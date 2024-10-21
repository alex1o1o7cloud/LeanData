import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_convergence_l513_51354

/-- The infinite series ∑[n=1 to ∞] (n³ - n) / ((n+3)!) converges to 1/6 -/
theorem infinite_sum_convergence :
  ∑' n : ℕ, (n^3 - n : ℝ) / (Nat.factorial (n + 3)) = 1/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_convergence_l513_51354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_M_l513_51352

def M : ℕ := 2^4 * 3^3 * 7^2

theorem number_of_factors_M : (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_M_l513_51352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_age_ratio_l513_51342

/- Define variables -/
variable (T N a : ℕ) -- Tom's current age, N years ago, and age of each younger child

/- Define the conditions -/
def tom_age (T a : ℕ) : Prop := T = 4 * a + 5
def children_ages_sum (T a : ℕ) : Prop := T = 3 * a + (a + 5)
def tom_age_n_years_ago (T N a : ℕ) : Prop := 
  T - N = 3 * (3 * (a - N) + ((a + 5) - N))

/- State the theorem -/
theorem tom_age_ratio (T N a : ℕ) : 
  tom_age T a → children_ages_sum T a → tom_age_n_years_ago T N a → T / N = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_age_ratio_l513_51342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l513_51392

-- Define the triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.A + t.B + t.C = Real.pi)
  (h2 : t.a * Real.cos (t.B - t.C) + t.a * Real.cos t.A = 2 * Real.sqrt 3 * t.b * Real.sin t.C * Real.cos t.A)
  (h3 : t.a + t.b + t.c = 8)
  (h4 : 2 * Real.sqrt 3 = t.a / Real.sin t.A) : 
  t.A = Real.pi / 3 ∧ 
  (1/2) * t.b * t.c * Real.sin t.A = (4 * Real.sqrt 3) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l513_51392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l513_51387

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  hd : d ≠ 0
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
  (h : seq.a 10 = S seq 4) : 
  S seq 8 / seq.a 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l513_51387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_value_sixty_three_incorrect_l513_51347

def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 5

def given_values : List ℝ := [10, 18, 29, 44, 63, 84, 111, 140]

theorem incorrect_value (x : Fin 8) :
  given_values[x] ≠ f (x.val + 1) :=
by
  sorry

theorem sixty_three_incorrect :
  given_values[4] ≠ f 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_value_sixty_three_incorrect_l513_51347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_range_l513_51381

theorem equation_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁ ∈ Set.Icc (k - 1) (k + 1) ∧ 
   x₂ ∈ Set.Icc (k - 1) (k + 1) ∧
   (|x₁ - k| = (Real.sqrt 2 / 2) * k * Real.sqrt x₁) ∧
   (|x₂ - k| = (Real.sqrt 2 / 2) * k * Real.sqrt x₂)) ↔ 
  (0 < k ∧ k ≤ 1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_range_l513_51381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l513_51332

/-- Definition of an ellipse with semi-major axis 1 and semi-minor axis b -/
def Ellipse (b : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 / b^2 = 1}

/-- The foci of the ellipse -/
noncomputable def Foci (b : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let c := Real.sqrt (1 - b^2)
  (-c, 0, c, 0)

theorem ellipse_property (b : ℝ) (hb : 0 < b ∧ b < 1) :
  ∃ (A B : ℝ × ℝ),
    A ∈ Ellipse b ∧
    B ∈ Ellipse b ∧
    (let (f1x, _, f2x, _) := Foci b
     (A.1 - f1x)^2 + A.2^2 = 4 * ((B.1 - f1x)^2 + B.2^2) ∧
     A.1 = f2x) →
    b^2 = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_property_l513_51332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_quadrant_l513_51315

theorem complex_number_quadrant : ∃ (z : ℂ), z = (1 : ℂ) / (1 + Complex.I) ∧ 
  z.re > 0 ∧ z.im < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_quadrant_l513_51315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_limit_is_infinity_l513_51368

noncomputable def my_sequence (n : ℕ) : ℝ :=
  ((n + 2)^4 - (n - 2)^4) / ((n + 5)^2 + (n - 5)^2)

theorem my_sequence_limit_is_infinity :
  Filter.Tendsto my_sequence Filter.atTop Filter.atTop :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_limit_is_infinity_l513_51368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_prob_theorem_l513_51320

-- Define the grid size
def grid_size : Nat := 4

-- Define the maximum number of hops
def max_hops : Nat := 6

-- Define a position on the grid
structure Position where
  x : Fin grid_size
  y : Fin grid_size

-- Define whether a position is on the edge
def is_edge (p : Position) : Bool :=
  p.x.val = 0 || p.x.val = grid_size - 1 || p.y.val = 0 || p.y.val = grid_size - 1

-- Define whether a position is central
def is_central (p : Position) : Bool :=
  p.x.val ∈ [1, 2] && p.y.val ∈ [1, 2]

-- Define the transition function (simplified for statement purposes)
noncomputable def transition (p : Position) : Position :=
  sorry

-- Define the probability of reaching an edge from a central position within n hops
noncomputable def prob_reach_edge (n : Nat) (p : Position) : Rat :=
  sorry

theorem frog_prob_theorem :
  ∃ (start : Position), is_central start ∧ 
    prob_reach_edge max_hops start = 211 / 256 := by
  sorry

#check frog_prob_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_prob_theorem_l513_51320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_cosine_l513_51337

theorem vector_angle_cosine (a b : ℝ × ℝ) (θ : ℝ) : 
  a = (-2, 1) → a + 2 • b = (2, 3) → Real.cos θ = -3/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_cosine_l513_51337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_longer_path_l513_51393

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A cube with side length 1 -/
def Cube : Set Point3D := {p : Point3D | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1 ∧ 0 ≤ p.z ∧ p.z ≤ 1}

/-- The spider's position on the cube -/
def spider : Point3D := ⟨0.5, 0, 0⟩

/-- The point opposite to the spider on the cube -/
def oppositePoint : Point3D := ⟨0.5, 1, 1⟩

/-- Function to calculate the shortest path between two points on the cube surface -/
noncomputable def shortestPath (p q : Point3D) : ℝ := sorry

/-- Theorem stating that there exists a point on the cube surface where the shortest path
    to the spider is greater than or equal to the shortest path from the opposite point -/
theorem exists_longer_path :
  ∃ p : Point3D, p ∈ Cube ∧ p ≠ oppositePoint ∧
  shortestPath p spider ≥ shortestPath oppositePoint spider := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_longer_path_l513_51393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_two_l513_51366

/-- A power function passing through (√2, 2√2) -/
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x^α

/-- The condition that f passes through (√2, 2√2) -/
axiom f_condition (α : ℝ) : f α (Real.sqrt 2) = 2 * Real.sqrt 2

/-- The theorem to prove -/
theorem f_at_two : ∃ α : ℝ, f α 2 = 8 := by
  -- We know α = 3 from our manual calculation
  use 3
  -- Simplify the goal
  simp [f]
  -- This is true by real number arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_two_l513_51366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_intersection_triangle_area_l513_51384

-- Define the points
variable (D A B C A1 A2 B1 B2 C1 C2 : EuclideanSpace ℝ (Fin 3))

-- Define the angles
variable (angle_ADB angle_ADC angle_BDC : ℝ)

-- Define the areas of the given triangles
variable (area_DA1B1 area_DA1C1 area_DB1C1 area_DA2B2 : ℝ)

-- State the theorem
theorem sphere_intersection_triangle_area :
  -- Conditions
  angle_ADB = 90 ∧ 
  angle_ADC = 90 ∧ 
  angle_BDC = 90 ∧
  area_DA1B1 = 15/4 ∧ 
  area_DA1C1 = 10 ∧ 
  area_DB1C1 = 6 ∧ 
  area_DA2B2 = 40 →
  -- Conclusion
  ∃ (area_A2B2C2 : ℝ), area_A2B2C2 = 50 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_intersection_triangle_area_l513_51384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_l513_51378

theorem cosine_value (θ : ℝ) 
  (h1 : Real.sin (θ + π/12) = 1/3) 
  (h2 : θ ∈ Set.Ioo (π/2) π) : 
  Real.cos (θ + π/3) = -(Real.sqrt 2 + 4)/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_value_l513_51378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_china_population_growth_rate_l513_51312

/-- The maximum annual average natural population growth rate (as a decimal) that keeps
    China's population under 1.4 billion after 10 years, starting from 1.3 billion. -/
noncomputable def max_growth_rate : ℝ :=
  (((1.4 / 1.3) ^ (1 / 10)) - 1)

theorem china_population_growth_rate :
  0.0073 < max_growth_rate ∧ max_growth_rate < 0.0075 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_china_population_growth_rate_l513_51312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_problem_solution_l513_51353

/-- The solution function for the Cauchy problem -/
noncomputable def y (x : ℝ) : ℝ := (2 - 3 * x) * Real.exp (2 * x)

/-- The differential equation -/
def diff_eq (f : ℝ → ℝ) : Prop :=
  ∀ x, (deriv (deriv f)) x - 4 * (deriv f x) + 4 * (f x) = 0

theorem cauchy_problem_solution :
  diff_eq y ∧ y 0 = 2 ∧ (deriv y) 0 = 1 := by
  sorry

#check cauchy_problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_problem_solution_l513_51353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_archery_variance_l513_51376

noncomputable def scores : List ℝ := [9, 10, 9, 7, 10]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m) ^ 2)).sum / xs.length

theorem archery_variance : variance scores = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_archery_variance_l513_51376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l513_51397

theorem negation_of_proposition :
  (¬(∀ x : ℝ, x > 0 → 2 * x^2 + 1 > 0)) ↔ (∃ x : ℝ, x > 0 ∧ 2 * x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l513_51397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_l513_51370

-- Define the piecewise function f(x)
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then (b - 3/2) * x + b - 1
  else -x^2 + (2-b) * x

-- State the theorem
theorem b_range (b : ℝ) :
  (∀ x y : ℝ, x < y → f b x < f b y) →
  (3/2 < b ∧ b ≤ 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_l513_51370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_max_zero_l513_51303

noncomputable def f (a x : ℝ) : ℝ := Real.cos x ^ 2 + a * Real.sin x + (5 * a / 8) - 5 / 2

theorem exists_a_max_zero :
  ∃ a : ℝ, (∀ x ∈ Set.Icc 0 Real.pi, f a x ≤ 0) ∧
           (∃ x ∈ Set.Icc 0 Real.pi, f a x = 0) ∧
           a = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_max_zero_l513_51303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l513_51395

/-- A quadratic function satisfying certain conditions -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_function_properties
  (a b c : ℝ)
  (h_a : a ≠ 0)
  (h_diff : ∀ x, QuadraticFunction a b c (x + 1) - QuadraticFunction a b c x = 2 * x)
  (h_f0 : QuadraticFunction a b c 0 = 1) :
  (∀ x, QuadraticFunction a b c x = x^2 - x + 1) ∧
  (∀ m, (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → QuadraticFunction a b c x > 2 * x + m) ↔ m < -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l513_51395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformed_data_l513_51338

variable (x₁ x₂ x₃ : ℝ)

noncomputable def mean (x₁ x₂ x₃ : ℝ) : ℝ := (x₁ + x₂ + x₃) / 3

noncomputable def variance (x₁ x₂ x₃ : ℝ) : ℝ :=
  ((x₁ - mean x₁ x₂ x₃)^2 + (x₂ - mean x₁ x₂ x₃)^2 + (x₃ - mean x₁ x₂ x₃)^2) / 3

theorem variance_transformed_data (h : variance x₁ x₂ x₃ = 3) :
  variance (2*x₁ + 3) (2*x₂ + 3) (2*x₃ + 3) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformed_data_l513_51338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l513_51345

/-- Given a geometric sequence {a_n} where a_2 = 3 and a_4 = 6, 
    prove that the common ratio q satisfies q = ± √2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * (a 1))
  (h_a2 : a 2 = 3)
  (h_a4 : a 4 = 6) :
  (a 1) = Real.sqrt 2 ∨ (a 1) = -Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l513_51345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_and_minimum_value_l513_51328

noncomputable def circle_C (a : ℝ) (θ : ℝ) : ℝ × ℝ := (a + a * Real.cos θ, a * Real.sin θ)

def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi/4) = 2 * Real.sqrt 2

noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem circle_intersection_and_minimum_value :
  ∃ (a : ℝ),
    (0 < a ∧ a < 5) ∧
    (∃ (θ₁ θ₂ : ℝ),
      line_l (distance (0, 0) (circle_C a θ₁)) θ₁ ∧
      line_l (distance (0, 0) (circle_C a θ₂)) θ₂ ∧
      distance (circle_C a θ₁) (circle_C a θ₂) = 2 * Real.sqrt 2) ∧
    a = 2 ∧
    (∀ (θ : ℝ),
      let f := λ θ' ↦ distance (0, 0) (circle_C a θ') + distance (0, 0) (circle_C a (θ' + Real.pi/3))
      f θ ≥ -4 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_and_minimum_value_l513_51328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_magnitude_of_OA_plus_OB_when_collinear_l513_51391

noncomputable section

-- Define the vectors
def OA (α : ℝ) : ℝ × ℝ := (Real.sin α, 1)
def OB (α : ℝ) : ℝ × ℝ := (Real.cos α, 0)
def OC (α : ℝ) : ℝ × ℝ := (-Real.sin α, 2)

-- Define point P
def P (α : ℝ) : ℝ × ℝ :=
  let AB := ((OB α).1 - (OA α).1, (OB α).2 - (OA α).2)
  ((OB α).1 + AB.1, (OB α).2 + AB.2)

-- Define function f
def f (α : ℝ) : ℝ :=
  let PB := ((P α).1 - (OB α).1, (P α).2 - (OB α).2)
  let CA := ((OC α).1 - (OA α).1, (OC α).2 - (OA α).2)
  PB.1 * CA.1 + PB.2 * CA.2

-- Theorem for part I
theorem smallest_positive_period_of_f : 
  ∃ (T : ℝ), T > 0 ∧ (∀ (α : ℝ), f (α + T) = f α) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (α : ℝ), f (α + T') = f α) → T ≤ T') :=
sorry

-- Theorem for part II
theorem magnitude_of_OA_plus_OB_when_collinear (α : ℝ) :
  (∃ (k : ℝ), P α = k • OC α) →
  Real.sqrt ((OA α).1 + (OB α).1)^2 + ((OA α).2 + (OB α).2)^2 = Real.sqrt 74 / 5 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_magnitude_of_OA_plus_OB_when_collinear_l513_51391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_part_of_proportional_division_l513_51348

theorem largest_part_of_proportional_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 156)
  (h_proportions : a = 2 ∧ b = 1/2 ∧ c = 1/4) :
  let parts := [a, b, c]
  let sum_parts := parts.sum
  let scaled_parts := parts.map (· * (total / sum_parts))
  scaled_parts.maximum? = some (112 * 8/11) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_part_of_proportional_division_l513_51348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_three_l513_51310

/-- The area of a triangle given by three points in R² --/
noncomputable def triangleArea (p q r : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := p
  let (x₂, y₂) := q
  let (x₃, y₃) := r
  (1/2) * abs (x₁*(y₂ - y₃) + x₂*(y₃ - y₁) + x₃*(y₁ - y₂))

/-- A point is on the line x + y = 10 --/
def onLine (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 = 10

theorem triangle_area_is_three :
  ∀ r : ℝ × ℝ, onLine r → triangleArea (4, 2) (2, 5) r = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_three_l513_51310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l513_51389

theorem trigonometric_identity (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.sin α + Real.cos α = Real.sqrt 3 / 3) :
  Real.cos α ^ 2 - Real.sin α ^ 2 = - Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l513_51389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l513_51313

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (1/3 * x - Real.pi/6)

theorem problem_solution :
  (f (5*Real.pi/4) = Real.sqrt 2) ∧
  (∀ α β : ℝ, 
    α ∈ Set.Icc 0 (Real.pi/2) → 
    β ∈ Set.Icc 0 (Real.pi/2) → 
    f (3*α + Real.pi/2) = 10/13 → 
    f (3*β + 2*Real.pi) = 6/5 → 
    Real.cos (α + β) = 16/65) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l513_51313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unpainted_area_specific_l513_51360

/-- The area of the unpainted region when two boards intersect -/
noncomputable def unpainted_area (width1 : ℝ) (width2 : ℝ) (angle : ℝ) : ℝ :=
  width1 * width2 / Real.sin angle

/-- Theorem: The area of the unpainted region on a 5-inch board when intersected
    by a 6-inch board at a 45-degree angle is 30√2 square inches -/
theorem unpainted_area_specific : 
  unpainted_area 5 6 (π/4) = 30 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unpainted_area_specific_l513_51360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_AOB_is_right_angle_l513_51364

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := y^2 = -x
def line (k x y : ℝ) : Prop := y = k * (x + 1)

-- Define the intersection points
def intersection_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), parabola x₁ y₁ ∧ line k x₁ y₁ ∧ parabola x₂ y₂ ∧ line k x₂ y₂ ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Theorem statement
theorem triangle_AOB_is_right_angle (k : ℝ) :
  intersection_points k →
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ * x₂ + y₁ * y₂ = 0 :=
by
  intro h
  rcases h with ⟨x₁, y₁, x₂, y₂, hparabola1, hline1, hparabola2, hline2, _⟩
  use x₁, y₁, x₂, y₂
  sorry -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_AOB_is_right_angle_l513_51364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_irrational_in_set_l513_51361

theorem one_irrational_in_set : ∃! x, x ∈ ({-3.14, 0, Real.pi, 22/7, 0.1010010001} : Set ℝ) ∧ Irrational x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_irrational_in_set_l513_51361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_equilateral_triangles_l513_51325

/-- Represents a point in the hexagonal lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The hexagonal lattice -/
def HexagonalLattice : Set LatticePoint :=
  {p : LatticePoint | p.x^2 + p.y^2 - p.x*p.y ≤ 4}

/-- Distance between two points in the lattice -/
noncomputable def distance (p q : LatticePoint) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 - (p.x - q.x)*(p.y - q.y))

/-- An equilateral triangle in the lattice -/
structure EquilateralTriangle where
  a : LatticePoint
  b : LatticePoint
  c : LatticePoint
  ha : a ∈ HexagonalLattice
  hb : b ∈ HexagonalLattice
  hc : c ∈ HexagonalLattice
  eq1 : distance a b = distance b c
  eq2 : distance b c = distance c a

/-- The set of all equilateral triangles in the lattice -/
def EquilateralTriangles : Set EquilateralTriangle := sorry

/-- Theorem stating the number of equilateral triangles in the lattice -/
theorem count_equilateral_triangles :
  ∃ (s : Finset EquilateralTriangle), s.card = 14 ∧ ∀ t, t ∈ s ↔ t ∈ EquilateralTriangles := by
  sorry

#check count_equilateral_triangles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_equilateral_triangles_l513_51325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_y_intercept_l513_51304

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 11

-- Define the point P
def P : ℝ × ℝ := (1, 12)

-- Define the slope of the tangent line at P
noncomputable def m : ℝ := 3 * P.1^2

-- Define the y-intercept of the tangent line
noncomputable def b : ℝ := P.2 - m * P.1

theorem tangent_y_intercept :
  b = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_y_intercept_l513_51304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crosswalk_problem_l513_51374

/-- Given a parallelogram with one side of length 20 feet and height 60 feet,
    and where the average length of the other two parallel sides is 62.5 feet,
    the length of the remaining side is approximately 19 feet. -/
theorem crosswalk_problem (side1 height avg_side2 : ℝ) 
  (h1 : side1 = 20)
  (h2 : height = 60)
  (h3 : avg_side2 = 62.5) : 
  ∃ (side2 : ℝ), abs (side2 - 19) < 0.5 ∧ side1 * height = avg_side2 * side2 := by
  sorry

#check crosswalk_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crosswalk_problem_l513_51374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_to_bike_speed_ratio_l513_51307

-- Define the given conditions
noncomputable def tractor_distance : ℝ := 575
noncomputable def tractor_time : ℝ := 23
noncomputable def car_distance : ℝ := 630
noncomputable def car_time : ℝ := 7

-- Define the relationships
noncomputable def tractor_speed : ℝ := tractor_distance / tractor_time
noncomputable def bike_speed : ℝ := 2 * tractor_speed
noncomputable def car_speed : ℝ := car_distance / car_time

-- Theorem to prove
theorem car_to_bike_speed_ratio :
  car_speed / bike_speed = 9 / 5 := by
  -- Expand definitions
  unfold car_speed bike_speed tractor_speed
  -- Simplify the expression
  simp [car_distance, car_time, tractor_distance, tractor_time]
  -- The proof is completed with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_to_bike_speed_ratio_l513_51307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_base_and_fractions_l513_51399

/-- Triangle area calculation -/
noncomputable def triangle_area (base height : ℝ) : ℝ := (1/2) * base * height

/-- Proper fraction definition -/
def is_proper_fraction (n d : ℤ) : Prop := 0 < n ∧ n < d

/-- Improper fraction definition -/
def is_improper_fraction (n d : ℤ) : Prop := n ≥ d ∧ d > 0

theorem triangle_base_and_fractions :
  ∃ (base : ℝ),
    triangle_area base 6 = 15 ∧
    is_proper_fraction 1 7 ∧
    is_improper_fraction 8 7 ∧
    base = 5 := by
  -- Proof goes here
  sorry

#check triangle_base_and_fractions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_base_and_fractions_l513_51399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_l513_51380

-- Define the hexagon PQRSTUV
def P : ℝ × ℝ := (0, 8)
def Q : ℝ × ℝ := (4, 8)
def R : ℝ × ℝ := (4, 5)
def S : ℝ × ℝ := (7, 5)
def T : ℝ × ℝ := (7, 0)
def U : ℝ × ℝ := (0, 0)

-- Define the line segments
def PQ : ℝ := 4
def QR : ℝ := 3
def UV : ℝ := 8
def ST' : ℝ := 7  -- Renamed to ST' to avoid conflict with S and T

-- Define right angles
axiom angle_PQU_right : Prop
axiom angle_QRS_right : Prop
axiom angle_STU_right : Prop

theorem hexagon_perimeter :
  let V : ℝ × ℝ := (0, UV)
  let perimeter := Real.sqrt ((P.1 - U.1)^2 + (P.2 - U.2)^2) + QR + (S.1 - R.1) + ST' +
                   Real.sqrt ((T.1 - U.1)^2 + (T.2 - U.2)^2) + UV
  perimeter = 27 + 4 * Real.sqrt 5 + Real.sqrt 74 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_l513_51380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l513_51318

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 3 * x

noncomputable def tangent_line (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ :=
  f a + (deriv f a) * (x - a)

theorem tangent_line_at_zero :
  tangent_line f 0 = λ x ↦ 4 * x + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l513_51318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_ninety_degrees_l513_51324

-- Define 90 degrees in radians
noncomputable def ninety_degrees : ℝ := Real.pi / 2

-- State the theorem
theorem sin_ninety_degrees : Real.sin ninety_degrees = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_ninety_degrees_l513_51324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l513_51371

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : 0 < a ∧ a < b

/-- Represents a line passing through two points -/
structure Line where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt (h.a^2 + h.b^2) / h.a

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (x : ℝ) (y : ℝ) (l : Line) : ℝ :=
  sorry  -- Definition of distance from point to line

theorem hyperbola_eccentricity (h : Hyperbola) (l : Line) 
  (h_line : l.x1 = h.a ∧ l.y1 = 0 ∧ l.x2 = 0 ∧ l.y2 = h.b)
  (h_distance : distance_point_to_line 0 0 l = (Real.sqrt 3 / 4) * Real.sqrt (h.a^2 + h.b^2)) :
  eccentricity h = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l513_51371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_solutions_l513_51323

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sqrt 3 * Real.cos x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, 2 * Real.sin x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_period_and_solutions :
  (∀ x, f (x + π) = f x) ∧
  (∀ t : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ ∈ Set.Icc 0 (π/2) ∧ x₂ ∈ Set.Icc 0 (π/2) ∧
    f x₁ - t = 1 ∧ f x₂ - t = 1) ↔ t ∈ Set.Icc 1 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_solutions_l513_51323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_sum_exists_l513_51321

/-- A set of marked numbers not exceeding 30 -/
def MarkedSet : Type := { s : Finset ℕ // s.card = 22 ∧ ∀ n ∈ s, n ≤ 30 }

/-- Theorem: In any set of 22 marked natural numbers not exceeding 30,
    there exists a marked number that is the sum of three other marked numbers -/
theorem marked_sum_exists (S : MarkedSet) :
  ∃ a b c d, a ∈ S.val ∧ b ∈ S.val ∧ c ∈ S.val ∧ d ∈ S.val ∧ a + b + c = d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marked_sum_exists_l513_51321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_with_tan_roots_l513_51351

theorem sum_of_angles_with_tan_roots (α β : Real) : 
  α ∈ Set.Ioo (-π/2) (π/2) →
  β ∈ Set.Ioo (-π/2) (π/2) →
  (∃ x y : Real, x = Real.tan α ∧ y = Real.tan β ∧ 
    x^2 + 3 * Real.sqrt 3 * x + 4 = 0 ∧
    y^2 + 3 * Real.sqrt 3 * y + 4 = 0) →
  α + β = -2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_with_tan_roots_l513_51351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_triangle_l513_51329

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis

/-- Check if a point lies on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Theorem statement -/
theorem largest_angle_in_triangle (e : Ellipse) (f1 f2 m : Point) :
  e.a = 4 ∧ e.b = 2 * Real.sqrt 3 ∧  -- Ellipse parameters
  isOnEllipse m e ∧  -- M is on the ellipse
  distance m f1 - distance m f2 = 2 →  -- Given condition
  ∃ θ : ℝ, θ = 90 ∧ θ = max (max (angle f1 m f2) (angle m f1 f2)) (angle m f2 f1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_in_triangle_l513_51329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_lamps_turn_on_l513_51355

/-- A lattice point in a two-dimensional Cartesian coordinate system. -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The distance between two lattice points. -/
noncomputable def distance (p q : LatticePoint) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A predicate that checks if all prime factors of a number are ≡ 1 (mod 4). -/
def all_prime_factors_congruent_1_mod_4 (n : ℕ) : Prop :=
  ∀ p, Nat.Prime p → p ∣ n → p % 4 = 1

/-- A path from the origin to a given lattice point using steps of length s. -/
def valid_path (s : ℕ) (target : LatticePoint) (path : List LatticePoint) : Prop :=
  path.head? = some ⟨0, 0⟩ ∧
  path.getLast? = some target ∧
  ∀ i j, i + 1 = j → j < path.length →
    distance (path.get ⟨i, by sorry⟩) (path.get ⟨j, by sorry⟩) = s

/-- The main theorem: for any integer s whose prime factors are all ≡ 1 (mod 4),
    and for any lattice point, there exists a valid path from the origin to that point. -/
theorem all_lamps_turn_on (s : ℕ) (target : LatticePoint) :
    s > 0 →
    all_prime_factors_congruent_1_mod_4 s →
    ∃ path : List LatticePoint, valid_path s target path := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_lamps_turn_on_l513_51355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_of_crayons_l513_51365

theorem least_number_of_crayons (n : ℕ) : 
  (∀ m : ℕ, m ∈ ({4, 6, 8, 9, 10} : Set ℕ) → n % m = 0) → 
  (∀ k : ℕ, k > 0 ∧ (∀ m : ℕ, m ∈ ({4, 6, 8, 9, 10} : Set ℕ) → k % m = 0) → k ≥ 360) →
  n = 360 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_of_crayons_l513_51365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logo_properties_l513_51344

/-- Logo structure with a square and four circles -/
structure Logo where
  square_side : ℝ
  circle_radius : ℝ
  circle_tangent_to_square : Prop
  circle_tangent_to_adjacent : Prop

/-- Calculate the total shaded area of the logo -/
noncomputable def total_shaded_area (l : Logo) : ℝ :=
  l.square_side ^ 2 - 4 * Real.pi * l.circle_radius ^ 2

/-- Calculate the total circumference of all circles in the logo -/
noncomputable def total_circumference (l : Logo) : ℝ :=
  8 * Real.pi * l.circle_radius

theorem logo_properties (l : Logo) 
  (h1 : l.square_side = 30)
  (h2 : l.circle_radius = 7.5)
  (h3 : l.circle_tangent_to_square)
  (h4 : l.circle_tangent_to_adjacent) :
  total_shaded_area l = 900 - 225 * Real.pi ∧
  total_circumference l = 60 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_logo_properties_l513_51344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l513_51349

/-- A hyperbola is defined by its equation and two points it passes through -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  p : ℝ × ℝ
  q : ℝ × ℝ

/-- The standard form of a hyperbola with vertical transverse axis -/
def verticalHyperbolaEquation (h : Hyperbola) (x y : ℝ) : Prop :=
  y^2 / h.b^2 - x^2 / h.a^2 = 1

/-- A point satisfies the hyperbola equation -/
def pointSatisfiesEquation (h : Hyperbola) (point : ℝ × ℝ) : Prop :=
  verticalHyperbolaEquation h point.fst point.snd

/-- The foci of a hyperbola with vertical transverse axis lie on the y-axis -/
def fociOnYAxis (h : Hyperbola) : Prop :=
  ∃ c : ℝ, c^2 = h.a^2 + h.b^2 ∧ (0, c) ∈ Set.range (λ t : ℝ ↦ (0, t))

/-- Main theorem: The hyperbola passes through given points and has foci on y-axis -/
theorem hyperbola_properties (h : Hyperbola) 
    (h_eq : h.a^2 = 75 ∧ h.b^2 = 25)
    (h_p : h.p = (-3, 2 * Real.sqrt 7))
    (h_q : h.q = (-6 * Real.sqrt 2, -7)) :
  pointSatisfiesEquation h h.p ∧
  pointSatisfiesEquation h h.q ∧
  fociOnYAxis h := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l513_51349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_value_l513_51363

def b (n : ℕ) : ℤ := n - 35

def a : ℕ → ℤ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | (n + 2) => a (n + 1) + 2^(n + 1)

def ratio (n : ℕ) : ℚ := (b n : ℚ) / (a n : ℚ)

theorem max_ratio_value :
  ∃ (k : ℕ), ∀ (n : ℕ), ratio n ≤ ratio k ∧ ratio k = 1 / 2^36 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_value_l513_51363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_integrals_existence_l513_51334

/-- A partition of [-1, 1] into black and white measurable subsets -/
structure BlackWhitePartition where
  black : Set ℝ
  white : Set ℝ
  black_measurable : MeasurableSet black
  white_measurable : MeasurableSet white
  partition : black ∪ white = Set.Icc (-1) 1
  disjoint : black ∩ white = ∅

/-- Polynomial of degree at most 2 -/
def QuadraticPolynomial := { f : ℝ → ℝ | ∃ a b c, ∀ x, f x = a*x^2 + b*x + c }

theorem equal_integrals_existence :
  ∃ (p : BlackWhitePartition),
    ∀ (f : ℝ → ℝ), f ∈ QuadraticPolynomial →
      ∫ x in p.black, f x = ∫ x in p.white, f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_integrals_existence_l513_51334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_intersection_theorem_l513_51343

/-- Point type represents a point in a plane -/
structure Point where

/-- ∠ X Y Z represents the angle formed by points X, Y, and Z with Y as the vertex -/
noncomputable def angle (X Y Z : Point) : ℝ := sorry

/-- IsAngleBisector X Y Z means that XY is the angle bisector of ∠XYZ -/
def IsAngleBisector (X Y Z : Point) : Prop := 
  ∃ O : Point, angle X Y Z / 2 = angle X Y O

theorem angle_bisector_intersection_theorem (A B C O : Point) 
  (h1 : angle A O B = 125)
  (h2 : IsAngleBisector A O B) 
  (h3 : IsAngleBisector B O A) : 
  angle A C B = 70 := by
  sorry

#check angle_bisector_intersection_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_intersection_theorem_l513_51343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_locus_ellipse_l513_51305

theorem complex_locus_ellipse (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), (z + z⁻¹ = x + y * Complex.I) →
  (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_locus_ellipse_l513_51305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_inscribed_spheres_in_cones_l513_51319

/-- The distance between the centers of spheres inscribed in two cones within a sphere -/
noncomputable def distance_between_inscribed_spheres (R α : ℝ) : ℝ :=
  (R * Real.sqrt 2 * Real.sin (α / 2)) / (2 * Real.cos (α / 8) * Real.cos (Real.pi / 4 - α / 8))

/-- Theorem stating the distance between the centers of spheres inscribed in two cones within a sphere -/
theorem distance_inscribed_spheres_in_cones 
  (R α : ℝ) 
  (h_R : R > 0) 
  (h_α : 0 < α ∧ α < Real.pi) :
  let sphere_radius := R
  let arc_angle := α
  distance_between_inscribed_spheres R α = 
    (R * Real.sqrt 2 * Real.sin (α / 2)) / (2 * Real.cos (α / 8) * Real.cos (Real.pi / 4 - α / 8)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_inscribed_spheres_in_cones_l513_51319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sinusoidal_function_l513_51359

noncomputable def f (x : ℝ) := Real.sin (2 * x)

noncomputable def g (x : ℝ) := Real.sin (4 * x + 2 * Real.pi / 3)

def shift_left (f : ℝ → ℝ) (c : ℝ) : ℝ → ℝ := λ x => f (x + c)

def compress (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f (k * x)

theorem transform_sinusoidal_function :
  compress (shift_left f (Real.pi / 3)) 2 = g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sinusoidal_function_l513_51359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_line_equation_l513_51379

/-- A line passing through (1,1) intersecting a circle with equation (x-2)^2 + (y-3)^2 = 9 -/
structure IntersectingLine where
  -- Slope of the line
  k : ℝ
  -- The line passes through (1,1)
  passes_through_point : k * 1 - 1 - k + 1 = 0
  -- The distance from the center of the circle to the line
  distance_to_center : |2*k - 3 - k + 1| / Real.sqrt (1 + k^2) = Real.sqrt 5

/-- The theorem stating the equation of the line -/
theorem intersecting_line_equation (l : IntersectingLine) : 
  l.k = -1/2 ∧ (fun x y ↦ x + 2*y - 3 = 0) = (fun x y ↦ l.k * x - y - l.k + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_line_equation_l513_51379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_measurements_l513_51386

/-- Represents a rectangular garden with given perimeter and breadth -/
structure RectangularGarden where
  perimeter : ℝ
  breadth : ℝ

/-- Calculates the length of a rectangular garden -/
noncomputable def garden_length (g : RectangularGarden) : ℝ :=
  (g.perimeter / 2) - g.breadth

/-- Calculates the diagonal of a rectangular garden using the Pythagorean theorem -/
noncomputable def garden_diagonal (g : RectangularGarden) : ℝ :=
  Real.sqrt ((garden_length g)^2 + g.breadth^2)

/-- Theorem stating the length and diagonal of a specific rectangular garden -/
theorem garden_measurements (g : RectangularGarden) 
  (h1 : g.perimeter = 600)
  (h2 : g.breadth = 200) :
  garden_length g = 100 ∧ garden_diagonal g = Real.sqrt 50000 := by
  sorry

#check garden_measurements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_measurements_l513_51386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_owner_gain_percentage_l513_51369

/-- Represents the transaction details of a store owner --/
structure Transaction where
  cost_price : ℚ
  selling_price : ℚ
  num_articles : ℕ
  discount_rate : ℚ
  tax_rate : ℚ

/-- Calculates the overall gain percentage for a given transaction --/
def overall_gain_percentage (t : Transaction) : ℚ :=
  let total_cost := t.cost_price * t.num_articles
  let discounted_price := t.selling_price * (1 - t.discount_rate)
  let total_discounted_sales := discounted_price * t.num_articles
  let tax_amount := total_discounted_sales * t.tax_rate
  let net_selling_price := total_discounted_sales - tax_amount
  let gain := net_selling_price - total_cost
  (gain / total_cost) * 100

/-- Theorem stating the overall gain percentage for the given transaction --/
theorem store_owner_gain_percentage :
  ∃ (t : Transaction),
    t.num_articles = 50 ∧
    t.cost_price * 50 = t.selling_price * 25 ∧
    t.discount_rate = 1/10 ∧
    t.tax_rate = 1/20 ∧
    overall_gain_percentage t = 71 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_owner_gain_percentage_l513_51369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_system_l513_51372

/-- Represents the length of the rope in feet -/
def x : ℝ := sorry

/-- Represents the length of the wood in feet -/
def y : ℝ := sorry

/-- The rope exceeds the wood by 4.5 feet when used to measure it -/
axiom rope_exceeds : x - y = 4.5

/-- Half the rope length plus 1 foot equals the wood length -/
axiom half_rope_short : (1/2) * x + 1 = y

/-- The system of equations correctly represents the relationship between x and y -/
theorem correct_system : (x - y = 4.5) ∧ ((1/2) * x + 1 = y) := by
  constructor
  · exact rope_exceeds
  · exact half_rope_short

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_system_l513_51372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l513_51316

theorem right_triangle_hypotenuse (m₁ m₂ : ℝ) (h₁ : m₁ = 6) (h₂ : m₂ = Real.sqrt 48) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a^2 + b^2 = c^2 ∧
    (a^2 / 4 + b^2 / 2 = m₁^2 ∨ b^2 / 4 + a^2 / 2 = m₁^2) ∧
    (a^2 / 4 + b^2 / 2 = m₂^2 ∨ b^2 / 4 + a^2 / 2 = m₂^2) ∧
    |c - 15.25| < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l513_51316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_from_perimeter_relation_l513_51309

open Real

/-- Predicate to assert that p, R, and r form a valid triangle -/
def IsTriangle (p R r : ℝ) : Prop := 
  p > 0 ∧ R > 0 ∧ r > 0 ∧ p > 2 * R ∧ p > 2 * r

/-- Predicate to assert that α, β, and γ form a valid triangle -/
def IsTriangleOfSides (α β γ : ℝ) : Prop := 
  α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β > γ ∧ β + γ > α ∧ γ + α > β

/-- Given a triangle with perimeter p, circumradius R, and inradius r,
    if there exists an angle φ (0 < φ < π) such that p = 2R sin φ + r cot(φ/2),
    then φ is one of the angles of the triangle. -/
theorem angle_from_perimeter_relation 
  (p R r : ℝ) (φ : ℝ) 
  (h_triangle : IsTriangle p R r)
  (h_angle_range : 0 < φ ∧ φ < π)
  (h_relation : p = 2 * R * sin φ + r * (cos (φ / 2) / sin (φ / 2))) :
  ∃ (α β γ : ℝ), IsTriangleOfSides α β γ ∧ (φ = α ∨ φ = β ∨ φ = γ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_from_perimeter_relation_l513_51309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_distance_theorem_l513_51311

/-- Represents the scale of a map as a ratio of map distance to actual distance -/
structure MapScale where
  map_distance : ℚ
  actual_distance : ℚ

/-- Calculates the actual distance given a map scale and measured map distance -/
def calculate_actual_distance (scale : MapScale) (map_distance : ℚ) : ℚ :=
  (scale.actual_distance / scale.map_distance) * map_distance

theorem actual_distance_theorem (scale : MapScale) (map_distance : ℚ) :
  scale.map_distance = 1 ∧ 
  scale.actual_distance = 25000 ∧ 
  map_distance = 5 →
  calculate_actual_distance scale map_distance = 125000 / 100000 := by
  sorry

#eval (125000 : ℚ) / 100000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_actual_distance_theorem_l513_51311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l513_51383

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.cos x + Real.sqrt 3 * Real.sin x)

-- Theorem statement
theorem f_properties :
  -- 1. Period of f is π
  (∀ x, f (x + π) = f x) ∧
  -- 2. Intervals of monotonic increase
  (∀ k : ℤ, StrictMonoOn f (Set.Ioo (k * π - π / 3) (k * π + π / 6))) ∧
  -- 3. Maximum value on [0, π/2]
  (∀ x ∈ Set.Icc 0 (π / 2), f x ≤ 3) ∧ (∃ x ∈ Set.Icc 0 (π / 2), f x = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l513_51383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_term_prime_l513_51301

/-- Defines the kth term of the sequence -/
def a (k : ℕ) : ℕ := 17 * (Finset.sum (Finset.range k) (λ i => 10^(2*i)))

/-- States that only the first term of the sequence is prime -/
theorem only_first_term_prime (n : ℕ) (hn : n > 0) :
  (∃! k : ℕ, k ≤ n ∧ Nat.Prime (a k)) ∧
  (∀ k : ℕ, k ≤ n → Nat.Prime (a k) → k = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_term_prime_l513_51301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l513_51335

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then -2 * x - 3 else 2^(-x)

-- State the theorem
theorem f_composition_value : f (f (-3)) = 1/8 := by
  -- Evaluate f(-3)
  have h1 : f (-3) = 3 := by
    simp [f]
    norm_num
  
  -- Evaluate f(3)
  have h2 : f 3 = 1/8 := by
    simp [f]
    norm_num
  
  -- Combine the results
  calc f (f (-3))
    = f 3 := by rw [h1]
    _ = 1/8 := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l513_51335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_equilateral_triangle_l513_51382

/-- Given a line that intersects a circle centered at the origin, forming an equilateral triangle with two intersection points and the origin, prove the slope of the line. -/
theorem line_circle_intersection_equilateral_triangle (a : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    (A.1 + a * A.2 + 3 = 0) ∧ 
    (B.1 + a * B.2 + 3 = 0) ∧ 
    (A.1^2 + A.2^2 = 4) ∧ 
    (B.1^2 + B.2^2 = 4) ∧ 
    (A.1^2 + A.2^2 = B.1^2 + B.2^2) ∧
    (A.1^2 + A.2^2 = 4)) →
  a = Real.sqrt 2 ∨ a = -Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_equilateral_triangle_l513_51382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_plus_sin_value_l513_51300

theorem cos_plus_sin_value (α : ℝ) 
  (h : (Real.cos (2 * α)) / (Real.sin (α - Real.pi/4)) = -Real.sqrt 2/2) : 
  Real.cos α + Real.sin α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_plus_sin_value_l513_51300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l513_51340

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 1

noncomputable def g (x : ℝ) : ℝ := (1/2) * abs (f (x + Real.pi/12)) + (1/2) * abs (f (x + Real.pi/3))

theorem g_properties :
  (∀ x, g (-x) = g x) ∧ 
  (∀ x, g (x + Real.pi/4) = g x) ∧
  (∀ T, 0 < T ∧ T < Real.pi/4 → ∃ x, g (x + T) ≠ g x) ∧
  (∀ x, 1 ≤ g x ∧ g x ≤ Real.sqrt 2) ∧
  (∃ x₁ x₂, g x₁ = 1 ∧ g x₂ = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l513_51340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_price_l513_51341

/-- The selling price of an item given its cost price and markup percentage -/
noncomputable def selling_price (cost_price : ℝ) (markup_percentage : ℝ) : ℝ :=
  cost_price * (1 + markup_percentage / 100)

/-- Theorem: The selling price of a computer table with cost price 500 and 100% markup is 1000 -/
theorem computer_table_price : selling_price 500 100 = 1000 := by
  -- Unfold the definition of selling_price
  unfold selling_price
  -- Simplify the expression
  simp
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_price_l513_51341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_requirement_l513_51326

/-- Calculates the required fencing for a rectangular field with given conditions -/
noncomputable def calculate_fencing (area : ℝ) (side : ℝ) (height_diff : ℝ) : ℝ :=
  let length := area / side
  let perimeter := 2 * length + side
  let extra_fencing := (height_diff / 5) * 2
  perimeter + extra_fencing

/-- Theorem stating the required fencing for the given field -/
theorem fencing_requirement :
  let area : ℝ := 260
  let side : ℝ := 25
  let height_diff : ℝ := 15
  calculate_fencing area side height_diff = 51.8 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_fencing 260 25 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_requirement_l513_51326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_three_consecutive_ones_l513_51317

def sequence_length : ℕ := 12

def valid_sequence (s : List ℕ) : Prop :=
  s.length = sequence_length ∧ 
  (∀ x ∈ s, x = 0 ∨ x = 1) ∧
  ¬(∃ i, i + 2 < s.length ∧ s[i]? = some 1 ∧ s[i+1]? = some 1 ∧ s[i+2]? = some 1)

def b : ℕ → ℕ
| 0 => 1
| 1 => 2
| 2 => 4
| n + 3 => b (n + 2) + b (n + 1) + b n

theorem probability_no_three_consecutive_ones :
  (b sequence_length : ℚ) / (2^sequence_length : ℚ) = 1705 / 4096 := by sorry

#eval b sequence_length
#eval 2^sequence_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_three_consecutive_ones_l513_51317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_and_triangle_area_l513_51375

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if three points form an equilateral triangle -/
def isEquilateralTriangle (p1 p2 p3 : Point) : Prop :=
  distance p1 p2 = distance p2 p3 ∧ distance p2 p3 = distance p3 p1 ∧ distance p3 p1 = 3

/-- Calculate the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  let s := (distance p1 p2 + distance p2 p3 + distance p3 p1) / 2
  Real.sqrt (s * (s - distance p1 p2) * (s - distance p2 p3) * (s - distance p3 p1))

/-- Main theorem -/
theorem circle_intersection_and_triangle_area 
  (A B C : Point) (circleA circleB circleC : Circle) (E F : Point) :
  circleA.radius = 3 ∧ circleB.radius = 3 ∧ circleC.radius = 3 →
  circleA.center = A ∧ circleB.center = B ∧ circleC.center = C →
  isEquilateralTriangle A B C →
  -- E is on circles A and B
  distance E A = 3 ∧ distance E B = 3 →
  -- F is on circles A and C
  distance F A = 3 ∧ distance F C = 3 →
  -- E is outside circle C and F is outside circle B
  distance E C > 3 ∧ distance F B > 3 →
  distance E F = 3 ∧ triangleArea A B C = Real.sqrt 31.5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_and_triangle_area_l513_51375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluation_related_to_gender_probability_different_genders_l513_51306

/-- Contingency table data --/
def male_thumbs_up : ℕ := 120
def male_thumbs_down : ℕ := 80
def female_thumbs_up : ℕ := 90
def female_thumbs_down : ℕ := 110

/-- Chi-square calculation --/
noncomputable def chi_square : ℝ := 
  let n := male_thumbs_up + male_thumbs_down + female_thumbs_up + female_thumbs_down
  let a := male_thumbs_up
  let b := male_thumbs_down
  let c := female_thumbs_up
  let d := female_thumbs_down
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Critical value for 99.5% certainty --/
def critical_value : ℝ := 7.879

/-- Stratified sampling calculation --/
def male_sample : ℕ := 4
def female_sample : ℕ := 3
def total_sample : ℕ := male_sample + female_sample

/-- Theorems to prove --/
theorem evaluation_related_to_gender : chi_square > critical_value := by sorry

theorem probability_different_genders : 
  (male_sample * female_sample : ℝ) / (total_sample * (total_sample - 1) / 2) = 4 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluation_related_to_gender_probability_different_genders_l513_51306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_numerator_simplification_l513_51336

theorem product_numerator_simplification (a b c d s : ℝ) (h : 2 * s = a + b + c + d) :
  let expr := (1 + (a^2 + b^2 - c^2 - d^2) / (2 * (a * b + c * d))) *
               (1 - (a^2 + b^2 - c^2 - d^2) / (2 * (a * b + c * d)))
  expr = 16 * (s - a) * (s - b) * (s - c) * (s - d) / (4 * (a * b + c * d)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_numerator_simplification_l513_51336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nineteenth_term_equals_target_l513_51385

/-- The nth term of the sequence -/
noncomputable def a (n : ℕ+) : ℝ := Real.log (4 * n - 1)

/-- The theorem stating that the 19th term of the sequence equals 2ln5 + ln3 -/
theorem nineteenth_term_equals_target : a 19 = 2 * Real.log 5 + Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nineteenth_term_equals_target_l513_51385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_polygon_has_equal_sides_l513_51327

/-- A convex polygon divided into isosceles triangles by non-intersecting diagonals -/
structure IsoscelesPolygon where
  /-- The set of vertices of the polygon -/
  vertices : Set (ℝ × ℝ)
  /-- The set of edges of the polygon -/
  edges : Set (ℝ × ℝ × ℝ × ℝ)
  /-- The set of diagonals dividing the polygon -/
  diagonals : Set (ℝ × ℝ × ℝ × ℝ)
  /-- Predicate ensuring the polygon is convex -/
  convex : Bool
  /-- Predicate ensuring the diagonals are non-intersecting -/
  non_intersecting : Bool
  /-- Predicate ensuring all triangles formed by diagonals are isosceles -/
  isosceles_triangles : Bool

/-- Function to calculate the length of an edge -/
def EdgeLength (e : ℝ × ℝ × ℝ × ℝ) : ℝ := sorry

/-- Theorem stating that a convex polygon divided into isosceles triangles by non-intersecting diagonals has at least two equal sides -/
theorem isosceles_polygon_has_equal_sides (p : IsoscelesPolygon) : 
  ∃ (e1 e2 : ℝ × ℝ × ℝ × ℝ), e1 ∈ p.edges ∧ e2 ∈ p.edges ∧ e1 ≠ e2 ∧ EdgeLength e1 = EdgeLength e2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_polygon_has_equal_sides_l513_51327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_eq_eight_l513_51322

/-- A square with side length 4 cm -/
structure Square where
  side : ℝ
  side_positive : side > 0
  side_eq_four : side = 4

/-- A rhombus formed by joining the midpoints of a square's sides -/
structure Rhombus (s : Square) where
  diagonal : ℝ
  diagonal_eq_side : diagonal = s.side

/-- The area of a rhombus -/
noncomputable def rhombus_area (s : Square) (r : Rhombus s) : ℝ :=
  r.diagonal * r.diagonal / 2

theorem rhombus_area_eq_eight (s : Square) (r : Rhombus s) :
  rhombus_area s r = 8 := by
  sorry

#check rhombus_area_eq_eight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_eq_eight_l513_51322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enterprise_financials_l513_51377

/-- Represents the annual turnover and profit of an enterprise over time -/
structure Enterprise where
  initialTurnover : ℝ  -- Initial annual turnover in billion yuan
  turnoverIncrease : ℝ  -- Annual increase in turnover in billion yuan
  initialProfit : ℝ  -- Initial annual profit in billion yuan
  profitGrowthRate : ℝ  -- Annual profit growth rate

/-- Calculates the sum of turnover for the first n quarters -/
noncomputable def sumTurnover (e : Enterprise) (n : ℕ) : ℝ :=
  n * e.initialTurnover + n * (n - 1) / 2 * e.turnoverIncrease / 4

/-- Calculates the turnover for the nth quarter -/
noncomputable def turnoverAtQuarter (e : Enterprise) (n : ℕ) : ℝ :=
  e.initialTurnover + (n - 1) * e.turnoverIncrease / 4

/-- Calculates the profit for the nth quarter -/
noncomputable def profitAtQuarter (e : Enterprise) (n : ℕ) : ℝ :=
  e.initialProfit * (1 + e.profitGrowthRate / 4) ^ (n - 1)

/-- The main theorem about the enterprise's financials -/
theorem enterprise_financials (e : Enterprise) 
  (h1 : e.initialTurnover = 1.1)
  (h2 : e.turnoverIncrease = 0.05)
  (h3 : e.initialProfit = 0.16)
  (h4 : e.profitGrowthRate = 0.04) :
  (sumTurnover e 20 = 31.5) ∧ 
  (∀ n < 25, turnoverAtQuarter e n > 0.18 * profitAtQuarter e n) ∧
  (turnoverAtQuarter e 25 ≤ 0.18 * profitAtQuarter e 25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enterprise_financials_l513_51377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_gt_one_neither_sufficient_nor_necessary_l513_51398

/-- An exponential sequence with common ratio q -/
def exponential_sequence (a₀ : ℝ) (q : ℝ) : ℕ → ℝ := fun n ↦ a₀ * q^n

/-- A sequence is increasing -/
def is_increasing (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, s n < s (n + 1)

/-- The condition q > 1 is neither sufficient nor necessary for an exponential sequence to be increasing -/
theorem q_gt_one_neither_sufficient_nor_necessary :
  (∃ q > 1, ∃ a₀, ¬is_increasing (exponential_sequence a₀ q)) ∧
  (∃ a₀ q, q ≤ 1 ∧ is_increasing (exponential_sequence a₀ q)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_gt_one_neither_sufficient_nor_necessary_l513_51398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_number_count_is_36_l513_51356

/-- A function that returns true if a number is a valid 3-digit number with the units digit exactly twice the tens digit -/
def isValidNumber (n : ℕ) : Bool :=
  100 ≤ n ∧ n < 1000 ∧ (n % 10 = 2 * ((n / 10) % 10))

/-- The count of valid 3-digit numbers -/
def validNumberCount : ℕ :=
  (List.range 1000).filter isValidNumber |>.length

/-- Theorem stating that the count of valid 3-digit numbers is 36 -/
theorem valid_number_count_is_36 : validNumberCount = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_number_count_is_36_l513_51356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l513_51331

-- Define complex numbers α and β
variable (α β : ℂ)

-- Define the conditions
def condition1 (α β : ℂ) : Prop := (α + β).re > 0
def condition2 (α β : ℂ) : Prop := (Complex.I * (2 * α - β)).re > 0
def condition3 (α β : ℂ) : Prop := β = 2 + 3 * Complex.I

-- Theorem statement
theorem alpha_value :
  ∀ α β : ℂ, condition1 α β → condition2 α β → condition3 α β →
  α = 6 + 4 * Complex.I :=
by
  intros α β h1 h2 h3
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l513_51331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_constraint_l513_51330

-- Define the curve C: y = e^x
noncomputable def C : ℝ → ℝ := fun x ↦ Real.exp x

-- Define the tangent line l
noncomputable def l : ℝ → ℝ := fun x ↦ Real.exp 1 * x

-- Define point A
def A : ℝ × ℝ := (0, 1)

-- Define point B
noncomputable def B (b : ℝ) : ℝ × ℝ := (b, Real.exp b)

-- Theorem statement
theorem tangent_line_parallel_constraint (b : ℝ) :
  (∀ x, (deriv C) x = l x) →  -- l is tangent to C
  l 0 = 0 →  -- l passes through the origin
  (B b).2 - A.2 = (Real.exp 1) * (B b).1 →  -- AB is parallel to l
  1 < b ∧ b < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_constraint_l513_51330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_distinct_colorings_l513_51373

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  faces : Fin 6 → Bool

/-- A rotation of a cube is a permutation of its faces that preserves its structure -/
def CubeRotation := Equiv.Perm (Fin 6)

/-- Two colorings are equivalent if one can be obtained from the other by rotation -/
def ColoringEquivalence (c₁ c₂ : Cube) : Prop :=
  ∃ (r : CubeRotation), ∀ (i : Fin 6), c₁.faces i = c₂.faces (r.toFun i)

/-- A valid coloring has at least one face painted -/
def ValidColoring (c : Cube) : Prop :=
  ∃ (i : Fin 6), c.faces i = true

/-- The number of distinct ways to paint a cube -/
noncomputable def DistinctColorings : ℕ := sorry

/-- Theorem: There are 7 distinct ways to paint a cube -/
theorem seven_distinct_colorings : DistinctColorings = 7 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_distinct_colorings_l513_51373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_and_tarun_completion_time_l513_51302

/-- Represents the total amount of work to be done -/
noncomputable def total_work : ℝ := 1

/-- Arun's work rate (portion of work completed per day) -/
noncomputable def arun_rate : ℝ := total_work / 70

/-- The number of days Arun and Tarun worked together before Tarun left -/
noncomputable def days_together : ℝ := 4

/-- The number of days Arun worked alone to complete the remaining work -/
noncomputable def days_arun_alone : ℝ := 42

/-- Theorem stating that Arun and Tarun can complete the work together in 10 days -/
theorem arun_and_tarun_completion_time :
  ∃ (tarun_rate : ℝ),
    tarun_rate > 0 ∧
    days_together * (arun_rate + tarun_rate) + days_arun_alone * arun_rate = total_work ∧
    (arun_rate + tarun_rate) = total_work / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_and_tarun_completion_time_l513_51302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_equals_fraction_l513_51362

/-- The decimal representation 0.3̄23 as a real number -/
noncomputable def decimal_rep : ℝ := 0.3 + (23 : ℝ) / 99

/-- The fraction 527/990 as a real number -/
noncomputable def fraction_rep : ℝ := 527 / 990

/-- Theorem stating that the decimal representation 0.3̄23 is equal to the fraction 527/990 -/
theorem decimal_equals_fraction : decimal_rep = fraction_rep := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_equals_fraction_l513_51362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l513_51339

theorem exponential_equation_solution (x : ℝ) : (25 : ℝ)^x * (125 : ℝ)^(3*x) = (25 : ℝ)^15 → x = 30/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l513_51339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_at_31_degrees_l513_51333

/-- Represents the volume of gas at a given temperature -/
structure GasVolume where
  temp : ℝ
  volume : ℝ

/-- The rate of volume change per 2° temperature change -/
noncomputable def volumeChangeRate : ℝ := 3

/-- The reference temperature -/
noncomputable def referenceTemp : ℝ := 45

/-- The reference volume at the reference temperature -/
noncomputable def referenceVolume : ℝ := 30

/-- The target temperature for which we want to find the volume -/
noncomputable def targetTemp : ℝ := 31

/-- Calculates the volume of gas at a given temperature -/
noncomputable def calculateVolume (t : ℝ) : ℝ :=
  referenceVolume - volumeChangeRate * ((referenceTemp - t) / 2)

/-- Theorem stating that the volume at 31° is 9 cubic centimeters -/
theorem volume_at_31_degrees :
  calculateVolume targetTemp = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_at_31_degrees_l513_51333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_point_z_coordinate_l513_51314

/-- Given a line passing through points (3,4,1) and (8,2,3), 
    the z-coordinate of a point on this line with x-coordinate 7 is 13/5 -/
theorem line_point_z_coordinate :
  let p1 : Fin 3 → ℝ := ![3, 4, 1]
  let p2 : Fin 3 → ℝ := ![8, 2, 3]
  let direction : Fin 3 → ℝ := λ i => p2 i - p1 i
  let line (t : ℝ) : Fin 3 → ℝ := λ i => p1 i + t * direction i
  let t_value : ℝ := (7 - p1 0) / direction 0
  line t_value 2 = 13/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_point_z_coordinate_l513_51314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l513_51390

noncomputable def f (x : ℝ) : ℝ := Real.cos (x - Real.pi / 3)

noncomputable def g (x : ℝ) : ℝ := Real.cos (x / 2 - Real.pi / 4)

theorem axis_of_symmetry :
  (∀ x : ℝ, g (Real.pi - x) = g (Real.pi + x)) ∧
  (∀ x : ℝ, g x = f ((2 * x + Real.pi / 3) / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_l513_51390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_south_notation_l513_51367

/-- Represents the direction of walking --/
inductive Direction
  | North
  | South

/-- Represents a walk with a direction and distance --/
structure Walk where
  direction : Direction
  distance : ℝ

/-- Notation for a walk --/
def walkNotation (w : Walk) : ℝ :=
  match w.direction with
  | Direction.North => w.distance
  | Direction.South => -w.distance

theorem south_notation (north_walk south_walk : Walk) 
  (h1 : north_walk.direction = Direction.North)
  (h2 : north_walk.distance = 3)
  (h3 : walkNotation north_walk = 3)
  (h4 : south_walk.direction = Direction.South)
  (h5 : south_walk.distance = 2) :
  walkNotation south_walk = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_south_notation_l513_51367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l513_51358

noncomputable def f (x : ℝ) : ℝ := Real.tan (Real.arcsin (x^2))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Icc (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l513_51358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l513_51394

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola in 2D space -/
structure Parabola where
  p : ℝ

def origin : Point := ⟨0, 0⟩

def pointA : Point := ⟨1, -2⟩

def onParabola (C : Parabola) (P : Point) : Prop :=
  P.y^2 = 2 * C.p * P.x

def parallelToOA (L : Line) : Prop :=
  L.a / L.b = -2

noncomputable def distanceBetweenLines (L1 L2 : Line) : ℝ :=
  |L1.c - L2.c| / Real.sqrt (L1.a^2 + L1.b^2)

def intersectsParabola (L : Line) (C : Parabola) : Prop :=
  ∃ P : Point, onParabola C P ∧ L.a * P.x + L.b * P.y + L.c = 0

theorem parabola_line_intersection
  (C : Parabola)
  (h1 : C.p > 0)
  (h2 : onParabola C pointA) :
  ∃! L : Line,
    parallelToOA L ∧
    distanceBetweenLines L (Line.mk 2 1 0) = Real.sqrt 5 / 5 ∧
    intersectsParabola L C ∧
    L.a = 2 ∧ L.b = 1 ∧ L.c = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l513_51394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chain_lift_work_value_l513_51346

/-- The work done in lifting a chain of masses connected by springs -/
noncomputable def chain_lift_work (m D g l₀ : ℝ) : ℝ :=
  let n := 10  -- number of masses
  let E_r := (385 * m^2 * g^2) / (2 * D)  -- elastic energy
  let E_h := m * g * (45 * l₀ + 165 * m * g / D)  -- gravitational potential energy
  E_r + E_h

/-- Theorem stating the work done in lifting the chain -/
theorem chain_lift_work_value (m D g l₀ : ℝ) (hm : m > 0) (hD : D > 0) (hg : g > 0) (hl₀ : l₀ > 0) :
  chain_lift_work m D g l₀ = 1165 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chain_lift_work_value_l513_51346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_triangle_sides_l513_51388

/-- Represents a triangle with sides a, b, c forming an arithmetic progression --/
structure ArithmeticTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_arithmetic : b = (a + c) / 2
  positive : 0 < a ∧ 0 < b ∧ 0 < c

/-- The bisector of angle B in an arithmetic triangle --/
noncomputable def angle_bisector_B (t : ArithmeticTriangle) : ℝ :=
  Real.sqrt ((3 * t.a * t.c) / 4)

theorem arithmetic_triangle_sides (t : ArithmeticTriangle) 
  (h1 : angle_bisector_B t = t.a)
  (h2 : t.a + t.b + t.c = 21) :
  t.a = 6 ∧ t.b = 7 ∧ t.c = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_triangle_sides_l513_51388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_years_lent_to_B_is_two_l513_51357

/-- Represents the lending scenario described in the problem -/
structure LendingScenario where
  principal_B : ℚ
  principal_C : ℚ
  years_C : ℚ
  total_interest : ℚ
  interest_rate : ℚ

/-- Calculates the number of years A lent to B -/
def years_lent_to_B (scenario : LendingScenario) : ℚ :=
  (scenario.total_interest - scenario.principal_C * scenario.interest_rate * scenario.years_C) /
  (scenario.principal_B * scenario.interest_rate)

/-- Theorem stating that the number of years A lent to B is 2 -/
theorem years_lent_to_B_is_two (scenario : LendingScenario)
  (h1 : scenario.principal_B = 4000)
  (h2 : scenario.principal_C = 2000)
  (h3 : scenario.years_C = 4)
  (h4 : scenario.total_interest = 2200)
  (h5 : scenario.interest_rate = 11/80) :
  years_lent_to_B scenario = 2 := by
  sorry

def example_scenario : LendingScenario := {
  principal_B := 4000,
  principal_C := 2000,
  years_C := 4,
  total_interest := 2200,
  interest_rate := 11/80
}

#eval years_lent_to_B example_scenario

end NUMINAMATH_CALUDE_ERRORFEEDBACK_years_lent_to_B_is_two_l513_51357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_carrots_is_45_l513_51396

-- Define the number of other goats
def num_other_goats : ℕ := 9

-- Define the total number of carrots
def total_carrots : ℕ := 45

-- First condition: 6 carrots each for special goats, 3 for others, 6 left over
axiom condition1 : 6 + 6 + 3 * num_other_goats + 6 = total_carrots

-- Second condition: 7 carrots each for special goats, 5 for others, 14 short
axiom condition2 : 7 + 7 + 5 * num_other_goats = total_carrots + 14

-- Theorem to prove
theorem total_carrots_is_45 : total_carrots = 45 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_carrots_is_45_l513_51396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_minimum_coins_l513_51350

/-- Represents the coin distribution process --/
def coin_distribution (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n + 1 =>
    if n % 2 = 0 then n / 2
    else if n % 3 = 0 then n / 3
    else coin_distribution (n - (n % 3))

/-- Theorem stating the existence of N --/
theorem exists_minimum_coins :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → coin_distribution n > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_minimum_coins_l513_51350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_product_equality_l513_51308

theorem sqrt_sum_product_equality : 
  Real.sqrt 75 + (27 ^ (1/3 : ℝ)) * Real.sqrt 45 = 5 * Real.sqrt 3 + 9 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_product_equality_l513_51308
