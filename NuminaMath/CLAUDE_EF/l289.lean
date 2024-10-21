import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_negative_one_two_distinct_zeros_sum_of_zeros_positive_l289_28931

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * (x + 1)) / Real.exp x + (1 / 2) * x^2

theorem tangent_line_at_negative_one (a : ℝ) :
  a = 1 →
  ∃ m b : ℝ, m = Real.exp 1 - 1 ∧ b = Real.exp 1 - 1/2 ∧
  ∀ x y : ℝ, y = m * x + b ↔ y - f 1 (-1) = (deriv (f 1)) (-1) * (x + 1) :=
by sorry

theorem two_distinct_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ a < 0 :=
by sorry

theorem sum_of_zeros_positive (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ ≠ x₂ → f a x₁ = 0 → f a x₂ = 0 → x₁ + x₂ > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_negative_one_two_distinct_zeros_sum_of_zeros_positive_l289_28931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l289_28961

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (1 - x^2) + 1

def domain (f : ℝ → ℝ) : Set ℝ :=
  {x | ∃ y, f x = y ∧ 1 - x^2 > 0 ∧ x ≠ 0}

theorem f_domain :
  domain f = Set.Ioo (-1) 0 ∪ Set.Ioo 0 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l289_28961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_l289_28930

/-- A power function passing through (2, √2/2) with the given inequality condition -/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x ↦ x^a

/-- The theorem stating the range of b given the conditions -/
theorem b_range (a : ℝ) (h1 : f a 2 = Real.sqrt 2 / 2) 
  (h2 : ∀ b : ℝ, f a (2*b - 1) < f a (2 - b)) :
  ∀ b : ℝ, 1 < b ∧ b < 2 ↔ (f a (2*b - 1) < f a (2 - b) ∧ 2*b - 1 > 0 ∧ 2 - b > 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_range_l289_28930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_two_l289_28978

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 1 => -4/3
  | n + 2 => 1 / (sequenceA n + 1)
  | _ => 0  -- This case is not used in the problem, but needed for completeness

theorem seventh_term_is_two : sequenceA 7 = 2 := by
  -- Compute the intermediate terms
  have a3 : sequenceA 3 = -3 := by
    rw [sequenceA, sequenceA]
    norm_num
  
  have a5 : sequenceA 5 = -1/2 := by
    rw [sequenceA, a3]
    norm_num
  
  -- Compute the 7th term
  rw [sequenceA, a5]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_two_l289_28978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_equation_solution_l289_28934

-- Define the type for digits (0-9)
def Digit := Fin 10

-- Define the equation function
def satisfiesEquation (M X Y : Digit) (n : ℕ) : Prop :=
  let A := (10 : ℕ)^(n+1)
  let B := (A - 10) / 9
  ((M.val : ℕ) * A + (X.val : ℕ) * B + Y.val) * (10 * X.val + Y.val) = 
  ((X.val : ℕ) * A + (M.val : ℕ) * B + Y.val) * (10 * M.val + Y.val)

-- Define the main theorem
theorem digit_equation_solution :
  ∀ M X Y : Digit,
  M ≠ X ∧ X ≠ Y ∧ Y ≠ M →
  (∀ n : ℕ, n ≥ 1 → satisfiesEquation M X Y n) →
  ((M = ⟨1, by norm_num⟩ ∧ X = ⟨3, by norm_num⟩ ∧ Y = ⟨5, by norm_num⟩) ∨
   (M = ⟨3, by norm_num⟩ ∧ X = ⟨1, by norm_num⟩ ∧ Y = ⟨5, by norm_num⟩) ∨
   (M = ⟨0, by norm_num⟩ ∧ X = ⟨4, by norm_num⟩ ∧ Y = ⟨5, by norm_num⟩) ∨
   (M = ⟨4, by norm_num⟩ ∧ X = ⟨0, by norm_num⟩ ∧ Y = ⟨5, by norm_num⟩)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_equation_solution_l289_28934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_at_8_l289_28993

/-- A monic polynomial of degree 7 satisfying p(k) = k for k = 1, 2, 3, 4, 5, 6, 7 -/
def special_polynomial (p : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, p (x + y) = p x + p y - x * y) ∧ 
  (∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ, ∀ x : ℝ, p x = x^7 + a₆ * x^6 + a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀) ∧
  (∀ k : ℕ, k ≥ 1 ∧ k ≤ 7 → p k = k)

theorem special_polynomial_at_8 (p : ℝ → ℝ) (h : special_polynomial p) : p 8 = 5048 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_at_8_l289_28993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrilateral_pyramid_sphere_ratio_l289_28945

/-- Predicate to state that r is the radius of the inscribed sphere of a regular quadrilateral pyramid -/
def is_inscribed_sphere_radius (r : ℝ) : Prop :=
  ∃ (a h : ℝ), a > 0 ∧ h > 0 ∧ r = (a / h) * (Real.sqrt (a^2 + h^2) - a)

/-- Predicate to state that R is the radius of the circumscribed sphere of a regular quadrilateral pyramid -/
def is_circumscribed_sphere_radius (R : ℝ) : Prop :=
  ∃ (a h : ℝ), a > 0 ∧ h > 0 ∧ R = (2 * a^2 + h^2) / (2 * h)

/-- For a regular quadrilateral pyramid, the ratio of the radius of its circumscribed sphere
    to the radius of its inscribed sphere is greater than or equal to √2 + 1. -/
theorem regular_quadrilateral_pyramid_sphere_ratio (r R : ℝ) 
  (hr : r > 0) (hR : R > 0)
  (h_inscribed : is_inscribed_sphere_radius r)
  (h_circumscribed : is_circumscribed_sphere_radius R) :
  R / r ≥ Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quadrilateral_pyramid_sphere_ratio_l289_28945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_selection_l289_28917

def validSelection (S : Finset ℕ) : Prop :=
  S.card = 1011 ∧ 
  S ⊆ Finset.range 2022 ∧
  ∀ x y, x ∈ S → y ∈ S → x ≠ y → x + y ≠ 2021 ∧ x + y ≠ 2022

theorem unique_valid_selection :
  ∃! S, validSelection S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_selection_l289_28917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_splits_exist_l289_28998

def is_split (perm : List ℕ) (subset : Finset ℕ) : Prop :=
  ∃ i j k, i < j ∧ j < k ∧
    perm.get! i ∈ subset ∧
    perm.get! j ∉ subset ∧
    perm.get! k ∈ subset

theorem splits_exist (n : ℕ) (h : n ≥ 3) :
  ∀ (subsets : Finset (Finset ℕ)),
    subsets.card = n - 2 →
    (∀ s ∈ subsets, 2 ≤ s.card ∧ s.card ≤ n - 1) →
    (∀ s ∈ subsets, s ⊆ Finset.range n) →
    ∃ (perm : List ℕ), perm.toFinset = Finset.range n ∧
      ∀ s ∈ subsets, is_split perm s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_splits_exist_l289_28998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_condition_f_decreasing_interval_f_increasing_interval_l289_28989

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

-- Theorem for part I
theorem f_increasing_condition (a : ℝ) :
  (∀ x : ℝ, Monotone (f a)) ↔ a ≤ 0 := by sorry

-- Theorems for part II
theorem f_decreasing_interval :
  StrictMonoOn (f 1) (Set.Iio 0) := by sorry

theorem f_increasing_interval :
  StrictMonoOn (f 1) (Set.Ioi 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_condition_f_decreasing_interval_f_increasing_interval_l289_28989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_motion_l289_28976

-- Define the motion of the particle
noncomputable def s (t : ℝ) : ℝ := 2 * t^3

-- Define average speed
noncomputable def average_speed (f : ℝ → ℝ) (a b : ℝ) : ℝ := (f b - f a) / (b - a)

-- Define instantaneous speed (derivative)
noncomputable def instantaneous_speed (f : ℝ → ℝ) (t : ℝ) : ℝ := 
  deriv f t

-- Theorem statement
theorem particle_motion :
  average_speed s 1 2 = 14 ∧ instantaneous_speed s 1 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_motion_l289_28976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l289_28909

theorem negation_of_sin_inequality :
  (¬ ∀ x : ℝ, Real.sin x ≥ -1) ↔ (∃ x : ℝ, Real.sin x < -1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l289_28909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l289_28972

-- Define the inequality function
noncomputable def inequality (a : ℝ) (x : ℝ) : Prop :=
  (2 : ℝ)^(x^2 + 1) ≤ ((1/4) : ℝ)^(3/2 - a*x)

-- Define the theorem
theorem min_a_value :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 3 4 → inequality a x) →
  a ≥ 5/2 := by
  sorry

#check min_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l289_28972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l289_28919

-- Define the function g
def g (c d x : ℝ) : ℝ := c * x + d

-- State the theorem
theorem range_of_g (c d : ℝ) (h : c > 0) :
  Set.range (fun x => g c d x) ∩ Set.Icc (-1) 2 = Set.Icc (-c + d) (2*c + d) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l289_28919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zero_implies_a_bound_l289_28958

-- Define the natural exponential constant e
noncomputable def e : ℝ := Real.exp 1

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := e^x - a*x^2 - b*x - 1

-- State the theorem
theorem function_zero_implies_a_bound (a b : ℝ) :
  f a b 1 = 0 →
  (∃ x, 0 < x ∧ x < 1 ∧ f a b x = 0) →
  e - 2 < a ∧ a < 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_zero_implies_a_bound_l289_28958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_father_son_age_sum_l289_28948

theorem father_son_age_sum : ∀ (a b c : ℕ),
  (a ≥ 1 ∧ a ≤ 9) →  -- Father's age tens digit
  (b ≥ 0 ∧ b ≤ 9) →  -- Father's age units digit
  (c ≥ 3 ∧ c ≤ 9) →  -- Son's age units digit (teenager)
  (1000 * a + 100 * b + 10 + c) - ((10 * a + b) - (10 + c)) = 4289 →
  (10 * a + b) + (10 + c) = 59 := by
  sorry

#check father_son_age_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_father_son_age_sum_l289_28948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_five_halves_log_42_56_in_terms_of_a_b_l289_28953

-- Problem 1
theorem complex_expression_equals_five_halves :
  (2 + 1/4)^(1/2) - (-0.96)^0 - (3 + 3/8)^(-2/3) + (3/2)^(-2) + ((-32)^(-4))^(-3/4) = 5/2 := by
  sorry

-- Problem 2
theorem log_42_56_in_terms_of_a_b (a b : ℝ) (h1 : (14 : ℝ)^a = 6) (h2 : (14 : ℝ)^b = 7) :
  Real.log 56 / Real.log 42 = (3 - 2*b) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_five_halves_log_42_56_in_terms_of_a_b_l289_28953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_germination_problem_l289_28963

theorem seed_germination_problem (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate_plot1 total_germination_rate : ℚ) :
  seeds_plot1 = 500 →
  seeds_plot2 = 200 →
  germination_rate_plot1 = 30 / 100 →
  total_germination_rate = 35714285714285715 / 100000000000000000 →
  (let total_seeds := seeds_plot1 + seeds_plot2
   let germinated_seeds_plot1 := (seeds_plot1 : ℚ) * germination_rate_plot1
   let total_germinated_seeds := (total_seeds : ℚ) * total_germination_rate
   let germinated_seeds_plot2 := total_germinated_seeds - germinated_seeds_plot1
   germinated_seeds_plot2 / (seeds_plot2 : ℚ) = 1 / 2) := by
  sorry

#check seed_germination_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seed_germination_problem_l289_28963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olympic_number_property_l289_28926

-- Define the "Olympic number" property
def has_olympic_number_property (f : ℝ → ℝ) : Prop :=
  ∀ x₁ : ℝ, ∃! x₂ : ℝ, f x₁ + f x₂ = 2008

-- Define the functions
def f (x : ℝ) : ℝ := 2008 * x^3

noncomputable def g (x : ℝ) : ℝ := Real.log (2008 * x)

-- Theorem stating that f and g have the "Olympic number" property
theorem olympic_number_property :
  has_olympic_number_property f ∧ has_olympic_number_property g :=
by
  sorry

#check olympic_number_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olympic_number_property_l289_28926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_in_interval_l289_28903

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + Real.cos (2 * x)

theorem f_decreasing_in_interval :
  ∀ x ∈ Set.Ioo (-π/6 : ℝ) 0, 
    ∀ y ∈ Set.Ioo (-π/6 : ℝ) 0, 
      x < y → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_in_interval_l289_28903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l289_28914

/-- The inclination angle of a line is the angle between the positive x-axis and the line, 
    measured counterclockwise. -/
noncomputable def inclination_angle (a b c : ℝ) : ℝ := 
  Real.arctan (-a / b)

theorem line_inclination_angle : 
  inclination_angle 3 (Real.sqrt 3) 2 = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l289_28914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_min_value_achieved_l289_28996

noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x - x

theorem min_value_of_f (x : ℝ) (h : x ∈ Set.Icc (1/2) 2) : f x ≥ 1 := by
  sorry

theorem min_value_achieved : ∃ x ∈ Set.Icc (1/2) 2, f x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_min_value_achieved_l289_28996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_period_of_f_l289_28975

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi/3) * Real.sin (x + Real.pi/2)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_minimum_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  is_periodic f p ∧ p > 0 ∧ ∀ q, 0 < q ∧ q < p → ¬is_periodic f q

theorem minimum_period_of_f :
  is_minimum_positive_period f Real.pi := by
  sorry

#check minimum_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_period_of_f_l289_28975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_square_min_sum_l289_28987

theorem matrix_square_min_sum (p q r s : ℤ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 → s ≠ 0 →
  (Matrix.of !![p, q; r, s])^2 = Matrix.of !![8, 0; 0, 8] →
  (∃ (a b c d : ℤ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    (Matrix.of !![a, b; c, d])^2 = Matrix.of !![8, 0; 0, 8] ∧
    (abs a + abs b + abs c + abs d < abs p + abs q + abs r + abs s ∨
     abs a + abs b + abs c + abs d = abs p + abs q + abs r + abs s)) →
  abs p + abs q + abs r + abs s ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_square_min_sum_l289_28987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircles_ratio_theorem_l289_28918

/-- The number of congruent semicircles that fit on the diameter of a large semicircle -/
def N : ℕ := 13

/-- The radius of each small semicircle -/
noncomputable def r : ℝ := 1

/-- The ratio of the areas -/
def ratio : ℚ := 1 / 12

/-- The combined area of the small semicircles -/
noncomputable def A : ℝ := (N * Real.pi * r^2) / 2

/-- The area of the region inside the large semicircle but outside the small semicircles -/
noncomputable def B : ℝ := (Real.pi * (N * r)^2) / 2 - A

/-- Theorem stating that N is equal to 13 when the ratio of A to B is 1:12 -/
theorem semicircles_ratio_theorem (h : A / B = ratio) : N = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircles_ratio_theorem_l289_28918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crocodile_gena_investment_theorem_l289_28942

/-- Represents the investment and return for a project -/
structure Project where
  investment : ℝ
  return_value : ℝ

/-- The distance function between two projects -/
noncomputable def distance (p1 p2 : Project) : ℝ :=
  Real.sqrt ((p1.return_value - p2.return_value)^2 + (p1.investment - p2.investment)^2)

/-- The first project constraint -/
def first_project_constraint (p : Project) : Prop :=
  3 * p.return_value - 4 * p.investment - 30 = 0

/-- The second project constraint -/
def second_project_constraint (p : Project) : Prop :=
  p.investment^2 - 12 * p.investment + p.return_value^2 - 14 * p.return_value + 69 = 0

/-- The theorem stating the minimum distance and profitability -/
theorem crocodile_gena_investment_theorem
  (p1 p2 : Project)
  (h1 : first_project_constraint p1)
  (h2 : second_project_constraint p2)
  (h3 : p1.investment > 0)
  (h4 : p2.investment > 0)
  (h5 : ∀ q1 q2 : Project, first_project_constraint q1 → second_project_constraint q2 →
        q1.investment > 0 → q2.investment > 0 → distance p1 p2 ≤ distance q1 q2) :
  distance p1 p2 = 2.6 ∧ p1.return_value + p2.return_value - p1.investment - p2.investment = 16.84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crocodile_gena_investment_theorem_l289_28942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_APBQ_l289_28932

-- Define the curve C
def C (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the distance ratio condition
def distanceRatio (x y : ℝ) : Prop :=
  Real.sqrt ((x + 1)^2 + y^2) / abs (x + 2) = Real.sqrt 2 / 2

-- Define a chord AB through F(-1, 0) on C
def chordAB (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  C x₁ y₁ ∧ C x₂ y₂ ∧ ∃ m : ℝ, m ≠ 0 ∧ x₁ = m * y₁ - 1 ∧ x₂ = m * y₂ - 1

-- Define the midpoint M of AB
def midpointM (x₁ y₁ x₂ y₂ xₘ yₘ : ℝ) : Prop :=
  xₘ = (x₁ + x₂) / 2 ∧ yₘ = (y₁ + y₂) / 2

-- Define points P and Q where OM intersects C
def intersectionPQ (xₚ yₚ xq yq xₘ yₘ : ℝ) : Prop :=
  C xₚ yₚ ∧ C xq yq ∧ xₚ * yₘ = xₘ * yₚ ∧ xq * yₘ = xₘ * yq

-- Theorem statement
theorem min_area_APBQ :
  ∀ x₁ y₁ x₂ y₂ xₘ yₘ xₚ yₚ xq yq : ℝ,
  C x₁ y₁ → C x₂ y₂ → C xₚ yₚ → C xq yq →
  distanceRatio x₁ y₁ → distanceRatio x₂ y₂ → distanceRatio xₚ yₚ → distanceRatio xq yq →
  chordAB x₁ y₁ x₂ y₂ →
  midpointM x₁ y₁ x₂ y₂ xₘ yₘ →
  intersectionPQ xₚ yₚ xq yq xₘ yₘ →
  ∃ area : ℝ, area ≥ 2 ∧
    (∀ other_area : ℝ, (∃ x₁' y₁' x₂' y₂' xₘ' yₘ' xₚ' yₚ' xq' yq' : ℝ,
      C x₁' y₁' ∧ C x₂' y₂' ∧ C xₚ' yₚ' ∧ C xq' yq' ∧
      distanceRatio x₁' y₁' ∧ distanceRatio x₂' y₂' ∧ distanceRatio xₚ' yₚ' ∧ distanceRatio xq' yq' ∧
      chordAB x₁' y₁' x₂' y₂' ∧
      midpointM x₁' y₁' x₂' y₂' xₘ' yₘ' ∧
      intersectionPQ xₚ' yₚ' xq' yq' xₘ' yₘ' ∧
      other_area = abs ((x₁' - xₚ') * (y₂' - yₚ') - (x₂' - xₚ') * (y₁' - yₚ')) / 2) →
    area ≤ other_area) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_APBQ_l289_28932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_circumcircle_l289_28984

open EuclideanGeometry

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the angle bisector BK
variable (K : EuclideanSpace ℝ (Fin 2))

-- Define points M and N on sides BA and BC respectively
variable (M N : EuclideanSpace ℝ (Fin 2))

-- Define the condition that BK is an angle bisector
def is_angle_bisector (A B C K : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∠ A B K = ∠ C B K

-- Define the condition that M is on BA and N is on BC
def on_side (X Y Z : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = t • Y + (1 - t) • Z

-- Define the condition that ∠AKM = ∠CKN = 1/2 ∠ABC
def angle_condition (A B C K M N : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∠ A K M = ∠ C K N ∧ ∠ A K M = (1/2) * ∠ A B C

-- Define the circumcircle of a triangle
noncomputable def circumcircle (X Y Z : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {P | ∃ r : ℝ, r > 0 ∧ dist P X = r ∧ dist P Y = r ∧ dist P Z = r}

-- Define the tangent condition
def is_tangent (L : Set (EuclideanSpace ℝ (Fin 2))) (C : Set (EuclideanSpace ℝ (Fin 2))) (P : EuclideanSpace ℝ (Fin 2)) : Prop :=
  P ∈ L ∧ P ∈ C ∧ ∀ Q ∈ L, Q ≠ P → Q ∉ C

-- State the theorem
theorem tangent_to_circumcircle (A B C K M N : EuclideanSpace ℝ (Fin 2)) :
  is_angle_bisector A B C K →
  on_side M B A →
  on_side N B C →
  angle_condition A B C K M N →
  is_tangent {X | ∃ t : ℝ, X = t • A + (1 - t) • C} (circumcircle M B N) K :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_to_circumcircle_l289_28984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l289_28969

-- Define the curves
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := ((2 + t) / 6, Real.sqrt t)
noncomputable def C₂ (s : ℝ) : ℝ × ℝ := (-(2 + s) / 6, -Real.sqrt s)
noncomputable def C₃ (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the condition for C₃
def C₃_condition (p : ℝ × ℝ) : Prop :=
  2 * p.1 - p.2 = 0

-- Theorem statement
theorem intersection_points :
  ∃ (t₁ t₂ s₁ s₂ θ₁ θ₂ θ₃ θ₄ : ℝ),
    C₁ t₁ = C₃ θ₁ ∧ C₃_condition (C₃ θ₁) ∧ C₁ t₁ = (1/2, 1) ∧
    C₁ t₂ = C₃ θ₂ ∧ C₃_condition (C₃ θ₂) ∧ C₁ t₂ = (1, 2) ∧
    C₂ s₁ = C₃ θ₃ ∧ C₃_condition (C₃ θ₃) ∧ C₂ s₁ = (-1/2, -1) ∧
    C₂ s₂ = C₃ θ₄ ∧ C₃_condition (C₃ θ₄) ∧ C₂ s₂ = (-1, -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l289_28969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l289_28947

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Checks if the given equation represents the line -/
def is_equation_of_line (l : Line) (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, (x, y) ∈ {p | ∃ t : ℝ, p = (t, l.slope * (t - l.point.1) + l.point.2)} ↔ a * x + b * y + c = 0

theorem line_equation (l : Line) 
  (h1 : l.slope = 3)
  (h2 : l.point = (1, -2)) :
  is_equation_of_line l 3 (-1) (-5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l289_28947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_of_111111_l289_28938

theorem factorization_of_111111 : 
  ∃ (a b : ℕ), 
    (a * b = 111111) ∧ 
    (a % 10 ≠ 1) ∧ 
    (b % 10 ≠ 1) := by
  use 3, 37037
  constructor
  · norm_num
  constructor
  · norm_num
  · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_of_111111_l289_28938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_square_partition_l289_28967

theorem impossible_square_partition : ¬ ∃ (n m : ℕ) (x y : ℝ),
  x > 0 ∧ y > 0 ∧
  x - y = 1 ∧
  x * y = 1 ∧
  n * x + m * y = 10 ∧
  n + m = 100 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_square_partition_l289_28967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_range_l289_28966

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the focus
def focus : ℝ × ℝ := (0, 2)

-- Define the directrix
def directrix (y : ℝ) : Prop := y = -2

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem parabola_point_range (x0 y0 : ℝ) : 
  parabola x0 y0 → 
  (∃ (x y : ℝ), directrix y ∧ 
    distance x y (focus.1) (focus.2) = distance x0 y0 (focus.1) (focus.2)) → 
  y0 > 2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_range_l289_28966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coinciding_lines_l289_28994

-- Define a scalene triangle ABC
structure ScaleneTriangle where
  A : Real
  B : Real
  C : Real
  scalene : A ≠ B ∧ B ≠ C ∧ C ≠ A
  sum_180 : A + B + C = 180

-- Define the problem
theorem coinciding_lines (t : ScaleneTriangle) (h : t.B = 60) :
  let l₁ := fun (x y : Real) => x + (Real.sin t.A + Real.sin t.C) / Real.sqrt 3 * y + 1
  let l₂ := fun (x y : Real) => x * Real.tan (60 - t.C) + y * (Real.sin t.A - Real.sin t.C) - Real.tan ((t.C - t.A) / 2)
  ∀ x y, l₁ x y = 0 ↔ l₂ x y = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coinciding_lines_l289_28994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_classification_l289_28937

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)

-- Define the property of being an interior angle
def is_interior_angle (t : Triangle) (angle : Real) : Prop :=
  0 < angle ∧ angle < Real.pi

-- Define the property of being an obtuse triangle
def is_obtuse_triangle (t : Triangle) : Prop :=
  t.A > Real.pi/2 ∨ t.B > Real.pi/2 ∨ t.C > Real.pi/2

-- State the theorem
theorem triangle_classification (t : Triangle) :
  is_interior_angle t t.A →
  Real.sin t.A * Real.cos t.B * Real.tan t.C < 0 →
  is_obtuse_triangle t :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_classification_l289_28937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_cube_existence_l289_28928

theorem smaller_cube_existence (n : ℕ) (points : Finset (Fin n × Fin n × Fin n)) :
  n = 13 →
  points.card = 1956 →
  ∃ (i j k : Fin n), ∀ (x y z : Fin n),
    x.val < 1 →
    y.val < 1 →
    z.val < 1 →
    (i + x, j + y, k + z) ∉ points :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_cube_existence_l289_28928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_class_average_l289_28957

/-- Prove that given two classes with specific student counts and known averages, 
    we can determine the average marks of the first class. -/
theorem first_class_average (n₁ n₂ : ℕ) (avg₂ avg_all : ℝ) 
    (h₁ : n₁ = 55) (h₂ : n₂ = 48) 
    (h₃ : avg₂ = 58) (h₄ : avg_all = 59.067961165048544) : 
  (avg_all * (n₁ + n₂) - avg₂ * n₂) / n₁ = 60 := by
  sorry

#eval (59.067961165048544 * (55 + 48) - 58 * 48) / 55

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_class_average_l289_28957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_t_for_g_inequality_l289_28912

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - a^2) * x + Real.log x - 1/x

noncomputable def g (x : ℝ) : ℝ := x * f 1 x + x^2 + 1

theorem smallest_t_for_g_inequality :
  ∃ (t : ℤ), (∀ (s : ℤ), (∃ (x : ℝ), x > 0 ∧ s ≥ g x) → t ≤ s) ∧
  (∃ (x : ℝ), x > 0 ∧ t ≥ g x) ∧ t = 0 := by
  sorry

#check smallest_t_for_g_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_t_for_g_inequality_l289_28912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_appropriate_units_l289_28981

/-- Represents units of measurement -/
inductive MeasurementUnit
  | Meter
  | Gram
  | Other

/-- Represents a quantity with a numeric value and a unit -/
structure Quantity where
  value : ℕ
  unit : MeasurementUnit

/-- Determines if a unit is appropriate for a given quantity -/
def isAppropriateUnit (q : Quantity) (u : MeasurementUnit) : Prop :=
  match q.value, u with
  | 60, MeasurementUnit.Meter => q.unit = MeasurementUnit.Meter
  | 100, MeasurementUnit.Gram => q.unit = MeasurementUnit.Gram
  | _, _ => False

theorem appropriate_units 
  (walking_speed : Quantity) 
  (soap_weight : Quantity) 
  (h1 : walking_speed.value = 60)
  (h2 : soap_weight.value = 100) :
  isAppropriateUnit walking_speed MeasurementUnit.Meter ∧ 
  isAppropriateUnit soap_weight MeasurementUnit.Gram := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_appropriate_units_l289_28981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_car_is_big_green_without_trailer_l289_28980

-- Define the characteristics of a car
inductive Color
| Green
| Blue
| Other

inductive Size
| Small
| Big
| Other

structure Car where
  color : Color
  size : Size
  hasTrailer : Bool

-- Define the boys and their car collections
def Misha : List Car := [
  { color := Color.Other, size := Size.Other, hasTrailer := true },
  { color := Color.Other, size := Size.Small, hasTrailer := false },
  { color := Color.Green, size := Size.Other, hasTrailer := false }
]

def Vitya : List Car := [
  { color := Color.Other, size := Size.Other, hasTrailer := false },
  { color := Color.Green, size := Size.Small, hasTrailer := true }
]

def Kolya : List Car := [
  { color := Color.Other, size := Size.Big, hasTrailer := false },
  { color := Color.Blue, size := Size.Small, hasTrailer := true }
]

-- Define the theorem
theorem identical_car_is_big_green_without_trailer :
  ∃ (c : Car),
    c ∈ Misha ∧
    c ∈ Vitya ∧
    c ∈ Kolya ∧
    c.color = Color.Green ∧
    c.size = Size.Big ∧
    c.hasTrailer = false :=
by
  sorry

#check identical_car_is_big_green_without_trailer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_car_is_big_green_without_trailer_l289_28980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bishui_bay_pumping_time_l289_28925

/-- Represents a rectangular swimming pool -/
structure SwimmingPool where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the volume of a swimming pool in cubic meters -/
noncomputable def poolVolume (pool : SwimmingPool) : ℝ :=
  pool.length * pool.width * pool.depth

/-- Calculates the time required to pump out all water from a pool -/
noncomputable def pumpingTime (pool : SwimmingPool) (pumpRate : ℝ) : ℝ :=
  poolVolume pool / pumpRate

/-- Converts minutes to hours -/
noncomputable def minutesToHours (minutes : ℝ) : ℝ :=
  minutes / 60

theorem bishui_bay_pumping_time :
  let pool : SwimmingPool := ⟨50, 30, 1.8⟩
  let pumpRate : ℝ := 2.5
  minutesToHours (pumpingTime pool pumpRate) = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bishui_bay_pumping_time_l289_28925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_darcys_walking_speed_l289_28923

/-- Darcy's commute problem -/
theorem darcys_walking_speed :
  let distance_to_work : ℝ := 1.5
  let train_speed : ℝ := 20
  let additional_train_time : ℝ := 0.5 / 60
  let time_difference : ℝ := 25 / 60
  let train_time : ℝ := distance_to_work / train_speed + additional_train_time
  let walk_time : ℝ := train_time + time_difference
  let walk_speed : ℝ := distance_to_work / walk_time
  walk_speed = 3 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_darcys_walking_speed_l289_28923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ants_meeting_point_l289_28992

/-- Triangle with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point on the perimeter of a triangle -/
structure PerimeterPoint (t : Triangle) where
  distanceFromP : ℝ

/-- The triangle PQR -/
noncomputable def trianglePQR : Triangle := { a := 7, b := 8, c := 9 }

/-- The meeting point S -/
noncomputable def S : PerimeterPoint trianglePQR :=
  { distanceFromP := (7 + 8 + 9) / 2 }

/-- Distance from Q to S -/
noncomputable def QS (t : Triangle) (s : PerimeterPoint t) : ℝ :=
  if s.distanceFromP ≤ t.a then t.a - s.distanceFromP
  else if s.distanceFromP ≤ t.a + t.b then s.distanceFromP - t.a
  else t.c - (s.distanceFromP - t.a - t.b)

theorem ants_meeting_point :
  QS trianglePQR S = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ants_meeting_point_l289_28992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_parabola_midpoint_l289_28944

-- Define the parabola C: y² = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a line intersecting the parabola at two points
def intersecting_line (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ C x₁ y₁ ∧ C x₂ y₂ ∧ l x₁ y₁ ∧ l x₂ y₂

-- Define the midpoint of a line segment
def segment_midpoint (x₁ y₁ x₂ y₂ xₘ yₘ : ℝ) : Prop :=
  xₘ = (x₁ + x₂) / 2 ∧ yₘ = (y₁ + y₂) / 2

-- Theorem statement
theorem line_equation_through_parabola_midpoint :
  ∀ (l : ℝ → ℝ → Prop),
  intersecting_line l parabola →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), l x₁ y₁ ∧ l x₂ y₂ ∧ segment_midpoint x₁ y₁ x₂ y₂ 2 2) →
  (∀ (x y : ℝ), l x y ↔ x - y = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_through_parabola_midpoint_l289_28944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_decreasing_function_conditions_l289_28960

-- Define the function
noncomputable def f (k b x : ℝ) : ℝ := (k - 1) * x^(k^2 - 3) + b + 1

-- State the theorem
theorem linear_decreasing_function_conditions (k b : ℝ) :
  (∀ x, ∃ m c, f k b x = m * x + c) →  -- Linear function condition
  (∀ x₁ x₂, x₁ < x₂ → f k b x₁ > f k b x₂) →  -- Decreasing function condition
  k = -2 ∧ b ∈ Set.univ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_decreasing_function_conditions_l289_28960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_of_P_reflection_of_l_l289_28906

-- Define the line l: x + 2y - 2 = 0
def line_l (x y : ℝ) : Prop := x + 2 * y - 2 = 0

-- Define point P
def point_P : ℝ × ℝ := (-2, -1)

-- Define point A
def point_A : ℝ × ℝ := (1, 1)

-- Define the reflected point P'
noncomputable def reflected_P : ℝ × ℝ := (2/5, 19/5)

-- Define the reflected line l'
def line_l' (x y : ℝ) : Prop := x + 2 * y - 4 = 0

-- Theorem for the reflection of point P
theorem reflection_of_P :
  let (x, y) := reflected_P
  line_l ((x + point_P.1) / 2) ((y + point_P.2) / 2) ∧
  (y - point_P.2) / (x - point_P.1) = -1/2 := by
  sorry

-- Theorem for the reflection of line l
theorem reflection_of_l :
  ∀ (x y : ℝ), line_l x y ↔ line_l' (2 - x) (2 - y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_of_P_reflection_of_l_l289_28906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l289_28929

-- Define the sequence a_n
noncomputable def a : ℕ → ℝ := sorry

-- Define the sum S_n
noncomputable def S : ℕ → ℝ := sorry

-- Define the constant t
variable (t : ℝ)

-- Define the sequence b_n
noncomputable def b : ℕ → ℝ := sorry

-- Define the sequence c_n
noncomputable def c : ℕ → ℝ := sorry

-- Define the sum T_n
noncomputable def T : ℕ → ℝ := sorry

-- Conditions and theorems to prove
theorem sequence_properties :
  (t > 0) →
  (a 1 = 1) →
  (∀ n : ℕ, n ≥ 2 → 2*t*(S n) - (2*t + 1)*(S (n-1)) = 2*t) →
  (∀ n : ℕ, n ≥ 2 → b n = (1 + 1/(2*(1/(b (n-1) + 2)))) - 2) →
  (b 1 = 1) →
  (∀ n : ℕ, c n = n * (b n)) →
  (∃ r : ℝ, ∀ n : ℕ, n ≥ 2 → a (n+1) / (a n) = r) ∧
  (∀ n : ℕ, b n = (1/2)^(n-1)) ∧
  (∀ n : ℕ, T n = 4 - (2 + n : ℝ)/(2^(n-1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l289_28929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l289_28956

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (a + 1) / 2 * x^2 + 1

theorem f_properties :
  ∀ (a : ℝ),
  (∀ (x : ℝ), x > 0 → f a x = a * Real.log x + (a + 1) / 2 * x^2 + 1) →
  (a = -1/2 →
    (∃ (max min : ℝ),
      (∀ (x : ℝ), 1/Real.exp 1 ≤ x ∧ x ≤ Real.exp 1 → f a x ≤ max) ∧
      (∃ (x : ℝ), 1/Real.exp 1 ≤ x ∧ x ≤ Real.exp 1 ∧ f a x = max) ∧
      max = 1/2 + (Real.exp 1)^2/4 ∧
      (∀ (x : ℝ), 1/Real.exp 1 ≤ x ∧ x ≤ Real.exp 1 → min ≤ f a x) ∧
      (∃ (x : ℝ), 1/Real.exp 1 ≤ x ∧ x ≤ Real.exp 1 ∧ f a x = min) ∧
      min = 5/4)) ∧
  (a ≤ -1 →
    ∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂) ∧
  (a ≥ 0 →
    ∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
  (-1 < a ∧ a < 0 →
    ∃ (x₀ : ℝ), x₀ > 0 ∧
    (∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ x₀ → f a x₁ > f a x₂) ∧
    (∀ (x₁ x₂ : ℝ), x₀ ≤ x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) ∧
  (-1 < a ∧ a < 0 →
    (∀ (x : ℝ), x > 0 → f a x > 1 + a/2 * Real.log (-a)) →
    1/Real.exp 1 - 1 < a ∧ a < 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l289_28956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_saved_is_40_l289_28990

/-- A sale where 5 tickets are sold for the price of 3 -/
structure TicketSale where
  original_price : ℝ
  sale_price : ℝ
  sale_price_eq : sale_price = 3 * (original_price / 5)

/-- The percentage saved in the ticket sale -/
noncomputable def percent_saved (sale : TicketSale) : ℝ :=
  (5 * sale.original_price - sale.sale_price) / (5 * sale.original_price) * 100

/-- Theorem stating that the percentage saved is 40% -/
theorem percent_saved_is_40 (sale : TicketSale) : percent_saved sale = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_saved_is_40_l289_28990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_equals_28pi_over_15_l289_28927

-- Define the region
def region (x y : ℝ) : Prop :=
  0 ≤ y ∧ y ≤ 1 ∧ y = Real.sqrt (x - 1) ∧ x ≥ 0.5

-- Define the volume of revolution around y-axis
noncomputable def volume_of_revolution : ℝ := 
  Real.pi * ∫ y in Set.Icc 0 1, (y^2 + 1)^2

-- Theorem statement
theorem volume_equals_28pi_over_15 :
  volume_of_revolution = 28 * Real.pi / 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_equals_28pi_over_15_l289_28927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_row_sizes_l289_28941

def total_students : ℕ := 540

def is_valid_row_size (x : ℕ) : Bool :=
  x ≥ 20 && x ≤ 30 && (total_students % x = 0) && (total_students / x ≥ 12)

def valid_row_sizes : List ℕ :=
  (List.range 11).map (· + 20) |>.filter is_valid_row_size

theorem sum_of_valid_row_sizes :
  valid_row_sizes.sum = 77 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_row_sizes_l289_28941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_power_y_equals_y_power_x_l289_28979

-- Define the variables and conditions
variable (t : ℝ)
variable (h1 : t > 0)
variable (h2 : t ≠ 1)

-- Define x and y as functions of t
noncomputable def x (t : ℝ) : ℝ := t^(1/(t-1))
noncomputable def y (t : ℝ) : ℝ := t^(t/(t-1))

-- State the theorem
theorem x_power_y_equals_y_power_x (t : ℝ) (h1 : t > 0) (h2 : t ≠ 1) :
  (y t)^(x t) = (x t)^(y t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_power_y_equals_y_power_x_l289_28979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l289_28921

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := 4 * x^2 - (4/3) * y^2 = 1

-- Define the line
def tangent_line (x y : ℝ) : Prop := 4 * x - 3 * y - 6 = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- Define the left focus of the hyperbola
def left_focus : ℝ × ℝ := (-1, 0)

-- Theorem statement
theorem circle_equation :
  (∀ x y : ℝ, hyperbola x y → left_focus = (-1, 0)) →
  (∃ x y : ℝ, tangent_line x y ∧ ∃ r : ℝ, r > 0 ∧ ((x + 1)^2 + y^2 = r^2)) →
  ∀ x y : ℝ, circle_eq x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l289_28921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_geometric_progressions_l289_28922

def geometric_progression (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem sum_of_geometric_progressions
  (a b : ℕ → ℝ) (p q : ℝ) (ha : geometric_progression a p) (hb : geometric_progression b q) :
  (∃ r : ℝ, geometric_progression (fun n ↦ a n + b n) r) ↔ (p = q ∧ a 1 + b 1 ≠ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_geometric_progressions_l289_28922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_consistency_l289_28964

-- Define the system of equations
def equation1 (x a : ℝ) : Prop := 12 * x^2 + 48 * x - a + 36 = 0
def equation2 (x a : ℝ) : Prop := (a + 60) * x - 3 * (a - 20) = 0

-- Define the consistency of the system
def is_consistent (a : ℝ) : Prop :=
  ∃ x : ℝ, equation1 x a ∧ equation2 x a

-- Define the set of consistent a values
def consistent_a_values : Set ℝ := {-12, 0, 180}

-- Define the corresponding x values for each consistent a
noncomputable def x_for_a (a : ℝ) : ℝ :=
  if a = -12 then -2
  else if a = 0 then -1
  else if a = 180 then 2
  else 0  -- This case should never occur for consistent a values

-- Theorem statement
theorem system_consistency :
  ∀ a : ℝ, is_consistent a ↔ a ∈ consistent_a_values ∧
  ∀ x : ℝ, (equation1 x a ∧ equation2 x a) → x = x_for_a a :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_consistency_l289_28964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manufacturing_sector_degrees_l289_28949

/-- Represents the number of degrees in a full circle -/
def full_circle : ℚ := 360

/-- Represents the percentage of employees in the manufacturing department -/
def manufacturing_percentage : ℚ := 50

/-- Represents the total percentage (100%) -/
def total_percentage : ℚ := 100

/-- Calculates the number of degrees occupied by a sector in a circle graph
    given the percentage it represents -/
def sector_degrees (percentage : ℚ) : ℚ :=
  (percentage / total_percentage) * full_circle

/-- Theorem: The manufacturing sector occupies 180 degrees in the circle graph -/
theorem manufacturing_sector_degrees :
  sector_degrees manufacturing_percentage = 180 := by
  -- Unfold the definition of sector_degrees
  unfold sector_degrees
  -- Simplify the expression
  simp [manufacturing_percentage, total_percentage, full_circle]
  -- The proof is complete
  rfl

#eval sector_degrees manufacturing_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_manufacturing_sector_degrees_l289_28949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_triple_existence_l289_28905

/-- A quadratic triple is a triple of natural numbers (a, b, c) forming an arithmetic progression
    where b is coprime with a and c, and abc is a perfect square. -/
def QuadraticTriple (a b c : ℕ) : Prop :=
  ∃ d : ℕ, b - a = d ∧ c - b = d ∧
  Nat.Coprime b a ∧ Nat.Coprime b c ∧
  ∃ k : ℕ, a * b * c = k * k

/-- For any quadratic triple, there exists another quadratic triple
    that shares at least one number with it. -/
theorem quadratic_triple_existence (a b c : ℕ) (h : QuadraticTriple a b c) :
  ∃ x y z : ℕ, QuadraticTriple x y z ∧ (x = a ∨ x = b ∨ x = c ∨ y = a ∨ y = b ∨ y = c ∨ z = a ∨ z = b ∨ z = c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_triple_existence_l289_28905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_sequence_l289_28973

theorem divisibility_in_sequence (n : ℕ) (a : Fin (n + 1) → ℤ) :
  ∃ i j : Fin (n + 1), i ≠ j ∧ (n : ℤ) ∣ (a i - a j) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_sequence_l289_28973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_polar_equation_l289_28908

/-- Given a point P with polar coordinates (1,π), the polar coordinate equation
    of the line passing through P and perpendicular to the polar axis is ρ cos θ = -1. -/
theorem perpendicular_line_polar_equation (P : ℝ × ℝ) (h : P = (1, Real.pi)) :
  ∃ (ρ θ : ℝ), ρ * Real.cos θ = -1 ∧ 
    (ρ * Real.cos θ, ρ * Real.sin θ) = (-1, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_polar_equation_l289_28908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rubbish_clearing_days_l289_28913

/-- Represents the state of the rubbish piles -/
structure RubbishState where
  piles : List Nat

/-- Angel's possible moves -/
inductive AngelMove
  | clearPile : AngelMove
  | removeOne : AngelMove

/-- Demon's possible moves -/
inductive DemonMove
  | addOne : DemonMove
  | newPile : DemonMove

/-- The score for Angel's strategy -/
def angelScore (state : RubbishState) : Int :=
  (state.piles.map (fun x => min x 2)).sum - 1

/-- The score for Demon's strategy -/
def demonScore (state : RubbishState) : Nat :=
  let β := state.piles.filter (fun x => x ≥ 2) |>.length
  let α := state.piles.filter (fun x => x = 1) |>.length
  if α = 0 then 2 * β - 1 else 2 * β

/-- The initial state of the warehouse -/
def initialState : RubbishState :=
  { piles := List.replicate 100 100 }

/-- Helper function to apply moves (not implemented) -/
def applyMoves (n : Nat) (angelStrategy : Nat → AngelMove) (demonStrategy : Nat → DemonMove)
  (initialState : RubbishState) : RubbishState :=
sorry

/-- The main theorem stating that 199 days are both necessary and sufficient -/
theorem rubbish_clearing_days :
  (∃ (strategy : Nat → AngelMove), ∀ (demonStrategy : Nat → DemonMove),
    ∃ (n : Nat), n ≤ 199 ∧ applyMoves n strategy demonStrategy initialState = { piles := [] }) ∧
  (∃ (demonStrategy : Nat → DemonMove), ∀ (strategy : Nat → AngelMove),
    ∀ (n : Nat), n < 199 → applyMoves n strategy demonStrategy initialState ≠ { piles := [] }) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rubbish_clearing_days_l289_28913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_anne_sparkling_water_expenditure_l289_28965

/-- The annual cost of sparkling water for Mary Anne -/
def annual_sparkling_water_cost : ℚ :=
  let daily_consumption : ℚ := 1 / 5
  let days_in_year : ℕ := 365
  let bottle_cost : ℚ := 2
  daily_consumption * (days_in_year : ℚ) * bottle_cost

/-- Theorem stating Mary Anne's annual sparkling water expenditure -/
theorem mary_anne_sparkling_water_expenditure :
  annual_sparkling_water_cost = 146 := by
  unfold annual_sparkling_water_cost
  -- The proof steps would go here
  sorry

#eval annual_sparkling_water_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_anne_sparkling_water_expenditure_l289_28965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_clock_accuracy_l289_28982

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Converts time to minutes since midnight -/
def Time.toMinutes (t : Time) : ℕ :=
  t.hours * 60 + t.minutes

/-- Represents a clock that may run at a different speed than real time -/
structure Clock where
  speed : ℚ  -- Speed relative to real time (1 means accurate)

/-- Calculates the actual time given a clock reading and the clock's properties -/
noncomputable def actualTime (clockReading : Time) (clock : Clock) : Time :=
  sorry  -- Implementation not needed for the statement

theorem car_clock_accuracy : 
  ∀ (carClock : Clock) (initialTime shoppingEndTime finalTime : Time),
    -- Initial condition: both clocks show 12:00 noon
    initialTime = ⟨12, 0, by norm_num⟩ →
    -- After shopping: accurate watch shows 12:30, car clock shows 12:35
    (actualTime ⟨12, 35, by norm_num⟩ carClock).toMinutes - initialTime.toMinutes = 30 →
    -- Final car clock reading
    finalTime = ⟨19, 0, by norm_num⟩ →
    -- Prove that the actual time is 6:00 PM
    (actualTime finalTime carClock) = ⟨18, 0, by norm_num⟩ := by
  sorry

#check car_clock_accuracy

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_clock_accuracy_l289_28982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l289_28988

-- Define the sets M and N
def M : Set ℝ := {x | 1 < x + 1 ∧ x + 1 ≤ 3}
def N : Set ℝ := {x | x^2 - 2*x - 3 > 0}

-- Define the complement of a set in ℝ
def complement_R (S : Set ℝ) : Set ℝ := {x | x ∉ S}

-- State the theorem
theorem complement_intersection_theorem :
  (complement_R M) ∩ (complement_R N) = Set.Icc (-1 : ℝ) 0 ∪ Set.Ioc 2 3 := by
  sorry

#check complement_intersection_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_theorem_l289_28988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l289_28904

noncomputable section

-- Define the hyperbola
def Hyperbola (a b : ℝ) := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

-- Define distance from focus to asymptote
def focus_to_asymptote (b e : ℝ) : ℝ := b / e

-- Define focal length
def focal_length (a e : ℝ) : ℝ := 2 * a * e

-- Theorem statement
theorem hyperbola_focal_length 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : eccentricity a b = 2) 
  (h4 : focus_to_asymptote b 2 = Real.sqrt 3) : 
  focal_length a 2 = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_l289_28904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_implies_range_l289_28920

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(|x + a|)

theorem f_symmetry_implies_range (a : ℝ) (m n : ℝ) :
  (∀ x, f a (1 - x) = f a (1 + x)) →
  (∃ x₁ x₂, m ≤ x₁ ∧ x₁ ≤ n ∧ m ≤ x₂ ∧ x₂ ≤ n ∧ f a x₁ - f a x₂ = 3) →
  0 < n - m ∧ n - m ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_implies_range_l289_28920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l289_28959

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := tan x ^ 2 - 4 * tan x - 12 / tan x + 9 / (tan x ^ 2) - 3

-- State the theorem
theorem min_value_f :
  ∃ (min : ℝ), min = 3 + 8 * Real.sqrt 3 ∧
  ∀ x ∈ Set.Ioo (-π/2 : ℝ) 0, f x ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l289_28959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l289_28910

open Real Set

theorem symmetry_of_f (f : ℝ → ℝ) 
  (h1 : ∀ x ∈ Ioo 0 1, f x > 0)
  (h2 : ∀ x₁ x₂, x₁ ∈ Ioo 0 1 → x₂ ∈ Ioo 0 1 → f x₁ / f x₂ + f (1 - x₁) / f (1 - x₂) ≤ 2) :
  ∀ x ∈ Ioo 0 1, f x = f (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_f_l289_28910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_berthas_family_l289_28977

theorem berthas_family (total : ℕ) (daughters : ℕ) (granddaughters_per_mother : ℕ) :
  total = 40 →
  daughters = 8 →
  granddaughters_per_mother = 4 →
  ∃ (daughters_with_children : ℕ),
    daughters_with_children ≤ daughters ∧
    daughters_with_children * granddaughters_per_mother = total - daughters ∧
    total - daughters = 32 := by
  sorry

#check berthas_family

end NUMINAMATH_CALUDE_ERRORFEEDBACK_berthas_family_l289_28977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_alpha_l289_28915

noncomputable def α : ℝ := Real.arcsin (-Real.sqrt 10 / 10)

theorem tan_two_alpha (h1 : Real.sin α = -Real.sqrt 10 / 10) (h2 : π < α ∧ α < 3 * π / 2) : 
  Real.tan (2 * α) = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_alpha_l289_28915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_18_same_type_as_sqrt_2_sqrt_12_not_same_type_as_sqrt_2_sqrt_16_not_same_type_as_sqrt_2_sqrt_24_not_same_type_as_sqrt_2_l289_28907

-- Define the radical expressions as noncomputable
noncomputable def r12 : ℝ := Real.sqrt 12
noncomputable def r16 : ℝ := Real.sqrt 16
noncomputable def r18 : ℝ := Real.sqrt 18
noncomputable def r24 : ℝ := Real.sqrt 24

-- Theorem stating that √18 is of the same type as √2
theorem sqrt_18_same_type_as_sqrt_2 :
  ∃ (q : ℚ), r18 = q * Real.sqrt 2 :=
by sorry

-- Theorems stating that √12, √16, and √24 are not of the same type as √2
theorem sqrt_12_not_same_type_as_sqrt_2 :
  ¬ ∃ (q : ℚ), r12 = q * Real.sqrt 2 :=
by sorry

theorem sqrt_16_not_same_type_as_sqrt_2 :
  ¬ ∃ (q : ℚ), r16 = q * Real.sqrt 2 :=
by sorry

theorem sqrt_24_not_same_type_as_sqrt_2 :
  ¬ ∃ (q : ℚ), r24 = q * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_18_same_type_as_sqrt_2_sqrt_12_not_same_type_as_sqrt_2_sqrt_16_not_same_type_as_sqrt_2_sqrt_24_not_same_type_as_sqrt_2_l289_28907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carton_volume_theorem_l289_28943

/-- Represents a pyramid-shaped carton formed by four congruent triangles -/
structure Carton where
  edge1 : ℝ  -- Length of two opposite edges
  edge2 : ℝ  -- Length of the other four edges

/-- Calculates the volume of the carton -/
noncomputable def carton_volume (c : Carton) : ℝ :=
  (16 / 3) * Real.sqrt 23

/-- Theorem stating that the volume of the carton with given dimensions is (16/3) * √23 -/
theorem carton_volume_theorem (c : Carton) (h1 : c.edge1 = 4) (h2 : c.edge2 = 10) :
  carton_volume c = (16 / 3) * Real.sqrt 23 := by
  sorry

#check carton_volume_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carton_volume_theorem_l289_28943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_payback_time_l289_28985

-- Define the parameters
noncomputable def initial_cost : ℝ := 25000
noncomputable def monthly_revenue : ℝ := 4000
noncomputable def monthly_expenses : ℝ := 1500

-- Define monthly profit
noncomputable def monthly_profit : ℝ := monthly_revenue - monthly_expenses

-- Define the time to pay back
noncomputable def time_to_pay_back : ℝ := initial_cost / monthly_profit

-- Theorem statement
theorem store_payback_time :
  time_to_pay_back = 10 :=
by
  -- Unfold the definitions
  unfold time_to_pay_back initial_cost monthly_profit monthly_revenue monthly_expenses
  -- Perform the calculation
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_payback_time_l289_28985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_points_partition_l289_28955

/-- Function mapping rational numbers to natural numbers -/
def H : ℚ → ℕ := sorry

/-- Set A of points with rational coordinates -/
def A : Set (ℚ × ℚ) :=
  {p | H p.1 ≤ H p.2}

/-- Set B of points with rational coordinates -/
def B : Set (ℚ × ℚ) :=
  {p | H p.1 > H p.2}

/-- Theorem stating the properties of sets A and B -/
theorem rational_points_partition :
  (∀ y : ℚ, Set.Finite {x : ℚ | (x, y) ∈ A}) ∧
  (∀ x : ℚ, Set.Finite {y : ℚ | (x, y) ∈ B}) ∧
  A ∪ B = Set.prod Set.univ Set.univ ∧
  A ∩ B = ∅ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_points_partition_l289_28955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cooking_probability_l289_28952

-- Define a finite set of courses
def Courses : Type := Fin 4

-- Define the probability measure on the set of courses
noncomputable def P : Courses → ℝ := λ _ => 1 / 4

-- Theorem statement
theorem cooking_probability :
  ∀ (cooking : Courses), P cooking = 1 / 4 := by
  intro cooking
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cooking_probability_l289_28952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_acute_angle_l289_28997

theorem sin_minus_cos_acute_angle (θ : ℝ) (b : ℝ) 
  (h1 : 0 < θ ∧ θ < Real.pi / 2) 
  (h2 : Real.cos (2 * θ) = b) : 
  Real.sin θ - Real.cos θ = Real.sqrt (1 - b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_acute_angle_l289_28997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_partition_l289_28991

theorem sequence_partition (n : ℕ) (a : Fin (2 * n) → ℤ)
  (h : ∀ k : ℤ, (Finset.filter (fun i => a i = k) Finset.univ).card ≤ n) :
  ∃ (b c : Fin n → Fin (2 * n)),
    (StrictMono b ∧ StrictMono c) ∧
    (∀ i : Fin (2 * n), ∃! j : Fin n, b j = i ∨ c j = i) ∧
    (∀ i : Fin n, a (b i) ≠ a (c i)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_partition_l289_28991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_three_real_zeros_l289_28935

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 - x

-- Statement 1: Center of symmetry
theorem center_of_symmetry (a : ℝ) :
  (∀ x : ℝ, f a (2 - x) = 2 * f a 1 - f a x) ↔ a = -1 :=
by sorry

-- Statement 2: Three real zeros
theorem three_real_zeros (a : ℝ) :
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_three_real_zeros_l289_28935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_is_54_l289_28983

def class_size (total_stickers : ℕ) (friends : ℕ) (stickers_per_student : ℚ) (stickers_left : ℚ) : ℕ :=
  let stickers_to_friends := (friends * (friends + 1)) / 2
  let remaining_stickers := total_stickers - stickers_to_friends
  let other_students := ((remaining_stickers : ℚ) - stickers_left) / stickers_per_student
  ⌊other_students⌋.toNat + friends + 1

#eval class_size 300 10 (11/2) (15/2)

theorem class_size_is_54 :
  class_size 300 10 (11/2) (15/2) = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_size_is_54_l289_28983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_range_l289_28933

open Real

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  h1 : ∀ x > 0, 9 * f x < x * (deriv f) x ∧ x * (deriv f) x < 10 * f x
  h2 : ∀ x > 0, f x > 0

/-- The main theorem about the range of f(2)/f(1) -/
theorem special_function_range (φ : SpecialFunction) : 
  2^9 < φ.f 2 / φ.f 1 ∧ φ.f 2 / φ.f 1 < 2^10 := by
  sorry

#check special_function_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_range_l289_28933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_cost_per_metre_l289_28950

/-- The cost price per metre of cloth -/
noncomputable def cost_price_per_metre (total_cost : ℝ) (total_length : ℝ) : ℝ :=
  total_cost / total_length

/-- Theorem: Given a total cost of $407 for 9.25 m of cloth, the cost price per metre is $44 -/
theorem cloth_cost_per_metre :
  let total_cost : ℝ := 407
  let total_length : ℝ := 9.25
  cost_price_per_metre total_cost total_length = 44 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_cost_per_metre_l289_28950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_in_interval_l289_28940

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + x - 3

-- State the theorem
theorem f_has_one_zero_in_interval :
  (∀ x, x ∈ Set.Ioo 0 1 → ContinuousAt f x) →
  (∀ x y, x ∈ Set.Ioo 0 1 → y ∈ Set.Ioo 0 1 → x < y → f x < f y) →
  f 0 < 0 →
  f 1 > 0 →
  ∃! x, x ∈ Set.Ioo 0 1 ∧ f x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_in_interval_l289_28940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_intersection_length_l289_28974

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := ((1/2) * t, (Real.sqrt 3 / 2) * t - 1)

-- Define the curve C in polar coordinates
def curve_C (ρ θ : ℝ) : Prop := ρ^2 - 2*ρ*(Real.sin θ) - 3 = 0

-- Define the curve C in rectangular coordinates
def curve_C_rect (x y : ℝ) : Prop := (x - 0)^2 + (y - 1)^2 = 4

-- State the theorem
theorem line_curve_intersection_length :
  ∃ (a b : ℝ), 
    (∃ t, line_l t = (a, b)) →  -- Point (a, b) lies on line l
    curve_C_rect a b →          -- Point (a, b) lies on curve C
    (a^2 + b^2).sqrt = 2 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_curve_intersection_length_l289_28974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l289_28962

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x)

noncomputable def g (x : ℝ) : ℝ := -Real.sqrt 3 * Real.cos (4 * x)

def shift_right (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x => f (x - a)

def compress_horizontal (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => f (k * x)

theorem function_transformation (x : ℝ) : 
  (compress_horizontal (shift_right f (Real.pi / 4)) 2) x = g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l289_28962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_berries_cheese_cost_l289_28900

/-- The cost of a basket of berries, a carton of milk, a loaf of bread, and a block of cheese -/
def total_cost : ℝ := 25

/-- The price of a carton of milk is twice the price of a loaf of bread -/
def milk_bread_relation (m l : ℝ) : Prop := m = 2 * l

/-- The cost of a block of cheese is equal to the price of a basket of berries plus $2 -/
def cheese_berries_relation (c b : ℝ) : Prop := c = b + 2

/-- The sum of the prices of berries, milk, bread, and cheese equals the total cost -/
def sum_equals_total (b m l c : ℝ) : Prop := b + m + l + c = total_cost

/-- The cost of berries and cheese together is $10 -/
theorem berries_cheese_cost (b c : ℝ) : 
  (∃ (m l : ℝ), milk_bread_relation m l ∧ cheese_berries_relation c b ∧ sum_equals_total b m l c) → 
  b + c = 10 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_berries_cheese_cost_l289_28900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_five_five_l289_28968

-- Define the custom operation
def custom_op : ℕ → ℕ → ℕ := sorry

-- Axiom: custom_op is distributive over addition
axiom custom_op_distrib (a b c : ℕ) : custom_op a (b + c) = custom_op a b + custom_op a c

-- Given condition
axiom given_equation : custom_op 3 2 + custom_op 2 3 = 7

-- Theorem to prove
theorem custom_op_five_five : custom_op 5 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_five_five_l289_28968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_transport_cost_l289_28971

/-- The cost in dollars per kilogram for NASA Space Shuttle transport -/
noncomputable def transport_cost_per_kg : ℝ := 25000

/-- The mass of the sensor device in grams -/
noncomputable def sensor_mass_g : ℝ := 350

/-- The mass of the communication module in grams -/
noncomputable def comm_module_mass_g : ℝ := 150

/-- Conversion factor from grams to kilograms -/
noncomputable def g_to_kg : ℝ := 1 / 1000

theorem total_transport_cost :
  let sensor_mass_kg := sensor_mass_g * g_to_kg
  let comm_module_mass_kg := comm_module_mass_g * g_to_kg
  let total_mass_kg := sensor_mass_kg + comm_module_mass_kg
  total_mass_kg * transport_cost_per_kg = 12500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_transport_cost_l289_28971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l289_28902

/-- A triangle with sides 2, m, and 5 satisfies the triangle inequality theorem -/
def is_valid_triangle (m : ℝ) : Prop :=
  2 + m > 5 ∧ 2 + 5 > m ∧ m + 5 > 2

/-- The expression to be simplified -/
noncomputable def expression (m : ℝ) : ℝ :=
  (9 - 6*m + m^2).sqrt - (m^2 - 14*m + 49).sqrt

theorem simplify_expression (m : ℝ) (h : is_valid_triangle m) :
  expression m = 2*m - 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l289_28902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l289_28946

noncomputable section

-- Define the function f
def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ) + Real.sqrt 3 * Real.cos (2 * x + φ)

-- Define the function g
def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 3)

theorem function_properties :
  ∃ φ : ℝ, 0 < |φ| ∧ |φ| < Real.pi ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 4), StrictMonoOn (fun x => -(f x φ)) {x}) ∧
  (∀ x, f x φ = f (Real.pi / 2 - x) φ) ∧
  φ = 2 * Real.pi / 3 ∧
  (∀ x, g x = f (x + Real.pi / 3) φ) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l289_28946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_value_l289_28939

/-- The perpendicular bisector of a line segment passes through its midpoint -/
axiom perpendicular_bisector_passes_through_midpoint 
  (A B M : ℝ × ℝ) (c : ℝ) :
  (∀ (x y : ℝ), x + y = c ↔ (x, y) ∈ Set.range (λ t ↦ (t, c - t))) →
  (M = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2)) →
  (M.fst + M.snd = c)

/-- The value of c for which x + y = c is the perpendicular bisector 
    of the line segment from (2,4) to (6,8) -/
theorem perpendicular_bisector_value : 
  ∃ c : ℝ, (∀ (x y : ℝ), x + y = c ↔ (x, y) ∈ Set.range (λ t ↦ (t, c - t))) ∧
           c = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_value_l289_28939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_PQ_value_l289_28911

-- Define the points
variable (P Q R S T : ℝ × ℝ)

-- Define the conditions
axiom collinear : ∃ (a b : ℝ), P.1 < Q.1 ∧ Q.1 < R.1 ∧ R.1 < S.1 ∧
  P.2 = a * P.1 + b ∧ Q.2 = a * Q.1 + b ∧ R.2 = a * R.1 + b ∧ S.2 = a * S.1 + b

axiom not_collinear : ¬∃ (a b : ℝ), T.2 = a * T.1 + b ∧ 
  P.2 = a * P.1 + b ∧ Q.2 = a * Q.1 + b ∧ R.2 = a * R.1 + b ∧ S.2 = a * S.1 + b

noncomputable def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

axiom PQ_eq_SR : distance P Q = distance S R
axiom QR_eq_15 : distance Q R = 15
axiom QT_eq_RT : distance Q T = distance R T
axiom QT_eq_13 : distance Q T = 13

noncomputable def perimeter (A B C : ℝ × ℝ) : ℝ := distance A B + distance B C + distance C A

axiom perimeter_relation : perimeter P T S = 2 * perimeter Q T R

-- The theorem to prove
theorem PQ_value : distance P Q = 11.625 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_PQ_value_l289_28911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_AB_touches_angle_sides_l289_28936

-- Define the angle
variable (α : ℝ)

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the two circles
variable (circleA circleB : Circle)

-- Define the condition that the circles touch each other
def circles_touch (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

-- Define the condition that a circle touches both sides of the angle
def circle_touches_angle_sides (c : Circle) (α : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), 
    (x1 = c.center.1 + c.radius * Real.cos α ∧ y1 = c.center.2 + c.radius * Real.sin α) ∧
    (x2 = c.center.1 + c.radius ∧ y2 = c.center.2)

-- Define the circle with diameter AB
noncomputable def circle_AB (circleA circleB : Circle) : Circle :=
  { center := ((circleA.center.1 + circleB.center.1) / 2, (circleA.center.2 + circleB.center.2) / 2),
    radius := ((circleA.center.1 - circleB.center.1)^2 + (circleA.center.2 - circleB.center.2)^2).sqrt / 2 }

-- State the theorem
theorem circle_AB_touches_angle_sides 
  (h1 : circles_touch circleA circleB)
  (h2 : circle_touches_angle_sides circleA α)
  (h3 : circle_touches_angle_sides circleB α) :
  circle_touches_angle_sides (circle_AB circleA circleB) α := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_AB_touches_angle_sides_l289_28936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_6_simplest_l289_28924

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : ℚ) : Prop :=
  ∃ m : ℚ, m * m = n

-- Define a function to check if a number has any perfect square factors other than 1
def hasNoSquareFactors (n : ℚ) : Prop :=
  ∀ m : ℚ, m > 1 → m < n → isPerfectSquare m → ¬(∃ k : ℚ, n = m * k)

-- Define the simplicity criterion for square roots
def isSimplestSqrt (n : ℚ) : Prop :=
  n > 0 ∧ hasNoSquareFactors n

-- Theorem stating that √6 is the simplest among the given options
theorem sqrt_6_simplest :
  isSimplestSqrt 6 ∧
  (¬isSimplestSqrt (2/10) ∨ 
   ¬isSimplestSqrt (1/2) ∨ 
   ¬isSimplestSqrt 12) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_6_simplest_l289_28924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_participants_optimal_time_l289_28916

noncomputable def total_distance : ℝ := 84
noncomputable def walking_speed : ℝ := 5
noncomputable def cycling_speed : ℝ := 20

noncomputable def optimal_time (S : ℝ) (v_walk : ℝ) (v_cycle : ℝ) : ℝ :=
  let α : ℝ := 4 / 7
  (3 - 2 * α) / v_cycle * S

theorem three_participants_optimal_time :
  optimal_time total_distance walking_speed cycling_speed = 7.8 := by
  sorry

#check three_participants_optimal_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_participants_optimal_time_l289_28916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_property_l289_28901

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Define a chord passing through the left focus
def chord_through_left_focus (x1 y1 x2 y2 : ℝ) : Prop :=
  is_on_ellipse x1 y1 ∧ is_on_ellipse x2 y2 ∧
  ∃ t : ℝ, x1 + t * (x2 - x1) = left_focus.1 ∧ y1 + t * (y2 - y1) = left_focus.2

-- Define the perimeter of the incircle of triangle ABF2
noncomputable def incircle_perimeter (x1 y1 x2 y2 : ℝ) : ℝ := Real.pi

theorem chord_property (x1 y1 x2 y2 : ℝ) :
  is_on_ellipse x1 y1 ∧ is_on_ellipse x2 y2 ∧
  chord_through_left_focus x1 y1 x2 y2 ∧
  incircle_perimeter x1 y1 x2 y2 = Real.pi →
  |y1 - y2| = 5/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_property_l289_28901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_decreasing_intervals_l289_28954

noncomputable def f (x : ℝ) : ℝ := x^3 - (3/2)*x^2 - 6*x + 4

theorem f_increasing_decreasing_intervals :
  (∀ x y : ℝ, x < y ∧ ((x < -1 ∧ y < -1) ∨ (x > 2 ∧ y > 2)) → f x < f y) ∧
  (∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 2 → f x > f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_decreasing_intervals_l289_28954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_period_pi_l289_28986

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * (x + Real.pi / 4))

theorem f_is_odd_and_period_pi : 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x + Real.pi) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_and_period_pi_l289_28986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_function_a_range_l289_28999

-- Define the function f as noncomputable due to the use of Real.instPowReal
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2*a - 3)*x + 1 else a^x

-- State the theorem
theorem strictly_increasing_function_a_range (a : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 
  a > 3/2 ∧ a ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_strictly_increasing_function_a_range_l289_28999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_blue_cells_l289_28951

/-- Represents a color configuration of a grid -/
structure ColorConfig (m n : ℕ) :=
  (colors : Fin m → Fin m → Fin n)

/-- Checks if a color configuration satisfies the condition that 
    for any cell, its row and column contain all colors -/
def satisfiesCondition (m n : ℕ) (config : ColorConfig m n) : Prop :=
  ∀ i j : Fin m, ∀ c : Fin n, 
    (∃ k : Fin m, config.colors k j = c) ∨ 
    (∃ k : Fin m, config.colors i k = c)

/-- Counts the number of cells with a specific color -/
def countColor (m n : ℕ) (config : ColorConfig m n) (c : Fin n) : ℕ :=
  (Finset.univ.filter (λ i : Fin m × Fin m => config.colors i.1 i.2 = c)).card

/-- Theorem stating the maximum number of blue cells for n = 2 and n = 25 -/
theorem max_blue_cells :
  (∃ (config : ColorConfig 50 2), 
    satisfiesCondition 50 2 config ∧ 
    countColor 50 2 config 0 = 2450) ∧
  (∃ (config : ColorConfig 50 25), 
    satisfiesCondition 50 25 config ∧ 
    countColor 50 25 config 0 = 1300) := by
  sorry

#check max_blue_cells

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_blue_cells_l289_28951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_multiples_of_three_squares_l289_28970

theorem count_even_multiples_of_three_squares : 
  (Finset.filter (λ k : ℕ ↦ 36 * k * k < 5000) (Finset.range 12)).card = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_even_multiples_of_three_squares_l289_28970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l289_28995

theorem geometric_sequence_sum (a r : ℝ) (hr : r ≠ 1) :
  let s10 := a * (1 - r^10) / (1 - r)
  let s20 := a * (1 - r^20) / (1 - r)
  let s30 := a * (1 - r^30) / (1 - r)
  s20 = 21 ∧ s30 = 49 → s10 = 7 ∨ s10 = 63 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l289_28995
