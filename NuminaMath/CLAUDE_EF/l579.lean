import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_composition_l579_57939

/-- Represents the composition of a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  other : ℝ
  sum_to_one : ryegrass + other = 1

/-- The final mixture of X and Y -/
def final_mixture (x : SeedMixture) (y : SeedMixture) (prop_x : ℝ) : SeedMixture where
  ryegrass := prop_x * x.ryegrass + (1 - prop_x) * y.ryegrass
  other := prop_x * x.other + (1 - prop_x) * y.other
  sum_to_one := by sorry

theorem mixture_composition 
  (x : SeedMixture)
  (y : SeedMixture)
  (hx : x.ryegrass = 0.4)
  (hy : y.ryegrass = 0.25)
  (h_final : ∃ (prop_x : ℝ), (final_mixture x y prop_x).ryegrass = 0.3) :
  ∃ (prop_x : ℝ), abs (prop_x - 0.3333) < 0.0001 := by
  sorry

#check mixture_composition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_composition_l579_57939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_function_negative_range_l579_57970

-- Define the function f
noncomputable def f (x a c : ℝ) : ℝ := -1/2 * x^2 + 1/2 * a * (5-a) * x + c

-- Theorem 1
theorem inequality_solution (a : ℝ) :
  f 2 a 16 > 0 ↔ -2 < a ∧ a < 7 := by sorry

-- Theorem 2
theorem function_negative_range (c : ℝ) :
  (∀ x ≤ 1, f x 4 c < 0) ↔ c < -3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_function_negative_range_l579_57970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l579_57954

-- Define the function f(x) = ax^2 - x - ln(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x - Real.log x

-- Theorem statement
theorem f_properties :
  -- Part 1: Minimum value when a = 3/8
  (∃ (x : ℝ), x > 0 ∧ f (3/8) x = -1/2 - Real.log 2 ∧ ∀ (y : ℝ), y > 0 → f (3/8) y ≥ f (3/8) x) ∧
  -- Part 2: Exactly one zero when -1 ≤ a ≤ 0
  (∀ (a : ℝ), -1 ≤ a ∧ a ≤ 0 → ∃! (x : ℝ), x > 0 ∧ f a x = 0) ∧
  -- Part 3: Two zeros iff 0 < a < 1
  (∀ (a : ℝ), (∃ (x y : ℝ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ f a x = 0 ∧ f a y = 0) ↔ (0 < a ∧ a < 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l579_57954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l579_57924

theorem tan_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.tan (2*α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l579_57924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_group_c_forms_triangle_l579_57998

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle 
    must be greater than the length of the third side. -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if a group of stick lengths can form a triangle -/
def can_form_triangle (lengths : List ℝ) : Prop :=
  lengths.length = 3 ∧ 
  satisfies_triangle_inequality lengths[0]! lengths[1]! lengths[2]!

/-- The given groups of stick lengths -/
def stick_groups : List (List ℝ) :=
  [[1, 2, 3], [1, 2, 4], [2, 3, 4], [2, 2, 4]]

theorem only_group_c_forms_triangle : 
  ∃! group, group ∈ stick_groups ∧ can_form_triangle group ∧ group = [2, 3, 4] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_group_c_forms_triangle_l579_57998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_and_area_l579_57992

/-- Given a triangle ABC with specific cosine values for angles A and B, and side length a,
    prove the cosine of A+B and the area of the triangle. -/
theorem triangle_cosine_and_area (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  Real.cos A = (2 * Real.sqrt 5) / 5 →
  Real.cos B = (3 * Real.sqrt 10) / 10 →
  a = Real.sqrt 10 →
  -- Conclusions
  Real.cos (A + B) = Real.sqrt 2 / 2 ∧
  (1 / 2 : ℝ) * a * b * Real.sin C = 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_and_area_l579_57992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laptop_price_calculation_l579_57973

theorem laptop_price_calculation (original_price discount_rate tax_rate : ℝ) 
  (h1 : original_price = 1200)
  (h2 : discount_rate = 0.30)
  (h3 : tax_rate = 0.12) :
  original_price * (1 - discount_rate) * (1 + tax_rate) = 940.80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laptop_price_calculation_l579_57973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_path_length_max_path_achievable_l579_57903

/-- Represents a rectangular grid --/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a path on the grid --/
structure GridPath :=
  (grid : Grid)
  (length : ℕ)

/-- The maximum number of diagonals that can be traversed in a closed path --/
def max_diagonals (g : Grid) : ℕ := g.rows * (g.cols / 2)

theorem max_path_length (g : Grid) (p : GridPath) (h1 : g.rows = 5) (h2 : g.cols = 8) (h3 : p.grid = g) :
  p.length ≤ max_diagonals g :=
sorry

theorem max_path_achievable (g : Grid) (h1 : g.rows = 5) (h2 : g.cols = 8) :
  ∃ p : GridPath, p.grid = g ∧ p.length = max_diagonals g :=
sorry

#eval max_diagonals { rows := 5, cols := 8 }  -- Should output 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_path_length_max_path_achievable_l579_57903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_theorem_l579_57944

/-- The polar equation of the circle -/
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 + 2*ρ*(Real.cos θ + Real.sqrt 3 * Real.sin θ) = 5

/-- The line θ = 0 -/
def line_theta_zero (θ : ℝ) : Prop := θ = 0

/-- The length of the chord intercepted by θ = 0 -/
noncomputable def chord_length : ℝ := 2 * Real.sqrt 6

theorem chord_length_theorem :
  ∀ ρ θ : ℝ, polar_equation ρ θ → line_theta_zero θ →
  ∃ x y : ℝ, x^2 + y^2 + 2*x + 2*Real.sqrt 3*y = 5 ∧
  chord_length = 2 * Real.sqrt ((3:ℝ)^2 - (Real.sqrt 3)^2) := by
  sorry

#check chord_length_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_theorem_l579_57944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_is_correct_l579_57959

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = x^2 + 10*x + 19
def parabola2 (x y : ℝ) : Prop := x = y^2 + 36*y + 325

-- Define the point of tangency
noncomputable def tangent_point : ℝ × ℝ := (-9/2, -35/2)

-- Theorem statement
theorem tangent_point_is_correct :
  let (x, y) := tangent_point
  parabola1 x y ∧ parabola2 x y ∧
  ∀ (x' y' : ℝ), x' ≠ x ∨ y' ≠ y →
    ¬(parabola1 x' y' ∧ parabola2 x' y') :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_is_correct_l579_57959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_identity_l579_57928

theorem sine_cosine_identity (θ : Real) :
  Real.sin θ + Real.cos (θ + π / 6) = 1 → Real.sin (θ + π / 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_identity_l579_57928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_copies_for_discount_proof_l579_57907

/-- The minimum number of photocopies required to get a discount -/
def min_copies_for_discount : ℕ := 160

/-- The cost of one photocopy in dollars -/
def cost_per_copy : ℚ := 2 / 100

/-- The discount rate offered -/
def discount_rate : ℚ := 25 / 100

/-- The total savings when ordering 160 copies with the discount -/
def total_savings : ℚ := 80 / 100

theorem min_copies_for_discount_proof :
  min_copies_for_discount = 160 ∧
  (cost_per_copy * (1 - discount_rate) * min_copies_for_discount) = 
  (cost_per_copy * min_copies_for_discount - total_savings) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_copies_for_discount_proof_l579_57907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_19_11_l579_57921

theorem binomial_coefficient_19_11 (h1 : Nat.choose 17 10 = 24310) (h2 : Nat.choose 17 8 = 24310) : 
  Nat.choose 19 11 = 85306 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_19_11_l579_57921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l579_57918

theorem equation_solutions :
  ∀ x m n : ℕ,
  x > 0 → m > 0 → n > 0 →
  x^m = 2^(2*n+1) + 2^n + 1 →
  ((x = 2^(2*n+1) + 2^n + 1 ∧ m = 1) ∨ (x = 23 ∧ m = 2 ∧ n = 4)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l579_57918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_M_when_a_is_1_a_value_when_min_M_is_3_l579_57933

-- Define the functions f and g
noncomputable def f (a x : ℝ) : ℝ := x^2 + x + a^2 + a
noncomputable def g (a x : ℝ) : ℝ := x^2 - x + a^2 - a

-- Define M as the maximum of f and g
noncomputable def M (a x : ℝ) : ℝ := max (f a x) (g a x)

-- Theorem 1: When a = 1, the minimum value of M(x) is 7/4
theorem min_M_when_a_is_1 : 
  ∀ x : ℝ, M 1 x ≥ 7/4 ∧ ∃ y : ℝ, M 1 y = 7/4 := by
  sorry

-- Theorem 2: When the minimum value of M(x) is 3, a = (√14 - 1) / 2
theorem a_value_when_min_M_is_3 : 
  (∀ x : ℝ, M a x ≥ 3 ∧ ∃ y : ℝ, M a y = 3) → a = (Real.sqrt 14 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_M_when_a_is_1_a_value_when_min_M_is_3_l579_57933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_range_l579_57990

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (5 - a) * x + 1 else a^x

-- State the theorem
theorem increasing_function_a_range (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : ∀ x y : ℝ, x < y → f a x < f a y) : 
  a ≥ 3 ∧ a < 5 := by
  sorry

#check increasing_function_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_a_range_l579_57990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_decomposition_and_quadrant_theta_values_l579_57931

-- Define the angle α
def α : ℝ := 2010

-- Define the function to convert degrees to radians
noncomputable def deg_to_rad (x : ℝ) : ℝ := x * (Real.pi / 180)

-- Define the function to normalize an angle to [0, 2π)
noncomputable def normalize_angle (x : ℝ) : ℝ := x % (2 * Real.pi)

-- Define the function to determine the quadrant of an angle
noncomputable def quadrant (x : ℝ) : ℕ :=
  let normalized := normalize_angle x
  if 0 ≤ normalized ∧ normalized < Real.pi/2 then 1
  else if Real.pi/2 ≤ normalized ∧ normalized < Real.pi then 2
  else if Real.pi ≤ normalized ∧ normalized < 3*Real.pi/2 then 3
  else 4

-- Theorem for part 1
theorem alpha_decomposition_and_quadrant :
  ∃ (k : ℤ) (β : ℝ), 
    α = k * 360 + β ∧ 
    0 ≤ β ∧ β < 360 ∧
    quadrant (deg_to_rad α) = 3 := by
  sorry

-- Theorem for part 2
theorem theta_values :
  ∃ (θ₁ θ₂ θ₃ : ℝ),
    normalize_angle (deg_to_rad θ₁) = normalize_angle (deg_to_rad α) ∧
    normalize_angle (deg_to_rad θ₂) = normalize_angle (deg_to_rad α) ∧
    normalize_angle (deg_to_rad θ₃) = normalize_angle (deg_to_rad α) ∧
    -360 ≤ θ₁ ∧ θ₁ < 720 ∧
    -360 ≤ θ₂ ∧ θ₂ < 720 ∧
    -360 ≤ θ₃ ∧ θ₃ < 720 ∧
    θ₁ = -150 ∧ θ₂ = 210 ∧ θ₃ = 570 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_decomposition_and_quadrant_theta_values_l579_57931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_sum_l579_57984

theorem quadrilateral_area_sum (a b : ℤ) (ha : a > b) (hb : b > 0) :
  let P : ℤ × ℤ := (a + 1, b - 1)
  let Q : ℤ × ℤ := (b + 1, a - 1)
  let R : ℤ × ℤ := (-a - 1, -b + 1)
  let S : ℤ × ℤ := (-b - 1, -a + 1)
  let area := abs ((a + 1 - (-b - 1)) * (a - 1 - (b - 1)) - (b + 1 - (-a - 1)) * (b - 1 - (a - 1))) / 2
  area = 24 → a + b = 6 := by
  sorry

#check quadrilateral_area_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_sum_l579_57984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toy_bridge_weight_capacity_l579_57911

/-- The weight that a toy bridge must hold up given specific conditions. -/
theorem toy_bridge_weight_capacity 
  (num_soda_cans : ℕ)
  (soda_weight : ℕ)
  (empty_can_weight : ℕ)
  (num_empty_cans : ℕ)
  (empty_can_weight_2 : ℕ) :
  num_soda_cans = 6 →
  soda_weight = 12 →
  empty_can_weight = 2 →
  num_empty_cans = 2 →
  empty_can_weight_2 = 3 →
  (num_soda_cans * (soda_weight + empty_can_weight) +
   num_empty_cans * empty_can_weight_2 +
   2 * (soda_weight + empty_can_weight)) = 118 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toy_bridge_weight_capacity_l579_57911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_equivalence_l579_57974

theorem price_reduction_equivalence (P : ℝ) (h : P > 0) :
  let first_reduction := P * (1 - 0.25)
  let second_reduction := first_reduction * (1 - 0.5)
  second_reduction = P * (1 - 0.375) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_equivalence_l579_57974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_theorem_l579_57904

def num_soldiers : ℕ := 5

def arrangements_without_A_at_left (n : ℕ) : ℕ :=
  (n - 1) * Nat.factorial (n - 1)

theorem arrangements_theorem :
  arrangements_without_A_at_left num_soldiers = 96 := by
  rw [arrangements_without_A_at_left, num_soldiers]
  norm_num
  rfl

#eval arrangements_without_A_at_left num_soldiers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_theorem_l579_57904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l579_57942

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - Real.sqrt (7 - Real.sqrt (x + 1)))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1 ≤ x ∧ x ≤ 48} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l579_57942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_eats_91_slices_l579_57909

/-- The number of pickle slices Sammy can eat -/
def sammy_slices : ℕ := 25

/-- The number of pickle slices Tammy can eat -/
def tammy_slices : ℕ := 3 * sammy_slices

/-- The number of pickle slices Ron can eat -/
noncomputable def ron_slices : ℚ := tammy_slices - (15 / 100 * tammy_slices)

/-- The number of pickle slices Amy can eat -/
noncomputable def amy_slices : ℚ := sammy_slices + (50 / 100 * sammy_slices)

/-- The combined total of pickle slices Ron and Amy can eat -/
noncomputable def ron_amy_total : ℚ := ron_slices + amy_slices

/-- The number of pickle slices Tim can eat -/
noncomputable def tim_slices : ℚ := ron_amy_total - (10 / 100 * ron_amy_total)

/-- Theorem stating that Tim eats approximately 91 pickle slices -/
theorem tim_eats_91_slices : Int.floor tim_slices = 91 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tim_eats_91_slices_l579_57909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_leq_g_f_l579_57943

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 0 else x - 2

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 0 then -x else 0

theorem f_g_leq_g_f (x : ℝ) (h : x ≥ -2) : f (g x) ≤ g (f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_leq_g_f_l579_57943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_valid_cut_is_one_eighth_l579_57923

/-- Represents a cut on the pipe -/
structure Cut where
  x : ℝ
  y : ℝ
  h1 : 0 < x
  h2 : x < y
  h3 : y < 400

/-- Represents a valid cut where all pieces are at least 100 cm -/
def ValidCut (c : Cut) : Prop :=
  c.x ≥ 100 ∧ c.y - c.x ≥ 100 ∧ 400 - c.y ≥ 100

/-- The set of all possible cuts -/
def AllCuts : Set Cut :=
  {c : Cut | True}

/-- The set of valid cuts -/
def ValidCuts : Set Cut :=
  {c : Cut | ValidCut c}

/-- The area of the triangle representing all possible cuts -/
noncomputable def TotalArea : ℝ := (400 * 400) / 2

/-- The area of the triangle representing valid cuts -/
noncomputable def ValidArea : ℝ := (200 * 100) / 2

/-- The probability of a valid cut -/
noncomputable def ProbabilityValidCut : ℝ := ValidArea / TotalArea

theorem probability_valid_cut_is_one_eighth :
  ProbabilityValidCut = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_valid_cut_is_one_eighth_l579_57923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_molecular_weight_l579_57987

/-- The molecular weight of a compound with 2 Potassium atoms, 2 Chromium atoms, and 7 Oxygen atoms -/
def molecularWeight (k39_abundance k39_mass k41_abundance k41_mass
                     cr52_abundance cr52_mass cr53_abundance cr53_mass
                     o_mass : ℝ) : ℝ :=
  let k_avg_mass := k39_abundance * k39_mass + k41_abundance * k41_mass
  let cr_avg_mass := cr52_abundance * cr52_mass + cr53_abundance * cr53_mass
  2 * k_avg_mass + 2 * cr_avg_mass + 7 * o_mass

/-- The molecular weight of the compound is approximately 294.18 amu -/
theorem compound_molecular_weight :
  ∃ ε > 0, |molecularWeight 0.932581 38.9637 0.067302 40.9618
                              0.83789 51.9405 0.09501 52.9407
                              15.999 - 294.18| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_molecular_weight_l579_57987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_and_negativity_l579_57968

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x - Real.log x

theorem zeros_and_negativity (a b : ℝ) :
  (∀ x, x > 0 → (deriv (f a b)) 1 = 0) →
  (
    (b = 2 - a → (
      (0 ≤ a ∧ a < 4 * (1 + Real.log 2) → (∀ x, x > 0 → f a b x ≠ 0)) ∧
      ((a < 0 ∨ a = 4 * (1 + Real.log 2)) → (∃! x, x > 0 ∧ f a b x = 0)) ∧
      (a > 4 * (1 + Real.log 2) → (∃ x y, x > 0 ∧ y > 0 ∧ x ≠ y ∧ f a b x = 0 ∧ f a b y = 0))
    )) ∧
    (a > 0 → Real.log a + 2 * b < 0)
  ) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_and_negativity_l579_57968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_form_theorem_l579_57927

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) / 2

-- Define the property that g must satisfy
def satisfies_equation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, Real.sin x + Real.cos y = f x + f y + g x - g y

-- Theorem statement
theorem g_form_theorem (g : ℝ → ℝ) (h : satisfies_equation g) :
  ∃ C : ℝ, ∀ x : ℝ, g x = (Real.sin x - Real.cos x) / 2 + C :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_form_theorem_l579_57927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_lower_bound_l579_57967

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log x + 2 * Real.exp (x - 1) / x

theorem tangent_line_and_lower_bound (x : ℝ) (hx : x > 0) :
  (∃ (m b : ℝ), m = Real.exp 1 ∧ b = 2 ∧
    ∀ (y : ℝ), y = m * (x - 1) + b ↔ y = f 1 + (deriv f 1) * (x - 1)) ∧
  f x > 1 := by
  sorry

#check tangent_line_and_lower_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_lower_bound_l579_57967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_third_quadrant_l579_57908

/-- The angle in degrees -/
noncomputable def angle : ℝ := 2009

/-- The x-coordinate of the point P -/
noncomputable def x : ℝ := Real.cos (angle * Real.pi / 180)

/-- The y-coordinate of the point P -/
noncomputable def y : ℝ := Real.sin (angle * Real.pi / 180)

/-- Definition of being in the third quadrant -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- Theorem stating that the point (cos2009°, sin2009°) is in the third quadrant -/
theorem point_in_third_quadrant : in_third_quadrant x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_third_quadrant_l579_57908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Ba_approx_l579_57913

/-- Molar mass of Ba(OH)2 in g/mol -/
noncomputable def molar_mass_Ba_OH_2 : ℝ := 171.343

/-- Molar mass of Ba in g/mol -/
noncomputable def molar_mass_Ba : ℝ := 137.327

/-- Mass of Ba(OH)2 in the mixture in grams -/
noncomputable def mass_Ba_OH_2 : ℝ := 25

/-- Mass of NaNO3 in the mixture in grams -/
noncomputable def mass_NaNO3 : ℝ := 15

/-- Calculate the mass percentage of Ba in the mixture -/
noncomputable def mass_percentage_Ba : ℝ :=
  let moles_Ba_OH_2 := mass_Ba_OH_2 / molar_mass_Ba_OH_2
  let mass_Ba := moles_Ba_OH_2 * molar_mass_Ba
  let total_mass := mass_Ba_OH_2 + mass_NaNO3
  (mass_Ba / total_mass) * 100

theorem mass_percentage_Ba_approx :
  abs (mass_percentage_Ba - 50.075) < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Ba_approx_l579_57913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_range_l579_57957

-- Define the circle equation
def circle_eq (a x y : ℝ) : Prop := (x - 2*a)^2 + (y - a - 3)^2 = 4

-- Define the condition that a point is at distance 1 from the origin
def distance_from_origin (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Main theorem
theorem circle_intersection_range (a : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    circle_eq a x₁ y₁ ∧ circle_eq a x₂ y₂ ∧
    distance_from_origin x₁ y₁ ∧ distance_from_origin x₂ y₂) →
  -1 < a ∧ a < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_range_l579_57957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_candy_kinds_l579_57917

/-- Represents the arrangement of candies on a counter. -/
def CandyArrangement := List Nat

/-- Checks if the number of candies between any two candies of the same kind is even. -/
def validArrangement (arr : CandyArrangement) : Prop :=
  ∀ i j, i < j → arr.get? i = arr.get? j → Even (j - i - 1)

/-- The theorem stating the minimum number of kinds of candies. -/
theorem min_candy_kinds (arr : CandyArrangement) 
  (h1 : arr.length = 91)
  (h2 : validArrangement arr) :
  (arr.toFinset).card ≥ 46 := by
  sorry

#check min_candy_kinds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_candy_kinds_l579_57917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_multiple_l579_57963

-- Define the function f(x) = 2^(-x)
noncomputable def f (x : ℝ) : ℝ := 2^(-x)

-- Define the interval [1,3]
def D : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem geometric_mean_multiple :
  ∃ (C : ℝ), C = 1/4 ∧
  (∀ x₁, x₁ ∈ D → ∃! x₂, x₂ ∈ D ∧ Real.sqrt (f x₁ * f x₂) = C) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_multiple_l579_57963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_equals_one_l579_57916

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 2 = 0

-- Define a parallel line to l passing through (-1, 0)
def line_l1 (x y t : ℝ) : Prop :=
  x = -1 + (Real.sqrt 2 / 2) * t ∧ y = (Real.sqrt 2 / 2) * t

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t1 t2 : ℝ,
    line_l1 A.1 A.2 t1 ∧ circle_C A.1 A.2 ∧
    line_l1 B.1 B.2 t2 ∧ circle_C B.1 B.2

-- The main theorem
theorem distance_product_equals_one (A B : ℝ × ℝ) :
  intersection_points A B →
  (A.1 + 1)^2 + A.2^2 * ((B.1 + 1)^2 + B.2^2) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_equals_one_l579_57916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_property_eleven_opposite_twelve_l579_57960

def Circle := Fin 20

def A (k : Fin 20) : ℕ := sorry
def B (k : Fin 20) : ℕ := sorry

def opposite (n : Fin 20) : Fin 20 := sorry

theorem circle_property (k : Fin 20) : A k = B k := sorry

theorem eleven_opposite_twelve : opposite ⟨11, sorry⟩ = ⟨12, sorry⟩ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_property_eleven_opposite_twelve_l579_57960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametricCurveLength_l579_57980

noncomputable def parametricCurve (t : ℝ) : ℝ × ℝ := (3 * Real.sin t, 3 * Real.cos t)

noncomputable def curveLength (f : ℝ → ℝ × ℝ) (a b : ℝ) : ℝ :=
  ∫ t in a..b, Real.sqrt ((deriv (λ t => (f t).1) t) ^ 2 + (deriv (λ t => (f t).2) t) ^ 2)

theorem parametricCurveLength :
  curveLength parametricCurve 0 (2 * Real.pi) = 6 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametricCurveLength_l579_57980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_return_time_is_two_hours_l579_57988

/-- A bicycle trip with given conditions -/
structure BicycleTrip where
  outbound_speed : ℝ
  inbound_speed : ℝ
  route_difference : ℝ
  total_time : ℝ

/-- The return time for a bicycle trip -/
noncomputable def return_time (trip : BicycleTrip) : ℝ :=
  let outbound_distance := (trip.total_time * trip.outbound_speed * trip.inbound_speed - 2 * trip.inbound_speed) / (trip.outbound_speed + trip.inbound_speed)
  (outbound_distance + trip.route_difference) / trip.inbound_speed

/-- Theorem stating that under given conditions, the return time is 2 hours -/
theorem return_time_is_two_hours (trip : BicycleTrip) 
    (h1 : trip.outbound_speed = 3)
    (h2 : trip.inbound_speed = 4)
    (h3 : trip.route_difference = 2)
    (h4 : trip.total_time = 4) :
    return_time trip = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_return_time_is_two_hours_l579_57988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_distance_l579_57972

/-- The distance between the center of the circle with equation x^2 + y^2 = 6x - 4y + 20 and the point (-3, 5) is √85. -/
theorem circle_center_distance : 
  let circle_eq := fun (x y : ℝ) => x^2 + y^2 - 6*x + 4*y - 20 = 0
  let center := (3, -2)
  let point := (-3, 5)
  circle_eq center.1 center.2 → 
  Real.sqrt ((center.1 - point.1)^2 + (center.2 - point.2)^2) = Real.sqrt 85 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_distance_l579_57972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_to_river_and_cabin_l579_57997

-- Define the positions
def cowboy_position : ℝ × ℝ := (0, -6)
def cabin_position : ℝ × ℝ := (10, -15)

-- Define the river as a function y = x
def river (x : ℝ) : ℝ := x

-- Define the reflection of a point across y = x
def reflect (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem shortest_path_to_river_and_cabin :
  let C := cowboy_position
  let B := cabin_position
  let C' := reflect C
  distance C (C.1, river C.1) + distance C' B = 6 + Real.sqrt 457 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_to_river_and_cabin_l579_57997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_x_plus_y_less_than_three_l579_57937

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The probability of a point satisfying a condition in a rectangle --/
noncomputable def probability (r : Rectangle) (condition : ℝ × ℝ → Prop) [DecidablePred condition] : ℝ :=
  (∫ x in r.x_min..r.x_max, ∫ y in r.y_min..r.y_max, Set.indicator (Set.Icc r.x_min r.x_max ×ˢ Set.Icc r.y_min r.y_max) (fun p => if condition p then 1 else 0) (x, y)) /
  ((r.x_max - r.x_min) * (r.y_max - r.y_min))

/-- The main theorem --/
theorem probability_x_plus_y_less_than_three :
  let r : Rectangle := {
    x_min := 0,
    x_max := 4,
    y_min := 0,
    y_max := 3,
    h_x := by norm_num,
    h_y := by norm_num
  }
  probability r (fun (x, y) => x + y < 3) = 3/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_x_plus_y_less_than_three_l579_57937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milo_cash_reward_l579_57945

/-- Represents a subject with its grade and credit hours -/
structure Subject where
  grade : ℚ
  creditHours : ℚ

/-- Calculates the weighted average grade -/
def weightedAverageGrade (subjects : List Subject) : ℚ :=
  let totalWeightedPoints := subjects.foldl (λ acc s => acc + s.grade * s.creditHours) 0
  let totalCreditHours := subjects.foldl (λ acc s => acc + s.creditHours) 0
  totalWeightedPoints / totalCreditHours

/-- Calculates the cash reward based on the weighted average grade -/
def cashReward (subjects : List Subject) : ℚ :=
  5 * weightedAverageGrade subjects

/-- Theorem: Milo's cash reward is $15.33 -/
theorem milo_cash_reward :
  let subjects : List Subject := [
    ⟨2, 5⟩,  -- Mathematics
    ⟨3, 4⟩, ⟨3, 4⟩, ⟨3, 4⟩,  -- English (three times)
    ⟨3, 4⟩, ⟨3, 4⟩,  -- Science (two times)
    ⟨4, 3⟩,  -- History
    ⟨5, 2⟩   -- Art
  ]
  cashReward subjects = 1533 / 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milo_cash_reward_l579_57945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l579_57932

/-- Given a hyperbola and a line satisfying certain conditions, prove that the length of the real axis of the hyperbola is 2 -/
theorem hyperbola_real_axis_length 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hyperbola : ℝ → ℝ → Prop)
  (line : ℝ → ℝ → Prop)
  (hyperbola_eq : ∀ x y : ℝ, hyperbola x y ↔ x^2 / a^2 - y^2 / b^2 = 1)
  (line_eq : ∀ x y : ℝ, line x y ↔ x - Real.sqrt 3 * y + 2 = 0)
  (focus_on_line : ∃ x y : ℝ, hyperbola x y ∧ line x y ∧ (x^2 + y^2 = a^2 + b^2))
  (perpendicular_to_asymptote : ∃ m : ℝ, m * Real.sqrt 3 = -1 ∧ (∀ x : ℝ, hyperbola x (m * x))) :
  2 * a = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l579_57932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_condition_l579_57915

/-- Given vectors m and n, if m + n is perpendicular to m - n, then the x-coordinate of n is -1. -/
theorem perpendicular_vectors_condition (m n : ℝ × ℝ) (h : m = (1, 1)) 
  (h' : ∃ x : ℝ, n = (x, 2)) (h'' : (m + n) • (m - n) = 0) : 
  ∃ x : ℝ, n = (x, 2) ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_condition_l579_57915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_increasing_positive_l579_57914

/-- Given a geometric sequence {a_n} with common ratio q, T_n is the sum of the first n terms -/
noncomputable def geometric_sequence_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁
  else a₁ * (1 - q^n) / (1 - q)

/-- A sequence is monotonically increasing if each term is greater than or equal to the previous term -/
def monotonically_increasing (T : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, T (n + 1) ≥ T n

/-- If a geometric sequence sum is monotonically increasing, then it is positive for n ≥ 1 -/
theorem geometric_sum_increasing_positive
  (a₁ : ℝ) (q : ℝ) (T : ℕ → ℝ)
  (h_geom_sum : ∀ n : ℕ, T n = geometric_sequence_sum a₁ q n)
  (h_increasing : monotonically_increasing T) :
  ∀ n : ℕ, n ≥ 1 → T n > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_increasing_positive_l579_57914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finland_forest_percentage_approx_l579_57995

/-- The percentage of the world's forested area represented by Finland -/
noncomputable def finland_forest_percentage (finland_forest : ℝ) (world_forest : ℝ) : ℝ :=
  (finland_forest / world_forest) * 100

/-- Theorem stating that the percentage of the world's forested area
    represented by Finland is approximately 0.66% -/
theorem finland_forest_percentage_approx :
  let finland_forest := 53.42
  let world_forest := 8076.0
  ∃ ε > 0, |finland_forest_percentage finland_forest world_forest - 0.66| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_finland_forest_percentage_approx_l579_57995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_bag_value_l579_57983

/-- Represents the value of a coin in paise -/
def CoinValue : Type := Nat

/-- Converts paise to rupees -/
def paiseToRupees (paise : Nat) : Rat :=
  paise / 100

/-- Calculates the total value of coins in rupees -/
def totalValue (oneRupeeCount : Nat) (fiftyPaiseCount : Nat) (twentyFivePaiseCount : Nat)
  (oneRupeeValue : Nat) (fiftyPaiseValue : Nat) (twentyFivePaiseValue : Nat) : Rat :=
  (oneRupeeCount : Rat) * paiseToRupees oneRupeeValue +
  (fiftyPaiseCount : Rat) * paiseToRupees fiftyPaiseValue +
  (twentyFivePaiseCount : Rat) * paiseToRupees twentyFivePaiseValue

theorem coin_bag_value :
  totalValue 60 60 60 100 50 25 = 105 := by
  -- Unfold the definition of totalValue
  unfold totalValue
  -- Simplify the arithmetic expressions
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_bag_value_l579_57983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_l579_57982

variable {n : Type*} [Fintype n] [DecidableEq n]

theorem matrix_equation (A : Matrix n n ℝ) (h_inv : IsUnit A) 
  (h_eq : (A - 3 • 1) * (A - 5 • 1) = 0) :
  A + 9 • A⁻¹ = 7 • 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equation_l579_57982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l579_57953

noncomputable def J : ℝ × ℝ := (-3, -4)
noncomputable def K : ℝ × ℝ := (-3, 1)
noncomputable def L : ℝ × ℝ := (5, 7)
noncomputable def M : ℝ × ℝ := (5, -4)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def perimeter : ℝ :=
  distance J K + distance K L + distance L M + distance M J

theorem trapezoid_perimeter : perimeter = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l579_57953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l579_57912

noncomputable def f (x : ℝ) : ℝ := Real.log (2011^x - 1)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > 0} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l579_57912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_probability_l579_57965

def total_members : ℕ := 30
def boys : ℕ := 12
def girls : ℕ := 18
def committee_size : ℕ := 6

theorem committee_probability : 
  (Nat.choose total_members committee_size - 
   (Nat.choose boys committee_size + Nat.choose girls committee_size)) / 
   Nat.choose total_members committee_size = 19145 / 19793 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_probability_l579_57965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_negative_root_l579_57958

theorem greatest_negative_root (x : ℝ) :
  (∀ y < 0, y > -7/6 → (Real.sin (π * y) - Real.cos (2 * π * y)) ≠ 0) ∧
  (Real.sin (π * (-7/6)) - Real.cos (2 * π * (-7/6))) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_negative_root_l579_57958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outer_sphere_area_l579_57962

/-- Represents the area traced by a sphere moving between two concentric spheres -/
noncomputable def tracedArea (innerRadius outerRadius movingSphereRadius : ℝ) (innerArea : ℝ) : ℝ :=
  (outerRadius / innerRadius) * innerArea

/-- Theorem: The area traced on the outer sphere is 55.5 square cm -/
theorem outer_sphere_area 
  (innerRadius : ℝ) 
  (outerRadius : ℝ) 
  (movingSphereRadius : ℝ) 
  (innerArea : ℝ) 
  (h1 : innerRadius = 4)
  (h2 : outerRadius = 6)
  (h3 : movingSphereRadius = 1)
  (h4 : innerArea = 37) :
  tracedArea innerRadius outerRadius movingSphereRadius innerArea = 55.5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval tracedArea 4 6 1 37

end NUMINAMATH_CALUDE_ERRORFEEDBACK_outer_sphere_area_l579_57962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_to_g_l579_57946

open Real

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := sin x + cos x

/-- The transformed function g(x) -/
noncomputable def g (x : ℝ) : ℝ := Real.sqrt 2 * sin (2*x + 11*π/12)

/-- Theorem stating that g is the result of transforming f -/
theorem transform_f_to_g :
  ∀ x : ℝ, g x = f ((x + π/3) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_f_to_g_l579_57946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_height_l579_57991

/-- Represents the height of an object -/
def Height := ℝ

/-- Represents the length of a shadow -/
def ShadowLength := ℝ

/-- Represents a pair of an object's height and its shadow length -/
structure ObjectShadow where
  height : Height
  shadowLength : ShadowLength

/-- Determines if two ObjectShadows are under similar conditions -/
def similarConditions (obj1 obj2 : ObjectShadow) : Prop := sorry

theorem flagpole_height 
  (flagpole building : ObjectShadow)
  (h_similar : similarConditions flagpole building)
  (h_flagpole_shadow : flagpole.shadowLength = (45 : ℝ))
  (h_building_shadow : building.shadowLength = (50 : ℝ))
  (h_building_height : building.height = (20 : ℝ)) :
  flagpole.height = (18 : ℝ) := by
  sorry

#check flagpole_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_height_l579_57991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l579_57900

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse defined by three of its four axis endpoints -/
structure Ellipse where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Calculates the distance between the foci of an ellipse -/
noncomputable def focalDistance (e : Ellipse) : ℝ :=
  2 * Real.sqrt 15

/-- Theorem stating that an ellipse with the given endpoints has foci distance 2√15 -/
theorem ellipse_focal_distance :
  ∀ e : Ellipse,
  ((e.p1 = ⟨1, 5⟩ ∧ e.p2 = ⟨4, -3⟩ ∧ e.p3 = ⟨9, 5⟩) ∨
   (e.p1 = ⟨1, 5⟩ ∧ e.p2 = ⟨9, 5⟩ ∧ e.p3 = ⟨4, -3⟩) ∨
   (e.p1 = ⟨4, -3⟩ ∧ e.p2 = ⟨1, 5⟩ ∧ e.p3 = ⟨9, 5⟩) ∨
   (e.p1 = ⟨4, -3⟩ ∧ e.p2 = ⟨9, 5⟩ ∧ e.p3 = ⟨1, 5⟩) ∨
   (e.p1 = ⟨9, 5⟩ ∧ e.p2 = ⟨1, 5⟩ ∧ e.p3 = ⟨4, -3⟩) ∨
   (e.p1 = ⟨9, 5⟩ ∧ e.p2 = ⟨4, -3⟩ ∧ e.p3 = ⟨1, 5⟩)) →
  focalDistance e = 2 * Real.sqrt 15 :=
by
  sorry

#check ellipse_focal_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l579_57900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_even_property_not_odd_not_even_l579_57934

noncomputable def f (a b x : ℝ) : ℝ := 2 / (a^x - 1) + b

-- State the theorem
theorem odd_even_property (a b : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  (∀ x, f a b x = -f a b (-x)) ↔ b = 1 :=
by sorry

-- Additional theorem to show that f is neither odd nor even when b ≠ 1
theorem not_odd_not_even (a b : ℝ) (ha : a > 0) (ha' : a ≠ 1) (hb : b ≠ 1) :
  ¬(∀ x, f a b x = f a b (-x)) ∧ ¬(∀ x, f a b x = -f a b (-x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_even_property_not_odd_not_even_l579_57934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_in_5_minutes_theorem_l579_57940

/-- Represents the distance traveled by an automobile -/
noncomputable def distance_traveled (a : ℝ) (s : ℝ) : ℝ :=
  -- The automobile travels a/4 feet in s seconds
  a / 4

/-- Converts feet to yards -/
noncomputable def feet_to_yards (feet : ℝ) : ℝ :=
  feet / 3

/-- Calculates the distance in yards traveled in 5 minutes -/
noncomputable def distance_in_5_minutes (a : ℝ) (s : ℝ) : ℝ :=
  let rate_feet_per_second := distance_traveled a s / s
  let rate_yards_per_second := feet_to_yards rate_feet_per_second
  rate_yards_per_second * (5 * 60)

/-- Theorem stating the distance traveled in 5 minutes -/
theorem distance_in_5_minutes_theorem (a s : ℝ) (h1 : s ≠ 0) :
  distance_in_5_minutes a s = 25 * a / s := by
  -- Unfold the definitions and simplify
  unfold distance_in_5_minutes
  unfold feet_to_yards
  unfold distance_traveled
  -- Perform algebraic manipulations
  simp [h1]
  -- The proof is completed
  sorry

#check distance_in_5_minutes_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_in_5_minutes_theorem_l579_57940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_plus_n_l579_57964

/-- Two parallel lines with a given distance between them -/
structure ParallelLines where
  m : ℝ
  n : ℝ
  distance : ℝ
  line1 : ℝ → ℝ → ℝ := λ x y => x + 3 * y + m
  line2 : ℝ → ℝ → ℝ := λ x y => 2 * x + n * y + 9
  parallel : ∃ (k : ℝ), k ≠ 0 ∧ (2 : ℝ) = k * 1 ∧ n = k * 3
  dist_eq : distance = (3 * Real.sqrt 10) / 20

/-- The possible values of m + n for the given parallel lines -/
theorem parallel_lines_m_plus_n (pl : ParallelLines) :
  pl.m + pl.n = 12 ∨ pl.m + pl.n = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_plus_n_l579_57964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_configurations_l579_57948

-- Define the plane
structure Plane where
  point : ℝ × ℝ

-- Define a line in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define angle between lines (placeholder)
def angle_between_lines (a b : Line3D) : ℝ := sorry

-- Define slope of projection (placeholder)
def slope_of_projection (l : Line3D) : ℝ := sorry

-- Define angle with plane (placeholder)
def angle_with_plane (l : Line3D) : ℝ := sorry

-- Define the problem setup
def problem_setup (A B : Plane) (a b : Line3D) : Prop :=
  -- Angle between lines a and b is 30°
  angle_between_lines a b = 30 ∧
  -- Slopes of first projections of lines a and b are 30°
  slope_of_projection a = 30 ∧
  slope_of_projection b = 30 ∧
  -- Angle between planes of lines and first plane is 60°
  angle_with_plane a = 60 ∧
  angle_with_plane b = 60

-- Define the theorem
theorem intersecting_lines_configurations 
  (A B : Plane) : 
  ∃ (configurations : List (Line3D × Line3D)), 
    (∀ (config : Line3D × Line3D), config ∈ configurations → 
      problem_setup A B config.fst config.snd) ∧
    configurations.length = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_lines_configurations_l579_57948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_lg_three_roots_l579_57996

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the theorem
theorem sin_eq_lg_three_roots :
  ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∀ x : ℝ, x > 0 → (Real.sin x = lg x ↔ x = a ∨ x = b ∨ x = c)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_eq_lg_three_roots_l579_57996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l579_57951

theorem complex_equation_solution (z : ℂ) : (Complex.I - z = 2 - Complex.I) → z = -1 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l579_57951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_is_negative_five_l579_57910

/-- A sequence in geometric progression with the first three terms x, 2x + 5, and 3x + 10 -/
def geometric_sequence (x : ℝ) : ℕ → ℝ
| 0 => x
| 1 => 2*x + 5
| 2 => 3*x + 10
| (n+3) => geometric_sequence x 2  -- We define this to avoid the 'sorry'

/-- The common ratio of the geometric sequence -/
noncomputable def common_ratio (x : ℝ) : ℝ := (geometric_sequence x 1) / (geometric_sequence x 0)

theorem fourth_term_is_negative_five :
  ∃ x : ℝ, 
  (∀ n : ℕ, n ≥ 1 → geometric_sequence x (n+1) = (common_ratio x) * geometric_sequence x n) ∧
  geometric_sequence x 3 = -5 := by
  sorry

#check fourth_term_is_negative_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_is_negative_five_l579_57910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_max_distance_l579_57986

/-- The distance between clock hands when increasing most rapidly -/
theorem clock_hands_max_distance (minute_hand : ℝ) (hour_hand : ℝ) 
  (h_minute : minute_hand = 4) (h_hour : hour_hand = 3) : 
  ∃ θ : ℝ, 
    let d := λ (θ : ℝ) => Real.sqrt (minute_hand^2 + hour_hand^2 - 2 * minute_hand * hour_hand * Real.cos θ)
    let d_derivative := λ (θ : ℝ) => (minute_hand * hour_hand * Real.sin θ) / d θ
    (∀ φ : ℝ, |d_derivative θ| ≥ |d_derivative φ|) ∧ d θ = Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_max_distance_l579_57986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_and_triangle_area_l579_57978

-- Define the parabola C
def C (x y : ℝ) : Prop := x^2 = 4*y

-- Define the focus F
def F : ℝ × ℝ := (0, 1)

-- Define the line l1
def l1 (k b x y : ℝ) : Prop := y = k*x + b

-- Define the line l2
def l2 (m x y : ℝ) : Prop := x - m*y + 3*m + 2 = 0

-- Define the fixed point P
def P : ℝ × ℝ := (0, 2)

-- Define the fixed point Q
def Q : ℝ × ℝ := (-2, 3)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  (v1.1 * v2.1) + (v1.2 * v2.2)

theorem parabola_intersection_and_triangle_area 
  (k b : ℝ) 
  (D E : ℝ × ℝ) 
  (hD : C D.1 D.2 ∧ l1 k b D.1 D.2) 
  (hE : C E.1 E.2 ∧ l1 k b E.1 E.2) 
  (h_dot : dot_product D E = -4) :
  (∃ m, l2 m P.1 P.2) ∧ 
  (abs (F.1 - P.1) * abs (F.2 - Q.2)) / 2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_and_triangle_area_l579_57978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_journey_l579_57956

def driving_distances : List Int := [15, -3, 14, -11, 10, -12]

noncomputable def fuel_consumption_rate : ℝ := Real.pi  -- Using pi as a placeholder for 'a'

theorem taxi_journey :
  let final_distance : Int := driving_distances.sum
  let total_distance : Nat := driving_distances.map Int.natAbs |>.sum
  let total_fuel : ℝ := (total_distance : ℝ) * fuel_consumption_rate
  final_distance = 13 ∧ total_distance = 65 ∧ total_fuel = 65 * fuel_consumption_rate := by
  sorry

#eval driving_distances.sum
#eval driving_distances.map Int.natAbs |>.sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_journey_l579_57956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_radius_of_y_is_three_l579_57919

-- Define the circles x and y
variable (x y : ℝ → Prop)

-- Define the radius and area functions
noncomputable def radius (c : ℝ → Prop) : ℝ := sorry
noncomputable def area (c : ℝ → Prop) : ℝ := Real.pi * (radius c)^2

-- State the theorem
theorem half_radius_of_y_is_three
  (h1 : area x = area y)
  (h2 : 2 * Real.pi * radius x = 12 * Real.pi) :
  radius y / 2 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_radius_of_y_is_three_l579_57919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_third_gap_l579_57905

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- The race scenario -/
structure RaceScenario where
  first : Runner
  second : Runner
  third : Runner
  race_distance : ℝ
  first_second_gap : ℝ
  first_third_gap : ℝ

theorem second_third_gap (race : RaceScenario) 
  (h1 : race.race_distance = 10)
  (h2 : race.first_second_gap = 2)
  (h3 : race.first_third_gap = 4)
  (h4 : race.race_distance / race.first.speed = (race.race_distance - race.first_second_gap) / race.second.speed)
  (h5 : race.race_distance / race.first.speed = (race.race_distance - race.first_third_gap) / race.third.speed) :
  race.race_distance - race.third.speed * (race.race_distance / race.second.speed) = 2.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_third_gap_l579_57905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_properties_l579_57926

/-- Set of sixteen-digit natural numbers that are squares and have 1 as tens digit -/
def A : Set ℕ :=
  {n : ℕ | 10^15 ≤ n ∧ n < 10^16 ∧ ∃ m : ℕ, n = m^2 ∧ (n / 10) % 10 = 1}

theorem A_properties :
  (∀ n ∈ A, Even n) ∧ Set.encard A > 10^6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_properties_l579_57926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_minimum_value_l579_57949

/-- The function g(x) as defined in the problem -/
noncomputable def g (x : ℝ) : ℝ :=
  x + x / (x^2 + 2) + x * (x + 3) / (x^2 + 3) + 3 * (x + 1) / (x * (x^2 + 3))

/-- Theorem stating that g(x) ≥ 4 for all x > 0 -/
theorem g_minimum_value (x : ℝ) (hx : x > 0) : g x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_minimum_value_l579_57949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_variance_l579_57981

/-- A random variable following a normal distribution -/
structure NormalRandomVariable where
  μ : ℝ  -- mean
  σ : ℝ  -- standard deviation
  σ_pos : σ > 0

/-- The density function of a normal distribution -/
noncomputable def normalDensity (X : NormalRandomVariable) (x : ℝ) : ℝ :=
  (1 / (X.σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - X.μ)/X.σ)^2)

/-- The variance of a random variable -/
noncomputable def variance (X : NormalRandomVariable) : ℝ := sorry

/-- The theorem stating that the variance of a normal random variable is σ² -/
theorem normal_variance (X : NormalRandomVariable) :
  (∀ x : ℝ, normalDensity X x ≤ 1 / (X.σ * Real.sqrt (2 * Real.pi))) →
  (normalDensity X X.μ = 1 / (X.σ * Real.sqrt (2 * Real.pi))) →
  (variance X = X.σ^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_variance_l579_57981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l579_57935

theorem solve_exponential_equation :
  ∃ x : ℝ, 64 = 4 * (16 : ℝ) ^ (x - 2) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l579_57935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_one_l579_57925

/-- A polynomial with nonnegative integer coefficients less than 100 -/
def PolynomialWithConstraints (P : ℕ → ℤ) : Prop :=
  ∀ n, 0 ≤ P n ∧ P n < 100

theorem polynomial_value_at_one
  (P : ℕ → ℤ)  -- P represents the coefficients of the polynomial
  (h_constraints : PolynomialWithConstraints P)
  (h_eval_10 : (∑' n, P n * 10^n) = 331633)
  (h_eval_neg_10 : (∑' n, P n * (-10)^n) = 273373) :
  (∑' n, P n) = 100 := by
  sorry

#check polynomial_value_at_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_one_l579_57925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_quadrilateral_area_l579_57999

noncomputable section

-- Define the ellipse C₁
def C₁ (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := y^2 = x

-- Define the focal points
def F₁ : ℝ × ℝ := (2, 0)
def F₂ : ℝ × ℝ := (-2, 0)

-- Define the area of quadrilateral AF₁F₂D
noncomputable def S (t : ℝ) : ℝ := 8 * Real.sqrt 2 * Real.sqrt (t^2 + 1) / (t^2 + 2)

theorem ellipse_and_quadrilateral_area 
  (h₁ : ∀ x y, C₁ 2 2 x y ↔ x^2 / 8 + y^2 / 4 = 1)
  (h₂ : ∃ x y, C₁ 2 2 x y ∧ C₂ x y)
  (h₃ : ∃ l : Set (ℝ × ℝ), (∀ p, p ∈ l → (C₁ 2 2 p.1 p.2 ∧ C₂ p.1 p.2)) ∧ F₂ ∈ l)
  (h₄ : ∀ t, t^2 < 8 → ¬∃ x y, x = t * y - 2 ∧ C₂ x y)
  (h₅ : ∀ S', (∃ t, S' = S t) → 12 * Real.sqrt 2 / 5 < S' ∧ S' ≤ 4 * Real.sqrt 2) :
  (∀ x y, C₁ 2 2 x y ↔ x^2 / 8 + y^2 / 4 = 1) ∧
  (∀ S', (∃ t, S' = S t) → 12 * Real.sqrt 2 / 5 < S' ∧ S' ≤ 4 * Real.sqrt 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_quadrilateral_area_l579_57999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planA_more_profitable_l579_57938

/-- Represents a financial plan with initial loan, initial profit, and growth rate -/
structure FinancialPlan where
  initialLoan : ℝ
  initialProfit : ℝ
  growthRate : ℝ

/-- Calculates the total profit over 10 years for a given plan -/
noncomputable def totalProfit (plan : FinancialPlan) : ℝ := 
  if plan.growthRate = 0.5 then
    10 * plan.initialProfit + (10 * 9) / 2 * 0.5 * plan.initialProfit
  else
    plan.initialProfit * (1 - (1 + plan.growthRate)^10) / (- plan.growthRate)

/-- Calculates the total loan repayment over 10 years for a given plan -/
noncomputable def totalLoanRepayment (plan : FinancialPlan) : ℝ :=
  if plan.initialLoan = 10000 then
    1.05 * plan.initialLoan * (1.05^10 - 1) / 0.05
  else
    plan.initialLoan * 1.05^10

/-- Calculates the net profit for a given plan -/
noncomputable def netProfit (plan : FinancialPlan) : ℝ :=
  totalProfit plan - totalLoanRepayment plan

/-- The two plans being compared -/
def planA : FinancialPlan := ⟨100000, 10000, 0.3⟩
def planB : FinancialPlan := ⟨10000, 10000, 0.5⟩

/-- The theorem stating that Plan A yields more profit than Plan B -/
theorem planA_more_profitable : netProfit planA > netProfit planB := by
  sorry

/-- Given values for calculations -/
axiom pow_1_05_10 : (1.05 : ℝ)^10 = 1.629
axiom pow_1_3_10 : (1.3 : ℝ)^10 = 13.786
axiom pow_1_5_10 : (1.5 : ℝ)^10 = 57.665

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planA_more_profitable_l579_57938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_fold_theorem_l579_57920

noncomputable def equilateral_triangle_fold (side_length : ℝ) (fold_point_distance : ℝ) : ℕ × ℕ × ℕ :=
  let m : ℕ := 3
  let n : ℕ := 1
  let p : ℕ := 13
  (m, n, p)

theorem equilateral_triangle_fold_theorem :
  let (m, n, p) := equilateral_triangle_fold 12 9
  (m : ℝ) * Real.sqrt p / n = 3 * Real.sqrt 13 ∧
  Nat.gcd m n = 1 ∧
  ∀ (q : ℕ), Nat.Prime q → ¬(q ^ 2 ∣ p) ∧
  m + n + p = 17 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_fold_theorem_l579_57920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_not_externally_tangent_l579_57906

-- Define the circles
noncomputable def circle_C (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 3*p.1 + 5*p.2 = r^2 - 17/2}

def circle_D : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 9}

-- Define the center and radius of circle C
noncomputable def center_C : ℝ × ℝ := (3/2, -5/2)

def radius_C (r : ℝ) : ℝ := r

-- Define the center and radius of circle D
def center_D : ℝ × ℝ := (0, 0)

def radius_D : ℝ := 3

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ :=
  Real.sqrt ((center_C.1 - center_D.1)^2 + (center_C.2 - center_D.2)^2)

-- Theorem: Circles C and D cannot be externally tangent
theorem circles_not_externally_tangent (r : ℝ) (hr : r > 0) :
  ∀ (x y : ℝ), (x, y) ∈ circle_C r → (x, y) ∈ circle_D →
  distance_between_centers ≠ radius_C r + radius_D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_not_externally_tangent_l579_57906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neg_alpha_eq_three_fifths_l579_57977

theorem cos_neg_alpha_eq_three_fifths (α : ℝ) 
  (h : Real.sin (π / 2 + α) = -3 / 5) : 
  Real.cos (-α) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neg_alpha_eq_three_fifths_l579_57977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_and_calculation_l579_57929

/-- Given a point P(-4, 3) on the terminal side of angle θ, prove the trigonometric identities and calculation -/
theorem trig_identities_and_calculation (θ : ℝ) 
  (h : ∃ (r : ℝ), r > 0 ∧ r * Real.cos θ = -4 ∧ r * Real.sin θ = 3) : 
  Real.sin θ = 3/5 ∧ 
  Real.cos θ = -4/5 ∧ 
  Real.tan θ = -3/4 ∧ 
  (Real.cos (θ - π/2)) / (Real.sin (π/2 + θ)) * Real.sin (θ + π) * Real.cos (2*π - θ) = -9/25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_and_calculation_l579_57929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_theorem_l579_57993

/-- Represents the selling price of the sportswear in yuan -/
def x : Type := ℝ

/-- Represents the monthly sales volume of the sportswear -/
def y (x : ℝ) : ℝ := -2 * x + 400

/-- Represents the cost price of the sportswear in yuan -/
def cost_price : ℝ := 60

/-- Calculates the profit per piece of sportswear -/
def profit_per_piece (x : ℝ) : ℝ := x - cost_price

/-- Calculates the monthly profit from selling the sportswear -/
def monthly_profit (x : ℝ) : ℝ := profit_per_piece x * y x

/-- The theorem stating the maximum monthly profit and the corresponding selling price -/
theorem max_profit_theorem :
  ∃ (max_profit : ℝ) (max_price : ℝ),
    max_profit = 9800 ∧
    max_price = 130 ∧
    ∀ x, monthly_profit x ≤ max_profit ∧
    monthly_profit max_price = max_profit := by
  sorry

#check max_profit_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_theorem_l579_57993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_difference_l579_57922

/-- The hyperbola -/
def hyperbola (x y : ℝ) : Prop := x^2/9 - y^2/16 = 1

/-- The first circle -/
def circle1 (x y : ℝ) : Prop := (x+5)^2 + y^2 = 4

/-- The second circle -/
def circle2 (x y : ℝ) : Prop := (x-5)^2 + y^2 = 1

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem max_distance_difference :
  ∃ (C : ℝ),
    C = 9 ∧
    ∀ (px py mx my nx ny : ℝ),
      hyperbola px py →
      circle1 mx my →
      circle2 nx ny →
      distance px py mx my - distance px py nx ny ≤ C :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_difference_l579_57922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martian_calendar_months_l579_57941

/-- Represents the number of days in a Martian month -/
inductive MartianMonth
  | long  : MartianMonth  -- 100 days
  | short : MartianMonth  -- 77 days

/-- Calculates the number of days in a given MartianMonth -/
def daysInMonth (m : MartianMonth) : Nat :=
  match m with
  | MartianMonth.long  => 100
  | MartianMonth.short => 77

/-- Represents a Martian calendar year -/
structure MartianYear where
  months : List MartianMonth
  total_days_eq : List.sum (List.map daysInMonth months) = 5882

/-- Theorem: A Martian year has 74 months -/
theorem martian_calendar_months (year : MartianYear) : year.months.length = 74 := by
  sorry

#check martian_calendar_months

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martian_calendar_months_l579_57941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l579_57976

-- Define the vector type
def PlanarVector := ℝ × ℝ

-- Define the dot product
def dot_product (v w : PlanarVector) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude of a vector
noncomputable def magnitude (v : PlanarVector) : ℝ := Real.sqrt (dot_product v v)

-- Define the theorem
theorem vector_sum_magnitude 
  (a b c : PlanarVector)
  (h1 : magnitude a = 1)
  (h2 : magnitude b = 2)
  (h3 : magnitude c = 3)
  (h4 : ∃ θ : ℝ, dot_product a b = magnitude a * magnitude b * Real.cos θ ∧
                 dot_product a c = magnitude a * magnitude c * Real.cos θ ∧
                 dot_product b c = magnitude b * magnitude c * Real.cos θ) :
  magnitude (a.1 + b.1 + c.1, a.2 + b.2 + c.2) = Real.sqrt 3 ∨
  magnitude (a.1 + b.1 + c.1, a.2 + b.2 + c.2) = 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_l579_57976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_form_n_l579_57979

/-- The quadratic equation -/
def quadratic_equation (x : ℝ) : Prop := 3 * x^2 - 4 * x - 7 = 0

/-- The root form -/
def root_form (x m n p : ℝ) : Prop := x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p

/-- m, n, p are positive integers -/
def positive_integers (m n p : ℝ) : Prop := ∃ (m' n' p' : ℕ+), (m = m') ∧ (n = n') ∧ (p = p')

/-- The greatest common divisor of m, n, and p is 1 -/
def gcd_is_one (m n p : ℕ+) : Prop := Nat.gcd m.val (Nat.gcd n.val p.val) = 1

theorem quadratic_root_form_n (x m n p : ℝ) : 
  quadratic_equation x → 
  root_form x m n p → 
  positive_integers m n p → 
  (∃ (m' n' p' : ℕ+), (m = m') ∧ (n = n') ∧ (p = p') ∧ gcd_is_one m' n' p') →
  n = 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_form_n_l579_57979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_four_quadrants_l579_57902

noncomputable section

/-- The function f(x) that we're analyzing -/
def f (a x : ℝ) : ℝ := (1/3) * a * x^3 + a * x^2 - 3 * a * x + 1

/-- The condition for f(x) to pass through four quadrants -/
def passes_through_four_quadrants (a : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ x₄ : ℝ, 
    x₁ < 0 ∧ x₂ > 0 ∧ x₃ < 0 ∧ x₄ > 0 ∧
    f a x₁ > 0 ∧ f a x₂ < 0 ∧ f a x₃ < 0 ∧ f a x₄ > 0

/-- The theorem stating the range of a for which f(x) passes through four quadrants -/
theorem range_of_a_for_four_quadrants :
  ∀ a : ℝ, passes_through_four_quadrants a ↔ (a < -1/9 ∨ a > 3/5) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_four_quadrants_l579_57902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_k_min_dot_product_l579_57975

/-- Given planar vectors -/
def OA : Fin 2 → ℝ := ![1, 7]
def OB : Fin 2 → ℝ := ![5, 1]
def OP : Fin 2 → ℝ := ![2, 1]

/-- Theorem for part I -/
theorem parallel_vectors_k (k : ℝ) :
  (∃ l : ℝ, k • OA + 2 • OB = l • (2 • OA - OB)) → k = -4 := by sorry

/-- Theorem for part II -/
theorem min_dot_product :
  (∃ t : ℝ, ∀ Q : Fin 2 → ℝ, (∃ s : ℝ, Q = s • OP) →
    (OA - Q) • (OB - Q) ≥ -8) ∧
  (∃ Q : Fin 2 → ℝ, (∃ s : ℝ, Q = s • OP) ∧ (OA - Q) • (OB - Q) = -8) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_k_min_dot_product_l579_57975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sharik_cannot_eat_all_meatballs_l579_57955

/-- Represents a strategy for Sharik to eat meatballs and move flies. -/
def Strategy := ℕ → ℕ × ℕ

/-- The state of the meatballs and flies at any given time. -/
structure GameState where
  meatballs : ℕ → ℕ  -- Number of flies on each meatball
  eaten : ℕ          -- Number of meatballs eaten
  total_flies : ℕ    -- Total number of flies eaten

/-- Apply a strategy for a given number of turns. -/
def apply_strategy (s : Strategy) (initial : GameState) (turns : ℕ) : GameState :=
  sorry

/-- A strategy is valid if it never moves a fly from a non-existent meatball. -/
def valid_strategy (s : Strategy) : Prop :=
  sorry

theorem sharik_cannot_eat_all_meatballs :
  ∀ (s : Strategy), valid_strategy s →
    ∃ (n : ℕ), ∀ (turns : ℕ),
      (let final_state := apply_strategy s 
        { meatballs := λ _ => 1, eaten := 0, total_flies := 0 } turns;
       final_state.total_flies ≤ 1000000 → final_state.meatballs n > 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sharik_cannot_eat_all_meatballs_l579_57955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_convex_prime_sides_l579_57961

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- The four vertices of the quadrilateral -/
def A : Point := ⟨0, 0⟩
def B : Point := ⟨5, 0⟩
def C : Point := ⟨5, 17⟩
def D : Point := ⟨0, 29⟩

/-- The four side lengths of the quadrilateral -/
noncomputable def AB : ℝ := distance A B
noncomputable def BC : ℝ := distance B C
noncomputable def CD : ℝ := distance C D
noncomputable def DA : ℝ := distance D A

/-- Theorem: The quadrilateral ABCD is convex with prime side lengths -/
theorem quadrilateral_convex_prime_sides :
  (isPrime 5 ∧ isPrime 17 ∧ isPrime 13 ∧ isPrime 29) ∧
  (AB ≠ BC ∧ AB ≠ CD ∧ AB ≠ DA ∧ BC ≠ CD ∧ BC ≠ DA ∧ CD ≠ DA) ∧
  (∀ (p1 p2 p3 p4 : Point), 
    (p1 = A ∧ p2 = B ∧ p3 = C ∧ p4 = D) →
    (p2.x - p1.x) * (p3.y - p2.y) - (p3.x - p2.x) * (p2.y - p1.y) > 0 ∧
    (p3.x - p2.x) * (p4.y - p3.y) - (p4.x - p3.x) * (p3.y - p2.y) > 0 ∧
    (p4.x - p3.x) * (p1.y - p4.y) - (p1.x - p4.x) * (p4.y - p3.y) > 0 ∧
    (p1.x - p4.x) * (p2.y - p1.y) - (p2.x - p1.x) * (p1.y - p4.y) > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_convex_prime_sides_l579_57961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_palabras_bookstore_readers_l579_57947

theorem palabras_bookstore_readers (W : ℕ) (saramago_readers kureishi_readers : ℚ) : 
  W = 40 →
  saramago_readers = 1/4 →
  kureishi_readers = 5/8 →
  (W : ℚ) - (saramago_readers * W + kureishi_readers * W - (saramago_readers * kureishi_readers * W)) = 
    (saramago_readers * W - saramago_readers * kureishi_readers * W) - 1 →
  saramago_readers * kureishi_readers * W = 2 := by
  sorry

#check palabras_bookstore_readers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_palabras_bookstore_readers_l579_57947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_is_four_l579_57966

-- Define the sector
noncomputable def sector_circumference : ℝ := 8
noncomputable def sector_central_angle : ℝ := 2

-- Define the sector area function
noncomputable def sector_area (r : ℝ) (θ : ℝ) : ℝ := (1/2) * r^2 * θ

-- Theorem statement
theorem sector_area_is_four :
  ∃ (r : ℝ), r > 0 ∧ 2 * r + 2 * Real.pi * r = sector_circumference ∧
  sector_area r sector_central_angle = 4 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

#check sector_area_is_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_is_four_l579_57966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_first_part_speed_l579_57936

/-- A journey with two parts --/
structure Journey where
  total_distance : ℝ
  total_time : ℝ
  first_part_time : ℝ
  second_part_speed : ℝ

/-- The speed during the first part of the journey --/
noncomputable def first_part_speed (j : Journey) : ℝ :=
  (j.total_distance - j.second_part_speed * (j.total_time - j.first_part_time)) / j.first_part_time

/-- Theorem stating the speed during the first part of the journey --/
theorem journey_first_part_speed (j : Journey) 
  (h1 : j.total_distance = 240)
  (h2 : j.total_time = 5)
  (h3 : j.first_part_time = 3)
  (h4 : j.second_part_speed = 60) :
  first_part_speed j = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_first_part_speed_l579_57936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_max_a_for_positive_f_l579_57969

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + 1

-- Theorem for part I
theorem tangent_parallel_to_x_axis (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, (deriv (f a)) x = 0 ↔ x = 1) →
  a = Real.exp 1 :=
by sorry

-- Theorem for part II
theorem max_a_for_positive_f (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, x < 1 → f a x > 0) →
  a ≤ Real.exp 1 + 1 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_to_x_axis_max_a_for_positive_f_l579_57969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_odd_exists_l579_57901

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem f_increasing_and_odd_exists :
  (∀ a : ℝ, ∀ x y : ℝ, x < y → f a x < f a y) ∧
  (∃ a : ℝ, ∀ x : ℝ, f a (-x) = -(f a x)) ∧
  (∀ x : ℝ, f 1 (-x) = -(f 1 x)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_odd_exists_l579_57901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l579_57994

noncomputable def f (x : ℝ) : ℝ := (x^5 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 - 9)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -3 ∨ (-3 < x ∧ x < 3) ∨ 3 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l579_57994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_product_l579_57950

noncomputable section

/-- An ellipse with equation x^2/2 + y^2 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1}

/-- The right focus of the ellipse -/
def F : ℝ × ℝ := (1, 0)

/-- The fixed point M on the x-axis -/
def M : ℝ × ℝ := (5/4, 0)

/-- A line passing through F -/
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 - 1)}

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Theorem: For any line passing through F and intersecting the ellipse at A and B,
    the dot product of MA and MB is constant -/
theorem ellipse_constant_product :
  ∀ k : ℝ, ∀ A B : ℝ × ℝ,
    A ∈ Ellipse ∩ Line k → B ∈ Ellipse ∩ Line k → A ≠ B →
    dot_product (A.1 - M.1, A.2 - M.2) (B.1 - M.1, B.2 - M.2) = -7/16 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_product_l579_57950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_office_persons_count_l579_57989

/-- The number of persons in the office -/
def N : ℕ := sorry

/-- The average age of all persons in the office -/
def average_age : ℕ := 15

/-- The number of persons in the first group -/
def group1_count : ℕ := 5

/-- The average age of the first group -/
def group1_average : ℕ := 14

/-- The number of persons in the second group -/
def group2_count : ℕ := 9

/-- The average age of the second group -/
def group2_average : ℕ := 16

/-- The age of the 15th person -/
def person15_age : ℕ := 86

/-- Theorem stating that the number of persons in the office is 20 -/
theorem office_persons_count : N = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_office_persons_count_l579_57989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l579_57985

/-- Calculates the length of a train given its speed and time to cross a post -/
noncomputable def train_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * (1000 / 3600) * time

/-- Theorem stating the length of a train given specific conditions -/
theorem train_length_calculation (speed : ℝ) (time : ℝ) 
  (h1 : speed = 40) 
  (h2 : time = 19.8) :
  ∃ (ε : ℝ), ε > 0 ∧ |train_length speed time - 220.178| < ε :=
by
  sorry

/-- Approximate evaluation of train length -/
def approx_train_length : ℚ := 
  (40 : ℚ) * (1000 / 3600) * (198 / 10)

#eval approx_train_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l579_57985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_number_in_pascal_row_15_l579_57971

def pascal_row (n : ℕ) : List ℕ :=
  List.range (n + 1) |>.map (λ k => Nat.choose n k)

theorem fifth_number_in_pascal_row_15 :
  (pascal_row 15).get! 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_number_in_pascal_row_15_l579_57971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l579_57952

-- Define T as a noncomputable function of r
noncomputable def T (r : ℝ) := 24 / (1 - r)

-- State the theorem
theorem geometric_series_sum (b : ℝ) 
  (h1 : -1 < b) (h2 : b < 1) 
  (h3 : T b * T (-b) = 4032) : 
  T b + T (-b) = 336 := by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l579_57952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_events_mutually_exclusive_and_conditional_prob_l579_57930

-- Define the sample space for Bag A
def bag_a : Finset (Fin 8) := Finset.univ

-- Define the events A₁, A₂, A₃
def A₁ : Finset (Fin 8) := Finset.filter (λ x => x.val < 3) Finset.univ
def A₂ : Finset (Fin 8) := Finset.filter (λ x => 3 ≤ x.val ∧ x.val < 6) Finset.univ
def A₃ : Finset (Fin 8) := Finset.filter (λ x => 6 ≤ x.val) Finset.univ

-- Define the sample space for Bag B after adding a ball from Bag A
def bag_b : Finset (Fin 6) := Finset.univ

-- Define the event B (drawing a red ball from Bag B)
def B : Finset (Fin 6) := Finset.filter (λ x => 2 ≤ x.val ∧ x.val < 4) Finset.univ

-- Define the probability measure
noncomputable def P (A : Finset (Fin 8)) : ℚ := (A.card : ℚ) / 8

-- Define the conditional probability
noncomputable def P_cond (B : Finset (Fin 6)) (A : Finset (Fin 8)) : ℚ := 
  (B.card : ℚ) / 6

theorem events_mutually_exclusive_and_conditional_prob :
  (A₁ ∩ A₂).card = 0 ∧ 
  (A₁ ∩ A₃).card = 0 ∧ 
  (A₂ ∩ A₃).card = 0 ∧
  P_cond B A₁ = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_events_mutually_exclusive_and_conditional_prob_l579_57930
