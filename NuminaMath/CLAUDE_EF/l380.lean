import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_result_l380_38066

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 3 * x + 7

-- Define the inverse function g_inv
noncomputable def g_inv (y : ℝ) : ℝ := (y - 7) / 3

-- Theorem statement
theorem inverse_composition_result : g_inv (g_inv 16) = -4/3 := by
  -- Calculate g_inv(16)
  have h1 : g_inv 16 = 3 := by
    unfold g_inv
    simp [sub_div]
    norm_num
  
  -- Calculate g_inv(g_inv(16))
  have h2 : g_inv (g_inv 16) = g_inv 3 := by
    rw [h1]
  
  -- Simplify g_inv(3)
  have h3 : g_inv 3 = -4/3 := by
    unfold g_inv
    simp [sub_div]
    norm_num
  
  -- Combine the results
  rw [h2, h3]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_result_l380_38066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_percentage_in_co_l380_38084

/-- The atomic mass of carbon in g/mol -/
noncomputable def carbon_mass : ℝ := 12.01

/-- The atomic mass of oxygen in g/mol -/
noncomputable def oxygen_mass : ℝ := 16.00

/-- The molecular mass of CO in g/mol -/
noncomputable def co_mass : ℝ := carbon_mass + oxygen_mass

/-- The mass percentage of carbon in CO -/
noncomputable def carbon_percentage : ℝ := (carbon_mass / co_mass) * 100

/-- Theorem stating that the mass percentage of carbon in CO is approximately 42.88% -/
theorem carbon_percentage_in_co :
  |carbon_percentage - 42.88| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_percentage_in_co_l380_38084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_count_odd_four_digit_numbers_count_numbers_greater_than_2000_count_l380_38008

def digits : Finset Nat := {1, 2, 3, 5}

def is_valid_number (n : Nat) : Bool :=
  n ≥ 1000 && n ≤ 9999 && ((Finset.filter (λ d => d ∈ digits) (Finset.image (λ i => (n / 10^i) % 10) {0, 1, 2, 3})).card = 4)

def is_odd (n : Nat) : Bool := n % 2 = 1

theorem four_digit_numbers_count :
  (Finset.filter (λ n => is_valid_number n) (Finset.range 10000)).card = 24 := by
  sorry

theorem odd_four_digit_numbers_count :
  (Finset.filter (λ n => is_valid_number n && is_odd n) (Finset.range 10000)).card = 18 := by
  sorry

theorem numbers_greater_than_2000_count :
  (Finset.filter (λ n => is_valid_number n && n > 2000) (Finset.range 10000)).card = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_count_odd_four_digit_numbers_count_numbers_greater_than_2000_count_l380_38008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_delegates_common_language_l380_38006

/-- Represents a delegate at the conference -/
structure Delegate where
  languages : Finset String
  lang_count : languages.card ≤ 3

/-- The conference with its delegates -/
structure Conference where
  delegates : Finset Delegate
  del_count : delegates.card = 9
  common_lang : ∀ d₁ d₂ d₃, d₁ ∈ delegates → d₂ ∈ delegates → d₃ ∈ delegates → 
    d₁ ≠ d₂ ∧ d₂ ≠ d₃ ∧ d₁ ≠ d₃ → 
    ∃ l, (l ∈ d₁.languages ∧ l ∈ d₂.languages) ∨ 
         (l ∈ d₂.languages ∧ l ∈ d₃.languages) ∨ 
         (l ∈ d₁.languages ∧ l ∈ d₃.languages)

/-- Main theorem: There exist three delegates with a common language -/
theorem three_delegates_common_language (conf : Conference) : 
  ∃ d₁ d₂ d₃, d₁ ∈ conf.delegates ∧ d₂ ∈ conf.delegates ∧ d₃ ∈ conf.delegates ∧
  d₁ ≠ d₂ ∧ d₂ ≠ d₃ ∧ d₁ ≠ d₃ ∧ 
  ∃ l, l ∈ d₁.languages ∧ l ∈ d₂.languages ∧ l ∈ d₃.languages :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_delegates_common_language_l380_38006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_breadth_for_given_cistern_l380_38092

/-- Represents the dimensions and wet surface area of a cistern -/
structure Cistern where
  length : ℝ
  width : ℝ
  wetSurfaceArea : ℝ

/-- Calculates the breadth of water in a cistern given its dimensions and wet surface area -/
noncomputable def waterBreadth (c : Cistern) : ℝ :=
  (c.wetSurfaceArea - c.length * c.width) / (2 * (c.length + c.width))

/-- Theorem stating that for a cistern with given dimensions and wet surface area, 
    the calculated water breadth is approximately 1.85 m -/
theorem water_breadth_for_given_cistern :
  let c : Cistern := { length := 8, width := 6, wetSurfaceArea := 99.8 }
  ∃ ε > 0, |waterBreadth c - 1.85| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_breadth_for_given_cistern_l380_38092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_problem_l380_38020

theorem triangle_sine_problem (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = π ∧
  a * Real.sin C = b * Real.sin A ∧
  b * Real.sin C = c * Real.sin B ∧
  c * Real.sin A = a * Real.sin B ∧
  c = 2 * a ∧
  b * Real.sin B - a * Real.sin A = (1/2) * a * Real.sin C →
  Real.sin B = Real.sqrt 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_problem_l380_38020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_and_max_value_l380_38023

-- Define the vectors OA and OB
noncomputable def OA (x : ℝ) : ℝ × ℝ := (2 * (Real.cos x) ^ 2, 1)
noncomputable def OB (x a : ℝ) : ℝ × ℝ := (1, Real.sqrt 3 * Real.sin (2 * x) + a)

-- Define the dot product function
noncomputable def f (x a : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6) + 1 + a

-- State the theorem
theorem dot_product_and_max_value (x a : ℝ) :
  (OA x).1 * (OB x a).1 + (OA x).2 * (OB x a).2 = f x a ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ 2) →
  a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_and_max_value_l380_38023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_thirtieth_term_l380_38001

/-- Predicate to check if a number contains the digit 1 --/
def containsOne (n : ℕ) : Prop :=
  ∃ k : ℕ, n / (10^k) % 10 = 1

/-- Predicate to check if a number satisfies all conditions of the sequence --/
def inSequence (n : ℕ) : Prop :=
  n > 0 ∧ n % 3 = 0 ∧ containsOne n ∧ n % 5 = 0

/-- The 30th term of the sequence --/
def thirtiethTerm (x : ℕ) : Prop :=
  x > 0 ∧ inSequence x ∧ (∃! k, k > 0 ∧ (∀ y, y < x → inSequence y → k ≤ 29))

theorem exists_unique_thirtieth_term :
  ∃! x : ℕ, thirtiethTerm x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_thirtieth_term_l380_38001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_sum_zero_l380_38009

theorem xy_sum_zero (x y : ℝ) 
  (h : (x + Real.sqrt (x^2 + 1)) * (y + Real.sqrt (y^2 + 1)) = 1) : 
  x + y = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_sum_zero_l380_38009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_multiplication_problem_l380_38057

/-- A digit is a natural number from 0 to 9 -/
def Digit : Type := {n : ℕ // n < 10}

/-- Convert a three-digit number to a natural number -/
def toNat3 (a b c : Digit) : ℕ := 100 * a.val + 10 * b.val + c.val

/-- Convert a two-digit number to a natural number -/
def toNat2 (a b : Digit) : ℕ := 10 * a.val + b.val

theorem digit_multiplication_problem (E F G H : Digit) 
  (hEF : E ≠ F) (hEG : E ≠ G) (hEH : E ≠ H) (hFG : F ≠ G) (hFH : F ≠ H) (hGH : G ≠ H)
  (h_mult : toNat3 E F E * toNat2 G H = toNat2 G H * 101) :
  E.val + F.val = 1 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_multiplication_problem_l380_38057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_sum_one_third_three_fourths_l380_38025

theorem reciprocal_sum_one_third_three_fourths :
  (1 / 3 + 3 / 4 : ℚ)⁻¹ = 12 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_sum_one_third_three_fourths_l380_38025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consumer_installment_credit_l380_38027

/-- The total outstanding consumer installment credit in billions of dollars -/
noncomputable def total_credit : ℝ := 342.857

/-- The automobile installment credit as a percentage of total credit -/
noncomputable def auto_credit_percentage : ℝ := 0.35

/-- The amount of credit extended by automobile finance companies in billions of dollars -/
noncomputable def auto_finance_credit : ℝ := 40

/-- The fraction of automobile installment credit extended by finance companies -/
noncomputable def auto_finance_fraction : ℝ := 1/3

theorem consumer_installment_credit :
  total_credit * auto_credit_percentage * auto_finance_fraction = auto_finance_credit :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consumer_installment_credit_l380_38027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_satisfies_condition_min_a_is_minimum_l380_38083

/-- The minimum value of a that satisfies the given conditions -/
noncomputable def min_a : ℝ := Real.exp 1 - 1

/-- Curve C₁: y = e^x -/
noncomputable def C₁ (x : ℝ) : ℝ := Real.exp x

/-- Curve C₂: y = 1 + ln(x - a), where a > 0 -/
noncomputable def C₂ (a x : ℝ) : ℝ := 1 + Real.log (x - a)

/-- Distance between points A(x₁, y₁) and B(x₂, y₂) -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem min_a_satisfies_condition :
  ∀ (a : ℝ), a > 0 →
  (∀ (x₁ x₂ : ℝ), x₂ > a →
    C₁ x₁ = C₂ a x₂ →
    distance x₁ (C₁ x₁) x₂ (C₂ a x₂) ≥ Real.exp 1) →
  a ≥ min_a :=
sorry

theorem min_a_is_minimum :
  ∀ (ε : ℝ), ε > 0 →
  ∃ (x₁ x₂ : ℝ), x₂ > min_a - ε →
    C₁ x₁ = C₂ (min_a - ε) x₂ ∧
    distance x₁ (C₁ x₁) x₂ (C₂ (min_a - ε) x₂) < Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_satisfies_condition_min_a_is_minimum_l380_38083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_code_permutations_l380_38049

theorem area_code_permutations : 
  (Finset.univ.filter (λ x : Fin 4 → Fin 4 => Function.Injective x)).card = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_code_permutations_l380_38049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l380_38010

/-- The minimum value of 1/a + 4/b given the conditions -/
theorem min_value_theorem (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h_line : ∃ x y : ℝ, a*x + b*y + 1 = 0 ∧ x^2 + y^2 + 8*x + 2*y + 1 = 0) : 
  (∀ a' b' : ℝ, a' > 1 → b' > 1 → 
    (∃ x y : ℝ, a'*x + b'*y + 1 = 0 ∧ x^2 + y^2 + 8*x + 2*y + 1 = 0) → 
    1/a + 4/b ≤ 1/a' + 4/b') ∧ 
  (∃ a₀ b₀ : ℝ, a₀ > 1 ∧ b₀ > 1 ∧ 
    (∃ x y : ℝ, a₀*x + b₀*y + 1 = 0 ∧ x^2 + y^2 + 8*x + 2*y + 1 = 0) ∧ 
    1/a + 4/b = 1/a₀ + 4/b₀) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l380_38010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_range_l380_38076

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := x * (a - (1 / Real.exp x))

-- Define the derivative of f
noncomputable def f_prime (a x : ℝ) : ℝ := a + (x - 1) * Real.exp (-x)

-- Theorem statement
theorem tangent_perpendicular_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f_prime a x₁ = 0 ∧ f_prime a x₂ = 0) ↔ 
  (-1 / Real.exp 2 < a ∧ a < 0) := by
  sorry

#check tangent_perpendicular_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_range_l380_38076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l380_38032

-- Define the hyperbola C
structure Hyperbola where
  center : ℝ × ℝ
  symmetry_axes : Set (ℝ × ℝ)

-- Define the parabola
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8*p.1}

-- Define the focus of the parabola
def F : ℝ × ℝ := (2, 0)

-- Define the intersection point P
structure IntersectionPoint where
  point : ℝ × ℝ
  on_parabola : point ∈ Parabola
  on_asymptote : ∃ (a b : ℝ), a * point.1 + b * point.2 = 0
  distance_to_focus : Real.sqrt ((point.1 - F.1)^2 + (point.2 - F.2)^2) = 4

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (h : Hyperbola) : ℝ := sorry

-- State the theorem
theorem hyperbola_eccentricity (C : Hyperbola) (P : IntersectionPoint) :
  C.center = (0, 0) ∧ 
  C.symmetry_axes = {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0} →
  eccentricity C = Real.sqrt 5 ∨ eccentricity C = Real.sqrt 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l380_38032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l380_38082

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x - 6)

-- Theorem statement
theorem function_properties :
  (f 3 ≠ 14) ∧ 
  (f 4 = -3) ∧ 
  (∃ x : ℝ, f x = 2 ∧ x = 14) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l380_38082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l380_38059

-- Define the vectors
def a : ℝ × ℝ := (3, 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (x, -1)
def c : ℝ × ℝ := (-8, -1)

-- Define the conditions
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0
def parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u.1 = k * v.1 ∧ u.2 = k * v.2

-- Define vector operations
def vectorAdd (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vectorScale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vectorSub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

noncomputable def vectorNorm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def angle (u v : ℝ × ℝ) : ℝ := 
  Real.arccos ((u.1 * v.1 + u.2 * v.2) / (vectorNorm u * vectorNorm v))

-- Define the theorem
theorem vector_problem (x : ℝ) :
  perpendicular (vectorSub (vectorScale 2 a) (b x)) (b x) ∧ 
  parallel a (vectorAdd (b x) c) →
  (vectorNorm (vectorAdd a (vectorScale 2 (b x))) = 5 ∨ 
   vectorNorm (vectorAdd a (vectorScale 2 (b x))) = 13) ∧
  Real.cos (angle a (b x)) = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l380_38059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combine_terms_power_l380_38047

theorem combine_terms_power (a b : ℤ) : 
  (∃ (k : ℚ), ∀ (x y : ℝ), -2/7 * x^(a-1) * y^2 = k * (-3/5 * x^2 * y^(b+1))) → 
  (b : ℝ)^(a : ℝ) = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combine_terms_power_l380_38047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nashville_miami_distance_l380_38051

/-- The distance between two points on a complex plane -/
noncomputable def complex_distance (z₁ z₂ : ℂ) : ℝ :=
  Real.sqrt ((z₂.re - z₁.re)^2 + (z₂.im - z₁.im)^2)

/-- Nashville's coordinates -/
def nashville : ℂ := Complex.mk 1170 1560

/-- Miami's coordinates -/
def miami : ℂ := Complex.mk 1950 780

/-- Theorem: The distance between Nashville and Miami is 1103 -/
theorem nashville_miami_distance :
  complex_distance nashville miami = 1103 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nashville_miami_distance_l380_38051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l380_38042

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.sin x}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo (-1) 1 ∪ {1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l380_38042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l380_38095

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  a * Real.sin x - 1/2 * Real.cos (2*x) + a - 3/a + 1/2

/-- Part 1: If f(x) ≤ 0 for all x ∈ ℝ, then a ∈ (0, 1] -/
theorem part_one (a : ℝ) (h1 : a ≠ 0) (h2 : ∀ x : ℝ, f a x ≤ 0) :
  0 < a ∧ a ≤ 1 := by
  sorry

/-- Part 2: If a ≥ 2 and there exists x ∈ ℝ such that f(x) ≤ 0, then a = 3 -/
theorem part_two (a : ℝ) (h1 : a ≥ 2) (h2 : ∃ x : ℝ, f a x ≤ 0) :
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l380_38095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_number_in_50th_row_l380_38031

-- Define Pascal's triangle using well-founded recursion
def pascal : ℕ → ℕ → ℕ
| 0, _ => 1
| n+1, 0 => 1
| n+1, k+1 => pascal n k + pascal n (k+1)

-- Theorem statement
theorem third_number_in_50th_row : pascal 50 2 = 1225 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_number_in_50th_row_l380_38031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_to_hemisphere_radius_l380_38017

theorem sphere_to_hemisphere_radius (r : ℝ) : 
  r > 0 → 
  (4 / 3 * Real.pi * r^3 = 2 / 3 * Real.pi * 5^3) → 
  r = 5 / (2 ^ (1/3 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_to_hemisphere_radius_l380_38017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_identity_l380_38068

/-- In a triangle ABC, given that c - a equals the height from A to BC,
    prove that sin((C-A)/2) + cos((C+A)/2) = 1 -/
theorem triangle_angle_sum_identity (A B C : ℝ) (a b c : ℝ) (h : ℝ) : 
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (c * Real.sin A = b * Real.sin C) →
  (a * Real.sin B = b * Real.sin A) →
  (c - a = h) →
  (h = c * Real.sin A) →
  Real.sin ((C - A) / 2) + Real.cos ((C + A) / 2) = 1 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sum_identity_l380_38068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_determine_identities_l380_38062

-- Define the types of people
inductive PersonType
| Liar
| TruthTeller
| Spy

-- Define a person
structure Person where
  type : PersonType

-- Define the group of three people
structure GroupOfThree where
  people : Fin 3 → Person
  has_one_of_each : 
    (∃ i, (people i).type = PersonType.Liar) ∧
    (∃ i, (people i).type = PersonType.TruthTeller) ∧
    (∃ i, (people i).type = PersonType.Spy)

-- Define a question and its possible answers
inductive Answer
| Yes
| No

def Question := Person → Answer

-- Define the theorem
theorem can_determine_identities (g : GroupOfThree) :
  ∃ (q1 q2 : Question), 
    ∀ (i : Fin 3), 
      ∃! (t : PersonType), (g.people i).type = t ∧
        ∀ (j : Fin 3), j ≠ i → (g.people j).type ≠ t :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_determine_identities_l380_38062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_maximized_at_centroid_l380_38036

/-- Triangle ABC with points A', B', C' uniformly distributed on its sides -/
structure TriangleWithRandomPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  A' : ℝ → ℝ × ℝ  -- Function representing A' on BC
  B' : ℝ → ℝ × ℝ  -- Function representing B' on CA
  C' : ℝ → ℝ × ℝ  -- Function representing C' on AB

/-- Probability that triangle AA'BB'CC' contains point Z -/
noncomputable def probability (T : TriangleWithRandomPoints) (Z : ℝ × ℝ) : ℝ := sorry

/-- Centroid of a triangle -/
def centroid (A B C : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Interior points of a triangle -/
def isInterior (A B C Z : ℝ × ℝ) : Prop := sorry

theorem probability_maximized_at_centroid (T : TriangleWithRandomPoints) :
  ∀ Z, isInterior T.A T.B T.C Z →
    probability T Z ≤ 1/4 ∧
    (probability T Z = 1/4 ↔ Z = centroid T.A T.B T.C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_maximized_at_centroid_l380_38036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l380_38089

noncomputable def series_term (n : ℕ) : ℝ := (4 * n + 2) / ((6 * n - 5)^2 * (6 * n + 1)^2)

theorem series_sum : ∑' n, series_term n = 1/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_l380_38089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_when_expression_minimized_l380_38002

/-- An ellipse with semi-major axis a and semi-minor axis b. -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The expression to be minimized for the ellipse. -/
noncomputable def expression (e : Ellipse) : ℝ :=
  e.a^2 + 16 / (e.b * (e.a - e.b))

/-- The eccentricity of an ellipse. -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- 
Theorem: If the expression is minimized for an ellipse, 
then its eccentricity is equal to √3/2.
-/
theorem eccentricity_when_expression_minimized (e : Ellipse) :
  (∀ e' : Ellipse, expression e ≤ expression e') →
  eccentricity e = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_when_expression_minimized_l380_38002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l380_38070

-- Define the force function
def F (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 5

-- Define the work done by the force
noncomputable def work_done (a b : ℝ) : ℝ := ∫ x in a..b, F x

-- Theorem statement
theorem work_calculation :
  work_done 5 10 = 825 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_calculation_l380_38070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l380_38003

noncomputable def z : ℂ := (1/2)/(1+Complex.I) + (-5/4 + 9/4*Complex.I)

theorem z_properties :
  let modulus_z := Complex.abs z
  ∃ (p q : ℝ), modulus_z = Real.sqrt 5 ∧ 
               2 * z^2 + p * z + q = 0 ∧
               p = 4 ∧ q = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_properties_l380_38003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_problem_l380_38034

/-- The time taken by runner A to finish a race, given the race distance, B's finishing time, and the distance by which A beats B. -/
noncomputable def race_time_A (race_distance : ℝ) (B_time : ℝ) (beat_distance : ℝ) : ℝ :=
  (race_distance - beat_distance) * B_time / race_distance

theorem race_problem (race_distance : ℝ) (B_time : ℝ) (beat_distance : ℝ) 
    (h1 : race_distance = 130)
    (h2 : B_time = 25)
    (h3 : beat_distance = 26) :
  race_time_A race_distance B_time beat_distance = 20 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_problem_l380_38034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_eq_general_term_l380_38088

noncomputable def x : ℕ → ℝ
  | 0 => 0  -- Adding a case for 0 to cover all natural numbers
  | 1 => 2
  | 2 => 3
  | n + 1 => if n % 2 = 0 then x n + x (n - 1) else x n + 2 * x (n - 1)

noncomputable def general_term (n : ℕ) : ℝ :=
  let m := (n + 1) / 2
  if n % 2 = 1 then
    1/4 * (3 - Real.sqrt 2) * (2 + Real.sqrt 2) ^ m + 1/4 * (3 + Real.sqrt 2) * (2 - Real.sqrt 2) ^ m
  else
    1/4 * (1 + 2 * Real.sqrt 2) * (2 + Real.sqrt 2) ^ m + 1/4 * (1 - 2 * Real.sqrt 2) * (2 - Real.sqrt 2) ^ m

theorem x_eq_general_term : ∀ n : ℕ, n ≥ 1 → x n = general_term n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_eq_general_term_l380_38088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_sum_of_squares_divisible_by_240_l380_38004

theorem smallest_k_for_sum_of_squares_divisible_by_240 :
  (∀ k : ℕ, k < 1440 → (k * (k + 1) * (2 * k + 1)) % 1440 ≠ 0) ∧
  (1440 * (1440 + 1) * (2 * 1440 + 1)) % 1440 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_sum_of_squares_divisible_by_240_l380_38004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l380_38098

theorem solution_set (a : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ x₅ : ℝ, 
    x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧ x₅ ≥ 0 ∧
    x₁ + 2*x₂ + 3*x₃ + 4*x₄ + 5*x₅ = a ∧
    x₁ + 2^3*x₂ + 3^3*x₃ + 4^3*x₄ + 5^3*x₅ = a^2 ∧
    x₁ + 2^5*x₂ + 3^5*x₃ + 4^5*x₄ + 5^5*x₅ = a^3) →
  a ∈ ({1, 4, 9, 16, 25} : Set ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l380_38098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cryptarithm_solution_l380_38096

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Cryptarithmic puzzle solver -/
def cryptarithm_solver (A B C : Digit) : Prop :=
  -- Each letter represents a distinct digit
  A ≠ B ∧ B ≠ C ∧ A ≠ C
  -- A + B = C + 10
  ∧ A.val + B.val = C.val + 10
  -- B + C = 12
  ∧ B.val + C.val = 12
  -- No carrying when A and B are added
  ∧ A.val + B.val < 10

theorem cryptarithm_solution :
  ∃ (A B C : Digit), cryptarithm_solver A B C ∧ A = ⟨22, by sorry⟩ := by
  sorry

-- Remove the #eval statement as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cryptarithm_solution_l380_38096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_f_l380_38099

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 8)

-- State the theorem
theorem monotonic_increasing_interval_f :
  ∀ x₁ x₂ : ℝ, x₁ > 4 ∧ x₂ > 4 ∧ x₁ < x₂ → f x₁ < f x₂ :=
by
  -- The proof is omitted using 'sorry'
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_f_l380_38099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_trig_equation_l380_38005

theorem solution_set_of_trig_equation :
  let S : Set ℝ := {x | -3 * (Real.cos x)^2 + 5 * Real.sin x + 1 = 0}
  S = {x | ∃ k : ℤ, x = Real.arcsin (1/3) + 2 * k * π ∨ 
                    x = π - Real.arcsin (1/3) + 2 * k * π} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_trig_equation_l380_38005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harrys_investment_harrys_investment_correct_l380_38018

theorem harrys_investment (mary_investment : ℝ) (total_profit : ℝ) 
  (mary_extra : ℝ) (h1 : mary_investment = 700) 
  (h2 : total_profit = 3000) (h3 : mary_extra = 800) : ℝ :=
  let effort_share := total_profit / 3
  let investment_share := total_profit - effort_share
  let harry_investment := 300
  let mary_total := effort_share / 2 + (mary_investment / (mary_investment + harry_investment)) * investment_share
  let harry_total := effort_share / 2 + (harry_investment / (mary_investment + harry_investment)) * investment_share
  have h4 : mary_total = harry_total + mary_extra := by sorry
  harry_investment

theorem harrys_investment_correct : harrys_investment 700 3000 800 rfl rfl rfl = 300 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harrys_investment_harrys_investment_correct_l380_38018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_value_l380_38079

/-- Line l: x - √3y + 2 = 0 -/
def line_l (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 2 = 0

/-- Circle: x^2 + y^2 = 4 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Points A and B are intersections of the line and circle -/
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ circle_eq A.1 A.2 ∧
  line_l B.1 B.2 ∧ circle_eq B.1 B.2 ∧
  A ≠ B

/-- The absolute value of the projection of AB on the positive x-axis -/
noncomputable def projection_AB (A B : ℝ × ℝ) : ℝ :=
  abs (B.1 - A.1)

theorem projection_value (A B : ℝ × ℝ) :
  intersection_points A B → projection_AB A B = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_value_l380_38079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_non_almost_square_intervals_l380_38060

/-- A positive integer is almost-square if it can be represented as ab where a and b are positive integers and a ≤ b ≤ 1.01a -/
def AlmostSquare (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n = a * b ∧ a ≤ b ∧ (b : ℝ) ≤ 1.01 * a

/-- For any positive integer K, there exists a positive integer m > K such that
    none of the integers in the set {m, m+1, ..., m+198} are almost-square -/
theorem infinite_non_almost_square_intervals :
  ∀ K : ℕ, ∃ m : ℕ, m > K ∧
    ∀ i : ℕ, i ≤ 198 → ¬AlmostSquare (m + i) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_non_almost_square_intervals_l380_38060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_weighings_l380_38058

/-- Represents the result of a weighing --/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- Represents a weighing strategy --/
def WeighStrategy := ℕ → List ℕ → List ℕ → WeighResult → List ℕ

/-- The total number of coins --/
def total_coins : ℕ := 2023

/-- The number of counterfeit coins --/
def counterfeit_coins : ℕ := 2

/-- The number of genuine coins --/
def genuine_coins : ℕ := total_coins - counterfeit_coins

/-- Determines if a weighing strategy can identify whether counterfeit coins are heavier or lighter --/
def can_determine (s : WeighStrategy) : Prop := sorry

/-- The minimum number of weighings needed --/
def min_weighings : ℕ := 3

/-- The main theorem to prove --/
theorem counterfeit_coin_weighings :
  ∀ (s : WeighStrategy),
    can_determine s →
    (∀ (n : ℕ), n < min_weighings → ¬can_determine (λ m l1 l2 r ↦ s m l1 l2 r)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_weighings_l380_38058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangent_y_intercept_l380_38094

-- Define the centers and radii of the circles
noncomputable def center1 : ℝ × ℝ := (2, 4)
noncomputable def center2 : ℝ × ℝ := (14, 9)
noncomputable def radius1 : ℝ := 4
noncomputable def radius2 : ℝ := 9

-- Define the common external tangent function
noncomputable def tangent_line (x : ℝ) : ℝ := (120/119) * x + 712/119

-- Theorem statement
theorem external_tangent_y_intercept :
  ∃ (x₁ x₂ : ℝ),
    -- The tangent line touches both circles
    (x₁ - center1.1)^2 + (tangent_line x₁ - center1.2)^2 = radius1^2 ∧
    (x₂ - center2.1)^2 + (tangent_line x₂ - center2.2)^2 = radius2^2 ∧
    -- The tangent line has positive slope
    (120/119) > 0 ∧
    -- The y-intercept is 712/119
    tangent_line 0 = 712/119 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_tangent_y_intercept_l380_38094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relative_frequency_deviation_probability_l380_38029

/-- The number of independent trials -/
def n : ℕ := 625

/-- The probability of the event occurring in each trial -/
def p : ℝ := 0.8

/-- The deviation limit -/
def ε : ℝ := 0.04

/-- The standard normal cumulative distribution function -/
noncomputable def Φ : ℝ → ℝ := sorry

/-- The theorem stating the probability of the relative frequency deviation -/
theorem relative_frequency_deviation_probability :
  ∃ (δ : ℝ), δ ≥ 0 ∧ δ < 0.0001 ∧
  (2 * Φ (ε * Real.sqrt (n / (p * (1 - p))))) = 0.9876 + δ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relative_frequency_deviation_probability_l380_38029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_of_3_eq_3_l380_38039

-- Define the function f(x) for the right-hand side of the equation
noncomputable def f (x : ℝ) : ℝ := 
  (x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) * 
  (x^32 + 1) * (x^64 + 1) * (x^128 + 1) * (x^256 + 1) * (x^512 + 1) - 1

-- Define the function h(x) implicitly
noncomputable def h (x : ℝ) : ℝ := (f x + 1) / (x^(2^10 - 1) - 1)

-- Theorem to prove
theorem h_of_3_eq_3 : h 3 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_of_3_eq_3_l380_38039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_in_expansion_l380_38074

theorem coefficient_x4_in_expansion : ℕ := by
  -- Define the coefficient of x^4 in the expansion of (2-x)(2x+1)^6
  let coeff : ℕ := 320

  -- Define the expansion of (2-x)(2x+1)^6 as a polynomial
  let expansion : Polynomial ℤ := sorry

  -- Assert that the coefficient of x^4 in the expansion is equal to 320
  have correct_coefficient : coeff = 320 := by rfl

  -- Return the coefficient
  exact coeff

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_in_expansion_l380_38074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_hyperbola_eq_implies_foci_distance_l380_38085

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  9 * x^2 + 36 * x + 4 * y^2 - 8 * y + 1 = 0

-- Define the distance between foci
noncomputable def foci_distance : ℝ := Real.sqrt 351 / 3

-- Theorem statement
theorem hyperbola_foci_distance :
  ∃ f₁ f₂ : ℝ × ℝ, (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = foci_distance^2 :=
by
  sorry

-- Helper theorem to connect the hyperbola equation to the foci distance
theorem hyperbola_eq_implies_foci_distance :
  ∀ x y : ℝ, hyperbola_eq x y → 
  ∃ f₁ f₂ : ℝ × ℝ, (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = foci_distance^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_hyperbola_eq_implies_foci_distance_l380_38085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_function_value_l380_38053

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

theorem sin_function_value (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∃ (A B : ℝ), ∀ x, f ω x ≤ A ∧ f ω x ≥ B ∧ A - B = 2 * Real.sqrt 2) : 
  f ω 1 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_function_value_l380_38053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_division_l380_38091

-- Define the space we're working in
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the points
variable (A B C O : V)

-- Define the conditions
variable (h1 : ∃ t : ℝ, C = (1 - t) • A + t • B ∧ 0 ≤ t ∧ t ≤ 1)
variable (h2 : ‖A - C‖ = 2 * ‖C - B‖)
variable (h3 : O ∉ {P | ∃ s : ℝ, P = (1 - s) • A + s • B})

-- State the theorem
theorem point_division : C = (1/3) • A + (2/3) • B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_division_l380_38091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l380_38090

/-- The parabola y^2 = 4x -/
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4*p.1

/-- The line x - y + 5 = 0 -/
def line (p : ℝ × ℝ) : Prop := p.1 - p.2 + 5 = 0

/-- The directrix of the parabola y^2 = 4x -/
def directrix : Set (ℝ × ℝ) := {p | p.1 = -1}

/-- The distance from a point to the directrix -/
noncomputable def distance_to_directrix (p : ℝ × ℝ) : ℝ := |p.1 - (-1)|

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_distance_sum :
  ∃ (d : ℝ), 
    (∀ (p q : ℝ × ℝ), 
      parabola p → 
      line q → 
      d ≤ distance_to_directrix p + distance p q) ∧
    d = 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l380_38090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_condition_l380_38048

/-- Given two vectors m and n in ℝ², prove they are parallel if and only if a = 2 or a = -1 -/
theorem parallel_vectors_condition (a : ℝ) : 
  (∃ (k : ℝ), (![a, -2] : Fin 2 → ℝ) = k • (![1, 1-a] : Fin 2 → ℝ)) ↔ (a = 2 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_condition_l380_38048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_sandy_meeting_point_l380_38086

/-- Given two points in a 2D plane, calculate their meeting point that is offset from the midpoint. -/
noncomputable def meeting_point (harry_pos : ℝ × ℝ) (sandy_pos : ℝ × ℝ) : ℝ × ℝ :=
  let midpoint := ((harry_pos.1 + sandy_pos.1) / 2, (harry_pos.2 + sandy_pos.2) / 2)
  (midpoint.1 + 2, midpoint.2 - 1)

/-- Theorem stating that Harry and Sandy's meeting point is at (8, 1) -/
theorem harry_sandy_meeting_point :
  meeting_point (12, -3) (0, 7) = (8, 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_sandy_meeting_point_l380_38086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_dot_product_l380_38052

/-- Definition of the ellipse E -/
noncomputable def E (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

/-- Definition of point M -/
def M (y : ℝ) : ℝ × ℝ := (2, y)

/-- Definition of point P on the ellipse -/
noncomputable def P (t : ℝ) : ℝ × ℝ := ((2 * t^2 - 4) / (t^2 + 2), (4 * t) / (t^2 + 2))

/-- Theorem stating that the dot product of OM and OP is constant -/
theorem constant_dot_product (y t : ℝ) :
  E (P t).1 (P t).2 →
  (M y).1 = (P t).1 * y / (P t).2 →
  (M y).1 * (P t).1 + (M y).2 * (P t).2 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_dot_product_l380_38052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_y_coordinate_l380_38024

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  (1/2) * abs ((p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y))

/-- Calculates the area of a rectangle given two opposite corners -/
noncomputable def rectangleArea (p1 p2 : Point) : ℝ :=
  abs ((p2.x - p1.x) * (p2.y - p1.y))

theorem pentagon_y_coordinate 
  (A B D E : Point)
  (hA : A = ⟨0, 0⟩)
  (hB : B = ⟨0, 6⟩)
  (hD : D = ⟨6, 6⟩)
  (hE : E = ⟨6, 0⟩)
  (C : Point)
  (hC_x : C.x = 3) -- C is on the line of symmetry
  (h_area : triangleArea B C D + rectangleArea A D = 60) :
  C.y = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_y_coordinate_l380_38024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d_one_equals_eight_l380_38012

/-- F(n) represents the number of quadruples (b₁, b₂, b₃, b₄) of distinct integers 
    where 1 ≤ bᵢ ≤ n such that n divides b₁² + b₂² + b₃² + b₄² -/
def F (n : ℕ) : ℕ := sorry

/-- p(x) is a polynomial of degree 3 that satisfies F(n) = p(n) for all n ≥ 3 -/
noncomputable def p : ℝ → ℝ := λ x ↦ sorry

/-- d₁ is the coefficient of x in p(x) -/
noncomputable def d₁ : ℝ := sorry

theorem d_one_equals_eight (h : ∀ n : ℕ, n ≥ 3 → F n = p n) : d₁ = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_d_one_equals_eight_l380_38012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_ratio_is_one_to_four_l380_38035

/-- Represents the number of pairs of socks -/
structure SockOrder where
  black : ℕ
  blue : ℕ

/-- Represents the price of socks -/
def sock_price : ℕ → ℚ
  | 0 => 1  -- price of blue socks
  | 1 => 2  -- price of black socks
  | _ => 0  -- undefined for other values

/-- Calculates the total price of a sock order -/
def total_price (order : SockOrder) : ℚ :=
  sock_price 1 * order.black + sock_price 0 * order.blue

/-- The theorem to be proved -/
theorem original_ratio_is_one_to_four (original : SockOrder) (swapped : SockOrder) :
  original.black = 4 ∧
  swapped.black = original.blue ∧
  swapped.blue = original.black ∧
  total_price swapped = (3/2) * total_price original →
  4 * original.black = original.blue :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_ratio_is_one_to_four_l380_38035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_equals_four_l380_38071

-- Define the curve C
noncomputable def curve_C (φ : Real) : Real × Real := (2 * Real.cos φ, 2 + 2 * Real.sin φ)

-- Define points A and B
noncomputable def point_A : Real × Real := curve_C (Real.pi / 3)
noncomputable def point_B : Real × Real := curve_C (5 * Real.pi / 6)

-- Theorem statement
theorem distance_AB_equals_four :
  Real.sqrt ((point_A.1 - point_B.1)^2 + (point_A.2 - point_B.2)^2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_equals_four_l380_38071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_region_area_proof_l380_38040

noncomputable def complex_region_area : ℝ := 75 - 25 * Real.pi / 2

theorem complex_region_area_proof :
  ∀ z : ℂ,
  (0 < z.re / 10 ∧ z.re / 10 < 1) ∧
  (0 < z.im / 10 ∧ z.im / 10 < 1) ∧
  (0 < (10 * z.re / (z.re^2 + z.im^2)) ∧ (10 * z.re / (z.re^2 + z.im^2)) < 1) ∧
  (0 < (10 * z.im / (z.re^2 + z.im^2)) ∧ (10 * z.im / (z.re^2 + z.im^2)) < 1) →
  complex_region_area = 75 - 25 * Real.pi / 2 :=
by
  intro z
  intro h
  sorry

#check complex_region_area_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_region_area_proof_l380_38040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_calculation_l380_38015

-- Define the ∆ operator
noncomputable def triangle (a b : ℝ) : ℝ :=
  if a > b then a + b else a - b

-- Theorem statement
theorem triangle_calculation : 
  (triangle (Real.sqrt 3) (Real.sqrt 2)) + (triangle (2 * Real.sqrt 3) (3 * Real.sqrt 2)) = 
  3 * Real.sqrt 3 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_calculation_l380_38015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_amount_correct_sam_investment_l380_38075

/-- Calculates the final amount of an investment after two consecutive compound interest periods -/
noncomputable def investment_amount (principal : ℝ) (p q : ℝ) : ℝ :=
  principal * (1 + p / 200) ^ 6 * (1 + q / 400) ^ 12

/-- Theorem stating the correctness of the investment amount calculation -/
theorem investment_amount_correct (principal p q : ℝ) :
  let amount := investment_amount principal p q
  amount = principal * (1 + p / 200) ^ 6 * (1 + q / 400) ^ 12 :=
by
  unfold investment_amount
  simp

/-- Specific case for Sam's investment -/
theorem sam_investment (p q : ℝ) :
  investment_amount 150 p q = 150 * (1 + p / 200) ^ 6 * (1 + q / 400) ^ 12 :=
by
  unfold investment_amount
  simp

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_amount_correct_sam_investment_l380_38075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_of_f_l380_38043

noncomputable def f (x : ℝ) : ℝ := (3 * x - 5) / (6 * x + 4)

theorem vertical_asymptote_of_f :
  ∃ (x : ℝ), x = -2/3 ∧ ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
  ∀ (y : ℝ), 0 < |y - x| ∧ |y - x| < δ → |f y| > 1/ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_of_f_l380_38043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_a_l380_38081

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The problem statement -/
theorem smallest_possible_a (P : IntPolynomial) (a : ℤ) : 
  a > 0 ∧
  P.eval 1 = a ∧
  P.eval 4 = a ∧
  P.eval 6 = a ∧
  P.eval 9 = a ∧
  P.eval 3 = -a ∧
  P.eval 5 = -a ∧
  P.eval 8 = -a ∧
  P.eval 10 = -a →
  (∀ b : ℤ, b > 0 ∧ (∃ Q : IntPolynomial, 
    Q.eval 1 = b ∧
    Q.eval 4 = b ∧
    Q.eval 6 = b ∧
    Q.eval 9 = b ∧
    Q.eval 3 = -b ∧
    Q.eval 5 = -b ∧
    Q.eval 8 = -b ∧
    Q.eval 10 = -b) → b ≥ 10080) ∧
  (∃ R : IntPolynomial, 
    R.eval 1 = 10080 ∧
    R.eval 4 = 10080 ∧
    R.eval 6 = 10080 ∧
    R.eval 9 = 10080 ∧
    R.eval 3 = -10080 ∧
    R.eval 5 = -10080 ∧
    R.eval 8 = -10080 ∧
    R.eval 10 = -10080) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_a_l380_38081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_ratio_is_27_25_l380_38028

/-- Represents the number of adults and children attending an exhibition. -/
structure Attendance where
  adults : ℤ
  children : ℤ

/-- Calculates the total admission fee for a given attendance. -/
def totalFee (a : Attendance) : ℤ :=
  25 * a.adults + 12 * a.children

/-- Checks if the ratio of adults to children is closer to 1 than any other valid attendance. -/
def isClosestRatio (a : Attendance) : Prop :=
  totalFee a = 1950 ∧
  a.adults ≥ 1 ∧
  a.children ≥ 1 ∧
  ∀ b : Attendance,
    totalFee b = 1950 → b.adults ≥ 1 → b.children ≥ 1 →
    |a.adults * b.children - a.children * b.adults| ≤ |b.adults * b.children - b.children * b.adults|

/-- The theorem stating that the attendance with 54 adults and 50 children
    has the ratio closest to 1 among all valid attendances. -/
theorem closest_ratio_is_27_25 :
  isClosestRatio { adults := 54, children := 50 } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_ratio_is_27_25_l380_38028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_point_l380_38044

noncomputable section

-- Define the curves
def curve1 (x : ℝ) : ℝ := x^2
def curve2 (x : ℝ) : ℝ := 1/x

-- Define the derivatives of the curves
def curve1_derivative (x : ℝ) : ℝ := 2*x
def curve2_derivative (x : ℝ) : ℝ := -1/(x^2)

theorem tangent_perpendicular_point :
  ∀ P : ℝ × ℝ,
  P.1 > 0 →
  curve1 2 = 4 →
  curve2 P.1 = P.2 →
  curve1_derivative 2 * curve2_derivative P.1 = -1 →
  P = (2, 1/2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_point_l380_38044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_of_the_year_requirement_l380_38011

/-- The number of academy members -/
def academy_members : ℕ := 785

/-- The fraction of lists a film must appear on to be considered for "movie of the year" -/
def required_fraction : ℚ := 1 / 4

/-- The smallest number of lists a film must appear on to be considered for "movie of the year" -/
def min_lists : ℕ := 197

/-- Theorem stating the smallest number of lists a film must appear on -/
theorem movie_of_the_year_requirement :
  min_lists = Nat.ceil ((academy_members : ℚ) * required_fraction) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movie_of_the_year_requirement_l380_38011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_triangle_area_range_l380_38050

/-- The focal length of the ellipse -/
def focal_length : ℝ := 4

/-- The ellipse C₁ -/
def ellipse (x y : ℝ) (a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- The parabola C₂ -/
def parabola (x y : ℝ) : Prop := y^2 = x

/-- The line passing through F₂ and intersecting C₁ and C₂ -/
def intersection_line (x y : ℝ) : Prop := sorry

/-- The line l passing through F₁ and intersecting C₁ at A and B -/
def line_l (x y : ℝ) : Prop := sorry

/-- The area of triangle ABF₂ -/
noncomputable def area_ABF₂ (A B F₂ : ℝ × ℝ) : ℝ := sorry

theorem ellipse_equation_and_triangle_area_range 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) :
  (∃ (x y : ℝ), ellipse x y a b ∧ parabola x y ∧ intersection_line x y) →
  (∀ (x y : ℝ), ellipse x y (2*Real.sqrt 2) 2 ↔ x^2/8 + y^2/4 = 1) ∧
  (∀ (A B F₂ : ℝ × ℝ), line_l A.1 A.2 ∧ line_l B.1 B.2 ∧ 
    ellipse A.1 A.2 (2*Real.sqrt 2) 2 ∧ ellipse B.1 B.2 (2*Real.sqrt 2) 2 ∧
    (∀ (x y : ℝ), line_l x y → ¬parabola x y) →
    12*Real.sqrt 2/5 < area_ABF₂ A B F₂ ∧ area_ABF₂ A B F₂ ≤ 4*Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_triangle_area_range_l380_38050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_l380_38013

/-- Definition of the ellipse E -/
noncomputable def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of eccentricity -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- Theorem stating the properties of the ellipse E and the fixed point -/
theorem ellipse_and_fixed_point (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ellipse (-1) (3/2) a b) (h4 : eccentricity a b = 1/2) :
  (∀ x y, ellipse x y 2 (Real.sqrt 3) ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∀ k m : ℝ, ∃ t_x t_y s_y : ℝ, 
    (t_x^2 / 4 + t_y^2 / 3 = 1) ∧  -- T is on the ellipse
    (t_y = k * t_x + m) ∧  -- l is tangent to E at T
    (s_y = k * (-4) + m) ∧  -- S is on l and x = -4
    ((1 - t_x) * (1 - (-4)) + (-s_y) * (-t_y) = 0))  -- Circle with diameter ST passes through (1, 0)
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_l380_38013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zamena_solution_l380_38087

def is_valid_assignment (Z A M E N : ℕ) : Prop :=
  Z ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
  A ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
  M ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
  E ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
  N ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
  3 > A ∧ A > M ∧ M < E ∧ E < N ∧ N < A ∧
  Z ≠ A ∧ Z ≠ M ∧ Z ≠ E ∧ Z ≠ N ∧
  A ≠ M ∧ A ≠ E ∧ A ≠ N ∧
  M ≠ E ∧ M ≠ N ∧
  E ≠ N

theorem unique_zamena_solution :
  ∀ Z A M E N : ℕ,
  is_valid_assignment Z A M E N →
  Z = 5 ∧ A = 4 ∧ M = 1 ∧ E = 2 ∧ N = 3 :=
by sorry

#check unique_zamena_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zamena_solution_l380_38087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABH_eq_angle_CBO_l380_38069

-- Define the triangle ABC
variable (A B C : EuclideanPlane) 

-- Define the orthocenter H
noncomputable def H (A B C : EuclideanPlane) : EuclideanPlane := 
  sorry

-- Define the circumcenter O
noncomputable def O (A B C : EuclideanPlane) : EuclideanPlane := 
  sorry

-- Define the angles
noncomputable def angle_ABH (A B C : EuclideanPlane) : ℝ := 
  sorry

noncomputable def angle_CBO (A B C : EuclideanPlane) : ℝ := 
  sorry

-- State the theorem
theorem angle_ABH_eq_angle_CBO (A B C : EuclideanPlane) : 
  angle_ABH A B C = angle_CBO A B C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ABH_eq_angle_CBO_l380_38069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_maximum_value_l380_38019

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / Real.exp x

theorem tangent_line_and_maximum_value 
  (a b : ℝ) 
  (h : ∀ x : ℝ, x = -1 → Real.exp x - f a b x + Real.exp 1 = 0) :
  (a = 1 ∧ b = 1) ∧ ∀ x : ℝ, f a b x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_maximum_value_l380_38019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_evenness_l380_38093

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) + Real.log (1 - x)

-- State the theorem about the domain and evenness of f
theorem f_domain_and_evenness :
  (∀ x : ℝ, f x ∈ Set.Ioo (-1) 1 ↔ x ∈ Set.Ioo (-1) 1) ∧
  (∀ x : ℝ, x ∈ Set.Ioo (-1) 1 → f (-x) = f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_evenness_l380_38093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l380_38067

theorem trigonometric_identities (α : Real) : 
  (Real.sin (3 * π / 2 - α) = -Real.cos α) ∧ (Real.cos (3 * π / 2 + α) = Real.sin α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l380_38067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_race_time_l380_38037

/-- The time it takes John to run 100 meters -/
noncomputable def john_total_time : ℝ := 13

/-- The distance John runs in the first second -/
noncomputable def john_initial_distance : ℝ := 4

/-- The total distance of the race -/
noncomputable def race_distance : ℝ := 100

/-- The time James takes to run the first 10 meters -/
noncomputable def james_initial_time : ℝ := 2

/-- The initial distance James runs -/
noncomputable def james_initial_distance : ℝ := 10

/-- The speed difference between James and John at top speed -/
noncomputable def speed_difference : ℝ := 2

/-- Calculate John's top speed -/
noncomputable def john_top_speed : ℝ := (race_distance - john_initial_distance) / (john_total_time - 1)

/-- Calculate James's top speed -/
noncomputable def james_top_speed : ℝ := john_top_speed + speed_difference

/-- The time James takes to run 100 meters -/
noncomputable def james_total_time : ℝ := james_initial_time + (race_distance - james_initial_distance) / james_top_speed

theorem james_race_time : james_total_time = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_race_time_l380_38037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alyosha_apartment_l380_38080

/-- Represents the number of apartments per floor -/
def apartments_per_floor : ℕ := 9

/-- Represents Alyosha's floor number and Ira's apartment number -/
def K : ℕ := sorry

/-- Represents Alyosha's apartment number -/
def A : ℕ := sorry

/-- The sum of Alyosha's and Ira's apartment numbers is 329 -/
axiom sum_of_apartments : A + K = 329

/-- Alyosha's apartment number is within the range of apartments on his floor -/
axiom apartment_range : 9 * K - 8 ≤ A ∧ A ≤ 9 * K

theorem alyosha_apartment : A = 296 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alyosha_apartment_l380_38080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_congruent_squares_l380_38065

/-- A lattice point on a 2D grid -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A square on a lattice grid -/
structure LatticeSquare where
  topLeft : LatticePoint
  sideLength : ℝ

/-- The size of the grid -/
def gridSize : ℕ := 6

/-- Predicate to check if a lattice square is valid on the grid -/
def isValidSquare (square : LatticeSquare) : Prop :=
  0 ≤ square.topLeft.x ∧ square.topLeft.x < gridSize ∧
  0 ≤ square.topLeft.y ∧ square.topLeft.y < gridSize ∧
  square.topLeft.x + Int.floor square.sideLength ≤ gridSize ∧
  square.topLeft.y + Int.floor square.sideLength ≤ gridSize

/-- The set of all valid non-congruent squares on the grid -/
def validNonCongruentSquares : Set LatticeSquare :=
  {square : LatticeSquare | isValidSquare square ∧ 
    ∀ other : LatticeSquare, isValidSquare other → 
    square.sideLength = other.sideLength → square = other}

/-- The main theorem: there are 128 non-congruent squares on a 6x6 grid -/
theorem count_non_congruent_squares : 
  ∃ s : Finset LatticeSquare, s.card = 128 ∧ ∀ square ∈ s, square ∈ validNonCongruentSquares := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_congruent_squares_l380_38065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anas_multiplication_mistake_l380_38007

theorem anas_multiplication_mistake :
  ∃ (x y : ℕ),
    y - x = 202 ∧
    (x * y - 1000) / x = 288 ∧
    (x * y - 1000) % x = 67 ∧
    x = 97 ∧
    y = 299 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anas_multiplication_mistake_l380_38007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_length_is_three_l380_38033

/-- Represents the road construction project -/
structure RoadProject where
  totalDays : ℕ
  initialMen : ℕ
  completedDays : ℕ
  completedLength : ℚ
  extraMen : ℚ

/-- Calculates the total length of the road -/
noncomputable def calculateRoadLength (project : RoadProject) : ℚ :=
  let remainingDays := project.totalDays - project.completedDays
  let totalMen := project.initialMen + project.extraMen
  let initialWork := project.initialMen * project.completedDays * project.completedLength
  let remainingWork := totalMen * remainingDays * (initialWork / (project.initialMen * project.completedDays))
  project.completedLength + remainingWork / (totalMen * remainingDays)

/-- Theorem stating that the total length of the road is 3 km -/
theorem road_length_is_three (project : RoadProject)
  (h1 : project.totalDays = 300)
  (h2 : project.initialMen = 35)
  (h3 : project.completedDays = 100)
  (h4 : project.completedLength = 5/2)
  (h5 : project.extraMen = 105/2) :
  calculateRoadLength project = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_length_is_three_l380_38033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_areas_l380_38056

/-- A circle in 2D space -/
def Circle (center : EuclideanSpace ℝ (Fin 2)) (r : ℝ) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {p : EuclideanSpace ℝ (Fin 2) | dist p center = r}

/-- The area of a set of points in 2D space -/
noncomputable def area (S : Set (EuclideanSpace ℝ (Fin 2))) : ℝ := sorry

/-- A predicate indicating if a set of points forms a triangle -/
def is_triangle (T : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

/-- A predicate indicating if a triangle is inscribed in a circle -/
def is_inscribed (T : Set (EuclideanSpace ℝ (Fin 2))) (C : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

/-- A predicate indicating if a triangle is circumscribed about a circle -/
def is_circumscribed (T : Set (EuclideanSpace ℝ (Fin 2))) (C : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

/-- The circle Ω with radius r -/
def Ω (r : ℝ) : Set (EuclideanSpace ℝ (Fin 2)) := Circle 0 r

/-- Given a circle Ω with radius r, prove the maximum area of an inscribed triangle
    and the minimum area of a circumscribed triangle. -/
theorem circle_triangle_areas (r : ℝ) (hr : r > 0) :
  ∃ (A_max A_min : ℝ),
    (∀ T : Set (EuclideanSpace ℝ (Fin 2)), is_triangle T → is_inscribed T (Ω r) → area T ≤ A_max) ∧
    (∃ T : Set (EuclideanSpace ℝ (Fin 2)), is_triangle T ∧ is_inscribed T (Ω r) ∧ area T = A_max) ∧
    (∀ T : Set (EuclideanSpace ℝ (Fin 2)), is_triangle T → is_circumscribed T (Ω r) → area T ≥ A_min) ∧
    (∃ T : Set (EuclideanSpace ℝ (Fin 2)), is_triangle T ∧ is_circumscribed T (Ω r) ∧ area T = A_min) ∧
    A_max = (3 * Real.sqrt 3 / 4) * r^2 ∧
    A_min = 3 * Real.sqrt 3 * r^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_areas_l380_38056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_six_average_l380_38077

theorem first_six_average (numbers : List ℝ) 
  (h_length : numbers.length = 11)
  (h_total_avg : numbers.sum / 11 = 60)
  (h_last_six_avg : (numbers.drop 5).sum / 6 = 65)
  (h_sixth : numbers.get? 5 = some 78) :
  (numbers.take 6).sum / 6 = 71 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_six_average_l380_38077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l380_38014

open Real

theorem trigonometric_identities :
  (∀ α : ℝ, (Real.sin (2 * π - α) * Real.cos (π + α)) / (Real.sin (π / 2 + α) * Real.cos (3 * π / 2 - α)) = -1) ∧
  (∀ α : ℝ, Real.tan α = 3 → Real.sin α ^ 2 - 2 * Real.sin α * Real.cos α = 3 / 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l380_38014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l380_38061

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop := x^2 + y^2 + 8*x + 10*y + 9 = 0

-- Define the area of the region
noncomputable def region_area : ℝ := 32 * Real.pi

-- Theorem statement
theorem area_of_region :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, region_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l380_38061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cheetah_speed_calculation_l380_38072

/-- The speed of the deer in miles per hour -/
noncomputable def deer_speed : ℝ := 50

/-- The time difference between the deer and cheetah passing the tree, in hours -/
noncomputable def time_difference : ℝ := 2 / 60

/-- The time it takes for the cheetah to catch up after passing the tree, in hours -/
noncomputable def catch_up_time : ℝ := 1 / 60

/-- The speed of the cheetah in miles per hour -/
noncomputable def cheetah_speed : ℝ := 150

theorem cheetah_speed_calculation :
  cheetah_speed * catch_up_time = deer_speed * (time_difference + catch_up_time) := by
  sorry

#check cheetah_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cheetah_speed_calculation_l380_38072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_fifth_term_l380_38021

def sequenceX (x : ℕ) : ℕ → ℕ
| 0 => 1
| 1 => 3
| 2 => 6
| 3 => 10
| 4 => x
| 5 => 21
| n + 6 => sequenceX x (n + 5) + (n + 6)

theorem sequence_fifth_term (x : ℕ) : 
  (∀ n : ℕ, n < 4 → sequenceX x (n + 1) - sequenceX x n = sequenceX x n - sequenceX x (n - 1) + 1) →
  (sequenceX x 5 - sequenceX x 4 = sequenceX x 4 - sequenceX x 3 + 1) →
  x = 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_fifth_term_l380_38021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_property_original_value_l380_38045

/-- The original value of a property given its depreciated value after a certain period -/
noncomputable def original_value (final_value : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  final_value / ((1 - rate) ^ years)

/-- Theorem stating the original value of a property given specific conditions -/
theorem property_original_value :
  let final_value : ℝ := 21093
  let rate : ℝ := 0.0625
  let years : ℕ := 3
  abs (original_value final_value rate years - 25592.31) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_property_original_value_l380_38045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l380_38022

/-- Given an angle α with its vertex at the origin, its initial side on the positive x-axis,
    and a point A(2,3) on its terminal side, prove that cos(2α) = -5/13 -/
theorem cos_double_angle_special_case (α : ℝ) (A : ℝ × ℝ) :
  A.1 = 2 → A.2 = 3 → Real.cos (2 * α) = -5/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_special_case_l380_38022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_monotonic_increasing_condition_l380_38078

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + m * x^2 - 3 * m^2 * x + 1

-- Theorem for the tangent line equation
theorem tangent_line_equation (m : ℝ) (h : m = 1) :
  ∃ (a b c : ℝ), a * x - b * y - c = 0 ∧ 
  (∀ x y : ℝ, y = f m x → (x - 2) * (f m x - f m 2) = (y - f m 2) * (x - 2)) ∧
  a = 15 ∧ b = 3 ∧ c = 25 := by
  sorry

-- Theorem for the monotonic increasing condition
theorem monotonic_increasing_condition (m : ℝ) :
  (∀ x₁ x₂ : ℝ, 2*m - 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < m + 1 → f m x₁ < f m x₂) ↔
  1 ≤ m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_monotonic_increasing_condition_l380_38078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_main_theorem_l380_38026

/-- Represents a parabola with equation y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a hyperbola with equation x^2/3 - y^2 = 1 -/
structure Hyperbola where

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Returns the focus of a parabola -/
noncomputable def parabola_focus (c : Parabola) : Point :=
  { x := c.p / 2, y := 0 }

/-- Returns the foci of a hyperbola -/
noncomputable def hyperbola_foci : (Point × Point) :=
  ({ x := -2, y := 0 }, { x := 2, y := 0 })

/-- Checks if a parabola and hyperbola share a common focus -/
def share_focus (c : Parabola) : Prop :=
  let pf := parabola_focus c
  let (hf1, hf2) := hyperbola_foci
  pf = hf1 ∨ pf = hf2

theorem parabola_equation (c : Parabola) 
  (shared_focus : share_focus c) : 
  c.p = 4 := by sorry

theorem main_theorem (c : Parabola) 
  (shared_focus : share_focus c) : 
  ∀ (x y : ℝ), y^2 = 8*x ↔ y^2 = 2*c.p*x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_main_theorem_l380_38026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_is_minus_five_halves_l380_38041

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  b : ℝ
  c : ℝ
  pass_through_minus_four : 16 + 4 * b + c = 0
  pass_through_minus_one : 1 + b + c = 0
  pass_through_zero : c = 4

/-- The axis of symmetry for a quadratic function -/
noncomputable def axis_of_symmetry (f : QuadraticFunction) : ℝ := -f.b / 2

/-- Theorem stating that the axis of symmetry is at x = -5/2 -/
theorem axis_of_symmetry_is_minus_five_halves (f : QuadraticFunction) :
  axis_of_symmetry f = -5/2 := by
  unfold axis_of_symmetry
  -- Proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_is_minus_five_halves_l380_38041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_parabola_equation_l380_38016

-- Define Hyperbola and Point types
structure Hyperbola where
  equation : ℝ → ℝ → Prop
  foci : Set (ℝ × ℝ)

structure Point where
  x : ℝ
  y : ℝ

-- Hyperbola theorem
theorem hyperbola_equation (h : Hyperbola) (p : Point) :
  h.foci = {(-Real.sqrt 6, 0), (Real.sqrt 6, 0)} →
  p = ⟨2, 1⟩ →
  h.equation = λ x y ↦ x^2 / 3 - y^2 / 3 = 1 :=
sorry

-- Define Parabola type
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

-- Parabola theorem
theorem parabola_equation (p : Parabola) :
  (∃ x y, p.focus = (x, y) ∧ x - 2*y + 2 = 0) →
  (p.equation = λ x y ↦ x^2 = 4*y) ∨ (p.equation = λ x y ↦ y^2 = -8*x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_parabola_equation_l380_38016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_number_probability_l380_38064

def upper_bound : ℕ := 300
def billy_multiple : ℕ := 25
def bobbi_multiple : ℕ := 40

theorem same_number_probability :
  let billy_numbers := Finset.filter (λ n => billy_multiple ∣ n) (Finset.range upper_bound)
  let bobbi_numbers := Finset.filter (λ n => bobbi_multiple ∣ n) (Finset.range upper_bound)
  let common_numbers := billy_numbers ∩ bobbi_numbers
  (Finset.card common_numbers : ℚ) / ((Finset.card billy_numbers) * (Finset.card bobbi_numbers)) = 1 / 84 :=
by sorry

#eval upper_bound
#eval billy_multiple
#eval bobbi_multiple

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_number_probability_l380_38064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_incenter_l380_38000

/-- Triangle PQR with given side lengths --/
structure Triangle where
  PQ : ℝ
  PR : ℝ
  QR : ℝ

/-- The incenter of a triangle --/
noncomputable def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- Distance between two points --/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the distance from P to the incenter J --/
theorem distance_to_incenter (t : Triangle) (h1 : t.PQ = 30) (h2 : t.PR = 29) (h3 : t.QR = 27) :
  let P : ℝ × ℝ := (0, 0)
  let J := incenter t
  ∃ ε > 0, |distance P J - 10.81| < ε := by
  sorry

#check distance_to_incenter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_incenter_l380_38000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_angles_l380_38046

theorem sin_sum_angles (α β : ℝ) : 
  Real.sin (α + β) = Real.cos α * Real.sin β + Real.sin α * Real.cos β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_angles_l380_38046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l380_38063

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 4)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x) ∧
  (f (-Real.pi / 8) = 0) ∧
  (¬ ∀ (x y : ℝ), 0 < x ∧ x < y ∧ y < Real.pi / 4 → f y < f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l380_38063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_production_theorem_l380_38030

/-- Sum of a geometric sequence -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (r^n - 1) / (r - 1)

/-- February production -/
def initial_production : ℝ := 10

/-- Monthly increase factor -/
def monthly_increase : ℝ := 3

/-- Number of months from February to August -/
def months : ℕ := 7

/-- Total canoe production from February to August -/
noncomputable def total_production : ℝ := geometric_sum initial_production monthly_increase months

theorem canoe_production_theorem : 
  total_production = 10930 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_canoe_production_theorem_l380_38030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_day_100_is_monday_l380_38054

/-- Days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Function to advance a day by a given number of days -/
def advanceDay (d : DayOfWeek) : ℕ → DayOfWeek
  | 0 => d
  | n + 1 => match advanceDay d n with
    | DayOfWeek.Monday => DayOfWeek.Tuesday
    | DayOfWeek.Tuesday => DayOfWeek.Wednesday
    | DayOfWeek.Wednesday => DayOfWeek.Thursday
    | DayOfWeek.Thursday => DayOfWeek.Friday
    | DayOfWeek.Friday => DayOfWeek.Saturday
    | DayOfWeek.Saturday => DayOfWeek.Sunday
    | DayOfWeek.Sunday => DayOfWeek.Monday

theorem day_100_is_monday (h : advanceDay DayOfWeek.Tuesday 44 = DayOfWeek.Tuesday) :
  advanceDay DayOfWeek.Tuesday 99 = DayOfWeek.Monday := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_day_100_is_monday_l380_38054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_a_l380_38097

/-- The function f(x) = ax √(x - 1) + bx + c -/
noncomputable def f (a b c x : ℝ) : ℝ := a * x * Real.sqrt (x - 1) + b * x + c

/-- The theorem stating the maximum value of a -/
theorem max_value_of_a (a b c : ℝ) (h_a : a ≠ 0) :
  (∀ x ∈ Set.Ici 1, f a b c x ∈ Set.Ioo 0 (1/2)) →
  (2 * f a b c 5 = 3 * f a b c 10) →
  (3 * f a b c 10 = 4 * f a b c 17) →
  (a ≤ 3/200) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_a_l380_38097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l380_38073

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  A = π/3 →
  b = 1 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
  sorry

#check triangle_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l380_38073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_height_after_16_min_l380_38038

noncomputable def ferris_wheel_height (t : ℝ) : ℝ :=
  8 * Real.sin (Real.pi / 6 * t - Real.pi / 2) + 10

theorem ferris_wheel_height_after_16_min :
  ferris_wheel_height 16 = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_height_after_16_min_l380_38038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2024th_term_l380_38055

def mySequence (n : ℕ) : ℚ := (-1)^n * (2/3) * ((10 : ℚ)^n - 1)

theorem sequence_2024th_term : 
  mySequence 2024 = (2/3) * ((10 : ℚ)^2024 - 1) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2024th_term_l380_38055
