import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_13_39_l748_74839

/-- An arithmetic sequence with first term a and common difference d -/
noncomputable def arithmeticSequence (a d : ℝ) (n : ℕ) : ℝ := a + d * (n - 1)

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmeticSum (a d : ℝ) (n : ℕ) : ℝ := n * a + n * (n - 1) / 2 * d

theorem arithmetic_sequence_sum_13_39 (a d : ℝ) :
  arithmeticSum a d 13 = 39 →
  arithmeticSequence a d 6 + arithmeticSequence a d 7 + arithmeticSequence a d 8 = 9 := by
  intro h
  sorry

#check arithmetic_sequence_sum_13_39

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_13_39_l748_74839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_decreasing_f_l748_74889

/-- A function f that takes a real number and returns a real number. -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 2 * Real.sin x) / Real.cos x

/-- The theorem stating the range of m for which f is monotonically decreasing. -/
theorem range_of_m_for_decreasing_f :
  ∀ m : ℝ, (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < π/2 → f m x₁ > f m x₂) ↔ m ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_decreasing_f_l748_74889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_implies_constant_S15_l748_74826

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a₁ : ℝ  -- First term
  d : ℝ   -- Common difference

/-- The nth term of an arithmetic sequence -/
noncomputable def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a₁ + (n - 1 : ℝ) * seq.d

/-- The sum of the first n terms of an arithmetic sequence -/
noncomputable def ArithmeticSequence.sumFirstNTerms (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * seq.a₁ + (n - 1 : ℝ) * seq.d)

/-- The statement to be proved -/
theorem constant_sum_implies_constant_S15 :
  ∀ (seq₁ seq₂ : ArithmeticSequence),
    seq₁.nthTerm 5 + seq₁.nthTerm 8 + seq₁.nthTerm 11 =
    seq₂.nthTerm 5 + seq₂.nthTerm 8 + seq₂.nthTerm 11 →
    seq₁.sumFirstNTerms 15 = seq₂.sumFirstNTerms 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_implies_constant_S15_l748_74826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_origin_l748_74841

-- Define the two functions as noncomputable
noncomputable def f (x : ℝ) : ℝ := 3^x
noncomputable def g (x : ℝ) : ℝ := -3^(-x)

-- Theorem stating the symmetry about the origin
theorem symmetry_about_origin :
  ∀ a : ℝ, ∃ b : ℝ, f a = g b ∧ b = -a ∧ g b = -f a :=
by
  intro a
  use -a
  sorry  -- Skipping the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_origin_l748_74841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribing_sphere_area_l748_74884

/-- A right triangular pyramid with pairwise perpendicular edges -/
structure RightTriangularPyramid where
  /-- The areas of the three side faces -/
  face_area1 : ℝ
  face_area2 : ℝ
  face_area3 : ℝ

/-- The surface area of a sphere -/
noncomputable def sphere_surface_area (radius : ℝ) : ℝ := 4 * Real.pi * radius^2

/-- Theorem: The surface area of the circumscribing sphere of a right triangular pyramid
    with side face areas 4, 6, and 12 is 56π -/
theorem circumscribing_sphere_area (pyramid : RightTriangularPyramid)
  (h1 : pyramid.face_area1 = 4)
  (h2 : pyramid.face_area2 = 6)
  (h3 : pyramid.face_area3 = 12) :
  ∃ (radius : ℝ), sphere_surface_area radius = 56 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribing_sphere_area_l748_74884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operations_l748_74875

def a : ℝ × ℝ × ℝ := (3, 5, -4)
def b : ℝ × ℝ × ℝ := (2, 1, 8)

theorem vector_operations :
  (∃ (x y z : ℝ), 3 • a - 2 • b = (x, y, z) ∧ x = 5 ∧ y = 13 ∧ z = -28) ∧
  (a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2 = -21) ∧
  (∀ (l m : ℝ), (l • a + m • b).2.2 = 0 ↔ -4 * l + 8 * m = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operations_l748_74875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_canada_l748_74898

/-- Represents a book purchase in a specific currency -/
structure BookPurchase where
  quantity : ℕ
  total_price : ℚ

/-- Calculates the average price per book given a list of book purchases -/
noncomputable def average_price (purchases : List BookPurchase) : ℚ :=
  let total_quantity := purchases.map (λ p => p.quantity) |>.sum
  let total_price := purchases.map (λ p => p.total_price) |>.sum
  total_price / total_quantity

theorem average_price_canada : 
  let canada_purchases := [
    ⟨28, 980⟩,
    ⟨45, 600⟩
  ]
  average_price canada_purchases = 1580 / 73 := by
  sorry

#eval (1580 : ℚ) / 73

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_canada_l748_74898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_max_l748_74899

/-- Regular triangular pyramid -/
structure RegularTriangularPyramid where
  a : ℝ  -- Length of edges meeting at the apex
  α : ℝ  -- Angle between edges meeting at the apex

/-- Volume of the regular triangular pyramid -/
noncomputable def volume (p : RegularTriangularPyramid) : ℝ :=
  (1/18) * p.a^3 * (Real.sin (p.α/2))^2 * Real.sqrt (3 - 4 * (Real.sin (p.α/2))^2)

/-- Theorem: The volume of a regular triangular pyramid is maximized when α = 90° -/
theorem volume_max (p : RegularTriangularPyramid) :
  ∀ α', 0 < α' → α' < Real.pi → volume {a := p.a, α := Real.pi/2} ≥ volume {a := p.a, α := α'} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_max_l748_74899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_event_sum_l748_74820

-- Define the sample space
def Ω : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define events A and B
def A : Finset Nat := {1, 2, 3, 4}
def B : Finset Nat := {1, 2, 3, 5}

-- Define event C as a function of m and n
def C (m n : Nat) : Finset Nat := {1, m, n, 8}

-- Define probability measure
noncomputable def P (E : Finset Nat) : ℚ := (E.card : ℚ) / (Ω.card : ℚ)

-- Define the theorem
theorem event_sum (m n : Nat) : 
  (m ∈ Ω ∧ n ∈ Ω) → 
  (m ≠ n) →
  (P (A ∩ B ∩ C m n) = P A * P B * P (C m n)) →
  (P (A ∩ B) ≠ P A * P B) →
  (P (A ∩ C m n) ≠ P A * P (C m n)) →
  (P (B ∩ C m n) ≠ P B * P (C m n)) →
  m + n = 13 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_event_sum_l748_74820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_triangle_relation_l748_74873

noncomputable def f (x : ℝ) := 2 * Real.cos (2 * x + 2 * Real.pi / 3) + Real.sqrt 3 * Real.sin (2 * x)

theorem function_properties_and_triangle_relation :
  ∃ (A B C : ℝ),
    0 < A ∧ 0 < B ∧ 0 < C ∧
    A + B + C = Real.pi ∧
    f (C / 2) = -1/2 ∧
    (∀ x, f (x + Real.pi) = f x) ∧
    (∀ x, f x ≤ 1) ∧
    (∃ x, f x = 1) ∧
    Real.sin A = (3 * Real.sqrt 21) / 14 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_triangle_relation_l748_74873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_is_decreasing_in_0_2_l748_74814

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 + 4) / x

-- Theorem for odd function property
theorem f_is_odd : ∀ x : ℝ, x ≠ 0 → f (-x) = -f x := by
  intro x hx
  sorry

-- Theorem for decreasing function property in (0,2)
theorem f_is_decreasing_in_0_2 :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 2 → f x₁ > f x₂ := by
  intro x₁ x₂ h₁ h₂ h₃
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_is_decreasing_in_0_2_l748_74814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2005_is_cos_l748_74876

/-- The sequence of functions defined by repeated differentiation of sin x -/
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => Real.sin
  | n + 1 => deriv (f n)

/-- Theorem: The 2005th function in the sequence is cosine -/
theorem f_2005_is_cos : f 2005 = Real.cos := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2005_is_cos_l748_74876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_zero_l748_74836

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a parallelogram -/
noncomputable def area (p : Parallelogram) : ℝ :=
  abs ((p.C.x - p.B.x) * (p.A.y - p.B.y))

/-- Calculates the area of a parallelogram below the x-axis -/
noncomputable def areaBelowXAxis (p : Parallelogram) : ℝ :=
  sorry -- Implementation details omitted

/-- Calculates the probability of selecting a point not above the x-axis -/
noncomputable def probabilityNotAboveXAxis (p : Parallelogram) : ℝ :=
  (areaBelowXAxis p) / (area p)

/-- Theorem: The probability of selecting a point not above the x-axis
    in the given parallelogram is 0 -/
theorem probability_zero (p : Parallelogram)
  (h1 : p.A = ⟨1, 6⟩) (h2 : p.B = ⟨3, 0⟩) (h3 : p.C = ⟨5, 0⟩) (h4 : p.D = ⟨3, 6⟩) :
  probabilityNotAboveXAxis p = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_zero_l748_74836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_select_sets_not_unions_l748_74892

theorem select_sets_not_unions (k : ℕ) (S : Finset (Finset ℕ)) 
  (h₁ : k > 1993006) (h₂ : S.card > k) : 
  ∃ (T : Finset (Finset ℕ)), T ⊆ S ∧ T.card = 1996 ∧ 
  ∀ A B C, A ∈ T → B ∈ T → C ∈ T → A ∪ B = C → A = C ∨ B = C := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_select_sets_not_unions_l748_74892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l748_74856

def A : Set ℝ := {x : ℝ | x^2 + x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m*x + 1 = 0}

theorem sufficient_not_necessary_condition :
  ∃ (m : ℝ), (B m ⊂ A) ∧ (m ≠ -(1/2) → B m ⊂ A) := by
  sorry

#check sufficient_not_necessary_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l748_74856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weaver_production_l748_74895

/-- Represents the production rate of a weaver type -/
structure ProductionRate where
  weavers : ℕ
  items : ℕ
  days : ℕ

/-- Calculates the daily production rate per weaver -/
def dailyRate (rate : ProductionRate) : ℚ :=
  (rate.items : ℚ) / (rate.weavers * rate.days)

/-- Calculates the total production for a given number of weavers and days -/
def totalProduction (rate : ProductionRate) (newWeavers : ℕ) (newDays : ℕ) : ℕ :=
  (((newWeavers : ℚ) * (newDays : ℚ) * dailyRate rate).floor : ℤ).toNat

theorem weaver_production
  (matRate : ProductionRate)
  (carpetRate : ProductionRate)
  (rugRate : ProductionRate)
  (curtainRate : ProductionRate)
  (h1 : matRate = ⟨4, 4, 4⟩)
  (h2 : carpetRate = ⟨3, 2, 3⟩)
  (h3 : rugRate = ⟨2, 3, 5⟩)
  (h4 : curtainRate = ⟨1, 1, 6⟩) :
  totalProduction matRate 5 8 = 10 ∧
  totalProduction carpetRate 6 8 = 10 ∧
  totalProduction rugRate 4 8 = 9 ∧
  totalProduction curtainRate 2 8 = 2 := by
  sorry

#eval totalProduction ⟨4, 4, 4⟩ 5 8
#eval totalProduction ⟨3, 2, 3⟩ 6 8
#eval totalProduction ⟨2, 3, 5⟩ 4 8
#eval totalProduction ⟨1, 1, 6⟩ 2 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weaver_production_l748_74895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_sum_equals_half_power_l748_74871

/-- Given a polynomial equation, prove that a specific sum of its coefficients equals (1/2)^2018 -/
theorem coefficient_sum_equals_half_power (a : Fin 2019 → ℝ) :
  (∀ x : ℝ, (x + 1)^2 * (x + 2)^2016 = 
    (Finset.range 2019).sum (λ i ↦ a i * (x + 2)^i)) →
  (Finset.range 2018).sum (λ i ↦ a (i + 1) / (2^(i + 1))) = (1 / 2)^2018 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_sum_equals_half_power_l748_74871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_fraction_evaluation_l748_74805

theorem ceiling_fraction_evaluation :
  (⌈(19 : ℚ) / 5 - ⌈(23 : ℚ) / 19⌉⌉ : ℚ) / (⌈(23 : ℚ) / 5 + ⌈(5 * 19 : ℚ) / 23⌉⌉ : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_fraction_evaluation_l748_74805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extrema_sum_positive_implies_a_range_l748_74870

/-- Given a > 0 and f(x) = ln(1 + ax) - (2x / (x + 2)) defined for x > 0,
    if f(x) has two extrema x₁ and x₂ such that f(x₁) + f(x₂) > 0,
    then 1/2 < a < 1. -/
theorem extrema_sum_positive_implies_a_range (a : ℝ) (h_a : a > 0) :
  let f := fun (x : ℝ) => Real.log (1 + a * x) - (2 * x / (x + 2))
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧
    (∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧ ∀ (x : ℝ), 0 < |x - x₁| ∧ |x - x₁| < δ → f x < f x₁ + ε) ∧
    (∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧ ∀ (x : ℝ), 0 < |x - x₂| ∧ |x - x₂| < δ → f x < f x₂ + ε) ∧
    f x₁ + f x₂ > 0 →
  1/2 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extrema_sum_positive_implies_a_range_l748_74870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_shiny_after_fifth_probability_l748_74882

/-- The number of shiny pennies in the box -/
def shiny_pennies : ℕ := 4

/-- The number of dull pennies in the box -/
def dull_pennies : ℕ := 5

/-- The total number of pennies in the box -/
def total_pennies : ℕ := shiny_pennies + dull_pennies

/-- The probability of drawing the fourth shiny penny after the fifth draw -/
def prob_fourth_shiny_after_fifth : ℚ := 20 / 21

theorem fourth_shiny_after_fifth_probability :
  (Nat.choose 5 3 * Nat.choose 4 1 + Nat.choose 5 2 * Nat.choose 4 2 + Nat.choose 5 1 * Nat.choose 4 3 : ℚ) / Nat.choose total_pennies shiny_pennies = prob_fourth_shiny_after_fifth := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_shiny_after_fifth_probability_l748_74882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l748_74818

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) :
  (Real.sin t.A - Real.sin t.C * (Real.cos t.B + Real.sqrt 3 / 3 * Real.sin t.B) = 0) →
  (t.C = π / 3) ∧
  (t.c = 2 ∧ (1 / 2 * t.a * t.b * Real.sin t.C = Real.sqrt 3) → t.a = 2 ∧ t.b = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l748_74818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_16_minus_x_squared_plus_sin_l748_74852

open Real MeasureTheory

/-- The definite integral of (√(16-x^2) + sin(x)) from -2 to 2 is equal to 4√3 + 8π/3 -/
theorem integral_sqrt_16_minus_x_squared_plus_sin : 
  ∫ x in (-2)..2, (Real.sqrt (16 - x^2) + Real.sin x) = 4 * Real.sqrt 3 + 8 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_16_minus_x_squared_plus_sin_l748_74852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_undefined_at_one_l748_74888

noncomputable section

-- Define the function g
def g (x : ℝ) : ℝ := (x - 5) / (x - 6)

-- State the theorem
theorem inverse_g_undefined_at_one :
  ∀ x : ℝ, x ≠ 6 → (∃ y : ℝ, g y = x) → x ≠ 1 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_undefined_at_one_l748_74888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_B_eq_five_l748_74847

def A : Finset ℕ := {0, 1, 2}

def B : Finset ℤ := Finset.image (λ (p : ℕ × ℕ) => p.1 - p.2) (A.product A)

theorem card_B_eq_five : Finset.card B = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_B_eq_five_l748_74847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_cubic_meters_to_cubic_feet_l748_74808

/-- Conversion factor from meters to feet -/
def meters_to_feet : ℝ := 3.28084

/-- Theorem stating that 2 cubic meters is approximately equal to 70.6294 cubic feet -/
theorem two_cubic_meters_to_cubic_feet : 
  abs (2 * (meters_to_feet ^ 3) - 70.6294) < 0.0001 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_cubic_meters_to_cubic_feet_l748_74808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_condition_l748_74845

/-- Given parameters for triangle construction -/
structure TriangleParams where
  a : ℝ  -- Length of one side
  AM : ℝ  -- Length of angle bisector
  lambda : ℝ  -- Ratio of other two sides
  a_pos : 0 < a
  AM_pos : 0 < AM
  lambda_pos : 0 < lambda

/-- Theorem stating the condition for triangle construction -/
theorem triangle_construction_condition (params : TriangleParams) :
  ∃ (b c : ℝ), 
    b > 0 ∧ c > 0 ∧
    b / c = params.lambda ∧
    params.a = b + c ∧
    params.AM = 2 * b * c / (b + c) * 
      Real.sqrt (1 - (params.a / (b + c))^2 / 4) ↔
  params.AM < (2 * params.lambda / (1 - params.lambda^2)) * params.a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_condition_l748_74845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_terms_in_binomial_expansion_l748_74840

theorem odd_terms_in_binomial_expansion (m n : ℤ) (hm : Odd m) (hn : Odd n) :
  (Finset.filter (fun k => Odd ((Nat.choose 8 k : ℤ) * m^(8 - k) * n^k)) (Finset.range 9)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_terms_in_binomial_expansion_l748_74840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l748_74849

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x - 2/x - Real.log x

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f x - 4*x

-- State the theorem
theorem f_and_g_properties :
  ∀ x : ℝ, x > 0 →
  (((1 + 2/x^2 - 1/x) < 2 ↔ x > 1) ∧
   (∀ y : ℝ, 0 < y → y < 2/3 → (deriv g y) > 0) ∧
   (∀ z : ℝ, z > 2/3 → (deriv g z) < 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l748_74849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_selection_theorem_l748_74853

theorem athlete_selection_theorem (male_count : Nat) (female_count : Nat) 
  (h1 : male_count = 4) (h2 : female_count = 3) : 
  (Nat.choose (male_count + female_count) 3) - 
  (Nat.choose male_count 3) - 
  (Nat.choose female_count 3) = 30 := by
  sorry

#check athlete_selection_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_selection_theorem_l748_74853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l748_74872

/-- The time it takes for two trains moving in opposite directions to completely pass each other -/
noncomputable def passing_time (speed1 speed2 length1 length2 : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let total_length := length1 + length2
  total_length / (relative_speed * 1000 / 3600)

/-- Theorem stating that the passing time for the given conditions is approximately 4.91 seconds -/
theorem train_passing_time :
  let speed1 := (78 : ℝ)
  let speed2 := (65 : ℝ)
  let length1 := (110 : ℝ)
  let length2 := (85 : ℝ)
  ∃ ε > 0, |passing_time speed1 speed2 length1 length2 - 4.91| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l748_74872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bounds_l748_74819

-- Define the parametric equations for C₁
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)

-- Define the parametric equations for C₂
noncomputable def C₂ (t : ℝ) : ℝ × ℝ := (-3 + t, (3 + 3*t) / 4)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |6 * x - 4 * y + 12| / 5

-- Theorem statement
theorem distance_bounds :
  let max_distance := (12 + 2 * Real.sqrt 13) / 5
  let min_distance := (12 - 2 * Real.sqrt 13) / 5
  ∀ θ : ℝ, 
    min_distance ≤ distance_to_line (C₁ θ).1 (C₁ θ).2 ∧ 
    distance_to_line (C₁ θ).1 (C₁ θ).2 ≤ max_distance ∧
    (∃ θ₁ θ₂ : ℝ, 
      distance_to_line (C₁ θ₁).1 (C₁ θ₁).2 = min_distance ∧
      distance_to_line (C₁ θ₂).1 (C₁ θ₂).2 = max_distance) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bounds_l748_74819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_case_l748_74817

/-- The length of the chord cut by a line on a circle -/
noncomputable def chord_length (a b c : ℝ) (p q r s t : ℝ) : ℝ :=
  let line := fun (x y : ℝ) ↦ a * x + b * y + c
  let circle := fun (x y : ℝ) ↦ x^2 + y^2 + p * x + q * y + r
  let center_x := -p / 2
  let center_y := -q / 2
  let radius := Real.sqrt ((p^2 + q^2) / 4 - r)
  let d := (abs (line center_x center_y)) / Real.sqrt (a^2 + b^2)
  2 * Real.sqrt (radius^2 - d^2)

/-- The theorem stating the length of the chord -/
theorem chord_length_specific_case :
  chord_length 3 4 (-1) (-2) (-4) (-4) 0 0 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_specific_case_l748_74817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_approximation_l748_74879

theorem fraction_approximation : ∃ (x : ℝ), 
  x = (10^10 : ℝ) / (2 * 10^5 * 3) ∧ 
  Int.floor x = 16667 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_approximation_l748_74879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_repeating_decimals_l748_74833

theorem sum_of_repeating_decimals : 
  (3 : ℚ) / 9 + (7 : ℚ) / 99 + (8 : ℚ) / 999 = (418 : ℚ) / 999 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_repeating_decimals_l748_74833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_selection_l748_74835

def is_valid_selection (S : Finset ℕ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x ≠ 7 * y ∧ y ≠ 7 * x

def max_selection : ℕ :=
  Finset.card (Finset.filter (λ n => n ≤ 2014) (Finset.range 2015))

theorem max_valid_selection :
  ∃ (S : Finset ℕ), S.card = 1763 ∧ 
    (∀ n, n ∈ S → n ≤ 2014) ∧
    is_valid_selection S ∧
    (∀ T : Finset ℕ, (∀ n, n ∈ T → n ≤ 2014) → is_valid_selection T → T.card ≤ 1763) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_selection_l748_74835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f₁_in_M_f₂_not_in_M_l748_74824

-- Define the set M
def M : Set (ℝ → ℝ) :=
  {f | ∀ x₁ x₂ : ℝ, x₁ ∈ [-1, 1] → x₂ ∈ [-1, 1] → 
    |f x₁ - f x₂| ≤ 4 * |x₁ - x₂|}

-- Define the functions f₁ and f₂
def f₁ : ℝ → ℝ := λ x ↦ x^2 - 2*x + 5

noncomputable def f₂ : ℝ → ℝ := λ x ↦ Real.sqrt (abs x)

-- Theorem statement
theorem f₁_in_M_f₂_not_in_M : f₁ ∈ M ∧ f₂ ∉ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f₁_in_M_f₂_not_in_M_l748_74824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_denominator_of_sum_l748_74822

-- Define the denominators of the two irreducible fractions
def d1 : ℕ := 600
def d2 : ℕ := 700

-- Define a function to represent an irreducible fraction
def IrreducibleFraction (n d : ℕ) : Prop := Nat.Coprime n d

-- Define the sum of two fractions
def FractionSum (n1 d1 n2 d2 : ℕ) : ℚ := (n1 : ℚ) / d1 + (n2 : ℚ) / d2

-- Theorem statement
theorem smallest_denominator_of_sum :
  ∀ (n1 n2 : ℕ), IrreducibleFraction n1 d1 → IrreducibleFraction n2 d2 →
  ∃ (n : ℕ), FractionSum n1 d1 n2 d2 = (n : ℚ) / 168 ∧
  (∀ (m k : ℕ), FractionSum n1 d1 n2 d2 = (m : ℚ) / k → k ≥ 168) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_denominator_of_sum_l748_74822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_equals_10002_l748_74838

def b : ℕ → ℕ
  | 0 => 3  -- Adding the base case for 0
  | n + 1 => b n + 2 * n + 1

theorem b_100_equals_10002 : b 100 = 10002 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_equals_10002_l748_74838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_fuse_length_is_1_1_l748_74860

/-- The minimum fuse length to ensure safety in a blasting operation -/
noncomputable def minimum_fuse_length (safe_zone : ℝ) (fuse_speed : ℝ) (run_speed : ℝ) : ℝ :=
  safe_zone * fuse_speed / run_speed

/-- Theorem: The minimum fuse length for the given conditions is 1.1 meters -/
theorem minimum_fuse_length_is_1_1 :
  let safe_zone := (70 : ℝ)
  let fuse_speed := (0.112 : ℝ)
  let run_speed := (7 : ℝ)
  abs (minimum_fuse_length safe_zone fuse_speed run_speed - 1.1) < 0.05
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_fuse_length_is_1_1_l748_74860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l748_74828

/-- A hyperbola with center at the origin, focus on the x-axis, and asymptote equation 4 ± 3y = 0 -/
structure Hyperbola where
  /-- The ratio of y to x in the asymptote equation -/
  asymptote_ratio : ℚ
  /-- The asymptote equation is y = ± (asymptote_ratio * x) -/
  asymptote_eq : asymptote_ratio = 4 / 3

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.asymptote_ratio ^ 2)

/-- Theorem stating that the eccentricity of the given hyperbola is 5/3 -/
theorem hyperbola_eccentricity (h : Hyperbola) : eccentricity h = 5 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l748_74828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matroskin_winning_strategy_exists_fyodor_winning_strategy_exists_uncle_fyodor_cannot_always_win_l748_74802

/-- Represents the state of a sandwich (with or without sausage) -/
inductive SandwichState
| WithSausage
| WithoutSausage

/-- Represents a player in the game -/
inductive Player
| UncleFyodor
| Matroskin

/-- Represents the game state -/
structure GameState where
  sandwiches : List SandwichState
  currentPlayer : Player

/-- Represents a move in the game -/
inductive Move
| EatSandwich
| RemoveSausage (index : Nat)

/-- Defines the rules of the game -/
def nextState (state : GameState) (move : Move) : GameState :=
  sorry

/-- Determines if the game is over -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Determines the winner of the game -/
def winner (state : GameState) : Option Player :=
  sorry

/-- Represents a strategy for a player -/
def Strategy := GameState → Move

/-- Plays the game with given strategies -/
def play (N : ℕ) (matroskin_strategy : Strategy) (fyodor_strategy : Strategy) : GameState :=
  sorry

/-- Theorem: There exists a natural number N such that Cat Matroskin has a winning strategy -/
theorem matroskin_winning_strategy_exists :
  ∃ (N : ℕ), ∃ (strategy : Strategy),
    ∀ (fyodor_strategy : Strategy),
      winner (play N strategy fyodor_strategy) = some Player.Matroskin :=
by
  sorry

/-- Theorem: There exists a natural number N such that Uncle Fyodor has a winning strategy -/
theorem fyodor_winning_strategy_exists :
  ∃ (N : ℕ), ∃ (strategy : Strategy),
    ∀ (matroskin_strategy : Strategy),
      winner (play N matroskin_strategy strategy) = some Player.UncleFyodor :=
by
  sorry

/-- Theorem: Uncle Fyodor cannot always win for all natural numbers N -/
theorem uncle_fyodor_cannot_always_win :
  ¬ (∀ (N : ℕ), ∃ (strategy : Strategy),
    ∀ (matroskin_strategy : Strategy),
      winner (play N matroskin_strategy strategy) = some Player.UncleFyodor) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matroskin_winning_strategy_exists_fyodor_winning_strategy_exists_uncle_fyodor_cannot_always_win_l748_74802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l748_74815

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.cos (ω * x + φ) + 1

theorem phi_range (ω φ : ℝ) : 
  ω > 0 ∧ 
  abs φ < Real.pi / 2 ∧ 
  (∃ (k : ℝ), ∀ (x : ℝ), f ω φ x = 3 → f ω φ (x + 2 * Real.pi / 3) = 3) ∧
  (∀ (x : ℝ), -Real.pi / 12 < x ∧ x < Real.pi / 6 → f ω φ x > 1) →
  -Real.pi / 4 ≤ φ ∧ φ ≤ 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l748_74815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_line_average_output_l748_74813

/-- Represents an order with its quantity and production rate -/
structure Order where
  quantity : ℕ
  rate : ℕ
  inv_rate : ℚ

/-- Calculates the time taken to complete an order -/
def time_taken (order : Order) : ℚ :=
  (order.quantity : ℚ) * order.inv_rate

/-- Calculates the overall average output given a list of orders -/
def average_output (orders : List Order) : ℚ :=
  let total_cogs := orders.foldl (fun acc o => acc + (o.quantity : ℚ)) 0
  let total_time := orders.foldl (fun acc o => acc + time_taken o) 0
  total_cogs / total_time

theorem assembly_line_average_output :
  let orders := [
    { quantity := 60, rate := 20, inv_rate := 1 / 20 },
    { quantity := 60, rate := 60, inv_rate := 1 / 60 },
    { quantity := 80, rate := 40, inv_rate := 1 / 40 },
    { quantity := 50, rate := 70, inv_rate := 1 / 70 }
  ]
  average_output orders = 250 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_line_average_output_l748_74813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l748_74827

/-- The hyperbola C -/
noncomputable def hyperbola (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

/-- The square root function -/
noncomputable def sqrt_func (x y : ℝ) : Prop :=
  y = Real.sqrt x

/-- The left focus of the hyperbola -/
def left_focus : ℝ × ℝ := (-1, 0)

/-- The point of intersection P -/
noncomputable def intersection_point (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  hyperbola P.1 P.2 a b ∧ sqrt_func P.1 P.2

/-- The tangent line passing through the left focus and P -/
noncomputable def tangent_line (P : ℝ × ℝ) : Prop :=
  ∃ (m : ℝ), (P.2 - 0) = m * (P.1 - (-1)) ∧
             ∀ (x : ℝ), Real.sqrt x ≤ m * (x - P.1) + P.2

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

/-- The main theorem -/
theorem hyperbola_eccentricity (a b : ℝ) (P : ℝ × ℝ) :
  hyperbola P.1 P.2 a b →
  sqrt_func P.1 P.2 →
  tangent_line P →
  eccentricity a 1 = (Real.sqrt 5 + 1) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l748_74827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l748_74821

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (2*m) + y^2 / (15 - m) = 1 ∧ m > 0 ∧ m < 15

def q (m : ℝ) : Prop := ∃ (x y : ℝ), y^2 / 2 - x^2 / (3*m) = 1 ∧ 2 < m ∧ m < 16/3

-- Define the set of m values that satisfy the conditions
def M : Set ℝ := {m | (p m ∨ q m) ∧ ¬(p m ∧ q m)}

-- State the theorem
theorem range_of_m : M = Set.union (Set.Ioc 0 2) (Set.Ico 5 (16/3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l748_74821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_25_l748_74861

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a trapezoid given its four vertices -/
noncomputable def trapezoidArea (p q r s : Point) : ℝ :=
  let b1 := q.y - p.y
  let b2 := s.y - r.y
  let h := r.x - p.x
  (b1 + b2) * h / 2

/-- Theorem: The area of trapezoid PQRS with given vertices is 25 square units -/
theorem trapezoid_area_is_25 :
  let p : Point := ⟨1, 1⟩
  let q : Point := ⟨1, 4⟩
  let r : Point := ⟨6, 1⟩
  let s : Point := ⟨6, 8⟩
  trapezoidArea p q r s = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_25_l748_74861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_cosine_floor_l748_74893

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def angle_congruent (q : Quadrilateral) : Prop := sorry

def side_lengths (q : Quadrilateral) : Prop :=
  let (ex, ey) := q.E
  let (fx, fy) := q.F
  let (gx, gy) := q.G
  let (hx, hy) := q.H
  ((fx - ex)^2 + (fy - ey)^2) = 200^2 ∧
  ((hx - gx)^2 + (hy - gy)^2) = 200^2

def diagonals_not_equal (q : Quadrilateral) : Prop :=
  let (ex, ey) := q.E
  let (fx, fy) := q.F
  let (gx, gy) := q.G
  let (hx, hy) := q.H
  ((gx - ex)^2 + (gy - ey)^2) ≠ ((hx - fx)^2 + (hy - fy)^2)

noncomputable def perimeter (q : Quadrilateral) : ℝ :=
  let (ex, ey) := q.E
  let (fx, fy) := q.F
  let (gx, gy) := q.G
  let (hx, hy) := q.H
  Real.sqrt ((fx - ex)^2 + (fy - ey)^2) +
  Real.sqrt ((gx - fx)^2 + (gy - fy)^2) +
  Real.sqrt ((hx - gx)^2 + (hy - gy)^2) +
  Real.sqrt ((ex - hx)^2 + (ey - hy)^2)

noncomputable def angle_E (q : Quadrilateral) : ℝ := sorry

-- Define the theorem
theorem quadrilateral_cosine_floor (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_angle : angle_congruent q)
  (h_sides : side_lengths q)
  (h_diag : diagonals_not_equal q)
  (h_perim : perimeter q = 720) :
  ⌊1000 * Real.cos (angle_E q)⌋ = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_cosine_floor_l748_74893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l748_74804

-- Define the hyperbola
def hyperbola (a b : ℝ) : (ℝ → ℝ → Prop) := λ x y ↦ x^2 / a^2 - y^2 / b^2 = 1

-- Define the parabola
def parabola (p : ℝ) : (ℝ → ℝ → Prop) := λ x y ↦ y^2 = 2 * p * x

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Define the asymptote of the hyperbola
def asymptote (a b : ℝ) : (ℝ → ℝ → Prop) := λ x y ↦ y = (b / a) * x ∨ y = -(b / a) * x

-- Define the axis of symmetry of the parabola
def axis_of_symmetry (p : ℝ) : (ℝ → Prop) := λ x ↦ x = -p / 2

-- State the theorem
theorem hyperbola_equation (a b p : ℝ) :
  a > 0 → b > 0 → p > 0 →
  distance (-a) 0 2 0 = 4 →
  asymptote a b (-2) (-1) →
  axis_of_symmetry p (-2) →
  hyperbola 2 1 = hyperbola a b := by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l748_74804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_men_work_hours_l748_74830

/-- The number of men working on the project -/
def num_men : ℕ := 15

/-- The number of days men work -/
def days_men : ℕ := 21

/-- The number of women working on the project -/
def num_women : ℕ := 21

/-- The number of days women work -/
def days_women : ℕ := 36

/-- The number of hours women work per day -/
def hours_women : ℕ := 5

/-- The ratio of women's work rate to men's work rate -/
def work_rate_ratio : ℚ := 2 / 3

/-- The amount of work completed by the project -/
def total_work : ℚ := 1

/-- Calculate the number of hours men work per day -/
def hours_men : ℚ :=
  (num_women * days_women * hours_women * work_rate_ratio) / (num_men * days_men)

theorem men_work_hours : hours_men = 8 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_men_work_hours_l748_74830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_circle_l748_74859

-- Define the complex number z
variable (z : ℂ)

-- Define the condition |z+1-i| = 2
def circle_condition (z : ℂ) : Prop := Complex.abs (z + 1 - Complex.I) = 2

-- Define the function to be maximized
noncomputable def distance_function (z : ℂ) : ℝ := Complex.abs (z - 2 + Complex.I)

-- Theorem statement
theorem max_distance_on_circle :
  (∀ w, circle_condition w → distance_function w ≤ Real.sqrt 13 + 2) ∧
  (∃ w, circle_condition w ∧ distance_function w = Real.sqrt 13 + 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_circle_l748_74859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_roots_quadratic_l748_74829

theorem sin_product_roots_quadratic (α β : ℝ) : 
  α ∈ Set.Ioo 0 π → β ∈ Set.Ioo 0 π → 
  (∃ x y : ℝ, x = Real.cos α ∧ y = Real.cos β ∧ 5 * x^2 - 3 * x - 1 = 0 ∧ 5 * y^2 - 3 * y - 1 = 0) →
  Real.sin α * Real.sin β = Real.sqrt 7 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_product_roots_quadratic_l748_74829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_approx_l748_74809

/-- Represents the round trip of a car between San Diego and San Francisco -/
structure RoundTrip where
  initial_speed : ℝ
  fog_reduction : ℝ
  detour_increase : ℝ
  return_time_factor : ℝ

/-- Calculates the average speed of the entire round trip -/
noncomputable def average_round_trip_speed (trip : RoundTrip) : ℝ :=
  let initial_distance := 1 -- Normalized distance
  let one_way_time := (initial_distance / 2) / trip.initial_speed + 
                      (initial_distance / 2) / (trip.initial_speed * (1 - trip.fog_reduction))
  let total_distance := 2 * initial_distance * (1 + trip.detour_increase)
  let total_time := one_way_time * (1 + trip.return_time_factor)
  total_distance / total_time

/-- Theorem stating that the average speed of the round trip is approximately 32.47 mph -/
theorem round_trip_speed_approx (ε : ℝ) (h : ε > 0) :
  ∃ (trip : RoundTrip), 
    trip.initial_speed = 45 ∧ 
    trip.fog_reduction = 0.2 ∧ 
    trip.detour_increase = 0.15 ∧ 
    trip.return_time_factor = 1 ∧ 
    |average_round_trip_speed trip - 32.47| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_speed_approx_l748_74809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_inequality_l748_74858

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) + x^2 / 2

noncomputable def g (x : ℝ) : ℝ := Real.cos x + x^2 / 2

theorem f_g_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : f (Real.exp (a/2)) = g b - 1) : 
  f (b^2) + 1 > g (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_g_inequality_l748_74858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_10_div_2_l748_74886

/-- The length of the chord formed by the intersection of a line and a circle -/
noncomputable def chordLength (m a : ℝ) (center : ℝ × ℝ) (radius : ℝ) : ℝ :=
  2 * (radius ^ 2 - ((center.2 - (m * center.1 + a)) / (m ^ 2 + 1)) ^ 2) ^ (1/2)

/-- Theorem stating that the length of the chord is √10/2 -/
theorem chord_length_is_sqrt_10_div_2 :
  let line_slope : ℝ := Real.sqrt 3
  let line_intercept : ℝ := Real.sqrt 2 / 2
  let circle_center : ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)
  let circle_radius : ℝ := 1
  chordLength line_slope line_intercept circle_center circle_radius = Real.sqrt 10 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_sqrt_10_div_2_l748_74886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l748_74800

theorem angle_sum_theorem (α β : Real) : 
  α ∈ Set.Ioo 0 (π/2) →
  β ∈ Set.Ioo 0 (π/2) →
  Real.cos (α - β/2) = Real.sqrt 3/2 →
  Real.sin (α/2 - β) = -1/2 →
  α + β = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_theorem_l748_74800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_point_implies_a_eq_two_f_nonnegative_implies_a_in_zero_one_power_fraction_greater_than_e_l748_74846

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) - (a * x) / (x + 1)

-- Theorem 1
theorem critical_point_implies_a_eq_two (a : ℝ) (h : a > 0) :
  (∀ x, x ≠ -1 → (deriv (f a)) x = 0 → x = 1) → a = 2 := by sorry

-- Theorem 2
theorem f_nonnegative_implies_a_in_zero_one (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, x ≥ 0 → f a x ≥ 0) → 0 < a ∧ a ≤ 1 := by sorry

-- Theorem 3
theorem power_fraction_greater_than_e :
  (2017 / 2016 : ℝ) ^ 2017 > Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_point_implies_a_eq_two_f_nonnegative_implies_a_in_zero_one_power_fraction_greater_than_e_l748_74846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marco_beverage_calories_l748_74890

/-- Represents the beverage composition and calorie content --/
structure Beverage where
  lemon_juice : ℚ
  sugar : ℚ
  lime_juice : ℚ
  water : ℚ
  lemon_juice_calories : ℚ
  sugar_calories : ℚ
  lime_juice_calories : ℚ

/-- Calculates the total weight of the beverage --/
def total_weight (b : Beverage) : ℚ :=
  b.lemon_juice + b.sugar + b.lime_juice + b.water

/-- Calculates the total calories in the beverage --/
def total_calories (b : Beverage) : ℚ :=
  b.lemon_juice * b.lemon_juice_calories / 100 +
  b.sugar * b.sugar_calories / 100 +
  b.lime_juice * b.lime_juice_calories / 100

/-- Calculates the calories in a given weight of the beverage --/
def calories_in_weight (b : Beverage) (weight : ℚ) : ℚ :=
  (total_calories b * weight) / total_weight b

/-- Marco's beverage --/
def marco_beverage : Beverage :=
  { lemon_juice := 150
  , sugar := 200
  , lime_juice := 50
  , water := 500
  , lemon_juice_calories := 30
  , sugar_calories := 390
  , lime_juice_calories := 20 }

/-- Theorem: 300g of Marco's beverage contains 278 calories --/
theorem marco_beverage_calories :
  ⌊calories_in_weight marco_beverage 300⌋ = 278 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marco_beverage_calories_l748_74890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_monotone_decreasing_f_l748_74865

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 * x^2 - 8 * a * x + 3
  else a^x - a

-- State the theorem
theorem a_range_for_monotone_decreasing_f (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≥ f a y) →
  1/2 ≤ a ∧ a ≤ 5/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_for_monotone_decreasing_f_l748_74865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_on_axes_l748_74803

/-- A square in the coordinate system -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- Predicate to check if a square intersects both coordinate axes -/
def intersects_axes (s : Square) : Prop :=
  ∃ (x y : ℝ), (x = 0 ∨ y = 0) ∧ 
    (|x - s.center.1| + |y - s.center.2| ≤ s.side_length)

/-- Predicate to check if two squares have no common interior point -/
def no_common_interior (s1 s2 : Square) : Prop :=
  (s1.center.1 - s2.center.1)^2 + (s1.center.2 - s2.center.2)^2 ≥ s1.side_length^2

/-- The main theorem stating the maximum number of squares -/
theorem max_squares_on_axes : 
  ∃ (n : ℕ), n = 5 ∧ 
    (∃ (squares : Finset Square), 
      (∀ s, s ∈ squares → intersects_axes s) ∧
      (∀ s1 s2, s1 ∈ squares → s2 ∈ squares → s1 ≠ s2 → no_common_interior s1 s2) ∧
      squares.card = n) ∧
    (∀ (m : ℕ) (squares : Finset Square), m > n →
      (∀ s, s ∈ squares → intersects_axes s) →
      (∀ s1 s2, s1 ∈ squares → s2 ∈ squares → s1 ≠ s2 → no_common_interior s1 s2) →
      squares.card < m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_squares_on_axes_l748_74803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_term150_eq_2280_l748_74891

/-- The sequence of positive integers that are either powers of 3 or sums of distinct powers of 3 -/
def specialSequence : ℕ → ℕ := sorry

/-- The 150th term of the specialSequence -/
def term150 : ℕ := specialSequence 150

/-- Theorem stating that the 150th term of the specialSequence is 2280 -/
theorem term150_eq_2280 : term150 = 2280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_term150_eq_2280_l748_74891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_A_l748_74837

-- Define points in R²
def A : ℝ × ℝ := (0, 6)
def B : ℝ × ℝ := (0, 15)
def C : ℝ × ℝ := (3, 9)

-- Define the line y = x
def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

-- Define A' and B' on the line y = x
def A' : ℝ × ℝ := (6, 6)
def B' : ℝ × ℝ := (7, 7)

-- Define the property that AA' and BB' intersect at C
def intersect_at_C (p q r : ℝ × ℝ) : Prop :=
  ∃ t s : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ s ∧ s ≤ 1 ∧
  (C.1 = (1 - t) * p.1 + t * q.1) ∧
  (C.2 = (1 - t) * p.2 + t * q.2) ∧
  (C.1 = (1 - s) * p.1 + s * r.1) ∧
  (C.2 = (1 - s) * p.2 + s * r.2)

-- Theorem statement
theorem length_of_A'B'_is_sqrt_2 :
  line_y_eq_x A' ∧ line_y_eq_x B' ∧
  intersect_at_C A A' C ∧ intersect_at_C B B' C →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_A_l748_74837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f_A_is_quadratic_l748_74863

/-- Definition of a quadratic function -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The four functions given in the problem -/
def f_A : ℝ → ℝ := λ x => -9 + x^2
def f_B : ℝ → ℝ := λ x => -2*x + 1
noncomputable def f_C : ℝ → ℝ := λ x => Real.sqrt (x^2 + 4)
def f_D : ℝ → ℝ := λ x => -(x + 1) + 3

/-- Theorem stating that only f_A is quadratic among the given functions -/
theorem only_f_A_is_quadratic :
  is_quadratic f_A ∧ 
  ¬is_quadratic f_B ∧ 
  ¬is_quadratic f_C ∧ 
  ¬is_quadratic f_D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f_A_is_quadratic_l748_74863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_neg_two_eq_227_div_16_l748_74896

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 9

noncomputable def g (x : ℝ) : ℝ := 3 * ((f⁻¹ x) ^ 2) + 4 * (f⁻¹ x) - 2

-- State the theorem
theorem g_of_neg_two_eq_227_div_16 : g (-2) = 227 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_neg_two_eq_227_div_16_l748_74896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_extreme_points_l748_74887

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / x + a * (x - Real.log x)

-- State the theorem
theorem f_monotonicity_and_extreme_points (a : ℝ) :
  (a > 0 →
    (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₁ > f a x₂)) ∧
  (∃ x₁ x₂ x₃, 1/2 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < 2 ∧
    (∃ ε₁ > 0, ∀ y ∈ Set.Ioo (x₁ - ε₁) (x₁ + ε₁), f a y ≤ f a x₁) ∧
    (∃ ε₂ > 0, ∀ y ∈ Set.Ioo (x₂ - ε₂) (x₂ + ε₂), f a y ≤ f a x₂) ∧
    (∃ ε₃ > 0, ∀ y ∈ Set.Ioo (x₃ - ε₃) (x₃ + ε₃), f a y ≤ f a x₃) →
    -2 * Real.sqrt (Real.exp 1) < a ∧ a < -Real.exp 1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_extreme_points_l748_74887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_mpg_equals_highway_mpg_l748_74842

/-- The average mileage per gallon on the highway -/
def highway_mpg : ℚ := 12.2

/-- The maximum distance that can be driven on 24 gallons of gasoline -/
def max_distance : ℚ := 292.8

/-- The number of gallons of gasoline -/
def gallons : ℚ := 24

/-- The average mileage per gallon in the city -/
noncomputable def city_mpg : ℚ := max_distance / gallons

theorem city_mpg_equals_highway_mpg : city_mpg = highway_mpg := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_mpg_equals_highway_mpg_l748_74842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l748_74801

theorem triangle_area_ratio (A B C : ℝ) (hABC : 2 * Real.sin A * Real.cos (B - C) + Real.sin (2 * A) = 2/3) :
  let S₁ := (1/2) * Real.sin A * Real.sin B * Real.sin C * (4 * (1 / (2 * Real.sin A))^2)
  let S₂ := Real.pi * (1 / (2 * Real.sin A))^2
  S₁ / S₂ = 1 / (3 * Real.pi) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l748_74801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l748_74866

-- Define the function f(x) = a^(2x) + 2a^x - 1
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2*x) + 2*a^x - 1

-- Define the interval [-1, 1]
def interval : Set ℝ := { x : ℝ | -1 ≤ x ∧ x ≤ 1 }

-- State the theorem
theorem max_value_theorem (a : ℝ) : 
  (∀ x ∈ interval, f a x ≤ 14) ∧ 
  (∃ x ∈ interval, f a x = 14) ↔ 
  (a = 1/3 ∨ a = 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l748_74866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_when_a_is_one_f_odd_iff_a_is_one_or_neg_one_l748_74857

/-- The function f(x) defined in terms of a constant a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - Real.exp x) / (1 + a * Real.exp x)

/-- Theorem stating that f(x) is odd when a = 1 -/
theorem f_odd_when_a_is_one : 
  ∀ x : ℝ, f 1 (-x) = -(f 1 x) := by
  sorry

/-- Theorem stating that f(x) is odd iff a = 1 or a = -1 -/
theorem f_odd_iff_a_is_one_or_neg_one :
  ∀ a : ℝ, (∀ x : ℝ, f a (-x) = -(f a x)) ↔ (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_when_a_is_one_f_odd_iff_a_is_one_or_neg_one_l748_74857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_sum_l748_74807

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C D E F : ℝ) : ℝ :=
  abs (F - C) / Real.sqrt (A^2 + B^2)

/-- Theorem: For parallel lines 3x + 4y + 5 = 0 and 6x + ay + 30 = 0,
    where a makes the lines parallel and d is their distance, a + d = 10 -/
theorem parallel_lines_sum (a d : ℝ) : 
  (∀ x y, 3*x + 4*y + 5 = 0 ↔ 6*x + a*y + 30 = 0) →
  d = distance_between_parallel_lines 6 a 30 3 4 5 →
  a + d = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_sum_l748_74807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l748_74855

/-- Represents a single digit -/
def SingleDigit : Type := { n : ℕ // n < 10 }

/-- Represents a three-digit number -/
def ThreeDigitNumber : Type := { n : ℕ // 100 ≤ n ∧ n < 1000 }

/-- Represents a four-digit number -/
def FourDigitNumber : Type := { n : ℕ // 1000 ≤ n ∧ n < 10000 }

/-- Check if a number is the units digit of a square -/
def IsUnitsDigitOfSquare (n : SingleDigit) : Prop :=
  ∃ m : ℕ, (m * m) % 10 = n.val

/-- The main theorem -/
theorem unique_solution :
  ∃! (TOR : ThreeDigitNumber) (ROT : FourDigitNumber) (Γ : SingleDigit),
    Γ.val > (TOR.val % 10) ∧
    IsUnitsDigitOfSquare ⟨TOR.val % 10, by sorry⟩ ∧
    TOR.val * (Γ.val * Γ.val) = Γ.val * ROT.val ∧
    TOR.val = 1089 ∧ ROT.val = 9801 ∧ Γ.val = 9 :=
by sorry

#check unique_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l748_74855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_prism_volume_l748_74810

/-- Represents a triangular prism -/
structure TriangularPrism where
  -- We don't need to define the internal structure,
  -- just that it exists as a concept

/-- The volume of a triangular prism -/
noncomputable def volume (prism : TriangularPrism) : ℝ := sorry

/-- The area of a lateral face of a triangular prism -/
noncomputable def lateralFaceArea (prism : TriangularPrism) : ℝ := sorry

/-- The distance from a lateral face to the opposite edge in a triangular prism -/
noncomputable def distanceToOppositeEdge (prism : TriangularPrism) : ℝ := sorry

/-- Theorem: The volume of a triangular prism is equal to half the product of
    the area of a lateral face and the distance from this face to the opposite edge -/
theorem triangular_prism_volume (prism : TriangularPrism) :
  volume prism = (1/2 : ℝ) * lateralFaceArea prism * distanceToOppositeEdge prism := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_prism_volume_l748_74810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_XWY_l748_74862

/-- Represents a triangle with base and height -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- Calculates the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := (1 / 2) * t.base * t.height

/-- The problem statement -/
theorem area_of_XWY (XYZ XWZ : Triangle) (h1 : XYZ.base = 8) (h2 : XYZ.height = 5)
    (h3 : XWZ.base = 4) (h4 : XWZ.height = 2) :
  triangleArea XYZ - triangleArea XWZ = 16 := by
  -- Unfold the definition of triangleArea
  unfold triangleArea
  -- Substitute the known values
  rw [h1, h2, h3, h4]
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_XWY_l748_74862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l748_74878

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.sin x * Real.cos x)

-- Define the theorem
theorem f_monotone_decreasing (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * Real.pi + Real.pi / 4) (k * Real.pi + Real.pi / 2)) :=
by
  sorry -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l748_74878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_weekly_take_home_pay_l748_74831

/-- Calculates James' weekly take-home pay after taxes --/
noncomputable def james_take_home_pay (
  main_job_rate : ℝ)
  (second_job_rate_reduction : ℝ)
  (main_job_hours : ℝ)
  (overtime_hours : ℝ)
  (overtime_rate_multiplier : ℝ)
  (weekend_gig_pay : ℝ)
  (weekend_days : ℝ)
  (federal_tax_rate : ℝ)
  (state_tax_rate : ℝ) : ℝ :=
  let second_job_rate := main_job_rate * (1 - second_job_rate_reduction)
  let second_job_hours := main_job_hours / 2
  let main_job_pay := main_job_rate * main_job_hours + 
                      overtime_hours * main_job_rate * overtime_rate_multiplier
  let second_job_pay := second_job_rate * second_job_hours
  let weekend_pay := weekend_gig_pay * weekend_days
  let total_pay := main_job_pay + second_job_pay + weekend_pay
  let total_tax := total_pay * (federal_tax_rate + state_tax_rate)
  total_pay - total_tax

/-- Theorem stating that James' weekly take-home pay is $916.30 --/
theorem james_weekly_take_home_pay : 
  james_take_home_pay 20 0.2 30 5 1.5 100 2 0.18 0.05 = 916.30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_weekly_take_home_pay_l748_74831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_product_ones_eights_l748_74844

/-- The number of digits in each factor -/
def n : ℕ := 84

/-- The first factor: a number consisting of n ones -/
def ones (n : ℕ) : ℕ := (10^n - 1) / 9

/-- The second factor: a number consisting of n eights -/
def eights (n : ℕ) : ℕ := 8 * ones n

/-- Sum of digits function -/
def sum_of_digits : ℕ → ℕ
| 0 => 0
| n + 1 => (n + 1) % 10 + sum_of_digits (n / 10)

/-- Main theorem -/
theorem sum_of_digits_product_ones_eights :
  sum_of_digits (ones n * eights n) = 8 * n := by
  sorry

#eval sum_of_digits (ones n * eights n)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_product_ones_eights_l748_74844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_f_bounded_l748_74867

/-- Definition of the function f -/
noncomputable def f (a b c x : ℝ) : ℝ := a * (x - b) / ((x - b)^2 + c)

/-- Theorem: f is symmetric about a point on the x-axis -/
theorem f_symmetric (a b c : ℝ) (ha : a ≠ 0) (hc : c > 0) :
  ∃ k : ℝ, ∀ x : ℝ, f a b c (2*k - x) = f a b c x := by
  sorry

/-- Theorem: f is bounded -/
theorem f_bounded (a b c : ℝ) (ha : a ≠ 0) (hc : c > 0) :
  ∃ p q : ℝ, ∀ x : ℝ, p ≤ f a b c x ∧ f a b c x ≤ q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_f_bounded_l748_74867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_condition_l748_74885

theorem fraction_meaningful_condition (x : ℝ) :
  (x + 3) / (x + 4) ∈ Set.univ ↔ x ≠ -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_condition_l748_74885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_2008_equals_3124_l748_74883

/-- The nth term of the sequence v_n -/
def v (n : ℕ) : ℕ := sorry

/-- The number of terms in the kth group -/
def group_size (k : ℕ) : ℕ := 2 * k - 1

/-- The total number of terms up to and including the kth group -/
def total_terms_up_to (k : ℕ) : ℕ := k^2

/-- The first term of the kth group -/
def first_term_of_group (k : ℕ) : ℕ := 
  1 + 3 * (total_terms_up_to (k - 1) / 2)

/-- The main theorem: v_2008 = 3124 -/
theorem v_2008_equals_3124 : v 2008 = 3124 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_2008_equals_3124_l748_74883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_four_girls_l748_74850

/-- The number of children -/
def n : ℕ := 7

/-- The probability of having a girl -/
def p : ℝ := 0.55

/-- The probability of having a boy -/
def q : ℝ := 0.45

/-- The probability of having exactly k girls out of n children -/
def P (k : ℕ) : ℝ := Nat.choose n k * p^k * q^(n-k)

/-- The probability of having at least four girls out of seven children -/
def prob_at_least_four : ℝ := P 4 + P 5 + P 6 + P 7

theorem probability_at_least_four_girls :
  abs (prob_at_least_four - 0.59197745) < 1e-6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_four_girls_l748_74850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_grade_homework_forgotten_l748_74832

theorem sixth_grade_homework_forgotten : ℚ := by
  -- Define the number of students in each group
  let group_a : ℕ := 30
  let group_b : ℕ := 50
  -- Define the percentage of students who forgot homework in each group
  let forgot_a_percent : ℚ := 20 / 100
  let forgot_b_percent : ℚ := 12 / 100
  -- Theorem statement
  have : (((group_a : ℚ) * forgot_a_percent + (group_b : ℚ) * forgot_b_percent) / 
         ((group_a : ℚ) + (group_b : ℚ))) * 100 = 15 := by
    sorry
  exact 15


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_grade_homework_forgotten_l748_74832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l748_74848

-- Define T(r) as a noncomputable function
noncomputable def T (r : ℝ) : ℝ := 18 / (1 - r)

-- State the theorem
theorem geometric_series_sum (b : ℝ) 
  (h1 : -1 < b) (h2 : b < 1) 
  (h3 : T b * T (-b) = 3024) : 
  T b + T (-b) = 337.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l748_74848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_into_perfect_squares_l748_74816

theorem partition_into_perfect_squares (n : ℕ+) :
  ∃ (p : ℕ+), 
    let seq := Finset.range p
    ∃ (partition : Finset (Finset ℕ)),
      (partition.card = n) ∧ 
      (∀ segment ∈ partition, ∃ k : ℕ, (segment.sum id = k^2)) ∧
      (partition.biUnion id = seq) :=
by
  -- We define p as a natural number, not a positive natural number
  let p : ℕ := (3^(n : ℕ) - 1) / 2 + 1
  
  -- We assert the existence of p as a positive natural number
  have h_p_pos : 0 < p := by sorry
  let p_pos : ℕ+ := ⟨p, h_p_pos⟩

  -- We use p_pos in our existence proof
  use p_pos

  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partition_into_perfect_squares_l748_74816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l748_74864

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def frac (x : ℝ) : ℝ := x - (floor x : ℝ)

def equation (x : ℝ) : Prop :=
  (floor x : ℝ) * (frac x) + x = 2 * (frac x) + 6

theorem solution_set :
  ∃ (S : Set ℝ), S = {14/3, 21/4, 6} ∧
  ∀ x : ℝ, x ∈ S ↔ equation x := by
  sorry

#check solution_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l748_74864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_l748_74881

theorem cos_two_theta (θ : ℝ) (h : (2 : ℝ)^(1 - 3 * Real.cos θ) + 1 = (2 : ℝ)^((1/2) - Real.cos θ)) : 
  Real.cos (2 * θ) = -17/18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_l748_74881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_overtakes_at_four_and_half_hours_l748_74897

/-- Represents the chase scenario between a pirate ship and a trading vessel -/
structure ChaseScenario where
  initial_distance : ℝ
  pirate_initial_speed : ℝ
  pirate_reduced_speed : ℝ
  vessel_speed : ℝ
  speed_change_time : ℝ

/-- Calculates the time when the pirate ship overtakes the trading vessel -/
noncomputable def overtake_time (scenario : ChaseScenario) : ℝ :=
  let initial_relative_speed := scenario.pirate_initial_speed - scenario.vessel_speed
  let distance_after_speed_change := 
    scenario.initial_distance + scenario.vessel_speed * scenario.speed_change_time - 
    scenario.pirate_initial_speed * scenario.speed_change_time
  let final_relative_speed := scenario.pirate_reduced_speed - scenario.vessel_speed
  scenario.speed_change_time + distance_after_speed_change / final_relative_speed

/-- Theorem stating that the pirate ship overtakes the trading vessel after 4.5 hours -/
theorem pirate_overtakes_at_four_and_half_hours (scenario : ChaseScenario) :
  scenario.initial_distance = 15 ∧ 
  scenario.pirate_initial_speed = 14 ∧ 
  scenario.pirate_reduced_speed = 12 ∧ 
  scenario.vessel_speed = 10 ∧ 
  scenario.speed_change_time = 3 →
  overtake_time scenario = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_overtakes_at_four_and_half_hours_l748_74897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_range_l748_74854

/-- The circle equation -/
def circle_eq (x y r : ℝ) : Prop := (x - 3)^2 + (y + 5)^2 = r^2

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := 4*x - 3*y = 2

/-- Distance from a point to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |4*x - 3*y - 2| / Real.sqrt (4^2 + (-3)^2)

/-- There are exactly two points on the circle at distance 1 from the line -/
def two_points_at_distance_one (r : ℝ) : Prop :=
  ∃! (p q : ℝ × ℝ), p ≠ q ∧ 
    circle_eq p.1 p.2 r ∧ circle_eq q.1 q.2 r ∧
    distance_to_line p.1 p.2 = 1 ∧ distance_to_line q.1 q.2 = 1

theorem circle_radius_range :
  ∀ r : ℝ, two_points_at_distance_one r → 4 < r ∧ r < 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_range_l748_74854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_is_800_l748_74874

-- Define the initial amount
def initial_amount : ℝ := sorry

-- Define the original interest rate
def original_rate : ℝ := sorry

-- Define the time period in years
def time : ℝ := 3

-- Define the amount after 3 years at the original rate
def amount_original : ℝ := 956

-- Define the amount after 3 years at the increased rate
def amount_increased : ℝ := 1052

-- Define the rate increase
def rate_increase : ℝ := 4

-- Theorem stating that the initial amount is 800
theorem initial_amount_is_800 :
  (initial_amount * (1 + original_rate * time / 100) = amount_original) →
  (initial_amount * (1 + (original_rate + rate_increase) * time / 100) = amount_increased) →
  initial_amount = 800 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_is_800_l748_74874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_stack_height_proof_l748_74825

/-- The height of a stack of four cylindrical pipes with diameter 12 cm,
    arranged in a square-based pyramid with three pipes touching each other
    at the base and the fourth resting on top where the first three touch. -/
noncomputable def pipe_stack_height : ℝ :=
  6 * Real.sqrt 3 + 6

/-- Theorem stating that the height of the pipe stack is 6√3 + 6 cm -/
theorem pipe_stack_height_proof :
  pipe_stack_height = 6 * Real.sqrt 3 + 6 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_stack_height_proof_l748_74825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_equals_three_halves_l748_74812

/-- The sum of an infinite geometric series with first term a and common ratio r, where |r| < 1 -/
noncomputable def geometricSeriesSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- The first term of our geometric series -/
noncomputable def a : ℝ := 1

/-- The common ratio of our geometric series -/
noncomputable def r : ℝ := 1 / 3

theorem geometric_series_sum_equals_three_halves :
  geometricSeriesSum a r = 3 / 2 := by
  -- Unfold the definitions
  unfold geometricSeriesSum a r
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_equals_three_halves_l748_74812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_zeros_sum_l748_74894

noncomputable def Q (x : ℂ) : ℂ := ((x^24 - 1) / (x - 1))^2 - x^23

def is_zero_of_Q (z : ℂ) : Prop :=
  ∃ (r : ℝ) (α : ℝ), 
    z = r * (Complex.cos (2 * Real.pi * α) + Complex.I * Complex.sin (2 * Real.pi * α)) ∧
    r > 0 ∧ 0 < α ∧ α < 1 ∧ Q z = 0

theorem Q_zeros_sum (α₁ α₂ α₃ α₄ α₅ : ℝ) :
  (∃ (z₁ z₂ z₃ z₄ z₅ : ℂ),
    is_zero_of_Q z₁ ∧ is_zero_of_Q z₂ ∧ is_zero_of_Q z₃ ∧ is_zero_of_Q z₄ ∧ is_zero_of_Q z₅ ∧
    α₁ ≤ α₂ ∧ α₂ ≤ α₃ ∧ α₃ ≤ α₄ ∧ α₄ ≤ α₅ ∧
    (∃ (r₁ r₂ r₃ r₄ r₅ : ℝ),
      z₁ = r₁ * (Complex.cos (2 * Real.pi * α₁) + Complex.I * Complex.sin (2 * Real.pi * α₁)) ∧
      z₂ = r₂ * (Complex.cos (2 * Real.pi * α₂) + Complex.I * Complex.sin (2 * Real.pi * α₂)) ∧
      z₃ = r₃ * (Complex.cos (2 * Real.pi * α₃) + Complex.I * Complex.sin (2 * Real.pi * α₃)) ∧
      z₄ = r₄ * (Complex.cos (2 * Real.pi * α₄) + Complex.I * Complex.sin (2 * Real.pi * α₄)) ∧
      z₅ = r₅ * (Complex.cos (2 * Real.pi * α₅) + Complex.I * Complex.sin (2 * Real.pi * α₅)))) →
  α₁ + α₂ + α₃ + α₄ + α₅ = 161 / 575 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_zeros_sum_l748_74894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l748_74834

/-- Hyperbola with focus at (√3, 0) and eccentricity √3 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : a > 0 ∧ b > 0

/-- A chord of the hyperbola with midpoint (1, 2) -/
structure Chord (h : Hyperbola) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  midpoint_cond : (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 2
  on_hyperbola : A.1^2 / h.a^2 - A.2^2 / h.b^2 = 1 ∧
                 B.1^2 / h.a^2 - B.2^2 / h.b^2 = 1

/-- The perpendicular bisector of a chord -/
def perpBisector (h : Hyperbola) (c : Chord h) : ℝ × ℝ → Prop :=
  λ p ↦ (p.1 - 1) * (c.B.1 - c.A.1) + (p.2 - 2) * (c.B.2 - c.A.2) = 0

/-- Main theorem -/
theorem hyperbola_properties (h : Hyperbola) (c : Chord h) :
  (h.a = 1 ∧ h.b^2 = 2) ∧  -- Hyperbola equation
  (∀ x y, y = x + 1 ↔ (x, y) ∈ ({c.A, c.B} : Set (ℝ × ℝ))) ∧  -- Line AB equation
  (∃ C D : ℝ × ℝ,
    C.1^2 - C.2^2 / 2 = 1 ∧
    D.1^2 - D.2^2 / 2 = 1 ∧
    perpBisector h c C ∧
    perpBisector h c D ∧
    ∃ center : ℝ × ℝ, ∃ radius : ℝ, ∀ p ∈ ({c.A, c.B, C, D} : Set (ℝ × ℝ)),
      (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l748_74834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_area_implies_p_eq_four_l748_74806

/-- Parabola represented by the equation y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  pos_p : p > 0

/-- Line represented by the equation 2x - y + 6 = 0 -/
def line (x y : ℝ) : Prop := 2*x - y + 6 = 0

/-- Point on the parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2*C.p*x

/-- Minimum area of triangle formed by a point on the parabola and the line's intersections with axes -/
noncomputable def min_triangle_area (C : Parabola) (M : PointOnParabola C) : ℝ := 15/2

/-- Theorem stating that if the minimum area of the triangle is 15/2, then p = 4 -/
theorem parabola_min_area_implies_p_eq_four (C : Parabola) (M : PointOnParabola C) :
  min_triangle_area C M = 15/2 → C.p = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_min_area_implies_p_eq_four_l748_74806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinates_l748_74880

/-- Predicate to define a midpoint of a segment -/
def is_midpoint (m x y : ℝ × ℝ) : Prop :=
  dist m x = dist m y ∧ ∃ t : ℝ, t ∈ (Set.Icc 0 1) ∧ m = (1 - t) • x + t • y

/-- Given a segment with endpoints A(x₁, y₁) and B(x₂, y₂), and its midpoint M(x₀, y₀),
    prove that x₀ = (x₁ + x₂) / 2 and y₀ = (y₁ + y₂) / 2 -/
theorem midpoint_coordinates (x₀ x₁ x₂ y₀ y₁ y₂ : ℝ) 
  (h_midpoint : is_midpoint (x₀, y₀) ((x₁, y₁) : ℝ × ℝ) ((x₂, y₂) : ℝ × ℝ)) : 
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_coordinates_l748_74880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cucumber_weight_after_evaporation_l748_74869

/-- Calculates the new weight of cucumbers after water evaporation -/
theorem cucumber_weight_after_evaporation 
  (initial_weight : ℝ) 
  (initial_water_percent : ℝ) 
  (final_water_percent : ℝ) : 
  initial_weight = 100 ∧ 
  initial_water_percent = 95 ∧ 
  final_water_percent = 75 → 
  (initial_weight * (1 - initial_water_percent / 100)) / (1 - final_water_percent / 100) = 500 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cucumber_weight_after_evaporation_l748_74869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l748_74823

noncomputable def geometricSequence (a q : ℝ) : ℕ → ℝ := fun n => a * q ^ (n - 1)

noncomputable def geometricSum (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_ratio (S_1 a q : ℝ) (hq : q ≠ 0) :
  S_1 + 3 * (geometricSum a q 3) = 2 * (2 * geometricSum a q 2) →
  q = 1/3 := by
  sorry

#check geometric_sequence_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l748_74823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_equivalence_l748_74851

theorem contrapositive_equivalence (α : Real) :
  (¬(α = π/3) ↔ ¬(Real.sin α = Real.sqrt 3 / 2)) ↔
  ((Real.sin α ≠ Real.sqrt 3 / 2) → (α ≠ π/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_equivalence_l748_74851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_equation_one_real_root_l748_74877

theorem determinant_equation_one_real_root
  (a b c k : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hk : k > 0) :
  ∃! x : ℝ, Matrix.det 
    !![k * x, c, -b;
      -c, k * x, a;
      b, -a, k * x] = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_equation_one_real_root_l748_74877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_S_l748_74811

/-- The function S representing the square of the distance between (x, ln x) and (a, a) -/
noncomputable def S (x a : ℝ) : ℝ := (x - a)^2 + (Real.log x - a)^2

/-- The theorem stating that the minimum value of S is 1/2 -/
theorem min_value_S :
  ∃ (min : ℝ), min = 1/2 ∧ ∀ (x a : ℝ), x > 0 → S x a ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_S_l748_74811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_error_calculation_l748_74868

noncomputable def correct_calculation (x : ℝ) : ℝ := x * (5/3) - 3

noncomputable def incorrect_calculation (x : ℝ) : ℝ := x * (3/5) - 7

noncomputable def percentage_error (correct incorrect : ℝ) : ℝ :=
  (abs (correct - incorrect) / correct) * 100

theorem error_calculation :
  let x : ℝ := 12
  let correct := correct_calculation x
  let incorrect := incorrect_calculation x
  abs (percentage_error correct incorrect - 98.82) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_error_calculation_l748_74868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_f_l748_74843

noncomputable section

-- Define the integral of sin(x) from 0 to π
noncomputable def a : ℝ := ∫ x in (0 : ℝ)..Real.pi, Real.sin x

-- Define the function f(x) = (a√x - 1/√x)^6
noncomputable def f (x : ℝ) : ℝ := (a * Real.sqrt x - 1 / Real.sqrt x) ^ 6

-- Theorem statement
theorem constant_term_of_f : ∃ (c : ℝ), c = -160 ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < x ∧ x < δ → |f x - c| < ε) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_f_l748_74843
