import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_video_cassette_cost_l448_44875

/-- The cost of 7 audio cassettes and 3 video cassettes in rupees -/
def cost1 : ℕ := 1110

/-- The cost of 5 audio cassettes and 4 video cassettes in rupees -/
def cost2 : ℕ := 1350

/-- The cost of one audio cassette in rupees -/
def audio_cost : ℕ → ℕ := sorry

/-- The cost of one video cassette in rupees -/
def video_cost : ℕ → ℕ := sorry

theorem video_cassette_cost :
  ∃ (x : ℕ), (7 * audio_cost x + 3 * video_cost x = cost1) ∧
             (5 * audio_cost x + 4 * video_cost x = cost2) →
             video_cost x = 300 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_video_cassette_cost_l448_44875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l448_44849

-- Problem 1
theorem problem_1 (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) : 
  (x^2 - 2*x + 1) / (x^2 - x) / (1/x - 1) = -1 := by sorry

-- Problem 2
theorem problem_2 : 
  -Real.rpow 27 (1/3 : ℝ) + (3.14 - Real.pi)^(0 : ℝ) + (-1/2 : ℝ)^(-1 : ℝ) = -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l448_44849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_binary_multiple_of_24_l448_44815

/-- A function that checks if a positive integer is composed only of 0s and 1s -/
def isComposedOf0sAnd1s (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The theorem statement -/
theorem smallest_binary_multiple_of_24 :
  ∃ (T : ℕ),
    T > 0 ∧
    isComposedOf0sAnd1s T ∧
    T % 24 = 0 ∧
    (∀ (S : ℕ), S > 0 → isComposedOf0sAnd1s S → S % 24 = 0 → S ≥ T) ∧
    T / 24 = 4625 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_binary_multiple_of_24_l448_44815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l448_44867

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem about the specific triangle ABC -/
theorem triangle_abc_properties (t : Triangle) 
  (h1 : t.a * t.a - 2 * Real.sqrt 5 * t.a + 4 = 0)
  (h2 : t.b * t.b - 2 * Real.sqrt 5 * t.b + 4 = 0)
  (h3 : 2 * Real.cos (t.A + t.B) = 1) :
  t.C = Real.pi * 2 / 3 ∧ t.c = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l448_44867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_495_or_0_l448_44898

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds < 10
  h_tens : tens < 10
  h_ones : ones < 10

/-- Converts a ThreeDigitNumber to a natural number -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Reverses a ThreeDigitNumber -/
def reverse (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.ones
  tens := n.tens
  ones := n.hundreds
  h_hundreds := n.h_ones
  h_tens := n.h_tens
  h_ones := n.h_hundreds

/-- Performs one step of the operation -/
def step (n : ThreeDigitNumber) : ThreeDigitNumber :=
  let forward := n.toNat
  let backward := (reverse n).toNat
  if forward ≥ backward then
    sorry -- Convert (forward - backward) back to ThreeDigitNumber
  else
    sorry -- Convert (backward - forward) back to ThreeDigitNumber

/-- The main theorem -/
theorem eventually_495_or_0 (initial : ThreeDigitNumber) :
  ∃ (k : Nat), (Nat.iterate step k initial).toNat = 495 ∨ (Nat.iterate step k initial).toNat = 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_495_or_0_l448_44898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_implies_domain_l448_44828

noncomputable def f (x : ℝ) : ℝ := 4^x - 3*2^x + 3

theorem range_implies_domain :
  (∀ y ∈ Set.Icc 1 7, ∃ x, f x = y) →
  (∀ x, f x ∈ Set.Icc 1 7 ↔ x ∈ Set.Iic 0 ∪ Set.Icc 1 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_implies_domain_l448_44828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l448_44813

open Set Real

def M : Set ℝ := {x | x^2 < 1}
def N : Set ℝ := {x | Real.rpow 2 x > 1}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l448_44813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_range_l448_44809

/-- Circle O with radius r -/
def Circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

/-- Line segment AB -/
def LineSegmentAB : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p.1 = 4 * (1 - t) ∧ p.2 = 4 * t}

/-- Condition for point M being midpoint of PN -/
def IsMidpoint (p n m : ℝ × ℝ) : Prop :=
  m.1 = (p.1 + n.1) / 2 ∧ m.2 = (p.2 + n.2) / 2

/-- Main theorem -/
theorem circle_intersection_range (r : ℝ) :
  (r > 0) →
  (∀ p ∈ LineSegmentAB,
    ∃ l : Set (ℝ × ℝ),
      (p ∈ l) ∧
      (∃ m n : ℝ × ℝ,
        m ∈ l ∧ n ∈ l ∧
        m ∈ Circle r ∧ n ∈ Circle r ∧
        IsMidpoint p n m)) ↔
  (4/3 ≤ r ∧ r < 2 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_range_l448_44809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_abc_is_45_l448_44856

-- Define the weights of a, b, and c
variable (a b c : ℝ)

-- Define the conditions
noncomputable def average_ab : ℝ := (a + b) / 2
noncomputable def average_bc : ℝ := (b + c) / 2

-- State the theorem
theorem average_abc_is_45 
  (h1 : average_ab a b = 40)
  (h2 : average_bc b c = 43)
  (h3 : b = 31) :
  (a + b + c) / 3 = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_abc_is_45_l448_44856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_property_l448_44865

def complex_translation (w : ℂ) : ℂ → ℂ := fun z ↦ z + w

theorem translation_property :
  ∃ w : ℂ, complex_translation w (1 + 3*I) = 6 + 6*I ∧
           complex_translation w (2 - I) = 7 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_property_l448_44865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l448_44834

theorem triangle_cosine_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.cos A + Real.cos B * Real.cos C ≤ 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l448_44834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l448_44817

noncomputable def triangle_ABC (a b c A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = Real.pi ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C

theorem triangle_problem (a b c A B C : ℝ) 
  (h_triangle : triangle_ABC a b c A B C)
  (h_a : a = 10 * Real.sqrt 3)
  (h_A : A = Real.pi / 3) :
  (b = 10 * Real.sqrt 2 → B = Real.pi / 4 ∧ C = 5 * Real.pi / 12) ∧
  (c = 10 → b = 20 ∧ 1/2 * b * c * Real.sin A = 50 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l448_44817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_bankers_discount_approx_l448_44842

/-- Calculates the banker's discount for a bill -/
noncomputable def bankers_discount (principal : ℝ) (interest_rate : ℝ) (time : ℝ) : ℝ :=
  principal * interest_rate * time / (100 + interest_rate * time)

/-- The combined banker's discount is approximately 623.30 -/
theorem combined_bankers_discount_approx :
  let bd1 := bankers_discount 2560 5 0.5
  let bd2 := bankers_discount 3800 7 0.75
  let bd3 := bankers_discount 4500 8 1
  abs ((bd1 + bd2 + bd3) - 623.30) < 0.01 := by sorry

#check combined_bankers_discount_approx

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_bankers_discount_approx_l448_44842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_equal_two_sets_can_be_disjoint_l448_44840

-- Define the property for the sets
def SetProperty (S₁ S₂ S₃ : Set ℤ) : Prop :=
  ∀ (i j k : Fin 3), i ≠ j → j ≠ k → i ≠ k →
    ∀ (x y : ℤ), x ∈ (match i with
                      | 0 => S₁
                      | 1 => S₂
                      | 2 => S₃) →
                 y ∈ (match i with
                      | 0 => S₁
                      | 1 => S₂
                      | 2 => S₃) →
      x - y ∈ (match k with
               | 0 => S₁
               | 1 => S₂
               | 2 => S₃)

-- Theorem statement
theorem at_least_two_equal (S₁ S₂ S₃ : Set ℤ) 
  (h₁ : S₁.Nonempty) (h₂ : S₂.Nonempty) (h₃ : S₃.Nonempty)
  (h : SetProperty S₁ S₂ S₃) : 
  S₁ = S₂ ∨ S₂ = S₃ ∨ S₁ = S₃ := by
  sorry

-- Additional theorem for the second question
theorem two_sets_can_be_disjoint : 
  ∃ (S₁ S₂ S₃ : Set ℤ), S₁.Nonempty ∧ S₂.Nonempty ∧ S₃.Nonempty ∧ 
    SetProperty S₁ S₂ S₃ ∧ (S₁ ∩ S₂ = ∅ ∨ S₂ ∩ S₃ = ∅ ∨ S₁ ∩ S₃ = ∅) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_two_equal_two_sets_can_be_disjoint_l448_44840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l448_44837

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * Real.log x

-- Define the curve y = ax^3
def curve (a : ℝ) (x : ℝ) : ℝ := a * x^3

-- State the theorem
theorem tangent_line_equation (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1) →
  (λ (x y : ℝ) => 3 * x + y - 2 = 0) = (λ (x y : ℝ) => y - curve a 1 = (deriv (curve a)) 1 * (x - 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l448_44837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lock_combination_a_lock_combination_b_l448_44894

-- Define the type for digits
def Digit := Fin 10

-- Define the lock combination type
structure LockCombination (n : Nat) where
  digits : Fin n → Digit
  different : ∀ i j, i ≠ j → digits i ≠ digits j

-- Define a function to check if a list of digits sums to or multiplies to 14
def isValidTriangle (d1 d2 d3 : Digit) : Prop :=
  (d1.val + d2.val + d3.val = 14) ∨ (d1.val * d2.val * d3.val = 14)

-- Theorem for part a
theorem lock_combination_a :
  ∃ (comb : LockCombination 4),
    let d := comb.digits
    isValidTriangle (d 0) (d 1) (d 2) ∧
    isValidTriangle (d 1) (d 2) (d 3) ∧
    (d 0 = ⟨1, by norm_num⟩ ∧ d 1 = ⟨2, by norm_num⟩ ∧ d 2 = ⟨5, by norm_num⟩ ∧ d 3 = ⟨7, by norm_num⟩) := by
  sorry

-- Theorem for part b
theorem lock_combination_b :
  ∃ (comb : LockCombination 5),
    let d := comb.digits
    isValidTriangle (d 0) (d 1) (d 2) ∧
    isValidTriangle (d 2) (d 3) (d 4) ∧
    (d 0 = ⟨1, by norm_num⟩ ∧ d 1 = ⟨2, by norm_num⟩ ∧ d 2 = ⟨5, by norm_num⟩ ∧ d 3 = ⟨7, by norm_num⟩ ∧ d 4 = ⟨9, by norm_num⟩) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lock_combination_a_lock_combination_b_l448_44894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l448_44851

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- Our specific triangle satisfying the given conditions -/
noncomputable def special_triangle : Triangle where
  A := Real.arcsin (3/5)  -- derived from a = 3 and sin A = 3/5
  B := Real.arcsin (3/5) + Real.pi/2
  C := Real.pi - (Real.arcsin (3/5) + (Real.arcsin (3/5) + Real.pi/2))  -- A + B + C = π
  a := 3
  b := 4
  c := 5  -- derived from a = 3, b = 4, and sin C = 4/5

theorem special_triangle_properties (t : Triangle) 
  (h1 : t.a = 3) 
  (h2 : t.b = 4) 
  (h3 : t.B = t.A + Real.pi/2) : 
  (Real.cos t.B = -3/5) ∧ 
  (Real.sin (2 * t.A) + Real.sin t.C = 31/25) := by
  sorry

#check special_triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l448_44851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_line_parallel_plane_l448_44895

-- Define the types for planes and lines
variable (α β : Type*) -- Planes
variable (m : Type*) -- Line

-- Define the property of a line being in a plane
def line_in_plane (l : Type*) (p : Type*) : Prop := sorry

-- Define parallel relation for planes
def planes_parallel (p1 p2 : Type*) : Prop := sorry

-- Define parallel relation between a line and a plane
def line_parallel_plane (l : Type*) (p : Type*) : Prop := sorry

-- State the theorem
theorem parallel_planes_line_parallel_plane 
  {α β m : Type*}
  (h1 : line_in_plane m α) 
  (h2 : planes_parallel α β) : 
  line_parallel_plane m β := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_line_parallel_plane_l448_44895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cyclic_sin_sum_l448_44847

theorem max_cyclic_sin_sum (θ₁ θ₂ θ₃ θ₄ θ₅ : ℝ) :
  Real.sin (θ₁ - θ₂) + Real.sin (θ₂ - θ₃) + Real.sin (θ₃ - θ₄) + Real.sin (θ₄ - θ₅) + Real.sin (θ₅ - θ₁) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cyclic_sin_sum_l448_44847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_properties_l448_44853

/-- Represents the survey data and calculations -/
structure SurveyData where
  total : ℕ
  satisfiedBoth : ℕ
  satisfiedDedication : ℕ
  satisfiedManagement : ℕ
  chiSquareCritical : ℝ

/-- Calculates the chi-square value for the survey data -/
noncomputable def calculateChiSquare (data : SurveyData) : ℝ :=
  let a := data.satisfiedBoth
  let b := data.satisfiedDedication - a
  let c := data.satisfiedManagement - a
  let d := data.total - a - b - c
  let n := data.total
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Calculates the mathematical expectation for selecting 3 people -/
noncomputable def calculateExpectation (data : SurveyData) : ℝ :=
  (3 : ℝ) * data.satisfiedBoth / data.total

/-- Theorem stating the properties of the survey data -/
theorem survey_properties (data : SurveyData)
  (h1 : data.total = 200)
  (h2 : data.satisfiedBoth = 50)
  (h3 : data.satisfiedDedication = 80)
  (h4 : data.satisfiedManagement = 90)
  (h5 : data.chiSquareCritical = 6.635) :
  calculateChiSquare data > data.chiSquareCritical ∧
  calculateExpectation data = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_properties_l448_44853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_open_interval_one_two_l448_44803

-- Define the properties of function f
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

-- Define the set of x satisfying the inequality
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f x > f (x^2 - 2*x + 2)}

-- State the theorem
theorem solution_is_open_interval_one_two
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_incr : increasing_on f (Set.Ici 0)) :
  solution_set f = Set.Ioo 1 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_open_interval_one_two_l448_44803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_1498_to_1500_form_294_l448_44888

/-- A function that generates the list of positive integers with first digit 2 -/
def integerListWithFirstDigit2 : ℕ → ℕ
| 0 => 2
| n + 1 => 
  let prev := integerListWithFirstDigit2 n
  if prev < 20 then 20
  else if prev < 30 then prev + 1
  else if prev < 300 then prev + 1
  else prev + 1

/-- The digits of a natural number as a list -/
def digits (n : ℕ) : List ℕ :=
  if n < 10 then [n]
  else (digits (n / 10)) ++ [n % 10]

/-- The list of all digits in the sequence up to the nth number -/
def allDigits (n : ℕ) : List ℕ :=
  List.join (List.map digits (List.map integerListWithFirstDigit2 (List.range n)))

theorem digits_1498_to_1500_form_294 :
  (allDigits 1500).drop 1497 = [2, 9, 4] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_1498_to_1500_form_294_l448_44888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_monopoly_prefers_durable_l448_44805

/-- Represents the benefit from using a coffee machine per period -/
def benefit : ℝ := 10

/-- Represents the production cost of a durable coffee machine -/
def durable_cost : ℝ := 6

/-- Represents the production cost of a low-quality coffee machine -/
def C : ℝ → ℝ := λ x => x

/-- Represents the profit from selling a durable coffee machine -/
def durable_profit : ℝ := 2 * benefit - durable_cost

/-- Represents the profit from selling two low-quality coffee machines -/
def low_quality_profit (c : ℝ) : ℝ := 2 * (benefit - C c)

/-- 
Theorem stating that a monopoly will produce only durable coffee machines 
if and only if the production cost of low-quality machines is greater than 3
-/
theorem monopoly_prefers_durable (c : ℝ) : 
  durable_profit > low_quality_profit c ↔ C c > 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_monopoly_prefers_durable_l448_44805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_difference_equals_interval_l448_44873

def M : Set ℝ := {x | x^2 + x - 12 ≤ 0}

def N : Set ℝ := {y | ∃ x ≤ 1, y = 3^x}

theorem set_difference_equals_interval :
  {x : ℝ | x ∈ M ∧ x ∉ N} = Set.Icc (-4) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_difference_equals_interval_l448_44873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l448_44869

noncomputable section

variable (a b : ℝ × ℝ)

def angle_between (v w : ℝ × ℝ) : ℝ := Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def proj (v w : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (v.1 * w.1 + v.2 * w.2) / (w.1^2 + w.2^2)
  (scalar * w.1, scalar * w.2)

theorem projection_theorem (h1 : angle_between a b = π / 3) 
                           (h2 : magnitude a = 2) 
                           (h3 : magnitude b = 2) : 
  proj a b = (0.5 * b.1, 0.5 * b.2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l448_44869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_formula_sin_sum_specific_values_l448_44819

open Real

-- Define the cosine of the difference of two angles
axiom cos_diff (α β : ℝ) : cos (α - β) = cos α * cos β - sin α * sin β

-- Define the range for α
def α_range (α : ℝ) : Prop := π < α ∧ α < 3 * π / 2

-- Define the range for β
def β_range (β : ℝ) : Prop := π / 2 < β ∧ β < π

theorem sin_sum_formula (α β : ℝ) :
  sin (α + β) = sin α * cos β + cos α * sin β := by sorry

theorem sin_sum_specific_values (α β : ℝ) 
  (h1 : cos α = -4/5) (h2 : α_range α) 
  (h3 : tan β = -1/3) (h4 : β_range β) :
  sin (α + β) = sqrt 10 / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_formula_sin_sum_specific_values_l448_44819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_puppy_eats_nine_times_as_often_l448_44846

/-- The number of times a puppy eats compared to a dog -/
def puppy_eating_frequency : ℕ := 9

/-- The number of puppies -/
def num_puppies : ℕ := 4

/-- The number of dogs -/
def num_dogs : ℕ := 3

/-- The amount of food a dog eats in one meal (in pounds) -/
def dog_meal_amount : ℕ := 4

/-- The number of times a dog eats per day -/
def dog_eating_frequency : ℕ := 3

/-- The total amount of food eaten by dogs and puppies in a day (in pounds) -/
def total_food : ℕ := 108

/-- Theorem stating that a puppy eats 9 times as often as a dog -/
theorem puppy_eats_nine_times_as_often :
  puppy_eating_frequency = 9 ∧
  num_puppies * (dog_meal_amount / 2) * puppy_eating_frequency +
  num_dogs * dog_meal_amount * dog_eating_frequency = total_food :=
by
  apply And.intro
  · rfl
  · norm_num
    rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_puppy_eats_nine_times_as_often_l448_44846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_area_value_l448_44820

/-- The area bounded by the curves y = x^2 - 5x + 4 and y = 2x - 2 -/
noncomputable def bounded_area : ℝ :=
  ∫ x in (1:ℝ)..(6:ℝ), (2 * x - 2) - (x^2 - 5 * x + 4)

/-- Theorem stating that the bounded area is equal to 25/6 -/
theorem bounded_area_value : bounded_area = 25 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounded_area_value_l448_44820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_l448_44889

theorem smallest_sum_of_factors (a b : ℕ) (h : (2^10 * 7^3 : ℕ) = a^b) : 
  (∀ (c d : ℕ), (2^10 * 7^3 : ℕ) = c^d → a + b ≤ c + d) ∧ a + b = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_l448_44889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_shots_l448_44877

/-- Represents the number of shots made in a sequence of attempts -/
structure ShotSequence where
  initialShots : ℕ
  initialSuccessRate : ℚ
  additionalShots : ℕ
  finalSuccessRate : ℚ

/-- Calculates the number of successful shots in the additional attempts -/
def additionalSuccessfulShots (s : ShotSequence) : ℕ :=
  (s.finalSuccessRate * (s.initialShots + s.additionalShots)).floor.toNat -
  (s.initialSuccessRate * s.initialShots).floor.toNat

theorem sally_shots (s : ShotSequence)
  (h1 : s.initialShots = 20)
  (h2 : s.initialSuccessRate = 11/20)
  (h3 : s.additionalShots = 5)
  (h4 : s.finalSuccessRate = 14/25) :
  additionalSuccessfulShots s = 3 := by
  sorry

#eval additionalSuccessfulShots { initialShots := 20, initialSuccessRate := 11/20, additionalShots := 5, finalSuccessRate := 14/25 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sally_shots_l448_44877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_melted_spheres_radius_l448_44831

theorem melted_spheres_radius (r : ℝ) : r > 0 → (4 / 3 * Real.pi * r^3 = 2 * (4 / 3 * Real.pi * 1^3)) → r = (2 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_melted_spheres_radius_l448_44831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_angle_cone_l448_44881

/-- The solid angle of a cone with angle 2α at the vertex -/
noncomputable def solidAngleCone (α : Real) : Real :=
  2 * Real.pi * (1 - Real.cos α)

/-- Theorem: The solid angle of a cone with angle 2α at the vertex is 2π(1 - cos α) -/
theorem solid_angle_cone (α : Real) : 
  solidAngleCone α = 2 * Real.pi * (1 - Real.cos α) := by
  -- Unfold the definition of solidAngleCone
  unfold solidAngleCone
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_angle_cone_l448_44881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_B_l448_44866

def B : Set ℕ := {n | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2) ∧ x > 0}

theorem gcd_of_B : ∀ n ∈ B, ∀ m ∈ B, Nat.gcd n m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_B_l448_44866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_negative_four_point_seven_l448_44858

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem floor_of_negative_four_point_seven :
  floor (-4.7) = -5 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_negative_four_point_seven_l448_44858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concept_chain_is_ordered_general_to_specific_l448_44896

-- Define the concepts as inductive types
inductive GeometricFigure
inductive PlaneGeometricFigure : Type
inductive ClosedPlaneGeometricFigure : Type
inductive Polygon : Type
inductive Quadrilateral : Type
inductive Parallelogram : Type
inductive Rhombus : Type
inductive Rectangle : Type
inductive Square : Type
inductive SquareABCD : Type

-- Define the chain of concepts
def conceptChain : List Type := [
  GeometricFigure, PlaneGeometricFigure, ClosedPlaneGeometricFigure,
  Polygon, Quadrilateral, Parallelogram, Rhombus, Rectangle, Square, SquareABCD
]

-- Define a custom relation for "more specific than"
inductive MoreSpecificThan : Type → Type → Prop where
  | geom_plane : MoreSpecificThan PlaneGeometricFigure GeometricFigure
  | plane_closed : MoreSpecificThan ClosedPlaneGeometricFigure PlaneGeometricFigure
  | closed_polygon : MoreSpecificThan Polygon ClosedPlaneGeometricFigure
  | polygon_quad : MoreSpecificThan Quadrilateral Polygon
  | quad_para : MoreSpecificThan Parallelogram Quadrilateral
  | para_rhombus : MoreSpecificThan Rhombus Parallelogram
  | para_rect : MoreSpecificThan Rectangle Parallelogram
  | rhombus_square : MoreSpecificThan Square Rhombus
  | rect_square : MoreSpecificThan Square Rectangle
  | square_abcd : MoreSpecificThan SquareABCD Square

-- Theorem: The chain is correctly ordered from general to specific
theorem concept_chain_is_ordered_general_to_specific :
  ∀ (i j : Fin conceptChain.length), i < j →
  MoreSpecificThan (conceptChain.get j) (conceptChain.get i) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concept_chain_is_ordered_general_to_specific_l448_44896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_calculation_l448_44890

/-- Calculates the speed for the second half of a journey given the total distance,
    total time, and speed for the first half. -/
noncomputable def second_half_speed (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ) : ℝ :=
  let half_distance := total_distance / 2
  let first_half_time := half_distance / first_half_speed
  let second_half_time := total_time - first_half_time
  half_distance / second_half_time

/-- Theorem stating that for a journey of 224 km completed in 10 hours,
    with the first half traveled at 21 km/hr, the speed for the second half
    is approximately 24 km/hr. -/
theorem journey_speed_calculation :
  let total_distance := (224 : ℝ)
  let total_time := (10 : ℝ)
  let first_half_speed := (21 : ℝ)
  let calculated_speed := second_half_speed total_distance total_time first_half_speed
  ∃ ε > 0, abs (calculated_speed - 24) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_calculation_l448_44890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_condition_l448_44814

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero -/
def isPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number (2 + ai) / (1 + i) where a is a real number -/
noncomputable def complexNumber (a : ℝ) : ℂ :=
  (2 + a * Complex.I) / (1 + Complex.I)

theorem purely_imaginary_condition (a : ℝ) :
  isPurelyImaginary (complexNumber a) ↔ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_condition_l448_44814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l448_44839

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 + 2*x else -((-x)^2 + 2*(-x))

theorem a_range (a : ℝ) : 
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x ≥ 0, f x = x^2 + 2*x) →  -- definition for x ≥ 0
  f (2 - a^2) > f a →  -- given inequality
  a > -2 ∧ a < 1 :=  -- conclusion: a ∈ (-2, 1)
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l448_44839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_range_l448_44862

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 + a*x + a + 5) / Real.log 3

-- Define the property of f(x) being decreasing in (-∞, 1)
def is_decreasing_in_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y < 1 → f y < f x

-- Theorem statement
theorem f_decreasing_implies_a_range (a : ℝ) :
  is_decreasing_in_interval (f a) → a ∈ Set.Icc (-3) (-2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_implies_a_range_l448_44862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_meet_time_l448_44860

/-- Represents a cyclist with a given speed in km/h -/
structure Cyclist where
  speed : ℚ
  speed_positive : speed > 0

/-- Calculates the time (in minutes) for a cyclist to complete one lap -/
def lap_time (track_length : ℚ) (c : Cyclist) : ℚ :=
  track_length / (c.speed * 1000 / 60)

/-- Theorem: Four cyclists meet at the starting point after 24 minutes -/
theorem cyclists_meet_time (track_length : ℚ) (a b c d : Cyclist) 
  (h_track : track_length = 800)
  (h_a : a.speed = 32)
  (h_b : b.speed = 48)
  (h_c : c.speed = 36)
  (h_d : d.speed = 60) :
  ∃ (t : ℚ), t = 24 ∧ 
    (∀ (n : ℕ), (n : ℚ) * (lap_time track_length a) = t) ∧
    (∀ (n : ℕ), (n : ℚ) * (lap_time track_length b) = t) ∧
    (∀ (n : ℕ), (n : ℚ) * (lap_time track_length c) = t) ∧
    (∀ (n : ℕ), (n : ℚ) * (lap_time track_length d) = t) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_meet_time_l448_44860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l448_44868

open Real

theorem angle_equality (α β γ θ : ℝ) 
  (h1 : 0 < α ∧ α < π)
  (h2 : 0 < β ∧ β < π)
  (h3 : 0 < γ ∧ γ < π)
  (h4 : 0 < θ ∧ θ < π)
  (h5 : sin α / sin β = sin γ / sin θ)
  (h6 : sin α / sin β = sin (α - γ) / sin (β - θ))
  : α = β ∧ γ = θ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equality_l448_44868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_cheat_approx_29_82_l448_44829

/-- Represents the percentage by which the shop owner cheats while buying -/
noncomputable def buying_cheat_percentage : ℝ := 14

/-- Represents the percentage profit the shop owner makes -/
noncomputable def profit_percentage : ℝ := 42.5

/-- Calculates the percentage by which the shop owner cheats while selling -/
noncomputable def selling_cheat_percentage : ℝ :=
  (100 * profit_percentage) / (100 - buying_cheat_percentage + profit_percentage)

/-- Theorem stating that the selling cheat percentage is approximately 29.82% -/
theorem selling_cheat_approx_29_82 :
  abs (selling_cheat_percentage - 29.82) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_cheat_approx_29_82_l448_44829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l448_44871

structure Ellipse where
  a : ℝ
  b : ℝ
  h : a = 2 ∧ b = 1

def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

def Point := ℝ × ℝ

def Ellipse.on_ellipse (e : Ellipse) (p : Point) : Prop :=
  e.equation p.1 p.2

def perpendicular (o p q : Point) : Prop :=
  (p.1 - o.1) * (q.1 - o.1) + (p.2 - o.2) * (q.2 - o.2) = 0

noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def triangle_area (o p q : Point) : ℝ :=
  abs ((p.1 - o.1) * (q.2 - o.2) - (q.1 - o.1) * (p.2 - o.2)) / 2

theorem ellipse_properties (e : Ellipse) (o p q : Point) 
  (h_o : o = (0, 0))
  (h_p : e.on_ellipse p)
  (h_q : e.on_ellipse q)
  (h_perp : perpendicular o p q) :
  (1 / distance o p^2 + 1 / distance o q^2 = 5/4) ∧
  (4/5 ≤ triangle_area o p q ∧ triangle_area o p q ≤ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l448_44871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l448_44845

-- Define the train length in meters
noncomputable def train_length : ℝ := 150

-- Define the train speed in km/h
noncomputable def train_speed : ℝ := 84.99280057595394

-- Define the man's walking speed in km/h
noncomputable def man_speed : ℝ := 5

-- Define the conversion factor from km/h to m/s
noncomputable def km_h_to_m_s : ℝ := 5 / 18

-- Define the relative speed in m/s
noncomputable def relative_speed : ℝ := (train_speed + man_speed) * km_h_to_m_s

-- Define the time taken for the train to cross the man
noncomputable def crossing_time : ℝ := train_length / relative_speed

-- Theorem statement
theorem train_crossing_time_approx :
  |crossing_time - 6.00024| < 0.00001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_l448_44845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_example_l448_44838

/-- The speed of a car given its travel distance and time -/
noncomputable def car_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Theorem: A car traveling 260 miles in 4 hours has a speed of 65 miles per hour -/
theorem car_speed_example : car_speed 260 4 = 65 := by
  -- Unfold the definition of car_speed
  unfold car_speed
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_example_l448_44838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_and_c_l448_44800

noncomputable def a : ℝ := Real.exp (-4/5)
noncomputable def b : ℝ := 13/25
noncomputable def c : ℝ := 5/9

theorem a_less_than_b_and_c : a < b ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_and_c_l448_44800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_number_is_84_l448_44893

/-- Given two positive integers with specific HCF, LCM, and ratio, prove that the larger number is 84 -/
theorem larger_number_is_84 (x y : ℕ) 
  (hx : x > 0)
  (hy : y > 0)
  (hcf : Nat.gcd x y = 84)
  (lcm : Nat.lcm x y = 21)
  (ratio : 4 * x = y) : 
  y = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_number_is_84_l448_44893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l448_44832

open Real

-- Define the function f
noncomputable def f (α : ℝ) : ℝ :=
  (sin (π/2 + α) * sin (2*π - α)) / (cos (-π - α) * sin (3*π/2 + α))

-- Theorem for the first part
theorem part_one (α : ℝ) 
  (h1 : π < α ∧ α < 3*π/2)  -- α is in the third quadrant
  (h2 : cos (α - 3*π/2) = 1/5) : 
  f α = sqrt 6 / 12 := by sorry

-- Theorem for the second part
theorem part_two (α : ℝ) 
  (h : f α = -2) : 
  2 * sin α * cos α + cos α ^ 2 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l448_44832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_sequences_is_560_l448_44836

/-- Represents a coin flip sequence -/
def CoinFlipSequence := List Bool

/-- Counts the number of occurrences of a specific pattern in a coin flip sequence -/
def countPattern (seq : CoinFlipSequence) (pattern : List Bool) : Nat :=
  sorry

/-- Checks if a coin flip sequence satisfies the given conditions -/
def isValidSequence (seq : CoinFlipSequence) : Bool :=
  seq.length = 15 &&
  countPattern seq [true, true] = 2 &&
  countPattern seq [true, false] = 3 &&
  countPattern seq [false, true] = 4 &&
  countPattern seq [false, false] = 5

/-- Generates all possible coin flip sequences of length 15 -/
def allSequences : List CoinFlipSequence :=
  List.map (fun n => List.map (fun i => n >>> i &&& 1 = 1) (List.range 15)) (List.range (2^15))

/-- The number of valid coin flip sequences -/
def numValidSequences : Nat :=
  (List.filter isValidSequence allSequences).length

theorem num_valid_sequences_is_560 : numValidSequences = 560 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_valid_sequences_is_560_l448_44836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_mono_increasing_intervals_l448_44882

open Real

-- Define the functions f₁, f₂, and f
noncomputable def f₁ (x : ℝ) : ℝ := sin (3 * π / 2 + x) * cos x
noncomputable def f₂ (x : ℝ) : ℝ := sin x * sin (π + x)
noncomputable def f (x : ℝ) : ℝ := f₁ x - f₂ x

-- Define the property of being monotonically increasing on an interval
def MonoIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem f_mono_increasing_intervals :
  ∀ k : ℤ, MonoIncreasingOn f (k * π) (k * π + π / 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_mono_increasing_intervals_l448_44882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_formation_l448_44801

-- Define Quadrilateral as a structure
structure Quadrilateral (α : Type*) :=
(a b c d : α)

theorem quadrilateral_formation (a b c d : ℝ) 
  (positive_a : 0 < a) (positive_b : 0 < b) (positive_c : 0 < c) (positive_d : 0 < d)
  (sum_one : a + b + c + d = 1) : 
  (∃ (quad : Quadrilateral ℝ), 
    quad.a = a ∧ 
    quad.b = b ∧ 
    quad.c = c ∧ 
    quad.d = d) ↔ 
  (a < 1/2 ∧ b < 1/2 ∧ c < 1/2 ∧ d < 1/2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_formation_l448_44801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ray_not_shorter_than_line_l448_44830

-- Define the concept of a geometric line
def Line : Type := sorry

-- Define the concept of a geometric ray
def Ray : Type := sorry

-- Define a point in geometric space
def Point : Type := sorry

-- Define the concept of length for geometric objects
noncomputable def length : Line → ℝ := sorry
noncomputable def ray_length : Ray → ℝ := sorry

-- Define the property of being infinite
def is_infinite (α : Type) : Prop := sorry

-- Axiom: Lines are infinite
axiom line_is_infinite : ∀ (l : Line), is_infinite Line

-- Axiom: Rays are infinite in one direction
axiom ray_is_infinite : ∀ (r : Ray), is_infinite Ray

-- Theorem: It's not possible to compare the length of a ray and a line
theorem ray_not_shorter_than_line : 
  ¬∀ (r : Ray) (l : Line), ray_length r < length l :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ray_not_shorter_than_line_l448_44830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l448_44812

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 8*x - 6*y = 19

-- Define the area of the region
noncomputable def region_area : ℝ := 44 * Real.pi

-- Theorem statement
theorem area_of_region :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (x y : ℝ), region_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l448_44812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_subset_sums_formula_l448_44816

variable {α : Type*} [CommSemiring α]

def sum_of_subset_sums (s : Finset α) : α :=
  (Finset.powerset s).sum (λ t => t.sum id)

theorem sum_of_subset_sums_formula (s : Finset α) :
  sum_of_subset_sums s = 2^(s.card - 1) * s.sum id := by
  sorry

#check sum_of_subset_sums_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_subset_sums_formula_l448_44816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_for_specific_cone_l448_44826

/-- Represents a right circular cone -/
structure RightCircularCone where
  base_radius : ℝ
  slant_height : ℝ

/-- The minimum distance from the vertex to the shortest path on the surface -/
noncomputable def min_distance_to_path (cone : RightCircularCone) : ℝ :=
  (3 * Real.sqrt 3) / 2

/-- Theorem stating the minimum distance for a specific cone -/
theorem min_distance_for_specific_cone :
  let cone : RightCircularCone := ⟨1, 3⟩
  min_distance_to_path cone = (3 * Real.sqrt 3) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_for_specific_cone_l448_44826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l448_44822

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := λ x => Real.log x / Real.log 2
noncomputable def g : ℝ → ℝ := λ x => 2^x

-- Define the composite function h
noncomputable def h : ℝ → ℝ := λ x => f (4*x - x^2)

-- State the theorem
theorem monotonic_increase_interval :
  (∀ x, f (g x) = x ∧ g (f x) = x) →
  ∃ a b, a = 0 ∧ b = 2 ∧
    (∀ x y, a < x ∧ x < y ∧ y < b → h x < h y) ∧
    (∀ ε > 0, ∃ x₁ x₂, x₁ < a ∧ b < x₂ ∧ h x₁ ≥ h (a + ε) ∧ h x₂ ≤ h (b - ε)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l448_44822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_new_earnings_l448_44886

/-- Calculate the new weekly earnings after a percentage raise -/
noncomputable def new_weekly_earnings (initial_earnings : ℝ) (percentage_increase : ℝ) : ℝ :=
  initial_earnings * (1 + percentage_increase / 100)

/-- Theorem: John's new weekly earnings after a 25% raise -/
theorem johns_new_earnings :
  new_weekly_earnings 60 25 = 75 := by
  -- Unfold the definition of new_weekly_earnings
  unfold new_weekly_earnings
  -- Simplify the arithmetic
  simp [mul_add, mul_div_assoc]
  -- Check that the equality holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_new_earnings_l448_44886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_reconstruction_l448_44883

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the angle bisector
def AngleBisector (T : Triangle) (P : Point) : Prop :=
  ∃ (l : ℝ), 0 < l ∧ l < 1 ∧ 
    P = ⟨(1 - l) * T.A.x + l * T.C.x, (1 - l) * T.A.y + l * T.C.y⟩

-- State the theorem
theorem triangle_reconstruction 
  (A B' C' : Point) 
  (h1 : ∃ T : Triangle, AngleBisector T B' ∧ AngleBisector T C') :
  ∃! T : Triangle, T.A = A ∧ AngleBisector T B' ∧ AngleBisector T C' :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_reconstruction_l448_44883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_l448_44887

/-- Given a function f such that f(x-1) = x^2 + 1 for all x,
    prove that f(x) = (x + 1)^2 + 1 for all x. -/
theorem function_composition (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = x^2 + 1) :
  ∀ x, f x = (x + 1)^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_l448_44887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_g_8_times_3_l448_44879

-- Define the functions u and g
noncomputable def u (x : ℝ) : ℝ := Real.sqrt (5 * x + 2)
noncomputable def g (x : ℝ) : ℝ := 7 - u x

-- State the theorem
theorem u_g_8_times_3 : u (g 8) * 3 = 3 * Real.sqrt (37 - 5 * Real.sqrt 42) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_g_8_times_3_l448_44879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_quadrant_angle_property_l448_44848

theorem third_quadrant_angle_property (α : Real) (h : α ∈ Set.Ioo π (3*π/2)) :
  Real.tan α - Real.sin α ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_quadrant_angle_property_l448_44848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l448_44884

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + 1/2 * x^2 + 2*a*x

theorem f_properties :
  (∃ (a : ℝ), ∀ (x y : ℝ), 2/3 < x ∧ x < y → f a x < f a y) ↔ (∃ (a : ℝ), a > -1/9) ∧
  (let a := 1
   ∀ (x : ℝ), x ∈ Set.Icc 1 4 → f a x ≤ f a 2) ∧
  (f 1 2 = 10/3) ∧
  (let a := 1
   ∀ (x : ℝ), x ∈ Set.Icc 1 4 → f a x ≥ f a 4) ∧
  (f 1 4 = -16/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l448_44884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l448_44857

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t/2, 1 - (Real.sqrt 3 / 2) * t)

-- Define the circle C in polar coordinates
noncomputable def circle_C (θ : ℝ) : ℝ := 2 * Real.sin θ

-- State the theorem
theorem line_circle_intersection :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧
  (∃ (θ₁ θ₂ : ℝ), 
    line_l t₁ = (circle_C θ₁ * Real.cos θ₁, circle_C θ₁ * Real.sin θ₁) ∧
    line_l t₂ = (circle_C θ₂ * Real.cos θ₂, circle_C θ₂ * Real.sin θ₂)) ∧
  (∀ (t θ : ℝ), line_l t = (circle_C θ * Real.cos θ, circle_C θ * Real.sin θ) →
    t = t₁ ∨ t = t₂) ∧
  Real.sqrt ((line_l t₁).1 - (line_l t₂).1)^2 + ((line_l t₁).2 - (line_l t₂).2)^2 = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l448_44857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_some_number_value_l448_44872

/-- Given x > 0, if x / n + x / 25 = 0.14000000000000002 * x, then |n - 10| < 0.01 -/
theorem some_number_value (x n : ℝ) (hx : x > 0) 
  (h : x / n + x / 25 = 0.14000000000000002 * x) : 
  |n - 10| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_some_number_value_l448_44872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line1_equation_line2_equation_l448_44802

noncomputable section

-- Define the slope of the original line
def original_slope : ℝ := -Real.sqrt 3

-- Define the slope of the new lines
def new_slope : ℝ := Real.sqrt 3 / 3

-- Define the point that line 1 passes through
def point : ℝ × ℝ := (Real.sqrt 3, -1)

-- Define the y-intercept of line 2
def y_intercept : ℝ := -5

-- Theorem for line 1
theorem line1_equation : 
  ∃ (A B C : ℝ), A * Real.sqrt 3 + B * (-1) + C = 0 ∧ 
  A * point.1 + B * point.2 + C = 0 ∧
  B / A = new_slope := by sorry

-- Theorem for line 2
theorem line2_equation :
  ∃ (m b : ℝ), m = new_slope ∧ b = y_intercept ∧
  ∀ x y : ℝ, y = m * x + b := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line1_equation_line2_equation_l448_44802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_F_l448_44804

-- Define the function F
noncomputable def F (x : ℝ) : ℝ := Real.sqrt (x + 2)

-- Define the domain of F
def domain (x : ℝ) : Prop := x ≥ -2 ∧ x ≠ 1

-- Theorem statement
theorem range_of_F :
  Set.range F = {y : ℝ | y ∈ Set.Icc 0 (Real.sqrt 3) ∨ y > Real.sqrt 3} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_F_l448_44804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l448_44854

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular 5 (π/3) 2 = (5/2, 5 * Real.sqrt 3 / 2, 2) := by
  unfold cylindrical_to_rectangular
  simp [Real.cos_pi_div_three, Real.sin_pi_div_three]
  norm_num
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l448_44854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_prime_l448_44861

theorem existence_of_prime (S : Finset ℕ) : 
  S.card = 15 → 
  (∀ n ∈ S, n > 0 ∧ n < 1998) → 
  (∀ a b, a ∈ S → b ∈ S → a ≠ b → Nat.gcd a b = 1) → 
  ∃ p ∈ S, Nat.Prime p :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_prime_l448_44861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_games_for_condition_l448_44878

/-- Represents a football tournament --/
structure Tournament :=
  (teams : Nat)
  (games_played : Nat)

/-- Predicate to check if two teams have played against each other --/
def have_played (t : Tournament) (a b : Fin t.teams) : Prop := sorry

/-- Predicate to check if the tournament satisfies the condition --/
def satisfies_condition (t : Tournament) : Prop :=
  ∀ (a b c : Fin t.teams), a ≠ b ∧ b ≠ c ∧ a ≠ c →
    have_played t a b ∨ have_played t b c ∨ have_played t a c

/-- The main theorem --/
theorem min_games_for_condition (t : Tournament) :
  t.teams = 16 →
  (∀ a b : Fin t.teams, a ≠ b → have_played t a b ∨ ¬have_played t a b) →
  satisfies_condition t →
  t.games_played ≥ 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_games_for_condition_l448_44878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brandon_textbooks_weight_l448_44863

noncomputable def jon_textbook_weights : List ℝ := [2, 8, 5, 9]

noncomputable def total_jon_weight : ℝ := jon_textbook_weights.sum

noncomputable def brandon_weight : ℝ := total_jon_weight / 3

theorem brandon_textbooks_weight :
  brandon_weight = 8 := by
  -- Unfold definitions
  unfold brandon_weight
  unfold total_jon_weight
  unfold jon_textbook_weights
  -- Simplify the sum
  simp
  -- The proof step is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brandon_textbooks_weight_l448_44863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_solution_proof_l448_44808

-- Define the differential equation
def diff_eq (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv (deriv y)) x + 4 * (deriv y) x + 5 * y x = 8 * Real.cos x

-- Define the boundedness condition
def bounded_neg_infinity (f : ℝ → ℝ) : Prop :=
  ∃ M > 0, ∀ x ≤ 0, |f x| ≤ M

-- Define the proposed solution
noncomputable def proposed_solution (x : ℝ) : ℝ :=
  2 * (Real.cos x + Real.sin x)

-- Theorem statement
theorem particular_solution_proof :
  diff_eq proposed_solution ∧ bounded_neg_infinity proposed_solution := by
  sorry

#check particular_solution_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_solution_proof_l448_44808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_problem_l448_44850

/-- A polynomial of degree at most 2 -/
def Quadratic (R : Type*) [Ring R] := { p : Polynomial R // p.degree ≤ 2 }

variable {R : Type*} [CommRing R]

/-- The statement of the problem -/
theorem quadratic_polynomial_problem 
  (f g h : Quadratic R) 
  (hf : f.val ≠ 0)
  (hg : g.val ≠ 0)
  (hh : h.val ≠ 0)
  (h_eq : f.val.comp g.val = f.val + g.val * h.val)
  (h_g2 : g.val.eval 2 = 12) :
  ∃ (a b c : R), g.val = Polynomial.monomial 2 a + Polynomial.monomial 1 b + Polynomial.monomial 0 c ∧ 
                 a = 1 ∧ b = 3 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_problem_l448_44850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_projections_relationship_l448_44899

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Define the projection operation of a line onto a plane
variable (project : Line → Plane → Line)

-- Define the positional relationship between lines
inductive PositionalRelationship
  | Intersecting
  | Parallel
  | Skew

-- Define the theorem
theorem line_projections_relationship 
  (a : Line) (α β : Plane) (l b c : Line)
  (h1 : intersect α β = l)
  (h2 : a ≠ project a α)
  (h3 : a ≠ project a β)
  (h4 : b = project a α)
  (h5 : c = project a β) :
  (∃ r : PositionalRelationship, r = PositionalRelationship.Intersecting ∨ 
                                  r = PositionalRelationship.Parallel ∨ 
                                  r = PositionalRelationship.Skew) :=
by
  -- The proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_projections_relationship_l448_44899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_lower_bound_l448_44885

theorem vector_sum_magnitude_lower_bound
  (a b c : ℝ × ℝ)
  (ha : ‖a‖ = 1)
  (hab : a • b = 1)
  (hbc : b • c = 1)
  (hac : a • c = 2) :
  ‖a + b + c‖ ≥ 4 ∧ ∃ (a' b' c' : ℝ × ℝ), 
    ‖a'‖ = 1 ∧ a' • b' = 1 ∧ b' • c' = 1 ∧ a' • c' = 2 ∧ ‖a' + b' + c'‖ = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_magnitude_lower_bound_l448_44885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l448_44833

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 + 2*x + a) / Real.log 0.5

def q (a : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ > 2 → x₂ > 2 → x₁ < x₂ → (x₁ - a)^2 < (x₂ - a)^2

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l448_44833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l448_44880

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - x^2 else x^2 - x - 3

theorem f_composition_value : f (1 / f 3) = 8/9 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l448_44880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l448_44827

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ := (principal * rate * time) / 100

/-- Problem statement -/
theorem interest_rate_calculation (principal interest time : ℝ) 
  (h_principal : principal = 810)
  (h_interest : interest = 155)
  (h_time : time = 4) :
  ∃ (rate : ℝ), 
    simple_interest principal rate time = interest ∧ 
    (rate ≥ 4.78 ∧ rate < 4.79) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l448_44827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_three_non_parallel_lines_perpendicular_three_hexagon_sides_l448_44852

-- Define the basic structures
structure Line where

structure Plane where

-- Define the perpendicular relation
def perpendicular : (Line ⊕ Plane) → (Line ⊕ Plane) → Prop := sorry

-- Define the "within" relation for a line and a plane
def within : Line → Plane → Prop := sorry

-- Define a function to check if lines are non-parallel
def non_parallel : Line → Line → Line → Prop := sorry

-- Define a regular hexagon
structure RegularHexagon where
  plane : Plane
  side1 : Line
  side2 : Line
  side3 : Line
  within_plane : within side1 plane ∧ within side2 plane ∧ within side3 plane

-- Theorem 1
theorem perpendicular_three_non_parallel_lines 
  (l : Line) (p : Plane) (l1 l2 l3 : Line) 
  (h1 : within l p)
  (h2 : within l1 p ∧ within l2 p ∧ within l3 p)
  (h3 : non_parallel l1 l2 l3)
  (h4 : perpendicular (Sum.inl l) (Sum.inl l1) ∧ 
        perpendicular (Sum.inl l) (Sum.inl l2) ∧ 
        perpendicular (Sum.inl l) (Sum.inl l3)) :
  perpendicular (Sum.inl l) (Sum.inr p) :=
sorry

-- Theorem 2
theorem perpendicular_three_hexagon_sides
  (l : Line) (h : RegularHexagon)
  (h1 : within l h.plane)
  (h2 : perpendicular (Sum.inl l) (Sum.inl h.side1) ∧ 
        perpendicular (Sum.inl l) (Sum.inl h.side2) ∧ 
        perpendicular (Sum.inl l) (Sum.inl h.side3)) :
  perpendicular (Sum.inl l) (Sum.inr h.plane) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_three_non_parallel_lines_perpendicular_three_hexagon_sides_l448_44852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_max_volume_ratio_l448_44870

/-- Represents a cylinder --/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The surface area of a cylinder --/
noncomputable def surfaceArea (c : Cylinder) : ℝ := 2 * Real.pi * c.radius * (c.radius + c.height)

/-- The volume of a cylinder --/
noncomputable def volume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- Theorem stating that for a cylinder with surface area 12π, 
    when its volume is maximized, the ratio of its base radius to its height is 1/2 --/
theorem cylinder_max_volume_ratio (c : Cylinder) 
    (h_surface : surfaceArea c = 12 * Real.pi) 
    (h_max : ∀ c' : Cylinder, surfaceArea c' = 12 * Real.pi → volume c' ≤ volume c) : 
    c.radius / c.height = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_max_volume_ratio_l448_44870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_inequality_l448_44876

-- Define f as an increasing function on ℝ
def f : ℝ → ℝ := sorry

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := sorry

-- Define a, b, and c
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

-- State the theorem
theorem inverse_function_inequality (hf : StrictMono f) 
  (hf_inv : f_inv = f.invFun)
  (h1 : f 3 = 0)
  (h2 : f 2 = a)
  (h3 : f_inv 2 = b)
  (h4 : f_inv 0 = c) :
  b > c ∧ c > a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_inequality_l448_44876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_max_lower_bound_l448_44843

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |3*x + 1/a| + 3*|x - a|

-- Part I
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 8} = Set.Iic (-1) ∪ Set.Ici (5/3) :=
sorry

-- Part II
theorem max_lower_bound (a : ℝ) (ha : a > 0) :
  ∃ m : ℝ, (∀ x : ℝ, f a x ≥ m) ∧ 
    (∀ m' : ℝ, (∀ x : ℝ, f a x ≥ m') → m' ≤ m) ∧
    m = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_max_lower_bound_l448_44843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_lambda_condition_l448_44825

def a (n : ℕ+) (lambda : ℝ) : ℝ := 2 * (n : ℝ)^2 + lambda * (n : ℝ)

theorem increasing_sequence_lambda_condition (lambda : ℝ) :
  (∀ n : ℕ+, a n lambda < a (n + 1) lambda) ↔ lambda > -6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sequence_lambda_condition_l448_44825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_squares_l448_44821

theorem constant_sum_squares (w : ℂ) (h : Complex.abs (w - (3 - 3*Complex.I)) = 6) :
  (Complex.abs (w + (2 - Complex.I)))^2 + (Complex.abs (w - (7 - 5*Complex.I)))^2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_squares_l448_44821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_multiple_of_18_to_2509_l448_44891

theorem closest_multiple_of_18_to_2509 : 
  ∀ n : ℤ, n ≠ 2502 → n % 18 = 0 → |2509 - 2502| ≤ |2509 - n| := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_multiple_of_18_to_2509_l448_44891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_train_time_l448_44892

noncomputable def distance_to_work : ℝ := 1.5
noncomputable def walking_speed : ℝ := 3
noncomputable def train_speed : ℝ := 20
noncomputable def extra_walking_time_minutes : ℝ := 10

noncomputable def walking_time_minutes : ℝ := distance_to_work / walking_speed * 60
noncomputable def train_travel_time_minutes : ℝ := distance_to_work / train_speed * 60

theorem additional_train_time (x : ℝ) : 
  walking_time_minutes = train_travel_time_minutes + x + extra_walking_time_minutes → 
  x = 15.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_train_time_l448_44892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_booth_visibility_distances_l448_44811

/-- The angle between two 2D vectors -/
noncomputable def angle_between_vectors (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

/-- The minimum and maximum distances from which a square booth can be seen -/
theorem booth_visibility_distances (L : ℝ) (h : L > 0) :
  ∃ (ρ_min ρ_max : ℝ),
    ρ_min = L ∧
    ρ_max = (1 + Real.sqrt 2) / 2 * L ∧
    (∀ (d : ℝ), d ≥ ρ_min ∧ d ≤ ρ_max ↔ 
      ∃ (P : ℝ × ℝ), 
        (Real.sqrt ((P.1 ^ 2 + P.2 ^ 2) / L ^ 2) = d / L) ∧
        (∃ (A B : ℝ × ℝ), 
          (A.1 = 0 ∧ A.2 = 0) ∧ 
          ((B.1 = L ∧ B.2 = 0) ∨ (B.1 = 0 ∧ B.2 = L) ∨ (B.1 = L ∧ B.2 = L)) ∧
          (angle_between_vectors (A.1 - P.1, A.2 - P.2) (B.1 - P.1, B.2 - P.2) = π / 4))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_booth_visibility_distances_l448_44811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_type_quadratic_radical_l448_44824

theorem same_type_quadratic_radical (x : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ Real.sqrt (4 - 3*x) = a * Real.sqrt b ∧ Real.sqrt 8 = a * Real.sqrt b) → x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_type_quadratic_radical_l448_44824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l448_44864

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 1/a|

theorem problem_solution :
  (∀ x, x ∈ Set.Icc (-17/4) (11/4) ↔ f 2 x ≤ 7) ∧
  (∀ a, (∀ x, f a x ≥ 10/3) ↔ 
    a ∈ Set.Iic (-3) ∪ Set.Icc (-1/3) 0 ∪ Set.Ioc 0 (1/3) ∪ Set.Ici 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l448_44864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_is_60_minutes_l448_44874

-- Define the average speed of the bus
noncomputable def average_speed : ℝ := 40

-- Define the distances for the two trips
noncomputable def distance1 : ℝ := 360
noncomputable def distance2 : ℝ := 400

-- Define the function to calculate time given distance and speed
noncomputable def calculate_time (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

-- Define the function to convert hours to minutes
noncomputable def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

-- Theorem statement
theorem time_difference_is_60_minutes :
  hours_to_minutes (calculate_time distance2 average_speed - calculate_time distance1 average_speed) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_is_60_minutes_l448_44874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_of_given_parabola_l448_44823

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Calculates the y-coordinate of the directrix for a parabola -/
noncomputable def directrix_y (p : Parabola) : ℚ := 
  let vertex_x := -p.b / (2 * p.a)
  let vertex_y := p.a * vertex_x^2 + p.b * vertex_x + p.c
  vertex_y - 1 / (4 * p.a)

/-- The given parabola y = (x² - 8x + 12) / 16 -/
def given_parabola : Parabola := {
  a := 1/16,
  b := -1/2,
  c := 3/4
}

theorem directrix_of_given_parabola :
  directrix_y given_parabola = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_of_given_parabola_l448_44823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_l448_44841

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
noncomputable def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the volume of a cube given its side length -/
noncomputable def cubeVolume (side : ℝ) : ℝ :=
  side ^ 3

/-- Calculates the percentage of volume removed from a box -/
noncomputable def percentageVolumeRemoved (boxDim : BoxDimensions) (cubeSide : ℝ) : ℝ :=
  let totalRemoved := 8 * cubeVolume cubeSide
  (totalRemoved / boxVolume boxDim) * 100

/-- Theorem stating that removing 2 cm cubes from each corner of an 18 cm × 12 cm × 9 cm box
    removes approximately 3.29% of its volume -/
theorem volume_removed_percentage :
  let boxDim : BoxDimensions := ⟨18, 12, 9⟩
  let cubeSide : ℝ := 2
  abs (percentageVolumeRemoved boxDim cubeSide - 3.29) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_l448_44841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_permutation_maximizes_sum_l448_44835

def is_permutation (a : List ℕ) : Prop :=
  a.length = 2011 ∧ a.toFinset = Finset.range 2011

def sum_of_differences (a : List ℕ) : ℕ :=
  (List.zip a a.tail).foldl (λ sum pair => sum + (max pair.1 pair.2 - min pair.1 pair.2)) 0

def optimal_permutation : List ℕ :=
  (List.range 2010).map (λ i => i + 2) ++ [2011, 1]

theorem optimal_permutation_maximizes_sum :
  ∀ a : List ℕ, is_permutation a →
    sum_of_differences a ≤ sum_of_differences optimal_permutation :=
by
  sorry

#eval sum_of_differences optimal_permutation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_permutation_maximizes_sum_l448_44835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_combined_time_l448_44859

/-- The time it takes for two workers to complete a job together -/
noncomputable def combined_time (time_a time_b : ℝ) : ℝ :=
  1 / (1 / time_a + 1 / time_b)

/-- Theorem: Two workers with individual completion times of 10 and 12 hours
    will complete the job in 60/11 hours when working together -/
theorem workers_combined_time :
  combined_time 10 12 = 60 / 11 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_combined_time_l448_44859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l448_44818

def sequence_a (k : ℤ) : ℕ → ℤ
  | 0 => 0
  | n + 1 => k * sequence_a k n - Int.sqrt ((k^2 - 1) * (sequence_a k n)^2 + 1)

theorem sequence_properties (k : ℤ) :
  (∀ n : ℕ, ∃ m : ℤ, sequence_a k n = m) ∧
  (∀ n : ℕ, (2 * k) ∣ sequence_a k (2 * n)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l448_44818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l448_44844

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem function_properties (φ : ℝ) 
  (h1 : -π < φ ∧ φ < 0)
  (h2 : ∀ x, f x φ = f (π/4 - x) φ) : 
  (φ = -3*π/4) ∧ 
  (∀ m : ℤ, ∀ x ∈ Set.Icc (π/8 + m*π) (5*π/8 + m*π), 
    ∀ y ∈ Set.Icc (π/8 + m*π) (5*π/8 + m*π), 
    x ≤ y → f x φ ≤ f y φ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l448_44844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_E_properties_l448_44855

-- Define the hyperbola E
def hyperbola_E (x y : ℝ) : Prop := y^2 / 9 - x^2 / 4 = 1

-- Define the asymptote hyperbola
def asymptote_hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 9 = 1

-- Define a point on the hyperbola E
def point_on_E : Prop := hyperbola_E (-2) (3 * Real.sqrt 2)

-- Define that E shares asymptotes with the given hyperbola
def shared_asymptotes : Prop := ∀ x y : ℝ, hyperbola_E x y ↔ ∃ k : ℝ, x^2 / 4 - y^2 / 9 = k

-- Define a line intersecting E at two points
def intersecting_line (l : ℝ → ℝ → Prop) (A B : ℝ × ℝ) : Prop :=
  l A.1 A.2 ∧ l B.1 B.2 ∧ hyperbola_E A.1 A.2 ∧ hyperbola_E B.1 B.2

-- Define the tangent line at a point on E
def tangent_line (x₀ y₀ : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ 9 * x₀ * x - 4 * y₀ * y + 36 = 0

theorem hyperbola_E_properties :
  point_on_E →
  shared_asymptotes →
  (∀ l : ℝ → ℝ → Prop, ∀ A B : ℝ × ℝ,
    intersecting_line l A B →
    (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 4 →
    ∀ x y : ℝ, l x y ↔ 9 * x - 16 * y + 55 = 0) →
  (∀ x₀ y₀ : ℝ, hyperbola_E x₀ y₀ → tangent_line x₀ y₀ = λ x y ↦ 9 * x₀ * x - 4 * y₀ * y + 36 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_E_properties_l448_44855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pawn_traversal_iff_even_l448_44807

/-- Represents a square on the chessboard -/
structure Square where
  x : Nat
  y : Nat
deriving Inhabited

/-- Represents a move on the chessboard -/
inductive Move
  | Side   : Square → Square → Move  -- Move type (a)
  | Vertex : Square → Square → Move  -- Move type (b)
deriving Inhabited

/-- Checks if two squares are adjacent (sharing a side) -/
def adjacent (s1 s2 : Square) : Prop :=
  (s1.x = s2.x ∧ s1.y + 1 = s2.y) ∨
  (s1.x = s2.x ∧ s1.y = s2.y + 1) ∨
  (s1.x + 1 = s2.x ∧ s1.y = s2.y) ∨
  (s1.x = s2.x + 1 ∧ s1.y = s2.y)

/-- Checks if two squares are diagonally adjacent (sharing only a vertex) -/
def diagonallyAdjacent (s1 s2 : Square) : Prop :=
  (s1.x + 1 = s2.x ∧ s1.y + 1 = s2.y) ∨
  (s1.x + 1 = s2.x ∧ s1.y = s2.y + 1) ∨
  (s1.x = s2.x + 1 ∧ s1.y + 1 = s2.y) ∨
  (s1.x = s2.x + 1 ∧ s1.y = s2.y + 1)

/-- Checks if a move is legal -/
def legalMove : Move → Prop
  | Move.Side s1 s2   => adjacent s1 s2
  | Move.Vertex s1 s2 => diagonallyAdjacent s1 s2

/-- Checks if two moves are of different types -/
def differentMoveTypes : Move → Move → Prop
  | Move.Side _ _,   Move.Vertex _ _ => True
  | Move.Vertex _ _, Move.Side _ _   => True
  | _, _ => False

/-- Represents a sequence of moves on the board -/
def MoveSequence := List Move

/-- Checks if a move sequence is valid (alternating move types and legal moves) -/
def validMoveSequence : MoveSequence → Prop
  | [] => True
  | [m] => legalMove m
  | m1 :: m2 :: rest => legalMove m1 ∧ legalMove m2 ∧ differentMoveTypes m1 m2 ∧ validMoveSequence (m2 :: rest)

/-- Checks if a move sequence visits all squares exactly once -/
def visitsAllSquaresOnce (n : Nat) (seq : MoveSequence) : Prop :=
  ∃ (start : Square), 
    start.x ≤ n ∧ start.y ≤ n ∧
    (∀ (s : Square), s.x ≤ n ∧ s.y ≤ n → 
      ∃! (index : Nat), index < seq.length ∧
        (index = 0 ∧ s = start) ∨
        (∃ (prev : Square), 
          (seq.get! index = Move.Side prev s ∨ seq.get! index = Move.Vertex prev s)))

/-- The main theorem: A valid move sequence visiting all squares exactly once exists if and only if n is even -/
theorem pawn_traversal_iff_even (n : Nat) : 
  (n ≥ 2 ∧ ∃ (seq : MoveSequence), validMoveSequence seq ∧ visitsAllSquaresOnce n seq) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pawn_traversal_iff_even_l448_44807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_PAB_l448_44806

/-- Given a triangle PAB where the distance between fixed points A and B is 2,
    the ratio of distances PA to PB is √2, and P, A, and B are not collinear,
    the maximum area of triangle PAB is 2√2. -/
theorem max_area_triangle_PAB (A B P : ℝ × ℝ) 
  (dist : ℝ × ℝ → ℝ × ℝ → ℝ)
  (Collinear : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop)
  (area : ℝ × ℝ → ℝ) : 
  (dist A B = 2) →
  (dist P A = Real.sqrt 2 * dist P B) →
  ¬ Collinear A B P →
  (∃ (S : Set (ℝ × ℝ)), S = {Q | dist Q A = Real.sqrt 2 * dist Q B}) →
  (area P = ⨆ Q, area Q) →
  area P = 2 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_PAB_l448_44806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_max_k_l448_44810

theorem triangle_inequality_max_k (A B C : ℝ) (hABC : A + B + C = π) :
  (∀ k : ℝ, 2 * (Real.sin C)^2 + Real.sin A * Real.sin B > k * Real.sin B * Real.sin C) →
  (∃ k_max : ℝ, k_max = 2 * Real.sqrt 2 - 1 ∧
    (∀ k : ℝ, 2 * (Real.sin C)^2 + Real.sin A * Real.sin B > k * Real.sin B * Real.sin C → k ≤ k_max)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_max_k_l448_44810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_approximations_are_convergents_l448_44897

/-- Continued fraction expansion of a real number -/
noncomputable def continuedFraction (α : ℝ) : ℕ → ℤ := sorry

/-- Convergents of a continued fraction -/
noncomputable def convergent (α : ℝ) : ℕ → ℚ := sorry

/-- Best approximation of a real number -/
def bestApproximation (α : ℝ) (x y : ℤ) : Prop :=
  ∀ a b : ℤ, b > 0 → b ≤ y → |x - α * y| ≤ |a - α * b|

/-- Theorem: Best approximations are convergents of continued fraction expansion -/
theorem best_approximations_are_convergents (α : ℝ) :
  ∀ x y : ℤ, bestApproximation α x y ↔ ∃ n : ℕ, (x : ℚ) / y = convergent α n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_approximations_are_convergents_l448_44897
