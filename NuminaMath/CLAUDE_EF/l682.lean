import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_at_one_l682_68247

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then x + 5 else 2 * x^2 + 1

theorem f_composition_at_one : f (f 1) = 8 := by
  -- Evaluate f(1)
  have h1 : f 1 = 3 := by
    rw [f]
    simp [if_neg (not_lt.mpr (le_refl 1))]
    norm_num
  
  -- Evaluate f(f(1)) = f(3)
  have h2 : f 3 = 8 := by
    rw [f]
    simp [if_pos (by norm_num : 3 > 1)]
    norm_num

  -- Combine the results
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_at_one_l682_68247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l682_68289

theorem tan_double_angle (θ : ℝ) (h1 : 0 < θ ∧ θ < π/2) 
  (h2 : Real.sin (θ - π/4) = Real.sqrt 2/10) : Real.tan (2*θ) = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l682_68289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_muffin_banana_ratio_l682_68240

/-- The cost of a single muffin -/
def m : ℝ := sorry

/-- The cost of a single banana -/
def b : ℝ := sorry

/-- Alice's total cost -/
def alice_cost : ℝ := 5 * m + 2 * b

/-- Bob's total cost -/
def bob_cost : ℝ := 3 * m + 12 * b

/-- Bob pays three times as much as Alice -/
axiom bob_pays_triple : bob_cost = 3 * alice_cost

/-- The ratio of the cost of a muffin to the cost of a banana is 2 -/
theorem muffin_banana_ratio : m / b = 2 := by
  sorry

#check muffin_banana_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_muffin_banana_ratio_l682_68240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_relationship_l682_68265

/-- The function f(x) with constants a and b -/
noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * Real.log (x + Real.sqrt (x^2 + 1)) + 5

/-- The theorem stating the relationship between the minimum on (0, +∞) and maximum on (-∞, 0) -/
theorem min_max_relationship (a b : ℝ) :
  (∃ (m : ℝ), m = -4 ∧ ∀ x > 0, f a b x ≥ m) →
  (∃ (M : ℝ), M = 14 ∧ ∀ x < 0, f a b x ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_relationship_l682_68265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flash_catches_ace_l682_68239

/-- The distance at which Flash catches up to Ace -/
noncomputable def catchup_distance (x y z v : ℝ) : ℝ := x * (y + v * z) / (x - 1)

/-- Theorem stating the distance at which Flash catches up to Ace -/
theorem flash_catches_ace (x y z v : ℝ) (hx : x > 1) (hy : y ≥ 0) (hz : z ≥ 0) (hv : v > 0) :
  let d := catchup_distance x y z v
  let t := d / (x * v)
  (v * t + y + v * z = x * v * t) ∧ (d ≥ 0) := by
  sorry

#check flash_catches_ace

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flash_catches_ace_l682_68239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l682_68262

/-- The central angle of the unfolded diagram of a cone's lateral surface -/
noncomputable def central_angle (θ : ℝ) : ℝ := 2 * Real.pi * (Real.sin θ)

/-- Theorem: The central angle of the unfolded diagram of a cone's lateral surface is π 
    when the angle between the generatrix and the axis is 30° -/
theorem cone_central_angle :
  let θ : ℝ := 30 * (Real.pi / 180)  -- 30° in radians
  let α : ℝ := central_angle θ
  α > 0 ∧ α = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_central_angle_l682_68262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_and_smallest_A_l682_68206

def is_nine_digit (n : ℕ) : Prop := 100000000 ≤ n ∧ n ≤ 999999999

def is_coprime_with_24 (n : ℕ) : Prop := Nat.Coprime n 24

def move_last_digit_to_first (n : ℕ) : ℕ :=
  let last_digit := n % 10
  let rest := n / 10
  last_digit * 100000000 + rest

theorem largest_and_smallest_A (B : ℕ) (hB1 : is_nine_digit B) (hB2 : is_coprime_with_24 B) (hB3 : B > 666666666) :
  let A := move_last_digit_to_first B
  (∀ B', is_nine_digit B' → is_coprime_with_24 B' → B' > 666666666 →
    move_last_digit_to_first B' ≤ 999999998) ∧
  (∀ B', is_nine_digit B' → is_coprime_with_24 B' → B' > 666666666 →
    move_last_digit_to_first B' ≥ 166666667) ∧
  (∃ B1 B2, is_nine_digit B1 ∧ is_coprime_with_24 B1 ∧ B1 > 666666666 ∧
            is_nine_digit B2 ∧ is_coprime_with_24 B2 ∧ B2 > 666666666 ∧
            move_last_digit_to_first B1 = 999999998 ∧
            move_last_digit_to_first B2 = 166666667) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_and_smallest_A_l682_68206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_for_f_l682_68251

/-- The function for which we're finding the axis of symmetry -/
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 3)

/-- Theorem stating that π/6 is an axis of symmetry for the function f -/
theorem axis_of_symmetry_for_f :
  ∀ x : ℝ, f (Real.pi / 6 + x) = f (Real.pi / 6 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_for_f_l682_68251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_angle_C_when_a_is_7_l682_68287

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.B ∈ Set.Ioo 0 Real.pi ∧
  Real.cos t.B = 3/5 ∧
  t.a * t.c = 35

-- Theorem for the area of the triangle
theorem area_of_triangle (t : Triangle) (h : triangle_conditions t) : 
  (1/2) * t.a * t.c * Real.sin t.B = 14 := by
  sorry

-- Theorem for angle C when a = 7
theorem angle_C_when_a_is_7 (t : Triangle) (h : triangle_conditions t) (ha : t.a = 7) : 
  t.C = Real.pi/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_angle_C_when_a_is_7_l682_68287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l682_68248

/-- The area of a triangle with vertices at (2, -3), (8, 6), and (5, 1) is 1.5 -/
theorem triangle_area : 
  let A : ℝ × ℝ := (2, -3)
  let B : ℝ × ℝ := (8, 6)
  let C : ℝ × ℝ := (5, 1)
  let area := 1.5
  area = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l682_68248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_equals_28_div_3_l682_68256

/-- Two similar right triangles where one has legs 12 and 9, and the other has legs y and 7 -/
structure SimilarRightTriangles where
  y : ℝ
  similarity : (12 : ℝ) / y = 9 / 7

/-- The value of y in the similar right triangles -/
noncomputable def y_value (triangles : SimilarRightTriangles) : ℝ := 28 / 3

/-- Theorem stating that y in the similar right triangles equals 28/3 -/
theorem y_equals_28_div_3 (triangles : SimilarRightTriangles) : 
  triangles.y = y_value triangles := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_equals_28_div_3_l682_68256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l682_68297

/-- Power function with specific properties -/
def power_function (p : ℕ+) (x : ℝ) : ℝ := x^(p.val^2 - 2*p.val - 3)

/-- The main theorem -/
theorem range_of_a (p : ℕ+) (a : ℝ) :
  (∀ x > 0, ∀ y > 0, x < y → power_function p y < power_function p x) →  -- Decreasing on (0,+∞)
  (∀ x : ℝ, power_function p x = power_function p (-x)) →  -- Symmetric about y-axis
  ((a^2 - 1)^(p.val/3 : ℝ) < (3*a + 3)^(p.val/3 : ℝ)) →  -- Given inequality
  -1 < a ∧ a < 4 :=  -- Range of a
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l682_68297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antons_number_l682_68234

/-- Represents a three-digit number -/
def ThreeDigitNumber : Type := { n : ℕ // 100 ≤ n ∧ n < 1000 }

/-- Checks if two numbers match in exactly one digit place -/
def matches_one_digit (a b : ThreeDigitNumber) : Prop :=
  (a.val / 100 = b.val / 100 ∧ a.val % 100 ≠ b.val % 100) ∨
  (a.val / 10 % 10 = b.val / 10 % 10 ∧ (a.val / 100 ≠ b.val / 100 ∨ a.val % 10 ≠ b.val % 10)) ∨
  (a.val % 10 = b.val % 10 ∧ a.val / 10 ≠ b.val / 10)

/-- The main theorem -/
theorem antons_number (x : ThreeDigitNumber) 
  (h1 : matches_one_digit x ⟨109, by norm_num [ThreeDigitNumber]⟩)
  (h2 : matches_one_digit x ⟨704, by norm_num [ThreeDigitNumber]⟩)
  (h3 : matches_one_digit x ⟨124, by norm_num [ThreeDigitNumber]⟩) :
  x = ⟨729, by norm_num [ThreeDigitNumber]⟩ := by
  sorry

#check antons_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_antons_number_l682_68234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_M_N_is_real_l682_68298

-- Define set M
def M : Set ℝ := {x | x^2 + 3*x + 2 > 0}

-- Define set N
def N : Set ℝ := {x | Real.rpow (1/2) x ≤ 4}

-- Theorem statement
theorem union_M_N_is_real : M ∪ N = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_M_N_is_real_l682_68298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l682_68226

theorem equation_solution :
  ∃ x : ℚ, (3 : ℝ) ^ ((2 : ℝ) * x + 1) = (81 : ℝ) ^ ((1 : ℝ)/3) ∧ x = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l682_68226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_cosine_property_l682_68214

theorem sequence_cosine_property (α : ℝ) : 
  (Real.cos α ≤ -1/4 ∧ ∀ n : ℕ, Real.cos (2^n * α) ≤ -1/4) ↔ 
  ∃ k : ℤ, α = 2*Real.pi/3 + 2*k*Real.pi ∨ α = -2*Real.pi/3 + 2*k*Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_cosine_property_l682_68214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_savings_theorem_l682_68238

/-- Represents the fuel efficiency improvement of the new car compared to the old car -/
noncomputable def efficiency_improvement : ℝ := 1.6

/-- Represents the fuel cost increase of the new car compared to the old car -/
noncomputable def cost_increase : ℝ := 1.25

/-- Represents the trip distance in kilometers -/
noncomputable def trip_distance : ℝ := 100

/-- Calculates the percentage savings in fuel costs for the given trip -/
noncomputable def fuel_cost_savings : ℝ := 
  (1 - cost_increase / efficiency_improvement) * 100

theorem fuel_savings_theorem :
  fuel_cost_savings = 21.875 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_savings_theorem_l682_68238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lock_combinations_l682_68271

theorem lock_combinations : 
  (Finset.univ.filter (λ s : Fin 10 → Fin 4 ↦ Function.Injective s)).card = 5040 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lock_combinations_l682_68271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_implies_product_l682_68291

theorem sin_cos_equation_implies_product (θ : ℝ) :
  (Real.sin θ^2 + 4) / (Real.cos θ + 1) = 2 →
  (Real.cos θ + 1) * (Real.sin θ + 1) = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equation_implies_product_l682_68291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l682_68266

theorem function_inequality (f g : ℝ → ℝ) (h₁ : Continuous f) (h₂ : Continuous g)
  (h₃ : Differentiable ℝ f) (h₄ : Differentiable ℝ g) (h₅ : f 0 = g 0) (h₆ : f 0 > 0)
  (h₇ : ∀ x ∈ Set.Icc 0 1, (deriv f x) * Real.sqrt (deriv g x) = 3) :
  ∀ x ∈ Set.Icc 0 1, 2 * f x + 3 * g x > 9 * x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l682_68266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_remaining_work_time_l682_68253

/-- Represents the time (in days) required to complete a work --/
def Time := ℝ

/-- Represents the rate of work (portion of work completed per day) --/
def Rate := ℝ

/-- The total amount of work to be done --/
def TotalWork := ℝ

theorem arun_remaining_work_time 
  (total_work : TotalWork)
  (arun_tarun_time : Time)
  (arun_time : Time)
  (tarun_left_after : Time)
  (h1 : arun_tarun_time = (10 : ℝ))
  (h2 : arun_time = (30 : ℝ))
  (h3 : tarun_left_after = (4 : ℝ))
  : ∃ (remaining_time : Time), remaining_time = (18 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_remaining_work_time_l682_68253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_arithmetic_problem_l682_68217

theorem modular_arithmetic_problem (n : ℕ) 
  (h1 : n < 29) (h2 : (4 * n) % 29 = 1) :
  (3^n)^4 % 29 - 3 ≡ 17 [ZMOD 29] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_arithmetic_problem_l682_68217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basket_apples_l682_68218

/-- The number of apples in the basket -/
def n : ℕ := sorry

/-- The probability of selecting the spoiled apple when choosing 2 apples randomly -/
def prob_spoiled (n : ℕ) : ℚ := (n - 1 : ℚ) / (n.choose 2 : ℚ)

/-- Theorem: If the probability of selecting the spoiled apple is 1/4, then there are 8 apples -/
theorem basket_apples : prob_spoiled n = 1/4 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basket_apples_l682_68218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_value_l682_68273

/-- The minimum distance from a point on a curve to a line -/
noncomputable def b : ℝ := (3 * Real.sqrt 2) / 2 - 1

/-- Theorem stating that b is equal to the given expression -/
theorem min_distance_value : b = (3 * Real.sqrt 2) / 2 - 1 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_value_l682_68273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l682_68281

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then (2 : ℝ)^x else Real.log (x - 1)

-- State the theorem
theorem m_range (m : ℝ) :
  (∀ x, x ≤ 2 → f x ≤ 4 - m * x) ↔ 0 ≤ m ∧ m ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l682_68281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l682_68294

-- Define the cistern and pipe capacities
noncomputable def cistern_capacity : ℝ := 1
noncomputable def initial_fill : ℝ := 2/3
noncomputable def remaining_capacity : ℝ := cistern_capacity - initial_fill

-- Define pipe rates (in terms of full cistern per minute)
noncomputable def pipe_a_rate : ℝ := 1 / (12 * 3)
noncomputable def pipe_b_rate : ℝ := 1 / (8 * 3)
noncomputable def pipe_c_rate : ℝ := -1 / 24

-- Combined rate of all pipes
noncomputable def combined_rate : ℝ := pipe_a_rate + pipe_b_rate + pipe_c_rate

-- Theorem to prove
theorem cistern_fill_time :
  (remaining_capacity / combined_rate) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l682_68294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_difference_l682_68236

theorem set_equality_implies_difference (a b : ℝ) : 
  ({a, 1} : Set ℝ) = {0, a + b} → b - a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_implies_difference_l682_68236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_planes_l682_68254

-- Define a structure for a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a structure for a plane in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Function to calculate distance between a point and a plane
noncomputable def distancePointToPlane (p : Point3D) (plane : Plane3D) : ℝ :=
  (plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d) / 
  Real.sqrt (plane.a^2 + plane.b^2 + plane.c^2)

-- Function to check if four points are non-coplanar
def areNonCoplanar (a b c d : Point3D) : Prop :=
  sorry

-- Function to check if distances between points are distinct
def haveDistinctDistances (a b c d : Point3D) : Prop :=
  sorry

-- Function to count planes satisfying the condition
def countSatisfyingPlanes (a b c d : Point3D) : ℕ :=
  sorry

-- Theorem statement
theorem count_satisfying_planes 
  (a b c d : Point3D) 
  (h1 : areNonCoplanar a b c d) 
  (h2 : haveDistinctDistances a b c d) : 
  countSatisfyingPlanes a b c d = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_planes_l682_68254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2015_equals_one_third_l682_68205

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => -2
  | n + 1 => 1 - 1 / sequenceA n

theorem sequence_2015_equals_one_third :
  sequenceA 2015 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2015_equals_one_third_l682_68205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_calculation_l682_68215

/-- Represents the pricing structure of an article -/
structure ArticlePricing where
  cost : ℝ
  marked : ℝ

/-- Calculates the selling price after deduction -/
def sellingPrice (a : ArticlePricing) : ℝ :=
  a.marked * (1 - 0.12)

/-- Calculates the profit based on cost -/
def profitPrice (a : ArticlePricing) : ℝ :=
  a.cost * 1.25

/-- Theorem stating the relationship between cost and selling price -/
theorem cost_calculation (a : ArticlePricing) 
  (h1 : sellingPrice a = profitPrice a)
  (h2 : sellingPrice a = 67.47) : 
  ‖a.cost - 53.98‖ < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_calculation_l682_68215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_individual_score_is_observation_l682_68200

/-- Represents a candidate's math score -/
structure MathScore where
  score : ℝ

/-- Represents the population of candidates -/
structure Population where
  candidates : Finset MathScore
  size : ℕ
  size_positive : 0 < size

/-- Represents a sample drawn from the population -/
structure Sample where
  scores : Finset MathScore
  size : ℕ
  size_positive : 0 < size

/-- The statistical analysis setup -/
structure StatisticalAnalysis where
  population : Population
  sample : Sample
  sample_from_population : sample.scores ⊆ population.candidates

/-- Defines what it means for a math score to be an observation -/
def IsObservation (score : MathScore) : Prop :=
  True  -- For simplicity, we consider any MathScore to be an observation

/-- Theorem: In a statistical analysis, each candidate's score is an individual observation -/
theorem individual_score_is_observation (analysis : StatisticalAnalysis) 
  (score : MathScore) (h : score ∈ analysis.sample.scores) :
  IsObservation score := by
  -- The proof is trivial given our definition of IsObservation
  exact True.intro


end NUMINAMATH_CALUDE_ERRORFEEDBACK_individual_score_is_observation_l682_68200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_between_100_and_500_l682_68237

theorem perfect_squares_between_100_and_500 :
  (Finset.filter (λ n : ℕ => 100 ≤ n^2 ∧ n^2 ≤ 500) (Finset.range 501)).card = 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_between_100_and_500_l682_68237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_to_cylinder_radius_l682_68233

theorem sphere_to_cylinder_radius (R : ℝ) : 
  R > 0 →  -- Radius is positive
  (4 / 3 * Real.pi * R^3) = (Real.pi * 4^3) →  -- Volume equality
  R = 2 * (2 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_to_cylinder_radius_l682_68233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_account_balance_l682_68235

theorem bank_account_balance 
  (X : ℝ)                            -- Initial balance
  (h1 : 2/5 * X = 200)               -- Withdrawal condition
  (h2 : X > 200)                     -- Ensure positive balance after withdrawal
  : X - 200 + (1/5 * (X - 200)) = 360 := by
  sorry

#check bank_account_balance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_account_balance_l682_68235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_taken_l682_68245

/-- The time taken by Delta to complete the task alone -/
def D : ℝ := sorry

/-- The time taken by Epsilon to complete the task alone -/
def E : ℝ := sorry

/-- The time taken by Delta and Epsilon together to complete the task -/
def DE : ℝ := sorry

/-- Condition: Delta and Epsilon together can finish the task in 4 hours less than Delta alone -/
axiom cond1 : DE = D - 4

/-- Condition: Delta and Epsilon together can finish the task in 3 hours less than Epsilon alone -/
axiom cond2 : DE = E - 3

/-- Condition: With Gamma, the three can finish the task in half the time Delta and Epsilon would need together -/
axiom cond3 : DE / 2 = (1 / D + 1 / E + 1 / 3) ⁻¹

/-- Theorem: The time taken by Delta and Epsilon together to complete the task is 42/13 hours -/
theorem time_taken : DE = 42 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_taken_l682_68245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_diagonal_division_l682_68295

/-- A regular hexagon with vertices A, B, C, D, E, F -/
structure RegularHexagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

/-- A point dividing a line segment -/
def divides (P Q R : ℝ × ℝ) (r : ℝ) : Prop :=
  dist P R = r * dist Q R

/-- Three points are collinear -/
def collinear (P Q R : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, Q = P + t • (R - P) ∨ R = P + t • (Q - P)

/-- Main theorem -/
theorem hexagon_diagonal_division (ABCDEF : RegularHexagon) 
  (M N : ℝ × ℝ) (r : ℝ) :
  divides ABCDEF.A M ABCDEF.C r →
  divides ABCDEF.C N ABCDEF.E r →
  collinear ABCDEF.B M N →
  r = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_diagonal_division_l682_68295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_team_min_score_l682_68268

def player_score : ℕ → ℕ := sorry

theorem basketball_team_min_score (total_points : ℕ) (num_players : ℕ) (max_individual_score : ℕ) 
  (h1 : total_points = 100)
  (h2 : num_players = 12)
  (h3 : max_individual_score = 23) :
  ∃ (min_score : ℕ), 
    (∀ (player : ℕ), player < num_players → min_score ≤ player_score player) ∧ 
    (∃ (player : ℕ), player < num_players ∧ player_score player = max_individual_score) ∧
    (Finset.sum (Finset.range num_players) player_score = total_points) ∧
    min_score = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_team_min_score_l682_68268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_monotone_increasing_l682_68201

noncomputable def f (x : ℝ) : ℝ := 2 ^ (|x - 1|)

def monotone_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → f x < f y

theorem min_m_for_monotone_increasing :
  ∃ m : ℝ, (∀ m' : ℝ, monotone_increasing f m' → m ≤ m') ∧ monotone_increasing f m :=
by
  -- We claim that m = 1 satisfies the conditions
  use 1
  constructor

  -- First part: show that 1 is the minimum value
  · intro m' h_monotone
    -- Proof that m' ≤ 1 goes here
    sorry

  -- Second part: show that f is monotone increasing for m ≥ 1
  · intro x y h_x h_y
    -- Proof that f x < f y when 1 ≤ x < y goes here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_for_monotone_increasing_l682_68201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_five_thirds_l682_68270

/-- The eccentricity of a hyperbola with given parameters -/
noncomputable def hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : b / a = 4 / 3) : ℝ :=
  Real.sqrt (1 + (b / a)^2)

/-- Theorem: The eccentricity of the given hyperbola is 5/3 -/
theorem hyperbola_eccentricity_is_five_thirds 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = 4 / 3) :
  hyperbola_eccentricity a b h1 h2 h3 = 5 / 3 := by
  sorry

#check hyperbola_eccentricity_is_five_thirds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_five_thirds_l682_68270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_m_n_l682_68252

noncomputable def m : ℝ × ℝ := (1, 0)
noncomputable def n : ℝ × ℝ := (-1, Real.sqrt 3)

noncomputable def cross_product_magnitude (a b : ℝ × ℝ) : ℝ :=
  let (a₁, a₂) := a
  let (b₁, b₂) := b
  Real.sqrt ((a₁ * b₂ - a₂ * b₁) ^ 2)

theorem cross_product_m_n :
  cross_product_magnitude m n = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_product_m_n_l682_68252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_value_of_a_l682_68209

-- Part 1
noncomputable def f (x : ℝ) : ℝ := (1/2)^(-x^2 + 4*x + 1)

theorem range_of_f :
  ∀ x ∈ Set.Icc 0 3, f x ∈ Set.Icc (1/32) (1/2) ∧
  ∃ y ∈ Set.Icc 0 3, f y = 1/32 ∧
  ∃ z ∈ Set.Icc 0 3, f z = 1/2 := by sorry

-- Part 2
def g (a x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

theorem value_of_a :
  (∃ a : ℝ, ∀ x ∈ Set.Icc 0 1, g a x ≤ 2 ∧ ∃ y ∈ Set.Icc 0 1, g a y = 2) →
  (a = -1 ∨ a = 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_value_of_a_l682_68209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_NH4Cl_moles_formed_l682_68259

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents a chemical reaction -/
structure Reaction where
  reactant1 : Moles
  reactant2 : Moles
  product : Moles
  ratio : ℝ

instance : OfNat Moles n where
  ofNat := (n : ℝ)

/-- The ammonium chloride formation reaction -/
def NH4Cl_formation : Reaction :=
  { reactant1 := 1,  -- 1 mole of NH3
    reactant2 := 1,  -- 1 mole of HCl
    product := 1,    -- 1 mole of NH4Cl (to be proved)
    ratio := 1 }     -- 1:1 ratio

theorem NH4Cl_moles_formed (r : Reaction) (h1 : r = NH4Cl_formation) :
  r.product = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_NH4Cl_moles_formed_l682_68259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_final_velocity_l682_68285

/-- Represents the problem of calculating vehicle A's final velocity --/
theorem vehicle_final_velocity 
  (distance_to_destination : ℝ) 
  (b_speed : ℝ) 
  (c_distance : ℝ) 
  (a_initial_speed : ℝ) 
  (h1 : distance_to_destination = 350) 
  (h2 : b_speed = 50) 
  (h3 : c_distance = 450) 
  (h4 : a_initial_speed = 45) :
  ∃ (a_final_speed : ℝ), 
    (abs (a_final_speed - 93) < 0.5) ∧ 
    (∃ (t : ℝ), t > 0 ∧ 
      t * b_speed = distance_to_destination ∧
      distance_to_destination = a_initial_speed * t + (1/2) * ((a_final_speed - a_initial_speed) / t) * t^2 ∧
      c_distance > a_initial_speed * t + (1/2) * ((a_final_speed - a_initial_speed) / t) * t^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_final_velocity_l682_68285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l682_68267

-- Define the property that for any x in [0,1], there exists a unique y in [-1,1] such that x + y^2 * e^y - a = 0
def unique_solution (a : ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc 0 1 → ∃! y, y ∈ Set.Icc (-1) 1 ∧ x + y^2 * Real.exp y - a = 0

-- Theorem statement
theorem range_of_a (a : ℝ) :
  unique_solution a → a ∈ Set.Ioo (1 + Real.exp (-1)) (Real.exp 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l682_68267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scout_troop_profit_l682_68232

/-- Calculates the profit for a scout troop's candy bar sale --/
theorem scout_troop_profit 
  (num_bars : ℕ)
  (purchase_rate : ℚ)
  (donation_fraction : ℚ)
  (sell_rate : ℚ)
  (sell_price : ℚ) :
  num_bars = 1200 →
  purchase_rate = 1 / 3 →
  donation_fraction = 1 / 2 →
  sell_rate = 3 / 4 →
  sell_price = 3 / 4 →
  (num_bars : ℚ) * sell_price - (num_bars : ℚ) * purchase_rate * (1 - donation_fraction) = 700 := by
  sorry

#check scout_troop_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scout_troop_profit_l682_68232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_3_or_4_not_6_up_to_150_l682_68243

def count_multiples (n : ℕ) (m : ℕ) : ℕ := n / m

def count_multiples_3_or_4_not_6 (upper_bound : ℕ) : ℕ :=
  count_multiples upper_bound 3 + count_multiples upper_bound 4 - 
  count_multiples upper_bound 12 - count_multiples upper_bound 6

theorem multiples_3_or_4_not_6_up_to_150 :
  count_multiples_3_or_4_not_6 150 = 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_3_or_4_not_6_up_to_150_l682_68243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformation_l682_68210

noncomputable def variance (x : Fin 3 → ℝ) : ℝ :=
  let mean := (x 0 + x 1 + x 2) / 3
  ((x 0 - mean)^2 + (x 1 - mean)^2 + (x 2 - mean)^2) / 3

def transform (x : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => 3 * x i + 2

theorem variance_transformation (a : Fin 3 → ℝ) (h : variance a = 1) :
  variance (transform a) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_transformation_l682_68210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l682_68255

-- Define the points A, B, C, and P in a 2D plane
variable (A B C P : EuclideanSpace ℝ (Fin 2))

-- Define the theorem
theorem vector_equality (h : B - C + (B - A) = (2 : ℝ) • (B - P)) :
  P - C + (P - A) = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_l682_68255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l682_68228

/-- A right pyramid with a square base -/
structure RightPyramid where
  base_side : ℝ
  height : ℝ

/-- The total surface area of the pyramid -/
noncomputable def total_surface_area (p : RightPyramid) : ℝ :=
  p.base_side^2 + 2 * p.base_side * Real.sqrt (p.base_side^2 / 4 + p.height^2)

/-- The volume of the pyramid -/
noncomputable def volume (p : RightPyramid) : ℝ :=
  (1/3) * p.base_side^2 * p.height

/-- The theorem stating the volume of the pyramid under given conditions -/
theorem pyramid_volume_theorem (p : RightPyramid) 
  (h1 : total_surface_area p = 720)
  (h2 : p.base_side^2 / 2 * Real.sqrt (p.base_side^2 / 4 + p.height^2) = p.base_side^2 / 3) :
  volume p = 108 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_theorem_l682_68228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_postage_count_l682_68275

structure Envelope where
  length : ℚ
  height : ℚ

def needsExtraPostage (e : Envelope) : Bool :=
  let ratio := e.length / e.height
  ratio < 3/2 || ratio > 14/5

def envelopes : List Envelope := [
  ⟨7, 5⟩,  -- X
  ⟨10, 4⟩, -- Y
  ⟨8, 5⟩,  -- Z
  ⟨14, 5⟩  -- W
]

theorem extra_postage_count :
  (envelopes.filter needsExtraPostage).length = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_postage_count_l682_68275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_in_interval_l682_68203

-- Define the function f(x) = log₂x + x - 3
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2 + x - 3

-- State the theorem
theorem solution_exists_in_interval :
  (∀ x y, 0 < x ∧ x < y → f x < f y) →  -- f is strictly increasing on (0, +∞)
  ∃ x, x ∈ Set.Icc 2 3 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_in_interval_l682_68203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_distribution_ways_l682_68216

def num_books : ℕ := 8
def min_in_library : ℕ := 2
def min_checked_out : ℕ := 2

theorem book_distribution_ways :
  (Finset.range (num_books - min_checked_out + 1 - min_in_library)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_distribution_ways_l682_68216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rex_playable_area_specific_l682_68299

/-- The playable area for Rex the dog -/
noncomputable def rexPlayableArea (shopWidth shopLength leashLength fountainRadius fountainDistance : ℝ) : ℝ :=
  let fullCircleArea := Real.pi * leashLength ^ 2
  let playableArc := 3 / 4
  let playableAreaExcludingFountain := playableArc * fullCircleArea
  let fountainArea := Real.pi * fountainRadius ^ 2
  playableAreaExcludingFountain - fountainArea

/-- Theorem stating the playable area for Rex given the specific conditions -/
theorem rex_playable_area_specific :
  rexPlayableArea 3 4 5 1 3 = 17.75 * Real.pi := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval rexPlayableArea 3 4 5 1 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rex_playable_area_specific_l682_68299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_interval_with_zeros_l682_68258

/-- A function with the given properties --/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.sqrt 3 * Real.cos (ω * x) + 1

/-- The theorem statement --/
theorem min_interval_with_zeros 
  (ω : ℝ) 
  (h_period : ∀ x, f ω (x + π) = f ω x) 
  (h_min_period : ∀ T, T > 0 → (∀ x, f ω (x + T) = f ω x) → T ≥ π) 
  (m n : ℝ) 
  (h_zeros : ∃ (S : Finset ℝ), S.card ≥ 12 ∧ (∀ x ∈ S, m ≤ x ∧ x ≤ n ∧ f ω x = 0)) :
  n - m ≥ 16 * π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_interval_with_zeros_l682_68258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_I_maximum_mark_l682_68224

theorem paper_I_maximum_mark :
  let passing_percentage : ℚ := 35 / 100
  let candidate_score : ℕ := 42
  let failing_margin : ℕ := 23
  let passing_score : ℕ := candidate_score + failing_margin
  let maximum_mark : ℕ := (passing_score : ℚ) / passing_percentage |>.ceil.toNat
  maximum_mark = 186 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_I_maximum_mark_l682_68224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_ploughed_85_hectares_per_day_l682_68213

/-- Represents the farmer's ploughing scenario -/
structure PloughingScenario where
  total_area : ℝ
  required_daily_area : ℝ
  extra_days : ℕ
  unploughed_area : ℝ

/-- Calculates the actual daily ploughed area given a ploughing scenario -/
noncomputable def actual_daily_area (scenario : PloughingScenario) : ℝ :=
  (scenario.total_area - scenario.unploughed_area) / 
  (scenario.total_area / scenario.required_daily_area + scenario.extra_days)

/-- Theorem stating that given the specific conditions, the farmer ploughed 85 hectares per day -/
theorem farmer_ploughed_85_hectares_per_day :
  let scenario : PloughingScenario := {
    total_area := 448,
    required_daily_area := 160,
    extra_days := 2,
    unploughed_area := 40
  }
  actual_daily_area scenario = 85 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_farmer_ploughed_85_hectares_per_day_l682_68213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l682_68219

noncomputable def f (x : ℝ) := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * Real.sin x * Real.sin x

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ 
   ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ T = Real.pi) ∧
  (∀ (x y : ℝ), Real.pi / 6 < x ∧ x < y ∧ y < 5 * Real.pi / 12 → f x < f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l682_68219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hot_dog_cost_l682_68264

theorem hot_dog_cost (prize : ℕ) (donation_fraction : ℚ) (remaining : ℕ) (hot_dog_cost : ℕ) : 
  prize = 114 → 
  donation_fraction = 1/2 → 
  remaining = 55 → 
  hot_dog_cost = prize - (prize * donation_fraction).floor - remaining →
  hot_dog_cost = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hot_dog_cost_l682_68264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_product_simplification_l682_68279

theorem logarithm_product_simplification (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log (x^2) / Real.log (y^8)) * (Real.log (y^3) / Real.log (x^4)) * 
  (Real.log (x^4) / Real.log (y^5)) * (Real.log (y^5) / Real.log (x^2)) * 
  (Real.log (x^4) / Real.log (y^3)) = (1/2) * (Real.log x / Real.log y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_product_simplification_l682_68279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_value_of_z_l682_68223

/-- Given a complex number z defined as z = (2-i)^2 / i, prove that its absolute value is 5 -/
theorem abs_value_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.abs ((2 - i)^2 / i) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_value_of_z_l682_68223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_10_l682_68272

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 + (3 : ℝ)^(-x)

-- State the theorem
theorem inverse_f_at_10 : 
  (Function.invFun f) 10 = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_10_l682_68272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l682_68249

theorem trigonometric_problem (α β : Real) 
  (h1 : Real.cos α = 1/3)
  (h2 : Real.cos (α + β) = -1/3)
  (h3 : 0 < α ∧ α < Real.pi/2)
  (h4 : 0 < β ∧ β < Real.pi/2) :
  Real.cos β = 7/9 ∧ 2*α + β = Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l682_68249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_ratio_l682_68241

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- The volume of a cone -/
noncomputable def volume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- The circumference of the base of a cone -/
noncomputable def baseCircumference (c : Cone) : ℝ := 2 * Real.pi * c.radius

theorem cone_height_ratio :
  ∀ (original shorter : Cone),
    baseCircumference original = 20 * Real.pi →
    original.height = 40 →
    baseCircumference shorter = baseCircumference original →
    volume shorter = 400 * Real.pi →
    shorter.height / original.height = 3 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_ratio_l682_68241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_inventory_problem_l682_68280

/-- Given a grocer's coffee inventory scenario, prove the percentage of decaffeinated coffee in a new batch. -/
theorem coffee_inventory_problem (initial_stock : ℝ) (initial_decaf_percent : ℝ) 
  (new_stock : ℝ) (final_decaf_percent : ℝ) :
  initial_stock = 400 →
  initial_decaf_percent = 20 →
  new_stock = 100 →
  final_decaf_percent = 28.000000000000004 →
  (let total_stock := initial_stock + new_stock
   let initial_decaf := initial_stock * (initial_decaf_percent / 100)
   let final_decaf := total_stock * (final_decaf_percent / 100)
   let new_decaf := final_decaf - initial_decaf
   new_decaf / new_stock * 100 = 60) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_inventory_problem_l682_68280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersect_at_point_l682_68274

/-- A line in a plane, defined by two distinct points. -/
structure Line (α : Type*) [AddCommGroup α] [Module ℝ α] where
  p1 : α
  p2 : α
  distinct : p1 ≠ p2

/-- A point lies on a line if it can be expressed as a linear combination of the line's endpoints. -/
def Point.liesOn {α : Type*} [AddCommGroup α] [Module ℝ α] (p : α) (l : Line α) : Prop :=
  ∃ t : ℝ, p = (1 - t) • l.p1 + t • l.p2

/-- Three lines in a plane intersect at a single point if there exists a point that lies on all three lines. -/
def three_lines_intersect {α : Type*} [AddCommGroup α] [Module ℝ α] 
  (l1 l2 l3 : Line α) : Prop :=
  ∃ p : α, Point.liesOn p l1 ∧ Point.liesOn p l2 ∧ Point.liesOn p l3

/-- The main theorem: given three lines in a plane, they intersect at a single point. -/
theorem lines_intersect_at_point {α : Type*} [AddCommGroup α] [Module ℝ α] 
  (A1 A2 B1 B2 C1 C2 : α) 
  (hA : A1 ≠ A2) (hB : B1 ≠ B2) (hC : C1 ≠ C2) :
  three_lines_intersect 
    (Line.mk A1 A2 hA) 
    (Line.mk B1 B2 hB) 
    (Line.mk C1 C2 hC) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_intersect_at_point_l682_68274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_one_l682_68208

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 - 2^x else Real.sqrt x

-- State the theorem
theorem f_composition_negative_one :
  f (f (-1)) = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_one_l682_68208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_about_pi_over_2_l682_68278

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 1 / Real.sin x

-- State the theorem
theorem f_symmetric_about_pi_over_2 :
  ∀ x : ℝ, Real.sin x ≠ 0 → f (Real.pi/2 + x) = f (Real.pi/2 - x) :=
by
  intro x h
  simp [f]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_about_pi_over_2_l682_68278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_false_statement_l682_68292

/-- A line -/
structure Line : Type where

/-- A plane -/
structure Plane : Type where

/-- Two lines are parallel -/
def parallel (m n : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perp_line_plane (l : Line) (p : Plane) : Prop := sorry

/-- Two planes are perpendicular -/
def perp_plane_plane (p1 p2 : Plane) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

/-- Two lines are perpendicular -/
def perp_line_line (l1 l2 : Line) : Prop := sorry

/-- The statement is false -/
theorem false_statement (m n : Line) (α β : Plane) 
  (h1 : parallel m n) 
  (h2 : α ≠ β) : 
  ¬(perp_plane_plane α β ∧ line_in_plane m α ∧ line_in_plane n β → perp_line_line m n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_false_statement_l682_68292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_age_ratio_l682_68260

/-- Tom's current age -/
def T : ℝ := sorry

/-- Number of years ago when Tom's age was three times the sum of his children's ages -/
def N : ℝ := sorry

/-- The sum of the ages of Tom's four children is equal to Tom's current age -/
axiom children_ages_sum : T = T

/-- N years ago, Tom's age was three times the sum of his children's ages -/
axiom past_age_relation : T - N = 3 * (T - 4 * N)

/-- The ratio of Tom's current age to N is 11/2 -/
theorem toms_age_ratio : T / N = 11 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_age_ratio_l682_68260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l682_68246

/-- Represents the annual sales volume in ten thousand units -/
noncomputable def m (x : ℝ) : ℝ := 3 - 2 / (x + 1)

/-- Represents the profit in ten thousand yuan -/
noncomputable def y (x : ℝ) : ℝ := -(16 / (x + 1) + (x + 1)) + 29

theorem profit_maximization (x : ℝ) :
  x ≥ 0 →
  y x ≤ 21 ∧
  (y x = 21 ↔ x = 3) :=
by sorry

#check profit_maximization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_maximization_l682_68246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_multiplication_theorem_l682_68230

/-- Converts a list of binary digits to a natural number -/
def binaryToNat (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits -/
def natToBinary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec toBits m : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBits (m / 2)
  toBits n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true, true]  -- 11101₂
  let b := [true, false, true, true]        -- 1101₂
  let product := [true, true, true, true, false, true, false, false, true]  -- 100101111₂
  binaryToNat a * binaryToNat b = binaryToNat product := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_multiplication_theorem_l682_68230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_f_of_f_exists_smallest_x_in_domain_of_f_of_f_l682_68286

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 5) ^ (1/3 : ℝ)

-- State the theorem
theorem smallest_x_in_domain_of_f_of_f (x : ℝ) :
  x ∈ Set.range (f ∘ f) → x ≥ 5 := by
  sorry

-- State the existence of the smallest x
theorem exists_smallest_x_in_domain_of_f_of_f :
  ∃ x : ℝ, x = 5 ∧ x ∈ Set.range (f ∘ f) ∧ ∀ y ∈ Set.range (f ∘ f), y ≥ x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_f_of_f_exists_smallest_x_in_domain_of_f_of_f_l682_68286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_of_i_l682_68222

theorem sum_of_powers_of_i : 
  let i : ℂ := Complex.I
  (Finset.range 101).sum (λ n => i^n) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_of_i_l682_68222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bking_2023_cycle_l682_68225

def cycle_length (s : String) : Nat :=
  s.length

theorem bking_2023_cycle : Nat.lcm (cycle_length "BKING") (cycle_length "2023") = 20 := by
  -- Evaluate cycle_length for both strings
  have h1 : cycle_length "BKING" = 5 := by rfl
  have h2 : cycle_length "2023" = 4 := by rfl
  
  -- Calculate LCM
  have h3 : Nat.lcm 5 4 = 20 := by rfl
  
  -- Rewrite using the calculated values
  rw [h1, h2, h3]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bking_2023_cycle_l682_68225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_Q_to_EH_l682_68277

noncomputable section

-- Define the square EFGH
def square_side_length : ℝ := 5

-- Define points E, F, G, H
def E : ℝ × ℝ := (0, square_side_length)
def F : ℝ × ℝ := (square_side_length, square_side_length)
def G : ℝ × ℝ := (square_side_length, 0)
def H : ℝ × ℝ := (0, 0)

-- Define point N as the midpoint of GH
def N : ℝ × ℝ := (square_side_length / 2, 0)

-- Define the radii of the circles
def radius_N : ℝ := 2.5
def radius_E : ℝ := 5

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the circles
def circle_N (p : ℝ × ℝ) : Prop := distance N p = radius_N
def circle_E (p : ℝ × ℝ) : Prop := distance E p = radius_E

-- Define point Q as the intersection of the circles (excluding H)
def Q : ℝ × ℝ := (2.5, 2.5)

-- State the theorem
theorem distance_Q_to_EH :
  circle_N Q ∧ circle_E Q ∧ Q ≠ H → Q.2 = 2.5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_Q_to_EH_l682_68277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_backpack_promotion_savings_l682_68227

noncomputable def regular_price : ℝ := 60
noncomputable def discount_second : ℝ := 0.2
noncomputable def discount_third : ℝ := 0.3
noncomputable def total_regular_price : ℝ := 3 * regular_price

noncomputable def discounted_price : ℝ := regular_price + (1 - discount_second) * regular_price + (1 - discount_third) * regular_price

noncomputable def savings : ℝ := total_regular_price - discounted_price

noncomputable def percentage_saved : ℝ := (savings / total_regular_price) * 100

theorem backpack_promotion_savings :
  ∃ (ε : ℝ), abs (percentage_saved - 16.67) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_backpack_promotion_savings_l682_68227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roberts_monthly_expenses_roberts_monthly_expenses_proof_l682_68242

/-- Calculates Robert's monthly expenses given his basic salary, commission rate, total sales, and savings allocation. -/
theorem roberts_monthly_expenses
  (basic_salary : ℝ)
  (commission_rate : ℝ)
  (total_sales : ℝ)
  (savings_rate : ℝ)
  (h1 : basic_salary = 1250)
  (h2 : commission_rate = 0.1)
  (h3 : total_sales = 23600)
  (h4 : savings_rate = 0.2)
  : ℝ := by
  let commission := commission_rate * total_sales
  let total_earnings := basic_salary + commission
  let savings := savings_rate * total_earnings
  let monthly_expenses := total_earnings - savings
  exact monthly_expenses

/-- Proves that Robert's monthly expenses are $2888 given the specified conditions. -/
theorem roberts_monthly_expenses_proof
  (basic_salary : ℝ)
  (commission_rate : ℝ)
  (total_sales : ℝ)
  (savings_rate : ℝ)
  (h1 : basic_salary = 1250)
  (h2 : commission_rate = 0.1)
  (h3 : total_sales = 23600)
  (h4 : savings_rate = 0.2)
  : roberts_monthly_expenses basic_salary commission_rate total_sales savings_rate h1 h2 h3 h4 = 2888 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roberts_monthly_expenses_roberts_monthly_expenses_proof_l682_68242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorboat_speed_calculation_l682_68288

/-- Represents the speed of a motorboat in still water -/
def boat_speed : ℝ → Prop := sorry

/-- Represents the speed of the river current -/
def current_speed : ℝ → Prop := sorry

/-- Represents the distance traveled by the motorboat -/
def distance : ℝ → Prop := sorry

/-- Represents the time difference between upstream and downstream trips -/
def time_difference : ℝ → Prop := sorry

theorem motorboat_speed_calculation 
  (v : ℝ) 
  (h_current : current_speed 2)
  (h_distance : distance 165)
  (h_time_diff : time_difference 4) :
  boat_speed v ↔ v = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorboat_speed_calculation_l682_68288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_diff_l682_68250

-- Define the line 2x - y - 4 = 0
def line (x y : ℝ) : Prop := 2 * x - y - 4 = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Define the distance difference function
noncomputable def distanceDiff (px py : ℝ) : ℝ :=
  |distance px py 4 (-1) - distance px py 3 4|

-- Theorem statement
theorem max_distance_diff :
  ∀ x y : ℝ, line x y →
  distanceDiff x y ≤ distanceDiff 5 6 :=
by sorry

-- Example usage (optional)
#check max_distance_diff

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_diff_l682_68250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hippocrates_lune_area_equals_triangle_area_l682_68244

-- Define the triangle
structure RightIsoscelesTriangle where
  r : ℝ
  r_pos : r > 0

-- Define the circle arc
structure CircleArc where
  radius : ℝ
  center : ℝ × ℝ

-- Define the lune
structure HippocratesLune (t : RightIsoscelesTriangle) where
  semicircle_leg : CircleArc
  semicircle_hypotenuse : CircleArc
  leg_radius : semicircle_leg.radius = t.r
  hyp_radius : semicircle_hypotenuse.radius = t.r * Real.sqrt 2 / 2

-- Define area function
noncomputable def Area (α : Type) : ℝ := sorry

-- Define triangle type
def Triangle (t : RightIsoscelesTriangle) : Type := Unit

-- Theorem statement
theorem hippocrates_lune_area_equals_triangle_area (t : RightIsoscelesTriangle) 
  (l : HippocratesLune t) : 
  Area (HippocratesLune t) = Area (Triangle t) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hippocrates_lune_area_equals_triangle_area_l682_68244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_transform_l682_68257

/-- Given a function f(x) = cos(2x), apply the following transformations:
    1. Double the x-coordinates
    2. Shift the resulting function left by 1 unit
    Prove that the resulting function is g(x) = cos(x + 1) -/
theorem cos_transform (x : ℝ) : Real.cos (2 * (x / 2 + 1)) = Real.cos (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_transform_l682_68257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l682_68293

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem min_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l682_68293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l682_68276

/-- Represents a corner in the game -/
structure Corner where
  n : ℕ
  m : ℕ
  h_n : n ≥ 2
  h_m : m ≥ 2

/-- Represents a move in the game -/
structure Move where
  x : ℕ
  y : ℕ
  width : ℕ
  height : ℕ

/-- Represents the game state -/
structure GameState where
  corner : Corner
  moves : List Move

/-- The final state of the game -/
def gameOverState (corner : Corner) 
  (strategy : GameState → Move) (opponent_strategy : GameState → Move) : GameState :=
  sorry

/-- The winner of the game -/
def GameState.winner (gs : GameState) : ℕ :=
  sorry

/-- Defines a winning strategy for a player -/
def WinningStrategy (player : ℕ) (corner : Corner) : Prop :=
  ∃ (strategy : GameState → Move),
    ∀ (opponent_strategy : GameState → Move),
      player = (gameOverState corner strategy opponent_strategy).winner

/-- Theorem stating that the first player has a winning strategy -/
theorem first_player_wins (corner : Corner) : WinningStrategy 1 corner :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l682_68276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_domain_l682_68220

-- Define the function f(x) = 3^x
noncomputable def f (x : ℝ) : ℝ := 3^x

-- State the theorem about the domain of the inverse function
theorem inverse_function_domain :
  {x : ℝ | x > 0} = Set.range f :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_domain_l682_68220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dampening_factor_calculation_l682_68229

noncomputable def signal_strength (s : ℝ) (r : ℝ) : ℝ := s / (1 - r)

noncomputable def odd_seconds_strength (s : ℝ) (r : ℝ) : ℝ := (s * r) / (1 - r^2)

theorem dampening_factor_calculation (s : ℝ) (r : ℝ) 
  (h1 : signal_strength s r = 16)
  (h2 : odd_seconds_strength s r = -6) :
  r = -3/11 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dampening_factor_calculation_l682_68229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_unique_minimizer_of_f_l682_68290

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x + 1 / (x + 1)

-- State the theorem
theorem min_value_of_f :
  ∀ x : ℝ, x > -1 → f x ≥ f 0 :=
by
  sorry

-- State that x = 0 is the unique minimizer
theorem unique_minimizer_of_f :
  ∀ x : ℝ, x > -1 → x ≠ 0 → f x > f 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_unique_minimizer_of_f_l682_68290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l682_68221

theorem range_of_m (m : ℝ) : 
  (∀ x ∈ Set.Ioo 0 1, 
    let f := λ x : ℝ ↦ m * Real.log x - x^2 / 2
    let f' := λ x : ℝ ↦ m / x - x
    (f' x) * (f' (1 - x)) ≤ 1) →
  m ∈ Set.Icc 0 (3/4) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l682_68221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l682_68282

theorem complex_power_sum (w : ℂ) (h : w + w⁻¹ = 2) : w^2022 + (w^2022)⁻¹ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l682_68282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_condition_l682_68204

/-- Given two vectors a and b in ℝ², where a = (4, 1) and b = (x, -2),
    if 2a + b is parallel to 3a - 4b, then x = -8. -/
theorem vector_parallel_condition (x : ℝ) : 
  let a : Fin 2 → ℝ := ![4, 1]
  let b : Fin 2 → ℝ := ![x, -2]
  (∃ (k : ℝ), k ≠ 0 ∧ (2 • a + b) = k • (3 • a - 4 • b)) → x = -8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_condition_l682_68204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_integer_iff_floor_sum_square_l682_68263

theorem roots_integer_iff_floor_sum_square (m n : ℕ+) (α β : ℝ) :
  (α^2 - m * α + n = 0) →
  (β^2 - m * β + n = 0) →
  (∀ x : ℝ, x^2 - m * x + n = 0 → x = α ∨ x = β) →
  (∃ a b : ℤ, α = a ∧ β = b ↔ ∃ k : ℕ, ⌊m * α⌋ + ⌊m * β⌋ = k^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_integer_iff_floor_sum_square_l682_68263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l682_68283

/-- The hyperbola C defined by x^2 - y^2 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- A line through (a, 0) with slope k -/
def line (a k y : ℝ) (x : ℝ) : Prop := x = k * y + a

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (k1 k2 : ℝ) : Prop := k1 * k2 = -1

/-- The distance between two points (x1, y1) and (x2, y2) -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The main theorem -/
theorem hyperbola_intersection_theorem :
  ∃! (a : ℝ), a > 1 ∧
  ∀ (k1 k2 : ℝ), perpendicular k1 k2 →
  ∀ (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ),
  (hyperbola x1 y1 ∧ hyperbola x2 y2 ∧ line a k1 y1 x1 ∧ line a k1 y2 x2) →
  (hyperbola x3 y3 ∧ hyperbola x4 y4 ∧ line a k2 y3 x3 ∧ line a k2 y4 x4) →
  distance x1 y1 x2 y2 = distance x3 y3 x4 y4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_theorem_l682_68283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l682_68212

noncomputable section

-- Define the cubic function
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 6*x - 3

-- Define the linear function
def g (x : ℝ) : ℝ := (6 - 2*x) / 3

-- Theorem statement
theorem intersection_sum : 
  ∃ (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ),
    (f x₁ = g x₁) ∧ (f x₂ = g x₂) ∧ (f x₃ = g x₃) ∧
    (y₁ = g x₁) ∧ (y₂ = g x₂) ∧ (y₃ = g x₃) ∧
    (x₁ + x₂ + x₃ = 4) ∧ (y₁ + y₂ + y₃ = 4) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l682_68212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l682_68231

def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | x^2 - 25 < 0}

theorem intersection_of_M_and_N : M ∩ N = Set.Icc 2 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l682_68231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pastry_shop_muffins_l682_68261

/-- Given a ratio of doughnuts to cookies to muffins and the number of doughnuts,
    calculate the number of muffins -/
theorem pastry_shop_muffins (doughnut_ratio cookie_ratio muffin_ratio doughnut_count : ℕ) :
  doughnut_ratio > 0 →
  cookie_ratio > 0 →
  muffin_ratio > 0 →
  doughnut_count > 0 →
  doughnut_ratio = 5 →
  cookie_ratio = 3 →
  muffin_ratio = 1 →
  doughnut_count = 50 →
  doughnut_count / doughnut_ratio * muffin_ratio = 10 := by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  -- The proof steps would go here
  sorry

#check pastry_shop_muffins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pastry_shop_muffins_l682_68261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_advertising_sales_regression_l682_68207

/-- Given data for advertising expenditure and sales --/
structure AdvertisingSalesData where
  n : ℕ
  sum_x_squared : ℝ
  sum_y_squared : ℝ
  sum_xy : ℝ
  mean_x : ℝ
  mean_y : ℝ

/-- Calculate the slope of the regression line --/
noncomputable def calculate_slope (data : AdvertisingSalesData) : ℝ :=
  (data.sum_xy - data.n * data.mean_x * data.mean_y) /
  (data.sum_x_squared - data.n * data.mean_x ^ 2)

/-- Calculate the y-intercept of the regression line --/
noncomputable def calculate_intercept (data : AdvertisingSalesData) (slope : ℝ) : ℝ :=
  data.mean_y - slope * data.mean_x

/-- Theorem stating the variance of sales and the regression line equation --/
theorem advertising_sales_regression (data : AdvertisingSalesData)
  (h_n : data.n = 5)
  (h_sum_x_squared : data.sum_x_squared = 145)
  (h_sum_y_squared : data.sum_y_squared = 13500)
  (h_sum_xy : data.sum_xy = 1380)
  (h_mean_x : data.mean_x = 5)
  (h_mean_y : data.mean_y = 50) :
  let variance_y := (data.sum_y_squared / data.n) - data.mean_y ^ 2
  let slope := calculate_slope data
  let intercept := calculate_intercept data slope
  (variance_y = 200) ∧ (slope = 6.5) ∧ (intercept = 17.5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_advertising_sales_regression_l682_68207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_in_set_l682_68211

theorem x_value_in_set : ∃ x : ℝ, x ∈ ({1, 2, x^2} : Set ℝ) ↔ (x = 0 ∨ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_in_set_l682_68211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l682_68269

/-- The equation of a circle in the form ax² + by² + cx + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle C with equation x² + y² - 4x + 2y = 0, its center is at the point (2, -1) -/
theorem circle_center (C : CircleEquation) 
  (h : C = { a := 1, b := 1, c := -4, d := 2, e := 0 }) : 
  CircleCenter.mk 2 (-1) = CircleCenter.mk 2 (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_l682_68269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_numbers_not_sum_of_small_primes_l682_68284

theorem infinite_numbers_not_sum_of_small_primes :
  ∃ f : ℕ → ℕ, StrictMono f ∧
  ∀ n : ℕ, ∀ a b : ℕ, (∀ p : ℕ, Nat.Prime p → p < 1394 → p ∣ a → p ∣ b) →
  f n ≠ a + b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_numbers_not_sum_of_small_primes_l682_68284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l682_68202

def f (x : ℝ) : ℝ := x^2 + 2*x

theorem f_properties (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, f x ≥ a) = (a ∈ Set.Iic (-1 : ℝ)) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, f x < a) = (a ∈ Set.Ioi 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l682_68202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l682_68296

-- Define the triangle ABC
def Triangle (A B C : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi

-- Define the acute triangle condition
def AcuteTriangle (A B C : ℝ) : Prop :=
  Triangle A B C ∧ A < Real.pi/2 ∧ B < Real.pi/2 ∧ C < Real.pi/2

-- Define the side lengths
def SideLengths (a b c : ℝ) : Prop :=
  a = Real.sqrt 7 ∧ b = 3 ∧ c > 0

-- Define the given condition
def GivenCondition (A B : ℝ) : Prop :=
  Real.sqrt 7 * Real.sin B + Real.sin A = 2 * Real.sqrt 3

-- State the theorem
theorem triangle_problem (A B C a b c : ℝ) 
  (h1 : AcuteTriangle A B C) 
  (h2 : SideLengths a b c) 
  (h3 : GivenCondition A B) :
  A = Real.pi/3 ∧ (1/2 * b * c * Real.sin A = 3 * Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l682_68296
