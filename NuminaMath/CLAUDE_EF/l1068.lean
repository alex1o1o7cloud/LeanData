import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_hours_is_five_l1068_106814

/-- Represents the tutoring scenario for Mikaela -/
structure TutoringScenario where
  hourly_rate : ℚ
  first_month_hours : ℚ
  personal_needs_fraction : ℚ
  savings : ℚ

/-- Calculates the additional hours tutored in the second month -/
noncomputable def additional_hours (scenario : TutoringScenario) : ℚ :=
  ((scenario.savings / (1 - scenario.personal_needs_fraction)) - 
   (scenario.hourly_rate * scenario.first_month_hours * 2)) / 
   scenario.hourly_rate

/-- Theorem stating that the additional hours tutored in the second month is 5 -/
theorem additional_hours_is_five (scenario : TutoringScenario) 
  (h1 : scenario.hourly_rate = 10)
  (h2 : scenario.first_month_hours = 35)
  (h3 : scenario.personal_needs_fraction = 4/5)
  (h4 : scenario.savings = 150) :
  additional_hours scenario = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_hours_is_five_l1068_106814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_locus_theorem_l1068_106857

/-- A rhombus in a 2D plane -/
structure Rhombus where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_rhombus : 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 
    (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = 
    (D.1 - A.1)^2 + (D.2 - A.2)^2 ∧
    (A.1 - C.1)^2 + (A.2 - C.2)^2 = 
    (B.1 - D.1)^2 + (B.2 - D.2)^2

/-- Distance between two points in 2D plane -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Check if a point is on a line segment -/
def is_on_segment (p a b : ℝ × ℝ) : Prop :=
  distance a p + distance p b = distance a b

/-- The locus theorem for rhombus -/
theorem rhombus_locus_theorem (ABCD : Rhombus) (M : ℝ × ℝ) :
  (is_on_segment M ABCD.A ABCD.C ∨ is_on_segment M ABCD.B ABCD.D) ↔
  distance M ABCD.A * distance M ABCD.C + 
  distance M ABCD.B * distance M ABCD.D = 
  (distance ABCD.A ABCD.B)^2 := by
  sorry

#check rhombus_locus_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_locus_theorem_l1068_106857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_cos_value_l1068_106806

/-- Given a line with inclination angle α perpendicular to x + 2y - 4 = 0,
    prove that cos(2017π/2 - 2α) = 4/5 -/
theorem perpendicular_line_cos_value (α : ℝ) :
  (∃ m b : ℝ, (m * Real.tan α = -1) ∧ (m * 1 + b * 2 = 4)) →
  Real.cos ((2017 / 2 : ℝ) * π - 2 * α) = 4 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_cos_value_l1068_106806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_dot_product_and_cosine_l1068_106896

/-- Vector in 2D plane -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v w : Vector2D) : ℝ := v.x * w.x + v.y * w.y

/-- Distance between two points in 2D plane -/
noncomputable def distance (v w : Vector2D) : ℝ := Real.sqrt ((v.x - w.x)^2 + (v.y - w.y)^2)

/-- Given points A, B, O, and P -/
def A : Vector2D := ⟨1, 7⟩
def B : Vector2D := ⟨5, 1⟩
def O : Vector2D := ⟨0, 0⟩
def P : Vector2D := ⟨2, 1⟩

/-- M is on the line OP -/
def M_on_OP (m : Vector2D) : Prop := m.y = (1/2) * m.x

/-- Theorem: M(4,2) minimizes MA · MB and cos(∠AMB) = -4√17/17 -/
theorem minimize_dot_product_and_cosine :
  let m : Vector2D := ⟨4, 2⟩
  M_on_OP m ∧
  (∀ n : Vector2D, M_on_OP n → 
    dot_product (Vector2D.mk (A.x - m.x) (A.y - m.y)) (Vector2D.mk (B.x - m.x) (B.y - m.y)) ≤ 
    dot_product (Vector2D.mk (A.x - n.x) (A.y - n.y)) (Vector2D.mk (B.x - n.x) (B.y - n.y))) ∧
  (dot_product (Vector2D.mk (A.x - m.x) (A.y - m.y)) (Vector2D.mk (B.x - m.x) (B.y - m.y)) / 
   (distance A m * distance B m) = -4 * Real.sqrt 17 / 17) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_dot_product_and_cosine_l1068_106896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_equals_sec_plus_tan_l1068_106845

-- Define a geometric sequence (a_n)
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem fourth_term_equals_sec_plus_tan (x : ℝ) (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = Real.cos x →
  a 2 = Real.sin x →
  a 3 = (Real.cos x)⁻¹ →
  a 4 = (Real.cos x)⁻¹ + Real.sin x / Real.cos x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_equals_sec_plus_tan_l1068_106845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_within_interval_value_l1068_106812

/-- Represents a normal distribution with mean μ and variance σ² -/
structure NormalDistribution (μ σ : ℝ) where
  mean : ℝ := μ
  variance : ℝ := σ^2

/-- The probability that a value from a normal distribution falls within one standard deviation of the mean -/
noncomputable def prob_within_one_std : ℝ := 0.6827

/-- The probability that a value from a normal distribution falls within two standard deviations of the mean -/
noncomputable def prob_within_two_std : ℝ := 0.9545

/-- The probability that a value from the given normal distribution N(1, 4) falls within the interval (3, 5) -/
noncomputable def prob_within_interval : ℝ :=
  (prob_within_two_std - prob_within_one_std) / 2

theorem prob_within_interval_value :
  prob_within_interval = 0.1359 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_within_interval_value_l1068_106812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_natural_gas_price_this_year_l1068_106852

/-- The price increase rate of natural gas this year -/
def price_increase_rate : ℝ := 0.25

/-- The gas bill of Xiao Ying's family in December last year (in yuan) -/
def december_bill : ℝ := 96

/-- The reduction in gas consumption in May compared to December (in m³) -/
def consumption_reduction : ℝ := 10

/-- The gas bill of Xiao Ying's family in May this year (in yuan) -/
def may_bill : ℝ := 90

/-- The price of natural gas per cubic meter last year (in yuan/m³) -/
noncomputable def last_year_price : ℝ := 
  let x := december_bill / (may_bill / (1 + price_increase_rate) + consumption_reduction)
  x / (1 + price_increase_rate)

/-- The price of natural gas per cubic meter this year (in yuan/m³) -/
noncomputable def this_year_price : ℝ := last_year_price * (1 + price_increase_rate)

/-- Theorem stating that the price of natural gas this year is 3 yuan/m³ -/
theorem natural_gas_price_this_year : this_year_price = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_natural_gas_price_this_year_l1068_106852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_calculation_l1068_106895

/-- Given a total purchase amount, the cost of tax-free items, and the amount of sales tax,
    calculate the tax rate on taxable purchases. -/
noncomputable def calculate_tax_rate (total_purchase : ℝ) (tax_free_items : ℝ) (sales_tax : ℝ) : ℝ :=
  let taxable_items := total_purchase - tax_free_items
  (sales_tax / taxable_items) * 100

theorem tax_rate_calculation :
  let total_purchase := (25 : ℝ)
  let tax_free_items := (18.7 : ℝ)
  let sales_tax := (0.3 : ℝ)
  let calculated_rate := calculate_tax_rate total_purchase tax_free_items sales_tax
  abs (calculated_rate - 4.76) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_rate_calculation_l1068_106895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_diagonal_of_rectangle_l1068_106808

theorem min_diagonal_of_rectangle (l w : ℝ) :
  l > 0 → w > 0 → l + w = 15 → 
  ∀ l' w', l' > 0 → w' > 0 → l' + w' = 15 → 
  Real.sqrt (l^2 + w^2) ≤ Real.sqrt (l'^2 + w'^2) → 
  Real.sqrt (l^2 + w^2) = Real.sqrt 112.5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_diagonal_of_rectangle_l1068_106808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1068_106868

/-- Line l passing through (-2, -4) with slope 1 -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-2 + (Real.sqrt 2 / 2) * t, -4 + (Real.sqrt 2 / 2) * t)

/-- Curve C with equation y² = 4x -/
def curve_C (x y : ℝ) : Prop := y^2 = 4 * x

/-- The intersection points of line l and curve C -/
def intersection_points (t : ℝ) : Prop :=
  let (x, y) := line_l t
  curve_C x y

/-- The sum of distances from (-2, -4) to the intersection points is 12√2 -/
theorem intersection_distance_sum :
  ∃ t₁ t₂ : ℝ,
    intersection_points t₁ ∧ intersection_points t₂ ∧ t₁ ≠ t₂ ∧
    abs t₁ + abs t₂ = 12 * Real.sqrt 2 := by
  sorry

#check intersection_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1068_106868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1068_106881

theorem relationship_abc : (2 : ℝ)^(1/3) < (4 : ℝ)^(1/2) ∧ (4 : ℝ)^(1/2) < (5 : ℝ)^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1068_106881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypergeometric_probability_l1068_106846

/-- Hypergeometric distribution probability mass function -/
def hypergeometric_pmf (N r n k : ℕ) : ℚ :=
  (Nat.choose r k * Nat.choose (N - r) (n - k)) / Nat.choose N n

/-- Random variable X follows H(3,2,10) -/
def X : ℕ → ℚ := hypergeometric_pmf 10 2 3

theorem hypergeometric_probability :
  X 1 = 7 / 15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypergeometric_probability_l1068_106846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_value_l1068_106836

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

-- State the theorem
theorem tan_2x_value (x : ℝ) :
  (∀ x, deriv f x = 2 * f x) →
  Real.tan (2 * x) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2x_value_l1068_106836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l1068_106880

noncomputable section

-- Define the points
def M : ℝ × ℝ := (1, 0)
def A : ℝ × ℝ := (2, 1)
noncomputable def B : ℝ × ℝ := (0, Real.sqrt 3)

-- Define the line l
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 - M.2 = k * (p.1 - M.1)}

-- Define the line segment AB
def segment_AB : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))}

-- Define the intersection condition
def intersects (l : Set (ℝ × ℝ)) (s : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ l ∧ p ∈ s

-- Define the angle of inclination
noncomputable def angle_of_inclination (k : ℝ) : ℝ := Real.arctan k

-- Theorem statement
theorem angle_of_inclination_range :
  ∀ k : ℝ, intersects (line_l k) segment_AB →
    π/4 ≤ angle_of_inclination k ∧ angle_of_inclination k ≤ 2*π/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l1068_106880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_hit_targets_l1068_106835

/-- The expected number of hit targets when n boys randomly choose from n targets -/
noncomputable def E (n : ℕ) : ℝ := n * (1 - (1 - 1/n)^n)

/-- Theorem stating properties of the expected number of hit targets -/
theorem expected_hit_targets (n : ℕ) (hn : n > 0) :
  E n = n * (1 - (1 - 1/n)^n) ∧ E n ≥ n/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_hit_targets_l1068_106835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1068_106813

/-- A differentiable function f with f'(x) > 2x for all x -/
noncomputable def f : ℝ → ℝ := sorry

/-- The derivative of f -/
noncomputable def f' : ℝ → ℝ := sorry

/-- Assumption that f is differentiable -/
axiom f_differentiable : Differentiable ℝ f

/-- Assumption that f'(x) > 2x for all x -/
axiom f'_gt_2x : ∀ x, f' x > 2 * x

/-- The derivative of f is indeed f' -/
axiom f_deriv : deriv f = f'

theorem solution_set_of_inequality :
  {x : ℝ | f (4 - x) < f x - 8 * x + 16} = {x : ℝ | x > 2} := by
  sorry

#check solution_set_of_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1068_106813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_value_l1068_106887

/-- An equilateral triangle with height 13 -/
structure EquilateralTriangle where
  height : ℝ
  is_equilateral : height = 13

/-- The center of an equilateral triangle -/
noncomputable def center (t : EquilateralTriangle) : ℝ × ℝ := sorry

/-- A point inside the equilateral triangle -/
structure InnerPoint (t : EquilateralTriangle) where
  point : ℝ × ℝ
  is_inside : sorry -- Placeholder for the condition

/-- A circle of radius 1 centered at a point -/
def unit_circle (center : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- The probability that a randomly chosen point X inside the triangle, 
    such that a circle of radius 1 centered at X lies entirely inside the triangle, 
    contains the center of the triangle -/
noncomputable def probability (t : EquilateralTriangle) : ℝ := sorry

theorem probability_value (t : EquilateralTriangle) :
  probability t = Real.sqrt 3 * Real.pi / 121 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_value_l1068_106887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_line_sum_l1068_106833

def P : ℝ × ℝ := (0, 10)
def Q : ℝ × ℝ := (4, 0)
def R : ℝ × ℝ := (10, 0)

def triangle_PQR : Set (ℝ × ℝ) := {P, Q, R}

-- Remove the redefinition of midpoint and slope, as they are likely already defined in Mathlib

def y_intercept (m : ℝ) (A : ℝ × ℝ) : ℝ :=
  A.2 - m * A.1

theorem bisecting_line_sum :
  let M := ((P.1 + R.1) / 2, (P.2 + R.2) / 2)
  let m := (M.2 - Q.2) / (M.1 - Q.1)
  m + y_intercept m Q = -15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_line_sum_l1068_106833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lila_max_cookies_l1068_106873

/-- Represents the number of cookies that can be made given available ingredients and recipe requirements. -/
def max_cookies (sugar_available butter_available : ℚ) : ℕ :=
  let sugar_cookies := (sugar_available / 3) * 10
  let butter_cookies := (butter_available / 2) * 10
  (min sugar_cookies butter_cookies).floor.toNat

/-- Theorem stating the maximum number of cookies Lila can make. -/
theorem lila_max_cookies : max_cookies 15 8 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lila_max_cookies_l1068_106873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_l1068_106874

noncomputable def cube_side_length : ℝ := 10

def reflection_point (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ cube_side_length ∧
  0 ≤ y ∧ y ≤ cube_side_length ∧
  x = 3 ∧ y = 4

noncomputable def path_length (x y : ℝ) : ℝ :=
  Real.sqrt (cube_side_length^2 + x^2 + y^2)

def num_reflections : ℕ := 10

theorem light_path_length (x y : ℝ) 
  (h_reflection : reflection_point x y) : 
  (num_reflections : ℝ) * path_length x y = 50 * Real.sqrt 5 := by
  sorry

#check light_path_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_l1068_106874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_can_calculation_l1068_106837

/-- The cost of a 12-pack of soft drinks before discount -/
noncomputable def pack_cost : ℚ := 299 / 100

/-- The number of cans in a pack -/
def cans_per_pack : ℕ := 12

/-- The discount rate applied to bulk purchases -/
noncomputable def discount_rate : ℚ := 15 / 100

/-- The cost per can before the discount is applied -/
noncomputable def cost_per_can : ℚ := pack_cost / cans_per_pack

theorem cost_per_can_calculation :
  cost_per_can = pack_cost / cans_per_pack :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_can_calculation_l1068_106837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_sets_l1068_106817

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The solution set of an inequality -/
def SolutionSet (f : ℝ → ℝ) (rel : ℝ → ℝ → Prop) : Set ℝ := {x | rel (f x) x}

theorem inequality_solution_sets
  (k : ℝ)
  (f : ℝ → ℝ)
  (h_k_pos : k > 0)
  (h_f_odd : IsOdd f)
  (h_sol_set : SolutionSet f (fun x y => x > k * y) = {x | x < -4 ∨ 0 < x ∧ x < 4}) :
  SolutionSet (f ∘ abs) (fun x y => x ≤ k * y) = {x | x ≥ 4 ∨ x = 0} := by
  sorry

#check inequality_solution_sets

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_sets_l1068_106817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_a_exists_for_slope_condition_l1068_106843

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - 2/x - a * log x

noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 + 2/x^2 - a/x

noncomputable def line_slope (a : ℝ) (x₁ x₂ : ℝ) : ℝ := (f a x₁ - f a x₂) / (x₁ - x₂)

theorem no_a_exists_for_slope_condition (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : a > 0) 
  (h_x₁ : x₁ > 0) 
  (h_x₂ : x₂ > 0) 
  (h_x₁_ne_x₂ : x₁ ≠ x₂)
  (h_crit₁ : f_deriv a x₁ = 0)
  (h_crit₂ : f_deriv a x₂ = 0) :
  line_slope a x₁ x₂ ≠ 2 - a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_a_exists_for_slope_condition_l1068_106843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_in_special_sequence_l1068_106876

/-- A sequence of 8 increasing real numbers -/
def IncreasingSequence : Type := { s : Fin 8 → ℝ // ∀ i j, i < j → s i < s j }

/-- Check if a sequence of 4 numbers is an arithmetic progression with given common difference -/
def isAP (s : Fin 4 → ℝ) (d : ℝ) : Prop :=
  ∀ i : Fin 3, s (i.succ) - s i = d

/-- Check if a sequence of 4 numbers is a geometric progression -/
def isGP (s : Fin 4 → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ i : Fin 3, s (i.succ) / s i = r

/-- Get a subsequence of 4 consecutive numbers starting from index i -/
def getSubsequence (s : IncreasingSequence) (i : Fin 5) : Fin 4 → ℝ :=
  λ j ↦ s.val (i + j)

theorem largest_number_in_special_sequence (s : IncreasingSequence) 
  (h1 : ∃ i j, i ≠ j ∧ isAP (getSubsequence s i) 4 ∧ isAP (getSubsequence s j) 36)
  (h2 : ∃ k, isGP (getSubsequence s k)) :
  s.val 7 = 126 ∨ s.val 7 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_in_special_sequence_l1068_106876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_incircle_tangent_segments_ratio_sum_l1068_106840

/-- Given a triangle with sides a, b, c and segments a₁, b₁, c₁ parallel to sides a, b, c 
    respectively and tangent to the incircle, prove that (a₁/a) + (b₁/b) + (c₁/c) = 1. -/
theorem triangle_incircle_tangent_segments_ratio_sum (a b c a₁ b₁ c₁ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha₁ : a₁ > 0) (hb₁ : b₁ > 0) (hc₁ : c₁ > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_parallel : a₁ < a ∧ b₁ < b ∧ c₁ < c)
  (h_tangent : ∃ r, a₁ + b₁ + c₁ = a + b + c - 2 * r) : 
  a₁ / a + b₁ / b + c₁ / c = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_incircle_tangent_segments_ratio_sum_l1068_106840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_sin_greater_than_one_l1068_106879

theorem negation_of_exists_sin_greater_than_one :
  (¬ ∃ x : ℝ, Real.sin x > 1) ↔ (∀ x : ℝ, Real.sin x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_sin_greater_than_one_l1068_106879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_5_equals_121_l1068_106848

/-- A geometric sequence with special properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)
  property1 : a 2 * a 5 = 3 * a 3
  property2 : (a 4 + 9 * a 7) / 2 = 2

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sum_n (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.a 1 * (1 - (seq.a 2 / seq.a 1) ^ n) / (1 - (seq.a 2 / seq.a 1))

/-- The theorem to be proved -/
theorem sum_5_equals_121 (seq : GeometricSequence) : sum_n seq 5 = 121 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_5_equals_121_l1068_106848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_number_of_odd_degree_vertices_l1068_106884

/-- A graph is represented by its vertex set and a function describing the degree of each vertex. -/
structure Graph where
  V : Type
  degree : V → ℕ

/-- The theorem stating that any graph has an even number of vertices with odd degree. -/
theorem even_number_of_odd_degree_vertices (G : Graph) [Fintype G.V] : 
  Even (Finset.card (Finset.filter (fun v => Odd (G.degree v)) (Finset.univ : Finset G.V))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_number_of_odd_degree_vertices_l1068_106884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_ingot_rooms_l1068_106854

/-- Represents a storage room configuration --/
structure StorageConfig where
  totalRooms : Nat
  occupiedRooms : Finset Nat

/-- Checks if a room number is valid --/
def isValidRoom (config : StorageConfig) (room : Nat) : Prop :=
  1 ≤ room ∧ room ≤ config.totalRooms

/-- Finds the next optimal room for placing an ingot --/
def nextOptimalRoom (config : StorageConfig) : Finset Nat :=
  sorry

/-- The initial configuration with ingots in rooms 1 and 81 --/
def initialConfig : StorageConfig :=
  { totalRooms := 81, occupiedRooms := {1, 81} }

/-- Generates the configuration after placing n ingots --/
def nthConfig (n : Nat) : StorageConfig :=
  sorry

/-- Theorem stating the possible rooms for the 6th ingot --/
theorem sixth_ingot_rooms :
  let sixthConfig := nthConfig 6
  ∀ room, room ∈ nextOptimalRoom sixthConfig →
    room ∈ ({11, 31, 51, 71} : Finset Nat) :=
by sorry

#check sixth_ingot_rooms

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_ingot_rooms_l1068_106854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_a_more_stable_l1068_106802

/-- Represents the stability of a data group -/
structure DataStability where
  variance : ℝ

/-- Compares the stability of two data groups -/
def more_stable (a b : DataStability) : Prop :=
  a.variance < b.variance

/-- Theorem stating that group A is more stable than group B -/
theorem group_a_more_stable (group_a group_b : DataStability)
  (h_a : group_a.variance = 0.2)
  (h_b : group_b.variance = 0.5) :
  more_stable group_a group_b :=
by
  unfold more_stable
  rw [h_a, h_b]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_a_more_stable_l1068_106802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bathroom_module_cost_l1068_106863

/-- Represents the cost of a bathroom module -/
def bathroom_cost (cost : ℕ) : Prop := cost = 12000

/-- The total size of the home in square feet -/
def total_size : ℕ := 2000

/-- The size of a kitchen module in square feet -/
def kitchen_size : ℕ := 400

/-- The size of a bathroom module in square feet -/
def bathroom_size : ℕ := 150

/-- The cost of a kitchen module in dollars -/
def kitchen_cost : ℕ := 20000

/-- The cost per square foot for other modules in dollars -/
def other_cost_per_sqft : ℕ := 100

/-- The total cost of the home in dollars -/
def total_cost : ℕ := 174000

/-- The number of kitchen modules in the home -/
def num_kitchens : ℕ := 1

/-- The number of bathroom modules in the home -/
def num_bathrooms : ℕ := 2

theorem bathroom_module_cost : 
  bathroom_cost 12000 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bathroom_module_cost_l1068_106863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tic_winning_strategy_l1068_106856

-- Define permutations function
def permutations (s : Set ℚ) : Set (ℚ × ℚ × ℚ) :=
  { p | ∃ (x y z : ℚ), x ∈ s ∧ y ∈ s ∧ z ∈ s ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    p = (x, y, z) }

theorem tic_winning_strategy :
  ∃ (a b c : ℚ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b + c = 0 ∧
  ∀ (A B C : ℚ), (A, B, C) ∈ permutations {a, b, c} →
    ∃ (r1 r2 : ℚ), r1 ≠ r2 ∧
    ∀ (x : ℚ), A * x^2 + B * x + C = A * (x - r1) * (x - r2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tic_winning_strategy_l1068_106856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_earned_3000_y_percent_2_years_l1068_106828

/-- Calculate compound interest for a given principal, rate, and time period -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem: Interest earned on 3000 invested at y% for 2 years, compounded annually -/
theorem interest_earned_3000_y_percent_2_years (y : ℝ) :
  compoundInterest 3000 y 2 = 3000 * ((1 + y / 100) ^ 2 - 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_earned_3000_y_percent_2_years_l1068_106828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_transformation_integer_root_exists_l1068_106899

/-- Represents a quadratic trinomial ax^2 + bx + c --/
structure QuadraticTrinomial where
  a : Int := 1
  b : Int
  c : Int

/-- Evaluates the quadratic trinomial at a given x --/
def QuadraticTrinomial.evaluate (q : QuadraticTrinomial) (x : Int) : Int :=
  q.a * x^2 + q.b * x + q.c

/-- Represents a single step in the transformation process --/
inductive TransformationStep
  | ChangeB : (δ : Int) → TransformationStep
  | ChangeC : (δ : Int) → TransformationStep

/-- Applies a transformation step to a quadratic trinomial --/
def applyStep (q : QuadraticTrinomial) (step : TransformationStep) : QuadraticTrinomial :=
  match step with
  | TransformationStep.ChangeB δ => ⟨q.a, q.b + δ, q.c⟩
  | TransformationStep.ChangeC δ => ⟨q.a, q.b, q.c + δ⟩

/-- Checks if a quadratic trinomial has an integer root --/
def hasIntegerRoot (q : QuadraticTrinomial) : Prop :=
  ∃ (x : Int), q.evaluate x = 0

theorem quadratic_transformation_integer_root_exists :
  ∃ (steps : List TransformationStep),
    let initial := QuadraticTrinomial.mk 1 10 20
    let final := QuadraticTrinomial.mk 1 20 10
    let intermediates := List.scanl applyStep initial steps
    (intermediates.getLast? = some final) ∧
    (∃ q ∈ intermediates, hasIntegerRoot q) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_transformation_integer_root_exists_l1068_106899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_7350_l1068_106892

theorem divisors_of_7350 : Nat.card (Nat.divisors 7350) = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_7350_l1068_106892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_squared_differences_l1068_106864

theorem max_min_squared_differences (a b c lambda : ℝ) (h_pos : lambda > 0) (h_sum : a^2 + b^2 + c^2 = lambda) :
  ∃ (f : ℝ), f = min (min ((a - b)^2) ((b - c)^2)) ((c - a)^2) ∧
             f ≤ lambda / 2 ∧
             ∃ (a' b' c' : ℝ), a'^2 + b'^2 + c'^2 = lambda ∧
                                min (min ((a' - b')^2) ((b' - c')^2)) ((c' - a')^2) = lambda / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_squared_differences_l1068_106864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_1999_multiple_l1068_106885

theorem smallest_integer_1999_multiple (n : ℕ) : 
  (∀ m : ℕ, m < n → (1999 * m) % 10000 ≠ 2001) ∧ (1999 * n) % 10000 = 2001 → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_1999_multiple_l1068_106885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_polynomial_real_roots_degree_bound_l1068_106849

/-- A polynomial of degree n with coefficients either 1 or -1 -/
def AlternatingPolynomial (n : ℕ) := 
  { p : Polynomial ℝ // p.degree = n ∧ ∀ i, i ≤ n → p.coeff i = 1 ∨ p.coeff i = -1 }

/-- The property that all roots of a polynomial are real -/
def AllRootsReal (p : Polynomial ℝ) :=
  ∀ x : ℂ, (p.map Complex.ofReal).eval x = 0 → x.im = 0

theorem alternating_polynomial_real_roots_degree_bound 
  (n : ℕ) (p : AlternatingPolynomial n) (h : AllRootsReal p.val) : 
  n ≤ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_polynomial_real_roots_degree_bound_l1068_106849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l1068_106850

def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

theorem set_operations :
  (Set.univ \ (A ∪ B) = {x : ℝ | x ≤ 2 ∨ x ≥ 10}) ∧
  (Set.univ \ (A ∩ B) = {x : ℝ | x < 3 ∨ x ≥ 7}) ∧
  ((Set.univ \ A) ∩ B = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l1068_106850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sikh_percentage_is_ten_percent_l1068_106831

/-- Represents the composition of students in a school -/
structure SchoolComposition where
  total_boys : ℕ
  muslim_percentage : ℚ
  hindu_percentage : ℚ
  other_boys : ℕ

/-- Calculates the percentage of Sikh boys in the school -/
def sikh_percentage (sc : SchoolComposition) : ℚ :=
  let muslim_boys := (sc.muslim_percentage * sc.total_boys : ℚ).floor
  let hindu_boys := (sc.hindu_percentage * sc.total_boys : ℚ).floor
  let sikh_boys := sc.total_boys - muslim_boys - hindu_boys - sc.other_boys
  (sikh_boys : ℚ) / sc.total_boys * 100

/-- Theorem stating that for the given school composition, the percentage of Sikh boys is 10% -/
theorem sikh_percentage_is_ten_percent 
  (sc : SchoolComposition)
  (h1 : sc.total_boys = 850)
  (h2 : sc.muslim_percentage = 44 / 100)
  (h3 : sc.hindu_percentage = 32 / 100)
  (h4 : sc.other_boys = 119) :
  sikh_percentage sc = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sikh_percentage_is_ten_percent_l1068_106831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_difference_l1068_106894

def jeanine_initial : ℕ := 250

def clare_initial : ℚ := -((3 : ℚ) / 5) * jeanine_initial

def jeanine_to_abby : ℚ := (2 : ℚ) / 7
def jeanine_to_lea : ℚ := (5 : ℚ) / 11

def clare_add_fraction : ℚ := -(1 : ℚ) / 4

theorem pencil_difference : 
  let jeanine_final := (jeanine_initial : ℚ) - (jeanine_to_abby * jeanine_initial) - (jeanine_to_lea * jeanine_initial)
  let clare_final := clare_initial ^ 2 + (clare_add_fraction * jeanine_initial)
  ⌊clare_final - jeanine_final⌋ = 22372 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_difference_l1068_106894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_seven_l1068_106871

theorem count_divisible_by_seven : 
  (Finset.filter (fun n => n % 7 = 0) (Finset.range 401 \ Finset.range 200)).card = 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_seven_l1068_106871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l1068_106800

-- Define a random variable following normal distribution
structure NormalDist (μ σ : ℝ) where
  value : ℝ

-- Define the probability function
noncomputable def prob {α : Type} (X : α) (event : α → Prop) : ℝ := sorry

-- Define the cumulative distribution function
noncomputable def cdf {α : Type} (X : α) (x : ℝ) : ℝ := sorry

-- Theorem statement
theorem normal_distribution_probability 
  {σ : ℝ} (ξ : NormalDist 2 σ) 
  (h : prob ξ (λ x => x.value ≤ 4) = 0.84) : 
  prob ξ (λ x => x.value < 0) = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_distribution_probability_l1068_106800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_theorem_l1068_106872

-- Define the dimensions of the prism
variable (a b c : ℝ)

-- Define the conditions
def face_area_1 (a b : ℝ) : ℝ := a * b
def face_area_2 (a c : ℝ) : ℝ := a * c
def face_area_3 (b c : ℝ) : ℝ := b * c

-- Define the volume of the prism
def prism_volume (a b c : ℝ) : ℝ := a * b * c

-- State the theorem
theorem prism_volume_theorem (h1 : face_area_1 a b = 30) 
                              (h2 : face_area_2 a c = 50) 
                              (h3 : face_area_3 b c = 75) : 
  prism_volume a b c = 750 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_volume_theorem_l1068_106872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_expression_l1068_106875

theorem absolute_value_expression (x : ℝ) (h : x = -2500) :
  |abs (abs x - x) - abs x| + x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_expression_l1068_106875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_minus_sin_double_l1068_106805

theorem sin_squared_minus_sin_double (α : ℝ) (h : Real.tan α = 1/2) : 
  Real.sin α ^ 2 - Real.sin (2 * α) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_minus_sin_double_l1068_106805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_l1068_106860

/-- Two intersecting circles with equal radii -/
structure IntersectingCircles where
  radius : ℝ
  shaded_area : ℝ
  unshaded_area : ℝ

/-- The properties of the intersecting circles as described in the problem -/
noncomputable def problem_circles : IntersectingCircles where
  radius := (324 : ℝ).sqrt
  shaded_area := 216 * Real.pi
  unshaded_area := 108 * Real.pi

/-- The condition that the shaded area equals the sum of the two unshaded areas -/
axiom shaded_equals_unshaded_sum : 
  problem_circles.shaded_area = 2 * problem_circles.unshaded_area

/-- The theorem to be proved -/
theorem circle_circumference : 
  2 * Real.pi * problem_circles.radius = 36 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_l1068_106860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1068_106826

-- Define the function f(x) = 1/x^3
noncomputable def f (x : ℝ) : ℝ := 1 / (x^3)

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, y ≠ 0 → ∃ x : ℝ, x ≠ 0 ∧ f x = y :=
by
  -- The proof is skipped for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1068_106826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equations_l1068_106862

noncomputable def a : ℝ × ℝ := (3, 2)
noncomputable def b : ℝ × ℝ := (-1, 2)
noncomputable def c : ℝ × ℝ := (4, 1)

noncomputable def m : ℝ := 5/9
noncomputable def n : ℝ := 8/9

noncomputable def d₁ : ℝ × ℝ := ((20 + 2 * Real.sqrt 5) / 5, (5 + 4 * Real.sqrt 5) / 5)
noncomputable def d₂ : ℝ × ℝ := ((20 - 2 * Real.sqrt 5) / 5, (5 - 4 * Real.sqrt 5) / 5)

theorem vector_equations :
  (a = m • b + n • c) ∧
  (∃ (k : ℝ), d₁ - c = k • (a + b)) ∧
  (∃ (k : ℝ), d₂ - c = k • (a + b)) ∧
  ‖d₁ - c‖ = 1 ∧
  ‖d₂ - c‖ = 1 := by
  sorry

#check vector_equations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equations_l1068_106862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_one_problem_two_l1068_106801

-- Problem 1
theorem problem_one (x y : ℝ) (h : 2 * x + 3 * y - 1 = 0) :
  (9 : ℝ)^x * (27 : ℝ)^y = 3 := by sorry

-- Problem 2
theorem problem_two (x y : ℝ) (h1 : x + y = 4) (h2 : x * y = 2) :
  x^2 + y^2 = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_one_problem_two_l1068_106801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marks_initial_money_l1068_106842

/-- Represents the amount of money Mark had initially -/
noncomputable def initial_money : ℝ := 180

/-- Represents the amount spent at the first store -/
noncomputable def first_store_spending (m : ℝ) : ℝ := m / 2 + 14

/-- Represents the amount spent at the second store -/
noncomputable def second_store_spending (m : ℝ) : ℝ := m / 3 + 16

/-- Theorem stating that Mark's initial money was $180 -/
theorem marks_initial_money :
  initial_money - first_store_spending initial_money - second_store_spending initial_money = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marks_initial_money_l1068_106842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_fourth_difference_quotient_of_sine_l1068_106807

open Real

theorem limit_of_fourth_difference_quotient_of_sine :
  let f (x : Real) := sin x
  let a := Real.pi / 3
  let L := (Real.sqrt 3) / 2
  ∀ ε > 0, ∃ δ > 0, ∀ h, 0 < |h| ∧ |h| < δ →
    |((f (a + 4*h) - 4*f (a + 3*h) + 6*f (a + 2*h) - 4*f (a + h) + f a) / h^4) - L| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_fourth_difference_quotient_of_sine_l1068_106807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_deducible_l1068_106832

-- Define the set S
structure S : Type :=
  (element : Prop)

-- Define pib and maa
def pib : S → Prop := λ s => s.element
def maa : S → Prop := λ s => s.element

-- P1: Every pib is a collection of maas
axiom P1 : ∀ p, pib p → ∃ m, maa m

-- P2: Any two distinct pibs intersect in at most two maas
axiom P2 : ∀ p1 p2, pib p1 → pib p2 → p1 ≠ p2 →
  (∃ m1 m2, maa m1 ∧ maa m2 ∧ m1 ≠ m2 ∧
  (∀ m, maa m → (m = m1 ∨ m = m2)))

-- P3: Every maa belongs to at least two pibs and at most three pibs
axiom P3 : ∀ m, maa m →
  (∃ p1 p2, pib p1 ∧ pib p2 ∧ p1 ≠ p2) ∧
  (∀ p1 p2 p3 p4, pib p1 ∧ pib p2 ∧ pib p3 ∧ pib p4 →
    p1 = p2 ∨ p1 = p3 ∨ p1 = p4 ∨ p2 = p3 ∨ p2 = p4 ∨ p3 = p4)

-- P4: There are exactly five pibs
axiom P4 : ∃ p1 p2 p3 p4 p5,
  pib p1 ∧ pib p2 ∧ pib p3 ∧ pib p4 ∧ pib p5 ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
  p3 ≠ p4 ∧ p3 ≠ p5 ∧
  p4 ≠ p5 ∧
  (∀ p, pib p → (p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4 ∨ p = p5))

-- T1: There are exactly ten maas
def T1 : Prop :=
  ∃ m1 m2 m3 m4 m5 m6 m7 m8 m9 m10,
    maa m1 ∧ maa m2 ∧ maa m3 ∧ maa m4 ∧ maa m5 ∧
    maa m6 ∧ maa m7 ∧ maa m8 ∧ maa m9 ∧ maa m10 ∧
    m1 ≠ m2 ∧ m1 ≠ m3 ∧ m1 ≠ m4 ∧ m1 ≠ m5 ∧ m1 ≠ m6 ∧ m1 ≠ m7 ∧ m1 ≠ m8 ∧ m1 ≠ m9 ∧ m1 ≠ m10 ∧
    m2 ≠ m3 ∧ m2 ≠ m4 ∧ m2 ≠ m5 ∧ m2 ≠ m6 ∧ m2 ≠ m7 ∧ m2 ≠ m8 ∧ m2 ≠ m9 ∧ m2 ≠ m10 ∧
    m3 ≠ m4 ∧ m3 ≠ m5 ∧ m3 ≠ m6 ∧ m3 ≠ m7 ∧ m3 ≠ m8 ∧ m3 ≠ m9 ∧ m3 ≠ m10 ∧
    m4 ≠ m5 ∧ m4 ≠ m6 ∧ m4 ≠ m7 ∧ m4 ≠ m8 ∧ m4 ≠ m9 ∧ m4 ≠ m10 ∧
    m5 ≠ m6 ∧ m5 ≠ m7 ∧ m5 ≠ m8 ∧ m5 ≠ m9 ∧ m5 ≠ m10 ∧
    m6 ≠ m7 ∧ m6 ≠ m8 ∧ m6 ≠ m9 ∧ m6 ≠ m10 ∧
    m7 ≠ m8 ∧ m7 ≠ m9 ∧ m7 ≠ m10 ∧
    m8 ≠ m9 ∧ m8 ≠ m10 ∧
    m9 ≠ m10 ∧
    (∀ m, maa m → (m = m1 ∨ m = m2 ∨ m = m3 ∨ m = m4 ∨ m = m5 ∨ m = m6 ∨ m = m7 ∨ m = m8 ∨ m = m9 ∨ m = m10))

-- T2: There are exactly four maas in each pib
def T2 : Prop :=
  ∀ p, pib p → ∃ m1 m2 m3 m4,
    maa m1 ∧ maa m2 ∧ maa m3 ∧ maa m4 ∧
    m1 ≠ m2 ∧ m1 ≠ m3 ∧ m1 ≠ m4 ∧ m2 ≠ m3 ∧ m2 ≠ m4 ∧ m3 ≠ m4 ∧
    (∀ m, maa m → (m = m1 ∨ m = m2 ∨ m = m3 ∨ m = m4))

-- T3: For each maa, there is exactly one other maa not in any shared pib with it
def T3 : Prop :=
  ∀ m1, maa m1 → ∃! m2, maa m2 ∧ m1 ≠ m2 ∧
    (∀ p, pib p → (pib ⟨m1.element⟩ ↔ ¬pib ⟨m2.element⟩))

theorem not_deducible : ¬(T1 ∨ T2 ∨ T3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_deducible_l1068_106832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l1068_106898

/-- Definition of the ellipse -/
def is_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1

/-- Definition of the foci -/
structure Foci :=
  (F₁ : ℝ × ℝ)
  (F₂ : ℝ × ℝ)

/-- Definition of the intersection points -/
structure IntersectionPoints :=
  (M : ℝ × ℝ)
  (N : ℝ × ℝ)

/-- The line passing through F₁ intersects the ellipse at M and N -/
def line_intersects (f : Foci) (p : IntersectionPoints) : Prop :=
  ∃ (t₁ t₂ : ℝ), 
    p.M = (f.F₁.1 + t₁ * (p.M.1 - f.F₁.1), f.F₁.2 + t₁ * (p.M.2 - f.F₁.2)) ∧
    p.N = (f.F₁.1 + t₂ * (p.N.1 - f.F₁.1), f.F₁.2 + t₂ * (p.N.2 - f.F₁.2)) ∧
    is_ellipse p.M.1 p.M.2 ∧
    is_ellipse p.N.1 p.N.2

/-- The perimeter of the triangle MNF₂ -/
noncomputable def triangle_perimeter (f : Foci) (p : IntersectionPoints) : ℝ :=
  let d1 := ((p.M.1 - p.N.1)^2 + (p.M.2 - p.N.2)^2).sqrt
  let d2 := ((p.M.1 - f.F₂.1)^2 + (p.M.2 - f.F₂.2)^2).sqrt
  let d3 := ((p.N.1 - f.F₂.1)^2 + (p.N.2 - f.F₂.2)^2).sqrt
  d1 + d2 + d3

theorem ellipse_triangle_perimeter 
  (f : Foci) (p : IntersectionPoints) 
  (h1 : is_ellipse f.F₁.1 f.F₁.2) 
  (h2 : is_ellipse f.F₂.1 f.F₂.2)
  (h3 : line_intersects f p) : 
  triangle_perimeter f p = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l1068_106898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_intersection_l1068_106847

noncomputable section

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))
-- Define point D on BC
variable (D : EuclideanSpace ℝ (Fin 2))

-- State the conditions
variable (h1 : ¬ IsRightAngle B A C)
variable (h2 : dist A C ≥ max (dist A B) (dist B C))
variable (h3 : D ∈ sphere A (dist A C / 2))
variable (h4 : D ∈ segmentClosed B C)
variable (h5 : dist A D = 3)
variable (h6 : dist C D = 5)

-- State the theorem
theorem triangle_circle_intersection :
  dist B D = Real.sqrt 34 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_intersection_l1068_106847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_phone_numbers_sum_l1068_106809

theorem central_phone_numbers_sum (phone_numbers : Finset ℕ) : 
  (∀ n ∈ phone_numbers, n ≤ 9999) → 
  (Finset.card phone_numbers ≥ 5001) →
  ∃ a b c, a ∈ phone_numbers ∧ b ∈ phone_numbers ∧ c ∈ phone_numbers ∧ a + b = c :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_phone_numbers_sum_l1068_106809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_regular_tetrahedron_l1068_106853

/-- The surface area of a regular tetrahedron with edge length a is a^2 * √3. -/
theorem surface_area_regular_tetrahedron (a : ℝ) (ha : a > 0) :
  ∃ S : ℝ, S = a^2 * Real.sqrt 3 ∧ S > 0 := by
  -- Let S be the surface area of the tetrahedron
  let S := a^2 * Real.sqrt 3
  
  -- Show that S satisfies the required properties
  use S
  constructor
  · -- Prove S = a^2 * √3
    rfl
  · -- Prove S > 0
    apply mul_pos
    · -- Prove a^2 > 0
      exact pow_pos ha 2
    · -- Prove √3 > 0
      exact Real.sqrt_pos.2 (by norm_num)

-- The actual proof would involve more steps, including:
-- 1. Defining a regular tetrahedron
-- 2. Proving that its faces are equilateral triangles
-- 3. Calculating the area of one face
-- 4. Multiplying by 4 to get the total surface area
-- These steps are omitted for brevity and due to the complexity
-- of formalizing 3D geometry in Lean.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_regular_tetrahedron_l1068_106853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_equals_one_range_of_a_for_f_geq_6_min_value_of_f_l1068_106893

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - a|

-- Part 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x ≥ 4} = Set.Iic (-2) ∪ Set.Ici 2 := by sorry

-- Part 2
theorem range_of_a_for_f_geq_6 :
  (∀ x, f a x ≥ 6) ↔ a ∈ Set.Iic (-3) ∪ Set.Ici 3 := by sorry

-- Helper theorem for the minimum value of f
theorem min_value_of_f (a : ℝ) :
  ∀ x, f a x ≥ 2 * |a| := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_equals_one_range_of_a_for_f_geq_6_min_value_of_f_l1068_106893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1068_106855

theorem problem_statement (x y : ℝ) : 
  x = (2023 : ℝ)^(1012 : ℤ) - (2023 : ℝ)^(-(1012 : ℤ)) → 
  y = (2023 : ℝ)^(1012 : ℤ) + (2023 : ℝ)^(-(1012 : ℤ)) → 
  x^2 - y^2 = -4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1068_106855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_2_eq_14_l1068_106869

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
def g (x : ℝ) : ℝ := 2 * x^2 - 2 * x - 3

-- State the theorem
theorem f_of_g_of_2_eq_14 : f (g 2) = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_g_of_2_eq_14_l1068_106869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1068_106878

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin (x + Real.pi/3) - Real.sqrt 3 * (Real.sin x)^2 + Real.sin x * Real.cos x

theorem f_properties :
  let T : ℝ := Real.pi
  let min_value : ℝ := -2
  let min_points (k : ℤ) : ℝ := k * Real.pi - 5 * Real.pi / 12
  let inverse_domain : Set ℝ := Set.Icc (Real.pi/12) (7*Real.pi/12)
  (∀ x : ℝ, f (x + T) = f x) ∧ 
  (∀ x : ℝ, f x ≥ min_value) ∧
  (∀ k : ℤ, f (min_points k) = min_value) ∧
  (∃ g : ℝ → ℝ, (∀ x ∈ inverse_domain, g (f x) = x) ∧ 
                (∀ y ∈ Set.range f, y ∈ inverse_domain → f (g y) = y) ∧ 
                g 1 = Real.pi/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1068_106878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_right_triangle_tetrahedron_l1068_106859

/-- A triangle is a geometric shape with three sides. -/
structure Triangle where
  -- We'll leave this as a placeholder for now
  mk :: -- constructor

/-- A tetrahedron is a polyhedron with four faces. -/
structure Tetrahedron where
  faces : Fin 4 → Triangle

/-- A right triangle is a triangle with one right angle. -/
structure RightTriangle extends Triangle

/-- Theorem: There exists a tetrahedron where all four faces are right triangles. -/
theorem exists_right_triangle_tetrahedron :
  ∃ (t : Tetrahedron), ∀ (i : Fin 4), ∃ (rt : RightTriangle), t.faces i = rt.toTriangle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_right_triangle_tetrahedron_l1068_106859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_in_S_l1068_106841

-- Define the set of lattice points
def T : Set (ℤ × ℤ) := Set.univ

-- Define the adjacency relation
def adjacent (p q : ℤ × ℤ) : Prop :=
  (abs (p.1 - q.1) + abs (p.2 - q.2) = 1)

-- Define the set S
def S : Set (ℤ × ℤ) :=
  {p | (p.1 + 2 * p.2) % 5 = 0}

-- Theorem statement
theorem exactly_one_in_S :
  ∀ p ∈ T, (p ∈ S ∧ ∀ q ∈ T, adjacent p q → q ∉ S) ∨
           (p ∉ S ∧ ∃! q, q ∈ T ∧ adjacent p q ∧ q ∈ S) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_in_S_l1068_106841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_proof_l1068_106824

/-- Calculates the simple interest for a given principal, rate, and time period. -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculates the compound interest for a given principal, rate, and time period. -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 100) ^ time - principal

/-- Proves that the principal amount is 17000 given the compound and simple interests. -/
theorem principal_amount_proof :
  ∃ (principal rate : ℝ),
    compound_interest principal rate 2 = 11730 ∧
    simple_interest principal rate 2 = 10200 ∧
    principal = 17000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_amount_proof_l1068_106824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptotes_is_sqrt3_l1068_106886

/-- The distance from the focus of a parabola to the asymptotes of a hyperbola -/
noncomputable def distance_focus_to_asymptotes (p h : ℝ → ℝ → Prop) : ℝ :=
  let focus : ℝ × ℝ := (2, 0)
  let asymptote (x y : ℝ) : Prop := x = (Real.sqrt 3 / 3) * y ∨ x = -(Real.sqrt 3 / 3) * y
  Real.sqrt 3

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

theorem distance_focus_to_asymptotes_is_sqrt3 :
  distance_focus_to_asymptotes parabola hyperbola = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_focus_to_asymptotes_is_sqrt3_l1068_106886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_transformation_l1068_106877

-- Define the initial hyperbola equation
def initial_hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 4 = 1

-- Define the transformation function
def transform (f : ℝ → ℝ → Prop) : ℝ → ℝ → Prop :=
  λ x y ↦ f y (-x)

-- Define the resulting hyperbola equation
def resulting_hyperbola (x y : ℝ) : Prop := y^2 / 4 - x^2 / 9 = 1

-- Theorem statement
theorem hyperbola_transformation :
  ∀ x y : ℝ, transform initial_hyperbola x y ↔ resulting_hyperbola x y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_transformation_l1068_106877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_third_quadrant_l1068_106834

noncomputable def complex_number : ℂ := (1 - Complex.I) / (3 + 4 * Complex.I)

theorem complex_number_in_third_quadrant : 
  complex_number.re < 0 ∧ complex_number.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_third_quadrant_l1068_106834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l1068_106890

def is_valid_assignment (x y e f : ℕ) : Prop :=
  ({x, y, e, f} : Finset ℕ) = {1, 2, 3, 4}

def expression_value (x y e f : ℕ) : ℕ :=
  e * x^y - f

theorem max_expression_value :
  ∃ x y e f : ℕ,
    is_valid_assignment x y e f ∧
    expression_value x y e f = 36 ∧
    ∀ x' y' e' f' : ℕ,
      is_valid_assignment x' y' e' f' →
      expression_value x' y' e' f' ≤ 36 := by
  sorry

#check max_expression_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l1068_106890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_expression_l1068_106838

theorem absolute_value_expression (x : ℤ) (h : x = -504) :
  abs (abs (abs x - x) - abs x) - x = 1008 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_expression_l1068_106838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_x_prime_factorization_sum_l1068_106851

theorem min_x_prime_factorization_sum (x y a b e f : ℕ) : 
  (∀ z : ℕ, z > 0 → 13 * z^7 = 17 * y^11 → x ≤ z) →
  x > 0 →
  y > 0 →
  13 * x^7 = 17 * y^11 →
  x = a^e * b^f →
  Nat.Prime a →
  Nat.Prime b →
  a + b + e + f = 18 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_x_prime_factorization_sum_l1068_106851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_light_equation_l1068_106830

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- The equation of a line passing through two points -/
def line_equation (p1 p2 : ℝ × ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (y - p1.2) * (p2.1 - p1.1) = (x - p1.1) * (p2.2 - p1.2)

theorem reflected_light_equation :
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (0, 2)
  let A' : ℝ × ℝ := reflect_y A
  ∀ x y, line_equation A' B x y ↔ 2 * x - y + 2 = 0 :=
by
  intros x y
  unfold line_equation reflect_y
  simp
  -- The actual proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_light_equation_l1068_106830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quotient_1324_div_23_base5_l1068_106888

/-- Converts a base 5 number to base 10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a base 10 number to base 5 --/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else go (m / 5) ((m % 5) :: acc)
    go n []

/-- Represents a number in base 5 with an integer part and a fractional part --/
structure Base5Number where
  intPart : List Nat
  fracPart : List Nat

/-- The quotient of 1324₅ divided by 23₅ in base 5 --/
def quotientInBase5 : Base5Number :=
  { intPart := [3, 1], fracPart := [2, 1] }

/-- Approximates a rational number in base 5 --/
def approxBase5 (q : ℚ) (precision : Nat) : List Nat :=
  let rec go (n : Nat) (r : ℚ) (acc : List Nat) : List Nat :=
    if n = 0 then acc
    else
      let d := Int.floor (r * 5)
      go (n - 1) (r * 5 - d) (acc ++ [d.toNat])
  go precision (q - ⌊q⌋) []

theorem quotient_1324_div_23_base5 :
  let dividend := toBase10 [4, 2, 3, 1]
  let divisor := toBase10 [3, 2]
  let result := (dividend : ℚ) / divisor
  toBase5 (Int.floor result).toNat = quotientInBase5.intPart ∧
  approxBase5 (result - ↑(Int.floor result)) 2 = quotientInBase5.fracPart := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quotient_1324_div_23_base5_l1068_106888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_circle_equation_l1068_106816

/-- A circle in polar coordinates -/
structure PolarCircle where
  a : ℝ
  equation : ℝ → ℝ
  equation_def : equation = fun θ => a * Real.cos θ

/-- The point (3√2, π/4) in polar coordinates -/
noncomputable def point : ℝ × ℝ := (3 * Real.sqrt 2, Real.pi / 4)

theorem polar_circle_equation (C : PolarCircle) (h1 : C.a = 6) :
  (C.equation 0 = 0) ∧ 
  (C.equation (Real.pi / 4) = 3 * Real.sqrt 2) ∧
  (∀ θ, C.equation θ ≤ C.a) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_circle_equation_l1068_106816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insect_legs_total_l1068_106858

/-- The total number of animals on the street -/
def total_animals : ℕ := 300

/-- The proportion of cats on the street -/
def cat_proportion : ℚ := 1/2

/-- The proportion of dogs on the street -/
def dog_proportion : ℚ := 3/10

/-- The proportion of birds on the street -/
def bird_proportion : ℚ := 1/10

/-- The proportion of insects on the street -/
def insect_proportion : ℚ := 1/10

/-- The number of legs a cat has -/
def cat_legs : ℕ := 4

/-- The number of legs a dog has -/
def dog_legs : ℕ := 4

/-- The number of legs a bird has -/
def bird_legs : ℕ := 2

/-- The number of legs an insect has -/
def insect_legs : ℕ := 6

/-- Theorem stating that the total number of insect legs on the street is 180 -/
theorem insect_legs_total : 
  (insect_proportion * ↑total_animals).floor * insect_legs = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_insect_legs_total_l1068_106858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_square_theorem_l1068_106823

theorem base_square_theorem (b : ℕ) : 
  b > 4 → (∃ n : ℕ, 2 * b^2 + 4 = n^2) ↔ b ∈ ({4, 6, 8, 10} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_square_theorem_l1068_106823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_unconnected_cells_l1068_106825

/-- Given an odd number n ≥ 3, we define a grid and a coloring function. -/
def Grid (n : ℕ) := Fin n × Fin n

/-- A coloring function that assigns either black (false) or white (true) to each cell. -/
def Coloring (n : ℕ) := Grid n → Bool

/-- Two cells are adjacent if they share a vertex and have the same color. -/
def adjacent (n : ℕ) (coloring : Coloring n) (a b : Grid n) : Prop :=
  (a.1 = b.1 ∧ (a.2.val + 1 = b.2.val ∨ b.2.val + 1 = a.2.val)) ∨
  (a.2 = b.2 ∧ (a.1.val + 1 = b.1.val ∨ b.1.val + 1 = a.1.val)) ∨
  (a.1.val + 1 = b.1.val ∧ a.2.val + 1 = b.2.val) ∨
  (a.1.val + 1 = b.1.val ∧ b.2.val + 1 = a.2.val) ∧
  coloring a = coloring b

/-- Two cells are connected if there exists a path of adjacent cells between them. -/
def connected (n : ℕ) (coloring : Coloring n) (a b : Grid n) : Prop :=
  ∃ (path : List (Grid n)), path.head? = some a ∧ path.getLast? = some b ∧
  ∀ (i : Fin (path.length - 1)), adjacent n coloring (path[i]) (path[i.val + 1])

/-- The main theorem stating the maximum number of pairwise unconnected cells. -/
theorem max_unconnected_cells (n : ℕ) (h : n ≥ 3) (hodd : Odd n) :
  ∃ (M : ℕ) (coloring : Coloring n),
    M = (n + 1)^2 / 4 + 1 ∧
    ∃ (cells : Finset (Grid n)),
      cells.card = M ∧
      ∀ (a b : Grid n), a ∈ cells → b ∈ cells → a ≠ b →
        ¬connected n coloring a b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_unconnected_cells_l1068_106825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_x_percentage_l1068_106803

/-- Represents the composition of a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  other : ℝ
  composition_sum : ryegrass + other = 1

/-- The final mixture composed of X and Y -/
def FinalMixture (x y : SeedMixture) (p : ℝ) : SeedMixture where
  ryegrass := p * x.ryegrass + (1 - p) * y.ryegrass
  other := p * x.other + (1 - p) * y.other
  composition_sum := by sorry

/-- The theorem stating the percentage of mixture X in the final mixture -/
theorem mixture_x_percentage
  (x : SeedMixture)
  (y : SeedMixture)
  (h_x : x.ryegrass = 0.4 ∧ x.other = 0.6)
  (h_y : y.ryegrass = 0.25 ∧ y.other = 0.75)
  (p : ℝ)
  (h_final : (FinalMixture x y p).ryegrass = 0.32) :
  ∃ (ε : ℝ), abs (p - 0.4666) < ε ∧ ε > 0 := by
  sorry

#check mixture_x_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_x_percentage_l1068_106803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_intervals_max_value_g_l1068_106815

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := -x^3 + m*x^2 - m

-- Define the function g
def g (m : ℝ) (x : ℝ) : ℝ := |f m x|

-- Part 1: Decreasing intervals when m = 1
theorem decreasing_intervals :
  ∀ x : ℝ, (x < 0 ∨ x > 2/3) → (deriv (f 1)) x < 0 := by sorry

-- Part 2: Maximum value of g on [0, m]
theorem max_value_g (m : ℝ) (h : m > 0) :
  ∃ y_max : ℝ, 
    (∀ x : ℝ, x ∈ Set.Icc 0 m → g m x ≤ y_max) ∧
    (m ≥ 3 * Real.sqrt 6 / 2 → y_max = 4/27 * m^3 - m) ∧
    (m < 3 * Real.sqrt 6 / 2 → y_max = m) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_intervals_max_value_g_l1068_106815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_circle_l1068_106870

/-- Given a circle and a point R, this theorem proves that the locus of midpoints 
    of segments from R to points on the circle forms another circle. -/
theorem midpoint_locus_circle 
  (C : ℝ × ℝ) 
  (R : ℝ × ℝ) 
  (radius : ℝ) 
  (h_C : C = (0, 15))
  (h_R : R = (0, 0))
  (h_radius : radius = 10)
  (h_distance : Real.sqrt ((C.1 - R.1)^2 + (C.2 - R.2)^2) = 15) :
  ∃ (center : ℝ × ℝ) (new_radius : ℝ),
    center = (0, 10) ∧ 
    new_radius = 10/3 ∧
    ∀ (S : ℝ × ℝ),
      (S.1 - C.1)^2 + (S.2 - C.2)^2 = radius^2 →
      let M := ((R.1 + S.1)/2, (R.2 + S.2)/2)
      (M.1 - center.1)^2 + (M.2 - center.2)^2 = new_radius^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_locus_circle_l1068_106870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l1068_106821

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 + 5*x + 1 = 0

-- Define the roots of the equation
def roots (x₁ x₂ : ℝ) : Prop := quadratic_equation x₁ ∧ quadratic_equation x₂

-- Define the expression to be evaluated
noncomputable def expression (x₁ x₂ : ℝ) : ℝ := 
  (x₁ * Real.sqrt 6 / (1 + x₂))^2 + (x₂ * Real.sqrt 6 / (1 + x₁))^2

-- Theorem statement
theorem expression_value (x₁ x₂ : ℝ) : 
  roots x₁ x₂ → expression x₁ x₂ = 220 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l1068_106821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_not_covered_by_square_l1068_106810

/-- The diameter of the circle in meters -/
noncomputable def circle_diameter : ℝ := 12

/-- The radius of the circle in meters -/
noncomputable def circle_radius : ℝ := circle_diameter / 2

/-- The area of the entire circle in square meters -/
noncomputable def circle_area : ℝ := Real.pi * circle_radius ^ 2

/-- The side length of the square in meters -/
noncomputable def square_side : ℝ := circle_radius

/-- The area of the square in square meters -/
noncomputable def square_area : ℝ := square_side ^ 2

/-- The area of half the circle in square meters -/
noncomputable def half_circle_area : ℝ := circle_area / 2

/-- The theorem stating the area not covered by the square -/
theorem area_not_covered_by_square : 
  half_circle_area = 18 * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_not_covered_by_square_l1068_106810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_distance_coincident_tangents_a_range_l1068_106882

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 2*x + a else Real.log x

def tangent_slope (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 2*x + 2 else 1/x

theorem perpendicular_tangents_distance (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ → x₂ < 0 →
  tangent_slope a x₁ * tangent_slope a x₂ = -1 →
  x₂ - x₁ ≥ 1 := by sorry

theorem coincident_tangents_a_range (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ < 0 → 0 < x₂ →
  (∃ k b : ℝ, ∀ x : ℝ, 
    k * (x - x₁) + f a x₁ = k * (x - x₂) + f a x₂) →
  a > -Real.log 2 - 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_distance_coincident_tangents_a_range_l1068_106882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_of_solutions_l1068_106820

noncomputable def f (n : ℝ) : ℝ :=
  if n < 0 then n^2 + 3*n + 2 else 3*n - 15

theorem positive_difference_of_solutions : ∃ a₁ a₂ : ℝ,
  f (-3) + f 3 + f a₁ = 0 ∧
  f (-3) + f 3 + f a₂ = 0 ∧
  a₁ ≠ a₂ ∧
  |a₁ - a₂| = 25/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_of_solutions_l1068_106820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_theorem_l1068_106866

def jo_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def kate_sum (n : ℕ) : ℕ := 
  (n / 10) * 600

theorem sum_difference_theorem : 
  |Int.ofNat (kate_sum 150) - Int.ofNat (jo_sum 150)| = 2325 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_theorem_l1068_106866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_T_l1068_106811

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define the problem setup
structure ProblemSetup where
  C : Circle
  L : Line
  O : ℝ × ℝ
  P : ℝ × ℝ
  K : Circle
  T : ℝ × ℝ

-- Helper functions (not implemented, just for context)
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry
def is_tangent_point (p : ℝ × ℝ) (c1 c2 : Circle) : Prop := sorry
def perpendicular (l1 l2 : Line) : Prop := sorry
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop := sorry
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

-- Define the conditions
def Conditions (setup : ProblemSetup) : Prop :=
  (setup.C.center = setup.O) ∧
  (setup.L.point1 = setup.O) ∧
  (point_on_line setup.P setup.L) ∧
  (setup.K.center = setup.P) ∧
  (setup.K.radius = distance setup.P setup.O) ∧
  (is_tangent_point setup.T setup.C setup.K)

-- Define the locus of T
def LocusOfT (setup : ProblemSetup) : Prop :=
  ∃ (X : ℝ × ℝ),
    (point_on_circle X setup.C) ∧
    (point_on_line X setup.L) ∧
    (is_tangent_point setup.T setup.C setup.C) ∧
    (perpendicular (Line.mk setup.O X) (Line.mk setup.O setup.T))

-- State the theorem
theorem locus_of_T (setup : ProblemSetup) :
  Conditions setup → LocusOfT setup :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_T_l1068_106811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_face_product_sum_maximum_l1068_106897

theorem cube_face_product_sum_maximum : 
  ∀ (a b c d e f : ℕ), 
    ({a, b, c, d, e, f} : Finset ℕ) = {3, 4, 5, 6, 7, 8} →
    (a + b) * (c + d) * (e + f) ≤ 1331 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_face_product_sum_maximum_l1068_106897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rectangular_parts_l1068_106804

/-- A rectangular sheet with holes -/
structure PerforatedSheet :=
  (n : ℕ)  -- number of holes

/-- The minimum number of rectangular parts a perforated sheet can be divided into -/
def minRectangularParts (sheet : PerforatedSheet) : ℕ :=
  3 * sheet.n + 1

/-- A function to represent the number of parts in a given configuration -/
def numberOfParts (configuration : PerforatedSheet) : ℕ :=
  sorry -- This function is not implemented, but we define it to use in the theorem

/-- Theorem stating the minimum number of rectangular parts -/
theorem min_rectangular_parts (sheet : PerforatedSheet) :
  ∀ m : ℕ, (m ≥ minRectangularParts sheet) →
  ∃ (configuration : PerforatedSheet), 
    (configuration.n = sheet.n) ∧ 
    (numberOfParts configuration = m) :=
by
  sorry -- The proof is omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rectangular_parts_l1068_106804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_term_implies_positive_d_positive_S_implies_increasing_l1068_106889

-- Define the arithmetic sequence and its sum
noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

noncomputable def S (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

-- Statement 1
theorem min_term_implies_positive_d (a₁ : ℝ) (d : ℝ) (h : d ≠ 0) :
  (∃ (n : ℕ), ∀ (m : ℕ), S a₁ d n ≤ S a₁ d m) → d > 0 := by
  sorry

-- Statement 2
theorem positive_S_implies_increasing (a₁ : ℝ) (d : ℝ) (h : d ≠ 0) :
  (∀ (n : ℕ), n > 0 → S a₁ d n > 0) →
  (∀ (n m : ℕ), n < m → S a₁ d n < S a₁ d m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_term_implies_positive_d_positive_S_implies_increasing_l1068_106889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_existence_l1068_106829

open Real

/-- A function that maps real numbers to the interval [-1,1] -/
def BoundedFunction := {f : ℝ → ℝ | ∀ x, f x ∈ Set.Icc (-1) 1}

theorem periodic_function_existence
  (a : ℚ) (b c d : ℝ) (f : ℝ → ℝ)
  (hf : f ∈ BoundedFunction)
  (h : ∀ x : ℝ, f (x + a + b) - f (x + b) = c * (x + 2*a + ⌊x⌋ - 2*⌊x+a⌋ - ⌊b⌋) + d) :
  ∃ T > 0, ∀ x : ℝ, f x = f (x + T) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_existence_l1068_106829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_simplified_l1068_106818

/-- Taxi fare function for Taixing city -/
noncomputable def taxi_fare (x : ℝ) : ℝ :=
  if x ≤ 3 then 5
  else 5 + 1.4 * (x - 3)

/-- Theorem stating the simplified form of the taxi fare for distances greater than 3 km -/
theorem taxi_fare_simplified (x : ℝ) (h : x > 3) : taxi_fare x = 0.8 + 1.4 * x := by
  sorry

#check taxi_fare_simplified

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_fare_simplified_l1068_106818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l1068_106891

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}
def Q : Set ℝ := {y | ∃ x : ℝ, y = -2*x^2 + 2}

-- Theorem statement
theorem intersection_of_P_and_Q : P ∩ Q = Set.Icc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_P_and_Q_l1068_106891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_primes_not_dividing_f_l1068_106839

/-- The polynomial f(x) = x^4 + 2x^3 - 2x^2 - 4x + 4 -/
def f (x : ℤ) : ℤ := x^4 + 2*x^3 - 2*x^2 - 4*x + 4

/-- Theorem stating that there are infinitely many primes p such that
    for all positive integers m, p does not divide f(m) -/
theorem infinitely_many_primes_not_dividing_f :
  ∃ S : Set Nat, (Set.Infinite S) ∧ (∀ p ∈ S, Nat.Prime p) ∧
    (∀ p ∈ S, ∀ m : ℕ, m > 0 → ¬(↑p ∣ f (↑m))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_primes_not_dividing_f_l1068_106839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_C_asymptote_l1068_106844

/-- The hyperbola C is defined by the equation y²/4 - x²/5 = 1 -/
def hyperbola_C (x y : ℝ) : Prop := y^2 / 4 - x^2 / 5 = 1

/-- The asymptote equation of hyperbola C -/
def asymptote_equation (x y : ℝ) : Prop := y = (Real.sqrt 5 / 2) * x ∨ y = -(Real.sqrt 5 / 2) * x

/-- Theorem: The asymptote equation of the hyperbola C is y = ±(√5/2)x -/
theorem hyperbola_C_asymptote :
  ∀ x y : ℝ, hyperbola_C x y → asymptote_equation x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_C_asymptote_l1068_106844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_for_tax_15000_unique_income_for_tax_15000_l1068_106827

/-- Calculates the tax for a given income based on the tax brackets --/
noncomputable def calculate_tax (income : ℝ) : ℝ :=
  let tax1 := min income 20000 * 0.1
  let tax2 := max 0 (min income 50000 - 20000) * 0.15
  let tax3 := max 0 (min income 90000 - 50000) * 0.2
  let tax4 := max 0 (income - 90000) * 0.25
  tax1 + tax2 + tax3 + tax4

/-- Theorem stating that an income of $92,000 results in a tax of $15,000 --/
theorem income_for_tax_15000 :
  calculate_tax 92000 = 15000 := by sorry

/-- Theorem stating that $92,000 is the unique income that results in a tax of $15,000 --/
theorem unique_income_for_tax_15000 :
  ∀ income : ℝ, calculate_tax income = 15000 → income = 92000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_for_tax_15000_unique_income_for_tax_15000_l1068_106827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_upper_bound_l1068_106861

theorem log_sum_upper_bound (x y : ℝ) (h1 : x ≥ y) (h2 : y > 2) :
  Real.log (x / y) / Real.log x + Real.log (y / x) / Real.log y ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_upper_bound_l1068_106861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_gives_alice_zero_l1068_106819

/-- The amount Bob should give to Alice to equalize expenses -/
noncomputable def amount_bob_gives_alice (alice_paid bob_paid charlie_paid : ℚ) : ℚ :=
  let total_expenses := alice_paid + bob_paid + charlie_paid
  let equal_share := total_expenses / 3
  max (equal_share - bob_paid) 0

/-- Theorem: Bob should give $0 to Alice -/
theorem bob_gives_alice_zero :
  amount_bob_gives_alice 120 150 180 = 0 := by
  sorry

-- Use #eval only for computable functions
def computable_amount_bob_gives_alice (alice_paid bob_paid charlie_paid : Int) : Int :=
  let total_expenses := alice_paid + bob_paid + charlie_paid
  let equal_share := total_expenses / 3
  max (equal_share - bob_paid) 0

#eval computable_amount_bob_gives_alice 120 150 180

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_gives_alice_zero_l1068_106819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ap_perp_bc_iff_t_eq_half_l1068_106867

/-- Given a triangle ABC and a point P, prove that AP is perpendicular to BC
    if and only if t = 1/2, under the given conditions. -/
theorem ap_perp_bc_iff_t_eq_half
  (A B C P : ℝ × ℝ)  -- Points in 2D plane
  (t : ℝ)            -- Positive real number
  (h1 : A = (0, 0))  -- A is at the origin
  (h2 : B = (1/t, 0))  -- B is on the x-axis
  (h3 : C = (0, t))    -- C is on the y-axis
  (h4 : P = (1, 4))    -- P's coordinates
  (h5 : t > 0)         -- t is positive
  : (P - A) • (C - B) = 0 ↔ t = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ap_perp_bc_iff_t_eq_half_l1068_106867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_half_equals_fifteen_l1068_106865

-- Define the functions g and f
noncomputable def g (x : ℝ) : ℝ := 1 - 2 * x

noncomputable def f (x : ℝ) : ℝ := 
  if x = 0 then 0  -- Arbitrary value for x = 0
  else (1 - (g⁻¹ x)^2) / (g⁻¹ x)^2

-- Theorem statement
theorem f_one_half_equals_fifteen : f (1/2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_half_equals_fifteen_l1068_106865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_value_l1068_106822

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

-- Define the symmetry condition
def is_symmetric_about_zero (f : ℝ → ℝ) (φ : ℝ) : Prop :=
  ∀ x, f (x + φ) = f (-x + φ)

-- Theorem statement
theorem symmetry_implies_phi_value :
  ∃ φ, is_symmetric_about_zero f φ ∧ φ = π / 6 := by
  sorry

#check symmetry_implies_phi_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_value_l1068_106822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_separate_tangent_lines_circles_through_M_l1068_106883

-- Define the circles and point
def circle_O : Set (ℝ × ℝ) := {(x, y) | x^2 + y^2 = 4}
def circle_C : Set (ℝ × ℝ) := {(x, y) | x^2 + (y - 4)^2 = 1}
def point_M : ℝ × ℝ := (2, 0)

-- Part I: Circles are separate
theorem circles_separate : ∀ p q, p ∈ circle_O → q ∈ circle_C → p ≠ q := by sorry

-- Part II: Tangent lines
def tangent_line1 (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 4 = 0
def tangent_line2 (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 4 = 0

theorem tangent_lines : 
  ∃ l : Set (ℝ × ℝ), (∀ p, p ∈ l → tangent_line1 p.1 p.2 ∨ tangent_line2 p.1 p.2) ∧
  (∀ p, p ∈ l → p ∈ circle_O → (∃! q, q ∈ l ∧ q ≠ p ∧ q ∈ circle_O)) := by sorry

-- Part III: Circles passing through M
def circle_P1 : Set (ℝ × ℝ) := {(x, y) | 5*x^2 + 5*y^2 - 16*x - 8*y + 12 = 0}
def circle_P2 : Set (ℝ × ℝ) := {(x, y) | x^2 + y^2 = 4}

theorem circles_through_M :
  ∃ l : Set (ℝ × ℝ), (∀ p, p ∈ l → p.2 = 4) ∧
  (∃ A B, A ∈ circle_O ∧ B ∈ circle_O ∧ A ≠ B ∧ A ∈ l ∧ B ∈ l ∧
   (point_M ∈ circle_P1 ∨ point_M ∈ circle_P2) ∧
   ((∀ p, p ∈ circle_P1 → dist p A + dist p B = dist A B) ∨
    (∀ p, p ∈ circle_P2 → dist p A + dist p B = dist A B))) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_separate_tangent_lines_circles_through_M_l1068_106883
