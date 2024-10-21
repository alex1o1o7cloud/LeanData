import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_value_l612_61226

theorem tan_beta_value (α β : ℝ) (h_acute_α : 0 < α ∧ α < Real.pi / 2) (h_acute_β : 0 < β ∧ β < Real.pi / 2)
  (h_sin_α : Real.sin α = 4 / 5) (h_tan_diff : Real.tan (α - β) = 2 / 3) : 
  Real.tan β = 6 / 17 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_value_l612_61226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_F_position_l612_61282

-- Define the shape F
structure ShapeF where
  base : ℝ × ℝ → Prop
  stem : ℝ × ℝ → Prop

-- Define the initial position of F
def initial_F : ShapeF :=
  { base := λ (x, y) ↦ x < 0 ∧ y = 0,
    stem := λ (x, y) ↦ x = 0 ∧ y < 0 }

-- Define the transformations
def rotate_90_clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def rotate_half_turn (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- Apply all transformations
def transform (F : ShapeF) : ShapeF :=
  { base := λ p ↦ F.base (rotate_90_clockwise (reflect_y_axis (rotate_half_turn p))),
    stem := λ p ↦ F.stem (rotate_90_clockwise (reflect_y_axis (rotate_half_turn p))) }

-- Theorem statement
theorem final_F_position (F : ShapeF) (h : F = initial_F) :
  let transformed_F := transform F
  (∀ x y, transformed_F.base (x, y) ↔ y > 0 ∧ x = 0) ∧
  (∀ x y, transformed_F.stem (x, y) ↔ x > 0 ∧ y = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_F_position_l612_61282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_octagon_area_equality_l612_61223

/-- The area between inscribed and circumscribed circles of a regular polygon -/
noncomputable def area_between_circles (n : ℕ) (side_length : ℝ) : ℝ :=
  let central_angle := 2 * Real.pi / n
  let apothem := side_length / (2 * Real.tan (central_angle / 2))
  let circumradius := side_length / (2 * Real.sin (central_angle / 2))
  Real.pi * (circumradius^2 - apothem^2)

/-- The theorem stating that the areas between circles for hexagon and octagon are equal -/
theorem hexagon_octagon_area_equality :
  area_between_circles 6 2 = area_between_circles 8 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_octagon_area_equality_l612_61223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_tangency_l612_61203

-- Define the triangle AQM
def Triangle (A Q M : ℝ × ℝ) : Prop :=
  True

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) : Prop :=
  True

-- Define the tangency condition
def IsTangent (circle : Set (ℝ × ℝ)) (line : Set (ℝ × ℝ)) : Prop :=
  True

-- Define the condition for relatively prime integers
def RelativelyPrime (p q : ℕ) : Prop :=
  Nat.Coprime p q

-- Define helper functions
def Perimeter (A Q M : ℝ × ℝ) : ℝ := sorry
def Angle (Q A M : ℝ × ℝ) : ℝ := sorry
def Line (A B : ℝ × ℝ) : Set (ℝ × ℝ) := sorry
def Distance (A B : ℝ × ℝ) : ℝ := sorry

theorem triangle_circle_tangency 
  (A Q M O : ℝ × ℝ) 
  (p q : ℕ) 
  (h1 : Triangle A Q M)
  (h2 : Perimeter A Q M = 300)
  (h3 : Angle Q A M = 120)
  (h4 : Circle O 30)
  (h5 : O.1 ∈ Set.Icc A.1 Q.1 ∧ O.2 = A.2)  -- O is on AQ
  (h6 : IsTangent {x | Circle O 30} (Line A M))
  (h7 : IsTangent {x | Circle O 30} (Line Q M))
  (h8 : Distance O Q = p / q)
  (h9 : RelativelyPrime p q)
  (h10 : p > 0 ∧ q > 0) :
  p + q = 16 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_tangency_l612_61203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_madeline_work_hours_l612_61224

/-- Calculates the number of work hours needed given expenses and hourly wage -/
def work_hours_needed (rent groceries medical utilities savings wage : ℚ) : ℕ :=
  let total_needed := rent + groceries + medical + utilities + savings
  (total_needed / wage).ceil.toNat

/-- Theorem stating Madeline needs to work 138 hours -/
theorem madeline_work_hours :
  work_hours_needed 1200 400 200 60 200 15 = 138 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_madeline_work_hours_l612_61224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l612_61247

theorem triangle_angle_B (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a = b * Real.sin A / Real.sin B →
  b = c * Real.sin B / Real.sin C →
  c = a * Real.sin C / Real.sin A →
  (Real.cos C) / (Real.cos B) = (2 * a - c) / b →
  B = π / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_B_l612_61247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_property_l612_61235

/-- The function f(x) = 1 / (3^x + √3) -/
noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

/-- Theorem: For any x₁ and x₂ such that x₁ + x₂ = 1, f(x₁) + f(x₂) = √3/3 -/
theorem f_sum_property (x₁ x₂ : ℝ) (h : x₁ + x₂ = 1) :
  f x₁ + f x₂ = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_property_l612_61235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_cost_per_foot_l612_61273

theorem fence_cost_per_foot 
  (plot_area : ℝ) 
  (total_cost : ℝ) 
  (h1 : plot_area = 289) 
  (h2 : total_cost = 4012) : 
  total_cost / (4 * Real.sqrt plot_area) = 59 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fence_cost_per_foot_l612_61273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_for_two_prime_factors_l612_61207

/-- The function that constructs the number x based on n -/
def construct_x (n : ℕ) : ℕ :=
  let base_number := 12320
  let appended_threes := 10 * n + 1
  -- Simulate appending threes and interpreting as base 4
  (base_number * 4^appended_threes + (4^appended_threes - 1) / 3) - 1

/-- The property that x has exactly two distinct prime factors -/
def has_two_distinct_prime_factors (x : ℕ) : Prop :=
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ x = p * q

/-- The main theorem -/
theorem unique_n_for_two_prime_factors :
  ∃! n : ℕ, has_two_distinct_prime_factors (construct_x n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_for_two_prime_factors_l612_61207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_red_l612_61201

def total_balls : ℕ := 4
def red_balls : ℕ := 2
def white_balls : ℕ := 2
def drawn_balls : ℕ := 2

def probability_at_least_one_red : ℚ := 5/6

theorem prob_at_least_one_red :
  probability_at_least_one_red = 1 - (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_one_red_l612_61201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_zero_l612_61222

open Real

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + 3^(2/x))

theorem limit_f_at_zero :
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < x ∧ x < δ → |f x - 0| < ε) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, -δ < x ∧ x < 0 → |f x - 1| < ε) ∧
  ¬ ∃ L, ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f x - L| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_zero_l612_61222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_sqrt_N_l612_61216

def N : ℕ := (10^2017 - 1) * 10^2019 + (10^2018 - 1) * 20 + 5

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem sum_of_digits_sqrt_N : digit_sum (Int.toNat (Int.floor (Real.sqrt (N : ℝ)))) = 6056 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_sqrt_N_l612_61216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_video_production_time_l612_61292

/-- Represents the time in hours for various activities in video production -/
structure VideoProduction where
  setupTime : ℚ
  paintingTimePerVideo : ℚ
  cleanupTime : ℚ
  editingTimePerVideo : ℚ
  videosPerBatch : ℕ

/-- Calculates the time to produce a single video given the production details -/
def timePerVideo (vp : VideoProduction) : ℚ :=
  (vp.setupTime + vp.paintingTimePerVideo * vp.videosPerBatch + vp.cleanupTime + vp.editingTimePerVideo * vp.videosPerBatch) / vp.videosPerBatch

/-- Theorem stating that Rachel's video production time is 3 hours per video -/
theorem rachel_video_production_time :
  let vp : VideoProduction := {
    setupTime := 1,
    paintingTimePerVideo := 1,
    cleanupTime := 1,
    editingTimePerVideo := 3/2,
    videosPerBatch := 4
  }
  timePerVideo vp = 3 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rachel_video_production_time_l612_61292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gymnastics_square_formation_l612_61202

/-- Given a team of 48 members, this theorem proves the minimum number of people
    to add or remove to form a square formation. -/
theorem gymnastics_square_formation (team_size : ℕ) (h : team_size = 48) :
  (∃ (add : ℕ), add = 1 ∧ IsSquare (team_size + add) ∧
    ∀ (x : ℕ), x < add → ¬IsSquare (team_size + x)) ∧
  (∃ (remove : ℕ), remove = 12 ∧ IsSquare (team_size - remove) ∧
    ∀ (x : ℕ), x < remove → ¬IsSquare (team_size - x)) :=
by
  sorry

#check gymnastics_square_formation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gymnastics_square_formation_l612_61202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_FM_FN_eq_8_l612_61214

-- Define the parabola C: y² = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line l: y = 2/3(x+2)
def line (x y : ℝ) : Prop := y = 2/3 * (x + 2)

-- Define the focus F of the parabola
def F : ℝ × ℝ := (1, 0)

-- Define the intersection points M and N
noncomputable def M : ℝ × ℝ := sorry
noncomputable def N : ℝ × ℝ := sorry

-- Axioms stating that M and N are on both the parabola and the line
axiom M_on_parabola : parabola M.1 M.2
axiom M_on_line : line M.1 M.2
axiom N_on_parabola : parabola N.1 N.2
axiom N_on_line : line N.1 N.2

-- Define vectors FM and FN
noncomputable def FM : ℝ × ℝ := (M.1 - F.1, M.2 - F.2)
noncomputable def FN : ℝ × ℝ := (N.1 - F.1, N.2 - F.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem: The dot product of FM and FN is 8
theorem dot_product_FM_FN_eq_8 : dot_product FM FN = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_FM_FN_eq_8_l612_61214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_imply_sum_l612_61269

/-- A rational function with integer coefficients and specific asymptotes -/
noncomputable def rational_function (A B C : ℤ) : ℝ → ℝ := λ x => x / (x^3 + A * x^2 + B * x + C)

/-- The denominator of the rational function -/
def denominator (A B C : ℤ) : ℝ → ℝ := λ x => x^3 + A * x^2 + B * x + C

/-- Theorem: If a rational function has vertical asymptotes at -3, 0, and 3, then A + B + C = -9 -/
theorem asymptotes_imply_sum (A B C : ℤ) :
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 → denominator A B C x ≠ 0) →
  (denominator A B C (-3) = 0) →
  (denominator A B C 0 = 0) →
  (denominator A B C 3 = 0) →
  A + B + C = -9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_imply_sum_l612_61269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_is_30_degrees_l612_61255

/-- The acute angle between a line and the positive x-axis -/
noncomputable def acute_angle (a b c : ℝ) : ℝ :=
  Real.arctan (abs (b / a))

theorem line_angle_is_30_degrees :
  let line_equation : ℝ → ℝ → ℝ := λ x y => x - Real.sqrt 3 * y + 2016
  acute_angle 1 (-Real.sqrt 3) 2016 = 30 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_angle_is_30_degrees_l612_61255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l612_61290

/-- Represents the system of equations:
    x + (m + 1)y + m - 2 = 0
    2mx + 4y + 16 = 0 -/
def system_of_equations (m : ℝ) (x y : ℝ) : Prop :=
  x + (m + 1) * y + m - 2 = 0 ∧ 2 * m * x + 4 * y + 16 = 0

/-- The unique solution when m ≠ 1 and m ≠ -2 -/
noncomputable def unique_solution (m : ℝ) : ℝ × ℝ := (6 / (1 - m), (m - 4) / (1 - m))

theorem system_solution (m : ℝ) :
  (m ≠ 1 ∧ m ≠ -2 → ∃! p : ℝ × ℝ, system_of_equations m p.1 p.2 ∧ p = unique_solution m) ∧
  (m = 1 ∨ m = -2 → ¬∃ x y : ℝ, system_of_equations m x y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l612_61290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_multiple_of_18_l612_61289

-- Define the set of four-digit numbers
def four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define the property of being a multiple of 18
def multiple_of_18 (n : ℕ) : Prop := ∃ k : ℕ, n = 18 * k

-- State the theorem
theorem smallest_four_digit_multiple_of_18 : 
  ∀ n : ℕ, four_digit n → multiple_of_18 n → 1008 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_multiple_of_18_l612_61289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l612_61299

-- Define the ellipse structure
structure Ellipse where
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis
  c : ℝ  -- semi-focal distance
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_c_lt_a : c < a
  h_pythagorean : a^2 = b^2 + c^2

-- Define eccentricity
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

-- Define the condition for points M
def inside_condition (e : Ellipse) : Prop :=
  ∀ M : ℝ × ℝ, (M.1 - e.c)^2 + M.2^2 = (M.1 + e.c)^2 + M.2^2 → 
    M.1^2 / e.a^2 + M.2^2 / e.b^2 < 1

-- Theorem statement
theorem eccentricity_range (e : Ellipse) (h : inside_condition e) :
  0 < eccentricity e ∧ eccentricity e < Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l612_61299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_permutation_no_prime_sum_l612_61268

/-- A sequence of natural numbers -/
def Sequence := ℕ → ℕ

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- Checks if a sequence is a permutation of natural numbers -/
def isPermutation (s : Sequence) : Prop :=
  (∀ n : ℕ, ∃ i : ℕ, s i = n) ∧
  (∀ i j : ℕ, i ≠ j → s i ≠ s j)

/-- Checks if a sequence has no consecutive subsequence summing to a prime -/
def noPrimeSum (s : Sequence) : Prop :=
  ∀ start length : ℕ, length > 1 →
    ¬isPrime (Finset.sum (Finset.range length) (λ i => s (start + i)))

/-- There exists a permutation of natural numbers with no consecutive subsequence summing to a prime -/
theorem exists_permutation_no_prime_sum :
  ∃ s : Sequence, isPermutation s ∧ noPrimeSum s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_permutation_no_prime_sum_l612_61268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_bc_l612_61239

variable (a b c : Euclidean 3)

-- Define the conditions
def norm_a : ‖a‖ = 1 := by sorry
def norm_b : ‖b‖ = 1 := by sorry
def norm_c : ‖c‖ = 3 := by sorry
def cross_product_relation : a.cross (b.cross c) + b = 0 := by sorry

-- Define the angle between b and c
noncomputable def angle_bc : ℝ := Real.arccos ((b • c) / (‖b‖ * ‖c‖))

-- Theorem statement
theorem smallest_angle_bc :
  ∃ θ : ℝ, θ = angle_bc ∧ Real.cos θ = 2 * Real.sqrt 2 / 3 ∧
  ∀ φ : ℝ, φ = angle_bc → Real.cos φ ≥ Real.cos θ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_bc_l612_61239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_deduction_is_63_cents_l612_61241

/-- Represents Alicia's hourly wage in dollars -/
def hourly_wage : ℚ := 30

/-- Represents the local tax rate as a decimal -/
def tax_rate : ℚ := 21 / 1000

/-- Represents the number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Calculates the tax deduction in cents -/
def tax_deduction_cents : ℕ := 
  (hourly_wage * tax_rate * cents_per_dollar).floor.toNat

theorem tax_deduction_is_63_cents : 
  tax_deduction_cents = 63 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_deduction_is_63_cents_l612_61241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_sum_of_squares_3k_l612_61252

theorem no_sum_of_squares_3k (k : ℕ) :
  ¬∃ (x y : ℕ), x^2 + y^2 = 3^k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_sum_of_squares_3k_l612_61252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soap_cost_is_16_l612_61274

/-- The original cost of a box of soap given the conditions of the problem -/
noncomputable def original_soap_cost (chlorine_cost : ℚ) (chlorine_discount : ℚ) (soap_discount : ℚ) (total_savings : ℚ) : ℚ :=
  let chlorine_savings := chlorine_cost * chlorine_discount * 3
  let soap_savings := total_savings - chlorine_savings
  let soap_savings_per_box := soap_savings / 5
  soap_savings_per_box / soap_discount

/-- Theorem stating that the original cost of a box of soap is $16 given the problem conditions -/
theorem soap_cost_is_16 :
  original_soap_cost 10 (2/10) (1/4) 26 = 16 := by
  -- Unfold the definition of original_soap_cost
  unfold original_soap_cost
  -- Perform the calculation
  norm_num
  -- QED

-- We can't use #eval for noncomputable functions, so we'll use #reduce instead
#reduce original_soap_cost 10 (2/10) (1/4) 26

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soap_cost_is_16_l612_61274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2022_value_l612_61256

def sequence_a : ℕ → ℤ
  | 0 => 0
  | n+1 => -Int.natAbs (sequence_a n + n.succ)

theorem a_2022_value : sequence_a 2022 = -1011 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2022_value_l612_61256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l612_61254

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the function g
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (1 - 4*m) * Real.sqrt x

-- State the theorem
theorem find_a :
  ∀ (a m : ℝ),
  (a > 0) →
  (a ≠ 1) →
  (∀ x ∈ Set.Icc (-1) 2, f a x ≤ 4) →
  (∃ x ∈ Set.Icc (-1) 2, f a x = 4) →
  (∀ x ∈ Set.Icc (-1) 2, f a x ≥ m) →
  (∃ x ∈ Set.Icc (-1) 2, f a x = m) →
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x < y → g m x < g m y) →
  a = 1/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l612_61254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_range_l612_61294

theorem x_squared_range (x : ℝ) :
  (x + 12).rpow (1/3) - (x - 12).rpow (1/3) = 4 →
  105 < x^2 ∧ x^2 < 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_range_l612_61294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l612_61270

-- Define the logarithm base 10
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define a, b, and c
noncomputable def a : ℝ := log10 (Real.exp 1)
noncomputable def b : ℝ := (log10 (Real.exp 1)) ^ 2
noncomputable def c : ℝ := log10 (Real.sqrt (Real.exp 1))

-- Theorem statement
theorem log_inequality : a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l612_61270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_percentage_change_l612_61261

-- Define the price increase and discounts
def price_increase : ℚ := 1/2
def discount1 : ℚ := 1/5
def discount2 : ℚ := 3/20
def discount3 : ℚ := 1/10

-- Define the function to calculate the final price after all adjustments
def final_price (initial_price : ℚ) : ℚ :=
  initial_price * (1 + price_increase) * (1 - discount1) * (1 - discount2) * (1 - discount3)

-- Theorem stating the overall percentage change
theorem overall_percentage_change :
  ∀ initial_price : ℚ, initial_price > 0 →
  (final_price initial_price - initial_price) / initial_price = -41/500 := by
  sorry

-- Evaluate the result for initial_price = 100
#eval (final_price 100 - 100) / 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_percentage_change_l612_61261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l612_61277

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem -/
theorem triangle_problem (t : Triangle) (D : ℝ) :
  t.b = Real.sqrt 7 →
  t.c = 3 →
  Real.sqrt 3 * t.a = 2 * t.b * Real.sin t.A →
  D = (t.a + t.c) / 2 →
  (t.B = Real.pi / 3 ∨ t.B = 2 * Real.pi / 3) ∧
  (Real.sqrt ((t.a^2 + t.c * t.a + t.c^2) / 4) = Real.sqrt 13 / 2 ∨
   Real.sqrt ((t.a^2 + t.c * t.a + t.c^2) / 4) = Real.sqrt 19 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l612_61277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_intersection_l612_61237

noncomputable section

-- Define the curve C in polar coordinates
def C (θ : ℝ) : ℝ := Real.sqrt 2 * Real.sin (θ + Real.pi / 4)

-- Define the line in parametric form
def line_x (t : ℝ) : ℝ := -1 + (3/5) * t
def line_y (t : ℝ) : ℝ := -1 + (4/5) * t

-- State the theorem
theorem curve_and_intersection :
  -- Part I: Rectangular equation of curve C
  (∀ x y : ℝ, (x^2 + y^2 = (C (Real.arctan (y/x)))^2) → x^2 + y^2 - x - y = 0) ∧
  -- Part II: Distance between intersection points
  (∃ t₁ t₂ : ℝ, 
    (line_x t₁)^2 + (line_y t₁)^2 - (line_x t₁) - (line_y t₁) = 0 ∧
    (line_x t₂)^2 + (line_y t₂)^2 - (line_x t₂) - (line_y t₂) = 0 ∧
    Real.sqrt ((t₁ - t₂)^2) = Real.sqrt 41 / 5) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_intersection_l612_61237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_is_1000_l612_61296

/-- The side length of the square --/
noncomputable def side_length : ℝ := 100

/-- The speed ratio of A to B --/
noncomputable def speed_ratio : ℝ := 1.5

/-- The point where A and B meet --/
structure MeetingPoint where
  distance_A : ℝ  -- distance traveled by A
  distance_B : ℝ  -- distance traveled by B

/-- Calculate the meeting point of A and B --/
noncomputable def calculate_meeting_point : MeetingPoint := 
  { distance_A := 240,
    distance_B := 160 }

/-- Calculate the area of triangle ADE --/
noncomputable def area_ADE (mp : MeetingPoint) : ℝ :=
  (1 / 2) * (mp.distance_B - side_length) * side_length

/-- Calculate the area of triangle BCE --/
noncomputable def area_BCE (mp : MeetingPoint) : ℝ :=
  (1 / 2) * (mp.distance_A - 2 * side_length) * side_length

/-- The main theorem --/
theorem area_difference_is_1000 :
  let mp := calculate_meeting_point
  area_ADE mp - area_BCE mp = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_is_1000_l612_61296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_result_l612_61234

-- Define the reactants and their initial amounts
noncomputable def initial_KOH : ℝ := 3
noncomputable def initial_NH4I : ℝ := 2

-- Define the stoichiometric coefficients
noncomputable def coeff_KOH : ℝ := 1
noncomputable def coeff_NH4I : ℝ := 1
noncomputable def coeff_H2O : ℝ := 1

-- Define the reaction function
noncomputable def reaction (KOH NH4I : ℝ) : ℝ × ℝ × ℝ :=
  let limiting_amount := min (KOH / coeff_KOH) (NH4I / coeff_NH4I)
  let H2O_formed := limiting_amount * coeff_H2O
  let KOH_remaining := KOH - (limiting_amount * coeff_KOH)
  let NH4I_remaining := NH4I - (limiting_amount * coeff_NH4I)
  (H2O_formed, KOH_remaining, NH4I_remaining)

-- State the theorem
theorem reaction_result :
  let (H2O_formed, KOH_remaining, NH4I_remaining) := reaction initial_KOH initial_NH4I
  H2O_formed = 2 ∧ KOH_remaining = 1 ∧ NH4I_remaining = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reaction_result_l612_61234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_range_l612_61262

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = Real.pi
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define the theorem
theorem triangle_angle_range (t : Triangle) 
  (h : Real.sin t.A ^ 2 ≤ Real.sin t.B ^ 2 + Real.sin t.C ^ 2 - Real.sin t.B * Real.sin t.C) : 
  0 < t.A ∧ t.A ≤ Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_range_l612_61262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_schwarz_inequality_cauchy_schwarz_equality_condition_l612_61230

open BigOperators

variable {n : ℕ}
variable (x y : Fin n → ℝ)

theorem cauchy_schwarz_inequality :
  (∑ i, x i * y i) ^ 2 ≤ (∑ i, x i ^ 2) * (∑ i, y i ^ 2) := by
  sorry

theorem cauchy_schwarz_equality_condition :
  (∑ i, x i * y i) ^ 2 = (∑ i, x i ^ 2) * (∑ i, y i ^ 2) ↔
  ∃ c : ℝ, ∀ i, x i = c * y i := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_schwarz_inequality_cauchy_schwarz_equality_condition_l612_61230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_four_different_socks_l612_61229

/-- Represents the number of pairs of socks in the bag -/
def num_pairs : ℕ := 5

/-- Represents the number of socks drawn in each sample -/
def sample_size : ℕ := 4

/-- Represents the probability of drawing 4 different socks in the first draw -/
def p1 : ℚ := 8 / 21

/-- Represents the probability of drawing exactly one pair and two different socks in the first draw -/
def p2 : ℚ := 4 / 7

/-- Represents the probability of drawing 2 different socks in the next draw, given that we already have 3 different socks and one pair discarded -/
def p3 : ℚ := 4 / 15

/-- Theorem stating the probability of ending up with 4 socks of different colors -/
theorem probability_of_four_different_socks : 
  ∀ (bag : Finset (Finset Color)) (process : Finset (Finset Color) → ℚ),
  Finset.card bag = num_pairs ∧ 
  (∀ pair ∈ bag, Finset.card pair = 2) ∧
  (∀ pair1 pair2, pair1 ∈ bag → pair2 ∈ bag → pair1 ≠ pair2 → Disjoint pair1 pair2) →
  process bag = p1 + p2 * p3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_four_different_socks_l612_61229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_larger_x_l612_61275

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (x - 5)^2 / 7^2 - (y - 10)^2 / 3^2 = 1

-- Define the focus with larger x-coordinate
noncomputable def focus_larger_x : ℝ × ℝ := (5 + Real.sqrt 58, 10)

-- Theorem statement
theorem hyperbola_focus_larger_x :
  ∃ (f1 f2 : ℝ × ℝ), 
    (f1.1 ≠ f2.1) ∧ 
    (∀ (x y : ℝ), hyperbola x y → 
      ((x - f1.1)^2 + (y - f1.2)^2 = (x - f2.1)^2 + (y - f2.2)^2)) ∧
    (focus_larger_x = if f1.1 > f2.1 then f1 else f2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_larger_x_l612_61275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_king_card_probability_l612_61272

theorem king_card_probability (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 2) :
  let total_combinations := n.choose 2
  let at_least_one_king := total_combinations - (n - k).choose 2
  let no_kings := (n - k).choose 2
  (at_least_one_king : ℚ) / total_combinations > (no_kings : ℚ) / total_combinations :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_king_card_probability_l612_61272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l612_61279

-- Define the ☆ operation
noncomputable def star (a b : ℝ) : ℝ := a^2 + b^2

-- Define the ★ operation
noncomputable def diamond (a b : ℝ) : ℝ := (a * b) / 2

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, star 3 x = diamond x 12 :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l612_61279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_squared_l612_61210

open Real

/-- The curve C₁ -/
noncomputable def C₁ (x : ℝ) : ℝ := x^2 - log x

/-- The line L -/
def L (x y : ℝ) : Prop := x - y - 2 = 0

/-- The distance function between two points -/
def distance_squared (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (x₂ - x₁)^2 + (y₂ - y₁)^2

theorem min_distance_squared :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    y₁ = C₁ x₁ ∧ 
    L x₂ y₂ ∧
    (∀ (a₁ b₁ a₂ b₂ : ℝ), b₁ = C₁ a₁ → L a₂ b₂ → 
      distance_squared x₁ y₁ x₂ y₂ ≤ distance_squared a₁ b₁ a₂ b₂) ∧
    distance_squared x₁ y₁ x₂ y₂ = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_squared_l612_61210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_approx_l612_61265

/-- Represents the probability of giving money to another player -/
noncomputable def giveProbability : ℝ := 3/4

/-- Represents the probability of keeping money -/
noncomputable def keepProbability : ℝ := 1/4

/-- Represents the number of rounds in the game -/
def numRounds : ℕ := 50

/-- Represents the probability of maintaining the (1-1-1) state in a single round -/
noncomputable def maintainStateProbability : ℝ := keepProbability^3 + 2 * giveProbability^3

/-- Represents the final probability of all players having $1 after 50 rounds -/
noncomputable def finalProbability : ℝ := maintainStateProbability^numRounds

/-- Theorem stating that the final probability is approximately 0.005 -/
theorem game_probability_approx : 
  ∃ ε > 0, |finalProbability - 0.005| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_approx_l612_61265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l612_61288

def f (x : ℝ) := x^3 - x

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, x < -Real.sqrt 3 / 3 ∨ x > Real.sqrt 3 / 3 → (deriv f x) > 0) ∧
  (∀ M, ∃ N, ∀ x, x > N → f x > M) ∧
  f (-Real.sqrt 3 / 3) = -2 * Real.sqrt 3 / 9 ∧
  f (Real.sqrt 3 / 3) = -2 * Real.sqrt 3 / 9 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l612_61288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quad_prism_volume_l612_61227

/-- Regular quadrilateral prism with side length a and diagonal angle 30° -/
structure RegularQuadPrism (a : ℝ) where
  side_length : a > 0
  diagonal_angle : Real.pi / 6 = 30 * (Real.pi / 180)

/-- Volume of a regular quadrilateral prism -/
noncomputable def volume (a : ℝ) (p : RegularQuadPrism a) : ℝ := a^3 * Real.sqrt 2

/-- Theorem: The volume of a regular quadrilateral prism with side length a 
    and diagonal angle 30° is a³√2 -/
theorem regular_quad_prism_volume (a : ℝ) (p : RegularQuadPrism a) : 
  volume a p = a^3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_quad_prism_volume_l612_61227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l612_61276

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x => 
  if x ≥ 1 ∧ x ≤ 3 then Real.log x 
  else if x > 0 then 2 * Real.log (1/x) 
  else 0  -- Define a default value for x ≤ 0

-- Define the function g
noncomputable def g (a : ℝ) : ℝ → ℝ := fun x => f x - a * x

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, 
    (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
      1/3 ≤ x₁ ∧ x₁ ≤ 3 ∧ 
      1/3 ≤ x₂ ∧ x₂ ≤ 3 ∧ 
      1/3 ≤ x₃ ∧ x₃ ≤ 3 ∧ 
      g a x₁ = 0 ∧ g a x₂ = 0 ∧ g a x₃ = 0) ↔
    (Real.log 3 / 3 ≤ a ∧ a < 1 / Real.exp 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l612_61276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_dual_base_number_l612_61266

/-- Represents a three-digit number in a given base -/
structure ThreeDigitNumber (base : ℕ) where
  hundreds : ℕ
  tens : ℕ
  ones : ℕ
  hundreds_valid : hundreds < base
  tens_valid : tens < base
  ones_valid : ones < base

/-- Converts a ThreeDigitNumber to its decimal (base 10) representation -/
def to_decimal (base : ℕ) (num : ThreeDigitNumber base) : ℕ :=
  num.hundreds * base^2 + num.tens * base + num.ones

theorem largest_dual_base_number :
  ∀ n : ℕ,
  (∃ (a b c : ℕ),
    (a < 5 ∧ b < 5 ∧ c < 5) ∧
    (a < 9 ∧ b < 9 ∧ c < 9) ∧
    n = to_decimal 5 { hundreds := a, tens := b, ones := c, hundreds_valid := by sorry, tens_valid := by sorry, ones_valid := by sorry } ∧
    n = to_decimal 9 { hundreds := c, tens := b, ones := a, hundreds_valid := by sorry, tens_valid := by sorry, ones_valid := by sorry }) →
  n ≤ 126 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_dual_base_number_l612_61266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l612_61231

/-- Proves that when a and b complete a work in 6 days, and a can do it alone in 10 days,
    then a and b together also complete the work in 6 days. -/
theorem work_completion_time 
  (total_time : ℝ) 
  (a_time : ℝ) 
  (combined_rate : ℝ) 
  (a_rate : ℝ)
  (h1 : total_time = 6)
  (h2 : a_time = 10)
  (h3 : combined_rate = 1 / total_time)
  (h4 : a_rate = 1 / a_time) :
  total_time = 6 := by
  -- The proof is trivial as it's given in the hypothesis
  exact h1

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l612_61231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_intersection_existence_l612_61264

/-- Given a disk of radius n ≥ 1 containing 4n unit-length segments, 
    there exists a line parallel to one of the coordinate axes 
    that intersects at least two of these segments. -/
theorem segment_intersection_existence (n : ℝ) (h_n : n ≥ 1) :
  ∃ (line : Set (ℝ × ℝ)) (segments : Finset (Set (ℝ × ℝ))),
    (∀ p, p ∈ line → p.1 = 0 ∨ p.2 = 0) ∧ 
    (segments.card = 4 * Int.floor n) ∧
    (∀ s, s ∈ segments → ∃ a b : ℝ × ℝ, s = {x | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = a + t • (b - a)} ∧ 
                                    ‖b - a‖ = 1 ∧ 
                                    ‖a‖ ≤ n ∧ ‖b‖ ≤ n) ∧
    (∃ s1 s2, s1 ∈ segments ∧ s2 ∈ segments ∧ s1 ≠ s2 ∧ (s1 ∩ line).Nonempty ∧ (s2 ∩ line).Nonempty) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_intersection_existence_l612_61264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_max_volume_l612_61271

open Real

theorem cylinder_max_volume (S : ℝ) (h : S > 0) :
  let r := S / (4 * π)
  let h := S / (2 * π * r)
  r = S / (4 * π) :=
by
  -- Define the radius and height in terms of S
  let r := S / (4 * π)
  let h := S / (2 * π * r)

  -- The proof would go here
  sorry

#check cylinder_max_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_max_volume_l612_61271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_l612_61286

/-- Represents a triangular pyramid ABCD with a cross-section ADE --/
structure TriangularPyramid where
  /-- Length of edge AD --/
  a : ℝ
  /-- Area of cross-section ADE --/
  S : ℝ
  /-- Angle between cross-section ADE and face ACD --/
  α : ℝ
  /-- Angle between cross-section ADE and face ADB --/
  β : ℝ
  /-- E is the midpoint of BC --/
  E_is_midpoint : Bool

/-- The volume of the triangular pyramid --/
noncomputable def volume (pyramid : TriangularPyramid) : ℝ :=
  (8 * pyramid.S^2 * Real.sin pyramid.α * Real.sin pyramid.β) / 
  (3 * pyramid.a * Real.sin (pyramid.α + pyramid.β))

/-- Theorem stating the volume of the triangular pyramid --/
theorem triangular_pyramid_volume (pyramid : TriangularPyramid) :
  volume pyramid = (8 * pyramid.S^2 * Real.sin pyramid.α * Real.sin pyramid.β) / 
                   (3 * pyramid.a * Real.sin (pyramid.α + pyramid.β)) := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_pyramid_volume_l612_61286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_property_l612_61233

/-- A function representing the absolute value of sine -/
noncomputable def f (x : ℝ) := |Real.sin x|

/-- Theorem statement -/
theorem intersection_property (k α : ℝ) : 
  k > 0 ∧ 
  π < α ∧ 
  α < 3*π/2 ∧ 
  (∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ = α ∧ 
    f x₁ = k * x₁ ∧ 
    f x₂ = k * x₂ ∧ 
    f x₃ = k * x₃ ∧ 
    (∀ x, f x = k * x → x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  Real.cos α / (Real.sin α + Real.sin (3*α)) = (1 + α^2) / (4*α) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_property_l612_61233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expected_benefit_l612_61245

/-- The expected benefit function for fishing -/
noncomputable def expected_benefit (w k : ℝ) (n : ℕ) : ℝ :=
  w * n * (1 - 1/k)^n

/-- Theorem stating that the expected benefit is maximized when n = k - 1 -/
theorem max_expected_benefit (w k : ℝ) (h1 : w > 0) (h2 : k > 1) :
  ∃ (n : ℕ), ∀ (m : ℕ), expected_benefit w k n ≥ expected_benefit w k m ∧ n = ⌊k⌋ - 1 := by
  sorry

#check max_expected_benefit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expected_benefit_l612_61245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_problems_without_conditional_l612_61205

-- Define a function to represent whether a problem requires conditional statements
def requires_conditional (problem : Nat) : Bool :=
  match problem with
  | 1 => false  -- opposite number
  | 2 => false  -- square perimeter
  | 3 => true   -- maximum of three numbers
  | 4 => true   -- piecewise function
  | _ => false  -- other cases (not relevant for this problem)

-- Theorem statement
theorem two_problems_without_conditional : 
  (Finset.range 4).sum (fun i => if requires_conditional i then 0 else 1) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_problems_without_conditional_l612_61205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_student_weight_l612_61284

theorem new_student_weight (n : ℕ) (w₁ w₂ : ℝ) (h1 : n = 29) (h2 : w₁ = 28) (h3 : w₂ = 27.8) :
  (n + 1) * w₂ - n * w₁ = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_student_weight_l612_61284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_parallel_min_magnitude_sum_l612_61238

/-- Given vectors a and b with a real parameter m -/
def a (m : ℝ) : ℝ × ℝ := (4, 3 - m)
def b (m : ℝ) : ℝ × ℝ := (1, m)

/-- Theorem stating that vectors a and b are parallel when m = 3/5 -/
theorem vectors_parallel : 
  let m : ℝ := 3/5
  ∃ (k : ℝ), a m = k • b m := by
  sorry

/-- Theorem stating that the minimum value of |a + 2b| is 6 -/
theorem min_magnitude_sum : 
  (⨅ (m : ℝ), ‖a m + 2 • b m‖) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_parallel_min_magnitude_sum_l612_61238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_product_contains_97420_l612_61215

def count_digit (n d : ℕ) : ℕ := 
  if n = 0 then
    if d = 0 then 1 else 0
  else
    if n % 10 = d then 
      1 + count_digit (n / 10) d
    else 
      count_digit (n / 10) d

def is_valid_pair (a b : ℕ) : Prop :=
  a ≥ 10000 ∧ a < 100000 ∧ b ≥ 10000 ∧ b < 100000 ∧
  ∀ d : ℕ, d < 10 → (count_digit a d + count_digit b d = 1)

noncomputable def largest_product_pair : ℕ × ℕ :=
  sorry

theorem largest_product_contains_97420 :
  let (a, b) := largest_product_pair
  97420 = a ∨ 97420 = b :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_product_contains_97420_l612_61215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_term_position_l612_61260

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℚ  -- first term
  d : ℚ  -- common difference

/-- The nth term of an arithmetic progression -/
def nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  ap.a + (n - 1 : ℚ) * ap.d

/-- The sum of the first n terms of an arithmetic progression -/
def sumFirstNTerms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * ap.a + (n - 1 : ℚ) * ap.d)

theorem other_term_position 
  (ap : ArithmeticProgression) 
  (x : ℕ) 
  (h1 : nthTerm ap 4 + nthTerm ap x = 20)
  (h2 : sumFirstNTerms ap 10 = 100) : 
  x = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_term_position_l612_61260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_percentage_l612_61258

noncomputable def original_price : ℝ := 79.95
noncomputable def sale_price : ℝ := 59.95

noncomputable def price_decrease : ℝ := original_price - sale_price

noncomputable def percentage_decrease : ℝ := (price_decrease / original_price) * 100

theorem price_decrease_percentage : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |percentage_decrease - 25| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_percentage_l612_61258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_removed_black_cubes_divisible_by_four_l612_61249

/-- Represents a cube with dimensions 10 × 10 × 10 -/
structure LargeCube where
  dimension : ℕ
  total_cubes : ℕ
  black_cubes : ℕ
  white_cubes : ℕ
  removed_cubes : ℕ
  rows : ℕ
  cubes_per_row : ℕ

/-- Represents the properties of the given cube -/
def given_cube : LargeCube := {
  dimension := 10,
  total_cubes := 1000,
  black_cubes := 500,
  white_cubes := 500,
  removed_cubes := 100,
  rows := 300,
  cubes_per_row := 10
}

/-- Function to represent the removal of cubes from a row -/
def removed_cubes_from_row (cube : LargeCube) (row : ℕ) : Set ℕ := sorry

/-- Function to count the number of removed black cubes -/
def number_of_removed_black_cubes (cube : LargeCube) : ℕ := sorry

/-- The main theorem to prove -/
theorem removed_black_cubes_divisible_by_four (cube : LargeCube) 
  (h1 : cube.dimension = 10)
  (h2 : cube.total_cubes = 1000)
  (h3 : cube.black_cubes = 500)
  (h4 : cube.white_cubes = 500)
  (h5 : cube.removed_cubes = 100)
  (h6 : cube.rows = 300)
  (h7 : cube.cubes_per_row = 10)
  (h8 : ∀ row, ∃! cube_in_row, cube_in_row ∈ removed_cubes_from_row cube row) :
  ∃ k : ℕ, number_of_removed_black_cubes cube = 4 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_removed_black_cubes_divisible_by_four_l612_61249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_roots_sum_l612_61204

theorem tangent_roots_sum (α β : Real) : 
  α ∈ Set.Ioo 0 π → 
  β ∈ Set.Ioo 0 π → 
  (Real.tan α)^2 + 3*Real.sqrt 3*(Real.tan α) + 4 = 0 → 
  (Real.tan β)^2 + 3*Real.sqrt 3*(Real.tan β) + 4 = 0 → 
  α + β = 4*π/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_roots_sum_l612_61204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_existence_l612_61208

theorem divisor_existence (S : Finset ℕ) : 
  S.card = 1008 → (∀ n, n ∈ S → 1 ≤ n ∧ n ≤ 2014) →
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_existence_l612_61208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l612_61297

-- Define an equilateral triangle
noncomputable def equilateral_triangle (side_length : ℝ) : ℝ := 3 * side_length

-- Define an isosceles triangle
noncomputable def isosceles_triangle_leg (perimeter base : ℝ) : ℝ := (perimeter - base) / 2

theorem triangle_properties :
  (equilateral_triangle 12 = 36) ∧
  (isosceles_triangle_leg 72 28 = 22) := by
  constructor
  · -- Proof for equilateral triangle
    simp [equilateral_triangle]
    norm_num
  · -- Proof for isosceles triangle
    simp [isosceles_triangle_leg]
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l612_61297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_Z_l612_61242

theorem modulus_of_Z (Z : ℂ) (h : Z * (2 - 3*Complex.I) = 6 + 4*Complex.I) : Complex.abs Z = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_Z_l612_61242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_nine_digit_number_with_distinct_remainders_l612_61221

theorem no_nine_digit_number_with_distinct_remainders :
  ¬ ∃ (N : ℕ), 
    (100000000 ≤ N ∧ N < 1000000000) ∧  -- Nine-digit number
    (∀ d ∈ N.digits 10, d ≠ 0) ∧  -- No zero digits
    (N.digits 10).card = 9 ∧  -- All digits are distinct
    (∀ i j, i ∈ N.digits 10 → j ∈ N.digits 10 → i ≠ j → N % i ≠ N % j) :=  -- Distinct remainders
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_nine_digit_number_with_distinct_remainders_l612_61221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_solution_is_unique_l612_61217

/-- Represents the system of linear equations -/
noncomputable def linear_system (n : ℕ) (A B : ℝ) (x : ℕ → ℝ) : Prop :=
  2 * x 1 - 2 * x 2 = A ∧
  (∀ i ∈ Finset.range (n - 1), i ≥ 1 → 
    -(i : ℝ) * x i + 2 * (i + 1 : ℝ) * x (i + 1) - (i + 2 : ℝ) * x (i + 2) = 0) ∧
  -(n - 1 : ℝ) * x (n - 1) + 2 * n * x n = B

/-- The solution to the system of linear equations -/
noncomputable def solution (n : ℕ) (A B : ℝ) (i : ℕ) : ℝ :=
  ((n + 1 - i : ℝ) * A + (i : ℝ) * B) / ((i : ℝ) * (n + 1 : ℝ))

/-- Theorem stating that the solution satisfies the system of linear equations -/
theorem solution_satisfies_system (n : ℕ) (A B : ℝ) (h : n > 1) :
  linear_system n A B (solution n A B) := by
  sorry

/-- Theorem stating that the solution is unique -/
theorem solution_is_unique (n : ℕ) (A B : ℝ) (h : n > 1) (x : ℕ → ℝ) :
  linear_system n A B x → x = solution n A B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_solution_is_unique_l612_61217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_C_right_triangle_l612_61246

/-- Cosine of angle C in a right triangle with AB = 15 and AC = 10 -/
theorem cos_C_right_triangle (AB AC : ℝ) (h_right : AB = 15 ∧ AC = 10) :
  let BC := Real.sqrt (AB^2 + AC^2)
  Real.cos (Real.arccos (AB / BC)) = 3 * Real.sqrt 325 / 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_C_right_triangle_l612_61246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_between_A_and_B_AB_AC_relation_platform_C_location_l612_61206

-- Define the locations of platforms A and B
noncomputable def A : ℝ := 9
noncomputable def B : ℝ := 1/3

-- Define C as a real number between A and B
noncomputable def C : ℝ := 7

-- State the condition that C is between A and B
theorem C_between_A_and_B : B < C ∧ C < A := by
  sorry

-- Define the relationship between AB and AC
theorem AB_AC_relation : A - B = 13/3 * (A - C) := by
  sorry

-- Theorem to prove
theorem platform_C_location : C = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_between_A_and_B_AB_AC_relation_platform_C_location_l612_61206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_arch_height_l612_61257

/-- Represents a parabolic arch --/
structure ParabolicArch where
  a : ℝ
  k : ℝ

/-- Height of the arch at a given x-coordinate --/
def ParabolicArch.height (arch : ParabolicArch) (x : ℝ) : ℝ := arch.a * x^2 + arch.k

theorem parabolic_arch_height :
  ∃ (arch : ParabolicArch),
    (arch.height 0 = 20) ∧
    (arch.height 25 = 0) ∧
    (arch.height 10 = 16.8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_arch_height_l612_61257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l612_61244

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ  -- Vertex A as a point in 2D plane
  B : ℝ × ℝ  -- Vertex B as a point in 2D plane
  C : ℝ × ℝ  -- Vertex C as a point in 2D plane

-- Define vector operations
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

def vector_length_squared (v : ℝ × ℝ) : ℝ := v.1^2 + v.2^2

-- Define the theorem
theorem triangle_properties (ABC : Triangle) 
  (h1 : dot_product (vector_sub ABC.B ABC.A) (vector_sub ABC.C ABC.A) = 1)
  (h2 : dot_product (vector_sub ABC.A ABC.B) (vector_sub ABC.C ABC.B) = 1)
  (h3 : vector_length_squared (vector_sub ABC.B ABC.A) + 
        vector_length_squared (vector_sub ABC.C ABC.A) + 
        2 * dot_product (vector_sub ABC.B ABC.A) (vector_sub ABC.C ABC.A) = 36) :
  (∃ (angle_A angle_B : ℝ), angle_A = angle_B) ∧ 
  ∃ (c : ℝ), c = Real.sqrt 2 ∧
  ∃ (S : ℝ), S = (3 * Real.sqrt 7) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l612_61244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_for_special_angle_l612_61218

theorem tan_value_for_special_angle (α : Real) 
  (h1 : Real.sin α - Real.cos α = Real.sqrt 2) 
  (h2 : α ∈ Set.Ioo 0 Real.pi) : 
  Real.tan α = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_for_special_angle_l612_61218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_investment_l612_61213

/-- Represents the investment and profit share of a partner in the business -/
structure Partner where
  investment : ℚ
  profitShare : ℚ

/-- The business scenario with three partners -/
structure Business where
  a : Partner
  b : Partner
  c : Partner
  profitShareRatio : ℚ

/-- The given business scenario -/
def givenBusiness : Business where
  a := { investment := 8000, profitShare := 0 }
  b := { investment := 10000, profitShare := 2000 }
  c := { investment := 0, profitShare := 0 }
  profitShareRatio := 2000 / 10000

theorem c_investment (business : Business) :
  business.a.investment = 8000 →
  business.b.investment = 10000 →
  business.b.profitShare = 2000 →
  business.profitShareRatio = business.b.profitShare / business.b.investment →
  business.a.profitShare - business.c.profitShare = 800 →
  business.c.investment = 4000 := by
  sorry

#check c_investment

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_investment_l612_61213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_area_l612_61298

/-- Helper function to calculate the area of a triangle given its side lengths -/
noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Given a triangle with sides a, b, and c, and an inscribed circle of radius r,
    this theorem states that the total area of the inscribed circle and the three circles
    inscribed in the triangles formed by tangents parallel to the sides is equal to
    (a² + b² + c²) / (a + b + c)² * π * r² -/
theorem inscribed_circles_area (a b c r : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_inradius : r = 2 * (area_triangle a b c) / (a + b + c)) :
  let total_area := π * r^2 * (a^2 + b^2 + c^2) / (a + b + c)^2
  ∃ (r1 r2 r3 : ℝ), 
    r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧
    total_area = π * (r^2 + r1^2 + r2^2 + r3^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_area_l612_61298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_is_60_l612_61259

/-- An arithmetic sequence with 9 terms and common difference 3 -/
noncomputable def ArithmeticSequence (a₁ : ℝ) : Fin 9 → ℝ :=
  λ n => a₁ + 3 * n.val

/-- A random variable that takes values from the arithmetic sequence with equal probability -/
noncomputable def Xi (a₁ : ℝ) : Fin 9 → ℝ := ArithmeticSequence a₁

/-- The probability mass function for Xi -/
noncomputable def prob : Fin 9 → ℝ := λ _ => 1 / 9

/-- The expected value of Xi -/
noncomputable def expectation (a₁ : ℝ) : ℝ := Finset.sum (Finset.univ : Finset (Fin 9)) (λ n => prob n * Xi a₁ n)

/-- The variance of Xi -/
noncomputable def variance (a₁ : ℝ) : ℝ := Finset.sum (Finset.univ : Finset (Fin 9)) (λ n => prob n * (Xi a₁ n - expectation a₁)^2)

/-- The theorem stating that the variance of Xi is 60 -/
theorem variance_is_60 (a₁ : ℝ) : variance a₁ = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_is_60_l612_61259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_theorem_l612_61219

-- Define a segment as a pair of real numbers (start, end)
def Segment := ℝ × ℝ

-- Define a property that a point is contained in a segment
def contains (s : Segment) (p : ℝ) : Prop :=
  s.1 ≤ p ∧ p ≤ s.2

-- Define the property that for any three segments, there exist two points
-- such that each segment contains at least one of them
def three_segment_property (segments : List Segment) : Prop :=
  ∀ s1 s2 s3, s1 ∈ segments → s2 ∈ segments → s3 ∈ segments → ∃ p1 p2 : ℝ,
    (contains s1 p1 ∨ contains s1 p2) ∧
    (contains s2 p1 ∨ contains s2 p2) ∧
    (contains s3 p1 ∨ contains s3 p2)

-- The theorem to prove
theorem segment_theorem (segments : List Segment) :
  three_segment_property segments →
  ∃ p1 p2 : ℝ, ∀ s ∈ segments, contains s p1 ∨ contains s p2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_theorem_l612_61219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l612_61291

/-- Properties of triangles --/
theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  -- Define triangle ABC
  (0 < A ∧ A < Real.pi) ∧ (0 < B ∧ B < Real.pi) ∧ (0 < C ∧ C < Real.pi) ∧
  (A + B + C = Real.pi) ∧
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  (a / Real.sin A = b / Real.sin B) ∧ (b / Real.sin B = c / Real.sin C) →
  -- Statement B
  (A > B → Real.sin A > Real.sin B) ∧
  -- Statement C
  (c / b < Real.cos A → a^2 > b^2 + c^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l612_61291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_nonperiodic_sequence_with_periodic_subsequences_l612_61250

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- A sequence is periodic if there exists a positive integer T such that
    for all n, a(n + T) = a(n) -/
def IsPeriodic (a : Sequence) : Prop :=
  ∃ T : ℕ+, ∀ n : ℕ, a (n + T) = a n

/-- A subsequence of a given sequence with step k -/
def Subsequence (a : Sequence) (k : ℕ) : Sequence :=
  fun n => a (k * n)

/-- The main theorem -/
theorem exists_nonperiodic_sequence_with_periodic_subsequences :
  ∃ a : Sequence,
    (Set.Finite (Set.range a)) ∧
    (∀ k : ℕ, k > 1 → IsPeriodic (Subsequence a k)) ∧
    ¬(IsPeriodic a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_nonperiodic_sequence_with_periodic_subsequences_l612_61250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_mod_3_square_mod_3_mod_1967_3_sum_of_digits_of_square_not_1967_l612_61232

def SumOfDigits : ℤ → ℕ
| n => sorry  -- Definition of sum of digits function

theorem sum_of_digits_mod_3 (n : ℤ) : SumOfDigits n % 3 = n % 3 := by
  sorry

theorem square_mod_3 (n : ℤ) : (n^2) % 3 = 0 ∨ (n^2) % 3 = 1 := by
  sorry

theorem mod_1967_3 : 1967 % 3 = 2 := by
  sorry

theorem sum_of_digits_of_square_not_1967 (n : ℤ) : SumOfDigits (n^2) ≠ 1967 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_mod_3_square_mod_3_mod_1967_3_sum_of_digits_of_square_not_1967_l612_61232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_domain_of_log_l612_61293

-- Define the function f(x) = ln(x - 1)
noncomputable def f (x : ℝ) := Real.log (x - 1)

-- Define the domain of f as a set
def A : Set ℝ := {x | x > 1}

-- State the theorem
theorem complement_of_domain_of_log (x : ℝ) :
  x ∈ (Set.univ \ A) ↔ x ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_domain_of_log_l612_61293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_1944_optimal_distribution_l612_61240

/-- Represents the distribution of clothing styles between two stores -/
structure ClothingDistribution where
  storeA_styleA : ℕ
  storeA_styleB : ℕ
  storeB_styleA : ℕ
  storeB_styleB : ℕ

/-- Calculates the total profit for a given distribution -/
def totalProfit (d : ClothingDistribution) : ℕ :=
  30 * d.storeA_styleA + 40 * d.storeA_styleB + 27 * d.storeB_styleA + 36 * d.storeB_styleB

/-- Checks if a distribution is valid according to the problem constraints -/
def isValidDistribution (d : ClothingDistribution) : Prop :=
  d.storeA_styleA + d.storeA_styleB = 30 ∧
  d.storeB_styleA + d.storeB_styleB = 30 ∧
  d.storeA_styleA + d.storeB_styleA = 35 ∧
  d.storeA_styleB + d.storeB_styleB = 25 ∧
  27 * d.storeB_styleA + 36 * d.storeB_styleB ≥ 950

/-- Theorem stating that the maximum total profit is 1944 yuan -/
theorem max_profit_is_1944 :
  ∃ (d : ClothingDistribution), isValidDistribution d ∧
    totalProfit d = 1944 ∧
    ∀ (d' : ClothingDistribution), isValidDistribution d' → totalProfit d' ≤ 1944 :=
by sorry

/-- Corollary stating the optimal distribution -/
theorem optimal_distribution :
  ∃ (d : ClothingDistribution), isValidDistribution d ∧
    totalProfit d = 1944 ∧
    d.storeA_styleA = 21 ∧ d.storeA_styleB = 9 ∧
    d.storeB_styleA = 14 ∧ d.storeB_styleB = 16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_1944_optimal_distribution_l612_61240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_negative_integers_l612_61248

theorem max_negative_integers (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℤ) 
  (h : a₁ * a₂ * a₃ * a₄ * a₅ * a₆ * a₇ * a₈ < 0) :
  ∃ (n : ℕ), n ≤ 7 ∧ 
    (∃ (s : Finset ℤ), 
      (∀ i ∈ s, i < 0) ∧
      (s.card = n) ∧
      ((Finset.filter (· < 0) {a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈}).card ≤ n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_negative_integers_l612_61248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_solution_satisfies_equation_l612_61209

/-- The differential equation y'' - 2y' - 3y = 0 with initial conditions y(0) = 8 and y'(0) = 0 -/
def differential_equation (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv^[2] y) x - 2 * (deriv y) x - 3 * y x = 0 ∧ y 0 = 8 ∧ (deriv y) 0 = 0

/-- The particular solution y = 6e^(-x) + 2e^(3x) -/
noncomputable def particular_solution (x : ℝ) : ℝ :=
  6 * Real.exp (-x) + 2 * Real.exp (3*x)

/-- Theorem stating that the particular solution satisfies the differential equation -/
theorem particular_solution_satisfies_equation :
  differential_equation particular_solution := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particular_solution_satisfies_equation_l612_61209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt3_over_2_l612_61225

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop := true

-- Define the lengths of sides
noncomputable def AB (A B : ℝ × ℝ) : ℝ := Real.sqrt 3
def BC (_ _ : ℝ × ℝ) : ℝ := 1
noncomputable def AC (A C : ℝ × ℝ) : ℝ := 2  -- We know AC = 2 from the problem

-- Define the relationship between sin C and cos C
def angle_C_property (C : ℝ) : Prop := Real.sin C = Real.sqrt 3 * Real.cos C

-- Define the area of a triangle
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ := 
  let a := AB B C
  let b := BC C A
  let c := AC A B
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem triangle_area_is_sqrt3_over_2 
  (A B C : ℝ × ℝ) 
  (triangle : Triangle A B C)
  (ab_length : AB A B = Real.sqrt 3)
  (bc_length : BC B C = 1)
  (ac_length : AC A C = 2)
  (angle_prop : angle_C_property (Real.arccos ((AB A B)^2 + (BC B C)^2 - (AC A C)^2) / (2 * AB A B * BC B C))) :
  triangle_area A B C = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt3_over_2_l612_61225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_equipment_value_after_n_years_l612_61285

/-- Represents the value of equipment over time with depreciation -/
noncomputable def equipment_value (a : ℝ) (b : ℝ) (n : ℕ+) : ℝ :=
  a * (1 - b / 100) ^ n.val

/-- 
Theorem stating that the value of equipment after n years
is equal to a(1-b%)^n, given:
  a: initial value in ten thousand yuan
  b: yearly depreciation rate as a percentage
  n: number of years (positive integer)
-/
theorem equipment_value_after_n_years (a b : ℝ) (n : ℕ+) :
  equipment_value a b n = a * (1 - b / 100) ^ n.val :=
by
  -- Unfold the definition of equipment_value
  unfold equipment_value
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_equipment_value_after_n_years_l612_61285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_median_perpendicular_l612_61236

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define the median
noncomputable def median (t : Triangle) (v : ℝ × ℝ) : ℝ × ℝ :=
  ((v.1 + (t.Q.1 + t.R.1) / 2) / 2, (v.2 + (t.Q.2 + t.R.2) / 2) / 2)

-- Define perpendicularity of vectors
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Define the length of a vector
noncomputable def length (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

-- State the theorem
theorem triangle_median_perpendicular (t : Triangle) :
  perpendicular (median t t.P) (median t t.Q) →
  length (t.Q.1 - t.R.1, t.Q.2 - t.R.2) = 9 →
  length (t.P.1 - t.R.1, t.P.2 - t.R.2) = 8 →
  length (t.P.1 - t.Q.1, t.P.2 - t.Q.2) = Real.sqrt 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_median_perpendicular_l612_61236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_decomposition_l612_61212

theorem factorial_decomposition : 2^8 * 3^4 * 5^1 * 35 = Nat.factorial 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_decomposition_l612_61212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_ratio_calculation_l612_61220

/-- Given two partners p and q with investments and time periods, calculate their profit ratio -/
theorem profit_ratio_calculation (x : ℕ) (h : x > 0) : 
  (7 * x * 5 : ℚ) / (5 * x * 12) = 7 / 12 := by
  sorry

#check profit_ratio_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_ratio_calculation_l612_61220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_permutation_invariant_l612_61283

open BigOperators

variable {n : ℕ}
variable (b : Fin n → ℝ)
variable (σ : Equiv.Perm (Fin n))

theorem sum_permutation_invariant :
  ∑ i, b i = ∑ i, b (σ i) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_permutation_invariant_l612_61283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_theorem_l612_61228

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form y = kx + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- Represents a circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.k * p.x + l.b

/-- Check if a point lies on a circle -/
def pointOnCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.h)^2 + (p.y - c.k)^2 = c.r^2

/-- Main theorem -/
theorem line_circle_intersection_theorem (l : Line) (c : Circle) (M N : Point) :
  l.b = 3 ∧ 
  c.h = 2 ∧ 
  c.k = 3 ∧ 
  c.r = 2 ∧
  pointOnLine M l ∧
  pointOnLine N l ∧
  pointOnCircle M c ∧
  pointOnCircle N c ∧
  distance M N ≥ 2 * Real.sqrt 3 →
  -Real.sqrt 3 / 3 ≤ l.k ∧ l.k ≤ Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_theorem_l612_61228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_difference_l612_61280

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 1 - sqrt (2 - 3 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * log x

-- Define the theorem
theorem max_value_of_difference (x₁ x₂ : ℝ) :
  x₁ ≤ 2/3 → 
  x₂ > 0 → 
  f x₁ - g x₂ = 1/4 → 
  x₁ - x₂ ≤ -25/48 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_difference_l612_61280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l612_61263

noncomputable def f (α : ℝ) : ℝ :=
  (Real.sin (α - 3 * Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + (3/2) * Real.pi)) /
  (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α))

theorem f_value_in_third_quadrant (α : ℝ) 
  (h1 : α ∈ Set.Icc Real.pi ((3/2) * Real.pi)) -- α is in the third quadrant
  (h2 : Real.cos (α - (3/2) * Real.pi) = 1/5) : 
  f α = 2 * Real.sqrt 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l612_61263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l612_61253

/-- The function f(x) = (ln x - 2ax) / x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x - 2 * a * x) / x

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a :
  ∃ a : ℝ, (∃! (x₀ : ℤ), f a (↑x₀) > 1) ∧
  a ∈ Set.Icc (1/4 * Real.log 2 - 1/2) (1/6 * Real.log 3 - 1/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l612_61253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_function_properties_l612_61251

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 1 - x
  else if x < 2 then -2 + Real.sqrt (4 - (x - 1)^2)
  else 2*x - 4

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := f (x - 1)

-- Theorem statement
theorem transformed_function_properties :
  (∀ x ∈ Set.Icc (-2) 1, ∃ m b, g x = m*x + b) ∧
  (∀ x ∈ Set.Ioo 1 3, ∃ c r, (x - c)^2 + (g x - (r - 2))^2 = r^2) ∧
  (∀ x ∈ Set.Icc 3 4, ∃ m b, g x = m*x + b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_function_properties_l612_61251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stationery_store_pricing_strategy_l612_61281

/-- Represents the stationery store's pricing and sales model -/
structure StationeryStore where
  cost_price : ℚ
  base_price : ℚ
  base_sales : ℚ
  price_increment : ℚ
  sales_decrement : ℚ
  max_price_ratio : ℚ

/-- Calculate the daily profit given a selling price -/
def daily_profit (store : StationeryStore) (selling_price : ℚ) : ℚ :=
  let sales := store.base_sales - (selling_price - store.base_price) / store.price_increment * store.sales_decrement
  (selling_price - store.cost_price) * sales

/-- The main theorem about the stationery store's pricing strategy -/
theorem stationery_store_pricing_strategy 
  (store : StationeryStore)
  (h1 : store.cost_price = 2)
  (h2 : store.base_price = 3)
  (h3 : store.base_sales = 500)
  (h4 : store.price_increment = 1/10)
  (h5 : store.sales_decrement = 10)
  (h6 : store.max_price_ratio = 12/5) :
  ∃ (price_for_800_profit : ℚ) (max_profit_price : ℚ) (max_profit : ℚ),
    daily_profit store price_for_800_profit = 800 ∧
    price_for_800_profit = 4 ∧
    max_profit_price = 24/5 ∧
    daily_profit store max_profit_price = max_profit ∧
    max_profit = 896 ∧
    ∀ (p : ℚ), p ≤ store.max_price_ratio * store.cost_price → 
      daily_profit store p ≤ daily_profit store max_profit_price := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stationery_store_pricing_strategy_l612_61281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_coloring_l612_61211

/-- A type representing the set of colors {0, 1, ..., 9} -/
def Color := Fin 10

/-- A function that determines if two positive real numbers differ in exactly one decimal place -/
def differInOnePlace (a b : ℝ) : Prop := sorry

/-- The main theorem stating the existence of a valid coloring function -/
theorem exists_valid_coloring : ∃ (c : ℝ → Color), 
  ∀ (a b : ℝ), differInOnePlace a b → c a ≠ c b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_coloring_l612_61211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagram_angle_ratio_l612_61295

/-- An equilateral pentagram is a five-pointed star where each side is equal in length. -/
structure EquilateralPentagram where
  -- We don't need to define the structure fully, just declare it exists
  dummy : Unit

/-- Given an equilateral pentagram, α represents the angle at point A -/
def α (p : EquilateralPentagram) : ℝ := sorry

/-- Given an equilateral pentagram, β represents the angle CGH -/
def β (p : EquilateralPentagram) : ℝ := sorry

/-- Given an equilateral pentagram, γ represents the angle HIJ -/
def γ (p : EquilateralPentagram) : ℝ := sorry

/-- In an equilateral pentagram, the ratio of angles α : β : γ is 7 : 4 : 2 -/
theorem pentagram_angle_ratio (p : EquilateralPentagram) :
  ∃ (k : ℝ), α p = 7 * k ∧ β p = 4 * k ∧ γ p = 2 * k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagram_angle_ratio_l612_61295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_le_g_l612_61243

def f (n : ℕ+) : ℚ := (Finset.range n).sum (λ i => (1 : ℚ) / ((i + 1 : ℕ) ^ 3))

def g (n : ℕ+) : ℚ := (3 - 1 / (n : ℚ) ^ 2) / 2

theorem f_le_g : ∀ n : ℕ+, f n ≤ g n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_le_g_l612_61243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_segment_ratio_l612_61287

/-- Given a line segment AB extended to P and Q such that AQ:QP:PB = 5:2:1,
    prove that Q = (3/8)A + (5/8)B -/
theorem extended_segment_ratio (A B P Q : ℝ × ℝ) : 
  (dist A Q : ℝ) / (dist Q P : ℝ) = 5 / 2 ∧ 
  (dist Q P : ℝ) / (dist P B : ℝ) = 2 / 1 →
  Q = ((3 : ℝ) / 8) • A + ((5 : ℝ) / 8) • B :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_segment_ratio_l612_61287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_one_l612_61267

-- Define the expression as noncomputable
noncomputable def complex_expression : ℝ :=
  Real.sqrt 5 * 5^(1/2) + 18 / 3 * 2 - 8^(3/2) / 2

-- State the theorem
theorem complex_expression_equals_one :
  complex_expression = 1 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equals_one_l612_61267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_on_line_and_parallel_l612_61278

noncomputable def line_param (t : ℝ) : ℝ × ℝ := (3 * t + 1, 2 * t + 3)

noncomputable def vector : ℝ × ℝ := (23/2, 10)

def direction : ℝ × ℝ := (3, 2)

theorem vector_on_line_and_parallel :
  ∃ t : ℝ, line_param t = vector ∧ 
  ∃ k : ℝ, vector.1 = k * direction.1 ∧ vector.2 = k * direction.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_on_line_and_parallel_l612_61278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l612_61200

/-- Predicate to check if a line given by ρ₀ and θ₀ is tangent to a circle given by ρ and θ -/
def is_tangent_line (ρ₀ θ₀ ρ θ : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 = ρ^2 ∧
               x = ρ * Real.cos θ ∧
               y = ρ * Real.sin θ ∧
               x = ρ₀ * Real.cos θ₀

/-- Given a circle with equation ρ = 4cos θ in polar coordinates,
    prove that a line tangent to this circle has the equation ρ cos θ = 4 -/
theorem tangent_line_to_circle (ρ θ : ℝ) :
  ρ = 4 * Real.cos θ →
  ∃ (ρ₀ θ₀ : ℝ), ρ₀ * Real.cos θ₀ = 4 ∧ is_tangent_line ρ₀ θ₀ ρ θ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l612_61200
