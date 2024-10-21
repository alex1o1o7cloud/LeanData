import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_increasing_interval_l32_3228

-- Define the function f(x) = sin(2x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

-- Define the property of being monotonically increasing on an interval
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem sin_2x_increasing_interval :
  ∀ k : ℤ, MonotonicallyIncreasing f (k * π - π/4) (k * π + π/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_increasing_interval_l32_3228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l32_3214

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi

-- Angles form an arithmetic sequence
def anglesFormArithmeticSequence (t : Triangle) : Prop :=
  2 * t.B = t.A + t.C

-- Sides form a geometric sequence
def sidesFormGeometricSequence (t : Triangle) : Prop :=
  t.b^2 = t.a * t.c

-- Main theorem
theorem triangle_properties (t : Triangle) 
  (h1 : isValidTriangle t) 
  (h2 : anglesFormArithmeticSequence t) 
  (h3 : sidesFormGeometricSequence t) : 
  Real.cos t.B = 1/2 ∧ Real.sin t.A * Real.sin t.C = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l32_3214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_example_l32_3208

/-- Dilation followed by translation of a complex number -/
noncomputable def transform (z : ℂ) (center : ℂ) (scale : ℝ) (translation : ℂ) : ℂ :=
  (scale • (z - center)) + center + translation

/-- The transformation maps -2 + i to -5 + 17i -/
theorem transform_example : 
  transform (-2 + I) (0 - 3*I) 3 (1 + 2*I) = -5 + 17*I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_example_l32_3208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_inequality_l32_3273

/-- Given a convex quadrilateral ABCD with side lengths a, b, c, d, and area S,
    prove that for any permutation (x, y, z, w) of (a, b, c, d),
    the inequality S ≤ (1/2)(xy + zw) holds. -/
theorem quadrilateral_area_inequality (a b c d S : ℝ) :
  a > 0 → b > 0 → c > 0 → d > 0 → S > 0 →
  (∃ (x y z w : ℝ), List.Perm [a, b, c, d] [x, y, z, w]) →
  S ≤ (1/2) * (x*y + z*w) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_inequality_l32_3273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_square_inside_triangle_l32_3218

noncomputable section

-- Define the square
def square_side : ℝ := 5

-- Define the triangle
def triangle_base : ℝ := 10
def triangle_height : ℝ := square_side

-- Define the coordinates
def square_lower_right : ℝ × ℝ := (5, 0)
def triangle_upper_right : ℝ × ℝ := (15, triangle_height)

-- Define the areas
def square_area : ℝ := square_side * square_side
def triangle_area : ℝ := (1/2) * triangle_base * triangle_height

-- Theorem statement
theorem area_outside_square_inside_triangle : 
  triangle_area - square_area = 0 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_square_inside_triangle_l32_3218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_one_l32_3251

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a) / (x + 1)

-- State the theorem
theorem extreme_value_at_one (a : ℝ) : 
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1 ∧ |x - 1| < ε → f a x ≤ f a 1) →
  a = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_one_l32_3251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_championship_probability_l32_3244

-- Define the probability function
def probability_A_wins (p : ℝ) (wins_needed_A wins_needed_B : ℕ) : ℝ :=
  if wins_needed_A = 0 then 1
  else if wins_needed_B = 0 then 0
  else p * probability_A_wins p (wins_needed_A - 1) wins_needed_B +
       (1 - p) * probability_A_wins p wins_needed_A (wins_needed_B - 1)

theorem volleyball_championship_probability :
  ∀ (wins_needed_A wins_needed_B : ℕ),
  ∀ (p : ℝ),
  wins_needed_A = 1 →
  wins_needed_B = 2 →
  0 < p →
  p < 1 →
  p = 1 - p →
  probability_A_wins p wins_needed_A wins_needed_B = 3/4 :=
by
  intros wins_needed_A wins_needed_B p h1 h2 h3 h4 h5
  rw [h1, h2]
  simp [probability_A_wins]
  rw [h5]
  -- The actual proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_championship_probability_l32_3244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_votes_is_30000_l32_3255

/-- Represents the total number of votes polled in the election -/
def total_votes : ℕ := sorry

/-- Represents the number of votes for the winner -/
def winner_votes : ℕ := sorry

/-- Represents the number of votes for the loser -/
def loser_votes : ℕ := sorry

/-- The winner's margin is 10% of the total votes -/
axiom winner_margin : winner_votes = loser_votes + (total_votes / 10)

/-- If 3000 votes change, the loser wins by 10% -/
axiom votes_change : (loser_votes + 3000) = (winner_votes - 3000) + (total_votes / 10)

/-- The total votes is the sum of winner and loser votes -/
axiom total_votes_sum : total_votes = winner_votes + loser_votes

/-- Theorem stating that the total number of votes is 30000 -/
theorem total_votes_is_30000 : total_votes = 30000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_votes_is_30000_l32_3255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perceived_weight_calculation_l32_3227

/-- The number of weight plates used -/
def num_plates : ℕ := 10

/-- The weight of each plate in pounds -/
noncomputable def weight_per_plate : ℝ := 30

/-- The percentage increase in perceived weight during lowering -/
noncomputable def weight_increase_percentage : ℝ := 20

/-- The total perceived weight when lowering the plates -/
noncomputable def total_perceived_weight : ℝ := num_plates * weight_per_plate * (1 + weight_increase_percentage / 100)

theorem perceived_weight_calculation :
  total_perceived_weight = 360 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perceived_weight_calculation_l32_3227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l32_3267

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem function_properties (A ω φ : ℝ) (h1 : A > 0) (h2 : ω > 0)
  (h3 : ∀ x, f A ω φ (x + 2) = f A ω φ x)
  (h4 : f A ω φ (1/3) = 2 ∧ ∀ x, f A ω φ x ≤ 2) :
  (∀ x, f A ω φ x = 2 * Real.sin (π * x + π/6)) ∧
  (∃ y ∈ Set.Icc (21/4 : ℝ) (23/4 : ℝ), y = 16/3 ∧
    ∀ x, f A ω φ (y + x) = f A ω φ (y - x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l32_3267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_exists_l32_3253

theorem no_such_function_exists : ¬∃ f : ℕ → ℕ, ∀ n : ℕ, n > 2 → f (f (n - 1)) = f (n + 1) - f n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_exists_l32_3253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_of_angles_l32_3268

theorem cos_difference_of_angles (α β : ℝ) :
  (∃ (P Q : ℝ × ℝ),
    P.1^2 + P.2^2 = 1 ∧
    Q.1^2 + Q.2^2 = 1 ∧
    P.2 = (4 * Real.sqrt 3) / 7 ∧
    Q.1 = 13 / 14) →
  Real.cos (α - β) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_of_angles_l32_3268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l32_3221

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  let (xA, yA) := t.A
  let (xB, yB) := t.B
  let (xC, yC) := t.C
  -- Add conditions to ensure it's a valid triangle
  (xA ≠ xB ∨ yA ≠ yB) ∧ (xB ≠ xC ∨ yB ≠ yC) ∧ (xC ≠ xA ∨ yC ≠ yA)

-- Define the midpoint of BC
noncomputable def midpoint_BC (t : Triangle) : ℝ × ℝ :=
  let (xB, yB) := t.B
  let (xC, yC) := t.C
  ((xB + xC) / 2, (yB + yC) / 2)

-- Define the length of AM
noncomputable def length_AM (t : Triangle) : ℝ :=
  let (xA, yA) := t.A
  let (xM, yM) := midpoint_BC t
  Real.sqrt ((xA - xM)^2 + (yA - yM)^2)

-- Define the area of the triangle
noncomputable def triangle_area (t : Triangle) : ℝ :=
  let (xA, yA) := t.A
  let (xB, yB) := t.B
  let (xC, yC) := t.C
  abs ((xA * (yB - yC) + xB * (yC - yA) + xC * (yA - yB)) / 2)

-- Theorem statement
theorem triangle_side_length (t : Triangle) (a b c : ℝ) :
  is_valid_triangle t →
  length_AM t = Real.sqrt 17 / 2 →
  Real.cos (Real.arccos (3/5)) = 3/5 →
  triangle_area t = 4 →
  a = 4 ∨ a = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l32_3221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_bill_theorem_l32_3296

/-- Represents a three-rate electricity meter --/
structure ElectricityMeter where
  peak : ℝ
  day : ℝ
  night : ℝ

/-- Represents tariff rates for electricity --/
structure TariffRates where
  peak : ℝ
  day : ℝ
  night : ℝ

/-- Calculates the maximum additional amount and expected difference --/
noncomputable def calculate_electricity_bill 
  (current_readings : ElectricityMeter)
  (previous_readings : ElectricityMeter)
  (tariffs : TariffRates)
  (actual_payment : ℝ) : ℝ × ℝ :=
  sorry

/-- The main theorem to prove --/
theorem electricity_bill_theorem 
  (current_readings : ElectricityMeter)
  (previous_readings : ElectricityMeter)
  (tariffs : TariffRates)
  (actual_payment : ℝ) :
  let (max_additional, expected_diff) := 
    calculate_electricity_bill current_readings previous_readings tariffs actual_payment
  (max_additional = 397.34 ∧ expected_diff = 19.30) :=
  by
    sorry

/-- Example usage with given data --/
def example_calculation : Unit :=
  let current_readings : ElectricityMeter := ⟨1402, 1347, 1337⟩
  let previous_readings : ElectricityMeter := ⟨1298, 1270, 1214⟩
  let tariffs : TariffRates := ⟨4.03, 3.39, 1.01⟩
  let actual_payment : ℝ := 660.72
  let result := calculate_electricity_bill current_readings previous_readings tariffs actual_payment
  ()

#eval example_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_electricity_bill_theorem_l32_3296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approximation_l32_3286

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length_approximation (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 30 → time_s = 6 → ∃ (length_m : ℝ), abs (length_m - 50) < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approximation_l32_3286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l32_3243

theorem expression_equality (x y : ℝ) (h : x - 2*y - 1 = 0) :
  (2 : ℝ)^x / (4 : ℝ)^y * 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l32_3243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_permutation_l32_3240

theorem three_digit_permutation (n : ℕ) (k : ℕ) : n = 5 → k = 3 → n.factorial / (n - k).factorial = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_permutation_l32_3240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fifteen_equals_seventyfive_l32_3235

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 0 + seq.a (n - 1))

/-- The determinant condition from the problem -/
def determinant_condition (seq : ArithmeticSequence) : Prop :=
  1 * seq.a 8 - 1 * (10 - seq.a 6) = 0

theorem sum_fifteen_equals_seventyfive (seq : ArithmeticSequence) 
  (h : determinant_condition seq) : sum_n seq 15 = 75 := by
  sorry

#check sum_fifteen_equals_seventyfive

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_fifteen_equals_seventyfive_l32_3235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_circle_radius_l32_3230

/-- Given a segment AB of length 2R and three circles as described,
    prove that the radius of the third circle is 4R/9 -/
theorem third_circle_radius (R : ℝ) (R_pos : R > 0) : ∃ (x : ℝ),
  let AB := 2 * R
  let circle1_radius := R
  let circle2_radius := R / 2
  let circle3_radius := x
  (∃ (A B O₁ O₂ O₃ : ℝ × ℝ),
    -- Circle 1 properties
    norm (B - A) = AB ∧
    norm (O₁ - A) = R ∧ norm (O₁ - B) = R ∧
    -- Circle 2 properties
    norm (O₂ - A) = circle2_radius ∧
    norm (O₂ - O₁) = R - circle2_radius ∧
    -- Circle 3 properties
    norm (O₃ - O₁) = R - x ∧
    norm (O₃ - O₂) = circle2_radius + x ∧
    (∃ (M : ℝ × ℝ), M.1 = A.1 ∨ M.1 = B.1) ∧
    norm (O₃ - M) = x) →
  x = 4 * R / 9 := by
  sorry

#check third_circle_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_circle_radius_l32_3230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_theorem_l32_3293

def equilateral_triangle_problem (A B C D E F G : ℝ × ℝ) : Prop :=
  let circle_radius : ℝ := 3
  let triangle_ABC_equilateral : Prop := sorry
  let triangle_ABC_inscribed : Prop := sorry
  let AD_length : ℝ := 15
  let AE_length : ℝ := 14
  let l1_parallel_AE : Prop := sorry
  let l2_parallel_AD : Prop := sorry
  let F_intersection : Prop := sorry
  let G_on_circle : Prop := sorry
  let G_collinear_AF : Prop := sorry
  let G_distinct_A : Prop := sorry
  let triangle_CBG_area : ℝ := sorry
  let p : ℕ := 360
  let q : ℕ := 3
  let r : ℕ := 961
  triangle_ABC_equilateral ∧
  triangle_ABC_inscribed ∧
  l1_parallel_AE ∧
  l2_parallel_AD ∧
  F_intersection ∧
  G_on_circle ∧
  G_collinear_AF ∧
  G_distinct_A ∧
  triangle_CBG_area = (p : ℝ) * Real.sqrt q / r ∧
  p + q + r = 1324

theorem equilateral_triangle_theorem (A B C D E F G : ℝ × ℝ) :
  equilateral_triangle_problem A B C D E F G := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_theorem_l32_3293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l32_3265

noncomputable section

-- Define the equations of the lines
def line1 (y a : ℝ) : ℝ := (16 / 5) * y + a
def line2 (x b : ℝ) : ℝ := (8 / 15) * x + b

-- Define the theorem
theorem intersection_point_sum (a b : ℝ) :
  (line1 (-2) a = 3) ∧ (line2 3 b = -2) → a + b = 29/5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l32_3265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_length_l32_3271

/-- A triangle ABC with vertices on the parabola y = 2x^2, where A is at the origin and BC is horizontal -/
structure ParabolaTriangle where
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_parabola_B : B.2 = 2 * B.1^2
  h_parabola_C : C.2 = 2 * C.1^2
  h_horizontal : B.2 = C.2

/-- The theorem stating that if the area of the triangle is 72, then the length of BC is 2 × ∛36 -/
theorem parabola_triangle_length (t : ParabolaTriangle) (h_area : (1/2) * (t.C.1 - t.B.1) * t.C.2 = 72) :
  Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2) = 2 * (36 : ℝ)^(1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_length_l32_3271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_sum_equals_i_l32_3259

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_sum_equals_i : i^5 + i^13 + i^(-7 : ℤ) = i := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_sum_equals_i_l32_3259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_two_l32_3288

def f (x : ℚ) : ℚ :=
  if x < 0 then 2 * x + 3
  else if x > 0 then -2 * x + 5
  else 0

theorem f_negative_two : f (-2) = -1 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the if-then-else expression
  simp
  -- The result follows directly from arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_two_l32_3288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limited_factor_product_divisors_l32_3247

/-- A natural number with more than 2 but less than 5 factors -/
def LimitedFactorNumber (n : ℕ) : Prop :=
  2 < (Nat.factors n).length ∧ (Nat.factors n).length < 5

theorem limited_factor_product_divisors
  (a b c : ℕ)
  (ha : LimitedFactorNumber a)
  (hb : LimitedFactorNumber b)
  (hc : LimitedFactorNumber c)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hac : a ≠ c) :
  (Finset.card (Nat.divisors (a^2 * b^3 * c^4))) = 1250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limited_factor_product_divisors_l32_3247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_and_perpendicular_implies_perpendicular_unique_perpendicular_plane_l32_3212

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

/-- Two lines are perpendicular -/
def perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

/-- A line is parallel to a plane -/
def parallel_to_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line is perpendicular to a plane -/
def perpendicular_to_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A point is in a plane -/
def point_in_plane (point : ℝ × ℝ × ℝ) (p : Plane3D) : Prop :=
  sorry

theorem parallel_and_perpendicular_implies_perpendicular 
  (l1 l2 : Line3D) (p : Plane3D) :
  parallel_to_plane l1 p → perpendicular_to_plane l2 p → perpendicular l1 l2 :=
by sorry

theorem unique_perpendicular_plane 
  (l1 l2 : Line3D) :
  perpendicular l1 l2 →
  ∃! (p : Plane3D), point_in_plane l1.point p ∧ perpendicular_to_plane l2 p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_and_perpendicular_implies_perpendicular_unique_perpendicular_plane_l32_3212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_monotone_increasing_l32_3250

-- Define the function f(x) = x + k/x
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x + k / x

-- Theorem for the parity of f(x)
theorem f_is_odd (k : ℝ) (h : k ≠ 0) :
  ∀ x : ℝ, x ≠ 0 → f k (-x) = -(f k x) := by
  intro x hx
  simp [f]
  field_simp [hx]
  ring

-- Theorem for the monotonicity of f(x) when k > 0
theorem f_monotone_increasing (k : ℝ) (h : k > 0) :
  ∀ x₁ x₂ : ℝ, x₁ ≥ Real.sqrt k → x₂ ≥ Real.sqrt k → x₁ < x₂ → f k x₁ < f k x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_monotone_increasing_l32_3250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beam_equation_l32_3257

/-- The number of beams that can be bought under the given conditions -/
def num_beams (x : ℕ) : Prop := 
  3 * (x - 1) * x = 6210

/-- The total price of the beams in wen -/
def total_price : ℕ := 6210

/-- The transportation cost per beam in wen -/
def transport_cost : ℕ := 3

theorem beam_equation (x : ℕ) : 
  num_beams x → (transport_cost * (x - 1) * x = total_price) :=
by
  intro h
  exact h


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beam_equation_l32_3257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l32_3274

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (√3a, c) = 3(sin A, cos C), then C = π/3 and (3√3 + 3)/2 < a + b + c ≤ 9/2 -/
theorem triangle_property (a b c A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Triangle is acute
  A + B + C = π ∧          -- Sum of angles in a triangle
  (Real.sqrt 3 * a, c) = (3 * Real.sin A, 3 * Real.cos C) → -- Given condition
  C = π/3 ∧                -- First conclusion
  (3 * Real.sqrt 3 + 3)/2 < a + b + c ∧ a + b + c ≤ 9/2 -- Second conclusion
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l32_3274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l32_3210

/-- Curve C₁ defined by y² = 4x -/
def C₁ (x y : ℝ) : Prop := y^2 = 4 * x

/-- Curve C₂ defined by √3x - y - 2√3 = 0 -/
def C₂ (x y : ℝ) : Prop := Real.sqrt 3 * x - y - 2 * Real.sqrt 3 = 0

/-- Point P with coordinates (2,0) -/
def P : ℝ × ℝ := (2, 0)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

theorem intersection_distance_product :
  ∃ A B : ℝ × ℝ,
    C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧
    C₂ A.1 A.2 ∧ C₂ B.1 B.2 ∧
    A ≠ B ∧
    (distance P A) * (distance P B) = 32 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l32_3210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pauline_spend_l32_3276

/-- The total amount Pauline will spend on school supplies --/
noncomputable def total_spend (total_before_tax : ℝ) (discount_rate : ℝ) (discount_limit : ℝ) (tax_rate : ℝ) : ℝ :=
  let discounted_amount := min total_before_tax discount_limit
  let discount := discount_rate * discounted_amount
  let subtotal := total_before_tax - discount
  let tax := tax_rate * subtotal
  subtotal + tax

/-- Theorem stating the total amount Pauline will spend --/
theorem pauline_spend :
  total_spend 250 0.15 100 0.08 = 253.80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pauline_spend_l32_3276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l32_3295

/-- The eccentricity of an ellipse given specific conditions -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let C : ℝ → ℝ → Prop := λ x y ↦ x^2/a^2 + y^2/b^2 = 1
  let A : ℝ × ℝ := (1, Real.sqrt 3 / 2)
  ∀ (F₁ F₂ : ℝ × ℝ), 
    C A.1 A.2 →
    (∀ x y, C x y → Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) + 
                     Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2) = 2*a) →
    Real.sqrt ((A.1 - F₁.1)^2 + (A.2 - F₁.2)^2) + 
    Real.sqrt ((A.1 - F₂.1)^2 + (A.2 - F₂.2)^2) = 4 →
    Real.sqrt (a^2 - b^2) / a = Real.sqrt 3 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l32_3295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_a_quotient_l32_3202

def factorial_a (n a : ℕ) : ℕ :=
  let k := (n / a) - 1
  List.range (k + 1) |>.foldl (λ acc i => acc * (n - i * a)) 1

theorem factorial_a_quotient :
  (factorial_a 72 8) / (factorial_a 18 2) = 4^9 := by
  -- Proof steps would go here
  sorry

#eval factorial_a 72 8
#eval factorial_a 18 2
#eval (factorial_a 72 8) / (factorial_a 18 2)
#eval 4^9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_a_quotient_l32_3202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_count_l32_3200

noncomputable def initial_height : ℝ := 20
noncomputable def bounce_ratio : ℝ := 1/2
noncomputable def height_threshold : ℝ := 0.5

noncomputable def bounce_height (k : ℕ) : ℝ := initial_height * bounce_ratio ^ k

theorem ball_bounce_count :
  ∃ k : ℕ, (∀ n < k, bounce_height n ≥ height_threshold) ∧
           bounce_height k < height_threshold ∧
           k = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_count_l32_3200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_equivalence_to_power_of_two_l32_3223

theorem equality_equivalence_to_power_of_two (a b : ℝ) : a = b ↔ (2 : ℝ)^a = (2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_equivalence_to_power_of_two_l32_3223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_correction_l32_3287

/-- Represents the class exam scores -/
structure ExamScores where
  students : ℕ
  scores : List ℚ
  deriving Repr

/-- Calculates the average of a list of scores -/
noncomputable def average (scores : List ℚ) : ℚ :=
  scores.sum / scores.length

/-- Calculates the variance of a list of scores -/
noncomputable def variance (scores : List ℚ) : ℚ :=
  let avg := average scores
  (scores.map (fun x => (x - avg) ^ 2)).sum / scores.length

/-- Updates the scores list by replacing two incorrect scores -/
def updateScores (scores : List ℚ) (incorrect1 incorrect2 correct1 correct2 : ℚ) : List ℚ :=
  scores.map (fun x => if x == incorrect1 then correct1 else if x == incorrect2 then correct2 else x)

theorem exam_score_correction (originalScores : ExamScores) 
  (h1 : originalScores.students = 50)
  (h2 : average originalScores.scores = 70)
  (h3 : variance originalScores.scores = 102)
  (h4 : originalScores.scores.contains 50)
  (h5 : originalScores.scores.contains 90) :
  let correctedScores := updateScores originalScores.scores 50 90 80 60
  average correctedScores = 70 ∧ variance correctedScores = 90 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_score_correction_l32_3287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_pair_satisfying_inequality_l32_3277

theorem exists_pair_satisfying_inequality (S : Finset ℝ) (h : S.card = 4) :
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (1 + a * b) / (Real.sqrt (1 + a^2) * Real.sqrt (1 + b^2)) > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_pair_satisfying_inequality_l32_3277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l32_3216

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the property of being an odd function
def IsOdd (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

-- Define the concept of symmetry about a point for a function
def SymmetricAboutPoint (h : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, h x = y ↔ h (2*a - x) = 2*b - y

-- Define symmetry between two functions about the line x - y = 0
def SymmetricAboutDiagonal (h k : ℝ → ℝ) : Prop :=
  ∀ x y, h x = y ↔ k y = x

-- State the theorem
theorem center_of_symmetry
  (h1 : IsOdd (fun x ↦ f (x - 1)))
  (h2 : SymmetricAboutDiagonal f g) :
  SymmetricAboutPoint g 0 (-1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l32_3216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l32_3291

-- Define the function f(x) = √(x^2 + 2x - 3)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 2*x - 3)

-- Define the domain of f
def domain : Set ℝ := {x : ℝ | x^2 + 2*x - 3 ≥ 0}

-- Define the monotonic decreasing interval
def monotonic_decreasing_interval : Set ℝ := Set.Iic (-3 : ℝ)

-- Theorem statement
theorem f_monotonic_decreasing :
  ∀ x ∈ domain, ∀ y ∈ domain,
    x < y → x ∈ monotonic_decreasing_interval → y ∈ monotonic_decreasing_interval →
    f x ≥ f y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l32_3291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_cone_ratio_l32_3229

/-- A cone with the property that its side surface, when unfolded, is a quarter of a full circle -/
structure SpecialCone where
  r : ℝ  -- radius of the base
  l : ℝ  -- slant height
  h : 2 * Real.pi * r = (1/2) * Real.pi * l  -- condition for quarter circle

/-- The ratio of surface area to lateral area for a SpecialCone -/
noncomputable def surfaceToLateralRatio (cone : SpecialCone) : ℝ :=
  (Real.pi * cone.r^2 + Real.pi * cone.r * cone.l) / (Real.pi * cone.r * cone.l)

theorem special_cone_ratio (cone : SpecialCone) : 
  surfaceToLateralRatio cone = 5/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_cone_ratio_l32_3229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l32_3289

theorem solve_exponential_equation :
  ∃ x : ℝ, 8 * (4 : ℝ) ^ x = 2048 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l32_3289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l32_3226

/-- Represents the time taken for a train to cross a platform -/
noncomputable def cross_time (train_length platform_length : ℝ) (speed : ℝ) : ℝ :=
  (train_length + platform_length) / speed

theorem train_crossing_time 
  (train_length : ℝ) 
  (platform1_length : ℝ) 
  (platform2_length : ℝ) 
  (time2 : ℝ) :
  train_length = 100 →
  platform1_length = 200 →
  platform2_length = 300 →
  time2 = 20 →
  cross_time train_length platform1_length 
    ((train_length + platform2_length) / time2) = 15 := by
  sorry

#eval "Theorem defined successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l32_3226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l32_3290

theorem solve_equation (x y z k p : ℝ) : 
  x = 9 → y = 343 → z = 2 →
  (x / 3) ^ 32 * (y / 125) ^ k * (z^3 / 7^3) ^ p = 1 / (27 ^ 32 * z^15) →
  k = 5 ∧ p = -5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l32_3290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_special_case_l32_3224

/-- An isosceles right triangle with side length s -/
structure IsoscelesRightTriangle where
  s : ℝ
  s_pos : s > 0

/-- The perimeter of an isosceles right triangle -/
noncomputable def perimeter (t : IsoscelesRightTriangle) : ℝ :=
  2 * t.s + t.s * Real.sqrt 2

/-- The area of the circumscribed circle of an isosceles right triangle -/
noncomputable def circumscribedCircleArea (t : IsoscelesRightTriangle) : ℝ :=
  Real.pi * t.s^2 / 2

/-- The length of the hypotenuse of an isosceles right triangle -/
noncomputable def hypotenuse (t : IsoscelesRightTriangle) : ℝ :=
  t.s * Real.sqrt 2

/-- Theorem: For an isosceles right triangle whose perimeter equals the area of its circumscribed circle,
    the length of the hypotenuse is (2√2 * (2 + √2)) / π -/
theorem hypotenuse_length_special_case :
  ∀ t : IsoscelesRightTriangle,
    perimeter t = circumscribedCircleArea t →
    hypotenuse t = (2 * Real.sqrt 2 * (2 + Real.sqrt 2)) / Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_special_case_l32_3224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l32_3219

theorem inequality_proof (a b c : ℝ) 
  (ha : a = Real.log 2)
  (hb : b = 5^(-1/2 : ℝ))
  (hc : c = ∫ (x : ℝ) in Set.Icc 0 1, x) :
  b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l32_3219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_S_n_l32_3252

-- Define an arithmetic sequence with positive common difference
noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
noncomputable def S_n (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1) * d) / 2

-- Theorem statement
theorem minimize_S_n (a₁ : ℝ) (d : ℝ) (h_d : d > 0) :
  (arithmetic_sequence a₁ d 5)^2 = (arithmetic_sequence a₁ d 1) * (arithmetic_sequence a₁ d 6) →
  (∀ n : ℕ, S_n a₁ d n ≥ S_n a₁ d 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_S_n_l32_3252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_nested_brackets_l32_3256

theorem simplify_nested_brackets (a b c : ℝ) : -(a - (b - c)) = -a + b - c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_nested_brackets_l32_3256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_timmy_skateboard_speed_difference_l32_3272

/-- The difference between the required speed and the average of trial speeds -/
noncomputable def speed_difference (trial_speeds : List ℝ) (required_speed : ℝ) : ℝ :=
  required_speed - (trial_speeds.sum / trial_speeds.length)

/-- Theorem stating the speed difference for Timmy's skateboard ramp problem -/
theorem timmy_skateboard_speed_difference :
  let trial_speeds : List ℝ := [36, 34, 38]
  let required_speed : ℝ := 40
  speed_difference trial_speeds required_speed = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_timmy_skateboard_speed_difference_l32_3272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_transformation_l32_3206

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := 2 * x^2

/-- The transformed parabola function -/
def transformed_parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 + 3

/-- Function that moves a parabola to the right by h units and up by k units -/
def move_parabola (f : ℝ → ℝ) (h k : ℝ) : ℝ → ℝ := λ x ↦ f (x - h) + k

theorem parabola_transformation :
  ∀ x, transformed_parabola x = move_parabola original_parabola 1 3 x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_transformation_l32_3206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_quotient_theorem_l32_3285

theorem square_quotient_theorem (a b : ℕ+) (h : (a.val * b.val + 1) ∣ (a.val ^ 2 + b.val ^ 2)) :
  ∃ k : ℕ+, (a.val ^ 2 + b.val ^ 2) / (a.val * b.val + 1) = k.val ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_quotient_theorem_l32_3285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vertex_trajectory_l32_3241

/-- Given a triangle ABC with perimeter 20, vertex B at (0, -4), and vertex C at (0, 4),
    the trajectory of vertex A forms an ellipse with equation x^2/20 + y^2/36 = 1 (x ≠ 0) -/
theorem triangle_vertex_trajectory (A B C : ℝ × ℝ) : 
  (B = (0, -4) ∧ C = (0, 4)) → 
  (dist A B + dist A C + dist B C = 20) →
  ∃ (x y : ℝ), x ≠ 0 ∧ A = (x, y) ∧ x^2/20 + y^2/36 = 1 :=
by sorry

/-- Helper function to calculate the distance between two points -/
noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vertex_trajectory_l32_3241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_ratio_increasing_conditions_l32_3264

/-- A function f is first-order ratio increasing on (0, +∞) if f(x)/x is increasing on (0, +∞) -/
def FirstOrderRatioIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → (f x / x) < (f y / y)

/-- A function f is second-order ratio increasing on (0, +∞) if f(x)/x² is increasing on (0, +∞) -/
def SecondOrderRatioIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → (f x / x^2) < (f y / y^2)

/-- The main theorem about the range of m for which f(x) = x³ - 2mx² - mx
    is first-order ratio increasing but not second-order ratio increasing -/
theorem range_of_m_for_ratio_increasing_conditions (m : ℝ) :
  (∀ x, 0 < x → FirstOrderRatioIncreasing (fun x => x^3 - 2*m*x^2 - m*x)) ∧
  (¬ ∀ x, 0 < x → SecondOrderRatioIncreasing (fun x => x^3 - 2*m*x^2 - m*x)) ↔
  m < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_ratio_increasing_conditions_l32_3264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_twenty_percent_of_pentagon_l32_3245

/-- A pentagon formed by an isosceles right triangle on top of a square -/
structure IsoscelesRightTriangleSquarePentagon where
  /-- The side length of the isosceles right triangle -/
  x : ℝ

/-- The side length of the square -/
noncomputable def square_side (p : IsoscelesRightTriangleSquarePentagon) : ℝ :=
  p.x * Real.sqrt 2

/-- The area of the isosceles right triangle -/
noncomputable def triangle_area (p : IsoscelesRightTriangleSquarePentagon) : ℝ :=
  (p.x^2) / 2

/-- The area of the square -/
noncomputable def square_area (p : IsoscelesRightTriangleSquarePentagon) : ℝ :=
  (square_side p)^2

/-- The total area of the pentagon -/
noncomputable def pentagon_area (p : IsoscelesRightTriangleSquarePentagon) : ℝ :=
  triangle_area p + square_area p

/-- The theorem stating that the area of the triangle is 20% of the pentagon's area -/
theorem triangle_area_is_twenty_percent_of_pentagon 
  (p : IsoscelesRightTriangleSquarePentagon) : 
  triangle_area p / pentagon_area p = 1/5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_twenty_percent_of_pentagon_l32_3245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_numbers_l32_3280

def has_72_divisors (n : ℕ) : Prop :=
  (Finset.filter (λ d ↦ n % d = 0) (Finset.range (n + 1))).card = 72

def ends_with_seven_zeros (n : ℕ) : Prop :=
  n % 10000000 = 0

theorem sum_of_special_numbers :
  ∃ (a b : ℕ), a ≠ b ∧
    has_72_divisors a ∧
    has_72_divisors b ∧
    ends_with_seven_zeros a ∧
    ends_with_seven_zeros b ∧
    a + b = 70000000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_numbers_l32_3280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_stack_rearrangement_l32_3297

theorem card_stack_rearrangement (n : ℕ) : 
  (∃ (stack : List ℕ) (pile_a pile_b : List ℕ),
    -- Initial conditions
    stack.length = 2 * n ∧ 
    stack = List.range (2 * n) ∧
    pile_a.length = n ∧
    pile_b.length = n ∧
    -- Rearrangement process
    (∀ i, i < n → 
      (pile_a ++ pile_b).get! (2 * i) = pile_b.get! i ∧
      (pile_a ++ pile_b).get! (2 * i + 1) = pile_a.get! i) ∧
    -- Card 252 retains its position
    stack.get! 251 = (pile_a ++ pile_b).get! 251) →
  n = 252 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_stack_rearrangement_l32_3297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_public_consultation_benefits_l32_3279

/-- Represents the public consultation process in Jinan city -/
structure PublicConsultation where
  draft_documents : Bool
  public_response_channels : Bool

/-- Benefits of the public consultation process -/
inductive Benefit
  | reflect_opinion
  | enhance_enthusiasm

/-- Theorem stating that the public consultation process in Jinan city leads to specific benefits -/
theorem public_consultation_benefits (pc : PublicConsultation) 
  (h1 : pc.draft_documents = true) 
  (h2 : pc.public_response_channels = true) : 
  (Benefit.reflect_opinion ∈ Set.univ) ∧ (Benefit.enhance_enthusiasm ∈ Set.univ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_public_consultation_benefits_l32_3279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_to_reverse_100_chips_l32_3213

/-- Represents a strip of chips -/
def Strip := List Nat

/-- Cost of swapping adjacent chips -/
def adjacent_swap_cost : Nat := 1

/-- Reverses the order of chips in a strip -/
def reverse_strip (s : Strip) : Strip := s.reverse

/-- Represents a swap operation -/
inductive SwapOp
| Adjacent : Nat → SwapOp  -- Index of first chip in adjacent swap
| Free : Nat → SwapOp      -- Index of first chip in free swap (3 chips between)

/-- Applies a swap operation to a strip -/
def apply_swap (s : Strip) (op : SwapOp) : Strip :=
  match op with
  | SwapOp.Adjacent i => sorry
  | SwapOp.Free i => sorry

/-- Calculates the cost of a sequence of swap operations -/
def swap_sequence_cost (ops : List SwapOp) : Nat :=
  ops.foldl (fun acc op => match op with
    | SwapOp.Adjacent _ => acc + 1
    | SwapOp.Free _ => acc) 0

/-- Theorem: The minimum number of rubles to reverse 100 chips is 50 -/
theorem min_cost_to_reverse_100_chips :
  ∀ (s : Strip) (ops : List SwapOp),
    s.length = 100 →
    reverse_strip s = ops.foldl apply_swap s →
    swap_sequence_cost ops ≥ 50 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_to_reverse_100_chips_l32_3213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sample_fifth_element_l32_3220

/-- Represents a systematic sample of students -/
structure SystematicSample where
  totalStudents : Nat
  sampleSize : Nat
  samples : Fin sampleSize → Nat

/-- Checks if a given sample is valid according to systematic sampling rules -/
def isValidSystematicSample (s : SystematicSample) : Prop :=
  ∀ i j : Fin s.sampleSize, i.val < j.val →
    ∃ k : Nat, s.samples j - s.samples i = k * (s.totalStudents / s.sampleSize)

theorem systematic_sample_fifth_element
  (s : SystematicSample)
  (h1 : s.totalStudents = 40)
  (h2 : s.sampleSize = 5)
  (h3 : s.samples ⟨0, by simp [h2]⟩ = 2)
  (h4 : s.samples ⟨1, by simp [h2]⟩ = 10)
  (h5 : s.samples ⟨2, by simp [h2]⟩ = 18)
  (h6 : s.samples ⟨3, by simp [h2]⟩ = 34)
  (h7 : isValidSystematicSample s) :
  s.samples ⟨4, by simp [h2]⟩ = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sample_fifth_element_l32_3220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_function_correct_l32_3209

/-- The cost function for a product over time with yearly cost reduction -/
def cost_function (a : ℝ) (p : ℝ) (m : ℕ) : ℝ :=
  a * (1 - p) ^ m

/-- Theorem stating that the cost function correctly describes the product cost over time -/
theorem cost_function_correct (a : ℝ) (p : ℝ) (m : ℕ) :
  cost_function a p m = a * (1 - p) ^ m :=
by
  -- Unfold the definition of cost_function
  unfold cost_function
  -- The equality now holds by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_function_correct_l32_3209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l32_3269

def a : ℝ × ℝ := (4, 3)
def b : ℝ × ℝ := (-1, 2)

theorem vector_problem :
  (let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)));
   Real.cos θ = 2 * Real.sqrt 5 / 25) ∧
  (let lambda : ℝ := 52 / 9;
   (a.1 - lambda * b.1) * (2 * a.1 + b.1) + (a.2 - lambda * b.2) * (2 * a.2 + b.2) = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l32_3269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_game_optimal_strategy_l32_3266

/-- Represents the reward structure and probabilities for a two-question quiz game. -/
structure QuizGame where
  a1 : ℝ  -- Reward for correctly answering Question 1
  a2 : ℝ  -- Reward for correctly answering Question 2
  p1 : ℝ  -- Probability of correctly answering Question 1
  p2 : ℝ  -- Probability of correctly answering Question 2

/-- Calculates the expected reward for answering Question 1 first. -/
def expected_reward_q1_first (game : QuizGame) : ℝ :=
  game.a1 * game.p1^2 + (game.a1 + game.a2) * game.p1 * (1 - game.p1)

/-- Calculates the expected reward for answering Question 2 first. -/
def expected_reward_q2_first (game : QuizGame) : ℝ :=
  game.a2 * (1 - game.p1)^2 + (game.a1 + game.a2) * game.p1 * (1 - game.p1)

/-- Determines which question should be answered first based on the game parameters. -/
noncomputable def optimal_strategy (game : QuizGame) : String :=
  if game.p1 > Real.sqrt 2 - 1 then "Answer Question 1 first"
  else if game.p1 < Real.sqrt 2 - 1 then "Answer Question 2 first"
  else "The order doesn't matter"

theorem quiz_game_optimal_strategy (game : QuizGame) 
  (h1 : game.a1 = 2 * game.a2) 
  (h2 : game.p1 + game.p2 = 1) 
  (h3 : 0 < game.p1 ∧ game.p1 < 1) :
  optimal_strategy game = 
    if game.p1 > Real.sqrt 2 - 1 then "Answer Question 1 first"
    else if game.p1 < Real.sqrt 2 - 1 then "Answer Question 2 first"
    else "The order doesn't matter" :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_game_optimal_strategy_l32_3266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_same_coins_l32_3236

/-- Represents the number of jars -/
def num_jars : ℕ := 2017

/-- Represents the number of consecutive jars filled each day -/
def consecutive_jars : ℕ := 10

/-- Represents a configuration of coins in jars -/
def JarConfiguration := Fin num_jars → ℕ

/-- Represents a valid jar configuration after some number of days -/
def is_valid_configuration (config : JarConfiguration) : Prop :=
  ∃ (days : ℕ), ∀ i, config i ≤ days

/-- The number of jars with the same number of coins in a configuration -/
noncomputable def num_same_coins (config : JarConfiguration) : ℕ :=
  Finset.card (Finset.filter (λ n => ∃ i, config i = n ∧ n > 0) (Finset.range (num_jars + 1)))

/-- The main theorem: The maximum number of jars with the same number of coins is 2014 -/
theorem max_same_coins :
  ∃ (config : JarConfiguration), is_valid_configuration config ∧
    (∀ (other_config : JarConfiguration), is_valid_configuration other_config →
      num_same_coins other_config ≤ num_same_coins config) ∧
    num_same_coins config = 2014 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_same_coins_l32_3236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_numbers_l32_3207

def is_valid_number (n : ℕ) : Bool :=
  50 < n && n < 350 && n % 10 = 6

def valid_numbers : List ℕ := List.filter is_valid_number (List.range 351)

theorem sum_of_valid_numbers : List.sum valid_numbers = 6030 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_valid_numbers_l32_3207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_exactly_one_zero_l32_3246

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := |Real.exp (x * Real.log 2) - 1| - Real.exp (x * Real.log 3)

-- Theorem statement
theorem f_has_exactly_one_zero :
  ∃! x : ℝ, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_exactly_one_zero_l32_3246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_after_tipping_l32_3298

/-- Represents a rectangular prism with water -/
structure WaterPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  waterDepth : ℝ

/-- Calculates the volume of water in the prism -/
noncomputable def waterVolume (p : WaterPrism) : ℝ :=
  p.length * p.width * p.waterDepth

/-- Calculates the new water depth when the prism is tipped on its largest face -/
noncomputable def newWaterDepth (p : WaterPrism) : ℝ :=
  waterVolume p / (max (p.length * p.width) (max (p.length * p.height) (p.width * p.height)))

/-- Theorem stating that tipping the prism results in the expected water depth -/
theorem water_depth_after_tipping (p : WaterPrism) :
  p.length = 2 ∧ p.width = 5 ∧ p.height = 8 ∧ p.waterDepth = 6 →
  newWaterDepth p = 1.5 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_after_tipping_l32_3298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_restoration_percentage_l32_3292

theorem price_restoration_percentage (original_price : ℝ) (reduced_price : ℝ) 
  (h1 : reduced_price = original_price * (1 - 0.15)) 
  (h2 : original_price > 0) : 
  ∃ (increase_percentage : ℝ), 
    (reduced_price * (1 + increase_percentage) = original_price) ∧ 
    (abs (increase_percentage - 0.1765) < 0.0001) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_restoration_percentage_l32_3292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_characterization_l32_3281

noncomputable def curve1 (x : ℝ) : ℝ := 2 - Real.sqrt (1 - 4^x)
noncomputable def curve2 (a x : ℝ) : ℝ := a * 2^x

def intersection_points (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve1 p.1 = curve2 a p.1}

theorem intersection_characterization (a : ℝ) :
  (a < Real.sqrt 3 → intersection_points a = ∅) ∧
  (a = Real.sqrt 3 → intersection_points a = {(1/2 * Real.log 3 / Real.log 2, 3/2)}) ∧
  (Real.sqrt 3 < a ∧ a ≤ 2 →
    intersection_points a = {
      (Real.log ((2*a - Real.sqrt (a^2 - 3)) / (a^2 + 1)) / Real.log 2,
       2 - (2 + a * Real.sqrt (a^2 - 3)) / (a^2 + 1)),
      (Real.log ((2*a + Real.sqrt (a^2 - 3)) / (a^2 + 1)) / Real.log 2,
       2 + (a * Real.sqrt (a^2 - 3) - 2) / (a^2 + 1))
    }) ∧
  (2 < a →
    intersection_points a = {
      (Real.log ((2*a - Real.sqrt (a^2 - 3)) / (a^2 + 1)) / Real.log 2,
       2 - (2 + a * Real.sqrt (a^2 - 3)) / (a^2 + 1))
    }) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_characterization_l32_3281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l32_3263

def is_valid_number (n : ℕ) : Bool :=
  1000 ≤ n ∧ n ≤ 9999 ∧  -- four-digit number
  n % 2 = 0 ∧  -- even number
  let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
  List.Pairwise (· < ·) digits  -- strictly increasing order

theorem count_valid_numbers : 
  (Finset.filter (fun n => is_valid_number n) (Finset.range 10000)).card = 46 := by
  sorry

#eval (Finset.filter (fun n => is_valid_number n) (Finset.range 10000)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l32_3263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_condition_inequality_solution_set_l32_3283

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 2|

-- Theorem for part I
theorem min_value_condition (a : ℝ) :
  (∀ x, f a x ≥ 2) ∧ (∃ x, f a x = 2) ↔ a = 0 ∨ a = -4 :=
sorry

-- Theorem for part II
theorem inequality_solution_set :
  {x : ℝ | f 2 x ≤ 6} = Set.Icc (-3) 3 :=
sorry

-- Where Set.Icc is the closed interval notation in Lean

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_condition_inequality_solution_set_l32_3283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l32_3232

/-- The length of a train given its relative speed and time to pass -/
noncomputable def train_length (relative_speed : ℝ) (passing_time : ℝ) : ℝ :=
  relative_speed * passing_time

/-- Convert speed from km/h to m/s -/
noncomputable def kmph_to_mps (speed : ℝ) : ℝ :=
  speed * (1000 / 3600)

theorem faster_train_length
  (faster_speed slower_speed : ℝ)
  (passing_time : ℝ)
  (h1 : faster_speed = 54)
  (h2 : slower_speed = 36)
  (h3 : passing_time = 27) :
  train_length (kmph_to_mps (faster_speed - slower_speed)) passing_time = 135 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_train_length_l32_3232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_equivalence_l32_3237

/-- Represents a natural number as a list of its digits -/
def Digits := List Nat

/-- Reverses the digits of a number -/
def reverse_digits : Digits → Digits :=
  List.reverse

/-- Checks if a list of digits has at most one zero -/
def at_most_one_zero (d : Digits) : Prop :=
  (d.filter (· = 0)).length ≤ 1

/-- Converts a list of digits to a natural number -/
def digits_to_nat (d : Digits) : Nat :=
  d.foldl (fun acc x => acc * 10 + x) 0

/-- Generates a list of k 9's -/
def nines (k : Nat) : Digits :=
  List.replicate k 9

/-- The set of solutions to the problem -/
def solution_set : Set Nat :=
  {0, 1089} ∪ {digits_to_nat (1 :: 0 :: nines k ++ [8, 9]) | k : Nat}

/-- The main theorem stating the equivalence of the problem conditions and the solution set -/
theorem problem_equivalence (N : Nat) : 
  (∃ d : Digits, 
    N = digits_to_nat d ∧ 
    9 * N = digits_to_nat (reverse_digits d) ∧
    at_most_one_zero d) ↔ 
  N ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_equivalence_l32_3237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_onto_plane_l32_3201

noncomputable def normal_vector : Fin 3 → ℝ := ![2, -1, 2]

noncomputable def projection_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.of ![
    ![5/9,  2/9, -4/9],
    ![2/9, 10/9, -2/9],
    ![-4/9,  2/9,  5/9]
  ]

theorem projection_onto_plane (w : Fin 3 → ℝ) :
  let projected := Matrix.mulVec projection_matrix w
  (Matrix.dotProduct projected normal_vector = 0) ∧
  (w - projected = (Matrix.dotProduct w normal_vector / Matrix.dotProduct normal_vector normal_vector) • normal_vector) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_onto_plane_l32_3201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_monotonicity_l32_3260

def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem extreme_values_and_monotonicity 
  (a b c : ℝ) 
  (h1 : ∃ (k : ℝ), (deriv (f a b c)) (-2/3) = k ∧ (deriv (f a b c)) 1 = k) :
  (a = -1/2 ∧ b = -2) ∧ 
  (∀ x y : ℝ, 
    ((x < -2/3 ∧ y < -2/3) ∨ (x > 1 ∧ y > 1)) ∧ x < y → f a b c x < f a b c y) ∧
  (∀ x y : ℝ, -2/3 < x ∧ x < y ∧ y < 1 → f a b c x > f a b c y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_values_and_monotonicity_l32_3260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l32_3270

/-- The equation of the region -/
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 20*y = -100

/-- The area of the region -/
noncomputable def region_area : ℝ := 9 * Real.pi

/-- Theorem: The area enclosed by the region defined by the equation
    x^2 + y^2 - 6x + 20y = -100 is equal to 9π -/
theorem area_of_region :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y : ℝ, region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l32_3270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_at_least_five_l32_3282

/-- Checks if a positive integer has no zero digits -/
def has_no_zero_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d ≠ 0

/-- Generates all digit rearrangements of a positive integer -/
noncomputable def digit_rearrangements (n : ℕ) : Finset ℕ :=
  sorry

/-- Checks if a positive integer is composed entirely of ones -/
def all_ones (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 1

/-- Main theorem -/
theorem digit_at_least_five (n : ℕ) (h_pos : 0 < n) (h_no_zero : has_no_zero_digits n)
  (h_sum : ∃ r₁ r₂ r₃, r₁ ∈ digit_rearrangements n ∧ r₂ ∈ digit_rearrangements n ∧ 
           r₃ ∈ digit_rearrangements n ∧ all_ones (n + r₁ + r₂ + r₃)) :
  ∃ d ∈ n.digits 10, 5 ≤ d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_at_least_five_l32_3282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_third_quadrant_l32_3248

theorem complex_number_in_third_quadrant 
  (θ : ℝ) 
  (h : 0 < θ ∧ θ < π/2) : 
  let z : ℂ := Complex.mk (Real.cos (3*π/2 - θ)) (Real.sin (π + θ))
  z.re < 0 ∧ z.im < 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_third_quadrant_l32_3248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_segments_theorem_l32_3254

/-- Represents an intersection point of lines -/
structure IntersectionPoint where
  lambda : ℕ  -- Number of lines meeting at this point

/-- Represents a set of intersecting lines -/
structure IntersectingLines where
  n : ℕ  -- Number of lines
  points : List IntersectionPoint  -- List of intersection points

/-- The number of segments formed by intersecting lines -/
def num_segments (lines : IntersectingLines) : ℕ :=
  lines.n + (lines.points.map IntersectionPoint.lambda).sum

/-- Theorem stating that the number of segments equals n + ∑λ(P) -/
theorem num_segments_theorem (lines : IntersectingLines) :
  num_segments lines = lines.n + (lines.points.map IntersectionPoint.lambda).sum :=
by
  -- The proof is trivial due to the definition of num_segments
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_segments_theorem_l32_3254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l32_3204

/-- An ellipse with foci F₁ and F₂, and a point M on the ellipse -/
structure Ellipse :=
  (F₁ F₂ M : ℝ × ℝ)
  (center : ℝ × ℝ)
  (a : ℝ)
  (c : ℝ)

/-- The circle centered at F₂ -/
def ellipseCircle (E : Ellipse) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | dist p E.F₂ = E.c}

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity (E : Ellipse) : ℝ := E.c / E.a

/-- A line through two points -/
def Line (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | ∃ t : ℝ, r = (1 - t) • p + t • q}

/-- Predicate for a line being tangent to a circle -/
def isTangent (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ l ∧ p ∈ c ∧ ∀ q : ℝ × ℝ, q ∈ l ∧ q ∈ c → q = p

/-- The statement to be proved -/
theorem ellipse_eccentricity (E : Ellipse) 
  (h1 : E.center ∈ ellipseCircle E) 
  (h2 : E.M ∈ ellipseCircle E) 
  (h3 : isTangent (Line E.M E.F₁) (ellipseCircle E)) :
  eccentricity E = Real.sqrt 3 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l32_3204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l32_3211

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 2 then 2^x else x^2

-- State the theorem
theorem range_of_a (a : ℝ) :
  f (a + 1) ≥ f (2*a - 1) → a ≤ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l32_3211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_line_equation_l32_3231

/-- Two points are symmetric about a line if the line is perpendicular to the segment connecting the points and passes through its midpoint. -/
def IsSymmetric (P Q : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop := 
  ∃ (M : ℝ × ℝ), M ∈ l ∧ M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) ∧
  ∀ (x y : ℝ), (x, y) ∈ l → (y - M.2) = (x - M.1)

/-- Given two points are symmetric about a line, prove the equation of the line. -/
theorem symmetric_points_line_equation (P Q : ℝ × ℝ) (l : Set (ℝ × ℝ)) 
  (h_symmetric : IsSymmetric P Q l) 
  (h_P : P = (3, 2)) 
  (h_Q : Q = (1, 4)) :
  l = {(x, y) : ℝ × ℝ | x - y + 1 = 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_points_line_equation_l32_3231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l32_3225

noncomputable section

/-- The number of days it takes to complete the work -/
def D : ℝ := 11

/-- The total amount of work to be done -/
def W : ℝ := 1

/-- The rate at which person a works -/
noncomputable def rate_a : ℝ := W / 24

/-- The rate at which person b works -/
noncomputable def rate_b : ℝ := W / 30

/-- The rate at which person c works -/
noncomputable def rate_c : ℝ := W / 40

/-- The combined rate of all three people working together -/
noncomputable def combined_rate : ℝ := rate_a + rate_b + rate_c

/-- The combined rate of a and b working together -/
noncomputable def rate_ab : ℝ := rate_a + rate_b

theorem work_completion_time : 
  W = combined_rate * (D - 4) + rate_ab * 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l32_3225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l32_3239

/-- Two circles in a 2D plane -/
structure TwoCircles where
  C₁ : ℝ → ℝ → Prop := fun x y ↦ (x - 2)^2 + (y - 1)^2 = 4
  C₂ : ℝ → ℝ → Prop := fun x y ↦ x^2 + (y - 2)^2 = 9

/-- The line passing through the intersection points of two circles -/
def intersectionLine : ℝ → ℝ → Prop :=
  fun x y ↦ 2 * x - 3 * y = 0

/-- Theorem stating that the line passing through the intersection points
    of the given circles has the equation 2x - 3y = 0 -/
theorem intersection_line_equation (circles : TwoCircles) :
    ∀ x y : ℝ, circles.C₁ x y ∧ circles.C₂ x y → intersectionLine x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_equation_l32_3239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l32_3205

/-- The parabola y^2 = 8x -/
def Parabola (P : ℝ × ℝ) : Prop :=
  (P.2)^2 = 8 * P.1

/-- Distance between two points in ℝ² -/
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem min_distance_sum :
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (7, 4)
  ∃ (min : ℝ), min = 9 ∧
    ∀ P : ℝ × ℝ, Parabola P →
      distance A P + distance B P ≥ min :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l32_3205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_greater_than_fn_and_y_bounds_l32_3261

noncomputable def fn (n : ℕ) (x : ℝ) : ℝ :=
  Finset.sum (Finset.range (n + 1)) (λ k => (1 / (Nat.factorial k : ℝ)) * x^k)

theorem exponential_greater_than_fn_and_y_bounds
  (x : ℝ) (hx : x > 0) (n : ℕ) :
  (∃ y : ℝ, Real.exp x = fn n x + (1 / (Nat.factorial (n + 1) : ℝ)) * x^(n + 1) * Real.exp y) →
  (Real.exp x > fn n x ∧ ∃ y : ℝ, 0 < y ∧ y < x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_greater_than_fn_and_y_bounds_l32_3261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l32_3217

noncomputable def g (x : ℝ) : ℝ := Real.sin x ^ 6 + Real.cos x ^ 2

theorem g_range :
  Set.range g = Set.Icc ((3 * Real.sqrt 3 - 2) / (3 * Real.sqrt 3)) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l32_3217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l32_3275

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop := 3 * x + 4 * y = 0

-- Define a point on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop := 
  hyperbola P.1 P.2

-- Define the foci
def foci (F₁ F₂ : ℝ × ℝ) : Prop := 
  ∃ c : ℝ, F₁ = (-c, 0) ∧ F₂ = (c, 0) ∧ c^2 = 16 + 9

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ := 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem hyperbola_foci_distance 
  (P F₁ F₂ : ℝ × ℝ) 
  (h_point : point_on_hyperbola P) 
  (h_foci : foci F₁ F₂) 
  (h_dist : distance P F₁ = 10) :
  distance P F₂ = 2 ∨ distance P F₂ = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l32_3275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_divisible_by_three_l32_3284

def ball_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

def is_sum_divisible_by_three (x y : ℕ) : Bool :=
  (x + y) % 3 = 0

def favorable_outcomes : Finset (ℕ × ℕ) :=
  ball_numbers.product ball_numbers

def total_outcomes : ℕ :=
  ball_numbers.card.choose 2

theorem probability_sum_divisible_by_three :
  (favorable_outcomes.filter (λ (x, y) => x < y ∧ is_sum_divisible_by_three x y)).card / total_outcomes = 1 / 3 := by
  sorry

#eval (favorable_outcomes.filter (λ (x, y) => x < y ∧ is_sum_divisible_by_three x y)).card
#eval total_outcomes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_divisible_by_three_l32_3284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l32_3242

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a / (x + 1)

def decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem range_of_a (a : ℝ) :
  (decreasing (f a) 1 2 ∧ decreasing (g a) 1 2) → 0 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l32_3242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_gcd_condition_l32_3278

def a : ℕ → ℕ
  | 0 => 3
  | n + 1 => a n + n * (a n - 1)

theorem power_of_two_gcd_condition (m : ℕ) (h : m ≥ 2) :
  (∀ n, Nat.gcd m (a n) = 1) ↔ ∃ k : ℕ, m = 2^k ∧ k ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_gcd_condition_l32_3278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f₁_correct_domain_f₂_correct_l32_3299

-- Function 1
noncomputable def f₁ (x : ℝ) : ℝ := Real.log (1 + 1/x) + Real.sqrt (1 - x^2)

-- Function 2
noncomputable def f₂ (x : ℝ) : ℝ := Real.log (x + 1) / Real.sqrt (-x^2 - 3*x + 4)

-- Domain of f₁
def domain_f₁ : Set ℝ := Set.Ioc 0 1

-- Domain of f₂
def domain_f₂ : Set ℝ := Set.Ioo (-1) 1

theorem domain_f₁_correct :
  ∀ x, f₁ x ∈ Set.range f₁ ↔ x ∈ domain_f₁ := by sorry

theorem domain_f₂_correct :
  ∀ x, f₂ x ∈ Set.range f₂ ↔ x ∈ domain_f₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f₁_correct_domain_f₂_correct_l32_3299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_Sp_equals_137500_l32_3203

/-- Sum of arithmetic progression -/
def sum_ap (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Sum of Sp for p from 1 to 10 -/
def sum_Sp : ℕ :=
  Finset.sum (Finset.range 10) (fun p => sum_ap (p + 1) (2 * (p + 1)) 50)

theorem sum_Sp_equals_137500 : sum_Sp = 137500 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_Sp_equals_137500_l32_3203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l32_3234

/-- Definition of the function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - a / (2^x + 1)

theorem problem_solution :
  /- (1) If f(-1) = -1, then a = -3 -/
  (∀ a : ℝ, f a (-1) = -1 → a = -3) ∧
  /- (2) f(x) is an odd function if and only if a = 2 -/
  (∀ a : ℝ, (∀ x : ℝ, f a (-x) = -(f a x)) ↔ a = 2) ∧
  /- (3) The equation f(x) = 0 has a solution in ℝ if and only if a ∈ (1, +∞) -/
  (∀ a : ℝ, (∃ x : ℝ, f a x = 0) ↔ a > 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l32_3234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l32_3262

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3

-- State the theorem
theorem range_of_x (x : ℝ) : 
  (f (x - 1) ≤ 2) → (1 < x ∧ x ≤ 10) :=
by
  -- Introduce the hypothesis
  intro h
  -- Apply sorry to skip the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l32_3262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_conjugate_magnitude_l32_3238

theorem complex_conjugate_magnitude (z : ℂ) : (1 - Complex.I) = (2 + 4 * Complex.I) / z → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_conjugate_magnitude_l32_3238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_circle_l32_3222

/-- The line x - y = 0 -/
def line (x y : ℝ) : Prop := x = y

/-- The circle (x-8)^2 + y^2 = 2 -/
def circle_eq (x y : ℝ) : Prop := (x - 8)^2 + y^2 = 2

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem min_distance_line_circle :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    line x₁ y₁ ∧ circle_eq x₂ y₂ ∧
    (∀ (x₃ y₃ x₄ y₄ : ℝ), line x₃ y₃ → circle_eq x₄ y₄ → 
      distance x₁ y₁ x₂ y₂ ≤ distance x₃ y₃ x₄ y₄) ∧
    distance x₁ y₁ x₂ y₂ = 3 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_circle_l32_3222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_smallest_elements_l32_3294

/-- Given an integer n ≥ 4, M is the set {1, 2, 3, ..., n} -/
def M (n : ℕ) : Finset ℕ := Finset.range n

/-- A_i is a 3-element subset of M -/
def A (n : ℕ) : Finset (Finset ℕ) := (M n).powerset.filter (fun s => s.card = 3)

/-- m_i is the smallest element in A_i -/
noncomputable def m (n : ℕ) (s : Finset ℕ) : ℕ := s.min' sorry

/-- P_n is the sum of all m_i -/
noncomputable def P (n : ℕ) : ℕ := (A n).sum (fun s => m n s)

/-- The main theorem -/
theorem sum_smallest_elements (n : ℕ) (h : n ≥ 4) : P n = Nat.choose (n + 1) 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_smallest_elements_l32_3294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_constant_product_l32_3258

noncomputable section

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- A line passing through (1,0) -/
def line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

/-- Intersection points of the line and the ellipse -/
def intersection (k : ℝ) (x y : ℝ) : Prop := ellipse x y ∧ line k x y

/-- The fixed point E -/
def E : ℝ × ℝ := (17/8, 0)

/-- The dot product of vectors PE and QE -/
def dot_product (P Q : ℝ × ℝ) : ℝ :=
  let (px, py) := P
  let (qx, qy) := Q
  let (ex, ey) := E
  (ex - px) * (ex - qx) + (ey - py) * (ey - qy)

theorem ellipse_intersection_constant_product :
  ∀ (k : ℝ) (P Q : ℝ × ℝ),
  (∃ (px py qx qy : ℝ), P = (px, py) ∧ Q = (qx, qy) ∧
    intersection k px py ∧ intersection k qx qy ∧ P ≠ Q) →
  dot_product P Q = 33/64 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_constant_product_l32_3258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_l32_3233

/-- The radius of the base circle of a cone formed from a circular sector -/
noncomputable def base_circle_radius (sector_angle : ℝ) (sector_radius : ℝ) : ℝ :=
  (sector_angle / 360) * sector_radius

/-- Theorem: The radius of the base circle of a cone is 4/3 -/
theorem cone_base_radius :
  base_circle_radius 120 4 = 4/3 := by
  -- Unfold the definition of base_circle_radius
  unfold base_circle_radius
  -- Simplify the expression
  simp [mul_div_assoc, mul_comm]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_l32_3233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_height_max_length_l32_3215

noncomputable def f (x : ℝ) := 2 * Real.cos x * Real.sin (x + Real.pi / 6) + Real.sqrt 3 * Real.sin x * Real.cos x - Real.sin x ^ 2

theorem triangle_height_max_length 
  (A B C : ℝ × ℝ) 
  (h_angle : 0 < A.1 ∧ A.1 < Real.pi) 
  (h_f : f A.1 = 2) 
  (h_dot : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = Real.sqrt 3) :
  ∃ (D : ℝ × ℝ), 
    (D.1 - A.1) * (B.2 - A.2) = (D.2 - A.2) * (B.1 - A.1) ∧ 
    (D.1 - A.1) * (C.2 - A.2) = (D.2 - A.2) * (C.1 - A.1) ∧
    Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) ≤ (Real.sqrt 3 + 1) / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_height_max_length_l32_3215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_theorem_l32_3249

/-- The distance between intersection points of y = 3 and y = 5x^2 + x - 2 -/
noncomputable def intersection_distance : ℝ :=
  let x₁ := (-1 + Real.sqrt 101) / 10
  let x₂ := (-1 - Real.sqrt 101) / 10
  |x₁ - x₂|

/-- The numerator and denominator of the simplified distance fraction -/
def m : ℕ := 101
def n : ℕ := 5

theorem intersection_distance_theorem :
  intersection_distance = Real.sqrt m / n ∧ m - n = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_theorem_l32_3249
