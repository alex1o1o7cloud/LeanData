import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_sin_cos_inequality_l917_91766

theorem largest_n_sin_cos_inequality : 
  ∀ n : ℕ, n > 0 →
    (∀ x : ℝ, (Real.sin x)^(n : ℝ) + (Real.cos x)^(n : ℝ) ≥ 2 / (n : ℝ)) ↔ n ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_sin_cos_inequality_l917_91766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_locus_l917_91782

-- Define the circle C
noncomputable def C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Define the tangent line
def is_tangent_line (l : ℝ → ℝ → Prop) : Prop :=
  (∀ x y, l x y → C x y) ∧
  (l P.1 P.2) ∧
  (∀ x y, C x y → l x y → (x = P.1 ∧ y = P.2))

-- Define point M on circle C
noncomputable def M (x₀ y₀ : ℝ) : Prop := C x₀ y₀

-- Define point N
def N (y₀ : ℝ) : ℝ × ℝ := (0, y₀)

-- Define point Q as midpoint of MN
noncomputable def Q (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀/2, y₀)

theorem tangent_line_and_locus :
  (∃ l : ℝ → ℝ → Prop, is_tangent_line l ∧
    ((∀ x y, l x y ↔ x = 2) ∨
     (∀ x y, l x y ↔ 3*x + 4*y - 10 = 0))) ∧
  (∀ x y, (∃ x₀ y₀, M x₀ y₀ ∧ Q x₀ y₀ = (x, y)) ↔ x^2 + y^2/4 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_locus_l917_91782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_average_is_75_l917_91778

/-- Represents the number of wise people in a room of 100 people. -/
def num_wise : ℕ := sorry

/-- Represents the number of optimists in the room. -/
def num_optimists : ℕ := 100 - num_wise

/-- The sum of numbers written down by all people in the room. -/
def sum_written : ℕ := num_wise * num_wise + num_optimists * 100

/-- The average of numbers written down by all people in the room. -/
noncomputable def average_written : ℚ := sum_written / 100

/-- Theorem stating that the minimum average of numbers written down is 75. -/
theorem min_average_is_75 : 
  ∀ (num_wise : ℕ), num_wise ≤ 100 → average_written ≥ 75 := by
  sorry

#check min_average_is_75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_average_is_75_l917_91778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_function_property_l917_91765

/-- A function that is odd and strictly decreasing on [-1,0] -/
def OddDecreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ 0 → f x > f y)

/-- Acute angles of a triangle -/
def AcuteAnglesOfTriangle (α β : ℝ) : Prop :=
  0 < α ∧ α < Real.pi/2 ∧ 0 < β ∧ β < Real.pi/2 ∧ α + β < Real.pi

theorem odd_decreasing_function_property 
  (f : ℝ → ℝ) (α β : ℝ) 
  (hf : OddDecreasingFunction f) 
  (hαβ : AcuteAnglesOfTriangle α β) : 
  f (Real.sin α) < f (Real.cos β) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_function_property_l917_91765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_properties_l917_91754

-- Define a triangle type
structure Triangle where
  base : ℝ
  height : ℝ

-- Define the area of a triangle
noncomputable def area (t : Triangle) : ℝ := t.base * t.height / 2

theorem triangle_area_properties :
  (∃ t1 t2 : Triangle, area t1 = area t2 ∧ (t1.base ≠ t2.base ∨ t1.height ≠ t2.height)) ∧
  (∀ t1 t2 : Triangle, t1.base = t2.base ∧ t1.height = t2.height → area t1 = area t2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_properties_l917_91754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l917_91758

noncomputable def geometric_sequence (q : ℝ) (n : ℕ) : ℝ := q^(n-1)

noncomputable def geometric_sum (q : ℝ) (n : ℕ) : ℝ := (q^n - 1) / (q - 1)

theorem geometric_sequence_properties (q : ℝ) (h1 : q > 0) :
  let a := geometric_sequence q
  let S := geometric_sum q
  (4 * a 3 = a 2 * a 4) →
  (q = 2) ∧
  (a 3 = 4) ∧
  (∀ n : ℕ, n ≥ 1 → S n / a n < 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l917_91758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_drink_size_is_two_l917_91719

-- Define the given information
noncomputable def first_drink_size : ℝ := 12
noncomputable def first_drink_caffeine : ℝ := 250
noncomputable def total_caffeine : ℝ := 750
noncomputable def caffeine_multiplier : ℝ := 3

-- Define the function to calculate the size of the second drink
noncomputable def second_drink_size : ℝ :=
  let first_drink_concentration := first_drink_caffeine / first_drink_size
  let second_drink_concentration := caffeine_multiplier * first_drink_concentration
  let total_drinks_caffeine := total_caffeine / 2
  let second_drink_caffeine := total_drinks_caffeine - first_drink_caffeine
  second_drink_caffeine / second_drink_concentration

-- Theorem stating that the second drink size is 2 ounces
theorem second_drink_size_is_two :
  second_drink_size = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_drink_size_is_two_l917_91719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_g_piecewise_l917_91728

-- Define the function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 3
  else if x ≤ 1 then -x^2 + 2
  else if x ≤ 4 then x - 2
  else 0  -- for x outside [-4, 4]

-- State the theorem about |g(x)|
theorem abs_g_piecewise (x : ℝ) :
  abs (g x) =
    if x < -3 then -x - 3
    else if x ≤ -1 then x + 3
    else if x ≤ 1 then -x^2 + 2
    else if x < 2 then 2 - x
    else if x ≤ 4 then x - 2
    else 0 := by sorry

-- Additional lemmas to help prove the main theorem
lemma g_neg_four_to_neg_one (x : ℝ) (h : -4 ≤ x ∧ x ≤ -1) :
  g x = x + 3 := by sorry

lemma g_neg_one_to_one (x : ℝ) (h : -1 < x ∧ x ≤ 1) :
  g x = -x^2 + 2 := by sorry

lemma g_one_to_four (x : ℝ) (h : 1 < x ∧ x ≤ 4) :
  g x = x - 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_g_piecewise_l917_91728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_double_angle_formulas_l917_91797

theorem triangle_double_angle_formulas (A : ℝ) (h1 : Real.cos A = 5/13) (h2 : 0 < A) (h3 : A < Real.pi) :
  Real.sin (2 * A) = 120/169 ∧ Real.cos (2 * A) = -119/169 ∧ Real.tan (2 * A) = -120/119 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_double_angle_formulas_l917_91797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_value_m_range_l917_91752

-- Define the complex numbers z₁ and z₂
def z₁ (b : ℝ) : ℂ := 1 + b * Complex.I
def z₂ : ℂ := 2 - 3 * Complex.I

-- Define the vector Z₁Z₂
def vector_Z₁Z₂ (b : ℝ) : ℂ := z₂ - z₁ b

-- Define the condition for Z₁Z₂ being parallel to the real axis
def parallel_to_real_axis (b : ℝ) : Prop := Complex.im (vector_Z₁Z₂ b) = 0

-- Define z as a function of m
def z (m : ℝ) : ℂ := (m + z₁ (-3)) ^ 2

-- Define the condition for z being in the third quadrant
def in_third_quadrant (m : ℝ) : Prop :=
  Complex.re (z m) < 0 ∧ Complex.im (z m) < 0

-- Theorem statements
theorem b_value (b : ℝ) : parallel_to_real_axis b → b = -3 := by sorry

theorem m_range (m : ℝ) : in_third_quadrant m ↔ -1 < m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_value_m_range_l917_91752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_l917_91746

/-- Daniel's current age -/
def d : ℕ := sorry

/-- Daniel's father's current age -/
def f : ℕ := sorry

/-- Daniel's age is one-ninth of his father's age -/
axiom daniel_age : f = 9 * d

/-- One year from now, Daniel's father's age will be seven times Daniel's age -/
axiom future_age : f + 1 = 7 * (d + 1)

theorem age_difference : f - d = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_difference_l917_91746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_squares_factorization_l917_91792

/-- Represents a factorization transformation -/
def is_factorization (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g x ∧ ∃ p q : ℝ → ℝ, g x = p x * q x

/-- The transformation a^2 - 9 = (a-3)(a+3) represents factorization -/
theorem difference_of_squares_factorization :
  is_factorization (λ a ↦ a^2 - 9) (λ a ↦ (a - 3) * (a + 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_squares_factorization_l917_91792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_4_equals_32_l917_91732

-- Define the function g
def g : ℝ → ℝ := λ x ↦ 2 * x

-- State the theorem
theorem f_of_4_equals_32 
  (f : ℝ → ℝ) 
  (h : ∀ x y z : ℝ, f (x^2 + y * f z) = x * g x + z * g y) :
  f 4 = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_4_equals_32_l917_91732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_problem_l917_91786

/-- Two-digit positive integer -/
def TwoDigitPositiveInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Decimal representation of a number with x before the decimal point and 8y after -/
noncomputable def DecimalRep (x y : ℕ) : ℝ := x + (8 * y : ℝ) / 100

theorem smallest_integer_problem (x y : ℕ) 
  (hx : TwoDigitPositiveInt x) (hy : TwoDigitPositiveInt y)
  (h_avg : (x + y : ℝ) / 2 = DecimalRep x y) :
  min x y = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_problem_l917_91786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_loss_percentage_l917_91720

/-- Calculates the loss percentage for a shopkeeper --/
theorem shopkeeper_loss_percentage 
  (profit_rate : ℝ) 
  (theft_rate : ℝ) 
  (profit_rate_value : profit_rate = 0.1)
  (theft_rate_value : theft_rate = 0.2) :
  let selling_price := 1 + profit_rate
  let remaining_goods := 1 - theft_rate
  let remaining_selling_price := selling_price * remaining_goods
  let loss := theft_rate
  ∃ (ε : ℝ), abs (loss / remaining_selling_price - 0.2273) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_loss_percentage_l917_91720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l917_91727

noncomputable def f (x : ℝ) := 1 / (x - 5) + Real.sqrt (x + 2)

theorem domain_of_f :
  {x : ℝ | x ≥ -2 ∧ x ≠ 5} = {x : ℝ | f x ≠ 0 ∨ f x = 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l917_91727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_english_textbook_cost_is_7_50_l917_91750

/-- The cost of an English textbook given the following conditions:
  * 35 English textbooks and 35 geography textbooks are ordered
  * A geography book costs $10.50
  * The total amount of the order is $630
-/
noncomputable def english_textbook_cost : ℝ := by
  -- Define the number of each type of textbook
  let num_english : ℕ := 35
  let num_geography : ℕ := 35
  
  -- Define the cost of a geography textbook
  let geography_cost : ℝ := 10.50
  
  -- Define the total cost of the order
  let total_cost : ℝ := 630
  
  -- Calculate the cost of an English textbook
  let english_cost : ℝ := (total_cost - (↑num_geography * geography_cost)) / ↑num_english
  
  -- Return the cost of an English textbook
  exact english_cost

/-- Theorem stating that the cost of an English textbook is $7.50 -/
theorem english_textbook_cost_is_7_50 : english_textbook_cost = 7.50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_english_textbook_cost_is_7_50_l917_91750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_em_equals_mf_l917_91713

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the points
variable (A B C O D E F M : V)

-- Define the conditions
def is_acute_triangle (A B C : V) : Prop := sorry
def is_longer_side (A B C : V) : Prop := sorry
def is_circumcenter (O A B C : V) : Prop := sorry
def is_midpoint (D B C : V) : Prop := sorry
def on_circle_with_diameter (E A D : V) : Prop := sorry
def is_parallel (v w : V) : Prop := sorry
def lies_on_line (M E F : V) : Prop := sorry

-- State the theorem
theorem em_equals_mf 
  (h1 : is_acute_triangle A B C)
  (h2 : is_longer_side A B C)
  (h3 : is_circumcenter O A B C)
  (h4 : is_midpoint D B C)
  (h5 : on_circle_with_diameter E A D)
  (h6 : on_circle_with_diameter F A D)
  (h7 : is_parallel (D - M) (A - O))
  (h8 : lies_on_line M E F) :
  ‖E - M‖ = ‖M - F‖ := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_em_equals_mf_l917_91713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_B_is_correct_l917_91714

-- Define the faces of the cube
inductive Face
| White
| Grey
| U
| V
| Other1
| Other2

-- Define the cube as a structure
structure Cube where
  faces : List Face
  adjacency : Face → Face → Bool
  orientation : Face → Face → Bool

-- Define the cardboard layout
def cardboard_layout : Cube :=
{ faces := [Face.White, Face.Grey, Face.U, Face.V, Face.Other1, Face.Other2],
  adjacency := λ f1 f2 => 
    match f1, f2 with
    | Face.White, Face.U => true
    | Face.U, Face.White => true
    | Face.Grey, Face.U => true
    | Face.U, Face.Grey => true
    | Face.White, Face.Grey => false
    | Face.Grey, Face.White => false
    | Face.U, Face.V => false
    | Face.V, Face.U => false
    | _, _ => true
  orientation := λ f1 f2 =>
    match f1, f2 with
    | Face.White, Face.U => true
    | Face.U, Face.Grey => false
    | _, _ => true }

-- Define the cube configuration B
def cube_B : Cube :=
{ faces := [Face.White, Face.Grey, Face.U, Face.V, Face.Other1, Face.Other2],
  adjacency := λ _ _ => true,  -- Simplified for this example
  orientation := λ _ _ => true }  -- Simplified for this example

-- Theorem statement
theorem cube_B_is_correct : 
  ∀ (c : Cube), c = cardboard_layout → c = cube_B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_B_is_correct_l917_91714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_first_quadrant_l917_91715

theorem complex_number_in_first_quadrant (z : ℂ) (h : (2 - Complex.I) * z = 1 + Complex.I) : 
  z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_first_quadrant_l917_91715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l917_91737

noncomputable def f (x : ℝ) := Real.sin x * Real.cos x + Real.sqrt 3 * Real.cos (Real.pi - x) * Real.cos x

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, 0 < S → S < T → ∃ y, f (y + S) ≠ f y)) ∧
  (∃ M : ℝ, M = 1 - Real.sqrt 3 / 2 ∧ ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ M) ∧
  (∃ m : ℝ, m = -Real.sqrt 3 ∧ ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → m ≤ f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l917_91737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_possible_radii_l917_91723

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the regular hexagon
def regularHexagon : Set (ℝ × ℝ) := sorry

-- Define the circles
def circles : Fin 6 → Circle := sorry

-- Define the intersection points
def P (i : Fin 6) : ℝ × ℝ := sorry

-- Define the Q points
def Q (i : Fin 6) : ℝ × ℝ := sorry

-- Define a function to create a circle from a center and radius
def mkCircle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) = radius}

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  (r.2 - p.2) * (q.1 - p.1) = (q.2 - p.2) * (r.1 - p.1)

-- State the theorem
theorem five_possible_radii 
  (h1 : ∀ i : Fin 6, (circles i).center ∈ regularHexagon)
  (h2 : ∀ i : Fin 6, (circles i).radius = (circles 0).radius)
  (h3 : ∀ i : Fin 6, P i ∈ mkCircle (circles i).center (circles i).radius ∩ 
                               mkCircle (circles (i + 1)).center (circles (i + 1)).radius)
  (h4 : ∀ i : Fin 6, collinear (Q i) (P i) (Q (i + 1)))
  (h5 : ∀ i : Fin 6, Q i ∈ mkCircle (circles i).center (circles i).radius) :
  ∃ (s : Finset ℝ), s.card = 5 ∧ (circles 0).radius ∈ s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_possible_radii_l917_91723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l917_91707

theorem binomial_expansion_coefficient : 
  (Nat.choose 10 3) * (4/5 : ℚ)^3 * (-1/3 : ℚ)^7 = -256/91125 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l917_91707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_symmetry_l917_91736

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin (2 * x + Real.pi / 4)

noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f (x + φ)

theorem shift_symmetry (φ : ℝ) (h1 : 0 < φ) (h2 : φ < Real.pi / 2) :
  (∀ x, g φ x = g φ (-x)) → φ = Real.pi / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_symmetry_l917_91736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABF_l917_91724

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the line L
def L (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the left focus F
def F : ℝ × ℝ := (-1, 0)

-- Define points A and B as the intersection of C and L
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define the area of a triangle function
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem area_of_triangle_ABF :
  C A.1 A.2 ∧ C B.1 B.2 ∧ L A.1 A.2 ∧ L B.1 B.2 →
  area_triangle A B F = 12 * Real.sqrt 2 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ABF_l917_91724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_angle_in_range_l917_91716

-- Define the set of possible outcomes for a fair die roll
def DieOutcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the vector types
def Vector2D := ℝ × ℝ

-- Define the angle between two vectors
noncomputable def angle (v w : Vector2D) : ℝ := Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

-- State the theorem
theorem probability_angle_in_range :
  let m_n_pairs := DieOutcomes.product DieOutcomes
  let valid_pairs := m_n_pairs.filter (fun p => 0 < angle (p.1, p.2) (1, 0) ∧ angle (p.1, p.2) (1, 0) < Real.pi / 4)
  (valid_pairs.card : ℚ) / m_n_pairs.card = 5 / 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_angle_in_range_l917_91716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_zero_S_is_correct_l917_91718

/-- Sequence S defined recursively -/
def S : ℕ → ℤ
  | 0 => 0
  | (k + 1) => S k + (k + 1) * a (k + 1)
where
  /-- Definition of a -/
  a (i : ℕ) : ℤ :=
    if i = 0 then 0 else if S (i - 1) < i then 1 else -1

/-- The largest k ≤ 2010 such that S k = 0 -/
def largest_zero_S : ℕ := 1092

/-- Theorem stating that largest_zero_S is correct -/
theorem largest_zero_S_is_correct :
  S largest_zero_S = 0 ∧
  ∀ k, largest_zero_S < k → k ≤ 2010 → S k ≠ 0 := by
  sorry

#eval S largest_zero_S

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_zero_S_is_correct_l917_91718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_l917_91798

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is internally tangent to another circle -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c2.radius - c1.radius)^2

/-- Checks if three points are collinear -/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem circle_tangency_theorem (C1 C2 C3 : Circle) (m n p : ℕ) :
  are_externally_tangent C1 C2 →
  is_internally_tangent C1 C3 →
  is_internally_tangent C2 C3 →
  C1.radius = 5 →
  C2.radius = 11 →
  are_collinear C1.center C2.center C3.center →
  (m : ℝ) * Real.sqrt n / p = 2 * Real.sqrt ((C3.radius)^2 - 11^2) →
  Nat.Coprime m p →
  ∀ (q : ℕ), q > 1 → Nat.Prime q → ¬(q^2 ∣ n) →
  m + n + p = 108 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_l917_91798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_theorem_l917_91749

/-- Represents the production function for articles based on workers, hours, and days. -/
noncomputable def production_function (x y z : ℝ) : ℝ := y^3 / (x * z)

/-- Theorem stating the production function for y workers given the initial conditions for x workers. -/
theorem production_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let initial_production := x -- x men working x hours a day for x days produce x articles
  let productivity_decrease := z -- productivity decreases by a factor of z for each additional worker
  production_function x y z = y^3 / (x * z) :=
by
  -- Unfold the definition of production_function
  unfold production_function
  -- The equation holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_theorem_l917_91749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_density_functions_correct_l917_91742

-- Define the joint density function
noncomputable def joint_density (x y : ℝ) : ℝ := 
  (1 / Real.pi) * Real.exp (-(1/2) * (x^2 + 2*x*y + 3*y^2))

-- Define the marginal density function for X
noncomputable def marginal_density_x (x : ℝ) : ℝ := 
  Real.sqrt (2 / (3 * Real.pi)) * Real.exp (-(1/3) * x^2)

-- Define the marginal density function for Y
noncomputable def marginal_density_y (y : ℝ) : ℝ := 
  Real.sqrt (2 / Real.pi) * Real.exp (-y^2)

-- Define the conditional density function of X given Y
noncomputable def conditional_density_x_given_y (x y : ℝ) : ℝ := 
  (1 / Real.sqrt (2 * Real.pi)) * Real.exp (-(1/2) * (x + y)^2)

-- Define the conditional density function of Y given X
noncomputable def conditional_density_y_given_x (y x : ℝ) : ℝ := 
  Real.sqrt (3 / (2 * Real.pi)) * Real.exp (-(3/2) * (y + (1/3) * x)^2)

-- Theorem statement
theorem density_functions_correct :
  ∀ (x y : ℝ),
    (∫ y, joint_density x y) = marginal_density_x x ∧
    (∫ x, joint_density x y) = marginal_density_y y ∧
    joint_density x y / marginal_density_y y = conditional_density_x_given_y x y ∧
    joint_density x y / marginal_density_x x = conditional_density_y_given_x y x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_density_functions_correct_l917_91742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_cosine_curve_l917_91756

open Real MeasureTheory

-- Define the cosine function
noncomputable def f (x : ℝ) : ℝ := Real.cos x

-- Define the interval
noncomputable def a : ℝ := 0
noncomputable def b : ℝ := 3 * Real.pi / 2

-- State the theorem
theorem area_enclosed_by_cosine_curve : 
  ∫ x in a..b, max 0 (f x) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_cosine_curve_l917_91756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_is_guilty_B_is_guilty_l917_91773

-- Define the set of suspects
inductive Suspect : Type
  | A : Suspect
  | B : Suspect
  | C : Suspect

-- Define the concept of being involved in the theft
variable (involved : Suspect → Prop)

-- Define the concept of going on a job
variable (goesOnJob : Suspect → Prop)

-- Define the concept of knowing how to drive
variable (canDrive : Suspect → Prop)

-- Define the concept of being guilty
variable (guilty : Suspect → Prop)

theorem A_is_guilty :
  -- No one besides A, B, and C was involved in the theft
  (∀ s, involved s ↔ (s = Suspect.A ∨ s = Suspect.B ∨ s = Suspect.C)) →
  -- C never goes on a job without A
  (goesOnJob Suspect.C → goesOnJob Suspect.A) →
  -- B does not know how to drive
  (¬ canDrive Suspect.B) →
  -- Prove that A is guilty
  guilty Suspect.A :=
by
  sorry

-- For the second part of the problem
theorem B_is_guilty :
  -- No one besides A, B, and C was involved in the theft
  (∀ s, involved s ↔ (s = Suspect.A ∨ s = Suspect.B ∨ s = Suspect.C)) →
  -- A never goes on a job without at least one accomplice
  (goesOnJob Suspect.A → (goesOnJob Suspect.B ∨ goesOnJob Suspect.C)) →
  -- C is not guilty
  (¬ guilty Suspect.C) →
  -- Prove that B is guilty
  guilty Suspect.B :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_is_guilty_B_is_guilty_l917_91773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fan_rotation_high_setting_l917_91784

/-- Represents the rotation rate of a fan at different settings -/
inductive FanSetting
| Slow
| Medium
| High

/-- Calculates the rotation rate for a given fan setting -/
def rotationRate : FanSetting → ℕ
| FanSetting.Slow => 100
| FanSetting.Medium => 200
| FanSetting.High => 400

/-- Calculates the number of rotations for a given time in minutes and fan setting -/
def rotationsInTime (minutes : ℕ) (setting : FanSetting) : ℕ :=
  minutes * rotationRate setting

theorem fan_rotation_high_setting :
  rotationsInTime 15 FanSetting.High = 6000 := by
  unfold rotationsInTime
  unfold rotationRate
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fan_rotation_high_setting_l917_91784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l917_91738

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + 2 * Real.cos x

theorem period_of_f : ∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), 0 < q ∧ q < p → ∃ (y : ℝ), f (y + q) ≠ f y := by
  -- We claim that p = 2π
  let p := 2 * Real.pi
  use p
  
  have h_p_pos : p > 0 := by
    apply mul_pos
    · norm_num
    · exact Real.pi_pos

  constructor
  · exact h_p_pos
  · sorry  -- The actual proof would go here, but we'll use sorry as requested


end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_f_l917_91738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_k_value_l917_91751

noncomputable section

-- Define the ellipse M
def M (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define points A and P
def A : ℝ × ℝ := (-2, 0)
def P : ℝ × ℝ := (1, 3/2)

-- Define the slope k
variable (k : ℝ)

-- Assumptions
axiom k_positive : k > 0
axiom A_on_M : M A.1 A.2
axiom P_on_M : M P.1 P.2

-- Define points B and C (their exact coordinates are not needed for the statement)
variable (B C : ℝ × ℝ)

-- Assumptions about B and C
axiom B_on_M : M B.1 B.2
axiom C_on_M : M C.1 C.2
axiom PB_slope : (B.2 - P.2) / (B.1 - P.1) = k
axiom PC_slope : (C.2 - P.2) / (C.1 - P.1) = -k

-- Define what it means for PABC to be a parallelogram
def is_parallelogram (A B C P : ℝ × ℝ) : Prop :=
  (B.1 - A.1 = C.1 - P.1) ∧ (B.2 - A.2 = C.2 - P.2)

-- The theorem to prove
theorem parallelogram_k_value :
  is_parallelogram A B C P → k = 3/2 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_k_value_l917_91751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_parabolas_l917_91780

/-- The area enclosed by two parabolas -/
theorem area_between_parabolas : 
  let f (x : ℝ) := x^2 - 1
  let g (x : ℝ) := 2 - 2*x^2
  ∫ x in (-1 : ℝ)..1, (g x - f x) = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_parabolas_l917_91780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_circumcircle_equation_l917_91775

-- Define the points
def O : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 2)
def C : ℝ × ℝ := (4, 0)

-- Define the distance function
noncomputable def distance (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / Real.sqrt (a^2 + b^2)

-- Theorem for the line equation
theorem line_equation :
  ∃ (k : ℝ), 
    (distance O 1 (-k) (4*k) = distance B 1 (-k) (4*k) ∧
     (k = 1 ∨ k = -1/3)) := by
  sorry

-- Theorem for the circumcircle equation
theorem circumcircle_equation :
  ∀ (x y : ℝ),
    (x - O.1)^2 + (y - O.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
    (x - O.1)^2 + (y - O.2)^2 = (x - C.1)^2 + (y - C.2)^2 →
    x^2 + y^2 - 4*x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_circumcircle_equation_l917_91775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_home_runs_l917_91710

theorem mean_home_runs (players : ℕ) (home_runs : List (ℕ × ℕ)) 
  (h_players : players = 11)
  (h_home_runs : home_runs = [(5, 4), (6, 3), (7, 2), (8, 1), (11, 1)])
  : (home_runs.foldl (fun acc (hr, count) => acc + hr * count) 0) / players = 71 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_home_runs_l917_91710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l917_91777

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*(a + 1)*x + 3 + 2*a*Real.log x

-- State the theorem
theorem function_property (a : ℝ) :
  (∀ x : ℝ, x > 0 → x ≥ 1 → f a x ≥ 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l917_91777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_to_fraction_l917_91791

theorem recurring_decimal_to_fraction (x : ℚ) :
  (∃ n : ℕ, x = (6 : ℚ) / (10^n - 1)) →
  (∃ a b : ℤ, x = a / b ∧ Int.gcd a b = 1 ∧ b = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_to_fraction_l917_91791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_envelope_l917_91767

/-- The envelope of trajectories for a projectile motion --/
theorem projectile_envelope 
  (v₀ : ℝ) -- Initial velocity
  (g : ℝ) -- Acceleration due to gravity
  (h : g > 0) -- Gravity is positive
  (x y : ℝ) -- Coordinates in the plane
  : 
  (∃ α : ℝ, 
    y = v₀ * Real.sin α * (x / (v₀ * Real.cos α)) - 
        (g / 2) * (x / (v₀ * Real.cos α))^2 ∧ 
    0 ≤ α ∧ α < Real.pi/2) 
  → 
  y ≤ v₀^2 / (2*g) - (g / (2*v₀^2)) * x^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_envelope_l917_91767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l917_91793

theorem remainder_problem (k : ℕ) (hk : k > 0) (h : 120 % (k^2) = 24) : 180 % k = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l917_91793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_phi_l917_91729

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem symmetric_function_phi (ω φ : ℝ) (h_ω : ω > 0) (h_φ : -π/2 < φ ∧ φ < π/2)
  (h_symmetric_axes : ∀ x : ℝ, f ω φ (x + π/6) = f ω φ x)
  (h_symmetric_point : f ω φ (5*π/18) = 0) :
  φ = π/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_phi_l917_91729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l917_91788

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

-- Theorem stating that g is an odd function
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- Expand the definition of g
  simp [g, f]
  -- The actual proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l917_91788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_inequality_l917_91755

theorem cube_root_inequality (x : ℝ) (hx : x > 0) :
  x^(1/3) < 3 * x ↔ x > 1 / (3 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_inequality_l917_91755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_digging_time_l917_91757

/-- The time it takes for Jake, Paul, and Hari to dig a well together -/
noncomputable def combined_digging_time (jake_time paul_time hari_time : ℝ) : ℝ :=
  1 / (1 / jake_time + 1 / paul_time + 1 / hari_time)

/-- Theorem stating that Jake, Paul, and Hari can dig the well in 8 days -/
theorem well_digging_time :
  combined_digging_time 16 24 48 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_digging_time_l917_91757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l917_91764

/-- Calculates the increase in area when changing a rectangular garden from 60 feet by 15 feet
    to a new rectangle with the same perimeter and length three times the width. -/
theorem garden_area_increase : 
  let original_length : ℝ := 60
  let original_width : ℝ := 15
  let perimeter : ℝ := 2 * (original_length + original_width)
  let new_width : ℝ := perimeter / 8
  let new_length : ℝ := 3 * new_width
  let original_area : ℝ := original_length * original_width
  let new_area : ℝ := new_length * new_width
  new_area - original_area = 154.6875 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_increase_l917_91764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_of_overlapping_triangles_l917_91796

/-- Represents a right-angled triangle with two 45-degree angles -/
structure Triangle45_45_90 where
  hypotenuse : ℝ
  hypotenuse_positive : hypotenuse > 0

/-- Calculates the area of the common region of two overlapping 45-45-90 triangles -/
noncomputable def commonArea (t1 t2 : Triangle45_45_90) : ℝ :=
  (t1.hypotenuse ^ 2) / 4

/-- Theorem: The area common to two congruent 45-45-90 triangles with hypotenuses of 10 units 
    that overlap partly with coinciding hypotenuses is 25 square units -/
theorem common_area_of_overlapping_triangles :
  ∀ (t1 t2 : Triangle45_45_90), 
    t1.hypotenuse = 10 → 
    t2.hypotenuse = 10 → 
    commonArea t1 t2 = 25 := by
  sorry

#check common_area_of_overlapping_triangles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_area_of_overlapping_triangles_l917_91796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_container_volume_ratio_l917_91722

noncomputable def cylinder_volume (d h : ℝ) : ℝ := (Real.pi / 4) * d^2 * h

theorem container_volume_ratio :
  let jonathan_d : ℝ := 10
  let jonathan_h : ℝ := 15
  let chris_d : ℝ := 15
  let chris_h : ℝ := 10
  (cylinder_volume jonathan_d jonathan_h) / (cylinder_volume chris_d chris_h) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_container_volume_ratio_l917_91722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flow_rates_in_channels_l917_91763

/-- Represents a channel in the irrigation system -/
inductive Channel
| AB | BC | CD | DE | EF | FG | GH | HA | BG | DG

/-- Represents a node in the irrigation system -/
inductive Node
| A | B | C | D | E | F | G | H

/-- The flow rate through a channel -/
noncomputable def flow_rate (c : Channel) : ℝ := sorry

/-- The total input flow rate -/
noncomputable def q₀ : ℝ := sorry

/-- The irrigation system satisfies conservation of flow -/
axiom conservation_of_flow :
  ∀ (path : List Channel), List.sum (List.map flow_rate path) = q₀

/-- The irrigation system is symmetric -/
axiom symmetry :
  flow_rate Channel.BC = flow_rate Channel.CD ∧
  flow_rate Channel.BG = flow_rate Channel.DG

/-- Theorem stating the flow rates in specific channels -/
theorem flow_rates_in_channels :
  flow_rate Channel.DE = (4/7) * q₀ ∧
  flow_rate Channel.BC = (2/7) * q₀ ∧
  flow_rate Channel.FG = (3/7) * q₀ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flow_rates_in_channels_l917_91763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_triangle_area_l917_91779

noncomputable section

/-- The area of a triangle given its base and height -/
def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

/-- The area of a rectangle given its width and height -/
def rectangle_area (width height : ℝ) : ℝ := width * height

theorem central_triangle_area :
  let rectangle_width : ℝ := 6
  let rectangle_height : ℝ := 8
  let total_area := rectangle_area rectangle_width rectangle_height
  let triangle1_area := triangle_area rectangle_width 4
  let triangle2_area := triangle_area rectangle_width 2
  let triangle3_area := triangle_area 4 4
  let sum_small_triangles := triangle1_area + triangle2_area + triangle3_area
  total_area - sum_small_triangles = 22 := by
  -- Unfold definitions
  unfold rectangle_area triangle_area
  -- Simplify expressions
  simp
  -- The proof steps would go here, but we'll use sorry to skip the proof
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_triangle_area_l917_91779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l917_91748

-- Define the power function as noncomputable
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- State the theorem
theorem power_function_through_point (α : ℝ) :
  f α 2 = Real.sqrt 2 → f α 16 = 4 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l917_91748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l917_91762

noncomputable def f (x : ℝ) := Real.sqrt (7 - Real.sqrt (x^2 - 3*x + 2))

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | x ≤ 1 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l917_91762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_divisors_in_range_sum_of_divisors_20_l917_91771

def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (fun d => n % d = 0) (Finset.range (n + 1))

def divisorCount (n : ℕ) : ℕ := (divisors n).card

def divisorSum (n : ℕ) : ℕ := (divisors n).sum id

theorem greatest_divisors_in_range :
  ∀ n ∈ Finset.range 21,
    divisorCount n ≤ divisorCount 20 ∧
    (divisorCount n = divisorCount 20 → n = 20) :=
  sorry

theorem sum_of_divisors_20 : divisorSum 20 = 42 :=
  sorry

#eval divisorCount 20
#eval divisorSum 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_divisors_in_range_sum_of_divisors_20_l917_91771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_1979_l917_91709

/-- Given a list of positive integers with sum 1979, their product is maximized when the list consists of 659 threes and one two. -/
theorem max_product_sum_1979 (a : List ℕ) (h_pos : ∀ x ∈ a, x > 0) :
  (a.sum = 1979) →
  (a.prod ≤ 3^659 * 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_1979_l917_91709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l917_91702

-- Define the line
def line (x y : ℝ) : Prop := 2 * x + y - 4 = 0

-- Define the circle (renamed to avoid conflict with built-in circle)
def my_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Theorem statement
theorem intersection_distance : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    line x₁ y₁ ∧ my_circle x₁ y₁ ∧
    line x₂ y₂ ∧ my_circle x₂ y₂ ∧
    x₁ ≠ x₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l917_91702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bound_for_g_l917_91700

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (2 * x) + Real.sin (2 * x)

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 6) + 1

-- State the theorem
theorem min_bound_for_g : 
  (∃ (a : ℝ), ∀ (x : ℝ), |g x| ≤ a) ∧ 
  (∀ (b : ℝ), (∀ (x : ℝ), |g x| ≤ b) → b ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bound_for_g_l917_91700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_union_equals_22_l917_91770

-- Define the vertices of the original triangle
def A : ℚ × ℚ := (3, 4)
def B : ℚ × ℚ := (5, -2)
def C : ℚ × ℚ := (7, 3)

-- Define the reflection line
def reflection_line : ℚ := 5

-- Function to reflect a point across x = reflection_line
def reflect (p : ℚ × ℚ) : ℚ × ℚ :=
  (2 * reflection_line - p.1, p.2)

-- Define the reflected vertices
def A' : ℚ × ℚ := reflect A
def B' : ℚ × ℚ := reflect B
def C' : ℚ × ℚ := reflect C

-- Function to calculate the area of a triangle given its vertices
def triangle_area (p1 p2 p3 : ℚ × ℚ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))

-- Theorem to prove
theorem area_of_union_equals_22 :
  triangle_area A B C + triangle_area A' B' C' = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_union_equals_22_l917_91770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_51_value_l917_91790

def G : ℕ → ℚ
  | 0 => 3  -- Adding the base case for 0
  | 1 => 3
  | (n + 1) => (3 * G n + 2) / 3

theorem G_51_value : G 51 = 109 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_51_value_l917_91790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_card_is_diamonds_three_l917_91726

/-- Represents a playing card suit -/
inductive Suit where
  | Spades
  | Hearts
  | Diamonds
  | Clubs

/-- Represents a playing card rank -/
inductive Rank where
  | Ace
  | Two
  | Three
  | Four
  | Five
  | Six
  | Seven
  | Eight
  | Nine
  | Ten
  | Jack
  | Queen
  | King

/-- Represents a playing card -/
structure Card where
  suit : Suit
  rank : Rank

/-- Represents special cards -/
inductive SpecialCard where
  | Joker
  | SmallJoker

/-- Represents either a regular card or a special card -/
inductive DeckCard where
  | Regular (card : Card)
  | Special (special : SpecialCard)

/-- The process of discarding and moving cards -/
def discard_process (deck : List DeckCard) : DeckCard :=
  sorry

/-- The initial deck arrangement -/
def initial_deck : List DeckCard :=
  sorry

/-- Theorem stating that the last remaining card is Diamonds 3 -/
theorem last_card_is_diamonds_three :
  discard_process initial_deck = DeckCard.Regular ⟨Suit.Diamonds, Rank.Three⟩ :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_card_is_diamonds_three_l917_91726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_m_l917_91747

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- Whether there exists a positive integer n such that n! ends with exactly m zeros -/
def existsFactorialWithZeros (m : ℕ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ trailingZeros n = m

/-- The set of integers m between 1 and 30 (inclusive) for which there exists an n such that n! ends with exactly m zeros -/
def validM : Set ℕ :=
  {m | 1 ≤ m ∧ m ≤ 30 ∧ existsFactorialWithZeros m}

/-- Decidable predicate for membership in validM -/
instance : DecidablePred (· ∈ validM) := sorry

theorem count_valid_m : Finset.card (Finset.filter (· ∈ validM) (Finset.range 31)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_m_l917_91747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_inner_quadrilateral_l917_91768

/-- Represents a trapezoid with parallel sides and an altitude -/
structure Trapezoid where
  side1 : ℝ
  side2 : ℝ
  altitude : ℝ

/-- Represents a quadrilateral formed within a trapezoid -/
structure InnerQuadrilateral (T : Trapezoid) where
  base1 : ℝ
  base2 : ℝ
  altitude : ℝ

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoidArea (T : Trapezoid) : ℝ :=
  T.altitude * (T.side1 + T.side2) / 2

/-- Calculates the area of an inner quadrilateral -/
noncomputable def innerQuadrilateralArea (T : Trapezoid) (Q : InnerQuadrilateral T) : ℝ :=
  Q.altitude * (Q.base1 + Q.base2) / 2

/-- Main theorem: Area of quadrilateral GHFD in trapezoid ABCD -/
theorem area_of_inner_quadrilateral (T : Trapezoid) 
  (h1 : T.side1 = 10)
  (h2 : T.side2 = 26)
  (h3 : T.altitude = 15) :
  ∃ Q : InnerQuadrilateral T, 
    Q.base1 = (T.side1 + T.side2 / 2) / 2 ∧ 
    Q.base2 = T.side2 / 2 ∧
    Q.altitude = T.altitude / 2 ∧
    innerQuadrilateralArea T Q = 91.875 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_inner_quadrilateral_l917_91768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l917_91708

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The distance from the center to a focus -/
noncomputable def Ellipse.focalDistance (e : Ellipse) : ℝ :=
  e.a * e.eccentricity

/-- The distance from the center to a directrix -/
noncomputable def Ellipse.directrixDistance (e : Ellipse) : ℝ :=
  e.a / e.eccentricity

theorem ellipse_focus_distance 
  (e : Ellipse)
  (h_eq : e.a = 5 ∧ e.b = 3)
  (P : ℝ × ℝ)
  (h_on_ellipse : (P.1 / e.a)^2 + (P.2 / e.b)^2 = 1)
  (h_right_directrix : ∃ (d : ℝ), d = e.directrixDistance ∧ 
    Real.sqrt ((P.1 - d)^2 + P.2^2) = 5) :
  Real.sqrt ((P.1 + e.focalDistance)^2 + P.2^2) = 6 := by
  sorry

#check ellipse_focus_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l917_91708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_t_value_proposition_A_sufficient_not_necessary_l917_91799

-- Define the ellipse equation
noncomputable def ellipse_equation (x y t : ℝ) : Prop :=
  y^2 / (5 - t) + x^2 / (t - 1) = 1

-- Define the eccentricity
noncomputable def eccentricity (t : ℝ) : ℝ :=
  Real.sqrt ((5 - t) - (t - 1)) / Real.sqrt (5 - t)

-- Define the inequality from proposition B
def inequality_B (t : ℝ) : Prop :=
  t^2 - 3*t - 4 < 0

-- Theorem 1: The value of t is 2
theorem ellipse_t_value :
  ∀ t : ℝ, (∀ x y : ℝ, ellipse_equation x y t) →
  (eccentricity t = Real.sqrt 6 / 3) →
  t = 2 :=
by sorry

-- Theorem 2: Proposition A is a sufficient but not necessary condition for B
theorem proposition_A_sufficient_not_necessary :
  (∀ t : ℝ, (1 < t ∧ t < 3) → inequality_B t) ∧
  (∃ t : ℝ, inequality_B t ∧ (t ≤ 1 ∨ t ≥ 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_t_value_proposition_A_sufficient_not_necessary_l917_91799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c₁_c₂_not_collinear_l917_91789

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (2, 0, -5)
def b : ℝ × ℝ × ℝ := (1, -3, 4)

-- Define c₁ and c₂ in terms of a and b
def c₁ : ℝ × ℝ × ℝ := (2 * a.1 - 5 * b.1, 2 * a.2.1 - 5 * b.2.1, 2 * a.2.2 - 5 * b.2.2)
def c₂ : ℝ × ℝ × ℝ := (5 * a.1 - 2 * b.1, 5 * a.2.1 - 2 * b.2.1, 5 * a.2.2 - 2 * b.2.2)

-- Theorem statement
theorem c₁_c₂_not_collinear : ¬ ∃ (k : ℝ), c₁ = (k * c₂.1, k * c₂.2.1, k * c₂.2.2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c₁_c₂_not_collinear_l917_91789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l917_91711

noncomputable def f (x : ℝ) := Real.cos (Real.pi / 2 - x)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l917_91711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_circle_l917_91731

theorem point_on_circle (a b r : ℝ) : 
  (∃ x y : ℝ, a*x + b*y = r^2 ∧ x^2 + y^2 = r^2) → 
  a^2 + b^2 = r^2 := by
  intro h
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_circle_l917_91731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l917_91760

noncomputable def f (ω φ x : Real) : Real := Real.cos (ω * x + φ)

theorem max_omega_value (ω φ : Real) 
  (h_ω_pos : ω > 0) 
  (h_φ_range : 0 < φ ∧ φ < Real.pi) 
  (h_odd : ∀ x, f ω φ x = -f ω φ (-x)) 
  (h_decreasing : ∀ x y, -Real.pi/3 < x ∧ x < y ∧ y < Real.pi/6 → f ω φ x > f ω φ y) :
  ω ≤ 3/2 := by
  sorry

#check max_omega_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l917_91760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_values_l917_91783

/-- The function f(x) as defined in the problem -/
noncomputable def f (x : Real) : Real :=
  Real.sin (2 * Real.pi + x) + Real.sqrt 3 * Real.cos (2 * Real.pi - x) - Real.sin (2013 * Real.pi + Real.pi / 6)

/-- Theorem stating the maximum and minimum values of f(x) in the given interval -/
theorem f_max_min_values :
  ∃ (max min : Real),
    (∀ x, -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2 → f x ≤ max) ∧
    (∃ x, -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = max) ∧
    (∀ x, -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2 → min ≤ f x) ∧
    (∃ x, -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = min) ∧
    max = 5 / 2 ∧ min = -1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_values_l917_91783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_traced_on_smaller_sphere_l917_91753

noncomputable section

-- Define the radii of the spheres
def r_small : ℝ := 3
def r_large : ℝ := 5

-- Define the area traced on the larger sphere
def area_large : ℝ := 1

-- Define the function for the surface area of a sphere
noncomputable def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

-- State the theorem
theorem area_traced_on_smaller_sphere :
  (area_large * surface_area r_small) / surface_area r_large = 9/25 := by
  -- Expand the definition of surface_area
  unfold surface_area
  -- Simplify the expression
  simp [r_small, r_large, area_large]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_traced_on_smaller_sphere_l917_91753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_vertical_asymptote_l917_91712

-- Define the function f(x) with parameter c
noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (x^2 - x + c) / (x^2 + x - 18)

-- Theorem statement
theorem exactly_one_vertical_asymptote (c : ℝ) :
  (∃! x, (x^2 + x - 18 = 0 ∧ x^2 - x + c ≠ 0)) ↔ (c = -6 ∨ c = -42) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_vertical_asymptote_l917_91712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l917_91785

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3 * x - 5) / (x + 4)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≠ 3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l917_91785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_point_symmetry_about_line_symmetry_of_shifted_functions_all_statements_true_l917_91733

-- Define the function f
variable (f : ℝ → ℝ)

-- Statement 1
theorem symmetry_about_point (h1 : ∀ x, f (x + 2) + f (2 - x) = 4) :
  ∀ x y, f x = y ↔ f (4 - x) = 4 - y :=
by sorry

-- Statement 2
theorem symmetry_about_line (h2 : ∀ x, f (x + 2) = f (2 - x)) :
  ∀ x, f x = f (4 - x) :=
by sorry

-- Statement 3
theorem symmetry_of_shifted_functions (h3 : ∀ x, f x = f (4 - x)) :
  ∀ x, f (x - 2) = f (-x + 2) :=
by sorry

-- All statements are true
theorem all_statements_true (f : ℝ → ℝ) :
  (∀ x, f (x + 2) + f (2 - x) = 4) →
  (∀ x, f (x + 2) = f (2 - x)) →
  (∀ x, f x = f (4 - x)) →
  (∀ x y, f x = y ↔ f (4 - x) = 4 - y) ∧
  (∀ x, f x = f (4 - x)) ∧
  (∀ x, f (x - 2) = f (-x + 2)) :=
by
  intros h1 h2 h3
  exact ⟨symmetry_about_point f h1, symmetry_about_line f h2, symmetry_of_shifted_functions f h3⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_point_symmetry_about_line_symmetry_of_shifted_functions_all_statements_true_l917_91733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_symmetric_l917_91772

/-- Recursive definition of polynomial k_n -/
def k : ℕ → List ℝ → ℝ
  | 0, _ => 1
  | 1, [x] => x
  | 1, _ => 0  -- Handle the case for n=1 with empty or incorrect list
  | n, x::xs => 
      if n = xs.length + 1 then
        x * k (n-1) xs + (x^2 + xs.head!^2) * k (n-2) xs.tail
      else 0
  | _, [] => 0  -- Handle the case for empty list

/-- Theorem: k_n is symmetric for all non-negative n -/
theorem k_symmetric (n : ℕ) (xs : List ℝ) : 
  k n xs = k n xs.reverse :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_symmetric_l917_91772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pollution_analysis_l917_91769

-- Define the pollutant data
noncomputable def pollutant_data : List (ℝ × ℝ) := [(1, Real.exp 1), (2, Real.exp 3), (3, Real.exp 4), (4, Real.exp 5)]

-- Define the relationship between t and y
noncomputable def pollution_model (t : ℝ) : ℝ := Real.exp (1.3 * t)

-- Define the growth rate function
noncomputable def growth_rate (t : ℝ) : ℝ := pollution_model t / t

-- Theorem statement
theorem pollution_analysis :
  -- Part 1: The relationship between t and y is y = e^(1.3t)
  (∀ (p : ℝ × ℝ), p ∈ pollutant_data → p.2 = pollution_model p.1) ∧
  -- Part 2: The pollutant area grows at the slowest rate when t = 1/1.3 hours
  (∀ t > 0, growth_rate t ≥ growth_rate (1 / 1.3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pollution_analysis_l917_91769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jump_stride_difference_approx_l917_91706

/-- The number of strides Peter takes between consecutive poles -/
def peter_strides : ℕ := 66

/-- The number of jumps Miguel takes between consecutive poles -/
def miguel_jumps : ℕ := 18

/-- The number of poles along the trail -/
def num_poles : ℕ := 31

/-- The distance in feet from the first pole to the 31st pole -/
def total_distance : ℝ := 3960

/-- The length of Peter's stride in feet -/
noncomputable def peter_stride_length : ℝ := total_distance / ((num_poles - 1) * peter_strides)

/-- The length of Miguel's jump in feet -/
noncomputable def miguel_jump_length : ℝ := total_distance / ((num_poles - 1) * miguel_jumps)

/-- The difference between Miguel's jump length and Peter's stride length -/
noncomputable def jump_stride_difference : ℝ := miguel_jump_length - peter_stride_length

theorem jump_stride_difference_approx :
  abs (jump_stride_difference - 5.333) < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jump_stride_difference_approx_l917_91706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iterative_average_difference_l917_91781

def iterative_average (seq : List ℚ) : ℚ :=
  seq.foldl (λ acc x => (acc + x) / 2) seq.head!

def possible_sequences : List (List ℚ) :=
  [1, 2, 3, 4, 5, 6].permutations.filter (λ l => l.head! = 1 ∧ l.getLast! = 6)

theorem iterative_average_difference :
  let max_avg := (possible_sequences.map iterative_average).maximum?
  let min_avg := (possible_sequences.map iterative_average).minimum?
  ∀ ma mi, max_avg = some ma → min_avg = some mi → ma - mi = 71875 / 100000 := by
  sorry

#eval possible_sequences.map iterative_average

end NUMINAMATH_CALUDE_ERRORFEEDBACK_iterative_average_difference_l917_91781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_partition_l917_91744

/-- Given a triangle ABC with n points on each side (excluding vertices),
    and lines drawn from each point to the opposite vertex,
    where no three lines meet at a point other than A, B, and C,
    the number of regions the triangle is partitioned into is 3n^2 + 3n + 1. -/
theorem triangle_partition (n : ℕ) : ℕ := by
  -- The proof goes here
  sorry

-- Example evaluation (marked as noncomputable)
noncomputable def example_partition : ℕ := triangle_partition 5

#eval 3 * 5^2 + 3 * 5 + 1  -- This will evaluate to 91

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_partition_l917_91744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2A_eq_sin_2B_necessary_not_sufficient_for_a_eq_b_l917_91735

theorem sin_2A_eq_sin_2B_necessary_not_sufficient_for_a_eq_b 
  (A B C : Real) (a b c : Real) :
  (A + B + C = Real.pi) →  -- Triangle condition
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- Positive side lengths
  (a / Real.sin A = b / Real.sin B) →  -- Law of sines
  (a = b → Real.sin (2*A) = Real.sin (2*B)) ∧  -- Necessary condition
  ¬(Real.sin (2*A) = Real.sin (2*B) → a = b)  -- Not sufficient condition
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2A_eq_sin_2B_necessary_not_sufficient_for_a_eq_b_l917_91735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l917_91739

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 / 4 = 1

-- Define the real axis length
noncomputable def real_axis_length : ℝ := 2 * Real.sqrt 2

-- Theorem statement
theorem hyperbola_real_axis_length :
  ∀ x y : ℝ, hyperbola x y → real_axis_length = 2 * Real.sqrt 2 :=
by
  intros x y h
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_real_axis_length_l917_91739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_cities_l917_91701

/-- The distance between City A and City B in miles -/
noncomputable def distance : ℝ := 1100

/-- The average speed from A to B in miles per hour -/
noncomputable def speed_AB : ℝ := 50

/-- The average speed from B to A in miles per hour -/
noncomputable def speed_BA : ℝ := 60

/-- The rest stop time in hours -/
noncomputable def rest_time : ℝ := 1/3

/-- The average speed for the total trip in miles per hour -/
noncomputable def avg_speed : ℝ := 55

theorem distance_between_cities :
  distance = speed_AB * speed_BA * avg_speed * rest_time /
    (speed_AB * speed_BA - avg_speed * (speed_AB + speed_BA) / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_cities_l917_91701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l917_91703

noncomputable section

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f a b x = -f a b (-x)) →  -- f is odd
  f a b (1/2) = 2/5 →                                      -- f(1/2) = 2/5
  (∃ c : ℝ, c = a ∧ b = 0 ∧ 
    (∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 → f c 0 x = x / (1 + x^2)) ∧
    (∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → f c 0 x < f c 0 y) ∧
    {t : ℝ | f c 0 (t-1) + f c 0 t < 0} = Set.Ioo 0 (1/2)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l917_91703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_pi_sixth_l917_91704

theorem sec_pi_sixth : 1 / Real.cos (π / 6) = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_pi_sixth_l917_91704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_y_iff_q_equals_seven_l917_91740

theorem x_equals_y_iff_q_equals_seven (q : ℚ) :
  (55 + 2 * q = 4 * q + 41) ↔ q = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_y_iff_q_equals_seven_l917_91740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_positive_derivative_positive_function_from_positive_derivative_l917_91759

-- Define the differential equation
def differential_equation (f : ℝ → ℝ) : Prop :=
  ∀ x, (deriv (deriv f)) x - 2 * (deriv f) x + f x = 2 * Real.exp x

-- Statement 1
theorem not_always_positive_derivative
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x > 0)
  (h2 : differential_equation f) :
  ¬ (∀ x, (deriv f) x > 0) :=
sorry

-- Statement 2
theorem positive_function_from_positive_derivative
  (g : ℝ → ℝ)
  (h : ∀ x, (deriv g) x > 0) :
  ∀ x, g x > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_positive_derivative_positive_function_from_positive_derivative_l917_91759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_2_seconds_l917_91730

-- Define the motion equation
noncomputable def s (t : ℝ) : ℝ := 1/t + 2*t

-- Define the instantaneous velocity function
noncomputable def v (t : ℝ) : ℝ := deriv s t

-- Theorem statement
theorem instantaneous_velocity_at_2_seconds :
  v 2 = 7/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_instantaneous_velocity_at_2_seconds_l917_91730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l917_91725

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b = 1) :
  (1/2 < (2 : ℝ)^(a - Real.sqrt b) ∧ (2 : ℝ)^(a - Real.sqrt b) < 2) ∧ 
  (a + Real.sqrt b ≤ Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l917_91725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_monotone_interval_l917_91761

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x^2 + 2*x else -x^2 + 2*x

-- State the theorem
theorem odd_function_monotone_interval (m : ℝ) :
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x, x ≤ 0 → f x = x^2 + 2*x) →  -- definition for x ≤ 0
  (∀ x y, x ∈ Set.Icc (-1) (m-1) → y ∈ Set.Icc (-1) (m-1) → x < y → f x < f y) →  -- f is monotonically increasing on [-1, m-1]
  0 < m ∧ m ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_monotone_interval_l917_91761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l917_91743

theorem root_exists_in_interval : ∃! x : ℝ, x ∈ Set.Ioo 1 2 ∧ x * Real.log x - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l917_91743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l917_91741

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := (1 + x) / (1 - x)

-- Define the point of tangency
def point : ℝ × ℝ := (2, -3)

-- Define the slope of the tangent line
def m : ℝ := 2

-- Theorem statement
theorem tangent_line_equation :
  let (x₀, y₀) := point
  let tangent_eq (x y : ℝ) := 2 * x - y - 7 = 0
  tangent_eq x₀ y₀ ∧ 
  ∀ x y, tangent_eq x y → (y - y₀) = m * (x - x₀) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l917_91741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wood_measurement_equation_l917_91776

/-- Represents the length of a piece of wood in feet. -/
def x : ℝ := sorry

/-- The length of the rope in feet. -/
def rope_length (x : ℝ) : ℝ := x + 4.5

/-- The condition that when the rope is folded in half and used to measure the wood,
    there is 1 foot of rope left. -/
def folded_rope_condition (x : ℝ) : Prop :=
  (1/2) * rope_length x = x - 1

/-- Theorem stating that if a rope used to measure a piece of wood of length x feet
    leaves 4.5 feet when measuring directly and 1 foot when folded in half,
    then the equation ½(x + 4.5) = x - 1 holds true. -/
theorem wood_measurement_equation (x : ℝ) :
  folded_rope_condition x ↔ (1/2) * (x + 4.5) = x - 1 := by
  sorry

#check wood_measurement_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wood_measurement_equation_l917_91776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vinny_fifth_month_loss_l917_91794

/-- Calculates the weight loss for a given month based on the previous month's loss -/
noncomputable def monthlyWeightLoss (previousLoss : ℝ) : ℝ := previousLoss / 2

/-- Calculates the total weight loss for the first four months -/
noncomputable def firstFourMonthsLoss (initialLoss : ℝ) : ℝ :=
  initialLoss + monthlyWeightLoss initialLoss + 
  monthlyWeightLoss (monthlyWeightLoss initialLoss) + 
  monthlyWeightLoss (monthlyWeightLoss (monthlyWeightLoss initialLoss))

/-- Theorem stating that given the conditions of Vinny's diet, 
    the weight loss in the fifth month is 12 pounds -/
theorem vinny_fifth_month_loss 
  (initialWeight : ℝ) 
  (finalWeight : ℝ) 
  (firstMonthLoss : ℝ) 
  (hw : initialWeight = 300) 
  (hf : finalWeight = 250.5) 
  (hl : firstMonthLoss = 20) : 
  initialWeight - finalWeight - firstFourMonthsLoss firstMonthLoss = 12 :=
by
  sorry

#check vinny_fifth_month_loss

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vinny_fifth_month_loss_l917_91794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l917_91705

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_equality (x : ℝ) (h : x > 1) :
  (∀ y > 0, f (2/y + 1) = lg y) → f x = lg (2/(x-1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equality_l917_91705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_g_two_l917_91745

def T : Set ℝ := {x : ℝ | x ≠ 0}

def FunctionalEquation (g : T → T) : Prop :=
  ∀ x y : T, (x : ℝ) + (y : ℝ) ≠ 0 →
    (g x : ℝ) + (g y : ℝ) = g ⟨((x : ℝ) * (y : ℝ) * (g ⟨(x : ℝ) + (y : ℝ), sorry⟩ : ℝ)) / 2, sorry⟩

theorem unique_g_two (g : T → T) (h : FunctionalEquation g) :
  ∃! v : ℝ, v ∈ T ∧ g ⟨2, sorry⟩ = ⟨v, sorry⟩ ∧ v = 1/2 := by
  sorry

#check unique_g_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_g_two_l917_91745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_l917_91795

-- Define the original parabola
def original_parabola (x y : ℝ) : Prop := y = (x - 3)^2 - 4

-- Define the translation
def translate (x y : ℝ) : ℝ × ℝ := (x + 1, y + 2)

-- Define the resulting parabola
def resulting_parabola (x y : ℝ) : Prop := y = (x - 4)^2 - 2

-- Theorem stating that the translation of the original parabola results in the new parabola
theorem parabola_translation :
  ∀ x y : ℝ, original_parabola x y → 
  let (x', y') := translate x y
  resulting_parabola x' y' :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_l917_91795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l917_91721

noncomputable def f (φ : Real) (x : Real) : Real := Real.sqrt 5 * Real.sin (2 * x + φ)

theorem function_properties (φ : Real) 
  (h1 : 0 < φ ∧ φ < Real.pi)
  (h2 : ∀ x, f φ (Real.pi/3 - x) = f φ (Real.pi/3 + x)) :
  (φ = 5*Real.pi/6) ∧
  (∀ x ∈ Set.Icc (-Real.pi/12) (Real.pi/2), 
    f φ x ≤ Real.sqrt 15 / 2 ∧
    (f φ x = Real.sqrt 15 / 2 ↔ x = -Real.pi/12)) ∧
  (∀ x ∈ Set.Icc (-Real.pi/12) (Real.pi/2),
    f φ x ≥ -Real.sqrt 5 ∧
    (f φ x = -Real.sqrt 5 ↔ x = Real.pi/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l917_91721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_coordinates_above_line_l917_91787

noncomputable def points : List (ℝ × ℝ) := [(4, 14), (8, 25), (15, 45), (21, 55), (24, 65)]

noncomputable def isAboveLine (p : ℝ × ℝ) : Bool :=
  p.2 > 3 * p.1 + 5

noncomputable def sumXCoordinatesAboveLine (pts : List (ℝ × ℝ)) : ℝ :=
  (pts.filter isAboveLine).map Prod.fst |>.sum

theorem sum_x_coordinates_above_line :
  sumXCoordinatesAboveLine points = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_coordinates_above_line_l917_91787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_l917_91734

/-- The function g(x) -/
noncomputable def g (x : ℝ) : ℝ := (3*x^3 - 5*x^2 + x - 1) / (x^2 - 3*x + 4)

/-- The horizontal asymptote of g(x) -/
def horizontal_asymptote : ℝ := 0

/-- Theorem: g(x) crosses its horizontal asymptote at x = 1 and x = -1/3 -/
theorem g_crosses_asymptote :
  {x : ℝ | g x = horizontal_asymptote} = {1, -1/3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_l917_91734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_proof_l917_91717

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

-- Define the function g
def g (f : ℝ → ℝ) (m : ℝ) (x : ℝ) : ℝ := f x - m * x

-- State the theorem
theorem function_and_range_proof 
  (a b : ℝ) 
  (h_a : a ≠ 0)
  (h_f : ∀ x, f a b (x + 1) - f a b x = 2 * x - 1)
  (h_g : ∀ m, ∀ x₁ x₂, x₁ ∈ Set.Icc 1 2 → x₂ ∈ Set.Icc 1 2 → 
    |g (f a b) m x₁ - g (f a b) m x₂| ≤ 2) :
  (∀ x, f a b x = x^2 - 2*x + 3) ∧ 
  (∃ m_min m_max, m_min = -1 ∧ m_max = 3 ∧ 
    ∀ m, (∀ x₁ x₂, x₁ ∈ Set.Icc 1 2 → x₂ ∈ Set.Icc 1 2 → 
      |g (f a b) m x₁ - g (f a b) m x₂| ≤ 2) → 
      m_min ≤ m ∧ m ≤ m_max) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_range_proof_l917_91717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_roots_iff_b_lt_neg_three_l917_91774

-- Define the function f(x) = |xe^(x+1)|
noncomputable def f (x : ℝ) : ℝ := abs (x * Real.exp (x + 1))

-- Define the equation f²(x) + bf(x) + 2 = 0
def has_four_distinct_roots (b : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    (f x₁)^2 + b * f x₁ + 2 = 0 ∧
    (f x₂)^2 + b * f x₂ + 2 = 0 ∧
    (f x₃)^2 + b * f x₃ + 2 = 0 ∧
    (f x₄)^2 + b * f x₄ + 2 = 0

-- Theorem statement
theorem four_roots_iff_b_lt_neg_three :
  ∀ b : ℝ, has_four_distinct_roots b ↔ b < -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_roots_iff_b_lt_neg_three_l917_91774
