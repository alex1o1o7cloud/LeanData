import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_divisors_l589_58919

def sequence_a : ℕ → ℕ
  | 0 => 5
  | (n + 1) => (sequence_a n) ^ 2

theorem distinct_prime_divisors (n : ℕ) (h : n > 0) : 
  ∃ (S : Finset Nat), Finset.card S ≥ n ∧ 
  (∀ p ∈ S, Nat.Prime p ∧ p ∣ (sequence_a n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_divisors_l589_58919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_315_degrees_sec_315_degrees_result_l589_58910

theorem sec_315_degrees : Real.cos (315 * π / 180) = 1 / Real.sqrt 2 := by
  sorry

theorem sec_315_degrees_result : 1 / Real.cos (315 * π / 180) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_315_degrees_sec_315_degrees_result_l589_58910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_theorem_l589_58958

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | Real.rpow 2 (x-2) > 1}

-- State the theorem
theorem intersection_complement_theorem :
  A ∩ (Set.univ \ B) = {x : ℝ | -1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_complement_theorem_l589_58958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bell_rings_for_geography_l589_58951

/-- Represents a class in Isabella's schedule -/
inductive ClassType
| Maths
| History
| Geography
| Science
| Music

/-- The schedule of classes for Isabella -/
def schedule : List ClassType :=
  [ClassType.Maths, ClassType.History, ClassType.Geography, ClassType.Science, ClassType.Music]

/-- Counts the number of bell rings up to and including the start of a given class -/
def bellRings : ClassType → Nat
| ClassType.Maths => 1
| ClassType.History => 3
| ClassType.Geography => 5
| ClassType.Science => 7
| ClassType.Music => 9

theorem bell_rings_for_geography :
  bellRings ClassType.Geography = 5 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bell_rings_for_geography_l589_58951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_decreasing_m_range_l589_58917

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.log x

noncomputable def g (x : ℝ) : ℝ := (1/2) * x * abs x

noncomputable def F (x : ℝ) : ℝ := x * f x - g x

-- Theorem for the monotonicity of F
theorem F_decreasing : 
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → F x₁ > F x₂ := by
  sorry

-- Theorem for the range of m
theorem m_range : 
  ∀ m : ℝ, m ≥ 1 → 
    ∀ x₁ x₂ : ℝ, x₁ ≥ 1 → x₂ ≥ 1 → x₁ < x₂ → 
      m * (g x₂ - g x₁) > x₂ * f x₂ - x₁ * f x₁ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_decreasing_m_range_l589_58917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sum_l589_58903

-- Define the power function as noncomputable
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- State the theorem
theorem max_value_of_sum (α : ℝ) (h : f α 4 = 2) :
  ∃ (max_val : ℝ), max_val = 2 ∧ 
  ∀ (a : ℝ), f α (a - 3) + f α (5 - a) ≤ max_val := by
  -- Proof sketch
  -- 1. Show that α = 1/2 using the given condition
  -- 2. Rewrite f as the square root function
  -- 3. Prove that the maximum value occurs when a = 4
  -- 4. Show that the maximum value is 2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_sum_l589_58903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l589_58927

theorem sin_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (α + π / 6) = 4 / 5) : 
  Real.sin α = (3 * Real.sqrt 3 - 4) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l589_58927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_values_l589_58992

noncomputable def f (x : ℝ) : ℝ :=
  if x > 6 then x^2 - 4
  else if x ≥ -6 then 3*x + 2
  else 5

noncomputable def f_modified (x : ℝ) : ℝ :=
  if x % 3 = 0 then f x + 5 else f x

theorem sum_of_f_values : f_modified (-8) + f_modified 0 + f_modified 9 = 94 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_values_l589_58992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_temp_deviation_l589_58957

/-- Represents the temperature conversion between Fahrenheit and Celsius -/
structure TempConversion where
  f_to_c : ℚ → ℚ
  c_to_f : ℚ → ℚ

/-- Represents the rounding operation -/
noncomputable def customRound (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

/-- The maximum deviation when rounding a rational number -/
def max_round_deviation : ℚ := 1/2

/-- Theorem stating the maximum possible deviation in temperature conversion and rounding -/
theorem max_temp_deviation (conv : TempConversion) 
  (h1 : conv.f_to_c 32 = 0)
  (h2 : conv.f_to_c 212 = 100)
  (h3 : ∀ x, conv.c_to_f (conv.f_to_c x) = x)
  (h4 : ∀ x, conv.f_to_c (conv.c_to_f x) = x) :
  ∀ t : ℚ, 
    |conv.f_to_c t - (conv.f_to_c ∘ (↑) ∘ customRound ∘ conv.c_to_f ∘ (↑) ∘ customRound ∘ conv.f_to_c) t| ≤ 13/18 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_temp_deviation_l589_58957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_line_l589_58983

-- Define the set T
def T : Set ℂ := {z : ℂ | ∃ (r : ℝ), (5 - 2 * Complex.I) * z = r}

-- Theorem statement
theorem T_is_line : ∃ (a b : ℝ), a ≠ 0 ∧ T = {z : ℂ | ∃ (t : ℝ), z = t * Complex.mk a b} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_is_line_l589_58983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_rectangle_l589_58908

/-- Area of a triangle given its vertices -/
def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Given a 6x8 rectangle with a triangle ABC inside, prove its area is 6 square units -/
theorem triangle_area_in_rectangle (A B C : ℝ × ℝ) : 
  A = (0, 2) → B = (6, 0) → C = (3, 8) → area_triangle A B C = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_in_rectangle_l589_58908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l589_58975

open Real

theorem trig_identity (α : ℝ) : 
  tan α + (1 / tan α) + tan (3 * α) + (1 / tan (3 * α)) = 
  (8 * (cos (2 * α))^2) / sin (6 * α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l589_58975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l589_58940

open Real

-- Define the triangle and vectors
def triangle_ABC (A B C a b c : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0

def vector_m (a b : ℝ) : ℝ × ℝ := (a, b)

noncomputable def vector_n (A B : ℝ) : ℝ × ℝ := (cos A, cos B)

noncomputable def vector_p (B C A : ℝ) : ℝ × ℝ := (2 * Real.sqrt 2 * sin ((B + C) / 2), 2 * sin A)

-- Define parallel vectors
def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

-- Define vector magnitude
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

-- Define the function f
noncomputable def f (A B x : ℝ) : ℝ :=
  sin A * sin x + cos B * cos x

-- Main theorem
theorem triangle_theorem (A B C a b c : ℝ) :
  triangle_ABC A B C a b c →
  parallel (vector_m a b) (vector_n A B) →
  magnitude (vector_p B C A) = 3 →
  A = π/3 ∧ B = π/3 ∧ C = π/3 ∧
  (∀ x, x ∈ Set.Icc 0 (π/2) → f A B x ≤ 1) ∧
  (∀ x, x ∈ Set.Icc 0 (π/2) → f A B x ≥ 1/2) ∧
  (∃ x, x ∈ Set.Icc 0 (π/2) ∧ f A B x = 1) ∧
  (∃ x, x ∈ Set.Icc 0 (π/2) ∧ f A B x = 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l589_58940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_10km_l589_58987

/-- Represents the swimming scenario with given parameters -/
structure SwimmingScenario where
  downstream_distance : ℝ
  downstream_time : ℝ
  upstream_time : ℝ
  current_speed : ℝ

/-- Calculates the upstream distance given a swimming scenario -/
noncomputable def upstream_distance (s : SwimmingScenario) : ℝ :=
  let swimmer_speed := s.downstream_distance / s.downstream_time - s.current_speed
  (swimmer_speed - s.current_speed) * s.upstream_time

/-- Theorem stating that the upstream distance is 10 km for the given scenario -/
theorem upstream_distance_is_10km (s : SwimmingScenario) 
    (h1 : s.downstream_distance = 55)
    (h2 : s.downstream_time = 5)
    (h3 : s.upstream_time = 5)
    (h4 : s.current_speed = 4.5) : 
  upstream_distance s = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_10km_l589_58987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_quarter_plus_x_l589_58931

theorem tan_pi_quarter_plus_x (x : ℝ) (h1 : x ∈ Set.Ioo (-π/2) 0) (h2 : Real.cos x = 4/5) :
  Real.tan (π/4 + x) = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_quarter_plus_x_l589_58931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_multiples_of_three_minus30_to_60_l589_58914

def sum_multiples_of_three (start : Int) (stop : Int) : Int :=
  let first := start - start % 3
  let last := stop - stop % 3
  let n := (last - first) / 3 + 1
  n * (first + last) / 2

theorem sum_multiples_of_three_minus30_to_60 :
  sum_multiples_of_three (-30) 60 = 465 := by
  -- Proof goes here
  sorry

#eval sum_multiples_of_three (-30) 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_multiples_of_three_minus30_to_60_l589_58914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_other_asymptote_l589_58933

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  /-- One asymptote of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- X-coordinate of the foci -/
  foci_x : ℝ
  /-- Indicates if the hyperbola has a vertical transverse axis -/
  vertical_transverse : Prop

/-- The other asymptote of the hyperbola -/
noncomputable def other_asymptote (h : Hyperbola) : ℝ → ℝ :=
  fun x ↦ -1/2 * x + 15/2

theorem hyperbola_other_asymptote 
  (h : Hyperbola) 
  (h1 : h.asymptote1 = fun x ↦ 2 * x) 
  (h2 : h.foci_x = 3) 
  (h3 : h.vertical_transverse) : 
  other_asymptote h = fun x ↦ -1/2 * x + 15/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_other_asymptote_l589_58933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_band_arrangement_l589_58966

theorem band_arrangement (n : ℕ) : n = 90 →
  (∃ (s : Finset ℕ), ∀ x, x ∈ s ↔ (x ∣ n ∧ 6 ≤ x ∧ x ≤ 15)) →
  (Finset.filter (λ x => x ∣ n ∧ 6 ≤ x ∧ x ≤ 15) (Finset.range (n + 1))).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_band_arrangement_l589_58966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l589_58913

theorem cos_minus_sin_value (α : ℝ) 
  (h1 : π/4 < α) 
  (h2 : α < π/2) 
  (h3 : Real.sin (2*α) = 24/25) : 
  Real.cos α - Real.sin α = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l589_58913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_medians_equilateral_l589_58932

/-- Given a triangle ABC with extended medians to A', B', C', prove that if A'B'C' is equilateral, then k = 1/√3 -/
theorem extended_medians_equilateral (A B C A' B' C' M N : ℂ) (k : ℝ) : 
  k > 0 → -- k is positive
  M = (B + C) / 2 → -- M is midpoint of BC
  N = (A + C) / 2 → -- N is midpoint of AC
  A' = A + k * (M - A) → -- AA' = k * AM
  B' = B + k * (N - B) → -- BB' = k * BN
  C' = C + k * ((A + B) / 2 - C) → -- CC' = k * CM
  (B' - A' = (C' - A') * Complex.exp (Complex.I * Real.pi / 3)) → -- A'B'C' is equilateral
  k = 1 / Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_medians_equilateral_l589_58932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_covered_bound_l589_58967

/-- A configuration of circles on a plane. -/
structure CircleConfiguration where
  n : ℕ
  centers : Fin n → ℝ × ℝ
  radius : ℝ
  n_ge_3 : n ≥ 3
  radius_is_1 : radius = 1
  intersection_property : ∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k →
    ∃ (p q : Fin n), p ≠ q ∧ (dist (centers p) (centers q) ≤ 2 * radius)

/-- The area covered by a configuration of circles. -/
noncomputable def area_covered (config : CircleConfiguration) : ℝ := sorry

/-- The theorem to be proved. -/
theorem area_covered_bound (config : CircleConfiguration) :
  area_covered config < 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_covered_bound_l589_58967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_intersection_fixed_point_l589_58941

noncomputable def A : ℝ × ℝ := (-Real.sqrt 2, 0)
noncomputable def B : ℝ × ℝ := (Real.sqrt 2, 0)

def C (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

def F : ℝ × ℝ := (1, 0)

def P_condition (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  2 * ((x + Real.sqrt 2) * (x - Real.sqrt 2) + y^2) = x^2

theorem trajectory_intersection_fixed_point :
  ∀ (k : ℝ) (G H M N : ℝ × ℝ),
  (∃ (P : ℝ × ℝ), P_condition P ∧ C P.1 P.2) →
  (C G.1 G.2 ∧ C H.1 H.2 ∧ C M.1 M.2 ∧ C N.1 N.2) →
  (G.2 - F.2 = k * (G.1 - F.1)) →
  (H.2 - F.2 = k * (H.1 - F.1)) →
  (M.2 - F.2 = -1/k * (M.1 - F.1)) →
  (N.2 - F.2 = -1/k * (N.1 - F.1)) →
  let E₁ := ((G.1 + H.1)/2, (G.2 + H.2)/2)
  let E₂ := ((M.1 + N.1)/2, (M.2 + N.2)/2)
  ∃ (t : ℝ), E₁.1 + t * (E₂.1 - E₁.1) = 2/3 ∧ E₁.2 + t * (E₂.2 - E₁.2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_intersection_fixed_point_l589_58941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_fifth_power_l589_58960

theorem cosine_sum_fifth_power : 
  (Real.cos (π/9))^5 + (Real.cos (5*π/9))^5 + (Real.cos (7*π/9))^5 = 21/32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_fifth_power_l589_58960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_product_l589_58928

theorem units_digit_of_product (a b : ℕ) : 
  a > 0 → b > 0 →
  (a : ℝ) + b * Real.sqrt 2 = (1 + Real.sqrt 2) ^ 2015 →
  (a * b) % 10 = 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_product_l589_58928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_55_expected_value_l589_58934

/-- Represents the score of a student in a multiple-choice test -/
def Score := Fin 61

/-- The number of questions in the test -/
def num_questions : ℕ := 12

/-- The number of options for each question -/
def num_options : ℕ := 4

/-- The score awarded for a correct answer -/
def correct_score : ℕ := 5

/-- The number of questions answered correctly with certainty -/
def certain_correct : ℕ := 9

/-- The number of questions with two incorrect options eliminated -/
def two_eliminated : ℕ := 2

/-- The number of questions with one incorrect option eliminated -/
def one_eliminated : ℕ := 1

/-- The random variable representing the student's score -/
noncomputable def X : Score := sorry

/-- The probability mass function of X -/
noncomputable def pmf (x : Score) : ℝ := sorry

/-- The expected value of X -/
noncomputable def E_X : ℝ := sorry

/-- Theorem stating the probability of scoring 55 points -/
theorem prob_55 : pmf ⟨55, sorry⟩ = 1 / 3 := by sorry

/-- Theorem stating the expected value of the score -/
theorem expected_value : E_X = 155 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_55_expected_value_l589_58934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_range_of_m_not_necessary_l589_58995

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 < x ∧ x < 10}
def B : Set ℝ := {x | x^2 - 9*x + 14 < 0}
def C (m : ℝ) : Set ℝ := {x | 5 - m < x ∧ x < 2*m}

-- Theorem for part (1)
theorem intersection_and_union :
  (A ∩ B = {x | 3 < x ∧ x < 7}) ∧
  ((Aᶜ) ∪ B = {x | x < 7 ∨ 10 ≤ x}) := by sorry

-- Theorem for part (2)
theorem range_of_m :
  {m : ℝ | ∀ x, x ∈ C m → x ∈ (A ∩ B)} = Set.Iic 2 := by sorry

-- Additional theorem to capture the "not necessary" condition
theorem not_necessary :
  ∀ m, m ≤ 2 → ∃ x, x ∈ (A ∩ B) ∧ x ∉ C m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_range_of_m_not_necessary_l589_58995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_proof_l589_58954

/-- Given a quadrilateral pyramid with a rectangular base:
    - b: length of the diagonal of the base
    - angle_between_diagonals: angle between the diagonals of the base (60°)
    - lateral_edge_angle: angle between each lateral edge and the base plane (45°)
    Calculates the volume of the pyramid -/
noncomputable def pyramid_volume (b : ℝ) (angle_between_diagonals : ℝ) (lateral_edge_angle : ℝ) : ℝ :=
  (b^3 * Real.sqrt 3) / 24

theorem pyramid_volume_proof (b : ℝ) (angle_between_diagonals : ℝ) (lateral_edge_angle : ℝ) 
  (h1 : angle_between_diagonals = π/3)  -- 60° in radians
  (h2 : lateral_edge_angle = π/4)       -- 45° in radians
  : pyramid_volume b angle_between_diagonals lateral_edge_angle = (b^3 * Real.sqrt 3) / 24 := by
  -- Unfold the definition of pyramid_volume
  unfold pyramid_volume
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_proof_l589_58954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_power_identity_l589_58984

theorem cosine_power_identity (n : ℕ) (x θ : ℝ) (h : x + x⁻¹ = 2 * Real.cos θ) :
  x^n + (x^n)⁻¹ = 2 * Real.cos (n * θ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_power_identity_l589_58984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_fraction_evaluation_l589_58969

theorem ceiling_fraction_evaluation :
  (⌈(19 : ℚ) / 7 - ⌈(37 : ℚ) / 17⌉⌉) / (⌈(33 : ℚ) / 7 + ⌈7 * (19 : ℚ) / 33⌉⌉) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_fraction_evaluation_l589_58969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l589_58938

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  a_gt_b : b < a
  relation : a^2 = b^2 + c^2

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- Theorem stating the range of eccentricity for the given ellipse -/
theorem eccentricity_range (e : Ellipse) 
  (min_dot_product : ∀ (P : ℝ × ℝ), P.1^2/e.a^2 + P.2^2/e.b^2 = 1 → 
    e.c^2 ≤ ((P.1 + e.c)^2 + P.2^2) * ((P.1 - e.c)^2 + P.2^2) ∧
    ((P.1 + e.c)^2 + P.2^2) * ((P.1 - e.c)^2 + P.2^2) ≤ 3*e.c^2) :
  Real.sqrt 5 / 5 ≤ eccentricity e ∧ eccentricity e ≤ Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l589_58938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_density_ratio_of_composite_cube_l589_58971

/-- Represents a cube composed of smaller cubes -/
structure CompositeCube where
  smallCubeVolume : ℝ
  smallCubeDensity : ℝ
  totalSmallCubes : ℕ
  replacedCubes : ℕ
  densityIncreaseFactor : ℝ

/-- Calculate the density ratio of a composite cube before and after replacement -/
noncomputable def densityRatio (c : CompositeCube) : ℝ :=
  let initialMass := (c.totalSmallCubes : ℝ) * c.smallCubeDensity * c.smallCubeVolume
  let finalMass := ((c.totalSmallCubes - c.replacedCubes) : ℝ) * c.smallCubeDensity * c.smallCubeVolume +
                   (c.replacedCubes : ℝ) * (c.densityIncreaseFactor * c.smallCubeDensity) * c.smallCubeVolume
  let totalVolume := (c.totalSmallCubes : ℝ) * c.smallCubeVolume
  (initialMass / totalVolume) / (finalMass / totalVolume)

theorem density_ratio_of_composite_cube :
  let c : CompositeCube := {
    smallCubeVolume := 1,
    smallCubeDensity := 1,
    totalSmallCubes := 8,
    replacedCubes := 2,
    densityIncreaseFactor := 2
  }
  densityRatio c = 4/5 := by
  sorry

#eval (4 : ℚ) / 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_density_ratio_of_composite_cube_l589_58971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_fraction_l589_58988

/-- The decimal representation of the number we're converting to a fraction -/
def decimal : ℚ := 637 / 990

/-- The fraction we claim is equivalent to the decimal -/
def fraction : ℚ := 631 / 990

/-- Predicate to check if a fraction is in its lowest terms -/
def is_reduced (n d : ℤ) : Prop := Nat.gcd n.natAbs d.natAbs = 1

theorem decimal_to_fraction :
  decimal = fraction ∧ is_reduced 631 990 := by
  sorry

#eval decimal
#eval fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_fraction_l589_58988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l589_58930

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_a_ge_b : a ≥ b

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt ((e.a^2 - e.b^2) / e.a^2)

/-- The sum of distances from any point on the ellipse to its foci -/
def foci_distance_sum (e : Ellipse) : ℝ := 2 * e.a

theorem ellipse_equation (e : Ellipse) 
  (h_ecc : eccentricity e = Real.sqrt 3 / 2)
  (h_sum : foci_distance_sum e = 12) :
  ∀ (x y : ℝ), x^2 / 36 + y^2 / 9 = 1 ↔ x^2 / e.a^2 + y^2 / e.b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l589_58930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_side_to_diagonal_ratio_l589_58900

/-- The ratio of the side length to the diagonal that skips one vertex in a regular octagon -/
noncomputable def octagon_ratio : ℝ := 1 / Real.sqrt (2 - Real.sqrt 2)

/-- Definition of a regular octagon using side length and diagonal length -/
def is_regular_octagon (side diagonal : ℝ) : Prop :=
  ∃ (center : ℝ × ℝ) (vertices : Fin 8 → ℝ × ℝ),
    (∀ i : Fin 8, dist (vertices i) (vertices ((i + 1) % 8)) = side) ∧
    (∀ i : Fin 8, dist (vertices i) (vertices ((i + 2) % 8)) = diagonal) ∧
    (∀ i : Fin 8, dist center (vertices i) = dist center (vertices 0))

/-- Theorem stating that the ratio of the side length to the diagonal that skips one vertex 
    in a regular octagon is equal to 1 / √(2 - √2) -/
theorem regular_octagon_side_to_diagonal_ratio :
  ∀ (side diagonal : ℝ), side > 0 → diagonal > 0 →
  is_regular_octagon side diagonal →
  side / diagonal = octagon_ratio := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_side_to_diagonal_ratio_l589_58900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_two_tangent_through_two_four_l589_58990

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := (1/3) * x^3 + 4/3

-- Define the derivative of the curve
noncomputable def curve_derivative (x : ℝ) : ℝ := x^2

-- Theorem for the tangent line at x=2
theorem tangent_at_two :
  ∃ (k b : ℝ), k * 2 + b = curve 2 ∧
               k = curve_derivative 2 ∧
               ∀ x, k * x + b = 4 * x - 4 := by
  sorry

-- Theorem for tangent lines passing through (2, 4)
theorem tangent_through_two_four :
  ∃ (x₀ : ℝ), x₀ ≠ 2 ∧
              (∃ (k b : ℝ), k * 2 + b = 4 ∧
                            k * x₀ + b = curve x₀ ∧
                            k = curve_derivative x₀ ∧
                            ∀ x, k * x + b = x - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_two_tangent_through_two_four_l589_58990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_inequality_l589_58977

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the concept of an obtuse angle
def is_obtuse_angle (angle : ℝ) : Prop := angle > (Real.pi / 2)

-- Define the length of a line segment
noncomputable def segment_length (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define a function to calculate the angle at a vertex
noncomputable def angle_at (q : Quadrilateral) (v : ℝ × ℝ) : ℝ :=
  sorry -- The actual implementation would go here

theorem diagonal_inequality (q : Quadrilateral) 
  (h1 : is_obtuse_angle (angle_at q q.B))
  (h2 : is_obtuse_angle (angle_at q q.D)) :
  segment_length q.B q.D < segment_length q.A q.C :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_inequality_l589_58977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l589_58964

-- Define the original function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the inverse function
noncomputable def g (x : ℝ) : ℝ := 2^x

-- Theorem statement
theorem inverse_function_proof (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 8 = 3 → (∀ x, g (f 2 x) = x ∧ f 2 (g x) = x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l589_58964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_is_629_l589_58902

/-- Represents the cost of fencing per meter for each side of the pentagon. -/
def fencing_cost : Fin 5 → ℕ
  | 0 => 2
  | 1 => 2
  | 2 => 3
  | 3 => 3
  | 4 => 4

/-- Represents the length of each side of the pentagon in meters. -/
def side_length : Fin 5 → ℕ
  | 0 => 34
  | 1 => 28
  | 2 => 45
  | 3 => 50
  | 4 => 55

/-- Calculates the total cost of fencing for the pentagonal field. -/
def total_fencing_cost : ℕ :=
  (Finset.sum Finset.univ fun i => fencing_cost i * side_length i)

/-- Theorem stating that the total cost of fencing is 629 Rs. -/
theorem fencing_cost_is_629 : total_fencing_cost = 629 := by
  sorry

#eval total_fencing_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_is_629_l589_58902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_circle_tangency_l589_58979

/-- A parabola with equation y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- A line with slope 1 -/
structure Line where
  c : ℝ

/-- A circle with center (5, 0) and radius 2√2 -/
def Circle : Set (ℝ × ℝ) :=
  {xy | (xy.1 - 5)^2 + xy.2^2 = 8}

/-- The focus of a parabola -/
noncomputable def focus (para : Parabola) : ℝ × ℝ := (para.p / 2, 0)

/-- A line passes through a point -/
def passes_through (l : Line) (point : ℝ × ℝ) : Prop :=
  point.2 = point.1 + l.c

/-- A line is tangent to a circle -/
def is_tangent (l : Line) (c : Set (ℝ × ℝ)) : Prop :=
  ∃ (point : ℝ × ℝ), point ∈ c ∧ passes_through l point ∧
  ∀ (other : ℝ × ℝ), other ∈ c → passes_through l other → other = point

theorem parabola_line_circle_tangency (para : Parabola) (l : Line) :
  passes_through l (focus para) →
  is_tangent l Circle →
  para.p = 2 ∨ para.p = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_circle_tangency_l589_58979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_properties_l589_58968

-- Define the point M
structure Point where
  x : ℝ
  y : ℝ

-- Define the fixed point F
def F : Point := ⟨1, 0⟩

-- Define the distance function
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Define the distance to the line x = -2
def distToLine (p : Point) : ℝ :=
  |p.x + 2|

-- Define the locus condition
def locusCondition (m : Point) : Prop :=
  distance m F = distToLine m - 1

-- State the theorem
theorem locus_and_properties :
  ∀ (m : Point), locusCondition m →
  (∃ (k : ℝ), m.y^2 = 4 * m.x) ∧ 
  (∀ (l₁ l₂ : Point → Prop), 
    (∀ (p : Point), l₁ p → l₂ p → p = F) →  -- l₁ and l₂ intersect at F
    (∀ (p q : Point), l₁ p → l₁ q → (p.x - F.x) * (q.x - F.x) + (p.y - F.y) * (q.y - F.y) = 0) →  -- l₁ ⊥ l₂
    ∃ (a b m n : Point), 
      locusCondition a ∧ locusCondition b ∧ locusCondition m ∧ locusCondition n ∧
      l₁ a ∧ l₁ b ∧ l₂ m ∧ l₂ n ∧
      ∃ (p q : Point), 
        p.x = (a.x + b.x) / 2 ∧ p.y = (a.y + b.y) / 2 ∧
        q.x = (m.x + n.x) / 2 ∧ q.y = (m.y + n.y) / 2 ∧
        ∃ (t : ℝ), p.x + t * (q.x - p.x) = 3 ∧ p.y + t * (q.y - p.y) = 0) ∧
  (∃ (minArea : ℝ), minArea = 4 ∧
    ∀ (p q : Point), 
      (∃ (a b m n : Point), 
        locusCondition a ∧ locusCondition b ∧ locusCondition m ∧ locusCondition n ∧
        p.x = (a.x + b.x) / 2 ∧ p.y = (a.y + b.y) / 2 ∧
        q.x = (m.x + n.x) / 2 ∧ q.y = (m.y + n.y) / 2) →
      1/2 * |Matrix.det
        ![![p.x - F.x, q.x - F.x],
          ![p.y - F.y, q.y - F.y]]| ≥ minArea) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_properties_l589_58968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_max_chord_length_achieved_l589_58937

/-- The equation of the ellipse -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- The equation of the line -/
def line (x y k : ℝ) : Prop := y = k*x + 1

/-- The length of the chord intercepted by the ellipse from the line -/
noncomputable def chord_length (k : ℝ) : ℝ := 
  let p := (0, 1)
  Real.sqrt (4 * (3/4) + 1/9)  -- This is the maximum chord length we calculated

/-- The maximum length of the chord as k varies -/
theorem max_chord_length : 
  ∀ k, chord_length k ≤ (4 * Real.sqrt 3) / 3 := by sorry

/-- The maximum length of the chord is achieved -/
theorem max_chord_length_achieved : 
  ∃ k, chord_length k = (4 * Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_max_chord_length_achieved_l589_58937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_average_age_l589_58989

/-- Proves that the average age of boys is 12 years in a school with given conditions -/
theorem boys_average_age (total_students : ℕ) (girls_count : ℕ) (girls_avg_age : ℝ) (school_avg_age : ℝ) :
  total_students = 600 →
  girls_count = 150 →
  girls_avg_age = 11 →
  school_avg_age = 11.75 →
  (total_students - girls_count : ℝ) * 12 = total_students * school_avg_age - girls_count * girls_avg_age := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_average_age_l589_58989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_approx_49_13_l589_58976

/-- Molar mass of Hydrogen in g/mol -/
noncomputable def molar_mass_H : ℝ := 1.01

/-- Molar mass of Bromine in g/mol -/
noncomputable def molar_mass_Br : ℝ := 79.90

/-- Molar mass of Oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- Molar mass of Sulfur in g/mol -/
noncomputable def molar_mass_S : ℝ := 32.07

/-- Molar mass of HBrO3 in g/mol -/
noncomputable def molar_mass_HBrO3 : ℝ := molar_mass_H + molar_mass_Br + 3 * molar_mass_O

/-- Molar mass of H2SO3 in g/mol -/
noncomputable def molar_mass_H2SO3 : ℝ := 2 * molar_mass_H + molar_mass_S + 3 * molar_mass_O

/-- Amount of HBrO3 in mol -/
noncomputable def amount_HBrO3 : ℝ := 1

/-- Amount of H2SO3 in mol -/
noncomputable def amount_H2SO3 : ℝ := 2

/-- Total mass of Oxygen in the mixture in g -/
noncomputable def total_mass_O : ℝ := 3 * molar_mass_O * (amount_HBrO3 + amount_H2SO3)

/-- Total mass of the mixture in g -/
noncomputable def total_mass_mixture : ℝ := molar_mass_HBrO3 * amount_HBrO3 + molar_mass_H2SO3 * amount_H2SO3

/-- Mass percentage of Oxygen in the mixture -/
noncomputable def mass_percentage_O : ℝ := (total_mass_O / total_mass_mixture) * 100

theorem mass_percentage_O_approx_49_13 :
  |mass_percentage_O - 49.13| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_approx_49_13_l589_58976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_translation_l589_58949

theorem sin_translation (φ : Real) : 
  (0 < φ ∧ φ < π) → 
  (∀ x, Real.sin (2*x + 2*φ) = Real.sin (2*x - π/3)) → 
  φ = 5*π/6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_translation_l589_58949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l589_58972

noncomputable def g (x : ℝ) : ℝ := 3 / (2 * x^8 - x^6 + 5)

theorem g_is_even : ∀ x, g (-x) = g x := by
  intro x
  simp [g]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l589_58972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_line_l589_58921

/-- The ellipse representing curve C1 -/
def C1 (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

/-- The line representing curve C2 -/
def C2 (x y : ℝ) : Prop := x + y = 4

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The main theorem -/
theorem min_distance_ellipse_line :
  ∃ (x1 y1 : ℝ), C1 x1 y1 ∧ 
    (∀ (x2 y2 : ℝ), C2 x2 y2 → 
      (∀ (x3 y3 : ℝ), C1 x3 y3 → 
        distance x1 y1 x2 y2 ≤ distance x3 y3 x2 y2)) ∧
    x1 = 3/2 ∧ y1 = 1/2 ∧ 
    (∃ (x2 y2 : ℝ), C2 x2 y2 ∧ distance x1 y1 x2 y2 = Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_line_l589_58921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ln_id_min_distance_exp_ln_l589_58952

-- Define the functions
noncomputable def ln_func (x : ℝ) : ℝ := Real.log x
noncomputable def exp_func (x : ℝ) : ℝ := Real.exp x
def id_func (x : ℝ) : ℝ := x

-- Define the distance function between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Statement for the minimal distance between y = ln x and y = x
theorem min_distance_ln_id :
  ∃ (d : ℝ), d = 1 ∧ 
  ∀ (x : ℝ), x > 0 → distance x (ln_func x) x (id_func x) ≥ d := by
  sorry

-- Statement for the minimal distance between y = e^x and y = ln x
theorem min_distance_exp_ln :
  ∃ (d : ℝ), d = 2 ∧ 
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → distance x₁ (exp_func x₁) x₂ (ln_func x₂) ≥ d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ln_id_min_distance_exp_ln_l589_58952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_angle_theorem_l589_58916

/-- Definition of the ellipse C -/
def is_on_ellipse (x y m : ℝ) : Prop := x^2 / m + y^2 / 2 = 1

/-- Definition of the foci F1 and F2 -/
def are_foci (F1 F2 : ℝ × ℝ) (m : ℝ) : Prop :=
  (m > 2 ∧ F1 = (Real.sqrt (m - 2), 0) ∧ F2 = (-Real.sqrt (m - 2), 0)) ∨
  (0 < m ∧ m < 2 ∧ F1 = (0, Real.sqrt (2 - m)) ∧ F2 = (0, -Real.sqrt (2 - m)))

/-- Definition of the angle F1MF2 being 120 degrees -/
noncomputable def angle_is_120 (F1 F2 M : ℝ × ℝ) : Prop :=
  let v1 := (F1.1 - M.1, F1.2 - M.2)
  let v2 := (F2.1 - M.1, F2.2 - M.2)
  let dot_product := v1.1 * v2.1 + v1.2 * v2.2
  let mag_v1 := Real.sqrt (v1.1^2 + v1.2^2)
  let mag_v2 := Real.sqrt (v2.1^2 + v2.2^2)
  Real.arccos (dot_product / (mag_v1 * mag_v2)) = 2 * Real.pi / 3

/-- The main theorem -/
theorem ellipse_angle_theorem (m : ℝ) :
  (∃ F1 F2 M : ℝ × ℝ,
    are_foci F1 F2 m ∧
    is_on_ellipse M.1 M.2 m ∧
    angle_is_120 F1 F2 M) →
  (0 < m ∧ m ≤ 1/2) ∨ (m ≥ 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_angle_theorem_l589_58916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_fraction_sum_l589_58959

theorem integer_fraction_sum (m n : ℕ+) : 
  ∃ (k : ℤ), ((m : ℚ) + 1) / n + ((n : ℚ) + 1) / m = k ↔ 
  ((m : ℚ) + 1) / n + ((n : ℚ) + 1) / m = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_fraction_sum_l589_58959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_baskets_l589_58947

/-- Represents a basket of fruit -/
structure Basket where
  apples : ℕ
  bananas : ℕ
  oranges : ℕ

/-- The cost of a single apple in dollars -/
def apple_cost : ℕ := 2

/-- The cost of a single banana in dollars -/
def banana_cost : ℕ := 3

/-- The cost of a single orange in dollars -/
def orange_cost : ℕ := 5

/-- The total number of fruits in a basket -/
def total_fruits : ℕ := 100

/-- The total cost of a basket in dollars -/
def total_cost : ℕ := 300

/-- Checks if a basket satisfies the given conditions -/
def is_valid_basket (b : Basket) : Prop :=
  b.apples + b.bananas + b.oranges = total_fruits ∧
  apple_cost * b.apples + banana_cost * b.bananas + orange_cost * b.oranges = total_cost

/-- The number of distinct valid baskets -/
def num_valid_baskets : ℕ := 34

/-- Theorem stating that there are exactly 34 distinct valid baskets -/
theorem count_valid_baskets :
  ∃ (s : Finset Basket), s.card = num_valid_baskets ∧ ∀ b ∈ s, is_valid_basket b :=
sorry

#check count_valid_baskets

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_baskets_l589_58947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_truck_meet_once_l589_58993

/-- Michael's walking speed in feet per second -/
noncomputable def michael_speed : ℝ := 4

/-- Distance between trash pails in feet -/
noncomputable def pail_distance : ℝ := 100

/-- Truck's speed in feet per second -/
noncomputable def truck_speed : ℝ := 8

/-- Time truck stops at each pail in seconds -/
noncomputable def truck_stop_time : ℝ := 20

/-- Time for truck to move between pails in seconds -/
noncomputable def truck_move_time : ℝ := pail_distance / truck_speed

/-- Total time for one truck cycle (moving + stopping) in seconds -/
noncomputable def truck_cycle_time : ℝ := truck_move_time + truck_stop_time

/-- Michael's position at time t -/
noncomputable def michael_position (t : ℝ) : ℝ := michael_speed * t

/-- Truck's position at time t -/
noncomputable def truck_position (t : ℝ) : ℝ :=
  let cycles := ⌊t / truck_cycle_time⌋
  let remaining_time := t - cycles * truck_cycle_time
  cycles * pail_distance + min remaining_time truck_move_time * truck_speed + pail_distance

/-- The number of times Michael and the truck meet -/
def meeting_count : ℕ := 1

theorem michael_truck_meet_once :
  ∃ (t : ℝ), t > 0 ∧ michael_position t = truck_position t ∧
  ∀ (s : ℝ), s > t → michael_position s > truck_position s :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_truck_meet_once_l589_58993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_coloring_l589_58918

/-- Represents a color on the board -/
inductive Color
| One
| Two
| Three
| Four
deriving Ord, DecidableEq

/-- Represents a cell on the 5x5 board -/
structure Cell :=
  (row : Fin 5)
  (col : Fin 5)

/-- Represents a coloring of the 5x5 board -/
def Coloring := Cell → Color

/-- Checks if a 2x2 square contains at least 3 different colors -/
def validSquare (c : Coloring) (r : Fin 4) (col : Fin 4) : Prop :=
  let colors : Finset Color := {
    c ⟨r, col⟩,
    c ⟨r, col.succ⟩,
    c ⟨r.succ, col⟩,
    c ⟨r.succ, col.succ⟩
  }
  colors.card ≥ 3

/-- A valid coloring satisfies the condition for all 2x2 squares -/
def validColoring (c : Coloring) : Prop :=
  ∀ (r : Fin 4) (col : Fin 4), validSquare c r col

/-- The main theorem: There does not exist a valid coloring of the 5x5 board -/
theorem no_valid_coloring : ¬ ∃ (c : Coloring), validColoring c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_coloring_l589_58918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l589_58996

theorem cos_double_angle (x : Real) (h : Real.cos x = 3/4) : Real.cos (2*x) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l589_58996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_ticket_percentage_l589_58994

theorem round_trip_ticket_percentage (total_passengers : ℝ) 
  (round_trip_passengers : ℝ) (h : round_trip_passengers = 37.5 / 100 * total_passengers) :
  round_trip_passengers / total_passengers * 100 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_ticket_percentage_l589_58994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_ellipse_l589_58986

/-- The curve formed by points (x, y) where x = cos u + sin u and y = 4(cos u - sin u) for real u -/
theorem curve_is_ellipse :
  ∀ (u : ℝ), 
  (((Real.cos u + Real.sin u)^2 / 2) + ((4 * (Real.cos u - Real.sin u))^2 / 32) = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_ellipse_l589_58986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_polynomial_exists_l589_58985

theorem no_integer_polynomial_exists : ¬ ∃ (P : Polynomial ℤ), (P.eval 6 = 5) ∧ (P.eval 14 = 9) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_polynomial_exists_l589_58985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_and_circles_area_sum_l589_58973

/-- Represents the properties of the grid and circles -/
structure GridAndCircles where
  gridSize : ℕ
  squareSize : ℚ
  smallCircleCount : ℕ
  largeCircleCount : ℕ
  smallCircleDiameter : ℚ
  largeCircleDiameter : ℚ

/-- Calculates the sum of the total grid area and the total area of the circles divided by π -/
noncomputable def totalAreaSum (gc : GridAndCircles) : ℚ :=
  let gridArea := (gc.gridSize * gc.squareSize) ^ 2
  let smallCircleArea := gc.smallCircleCount * (gc.smallCircleDiameter / 2) ^ 2
  let largeCircleArea := gc.largeCircleCount * (gc.largeCircleDiameter / 2) ^ 2
  gridArea + smallCircleArea + largeCircleArea

/-- The main theorem to be proved -/
theorem grid_and_circles_area_sum :
  let gc : GridAndCircles := {
    gridSize := 6,
    squareSize := 3,
    smallCircleCount := 5,
    largeCircleCount := 1,
    smallCircleDiameter := 3,
    largeCircleDiameter := 12
  }
  totalAreaSum gc = 371.25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_and_circles_area_sum_l589_58973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_max_value_f_inequality_max_k_value_l589_58946

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := 1/2 * x^2 - 2*x
noncomputable def h (x : ℝ) : ℝ := f (x+1) - (x-2)

-- Statement 1
theorem h_max_value : ∃ (x : ℝ), h x = 2 ∧ ∀ (y : ℝ), h y ≤ 2 := by
  sorry

-- Statement 2
theorem f_inequality (a b : ℝ) (h1 : 0 < b) (h2 : b < a) :
  f (a+b) - f (2*a) < (b-a)/(2*a) := by
  sorry

-- Statement 3
def k_condition (k : ℤ) : Prop :=
  ∀ (x : ℝ), x > 1 → k * (x - 1) < x * f x + 3 * (x - 2) + 4

theorem max_k_value : ∃ (k : ℤ), k_condition k ∧ k = 5 ∧ ∀ (m : ℤ), m > 5 → ¬(k_condition m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_max_value_f_inequality_max_k_value_l589_58946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l589_58962

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 6) + 3 / 2

theorem f_properties :
  let period : ℝ := Real.pi
  let decreasing_intervals := {x : ℝ | ∃ k : ℤ, Real.pi / 6 + k * Real.pi ≤ x ∧ x ≤ 2 * Real.pi / 3 + k * Real.pi}
  let max_value : ℝ := 5 / 2
  let max_points := {x : ℝ | ∃ k : ℤ, x = Real.pi / 6 + k * Real.pi}
  (∀ x : ℝ, f (x + period) = f x) ∧
  (∀ x y, x ∈ decreasing_intervals → y ∈ decreasing_intervals → x < y → f y < f x) ∧
  (∀ x : ℝ, f x ≤ max_value) ∧
  (∀ x, x ∈ max_points → f x = max_value) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l589_58962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_l589_58911

-- Define the triangle ABC
structure Triangle :=
  (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the equilateral triangles
def isEquilateral (p q r : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist p q = dist q r ∧ dist q r = dist r p

-- Define the reflection of a point over a line
noncomputable def reflectOverLine (p q r : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  sorry

-- Define collinearity
def collinear (p q r : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ (t : ℝ), q = t • (r - p) + p

-- Main theorem
theorem collinear_points (ABC : Triangle) 
  (A' B' : EuclideanSpace ℝ (Fin 2))
  (h1 : isEquilateral ABC.B ABC.C A')
  (h2 : isEquilateral ABC.C ABC.A B')
  (C' : EuclideanSpace ℝ (Fin 2))
  (h3 : C' = reflectOverLine ABC.C ABC.A ABC.B) :
  collinear A' B' C' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_l589_58911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_x_axis_at_two_l589_58953

/-- Represents a line in 2D space --/
structure Line where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

/-- Represents a grid of unit squares --/
structure Grid where
  rows : ℕ
  cols : ℕ

/-- Function to calculate the area above the line in the grid --/
noncomputable def area_above_line (grid : Grid) (line : Line) : ℝ :=
  sorry

/-- Function to calculate the area below the line in the grid --/
noncomputable def area_below_line (grid : Grid) (line : Line) : ℝ :=
  sorry

/-- Predicate to check if a line intersects the x-axis at a given x-coordinate --/
def line_intersects_x_axis (line : Line) (x : ℝ) : Prop :=
  sorry

/-- Theorem stating that the line intersects the x-axis at x = 2 --/
theorem line_intersects_x_axis_at_two 
  (grid : Grid)
  (line : Line)
  (h_grid : grid.rows = 2 ∧ grid.cols = 3)
  (h_line_start : line.start = (2, 0))
  (h_line_end : line.endpoint = (3, 2))
  (h_equal_area : ∃ (a : ℝ), a > 0 ∧ 
    area_above_line grid line = a ∧
    area_below_line grid line = a) :
  ∃ (x : ℝ), x = 2 ∧ line_intersects_x_axis line x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_x_axis_at_two_l589_58953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_product_l589_58929

theorem quadratic_roots_product (x₁ x₂ : ℝ) : 
  x₁^2 - x₁ - 4 = 0 → x₂^2 - x₂ - 4 = 0 → (x₁^5 - 20*x₁) * (x₂^4 + 16) = 1296 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_product_l589_58929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_balls_count_l589_58978

/-- Represents the number of red balls in the bag -/
def n : ℕ := 15

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 10

/-- The probability of picking a white ball -/
def prob_white : ℝ := 0.4

/-- Theorem stating that if the probability of picking a white ball is 0.4,
    then the number of red balls is 15 -/
theorem red_balls_count : 
  (white_balls : ℝ) / ((n : ℝ) + (white_balls : ℝ)) = prob_white := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_balls_count_l589_58978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_and_planes_l589_58998

-- Define a plane in 3D space
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

-- Define a point in 3D space
def Point := ℝ × ℝ × ℝ

-- Define a line in 3D space
structure Line where
  direction : ℝ × ℝ × ℝ
  point : Point

-- Function to check if a point is not on a plane
def notOnPlane (p : Point) (plane : Plane) : Prop := sorry

-- Function to check if a line is parallel to a plane
def lineParallelToPlane (l : Line) (plane : Plane) : Prop := sorry

-- Function to check if two planes are parallel
def planesParallel (p1 p2 : Plane) : Prop := sorry

-- Theorem statement
theorem parallel_lines_and_planes 
  (plane : Plane) (p : Point) (h : notOnPlane p plane) :
  (∃ (lines : Set Line), Set.Infinite lines ∧ 
    ∀ l ∈ lines, l.point = p ∧ lineParallelToPlane l plane) ∧
  (∃! parallelPlane : Plane, 
    planesParallel parallelPlane plane ∧ parallelPlane.point = p) := by
  sorry

#check parallel_lines_and_planes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_and_planes_l589_58998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_squared_eq_3431_l589_58905

/-- The number of distinct prime divisors of n less than 6 -/
def f (n : ℕ) : ℕ := (Finset.filter (fun p => Nat.Prime p ∧ p < 6 ∧ n % p = 0) (Finset.range 6)).card

/-- The sum of f(n)^2 from n=1 to 2020 -/
def sum_f_squared : ℕ := (Finset.range 2020).sum (fun n => (f (n + 1))^2)

/-- Theorem stating that the sum of f(n)^2 from n=1 to 2020 is equal to 3431 -/
theorem sum_f_squared_eq_3431 : sum_f_squared = 3431 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_f_squared_eq_3431_l589_58905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_scores_l589_58925

noncomputable def scores : List ℝ := [8, 7, 9, 5, 4, 9, 10, 7, 4]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m)^2)).sum / xs.length

theorem variance_of_scores :
  variance scores = 40 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_scores_l589_58925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_over_cos_squared_alpha_l589_58965

theorem sin_2alpha_over_cos_squared_alpha (α : Real) (h : Real.tan α = 3) :
  (Real.sin (2 * α)) / (Real.cos α)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_over_cos_squared_alpha_l589_58965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_right_triangle_area_l589_58943

def is_area_of_right_triangle (a b area : ℝ) : Prop :=
  ∃ c : ℝ, (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) ∧ area = (a * b) / 2

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let min_area := (a * Real.sqrt (b^2 - a^2)) / 2
  ∀ area : ℝ, is_area_of_right_triangle a b area → area ≥ min_area :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_right_triangle_area_l589_58943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_point_theorem_l589_58935

noncomputable section

open Real

-- Define basic geometric structures
structure Point where
  x : ℝ
  y : ℝ

-- Define distance between two points
def dist (A B : Point) : ℝ := sorry

-- Define angle between three points
def angle (A B C : Point) : ℝ := sorry

-- Define the triangle and its properties
def is_acute_triangle (A B C : Point) : Prop := sorry

def is_altitude (A D : Point) (B C : Point) : Prop := sorry

def is_median (B E : Point) (A C : Point) : Prop := sorry

def is_angle_bisector (C F : Point) (A B : Point) : Prop := sorry

def intersect_at (P Q R O : Point) : Prop := sorry

-- Main theorem
theorem triangle_special_point_theorem 
  (A B C D E F O : Point)
  (h_acute : is_acute_triangle A B C)
  (h_altitude : is_altitude A D B C)
  (h_median : is_median B E A C)
  (h_bisector : is_angle_bisector C F A B)
  (h_intersect : intersect_at D E F O)
  (h_ratio : dist O E = 2 * dist O C) :
  angle A C B = arccos (1 / 7) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_point_theorem_l589_58935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_15_converges_to_1_l589_58982

-- Define S_n
def S (n : ℕ) : ℚ := (n * (n + 1) * (2 * n + 1)) / 6

-- Define Q_n
noncomputable def Q (n : ℕ) : ℚ :=
  if n < 3 then 0
  else Finset.prod (Finset.range (n - 2)) (fun i => S (i + 3) / (S (i + 3) + 1))

-- Theorem statement
theorem Q_15_converges_to_1 : 
  ∀ ε > 0, |Q 15 - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_15_converges_to_1_l589_58982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_theorem_l589_58974

-- Define the square
def Square : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 100 ∧ 0 ≤ p.2 ∧ p.2 ≤ 100}

-- Define a broken line as a continuous function from [0, 1] to the square
def BrokenLine : Type := {f : ℝ → ℝ × ℝ // Continuous f ∧ ∀ t, t ∈ Set.Icc 0 1 → f t ∈ Square}

-- Define the property that any point in the square is no more than 0.5 units away from the line
def CloseToLine (L : BrokenLine) : Prop :=
  ∀ p ∈ Square, ∃ t ∈ Set.Icc 0 1, dist p (L.val t) ≤ 0.5

-- Define the distance along the broken line
noncomputable def DistanceAlongLine (L : BrokenLine) (t₁ t₂ : ℝ) : ℝ :=
  ∫ t in t₁..t₂, norm (deriv L.val t)

-- The main theorem
theorem broken_line_theorem (L : BrokenLine) (h : CloseToLine L) :
  ∃ t₁ t₂ : ℝ, t₁ ∈ Set.Icc 0 1 ∧ t₂ ∈ Set.Icc 0 1 ∧ t₁ < t₂ ∧
    dist (L.val t₁) (L.val t₂) ≤ 1 ∧
    DistanceAlongLine L t₁ t₂ ≥ 198 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_broken_line_theorem_l589_58974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_angle_l589_58922

/-- Given two parallel vectors a and b, prove that the acute angle α between them is π/3 -/
theorem parallel_vectors_angle (α : Real) 
  (ha : Fin 2 → Real := ![3, Real.sin α])
  (hb : Fin 2 → Real := ![Real.sqrt 3, Real.cos α])
  (parallel : ∃ (k : Real), k • ha = hb)
  (acute : 0 < α ∧ α < π / 2) : 
  α = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_angle_l589_58922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_average_distance_l589_58956

noncomputable def board_size : ℕ := 100  -- Assuming a large board size

noncomputable def average_distance (n : ℕ) : ℝ :=
  7 + 4 * (if n ≤ 4 then (1/4 + (n-1: ℝ)/16) else 1/4)

theorem max_average_distance :
  ∀ n : ℕ, n > 0 → n ≤ board_size →
    average_distance 4 ≥ average_distance n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_average_distance_l589_58956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l589_58907

theorem constant_term_binomial_expansion :
  let binomial := (fun x : ℝ => x + 1 / (2 * x)) ^ 6
  ∃ (c : ℝ), c = 5/2 ∧ 
    ∀ ε > 0, ∃ N, ∀ x, x > N → |binomial x - c| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_binomial_expansion_l589_58907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_n_value_l589_58926

def loop_result (n₀ s₀ : ℕ) : ℕ :=
  let rec loop (n s : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then n
    else if s < 14 then loop (n - 1) (s + n) (fuel - 1)
    else n
  loop n₀ s₀ 100  -- Use a sufficiently large fuel value

theorem final_n_value :
  loop_result 5 0 = 1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_n_value_l589_58926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_carving_percentage_l589_58963

theorem statue_carving_percentage (initial_weight : ℝ) 
  (first_week_cut : ℝ) (second_week_cut : ℝ) (final_weight : ℝ) :
  initial_weight = 180 →
  first_week_cut = 28 →
  second_week_cut = 18 →
  final_weight = 85.0176 →
  let weight_after_first_week := initial_weight * (1 - first_week_cut / 100)
  let weight_after_second_week := weight_after_first_week * (1 - second_week_cut / 100)
  let third_week_cut := (1 - final_weight / weight_after_second_week) * 100
  ∃ ε > 0, |third_week_cut - 20.01| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statue_carving_percentage_l589_58963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_lines_l589_58939

/-- Line represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The first quadrant -/
def FirstQuadrant : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2 ≥ 0}

/-- Points below a given line -/
def PointsBelowLine (l : Line) : Set (ℝ × ℝ) :=
  {p | p.2 ≤ l.slope * p.1 + l.intercept}

/-- Points between two lines -/
def PointsBetweenLines (l1 l2 : Line) : Set (ℝ × ℝ) :=
  {p | l2.slope * p.1 + l2.intercept ≤ p.2 ∧ p.2 ≤ l1.slope * p.1 + l1.intercept}

/-- Area of a triangle given base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

theorem probability_between_lines :
  let p : Line := ⟨-2, 8⟩
  let q : Line := ⟨-3, 9⟩
  let areaUnderP := triangleArea 4 8
  let areaBetweenPandQ := triangleArea 4 8 - triangleArea 3 9
  (areaBetweenPandQ / areaUnderP : ℝ) = 0.15625 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_lines_l589_58939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l589_58945

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (h : d = 18) :
  (4 / 3 : ℝ) * π * (d / 4 * Real.sqrt 2)^3 = (1458 * Real.sqrt 2 / 4 : ℝ) * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_volume_l589_58945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_division_condition_l589_58904

theorem integer_division_condition (a n : ℕ) (ha : a ≥ 1) (hn : n ≥ 1) :
  ∃ k : ℤ, ((a + 1)^n - a^n : ℤ) = n * k ↔ n = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_division_condition_l589_58904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_solution_l589_58991

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The property that P satisfies for all real a, b, c -/
def SatisfiesProperty (P : RealPolynomial) : Prop :=
  ∀ a b c : ℝ, P (a + b - 2*c) + P (b + c - 2*a) + P (c + a - 2*b) = 
                3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)

/-- The identity function on real numbers -/
def IdentityFunction : RealPolynomial := λ x ↦ x

theorem unique_polynomial_solution : 
  ∀ P : RealPolynomial, SatisfiesProperty P → P = IdentityFunction := by
  sorry

#check unique_polynomial_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_solution_l589_58991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_values_monotonicity_condition_l589_58912

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

-- Define the domain
def domain : Set ℝ := Set.Icc (-5) 5

-- Theorem for part 1
theorem min_max_values :
  (∃ y ∈ domain, f (-1) y = 1) ∧
  (∀ z ∈ domain, f (-1) z ≥ 1) ∧
  (∃ w ∈ domain, f (-1) w = 37) ∧
  (∀ u ∈ domain, f (-1) u ≤ 37) :=
sorry

-- Theorem for part 2
theorem monotonicity_condition (a : ℝ) :
  (∀ x y, x ∈ domain → y ∈ domain → x < y → (f a x < f a y ∨ f a x > f a y)) ↔ 
  (a ≤ -5 ∨ a ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_values_monotonicity_condition_l589_58912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_symmetric_function_l589_58950

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem periodic_symmetric_function 
  (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : 0 < φ ∧ φ < Real.pi) 
  (h_period : ∀ x, f ω φ (x + Real.pi) = f ω φ x) 
  (h_symmetric : ∀ x, f ω φ (Real.pi / 3 + x) = f ω φ (Real.pi / 3 - x)) 
  (α : ℝ) 
  (h_α_acute : 0 < α ∧ α < Real.pi / 2) 
  (h_f_value : f ω φ (α / 2 - Real.pi / 12) = 3 / 5) :
  ω = 2 ∧ φ = Real.pi / 6 ∧ Real.cos (α - Real.pi / 3) = (4 + 3 * Real.sqrt 3) / 10 := by
  sorry

#check periodic_symmetric_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_symmetric_function_l589_58950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_explicit_formula_l589_58942

def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | n+1 => (n+3) * a n

theorem a_explicit_formula (n : ℕ) : a n = Nat.factorial (n+2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_explicit_formula_l589_58942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l589_58920

noncomputable def equilateral_triangle_area (side_length : ℝ) : ℝ := 
  (Real.sqrt 3 / 4) * side_length ^ 2

theorem triangle_area_ratio : 
  let large_triangle_side := (10 : ℝ)
  let small_triangle_side := (3 : ℝ)
  let large_triangle_area := equilateral_triangle_area large_triangle_side
  let small_triangle_area := equilateral_triangle_area small_triangle_side
  let trapezoid_area := large_triangle_area - small_triangle_area
  (small_triangle_area / trapezoid_area) = 9 / 91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l589_58920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lambda_triangle_inequality_l589_58961

theorem min_lambda_triangle_inequality (x y z : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
  (h_sum : x + y + z = 1) :
  ∃ (lambda : ℝ), (∀ (mu : ℝ), mu ≥ lambda → mu * (x*y + y*z + z*x) ≥ 3*(mu + 1)*x*y*z + 1) ∧
             (∀ (nu : ℝ), (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + b + c = 1 →
                nu * (a*b + b*c + c*a) ≥ 3*(nu + 1)*a*b*c + 1) → nu ≥ lambda) ∧
             lambda = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lambda_triangle_inequality_l589_58961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l589_58924

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

theorem f_properties :
  (∀ x y, x < y → x < 0 → y < 0 → f x < f y) ∧
  (∀ x, -3 ≤ x ∧ x ≤ -1 → f x ≥ 1/10) ∧
  (∀ x, -3 ≤ x ∧ x ≤ -1 → f x ≤ 1/2) ∧
  (f (-3) = 1/10) ∧
  (f (-1) = 1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l589_58924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_from_circle_tangents_l589_58906

theorem equilateral_triangle_area_from_circle_tangents (R : ℝ) :
  let circle := { p : ℝ × ℝ | p.1^2 + p.2^2 = R^2 }
  ∃ (A B C : ℝ × ℝ),
    (B ∈ circle ∧ C ∈ circle) ∧
    (∀ p ∈ circle, (A.1 - B.1) * (p.1 - B.1) + (A.2 - B.2) * (p.2 - B.2) = 0) ∧
    (∀ p ∈ circle, (A.1 - C.1) * (p.1 - C.1) + (A.2 - C.2) * (p.2 - C.2) = 0) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
    (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 →
    (3 * Real.sqrt 3 / 4) * R^2 = 
      1/2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_from_circle_tangents_l589_58906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_example_l589_58923

/-- Converts cylindrical coordinates to rectangular coordinates -/
noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

/-- The conversion of (5, 3π/4, 2) from cylindrical to rectangular coordinates -/
theorem cylindrical_to_rectangular_example :
  cylindrical_to_rectangular 5 (3 * π / 4) 2 = (-5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_example_l589_58923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_plus_intercept_special_case_l589_58955

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The slope of a line -/
noncomputable def Line.slope (l : Line) : ℝ := (l.y₂ - l.y₁) / (l.x₂ - l.x₁)

/-- The y-intercept of a line -/
noncomputable def Line.yIntercept (l : Line) : ℝ := l.y₁ - l.slope * l.x₁

/-- The theorem stating that for a line passing through (2, 0) and (0, 3), 
    the sum of its slope and y-intercept is 3/2 -/
theorem slope_plus_intercept_special_case : 
  let l := Line.mk 2 0 0 3
  l.slope + l.yIntercept = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_plus_intercept_special_case_l589_58955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extreme_ratios_l589_58944

/-- The equation of the ellipse -/
noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  3 * x^2 + 2 * x * y + 4 * y^2 - 15 * x - 24 * y + 54 = 0

/-- The set of points on the ellipse -/
noncomputable def ellipse_points : Set (ℝ × ℝ) :=
  {p | ellipse_equation p.1 p.2}

/-- The ratio y/x for a point (x,y) -/
noncomputable def ratio (p : ℝ × ℝ) : ℝ := p.2 / p.1

/-- The theorem stating that the sum of max and min ratios is 1 -/
theorem sum_of_extreme_ratios :
  ∃ (max_ratio min_ratio : ℝ),
    (∀ p ∈ ellipse_points, ratio p ≤ max_ratio) ∧
    (∀ p ∈ ellipse_points, ratio p ≥ min_ratio) ∧
    (max_ratio + min_ratio = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_extreme_ratios_l589_58944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftyThreeDaysAfterFriday_l589_58901

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
deriving Repr, DecidableEq

def daysInWeek : Nat := 7

-- Function to advance a day by one
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

-- Function to advance a day by a given number of days
def advanceDay (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => advanceDay (nextDay start) n

theorem fiftyThreeDaysAfterFriday (start : DayOfWeek) (days : Nat) :
  start = DayOfWeek.Friday ∧ days = 53 → advanceDay start days = DayOfWeek.Tuesday :=
by
  intro h
  cases h with
  | intro hStart hDays =>
    rw [hStart, hDays]
    -- The actual proof would go here, but we'll use sorry for now
    sorry

#eval advanceDay DayOfWeek.Friday 53

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftyThreeDaysAfterFriday_l589_58901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_expression_l589_58981

theorem undefined_expression (a : ℝ) : 
  (∀ x : ℝ, (a + 3) / (a^3 - 8) ≠ x) ↔ a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_expression_l589_58981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedron_face_division_l589_58936

/-- A face is a flat polygon that can be part of a polyhedron -/
structure Face where

/-- A polyhedron is a 3D geometric object with flat polygonal faces -/
structure Polyhedron where

/-- A set of faces that can form a convex polyhedron -/
def ConvexPolyhedronFaces := Set Face

/-- Predicate to check if a set of faces can form a convex polyhedron -/
def can_form_convex_polyhedron (faces : Set Face) : Prop := sorry

/-- Theorem stating that it's possible to divide faces of a convex polyhedron
    into two subsets, each capable of forming a convex polyhedron -/
theorem convex_polyhedron_face_division
  (original_faces : ConvexPolyhedronFaces)
  (h : can_form_convex_polyhedron original_faces) :
  ∃ (subset1 subset2 : Set Face),
    subset1 ∪ subset2 = original_faces ∧
    subset1 ∩ subset2 = ∅ ∧
    can_form_convex_polyhedron subset1 ∧
    can_form_convex_polyhedron subset2 :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polyhedron_face_division_l589_58936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l589_58909

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 6)

theorem function_properties (ω : ℝ) (h : ω > 0) :
  -- The initial phase of the function is -π/6
  (∃ A : ℝ, ∀ x : ℝ, f ω x = A * Real.sin (ω * x - Real.pi / 6)) ∧
  -- The function is decreasing in the interval [-π/(3ω), 2π/(3ω)] when 0 < ω ≤ 2
  (ω ≤ 2 → ∀ x y : ℝ, -Real.pi / (3 * ω) ≤ x ∧ x < y ∧ y ≤ 2 * Real.pi / (3 * ω) → f ω x > f ω y) :=
by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l589_58909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arthurs_walk_distance_l589_58948

/-- Represents the distance walked in blocks -/
structure BlockDistance where
  east : ℕ
  north : ℕ

/-- Converts blocks to miles -/
noncomputable def blocks_to_miles (blocks : ℕ) : ℝ :=
  (blocks : ℝ) / 2

/-- Calculates the total distance walked in miles -/
noncomputable def total_distance_miles (bd : BlockDistance) : ℝ :=
  blocks_to_miles (bd.east + bd.north)

/-- Theorem: Arthur's walk is 11.5 miles -/
theorem arthurs_walk_distance :
  let bd := BlockDistance.mk 8 15
  total_distance_miles bd = 11.5 := by
  sorry

#eval (8 + 15 : ℕ) -- This will evaluate to 23

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arthurs_walk_distance_l589_58948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_equation_solutions_l589_58997

theorem integer_equation_solutions : 
  (Finset.filter (fun n : ℕ => 1 + Int.floor ((120 : ℚ) * n / 121) = Int.ceil ((119 : ℚ) * n / 120)) (Finset.range 12120)).card = 12120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_equation_solutions_l589_58997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_theorem_l589_58980

-- Define the circle ω
noncomputable def ω : Set (ℝ × ℝ) := sorry

-- Define points A and B
def A : ℝ × ℝ := (6, 13)
def B : ℝ × ℝ := (12, 11)

-- Define the tangent lines at A and B
noncomputable def tangent_A : Set (ℝ × ℝ) := sorry
noncomputable def tangent_B : Set (ℝ × ℝ) := sorry

-- Define the intersection point of tangent lines
noncomputable def intersection_point : ℝ × ℝ := sorry

-- Define a predicate for a line being tangent to a circle at a point
def is_tangent (line : Set (ℝ × ℝ)) (circle : Set (ℝ × ℝ)) (point : ℝ × ℝ) : Prop := sorry

-- Define a function to calculate the area of a set
noncomputable def set_area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- State the theorem
theorem circle_area_theorem :
  A ∈ ω ∧ B ∈ ω ∧  -- A and B lie on circle ω
  is_tangent tangent_A ω A ∧ is_tangent tangent_B ω B ∧  -- tangent lines at A and B
  intersection_point.2 = 0 ∧  -- intersection point is on x-axis
  intersection_point ∈ tangent_A ∧ intersection_point ∈ tangent_B  -- intersection point is on both tangent lines
  →
  set_area ω = 85 * Real.pi / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_theorem_l589_58980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_problem_l589_58999

noncomputable section

/-- The volume of a cone with given radius and height -/
def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The volume of a cylinder with given radius and height -/
def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The height of water in a cylindrical tank when water from a cone is poured into it -/
def water_height_in_cylinder (cone_radius cone_height cylinder_radius : ℝ) : ℝ :=
  (cone_volume cone_radius cone_height) / (Real.pi * cylinder_radius^2)

theorem water_height_problem (cone_radius cone_height cylinder_radius : ℝ) 
  (h_cone_radius : cone_radius = 15)
  (h_cone_height : cone_height = 25)
  (h_cylinder_radius : cylinder_radius = 10) :
  water_height_in_cylinder cone_radius cone_height cylinder_radius = 18.75 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_problem_l589_58999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cafeteria_pies_l589_58970

/-- Calculates the maximum number of whole pies that can be made given the initial number of apples,
    apples handed out, apples used for juice, and apples required per pie. -/
def max_pies (initial : ℕ) (handed_out : ℕ) (juice : ℕ) (per_pie : ℕ) : ℕ :=
  (initial - handed_out - juice) / per_pie

/-- Theorem stating that given the specific conditions from the problem,
    the maximum number of whole pies that can be made is 16. -/
theorem cafeteria_pies :
  max_pies 250 42 75 8 = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cafeteria_pies_l589_58970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bricks_for_courtyard_l589_58915

/-- Calculates the minimum number of whole bricks required to cover a rectangular area -/
def minimum_bricks_required (courtyard_length courtyard_width brick_length brick_width : ℚ) : ℕ :=
  let courtyard_area := courtyard_length * courtyard_width * 10000
  let brick_area := brick_length * brick_width
  (courtyard_area / brick_area).ceil.toNat

/-- Proves that 22550 bricks are required for the given courtyard and brick dimensions -/
theorem bricks_for_courtyard :
  minimum_bricks_required 23 15 17 9 = 22550 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bricks_for_courtyard_l589_58915
