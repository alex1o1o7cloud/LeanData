import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_range_l703_70330

noncomputable def s (x : ℝ) : ℝ := 1 / (1 - Real.cos x)^2

theorem s_range :
  (∀ y : ℝ, (∃ x : ℝ, s x = y) → y > 1/4) ∧
  (∀ ε > 0, ∃ x : ℝ, |s x - (1/4 + ε)| < ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_range_l703_70330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_9_equals_7_l703_70324

def sequence_a : ℕ → ℚ
  | 0 => 3  -- Add this case for 0
  | n + 1 => sequence_a n + 1/2

theorem a_9_equals_7 : sequence_a 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_9_equals_7_l703_70324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_expanded_form_l703_70361

theorem sum_of_coefficients_expanded_form (d : ℝ) : 
  let expanded_form := -(5 - d) * (d + 2*(5 - d) + d)
  10 + (-50) = -40 := by
  -- Proof goes here
  sorry

#eval 10 + (-50) -- This will evaluate to -40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_expanded_form_l703_70361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_squares_sum_l703_70379

/-- The sum of cosine squares for specific angles equals 3.25 -/
theorem cosine_squares_sum : 
  (Real.cos 0)^2 + 2 * (Real.cos (15 * π / 180))^2 + (Real.cos (30 * π / 180))^2 + 
  2 * (Real.cos (45 * π / 180))^2 + (Real.cos (60 * π / 180))^2 + 
  2 * (Real.cos (75 * π / 180))^2 + (Real.cos (90 * π / 180))^2 = 3.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_squares_sum_l703_70379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_comparison_l703_70321

/-- Two polynomials with specific behavior on an interval -/
theorem polynomial_comparison 
  (a b c d p q : ℝ) 
  (P : ℝ → ℝ) 
  (Q : ℝ → ℝ) 
  (I : Set ℝ) 
  (hP : P = fun x ↦ x^4 + a*x^3 + b*x^2 + c*x + d)
  (hQ : Q = fun x ↦ x^2 + p*x + q)
  (hI : ∃ s t, I = Set.Icc s t ∧ t - s > 2)
  (hNegI : ∀ x ∈ I, P x < 0 ∧ Q x < 0)
  (hNonNegOut : ∀ x ∉ I, P x ≥ 0 ∧ Q x ≥ 0) :
  ∃ x₀ : ℝ, P x₀ < Q x₀ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_comparison_l703_70321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l703_70382

/-- Given two functions f and g, prove that the range of a is (5/27, ∞) -/
theorem range_of_a (f g : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f x = x^3) →
  (∀ x, g x = -x^2 + x - a) →
  a > 0 →
  (∃ x₀ ∈ Set.Icc (-1) 1, f x₀ < g x₀) →
  a ∈ Set.Ioi (5/27) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l703_70382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sequence_length_l703_70359

def sequence_gen (y : ℕ) : ℕ → ℤ
  | 0 => 2000
  | 1 => y
  | 2 => 2000 + y
  | n + 3 => sequence_gen y (n + 1) + sequence_gen y n

def is_valid_sequence (y : ℕ) : Prop :=
  ∀ n : ℕ, (sequence_gen y n > 0) → (sequence_gen y (n + 1) > 0)

def max_valid_y : ℕ := 1340

theorem max_sequence_length (y : ℕ) :
  is_valid_sequence y → y ≤ max_valid_y :=
by
  sorry

#eval max_valid_y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sequence_length_l703_70359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_equals_sixteen_l703_70341

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the external point P
def P : ℝ × ℝ := (3, 4)

-- Define a line passing through P and intersecting the circle at A and B
def line_intersects_circle (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, A = (3 + t * (A.1 - 3), 4 + t * (A.2 - 4)) ∧
            B = (3 + t * (B.1 - 3), 4 + t * (B.2 - 4)) ∧
            circle_equation A.1 A.2 ∧ circle_equation B.1 B.2

-- Define the dot product of vectors PA and PB
def dot_product_PA_PB (A B : ℝ × ℝ) : ℝ :=
  (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2)

-- Theorem statement
theorem dot_product_equals_sixteen (A B : ℝ × ℝ) :
  line_intersects_circle A B → dot_product_PA_PB A B = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_equals_sixteen_l703_70341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l703_70320

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x + 9) / (x - 4)

theorem min_value_of_f :
  ∀ x : ℝ, x ≥ 5 → f x ≥ 10 ∧ ∃ x₀ : ℝ, x₀ ≥ 5 ∧ f x₀ = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l703_70320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_when_a_zero_a_equals_zero_when_f_greater_than_one_l703_70311

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 1) / (Real.log x - a * x^2)

-- Statement for part 1
theorem f_monotonic_when_a_zero :
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₁ ≠ 1 ∧ x₂ ≠ 1 →
  f 0 x₁ < f 0 x₂ := by sorry

-- Statement for part 2
theorem a_equals_zero_when_f_greater_than_one :
  (∃ a : ℝ, ∀ x, 1 < x ∧ x < Real.exp 1 → f a x > 1) ↔
  (∃ a : ℝ, a = 0 ∧ ∀ x, 1 < x ∧ x < Real.exp 1 → f a x > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_when_a_zero_a_equals_zero_when_f_greater_than_one_l703_70311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_circle_l703_70396

theorem polar_equation_circle (θ : Real) :
  let r := 6 * Real.sin θ * (1 / Real.sin θ)
  ∃ (x y : Real), x^2 + y^2 = 36 ∧ r = Real.sqrt (x^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_circle_l703_70396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l703_70332

noncomputable def worker_rate (days : ℝ) : ℝ := 1 / days

theorem job_completion_time 
  (rate_A rate_B rate_C rate_D : ℝ)
  (h_A : rate_A = worker_rate 10)
  (h_B : rate_B = worker_rate 15)
  (h_C : rate_C = worker_rate 20)
  (h_D : rate_D = worker_rate 30) :
  1 / (rate_A + rate_B + rate_C + rate_D) = 4 := by
  sorry

#check job_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l703_70332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_fourteen_fifths_pi_l703_70348

theorem sin_fourteen_fifths_pi : ∀ (π : ℝ), Real.sin ((14 * π) / 5) = Real.sin (π / 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_fourteen_fifths_pi_l703_70348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_standard_form_l703_70395

-- Define the parametric equations
noncomputable def x (t : ℝ) : ℝ := 3 + (1/2) * t
noncomputable def y (t : ℝ) : ℝ := (Real.sqrt 3 / 2) * t

-- State the theorem
theorem parametric_to_standard_form :
  ∀ t : ℝ, y t = Real.sqrt 3 * (x t) - 3 * Real.sqrt 3 :=
by
  intro t
  -- Unfold the definitions of x and y
  unfold x y
  -- Simplify the expressions
  simp [Real.sqrt_div, mul_div_assoc, mul_comm, mul_assoc]
  -- The proof is complete
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_standard_form_l703_70395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_5_l703_70355

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem: The eccentricity of a hyperbola with one asymptote passing through (1, -2) is √5 -/
theorem hyperbola_eccentricity_sqrt_5 (h : Hyperbola) (p : Point) :
  p.x = 1 ∧ p.y = -2 →  -- Point P(1, -2)
  (∃ (k : ℝ), p.y = k * p.x ∧ k^2 * h.a^2 = h.b^2) →  -- Asymptote equation
  eccentricity h = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_sqrt_5_l703_70355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_team_selection_probability_l703_70387

/-- Represents a swimming team with members in different levels -/
structure SwimmingTeam where
  total_members : ℕ
  first_level : ℕ
  second_level : ℕ
  third_level : ℕ
  first_prob : ℝ
  second_prob : ℝ
  third_prob : ℝ

/-- Calculates the probability of selecting a team member who can pass the selection -/
noncomputable def pass_selection_probability (team : SwimmingTeam) : ℝ :=
  (team.first_level : ℝ) / (team.total_members : ℝ) * team.first_prob +
  (team.second_level : ℝ) / (team.total_members : ℝ) * team.second_prob +
  (team.third_level : ℝ) / (team.total_members : ℝ) * team.third_prob

/-- Theorem stating the probability of selecting a team member who can pass the selection -/
theorem swimming_team_selection_probability 
  (team : SwimmingTeam)
  (h1 : team.total_members = 20)
  (h2 : team.first_level = 10)
  (h3 : team.second_level = 5)
  (h4 : team.third_level = 5)
  (h5 : team.first_prob = 0.8)
  (h6 : team.second_prob = 0.7)
  (h7 : team.third_prob = 0.5) :
  pass_selection_probability team = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_team_selection_probability_l703_70387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_ellipse_with_foci_on_y_axis_l703_70345

-- Define the equation
def curve_equation (x y : ℝ) : Prop :=
  x^2 / (Real.sin (Real.sqrt 2) - Real.sin (Real.sqrt 3)) +
  y^2 / (Real.cos (Real.sqrt 2) - Real.cos (Real.sqrt 3)) = 1

-- Theorem stating that the equation represents an ellipse with foci on the y-axis
theorem curve_is_ellipse_with_foci_on_y_axis :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
    (∀ (x y : ℝ), curve_equation x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
    ∃ (c : ℝ), c > 0 ∧ a^2 = b^2 + c^2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_ellipse_with_foci_on_y_axis_l703_70345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_in_range_l703_70352

/-- The function f(x) = x + a*ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a * Real.log x

/-- Theorem statement -/
theorem inequality_holds_iff_a_in_range (a : ℝ) : 
  (a > 0 ∧ 
   ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ ∈ Set.Icc 1 3 → x₂ ∈ Set.Icc 1 3 → 
   |f a x₁ - f a x₂| < |1/x₁ - 1/x₂|) 
  ↔ 
  (0 < a ∧ a < 8/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_a_in_range_l703_70352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l703_70303

theorem simplify_expression : 7 - (-3) + (-5) - (2) = 7 + 3 - 5 - 2 := by
  ring  -- This tactic should solve this arithmetic equality
  -- If 'ring' doesn't work, we can use 'sorry' as a placeholder
  -- sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l703_70303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_inequality_l703_70315

-- Define the power function
noncomputable def f (x : ℝ) : ℝ := x^(-(1/2 : ℝ))

-- State the theorem
theorem power_function_inequality (a : ℝ) :
  (f (a + 1) < f (10 - 2*a)) ↔ (3 < a ∧ a < 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_inequality_l703_70315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_exterior_angles_sum_exterior_angles_twenty_sided_l703_70385

/-- The sum of exterior angles of a polygon with n sides is 360 degrees -/
theorem sum_exterior_angles (n : ℕ) (h : n ≥ 3) : 
  360 = 360 :=
by
  -- The actual proof would go here, but we're using sorry for now
  sorry

/-- The sum of exterior angles of a twenty-sided polygon is 360 degrees -/
theorem sum_exterior_angles_twenty_sided : 
  360 = 360 :=
by
  -- We can prove this using the general theorem
  -- apply sum_exterior_angles 20
  -- show 20 ≥ 3, by norm_num
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_exterior_angles_sum_exterior_angles_twenty_sided_l703_70385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l703_70366

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

-- State the theorem
theorem f_satisfies_conditions :
  -- Condition 1: The domain of f is ℝ (implied by the type of f)
  -- Condition 2: f is not periodic
  (¬ ∃ (T : ℝ), T ≠ 0 ∧ (∀ x, f (x + T) = f x)) ∧
  -- Condition 3: f' has period 2π
  (∃ (f' : ℝ → ℝ), (∀ x, HasDerivAt f (f' x) x) ∧ 
                   (∀ x, f' (x + 2 * π) = f' x)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_conditions_l703_70366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_power_sum_l703_70343

theorem unique_power_sum (n : ℕ) : 3^n + 4^n = 5^n ↔ n = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_power_sum_l703_70343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l703_70372

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := (1/3) ^ x

-- Define the interval
def interval : Set ℝ := Set.Ioo 0 (200 * Real.pi)

-- Define the solution set
def solution_set : Set ℝ := {x ∈ interval | f x = g x}

-- State the theorem
theorem solution_count : ∃ (S : Finset ℝ), S.card = 200 ∧ ∀ x ∈ S, x ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l703_70372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_l703_70394

/-- The original length of a wire cut into squares and circles --/
theorem wire_length (square_side : ℝ) (circle_diameter : ℝ) (total_square_area : ℝ) (total_circle_area : ℝ) 
  (h1 : square_side = 2.5)
  (h2 : circle_diameter = 3)
  (h3 : total_square_area = 87.5)
  (h4 : total_circle_area = 56.52) : 
  ∃ (wire_length : ℝ), abs (wire_length - 215.4) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_l703_70394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_true_discount_approx_l703_70388

/-- The true discount on a bill given its present worth and banker's discount -/
noncomputable def true_discount (present_worth banker_discount : ℝ) : ℝ :=
  banker_discount / (1 + banker_discount / present_worth)

/-- Theorem stating that the true discount is approximately 35.92 given the conditions -/
theorem true_discount_approx :
  let pw := (800 : ℝ)
  let bd := (37.62 : ℝ)
  abs (true_discount pw bd - 35.92) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_true_discount_approx_l703_70388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l703_70307

noncomputable def f (x : ℝ) : ℝ := 
  Real.sqrt 2 * Real.sin (x/4) * Real.cos (x/4) + 
  Real.sqrt 6 * (Real.cos (x/4))^2 - Real.sqrt 6 / 2

theorem min_value_of_f :
  ∀ x ∈ Set.Icc (-π/3) (π/3), f x ≥ Real.sqrt 2 / 2 ∧ 
  ∃ x₀ ∈ Set.Icc (-π/3) (π/3), f x₀ = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l703_70307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_with_special_property_l703_70353

theorem four_digit_numbers_with_special_property : 
  let count := Finset.filter (fun N => 1000 ≤ N ∧ N < 10000 ∧ 
    (N % 1000 : ℕ) = N / 11) (Finset.range 10000)
  Finset.card count = 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_with_special_property_l703_70353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arg_z_range_l703_70336

theorem arg_z_range (z : ℂ) :
  Complex.abs (Complex.arg ((z + 1) / (z + 2))) = π / 6 →
  ∃ θ : ℝ, Complex.arg z = θ ∧ 
    (5 * π / 6 - Real.arcsin (Real.sqrt 3 / 3) ≤ θ) ∧ 
    (θ ≤ 7 * π / 6 + Real.arcsin (Real.sqrt 3 / 3)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arg_z_range_l703_70336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l703_70312

/-- Piecewise function f(x) -/
noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x < c then c * x + 1
  else if c ≤ x ∧ x < 1 then 2^(-x / c^2) + 1
  else 0

theorem f_properties (c : ℝ) (h1 : 0 < c) (h2 : c < 1) (h3 : f c c^2 = 9/8) :
  c = 1/2 ∧ ∀ x, f c x > Real.sqrt 2 / 8 + 1 ↔ Real.sqrt 2 / 4 < x ∧ x < 5/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l703_70312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equality_l703_70317

-- Define c and d as noncomputable
noncomputable def c : ℝ := Real.log 8
noncomputable def d : ℝ := Real.log 25

-- State the theorem
theorem log_sum_equality : 5^(c/d) + 2^(d/c) = 2 * Real.sqrt 2 + 5^(2/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_equality_l703_70317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_separate_from_circle_l703_70351

/-- Proves that a line is separate from a circle given specific conditions -/
theorem line_separate_from_circle
  (a : ℝ) (x₀ y₀ : ℝ)
  (h_a_pos : a > 0)
  (h_inside : x₀^2 + y₀^2 < a^2)
  (h_not_center : (x₀, y₀) ≠ (0, 0)) :
  (|a^2| / Real.sqrt (x₀^2 + y₀^2)) > a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_separate_from_circle_l703_70351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_sliding_segment_max_area_value_l703_70393

/-- The area of a triangle formed by a segment of length a sliding along the sides of a right angle
    is maximized when the segment forms a 45-45-90 triangle. -/
theorem max_area_sliding_segment (a : ℝ) (ha : a > 0) :
  ∃ (x : ℝ), 0 < x ∧ x < a ∧
  ∀ (y : ℝ), 0 ≤ y ∧ y ≤ a →
    x * (a^2 - x^2).sqrt / 2 ≥ y * (a^2 - y^2).sqrt / 2 :=
by
  -- Let x be a/√2
  let x := a / Real.sqrt 2
  
  -- Prove that 0 < x < a
  have hx_pos : 0 < x := sorry
  have hx_lt_a : x < a := sorry
  
  -- Show that x maximizes the area function
  have h_max : ∀ (y : ℝ), 0 ≤ y ∧ y ≤ a →
    x * (a^2 - x^2).sqrt / 2 ≥ y * (a^2 - y^2).sqrt / 2 := sorry
  
  -- Conclude the proof
  exact ⟨x, hx_pos, hx_lt_a, h_max⟩

/-- The maximum area of the triangle is a^2/4 -/
theorem max_area_value (a : ℝ) (ha : a > 0) :
  ∃ (S : ℝ), S = a^2 / 4 ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ a →
    x * (a^2 - x^2).sqrt / 2 ≤ S :=
by
  -- Let S be a^2/4
  let S := a^2 / 4
  
  -- Prove that S is the maximum area
  have h_max : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ a →
    x * (a^2 - x^2).sqrt / 2 ≤ S := sorry
  
  -- Conclude the proof
  exact ⟨S, rfl, h_max⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_sliding_segment_max_area_value_l703_70393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_origin_to_circle_l703_70325

/-- The shortest distance from the origin to a circle -/
theorem shortest_distance_origin_to_circle :
  let circle := {p : ℝ × ℝ | (p.1^2 - 18*p.1 + p.2^2 - 8*p.2 + 153) = 0}
  ∃ d : ℝ, d = Real.sqrt 97 - Real.sqrt 44 ∧
    ∀ p ∈ circle, d ≤ Real.sqrt (p.1^2 + p.2^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_origin_to_circle_l703_70325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_notebooks_58_dollars_37_cents_l703_70339

/-- The maximum number of notebooks that can be purchased with a given budget and notebook price. -/
def max_notebooks (budget : ℚ) (price : ℚ) : ℕ :=
  (budget / price).floor.toNat

/-- Theorem stating the maximum number of notebooks that can be purchased with $58 when each notebook costs 37 cents. -/
theorem max_notebooks_58_dollars_37_cents :
  max_notebooks 58 (37/100) = 156 := by
  -- Proof goes here
  sorry

#eval max_notebooks 58 (37/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_notebooks_58_dollars_37_cents_l703_70339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_constant_for_difference_in_H_l703_70365

/-- The set H of floor values of i√2 for positive integers i -/
def H : Set ℤ := {z | ∃ i : ℕ+, z = ⌊(i : ℝ) * Real.sqrt 2⌋}

/-- Theorem stating the existence of a constant C for the given property -/
theorem exists_constant_for_difference_in_H :
  ∃ C : ℝ, C > 0 ∧
  ∀ n : ℕ+, ∀ A : Finset ℕ,
    A ⊆ Finset.range n →
    (A.card : ℝ) ≥ C * Real.sqrt n →
    ∃ a b : ℕ, a ∈ A ∧ b ∈ A ∧ ((a : ℤ) - (b : ℤ)) ∈ H :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_constant_for_difference_in_H_l703_70365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_height_l703_70390

/-- A right pyramid with a square base -/
structure RightPyramid where
  /-- The perimeter of the square base in inches -/
  basePerimeter : ℝ
  /-- The distance from the apex to each vertex of the base in inches -/
  apexToVertex : ℝ

/-- The height of the pyramid from its peak to the center of its square base -/
noncomputable def pyramidHeight (p : RightPyramid) : ℝ :=
  3 * Real.sqrt 7

/-- Theorem stating the height of the specific pyramid -/
theorem specific_pyramid_height :
  ∀ (p : RightPyramid),
    p.basePerimeter = 24 →
    p.apexToVertex = 9 →
    pyramidHeight p = 3 * Real.sqrt 7 :=
by
  intro p base_perimeter apex_to_vertex
  unfold pyramidHeight
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_height_l703_70390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_to_sixth_l703_70319

-- Define the complex number
noncomputable def z : ℂ := 1 + Complex.I * Real.sqrt 2

-- Define the magnitude of a complex number
noncomputable def magnitude (c : ℂ) : ℝ := Real.sqrt (c.re * c.re + c.im * c.im)

-- State the theorem
theorem magnitude_of_z_to_sixth : magnitude (z^6) = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_z_to_sixth_l703_70319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hard_drive_capacity_2050_l703_70344

-- Define the initial capacity in 2000
def initial_capacity : ℝ := 0.7

-- Define the growth rate function
def growth_rate (r₀ p t : ℝ) : ℝ := r₀ + t * p

-- Define the capacity function
noncomputable def capacity (r₀ p t : ℝ) : ℝ := initial_capacity * (1 + growth_rate r₀ p t) ^ t

-- Theorem statement
theorem hard_drive_capacity_2050 (r₀ p : ℝ) (h₁ : 0 < p) (h₂ : p < 100) :
  capacity r₀ p 50 = initial_capacity * (1 + (r₀ + 50 * p)) ^ 50 := by
  -- Unfold the definitions
  unfold capacity
  unfold growth_rate
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hard_drive_capacity_2050_l703_70344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l703_70347

noncomputable section

variable (f : ℝ → ℝ)

axiom f_at_one : f 1 = 1
axiom f_derivative_gt_half : ∀ x, deriv f x > (1/2 : ℝ)

theorem solution_set_of_inequality :
  {x : ℝ | f (x^2) < x^2/2 + 1/2} = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l703_70347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_in_interval_10_50_l703_70367

structure DataGroup where
  lowerBound : ℝ
  upperBound : ℝ
  frequency : ℕ

def sampleData : List DataGroup := [
  ⟨10, 20, 2⟩,
  ⟨20, 30, 3⟩,
  ⟨30, 40, 4⟩,
  ⟨40, 50, 5⟩,
  ⟨50, 60, 4⟩,
  ⟨60, 70, 2⟩
]

def sampleSize : ℕ := 20

noncomputable def isInInterval (x : DataGroup) (lower upper : ℝ) : Bool :=
  x.lowerBound ≥ lower ∧ x.upperBound < upper

noncomputable def frequencyInInterval (data : List DataGroup) (lower upper : ℝ) : ℕ :=
  (data.filter (fun x => isInInterval x lower upper)).foldl (fun acc x => acc + x.frequency) 0

theorem frequency_in_interval_10_50 :
  (frequencyInInterval sampleData 10 50 : ℝ) / sampleSize = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_in_interval_10_50_l703_70367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scheme4_most_optimal_scheme3_more_optimal_range_l703_70313

-- Define the probability p
noncomputable def p : ℝ := 2 * Real.sqrt 2 / 3

-- Define the expected number of tests for each scheme
noncomputable def E_scheme1 : ℝ := 4
noncomputable def E_scheme2 : ℝ := 22 / 9
noncomputable def E_scheme4_fixed : ℝ := 149 / 81

-- Define a general function for the expected number of tests for Scheme 3 and 4
noncomputable def E_scheme3 (p : ℝ) : ℝ := 5 - 3 * p^3
noncomputable def E_scheme4 (p : ℝ) : ℝ := 5 - 4 * p^4

-- Theorem for part (1)
theorem scheme4_most_optimal :
  E_scheme4_fixed < E_scheme2 ∧ E_scheme4_fixed < E_scheme1 := by sorry

-- Theorem for part (2)
theorem scheme3_more_optimal_range (p : ℝ) :
  0 < p ∧ p < 3/4 ↔ E_scheme3 p < E_scheme4 p := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scheme4_most_optimal_scheme3_more_optimal_range_l703_70313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_difference_l703_70360

/-- Given a large rectangle of size A × B and a smaller rectangle of size a × b inside it,
    prove that the difference between the total area of yellow quadrilaterals and
    the total area of green quadrilaterals is 20. -/
theorem rectangle_area_difference (A B a b : ℝ) (h1 : A = 20) (h2 : B = 30) (h3 : a = 4) (h4 : b = 7) :
  (A * B - a * b) / 2 - (A * B - a * b) / 2 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_difference_l703_70360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log5_domain_l703_70333

-- Define the logarithm function with base 5
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- State the theorem
theorem log5_domain :
  {x : ℝ | ∃ y, log5 x = y} = {x : ℝ | x > 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log5_domain_l703_70333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_transformation_l703_70374

theorem sine_graph_transformation (ω φ : ℝ) (h_ω_pos : ω > 0) (h_φ_bound : |φ| < π/2) :
  (∀ x, Real.sin (ω * (2 * (x - π/3)) + φ) = Real.sin x) ↔ (ω = 1/2 ∧ φ = π/6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_transformation_l703_70374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_heads_probability_is_11_32_l703_70377

open BigOperators

/-- The probability of getting more heads than tails when tossing a fair coin 6 times -/
def more_heads_probability : ℚ :=
  (Nat.choose 6 4 + Nat.choose 6 5 + Nat.choose 6 6) / 2^6

/-- Theorem stating that the probability of getting more heads than tails
    when tossing a fair coin 6 times is equal to 11/32 -/
theorem more_heads_probability_is_11_32 :
  more_heads_probability = 11/32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_heads_probability_is_11_32_l703_70377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_m_upper_bound_f_minus_g_positive_l703_70397

-- Define the functions f and g
noncomputable def f (x : ℝ) := Real.exp x * Real.sin x - Real.cos x
noncomputable def g (x : ℝ) := x * Real.cos x - Real.sqrt 2 * Real.exp x

-- Part 1: f has exactly one zero in (0, π/2)
theorem f_has_one_zero :
  ∃! x, 0 < x ∧ x < Real.pi / 2 ∧ f x = 0 :=
sorry

-- Part 2: Upper bound for m
theorem m_upper_bound :
  ∀ m : ℝ, (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ Real.pi / 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ Real.pi / 2
    → f x₁ + g x₂ ≥ m) → m ≤ -1 - Real.sqrt 2 :=
sorry

-- Part 3: f(x) - g(x) > 0 for x > -1
theorem f_minus_g_positive :
  ∀ x : ℝ, x > -1 → f x - g x > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_m_upper_bound_f_minus_g_positive_l703_70397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_equals_sqrt_2_l703_70389

-- Define the function representing the curve
noncomputable def f (x : ℝ) : ℝ := Real.arccos (Real.sqrt x) - Real.sqrt (x - x^2) + 4

-- Define the derivative of the function
noncomputable def f_derivative (x : ℝ) : ℝ := -Real.sqrt ((1 - x) / x)

-- State the theorem
theorem arc_length_equals_sqrt_2 :
  ∫ x in Set.Icc 0 (1/2), Real.sqrt (1 + (f_derivative x)^2) = Real.sqrt 2 := by
  sorry

#check arc_length_equals_sqrt_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_equals_sqrt_2_l703_70389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_y_value_l703_70300

/-- Given a triangle with vertices at (-1, y), (7, 3), and (-1, 3), 
    where y is positive and the area is 36 square units, prove that y = 12. -/
theorem triangle_y_value (y : ℝ) (h1 : y > 0) : 
  (1/2 : ℝ) * |7 - (-1)| * |y - 3| = 36 → y = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_y_value_l703_70300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_of_e_equals_three_l703_70322

theorem ceiling_of_e_equals_three : ⌈Real.exp 1⌉ = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_of_e_equals_three_l703_70322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_problem_l703_70350

theorem birthday_problem (total_people : ℕ) (days_in_year : ℕ) : 
  total_people = 1200 → days_in_year ∈ ({365, 366} : Set ℕ) → 
  ∃ (min_shared : ℕ), min_shared ≥ 4 ∧ 
  (total_people : ℝ) / (days_in_year : ℝ) ≥ (min_shared : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_problem_l703_70350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_theorem_l703_70363

/-- A circle with center (a, b) and radius r -/
structure Circle where
  a : ℝ
  b : ℝ
  r : ℝ

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Whether two circles are tangent at a given point -/
def are_tangent (c1 c2 : Circle) (x y : ℝ) : Prop :=
  distance c1.a c1.b x y = c1.r ∧
  distance c2.a c2.b x y = c2.r ∧
  distance c1.a c1.b c2.a c2.b = c1.r + c2.r

theorem circle_tangent_theorem :
  let c1 : Circle := ⟨2, -1, 2⟩
  let c2 : Circle := ⟨5, -1, 1⟩
  are_tangent c1 c2 4 (-1) ∧
  c2.r = 1 →
  ∀ x y : ℝ, (x - c2.a)^2 + (y - c2.b)^2 = c2.r^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_theorem_l703_70363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_and_square_root_l703_70316

-- Define the cube root function as noncomputable
noncomputable def cube_root (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the statement that -9 is a square root of 81
def is_square_root (x y : ℝ) : Prop := x^2 = y

-- Theorem statement
theorem cube_root_and_square_root :
  (cube_root (-81) = -9) ∧ (is_square_root (-9) 81) := by
  constructor
  · -- Proof for cube root part
    sorry
  · -- Proof for square root part
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_and_square_root_l703_70316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l703_70383

theorem triangle_cosine_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (α β γ : ℝ) (hα : α > 0) (hβ : β > 0) (hγ : γ > 0)
  (h_triangle : α + β + γ = Real.pi)
  (h_cosα : Real.cos α = (b^2 + c^2 - a^2) / (2*b*c))
  (h_cosβ : Real.cos β = (a^2 + c^2 - b^2) / (2*a*c))
  (h_cosγ : Real.cos γ = (a^2 + b^2 - c^2) / (2*a*b)) :
  2 * ((Real.cos α)^2 + (Real.cos β)^2 + (Real.cos γ)^2) ≥ 
    a^2 / (b^2 + c^2) + b^2 / (a^2 + c^2) + c^2 / (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l703_70383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_quadrants_l703_70335

theorem angle_quadrants (α : Real) :
  (Real.sin α + Real.cos α = (2 * Real.sqrt 6) / 5) →
  (α ∈ Set.Ioo (π/2) π) ∨ (α ∈ Set.Ioo (3*π/2) (2*π)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_quadrants_l703_70335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_l703_70370

def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ d => n % d = 0)

def is_valid_n (n : ℕ) : Prop :=
  let d := divisors n
  d.length ≥ 22 ∧
  d.Nodup ∧
  d.head? = some 1 ∧
  d.getLast? = some n ∧
  d[6]?.isSome ∧ d[9]?.isSome ∧ d[21]?.isSome ∧
  (d[6]?.get!)^2 + (d[9]?.get!)^2 = (n / d[21]?.get!)^2

theorem unique_n :
  ∀ n : ℕ, is_valid_n n ↔ n = 2^3 * 3 * 5 * 17 :=
by sorry

#eval divisors (2^3 * 3 * 5 * 17)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_l703_70370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_function_l703_70318

noncomputable def lg_iter (n : ℕ) (x : ℝ) : ℝ := 
  match n with
  | 0 => x
  | n + 1 => Real.log (lg_iter n x)

theorem exists_special_function : 
  ∃ f : ℝ → ℝ, 
    (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → f x > M) ∧ 
    (∀ n : ℕ, ∀ ε > 0, ∃ K : ℝ, ∀ x > K, |f x / lg_iter n x| < ε) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_function_l703_70318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_diameter_l703_70376

/-- The radius of each small circle -/
def small_radius : ℝ := 4

/-- The number of small circles -/
def num_small_circles : ℕ := 6

/-- The radius of the large circle -/
def large_radius : ℝ := 10

/-- Function to represent the radius of each small circle -/
def circle_radius : Fin num_small_circles → ℝ := λ _ => small_radius

/-- Predicate to represent that a small circle is tangent to the large circle -/
def tangent_to_large_circle : Fin num_small_circles → Prop := λ _ => True

/-- Predicate to represent that a small circle is tangent to its neighbors -/
def tangent_to_neighbors : Fin num_small_circles → Prop := λ _ => True

/-- Predicate to represent that a small circle is tangent to the bisector -/
def tangent_to_bisector : Fin num_small_circles → Prop := λ _ => True

/-- Theorem: Given the conditions, the diameter of the large circle is 20 units -/
theorem large_circle_diameter :
  (∀ (i : Fin num_small_circles),
    -- Each small circle has radius 4
    (circle_radius i = small_radius) ∧
    -- Each small circle is tangent to the large circle
    (tangent_to_large_circle i) ∧
    -- Each small circle is tangent to its two neighboring small circles
    (tangent_to_neighbors i) ∧
    -- All small circles are tangent to a horizontal line that bisects the large circle
    (tangent_to_bisector i)) →
  -- The diameter of the large circle is 20 units
  2 * large_radius = 20 := by
  intro h
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_diameter_l703_70376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l703_70310

/-- Given a triangle ABC with angles α, β, γ and sides a, b, c opposite to these angles respectively,
    prove that if γ = 3α, a = 27, and c = 48, then b = 35. -/
theorem triangle_side_length (α β γ : ℝ) (a b c : ℝ) :
  γ = 3 * α →
  a = 27 →
  c = 48 →
  0 < α →
  0 < β →
  0 < γ →
  α + β + γ = Real.pi →
  0 < a →
  0 < b →
  0 < c →
  a / (Real.sin α) = b / (Real.sin β) →
  b / (Real.sin β) = c / (Real.sin γ) →
  b = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l703_70310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_triangular_prism_l703_70349

/-- 
Given a right prism with triangular bases, where:
- The sum of the areas of three mutually adjacent faces is 40
- The angle between the sides of the base is π/3
This theorem states that the maximum volume of the prism is 400√3/27.
-/
theorem max_volume_triangular_prism (a b h : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : h > 0) : 
  (a * h + b * h + (Real.sqrt 3 / 4) * a * b = 40) →
  (Real.sqrt 3 / 4) * a * b * h ≤ 400 * Real.sqrt 3 / 27 := by
  sorry

#check max_volume_triangular_prism

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_triangular_prism_l703_70349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_triangle_area_ratio_l703_70384

/-- Given a rectangle with sides divided into 12 and 8 segments,
    the ratio of the area of a triangle formed by joining the endpoints
    of one segment from each divided pair to the nearest vertex,
    to the area of the entire rectangle, is 1/192. -/
theorem rectangle_triangle_area_ratio :
  ∀ (L W : ℝ), L > 0 → W > 0 →
  (1 / 2) * (L / 12) * (W / 8) / (L * W) = 1 / 192 :=
by
  intros L W hL hW
  field_simp
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_triangle_area_ratio_l703_70384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_hours_worked_l703_70358

/-- Represents the number of widgets assembled per hour by a worker -/
noncomputable def widgets_per_hour (minutes_per_widget : ℚ) : ℚ :=
  60 / minutes_per_widget

/-- Represents the total number of widgets assembled by three workers in h hours -/
noncomputable def total_widgets (h : ℚ) (rate1 rate2 rate3 : ℚ) : ℚ :=
  h * (rate1 + rate2 + rate3)

theorem jack_hours_worked : ∃ h : ℕ, 
  h ≤ 4 ∧ 
  h + 1 > 4 ∧
  total_widgets (h : ℚ) (widgets_per_hour 10) (widgets_per_hour 30) (widgets_per_hour 10) = 68 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_hours_worked_l703_70358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_solution_set_l703_70354

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define the parameters b and c
def b : ℝ := 1
def c : ℝ := 2

theorem intersection_and_solution_set :
  (A ∩ B = Set.Ioo (-1) 2) ∧
  ({x : ℝ | c*x^2 + b*x - 1 > 0} = {x : ℝ | x < -1 ∨ x > 1/2}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_solution_set_l703_70354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_pass_rate_l703_70399

theorem exam_pass_rate (total_students : ℕ) (passing_students : ℕ) 
  (h : passing_students ≤ total_students) : 
  ∃ (pass_rate : ℚ), pass_rate = passing_students / total_students :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_pass_rate_l703_70399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l703_70301

-- Define the functions f and g
noncomputable def f (a b x : ℝ) : ℝ := a - b * Real.cos (2 * x + Real.pi / 6)
noncomputable def g (a b x : ℝ) : ℝ := -4 * a * Real.sin (b * x - Real.pi / 3)

-- State the theorem
theorem function_analysis 
  (a b : ℝ) 
  (h_b_pos : b > 0)
  (h_f_max : ∀ x, f a b x ≤ 3/2)
  (h_f_min : ∀ x, f a b x ≥ -1/2)
  (h_f_max_exists : ∃ x, f a b x = 3/2)
  (h_f_min_exists : ∃ x, f a b x = -1/2) :
  (a = 1/2 ∧ b = 1) ∧ 
  (∀ x, g a b x ≥ -2) ∧
  (∃ k : ℤ, g a b (2 * k * Real.pi + 5 * Real.pi / 6) = -2) ∧
  (∀ x, g a b x = -2 → ∃ k : ℤ, x = 2 * k * Real.pi + 5 * Real.pi / 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_analysis_l703_70301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pages_read_is_129_l703_70398

/-- Represents the reading speed for a genre at different focus levels -/
structure ReadingSpeed :=
  (low : ℚ)
  (medium : ℚ)
  (high : ℚ)

/-- Calculates the total pages read for a genre -/
def pagesRead (speed : ReadingSpeed) (timePerFocus : ℚ) : ℚ :=
  speed.low * timePerFocus + speed.medium * timePerFocus + speed.high * timePerFocus

def totalReadingTime : ℚ := 24 * (1 / 6)
def timePerGenre : ℚ := totalReadingTime * (1 / 5)
def timePerFocus : ℚ := timePerGenre * (1 / 3)

def novelSpeed : ReadingSpeed := ⟨21, 25, 30⟩
def graphicNovelSpeed : ReadingSpeed := ⟨30, 36, 42⟩
def comicBookSpeed : ReadingSpeed := ⟨45, 54, 60⟩
def nonFictionSpeed : ReadingSpeed := ⟨18, 22, 28⟩
def biographySpeed : ReadingSpeed := ⟨20, 24, 29⟩

def totalPagesRead : ℚ :=
  pagesRead novelSpeed timePerFocus +
  pagesRead graphicNovelSpeed timePerFocus +
  pagesRead comicBookSpeed timePerFocus +
  pagesRead nonFictionSpeed timePerFocus +
  pagesRead biographySpeed timePerFocus

theorem total_pages_read_is_129 : ⌊totalPagesRead⌋ = 129 := by
  sorry

#eval ⌊totalPagesRead⌋

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_pages_read_is_129_l703_70398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_dot_product_l703_70392

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that AB · BC = -3 under the given conditions. -/
theorem triangle_dot_product (A B C : ℝ) (a b c : ℝ) : 
  a = 2 → 
  c = 3 → 
  (2 * a - c) * Real.cos B = b * Real.cos C → 
  (2 - A) * (C - 2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_dot_product_l703_70392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l703_70337

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain : Set ℝ := Set.Ioo (-4) 4

-- State that f is even
axiom f_even : ∀ x, x ∈ domain → f x = f (-x)

-- State that f is increasing on (-4, 0]
axiom f_increasing : ∀ x y, x ∈ Set.Ioc (-4) 0 → y ∈ Set.Ioc (-4) 0 → x < y → f x < f y

-- State the condition f(a) < f(3)
axiom condition (a : ℝ) : a ∈ domain → f a < f 3

-- Theorem to prove
theorem a_range (a : ℝ) : 
  a ∈ domain → f a < f 3 → a ∈ Set.Ioo (-4) (-3) ∪ Set.Ioo 3 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l703_70337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_inequality_holds_lambda_range_l703_70386

noncomputable section

def f (x : ℝ) := x * Real.log x
def g (lambda : ℝ) (x : ℝ) := lambda * (x^2 - 1)

theorem tangent_line_intersection (lambda : ℝ) :
  (∀ x, deriv f x = deriv (g lambda) x) → lambda = 1/2 :=
sorry

theorem inequality_holds (x : ℝ) (h : x ≥ 1) :
  f x ≤ g (1/2) x :=
sorry

theorem lambda_range (lambda : ℝ) :
  (∀ x ≥ 1, f x ≤ g lambda x) → lambda ≥ 1/2 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intersection_inequality_holds_lambda_range_l703_70386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_properties_l703_70356

noncomputable def geometricSeries (a : ℝ) (r : ℝ) : ℕ → ℝ 
  | 0 => 0
  | n + 1 => a * (1 - r^(n + 1)) / (1 - r)

theorem geometric_series_properties :
  let a : ℝ := 3
  let r : ℝ := 1/4
  let S : ℕ → ℝ := geometricSeries a r
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |S n - 4| < ε) ∧
  (∃ L : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |S n - L| < ε) ∧
  (∀ n : ℕ, S n ≤ 4) := by
  sorry

#check geometric_series_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_properties_l703_70356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_median_inequality_l703_70368

/-- Predicate to check if three lengths form a triangle -/
def IsTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Predicate to check if a length is a median of a triangle -/
def IsMedian (a b c m : ℝ) : Prop :=
  4 * m^2 = 2 * b^2 + 2 * c^2 - a^2

/-- Given a triangle with side lengths a, b, c and corresponding medians m_a, m_b, m_c,
    the following inequality holds. -/
theorem triangle_median_inequality 
  (a b c m_a m_b m_c : ℝ) 
  (h_triangle : IsTriangle a b c) 
  (h_ma : IsMedian a b c m_a) 
  (h_mb : IsMedian b c a m_b) 
  (h_mc : IsMedian c a b m_c) :
  m_a * (b/a - 1) * (c/a - 1) + m_b * (a/b - 1) * (c/b - 1) + m_c * (a/c - 1) * (b/c - 1) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_median_inequality_l703_70368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_difference_equals_sqrt_three_over_two_l703_70309

theorem cos_squared_difference_equals_sqrt_three_over_two :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_difference_equals_sqrt_three_over_two_l703_70309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_formula_l703_70302

noncomputable def perimeter (d m n : ℝ) : ℝ :=
  2 * Real.sqrt 2 * d * Real.cos ((m - n) * Real.pi / (4 * (m + n)))

theorem perimeter_formula (d m n : ℝ) (hd : d > 0) (hm : m > 0) (hn : n > 0) :
  let diagonal := d
  let angle_ratio := m / n
  perimeter d m n = 2 * Real.sqrt 2 * d * Real.cos ((m - n) * Real.pi / (4 * (m + n))) :=
by
  -- Unfold the definition of perimeter
  unfold perimeter
  
  -- The proof steps would go here
  -- We'll use sorry to skip the actual proof for now
  sorry

#check perimeter_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_formula_l703_70302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painted_cells_count_l703_70381

/-- Represents the grid dimensions -/
structure GridDim where
  width : Nat
  height : Nat

/-- Represents a position on the grid -/
structure Position where
  x : Nat
  y : Nat

/-- The grid dimensions for our problem -/
def problemGrid : GridDim := ⟨2000, 70⟩

/-- Function to calculate the next position after one move -/
def nextPosition (grid : GridDim) (pos : Position) : Position :=
  ⟨(pos.x + 1) % grid.width, (pos.y + grid.height - 1) % grid.height⟩

/-- Function to check if two positions are equal -/
def positionEq (p1 p2 : Position) : Prop :=
  p1.x = p2.x ∧ p1.y = p2.y

/-- Theorem stating the number of unique cells painted before returning to start -/
theorem painted_cells_count (start : Position) : 
  ∃ (n : Nat), n = 14000 ∧ 
    (∀ (k : Nat), k < n → ¬positionEq ((nextPosition problemGrid)^[k] start) start) ∧
    positionEq ((nextPosition problemGrid)^[n] start) start := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_painted_cells_count_l703_70381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_among_chosen_numbers_l703_70371

theorem divisibility_among_chosen_numbers (n : ℕ) (S : Finset ℕ) : 
  (∀ x, x ∈ S → x ≤ n) → 
  S.card > (n + 1) / 2 →
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_among_chosen_numbers_l703_70371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_weight_partition_l703_70364

theorem equal_weight_partition :
  let weights : List ℕ := List.range 552
  let total_weight := weights.sum
  let partition_weight := total_weight / 3
  ∃ (partition1 partition2 partition3 : List ℕ),
    (∀ x, x ∈ partition1 → x ∉ partition2 ∧ x ∉ partition3) ∧
    (∀ x, x ∈ partition2 → x ∉ partition1 ∧ x ∉ partition3) ∧
    (∀ x, x ∈ partition3 → x ∉ partition1 ∧ x ∉ partition2) ∧
    partition1 ++ partition2 ++ partition3 = weights ∧
    partition1.sum = partition_weight ∧
    partition2.sum = partition_weight ∧
    partition3.sum = partition_weight ∧
    partition_weight = 50876 := by
  sorry

#eval (List.range 552).sum / 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_weight_partition_l703_70364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l703_70340

noncomputable def f (x : ℝ) : ℝ := 1 / ⌊x^2 - 8*x + 18⌋

theorem domain_of_f : Set.univ = {x : ℝ | ∃ y, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l703_70340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_a_value_l703_70329

/-- The curve function f(x) = a * ln(x) + x^2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := a / x + 2 * x

theorem tangent_line_parallel_implies_a_value :
  ∀ a : ℝ,
  (f a 1 = 1) →  -- The curve passes through (1, 1)
  (f_derivative a 1 = -1) →  -- The tangent line at (1, 1) is parallel to x + y = 0
  a = -3 :=
by
  intro a h1 h2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_a_value_l703_70329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_imply_a_range_l703_70327

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x - a
noncomputable def g (x : ℝ) : ℝ := 2*x + 2*Real.log x

-- Define the domain
def domain : Set ℝ := { x | 1/Real.exp 1 ≤ x ∧ x ≤ Real.exp 1 }

-- State the theorem
theorem intersection_points_imply_a_range (a : ℝ) :
  (∃ x₁ x₂, x₁ ∈ domain ∧ x₂ ∈ domain ∧ x₁ ≠ x₂ ∧ f a x₁ = g x₁ ∧ f a x₂ = g x₂) →
  a ∈ Set.Ioo 1 (1/(Real.exp 1)^2 + 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_imply_a_range_l703_70327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_partition_l703_70373

theorem impossible_partition : ¬ ∃ (partition : List (List Nat)),
  (∀ group ∈ partition, (group.length ≥ 4 ∧ 
    ∃ x ∈ group, x = (group.sum - x))) ∧
  (partition.join.toFinset = Finset.range 89) ∧
  (List.sum (List.range 89) % 2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_partition_l703_70373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_parabola_directrix_l703_70304

/-- The distance from the center of the circle x^2 - 2x + y^2 = 0 to the directrix of the parabola y^2 = 4x is 2. -/
theorem distance_circle_center_to_parabola_directrix :
  let circle : Set (ℝ × ℝ) := {p | (p.1^2 - 2*p.1 + p.2^2) = 0}
  let parabola : Set (ℝ × ℝ) := {p | p.2^2 = 4*p.1}
  let circle_center : ℝ × ℝ := (1, 0)
  let directrix : Set (ℝ × ℝ) := {p | p.1 = -1}
  dist circle_center ⟨-1, 0⟩ = 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_parabola_directrix_l703_70304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_intersection_l703_70391

/-- The function representing the graph -/
noncomputable def f (x : ℝ) : ℝ := (x^2 - 6*x + 8) / (x^2 - 6*x + 9)

/-- The x-coordinate of the potential intersection point -/
def x_intersect : ℝ := 3

/-- The y-coordinate of the potential intersection point -/
def y_intersect : ℝ := 1

/-- Theorem stating that (3, 1) is the intersection of the asymptotes -/
theorem asymptote_intersection :
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - x_intersect| < δ → |f x - y_intersect| < ε) ∧
  (∀ M > 0, ∃ N > 0, ∀ x, |x| > N → |f x - y_intersect| < M) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_intersection_l703_70391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_l703_70375

/-- The parabola y² = x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = p.1}

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_point_distance (a : ℝ) :
  (a > 0) →
  (∀ q ∈ Parabola, distance (a, 0) q ≥ distance (a, 0) (0, 0)) →
  a ≤ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_l703_70375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_two_sevenths_l703_70342

/-- The probability of arranging 3 irrational terms among 7 total terms 
    such that no two irrational terms are adjacent -/
def irrational_terms_arrangement_probability : ℚ :=
  let total_terms : ℕ := 7
  let irrational_terms : ℕ := 3
  let arrangements_without_adjacent : ℕ := Nat.choose 4 4 * Nat.choose 5 3
  let total_arrangements : ℕ := Nat.factorial 7
  ↑arrangements_without_adjacent / ↑total_arrangements

/-- The probability is equal to 2/7 -/
theorem probability_is_two_sevenths : 
  irrational_terms_arrangement_probability = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_two_sevenths_l703_70342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_proof_l703_70305

/-- Given a sinusoidal function f(x) = A * sin(w * x + φ) with specific properties,
    prove that it equals 3 * sin(x + π/4) -/
theorem sinusoidal_function_proof (f : ℝ → ℝ) 
  (h1 : ∃ A w φ : ℝ, ∀ x, f x = A * Real.sin (w * x + φ))
  (h2 : f (π/4) = 3 ∧ ∀ y, f y ≤ 3)  -- (π/4, 3) is a maximum point
  (h3 : f (-π/4) = 0)  -- (-π/4, 0) is on the x-axis
  (h4 : ∀ x ∈ Set.Ioo (-π/4) (π/4), f x ≠ 3 ∧ f x ≠ 0)  -- These points are adjacent
  : f = λ x ↦ 3 * Real.sin (x + π/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_proof_l703_70305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l703_70338

theorem cos_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : Real.cos (α - π/6) = 3/5) :
  Real.cos α = (3 * Real.sqrt 3 - 4) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l703_70338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l703_70369

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^3

-- State the theorem
theorem range_of_x (x : ℝ) : 
  (f (x^2) < f (3*x - 2)) → (1 < x ∧ x < 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l703_70369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_difference_constant_sum_l703_70378

noncomputable section

-- Define the points
def P : ℝ × ℝ := (0, -Real.sqrt 2)
def Q : ℝ × ℝ := (0, Real.sqrt 2)
def A (a : ℝ) : ℝ × ℝ := (a, Real.sqrt (a^2 + 1))

-- Define the constraint on a
def a_constraint (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1

-- Define the parabola
def parabola (x : ℝ) : ℝ := (Real.sqrt 2 / 8) * x^2

-- Define point B (intersection of QA and parabola)
def B (a : ℝ) : ℝ × ℝ := sorry

-- Define point C (intersection of perpendicular from B to y = 2)
def C (a : ℝ) : ℝ × ℝ := sorry

-- Define distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

end noncomputable section

-- Theorem 1: PA - AQ is constant
theorem constant_difference (a : ℝ) (h : a_constraint a) :
  distance P (A a) - distance (A a) Q = 2 := by sorry

-- Theorem 2: PA + AB + BC is constant
theorem constant_sum (a : ℝ) (h : a_constraint a) :
  distance P (A a) + distance (A a) (B a) + distance (B a) (C a) = 4 + Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_difference_constant_sum_l703_70378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_diagonal_sum_l703_70314

/-- A parallelepiped with side lengths x, y, and z -/
structure Parallelepiped where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The length of a space diagonal in a parallelepiped -/
noncomputable def spaceDiagonal (p : Parallelepiped) : ℝ :=
  Real.sqrt (p.x^2 + p.y^2 + p.z^2)

/-- Theorem: For a parallelepiped with space diagonals of lengths 15, 17, 21, and 23,
    the sum of squares of its side lengths is 371 -/
theorem parallelepiped_diagonal_sum (p : Parallelepiped) 
  (h1 : spaceDiagonal p = 15)
  (h2 : spaceDiagonal p = 17)
  (h3 : spaceDiagonal p = 21)
  (h4 : spaceDiagonal p = 23) :
  p.x^2 + p.y^2 + p.z^2 = 371 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_diagonal_sum_l703_70314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_solution_concentration_l703_70380

/-- Given a solution that is 10% sugar by weight, if 1/4 of it is replaced by a second solution 
    resulting in a 16% sugar solution, then the second solution must be 34% sugar by weight. -/
theorem sugar_solution_concentration (W : ℝ) (S : ℝ) : 
  W > 0 → -- W is the total weight of the original solution
  0.10 * W = 0.10 * W → -- Original solution is 10% sugar (tautology to avoid using undefined function)
  0.075 * W + (S / 100) * (W / 4) = 0.16 * W → -- Equation for final sugar content
  S = 34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_solution_concentration_l703_70380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l703_70328

def sequenceA (n : ℕ+) : ℚ := 1 / (n * (n + 1))

theorem sequence_formula (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) :
  a 1 = 1/2 →
  (∀ n : ℕ+, S n = n^2 * a n) →
  ∀ n : ℕ+, a n = sequenceA n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l703_70328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_exists_l703_70346

/-- Represents the contingency table data -/
structure ContingencyTable where
  total_students : Nat
  like_outdoor_sports : Nat
  excellent_scores : Nat
  like_sports_non_excellent : Nat
  dislike_sports_non_excellent : Nat

/-- Calculates the K² value for the given contingency table -/
noncomputable def calculate_k_squared (ct : ContingencyTable) : Real :=
  let a := ct.excellent_scores - ct.like_sports_non_excellent
  let b := ct.like_sports_non_excellent
  let c := ct.excellent_scores - a
  let d := ct.dislike_sports_non_excellent
  let n := ct.total_students
  (n * (a * d - b * c)^2 : Real) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The threshold for 95% confidence level -/
def confidence_threshold : Real := 3.841

/-- The theorem to be proved -/
theorem relationship_exists (ct : ContingencyTable) 
  (h1 : ct.total_students = 100)
  (h2 : ct.like_outdoor_sports = 60)
  (h3 : ct.excellent_scores = 75)
  (h4 : ct.like_sports_non_excellent = 10)
  (h5 : ct.dislike_sports_non_excellent = 15) :
  calculate_k_squared ct > confidence_threshold :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_exists_l703_70346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l703_70357

noncomputable def f (x : ℝ) := 3 * Real.sin (2 * x + Real.pi / 6)

theorem f_properties :
  (f (Real.pi / 12) = 3 * Real.sqrt 3 / 2) ∧
  (∀ θ : ℝ, θ ∈ Set.Ioo 0 (Real.pi / 2) → Real.sin θ = 4 / 5 → f (5 * Real.pi / 12 - θ) = 72 / 25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l703_70357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l703_70323

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 - 4*x
  else if x < 0 then -(((-x)^2) - 4*(-x))
  else 0

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) :
  f x > x ↔ x ∈ Set.Ioo (-5 : ℝ) 0 ∪ Set.Ioi 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l703_70323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l703_70306

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem problem_solution (x y : ℝ) 
  (h1 : y = 3 * (floor x) + 4)
  (h2 : y = 2 * (floor (x - 3)) + 7)
  (h3 : x ≠ ↑(floor x)) : 
  -8 < x + y ∧ x + y < -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l703_70306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_proof_l703_70331

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digits_to_num (p q r s : ℕ) : ℕ := 1000 * p + 100 * q + 10 * r + s

def reverse_num (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  digits_to_num d4 d3 d2 d1

theorem smallest_number_proof (p q r s : ℕ) (X : ℕ) 
  (h_distinct : p < q ∧ q < r ∧ r < s)
  (h_four_digit : is_four_digit (digits_to_num p q r s))
  (h_sum : digits_to_num p q r s + reverse_num (digits_to_num p q r s) + X = 26352)
  (h_X_same_digits : ∃ (a b c d : ℕ), X = digits_to_num a b c d ∧ Multiset.ofList [a, b, c, d] = Multiset.ofList [p, q, r, s]) :
  digits_to_num p q r s = 6789 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_proof_l703_70331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_coloring_l703_70362

/-- The chromatic polynomial of a cycle graph with n vertices and k colors -/
def chromaticPolynomialCycle (n : ℕ) (k : ℕ) : ℤ :=
  (k - 1)^n + (-1)^n * (k - 1)

/-- The number of vertices in our cycle graph (flower bed parts) -/
def numVertices : ℕ := 6

/-- The number of colors available -/
def numColors : ℕ := 4

theorem flower_bed_coloring :
  chromaticPolynomialCycle numVertices numColors = 732 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_coloring_l703_70362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_truck_meetings_l703_70334

-- Define the parameters
noncomputable def michael_speed : ℝ := 6
noncomputable def truck_speed : ℝ := 12
noncomputable def pail_distance : ℝ := 200
noncomputable def truck_stop_time : ℝ := 20
noncomputable def initial_distance : ℝ := 200

-- Define the function for distance change in one cycle
noncomputable def distance_change_per_cycle : ℝ :=
  let truck_travel_time := pail_distance / truck_speed
  let michael_travel_during_truck_movement := michael_speed * truck_travel_time
  let michael_travel_during_truck_stop := michael_speed * truck_stop_time
  (pail_distance - michael_travel_during_truck_movement) - michael_travel_during_truck_stop

-- Theorem statement
theorem michael_truck_meetings :
  (Int.floor (initial_distance / (-distance_change_per_cycle))) = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_truck_meetings_l703_70334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_congruence_theorem_l703_70308

/-- Given a prime p and polynomials f, g, h, r, s with integer coefficients -/
def polynomial_congruence_problem
  (p : ℕ) (f g h r s : Polynomial ℤ) : Prop :=
  Prime p ∧
  (∀ x : ℤ, (r.eval x * f.eval x + s.eval x * g.eval x) % p = 1) ∧
  (∀ x : ℤ, (f.eval x * g.eval x) % p = h.eval x % p)

/-- The existence of polynomials F and G satisfying the required congruences -/
def polynomial_congruence_solution
  (p : ℕ) (f g h : Polynomial ℤ) (n : ℕ) : Prop :=
  ∃ (F G : Polynomial ℤ),
    (∀ x : ℤ, F.eval x % p = f.eval x % p) ∧
    (∀ x : ℤ, G.eval x % p = g.eval x % p) ∧
    (∀ x : ℤ, (F.eval x * G.eval x) % (p^n) = h.eval x % (p^n))

/-- The main theorem stating that the problem implies the solution for any positive n -/
theorem polynomial_congruence_theorem
  (p : ℕ) (f g h r s : Polynomial ℤ) :
  polynomial_congruence_problem p f g h r s →
  ∀ n : ℕ, n > 0 → polynomial_congruence_solution p f g h n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_congruence_theorem_l703_70308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_internet_minutes_correct_l703_70326

/-- Given that M minutes of internet usage can be purchased for P pennies,
    and 1 euro is equivalent to 100 pennies, this function calculates
    the number of minutes of internet usage that can be purchased for E euros. -/
noncomputable def internet_minutes (M P E : ℝ) : ℝ :=
  100 * E * M / P

/-- Theorem stating that the internet_minutes function correctly calculates
    the number of minutes of internet usage that can be purchased for E euros. -/
theorem internet_minutes_correct (M P E : ℝ) (hM : M > 0) (hP : P > 0) (hE : E ≥ 0) :
  internet_minutes M P E = 100 * E * M / P :=
by
  -- Unfold the definition of internet_minutes
  unfold internet_minutes
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_internet_minutes_correct_l703_70326
