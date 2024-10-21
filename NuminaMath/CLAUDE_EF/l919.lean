import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_result_l919_91949

-- Define the K² statistic
def K_squared : ℝ := 5

-- Define the critical values and their corresponding probabilities
def critical_value_95 : ℝ := 3.84
def critical_value_99 : ℝ := 6.635

-- Define the probabilities
def P_95 : ℝ := 0.05
def P_99 : ℝ := 0.01

-- Define the confidence level
def confidence_level : ℝ := 1 - P_95

-- Theorem statement
theorem independence_test_result :
  K_squared > critical_value_95 →
  confidence_level = 0.95 →
  True := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_result_l919_91949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l919_91935

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the line
def line (k m x y : ℝ) : Prop := y = k*x + m

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem ellipse_and_line_intersection 
  (k m : ℝ) (hk : k ≠ 0) :
  (∀ x y, ellipse x y → 
    (x = 1 ∧ y = 3/2) → 
    (∃ e_hyp, ∀ x y, hyperbola x y → e_hyp^2 = 4/3) →
    (∃ x1 y1 x2 y2, 
      x1 ≠ x2 ∧ 
      ellipse x1 y1 ∧ 
      ellipse x2 y2 ∧ 
      line k m x1 y1 ∧ 
      line k m x2 y2 ∧
      distance x1 y1 (1/5) 0 = distance x2 y2 (1/5) 0)) →
  (k < -Real.sqrt 7 / 7 ∨ k > Real.sqrt 7 / 7) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_intersection_l919_91935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_when_a_zero_f_defined_iff_a_gt_neg_one_l919_91977

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 + 2*a*x + a^2 + a + 1)

-- Statement 1: f is even when a = 0
theorem f_even_when_a_zero : 
  ∀ x : ℝ, f 0 x = f 0 (-x) := by sorry

-- Statement 2: f is defined for all real x iff a > -1
theorem f_defined_iff_a_gt_neg_one (a : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ a > -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_when_a_zero_f_defined_iff_a_gt_neg_one_l919_91977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_horizontal_compression_l919_91924

-- Define the original function g
noncomputable def g : ℝ → ℝ := λ x => 
  if -2 ≤ x ∧ x ≤ 3 then x^2 - 2*x - 3
  else if 3 < x ∧ x ≤ 5 then 2*x - 9
  else 0

-- Define the transformed function
noncomputable def g_transformed : ℝ → ℝ := λ x => g (2*x)

-- Theorem statement
theorem g_horizontal_compression :
  ∀ x : ℝ, g_transformed x = g (2*x) :=
by
  intro x
  rfl  -- reflexivity proves the equality


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_horizontal_compression_l919_91924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_catch_up_time_l919_91926

/-- Represents the pursuit of a cargo ship by a pirate ship -/
structure ShipPursuit where
  initial_distance : ℚ
  initial_pirate_speed : ℚ
  initial_cargo_speed : ℚ
  initial_pursuit_time : ℚ
  post_storm_pirate_distance : ℚ
  post_storm_cargo_distance : ℚ

/-- Calculates the total time for the pirate ship to catch the cargo ship -/
def catch_up_time (pursuit : ShipPursuit) : ℚ :=
  pursuit.initial_pursuit_time +
  (pursuit.initial_distance -
   (pursuit.initial_pursuit_time * (pursuit.initial_pirate_speed - pursuit.initial_cargo_speed))) /
  ((pursuit.post_storm_pirate_distance / pursuit.post_storm_cargo_distance) * pursuit.initial_cargo_speed - pursuit.initial_cargo_speed)

/-- Theorem stating that the total catch-up time is 9 hours -/
theorem pirate_catch_up_time :
  let pursuit : ShipPursuit := {
    initial_distance := 15,
    initial_pirate_speed := 12,
    initial_cargo_speed := 9,
    initial_pursuit_time := 3,
    post_storm_pirate_distance := 10,
    post_storm_cargo_distance := 9
  }
  catch_up_time pursuit = 9 := by
  sorry

#eval catch_up_time {
  initial_distance := 15,
  initial_pirate_speed := 12,
  initial_cargo_speed := 9,
  initial_pursuit_time := 3,
  post_storm_pirate_distance := 10,
  post_storm_cargo_distance := 9
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pirate_catch_up_time_l919_91926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sum_distances_l919_91938

-- Define the triangle type
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the function for sum of distances to sidelines
noncomputable def sum_distances (T : Triangle) (P : ℝ × ℝ) : ℝ :=
  sorry

-- Define the property of having common incircle and circumcircle
def common_circles (T1 T2 : Triangle) : Prop :=
  sorry

theorem equal_sum_distances (ABC A'B'C' : Triangle) (P : ℝ × ℝ) :
  common_circles ABC A'B'C' →
  sum_distances ABC P = sum_distances A'B'C' P := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sum_distances_l919_91938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_picnic_attendance_theorem_l919_91934

/-- Represents the percentage of employees who attended the picnic -/
noncomputable def picnic_attendance (total_employees : ℕ) (men_percentage : ℚ) (men_attendance : ℚ) (women_attendance : ℚ) : ℚ :=
  let men := men_percentage * total_employees
  let women := (1 - men_percentage) * total_employees
  let men_attended := men_attendance * men
  let women_attended := women_attendance * women
  (men_attended + women_attended) / total_employees * 100

/-- Proves that given the conditions, 31% of all employees went to the picnic -/
theorem picnic_attendance_theorem :
  ∀ (total_employees : ℕ), total_employees > 0 →
  picnic_attendance total_employees (45/100) (20/100) (40/100) = 31 :=
by
  intro total_employees h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_picnic_attendance_theorem_l919_91934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_group_average_weight_l919_91943

def initial_group_size : ℕ := 30
def additional_group_size : ℕ := 30
def average_weight_increase : ℝ := 10

theorem additional_group_average_weight 
  (some : ℝ) -- Average weight of whole group after addition
  (h : some > 0) -- Assumption that the average weight is positive
  : ∃ (initial_avg : ℝ),
    initial_avg > 0 ∧
    some = initial_avg + average_weight_increase ∧
    (initial_avg * (initial_group_size : ℝ) + 
     (some + 10) * (additional_group_size : ℝ)) / 
     ((initial_group_size + additional_group_size) : ℝ) = some :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_group_average_weight_l919_91943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_negative_one_l919_91910

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 : ℝ)^x + a * (2 : ℝ)^(-x)

-- State the theorem
theorem odd_function_implies_a_equals_negative_one :
  (∀ x : ℝ, f a x = -f a (-x)) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_equals_negative_one_l919_91910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_condition_for_obtuse_angle_l919_91906

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (2, 3)
def n : ℝ × ℝ := (3, -1)

/-- Vector m as a function of lambda -/
def m (lambda : ℝ) : ℝ × ℝ := (lambda * a.1 + b.1, lambda * a.2 + b.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The angle between two vectors is obtuse if their dot product is negative -/
def is_obtuse_angle (v w : ℝ × ℝ) : Prop := dot_product v w < 0

/-- Main theorem: lambda < -4 is a sufficient but not necessary condition for obtuse angle -/
theorem lambda_condition_for_obtuse_angle :
  (∀ lambda, lambda < -4 → is_obtuse_angle (m lambda) n) ∧
  ¬(∀ lambda, is_obtuse_angle (m lambda) n → lambda < -4) := by
  sorry

#check lambda_condition_for_obtuse_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_condition_for_obtuse_angle_l919_91906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosecant_150_degrees_csc_150_degrees_l919_91998

theorem cosecant_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

theorem csc_150_degrees : 1 / Real.sin (150 * π / 180) = 2 := by
  have h : Real.sin (150 * π / 180) = 1 / 2 := cosecant_150_degrees
  rw [h]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosecant_150_degrees_csc_150_degrees_l919_91998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l919_91933

noncomputable def sine_function (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem sine_function_properties (A ω φ : ℝ) 
  (h1 : A > 0) (h2 : ω > 0) (h3 : φ > -π/2 ∧ φ < π/2)
  (h4 : sine_function A ω φ (π/12) = 0)
  (h5 : sine_function A ω φ (π/3) = 5) :
  ∃ (k : ℤ),
    (∀ x, sine_function A ω φ x = 5 * Real.sin (2*x - π/6)) ∧
    (5 = A) ∧
    (∀ x, sine_function A ω φ x ≤ 5) ∧
    (∀ x, sine_function A ω φ x = 5 → ∃ k : ℤ, x = k*π + π/3) ∧
    (∀ x, sine_function A ω φ x ≤ 0 ↔ 
      ∃ k : ℤ, k*π - 5*π/12 ≤ x ∧ x ≤ k*π + π/12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l919_91933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_inequality_l919_91994

theorem root_difference_inequality (a : ℝ) : 
  (∀ m : ℝ, m ∈ Set.Icc (-1 : ℝ) 1 → 
    ∃ x₁ x₂ : ℝ, (x₁^2 - m*x₁ - 2 = 0) ∧ (x₂^2 - m*x₂ - 2 = 0) ∧ 
    (a^2 - 5*a - 3 ≥ |x₁ - x₂|)) ↔ 
  (a ≥ 6 ∨ a ≤ -1) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_difference_inequality_l919_91994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angles_trigonometric_relations_l919_91912

open Real

theorem acute_angles_trigonometric_relations (α β : ℝ) 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_relation : sin β / sin α = cos (α + β)) :
  (∃ (C : ℝ), tan (α + β) * (cos α / sin α) = C) ∧
  (∃ (M : ℝ), ∀ β', 0 < β' ∧ β' < π / 2 → tan β' ≤ M) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angles_trigonometric_relations_l919_91912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rounds_for_property_l919_91900

/-- A tournament schedule for 2m players over n rounds -/
structure TournamentSchedule (m n : ℕ) :=
  (schedule : Fin n → Fin (2*m) → Fin (2*m))
  (valid : ∀ (r : Fin n) (p : Fin (2*m)), schedule r p ≠ p)
  (complete : ∀ (p q : Fin (2*m)), p ≠ q → ∃ (r : Fin n), schedule r p = q ∨ schedule r q = p)

/-- The property that any four players have either not competed or competed at least twice -/
def SatisfiesProperty (m n : ℕ) (s : TournamentSchedule m n) : Prop :=
  ∀ (p1 p2 p3 p4 : Fin (2*m)),
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 →
    (∀ (r : Fin n), s.schedule r p1 ≠ p2 ∧ s.schedule r p1 ≠ p3 ∧ s.schedule r p1 ≠ p4 ∧
                    s.schedule r p2 ≠ p3 ∧ s.schedule r p2 ≠ p4 ∧ s.schedule r p3 ≠ p4) ∨
    (∃ (r1 r2 : Fin n), r1 ≠ r2 ∧
      ((s.schedule r1 p1 = p2 ∨ s.schedule r1 p1 = p3 ∨ s.schedule r1 p1 = p4 ∨
        s.schedule r1 p2 = p3 ∨ s.schedule r1 p2 = p4 ∨ s.schedule r1 p3 = p4) ∧
       (s.schedule r2 p1 = p2 ∨ s.schedule r2 p1 = p3 ∨ s.schedule r2 p1 = p4 ∨
        s.schedule r2 p2 = p3 ∨ s.schedule r2 p2 = p4 ∨ s.schedule r2 p3 = p4)))

theorem smallest_rounds_for_property (m : ℕ) (h : m ≥ 3) :
  (∃ (n : ℕ) (s : TournamentSchedule m n), SatisfiesProperty m n s) ∧
  (∀ (k : ℕ), k < m - 1 → ¬∃ (s : TournamentSchedule m k), SatisfiesProperty m k s) := by
  sorry

#check smallest_rounds_for_property

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rounds_for_property_l919_91900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relationship_triangle_perimeter_l919_91915

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if sin C * sin(A-B) = sin B * sin(C-A), then 2a² = b² + c² -/
theorem triangle_side_relationship (a b c A B C : ℝ) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : 0 < A ∧ A < π)
  (h3 : 0 < B ∧ B < π)
  (h4 : 0 < C ∧ C < π)
  (h5 : A + B + C = π)
  (h6 : a / Real.sin A = b / Real.sin B)
  (h7 : b / Real.sin B = c / Real.sin C)
  (h8 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) :
  2 * a^2 = b^2 + c^2 := by
  sorry

/-- If a = 5 and cos A = 25/31 in the above triangle, its perimeter is 14 -/
theorem triangle_perimeter (a b c A B C : ℝ) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : 0 < A ∧ A < π)
  (h3 : 0 < B ∧ B < π)
  (h4 : 0 < C ∧ C < π)
  (h5 : A + B + C = π)
  (h6 : a / Real.sin A = b / Real.sin B)
  (h7 : b / Real.sin B = c / Real.sin C)
  (h8 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A))
  (h9 : a = 5)
  (h10 : Real.cos A = 25/31) :
  a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relationship_triangle_perimeter_l919_91915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l919_91964

theorem log_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hna : a ≠ 1) (hnb : b ≠ 1) (hlog : Real.log b / Real.log a > 1) :
  (b - 1) * (b - a) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l919_91964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_implies_equal_areas_and_sides_equal_areas_and_sides_necessary_for_congruence_l919_91986

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)  -- side lengths
  (angleA angleB angleC : ℝ)  -- angles
  (valid : a > 0 ∧ b > 0 ∧ c > 0 ∧ angleA > 0 ∧ angleB > 0 ∧ angleC > 0 ∧ angleA + angleB + angleC = π)

-- Define congruence of triangles
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c ∧
  t1.angleA = t2.angleA ∧ t1.angleB = t2.angleB ∧ t1.angleC = t2.angleC

-- Define the area of a triangle
noncomputable def area (t : Triangle) : ℝ :=
  (1/2) * t.a * t.b * Real.sin t.angleC

-- Theorem statement
theorem congruent_implies_equal_areas_and_sides (t1 t2 : Triangle) :
  congruent t1 t2 → area t1 = area t2 ∧ t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c := by
  sorry

-- Theorem for the problem statement
theorem equal_areas_and_sides_necessary_for_congruence :
  ∀ t1 t2 : Triangle, congruent t1 t2 → area t1 = area t2 ∧ t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_congruent_implies_equal_areas_and_sides_equal_areas_and_sides_necessary_for_congruence_l919_91986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quarterly_interest_payment_l919_91968

/-- Calculate quarterly interest payment for a debenture investment -/
theorem quarterly_interest_payment
  (principal : ℝ)
  (annual_rate : ℝ)
  (period_months : ℕ)
  (h1 : principal = 10000)
  (h2 : annual_rate = 0.095)
  (h3 : period_months = 18) :
  (principal * annual_rate * (period_months / 12 : ℝ)) / (period_months / 3 : ℝ) = 237.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quarterly_interest_payment_l919_91968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_length_l919_91913

/-- The length of a train passing an observer --/
noncomputable def train_length (v_observer v_train : ℝ) (t : ℝ) : ℝ :=
  (v_observer + v_train) * t * (1000 / 3600)

/-- Theorem stating the length of the goods train --/
theorem goods_train_length :
  let v_man := (50 : ℝ) -- km/h
  let v_goods := (62 : ℝ) -- km/h
  let t := (9 : ℝ) -- seconds
  ∃ ε > 0, |train_length v_man v_goods t - 280| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_length_l919_91913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l919_91996

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := 2^(x - b)

-- State the theorem
theorem range_of_f (b : ℝ) :
  (∃ (x : ℝ), 2 ≤ x ∧ x ≤ 4 ∧ f b x = 1) →
  (∀ (y : ℝ), (∃ (x : ℝ), 2 ≤ x ∧ x ≤ 4 ∧ f b x = y) ↔ 1/2 ≤ y ∧ y ≤ 2) :=
by
  sorry

#check range_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l919_91996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_relationship_domain_f_l919_91999

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the domain of y = f(2x+1)
def domain_f_2x_plus_1 : Set ℝ := Set.Icc 1 2

-- Theorem stating the relationship between the domains
theorem domain_relationship :
  (Set.Icc 3 5 : Set ℝ) = {x | ∃ y ∈ domain_f_2x_plus_1, 2*y + 1 = x} := by
  sorry

-- Corollary: If the domain of f(2x+1) is [1,2], then the domain of f(x) is [3,5]
theorem domain_f :
  (∀ x, x ∈ domain_f_2x_plus_1 ↔ f (2*x + 1) ≠ 0) →
  (∀ x, x ∈ Set.Icc 3 5 ↔ f x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_relationship_domain_f_l919_91999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l919_91972

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem function_properties (ω φ : ℝ) 
  (h_ω_pos : ω > 0)
  (h_φ_bound : |φ| < π / 2)
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x)
  (h_symmetry : ∀ x, f ω φ (π / 3 - x) = f ω φ (π / 3 + x)) :
  ω = 2 ∧ 
  φ = π / 6 ∧
  (∀ x, f ω φ (5 * π / 12 - x) = f ω φ (5 * π / 12 + x)) ∧
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π / 6 → f ω φ x < f ω φ y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l919_91972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_in_expansion_l919_91960

noncomputable def binomial_expansion (x : ℝ) (n : ℕ+) : ℕ → ℝ :=
  fun r => (-1)^r * 3^(18 - 3*r) * (Nat.choose n.val r) * x^(n.val - 3*r/2)

theorem constant_term_in_expansion (n : ℕ+) :
  (∃ k : ℕ, binomial_expansion 1 n k = 36) →
  (∃ r : ℕ, binomial_expansion 0 n r = 84) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_in_expansion_l919_91960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_everyone_knows_same_number_l919_91920

/-- Represents a gathering of people with specific acquaintance rules -/
structure Gathering where
  n : ℕ                -- number of people
  knows : Fin n → Fin n → Bool  -- relation representing who knows whom

/-- The acquaintance rules for the gathering -/
class GatheringRules (G : Gathering) where
  mutual_acquaintances : 
    ∀ (x y : Fin G.n), ¬G.knows x y → ∃! (a b : Fin G.n), 
      a ≠ b ∧ G.knows x a = true ∧ G.knows y a = true ∧ G.knows x b = true ∧ G.knows y b = true
  no_mutual_friends : 
    ∀ (x y z : Fin G.n), G.knows x y = true → ¬(G.knows x z = true ∧ G.knows y z = true)

/-- The number of people each person knows -/
def num_known (G : Gathering) (x : Fin G.n) : ℕ := 
  (Finset.filter (fun y => G.knows x y) (Finset.univ : Finset (Fin G.n))).card

/-- The main theorem: everyone knows the same number of people -/
theorem everyone_knows_same_number (G : Gathering) [GatheringRules G] : 
  ∀ (x y : Fin G.n), num_known G x = num_known G y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_everyone_knows_same_number_l919_91920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curvature_cubic_plus_one_curvature_cubic_plus_two_range_l919_91981

-- Define the curvature function φ
noncomputable def φ (f : ℝ → ℝ) (x₁ x₂ : ℝ) : ℝ :=
  let y₁ := f x₁
  let y₂ := f x₂
  let k₁ := (deriv f) x₁
  let k₂ := (deriv f) x₂
  |k₁ - k₂| / Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Part 1
theorem curvature_cubic_plus_one :
  φ (λ x => x^3 + 1) 1 2 = (9 * Real.sqrt 2) / 10 := by sorry

-- Part 2
theorem curvature_cubic_plus_two_range (x₁ x₂ : ℝ) (h : x₁ * x₂ = 1) :
  0 < φ (λ x => x^3 + 2) x₁ x₂ ∧ φ (λ x => x^3 + 2) x₁ x₂ < (3 * Real.sqrt 10) / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curvature_cubic_plus_one_curvature_cubic_plus_two_range_l919_91981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_solution_l919_91987

/-- Represents the theater problem with given conditions --/
def TheaterProblem (seats : ℕ) (cost_per_sqft : ℝ) (construction_factor : ℝ) 
                   (partner_share : ℝ) (tom_spent : ℝ) (sqft_per_seat : ℝ) : Prop :=
  let total_cost := seats * cost_per_sqft * construction_factor * sqft_per_seat
  tom_spent = (1 - partner_share) * total_cost

/-- Theorem stating the solution to the theater problem --/
theorem theater_solution :
  ∃ (sqft_per_seat : ℝ),
    TheaterProblem 500 5 3 0.4 54000 sqft_per_seat ∧ 
    sqft_per_seat = 12 := by
  sorry

#check theater_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theater_solution_l919_91987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_inequality_l919_91982

theorem triangle_sine_sum_inequality (A B C : ℝ) :
  A > 0 → B > 0 → C > 0 → A + B + C = Real.pi →
  Real.sin A + Real.sin B + Real.sin C ≤ 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_inequality_l919_91982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l919_91916

noncomputable def line_l₁ (t : ℝ) : ℝ × ℝ := (t, Real.sqrt 3 * t)

def circle_C₁ (x y : ℝ) : Prop :=
  (x - Real.sqrt 3)^2 + (y - 2)^2 = 1

def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, p = line_l₁ t ∧ circle_C₁ p.1 p.2}

theorem area_of_triangle : ∃ (area : ℝ), area = Real.sqrt 3 / 4 ∧ 
  ∃ M N, M ∈ intersection_points ∧ N ∈ intersection_points ∧ M ≠ N ∧
    let center : ℝ × ℝ := (Real.sqrt 3, 2)
    area = (1/2) * abs (center.1 * (M.2 - N.2) + M.1 * (N.2 - center.2) + N.1 * (center.2 - M.2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_l919_91916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_problem_l919_91944

theorem permutation_problem (m : ℕ) : (Nat.factorial 6 / Nat.factorial (6 - m) = 6 * 5 * 4) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_problem_l919_91944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_asymptote_intersection_l919_91942

/-- The function representing the given graph -/
noncomputable def f (x : ℝ) : ℝ := (x^2 - 9*x + 20) / (x^2 - 9*x + 21)

/-- The horizontal asymptote of the function -/
def horizontal_asymptote : ℝ := 1

/-- Theorem stating that there is no intersection of asymptotes -/
theorem no_asymptote_intersection :
  ¬ ∃ (x y : ℝ), (∀ ε > 0, ∃ δ > 0, ∀ t, |t - x| < δ → |f t - y| < ε) ∧
                 y ≠ horizontal_asymptote :=
by
  sorry

#check no_asymptote_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_asymptote_intersection_l919_91942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_n_with_k_prime_factors_l919_91903

/-- Number of distinct prime factors of a natural number -/
def num_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.card

/-- 
Theorem: For any positive integer k and positive odd number m, 
there exists a positive integer n such that m^n + n^m has at least k distinct prime factors.
-/
theorem existence_of_n_with_k_prime_factors 
  (k : ℕ) (m : ℕ) 
  (hk : k > 0) (hm : m > 0) (hm_odd : m % 2 = 1) :
  ∃ n : ℕ, n > 0 ∧ (num_prime_factors (m^n + n^m) ≥ k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_n_with_k_prime_factors_l919_91903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_normal_vectors_l919_91918

/-- Given two planes α and β with normal vectors, prove that if they are parallel, then x = 1/2 -/
theorem parallel_planes_normal_vectors (x : ℝ) : 
  let a : Fin 3 → ℝ := ![(-1 : ℝ), 2, 4]
  let b : Fin 3 → ℝ := ![x, (-1 : ℝ), (-2 : ℝ)]
  (∃ (k : ℝ), ∀ i, a i = k * b i) → x = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_normal_vectors_l919_91918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_l919_91984

noncomputable section

open Real

-- Define the dihedral angle
def dihedral_angle : ℝ := 5 * π / 6

-- Define the planes and lines
variable (α β : Set ℝ) (l a b : Set ℝ)

-- State the conditions
variable (h1 : DihAng α l β = dihedral_angle)
variable (h2 : Perpendicular a α)
variable (h3 : b ⊆ β)

-- Define the angle between lines a and b
def angle_ab (a b : Set ℝ) : ℝ := sorry

-- State the theorem
theorem angle_range : π/3 ≤ angle_ab a b ∧ angle_ab a b ≤ π/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_range_l919_91984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_l919_91989

theorem equality_condition (a b c d : ℕ+) (hd : d > 1) :
  Real.sqrt ((a : ℝ) ^ 2 + (b : ℝ) ^ (d : ℕ) / c) = (a : ℝ) ^ (d : ℕ) * Real.sqrt ((b : ℝ) / c) ↔
  (c : ℝ) = ((a : ℝ) ^ (2 * (d : ℕ)) * b - (b : ℝ) ^ (d : ℕ)) / (a : ℝ) ^ 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_l919_91989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_sum_l919_91985

theorem factorization_sum (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 13*x + 40 = (x + a)*(x + b)) →
  (∀ x : ℝ, x^2 - 19*x + 88 = (x - b)*(x - c)) →
  a + b + c = 24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_sum_l919_91985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_spheres_volume_calculation_l919_91967

noncomputable section

/-- The volume of a sphere with radius r -/
def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The volume of the region between two concentric spheres -/
def concentricSpheresVolume (r₁ r₂ : ℝ) : ℝ := sphereVolume r₂ - sphereVolume r₁

theorem concentric_spheres_volume_calculation :
  concentricSpheresVolume 4 7 = 372 * Real.pi := by
  -- Unfold the definitions
  unfold concentricSpheresVolume sphereVolume
  -- Simplify the expression
  simp [Real.pi]
  -- The rest of the proof is omitted
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concentric_spheres_volume_calculation_l919_91967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_time_approx_45_hours_l919_91931

/-- The radius of Earth at the equator in miles -/
def earth_radius : ℝ := 3960

/-- The speed of the jet in miles per hour -/
def jet_speed : ℝ := 550

/-- The circumference of Earth at the equator -/
noncomputable def earth_circumference : ℝ := 2 * Real.pi * earth_radius

/-- The time taken for the jet to fly around the Earth -/
noncomputable def flight_time : ℝ := earth_circumference / jet_speed

/-- Theorem stating that the flight time is approximately 45 hours -/
theorem flight_time_approx_45_hours :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |flight_time - 45| < ε := by
  sorry

#eval earth_radius
#eval jet_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flight_time_approx_45_hours_l919_91931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C_to_l_l919_91995

/-- The line l in the Cartesian plane -/
def line_l (x y : ℝ) : Prop := y = x - 4

/-- The circle C in the Cartesian plane -/
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

/-- A point P on the circle C -/
def point_on_C (P : ℝ × ℝ) : Prop := circle_C P.1 P.2

/-- The distance from a point to the line l -/
noncomputable def distance_to_line (P : ℝ × ℝ) : ℝ :=
  |P.2 - P.1 + 4| / Real.sqrt 2

/-- The theorem stating the minimum distance from any point on C to line l -/
theorem min_distance_C_to_l :
  ∀ P : ℝ × ℝ, point_on_C P → 
  ∃ Q : ℝ × ℝ, point_on_C Q ∧ 
  ∀ R : ℝ × ℝ, point_on_C R → distance_to_line Q ≤ distance_to_line R ∧
  distance_to_line Q = 3 * Real.sqrt 2 - 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_C_to_l_l919_91995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tradesman_gain_percentage_l919_91993

/-- Represents the percentage of fraud applied by the tradesman -/
noncomputable def fraudPercentage : ℝ := 20

/-- Calculates the buying price after fraud -/
noncomputable def buyingPrice (trueValue : ℝ) : ℝ := trueValue * (1 - fraudPercentage / 100)

/-- Calculates the selling price after fraud -/
noncomputable def sellingPrice (trueValue : ℝ) : ℝ := trueValue * (1 + fraudPercentage / 100)

/-- Calculates the gain percentage on outlay -/
noncomputable def gainPercentage (buyPrice sellingPrice : ℝ) : ℝ :=
  ((sellingPrice - buyPrice) / buyPrice) * 100

theorem tradesman_gain_percentage (trueValue : ℝ) (trueValue_pos : trueValue > 0) :
  gainPercentage (buyingPrice trueValue) (sellingPrice trueValue) = 50 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tradesman_gain_percentage_l919_91993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_palindromic_prime_l919_91974

/-- A number is palindromic if it reads the same backwards as forwards -/
def isPalindromic (n : ℕ) : Prop :=
  (n.repr.toList).reverse = n.repr.toList

/-- A number is four-digit if it's between 1000 and 9999 inclusive -/
def isFourDigit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_palindromic_prime : 
  ∀ n : ℕ, isFourDigit n → isPalindromic n → Nat.Prime n → 1101 ≤ n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_palindromic_prime_l919_91974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l919_91965

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^4 - 5*x^2 + 6) / |x - 2|

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x ≠ 2}

-- Theorem stating that the domain of f is (-∞, 2) ∪ (2, ∞)
theorem domain_of_f : domain_f = Set.Iio 2 ∪ Set.Ioi 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l919_91965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_of_37_l919_91930

theorem sum_of_divisors_of_37 (h : Nat.Prime 37) : 
  (Finset.filter (λ x => 37 % x = 0) (Finset.range 38)).sum id = 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_of_37_l919_91930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l919_91961

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 5 - (x - 1)^2 / 3

-- Define the theorem
theorem intersection_point_sum :
  ∃ (x y : ℝ), f x = f (x - 4) ∧ f x = y ∧ x + y = 20/3 := by
  -- We'll use x = 3 and y = 11/3 as our witnesses
  use 3, 11/3
  -- Now we need to prove the three conditions
  apply And.intro
  · -- Prove f 3 = f (3 - 4)
    simp [f]
    -- Numerical calculation
    norm_num
  · -- Prove f 3 = 11/3
    apply And.intro
    · simp [f]
      -- Numerical calculation
      norm_num
    · -- Prove 3 + 11/3 = 20/3
      norm_num

-- The proof is complete, so we don't need 'sorry'

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_sum_l919_91961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_mold_radius_equals_expected_l919_91946

/-- The volume of a hemisphere with radius r -/
noncomputable def hemisphere_volume (r : ℝ) : ℝ := (2 / 3) * Real.pi * r^3

/-- The radius of smaller molds given the conditions of the problem -/
noncomputable def smaller_mold_radius : ℝ :=
  let large_bowl_radius : ℝ := 2
  let fill_ratio : ℝ := 3 / 4
  let num_small_molds : ℕ := 8
  let large_bowl_volume : ℝ := hemisphere_volume large_bowl_radius
  let chocolate_volume : ℝ := fill_ratio * large_bowl_volume
  ((3 * chocolate_volume) / (num_small_molds * 4 * Real.pi))^(1/3)

theorem smaller_mold_radius_equals_expected : 
  smaller_mold_radius = (3^(1/3)) / (2^(2/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_mold_radius_equals_expected_l919_91946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_product_equation_l919_91980

theorem sqrt_product_equation (x : ℝ) (h : x > 0) :
  Real.sqrt (12 * x) * Real.sqrt (20 * x) * Real.sqrt (6 * x) * Real.sqrt (30 * x) = 60 →
  x = (30 ^ (1/4 : ℝ)) / 60 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_product_equation_l919_91980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_man_time_l919_91908

-- Define constants
def platform_length : ℝ := 225.018
def platform_pass_time : ℝ := 35
def train_speed_kmh : ℝ := 54

-- Define the theorem
theorem train_pass_man_time :
  let train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600
  let train_length : ℝ := train_speed_ms * platform_pass_time - platform_length
  let man_pass_time : ℝ := train_length / train_speed_ms
  ⌊man_pass_time⌋₊ = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_man_time_l919_91908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_genotan_is_guilty_l919_91973

-- Define the set of people
inductive Person : Type
  | Isobel : Person
  | Josh : Person
  | Genotan : Person
  | Tegan : Person

-- Define a function to represent the statements made by each person
def statement (guilty : Person) (p : Person) : Prop :=
  match p with
  | Person.Isobel => Person.Josh ≠ guilty
  | Person.Josh => Person.Genotan = guilty
  | Person.Genotan => Person.Tegan = guilty
  | Person.Tegan => Person.Isobel ≠ guilty

-- Define the guilty person
def guilty : Person := Person.Genotan

-- Theorem stating that Genotan is guilty
theorem genotan_is_guilty :
  (∃! p : Person, p = guilty) ∧
  (∀ p : Person, p ≠ guilty → statement guilty p) ∧
  (¬statement guilty guilty) →
  guilty = Person.Genotan :=
by sorry

#check genotan_is_guilty

end NUMINAMATH_CALUDE_ERRORFEEDBACK_genotan_is_guilty_l919_91973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_zero_function_l919_91979

noncomputable def f (m : ℝ) (x : ℝ) := -2 * Real.tan x + m

theorem range_of_m_for_zero_function :
  ∀ m : ℝ, (∃ x ∈ Set.Icc (-π/4 : ℝ) (π/3 : ℝ), f m x = 0) ↔ m ∈ Set.Icc (-2 : ℝ) (2 * Real.sqrt 3) :=
by sorry

#check range_of_m_for_zero_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_zero_function_l919_91979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_equation_l919_91952

theorem certain_number_equation (a : ℝ) :
  (625 : ℝ)^(-(1/4 : ℝ)) + 25^(-(1/2 : ℝ)) + a^(-(1 : ℝ)) = 11 → a = 5/53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_equation_l919_91952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_characterization_l919_91975

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the conditions for the function
def SatisfiesConditions (f : RealFunction) : Prop :=
  (∀ m : ℤ, f m = m) ∧
  (∀ a b c d : ℤ, c > 0 → d > 0 → |a * d - b * c| = 1 →
    f ((a + b : ℝ) / (c + d)) = (f (a / c) + f (b / d)) / 2) ∧
  (∀ x y : ℝ, x < y → f x < f y)

-- Define the continued fraction expansion
noncomputable def continuedFraction (x : ℝ) : ℕ → ℤ
  | 0 => ⌊x⌋
  | n + 1 => ⌊1 / (x - ↑⌊x⌋)⌋

-- Define the partial sum of continued fraction terms
noncomputable def A (x : ℝ) : ℕ → ℕ
  | 0 => 0
  | n + 1 => A x n + (continuedFraction x (n + 1)).natAbs

-- Define the sum in the formula
noncomputable def SumFormula (x : ℝ) : ℝ :=
  2 * ∑' n, (-1 : ℝ)^n / 2^(A x n)

-- The main theorem
theorem unique_function_characterization {f : RealFunction} (h : SatisfiesConditions f) :
  ∀ x : ℝ, f x = (continuedFraction x 0 : ℝ) - SumFormula x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_characterization_l919_91975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_c_value_l919_91927

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi ∧
  0 < t.B ∧ t.B < Real.pi ∧
  0 < t.C ∧ t.C < Real.pi ∧
  t.A + t.B + t.C = Real.pi ∧
  Real.cos t.A = 1/3 ∧
  (1/2 * t.b * t.c * Real.sin t.A) = 4 * Real.sqrt 2 ∧
  Real.sin (t.A - t.B) = 2 * Real.sin t.B * (1 - 2 * Real.cos t.A)

-- Theorem statement
theorem triangle_side_c_value (t : Triangle) (h : triangle_conditions t) : t.c = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_c_value_l919_91927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_represents_opposite_quantities_l919_91945

-- Define the possible options
inductive QuantityOption
  | A
  | B
  | C
  | D

-- Define a function to check if an option represents opposite quantities
def represents_opposite_quantities (o : QuantityOption) : Prop :=
  match o with
  | QuantityOption.A => False  -- Winning games and losing money are not opposites
  | QuantityOption.B => False  -- Traveling east and north are not opposites
  | QuantityOption.C => False  -- Transporting and selling apples are not opposites
  | QuantityOption.D => True   -- Water level rising and dropping are opposites

-- Theorem stating that only Option D represents opposite quantities
theorem only_D_represents_opposite_quantities :
  ∀ o : QuantityOption, represents_opposite_quantities o ↔ o = QuantityOption.D :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_represents_opposite_quantities_l919_91945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_visit_pattern_l919_91925

-- Define the cube structure
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 8 × Fin 8)
  is_adjacent : Fin 8 → Fin 8 → Bool

-- Define the ant's path
def AntPath (cube : Cube) := List (Fin 8)

-- Define the property of not turning back
def NoTurningBack (cube : Cube) (path : AntPath cube) : Prop :=
  ∀ i j k, i < j → j < k → path.get i = path.get k → path.get i ≠ path.get j

-- Define the visit count for each vertex
def VisitCount (cube : Cube) (path : AntPath cube) (v : Fin 8) : Nat :=
  (path.filter (· = v)).length

-- The main theorem
theorem impossible_visit_pattern (cube : Cube) :
  ¬ ∃ (path : AntPath cube),
    NoTurningBack cube path ∧
    (∃ (v : Fin 8), VisitCount cube path v = 25 ∧
      ∀ (w : Fin 8), w ≠ v → VisitCount cube path w = 20) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_visit_pattern_l919_91925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_system_solution_l919_91929

theorem logarithmic_system_solution :
  ∀ (x y : ℝ), x > 0 ∧ x ≠ 1 ∧ y > 0 ∧ y ≠ 1 →
  (((Real.log x / Real.log y - Real.log y / Real.log x = 8/3) ∧ (x * y = 16)) ↔ 
   ((x = 8 ∧ y = 2) ∨ (x = 1/4 ∧ y = 64))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_system_solution_l919_91929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_sequence_with_contradictory_sums_l919_91962

theorem no_sequence_with_contradictory_sums :
  ¬ ∃ (a : Fin 17 → ℝ),
    (∀ i : Fin 11, (Finset.range 7).sum (fun j ↦ a ⟨(i + j : Fin 17), sorry⟩) > 0) ∧
    (∀ i : Fin 7, (Finset.range 11).sum (fun j ↦ a ⟨(i + j : Fin 17), sorry⟩) < 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_sequence_with_contradictory_sums_l919_91962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_percentage_approx_l919_91956

def votes : List Nat := [12136, 17636, 23840, 19568, 17126, 20640, 26228]

def total_votes : Nat := votes.sum

def winning_votes : Nat := (votes.maximum?).getD 0

theorem winning_percentage_approx (ε : Real) (h : ε > 0) :
  ∃ p : Real, |p - 19.11| < ε ∧ p = (winning_votes * 100 : Real) / total_votes :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_percentage_approx_l919_91956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_parallel_line_equation_l919_91914

/-- The intersection point of two lines -/
noncomputable def intersection_point (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ × ℝ :=
  ((b₁ * c₂ - b₂ * c₁) / (a₁ * b₂ - a₂ * b₁),
   (a₂ * c₁ - a₁ * c₂) / (a₁ * b₂ - a₂ * b₁))

/-- The equation of a line passing through a point with a given slope -/
def line_equation (x₀ y₀ m : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = m * (x - x₀)

theorem line_through_intersection :
  let p := intersection_point 1 (-1) (-1) 2 (-1) 0
  ∀ x y : ℝ, line_equation p.1 p.2 3 x y ↔ 3 * x - y - 1 = 0 := by
  sorry

/-- The equation of a line parallel to another line at a given distance -/
theorem parallel_line_equation (a b c d : ℝ) :
  ∀ x y : ℝ, (a * x + b * y + c = 0 ∨ a * x + b * y + d = 0) ↔
  (|c| / Real.sqrt (a^2 + b^2) = Real.sqrt 10 / 10 ∨
   |d| / Real.sqrt (a^2 + b^2) = Real.sqrt 10 / 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_parallel_line_equation_l919_91914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_three_digit_numbers_l919_91902

def Card := Fin 2 → Nat

def cards : Fin 3 → Card
| 0 => ![0, 2]
| 1 => ![3, 4]
| 2 => ![5, 6]

def is_valid_number (n : Nat) : Prop :=
  ∃ (i j k : Fin 3) (x y z : Fin 2),
    i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    n = 100 * (cards i x) + 10 * (cards j y) + (cards k z) ∧
    100 ≤ n ∧ n < 1000

-- Add this instance to make is_valid_number decidable
instance : DecidablePred is_valid_number :=
  fun n => decidable_of_iff (∃ (i j k : Fin 3) (x y z : Fin 2),
    i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    n = 100 * (cards i x) + 10 * (cards j y) + (cards k z) ∧
    100 ≤ n ∧ n < 1000) 
    (by simp [is_valid_number])

def count_valid_numbers : Nat :=
  (Finset.filter is_valid_number (Finset.range 1000)).card

theorem distinct_three_digit_numbers :
  count_valid_numbers = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_three_digit_numbers_l919_91902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_in_boxes_probability_l919_91919

theorem balls_in_boxes_probability 
  (n i r k l : ℕ) 
  (h1 : k + l ≤ r) 
  (h2 : i > 0) 
  (h3 : i ≤ n) :
  (Nat.choose r k * Nat.choose (r-k) l * (i-1)^l * (n-i)^(r-k-l) : ℚ) / n^r = 
  (Nat.choose r k * Nat.choose (r-k) l * (i-1)^l * (n-i)^(r-k-l) : ℚ) / n^r :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_in_boxes_probability_l919_91919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_geometric_sequence_l919_91947

-- Define a function to check if three digits form a geometric sequence
def is_geometric_sequence (a b c : ℕ) : Prop :=
  ∃ r : ℚ, b = Int.floor (a * r) ∧ c = Int.floor (b * r)

-- Define a function to check if a number is a three-digit integer
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

-- Define a function to get the digits of a three-digit number
def digits (n : ℕ) : (ℕ × ℕ × ℕ) :=
  (n / 100, (n / 10) % 10, n % 10)

-- Define the main theorem
theorem largest_three_digit_geometric_sequence :
  ∀ n : ℕ, is_three_digit n →
    (let (d1, d2, d3) := digits n
     d1 = 8 ∧
     d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3 ∧
     is_geometric_sequence d1 d2 d3) →
    n ≤ 842 :=
by
  sorry

#check largest_three_digit_geometric_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_geometric_sequence_l919_91947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_y_sum_l919_91909

/-- Predicate indicating that a set of points forms a rectangle -/
def IsRectangle (s : Set (ℝ × ℝ)) : Prop := sorry

/-- Predicate indicating that two points are opposite vertices in a rectangle -/
def IsOpposite (s : Set (ℝ × ℝ)) (p q : ℝ × ℝ) : Prop := sorry

/-- Given a rectangle with opposite vertices at (5, 20) and (15, -10),
    the sum of the y-coordinates of the other two vertices is 10. -/
theorem rectangle_y_sum : 
  ∀ (a b : ℝ × ℝ),
  (∃ (r : Set (ℝ × ℝ)), IsRectangle r ∧ (5, 20) ∈ r ∧ (15, -10) ∈ r ∧ a ∈ r ∧ b ∈ r ∧ 
   IsOpposite r (5, 20) (15, -10) ∧ IsOpposite r a b) →
  a.2 + b.2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_y_sum_l919_91909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_discount_percentage_l919_91911

/-- Given an initial price and two successive discounts, calculate the total percentage reduction -/
theorem total_discount_percentage (P : ℝ) (h : P > 0) : 
  (P - P * (1 - 0.3) * (1 - 0.5)) / P * 100 = 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_discount_percentage_l919_91911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l919_91958

/-- Simple interest calculation --/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem interest_rate_proof (principal interest time : ℝ) 
  (h_principal : principal = 12000)
  (h_interest : interest = 4320)
  (h_time : time = 3) :
  ∃ rate : ℝ, simple_interest principal rate time = interest ∧ rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_proof_l919_91958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_of_chord_l919_91970

-- Define the line
def line (x y : ℝ) : Prop := 2 * x + 3 * y + 1 = 0

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y = 0

-- Define the perpendicular bisector
def perp_bisector (x y : ℝ) : Prop := 3 * x - 2 * y - 7 = 0

-- Theorem statement
theorem perpendicular_bisector_of_chord 
  (A B : ℝ × ℝ) 
  (h_line_A : line A.1 A.2)
  (h_line_B : line B.1 B.2)
  (h_circle_A : my_circle A.1 A.2)
  (h_circle_B : my_circle B.1 B.2)
  (h_distinct : A ≠ B) :
  perp_bisector ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_of_chord_l919_91970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_determinant_l919_91992

theorem triangle_angle_determinant (P Q R : ℝ) : 
  P + Q + R = Real.pi →
  Matrix.det !![Real.cos P, Real.sin P, 1; Real.cos Q, Real.sin Q, 1; Real.cos R, Real.sin R, 1] = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_determinant_l919_91992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_l919_91959

-- Define the hexagon structure
structure Hexagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

-- Define helper functions
def isConvex (h : Hexagon) : Prop := sorry
def isEquilateral (h : Hexagon) : Prop := sorry
def angleEqual (p1 p2 p3 : ℝ × ℝ) (angle : ℝ) : Prop := sorry
def parallel (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry
def distinctYCoordinates (h : Hexagon) (s : Set ℝ) : Prop := sorry

-- Define the conditions for the hexagon
def isValidHexagon (h : Hexagon) : Prop :=
  h.A = (0, 0) ∧
  h.B.2 = 3 ∧
  (∃ b : ℝ, h.B = (b, 3)) ∧
  isConvex h ∧
  isEquilateral h ∧
  angleEqual h.F h.A h.B 120 ∧
  parallel h.A h.B h.D h.E ∧
  parallel h.B h.C h.E h.F ∧
  parallel h.C h.D h.F h.A ∧
  distinctYCoordinates h {0, 3, 6, 9, 12, 15}

-- Define the area function
noncomputable def area (h : Hexagon) : ℝ := sorry

-- Theorem statement
theorem hexagon_area (h : Hexagon) (hValid : isValidHexagon h) : 
  area h = 108 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_l919_91959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l919_91923

/-- A trapezoid with specific measurements -/
structure Trapezoid :=
  (A B C D : ℝ × ℝ)
  (is_trapezoid : sorry)
  (AB_CD_parallel : sorry)
  (AB_CD_equal : sorry)
  (vertical_leg_AB : |A.2 - B.2| = 6)
  (vertical_leg_CD : |C.2 - D.2| = 6)
  (horizontal_diff_AB : |A.1 - B.1| = 8)
  (horizontal_diff_CD : |C.1 - D.1| = 8)

/-- The perimeter of a trapezoid -/
noncomputable def perimeter (t : Trapezoid) : ℝ :=
  let AB := Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2)
  let BC := |t.B.1 - t.C.1|
  let CD := Real.sqrt ((t.C.1 - t.D.1)^2 + (t.C.2 - t.D.2)^2)
  let DA := |t.D.1 - t.A.1|
  AB + BC + CD + DA

/-- The perimeter of the specific trapezoid is 44 -/
theorem trapezoid_perimeter (t : Trapezoid) : perimeter t = 44 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_l919_91923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_theorem_l919_91905

theorem power_sum_theorem (x y : ℝ) (h1 : (2 : ℝ)^x = 3) (h2 : (2 : ℝ)^y = 5) : 
  (2 : ℝ)^(x + 2*y) = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_theorem_l919_91905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonpositive_implies_k_geq_one_l919_91932

-- Define the function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log (x - 1) - k * (x - 1) + 1

-- State the theorem
theorem f_nonpositive_implies_k_geq_one :
  ∀ k : ℝ, (∀ x : ℝ, x > 1 → f k x ≤ 0) → k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonpositive_implies_k_geq_one_l919_91932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_is_twelve_l919_91922

-- Define the parallelogram as a structure
structure Parallelogram where
  base : ℝ
  height : ℝ

-- Define the area function for a parallelogram
def area (p : Parallelogram) : ℝ := p.base * p.height

-- Theorem statement
theorem parallelogram_area_is_twelve :
  ∀ p : Parallelogram, p.base = 3 ∧ p.height = 4 → area p = 12 :=
by
  intro p h
  have h1 : p.base = 3 := h.left
  have h2 : p.height = 4 := h.right
  calc
    area p = p.base * p.height := rfl
    _ = 3 * 4 := by rw [h1, h2]
    _ = 12 := by norm_num

#check parallelogram_area_is_twelve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_is_twelve_l919_91922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_range_boundaries_l919_91950

/-- The function f(x) = 3 / (3 + 3x^2 + 6x) -/
noncomputable def f (x : ℝ) : ℝ := 3 / (3 + 3 * x^2 + 6 * x)

/-- The lower bound of the range of f -/
def c : ℝ := 0

/-- The upper bound of the range of f -/
def d : ℝ := 1

theorem sum_of_range_boundaries :
  c + d = 1 ∧ Set.Icc c d = Set.range f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_range_boundaries_l919_91950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_correct_guesses_l919_91917

/-- A strategy for Alice and Bob in the sequence guessing game -/
structure Strategy (k : ℕ) where
  /-- Alice's choice of which bit to reveal -/
  alice_reveal : (Fin (2^k) → Bool) → Fin (2^k)
  /-- Bob's guesses based on Alice's reveal -/
  bob_guess : Fin (2^k) × Bool → Finset (Fin (2^k) × Bool)

/-- The main theorem stating that k+1 is the maximum number of correct guesses -/
theorem max_correct_guesses (k : ℕ) :
  (∃ (s : Strategy k), ∀ (seq : Fin (2^k) → Bool),
    (s.bob_guess (s.alice_reveal seq, seq (s.alice_reveal seq))).card = k + 1 ∧
    ∀ (i : Fin (2^k)), (i, seq i) ∈ s.bob_guess (s.alice_reveal seq, seq (s.alice_reveal seq))) ∧
  (∀ (s : Strategy k), ∃ (seq : Fin (2^k) → Bool),
    (s.bob_guess (s.alice_reveal seq, seq (s.alice_reveal seq))).card ≤ k + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_correct_guesses_l919_91917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_S_T_l919_91948

-- Define the sets S and T
def S : Set ℝ := {x : ℝ | x > -2}
def T : Set ℝ := {x : ℝ | -4 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem intersection_S_T : S ∩ T = Set.Ioc (-2) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_S_T_l919_91948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_shift_l919_91904

noncomputable def f (x : ℝ) := Real.sin (3 * x - Real.pi / 3)
noncomputable def g (x : ℝ) := Real.sin (3 * x)

theorem sine_graph_shift :
  ∀ x : ℝ, f (x + Real.pi / 9) = g x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_shift_l919_91904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_leaves_needed_l919_91953

/-- The number of leaves Sabrina needs for her poultice -/
structure LeavesNeeded where
  basil : ℕ
  sage : ℕ
  verbena : ℕ
  chamomile : ℕ

/-- The conditions for Sabrina's herb collection -/
def HerbConditions (l : LeavesNeeded) : Prop :=
  l.basil = 3 * l.sage ∧
  l.verbena = l.sage + 8 ∧
  l.chamomile = 2 * l.sage + 7 ∧
  l.basil = 36

/-- The theorem stating the total number of leaves needed -/
theorem total_leaves_needed (l : LeavesNeeded) 
  (h : HerbConditions l) : 
  l.basil + l.sage + l.verbena + l.chamomile = 99 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_leaves_needed_l919_91953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_two_flips_l919_91978

/-- The expected value of winnings after two flips of a biased coin -/
theorem expected_value_two_flips 
  (p_heads : ℝ) (p_tails : ℝ) (win_amount : ℝ) (lose_amount : ℝ) 
  (h1 : p_heads = 2/5)
  (h2 : p_tails = 3/5)
  (h3 : win_amount = 4)
  (h4 : lose_amount = 1) :
  2 * (p_heads * win_amount - p_tails * lose_amount) = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_two_flips_l919_91978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_sectors_area_l919_91969

/-- The area of the overlapping region of two 60° sectors in a circle with radius 15 -/
theorem overlapping_sectors_area : 
  let r : ℝ := 15
  let angle : ℝ := 60 * Real.pi / 180  -- 60° in radians
  let sector_area : ℝ := 1/2 * r^2 * angle
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * r^2
  let overlap_area : ℝ := 2 * (sector_area - triangle_area)
  overlap_area = 75 * Real.pi - 112.5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_sectors_area_l919_91969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_floors_minimize_cost_l919_91937

/-- The optimal number of floors that minimizes the average comprehensive cost -/
def optimal_floors : ℕ := 15

/-- The floor area in square meters -/
noncomputable def floor_area : ℝ := 2000

/-- The total land cost in yuan -/
noncomputable def land_cost : ℝ := 21600000

/-- The average construction cost per square meter as a function of the number of floors -/
noncomputable def avg_construction_cost (x : ℝ) : ℝ := 560 + 48 * x

/-- The average comprehensive cost per square meter as a function of the number of floors -/
noncomputable def avg_comprehensive_cost (x : ℝ) : ℝ :=
  avg_construction_cost x + land_cost / (floor_area * x)

/-- Theorem stating that 15 floors minimizes the average comprehensive cost -/
theorem optimal_floors_minimize_cost :
  ∀ x : ℝ, x ≥ 10 → avg_comprehensive_cost (optimal_floors : ℝ) ≤ avg_comprehensive_cost x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_floors_minimize_cost_l919_91937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_approximation_l919_91907

noncomputable def f (x : ℝ) : ℝ := x^2 / (x^2 + 2*x + 2)

theorem integral_approximation :
  ∃ (I : ℝ), (∫ (x : ℝ) in Set.Icc (-1) 1, f x) = I ∧ |I - 0.4| < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_approximation_l919_91907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_event_equivalence_l919_91941

/-- Represents a shoe --/
structure Shoe where
  pair_id : Nat
  is_left : Bool

/-- The set of all shoes from four pairs --/
def all_shoes : Finset Shoe :=
  sorry

/-- A selection of four shoes --/
def selection : Finset Shoe :=
  sorry

/-- Predicate: all shoes in the selection form pairs --/
def all_form_pairs (s : Finset Shoe) : Prop :=
  ∀ shoe ∈ s, ∃ other_shoe ∈ s, shoe.pair_id = other_shoe.pair_id ∧ shoe.is_left ≠ other_shoe.is_left

/-- Predicate: at least two shoes in the selection do not form a pair --/
def at_least_two_dont_pair (s : Finset Shoe) : Prop :=
  ∃ shoe1 shoe2, shoe1 ∈ s ∧ shoe2 ∈ s ∧ shoe1 ≠ shoe2 ∧ 
    (∀ other_shoe ∈ s, shoe1.pair_id ≠ other_shoe.pair_id ∨ shoe1.is_left = other_shoe.is_left) ∧
    (∀ other_shoe ∈ s, shoe2.pair_id ≠ other_shoe.pair_id ∨ shoe2.is_left = other_shoe.is_left)

theorem complementary_event_equivalence :
  ¬(all_form_pairs selection) ↔ at_least_two_dont_pair selection :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_event_equivalence_l919_91941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_fuel_theorem_l919_91939

/-- Represents the characteristics of a truck's fuel consumption -/
structure TruckFuel where
  distance : ℝ  -- Distance traveled in miles
  fuel : ℝ      -- Fuel consumed in gallons

/-- Calculates the distance a truck can travel given a certain amount of fuel -/
noncomputable def calculateDistance (truck : TruckFuel) (newFuel : ℝ) : ℝ :=
  (newFuel / truck.fuel) * truck.distance

/-- Calculates the fuel efficiency of a truck in miles per gallon -/
noncomputable def fuelEfficiency (truck : TruckFuel) : ℝ :=
  truck.distance / truck.fuel

/-- Theorem stating the distance a truck can travel on 15 gallons and its fuel efficiency -/
theorem truck_fuel_theorem (truck : TruckFuel) 
    (h1 : truck.distance = 300)
    (h2 : truck.fuel = 10) :
    calculateDistance truck 15 = 450 ∧ fuelEfficiency truck = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_fuel_theorem_l919_91939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_k_values_l919_91940

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {a b c d e f : ℝ} :
  (a * d = b * c) ↔ (∀ x y, a * x + b * y + e = 0 ↔ c * x + d * y + f = 0)

/-- Definition of line l₁ -/
def l₁ (k : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (k - 3) * x + (4 - k) * y + 1 = 0

/-- Definition of line l₂ -/
def l₂ (k : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ 2 * (k - 3) * x - 2 * y + 3 = 0

/-- Theorem stating that if l₁ is parallel to l₂, then k = 3 or k = 5 -/
theorem parallel_lines_k_values (k : ℝ) :
  (∀ x y, l₁ k x y ↔ l₂ k x y) → k = 3 ∨ k = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_k_values_l919_91940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l919_91976

/-- The distance from a point on the line y = ax - 2a + 5 to the line x - 2y + 3 = 0 is √5 -/
theorem distance_to_line :
  ∀ (a : ℝ),
  let A : ℝ × ℝ := (2, 5)
  let l₁ : ℝ → ℝ := fun x ↦ a * x - 2 * a + 5
  let l : ℝ × ℝ → ℝ := fun p ↦ p.1 - 2 * p.2 + 3
  (l₁ A.1 = A.2) →
  (abs (l A) / Real.sqrt (1 + 4) = Real.sqrt 5) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l919_91976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l919_91921

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin (Real.pi / 4 + x / 2) * Real.sin (Real.pi / 4 - x / 2) - Real.sin (Real.pi + x)

noncomputable def g (x : ℝ) : ℝ := f (Real.pi / 2 - x)

theorem problem_statement :
  (∃ m : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → (g x)^2 - m * g x + 2 = 0) ↔ m ∈ Set.Icc (2 * Real.sqrt 2) 3) ∧
  (∃ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 (11 * Real.pi / 12) → f x + a * g (-x) > 0) ↔ a ∈ Set.Iio (-Real.sqrt 2) ∪ Set.Ioi (Real.sqrt 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l919_91921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_approx_17_l919_91954

/-- Represents the cost and profit structure of a product sold through multiple levels -/
structure SaleChain where
  manufacturer_profit : ℚ
  wholesaler_profit : ℚ
  retailer_profit : ℚ
  final_price : ℚ

/-- Calculates the initial cost price given a SaleChain -/
noncomputable def initial_cost_price (chain : SaleChain) : ℚ :=
  chain.final_price / ((1 + chain.manufacturer_profit) * (1 + chain.wholesaler_profit) * (1 + chain.retailer_profit))

/-- Theorem stating that for the given profit margins and final price, the initial cost is approximately 17 -/
theorem cost_price_approx_17 (chain : SaleChain) 
    (h1 : chain.manufacturer_profit = 18/100)
    (h2 : chain.wholesaler_profit = 20/100)
    (h3 : chain.retailer_profit = 25/100)
    (h4 : chain.final_price = 3009/100) :
    abs (initial_cost_price chain - 17) < 1/100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_approx_17_l919_91954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_over_n_squared_l919_91997

/-- 
f(n) is the exponent of the largest power of 5 dividing 1¹2²3³...nⁿ
-/
def f (n : ℕ) : ℕ := sorry

/-- 
The limit of f(n)/n² as n approaches infinity is 1/8
-/
theorem limit_f_over_n_squared :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |f n / (n^2 : ℝ) - 1/8| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_over_n_squared_l919_91997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_problem_l919_91966

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point on a line segment
def PointOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

-- Define the distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem triangle_ratio_problem (ABC : Triangle) (D : ℝ × ℝ) :
  distance ABC.A ABC.B = 8 →
  distance ABC.B ABC.C = 10 →
  distance ABC.A ABC.C = 6 →
  PointOnSegment D ABC.A ABC.C →
  distance ABC.B D = 8 →
  ∃ (AD DC : ℝ), AD = 0 ∧ DC ≠ 0 ∧ AD / DC = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_problem_l919_91966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_proof_l919_91983

/-- Given a line L1 and a point P, the symmetric line L2 with respect to P 
    is such that for any point Q on L1, its symmetric point Q' with respect to P 
    lies on L2. -/
def is_symmetric_line (L1 L2 : ℝ → ℝ → Prop) (P : ℝ × ℝ) : Prop :=
  ∀ Q : ℝ × ℝ, L1 Q.1 Q.2 → 
    L2 (2 * P.1 - Q.1) (2 * P.2 - Q.2)

theorem symmetric_line_proof :
  let L1 (x y : ℝ) := 2 * x + 3 * y - 6 = 0
  let L2 (x y : ℝ) := 2 * x + 3 * y + 8 = 0
  let P : ℝ × ℝ := (1, -1)
  is_symmetric_line L1 L2 P := by
  sorry

#check symmetric_line_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_proof_l919_91983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l919_91963

-- Define the geometric sequence
def geometric_sequence : ℕ → ℝ := sorry

-- Define the common ratio
def q : ℝ := sorry

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - (2*geometric_sequence 2 - 1)*x + 2*geometric_sequence 2 - 2

-- State the theorem
theorem geometric_sequence_product :
  (∀ n : ℕ, geometric_sequence (n+1) > geometric_sequence n) →  -- increasing sequence
  (f (geometric_sequence 2 - 2) = 0) →                         -- a₂-2 is a zero of f
  (f (geometric_sequence 6 - 3) = 0) →                         -- a₆-3 is a zero of f
  (∀ x : ℝ, f x = f (-x)) →                                    -- f is an even function
  (∀ n : ℕ, geometric_sequence (n+1) = geometric_sequence n * q) →  -- geometric sequence property
  (Finset.prod (Finset.range 7) (fun i => geometric_sequence (i+1))) = 128 :=    -- T₇ = 128
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_l919_91963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_five_digit_divisible_by_smallest_primes_l919_91951

theorem smallest_five_digit_divisible_by_smallest_primes :
  let smallest_primes : List Nat := [2, 3, 5, 7, 11]
  let is_divisible (n : Nat) (d : Nat) := n % d = 0
  let is_five_digit (n : Nat) := 10000 ≤ n ∧ n < 100000
  11550 = (List.filter (fun n => is_five_digit n ∧ 
    (∀ p ∈ smallest_primes, is_divisible n p)) (List.range 100000)).head! :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_five_digit_divisible_by_smallest_primes_l919_91951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l919_91988

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - 2 / (2 * a^(x-1) + 1)

-- Theorem statement
theorem function_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  -- Part 1: The value of a that makes f(x) an odd function is 2
  (∀ x, f a x = -f a (-x)) → a = 2 ∧
  
  -- Part 2: The function f(x) with a = 2 is increasing on ℝ
  (∀ x y, x < y → f 2 x < f 2 y) ∧
  
  -- Part 3: For x ∈ [1, +∞), the maximum value of m that satisfies mf(x) ≤ 2^x - 2 is 0
  (∀ m, (∀ x, x ≥ 1 → m * f 2 x ≤ 2^x - 2) → m ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l919_91988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l919_91901

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the point of tangency
def P : ℝ × ℝ := (0, 1)

-- State the theorem
theorem tangent_line_at_P : 
  ∃ (m b : ℝ), ∀ x y : ℝ, 
    (x = P.1 ∧ y = f x) → 
    (y - P.2 = m * (x - P.1)) ∧
    (y = m * x + b) ∧
    (x - y + 1 = 0) := by
  -- Proof goes here
  sorry

#check tangent_line_at_P

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P_l919_91901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_entry_arrangements_l919_91971

theorem entry_arrangements (n : ℕ) (k : ℕ) : 
  n = 6 → k = 2 → (k * Nat.factorial (n - k)) = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_entry_arrangements_l919_91971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_bottle_probability_expected_pills_approximation_l919_91957

/-- Represents the state of pills in two bottles -/
structure PillState where
  bottle1 : ℕ
  bottle2 : ℕ

/-- The probability of selecting a bottle -/
noncomputable def bottle_selection_prob : ℝ := 1/2

/-- The number of days before checking for an empty bottle -/
def days_before_check : ℕ := 13

/-- The total number of pills in both bottles initially -/
def total_pills : ℕ := 20

/-- Calculates the probability of finding an empty bottle on the 14th day for the first time -/
noncomputable def prob_empty_bottle_day14 (s : PillState) : ℝ :=
  sorry

/-- Calculates the expected number of pills taken when discovering an empty bottle -/
noncomputable def expected_pills_taken (s : PillState) : ℝ :=
  sorry

theorem empty_bottle_probability (s : PillState) 
  (h1 : s.bottle1 = 10) (h2 : s.bottle2 = 10) :
  prob_empty_bottle_day14 s = 143 / 4096 := by
  sorry

theorem expected_pills_approximation (s : PillState) 
  (h1 : s.bottle1 = 10) (h2 : s.bottle2 = 10) :
  abs (expected_pills_taken s - 17.3) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_bottle_probability_expected_pills_approximation_l919_91957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_line_l_cartesian_line_intersects_circle_l919_91936

noncomputable section

-- Define the polar coordinates of point A
def point_A : ℝ × ℝ := (Real.sqrt 2, Real.pi / 4)

-- Define the polar equation of line l
def line_l (ρ θ a : ℝ) : Prop := ρ * Real.cos (θ - Real.pi / 4) = a

-- Define that point A lies on line l
axiom A_on_l : line_l point_A.1 point_A.2 (Real.sqrt 2)

-- Define the parametric equation of circle C
def circle_C (α : ℝ) : ℝ × ℝ := (1 + Real.cos α, Real.sin α)

-- Theorem 1: The value of a is √2
theorem a_value : ∃ a : ℝ, a = Real.sqrt 2 ∧ line_l point_A.1 point_A.2 a := by sorry

-- Theorem 2: The Cartesian equation of line l is x + y - 2 = 0
theorem line_l_cartesian : ∃ f : ℝ × ℝ → ℝ, ∀ x y : ℝ, f (x, y) = 0 ↔ x + y - 2 = 0 := by sorry

-- Theorem 3: Line l intersects with circle C
theorem line_intersects_circle : ∃ α : ℝ, ∃ x y : ℝ, circle_C α = (x, y) ∧ x + y - 2 = 0 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_line_l_cartesian_line_intersects_circle_l919_91936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_spheres_l919_91928

def sphere_center_1 : ℝ × ℝ × ℝ := (-5, -15, 10)
def sphere_radius_1 : ℝ := 25

def sphere_center_2 : ℝ × ℝ × ℝ := (20, 15, -25)
def sphere_radius_2 : ℝ := 95

theorem max_distance_between_spheres :
  let dist := λ (p q : ℝ × ℝ × ℝ) => Real.sqrt (
    (p.1 - q.1)^2 + (p.2.1 - q.2.1)^2 + (p.2.2 - q.2.2)^2
  )
  ∃ (p : ℝ × ℝ × ℝ) (q : ℝ × ℝ × ℝ),
    dist p sphere_center_1 = sphere_radius_1 ∧
    dist q sphere_center_2 = sphere_radius_2 ∧
    ∀ (x y : ℝ × ℝ × ℝ),
      dist x sphere_center_1 = sphere_radius_1 →
      dist y sphere_center_2 = sphere_radius_2 →
      dist x y ≤ dist p q ∧
      dist p q = 120 + 5 * Real.sqrt 110 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_spheres_l919_91928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l919_91955

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.b / t.c = 2 * Real.sqrt 3 / 3)
  (h2 : t.A + 3 * t.C = Real.pi) :
  (Real.cos t.C + Real.cos t.B = (Real.sqrt 3 - 1) / 3) ∧
  (t.b = 3 * Real.sqrt 3 → t.a * t.b * Real.sin t.C / 2 = 9 * Real.sqrt 2 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l919_91955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_intercept_sum_l919_91990

/-- Given points A, B, and C in a 2D plane, with D as the midpoint of AC,
    prove that the sum of the slope and y-intercept of line CD is 36/5 -/
theorem slope_intercept_sum (A B C D : ℝ × ℝ) : 
  A = (0, 8) →
  B = (0, 0) →
  C = (10, 0) →
  D = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) →
  let m := (D.2 - C.2) / (D.1 - C.1)
  let b := D.2 - m * D.1
  m + b = 36 / 5 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_intercept_sum_l919_91990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_cross_section_area_l919_91991

/-- The area of a specific cross-section in a right triangular truncated pyramid. -/
noncomputable def crossSectionArea (a b α : ℝ) : ℝ :=
  (7 * a + 3 * b) / (144 * Real.cos α) * Real.sqrt (3 * (a^2 + b^2 + 2 * a * b * Real.cos (2 * α)))

/-- Theorem: The area of the cross-section of a right triangular truncated pyramid,
    passing through the midline of the lateral face and the center of the lower base. -/
theorem truncated_pyramid_cross_section_area 
  (a b α : ℝ) 
  (h1 : a > b) 
  (h2 : α > 0) 
  (h3 : α < π / 2) :
  ∃ (S : ℝ), S = crossSectionArea a b α ∧ 
  S = (7 * a + 3 * b) / (144 * Real.cos α) * Real.sqrt (3 * (a^2 + b^2 + 2 * a * b * Real.cos (2 * α))) := by
  sorry

#check truncated_pyramid_cross_section_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_cross_section_area_l919_91991
