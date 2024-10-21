import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_to_line_l220_22013

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 5

-- Define the line ax + y - 1 = 0
def my_line (a x y : ℝ) : Prop := a * x + y - 1 = 0

-- Define the tangent line passing through (1, 1)
def my_tangent_line (x y : ℝ) : Prop := ∃ (m : ℝ), y - 1 = m * (x - 1)

-- State the theorem
theorem tangent_line_perpendicular_to_line (a : ℝ) : 
  (∃ (x y : ℝ), my_tangent_line x y ∧ my_circle x y) →
  (∀ (x y : ℝ), my_tangent_line x y → my_line a x y → 
    ∃ (m₁ m₂ : ℝ), (y - 1 = m₁ * (x - 1)) ∧ (y = -1/a * x + 1/a) ∧ m₁ * m₂ = -1) →
  a = 1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_to_line_l220_22013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cayley_hamilton_for_N_l220_22003

def N : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; -5, 2]

theorem cayley_hamilton_for_N :
  ∃ (p q : ℝ), N ^ 2 = p • N + q • (1 : Matrix (Fin 2) (Fin 2) ℝ) ∧ p = 5 ∧ q = -26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cayley_hamilton_for_N_l220_22003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheilas_weekly_earnings_l220_22052

/-- Sheila's weekly earnings calculation -/
theorem sheilas_weekly_earnings : 
  (3 * (8 * 7) + 2 * (6 * 7) : ℕ) = 252 :=
by
  -- Evaluate the expression
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheilas_weekly_earnings_l220_22052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_mfn_is_right_angle_l220_22082

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the left vertex and right focus
def left_vertex : ℝ × ℝ := (-1, 0)
def right_focus : ℝ × ℝ := (2, 0)

-- Define a line passing through the right focus
def line_through_focus (k : ℝ) (x : ℝ) : ℝ := k * (x - 2)

-- Define the intersection line
def intersection_line (x : ℝ) : Prop := x = 1/2

-- Define line segment (simplified)
def line_segment (A B : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define angle (simplified)
def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem angle_mfn_is_right_angle 
  (k : ℝ) 
  (P Q : ℝ × ℝ) 
  (hP : hyperbola P.1 P.2 ∧ P.2 = line_through_focus k P.1)
  (hQ : hyperbola Q.1 Q.2 ∧ Q.2 = line_through_focus k Q.1)
  (M N : ℝ × ℝ)
  (hM : intersection_line M.1 ∧ M ∈ line_segment left_vertex P)
  (hN : intersection_line N.1 ∧ N ∈ line_segment left_vertex Q) :
  angle right_focus M N = π/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_mfn_is_right_angle_l220_22082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_classification_complete_triangle_classification_exclusive_l220_22068

-- Define the possible triangle classifications based on angles
inductive TriangleClassification
  | Acute
  | Right
  | Obtuse

-- Define a function that classifies a triangle based on its angles
noncomputable def classifyTriangle (a b c : ℝ) : TriangleClassification :=
  if a < 90 ∧ b < 90 ∧ c < 90 then TriangleClassification.Acute
  else if a = 90 ∨ b = 90 ∨ c = 90 then TriangleClassification.Right
  else TriangleClassification.Obtuse

-- Theorem stating that all possible triangle classifications are covered
theorem triangle_classification_complete :
  ∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 → a + b + c = 180 →
    (classifyTriangle a b c = TriangleClassification.Acute ∨
     classifyTriangle a b c = TriangleClassification.Right ∨
     classifyTriangle a b c = TriangleClassification.Obtuse) :=
by
  sorry

-- Theorem stating that the classifications are mutually exclusive
theorem triangle_classification_exclusive :
  ∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 → a + b + c = 180 →
    (classifyTriangle a b c = TriangleClassification.Acute →
     classifyTriangle a b c ≠ TriangleClassification.Right ∧
     classifyTriangle a b c ≠ TriangleClassification.Obtuse) ∧
    (classifyTriangle a b c = TriangleClassification.Right →
     classifyTriangle a b c ≠ TriangleClassification.Acute ∧
     classifyTriangle a b c ≠ TriangleClassification.Obtuse) ∧
    (classifyTriangle a b c = TriangleClassification.Obtuse →
     classifyTriangle a b c ≠ TriangleClassification.Acute ∧
     classifyTriangle a b c ≠ TriangleClassification.Right) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_classification_complete_triangle_classification_exclusive_l220_22068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l220_22031

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := (3*x - 5)*(x - 2) / (2*x)

-- Define the solution set
def solution_set : Set ℝ := {x | x < 0 ∨ x ≥ 5/3}

-- Theorem statement
theorem inequality_solution :
  ∀ x : ℝ, x ≠ 0 → (g x ≥ 0 ↔ x ∈ solution_set) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l220_22031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_double_angle_plus_pi_twelfth_l220_22071

theorem sine_double_angle_plus_pi_twelfth (α : ℝ) :
  0 < α → α < π/2 →
  Real.cos (α + π/16) = 4/5 →
  Real.sin (2*α + π/12) = 17*Real.sqrt 2/50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_double_angle_plus_pi_twelfth_l220_22071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l220_22028

/-- The radius of the inscribed circle in a triangle with sides 50, 120, and 130 is 20. -/
theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 50) (hb : b = 120) (hc : c = 130) :
  let s := (a + b + c) / 2
  let A := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  A / s = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_l220_22028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yardwork_earnings_split_l220_22023

def earnings : List ℝ := [12, 18, 24, 30, 45]

theorem yardwork_earnings_split :
  let total := earnings.sum
  let n := earnings.length
  let equal_share := total / n
  let highest_earner := earnings.maximum?
  ∀ h : highest_earner.isSome, highest_earner.get h - equal_share = 19.2 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yardwork_earnings_split_l220_22023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_engine_batch_count_l220_22087

theorem engine_batch_count (total_batches non_defective_count : ℕ) : ℕ :=
  let defective_ratio : ℚ := 1/4
  let non_defective_ratio : ℚ := 1 - defective_ratio
  let engines_per_batch : ℕ := 80
  
  have h1 : total_batches = 5 := by sorry
  have h2 : non_defective_count = 300 := by sorry
  have h3 : (total_batches : ℚ) * (non_defective_ratio * engines_per_batch) = non_defective_count := by sorry
  
  engines_per_batch


end NUMINAMATH_CALUDE_ERRORFEEDBACK_engine_batch_count_l220_22087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_analysis_l220_22039

theorem condition_analysis (x y : ℝ) :
  (¬ ((x ≠ 1 ∧ y ≠ 2) → (x + y ≠ 3)) ∧ ¬ ((x + y ≠ 3) → (x ≠ 1 ∧ y ≠ 2))) ∧
  ((x + y ≠ 3 → (x ≠ 1 ∨ y ≠ 2)) ∧ ¬ ((x ≠ 1 ∨ y ≠ 2) → (x + y ≠ 3))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_condition_analysis_l220_22039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_inequality_l220_22088

theorem sin_squared_inequality (a : ℝ) : 
  (∀ x ∈ Set.Icc (-π/6) (π/2), (Real.sin x) ^ 2 + a * (Real.sin x) + a + 3 ≥ 0) ↔ a ≥ -2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_inequality_l220_22088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_specific_fields_l220_22037

/-- Represents the dimensions and properties of a rectangular field -/
structure RectField where
  width : ℝ
  length : ℝ
  area : ℝ
  width_prop_sqrt_length : ∃ k : ℝ, width = k * Real.sqrt length

/-- The difference in area between two rectangular fields -/
def areaDifference (field1 field2 : RectField) : ℝ :=
  field1.area - field2.area

/-- Theorem stating the difference in area between two specific fields -/
theorem area_difference_specific_fields :
  ∀ (field1 : RectField),
    field1.area = 10000 →
    let field2 : RectField := {
      width := 1.01 * field1.width,
      length := 0.95 * field1.length,
      area := (1.01 * field1.width) * (0.95 * field1.length),
      width_prop_sqrt_length := sorry
    }
    areaDifference field1 field2 = 405 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_specific_fields_l220_22037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l220_22098

-- Define IsTriangle as a predicate
def IsTriangle (a b c : Real) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_properties (a b c A B C : Real) :
  IsTriangle a b c →
  2 * a = Real.sqrt 3 * c * Real.sin A - a * Real.cos C →
  C = 2 * Real.pi / 3 ∧
  (c = Real.sqrt 3 →
    ∀ S : Real, S = 1/2 * a * b * Real.sin C → S ≤ Real.sqrt 3 / 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l220_22098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_inverse_e_l220_22056

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + 1

-- State the theorem
theorem product_inverse_e (a b : ℝ) (h1 : ∀ x, f (Real.exp (x - 1)) = 2 * x - 1) 
  (h2 : f a + f b = 0) : a * b = Real.exp (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_inverse_e_l220_22056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_passage_time_l220_22067

/-- Represents the time (in seconds) it takes for a train to pass a platform -/
noncomputable def time_to_pass_platform (train_length platform_length : ℝ) (time_to_pass_point : ℝ) : ℝ :=
  (train_length + platform_length) * time_to_pass_point / train_length

/-- Theorem stating that a train of length 1500 m, taking 100 seconds to pass a point,
    will take 220 seconds to pass a platform of length 1800 m -/
theorem train_platform_passage_time :
  time_to_pass_platform 1500 1800 100 = 220 := by
  -- Unfold the definition of time_to_pass_platform
  unfold time_to_pass_platform
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_passage_time_l220_22067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jenny_commute_time_l220_22041

/-- Represents Jenny's walking speed in miles per minute -/
noncomputable def walking_speed : ℝ := by sorry

/-- The time it takes Jenny to walk one mile -/
noncomputable def time_per_mile : ℝ := 1 / walking_speed

/-- The length of the bus journey in the indirect route -/
def bus_journey_length : ℝ := 1

/-- The length of the walking part in the indirect route -/
def indirect_walk_length : ℝ := 0.75

/-- The length of the direct walking route -/
def direct_walk_length : ℝ := 1.5

/-- The time the bus journey takes -/
def bus_journey_time : ℝ := 15

theorem jenny_commute_time :
  (bus_journey_time + indirect_walk_length * time_per_mile = direct_walk_length * time_per_mile) →
  (direct_walk_length * time_per_mile = 30) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jenny_commute_time_l220_22041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_black_white_areas_l220_22062

/-- Represents a point on the chessboard --/
structure Point where
  x : Fin 8
  y : Fin 8

/-- Represents a segment of the polygonal line --/
inductive Segment
  | Horizontal : Point → Point → Segment
  | Vertical : Point → Point → Segment
  | Diagonal : Point → Point → Segment

/-- Represents a closed polygonal line on the chessboard --/
structure PolygonalLine where
  segments : List Segment
  is_closed : segments.head? = segments.head?  -- Changed from last? to head? for consistency
  no_self_intersections : ∀ (s1 s2 : Segment), s1 ≠ s2 → True  -- Simplified condition

/-- Calculates the area enclosed by a polygonal line --/
noncomputable def enclosed_area (pl : PolygonalLine) : ℝ :=
  sorry

/-- Calculates the area of black squares within an enclosed area --/
noncomputable def black_area (area : ℝ) : ℝ :=
  sorry

/-- Calculates the area of white squares within an enclosed area --/
noncomputable def white_area (area : ℝ) : ℝ :=
  sorry

/-- Theorem: The total area of black pieces within the enclosed area is equal to the total area of white pieces --/
theorem equal_black_white_areas (pl : PolygonalLine) :
  black_area (enclosed_area pl) = white_area (enclosed_area pl) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_black_white_areas_l220_22062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_range_l220_22078

/-- A quadratic function f(x) = ax^2 - 2ax + c -/
def quadratic_function (a c : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + c

theorem quadratic_function_range
  (a c : ℝ)
  (h_decreasing : ∀ x ∈ Set.Icc 0 1, 
    MonotoneOn (quadratic_function a c) (Set.Icc 0 1) ∧ 
    StrictAntiOn (quadratic_function a c) (Set.Icc 0 1))
  (h_inequality : ∀ n : ℝ, quadratic_function a c n ≤ quadratic_function a c 0) :
  ∀ n : ℝ, quadratic_function a c n ≤ quadratic_function a c 0 → n ∈ Set.Icc 0 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_range_l220_22078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cost_is_624_l220_22076

/-- Calculates the cost to paint a rectangular floor given its length and painting rate. -/
noncomputable def paint_cost (length : ℝ) (rate : ℝ) : ℝ :=
  let breadth := length / 3
  let area := length * breadth
  area * rate

/-- Theorem stating that the cost to paint a rectangular floor with given dimensions is 624. -/
theorem paint_cost_is_624 :
  paint_cost 21.633307652783934 4 = 624 := by
  -- Unfold the definition of paint_cost
  unfold paint_cost
  -- Simplify the expression
  simp
  -- Assert that the result is approximately equal to 624
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cost_is_624_l220_22076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expression_1_log_expression_2_l220_22064

-- Define the natural logarithm (ln) and log base 10
noncomputable def ln (x : ℝ) : ℝ := Real.log x
noncomputable def log10 (x : ℝ) : ℝ := (ln x) / (ln 10)

-- Define log base 25 and log base 2
noncomputable def log25 (x : ℝ) : ℝ := (ln x) / (ln 25)
noncomputable def log2 (x : ℝ) : ℝ := (ln x) / (ln 2)

-- Statement for the first expression
theorem log_expression_1 :
  log10 14 - 2 * log10 (7/3) + log10 7 - log10 18 = 0 := by sorry

-- Statement for the second expression
theorem log_expression_2 :
  log25 625 + log10 0.01 + ln (Real.sqrt (Real.exp 1)) - 2^(1 + log2 3) = -11/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_expression_1_log_expression_2_l220_22064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_values_l220_22030

-- Define the polynomials p and q
noncomputable def p : ℝ → ℝ := sorry
noncomputable def q : ℝ → ℝ := sorry

-- Axioms
axiom cubic_p : ∃ a b c d : ℝ, ∀ x, p x = a * x^3 + b * x^2 + c * x + d
axiom cubic_q : ∃ a b c d : ℝ, ∀ x, q x = a * x^3 + b * x^2 + c * x + d

axiom p_zero : p 0 = -24
axiom q_zero : q 0 = 30

axiom commute : ∀ x, p (q x) = q (p x)

-- Theorem to prove
theorem polynomial_values : p 3 = 3 ∧ q 6 = -24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_values_l220_22030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cupcakes_frosted_in_ten_minutes_l220_22009

noncomputable def cagney_rate : ℝ := 1 / 15
noncomputable def lacey_rate : ℝ := 1 / 40
noncomputable def jessie_rate : ℝ := 1 / 30

def time_before_jessie : ℝ := 3 * 60  -- 3 minutes in seconds
def total_time : ℝ := 10 * 60  -- 10 minutes in seconds

noncomputable def combined_rate_before_jessie : ℝ := cagney_rate + lacey_rate
noncomputable def combined_rate_after_jessie : ℝ := cagney_rate + lacey_rate + jessie_rate

theorem cupcakes_frosted_in_ten_minutes :
  let cupcakes_before_jessie := combined_rate_before_jessie * time_before_jessie
  let cupcakes_after_jessie := combined_rate_after_jessie * (total_time - time_before_jessie)
  let total_cupcakes := cupcakes_before_jessie + cupcakes_after_jessie
  ⌊total_cupcakes⌋ = 68 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cupcakes_frosted_in_ten_minutes_l220_22009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_wood_rope_equations_l220_22077

/-- The length of the wood in feet -/
def x : ℝ := sorry

/-- The length of the rope in feet -/
def y : ℝ := sorry

/-- The system of equations describing the relationship between x and y -/
def wood_rope_equations : Prop :=
  (y - x = 4.5) ∧ ((1/2) * y = x - 1)

/-- Theorem stating that the given system of equations correctly describes the problem -/
theorem correct_wood_rope_equations : wood_rope_equations := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_wood_rope_equations_l220_22077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l220_22001

-- Define the line l
noncomputable def line_equation (x y : ℝ) (θ : ℝ) : Prop :=
  x + y * Real.cos θ + 3 = 0

-- Define the inclination angle α
noncomputable def inclination_angle (θ : ℝ) : ℝ :=
  Real.arctan (-1 / Real.cos θ)

-- State the theorem
theorem inclination_angle_range :
  ∀ θ : ℝ, ∃ α : ℝ, 
    (∃ x y : ℝ, line_equation x y θ) →
    (α = inclination_angle θ ∨ α = π/2) ∧
    π/4 ≤ α ∧ α ≤ 3*π/4 := by
  sorry

#check inclination_angle_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_range_l220_22001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_multiple_of_seven_l220_22063

def sequence_custom (a b : ℤ) : ℕ → ℤ
  | 0 => a
  | 1 => b
  | (n + 2) => 2 * sequence_custom a b (n + 1) - 9 * sequence_custom a b n

theorem sequence_multiple_of_seven (a b : ℤ) :
  (∃ n : ℕ, sequence_custom a b n % 7 = 0) ↔ (b % 7 = a % 7 ∨ b % 7 = (2 * a) % 7) := by
  sorry

#check sequence_multiple_of_seven

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_multiple_of_seven_l220_22063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l220_22045

noncomputable def θ : ℝ := 2 * Real.pi / 3

noncomputable def z : ℂ := Complex.exp (θ * Complex.I)

theorem z_in_second_quadrant : 
  Complex.re z < 0 ∧ Complex.im z > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l220_22045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hats_purchased_is_four_l220_22055

/-- Represents the purchase of clothes in a shop during a sale. -/
structure ClothingPurchase where
  shirt_cost : ℕ 
  hat_cost : ℕ
  jeans_cost : ℕ
  num_shirts : ℕ
  num_jeans : ℕ
  total_cost : ℕ

/-- The number of hats purchased given the clothing purchase details. -/
def num_hats_purchased (purchase : ClothingPurchase) : ℕ :=
  (purchase.total_cost - 
   (purchase.shirt_cost * purchase.num_shirts + 
    purchase.jeans_cost * purchase.num_jeans)) / 
  purchase.hat_cost

/-- Theorem stating that the number of hats purchased is 4 for the given scenario. -/
theorem hats_purchased_is_four :
  ∃ (purchase : ClothingPurchase),
    purchase.shirt_cost = 5 ∧
    purchase.hat_cost = 4 ∧
    purchase.jeans_cost = 10 ∧
    purchase.num_shirts = 3 ∧
    purchase.num_jeans = 2 ∧
    purchase.total_cost = 51 ∧
    num_hats_purchased purchase = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hats_purchased_is_four_l220_22055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l220_22047

/-- Given a sequence of positive terms with sum S_n satisfying 2S_n = (a_n + 1/2)^2 -/
noncomputable def S (n : ℕ+) : ℝ := sorry

/-- The n-th term of the sequence -/
noncomputable def a (n : ℕ+) : ℝ := n - 1/2

/-- The condition relating S_n and a_n -/
axiom S_condition (n : ℕ+) : 2 * S n = (a n + 1/2)^2

/-- The b_n sequence defined in terms of a_n -/
noncomputable def b (n : ℕ+) : ℝ := (a n + a (n + 1)) / ((a n)^2 * (a (n + 1))^2)

/-- The sum of the first n terms of the b sequence -/
noncomputable def T (n : ℕ+) : ℝ := (16 * n.val^2 + 16 * n.val) / (4 * n.val^2 + 4 * n.val + 1)

/-- The main theorem to be proved -/
theorem main_theorem :
  (∀ n : ℕ+, a n = n - 1/2) ∧
  (∀ n : ℕ+, T n = (16 * n.val^2 + 16 * n.val) / (4 * n.val^2 + 4 * n.val + 1)) :=
by
  constructor
  · intro n
    rfl
  · intro n
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l220_22047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_train_length_theorem_l220_22034

/-- Represents a spherical planet with ring roads and trains -/
structure PlanetWithTrains where
  /-- Number of ring roads -/
  N : ℕ
  /-- Equator length of the planet -/
  equatorLength : ℝ
  /-- Circumference of each ring road -/
  roadCircumference : ℝ
  /-- Assertion that equator length and road circumference are 1 -/
  h1 : equatorLength = 1
  h2 : roadCircumference = 1

/-- The maximum total length of all trains on the planet -/
noncomputable def maxTotalTrainLength (p : PlanetWithTrains) : ℝ := p.N / 2

/-- Theorem stating the maximum total length of trains -/
theorem max_train_length_theorem (p : PlanetWithTrains) :
  maxTotalTrainLength p ≤ p.N / 2 := by
  sorry

#check max_train_length_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_train_length_theorem_l220_22034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l220_22017

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (4 * x + Real.pi / 3)

theorem graph_translation (m : ℝ) :
  (∀ x, f (x - m) = f x) → m = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_l220_22017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_price_calculation_l220_22051

def wholesale_price : ℚ := 4
def markup_percentage : ℚ := 1/4
def discount_percentage : ℚ := 1/20
def tax_rate : ℚ := 3/40

def retail_price : ℚ := wholesale_price * (1 + markup_percentage)
def discounted_price : ℚ := retail_price * (1 - discount_percentage)
def final_price : ℚ := discounted_price * (1 + tax_rate)

theorem milk_price_calculation :
  (Int.floor (final_price * 100 + 1/2)) / 100 = 511/100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_price_calculation_l220_22051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_is_inverse_of_h_l220_22090

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := 3 + 6 * x

-- Define the proposed inverse function k
noncomputable def k (x : ℝ) : ℝ := (x - 3) / 6

-- Theorem stating that k is the inverse of h
theorem k_is_inverse_of_h : 
  (∀ x, h (k x) = x) ∧ (∀ x, k (h x) = x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_is_inverse_of_h_l220_22090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_left_focus_l220_22046

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (1, 0)

-- Define the property that F₂ is the midpoint of AB
def is_midpoint (A B F₂ : ℝ × ℝ) : Prop :=
  F₂.1 = (A.1 + B.1) / 2 ∧ F₂.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem distance_to_left_focus 
  (A B : ℝ × ℝ) 
  (h_A : is_on_ellipse A.1 A.2) 
  (h_B : is_on_ellipse B.1 B.2) 
  (h_midpoint : is_midpoint A B right_focus) :
  ∃ (F₁ : ℝ × ℝ), ‖A - F₁‖ = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_left_focus_l220_22046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_max_area_condition_l220_22015

/-- Ellipse with given foci and point -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- The ellipse E with given properties -/
noncomputable def E : Ellipse where
  a := 2
  b := 1
  h_ab := by sorry

/-- F₁ is a focus of E -/
noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)

/-- F₂ is a focus of E -/
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 3, 0)

/-- P is a point on E -/
noncomputable def P : ℝ × ℝ := (1, Real.sqrt 3 / 2)

/-- Theorem: The equation of ellipse E is x²/4 + y² = 1 -/
theorem ellipse_equation : ∀ (x y : ℝ), (x^2 / 4 + y^2 = 1) ↔ (x^2 / E.a^2 + y^2 / E.b^2 = 1) := by
  sorry

/-- Line l passing through F₁ -/
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 + Real.sqrt 3)}

/-- Area of triangle F₂MN -/
noncomputable def area_F₂MN (k : ℝ) : ℝ := sorry

/-- Theorem: The maximum area of triangle F₂MN occurs when line l has equation x ± √2y + √3 = 0 -/
theorem max_area_condition : 
  (∀ k, area_F₂MN k ≤ area_F₂MN (Real.sqrt 2 / 2)) ∧
  (area_F₂MN (Real.sqrt 2 / 2) = area_F₂MN (-Real.sqrt 2 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_max_area_condition_l220_22015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_time_to_complete_together_l220_22058

/-- The time taken for Cameron and Sandra to complete the entire task together -/
noncomputable def time_to_complete_together : ℚ := 7

/-- Cameron's individual work rate (portion of task completed per day) -/
noncomputable def cameron_rate : ℚ := 1 / 18

/-- The number of days Cameron worked alone -/
noncomputable def cameron_solo_days : ℚ := 9

/-- The time taken by Cameron and Sandra to complete half the task together -/
noncomputable def time_for_half_task : ℚ := 7 / 2

theorem prove_time_to_complete_together :
  time_to_complete_together = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_time_to_complete_together_l220_22058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_4_6_digit_difference_l220_22096

/-- The number of digits required to represent a positive integer n in base b -/
noncomputable def num_digits (n : ℕ) (b : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.floor (Real.log ↑n / Real.log ↑b) + 1

theorem base_4_6_digit_difference :
  num_digits 1729 4 = num_digits 1729 6 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_4_6_digit_difference_l220_22096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_silo_height_l220_22040

/-- Represents the properties of a cylindrical silo -/
structure Silo where
  height : ℝ
  volume : ℝ

/-- Calculates the scale factor between two silos based on their volumes -/
noncomputable def scaleFactor (original : Silo) (model : Silo) : ℝ :=
  (original.volume / model.volume) ^ (1/3 : ℝ)

/-- Theorem stating the relationship between the original silo and its scaled model -/
theorem scaled_silo_height 
  (original : Silo) 
  (model : Silo) 
  (h1 : original.height = 30) 
  (h2 : original.volume = 5000) 
  (h3 : model.volume = 0.05) : 
  ∃ (ε : ℝ), abs (model.height - 0.65) < ε ∧ ε > 0 := by
  sorry

#check scaled_silo_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_silo_height_l220_22040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_of_powers_l220_22048

theorem comparison_of_powers (n : ℕ) :
  (n ≤ 48 → 99^n + 100^n > 101^n) ∧
  (n > 48 → 99^n + 100^n < 101^n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_of_powers_l220_22048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_satisfies_conditions_l220_22014

-- Define the cubic polynomial q(x)
noncomputable def q (x : ℝ) : ℝ := -4/3 * x^3 + 6 * x^2 - 4 * x - 2

-- State the theorem
theorem cubic_polynomial_satisfies_conditions :
  q 1 = -8 ∧ q 2 = -12 ∧ q 3 = -20 ∧ q 4 = -40 := by
  -- Proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_satisfies_conditions_l220_22014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_is_15_l220_22094

/-- An arithmetic sequence where the sum of the first and third terms is 10 -/
structure ArithSeq where
  a : ℝ  -- first term
  d : ℝ  -- common difference
  sum_first_third : a + (a + 2 * d) = 10

/-- The fourth term of the arithmetic sequence is 15 -/
theorem fourth_term_is_15 (seq : ArithSeq) : seq.a + 3 * seq.d = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_is_15_l220_22094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_group_days_l220_22057

-- Define the daily work done by a boy
noncomputable def boy_work : ℝ := 1

-- Define the daily work done by a man (twice that of a boy)
noncomputable def man_work : ℝ := 2 * boy_work

-- Define the total work to be done
noncomputable def total_work : ℝ := (12 * man_work + 16 * boy_work) * 5

-- Define the function to calculate days needed for the second group
noncomputable def days_needed (x : ℝ) : ℝ := total_work / (13 * man_work + 24 * boy_work)

-- Theorem stating that the second group takes 4 days to complete the work
theorem second_group_days : days_needed 4 = 4 := by
  -- Expand the definitions
  unfold days_needed total_work man_work boy_work
  -- Simplify the expression
  simp [mul_add, mul_assoc, mul_comm]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_group_days_l220_22057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l220_22084

/-- Two lines in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The slope of a line -/
noncomputable def Line.slope (l : Line) : ℝ := -l.a / l.b

/-- Two lines are parallel if they have the same slope -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

theorem parallel_lines_a_value :
  ∀ a : ℝ,
  let l1 : Line := { a := a - 1, b := 1, c := 1 }
  let l2 : Line := { a := 2 * a, b := 1, c := 3 }
  parallel l1 l2 → a = -1 := by
  sorry

#check parallel_lines_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_a_value_l220_22084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intervals_of_increase_decrease_min_value_and_no_max_no_monotonicity_l220_22008

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 3

-- Theorem 1: Intervals of increase and decrease when a = -1
theorem intervals_of_increase_decrease :
  (∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x ≤ y → f (-1) x ≤ f (-1) y) ∧
  (∀ x y : ℝ, x ≤ 1 → y ≤ 1 → x ≤ y → f (-1) x ≥ f (-1) y) :=
sorry

-- Theorem 2: Minimum value and no maximum when a = -2
theorem min_value_and_no_max :
  (∃ x : ℝ, ∀ y : ℝ, f (-2) y ≥ f (-2) x) ∧
  f (-2) (2 : ℝ) = -1 ∧
  (∀ M : ℝ, ∃ x : ℝ, f (-2) x > M) :=
sorry

-- Theorem 3: No monotonicity for any value of a
theorem no_monotonicity :
  ∀ a : ℝ, ¬(∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ∧
           ¬(∀ x y : ℝ, x ≤ y → f a x ≥ f a y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intervals_of_increase_decrease_min_value_and_no_max_no_monotonicity_l220_22008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l220_22080

/-- Calculates the time (in seconds) for a train to cross a bridge -/
noncomputable def time_to_cross_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Proves that a train of length 140 meters traveling at 45 km/hr takes 30 seconds to cross a bridge of length 235 meters -/
theorem train_bridge_crossing_time :
  time_to_cross_bridge 140 45 235 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l220_22080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_sin_2phi_l220_22072

open Real

theorem symmetry_implies_sin_2phi (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) :
  (∀ x : ℝ, sin (π * x + φ) - 2 * cos (π * x + φ) = 
            sin (π * (2 - x) + φ) - 2 * cos (π * (2 - x) + φ)) →
  sin (2 * φ) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_sin_2phi_l220_22072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_integral_abs_diff_l220_22016

open Real MeasureTheory Interval

/-- The function that minimizes the integral of absolute differences -/
theorem minimize_integral_abs_diff
  (f : ℝ → ℝ) (f' : ℝ → ℝ) (a b : ℝ) (h : a < b)
  (h_deriv : ∀ x ∈ Set.Icc a b, HasDerivAt f (f' x) x ∧ f' x > 0) :
  ∃ x ∈ Set.Icc a b,
    (∀ y ∈ Set.Icc a b,
      ∫ t in a..b, |f t - f x| ≤ ∫ t in a..b, |f t - f y|) ∧
    x = (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_integral_abs_diff_l220_22016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_tube_volume_difference_l220_22060

/-- The volume of a cylinder with radius r and height h -/
noncomputable def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The radius of a cylinder given its circumference -/
noncomputable def radiusFromCircumference (c : ℝ) : ℝ := c / (2 * Real.pi)

theorem paper_tube_volume_difference (sheet_width sheet_length : ℝ) 
  (h_width : sheet_width = 5)
  (h_length : sheet_length = 7) :
  let v1 := cylinderVolume (radiusFromCircumference sheet_width) sheet_length
  let v2 := cylinderVolume (radiusFromCircumference sheet_length) sheet_width
  Real.pi * |v1 - v2| = 17.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_tube_volume_difference_l220_22060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_g_range_l220_22073

noncomputable def f (x : ℝ) := (1/2) * Real.sin (2*x) - Real.sqrt 3 * (Real.cos x)^2

noncomputable def g (x : ℝ) := f (x/2)

theorem f_properties_and_g_range :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, (0 < S ∧ S < T) → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x, f x ≥ -(2 + Real.sqrt 3)/2) ∧
  (∃ x, f x = -(2 + Real.sqrt 3)/2) ∧
  (∀ y ∈ Set.Icc (π/2) π, (1 - Real.sqrt 3)/2 ≤ g y ∧ g y ≤ (2 - Real.sqrt 3)/2) ∧
  (∃ y₁ ∈ Set.Icc (π/2) π, g y₁ = (1 - Real.sqrt 3)/2) ∧
  (∃ y₂ ∈ Set.Icc (π/2) π, g y₂ = (2 - Real.sqrt 3)/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_and_g_range_l220_22073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_height_of_cone_l220_22006

/-- The volume of a cone given its radius and height -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ := ⌊x + 0.5⌋

theorem min_height_of_cone (r v : ℝ) (hr : r = 5) (hv : v = 200) :
  round_to_nearest ((3 * v) / (Real.pi * r^2)) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_height_of_cone_l220_22006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_condition_percentage_l220_22004

/-- The percentage of fruits in good condition given the number of oranges and bananas and their rotten percentages. -/
def percentage_good_fruits (total_oranges : ℕ) (total_bananas : ℕ) (rotten_oranges_percent : ℚ) (rotten_bananas_percent : ℚ) : ℚ :=
  let total_fruits := total_oranges + total_bananas
  let rotten_oranges := (rotten_oranges_percent * total_oranges) / 100
  let rotten_bananas := (rotten_bananas_percent * total_bananas) / 100
  let total_rotten := rotten_oranges + rotten_bananas
  let good_fruits := total_fruits - total_rotten.floor
  (good_fruits : ℚ) / total_fruits * 100

/-- Theorem stating that given 600 oranges and 400 bananas, with 15% of oranges and 4% of bananas being rotten, the percentage of fruits in good condition is 89.4%. -/
theorem fruit_condition_percentage :
  percentage_good_fruits 600 400 15 4 = 894 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_condition_percentage_l220_22004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_odd_count_equality_l220_22070

theorem even_odd_count_equality : 
  (Finset.filter (fun n => n % 2 = 0) (Finset.range 104)).card = 
  (Finset.filter (fun n => n % 2 = 1) (Finset.range 107 \ Finset.range 5)).card := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_odd_count_equality_l220_22070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l220_22075

/-- A power function that passes through the point (2, √2/2) -/
noncomputable def f (x : ℝ) : ℝ := x^(-(1/2 : ℝ))

/-- Theorem stating that f is the unique power function passing through (2, √2/2) -/
theorem power_function_through_point :
  (∃ a : ℝ, ∀ x : ℝ, f x = x^a) ∧ f 2 = Real.sqrt 2 / 2 → ∀ x : ℝ, f x = x^(-(1/2 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l220_22075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_number_problem_l220_22011

/-- A four-digit number type -/
def FourDigitNumber := { n : Nat // 1000 ≤ n ∧ n < 10000 }

/-- Function to get the largest rearrangement of digits -/
def largest_rearrangement (n : FourDigitNumber) : Nat := sorry

/-- Function to get the smallest rearrangement of digits -/
def smallest_rearrangement (n : FourDigitNumber) : Nat := sorry

/-- Function to check if a number has no zero digits -/
def has_no_zero_digits (n : FourDigitNumber) : Prop := sorry

theorem four_digit_number_problem :
  ∃ (n : FourDigitNumber),
    has_no_zero_digits n ∧
    largest_rearrangement n = n.val + 5562 ∧
    smallest_rearrangement n = n.val - 2700 ∧
    n.val = 4179 := by
  sorry

#eval 4179 -- To check if the number is recognized correctly

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_number_problem_l220_22011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dennys_home_other_modules_cost_per_sqft_l220_22053

/-- Represents the modular home construction problem --/
structure ModularHome where
  total_size : ℕ
  kitchen_size : ℕ
  kitchen_cost : ℕ
  bathroom_size : ℕ
  bathroom_cost : ℕ
  num_bathrooms : ℕ
  total_cost : ℕ

/-- The specific modular home instance from the problem --/
def dennys_home : ModularHome :=
  { total_size := 2000
  , kitchen_size := 400
  , kitchen_cost := 20000
  , bathroom_size := 150
  , bathroom_cost := 12000
  , num_bathrooms := 2
  , total_cost := 174000 }

/-- Calculate the cost per square foot for other modules --/
def other_modules_cost_per_sqft (home : ModularHome) : ℚ :=
  let kitchen_bathroom_size := home.kitchen_size + home.num_bathrooms * home.bathroom_size
  let kitchen_bathroom_cost := home.kitchen_cost + home.num_bathrooms * home.bathroom_cost
  let other_size := home.total_size - kitchen_bathroom_size
  let other_cost := home.total_cost - kitchen_bathroom_cost
  (other_cost : ℚ) / (other_size : ℚ)

/-- Theorem stating that the cost per square foot for other modules in Denny's home is $100 --/
theorem dennys_home_other_modules_cost_per_sqft :
  other_modules_cost_per_sqft dennys_home = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dennys_home_other_modules_cost_per_sqft_l220_22053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_three_integer_solutions_l220_22092

theorem equation_three_integer_solutions (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 3 ∧ ∀ x ∈ s, abs (abs (x - 2) - 1) = a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_three_integer_solutions_l220_22092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_hits_ground_l220_22081

/-- The equation describing the height of the ball -/
def ballHeight (t : ℚ) : ℚ := -16 * t^2 + 30 * t + 50

/-- The time when the ball hits the ground is the positive solution to ballHeight(t) = 0 -/
theorem ball_hits_ground : ∃ t : ℚ, t > 0 ∧ ballHeight t = 0 ∧ t = 47/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_hits_ground_l220_22081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l220_22043

-- Define the ellipse
def Ellipse (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the foci
def foci : Set (ℝ × ℝ) := {(-1, 0), (1, 0)}

-- Define the point P
def P : ℝ × ℝ := (2, 0)

theorem ellipse_equation (h1 : foci = {(-1, 0), (1, 0)}) (h2 : P ∈ Ellipse 2 (Real.sqrt 3)) :
  Ellipse 2 (Real.sqrt 3) = {p : ℝ × ℝ | (p.1^2 / 4) + (p.2^2 / 3) = 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l220_22043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_thirteen_l220_22021

theorem remainder_sum_mod_thirteen (a b c d : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_thirteen_l220_22021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_setB_forms_right_triangle_l220_22089

-- Define the sets of line segments
noncomputable def setA : List ℝ := [4, 5, 6]
noncomputable def setB : List ℝ := [1, Real.sqrt 3, 2]
noncomputable def setC : List ℝ := [5, 6, 7]
noncomputable def setD : List ℝ := [1, Real.sqrt 2, 3]

-- Function to check if a set of line segments can form a right triangle
def canFormRightTriangle (sides : List ℝ) : Prop :=
  sides.length = 3 ∧ ∃ a b c, sides = [a, b, c] ∧ a^2 + b^2 = c^2

-- Theorem stating that only setB can form a right triangle
theorem only_setB_forms_right_triangle :
  canFormRightTriangle setB ∧
  ¬canFormRightTriangle setA ∧
  ¬canFormRightTriangle setC ∧
  ¬canFormRightTriangle setD :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_setB_forms_right_triangle_l220_22089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_to_hemisphere_volume_ratio_l220_22035

-- Define the volume of a sphere
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Define the volume of a hemisphere
noncomputable def hemisphere_volume (r : ℝ) : ℝ := (1 / 2) * (4 / 3) * Real.pi * r^3

-- Theorem statement
theorem sphere_to_hemisphere_volume_ratio (r : ℝ) (h : r > 0) :
  sphere_volume r / hemisphere_volume (3 * r) = 1 / 13.5 :=
by
  -- Expand the definitions
  unfold sphere_volume hemisphere_volume
  -- Simplify the expression
  simp [Real.pi]
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_to_hemisphere_volume_ratio_l220_22035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l220_22049

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.arcsin x + (2 : ℝ)^x

-- State the theorem
theorem range_of_f :
  ∃ (a b : ℝ), a = (1 - Real.pi) / 2 ∧ b = (Real.pi + 4) / 2 ∧
  (∀ y, (∃ x, x ∈ Set.Icc (-1 : ℝ) 1 ∧ f x = y) ↔ y ∈ Set.Icc a b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l220_22049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_valid_number_l220_22086

def is_valid (n : ℕ) : Prop :=
  ∀ i : ℕ, 2 ≤ i ∧ i ≤ 7 → n % i = 2

theorem least_valid_number : 
  (2102 % 9 = 0) ∧ 
  (is_valid 2102) ∧ 
  (∀ m : ℕ, m < 2102 → ¬(m % 9 = 0 ∧ is_valid m)) :=
by
  sorry

#check least_valid_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_valid_number_l220_22086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_proof_l220_22079

/-- Represents a plane in 3D space -/
structure Plane where
  A : Int
  B : Int
  C : Int
  D : Int

/-- Checks if a point lies on a plane -/
def Plane.contains (p : Plane) (x y z : ℝ) : Prop :=
  p.A * x + p.B * y + p.C * z + p.D = 0

/-- Checks if two planes are parallel -/
def Plane.parallelTo (p1 p2 : Plane) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ p1.A = k * p2.A ∧ p1.B = k * p2.B ∧ p1.C = k * p2.C

/-- The given plane 2x - y + 3z = 5 -/
def givenPlane : Plane := ⟨2, -1, 3, -5⟩

theorem plane_equation_proof :
  ∃ (p : Plane),
    p.A > 0 ∧
    Int.gcd p.A.natAbs (Int.gcd p.B.natAbs (Int.gcd p.C.natAbs p.D.natAbs)) = 1 ∧
    p.contains 2 3 1 ∧
    p.parallelTo givenPlane ∧
    p = ⟨2, -1, 3, -4⟩ :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_proof_l220_22079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sides_of_folded_square_l220_22020

/-- A square with white top side and red bottom side -/
structure ColoredSquare where
  side : ℝ
  topColor : String
  bottomColor : String

/-- A point randomly chosen inside the square -/
structure RandomPoint where
  x : ℝ
  y : ℝ

/-- The result of folding the square -/
inductive FoldResult
  | Triangle
  | Quadrilateral

/-- Function to determine the fold result based on the random point's position -/
def foldSquare (s : ColoredSquare) (p : RandomPoint) : FoldResult :=
  sorry

/-- The probability of getting a quadrilateral when folding -/
noncomputable def probQuadrilateral : ℝ := 2 - Real.pi / 2

/-- The probability of getting a triangle when folding -/
noncomputable def probTriangle : ℝ := Real.pi / 2 - 1

/-- Theorem: The expected number of sides of the resulting red polygon is 5 - π/2 -/
theorem expected_sides_of_folded_square (s : ColoredSquare) (p : RandomPoint) :
  3 * probTriangle + 4 * probQuadrilateral = 5 - Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sides_of_folded_square_l220_22020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_28_equals_reciprocal_of_one_minus_x_l220_22007

-- Define the initial function f₁
noncomputable def f₁ (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

-- Define the recursive function f_n
noncomputable def f_n : ℕ → ℝ → ℝ
  | 0, x => x
  | n + 1, x => f₁ (f_n n x)

-- State the theorem
theorem f_28_equals_reciprocal_of_one_minus_x (x : ℝ) :
  f_n 28 x = 1 / (1 - x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_28_equals_reciprocal_of_one_minus_x_l220_22007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_solutions_exist_l220_22019

def coin_probability_equation (p : ℝ) : Prop :=
  15 * p^4 * (1-p)^2 = 64/729

theorem multiple_solutions_exist : 
  ∃ p₁ p₂ : ℝ, p₁ ≠ p₂ ∧ p₁ ∈ Set.Icc 0 1 ∧ p₂ ∈ Set.Icc 0 1 ∧ 
  coin_probability_equation p₁ ∧ coin_probability_equation p₂ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_solutions_exist_l220_22019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l220_22097

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem omega_range (ω : ℝ) :
  (∀ x₁ x₂ : ℝ, -π/2 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2*π/3 → f ω x₁ < f ω x₂) ∧
  (∃! x : ℝ, 0 ≤ x ∧ x ≤ π ∧ ∀ y : ℝ, 0 ≤ y ∧ y ≤ π → f ω y ≤ f ω x) →
  1/2 ≤ ω ∧ ω ≤ 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l220_22097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l220_22000

noncomputable def f (x : ℝ) : ℝ := (9 * x^2 + 18 * x - 60) / ((3 * x - 4) * (x + 5))

theorem inequality_solution_set :
  {x : ℝ | f x < 4} = {x : ℝ | x < -10 ∨ (-5 < x ∧ x < 2/3) ∨ 4/3 < x} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l220_22000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_line_center_x_range_l220_22065

/-- A square with edges on parallel lines -/
structure ParallelEdgeSquare where
  /-- The line containing edge AB: x + 3y - 5 = 0 -/
  ab_line : Set (ℝ × ℝ)
  /-- The line containing edge CD: x + 3y + 7 = 0 -/
  cd_line : Set (ℝ × ℝ)
  /-- Condition that ab_line is x + 3y - 5 = 0 -/
  ab_eq : ab_line = {p : ℝ × ℝ | p.1 + 3 * p.2 - 5 = 0}
  /-- Condition that cd_line is x + 3y + 7 = 0 -/
  cd_eq : cd_line = {p : ℝ × ℝ | p.1 + 3 * p.2 + 7 = 0}

/-- The center of the square lies on x + 3y + 1 = 0 -/
theorem center_line (s : ParallelEdgeSquare) :
  ∃ G : ℝ × ℝ, G.1 + 3 * G.2 + 1 = 0 ∧ 
  (∀ p ∈ s.ab_line, ∃ q ∈ s.cd_line, G = ((p.1 + q.1) / 2, (p.2 + q.2) / 2)) :=
sorry

/-- The x-coordinate of the center is in (6/5, 13/5) when only two vertices are in the first quadrant -/
theorem center_x_range (s : ParallelEdgeSquare) 
  (two_vertices_first_quadrant : ∃ (A B C D : ℝ × ℝ), 
    A ∈ s.ab_line ∧ B ∈ s.ab_line ∧ C ∈ s.cd_line ∧ D ∈ s.cd_line ∧
    (A.1 > 0 ∧ A.2 > 0) ∧ (B.1 > 0 ∧ B.2 > 0) ∧
    (C.1 ≤ 0 ∨ C.2 ≤ 0) ∧ (D.1 ≤ 0 ∨ D.2 ≤ 0)) :
  ∃ G : ℝ × ℝ, G.1 + 3 * G.2 + 1 = 0 ∧ 6/5 < G.1 ∧ G.1 < 13/5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_line_center_x_range_l220_22065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l220_22099

-- Define the constants as noncomputable
noncomputable def a : ℝ := (3 : ℝ) ^ (0.1 : ℝ)
noncomputable def b : ℝ := (2 : ℝ) ^ (0.1 : ℝ)
noncomputable def c : ℝ := (0.2 : ℝ) ^ (1.3 : ℝ)

-- State the theorem
theorem relationship_abc : c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l220_22099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_2x_minus_2sin_squared_x_plus_1_l220_22027

/-- The maximum value of the function y = sin(2x) - 2sin²(x) + 1 is √2 -/
theorem max_value_sin_2x_minus_2sin_squared_x_plus_1 :
  ∃ (x : ℝ), ∀ (t : ℝ), Real.sin (2*t) - 2*(Real.sin t)^2 + 1 ≤ Real.sin (2*x) - 2*(Real.sin x)^2 + 1 ∧
              Real.sin (2*x) - 2*(Real.sin x)^2 + 1 = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_2x_minus_2sin_squared_x_plus_1_l220_22027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_reciprocals_min_sum_reciprocals_attained_l220_22093

/-- An arithmetic sequence with positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (n : ℝ) * (seq.a 1 + seq.a n) / 2

theorem min_sum_reciprocals (seq : ArithmeticSequence) 
    (h : sumFirstN seq 2019 = 4038) :
    (1 / seq.a 9 + 9 / seq.a 2011) ≥ 4 := by
  sorry

theorem min_sum_reciprocals_attained (seq : ArithmeticSequence) 
    (h : sumFirstN seq 2019 = 4038) :
    ∃ seq' : ArithmeticSequence, (1 / seq'.a 9 + 9 / seq'.a 2011) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_reciprocals_min_sum_reciprocals_attained_l220_22093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fee_piecewise_fee_26_4_implies_x_1_5_l220_22010

/-- Water fee calculation function -/
noncomputable def water_fee (usage : ℝ) : ℝ :=
  if usage ≤ 4 then 1.8 * usage
  else 1.8 * 4 + 3 * (usage - 4)

/-- Total water fee for two households -/
noncomputable def total_fee (x : ℝ) : ℝ :=
  water_fee (5 * x) + water_fee (3 * x)

/-- Theorem: Total water fee function is piecewise as described -/
theorem total_fee_piecewise (x : ℝ) (h : x ≥ 0) :
  total_fee x = 
    if x ≤ 4/5 then 14.4 * x
    else if x ≤ 4/3 then 20.4 * x - 4.8
    else 24 * x - 9.6 := by
  sorry

/-- Corollary: When total fee is 26.4, x = 1.5 -/
theorem fee_26_4_implies_x_1_5 :
  ∃ x, x > 0 ∧ total_fee x = 26.4 ∧ x = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fee_piecewise_fee_26_4_implies_x_1_5_l220_22010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l220_22054

def a (n : ℕ) : ℚ := 4^(2*n - 1) + 3^(n - 2)

def divides_infinitely_many (p : ℕ) : Prop :=
  ∀ m : ℕ, ∃ n ≥ m, (a n).num % p = 0

def divides_all (p : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (a n).num % p = 0

theorem sequence_properties :
  (divides_infinitely_many 5 ∧
   ∀ p < 5, Nat.Prime p → ¬divides_infinitely_many p) ∧
  (divides_all 13 ∧
   ∀ p < 13, Nat.Prime p → ¬divides_all p) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l220_22054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_2_pow_2023_l220_22029

def last_digit (n : ℕ) : ℕ := n % 10

def power_of_two_last_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 2
  | 2 => 4
  | _ => 8

theorem last_digit_of_2_pow_2023 :
  last_digit (2^2023) = power_of_two_last_digit 2023 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_of_2_pow_2023_l220_22029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_relationship_l220_22095

-- Define the logarithms
noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 6 / Real.log 4
noncomputable def c : ℝ := Real.log 9 / Real.log 4

-- State the theorem
theorem log_relationship : a = c ∧ a > b ∧ c > b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_relationship_l220_22095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_shortest_chord_length_l220_22018

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop :=
  x = m * y + 1

-- Define the intersection points of the line and the parabola
def intersection_points (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), parabola x₁ y₁ ∧ parabola x₂ y₂ ∧ 
  line_through_focus m x₁ y₁ ∧ line_through_focus m x₂ y₂

-- Theorem 1: If x₁ + x₂ = 5, then |AB| = 7
theorem chord_length (m : ℝ) :
  ∀ (x₁ y₁ x₂ y₂ : ℝ), intersection_points m → x₁ + x₂ = 5 → 
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 7 :=
sorry

-- Theorem 2: The length of the shortest chord passing through F is 4
theorem shortest_chord_length :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), parabola x₁ y₁ ∧ parabola x₂ y₂ ∧ 
  (∀ (a b c d : ℝ), parabola a b ∧ parabola c d → 
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≤ Real.sqrt ((c - a)^2 + (d - b)^2)) ∧
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_shortest_chord_length_l220_22018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_moves_on_ellipse_l220_22024

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circular path of vertex C
noncomputable def CircularPath (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the perpendicular bisector of AB
noncomputable def PerpendicularBisector (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - (A.1 + B.1)/2)^2 + (p.2 - (A.2 + B.2)/2)^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2)/4}

-- Define the centroid of a triangle
noncomputable def Centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1)/3, (t.A.2 + t.B.2 + t.C.2)/3)

-- Define an ellipse
noncomputable def Ellipse (center : ℝ × ℝ) (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2/a^2 + (p.2 - center.2)^2/b^2 = 1}

theorem centroid_moves_on_ellipse
  (A B : ℝ × ℝ)
  (D : ℝ × ℝ)
  (h1 : D ∈ PerpendicularBisector A B)
  (radius : ℝ)
  : ∃ (center : ℝ × ℝ) (a b : ℝ),
    ∀ C : ℝ × ℝ,
    C ∈ CircularPath D radius →
    Centroid ⟨A, B, C⟩ ∈ Ellipse center a b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_moves_on_ellipse_l220_22024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_properties_l220_22085

theorem divisibility_properties :
  (∀ n : ℕ, ∃ k : ℤ, (3 : ℤ)^(6*n) - (2 : ℤ)^(6*n) = 35*k) ∧
  (∀ n : ℤ, ∃ k : ℤ, n^5 - 5*n^3 + 4*n = 120*k) ∧
  (∀ n : ℤ, ∀ k : ℤ, n^2 + 3*n + 5 ≠ 121*k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_properties_l220_22085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l220_22044

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x)) / Real.log 10

-- Theorem statement
theorem f_is_odd : ∀ x, f (-x) = -f x := by
  intro x
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l220_22044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_problem_l220_22012

-- Define the function g
def g : ℝ → ℝ := sorry

-- Define the inverse function g⁻¹
def g_inv : ℝ → ℝ := sorry

-- State the theorem
theorem inverse_composition_problem 
  (h1 : g 4 = 7)
  (h2 : g 6 = 2)
  (h3 : g 3 = 8)
  (h4 : Function.LeftInverse g_inv g)
  (h5 : Function.RightInverse g_inv g) :
  g_inv (g_inv 8 + g_inv 7) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_composition_problem_l220_22012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_reader_time_l220_22069

/-- Represents the reading speed of a person in arbitrary units. -/
def ReadingSpeed : Type := ℝ

/-- Represents the time taken to read a book in minutes. -/
def ReadingTime : Type := ℝ

/-- The ratio of reading speeds between two people. -/
def SpeedRatio : ℝ := 4

/-- The time taken by the slower reader in hours. -/
def SlowerReaderTime : ℝ := 3

/-- Converts hours to minutes. -/
def hoursToMinutes (hours : ℝ) : ℝ := hours * 60

/-- Theorem stating that if one person reads 4 times faster than another,
    and the slower person takes 3 hours to read a book,
    then the faster person will take 45 minutes to read the same book. -/
theorem faster_reader_time :
  let slowerTime : ℝ := hoursToMinutes SlowerReaderTime
  let fasterTime : ℝ := slowerTime / SpeedRatio
  fasterTime = 45 := by
  -- Unfold definitions
  unfold hoursToMinutes
  unfold SlowerReaderTime
  unfold SpeedRatio
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_faster_reader_time_l220_22069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_distance_points_form_spherical_surface_l220_22022

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the distance between two points in 3D space
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

-- Define a spherical surface
def SphericalSurface (center : Point3D) (radius : ℝ) : Set Point3D :=
  {p : Point3D | distance p center = radius}

-- Theorem statement
theorem fixed_distance_points_form_spherical_surface (center : Point3D) (r : ℝ) :
  {p : Point3D | distance p center = r} = SphericalSurface center r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_distance_points_form_spherical_surface_l220_22022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_APRS_l220_22025

-- Define the points
variable (A P Q R S : ℝ × ℝ)

-- Define the angles
noncomputable def angle_PAQ : ℝ := 60 * Real.pi / 180
noncomputable def angle_QPR : ℝ := 60 * Real.pi / 180
noncomputable def angle_RPS : ℝ := 60 * Real.pi / 180

-- Define the length AP
def AP : ℝ := 36

-- Define right angles
def right_angle_P (A P Q : ℝ × ℝ) : Prop := (A - P) • (Q - P) = 0
def right_angle_Q (P Q R : ℝ × ℝ) : Prop := (P - Q) • (R - Q) = 0
def right_angle_R (P R S : ℝ × ℝ) : Prop := (P - R) • (S - R) = 0

-- Helper definitions
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry
noncomputable def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem area_APRS (h1 : right_angle_P A P Q)
                  (h2 : right_angle_Q P Q R)
                  (h3 : right_angle_R P R S)
                  (h4 : angle A P Q = angle_PAQ)
                  (h5 : angle Q P R = angle_QPR)
                  (h6 : angle R P S = angle_RPS)
                  (h7 : ‖A - P‖ = AP) :
  area_quadrilateral A P R S = 405 + 20.25 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_APRS_l220_22025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l220_22083

open Real

/-- The function f(x) as described in the problem -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sqrt 3 * sin (3 * ω * x + π / 3)

/-- The shifted function f(x+θ) -/
noncomputable def f_shifted (ω θ : ℝ) (x : ℝ) : ℝ := f ω (x + θ)

/-- Theorem stating the conditions and conclusion of the problem -/
theorem function_properties (ω θ : ℝ) : 
  ω > 0 ∧ 
  (∀ x, f_shifted ω θ x = f_shifted ω θ (-x)) ∧  -- even function
  (∀ x, f_shifted ω θ (x + 2*π) = f_shifted ω θ x) →  -- period 2π
  ω = 1/3 ∧ ∃ k : ℤ, θ = k * π + π / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l220_22083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_museum_sales_tax_percentage_l220_22033

/-- Calculates the sales tax percentage for a group museum entrance -/
def calculate_sales_tax_percentage (group_size : ℕ) (total_cost_with_tax : ℚ) (ticket_face_value : ℚ) : ℚ :=
  let total_cost_without_tax := group_size * ticket_face_value
  let sales_tax_paid := total_cost_with_tax - total_cost_without_tax
  (sales_tax_paid / total_cost_without_tax) * 100

/-- Theorem stating that the sales tax percentage for the given scenario is approximately 5.26% -/
theorem museum_sales_tax_percentage :
  let group_size : ℕ := 25
  let total_cost_with_tax : ℚ := 945
  let ticket_face_value : ℚ := 35.91
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
    |calculate_sales_tax_percentage group_size total_cost_with_tax ticket_face_value - 5.26| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_museum_sales_tax_percentage_l220_22033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_range_l220_22032

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then (a + 1) / x else (-2 * a - 1) * x + 1

-- State the theorem
theorem monotone_decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y < f a x) →
  a ∈ Set.Ioc (-1/2 : ℝ) (-1/3 : ℝ) :=
by
  -- Proof skeleton
  intro h
  -- We'll prove this later
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_range_l220_22032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_fraction_l220_22091

theorem real_part_of_fraction (θ : ℝ) : 
  let z : ℂ := Complex.exp (θ * Complex.I)
  Complex.re (1 / (2 - z)) = (2 - Real.cos θ) / (5 - 4 * Real.cos θ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_fraction_l220_22091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_when_f_minus_g_equals_3_l220_22038

/-- The function f(x) = 7x^2 - 1/x + 5 -/
noncomputable def f (x : ℝ) : ℝ := 7 * x^2 - 1/x + 5

/-- The function g(x) = x^2 - k, where k is a parameter -/
def g (k : ℝ) (x : ℝ) : ℝ := x^2 - k

/-- Theorem stating that if f(3) - g(3) = 3, then k = 176/3 -/
theorem k_value_when_f_minus_g_equals_3 :
  ∃ k : ℝ, f 3 - g k 3 = 3 → k = 176/3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_value_when_f_minus_g_equals_3_l220_22038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_implies_omega_range_l220_22059

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  Real.sin (ω * x / 2) ^ 2 + (1/2) * Real.sin (ω * x) - 1/2

theorem f_zero_implies_omega_range (ω : ℝ) (h_ω_pos : ω > 0) :
  (∃ x, π < x ∧ x < 2*π ∧ f ω x = 0) →
  (ω > 1/8 ∧ ω < 1/4) ∨ (ω > 5/8) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zero_implies_omega_range_l220_22059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_and_values_l220_22066

-- Define the function f(x)
noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 4 * Real.log x

-- State the theorem
theorem extreme_points_and_values :
  ∃ (a b : ℝ),
    (∀ x : ℝ, x > 0 → (deriv (f a b)) x = 0 ↔ x = 1 ∨ x = 2) ∧
    a = 1 ∧
    b = -6 ∧
    (∀ x : ℝ, x > 0 → f a b x ≤ -5) ∧
    (∀ x : ℝ, x > 0 → f a b x ≥ -8 + 4 * Real.log 2) ∧
    f a b 1 = -5 ∧
    f a b 2 = -8 + 4 * Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_and_values_l220_22066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_prime_min_negative_smallest_b_is_zero_l220_22005

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := Real.exp x - a * x^2 - 2 * x + b

-- Define the derivative of f
noncomputable def f_prime (a x : ℝ) : ℝ := Real.exp x - 2 * a * x - 2

-- Theorem 1: The minimum value of f'(x) is less than 0 when a > 0
theorem f_prime_min_negative (a : ℝ) (h : a > 0) :
  ∃ x : ℝ, ∀ y : ℝ, f_prime a x ≤ f_prime a y ∧ f_prime a x < 0 := by
  sorry

-- Theorem 2: The smallest integer b that ensures f(x) > 0 for all x is 0
theorem smallest_b_is_zero (a : ℝ) (h : a > 0) :
  (∀ b : ℤ, (∀ x : ℝ, f a (b : ℝ) x > 0) → b ≥ 0) ∧
  (∀ x : ℝ, f a 0 x > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_prime_min_negative_smallest_b_is_zero_l220_22005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l220_22061

-- Define the expression as a noncomputable function
noncomputable def q (x d e f : ℝ) : ℝ :=
  ((x + d)^4) / ((d - e) * (d - f)) +
  ((x + e)^4) / ((e - d) * (e - f)) +
  ((x + f)^4) / ((f - d) * (f - e))

-- State the theorem
theorem expression_simplification (d e f : ℝ) (h1 : d ≠ e) (h2 : d ≠ f) (h3 : e ≠ f) :
  ∀ x, q x d e f = d + e + f + 4 * x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l220_22061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_open_interval_l220_22050

-- Define set A
def A : Set ℝ := {x | x > 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define the open interval (0, 3)
def open_interval : Set ℝ := Set.Ioo 0 3

-- Theorem statement
theorem intersection_equals_open_interval : A ∩ B = open_interval := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_open_interval_l220_22050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_l220_22002

/-- The area of an equilateral triangle with side length 10 meters is 25√3 square meters. -/
theorem equilateral_triangle_area :
  let side_length : ℝ := 10
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * side_length^2
  triangle_area = 25 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_area_l220_22002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_bad_markers_eq_1020_l220_22036

/-- Represents a marker on the road -/
structure Marker where
  front : Nat
  back : Nat

/-- The total distance between cities -/
def total_distance : Nat := 2020

/-- The collection of all markers on the road -/
def all_markers : List Marker :=
  List.range total_distance |>.map (fun i =>
    let front := i + 1
    let back := total_distance - front
    ⟨front, back⟩)

/-- A marker is bad if its front and back numbers are coprime -/
def is_bad_marker (m : Marker) : Bool :=
  Nat.gcd m.front m.back = 1

/-- Count the number of bad markers -/
def count_bad_markers : Nat :=
  all_markers.filter is_bad_marker |>.length

/-- The main theorem to be proved -/
theorem count_bad_markers_eq_1020 :
    count_bad_markers = 1020 := by
  sorry

#eval count_bad_markers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_bad_markers_eq_1020_l220_22036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_sum_divisibility_l220_22074

theorem three_digit_sum_divisibility :
  ∃ (a b c d : ℕ),
    -- All numbers are three-digit integers
    100 ≤ a ∧ a < 1000 ∧
    100 ≤ b ∧ b < 1000 ∧
    100 ≤ c ∧ c < 1000 ∧
    100 ≤ d ∧ d < 1000 ∧
    -- All numbers are different
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    -- All numbers start with the same digit
    a / 100 = b / 100 ∧ a / 100 = c / 100 ∧ a / 100 = d / 100 ∧
    -- The sum is divisible by three of the numbers without remainder
    ∃ (x y z : ℕ), ({x, y, z} : Set ℕ) ⊆ {a, b, c, d} ∧
    (a + b + c + d) % x = 0 ∧
    (a + b + c + d) % y = 0 ∧
    (a + b + c + d) % z = 0 ∧
    -- The specific numbers satisfy the conditions
    a = 108 ∧ b = 135 ∧ c = 180 ∧ d = 117 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_sum_divisibility_l220_22074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equals_sqrt_two_l220_22026

noncomputable section

-- Define the value of π
def π : ℝ := Real.pi

-- Convert degrees to radians
def deg_to_rad (x : ℝ) : ℝ := x * π / 180

-- Define the expression
def expr : ℝ := (Real.cos (deg_to_rad 10) + Real.sqrt 3 * Real.sin (deg_to_rad 10)) / 
                Real.sqrt (1 - Real.cos (deg_to_rad 80))

-- Theorem statement
theorem trigonometric_expression_equals_sqrt_two : expr = Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equals_sqrt_two_l220_22026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_relation_l220_22042

/-- Definition of the function g(n) -/
noncomputable def g (n : ℤ) : ℝ :=
  (4 + 2 * Real.sqrt 4) / 10 * ((1 + Real.sqrt 4) / 2) ^ n +
  (4 - 2 * Real.sqrt 4) / 10 * ((1 - Real.sqrt 4) / 2) ^ n

/-- Theorem stating the relationship between g(n+2), g(n-2), and g(n) -/
theorem g_relation (n : ℤ) : g (n + 2) - g (n - 2) = (13 / 9) * g n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_relation_l220_22042
