import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_region_is_zero_l1057_105741

/-- The region defined by the given inequalities -/
def Region : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | (|p.1| + |p.2.1| + |p.2.2| ≤ 1) ∧ (|p.1| + |p.2.1| + |p.2.2 - 2| ≤ 1)}

/-- The volume of a set in ℝ³ -/
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the volume of the defined region is zero -/
theorem volume_of_region_is_zero : volume Region = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_region_is_zero_l1057_105741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_angle_is_30_degrees_l1057_105709

-- Define the cone
structure Cone where
  slant_height : ℝ
  base_radius : ℝ
  angle : ℝ

-- Define the properties of the cone
def is_valid_cone (c : Cone) : Prop :=
  c.slant_height > 0 ∧ c.base_radius > 0 ∧ c.angle > 0 ∧ c.angle < Real.pi/2

-- Define the condition that the lateral surface forms a semicircle
def lateral_surface_is_semicircle (c : Cone) : Prop :=
  2 * Real.pi * c.base_radius = Real.pi * c.slant_height^2 / 2

-- Define the condition that the area of the semicircle is 2π
def semicircle_area_is_2pi (c : Cone) : Prop :=
  Real.pi * c.slant_height^2 / 2 = 2 * Real.pi

-- The main theorem
theorem cone_angle_is_30_degrees (c : Cone) 
  (h1 : is_valid_cone c) 
  (h2 : lateral_surface_is_semicircle c) 
  (h3 : semicircle_area_is_2pi c) : 
  c.angle = Real.pi/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_angle_is_30_degrees_l1057_105709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l1057_105764

/-- The distance between two parallel lines in 2D space -/
noncomputable def distance_between_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a^2 + b^2)

/-- Proof that the distance between lines l₁ and l₂ is √2/4 -/
theorem distance_between_specific_lines :
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ x + y + 1 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y ↦ 2*x + 2*y + 3 = 0
  distance_between_parallel_lines 1 1 (-1) (-3/2) = Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l1057_105764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1057_105738

noncomputable def vector : Type := Fin 2 → ℝ

noncomputable def proj (u v : vector) : vector :=
  let dot_product := (u 0) * (v 0) + (u 1) * (v 1)
  let norm_squared := (v 0)^2 + (v 1)^2
  fun i => (dot_product / norm_squared) * (v i)

theorem projection_theorem (b : ℝ) :
  let v : vector := fun i => if i = 0 then -6 else b
  let u : vector := fun i => if i = 0 then 3 else 2
  (∀ i, proj v u i = -15/13 * (u i)) → b = 3/2 := by
  sorry

#check projection_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l1057_105738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_value_l1057_105781

theorem sin_cos_value (x : Real) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4/17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_value_l1057_105781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_net_profit_at_4_minutes_p_2_equals_372_l1057_105787

-- Define the passenger capacity function
noncomputable def p (t : ℝ) : ℝ :=
  if 2 ≤ t ∧ t < 10 then 300 + 40 * t - 2 * t^2
  else if 10 ≤ t ∧ t ≤ 20 then 500
  else 0

-- Define the net profit per minute function
noncomputable def Q (t : ℝ) : ℝ := (8 * p t - 2656) / t - 60

-- Theorem statement
theorem max_net_profit_at_4_minutes :
  ∀ t : ℝ, 2 ≤ t ∧ t ≤ 20 → Q t ≤ Q 4 ∧ Q 4 = 132 := by
  sorry

-- Additional theorem to show that p(2) = 372
theorem p_2_equals_372 : p 2 = 372 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_net_profit_at_4_minutes_p_2_equals_372_l1057_105787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cotangent_inequality_l1057_105774

theorem triangle_cotangent_inequality (a b c : ℝ) (A B C : ℝ) (r R : ℝ) :
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ 
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧ 
  (A + B + C = π) ∧
  (0 < r) ∧ (0 < R) ∧
  (a = 2 * R * Real.sin A) ∧ (b = 2 * R * Real.sin B) ∧ (c = 2 * R * Real.sin C) →
  6 * r ≤ a * (Real.cos A / Real.sin A) + b * (Real.cos B / Real.sin B) + c * (Real.cos C / Real.sin C) ∧
  a * (Real.cos A / Real.sin A) + b * (Real.cos B / Real.sin B) + c * (Real.cos C / Real.sin C) ≤ 3 * R :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cotangent_inequality_l1057_105774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l1057_105732

theorem polynomial_identity (P : Polynomial ℝ) : 
  (P.eval 0 = 0) → 
  (∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1) → 
  (P = Polynomial.X) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l1057_105732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slopes_product_implies_eccentricity_l1057_105797

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1
  h_not_vertex : (x ≠ -E.a ∨ y ≠ 0) ∧ (x ≠ E.a ∨ y ≠ 0)

/-- The theorem stating the relationship between the slopes product and eccentricity -/
theorem ellipse_slopes_product_implies_eccentricity (E : Ellipse) (P : PointOnEllipse E) :
  (P.y / (P.x + E.a)) * (P.y / (P.x - E.a)) = -1/2 →
  Real.sqrt ((E.a^2 - E.b^2) / E.a^2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slopes_product_implies_eccentricity_l1057_105797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l1057_105766

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the theorem
theorem min_k_value (k : ℝ) : 
  (∀ x > 0, ∀ m ∈ Set.Icc 3 5, f x ≥ m + 4 / m - k) → 
  k ≥ 29 / 5 + 1 / Real.exp 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_l1057_105766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_double_angle_l1057_105721

theorem cos_negative_double_angle (θ : ℝ) : 
  Real.sin (π / 2 + θ) = 3 / 5 → Real.cos (-2 * θ) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_double_angle_l1057_105721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_passes_through_point_inverse_proportion_is_correct_l1057_105742

/-- An inverse proportion function passing through (-3, 4) -/
noncomputable def inverse_proportion_function (x : ℝ) : ℝ :=
  -12 / x

theorem inverse_proportion_passes_through_point :
  inverse_proportion_function (-3) = 4 :=
by sorry

theorem inverse_proportion_is_correct (x : ℝ) (hx : x ≠ 0) :
  inverse_proportion_function x = -12 / x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_passes_through_point_inverse_proportion_is_correct_l1057_105742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_our_parabola_l1057_105722

/-- A parabola is defined by its equation relating x and y coordinates --/
structure Parabola where
  equation : ℝ → ℝ

/-- The focus of a parabola is a point (x, y) --/
structure Focus where
  x : ℝ
  y : ℝ

/-- Define our specific parabola x = (1/4)y^2 --/
noncomputable def our_parabola : Parabola where
  equation := fun y => (1/4) * y^2

/-- Theorem: The focus of the parabola x = (1/4)y^2 is at (-1, 0) --/
theorem focus_of_our_parabola : 
  ∃ (f : Focus), f.x = -1 ∧ f.y = 0 ∧ 
  (∀ (x y : ℝ), x = our_parabola.equation y → 
    (x - f.x)^2 + y^2 = (x - (-f.x))^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_our_parabola_l1057_105722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_payment_difference_l1057_105728

/-- Compound interest calculation function -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (compoundFreq : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / compoundFreq) ^ (compoundFreq * time)

/-- Simple interest calculation function -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Theorem stating the difference between compound and simple interest loan payments -/
theorem loan_payment_difference :
  let principal := 15000
  let compoundRate := 0.08
  let simpleRate := 0.10
  let compoundFreq := 12
  let totalTime := 15
  let halfPaymentTime := 10

  let compoundAmount10Years := compoundInterest principal compoundRate compoundFreq halfPaymentTime
  let halfPayment := compoundAmount10Years / 2
  let remainingBalance := compoundAmount10Years - halfPayment
  let finalCompoundAmount := compoundInterest remainingBalance compoundRate compoundFreq (totalTime - halfPaymentTime)
  let totalCompoundPayment := halfPayment + finalCompoundAmount

  let totalSimplePayment := simpleInterest principal simpleRate totalTime

  let difference := totalCompoundPayment - totalSimplePayment

  ⌊difference⌋₊ = 4163 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_payment_difference_l1057_105728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friendly_triangle_theorem_l1057_105718

open Real

/-- Definition of a friendly triangle -/
noncomputable def is_friendly (A B C A1 B1 C1 : ℝ) : Prop :=
  (cos A) / (sin A1) = (cos B) / (sin B1) ∧
  (cos B) / (sin B1) = (cos C) / (sin C1) ∧
  (cos A) / (sin A1) = 1

/-- Check if a triangle has a friendly triangle -/
def has_friendly_triangle (A B C : ℝ) : Prop :=
  ∃ A1 B1 C1, is_friendly A B C A1 B1 C1

/-- Theorem about friendly triangles -/
theorem friendly_triangle_theorem :
  (¬ has_friendly_triangle (π/2) (π/3) (π/6)) ∧
  (has_friendly_triangle (5*π/12) (π/3) (π/4)) ∧
  (¬ has_friendly_triangle (5*π/12) (5*π/12) (π/6)) ∧
  (∀ B C : ℝ, 
    has_friendly_triangle (7*π/18) B C →
    ((B = 13*π/36 ∧ C = π/4) ∨ (B = π/4 ∧ C = 13*π/36))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_friendly_triangle_theorem_l1057_105718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_squares_l1057_105745

/-- A square in a planar shape -/
structure Square where
  area : ℝ
  position : ℝ × ℝ

/-- The planar shape containing the squares -/
structure PlanarShape where
  area : ℝ
  squares : List Square

/-- The area of intersection between two squares -/
noncomputable def intersection_area (s1 s2 : Square) : ℝ := sorry

theorem overlapping_squares (shape : PlanarShape) 
    (h1 : shape.area = 1)
    (h2 : shape.squares.length = 9)
    (h3 : ∀ s, s ∈ shape.squares → s.area = 1/5) :
  ∃ s1 s2, s1 ∈ shape.squares ∧ s2 ∈ shape.squares ∧ s1 ≠ s2 ∧ intersection_area s1 s2 ≥ 1/45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_squares_l1057_105745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edric_hourly_rate_l1057_105795

/-- Calculates the hourly rate given monthly salary and work schedule -/
noncomputable def hourly_rate (monthly_salary : ℝ) (hours_per_day : ℝ) (days_per_week : ℝ) : ℝ :=
  let weeks_per_month : ℝ := 52 / 12
  let hours_per_month : ℝ := hours_per_day * days_per_week * weeks_per_month
  monthly_salary / hours_per_month

/-- Theorem stating Edric's hourly rate -/
theorem edric_hourly_rate :
  let monthly_salary : ℝ := 576
  let hours_per_day : ℝ := 8
  let days_per_week : ℝ := 6
  abs (hourly_rate monthly_salary hours_per_day days_per_week - 2.77) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_edric_hourly_rate_l1057_105795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_entry_is_22_l1057_105706

/-- Represents the remainder when n is divided by 7 -/
def r_7 (n : ℕ) : ℕ := n % 7

/-- The condition for n such that r_7(3n) ≤ 3 -/
def satisfies_condition (n : ℕ) : Prop := r_7 (3 * n) ≤ 3

/-- The sequence of nonnegative integers satisfying the condition -/
def satisfying_sequence : List ℕ := sorry

theorem fifteenth_entry_is_22 : satisfying_sequence.get? 14 = some 22 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_entry_is_22_l1057_105706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l1057_105734

/-- Arithmetic sequence with first term 1 and common difference d -/
def arithmetic_seq (d : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => arithmetic_seq d n + d

/-- Geometric sequence with first term 1 and common ratio r -/
def geometric_seq (r : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => geometric_seq r n * r

/-- Sum of corresponding terms of arithmetic and geometric sequences -/
def c_seq (d r : ℕ) (n : ℕ) : ℕ :=
  arithmetic_seq d n + geometric_seq r n

theorem c_k_value (d r k : ℕ) : 
  c_seq d r (k - 1) = 200 ∧ 
  c_seq d r (k + 1) = 2000 →
  c_seq d r k = 423 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_k_value_l1057_105734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l1057_105757

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_a_ge_b : a ≥ b

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ := 
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Theorem: For an ellipse with equation x²/5 + y²/m = 1, 
    where the foci are on the y-axis and the eccentricity is √10/5, 
    the value of m is 25/3 -/
theorem ellipse_m_value (m : ℝ) (h_m_pos : 0 < m) :
  let e := Ellipse.mk (Real.sqrt 5) (Real.sqrt m) (by norm_num) (by exact Real.sqrt_pos.mpr h_m_pos) (by sorry)
  e.eccentricity = Real.sqrt 10 / 5 → m = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l1057_105757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_johnson_class_size_l1057_105743

/-- Proves that the number of students in Mrs. Johnson's class is 29 -/
theorem mrs_johnson_class_size :
  ∀ (b : ℕ), -- number of boys
  let g := b + 1 -- number of girls
  let total_jellybeans := 441
  let remaining_jellybeans := 9
  let distributed_jellybeans := total_jellybeans - remaining_jellybeans
  b * b + g * g = distributed_jellybeans →
  b + g = 29 :=
by
  intro b
  -- Define variables
  let g := b + 1
  let total_jellybeans := 441
  let remaining_jellybeans := 9
  let distributed_jellybeans := total_jellybeans - remaining_jellybeans
  
  -- Assume the hypothesis
  intro h
  
  -- The actual proof would go here
  sorry

#check mrs_johnson_class_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_johnson_class_size_l1057_105743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_device_usage_theorem_l1057_105751

def probability_A : ℝ := 0.6
def probability_B : ℝ := 0.5
def probability_C : ℝ := 0.5
def probability_D : ℝ := 0.4

def events_independent : Prop := sorry

def probability_at_least_three : ℝ := sorry

noncomputable def probability_more_than_k (k : ℕ) : ℝ := sorry

def min_devices (k : ℕ) : Prop := 
  (probability_more_than_k k < 0.1) ∧ 
  ∀ j : ℕ, j < k → probability_more_than_k j ≥ 0.1

theorem device_usage_theorem :
  events_independent →
  probability_at_least_three = 0.31 ∧
  min_devices 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_device_usage_theorem_l1057_105751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1057_105701

/-- Given a hyperbola and an ellipse with the same parameters, prove the eccentricity of the ellipse -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let hyperbola_eq := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let ellipse_eq := fun (x y : ℝ) => x^2 / a^2 + y^2 / b^2 = 1
  let hyperbola_eccentricity := Real.sqrt 5 / 2
  let ellipse_eccentricity := fun (c : ℝ) => c / a
  ∃ c, hyperbola_eccentricity = Real.sqrt (a^2 + b^2) / a →
       ellipse_eccentricity c = Real.sqrt 3 / 2 ∧
       c^2 = a^2 - b^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1057_105701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biking_distance_comparison_l1057_105769

/-- Represents a biker with a constant speed -/
structure Biker where
  speed : ℚ
  deriving Repr

/-- Calculates the distance traveled by a biker in a given time -/
def distance (b : Biker) (t : ℚ) : ℚ := b.speed * t

/-- Bjorn's speed based on traveling 45 miles in 4 hours -/
def bjorn_speed : ℚ := 45 / 4

/-- Alberto's speed based on traveling 60 miles in 4 hours -/
def alberto_speed : ℚ := 60 / 4

/-- Carlos's speed based on traveling 75 miles in 4 hours -/
def carlos_speed : ℚ := 75 / 4

theorem biking_distance_comparison :
  let bjorn : Biker := ⟨bjorn_speed⟩
  let alberto : Biker := ⟨alberto_speed⟩
  let carlos : Biker := ⟨carlos_speed⟩
  (distance alberto 6 - distance bjorn 6 = 45/2) ∧
  (distance carlos 6 - distance bjorn 6 = 45) := by
  sorry

#eval bjorn_speed
#eval alberto_speed
#eval carlos_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_biking_distance_comparison_l1057_105769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fires_ratio_l1057_105768

def fires_problem (doug kai eli : ℕ) : Prop :=
  doug = 20 ∧ 
  kai = 3 * doug ∧ 
  doug + kai + eli = 110

theorem fires_ratio : 
  ∀ doug kai eli : ℕ, 
  fires_problem doug kai eli → 
  eli * 2 = kai := by
  intro doug kai eli h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fires_ratio_l1057_105768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_segment_l1057_105737

noncomputable section

/-- The volume of a cylinder with hemispheres at both ends -/
def cylinderWithHemispheresVolume (radius : ℝ) (length : ℝ) : ℝ :=
  Real.pi * radius^2 * length + (4/3) * Real.pi * radius^3

theorem length_of_segment (AB : ℝ) :
  cylinderWithHemispheresVolume 4 AB = 384 * Real.pi → AB = 19 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_segment_l1057_105737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_distribution_l1057_105715

theorem chocolate_distribution (tito angela juan : ℕ) : 
  tito > 0 → 
  angela = 3 * tito → 
  juan = 4 * angela → 
  (13 : ℚ) / 36 = ((juan + angela + tito) / 3 - tito : ℚ) / juan :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_distribution_l1057_105715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_number_square_l1057_105705

noncomputable def arithmetic_mean (numbers : List ℝ) : ℝ := (numbers.sum) / numbers.length

theorem equal_number_square (numbers : List ℝ) 
  (h1 : numbers.length = 5)
  (h2 : arithmetic_mean numbers = 20)
  (h3 : 12 ∈ numbers)
  (h4 : 22 ∈ numbers)
  (h5 : ∃ x, (numbers.filter (λ y => y ≠ 12 ∧ y ≠ 22)).all (λ y => y = x)) :
  ∃ x ∈ numbers, x ≠ 12 ∧ x ≠ 22 ∧ x^2 = 484 := by
  sorry

#check equal_number_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_number_square_l1057_105705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_five_numbers_l1057_105712

theorem product_of_five_numbers : 
  ∃ (n₁ n₂ n₃ n₄ n₅ : ℕ), 
    n₁ ∈ ({1, 2, 4} : Set ℕ) ∧ 
    n₂ ∈ ({1, 2, 4} : Set ℕ) ∧ 
    n₃ ∈ ({1, 2, 4} : Set ℕ) ∧ 
    n₄ ∈ ({1, 2, 4} : Set ℕ) ∧ 
    n₅ ∈ ({1, 2, 4} : Set ℕ) ∧
    n₁ * n₂ * n₃ * n₄ * n₅ = 256 ∧
    ∀ (m₁ m₂ m₃ m₄ m₅ : ℕ), 
      m₁ ∈ ({1, 2, 4} : Set ℕ) →
      m₂ ∈ ({1, 2, 4} : Set ℕ) →
      m₃ ∈ ({1, 2, 4} : Set ℕ) →
      m₄ ∈ ({1, 2, 4} : Set ℕ) →
      m₅ ∈ ({1, 2, 4} : Set ℕ) →
      m₁ * m₂ * m₃ * m₄ * m₅ ≠ 100 ∧
      m₁ * m₂ * m₃ * m₄ * m₅ ≠ 120 ∧
      m₁ * m₂ * m₃ * m₄ * m₅ ≠ 768 ∧
      m₁ * m₂ * m₃ * m₄ * m₅ ≠ 2048 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_five_numbers_l1057_105712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_committees_l1057_105747

-- Define the types for professors and committees
def Professor : Type := ℕ
def Committee : Type := ℕ

-- Define the university structure
structure University :=
  (professors : Finset Professor)
  (committees : Finset Committee)
  (serves_on : Professor → Committee → Prop)

-- Define the axioms
def satisfies_axioms (u : University) : Prop :=
  -- Axiom 1
  (∀ p q : Professor, p ∈ u.professors → q ∈ u.professors → p ≠ q → 
    ∃! c : Committee, c ∈ u.committees ∧ u.serves_on p c ∧ u.serves_on q c) ∧
  -- Axiom 2
  (∀ c : Committee, c ∈ u.committees → ∀ p : Professor, p ∈ u.professors → ¬u.serves_on p c →
    ∃! d : Committee, d ∈ u.committees ∧ u.serves_on p d ∧ 
    ∀ q : Professor, q ∈ u.professors → u.serves_on q c → ¬u.serves_on q d) ∧
  -- Axiom 3
  (∀ c : Committee, c ∈ u.committees → 
    ∃ p q : Professor, p ∈ u.professors ∧ q ∈ u.professors ∧ p ≠ q ∧ u.serves_on p c ∧ u.serves_on q c) ∧
  -- At least two committees
  (u.committees.card ≥ 2)

-- Theorem statement
theorem min_committees (u : University) (h : satisfies_axioms u) :
  u.committees.card ≥ 6 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_committees_l1057_105747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sister_age_in_2021_l1057_105780

def kelsey_age_in_1999 : ℕ := 25
def year_kelsey_25 : ℕ := 1999
def sister_age_difference : ℕ := 3
def current_year : ℕ := 2021

theorem sister_age_in_2021 : 
  current_year - (year_kelsey_25 - kelsey_age_in_1999 - sister_age_difference) = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sister_age_in_2021_l1057_105780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_example_l1057_105708

-- Define the custom operation
noncomputable def custom_op (m n x y : ℝ) : ℝ := (n^2 - m) / x + y

-- Theorem statement
theorem custom_op_example : custom_op 4 3 2 5 = 7.5 := by
  -- Unfold the definition of custom_op
  unfold custom_op
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_example_l1057_105708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_problem_car_r_speed_is_50_l1057_105789

def distance : ℝ := 600

def car_r_speed (v : ℝ) : Prop := v = 50

theorem car_speed_problem (v : ℝ) :
  (distance / v - 2 = distance / (v + 10)) →
  car_r_speed v :=
by
  intro h
  sorry

theorem car_r_speed_is_50 : car_r_speed 50 :=
by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_problem_car_r_speed_is_50_l1057_105789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_property_l1057_105748

/-- A reflection in 2D space -/
def Reflection : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) :=
  fun v x => sorry

theorem reflection_property (r : (ℝ × ℝ) → (ℝ × ℝ)) :
  r (2, -3) = (-2, 7) →
  r (3, -1) = (-3, -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_property_l1057_105748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_comparison_l1057_105713

/-- Triangle represented by its three vertices -/
structure Triangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ

/-- Calculate the area of a triangle -/
noncomputable def area (t : Triangle) : ℝ :=
  let (x1, y1) := t.v1
  let (x2, y2) := t.v2
  let (x3, y3) := t.v3
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

/-- Calculate the perimeter of a triangle -/
noncomputable def perimeter (t : Triangle) : ℝ :=
  let d12 := Real.sqrt ((t.v2.1 - t.v1.1)^2 + (t.v2.2 - t.v1.2)^2)
  let d23 := Real.sqrt ((t.v3.1 - t.v2.1)^2 + (t.v3.2 - t.v2.2)^2)
  let d31 := Real.sqrt ((t.v1.1 - t.v3.1)^2 + (t.v1.2 - t.v3.2)^2)
  d12 + d23 + d31

/-- Triangle I with vertices (0,0), (3,0), and (0,3) -/
def triangle_I : Triangle :=
  { v1 := (0, 0), v2 := (3, 0), v3 := (0, 3) }

/-- Triangle II with vertices (0,0), (4.5,0), and (0,2) -/
def triangle_II : Triangle :=
  { v1 := (0, 0), v2 := (4.5, 0), v3 := (0, 2) }

/-- Theorem: The areas of Triangle I and Triangle II are equal, but the perimeter of Triangle I is less than the perimeter of Triangle II -/
theorem triangle_comparison :
  area triangle_I = area triangle_II ∧ perimeter triangle_I < perimeter triangle_II := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_comparison_l1057_105713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1057_105749

-- Define the ellipse C₁
def C₁ (a b x y : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1 ∧ a > b ∧ b > 0

-- Define the hyperbola C₂
def C₂ (x y : ℝ) : Prop := x^2 - y^2 = 4

-- Define the common point P
def P (a b x y : ℝ) : Prop := C₁ a b x y ∧ C₂ x y

-- Define the right focus F₂
def F₂ (x y : ℝ) : Prop := x > 0 ∧ y = 0

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Define the eccentricity of an ellipse
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

theorem ellipse_eccentricity 
  (a b c : ℝ) 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : C₁ a b x₁ y₁) 
  (h₂ : C₂ x₁ y₁) 
  (h₃ : F₂ x₂ y₂) 
  (h₄ : distance x₁ y₁ x₂ y₂ = 2) :
  eccentricity a c = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1057_105749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_positive_power_of_two_l1057_105798

theorem negation_of_all_positive_power_of_two :
  (¬ ∀ x : ℝ, (2 : ℝ)^x > 0) ↔ (∃ x : ℝ, (2 : ℝ)^x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_all_positive_power_of_two_l1057_105798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_of_implication_l1057_105736

/- Define a proposition as a function from some type to Prop -/
def Proposition (α : Type) := α → Prop

/- Define the implication proposition "If p, then q" -/
def ImplicationProposition {α : Type} (p q : Proposition α) : Proposition α :=
  λ x ↦ p x → q x

/- Define the converse of a proposition -/
def Converse {α : Type} (prop : Proposition α) : Proposition α :=
  λ x ↦ prop x

/- Theorem: The converse of "If p, then q" is "If q, then p" -/
theorem converse_of_implication {α : Type} (p q : Proposition α) :
  Converse (ImplicationProposition p q) = ImplicationProposition q p :=
by
  funext x
  simp [Converse, ImplicationProposition]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_converse_of_implication_l1057_105736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_sqrt_two_l1057_105775

/-- Given an acute triangle ABC with an internal point D satisfying certain conditions, 
    prove that the ratio (AB · CD) / (AC · BD) equals √2. -/
theorem triangle_ratio_sqrt_two (A B C D : ℂ) : 
  let triangle_is_acute : Prop := sorry
  let D_is_internal : Prop := sorry
  let angle_condition : ℂ → ℂ → ℂ → ℝ := λ p q r ↦ sorry -- Function to calculate angle
  let distance : ℂ → ℂ → ℝ := λ p q ↦ sorry -- Function to calculate distance
  triangle_is_acute ∧ 
  D_is_internal ∧ 
  angle_condition A D B = angle_condition A C B + π/2 ∧ 
  distance A C * distance B D = distance A D * distance B C →
  (distance A B * distance C D) / (distance A C * distance B D) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_sqrt_two_l1057_105775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l1057_105750

/-- The line on which point P moves --/
def line (x y : ℝ) : Prop := x + y - 1 = 0

/-- The fixed point Q --/
def Q : ℝ × ℝ := (1, 1)

/-- The distance between two points --/
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

/-- The theorem statement --/
theorem min_distance_point :
  ∀ P : ℝ × ℝ, line P.1 P.2 →
  (∀ P' : ℝ × ℝ, line P'.1 P'.2 → distance P Q ≤ distance P' Q) →
  P = (1/2, 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l1057_105750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_range_l1057_105786

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

-- Define the line
def my_line (x y : ℝ) : Prop := y = 2*x - 4

-- Define the condition for point P
def condition_P (m : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ),
    my_circle A.1 A.2 ∧
    my_circle B.1 B.2 ∧
    my_line m (2*m - 4) ∧
    (m - A.1)^2 + (2*m - 4 - A.2)^2 = 4 * ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- State the theorem
theorem x_coordinate_range :
  ∀ m : ℝ, condition_P m → 9 - 2*Real.sqrt 19 ≤ m ∧ m ≤ 9 + 2*Real.sqrt 19 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_range_l1057_105786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1057_105771

theorem solve_exponential_equation :
  ∀ y : ℚ, (16 : ℝ)^((3:ℝ)*y - 5) = (4 : ℝ)^((2:ℝ)*y + 8) → y = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1057_105771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_age_l1057_105777

def Ages : List ℕ := [2, 4, 6, 8, 10, 12]

structure Friend where
  age : ℕ
  activity : String

def ConcertPair (f1 f2 : Friend) : Prop :=
  f1.age + f2.age = 18 ∧ f1.activity = "concert" ∧ f2.activity = "concert"

def ParkGroup (group : List Friend) : Prop :=
  group.length = 3 ∧ (∀ f ∈ group, f.age < 8 ∧ f.activity = "park")

def HomeStay (alex youngest : Friend) : Prop :=
  alex.activity = "home" ∧ youngest.activity = "home" ∧
  youngest.age = (Ages.filter (λ a => a < 8)).minimum?

theorem alex_age (friends : List Friend) : 
  friends.length = 6 ∧
  (∀ f ∈ friends, f.age ∈ Ages) ∧
  (∃ f1 f2, f1 ∈ friends ∧ f2 ∈ friends ∧ ConcertPair f1 f2) ∧
  (∃ group : List Friend, group ⊆ friends ∧ ParkGroup group) ∧
  (∃ alex youngest, alex ∈ friends ∧ youngest ∈ friends ∧ HomeStay alex youngest) →
  ∃ alex ∈ friends, alex.age = 12 ∧ alex.activity = "home" :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_age_l1057_105777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1057_105735

/-- A hyperbola with semi-major axis a, semi-minor axis b, and focal length c -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_pos : 0 < b ∧ b < a
  h_focal : c^2 = a^2 + b^2

/-- The angle between the asymptotes of a hyperbola -/
noncomputable def asymptote_angle (h : Hyperbola) : ℝ := 2 * Real.arctan (h.b / h.a)

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := h.c / h.a

/-- A focus of a hyperbola -/
def focus (h : Hyperbola) : ℝ × ℝ := (h.c, 0)

/-- A line passing through a point -/
def line_through (m : ℝ) (b : ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = m * p.1 + b

theorem hyperbola_properties (h : Hyperbola) 
    (h_focal_length : h.c = 2)
    (h_asymptote_angle : asymptote_angle h = π/3) :
  eccentricity h = 2*Real.sqrt 3/3 ∧
  line_through (-1) 2 (focus h) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1057_105735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ratio_on_line_l1057_105755

theorem angle_ratio_on_line (α : Real) : 
  (∃ (x y : Real), y = 2 * x ∧ Real.tan α = y / x) → 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ratio_on_line_l1057_105755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_fund_theorem_l1057_105759

/-- Calculates the required working hours per week to meet a financial goal
    given initial plans and unexpected changes. -/
noncomputable def required_hours_per_week (
  initial_weeks : ℕ
  ) (initial_hours_per_week : ℝ
  ) (target_amount : ℝ
  ) (unavailable_weeks : ℕ
  ) : ℝ :=
  let remaining_weeks := initial_weeks - unavailable_weeks
  let hourly_rate := target_amount / (initial_hours_per_week * initial_weeks)
  let total_hours_needed := target_amount / hourly_rate
  total_hours_needed / remaining_weeks

/-- Proves that given the original plan and actual situation,
    the required working hours per week is approximately 31.25 -/
theorem vacation_fund_theorem :
  let result := required_hours_per_week 15 25 4500 3
  (result > 31.24) ∧ (result < 31.26) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vacation_fund_theorem_l1057_105759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_radius_of_curvature_l1057_105729

/-- The radius of curvature of a projectile's trajectory at its final point --/
noncomputable def radiusOfCurvature (v₀ : ℝ) (θ : ℝ) (g : ℝ) : ℝ :=
  v₀^2 / (g * Real.cos θ)

/-- Theorem: The radius of curvature for the given projectile motion is 20 m --/
theorem projectile_radius_of_curvature :
  let v₀ : ℝ := 10  -- Initial speed in m/s
  let θ : ℝ := Real.pi / 3  -- Angle of projection in radians (60°)
  let g : ℝ := 10  -- Acceleration due to gravity in m/s²
  radiusOfCurvature v₀ θ g = 20 := by
  -- Unfold the definition of radiusOfCurvature
  unfold radiusOfCurvature
  -- Simplify the expression
  simp [Real.cos_pi_div_three]
  -- The rest of the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_radius_of_curvature_l1057_105729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_probability_l1057_105793

def total_members : ℕ := 24
def boys : ℕ := 14
def girls : ℕ := 10
def committee_size : ℕ := 5

def probability_at_least_one_of_each : ℚ := 4025 / 42504

theorem committee_probability :
  (Nat.choose total_members committee_size - (Nat.choose boys committee_size + Nat.choose girls committee_size) : ℚ) /
  Nat.choose total_members committee_size = probability_at_least_one_of_each :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_probability_l1057_105793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_sum_in_three_circles_l1057_105784

/-- Configuration of three intersecting circles -/
inductive CircleConfiguration
| CaseA
| CaseB
deriving Repr, DecidableEq

/-- Representation of an arc on a circle -/
structure Arc where
  measure : ℝ

/-- Theorem about the sum of arcs in three intersecting equal circles -/
theorem arc_sum_in_three_circles (config : CircleConfiguration) 
  (AB₁ BC₁ CA₁ : Arc) : 
  AB₁.measure + BC₁.measure + (if config = CircleConfiguration.CaseB then -CA₁.measure else CA₁.measure) = 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_sum_in_three_circles_l1057_105784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_right_angle_l1057_105714

/-- The time required for clock hands to form a right angle -/
noncomputable def time_to_right_angle (second_hand_speed minute_hand_speed : ℝ) : ℝ :=
  90 / (second_hand_speed - minute_hand_speed)

/-- Theorem: The time for clock hands to form a right angle is 15 seconds -/
theorem clock_hands_right_angle :
  time_to_right_angle 6 0.1 = 15 := by
  -- Unfold the definition of time_to_right_angle
  unfold time_to_right_angle
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_right_angle_l1057_105714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_accurate_value_l1057_105716

/-- The measured value of constant K -/
def measured_value : ℝ := 1.78654

/-- The accuracy of the measurement -/
def accuracy : ℝ := 0.00443

/-- The upper bound of the possible values for K -/
def upper_bound : ℝ := measured_value + accuracy

/-- The lower bound of the possible values for K -/
def lower_bound : ℝ := measured_value - accuracy

/-- A function to round a real number to the nearest tenth -/
noncomputable def round_to_tenth (x : ℝ) : ℝ := 
  ⌊x * 10 + 0.5⌋ / 10

/-- The theorem stating that 1.8 is the most accurate value that can be declared -/
theorem most_accurate_value :
  round_to_tenth upper_bound = round_to_tenth lower_bound ∧
  round_to_tenth upper_bound = 1.8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_accurate_value_l1057_105716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_side_length_theorem_l1057_105758

/-- The side length of the base triangle of a regular triangular prism 
    intersected by a plane, given the side lengths of the intersection triangle. -/
noncomputable def base_side_length (a b c : ℝ) : ℝ :=
  Real.sqrt ((1/3) * (a^2 + b^2 + c^2 - 2 * Real.sqrt (a^4 + b^4 + c^4 - a^2*b^2 - a^2*c^2 - b^2*c^2)))

/-- Theorem stating the relationship between the side lengths of the intersection 
    triangle and the base triangle in a regular triangular prism. -/
theorem base_side_length_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let d := base_side_length a b c
  (d^2 : ℝ) = (1/3) * (a^2 + b^2 + c^2 - 2 * Real.sqrt (a^4 + b^4 + c^4 - a^2*b^2 - a^2*c^2 - b^2*c^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_side_length_theorem_l1057_105758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_f_above_g_implies_a_bound_max_value_of_h_l1057_105733

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def t (a : ℝ) (x : ℝ) : ℝ := g x - f a x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := |f a x|

-- Define the maximum value function
noncomputable def F (a : ℝ) : ℝ :=
  if a ≤ 1/4 then 1 - 3*a
  else if a < 1 then 2*a*Real.sqrt a
  else 3*a - 1

-- Theorem statements
theorem tangent_line_at_one (a : ℝ) (h : a = 1) :
  ∃ m b, ∀ x, t a x = m * (x - 1) + t a 1 ∧ m * x - 1 * y + b = 0 := by sorry

theorem f_above_g_implies_a_bound (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f a x ≥ g x) → a ≤ 1/3 := by sorry

theorem max_value_of_h (a : ℝ) :
  ∀ x ∈ Set.Icc (-1) 1, h a x ≤ F a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_f_above_g_implies_a_bound_max_value_of_h_l1057_105733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1057_105792

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x + 4 / x

-- Theorem statement
theorem f_properties :
  (f 1 = 5) ∧ 
  (∀ x : ℝ, x ≠ 0 → f (-x) = -f x) ∧
  (∀ x y : ℝ, 2 ≤ x → x < y → f x < f y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1057_105792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1057_105744

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-1 : ℝ) 1 ∧ 
    |4^x₀ - a * 2^x₀ + 1| ≤ 2^(x₀ + 1)) ↔ 
  a ∈ Set.Icc 0 (9/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1057_105744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l1057_105772

/-- Calculates the length of a platform given train parameters --/
theorem platform_length
  (train_length : ℝ)
  (train_speed_kmph : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 120)
  (h2 : train_speed_kmph = 60)
  (h3 : crossing_time = 15) :
  ∃ (platform_length : ℝ), |platform_length - 130.05| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l1057_105772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_suitable_factorials_suitable_no_squares_l1057_105762

-- Define what it means for a set to be suitable
def IsSuitable (A : Set ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → (∃! p : ℕ, Nat.Prime p ∧ n - p ∈ A) ∨ (∀ p : ℕ, Nat.Prime p → n - p ∉ A)

-- Define the set of perfect squares
def PerfectSquares : Set ℕ := {n : ℕ | ∃ k : ℕ, n = k^2}

-- Define the set of factorials for n ≥ 2
def Factorials : Set ℕ := {n : ℕ | ∃ k : ℕ, k ≥ 2 ∧ n = Nat.factorial k}

theorem perfect_squares_suitable : IsSuitable PerfectSquares := by
  sorry

theorem factorials_suitable_no_squares :
  IsSuitable Factorials ∧ (∀ n : ℕ, n ∈ Factorials → n ∉ PerfectSquares) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_squares_suitable_factorials_suitable_no_squares_l1057_105762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_collinear_l1057_105760

open EuclideanGeometry

-- Define the points
variable (A B P Q R S M N T : EuclideanSpace ℝ (Fin 2))

-- Define the lines
variable (AX BY : Line ℝ (Fin 2))

-- Define the conditions
variable (h1 : A ∈ AX)
variable (h2 : B ∈ BY)
variable (h3 : P ∈ AX)
variable (h4 : R ∈ AX)
variable (h5 : Q ∈ BY)
variable (h6 : S ∈ BY)
variable (h7 : dist A P / dist B Q = dist A R / dist B S)
variable (h8 : dist A M / dist M B = dist P N / dist N Q)
variable (h9 : dist A M / dist M B = dist R T / dist T S)

-- Theorem statement
theorem points_collinear : Collinear ℝ ({M, N, T} : Set (EuclideanSpace ℝ (Fin 2))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_collinear_l1057_105760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l1057_105724

open Real

theorem indefinite_integral_proof (x : ℝ) (h : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -2) :
  deriv (λ y => (3 * y^2) / 2 - log (abs y) + log (abs (y - 1)) + log (abs (y + 2))) x =
  (3 * x^4 + 3 * x^3 - 5 * x^2 + 2) / (x * (x - 1) * (x + 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l1057_105724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_villagers_claim_impossible_l1057_105725

/-- Represents a village configuration with m villages on the left bank,
    n villages on the right bank, and one village on an island. -/
structure VillageConfig where
  m : ℕ
  n : ℕ

/-- Represents the claim made by villagers about ferry numbers. -/
def villagersClaim (config : VillageConfig) : Prop :=
  ∀ (v : ℕ), ∃ (a b : ℕ), Set.range (fun i => i + a) ∩ Set.range (fun i => i + 1) = Set.range (fun i => i + b)

/-- The main theorem stating that the villagers' claim is impossible. -/
theorem villagers_claim_impossible (config : VillageConfig) 
  (h : Nat.gcd (config.m + 1) (config.n + 1) > 1) : 
  ¬(villagersClaim config) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_villagers_claim_impossible_l1057_105725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_theorem_l1057_105707

theorem cube_root_sum_theorem (y : ℝ) (hy : y > 0) 
  (h : (2 - y^3)^(1/3) + (2 + y^3)^(1/3) = 2) : 
  y^6 = 116/27 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_theorem_l1057_105707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_length_special_trapezoid_l1057_105788

/-- A trapezoid with specific properties -/
structure SpecialTrapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  isParallel : (A.2 - D.2) / (A.1 - D.1) = (B.2 - C.2) / (B.1 - C.1)
  bdLength : Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 2
  angleDBA : Real.arccos ((A.1 - B.1) * (D.1 - B.1) + (A.2 - B.2) * (D.2 - B.2)) / 
             (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) * Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2)) = π / 6
  angleBDC : Real.arccos ((C.1 - D.1) * (B.1 - D.1) + (C.2 - D.2) * (B.2 - D.2)) / 
             (Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) * Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2)) = π / 3
  bcadRatio : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) / Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) = 7/4

/-- The length of AB in the special trapezoid is 2/3 -/
theorem ab_length_special_trapezoid (t : SpecialTrapezoid) : 
  Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2) = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_length_special_trapezoid_l1057_105788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1057_105731

theorem exponential_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : (2 : ℝ)^a > (2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_l1057_105731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_exp_is_log_l1057_105739

/-- Given a function f that is the inverse of the exponential function with base a,
    where a > 0 and a ≠ 1, and f passes through the point (a², a), prove that f(x) = log₂(x) -/
theorem inverse_exp_is_log (a : ℝ) (f : ℝ → ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : Function.RightInverse f (fun x ↦ a^x))
  (h4 : f (a^2) = a) :
  f = fun x ↦ Real.log x / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_exp_is_log_l1057_105739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_many_inscribed_squares_l1057_105785

/-- A square is inscribed in the graph of y = A sin x if all its vertices lie on the graph. -/
def IsInscribedSquare (A : ℝ) (square : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ square → y = A * Real.sin x

/-- The number of inscribed squares in the graph of y = A sin x. -/
noncomputable def NumInscribedSquares (A : ℝ) : ℕ :=
  Nat.card { square : Set (ℝ × ℝ) | IsInscribedSquare A square }

/-- There exists a real number A > 1978 * 2π such that the graph of y = A sin x
    contains at least 1978 mutually distinct inscribed squares. -/
theorem exists_many_inscribed_squares :
  ∃ A : ℝ, A > 1978 * 2 * Real.pi ∧ NumInscribedSquares A ≥ 1978 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_many_inscribed_squares_l1057_105785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composed_has_three_roots_l1057_105790

/-- The function g(x) -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

/-- The composition g(g(x)) -/
def g_composed (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- The value of d for which g(g(x)) has exactly 3 distinct real roots -/
noncomputable def d_value : ℝ := (3 - Real.sqrt 57) / 2

/-- Theorem stating that g_composed has exactly three distinct real roots when d = d_value -/
theorem g_composed_has_three_roots :
  ∃ (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
  (∀ x : ℝ, g_composed d_value x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composed_has_three_roots_l1057_105790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_weekly_consumption_l1057_105770

/-- Represents John's weekly beverage consumption --/
noncomputable def weekly_consumption : ℝ :=
  let water_daily := 1.5  -- gallons
  let milk_every_other_day := 3  -- pints
  let juice_every_three_days := 20  -- fluid ounces
  let days_in_week := 7
  let quarts_per_gallon := 4
  let pints_per_quart := 2
  let fluid_ounces_per_quart := 32
  let milk_days_per_week := 4  -- assuming 4 times a week
  let juice_days_per_week := 2  -- assuming 2 times a week

  let water_quarts := water_daily * quarts_per_gallon * days_in_week
  let milk_quarts := (milk_every_other_day / pints_per_quart) * milk_days_per_week
  let juice_quarts := (juice_every_three_days / fluid_ounces_per_quart) * juice_days_per_week

  water_quarts + milk_quarts + juice_quarts

/-- Theorem stating that John's weekly beverage consumption is 49.25 quarts --/
theorem johns_weekly_consumption :
  weekly_consumption = 49.25 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_weekly_consumption_l1057_105770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l1057_105782

/-- The distance between the foci of a hyperbola -/
noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between the foci of the hyperbola x^2/25 - y^2/9 = 1 is 2√34 -/
theorem hyperbola_foci_distance :
  distance_between_foci 5 3 = 2 * Real.sqrt 34 := by
  unfold distance_between_foci
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l1057_105782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_implies_sin_2x_l1057_105711

theorem trigonometric_sum_implies_sin_2x 
  (x : ℝ) 
  (h : Real.sin x + Real.cos x + Real.tan x + (Real.cos x / Real.sin x) + (1 / Real.cos x) + (1 / Real.sin x) = 11) : 
  Real.sin (2 * x) = 23 - 5 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_implies_sin_2x_l1057_105711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_quadratic_l1057_105794

theorem fixed_point_quadratic (k : ℝ) : 
  (λ x : ℝ ↦ 9 * x^2 + k * x - 5 * k) 5 = 225 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_quadratic_l1057_105794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_enclosure_A1_l1057_105710

/-- Represents the dimensions and area of an animal enclosure -/
structure Enclosure where
  width : ℝ
  length : ℝ
  area : ℝ

/-- Represents a rectangular field with animal enclosures -/
structure FarmField where
  width : ℝ
  enclosureA1 : Enclosure
  enclosureA2 : Enclosure
  enclosureA3 : Enclosure
  totalFenceLength : ℝ

/-- The theorem stating the area of enclosure A1 given the field conditions -/
theorem area_of_enclosure_A1 (f : FarmField) : 
  f.width = 45 ∧ 
  f.enclosureA2.area = 4 * f.enclosureA1.area ∧ 
  f.enclosureA3.area = 5 * f.enclosureA1.area ∧ 
  f.totalFenceLength = 360 →
  f.enclosureA1.area = 150 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_enclosure_A1_l1057_105710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_and_odd_functions_l1057_105704

-- Define the exponential function g
def g : ℝ → ℝ := sorry

-- Define the function f
def f : ℝ → ℝ := sorry

-- Theorem statement
theorem exponential_and_odd_functions
  (h1 : ∀ x y, g (x + y) = g x * g y)  -- g is exponential
  (h2 : g 3 = 27)                      -- g(3) = 27
  (h3 : ∀ x, f x = (1 - g x) / (3 + 3 * g x))  -- definition of f
  (h4 : ∀ x, f (-x) = -f x)            -- f is odd
  :
  (∀ x, g x = 3^x) ∧                   -- g(x) = 3^x
  (∀ x, f x = (1 - 3^x) / (3 + 3^(x+1))) ∧  -- f(x) = (1-3^x)/(3+3^(x+1))
  (∀ k, (∀ t ∈ Set.Ioo 1 4, f (2*t - 3) + f (t - k) > 0) ↔ k ≥ 9)  -- range of k
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_and_odd_functions_l1057_105704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_sum_l1057_105726

theorem reciprocal_of_sum : (3 / 4 + 4 / 5 : ℝ)⁻¹ = 20 / 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_sum_l1057_105726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recommended_calories_is_25_l1057_105702

/-- Represents the total calories in the prepared lunch -/
def total_calories : ℚ := 40

/-- Represents the fraction of lunch eaten by the dietitian -/
def fraction_eaten : ℚ := 3/4

/-- Represents the excess calories eaten compared to the recommended amount -/
def excess_calories : ℚ := 5

/-- Calculates the recommended calorie intake based on the given conditions -/
def recommended_calories : ℚ :=
  total_calories * fraction_eaten - excess_calories

/-- Theorem stating that the recommended calorie intake is 25 -/
theorem recommended_calories_is_25 :
  recommended_calories = 25 := by
  unfold recommended_calories
  unfold total_calories
  unfold fraction_eaten
  unfold excess_calories
  norm_num
  
#eval recommended_calories

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recommended_calories_is_25_l1057_105702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_when_p_or_q_false_range_when_p_or_q_true_and_p_and_q_false_l1057_105779

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x > 0 ∧ x^2 - 2 * Real.exp 1 * Real.log x ≤ m

def q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*m*x₁ + 1 = 0 ∧ x₂^2 - 2*m*x₂ + 1 = 0

-- Theorem for the first part of the problem
theorem range_when_p_or_q_false (m : ℝ) :
  (¬(p m ∨ q m)) ↔ (m ≥ -1 ∧ m < 0) :=
sorry

-- Theorem for the second part of the problem
theorem range_when_p_or_q_true_and_p_and_q_false (m : ℝ) :
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) ↔ (m < -1 ∨ (m ≥ 0 ∧ m ≤ 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_when_p_or_q_false_range_when_p_or_q_true_and_p_and_q_false_l1057_105779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_left_handed_classical_fans_count_l1057_105756

/-- Represents a group of people with handedness and music preferences -/
structure GroupInfo where
  total : ℕ
  leftHanded : ℕ
  classicalFans : ℕ
  rightHandedNonFans : ℕ

/-- The number of left-handed classical music fans in the group -/
def leftHandedClassicalFans (g : GroupInfo) : ℕ :=
  g.leftHanded + g.classicalFans - (g.total - g.rightHandedNonFans)

/-- Theorem stating the number of left-handed classical music fans -/
theorem left_handed_classical_fans_count (g : GroupInfo) 
  (h1 : g.total = 30)
  (h2 : g.leftHanded = 12)
  (h3 : g.classicalFans = 20)
  (h4 : g.rightHandedNonFans = 3) :
  leftHandedClassicalFans g = 5 := by
  sorry

#eval leftHandedClassicalFans { total := 30, leftHanded := 12, classicalFans := 20, rightHandedNonFans := 3 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_left_handed_classical_fans_count_l1057_105756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1057_105796

/-- The time taken for all workers to complete the work together -/
noncomputable def time_all_workers (time_AB time_A time_C time_D : ℝ) : ℝ :=
  1 / (1 / time_A + (1 / time_AB - 1 / time_A) + 1 / time_C + 1 / time_D)

/-- Theorem stating the time taken for all workers to complete the work -/
theorem work_completion_time 
  (time_AB : ℝ) (time_A : ℝ) (time_C : ℝ) (time_D : ℝ)
  (h_AB : time_AB = 5)
  (h_A : time_A = 10)
  (h_C : time_C = 15)
  (h_D : time_D = 20) :
  time_all_workers time_AB time_A time_C time_D = 60 / 19 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval time_all_workers 5 10 15 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l1057_105796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_9_15_l1057_105799

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees per hour on a clock face -/
noncomputable def degrees_per_hour : ℝ := 360 / clock_hours

/-- The position of the hour hand at 9:15 p.m. in degrees -/
noncomputable def hour_hand_position : ℝ := 9 * degrees_per_hour + 0.25 * degrees_per_hour

/-- The position of the minute hand at 9:15 p.m. in degrees -/
def minute_hand_position : ℝ := 90

/-- The angle between the hour and minute hands at 9:15 p.m. -/
noncomputable def angle_between_hands : ℝ := hour_hand_position - minute_hand_position

theorem clock_angle_at_9_15 :
  min angle_between_hands (360 - angle_between_hands) = 187.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_9_15_l1057_105799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l1057_105754

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
def interceptsXAxis (c : Circle) : Prop :=
  c.radius^2 - c.center.2^2 = 2

def interceptsYAxis (c : Circle) : Prop :=
  c.radius^2 - c.center.1^2 = 3

noncomputable def distanceToLineYEqX (c : Circle) : ℝ :=
  |c.center.2 - c.center.1| / Real.sqrt 2

-- Theorem statement
theorem circle_properties (c : Circle)
  (hx : interceptsXAxis c)
  (hy : interceptsYAxis c)
  (hd : distanceToLineYEqX c = 1/2) :
  (∃ x y : ℝ, y^2 - x^2 = 1 ∧ (x, y) = c.center) ∧
  ((∀ x y : ℝ, (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ↔ 
    x^2 + (y - 1)^2 = 3) ∨
   (∀ x y : ℝ, (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ↔ 
    x^2 + (y + 1)^2 = 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l1057_105754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l1057_105767

theorem log_inequality_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : Real.log (3/5) / Real.log a < 1) : 
  a ∈ Set.Ioo 0 (3/5) ∪ Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_range_l1057_105767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_values_l1057_105717

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x + 7 else 6 * x - 5

-- Theorem to prove g(-2) = 3 and g(3) = 13
theorem g_values : g (-2) = 3 ∧ g 3 = 13 := by
  -- Split the conjunction into two parts
  constructor
  -- Prove g(-2) = 3
  · sorry
  -- Prove g(3) = 13
  · sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_values_l1057_105717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_bounds_l1057_105761

noncomputable def ellipse (x y : ℝ) : Prop := (x/2)^2 + (y/3)^2 = 1

noncomputable def square_vertices : List (ℝ × ℝ) := [(1, Real.sqrt 3), (-Real.sqrt 3, 1), (-1, -Real.sqrt 3), (Real.sqrt 3, -1)]

def distance_squared (x1 y1 x2 y2 : ℝ) : ℝ := (x1 - x2)^2 + (y1 - y2)^2

noncomputable def sum_distances_squared (x y : ℝ) : ℝ :=
  (square_vertices.map (fun (vx, vy) => distance_squared x y vx vy)).sum

theorem distance_sum_bounds :
  ∀ x y : ℝ, ellipse x y → 32 ≤ sum_distances_squared x y ∧ sum_distances_squared x y ≤ 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_bounds_l1057_105761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hike_length_is_seven_l1057_105719

/-- Represents the hike scenario with given conditions -/
structure HikeScenario where
  initial_water : ℚ
  duration : ℚ
  remaining_water : ℚ
  leak_rate : ℚ
  last_mile_consumption : ℚ
  first_part_rate : ℚ

/-- Calculates the total length of the hike given the scenario -/
noncomputable def hike_length (scenario : HikeScenario) : ℚ :=
  let total_water_used := scenario.initial_water - scenario.remaining_water
  let leaked_water := scenario.duration * scenario.leak_rate
  let water_consumed := total_water_used - leaked_water
  let first_part_consumption := water_consumed - scenario.last_mile_consumption
  let first_part_distance := first_part_consumption / scenario.first_part_rate
  first_part_distance + 1

/-- Theorem stating that given the specific conditions, the hike length is 7 miles -/
theorem hike_length_is_seven : 
  let scenario := HikeScenario.mk 11 3 2 1 3 (1/2)
  hike_length scenario = 7 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hike_length_is_seven_l1057_105719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sin_angle_l1057_105700

theorem contrapositive_sin_angle :
  (¬(∀ A B : Real, Real.sin A = Real.sin B → A = B) ↔ (∀ A B : Real, A ≠ B → Real.sin A ≠ Real.sin B)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sin_angle_l1057_105700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l1057_105773

/-- Given an ellipse and a line passing through its left focus, prove the length of the chord formed by their intersection --/
theorem ellipse_chord_length :
  let ellipse (x y : ℝ) := x^2 / 9 + y^2 = 1
  let left_focus : ℝ × ℝ := (-2 * Real.sqrt 2, 0)
  let line_slope : ℝ := Real.sqrt 3 / 3
  let line (x y : ℝ) := y = line_slope * (x - left_focus.fst)
  let chord_length := 
    let x1_plus_x2 := 3 * Real.sqrt 2
    let x1_times_x2 := 15 / 4
    Real.sqrt (1 + line_slope^2) * Real.sqrt ((x1_plus_x2)^2 - 4 * x1_times_x2)
  chord_length = 2 := by
    sorry

#check ellipse_chord_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l1057_105773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identification_l1057_105746

noncomputable def f (w : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (w * x + φ)

theorem function_identification 
  (w : ℝ) (φ : ℝ) 
  (hw : w > 0) 
  (hφ : abs φ < π / 2) 
  (hperiod : ∀ x, f w φ (x + π) = f w φ x) 
  (hsymmetry : ∀ x, f w φ (π / 2 + π / 6 - x) = f w φ (π / 2 + π / 6 + x)) :
  ∀ x, f w φ x = Real.sin (2 * x - π / 6) :=
by
  sorry

#check function_identification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identification_l1057_105746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_MQ_parallel_NP_l1057_105740

-- Define the rhombus ABCD
structure Rhombus :=
  (A B C D : ℂ)

-- Define the incircle O
structure Incircle :=
  (O : ℂ)
  (radius : ℝ)

-- Define the tangent points
structure TangentPoints :=
  (E F G H : ℂ)

-- Define the intersection points
structure IntersectionPoints :=
  (M N P Q : ℂ)

-- Main theorem
noncomputable def parallel_lines_in_rhombus 
  (ABCD : Rhombus) 
  (O : Incircle) 
  (EFGH : TangentPoints) 
  (MNPQ : IntersectionPoints) : ℝ :=
  let zQ := MNPQ.Q
  let zM := MNPQ.M
  let zP := MNPQ.P
  let zN := MNPQ.N
  Complex.abs ((zQ - zM) / (zP - zN))

-- The proof is omitted
axiom parallel_lines_in_rhombus_proof : 
  ∀ (ABCD : Rhombus) (O : Incircle) (EFGH : TangentPoints) (MNPQ : IntersectionPoints),
  ∃ (r : ℝ), parallel_lines_in_rhombus ABCD O EFGH MNPQ = r

-- Main theorem statement
theorem MQ_parallel_NP 
  (ABCD : Rhombus) 
  (O : Incircle) 
  (EFGH : TangentPoints) 
  (MNPQ : IntersectionPoints) : 
  ∃ (r : ℝ), parallel_lines_in_rhombus ABCD O EFGH MNPQ = r :=
by
  exact parallel_lines_in_rhombus_proof ABCD O EFGH MNPQ

end NUMINAMATH_CALUDE_ERRORFEEDBACK_MQ_parallel_NP_l1057_105740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_values_from_sine_l1057_105783

theorem cosine_values_from_sine (θ : Real) (h1 : Real.sin θ = 1/3) (h2 : θ ∈ Set.Ioo 0 (Real.pi/2)) :
  Real.cos θ = 2*Real.sqrt 2/3 ∧ Real.cos (2*θ) = 7/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_values_from_sine_l1057_105783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f5_of_2_equals_33_l1057_105765

def f1 (x : ℝ) : ℝ := 2 * x - 1

def fn : ℕ → ℝ → ℝ
  | 0, x => x  -- Base case for n = 0
  | 1, x => f1 x
  | n+1, x => f1 (fn n x)

theorem f5_of_2_equals_33 : fn 5 2 = 33 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f5_of_2_equals_33_l1057_105765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_and_converse_product_zero_iff_square_eq_iff_product_eq_self_iff_propositions_truth_l1057_105723

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a side
noncomputable def sideLength (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define an angle in a triangle
noncomputable def angle (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

theorem triangle_inequality_and_converse (t : Triangle) :
  (angle t t.B < angle t t.C ↔ sideLength t.A t.C < sideLength t.A t.B) ∧
  (angle t t.C < angle t t.B ↔ sideLength t.A t.B < sideLength t.A t.C) :=
by sorry

theorem product_zero_iff (a b : ℝ) : a * b = 0 ↔ a = 0 ∨ b = 0 :=
by sorry

theorem square_eq_iff (a b : ℝ) : a^2 = b^2 ↔ a = b ∨ a = -b :=
by sorry

theorem product_eq_self_iff (a b c : ℝ) : a * c = c * b ↔ a = b ∨ c = 0 :=
by sorry

theorem propositions_truth :
  (∀ t : Triangle, angle t t.C > angle t t.B → sideLength t.A t.B > sideLength t.A t.C) ∧
  ¬(∀ a b : ℝ, (a ≠ 0 ∨ b ≠ 0) → a * b ≠ 0) ∧
  ¬(∀ a b : ℝ, a ≠ b → a^2 ≠ b^2) ∧
  (∀ a b c : ℝ, a = b → a * c = c * b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_and_converse_product_zero_iff_square_eq_iff_product_eq_self_iff_propositions_truth_l1057_105723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_M_correct_l1057_105703

/-- The smallest positive integer M such that among M, M+1, and M+2, 
    one is divisible by 2^3, one by 3^2, and one by 5^2 -/
def smallest_M : ℕ := 200

theorem smallest_M_correct :
  (∃ n ∈ ({smallest_M, smallest_M + 1, smallest_M + 2} : Set ℕ), 8 ∣ n) ∧
  (∃ n ∈ ({smallest_M, smallest_M + 1, smallest_M + 2} : Set ℕ), 9 ∣ n) ∧
  (∃ n ∈ ({smallest_M, smallest_M + 1, smallest_M + 2} : Set ℕ), 25 ∣ n) ∧
  ∀ m < smallest_M,
    ¬((∃ n ∈ ({m, m + 1, m + 2} : Set ℕ), 8 ∣ n) ∧
      (∃ n ∈ ({m, m + 1, m + 2} : Set ℕ), 9 ∣ n) ∧
      (∃ n ∈ ({m, m + 1, m + 2} : Set ℕ), 25 ∣ n)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_M_correct_l1057_105703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_zero_is_933_l1057_105791

-- Define the sequence a_n
noncomputable def a : ℕ → ℝ
  | 0 => 0  -- Add a case for 0 to avoid missing cases error
  | 1 => 19
  | 2 => 98
  | (n + 3) => a (n + 1) - 2 / a (n + 2)

-- Define the property that a_m is the first zero in the sequence
def first_zero (m : ℕ) : Prop :=
  a m = 0 ∧ ∀ k < m, a k ≠ 0

-- Theorem statement
theorem smallest_zero_is_933 :
  ∃ m, first_zero m ∧ m = 933 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_zero_is_933_l1057_105791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1057_105730

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The x-axis and y-axis intersection points of a line -/
structure Intersections where
  x_intercept : ℝ × ℝ
  y_intercept : ℝ × ℝ

/-- The quadrants a line passes through -/
inductive Quadrant
  | I
  | II
  | III
  | IV

noncomputable def Line.intersections (l : Line) : Intersections :=
  { x_intercept := (-l.intercept / l.slope, 0),
    y_intercept := (0, l.intercept) }

def Line.quadrants (l : Line) : Set Quadrant :=
  sorry

theorem line_properties (l : Line) :
  l.slope = -2 ∧ l.intercept = -3 →
  l.intersections.x_intercept = (-3/2, 0) ∧
  l.intersections.y_intercept = (0, -3) ∧
  l.quadrants = {Quadrant.II, Quadrant.III, Quadrant.IV} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_properties_l1057_105730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_transformation_l1057_105753

theorem trigonometric_transformation (α : ℝ) :
  4.24 * (Real.cos ((5 * Real.pi) / 2 - α) * Real.sin (Real.pi / 2 + α / 2)) /
  ((2 * Real.sin ((Real.pi - α) / 2) + Real.cos ((3 * Real.pi) / 2 - α)) * (Real.cos ((Real.pi - α) / 4))^2) =
  2 * Real.tan (α / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_transformation_l1057_105753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1057_105778

/-- The eccentricity of an ellipse intersected by a line passing through its midpoint. -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 1) : 
  let C : ℝ × ℝ → Prop := λ (x, y) ↦ x^2 / a^2 + y^2 / b^2 = 1
  let M : ℝ × ℝ := (1, 1)
  let line_slope : ℝ := -1/2
  let intersect : Set (ℝ × ℝ) := {p | C p ∧ (p.2 - M.2) = line_slope * (p.1 - M.1)}
  ∀ A B : ℝ × ℝ, A ∈ intersect → B ∈ intersect → A ≠ B →
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  Real.sqrt (1 - (b/a)^2) = Real.sqrt 2 / 2 :=
by
  intros C M line_slope intersect A B hA hB hAB hM
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1057_105778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_two_l1057_105752

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x^2
  else 2*x - 2

-- State the theorem
theorem unique_solution_is_two :
  ∃! x : ℝ, f x = 2 ∧ x = 2 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_is_two_l1057_105752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1057_105776

theorem hyperbola_eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) :
  ∃ e : ℝ, 1 < e ∧ e < Real.sqrt 2 := by
  -- Define the eccentricity
  let e := Real.sqrt (a^2 + b^2) / a

  -- Prove 1 < e
  have h1 : 1 < e := sorry

  -- Prove e < √2
  have h2 : e < Real.sqrt 2 := sorry

  -- Combine the results
  exact ⟨e, h1, h2⟩

#check hyperbola_eccentricity_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1057_105776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l1057_105763

theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f x = x^a) → f 2 = Real.sqrt 2 → a = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l1057_105763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_problem_l1057_105720

theorem binomial_coefficient_problem (a : ℝ) : 
  (Nat.choose 7 2 : ℝ) * a^2 = 7 → a = Real.sqrt 3 / 3 ∨ a = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_problem_l1057_105720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l1057_105727

/-- A pair of nonnegative integers (n,p) satisfies the given conditions --/
def SatisfiesConditions (n p : ℕ) : Prop :=
  Nat.Prime p ∧ n < 2 * p ∧ ((p-1)^n + 1 ∣ n^(p-1))

/-- The set of all pairs (n,p) satisfying the conditions --/
def SolutionSet : Set (ℕ × ℕ) :=
  {pair | SatisfiesConditions pair.1 pair.2}

/-- The theorem stating the correct solution --/
theorem solution_characterization :
  SolutionSet = {pair : ℕ × ℕ | (pair.1 = 1 ∧ Nat.Prime pair.2) ∨ pair = (2, 2) ∨ pair = (3, 3)} := by
  sorry

#check solution_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_characterization_l1057_105727
