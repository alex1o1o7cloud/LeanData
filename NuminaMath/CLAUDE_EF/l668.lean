import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_intersection_l668_66815

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 + y^2/4 = 1

-- Define the line that intersects C
def Line (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem curve_and_intersection :
  ∀ (x y k : ℝ),
  (∀ (x y : ℝ), C x y → 
    distance x y 0 (-Real.sqrt 3) + distance x y 0 (Real.sqrt 3) = 4) →
  (∃ (x1 y1 x2 y2 : ℝ), C x1 y1 ∧ C x2 y2 ∧ Line k x1 y1 ∧ Line k x2 y2) →
  (C x y ↔ x^2 + y^2/4 = 1) ∧
  ((∃ (x1 y1 x2 y2 : ℝ), C x1 y1 ∧ C x2 y2 ∧ Line k x1 y1 ∧ Line k x2 y2 ∧ 
    x1*x2 + y1*y2 = 0) → (k = 1/2 ∨ k = -1/2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_intersection_l668_66815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l668_66811

def sequence_a : ℕ → ℕ
  | 0 => 1  -- Add this case to handle Nat.zero
  | 1 => 1
  | (n + 1) => sequence_a n + n

theorem a_100_value : sequence_a 100 = 4951 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l668_66811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cubes_required_l668_66894

/-- Represents a cube with 6 faces, each containing a digit. -/
structure Cube where
  faces : Fin 6 → Fin 10

/-- The set of all possible 30-digit numbers. -/
def AllNumbers : Set (Fin 30 → Fin 10) :=
  {f | ∀ i, f i < 10}

/-- Given a set of cubes, determines if it can form all 30-digit numbers. -/
def CanFormAllNumbers (cubes : Finset Cube) : Prop :=
  ∀ n ∈ AllNumbers, ∃ (arrangement : Fin 30 → Cube × Fin 6),
    ∀ i, (arrangement i).1.faces (arrangement i).2 = n i ∧ (arrangement i).1 ∈ cubes

/-- The main theorem stating the minimum number of cubes required. -/
theorem min_cubes_required :
  ∃ (n : ℕ), n = 50 ∧ 
  (∃ (cubes : Finset Cube), Finset.card cubes = n ∧ CanFormAllNumbers cubes) ∧
  (∀ (m : ℕ) (cubes : Finset Cube), m < n → Finset.card cubes = m → ¬CanFormAllNumbers cubes) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cubes_required_l668_66894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l668_66871

open Real MeasureTheory

noncomputable def S₁ : ℝ := ∫ x in (0)..(π/2), cos x
noncomputable def S₂ : ℝ := ∫ x in (1)..2, 1/x
noncomputable def S₃ : ℝ := ∫ x in (1)..2, exp x

theorem integral_inequality : S₂ < S₁ ∧ S₁ < S₃ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_inequality_l668_66871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_rotated_legs_and_incircle_l668_66880

/-- 
For a right triangle with legs a and b, and hypotenuse c, 
the expression a + b - c represents both the length of the common part 
when the legs are rotated to lie on the hypotenuse and the diameter of 
the inscribed circle.
-/
theorem right_triangle_rotated_legs_and_incircle 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  a + b - c = 2 * ((a + b - c) / 2) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_rotated_legs_and_incircle_l668_66880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_S_to_T_l668_66840

-- Define the set T
def T : Set (ℝ × ℝ × ℝ) :=
  {p | p.fst ≥ 0 ∧ p.snd ≥ 0 ∧ p.snd.snd ≥ 0 ∧ p.fst + p.snd.fst + p.snd.snd = 1}

-- Define the supports relation
def supports (p : ℝ × ℝ × ℝ) (a b c : ℝ) : Prop :=
  (p.fst ≥ a ∧ p.snd.fst ≥ b) ∨ (p.fst ≥ a ∧ p.snd.snd ≥ c) ∨ (p.snd.fst ≥ b ∧ p.snd.snd ≥ c)

-- Define the set S
def S : Set (ℝ × ℝ × ℝ) :=
  {p ∈ T | supports p (1/3) (1/4) (1/5)}

-- State the theorem
theorem area_ratio_S_to_T : MeasureTheory.volume S / MeasureTheory.volume T = 19/120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_S_to_T_l668_66840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l668_66893

noncomputable def f (x : ℝ) := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x + 2 * Real.cos x ^ 2

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x ∧
    ∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (- Real.pi / 3 + k * Real.pi) (Real.pi / 6 + k * Real.pi))) ∧
  (∀ x : ℝ, x ∈ Set.Icc (- Real.pi / 3) (Real.pi / 12) → f x ≤ (Real.sqrt 3 + 3) / 2) ∧
  (∃ x : ℝ, x ∈ Set.Icc (- Real.pi / 3) (Real.pi / 12) ∧ f x = (Real.sqrt 3 + 3) / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l668_66893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_fifty_is_zero_l668_66813

open Real

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → x * g y - y * g x = g (x / y) + g (x * y)

/-- The main theorem -/
theorem g_fifty_is_zero
    (g : ℝ → ℝ)
    (h : FunctionalEquation g) :
    g 50 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_fifty_is_zero_l668_66813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l668_66859

/-- Definition of the function f(x) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := |x + 8/m| + |x - 2*m|

/-- Theorem stating the minimum value of f(x) and the range of m satisfying f(1) > 10 -/
theorem f_properties (m : ℝ) (h : m > 0) :
  (∃ (min : ℝ), ∀ x, f m x ≥ min ∧ ∃ x₀, f m x₀ = min) ∧
  (f m 1 > 10 ↔ m ∈ Set.Ioo 0 1 ∪ Set.Ioi 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l668_66859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_20_l668_66824

def sequence_a : ℕ → ℕ
  | 0 => 2  -- Add this case for n = 0
  | 1 => 2
  | n + 2 => if n % 2 = 0 then sequence_a (n + 1) + 2 else 2 * sequence_a (n + 1)

theorem a_5_equals_20 : sequence_a 5 = 20 := by
  -- Unfold the definition of sequence_a
  unfold sequence_a
  -- Simplify the expressions
  simp
  -- Evaluate the sequence step by step
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_20_l668_66824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l668_66835

/-- The area of a triangle given by three points in the coordinate plane -/
noncomputable def triangleArea (p q r : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := p
  let (x₂, y₂) := q
  let (x₃, y₃) := r
  (1/2) * abs (x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂))

/-- The area of triangle PQR with given coordinates is 31/2 -/
theorem triangle_PQR_area :
  triangleArea (3, -4) (-2, 5) (-1, -3) = 31/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l668_66835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_given_sin_double_l668_66825

theorem sin_plus_cos_given_sin_double (A : ℝ) (h1 : Real.sin (2 * A) = 2 / 3) (h2 : 0 < A) (h3 : A < π / 2) :
  Real.sin A + Real.cos A = Real.sqrt 15 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_given_sin_double_l668_66825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_investment_interest_l668_66838

/-- Calculates the interest earned on an investment with compound interest -/
noncomputable def interest_earned (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem lisa_investment_interest :
  round_to_nearest (interest_earned 500 0.02 10) = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_investment_interest_l668_66838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l668_66870

theorem inequality_solution (x : ℝ) : (x - 2) / (x - 4) ≥ 3 ↔ x ∈ Set.Ioc 4 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l668_66870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_number_in_overlapping_groups_l668_66877

theorem common_number_in_overlapping_groups (numbers : List ℝ) 
  (h1 : numbers.length = 8)
  (h2 : (numbers.take 5).sum / 5 = 6)
  (h3 : (numbers.drop 4).sum / 4 = 10)
  (h4 : numbers.sum / 8 = 8) : 
  numbers[4]! = 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_number_in_overlapping_groups_l668_66877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steves_speed_l668_66820

/-- Prove that given the race conditions, Steve's speed was 3.7 m/s -/
theorem steves_speed (john_initial_distance john_speed race_duration john_final_lead : Real) : 
  john_initial_distance = 16 →
  john_speed = 4.2 →
  race_duration = 36 →
  john_final_lead = 2 →
  (john_speed * race_duration - (john_initial_distance + john_final_lead)) / race_duration = 3.7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_steves_speed_l668_66820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_fair_dice_classical_model_l668_66874

/-- Represents a die with n faces -/
structure Die (n : ℕ) where
  faces : Fin n → ℕ

/-- A fair die has equally likely outcomes -/
def is_fair_die {n : ℕ} (d : Die n) : Prop :=
  ∀ (i j : Fin n), d.faces i = d.faces j

/-- The sample space of rolling two dice -/
def two_dice_sample_space {n : ℕ} (d1 d2 : Die n) : Set (Fin n × Fin n) :=
  Set.univ

/-- The event of getting a specific sum when rolling two dice -/
def sum_event {n : ℕ} (d1 d2 : Die n) (sum : ℕ) : Set (Fin n × Fin n) :=
  {p | d1.faces p.1 + d2.faces p.2 = sum}

/-- A probability model for rolling two dice -/
structure TwoDiceModel (n : ℕ) where
  die1 : Die n
  die2 : Die n
  fair1 : is_fair_die die1
  fair2 : is_fair_die die2

/-- Theorem: Rolling two fair dice is a classical probability model -/
theorem two_fair_dice_classical_model {n : ℕ} (model : TwoDiceModel n) :
  (∀ (i j : Fin n), model.die1.faces i = model.die2.faces j) →
  Finite (two_dice_sample_space model.die1 model.die2) →
  ∀ (sum : ℕ), Finite (sum_event model.die1 model.die2 sum) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_fair_dice_classical_model_l668_66874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_count_l668_66802

theorem triangle_third_side_count : ℕ := by
  -- Define the two known sides of the triangle
  let a : ℕ := 8
  let b : ℕ := 11

  -- Define the set of possible integer lengths for the third side
  let possible_lengths : Finset ℕ := Finset.range 19 \ Finset.range 4

  -- The count of possible lengths is the cardinality of the set
  have h : (possible_lengths.card : ℕ) = 15 := by sorry

  -- The result is the cardinality of the set of possible lengths
  exact 15


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_count_l668_66802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_length_is_ln2_l668_66855

open Real MeasureTheory Set

-- Define the parameterization functions
noncomputable def x (t : ℝ) : ℝ := ∫ (u : ℝ) in Ici t, (cos u / u)
noncomputable def y (t : ℝ) : ℝ := ∫ (u : ℝ) in Ici t, (sin u / u)

-- Define the domain of the parameter t
def t_domain : Set ℝ := Icc 1 2

-- State the theorem
theorem curve_length_is_ln2 :
  ∫ t in t_domain, sqrt ((deriv x t) ^ 2 + (deriv y t) ^ 2) = log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_length_is_ln2_l668_66855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sequence_length_exists_sequence_of_length_16_maximal_sequence_length_l668_66846

def is_valid_sequence (s : List ℤ) : Prop :=
  (∀ i, i + 6 < s.length → (List.sum (List.take 7 (List.drop i s)) > 0)) ∧
  (∀ i, i + 10 < s.length → (List.sum (List.take 11 (List.drop i s)) < 0)) ∧
  (∀ x ∈ s, x ≠ 0)

theorem max_sequence_length :
  ∀ s : List ℤ, is_valid_sequence s → s.length ≤ 16 :=
by sorry

theorem exists_sequence_of_length_16 :
  ∃ s : List ℤ, is_valid_sequence s ∧ s.length = 16 :=
by sorry

theorem maximal_sequence_length :
  (∃ s : List ℤ, is_valid_sequence s ∧ s.length = 16) ∧
  (∀ s : List ℤ, is_valid_sequence s → s.length ≤ 16) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sequence_length_exists_sequence_of_length_16_maximal_sequence_length_l668_66846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l668_66857

open Real

-- Define the vectors a and b
noncomputable def a (x : ℝ) : ℝ × ℝ := (sin (2 * x), 2 * (cos x) ^ 2 - 1)
noncomputable def b (θ : ℝ) : ℝ × ℝ := (sin θ, cos θ)

-- Define the function f
noncomputable def f (x θ : ℝ) : ℝ := (a x).1 * (b θ).1 + (a x).2 * (b θ).2

-- State the theorem
theorem problem_solution (θ : ℝ) (h1 : 0 < θ) (h2 : θ < π) (h3 : f (π/6) θ = 1) :
  θ = π/3 ∧
  (∀ x, f x θ = f (x + π) θ) ∧
  (∀ x ∈ Set.Icc (-π/6) (π/4), f x θ ≤ 1) ∧
  (f (π/6) θ = 1) ∧
  (∀ x ∈ Set.Icc (-π/6) (π/4), f x θ ≥ -1/2) ∧
  (f (-π/6) θ = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l668_66857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_problem_l668_66885

theorem cubic_roots_problem (u v c d : ℝ) : 
  (∃ w : ℝ, u^3 + c*u + d = 0 ∧ v^3 + c*v + d = 0 ∧ w^3 + c*w + d = 0) →
  (∃ w' : ℝ, (u+3)^3 + c*(u+3) + d + 153 = 0 ∧ 
             (v-2)^3 + c*(v-2) + d + 153 = 0 ∧ 
             w'^3 + c*w' + d + 153 = 0) →
  d = -13.125 ∨ |d - 39.448| < 0.001 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_problem_l668_66885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_dividing_line_segment_l668_66812

/-- Given two points M and N, and a point P on the line segment MN that divides it in a specific ratio, 
    this theorem proves that the x-coordinate of P is 7. -/
theorem point_dividing_line_segment (M N P : ℝ × ℝ) (s t : ℝ) :
  M = (3, 2) →
  N = (9, 5) →
  P = (s, t) →
  P ∈ Set.Icc M N →
  (dist M P) / (dist P N) = 4 / 2 →
  s = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_dividing_line_segment_l668_66812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_lim_revenue_l668_66895

/-- Calculates the total revenue for Mrs. Lim's milk sales --/
def calculate_total_revenue (
  yesterday_morning : ℕ
) (yesterday_evening : ℕ
) (morning_price : ℚ
) (evening_price : ℚ
) (delivery_fee : ℕ
) (storage_fee : ℕ
) (unsold_gallons : ℕ
) : ℚ :=
  let this_morning := yesterday_morning - 18
  let total_milk := yesterday_morning + yesterday_evening + this_morning
  let sold_milk := total_milk - unsold_gallons
  let average_price := (morning_price + evening_price) / 2
  let milk_revenue := (sold_milk : ℚ) * average_price
  let total_fees := (2 * delivery_fee + unsold_gallons * storage_fee : ℚ)
  milk_revenue - total_fees

/-- Theorem stating that Mrs. Lim's total revenue is $380 --/
theorem mrs_lim_revenue :
  calculate_total_revenue 68 82 (7/2) 4 20 10 24 = 380 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_lim_revenue_l668_66895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_integral_relation_l668_66899

open Set
open MeasureTheory

variable {a b c : ℝ}
variable {f : ℝ → ℝ}

theorem right_triangle_integral_relation
  (h_triangle : a^2 + b^2 = c^2)
  (h_order : a < b)
  (h_continuous : ContinuousOn f (Icc a b))
  (h_differentiable : DifferentiableOn ℝ f (Ioo a b))
  (h_nonzero_derivative : ∀ x ∈ Ioo a b, deriv f x ≠ 0)
  (h_equation : ∀ x ∈ Icc a b, (1/2) * f x * (x - a) * (x - b) = c^2) :
  (1/2) * (∫ x in a..b, f x * x^2) - 
  (1/2) * (a + b) * (∫ x in a..b, f x * x) + 
  (1/2) * a * b * (∫ x in a..b, f x) = 
  c^2 * (b - a) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_integral_relation_l668_66899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_l668_66823

noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

theorem work_completion 
  (a_days b_days abc_days c_days : ℝ)
  (ha : a_days = 4)
  (hb : b_days = 6)
  (habc : abc_days = 2)
  (h_total : work_rate a_days + work_rate b_days + work_rate c_days = work_rate abc_days) :
  c_days = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_l668_66823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_section_b_average_weight_l668_66891

/-- Proves that the average weight of section B is 60 kg given the conditions of the problem -/
theorem section_b_average_weight (students_a : ℕ) (students_b : ℕ) (avg_weight_a : ℝ) (avg_weight_total : ℝ) : 
  students_a = 40 →
  students_b = 30 →
  avg_weight_a = 50 →
  avg_weight_total = 54.285714285714285 →
  ((students_a : ℝ) * avg_weight_a + (students_b : ℝ) * ((students_a + students_b : ℝ) * avg_weight_total - (students_a : ℝ) * avg_weight_a) / (students_b : ℝ)) / ((students_a + students_b : ℝ)) = avg_weight_total →
  ((students_a + students_b : ℝ) * avg_weight_total - (students_a : ℝ) * avg_weight_a) / (students_b : ℝ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_section_b_average_weight_l668_66891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ZQY_measure_l668_66801

-- Define the points
variable (X Y Z W Q : ℝ × ℝ)

-- Define the conditions
axiom midpoint_XY : Z = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)
axiom midpoint_YZ : W = ((Y.1 + Z.1) / 2, (Y.2 + Z.2) / 2)

-- Define the semicircles
def semicircle_XY := {p : ℝ × ℝ | dist p X + dist p Y = dist X Y ∧ (p.2 - X.2) * (Y.1 - X.1) ≥ (p.1 - X.1) * (Y.2 - X.2)}
def semicircle_YZ := {p : ℝ × ℝ | dist p Y + dist p Z = dist Y Z ∧ (p.2 - Y.2) * (Z.1 - Y.1) ≥ (p.1 - Y.1) * (Z.2 - Y.2)}

-- Define the areas
noncomputable def area_XY := (Real.pi / 2) * (dist X Y / 2)^2
noncomputable def area_YZ := (Real.pi / 2) * (dist Y Z / 2)^2

-- Define the condition that ZQ splits the combined area into two equal parts
axiom equal_areas : area_XY X Y / 2 + area_YZ Y Z = (area_XY X Y + area_YZ Y Z) / 2

-- Define the angle ZQY
noncomputable def angle_ZQY := Real.arccos ((dist Z Q^2 + dist Q Y^2 - dist Z Y^2) / (2 * dist Z Q * dist Q Y))

-- State the theorem
theorem angle_ZQY_measure :
  angle_ZQY Z Q Y = 112.5 * Real.pi / 180 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_ZQY_measure_l668_66801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_matching_numbers_eq_203_l668_66844

/-- First numbering system (row-wise) -/
def f (i j : ℕ) : ℕ := 13 * (i - 1) + j

/-- Second numbering system (column-wise) -/
def g (i j : ℕ) : ℕ := 11 * (j - 1) + i

/-- The set of all pairs (i, j) where the numbering systems match -/
def matching_pairs : List (ℕ × ℕ) :=
  [(1, 1), (6, 7), (11, 13)]

/-- The sum of numbers that are the same in both numbering systems -/
def sum_of_matching_numbers : ℕ :=
  (matching_pairs.map (fun (p : ℕ × ℕ) => f p.1 p.2)).sum

theorem sum_of_matching_numbers_eq_203 : sum_of_matching_numbers = 203 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_matching_numbers_eq_203_l668_66844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonicity_l668_66842

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x - a) / Real.log (1/3)

theorem function_monotonicity (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ < -1/2 ∧ x₂ < -1/2 → (f a x₂ - f a x₁) / (x₂ - x₁) > 0) ↔ 
  -1 ≤ a ∧ a ≤ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonicity_l668_66842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_simplification_l668_66836

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_sum_simplification :
  i^8 + i^20 + i^(-32 : ℤ) + 2*i = 3 + 2*i :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sum_simplification_l668_66836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_condition_l668_66829

/-- Given a quadratic equation x^2 - (m+3)x + m+2 = 0 with roots x₁ and x₂,
    prove that (1/x₁ + 1/x₂ > 1/2 and x₁² + x₂² < 5) is equivalent to m ∈ (-4, -2) ∪ (-2, 0) -/
theorem quadratic_roots_condition (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, (∀ x, x^2 - (m+3)*x + m+2 = 0 → x = x₁ ∨ x = x₂) →
  (1/x₁ + 1/x₂ > 1/2 ∧ x₁^2 + x₂^2 < 5) ↔ 
  (m > -4 ∧ m < -2) ∨ (m > -2 ∧ m < 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_condition_l668_66829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_21_sqrt_7_over_8_l668_66809

/-- Triangle ABC with points D and E on side BC, where E is between B and D -/
structure TriangleABC where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  E_between_B_D : (E.1 - B.1) * (D.1 - E.1) ≥ 0

/-- Properties of the triangle -/
def TriangleProperties (t : TriangleABC) : Prop :=
  let (ax, ay) := t.A
  let (bx, _) := t.B
  let (cx, _) := t.C
  let (dx, _) := t.D
  let (ex, _) := t.E
  -- BE = 1
  (ex - bx = 1) ∧
  -- ED = DC = 3
  (dx - ex = 3) ∧
  (cx - dx = -3) ∧
  -- ∠BAD = ∠EAC = 90°
  ((ax - dx) * (ax - bx) + ay * ay = 0) ∧
  ((ax - ex) * (ax - cx) + ay * ay = 0)

/-- The area of the triangle -/
noncomputable def TriangleArea (t : TriangleABC) : ℝ :=
  let (ax, ay) := t.A
  let (bx, _) := t.B
  let (cx, _) := t.C
  (1/2) * abs (cx - bx) * abs ay

theorem triangle_area_is_21_sqrt_7_over_8 (t : TriangleABC) 
  (h : TriangleProperties t) : TriangleArea t = 21 * Real.sqrt 7 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_21_sqrt_7_over_8_l668_66809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_is_0_18_l668_66822

-- Define the slope of the line
def m : ℝ := -3

-- Define the x-intercept point
def x_intercept : ℝ × ℝ := (6, 0)

-- Define a line using slope and a point
def line (slope : ℝ) (p : ℝ × ℝ) (x : ℝ) : ℝ :=
  slope * (x - p.1) + p.2

-- Theorem: The y-intercept of the line is (0, 18)
theorem y_intercept_is_0_18 :
  line m x_intercept 0 = 18 := by
  -- Expand the definition of the line
  unfold line
  -- Simplify the expression
  simp [m, x_intercept]
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_is_0_18_l668_66822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_result_l668_66879

theorem complex_division_result : 
  (1 + 3 * Complex.I) / (1 - Complex.I) = -1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_division_result_l668_66879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l668_66881

-- Define the function f
def f (x : ℝ) : ℝ := 6 * x^2 + x - 1

-- Define α as a real number (since Lean doesn't have a built-in concept of "acute angle")
variable (α : ℝ)

-- State the theorem
theorem function_properties :
  (0 < α) → (α < π / 2) →  -- α is an acute angle
  f (Real.sin α) = 0 →        -- sin(α) is a zero of f(x)
  (Real.sin α = 1 / 3 ∧
   (Real.tan (π + α) * Real.cos (-α)) / (Real.cos (π / 2 - α) * Real.sin (π - α)) = 3 ∧
   Real.sin (α + π / 6) = (Real.sqrt 3 + 2 * Real.sqrt 2) / 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l668_66881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_l668_66831

/-- An equilateral triangle in the first quadrant -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  first_quadrant : A.1 ≥ 0 ∧ A.2 ≥ 0 ∧ B.1 ≥ 0 ∧ B.2 ≥ 0 ∧ C.1 ≥ 0 ∧ C.2 ≥ 0
  equilateral : ‖A - B‖ = ‖B - C‖ ∧ ‖B - C‖ = ‖C - A‖

/-- Points on the lines of the triangle -/
structure TrianglePoints where
  on_CA : ℝ × ℝ
  on_AB : ℝ × ℝ
  on_BC : ℝ × ℝ

/-- The centroid of a triangle -/
noncomputable def centroid (t : EquilateralTriangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

/-- The theorem to be proved -/
theorem centroid_sum (t : EquilateralTriangle) (p : TrianglePoints) 
  (h1 : p.on_CA = (4, 0)) (h2 : p.on_AB = (6, 0)) (h3 : p.on_BC = (12, 0)) :
  let g := centroid t
  g.1 + g.2 = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_sum_l668_66831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_finishes_in_four_days_l668_66888

noncomputable section

/-- The number of days A needs to finish the work alone -/
def a_days : ℝ := 5

/-- The number of days B needs to finish the work alone -/
def b_days : ℝ := 10

/-- The number of days A and B work together before A leaves -/
def days_together : ℝ := 2

/-- The work rate of A per day -/
noncomputable def a_rate : ℝ := 1 / a_days

/-- The work rate of B per day -/
noncomputable def b_rate : ℝ := 1 / b_days

/-- The combined work rate of A and B per day -/
noncomputable def combined_rate : ℝ := a_rate + b_rate

/-- The amount of work completed when A and B work together -/
noncomputable def work_completed : ℝ := combined_rate * days_together

/-- The amount of work remaining after A leaves -/
noncomputable def work_remaining : ℝ := 1 - work_completed

/-- The number of days B needs to finish the remaining work -/
noncomputable def days_for_b_to_finish : ℝ := work_remaining / b_rate

theorem b_finishes_in_four_days : 
  days_for_b_to_finish = 4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_finishes_in_four_days_l668_66888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalence_of_inequalities_l668_66867

theorem equivalence_of_inequalities (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (|a - b * Real.sqrt c| < 1 / (2 * b)) ↔ (|a^2 - b^2 * c| < Real.sqrt c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equivalence_of_inequalities_l668_66867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squared_distances_l668_66843

/-- Given collinear points A, B, C, D, and E in that order, with specified distances between them,
    this theorem states the minimum value of the sum of squared distances from these points
    to any point P in space. -/
theorem min_sum_squared_distances (A B C D E : ℝ) (P : ℝ) :
  A < B ∧ B < C ∧ C < D ∧ D < E →
  B - A = 2 →
  C - B = 2 →
  D - C = 3 →
  E - D = 7 →
  ∃ (min : ℝ), min = 133.2 ∧
    ∀ P, (P - A)^2 + (P - B)^2 + (P - C)^2 + (P - D)^2 + (P - E)^2 ≥ min :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squared_distances_l668_66843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l668_66806

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

-- Define the conditions given in the problem
def satisfies_conditions (t : Triangle) : Prop :=
  is_valid_triangle t ∧
  t.a = 2 * t.b ∧
  ∃ k, Real.sin t.A + Real.sin t.B = 2 * Real.sin t.C + k ∧
  (1/2) * t.b * t.c * Real.sin t.A = (8 * Real.sqrt 15) / 3

-- State the theorem
theorem triangle_problem (t : Triangle) (h : satisfies_conditions t) :
  Real.cos (t.B + t.C) = 1/4 ∧ t.c = 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l668_66806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_l668_66866

/-- The circle with center (1,2) and radius 2 -/
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

/-- The point M through which the tangent line passes -/
def M : ℝ × ℝ := (3, 1)

/-- A line is tangent to the circle if it touches the circle at exactly one point -/
def is_tangent (a b c : ℝ) : Prop :=
  ∃! (x y : ℝ), my_circle x y ∧ a * x + b * y + c = 0

theorem tangent_lines :
  (∀ (a b c : ℝ), is_tangent a b c ∧ a * M.1 + b * M.2 + c = 0 →
    (a = 3 ∧ b = -4 ∧ c = -5) ∨ (a = 1 ∧ b = 0 ∧ c = -3)) ∧
  is_tangent 3 (-4) (-5) ∧ is_tangent 1 0 (-3) ∧
  3 * M.1 + (-4) * M.2 + (-5) = 0 ∧ 1 * M.1 + 0 * M.2 + (-3) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_l668_66866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameter_range_l668_66814

/-- Given that the equation (x^2 / (2+m)) + (y^2 / (1-m)) = 1 represents an ellipse
    with foci on the x-axis, this theorem states the range of the real number m. -/
theorem ellipse_parameter_range (m : ℝ) :
  (∀ x y : ℝ, (x^2 / (2+m)) + (y^2 / (1-m)) = 1 → 
   ∃ a b : ℝ, a > b ∧ b > 0 ∧ 
   ∀ t : ℝ, (x = a * Real.cos t ∧ y = b * Real.sin t) ∨ (x = -a * Real.cos t ∧ y = b * Real.sin t)) ↔
  (-1/2 < m ∧ m < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parameter_range_l668_66814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l668_66807

theorem function_property (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x + y) = 2*x + f (f y - x)) : 
  ∃ a : ℝ, ∀ x : ℝ, f x = x - a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_property_l668_66807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l668_66834

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def is_acute_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧ 0 < t.B ∧ t.B < Real.pi/2 ∧ 0 < t.C ∧ t.C < Real.pi/2

def sides_are_roots (t : Triangle) : Prop :=
  t.a^2 - 2*t.a + 2 = 0 ∧ t.b^2 - 2*t.b + 2 = 0

def angles_satisfy_equation (t : Triangle) : Prop :=
  2 * Real.sin (t.A + t.B) - 1 = 0

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h_acute : is_acute_triangle t)
  (h_roots : sides_are_roots t)
  (h_angles : angles_satisfy_equation t) :
  t.C = Real.pi/3 ∧ t.c = Real.sqrt 6 ∧ (1/2 * t.a * t.b * Real.sin t.C) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l668_66834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_ratio_theorem_l668_66883

/-- Represents the density of a metal relative to water -/
structure MetalDensity where
  density : ℝ
  density_positive : density > 0

/-- Represents the composition of an alloy -/
structure AlloyComposition where
  gold : ℝ
  copper : ℝ
  silver : ℝ
  all_nonnegative : gold ≥ 0 ∧ copper ≥ 0 ∧ silver ≥ 0

/-- The density of gold relative to water -/
def gold_density : MetalDensity := ⟨11, by norm_num⟩

/-- The density of copper relative to water -/
def copper_density : MetalDensity := ⟨5, by norm_num⟩

/-- The density of silver relative to water -/
def silver_density : MetalDensity := ⟨7, by norm_num⟩

/-- The target density of the alloy relative to water -/
def target_density : MetalDensity := ⟨9, by norm_num⟩

/-- Calculates the density of an alloy given its composition -/
noncomputable def alloy_density (c : AlloyComposition) : ℝ :=
  (c.gold * gold_density.density + c.copper * copper_density.density + c.silver * silver_density.density) /
  (c.gold + c.copper + c.silver)

/-- Theorem: The ratio of gold:copper:silver that produces an alloy with the target density is 1:2:1 -/
theorem alloy_ratio_theorem :
  ∃ (c : AlloyComposition), alloy_density c = target_density.density ∧
  c.gold = 1 ∧ c.copper = 2 ∧ c.silver = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_ratio_theorem_l668_66883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birth_probability_l668_66830

def num_children : ℕ := 5

def prob_boy : ℚ := 1/2
def prob_girl : ℚ := 1/2

def prob_all_same_gender : ℚ := (1/2)^num_children

def prob_three_two : ℚ := (Nat.choose num_children 3) * ((1/2)^num_children)

def prob_four_one : ℚ := 2 * (Nat.choose num_children 1) * ((1/2)^num_children)

theorem birth_probability :
  prob_three_two = prob_four_one ∧
  prob_three_two > prob_all_same_gender := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birth_probability_l668_66830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_divisible_by_four_l668_66848

theorem at_least_one_divisible_by_four (a b c : ℤ) 
  (ha : Odd a) (hb : Odd b) (hc : Odd c) : 
  ∃ x ∈ ({a * b - 1, b * c - 1, c * a - 1} : Set ℤ), 4 ∣ x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_divisible_by_four_l668_66848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l668_66841

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (3 * x + 1) + 1 / (2 - x)

-- Define the domain
def domain : Set ℝ := {x | x > -1/3 ∧ x ≠ 2}

-- Theorem statement
theorem f_domain : 
  ∀ x : ℝ, f x ∈ Set.range f ↔ x ∈ domain := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l668_66841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l668_66818

/-- The complex number z satisfying the given equation lies in the second quadrant -/
theorem z_in_second_quadrant : 
  ∃ z : ℂ, z * (1 - Complex.I) = (1 + 2 * Complex.I) * Complex.I ∧ 
  z.re < 0 ∧ z.im > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_second_quadrant_l668_66818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_labeling_density_l668_66837

/-- A function that represents a valid labeling of an n×n grid -/
def ValidLabeling (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if three consecutive cells are labeled -/
def HasThreeConsecutive (n : ℕ) (l : ValidLabeling n) : Prop := sorry

/-- Counts the number of labeled cells in a grid -/
def CountLabeled (n : ℕ) (l : ValidLabeling n) : ℕ := sorry

/-- The maximum density of labeled cells -/
noncomputable def MaxDensity : ℝ := 2/3

theorem max_labeling_density :
  ∀ d : ℝ, d > 0 →
  (∀ n : ℕ, n > 0 →
    ∃ l : ValidLabeling n, ¬HasThreeConsecutive n l ∧ CountLabeled n l ≥ ⌊d * n^2⌋) →
  d ≤ MaxDensity := by
  sorry

#check max_labeling_density

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_labeling_density_l668_66837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_ABCD_l668_66808

-- Define the variables
noncomputable def A : ℝ := Real.sqrt 3000 + Real.sqrt 3001
noncomputable def B : ℝ := -Real.sqrt 3000 - Real.sqrt 3001
noncomputable def C : ℝ := Real.sqrt 3000 - Real.sqrt 3001
noncomputable def D : ℝ := Real.sqrt 3001 - Real.sqrt 3000

-- State the theorem
theorem product_ABCD : A * B * C * D = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_ABCD_l668_66808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_5_of_72_l668_66821

def factors (n : ℕ) : Finset ℕ := 
  Finset.filter (λ m => n % m = 0) (Finset.range (n + 1))

def lessThan5 (n : ℕ) : Bool := n < 5

theorem probability_factor_less_than_5_of_72 :
  let all_factors := factors 72
  let factors_less_than_5 := all_factors.filter (λ n => lessThan5 n = true)
  (factors_less_than_5.card : ℚ) / all_factors.card = 1 / 3 := by
  sorry

#eval factors 72
#eval (factors 72).filter (λ n => lessThan5 n = true)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_5_of_72_l668_66821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_l668_66851

/-- The system of differential equations and initial conditions -/
def system (x y : ℝ → ℝ) : Prop :=
  (∀ t, deriv x t = 3 * x t + 8 * y t) ∧
  (∀ t, deriv y t = -x t - 3 * y t) ∧
  x 0 = 6 ∧
  y 0 = -2

/-- The solution functions -/
noncomputable def x_solution (t : ℝ) : ℝ := 4 * Real.exp t + 2 * Real.exp (-t)
noncomputable def y_solution (t : ℝ) : ℝ := -(Real.exp t) - Real.exp (-t)

/-- Theorem stating that the solution functions satisfy the system -/
theorem solution_satisfies_system :
  system x_solution y_solution := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_l668_66851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insurance_cost_saving_and_appendix_l668_66884

/-- Represents an insurance cost-saving tool that decreases after each payout -/
structure AggregateSumInsurance : Type := mk ::

/-- Represents an insurance cost-saving tool that exempts the insurer from paying damages of a certain size -/
structure Deductible : Type := mk ::

/-- Represents a document containing the main provisions of the insurance contract -/
structure InsuranceRules : Type := mk ::

/-- Represents tools used to save on insurance costs -/
structure InsuranceCostSavingTools : Type :=
  (aggregate_sum : AggregateSumInsurance)
  (deductible : Deductible)

/-- Represents the appendix to the insurance contract -/
def InsuranceContractAppendix : Type := InsuranceRules

theorem insurance_cost_saving_and_appendix 
  (aggregate_sum : AggregateSumInsurance) 
  (deductible : Deductible) 
  (insurance_rules : InsuranceRules) :
  ∃ (tools : InsuranceCostSavingTools) (appendix : InsuranceContractAppendix),
    tools.aggregate_sum = aggregate_sum ∧
    tools.deductible = deductible ∧
    appendix = insurance_rules := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_insurance_cost_saving_and_appendix_l668_66884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l668_66897

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 6))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 9/2 ∨ x > 9/2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l668_66897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l668_66854

/-- Given a function g(x) = (2ax - 3a) / (3dx - 2d) where a, d ≠ 0,
    if g(g(x)) = x for all x in the domain of g, then 2a - 2d = 0 -/
theorem inverse_function_condition (a d : ℝ) (ha : a ≠ 0) (hd : d ≠ 0) :
  let g := λ x ↦ (2 * a * x - 3 * a) / (3 * d * x - 2 * d)
  (∀ x, x ∈ Set.range g → g (g x) = x) →
  2 * a - 2 * d = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_condition_l668_66854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_triangle_area_l668_66896

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
axiom triangle_condition (t : Triangle) : 
  4 * Real.sin ((t.B + t.C) / 2) ^ 2 - Real.cos (2 * t.A) = 7/2

axiom side_sum (t : Triangle) : t.a + t.c = 3 * Real.sqrt 3 / 2

axiom side_b (t : Triangle) : t.b = Real.sqrt 3

-- Theorem statements
theorem angle_A_measure (t : Triangle) : t.A = Real.pi / 3 := by sorry

theorem triangle_area (t : Triangle) : 
  (1/2) * t.b * t.c * Real.sin t.A = 15 * Real.sqrt 3 / 32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_triangle_area_l668_66896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l668_66805

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.arcsin (x - 1)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-Real.pi/2) (Real.pi/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l668_66805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_increase_percentage_l668_66817

/-- The percentage increase needed to restore a price to 120% of its original value after a 15% reduction -/
theorem price_increase_percentage (original_price : ℝ) (original_price_pos : original_price > 0) :
  let reduced_price := original_price * (1 - 0.15)
  let target_price := original_price * 1.2
  let increase_factor := target_price / reduced_price
  ∃ ε > 0, abs ((increase_factor - 1) * 100 - 41.18) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_increase_percentage_l668_66817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_water_usage_l668_66890

-- Define the water usage types
structure WaterUsage where
  spring : ℝ
  tap : ℝ

-- Define the pricing function for spring water
noncomputable def springWaterCost (x : ℝ) : ℝ :=
  if x ≤ 5 then 8 * x
  else if x ≤ 8 then 40 + 12 * (x - 5)
  else 76 + 16 * (x - 8)

-- Define the pricing function for tap water
def tapWaterCost (x : ℝ) : ℝ := 2 * x

-- Define the total cost function
noncomputable def totalCost (usage : WaterUsage) : ℝ :=
  springWaterCost usage.spring + tapWaterCost usage.tap

-- State the theorem
theorem unique_water_usage :
  ∃! usage : WaterUsage,
    usage.spring + usage.tap = 16 ∧
    totalCost usage = 72 ∧
    usage.spring = 6 ∧
    usage.tap = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_water_usage_l668_66890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_square_range_l668_66847

theorem count_integers_in_square_range : 
  ∃! n : ℕ, n = (Finset.filter (fun x => 225 ≤ x^2 ∧ x^2 ≤ 400) (Finset.range 401)).card ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_square_range_l668_66847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_receives_correct_amount_l668_66832

/-- The amount each friend receives when a person wins $100 and gives away 50% equally among 3 friends -/
def amount_per_friend (total_winnings : ℚ) (giveaway_percentage : ℚ) (num_friends : ℕ+) : ℚ :=
  (total_winnings * giveaway_percentage) / num_friends

/-- Theorem stating that each friend receives $16.67 (rounded to two decimal places) -/
theorem friend_receives_correct_amount :
  let result := amount_per_friend 100 (1/2) 3
  ⌊result * 100⌋ / 100 = 167 / 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_receives_correct_amount_l668_66832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_p_properties_l668_66872

/-- Circle P with diameter endpoints (3, 5) and (-5, -3) -/
structure CircleP where
  endpoint1 : ℝ × ℝ := (3, 5)
  endpoint2 : ℝ × ℝ := (-5, -3)

/-- The center of CircleP -/
noncomputable def center (c : CircleP) : ℝ × ℝ :=
  ((c.endpoint1.1 + c.endpoint2.1) / 2, (c.endpoint1.2 + c.endpoint2.2) / 2)

/-- The radius of CircleP -/
noncomputable def radius (c : CircleP) : ℝ :=
  Real.sqrt ((center c).1 - c.endpoint1.1)^2 + ((center c).2 - c.endpoint1.2)^2

theorem circle_p_properties (c : CircleP) :
  center c = (-1, 1) ∧ radius c = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_p_properties_l668_66872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_zeros_imply_omega_range_l668_66852

theorem sine_zeros_imply_omega_range (ω : ℝ) :
  ω > 0 →
  (∃ (s : Finset ℝ), s.card = 11 ∧ 
    (∀ x ∈ s, x ∈ Set.Icc (-π/2) (π/2) ∧ Real.sin (ω * x) = 0) ∧
    (∀ x ∈ Set.Icc (-π/2) (π/2), Real.sin (ω * x) = 0 → x ∈ s)) →
  10 ≤ ω ∧ ω < 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_zeros_imply_omega_range_l668_66852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l668_66828

-- Define the circles and line
def circle_C (m : ℝ) := {(x, y) : ℝ × ℝ | x^2 + y^2 - 4*x + m = 0}
def circle_2 := {(x, y) : ℝ × ℝ | (x - 3)^2 + (y + 2*Real.sqrt 2)^2 = 4}
def line := {(x, y) : ℝ × ℝ | 3*x - 4*y + 4 = 0}

-- Define the condition of external tangency
def externally_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), (x, y) ∈ circle_C m ∧ (x, y) ∈ circle_2

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |3*x - 4*y + 4| / Real.sqrt 25

-- State the theorem
theorem max_distance_to_line :
  ∃ (m : ℝ), externally_tangent m →
    (∀ (x y : ℝ), (x, y) ∈ circle_C m → distance_to_line x y ≤ 3) ∧
    (∃ (x y : ℝ), (x, y) ∈ circle_C m ∧ distance_to_line x y = 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l668_66828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cities_distance_l668_66845

/-- Given a map distance and a scale, calculate the actual distance between cities. -/
noncomputable def actual_distance (map_distance : ℝ) (scale_map : ℝ) (scale_actual : ℝ) : ℝ :=
  map_distance * (scale_actual / scale_map)

/-- Theorem: The actual distance between cities is 216 miles. -/
theorem cities_distance : 
  let map_distance : ℝ := 18
  let scale_map : ℝ := 0.5
  let scale_actual : ℝ := 6
  actual_distance map_distance scale_map scale_actual = 216 := by
  -- Unfold the definition of actual_distance
  unfold actual_distance
  -- Simplify the arithmetic
  simp [mul_div_assoc]
  -- Evaluate the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cities_distance_l668_66845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_theorem_l668_66858

-- Define a triangle ABC
structure Triangle (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C : V)

-- Define the concept of an angle
noncomputable def angle {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (A B C : V) : ℝ := sorry

-- Define the concept of an angle bisector
def is_angle_bisector {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (A B C K : V) : Prop := sorry

-- Define the theorem
theorem angle_bisector_theorem {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
    (t : Triangle V) (K : V) :
  angle t.A t.B t.C = 2 * angle t.A t.B t.C + angle t.B t.A t.C →
  is_angle_bisector t.A t.B t.C K →
  dist t.B K = dist t.B t.C →
  angle K t.B t.C = 2 * angle K t.B t.A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_theorem_l668_66858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l668_66856

noncomputable def f (x : ℝ) : ℝ := 
  (2 - Real.cos (Real.pi/4 * (1-x)) + Real.sin (Real.pi/4 * (1-x))) / (x^2 + 4*x + 5)

theorem max_value_of_f :
  ∃ (M : ℝ), M = 2 + Real.sqrt 2 ∧
  (∀ x : ℝ, -4 ≤ x ∧ x ≤ 0 → f x ≤ M) ∧
  (∃ x : ℝ, -4 ≤ x ∧ x ≤ 0 ∧ f x = M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l668_66856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l668_66865

noncomputable def f (x : ℝ) := 1 / Real.sqrt (2 - x) + Real.log (x + 1)

theorem domain_of_f :
  ∀ x : ℝ, (x > -1 ∧ x < 2) ↔ (2 - x > 0 ∧ x + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l668_66865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l668_66853

/-- The asymptote equations of a hyperbola with given properties -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.sqrt (a^2 + b^2) / a = Real.sqrt 3) →
  (∃ k : ℝ, k = Real.sqrt 2 ∧ 
    ∀ x : ℝ, (y = k * x ∨ y = -k * x) ↔ y^2 / x^2 = b^2 / a^2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l668_66853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_function_l668_66804

theorem max_value_trig_function :
  (∀ x : ℝ, Real.sqrt 3 * Real.sin x + Real.cos x ≤ 2) ∧
  (∃ x : ℝ, Real.sqrt 3 * Real.sin x + Real.cos x = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_function_l668_66804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_on_y_axis_hyperbola_asymptotes_two_straight_lines_three_correct_statements_l668_66886

-- Define the curve C: mx² + ny² = 1
def C (m n : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | m * p.1^2 + n * p.2^2 = 1}

-- Statement 1
theorem ellipse_foci_on_y_axis (m n : ℝ) (h : m > n ∧ n > 0) :
  ∃ a b : ℝ, a > b ∧ C m n = {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1} :=
sorry

-- Statement 2
theorem hyperbola_asymptotes (m n : ℝ) (h : m * n < 0) :
  ∃ k : ℝ, C m n = {p : ℝ × ℝ | p.2 = k * p.1 ∨ p.2 = -k * p.1} ∧ k^2 = -m/n :=
sorry

-- Statement 3
theorem two_straight_lines (n : ℝ) (h : n > 0) :
  C 0 n = {p : ℝ × ℝ | p.2 = 1/Real.sqrt n ∨ p.2 = -1/Real.sqrt n} :=
sorry

-- Theorem stating that all three statements are correct
theorem three_correct_statements :
  (∀ m n : ℝ, m > n ∧ n > 0 → ∃ a b : ℝ, a > b ∧ C m n = {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}) ∧
  (∀ m n : ℝ, m * n < 0 → ∃ k : ℝ, C m n = {p : ℝ × ℝ | p.2 = k * p.1 ∨ p.2 = -k * p.1} ∧ k^2 = -m/n) ∧
  (∀ n : ℝ, n > 0 → C 0 n = {p : ℝ × ℝ | p.2 = 1/Real.sqrt n ∨ p.2 = -1/Real.sqrt n}) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_on_y_axis_hyperbola_asymptotes_two_straight_lines_three_correct_statements_l668_66886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l668_66862

def plane (x y z : ℝ) : Prop := 5 * x - 3 * y + 2 * z = 40

noncomputable def distance (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2)

theorem closest_point_on_plane :
  let x₀ := 92 / 19
  let y₀ := -2 / 19
  let z₀ := 90 / 19
  plane x₀ y₀ z₀ ∧
  ∀ x y z, plane x y z →
    distance x y z 3 1 4 ≥ distance x₀ y₀ z₀ 3 1 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_plane_l668_66862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_g_sum_zero_l668_66887

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := f (x - 1) + 1

-- Theorem stating that g is an odd function
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  -- The proof steps would go here
  sorry

-- Additional theorem to show that g(-x) + g(x) = 0 for all x
theorem g_sum_zero : ∀ x : ℝ, g (-x) + g x = 0 := by
  intro x
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_g_sum_zero_l668_66887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_result_l668_66849

-- Define the function g
noncomputable def g (x : ℝ) : ℝ :=
  if x > 5 then Real.sqrt (x + 1)
  else x^3

-- State the theorem
theorem g_composition_result :
  g (g (g 3)) = Real.sqrt 6.29 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_result_l668_66849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_workers_needed_l668_66810

/-- Represents the project parameters and progress -/
structure ProjectData where
  totalDays : ℕ
  daysPassed : ℕ
  percentComplete : ℚ
  initialWorkers : ℕ

/-- Calculates the minimum number of workers needed to complete the project on time -/
def minWorkersNeeded (data : ProjectData) : ℕ :=
  let remainingDays := data.totalDays - data.daysPassed
  let remainingWork := 1 - data.percentComplete
  let dailyWorkRate := data.percentComplete / data.daysPassed
  let workPerWorker := dailyWorkRate / data.initialWorkers
  (((remainingWork / remainingDays) / workPerWorker).ceil).toNat

/-- Theorem stating that for the given project data, 5 workers are needed -/
theorem five_workers_needed :
  let data : ProjectData := {
    totalDays := 40,
    daysPassed := 10,
    percentComplete := 2/5,
    initialWorkers := 10
  }
  minWorkersNeeded data = 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_workers_needed_l668_66810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_theorem_l668_66800

/-- Calculates the profit percentage without discount given the discount percentage and profit percentage with discount -/
noncomputable def profit_without_discount (discount_percent : ℝ) (profit_with_discount_percent : ℝ) : ℝ :=
  let selling_price_with_discount := 100 + profit_with_discount_percent
  let selling_price_without_discount := selling_price_with_discount / (1 - discount_percent / 100)
  (selling_price_without_discount - 100) / 100 * 100

/-- Theorem: If a shopkeeper offers a 5% discount and earns a 23.5% profit, 
    then the profit percentage without the discount would be 30% -/
theorem shopkeeper_profit_theorem :
  profit_without_discount 5 23.5 = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_theorem_l668_66800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_eighteen_consecutive_good_l668_66833

/-- A natural number is good if it has exactly two prime divisors -/
def is_good (n : ℕ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ∣ n ∧ q ∣ n ∧
  ∀ r : ℕ, Nat.Prime r → r ∣ n → (r = p ∨ r = q)

/-- There does not exist a sequence of 18 consecutive natural numbers where each number is good -/
theorem no_eighteen_consecutive_good :
  ¬ ∃ k : ℕ, ∀ i : ℕ, i < 18 → is_good (k + i) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_eighteen_consecutive_good_l668_66833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_plate_mass_l668_66819

/-- The mass of a square plate with density proportional to the sum of distances from diagonals -/
theorem square_plate_mass (a K : ℝ) (h_a : a > 0) (h_K : K > 0) :
  let ρ : ℝ × ℝ → ℝ := fun p ↦ K * (p.1 + p.2)
  let Ω : Set (ℝ × ℝ) := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ a ∧ 0 ≤ p.2 ∧ p.2 ≤ a}
  (∫ p in Ω, ρ p) = (4/3) * K * a^3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_plate_mass_l668_66819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_48π_l668_66869

-- Define the points and distances
def A : ℝ := 0
def B : ℝ := 2
def C : ℝ := 6
def D : ℝ := 12
def E : ℝ := 16
def F : ℝ := 18

-- Define the function to calculate semi-circle area
noncomputable def semicircle_area (d : ℝ) : ℝ := (Real.pi * d^2) / 8

-- Define the shaded area function
noncomputable def shaded_area : ℝ :=
  semicircle_area (F - A) - semicircle_area (B - A) - semicircle_area (F - E) +
  semicircle_area (C - B) + semicircle_area (D - C) + semicircle_area (E - D)

-- Theorem statement
theorem shaded_area_is_48π : shaded_area = 48 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_48π_l668_66869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_and_intersection_l668_66861

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point P
def P : ℝ × ℝ := (1, 2)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem circle_tangent_and_intersection :
  -- Part 1
  (∀ x y : ℝ, y = 2 → (C x y → x = 1)) ∧
  (∀ x y : ℝ, 4*x + 3*y - 10 = 0 → (C x y → (x = 1 ∧ y = 2))) ∧
  -- Part 2
  (∃ A B : ℝ × ℝ, C A.1 A.2 ∧ C B.1 B.2 ∧
    (3*A.1 - 4*A.2 + 5 = 0) ∧ (3*B.1 - 4*B.2 + 5 = 0) ∧
    distance A B = 2*Real.sqrt 3 ∧
    (3*P.1 - 4*P.2 + 5 = 0)) ∧
  (∃ A B : ℝ × ℝ, C A.1 A.2 ∧ C B.1 B.2 ∧
    A.1 = 1 ∧ B.1 = 1 ∧
    distance A B = 2*Real.sqrt 3 ∧
    P.1 = 1) ∧
  -- Part 3
  (∀ x₀ y₀ x y : ℝ, C x₀ y₀ →
    x = x₀/2 ∧ y = y₀ →
    x^2 + y^2/4 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_and_intersection_l668_66861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_small_fermat_solution_l668_66816

theorem no_small_fermat_solution : 
  ∀ (x y z k : ℕ), x > 0 → y > 0 → z > 0 → k > 0 →
  (x : ℤ)^k + (y : ℤ)^k = (z : ℤ)^k → x ≥ k ∨ y ≥ k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_small_fermat_solution_l668_66816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_max_area_l668_66839

/-- A triangle with a fixed base and perimeter -/
structure FixedBasePerimeterTriangle where
  base : ℝ
  perimeter : ℝ
  side1 : ℝ
  side2 : ℝ
  base_positive : 0 < base
  perimeter_positive : 0 < perimeter
  sides_positive : 0 < side1 ∧ 0 < side2
  perimeter_constraint : base + side1 + side2 = perimeter

/-- The area of a triangle given its three sides -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with fixed base and perimeter is maximized when it is isosceles -/
theorem isosceles_max_area (t : FixedBasePerimeterTriangle) :
  ∀ (other : FixedBasePerimeterTriangle), 
    other.base = t.base → 
    other.perimeter = t.perimeter → 
    t.side1 = t.side2 →
    triangle_area t.base t.side1 t.side2 ≥ triangle_area other.base other.side1 other.side2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_max_area_l668_66839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excircle_centers_distance_l668_66889

-- Define a triangle inscribed in a unit circle
structure InscribedTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  inscribed : (A.1 - 0)^2 + (A.2 - 0)^2 = 1 ∧
              (B.1 - 0)^2 + (B.2 - 0)^2 = 1 ∧
              (C.1 - 0)^2 + (C.2 - 0)^2 = 1

-- Define the center of an excircle
def excircleCenter (t : InscribedTriangle) (v : Fin 3) : ℝ × ℝ := sorry

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem excircle_centers_distance (t : InscribedTriangle) :
  ∀ i j, i ≠ j → 0 < distance (excircleCenter t i) (excircleCenter t j) ∧
                 distance (excircleCenter t i) (excircleCenter t j) < 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_excircle_centers_distance_l668_66889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equidistant_from_B_and_C_l668_66875

noncomputable def point_A : ℝ × ℝ × ℝ := (0, 1/2, 0)
def point_B : ℝ × ℝ × ℝ := (3, 0, 3)
def point_C : ℝ × ℝ × ℝ := (0, 2, 4)

noncomputable def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

theorem A_equidistant_from_B_and_C :
  distance point_A point_B = distance point_A point_C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equidistant_from_B_and_C_l668_66875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l668_66873

/-- Given a hyperbola with equation x²/a² - y² = 1 where a > 0 and eccentricity √2, 
    its asymptote equation is y = ±x -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) :
  let e := Real.sqrt 2  -- eccentricity
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 = 1
  let asymptote := fun (x y : ℝ) ↦ y = x ∨ y = -x
  (∀ x y, hyperbola x y → e = Real.sqrt (1 + 1/a^2)) →
  (∀ x y, asymptote x y ↔ hyperbola x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l668_66873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_garden_area_l668_66803

/-- Given a rectangular field with specific dimensions and a circular garden,
    prove that the area of the garden is 56.25π square meters. -/
theorem circular_garden_area (w : ℝ) : 
  w > 0 →  -- width is positive
  (2 * w - 3) > 0 →  -- length is positive
  2 * (2 * w - 3) + 2 * w = 84 →  -- perimeter condition
  π * (w / 2)^2 = 56.25 * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_garden_area_l668_66803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_3375_l668_66882

theorem cube_root_3375 :
  ∃ (a b : ℕ+), (a : ℝ) * (b : ℝ)^(1/3) = 3375^(1/3) ∧
  (∀ (c d : ℕ+), (c : ℝ) * (d : ℝ)^(1/3) = 3375^(1/3) → b ≤ d) ∧
  a = 15 ∧ b = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_3375_l668_66882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_small_triangle_l668_66878

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is inside or on the boundary of a unit square -/
def isInUnitSquare (p : Point) : Prop :=
  0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1

/-- Checks if three points are collinear -/
def areCollinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Calculates the area of a triangle formed by three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  abs ((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2)

/-- The main theorem to be proved -/
theorem exists_small_triangle (points : Finset Point) :
  points.card = 101 →
  (∀ p, p ∈ points → isInUnitSquare p) →
  (∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points → p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬areCollinear p1 p2 p3) →
  ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ triangleArea p1 p2 p3 ≤ 0.01 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_small_triangle_l668_66878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_point_infinitely_many_k_with_max_l668_66827

-- Define the function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log x - k * Real.sin x - 1

-- Theorem 1: For k ≥ 0, f(x) has at least one zero point in (0, π]
theorem f_has_zero_point (k : ℝ) (h : k ≥ 0) : 
  ∃ x : ℝ, x ∈ Set.Ioo 0 Real.pi ∧ f k x = 0 := by sorry

-- Theorem 2: There exist infinitely many k such that f(x) has a maximum value in (0, π]
theorem infinitely_many_k_with_max :
  ∃ S : Set ℝ, Set.Infinite S ∧ ∀ k ∈ S, ∃ x : ℝ, x ∈ Set.Ioo 0 Real.pi ∧ 
    ∀ y ∈ Set.Ioo 0 Real.pi, f k y ≤ f k x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_point_infinitely_many_k_with_max_l668_66827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_leq_6_range_of_a_for_nonempty_solution_l668_66864

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x + 3|

-- Theorem 1: Solution set of f(x) ≤ 6
theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = Set.Icc (-5/2) (1/2) := by sorry

-- Theorem 2: Range of a for non-empty solution set of f(x) ≤ |a-1|
theorem range_of_a_for_nonempty_solution :
  {a : ℝ | ∃ x, f x ≤ |a - 1|} = {a : ℝ | a ≥ 3 ∨ a ≤ -1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_leq_6_range_of_a_for_nonempty_solution_l668_66864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_wins_iff_k_lt_winning_value_l668_66863

/-- The number of zeroes in the circle -/
def n : Nat := 80

/-- The number of consecutive numbers B selects -/
def m : Nat := 10

/-- The sum A adds each turn -/
noncomputable def sum_added : ℝ := 1

/-- The winning condition for A -/
noncomputable def winning_value : ℝ := 503 / 140

/-- Represents the state of the game -/
def GameState := Fin n → ℝ

/-- A's strategy is a function that takes the current state and returns the new state after A's move -/
def Strategy := GameState → GameState

/-- B's move: selects m consecutive numbers with the largest sum and reduces them to 0 -/
noncomputable def b_move (state : GameState) : GameState := sorry

/-- Checks if A has won -/
def has_won (k : ℝ) (state : GameState) : Prop :=
  ∃ i, state i ≥ k

/-- The main theorem: A can always win if and only if k < winning_value -/
theorem a_wins_iff_k_lt_winning_value (k : ℝ) :
  (∃ (strategy : Strategy), ∀ (initial_state : GameState),
    ∃ (t : ℕ), has_won k ((b_move ∘ strategy)^[t] initial_state)) ↔
  k < winning_value := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_wins_iff_k_lt_winning_value_l668_66863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l668_66898

/-- Two lines ax + by + c = 0 and dx + ey + f = 0 are parallel if and only if a/b = d/e -/
axiom parallel_lines (a b c d e f : ℝ) : 
  (a * e = b * d) ↔ (∀ (x y : ℝ), a*x + b*y + c = 0 ↔ d*x + e*y + f = 0)

theorem parallel_lines_m_value : 
  ∃ m : ℝ, (2 * m = -(m - 3)) ↔ 
    (∀ (x y : ℝ), 2*m*x + y + 6 = 0 ↔ (m - 3)*x - y + 7 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l668_66898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l668_66860

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Define for 0 to cover all natural numbers
  | 1 => 1
  | (n + 2) => 2 * sequence_a (n + 1) / (sequence_a (n + 1) + 2)

theorem a_10_value : sequence_a 10 = 2/11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l668_66860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l668_66876

noncomputable def f (x : ℝ) := 4 * Real.sin (2 * x + Real.pi / 3)

theorem f_properties :
  (∀ x, f x = 4 * Real.cos (2 * x - Real.pi / 6)) ∧
  (∀ t, f (-Real.pi / 6 + t) = f (-Real.pi / 6 - t)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l668_66876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_theorem_l668_66850

/-- Regular octagon with side length 16 cm -/
noncomputable def regular_octagon : ℝ := 16

/-- Area of one isosceles right triangle with hypotenuse 16 cm -/
noncomputable def triangle_area : ℝ := (1/2) * regular_octagon * regular_octagon

/-- Number of triangles -/
def num_triangles : ℕ := 8

/-- Total area of all triangles -/
noncomputable def total_triangle_area : ℝ := num_triangles * triangle_area

/-- Theorem: The difference in area between the outer region (excluding the inner octagon) 
    and the inner octagon in the described configuration is 512 square centimeters -/
theorem area_difference_theorem : 
  total_triangle_area - (2 * regular_octagon * regular_octagon - total_triangle_area) = 512 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_difference_theorem_l668_66850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_bunny_unfriendly_l668_66826

def is_bunny_unfriendly (n : ℕ+) : Prop :=
  ∀ a d : ℕ+, ∃ k : ℕ, k ≤ 2013 ∧
    (a + k * d ≥ n ∨ ¬(Nat.Coprime (a + k * d) n))

def product_of_primes_less_than (m : ℕ) : ℕ :=
  (List.filter (fun p ↦ Nat.Prime p ∧ p < m) (List.range m)).prod

theorem max_bunny_unfriendly :
  ∃ n : ℕ+, is_bunny_unfriendly n ∧
    (∀ m : ℕ+, is_bunny_unfriendly m → m ≤ n) ∧
    n = 2013 * product_of_primes_less_than 2014 := by
  sorry

#check max_bunny_unfriendly

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_bunny_unfriendly_l668_66826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l668_66868

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sin (abs x)

-- State the theorem
theorem f_range : Set.range f = Set.Icc (-2) 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l668_66868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_equality_l668_66892

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : EuclideanSpace ℝ (Fin 2))

-- Define the angles
noncomputable def angle (Q : Quadrilateral) (X Y Z : Fin 4) : ℝ :=
  sorry

-- Define the theorem
theorem quadrilateral_equality 
  (Q : Quadrilateral)
  (α β γ δ ε : ℝ)
  (h1 : α = angle Q 3 0 1)
  (h2 : β = angle Q 0 3 1)
  (h3 : γ = angle Q 0 2 1)
  (h4 : δ = angle Q 3 1 2)
  (h5 : ε = angle Q 3 1 0)
  (h6 : α < π / 2)
  (h7 : β + γ = π / 2)
  (h8 : δ + 2 * ε = π) :
  (dist Q.D Q.B + dist Q.B Q.C)^2 = (dist Q.A Q.D)^2 + (dist Q.A Q.C)^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_equality_l668_66892
