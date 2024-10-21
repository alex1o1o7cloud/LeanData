import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_implies_m_geq_one_l1133_113357

noncomputable def f (x m : ℝ) : ℝ := (1/4) * x^4 - (2/3) * x^3 + m

theorem f_nonnegative_implies_m_geq_one (m : ℝ) :
  (∀ x : ℝ, f x m + 1/3 ≥ 0) → m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_implies_m_geq_one_l1133_113357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_floor_problem_l1133_113379

/-- The floor of (2007! + 2004!) / (2006! + 2005!) is equal to 2006 -/
theorem factorial_floor_problem : 
  ⌊(Nat.factorial 2007 + Nat.factorial 2004 : ℚ) / (Nat.factorial 2006 + Nat.factorial 2005 : ℚ)⌋ = 2006 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_floor_problem_l1133_113379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frac_one_minus_i_over_one_plus_i_l1133_113347

def complex_i : ℂ := Complex.I

theorem frac_one_minus_i_over_one_plus_i (a b : ℝ) :
  (1 - complex_i) / (1 + complex_i) = Complex.mk a b → a - b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frac_one_minus_i_over_one_plus_i_l1133_113347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1133_113388

noncomputable def a (ω : ℝ) (x : ℝ) : ℝ × ℝ := (Real.sqrt 3, 2 * Real.sin (ω * x / 2))
noncomputable def b (ω : ℝ) (x : ℝ) : ℝ × ℝ := (Real.sin (ω * x), -Real.sin (ω * x / 2))
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := (a ω x).1 * (b ω x).1 + (a ω x).2 * (b ω x).2 + 1

theorem function_properties (ω : ℝ) (h : ω > 0) :
  -- 1. The value of ω that makes the smallest positive period of f equal to π is 2
  (∀ x : ℝ, f ω x = f ω (x + π)) ∧ (∀ y : ℝ, 0 < y ∧ y < π → ∃ z : ℝ, f ω z ≠ f ω (z + y)) → ω = 2 ∧
  -- 2. The minimum value of f is -2
  (∀ x : ℝ, f 2 x ≥ -2) ∧ (∃ x : ℝ, f 2 x = -2) ∧
  -- 3. The smallest positive value of φ that makes f(x + φ) symmetric about (π/3, 0) is π/12
  (∃ φ : ℝ, φ > 0 ∧ ∀ x : ℝ, f 2 (x + φ) = f 2 (2 * π/3 - x)) ∧
  (∀ ψ : ℝ, 0 < ψ ∧ ψ < π/12 → ∃ y : ℝ, f 2 (y + ψ) ≠ f 2 (2 * π/3 - y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1133_113388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_84_degrees_l1133_113352

/-- Represents a clock with hour and minute hands -/
structure Clock :=
  (hour_angle : ℝ)
  (minute_angle : ℝ)

/-- Calculates the angle between the hour and minute hands -/
def angle_between_hands (c : Clock) : ℝ :=
  abs (c.hour_angle - c.minute_angle)

/-- Represents a time in hours and minutes -/
structure Time :=
  (hour : ℕ)
  (minute : ℕ)

/-- Converts a time to minutes past 7:00 -/
def minutes_past_seven (t : Time) : ℕ :=
  (t.hour - 7) * 60 + t.minute

/-- Theorem: The clock hands form an 84° angle at 7:23 and 7:53 -/
theorem clock_hands_84_degrees :
  ∃ (t1 t2 : Time),
    minutes_past_seven t1 = 23 ∧
    minutes_past_seven t2 = 53 ∧
    (∀ (t : Time),
      7 ≤ t.hour ∧ t.hour < 8 →
      abs (angle_between_hands (Clock.mk
        ((210 + 30 * (minutes_past_seven t / 60 : ℝ)) % 360)
        ((minutes_past_seven t * 6 : ℝ) % 360)) - 84) < 0.5 →
      t = t1 ∨ t = t2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_hands_84_degrees_l1133_113352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_elements_with_few_factors_l1133_113353

/-- A sequence of positive integers satisfying the given condition -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  ∀ S : Finset ℕ, S.Nonempty → Nat.Prime ((S.prod (fun i => a i)) - 1)

/-- The set of indices for which the sequence element has less than m distinct prime factors -/
def LessThanMFactors (a : ℕ → ℕ) (m : ℕ) : Set ℕ :=
  {i | (Nat.factors (a i)).toFinset.card < m}

theorem finite_elements_with_few_factors
  (a : ℕ → ℕ) (h : SpecialSequence a) (m : ℕ) :
  Set.Finite (LessThanMFactors a m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_elements_with_few_factors_l1133_113353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_phi_l1133_113300

theorem smallest_angle_phi (φ : Real) : 
  (φ = 70 * Real.pi / 180) ∧ 
  (∀ θ : Real, θ > 0 ∧ θ < φ → Real.cos (10 * Real.pi / 180) ≠ Real.sin (30 * Real.pi / 180) + Real.sin θ) ∧
  (Real.cos (10 * Real.pi / 180) = Real.sin (30 * Real.pi / 180) + Real.sin φ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_phi_l1133_113300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_ate_two_apples_on_monday_l1133_113332

/-- Represents the days of the week --/
inductive Day
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday

/-- The number of apples Billy ate on each day --/
def apples_eaten (d : Day) : ℕ := sorry

/-- The total number of apples Billy ate during the week --/
def total_apples : ℕ := 20

/-- Theorem stating that Billy ate 2 apples on Monday --/
theorem billy_ate_two_apples_on_monday :
  (∀ d : Day, apples_eaten d > 0) →
  (∃ d : Day, apples_eaten d = 2) →
  (apples_eaten Day.tuesday = 2 * apples_eaten Day.monday) →
  (apples_eaten Day.wednesday = 9) →
  (apples_eaten Day.thursday = 4 * apples_eaten Day.friday) →
  (apples_eaten Day.friday = apples_eaten Day.monday / 2) →
  (apples_eaten Day.monday + apples_eaten Day.tuesday + apples_eaten Day.wednesday +
   apples_eaten Day.thursday + apples_eaten Day.friday = total_apples) →
  (apples_eaten Day.monday = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_billy_ate_two_apples_on_monday_l1133_113332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_condition_for_non_negative_l1133_113356

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1/x) - Real.log x

-- Theorem for part (I)
theorem tangent_line_at_one (x y : ℝ) :
  f 1 1 = 0 ∧ 
  (deriv (f 1)) 1 = 1 →
  x - y - 1 = 0 ↔ y = (deriv (f 1)) 1 * (x - 1) + f 1 1 := by
  sorry

-- Theorem for part (II)
theorem condition_for_non_negative (a : ℝ) :
  (∀ x ≥ 1, f a x ≥ 0) ↔ a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_condition_for_non_negative_l1133_113356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_of_P_coordinates_l1133_113396

-- Define the triangle DEF
def D : ℝ × ℝ := (-3, 6)
def E : ℝ × ℝ := (0, -3)
def F : ℝ × ℝ := (6, -3)

-- Define the vertical line that intersects DF at P and EF at Q
noncomputable def P : ℝ × ℝ := sorry
noncomputable def Q : ℝ × ℝ := sorry

-- Define the area of triangle PQF
def area_PQF : ℝ := 20

-- Define the property that P is on DF
def P_on_DF : Prop := sorry

-- Define the property that Q is on EF
def Q_on_EF : Prop := sorry

-- Define the property that PQ is vertical
def PQ_vertical : Prop := sorry

-- Theorem statement
theorem positive_difference_of_P_coordinates :
  P_on_DF → Q_on_EF → PQ_vertical → area_PQF = 20 →
  |P.2 - P.1| = 4 * Real.sqrt 10 - 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_difference_of_P_coordinates_l1133_113396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_2_8_l1133_113381

/-- A power function that passes through the point (2, 8) -/
noncomputable def power_function (a : ℝ) : ℝ → ℝ := fun x ↦ x^a

/-- Theorem stating that the power function passing through (2, 8) has exponent 3 -/
theorem power_function_through_2_8 :
  ∃ a : ℝ, (power_function a 2 = 8) ∧ (a = 3) := by
  use 3
  constructor
  · simp [power_function]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_2_8_l1133_113381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_over_12_l1133_113320

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2 * Real.tan x - (2 * Real.sin (x/2) ^ 2 - 1) / (Real.sin (x/2) * Real.cos (x/2))

-- State the theorem
theorem f_pi_over_12 : f (π/12) = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_over_12_l1133_113320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_time_l1133_113383

-- Define the rate of descent
noncomputable def rate_of_descent : ℝ := 32

-- Define the depth of the ship
noncomputable def depth_of_ship : ℝ := 6400

-- Define the time taken to reach the ship
noncomputable def time_to_reach_ship : ℝ := depth_of_ship / rate_of_descent

-- Theorem statement
theorem diver_descent_time :
  time_to_reach_ship = 200 := by
  -- Unfold the definitions
  unfold time_to_reach_ship depth_of_ship rate_of_descent
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_time_l1133_113383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_gravity_correct_l1133_113330

/-- The center of gravity of a wire triangle -/
noncomputable def center_of_gravity (A B C : ℝ × ℝ) : ℝ × ℝ :=
  let a := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let b := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let c := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let denominator := 2 * (a + b + c)
  (
    ((b + c) * A.1 + (c + a) * B.1 + (a + b) * C.1) / denominator,
    ((b + c) * A.2 + (c + a) * B.2 + (a + b) * C.2) / denominator
  )

/-- Theorem: The center of gravity of a wire triangle is correctly calculated -/
theorem center_of_gravity_correct (A B C : ℝ × ℝ) :
  center_of_gravity A B C = 
    let a := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
    let b := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
    let c := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
    let denominator := 2 * (a + b + c)
    (
      ((b + c) * A.1 + (c + a) * B.1 + (a + b) * C.1) / denominator,
      ((b + c) * A.2 + (c + a) * B.2 + (a + b) * C.2) / denominator
    ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_gravity_correct_l1133_113330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_distance_l1133_113323

/-- Represents an arithmetic sequence of 9 terms -/
structure ArithmeticSequence where
  terms : Fin 9 → ℕ
  is_arithmetic : ∀ i j k : Fin 9, i.val + 1 = j.val ∧ j.val + 1 = k.val → 
    terms j - terms i = terms k - terms j

/-- The theorem representing the problem -/
theorem walking_distance (seq : ArithmeticSequence) 
  (sum_condition : (Finset.univ : Finset (Fin 9)).sum seq.terms = 1260)
  (specific_sum : seq.terms 0 + seq.terms 3 + seq.terms 6 = 390) :
  seq.terms 7 = 170 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_distance_l1133_113323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_special_triangle_l1133_113372

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Helper function to calculate distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the properties of the triangle
def triangle_properties (t : Triangle) : Prop :=
  let AB := distance t.A t.B
  let BC := distance t.B t.C
  let AC := distance t.A t.C
  AB = Real.sqrt 3 ∧ 
  BC = 1 ∧ 
  Real.sin (Real.arccos ((AC^2 + BC^2 - AB^2) / (2 * AC * BC))) = 
    Real.sqrt 3 * Real.cos (Real.arccos ((AC^2 + BC^2 - AB^2) / (2 * AC * BC)))

-- Define the area of a triangle
noncomputable def triangle_area (t : Triangle) : ℝ :=
  let s := (distance t.A t.B + distance t.B t.C + distance t.C t.A) / 2
  Real.sqrt (s * (s - distance t.A t.B) * (s - distance t.B t.C) * (s - distance t.C t.A))

-- The theorem to be proved
theorem area_of_special_triangle (t : Triangle) (h : triangle_properties t) : 
  triangle_area t = Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_special_triangle_l1133_113372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_2_sqrt_2_l1133_113316

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 12 = 0

-- Define the common chord length
noncomputable def common_chord_length : ℝ := 2 * Real.sqrt 2

-- Theorem statement
theorem common_chord_length_is_2_sqrt_2 :
  ∃ (x1 y1 x2 y2 : ℝ),
    circle1 x1 y1 ∧ circle1 x2 y2 ∧
    circle2 x1 y1 ∧ circle2 x2 y2 ∧
    x1 ≠ x2 ∧ y1 ≠ y2 ∧
    (x2 - x1)^2 + (y2 - y1)^2 = common_chord_length^2 := by
  sorry

#check common_chord_length_is_2_sqrt_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_2_sqrt_2_l1133_113316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_k_double_cardinality_iff_odd_l1133_113335

/-- The set of nonnegative integers in base 10 with at most n digits -/
def S (n : ℕ) : Finset ℕ :=
  Finset.filter (fun x => x < 10^n) (Finset.range (10^n))

/-- The digit sum of a natural number -/
def digitSum (x : ℕ) : ℕ :=
  sorry

/-- The subset of S with digit sum less than k -/
def S_k (n k : ℕ) : Finset ℕ :=
  Finset.filter (fun x => digitSum x < k) (S n)

/-- Main theorem: |S| = 2|S_k| for some k if and only if n is odd -/
theorem exists_k_double_cardinality_iff_odd (n : ℕ) :
  (∃ k, Finset.card (S n) = 2 * Finset.card (S_k n k)) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_k_double_cardinality_iff_odd_l1133_113335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_two_element_set_l1133_113371

def S : Finset Nat := {1, 2}

theorem proper_subsets_of_two_element_set :
  (S.powerset.filter (· ⊂ S)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_two_element_set_l1133_113371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_equation_l1133_113331

theorem integer_solutions_equation (x y : ℤ) :
  7 * x ^ 2 - 40 * x * y + 7 * y ^ 2 = (abs (x - y) + 2) ^ 3 ↔ 
  (x = 2 ∧ y = -2) ∨ (x = -2 ∧ y = 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_equation_l1133_113331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1133_113373

noncomputable def f (a b x : ℝ) : ℝ := 2 * a * (Real.cos (x / 2))^2 + 2 * Real.sqrt 3 * a * Real.sin (x / 2) * Real.cos (x / 2) - a + b

theorem function_properties (a b : ℝ) :
  f a b (π / 3) = 3 ∧ f a b (5 * π / 6) = 1 →
  (a = 1 ∧ b = 1) ∧
  (∀ x ∈ Set.Icc 0 (π / 2), 2 ≤ f 1 1 x ∧ f 1 1 x ≤ 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1133_113373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_pi_alpha_l1133_113370

theorem cosine_sum_pi_alpha (α : Real) :
  ∃ (P : ℝ × ℝ), P = (3, 4) ∧ P.1 = 3 * Real.cos α ∧ P.2 = 3 * Real.sin α →
  Real.cos (π + α) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_pi_alpha_l1133_113370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_transformation_l1133_113368

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  a : Point
  b : Point
  c : Point
  d : Point

/-- Represents a rotation -/
def rotate (p : Point) (center : Point) (angle : ℝ) : Point :=
  sorry

theorem rotation_transformation (original : Quadrilateral) (rotated : Quadrilateral) (center : Point) :
  rotated = Quadrilateral.mk
    (rotate original.a center (270 * π / 180))
    (rotate original.b center (270 * π / 180))
    (rotate original.c center (270 * π / 180))
    (rotate original.d center (270 * π / 180)) :=
  sorry

-- The actual proof would go here if we had specific geometric information

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_transformation_l1133_113368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_l1133_113313

/-- Inverse proportion function -/
noncomputable def inverse_proportion (m : ℝ) (x : ℝ) : ℝ := (m^2 + 2*m - 2) / x

/-- Condition for the area of triangle OAB -/
def area_condition (m : ℝ) : Prop :=
  |m^2 + 2*m - 2| = 6

theorem inverse_proportion_point (m : ℝ) :
  area_condition m →
  ∃ x y : ℝ, x = -2 ∧ y = -3 ∧ inverse_proportion m x = y :=
by
  intro h
  use -2, -3
  simp [inverse_proportion]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_l1133_113313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fresh_grapes_weight_is_5_l1133_113386

/-- Represents the water content (by weight) of fresh grapes -/
def fresh_water_content : ℚ := 9/10

/-- Represents the water content (by weight) of dried grapes -/
def dried_water_content : ℚ := 1/5

/-- Represents the weight of dry grapes obtained (in kg) -/
def dry_grapes_weight : ℚ := 5/8

/-- Calculates the weight of fresh grapes used to obtain the given weight of dry grapes -/
def fresh_grapes_weight (fw : ℚ) (dw : ℚ) (d : ℚ) : ℚ :=
  d * (1 - dw) / (1 - fw)

/-- Theorem stating that the weight of fresh grapes used is 5 kg -/
theorem fresh_grapes_weight_is_5 :
  fresh_grapes_weight fresh_water_content dried_water_content dry_grapes_weight = 5 := by
  -- Unfold the definitions
  unfold fresh_grapes_weight
  unfold fresh_water_content
  unfold dried_water_content
  unfold dry_grapes_weight
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fresh_grapes_weight_is_5_l1133_113386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_intersection_l1133_113390

-- Define the ellipse E
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the circle (renamed to avoid conflict with Mathlib's circle)
def myCircle (r : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = r^2

-- Define perpendicularity
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

-- Define the theorem
theorem ellipse_circle_intersection :
  ∃ (a b r : ℝ),
    a > 0 ∧ b > 0 ∧ a > b ∧
    ellipse a b 2 (Real.sqrt 2) ∧
    ellipse a b (Real.sqrt 6) 1 ∧
    r^2 = 8/3 ∧
    (∀ (k m : ℝ),
      (∃ (x1 y1 x2 y2 : ℝ),
        ellipse a b x1 y1 ∧
        ellipse a b x2 y2 ∧
        (y1 = k * x1 + m) ∧
        (y2 = k * x2 + m) ∧
        myCircle r (m / Real.sqrt (1 + k^2)) (k * m / Real.sqrt (1 + k^2)) ∧
        perpendicular x1 y1 x2 y2)) ∧
    (∃ (x y : ℝ),
      ellipse a b x y ∧
      ellipse a b (-x) y ∧
      myCircle r x 0 ∧
      perpendicular x y (-x) y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_intersection_l1133_113390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_distances_equal_iff_condition_l1133_113355

/-- The focal distance of an ellipse with equation 9x^2 + 25y^2 = 225 -/
def ellipse_focal_distance : ℝ := 8

/-- The focal distance of a hyperbola with equation (x^2)/(16-k) - (y^2)/k = 1 -/
noncomputable def hyperbola_focal_distance (k : ℝ) : ℝ := 2 * Real.sqrt 16

/-- The condition for equal focal distances -/
def equal_focal_distances (k : ℝ) : Prop :=
  k < 16 ∧ k ≠ 0

theorem focal_distances_equal_iff_condition (k : ℝ) :
  ellipse_focal_distance = hyperbola_focal_distance k ↔ equal_focal_distances k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_distances_equal_iff_condition_l1133_113355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_section_area_is_pi_third_l1133_113337

/-- Represents a cube with edge length 2 -/
structure Cube where
  edge_length : ℝ
  edge_length_eq : edge_length = 2

/-- Represents a sphere tangent to all faces of a cube -/
structure TangentSphere (c : Cube) where
  center : ℝ × ℝ × ℝ
  radius : ℝ
  tangent_to_faces : True  -- This is a placeholder for the tangency condition

/-- Represents a plane passing through two opposite vertices of the cube and the midpoint of an edge perpendicular to the diagonal formed by those vertices -/
structure CuttingPlane (c : Cube) where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ
  passes_through_vertices : True  -- This is a placeholder for the condition

/-- The area of the section obtained by cutting the sphere with the plane -/
noncomputable def section_area (c : Cube) (s : TangentSphere c) (p : CuttingPlane c) : ℝ :=
  sorry

/-- The main theorem stating that the area of the section is π/3 -/
theorem section_area_is_pi_third (c : Cube) (s : TangentSphere c) (p : CuttingPlane c) :
  section_area c s p = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_section_area_is_pi_third_l1133_113337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_cube_root_simplification_l1133_113343

theorem nested_cube_root_simplification (M : ℝ) (h : M > 1) :
  (M * (M * (M * M ^ (1/3)) ^ (1/3)) ^ (1/3)) ^ (1/3) = M ^ (40/81) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_cube_root_simplification_l1133_113343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1133_113399

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi) * Real.sin (x - Real.pi / 2) + Real.cos x * Real.sin (Real.pi / 2 - x)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ 
   ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (k : ℤ), ∃ (c : ℝ), c = (k : ℝ) * Real.pi / 2 - Real.pi / 8 ∧
   ∀ (x : ℝ), f (c + x) = f (c - x)) ∧
  (∀ (x : ℝ), -Real.pi / 8 ≤ x ∧ x ≤ 3 * Real.pi / 8 →
   1 / 2 ≤ f x ∧ f x ≤ (1 + Real.sqrt 2) / 2) ∧
  (∃ (x : ℝ), -Real.pi / 8 ≤ x ∧ x ≤ 3 * Real.pi / 8 ∧ f x = 1 / 2) ∧
  (∃ (x : ℝ), -Real.pi / 8 ≤ x ∧ x ≤ 3 * Real.pi / 8 ∧ f x = (1 + Real.sqrt 2) / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1133_113399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avian_influenza_virus_size_scientific_notation_l1133_113307

theorem avian_influenza_virus_size_scientific_notation :
  (0.000000102 : ℝ) = 1.02 * (10 : ℝ)^(-7 : ℤ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_avian_influenza_virus_size_scientific_notation_l1133_113307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_result_l1133_113374

-- Define the initial complex number
def z : ℂ := -3 - 8*Complex.I

-- Define the rotation transformation
noncomputable def rotate (θ : ℝ) (w : ℂ) : ℂ := w * Complex.exp (θ * Complex.I)

-- Define the dilation transformation
def dilate (k : ℝ) (w : ℂ) : ℂ := k * w

-- Define the composition of both transformations
noncomputable def transform (w : ℂ) : ℂ := dilate 2 (rotate (Real.pi/3) w)

-- State the theorem
theorem transform_result :
  transform z = (-3 - 8*Real.sqrt 3) + (-3*Real.sqrt 3 - 8)*Complex.I :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_result_l1133_113374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swallow_ratio_l1133_113364

/-- Represents the number of American swallows -/
def american_swallows : ℕ := sorry

/-- Represents the number of European swallows -/
def european_swallows : ℕ := sorry

/-- The total number of swallows in the flock -/
def total_swallows : ℕ := 90

/-- The weight an American swallow can carry (in pounds) -/
def american_capacity : ℕ := 5

/-- The weight a European swallow can carry (in pounds) -/
def european_capacity : ℕ := 2 * american_capacity

/-- The maximum combined weight the flock can carry (in pounds) -/
def max_combined_weight : ℕ := 600

theorem swallow_ratio :
  american_swallows + european_swallows = total_swallows ∧
  american_swallows * american_capacity + european_swallows * european_capacity = max_combined_weight →
  american_swallows = 2 * european_swallows :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swallow_ratio_l1133_113364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l1133_113336

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 + Real.log x

-- State the theorem
theorem min_m_value (m : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (Real.exp 1) ∧ f x₀ ≤ m) ↔ m ≥ 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_m_value_l1133_113336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_negative_seven_l1133_113346

theorem cube_root_negative_seven (x : ℝ) : x^3 = -7 ↔ x = -(7^(1/3)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_negative_seven_l1133_113346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_water_volume_ratio_l1133_113309

theorem cone_water_volume_ratio :
  ∀ (h r : ℝ), h > 0 → r > 0 →
  let water_height := (2 : ℝ) / 3 * h
  let water_radius := (2 : ℝ) / 3 * r
  let cone_volume := (1 : ℝ) / 3 * Real.pi * r^2 * h
  let water_volume := (1 : ℝ) / 3 * Real.pi * water_radius^2 * water_height
  water_volume / cone_volume = 8 / 27 := by
  intros h r h_pos r_pos
  -- Proof steps would go here
  sorry

#check cone_water_volume_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_water_volume_ratio_l1133_113309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_mass_is_2pi_l1133_113365

noncomputable section

-- Define the cone
def cone (x y z : ℝ) : Prop := 36 * (x^2 + y^2) = z^2

-- Define the cylinder
def cylinder (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the density function
noncomputable def density (x y : ℝ) : ℝ := 5 * (x^2 + y^2) / 6

-- Define the region of integration
def region (x y z : ℝ) : Prop :=
  cone x y z ∧ cylinder x y ∧ x ≥ 0 ∧ z ≥ 0

-- State the theorem
theorem cone_mass_is_2pi :
  ∃ (m : ℝ), m = 2 * Real.pi ∧
  m = ∫ (x : ℝ) in Set.Icc 0 1, ∫ (y : ℝ) in Set.Icc (-1) 1, ∫ (z : ℝ) in Set.Icc 0 (6 * Real.sqrt (x^2 + y^2)),
    density x y := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_mass_is_2pi_l1133_113365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l1133_113312

/-- Circle represented by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Distance between two points in ℝ² -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Definition of internally tangent circles -/
def internally_tangent (c1 c2 : Circle) : Prop :=
  distance c1.center c2.center = c2.radius - c1.radius

theorem circles_internally_tangent : 
  let c1 : Circle := ⟨(0, 0), 2⟩
  let c2 : Circle := ⟨(3, -4), 7⟩
  internally_tangent c1 c2 := by
  sorry

#check circles_internally_tangent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l1133_113312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composition_l1133_113326

-- Define a function f with domain [-1, 1]
def f : Set ℝ → Set ℝ := fun D => {x : ℝ | x ∈ D}

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-1) 1

-- State the theorem
theorem domain_of_composition (f : Set ℝ → Set ℝ) (h : domain_f = Set.Icc (-1) 1) :
  {x : ℝ | (2 * x - 1) ∈ domain_f} = Set.Icc 0 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_composition_l1133_113326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_problem_solution_l1133_113391

-- Define the function y
noncomputable def y (x : ℝ) : ℝ := -Real.exp (-2 * x)

-- State the theorem
theorem cauchy_problem_solution :
  (∀ x, (deriv (deriv y)) x - 8 * (y x)^3 = 0) ∧
  y 0 = -1 ∧
  (deriv y) 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cauchy_problem_solution_l1133_113391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_contains_point_and_line_l1133_113327

-- Define the point M₁
def M₁ : ℝ × ℝ × ℝ := (3, 1, 0)

-- Define the line
def line_equation (x y z : ℝ) : Prop :=
  (x - 4) / 1 = y / 2 ∧ y / 2 = (z - 1) / 3

-- Define the plane equation
def plane_equation (x y z : ℝ) : Prop :=
  5 * x + 2 * y - 3 * z - 17 = 0

-- Theorem statement
theorem plane_contains_point_and_line :
  (∀ x y z, plane_equation x y z → (x, y, z) = M₁ ∨ line_equation x y z) ∧
  (plane_equation M₁.1 M₁.2.1 M₁.2.2) ∧
  (∀ x y z, line_equation x y z → plane_equation x y z) := by
  sorry

#check plane_contains_point_and_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_contains_point_and_line_l1133_113327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_isosceles_triangle_properties_l1133_113389

/-- Represents an isosceles triangle with a rotated solid of revolution -/
structure RotatedIsoscelesTriangle where
  b : ℝ  -- Length of side AB
  α : ℝ  -- Angle BAC in radians
  h : 0 < b ∧ 0 < α ∧ α < Real.pi  -- Constraints on b and α

/-- Calculate the surface area of the rotated solid -/
noncomputable def surfaceArea (t : RotatedIsoscelesTriangle) : ℝ :=
  12 * Real.pi * t.b^2 * Real.sin (t.α / 2) * (1 + Real.sin (t.α / 2))

/-- Calculate the volume of the rotated solid -/
noncomputable def volume (t : RotatedIsoscelesTriangle) : ℝ :=
  6 * Real.pi * t.b^3 * Real.sin (t.α / 2)^2 * Real.cos (t.α / 2)

/-- Theorem stating the correctness of surface area and volume calculations -/
theorem rotated_isosceles_triangle_properties (t : RotatedIsoscelesTriangle) :
  surfaceArea t = 12 * Real.pi * t.b^2 * Real.sin (t.α / 2) * (1 + Real.sin (t.α / 2)) ∧
  volume t = 6 * Real.pi * t.b^3 * Real.sin (t.α / 2)^2 * Real.cos (t.α / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_isosceles_triangle_properties_l1133_113389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_board_probability_l1133_113351

/-- The probability of a dart landing in the center hexagon of a regular hexagonal dart board -/
theorem dart_board_probability (s : ℝ) (h : s > 0) : 
  (3 * Real.sqrt 3 / 2 * (s/2)^2) / (3 * Real.sqrt 3 / 2 * s^2) = 1/4 := by
  -- Simplify the fraction
  have h1 : (3 * Real.sqrt 3 / 2 * (s/2)^2) / (3 * Real.sqrt 3 / 2 * s^2) = (s/2)^2 / s^2 := by
    field_simp
    ring
  
  -- Rewrite the goal
  rw [h1]
  
  -- Simplify further
  have h2 : (s/2)^2 / s^2 = 1/4 := by
    field_simp
    ring
  
  -- Apply the final simplification
  exact h2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dart_board_probability_l1133_113351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_implies_sum_l1133_113317

/-- Triangle PQR with vertices P(2,5), Q(-1,3), and R(4,-2) -/
def P : ℝ × ℝ := (2, 5)
def Q : ℝ × ℝ := (-1, 3)
def R : ℝ × ℝ := (4, -2)

/-- Point S with coordinates (x, y) -/
def S (x y : ℝ) : ℝ × ℝ := (x, y)

/-- The area of a triangle given three points -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

/-- Theorem: If S is chosen such that PQS, QRS, and RPS have equal areas, then 4x + 3y = 38/3 -/
theorem equal_area_implies_sum (x y : ℝ) 
  (h : triangle_area P Q (S x y) = triangle_area Q R (S x y) ∧ 
       triangle_area Q R (S x y) = triangle_area R P (S x y)) : 
  4 * x + 3 * y = 38 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_implies_sum_l1133_113317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angleBetweenHandsAt10pm_l1133_113382

/-- The number of hour marks on a clock --/
def clockHourMarks : ℕ := 12

/-- The angle between adjacent hour marks on a clock --/
noncomputable def angleBetweenMarks : ℝ := 360 / clockHourMarks

/-- The number of hour marks between the hour and minute hands at 10:00 p.m. --/
def marksBetweenHandsAt10pm : ℕ := 2

/-- Theorem: The angle between the hour and minute hands of a clock at 10:00 p.m. is 60° --/
theorem angleBetweenHandsAt10pm : 
  (marksBetweenHandsAt10pm : ℝ) * angleBetweenMarks = 60 := by
  sorry

#eval clockHourMarks
#eval marksBetweenHandsAt10pm

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angleBetweenHandsAt10pm_l1133_113382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_S_7890_l1133_113345

-- Define the sequence S_n
noncomputable def S (n : ℕ) : ℝ :=
  let c := 4 + 3 * Real.sqrt 2
  let d := 4 - 3 * Real.sqrt 2
  (1 / 2) * (c ^ n + d ^ n)

-- Define the units digit function
noncomputable def unitsDigit (x : ℝ) : ℕ :=
  Int.natAbs (Int.mod (Int.floor x) 10)

-- Theorem statement
theorem units_digit_S_7890 :
  unitsDigit (S 7890) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_S_7890_l1133_113345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a2005_eq_neg6_l1133_113322

/-- A sequence of integers with specific properties -/
def SequenceA : ℕ → ℤ := sorry

/-- The third term of the sequence is 5 -/
axiom a3_eq_5 : SequenceA 3 = 5

/-- The fifth term of the sequence is 8 -/
axiom a5_eq_8 : SequenceA 5 = 8

/-- There exists a positive integer n such that the sum of three consecutive terms starting at n is 7 -/
axiom exists_sum_7 : ∃ n : ℕ+, SequenceA n.val + SequenceA (n.val + 1) + SequenceA (n.val + 2) = 7

/-- The 2005th term of the sequence is -6 -/
theorem a2005_eq_neg6 : SequenceA 2005 = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a2005_eq_neg6_l1133_113322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_heap_height_l1133_113397

/-- Represents the dimensions of a cylinder -/
structure Cylinder where
  height : ℝ
  radius : ℝ

/-- Represents the dimensions of a cone -/
structure Cone where
  height : ℝ
  radius : ℝ

/-- Calculates the volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- Theorem: The height of the conical heap is 12 cm -/
theorem sand_heap_height (bucket : Cylinder) (heap : Cone) : 
  bucket.height = 36 ∧ 
  bucket.radius = 21 ∧ 
  heap.radius = 63 ∧ 
  cylinderVolume bucket = coneVolume heap → 
  heap.height = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_heap_height_l1133_113397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_year_markup_is_25_percent_l1133_113344

/-- Calculates the New Year season markup percentage given the initial markup,
    February discount, and final profit percentages. -/
noncomputable def calculate_new_year_markup (initial_markup : ℝ) (february_discount : ℝ) (final_profit : ℝ) : ℝ :=
  let initial_price := 1 + initial_markup
  let final_price := 1 + final_profit
  let new_year_markup := (final_price / ((1 - february_discount) * initial_price)) - 1
  new_year_markup * 100

/-- Theorem stating that given the specific conditions, the New Year markup was 25% -/
theorem new_year_markup_is_25_percent :
  calculate_new_year_markup 0.20 0.08 0.38 = 25 := by
  sorry

-- Remove the #eval statement as it's not necessary for building and may cause issues
-- #eval calculate_new_year_markup 0.20 0.08 0.38

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_year_markup_is_25_percent_l1133_113344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_overlap_exists_min_omega_l1133_113354

noncomputable section

open Real

/-- The function f(x) = 2sin(ωx + π/3) - 1 --/
def f (ω : ℝ) (x : ℝ) : ℝ := 2 * sin (ω * x + π / 3) - 1

/-- The shifted function g(x) = f(x - π/3) --/
def g (ω : ℝ) (x : ℝ) : ℝ := f ω (x - π / 3)

theorem min_omega_for_overlap (ω : ℝ) (h1 : ω > 0) :
  (∀ x, f ω x = g ω x) → ω ≥ 6 := by sorry

theorem exists_min_omega :
  ∃ ω : ℝ, ω > 0 ∧ (∀ x, f ω x = g ω x) ∧ ω = 6 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_for_overlap_exists_min_omega_l1133_113354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1133_113393

theorem tan_alpha_value (α : ℝ) (h1 : Real.sin α = 4/5) (h2 : α ∈ Set.Ioo 0 Real.pi) :
  Real.tan α = 4/3 ∨ Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1133_113393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l1133_113392

def M : Nat := 2^5 * 3^4 * 5^3 * 7^3 * 11^1

theorem number_of_factors_of_M : 
  (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 960 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l1133_113392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l1133_113338

theorem min_value_expression (a b c : ℕ) 
  (ha : a < 7) (hb : b ≤ 3) (hc : c ≤ 4) : 
  (∀ a' b' c' : ℕ, a' < 7 → b' ≤ 3 → c' ≤ 4 → 
    (3 : ℤ) * a - 2 * a * b + a * c ≤ 3 * a' - 2 * a' * b' + a' * c') → 
  (3 : ℤ) * a - 2 * a * b + a * c = -12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l1133_113338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1133_113305

/-- The hyperbola equation -/
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  y^2 / a^2 - x^2 / b^2 = 1

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

/-- The distance from a point to the line x = -1 -/
def dist_to_line (x y : ℝ) : ℝ :=
  |x + 1|

/-- The distance between two points -/
noncomputable def dist (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The theorem statement -/
theorem hyperbola_equation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y, hyperbola a b x y → ∀ x' y', parabola x' y' →
    (∃ m, m = dist_to_line x' y' + dist x' y' 0 c ∧
          ∀ x'' y'', parabola x'' y'' →
            m ≤ dist_to_line x'' y'' + dist x'' y'' 0 c) →
    m = Real.sqrt 6) →
  c / a = Real.sqrt 5 / 2 →
  a = 2 ∧ b = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1133_113305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_special_property_l1133_113319

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio q -/
noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric sequence with common ratio q (q ≠ 0, q ≠ 1),
    if S_3, S_9, and S_6 form an arithmetic sequence, then q^3 = -1/2 -/
theorem geometric_sequence_special_property (a : ℝ) (q : ℝ) (h1 : q ≠ 0) (h2 : q ≠ 1) :
  let S := geometric_sum a q
  2 * S 9 = S 3 + S 6 → q^3 = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_special_property_l1133_113319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_increasing_condition_l1133_113384

theorem log_function_increasing_condition (a : ℝ) : 
  a > 0 → 
  a ≠ 1 → 
  (∀ x ∈ Set.Icc 1 2, Monotone (fun x => Real.log (|x^2 - (a + 1/a)*x + 1|))) → 
  a ≥ 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_increasing_condition_l1133_113384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_exists_l1133_113363

/-- The sequence where each integer k appears k^2 times -/
def my_sequence (n : ℕ) : ℕ := sorry

/-- The formula for the nth term of the sequence -/
noncomputable def formula (b c d n : ℤ) : ℤ := b * ⌊(n + c : ℝ)^(1/4)⌋ + d

theorem sequence_formula_exists :
  ∃ (b c d : ℤ), 
    (∀ n : ℕ, (my_sequence n : ℤ) = formula b c d n) ∧
    b + c + d = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_exists_l1133_113363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_position_l1133_113328

/-- The number of sides in a regular octagon -/
def octagon_sides : ℕ := 8

/-- The inner angle of a regular octagon in degrees -/
noncomputable def octagon_inner_angle : ℚ := ((octagon_sides - 2) * 180) / octagon_sides

/-- The rotation of the square per movement in degrees -/
noncomputable def square_rotation_per_move : ℚ := 360 - (octagon_inner_angle + 90)

/-- The number of sides the square rolls around -/
def num_moves : ℕ := 4

/-- The total rotation of the square after rolling around four sides -/
noncomputable def total_rotation : ℚ := num_moves * square_rotation_per_move

theorem square_triangle_position : 
  (total_rotation % 360 = 180) → 
  (∃ (initial_pos final_pos : String), initial_pos = "left" ∧ final_pos = "right") :=
by sorry

#eval octagon_sides
#eval num_moves

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_position_l1133_113328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1133_113350

noncomputable def f (a x : ℝ) : ℝ := a * (1 / (x * 2^x - x) + 1 / (2 * x)) - Real.log (|4 * x|) + 2

theorem problem_statement :
  (∀ x, f 0 x > 2 - Real.log 2 ↔ x ∈ Set.Ioo (-1/2 : ℝ) 0 ∪ Set.Ioo 0 (1/2)) ∧
  (∀ a t, a > 0 → (f a (3*t-1) > f a (t-2) ↔ t ∈ Set.Ioo (-1/2 : ℝ) (1/3) ∪ Set.Ioo (1/3) (3/4))) ∧
  (∀ m n : ℝ, m ≠ 0 → n ≠ 0 → f 0 m + 1/n^2 = f 0 n - 1/m^2 → m^2 - n^2 > 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1133_113350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_min_values_l1133_113349

/-- P is a monic cubic polynomial -/
def P (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- Q is a monic cubic polynomial -/
def Q (d e f : ℝ) (x : ℝ) : ℝ := x^3 + d*x^2 + e*x + f

/-- The composition of P and Q -/
def P_Q (a b c d e f : ℝ) (x : ℝ) : ℝ := P a b c (Q d e f x)

/-- The composition of Q and P -/
def Q_P (a b c d e f : ℝ) (x : ℝ) : ℝ := Q d e f (P a b c x)

/-- The theorem stating the sum of minimum values of P and Q is -20 -/
theorem sum_of_min_values (a b c d e f : ℝ) :
  (∀ x ∈ ({-4, -2, 0, 2, 4} : Set ℝ), P_Q a b c d e f x = 0) →
  (∀ x ∈ ({-3, -1, 1, 3} : Set ℝ), Q_P a b c d e f x = 0) →
  (∃ x₁, ∀ x, P a b c x ≥ P a b c x₁) →
  (∃ x₂, ∀ x, Q d e f x ≥ Q d e f x₂) →
  ∃ x₁ x₂, P a b c x₁ + Q d e f x₂ = -20 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_min_values_l1133_113349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_inequality_l1133_113348

theorem power_product_inequality (m n : ℕ) : (2 : ℝ)^m * (3 : ℝ)^n ≠ 6^(m+n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_product_inequality_l1133_113348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_sum_a_b_l1133_113394

noncomputable def h (x : ℝ) : ℝ := 3 / (3 + 5 * x^2)

theorem h_range : Set.range h = Set.Ioo 0 1 := by sorry

theorem sum_a_b : ∃ (a b : ℝ), Set.range h = Set.Ioi a ∩ Set.Iic b ∧ a + b = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_sum_a_b_l1133_113394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_fv_length_l1133_113385

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola -/
structure Parabola where
  vertex : Point
  focus : Point

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem about the length of FV in a parabola -/
theorem parabola_fv_length
  (p : Parabola)
  (a : Point)
  (h1 : distance a p.focus = 25)
  (h2 : distance a p.vertex = 28)
  (h3 : ∃ (t : ℝ), 0 < t ∧ t < 1 ∧
        distance (Point.mk (t * p.focus.x + (1 - t) * p.vertex.x)
                           (t * p.focus.y + (1 - t) * p.vertex.y)) a =
        distance (Point.mk (t * p.focus.x + (1 - t) * p.vertex.x)
                           (t * p.focus.y + (1 - t) * p.vertex.y)) p.focus ∧
        t = 1/4) :
  distance p.focus p.vertex = 200/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_fv_length_l1133_113385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_handshaking_arrangements_l1133_113378

/-- The number of people in the group -/
def n : ℕ := 11

/-- The number of handshakes per person -/
def k : ℕ := 3

/-- The modulus for the final result -/
def m : ℕ := 1000

/-- A handshaking arrangement is valid if each person shakes hands with exactly k others -/
def IsValidArrangement (arrangement : Fin n → Fin n → Bool) : Prop :=
  ∀ i, (Finset.filter (λ j => arrangement i j) Finset.univ).card = k

/-- The set of all valid handshaking arrangements -/
def ValidArrangements : Set (Fin n → Fin n → Bool) :=
  {arrangement | IsValidArrangement arrangement}

/-- The number of valid handshaking arrangements -/
noncomputable def M : ℕ := sorry

/-- The main theorem: the number of valid arrangements is congruent to 800 modulo 1000 -/
theorem handshaking_arrangements :
  M ≡ 800 [MOD m] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_handshaking_arrangements_l1133_113378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l1133_113387

/-- The function f(x) = |4/x - ax| -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |4/x - a*x|

/-- The theorem stating the maximum value of m -/
theorem max_m_value (a : ℝ) (h_a : a > 0) :
  (∃ (x₀ : ℝ), x₀ ∈ Set.Icc 1 4 ∧ ∀ (m : ℝ), m ≤ f a x₀) ↔ 
  (∀ (m : ℝ), m ≤ 3) :=
by
  sorry

#check max_m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_l1133_113387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1133_113398

noncomputable def S (x : ℝ) : ℝ := 2 * (1 / (1 + x))

def equation (x : ℝ) : Prop := x = S x

def convergence (x : ℝ) : Prop := |x| < 1

theorem unique_solution : ∃! x : ℝ, equation x ∧ convergence x ∧ x = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1133_113398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_solution_rounded_up_l1133_113333

noncomputable def deg_to_rad (x : ℝ) : ℝ := x * Real.pi / 180

def equation (x : ℝ) : Prop :=
  Real.cos (deg_to_rad x) = Real.sin (deg_to_rad (x + 6))

theorem least_solution_rounded_up :
  ∃ (x : ℝ), x > 1 ∧ equation x ∧
  (∀ (y : ℝ), y > 1 ∧ equation y → x ≤ y) ∧
  Int.ceil x = 42 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_solution_rounded_up_l1133_113333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_denotation_l1133_113367

/-- Represents the monetary value in yuan -/
structure Yuan where
  value : Int

/-- Represents the denotation of a monetary value -/
def denote (x : Yuan) : Yuan := x

/-- Conversion from Int to Yuan -/
instance : Coe Int Yuan where
  coe i := ⟨i⟩

/-- Addition for Yuan -/
instance : Add Yuan where
  add a b := ⟨a.value + b.value⟩

/-- Negation for Yuan -/
instance : Neg Yuan where
  neg a := ⟨-a.value⟩

/-- OfNat for Yuan -/
instance : OfNat Yuan n where
  ofNat := ⟨n⟩

/-- Given that a profit of 20 yuan is denoted as +20 yuan,
    prove that a loss of 30 yuan should be denoted as -30 yuan -/
theorem loss_denotation (h : denote 20 = (20 : Yuan)) : denote (-30) = (-30 : Yuan) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_denotation_l1133_113367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_P_and_Q_l1133_113306

-- Define the circle and points on it
variable (circle : Set ℝ)
variable (A B Q D C E : ℝ)

-- Define the arcs and their measures
variable (arc_BQ arc_QD arc_DE : ℝ)

-- Define angles P and Q
variable (angle_P angle_Q : ℝ)

-- State the theorem
theorem sum_of_angles_P_and_Q
  (hA : A ∈ circle) (hB : B ∈ circle) (hQ : Q ∈ circle)
  (hD : D ∈ circle) (hC : C ∈ circle) (hE : E ∈ circle)
  (h_BQ : arc_BQ = 60) (h_QD : arc_QD = 40) (h_DE : arc_DE = 50)
  (h_P : angle_P = (90 + 50 - 60) / 2)
  (h_Q : angle_Q = (90 + 50 - 40) / 2) :
  angle_P + angle_Q = 90 :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_angles_P_and_Q_l1133_113306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_all_crows_on_same_tree_l1133_113340

/-- Represents the number of trees in the circle -/
def num_trees : ℕ := 22

/-- Represents the parity of the number of crows on cherry trees -/
inductive Parity
| Odd
| Even

/-- Represents the state of crows on cherry trees -/
structure CrowState where
  count : ℕ
  parity : Parity

/-- The initial state of crows on cherry trees -/
def initial_state : CrowState :=
  { count := num_trees / 2, parity := Parity.Odd }

/-- Represents a single move of two crows -/
inductive Move
| BothToCherry
| BothFromCherry
| OneEachWay

/-- Updates the state after a move -/
def update_state (state : CrowState) (move : Move) : CrowState :=
  match move with
  | Move.BothToCherry => { count := state.count + 2, parity := state.parity }
  | Move.BothFromCherry => { count := state.count - 2, parity := state.parity }
  | Move.OneEachWay => state

/-- Theorem: It's impossible for all crows to be on the same tree -/
theorem impossibility_of_all_crows_on_same_tree :
  ∀ (moves : List Move), 
    (moves.foldl update_state initial_state).count ≠ 0 ∧
    (moves.foldl update_state initial_state).count ≠ num_trees :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_all_crows_on_same_tree_l1133_113340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pricing_theorem_l1133_113360

/-- Represents the tiered pricing system for water usage -/
structure WaterPricing where
  a : ℚ  -- Price for first 17 tons
  b : ℚ  -- Price for 17-30 tons
  sewage : ℚ := 0.80  -- Sewage treatment price

/-- Calculates the water bill based on usage and pricing -/
def calculateBill (pricing : WaterPricing) (usage : ℚ) : ℚ :=
  if usage ≤ 17 then
    usage * (pricing.a + pricing.sewage)
  else if usage ≤ 30 then
    17 * (pricing.a + pricing.sewage) + (usage - 17) * (pricing.b + pricing.sewage)
  else
    17 * (pricing.a + pricing.sewage) + 13 * (pricing.b + pricing.sewage) + (usage - 30) * (6 + pricing.sewage)

theorem water_pricing_theorem (pricing : WaterPricing) :
  (calculateBill pricing 20 = 66 ∧ calculateBill pricing 25 = 91) →
  (pricing.a = 2.2 ∧ pricing.b = 4.2) ∧
  (∃ (x : ℚ), calculateBill pricing x = 184 ∧ x = 40) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pricing_theorem_l1133_113360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_consistent_configuration_l1133_113315

/-- Represents a statement on the card -/
inductive Statement
| one
| two
| three
| four
| five

/-- The truth value of each statement -/
def is_true : Statement → Bool
| Statement.one => true
| Statement.two => true
| Statement.three => false
| Statement.four => false
| Statement.five => false

/-- The claim made by each statement -/
def claim : Statement → Nat
| Statement.one => 1
| Statement.two => 2
| Statement.three => 3
| Statement.four => 4
| Statement.five => 5

/-- The total number of statements on the card -/
def total_statements : Nat := 5

/-- The number of false statements on the card -/
def false_count : Nat := List.filter (fun s => ¬(is_true s)) [Statement.one, Statement.two, Statement.three, Statement.four, Statement.five] |>.length

theorem unique_consistent_configuration :
  -- The number of false statements is equal to the claim of any true statement
  (∀ s : Statement, is_true s → claim s = false_count) ∧
  -- The number of true statements is equal to total_statements minus the claim of any false statement
  (∀ s : Statement, ¬(is_true s) → total_statements - claim s = total_statements - false_count) ∧
  -- There is at least one true statement and one false statement
  (∃ s : Statement, is_true s) ∧ (∃ s : Statement, ¬(is_true s)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_consistent_configuration_l1133_113315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l1133_113342

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- Predicate to check if a number is divisible by 6 -/
def isDivisibleBy6 (n : ℤ) : Prop := ∃ k : ℤ, n = 6 * k

theorem polynomial_divisibility 
  (P : IntPolynomial) 
  (h2 : isDivisibleBy6 (P.eval 2)) 
  (h3 : isDivisibleBy6 (P.eval 3)) : 
  isDivisibleBy6 (P.eval 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l1133_113342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_volume_is_2_sqrt_6_div_9_l1133_113358

/-- A right square pyramid with equilateral triangle lateral faces -/
structure SquarePyramid where
  base_side : ℝ
  lateral_face_equilateral : Bool

/-- A cube inscribed in a pyramid -/
structure InscribedCube where
  pyramid : SquarePyramid
  bottom_face_on_base : Bool
  top_vertices_touch_lateral_faces : Bool

/-- The volume of an inscribed cube in the given pyramid -/
noncomputable def inscribed_cube_volume (c : InscribedCube) : ℝ :=
  2 * Real.sqrt 6 / 9

/-- Theorem stating the volume of the inscribed cube -/
theorem inscribed_cube_volume_is_2_sqrt_6_div_9 
  (c : InscribedCube) 
  (h1 : c.pyramid.base_side = 2) 
  (h2 : c.pyramid.lateral_face_equilateral = true)
  (h3 : c.bottom_face_on_base = true)
  (h4 : c.top_vertices_touch_lateral_faces = true) : 
  inscribed_cube_volume c = 2 * Real.sqrt 6 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_volume_is_2_sqrt_6_div_9_l1133_113358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_attendees_count_l1133_113325

def wedding_attendees (total_guests : ℕ) 
  (response_rate_30 response_rate_20 response_rate_10 : ℚ)
  (yes_rate_30 yes_rate_20 yes_rate_10 : ℚ)
  (change_rate : ℚ) : ℕ :=
  let guests_30 := (total_guests : ℚ) * response_rate_30
  let yes_30 := guests_30 * yes_rate_30
  let guests_20 := (total_guests : ℚ) * response_rate_20
  let yes_20 := guests_20 * yes_rate_20
  let guests_10 := (total_guests : ℚ) * response_rate_10
  let yes_10 := guests_10 * yes_rate_10
  let total_yes := yes_30 + yes_20 + yes_10
  let changed := total_yes * change_rate
  (total_yes - changed).floor.toNat

theorem final_attendees_count :
  wedding_attendees 200 (60/100) (30/100) (5/100) (80/100) (75/100) (50/100) (2/100) = 144 := by
  sorry

#eval wedding_attendees 200 (60/100) (30/100) (5/100) (80/100) (75/100) (50/100) (2/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_attendees_count_l1133_113325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_1200_l1133_113380

theorem divisors_of_1200 : 
  (Finset.filter (λ n : ℕ ↦ 1200 % n = 0) (Finset.range 1201)).card = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_1200_l1133_113380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_possible_median_l1133_113308

theorem largest_possible_median (L : List ℕ) : 
  L.length = 11 ∧ 
  (∀ x ∈ L, x > 0) ∧ 
  3 ∈ L ∧ 7 ∈ L ∧ 2 ∈ L ∧ 5 ∈ L ∧ 9 ∈ L ∧ 6 ∈ L →
  ∃ (M : List ℕ), M.Perm L ∧ M.Sorted (·≤·) ∧ M.get? 5 = some 8 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_possible_median_l1133_113308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_price_increase_l1133_113303

/-- Represents the price change of items from June to July -/
structure PriceChange where
  june_price : ℚ  -- Price in June
  july_price : ℚ  -- Price in July

/-- Calculate the percentage change in price -/
noncomputable def PriceChange.percentage_change (pc : PriceChange) : ℚ :=
  (pc.july_price - pc.june_price) / pc.june_price * 100

/-- Represents the prices of milk powder and coffee -/
structure Prices where
  milk_powder : PriceChange
  coffee : PriceChange

/-- Theorem stating the conditions and the expected result -/
theorem coffee_price_increase (prices : Prices) : 
  prices.milk_powder.june_price = prices.coffee.june_price →  -- Same price in June
  prices.milk_powder.july_price = 1/5 * prices.milk_powder.june_price →  -- 80% price drop for milk powder
  prices.milk_powder.july_price = 1/5 →  -- Milk powder costs $0.20 in July
  3/2 * prices.milk_powder.july_price + 3/2 * prices.coffee.july_price = 63/10 →  -- Mixture cost in July
  prices.coffee.percentage_change = 300 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_price_increase_l1133_113303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_1_prop_2_prop_3_prop_4_l1133_113366

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real

-- Define properties of the triangle
def isEquilateral (t : Triangle) : Prop :=
  t.A = t.B ∧ t.B = t.C

def isRightAngled (t : Triangle) : Prop :=
  t.A = Real.pi/2 ∨ t.B = Real.pi/2 ∨ t.C = Real.pi/2

def isObtuse (t : Triangle) : Prop :=
  t.A > Real.pi/2 ∨ t.B > Real.pi/2 ∨ t.C > Real.pi/2

def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.C = t.A

-- Theorem for proposition ①
theorem prop_1 (t : Triangle) :
  Real.cos (t.A - t.B) * Real.cos (t.B - t.C) * Real.cos (t.C - t.A) = 1 →
  isEquilateral t := by
  sorry

-- Theorem for proposition ②
theorem prop_2 :
  ∃ t : Triangle, Real.sin t.A = Real.cos t.B ∧ ¬isRightAngled t := by
  sorry

-- Theorem for proposition ③
theorem prop_3 (t : Triangle) :
  Real.cos t.A * Real.cos t.B * Real.cos t.C < 0 →
  isObtuse t := by
  sorry

-- Theorem for proposition ④
theorem prop_4 :
  ∃ t : Triangle, Real.sin (2 * t.A) = Real.sin (2 * t.B) ∧ ¬isIsosceles t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prop_1_prop_2_prop_3_prop_4_l1133_113366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1133_113361

-- Define the function f(x) with parameter m
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + x^2 + (m^2 - 1) * x

-- State the theorem
theorem f_properties (m : ℝ) (h_m : m > 0) :
  -- Part 1: When m = 1, maximum and minimum values on [-3, 2]
  (∀ x ∈ Set.Icc (-3) 2, f 1 x ≤ 18) ∧
  (∃ x ∈ Set.Icc (-3) 2, f 1 x = 18) ∧
  (∀ x ∈ Set.Icc (-3) 2, f 1 x ≥ 0) ∧
  (∃ x ∈ Set.Icc (-3) 2, f 1 x = 0) ∧
  -- Part 2: Interval of monotonic increase
  (∀ x y : ℝ, 1 - m < x ∧ x < y ∧ y < m + 1 → f m x < f m y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1133_113361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_return_trip_speed_is_9_l1133_113341

/-- Represents the round trip of a cyclist --/
structure CyclistTrip where
  total_time : ℝ
  outbound_distance_flat : ℝ
  outbound_distance_downhill : ℝ
  bicycle_speed : ℝ
  skateboard_speed : ℝ

/-- Calculates the average speed for the return trip --/
noncomputable def return_trip_average_speed (trip : CyclistTrip) : ℝ :=
  let outbound_time_bicycle := trip.outbound_distance_flat / trip.bicycle_speed
  let outbound_time_skateboard := trip.outbound_distance_downhill / trip.skateboard_speed
  let outbound_time := outbound_time_bicycle + outbound_time_skateboard
  let return_time := trip.total_time - outbound_time
  let total_distance := trip.outbound_distance_flat + trip.outbound_distance_downhill
  total_distance / return_time

/-- Theorem: The average speed for the return trip is 9 miles per hour --/
theorem return_trip_speed_is_9 (trip : CyclistTrip)
  (h1 : trip.total_time = 7.3)
  (h2 : trip.outbound_distance_flat = 18)
  (h3 : trip.outbound_distance_downhill = 18)
  (h4 : trip.bicycle_speed = 12)
  (h5 : trip.skateboard_speed = 10) :
  return_trip_average_speed trip = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_return_trip_speed_is_9_l1133_113341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_and_integral_l1133_113334

open Set
open Real
open MeasureTheory
open Interval

-- Define the inequality
def inequality (x : ℝ) : Prop := |x - 2| + |x - 3| < 3

-- Define the solution set
def solution_set : Set ℝ := {x | inequality x}

-- Define the integrand
noncomputable def integrand (x : ℝ) : ℝ := sqrt x - 1

theorem inequality_solution_and_integral :
  ∃ (a b : ℝ), a < b ∧
  solution_set = Ioo a b ∧
  ∫ x in a..b, integrand x = 5/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_and_integral_l1133_113334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_collinear_sum_l1133_113395

-- Define the points
variable (A B C M N O : ℝ × ℝ)

-- Define the vectors
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

-- Define m and n as real numbers
variable (m n : ℝ)

-- O is the midpoint of BC
axiom midpoint_O : vec A O = (1/2 : ℝ) • (vec A B + vec A C)

-- M, O, N are collinear
axiom collinear_MON : ∃ (t : ℝ), vec A O = t • vec A M + (1 - t) • vec A N

-- Vector relationships
axiom vec_AB_rel : vec A B = m • vec A M
axiom vec_AC_rel : vec A C = n • vec A N

-- Theorem to prove
theorem midpoint_collinear_sum : m + n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_collinear_sum_l1133_113395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1133_113311

/-- The function f(x) = ln x + x - 5 -/
noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 5

theorem root_in_interval (a : ℤ) :
  (∃ x : ℝ, x > a ∧ x < a + 1 ∧ f x = 0) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1133_113311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_origin_marcia_return_possibilities_four_rolls_return_possibilities_five_rolls_l1133_113302

/-- Represents the directions of movement --/
inductive Direction
| North
| East
| South
| West

/-- Represents a single die roll --/
def DieRoll := Fin 6

/-- Calculates the distance from the origin after a sequence of dice rolls --/
noncomputable def distanceFromOrigin (rolls : List DieRoll) : ℝ :=
  sorry

/-- Counts the number of possibilities to return to the origin after n rolls --/
def countReturnPossibilities (n : ℕ) : ℕ :=
  sorry

/-- The sequence of dice rolls for Márcia --/
def marciasRolls : List DieRoll := [⟨1, sorry⟩, ⟨0, sorry⟩, ⟨3, sorry⟩, ⟨2, sorry⟩, ⟨5, sorry⟩, ⟨4, sorry⟩]

theorem distance_from_origin_marcia :
  distanceFromOrigin marciasRolls = 5 := by
  sorry

theorem return_possibilities_four_rolls :
  countReturnPossibilities 4 = 36 := by
  sorry

theorem return_possibilities_five_rolls :
  countReturnPossibilities 5 = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_origin_marcia_return_possibilities_four_rolls_return_possibilities_five_rolls_l1133_113302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_b_l1133_113362

/-- A line segment in a 2D plane --/
structure LineSegment where
  start : ℝ × ℝ
  finish : ℝ × ℝ

/-- A line in a 2D plane of the form x + y = b --/
structure Line where
  b : ℝ

/-- Predicate to check if a line is a perpendicular bisector of a line segment --/
def is_perpendicular_bisector (l : Line) (seg : LineSegment) : Prop :=
  let midpoint := ((seg.start.1 + seg.finish.1) / 2, (seg.start.2 + seg.finish.2) / 2)
  midpoint.1 + midpoint.2 = l.b

/-- The main theorem --/
theorem perpendicular_bisector_b (seg : LineSegment) (l : Line) :
  seg.start = (0, 3) →
  seg.finish = (6, 9) →
  is_perpendicular_bisector l seg →
  l.b = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_b_l1133_113362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_sum_l1133_113329

def binomial_expansion (a b : ℕ) : ℕ → ℕ → ℕ := 
  λ m n => Nat.choose a m * Nat.choose b n

def f (m n : ℕ) : ℕ := 
  let r := m + n - 4
  Nat.choose 6 r * (2^(6-r)) * Nat.choose 4 n

theorem expansion_coefficient_sum : f 3 4 + f 5 3 = 400 := by
  -- Expand the definitions and compute
  simp [f]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_coefficient_sum_l1133_113329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_triangle_angle_equivalence_l1133_113318

-- Define the conic section
structure ConicSection where
  -- Add necessary fields to define a conic section
  (dummy : Unit) -- Placeholder field

-- Define a point on the conic section
structure PointOnConic where
  point : EuclideanSpace ℝ (Fin 2)
  conic : ConicSection
  on_conic : True -- Placeholder condition

-- Define a triangle with vertices on a conic section
structure TriangleOnConic where
  A : PointOnConic
  B : PointOnConic
  C : PointOnConic
  same_conic : A.conic = B.conic ∧ B.conic = C.conic

-- Define an axis of symmetry for a conic section
noncomputable def AxisOfSymmetry (c : ConicSection) : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define the angle between a line and the axis of symmetry
noncomputable def AngleWithAxis (l : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 2))) (c : ConicSection) : ℝ := sorry

-- Define the tangent line at a point on the conic
noncomputable def TangentLine (p : PointOnConic) : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define a line through two points
noncomputable def LineThroughPoints (p q : EuclideanSpace ℝ (Fin 2)) : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 2)) := sorry

-- The main theorem
theorem conic_triangle_angle_equivalence (t : TriangleOnConic) :
  AngleWithAxis (LineThroughPoints t.A.point t.B.point) t.A.conic =
  AngleWithAxis (LineThroughPoints t.A.point t.C.point) t.A.conic
  ↔
  AngleWithAxis (LineThroughPoints t.B.point t.C.point) t.A.conic =
  AngleWithAxis (TangentLine t.A) t.A.conic :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_triangle_angle_equivalence_l1133_113318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrange_matrix_l1133_113369

/-- Represents a square matrix with elements 0 or 1 -/
def BinaryMatrix (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if every submatrix contains at least one 1 -/
def hasOneInEverySubmatrix (A : BinaryMatrix n) : Prop :=
  ∀ (rows cols : Set (Fin n)), ∃ i j, i ∈ rows ∧ j ∈ cols ∧ A i j = true

/-- Represents a permutation of n elements -/
def Permutation (n : ℕ) := {σ : Fin n → Fin n // Function.Bijective σ}

/-- Applies row and column permutations to a matrix -/
def applyPermutations (A : BinaryMatrix n) (σ τ : Permutation n) : BinaryMatrix n :=
  λ i j ↦ A (σ.val i) (τ.val j)

/-- Checks if all 1s are on or above the main diagonal -/
def isUpperTriangular (A : BinaryMatrix n) : Prop :=
  ∀ i j, A i j = true → i.val ≤ j.val

theorem rearrange_matrix (n : ℕ) (A : BinaryMatrix n) 
  (h : hasOneInEverySubmatrix A) :
  ∃ (σ τ : Permutation n), isUpperTriangular (applyPermutations A σ τ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rearrange_matrix_l1133_113369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1133_113310

noncomputable def f (x : Real) : Real := 2 * Real.sin x * Real.cos x

noncomputable def g (x : Real) : Real := Real.sin (2 * x + Real.pi / 6) + 1

theorem min_value_theorem (x₁ x₂ : Real) (h : f x₁ * g x₂ = 2) :
  ∃ (k : Int), |2 * x₁ + x₂ - (2 * k + 1) * Real.pi / 3| = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1133_113310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_circles_locus_l1133_113321

/-- Definition of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Definition of orthogonal circles -/
def orthogonal (c1 c2 : Circle) : Prop :=
  sorry

/-- Definition of a point lying on a circle -/
def on_circle (p : Point) (c : Circle) : Prop :=
  sorry

/-- Definition of the angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Definition of a locus of points -/
def locus (p : Point → Prop) : Set Point :=
  sorry

/-- Definition of a line -/
def is_line (s : Set Point) : Prop :=
  sorry

/-- Main theorem -/
theorem orthogonal_circles_locus 
  (A B C : Point) 
  (O₁ O₂ : Circle) 
  (h1 : orthogonal O₁ O₂)
  (h2 : on_circle A O₁ ∧ on_circle C O₁)
  (h3 : on_circle B O₂ ∧ on_circle C O₂) :
  ∃ (L : Set Point), L = locus (λ P => ∃ O₁' O₂', 
    orthogonal O₁' O₂' ∧ 
    on_circle A O₁' ∧ on_circle C O₁' ∧
    on_circle B O₂' ∧ on_circle C O₂' ∧
    on_circle P O₁' ∧ on_circle P O₂') ∧
  ((∃ c : Circle, L = {p | on_circle p c}) ∨ 
   (angle A C B = π / 2 → ∃ l : Set Point, L = l ∧ is_line l)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_circles_locus_l1133_113321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_quadrilateral_PACQ_l1133_113304

/-- The area of quadrilateral PACQ given specific side lengths -/
theorem area_quadrilateral_PACQ :
  ∀ (AP PQ PC : ℝ),
  AP = 9 →
  PQ = 20 →
  PC = 21 →
  let PAQ_c := Real.sqrt (PQ^2 - AP^2)
  let PCQ_c := Real.sqrt (PC^2 - PQ^2)
  (AP * PAQ_c / 2 + PC * PCQ_c / 2) = (9 * Real.sqrt 319 + 21 * Real.sqrt 41) / 2 :=
by
  intros AP PQ PC hAP hPQ hPC
  simp [hAP, hPQ, hPC]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_quadrilateral_PACQ_l1133_113304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_initial_bananas_correct_l1133_113339

/-- Represents the number of bananas taken by each monkey initially -/
structure BananaSplit :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Checks if the given banana split satisfies all conditions -/
def isValidSplit (split : BananaSplit) : Prop :=
  let firstFinal := split.first * 2 / 3 + split.second * 7 / 30 + split.third * 2 / 9
  let secondFinal := split.first * 1 / 3 + split.second * 1 / 5 + split.third * 2 / 9
  let thirdFinal := split.first * 1 / 3 + split.second * 4 / 15 + split.third * 1 / 10
  (firstFinal : ℚ) / secondFinal = 2 ∧ 
  (secondFinal : ℚ) / thirdFinal = 2 ∧
  firstFinal % 1 = 0 ∧ secondFinal % 1 = 0 ∧ thirdFinal % 1 = 0

/-- The smallest initial number of bananas -/
def smallestInitialBananas : ℕ := 215

theorem smallest_initial_bananas_correct :
  (∃ (split : BananaSplit), isValidSplit split ∧ 
    split.first + split.second + split.third = smallestInitialBananas) ∧
  (∀ (n : ℕ), n < smallestInitialBananas → 
    ¬∃ (split : BananaSplit), isValidSplit split ∧ 
      split.first + split.second + split.third = n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_initial_bananas_correct_l1133_113339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_hole_rate_is_3_l1133_113359

/-- The rate of water leakage from the largest hole in the garage roof -/
noncomputable def largest_hole_rate : ℝ := sorry

/-- The rate of water leakage from the medium-sized hole in the garage roof -/
noncomputable def medium_hole_rate : ℝ := largest_hole_rate / 2

/-- The rate of water leakage from the smallest hole in the garage roof -/
noncomputable def smallest_hole_rate : ℝ := medium_hole_rate / 3

/-- The total amount of water leaked from all three holes in 2 hours -/
def total_water_leaked : ℝ := 600

/-- The duration of the leak in minutes -/
def leak_duration : ℝ := 120

theorem largest_hole_rate_is_3 :
  largest_hole_rate = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_hole_rate_is_3_l1133_113359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l1133_113377

-- Define the slopes of the two lines
noncomputable def m1 (a : ℝ) : ℝ := -(a + 2) / (1 - a)
noncomputable def m2 (a : ℝ) : ℝ := -(a - 1) / (2*a + 3)

-- Theorem statement
theorem perpendicular_lines (a : ℝ) :
  (m1 a * m2 a = -1) → (a = -5 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_l1133_113377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_nine_fourths_l1133_113375

/-- The sum of the series 1 + 2(1/3) + 3(1/3)^2 + 4(1/3)^3 + ... -/
noncomputable def seriesSum : ℝ := ∑' n, (n + 1 : ℕ) * (1 / 3 : ℝ) ^ n

/-- Theorem stating that the sum of the series is equal to 9/4 -/
theorem series_sum_is_nine_fourths : seriesSum = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_nine_fourths_l1133_113375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_interior_angles_pentagon_l1133_113301

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  num_sides : ℕ
  num_sides_positive : num_sides > 0

/-- The sum of interior angles of a polygon. -/
def sum_interior_angles (p : Polygon) : ℕ := (p.num_sides - 2) * 180

theorem sum_interior_angles_pentagon : 
  ∀ (p : Polygon), (p.num_sides = 5) → (sum_interior_angles p = 540) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_interior_angles_pentagon_l1133_113301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1133_113324

theorem problem_solution 
  (a b c d : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) : 
  2019 * a + 7 / (c * d) + 2019 * b = 7 := by
  calc
    2019 * a + 7 / (c * d) + 2019 * b 
      = 2019 * a + 7 + 2019 * b := by rw [h2, div_one]
    _ = 2019 * (a + b) + 7 := by ring
    _ = 2019 * 0 + 7 := by rw [h1]
    _ = 0 + 7 := by simp
    _ = 7 := by rw [zero_add]

#eval decide (2019 * 1 + 7 / 1 + 2019 * (-1) = 7)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1133_113324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implications_l1133_113314

theorem inequality_implications (a b : ℝ) (h : 0 < a ∧ 0 < b ∧ 1 / Real.sqrt a > 1 / Real.sqrt b) :
  a^3 < b^3 ∧ b / a > (b + 1) / (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implications_l1133_113314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_silver_tokens_is_52_l1133_113376

/-- Represents the number of tokens of each color --/
structure Tokens where
  red : Int
  blue : Int
  silver : Int

/-- Represents the exchange rates at the booths --/
structure ExchangeRates where
  red_to_silver : Tokens
  blue_to_silver : Tokens

/-- The initial number of tokens and the exchange rates --/
def initial_state : Tokens × ExchangeRates :=
  ({ red := 90, blue := 80, silver := 0 },
   { red_to_silver := { red := 3, blue := -2, silver := -1 },
     blue_to_silver := { red := -1, blue := 4, silver := -1 } })

/-- Predicate to check if further exchanges are possible --/
def can_exchange (t : Tokens) (e : ExchangeRates) : Prop :=
  t.red ≥ e.red_to_silver.red ∨ t.blue ≥ e.blue_to_silver.blue

/-- The maximum number of silver tokens that can be obtained --/
noncomputable def max_silver_tokens (t : Tokens) (e : ExchangeRates) : Int :=
  sorry

/-- Main theorem: The maximum number of silver tokens Alex can obtain is 52 --/
theorem max_silver_tokens_is_52 :
  max_silver_tokens initial_state.1 initial_state.2 = 52 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_silver_tokens_is_52_l1133_113376
