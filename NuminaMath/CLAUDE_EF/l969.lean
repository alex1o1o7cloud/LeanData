import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_row_fourth_number_l969_96969

theorem ninth_row_fourth_number : ∀ n : ℕ, n = 9 → (7 * n - 3 = 60) :=
  fun n h =>
  calc
    7 * n - 3 = 7 * 9 - 3 := by rw [h]
    _ = 63 - 3 := by rfl
    _ = 60 := by rfl

#check ninth_row_fourth_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_row_fourth_number_l969_96969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercept_line_equation_l969_96983

/-- A line passing through a point with equal intercepts on both axes -/
structure EqualInterceptLine where
  -- The slope of the line
  slope : ℝ
  -- The y-intercept of the line
  y_intercept : ℝ
  -- The line passes through (5, -2)
  point_condition : -2 = slope * 5 + y_intercept
  -- The intercepts on both axes are equal
  equal_intercepts : y_intercept = -y_intercept / slope

/-- The equation of a line with equal intercepts passing through (5, -2) -/
def line_equation (l : EqualInterceptLine) : ℝ → ℝ → Prop :=
  λ x y ↦ y = l.slope * x + l.y_intercept

theorem equal_intercept_line_equation :
  ∀ l : EqualInterceptLine, 
  (∀ x y, line_equation l x y ↔ x + y - 3 = 0) ∨
  (∀ x y, line_equation l x y ↔ 2*x + 5*y = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_intercept_line_equation_l969_96983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_690_degrees_l969_96952

theorem cos_690_degrees : Real.cos (690 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_690_degrees_l969_96952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_number_29_pairing_equation_perfect_expression_equation_l969_96977

-- Definition of a perfect number
def isPerfectNumber (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + b^2

-- Definition of a perfect expression
def isPerfectExpression (e : ℝ → ℝ → ℝ) : Prop :=
  ∃ M N : ℝ → ℝ → ℝ, ∀ x y, e x y = (M x y)^2 + (N x y)^2

theorem perfect_number_29 : isPerfectNumber 29 := by
  sorry

theorem pairing_equation : 
  ∃ m n : ℝ, (∀ x, x^2 - 4*x + 5 = (x - m)^2 + n) ∧ m * n = 2 := by
  sorry

theorem perfect_expression_equation :
  isPerfectExpression (λ x y ↦ x^2 + y^2 - 2*x + 4*y + 5) →
  ∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 5 = 0 → x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_number_29_pairing_equation_perfect_expression_equation_l969_96977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_k_closing_price_l969_96991

/-- Calculates the closing price of a stock given the opening price and percent increase -/
noncomputable def closing_price (opening_price : ℝ) (percent_increase : ℝ) : ℝ :=
  opening_price * (1 + percent_increase / 100)

/-- Theorem stating that given the specific opening price and percent increase, 
    the closing price of stock K is $29.00 -/
theorem stock_k_closing_price :
  let opening_price : ℝ := 28
  let percent_increase : ℝ := 3.571428571428581
  closing_price opening_price percent_increase = 29 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval closing_price 28 3.571428571428581

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_k_closing_price_l969_96991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_length_calculation_l969_96922

/-- Proves that the length of a rectangular plot is approximately 17.7 meters given specific conditions -/
theorem plot_length_calculation (width : ℝ) (path_width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) 
  (h1 : width = 70)
  (h2 : path_width = 2.5)
  (h3 : cost_per_sqm = 0.9)
  (h4 : total_cost = 742.5) :
  ∃ (length : ℝ), (length ≥ 17.69 ∧ length ≤ 17.71) ∧ 
  total_cost = (length - 2 * path_width) * (width - 2 * path_width) * cost_per_sqm := by
  sorry

#check plot_length_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_length_calculation_l969_96922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_interval_l969_96927

open Real

/-- The function f(x) defined on positive real numbers -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x - (1/2) * a * x^2 - 2*x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 1/x - a*x - 2

theorem monotone_decreasing_interval (a : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ x₁ < x₂ ∧
    ∀ x ∈ Set.Ioo x₁ x₂, f_prime a x < 0) ↔ a > -1 := by
  sorry

#check monotone_decreasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_decreasing_interval_l969_96927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l969_96941

theorem equation_solution : ∃ X : ℝ, 
  (0.625 * 0.0729 * X) / (0.0017 * 0.025 * 8.1) = 382.5 ∧ 
  |X - 2.33075| < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l969_96941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l969_96909

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side length opposite to A
  b : ℝ  -- Side length opposite to B
  c : ℝ  -- Side length opposite to C
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the vectors m and n
noncomputable def m (t : Triangle) : ℝ × ℝ := (t.a / Real.sin (t.A + t.B), t.c - 2 * t.b)
noncomputable def n (t : Triangle) : ℝ × ℝ := (Real.sin (2 * t.C), 1)

-- Define the dot product of m and n
noncomputable def dot_product (t : Triangle) : ℝ := (m t).1 * (n t).1 + (m t).2 * (n t).2

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : dot_product t = 0) : 
  t.A = π / 3 ∧ 
  (t.a = 1 → 2 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l969_96909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_shaded_area_coefficients_sum_l969_96995

noncomputable section

-- Define the grid
def grid_side : ℝ := 4
def square_side : ℝ := 2

-- Define the shapes
def ellipse_major_axis : ℝ := 4
def ellipse_minor_axis : ℝ := 2
def circle_radius : ℝ := 1
def num_circles : ℕ := 3

-- Calculate areas
noncomputable def grid_area : ℝ := grid_side * grid_side * square_side * square_side
noncomputable def ellipse_area : ℝ := Real.pi * (ellipse_major_axis / 2) * (ellipse_minor_axis / 2)
noncomputable def circles_area : ℝ := num_circles * Real.pi * circle_radius * circle_radius

-- Define the visible shaded area
noncomputable def visible_shaded_area : ℝ := grid_area - (ellipse_area + circles_area)

-- Theorem to prove
theorem visible_shaded_area_coefficients_sum :
  ∃ (A B : ℝ), visible_shaded_area = A - B * Real.pi ∧ A + B = 69 := by
  -- We'll use A = 64 and B = 5 as in the solution
  use 64, 5
  constructor
  · -- Prove that visible_shaded_area = 64 - 5 * Real.pi
    sorry
  · -- Prove that 64 + 5 = 69
    norm_num

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_visible_shaded_area_coefficients_sum_l969_96995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_inscribed_circle_ratio_l969_96949

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Center of the inscribed circle
  O : ℝ × ℝ
  -- Assertion that ABC is a right triangle
  is_right_triangle : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  -- Assertion that O is the center of the inscribed circle
  is_inscribed_center : True  -- This would need a more complex definition in practice
  -- Assertion that O is closer to one end of the hypotenuse
  O_closer_to_hypotenuse_end : True  -- This would need a more precise definition

/-- The main theorem -/
theorem right_triangle_inscribed_circle_ratio
  (t : RightTriangleWithInscribedCircle)
  (half_hypotenuse_right_angle : True)  -- This condition needs a more precise definition
  : ∃ (k : ℝ), 
    (Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2) = 3*k) ∧
    (Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2) = 4*k) ∧
    (Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2) = 5*k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_inscribed_circle_ratio_l969_96949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sequence_l969_96954

/-- A strictly decreasing sequence of positive integers -/
def StrictlyDecreasingSequence (n : ℕ) := { s : Fin n → ℕ+ // ∀ i j, i < j → s i > s j }

/-- The property that no term in the sequence divides any other term -/
def NoDivision {n : ℕ} (s : StrictlyDecreasingSequence n) :=
  ∀ i j, i ≠ j → ¬(s.val i ∣ s.val j)

/-- The set S_n of all strictly decreasing sequences of n positive integers with no division property -/
def S_n (n : ℕ) := { s : StrictlyDecreasingSequence n // NoDivision s }

/-- The ordering relation between two sequences -/
def LexOrder {n : ℕ} (a b : S_n n) :=
  ∃ k : Fin n, (a.val.val k < b.val.val k) ∧ (∀ i : Fin n, i < k → a.val.val i = b.val.val i)

/-- Construct the sequence A -/
noncomputable def ConstructA (n : ℕ) : S_n n := sorry

/-- The main theorem statement -/
theorem smallest_sequence (n : ℕ) :
  ∀ B : S_n n, B ≠ ConstructA n → LexOrder (ConstructA n) B :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sequence_l969_96954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_problem_l969_96994

/-- Represents a person's age -/
def Age := ℕ

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Checks if one number is a multiple of another -/
def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem birthday_problem (michael sarah tim : ℕ) (n : ℕ) : 
  michael = sarah + 2 →
  sarah = tim + 8 →
  tim = 2 →
  (∀ k : ℕ, k < 4 → is_multiple (sarah + k) (tim + k)) →
  (∀ m : ℕ, m < n → ¬ is_multiple (michael + m) (tim + m)) →
  is_multiple (michael + n) (tim + n) →
  sum_of_digits (michael + n) = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_problem_l969_96994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_theorem_l969_96935

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x)

noncomputable def g (x φ : ℝ) : ℝ := 2 * Real.sin (2 * (x - φ))

theorem translation_theorem (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi / 2) :
  (∃ x₁ x₂ : ℝ, |f x₁ - g x₂ φ| = 4 ∧ 
    ∀ y₁ y₂ : ℝ, |f y₁ - g y₂ φ| = 4 → |x₁ - x₂| ≤ |y₁ - y₂|) →
  (∀ x₁ x₂ : ℝ, |f x₁ - g x₂ φ| = 4 → |x₁ - x₂| ≥ Real.pi / 6) →
  (∃ x₁ x₂ : ℝ, |f x₁ - g x₂ φ| = 4 ∧ |x₁ - x₂| = Real.pi / 6) →
  φ = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_theorem_l969_96935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_a_given_sum_identity_l969_96951

theorem max_sin_a_given_sum_identity (a b : ℝ) :
  (∀ a b : ℝ, Real.sin (a + b) = Real.sin a + Real.sin b) →
  (∃ a : ℝ, Real.sin a = 1) ∧ (∀ a : ℝ, Real.sin a ≤ 1) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_a_given_sum_identity_l969_96951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_device_improvement_l969_96979

noncomputable def old_data : List ℝ := [9.8, 10.3, 10.0, 10.2, 9.9, 9.8, 10.0, 10.1, 10.2, 9.7]
noncomputable def new_data : List ℝ := [10.1, 10.4, 10.1, 10.0, 10.1, 10.3, 10.6, 10.5, 10.4, 10.5]

noncomputable def mean (data : List ℝ) : ℝ := (data.sum) / (data.length : ℝ)

noncomputable def variance (data : List ℝ) : ℝ :=
  let m := mean data
  (data.map (λ x => (x - m)^2)).sum / (data.length : ℝ)

noncomputable def significant_improvement (old_data new_data : List ℝ) : Prop :=
  let x_bar := mean old_data
  let y_bar := mean new_data
  let s1_squared := variance old_data
  let s2_squared := variance new_data
  y_bar - x_bar ≥ 2 * Real.sqrt ((s1_squared + s2_squared) / 10)

theorem new_device_improvement : significant_improvement old_data new_data := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_device_improvement_l969_96979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_initial_number_l969_96918

theorem largest_initial_number :
  ∃ (a b c d e : ℕ),
    189 ∉ ({a, b, c, d, e} : Set ℕ) ∧
    ¬(189 ∣ a) ∧ ¬(189 ∣ b) ∧ ¬(189 ∣ c) ∧ ¬(189 ∣ d) ∧ ¬(189 ∣ e) ∧
    189 + a + b + c + d + e = 200 ∧
    ∀ n > 189, ¬∃ (x y z w v : ℕ),
      n ∉ ({x, y, z, w, v} : Set ℕ) ∧
      ¬(n ∣ x) ∧ ¬(n ∣ y) ∧ ¬(n ∣ z) ∧ ¬(n ∣ w) ∧ ¬(n ∣ v) ∧
      n + x + y + z + w + v = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_initial_number_l969_96918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_power_sum_l969_96965

theorem sin_power_sum (φ : ℝ) (x : ℂ) (n : ℕ) 
  (h1 : 0 < φ) (h2 : φ < π / 2) (h3 : x + 1 / x = 2 * Real.sin φ) : 
  x^n + (1 / x)^n = 2 * Real.sin (n * φ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_power_sum_l969_96965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reader_distance_to_C_l969_96912

/-- Represents a location --/
inductive Location where
  | A
  | B
  | C

/-- Represents a time of day --/
structure Time where
  hour : Nat
  minute : Nat

/-- Represents a person --/
inductive Person where
  | Reader
  | Friend

/-- Represents the problem setup --/
structure ProblemSetup where
  distance_AB : Real
  reader_start : Time
  friend_start : Time
  friend_speed : Real
  joint_speed : Real
  arrival_time : Time

/-- Represents the movement of a person --/
structure Movement where
  person : Person
  start : Location
  finish : Location
  start_time : Time
  speed : Real

/-- Helper function to calculate distance covered --/
def distance_covered_by_reader_to_C (setup : ProblemSetup) : Real :=
  sorry

/-- Main theorem --/
theorem reader_distance_to_C (setup : ProblemSetup) 
  (h1 : setup.distance_AB = 1)
  (h2 : setup.reader_start = ⟨12, 0⟩)
  (h3 : setup.friend_start = ⟨12, 15⟩)
  (h4 : setup.friend_speed = 5)
  (h5 : setup.joint_speed = 4)
  (h6 : setup.arrival_time = ⟨13, 0⟩) :
  ∃ (d : Real), d = 2/3 ∧ d = distance_covered_by_reader_to_C setup := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reader_distance_to_C_l969_96912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lauren_total_earnings_l969_96932

/-- Represents Lauren's earnings over a period --/
structure LaurenEarnings where
  -- Monday earnings in USD
  monday_commercial_rate : ℚ
  monday_subscription_rate : ℚ
  monday_commercial_views : ℕ
  monday_subscriptions : ℕ
  
  -- Tuesday earnings in EUR
  tuesday_commercial_rate : ℚ
  tuesday_subscription_rate : ℚ
  tuesday_commercial_views : ℕ
  tuesday_subscriptions : ℕ
  
  -- Weekend earnings in GBP
  weekend_merchandise_sales : ℚ
  weekend_earnings_percentage : ℚ
  
  -- Exchange rates
  usd_to_eur_rate : ℚ
  gbp_to_usd_rate : ℚ

/-- Calculates Lauren's total earnings in USD --/
def total_earnings (e : LaurenEarnings) : ℚ :=
  -- Monday earnings
  (e.monday_commercial_rate * e.monday_commercial_views +
   e.monday_subscription_rate * e.monday_subscriptions) +
  -- Tuesday earnings converted to USD
  ((e.tuesday_commercial_rate * e.tuesday_commercial_views +
    e.tuesday_subscription_rate * e.tuesday_subscriptions) / e.usd_to_eur_rate) +
  -- Weekend earnings converted to USD
  (e.weekend_merchandise_sales * e.weekend_earnings_percentage * e.gbp_to_usd_rate)

/-- Theorem stating Lauren's total earnings --/
theorem lauren_total_earnings : 
  ∀ (e : LaurenEarnings), 
    e.monday_commercial_rate = 2/5 ∧
    e.monday_subscription_rate = 4/5 ∧
    e.monday_commercial_views = 80 ∧
    e.monday_subscriptions = 20 ∧
    e.tuesday_commercial_rate = 2/5 ∧
    e.tuesday_subscription_rate = 3/4 ∧
    e.tuesday_commercial_views = 100 ∧
    e.tuesday_subscriptions = 27 ∧
    e.weekend_merchandise_sales = 100 ∧
    e.weekend_earnings_percentage = 1/10 ∧
    e.usd_to_eur_rate = 17/20 ∧
    e.gbp_to_usd_rate = 69/50 →
    total_earnings e = 3317/25 := by
  sorry

#eval (3317 : ℚ) / 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lauren_total_earnings_l969_96932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l969_96955

noncomputable section

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2/36 + y^2/16 = 1

/-- Definition of the focus -/
noncomputable def focus : ℝ × ℝ := (2 * Real.sqrt 5, 0)

/-- Definition of point A -/
noncomputable def point_A : ℝ × ℝ := (4.8, 8 * Real.sqrt 7 / 6)

/-- Definition of point B -/
noncomputable def point_B : ℝ × ℝ := (2 * Real.sqrt 5, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem ellipse_chord_length :
  is_on_ellipse point_A.1 point_A.2 ∧
  is_on_ellipse point_B.1 point_B.2 ∧
  distance point_A focus = 2 ∧
  -- Angle condition is implicit in the point definitions
  distance point_B focus = 8 * Real.sqrt 7 / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_length_l969_96955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_difference_theorem_l969_96970

/-- Represents the number of hours passed -/
noncomputable def t : ℝ := 15

/-- The time shown by the fast clock after t hours -/
noncomputable def fast_clock (t : ℝ) : ℝ := t + t / 30

/-- The time shown by the slow clock after t hours -/
noncomputable def slow_clock (t : ℝ) : ℝ := t - t / 30

/-- Theorem stating that after 15 hours, the fast clock exceeds the slow clock by 1 hour -/
theorem clock_difference_theorem : fast_clock t - slow_clock t = 1 := by
  -- Unfold the definitions
  unfold fast_clock slow_clock t
  -- Simplify the expression
  simp [add_sub_cancel]
  -- Prove the equality
  ring

#check clock_difference_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_difference_theorem_l969_96970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_B_given_A_l969_96919

-- Define the set of people and schools
inductive Person : Type
  | A | B | C | D

inductive School : Type
  | A | B | C | D

-- Define an assignment as a function from Person to School
def Assignment := Person → School

-- Define the probability space
def Ω : Type := Assignment

-- Define the event that A does not go to school A
def EventA (ω : Ω) : Prop := ω Person.A ≠ School.A

-- Define the event that B does not go to school B
def EventB (ω : Ω) : Prop := ω Person.B ≠ School.B

-- Define the probability measure
noncomputable def P : Set Ω → ℝ := sorry

-- State the theorem
theorem prob_B_given_A :
  P {ω | EventB ω ∧ EventA ω} / P {ω | EventA ω} = 7 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_B_given_A_l969_96919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_at_2023_l969_96989

/-- The repeating decimal expansion of 7/26 -/
def repeating_decimal : List Nat := [2, 6, 9, 2, 3, 0, 7, 6, 9, 2, 3, 0]

/-- The length of the repeating sequence -/
def repeat_length : Nat := 12

/-- The position we're interested in -/
def target_position : Nat := 2023

theorem digit_at_2023 :
  (repeating_decimal.get! ((target_position - 1) % repeat_length)) = 0 := by
  sorry

#eval repeating_decimal.get! ((target_position - 1) % repeat_length)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_at_2023_l969_96989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_and_initial_phase_l969_96985

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 3 * sin (-x + π/6)

-- State the theorem
theorem phase_and_initial_phase :
  ∃ (phase : ℝ → ℝ) (initial_phase : ℝ),
    (∀ x, f x = 3 * sin (phase x)) ∧
    (phase 0 = initial_phase) ∧
    (phase = λ x ↦ x + 5*π/6) ∧
    (initial_phase = 5*π/6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_and_initial_phase_l969_96985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l969_96903

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x / (9 - x^2)

-- Define the domain of f(x)
def domain : Set ℝ := Set.Ioo (-3) 3

-- State the theorem
theorem f_properties :
  -- f(x) is an odd function
  (∀ x, x ∈ domain → f (-x) = -f x) ∧
  -- f(1) = 1/8
  f 1 = 1/8 ∧
  -- f(x) is monotonically increasing on (-3, 3)
  (∀ x y, x ∈ domain → y ∈ domain → x < y → f x < f y) ∧
  -- The solution set for f(t-1) + f(t) < 0 is (-2, 1/2)
  (∀ t : ℝ, f (t-1) + f t < 0 ↔ t ∈ Set.Ioo (-2) (1/2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l969_96903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sale_ratio_l969_96958

theorem book_sale_ratio :
  ∀ (x y : ℝ),
  x + y = 10 →
  2.5 * x + 2 * y = 22 →
  x ≥ 0 →
  y ≥ 0 →
  x / (x + y) = 2 / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_sale_ratio_l969_96958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_theorem_l969_96936

noncomputable section

def vector_b : ℝ × ℝ := (0, 1)

def angle_between (v w : ℝ × ℝ) : ℝ := Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem vector_angle_theorem (m : ℝ) :
  let a : ℝ × ℝ := (m, 1)
  angle_between a vector_b = π / 3 →
  m = Real.sqrt 3 ∨ m = -Real.sqrt 3 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_theorem_l969_96936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l969_96972

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (s : ℝ) :
  (0 < A) ∧ (A < Real.pi) ∧ (0 < B) ∧ (B < Real.pi) ∧ (0 < C) ∧ (C < Real.pi) ∧
  (A + B + C = Real.pi) →
  2 * a * Real.sin B = Real.sqrt 3 * b →
  Real.sin (2 * A + Real.pi / 6) = 1 / 2 →
  b = 1 →
  1 / 2 * b * c * Real.sin A = Real.sqrt 3 / 2 →
  (A = Real.pi / 3) ∧ (a^2 + b^2 = c^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l969_96972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_salary_problem_l969_96967

theorem employee_salary_problem (num_employees : ℕ) (manager_salary : ℕ) 
  (average_increase : ℕ) (h1 : num_employees = 20) (h2 : manager_salary = 3700) 
  (h3 : average_increase = 100) :
  ∃ (average_employees : ℕ),
    (num_employees + 1) * (average_employees + average_increase) =
    num_employees * average_employees + manager_salary ∧
    average_employees = 1600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employee_salary_problem_l969_96967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l969_96975

noncomputable def f (x : ℝ) : ℝ := if x ≤ 0 then -x + 2 else x + 2

theorem solution_set_of_inequality :
  ∀ x : ℝ, (f x ≥ x^2) ↔ x ∈ Set.Icc (-2) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l969_96975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_h_increasing_l969_96976

-- Define the interval (0, +∞)
def openPositiveReals : Set ℝ := { x : ℝ | x > 0 }

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.cos x
def h (x : ℝ) : ℝ := x^2
def k (x : ℝ) : ℝ := 1  -- x^0 is always 1 for x ≠ 0

-- Define what it means for a function to be increasing on an interval
def IsIncreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x < f y

-- State the theorem
theorem only_h_increasing :
  ¬(IsIncreasingOn f openPositiveReals) ∧
  ¬(IsIncreasingOn g openPositiveReals) ∧
  (IsIncreasingOn h openPositiveReals) ∧
  ¬(IsIncreasingOn k openPositiveReals) :=
by
  sorry -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_h_increasing_l969_96976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_eq_neg_two_inequality_solution_set_l969_96930

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 1 + m / (3^x + 1)

-- Part 1: Proving m = -2 if f is an odd function
theorem odd_function_implies_m_eq_neg_two (m : ℝ) :
  (∀ x, f m x = -f m (-x)) → m = -2 := by sorry

-- Part 2: Proving the solution set of the inequality
theorem inequality_solution_set (x : ℝ) :
  (f (-2) (x^2 - x - 1) + 1/2 < 0) ↔ (0 < x ∧ x < 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_m_eq_neg_two_inequality_solution_set_l969_96930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_plus_pi_3_l969_96978

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin (Real.pi + x) * Real.cos (-3 * Real.pi - x) - 2 * Real.sin (Real.pi / 2 - x) * Real.cos (Real.pi - x)

theorem cos_2alpha_plus_pi_3 (α : ℝ) (h1 : f (α / 2 - Real.pi / 12) = 3 / 2) (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  Real.cos (2 * α + Real.pi / 3) = (7 + 3 * Real.sqrt 5) / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2alpha_plus_pi_3_l969_96978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_series_sum_equals_2033136_l969_96946

/-- The sum of the alternating series of products of consecutive integers up to 2015×2016 -/
def alternating_series_sum : ℕ → ℤ
  | 0 => 0
  | n + 1 => alternating_series_sum n + (if n % 2 = 0 then ((2*n + 1) * (2*n + 2) : ℤ) else -((2*n + 1) * (2*n + 2) : ℤ))

/-- The theorem stating that the sum of the series equals 2033136 -/
theorem alternating_series_sum_equals_2033136 : alternating_series_sum 1008 = 2033136 := by
  sorry

#eval alternating_series_sum 1008

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_series_sum_equals_2033136_l969_96946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pentagon_area_l969_96933

/-- Pentagon inscribed in a unit circle -/
structure InscribedPentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  radius : ℝ
  AB_length : ℝ
  angle_ABE : ℝ
  angle_EBD : ℝ
  BC_equals_CD : Prop

/-- Conditions for the specific pentagon -/
def SpecificPentagon (p : InscribedPentagon) : Prop :=
  p.radius = 1 ∧
  p.AB_length = Real.sqrt 2 ∧
  p.angle_ABE = Real.pi / 4 ∧
  p.angle_EBD = Real.pi / 6 ∧
  p.BC_equals_CD

/-- Area of the pentagon -/
noncomputable def PentagonArea (p : InscribedPentagon) : ℝ := sorry

/-- Theorem stating the area of the specific pentagon -/
theorem specific_pentagon_area (p : InscribedPentagon) (h : SpecificPentagon p) :
  PentagonArea p = 1 + (3 * Real.sqrt 3) / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pentagon_area_l969_96933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l969_96929

noncomputable def g (x : ℝ) : ℝ := (3*x - 8) * (x - 4) * (x + 2) / x

theorem inequality_solution (x : ℝ) (h : x ≠ 0) :
  g x ≥ 0 ↔ x ∈ Set.Ici (-2) ∪ Set.Icc (8/3) 4 ∪ Set.Ioi 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l969_96929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_increasing_condition_log_increasing_iff_l969_96904

-- Define the logarithmic function
noncomputable def log_function (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define what it means for a function to be increasing
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem log_increasing_condition (a : ℝ) :
  (a > 2 → is_increasing (log_function a)) ∧
  ¬(is_increasing (log_function a) → a > 2) := by
  sorry

-- Additional lemma to show that a > 1 is necessary and sufficient
theorem log_increasing_iff (a : ℝ) :
  is_increasing (log_function a) ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_increasing_condition_log_increasing_iff_l969_96904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_cycling_trip_l969_96917

/-- The remaining distance of Pascal's cycling trip. -/
noncomputable def remaining_distance : ℝ := 256

/-- Pascal's current speed in miles per hour. -/
noncomputable def current_speed : ℝ := 8

/-- The reduced speed in miles per hour. -/
noncomputable def reduced_speed : ℝ := current_speed - 4

/-- The increased speed in miles per hour. -/
noncomputable def increased_speed : ℝ := current_speed * 1.5

/-- Time taken at current speed in hours. -/
noncomputable def current_time : ℝ := remaining_distance / current_speed

/-- Time taken at reduced speed in hours. -/
noncomputable def reduced_time : ℝ := remaining_distance / reduced_speed

/-- Time taken at increased speed in hours. -/
noncomputable def increased_time : ℝ := remaining_distance / increased_speed

theorem pascal_cycling_trip :
  (reduced_time - current_time = 16) ∧
  (reduced_time - increased_time = 16) →
  remaining_distance = 256 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_cycling_trip_l969_96917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_on_circle_l969_96943

noncomputable def circleEq (x y : ℝ) : Prop := (x - 11)^2 + (y - 13)^2 = 116

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := 
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem farthest_point_on_circle : 
  ∀ x y : ℝ, circleEq x y → 
    distance x y 41 25 ≤ distance 1 9 41 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_on_circle_l969_96943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l969_96996

theorem perpendicular_vectors (e₁ e₂ : ℝ × ℝ × ℝ) (k : ℝ) : 
  ‖e₁‖ = 1 →
  ‖e₂‖ = 1 →
  e₁ • e₂ = -1/2 →
  let a := e₁ - 2 • e₂
  let b := k • e₁ + e₂
  a • b = 0 →
  k = 5/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l969_96996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_bicycle_sale_price_l969_96944

/-- The sale price of a bicycle at Store Q given its price at Store P and percentage increases --/
def bicycle_sale_price 
  (price_p : ℝ)                    -- Price at Store P
  (regular_increase : ℝ)           -- Percentage increase for regular price at Store Q
  (discount : ℝ)                   -- Discount percentage at Store Q
  : ℝ :=
  
  let regular_q := price_p * (1 + regular_increase)
  let sale_price := regular_q * (1 - discount)
  
  sale_price

/-- Proof of the theorem --/
theorem prove_bicycle_sale_price : 
  ∃ (price_p regular_increase discount : ℝ),
    price_p = 200 ∧
    regular_increase = 0.15 ∧
    discount = 0.10 ∧
    bicycle_sale_price price_p regular_increase discount = 207 :=
by
  -- Provide the values
  let price_p : ℝ := 200
  let regular_increase : ℝ := 0.15
  let discount : ℝ := 0.10

  -- Check that bicycle_sale_price with these values equals 207
  have h : bicycle_sale_price price_p regular_increase discount = 207 := by
    -- Unfold the definition and simplify
    unfold bicycle_sale_price
    simp
    -- Perform the calculation
    norm_num

  -- Prove the existence
  exact ⟨price_p, regular_increase, discount, rfl, rfl, rfl, h⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_bicycle_sale_price_l969_96944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l969_96921

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x - (1/2) * x^2 + x

-- Define the golden ratio
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

-- Theorem statement
theorem monotonic_increasing_interval :
  ∀ x y : ℝ, 0 < y → y < x → x < φ →
  f y < f x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_l969_96921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_P_less_than_Q_l969_96953

/-- Polynomial of degree 4 -/
def P (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

/-- Polynomial of degree 2 -/
def Q (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

/-- The theorem statement -/
theorem exists_point_P_less_than_Q 
  (a b c d p q : ℝ) 
  (I : Set ℝ) 
  (h_length : ∃ x y, x ∈ I ∧ y ∈ I ∧ |x - y| > 2)
  (h_neg_on_I : ∀ x, x ∈ I → P a b c d x < 0 ∧ Q p q x < 0)
  (h_nonneg_outside_I : ∀ x, x ∉ I → P a b c d x ≥ 0 ∧ Q p q x ≥ 0) :
  ∃ x₀ : ℝ, P a b c d x₀ < Q p q x₀ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_point_P_less_than_Q_l969_96953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l969_96993

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (1/2) * x - 1 else 1/x

-- State the theorem
theorem f_inequality_range :
  {a : ℝ | f a ≤ a} = {a : ℝ | a ≥ -1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l969_96993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sedan_overtake_truck_time_l969_96982

/-- The time taken for a sedan to overtake a truck on a highway -/
theorem sedan_overtake_truck_time : 
  ∀ (sedan_length truck_length : ℝ) 
    (sedan_speed truck_speed : ℝ) 
    (overtake_time : ℝ),
  sedan_length = 4 →
  truck_length = 12 →
  sedan_speed = 110 →
  truck_speed = 100 →
  overtake_time * sedan_speed / 3600 = 
    overtake_time * truck_speed / 3600 + (sedan_length + truck_length) / 1000 →
  (↑(Int.floor (overtake_time * 10 + 0.5)) / 10 : ℝ) = 5.8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sedan_overtake_truck_time_l969_96982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l969_96934

/-- Triangle ABC with sides a, b, c corresponding to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  area : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  0 < t.a ∧ 0 < t.b ∧ 0 < t.c ∧
  0 < t.A ∧ t.A < Real.pi ∧ 0 < t.B ∧ t.B < Real.pi ∧ 0 < t.C ∧ t.C < Real.pi ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a * Real.cos t.B + t.b * Real.cos t.A = 2 * t.a

theorem triangle_theorem (t : Triangle) (h : TriangleConditions t) :
  (t.B = Real.pi / 3 → t.A = Real.pi / 6) ∧
  (t.B ≥ t.A ∧ t.B ≥ t.C →
    ((t.a^2 + t.c^2 + t.a*t.c = t.b^2 ∧ t.b = Real.sqrt 7) →
      t.area = Real.sqrt 3 / 2) ∧
    ((t.a^2 + t.c^2 + t.a*t.c = t.b^2 ∧ t.area = Real.sqrt 3 / 2) →
      t.b = Real.sqrt 7) ∧
    ((t.b = Real.sqrt 7 ∧ t.area = Real.sqrt 3 / 2) →
      t.a^2 + t.c^2 + t.a*t.c = t.b^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l969_96934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_plane_intersection_l969_96984

-- Define a plane in 3D space
structure Plane3D where
  -- You might define a plane using a normal vector and a point, or other suitable representation
  -- This is a placeholder definition
  dummy : Unit

-- Define the concept of intersection between planes
def intersects (p1 p2 : Plane3D) : Prop :=
  -- This is a placeholder definition
  True

-- Define the number of intersection lines between three planes
def numIntersectionLines (p1 p2 p3 : Plane3D) : ℕ :=
  -- This is a placeholder definition
  0

-- Theorem statement
theorem three_plane_intersection (α β γ : Plane3D) 
  (h1 : intersects α β) (h2 : intersects α γ) :
  ∃ n : ℕ, n ∈ ({1, 2, 3} : Set ℕ) ∧ numIntersectionLines α β γ = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_plane_intersection_l969_96984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l969_96981

noncomputable def f (x : ℝ) := 3 * x^2 - 3 * Real.log x

theorem f_decreasing_interval :
  ∀ x ∈ Set.Ioo (0 : ℝ) (Real.sqrt 2 / 2),
  ∀ y ∈ Set.Ioo (0 : ℝ) (Real.sqrt 2 / 2),
  x < y → f x > f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l969_96981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_slope_product_constant_l969_96948

/-- Defines an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Defines a line with slope k and y-intercept b -/
structure Line where
  k : ℝ
  b : ℝ
  h_not_axis : k ≠ 0 ∧ b ≠ 0

/-- Defines a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines membership of a point in an ellipse -/
def Point.mem_ellipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Defines membership of a point on a line -/
def Point.mem_line (p : Point) (l : Line) : Prop :=
  p.y = l.k * p.x + l.b

/-- Theorem: For any ellipse with given properties and any line satisfying given conditions,
    the product of slopes of OM and the line is constant -/
theorem ellipse_line_slope_product_constant
  (C : Ellipse)
  (h_ecc : C.a / Real.sqrt (C.a^2 - C.b^2) = Real.sqrt 2)
  (h_point : C.a^2 * 2 + C.b^2 * 2 = (C.a * C.b)^2)
  (l : Line)
  (A B M : Point)
  (h_intersect : A.mem_ellipse C ∧ B.mem_ellipse C ∧ A.mem_line l ∧ B.mem_line l)
  (h_midpoint : M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2) :
  (M.y / M.x) * l.k = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_slope_product_constant_l969_96948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anne_solo_cleaning_time_l969_96920

/-- Represents the time it takes Anne to clean the house alone -/
def annes_solo_time : ℝ := 12

/-- Represents Bruce's cleaning rate in houses per hour -/
def bruce_rate : ℝ := sorry

/-- Represents Anne's cleaning rate in houses per hour -/
def anne_rate : ℝ := sorry

/-- The house can be cleaned in 4 hours when Bruce and Anne work together -/
axiom combined_time : bruce_rate + anne_rate = 1 / 4

/-- The house can be cleaned in 3 hours when Bruce works at his rate and Anne works at twice her rate -/
axiom doubled_anne_time : bruce_rate + 2 * anne_rate = 1 / 3

/-- Theorem stating that Anne's solo cleaning time is 12 hours -/
theorem anne_solo_cleaning_time : 1 / anne_rate = annes_solo_time := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anne_solo_cleaning_time_l969_96920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l969_96911

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - Real.cos (2 * x + Real.pi / 2)

theorem f_properties :
  (f (Real.pi / 8) = Real.sqrt 2 + 1) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ Real.pi) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Ioo (k * Real.pi + Real.pi / 8) (k * Real.pi + 5 * Real.pi / 8),
    ∀ y ∈ Set.Ioo (k * Real.pi + Real.pi / 8) (k * Real.pi + 5 * Real.pi / 8),
    x < y → f y < f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l969_96911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_jiong_circle_area_l969_96968

-- Define the Jiong function
noncomputable def jiong_function (x : ℝ) : ℝ := 1 / (|x| - 1)

-- Define the Jiong point
def jiong_point : ℝ × ℝ := (0, 1)

-- Define a Jiong circle
def is_jiong_circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  center = jiong_point ∧ 
  ∃ x : ℝ, (x - center.1)^2 + (jiong_function x - center.2)^2 = radius^2

-- Theorem: The minimum area of all Jiong circles is 3π
theorem min_jiong_circle_area :
  ∃ (min_area : ℝ), 
    min_area = 3 * Real.pi ∧
    (∀ (center : ℝ × ℝ) (radius : ℝ), 
      is_jiong_circle center radius → Real.pi * radius^2 ≥ min_area) ∧
    (∃ (center : ℝ × ℝ) (radius : ℝ), 
      is_jiong_circle center radius ∧ Real.pi * radius^2 = min_area) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_jiong_circle_area_l969_96968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_logarithmic_curves_l969_96986

theorem tangent_line_to_logarithmic_curves (k b : ℝ) :
  (∃ x₁ : ℝ, x₁ > 0 ∧ k * x₁ + b = Real.log x₁ + 2 ∧ k = 1 / x₁) ∧
  (∃ x₂ : ℝ, x₂ > -1 ∧ k * x₂ + b = Real.log (x₂ + 1) ∧ k = 1 / (x₂ + 1)) →
  b = 1 - Real.log 2 := by
  intro h
  sorry

#check tangent_line_to_logarithmic_curves

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_logarithmic_curves_l969_96986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_couch_to_table_ratio_l969_96950

variable (chair_price table_price couch_price : ℚ)

axiom table_price_def : table_price = 3 * chair_price
axiom total_cost : chair_price + table_price + couch_price = 380
axiom couch_price_def : couch_price = 300

theorem couch_to_table_ratio : couch_price / table_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_couch_to_table_ratio_l969_96950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_is_twenty_percent_l969_96966

-- Define the problem parameters
noncomputable def purchase_price : ℝ := 56
noncomputable def markup_percentage : ℝ := 0.3
noncomputable def gross_profit : ℝ := 8

-- Define the selling price
noncomputable def selling_price : ℝ := purchase_price / (1 - markup_percentage)

-- Define the discounted price
noncomputable def discounted_price : ℝ := purchase_price + gross_profit

-- Theorem statement
theorem discount_percentage_is_twenty_percent :
  (selling_price - discounted_price) / selling_price = 0.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_is_twenty_percent_l969_96966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_real_roots_triangle_perimeter_l969_96997

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := x^2 - (k+2)*x + 2*k

-- Theorem 1: The equation always has real roots
theorem equation_has_real_roots (k : ℝ) : 
  ∃ x : ℝ, quadratic_equation k x = 0 := by sorry

-- Define a right triangle with hypotenuse 3 and other sides as roots of the equation
def right_triangle (k : ℝ) : 
  {sides : ℝ × ℝ × ℝ // sides.2.1^2 + sides.2.2^2 = sides.1^2 ∧ 
                       sides.1 = 3 ∧ 
                       quadratic_equation k sides.2.1 = 0 ∧ 
                       quadratic_equation k sides.2.2 = 0} :=
  sorry

-- Theorem 2: The perimeter of the triangle is 5 + √5
theorem triangle_perimeter (k : ℝ) :
  let triangle := right_triangle k
  (triangle.val.1 + triangle.val.2.1 + triangle.val.2.2) = 5 + Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_real_roots_triangle_perimeter_l969_96997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_special_product_l969_96900

/-- The product of factors where each factor has twice as many digits as the previous one -/
def specialProduct (n : ℕ) : ℕ :=
  Finset.prod (Finset.range (n + 1)) (fun k => 10^(2^k) - 1)

/-- The sum of digits of a natural number -/
def sumOfDigits : ℕ → ℕ
  | 0 => 0
  | n + 1 => (n + 1) % 10 + sumOfDigits (n / 10)

/-- Theorem stating the sum of digits of the special product -/
theorem sum_of_digits_special_product (n : ℕ) :
  sumOfDigits (specialProduct n) = 9 * (2^(n+1) - 1) := by
  sorry

#check sum_of_digits_special_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_special_product_l969_96900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_distance_to_line_l969_96905

-- Define the hyperbola C₁
def C₁ (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1

-- Define the ellipse C₂
def C₂ (x y : ℝ) : Prop := 4 * x^2 + y^2 = 1

-- Define perpendicularity of two vectors
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

-- Define the distance from a point to a line
noncomputable def distancePointToLine (x₀ y₀ a b c : ℝ) : ℝ :=
  |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)

-- Theorem statement
theorem constant_distance_to_line (x₁ y₁ x₂ y₂ : ℝ) :
  C₁ x₁ y₁ → C₂ x₂ y₂ → perpendicular x₁ y₁ x₂ y₂ →
  ∃ (a b c : ℝ), distancePointToLine 0 0 a b c = Real.sqrt 3 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_distance_to_line_l969_96905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_2a_plus_b_l969_96902

theorem min_value_2a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b = 1) :
  2 * Real.sqrt 2 ≤ 2 * a + b ∧ 
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ * b₀ = 1 ∧ 2 * a₀ + b₀ = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_2a_plus_b_l969_96902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_and_square_property_l969_96907

noncomputable def a : ℕ → ℚ
  | 0 => 1/3  -- Add this case for Nat.zero
  | 1 => 1/3
  | 2 => 1/3
  | (n+3) => let a_n_minus_2 := a (n+1)
              let a_n_minus_1 := a (n+2)
              ((1 - 2*a_n_minus_2) * a_n_minus_1^2) / (2*a_n_minus_1^2 - 4*a_n_minus_2*a_n_minus_1^2 + a_n_minus_2)

noncomputable def general_term (n : ℕ) : ℝ :=
  ((13/5 - 5/2*Real.sqrt 3) * (7 + 4*Real.sqrt 3)^n + 
   (13/3 + 5/2*Real.sqrt 3) * (7 - 4*Real.sqrt 3)^n + 
   7/3)⁻¹

theorem sequence_formula_and_square_property :
  (∀ n : ℕ, n ≥ 1 → a n = general_term n) ∧
  (∀ n : ℕ, n ≥ 1 → ∃ k : ℤ, (a n)⁻¹ - 2 = k^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_and_square_property_l969_96907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_point_l969_96957

-- Define the two curves
noncomputable def curve1 (x : ℝ) : ℝ := x * Real.log x
noncomputable def curve2 (x : ℝ) : ℝ := 4 / x

-- Define the derivatives of the curves
noncomputable def derivative1 (x : ℝ) : ℝ := 1 + Real.log x
noncomputable def derivative2 (x : ℝ) : ℝ := -4 / (x^2)

-- Theorem statement
theorem tangent_perpendicular_point :
  ∃ x : ℝ, x > 0 ∧ 
  (derivative1 1) * (derivative2 x) = -1 ∧ 
  (x = 2 ∨ x = -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_point_l969_96957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l969_96914

theorem max_product_sum (f g h j : ℕ) : 
  f ∈ ({9, 10, 11, 12} : Set ℕ) →
  g ∈ ({9, 10, 11, 12} : Set ℕ) →
  h ∈ ({9, 10, 11, 12} : Set ℕ) →
  j ∈ ({9, 10, 11, 12} : Set ℕ) →
  f ≠ g → f ≠ h → f ≠ j → g ≠ h → g ≠ j → h ≠ j →
  (f * g + g * h + h * j + f * j) ≤ 441 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l969_96914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_sum_17_l969_96962

def is_valid_digit (d : Nat) : Bool := d = 2 ∨ d = 3 ∨ d = 4

def digits_sum_to_17 (n : Nat) : Prop :=
  let digits := n.digits 10
  digits.sum = 17 ∧ digits.all (λ d => is_valid_digit d)

def is_largest_valid_number (n : Nat) : Prop :=
  digits_sum_to_17 n ∧ ∀ m : Nat, digits_sum_to_17 m → m ≤ n

theorem largest_number_with_sum_17 :
  is_largest_valid_number 43333 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_with_sum_17_l969_96962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_log3_implies_exp3_l969_96974

noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

def symmetric_wrt_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

theorem symmetric_log3_implies_exp3 (f : ℝ → ℝ) 
    (h : symmetric_wrt_y_eq_x f (log3 ∘ (λ x => x + 1))) :
  f = λ x => 3^x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_log3_implies_exp3_l969_96974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sets_l969_96987

def is_valid_set (A : Finset ℕ) : Prop :=
  (2015 ∈ A) ∧ 
  (∀ i j, i ∈ A → j ∈ A → i ≠ j → Nat.Prime (Int.natAbs (i - j)))

theorem valid_sets :
  let sets : List (Finset ℕ) := [
    ({2008, 2010, 2013, 2015} : Finset ℕ),
    ({2013, 2015, 2018, 2020} : Finset ℕ),
    ({2010, 2012, 2015, 2017} : Finset ℕ),
    ({2015, 2017, 2020, 2022} : Finset ℕ)
  ]
  ∀ A : Finset ℕ, (A.card ≥ 4 ∧ is_valid_set A) → A ∈ sets := by
  sorry

#check valid_sets

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sets_l969_96987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lucky_license_plates_count_l969_96916

def allowed_letters : Finset Char := {'А', 'В', 'Е', 'К', 'М', 'Н', 'О', 'Р', 'С', 'Т', 'У', 'Х'}
def vowels : Finset Char := {'А', 'Е', 'О', 'У'}
def digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def odd_digits : Finset Nat := {1, 3, 5, 7, 9}
def even_digits : Finset Nat := {0, 2, 4, 6, 8}

structure LicensePlate where
  first_letter : Char
  second_letter : Char
  third_letter : Char
  first_digit : Nat
  second_digit : Nat
  third_digit : Nat

def is_lucky (plate : LicensePlate) : Prop :=
  (plate.second_letter ∈ vowels) ∧ (plate.second_digit ∈ odd_digits) ∧ (plate.third_digit ∈ even_digits)

def is_valid (plate : LicensePlate) : Prop :=
  (plate.first_letter ∈ allowed_letters) ∧
  (plate.second_letter ∈ allowed_letters) ∧
  (plate.third_letter ∈ allowed_letters) ∧
  (plate.first_digit ∈ digits) ∧
  (plate.second_digit ∈ digits) ∧
  (plate.third_digit ∈ digits) ∧
  ¬(plate.first_digit = 0 ∧ plate.second_digit = 0 ∧ plate.third_digit = 0)

-- Assuming finiteness of LicensePlate
instance : Fintype LicensePlate := sorry

-- Assuming decidability of is_lucky and is_valid
instance : DecidablePred is_lucky := sorry
instance : DecidablePred is_valid := sorry

theorem lucky_license_plates_count :
  (Finset.filter (λ plate => is_lucky plate ∧ is_valid plate) (Finset.univ : Finset LicensePlate)).card = 359999 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lucky_license_plates_count_l969_96916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_eq_neg_twenty_l969_96947

def A (k : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![2, -1, 3; 0, 4, -k; 3, -1, 2]

theorem det_A_eq_neg_twenty (k : ℝ) : Matrix.det (A k) = -20 ↔ k = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_A_eq_neg_twenty_l969_96947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_ten_goal_scenario_l969_96925

/-- Represents a player's statements about goal scores -/
structure PlayerStatements where
  self_goals : ℕ
  other_player : String
  other_goals : ℕ

/-- Represents a player in the football match -/
structure Player where
  name : String
  actual_goals : ℕ
  statements : PlayerStatements

/-- Checks if a player's statements satisfy the truth/lie condition -/
def valid_statements (p : Player) (andrey pasha vanya : Player) : Prop :=
  (p.actual_goals = p.statements.self_goals ∧ p.statements.other_goals ≠ (match p.statements.other_player with
    | "Andrey" => andrey.actual_goals
    | "Pasha" => pasha.actual_goals
    | "Vanya" => vanya.actual_goals
    | _ => 0
  )) ∨
  (p.actual_goals ≠ p.statements.self_goals ∧ p.statements.other_goals = (match p.statements.other_player with
    | "Andrey" => andrey.actual_goals
    | "Pasha" => pasha.actual_goals
    | "Vanya" => vanya.actual_goals
    | _ => 0
  ))

/-- The football match scenario -/
def football_match (andrey pasha vanya : Player) : Prop :=
  andrey.name = "Andrey" ∧
  pasha.name = "Pasha" ∧
  vanya.name = "Vanya" ∧
  valid_statements andrey andrey pasha vanya ∧
  valid_statements pasha andrey pasha vanya ∧
  valid_statements vanya andrey pasha vanya ∧
  andrey.actual_goals + pasha.actual_goals + vanya.actual_goals = 10

/-- Theorem: It's impossible for the players to score 10 goals under the given conditions -/
theorem no_valid_ten_goal_scenario :
  ¬∃ (andrey pasha vanya : Player), football_match andrey pasha vanya :=
by
  sorry  -- The proof is omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_ten_goal_scenario_l969_96925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_C_to_line_l_l969_96973

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 4 = 0

-- Define the curve C
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos α, Real.sin α)

-- Define the distance function between a point and line l
noncomputable def distance_to_line_l (x y : ℝ) : ℝ :=
  (abs (x - y + 4)) / Real.sqrt 2

-- Theorem statement
theorem min_distance_curve_C_to_line_l :
  ∃ (d : ℝ), d = Real.sqrt 2 ∧
  ∀ (α : ℝ), distance_to_line_l (curve_C α).1 (curve_C α).2 ≥ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_C_to_line_l_l969_96973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bad_arrangements_count_l969_96956

/-- A circular arrangement of the numbers 1 to 6 -/
def CircularArrangement := Fin 6 → Fin 6

/-- An arrangement is bad if it's not true that for every n from 1 to 20, 
    a subset of consecutive numbers sums to n -/
def is_bad (arr : CircularArrangement) : Prop :=
  ∃ n : Fin 20, ∀ subset : List (Fin 6), 
    (∀ i j, i ∈ subset → j ∈ subset → (i - j) % 6 ≤ 1 ∨ (j - i) % 6 ≤ 1) →
    (subset.map (λ i => (arr i).val + 1)).sum ≠ n.val + 1

/-- Two arrangements are considered the same if they differ by rotation or reflection -/
def same_arrangement (arr1 arr2 : CircularArrangement) : Prop :=
  (∃ k : Fin 6, ∀ i, arr1 i = arr2 ((i + k) % 6)) ∨
  (∃ k : Fin 6, ∀ i, arr1 i = arr2 ((k - i) % 6))

/-- The number of different bad arrangements -/
noncomputable def num_bad_arrangements : ℕ := sorry

theorem bad_arrangements_count : num_bad_arrangements = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bad_arrangements_count_l969_96956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_percent_stock_yield_l969_96938

/-- Calculates the yield of a stock given its dividend rate, par value, and market value. -/
noncomputable def stock_yield (dividend_rate : ℝ) (par_value : ℝ) (market_value : ℝ) : ℝ :=
  (dividend_rate * par_value / market_value) * 100

/-- Theorem stating that a 5% stock with par value $100 and market value $50 has a yield of 10%. -/
theorem five_percent_stock_yield :
  let dividend_rate : ℝ := 0.05
  let par_value : ℝ := 100
  let market_value : ℝ := 50
  stock_yield dividend_rate par_value market_value = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_percent_stock_yield_l969_96938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gnome_falling_theorem_l969_96942

/-- Probability of exactly k gnomes falling -/
noncomputable def prob_k_fall (n k : ℕ) (p : ℝ) : ℝ :=
  p * (1 - p) ^ (n - k)

/-- Expected number of fallen gnomes -/
noncomputable def expected_fallen (n : ℕ) (p : ℝ) : ℝ :=
  n + 1 - 1/p + (1 - p)^(n + 1) / p

/-- Theorem for gnome falling probabilities -/
theorem gnome_falling_theorem (n k : ℕ) (p : ℝ) 
  (h1 : 0 < p) (h2 : p < 1) : 
  (∀ (k : ℕ), k ≤ n → prob_k_fall n k p = p * (1 - p) ^ (n - k)) ∧
  expected_fallen n p = n + 1 - 1/p + (1 - p)^(n + 1) / p :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gnome_falling_theorem_l969_96942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_1002_solutions_l969_96964

/-- Definition of g₁ -/
noncomputable def g₁ (x : ℝ) : ℝ := 3/4 - 4/(4*x + 1)

/-- Recursive definition of gₙ -/
noncomputable def g : ℕ → ℝ → ℝ
| 0, x => x
| 1, x => g₁ x
| (n+2), x => g₁ (g (n+1) x)

/-- Theorem stating the solutions of g₁₀₀₂(x) = x - 4 -/
theorem g_1002_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ = 543/51 ∧ x₂ = 5/32 ∧ 
  g 1002 x₁ = x₁ - 4 ∧ g 1002 x₂ = x₂ - 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_1002_solutions_l969_96964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_pi_fourth_l969_96992

theorem tan_alpha_minus_pi_fourth (α : Real) 
  (h1 : α ∈ Set.Ioo (-π/2) (-π/4))
  (h2 : Real.cos α ^ 2 + Real.cos (3*π/2 + 2*α) = -1/2) : 
  Real.tan (α - π/4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_minus_pi_fourth_l969_96992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_female_count_correct_l969_96960

/-- Calculates the number of female employees to be drawn in a stratified sample -/
def stratified_sample_female_count 
  (total_employees : ℕ) 
  (male_employees : ℕ) 
  (sample_size : ℕ) : ℕ := 
  (total_employees - male_employees) * sample_size / total_employees

theorem stratified_sample_female_count_correct
  (total_employees : ℕ) 
  (male_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 750)
  (h2 : male_employees = 300)
  (h3 : sample_size = 45)
  : stratified_sample_female_count total_employees male_employees sample_size = 27 := by
  sorry

#eval stratified_sample_female_count 750 300 45

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sample_female_count_correct_l969_96960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_f_inv_solutions_l969_96999

/-- The function f(x) = x^2 - 3x - 4 -/
def f (x : ℝ) : ℝ := x^2 - 3*x - 4

/-- f_inv is the inverse function of f -/
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

/-- Theorem stating that f(x) = f^(-1)(x) has solutions x = 2 + 2√2 and x = 2 - 2√2 -/
theorem f_equals_f_inv_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ = 2 + 2 * Real.sqrt 2 ∧ 
                 x₂ = 2 - 2 * Real.sqrt 2 ∧ 
                 f x₁ = f_inv x₁ ∧ 
                 f x₂ = f_inv x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_f_inv_solutions_l969_96999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_part_speed_l969_96926

theorem first_part_speed 
  (total_distance : ℝ) 
  (first_part_distance : ℝ) 
  (second_part_speed : ℝ) 
  (average_speed : ℝ) 
  (h1 : total_distance = 80) 
  (h2 : first_part_distance = 30) 
  (h3 : second_part_speed = 50) 
  (h4 : average_speed = 40) :
  ∃ v : ℝ, v > 0 ∧ (total_distance - first_part_distance) / second_part_speed + first_part_distance / v = total_distance / average_speed ∧ v = 30 := by
  sorry

#eval 30 -- This line is added to check if 30 is recognized as a number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_part_speed_l969_96926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l969_96901

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + Real.sin (7 * Real.pi / 6 - 2 * x) - 1

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_side_value (abc : Triangle) :
  f abc.A = 1/2 ∧ 
  2 * abc.a = abc.b + abc.c ∧ 
  abc.c * abc.a * Real.cos abc.B = -9 →
  abc.a = 3 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l969_96901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_cube_root_relation_l969_96990

/-- A function representing the inverse variation of y with the cube root of x -/
noncomputable def inverse_cube_root_variation (k : ℝ) (x : ℝ) : ℝ := k / (x^(1/3))

/-- Theorem stating the relationship between x and y when y varies inversely as the cube root of x -/
theorem inverse_cube_root_relation (k : ℝ) :
  (inverse_cube_root_variation k 8 = 2) →
  (inverse_cube_root_variation k (1/8) = 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_cube_root_relation_l969_96990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l969_96906

theorem min_value_of_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_perp : m * 1 + 1 * (n - 2) = 0) :
  ∃ (x y : ℝ) (hx : x > 0) (hy : y > 0),
    (∀ a b, a > 0 → b > 0 → a * 1 + 1 * (b - 2) = 0 → 1/a + 2/b ≥ 1/x + 2/y) ∧
    1/x + 2/y = 3/2 + Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l969_96906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l969_96959

theorem power_equation_solution (x : ℝ) : (2 : ℝ)^16^(1/2) = (256 : ℝ)^x → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l969_96959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_less_than_2_5_l969_96924

theorem integers_less_than_2_5 : 
  {x : ℤ | |x| < (5/2 : ℚ)} = {-2, -1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integers_less_than_2_5_l969_96924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l969_96928

-- Define set A
def A : Set ℝ := {x : ℝ | 2 * x^2 - 5 * x - 3 ≤ 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x : ℝ | (x - (2*a + 1)) * ((a - 1) - x) < 0}

-- State that B is nonempty
axiom B_nonempty (a : ℝ) : (B a).Nonempty

-- Theorem for part (I)
theorem part_one : A ∪ B 0 = Set.Ioc (-1) 3 → 0 = 0 := by sorry

-- Theorem for part (II)
theorem part_two (a : ℝ) : A ∩ B a = ∅ → 
  (a ≤ -3/4 ∨ a ≥ 4) ∧ a ≠ -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l969_96928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_l969_96988

/-- The function f(x) = (1/2)ax^2 + (a-1)x - ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 + (a-1) * x - Real.log x

/-- Theorem: For a > 0 and x > 0, f(x) ≥ 2 - (3/(2a)) -/
theorem f_lower_bound (a : ℝ) (x : ℝ) (ha : a > 0) (hx : x > 0) :
  f a x ≥ 2 - (3/(2*a)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_lower_bound_l969_96988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_upper_bound_for_phi_d_ratio_l969_96931

/-- d(n) is the number of positive divisors of n -/
def d (n : ℕ+) : ℕ+ := sorry

/-- φ(n) is Euler's totient function -/
def φ (n : ℕ+) : ℕ+ := sorry

/-- The statement to be proven -/
theorem no_upper_bound_for_phi_d_ratio :
  ∀ C : ℝ, C > 0 → ∃ n : ℕ+, (φ (d n) : ℝ) / (d (φ n) : ℝ) > C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_upper_bound_for_phi_d_ratio_l969_96931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_position_after_150_moves_l969_96910

noncomputable def initial_position : ℂ := 5

noncomputable def rotation_angle : ℝ := Real.pi / 4

noncomputable def translation_distance : ℝ := 10

noncomputable def move (z : ℂ) : ℂ := z * Complex.exp (Complex.I * rotation_angle) + translation_distance

noncomputable def final_position (n : ℕ) : ℂ := (move^[n]) initial_position

theorem particle_position_after_150_moves :
  final_position 150 = Complex.ofReal (-5 * Real.sqrt 2) + Complex.I * Complex.ofReal (5 + 5 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_position_after_150_moves_l969_96910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_tangents_iff_a_gt_e_l969_96913

/-- The function f(x) = x ln x -/
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

/-- The slope of the tangent line to f at point m -/
noncomputable def tangent_slope (m : ℝ) : ℝ := 1 + Real.log m

/-- The condition for the line through (a,a) to be tangent to f at (m, f(m)) -/
def is_tangent (a m : ℝ) : Prop :=
  tangent_slope m = (f m - a) / (m - a)

/-- The main theorem: there are exactly two tangent lines through (a,a) iff a > e -/
theorem two_tangents_iff_a_gt_e (a : ℝ) : 
  (∃ m₁ m₂ : ℝ, m₁ ≠ m₂ ∧ is_tangent a m₁ ∧ is_tangent a m₂ ∧ 
    ∀ m : ℝ, is_tangent a m → m = m₁ ∨ m = m₂) ↔ 
  a > Real.exp 1 := by
  sorry

#check two_tangents_iff_a_gt_e

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_tangents_iff_a_gt_e_l969_96913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l969_96961

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + Real.cos (2*x + Real.pi/3) - 1

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x) ∧
    (∀ q, q > 0 → (∀ x, f (x + q) = f x) → p ≤ q)) ∧
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi → f (5*Real.pi/12 + (5*Real.pi/12 - x)) = f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l969_96961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squared_factor_l969_96963

-- Define the polynomial P
variable (P : ℝ → ℝ → ℝ)

-- Define the symmetry condition
axiom symmetry : ∀ x y, P x y = P y x

-- Define the factor condition
axiom factor : ∀ x y, ∃ Q : ℝ → ℝ → ℝ, P x y = (x - y) * Q x y

-- Theorem to prove
theorem squared_factor : ∀ x y, ∃ R : ℝ → ℝ → ℝ, P x y = (x - y)^2 * R x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_squared_factor_l969_96963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_rearrangement_l969_96923

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

/-- Calculates the area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

theorem isosceles_triangle_rearrangement (t1 t2 : Triangle) :
  t1.a = 13 ∧ t1.b = 13 ∧ t1.c = 10 ∧
  t2.a = 13 ∧ t2.b = 13 ∧ t2.c = 24 ∧
  t1.isIsosceles ∧ t2.isIsosceles →
  ∃ (h1 h2 : ℝ),
    triangleArea t1.c h1 = triangleArea t2.c h2 ∧
    h1 = 12 ∧ h2 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_rearrangement_l969_96923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ploughing_time_l969_96980

/-- Given that two workers A and B can plough a field together in 10 hours,
    and A alone can plough the field in 15 hours, prove that B alone
    would take 30 hours to plough the same field. -/
theorem ploughing_time (time_together time_A : ℝ) (h1 : time_together = 10)
    (h2 : time_A = 15) : 
    (1 / (1 / time_together - 1 / time_A)) = 30 := by
  -- Define rate_A and rate_B
  let rate_A := 1 / time_A
  let rate_B := 1 / time_together - rate_A
  
  -- The main proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ploughing_time_l969_96980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bananas_per_chimp_is_correct_l969_96945

/-- The number of bananas eaten per chimp per day at the Central City Zoo -/
noncomputable def bananas_per_chimp : ℝ :=
  let total_chimps : ℕ := 45
  let total_bananas : ℕ := 72
  (total_bananas : ℝ) / total_chimps

/-- Theorem stating that the number of bananas eaten per chimp per day is 1.6 -/
theorem bananas_per_chimp_is_correct : bananas_per_chimp = 1.6 := by
  -- Unfold the definition of bananas_per_chimp
  unfold bananas_per_chimp
  -- Simplify the expression
  simp
  -- Check that 72 / 45 = 1.6
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bananas_per_chimp_is_correct_l969_96945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_multiplication_for_pattern_l969_96998

-- Define a function to check if a digit is even
def isEven (d : Nat) : Bool :=
  d % 2 = 0

-- Define a function to convert a number to its OE representation
def toOERepresentation (n : Nat) : String :=
  String.mk (List.map (fun c =>
    if isEven (c.toNat - '0'.toNat) then 'E' else 'O') (String.toList (toString n)))

-- Define the theorem
theorem unique_multiplication_for_pattern : ∃! (a b : Nat),
  a ≥ 100 ∧ a < 1000 ∧ b ≥ 10 ∧ b < 100 ∧
  toOERepresentation a = "OEE" ∧
  toOERepresentation b = "E" ∧
  toOERepresentation (a * b) = "EOEE" ∧
  a = 346 ∧ b = 28 :=
by
  sorry

#eval toOERepresentation 346  -- Should output "OEE"
#eval toOERepresentation 28   -- Should output "E"
#eval toOERepresentation 9688 -- Should output "EOEE"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_multiplication_for_pattern_l969_96998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l969_96908

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 - x^2) + 3 / (x - 1)

-- Define the domain of f
def domain_f : Set ℝ := {x | x ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) ∧ x ≠ 1}

-- Theorem stating that the domain of f is correct
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = domain_f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l969_96908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l969_96937

def T (n : ℕ) : Set ℕ := {i | 2 ≤ i ∧ i ≤ n}

def has_sum_triple (S : Set ℕ) : Prop :=
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a + b = c

def valid_partition (n : ℕ) : Prop :=
  ∀ A B : Set ℕ, A ∪ B = T n → A ∩ B = ∅ →
    has_sum_triple A ∨ has_sum_triple B

theorem smallest_valid_n : 
  (∀ n ≥ 4, valid_partition n) ∧
  (∀ n < 4, ¬valid_partition n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l969_96937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_part_speed_is_60_l969_96915

/-- Represents a two-part trip with given distances and speeds -/
structure TwoPart_Trip where
  total_distance : ℝ
  first_part_distance : ℝ
  second_part_distance : ℝ
  second_part_speed : ℝ
  average_speed : ℝ

/-- Calculates the speed of the first part of the trip -/
noncomputable def first_part_speed (trip : TwoPart_Trip) : ℝ :=
  let total_time := trip.total_distance / trip.average_speed
  let second_part_time := trip.second_part_distance / trip.second_part_speed
  trip.first_part_distance / (total_time - second_part_time)

/-- Theorem stating that for the given trip conditions, the first part speed is 60 mph -/
theorem first_part_speed_is_60 (trip : TwoPart_Trip) 
  (h1 : trip.total_distance = 300)
  (h2 : trip.first_part_distance = 180)
  (h3 : trip.second_part_distance = 120)
  (h4 : trip.second_part_speed = 40)
  (h5 : trip.average_speed = 50) :
  first_part_speed trip = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_part_speed_is_60_l969_96915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_area_parallelogram_division_l969_96940

/-- Regular octagon with side length 1 -/
structure RegularOctagon :=
  (sideLength : ℝ)
  (area : ℝ)
  (h_side : sideLength = 1)
  (h_area : area = 2 + 2 * Real.sqrt 2)

/-- Parallelogram with rational area -/
structure Parallelogram :=
  (area : ℚ)

/-- A division of an octagon into parallelograms -/
structure OctagonDivision :=
  (octagon : RegularOctagon)
  (parallelograms : List Parallelogram)
  (h_equal_areas : ∀ p q, p ∈ parallelograms → q ∈ parallelograms → p.area = q.area)
  (h_sum_areas : (parallelograms.map Parallelogram.area).sum = octagon.area)

/-- Theorem: A regular octagon cannot be divided into parallelograms of equal area -/
theorem no_equal_area_parallelogram_division :
  ¬ ∃ (d : OctagonDivision), d.parallelograms ≠ [] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_area_parallelogram_division_l969_96940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_condition_l969_96971

theorem log_inequality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, Real.log a > Real.log b → a > b^3) ∧ 
  ¬(∀ a b, a > b^3 → Real.log a > Real.log b) :=
by
  constructor
  · intro a b h
    sorry
  · push_neg
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_condition_l969_96971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_counts_l969_96939

/-- The set of digits from 0 to 9 -/
def Digits : Finset Nat := Finset.range 10

/-- Count of unique three-digit numbers without repetition -/
def uniqueThreeDigitCount : Nat :=
  Finset.card (Finset.filter (fun n =>
    n ∈ Finset.range 1000 \ Finset.range 100 ∧ 
    (n / 100) ≠ ((n / 10) % 10) ∧ 
    (n / 100) ≠ (n % 10) ∧ 
    ((n / 10) % 10) ≠ (n % 10)) (Finset.range 1000))

/-- Count of unique four-digit even numbers without repetition -/
def uniqueFourDigitEvenCount : Nat :=
  Finset.card (Finset.filter (fun n =>
    n ∈ Finset.range 10000 \ Finset.range 1000 ∧ 
    n % 2 = 0 ∧
    (n / 1000) ≠ ((n / 100) % 10) ∧
    (n / 1000) ≠ ((n / 10) % 10) ∧
    (n / 1000) ≠ (n % 10) ∧
    ((n / 100) % 10) ≠ ((n / 10) % 10) ∧
    ((n / 100) % 10) ≠ (n % 10) ∧
    ((n / 10) % 10) ≠ (n % 10)) (Finset.range 10000))

theorem unique_digit_counts : 
  uniqueThreeDigitCount = 648 ∧ uniqueFourDigitEvenCount = 2296 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_digit_counts_l969_96939
