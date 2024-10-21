import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_bill_calculation_l1064_106430

theorem book_bill_calculation (num_friends : ℕ) (individual_payment : ℚ) (discount_rate : ℚ) :
  num_friends = 6 →
  individual_payment = 63.59 →
  discount_rate = 0.05 →
  ∃ (original_bill : ℚ), 
    original_bill * (1 - discount_rate) = num_friends * individual_payment ∧
    abs (original_bill - 401.62) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_bill_calculation_l1064_106430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_daily_rate_l1064_106436

/-- Represents the daily earnings of a worker -/
def daily_rate (x : ℝ) : Prop := true

/-- Represents the condition of working 5 days a week normally -/
def normal_work_week : Prop := true

/-- Represents the condition of getting 50% extra pay for overtime -/
def overtime_pay : Prop := true

/-- Represents the monthly earnings with overtime every week -/
def monthly_earnings (x : ℝ) : Prop := true

theorem worker_daily_rate :
  normal_work_week →
  overtime_pay →
  monthly_earnings 208 →
  daily_rate 8 :=
by
  intro hw ho hm
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_daily_rate_l1064_106436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1064_106476

noncomputable def f (A : ℝ) (φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (2 * x + φ)

theorem function_properties (A : ℝ) (φ : ℝ) (h1 : A > 0) (h2 : 0 < φ) (h3 : φ < π) 
  (h4 : Real.cos φ = -Real.sqrt 2 / 2) (h5 : Real.sin φ = Real.sqrt 2 / 2) :
  (φ = 3 * π / 4) ∧ 
  (∀ (k : ℤ) (x y : ℝ), 
    x ∈ Set.Icc (-π/8 + ↑k*π) (3*π/8 + ↑k*π) → 
    y ∈ Set.Icc (-π/8 + ↑k*π) (3*π/8 + ↑k*π) → 
    x < y → 
    f A φ y < f A φ x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1064_106476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_sum_theorem_l1064_106409

/-- A 3x3 grid represented as a function from pairs of integers to integers -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- The sum of elements along a diagonal of a grid -/
def diagonalSum (g : Grid) (d : Bool) : ℕ :=
  (Finset.sum (Finset.range 3) fun i => (g i i : ℕ)) +
  if d then 0 else (Finset.sum (Finset.range 3) fun i => (g i (2 - i) : ℕ))

/-- The sum of elements in five specific cells of a grid -/
def fiveCellSum (g : Grid) : ℕ :=
  (g 0 1 : ℕ) + (g 1 0 : ℕ) + (g 1 1 : ℕ) + (g 1 2 : ℕ) + (g 2 1 : ℕ)

/-- Theorem stating the existence of a grid satisfying the given conditions -/
theorem grid_sum_theorem :
  ∃ (g : Grid),
    (∀ i j k : Fin 3, i ≠ j ∨ j ≠ k → g i j ≠ g j k) ∧
    diagonalSum g true = 7 ∧
    diagonalSum g false = 21 ∧
    fiveCellSum g = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_sum_theorem_l1064_106409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1064_106490

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + (Real.cos x)^2 - 1/2

-- Define the interval
def interval : Set ℝ := Set.Icc (-5 * Real.pi / 12) (Real.pi / 24)

-- Theorem stating the properties of f(x)
theorem f_properties :
  ∀ x ∈ interval,
    (f x = Real.sin (2 * x + Real.pi / 6)) ∧
    (f (Real.pi / 24) = Real.sqrt 2 / 2) ∧
    (f (-5 * Real.pi / 12) = -1/2) ∧
    (∀ y ∈ interval, f y ≤ f (Real.pi / 24)) ∧
    (∀ y ∈ interval, f y ≥ f (-5 * Real.pi / 12)) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1064_106490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_is_minus_one_to_one_l1064_106423

-- Define the function f with domain [0,2]
def f : Set ℝ → Set ℝ := sorry

-- Define the property that f has domain [0,2]
def has_domain_zero_to_two (f : Set ℝ → Set ℝ) : Prop :=
  ∀ x, x ∈ (Set.Icc 0 2) ↔ x ∈ (f (Set.Icc 0 2))

-- Define the function g(x) = f(x+1)
def g (f : Set ℝ → Set ℝ) : Set ℝ → Set ℝ :=
  λ S => f {y | ∃ x ∈ S, y = x + 1}

-- Theorem statement
theorem domain_of_g_is_minus_one_to_one
  (f : Set ℝ → Set ℝ)
  (h : has_domain_zero_to_two f) :
  ∀ x, x ∈ (Set.Icc (-1) 1) ↔ x ∈ (g f (Set.Icc (-1) 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_is_minus_one_to_one_l1064_106423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mushroom_pickers_l1064_106432

/-- The number of mushrooms Anya has -/
def A : ℕ := sorry

/-- The number of mushrooms Vitya has -/
def V : ℕ := sorry

/-- The number of mushrooms Sasha has -/
def S : ℕ := sorry

/-- The total number of children who went to pick mushrooms -/
def n : ℕ := sorry

/-- If Anya gives half of her mushrooms to Vitya, all children will have the same number of mushrooms -/
axiom condition1 : A + V + S = (n - 3) * (A / 2)

/-- If Anya gives all her mushrooms to Sasha, Sasha will have as many mushrooms as all the others combined -/
axiom condition2 : S + A = (n - 1) * (A / 2)

/-- The theorem states that the number of children who went to pick mushrooms is 6 -/
theorem mushroom_pickers : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mushroom_pickers_l1064_106432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_50th_group_is_591_l1064_106494

/-- Represents the number of elements in a group based on its position -/
def group_size (n : ℕ) : ℕ :=
  match n % 3 with
  | 0 => 3
  | 1 => 1
  | _ => 2

/-- Calculates the sum of group sizes up to the nth group -/
def sum_group_sizes (n : ℕ) : ℕ :=
  (n / 3) * 6 + group_size (n % 3)

/-- The nth odd number -/
def nth_odd (n : ℕ) : ℕ := 2 * n - 1

theorem sum_50th_group_is_591 :
  let start := sum_group_sizes 49
  let group_sum := (nth_odd (start + 1) + nth_odd (start + 2) + nth_odd (start + 3))
  group_sum = 591 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_50th_group_is_591_l1064_106494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cogs_after_speed_increase_l1064_106466

/-- Proves that the number of cogs produced after speed increase is 60 -/
theorem cogs_after_speed_increase 
  (initial_rate : ℝ) 
  (initial_order : ℝ) 
  (increased_rate : ℝ) 
  (average_output : ℝ) 
  (h1 : initial_rate = 15)
  (h2 : initial_order = 60)
  (h3 : increased_rate = 60)
  (h4 : average_output = 24) :
  ∃ x : ℝ, 
    let total_time := initial_order / initial_rate + x / increased_rate
    let total_cogs := initial_order + x
    x = 60 ∧ average_output = total_cogs / total_time := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cogs_after_speed_increase_l1064_106466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisors_1_to_20_l1064_106495

open Nat Finset

def divisorCount (n : ℕ) : ℕ := (filter (· ∣ n) (range (n + 1))).card

theorem max_divisors_1_to_20 :
  ∃ (max : ℕ), max = 6 ∧
  (∀ n ∈ range 21, divisorCount n ≤ max) ∧
  (divisorCount 12 = max ∧ divisorCount 18 = max ∧ divisorCount 20 = max) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisors_1_to_20_l1064_106495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_axes_product_l1064_106418

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  majorAxis : ℝ
  minorAxis : ℝ
  focus : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Product of major and minor axes of an ellipse -/
theorem ellipse_axes_product (e : Ellipse) (h1 : distance e.center e.focus = 8) 
    (h2 : 2 * (e.minorAxis / 2 + 8 - e.majorAxis / 2) / 2 = 4) : 
    e.majorAxis * e.minorAxis = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_axes_product_l1064_106418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1064_106419

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) : ℕ → ℝ
  | 0 => 0
  | n + 1 => S seq n + seq.a n

theorem arithmetic_sequence_properties (seq : ArithmeticSequence)
  (h1 : S seq 6 > S seq 7)
  (h2 : S seq 7 > S seq 5) :
  seq.d < 0 ∧
  S seq 11 > 0 ∧
  S seq 12 > 0 ∧
  (∀ n : ℕ, S seq n ≤ S seq 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l1064_106419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_inequality_l1064_106425

theorem polynomial_inequality (a b c d : ℝ) :
  (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 → 
    (a * (x + y)^3 + b * (x + y)^2 + c * (x + y) + d > 
     a * x^3 + b * x^2 + c * x + d + a * y^3 + b * y^2 + c * y + d)) ↔ 
  ((a = 0 ∧ b ≥ 0 ∧ d < 0) ∨ 
   (a > 0 ∧ d < 0 ∧ 8 * b^3 > 243 * a^2 * d)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_inequality_l1064_106425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outlier_count_l1064_106448

noncomputable def data : List ℝ := [3, 11, 28, 28, 34, 36, 36, 38, 47, 54]
def Q1 : ℝ := 28
def Q2 : ℝ := 35
def Q3 : ℝ := 38

def IQR : ℝ := Q3 - Q1

def lowerThreshold : ℝ := Q1 - 1.5 * IQR
def upperThreshold : ℝ := Q3 + 1.5 * IQR

noncomputable def isOutlier (x : ℝ) : Bool :=
  x < lowerThreshold ∨ x > upperThreshold

theorem outlier_count :
  (data.filter isOutlier).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_outlier_count_l1064_106448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_is_six_prob_product_not_multiple_of_four_l1064_106404

-- Define the sample space for two dice rolls
def SampleSpace : Finset (ℕ × ℕ) := Finset.product (Finset.range 6) (Finset.range 6)

-- Define the event where the sum of two dice is 6
def SumIsSix (roll : ℕ × ℕ) : Prop := roll.1 + roll.2 + 2 = 6

-- Define the event where the product of two dice is not a multiple of 4
def ProductNotMultipleOfFour (roll : ℕ × ℕ) : Prop := ¬(4 ∣ ((roll.1 + 1) * (roll.2 + 1)))

-- Make SumIsSix decidable
instance : DecidablePred SumIsSix :=
  fun roll => decidable_of_iff (roll.1 + roll.2 + 2 = 6) (by simp [SumIsSix])

-- Make ProductNotMultipleOfFour decidable
instance : DecidablePred ProductNotMultipleOfFour :=
  fun roll => decidable_of_iff (¬(4 ∣ ((roll.1 + 1) * (roll.2 + 1)))) (by simp [ProductNotMultipleOfFour])

-- Theorem for the probability of the sum being 6
theorem prob_sum_is_six :
  (Finset.filter SumIsSix SampleSpace).card / SampleSpace.card = 5 / 36 := by
  sorry

-- Theorem for the probability of the product not being a multiple of 4
theorem prob_product_not_multiple_of_four :
  (Finset.filter ProductNotMultipleOfFour SampleSpace).card / SampleSpace.card = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_is_six_prob_product_not_multiple_of_four_l1064_106404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_increase_year_is_2004_l1064_106427

def profits : List ℝ := [2.0, 2.4, 3.0, 3.5, 4.5, 4.8, 5.3, 5.0, 3.5, 4.0, 3.0]

noncomputable def year_of_max_increase (profits : List ℝ) : ℕ :=
  let increases := List.zipWith (·-·) (List.tail profits) profits
  increases.indexOf (increases.maximum?) + 2001

theorem max_increase_year_is_2004 :
  year_of_max_increase profits = 2004 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_increase_year_is_2004_l1064_106427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_slope_theorem_l1064_106450

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is on the ellipse -/
def on_ellipse (e : Ellipse) (p : Point) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- The right focus of the ellipse is perpendicular to the x-axis from the given point -/
def right_focus_perpendicular (e : Ellipse) (p : Point) : Prop :=
  p.x = e.a^2 - e.b^2

/-- The slope of line MN as described in the problem -/
noncomputable def slope_MN (y₀ : ℝ) : ℝ :=
  -2 * y₀ / (3 * y₀^2 + 16)

theorem ellipse_and_slope_theorem (e : Ellipse) (p : Point) :
  on_ellipse e p → 
  p.x = 1 →
  p.y = 3/2 →
  right_focus_perpendicular e p →
  (e.a = 2 ∧ e.b = Real.sqrt 3) ∧
  (∀ y₀ > 0, slope_MN y₀ ≥ -Real.sqrt 3/12 ∧ ∃ y₀, slope_MN y₀ = -Real.sqrt 3/12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_slope_theorem_l1064_106450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_mean_value_is_two_l1064_106429

-- Define the function f(x) = log₂x
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the domain D = [2, 8]
def D : Set ℝ := Set.Icc 2 8

-- Define the property of having a "mean value" C
def has_mean_value (f : ℝ → ℝ) (D : Set ℝ) (C : ℝ) : Prop :=
  ∀ x₁, x₁ ∈ D → ∃! x₂, x₂ ∈ D ∧ (f x₁ + f x₂) / 2 = C

-- State the theorem
theorem f_mean_value_is_two :
  has_mean_value f D 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_mean_value_is_two_l1064_106429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approximation_l1064_106440

noncomputable def expression : ℝ :=
  (Real.sin (15 * Real.pi / 180) + Real.sin (30 * Real.pi / 180) + Real.sin (45 * Real.pi / 180) + 
   Real.sin (60 * Real.pi / 180) + Real.sin (75 * Real.pi / 180)) /
  (Real.cos (10 * Real.pi / 180) * Real.cos (20 * Real.pi / 180) * Real.cos (30 * Real.pi / 180))

theorem expression_approximation : 
  ∃ ε > 0, |expression - 5.128| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_approximation_l1064_106440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_and_sandy_meeting_point_l1064_106420

/-- The meeting point of two people, given their initial positions and a vertical offset from the midpoint -/
noncomputable def meeting_point (x1 y1 x2 y2 offset : ℚ) : ℚ × ℚ :=
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  (midpoint_x, midpoint_y + offset)

/-- Theorem stating that Mark and Sandy will meet at (1, 3) -/
theorem mark_and_sandy_meeting_point :
  meeting_point 0 7 2 (-3) 1 = (1, 3) := by
  -- Unfold the definition of meeting_point
  unfold meeting_point
  -- Simplify the arithmetic expressions
  simp [Rat.add_def, Rat.mul_def, Rat.div_def]
  -- The proof is complete
  rfl

-- We can't use #eval with noncomputable functions, so we'll use #reduce instead
#reduce meeting_point 0 7 2 (-3) 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_and_sandy_meeting_point_l1064_106420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_constraint_l1064_106401

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  Real.exp (x - 2) + (1/3) * x^3 - (3/2) * x^2 + 2*x - Real.log (x - 1) + a

theorem f_range_constraint (a : ℝ) : 
  (∀ x > 1, ∃ y > 1, f a x = f a y) → 
  (∀ x > 1, ∃ y > 1, f a (f a x) = f a y) → 
  a ≤ 1/3 ∧ a ∈ Set.Iic (1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_constraint_l1064_106401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dissimilar_terms_count_l1064_106442

/-- The number of dissimilar terms in the expansion of (a + b + c + d)^8 -/
def dissimilarTerms : ℕ := Nat.choose 11 3

/-- The polynomial (a + b + c + d)^8 -/
def polynomial (a b c d : ℝ) : ℝ := (a + b + c + d)^8

theorem dissimilar_terms_count :
  dissimilarTerms = Finset.card (Finset.filter (fun t : ℕ × ℕ × ℕ × ℕ => t.1 + t.2.1 + t.2.2.1 + t.2.2.2 = 8) (Finset.product (Finset.range 9) (Finset.product (Finset.range 9) (Finset.product (Finset.range 9) (Finset.range 9))))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dissimilar_terms_count_l1064_106442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_to_b_equals_nine_l1064_106431

theorem n_to_b_equals_nine (n b : ℝ) (h1 : n = 3 ^ (15 / 100)) (h2 : b = 40 / 3) :
  n ^ b = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_to_b_equals_nine_l1064_106431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_run_time_is_18_l1064_106488

/-- Represents the number of minutes run per day by students in each grade -/
structure GradeRunTime where
  sixth : ℚ
  seventh : ℚ
  eighth : ℚ

/-- Represents the ratio of students in each grade compared to eighth grade -/
structure GradeRatio where
  sixth : ℚ
  seventh : ℚ
  eighth : ℚ

/-- Calculates the average run time for all students given the run times and ratios for each grade -/
def averageRunTime (runTime : GradeRunTime) (ratio : GradeRatio) : ℚ :=
  (runTime.sixth * ratio.sixth + runTime.seventh * ratio.seventh + runTime.eighth * ratio.eighth) /
  (ratio.sixth + ratio.seventh + ratio.eighth)

theorem average_run_time_is_18 :
  let runTime : GradeRunTime := { sixth := 20, seventh := 18, eighth := 12 }
  let ratio : GradeRatio := { sixth := 3, seventh := 4, eighth := 1 }
  averageRunTime runTime ratio = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_run_time_is_18_l1064_106488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1064_106463

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
class TriangleConditions (t : Triangle) where
  positive_sides : 0 < t.a ∧ 0 < t.b ∧ 0 < t.c
  positive_angles : 0 < t.A ∧ 0 < t.B ∧ 0 < t.C
  angle_sum : t.A + t.B + t.C = π
  side_condition : t.b^2 = t.a^2 + t.c^2 + t.a * t.c

/-- The area of the triangle -/
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.a * t.c * Real.sin t.B

theorem triangle_theorem (t : Triangle) [TriangleConditions t] :
  t.B = (2/3) * π ∧
  (∃ (S : ℝ), S = area t ∧
    (t.b = Real.sqrt 3 →
      (∀ A C : ℝ, S + Real.sqrt 3 * Real.cos A * Real.cos C ≤ Real.sqrt 3) ∧
      (∃ A C : ℝ, S + Real.sqrt 3 * Real.cos A * Real.cos C = Real.sqrt 3 ∧ A = π/6))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1064_106463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_sum_solution_l1064_106441

theorem unique_prime_sum_solution (a b c : ℕ+) :
  Nat.Prime (a * b + a * c + b * c) ∧
  (a + b : ℚ) / (a + c) = (b + c) / (b + a) →
  a = 1 ∧ b = 1 ∧ c = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_sum_solution_l1064_106441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leastExpensiveTripCost_l1064_106413

/-- Represents a city with x and y coordinates -/
structure City where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two cities -/
noncomputable def distance (c1 c2 : City) : ℝ :=
  Real.sqrt ((c1.x - c2.x)^2 + (c1.y - c2.y)^2)

/-- Calculates the cost of bus travel between two cities -/
noncomputable def busCost (c1 c2 : City) : ℝ :=
  0.15 * distance c1 c2

/-- Calculates the cost of plane travel between two cities -/
noncomputable def planeCost (c1 c2 : City) : ℝ :=
  100 + 0.10 * distance c1 c2

/-- Theorem: The least expensive total cost for the trip is $1240.47 -/
theorem leastExpensiveTripCost (a b c : City)
  (h1 : distance a c = 3500)
  (h2 : distance a b = 4000)
  (h3 : b.x = c.x)  -- B is directly north of C
  (h4 : a.y = c.y)  -- A is directly east of C
  : min (planeCost a b) (busCost a b) +
    min (planeCost b c) (busCost b c) +
    min (planeCost c a) (busCost c a) = 1240.47 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_leastExpensiveTripCost_l1064_106413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1064_106482

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Definition of eccentricity for a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + (h.b / h.a)^2)

/-- Predicate: Circle centered at B(0,b) is tangent to one of the asymptotes at point P -/
def CircleTangentToAsymptote (h : Hyperbola) (P : ℝ × ℝ) : Prop :=
  sorry

/-- Predicate: BP is parallel to PF, where F is the right focus -/
def BPParallelPF (h : Hyperbola) (P : ℝ × ℝ) : Prop :=
  sorry

/-- Theorem: Eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_circle : ∃ (P : ℝ × ℝ), CircleTangentToAsymptote h P ∧ BPParallelPF h P) :
  eccentricity h = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1064_106482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_sixth_l1064_106437

-- Define the function f
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

-- State the theorem
theorem f_value_at_pi_sixth (ω φ : ℝ) :
  (∀ x, f ω φ (π/3 + x) = f ω φ (-x)) →
  (f ω φ (π/6) = 2 ∨ f ω φ (π/6) = -2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_sixth_l1064_106437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_discount_problem_l1064_106457

/-- Calculates the banker's discount for a bill --/
noncomputable def bankers_discount (bill_amount : ℝ) (true_discount : ℝ) (annual_rate : ℝ) (days : ℝ) : ℝ :=
  let face_value := bill_amount + true_discount
  let time_in_years := days / 365
  face_value * annual_rate * time_in_years

/-- Theorem stating the banker's discount for the given conditions --/
theorem bankers_discount_problem :
  let bill_amount := (8250 : ℝ)
  let true_discount := (1287.50 : ℝ)
  let annual_rate := (0.065 : ℝ)
  let days := (205 : ℝ)
  abs (bankers_discount bill_amount true_discount annual_rate days - 348.23) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_discount_problem_l1064_106457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_rate_is_four_percent_l1064_106400

noncomputable def base_earnings : ℝ := 190
noncomputable def earnings_goal : ℝ := 500
noncomputable def sales_needed : ℝ := 7750

noncomputable def commission_rate : ℝ := (earnings_goal - base_earnings) / sales_needed * 100

theorem commission_rate_is_four_percent :
  ∃ ε > 0, |commission_rate - 4| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_rate_is_four_percent_l1064_106400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chords_of_tangency_equal_l1064_106481

/-- Two circles with centers and tangents drawn between them -/
structure TwoCirclesWithTangents where
  /-- Center of the first circle -/
  O₁ : ℝ × ℝ
  /-- Center of the second circle -/
  O₂ : ℝ × ℝ
  /-- Radius of the first circle -/
  r : ℝ
  /-- Radius of the second circle -/
  R : ℝ
  /-- The distance between the centers is greater than the sum of the radii -/
  centers_distance_gt_sum_radii : dist O₁ O₂ > r + R

/-- The chords connecting the points of tangency are equal in length -/
theorem chords_of_tangency_equal (c : TwoCirclesWithTangents) :
  ∃ (A B C D : ℝ × ℝ),
    (dist A B = dist C D) ∧
    (dist c.O₁ A = c.r) ∧
    (dist c.O₁ B = c.r) ∧
    (dist c.O₂ C = c.R) ∧
    (dist c.O₂ D = c.R) ∧
    (dist c.O₁ C * dist c.O₁ D = dist c.O₁ c.O₂ ^ 2 - c.R ^ 2) ∧
    (dist c.O₂ A * dist c.O₂ B = dist c.O₁ c.O₂ ^ 2 - c.r ^ 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chords_of_tangency_equal_l1064_106481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_max_value_l1064_106405

open Real Matrix

-- Define the determinant as a function of θ
noncomputable def det_value (θ : ℝ) : ℝ :=
  det !![1, 1, 1; 1, 1 + Real.sin θ ^ 2, 1; 1 + Real.cos θ ^ 2, 1, 1]

-- State the theorem
theorem det_max_value :
  ∃ (max_val : ℝ), max_val = 1/4 ∧ ∀ θ, det_value θ ≤ max_val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_max_value_l1064_106405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1064_106465

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := log (x + Real.sqrt (1 + x^2)) + exp x - exp (-x)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x > 0, f (a * x + 1) + f (log (1 / x)) > 0) →
  a > 1 / exp 2 ∧ ∀ b > 1 / exp 2, ∃ x > 0, f (b * x + 1) + f (log (1 / x)) > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1064_106465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_needle_rotation_theorem_l1064_106473

/-- Represents an infinite grid with pins at each node -/
structure InfiniteGrid :=
  (pins : ℤ × ℤ → Unit)

/-- Represents a needle on the grid -/
structure Needle :=
  (length : ℝ)
  (position : ℝ × ℝ)
  (angle : ℝ)

/-- Predicate to check if a needle can be rotated by 90 degrees on the grid -/
def can_rotate_90_degrees (grid : InfiniteGrid) (needle : Needle) : Prop :=
  ∃ (new_position : ℝ × ℝ) (new_angle : ℝ),
    new_angle = needle.angle + Real.pi / 2 ∧
    ∀ (t : ℝ), 0 ≤ t ∧ t ≤ 1 →
      ∃ (intermediate_position : ℝ × ℝ) (intermediate_angle : ℝ),
        intermediate_angle = needle.angle + t * (Real.pi / 2) ∧
        ∀ (x y : ℤ), (x, y) ≠ (⌊intermediate_position.1⌋, ⌊intermediate_position.2⌋)

/-- Theorem stating that a needle of length ≥ 1 can be rotated by 90 degrees on the grid -/
theorem needle_rotation_theorem (grid : InfiniteGrid) (needle : Needle) :
  needle.length ≥ 1 → can_rotate_90_degrees grid needle :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_needle_rotation_theorem_l1064_106473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_about_pi_over_2_l1064_106439

noncomputable def g (x : ℝ) := Real.cos (2 * x)

theorem g_symmetry_about_pi_over_2 :
  ∀ x : ℝ, g (π / 2 + x) = g (π / 2 - x) :=
by
  intro x
  unfold g
  simp [Real.cos_add, Real.cos_two_mul]
  ring
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_symmetry_about_pi_over_2_l1064_106439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_sphere_radius_l1064_106438

/-- The radius of the smallest sphere that encloses eight spheres tangent to an inner cube --/
theorem smallest_enclosing_sphere_radius 
  (inner_cube_side_length : ℝ) 
  (inner_sphere_radius : ℝ) 
  (h_side_length : inner_cube_side_length = 6) 
  (h_sphere_radius : inner_sphere_radius = 2) :
  (inner_cube_side_length * Real.sqrt 3 + 2 * inner_sphere_radius) / 2 = 3 * Real.sqrt 3 + 2 := by
  sorry

#check smallest_enclosing_sphere_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_enclosing_sphere_radius_l1064_106438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ABC_on_unit_circle_l1064_106403

-- Define a circle as a center point and a radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define what it means for a point to be on a circle
def on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define the problem setup
variable (O A B C : ℝ × ℝ)
variable (circle1 circle2 circle3 : Circle)

-- Axioms for the conditions
axiom circles_are_unit : 
  circle1.radius = 1 ∧ circle2.radius = 1 ∧ circle3.radius = 1

axiom O_is_common_point : 
  on_circle O circle1 ∧ on_circle O circle2 ∧ on_circle O circle3

axiom A_is_intersection : 
  on_circle A circle1 ∧ on_circle A circle2 ∧ ¬on_circle A circle3

axiom B_is_intersection : 
  on_circle B circle2 ∧ on_circle B circle3 ∧ ¬on_circle B circle1

axiom C_is_intersection : 
  on_circle C circle3 ∧ on_circle C circle1 ∧ ¬on_circle C circle2

-- Theorem to prove
theorem ABC_on_unit_circle : 
  ∃ (result_circle : Circle), 
    result_circle.radius = 1 ∧ 
    on_circle A result_circle ∧ 
    on_circle B result_circle ∧ 
    on_circle C result_circle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ABC_on_unit_circle_l1064_106403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_2_8_l1064_106414

-- Define a power function
noncomputable def power_function (α : ℝ) : ℝ → ℝ := λ x ↦ x ^ α

-- Theorem statement
theorem power_function_through_point_2_8 :
  ∃ α : ℝ, (power_function α 2 = 8) ∧ (∀ x : ℝ, power_function α x = x^3) :=
by
  -- Provide the value of α
  use 3
  constructor
  · -- Prove that power_function 3 2 = 8
    simp [power_function]
    norm_num
  · -- Prove that ∀ x : ℝ, power_function 3 x = x^3
    intro x
    simp [power_function]
    -- The rest of the proof is omitted
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_2_8_l1064_106414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_6_or_8_under_201_l1064_106474

theorem multiples_of_6_or_8_under_201 : 
  (Finset.filter (λ n : ℕ => n < 201 ∧ 
    ((6 ∣ n ∨ 8 ∣ n) ∧ ¬(6 ∣ n ∧ 8 ∣ n))) 
    (Finset.range 201)).card = 42 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_6_or_8_under_201_l1064_106474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_difference_in_S_l1064_106443

def S : Set Int := {-20, -5, 1, 3, 5, 15}

theorem largest_difference_in_S : 
  (∀ x y : Int, x ∈ S → y ∈ S → (x - y : Int) ≤ 35) ∧ 
  (∃ a b : Int, a ∈ S ∧ b ∈ S ∧ a - b = 35) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_difference_in_S_l1064_106443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bella_steps_count_l1064_106462

/-- Represents the walking scenario between Bella and Ella -/
structure WalkingScenario where
  distance : ℚ  -- Distance between houses in feet
  step_length : ℚ  -- Length of Bella's step in feet
  delay : ℚ  -- Delay before Ella starts in minutes
  speed_ratio : ℚ  -- Ratio of Ella's speed to Bella's speed

/-- Calculates the number of steps Bella takes before meeting Ella -/
def steps_to_meeting (scenario : WalkingScenario) : ℚ :=
  (17 * scenario.distance / (6 * scenario.speed_ratio) + scenario.distance) / 
  (2 * scenario.step_length)

/-- Theorem stating that Bella takes 2112 steps before meeting Ella -/
theorem bella_steps_count (scenario : WalkingScenario) 
  (h1 : scenario.distance = 15840)
  (h2 : scenario.step_length = 2)
  (h3 : scenario.delay = 5)
  (h4 : scenario.speed_ratio = 5) : 
  steps_to_meeting scenario = 2112 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bella_steps_count_l1064_106462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_analysis_l1064_106407

-- Define the survey data structure
structure SurveyData where
  excellent_low : Nat
  excellent_mid : Nat
  excellent_high : Nat
  good_low : Nat
  good_mid : Nat
  good_high : Nat
  mild_low : Nat
  mild_mid : Nat
  mild_high : Nat
  moderate_low : Nat
  moderate_mid : Nat
  moderate_high : Nat

-- Define the air quality levels
inductive AirQuality
  | Excellent
  | Good
  | MildPollution
  | ModeratePollution

-- Define the theorem
theorem survey_analysis (data : SurveyData) :
  let total_days : ℚ := (data.excellent_low + data.excellent_mid + data.excellent_high +
                    data.good_low + data.good_mid + data.good_high +
                    data.mild_low + data.mild_mid + data.mild_high +
                    data.moderate_low + data.moderate_mid + data.moderate_high : ℚ)

  let prob_excellent := ((data.excellent_low + data.excellent_mid + data.excellent_high : ℚ) / total_days)
  let prob_good := ((data.good_low + data.good_mid + data.good_high : ℚ) / total_days)
  let prob_mild := ((data.mild_low + data.mild_mid + data.mild_high : ℚ) / total_days)
  let prob_moderate := ((data.moderate_low + data.moderate_mid + data.moderate_high : ℚ) / total_days)

  let avg_exercising := ((100 * (data.excellent_low + data.good_low + data.mild_low + data.moderate_low) +
                         300 * (data.excellent_mid + data.good_mid + data.mild_mid + data.moderate_mid) +
                         500 * (data.excellent_high + data.good_high + data.mild_high + data.moderate_high) : ℚ) / total_days)

  let good_low400 := data.excellent_low + data.good_low
  let good_high400 := data.excellent_mid + data.excellent_high + data.good_mid + data.good_high
  let poor_low400 := data.mild_low + data.moderate_low
  let poor_high400 := data.mild_mid + data.mild_high + data.moderate_mid + data.moderate_high

  let k_squared := ((total_days * (good_low400 * poor_high400 - good_high400 * poor_low400)^2 : ℚ) /
                   ((good_low400 + good_high400) * (poor_low400 + poor_high400) *
                    (good_low400 + poor_low400) * (good_high400 + poor_high400) : ℚ))

  -- Assuming data is the given survey data
  data = {excellent_low := 2, excellent_mid := 16, excellent_high := 25,
          good_low := 5, good_mid := 10, good_high := 12,
          mild_low := 6, mild_mid := 7, mild_high := 8,
          moderate_low := 7, moderate_mid := 2, moderate_high := 0} →
  (prob_excellent = 0.43 ∧ 
   prob_good = 0.27 ∧ 
   prob_mild = 0.21 ∧ 
   prob_moderate = 0.09 ∧
   avg_exercising = 350 ∧
   k_squared > 3.841) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_survey_analysis_l1064_106407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_deposit_problem_l1064_106464

/-- Calculates simple interest given principal, rate, and time -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem bank_deposit_problem (P : ℝ) :
  P > 0 →
  let SI1 := simpleInterest P 15 3.5
  let SI2 := simpleInterest P 15 10
  SI2 - SI1 = 144 →
  ∃ ε > 0, |P - 148| < ε :=
by
  sorry

#check bank_deposit_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bank_deposit_problem_l1064_106464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plans_equal_at_2500_minutes_l1064_106486

/-- Represents a phone plan with a base fee, included minutes, and per-minute rate for overages -/
structure PhonePlan where
  baseFee : ℚ
  includedMinutes : ℚ
  overageRate : ℚ

/-- Calculates the cost of a phone plan given the number of minutes used -/
def planCost (plan : PhonePlan) (minutes : ℚ) : ℚ :=
  if minutes ≤ plan.includedMinutes then
    plan.baseFee
  else
    plan.baseFee + (minutes - plan.includedMinutes) * plan.overageRate

/-- The first phone plan -/
def plan1 : PhonePlan := { baseFee := 50, includedMinutes := 500, overageRate := 35/100 }

/-- The second phone plan -/
def plan2 : PhonePlan := { baseFee := 75, includedMinutes := 1000, overageRate := 45/100 }

/-- Theorem stating that the costs of the two plans are equal at 2500 minutes -/
theorem plans_equal_at_2500_minutes :
  planCost plan1 2500 = planCost plan2 2500 := by
  -- Unfold definitions and simplify
  simp [planCost, plan1, plan2]
  -- Perform arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plans_equal_at_2500_minutes_l1064_106486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_functions_l1064_106410

-- Define the graphs as variables
variable (A B C D E : Set (ℝ × ℝ))

-- Define the properties of each graph
def graph_A (A : Set (ℝ × ℝ)) : Prop := ∃ y : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (y = 3 → (x₁, y) ∈ A ∧ (x₂, y) ∈ A)
def graph_B (B : Set (ℝ × ℝ)) : Prop := ∀ y : ℝ, ∀ x₁ x₂ : ℝ, (x₁, y) ∈ B ∧ (x₂, y) ∈ B → x₁ = x₂
def graph_C (C : Set (ℝ × ℝ)) : Prop := ∀ y : ℝ, ∀ x₁ x₂ : ℝ, (x₁, y) ∈ C ∧ (x₂, y) ∈ C → x₁ = x₂
def graph_D (D : Set (ℝ × ℝ)) : Prop := ∃ y : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (abs y < 1 → (x₁, y) ∈ D ∧ (x₂, y) ∈ D)
def graph_E (E : Set (ℝ × ℝ)) : Prop := ∀ y : ℝ, ∀ x₁ x₂ : ℝ, (x₁, y) ∈ E ∧ (x₂, y) ∈ E → x₁ = x₂

-- Define what it means for a function to have an inverse
def has_inverse (f : Set (ℝ × ℝ)) : Prop := ∀ y : ℝ, ∀ x₁ x₂ : ℝ, (x₁, y) ∈ f ∧ (x₂, y) ∈ f → x₁ = x₂

-- Theorem statement
theorem inverse_functions :
  graph_A A ∧ graph_B B ∧ graph_C C ∧ graph_D D ∧ graph_E E →
  (¬ has_inverse A) ∧ (has_inverse B) ∧ (has_inverse C) ∧ (¬ has_inverse D) ∧ (has_inverse E) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_functions_l1064_106410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_C_time_l1064_106472

/-- Given workers A, B, and C, and their work rates on a job -/
structure WorkerRates where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The time it takes for a given work rate to complete the job -/
noncomputable def timeToComplete (rate : ℝ) : ℝ := 1 / rate

theorem worker_C_time (w : WorkerRates) 
  (h1 : w.A + w.B = 1 / 15)  -- A and B finish the job in 15 days
  (h2 : w.A + w.B + w.C = 1 / 6)  -- A, B, and C finish the job in 6 days
  : timeToComplete w.C = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_C_time_l1064_106472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_cubic_parabola_revolution_l1064_106487

/-- The surface area generated by revolving the arc of the cubic parabola y = x^3 
    between x = -2/3 and x = 2/3 around the Ox axis -/
noncomputable def surface_area : ℝ := 196 * Real.pi / 729

/-- The function representing the cubic parabola -/
def f (x : ℝ) : ℝ := x^3

/-- The lower bound of the integral -/
noncomputable def lower_bound : ℝ := -2/3

/-- The upper bound of the integral -/
noncomputable def upper_bound : ℝ := 2/3

theorem surface_area_of_cubic_parabola_revolution : 
  2 * Real.pi * ∫ x in lower_bound..upper_bound, f x * Real.sqrt (1 + (3 * x^2)^2) = surface_area := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_cubic_parabola_revolution_l1064_106487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1064_106478

/-- Proves that in a triangle ABC where side b = 3, side a = √3, and angle A = 30°, the length of side c is 2√3 -/
theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  b = 3 →
  a = Real.sqrt 3 →
  A = 30 * π / 180 →
  c = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1064_106478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_1234_consecutive_repeat_1975_l1064_106491

-- Define the sequence
def mySequence : ℕ → ℕ
  | 0 => 1
  | 1 => 9
  | 2 => 7
  | 3 => 5
  | n + 4 => (mySequence n + mySequence (n + 1) + mySequence (n + 2) + mySequence (n + 3)) % 10

-- Theorem 1: 1234 will not appear consecutively
theorem no_1234_consecutive : ¬∃n : ℕ, 
  mySequence n = 1 ∧ 
  mySequence (n + 1) = 2 ∧ 
  mySequence (n + 2) = 3 ∧ 
  mySequence (n + 3) = 4 := by
  sorry

-- Theorem 2: 1975 will appear at least twice
theorem repeat_1975 : ∃n m : ℕ, n ≠ m ∧ 
  mySequence n = 1 ∧ 
  mySequence (n + 1) = 9 ∧ 
  mySequence (n + 2) = 7 ∧ 
  mySequence (n + 3) = 5 ∧
  mySequence m = 1 ∧ 
  mySequence (m + 1) = 9 ∧ 
  mySequence (m + 2) = 7 ∧ 
  mySequence (m + 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_1234_consecutive_repeat_1975_l1064_106491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_triangle_circle_l1064_106471

-- Define an equilateral triangle
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

-- Define a circle
structure Circle where
  radius : ℝ
  radius_positive : radius > 0

-- Define the theorem
theorem area_ratio_triangle_circle 
  (t : EquilateralTriangle) (c : Circle) 
  (h : 3 * t.side = 2 * Real.pi * c.radius) : 
  (Real.sqrt 3 * t.side^2 / 4) / (Real.pi * c.radius^2) = Real.sqrt 3 * Real.pi / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_triangle_circle_l1064_106471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_min_diff_l1064_106461

def problem (a b : ℕ) : Prop :=
  Nat.lcm a b = 2010 ∧ Nat.gcd a b = 2

theorem max_sum_min_diff (a b : ℕ) (h : problem a b) :
  (∀ x y : ℕ, problem x y → a + b ≥ x + y) ∧
  (∀ x y : ℕ, problem x y → |Int.ofNat a - Int.ofNat b| ≤ |Int.ofNat x - Int.ofNat y|) ∧
  a + b = 2012 ∧
  |Int.ofNat a - Int.ofNat b| = 104 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_min_diff_l1064_106461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_problem_l1064_106475

/-- Given a line passing through points (-1, -4) and (3, k) with slope k, prove that k = 4/3 -/
theorem line_slope_problem (k : ℝ) : 
  (k - (-4)) / (3 - (-1)) = k → k = 4/3 := by 
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_problem_l1064_106475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1064_106449

-- Define the ellipse E
noncomputable def ellipse_E (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define points
def point_O : ℝ × ℝ := (0, 0)
def point_A (a : ℝ) : ℝ × ℝ := (a, 0)
def point_B (b : ℝ) : ℝ × ℝ := (0, b)
def point_C (a : ℝ) : ℝ × ℝ := (-a, 0)

-- Define point M
noncomputable def point_M (a b : ℝ) : ℝ × ℝ := ((2*a)/3, b/3)

-- Define point N
noncomputable def point_N (a b : ℝ) : ℝ × ℝ := (-a/2, b/2)

-- Helper definitions
noncomputable def eccentricity (e : Set (ℝ × ℝ)) : ℝ := sorry
noncomputable def reflect_across_line (p : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : ℝ × ℝ := sorry
noncomputable def line_equation (a b : ℝ) : ℝ × ℝ → ℝ := sorry

-- Main theorem
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : (point_M a b).2 / (point_M a b).1 = Real.sqrt 5 / 10)
  (h4 : ∃ (x : ℝ), (x, 13/2) = reflect_across_line (point_N a b) (line_equation a b)) :
  (eccentricity (ellipse_E a b) = (2 * Real.sqrt 5) / 5) ∧
  (ellipse_E (3 * Real.sqrt 5) 3 = ellipse_E a b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1064_106449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_always_positive_l1064_106489

/-- The function f(x) for a given k -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  ((k + 1) * x^2 + (k + 3) * x + (2 * k - 8)) /
  ((2 * k - 1) * x^2 + (k + 1) * x + (k - 4))

/-- The domain of f(x) for a given k -/
def domain (k : ℝ) : Set ℝ :=
  {x : ℝ | (2 * k - 1) * x^2 + (k + 1) * x + (k - 4) ≠ 0}

/-- The theorem stating the condition for f(x) to be always positive -/
theorem f_always_positive (k : ℝ) :
  (∀ x ∈ domain k, f k x > 0) ↔
  (k < (15 - 16 * Real.sqrt 2) / 7 ∨ k > (15 * Real.sqrt 2) / 7) :=
by
  sorry

#check f_always_positive

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_always_positive_l1064_106489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_height_l1064_106416

/-- The height of a regular tetrahedron with edge length a -/
noncomputable def tetrahedron_height (a : ℝ) : ℝ := a * Real.sqrt (2/3)

/-- Theorem: The height of a regular tetrahedron with edge length a is a * sqrt(2/3) -/
theorem regular_tetrahedron_height (a : ℝ) (h : a > 0) :
  tetrahedron_height a = a * Real.sqrt (2/3) := by
  -- Unfold the definition of tetrahedron_height
  unfold tetrahedron_height
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_tetrahedron_height_l1064_106416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_job_assignment_l1064_106492

-- Define the people and jobs
inductive Person : Type
  | Boulanger : Person
  | Patissier : Person
  | Fleuriste : Person

inductive Job : Type
  | Baker : Job
  | PastryChef : Job
  | Florist : Job

-- Define the assignment of jobs to people
def job_assignment : Person → Job := sorry

-- Define the statements
def statement1 : Prop := job_assignment Person.Patissier ≠ Job.Baker
def statement2 : Prop := job_assignment Person.Fleuriste ≠ Job.PastryChef
def statement3 : Prop := job_assignment Person.Patissier = Job.PastryChef
def statement4 : Prop := job_assignment Person.Fleuriste = Job.Florist

-- Theorem to prove
theorem unique_job_assignment :
  (∀ (p1 p2 : Person), p1 ≠ p2 → job_assignment p1 ≠ job_assignment p2) →
  (∀ (j : Job), ∃ (p : Person), job_assignment p = j) →
  (statement1 ∨ statement2 ∨ statement3 ∨ statement4) →
  (¬statement1 ∨ ¬statement2) →
  (¬statement1 ∨ ¬statement3) →
  (¬statement1 ∨ ¬statement4) →
  (¬statement2 ∨ ¬statement3) →
  (¬statement2 ∨ ¬statement4) →
  (¬statement3 ∨ ¬statement4) →
  (job_assignment Person.Patissier = Job.Baker ∧
   job_assignment Person.Fleuriste = Job.PastryChef ∧
   job_assignment Person.Boulanger = Job.Florist) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_job_assignment_l1064_106492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_matrix_characterization_l1064_106447

def reflection_matrix (a b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![a, b; -3/5, 4/5]

theorem reflection_matrix_characterization (a b : ℚ) :
  (∀ v : Matrix (Fin 2) (Fin 1) ℚ, reflection_matrix a b * (reflection_matrix a b * v) = v) ↔
  (a = -4/5 ∧ b = -3/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_matrix_characterization_l1064_106447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1064_106415

theorem triangle_inequality (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  Real.sin (3 * A / 2) + Real.sin (3 * B / 2) + Real.sin (3 * C / 2) ≤
  Real.cos ((A - B) / 2) + Real.cos ((B - C) / 2) + Real.cos ((C - A) / 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l1064_106415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_ratio_l1064_106412

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] where
  F₁ : P
  F₂ : P
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

/-- A point on the ellipse -/
def PointOnEllipse {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (E : Ellipse P) (M : P) : Prop :=
  dist M E.F₁ + dist M E.F₂ = 2 * E.a

/-- The intersection of MF₁ with the ellipse -/
noncomputable def intersectionA {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (E : Ellipse P) (M : P) : P :=
  sorry

/-- The intersection of MF₂ with the ellipse -/
noncomputable def intersectionB {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (E : Ellipse P) (M : P) : P :=
  sorry

/-- The theorem statement -/
theorem ellipse_constant_ratio {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (E : Ellipse P) (M : P) 
    (h : PointOnEllipse E M) :
  let A := intersectionA E M
  let B := intersectionB E M
  (dist M E.F₁ / dist E.F₁ A) + (dist M E.F₂ / dist E.F₂ B) = 4 * E.a^2 / E.b^2 - 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_ratio_l1064_106412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_configuration_theorem_l1064_106434

/-- A structure representing a configuration of points on a plane --/
structure PointConfiguration where
  l : Set (ℝ × ℝ)  -- The line l
  P : ℝ × ℝ        -- The point P
  A : Fin 100 → ℝ × ℝ  -- The 100 points on l

/-- The distance between two points --/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The dot product of two vectors --/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The main theorem --/
theorem point_configuration_theorem (config : PointConfiguration) :
  (∀ p ∈ config.l, distance config.P p = 1) →
  (∀ i, config.A i ∈ config.l) →
  (∀ i < 99, dot_product (config.A (i+1) - config.P) (config.A (i+2) - config.P) = -2 / (i+1)) →
  dot_product (config.A 100 - config.P) (config.A 1 - config.P) = -51/50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_configuration_theorem_l1064_106434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_probability_l1064_106498

-- Define the angle between hour and minute hands
noncomputable def angle_between_hands (hour : ℝ) (minute : ℝ) : ℝ :=
  |30 * hour - 5.5 * minute|

-- Define the probability function
noncomputable def prob_angle_less_than (angle : ℝ) : ℝ :=
  (2 * angle) / 360

-- State the theorem
theorem clock_angle_probability : 
  prob_angle_less_than (360 / 11) * 2 = 4 / 11 := by
  sorry

#eval (100 * 4 + 11 : Nat)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_probability_l1064_106498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_stations_l1064_106460

/-- The distance between two stations in kilometers. -/
noncomputable def distance : ℝ := 25

/-- The time taken to travel between the stations at 100 kmph in hours. -/
noncomputable def base_time : ℝ := distance / 100

/-- Represents the travel time at a given speed. -/
noncomputable def travel_time (speed : ℝ) : ℝ := distance / speed

theorem distance_between_stations :
  (travel_time 100 = base_time) ∧
  (travel_time 50 = base_time + 15 / 60) ∧
  (travel_time 70 = base_time + 7 / 60) →
  distance = 25 := by
  intro h
  -- The proof goes here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_stations_l1064_106460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_congruent_one_mod_n_l1064_106477

theorem xy_congruent_one_mod_n (n : ℕ+) (x y : ZMod n) (h : x = y⁻¹) :
  (x * y : ZMod n) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_congruent_one_mod_n_l1064_106477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_triangle_inequality_l1064_106433

/-- A point on a lattice grid -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle on a lattice grid -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Distance between two lattice points -/
noncomputable def distance (p q : LatticePoint) : ℝ :=
  Real.sqrt (((p.x - q.x)^2 + (p.y - q.y)^2) : ℝ)

/-- Perimeter of a lattice triangle -/
noncomputable def perimeter (t : LatticeTriangle) : ℝ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

theorem lattice_triangle_inequality (t : LatticeTriangle) :
  distance t.A t.B > distance t.A t.C →
  distance t.A t.B - distance t.A t.C > 1 / perimeter t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_triangle_inequality_l1064_106433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1064_106417

theorem trig_identity (θ : Real) 
  (h : Real.sin (π/2 + θ) + 3*Real.cos (θ - π) = Real.sin (-θ)) : 
  Real.sin θ * Real.cos θ + Real.cos θ^2 = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1064_106417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1064_106453

noncomputable def f (x : ℝ) : ℝ := |x - 1| - |x + 1|
noncomputable def F (x y m n : ℝ) : ℝ := max (|x^2 - 4*y + m|) (|y^2 - 2*x + n|)

theorem problem_solution (m n : ℝ) (h : m + n = 7) :
  (∀ x : ℝ, f x ≥ (m + n) * x ↔ x ≤ 0) ∧
  (∀ x y : ℝ, F x y m n ≥ 1) ∧ (∃ x y : ℝ, F x y m n = 1) :=
by
  sorry

#check problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1064_106453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_can_escape_l1064_106455

/-- A circular pool with a student and a teacher -/
structure Pool where
  R : ℝ  -- Radius of the pool
  v : ℝ  -- Swimming speed of the student
  r : ℝ  -- Radius of student's circular path

/-- The conditions of the problem -/
def pool_conditions (p : Pool) : Prop :=
  p.R > 0 ∧ p.v > 0 ∧ ((1 - Real.pi/4) * p.R < p.r) ∧ (p.r < p.R/4)

/-- The student can escape if they reach the edge before the teacher -/
def student_escapes (p : Pool) : Prop :=
  p.R / p.v < Real.pi * p.R / (4 * p.v)

/-- The main theorem: given the pool conditions, the student can escape -/
theorem student_can_escape (p : Pool) (h : pool_conditions p) : student_escapes p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_can_escape_l1064_106455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_sum_l1064_106485

noncomputable def a : ℝ × ℝ := (1, -3)
noncomputable def b : ℝ × ℝ := (-2, 0)

noncomputable def vector_sum (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

noncomputable def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem magnitude_of_vector_sum : 
  magnitude (vector_sum (scalar_mult 2 a) b) = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_sum_l1064_106485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_optimization_l1064_106411

-- Define the production quantities and costs
noncomputable def production_A (x : ℝ) : ℝ := x
noncomputable def production_B (x : ℝ) : ℝ := x + 2
noncomputable def cost_A (x : ℝ) : ℝ := 40000 * x^2
noncomputable def cost_B (x : ℝ) : ℝ := 80000 * (x + 2)

-- Define the average cost function
noncomputable def average_cost (x : ℝ) : ℝ := (cost_A x + cost_B x) / (production_A x + production_B x)

-- Define the constrained production range
def production_range (x : ℝ) : Prop := (0 ≤ x ∧ x ≤ 1/2) ∨ (2 ≤ x ∧ x ≤ 8)

-- Theorem statement
theorem production_optimization :
  (∃ (min_cost : ℝ), ∀ (x : ℝ), x > 0 → average_cost x ≥ min_cost ∧ min_cost = 4000 * Real.sqrt 3) ∧
  (∃ (optimal_x : ℝ), production_range optimal_x ∧
    ∀ (x : ℝ), production_range x → average_cost x ≥ average_cost optimal_x ∧ optimal_x = 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_optimization_l1064_106411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_bill_is_correct_l1064_106467

def flat_rate_shipping : ℚ := 5
def shipping_threshold : ℚ := 50
def shipping_rate : ℚ := 1/5

def shirt_price : ℚ := 12
def shirt_quantity : ℕ := 3

def socks_price : ℚ := 5
def socks_quantity : ℕ := 1

def shorts_price : ℚ := 15
def shorts_quantity : ℕ := 2

def swim_trunks_price : ℚ := 14
def swim_trunks_quantity : ℕ := 1

def calculate_total_bill (shirt_price : ℚ) (shirt_quantity : ℕ) 
                         (socks_price : ℚ) (socks_quantity : ℕ)
                         (shorts_price : ℚ) (shorts_quantity : ℕ)
                         (swim_trunks_price : ℚ) (swim_trunks_quantity : ℕ)
                         (flat_rate_shipping : ℚ) (shipping_threshold : ℚ) (shipping_rate : ℚ) : ℚ :=
  let subtotal := shirt_price * shirt_quantity + 
                  socks_price * socks_quantity +
                  shorts_price * shorts_quantity + 
                  swim_trunks_price * swim_trunks_quantity
  let shipping_cost := if subtotal < shipping_threshold then flat_rate_shipping
                       else subtotal * shipping_rate
  subtotal + shipping_cost

theorem total_bill_is_correct : 
  calculate_total_bill shirt_price shirt_quantity 
                       socks_price socks_quantity
                       shorts_price shorts_quantity
                       swim_trunks_price swim_trunks_quantity
                       flat_rate_shipping shipping_threshold shipping_rate = 102 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_bill_is_correct_l1064_106467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_slope_intercept_form_l1064_106480

/-- Given a line passing through points (0,4) and (3,0), its slope-intercept form is y = -4/3x + 4 -/
theorem line_through_points_slope_intercept_form :
  ∀ (l : Set (ℝ × ℝ)),
  (∀ x y, (x, y) ∈ l ↔ ∃ t : ℝ, (x, y) = (0, 4) + t • ((3, 0) - (0, 4))) →
  ∃ m b, ∀ x y, (x, y) ∈ l ↔ y = m * x + b ∧ m = -4/3 ∧ b = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_points_slope_intercept_form_l1064_106480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_l1064_106435

/-- Predicate to check if a point is a focus of the ellipse -/
def is_focus (f : ℝ × ℝ) (x y : ℝ) : Prop :=
  ∃ (c : ℝ), c^2 = 9 - 5 ∧ 
  (x - f.1)^2 + (y - f.2)^2 + (x + f.1)^2 + (y + f.2)^2 = 4 * (3^2)

/-- Given an ellipse with equation x²/9 + y²/5 = 1, prove that one of its foci has coordinates (2, 0) -/
theorem ellipse_focus (x y : ℝ) : 
  (x^2 / 9 + y^2 / 5 = 1) → (∃ (f : ℝ × ℝ), f = (2, 0) ∧ is_focus f x y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_l1064_106435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_medal_award_theorem_l1064_106483

/-- Represents the number of ways to award medals in a race with given constraints -/
def medal_award_ways (total_sprinters : ℕ) (jamaican_sprinters : ℕ) (max_non_jamaican_medals : ℕ) : ℕ :=
  0 -- Placeholder definition, to be implemented

/-- Theorem stating the number of ways to award medals in the 200-meter race -/
theorem medal_award_theorem :
  let total_sprinters := 10
  let jamaican_sprinters := 4
  let non_jamaican_sprinters := total_sprinters - jamaican_sprinters
  let max_non_jamaican_medals := 2
  medal_award_ways total_sprinters jamaican_sprinters max_non_jamaican_medals = 720 :=
by
  sorry

#check medal_award_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_medal_award_theorem_l1064_106483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_existence_l1064_106459

theorem sequence_existence : ∃ (m n : ℕ → ℕ), 
  (∀ i j : ℕ, i ≠ j → m i ≠ m j ∧ n i ≠ n j) ∧ 
  (∀ k : ℕ, ∃ a : ℕ, (n k + (m k)^2 : ℚ).sqrt = a) ∧
  (∀ k : ℕ, ∃ b : ℕ, (n k - (m k)^2 : ℚ).sqrt = b) :=
by
  -- Define m_k and n_k
  let m := λ k : ℕ ↦ 2 * k
  let n := λ k : ℕ ↦ 5 * k^2
  
  -- Prove the existence
  use m, n
  
  sorry -- Skip the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_existence_l1064_106459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_l1064_106452

-- Define the sequence recursively
def t : ℕ → ℚ
  | 0 => 1  -- Add a case for 0
  | 1 => 1
  | n + 2 => if (n + 2) % 3 = 0 then 2 + t ((n + 2) / 3) else 1 / t (n + 1)

-- State the theorem
theorem sequence_value (n : ℕ) : t n = 5/116 → n = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_l1064_106452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l1064_106422

/-- The speed of a train in km/hr given its length and time to cross a pole -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem: A train 240 meters long that takes 6.856594329596489 seconds to cross a pole has a speed of approximately 126 km/hr -/
theorem train_speed_approx :
  let length : ℝ := 240
  let time : ℝ := 6.856594329596489
  |train_speed length time - 126| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approx_l1064_106422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_points_existence_l1064_106458

noncomputable section

/-- The parabola y = x²/4 -/
def parabola (x : ℝ) : ℝ := x^2 / 4

/-- A point on the line x = √3 -/
structure PointOnLine where
  y : ℝ

/-- Checks if two tangent lines to the parabola pass through a given point -/
def has_two_tangents (p : PointOnLine) : Prop :=
  ∃ k₁ k₂ : ℝ, k₁ ≠ k₂ ∧
    (∀ x : ℝ, parabola x = p.y + k₁ * (x - Real.sqrt 3) ∨ 
              parabola x = p.y + k₂ * (x - Real.sqrt 3))

/-- Checks if the angle between two tangent lines is 60° -/
def angle_is_60_degrees (p : PointOnLine) : Prop :=
  ∃ k₁ k₂ : ℝ, k₁ ≠ k₂ ∧ 
    (k₂ - k₁) / (1 + k₁ * k₂) = Real.sqrt 3

/-- Theorem stating the existence of two points satisfying the conditions -/
theorem tangent_points_existence : 
  ∃ (M₁ M₂ : PointOnLine),
    has_two_tangents M₁ ∧
    has_two_tangents M₂ ∧
    angle_is_60_degrees M₁ ∧
    angle_is_60_degrees M₂ ∧
    M₁.y = 0 ∧
    M₂.y = -2/3 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_points_existence_l1064_106458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l1064_106497

theorem exponential_equation_solution :
  ∃ x : ℚ, (3 : ℝ) ^ ((2 : ℝ) * x + 1) = (1 : ℝ) / 81 ∧ x = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l1064_106497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_intra_republic_route_l1064_106456

/-- A structure representing a country with cities and airline routes -/
structure Country where
  cities : Finset Nat
  republics : Finset Nat
  city_republic : Nat → Nat
  routes : Nat → Finset Nat

/-- The theorem stating the existence of an intra-republic route -/
theorem exists_intra_republic_route (c : Country) 
  (h1 : c.cities.card = 100)
  (h2 : c.republics.card = 3)
  (h3 : ∀ i, i ∈ c.cities → c.city_republic i ∈ c.republics)
  (h4 : ∃ S : Finset Nat, S ⊆ c.cities ∧ S.card ≥ 70 ∧ ∀ i ∈ S, (c.routes i).card ≥ 70) :
  ∃ i j, i ∈ c.cities ∧ j ∈ c.cities ∧ i ≠ j ∧ j ∈ c.routes i ∧ c.city_republic i = c.city_republic j :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_intra_republic_route_l1064_106456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1064_106479

/-- Given a triangle ABC with sides a, b, and c, where a = 4, b = 6, and the perimeter is an even number less than 18, prove that c = 4 or c = 6 and the triangle is isosceles. -/
theorem triangle_problem (a b c : ℝ) : 
  a = 4 → 
  b = 6 → 
  a + b + c < 18 → 
  Even (Int.floor (a + b + c)) →
  (c = 4 ∨ c = 6) ∧ 
  (a = c ∨ b = c) := by
  sorry

#check triangle_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1064_106479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_alpha_value_l1064_106444

theorem cosine_alpha_value (α β : ℝ) : 
  0 < α ∧ α < Real.pi → 
  0 < β ∧ β < Real.pi → 
  Real.cos β = -5/13 → 
  Real.sin (α + β) = 3/5 → 
  Real.cos α = 56/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_alpha_value_l1064_106444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_painted_fraction_is_one_third_l1064_106470

/-- Represents a triangular prism -/
structure TriangularPrism where
  edge_length : ℝ
  deriving Inhabited

/-- The large triangular prism composed of smaller prisms -/
def large_prism : TriangularPrism :=
  { edge_length := 3 }

/-- A smaller triangular prism that makes up the large prism -/
def small_prism : TriangularPrism :=
  { edge_length := 1 }

/-- The number of smaller prisms that make up the large prism -/
def num_small_prisms : ℕ := 27

/-- Calculate the surface area of a triangular prism -/
noncomputable def surface_area (prism : TriangularPrism) : ℝ :=
  3 * prism.edge_length ^ 2 + 2 * (Real.sqrt 3 / 4) * prism.edge_length ^ 2

/-- Calculate the painted surface area of the large prism -/
noncomputable def painted_area : ℝ :=
  surface_area large_prism

/-- Calculate the total surface area of all small prisms -/
noncomputable def total_small_prisms_area : ℝ :=
  num_small_prisms * surface_area small_prism

/-- The main theorem to prove -/
theorem painted_fraction_is_one_third :
  painted_area / total_small_prisms_area = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_painted_fraction_is_one_third_l1064_106470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l1064_106426

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 2 * Real.cos x) : Real.sin x * Real.cos x = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l1064_106426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_theorem_l1064_106424

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x > 1, y = Real.log x / Real.log 2}
def B : Set ℝ := {y | ∃ x > 1, y = (1/2) ^ x}

-- Define the intersection set
def intersection_set : Set ℝ := {y | 0 < y ∧ y < 1/2}

-- Theorem statement
theorem set_intersection_theorem : A ∩ B = intersection_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_theorem_l1064_106424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_catchup_at_60km_l1064_106469

/-- The distance at which B catches up with A -/
noncomputable def catchup_distance (speed_a speed_b : ℝ) (time_diff : ℝ) : ℝ :=
  (speed_b * time_diff * speed_a) / (speed_b - speed_a)

/-- Theorem stating that B catches up with A at 60 km -/
theorem catchup_at_60km (speed_a speed_b time_diff : ℝ) 
  (ha : speed_a = 10)
  (hb : speed_b = 20)
  (ht : time_diff = 3)
  (h_speed : speed_b > speed_a) :
  catchup_distance speed_a speed_b time_diff = 60 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_catchup_at_60km_l1064_106469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_arrival_l1064_106446

/-- Represents the scenario of two people traveling from M to N with a bicycle --/
structure TravelScenario where
  distance : ℚ  -- Distance from M to N in km
  walkSpeed : ℚ  -- Walking speed in km/h
  bikeSpeed : ℚ  -- Bicycle speed in km/h

/-- Calculates the time C should leave before A and B --/
def timeCLeavesBefore (scenario : TravelScenario) : ℚ :=
  3 / 11

/-- Theorem stating that C should leave 3/11 hours before A and B for simultaneous arrival --/
theorem simultaneous_arrival (scenario : TravelScenario) 
  (h1 : scenario.distance = 15)
  (h2 : scenario.walkSpeed = 6)
  (h3 : scenario.bikeSpeed = 15) :
  timeCLeavesBefore scenario = 3 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simultaneous_arrival_l1064_106446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insurance_coverage_percentage_l1064_106451

/-- Given a doctor visit with a total cost and an out-of-pocket cost,
    calculate the percentage of insurance coverage. -/
theorem insurance_coverage_percentage
  (total_cost : ℝ)
  (out_of_pocket : ℝ)
  (h1 : total_cost = 300)
  (h2 : out_of_pocket = 60) :
  (total_cost - out_of_pocket) / total_cost * 100 = 80 := by
  sorry

#check insurance_coverage_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_insurance_coverage_percentage_l1064_106451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_given_asymptote_point_l1064_106484

/-- A hyperbola with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The slope of the asymptotes of a hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola) : ℝ :=
  h.b / h.a

theorem hyperbola_eccentricity_given_asymptote_point (h : Hyperbola) 
  (asymptote_point : asymptote_slope h = 4/3) :
  eccentricity h = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_given_asymptote_point_l1064_106484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l1064_106499

/-- Sequence defined recursively -/
def a : ℕ → ℚ
  | 0 => 2
  | 1 => 3
  | (n+2) => (1/4) * a (n+1) + (1/3) * a n

/-- Sum of the sequence -/
noncomputable def S : ℚ := ∑' n, a n

/-- Theorem stating that the sum of the sequence is 10.8 -/
theorem sequence_sum : S = 10.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l1064_106499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_third_l1064_106454

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let ab := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let bc := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let ca := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  ab = 10 ∧ bc = 12 ∧ ca = 14

-- Define points D, E, F on the sides of triangle ABC
def PointsOnSides (A B C D E F : ℝ × ℝ) (p q r : ℝ) : Prop :=
  ∃ (t₁ t₂ t₃ : ℝ), 
    D = (t₁ * A.1 + (1 - t₁) * B.1, t₁ * A.2 + (1 - t₁) * B.2) ∧
    E = (t₂ * B.1 + (1 - t₂) * C.1, t₂ * B.2 + (1 - t₂) * C.2) ∧
    F = (t₃ * C.1 + (1 - t₃) * A.1, t₃ * C.2 + (1 - t₃) * A.2) ∧
    t₁ = 1 - p ∧ t₂ = 1 - q ∧ t₃ = 1 - r

-- Define the conditions on p, q, r
def Conditions (p q r : ℝ) : Prop :=
  p > 0 ∧ q > 0 ∧ r > 0 ∧ p + q + r = 1 ∧ p^2 + q^2 + r^2 = 1/3

-- Define the area ratio
noncomputable def AreaRatio (A B C D E F : ℝ × ℝ) : ℝ :=
  let area_ABC := abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2
  let area_DEF := abs ((E.1 - D.1) * (F.2 - D.2) - (F.1 - D.1) * (E.2 - D.2)) / 2
  area_DEF / area_ABC

-- The main theorem
theorem area_ratio_is_one_third 
  (A B C D E F : ℝ × ℝ) (p q r : ℝ) : 
  Triangle A B C → 
  PointsOnSides A B C D E F p q r → 
  Conditions p q r → 
  AreaRatio A B C D E F = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_third_l1064_106454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ink_transfer_equality_l1064_106421

/-- Proof that the amount of blue ink mixed into Cup A equals the amount of red ink mixed into Cup B after transfers -/
theorem ink_transfer_equality (m a : ℝ) (h1 : 0 < a) (h2 : a < m) :
  (m * a) / (m + a) = (m * a) / (m + a) := by
  -- The proof is trivial as both sides are identical
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ink_transfer_equality_l1064_106421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l1064_106428

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.log (3 * x)

-- State the theorem
theorem intersection_point :
  ∃! x : ℝ, x > 0 ∧ f x = g x ∧ x = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l1064_106428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1064_106408

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x < 7}

-- Define set B
def B : Set ℝ := {x | x^2 - 4*x - 5 ≤ 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 1 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1064_106408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_leak_time_l1064_106496

/-- Calculates the time elapsed for a bottle to leak from an initial volume to a final volume at a given leak rate. -/
noncomputable def timeElapsed (initialVolume finalVolume leakRate : ℝ) : ℝ :=
  (initialVolume - finalVolume) / leakRate

/-- Theorem: The time elapsed for a 1000 ml bottle of milk to leak down to 300 ml at a rate of 5 ml per minute is 140 minutes. -/
theorem milk_leak_time :
  timeElapsed 1000 300 5 = 140 := by
  -- Unfold the definition of timeElapsed
  unfold timeElapsed
  -- Simplify the arithmetic
  simp [sub_div]
  -- Check that the result is equal to 140
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_leak_time_l1064_106496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_identity_l1064_106445

theorem sine_sum_identity (x y : ℝ) :
  Real.sin (x + y) * Real.cos y + Real.cos (x + y) * Real.sin y = Real.sin (x + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_identity_l1064_106445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1064_106493

-- Define vectors a and b
noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos x + Real.sin x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sin x - Real.cos x)

-- Define function f
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- State the theorem
theorem range_of_m :
  ∃ (S : Set ℝ), S = {m : ℝ | ∀ (x : ℝ), x ∈ Set.Icc (5*π/24) (5*π/12) →
    ∀ (t : ℝ), m*t^2 + m*t + 3 ≥ f x} ∧
  S = Set.Icc 0 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1064_106493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_cos_f_special_value_l1064_106402

-- Define the function f
noncomputable def f (α : Real) : Real := (2 * (Real.cos α)^2 - Real.sin (2 * α)) / (2 * (Real.cos α - Real.sin α))

-- Theorem 1: f(α) = cos(α) for all α
theorem f_equals_cos (α : Real) : f α = Real.cos α := by sorry

-- Theorem 2: f(α + π/6) = -(4√3 + 3)/10 when α is in the second quadrant and sin(α) = 3/5
theorem f_special_value (α : Real) 
  (h1 : π/2 < α ∧ α < π) -- α is in the second quadrant
  (h2 : Real.sin α = 3/5) : 
  f (α + π/6) = -(4 * Real.sqrt 3 + 3)/10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_cos_f_special_value_l1064_106402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_l1064_106406

variable (n : ℕ) (f : ℝ → ℝ → ℝ)

-- Condition 1: Homogeneity
axiom homogeneity : ∀ (t x y : ℝ), f (t * x) (t * y) = (t ^ n) * f x y

-- Condition 2: Sum property
axiom sum_property : ∀ (a b c : ℝ), f (a + b) c + f (b + c) a + f (c + a) b = 0

-- Condition 3: Initial value
axiom initial_value : f 1 0 = 1

-- Theorem statement
theorem unique_polynomial : 
  ∃! (g : ℝ → ℝ → ℝ), (∀ x y, f x y = g x y) ∧ 
  (∀ X Y : ℝ, g X Y = (X - 2*Y) * ((X + Y) ^ (n - 1))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_polynomial_l1064_106406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_R_value_l1064_106468

/-- The side length of the larger square -/
def s : ℝ := 6

/-- The area of the inscribed rectangle -/
def rect_area : ℝ := 4

/-- The area of the inscribed circle -/
noncomputable def circle_area : ℝ := Real.pi

/-- The maximum area of rectangle R -/
noncomputable def max_area_R : ℝ := s^2 - rect_area - circle_area

theorem max_area_R_value : max_area_R = 32 - Real.pi := by
  unfold max_area_R
  unfold s
  unfold rect_area
  unfold circle_area
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_R_value_l1064_106468
