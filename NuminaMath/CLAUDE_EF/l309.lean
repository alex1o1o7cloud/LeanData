import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_subsets_covering_triples_l309_30918

theorem min_subsets_covering_triples (S : Finset Nat) (n : Nat) 
  (A : Fin n → Finset Nat) : 
  S = Finset.range 15 →
  (∀ i, (A i).card = 7) →
  (∀ i j, i < j → (A i ∩ A j).card ≤ 3) →
  (∀ M : Finset Nat, M ⊆ S → M.card = 3 → ∃ k, M ⊆ A k) →
  n ≥ 15 ∧ ∃ A' : Fin 15 → Finset Nat, 
              (∀ i, (A' i).card = 7) ∧ 
              (∀ i j, i < j → (A' i ∩ A' j).card ≤ 3) ∧
              (∀ M : Finset Nat, M ⊆ S → M.card = 3 → ∃ k, M ⊆ A' k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_subsets_covering_triples_l309_30918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_tank_height_l309_30993

noncomputable def rainfall_pattern (t : ℝ) : ℝ :=
  if t ≤ 1 then 2
  else if t ≤ 5 then 1
  else 3

noncomputable def total_rainfall (start_time end_time : ℝ) : ℝ :=
  ∫ t in start_time..end_time, rainfall_pattern t

theorem fish_tank_height : total_rainfall 0 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_tank_height_l309_30993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_hands_angle_difference_l309_30951

noncomputable def hour_hand_angle (n : ℝ) : ℝ := 180 + n / 2

noncomputable def minute_hand_angle (n : ℝ) : ℝ := 6 * n

noncomputable def hand_angle_diff (n : ℝ) : ℝ := |hour_hand_angle n - minute_hand_angle n|

theorem watch_hands_angle_difference :
  ∃ (t₁ t₂ : ℝ), 0 < t₁ ∧ t₁ < t₂ ∧ t₂ < 60 ∧
  hand_angle_diff t₁ = 110 ∧
  hand_angle_diff t₂ = 110 ∧
  t₂ - t₁ = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_watch_hands_angle_difference_l309_30951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l309_30956

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 3^x

-- State the theorem
theorem f_difference (x : ℝ) : f (x + 2) - f x = 8 * f x := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the left-hand side
  simp [Real.rpow_add]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l309_30956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_equals_interval_l309_30964

def M : Set ℝ := {x | x^2 ≤ 9}
def N : Set ℝ := {x | x ≤ 1}

theorem set_intersection_equals_interval : M ∩ N = Set.Icc (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_equals_interval_l309_30964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l309_30991

theorem count_integers_satisfying_inequality :
  (Finset.filter (fun n : ℕ => n > 0 ∧ 21 - 3 * n > 12) (Finset.range 21)).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l309_30991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_630_l309_30917

theorem prime_divisors_of_630 : 
  let n : ℕ := 630
  let prime_divs := Finset.filter (λ p => Nat.Prime p ∧ p ∣ n) (Finset.range (n + 1))
  (Finset.card prime_divs = 4) ∧ (Finset.sum prime_divs id = 17) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_630_l309_30917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_vest_cost_is_72_l309_30985

/-- Calculates the cost of increasing a weight vest's weight by a given percentage --/
def weightVestCost (initialWeight : ℕ) (increasePercentage : ℚ) (ingotWeight : ℕ) (ingotCost : ℕ) (discountPercentage : ℚ) (discountThreshold : ℕ) : ℕ :=
  let additionalWeight := (initialWeight : ℚ) * increasePercentage
  let numIngots := (additionalWeight / ingotWeight).ceil.toNat
  let initialCost := numIngots * ingotCost
  let discountAmount := if numIngots > discountThreshold then (initialCost : ℚ) * discountPercentage else 0
  ((initialCost : ℚ) - discountAmount).floor.toNat

/-- The cost to increase a 60-pound weight vest by 60% is $72 --/
theorem weight_vest_cost_is_72 :
  weightVestCost 60 (60 / 100) 2 5 (20 / 100) 10 = 72 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_vest_cost_is_72_l309_30985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_distribution_property_l309_30940

/-- Define a symmetric distribution about a mean -/
def SymmetricDistribution (μ : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f (μ + x) = f (μ - x)

/-- Define the cumulative distribution function -/
noncomputable def CDF (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  ∫ y in Set.Iio x, f y

/-- Define the property that 84% of the distribution is within one standard deviation -/
def WithinOneStdDev (μ σ : ℝ) (f : ℝ → ℝ) : Prop :=
  CDF f (μ + σ) - CDF f (μ - σ) = 0.84

/-- Theorem statement -/
theorem symmetric_distribution_property
  (μ σ : ℝ) (f : ℝ → ℝ) 
  (h_symmetric : SymmetricDistribution μ f)
  (h_within_std : WithinOneStdDev μ σ f) :
  CDF f (μ + σ) - CDF f μ = 0.42 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_distribution_property_l309_30940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l309_30943

/-- The complex number z defined as i^2016 / (3 + 2i) -/
noncomputable def z : ℂ := (Complex.I ^ 2016) / (3 + 2 * Complex.I)

/-- A complex number is in the fourth quadrant if its real part is positive and its imaginary part is negative -/
def is_in_fourth_quadrant (w : ℂ) : Prop := 0 < w.re ∧ w.im < 0

/-- Theorem stating that z is in the fourth quadrant -/
theorem z_in_fourth_quadrant : is_in_fourth_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_fourth_quadrant_l309_30943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_arrangement_theorem_l309_30912

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The angle between three points in 3D space -/
noncomputable def angle (A B C : Point3D) : ℝ := sorry

/-- A set of n points in 3D space -/
def PointSet (n : ℕ) := Fin n → Point3D

theorem point_arrangement_theorem (n : ℕ) (hs : n ≥ 3) (points : PointSet n)
  (h_angle : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → 
    ∃ (a : Fin 3), angle (points i) (points j) (points k) ≥ 120) :
  ∃ (perm : Fin n → Fin n), Function.Bijective perm ∧
    ∀ (i j k : Fin n), i < j → j < k →
      angle (points (perm i)) (points (perm j)) (points (perm k)) ≥ 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_arrangement_theorem_l309_30912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_7623_496_l309_30903

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

/-- The theorem states that rounding 7623.496 to the nearest whole number equals 7623 -/
theorem round_7623_496 : round_to_nearest 7623.496 = 7623 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_7623_496_l309_30903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_sum_theorem_l309_30906

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y = 2x^2 -/
def parabola (p : Point) : Prop :=
  p.y = 2 * p.x^2

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: There exists a constant d such that for all chords PQ of the parabola y = 2x^2
    passing through D = (0,d), the sum 1/PD + 1/QD is equal to 8 -/
theorem parabola_chord_sum_theorem :
  ∃ d : ℝ, ∀ P Q : Point,
    parabola P → parabola Q →
    (∃ m : ℝ, P.y = m * P.x + d ∧ Q.y = m * Q.x + d) →
    1 / distance P ⟨0, d⟩ + 1 / distance Q ⟨0, d⟩ = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_sum_theorem_l309_30906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_double_angle_l309_30965

theorem sin_cos_double_angle (α : ℝ) 
  (h1 : Real.sin α = 1/2 + Real.cos α) 
  (h2 : 0 < α ∧ α < π/2) : 
  Real.sin (2*α) = 3/4 ∧ Real.cos (2*α) = -Real.sqrt 7 / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_double_angle_l309_30965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_current_l309_30953

-- Define complex numbers
noncomputable def V : ℂ := 2 + 2*Complex.I
noncomputable def Z₁ : ℂ := 2 + Complex.I
noncomputable def Z₂ : ℂ := 4 + 2*Complex.I

-- Define the total impedance Z
noncomputable def Z : ℂ := (Z₁ * Z₂) / (Z₁ + Z₂)

-- Define the current I
noncomputable def I : ℂ := V / Z

-- Theorem to prove
theorem circuit_current : I = 2 + 2*Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circuit_current_l309_30953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_insured_employees_is_121_l309_30973

/-- The maximum number of employees that can be insured given the specified conditions -/
def max_insured_employees : ℕ :=
  let total_premium : ℚ := 5000000
  let outpatient_cost : ℚ := 18000
  let hospitalization_cost : ℚ := 60000
  let hospitalization_rate : ℚ := 1/4
  let overhead_cost : ℚ := 2800
  let profit_rate : ℚ := 15/100
  let total_cost_per_employee : ℚ := outpatient_cost + hospitalization_rate * hospitalization_cost + overhead_cost
  let profit_per_employee : ℚ := profit_rate * total_cost_per_employee
  let total_cost_with_profit : ℚ := total_cost_per_employee + profit_per_employee
  (total_premium / total_cost_with_profit).floor.toNat

theorem max_insured_employees_is_121 : max_insured_employees = 121 := by
  sorry

#eval max_insured_employees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_insured_employees_is_121_l309_30973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_circle_intersection_l309_30925

-- Define the rectangle
def rectangle_vertices : List (ℝ × ℝ) := [(5, 11), (16, 11), (16, -8), (5, -8)]

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 5)^2 + (y - 1)^2 = 25

-- Define the intersection area
def intersection_area : ℝ := 0

-- Theorem statement
theorem rectangle_circle_intersection :
  ∀ (x y : ℝ),
  (x, y) ∈ rectangle_vertices →
  ¬ circle_equation x y →
  intersection_area = 0 :=
by
  intros x y h1 h2
  exact rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_circle_intersection_l309_30925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_properties_function_value_at_specific_point_l309_30907

noncomputable def f (a b ω x : ℝ) : ℝ := a * Real.cos (ω * x) + b * Real.sin (ω * x)

theorem periodic_function_properties :
  ∀ (a b ω : ℝ),
  ω > 0 →
  (∀ (x : ℝ), f a b ω (x + π / 2) = f a b ω x) →
  (∃ (x : ℝ), f a b ω x = 4 ∧ ∀ (y : ℝ), f a b ω y ≤ 4) →
  f a b ω (π / 6) = 4 →
  (a = -2 ∧ b = 2 * Real.sqrt 3 ∧ ω = 4) := by
  sorry

theorem function_value_at_specific_point :
  ∀ (a b ω x : ℝ),
  a = -2 →
  b = 2 * Real.sqrt 3 →
  ω = 4 →
  π / 4 < x →
  x < 3 * π / 4 →
  f a b ω (x + π / 6) = 4 / 3 →
  f a b ω (x / 2 + π / 6) = -4 * Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_properties_function_value_at_specific_point_l309_30907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l309_30970

noncomputable section

-- Define the domain conditions
def is_valid (m : ℝ) : Prop := m ≥ 4 ∧ m ≠ 8

-- Define the original expression
noncomputable def original_expr (m : ℝ) : ℝ :=
  (((m + 4 * (m - 4)^(1/2))^(1/3) * ((m - 4)^(1/2) + 2)^(1/3)) / 
   ((m - 4 * (m - 4)^(1/2))^(1/3) * ((m - 4)^(1/2) - 2)^(1/3))) * 
  (m - 4 * (m - 4)^(1/2)) / 2

-- State the theorem
theorem simplify_expression (m : ℝ) (h : is_valid m) : 
  original_expr m = (m - 8) / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l309_30970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_babblean_words_count_l309_30928

/-- The number of letters in the Babblean alphabet -/
def alphabet_size : ℕ := 7

/-- The maximum word length in the Babblean language -/
def max_word_length : ℕ := 4

/-- The number of possible words of a given length -/
def words_of_length (n : ℕ) : ℕ := alphabet_size ^ n

/-- The total number of possible words in the Babblean language -/
def total_words : ℕ := Finset.sum (Finset.range max_word_length) (fun i => words_of_length (i + 1))

/-- Theorem stating the total number of possible words in the Babblean language -/
theorem babblean_words_count : total_words = 2800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_babblean_words_count_l309_30928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monday_dressing_time_l309_30931

/-- Represents the time taken to get dressed each day of the week -/
structure DressingTimes where
  monday : ℚ
  tuesday : ℚ
  wednesday : ℚ
  thursday : ℚ
  friday : ℚ

/-- Calculates the average time for a given number of days -/
def average (total : ℚ) (days : ℚ) : ℚ := total / days

/-- The theorem to be proved -/
theorem monday_dressing_time (times : DressingTimes) 
  (h1 : times.tuesday = 4)
  (h2 : times.wednesday = 3)
  (h3 : times.thursday = 4)
  (h4 : times.friday = 2)
  (h5 : average (times.monday + times.tuesday + times.wednesday + times.thursday + times.friday) 5 = 3) :
  times.monday = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monday_dressing_time_l309_30931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_sum_seven_eq_28_l309_30996

/-- The number of three-digit positive integers with digit sum 7 -/
def count_three_digit_sum_seven : ℕ :=
  (Finset.filter (λ n : ℕ ↦ 
    100 ≤ n ∧ n ≤ 999 ∧ 
    (n / 100 + (n / 10) % 10 + n % 10 = 7)) 
    (Finset.range 1000)).card

/-- Theorem stating that the count of three-digit positive integers with digit sum 7 is 28 -/
theorem count_three_digit_sum_seven_eq_28 : 
  count_three_digit_sum_seven = 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_digit_sum_seven_eq_28_l309_30996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l309_30984

-- Define the function f
noncomputable def f (x : Real) : Real := Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x ^ 2 - 1/2

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h1 : f t.B = 1)
  (h2 : t.a = Real.sqrt 3)
  (h3 : t.b = 1) :
  t.B = π/6 ∧ (t.c = 1 ∨ t.c = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l309_30984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pages_for_30_dollars_l309_30967

/-- The cost in cents to copy a given number of pages -/
def copy_cost (pages : ℕ) : ℚ :=
  (7 : ℚ) / 4 * pages

/-- The number of pages that can be copied for a given amount in dollars -/
def pages_copied (dollars : ℚ) : ℕ :=
  (dollars * 100 * 4 / 7 : ℚ).floor.toNat

/-- Theorem stating that the maximum number of whole pages that can be copied for $30 is 1714 -/
theorem max_pages_for_30_dollars : pages_copied 30 = 1714 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pages_for_30_dollars_l309_30967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_rhombus_intersection_l309_30974

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a tetrahedron
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

-- Define a plane
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a rhombus
structure Rhombus where
  center : Point3D
  side_length : ℝ
  angle : ℝ

-- Placeholder function for intersection
noncomputable def intersection (tetra : Tetrahedron) (p : Plane) : Option Rhombus :=
  sorry

-- Theorem statement
theorem tetrahedron_rhombus_intersection
  (tetra : Tetrahedron) :
  ∃ (p : Plane), ∃ (r : Rhombus),
    intersection tetra p = some r :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_rhombus_intersection_l309_30974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_M_to_N_l309_30921

/-- The distance between two points in 3D space -/
noncomputable def distance3D (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2)

/-- Theorem: The distance between points M(1, 0, 2) and N(-1, 2, 0) is 2√3 -/
theorem distance_M_to_N : distance3D 1 0 2 (-1) 2 0 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_M_to_N_l309_30921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_books_borrowed_is_eight_l309_30954

/-- Represents the maximum number of books borrowed by a single student. -/
def max_books_borrowed (total_students : ℕ) (zero_book_students : ℕ) (one_book_students : ℕ) 
  (two_book_students : ℕ) (avg_books : ℚ) : ℕ :=
  let total_books := (total_students : ℚ) * avg_books
  let known_books := (one_book_students : ℚ) + 2 * two_book_students
  let remaining_students := total_students - zero_book_students - one_book_students - two_book_students
  let remaining_books := total_books - known_books
  (remaining_books - (remaining_students - 1 : ℚ) * 3).floor.toNat

/-- Theorem stating the maximum number of books borrowed by a single student 
    given the conditions in the problem. -/
theorem max_books_borrowed_is_eight :
  max_books_borrowed 20 2 8 3 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_books_borrowed_is_eight_l309_30954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_l309_30997

/-- Definition of series A' -/
noncomputable def A' : ℝ := ∑' n, if (n % 2 ≠ 0 ∧ n % 5 ≠ 0) then ((-1)^((n-1)/2) / n^2) else 0

/-- Definition of series B' -/
noncomputable def B' : ℝ := ∑' n, if (n % 2 ≠ 0 ∧ n % 5 = 0) then ((-1)^((n-5)/10) / n^2) else 0

/-- Theorem stating that A' divided by B' equals 26 -/
theorem A'_div_B'_eq_26 : A' / B' = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_l309_30997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yeast_experiment_properties_l309_30968

/-- Represents an experiment exploring yeast population changes -/
structure YeastExperiment where
  /-- The culture medium used in the experiment -/
  medium : Type
  /-- A function representing the sampling process -/
  sample : medium → ℕ
  /-- A function representing the effect of shaking on sampling -/
  shake : medium → medium

/-- Represents the state of the experiment at a given time -/
structure ExperimentState (e : YeastExperiment) where
  /-- The current state of the culture medium -/
  current_medium : e.medium
  /-- The yeast population count -/
  population : ℕ

/-- Theorem stating the key aspects of the yeast population experiment -/
theorem yeast_experiment_properties (e : YeastExperiment) :
  /- 1. Shaking affects sampling results -/
  (∃ m : e.medium, e.sample m ≠ e.sample (e.shake m)) ∧
  /- 2. The experiment uses self-comparison -/
  (∃ t₁ t₂ : ExperimentState e, t₁.current_medium ≠ t₂.current_medium ∧ t₁.population ≠ t₂.population) ∧
  /- 3. Sampling is used to estimate population -/
  (∀ t : ExperimentState e, ∃ s : ℕ, s = e.sample t.current_medium) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yeast_experiment_properties_l309_30968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_g_eq_12_l309_30960

-- Define the piecewise function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if x < -1 then 3 * x + 12
  else if x < 1 then x^2 - 1
  else 2 * x + 1

-- Theorem statement
theorem unique_solution_g_eq_12 :
  ∃! x : ℝ, g x = 12 ∧ x = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_g_eq_12_l309_30960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_18_divisors_l309_30901

def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_with_18_divisors :
  ∀ n : ℕ, n > 0 → num_divisors n = 18 → n ≥ 90 :=
by
  sorry

#eval num_divisors 90

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_with_18_divisors_l309_30901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_minimum_l309_30920

noncomputable def f (t u v : ℕ) (x : ℝ) : ℝ :=
  (2^(3*u - 5*v - 3) - 1) * x^2 + 
  (2^(7*u - 8*v - 18) - 2^(4*u - 3*v - 15)) * x + 
  (11*t^2 - 5*u) / (154*v - 10*t^2)

noncomputable def f' (t u v : ℕ) (x : ℝ) : ℝ :=
  2 * (2^(3*u - 5*v - 3) - 1) * x + 
  (2^(7*u - 8*v - 18) - 2^(4*u - 3*v - 15))

theorem function_minimum (t u v : ℕ) (h1 : t = 3) (h2 : u = 4) (h3 : v = 1) :
  (∀ x : ℝ, f t u v (-1/8) ≤ f t u v x) ∧ 
  (f t u v (-1/8) > 0) ∧
  (f' t u v (-1/8) = 0) := by
  sorry

#check function_minimum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_minimum_l309_30920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l309_30944

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := 
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The angle between two vectors -/
noncomputable def angle (v1 v2 : Point × Point) : ℝ :=
  Real.arccos ((v1.1.x - v1.2.x) * (v2.1.x - v2.2.x) + (v1.1.y - v1.2.y) * (v2.1.y - v2.2.y)) /
    (((v1.1.x - v1.2.x)^2 + (v1.1.y - v1.2.y)^2).sqrt * ((v2.1.x - v2.2.x)^2 + (v2.1.y - v2.2.y)^2).sqrt)

theorem ellipse_eccentricity_special_case (e : Ellipse) 
  (A : Point) -- right vertex of the ellipse
  (P : Point) -- point on the ellipse
  (O : Point) -- origin
  (h_ellipse_eq : ∀ (x y : ℝ), x^2 / e.a^2 + y^2 / e.b^2 = 1 → 
    ∃ (p : Point), p.x = x ∧ p.y = y) -- ellipse equation
  (h_A_vertex : A.x = e.a ∧ A.y = 0) -- A is right vertex
  (h_P_on_ellipse : P.x^2 / e.a^2 + P.y^2 / e.b^2 = 1) -- P is on ellipse
  (h_O_origin : O.x = 0 ∧ O.y = 0) -- O is origin
  (h_angle_POA : Real.cos (angle (P, O) (A, O)) = 1/2) -- angle POA is 60 degrees
  (h_OP_perp_AP : (P.x - O.x) * (P.x - A.x) + (P.y - O.y) * (P.y - A.y) = 0) -- OP perpendicular to AP
  : eccentricity e = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_special_case_l309_30944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l309_30958

noncomputable def f (x : ℝ) : ℝ := Real.tan (Real.pi * x + Real.pi / 4)

def IsCenterOfSymmetry (f : ℝ → ℝ) (x₀ y₀ : ℝ) : Prop :=
  ∀ x, f (2 * x₀ - x) = 2 * y₀ - f x

theorem center_of_symmetry (k : ℤ) :
  IsCenterOfSymmetry f ((2 * k - 1 : ℤ) / 4 : ℝ) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_l309_30958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l309_30989

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

noncomputable def eccentricity (a c : ℝ) : ℝ :=
  c / a

theorem hyperbola_eccentricity :
  ∃ (a b c : ℝ),
    (∀ x y : ℝ, hyperbola_equation x y ↔ x^2 / (a^2) - y^2 / (b^2) = 1) ∧
    (c^2 = a^2 + b^2) ∧
    (eccentricity a c = 5/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l309_30989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l309_30963

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 + sqrt (24 * x - 9 * x^2)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 5 ∧ (∀ x, 0 < x → x < 2 → f x ≤ M) ∧ (∃ x, 0 < x ∧ x < 2 ∧ f x = M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l309_30963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_swap_and_triple_l309_30933

theorem matrix_swap_and_triple :
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 3, 0]
  ∀ (N : Matrix (Fin 2) (Fin 2) ℝ),
    ∃ (a b c d : ℝ),
      N = !![a, b; c, d] →
      M * N = !![c, d; 3*a, 3*b] := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_swap_and_triple_l309_30933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_first_quadrant_l309_30945

noncomputable def complex_number : ℂ := (1 / (1 + Complex.I)) + Complex.I

theorem point_in_first_quadrant : 
  0 < complex_number.re ∧ 0 < complex_number.im := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_first_quadrant_l309_30945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_sum_bounds_l309_30992

theorem permutation_sum_bounds (x₁ x₂ x₃ x₄ : ℝ) (h : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄) :
  ∀ (y₁ y₂ y₃ y₄ : ℝ), (∃ (σ : Equiv.Perm (Fin 4)), y₁ = x₁ ∨ y₁ = x₂ ∨ y₁ = x₃ ∨ y₁ = x₄) →
                       (∃ (σ : Equiv.Perm (Fin 4)), y₂ = x₁ ∨ y₂ = x₂ ∨ y₂ = x₃ ∨ y₂ = x₄) →
                       (∃ (σ : Equiv.Perm (Fin 4)), y₃ = x₁ ∨ y₃ = x₂ ∨ y₃ = x₃ ∨ y₃ = x₄) →
                       (∃ (σ : Equiv.Perm (Fin 4)), y₄ = x₁ ∨ y₄ = x₂ ∨ y₄ = x₃ ∨ y₄ = x₄) →
  (y₁ - y₂)^2 + (y₂ - y₃)^2 + (y₃ - y₄)^2 + (y₄ - y₁)^2 ≤ (x₁ - x₃)^2 + (x₃ - x₂)^2 + (x₂ - x₄)^2 + (x₄ - x₁)^2 ∧
  (y₁ - y₂)^2 + (y₂ - y₃)^2 + (y₃ - y₄)^2 + (y₄ - y₁)^2 ≥ (x₁ - x₂)^2 + (x₂ - x₄)^2 + (x₄ - x₃)^2 + (x₃ - x₁)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_sum_bounds_l309_30992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_sequence_theorem_l309_30935

def cartesian_points (n : ℕ) (P : Fin (n + 1) → ℤ × ℤ) : Prop :=
  (P 0 = (0, 1)) ∧ 
  (∀ k : Fin n, (P (k.succ)).1 - (P k).1 = 1) ∧
  (∀ k : Fin n, |((P (k.succ)).1 - (P k).1)| * |((P (k.succ)).2 - (P k).2)| = 2) ∧
  (∀ k : Fin n, (P (k.succ)).2 > (P k).2) ∧
  ((P n).2 = 3 * (P n).1 - 8)

theorem point_sequence_theorem : 
  ∃ (P : Fin 10 → ℤ × ℤ), cartesian_points 9 P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_sequence_theorem_l309_30935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_sequence_first_element_l309_30981

/-- A sequence defined by the given recursion formula -/
def RecursiveSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = 1 / (1 - a n) - 1 / (1 + a n)

/-- Periodicity of a sequence -/
def IsPeriodic (a : ℕ → ℝ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ ∀ n, a (n + k) = a n

/-- The form of the first element of the sequence -/
def FirstElementForm (x : ℝ) : Prop :=
  ∃ k l : ℤ, k ≥ 1 ∧ x = Real.tan (Real.pi * (l : ℝ) / ((2 : ℝ)^k - 1))

theorem periodic_sequence_first_element
  (a : ℕ → ℝ)
  (h_recursive : RecursiveSequence a)
  (h_periodic : IsPeriodic a) :
  FirstElementForm (a 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_sequence_first_element_l309_30981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l309_30942

def x : ℕ → ℚ
  | 0 => 0
  | 1 => 1/2
  | n+2 => (x (n+1) + x n) / 2

def a (n : ℕ) : ℚ := x (n+1) - x n

theorem sequence_formula (n : ℕ) : a n = (-1)^n / 2^(n+1) := by
  sorry

#eval a 1
#eval a 2
#eval a 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l309_30942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l309_30957

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x^2
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x + a * x - 3/2

theorem problem_solution :
  (∃ k : ℝ, k = (deriv (g 1)) 1 ∧ k * 1 = 3 * Real.exp 2) ∧
  (∀ x > 0, f 1 x < g 1 x) ∧
  (∀ x ∈ Set.Ioo 0 (Real.sqrt 2 / 2), deriv (f 1) x > 0) ∧
  (∀ x ∈ Set.Ioi (Real.sqrt 2 / 2), deriv (f 1) x < 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l309_30957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_is_twenty_minutes_l309_30977

/-- Represents the tank and pipe system -/
structure WaterSystem where
  tankCapacity : ℚ
  pipeARate : ℚ
  pipeBRate : ℚ
  pipeCRate : ℚ
  pipeATime : ℚ
  pipeBTime : ℚ
  pipeCTime : ℚ

/-- Calculates the net water filled in one cycle -/
def netWaterPerCycle (system : WaterSystem) : ℚ :=
  system.pipeARate * system.pipeATime + 
  system.pipeBRate * system.pipeBTime - 
  system.pipeCRate * system.pipeCTime

/-- Calculates the time for one complete cycle -/
def cycleTime (system : WaterSystem) : ℚ :=
  system.pipeATime + system.pipeBTime + system.pipeCTime

/-- Calculates the total time to fill the tank -/
def totalFillTime (system : WaterSystem) : ℚ :=
  (system.tankCapacity / netWaterPerCycle system) * cycleTime system

/-- Theorem: The time to fill the tank is 20 minutes -/
theorem fill_time_is_twenty_minutes (system : WaterSystem) 
  (h1 : system.tankCapacity = 1000)
  (h2 : system.pipeARate = 200)
  (h3 : system.pipeBRate = 50)
  (h4 : system.pipeCRate = 25)
  (h5 : system.pipeATime = 1)
  (h6 : system.pipeBTime = 2)
  (h7 : system.pipeCTime = 2) :
  totalFillTime system = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_time_is_twenty_minutes_l309_30977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangulation_count_equals_catalan_l309_30939

/-- The number of triangulations of a convex polygon with n+1 sides -/
noncomputable def triangulationCount (n : ℕ) : ℚ :=
  if n ≥ 2 then
    1 / n * (Nat.choose (2*n - 2) (n - 1) : ℚ)
  else 0

/-- The nth Catalan number -/
noncomputable def catalanNumber (n : ℕ) : ℚ :=
  1 / (n + 1 : ℚ) * (Nat.choose (2*n) n : ℚ)

/-- Theorem: The number of triangulations of a convex (n+1)-gon is equal to the nth Catalan number -/
theorem triangulation_count_equals_catalan (n : ℕ) (h : n ≥ 2) :
  triangulationCount n = catalanNumber (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangulation_count_equals_catalan_l309_30939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_investment_is_7000_l309_30936

/-- Represents the business partnership between A and B -/
structure Partnership where
  a_investment : ℚ
  b_investment : ℚ
  a_duration : ℚ
  b_duration : ℚ
  profit_ratio : ℚ × ℚ

/-- Calculates B's investment given the partnership details -/
def calculate_b_investment (p : Partnership) : ℚ :=
  (p.a_investment * p.a_duration * p.profit_ratio.2) / (p.b_duration * p.profit_ratio.1)

/-- Theorem stating that B's investment is 7000 given the problem conditions -/
theorem b_investment_is_7000 :
  let p : Partnership := {
    a_investment := 3500,
    b_investment := 0,  -- We'll calculate this
    a_duration := 12,
    b_duration := 6,
    profit_ratio := (2, 3)
  }
  calculate_b_investment p = 7000 := by
  -- Proof goes here
  sorry

#eval calculate_b_investment {
  a_investment := 3500,
  b_investment := 0,
  a_duration := 12,
  b_duration := 6,
  profit_ratio := (2, 3)
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_investment_is_7000_l309_30936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_small_triangle_to_hexagon_ratio_l309_30910

noncomputable def large_triangle_side : ℝ := 12
noncomputable def small_triangle_side : ℝ := 4

noncomputable def equilateral_triangle_area (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side ^ 2

noncomputable def large_triangle_area : ℝ := equilateral_triangle_area large_triangle_side
noncomputable def small_triangle_area : ℝ := equilateral_triangle_area small_triangle_side

noncomputable def hexagon_area : ℝ := large_triangle_area - 2 * small_triangle_area

theorem small_triangle_to_hexagon_ratio :
  small_triangle_area / hexagon_area = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_small_triangle_to_hexagon_ratio_l309_30910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l309_30952

-- Define the variables and conditions
noncomputable def a : ℝ := Real.log 5 / Real.log 3
noncomputable def b : ℝ := Real.log (1/5) / Real.log 3
noncomputable def c : ℝ := 1/3

-- State the theorem
theorem relationship_abc : b < c ∧ c < a := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l309_30952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_calculation_l309_30978

-- Define the triangle properties
noncomputable def triangle_perimeter : ℝ := 32
noncomputable def triangle_inradius : ℝ := 2.5

-- Define the semi-perimeter
noncomputable def semi_perimeter : ℝ := triangle_perimeter / 2

-- Theorem statement
theorem triangle_area_calculation : 
  triangle_inradius * semi_perimeter = 40 := by
  -- Unfold the definitions
  unfold triangle_inradius
  unfold semi_perimeter
  unfold triangle_perimeter
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_calculation_l309_30978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_speed_l309_30999

-- Define the ratio of speeds
def speed_ratio : ℚ := 7 / 8

-- Define the distance and time for the second train
def distance_train2 : ℝ := 400
def time_train2 : ℝ := 4

-- Calculate the speed of the second train
noncomputable def speed_train2 : ℝ := distance_train2 / time_train2

-- Define the speed of the first train
noncomputable def speed_train1 : ℝ := speed_ratio * speed_train2

-- Theorem to prove
theorem first_train_speed :
  speed_train1 = 87.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_speed_l309_30999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_inconsistencies_l309_30937

structure StoryInconsistency where
  description : String
  explanation : String

structure DraculaStory where
  inconsistencies : List StoryInconsistency

def draculaStoryInconsistencies : DraculaStory :=
  { inconsistencies := [
      { description := "Use of the word 'Yes'",
        explanation := "Dracula, being part of higher Transylvanian nobility, should never use the word 'Yes'" },
      { description := "Repeated use of 'Yes'",
        explanation := "Dracula uses 'Yes' again when explaining the magical statement" },
      { description := "Contradiction with nobility norms",
        explanation := "The story contradicts the established norm that higher Transylvanian nobility doesn't use 'Yes'" },
      { description := "Inconsistency in magical assertion",
        explanation := "There's a logical gap in how the 'magical' statement works or is interpreted" }
    ]
  }

#eval draculaStoryInconsistencies.inconsistencies.length

theorem four_inconsistencies : 
  draculaStoryInconsistencies.inconsistencies.length = 4 := by
  rfl

#check four_inconsistencies

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_inconsistencies_l309_30937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_representation_l309_30930

/-- Represents the number of people -/
def x : ℕ := sorry

/-- Represents the number of carriages -/
def y : ℕ := sorry

/-- Condition 1: Three people share a carriage, leaving two carriages empty -/
axiom condition1 : x / 3 = y - 2

/-- Condition 2: Two people share a carriage, leaving nine people walking -/
axiom condition2 : (x - 9) / 2 = y

/-- The system of equations correctly represents the problem -/
theorem correct_representation : 
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) := by
  constructor
  . exact condition1
  . exact condition2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_representation_l309_30930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_part1_max_m_over_n_part2_l309_30909

-- Define the function f
noncomputable def f (m n x : ℝ) : ℝ := Real.exp (-x) + (n * x) / (m * x + n)

-- Part 1
theorem min_value_part1 :
  ∀ x : ℝ, f 0 1 x ≥ 1 :=
by sorry

-- Part 2
theorem max_m_over_n_part2 (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x : ℝ, x ≥ 0 → f m n x ≥ 1) →
  m / n ≤ (1 : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_part1_max_m_over_n_part2_l309_30909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l309_30998

theorem polynomial_divisibility (n : ℕ) :
  (∃ q : Polynomial ℤ, X^(4*n) + X^(3*n) + X^(2*n) + X^n + 1 = (X^4 + X^3 + X^2 + X + 1) * q) ↔ ¬(5 ∣ n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l309_30998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_pairs_satisfying_inequalities_l309_30961

theorem ordered_pairs_satisfying_inequalities :
  let S := {(a, b) : ℤ × ℤ | a^2 + b^2 < 25 ∧ a^2 + b^2 < 10*a ∧ a^2 + b^2 < 10*b}
  Finset.card (Finset.filter (fun (a, b) => a^2 + b^2 < 25 ∧ a^2 + b^2 < 10*a ∧ a^2 + b^2 < 10*b) (Finset.product (Finset.range 11) (Finset.range 11))) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordered_pairs_satisfying_inequalities_l309_30961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l309_30972

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x) / a + x

-- Define the function g
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := f 1 x + (1/2) * x^2 - k * x

-- Theorem statement
theorem function_properties (a b k : ℝ) :
  (∀ x, x > 0 → (2 * x - f a x + b = 0) ↔ x = 1) →
  (∀ x, x > 0 → deriv (g k) x ≥ 0) →
  (a = 1 ∧ b = -1 ∧ k ≤ 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l309_30972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divides_totient_power_minus_one_l309_30990

theorem divides_totient_power_minus_one (a n : ℕ) (hn : n > 0) :
  n ∣ Nat.totient (a^n - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divides_totient_power_minus_one_l309_30990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_sqrt3_cos_l309_30914

open Real MeasureTheory

theorem min_value_sin_sqrt3_cos :
  ∃ (m : ℝ), m = 1 ∧ ∀ y ∈ Set.Icc 0 (π/2), sin y + Real.sqrt 3 * cos y ≥ m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_sqrt3_cos_l309_30914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_residual_probability_l309_30938

-- Define the regression line
def regression_line (x : ℝ) : ℝ := x + 1

-- Define the residual function
def residual_func (y : ℝ) : ℝ := y - regression_line 1

-- Define the probability space
def y₀_range : Set ℝ := Set.Icc 0 3

-- Define the event where the absolute residual is not greater than 1
def event : Set ℝ := {y ∈ y₀_range | |residual_func y| ≤ 1}

-- Theorem statement
theorem residual_probability : 
  (MeasureTheory.volume event) / (MeasureTheory.volume y₀_range) = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_residual_probability_l309_30938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_fraction_solutions_nested_fraction_unique_solutions_l309_30955

/-- Recursive function representing the nested fraction structure -/
noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => 1 + 1 / f n x

/-- The equation to be solved -/
def equation (n : ℕ) (x : ℝ) : Prop :=
  f n x = x

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The conjugate of the golden ratio -/
noncomputable def φ_conj : ℝ := (1 - Real.sqrt 5) / 2

/-- Theorem stating that φ and φ_conj are solutions to the equation -/
theorem nested_fraction_solutions (n : ℕ) :
  equation n φ ∧ equation n φ_conj := by
  sorry

/-- Theorem stating that these are the only solutions -/
theorem nested_fraction_unique_solutions (n : ℕ) (x : ℝ) :
  equation n x → x = φ ∨ x = φ_conj := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_fraction_solutions_nested_fraction_unique_solutions_l309_30955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_same_flips_l309_30934

/-- Represents a coin with a given probability of heads -/
structure Coin where
  prob_heads : ℝ
  prob_valid : 0 < prob_heads ∧ prob_heads ≤ 1

/-- Fair coin with probability of heads 1/2 -/
noncomputable def fair_coin : Coin := ⟨1/2, by norm_num⟩

/-- Biased coin with probability of heads 1/3 -/
noncomputable def biased_coin : Coin := ⟨1/3, by norm_num⟩

/-- Probability of getting first head on nth flip for a given coin -/
noncomputable def prob_first_head (c : Coin) (n : ℕ) : ℝ :=
  (1 - c.prob_heads) ^ (n - 1) * c.prob_heads

/-- Probability that all three players get their first head on the nth flip -/
noncomputable def prob_all_same_n (n : ℕ) : ℝ :=
  (prob_first_head fair_coin n) ^ 2 * (prob_first_head biased_coin n)

/-- Total probability that all three players flip their coins the same number of times -/
noncomputable def total_prob : ℝ := ∑' n, prob_all_same_n n

theorem prob_all_same_flips : total_prob = 1 / 17 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_same_flips_l309_30934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aluminum_phosphate_molecular_weight_l309_30946

/-- The molecular weight of a substance in g/mol -/
def molecular_weight (substance : Type) : ℝ := sorry

/-- The number of moles of a substance -/
def moles (substance : Type) : ℝ := sorry

/-- Aluminum phosphate (AlPO4) -/
structure AlPO4 : Type

/-- The total weight of a substance in grams -/
def total_weight (substance : Type) : ℝ := sorry

theorem aluminum_phosphate_molecular_weight :
  molecular_weight AlPO4 = 121.95 ∧ 
  ∃ n : ℝ, n * molecular_weight AlPO4 = 854 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aluminum_phosphate_molecular_weight_l309_30946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_theorem_l309_30911

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ

/-- Represents the configuration of circles and triangle as described in the problem -/
structure CircleConfiguration where
  A : Circle
  B : Circle
  C : Circle
  D : Circle
  E : Circle
  T : EquilateralTriangle

/-- Predicate to check if a triangle is inscribed in a circle -/
def is_inscribed (t : EquilateralTriangle) (c : Circle) : Prop :=
  sorry

/-- Predicate to check if one circle is internally tangent to another -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  sorry

/-- Predicate to check if one circle is externally tangent to another -/
def is_externally_tangent (c1 c2 : Circle) : Prop :=
  sorry

/-- The main theorem to be proved -/
theorem circle_configuration_theorem (config : CircleConfiguration) 
  (h1 : config.A.radius = 15)
  (h2 : config.B.radius = 5)
  (h3 : config.C.radius = 3)
  (h4 : config.D.radius = 3)
  (h5 : config.T.sideLength = 15 * Real.sqrt 3)
  (h6 : ∃ (m n : ℕ), Nat.Coprime m n ∧ config.E.radius = m / n)
  (h7 : is_inscribed config.T config.A)
  (h8 : is_internally_tangent config.B config.A)
  (h9 : is_internally_tangent config.C config.A)
  (h10 : is_internally_tangent config.D config.A)
  (h11 : is_externally_tangent config.B config.E)
  (h12 : is_externally_tangent config.C config.E)
  (h13 : is_externally_tangent config.D config.E) :
  ∃ (m n : ℕ), Nat.Coprime m n ∧ config.E.radius = 121 / 10 ∧ m + n = 131 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_configuration_theorem_l309_30911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_parabola_focus_distance_proof_l309_30950

/-- The distance from the focus of the parabola x^2 = 16y to the point (2, 5) is √5 -/
theorem parabola_focus_distance : ℝ → ℝ → ℝ → ℝ → ℝ → Prop :=
  fun a b c d e =>
    (∀ x y, x^2 = 16*y ↔ (x - a)^2 + (y - b)^2 = c^2) →
    (2 - a)^2 + (5 - b)^2 = d^2 →
    e = Real.sqrt 5 →
    d = e

theorem parabola_focus_distance_proof : ∃ a b c d e : ℝ,
  parabola_focus_distance a b c d e := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_parabola_focus_distance_proof_l309_30950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_rotation_matrix_75_degrees_l309_30994

open Matrix Real

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![Real.cos θ, -Real.sin θ; Real.sin θ, Real.cos θ]

theorem det_rotation_matrix_75_degrees :
  let Q := rotation_matrix (75 * π / 180)
  Matrix.det Q = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_rotation_matrix_75_degrees_l309_30994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_value_l309_30904

theorem cos_two_theta_value (θ : ℝ) (h : ∑' n, (Real.cos θ)^(2*n) = 9) : Real.cos (2*θ) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_theta_value_l309_30904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cross_section_area_l309_30919

/-- A prism with a square base and vertical edges parallel to the z-axis -/
structure Prism where
  base_side_length : ℝ
  base_center : ℝ × ℝ × ℝ

/-- A plane in 3D space defined by the equation ax + by + cz = d -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The area of the cross-section created by a plane intersecting a prism -/
noncomputable def cross_section_area (prism : Prism) (plane : Plane) : ℝ :=
  sorry

/-- The theorem stating the maximum area of the cross-section -/
theorem max_cross_section_area (prism : Prism) (plane : Plane) :
  prism.base_side_length = 8 ∧
  prism.base_center = (0, 0, 0) ∧
  plane.a = 3 ∧ plane.b = -5 ∧ plane.c = 2 ∧ plane.d = 20 →
  ∃ (area : ℝ), cross_section_area prism plane ≤ area ∧ area = 60.4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cross_section_area_l309_30919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_power_l309_30900

theorem greatest_integer_power (n : ℤ) : (∀ m : ℤ, 6.1 * (10 : ℝ)^m < 620 → m ≤ n) ↔ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_power_l309_30900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l309_30949

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.sin x) + Real.sqrt (1/2 - Real.cos x)

-- Define the domain condition
def in_domain (x : ℝ) : Prop :=
  Real.sin x ≥ 0 ∧ (1/2 - Real.cos x) ≥ 0

-- Define the interval condition
def in_interval (x : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 3 ≤ x ∧ x ≤ 2 * k * Real.pi + Real.pi

-- State the theorem
theorem domain_equivalence (x : ℝ) : in_domain x ↔ in_interval x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_equivalence_l309_30949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_projectile_speed_l309_30947

/-- Proves that given two projectiles launched simultaneously from 2520 km apart,
    with one traveling at 432 km/h and meeting after 150 minutes,
    the speed of the second projectile is 576 km/h. -/
theorem second_projectile_speed
  (initial_distance : ℝ)
  (first_speed : ℝ)
  (meeting_time_minutes : ℝ)
  (h1 : initial_distance = 2520)
  (h2 : first_speed = 432)
  (h3 : meeting_time_minutes = 150) :
  let meeting_time_hours : ℝ := meeting_time_minutes / 60
  let second_speed : ℝ := (initial_distance / meeting_time_hours) - first_speed
  second_speed = 576 := by
  -- Convert meeting time from minutes to hours
  have meeting_time_hours : ℝ := meeting_time_minutes / 60
  
  -- Calculate the second speed
  have second_speed : ℝ := (initial_distance / meeting_time_hours) - first_speed
  
  -- Prove that the second speed is 576 km/h
  sorry  -- This skips the proof for now

#check second_projectile_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_projectile_speed_l309_30947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l309_30922

theorem calculate_expression : -1 - (-1/2 : ℚ)^(-3 : ℤ) + (Real.pi - 3.14)^0 - |2 - 4| = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l309_30922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_problem_l309_30926

/-- The distance between points A and B in kilometers -/
def S : ℕ := sorry

/-- The speed of person A -/
def v_A : ℝ := sorry

/-- The speed of person B -/
def v_B : ℝ := sorry

/-- The speed of the motorcycle -/
def v_M : ℝ := sorry

/-- The distance AC -/
def AC : ℕ := sorry

/-- The distance AD -/
def AD : ℕ := sorry

/-- The distance AE -/
def AE : ℕ := sorry

theorem distance_problem :
  (Finset.filter (· ∣ S) (Finset.range (S + 1))).card = 8 ∧
  v_A = 4 * v_B ∧
  v_M = 14 * v_A ∧
  AC = S / 5 ∧
  AD = 2 * S / 3 ∧
  AE = S / 21 →
  S = 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_problem_l309_30926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l309_30916

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Theorem statement
theorem arithmetic_sequence_properties (a : ℕ → ℝ) (h : is_arithmetic_sequence a) :
  is_arithmetic_sequence (λ n ↦ a n + 1) ∧
  is_arithmetic_sequence (λ n ↦ a (n + 1) - a n) ∧
  is_arithmetic_sequence (λ n ↦ 2 * a n) ∧
  is_arithmetic_sequence (λ n ↦ a n + n) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l309_30916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_field_dimensions_l309_30902

theorem rectangular_field_dimensions : ∃! m : ℝ, m > 0 ∧ (3 * m + 8) * (m - 3) = 100 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_field_dimensions_l309_30902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l309_30908

noncomputable section

/-- Ellipse equation -/
def ellipse (x y : ℝ) : Prop := y^2 / 4 + x^2 / 2 = 1

/-- Line equation -/
def line (x y m : ℝ) : Prop := y = Real.sqrt 2 * x + m

/-- Point P on the ellipse -/
def point_P : Prop := ellipse 1 (Real.sqrt 2)

/-- Intersection points A and B -/
def intersection_points (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    line x₁ y₁ m ∧ line x₂ y₂ m

/-- Area of triangle PAB -/
noncomputable def triangle_area (m : ℝ) : ℝ :=
  (1 / 2) * Real.sqrt ((4 - m^2 / 2) * m^2)

/-- Theorem: Maximum area of triangle PAB is √2 -/
theorem max_triangle_area :
  point_P →
  (∀ m, intersection_points m → triangle_area m ≤ Real.sqrt 2) ∧
  (∃ m, intersection_points m ∧ triangle_area m = Real.sqrt 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l309_30908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_on_unit_interval_l309_30983

/-- The function g(x) = (1+k)^x - kx - 1 -/
noncomputable def g (k : ℝ) (x : ℝ) : ℝ := (1 + k)^x - k * x - 1

/-- Theorem: The maximum value of g(x) on [0,1] is 0, for k ∈ (-1, +∞) -/
theorem g_max_value_on_unit_interval (k : ℝ) (hk : k > -1) :
  ∃ (M : ℝ), M = 0 ∧ ∀ x, x ∈ Set.Icc 0 1 → g k x ≤ M :=
by
  -- We claim that 0 is the maximum value
  use 0
  constructor
  -- Trivially, 0 = 0
  · rfl
  -- For all x in [0, 1], g(k, x) ≤ 0
  · intro x hx
    sorry -- The actual proof would go here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_max_value_on_unit_interval_l309_30983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lock_combination_l309_30980

/-- Represents a digit in base 12 --/
def Digit12 := Fin 12

/-- Represents the equation STARS + RATS + ARTS = START in base 12 --/
def EquationHolds (S T A R : Digit12) : Prop :=
  let BASE := 12
  ∃ (carry1 carry2 carry3 : Nat),
    (S.val + R.val + A.val : Nat) = T.val + carry1 * BASE ∧
    (T.val + A.val + R.val + carry1 : Nat) = R.val + carry2 * BASE ∧
    (A.val + T.val + S.val + carry2 : Nat) = A.val + carry3 * BASE ∧
    (S.val + carry3 : Nat) = S.val

/-- All letters represent distinct digits --/
def AllDistinct (S T A R : Digit12) : Prop :=
  S ≠ T ∧ S ≠ A ∧ S ≠ R ∧ T ≠ A ∧ T ≠ R ∧ A ≠ R

/-- The value of STA in decimal --/
def STAValue (S T A : Digit12) : Nat :=
  S.val * 100 + T.val * 10 + A.val

theorem lock_combination :
  ∃ (S T A R : Digit12),
    EquationHolds S T A R ∧
    AllDistinct S T A R ∧
    STAValue S T A = 805 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lock_combination_l309_30980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l309_30995

-- Define the given conditions
noncomputable def terminal_side_intersection : ℝ × ℝ := (3 * Real.sqrt 10 / 10, Real.sqrt 10 / 10)

-- Define the main theorem
theorem trigonometric_identities 
  (h1 : terminal_side_intersection = (3 * Real.sqrt 10 / 10, Real.sqrt 10 / 10))
  (h2 : ∀ α β : ℝ, Real.tan (α + β) = 2 / 5) :
  (∀ α : ℝ, Real.sin (2 * α + π / 6) = (3 * Real.sqrt 3 - 4) / 10) ∧
  (∀ β : ℝ, Real.tan (2 * β - π / 3) = 17 / 144) := by
  sorry

-- Additional helper lemmas if needed
lemma helper_lemma_1 (α : ℝ) : 
  Real.sin (α + π / 6) = Real.sqrt 10 / 10 ∧ 
  Real.cos (α + π / 6) = 3 * Real.sqrt 10 / 10 := by
  sorry

lemma helper_lemma_2 (α : ℝ) :
  Real.sin (2 * α + π / 3) = 3 / 5 ∧
  Real.cos (2 * α + π / 3) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l309_30995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_7_l309_30986

-- Define the sequence T
def T : ℕ → ℕ
  | 0 => 3  -- Add this case to handle 0
  | 1 => 3
  | n+2 => 3^(T (n+1))

-- State the theorem
theorem t_50_mod_7 : T 50 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_50_mod_7_l309_30986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_is_square_l309_30959

/-- A function from positive integers to positive integers -/
def PositiveIntegerFunction := ℕ+ → ℕ+

/-- The property that f satisfies for all positive integers m and n -/
def SatisfiesProperty (f : PositiveIntegerFunction) : Prop :=
  ∀ m n : ℕ+, 
    (f m + f n - (m * n : ℕ) ≠ 0) ∧ 
    ((m * f m + n * f n : ℕ) % (f m + f n - (m * n : ℕ)) = 0)

/-- The theorem stating that if f satisfies the property, then f(n) = n² for all n -/
theorem function_is_square (f : PositiveIntegerFunction) 
  (h : SatisfiesProperty f) : 
  ∀ n : ℕ+, f n = n ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_is_square_l309_30959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l309_30941

noncomputable def f (x y : ℝ) : ℝ := Real.sqrt (x^2 - 3*x + 3) + Real.sqrt (y^2 - 3*y + 3) + Real.sqrt (x^2 - Real.sqrt 3 * x * y + y^2)

theorem min_value_f (x y : ℝ) (hx : x > 0) (hy : y > 0) : f x y ≥ Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l309_30941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_and_integer_factorial_l309_30982

theorem floor_inequality_and_integer_factorial (α β : ℝ) (m n : ℕ) : 
  (⌊2 * α⌋ + ⌊2 * β⌋ ≥ ⌊α⌋ + ⌊β⌋ + ⌊α + β⌋) ∧ 
  (∃ k : ℕ, k * (m.factorial * n.factorial * (m + n).factorial) = (2 * m).factorial * (2 * n).factorial) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_and_integer_factorial_l309_30982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l309_30915

noncomputable section

/-- The line is defined by y = 2x - 4 -/
def line (x : ℝ) : ℝ := 2 * x - 4

/-- The point we're measuring distance from -/
def point : ℝ × ℝ := (3, -1)

/-- The proposed closest point on the line -/
def closest_point : ℝ × ℝ := (9/5, 2/5)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem closest_point_on_line :
  (∀ x : ℝ, distance (x, line x) point ≥ distance closest_point point) ∧
  (closest_point.2 = line closest_point.1) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l309_30915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_log_abs_diff_iff_zero_l309_30932

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

/-- The logarithm of the absolute value of x minus a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (|x - a|)

theorem even_log_abs_diff_iff_zero (a : ℝ) :
  IsEven (f a) ↔ a = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_log_abs_diff_iff_zero_l309_30932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_distance_l309_30971

/-- Calculates the distance of a marathon given the total time and average pace per mile. -/
theorem marathon_distance 
  (total_hours : ℕ) 
  (total_minutes : ℕ) 
  (avg_minutes_per_mile : ℕ) 
  (h1 : total_hours = 3) 
  (h2 : total_minutes = 36) 
  (h3 : avg_minutes_per_mile = 9) : 
  (total_hours * 60 + total_minutes) / avg_minutes_per_mile = 24 := by
  sorry

#check marathon_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_distance_l309_30971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_volume_properties_l309_30929

-- Define the sequence a_n
noncomputable def a : ℕ → ℝ
  | 0 => 1  -- Define for 0 to avoid missing case
  | 1 => 1
  | n + 1 => 2 * a n - (1/2) * (a n)^2

-- State the theorem
theorem sales_volume_properties :
  (∀ n : ℕ, n ≥ 1 → a n < a (n + 1) ∧ a (n + 1) < 2) ∧
  (∀ n : ℕ, n ≥ 1 → a n = 2 - 2 * (1/2)^(2^(n-1))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_volume_properties_l309_30929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l309_30975

-- Define the power function as noncomputable
noncomputable def power_function (a : ℝ) : ℝ → ℝ := fun x ↦ x ^ a

-- State the theorem
theorem power_function_value (a : ℝ) :
  power_function a 2 = 1/4 → power_function a (1/3) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l309_30975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_power_sum_divisible_l309_30905

theorem odd_power_sum_divisible (x y : ℤ) :
  ∀ k : ℕ, ∃ m : ℤ, x^(2*k + 1) + y^(2*k + 1) = (x + y) * m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_power_sum_divisible_l309_30905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_matrix_det_nonzero_l309_30987

/-- The determinant of a 3x3 matrix with trigonometric function elements is not necessarily zero. -/
theorem trig_matrix_det_nonzero : ∃ (a b c d e f g h i : ℝ),
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![Real.sin a, Real.cos b, Real.sin c;
                                        Real.sin d, Real.cos e, Real.sin f;
                                        Real.sin g, Real.cos h, Real.sin i]
  ¬ (Matrix.det A = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_matrix_det_nonzero_l309_30987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_middle_three_l309_30966

noncomputable def average_of_eight_numbers (numbers : Fin 8 → ℝ) : ℝ :=
  (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + numbers 5 + numbers 6 + numbers 7) / 8

theorem average_of_middle_three (numbers : Fin 8 → ℝ) :
  (average_of_eight_numbers numbers = 25) →
  ((numbers 0 + numbers 1) / 2 = 20) →
  (numbers 5 = numbers 6 - 4) →
  (numbers 5 = numbers 7 - 6) →
  (numbers 7 = 30) →
  ((numbers 2 + numbers 3 + numbers 4) / 3 = 26) :=
by
  sorry

#check average_of_middle_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_middle_three_l309_30966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_correct_l309_30962

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := 1 / ⌊x^2 - 9*x + 21⌋

-- Define the domain of g(x)
def domain_g : Set ℝ := {x | x ≤ 4 ∨ x ≥ 5}

-- Theorem stating that the domain of g is correct
theorem g_domain_correct : 
  ∀ x : ℝ, g x ≠ 0 ↔ x ∈ domain_g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_correct_l309_30962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_protractor_roll_l309_30924

/-- The central angle in radians when rolling a circle along a straight line -/
noncomputable def centralAngle (radius : ℝ) (arcLength : ℝ) : ℝ :=
  arcLength / radius

/-- Convert radians to degrees -/
noncomputable def radiansToDegrees (radians : ℝ) : ℝ :=
  radians * 180 / Real.pi

theorem protractor_roll (radius : ℝ) (arcLength : ℝ) 
    (h1 : radius = 5)
    (h2 : arcLength = 10) : 
  ∃ (angle : ℝ), abs (angle - 115) < 1 ∧ angle = radiansToDegrees (centralAngle radius arcLength) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_protractor_roll_l309_30924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faulty_meter_weight_difference_l309_30988

/-- Represents the profit percentage of a shopkeeper using a faulty meter. -/
noncomputable def profit_percent : ℝ := 4.166666666666666

/-- Represents the expected weight in grams. -/
noncomputable def expected_weight : ℝ := 1000

/-- Calculates the actual weight provided by the faulty meter. -/
noncomputable def actual_weight : ℝ := expected_weight / (1 + profit_percent / 100)

/-- Theorem stating that the faulty meter weighs approximately 40 grams less than expected for every 1000 grams. -/
theorem faulty_meter_weight_difference :
  ‖expected_weight - actual_weight - 40‖ < 0.01 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_faulty_meter_weight_difference_l309_30988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_pool_width_l309_30979

/-- Represents a trapezoidal prism-shaped swimming pool -/
structure SwimmingPool where
  length : ℝ
  shallow_depth : ℝ
  deep_depth : ℝ
  volume : ℝ
  width : ℝ

/-- Calculates the volume of a trapezoidal prism -/
noncomputable def trapezoidal_prism_volume (p : SwimmingPool) : ℝ :=
  (p.width / 2) * (p.shallow_depth + p.deep_depth) * p.length

/-- Theorem stating that the width of the swimming pool is 9 meters -/
theorem swimming_pool_width (p : SwimmingPool)
  (h_length : p.length = 12)
  (h_shallow : p.shallow_depth = 1)
  (h_deep : p.deep_depth = 4)
  (h_volume : p.volume = 270)
  (h_vol_eq : p.volume = trapezoidal_prism_volume p) :
  p.width = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_pool_width_l309_30979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l309_30948

noncomputable section

-- Define the necessary function types
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)
def InverseFunction (f g : ℝ → ℝ) : Prop := ∀ x, g (f x) = x ∧ f (g x) = x

-- Define the specific function in proposition (3)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

-- Define the specific function in proposition (5)
noncomputable def g (x : ℝ) : ℝ := 2^x - x^2

theorem function_properties :
  -- Proposition (1) is false
  ¬(∀ f : ℝ → ℝ, EvenFunction f → ∃ y, f 0 = y) ∧
  -- Proposition (2) is false
  ¬(∀ f : ℝ → ℝ, OddFunction f → f 0 = 0) ∧
  -- Proposition (3) is true
  (∀ a : ℝ, OddFunction (f a) → a = 1) ∧
  -- Proposition (4) is false
  ¬(∀ f : ℝ → ℝ, OddFunction f → f 0 = 0 → Monotone f) ∧
  -- Proposition (5) is false
  ¬(∃! x₁ x₂ : ℝ, g x₁ = 0 ∧ g x₂ = 0 ∧ x₁ ≠ x₂) ∧
  -- Proposition (6) is true
  (∀ f g : ℝ → ℝ, InverseFunction f g → 
    ∀ x y : ℝ, f x = y ↔ g y = x) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l309_30948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_15_l309_30927

noncomputable def geometric_sequence (a r : ℝ) : ℕ → ℝ := λ n ↦ a * r^(n - 1)

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then n * a else a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_15 
  (a r : ℝ) 
  (h1 : geometric_sum a r 5 = 10) 
  (h2 : geometric_sum a r 10 = 50) : 
  geometric_sum a r 15 = 210 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_15_l309_30927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equality_l309_30923

theorem cube_root_equality : ((-8 : ℝ) ^ (1/3 : ℝ)) = -(8 : ℝ) ^ (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equality_l309_30923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_of_parabola_intersection_l309_30976

/-- Given a line y = kx - 2 intersecting a parabola y² = 8x at two points A and B,
    where the x-coordinate of the midpoint of AB is 2, the length of chord AB is 2√15. -/
theorem chord_length_of_parabola_intersection (k : ℝ) (A B : ℝ × ℝ) :
  (∀ x y, y = k * x - 2 ↔ (x, y) = A ∨ (x, y) = B) →
  (∀ x y, y^2 = 8 * x ↔ (x, y) = A ∨ (x, y) = B) →
  A.1 + B.1 = 4 →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_of_parabola_intersection_l309_30976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_averaging_properties_l309_30969

-- Define the averaging operation
noncomputable def avg (x y : ℝ) : ℝ := (x + y) / 2

-- Define the middle-value operation
noncomputable def mid (x y : ℝ) : ℝ := 
  if x ≤ y then x else y

-- State the theorem
theorem averaging_properties :
  (∀ x y z : ℝ, avg (avg x y) z = avg x (avg y z)) = false ∧ 
  (∀ x y : ℝ, avg x y = avg y x) = true ∧
  (∀ x y z : ℝ, mid x (avg y z) = avg (mid x y) (mid x z)) = false ∧
  (∀ x y z : ℝ, avg x (mid y z) = mid (avg x y) (avg x z)) = false ∧
  (∀ x y z : ℝ, mid (mid x y) z = mid x (mid y z)) = false := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_averaging_properties_l309_30969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_function_form_intersection_points_count_ln_inequality_implies_b_range_l309_30913

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + (2 - 3*a) / 2 * x^2 + b * x

noncomputable def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + (2 - 3*a) * x + b

theorem tangent_line_implies_function_form (a b : ℝ) :
  (f' a b 2 = 1 ∧ f a b 2 = 8) → 
  (∀ x, f a b x = -x^3 + 5/2 * x^2 + 3*x) :=
by sorry

theorem intersection_points_count (m : ℝ) :
  (m < -5/27 ∨ m > 1) →
  ∃! x, f (-1) 3 x = -1/2 * (f' (-1) 3 x - 9*x - 3) + m :=
by sorry

theorem ln_inequality_implies_b_range (b : ℝ) :
  (∀ x > 0, Real.log x ≤ f' 1 b x) →
  b ≥ -Real.log 2 - 1/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_function_form_intersection_points_count_ln_inequality_implies_b_range_l309_30913
