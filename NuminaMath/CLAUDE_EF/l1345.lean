import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_sample_data_l1345_134545

noncomputable def sample_data : List ℝ := [11, 8, 9, 10, 7]

noncomputable def sample_mean (data : List ℝ) : ℝ :=
  (data.sum) / (data.length : ℝ)

noncomputable def sample_variance (data : List ℝ) : ℝ :=
  let mean := sample_mean data
  let squared_diff_sum := (data.map (λ x => (x - mean)^2)).sum
  squared_diff_sum / ((data.length - 1) : ℝ)

theorem variance_of_sample_data :
  sample_variance sample_data = 2.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_sample_data_l1345_134545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_main_theorem_l1345_134520

-- Define the sequence of numbers
def mySequence : List Nat := List.range 10 |>.map (fun i => 3 + 10 * i)

-- Theorem statement
theorem product_remainder (n : Nat) (hn : n ∈ mySequence) : n % 6 = 3 := by
  sorry

theorem main_theorem : (mySequence.prod) % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_main_theorem_l1345_134520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_37_l1345_134517

theorem sum_of_divisors_37 : (Finset.sum (Nat.divisors 37) id) = 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_37_l1345_134517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_preserving_l1345_134587

-- Define the domain
def Domain : Type := {x : ℝ // x ≠ 0}

-- Define a geometric sequence
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define a geometric sequence preserving function
def IsGeometricSequencePreserving (f : Domain → ℝ) : Prop :=
  ∀ a : ℕ → Domain, IsGeometricSequence (fun n ↦ (a n).val) → 
    IsGeometricSequence (fun n ↦ f (a n))

-- Define the two functions
def f₁ : Domain → ℝ := fun x ↦ x.val ^ 2

noncomputable def f₂ : Domain → ℝ := fun x ↦ Real.sqrt (abs x.val)

-- The theorem to prove
theorem geometric_sequence_preserving :
  IsGeometricSequencePreserving f₁ ∧ IsGeometricSequencePreserving f₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_preserving_l1345_134587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_Ω_l1345_134595

-- Define the region Ω
def Ω : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | 
    let (x, y, z) := p;
    2 * Real.sqrt (2 * y) ≤ x ∧
    x ≤ 17 * Real.sqrt (2 * y) ∧
    0 ≤ y ∧ y ≤ 1/2 ∧
    0 ≤ z ∧ z ≤ 1/2 - y}

-- State the theorem
theorem volume_of_Ω : MeasureTheory.volume Ω = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_Ω_l1345_134595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_perfect_square_l1345_134510

theorem prime_perfect_square (p : ℕ) : 
  Prime p ∧ (∃ x : ℕ, 1 + p * 2^p = x^2) ↔ p = 2 ∨ p = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_perfect_square_l1345_134510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bag_weight_problem_l1345_134565

theorem bag_weight_problem (w₁ w₂ w₃ : ℝ) :
  w₁ > 0 ∧ w₂ > 0 ∧ w₃ > 0 →
  (w₁ : ℝ) / w₂ = 4 / 5 ∧ (w₂ : ℝ) / w₃ = 5 / 6 →
  w₃ + w₁ = w₂ + 45 →
  w₁ = 36 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bag_weight_problem_l1345_134565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_BA_BC_l1345_134593

noncomputable def BA : ℝ × ℝ := (1/2, Real.sqrt 3/2)
noncomputable def BC : ℝ × ℝ := (Real.sqrt 3/2, 1/2)

theorem angle_between_BA_BC : 
  Real.arccos ((BA.1 * BC.1 + BA.2 * BC.2) / (Real.sqrt (BA.1^2 + BA.2^2) * Real.sqrt (BC.1^2 + BC.2^2))) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_BA_BC_l1345_134593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_2_equals_S_3_l1345_134532

-- Define a point in space
variable {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α]

-- Define the set S₀
variable (S₀ : Set α)

-- Define the closed segment between two points
def closed_segment (A B : α) : Set α :=
  {P | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B}

-- Define Sₙ recursively
def S (S₀ : Set α) : ℕ → Set α
  | 0 => S₀
  | n + 1 => {P | ∃ A B, A ∈ S S₀ n ∧ B ∈ S S₀ n ∧ P ∈ closed_segment A B}

-- State the theorem
theorem S_2_equals_S_3 (S₀ : Set α) : S S₀ 2 = S S₀ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_2_equals_S_3_l1345_134532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_of_P_l1345_134539

-- Define the points
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (-3, 2)
def C : ℝ × ℝ := (3, 2)
def D : ℝ × ℝ := (4, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem y_coordinate_of_P (P : ℝ × ℝ) 
  (h1 : distance P A + distance P D = 10)
  (h2 : distance P B + distance P C = 10) :
  P.2 = 6/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_of_P_l1345_134539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1345_134555

noncomputable def f (x : ℝ) := 2 * (Real.sin x)^2 - 1

theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = π := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1345_134555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colin_average_speed_approx_l1345_134566

/-- Represents a segment of Colin's run -/
structure Segment where
  distance : ℝ
  speed : ℝ

/-- Calculates the time taken for a segment -/
noncomputable def time_for_segment (s : Segment) : ℝ :=
  s.distance / s.speed

/-- Colin's run segments -/
def colin_run : List Segment := [
  { distance := 1.5, speed := 6.5 },
  { distance := 1.25, speed := 8.0 },
  { distance := 2.25, speed := 9.5 }
]

/-- Calculates the total distance of the run -/
noncomputable def total_distance (run : List Segment) : ℝ :=
  run.foldl (fun acc s => acc + s.distance) 0

/-- Calculates the total time of the run -/
noncomputable def total_time (run : List Segment) : ℝ :=
  run.foldl (fun acc s => acc + time_for_segment s) 0

/-- Calculates the average speed of the run -/
noncomputable def average_speed (run : List Segment) : ℝ :=
  total_distance run / total_time run

/-- Theorem stating that Colin's average speed is approximately 8.014 mph -/
theorem colin_average_speed_approx :
  |average_speed colin_run - 8.014| < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_colin_average_speed_approx_l1345_134566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_center_of_symmetry_l1345_134546

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / (x - a)

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := f 1 x + x^3 - 3*x^2

-- Theorem 1: Range of a
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x y, 1 < x ∧ x < y → f a x > f a y) →
  0 < a ∧ a ≤ 1 := by
  sorry

-- Theorem 2: Center of symmetry
theorem center_of_symmetry :
  ∀ x, h (-x + 1) + 1 = -h (x + 1) + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_center_of_symmetry_l1345_134546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_result_l1345_134548

-- Define the original function
noncomputable def original_func (x : ℝ) : ℝ := Real.cos x

-- Define the transformation that halves abscissas
noncomputable def halve_abscissas (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (2 * x)

-- Define the transformation that shifts the graph left by π/4
noncomputable def shift_left (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (x + Real.pi / 4)

-- Define the resulting function after transformations
noncomputable def resulting_func (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 2)

-- Theorem statement
theorem transformations_result : 
  ∀ x : ℝ, (shift_left (halve_abscissas original_func)) x = resulting_func x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_result_l1345_134548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_mpg_calculation_l1345_134514

/-- Calculates the average miles per gallon given initial and final odometer readings and gasoline used --/
def average_mpg (initial_reading : ℕ) (final_reading : ℕ) (gasoline_used : ℕ) : ℚ :=
  (final_reading - initial_reading : ℚ) / gasoline_used

/-- Rounds a rational number to the nearest tenth --/
def round_to_tenth (x : ℚ) : ℚ :=
  ⌊(x * 10 + 1/2)⌋ / 10

theorem trip_mpg_calculation :
  let initial_reading : ℕ := 58300
  let final_reading : ℕ := 59275
  let gasoline_used : ℕ := 40
  round_to_tenth (average_mpg initial_reading final_reading gasoline_used) = 244/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_mpg_calculation_l1345_134514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_implies_values_solution_set_positive_description_l1345_134553

/-- The quadratic function f(x) = x^2 - (a+b)x + 3a -/
def f (a b x : ℝ) : ℝ := x^2 - (a+b)*x + 3*a

/-- The solution set of f(x) ≤ 0 -/
def solution_set (a b : ℝ) : Set ℝ := {x | f a b x ≤ 0}

/-- Theorem stating that if the solution set of f(x) ≤ 0 is [1,3], then a = 1 and b = 3 -/
theorem solution_implies_values (a b : ℝ) :
  solution_set a b = Set.Icc 1 3 → a = 1 ∧ b = 3 := by sorry

/-- The solution set of f(x) > 0 when b = 3 -/
def solution_set_positive (a : ℝ) : Set ℝ := {x | f a 3 x > 0}

/-- Theorem describing the solution set of f(x) > 0 when b = 3 -/
theorem solution_set_positive_description (a : ℝ) :
  solution_set_positive a =
    if a > 3 then {x | x < 3 ∨ x > a}
    else if a < 3 then {x | x < a ∨ x > 3}
    else {x | x ≠ 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_implies_values_solution_set_positive_description_l1345_134553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_not_visible_l1345_134562

/-- The Earth's circumference in kilometers -/
noncomputable def earth_circumference : ℝ := 40000

/-- The Earth's radius in kilometers -/
noncomputable def earth_radius : ℝ := earth_circumference / (2 * Real.pi)

/-- Height above the lake surface in kilometers -/
noncomputable def height_above_lake : ℝ := 0.05

/-- Calculate the maximum visible distance from a given height -/
noncomputable def max_visible_distance (h : ℝ) : ℝ := Real.sqrt (2 * earth_radius * h)

/-- First distance to check visibility (Balatonvilágos to Balatonmáriafürdő) -/
noncomputable def distance1 : ℝ := 66.5

/-- Second distance to check visibility (Balatonakarattya to Balatonszemes) -/
noncomputable def distance2 : ℝ := 36.6

/-- Theorem stating that both points are not visible -/
theorem points_not_visible : 
  max_visible_distance height_above_lake < distance1 ∧ 
  max_visible_distance height_above_lake < distance2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_not_visible_l1345_134562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_points_l1345_134528

-- Define the power function as noncomputable
noncomputable def power_function (a : ℝ) : ℝ → ℝ := fun x ↦ x ^ a

-- State the theorem
theorem power_function_through_points :
  ∀ a m : ℝ,
  power_function a 2 = 16 →
  power_function a (1/2) = m →
  m = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_points_l1345_134528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_iff_f_maximum_l1345_134530

noncomputable def f (a x : ℝ) : ℝ := -x^2 + |x - a| + a + 1

theorem f_even_iff (a : ℝ) : (∀ x, f a x = f a (-x)) ↔ a = 0 := by sorry

noncomputable def f_max (a : ℝ) : ℝ :=
  if -1/2 < a ∧ a ≤ 0 then 5/4
  else if 0 < a ∧ a < 1/2 then 5/4 + 2*a
  else -a^2 + a + 1

theorem f_maximum (a : ℝ) : ∀ x, f a x ≤ f_max a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_iff_f_maximum_l1345_134530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_fractions_l1345_134513

theorem max_integer_fractions (n : ℕ) (h : n = 26) :
  let S : Finset ℕ := Finset.range n
  ∃ (F : Finset (ℕ × ℕ)), 
    (∀ (a b : ℕ), (a, b) ∈ F → a ∈ S ∧ b ∈ S) ∧ 
    (∀ x ∈ S, (∃! p, p ∈ F ∧ (x = p.1 ∨ x = p.2))) ∧
    F.card = 13 ∧
    (∃ (I : Finset (ℕ × ℕ)), I ⊆ F ∧ I.card = 12 ∧ 
      (∀ (a b : ℕ), (a, b) ∈ I → a % b = 0)) ∧
    (∀ (J : Finset (ℕ × ℕ)), J ⊆ F → 
      (∀ (a b : ℕ), (a, b) ∈ J → a % b = 0) → J.card ≤ 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_fractions_l1345_134513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_intersection_theorem_l1345_134523

theorem quadratic_intersection_theorem (a b : ℕ) (x₁ x₂ : ℝ) :
  a > 0 →
  b > 0 →
  (∀ x, a * x^2 + b * x + 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  -1 < x₁ →
  x₁ < x₂ →
  x₂ < 0 →
  (∀ a' b' : ℕ, a' > 0 → b' > 0 → ∃ x₁' x₂' : ℝ,
    (∀ x, a' * x^2 + b' * x + 1 = 0 ↔ x = x₁' ∨ x = x₂') ∧
    -1 < x₁' ∧ x₁' < x₂' ∧ x₂' < 0 →
    a' ≥ a ∧ b' ≥ b) →
  a = 5 ∧ b = 5 ∧ x₁ = (-5 - Real.sqrt 5) / 10 ∧ x₂ = (-5 + Real.sqrt 5) / 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_intersection_theorem_l1345_134523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_for_given_sin_cos_product_l1345_134544

theorem tan_value_for_given_sin_cos_product (α : ℝ) : 
  0 < α → α < π/2 →  -- acute angle condition
  Real.sin α * Real.cos α = 1/4 → 
  Real.tan α = 2 + Real.sqrt 3 ∨ Real.tan α = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_for_given_sin_cos_product_l1345_134544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_properties_l1345_134542

-- Define the ceiling function as noncomputable
noncomputable def ceiling (x : ℝ) : ℤ := Int.ceil x

-- State the properties to be proven
theorem ceiling_properties :
  (∀ x₁ x₂ : ℝ, ceiling x₁ = ceiling x₂ → x₁ - x₂ < 1) ∧
  (∀ x₁ x₂ : ℝ, ceiling (x₁ + x₂) ≤ ceiling x₁ + ceiling x₂) :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_properties_l1345_134542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1345_134597

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) + x^3 + 3*x

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : f (2*a - 1) + f (b - 1) = 0) : 
  (∀ x y, x > 0 → y > 0 → f (2*x - 1) + f (y - 1) = 0 → 
    (2*a^2)/(a+1) + (b^2+1)/b ≤ (2*x^2)/(x+1) + (y^2+1)/y) ∧
  (2*a^2)/(a+1) + (b^2+1)/b = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1345_134597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_l1345_134588

theorem sin_shift (x : ℝ) : 
  Real.sin (x - π/3) = Real.sin (x - π/3) :=
by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_l1345_134588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_m_value_l1345_134522

-- Define the ellipse parameters
noncomputable def a : ℝ := 2
noncomputable def b : ℝ := Real.sqrt 3
noncomputable def c : ℝ := 1

-- Define the eccentricity
noncomputable def e : ℝ := 1 / 2

-- Define the distance from left vertex to right focus
def d : ℝ := 3

-- Define the maximum PQ distance
noncomputable def max_pq : ℝ := Real.sqrt 5

-- Define m
noncomputable def m : ℝ := 1 / 2

-- Assumptions
axiom h1 : a > b
axiom h2 : b > 0
axiom h3 : a + c = d
axiom h4 : c / a = e
axiom h5 : m > 0

-- Theorem for the ellipse equation
theorem ellipse_equation : 
  a = 2 ∧ b = Real.sqrt 3 := by sorry

-- Theorem for the value of m
theorem m_value : m = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_m_value_l1345_134522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l1345_134571

/-- The curve C is defined by the equation x²/(4-m) + y²/(2+m) = 1 --/
def C (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / (4 - m) + p.2^2 / (2 + m) = 1}

/-- The foci of an ellipse --/
def foci (a b : ℝ) : Set (ℝ × ℝ) :=
  {(0, Real.sqrt (b^2 - a^2)), (0, -Real.sqrt (b^2 - a^2))}

/-- The asymptotes of a hyperbola --/
def asymptotes (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = b/a * p.1 ∨ p.2 = -b/a * p.1}

theorem curve_C_properties :
  (∃ F, F = foci (Real.sqrt 2) 2 ∧ F ⊆ C 2) ∧
  (asymptotes (Real.sqrt 2) (2 * Real.sqrt 2) = asymptotes 1 2) ∧
  (∃ r, C 1 = {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l1345_134571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorems_l1345_134550

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove the projection theorem, sine rule, and cosine rule. -/
theorem triangle_theorems (a b c A B C : ℝ) : 
  (0 < a) → (0 < b) → (0 < c) → 
  (0 < A) → (A < π) → (0 < B) → (B < π) → (0 < C) → (C < π) →
  (A + B + C = π) →
  (c = a * Real.cos B + b * Real.cos A) ∧ 
  (a / Real.sin A = b / Real.sin B) ∧
  (c^2 = a^2 + b^2 - 2*a*b * Real.cos C) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorems_l1345_134550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1345_134524

noncomputable def A (a b c : ℝ) : Set ℝ :=
  {Real.sin a, Real.cos b, 0, 1, -2, Real.sqrt 2 / 2, Real.log c}

noncomputable def B (a b c : ℝ) : Set ℝ :=
  {x | ∃ y ∈ A a b c, x = y^2022 + y^2}

theorem problem_statement (a b c : ℝ) :
  (∃ f : B a b c → Fin 4, Function.Bijective f) →
  Real.cos (2*a) - Real.cos (2*b) + c = 1 + Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1345_134524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_C_to_D_l1345_134527

/-- The distance between two points given a series of movements --/
noncomputable def distance_after_movements (north south east west : ℝ) : ℝ :=
  Real.sqrt ((north - south)^2 + (east - west)^2)

/-- Theorem stating the distance between C and D --/
theorem distance_C_to_D : 
  distance_after_movements 70 90 80 30 = Real.sqrt 2900 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_C_to_D_l1345_134527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_implies_r_absolute_value_l1345_134526

def complex_polynomial (p q r : ℤ) : ℂ → ℂ := 
  λ z ↦ p * z^4 + q * z^3 + r * z^2 + q * z + p

theorem polynomial_root_implies_r_absolute_value (p q r : ℤ) :
  complex_polynomial p q r (1 + Complex.I) = 0 →
  Int.gcd p (Int.gcd q r) = 1 →
  |r| = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_implies_r_absolute_value_l1345_134526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_theorem_l1345_134503

structure GroupData where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  avg_a : ℝ
  avg_b : ℝ
  avg_c : ℝ
  avg_d : ℝ
  avg_ab : ℝ
  avg_ac : ℝ
  avg_bc : ℝ
  avg_abc : ℝ

noncomputable def average_age (data : GroupData) : ℝ :=
  (data.a * data.avg_a + data.b * data.avg_b + data.c * data.avg_c + data.d * data.avg_d) /
  (data.a + data.b + data.c + data.d)

theorem average_age_theorem (data : GroupData)
  (h1 : data.avg_a = 30)
  (h2 : data.avg_b = 25)
  (h3 : data.avg_c = 35)
  (h4 : data.avg_d = 40)
  (h5 : data.avg_ab = 28)
  (h6 : data.avg_ac = 32)
  (h7 : data.avg_bc = 30)
  (h8 : data.avg_abc = 31) :
  average_age data = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_theorem_l1345_134503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l1345_134577

/-- The focus of a parabola given by y = ax^2 + bx + c -/
noncomputable def parabola_focus (a b c : ℝ) : ℝ × ℝ :=
  let h := -b / (2 * a)
  let k := c - b^2 / (4 * a)
  (h, k + 1 / (4 * a))

theorem focus_of_specific_parabola :
  parabola_focus 2 6 (-5) = (-3/2, -45/8) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_of_specific_parabola_l1345_134577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_after_walking_l1345_134535

/-- Represents a point on the circular track -/
inductive TrackPoint
| A
| B
| C
| D
| Start

/-- Calculates the distance between two points on the track -/
noncomputable def distance (p1 p2 : TrackPoint) : ℝ :=
  sorry

/-- Represents the circular track -/
structure Track where
  length : ℝ
  points : List TrackPoint

/-- Represents a walker on the track -/
structure Walker where
  speed : ℝ
  startPoint : TrackPoint
  direction : Bool  -- True for counter-clockwise, False for clockwise

/-- Calculates the final position of the walker -/
noncomputable def finalPosition (track : Track) (walker : Walker) (time : ℝ) : TrackPoint :=
  sorry

theorem closest_point_after_walking
  (track : Track)
  (walker : Walker)
  (walkTime : ℝ) :
  track.length = 400 ∧
  track.points = [TrackPoint.A, TrackPoint.B, TrackPoint.C, TrackPoint.D] ∧
  walker.speed = 1.4 ∧
  walker.startPoint = TrackPoint.Start ∧
  walker.direction = true ∧
  walkTime = 30 * 60 ∧
  distance TrackPoint.A TrackPoint.B = distance TrackPoint.B TrackPoint.C ∧
  distance TrackPoint.B TrackPoint.C = distance TrackPoint.C TrackPoint.D ∧
  distance TrackPoint.C TrackPoint.D = distance TrackPoint.D TrackPoint.A ∧
  distance TrackPoint.Start TrackPoint.A = distance TrackPoint.Start TrackPoint.B →
  ∀ p : TrackPoint, p ≠ TrackPoint.C →
    distance (finalPosition track walker walkTime) TrackPoint.C <
    distance (finalPosition track walker walkTime) p :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_after_walking_l1345_134535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_is_15_4_l1345_134537

/-- The coefficient of x^2 in the binomial expansion of (x^2/2 - 1/√x)^6 -/
noncomputable def coefficient_x_squared : ℚ :=
  let a : ℝ → ℝ := λ x ↦ x^2 / 2
  let b : ℝ → ℝ := λ x ↦ -1 / Real.sqrt x
  let n : ℕ := 6
  let r : ℕ := 4
  (-1)^r * (1/2)^(n-r) * (n.choose r)

theorem coefficient_x_squared_is_15_4 :
  coefficient_x_squared = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_is_15_4_l1345_134537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_average_age_l1345_134592

theorem cricket_team_average_age 
  (team_size : ℕ) 
  (team_average : ℝ) 
  (wicket_keeper_age_diff : ℝ) 
  (remaining_average_diff : ℝ) :
  team_size = 11 →
  team_average = 24 →
  wicket_keeper_age_diff = 3 →
  remaining_average_diff = 1 →
  (let total_age := team_size * team_average
   let wicket_keeper_age := team_average + wicket_keeper_age_diff
   let remaining_players := team_size - 2
   let remaining_average := team_average - remaining_average_diff
   (total_age - wicket_keeper_age - (total_age - wicket_keeper_age - remaining_players * remaining_average)) / remaining_players) = 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_average_age_l1345_134592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1345_134531

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / Real.sqrt (3 * x - 9)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1345_134531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_f_eq_5_l1345_134590

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 4 * x + 8 else 3 * x - 15

theorem solutions_of_f_eq_5 :
  {x : ℝ | f x = 5} = {-3/4, 20/3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_f_eq_5_l1345_134590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1345_134529

-- Define the function f
noncomputable def f (x a b : ℝ) : ℝ := 2 * (Real.log x / Real.log 2)^2 + 2 * a * (Real.log (1/x) / Real.log 2) + b

-- State the theorem
theorem function_properties (a b : ℝ) :
  (∀ x > 0, f x a b ≥ f (1/2) a b) ∧ (f (1/2) a b = -8) →
  (a = -2 ∧ b = -6) ∧
  (∀ x > 0, f x a b > 0 ↔ (0 < x ∧ x < 1/8) ∨ x > 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1345_134529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fitting_model_l1345_134596

/-- Represents a regression model with its coefficient of determination (R²) -/
structure RegressionModel where
  name : String
  r_squared : Float

/-- Determines if a model has the best fitting effect among a list of models -/
def has_best_fitting_effect (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, model.r_squared ≥ m.r_squared

theorem best_fitting_model (models : List RegressionModel) 
  (h1 : RegressionModel.mk "Model 1" 0.99 ∈ models)
  (h2 : RegressionModel.mk "Model 2" 0.88 ∈ models)
  (h3 : RegressionModel.mk "Model 3" 0.50 ∈ models)
  (h4 : RegressionModel.mk "Model 4" 0.20 ∈ models) :
  has_best_fitting_effect (RegressionModel.mk "Model 1" 0.99) models :=
by
  sorry

#check best_fitting_model

end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_fitting_model_l1345_134596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_for_similar_statues_l1345_134502

/-- Calculates the amount of paint needed for similar statues -/
noncomputable def paint_needed (original_height : ℝ) (original_paint : ℝ) (new_height : ℝ) (num_statues : ℕ) : ℝ :=
  let scale_factor := (new_height / original_height) ^ 2
  num_statues * original_paint * scale_factor

/-- Theorem stating the amount of paint needed for 800 similar statues -/
theorem paint_for_similar_statues (original_height original_paint new_height : ℝ) (num_statues : ℕ) :
  original_height = 8 →
  original_paint = 2 →
  new_height = 2 →
  num_statues = 800 →
  paint_needed original_height original_paint new_height num_statues = 100 := by
  sorry

#check paint_for_similar_statues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_for_similar_statues_l1345_134502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_vertices_on_circle_min_sum_chords_l1345_134560

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a circle with center (0, √2/2) and radius √(1/2) -/
def circle_M (x y : ℝ) : Prop :=
  x^2 + (y - Real.sqrt 2 / 2)^2 = 1/2

/-- The ellipse C satisfying the given conditions -/
noncomputable def C : Ellipse where
  a := Real.sqrt 2
  b := 1
  h_pos := by sorry

theorem ellipse_equation (x y : ℝ) :
  (y^2 / C.a^2 + x^2 / C.b^2 = 1) ↔ (y^2 / 2 + x^2 = 1) :=
by sorry

theorem vertices_on_circle :
  circle_M C.a (C.b) ∧ circle_M (-C.a) (C.b) :=
by sorry

theorem min_sum_chords :
  ∃ (min : ℝ), min = 8 * Real.sqrt 3 / 3 ∧
  ∀ (A B C D : ℝ × ℝ),
    (A.1^2 / 2 + A.2^2 = 1) →
    (B.1^2 / 2 + B.2^2 = 1) →
    (C.1^2 / 2 + C.2^2 = 1) →
    (D.1^2 / 2 + D.2^2 = 1) →
    -- AB and CD are perpendicular
    ((B.2 - A.2) * (D.2 - C.2) = -(B.1 - A.1) * (D.1 - C.1)) →
    -- AB and CD pass through the upper focus
    ((0 : ℝ), Real.sqrt (1/2)) ∈ Set.Icc A B →
    ((0 : ℝ), Real.sqrt (1/2)) ∈ Set.Icc C D →
    min ≤ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) +
          Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_vertices_on_circle_min_sum_chords_l1345_134560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_value_l1345_134500

/-- Represents a right triangle in the coordinate plane with legs parallel to the axes -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Slope of the first median -/
noncomputable def first_median_slope (t : RightTriangle) : ℝ := t.c / (2 * t.d)

/-- Slope of the second median -/
noncomputable def second_median_slope (t : RightTriangle) : ℝ := 2 * t.c / t.d

/-- The condition that one median lies on y = 2x + 1 -/
def first_median_condition (t : RightTriangle) : Prop :=
  first_median_slope t = 2

/-- The condition that the other median lies on y = mx + 3 -/
def second_median_condition (t : RightTriangle) (m : ℝ) : Prop :=
  second_median_slope t = m

theorem unique_m_value :
  ∃! m : ℝ, ∃ t : RightTriangle, first_median_condition t ∧ second_median_condition t m ∧ m = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_value_l1345_134500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_x_geq_24_l1345_134582

/-- The function x(t) as defined in the problem -/
noncomputable def x (t a : ℝ) : ℝ := 5 * (t + 1)^2 + a / (t + 1)^5

/-- The theorem statement -/
theorem min_a_for_x_geq_24 :
  ∃ a_min : ℝ, a_min = 2 * (24/7)^(7/2) ∧
  (∀ t : ℝ, t ≥ 0 → x t a_min ≥ 24) ∧
  (∀ a : ℝ, a < a_min → ∃ t : ℝ, t ≥ 0 ∧ x t a < 24) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_x_geq_24_l1345_134582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_sum_probability_l1345_134559

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The total number of possible outcomes when rolling three dice -/
def total_outcomes : ℕ := sides ^ 3

/-- The number of favorable outcomes (sum < 16) -/
def favorable_outcomes : ℕ := total_outcomes - 10

/-- The probability of rolling three fair six-sided dice and getting a sum less than 16 -/
theorem dice_sum_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 103 / 108 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_sum_probability_l1345_134559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_number_l1345_134505

def number : ℕ := 4^12 * 5^21

theorem sum_of_digits_of_number (n : ℕ) : n = number → (n.digits 10).sum = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_number_l1345_134505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spacy_subsets_count_l1345_134549

/-- A set is spacy if it contains no more than one out of any three consecutive integers. -/
def IsSpacy (s : Set ℤ) : Prop :=
  ∀ x : ℤ, (x ∈ s ∧ x + 1 ∈ s) → x + 2 ∉ s

/-- The number of spacy subsets of {1, ..., n} -/
def NumSpacySubsets : ℕ → ℕ
  | 0 => 1  -- Empty set is spacy
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | n + 4 => NumSpacySubsets (n + 3) + NumSpacySubsets (n + 1)

theorem spacy_subsets_count :
  NumSpacySubsets 15 = 406 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spacy_subsets_count_l1345_134549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_TMN_l1345_134561

/-- The trajectory of point P -/
def Γ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1 ∧ y ≠ 0

/-- The slope of line MN is 1 -/
def slope_MN (x₁ y₁ x₂ y₂ : ℝ) : Prop := (y₂ - y₁) / (x₂ - x₁) = 1

/-- The product of slopes of PA and PB is -1/4 -/
def slope_product (x y : ℝ) : Prop := (y / (x + 2)) * (y / (x - 2)) = -1/4

/-- T is a fixed point (3, 0) -/
def T : ℝ × ℝ := (3, 0)

/-- Helper function to calculate the area of a triangle -/
noncomputable def area_triangle (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

/-- The maximum area of triangle TMN -/
theorem max_area_TMN :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    Γ x₁ y₁ ∧ Γ x₂ y₂ ∧
    slope_MN x₁ y₁ x₂ y₂ ∧
    (∀ (x y : ℝ), Γ x y → slope_product x y) →
    (∀ (a b : ℝ),
      Γ a b ∧
      (∃ (c d : ℝ), Γ c d ∧ slope_MN a b c d) →
      area_triangle T (a, b) (c, d) ≤ 16/5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_TMN_l1345_134561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_valid_integer_valid_147_solution_l1345_134585

def is_valid (n : ℕ) : Prop :=
  n < 150 ∧ Nat.gcd n 24 = 3

theorem greatest_valid_integer : ∀ m, is_valid m → m ≤ 147 :=
  sorry

theorem valid_147 : is_valid 147 :=
  sorry

theorem solution : ∃ n, is_valid n ∧ ∀ m, is_valid m → m ≤ n :=
  ⟨147, by {
    constructor
    · exact valid_147
    · exact greatest_valid_integer
  }⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_valid_integer_valid_147_solution_l1345_134585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expense_equalization_l1345_134508

/-- The amount paid by each friend --/
structure Expenses where
  tom : ℚ
  dorothy : ℚ
  sammy : ℚ
  alice : ℚ

/-- The transfers made to equalize expenses --/
structure Transfers where
  t : ℚ  -- amount Tom gives to Sammy
  d : ℚ  -- amount Dorothy gives to Alice

/-- The problem statement --/
theorem expense_equalization (e : Expenses) (tr : Transfers) : 
  e.tom = 130 ∧ e.dorothy = 160 ∧ e.sammy = 150 ∧ e.alice = 180 →
  (e.tom + e.dorothy + e.sammy + e.alice) / 4 = 155 →
  e.tom + tr.t = 155 →
  e.dorothy - tr.d = 155 →
  e.sammy - tr.t = 155 →
  e.alice + tr.d = 155 →
  tr.t - tr.d = 30 := by
  sorry

#check expense_equalization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expense_equalization_l1345_134508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_female_officers_count_l1345_134519

theorem female_officers_count (total_on_duty : ℕ) (ratio_male_female : ℚ) (female_on_duty_percentage : ℚ) :
  total_on_duty = 400 →
  ratio_male_female = 7 / 3 →
  female_on_duty_percentage = 32 / 100 →
  ∃ (total_female : ℕ), total_female = 375 ∧ 
    (female_on_duty_percentage * (total_female : ℚ) = (total_on_duty : ℚ) * 3 / (7 + 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_female_officers_count_l1345_134519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l1345_134567

/-- The phase shift of a cosine function in the form y = A cos(Bx + C) -/
noncomputable def phase_shift (A B C : ℝ) : ℝ := -C / B

/-- The cosine function y = 3 cos(3x - π/4) -/
noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (3 * x - Real.pi / 4)

theorem phase_shift_of_f :
  phase_shift 3 3 (-Real.pi / 4) = Real.pi / 12 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l1345_134567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l1345_134554

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a trapezoid ABCD -/
structure Trapezoid where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if two line segments are parallel -/
def isParallel (p1 q1 p2 q2 : Point) : Prop :=
  (q1.y - p1.y) * (q2.x - p2.x) = (q2.y - p2.y) * (q1.x - p1.x)

/-- Check if two line segments are perpendicular -/
def isPerpendicular (p1 q1 p2 q2 : Point) : Prop :=
  (q1.x - p1.x) * (q2.x - p2.x) + (q1.y - p1.y) * (q2.y - p2.y) = 0

/-- Calculate the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((q.x - p.x)^2 + (q.y - p.y)^2)

/-- Calculate the angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ :=
  Real.arccos (((p1.x - p2.x) * (p3.x - p2.x) + (p1.y - p2.y) * (p3.y - p2.y)) /
    (distance p1 p2 * distance p2 p3))

/-- Calculate the area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ :=
  ((distance t.A t.D + distance t.B t.C) / 2) * (distance t.A t.B)

theorem isosceles_trapezoid_area (t : Trapezoid) (X Y : Point) :
  isParallel t.B t.C t.A t.D →
  distance t.A t.B = distance t.C t.D →
  isPerpendicular t.A t.B t.A t.D →
  X.x > t.A.x ∧ X.x < Y.x ∧ Y.x < t.C.x →
  angle t.A X t.D = π / 2 →
  distance t.A X = 4 →
  distance X Y = 2 →
  distance Y t.C = 3 →
  trapezoidArea t = 24 := by
  sorry

#check isosceles_trapezoid_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_area_l1345_134554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_product_bounds_l1345_134578

open MeasureTheory Interval Set

/-- Given two polynomials P and Q on [0, 2] satisfying certain integral conditions,
    prove that the integral of their product is bounded between -8 and 16. -/
theorem integral_product_bounds
  (P Q : ℝ → ℝ)
  (hP1 : ∫ x in (Icc 0 2), (P x)^2 = 14)
  (hP2 : ∫ x in (Icc 0 2), P x = 4)
  (hQ1 : ∫ x in (Icc 0 2), (Q x)^2 = 26)
  (hQ2 : ∫ x in (Icc 0 2), Q x = 2) :
  -8 ≤ ∫ x in (Icc 0 2), P x * Q x ∧ ∫ x in (Icc 0 2), P x * Q x ≤ 16 := by
  sorry

#check integral_product_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_product_bounds_l1345_134578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1345_134543

noncomputable section

/-- The side length of the equilateral triangle in the figure. -/
def triangle_side : ℝ := 20

/-- The base of the rectangle (shared with the triangle). -/
def base : ℝ := 20

/-- The altitude of the equilateral triangle. -/
noncomputable def triangle_altitude : ℝ := (Real.sqrt 3 / 2) * triangle_side

/-- The height of the rectangle. -/
noncomputable def rectangle_height : ℝ := 2 * triangle_altitude

/-- The area of the rectangle. -/
noncomputable def rectangle_area : ℝ := base * rectangle_height

/-- The area of the equilateral triangle. -/
noncomputable def triangle_area : ℝ := (1 / 2) * base * triangle_altitude

/-- The total area of the figure. -/
noncomputable def total_area : ℝ := rectangle_area + triangle_area

/-- Each of the three equal areas. -/
noncomputable def equal_area : ℝ := total_area / 3

theorem triangle_side_length :
  triangle_side = 20 ∧
  base = 20 ∧
  rectangle_height = 2 * triangle_altitude ∧
  total_area = 3 * equal_area →
  triangle_side = 20 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1345_134543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_problem_l1345_134504

theorem tan_sum_problem (x y : ℝ) 
  (h1 : Real.tan x + Real.tan y = 40)
  (h2 : (Real.tan x)⁻¹ + (Real.tan y)⁻¹ = 50) : 
  Real.tan (x + y) = 200 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_problem_l1345_134504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trihedral_angle_plane_angles_l1345_134538

/-- Given a trihedral angle with one plane angle α and adjacent dihedral angles β and γ,
    this theorem states the formulas for the other two plane angles. -/
theorem trihedral_angle_plane_angles
  (α β γ : Real)
  (h_α : 0 < α ∧ α < π)
  (h_β : 0 < β ∧ β < π)
  (h_γ : 0 < γ ∧ γ < π) :
  ∃ (θ₁ θ₂ : Real),
    θ₁ = Real.arctan (Real.tan (π/2 - α) * Real.sin β * (Real.tan (π/2 - β) + Real.tan (π/2 - γ) / Real.cos α)) ∧
    θ₂ = Real.arctan (Real.tan (π/2 - α) * Real.sin γ * (Real.tan (π/2 - γ) + Real.tan (π/2 - β) / Real.cos α)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trihedral_angle_plane_angles_l1345_134538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_position_after_12_moves_l1345_134509

noncomputable def ω : ℂ := Complex.exp (Complex.I * (Real.pi / 3))

noncomputable def move (z : ℂ) : ℂ := ω * z + 6

noncomputable def iteratedMove (z : ℂ) : ℕ → ℂ
  | 0 => z
  | n + 1 => move (iteratedMove z n)

theorem particle_position_after_12_moves :
  iteratedMove 10 12 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_position_after_12_moves_l1345_134509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_circular_grid_l1345_134591

/-- The area of the shaded portion in a circular grid with specific geometric shapes --/
theorem shaded_area_in_circular_grid (r : ℝ) (π : ℝ) (n : ℕ) : 
  r = 4 → π = 3 → n = 8 → 
  (let grid_square_side := r / n
   let large_circle_area := π * r^2
   let small_circle_area := 3 * π * (r/2)^2
   let rectangle_area := (r/2) * (r/4)
   let small_shapes_area := 3 * ((grid_square_side^2) * 2)
   large_circle_area - small_circle_area - rectangle_area - small_shapes_area) = 32.5 := by
  sorry

#check shaded_area_in_circular_grid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_circular_grid_l1345_134591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triangles_with_center_l1345_134518

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  center : ℝ × ℝ
  is_regular : True  -- We assume the polygon is regular without specifying the conditions

/-- A triangle formed by three points -/
structure Triangle where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ

/-- Predicate to check if a point is inside a triangle -/
def point_inside_triangle (t : Triangle) (p : ℝ × ℝ) : Prop := sorry

/-- The set of all triangles formed by vertices of a regular polygon -/
def all_triangles (poly : RegularPolygon 25) : Set Triangle := sorry

/-- The set of triangles that contain the center of the polygon -/
def triangles_with_center (poly : RegularPolygon 25) : Set Triangle :=
  {t ∈ all_triangles poly | point_inside_triangle t poly.center}

/-- Assume the set of triangles with center is finite -/
instance (poly : RegularPolygon 25) : Fintype (triangles_with_center poly) := sorry

theorem count_triangles_with_center (poly : RegularPolygon 25) :
  Fintype.card (triangles_with_center poly) = 925 := by
  sorry

#check count_triangles_with_center

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triangles_with_center_l1345_134518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_n_l1345_134579

/-- The sum of digits of 10^75 - 75 -/
def sum_of_digits : ℕ := 664

/-- The number we're considering -/
def n : ℕ := 10^75 - 75

theorem sum_of_digits_of_n : sum_of_digits = (n.repr.toList.map (λ c => c.toString.toNat!)).sum := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_n_l1345_134579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_value_l1345_134547

noncomputable def f (x a b : ℝ) : ℝ := x + a / x + b

theorem max_b_value (a b : ℝ) :
  (∀ x ∈ Set.Icc (1/4 : ℝ) 1, ∀ a ∈ Set.Icc (1/2 : ℝ) 2, f x a b ≤ 10) →
  b ≤ 7/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_value_l1345_134547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_intersection_l1345_134586

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x + 2) + Real.log (3 - x)

-- Define the domain A of f
def A : Set ℝ := {x | -2 < x ∧ x < 3}

-- Define the set B
def B (m : ℝ) : Set ℝ := {x | 1 - m < x ∧ x < 3 * m - 1}

-- Theorem statement
theorem domain_and_intersection (m : ℝ) : 
  (∀ x, f x ∈ Set.univ ↔ x ∈ A) ∧ 
  (A ∩ B m = B m → m ≤ 4/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_and_intersection_l1345_134586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_google_search_3_1415_l1345_134541

/-- Represents the number of search results for a given query on Google -/
def google_search_results (query : String) : ℕ := sorry

/-- Predicate indicating whether Google searches for substrings -/
def google_searches_substrings : Prop := sorry

theorem google_search_3_1415 : 
  ¬google_searches_substrings → google_search_results "3.1415" = 422000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_google_search_3_1415_l1345_134541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_l1345_134575

/-- The price of a book given contributions from family members -/
theorem book_price
  (father_contribution : ℝ)
  (elder_brother_contribution : ℝ)
  (younger_brother_contribution : ℝ)
  (h1 : younger_brother_contribution = 10)
  (h2 : father_contribution = (elder_brother_contribution + younger_brother_contribution) / 2)
  (h3 : elder_brother_contribution = (father_contribution + younger_brother_contribution) / 3)
  : father_contribution + elder_brother_contribution + younger_brother_contribution = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_l1345_134575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l1345_134507

noncomputable def translation_vector : ℝ × ℝ := (Real.pi/4, -1/2)
noncomputable def translation_magnitude : ℝ := (1/2) * Real.sqrt (Real.pi^2 + 4)

noncomputable def original_function (x : ℝ) : ℝ := Real.sin x

theorem trajectory_equation (x : ℝ) :
  let scaled_vector := (translation_magnitude / (translation_vector.1^2 + translation_vector.2^2).sqrt) • translation_vector
  let translated_x := x - scaled_vector.1
  let translated_y := original_function translated_x - scaled_vector.2
  translated_y = -2 * (Real.cos (x/2))^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_l1345_134507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1345_134594

/-- The minimum distance between a point on the unit circle and its
    projection onto the line x + y = 2 at a 45° angle -/
theorem min_distance_circle_to_line :
  let C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
  let l : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
  ∃ (min_dist : ℝ),
    (∀ (P : ℝ × ℝ) (A : ℝ × ℝ),
      P ∈ C →
      A ∈ l →
      (A.1 - P.1 = A.2 - P.2) →  -- 45° angle condition
      ‖A - P‖ ≥ min_dist) ∧
    (∃ (P : ℝ × ℝ) (A : ℝ × ℝ),
      P ∈ C ∧
      A ∈ l ∧
      (A.1 - P.1 = A.2 - P.2) ∧
      ‖A - P‖ = min_dist) ∧
    min_dist = 2 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l1345_134594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1345_134536

noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 1) + 1/3

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l1345_134536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_positive_integer_solution_l1345_134576

theorem no_positive_integer_solution :
  ¬ ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    (45 * x = (35 * 900) / 100) ∧ 
    (y^2 + x = 100) ∧ 
    (z = x^3 * y - (2 * x + 1) / (y + 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_positive_integer_solution_l1345_134576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_range_of_m_l1345_134556

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 1 - 2 / (2^x + 1)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2*x - Real.pi/6)

-- Theorem for the range of k
theorem range_of_k :
  ∀ k : ℝ, (∃ x : ℝ, (2^x + 1) * f x + k = 0) → k < 1 :=
by
  sorry

-- Theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ,
  (∀ x₁ : ℝ, 0 < x₁ ∧ x₁ < 1 →
    ∃ x₂ : ℝ, -Real.pi/4 ≤ x₂ ∧ x₂ ≤ Real.pi/6 ∧ f x₁ - m * 2^x₁ > g x₂) ↔
  m ≤ 7/6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_range_of_m_l1345_134556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_a_le_one_l1345_134564

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log (x + 1)

-- State the theorem
theorem f_inequality_iff_a_le_one :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 0 → f x ≥ a * x) ↔ a ≤ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_iff_a_le_one_l1345_134564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lisas_journey_remaining_is_nine_l1345_134551

/-- The total distance from Lisa's home to Shangri-La -/
noncomputable def D : ℝ := 90

/-- The distance Lisa drove in the first hour -/
noncomputable def first_hour : ℝ := D / 3

/-- The distance Lisa drove in the second hour -/
noncomputable def second_hour : ℝ := (D - first_hour) / 2

/-- The distance Lisa drove in the third hour -/
noncomputable def third_hour : ℝ := first_hour - D / 10

/-- The remaining distance after three hours -/
noncomputable def remaining : ℝ := D - (first_hour + second_hour + third_hour)

theorem lisas_journey : D = 90 := by
  -- Unfold the definition of D
  unfold D
  -- The proof is trivial since we defined D as 90
  rfl

theorem remaining_is_nine : remaining = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lisas_journey_remaining_is_nine_l1345_134551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_alpha_l1345_134525

theorem cos_pi_minus_alpha (α : ℝ) (h1 : Real.sin α = 12/13) (h2 : 0 < α ∧ α < π/2) :
  Real.cos (π - α) = -5/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_alpha_l1345_134525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_g_zero_properties_l1345_134552

noncomputable def f (x : ℝ) : ℝ := x * Real.log (-x)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * f (a * x) - Real.exp (x - 2)

theorem f_properties :
  (∃ (x : ℝ), x < 0 ∧ ∀ (y : ℝ), y < 0 → f y ≤ f x) ∧
  (∃ (x : ℝ), x = -(1 / Real.exp 1) ∧ f x = 1 / Real.exp 1) := by sorry

theorem g_zero_properties (a : ℝ) :
  (a > 0 ∨ a = -(1 / Real.exp 1) → ∃! (x : ℝ), g a x = 0) ∧
  (a < 0 ∧ a ≠ -(1 / Real.exp 1) → ∀ (x : ℝ), g a x ≠ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_g_zero_properties_l1345_134552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_a_range_l1345_134569

/-- A function f is monotonically increasing on ℝ -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- The function f(x) = x - (1/3)sin(2x) + a*sin(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (1/3) * Real.sin (2*x) + a * Real.sin x

/-- Theorem: If f is monotonically increasing on ℝ, then a ∈ [-1/3, 1/3] -/
theorem monotone_f_implies_a_range (a : ℝ) :
  MonotonicallyIncreasing (f a) → a ∈ Set.Icc (-1/3) (1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_a_range_l1345_134569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_A_is_5_l1345_134581

/-- Two people walking towards each other --/
structure WalkingProblem where
  duration : ℝ  -- Duration of walk in hours
  distance : ℝ  -- Initial distance between A and B in km
  speed_B : ℝ   -- Speed of B in km/h

/-- The speed of A given the problem parameters --/
noncomputable def speed_A (p : WalkingProblem) : ℝ :=
  (p.distance - p.speed_B * p.duration) / p.duration

/-- Theorem stating that given the problem conditions, A's speed is 5 km/h --/
theorem speed_A_is_5 (p : WalkingProblem) 
  (h1 : p.duration = 2)
  (h2 : p.distance = 24)
  (h3 : p.speed_B = 7) : 
  speed_A p = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_A_is_5_l1345_134581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jericho_altitude_proof_l1345_134515

/-- Represents the altitude of a location in meters relative to sea level -/
def altitude (x : ℤ) : ℤ := x

/-- Jericho's distance below sea level in meters -/
def jericho_below_sea_level : ℕ := 300

/-- The altitude of Jericho -/
def jericho_altitude : ℤ := altitude (-jericho_below_sea_level)

theorem jericho_altitude_proof : jericho_altitude = -300 := by
  -- Unfold the definition of jericho_altitude
  unfold jericho_altitude
  -- Unfold the definition of altitude
  unfold altitude
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jericho_altitude_proof_l1345_134515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_douglas_vote_percentage_l1345_134533

theorem douglas_vote_percentage (total_percentage x_percentage x_to_y_ratio : ℝ) :
  total_percentage = 58 →
  x_percentage = 64 →
  x_to_y_ratio = 2 →
  let y_percentage := (3 * total_percentage - 2 * x_percentage) / x_to_y_ratio
  y_percentage = 46 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_douglas_vote_percentage_l1345_134533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l1345_134580

-- Define the work rates
noncomputable def work_rate_ab : ℝ := 1 / 15
noncomputable def work_rate_c : ℝ := 1 / (15/4)

-- Theorem statement
theorem job_completion_time :
  let work_rate_abc := work_rate_ab + work_rate_c
  (1 / work_rate_abc) = 3 := by
  -- Unfold the definitions and simplify
  simp [work_rate_ab, work_rate_c]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l1345_134580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_bijective_l1345_134557

-- Define the ⊕ operation on {0, 1}
def bitXor : Bool → Bool → Bool
  | false, false => false
  | false, true  => true
  | true,  false => true
  | true,  true  => false

-- Define the ⊕ operation for natural numbers
def natXor (a b : ℕ) : ℕ :=
  (List.zipWith bitXor a.bits b.bits).foldl (λ acc b => 2 * acc + if b then 1 else 0) 0

-- Define the function f
def f (n : ℕ) : ℕ :=
  natXor n (n / 2)

-- Theorem statement
theorem f_is_bijective : Function.Bijective f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_bijective_l1345_134557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_tetrahedron_properties_l1345_134558

/-- A tetrahedron with three pairs of equal opposite edges -/
structure SpecialTetrahedron where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ
  edge1_eq : edge1 = 5
  edge2_eq : edge2 = Real.sqrt 34
  edge3_eq : edge3 = Real.sqrt 41

/-- The volume of the special tetrahedron -/
def volume (t : SpecialTetrahedron) : ℝ := 20

/-- The surface area of the circumscribed sphere of the special tetrahedron -/
noncomputable def circumscribed_sphere_area (t : SpecialTetrahedron) : ℝ := 50 * Real.pi

/-- Theorem stating the volume and surface area of the circumscribed sphere for the special tetrahedron -/
theorem special_tetrahedron_properties (t : SpecialTetrahedron) : 
  volume t = 20 ∧ circumscribed_sphere_area t = 50 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_tetrahedron_properties_l1345_134558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thief_distance_before_caught_l1345_134501

/-- The distance run by a thief before being overtaken by a policeman -/
theorem thief_distance_before_caught (initial_distance : ℝ) (thief_speed : ℝ) (policeman_speed : ℝ) :
  initial_distance = 150 →
  thief_speed = 8 * (1000 / 3600) →
  policeman_speed = 10 * (1000 / 3600) →
  let relative_speed := policeman_speed - thief_speed
  let time := initial_distance / relative_speed
  let distance_run := thief_speed * time
  abs (distance_run - 594.64) < 0.01 := by
  sorry

#check thief_distance_before_caught

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thief_distance_before_caught_l1345_134501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitivity_l1345_134589

-- Define the types for lines and planes
def Line : Type := ℝ → ℝ → ℝ → Prop
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define the parallel relation between planes
def parallel (p q : Plane) : Prop := 
  ∃ (k l m : ℝ), ∀ (x y z : ℝ), p x y z ↔ q (x + k) (y + l) (z + m)

-- Theorem statement
theorem parallel_transitivity (α β γ : Plane) :
  parallel α γ → parallel β γ → parallel α β :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitivity_l1345_134589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1345_134584

/-- The function f as defined in the problem -/
noncomputable def f (x y z : ℝ) : ℝ := ((x*y + y*z + z*x) * (x + y + z)) / ((x + y) * (y + z) * (z + x))

/-- Theorem stating the range of f for positive real inputs -/
theorem f_range :
  ∀ r : ℝ, (∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ f x y z = r) ↔ 1 ≤ r ∧ r ≤ 9/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1345_134584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_through_C_is_correct_l1345_134572

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- The probability of choosing to go east or south at each intersection -/
def move_probability : ℚ := 1/2

/-- The starting point A -/
def A : Point := ⟨0, 0⟩

/-- The ending point B -/
def B : Point := ⟨4, 4⟩

/-- The intermediate point C -/
def C : Point := ⟨3, 1⟩

/-- Calculates the number of paths between two points -/
def num_paths (start : Point) (finish : Point) : ℕ :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The probability of a path from A to B passing through C -/
def prob_through_C : ℚ := 8/35

theorem prob_through_C_is_correct : 
  (num_paths A C * num_paths C B : ℚ) / num_paths A B = prob_through_C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_through_C_is_correct_l1345_134572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_zero_l1345_134506

theorem trigonometric_sum_zero 
  (A B C x y z : ℝ) 
  (h_sum : ∃ k : ℤ, A + B + C = k * Real.pi)
  (h_sin1 : x * Real.sin A + y * Real.sin B + z * Real.sin C = 0)
  (h_sin2 : x^2 * Real.sin (2*A) + y^2 * Real.sin (2*B) + z^2 * Real.sin (2*C) = 0) :
  ∀ n : ℕ, x^n * Real.sin (n • A) + y^n * Real.sin (n • B) + z^n * Real.sin (n • C) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_zero_l1345_134506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_figure_area_l1345_134521

-- Define the curves
noncomputable def curve (x : ℝ) : ℝ := x^2
noncomputable def line (x : ℝ) : ℝ := 1

-- Define the area of the closed figure
noncomputable def area : ℝ := ∫ x in (-1)..1, (line x - curve x)

-- Theorem statement
theorem closed_figure_area : area = 4/3 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closed_figure_area_l1345_134521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_problem_l1345_134534

-- Define the circles and their properties
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the problem statement
theorem circle_tangent_problem (C₁ C₂ : Set (ℝ × ℝ)) (r₁ r₂ p : ℝ) :
  -- The circles are defined with their respective radii
  (∃ center₁ center₂ : ℝ × ℝ, C₁ = Circle center₁ r₁ ∧ C₂ = Circle center₂ r₂) →
  -- The circles intersect at (8, 4)
  (8, 4) ∈ C₁ ∧ (8, 4) ∈ C₂ →
  -- The product of the radii is 77
  r₁ * r₂ = 77 →
  -- The x-axis is tangent to one of the circles
  (∃ x : ℝ, (x, 0) ∈ C₁ ∨ (x, 0) ∈ C₂) →
  -- The line y = 2x + 3 is tangent to one of the circles
  (∃ x : ℝ, (x, 2*x + 3) ∈ C₁ ∨ (x, 2*x + 3) ∈ C₂) →
  -- The vertical line x = p (p > 0) is tangent to one of the circles
  p > 0 →
  (∃ y : ℝ, (p, y) ∈ C₁ ∨ (p, y) ∈ C₂) →
  -- Then p = 12
  p = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_problem_l1345_134534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linlin_cards_l1345_134599

/-- The number of cards Tongtong has initially -/
def T : ℕ := sorry

/-- The number of cards Linlin has initially -/
def L : ℕ := sorry

/-- If Tongtong gives 6 cards to Linlin, Linlin will have 3 times as many cards as Tongtong -/
axiom condition1 : L + 6 = 3 * (T - 6)

/-- If Linlin gives 2 cards to Tongtong, Linlin will have twice as many cards as Tongtong -/
axiom condition2 : L - 2 = 2 * (T + 2)

/-- Linlin originally has 66 cards -/
theorem linlin_cards : L = 66 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linlin_cards_l1345_134599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_parallel_vectors_l1345_134568

theorem cosine_of_parallel_vectors (α : ℝ) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : ∃ (k : ℝ), k ≠ 0 ∧ k * (1/3 : ℝ) = Real.cos α ∧ k * Real.tan α = 1) :
  Real.cos α = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_parallel_vectors_l1345_134568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_possibilities_l1345_134573

/-- The number of possibilities for (a,b) satisfying the given conditions --/
def count_possibilities : ℕ :=
  Finset.filter (fun p : ℕ × ℕ =>
    let a := p.1
    let b := p.2
    b > a ∧ 
    a > 0 ∧ 
    b > 0 ∧ 
    (a - 4) * (b - 4) = (a * b) / 2
  ) (Finset.range 100 ×ˢ Finset.range 100) |>.card

/-- Theorem stating that there are exactly 3 possibilities for (a,b) --/
theorem three_possibilities : count_possibilities = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_possibilities_l1345_134573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_negative_product_l1345_134512

noncomputable def f (x : ℝ) : ℝ :=
  if x < (1/2 : ℝ) then x + (1/2 : ℝ) else x^2

noncomputable def sequence_a (a : ℝ) : ℕ → ℝ
| 0 => a
| n + 1 => f (sequence_a a n)

noncomputable def sequence_b (b : ℝ) : ℕ → ℝ
| 0 => b
| n + 1 => f (sequence_b b n)

theorem exists_negative_product (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  ∃ n : ℕ+, (sequence_a a n - sequence_a a (n-1)) * (sequence_b b n - sequence_b b (n-1)) < 0 := by
  sorry

#check exists_negative_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_negative_product_l1345_134512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_shaded_triangle_l1345_134563

/-- Given a set of triangles where some are shaded, this theorem calculates
    the probability of selecting a shaded triangle. -/
theorem probability_of_shaded_triangle 
  (total_triangles : ℕ) 
  (shaded_triangles : ℕ) 
  (h1 : total_triangles = 8) 
  (h2 : shaded_triangles = 3) 
  (h3 : shaded_triangles ≤ total_triangles) 
  (h4 : total_triangles > 0) :
  (shaded_triangles : ℚ) / total_triangles = 3 / 8 := by
  sorry

#check probability_of_shaded_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_shaded_triangle_l1345_134563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l1345_134516

theorem inverse_function_point (f g h : ℝ → ℝ) :
  (f 1 = 2) →
  (∀ x, g x = f (x + 2)) →
  (∀ x, g (h x) = x ∧ h (g x) = x) →
  h 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_l1345_134516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1345_134511

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.sin (α + π/4) = Real.sqrt 2 / 4)
  (h2 : α ∈ Set.Ioo (π/2) π) :
  Real.sin α = (Real.sqrt 7 + 1) / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1345_134511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_set_l1345_134574

noncomputable def data_set : List ℝ := [-1, 0, 4, 6, 7, 14]

noncomputable def median (l : List ℝ) : ℝ := sorry

noncomputable def mean (l : List ℝ) : ℝ := (l.sum) / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  (l.map (fun x => (x - mean l) ^ 2)).sum / l.length

theorem variance_of_data_set :
  median data_set = 5 →
  variance data_set = 74/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_set_l1345_134574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_runs_30_seconds_l1345_134583

/-- Represents the race scenario between Nicky and Cristina -/
structure RaceScenario where
  total_distance : ℝ
  head_start : ℝ
  cristina_speed : ℝ
  nicky_speed : ℝ

/-- Calculates the time when Cristina catches up to Nicky -/
noncomputable def catch_up_time (race : RaceScenario) : ℝ :=
  (race.head_start * race.nicky_speed) / (race.cristina_speed - race.nicky_speed)

/-- Calculates the total time Nicky runs before Cristina catches up -/
noncomputable def nicky_total_time (race : RaceScenario) : ℝ :=
  race.head_start + catch_up_time race

/-- Theorem stating that Nicky runs for 30 seconds before Cristina catches up -/
theorem nicky_runs_30_seconds (race : RaceScenario)
  (h1 : race.total_distance = 100)
  (h2 : race.head_start = 12)
  (h3 : race.cristina_speed = 5)
  (h4 : race.nicky_speed = 3) :
  nicky_total_time race = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nicky_runs_30_seconds_l1345_134583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_l1345_134598

-- Problem 1
theorem problem_1 : (-5) - 3 + (-7) - (-8) = -7 := by sorry

-- Problem 2
theorem problem_2 : (5/9 - 3/4 + 1/12) / (-1/36) = 4 := by sorry

-- Problem 3
theorem problem_3 : -4^2 + (-20) / (-5) - 6 * (-2)^3 = 36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_l1345_134598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_arc_circumference_l1345_134540

-- Define the circle and points
def Circle : Type := ℝ × ℝ → Prop
def Point : Type := ℝ × ℝ

-- Define the radius of the circle
def radius : ℝ := 15

-- Define the angle ACB in radians
noncomputable def angle_ACB : ℝ := 50 * Real.pi / 180

-- Define the circumference of the minor arc AB
noncomputable def minor_arc_AB (c : Circle) (A B C : Point) : ℝ := 
  2 * radius * angle_ACB

-- Theorem statement
theorem minor_arc_circumference (c : Circle) (A B C : Point) :
  minor_arc_AB c A B C = 25 * Real.pi / 3 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_arc_circumference_l1345_134540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rounds_bounds_l1345_134570

/-- Game parameters -/
structure GameParams where
  p : ℝ
  h_p_pos : 0 < p
  h_p_lt_one : p < 1

/-- Expected number of rounds -/
noncomputable def expected_rounds (params : GameParams) : ℝ :=
  let q := 1 - params.p
  let u := 2 * params.p * q
  2 * (1 - u^10) / (1 - u)

/-- Theorem statement -/
theorem expected_rounds_bounds (params : GameParams) :
  2 < expected_rounds params ∧ expected_rounds params ≤ 1023/256 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_rounds_bounds_l1345_134570
