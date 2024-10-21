import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_inequality_l966_96650

theorem power_sum_inequality (a b c : ℝ) 
  (sum_eq_one : a + b + c = 1)
  (a_ge_quarter : a ≥ (1/4 : ℝ))
  (b_ge_quarter : b ≥ (1/4 : ℝ))
  (c_ge_quarter : c ≥ (1/4 : ℝ)) :
  (16 : ℝ)^a + (16 : ℝ)^b + (16 : ℝ)^c ≤ 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_inequality_l966_96650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l966_96692

/-- Represents the sum of the first n terms of the sequence a_n -/
noncomputable def S (n : ℕ) : ℝ := sorry

/-- Represents the n-th term of the sequence a_n -/
noncomputable def a (n : ℕ) : ℝ := sorry

/-- Represents the n-th term of the sequence b_n -/
noncomputable def b (n : ℕ) : ℝ := 3 / (a n * a (n + 1))

/-- Represents the sum of the first n terms of the sequence b_n -/
noncomputable def T (n : ℕ) : ℝ := sorry

theorem sequence_properties (n : ℕ) :
  (∀ k, a k > 0) ∧
  (∀ k, a k ^ 2 + 3 * a k = 6 * S k + 4) →
  (a n = 3 * n + 1) ∧
  (T n = 1 / 4 - 1 / (3 * n + 4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l966_96692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_division_l966_96638

theorem impossibility_of_division : ¬ ∃ (A B C D : Finset ℕ) (s : ℕ),
  (∀ n, n ∈ A ∪ B ∪ C ∪ D → 1 ≤ n ∧ n ≤ 1980) ∧
  (A ∩ B = ∅) ∧ (A ∩ C = ∅) ∧ (A ∩ D = ∅) ∧ (B ∩ C = ∅) ∧ (B ∩ D = ∅) ∧ (C ∩ D = ∅) ∧
  (A.card + B.card + C.card + D.card = 1980) ∧
  (A.sum id = s) ∧ (B.sum id = s + 10) ∧ (C.sum id = s + 20) ∧ (D.sum id = s + 30) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_division_l966_96638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_all_reals_l966_96671

/-- A rational function f(x) with numerator 3x^2 - 4x + 1 and denominator -7x^2 + 6x + c -/
noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (3 * x^2 - 4 * x + 1) / (-7 * x^2 + 6 * x + c)

/-- The domain of f(x) is the set of all real numbers if and only if c < -9/7 -/
theorem domain_f_all_reals (c : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f c x = y) ↔ c < -9/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_all_reals_l966_96671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jeongyeon_long_jump_record_l966_96673

/-- Calculates Jeongyeon's long jump record given Eunseol's record and the difference between their jumps. -/
noncomputable def jeongyeon_record (eunseol_record : ℝ) (difference_cm : ℝ) : ℝ :=
  eunseol_record + difference_cm / 100

theorem jeongyeon_long_jump_record :
  let eunseol_record : ℝ := 1.35
  let difference_cm : ℝ := 9
  jeongyeon_record eunseol_record difference_cm = 1.44 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jeongyeon_long_jump_record_l966_96673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_theorem_l966_96680

/-- Represents the result of a key press on the malfunctioning calculator -/
inductive CalcResult
  | Digit (n : ℕ)
  | Operation (op : ℕ → ℕ → ℕ)
deriving Inhabited

/-- Models the behavior of the malfunctioning calculator -/
def calculatorPress (key : ℕ) : CalcResult :=
  match key with
  | 0 => CalcResult.Digit 0
  | 1 => CalcResult.Digit 2
  | 2 => CalcResult.Digit 4
  | 3 => CalcResult.Digit 6
  | 4 => CalcResult.Digit 8
  | 5 => CalcResult.Digit 10
  | 6 => CalcResult.Digit 12
  | 7 => CalcResult.Digit 14
  | 8 => CalcResult.Digit 16
  | 9 => CalcResult.Digit 18
  | _ => CalcResult.Digit 0

/-- Represents a sequence of key presses on the calculator -/
def KeySequence := List CalcResult

/-- Evaluates a sequence of key presses to produce a final result -/
def evaluateSequence (seq : KeySequence) : ℕ := sorry

theorem calculator_theorem :
  (∃ (seq : KeySequence), seq.length = 3 ∧ evaluateSequence seq = 80) ∧
  (∃ (seq : KeySequence), seq.length = 5 ∧ evaluateSequence seq = 50 ∧
    (∀ i, i % 2 = 0 → ∃ (n : ℕ), seq.get? i = some (CalcResult.Digit n)) ∧
    (∀ i, i % 2 = 1 → ∃ (op : ℕ → ℕ → ℕ), seq.get? i = some (CalcResult.Operation op))) ∧
  (∀ (seq : KeySequence), evaluateSequence seq = 23 → seq.length ≥ 4) ∧
  (∃ (seq : KeySequence), seq.length = 4 ∧ evaluateSequence seq = 23) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_theorem_l966_96680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_when_a_half_condition_satisfied_iff_a_ge_one_l966_96621

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * x * Real.log x - 2 * a * x^2

-- Theorem 1: f is monotonically decreasing on (0, +∞) when a = 1/2
theorem f_decreasing_when_a_half :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f (1/2) x₂ ≤ f (1/2) x₁ :=
by
  sorry

-- Theorem 2: Condition is satisfied iff a ∈ [1, +∞)
theorem condition_satisfied_iff_a_ge_one :
  ∀ a : ℝ, (∀ x : ℝ, x > 1 → f a x ≤ (deriv (f a)) x / 2 - Real.log x - 1) ↔ a ≥ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_when_a_half_condition_satisfied_iff_a_ge_one_l966_96621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_test_score_l966_96698

/-- Represents a set of test scores -/
def TestScores := List Nat

/-- Check if all elements in a list are unique -/
def all_unique (l : List Nat) : Prop :=
  l.Nodup

/-- Check if all elements in a list are within a given range -/
def all_in_range (l : List Nat) (low high : Nat) : Prop :=
  ∀ x ∈ l, low ≤ x ∧ x ≤ high

/-- Check if the average of a list of numbers is an integer -/
def integer_average (l : List Nat) : Prop :=
  ∃ n : Nat, l.sum = n * l.length

/-- Main theorem -/
theorem seventh_test_score 
  (scores : TestScores) 
  (h1 : scores.length = 8)
  (h2 : all_unique scores)
  (h3 : all_in_range scores 94 100)
  (h4 : ∀ k : Nat, k < 8 → integer_average (scores.take (k + 1)))
  (h5 : scores.getLast? = some 97) :
  scores.get? 6 = some 94 := by
  sorry

#check seventh_test_score

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_test_score_l966_96698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_l966_96605

-- Define the constant term in the expansion
def constant_term (a : ℝ) : ℝ := (Nat.choose 6 3) * a^3

-- Define the integral
noncomputable def integral (a : ℝ) : ℝ := ∫ x in (Set.Icc 0 1), x^a

-- Theorem statement
theorem integral_value :
  ∃ a : ℝ, constant_term a = 40 ∧ integral a = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_value_l966_96605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_vector_equation_l966_96610

theorem min_value_vector_equation (A B C D : EuclideanSpace ℝ (Fin 2)) (m n : ℝ) :
  (∃ t : ℝ, t ∈ Set.Ioo 0 1 ∧ D = B + t • (C - B)) →  -- D is on BC, excluding endpoints
  (A - D = m • (A - B) + n • (A - C)) →  -- Vector equation
  m > 0 →
  n > 0 →
  1/m + 4/n ≥ 9 ∧ ∃ m₀ n₀ : ℝ, 1/m₀ + 4/n₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_vector_equation_l966_96610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mappings_and_bijections_count_l966_96688

theorem mappings_and_bijections_count (A : Type) [Fintype A] [DecidableEq A] (h : Fintype.card A = 4) :
  (Fintype.card (A → A) = 256) ∧ (Fintype.card (Equiv.Perm A) = 24) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mappings_and_bijections_count_l966_96688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_pattern_l966_96695

theorem perfect_square_pattern (n : ℕ) :
  let num := (10^n - 1) * 10^n / 9 + 5 * (10^n - 1) / 9 + 6
  num = ((10^n + 2) / 3)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_pattern_l966_96695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_after_pouring_l966_96625

/-- Represents a cylinder with a given radius and height -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

theorem water_depth_after_pouring (c1 c2 : Cylinder) 
  (h1 : c1.radius = 2)
  (h2 : c1.height = 8)
  (h3 : c2.radius = 4)
  (h4 : c2.height = 8) :
  let v1 := cylinderVolume c1
  let depth := v1 / (Real.pi * c2.radius^2)
  depth = 2 := by
  sorry

#check water_depth_after_pouring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_after_pouring_l966_96625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygonal_line_length_bound_l966_96689

/-- Represents a board with a polygonal line -/
structure Board :=
  (size : ℕ)
  (polygonal_line : Set (ℕ × ℕ))

/-- Predicate to check if a set of points forms a closed line -/
def is_closed (line : Set (ℕ × ℕ)) : Prop := sorry

/-- Predicate to check if a set of points forms a non-self-intersecting line -/
def is_non_self_intersecting (line : Set (ℕ × ℕ)) : Prop := sorry

/-- Predicate to check if a set of points is symmetric with respect to the diagonal -/
def is_symmetric_wrt_diagonal (line : Set (ℕ × ℕ)) : Prop := sorry

/-- Predicate to check if the board satisfies the given conditions -/
def valid_board (b : Board) : Prop :=
  b.size = 15 ∧
  is_closed b.polygonal_line ∧
  is_non_self_intersecting b.polygonal_line ∧
  is_symmetric_wrt_diagonal b.polygonal_line

/-- The length of the polygonal line -/
def line_length (b : Board) : ℕ := sorry

/-- Theorem stating that the length of the polygonal line is no more than 200 -/
theorem polygonal_line_length_bound (b : Board) (h : valid_board b) : 
  line_length b ≤ 200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygonal_line_length_bound_l966_96689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_shift_sin_2x_period_cos_2x_period_l966_96615

open Real

theorem sin_cos_shift (x : ℝ) : 
  sin (2 * x) = cos (2 * (x - π / 4)) :=
by sorry

theorem sin_2x_period (x : ℝ) : 
  sin (2 * (x + π)) = sin (2 * x) :=
by sorry

theorem cos_2x_period (x : ℝ) : 
  cos (2 * (x + π)) = cos (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_shift_sin_2x_period_cos_2x_period_l966_96615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l966_96663

/-- The equation of the tangent line to y = 2 ln x at (1, 0) is y = 2x - 2 -/
theorem tangent_line_equation : 
  let f : ℝ → ℝ := λ x => 2 * Real.log x
  let tangent_point : ℝ × ℝ := (1, 0)
  let tangent_line : ℝ → ℝ := λ x => 2 * x - 2
  (∀ x, tangent_line x = (deriv f) tangent_point.1 * (x - tangent_point.1) + tangent_point.2) ∧
  (f tangent_point.1 = tangent_point.2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l966_96663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_slope_ratio_l966_96642

/-- The parabola E with equation y² = 2x -/
def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 2 * p.1}

/-- Point S -/
def S : ℝ × ℝ := (2, 0)

/-- Point Q -/
def Q : ℝ × ℝ := (1, 0)

/-- Line through S with slope k₁ -/
def line_through_S (k₁ : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k₁ * (p.1 - S.1)}

/-- Collinearity of three points -/
def collinear (P Q R : ℝ × ℝ) : Prop :=
  (R.2 - P.2) * (Q.1 - P.1) = (Q.2 - P.2) * (R.1 - P.1)

theorem parabola_intersection_slope_ratio 
  (k₁ : ℝ) 
  (A B : ℝ × ℝ) 
  (hA : A ∈ E ∩ line_through_S k₁) 
  (hB : B ∈ E ∩ line_through_S k₁) 
  (C D : ℝ × ℝ) 
  (hC : C ∈ E ∧ collinear Q A C) 
  (hD : D ∈ E ∧ collinear Q B D) 
  (k₂ : ℝ) 
  (hk₂ : k₂ = (D.2 - C.2) / (D.1 - C.1)) :
  k₂ / k₁ = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_slope_ratio_l966_96642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_rounded_sum_l966_96669

/-- Rounds a natural number to the nearest multiple of 5, rounding up for .5 -/
def roundToNearestFive (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

/-- Sum of integers from 1 to n -/
def sumToN (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Sum of rounded integers from 1 to n -/
def sumRoundedToN (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun i => roundToNearestFive (i + 1))

theorem sum_equals_rounded_sum :
  sumToN 120 = sumRoundedToN 120 := by
  sorry

#eval sumToN 120
#eval sumRoundedToN 120

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_rounded_sum_l966_96669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_imply_x_value_dot_product_imply_sin_value_l966_96628

noncomputable section

open Real

def x : ℝ := sorry
axiom x_range : 0 ≤ x ∧ x ≤ π/3

def m : ℝ × ℝ := (sin x, cos x)
def n : ℝ × ℝ := (Real.sqrt 3/2, 1/2)

theorem parallel_vectors_imply_x_value :
  (∃ (k : ℝ), m = k • n) → x = π/3 := by sorry

theorem dot_product_imply_sin_value :
  m.1 * n.1 + m.2 * n.2 = 3/5 → sin (x - π/12) = -(Real.sqrt 2)/10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_imply_x_value_dot_product_imply_sin_value_l966_96628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_45_properties_l966_96655

/-- A regular polygon with exterior angles of 45 degrees -/
structure RegularPolygon45 where
  /-- The number of sides in the polygon -/
  sides : ℕ
  /-- The measure of each exterior angle is 45 degrees -/
  exterior_angle : sides * 45 = 360

theorem regular_polygon_45_properties (p : RegularPolygon45) :
  (p.sides - 2) * 180 = 1080 ∧
  p.sides * (p.sides - 3) / 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_45_properties_l966_96655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_value_l966_96672

/-- Given a linear function T of P with slope h and y-intercept 3,
    prove that if T = 20 when P = 7, then T = 32 1/7 when P = 12 -/
theorem linear_function_value (h : ℚ) (T : ℚ → ℚ) :
  (∀ P, T P = h * P + 3) →
  T 7 = 20 →
  T 12 = 32 + 1 / 7 :=
by
  intros h_eq h_7
  
  -- Find the value of h
  have h_value : h = 17 / 7 := by
    rw [h_eq] at h_7
    linarith
  
  -- Calculate T 12
  have T_12 : T 12 = 32 + 1 / 7 := by
    rw [h_eq, h_value]
    norm_num
  
  exact T_12


end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_value_l966_96672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_select_three_balls_l966_96649

def num_balls : ℕ := 10
def less_than_five : Finset ℕ := Finset.filter (fun n => 1 ≤ n ∧ n < 5) (Finset.range (num_balls + 1))
def greater_than_five : Finset ℕ := Finset.filter (fun n => 5 < n ∧ n ≤ num_balls) (Finset.range (num_balls + 1))

theorem select_three_balls : 
  (Finset.card less_than_five) * 
  (Finset.card {5}) * 
  (Finset.card greater_than_five) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_select_three_balls_l966_96649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l966_96644

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a * Real.log x

-- State the theorem
theorem function_inequality (a : ℝ) :
  (∀ t : ℝ, t ≥ 1 → f a (2*t - 1) ≥ 2 * f a t - 3) →
  a ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l966_96644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_limit_l966_96685

def my_sequence (n : ℕ) : ℚ := (n^2 + n + 1) / (2*n^2 + 3*n + 2)

theorem my_sequence_limit : 
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |my_sequence n - 1/2| < ε := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_limit_l966_96685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_inscribed_rectangle_perimeter_l966_96683

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  topLeft : Point
  bottomRight : Point

/-- Represents a rhombus -/
structure Rhombus where
  vertices : Fin 4 → Point

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a rhombus is inscribed in a rectangle -/
def isInscribed (r : Rhombus) (rect : Rectangle) : Prop :=
  ∃ (e f g h : Point),
    r.vertices 0 = e ∧ r.vertices 1 = f ∧ r.vertices 2 = g ∧ r.vertices 3 = h ∧
    e.x = rect.topLeft.x ∧ f.y = rect.topLeft.y ∧
    g.x = rect.bottomRight.x ∧ h.y = rect.bottomRight.y

/-- Calculate the perimeter of a rectangle -/
def rectanglePerimeter (rect : Rectangle) : ℝ :=
  2 * (rect.bottomRight.x - rect.topLeft.x + rect.topLeft.y - rect.bottomRight.y)

theorem rhombus_inscribed_rectangle_perimeter
  (r : Rhombus) (rect : Rectangle) (i j k l : Point)
  (h_inscribed : isInscribed r rect)
  (h_ie : distance i (r.vertices 0) = 12)
  (h_jf : distance j (r.vertices 1) = 16)
  (h_eg : distance (r.vertices 0) (r.vertices 2) = 34)
  (h_fh : distance (r.vertices 1) (r.vertices 3) = 34) :
  rectanglePerimeter rect = 148 := by
  sorry

#check rhombus_inscribed_rectangle_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_inscribed_rectangle_perimeter_l966_96683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_lambda_bound_l966_96617

theorem sequence_increasing_lambda_bound (lambda : ℝ) :
  (∀ n : ℕ, (n + 1)^2 + lambda * (n + 1) > n^2 + lambda * n) →
  lambda > -3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_increasing_lambda_bound_l966_96617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_citizens_for_minister_five_satisfies_condition_max_citizens_is_five_l966_96633

theorem max_citizens_for_minister (n : ℕ) : (Nat.choose n 4 < Nat.choose n 2) → n ≤ 5 :=
by sorry

theorem five_satisfies_condition : Nat.choose 5 4 < Nat.choose 5 2 :=
by sorry

theorem max_citizens_is_five : ∃ (n : ℕ), n = 5 ∧ Nat.choose n 4 < Nat.choose n 2 ∧ ∀ (m : ℕ), m > n → Nat.choose m 4 ≥ Nat.choose m 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_citizens_for_minister_five_satisfies_condition_max_citizens_is_five_l966_96633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_equals_two_l966_96636

-- Define the function f
def f (z : ℝ) : ℝ := z^3 + 2015*z

-- State the theorem
theorem x_plus_y_equals_two (x y : ℝ) 
  (hx : f (x - 1) = -1) 
  (hy : f (y - 1) = 1) : 
  x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_equals_two_l966_96636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l966_96693

/-- Represents the speed of the train in km/hr -/
noncomputable def train_speed : ℝ := 144

/-- Represents the time taken by the train to cross the electric pole in seconds -/
noncomputable def crossing_time : ℝ := 1.4998800095992322

/-- Converts speed from km/hr to m/s -/
noncomputable def speed_km_hr_to_m_s (speed_km_hr : ℝ) : ℝ := speed_km_hr * 1000 / 3600

/-- Calculates the length of the train in meters -/
noncomputable def train_length : ℝ := speed_km_hr_to_m_s train_speed * crossing_time

theorem train_length_approx :
  ∃ ε > 0, |train_length - 59.995| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l966_96693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_skew_lines_l966_96664

/-- Regular quadrangular pyramid -/
structure RegularQuadPyramid where
  S : Point
  A : Point
  B : Point
  C : Point
  D : Point
  is_regular : Prop  -- We'll leave this as a proposition for now

/-- Dihedral angle between faces A-SB and B-SD -/
noncomputable def dihedralAngle (p : RegularQuadPyramid) : ℝ := sorry

/-- Angle between skew lines SA and BC -/
noncomputable def skewLinesAngle (p : RegularQuadPyramid) : ℝ := sorry

theorem angle_between_skew_lines 
  (p : RegularQuadPyramid) 
  (h : Real.sin (dihedralAngle p) = Real.sqrt 6 / 3) : 
  skewLinesAngle p = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_skew_lines_l966_96664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_has_real_zero_l966_96668

theorem polynomial_has_real_zero (P : ℝ → ℝ) (c d : ℝ) :
  (c > 0) →
  (d > 0) →
  (∀ n : ℕ, c * (n : ℝ)^3 ≤ |P n| ∧ |P n| ≤ d * (n : ℝ)^3) →
  (∃ a₀ a₁ a₂ a₃ : ℝ, ∀ x, P x = a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀) →
  ∃ x : ℝ, P x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_has_real_zero_l966_96668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_crease_length_l966_96690

/-- The length of the crease when folding a rectangular paper -/
noncomputable def creaseLength (paperWidth : ℝ) (θ : ℝ) : ℝ :=
  (paperWidth * Real.tan θ) / (1 + Real.tan θ)

/-- Theorem stating the length of the crease for a specific paper width -/
theorem fold_crease_length (θ : ℝ) :
  creaseLength 8 θ = (8 * Real.tan θ) / (1 + Real.tan θ) := by
  -- Unfold the definition of creaseLength
  unfold creaseLength
  -- The rest of the proof is omitted
  sorry

#check fold_crease_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_crease_length_l966_96690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_none_given_not_A_l966_96634

-- Define the sample space
variable (Ω : Type)
-- Define the probability measure
variable (P : Set Ω → ℝ)

-- Define events for each hazard
variable (A B C : Set Ω)

-- Define the conditions
variable (h1 : P (A \ (B ∪ C)) = 0.2)
variable (h2 : P (B \ (A ∪ C)) = 0.2)
variable (h3 : P (C \ (A ∪ B)) = 0.2)
variable (h4 : P ((A ∩ B) \ C) = 0.1)
variable (h5 : P ((B ∩ C) \ A) = 0.1)
variable (h6 : P ((A ∩ C) \ B) = 0.1)
variable (h7 : P (A ∩ B ∩ C) / P (A ∩ B) = 1/2)

-- Define the theorem
theorem probability_none_given_not_A :
  P ((Set.univ \ (A ∪ B ∪ C))) / P (Set.univ \ A) = 11/9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_none_given_not_A_l966_96634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_magnitude_of_vector_sum_l966_96640

/-- The minimum magnitude of a + λb where a = (2,1), b = (1,2), and λ ∈ ℝ is 3√5/5 -/
theorem min_magnitude_of_vector_sum (a b : ℝ × ℝ) :
  a = (2, 1) →
  b = (1, 2) →
  (∃ (min : ℝ), min = (3 * Real.sqrt 5) / 5 ∧
    ∀ (l : ℝ), ‖a + l • b‖ ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_magnitude_of_vector_sum_l966_96640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_to_planes_parallel_to_intersection_l966_96654

/-- Represents a 3D space -/
structure Space3D where
  -- Add necessary structure here

/-- Represents a plane in 3D space -/
structure Plane where
  -- Add necessary structure here

/-- Represents a line in 3D space -/
structure Line where
  -- Add necessary structure here

/-- Defines the intersection of two planes -/
def plane_intersection (α β : Plane) : Line :=
  sorry

/-- Defines when a line is parallel to a plane -/
def line_parallel_to_plane (m : Line) (α : Plane) : Prop :=
  sorry

/-- Defines when two lines are parallel -/
def lines_parallel (l m : Line) : Prop :=
  sorry

/-- Theorem: If a line is parallel to two intersecting planes, it is parallel to their intersection line -/
theorem parallel_to_planes_parallel_to_intersection 
  (α β : Plane) (l m : Line) 
  (h_intersection : plane_intersection α β = l)
  (h_parallel_α : line_parallel_to_plane m α)
  (h_parallel_β : line_parallel_to_plane m β) :
  lines_parallel m l := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_to_planes_parallel_to_intersection_l966_96654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_three_element_set_l966_96647

theorem subsets_of_three_element_set :
  let S : Finset ℕ := {1, 2, 3}
  Finset.card (Finset.powerset S) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subsets_of_three_element_set_l966_96647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diff_eq_solution_l966_96659

/-- Given a second-order linear differential equation y'' + 10y' + 25y = 4e^(-5x),
    prove that y(x) = (C₁ + C₂x)e^(-5x) + 2x²e^(-5x) is its general solution,
    where C₁ and C₂ are arbitrary constants. -/
theorem diff_eq_solution (x : ℝ) (C₁ C₂ : ℝ) :
  let y : ℝ → ℝ := λ x => (C₁ + C₂ * x) * Real.exp (-5 * x) + 2 * x^2 * Real.exp (-5 * x)
  (deriv^[2] y) + 10 * (deriv y) + 25 * y = λ x => 4 * Real.exp (-5 * x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diff_eq_solution_l966_96659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_side_length_sum_l966_96656

/-- QuadrilateralABCD represents a quadrilateral with vertices A, B, C, and D -/
def QuadrilateralABCD (A B C D : EuclideanPlane) : Prop := sorry

/-- SegmentLength represents the length of a line segment between two points -/
def SegmentLength (P Q : EuclideanPlane) : ℝ := sorry

/-- MeasureAngle represents the measure of an angle in degrees -/
def MeasureAngle (P Q R : EuclideanPlane) : ℝ := sorry

/-- Given a quadrilateral ABCD with specified side lengths and angles, 
    prove that the sum of p and q in the expression AB = p + √q equals 14. -/
theorem quadrilateral_side_length_sum (A B C D : EuclideanPlane) (p q : ℤ) : 
  QuadrilateralABCD A B C D →
  SegmentLength B C = 10 →
  SegmentLength C D = 15 →
  SegmentLength A D = 13 →
  MeasureAngle B A D = 70 →
  MeasureAngle A B C = 70 →
  SegmentLength A B = p + Real.sqrt (q : ℝ) →
  p + q = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_side_length_sum_l966_96656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_sum_distances_verify_m_plus_n_l966_96687

-- Define the points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (12, 0)
def C : ℝ × ℝ := (5, 7)
def P : ℝ × ℝ := (5, 3)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem fermat_point_sum_distances :
  distance P A + distance P B + distance P C = Real.sqrt 34 + Real.sqrt 58 + 4 := by
  sorry

-- The sum of coefficients
def m_plus_n : ℕ := 2

-- Verify that m_plus_n is indeed 2
theorem verify_m_plus_n : m_plus_n = 2 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fermat_point_sum_distances_verify_m_plus_n_l966_96687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_composites_theorem_l966_96645

/-- A function that checks if a number is composite -/
def isComposite (k : ℕ) : Prop :=
  k > 1 ∧ ∃ m, 1 < m ∧ m < k ∧ k % m = 0

/-- A function that checks if there exist n consecutive composite numbers less than n factorial -/
def existsNConsecutiveComposites (n : ℕ) : Prop :=
  ∃ k, k > 0 ∧ k + n ≤ Nat.factorial n ∧ ∀ i, k ≤ i ∧ i < k + n → isComposite i

/-- Theorem stating the condition for n -/
theorem consecutive_composites_theorem (n : ℕ) :
  n > 0 ∧ ¬(existsNConsecutiveComposites n) ↔ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := by
  sorry

#check consecutive_composites_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_composites_theorem_l966_96645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_k_prime_y_l966_96674

def sequence_x : ℕ → ℕ → ℕ
  | a, 0 => a
  | a, n + 1 => 2 * sequence_x a n + 1

def sequence_y (a : ℕ) (i : ℕ) : ℕ :=
  2^(sequence_x a i) - 1

theorem largest_k_prime_y (a : ℕ) :
  ∃ k : ℕ, k = 2 ∧
  (∀ i ≤ k, Nat.Prime (sequence_y a i)) ∧
  (∀ m > k, ¬(∀ i ≤ m, Nat.Prime (sequence_y a i))) :=
by
  sorry

#eval sequence_x 2 3
#eval sequence_y 2 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_k_prime_y_l966_96674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l966_96653

theorem trig_identity (α : ℝ) : 
  Real.sin (π + α)^2 + Real.cos (2*π + α) * Real.cos (-α) - 1 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l966_96653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l966_96651

/-- Geometric sequence sum -/
noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

/-- Theorem: In a geometric sequence where S_2 = 7 and S_6 = 91, S_4 = 28 -/
theorem geometric_sequence_sum (a : ℝ) (q : ℝ) :
  geometric_sum a q 2 = 7 →
  geometric_sum a q 6 = 91 →
  geometric_sum a q 4 = 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l966_96651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_sides_l966_96652

noncomputable def point_A : ℝ × ℝ := (-1, -1)
noncomputable def point_B : ℝ × ℝ := (0, 2)

noncomputable def parabola1 (x : ℝ) : ℝ := 2*x^2 + 4*x
noncomputable def parabola2 (x : ℝ) : ℝ := x^2/2 - x - 3/2
noncomputable def parabola3 (x : ℝ) : ℝ := -x^2 + 2*x - 1
noncomputable def parabola4 (x : ℝ) : ℝ := -x^2 - 4*x - 3
noncomputable def parabola5 (x : ℝ) : ℝ := -x^2 + 3

def same_side (f : ℝ → ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  (f p1.1 - p1.2) * (f p2.1 - p2.2) > 0

theorem parabola_sides :
  (same_side parabola1 point_A point_B) ∧
  (¬ same_side parabola2 point_A point_B) ∧
  (same_side parabola3 point_A point_B) ∧
  (¬ same_side parabola4 point_A point_B) ∧
  (same_side parabola5 point_A point_B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_sides_l966_96652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l966_96600

theorem exponential_equation_solution (x : ℝ) : (3 : ℝ)^(2*x + 1) = 81 → x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l966_96600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_given_prob_definition_polygon_has_nine_sides_l966_96679

/-- The number of sides in a regular polygon. -/
def num_sides : ℕ := 9

/-- The probability of selecting a shortest diagonal from all diagonals. -/
def prob_shortest_diagonal : ℚ := 1/3

/-- The total number of diagonals in the polygon. -/
def total_diagonals : ℕ := num_sides * (num_sides - 3) / 2

/-- The number of shortest diagonals in the polygon. -/
def num_shortest_diagonals : ℕ := num_sides

/-- The probability of selecting a shortest diagonal is the given value. -/
theorem prob_given : prob_shortest_diagonal = 1/3 := by rfl

/-- The probability of selecting a shortest diagonal is the ratio of shortest diagonals to total diagonals. -/
theorem prob_definition : prob_shortest_diagonal = (num_shortest_diagonals : ℚ) / (total_diagonals : ℚ) := by
  sorry

/-- Theorem: The regular polygon has 9 sides. -/
theorem polygon_has_nine_sides : num_sides = 9 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_given_prob_definition_polygon_has_nine_sides_l966_96679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_problem_l966_96629

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Sum of reciprocals of Fibonacci products -/
def fibSum (n : ℕ) : ℚ :=
  (List.range n).foldl (fun acc i => acc + 1 / ((List.range (i + 1)).foldl (fun prod j => prod * fib j) 1)) 0

/-- Main theorem -/
theorem fibonacci_problem (n : ℕ) (hn : n > 0) :
  (∃ a : ℕ, a > 0 ∧ fib n ≤ a ∧ a ≤ fib (n + 1) ∧ (a * fibSum n).num % (a * fibSum n).den = 0) ↔ n ∈ ({1, 2, 3} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_problem_l966_96629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_replacement_factor_five_l966_96657

theorem digit_replacement_factor_five (N : ℕ) : 
  (N ≥ 10^199 ∧ N < 10^200) →
  (∃ (k : ℕ) (a : ℕ), a ≤ 9 ∧ 
    N = (N / (10^(k+1))) * 10^(k+1) + a * 10^k + (N % 10^k) ∧
    5 * ((N / (10^(k+1))) * 10^(k+1) + (N % 10^k)) = N) →
  ∃ (a : ℕ), a ∈ ({1, 2, 3} : Set ℕ) ∧ N = 125 * a * 10^197 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_replacement_factor_five_l966_96657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l966_96623

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom f_differentiable : Differentiable ℝ f

axiom f_equation : ∀ x > 0, x * (deriv f x) + f x = (log x) / x

axiom f_at_e : f (Real.exp 1) = 1 / (Real.exp 1)

theorem f_inequality (x : ℝ) :
  x > 0 → (f x - x > 1 / (Real.exp 1) - (Real.exp 1) ↔ 0 < x ∧ x < (Real.exp 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l966_96623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_parabola_and_line_l966_96667

-- Define the parabola and line functions
def parabola (x : ℝ) : ℝ := 2 * x^2
def line (x : ℝ) : ℝ := 2 * x

-- Define the area function
noncomputable def area_between_curves (f g : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, f x - g x

-- Theorem statement
theorem area_enclosed_by_parabola_and_line :
  area_between_curves line parabola 0 1 = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_parabola_and_line_l966_96667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_l966_96637

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Whether a point is in the second quadrant -/
def inSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The origin point (0, 0) -/
def origin : Point :=
  ⟨0, 0⟩

/-- A point on the y-axis with the same y-coordinate as the given point -/
def yAxisProjection (p : Point) : Point :=
  ⟨0, p.y⟩

/-- A point on the x-axis with the same x-coordinate as the given point -/
def xAxisProjection (p : Point) : Point :=
  ⟨p.x, 0⟩

theorem point_coordinates (M : Point) 
  (h1 : inSecondQuadrant M)
  (h2 : M.y > 0)
  (h3 : distance M (xAxisProjection M) = 2)
  (h4 : distance M (yAxisProjection M) = 1) :
  M.x = -1 ∧ M.y = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_l966_96637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continued_fractions_and_golden_ratio_l966_96658

/-- The continued fraction [0; 1,1,1, ...] -/
noncomputable def continued_fraction_0 : ℝ :=
  Real.sqrt 5 / 2 - 1 / 2

/-- The continued fraction [1; 1,1,1, ...] -/
noncomputable def continued_fraction_1 : ℝ :=
  (1 + Real.sqrt 5) / 2

/-- The golden ratio -/
noncomputable def φ : ℝ :=
  (1 + Real.sqrt 5) / 2

theorem continued_fractions_and_golden_ratio :
  continued_fraction_0 = 1 / φ ∧
  continued_fraction_1 = φ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continued_fractions_and_golden_ratio_l966_96658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_sum_l966_96678

/-- Two-dimensional vector -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v w : Vec2D) : ℝ := v.x * w.x + v.y * w.y

/-- Magnitude (length) of a 2D vector -/
noncomputable def magnitude (v : Vec2D) : ℝ := Real.sqrt (dot_product v v)

/-- Angle between two 2D vectors in radians -/
noncomputable def angle (v w : Vec2D) : ℝ := Real.arccos ((dot_product v w) / (magnitude v * magnitude w))

/-- Scalar multiplication of a 2D vector -/
def scalar_mul (k : ℝ) (v : Vec2D) : Vec2D := Vec2D.mk (k * v.x) (k * v.y)

/-- Addition of two 2D vectors -/
def vec_add (v w : Vec2D) : Vec2D := Vec2D.mk (v.x + w.x) (v.y + w.y)

theorem vector_magnitude_sum (a b : Vec2D) : 
  angle a b = π / 3 → 
  a = Vec2D.mk 2 0 → 
  magnitude b = 1 → 
  magnitude (vec_add a (scalar_mul 2 b)) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_sum_l966_96678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_same_color_specific_distances_l966_96603

-- Define a type for colors
inductive Color
| Black
| Red

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define the distance function between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- State the theorem
theorem three_points_same_color_specific_distances :
  ∃ (p1 p2 p3 : Point),
    coloring p1 = coloring p2 ∧
    coloring p2 = coloring p3 ∧
    ((distance p1 p2 = 1 ∨ distance p1 p2 = Real.sqrt 3) ∧
     (distance p2 p3 = 1 ∨ distance p2 p3 = Real.sqrt 3) ∧
     (distance p3 p1 = 1 ∨ distance p3 p1 = Real.sqrt 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_same_color_specific_distances_l966_96603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrad_not_four_chromosomes_l966_96604

/-- Represents the properties of a tetrad -/
structure Tetrad :=
  (centromeres : Nat)
  (dna_molecules : Nat)
  (sister_chromatid_pairs : Nat)
  (chromosomes : Nat)

/-- The correct description of a tetrad -/
def correct_tetrad : Tetrad :=
  { centromeres := 2
  , dna_molecules := 4
  , sister_chromatid_pairs := 2
  , chromosomes := 2 }

/-- Theorem stating that a tetrad does not have 4 chromosomes -/
theorem tetrad_not_four_chromosomes : correct_tetrad.chromosomes ≠ 4 := by
  -- The proof would go here, but we'll use sorry for now
  sorry

#check tetrad_not_four_chromosomes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrad_not_four_chromosomes_l966_96604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_distance_l966_96627

/-- A parabola with equation y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  pos_p : p > 0

/-- A line passing through the point (0, 1) -/
structure Line where
  slope : Option ℝ  -- None represents infinite slope

/-- The focus of a parabola -/
noncomputable def focus (par : Parabola) : ℝ × ℝ := (par.p / 2, 0)

/-- Minimum distance from a point on the parabola to its focus -/
def min_distance_to_focus (par : Parabola) : ℝ := 2

/-- Condition that the line intersects the parabola at only one point -/
def intersects_once (par : Parabola) (l : Line) : Prop := sorry

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (p : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- The theorem to be proved -/
theorem parabola_line_distance (par : Parabola) (l : Line) :
  min_distance_to_focus par = 2 →
  intersects_once par l →
  distance_point_to_line (focus par) l ∈ ({1, 2, Real.sqrt 5} : Set ℝ) := by
  sorry

#check parabola_line_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_distance_l966_96627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_segment_length_l966_96611

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  base_diff : ℝ
  midpoint_ratio : ℝ × ℝ
  equal_area_segment : ℝ

/-- The trapezoid satisfying the problem conditions -/
def problem_trapezoid (x : ℝ) : Trapezoid where
  base_diff := 50
  midpoint_ratio := (3, 2)
  equal_area_segment := x

/-- The theorem statement -/
theorem trapezoid_segment_length (x : ℝ) (t : Trapezoid) 
  (h1 : t.base_diff = 50)
  (h2 : t.midpoint_ratio = (3, 2))
  (h3 : t.equal_area_segment = x)
  : ⌊x^2 / 50⌋ = 112 := by
  sorry

#check trapezoid_segment_length


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_segment_length_l966_96611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_people_time_l966_96646

/-- Represents the time taken to paint a house given a number of people -/
def paintTime (people : ℕ) : ℚ := sorry

/-- The constant representing the total work required to paint the house -/
def workConstant : ℚ := 5 * 4

/-- Assumption that 5 people can paint the house in 4 hours -/
axiom five_people_time : paintTime 5 = 4

/-- Assumption that the work rate is inversely proportional to the number of people -/
axiom inverse_proportion (n : ℕ) : n * paintTime n = workConstant

/-- Theorem stating that 6 people can paint the house in 10/3 hours -/
theorem six_people_time : paintTime 6 = 10 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_people_time_l966_96646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_sequence_l966_96665

noncomputable def a : ℕ → ℚ
  | 0 => 1994
  | n + 1 => (a n) ^ 2 / (a n + 1)

theorem floor_of_sequence (n : ℕ) (h : 1 ≤ n ∧ n ≤ 998) :
  ⌊a n⌋ = 1994 - n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_sequence_l966_96665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_problem_l966_96662

theorem exponential_problem (y : ℝ) (h : (3 : ℝ)^y = 81) : (3 : ℝ)^(y+3) = 2187 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_problem_l966_96662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_m_neg_one_max_value_of_f_l966_96699

-- Define the function
noncomputable def f (x m : ℝ) : ℝ := 2 * (Real.sin x)^2 + m * Real.cos x - 1/8

-- Theorem for part (1)
theorem range_of_f_when_m_neg_one (x : ℝ) (h : -π/3 ≤ x ∧ x ≤ 2*π/3) :
  ∃ y, f x (-1) = y ∧ -9/8 ≤ y ∧ y ≤ 2 := by
  sorry

-- Theorem for part (2)
theorem max_value_of_f (x m : ℝ) :
  ∃ max_y, (∀ x, f x m ≤ max_y) ∧
  (m < -4 → max_y = -m - 1/8) ∧
  (m > 4 → max_y = m - 1/8) ∧
  (-4 ≤ m ∧ m ≤ 4 → max_y = (m^2 + 15)/8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_m_neg_one_max_value_of_f_l966_96699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_problem_l966_96614

noncomputable def ferris_wheel_travel_time (radius : ℝ) (revolution_time : ℝ) (height : ℝ) : ℝ :=
  let angle := Real.arccos ((radius + height) / (2 * radius) - 1)
  2 * revolution_time * angle / (2 * Real.pi)

theorem ferris_wheel_problem :
  ferris_wheel_travel_time 30 120 15 = 20 := by
  -- Unfold the definition of ferris_wheel_travel_time
  unfold ferris_wheel_travel_time
  -- Simplify the expression
  simp
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_problem_l966_96614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joint_cdf_product_of_marginals_l966_96616

noncomputable section

/-- Density function for X -/
def p₁ (x : ℝ) : ℝ :=
  if x < -1 ∨ x > 1 then 0
  else if -1 < x ∧ x ≤ 1 then 0.5
  else 0

/-- Density function for Y -/
def p₂ (y : ℝ) : ℝ :=
  if y < 0 ∨ y > 2 then 0
  else if 0 < y ∧ y ≤ 2 then 0.5
  else 0

/-- Cumulative distribution function for X -/
def F₁ (x : ℝ) : ℝ :=
  if x ≤ -1 then 0
  else if -1 < x ∧ x ≤ 1 then 0.5 * (x + 1)
  else 1

/-- Cumulative distribution function for Y -/
def F₂ (y : ℝ) : ℝ :=
  if y ≤ 0 then 0
  else if 0 < y ∧ y ≤ 2 then 0.5 * y
  else 1

/-- Joint cumulative distribution function -/
def F (x y : ℝ) : ℝ := F₁ x * F₂ y

theorem joint_cdf_product_of_marginals (x y : ℝ) :
  F x y = F₁ x * F₂ y := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joint_cdf_product_of_marginals_l966_96616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l966_96609

/-- The infinite series 1 - x + x^2 - x^3 + x^4 - ... -/
noncomputable def infiniteSeries (x : ℝ) : ℝ := 1 / (1 + x)

/-- The theorem stating that the unique solution to the equation satisfying the convergence condition is (√5 - 1) / 2 -/
theorem unique_solution :
  ∃! x : ℝ, x = infiniteSeries x ∧ |x| < 1 ∧ x = (Real.sqrt 5 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l966_96609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l966_96677

theorem lambda_range (x y lambda : ℝ) :
  (3 * x^2 + 4 * y^2 = 1) →
  (∀ x' y', 3 * x'^2 + 4 * y'^2 = 1 →
    |3 * x + 4 * y - lambda| + |lambda + 7 - 3 * x - 4 * y| =
    |3 * x' + 4 * y' - lambda| + |lambda + 7 - 3 * x' - 4 * y'|) →
  lambda ∈ Set.Icc (Real.sqrt 7 - 7) (-Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l966_96677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_hexagram_arrangements_eq_factorial_div_symmetries_l966_96618

/-- A regular hexagram with 12 points -/
structure Hexagram where
  points : Fin 12 → Point

/-- The group of symmetries of a regular hexagram -/
def hexagram_symmetries : Type := sorry

/-- The number of symmetries of a regular hexagram -/
def num_hexagram_symmetries : ℕ := 12

/-- The number of ways to arrange n distinct objects -/
def num_arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of distinct arrangements on a hexagram -/
def distinct_hexagram_arrangements : ℕ :=
  num_arrangements 12 / num_hexagram_symmetries

theorem distinct_hexagram_arrangements_eq_factorial_div_symmetries :
  distinct_hexagram_arrangements = 479001600 := by
  sorry

#eval distinct_hexagram_arrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_hexagram_arrangements_eq_factorial_div_symmetries_l966_96618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_is_product_of_nonreal_zeros_l966_96682

/-- A cubic polynomial with three distinct real roots -/
structure CubicPolynomial where
  p : ℝ
  q : ℝ
  r : ℝ
  has_three_distinct_real_roots : True

/-- The polynomial function -/
def Q (c : CubicPolynomial) (x : ℝ) : ℝ := x^3 + c.p * x^2 + c.q * x + c.r

/-- The product of all zeros of Q -/
def product_of_zeros (c : CubicPolynomial) : ℝ := -c.r

/-- The sum of the real zeros of Q -/
def sum_of_real_zeros (c : CubicPolynomial) : ℝ := -c.p

theorem smallest_is_product_of_nonreal_zeros (c : CubicPolynomial) :
  let a := Q c (-1)
  let b := product_of_zeros c
  let c_val := (0 : ℝ) -- product of non-real zeros (which don't exist in this case)
  let d := 1 + c.p + c.q + c.r -- sum of coefficients
  let e := sum_of_real_zeros c
  c_val ≤ a ∧ c_val ≤ b ∧ c_val ≤ d ∧ c_val ≤ e := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_is_product_of_nonreal_zeros_l966_96682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lcm_with_18_l966_96686

def S : Finset Nat := {2, 4, 6, 9, 12, 16}

theorem largest_lcm_with_18 : 
  (Finset.sup S (fun x => Nat.lcm 18 x)) = 144 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_lcm_with_18_l966_96686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_points_concentric_circles_l966_96631

/-- Two non-intersecting circles -/
structure TwoCircles where
  O₁ : ℝ × ℝ  -- Center of first circle
  O₂ : ℝ × ℝ  -- Center of second circle
  R₁ : ℝ      -- Radius of first circle
  R₂ : ℝ      -- Radius of second circle
  a : ℝ       -- Distance between centers
  h₁ : R₁ > 0
  h₂ : R₂ > 0
  h₃ : a > R₁ + R₂  -- Circles do not intersect

/-- Points of tangency for external tangents -/
noncomputable def external_tangency_points (c : TwoCircles) : Set (ℝ × ℝ) := sorry

/-- Points of tangency for internal tangents -/
noncomputable def internal_tangency_points (c : TwoCircles) : Set (ℝ × ℝ) := sorry

/-- Points of intersection of internal and external tangents -/
noncomputable def tangent_intersection_points (c : TwoCircles) : Set (ℝ × ℝ) := sorry

/-- A circle in 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def on_circle (p : ℝ × ℝ) (C : Circle) : Prop :=
  (p.1 - C.center.1) ^ 2 + (p.2 - C.center.2) ^ 2 = C.radius ^ 2

/-- Theorem stating the main claims -/
theorem tangency_points_concentric_circles (c : TwoCircles) :
  ∃ (C₁ C₂ C₃ : Circle),
    (∀ p ∈ external_tangency_points c, on_circle p C₁) ∧
    (∀ p ∈ internal_tangency_points c, on_circle p C₂) ∧
    (∀ p ∈ tangent_intersection_points c, on_circle p C₃) ∧
    C₁.center = C₂.center ∧ C₂.center = C₃.center := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_points_concentric_circles_l966_96631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_lcm_bound_l966_96619

theorem sequence_lcm_bound (n : ℕ) (k : ℕ) (a : Fin k → ℕ) 
  (h0 : 0 < k)
  (h1 : ∀ i : Fin k, a i > 0) 
  (h2 : ∀ i j : Fin k, i.val < j.val → a i > a j) 
  (h3 : n ≥ a ⟨0, h0⟩) 
  (h4 : ∀ i j : Fin k, Nat.lcm (a i) (a j) ≤ n) : 
  ∀ i : Fin k, (i.val + 1) * a i ≤ n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_lcm_bound_l966_96619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l966_96694

def sequenceA (n : ℕ) : ℚ :=
  match n with
  | 0 => 0
  | 1 => 2
  | n+2 => 2 * (2 * n + 3)^2 * sequenceA (n+1) - 4 * (n+1)^2 * (2 * n + 1) * (2 * n + 3) * sequenceA n

theorem sequence_formula (n : ℕ) : sequenceA n = 2^n * n.factorial * (n+1).factorial := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l966_96694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l966_96660

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2 - 4*x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the interval [1/e, e]
def interval (x : ℝ) : Prop := 1/Real.exp 1 ≤ x ∧ x ≤ Real.exp 1

-- Define the mean value function property
def is_mean_value_function (h : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    (deriv h) ((x₁ + x₂) / 2) = (h x₂ - h x₁) / (x₂ - x₁)

theorem f_properties (a : ℝ) :
  (a = 1 → ∃ x y : ℝ, x = 1 ∧ f 1 x = y ∧ tangent_line x y) ∧
  (∀ x : ℝ, interval x → (f a x ≥ g a x) ↔ a ≤ -1) ∧
  (is_mean_value_function (f a) ↔ a = 0) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l966_96660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_maximum_l966_96630

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

-- State the theorem
theorem f_has_maximum : ∃ (x : ℝ), ∀ (y : ℝ), f y ≤ f x ∧ f x = 28/3 := by
  -- We'll use x = -2 as the maximum point
  use -2
  intro y
  have h1 : f (-2) = 28/3 := by
    -- Evaluate f(-2)
    simp [f]
    -- Simplify the arithmetic
    ring
  
  have h2 : f y ≤ f (-2) := by
    -- This part requires calculus and optimization techniques
    -- which are beyond the scope of this basic proof
    sorry

  -- Combine the two parts of the theorem
  constructor
  · exact h2
  · exact h1


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_maximum_l966_96630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_difference_l966_96661

theorem cube_root_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (x₁ ≠ x₂) ∧ 
  ((9 - x₁^2 / 4)^(1/3 : ℝ) = -3) ∧ 
  ((9 - x₂^2 / 4)^(1/3 : ℝ) = -3) ∧ 
  (|x₁ - x₂| = 24) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_difference_l966_96661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_number_deduction_l966_96639

def consecutive_integers (start : ℤ) : List ℤ :=
  List.range 10 |>.map (λ i => start + i)

theorem third_number_deduction
  (start : ℤ)
  (h1 : (consecutive_integers start).sum / 10 = 20)
  (h2 : let new_list := (consecutive_integers start).zipWith
          (λ x y => x - (9 - y : ℤ)) (List.range 10)
        (new_list.sum / (10 : ℚ)) = 31/2)
  : (consecutive_integers start)[2]! - ((consecutive_integers start).zipWith
      (λ x y => x - (9 - y : ℤ)) (List.range 10))[2]! = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_number_deduction_l966_96639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girl_scouts_percentage_approx_l966_96643

/-- Percentage of girl scouts with signed permission slips -/
def girl_scouts_with_slips (total_scouts : ℚ) : ℚ :=
  let total_with_slips := 0.8 * total_scouts
  let boy_scouts := 0.4 * total_scouts
  let boy_scouts_with_slips := 0.75 * boy_scouts
  let girl_scouts := total_scouts - boy_scouts
  let girl_scouts_with_slips := total_with_slips - boy_scouts_with_slips
  (girl_scouts_with_slips / girl_scouts) * 100

/-- Theorem stating that the percentage of girl scouts with signed permission slips
    is approximately 83% (rounded to the nearest percent) -/
theorem girl_scouts_percentage_approx (total_scouts : ℚ) (h : total_scouts > 0) :
  ∃ (ε : ℚ), abs ε < 0.5 ∧ girl_scouts_with_slips total_scouts = 83 + ε :=
by
  sorry

#eval girl_scouts_with_slips 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girl_scouts_percentage_approx_l966_96643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l966_96670

noncomputable def f (x : ℝ) : ℝ := 1 / (2 * x - 1)

theorem domain_of_f : Set ℝ = {x | x ≠ 1/2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l966_96670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l966_96696

structure Ellipse where
  C : Set (ℝ × ℝ)
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  isFoci : F₁ ∈ C ∧ F₂ ∈ C

def isOnEllipse (E : Ellipse) (P : ℝ × ℝ) : Prop := P ∈ E.C

def isPerpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

noncomputable def areaTriangle (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

noncomputable def perimeterTriangle (A B C : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) +
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) +
  Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)

def isStandardEllipseEquation (E : Ellipse) : Prop :=
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, (x, y) ∈ E.C ↔ x^2/a^2 + y^2/b^2 = 1) ∧
    (a^2 = 25 ∧ b^2 = 16 ∨ a^2 = 16 ∧ b^2 = 25))

theorem ellipse_theorem (E : Ellipse) (P : ℝ × ℝ) :
  isOnEllipse E P →
  isPerpendicular (P.1 - E.F₁.1, P.2 - E.F₁.2) (P.1 - E.F₂.1, P.2 - E.F₂.2) →
  areaTriangle P E.F₁ E.F₂ = 16 →
  perimeterTriangle P E.F₁ E.F₂ = 16 →
  isStandardEllipseEquation E :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l966_96696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l966_96641

/-- The distance from the center of the circle (x^2 + (y-1)^2 = 1) to the line (y = -x - 2) is √2/2 -/
theorem distance_circle_center_to_line :
  let circle := {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 1}
  let line := {p : ℝ × ℝ | p.2 = -p.1 - 2}
  let center : ℝ × ℝ := (0, 1)
  ∃ (closest : ℝ × ℝ), closest ∈ line ∧ dist center closest = Real.sqrt 2 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_circle_center_to_line_l966_96641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_word_sum_theorem_l966_96681

/-- The set of letters used to form words -/
inductive Letter : Type
| x : Letter
| y : Letter
| z : Letter

/-- A word is a list of letters -/
def Word := List Letter

/-- The set of all words -/
def S : Set Word := {w | True}

/-- Two words are similar if one can be obtained from the other by inserting
    a string from {xyz, yzx, zxy} somewhere in the word -/
def similar (u v : Word) : Prop := sorry

/-- A word is trivial if it can be obtained from the empty word by a sequence
    of similarity transformations -/
def is_trivial (w : Word) : Prop := sorry

/-- The number of trivial words of length 3n -/
def f (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem word_sum_theorem (p q : ℕ) (hpq : Nat.Coprime p q) :
  (p : ℚ) / q = ∑' n, (f n : ℚ) * (225 / 8192) ^ n → p + q = 61 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_word_sum_theorem_l966_96681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l966_96666

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.cos x - 3/4

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧
  f x = 1 ∧
  ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≤ 1 :=
by
  -- Proof goes here
  sorry

-- You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l966_96666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_profit_is_100_percent_l966_96675

-- Define the cost price and selling price
variable (C : ℝ) -- Cost price
variable (S : ℝ) -- Selling price

-- Define the profit percentage function
noncomputable def profit_percentage (cost : ℝ) (sell : ℝ) : ℝ :=
  ((sell - cost) / cost) * 100

-- State the theorem
theorem original_profit_is_100_percent 
  (h : profit_percentage C (2 * S) = 300) : 
  profit_percentage C S = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_profit_is_100_percent_l966_96675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_teacher_apples_l966_96608

theorem max_teacher_apples (n : ℕ) (children : ℕ) : 
  children = 8 → 
  ∃ (q : ℕ), n = children * q + (children - 1) ∧ 
  ∀ (r : ℕ), r < children → ∃ (q' : ℕ), n = children * q' + r :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_teacher_apples_l966_96608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exponential_sum_min_value_is_eight_min_value_achieved_l966_96648

theorem min_value_of_exponential_sum (a b : ℝ) (h : 2 * a + 3 * b = 4) :
  ∀ x y : ℝ, 2 * x + 3 * y = 4 → (4 : ℝ)^a + (8 : ℝ)^b ≤ (4 : ℝ)^x + (8 : ℝ)^y :=
by
  sorry

theorem min_value_is_eight (a b : ℝ) (h : 2 * a + 3 * b = 4) :
  (4 : ℝ)^a + (8 : ℝ)^b ≥ 8 :=
by
  sorry

theorem min_value_achieved (a b : ℝ) (h : 2 * a + 3 * b = 4) :
  ∃ x y : ℝ, 2 * x + 3 * y = 4 ∧ (4 : ℝ)^x + (8 : ℝ)^y = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exponential_sum_min_value_is_eight_min_value_achieved_l966_96648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tablet_battery_usage_time_l966_96635

/-- Represents the battery life of a tablet -/
structure TabletBattery where
  initialCharge : ℝ
  decreaseRate : ℝ
  timeUsed : ℝ
  remainingCharge : ℝ

/-- Calculates the additional time a tablet can be used before the battery reaches 0% -/
noncomputable def additionalUsageTime (tb : TabletBattery) : ℝ :=
  (tb.initialCharge / tb.decreaseRate) - tb.timeUsed

/-- Theorem stating the additional usage time for the given tablet scenario -/
theorem tablet_battery_usage_time 
  (tb : TabletBattery) 
  (h1 : tb.initialCharge = 100)
  (h2 : tb.timeUsed = 60)
  (h3 : tb.remainingCharge = 68)
  (h4 : tb.decreaseRate = (tb.initialCharge - tb.remainingCharge) / tb.timeUsed) :
  additionalUsageTime tb = 127.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tablet_battery_usage_time_l966_96635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equation_sum_l966_96606

theorem exponent_equation_sum (x y : ℝ) : 
  (7 : ℝ)^(3*x - 1) * (3 : ℝ)^(4*y - 3) = (49 : ℝ)^x * (27 : ℝ)^y → x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_equation_sum_l966_96606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_path_theorem_l966_96620

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a point on the surface of the cone -/
structure ConePoint where
  distanceFromVertex : ℝ
  angle : ℝ

/-- The least distance between two points on a cone's surface -/
noncomputable def leastDistance (c : Cone) (p1 p2 : ConePoint) : ℝ :=
  sorry

theorem fly_path_theorem (c : Cone) (start finish : ConePoint) :
  c.baseRadius = 500 →
  c.height = 300 * Real.sqrt 3 →
  start.distanceFromVertex = 150 →
  finish.distanceFromVertex = 450 * Real.sqrt 2 →
  finish.angle = start.angle + 5 * Real.pi / (2 * Real.sqrt 13) →
  leastDistance c start finish =
    Real.sqrt ((finish.distanceFromVertex * Real.cos finish.angle - start.distanceFromVertex * Real.cos start.angle)^2 +
               (finish.distanceFromVertex * Real.sin finish.angle - start.distanceFromVertex * Real.sin start.angle)^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_path_theorem_l966_96620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_k_in_range_l966_96676

open Real

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.exp 2 * x^2 + 1) / x
noncomputable def g (x : ℝ) : ℝ := (Real.exp 2 * x) / (Real.exp x)

-- State the theorem
theorem inequality_holds_iff_k_in_range :
  ∀ k : ℝ, k > 0 →
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → g x₁ / k ≤ f x₂ / (k + 1)) ↔
  k ≥ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_k_in_range_l966_96676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_is_constant_l966_96626

/-- A function satisfying the given properties on (0,1) -/
structure SpecialFunction where
  f : ℝ → ℝ
  domain : Set ℝ := Set.Ioo 0 1
  pos : ∀ x, x ∈ domain → f x > 0
  ineq : ∀ x y, x ∈ domain → y ∈ domain → f x / f y + f (1 - x) / f (1 - y) ≤ 2

/-- The main theorem: any function satisfying the properties is constant -/
theorem special_function_is_constant (sf : SpecialFunction) :
  ∀ x y, x ∈ sf.domain → y ∈ sf.domain → sf.f x = sf.f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_is_constant_l966_96626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangles_is_66_l966_96632

/-- The side length of the square paper in centimeters -/
def square_side : ℚ := 10

/-- The base of the right triangle in centimeters -/
def triangle_base : ℚ := 1

/-- The height of the right triangle in centimeters -/
def triangle_height : ℚ := 3

/-- The area of the square paper in square centimeters -/
def square_area : ℚ := square_side ^ 2

/-- The area of one right triangle in square centimeters -/
def triangle_area : ℚ := (triangle_base * triangle_height) / 2

/-- The maximum number of whole right triangles that can be cut from the square paper -/
def max_triangles : ℕ := (square_area / triangle_area).floor.toNat

/-- Theorem stating that the maximum number of triangles is 66 -/
theorem max_triangles_is_66 : max_triangles = 66 := by
  -- The proof goes here
  sorry

#eval max_triangles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangles_is_66_l966_96632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_move_adjacent_l966_96691

/-- Represents the state of a regular 12-sided polygon with vertices labeled 1 or -1 -/
def PolygonState := Fin 12 → Int

/-- Checks if a PolygonState is valid (has exactly one -1) -/
def isValidState (s : PolygonState) : Prop :=
  (∃! i, s i = -1) ∧ (∀ i, s i = 1 ∨ s i = -1)

/-- Applies the operation of changing signs of k consecutive vertices -/
def applyOperation (s : PolygonState) (start : Fin 12) (k : Nat) : PolygonState :=
  fun i => if (start.val + i.val) % 12 < k then -s ((start + i) % 12) else s i

/-- Checks if two states are adjacent (differ by one position) -/
def areAdjacent (s1 s2 : PolygonState) : Prop :=
  ∃ i j, i ≠ j ∧ s1 i = -1 ∧ s2 j = -1 ∧ (j = (i + 1) % 12 ∨ j = (i - 1 + 12) % 12) ∧
    (∀ l, l ≠ i ∧ l ≠ j → s1 l = s2 l)

/-- Theorem stating the impossibility of moving -1 to an adjacent vertex -/
theorem impossible_to_move_adjacent (k : Nat) (hk : k = 3 ∨ k = 4 ∨ k = 6) :
  ∀ s1 s2 : PolygonState, isValidState s1 → isValidState s2 → areAdjacent s1 s2 →
    ¬∃ (steps : List (Fin 12)), s2 = steps.foldl (fun acc i => applyOperation acc i k) s1 :=
by
  sorry

#check impossible_to_move_adjacent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_move_adjacent_l966_96691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_division_l966_96613

/-- Represents a cube with integer edge length -/
structure Cube where
  edge : ℕ

/-- Represents a collection of cubes -/
structure CubeCollection where
  cubes : List Cube

def volume (c : Cube) : ℕ := c.edge ^ 3

def totalVolume (cc : CubeCollection) : ℕ :=
  cc.cubes.map volume |>.sum

def isWholeNumber (n : ℕ) : Prop := True

theorem cube_division (cc : CubeCollection) : 
  (∃ c1 c2, c1 ∈ cc.cubes ∧ c2 ∈ cc.cubes ∧ c1.edge ≠ c2.edge) →
  (∀ c, c ∈ cc.cubes → isWholeNumber c.edge) →
  totalVolume cc = volume { edge := 3 } →
  cc.cubes.length = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_division_l966_96613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_value_l966_96622

/-- The perimeter of a triangle with vertices A(1, 2), B(6, 8), and C(1, 5) -/
noncomputable def trianglePerimeter : ℝ :=
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (6, 8)
  let C : ℝ × ℝ := (1, 5)
  let distAB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let distBC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let distCA := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  distAB + distBC + distCA

/-- Theorem: The perimeter of the triangle is equal to √61 + √34 + 3 -/
theorem triangle_perimeter_value :
  trianglePerimeter = Real.sqrt 61 + Real.sqrt 34 + 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_value_l966_96622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l966_96602

structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

theorem triangle_properties (t : Triangle) 
  (h1 : t.c = 2)
  (h2 : t.C = π/3) :
  (t.a = 2*Real.sqrt 3/3 → t.A = π/6) ∧ 
  (Real.sin t.B = 2 * Real.sin t.A → 
    (1/2) * t.a * t.b * Real.sin t.C = 2*Real.sqrt 3/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l966_96602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l966_96607

/-- A power function that passes through the point (9, 3) -/
noncomputable def f (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

/-- The theorem stating that f(1/3) = √3/3 for a power function passing through (9, 3) -/
theorem power_function_value (α : ℝ) (h : f α 9 = 3) : f α (1/3) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l966_96607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_circumcenters_l966_96601

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Represents a triangle -/
structure Triangle where
  P : Point2D
  Q : Point2D
  R : Point2D

/-- Returns the center of the circumscribed circle of a triangle -/
noncomputable def circumcenter (t : Triangle) : Point2D := sorry

/-- Represents a line segment between two points -/
def Segment (A B : Point2D) : Set Point2D := sorry

/-- Checks if a point is on a segment -/
def Point_on_segment (P : Point2D) (seg : Set Point2D) : Prop := P ∈ seg

/-- Checks if four points form a parallelogram -/
def is_parallelogram (A B C D : Point2D) : Prop := sorry

/-- The main theorem -/
theorem parallelogram_circumcenters 
  (ABCD : Parallelogram) 
  (P Q R S : Point2D)
  (h1 : Point_on_segment P (Segment ABCD.A ABCD.B))
  (h2 : Point_on_segment Q (Segment ABCD.B ABCD.C))
  (h3 : Point_on_segment R (Segment ABCD.C ABCD.D))
  (h4 : Point_on_segment S (Segment ABCD.D ABCD.A))
  : let t1 := Triangle.mk P Q ABCD.B
    let t2 := Triangle.mk Q R ABCD.C
    let t3 := Triangle.mk R S ABCD.D
    let t4 := Triangle.mk S P ABCD.A
    let c1 := circumcenter t1
    let c2 := circumcenter t2
    let c3 := circumcenter t3
    let c4 := circumcenter t4
    is_parallelogram c1 c2 c3 c4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_circumcenters_l966_96601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_rotated_line_l966_96697

/-- The equation of line l -/
def line_l (x y : ℝ) : Prop := 2 * x - 3 * y + 30 = 0

/-- The rotation angle in radians -/
noncomputable def rotation_angle : ℝ := Real.pi / 6

/-- The center of rotation -/
def rotation_center : ℝ × ℝ := (10, 10)

/-- The x-coordinate of the x-intercept of line k -/
noncomputable def x_intercept : ℝ := 15 * Real.sqrt 3 - 15

/-- Theorem stating that the x-coordinate of the x-intercept of line k is 15√3 - 15 -/
theorem x_intercept_of_rotated_line :
  ∃ (k : ℝ → ℝ → Prop),
    (∀ x y, line_l x y ↔ k (x * Real.cos rotation_angle - y * Real.sin rotation_angle + rotation_center.1)
                            (x * Real.sin rotation_angle + y * Real.cos rotation_angle + rotation_center.2)) →
    ∃ x, k x 0 ∧ x = x_intercept := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_rotated_line_l966_96697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_proof_l966_96624

noncomputable def circle_radius : ℝ := 1
noncomputable def central_angle_degrees : ℝ := 60
noncomputable def central_angle_radians : ℝ := central_angle_degrees * (Real.pi / 180)

theorem sector_area_proof :
  (1/2 : ℝ) * circle_radius^2 * central_angle_radians = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_proof_l966_96624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_A_l966_96684

noncomputable section

-- Define the curve
def f (x : ℝ) : ℝ := 1 / x

-- Define the point of tangency
def A : ℝ × ℝ := (1, 1)

-- Define the equation of the tangent line
def tangent_line (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem statement
theorem tangent_line_at_A : 
  ∀ x y : ℝ, tangent_line x y ↔ 
  (∃ m : ℝ, m = -(deriv f A.fst) ∧ y - A.snd = m * (x - A.fst)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_A_l966_96684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_sqrt_2_l966_96612

/-- The curve function y = x^2 - ln(x) --/
noncomputable def f (x : ℝ) : ℝ := x^2 - Real.log x

/-- The line function y = x - 2 --/
def g (x : ℝ) : ℝ := x - 2

/-- A point on the curve --/
structure PointOnCurve where
  x : ℝ
  y : ℝ
  h : y = f x

/-- The distance function from a point to the line y = x - 2 --/
noncomputable def distance_to_line (p : PointOnCurve) : ℝ :=
  |p.x + (-1) * p.y + 2| / Real.sqrt 2

theorem min_distance_is_sqrt_2 :
  ∃ (p : PointOnCurve), ∀ (q : PointOnCurve), distance_to_line p ≤ distance_to_line q ∧ distance_to_line p = Real.sqrt 2 := by
  sorry

#check min_distance_is_sqrt_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_sqrt_2_l966_96612
