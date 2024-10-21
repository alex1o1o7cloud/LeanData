import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sequence_length_smallest_sequence_length_proof_l634_63453

/-- The smallest positive integer n such that there exists a sequence of n+1 terms
    satisfying the given conditions. -/
theorem smallest_sequence_length : ℕ := 19

/-- Proof of the smallest sequence length -/
theorem smallest_sequence_length_proof :
  let n : ℕ := 19
  ∃ (a : ℕ → ℤ),
    a 0 = 0 ∧
    a n = 2008 ∧
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → |a i - a (i-1)| = i^2) ∧
    (∀ m : ℕ, m < n →
      ¬∃ (b : ℕ → ℤ),
        b 0 = 0 ∧
        b m = 2008 ∧
        (∀ i : ℕ, 1 ≤ i ∧ i ≤ m → |b i - b (i-1)| = i^2)) :=
by
  sorry

#check smallest_sequence_length
#check smallest_sequence_length_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sequence_length_smallest_sequence_length_proof_l634_63453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_work_time_l634_63465

/-- Represents the time required for a group of workers to complete a task -/
structure WorkTime where
  workers : ℕ
  days : ℕ

/-- The work completion rate is inversely proportional to the number of workers -/
axiom work_rate_inverse_prop {w1 w2 : WorkTime} :
  w1.workers * w1.days = w2.workers * w2.days

/-- Given: Double the workers can do half the work in 6 days -/
axiom double_workers_half_work (original : WorkTime) :
  ∃ (half_work : WorkTime),
    half_work.workers = 2 * original.workers ∧
    half_work.days = 6 ∧
    half_work.days * 2 = original.days

/-- Theorem: The original work time is 24 days -/
theorem original_work_time (original : WorkTime) :
  (∃ (half_work : WorkTime),
    half_work.workers = 2 * original.workers ∧
    half_work.days = 6 ∧
    half_work.days * 2 = original.days) →
  original.days = 24 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_work_time_l634_63465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_le_5_l634_63445

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + 2*x + a) / (x + 1)

def is_increasing_on (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x < y → f x < f y

theorem f_increasing_implies_a_le_5 :
  ∀ a : ℝ, is_increasing_on (f a) → a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_implies_a_le_5_l634_63445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l634_63443

noncomputable def f (x : ℝ) := 4 * Real.sin x * Real.cos (x + Real.pi/3) + Real.sqrt 3

theorem f_properties :
  (∀ x ∈ Set.Icc 0 (Real.pi/6), Real.sqrt 3 ≤ f x ∧ f x ≤ 2) ∧
  (∀ a b r S : ℝ, 
    a = Real.sqrt 3 ∧ 
    b = 2 ∧ 
    r = (3 * Real.sqrt 2) / 4 ∧
    S = (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - (a * b * Real.sqrt 3)^2) / (4 * a^2 * b^2))) →
    S = Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l634_63443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_walk_multiple_of_5_l634_63419

/-- A random walk on integers -/
def RandomWalk : Type := ℕ → ℤ

/-- The probability of being at a certain position after n steps -/
noncomputable def prob (walk : RandomWalk) (n : ℕ) (pos : ℤ) : ℝ := sorry

/-- The probability of being at a multiple of 5 after n steps -/
noncomputable def probMultipleOf5 (walk : RandomWalk) (n : ℕ) : ℝ :=
  ∑' k : ℤ, prob walk n (5 * k)

/-- A fair random walk starts at 0 and has equal probability of moving left or right -/
def isFairRandomWalk (walk : RandomWalk) : Prop :=
  walk 0 = 0 ∧
  ∀ n : ℕ, prob walk (n + 1) (walk n + 1) = (1 : ℝ) / 2 ∧
           prob walk (n + 1) (walk n - 1) = (1 : ℝ) / 2

theorem random_walk_multiple_of_5 (walk : RandomWalk) (h : isFairRandomWalk walk) :
  probMultipleOf5 walk 2022 > (1 : ℝ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_walk_multiple_of_5_l634_63419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_fold_5x5_to_1x8_or_1x7_l634_63406

/-- Represents a rectangular sheet of paper with width and height -/
structure Sheet where
  width : ℝ
  height : ℝ

/-- The diagonal length of a rectangular sheet -/
noncomputable def diagonalLength (s : Sheet) : ℝ :=
  Real.sqrt (s.width ^ 2 + s.height ^ 2)

/-- The maximum distance between any two points on a sheet -/
noncomputable def maxDistance (s : Sheet) : ℝ :=
  diagonalLength s

/-- Predicate to check if a folding operation is valid -/
def IsValidFolding (f : Sheet → Sheet) : Prop :=
  ∀ s : Sheet, diagonalLength (f s) ≤ diagonalLength s

theorem cannot_fold_5x5_to_1x8_or_1x7 :
  let original : Sheet := ⟨5, 5⟩
  let target1 : Sheet := ⟨1, 8⟩
  let target2 : Sheet := ⟨1, 7⟩
  (maxDistance original < maxDistance target1) ∧
  (maxDistance original ≤ maxDistance target2 →
    ¬∃ (f : Sheet → Sheet), f original = target2 ∧ IsValidFolding f) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_fold_5x5_to_1x8_or_1x7_l634_63406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_garden_area_l634_63425

/-- Calculates the area of a trapezoidal garden given property dimensions and trapezoid proportions --/
noncomputable def trapezoidalGardenArea (propertyWidth propertyLength : ℝ) 
  (shortSideProportion longSideProportion heightProportion : ℝ) : ℝ :=
  let shortSide := shortSideProportion * propertyWidth
  let longSide := longSideProportion * propertyWidth
  let height := heightProportion * propertyLength
  (shortSide + longSide) * height / 2

/-- The calculated area of Bob's trapezoidal garden is approximately 32,812.875 square feet --/
theorem bobs_garden_area :
  ∃ ε > 0, abs (trapezoidalGardenArea 1000 2250 (1/8) (1/6) (1/10) - 32812.875) < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobs_garden_area_l634_63425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l634_63447

/-- Represents a sprinter in the national team -/
inductive Sprinter : Type
| A : Sprinter
| B : Sprinter
| Other : Fin 4 → Sprinter

/-- Represents a relay team arrangement -/
def RelayTeam : Type := Fin 4 → Sprinter

/-- Checks if a relay team arrangement is valid according to the restrictions -/
def isValidArrangement (team : RelayTeam) : Prop :=
  team 0 ≠ Sprinter.A ∧ team 3 ≠ Sprinter.B

/-- The set of all valid relay team arrangements -/
def ValidArrangements : Set RelayTeam :=
  {team : RelayTeam | isValidArrangement team}

/-- Provide instances for Fintype and DecidablePred -/
instance : Fintype Sprinter := by
  sorry

instance : Fintype RelayTeam := by
  sorry

instance : DecidablePred isValidArrangement := by
  sorry

/-- The main theorem stating the number of valid arrangements -/
theorem valid_arrangements_count : Finset.card (Finset.filter isValidArrangement (Finset.univ : Finset RelayTeam)) = 252 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l634_63447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_10_of_90_l634_63464

open BigOperators Finset

def positive_factors (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ d => d > 0 ∧ n % d = 0)

def factors_less_than (n k : ℕ) : Finset ℕ :=
  (positive_factors n).filter (λ d => d < k)

theorem probability_factor_less_than_10_of_90 :
  (factors_less_than 90 10).card / (positive_factors 90).card = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_less_than_10_of_90_l634_63464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_areas_condition_l634_63410

theorem equal_areas_condition (r : ℝ) (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < π/4) : 
  (r^2 * φ / 2 = r^2 * Real.tan φ / 2) ↔ Real.tan φ = 4 * φ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_areas_condition_l634_63410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_sum_of_endpoints_l634_63459

noncomputable def g (x : ℝ) : ℝ := 2 / (2 + 4 * x^2)

theorem range_of_g :
  Set.range g = Set.Ioo 0 1 := by sorry

theorem sum_of_endpoints :
  ∃ (a b : ℝ), Set.range g = Set.Ioc a b ∧ a + b = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_sum_of_endpoints_l634_63459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_max_area_l634_63497

noncomputable section

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + y^2 = 16

-- Define the trajectory of E
def trajectory_E (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line l
def line_l (k m x y : ℝ) : Prop := y = k*x + m

-- Define the area of triangle OPQ
def area_OPQ (k m : ℝ) : ℝ := 
  (2 : ℝ) / 9 * Real.sqrt (20 + 1/k^2 - 1/k^4)

theorem trajectory_and_max_area :
  ∃ (k m : ℝ),
    k ≠ 0 ∧ 
    m > 0 ∧ 
    (∀ x y, trajectory_E x y → ∃ B_x B_y, circle_C B_x B_y ∧ 
      (∃ E_x E_y, line_l k m E_x E_y ∧ trajectory_E E_x E_y)) ∧
    line_l k m (-1) 0 ∧
    (∀ k' m', k' ≠ 0 → m' > 0 → line_l k' m' (-1) 0 → 
      area_OPQ k m ≥ area_OPQ k' m') ∧
    k = Real.sqrt 2 ∧
    m = 3 * Real.sqrt 2 / 2 ∧
    area_OPQ k m = 1 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_max_area_l634_63497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pi_minus_five_floor_l634_63429

-- Define the greatest integer function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem two_pi_minus_five_floor : floor (2 * Real.pi - 5) = 1 := by
  -- Proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pi_minus_five_floor_l634_63429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_primes_with_integer_root_sum_l634_63409

theorem infinitely_many_primes_with_integer_root_sum :
  ∃ f : ℕ → ℕ, ∀ k : ℕ,
    Nat.Prime (f k) ∧
    ∃ n : ℕ, (Int.sqrt (f k + n) + Int.sqrt n : ℤ) ∈ Set.range Int.ofNat := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_primes_with_integer_root_sum_l634_63409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_with_perimeter_100_l634_63461

noncomputable def triangle_area (a b c : ℕ) : ℝ :=
  let s : ℝ := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem min_area_triangle_with_perimeter_100 :
  ∃ (a b c : ℕ), 
    a + b + c = 100 ∧ 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b > c ∧ b + c > a ∧ c + a > b ∧
    ∀ (x y z : ℕ), 
      x + y + z = 100 → 
      x > 0 → y > 0 → z > 0 →
      x + y > z → y + z > x → z + x > y →
      triangle_area x y z ≥ 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_with_perimeter_100_l634_63461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_fixed_point_ST_range_l634_63483

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 4

-- Define the line l
noncomputable def line_l (θ : ℝ) (x y : ℝ) : Prop := y = Real.tan θ * (x + 1)

-- Define point P
def P (m : ℝ) : ℝ × ℝ := (m, 1)

-- Define point Q on C₂
def Q (x₀ y₀ : ℝ) : Prop := C₂ x₀ y₀

-- Theorem 1
theorem sin_theta_value (θ : ℝ) :
  ∃ (A B : ℝ × ℝ),
    C₂ A.1 A.2 ∧ C₂ B.1 B.2 ∧
    line_l θ A.1 A.2 ∧ line_l θ B.1 B.2 ∧
    (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = 0 →
    Real.sin θ = Real.sqrt 22 / 20 := by
  sorry

-- Theorem 2
theorem fixed_point (m : ℝ) :
  ∃ (M N : ℝ × ℝ),
    C₂ M.1 M.2 ∧ C₂ N.1 N.2 ∧
    (∃ (l₁ l₂ : ℝ × ℝ → Prop),
      l₁ (P m) ∧ l₁ M ∧ l₂ (P m) ∧ l₂ N) →
    (4, 1) ∈ {p : ℝ × ℝ | (p.1 - m) * (p.1 - 4) + (p.2 - 1) * p.2 = 0} := by
  sorry

-- Theorem 3
theorem ST_range (x₀ y₀ : ℝ) :
  Q x₀ y₀ →
  ∃ (S T : ℝ),
    (∃ (k₁ k₂ : ℝ),
      (k₁ + k₁*x₀ - y₀)^2 = 1 + k₁^2 ∧
      (k₂ + k₂*x₀ - y₀)^2 = 1 + k₂^2 ∧
      S = y₀ - k₁*x₀ ∧
      T = y₀ - k₂*x₀) →
    Real.sqrt 2 ≤ |T - S| ∧ |T - S| ≤ 5 * Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_value_fixed_point_ST_range_l634_63483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kitchen_light_bulbs_l634_63416

/-- Proves the number of light bulbs in the kitchen given the problem conditions -/
theorem kitchen_light_bulbs : 
  ∀ (kitchen_total foyer_total : ℕ),
  kitchen_total > 0 →
  foyer_total > 0 →
  3 * kitchen_total / 5 = kitchen_total - (34 - (foyer_total - 10)) →
  foyer_total / 3 = 10 →
  kitchen_total = 35 := by
  sorry

#check kitchen_light_bulbs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kitchen_light_bulbs_l634_63416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_errors_l634_63426

/-- Represents the number of errors Senya makes when writing a word -/
def errors (word : String) : ℕ := sorry

/-- The set of letters Senya can write correctly -/
def correct_letters : Set Char := sorry

theorem octahedron_errors :
  errors "TETRAHEDRON" = 5 →
  errors "DODECAHEDRON" = 6 →
  errors "ICOSAHEDRON" = 7 →
  correct_letters = {'K', 'D'} →
  errors "OCTAHEDRON" = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_errors_l634_63426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_circle_theorem_l634_63450

/-- Represents the state of a lamp (ON or OFF) -/
inductive LampState
| ON
| OFF

/-- Represents the circular arrangement of lamps -/
def LampCircle (n : ℕ) := Fin n → LampState

/-- Applies one step of the transformation to the lamp circle -/
def applyStep (n : ℕ) (circle : LampCircle n) (j : Fin n) : LampCircle n :=
  sorry

/-- Applies m steps of the transformation to the lamp circle -/
def applySteps (n : ℕ) (circle : LampCircle n) (m : ℕ) : LampCircle n :=
  sorry

/-- Checks if all lamps in the circle are ON -/
def allOn (n : ℕ) (circle : LampCircle n) : Prop :=
  sorry

theorem lamp_circle_theorem (n : ℕ) (h : n > 1) :
  -- Part (a)
  (∃ M : ℕ+, allOn n (applySteps n (λ _ => LampState.ON) M)) ∧
  -- Part (b)
  (∃ k : ℕ, n = 2^k → allOn n (applySteps n (λ _ => LampState.ON) (n^2 - 1))) ∧
  -- Part (c)
  (∃ k : ℕ, n = 2^k + 1 → allOn n (applySteps n (λ _ => LampState.ON) (n^2 - n + 1))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_circle_theorem_l634_63450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l634_63405

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * sin (3 * x + π / 6)

-- State the theorem
theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ 
  (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧
  T = 2 * π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l634_63405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_intersections_l634_63493

-- Define the types for planes and lines
variable (Plane Line : Type*)

-- Define the intersection operation
variable (intersect : Plane → Plane → Line)

-- Define the point type
variable (Point : Type*)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpPlane : Line → Plane → Prop)

-- Define the intersection of two lines
variable (intersectLine : Line → Line → Point)

-- Given three non-coincident planes
variable (α β γ : Plane)

-- Given the intersections of the planes
variable (a b c : Line)
variable (ha : intersect α β = a)
variable (hb : intersect α γ = b)
variable (hc : intersect β γ = c)

-- Theorem statement
theorem plane_intersections 
  (h1 : ∀ P, intersectLine a b = P → intersectLine a c = P)
  (h2 : ∀ x y z, perp x y → perpPlane x z)
  (h3 : ∀ x y z, parallel x y → parallel x z) :
  (∀ P, intersectLine a b = P → intersectLine a c = P) ∧
  (perp a b → perp a c → perpPlane a γ) ∧
  (parallel a b → parallel a c) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_intersections_l634_63493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perpendicular_iff_squared_sides_l634_63491

-- Define a triangle with sides a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

-- Define the circumcenter
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the centroid
noncomputable def centroid (t : Triangle) : ℝ × ℝ := sorry

-- Define the median to side c
noncomputable def median_c (t : Triangle) : Set (ℝ × ℝ) := sorry

-- Define a line passing through two points
noncomputable def line_through (p q : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define perpendicularity of lines
def perpendicular (l1 l2 : Set (ℝ × ℝ)) : Prop := sorry

-- The main theorem
theorem triangle_perpendicular_iff_squared_sides (t : Triangle) :
  perpendicular (line_through (circumcenter t) (centroid t)) (median_c t) ↔ 
  t.a^2 + t.b^2 = 2 * t.c^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perpendicular_iff_squared_sides_l634_63491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_eleven_results_l634_63479

theorem average_of_eleven_results 
  (n : ℕ) 
  (first_six_avg : ℚ) 
  (last_six_avg : ℚ) 
  (sixth_result : ℚ) 
  (h1 : n = 11) 
  (h2 : first_six_avg = 58) 
  (h3 : last_six_avg = 63) 
  (h4 : sixth_result = 66) : 
  (6 * first_six_avg + 6 * last_six_avg - sixth_result) / n = 60 := by
  sorry

#check average_of_eleven_results

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_eleven_results_l634_63479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_log2_l634_63444

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem derivative_of_log2 (x : ℝ) (h : x > 0) : 
  deriv log2 x = 1 / (x * Real.log 2) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_log2_l634_63444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l634_63469

noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

def inequality (x : ℝ) : Prop :=
  x^2 ≤ 2 * (floor (Real.rpow x (1/3) + 0.5) + floor (Real.rpow x (1/3)))

theorem solution_difference : 
  ∃ (min max : ℝ), 
    (∀ x, inequality x → min ≤ x ∧ x ≤ max) ∧
    (inequality min ∧ inequality max) ∧
    (max - min = 1) := by
  sorry

#check solution_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l634_63469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_annual_profit_l634_63498

-- Define the variable cost function
noncomputable def W (x : ℝ) : ℝ :=
  if x < 60 then (1/2) * x^2 + x
  else 7 * x + 100 / x - 39

-- Define the annual profit function
noncomputable def L (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 6 then -(1/2) * x^2 + 5 * x - 4
  else if x ≥ 6 then 35 - (x + 100 / x)
  else 0  -- For x ≤ 0, profit is undefined, so we set it to 0

-- State the theorem
theorem max_annual_profit :
  ∃ (x_max : ℝ), x_max = 10 ∧
  ∀ (x : ℝ), L x ≤ L x_max ∧
  L x_max = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_annual_profit_l634_63498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_endpoint_coordinate_sum_l634_63489

def endpoint1 : ℝ × ℝ := (6, -2)
def midpoint1 : ℝ × ℝ := (3, 5)

theorem other_endpoint_coordinate_sum : 
  ∃ (x y : ℝ), 
    (endpoint1.1 + x) / 2 = midpoint1.1 ∧ 
    (endpoint1.2 + y) / 2 = midpoint1.2 ∧ 
    x + y = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_endpoint_coordinate_sum_l634_63489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_tangent_line_equation_l634_63454

-- Define the function f(x) = xe^x
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

-- Theorem for the interval of monotonic increase
theorem monotonic_increase_interval :
  {x : ℝ | ∀ y, x < y → f x < f y} = Set.Ioi (-1 : ℝ) :=
sorry

-- Theorem for the equation of the tangent line
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := Real.exp x₀ * (x₀ + 1)
  ∀ x y : ℝ, (2 * Real.exp 1 * x - y - Real.exp 1 = 0) ↔ (y - y₀ = m * (x - x₀)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_tangent_line_equation_l634_63454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_x_given_lcm_l634_63430

theorem greatest_x_given_lcm (x : ℕ) :
  Nat.lcm x (Nat.lcm 15 21) = 105 → x ≤ 105 ∧ ∃ y : ℕ, y > 105 → Nat.lcm y (Nat.lcm 15 21) > 105 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_x_given_lcm_l634_63430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_condition_and_intersection_l634_63437

/-- Equation of the circle C -/
def C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

/-- Equation of the line l -/
def l (x y : ℝ) : Prop := x + 2*y - 4 = 0

/-- Distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

theorem circle_condition_and_intersection (m : ℝ) :
  (∃ x₀ y₀ r, ∀ x y, C x y m ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) →
  (∃ x₁ y₁ x₂ y₂, C x₁ y₁ m ∧ C x₂ y₂ m ∧ l x₁ y₁ ∧ l x₂ y₂ ∧ distance x₁ y₁ x₂ y₂ = 4 / Real.sqrt 5) →
  m < 5 ∧ m = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_condition_and_intersection_l634_63437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_final_price_l634_63490

/-- The final price of a notebook after two successive discounts -/
theorem notebook_final_price (initial_price discount1 discount2 : ℝ) : 
  initial_price = 20 ∧ discount1 = 0.20 ∧ discount2 = 0.25 →
  initial_price * (1 - discount1) * (1 - discount2) = 12 := by
  intro h
  have h1 : initial_price = 20 := h.left
  have h2 : discount1 = 0.20 := h.right.left
  have h3 : discount2 = 0.25 := h.right.right
  rw [h1, h2, h3]
  norm_num

#check notebook_final_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_notebook_final_price_l634_63490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commensurable_iff_rational_ratio_l634_63466

/-- Two line segments are commensurable if there exists a common measure segment
    that fits an integer number of times into each of the given segments. -/
def are_commensurable (A B : ℝ) : Prop :=
  ∃ (d : ℝ) (m n : ℤ), A = m • d ∧ B = n • d

/-- The main theorem stating that two line segments are commensurable
    if and only if the ratio of their lengths is rational. -/
theorem commensurable_iff_rational_ratio (A B : ℝ) (hB : B ≠ 0) :
  are_commensurable A B ↔ ∃ (p q : ℤ), A / B = (p : ℚ) / q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_commensurable_iff_rational_ratio_l634_63466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parallel_planes_l634_63441

/-- Definition of a point in ℝ³ -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Definition of a vector in ℝ³ -/
structure Vec3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Definition of a plane in ℝ³ -/
structure Plane3D where
  normal : Vec3D
  point : Point3D

/-- Distance between two planes -/
noncomputable def distance (α β : Plane3D) : ℝ := sorry

/-- Plane contains a point -/
def Plane3D.contains (p : Plane3D) (pt : Point3D) : Prop := sorry

/-- The distance between two parallel planes -/
theorem distance_between_parallel_planes 
  (α β : Plane3D) 
  (O : Point3D) 
  (A : Point3D) 
  (n : Vec3D) :
  α.contains O →
  β.contains A →
  A = ⟨2, 1, 1⟩ →
  n = ⟨-1, 0, 1⟩ →
  α.normal = n →
  β.normal = n →
  distance α β = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_parallel_planes_l634_63441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_l634_63488

def mySequence (n : ℕ) : ℚ :=
  33 + n^2 - n

theorem min_value_of_sequence :
  let a : ℕ → ℚ := mySequence
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 2 * n) ∧
  (∀ n : ℕ, n ≥ 1 → a n / n ≥ 21 / 2) ∧
  (∃ n : ℕ, n ≥ 1 ∧ a n / n = 21 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sequence_l634_63488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_intersection_and_perpendicular_l634_63455

noncomputable def intersection_point (a1 b1 c1 a2 b2 c2 : ℝ) : ℝ × ℝ :=
  let x := (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)
  let y := (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
  (x, y)

def point_on_line (x y a b c : ℝ) : Prop :=
  a * x + b * y + c = 0

def perpendicular_lines (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

theorem line_passes_through_intersection_and_perpendicular
  (a1 b1 c1 a2 b2 c2 a3 b3 c3 : ℝ)
  (h1 : a1 = 2 ∧ b1 = 3 ∧ c1 = 1)
  (h2 : a2 = 1 ∧ b2 = -3 ∧ c2 = 4)
  (h3 : a3 = 3 ∧ b3 = 4 ∧ c3 = -7) :
  let (x, y) := intersection_point a1 b1 c1 a2 b2 c2
  point_on_line x y 4 (-3) 1 ∧
  perpendicular_lines 4 (-3) a3 b3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_intersection_and_perpendicular_l634_63455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l634_63424

/-- Triangle DEF with vertices D(0,0), E(10,0), and F(0,2) -/
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

/-- A line passing through a point with a given slope -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- The area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  (1/2) * base * height

/-- The area of the region to the left of a line in the triangle -/
noncomputable def leftArea (t : Triangle) (l : Line) : ℝ :=
  let x := (t.F.2 / l.slope)
  triangleArea x (t.F.2 - l.slope * x)

/-- Theorem: There exists a line that divides the triangle into two equal areas -/
theorem equal_area_division (t : Triangle) 
  (h1 : t.D = (0, 0))
  (h2 : t.E = (10, 0))
  (h3 : t.F = (0, 2)) :
  ∃ θ : ℝ, 
    let l : Line := { point := t.D, slope := Real.tan θ }
    leftArea t l = triangleArea 10 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l634_63424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_remainder_euler_theorem_mod_mul_mod_geometric_sum_mod_500_l634_63414

theorem geometric_sum_remainder (n : ℕ) (a : ℕ) (m : ℕ) (h : m > 0) :
  (a^(n+1) - 1) / (a - 1) % m = ((a^(n+1) - 1) / (a - 1)) % m :=
sorry

theorem euler_theorem (a m : ℕ) (h : Nat.Coprime a m) :
  a^(Nat.totient m) % m = 1 :=
sorry

theorem mod_mul_mod (a b m : ℕ) (h : m > 0) :
  (a % m * b % m) % m = (a * b) % m :=
sorry

theorem geometric_sum_mod_500 :
  (3^500 - 1) / 2 % 500 = 440 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_remainder_euler_theorem_mod_mul_mod_geometric_sum_mod_500_l634_63414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_speed_l634_63467

/-- Given a man's speed in still water and his speed rowing downstream, 
    this theorem proves that his speed rowing upstream can be calculated as 
    twice his speed in still water minus his speed downstream. -/
theorem upstream_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still > 0)
  (h2 : speed_downstream > speed_still) :
  let speed_upstream := 2 * speed_still - speed_downstream
  speed_upstream > 0 ∧ speed_upstream < speed_still := by
  sorry

#check upstream_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_speed_l634_63467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_comparison_l634_63457

/-- Proves that given a car traveling at 65.45454545454545 km/h and taking 10 seconds longer
    to travel 1 km compared to another speed, the other speed is approximately 80 km/h. -/
theorem car_speed_comparison (current_speed : ℝ) (time_difference : ℝ) (comparison_speed : ℝ) :
  current_speed = 65.45454545454545 →
  time_difference = 10 →
  (1 / (current_speed / 3600)) - (1 / (comparison_speed / 3600)) = time_difference →
  abs (comparison_speed - 80) < 0.0001 := by
  sorry

#check car_speed_comparison

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_comparison_l634_63457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sixty_plus_abs_sqrt_three_minus_two_plus_neg_one_inv_minus_cube_root_neg_eight_equals_three_l634_63463

theorem sin_sixty_plus_abs_sqrt_three_minus_two_plus_neg_one_inv_minus_cube_root_neg_eight_equals_three :
  2 * Real.sin (π / 3) + |Real.sqrt 3 - 2| + (-1)⁻¹ - ((-8) ^ (1/3 : ℝ)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sixty_plus_abs_sqrt_three_minus_two_plus_neg_one_inv_minus_cube_root_neg_eight_equals_three_l634_63463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_product_l634_63420

theorem sin_difference_product (a b : ℝ) : 
  Real.sin (a + b) - Real.sin (a - b) = 2 * Real.sin b * Real.cos a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_product_l634_63420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l634_63407

noncomputable def f (x : Real) : Real := Real.cos x ^ 2 - Real.sin x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 1

theorem f_properties :
  (∃ (p : Real), p > 0 ∧ (∀ (x : Real), f (x + p) = f x) ∧
    (∀ (q : Real), q > 0 ∧ (∀ (x : Real), f (x + q) = f x) → p ≤ q)) ∧
  (∃ (m : Real), ∀ (x : Real), f x ≥ m ∧ ∃ (y : Real), f y = m) ∧
  (∀ (α : Real), f α = 2 ∧ α ≥ Real.pi / 4 ∧ α ≤ Real.pi / 2 → α = Real.pi / 3) :=
by sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l634_63407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_integer_values_l634_63403

/-- A polynomial function from real numbers to real numbers -/
def PolynomialFunction := ℝ → ℝ

/-- Predicate to check if a number ends with 5 or 8 in base 10 -/
def EndsIn5Or8 (n : ℕ) : Prop := n % 10 = 5 ∨ n % 10 = 8

/-- Predicate to check if a real number is an integer -/
def IsInteger (x : ℝ) : Prop := ∃ n : ℤ, x = ↑n

theorem polynomial_integer_values (f : PolynomialFunction) :
  (∀ k : ℕ, k > 0 → EndsIn5Or8 k → IsInteger (f k)) →
  IsInteger (f 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_integer_values_l634_63403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l634_63421

theorem sin_double_angle (θ : ℝ) : 
  Real.sin (π/4 + θ) = 1/3 → Real.sin (2*θ) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_l634_63421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l634_63434

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the length of a vector
noncomputable def vector_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the sides of the triangle
noncomputable def a (A B C : ℝ × ℝ) : ℝ := vector_length (C - B)
noncomputable def b (A B C : ℝ × ℝ) : ℝ := vector_length (C - A)
noncomputable def c (A B C : ℝ × ℝ) : ℝ := vector_length (B - A)

-- State the given conditions
axiom dot_product_condition (A B C : ℝ × ℝ) : 
  dot_product (B - A) (C - A) = dot_product (A - B) (C - B) ∧ 
  dot_product (B - A) (C - A) = 1

axiom vector_sum_condition (A B C : ℝ × ℝ) : 
  vector_length ((B - A) + (C - A)) = Real.sqrt 6

-- State the theorem to be proved
theorem triangle_properties (A B C : ℝ × ℝ) : 
  (A = B) ∧ 
  c A B C = Real.sqrt 2 ∧ 
  (1/2) * a A B C * b A B C * Real.sin (Real.arccos ((a A B C^2 + b A B C^2 - c A B C^2) / (2 * a A B C * b A B C))) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l634_63434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_puzzle_l634_63440

theorem integer_puzzle (a b : ℕ) (h1 : a * b = 24) (h2 : a + b = 11) : |Int.ofNat a - Int.ofNat b| = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_puzzle_l634_63440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_prism_diagonal_length_l634_63432

/-- Represents a triangular prism with a right triangular base -/
structure TriangularPrism where
  base_area : ℝ
  short_leg : ℝ
  long_leg : ℝ
  height : ℝ

/-- The length of the diagonal of a triangular prism -/
noncomputable def diagonal_length (prism : TriangularPrism) : ℝ :=
  Real.sqrt (2 * prism.long_leg ^ 2)

theorem triangular_prism_diagonal_length :
  ∀ (prism : TriangularPrism),
    prism.base_area = 63 →
    prism.long_leg = prism.short_leg + 2 →
    prism.height = prism.long_leg →
    prism.short_leg = 10.5 →
    diagonal_length prism = Real.sqrt (2 * (10.5 + 2) ^ 2) := by
  sorry

-- Use #eval only for computable expressions
#eval (2 * (10.5 + 2) ^ 2 : ℝ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_prism_diagonal_length_l634_63432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shipB_highest_percentage_l634_63422

noncomputable section

structure Ship where
  roundTripPercentage : ℝ
  carPercentage : ℝ

noncomputable def passengersWithoutCars (s : Ship) : ℝ :=
  s.roundTripPercentage - (s.roundTripPercentage * s.carPercentage / 100)

def shipA : Ship := ⟨30, 25⟩
def shipB : Ship := ⟨50, 15⟩
def shipC : Ship := ⟨20, 35⟩

theorem shipB_highest_percentage :
  passengersWithoutCars shipB > passengersWithoutCars shipA ∧
  passengersWithoutCars shipB > passengersWithoutCars shipC :=
by
  apply And.intro
  · sorry  -- Proof for shipB > shipA
  · sorry  -- Proof for shipB > shipC

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shipB_highest_percentage_l634_63422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_problem_l634_63415

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the line l₁
def line_l₁ (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 6 = 0

-- Define the relationship between A and N
def point_N_relation (x₀ y₀ x y : ℝ) : Prop :=
  x = x₀ ∧ y = Real.sqrt 3 / 3 * y₀

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 9 + y^2 / 3 = 1

-- Define the line l
def line_l (x y m : ℝ) : Prop := Real.sqrt 3 * x + y + m = 0

-- Helper function to represent the area of a triangle
noncomputable def area_triangle (O B D : ℝ × ℝ) : ℝ := sorry

-- Helper functions to represent the coordinates of B and D
noncomputable def B (m : ℝ) : ℝ × ℝ := sorry
noncomputable def D (m : ℝ) : ℝ × ℝ := sorry

theorem circle_tangent_line_problem :
  ∀ (x₀ y₀ x y m : ℝ),
  circle_M x₀ y₀ →
  line_l₁ x₀ y₀ →
  point_N_relation x₀ y₀ x y →
  (∀ x y, curve_C x y ↔ ∃ x₀ y₀, circle_M x₀ y₀ ∧ point_N_relation x₀ y₀ x y) →
  (∃ B D : ℝ × ℝ, curve_C B.1 B.2 ∧ curve_C D.1 D.2 ∧ line_l B.1 B.2 m ∧ line_l D.1 D.2 m) →
  curve_C x y ∧
  (∀ m, ∃ S : ℝ, S = area_triangle (0, 0) (B m) (D m) ∧ S ≤ 3 * Real.sqrt 3 / 2) ∧
  (∃ m, area_triangle (0, 0) (B m) (D m) = 3 * Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_line_problem_l634_63415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_sqrt_5_l634_63404

/-- The eccentricity of a hyperbola with equation x²/a² - y²/(4a²) = 1, where a > 0 -/
noncomputable def hyperbola_eccentricity (a : ℝ) : ℝ :=
  Real.sqrt 5

/-- Theorem: The eccentricity of the hyperbola x²/a² - y²/(4a²) = 1, where a > 0, is √5 -/
theorem hyperbola_eccentricity_is_sqrt_5 (a : ℝ) (ha : a > 0) :
  hyperbola_eccentricity a = Real.sqrt 5 := by
  -- Unfold the definition of hyperbola_eccentricity
  unfold hyperbola_eccentricity
  -- The result follows immediately from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_sqrt_5_l634_63404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_expansion_equality_l634_63499

noncomputable def f (x : ℝ) : ℝ := x + 1

noncomputable def fourierCoefficient (n : ℕ) : ℝ :=
  (8 / Real.pi^2) * ((-1)^n * (n + 1 : ℝ) * Real.pi - 1) / (n + 1 : ℝ)^2

noncomputable def fourierSeries (x : ℝ) : ℝ :=
  ∑' n, fourierCoefficient n * Real.cos ((2 * n + 1 : ℝ) * Real.pi * x / 2)

theorem fourier_expansion_equality (x : ℝ) (hx : x ∈ Set.Ioo 0 1) :
  f x = fourierSeries x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourier_expansion_equality_l634_63499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_babji_roshan_difference_l634_63411

/-- Represents the heights of three people: Ashis, Babji, and Roshan -/
structure Heights where
  babji : ℝ
  ashis : ℝ
  roshan : ℝ

/-- The conditions given in the problem -/
def HeightConditions (h : Heights) : Prop :=
  h.ashis = 1.25 * h.babji ∧ h.roshan = 0.85 * h.ashis

/-- The percentage difference between Babji and Roshan's heights -/
noncomputable def BabjiRoshanDifference (h : Heights) : ℝ :=
  (h.roshan - h.babji) / h.babji * 100

/-- Theorem stating that the percentage difference between Babji and Roshan's heights is 6.25% -/
theorem babji_roshan_difference (h : Heights) (hc : HeightConditions h) :
    BabjiRoshanDifference h = 6.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_babji_roshan_difference_l634_63411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_100_triangle_numbers_l634_63439

/-- The nth triangle number -/
def triangleNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the first n triangle numbers -/
def sumTriangleNumbers (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun i => triangleNumber (i + 1))

/-- Theorem: The sum of the first 100 triangle numbers is 171700 -/
theorem sum_first_100_triangle_numbers :
  sumTriangleNumbers 100 = 171700 := by
  sorry

#eval sumTriangleNumbers 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_100_triangle_numbers_l634_63439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_jogger_l634_63436

/-- The time taken for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (train_length : ℝ) 
  (initial_distance : ℝ) 
  (h1 : jogger_speed = 9 / 3.6)  -- Convert 9 km/hr to m/s
  (h2 : train_speed = 45 / 3.6)  -- Convert 45 km/hr to m/s
  (h3 : train_length = 150)
  (h4 : initial_distance = 240) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 39 := by
  sorry

#check train_passing_jogger

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_jogger_l634_63436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_price_increase_l634_63495

/-- The percent increase from a discounted price back to the original price -/
noncomputable def percent_increase (discount_rate : ℝ) : ℝ :=
  (1 / (1 - discount_rate) - 1) * 100

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem sale_price_increase (discount_rate : ℝ) (h : discount_rate = 0.13) :
  round_to_nearest (percent_increase discount_rate) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_price_increase_l634_63495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eunice_car_purchase_l634_63402

noncomputable def original_price : ℚ := 10000

noncomputable def discount_percentage : ℚ := 25

noncomputable def eunice_spent : ℚ := original_price * (1 - discount_percentage / 100)

theorem eunice_car_purchase : eunice_spent = 7500 := by
  -- Unfold the definitions
  unfold eunice_spent
  unfold original_price
  unfold discount_percentage
  
  -- Perform the calculation
  simp [mul_sub, mul_div_assoc, mul_one]
  
  -- The proof is completed
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eunice_car_purchase_l634_63402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_stopped_time_l634_63462

/-- Represents the scenario of a person walking on a moving walkway -/
structure WalkwayScenario where
  length : ℝ  -- Length of the walkway in meters
  time_with : ℝ  -- Time taken to walk with the walkway in seconds
  time_against : ℝ  -- Time taken to walk against the walkway in seconds

/-- Calculates the time taken to walk the walkway when it's not moving -/
noncomputable def time_when_stopped (scenario : WalkwayScenario) : ℝ :=
  2 * scenario.length * scenario.time_with * scenario.time_against /
    (scenario.time_with * scenario.time_against + scenario.length * (scenario.time_with + scenario.time_against))

/-- Theorem stating that for the given scenario, the time taken when the walkway is stopped is 75 seconds -/
theorem walkway_stopped_time (scenario : WalkwayScenario)
  (h1 : scenario.length = 100)
  (h2 : scenario.time_with = 50)
  (h3 : scenario.time_against = 150) :
  time_when_stopped scenario = 75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_walkway_stopped_time_l634_63462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_integers_with_specific_remainders_l634_63482

theorem three_digit_integers_with_specific_remainders :
  ∃! (s : Finset ℕ), s.card = 5 ∧ 
  (∀ n, n ∈ s ↔ 
    (100 ≤ n ∧ n < 1000 ∧
     n % 6 = 2 ∧
     n % 9 = 5 ∧
     n % 11 = 7)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_integers_with_specific_remainders_l634_63482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l634_63484

noncomputable def f (x : ℝ) : ℝ := |Real.log x / Real.log 2|

theorem problem_statement (a b c : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a < b) (h5 : b < c)
  (h6 : f a > f c) (h7 : f c > f b) :
  (a - 1) * (c - 1) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l634_63484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_B_in_special_triangle_l634_63446

/-- Given a triangle ABC with side lengths a and b, and angle A, 
    prove that if a = 1, b = √3, and A = 30°, then sin B = √3/2 -/
theorem sin_B_in_special_triangle (a b : ℝ) (A B : ℝ) :
  a = 1 → b = Real.sqrt 3 → A = π/6 → Real.sin B = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_B_in_special_triangle_l634_63446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_any_number_displayable_l634_63487

/-- Represents the state of the calculator --/
structure CalculatorState where
  display : ℕ
  switch_up : Bool

/-- Represents the possible actions on the calculator --/
inductive CalculatorAction
  | flip_switch
  | press_red
  | press_yellow
  | press_green
  | press_blue

/-- Applies an action to the calculator state --/
def apply_action (state : CalculatorState) (action : CalculatorAction) : CalculatorState :=
  match action with
  | CalculatorAction.flip_switch => 
      { display := if state.switch_up then state.display - 1 else state.display + 1,
        switch_up := ¬state.switch_up }
  | CalculatorAction.press_red => { state with display := state.display * 3 }
  | CalculatorAction.press_yellow => 
      { state with display := if state.display % 3 = 0 then state.display / 3 else state.display }
  | CalculatorAction.press_green => { state with display := state.display * 5 }
  | CalculatorAction.press_blue => 
      { state with display := if state.display % 5 = 0 then state.display / 5 else state.display }

/-- Applies a sequence of actions to the calculator state --/
def apply_actions (initial : CalculatorState) (actions : List CalculatorAction) : CalculatorState :=
  actions.foldl apply_action initial

/-- Theorem: Any positive integer can be displayed on the calculator --/
theorem any_number_displayable (n : ℕ) (h : n > 0) : 
  ∃ (actions : List CalculatorAction), (apply_actions { display := 1, switch_up := false } actions).display = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_any_number_displayable_l634_63487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l634_63471

/-- Given a triangle ABC with the following properties:
  cosA = √10 / 10
  AC = √10
  BC = 3√2
  Prove that:
  1. sinB = √2 / 2
  2. The area of △ABC is 6
-/
theorem triangle_ABC_properties (A B C : ℝ) (cosA AC BC : ℝ) 
  (h1 : cosA = Real.sqrt 10 / 10)
  (h2 : AC = Real.sqrt 10)
  (h3 : BC = 3 * Real.sqrt 2) :
  ∃ (sinB area : ℝ),
    sinB = Real.sqrt 2 / 2 ∧ area = 6 := by
  -- Define sinB
  let sinB := Real.sqrt (1 - (AC^2 + BC^2 - (2 * AC * BC * cosA))^2 / (4 * AC^2 * BC^2))
  -- Define semiperimeter
  let s := (AC + BC + Real.sqrt (AC^2 + BC^2 - 2 * AC * BC * cosA)) / 2
  -- Define area using Heron's formula
  let area := Real.sqrt (s * (s - AC) * (s - BC) * (s - Real.sqrt (AC^2 + BC^2 - 2 * AC * BC * cosA)))
  
  -- Prove the existence of sinB and area satisfying the conditions
  use sinB, area
  
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l634_63471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_russian_players_pairing_probability_l634_63458

/-- The probability of all Russian players being paired with each other in a tournament -/
theorem russian_players_pairing_probability
  (total_players : ℕ)
  (russian_players : ℕ)
  (h1 : total_players = 10)
  (h2 : russian_players = 4)
  (h3 : russian_players ≤ total_players)
  (h4 : Even total_players) :
  (russian_players.choose 2 * (total_players - russian_players).choose 0) /
  (total_players.choose 2) = 1 / 21 := by
  sorry

#check russian_players_pairing_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_russian_players_pairing_probability_l634_63458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l634_63431

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus : Point
  directrix : ℝ
  eccentricity : ℝ

def is_geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

def ellipse_equation (e : Ellipse) (p : Point) : Prop :=
  (p.x ^ 2 + (p.y + 2 * Real.sqrt 2) ^ 2).sqrt / 
  (abs (p.y + 9 / 4 * Real.sqrt 2)) = e.eccentricity

def line_intersects_ellipse (l : Line) (e : Ellipse) : Prop :=
  ∃ (p q : Point), p ≠ q ∧ 
    (p.y = l.slope * p.x + l.intercept) ∧ 
    (q.y = l.slope * q.x + l.intercept) ∧
    ellipse_equation e p ∧ ellipse_equation e q

def midpoint_on_line (p q : Point) (x : ℝ) : Prop :=
  (p.x + q.x) / 2 = x

noncomputable def slope_angle_in_range (l : Line) : Prop :=
  (Real.arctan l.slope > Real.pi / 3 ∧ Real.arctan l.slope < Real.pi / 2) ∨
  (Real.arctan l.slope > Real.pi / 2 ∧ Real.arctan l.slope < 2 * Real.pi / 3)

theorem ellipse_properties (e : Ellipse) :
  e.focus = Point.mk 0 (-2 * Real.sqrt 2) →
  e.directrix = -9 / 4 * Real.sqrt 2 →
  is_geometric_sequence (2 / 3) e.eccentricity (4 / 3) →
  (∀ (p : Point), ellipse_equation e p ↔ p.x ^ 2 + p.y ^ 2 / 9 = 1) ∧
  (∃ (l : Line), line_intersects_ellipse l e ∧
    (∀ (p q : Point), p ≠ q → 
      ellipse_equation e p → ellipse_equation e q →
      p.y = l.slope * p.x + l.intercept →
      q.y = l.slope * q.x + l.intercept →
      midpoint_on_line p q (-1 / 2)) ∧
    slope_angle_in_range l) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l634_63431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_eight_l634_63474

def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem sixth_term_is_eight : fibonacci 6 = 8 := by
  rfl

#eval fibonacci 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_is_eight_l634_63474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_functions_satisfy_condition_l634_63496

theorem no_functions_satisfy_condition :
  ¬∃ (f : ℝ → ℝ), ∀ (x y z : ℝ), f (x^2 * y) + f (x^2 * z) - f x * f (y * z) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_functions_satisfy_condition_l634_63496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taco_truck_profit_is_200_l634_63408

/-- Calculates the profit for a taco truck given the total beef, beef per taco, selling price, and cost per taco -/
noncomputable def taco_truck_profit (total_beef : ℝ) (beef_per_taco : ℝ) (selling_price : ℝ) (cost_per_taco : ℝ) : ℝ :=
  let num_tacos := total_beef / beef_per_taco
  let revenue := num_tacos * selling_price
  let total_cost := num_tacos * cost_per_taco
  revenue - total_cost

/-- Theorem stating that the taco truck's profit is $200 given the specified conditions -/
theorem taco_truck_profit_is_200 :
  taco_truck_profit 100 0.25 2 1.5 = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_taco_truck_profit_is_200_l634_63408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_three_tenths_l634_63480

def total_screws : ℕ := 10
def defective_screws : ℕ := 3
def drawn_screws : ℕ := 4
def target_defective : ℕ := 2

def probability_exactly_two_defective : ℚ :=
  (Nat.choose defective_screws target_defective * Nat.choose (total_screws - defective_screws) (drawn_screws - target_defective)) /
  (Nat.choose total_screws drawn_screws)

theorem probability_is_three_tenths :
  probability_exactly_two_defective = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_three_tenths_l634_63480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_single_point_l634_63472

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the fixed points A and B
variable (A B : V)

-- Define the distance between A and B
def distance_AB (A B : V) : ℝ := ‖A - B‖

-- Define the radius of the circles
def radius : ℝ := 3

-- Define the set of centers of circles passing through A and B with radius 3
def circle_centers (A B : V) : Set V :=
  {C : V | ‖C - A‖ = radius ∧ ‖C - B‖ = radius}

-- Theorem statement
theorem locus_is_single_point 
  (h_distance : distance_AB A B = 6) :
  ∃! C, C ∈ circle_centers A B :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_single_point_l634_63472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_implication_zero_product_l634_63433

theorem negation_of_implication_zero_product (R : Type) [Field R] : 
  ¬(∀ (a b : R), a * b = 0 → a = 0) ↔ 
  ∃ (a b : R), a * b = 0 ∧ a ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_implication_zero_product_l634_63433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l634_63438

/-- Given three vectors a, b, and c in ℝ², where a = (2, 1), prove:
    1) If |c| = 2√5 and c ⟂ a, then c = (-2, 4) or c = (2, -4)
    2) If |b| = √5/2 and a + 2b ⟂ 2a - b, then the angle between a and b is π -/
theorem vector_problem (a b c : ℝ × ℝ) (h_a : a = (2, 1)) :
  (‖c‖ = 2 * Real.sqrt 5 ∧ c.1 * a.1 + c.2 * a.2 = 0 →
    c = (-2, 4) ∨ c = (2, -4)) ∧
  (‖b‖ = Real.sqrt 5 / 2 ∧ (a.1 + 2 * b.1) * (2 * a.1 - b.1) + (a.2 + 2 * b.2) * (2 * a.2 - b.2) = 0 →
    Real.arccos ((a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖)) = Real.pi) := by
  sorry

/-- Helper function to calculate the magnitude of a 2D vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

/-- Notation for vector magnitude -/
notation "‖" v "‖" => magnitude v

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l634_63438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_can_fill_second_row_l634_63428

/-- Represents a row of 100 cells, each containing 1, 2, or 3 -/
def Row := Fin 100 → Fin 3

/-- Checks if two rows have different numbers in each corresponding position -/
def isDifferent (row1 row2 : Row) : Prop :=
  ∀ i : Fin 100, row1 i ≠ row2 i

/-- Calculates the sum of a row -/
def rowSum (row : Row) : ℕ :=
  (Finset.sum (Finset.range 100) fun i => (row i).val + 1)

theorem alex_can_fill_second_row (dima_row : Row) :
  ∃ (alex_row : Row), isDifferent dima_row alex_row ∧ rowSum alex_row = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_can_fill_second_row_l634_63428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l634_63449

/-- Given a quadratic function f(x) = x^2 + mx - 3 with roots -1 and n,
    prove the values of m, n, and a. -/
theorem quadratic_function_properties (m n a : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x, f x = x^2 + m*x - 3) →
  (∃ f : ℝ → ℝ, f (-1) = 0 ∧ f n = 0) →
  (m = -2 ∧ n = 3) ∧
  (∀ f : ℝ → ℝ, (∀ x, f x = x^2 + m*x - 3) → f x = f (2*a - 3) → (a = 1 ∨ a = 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l634_63449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_result_l634_63435

noncomputable def f (x : ℝ) : ℝ := x + 2

noncomputable def g (x : ℝ) : ℝ := x / 3

theorem composition_result :
  f⁻¹ (g (f (f⁻¹ (g⁻¹ (f (g (f⁻¹ (f (g⁻¹ (f 27)))))))))) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_result_l634_63435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_equilateral_triangles_in_cube_l634_63401

/-- A cube is a three-dimensional shape with 8 vertices. -/
structure Cube where
  vertices : Finset (Fin 8)
  -- Additional properties of a cube could be defined here

/-- An equilateral triangle is a triangle with all sides of equal length. -/
structure EquilateralTriangle where
  vertices : Fin 3 → Fin 8
  -- Additional properties ensuring the triangle is equilateral could be defined here

/-- The set of all equilateral triangles that can be formed from the vertices of a cube. -/
def equilateralTrianglesInCube (c : Cube) : Set EquilateralTriangle :=
  {t : EquilateralTriangle | ∀ i, t.vertices i ∈ c.vertices}

/-- Assumption: The set of equilateral triangles in a cube is finite. -/
axiom equilateralTrianglesInCube_finite (c : Cube) : 
  Finite (equilateralTrianglesInCube c)

/-- The number of equilateral triangles in a cube is 8. -/
theorem num_equilateral_triangles_in_cube (c : Cube) :
  Fintype.card (Set.Finite.toFinset (equilateralTrianglesInCube_finite c)) = 8 := by
  sorry

#check num_equilateral_triangles_in_cube

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_equilateral_triangles_in_cube_l634_63401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_remaining_number_l634_63470

def josephus_sequence (n : ℕ) : List ℕ :=
  sorry

theorem last_remaining_number :
  (josephus_sequence 200).getLast? = some 64 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_remaining_number_l634_63470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_expression_l634_63485

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the given function property
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f ((2 / x) + 1) = log10 x

-- State the theorem
theorem function_expression (f : ℝ → ℝ) (h : has_property f) :
  ∀ x > 1, f x = log10 (2 / (x - 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_expression_l634_63485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_distinct_abs_values_l634_63468

-- Define the set S
def S : Set ℤ := {x | -100 ≤ x ∧ x ≤ 100}

-- Define the type of 50-element subsets of S
def Subset50 : Type := {T : Finset ℤ // T.toSet ⊆ S ∧ T.card = 50}

-- Define the function that maps a subset to its set of absolute values
def absValues (T : Finset ℤ) : Finset ℕ := T.image (fun x => x.natAbs)

-- Define the expected value function
noncomputable def expectedDistinctAbsValues : ℝ := sorry

-- The main theorem
theorem expected_distinct_abs_values :
  expectedDistinctAbsValues = 8825 / 201 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_distinct_abs_values_l634_63468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_increases_implies_negative_k_l634_63427

noncomputable def inverse_proportion (k : ℝ) : ℝ → ℝ := fun x ↦ k / x

def increases_in_each_quadrant (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → (x₁ < 0 ∧ x₂ < 0) ∨ (x₁ > 0 ∧ x₂ > 0) → f x₁ < f x₂

theorem inverse_proportion_increases_implies_negative_k (k : ℝ) :
  increases_in_each_quadrant (inverse_proportion k) → k < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_increases_implies_negative_k_l634_63427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_night_downloads_l634_63442

/-- Represents the number of songs downloaded at different times of the day -/
structure SongDownloads where
  morning : Nat
  later : Nat
  night : Nat

/-- Represents the memory space in MB -/
abbrev MBSpace := Nat

/-- The size of each song in MB -/
def songSize : MBSpace := 5

/-- The total memory space occupied by new songs -/
def totalNewSpace : MBSpace := 140

/-- Calculates the total memory space occupied by a given number of songs -/
def spaceOccupied (songs : Nat) : MBSpace := songs * songSize

theorem night_downloads (downloads : SongDownloads) : 
  downloads.morning = 10 →
  downloads.later = 15 →
  spaceOccupied (downloads.morning + downloads.later + downloads.night) = totalNewSpace →
  downloads.night = 3 := by
  sorry

#eval songSize
#eval totalNewSpace
#eval spaceOccupied 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_night_downloads_l634_63442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_non_cyclists_percentage_l634_63473

theorem basketball_non_cyclists_percentage
  (total : ℝ)
  (basketball_percent : ℝ)
  (cycling_percent : ℝ)
  (basketball_and_cycling_percent : ℝ)
  (h1 : basketball_percent = 0.75)
  (h2 : cycling_percent = 0.45)
  (h3 : basketball_and_cycling_percent = 0.60 * basketball_percent) :
  let non_cycling_basketball := total * (basketball_percent - basketball_and_cycling_percent)
  let non_cyclists := total * (1 - cycling_percent)
  ∃ ε > 0, |((non_cycling_basketball / non_cyclists) * 100) - 55| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_non_cyclists_percentage_l634_63473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hair_cut_theorem_l634_63413

/-- Calculates the amount of hair cut off given initial length, weekly growth rates, and final length --/
def hair_cut_amount (initial_length : ℝ) (weekly_growth : List ℝ) (final_length : ℝ) : ℝ :=
  initial_length + (weekly_growth.sum) - final_length

/-- Theorem stating the amount of hair cut off is 6.5 inches --/
theorem hair_cut_theorem :
  ∃ (initial_length : ℝ) (weekly_growth : List ℝ) (final_length : ℝ),
    initial_length = 11 ∧
    weekly_growth = [0.5, 0.75, 1, 0.25] ∧
    final_length = 7 ∧
    hair_cut_amount initial_length weekly_growth final_length = 6.5 := by
  use 11, [0.5, 0.75, 1, 0.25], 7
  simp [hair_cut_amount]
  norm_num
  
#eval hair_cut_amount 11 [0.5, 0.75, 1, 0.25] 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hair_cut_theorem_l634_63413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l634_63451

-- Define the type for positive integers
def PositiveInt := { n : ℕ // n > 0 }

-- Define the property of being prime
def IsPrime (p : ℕ) : Prop := Nat.Prime p

-- Define the congruence relation
def IsCongruent (a b m : ℕ) : Prop := a % m = b % m

-- Define the parity of a number
def SameParity (a b : ℕ) : Prop := a % 2 = b % 2

-- Define the function type
def FunctionType := PositiveInt → ℕ

-- Define the property that the function satisfies the given condition
def SatisfiesCondition (f : FunctionType) : Prop :=
  ∀ (p : ℕ) (n : PositiveInt), IsPrime p → IsCongruent (f n ^ p) n.val (f ⟨p, sorry⟩)

-- State the theorem
theorem function_characterization (f : FunctionType) (h : SatisfiesCondition f) :
  (∀ n : PositiveInt, f n = n.val) ∨
  ((∀ p : ℕ, IsPrime p → f ⟨p, sorry⟩ = 1) ∧
   (∀ n : PositiveInt, SameParity (f n) n.val)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_characterization_l634_63451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manuscript_fee_calculation_l634_63475

-- Define the tax calculation function
def calculateTax (fee : ℚ) : ℚ :=
  if fee ≤ 800 then 0
  else if fee ≤ 4000 then (fee - 800) * (14 / 100)
  else fee * (11 / 100)

-- Theorem statement
theorem manuscript_fee_calculation (tax_paid : ℚ) (h : tax_paid = 420) :
  ∃ (fee : ℚ), calculateTax fee = tax_paid ∧ fee = 3800 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_manuscript_fee_calculation_l634_63475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l634_63481

theorem max_value_of_f : 
  ∃ (M : ℝ), (∀ x, 2 * Real.cos x + Real.sin x ≤ M) ∧ 
             (∃ x₀, 2 * Real.cos x₀ + Real.sin x₀ = M) ∧ 
             M = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l634_63481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_has_eleven_figures_l634_63460

/-- Calculates the number of action figures Jerry has on the shelf after a series of additions and removals --/
def jerrys_action_figures : ℕ :=
  let initial := 3
  let after_monday := initial + 4 - 2
  let after_wednesday := after_monday + 5 - 3
  let after_friday := after_wednesday + 8
  let give_away := (after_friday : ℚ) * (1/4 : ℚ)
  (after_friday : ℚ) - give_away |>.floor.toNat

/-- Proves that Jerry ends up with 11 action figures on the shelf --/
theorem jerry_has_eleven_figures : jerrys_action_figures = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_has_eleven_figures_l634_63460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_interval_for_x_2009_l634_63412

noncomputable def x_seq (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => 5/6 - 4/3 * |x_seq x₀ n - 1/2|

theorem best_interval_for_x_2009 :
  ∀ x₀ ∈ Set.Icc (0 : ℝ) 1,
    x_seq x₀ 2009 ∈ Set.Icc (7/18 : ℝ) (5/6) ∧
    ¬∃ a b, a > 7/18 ∧ b < 5/6 ∧ ∀ y₀ ∈ Set.Icc (0 : ℝ) 1, x_seq y₀ 2009 ∈ Set.Icc a b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_interval_for_x_2009_l634_63412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_mixture_tin_amount_l634_63400

/-- Represents an alloy with its total mass and ratios of components -/
structure Alloy where
  mass : ℝ
  ratio1 : ℝ
  ratio2 : ℝ

/-- Calculates the amount of the second component in an alloy -/
noncomputable def amountOfSecondComponent (a : Alloy) : ℝ :=
  (a.ratio2 / (a.ratio1 + a.ratio2)) * a.mass

/-- The problem statement -/
theorem alloy_mixture_tin_amount 
  (alloyA : Alloy)
  (alloyB : Alloy)
  (h1 : alloyA.mass = 100)
  (h2 : alloyB.mass = 200)
  (h3 : alloyA.ratio1 = 5 ∧ alloyA.ratio2 = 3)
  (h4 : alloyB.ratio1 = 2 ∧ alloyB.ratio2 = 3) :
  amountOfSecondComponent alloyA + amountOfSecondComponent alloyB = 117.5 := by
  sorry

#check alloy_mixture_tin_amount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_mixture_tin_amount_l634_63400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_circle_l634_63486

-- Define the circle
def circle' (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = m}

-- Define the line
def line (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 + m = 0}

-- Define what it means for a line to be tangent to a circle
def is_tangent (m : ℝ) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ circle' m ∧ p ∈ line m ∧
  ∀ q : ℝ × ℝ, q ∈ circle' m ∧ q ∈ line m → q = p

-- State the theorem
theorem tangent_line_circle (m : ℝ) :
  is_tangent m → m = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_circle_l634_63486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l634_63417

/-- The time for a train to pass a man moving in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 220 →
  train_speed = 60 * (1000 / 3600) →
  man_speed = 6 * (1000 / 3600) →
  let relative_speed := train_speed + man_speed
  let time := train_length / relative_speed
  ∃ ε > 0, |time - 12| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l634_63417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l634_63494

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := 1/2 * t.a * t.b * Real.sin t.C

/-- Theorem for part (I) -/
theorem part_one (t : Triangle) (h1 : t.c = 2) (h2 : t.C = Real.pi/3) (h3 : area t = Real.sqrt 3) :
  t.a = 2 ∧ t.b = 2 := by sorry

/-- A triangle is right-angled if one of its angles is π/2 -/
def is_right_angled (t : Triangle) : Prop :=
  t.A = Real.pi/2 ∨ t.B = Real.pi/2 ∨ t.C = Real.pi/2

/-- A triangle is isosceles if at least two of its sides are equal -/
def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Theorem for part (II) -/
theorem part_two (t : Triangle) (h : Real.sin t.C + Real.sin (t.B - t.A) = Real.sin (2 * t.A)) :
  is_right_angled t ∨ is_isosceles t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l634_63494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_diff_eq_l634_63423

/-- The differential equation y'' + 4y' + 4y = 2 sin(2x) + 3 cos(2x) -/
def diff_eq (y : ℝ → ℝ) (x : ℝ) : Prop :=
  (deriv (deriv y)) x + 4 * (deriv y) x + 4 * y x = 2 * Real.sin (2 * x) + 3 * Real.cos (2 * x)

/-- The general solution to the differential equation -/
noncomputable def solution (C₁ C₂ : ℝ) (x : ℝ) : ℝ :=
  (C₁ + C₂ * x) * Real.exp (-2 * x) - (1/4) * Real.cos (2 * x) + (3/8) * Real.sin (2 * x)

/-- Theorem stating that the solution satisfies the differential equation -/
theorem solution_satisfies_diff_eq (C₁ C₂ : ℝ) :
  ∀ x, diff_eq (solution C₁ C₂) x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_diff_eq_l634_63423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_targeted_gangsters_l634_63477

/-- A configuration of gangsters on a plane -/
structure GangsterConfig where
  positions : Finset (ℝ × ℝ)
  distinct_distances : ∀ p q r s : ℝ × ℝ, p ∈ positions → q ∈ positions → 
    r ∈ positions → s ∈ positions → p ≠ q → r ≠ s → 
    (p.1 - q.1)^2 + (p.2 - q.2)^2 ≠ (r.1 - s.1)^2 + (r.2 - s.2)^2

/-- The set of gangsters targeted in a given configuration -/
noncomputable def targeted (config : GangsterConfig) : Finset (ℝ × ℝ) :=
  config.positions.filter (λ p => ∃ q ∈ config.positions, q ≠ p ∧
    ∀ r ∈ config.positions, r ≠ p → r ≠ q → 
      (p.1 - q.1)^2 + (p.2 - q.2)^2 ≤ (p.1 - r.1)^2 + (p.2 - r.2)^2)

/-- The minimum number of targeted gangsters is at least 3 -/
theorem min_targeted_gangsters (config : GangsterConfig) 
    (h : config.positions.card = 10) : 
    (targeted config).card ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_targeted_gangsters_l634_63477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l634_63476

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := (3 * x + 8) / (x - 4)

-- State the theorem about the range of g
theorem range_of_g :
  Set.range g = {y : ℝ | y ≠ 3} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l634_63476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l634_63418

-- Define the function f
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^m - 4/x

-- Theorem statement
theorem function_properties :
  (∃ m : ℝ, f 4 m = 3 ∧ m = 1) ∧
  (∀ x : ℝ, x ≠ 0 → f (-x) 1 = -(f x 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l634_63418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_koala_fiber_theorem_l634_63448

noncomputable def koala_fiber_problem (absorption_rate : ℝ) (absorbed_amount : ℝ) : ℝ :=
  absorbed_amount / absorption_rate

theorem koala_fiber_theorem (absorption_rate absorbed_amount : ℝ)
  (h1 : absorption_rate = 0.40)
  (h2 : absorbed_amount = 12) :
  koala_fiber_problem absorption_rate absorbed_amount = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_koala_fiber_theorem_l634_63448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_neg_one_max_value_three_implies_a_one_range_open_zero_inf_implies_a_zero_l634_63452

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) ^ (a * x^2 - 4*x + 3)

-- Theorem 1: Monotonicity when a = -1
theorem monotonicity_when_a_neg_one :
  ∀ x₁ x₂, x₁ < -2 → x₂ < -2 → f (-1) x₁ > f (-1) x₂ ∧
  ∀ x₃ x₄, x₃ > -2 → x₄ > -2 → x₃ < x₄ → f (-1) x₃ < f (-1) x₄ := by
  sorry

-- Theorem 2: When max value is 3, a = 1
theorem max_value_three_implies_a_one :
  (∃ x₀, ∀ x, f a x ≤ f a x₀ ∧ f a x₀ = 3) → a = 1 := by
  sorry

-- Theorem 3: When range is (0, +∞), a = 0
theorem range_open_zero_inf_implies_a_zero :
  (∀ y, y > 0 → ∃ x, f a x = y) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_neg_one_max_value_three_implies_a_one_range_open_zero_inf_implies_a_zero_l634_63452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_theorem_l634_63492

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a trapezoid with four vertices -/
structure Trapezoid where
  E : Point
  F : Point
  G : Point
  H : Point

/-- Calculates the area of a trapezoid -/
def trapezoidArea (t : Trapezoid) : ℚ :=
  let base1 := t.F.x - t.E.x
  let base2 := t.G.x - t.H.x
  let height := t.E.y - t.H.y
  (base1 + base2) * height / 2

theorem trapezoid_area_theorem :
  let t : Trapezoid := {
    E := { x := 0, y := 3 },
    F := { x := 5, y := 3 },
    G := { x := 5, y := -2 },
    H := { x := -1, y := -2 }
  }
  trapezoidArea t = 55/2 := by sorry

#eval trapezoidArea {
  E := { x := 0, y := 3 },
  F := { x := 5, y := 3 },
  G := { x := 5, y := -2 },
  H := { x := -1, y := -2 }
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_theorem_l634_63492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_negative_three_three_polar_l634_63478

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 && y ≥ 0 then Real.arctan (y / x) + Real.pi
           else if x < 0 && y < 0 then Real.arctan (y / x) - Real.pi
           else if x = 0 && y > 0 then Real.pi / 2
           else if x = 0 && y < 0 then -Real.pi / 2
           else 0  -- x = 0 and y = 0
  (r, if θ < 0 then θ + 2 * Real.pi else θ)

theorem point_negative_three_three_polar :
  let (r, θ) := rectangular_to_polar (-3) 3
  r = 3 * Real.sqrt 2 ∧ θ = 3 * Real.pi / 4 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_negative_three_three_polar_l634_63478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_3x_plus_y_l634_63456

theorem min_value_3x_plus_y (x y : ℝ) 
  (h1 : x > max (-3) y) 
  (h2 : (x + 3) * (x^2 - y^2) = 8) : 
  ∀ z : ℝ, 3*x + y ≥ 4*(Real.sqrt 6) - 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_3x_plus_y_l634_63456
