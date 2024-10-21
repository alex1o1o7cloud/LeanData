import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_cos_product_l104_10465

theorem max_value_sin_cos_product : 
  ∀ x y z : ℝ, (Real.sin x + Real.sin (2*y) + Real.sin (3*z)) * (Real.cos x + Real.cos (2*y) + Real.cos (3*z)) ≤ 4.5 ∧ 
  ∃ a b c : ℝ, (Real.sin a + Real.sin (2*b) + Real.sin (3*c)) * (Real.cos a + Real.cos (2*b) + Real.cos (3*c)) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_cos_product_l104_10465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_set_l104_10479

def A : Set ℤ := {x | (x + 2) / (x - 3) ≤ 0}

def B : Set ℝ := {x | x ≤ -1 ∨ x > 3}

def B_int : Set ℤ := {x : ℤ | (x : ℝ) ∈ B}

theorem intersection_equals_set : A ∩ (Set.univ \ B_int) = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_set_l104_10479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factor_of_consecutive_sum_l104_10441

theorem prime_factor_of_consecutive_sum : ∃! p : ℕ, Prime p ∧ ∀ n : ℕ, p ∣ (4 * n + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factor_of_consecutive_sum_l104_10441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_length_is_5_5_l104_10448

/-- A room with a rectangular floor -/
structure Room where
  width : ℝ
  pavingCost : ℝ
  pavingRate : ℝ

/-- Calculate the length of a room given its properties -/
noncomputable def roomLength (r : Room) : ℝ :=
  (r.pavingCost / r.pavingRate) / r.width

/-- Theorem stating that a room with given properties has a length of 5.5 meters -/
theorem room_length_is_5_5 (r : Room) 
  (h1 : r.width = 3.75)
  (h2 : r.pavingCost = 24750)
  (h3 : r.pavingRate = 1200) : 
  roomLength r = 5.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_length_is_5_5_l104_10448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_time_bounds_l104_10439

/-- Represents the flow rate of water in gallons per minute -/
noncomputable def flow_rate (pressure : ℝ) : ℝ := 1.2 + 0.4 * pressure

/-- Represents the net flow rate after accounting for the leak -/
noncomputable def net_flow_rate (pressure : ℝ) : ℝ := flow_rate pressure - 0.1

/-- Represents the time required to fill the pool in minutes -/
noncomputable def fill_time (pressure : ℝ) : ℝ := 60 / net_flow_rate pressure

theorem pool_fill_time_bounds :
  ∀ p : ℝ, 1 ≤ p ∧ p ≤ 10 →
  11.76 ≤ fill_time p ∧ fill_time p ≤ 40 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_time_bounds_l104_10439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_height_change_rate_l104_10481

/-- Represents the volume of the solution as a function of time t (in seconds) --/
noncomputable def V (t : ℝ) : ℝ := Real.pi * t^3 + 2 * Real.pi * t^2

/-- Represents the radius of the cup's base in cm --/
def r : ℝ := 4

/-- Represents the height of the cup in cm --/
def cup_height : ℝ := 20

/-- Represents the area of the cup's base in cm² --/
noncomputable def S : ℝ := Real.pi * r^2

/-- Represents the height of the solution in the cup as a function of time t --/
noncomputable def h (t : ℝ) : ℝ := V t / S

/-- States that the instantaneous rate of change of the solution height at t=4s is 4cm/s --/
theorem solution_height_change_rate : 
  (deriv h) 4 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_height_change_rate_l104_10481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_adjacent_to_7_l104_10409

def divisors_147 : List Nat := [3, 7, 21, 49, 147]

def valid_arrangement (arr : List Nat) : Prop :=
  ∀ i j, (i + 1) % arr.length = j % arr.length →
    ∃ k > 1, k ∣ arr[i]! ∧ k ∣ arr[j]!

def adjacent_sum_to_7 (arr : List Nat) : Nat :=
  let i := arr.indexOf 7
  arr[(i - 1 + arr.length) % arr.length]! + arr[(i + 1) % arr.length]!

theorem sum_adjacent_to_7 :
  ∀ arr : List Nat, arr.Perm divisors_147 →
  valid_arrangement arr →
  adjacent_sum_to_7 arr = 70 := by
  sorry

#eval adjacent_sum_to_7 divisors_147

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_adjacent_to_7_l104_10409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pair_count_eq_odd_product_l104_10434

/-- The number of ways to pair 2n players -/
def pair_count (n : ℕ) : ℕ := (2 * n).factorial / (n.factorial * 2^n)

/-- The product of the first n odd numbers -/
def odd_product (n : ℕ) : ℕ := Finset.prod (Finset.range n) (λ i => 2 * i + 1)

/-- Theorem stating that the number of ways to pair 2n players
    is equal to the product of the first n odd numbers -/
theorem pair_count_eq_odd_product (n : ℕ) :
  pair_count n = odd_product n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pair_count_eq_odd_product_l104_10434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_cube_root_unity_l104_10483

/-- A cube root of unity -/
noncomputable def ω : ℂ :=
  sorry

/-- n is a multiple of 3 -/
def n (m : ℕ) : ℕ :=
  3 * m

/-- The sum s = 1 + 2ω + 3ω^2 + 4ω^3 + ... + (n+1)ω^n -/
noncomputable def s (m : ℕ) : ℂ :=
  sorry

theorem sum_cube_root_unity (m : ℕ) :
    ω ^ 3 = 1 ∧ ω ≠ 1 → s m = (3 * m : ℂ) + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_cube_root_unity_l104_10483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_score_is_43_l104_10401

def scores : List ℕ := [10, 7, 6, 8, 5, 9, 8, 8, 5, 6]
def times : List ℚ := [2/3, 1/2, 1/3, 2/3, 1/4, 2/3, 1/2, 2/5, 1/5, 1/4]
def costs : List ℕ := [1000, 700, 300, 800, 200, 900, 900, 600, 400, 600]

def isValidVector (v : List ℕ) : Prop :=
  v.length = 10 ∧
  v.all (λ x => x = 0 ∨ x = 1) ∧
  (List.sum (List.zipWith (·*·) v times) : ℚ) ≤ 3 ∧
  List.sum (List.zipWith (·*·) v costs) ≤ 3500

def vectorScore (v : List ℕ) : ℕ :=
  List.sum (List.zipWith (·*·) v scores)

theorem max_score_is_43 :
  ∃ (v : List ℕ), isValidVector v ∧ vectorScore v = 43 ∧
  ∀ (w : List ℕ), isValidVector w → vectorScore w ≤ 43 := by
  sorry

#eval "The theorem has been stated successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_score_is_43_l104_10401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_reciprocal_l104_10444

-- Define the function f(x) = 1/x
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- State the theorem
theorem derivative_of_reciprocal (x : ℝ) (hx : x ≠ 0) :
  deriv f x = -1 / (x^2) := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_reciprocal_l104_10444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_pi_over_four_l104_10416

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 1 - Real.sqrt (1 - x^2)
noncomputable def g (x : ℝ) : ℝ := 1 + Real.sqrt (1 - x^2)

-- Define the domain
def domain : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem enclosed_area_is_pi_over_four :
  (∫ x in domain, (g x - f x)) = π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_pi_over_four_l104_10416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_souvenir_pricing_and_plans_l104_10491

theorem souvenir_pricing_and_plans :
  ∃ (price_A price_B : ℕ) (num_plans : ℕ),
    price_A + 5 * price_B = 52 ∧
    3 * price_A + 4 * price_B = 68 ∧
    price_A = 12 ∧
    price_B = 8 ∧
    num_plans = (Finset.filter (λ t : ℕ ↦
      992 ≤ 12 * t + 8 * (100 - t) ∧
      12 * t + 8 * (100 - t) ≤ 1002
    ) (Finset.range 101)).card ∧
    num_plans = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_souvenir_pricing_and_plans_l104_10491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l104_10417

theorem sqrt_equation_solution (x : ℚ) : 
  (Real.sqrt (7 * x) / Real.sqrt (4 * (x - 2)) = 3) → x = 72 / 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_solution_l104_10417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l104_10468

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x + 4) + Real.sqrt (1 - x)) / x

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Icc (-4) 0 ∪ Set.Ioo 0 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l104_10468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sequence_properties_l104_10484

theorem roots_sequence_properties (x₁ x₂ : ℝ) (h : x₁^2 - 6*x₁ + 1 = 0 ∧ x₂^2 - 6*x₂ + 1 = 0) :
  let a : ℕ → ℝ := fun n => x₁^n + x₂^n
  ∀ n : ℕ, n > 0 → (∃ k : ℤ, a n = k) ∧ (∃ m : ℤ, a n = 5 * m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sequence_properties_l104_10484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_race_time_l104_10446

/-- A relay race with three runners --/
structure RelayRace where
  total_distance : ℚ
  sadie_speed : ℚ
  sadie_time : ℚ
  ariana_speed : ℚ
  ariana_time : ℚ
  sarah_speed : ℚ

/-- Calculate the total time for the relay race --/
def total_race_time (race : RelayRace) : ℚ :=
  let sadie_distance := race.sadie_speed * race.sadie_time
  let ariana_distance := race.ariana_speed * race.ariana_time
  let sarah_distance := race.total_distance - sadie_distance - ariana_distance
  let sarah_time := sarah_distance / race.sarah_speed
  race.sadie_time + race.ariana_time + sarah_time

/-- Theorem stating that the total race time is 4.5 hours --/
theorem relay_race_time : total_race_time
  { total_distance := 17
    sadie_speed := 3
    sadie_time := 2
    ariana_speed := 6
    ariana_time := 1/2
    sarah_speed := 4 } = 9/2 := by
  sorry

#eval total_race_time
  { total_distance := 17
    sadie_speed := 3
    sadie_time := 2
    ariana_speed := 6
    ariana_time := 1/2
    sarah_speed := 4 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relay_race_time_l104_10446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_K_nonnegative_when_L_nonpositive_l104_10410

/-- Prime factorization of a positive integer -/
def prime_factorization (n : ℕ+) : List (ℕ × ℕ) := sorry

/-- λ function as defined in the problem -/
def lambda (n : ℕ+) : Int :=
  let factors := prime_factorization n
  (-1) ^ (factors.map (·.2)).sum

/-- L function as defined in the problem -/
def L (n : ℕ) : Int :=
  (Finset.range n).sum (fun x => lambda ⟨x + 1, Nat.succ_pos x⟩)

/-- K function as defined in the problem -/
def K (n : ℕ) : Int :=
  (Finset.filter (fun x => ¬ Nat.Prime (x + 1)) (Finset.range n)).sum 
    (fun x => lambda ⟨x + 1, Nat.succ_pos x⟩)

/-- Main theorem -/
theorem K_nonnegative_when_L_nonpositive (N : ℕ) (h : ∀ n, 2 ≤ n → n ≤ N → L n ≤ 0) :
  ∀ n, 2 ≤ n → n ≤ N → K n ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_K_nonnegative_when_L_nonpositive_l104_10410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l104_10467

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - 3 * Real.pi / 4)

theorem min_positive_period_of_f (T : ℝ) :
  (∀ x, f (x + T) = f x) ∧
  (∀ T', 0 < T' ∧ T' < T → ∃ x, f (x + T') ≠ f x) →
  T = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l104_10467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sum_is_42_l104_10447

/-- The type representing a 7x7 matrix of integers -/
def Matrix7x7 := Fin 7 → Fin 7 → ℤ

/-- The sum of integers from -18 to 30, inclusive -/
def total_sum : ℤ := 294

/-- Checks if a given matrix contains all integers from -18 to 30 exactly once -/
def valid_matrix (m : Matrix7x7) : Prop := sorry

/-- The sum of a row in the matrix -/
def row_sum (m : Matrix7x7) (i : Fin 7) : ℤ := sorry

/-- The sum of a column in the matrix -/
def col_sum (m : Matrix7x7) (j : Fin 7) : ℤ := sorry

/-- The sum of the main diagonal (top-left to bottom-right) -/
def diag1_sum (m : Matrix7x7) : ℤ := sorry

/-- The sum of the other main diagonal (top-right to bottom-left) -/
def diag2_sum (m : Matrix7x7) : ℤ := sorry

/-- Theorem stating that if a valid 7x7 matrix of integers from -18 to 30 has equal sums
    for all rows, columns, and main diagonals, then this common sum must be 42 -/
theorem equal_sum_is_42 (m : Matrix7x7) (h : valid_matrix m) 
  (h_rows : ∀ i j : Fin 7, row_sum m i = row_sum m j)
  (h_cols : ∀ i j : Fin 7, col_sum m i = col_sum m j)
  (h_diag1 : ∀ i : Fin 7, row_sum m i = diag1_sum m)
  (h_diag2 : ∀ i : Fin 7, row_sum m i = diag2_sum m) :
  row_sum m 0 = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_sum_is_42_l104_10447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_shift_l104_10442

noncomputable def f (x : ℝ) := Real.cos (2 * x + Real.pi / 3)
noncomputable def g (x : ℝ) := Real.sin (2 * x + Real.pi / 3)

theorem cos_sin_shift :
  ∀ x : ℝ, f x = g (x + Real.pi / 4) :=
by
  intro x
  simp [f, g]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_shift_l104_10442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_circle_l104_10406

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The line 2x + y + 1 = 0 -/
def tangentLine (x y : ℝ) : Prop :=
  2 * x + y + 1 = 0

/-- The curve y = 2/x, where x > 0 -/
def centerCurve (x y : ℝ) : Prop :=
  x > 0 ∧ y = 2 / x

/-- Check if a circle is tangent to the line 2x + y + 1 = 0 -/
def isTangent (c : Circle) : Prop :=
  ∃ (x y : ℝ), tangentLine x y ∧ 
    (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- Check if a circle's center is on the curve y = 2/x, x > 0 -/
def hasCenterOnCurve (c : Circle) : Prop :=
  centerCurve c.center.1 c.center.2

/-- The specific circle (x-1)^2 + (y-2)^2 = 5 -/
noncomputable def smallestCircle : Circle :=
  { center := (1, 2), radius := Real.sqrt 5 }

theorem smallest_area_circle :
  ∀ c : Circle, isTangent c → hasCenterOnCurve c →
    c.radius^2 ≥ smallestCircle.radius^2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_circle_l104_10406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l104_10435

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

theorem f_property (a : ℝ) (m : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 3, ∀ y ∈ Set.Icc (-1 : ℝ) 3,
    a ∈ Set.Icc (-3 : ℝ) 0 →
    (∀ x₁ ∈ Set.Icc (0 : ℝ) 2, ∀ x₂ ∈ Set.Icc (0 : ℝ) 2,
      m - a * m^2 ≥ |f a x₁ - f a x₂|)) →
  m ∈ Set.Ioi 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l104_10435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_f_of_f_smallest_x_is_21_l104_10474

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 5)

theorem smallest_x_in_domain_of_f_of_f (x : ℝ) :
  x ∈ Set.range (f ∘ f) ↔ x ≥ 21 := by
  sorry

theorem smallest_x_is_21 :
  ∃ (x : ℝ), x ∈ Set.range (f ∘ f) ∧ ∀ (y : ℝ), y ∈ Set.range (f ∘ f) → y ≥ x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_f_of_f_smallest_x_is_21_l104_10474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_three_l104_10415

/-- A polynomial of degree 6 with integer coefficients between 0 and 4 inclusive -/
def MyPolynomial (b₀ b₁ b₂ b₃ b₄ b₅ b₆ : ℕ) (h₀ : b₀ < 5) (h₁ : b₁ < 5) (h₂ : b₂ < 5) 
  (h₃ : b₃ < 5) (h₄ : b₄ < 5) (h₅ : b₅ < 5) (h₆ : b₆ < 5) (x : ℝ) : ℝ :=
  b₀ + b₁ * x + b₂ * x^2 + b₃ * x^3 + b₄ * x^4 + b₅ * x^5 + b₆ * x^6

theorem polynomial_value_at_three 
  (b₀ b₁ b₂ b₃ b₄ b₅ b₆ : ℕ) 
  (h₀ : b₀ < 5) (h₁ : b₁ < 5) (h₂ : b₂ < 5) (h₃ : b₃ < 5) (h₄ : b₄ < 5) (h₅ : b₅ < 5) (h₆ : b₆ < 5)
  (h : MyPolynomial b₀ b₁ b₂ b₃ b₄ b₅ b₆ h₀ h₁ h₂ h₃ h₄ h₅ h₆ (Real.sqrt 5) = 35 + 26 * Real.sqrt 5) :
  MyPolynomial b₀ b₁ b₂ b₃ b₄ b₅ b₆ h₀ h₁ h₂ h₃ h₄ h₅ h₆ 3 = 437 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_three_l104_10415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_count_l104_10425

theorem undefined_values_count : 
  ∃! (S : Finset ℝ), (∀ x ∈ S, (x^2 + 3*x - 4) * (x - 4) = 0) ∧ Finset.card S = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_undefined_values_count_l104_10425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l104_10423

noncomputable def complex_number_z : ℂ := (1 - Complex.I) / (2 * Complex.I)

theorem z_in_third_quadrant :
  let z := complex_number_z
  Complex.re z < 0 ∧ Complex.im z < 0 :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l104_10423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_proof_l104_10440

noncomputable def α : ℝ := Real.arctan 3

theorem tan_alpha_proof :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 4/5 ∧
  Real.sin α ^ 2 + Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_proof_l104_10440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l104_10494

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem max_value_of_f :
  ∃ (c : ℝ), c > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≤ f c ∧ f c = 1 / exp 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l104_10494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PB_l104_10427

/-- The ellipse C is defined by the equation x²/5 + y² = 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 5 + p.2^2 = 1}

/-- B is the upper vertex of the ellipse C -/
def B : ℝ × ℝ := (0, 1)

/-- P is a point on the ellipse C -/
def P : C → ℝ × ℝ := fun p => p

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating that the maximum distance between P and B is 5/2 -/
theorem max_distance_PB :
  ∃ (p : C), ∀ (q : C), distance (P p) B ≤ distance (P q) B →
    distance (P p) B = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PB_l104_10427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semi_circle_perimeter_formula_semi_circle_perimeter_approx_l104_10428

/-- The perimeter of a semi-circle with radius 6.3 cm -/
noncomputable def semiCirclePerimeter : ℝ := Real.pi * 6.3 + 2 * 6.3

/-- Theorem stating that the perimeter of a semi-circle with radius 6.3 cm
    is equal to π * 6.3 + 2 * 6.3 cm -/
theorem semi_circle_perimeter_formula :
  semiCirclePerimeter = Real.pi * 6.3 + 2 * 6.3 := by
  rfl

/-- Theorem stating that the perimeter of a semi-circle with radius 6.3 cm
    is approximately 32.393 cm -/
theorem semi_circle_perimeter_approx :
  abs (semiCirclePerimeter - 32.393) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semi_circle_perimeter_formula_semi_circle_perimeter_approx_l104_10428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AD_approx_40_l104_10497

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The configuration of points A, B, C, and D -/
structure Configuration where
  A : Point
  B : Point
  C : Point
  D : Point
  B_east_of_A : B.x > A.x ∧ B.y = A.y
  C_north_of_B : C.x = B.x ∧ C.y > B.y
  AC_distance : distance A C = 15 * Real.sqrt 2
  BAC_angle : Real.arctan ((C.y - A.y) / (C.x - A.x)) = 30 * π / 180
  D_north_of_C : D.x = C.x ∧ D.y = C.y + 30

/-- The distance between A and D is approximately 40 -/
theorem distance_AD_approx_40 (config : Configuration) :
  ∃ ε > 0, |distance config.A config.D - 40| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AD_approx_40_l104_10497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_power_n_plus_reciprocal_l104_10463

theorem x_power_n_plus_reciprocal (θ : ℝ) (x : ℂ) (n : ℕ) 
  (h1 : 0 < θ) (h2 : θ < π) (h3 : x + 1/x = 2 * Real.sin θ) : 
  x^n + 1/x^n = 2 * Real.cos (n * θ + π/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_power_n_plus_reciprocal_l104_10463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_asymptote_distance_l104_10403

-- Define the hyperbola
def hyperbola (x y b : ℝ) : Prop := y^2 / 4 - x^2 / b^2 = 1

-- Define the condition that vertices bisect the focal distance
def vertices_bisect_focal_distance (a c : ℝ) : Prop := 2 * a = 2 * c / 3

-- Define the distance from focus to asymptote
noncomputable def distance_focus_to_asymptote (b : ℝ) : ℝ := 4 * Real.sqrt 2

-- Theorem statement
theorem hyperbola_focus_asymptote_distance :
  ∀ (b : ℝ), b > 0 →
  (∃ (x y : ℝ), hyperbola x y b) →
  (∃ (a c : ℝ), a > 0 ∧ c > 0 ∧ vertices_bisect_focal_distance a c) →
  distance_focus_to_asymptote b = 4 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_asymptote_distance_l104_10403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l104_10443

/-- A hyperbola with foci F₁ and F₂ -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A point on a hyperbola -/
def PointOnHyperbola (C : Hyperbola) (P : ℝ × ℝ) : Prop :=
  ∃ (a : ℝ), |P.1 - C.F₁.1| - |P.1 - C.F₂.1| = 2 * a

/-- The angle between two vectors -/
noncomputable def angle (v w : ℝ × ℝ) : ℝ := sorry

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (C : Hyperbola) : ℝ := sorry

/-- The main theorem -/
theorem hyperbola_eccentricity (C : Hyperbola) (P : ℝ × ℝ) 
  (h1 : PointOnHyperbola C P)
  (h2 : angle (C.F₁.1 - P.1, C.F₁.2 - P.2) (C.F₂.1 - P.1, C.F₂.2 - P.2) = π / 3)
  (h3 : (C.F₁.1 - P.1)^2 + (C.F₁.2 - P.2)^2 = 9 * ((C.F₂.1 - P.1)^2 + (C.F₂.2 - P.2)^2)) :
  eccentricity C = Real.sqrt 7 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l104_10443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_valid_colorings_l104_10414

/-- A coloring of integer points on a number line. -/
def Coloring := ℤ → Bool

/-- Check if a coloring is valid according to the given rules. -/
def is_valid_coloring (c : Coloring) : Prop :=
  (∀ x : ℤ, c (x + 7) = c x) ∧
  c 20 = true ∧ c 14 = true ∧
  c 71 = false ∧ c 143 = false

/-- The number of distinct valid colorings. -/
def num_valid_colorings : ℕ := 8

/-- Theorem stating that there are exactly 8 valid colorings. -/
theorem eight_valid_colorings : 
  ∃ (s : Finset Coloring), s.card = num_valid_colorings ∧ 
  (∀ c ∈ s, is_valid_coloring c) ∧
  (∀ c, is_valid_coloring c → c ∈ s) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_valid_colorings_l104_10414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l104_10469

open Real MeasureTheory

-- Define the functions f and g
noncomputable def f (a x : ℝ) := log x - x + 1 + a
noncomputable def g (x : ℝ) := x^2 * exp x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x₁ ∈ Set.Icc (1/exp 1) 1, ∃ x₂ ∈ Set.Icc 0 1, f a x₁ = g x₂) →
  a ∈ Set.Ioo (1/exp 1) (exp 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l104_10469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_pi_over_three_max_sin_A_plus_sin_B_max_sin_A_plus_sin_B_equality_l104_10476

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_angles : 0 < A ∧ 0 < B ∧ 0 < C
  h_sum_angles : A + B + C = π

-- Define the area of the triangle
noncomputable def area (t : Triangle) : ℝ := Real.sqrt 3 / 4 * (t.a^2 + t.b^2 - t.c^2)

theorem angle_C_is_pi_over_three (t : Triangle) (h : area t = Real.sqrt 3 / 4 * (t.a^2 + t.b^2 - t.c^2)) :
  t.C = π / 3 := by sorry

theorem max_sin_A_plus_sin_B (t : Triangle) (h : area t = Real.sqrt 3 / 4 * (t.a^2 + t.b^2 - t.c^2)) :
  Real.sin t.A + Real.sin t.B ≤ Real.sqrt 3 := by sorry

theorem max_sin_A_plus_sin_B_equality (t : Triangle) (h : area t = Real.sqrt 3 / 4 * (t.a^2 + t.b^2 - t.c^2)) :
  ∃ (t' : Triangle), Real.sin t'.A + Real.sin t'.B = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_pi_over_three_max_sin_A_plus_sin_B_max_sin_A_plus_sin_B_equality_l104_10476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OPQ_l104_10400

-- Define the curve C
noncomputable def C (x y : ℝ) : Prop := x^2 = 4*y

-- Define the point P
noncomputable def P : ℝ × ℝ := (Real.sqrt 2, 1/2)

-- Define the tangent line slope at P
noncomputable def tangent_slope (p : ℝ × ℝ) : ℝ := p.1 / 2

-- Define the perpendicular line l
noncomputable def l (x y : ℝ) : Prop :=
  y = -Real.sqrt 2 * x + 5/2

-- Define the other intersection point Q
noncomputable def Q : ℝ × ℝ := (-3*Real.sqrt 2, (54-30*Real.sqrt 2)/8)

-- Theorem statement
theorem area_of_triangle_OPQ :
  let O : ℝ × ℝ := (0, 0)
  C P.1 P.2 ∧ l Q.1 Q.2 ∧ C Q.1 Q.2 →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) * Real.sqrt (25/12) / 2 = 15 * Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OPQ_l104_10400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_part1_problem_part2_l104_10449

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (2 * x), Real.cos (2 * x))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (2 * x), -Real.cos (2 * x))

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem problem_part1 (x : ℝ) (h1 : x > 7 * Real.pi / 24) (h2 : x < 5 * Real.pi / 12)
  (h3 : dot_product (a x) (b x) + 1/2 = -3/5) :
  Real.cos (4 * x) = (3 - 4 * Real.sqrt 3) / 10 := by sorry

theorem problem_part2 (x : ℝ) (h1 : Real.cos x ≥ 1/2) (h2 : x > 0) (h3 : x < Real.pi)
  (h4 : ∃! m : ℝ, dot_product (a x) (b x) + 1/2 = m) :
  ∃ m : ℝ, (m = 1 ∨ m = -1/2) ∧ dot_product (a x) (b x) + 1/2 = m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_part1_problem_part2_l104_10449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_configuration_l104_10492

/-- A configuration of points and lines on a plane. -/
structure Configuration where
  points : Finset (Fin 7)
  lines : Finset (Fin 7)

/-- The incidence relation between points and lines. -/
def incidence (c : Configuration) : Fin 7 → Fin 7 → Bool := sorry

/-- A configuration is valid if each point is incident to exactly 3 lines
    and each line is incident to exactly 3 points. -/
def is_valid (c : Configuration) : Prop :=
  (∀ p ∈ c.points, (c.lines.filter (λ l => incidence c p l)).card = 3) ∧
  (∀ l ∈ c.lines, (c.points.filter (λ p => incidence c p l)).card = 3)

/-- Theorem: There exists no valid configuration of 7 points and 7 lines. -/
theorem no_valid_configuration : ¬ ∃ c : Configuration, is_valid c := by
  sorry

#check no_valid_configuration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_configuration_l104_10492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_3_negative_l104_10496

theorem tan_3_negative : Real.tan 3 < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_3_negative_l104_10496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_short_distance_A_short_distance_B_equidistant_C_D_l104_10462

noncomputable def short_distance (x y : ℝ) : ℝ := min (abs x) (abs y)

def equidistant (x1 y1 x2 y2 : ℝ) : Prop :=
  short_distance x1 y1 = short_distance x2 y2

-- Theorem statements
theorem short_distance_A : short_distance (-5) (-2) = 2 := by sorry

theorem short_distance_B (m : ℝ) : 
  short_distance (-2) (-2*m + 1) = 1 ↔ m = 0 ∨ m = 1 := by sorry

theorem equidistant_C_D (k : ℝ) : 
  equidistant (-1) (k + 3) 4 (2*k - 3) ↔ k = 1 ∨ k = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_short_distance_A_short_distance_B_equidistant_C_D_l104_10462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_line_l104_10411

noncomputable def f (x : ℝ) : ℝ := 1 / x

def line (x y : ℝ) : Prop := y = -x - 1

noncomputable def distance_to_line (x y a b c : ℝ) : ℝ :=
  |a * x + b * y + c| / Real.sqrt (a^2 + b^2)

theorem shortest_distance_to_line :
  ∃ (x₀ y₀ : ℝ), y₀ = f x₀ ∧
    (∀ (x y : ℝ), y = f x →
      distance_to_line x y 1 1 1 ≥ distance_to_line x₀ y₀ 1 1 1) ∧
    distance_to_line x₀ y₀ 1 1 1 = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_line_l104_10411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_in_interval_l104_10461

theorem no_solutions_in_interval (x : ℝ) : 
  x ∈ Set.Icc 0 π → 
  Real.sin (π * Real.sin x) ≠ Real.cos (π * Real.cos x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_in_interval_l104_10461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l104_10498

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 2| - 5

-- Define the function g
def g (x m : ℝ) : ℝ := f x + |x - m|

-- Define the set of x where f(x) ≥ |x-1|
def solution_set : Set ℝ := {x | f x ≥ |x - 1|}

-- Define the set of m where g(x) forms a triangle with x-axis when m ≥ -1
def m_range : Set ℝ := {m | m ≥ -1 ∧ ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧ g x₁ m = 0 ∧ g x₂ m = 0 ∧ ∀ x, x₁ < x ∧ x < x₂ → g x m > 0}

theorem problem_solution :
  solution_set = Set.Iic (-8) ∪ Set.Ici 2 ∧
  m_range = Set.Icc (3/2) 4 ∪ {-1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l104_10498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_three_l104_10433

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x else x^(1/3)

-- State the theorem
theorem f_composition_negative_three : f (f (-3)) = 1/2 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_three_l104_10433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_may_earnings_percentage_l104_10460

/-- Represents the percentage of Mrs. Bil's earnings in the family's total income --/
noncomputable def earnings_percentage (may_earnings rest_income : ℝ) : ℝ :=
  may_earnings / (may_earnings + rest_income)

/-- Theorem stating the relationship between May and June earnings --/
theorem may_earnings_percentage 
  (may_earnings rest_income : ℝ)
  (may_earnings_positive : may_earnings > 0)
  (rest_income_nonnegative : rest_income ≥ 0)
  (june_percentage : (1.1 * may_earnings) / (1.1 * may_earnings + rest_income) = 0.7196) :
  ∃ (ε : ℝ), abs (earnings_percentage may_earnings rest_income - 0.7000) < ε ∧ ε > 0 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_may_earnings_percentage_l104_10460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equality_l104_10478

theorem trigonometric_expression_equality (α : Real) 
  (h1 : α ∈ Set.Icc π (3*π/2))  -- α is in the third quadrant
  (h2 : Real.tan α = 2) :         -- tan α = 2
  (Real.sin (π/2 - α) * Real.cos (π + α)) / Real.sin (3*π/2 + α) = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_equality_l104_10478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l104_10499

-- Define the ellipse E
noncomputable def E (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1

-- Define the left focus F
def F : ℝ × ℝ := (-2, 0)

-- Define a line l passing through F with slope k
def l (k : ℝ) (x y : ℝ) : Prop := x = k * y - 2

-- Define the condition for a point to be on both E and l
def on_E_and_l (k x y : ℝ) : Prop := E x y ∧ l k x y

-- Define the midpoint C of AB
noncomputable def C (k : ℝ) : ℝ × ℝ := (-4/(k^2 + 2), 2*k/(k^2 + 2))

-- Define point D where OC intersects x=-4
def D (k : ℝ) : ℝ × ℝ := (-4, 2*k)

-- Define the condition for ADF to be an isosceles right triangle
def isosceles_right_ADF (k x y : ℝ) : Prop :=
  (x + 2)^2 + y^2 = 4 + 4*k^2

-- Main theorem
theorem ellipse_line_intersection (k : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    on_E_and_l k x₁ y₁ ∧ 
    on_E_and_l k x₂ y₂ ∧ 
    x₁ ≠ x₂ ∧
    isosceles_right_ADF k x₁ y₁) →
  k = 1 ∨ k = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l104_10499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_sequence_exists_l104_10454

theorem no_sequence_exists : ¬ ∃ (a : ℕ → ℕ), 
  (∀ n, a n > 0) ∧ 
  (∀ n, a (n + 2) = a (n + 1) + Int.floor (Real.sqrt (↑(a n) + ↑(a (n + 1))))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_sequence_exists_l104_10454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_is_thirteen_l104_10445

/-- A number is considered valid if it's between 1 and 50, inclusive. -/
def IsValid (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 50

/-- A number is considered prime if it's greater than 1 and has no divisors other than 1 and itself. -/
def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- Alice's number -/
def A : ℕ := sorry

/-- Bob's number -/
def B : ℕ := sorry

/-- Alice can't tell who has the larger number -/
axiom alice_uncertainty : ∃ (x : ℕ), IsValid x ∧ x ≠ A ∧ (x < A ∨ x > A)

/-- Bob knows who has the larger number -/
axiom bob_certainty : ∀ (x : ℕ), IsValid x → x < B → ¬IsPrime x

/-- Bob's number is prime -/
axiom bob_prime : IsPrime B

/-- The sum of Alice's number and 130 times Bob's number is a perfect square -/
axiom perfect_square : ∃ (k : ℕ), A + 130 * B = k^2

theorem sum_is_thirteen : A + B = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_is_thirteen_l104_10445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_knockout_tournament_semifinal_final_l104_10493

structure Tournament :=
  (players : Nat)
  (years : Nat)
  (matches_per_year : Nat)
  (total_matches : Nat)

def knockout_tournament (t : Tournament) : Prop :=
  t.players = 8 ∧
  t.matches_per_year = 7 ∧
  t.total_matches = t.players * (t.players - 1) / 2 ∧
  t.years * t.matches_per_year = t.total_matches

-- Define semifinal_appearances and final_appearances as functions
def semifinal_appearances (t : Tournament) (p : Nat) : Nat := sorry

def final_appearances (t : Tournament) (p : Nat) : Nat := sorry

theorem knockout_tournament_semifinal_final 
  (t : Tournament) 
  (h : knockout_tournament t) : 
  (∀ p, semifinal_appearances t p > 1) ∧
  (∀ p, final_appearances t p ≥ 1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_knockout_tournament_semifinal_final_l104_10493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diagonals_l104_10437

/-- A diagonal in a grid cell -/
structure Diagonal where
  x : Fin 10
  y : Fin 10
  direction : Bool

/-- The distance between two points in a 10x10 grid -/
noncomputable def distance (p1 p2 : Fin 10 × Fin 10) : ℝ :=
  Real.sqrt (((p1.1 - p2.1).val ^ 2 + (p1.2 - p2.2).val ^ 2) : ℝ)

/-- The endpoints of a diagonal -/
def endpoints (d : Diagonal) : Fin 10 × Fin 10 × Fin 10 × Fin 10 :=
  if d.direction then
    (d.x, d.y, d.x + 1, d.y + 1)
  else
    (d.x, d.y + 1, d.x + 1, d.y)

/-- Check if two diagonals satisfy the condition -/
noncomputable def satisfyCondition (d1 d2 : Diagonal) : Prop :=
  let (x1, y1, x2, y2) := endpoints d1
  let (x3, y3, x4, y4) := endpoints d2
  (x2 = x3 ∧ y2 = y3) ∨ (x1 = x4 ∧ y1 = y4) ∨
  (distance (x1, y1) (x3, y3) ≥ 2 ∧ distance (x1, y1) (x4, y4) ≥ 2 ∧
   distance (x2, y2) (x3, y3) ≥ 2 ∧ distance (x2, y2) (x4, y4) ≥ 2)

/-- The main theorem -/
theorem max_diagonals :
  ∃ (diagonals : List Diagonal),
    diagonals.length = 48 ∧
    (∀ d1 d2, d1 ∈ diagonals → d2 ∈ diagonals → d1 ≠ d2 → satisfyCondition d1 d2) ∧
    ∀ (larger_set : List Diagonal),
      (∀ d1 d2, d1 ∈ larger_set → d2 ∈ larger_set → d1 ≠ d2 → satisfyCondition d1 d2) →
      larger_set.length ≤ 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_diagonals_l104_10437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_changes_min_k_when_a_zero_l104_10429

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - (a+2) * x + 2 * a * Real.log x

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := x - (a + 2) + 2 * a / x

theorem monotonicity_changes (a : ℝ) :
  ∃ (x₁ x₂ : ℝ), 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧
  (∀ h : 0 < x₁, HasDerivAt (f a) ((f_derivative a) x₁) x₁ ∧ (f_derivative a) x₁ < 0) ∧
  (∀ h : 0 < x₂, HasDerivAt (f a) ((f_derivative a) x₂) x₂ ∧ (f_derivative a) x₂ > 0) :=
sorry

theorem min_k_when_a_zero :
  ∃ (k : ℤ), k = 1 ∧
  (∀ x : ℝ, x > 0 → (↑k : ℝ) * x + (f_derivative 0 x) * Real.log x > 0) ∧
  (∀ k' : ℤ, k' < k →
    ∃ x : ℝ, x > 0 ∧ (↑k' : ℝ) * x + (f_derivative 0 x) * Real.log x ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_changes_min_k_when_a_zero_l104_10429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l104_10430

noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 1) + 1/3

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  simp [g]
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l104_10430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_happy_cattle_ranch_calf_ratio_l104_10436

/-- The ratio of new calves born each year to the number of cows on Happy Cattle Ranch -/
noncomputable def calf_ratio (initial_cows : ℝ) (final_cows : ℝ) (years : ℝ) : ℝ :=
  (final_cows / initial_cows) ^ (1 / years) - 1

/-- Theorem stating that the calf ratio is 0.5 given the conditions of Happy Cattle Ranch -/
theorem happy_cattle_ranch_calf_ratio :
  calf_ratio 200 450 2 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_happy_cattle_ranch_calf_ratio_l104_10436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_square_perimeter_l104_10458

/-- The set of possible perimeters of the outermost square formed by four kite-shaped shapes --/
def possible_perimeters : Set ℕ := {24, 40, 56, 72}

/-- Theorem stating the possible perimeters of the outermost square --/
theorem kite_square_perimeter (a b : ℕ) :
  a > b →
  a < 10 →
  b < 10 →
  Nat.Coprime a b →
  2 * b = 4 →
  (∃ (p : ℕ), p ∈ possible_perimeters ∧ p = 8 * a) :=
by
  sorry

#check kite_square_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kite_square_perimeter_l104_10458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_tangent_cotangent_function_l104_10408

open Real

theorem min_value_tangent_cotangent_function :
  ∀ x : ℝ, 0 < x → x < π / 2 →
    tan x ^ 2 + 3 * tan x + 6 / tan x + 4 / (tan x ^ 2) - 1 ≥ 3 + 6 * sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_tangent_cotangent_function_l104_10408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l104_10421

/-- A function f is decreasing on ℝ -/
def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The set of real numbers x that satisfies f(x^2-2x) < f(3) -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | f (x^2 - 2*x) < f 3}

theorem solution_set_characterization (f : ℝ → ℝ) (h : IsDecreasing f) :
  SolutionSet f = Set.Iic (-1) ∪ Set.Ioi 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_characterization_l104_10421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unequal_gender_probability_l104_10405

def num_grandchildren : ℕ := 12
def num_unknown_gender : ℕ := 11

noncomputable def probability_unequal_gender_distribution : ℚ :=
  1 - (2 * Nat.choose num_unknown_gender (num_unknown_gender / 2)) / (2 ^ num_unknown_gender : ℚ)

theorem unequal_gender_probability :
  probability_unequal_gender_distribution = 281 / 512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unequal_gender_probability_l104_10405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sides_is_75_l104_10464

/-- Represents a quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ
  -- Conditions
  ab_shortest : ab ≤ bc ∧ ab ≤ cd ∧ ab ≤ da
  right_angle : ab * da = 0  -- Perpendicular sides have dot product 0
  arithmetic_progression : ∃ d : ℝ, bc = ab + d ∧ cd = ab + 2*d ∧ da = ab + 3*d

/-- The sum of possible values for sides other than the shortest in the special quadrilateral -/
def sum_of_possible_sides (q : SpecialQuadrilateral) : ℝ :=
  q.bc + q.cd + q.da

/-- Main theorem stating the sum of possible side lengths -/
theorem sum_of_sides_is_75 (q : SpecialQuadrilateral) (h : q.ab = 15) :
  sum_of_possible_sides q = 75 := by
  sorry

#check sum_of_sides_is_75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sides_is_75_l104_10464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_contains_all_naturals_l104_10472

def mySequence : Nat → Nat
  | 0 => 1
  | 1 => 2
  | (n + 2) => sorry  -- Definition of subsequent terms

theorem sequence_contains_all_naturals :
  ∀ (m : Nat), ∃ (n : Nat), mySequence n = m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_contains_all_naturals_l104_10472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gyroscope_initial_speed_l104_10431

/-- The initial speed of a gyroscope with specific doubling behavior -/
noncomputable def initial_speed (doubling_time : ℝ) (total_time : ℝ) (final_speed : ℝ) : ℝ :=
  final_speed / (2 ^ (total_time / doubling_time))

/-- Theorem: The initial speed of the gyroscope is 6.25 m/s -/
theorem gyroscope_initial_speed :
  initial_speed 15 90 400 = 6.25 := by
  -- Unfold the definition of initial_speed
  unfold initial_speed
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gyroscope_initial_speed_l104_10431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_scalar_l104_10489

/-- Given a line passing through points (3, -1) and (-1, 4) with a direction vector of the form (b, -2), prove that b = 8/5 -/
theorem direction_vector_scalar (b : ℚ) : 
  let p1 : ℚ × ℚ := (3, -1)
  let p2 : ℚ × ℚ := (-1, 4)
  let direction : ℚ × ℚ := (p2.1 - p1.1, p2.2 - p1.2)
  let scaled_direction : ℚ × ℚ := (b, -2)
  (∃ (k : ℚ), (k * direction.1, k * direction.2) = scaled_direction) → b = 8/5 :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_scalar_l104_10489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_derivative_zero_l104_10480

open Function Real

-- Define the property of being an extreme value point
def IsExtremeValuePoint (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f x₀ ≤ f x ∨ f x₀ ≥ f x

-- State the theorem
theorem extreme_value_derivative_zero
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) (x₀ : ℝ) :
  (IsExtremeValuePoint f x₀ → deriv f x₀ = 0) ∧
  ∃ g : ℝ → ℝ, ∃ a : ℝ, Differentiable ℝ g ∧ deriv g a = 0 ∧ ¬IsExtremeValuePoint g a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_derivative_zero_l104_10480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bottles_per_case_l104_10482

/-- Proves the number of bottles in a case given the conditions of Harper's mineral water consumption --/
theorem bottles_per_case
  (daily_consumption : ℚ)
  (duration : ℕ)
  (total_cost : ℕ)
  (cost_per_case : ℕ)
  (h1 : daily_consumption = 1 / 2)
  (h2 : duration = 240)
  (h3 : total_cost = 60)
  (h4 : cost_per_case = 12) :
  (daily_consumption * duration) / (total_cost / cost_per_case : ℚ) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bottles_per_case_l104_10482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l104_10456

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2*a)^x else a/x + 4

-- State the theorem
theorem range_of_a_for_increasing_f (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → a < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l104_10456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_subsequence_exists_l104_10407

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The sequence x_n defined in the problem -/
def x (a b : ℝ) (n : ℕ) : ℕ := sumOfDigits (Int.toNat ⌊a * n + b⌋)

/-- Main theorem -/
theorem constant_subsequence_exists (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (c : ℕ) (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, x a b n = c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_subsequence_exists_l104_10407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C₂_and_distance_AB_l104_10413

-- Define the polar curve C₁
noncomputable def C₁ (θ : Real) : Real := 2 * Real.cos θ

-- Define the relationship between OP and OQ
def OQ_relation (r₁ r₂ : Real) : Prop := r₁ * r₂ = 6

-- Define the rectangular equation of C₂
def C₂ (x : Real) : Prop := x = 3

-- Define the line l
noncomputable def line_l (x : Real) : Real := Real.sqrt 3 * x

-- Define point A as the intersection of C₁ and line l
noncomputable def point_A : Real × Real := (1/2, Real.sqrt 3 / 2)

-- Define point B as the intersection of C₂ and line l
noncomputable def point_B : Real × Real := (3, 3 * Real.sqrt 3)

theorem curve_C₂_and_distance_AB :
  (∀ x, C₂ x ↔ x = 3) ∧
  Real.sqrt ((point_A.1 - point_B.1)^2 + (point_A.2 - point_B.2)^2) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C₂_and_distance_AB_l104_10413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_l104_10470

/-- The perimeter of a hexagon ABCDEF with given coordinates is 12 -/
theorem hexagon_perimeter : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (2, 0)
  let C : ℝ × ℝ := (3, Real.sqrt 3)
  let D : ℝ × ℝ := (2, 2 * Real.sqrt 3)
  let E : ℝ × ℝ := (0, 2 * Real.sqrt 3)
  let F : ℝ × ℝ := (-1, Real.sqrt 3)
  let perimeter := 
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) +
    Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) +
    Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) +
    Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) +
    Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2) +
    Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2)
  perimeter = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_l104_10470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_eq_2_is_local_minimum_l104_10486

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 / x + Real.log x

-- State the theorem
theorem x_eq_2_is_local_minimum :
  ∃ δ > 0, ∀ x, x > 0 → |x - 2| < δ → x ≠ 2 → f x > f 2 :=
by
  sorry

-- Additional lemmas that might be useful for the proof
lemma f_deriv (x : ℝ) (hx : x > 0) : 
  deriv f x = (x - 2) / (x^2) :=
by
  sorry

lemma f_deriv_pos (x : ℝ) (hx : x > 2) : 
  deriv f x > 0 :=
by
  sorry

lemma f_deriv_neg (x : ℝ) (hx1 : x > 0) (hx2 : x < 2) : 
  deriv f x < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_eq_2_is_local_minimum_l104_10486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_train_problem_l104_10452

/-- The distance between two points given the speeds of two trains and their arrival time difference --/
theorem distance_between_points (v1 v2 t : ℝ) : 
  v1 > 0 → v2 > 0 → v1 > v2 → t > 0 → 
  (t * v1 * v2) / (v1 - v2) / v1 + t = (t * v1 * v2) / (v1 - v2) / v2 := by
  sorry

/-- The specific problem instance --/
theorem train_problem : 
  (5 * 29 * 65) / 36 = (5 * 65 * 29) / (65 - 29) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_train_problem_l104_10452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_roots_l104_10420

theorem absolute_difference_of_roots (x : ℝ) :
  |Real.sqrt (x^2 - 4*x + 4) - Real.sqrt (x^2 + 4*x + 4)| = abs (abs (x - 2) - abs (x + 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_roots_l104_10420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_width_curve_diameters_l104_10466

-- Define a curve of constant width
structure ConstantWidthCurve where
  -- Add necessary properties
  width : ℝ
  -- Additional properties can be added as needed

-- Define a diameter of a curve
def Diameter (c : ConstantWidthCurve) : Type := sorry

-- Define a line
def Line : Type := sorry

-- Define a point
def Point : Type := sorry

-- Define the intersection of two lines
def Intersection (l1 l2 : Line) : Point := sorry

-- Define a point being inside or on a curve
def InsideOrOn (p : Point) (c : ConstantWidthCurve) : Prop := sorry

-- Define a corner point of a curve
def CornerPoint (p : Point) (c : ConstantWidthCurve) : Prop := sorry

-- Define the external angle at a point on a curve
noncomputable def ExternalAngle (p : Point) (c : ConstantWidthCurve) : ℝ := sorry

-- Define the angle between two lines
noncomputable def AngleBetween (l1 l2 : Line) : ℝ := sorry

-- Define membership for points in a curve
instance : Membership Point ConstantWidthCurve where
  mem := fun p c => InsideOrOn p c

theorem constant_width_curve_diameters 
  (c : ConstantWidthCurve) (d1 d2 : Diameter c) :
  let p := Intersection (d1 : Line) (d2 : Line)
  InsideOrOn p c ∧
  (p ∈ c → CornerPoint p c) ∧
  (p ∈ c → ExternalAngle p c ≥ AngleBetween (d1 : Line) (d2 : Line)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_width_curve_diameters_l104_10466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l104_10432

noncomputable def f (x : ℝ) := 3 - 2 * Real.cos (2 * x - Real.pi / 3)

theorem monotonic_decreasing_interval (k : ℤ) :
  StrictMonoOn f (Set.Ioo (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6)) :=
by
  sorry

#check monotonic_decreasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l104_10432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_root_theorem_l104_10490

theorem complex_root_theorem (a : ℝ) (b : ℝ) :
  (∃ x : ℝ, x^2 + (4 + I) * x + (4 : ℂ) + a * I = 0) →
  (a + b * I : ℂ) = 2 - 2 * I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_root_theorem_l104_10490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_9n_is_9_l104_10419

/-- A natural number where each digit is strictly greater than the digit to its left -/
def StrictlyIncreasingDigits (n : ℕ) : Prop :=
  ∀ i j, i < j → (n.repr.get! i).toNat < (n.repr.get! j).toNat

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  n.repr.toList.map Char.toNat |>.sum

/-- Theorem: For any natural number with strictly increasing digits,
    the sum of digits of 9 times that number is always 9 -/
theorem sum_of_digits_of_9n_is_9 (n : ℕ) (h : StrictlyIncreasingDigits n) :
  sumOfDigits (9 * n) = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_9n_is_9_l104_10419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_speed_of_trains_l104_10485

/-- Calculates the speed of a train in km/h given its length and time to pass -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Calculates the actual speed of a train relative to the ground -/
noncomputable def actual_speed (relative_speed : ℝ) (observer_speed : ℝ) : ℝ :=
  relative_speed + observer_speed

theorem combined_speed_of_trains (train_a_speed : ℝ) 
  (train_b_length : ℝ) (train_b_time : ℝ)
  (train_c_length : ℝ) (train_c_time : ℝ)
  (train_d_length : ℝ) (train_d_time : ℝ) :
  train_a_speed = 60 →
  train_b_length = 280 →
  train_b_time = 9 →
  train_c_length = 360 →
  train_c_time = 12 →
  train_d_length = 450 →
  train_d_time = 15 →
  abs ((actual_speed (train_speed train_b_length train_b_time) train_a_speed +
   actual_speed (train_speed train_c_length train_c_time) train_a_speed +
   actual_speed (train_speed train_d_length train_d_time) train_a_speed) - 507.996) < 0.001 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_speed_of_trains_l104_10485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_distance_to_line_l104_10404

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

noncomputable def line (y : ℝ) : Prop := y = 2 * Real.sqrt 3

def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

noncomputable def distance_to_line (a b c : ℝ) : ℝ := |c| / Real.sqrt (a^2 + b^2)

theorem constant_distance_to_line :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ →
    line y₂ →
    perpendicular x₁ y₁ x₂ y₂ →
    ∃ (a b c : ℝ),
      a * x₁ + b * y₁ + c = 0 ∧
      a * x₂ + b * y₂ + c = 0 ∧
      distance_to_line a b c = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_distance_to_line_l104_10404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_B_time_l104_10453

/-- Represents the time it takes for pipe B to fill the tank alone -/
noncomputable def time_for_pipe_B : ℝ := 35

/-- Represents the time it takes for all pipes together to fill the tank -/
noncomputable def time_for_all_pipes : ℝ := 10

/-- Represents the rate at which pipe C fills the tank -/
noncomputable def rate_C : ℝ := 1 / 70

/-- Represents the rate at which pipe B fills the tank -/
noncomputable def rate_B : ℝ := 2 * rate_C

/-- Represents the rate at which pipe A fills the tank -/
noncomputable def rate_A : ℝ := 2 * rate_B

/-- Theorem stating that pipe B alone takes 35 hours to fill the tank -/
theorem pipe_B_time : 
  (rate_A + rate_B + rate_C) * time_for_all_pipes = 1 ∧ 
  rate_A = 2 * rate_B ∧ 
  rate_B = 2 * rate_C → 
  time_for_pipe_B = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_B_time_l104_10453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_implies_product_l104_10495

theorem divisibility_implies_product (f b : Nat) : 
  f < 10 → b < 10 → 
  (52000 + f * 1000 + 300 + b) % 7 = 0 → 
  (52000 + f * 1000 + 300 + b) % 13 = 0 → 
  (52000 + f * 1000 + 300 + b) % 89 = 0 → 
  f * 2 * 3 * b = 108 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_implies_product_l104_10495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_non_equivalent_schemes_l104_10402

/-- Represents a 7x7 board with two yellow squares --/
def Board := Fin 49 × Fin 49

/-- Two boards are equivalent if one can be obtained from the other by rotation --/
def equivalent (b1 b2 : Board) : Prop := sorry

/-- The set of all possible boards --/
def allBoards : Finset Board := sorry

/-- The set of non-equivalent boards --/
def nonEquivalentBoards : Finset Board := sorry

/-- The number of non-equivalent color schemes --/
def numNonEquivalentSchemes : ℕ := Finset.card nonEquivalentBoards

theorem num_non_equivalent_schemes :
  numNonEquivalentSchemes = 300 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_non_equivalent_schemes_l104_10402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_integers_satisfying_inequality_l104_10418

theorem count_positive_integers_satisfying_inequality : 
  (Finset.filter (fun n : ℕ => n > 0 ∧ (n + 9) * (n - 4) * (n - 13) < 0) (Finset.range 14)).card = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_positive_integers_satisfying_inequality_l104_10418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_for_increasing_sequence_l104_10475

/-- An increasing sequence of real numbers indexed by positive natural numbers. -/
def IncreasingSequence (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n < a (n + 1)

/-- The theorem stating the range of λ for an increasing sequence satisfying a_n = n^2 + λn. -/
theorem lambda_range_for_increasing_sequence (a : ℕ+ → ℝ) (l : ℝ) :
  IncreasingSequence a ∧ (∀ n : ℕ+, a n = n.val^2 + l * n.val) →
  ∀ x : ℝ, x > -3 ↔ ∃ l' : ℝ, l' = x ∧
    IncreasingSequence (fun n ↦ n.val^2 + l' * n.val) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_for_increasing_sequence_l104_10475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_omega_two_thirds_achievable_l104_10457

open Real

/-- The function f(x) = sin(ωx + φ) with given properties -/
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := sin (ω * x + φ)

/-- The theorem stating the minimum value of ω -/
theorem min_omega_value (ω φ : ℝ) (h_ω_pos : ω > 0) 
  (h_symmetry : ∃ k : ℤ, ω * (π / 2) + φ = k * π)
  (h_quarter_pi : f ω φ (π / 4) = 1 / 2) :
  2 / 3 ≤ ω := by
  sorry

/-- The theorem proving that 2/3 is achievable -/
theorem omega_two_thirds_achievable :
  ∃ ω φ : ℝ, ω = 2 / 3 ∧ ω > 0 ∧
  (∃ k : ℤ, ω * (π / 2) + φ = k * π) ∧
  f ω φ (π / 4) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_omega_two_thirds_achievable_l104_10457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twice_as_many_european_carry_double_total_weight_eq_max_flock_size_l104_10426

/-- Represents the number of European swallows in the flock -/
def european_swallows : ℕ := 30

/-- Represents the number of American swallows in the flock -/
def american_swallows : ℕ := 60

/-- The weight an American swallow can carry -/
def american_carry_weight : ℕ := 5

/-- The maximum combined weight the flock can carry -/
def max_combined_weight : ℕ := 600

/-- There are twice as many American swallows as European swallows -/
theorem twice_as_many : american_swallows = 2 * european_swallows := by
  rfl

/-- A European swallow can carry twice the weight of an American swallow -/
theorem european_carry_double : 
  american_carry_weight * 2 = american_carry_weight + american_carry_weight := by
  rfl

/-- The total weight the flock can carry equals the maximum combined weight -/
theorem total_weight_eq_max : 
  american_swallows * american_carry_weight + 
  european_swallows * (american_carry_weight * 2) = max_combined_weight := by
  rfl

/-- The total number of swallows in the flock -/
def total_swallows : ℕ := european_swallows + american_swallows

/-- Theorem: The total number of swallows in the flock is 90 -/
theorem flock_size : total_swallows = 90 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twice_as_many_european_carry_double_total_weight_eq_max_flock_size_l104_10426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_implicit_partial_derivatives_l104_10450

-- Define the implicit function
noncomputable def F (x y z : ℝ) : ℝ := Real.exp (z^2) - x^2 * y^2 * z^2

-- Define the partial derivatives of F with respect to x, y, and z
def F_x (x y z : ℝ) : ℝ := -2 * x * y^2 * z^2
def F_y (x y z : ℝ) : ℝ := -2 * x^2 * y * z^2
noncomputable def F_z (x y z : ℝ) : ℝ := 2 * z * Real.exp (z^2) - 2 * x^2 * y^2 * z

-- State the theorem
theorem implicit_partial_derivatives
  (x y z : ℝ) (hz : F x y z = 0) (hz_nonzero : z ≠ 0) (hx_nonzero : x ≠ 0) (hy_nonzero : y ≠ 0) :
  (- F_x x y z / F_z x y z = z / (x * (z^2 - 1))) ∧
  (- F_y x y z / F_z x y z = z / (y * (z^2 - 1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_implicit_partial_derivatives_l104_10450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_size_increase_proof_l104_10438

/-- Represents the increase in length per unit size of a shoe -/
def length_increase : ℝ := 0.2

/-- The length of a shoe of a given size -/
def shoe_length (size : ℕ) : ℝ :=
  5.9 + length_increase * (size - 15 : ℝ)

theorem shoe_size_increase_proof :
  (∀ size, 8 ≤ size → size ≤ 17 → Int.floor (shoe_length size) = size) ∧
  (shoe_length 17 = shoe_length 8 * 1.4) ∧
  (shoe_length 15 = 5.9) →
  length_increase = 0.2 := by
  sorry

#check shoe_size_increase_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_size_increase_proof_l104_10438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_monotone_increasing_interval_l104_10422

theorem sin_monotone_increasing_interval (x : ℝ) :
  ∃ k : ℤ, (∀ x₁ x₂, x₁ ∈ Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12) →
    x₂ ∈ Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12) →
    x₁ ≤ x₂ → Real.sin (2 * x₁ - Real.pi / 3) ≤ Real.sin (2 * x₂ - Real.pi / 3)) ∧
  (∀ x₀, x₀ ∉ ⋃ (k : ℤ), Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12) →
    ∃ ε > 0, ∃ x₁ x₂, x₁ ∈ Set.Ioo (x₀ - ε) (x₀ + ε) ∧
      x₂ ∈ Set.Ioo (x₀ - ε) (x₀ + ε) ∧
      x₁ < x₂ ∧ Real.sin (2 * x₁ - Real.pi / 3) > Real.sin (2 * x₂ - Real.pi / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_monotone_increasing_interval_l104_10422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rainfall_rate_proof_l104_10477

/-- Calculates the rate of rainfall given puddle dimensions and time -/
noncomputable def rainfall_rate (base_area : ℝ) (depth : ℝ) (time : ℝ) : ℝ :=
  depth / time

/-- Proves that the rainfall rate is 10 cm/hour given the problem conditions -/
theorem rainfall_rate_proof (base_area depth time : ℝ) 
  (h1 : base_area = 300) 
  (h2 : depth = 30)
  (h3 : time = 3) :
  rainfall_rate base_area depth time = 10 := by
  unfold rainfall_rate
  rw [h2, h3]
  norm_num

#check rainfall_rate_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rainfall_rate_proof_l104_10477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_area_of_specific_pyramid_l104_10412

/-- A rectangular pyramid with given dimensions -/
structure RectangularPyramid where
  base_length : ℝ
  base_width : ℝ
  height : ℝ
  perpendicular_edge : Bool

/-- The surface area of the circumscribed sphere of a rectangular pyramid -/
noncomputable def circumscribed_sphere_area (p : RectangularPyramid) : ℝ :=
  4 * Real.pi * ((p.base_length ^ 2 + p.base_width ^ 2 + p.height ^ 2) / 4)

/-- The theorem stating the surface area of the circumscribed sphere for the given pyramid -/
theorem circumscribed_sphere_area_of_specific_pyramid :
  let p : RectangularPyramid := {
    base_length := 7,
    base_width := 5,
    height := 8,
    perpendicular_edge := true
  }
  circumscribed_sphere_area p = 138 * Real.pi := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_area_of_specific_pyramid_l104_10412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_imply_m_and_n_l104_10451

/-- Two monomials are like terms if their variables have the same exponents -/
def are_like_terms (term1 term2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ (x y : ℕ), term1 x y ≠ 0 ∧ term2 x y ≠ 0 → 
    ∃ (c1 c2 : ℚ), term1 x y = c1 * x^(Int.natAbs (Rat.num (term1 x y))) * y^(Int.natAbs (Rat.num (term1 x y))) ∧
                   term2 x y = c2 * x^(Int.natAbs (Rat.num (term2 x y))) * y^(Int.natAbs (Rat.num (term2 x y))) ∧
                   Int.natAbs (Rat.num (term1 x y)) = Int.natAbs (Rat.num (term2 x y))

/-- The first term in the problem -/
def term1 (n : ℕ) (x y : ℕ) : ℚ := 2 * (x^(n+2) * y^3 : ℚ)

/-- The second term in the problem -/
def term2 (m : ℕ) (x y : ℕ) : ℚ := -3 * (x^3 * y^(2*m-1) : ℚ)

/-- The main theorem -/
theorem like_terms_imply_m_and_n (m n : ℕ) : 
  are_like_terms (term1 n) (term2 m) → m = 2 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_like_terms_imply_m_and_n_l104_10451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l104_10455

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 - x) / x + Real.log x

-- Define the theorem
theorem max_a_value (a : ℝ) :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, f x ≥ a) →
  a ≤ 0 :=
by
  -- Introduce the hypothesis
  intro h
  -- The proof goes here, but we'll use sorry for now
  sorry

#check max_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l104_10455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_possible_sums_l104_10473

/-- The set of integers from 1 to 120 -/
def fullSet : Finset ℕ := Finset.range 120

/-- An 80-element subset of the full set -/
def subsetC : Set (Finset ℕ) := {C : Finset ℕ | C ⊆ fullSet ∧ C.card = 80}

/-- The sum of elements in a subset -/
def sumOfSet (C : Finset ℕ) : ℕ := C.sum id

/-- The set of all possible sums for 80-element subsets -/
def possibleSums : Set ℕ := {S : ℕ | ∃ C ∈ subsetC, S = sumOfSet C}

/-- The main theorem: there are 3201 possible sum values -/
theorem count_possible_sums : Finset.card (Finset.range 3201) = 3201 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_possible_sums_l104_10473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_lines_bound_l104_10487

/-- Represents a convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_convex : Prop := sorry
  sides_non_parallel : Prop := sorry

/-- A point inside the polygon -/
noncomputable def interior_point (p : ConvexPolygon n) : ℝ × ℝ := sorry

/-- A line that bisects the area of the polygon -/
def bisecting_line (p : ConvexPolygon n) (o : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

/-- The set of all bisecting lines through a given point -/
def bisecting_lines (p : ConvexPolygon n) (o : ℝ × ℝ) : Set (Set (ℝ × ℝ)) := sorry

/-- The theorem stating that the number of bisecting lines is at most n -/
theorem bisecting_lines_bound (n : ℕ) (p : ConvexPolygon n) :
  let o := interior_point p
  ∃ (s : Finset (Set (ℝ × ℝ))), s.card ≤ n ∧ ∀ l ∈ bisecting_lines p o, l ∈ s := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisecting_lines_bound_l104_10487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l104_10471

theorem min_value_expression (a b : ℤ) 
  (ha : 0 < a ∧ a < 11) (hb : 0 < b ∧ b < 11) : 
  (∀ x y : ℤ, 0 < x ∧ x < 11 → 0 < y ∧ y < 11 → 2 * x - x * y ≥ 2 * a - a * b) → 
  2 * a - a * b = -80 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l104_10471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_solution_l104_10488

theorem unique_function_solution (f : ℕ → ℕ) : 
  (∀ n : ℕ, f (f n) < f (n + 1)) → (∀ n : ℕ, f n = n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_solution_l104_10488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_of_complex_fraction_l104_10459

theorem subset_count_of_complex_fraction : 
  ∃ (x y : ℝ), (2 - Complex.I) / (1 + 2*Complex.I) = x + y*Complex.I ∧ 
  Finset.card (Finset.powerset {x, Real.exp (x * Real.log 2), y}) = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_count_of_complex_fraction_l104_10459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lunch_break_is_one_hour_l104_10424

/-- Represents the work rate of a person or group in task percentage per hour -/
structure WorkRate where
  value : ℝ

/-- Represents the length of time in hours -/
structure Time where
  value : ℝ

/-- The work completed by a group in a given time period -/
def work_completed (rate : WorkRate) (time : Time) : ℝ := rate.value * time.value

/-- Sarah's work rate -/
noncomputable def sarah_rate : WorkRate := ⟨1⟩

/-- Combined work rate of the two assistants -/
noncomputable def assistants_rate : WorkRate := ⟨1⟩

/-- Length of the lunch break in hours -/
noncomputable def lunch_break : Time := ⟨1⟩

theorem lunch_break_is_one_hour :
  ∃ (sarah_rate assistants_rate : WorkRate) (lunch_break : Time),
    -- Monday's work
    work_completed ⟨sarah_rate.value + assistants_rate.value⟩ ⟨8 - lunch_break.value⟩ = 0.6 ∧
    -- Tuesday's work
    work_completed assistants_rate ⟨6.4 - lunch_break.value⟩ = 0.28 ∧
    -- Wednesday's work
    work_completed sarah_rate ⟨9.6 - lunch_break.value⟩ = 0.12 ∧
    -- Lunch break is 1 hour (60 minutes)
    lunch_break.value = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lunch_break_is_one_hour_l104_10424
