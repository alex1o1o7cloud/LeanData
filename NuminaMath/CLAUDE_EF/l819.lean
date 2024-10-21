import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_zero_iff_f_even_l819_81953

-- Define the function f(x) = ln|x-a|
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (abs (x - a))

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Theorem statement
theorem a_zero_iff_f_even (a : ℝ) :
  a = 0 ↔ is_even (f a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_zero_iff_f_even_l819_81953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_by_fraction_twelve_divided_by_one_fourth_l819_81972

theorem division_by_fraction (a b : ℚ) (hb : b ≠ 0) : 
  a / (1 / b) = a * b := by sorry

theorem twelve_divided_by_one_fourth : 
  (12 : ℚ) / (1 / 4) = 48 := by
  have h : (4 : ℚ) ≠ 0 := by norm_num
  rw [division_by_fraction 12 4 h]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_by_fraction_twelve_divided_by_one_fourth_l819_81972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_nine_factorial_l819_81979

theorem divisors_of_nine_factorial (n : ℕ) : n = 9 →
  (Finset.filter (λ d ↦ d ∣ n.factorial ∧ d > (n-1).factorial) (Finset.range (n.factorial + 1))).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_nine_factorial_l819_81979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_ordering_l819_81976

noncomputable section

-- Define the function
def f (x : ℝ) : ℝ := 4 / x

-- Define the points
variable (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ)

-- Define the conditions
axiom points_on_graph : (y₁ = f x₁) ∧ (y₂ = f x₂) ∧ (y₃ = f x₃)
axiom x_order : x₁ < x₂ ∧ x₂ < 0
axiom x₃_positive : x₃ > 0

-- State the theorem
theorem inverse_proportion_ordering : y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_ordering_l819_81976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_l819_81955

/-- Definition of circle w₁ -/
def w₁ (x y : ℝ) : Prop := x^2 + y^2 + 12*x + 10*y - 50 = 0

/-- Definition of circle w₂ -/
def w₂ (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 10*y + 60 = 0

/-- Definition of a line with slope m passing through (x, y) -/
def line_with_slope (m x y : ℝ) : Prop := y = m * x

/-- Definition of external tangency -/
def externally_tangent (x₁ y₁ r₁ x₂ y₂ r₂ : ℝ) : Prop :=
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = (r₁ + r₂)^2

/-- Definition of internal tangency -/
def internally_tangent (x₁ y₁ r₁ x₂ y₂ r₂ : ℝ) : Prop :=
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = (r₁ - r₂)^2

/-- The main theorem -/
theorem circle_tangency_theorem :
  ∃ (m : ℝ) (p q : ℕ), 
    (∀ m' > 0, m ≤ m') →
    (m^2 = p / q) →
    (Nat.Coprime p q) →
    (∃ (x y r : ℝ),
      line_with_slope m x y ∧
      (∃ (x₁ y₁ r₁ : ℝ), w₁ x₁ y₁ ∧ internally_tangent x y r x₁ y₁ r₁) ∧
      (∃ (x₂ y₂ r₂ : ℝ), w₂ x₂ y₂ ∧ externally_tangent x y r x₂ y₂ r₂)) →
    p + q = 177 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_theorem_l819_81955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_account_balance_l819_81977

/-- Calculates the final balance in a savings account after two years with specific conditions --/
theorem savings_account_balance 
  (initial_deposit : ℝ) 
  (first_year_rate : ℝ) 
  (second_year_rate : ℝ) 
  (withdrawal_fraction : ℝ) : 
  initial_deposit = 1000 →
  first_year_rate = 0.20 →
  second_year_rate = 0.15 →
  withdrawal_fraction = 0.5 →
  (initial_deposit * (1 + first_year_rate) * (1 - withdrawal_fraction) * (1 + second_year_rate)) = 690 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_account_balance_l819_81977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_min_first_level_energy_l819_81928

/-- Represents the energy at each trophic level in the food chain --/
def TrophicEnergy : Nat → ℝ := sorry

/-- The number of trophic levels in the food chain --/
def numLevels : Nat := 6

/-- The minimum energy flow efficiency between trophic levels --/
def minEfficiency : ℝ := 0.1

/-- The energy received by the last trophic level --/
def lastLevelEnergy : ℝ := 10

/-- 
Theorem stating that if the last trophic level receives 10 KJ of energy,
and the energy flow efficiency between levels is at least 10%,
then the first trophic level must provide at least 10^6 KJ of energy.
--/
theorem min_first_level_energy : 
  (∀ n : Nat, n < numLevels - 1 → TrophicEnergy (n + 1) ≥ minEfficiency * TrophicEnergy n) →
  TrophicEnergy (numLevels - 1) = lastLevelEnergy →
  TrophicEnergy 0 ≥ 10^6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_min_first_level_energy_l819_81928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_8_neg8sqrt2_l819_81985

noncomputable def rectangular_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 && y ≥ 0 then Real.arctan (y / x) + Real.pi
           else if x < 0 && y < 0 then Real.arctan (y / x) - Real.pi
           else if x = 0 && y > 0 then Real.pi / 2
           else if x = 0 && y < 0 then -Real.pi / 2
           else 0  -- x = 0 and y = 0
  (r, if θ < 0 then θ + 2*Real.pi else θ)

theorem rectangular_to_polar_8_neg8sqrt2 :
  let (r, θ) := rectangular_to_polar 8 (-8*Real.sqrt 2)
  r = 8*Real.sqrt 3 ∧ θ = 7*Real.pi/4 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2*Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_8_neg8sqrt2_l819_81985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_length_l819_81954

/-- Given an isosceles triangle with two sides of length 10 units and the angle between them
    measuring 80°, the length of the third side (base) is approximately 12.862 units. -/
theorem isosceles_triangle_base_length :
  ∀ (a b c : ℝ) (A B C : ℝ),
  a = 10 ∧ b = 10 ∧ C = 80 * π / 180 →
  A = B →
  A + B + C = π →
  abs (c - 12.862) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_base_length_l819_81954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circumcenter_relation_l819_81952

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the orthocenter, circumcenter, and circumradius
noncomputable def orthocenter (t : Triangle) : ℝ × ℝ := sorry
noncomputable def circumcenter (t : Triangle) : ℝ × ℝ := sorry
noncomputable def circumradius (t : Triangle) : ℝ := sorry

-- Define the angles of the triangle
noncomputable def angle_A (t : Triangle) : ℝ := sorry
noncomputable def angle_B (t : Triangle) : ℝ := sorry
noncomputable def angle_C (t : Triangle) : ℝ := sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Define what it means for a triangle to be equilateral
def is_equilateral (t : Triangle) : Prop := sorry

-- Theorem statement
theorem orthocenter_circumcenter_relation (t : Triangle) :
  let O := circumcenter t
  let H := orthocenter t
  let R := circumradius t
  let α := angle_A t
  let β := angle_B t
  let γ := angle_C t
  (distance O H = R * Real.sqrt (1 - 8 * Real.cos α * Real.cos β * Real.cos γ)) ∧
  (O = H ↔ is_equilateral t) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_circumcenter_relation_l819_81952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_in_ellipse_ratio_l819_81973

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the equilateral triangle ABC
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_equilateral : A.1 = B.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ B.2 = C.2

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Theorem statement
theorem triangle_in_ellipse_ratio 
  (ABC : EquilateralTriangle)
  (h_B : ABC.B = (0, 2))
  (h_AC_parallel : ABC.A.2 = ABC.C.2)
  (h_A_in_ellipse : ellipse ABC.A.1 ABC.A.2)
  (h_B_in_ellipse : ellipse ABC.B.1 ABC.B.2)
  (h_C_in_ellipse : ellipse ABC.C.1 ABC.C.2)
  (h_F₁_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F₁ = (1 - t) • ABC.B + t • ABC.C)
  (h_F₂_on_AB : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ F₂ = (1 - s) • ABC.A + s • ABC.B)
  : ‖ABC.A - ABC.B‖ / ‖F₁ - F₂‖ = 3 / (2 * Real.sqrt 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_in_ellipse_ratio_l819_81973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l819_81964

-- Define the parameter a
variable (a : ℝ)

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | -2 ≤ x ∧ x ≤ a}
def B (a : ℝ) : Set ℝ := {y | ∃ x ∈ A a, y = 2*x + 3}
def C (a : ℝ) : Set ℝ := {t | ∃ x ∈ A a, t = x^2}

-- State the theorem
theorem a_range (h1 : a ≥ 2) (h2 : C a ⊆ B a) : 2 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l819_81964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l819_81910

noncomputable def arithmeticSequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

noncomputable def arithmeticSum (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum (a₁ d : ℝ) :
  3 * (arithmeticSequence a₁ d 2 + arithmeticSequence a₁ d 6) +
  2 * (arithmeticSequence a₁ d 5 + arithmeticSequence a₁ d 10 + arithmeticSequence a₁ d 15) = 24 →
  arithmeticSum a₁ d 13 = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l819_81910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_negative_two_l819_81901

noncomputable def ζ : ℂ := Complex.exp (2 * Real.pi * Complex.I / 89)

noncomputable def sum_function (k : ℕ) : ℂ :=
  Complex.sin (2^(k + 4) * Real.pi / 89) / Complex.sin (2^k * Real.pi / 89)

theorem sum_equals_negative_two :
  (Finset.sum (Finset.range 11) (fun k => sum_function (k + 1))) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_negative_two_l819_81901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_l819_81966

/-- Proves that the speed of a boat in still water is 20 km/hr given the specified conditions -/
theorem boat_speed (current_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) 
  (h1 : current_speed = 5)
  (h2 : downstream_distance = 10)
  (h3 : downstream_time = 24 / 60) : 
  (downstream_distance / downstream_time) - current_speed = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_l819_81966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_sum_greater_than_two_l819_81935

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m * x - 1) / x - Real.log x

-- State the theorem
theorem two_zeros_sum_greater_than_two (m : ℝ) (x₁ x₂ : ℝ) :
  0 < x₁ → x₁ < x₂ →
  f m x₁ = 0 → f m x₂ = 0 →
  (∀ x, 0 < x → f m x = 0 → x = x₁ ∨ x = x₂) →
  x₁ + x₂ > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_sum_greater_than_two_l819_81935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_18_l819_81960

/-- Represents the swimming scenario with given conditions -/
structure SwimmingScenario where
  downstream_distance : ℝ
  downstream_time : ℝ
  upstream_time : ℝ
  still_water_speed : ℝ

/-- Calculates the upstream distance given a swimming scenario -/
noncomputable def upstream_distance (s : SwimmingScenario) : ℝ :=
  let downstream_speed := s.downstream_distance / s.downstream_time
  let current_speed := downstream_speed - s.still_water_speed
  let upstream_speed := s.still_water_speed - current_speed
  upstream_speed * s.upstream_time

/-- Theorem stating that under the given conditions, the upstream distance is 18 km -/
theorem upstream_distance_is_18 :
  let scenario : SwimmingScenario := {
    downstream_distance := 36,
    downstream_time := 6,
    upstream_time := 6,
    still_water_speed := 4.5
  }
  upstream_distance scenario = 18 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_distance_is_18_l819_81960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_vectors_l819_81993

theorem cosine_of_angle_between_vectors :
  let a : Fin 3 → ℝ := ![1, 1, 2]
  let b : Fin 3 → ℝ := ![2, -1, 2]
  let dot_product := (a 0) * (b 0) + (a 1) * (b 1) + (a 2) * (b 2)
  let magnitude_a := Real.sqrt ((a 0)^2 + (a 1)^2 + (a 2)^2)
  let magnitude_b := Real.sqrt ((b 0)^2 + (b 1)^2 + (b 2)^2)
  dot_product / (magnitude_a * magnitude_b) = 5 * Real.sqrt 6 / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_vectors_l819_81993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chameleons_cannot_all_be_white_l819_81912

/-- Represents the number of chameleons of each color -/
structure ChameleonCount where
  blue : ℕ
  white : ℕ
  red : ℕ

/-- The initial count of chameleons -/
def initialCount : ChameleonCount :=
  { blue := 12, white := 25, red := 8 }

/-- The rule for chameleon color changes -/
def colorChangeRule (c : ChameleonCount) : ChameleonCount :=
  { blue := c.blue, white := c.white, red := c.red }

/-- Theorem stating it's impossible for all chameleons to become white -/
theorem chameleons_cannot_all_be_white (c : ChameleonCount) :
  c = initialCount → ¬∃ n : ℕ, (Nat.iterate colorChangeRule n c).white = c.blue + c.white + c.red := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chameleons_cannot_all_be_white_l819_81912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_randy_blocks_l819_81945

theorem randy_blocks 
  (initial : ℕ)
  (used : ℕ)
  (given_away : ℕ)
  (bought : ℕ)
  (sister_sets : ℕ)
  (blocks_per_set : ℕ)
  (h1 : initial = 78)
  (h2 : used = 19)
  (h3 : given_away = 25)
  (h4 : bought = 36)
  (h5 : sister_sets = 3)
  (h6 : blocks_per_set = 12) :
  (initial - used - given_away + bought + sister_sets * blocks_per_set) / 2 = 53 :=
by
  sorry

#check randy_blocks

end NUMINAMATH_CALUDE_ERRORFEEDBACK_randy_blocks_l819_81945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_power_mod_five_l819_81927

theorem sum_power_mod_five (n : ℕ) :
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_power_mod_five_l819_81927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_imply_a_range_l819_81916

theorem equation_roots_imply_a_range (a : ℝ) :
  (∃ (x₁ x₂ x₃ x₄ : ℝ), 
    (∀ i : Fin 4, (Fin.val i + 1 : ℝ) > 0) ∧
    (∀ i j : Fin 4, i ≠ j → (Fin.val i + 1 : ℝ) ≠ (Fin.val j + 1 : ℝ)) ∧
    (∀ i : Fin 4, let x := (Fin.val i + 1 : ℝ); x^3 - 2*a*x^2 + (a^2 + 2)*x = 4*a - 4/x)) →
  a > 3 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_imply_a_range_l819_81916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_m_values_for_given_chord_length_l819_81921

/-- The line equation -/
def line_equation (m x y : ℝ) : Prop :=
  (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 25

/-- The fixed point that the line always passes through -/
def fixed_point : ℝ × ℝ := (3, 1)

/-- The chord length -/
noncomputable def chord_length : ℝ := 4 * Real.sqrt 6

/-- Theorem: The line always passes through the fixed point -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_equation m (fixed_point.1) (fixed_point.2) := by
  sorry

/-- Theorem: If the chord length is 4√6, then m = -1/2 or m = 1/2 -/
theorem m_values_for_given_chord_length :
  ∀ m : ℝ, (∃ x y : ℝ, line_equation m x y ∧ circle_equation x y ∧
    ∃ x' y' : ℝ, line_equation m x' y' ∧ circle_equation x' y' ∧
    ((x - x')^2 + (y - y')^2 = chord_length^2)) →
  (m = -1/2 ∨ m = 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_m_values_for_given_chord_length_l819_81921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_letter_words_with_vowels_l819_81923

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def word_length : Nat := 5

def total_words : Nat := (Finset.card letters) ^ word_length
def words_without_vowels : Nat := (Finset.card (letters \ vowels)) ^ word_length

theorem five_letter_words_with_vowels :
  total_words - words_without_vowels = 6752 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_letter_words_with_vowels_l819_81923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_solution_liquid_x_percentage_l819_81998

/-- Represents the composition of a solution --/
structure Solution where
  total : ℚ
  liquid_x : ℚ
  water : ℚ

/-- Calculates the percentage of liquid X in a solution --/
def percentage_liquid_x (s : Solution) : ℚ :=
  s.liquid_x / s.total * 100

/-- The initial solution Y --/
def initial_solution : Solution :=
  { total := 6
  , liquid_x := 6 * 3 / 10
  , water := 6 * 7 / 10 }

/-- The solution after water evaporation --/
def evaporated_solution : Solution :=
  { total := initial_solution.total - 2
  , liquid_x := initial_solution.liquid_x
  , water := initial_solution.water - 2 }

/-- The additional solution Y to be added --/
def additional_solution : Solution :=
  { total := 2
  , liquid_x := 2 * 3 / 10
  , water := 2 * 7 / 10 }

/-- The final solution after adding the additional solution --/
def final_solution : Solution :=
  { total := evaporated_solution.total + additional_solution.total
  , liquid_x := evaporated_solution.liquid_x + additional_solution.liquid_x
  , water := evaporated_solution.water + additional_solution.water }

theorem final_solution_liquid_x_percentage :
  percentage_liquid_x final_solution = 40 := by
  sorry

#eval percentage_liquid_x final_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_solution_liquid_x_percentage_l819_81998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_theorem_l819_81934

/-- A parabola with vertex at the origin and focus on the y-axis -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- A point on the parabola -/
structure ParabolaPoint (para : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : x^2 = -2 * para.p * y

/-- The focus of the parabola -/
def focus (para : Parabola) : ℝ × ℝ := (0, para.p)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The main theorem -/
theorem parabola_point_theorem (para : Parabola) (k : ℝ) :
  let P : ParabolaPoint para := ⟨k, -2, by sorry⟩
  distance (k, -2) (focus para) = 4 → k = 4 ∨ k = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_theorem_l819_81934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jills_remaining_amount_l819_81904

noncomputable def net_salary : ℝ := 3500

noncomputable def discretionary_income (salary : ℝ) : ℝ := salary / 5

noncomputable def vacation_fund (income : ℝ) : ℝ := 0.30 * income

noncomputable def savings (income : ℝ) : ℝ := 0.20 * income

noncomputable def eating_out (income : ℝ) : ℝ := 0.35 * income

noncomputable def remaining_amount (income : ℝ) : ℝ :=
  income - (vacation_fund income + savings income + eating_out income)

theorem jills_remaining_amount :
  remaining_amount (discretionary_income net_salary) = 105 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jills_remaining_amount_l819_81904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l819_81986

/-- The distance between two points in 3D space -/
noncomputable def distance3D (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2)

/-- Theorem: The distance between points A(1, 1, 1) and B(-3, -3, -3) is 4√3 -/
theorem distance_A_to_B : distance3D 1 1 1 (-3) (-3) (-3) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l819_81986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l819_81981

/-- Curve C in polar coordinates -/
noncomputable def curve_C (a : ℝ) (θ : ℝ) : ℝ := 2 * a * Real.sin θ

/-- Line l in parametric form -/
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (-Real.sqrt 2 / 2 * t - 1, Real.sqrt 2 / 2 * t)

/-- Condition for common points between curve C and line l -/
def has_common_points (a : ℝ) : Prop :=
  ∃ t θ, curve_C a θ * Real.cos θ = (line_l t).1 ∧
         curve_C a θ * Real.sin θ = (line_l t).2

/-- The range of values for a -/
def a_range (a : ℝ) : Prop :=
  a ≤ (1 - 4 * Real.sqrt 2) / 7 ∨ a ≥ (1 + 4 * Real.sqrt 2) / 7

/-- Main theorem -/
theorem curve_line_intersection (a : ℝ) (ha : a ≠ 0) :
  has_common_points a → a_range a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l819_81981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_circles_shortest_distance_between_circles_proof_l819_81996

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles (x y : ℝ) : ℝ :=
  let circle1 := x^2 - 6*x + y^2 - 8*y = 1
  let circle2 := x^2 + 10*x + y^2 - 6*y = 25
  0

/-- Proof of the theorem -/
theorem shortest_distance_between_circles_proof (x y : ℝ) : 
  shortest_distance_between_circles x y = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_between_circles_shortest_distance_between_circles_proof_l819_81996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_properties_l819_81913

open Real

-- Define the original line l
def line_l (x y : ℝ) : Prop := x + y - 1 = 0

-- Define a point
def point : ℝ × ℝ := (1, 1)

-- Define the slope of a line given its equation
noncomputable def slope_of_line (f : ℝ → ℝ → Prop) : ℝ := -1

-- Define the angle of inclination given a slope
noncomputable def angle_of_inclination (m : ℝ) : ℝ :=
  Real.pi - Real.arctan (-m)

-- Define the equation of a line given a point and slope
def line_equation (p : ℝ × ℝ) (m : ℝ) (x y : ℝ) : Prop :=
  y - p.2 = m * (x - p.1)

-- State the theorem
theorem parallel_line_properties :
  let m := slope_of_line line_l
  let α := angle_of_inclination m
  let parallel_line := line_equation point m
  (α = Real.pi * 3 / 4) ∧ 
  (∀ x y, parallel_line x y ↔ x + y - 2 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_line_properties_l819_81913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_over_two_hundred_l819_81937

noncomputable def series_term (n : ℕ) : ℝ := (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

noncomputable def series_sum : ℝ := ∑' n, series_term n

theorem series_sum_equals_one_over_two_hundred : series_sum = 1 / 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_over_two_hundred_l819_81937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_never_stops_l819_81997

/-- Represents the transformation rule described in the problem -/
def transform (n : ℕ) : ℕ :=
  let a := n / 100
  let b := n % 100
  2 * a + 8 * b

/-- The initial number with 900 digits all being 1 -/
def initial_number : ℕ := 10^900 - 1

/-- Theorem stating that the transformation process will never stop -/
theorem transformation_never_stops :
  ∀ k : ℕ, Nat.iterate transform k initial_number ≥ 100 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_never_stops_l819_81997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_of_product_l819_81943

theorem unit_digit_of_product (a b : ℕ) : 
  (a : ℝ) - (b : ℝ) * Real.sqrt 3 = (2 - Real.sqrt 3) ^ 100 →
  (a * b) % 10 = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_of_product_l819_81943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_less_than_sum_of_sins_l819_81987

theorem sin_sum_less_than_sum_of_sins (α β : Real) 
  (h1 : 0 < α) (h2 : α < π/2) (h3 : 0 < β) (h4 : β < π/2) : 
  Real.sin (α + β) < Real.sin α + Real.sin β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_less_than_sum_of_sins_l819_81987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_three_zeros_a_range_l819_81920

/-- The function f(x) defined as (ln x)^2 - (a/2)x ln x + (a/e)x^2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x)^2 - (a/2) * x * Real.log x + (a/Real.exp 1) * x^2

/-- Theorem stating that if f(x) has three distinct real zeros, then a is in the range (-2/e, 0) -/
theorem f_three_zeros_a_range (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) →
  -2 / Real.exp 1 < a ∧ a < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_three_zeros_a_range_l819_81920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_values_l819_81906

def U : Set ℝ := {1, 3, 5, 7, 9}

def A (a : ℝ) : Set ℝ := {1, |a + 1|, 9}

def complement_A (a : ℝ) : Set ℝ := {x ∈ U | x ∉ A a}

theorem a_values (a : ℝ) : 
  (complement_A a = {5, 7}) → (a = 2 ∨ a = -4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_values_l819_81906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_meeting_point_l819_81907

/-- The slope length in meters -/
noncomputable def slope_length : ℝ := 110

/-- The male athlete's uphill speed in meters per second -/
noncomputable def male_uphill_speed : ℝ := 3

/-- The male athlete's downhill speed in meters per second -/
noncomputable def male_downhill_speed : ℝ := 5

/-- The female athlete's uphill speed in meters per second -/
noncomputable def female_uphill_speed : ℝ := 2

/-- The female athlete's downhill speed in meters per second -/
noncomputable def female_downhill_speed : ℝ := 3

/-- The function to calculate the time for an athlete to complete a round trip -/
noncomputable def round_trip_time (uphill_speed downhill_speed : ℝ) : ℝ :=
  slope_length / uphill_speed + slope_length / downhill_speed

/-- The theorem stating where the athletes meet for the second time -/
theorem second_meeting_point :
  ∃ x : ℝ, x = 47 + 1/7 ∧
  round_trip_time male_uphill_speed male_downhill_speed + x / male_downhill_speed =
  round_trip_time female_uphill_speed female_downhill_speed + (slope_length - x) / female_uphill_speed :=
by sorry

#check second_meeting_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_meeting_point_l819_81907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_ratio_l819_81915

/-- Calculate simple interest -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Calculate compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem: The ratio of simple interest to compound interest is 1:2 -/
theorem interest_ratio :
  let si := simple_interest 603.75 14 6
  let ci := compound_interest 7000 7 2
  si / ci = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_ratio_l819_81915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_y_equals_closed_form_l819_81942

open BigOperators

/-- The sequence defined in the problem -/
def y (n : ℕ) : ℕ → ℚ
  | 0 => 1
  | 1 => n
  | (k + 2) => ((n + 1) * y n (k + 1) - (n - k) * y n k) / (k + 2)

/-- The sum of the sequence up to the n-th term -/
def sum_y (n : ℕ) : ℚ :=
  ∑ k in Finset.range (n + 1), y n k

/-- The closed form of the sum -/
def closed_form_sum (n : ℕ) : ℚ :=
  ∑ k in Finset.range (n + 1), (n + k).choose k

theorem sum_y_equals_closed_form (n : ℕ) :
  sum_y n = closed_form_sum n := by
  sorry  -- The proof is omitted for brevity

#check sum_y_equals_closed_form

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_y_equals_closed_form_l819_81942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_l819_81940

/-- Two lines are parallel if their slopes are equal -/
def parallel (m1 m2 : ℝ) : Prop := m1 = m2

/-- The slope of the line ax + y - 1 = 0 -/
noncomputable def slope1 (a : ℝ) : ℝ := -a

/-- The slope of the line x + ay + 1 = 0 -/
noncomputable def slope2 (a : ℝ) : ℝ := -1/a

theorem parallel_lines (a : ℝ) :
  (a = 1 → parallel (slope1 a) (slope2 a)) ∧
  (∃ b : ℝ, b ≠ 1 ∧ parallel (slope1 b) (slope2 b)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_l819_81940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_reciprocals_l819_81929

/-- Curve C₁ in polar coordinates -/
def C₁ (ρ θ : ℝ) : Prop := ρ^2 - 4*ρ*(Real.cos θ) - 4*ρ*(Real.sin θ) + 7 = 0

/-- Line C₂ in polar coordinates -/
def C₂ (θ : ℝ) : Prop := θ = Real.pi/3

/-- Intersection points of C₁ and C₂ -/
def intersection (ρ : ℝ) : Prop := C₁ ρ (Real.pi/3) ∧ C₂ (Real.pi/3)

theorem intersection_sum_reciprocals :
  ∃ ρ₁ ρ₂ : ℝ, 
    intersection ρ₁ ∧ 
    intersection ρ₂ ∧ 
    ρ₁ ≠ ρ₂ ∧
    (1 / ρ₁ + 1 / ρ₂ = (2 * Real.sqrt 3 + 2) / 7) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_reciprocals_l819_81929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_y_coordinate_l819_81989

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/8 = 1

-- Define the right focus F
def right_focus : ℝ × ℝ := (3, 0)

-- Define point A
noncomputable def point_A : ℝ × ℝ := (0, 6 * Real.sqrt 6)

-- Define a point P on the left branch of the hyperbola
def point_P (x y : ℝ) : Prop := 
  hyperbola x y ∧ x < 0

-- Define the perimeter of triangle APF
noncomputable def perimeter (px py : ℝ) : ℝ :=
  let (ax, ay) := point_A
  let (fx, fy) := right_focus
  Real.sqrt ((px - ax)^2 + (py - ay)^2) +
  Real.sqrt ((px - fx)^2 + (py - fy)^2) +
  Real.sqrt ((ax - fx)^2 + (ay - fy)^2)

-- Theorem statement
theorem min_perimeter_y_coordinate :
  ∃ (px py : ℝ), point_P px py ∧
  (∀ (qx qy : ℝ), point_P qx qy → perimeter px py ≤ perimeter qx qy) ∧
  py = 2 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_y_coordinate_l819_81989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l819_81969

theorem division_problem (x y : ℕ) (h1 : x % y = 9) (h2 : (x : ℝ) / (y : ℝ) = 96.15) : y = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l819_81969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_satisfies_conditions_l819_81911

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

noncomputable def perimeter (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  distance x1 y1 x2 y2 + distance x2 y2 x3 y3 + distance x3 y3 x1 y1

noncomputable def area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem no_triangle_satisfies_conditions : ¬ ∃ (x y : ℝ), 
  (distance 0 0 8 0 = 8) ∧ 
  (perimeter 0 0 8 0 x y = 40) ∧ 
  (area 0 0 8 0 x y = 80) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_triangle_satisfies_conditions_l819_81911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l819_81947

theorem inequality_solution (x : ℝ) :
  x ≠ 5 →
  (x * (x + 3)) / ((x - 5)^2) ≥ 15 ↔ x ∈ Set.Icc 3 5 ∪ Set.Ico 5 (125/14) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l819_81947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_inequality_inequality_condition_l819_81903

noncomputable section

def f (x : ℝ) := Real.exp x

-- Part 1
theorem tangent_line_inequality (k b : ℝ) (h : ∃ t, f t = k * t + b ∧ (deriv f) t = k) :
  ∀ x, f x ≥ k * x + b :=
sorry

-- Part 2
theorem inequality_condition (k b : ℝ) 
  (h : ∀ x ≥ 0, f x ≥ k * x + b) :
  (k ≤ 1 ∧ b ≤ 1) ∨ (k > 1 ∧ b ≤ k * (1 - Real.log k)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_inequality_inequality_condition_l819_81903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_decreasing_l819_81949

-- Define the function f(x) = -x/2
noncomputable def f (x : ℝ) : ℝ := -x/2

-- Theorem statement
theorem f_is_decreasing : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ :=
by
  -- Introduce variables and hypothesis
  intro x₁ x₂ h
  -- Unfold the definition of f
  unfold f
  -- Apply properties of real numbers
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_decreasing_l819_81949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_problem_l819_81925

-- Define the circles and their properties
def circle_O (x : ℝ) := {diameter : ℝ // ∃ area : ℝ, area = 12 + 2*x}
def circle_P (x : ℝ) := {diameter : ℝ // ∃ area : ℝ, area = 24 + x}
def circle_Q (x : ℝ) := {diameter : ℝ // ∃ area : ℝ, area = 108 - x}

-- Define points A, B, and C
noncomputable def point_A : ℝ × ℝ := sorry
noncomputable def point_B : ℝ × ℝ := sorry
noncomputable def point_C : ℝ × ℝ := sorry

-- Define the condition that C is on circle O
def C_on_circle_O (x : ℝ) : Prop := sorry

-- Define the relationship between circle areas and diameters
axiom circle_area_diameter_relation (diameter area : ℝ) : area = Real.pi * (diameter / 2)^2

-- Theorem to prove
theorem circle_problem (x : ℝ) 
  (hO : circle_O x)
  (hP : circle_P x)
  (hQ : circle_Q x)
  (hC : C_on_circle_O x) :
  x = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_problem_l819_81925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walker_speed_is_pi_over_three_l819_81944

/-- Represents a track with straight sides and semicircular ends -/
structure Track where
  width : ℝ
  time_difference : ℝ

/-- Calculates the speed of a walker on the given track -/
noncomputable def walker_speed (track : Track) : ℝ :=
  (2 * Real.pi * track.width) / track.time_difference

theorem walker_speed_is_pi_over_three (track : Track) 
  (h1 : track.width = 6)
  (h2 : track.time_difference = 36) : 
  walker_speed track = Real.pi / 3 := by
  sorry

#check walker_speed_is_pi_over_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walker_speed_is_pi_over_three_l819_81944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_expected_rainfall_l819_81970

-- Define the weather probabilities for each week
def week1_probabilities : List (Float × Float) := [(0.3, 0), (0.4, 2), (0.3, 8)]
def week2_probabilities : List (Float × Float) := [(0.2, 0), (0.3, 6), (0.5, 12)]

-- Define the number of days per week
def days_per_week : Float := 5

-- Calculate the expected rainfall for a single day given probabilities
def expected_rainfall_per_day (probabilities : List (Float × Float)) : Float :=
  probabilities.foldr (fun (p, r) acc => acc + p * r) 0

-- Calculate the expected rainfall for a week
def expected_rainfall_per_week (probabilities : List (Float × Float)) : Float :=
  expected_rainfall_per_day probabilities * days_per_week

-- Theorem to prove
theorem total_expected_rainfall :
  expected_rainfall_per_week week1_probabilities + expected_rainfall_per_week week2_probabilities = 55.0 := by
  sorry

#eval expected_rainfall_per_week week1_probabilities + expected_rainfall_per_week week2_probabilities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_expected_rainfall_l819_81970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_youngest_dwarf_age_is_16_l819_81950

/-- The age of Snow White now -/
def snow_white_age : ℕ := sorry

/-- The sum of ages of all seven dwarfs a year ago -/
def sum_dwarfs_year_ago : ℕ := sorry

/-- The sum of ages of the six eldest dwarfs now -/
def sum_six_eldest_now : ℕ := sorry

/-- Snow White's age a year ago equals the sum of all seven dwarfs' ages a year ago -/
axiom snow_white_year_ago : snow_white_age - 1 = sum_dwarfs_year_ago

/-- Snow White's age in two years equals the sum of the six eldest dwarfs' ages now -/
axiom snow_white_two_years : snow_white_age + 2 = sum_six_eldest_now

/-- The age of the youngest dwarf now -/
def youngest_dwarf_age : ℕ := 16

/-- The sum of all seven dwarfs' ages now -/
def sum_dwarfs_now : ℕ := sum_six_eldest_now + youngest_dwarf_age

theorem youngest_dwarf_age_is_16 : 
  youngest_dwarf_age = 16 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_youngest_dwarf_age_is_16_l819_81950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_g_equals_half_l819_81909

/-- Definition of g(n) for positive integers n -/
noncomputable def g (n : ℕ+) : ℝ := ∑' k : ℕ+, (1 : ℝ) / (k + 2 : ℝ) ^ n.val

/-- The sum of g(n) from n=2 to infinity equals 1/2 -/
theorem sum_g_equals_half : ∑' n : ℕ+, g (n + 1) = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_g_equals_half_l819_81909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_l819_81983

/-- Superhero's speed in miles per 4 minutes -/
noncomputable def superhero_speed : ℝ := 10

/-- Supervillain's speed in miles per hour -/
noncomputable def supervillain_speed : ℝ := 100

/-- Number of minutes in an hour -/
noncomputable def minutes_per_hour : ℝ := 60

/-- Number of 4-minute intervals in an hour -/
noncomputable def intervals_per_hour : ℝ := minutes_per_hour / 4

/-- Distance the superhero can run in an hour -/
noncomputable def superhero_distance : ℝ := superhero_speed * intervals_per_hour

/-- Distance the supervillain can drive in an hour -/
noncomputable def supervillain_distance : ℝ := supervillain_speed

/-- Theorem stating the difference in distance traveled -/
theorem distance_difference : superhero_distance - supervillain_distance = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_difference_l819_81983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l819_81968

/-- A function f defined on [-1, 1] with specific properties -/
noncomputable def f (x : ℝ) : ℝ := x / (1 + x^2)

/-- Theorem stating the properties and behavior of function f -/
theorem f_properties :
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f (-x) = -f x) ∧  -- f is odd
  (f (1/3) = 3/10) ∧  -- f(1/3) = 3/10
  (∀ x y, x ∈ Set.Icc (-1 : ℝ) 1 → y ∈ Set.Icc (-1 : ℝ) 1 → x < y → f x < f y) ∧  -- f is increasing on [-1, 1]
  (∀ m : ℝ, (∃ x, x ∈ Set.Icc (1/2 : ℝ) 1 ∧ ∀ x, x ∈ Set.Icc (1/2 : ℝ) 1 → f (m*x - x) + f (x^2 - 1) > 0) ↔ 1 < m ∧ m ≤ 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l819_81968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_k_is_positive_integer_l819_81995

theorem sqrt_k_is_positive_integer (k m : ℕ+) 
  (h : ∃ n : ℤ, (1/2 : ℝ) * (Real.sqrt (k + 4 * Real.sqrt m) - Real.sqrt k) = n) :
  ∃ n : ℕ+, Real.sqrt k = n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_k_is_positive_integer_l819_81995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_recurring_digit_sum_l819_81974

-- Define a polynomial with integer coefficients
def MyPolynomial (α : Type) := List (α × ℕ)

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define the sequence a_n
def mySequence (P : MyPolynomial ℤ) (n : ℕ) : ℕ := 
  sumOfDigits (sorry) -- Replace with actual evaluation of P at n when implemented

theorem infinitely_recurring_digit_sum 
  (P : MyPolynomial ℤ) : 
  ∃ (k : ℕ), ∀ (N : ℕ), ∃ (n : ℕ), n > N ∧ mySequence P n = k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_recurring_digit_sum_l819_81974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_2_sqrt_5_l819_81905

/-- The length of the common chord between two circles -/
noncomputable def common_chord_length (r₁ : ℝ) (a b c : ℝ) : ℝ :=
  let d := |c| / Real.sqrt (a^2 + b^2)
  2 * Real.sqrt (r₁^2 - d^2)

/-- Theorem stating that the length of the common chord between the given circles is 2√5 -/
theorem common_chord_length_is_2_sqrt_5 :
  common_chord_length 50 12 6 (-10) = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_2_sqrt_5_l819_81905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l819_81938

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  BD : Real

-- State the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : 2 * t.a * Real.cos t.C - t.c = 2 * t.b)
  (h2 : t.c = Real.sqrt 2)
  (h3 : t.BD = Real.sqrt 3) :
  t.A = 2 * Real.pi / 3 ∧ t.a = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l819_81938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_max_min_exp_on_unit_interval_l819_81926

-- Define the function f(x) = 2^x
noncomputable def f (x : ℝ) : ℝ := 2^x

-- State the theorem
theorem sum_max_min_exp_on_unit_interval :
  (⨆ x ∈ Set.Icc 0 1, f x) + (⨅ x ∈ Set.Icc 0 1, f x) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_max_min_exp_on_unit_interval_l819_81926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_nature_l819_81951

theorem quadratic_roots_nature :
  ∃! r : ℝ, r^2 - 6*r*Real.sqrt 2 + 18 = 0 ∧ r = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_nature_l819_81951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cos_sin_function_l819_81939

theorem min_value_cos_sin_function :
  (∀ x : ℝ, 2 * (Real.cos x)^2 + Real.sin x + 3 ≥ 2) ∧
  (∃ x : ℝ, 2 * (Real.cos x)^2 + Real.sin x + 3 = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_cos_sin_function_l819_81939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_probability_l819_81978

/-- Two circles with equations x² + y² = 1 and (x-a)² + (y-a)² = 1, where a ≠ 0 -/
def Circle1 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
def Circle2 (a : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + (p.2 - a)^2 = 1}

/-- The circles have a common point -/
def HaveCommonPoint (a : ℝ) : Prop := ∃ p : ℝ × ℝ, p ∈ Circle1 ∧ p ∈ Circle2 a

/-- The length of the chord of intersection -/
noncomputable def ChordLength (a : ℝ) : ℝ := Real.sqrt (4 - 2 * a^2)

/-- The probability that the length of the chord of intersection is not less than √2 -/
noncomputable def Probability (a : ℝ) : ℝ := 
  if -1 ≤ a ∧ a ≤ 1 then Real.sqrt 2 / 2 else 0

/-- The main theorem -/
theorem intersection_chord_probability (a : ℝ) (ha : a ≠ 0) (h : HaveCommonPoint a) : 
  Probability a = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_probability_l819_81978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_brasil_paths_l819_81917

/-- Represents a move in the grid -/
inductive GridMove
| H  -- Horizontal move
| V  -- Vertical move

/-- Represents a path in the grid -/
def GridPath := List GridMove

/-- The length of a valid path -/
def pathLength : Nat := 6

/-- The size of the grid -/
def gridSize : Nat := 6

/-- Checks if a path is valid (has correct length) -/
def isValidPath (p : GridPath) : Bool :=
  p.length = pathLength

/-- Counts the number of valid paths for a given starting row -/
def countPathsForRow (row : Nat) : Nat :=
  match row with
  | 0 => 1  -- First row: only one path (VVVVV)
  | 1 => 5  -- Second row: 5 paths
  | 2 => 10 -- Third row: 10 paths
  | 3 => 10 -- Fourth row: 10 paths
  | 4 => 5  -- Fifth row: 5 paths
  | 5 => 1  -- Sixth row: only one path (HHHHH)
  | _ => 0  -- Invalid row

/-- Counts the total number of valid paths in the grid -/
def totalValidPaths : Nat :=
  List.range gridSize
  |> List.map countPathsForRow
  |> List.sum

theorem count_brasil_paths :
  totalValidPaths = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_brasil_paths_l819_81917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l819_81956

/-- A circle centered at the origin with radius 3 -/
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 9

/-- A point P with coordinates (3,1) -/
def point_P : ℝ × ℝ := (3, 1)

/-- A line is tangent to the circle if it intersects the circle at exactly one point -/
def is_tangent_line (a b c : ℝ) : Prop :=
  ∃! (x y : ℝ), my_circle x y ∧ a*x + b*y + c = 0

/-- A line passes through point P -/
def passes_through_P (a b c : ℝ) : Prop :=
  a * point_P.1 + b * point_P.2 + c = 0

theorem tangent_lines_to_circle :
  (∀ a b c : ℝ, is_tangent_line a b c ∧ passes_through_P a b c →
    (a = 4 ∧ b = 3 ∧ c = -15) ∨ (a = 1 ∧ b = 0 ∧ c = -3)) ∧
  is_tangent_line 4 3 (-15) ∧ passes_through_P 4 3 (-15) ∧
  is_tangent_line 1 0 (-3) ∧ passes_through_P 1 0 (-3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l819_81956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_for_given_triangle_l819_81924

-- Define a right triangle
structure RightTriangle where
  area : ℝ
  height : ℝ
  hypotenuse : ℝ

-- Define the median on the hypotenuse
noncomputable def median_on_hypotenuse (t : RightTriangle) : ℝ := t.hypotenuse / 2

-- Theorem statement
theorem median_length_for_given_triangle :
  ∀ t : RightTriangle,
  t.area = 8 ∧ t.height = 2 →
  median_on_hypotenuse t = 4 := by
  intro t ht
  have h1 : t.hypotenuse = 8 := by
    -- Proof steps would go here
    sorry
  unfold median_on_hypotenuse
  rw [h1]
  norm_num
  
#check median_length_for_given_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_for_given_triangle_l819_81924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_fee_formula_for_high_consumption_l819_81900

/-- Represents the gas consumption tiers --/
inductive GasTier
| Tier1
| Tier2
| Tier3

/-- Calculates the gas fee based on consumption and household size --/
noncomputable def gasFee (consumption : ℝ) (householdSize : ℕ) : ℝ :=
  let tier1Limit := if householdSize > 4 then 400 + 100 * (householdSize - 4 : ℝ) else 400
  let tier2Limit := if householdSize > 4 then 1200 + 200 * (householdSize - 4 : ℝ) else 1200
  if consumption ≤ tier1Limit then
    consumption * 2.67
  else if consumption ≤ tier2Limit then
    tier1Limit * 2.67 + (consumption - tier1Limit) * 3.15
  else
    tier1Limit * 2.67 + (tier2Limit - tier1Limit) * 3.15 + (consumption - tier2Limit) * 3.63

/-- Theorem: For a household with no more than 4 people and consumption > 1200 m³,
    the gas fee is 3.63x - 768 yuan --/
theorem gas_fee_formula_for_high_consumption
  (consumption : ℝ) (householdSize : ℕ) 
  (h1 : consumption > 1200) 
  (h2 : householdSize ≤ 4) :
  gasFee consumption householdSize = 3.63 * consumption - 768 := by
  sorry

#check gas_fee_formula_for_high_consumption

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_fee_formula_for_high_consumption_l819_81900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l819_81941

noncomputable def curve_C1 (x y : ℝ) : Prop := x^2/4 + y^2 = 1

noncomputable def curve_C2 (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 1

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_range :
  ∀ (x1 y1 x2 y2 : ℝ),
    curve_C1 x1 y1 → curve_C2 x2 y2 →
    1 ≤ distance x1 y1 x2 y2 ∧ distance x1 y1 x2 y2 ≤ 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_l819_81941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l819_81957

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Define for 0 to cover all natural numbers
  | 1 => 1
  | (n + 2) => 2 * sequence_a (n + 1) / (sequence_a (n + 1) + 2)

theorem a_10_value : sequence_a 10 = 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_value_l819_81957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_natasha_hill_climb_l819_81961

/-- Represents a hill climbing journey with ascent and descent -/
structure HillClimb where
  ascent_time : ℚ
  descent_time : ℚ
  total_avg_speed : ℚ

/-- Calculates the average speed during ascent given a HillClimb -/
def ascent_avg_speed (hc : HillClimb) : ℚ :=
  (hc.total_avg_speed * (hc.ascent_time + hc.descent_time)) / (2 * hc.ascent_time)

/-- Theorem stating that for the given conditions, the ascent average speed is 3 km/h -/
theorem natasha_hill_climb :
  let hc : HillClimb := {
    ascent_time := 4,
    descent_time := 2,
    total_avg_speed := 4
  }
  ascent_avg_speed hc = 3 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_natasha_hill_climb_l819_81961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faye_age_l819_81991

-- Define the ages of Chad, Diana, Eduardo, and Faye
variable (chad diana eduardo faye : ℕ)

-- State the conditions
axiom diana_eduardo : diana = eduardo - 2
axiom eduardo_chad : eduardo = chad + 5
axiom faye_chad : faye = chad + 4
axiom diana_age : diana = 15

-- State the theorem
theorem faye_age : faye = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faye_age_l819_81991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ada_original_seat_l819_81902

/- Define the seat numbers -/
def Seat := Fin 5

/- Define the friends -/
inductive Friend
| Ada
| Bea
| Cee
| Dee
| Ed

/- Define the seating arrangement as a function from Friend to Seat -/
def Seating := Friend → Seat

/- Define the original seating arrangement -/
noncomputable def original_seating : Seating := sorry

/- Define the final seating arrangement after movements -/
noncomputable def final_seating : Seating := sorry

/- State the theorem -/
theorem ada_original_seat :
  /- Bea moved two seats to the right -/
  (final_seating Friend.Bea).val = (original_seating Friend.Bea).val + 2 ∧
  /- Cee moved one seat to the left -/
  (final_seating Friend.Cee).val = (original_seating Friend.Cee).val - 1 ∧
  /- Dee and Ed exchanged seats -/
  (final_seating Friend.Dee = original_seating Friend.Ed ∧
   final_seating Friend.Ed = original_seating Friend.Dee) ∧
  /- Ada is in the leftmost seat after movements -/
  final_seating Friend.Ada = ⟨0, sorry⟩ ∧
  /- The seating is a bijection (one-to-one correspondence) -/
  Function.Bijective original_seating ∧
  Function.Bijective final_seating →
  /- Ada's original seat was 2 -/
  (original_seating Friend.Ada).val = 1 := by
  sorry

#eval Fin.val (⟨1, sorry⟩ : Fin 5)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ada_original_seat_l819_81902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_points_l819_81963

/-- Calculates the points earned from a shopping trip --/
def calculate_points (amount_spent : ℚ) (is_fifth_visit : Bool) : ℕ :=
  let base_points := (amount_spent * 5).floor.toNat
  let over_hundred_bonus := if amount_spent > 100 then 250 else 0
  let loyalty_bonus := if is_fifth_visit then 100 else 0
  base_points + over_hundred_bonus + loyalty_bonus

/-- Theorem stating that Martha earns 850 points --/
theorem martha_points : calculate_points 106.10 true = 850 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martha_points_l819_81963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_solution_decomposition_l819_81932

/-- The equation from the original problem -/
def original_equation (x : ℝ) : Prop :=
  2 / (x - 2) + 7 / (x - 7) + 11 / (x - 11) + 13 / (x - 13) = x^2 - 9*x - 7

/-- Checks if a given real number is a solution to the original equation -/
def is_solution (n : ℝ) : Prop :=
  original_equation n

/-- Checks if a given real number is the largest solution to the original equation -/
def is_largest_solution (n : ℝ) : Prop :=
  is_solution n ∧ ∀ x, is_solution x → x ≤ n

/-- The theorem to be proved -/
theorem largest_solution_decomposition :
  ∃ (n : ℝ) (d e f : ℕ), is_largest_solution n ∧ 
    n = d + Real.sqrt (e + Real.sqrt (f : ℝ)) ∧
    d + e + f = 343 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_solution_decomposition_l819_81932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_purchase_ratio_l819_81922

/-- Represents the number of people who purchased a book or combination of books. -/
structure BookPurchases where
  only_a : ℕ
  only_b : ℕ
  both : ℕ

/-- The ratio of two natural numbers. -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

theorem book_purchase_ratio (purchases : BookPurchases)
  (h1 : purchases.both = 500)
  (h2 : purchases.both = 2 * purchases.only_b)
  (h3 : purchases.only_a = 1000)
  (h4 : ∃ k : ℕ, purchases.only_a + purchases.both = k * (purchases.only_b + purchases.both)) :
  Ratio.mk (purchases.only_a + purchases.both) (purchases.only_b + purchases.both) = Ratio.mk 2 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_purchase_ratio_l819_81922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_pass_midpoint_is_90_seconds_l819_81958

/-- Represents a train with its length and speed -/
structure Train where
  length : ℝ  -- length in meters
  speed : ℝ   -- speed in km/hr

/-- Calculates the time taken by the slowest train to pass a point equidistant from the drivers of the faster trains -/
noncomputable def time_to_pass_midpoint (train1 train2 train3 : Train) : ℝ :=
  let slowest_speed := min (min train1.speed train2.speed) train3.speed
  let fastest_two_sum := train1.length + train2.length + train3.length - 
                         (if train1.speed = slowest_speed then train1.length
                          else if train2.speed = slowest_speed then train2.length
                          else train3.length)
  (fastest_two_sum / 2) / (slowest_speed * 1000 / 3600)

/-- Theorem stating that for the given train configurations, the time to pass the midpoint is 90 seconds -/
theorem time_to_pass_midpoint_is_90_seconds :
  let train1 : Train := { length := 650, speed := 45 }
  let train2 : Train := { length := 750, speed := 30 }
  let train3 : Train := { length := 850, speed := 60 }
  time_to_pass_midpoint train1 train2 train3 = 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_pass_midpoint_is_90_seconds_l819_81958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_roll_volume_difference_l819_81980

noncomputable section

/-- The volume of a cylinder with radius r and height h -/
def cylinderVolume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The radius of a cylinder formed by rolling a rectangular sheet -/
def cylinderRadius (circumference : ℝ) : ℝ := circumference / (2 * Real.pi)

theorem paper_roll_volume_difference : 
  let sheet_width : ℝ := 7
  let sheet_length : ℝ := 9
  let alice_radius := cylinderRadius sheet_width
  let bob_radius := cylinderRadius sheet_length
  let alice_volume := cylinderVolume alice_radius sheet_length
  let bob_volume := cylinderVolume bob_radius sheet_width
  Real.pi * |alice_volume - bob_volume| = 850.5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_roll_volume_difference_l819_81980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nh4oh_ionization_constant_l819_81999

/-- The ionization constant of NH4OH -/
noncomputable def K_ion (α : ℝ) (C_total : ℝ) : ℝ :=
  (α * C_total)^2 / ((1 - α) * C_total)

/-- Theorem: The ionization constant of NH4OH is approximately 1.76e-5 -/
theorem nh4oh_ionization_constant :
  let α : ℝ := 1.33e-2
  let C_total : ℝ := 0.1
  |K_ion α C_total - 1.76e-5| < 1e-7 := by
  sorry

#check nh4oh_ionization_constant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nh4oh_ionization_constant_l819_81999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_mono_decreasing_implies_a_range_a_range_implies_f_mono_decreasing_l819_81930

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 - a * x)

-- Define the property of being monotonically decreasing on an interval
def MonoDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f y < f x

-- State the theorem
theorem f_mono_decreasing_implies_a_range :
  ∀ a : ℝ, (MonoDecreasing (f a) 0 1) → 0 < a ∧ a ≤ 1 := by
  sorry

-- Prove the converse (not required by the original problem, but included for completeness)
theorem a_range_implies_f_mono_decreasing :
  ∀ a : ℝ, 0 < a ∧ a ≤ 1 → (MonoDecreasing (f a) 0 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_mono_decreasing_implies_a_range_a_range_implies_f_mono_decreasing_l819_81930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_containing_square_l819_81918

/-- Represents a square with a given side length -/
structure Square where
  side : ℝ

/-- Defines the sequence of squares with side lengths 1, 1/2, 1/3, ... -/
noncomputable def square_sequence : ℕ → Square :=
  λ n => ⟨1 / (n + 1 : ℝ)⟩

/-- Defines what it means for a square to contain another square -/
def contains (outer inner : Square) : Prop :=
  outer.side ≥ inner.side

/-- Defines what it means for a square to contain a sequence of squares without overlapping -/
def contains_all_without_overlap (outer : Square) : Prop :=
  ∃ (arrangement : ℕ → ℝ × ℝ),
    (∀ n, let (x, y) := arrangement n;
      0 ≤ x ∧ x + (square_sequence n).side ≤ outer.side ∧
      0 ≤ y ∧ y + (square_sequence n).side ≤ outer.side) ∧
    (∀ m n, m ≠ n →
      let (x₁, y₁) := arrangement m
      let (x₂, y₂) := arrangement n;
      x₁ + (square_sequence m).side ≤ x₂ ∨
      x₂ + (square_sequence n).side ≤ x₁ ∨
      y₁ + (square_sequence m).side ≤ y₂ ∨
      y₂ + (square_sequence n).side ≤ y₁)

/-- The main theorem statement -/
theorem smallest_containing_square :
  (∃ (s : Square), contains_all_without_overlap s) ∧
  (∀ (s : Square), contains_all_without_overlap s → s.side ≥ 1.5) ∧
  contains_all_without_overlap ⟨1.5⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_containing_square_l819_81918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_z_implies_a_equals_two_l819_81914

-- Define the complex number z as a function of a
def z (a : ℝ) : ℂ := Complex.mk (a^2 - a - 2) (a + 1)

-- State the theorem
theorem purely_imaginary_z_implies_a_equals_two :
  ∀ a : ℝ, (z a).re = 0 ∧ (z a).im ≠ 0 → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_z_implies_a_equals_two_l819_81914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strict_concavity_condition_max_interval_length_l819_81962

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/12) * x^4 - (m/6) * x^3 - (3/2) * x^2

-- Define the second derivative of f
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x - 3

-- Part I: Strict concavity condition on [0,3]
theorem strict_concavity_condition (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 3, g m x < 0) ↔ m > 2 := by sorry

-- Part II: Maximum interval length for strict concavity when |m| ≤ 2
theorem max_interval_length : 
  (∃ a b : ℝ, a < b ∧ b - a = 2 ∧ 
    ∀ m : ℝ, |m| ≤ 2 → ∀ x ∈ Set.Ioo a b, g m x < 0) ∧
  (∀ a b : ℝ, a < b ∧ (∀ m : ℝ, |m| ≤ 2 → ∀ x ∈ Set.Ioo a b, g m x < 0) 
    → b - a ≤ 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_strict_concavity_condition_max_interval_length_l819_81962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertices_distance_l819_81933

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  16 * x^2 - 32 * x - y^2 + 4 * y + 48 = 0

-- Define the distance between vertices
noncomputable def distance_between_vertices (eq : (ℝ → ℝ → Prop)) : ℝ :=
  3/2

-- Theorem statement
theorem hyperbola_vertices_distance :
  distance_between_vertices hyperbola_equation = 3/2 :=
by
  -- The proof steps would go here
  sorry

#check hyperbola_vertices_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertices_distance_l819_81933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_backpack_sales_theorem_l819_81946

/-- Represents the backpack sales model -/
structure BackpackSales where
  purchase_price : ℚ
  initial_price : ℚ
  initial_sales : ℚ
  price_increment : ℚ
  sales_decrement : ℚ
  min_sales : ℚ

/-- Calculates the sales volume based on the selling price -/
def sales_volume (model : BackpackSales) (price : ℚ) : ℚ :=
  model.initial_sales - ((price - model.initial_price) / model.price_increment) * model.sales_decrement

/-- Calculates the profit based on the selling price -/
def profit (model : BackpackSales) (price : ℚ) : ℚ :=
  (price - model.purchase_price) * (sales_volume model price)

/-- The main theorem about the backpack sales model -/
theorem backpack_sales_theorem (model : BackpackSales) 
  (h_purchase : model.purchase_price = 30)
  (h_initial_price : model.initial_price = 40)
  (h_initial_sales : model.initial_sales = 280)
  (h_price_increment : model.price_increment = 2)
  (h_sales_decrement : model.sales_decrement = 20)
  (h_min_sales : model.min_sales = 130) :
  (∃ (max_price : ℚ), max_price = 54 ∧ 
    ∀ (p : ℚ), sales_volume model p ≥ model.min_sales → p ≤ max_price) ∧
  (∃ (profit_price : ℚ), profit_price = 42 ∧ profit model profit_price = 3120) ∧
  (¬∃ (p : ℚ), profit model p = 3700) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_backpack_sales_theorem_l819_81946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l819_81931

theorem triangle_properties (a b c A B C : ℝ) :
  -- Given conditions
  c = 2 →
  C = π/3 →
  -- Part I
  ((1/2) * a * b * Real.sin C = Real.sqrt 3 →
   a = 2 ∧ b = 2) ∧
  -- Part II
  (Real.sin C + Real.sin (B - A) = 2 * Real.sin (2 * A) →
   A = π/2 ∨ A = π/6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l819_81931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_value_of_f_l819_81988

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 4) + Real.sin (x / 7)

theorem smallest_max_value_of_f :
  let x := 2610 * (π / 180)  -- Convert 2610° to radians
  (∀ y : ℝ, 0 < y → y < x → f y < f x) ∧
  (∀ z : ℝ, f z ≤ f x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_value_of_f_l819_81988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_T_l819_81982

-- Define the solid T
def T : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p;
                    (|x| + |y| ≤ 2) ∧ (|x| + |z| ≤ 2) ∧ (|y| + |z| ≤ 2)}

-- Define the volume function
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem volume_of_T : volume T = 32/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_T_l819_81982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sum_theorem_l819_81967

/-- Sequence of numbers 1, 19, 199, 1999, ... -/
def specialSequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => 10 * specialSequence n + 9

/-- Predicate to check if a number has all digits as 2 except one -/
def allTwosExceptOne (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ d ≠ 2 ∧ ∃ k : ℕ, n = k * 10 + d ∧ ∀ m, m < Nat.log 10 k → (k / 10^m) % 10 = 2

theorem special_sum_theorem :
  ∃ (S : Finset ℕ) (k : ℕ → ℕ),
    S.card ≥ 3 ∧
    (∀ i ∈ S, k i < k (i + 1)) ∧
    (∀ i ∈ S, specialSequence (k i) ∈ S) ∧
    allTwosExceptOne (S.sum id) ∧
    (∃ d, (d = 0 ∨ d = 1) ∧ allTwosExceptOne ((S.sum id) + d)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sum_theorem_l819_81967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_intersection_A_B_l819_81994

-- Define the function (marked as noncomputable due to Real.sqrt)
noncomputable def f (x : ℝ) := Real.sqrt (x^2 - 9)

-- Define the domain set A
def A : Set ℝ := {x | x ≥ 3 ∨ x ≤ -3}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for the domain of f
theorem domain_of_f : {x : ℝ | ∃ y, f x = y} = A := by sorry

-- Theorem for the intersection of A and B
theorem intersection_A_B (a : ℝ) :
  A ∩ B a = 
    if a ≤ -3 then
      {x | x < a}
    else if a ≤ 3 then
      {x | x ≤ -3}
    else
      {x | x ≤ -3 ∨ (3 ≤ x ∧ x < a)} := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_intersection_A_B_l819_81994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_paths_count_l819_81965

def grid_size : ℕ := 5

def start : ℕ × ℕ := (5, 0)
def finish : ℕ × ℕ := (0, 5)
def avoid : ℕ × ℕ := (1, 1)

def is_valid_path (path : List (ℕ × ℕ)) : Prop :=
  path.length = grid_size * 2 ∧
  path.head? = some start ∧
  path.getLast? = some finish ∧
  avoid ∉ path ∧
  ∀ i, i < path.length - 1 →
    let (x₁, y₁) := path[i]!
    let (x₂, y₂) := path[i+1]!
    ((x₁ = x₂ ∧ y₂ = y₁ + 1) ∨ (y₁ = y₂ ∧ x₂ = x₁ - 1))

def count_valid_paths : ℕ := sorry

theorem valid_paths_count : count_valid_paths = 56 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_paths_count_l819_81965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_hourly_rate_l819_81984

-- Define the work completion times and earnings for x and y
noncomputable def x_days : ℝ := 15
noncomputable def y_days : ℝ := 10
noncomputable def x_earnings : ℝ := 450
noncomputable def y_earnings : ℝ := 600

-- Define the total time taken when working together
noncomputable def total_days : ℝ := 5

-- Define z's working hours per day
noncomputable def z_hours_per_day : ℝ := 4

-- Define the total earnings
noncomputable def total_earnings : ℝ := x_earnings + y_earnings

-- Define the work done by x and y in 5 days
noncomputable def x_work : ℝ := total_days / x_days
noncomputable def y_work : ℝ := total_days / y_days

-- Define the total work done by x and y
noncomputable def xy_total_work : ℝ := x_work + y_work

-- Define z's work
noncomputable def z_work : ℝ := 1 - xy_total_work

-- Define z's total working hours
noncomputable def z_total_hours : ℝ := z_hours_per_day * total_days

-- Theorem to prove
theorem z_hourly_rate : 
  z_work / 1 * total_earnings / z_total_hours = 8.75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_hourly_rate_l819_81984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l819_81919

-- Define the terms of the geometric sequence
noncomputable def x₁ (a : ℝ) : ℝ := -a + Real.log 2017 / Real.log 3
noncomputable def x₂ (a : ℝ) : ℝ := 2*a + Real.log 2017 / Real.log 2
noncomputable def x₃ (a : ℝ) : ℝ := -4*a + Real.log 2017 / Real.log 21

-- State the theorem
theorem geometric_sequence_common_ratio (a : ℝ) :
  (x₃ a - x₂ a) / (x₂ a - x₁ a) = 8 / 15 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l819_81919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_negative_one_range_of_a_for_subset_condition_l819_81975

def f (a x : ℝ) : ℝ := |2*x - a| + |2*x - 1|

theorem solution_set_for_negative_one :
  {x : ℝ | f (-1) x ≤ 2} = Set.Icc (-1/2) (1/2) := by sorry

theorem range_of_a_for_subset_condition :
  {a : ℝ | ∀ x ∈ Set.Icc (1/2) 1, f a x ≤ |2*x + 1|} = Set.Icc 0 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_negative_one_range_of_a_for_subset_condition_l819_81975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_area_l819_81936

-- Define the cylinder type
structure Cylinder where
  lateral_length : ℝ
  lateral_width : ℝ

-- Define the surface area function
noncomputable def surface_area (c : Cylinder) : ℝ :=
  if c.lateral_length = 2 * Real.pi * (c.lateral_width / (2 * Real.pi)) then
    c.lateral_length * c.lateral_width + 2 * Real.pi * (c.lateral_width / (2 * Real.pi))^2
  else
    c.lateral_length * c.lateral_width + 2 * Real.pi * (c.lateral_length / (2 * Real.pi))^2

-- Theorem statement
theorem cylinder_surface_area (c : Cylinder) 
  (h1 : c.lateral_length = 4 ∧ c.lateral_width = 6) :
  surface_area c = 24 + 18/Real.pi ∨ surface_area c = 24 + 8/Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_area_l819_81936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_with_geometric_progression_exists_l819_81971

/-- Represents a quadrilateral in 2D space -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

/-- Calculates the distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Checks if a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop := sorry

/-- Checks if a list of real numbers forms a geometric progression -/
def is_geometric_progression (l : List ℝ) : Prop := sorry

/-- Theorem: There exists a convex quadrilateral whose side lengths and diagonals form a geometric progression -/
theorem convex_quadrilateral_with_geometric_progression_exists : 
  ∃ (q : Quadrilateral), 
    is_convex q ∧ 
    is_geometric_progression 
      [distance q.A q.B, distance q.B q.C, distance q.C q.D, distance q.D q.A, 
       distance q.A q.C, distance q.B q.D] :=
by sorry

#check convex_quadrilateral_with_geometric_progression_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_with_geometric_progression_exists_l819_81971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_sequences_bounded_l819_81990

def g₁ : ℕ → ℕ
  | 0 => 1
  | n + 1 => sorry  -- Definition based on prime factorization

def g (m : ℕ) : ℕ → ℕ
  | 0 => g₁ 0
  | n + 1 => match m with
    | 0 => n + 1
    | m + 1 => g₁ (g m (n + 1))

def is_bounded (seq : ℕ → ℕ) : Prop :=
  ∃ M, ∀ n, seq n ≤ M

theorem all_sequences_bounded :
  ∀ N : ℕ, N ≤ 100 → is_bounded (λ m ↦ g m N) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_sequences_bounded_l819_81990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l819_81908

def U : Set ℝ := Set.univ

def A : Set ℝ := {-1, 0, 1, 2, 3, 4, 5, 6}

def B : Set ℝ := {x : ℝ | ∃ n : ℕ, x = n ∧ n < 4}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l819_81908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_5_l819_81959

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (2 * x + 1)

-- Define the sequence of functions f_n
noncomputable def f_n : ℕ → (ℝ → ℝ)
  | 0 => f
  | n + 1 => f ∘ f_n n

-- State the theorem
theorem min_value_f_5 :
  ∀ x : ℝ, x ≥ 1/2 → x ≤ 1 → f_n 5 x ≥ 1/12 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_5_l819_81959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_existence_l819_81948

theorem system_solution_existence (b : ℝ) : 
  (∃ (a x y : ℝ), x = abs (y - b) + 3 / b ∧ 
    x^2 + y^2 + 32 = a * (2 * y - a) + 12 * x) ↔ 
  b < 0 ∨ b ≥ 3 / 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_existence_l819_81948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_art_gallery_pieces_l819_81992

/-- Represents the total number of pieces of art in the gallery -/
def total_art : ℕ := 900

/-- Represents the number of sculptures not on display -/
def sculptures_not_displayed : ℕ := 400

theorem art_gallery_pieces :
  (total_art / 3 : ℚ) = (total_art - sculptures_not_displayed : ℚ) ∧
  (total_art / 18 : ℚ) = ((total_art / 3 : ℚ) / 6 : ℚ) ∧
  ((2 * total_art / 3 : ℚ) / 3 : ℚ) = ((total_art - sculptures_not_displayed : ℚ) / 3 : ℚ) ∧
  sculptures_not_displayed = (2 * total_art / 3 : ℚ).floor * 2 / 3 :=
by sorry

#eval total_art

end NUMINAMATH_CALUDE_ERRORFEEDBACK_art_gallery_pieces_l819_81992
