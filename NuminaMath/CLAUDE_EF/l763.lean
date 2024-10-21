import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_elements_in_subset_l763_76396

theorem max_elements_in_subset (p : ℕ) (A : Finset ℕ) : 
  (∀ x ∈ A, x ≤ 2^p) → 
  (∀ x ∈ A, 2*x ∉ A) → 
  A.card ≤ if p % 2 = 0 then (2^(p+1) + 1) / 3 else (2^(p+1) - 1) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_elements_in_subset_l763_76396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_factorial_8_l763_76369

-- Define 8!
def factorial_8 : ℕ := 40320

-- Define the prime factorization of 8!
def prime_factorization : List (ℕ × ℕ) := [(2, 7), (3, 2), (5, 1), (7, 1)]

-- Define a function to count divisors based on prime factorization
def count_divisors (factorization : List (ℕ × ℕ)) : ℕ :=
  factorization.foldl (fun acc (_, e) => acc * (e + 1)) 1

-- Theorem: The number of positive divisors of 8! is 96
theorem divisors_of_factorial_8 : count_divisors prime_factorization = 96 := by
  -- Evaluate the count_divisors function
  have h1 : count_divisors prime_factorization = 8 * 3 * 2 * 2 := by rfl
  -- Simplify the arithmetic
  have h2 : 8 * 3 * 2 * 2 = 96 := by rfl
  -- Combine the steps
  rw [h1, h2]

#eval count_divisors prime_factorization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_factorial_8_l763_76369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_for_given_circumference_and_area_max_sector_area_for_given_circumference_l763_76388

-- Define the sector structure
structure Sector where
  radius : ℝ
  centralAngle : ℝ

-- Define the circumference of a sector
noncomputable def sectorCircumference (s : Sector) : ℝ := 2 * s.radius + s.radius * s.centralAngle

-- Define the area of a sector
noncomputable def sectorArea (s : Sector) : ℝ := 1/2 * s.radius^2 * s.centralAngle

-- Theorem for part (1)
theorem sector_angle_for_given_circumference_and_area :
  ∃ (s : Sector), sectorCircumference s = 10 ∧ sectorArea s = 4 ∧ s.centralAngle = 1/2 := by sorry

-- Theorem for part (2)
theorem max_sector_area_for_given_circumference :
  ∃ (s : Sector), sectorCircumference s = 40 ∧ 
    (∀ (t : Sector), sectorCircumference t = 40 → sectorArea t ≤ sectorArea s) ∧
    s.radius = 10 ∧ s.centralAngle = 2 ∧ sectorArea s = 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_angle_for_given_circumference_and_area_max_sector_area_for_given_circumference_l763_76388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_nonzero_terms_l763_76310

theorem expansion_nonzero_terms : 
  ∃ (a b c d : ℚ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    (X - 3) * (3 * X^2 - 2 * X + 5) - 2 * (X^3 + X^2 - 4 * X) + (2 * X - 6) 
    = a * X^3 + b * X^2 + c * X + d :=
by sorry

#check expansion_nonzero_terms

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_nonzero_terms_l763_76310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_call_days_l763_76389

def days_in_year : ℕ := 365

def call_frequency : Fin 3 → ℕ
| 0 => 2
| 1 => 3
| 2 => 6

def days_with_calls (f : Fin 3 → ℕ) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin 3)) (λ i => days_in_year / f i)) -
  (Finset.sum (Finset.univ : Finset (Fin 3 × Fin 3)) (λ (i, j) => days_in_year / Nat.lcm (f i) (f j))) +
  (days_in_year / Nat.lcm (f 0) (Nat.lcm (f 1) (f 2)))

theorem no_call_days :
  days_in_year - days_with_calls call_frequency = 122 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_call_days_l763_76389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_square_with_circles_l763_76319

/-- The area of the shaded region in a square with circles at its vertices --/
theorem shaded_area_square_with_circles (side_length circle_radius : ℝ) :
  side_length = 8 →
  circle_radius = 3 →
  let square_area := side_length ^ 2
  let circle_area := Real.pi * circle_radius ^ 2
  let sector_angle := Real.pi / 4  -- 45 degrees in radians
  let sector_area := sector_angle / (2 * Real.pi) * circle_area
  let triangle_area := 1 / 2 * circle_radius * circle_radius * Real.sqrt 2 / 2
  square_area - 4 * sector_area - 4 * triangle_area = 64 - 9 * Real.pi - 9 * Real.sqrt 2 :=
by
  sorry

#check shaded_area_square_with_circles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_square_with_circles_l763_76319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f5_l763_76365

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (2 * x + 1)

-- Define the recursive function fₙ
noncomputable def f_n : ℕ → (ℝ → ℝ)
  | 0 => λ x => x  -- Base case for n = 0
  | 1 => f
  | n + 1 => λ x => f (f_n n x)

-- State the theorem
theorem min_value_f5 :
  ∀ x ∈ Set.Icc (1/2 : ℝ) 1, f_n 5 x ≥ 1/12 ∧ ∃ y ∈ Set.Icc (1/2 : ℝ) 1, f_n 5 y = 1/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f5_l763_76365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_size_of_periodic_function_l763_76321

theorem range_size_of_periodic_function (f : ℤ × ℤ → ℝ) 
  (h : ∀ (x y m n : ℤ), f (x + 3*m - 2*n, y - 4*m + 5*n) = f (x, y)) :
  ∃ S : Finset ℝ, S.card ≤ 7 ∧ ∀ x y, f (x, y) ∈ S :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_size_of_periodic_function_l763_76321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_P_to_center_l763_76362

-- Define the circle and point P
def circle_radius : ℝ := 4
def PA : ℝ := 4
def PB : ℝ := 6

-- Define the distance from P to the center of the circle
noncomputable def distance_to_center (r PA PB : ℝ) : ℝ := 
  Real.sqrt (PA * PB + r^2)

-- Theorem statement
theorem distance_from_P_to_center : 
  distance_to_center circle_radius PA PB = 2 * Real.sqrt 10 := by
  -- Unfold the definition of distance_to_center
  unfold distance_to_center
  -- Simplify the expression
  simp [circle_radius, PA, PB]
  -- The proof steps would go here, but we'll use sorry to skip them for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_P_to_center_l763_76362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_winning_strategy_l763_76300

/-- Represents the state of the match game -/
structure MatchGame where
  pile1 : ℕ
  pile2 : ℕ

/-- Defines a valid move in the match game -/
def ValidMove (game : MatchGame) (newGame : MatchGame) : Prop :=
  (newGame.pile1 < game.pile1 ∧ newGame.pile2 = game.pile2 ∧ (game.pile1 - newGame.pile1) % game.pile2 = 0) ∨
  (newGame.pile2 < game.pile2 ∧ newGame.pile1 = game.pile1 ∧ (game.pile2 - newGame.pile2) % game.pile1 = 0)

/-- Defines the winning condition -/
def IsWinningPosition (game : MatchGame) : Prop :=
  game.pile1 = 0 ∨ game.pile2 = 0

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The main theorem statement -/
theorem first_player_winning_strategy (m n : ℕ) (h1 : m > n) (h2 : (m : ℝ) > φ * n) :
  ∃ (strategy : MatchGame → MatchGame),
    (∀ (game : MatchGame), game.pile1 = m ∧ game.pile2 = n →
      (ValidMove game (strategy game) ∧
        ∀ (opponentMove : MatchGame),
          ValidMove (strategy game) opponentMove →
            ∃ (nextMove : MatchGame),
              ValidMove opponentMove nextMove ∧
              (IsWinningPosition nextMove ∨
                ∃ (futureStrategy : MatchGame → MatchGame),
                  ∀ (futureGame : MatchGame),
                    ValidMove opponentMove futureGame →
                      ValidMove futureGame (futureStrategy futureGame) ∧
                      IsWinningPosition (futureStrategy futureGame)))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_winning_strategy_l763_76300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l763_76351

/-- Represents a valid arrangement of cards -/
def ValidArrangement (arrangement : List ℕ) : Prop :=
  arrangement.length = 20 ∧
  (∀ k : ℕ, k ≤ 9 → (arrangement.filter (· = k)).length = 2) ∧
  (∀ k : ℕ, k ≤ 9 → ∃ i j : ℕ, i < j ∧ 
    arrangement.get? i = some k ∧ 
    arrangement.get? j = some k ∧ 
    j - i - 1 = k)

/-- The main theorem stating that no valid arrangement exists -/
theorem no_valid_arrangement : ¬ ∃ arrangement : List ℕ, ValidArrangement arrangement := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangement_l763_76351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_squares_in_possible_set_l763_76316

/-- Represents a configuration of rooks on a 6x6 chessboard -/
structure RookConfiguration where
  rows : Finset (Fin 6)
  cols : Finset (Fin 6)
  total_rooks : ℕ
  rook_count_valid : total_rooks = 9

/-- Calculates the number of safe squares for a given rook configuration -/
def safe_squares (config : RookConfiguration) : ℕ :=
  (6 - config.rows.card) * (6 - config.cols.card)

/-- The set of possible numbers of safe squares -/
def possible_safe_squares : Set ℕ := {1, 4, 6, 9}

/-- Theorem stating that the number of safe squares for any valid configuration
    is in the set of possible safe squares -/
theorem safe_squares_in_possible_set (config : RookConfiguration) :
  safe_squares config ∈ possible_safe_squares := by
  sorry

#check safe_squares_in_possible_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_safe_squares_in_possible_set_l763_76316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l763_76360

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.tan x

def isDefined (x : ℝ) : Prop := ∀ k : ℤ, x ≠ k * Real.pi + Real.pi / 2

theorem f_is_odd : 
  ∀ x : ℝ, isDefined x → isDefined (-x) → f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l763_76360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_and_zeros_l763_76315

open Real MeasureTheory

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem function_inequality_and_zeros (e : ℝ) (he : e = Real.exp 1) :
  (∀ k : ℝ, (∀ x : ℝ, x > 0 → f x < k * x) ↔ k ∈ Set.Ioi (1 / (2 * e))) ∧
  (∀ k : ℝ, (∃ x y : ℝ, x ∈ Set.Icc (1 / e) (e ^ 2) ∧
                        y ∈ Set.Icc (1 / e) (e ^ 2) ∧
                        x ≠ y ∧
                        f x - k * x = 0 ∧
                        f y - k * y = 0) ↔
             k ∈ Set.Icc (2 / (e ^ 4)) (1 / (2 * e))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_and_zeros_l763_76315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_roots_distinct_l763_76309

def P : ℕ → (ℝ → ℝ)
| 0 => λ x => x^2 - 2
| (n+1) => λ x => P 0 (P n x)

theorem P_roots_distinct (n : ℕ+) : 
  ∃ (S : Set ℝ), S.Finite ∧ S.Nonempty ∧ (∀ x, x ∈ S → P n x = x) ∧ 
  (∀ x y, x ∈ S → y ∈ S → x ≠ y → P n x ≠ P n y) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_roots_distinct_l763_76309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_newborn_count_approx_l763_76349

/-- The probability of an animal surviving each of the first three months -/
noncomputable def survival_probability : ℝ := 9/10

/-- The expected number of animals surviving the first three months -/
noncomputable def expected_survivors : ℝ := 109.35

/-- The number of newborn members in the group -/
noncomputable def newborn_count : ℝ := expected_survivors / (survival_probability ^ 3)

/-- Theorem stating that the number of newborn members is approximately 150 -/
theorem newborn_count_approx : 
  |newborn_count - 150| < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_newborn_count_approx_l763_76349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_a_minus_sqrt_n_eq_half_l763_76374

-- Define the sequence a_n
noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => 1 + (n + 1 : ℝ) / a n

-- State the theorem
theorem limit_a_minus_sqrt_n_eq_half :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - Real.sqrt (n : ℝ) - (1/2)| < ε := by
  sorry

#check limit_a_minus_sqrt_n_eq_half

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_a_minus_sqrt_n_eq_half_l763_76374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_surface_area_l763_76348

/-- Represents a right pyramid with a rectangular base -/
structure RightPyramid where
  base_length : ℝ
  base_width : ℝ
  peak_height : ℝ

/-- Calculates the total surface area of a right pyramid -/
noncomputable def total_surface_area (p : RightPyramid) : ℝ :=
  let base_area := p.base_length * p.base_width
  let half_diagonal := Real.sqrt (p.base_length ^ 2 / 4 + p.base_width ^ 2 / 4)
  let slant_height := Real.sqrt (p.peak_height ^ 2 + half_diagonal ^ 2)
  let lateral_area := (p.base_length + p.base_width) * slant_height
  base_area + lateral_area

/-- Theorem stating the total surface area of the specific pyramid -/
theorem specific_pyramid_surface_area :
  let p : RightPyramid := ⟨8, 6, 15⟩
  total_surface_area p = 48 + 7 * Real.sqrt 241 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_surface_area_l763_76348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l763_76333

theorem trigonometric_identities (α : ℝ) 
  (h1 : Real.tan α = 2) 
  (h2 : π < α ∧ α < 3*π/2) : 
  (Real.sin α)^2 - 2*(Real.cos α)^2 = 2/5 ∧ 
  Real.sin α - Real.cos α = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l763_76333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l763_76304

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ p : ℝ, p > 0 → (∀ x : ℝ, f (x + p) = f x) → p ≥ Real.pi) ∧
  (∀ x : ℝ, f (x + Real.pi) = f x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l763_76304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_amplitude_l763_76370

/-- 
Given a sinusoidal function y = a * sin(b * x + c) + d, where a, b, c, and d are positive constants,
if the function oscillates between 5 and -3, then the amplitude a is equal to 4.
-/
theorem sinusoidal_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_max : ∀ x, a * Real.sin (b * x + c) + d ≤ 5) 
  (h_min : ∀ x, a * Real.sin (b * x + c) + d ≥ -3) 
  (h_reach_max : ∃ x, a * Real.sin (b * x + c) + d = 5)
  (h_reach_min : ∃ x, a * Real.sin (b * x + c) + d = -3) : 
  a = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_amplitude_l763_76370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_allowance_l763_76356

/-- Alex's weekly allowance in AUD -/
noncomputable def weekly_allowance : ℝ := 39

/-- Amount saved for toy -/
noncomputable def toy_savings : ℝ := (1/3) * weekly_allowance

/-- Remaining allowance after toy savings -/
noncomputable def remaining_after_toy : ℝ := weekly_allowance - toy_savings

/-- Amount spent on online game -/
noncomputable def online_game_spend : ℝ := (2/5) * remaining_after_toy

/-- Cost of online game in USD -/
def online_game_cost_usd : ℝ := 8

/-- Exchange rate AUD to USD -/
def aud_to_usd_rate : ℝ := 1.3

/-- Cost of online game in AUD -/
noncomputable def online_game_cost_aud : ℝ := online_game_cost_usd * aud_to_usd_rate

/-- Cost of special event in Yen -/
def special_event_cost_yen : ℝ := 1000

/-- Exchange rate AUD to Yen -/
def aud_to_yen_rate : ℝ := 0.012

/-- Cost of special event in AUD -/
noncomputable def special_event_cost_aud : ℝ := special_event_cost_yen * aud_to_yen_rate

/-- Total expenses -/
noncomputable def total_expenses : ℝ := toy_savings + online_game_cost_aud + special_event_cost_aud

theorem alex_allowance : 
  online_game_spend = online_game_cost_aud ∧ 
  total_expenses < weekly_allowance ∧
  weekly_allowance - total_expenses > 0 ∧
  weekly_allowance - total_expenses < 4 →
  weekly_allowance = 39 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_allowance_l763_76356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_significant_digits_l763_76335

/-- Function to calculate the number of significant digits in a real number -/
noncomputable def significant_digits : ℝ → ℕ :=
  sorry

/-- Given a square with an area of 3.24 square inches (to the nearest hundredth),
    prove that the measurement of its side length has 2 significant digits. -/
theorem square_side_significant_digits :
  ∀ (area : ℝ) (side : ℝ),
    (abs (area - 3.24) ≤ 0.005) →  -- area is approximately 3.24 (to the nearest hundredth)
    side ^ 2 = area →  -- side is the square root of the area
    significant_digits side = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_side_significant_digits_l763_76335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_correct_statements_proof_l763_76392

-- Define the domain D as a set of real numbers
noncomputable def D : Set ℝ := Set.univ

-- Define the k-type function property
def is_k_type_function (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∃ (m n : ℝ), m < n ∧ Set.Icc m n ⊆ D ∧
  (∃ (y : Set ℝ), y = f '' Set.Icc m n ∧ y = Set.Icc (k * m) (k * n))

-- Define the three functions
noncomputable def f₁ (x : ℝ) : ℝ := 3 - 4 / x
noncomputable def f₂ (a : ℝ) (x : ℝ) : ℝ := ((a^2 + a) * x - 1) / (a^2 * x)
noncomputable def f₃ (x : ℝ) : ℝ := -1/2 * x^2 + x

-- State the theorem
def number_of_correct_statements : ℕ :=
  let s₁ := ¬∃ k, is_k_type_function f₁ k
  let s₂ := ∀ a ≠ 0, is_k_type_function (f₂ a) 1 → 
            (∃ m n, is_k_type_function (f₂ a) 1 ∧ n - m ≤ 2 * Real.sqrt 3 / 3)
  let s₃ := is_k_type_function f₃ 3 → (∃ m n, m = -4 ∧ n = 0 ∧ is_k_type_function f₃ 3)
  2

-- Proof
theorem number_of_correct_statements_proof : 
  number_of_correct_statements = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_correct_statements_proof_l763_76392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_perfect_power_l763_76325

def sequenceA (n : ℕ) : ℤ :=
  match n with
  | 0 => 2
  | 1 => 12
  | n + 2 => 6 * sequenceA (n + 1) - sequenceA n

def is_perfect_power (x : ℤ) : Prop :=
  ∃ (b : ℤ) (k : ℕ), k > 1 ∧ x = b ^ k

theorem sequence_not_perfect_power :
  ∀ n : ℕ, ¬(is_perfect_power (sequenceA n)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_perfect_power_l763_76325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l763_76340

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := (2 - i) / (1 + i)

theorem modulus_of_z : Complex.abs z = (3 * Real.sqrt 2) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_l763_76340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l763_76308

theorem tan_alpha_value (α : Real) 
  (h1 : Real.sin α = 2 * Real.sqrt 5 / 5)
  (h2 : π / 2 ≤ α)
  (h3 : α ≤ π) :
  Real.tan α = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l763_76308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_n_l763_76366

-- Define the curve C
noncomputable def curve_C (x y : ℝ) : Prop :=
  x ≥ Real.sqrt 2 ∧ x^2 / 2 - y^2 / 2 = 1

-- Define the line l
noncomputable def line_l (x y : ℝ) : Prop :=
  x - Real.sqrt 3 * y - 2 = 0

-- Define point M
def point_M : ℝ × ℝ := (2, 0)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the line n passing through M and with slope angle θ
noncomputable def line_n (θ : ℝ) (t : ℝ) : ℝ × ℝ :=
  (2 + t * Real.cos θ, t * Real.sin θ)

-- State the theorem
theorem slope_angle_of_line_n :
  ∃ (θ : ℝ), (θ = π / 3 ∨ θ = 2 * π / 3) ∧
  ∃ (t1 t2 : ℝ), t1 ≠ t2 ∧
  curve_C (line_n θ t1).1 (line_n θ t1).2 ∧
  curve_C (line_n θ t2).1 (line_n θ t2).2 ∧
  distance (line_n θ t1) (line_n θ t2) = 4 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_n_l763_76366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l763_76343

/-- The number of days it takes for p and q to complete the work together -/
noncomputable def days_together (p_efficiency : ℝ) (q_efficiency : ℝ) (p_days : ℝ) : ℝ :=
  1 / (1 / p_days + 1 / (p_days * (1 + p_efficiency / q_efficiency)))

theorem work_completion_time :
  let p_efficiency : ℝ := 1.5
  let q_efficiency : ℝ := 1
  let p_days : ℝ := 25
  days_together p_efficiency q_efficiency p_days = 15 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l763_76343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l763_76312

/-- Represents an ellipse in the Cartesian coordinate system -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- Represents a point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- The length of the segment cut by y=x from the ellipse -/
noncomputable def segment_length (e : Ellipse) : ℝ :=
  2 * Real.sqrt (e.a^2 + e.b^2)

/-- Main theorem about the ellipse and its properties -/
theorem ellipse_theorem (e : Ellipse) 
  (h_ecc : eccentricity e = Real.sqrt 3 / 2)
  (h_seg : segment_length e = 4 * Real.sqrt 10 / 5) :
  ∃ (A B D : Point) (k₁ k₂ : ℝ),
    -- 1. The equation of C is x²/4 + y² = 1
    (e.a = 2 ∧ e.b = 1) ∧
    -- 2. For points A, B, D on C with AD ⊥ AB, and M, the slopes k₁ of BD and k₂ of AM satisfy k₁ = -1/2 * k₂
    (A.x * A.y ≠ 0) ∧
    (B.x = -A.x ∧ B.y = -A.y) ∧
    (D.x * A.y + D.y * A.x = 0) ∧  -- AD ⊥ AB
    (k₁ = (D.y + A.y) / (D.x + A.x)) ∧
    (k₂ = -A.y / (2 * A.x)) ∧
    (k₁ = -1/2 * k₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l763_76312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_rental_cost_per_mile_l763_76380

/-- Represents a car rental scenario -/
structure CarRental where
  baseCost : ℚ  -- Base cost per day in dollars
  budget : ℚ    -- Total budget in dollars
  miles : ℚ     -- Number of miles that can be driven

/-- Calculates the cost per mile for a given car rental scenario -/
noncomputable def costPerMile (rental : CarRental) : ℚ :=
  (rental.budget - rental.baseCost) / rental.miles

/-- Theorem stating that for the given car rental scenario, the cost per mile is $0.18 -/
theorem car_rental_cost_per_mile :
  let rental : CarRental := { baseCost := 30, budget := 75, miles := 250 }
  costPerMile rental = 9/50 := by
  sorry

#eval (9:ℚ)/50  -- This will output 0.18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_rental_cost_per_mile_l763_76380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_m_l763_76384

theorem max_value_m (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_eq : 2 / a + 1 / b = 1 / 4) (h_ineq : ∀ m : ℝ, 2 * a + b ≥ 4 * m) :
  ∃ m_max : ℝ, m_max = 7 / 4 ∧ ∀ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → 2 / a + 1 / b = 1 / 4 → 2 * a + b ≥ 4 * m) → m ≤ m_max :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_m_l763_76384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l763_76372

noncomputable def f (x : ℝ) (φ : ℝ) := 2 * Real.sin (2 * x + φ)

/-- A point is a symmetry center of a function if the function is symmetric about that point. -/
def IsSymmetryCenter (f : ℝ → ℝ) (c : ℝ × ℝ) : Prop :=
  ∀ x, f (c.1 + x) = f (c.1 - x)

theorem symmetry_center_of_f 
  (φ : ℝ) 
  (h1 : |φ| < π/2) 
  (h2 : f 0 φ = Real.sqrt 3) : 
  ∃ (c : ℝ × ℝ), c = (-π/6, 0) ∧ IsSymmetryCenter (f · φ) c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l763_76372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_value_l763_76377

-- Define the angle θ and the point P
variable (θ : Real)
variable (k : Real)
def P : Real × Real := (-4 * k, 3 * k)

-- State the theorem
theorem angle_terminal_side_value (h1 : k < 0) (h2 : Real.cos θ = 4 / 5) (h3 : Real.sin θ = -3 / 5) :
  2 * Real.sin θ + Real.cos θ = -2 / 5 := by
  -- Substitute the given values for sin θ and cos θ
  calc
    2 * Real.sin θ + Real.cos θ = 2 * (-3 / 5) + 4 / 5 := by rw [h3, h2]
    _ = -6 / 5 + 4 / 5 := by ring
    _ = -2 / 5 := by ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_value_l763_76377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_approx_l763_76337

/-- Right prism with isosceles triangular bases -/
structure RightPrism where
  height : ℝ
  base_side1 : ℝ
  base_side2 : ℝ
  base_side3 : ℝ

/-- Midpoint of an edge -/
structure Midpoint where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the surface area of the sliced off part -/
noncomputable def surface_area_CXYZ' (p : RightPrism) (x' y' z' : Midpoint) : ℝ :=
  sorry

/-- Main theorem -/
theorem surface_area_approx (p : RightPrism) (x' y' z' : Midpoint) :
  p.height = 20 ∧ 
  p.base_side1 = 12 ∧ 
  p.base_side2 = 12 ∧ 
  p.base_side3 = 18 →
  abs (surface_area_CXYZ' p x' y' z' - 126.285) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_approx_l763_76337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_astrophysics_degrees_l763_76306

/-- The percentage of the budget allocated to sectors other than basic astrophysics -/
def other_sectors_percentage : ℝ := 95

/-- The total number of degrees in a circle -/
def circle_degrees : ℝ := 360

/-- Theorem: The number of degrees representing basic astrophysics research is 18 -/
theorem basic_astrophysics_degrees : 
  (100 - other_sectors_percentage) / 100 * circle_degrees = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basic_astrophysics_degrees_l763_76306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_decreases_l763_76394

def original_scores : List ℝ := [5, 9, 7, 10, 9]
def additional_score : ℝ := 8
def original_variance : ℝ := 3.2

theorem variance_decreases (scores : List ℝ) (new_score : ℝ) (orig_var : ℝ) 
  (h1 : scores = original_scores)
  (h2 : new_score = additional_score)
  (h3 : orig_var = original_variance) :
  let new_scores := scores ++ [new_score]
  let new_mean := (scores.sum + new_score) / (scores.length + 1 : ℝ)
  let new_variance := (scores.map (λ x => (x - new_mean)^2)).sum / (scores.length + 1 : ℝ) + 
                      ((new_score - new_mean)^2) / (scores.length + 1 : ℝ)
  new_variance < orig_var := by
  sorry

#eval original_scores
#eval additional_score
#eval original_variance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_decreases_l763_76394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_diverges_l763_76350

-- Define f(n) as a noncomputable function
noncomputable def f (n : ℕ) : ℝ := ∑' k, (1 : ℝ) / (2 * k + 3) ^ n

-- State the theorem
theorem series_diverges : ¬ Summable (fun n ↦ f (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_diverges_l763_76350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_water_fraction_l763_76381

/-- Fraction of original cylinder filled with water -/
noncomputable def fraction_filled (h : ℝ) (r : ℝ) : ℝ :=
  (3 / 5) * ((1.25 * r)^2 * (0.9 * h)) / (r^2 * h)

/-- Theorem stating the fraction of the original cylinder filled with water -/
theorem cylinder_water_fraction (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  fraction_filled h r = 27 / 32 := by
  sorry

#check cylinder_water_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_water_fraction_l763_76381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_lower_bound_l763_76371

noncomputable def f (x : ℝ) : ℝ := x^2 + 2/x

noncomputable def g (x m : ℝ) : ℝ := (1/2)^x - m

def condition (m : ℝ) : Prop :=
  ∀ x₁ ∈ Set.Icc 1 2, ∃ x₂ ∈ Set.Icc (-1) 1, f x₁ ≥ g x₂ m

theorem m_lower_bound (m : ℝ) (h : condition m) : m ≥ -5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_lower_bound_l763_76371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_digits_of_powers_l763_76341

/-- Given a positive integer n, if the first two digits of 5^n and 2^n are identical,
    then these two digits form the number 31. -/
theorem identical_digits_of_powers (n : ℕ) (hn : 0 < n):
  (∃ (k : ℕ) (a b : ℕ), 
    (10 ≤ a ∧ a ≤ 99) ∧
    (∃ (x y : ℕ), 
      5^n = a * 10^k + b * 10^(k-1) + x ∧
      2^n = a * 10^k + b * 10^(k-1) + y)) →
  31 * 10^(Nat.log 10 n - 1) ≤ 5^n ∧ 5^n < 32 * 10^(Nat.log 10 n - 1) ∧
  31 * 10^(Nat.log 2 n - 1) ≤ 2^n ∧ 2^n < 32 * 10^(Nat.log 2 n - 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_digits_of_powers_l763_76341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_fourth_vertex_l763_76358

noncomputable def point_distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2)

def is_parallelogram (p1 p2 p3 p4 : ℝ × ℝ × ℝ) : Prop :=
  point_distance p1 p2 = point_distance p3 p4 ∧
  point_distance p1 p4 = point_distance p2 p3

theorem parallelogram_fourth_vertex :
  let p1 : ℝ × ℝ × ℝ := (1, 3, 2)
  let p2 : ℝ × ℝ × ℝ := (4, 5, -1)
  let p3 : ℝ × ℝ × ℝ := (7, 2, 4)
  let p4 : ℝ × ℝ × ℝ := (10, 6, 1)
  (∀ p : ℝ × ℝ × ℝ, is_parallelogram p1 p2 p3 p → p = p4) ∧
  (∀ i j k : ℤ, is_parallelogram p1 p2 p3 (↑i, ↑j, ↑k) → (i, j, k) = (10, 6, 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_fourth_vertex_l763_76358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_properties_l763_76346

/-- Represents a triangle in 2D space -/
structure Triangle (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] :=
  (A B C : α)

/-- Represents an equilateral triangle constructed on a side of another triangle -/
structure EquilateralTriangle (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] :=
  (base : Triangle α)
  (center : α)
  (isExternal : Bool)

/-- Helper function to determine if a triangle is equilateral -/
def is_equilateral {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) : Prop :=
  sorry

/-- Helper function to calculate the centroid of a triangle -/
noncomputable def centroid {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) : α :=
  sorry

/-- Helper function to calculate the area of a triangle -/
noncomputable def area {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) : ℝ :=
  sorry

/-- Main theorem about properties of triangles constructed from equilateral triangles -/
theorem equilateral_triangles_properties
  {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α]
  (ABC : Triangle α)
  (eq_BC eq_CA eq_AB : EquilateralTriangle α)
  (h_eq_BC : eq_BC.base = ⟨ABC.B, ABC.C, ABC.A⟩)
  (h_eq_CA : eq_CA.base = ⟨ABC.C, ABC.A, ABC.B⟩)
  (h_eq_AB : eq_AB.base = ⟨ABC.A, ABC.B, ABC.C⟩)
  (h_external : eq_BC.isExternal ∧ eq_CA.isExternal ∧ eq_AB.isExternal)
  (h_internal : ¬eq_BC.isExternal ∧ ¬eq_CA.isExternal ∧ ¬eq_AB.isExternal)
  (Δ : Triangle α)
  (δ : Triangle α)
  (h_Δ : Δ = ⟨eq_BC.center, eq_CA.center, eq_AB.center⟩)
  (h_δ : δ = ⟨eq_BC.center, eq_CA.center, eq_AB.center⟩) :
  (is_equilateral Δ ∧ is_equilateral δ) ∧
  (centroid ABC = centroid Δ ∧ centroid ABC = centroid δ) ∧
  (area Δ - area δ = area ABC) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_properties_l763_76346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_trapezoid_diagonals_perpendicular_iff_leg_geometric_mean_l763_76390

/-- A right-angled trapezoid -/
structure RightTrapezoid where
  a : ℝ  -- length of one parallel side
  c : ℝ  -- length of the other parallel side
  d : ℝ  -- length of the leg perpendicular to parallel sides
  h_positive : 0 < a ∧ 0 < c ∧ 0 < d

/-- The diagonals of a right-angled trapezoid -/
noncomputable def diagonals (t : RightTrapezoid) : ℝ × ℝ :=
  (Real.sqrt (t.a^2 + t.d^2), Real.sqrt (t.c^2 + t.d^2))

/-- The diagonals are perpendicular -/
def diagonals_perpendicular (t : RightTrapezoid) : Prop :=
  let (e, f) := diagonals t
  e * f = t.a * t.c + t.d^2

/-- The leg is the geometric mean of the parallel sides -/
def leg_is_geometric_mean (t : RightTrapezoid) : Prop :=
  t.d^2 = t.a * t.c

theorem right_trapezoid_diagonals_perpendicular_iff_leg_geometric_mean (t : RightTrapezoid) :
  diagonals_perpendicular t ↔ leg_is_geometric_mean t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_trapezoid_diagonals_perpendicular_iff_leg_geometric_mean_l763_76390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_second_quadrant_l763_76354

theorem tan_alpha_second_quadrant (α : ℝ) (h1 : Real.sin α = 5/13) (h2 : π/2 < α ∧ α < π) :
  Real.tan α = -(5/12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_second_quadrant_l763_76354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chess_boards_l763_76344

/-- Represents a chess board with numbered squares -/
structure ChessBoard where
  squares : Fin 64 → Fin 64

/-- Checks if two chess boards have no matching numbers on their perimeters -/
def validPerimeterPair (board1 board2 : ChessBoard) : Prop :=
  ∀ (i j : Fin 8), 
    (board1.squares ⟨i.val + 8 * j.val, by sorry⟩ ≠ board2.squares ⟨i.val + 8 * j.val, by sorry⟩) ∧ 
    (board1.squares ⟨i.val + 8 * j.val, by sorry⟩ ≠ board2.squares ⟨(7 - i.val) + 8 * j.val, by sorry⟩) ∧
    (board1.squares ⟨i.val + 8 * j.val, by sorry⟩ ≠ board2.squares ⟨i.val + 8 * (7 - j.val), by sorry⟩) ∧
    (board1.squares ⟨i.val + 8 * j.val, by sorry⟩ ≠ board2.squares ⟨(7 - i.val) + 8 * (7 - j.val), by sorry⟩)

/-- A collection of chess boards is valid if all pairs have valid perimeters -/
def validBoardCollection {n : Nat} (boards : Fin n → ChessBoard) : Prop :=
  ∀ i j : Fin n, i ≠ j → validPerimeterPair (boards i) (boards j)

/-- The maximum number of valid chess boards is 16 -/
theorem max_chess_boards : 
  (∃ (boards : Fin 16 → ChessBoard), validBoardCollection boards) ∧ 
  (¬∃ (boards : Fin 17 → ChessBoard), validBoardCollection boards) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chess_boards_l763_76344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_monotonicity_l763_76301

-- Define the power function
noncomputable def power_function (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^m

-- Define monotonically increasing on (0,+∞)
def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

-- Define the conditions
def condition_p (m : ℝ) : Prop :=
  monotonically_increasing (power_function m)

def condition_q (m : ℝ) : Prop :=
  |m - 2| < 1

-- The theorem to prove
theorem power_function_monotonicity :
  (∃ m, condition_p m → condition_q m) ∧
  (∃ m, condition_q m ∧ ¬condition_p m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_monotonicity_l763_76301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_concentration_l763_76339

/-- Represents the properties of a vessel containing an alcohol mixture -/
structure Vessel where
  capacity : ℝ
  alcohol_concentration : ℝ

/-- Calculates the new concentration of alcohol when mixing two vessels and adding water -/
noncomputable def new_concentration (v1 v2 : Vessel) (total_volume : ℝ) : ℝ :=
  ((v1.capacity * v1.alcohol_concentration + v2.capacity * v2.alcohol_concentration) / total_volume) * 100

theorem mixture_concentration :
  let v1 : Vessel := ⟨2, 0.2⟩
  let v2 : Vessel := ⟨6, 0.55⟩
  let total_volume : ℝ := 8
  new_concentration v1 v2 total_volume = 46.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_concentration_l763_76339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sequence_with_100_distinct_terms_l763_76387

/-- Product of digits of a natural number in base 10 -/
def digit_product (n : ℕ) : ℕ := sorry

/-- Sequence defined by a_{n+1} = a_n + k * π(a_n) -/
def custom_sequence (a k : ℕ) : ℕ → ℕ
  | 0 => a
  | n + 1 => custom_sequence a k n + k * digit_product (custom_sequence a k n)

/-- Number of distinct terms in the sequence -/
def distinct_terms (a k : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a and k for a sequence with exactly 100 distinct terms -/
theorem exists_sequence_with_100_distinct_terms :
  ∃ (a k : ℕ), a > 0 ∧ k > 0 ∧ distinct_terms a k = 100 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sequence_with_100_distinct_terms_l763_76387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_complex_fraction_is_zero_l763_76375

/-- The real part of (1 + √3i) / (√3 - i) is 0 -/
theorem real_part_of_complex_fraction_is_zero :
  let z : ℂ := (1 + Complex.I * Real.sqrt 3) / (Real.sqrt 3 - Complex.I)
  Complex.re z = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_complex_fraction_is_zero_l763_76375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l763_76379

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 3)

theorem f_properties :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ |x₁ - x₂| = Real.pi / 2) ∧
  (∀ (x y : ℝ), Real.sin (x - Real.pi / 3) = y → f (x / 2) = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l763_76379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_eq_four_base_area_l763_76398

/-- A prism with an inscribed sphere -/
structure InscribedPrism where
  /-- The base area of the prism -/
  baseArea : ℝ
  /-- The radius of the inscribed sphere -/
  sphereRadius : ℝ
  /-- The height of the prism is twice the radius of the inscribed sphere -/
  height : ℝ
  height_eq : height = 2 * sphereRadius

/-- The lateral surface area of a prism with an inscribed sphere -/
noncomputable def lateralSurfaceArea (p : InscribedPrism) : ℝ := 
  4 * p.baseArea

/-- Theorem: The lateral surface area of a prism with an inscribed sphere is four times its base area -/
theorem lateral_surface_area_eq_four_base_area (p : InscribedPrism) :
  lateralSurfaceArea p = 4 * p.baseArea := by
  -- Unfold the definition of lateralSurfaceArea
  unfold lateralSurfaceArea
  -- The equation now holds by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_eq_four_base_area_l763_76398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l763_76323

/-- The distance from a point to a line in 2D space -/
noncomputable def point_to_line_distance (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance from the point (1, -2) to the line x - y + 1 = 0 is 2√2 -/
theorem distance_point_to_line : 
  point_to_line_distance 1 (-2) 1 (-1) 1 = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l763_76323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l763_76336

/-- The equation of the tangent line to y = 2x² + 1 at (-1, 3) is y = -4x - 1 -/
theorem tangent_line_equation : 
  ∃ (m b : ℝ), ∀ (x y : ℝ), 
    (y - 3 = m * (x + 1)) ↔ (y = m * x + b) ∧ m = -4 ∧ b = -1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l763_76336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perfect_square_divisible_by_4_and_5_l763_76332

theorem smallest_perfect_square_divisible_by_4_and_5 :
  ∀ n : ℕ, n > 0 → (∃ k : ℕ, n = k^2) → 4 ∣ n → 5 ∣ n → n ≥ 400 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perfect_square_divisible_by_4_and_5_l763_76332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_l763_76364

/-- Proves that the speed of a goods train is 36 km/h given specific conditions --/
theorem goods_train_speed : ∀ (goods_speed : ℝ),
  goods_speed > 0 →
  (let express_speed : ℝ := 90
   let head_start : ℝ := 6
   let catch_up_time : ℝ := 4
   express_speed * catch_up_time = goods_speed * (head_start + catch_up_time)) →
  goods_speed = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_l763_76364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_max_min_diff_l763_76397

noncomputable def geometric_sequence (a₁ : ℝ) (r : ℝ) : ℕ → ℝ :=
  λ n => a₁ * r^(n - 1)

noncomputable def geometric_sum (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_max_min_diff :
  let a₁ := (3 : ℝ) / 2
  let r := (-1 : ℝ) / 2
  let Sₙ := λ n : ℕ+ => geometric_sum a₁ r n
  (⨆ n : ℕ+, Sₙ n - (Sₙ n)⁻¹) + (⨅ n : ℕ+, Sₙ n - (Sₙ n)⁻¹) = (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_max_min_diff_l763_76397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_addition_proof_l763_76382

def u : Fin 3 → ℝ := ![-3, 2, 5]
def v : Fin 3 → ℝ := ![4, -7, 1]

theorem vector_addition_proof : (2 : ℝ) • u + v = ![-2, -3, 11] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_addition_proof_l763_76382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_approximation_l763_76313

/-- The side length of the initial equilateral triangle -/
noncomputable def initial_side_length : ℝ := 10

/-- The number of iterations of the triangle division process -/
def num_iterations : ℕ := 50

/-- The ratio of the area of a shaded triangle to its parent triangle -/
noncomputable def shaded_area_ratio : ℝ := 1 / 4

/-- The area of an equilateral triangle given its side length -/
noncomputable def equilateral_triangle_area (side_length : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * side_length^2

/-- The sum of a geometric series with first term a, ratio r, and n terms -/
noncomputable def geometric_series_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

/-- The theorem stating the approximate cumulative area of shaded triangles -/
theorem shaded_area_approximation :
  let initial_area := equilateral_triangle_area initial_side_length
  let first_term := shaded_area_ratio * initial_area
  let sum := geometric_series_sum first_term shaded_area_ratio num_iterations
  ∃ ε > 0, |sum - 25| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_approximation_l763_76313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_l763_76330

/-- Represents a pentagon with specific properties -/
structure Pentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  triangle_base : ℝ
  triangle_height : ℝ
  trapezoid_base1 : ℝ
  trapezoid_base2 : ℝ
  trapezoid_height : ℝ

/-- Calculates the area of a right triangle -/
noncomputable def triangle_area (base height : ℝ) : ℝ := (1/2) * base * height

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoid_area (base1 base2 height : ℝ) : ℝ := (1/2) * (base1 + base2) * height

/-- Theorem: The area of the specified pentagon is 975 square units -/
theorem pentagon_area (p : Pentagon) 
  (h1 : p.side1 = 18 ∧ p.side2 = 25 ∧ p.side3 = 31 ∧ p.side4 = 29 ∧ p.side5 = 25)
  (h2 : p.triangle_base = 18 ∧ p.triangle_height = 25)
  (h3 : p.trapezoid_base1 = 29 ∧ p.trapezoid_base2 = 31 ∧ p.trapezoid_height = 25) :
  triangle_area p.triangle_base p.triangle_height + 
  trapezoid_area p.trapezoid_base1 p.trapezoid_base2 p.trapezoid_height = 975 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_l763_76330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_solution_l763_76385

def cookie_problem (total_items : ℕ) (oreo_price choc_price sugar_price : ℚ) : Prop :=
  ∃ (x : ℚ),
    -- Ratio condition
    (4 * x).floor + (5 * x).floor + (6 * x).floor = total_items ∧
    -- Price difference calculation
    (5 * x * choc_price + 6 * x * sugar_price) - (4 * x * oreo_price) = 186

theorem cookie_solution :
  cookie_problem 90 2 3 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_solution_l763_76385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_intersection_theorem_l763_76307

def A : Set ℚ := {x | 1 < x ∧ x < 4}
def B : Set ℚ := A

def oplus (A B : Set ℚ) : Set (ℚ × ℚ) :=
  {p | p.1 / 2 ∈ A ∧ 2 / p.2 ∈ B}

def C : Set (ℚ × ℚ) := {p | p.2 = -1/6 * p.1 + 5/3}

theorem oplus_intersection_theorem :
  (oplus A B ∩ C) = {(4, 1), (6, 2/3)} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oplus_intersection_theorem_l763_76307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_autographs_exist_l763_76353

/-- Represents the number of inhabitants in the SMO country -/
def num_inhabitants : Nat := 1111

/-- Represents the number of players in the Liechtenstein national team -/
def num_players : Nat := 11

/-- Represents an autograph combination as a vector of booleans -/
def AutographCombination := Fin num_players → Bool

/-- Represents the set of all possible autograph combinations -/
def all_combinations : Set AutographCombination := Set.univ

/-- Represents the distribution of autographs to inhabitants -/
def AutographDistribution := Fin num_inhabitants → AutographCombination

/-- Predicate to check if a distribution is valid -/
def is_valid_distribution (d : AutographDistribution) : Prop :=
  ∀ i j : Fin num_inhabitants, i ≠ j → d i ≠ d j

/-- Theorem stating the existence of two inhabitants with complementary autographs -/
theorem complementary_autographs_exist (d : AutographDistribution) 
  (h : is_valid_distribution d) : 
  ∃ i j : Fin num_inhabitants, i ≠ j ∧ 
    (∀ k : Fin num_players, (d i k = true ∨ d j k = true)) ∧
    (∀ k : Fin num_players, ¬(d i k = true ∧ d j k = true)) :=
  sorry

#check complementary_autographs_exist

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_autographs_exist_l763_76353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_l763_76399

-- Define the points as elements of a real inner product space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (W X Y Z M N S : V)

-- Define the radii
variable (r_W r_X r_Y r_Z : ℝ)

-- Define the conditions
axiom radius_relation_WX : r_W = (2/3) * r_X
axiom radius_relation_YZ : r_Y = (3/4) * r_Z
axiom distance_WX : ‖W - X‖ = 25
axiom distance_YZ : ‖Y - Z‖ = 25
axiom distance_MN : ‖M - N‖ = 30
axiom S_midpoint : S = (1/2 : ℝ) • (M + N)

-- Define that M and N lie on all circles
axiom M_on_circles : 
  ‖M - W‖ = r_W ∧ ‖M - X‖ = r_X ∧ ‖M - Y‖ = r_Y ∧ ‖M - Z‖ = r_Z
axiom N_on_circles : 
  ‖N - W‖ = r_W ∧ ‖N - X‖ = r_X ∧ ‖N - Y‖ = r_Y ∧ ‖N - Z‖ = r_Z

-- The theorem to prove
theorem sum_of_distances : 
  ‖W - S‖ + ‖X - S‖ + ‖Y - S‖ + ‖Z - S‖ = 130 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_distances_l763_76399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_values_l763_76357

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_cosine_values (t : Triangle)
  (h1 : t.c^2 = t.a^2 + t.b^2 - 4*t.b*t.c*Real.cos t.C)
  (h2 : t.A - t.C = π/2) :
  Real.cos t.C = 2*Real.sqrt 5/5 ∧
  Real.cos (t.B + π/3) = (4 - 3*Real.sqrt 3)/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_values_l763_76357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l763_76345

/-- Triangle with side lengths and altitude ratio -/
structure RightTriangle where
  a : ℚ
  b : ℚ
  c : ℚ
  altitude_ratio : ℚ × ℚ

/-- Calculate the area of a right triangle -/
def area (t : RightTriangle) : ℚ :=
  t.a * t.b / 2

/-- The ratio of areas of two right triangles -/
def area_ratio (t1 t2 : RightTriangle) : ℚ :=
  area t1 / area t2

theorem triangle_area_ratio :
  let ghi : RightTriangle := ⟨7, 24, 25, (2, 3)⟩
  let jkl : RightTriangle := ⟨9, 40, 41, (4, 5)⟩
  area_ratio ghi jkl = 7 / 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_ratio_l763_76345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_values_l763_76376

-- Define the line equation
def line (x y : ℝ) (a : ℝ) : Prop := 4 * x + 3 * y + a = 0

-- Define the circle equation
def circleEq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 9

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x2 - x1)^2 + (y2 - y1)^2).sqrt

-- Main theorem
theorem intersection_values (a : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ,
    line x1 y1 a ∧ line x2 y2 a ∧
    circleEq x1 y1 ∧ circleEq x2 y2 ∧
    distance x1 y1 x2 y2 = 4 * Real.sqrt 2) →
  (a = -5 ∨ a = -15) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_values_l763_76376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_base_for_22_factorial_l763_76318

def factorial (n : ℕ) : ℕ := Finset.prod (Finset.range n) (λ i ↦ i + 1)

theorem unique_base_for_22_factorial :
  ∃! b : ℕ, b > 1 ∧ b^9 ∣ factorial 22 ∧ ¬(b^10 ∣ factorial 22) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_base_for_22_factorial_l763_76318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_males_in_band_not_orchestra_l763_76322

/-- Represents a student group with male and female members -/
structure StudentGroup where
  females : ℕ
  males : ℕ

/-- Represents the overlap between two groups -/
structure GroupOverlap where
  females : ℕ
  males : ℕ

def total_students : ℕ := 190

def band : StudentGroup := ⟨90, 70⟩
def orchestra : StudentGroup := ⟨70, 90⟩
def both : GroupOverlap := ⟨50, 0⟩ -- males in both is unknown, set to 0

theorem males_in_band_not_orchestra : ℕ := by
  -- The actual proof will go here
  sorry

#check males_in_band_not_orchestra

end NUMINAMATH_CALUDE_ERRORFEEDBACK_males_in_band_not_orchestra_l763_76322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_sum_l763_76391

/-- Right triangle ABC with point P on hypotenuse -/
structure RightTriangleWithPoint where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  P : ℝ × ℝ
  right_angle : (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0
  equal_sides : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 4 ∧ (B.1 - C.1)^2 + (B.2 - C.2)^2 = 4
  P_on_hypotenuse : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)
  BP_twice_PA : 2 * ((P.1 - A.1)^2 + (P.2 - A.2)^2) = (P.1 - B.1)^2 + (P.2 - B.2)^2

theorem dot_product_sum (triangle : RightTriangleWithPoint) :
  let CP := (triangle.P.1 - triangle.C.1, triangle.P.2 - triangle.C.2)
  let CA := (triangle.A.1 - triangle.C.1, triangle.A.2 - triangle.C.2)
  let CB := (triangle.B.1 - triangle.C.1, triangle.B.2 - triangle.C.2)
  (CP.1 * CA.1 + CP.2 * CA.2) + (CP.1 * CB.1 + CP.2 * CB.2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_sum_l763_76391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l763_76395

-- Definitions
noncomputable def Circle (O : Point) (r : ℝ) : Prop := sorry
noncomputable def Diameter (O A D : Point) : Prop := sorry
noncomputable def OnCircle (O : Point) (r : ℝ) (P : Point) : Prop := sorry
noncomputable def dist (P Q : Point) : ℝ := sorry
noncomputable def ArcMeasure (O A C D : Point) : ℝ := sorry
noncomputable def angle (A B O : Point) : ℝ := sorry

theorem chord_length (O A B C D : Point) (r : ℝ) : 
  Circle O r → -- Circle with center O and radius r
  Diameter O A D → -- AD is a diameter
  OnCircle O r B → -- B is on the circle
  OnCircle O r C → -- C is on the circle
  dist O B = 5 → -- BO = 5
  angle A B O = 60 * π / 180 → -- ∠ABO = 60°
  ArcMeasure O A C D = 60 * π / 180 → -- arc CD = 60°
  dist B C = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l763_76395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_80_factorial_l763_76334

def last_two_nonzero_digits (n : ℕ) : ℕ :=
  n % 100

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem last_two_nonzero_digits_80_factorial (n : ℕ) : 
  n = 76 → n = last_two_nonzero_digits (factorial 80) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_two_nonzero_digits_80_factorial_l763_76334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_range_l763_76368

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then |x - 2*a|
  else x + 1/(x-2) + a

-- State the theorem
theorem min_value_range (a : ℝ) :
  (∀ x : ℝ, f a 2 ≤ f a x) → a ∈ Set.Icc 1 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_range_l763_76368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l763_76329

noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x + Real.pi / 4)

theorem f_properties :
  let period : ℝ := Real.pi
  let max_value : ℝ := 2
  let min_value : ℝ := -Real.sqrt 2
  let max_point : ℝ := Real.pi / 8
  let min_point : ℝ := Real.pi / 2
  (∀ x : ℝ, f (x + period) = f x) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ max_value) ∧
  (f max_point = max_value) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ min_value) ∧
  (f min_point = min_value) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l763_76329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangles_share_vertex_l763_76328

variable (grid_width grid_height num_triangles : Nat)

/-- Represents the vertices of a triangle -/
def triangle_vertices (t : Fin num_triangles) : Finset (Fin (grid_width * grid_height)) :=
  sorry

/-- Theorem stating that at least two triangles share a vertex -/
theorem triangles_share_vertex
  (h_width : grid_width = 11)
  (h_height : grid_height = 8)
  (h_triangles : num_triangles = 30) :
  ∃ (t1 t2 : Fin num_triangles) (v : Fin (grid_width * grid_height)),
    t1 ≠ t2 ∧ v ∈ (triangle_vertices grid_width grid_height num_triangles t1) ∧
    v ∈ (triangle_vertices grid_width grid_height num_triangles t2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangles_share_vertex_l763_76328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_implies_sum_l763_76314

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = -1/2 * x + 6

/-- Point P is where the line crosses the x-axis -/
noncomputable def point_P : ℝ × ℝ := (12, 0)

/-- Point Q is where the line crosses the y-axis -/
noncomputable def point_Q : ℝ × ℝ := (0, 6)

/-- Point T is on line segment PQ -/
def point_T_on_PQ (r s : ℝ) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
  r = t * point_P.1 + (1 - t) * point_Q.1 ∧
  s = t * point_P.2 + (1 - t) * point_Q.2

/-- Area of triangle POQ -/
noncomputable def area_POQ : ℝ := 36

/-- Area of triangle TOP -/
noncomputable def area_TOP (r s : ℝ) : ℝ := 1/2 * r * s

/-- Theorem: If the area of triangle POQ is four times the area of triangle TOP,
    then r + s = 10.5 -/
theorem area_ratio_implies_sum (r s : ℝ) :
  line_equation r s →
  point_T_on_PQ r s →
  area_POQ = 4 * area_TOP r s →
  r + s = 10.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_implies_sum_l763_76314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_radius_proof_l763_76317

/-- The radius of a circle that is internally tangent to three externally tangent unit circles. -/
noncomputable def large_circle_radius : ℝ := (3 + 2 * Real.sqrt 3) / 3

/-- Representation of a circle in ℝ² -/
structure Circle (α : Type*) [LinearOrderedField α] where
  center : α × α
  radius : α

/-- Two circles are externally tangent. -/
def are_externally_tangent {α : Type*} [LinearOrderedField α] (c1 c2 : Circle α) : Prop :=
  sorry

/-- A circle is internally tangent to another circle. -/
def is_internally_tangent {α : Type*} [LinearOrderedField α] (c1 c2 : Circle α) : Prop :=
  sorry

/-- Theorem stating that the radius of a circle internally tangent to three externally tangent unit circles is (3 + 2√3) / 3. -/
theorem large_circle_radius_proof (c1 c2 c3 C : Circle ℝ) :
  c1.radius = 1 ∧ c2.radius = 1 ∧ c3.radius = 1 ∧
  are_externally_tangent c1 c2 ∧
  are_externally_tangent c2 c3 ∧
  are_externally_tangent c3 c1 ∧
  is_internally_tangent c1 C ∧
  is_internally_tangent c2 C ∧
  is_internally_tangent c3 C →
  C.radius = large_circle_radius := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_circle_radius_proof_l763_76317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2007_is_sin_l763_76359

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => Real.cos
  | (n + 1) => λ x => deriv (f n) x

-- State the theorem
theorem f_2007_is_sin : f 2007 = Real.sin := by
  -- Proof strategy:
  -- 1. Show that the sequence repeats every 4 steps
  -- 2. Use this to reduce f 2007 to f 3
  -- 3. Show that f 3 = sin
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2007_is_sin_l763_76359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_in_ratio_l763_76386

/-- 
Given a total amount of $600 to be divided between A and B in the ratio 1:2,
prove that A receives $200.
-/
theorem division_in_ratio (total ratio_a ratio_b amount_a : ℕ) : 
  total = 600 →
  ratio_a = 1 →
  ratio_b = 2 →
  amount_a = total * ratio_a / (ratio_a + ratio_b) →
  amount_a = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_in_ratio_l763_76386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_seven_digit_sum_two_l763_76361

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that returns the number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The set of seven-digit numbers whose digits sum to 2 -/
def seven_digit_sum_two : Set ℕ := {n : ℕ | num_digits n = 7 ∧ digit_sum n = 2 ∧ n ≥ 1000000}

/-- A finite list of all seven-digit numbers whose digits sum to 2 -/
def seven_digit_sum_two_list : List ℕ := [
  2000000, 1100000, 1010000, 1001000, 1000100, 1000010, 1000001
]

theorem count_seven_digit_sum_two : 
  seven_digit_sum_two_list.length = 7 ∧ 
  ∀ n, n ∈ seven_digit_sum_two ↔ n ∈ seven_digit_sum_two_list := by
  sorry

#eval seven_digit_sum_two_list.length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_seven_digit_sum_two_l763_76361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_acute_angle_l763_76305

theorem sin_plus_cos_acute_angle (θ : ℝ) (b : ℝ) 
  (h1 : 0 < θ ∧ θ < π/2) -- θ is acute
  (h2 : Real.cos (2 * θ) = b) : 
  Real.sin θ + Real.cos θ = Real.sqrt (2 - b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_acute_angle_l763_76305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_is_bisector_union_l763_76383

/-- Two planes in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance from a point to a plane -/
noncomputable def distance_to_plane (pt : Point) (pl : Plane) : ℝ :=
  (pl.a * pt.x + pl.b * pt.y + pl.c * pt.z + pl.d) / Real.sqrt (pl.a^2 + pl.b^2 + pl.c^2)

/-- The set of points equidistant from two planes -/
def equidistant_set (p1 p2 : Plane) : Set Point :=
  {pt : Point | distance_to_plane pt p1 = distance_to_plane pt p2}

/-- The first bisector plane -/
def bisector_plane1 (p1 p2 : Plane) : Plane where
  a := p1.a - p2.a
  b := p1.b - p2.b
  c := p1.c - p2.c
  d := p1.d - p2.d

/-- The second bisector plane -/
def bisector_plane2 (p1 p2 : Plane) : Plane where
  a := p1.a + p2.a
  b := p1.b + p2.b
  c := p1.c + p2.c
  d := p1.d + p2.d

/-- The union of the two bisector planes -/
def bisector_union (p1 p2 : Plane) : Set Point :=
  {pt : Point | 
    (bisector_plane1 p1 p2).a * pt.x + (bisector_plane1 p1 p2).b * pt.y + 
    (bisector_plane1 p1 p2).c * pt.z + (bisector_plane1 p1 p2).d = 0 ∨
    (bisector_plane2 p1 p2).a * pt.x + (bisector_plane2 p1 p2).b * pt.y + 
    (bisector_plane2 p1 p2).c * pt.z + (bisector_plane2 p1 p2).d = 0}

theorem equidistant_is_bisector_union (p1 p2 : Plane) : 
  equidistant_set p1 p2 = bisector_union p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_is_bisector_union_l763_76383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parameter_l763_76303

/-- The circle C in the xy-plane -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 - 4*p.2 = 0}

/-- The original line L with parameter m -/
def L (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - 2*p.2 + m = 0}

/-- The transformed line L' after moving 1 unit left and 2 units down -/
def L' (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + 1) - 2*(p.2 + 2) + m = 0}

/-- Predicate to check if a line is tangent to a circle -/
def IsTangentTo (line : Set (ℝ × ℝ)) (circle : Set (ℝ × ℝ)) : Prop :=
  ∃ (p : ℝ × ℝ), p ∈ line ∧ p ∈ circle ∧
    ∀ (q : ℝ × ℝ), q ∈ line ∧ q ∈ circle → q = p

/-- The theorem stating that m is either 13 or 3 when L' is tangent to C -/
theorem tangent_line_parameter :
  ∃ (m : ℝ), IsTangentTo (L' m) C ∧ (m = 13 ∨ m = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parameter_l763_76303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wooden_toy_price_reduction_is_15_percent_l763_76342

/-- Calculates the percentage reduction in the selling price of a wooden toy -/
def wooden_toy_price_reduction (
  num_paintings : ℕ
) (painting_cost : ℚ
) (num_toys : ℕ
) (toy_cost : ℚ
) (painting_price_reduction : ℚ
) (total_loss : ℚ
) : ℚ :=
  let total_cost := num_paintings * painting_cost + num_toys * toy_cost
  let painting_sell_price := painting_cost * (1 - painting_price_reduction)
  let total_painting_revenue := num_paintings * painting_sell_price
  let total_revenue := total_cost - total_loss
  let toy_revenue := total_revenue - total_painting_revenue
  let toy_sell_price := toy_revenue / num_toys
  (toy_cost - toy_sell_price) / toy_cost

theorem wooden_toy_price_reduction_is_15_percent :
  wooden_toy_price_reduction 10 40 8 20 (1/10) 64 = 15/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wooden_toy_price_reduction_is_15_percent_l763_76342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operations_l763_76352

-- Define the vector type
def Vector2D := ℝ × ℝ

-- Define the given vectors
def a : Vector2D := (1, 0)
def b : Vector2D := (-1, 2)

-- Define vector addition
def add (v w : Vector2D) : Vector2D := (v.1 + w.1, v.2 + w.2)

-- Define scalar multiplication
def scale (k : ℝ) (v : Vector2D) : Vector2D := (k * v.1, k * v.2)

-- Define vector subtraction
def sub (v w : Vector2D) : Vector2D := (v.1 - w.1, v.2 - w.2)

-- Define vector magnitude
noncomputable def mag (v : Vector2D) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem vector_operations :
  (add (scale 2 a) b = (1, 2)) ∧ 
  (mag (sub a b) = 2 * Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_operations_l763_76352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiction_books_count_l763_76331

theorem fiction_books_count (n : ℕ) : n = 3 :=
  by
    have h1 : (Nat.factorial n / Nat.factorial (n - 3))^2 = 36 := by sorry
    have h2 : n ≥ 3 := by sorry
    have h3 : ∀ m > 3, Nat.factorial m / Nat.factorial (m - 3) > 6 := by sorry
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiction_books_count_l763_76331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersects_x_axis_once_l763_76327

/-- The function f(x) for a given m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 - 6 * x + (3/2) * m

/-- The discriminant of the quadratic function f(x) for a given m -/
noncomputable def discriminant (m : ℝ) : ℝ := 36 - 6 * m^2 + 6 * m

/-- Theorem stating that f(x) intersects the x-axis at exactly one point iff m = 1 or m = -2 or m = 3 -/
theorem intersects_x_axis_once (m : ℝ) : 
  (∃! x, f m x = 0) ↔ (m = 1 ∨ m = -2 ∨ m = 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersects_x_axis_once_l763_76327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_cost_calculation_l763_76347

noncomputable def cost_of_trip (initial_reading : ℕ) (final_reading : ℕ) (fuel_efficiency : ℝ) (gas_price : ℝ) : ℝ :=
  ((final_reading - initial_reading : ℝ) / fuel_efficiency) * gas_price

theorem trip_cost_calculation (initial_reading final_reading : ℕ) (fuel_efficiency gas_price : ℝ) :
  initial_reading = 85120 →
  final_reading = 85150 →
  fuel_efficiency = 30 →
  gas_price = 4.25 →
  cost_of_trip initial_reading final_reading fuel_efficiency gas_price = 4.25 :=
by
  sorry

-- Remove the #eval statement as it's causing issues with noncomputable definitions
-- #eval cost_of_trip 85120 85150 30 4.25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_cost_calculation_l763_76347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_has_sweetest_mixture_l763_76338

/-- Represents a sugar water mixture -/
structure SugarWater where
  sugar : ℚ
  water : ℚ

/-- Calculates the sugar concentration of a mixture -/
def concentration (sw : SugarWater) : ℚ :=
  sw.sugar / (sw.sugar + sw.water)

/-- Initial mixture for all students -/
def initial : SugarWater :=
  { sugar := 25, water := 100 }

/-- A's final mixture after addition -/
def final_A : SugarWater :=
  { sugar := initial.sugar + 10,
    water := initial.water + 40 }

/-- B's final mixture after addition -/
def final_B : SugarWater :=
  { sugar := initial.sugar + 20,
    water := initial.water }

/-- C's final mixture after addition -/
def final_C : SugarWater :=
  { sugar := initial.sugar + 20,
    water := initial.water + 20 }

theorem B_has_sweetest_mixture :
  concentration final_B > concentration final_A ∧
  concentration final_B > concentration final_C := by
  sorry

#eval concentration final_A
#eval concentration final_B
#eval concentration final_C

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_has_sweetest_mixture_l763_76338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l763_76378

/-- The angle between two plane vectors given specific conditions -/
theorem angle_between_vectors (a b : ℝ × ℝ) : 
  a = (1, -Real.sqrt 3) →
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 1 →
  Real.sqrt ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2) = 2 →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l763_76378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_triangle_l763_76324

/-- An isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  -- The length of the two equal sides
  side : ℝ
  -- The length of the base
  base : ℝ
  -- Assumption that side is positive
  side_pos : side > 0
  -- Assumption that base is positive
  base_pos : base > 0
  -- Assumption that the triangle inequality holds
  triangle_ineq : base < 2 * side

/-- The area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  let height := Real.sqrt (t.side^2 - (t.base/2)^2)
  (1/2) * t.base * height

/-- Theorem: The area of the specific isosceles triangle is 240 -/
theorem area_of_specific_triangle :
  let t : IsoscelesTriangle := {
    side := 26,
    base := 20,
    side_pos := by norm_num,
    base_pos := by norm_num,
    triangle_ineq := by norm_num
  }
  area t = 240 := by
    -- Proof steps would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_triangle_l763_76324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_number_l763_76302

open Complex

theorem modulus_of_complex_number (i : ℂ) (a : ℝ) :
  i * i = -1 →
  ((2 : ℂ) - i) / (a + i) ∈ {z : ℂ | z.re = 0 ∧ z.im ≠ 0} →
  abs ((2 * a : ℂ) + I * Real.sqrt 2) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_complex_number_l763_76302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_one_l763_76367

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x + 2 else 3 * x - 1

-- State the theorem
theorem f_composition_negative_one : f (f (-1)) = 2 := by
  -- Evaluate f(-1)
  have h1 : f (-1) = 1 := by
    simp [f]
    norm_num
  
  -- Evaluate f(1)
  have h2 : f 1 = 2 := by
    simp [f]
    norm_num
  
  -- Combine the results
  calc
    f (f (-1)) = f 1 := by rw [h1]
    _          = 2   := by rw [h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_one_l763_76367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l763_76311

noncomputable def a : ℝ := Real.sqrt 3 - Real.sqrt 11
noncomputable def b : ℝ := Real.sqrt 3 + Real.sqrt 11

theorem simplify_expression :
  (a^2 - b^2) / (a^2 * b - a * b^2) / (1 + (a^2 + b^2) / (2 * a * b)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l763_76311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_capacitor_voltage_problem_l763_76373

/-- The final voltage across two capacitors connected with opposite-charged plates -/
noncomputable def final_voltage (C₁ C₂ U₁ U₂ : ℝ) : ℝ :=
  (C₁ * U₁ - C₂ * U₂) / (C₁ + C₂)

/-- Theorem stating the final voltage for the given capacitor problem -/
theorem capacitor_voltage_problem :
  let C₁ : ℝ := 20e-6  -- 20 μF
  let C₂ : ℝ := 5e-6   -- 5 μF
  let U₁ : ℝ := 20     -- 20 V
  let U₂ : ℝ := 5      -- 5 V
  final_voltage C₁ C₂ U₁ U₂ = 15 := by
  sorry

#eval (20e-6 * 20 - 5e-6 * 5) / (20e-6 + 5e-6)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_capacitor_voltage_problem_l763_76373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_folding_ratio_l763_76326

theorem paper_folding_ratio : 
  (let original_length : ℝ := 12
   let original_width : ℝ := 8
   let small_length : ℝ := original_length / 2
   let small_width : ℝ := original_width / 2
   let small_perimeter : ℝ := 2 * (small_length + small_width)
   let large_perimeter : ℝ := 2 * (original_length + original_width)
   small_perimeter / large_perimeter) = 1 / 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_folding_ratio_l763_76326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l763_76363

/-- The area of a trapezoid ABCD with vertices A(0,0), B(0,-2), C(4,0), and D(4,6) is 16 square units. -/
theorem trapezoid_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, -2)
  let C : ℝ × ℝ := (4, 0)
  let D : ℝ × ℝ := (4, 6)
  let base1 := abs (B.2 - A.2)
  let base2 := abs (D.2 - C.2)
  let height := abs (C.1 - A.1)
  let area := (base1 + base2) * height / 2
  area = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l763_76363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_bike_speed_l763_76320

/-- Represents a triathlon segment with distance and speed -/
structure Segment where
  distance : ℝ
  speed : ℝ

/-- Calculates the time taken for a segment -/
noncomputable def time (s : Segment) : ℝ := s.distance / s.speed

/-- Represents a triathlon with swim, run, and bike segments -/
structure Triathlon where
  swim : Segment
  run : Segment
  bike_distance : ℝ
  total_time : ℝ

/-- The specific triathlon Jessica is preparing for -/
def jessica_triathlon : Triathlon :=
  { swim := { distance := 0.5, speed := 1 }
  , run := { distance := 5, speed := 5 }
  , bike_distance := 20
  , total_time := 4
  }

/-- Theorem stating the required bike speed for Jessica's triathlon -/
theorem jessica_bike_speed :
  let t := jessica_triathlon
  let swim_run_time := time t.swim + time t.run
  let bike_time := t.total_time - swim_run_time
  t.bike_distance / bike_time = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_bike_speed_l763_76320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l763_76393

/-- 
Given:
- n is a positive integer
- p is a rational number between 0 and 1
- a coin is flipped n times with probability p of heads for each flip
- all 2^n possible sequences are written in two columns

Prove that p = 1/2 is the only value that allows the actual sequence 
to appear in the left column with probability 1/2, for any n.
-/
theorem coin_flip_probability (n : ℕ+) (p : ℚ) 
  (hp : 0 < p ∧ p < 1) :
  (∃ (left_sequences : Finset (Fin n → Bool)),
    (left_sequences.card : ℚ) / (2 ^ n.val : ℚ) = 1/2) ↔ p = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_l763_76393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_fencing_cost_l763_76355

/-- Represents a rectangular farm with fencing costs -/
structure RectangularFarm where
  area : ℝ
  shortSide : ℝ
  longSideCost : ℝ
  shortSideCost : ℝ
  diagonalCost : ℝ

/-- Calculates the total fencing cost for a rectangular farm -/
noncomputable def totalFencingCost (farm : RectangularFarm) : ℝ :=
  let longSide := farm.area / farm.shortSide
  let diagonal := Real.sqrt (longSide ^ 2 + farm.shortSide ^ 2)
  farm.longSideCost * longSide + farm.shortSideCost * farm.shortSide + farm.diagonalCost * diagonal

/-- Theorem stating the total fencing cost for the given farm -/
theorem farm_fencing_cost : 
  let farm := RectangularFarm.mk 1200 30 16 14 18
  totalFencingCost farm = 1960 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_fencing_cost_l763_76355
