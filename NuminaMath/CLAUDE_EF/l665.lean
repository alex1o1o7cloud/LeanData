import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_foci_vertices_ratio_l665_66513

noncomputable section

/-- Parabola P' defined by y = 2x^2 -/
def P' : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 2 * p.1^2}

/-- Vertex of P' -/
def V1' : ℝ × ℝ := (0, 0)

/-- Focus of P' -/
def F1' : ℝ × ℝ := (0, 1/8)

/-- Condition for points C and D on P' forming a right angle with V1' -/
def right_angle_condition (C D : ℝ × ℝ) : Prop :=
  C ∈ P' ∧ D ∈ P' ∧ (C.1 * D.1 = -1/4)

/-- Midpoint of two points -/
def mid (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

/-- Locus Q' of midpoints of CD -/
def Q' : Set (ℝ × ℝ) := {q : ℝ × ℝ | ∃ C D, right_angle_condition C D ∧ q = mid C D}

/-- Vertex of Q' -/
def V2' : ℝ × ℝ := (0, 1/2)

/-- Focus of Q' -/
def F2' : ℝ × ℝ := (0, 3/4)

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_foci_vertices_ratio :
  distance F1' F2' / distance V1' V2' = 3/2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_foci_vertices_ratio_l665_66513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_exists_l665_66568

noncomputable def f (x : ℝ) : ℝ := -3 * Real.cos (Real.pi * x / 2)

def is_solution (x : ℝ) : Prop :=
  -3 ≤ x ∧ x ≤ 3 ∧ f (f (f x)) = f x

theorem solution_count_exists : ∃ n : ℕ, n > 0 ∧ 
  (∃ s : Finset ℝ, (∀ x ∈ s, is_solution x) ∧ s.card = n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_exists_l665_66568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l665_66549

theorem max_value_sqrt_sum :
  ∀ x : ℝ, x ∈ Set.Icc (-16) 16 →
  (Real.sqrt (16 + x) + Real.sqrt (16 - x)) ≤ 8 ∧
  ∃ x₀ ∈ Set.Icc (-16) 16, Real.sqrt (16 + x₀) + Real.sqrt (16 - x₀) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l665_66549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l665_66505

/-- The time taken for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that a 110-meter train traveling at 45 km/h takes 30 seconds to cross a 265-meter bridge -/
theorem train_crossing_bridge_time : 
  train_crossing_time 110 45 265 = 30 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l665_66505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_inequality_l665_66587

open Real

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := log x - m * x

-- State the theorem
theorem roots_sum_inequality {m : ℝ} {x₁ x₂ : ℝ} 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂)
  (h₄ : f m x₁ = 0) (h₅ : f m x₂ = 0) :
  m * (x₁ + x₂) > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_inequality_l665_66587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tinas_earnings_l665_66540

/-- Calculates the total earnings for a worker given their hourly rate, 
    overtime threshold, days worked, and hours worked per day. -/
noncomputable def calculate_earnings (hourly_rate : ℚ) (overtime_threshold : ℚ) 
                       (days_worked : ℕ) (hours_per_day : ℚ) : ℚ :=
  let regular_hours := min overtime_threshold hours_per_day
  let overtime_hours := max 0 (hours_per_day - overtime_threshold)
  let overtime_rate := hourly_rate * (3/2)
  let daily_earnings := regular_hours * hourly_rate + overtime_hours * overtime_rate
  daily_earnings * days_worked

/-- Theorem stating that Tina's earnings for the week are $990.00 -/
theorem tinas_earnings : 
  calculate_earnings 18 8 5 10 = 990 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tinas_earnings_l665_66540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_exponent_reciprocal_l665_66584

theorem negative_exponent_reciprocal (x : ℝ) (h : x ≠ 0) : x^(-2 : ℤ) = 1 / x^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_exponent_reciprocal_l665_66584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_fraction_l665_66585

-- Define the expressions
noncomputable def expr_A (a b : ℝ) : ℝ := 1 / (a - b)
noncomputable def expr_B (a b : ℝ) : ℝ := (b - a) / (b^2 - a^2)
noncomputable def expr_C (a b : ℝ) : ℝ := 2 / (6 * a * b)
noncomputable def expr_D (a b : ℝ) : ℝ := (a * b - a^2) / a

-- Define what it means for a fraction to be simplified
def is_simplified (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b, a ≠ b → (∀ k : ℝ, k ≠ 0 → f a b ≠ k * f a b)

-- Theorem statement
theorem simplified_fraction :
  is_simplified expr_A ∧
  ¬is_simplified expr_B ∧
  ¬is_simplified expr_C ∧
  ¬is_simplified expr_D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplified_fraction_l665_66585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l665_66554

theorem trigonometric_identity (β : ℝ) :
  3.413 * (Real.sin (2 * β))^3 * Real.cos (6 * β) + (Real.cos (2 * β))^3 * Real.sin (6 * β) =
  3.413 * (Real.sin (2 * β))^3 * Real.cos (6 * β) + (Real.cos (2 * β))^3 * Real.sin (6 * β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l665_66554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l665_66545

/-- Given a geometric sequence where the fourth term is 6! and the seventh term is 7!,
    prove that the first term is 720/7 -/
theorem geometric_sequence_first_term :
  ∀ (a : ℝ) (r : ℝ),
    (a * r^3 = Nat.factorial 6) →
    (a * r^6 = Nat.factorial 7) →
    a = 720 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l665_66545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l665_66547

-- Define the sets A and B
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem intersection_A_B : A ∩ B = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l665_66547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_sheet_diagonal_l665_66574

-- Define the cookie sheet dimensions
def width : ℝ := 15.2
def length : ℝ := 3.7

-- Define the conversion factor
def inches_to_cm : ℝ := 2.54

-- Define the function to calculate the diagonal using Pythagorean theorem
noncomputable def diagonal (w l : ℝ) : ℝ := Real.sqrt (w^2 + l^2)

-- Define the function to convert inches to centimeters
def to_cm (inches : ℝ) : ℝ := inches * inches_to_cm

-- Define the function to round to the nearest tenth
noncomputable def round_to_tenth (x : ℝ) : ℝ := ⌊x * 10 + 0.5⌋ / 10

-- Theorem statement
theorem cookie_sheet_diagonal :
  round_to_tenth (to_cm (diagonal width length)) = 39.7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_sheet_diagonal_l665_66574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_denominator_1_common_denominator_2_l665_66519

-- Define variables
variable (x y c a b : ℚ)

-- Define the fractions for the first set
def frac1_1 (x y : ℚ) : ℚ := x / (3 * y)
def frac1_2 (x y : ℚ) : ℚ := (3 * x) / (2 * y^2)

-- Define the fractions for the second set
def frac2_1 (c a b : ℚ) : ℚ := (6 * c) / (a^2 * b)
def frac2_2 (c a b : ℚ) : ℚ := c / (3 * a * b^2)

-- Theorem for the first set of fractions
theorem common_denominator_1 (x y : ℚ) (h : y ≠ 0) : ∃ (k : ℚ), k ≠ 0 ∧ 
  (∃ (m n : ℚ), frac1_1 x y = m / k ∧ frac1_2 x y = n / k) ∧ 
  k = 6 * y^2 := by
  sorry

-- Theorem for the second set of fractions
theorem common_denominator_2 (c a b : ℚ) (h1 : a ≠ 0) (h2 : b ≠ 0) : ∃ (k : ℚ), k ≠ 0 ∧ 
  (∃ (m n : ℚ), frac2_1 c a b = m / k ∧ frac2_2 c a b = n / k) ∧ 
  k = 3 * a^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_denominator_1_common_denominator_2_l665_66519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spearman_correlation_approx_l665_66501

-- Define the rankings for inspectors A and B
noncomputable def rankings_A : List ℝ := [1, 2, 4, 4, 4, 7.5, 7.5, 7.5, 7.5]
noncomputable def rankings_B : List ℝ := [2, 1, 4, 3, 5, 6.5, 6.5, 8, 9]

-- Define the number of samples
def n : Nat := 9

-- Define the Spearman rank correlation coefficient formula
noncomputable def spearman_correlation (x y : List ℝ) : ℝ :=
  1 - (6 * (List.sum (List.zipWith (fun a b => (a - b)^2) x y))) / (n * (n^2 - 1))

-- Theorem statement
theorem spearman_correlation_approx :
  ∃ ε > 0, ε < 0.01 ∧ |spearman_correlation rankings_A rankings_B - 0.93| < ε := by
  sorry

-- This is now a noncomputable expression
noncomputable example : ℝ := spearman_correlation rankings_A rankings_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spearman_correlation_approx_l665_66501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l665_66522

theorem max_value_expression (a b c d : ℕ) : 
  a ∈ ({0, 1, 3, 4} : Set ℕ) → 
  b ∈ ({0, 1, 3, 4} : Set ℕ) → 
  c ∈ ({0, 1, 3, 4} : Set ℕ) → 
  d ∈ ({0, 1, 3, 4} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d → 
  (∀ a' b' c' d' : ℕ, 
    a' ∈ ({0, 1, 3, 4} : Set ℕ) → 
    b' ∈ ({0, 1, 3, 4} : Set ℕ) → 
    c' ∈ ({0, 1, 3, 4} : Set ℕ) → 
    d' ∈ ({0, 1, 3, 4} : Set ℕ) → 
    a' ≠ b' → a' ≠ c' → a' ≠ d' → b' ≠ c' → b' ≠ d' → c' ≠ d' → 
    d * b^c - a ≥ d' * b'^c' - a') :=
by
  sorry

#check max_value_expression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l665_66522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagonal_cube_removed_volume_l665_66503

/-- The edge length of the original cube -/
noncomputable def cube_edge : ℝ := 2

/-- The volume of tetrahedra removed when slicing corners off a cube to make octagonal faces -/
noncomputable def removed_volume (edge : ℝ) : ℝ :=
  let y := edge * (Real.sqrt 2 - 1)
  let height := edge - y
  let base_area := (1 / 2) * y^2
  8 * (1 / 3) * base_area * height

theorem octagonal_cube_removed_volume :
  removed_volume cube_edge = (8 / 3) * (5 - 4 * Real.sqrt 2) * (4 - 2 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagonal_cube_removed_volume_l665_66503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l665_66569

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Add a case for 0 to make the function total
  | 1 => 2
  | n + 2 => 2 * sequence_a (n + 1) / (sequence_a (n + 1) + 2)

theorem sequence_a_formula : ∀ n : ℕ, n > 0 → sequence_a n = 2 / n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l665_66569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_three_l665_66593

/-- A square with side length 2 -/
structure Square where
  side : ℚ
  is_square : side = 2

/-- An isosceles trapezoid with parallel sides CK and MN -/
structure IsoscelesTrapezoid where
  ck : ℚ
  mn : ℚ
  h : ℚ
  is_isosceles : mn = ck / 2
  ck_equals_square_side : ck = 2

/-- The area of an isosceles trapezoid -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℚ :=
  (t.ck + t.mn) * t.h / 2

/-- Main theorem: The area of the trapezoid CMNK is 3 square inches -/
theorem trapezoid_area_is_three 
  (s : Square) 
  (t : IsoscelesTrapezoid) 
  (h_square_area : s.side * s.side = 4) 
  (h_trapezoid_height : t.h = s.side) : 
  trapezoid_area t = 3 := by
  sorry

#check trapezoid_area_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_three_l665_66593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stick_swap_impossibility_l665_66517

def is_ngon (sides : List ℝ) : Prop :=
  sides.length ≥ 3 ∧ 
  ∀ i, i < sides.length → sides.get! i < (sides.sum - sides.get! i)

theorem stick_swap_impossibility (N : ℕ) (h : N ≥ 3) : 
  ∃ (blue red : List ℝ),
    blue.length = N ∧ 
    red.length = N ∧ 
    blue.sum = red.sum ∧
    is_ngon blue ∧ 
    is_ngon red ∧
    ∀ (b r : ℝ), b ∈ blue → r ∈ red → 
      ¬(is_ngon ((blue.filter (λ x => x ≠ b)) ++ [r]) ∧ 
        is_ngon ((red.filter (λ x => x ≠ r)) ++ [b])) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stick_swap_impossibility_l665_66517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_for_cubic_derivative_l665_66512

/-- Given a cubic function f(x) = x³ - (9/2)x² + 6x - 5, 
    this theorem states that the maximum value of m for which 
    f'(x) ≥ m holds for all x is -3/4. -/
theorem max_m_for_cubic_derivative (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - (9/2)*x^2 + 6*x - 5) :
  (∃ m : ℝ, (∀ x, deriv f x ≥ m) ∧ 
   (∀ m' : ℝ, (∀ x, deriv f x ≥ m') → m' ≤ m)) ∧ 
  (∀ m : ℝ, (∀ x, deriv f x ≥ m) → m ≤ -3/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_for_cubic_derivative_l665_66512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_derivative_at_two_l665_66521

noncomputable def f (x : ℝ) : ℝ := (1 - x) / x + Real.log x

theorem third_derivative_at_two :
  (deriv^[3] f) 2 = -(1/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_derivative_at_two_l665_66521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_rainbow_cycle_l665_66508

/-- A graph representing airports and flights -/
structure AirportGraph where
  n : ℕ
  vertices : Finset (Fin n)
  cycles : Finset (List (Fin n))
  h_n_ge_3 : n ≥ 3
  h_cycles_count : cycles.card = n
  h_cycles_odd : ∀ c ∈ cycles, Odd c.length ∧ c.length ≥ 3
  h_cycles_valid : ∀ c ∈ cycles, c.toFinset ⊆ vertices

/-- Definition of a rainbow cycle -/
def isRainbowCycle (g : AirportGraph) (cycle : List (Fin g.n)) : Prop :=
  Odd cycle.length ∧ 
  cycle.toFinset ⊆ g.vertices ∧
  ∀ (i j : Fin cycle.length), i ≠ j → 
    ∃ c ∈ g.cycles, (cycle[i], cycle[j]) ∈ List.zip c (c.rotate 1)

/-- The main theorem -/
theorem exists_rainbow_cycle (g : AirportGraph) : 
  ∃ cycle, isRainbowCycle g cycle := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_rainbow_cycle_l665_66508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l665_66563

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) : ℝ → ℝ → Prop := λ x y => x + a * y + 1 = 0
def l₂ (a : ℝ) : ℝ → ℝ → Prop := λ x y => (a - 1) * x + 2 * y + 2 * a = 0

-- Define the condition that the lines are parallel
def are_parallel (a : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ (1 : ℝ) / a = k * ((a - 1) / 2)

-- Define the distance between two lines
noncomputable def distance_between_lines (A B C₁ C₂ : ℝ) : ℝ :=
  abs (C₁ - C₂) / Real.sqrt (A^2 + B^2)

-- State the theorem
theorem parallel_lines_distance (a : ℝ) :
  are_parallel a →
  ∃ A B C₁ C₂ : ℝ, l₁ a = λ x y => A * x + B * y + C₁ = 0 ∧
                   l₂ a = λ x y => A * x + B * y + C₂ = 0 ∧
                   distance_between_lines A B C₁ C₂ = 3 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l665_66563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_businessman_income_distribution_l665_66570

/-- Businessman's income distribution problem -/
theorem businessman_income_distribution (I : ℝ) 
  (h1 : I > 0) 
  (h2 : 0.2 * I + 0.22 * I + 0.15 * I + 0.1 * I + 0.05 * I = 0.72 * I) 
  (h3 : 0.92 * (0.28 * I) = 25000) : 
  ‖I - 97049.68‖ < 0.01 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_businessman_income_distribution_l665_66570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_comparison_l665_66575

-- Define a function to convert degrees, minutes, and seconds to decimal degrees
noncomputable def to_decimal_degrees (degrees : ℕ) (minutes : ℕ) (seconds : ℕ) : ℝ :=
  (degrees : ℝ) + (minutes : ℝ) / 60 + (seconds : ℝ) / 3600

-- Define the angles
noncomputable def angle_A : ℝ := to_decimal_degrees 60 24 0
noncomputable def angle_B : ℝ := 60.24
noncomputable def angle_C : ℝ := to_decimal_degrees 60 14 24

-- State the theorem
theorem angle_comparison : angle_A > angle_B ∧ angle_B = angle_C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_comparison_l665_66575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extrema_of_f_l665_66592

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x / 8 + 2 / x

-- Define the interval
def I : Set ℝ := Set.Ioo (-5 : ℝ) 10

-- State the theorem
theorem extrema_of_f :
  ∃ (x_max x_min : ℝ),
    x_max ∈ I ∧ x_min ∈ I ∧
    x_max = -4 ∧ x_min = 4 ∧
    f x_max = -1 ∧ f x_min = 1 ∧
    (∀ x ∈ I, f x ≤ f x_max) ∧
    (∀ x ∈ I, f x ≥ f x_min) ∧
    (∀ x ∈ I, (f x = f x_max ∨ f x = f x_min) → (x = x_max ∨ x = x_min)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extrema_of_f_l665_66592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_game_cost_proof_l665_66546

/-- The cost of a board game given the following conditions:
  * A jump rope costs $7
  * A playground ball costs $4
  * Dalton has saved $6 from his allowance
  * Dalton's uncle gave him $13
  * Dalton needs $4 more to buy everything
-/
def board_game_cost : ℕ := 12

/-- Proof that the board game costs $12 -/
theorem board_game_cost_proof :
  let jump_rope_cost : ℕ := 7
  let ball_cost : ℕ := 4
  let savings : ℕ := 6
  let uncle_gift : ℕ := 13
  let additional_needed : ℕ := 4
  board_game_cost = 12 := by
  -- Unfold the definition of board_game_cost
  unfold board_game_cost
  -- The rest of the proof would go here
  sorry

#eval board_game_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_game_cost_proof_l665_66546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l665_66537

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the circle
def circleC (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem min_distance_sum :
  ∃ (min : ℝ), min = 4 ∧
  ∀ (m : ℝ × ℝ) (a : ℝ × ℝ),
    parabola m.1 m.2 → circleC a.1 a.2 →
    distance m a + distance m focus ≥ min :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l665_66537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_and_triangle_areas_l665_66543

/-- Represents a rectangle with two points on its sides -/
structure RectangleWithPoints where
  AB : ℝ
  AD : ℝ
  AN : ℝ
  NC : ℝ
  AM : ℝ
  MB : ℝ

/-- Calculate the area of the rectangle -/
noncomputable def rectangleArea (r : RectangleWithPoints) : ℝ :=
  r.AB * r.AD

/-- Calculate the area of the triangle MNC -/
noncomputable def triangleArea (r : RectangleWithPoints) : ℝ :=
  rectangleArea r - (r.AM * r.AN / 2) - (r.MB * r.AD / 2) - (r.AB * (r.AD - r.AN) / 2)

/-- Main theorem statement -/
theorem rectangle_and_triangle_areas 
  (r : RectangleWithPoints) 
  (h1 : r.AN = 7)
  (h2 : r.NC = 39)
  (h3 : r.AM = 12)
  (h4 : r.MB = 3)
  (h5 : r.AB = r.AM + r.MB)
  (h6 : r.AD = r.AN + r.NC) :
  rectangleArea r = 645 ∧ triangleArea r = 268.5 := by
  sorry

#check rectangle_and_triangle_areas

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_and_triangle_areas_l665_66543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_parallel_diagonal_l665_66529

-- Define a polygon
def Polygon (n : ℕ) := Fin (2*n) → ℝ × ℝ

-- Define convexity for a polygon
def is_convex (p : Polygon n) : Prop := sorry

-- Define a diagonal of a polygon
def is_diagonal {n : ℕ} (p : Polygon n) (i j : Fin (2*n)) : Prop :=
  i ≠ j ∧ (i.val + 1) % (2*n) ≠ j.val ∧ (j.val + 1) % (2*n) ≠ i.val

-- Define parallelism between two line segments
def are_parallel (a b c d : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem exists_non_parallel_diagonal {n : ℕ} (p : Polygon n) (h : is_convex p) :
  ∃ (i j : Fin (2*n)), is_diagonal p i j ∧
    ∀ (k : Fin (2*n)), ¬(are_parallel (p i) (p j) (p k) (p ⟨(k.val + 1) % (2*n), sorry⟩)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_parallel_diagonal_l665_66529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_sweep_area_l665_66594

theorem chord_sweep_area (r : ℝ) (r_pos : r > 0) :
  let swept_area := r^2 * (7 * Real.pi - 4) / 16
  swept_area = r^2 * (7 * Real.pi - 4) / 16 := by
  -- Define the swept area
  let swept_area := r^2 * (7 * Real.pi - 4) / 16
  
  -- The proof would go here, but we'll use sorry for now
  sorry

#check chord_sweep_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_sweep_area_l665_66594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_not_expressible_l665_66506

theorem min_value_not_expressible : 
  ∃ n : ℕ, n = 35 ∧
  (n > 0 ∧ ¬(2 ∣ n) ∧ ¬(3 ∣ n)) ∧
  (∀ a b : ℕ, (2^a : ℤ) - (3^b : ℤ) ≠ n ∧ (3^b : ℤ) - (2^a : ℤ) ≠ n) ∧
  (∀ m : ℕ, m < n → (m > 0 ∧ ¬(2 ∣ m) ∧ ¬(3 ∣ m)) → 
    ∃ a b : ℕ, ((2^a : ℤ) - (3^b : ℤ) = m ∨ (3^b : ℤ) - (2^a : ℤ) = m)) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_not_expressible_l665_66506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_undefined_at_one_l665_66586

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (x - 5) / (x - 7)

-- State the theorem
theorem inverse_g_undefined_at_one :
  ¬∃ y, g y = 1 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_undefined_at_one_l665_66586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_intervals_range_for_three_zeros_l665_66530

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x - 4 + a/x

-- Part 1: Monotonic intervals when a = 4
theorem monotonic_intervals :
  let f₄ := f 4
  (∀ x₁ x₂, x₁ < x₂ ∧ x₁ < 0 ∧ x₂ < 0 → f₄ x₁ > f₄ x₂) ∧
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f₄ x₁ > f₄ x₂) ∧
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f₄ x₁ < f₄ x₂) :=
by
  sorry

-- Part 2: Range of a for three zeros
theorem range_for_three_zeros :
  ∀ a : ℝ, (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) ↔
  (-8 < a ∧ a < 0) ∨ (0 < a ∧ a < 40/27) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_intervals_range_for_three_zeros_l665_66530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_flow_at_3_30pm_l665_66531

/-- Traffic flow function -/
noncomputable def traffic_flow (A : ℝ) (t : ℝ) : ℝ := A * Real.sin (Real.pi/4 * t - 13*Real.pi/8) + 300

/-- Theorem stating the approximate traffic flow at 3:30 PM -/
theorem traffic_flow_at_3_30pm
  (h1 : ∀ t, 6 ≤ t → t ≤ 18 → ∃ A, traffic_flow A t = traffic_flow A 8.5)
  (h2 : ∃ A, traffic_flow A 8.5 = 500) :
  ∃ A, |traffic_flow A 15.5 - 441| < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_flow_at_3_30pm_l665_66531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_volume_solid_volume_proof_l665_66520

/-- The volume of the solid described by the three views is 60 -/
theorem solid_volume : ℕ :=
  60

/-- Proof that the volume of the solid is 60 -/
theorem solid_volume_proof : solid_volume = 60 := by
  -- The proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_volume_solid_volume_proof_l665_66520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unchanged_stats_l665_66557

def original_set : Finset Int := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

def mean (s : Finset Int) : ℚ :=
  (s.sum (λ x => (x : ℚ))) / s.card

def variance (s : Finset Int) : ℚ :=
  (s.sum (λ x => (x : ℚ)^2)) / s.card - (mean s)^2

def new_set1 : Finset Int := {-5, 1, -5, -3, -2, -1, 0, 1, 2, 3, 4, 5}
def new_set2 : Finset Int := {-5, -4, -3, -2, -1, 0, 1, 2, 3, -1, 5, 5}

theorem unchanged_stats :
  (mean original_set = mean new_set1) ∧
  (variance original_set = variance new_set1) ∧
  (mean original_set = mean new_set2) ∧
  (variance original_set = variance new_set2) :=
by sorry

#eval mean original_set
#eval variance original_set
#eval mean new_set1
#eval variance new_set1
#eval mean new_set2
#eval variance new_set2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unchanged_stats_l665_66557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_beetles_to_meet_l665_66599

/-- Represents a beetle on the grid -/
structure Beetle where
  x : Nat
  y : Nat

/-- The grid size -/
def gridSize : Nat := 2023

/-- Function to determine if a cell is black in the checkerboard pattern -/
def isBlackCell (x y : Nat) : Bool :=
  (x + y) % 2 = 0

/-- Function to determine if a beetle is in a happy state -/
def isHappyBeetle (b : Beetle) : Bool :=
  isBlackCell b.x b.y && (b.x % 2 = 0)

/-- Function to determine if a beetle is in a sad state -/
def isSadBeetle (b : Beetle) : Bool :=
  isBlackCell b.x b.y && (b.x % 2 = 1)

/-- The theorem to be proved -/
theorem minimum_beetles_to_meet (beetles : List Beetle) : 
  (∀ b1 b2, b1 ∈ beetles → b2 ∈ beetles → b1 ≠ b2 → ∃ t : Nat, 
    (b1.x + t) % gridSize ≠ (b2.x + t) % gridSize ∨ 
    (b1.y + t) % gridSize ≠ (b2.y + t) % gridSize) →
  beetles.length ≤ 4088484 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_beetles_to_meet_l665_66599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_unit_expression_l665_66590

theorem imaginary_unit_expression : 
  Complex.I + 2 * Complex.I^2 + 3 * Complex.I^3 + 4 * Complex.I^4 = 2 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_unit_expression_l665_66590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_a_l665_66509

theorem triangle_angle_a (a b c : ℝ) (h : a^2 - c^2 = b^2 - b*c) :
  Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_a_l665_66509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_sum_l665_66553

/-- A cubic polynomial function -/
def cubic_polynomial (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + c * x + d

theorem cubic_polynomial_sum (a b c d : ℝ) :
  cubic_polynomial a b c d 1 = 20 →
  cubic_polynomial a b c d (-1) = 16 →
  b + d = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_sum_l665_66553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_rectangle_perimeter_l665_66579

-- Define the locus W
def W : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; |y| = Real.sqrt (x^2 + (y - 1/2)^2)}

-- State the theorem
theorem locus_and_rectangle_perimeter :
  -- Part 1: The equation of W
  (∀ p : ℝ × ℝ, p ∈ W ↔ p.2 = p.1^2 + 1/4) ∧
  -- Part 2: Perimeter of rectangle ABCD
  (∀ A B C D : ℝ × ℝ,
    A ∈ W → B ∈ W → C ∈ W →
    (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0 →
    2 * (((B.1 - A.1)^2 + (B.2 - A.2)^2).sqrt + ((C.1 - B.1)^2 + (C.2 - B.2)^2).sqrt) > 3 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_rectangle_perimeter_l665_66579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_foci_hyperbola_l665_66511

/-- The distance between the foci of a hyperbola -/
noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 + b^2)

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 18 * x - 16 * y^2 - 32 * y = 144

/-- Theorem: The distance between the foci of the given hyperbola is √3047/6 -/
theorem distance_foci_hyperbola :
  ∃ a b : ℝ, (∀ x y : ℝ, hyperbola_equation x y ↔ 
    (x - 1)^2 / a^2 - (y + 1)^2 / b^2 = 1) ∧
    distance_between_foci a b = Real.sqrt 3047 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_foci_hyperbola_l665_66511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_from_rectangle_l665_66548

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a triangular pyramid -/
structure TriangularPyramid where
  base : Point3D
  vertex : Point3D

/-- Calculate the volume of a triangular pyramid -/
noncomputable def pyramidVolume (pyramid : TriangularPyramid) (baseArea : ℝ) : ℝ :=
  (1 / 3) * baseArea * pyramid.vertex.z

theorem pyramid_volume_from_rectangle (rect : Rectangle) 
  (h1 : rect.length = 10 * Real.sqrt 2)
  (h2 : rect.width = 15 * Real.sqrt 2)
  (pyramid : TriangularPyramid)
  (h3 : pyramid.base.x = 5 * Real.sqrt 2)
  (h4 : pyramid.base.y = 0)
  (h5 : pyramid.base.z = 0)
  (h6 : pyramid.vertex.x = 0)
  (h7 : pyramid.vertex.y = 275 / (2 * Real.sqrt 275))
  (h8 : pyramid.vertex.z = 225 / (2 * Real.sqrt 31))
  (baseArea : ℝ)
  (h9 : baseArea = 15 * Real.sqrt 31) :
  pyramidVolume pyramid baseArea = 1687.5 := by
  sorry

#check pyramid_volume_from_rectangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_from_rectangle_l665_66548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangles_equal_area_l665_66541

-- Define the triangles
def triangle1 : ℝ × ℝ × ℝ := (13, 13, 10)
def triangle2 : ℝ × ℝ × ℝ := (13, 13, 24)

-- Define a function to calculate the area of a triangle using Heron's formula
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem isosceles_triangles_equal_area :
  triangleArea triangle1.fst triangle1.snd.fst triangle1.snd.snd =
  triangleArea triangle2.fst triangle2.snd.fst triangle2.snd.snd := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangles_equal_area_l665_66541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_class_size_l665_66581

theorem math_class_size (total_students : ℕ) (both_subjects : ℕ) 
  (h1 : total_students = 52)
  (h2 : both_subjects = 6)
  (h3 : ∃ (math_only physics_only : ℕ), 
    total_students = math_only + physics_only + both_subjects ∧
    math_only + both_subjects = 2 * (physics_only + both_subjects)) :
  ∃ (math_class : ℕ), math_class = 38 ∧ 
    math_class = total_students - (total_students - both_subjects) / 3 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_class_size_l665_66581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_sixteen_l665_66567

theorem divisible_by_sixteen (n : ℕ) : 
  16 ∣ (12 * n^2 + 8 * n + ((-1 : ℤ)^n * (9 + (-1 : ℤ)^n * 7) * 2 : ℤ)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_sixteen_l665_66567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l665_66526

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 - 4*x - 9*y^2 - 18*y = 45

/-- The distance between the foci of the hyperbola -/
noncomputable def foci_distance : ℝ := 40/3

/-- Theorem stating that the distance between the foci of the given hyperbola is 40/3 -/
theorem hyperbola_foci_distance :
  ∀ (x y : ℝ), hyperbola_eq x y → foci_distance = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l665_66526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_is_six_l665_66588

def exercise_data : List (Nat × Nat) := [(5, 2), (6, 6), (7, 5), (8, 2)]

def total_students : Nat := 15

theorem median_is_six (data : List (Nat × Nat)) (total : Nat) :
  data = exercise_data → total = total_students → 
  ∃ (ordered_list : List Nat), 
    (ordered_list.length = total) ∧ 
    (List.Sorted (· ≤ ·) ordered_list) ∧
    (ordered_list.get! ((total - 1) / 2) = 6) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_is_six_l665_66588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decaf_percentage_approx_l665_66504

/-- Represents the coffee inventory and calculates the percentage of decaffeinated coffee -/
noncomputable def coffee_inventory (initial_stock : ℝ) (type_a_percent : ℝ) (type_b_percent : ℝ)
  (type_a_decaf : ℝ) (type_b_decaf : ℝ) (type_c_decaf : ℝ) (additional_type_c : ℝ) : ℝ :=
  let type_a_weight := initial_stock * type_a_percent
  let type_b_weight := initial_stock * type_b_percent
  let type_c_weight := initial_stock * (1 - type_a_percent - type_b_percent)
  let type_a_decaf_weight := type_a_weight * type_a_decaf
  let type_b_decaf_weight := type_b_weight * type_b_decaf
  let type_c_decaf_weight := type_c_weight * type_c_decaf
  let new_type_c_weight := type_c_weight + additional_type_c
  let new_type_c_decaf_weight := new_type_c_weight * type_c_decaf
  let total_decaf_weight := type_a_decaf_weight + type_b_decaf_weight + new_type_c_decaf_weight
  let new_total_weight := initial_stock + additional_type_c
  (total_decaf_weight / new_total_weight) * 100

/-- Theorem stating that the percentage of decaffeinated coffee is approximately 47.24% -/
theorem decaf_percentage_approx :
  ∃ ε > 0, |coffee_inventory 700 0.4 0.35 0.3 0.5 0.6 150 - 47.24| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decaf_percentage_approx_l665_66504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_needed_l665_66523

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of boxes needed given the total cost and cost per box -/
def numBoxesNeeded (totalCost costPerBox : ℚ) : ℕ :=
  (totalCost / costPerBox).ceil.toNat

/-- Theorem: Given the conditions, the total volume needed to package the collection is 3,060,000 cubic inches -/
theorem total_volume_needed
  (boxDim : BoxDimensions)
  (costPerBox totalCost : ℚ)
  (h1 : boxDim.length = 20)
  (h2 : boxDim.width = 20)
  (h3 : boxDim.height = 15)
  (h4 : costPerBox = 13/10)
  (h5 : totalCost = 663) :
  (numBoxesNeeded totalCost costPerBox) * (boxVolume boxDim) = 3060000 := by
  sorry

#eval boxVolume { length := 20, width := 20, height := 15 }
#eval numBoxesNeeded 663 (13/10)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_needed_l665_66523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l665_66542

theorem triangle_inequalities (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) > Real.sqrt (c^2 - c*a + a^2)) ∧
  (Real.sqrt (a^2 + a*b + b^2) + Real.sqrt (b^2 + b*c + c^2) > Real.sqrt (c^2 + c*a + a^2)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l665_66542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equals_26_l665_66551

theorem ceiling_sum_equals_26 : 
  ⌈Real.sqrt (25/9 : ℝ)⌉ + ⌈(25/9 : ℝ)^3⌉ + ⌈(25/9 : ℝ)^(1/3)⌉ = 26 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equals_26_l665_66551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_a_value_l665_66507

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 * Real.cos x + 1

-- State the theorem
theorem f_minus_a_value (a : ℝ) (h : f a = 11) : f (-a) = -9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minus_a_value_l665_66507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_interval_l665_66539

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x - Real.pi / 4)

noncomputable def g (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (x / 2 - Real.pi / 12)

theorem g_decreasing_interval :
  ∀ x y, -17*Real.pi/6 ≤ x ∧ x < y ∧ y ≤ -5*Real.pi/6 → g y < g x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_decreasing_interval_l665_66539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_root_l665_66560

/-- The function f(x) = 2x - sin(x) --/
noncomputable def f (x : ℝ) := 2 * x - Real.sin x

/-- Theorem stating that f(x) has exactly one root --/
theorem f_has_unique_root : ∃! x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_root_l665_66560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l665_66514

noncomputable def f (x : ℝ) := Real.exp (x - 1) + Real.exp (1 - x)

theorem f_inequality_range (x : ℝ) :
  f (x - 1) < Real.exp 1 + Real.exp (-1) ↔ 1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l665_66514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrochloric_acid_solution_volume_l665_66516

/-- Given an initial solution of hydrochloric acid and pure acid added, 
    calculate the initial volume to achieve a target concentration -/
noncomputable def initial_volume (initial_concentration : ℝ) (final_concentration : ℝ) 
                   (pure_acid_added : ℝ) : ℝ :=
  (pure_acid_added * final_concentration) / (final_concentration - initial_concentration)

/-- Theorem stating that the initial volume of 10% hydrochloric acid solution 
    is 60 liters when 3.52941176471 liters of pure acid is added to make a 15% solution -/
theorem hydrochloric_acid_solution_volume : 
  initial_volume 0.10 0.15 3.52941176471 = 60 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hydrochloric_acid_solution_volume_l665_66516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l665_66534

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 3 / (x^2 - 4*x + 5)

-- Define the function f as a composition of g
noncomputable def f (x : ℝ) : ℝ := g (g (g x))

-- Statement of the theorem
theorem range_of_f :
  Set.range f = Set.Icc (3/50) 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l665_66534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_l665_66572

/-- Calculates the total time for a round trip boat journey given the boat's speed in still water,
    the stream's speed, and the distance to the destination. -/
theorem round_trip_time
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (distance : ℝ)
  (h1 : boat_speed = 5)
  (h2 : stream_speed = 2)
  (h3 : distance = 252)
  : (distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed)) = 120 := by
  have downstream_speed : ℝ := boat_speed + stream_speed
  have upstream_speed : ℝ := boat_speed - stream_speed
  have downstream_time : ℝ := distance / downstream_speed
  have upstream_time : ℝ := distance / upstream_speed
  have total_time : ℝ := downstream_time + upstream_time
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_l665_66572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_formula_l665_66591

/-- Given three sequences of points in a Cartesian plane -/
def A : ℕ → ℝ × ℝ := sorry
def B : ℕ → ℝ × ℝ := sorry
def C : ℕ → ℝ × ℝ := sorry

/-- The x-coordinate of A₍ₙ₎ is n -/
axiom A_x (n : ℕ) : (A n).1 = n

/-- The x-coordinate of B₍ₙ₎ is n -/
axiom B_x (n : ℕ) : (B n).1 = n

/-- C₍ₙ₎ has coordinates (n-1, 0) -/
axiom C_coord (n : ℕ) : C n = ((n - 1 : ℝ), 0)

/-- Vector A₍ₙ₎A₍ₙ₊₁₎ is collinear with vector B₍ₙ₎C₍ₙ₎ -/
axiom collinear (n : ℕ) : ∃ (k : ℝ), k ≠ 0 ∧ 
  (A (n + 1)).1 - (A n).1 = k * ((C n).1 - (B n).1) ∧
  (A (n + 1)).2 - (A n).2 = k * ((C n).2 - (B n).2)

/-- The difference between consecutive y-coordinates of B is 6 -/
axiom B_diff (n : ℕ) : (B (n + 1)).2 - (B n).2 = 6

/-- Initial conditions -/
axiom initial : (A 1).2 = 0 ∧ (B 1).2 = 0

/-- The main theorem: aₙ = 3n² - 9n + 6 -/
theorem a_n_formula (n : ℕ) : (A n).2 = 3 * n^2 - 9 * n + 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_n_formula_l665_66591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inlet_flow_rate_l665_66589

/-- Represents the rate at which water flows, in litres per hour -/
abbrev FlowRate := ℝ

/-- Represents the capacity of the tank in litres -/
def TankCapacity : ℝ := 1440

/-- Represents the time it takes for the leak to empty the full tank, in hours -/
def LeakEmptyTime : ℝ := 3

/-- Represents the time it takes for the tank to empty with both leak and inlet open, in hours -/
def CombinedEmptyTime : ℝ := 12

/-- Calculates the flow rate of the leak in litres per hour -/
noncomputable def leakRate : FlowRate := TankCapacity / LeakEmptyTime

/-- Calculates the net flow rate when both inlet and leak are active, in litres per hour -/
noncomputable def netFlowRate : FlowRate := TankCapacity / CombinedEmptyTime

/-- Calculates the inlet flow rate in litres per hour -/
noncomputable def inletRate : FlowRate := netFlowRate + leakRate

/-- Converts the inlet flow rate from litres per hour to litres per minute -/
noncomputable def inletRatePerMinute : ℝ := inletRate / 60

theorem inlet_flow_rate :
  inletRatePerMinute = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inlet_flow_rate_l665_66589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_l665_66598

noncomputable def distance_AB : ℝ := 12

noncomputable def angle_Alice : ℝ := Real.pi / 4  -- 45 degrees in radians

noncomputable def angle_Bob : ℝ := Real.pi / 6  -- 30 degrees in radians

noncomputable def angle_northeast : ℝ := Real.pi / 4  -- 45 degrees in radians

theorem airplane_altitude (h : ℝ) :
  h = distance_AB / Real.sqrt 2 → (
    Real.tan angle_Alice = h / (distance_AB / Real.sqrt 2) ∧
    Real.tan angle_Bob = h / (distance_AB * (1 - 1 / Real.sqrt 2)) ∧
    Real.sin angle_northeast * distance_AB = distance_AB / Real.sqrt 2
  ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_l665_66598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_after_discounts_l665_66566

/-- Represents a book category with its price, quantity, and discount rule. -/
structure BookCategory where
  price : ℚ
  quantity : ℕ
  discountRate : ℚ
  discountThreshold : ℕ

/-- Calculates the total cost for a book category after applying discounts. -/
def calculateCategoryCost (category : BookCategory) : ℚ :=
  let discountedGroups := category.quantity / category.discountThreshold
  let remainingBooks := category.quantity % category.discountThreshold
  let discountedPrice := category.price * (1 - category.discountRate)
  (discountedGroups * category.discountThreshold : ℚ) * discountedPrice +
  (remainingBooks : ℚ) * category.price

/-- Represents the bookstore scenario with all book categories. -/
def bookstore : List BookCategory :=
  [{ price := 15, quantity := 6, discountRate := 1/5, discountThreshold := 3 },
   { price := 12, quantity := 4, discountRate := 3/20, discountThreshold := 2 },
   { price := 10, quantity := 8, discountRate := 1/10, discountThreshold := 5 },
   { price := 15/2, quantity := 10, discountRate := 1/4, discountThreshold := 4 },
   { price := 5, quantity := 5, discountRate := 1/20, discountThreshold := 1 }]

/-- Theorem stating that the total cost after discounts is $271.55. -/
theorem total_cost_after_discounts :
  (bookstore.map calculateCategoryCost).sum = 54311/200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_after_discounts_l665_66566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l665_66558

/-- The positive slope of the asymptotes for the hyperbola x^2/49 - y^2/36 = 1 -/
noncomputable def asymptote_slope : ℝ := 6/7

/-- The equation of the hyperbola -/
def is_on_hyperbola (x y : ℝ) : Prop := x^2/49 - y^2/36 = 1

theorem hyperbola_asymptote_slope :
  ∀ ε > 0, ∃ X > 0, ∀ x y, is_on_hyperbola x y → x > X →
    abs (y/x - asymptote_slope) < ε ∨ abs (y/x + asymptote_slope) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_slope_l665_66558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frame_division_theorem_l665_66597

/-- Represents a piece of the frame -/
structure Piece where
  shape : Set (ℕ × ℕ)
  is_connected : Bool

/-- Represents the rectangular frame -/
structure Frame where
  width : ℕ
  height : ℕ
  outline : Set (ℕ × ℕ)

/-- Divides the frame into pieces -/
def divide_frame (f : Frame) : List Piece :=
  sorry

/-- Checks if a list of pieces can form a 6x6 square -/
def can_form_square (pieces : List Piece) : Bool :=
  sorry

/-- Checks if all pieces in a list are distinct -/
def all_distinct (pieces : List Piece) : Bool :=
  sorry

/-- Main theorem: There exists a frame that can be divided into 9 distinct pieces
    that can form a 6x6 square -/
theorem frame_division_theorem :
  ∃ (f : Frame),
    let pieces := divide_frame f
    pieces.length = 9 ∧
    all_distinct pieces ∧
    can_form_square pieces :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frame_division_theorem_l665_66597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_younger_brother_silver_fraction_l665_66550

theorem younger_brother_silver_fraction
  (gold silver : ℝ)
  (hg : gold > 0)
  (hs : silver > 0)
  (htotal : gold + silver = 600)
  (helder : gold / 5 + silver / 7 = 100) :
  gold / 7 + (9 / 49) * silver = 100 :=
by
  sorry

#check younger_brother_silver_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_younger_brother_silver_fraction_l665_66550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_multiple_of_15_with_9_and_0_digits_l665_66518

def is_multiple_of_15 (n : ℕ) : Prop := ∃ k, n = 15 * k

def digits_are_9_or_0 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 9 ∨ d = 0

def largest_valid_number : ℕ := 9990

theorem largest_multiple_of_15_with_9_and_0_digits :
  is_multiple_of_15 largest_valid_number ∧
  digits_are_9_or_0 largest_valid_number ∧
  (∀ n : ℕ, n > largest_valid_number →
    ¬(is_multiple_of_15 n ∧ digits_are_9_or_0 n)) ∧
  largest_valid_number / 15 = 666 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_multiple_of_15_with_9_and_0_digits_l665_66518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_example_l665_66596

/-- Dilation of a complex number -/
noncomputable def dilation (center : ℂ) (scale : ℝ) (z : ℂ) : ℂ :=
  center + scale • (z - center)

theorem dilation_example :
  let center : ℂ := 2 - 3*I
  let scale : ℝ := 3
  let z : ℂ := -1 + 2*I
  dilation center scale z = -7 + 12*I := by
  -- Expand the definition of dilation
  unfold dilation
  -- Simplify the expression
  simp [Complex.I]
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_example_l665_66596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_part1_calculation_part2_l665_66538

-- Part 1
theorem calculation_part1 : 
  (2 + 3/5 : ℝ)^0 + 2^(-2 : ℤ) * (2 + 1/4 : ℝ)^(-1/2 : ℝ) - (1/100 : ℝ)^(1/2 : ℝ) = 16/15 := by sorry

-- Part 2
theorem calculation_part2 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a^(-2 : ℤ) * b^(-3 : ℤ)) * (-4 * a^(-1 : ℤ) * b) / (12 * a^(-4 : ℤ) * b^(-2 : ℤ) * c) = -a / (3 * c) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_part1_calculation_part2_l665_66538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_alignment_l665_66535

noncomputable def minute_hand_position (minutes_past_3 : ℝ) : ℝ :=
  6 * minutes_past_3

noncomputable def hour_hand_position (minutes_past_3 : ℝ) : ℝ :=
  90 + 0.5 * minutes_past_3

noncomputable def time_now : ℝ := 7 + 27 / 60

theorem clock_alignment :
  minute_hand_position (time_now + 8) = hour_hand_position (time_now - 2) := by
  -- Unfold definitions
  unfold minute_hand_position hour_hand_position time_now
  -- Simplify expressions
  simp [mul_add, add_mul]
  -- Check equality (this step would typically involve more detailed calculations)
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_alignment_l665_66535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_greater_than_f_y_between_zero_and_x_l665_66515

/-- The partial sum of the Taylor series for e^x up to n terms -/
noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
  Finset.sum (Finset.range (n+1)) (λ k => x^k / (Nat.factorial k : ℝ))

theorem exp_greater_than_f (x : ℝ) (n : ℕ) (hx : x > 0) :
  Real.exp x > f n x := by sorry

theorem y_between_zero_and_x (x y : ℝ) (n : ℕ) (hx : x > 0) 
  (h : Real.exp x = f n x + Real.exp y / (Nat.factorial (n+1) : ℝ) * x^(n+1)) :
  0 < y ∧ y < x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_greater_than_f_y_between_zero_and_x_l665_66515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_divides_circle_ratio_l665_66576

/-- The line that divides the circle -/
def dividing_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 2 = 0

/-- The circle that is divided -/
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- The ratio of the arc lengths -/
def arc_ratio : ℚ × ℚ := (1, 2)

/-- Theorem stating that the line divides the circle in the ratio 1:2 -/
theorem line_divides_circle_ratio :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    dividing_line x₁ y₁ ∧ dividing_line x₂ y₂ ∧
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
    arc_ratio = (1, 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_divides_circle_ratio_l665_66576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_product_fourth_root_floor_even_l665_66582

theorem consecutive_product_fourth_root_floor_even (x : ℕ) :
  ∃ k : ℕ, k * 2 = ⌊((x : ℝ) * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5) * (x + 6) * (x + 7)) ^ (1/4)⌋ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_product_fourth_root_floor_even_l665_66582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratios_l665_66502

/-- Triangle PQR with sides a, b, c -/
structure TrianglePQR where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 7
  hb : b = 24
  hc : c = 25

/-- Triangle XYZ with sides d, e, f -/
structure TriangleXYZ where
  d : ℝ
  e : ℝ
  f : ℝ
  hd : d = 9
  he : e = 40
  hf : f = 41

/-- The area of a triangle given two sides -/
noncomputable def triangleArea (s1 s2 : ℝ) : ℝ := (1/2) * s1 * s2

/-- The perimeter of a triangle given three sides -/
def trianglePerimeter (s1 s2 s3 : ℝ) : ℝ := s1 + s2 + s3

theorem triangle_ratios (pqr : TrianglePQR) (xyz : TriangleXYZ) :
  (triangleArea pqr.a pqr.b / triangleArea xyz.d xyz.e = 7/15) ∧
  (trianglePerimeter pqr.a pqr.b pqr.c / trianglePerimeter xyz.d xyz.e xyz.f = 28/45) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratios_l665_66502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_traffic_flow_rate_speed_range_for_min_flow_rate_l665_66544

-- Define the traffic flow rate function
noncomputable def y (v : ℝ) : ℝ := 920 * v / (v^2 + 3 * v + 1600)

-- Theorem for the maximum traffic flow rate
theorem max_traffic_flow_rate :
  ∃ (v_max : ℝ), v_max > 0 ∧
    (∀ (v : ℝ), v > 0 → y v ≤ y v_max) ∧
    v_max = 40 ∧ y v_max = 920 / 83 := by
  sorry

-- Theorem for the range of average speed
theorem speed_range_for_min_flow_rate :
  ∀ (v : ℝ), v > 0 →
    (y v ≥ 10 ↔ 25 ≤ v ∧ v ≤ 64) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_traffic_flow_rate_speed_range_for_min_flow_rate_l665_66544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_30_prime_factors_l665_66573

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

def count_distinct_prime_factors (n : ℕ) : ℕ :=
  (Finset.filter Nat.Prime (Finset.range (n + 1))).card

theorem factorial_30_prime_factors :
  count_distinct_prime_factors (factorial 30) = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_30_prime_factors_l665_66573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_is_odd_f_has_period_pi_is_smallest_period_l665_66556

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi)

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧ 
  (∃ p > 0, ∀ x, f (x + p) = f x) ∧
  (∀ p > 0, (∀ x, f (x + p) = f x) → p ≥ Real.pi) := by
  sorry

-- Theorem stating that f is odd
theorem f_is_odd : ∀ x, f (-x) = -f x := by
  sorry

-- Theorem stating that f has a positive period
theorem f_has_period : ∃ p > 0, ∀ x, f (x + p) = f x := by
  sorry

-- Theorem stating that pi is the smallest positive period
theorem pi_is_smallest_period : 
  ∀ p > 0, (∀ x, f (x + p) = f x) → p ≥ Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_is_odd_f_has_period_pi_is_smallest_period_l665_66556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_tetrahedron_volume_centroid_to_inscribed_ratio_l665_66510

/-- A parallelepiped in 3D space -/
structure Parallelepiped where
  volume : ℝ

/-- A tetrahedron in 3D space -/
structure Tetrahedron where
  volume : ℝ

/-- An inscribed tetrahedron in a parallelepiped -/
noncomputable def inscribed_tetrahedron (p : Parallelepiped) : Tetrahedron :=
  { volume := p.volume / 3 }

/-- The tetrahedron formed by the centroids of cut-off tetrahedra -/
noncomputable def centroid_tetrahedron (p : Parallelepiped) : Tetrahedron :=
  { volume := p.volume / 24 }

/-- Theorem stating the volume relationship between the centroid tetrahedron and the parallelepiped -/
theorem centroid_tetrahedron_volume (p : Parallelepiped) :
  (centroid_tetrahedron p).volume = p.volume / 24 := by
  sorry

/-- Theorem stating the volume relationship between the centroid tetrahedron and the inscribed tetrahedron -/
theorem centroid_to_inscribed_ratio (p : Parallelepiped) :
  (centroid_tetrahedron p).volume = (inscribed_tetrahedron p).volume / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_tetrahedron_volume_centroid_to_inscribed_ratio_l665_66510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l665_66595

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

-- State the theorem
theorem f_inequality (a : ℝ) : f (a + 3) > f (2 * a) ↔ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l665_66595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_gcd_problem_l665_66532

theorem lcm_gcd_problem (a b : ℕ) : 
  Nat.lcm a b = 2310 → 
  Nat.gcd a b = 30 → 
  b = 330 → 
  a = 210 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_gcd_problem_l665_66532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eliza_height_l665_66561

/-- Represents the height of a person in inches -/
def Height := ℕ

/-- Represents a group of siblings -/
structure SiblingGroup where
  total_siblings : ℕ
  total_height : ℕ
  known_heights : List ℕ
  unknown_sibling_height : ℕ → ℕ

/-- The problem setup -/
def eliza_siblings : SiblingGroup :=
  { total_siblings := 6
  , total_height := 435
  , known_heights := [66, 66, 60, 75]
  , unknown_sibling_height := fun h => h + 4
  }

/-- Theorem stating Eliza's height -/
theorem eliza_height : 
  ∃ (h : ℕ), 
    h + eliza_siblings.unknown_sibling_height h + 
    (eliza_siblings.known_heights.sum) = 
    eliza_siblings.total_height ∧ h = 82 := by
  sorry

#eval eliza_siblings.total_height
#eval eliza_siblings.known_heights.sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eliza_height_l665_66561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_committee_probability_l665_66528

/-- The number of members in the club -/
def total_members : ℕ := 30

/-- The number of boys in the club -/
def num_boys : ℕ := 12

/-- The number of girls in the club -/
def num_girls : ℕ := 18

/-- The size of the committee -/
def committee_size : ℕ := 6

/-- The probability of choosing a committee with at least one boy and one girl -/
def prob_mixed_committee : ℚ := 574287 / 593775

theorem mixed_committee_probability :
  prob_mixed_committee = 1 - (Nat.choose num_boys committee_size + Nat.choose num_girls committee_size : ℚ) / Nat.choose total_members committee_size :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixed_committee_probability_l665_66528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_two_l665_66524

/-- Rectangle OABC with area 4 -/
structure Rectangle where
  a : ℝ
  b : ℝ
  area_eq_4 : a * b = 4

/-- Inverse proportional function y = k/x -/
noncomputable def inverse_prop (k : ℝ) (x : ℝ) : ℝ := k / x

/-- Point on the graph of y = k/x -/
structure PointOnGraph (k : ℝ) where
  x : ℝ
  y : ℝ
  x_pos : x > 0
  on_graph : y = inverse_prop k x

/-- The theorem to be proved -/
theorem min_distance_is_two (rect : Rectangle) 
  (k : ℝ) 
  (passes_midpoint : inverse_prop k (rect.a / 2) = rect.b / 2) :
  ∃ (min_dist : ℝ), 
    min_dist = 2 ∧ 
    ∀ (p : PointOnGraph k), 
      Real.sqrt (p.x ^ 2 + p.y ^ 2) ≥ min_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_two_l665_66524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_eight_l665_66559

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then x^2 + 2 else 2*x

-- State the theorem
theorem function_value_eight (x₀ : ℝ) : f x₀ = 8 → x₀ = 4 ∨ x₀ = -Real.sqrt 6 := by
  intro h
  by_cases h₁ : x₀ ≤ 2
  · -- Case: x₀ ≤ 2
    have : x₀^2 + 2 = 8 := by
      rw [f] at h
      simp [h₁] at h
      exact h
    sorry -- Proof omitted
  · -- Case: x₀ > 2
    have : 2*x₀ = 8 := by
      rw [f] at h
      simp [h₁] at h
      exact h
    have : x₀ = 4 := by
      linarith
    left
    exact this


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_value_eight_l665_66559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_fourth_vertex_distance_l665_66525

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  normal : Point3D
  d : ℝ

/-- Represents a parallelogram in 3D space -/
structure Parallelogram3D where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Distance from a point to a plane -/
noncomputable def distanceToPlane (p : Point3D) (plane : Plane3D) : ℝ :=
  sorry

/-- Theorem: Distance from the fourth vertex of a parallelogram to a plane -/
theorem parallelogram_fourth_vertex_distance 
  (ABCD : Parallelogram3D) (M : Plane3D) 
  (a b c : ℝ) 
  (ha : distanceToPlane ABCD.A M = a)
  (hb : distanceToPlane ABCD.B M = b)
  (hc : distanceToPlane ABCD.C M = c) :
  ∃ d : ℝ, distanceToPlane ABCD.D M = d ∧ d = |a + c - b| := by
  sorry

#check parallelogram_fourth_vertex_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_fourth_vertex_distance_l665_66525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l665_66562

-- Define the train properties
noncomputable def train1_length : ℝ := 100
noncomputable def train2_length : ℝ := 200
noncomputable def initial_distance : ℝ := 140
noncomputable def train1_speed_kmh : ℝ := 54
noncomputable def train2_speed_kmh : ℝ := 72

-- Convert km/h to m/s
noncomputable def km_per_hour_to_m_per_second (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

-- Calculate the meeting time
noncomputable def meeting_time : ℝ :=
  let train1_speed_ms := km_per_hour_to_m_per_second train1_speed_kmh
  let train2_speed_ms := km_per_hour_to_m_per_second train2_speed_kmh
  let relative_speed := train1_speed_ms + train2_speed_ms
  let total_distance := train1_length + train2_length + initial_distance
  total_distance / relative_speed

-- Theorem statement
theorem trains_meet_time :
  ∃ ε > 0, |meeting_time - 12.57| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l665_66562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_divisibility_property_l665_66583

theorem function_divisibility_property 
  (k : ℕ+) 
  (f : ℕ+ → ℕ+) 
  (h : ∀ (m n : ℕ+), (f m + f n) ∣ (m + n)^(k : ℕ)) :
  ∃ c : ℕ, ∀ n : ℕ+, f n = n + c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_divisibility_property_l665_66583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l665_66555

noncomputable section

/-- Definition of the ellipse E -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The ellipse passes through the point (0, √2) -/
def passes_through_point (a b : ℝ) : Prop :=
  ellipse a b 0 (Real.sqrt 2)

/-- The eccentricity of the ellipse -/
def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- The left focus of the ellipse -/
def left_focus (a b : ℝ) : ℝ × ℝ :=
  (-Real.sqrt (a^2 - b^2), 0)

/-- The dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem ellipse_properties (a b : ℝ) 
    (h1 : a > b) (h2 : b > 0) 
    (h3 : passes_through_point a b)
    (h4 : eccentricity a b = Real.sqrt 6 / 3) :
  -- 1) The standard equation of the ellipse
  (∀ x y : ℝ, ellipse a b x y ↔ x^2 / 6 + y^2 / 2 = 1) ∧
  -- 2) The range of OP · OQ
  (∀ P Q : ℝ × ℝ, 
    ellipse a b P.1 P.2 → 
    ellipse a b Q.1 Q.2 → 
    ∃ k : ℝ, P.2 - (left_focus a b).2 = k * (P.1 - (left_focus a b).1) ∧
             Q.2 - (left_focus a b).2 = k * (Q.1 - (left_focus a b).1) →
    -6 ≤ dot_product P Q ∧ dot_product P Q ≤ 10/3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l665_66555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_l665_66565

/-- Represents the scenario of Yan's journey between home and park -/
structure JourneyScenario where
  walkSpeed : ℝ
  homeDistance : ℝ
  parkDistance : ℝ
  scooterSpeedMultiplier : ℝ
  homeDistancePositive : homeDistance > 0
  parkDistancePositive : parkDistance > 0
  walkSpeedPositive : walkSpeed > 0
  scooterSpeedMultiplierValue : scooterSpeedMultiplier = 5

/-- The time taken for the direct walk to the park equals the time taken to walk home and scooter to the park -/
def equalTime (s : JourneyScenario) : Prop :=
  s.parkDistance / s.walkSpeed = 
    s.homeDistance / s.walkSpeed + (s.homeDistance + s.parkDistance) / (s.scooterSpeedMultiplier * s.walkSpeed)

/-- The theorem stating the ratio of distances -/
theorem distance_ratio (s : JourneyScenario) (h : equalTime s) : 
  s.homeDistance / s.parkDistance = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_l665_66565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_count_theorem_l665_66571

theorem divisor_count_theorem (n : ℕ) (d : ℕ → ℕ) (k : ℕ) : 
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → d i < d j) →  -- divisors are strictly increasing
  (d 1 = 1 ∧ d k = n) →  -- first and last divisors
  (∀ i, 1 ≤ i ∧ i ≤ k → n % d i = 0) →  -- all d_i are divisors of n
  (∀ i, 1 ≤ i ∧ i < k → ∀ m, n % m = 0 → m = d i ∨ m < d i ∨ m > d (i+1)) →  -- d_i are all divisors
  n = d 2 * d 3 + d 2 * d 5 + d 3 * d 5 →  -- given equation
  k = 8 ∨ k = 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_count_theorem_l665_66571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_convergence_l665_66500

/-- Represents the state of candy distribution among children -/
structure CandyDistribution where
  n : ℕ  -- number of children
  candies : Fin n → ℕ  -- candies for each child
  all_even : ∀ i, Even (candies i)

/-- Represents one round of candy distribution -/
def distribute_candies (d : CandyDistribution) : CandyDistribution :=
  sorry

/-- Predicate to check if all children have the same number of candies -/
def uniform_distribution (d : CandyDistribution) : Prop :=
  ∀ i j, d.candies i = d.candies j

/-- The main theorem to prove -/
theorem candy_distribution_convergence
  (initial : CandyDistribution) :
  ∃ k : ℕ, uniform_distribution (Nat.iterate distribute_candies k initial) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_convergence_l665_66500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l665_66580

noncomputable def f₁ (x : ℝ) : ℝ := 3 * x + 3
noncomputable def f₂ (x : ℝ) : ℝ := (1/3) * x + 2
noncomputable def f₃ (x : ℝ) : ℝ := -(1/2) * x + 8

noncomputable def g (x : ℝ) : ℝ := min (f₁ x) (min (f₂ x) (f₃ x))

theorem max_value_of_g :
  ∃ (x_max : ℝ), ∀ (x : ℝ), g x ≤ g x_max ∧ g x_max = 22/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l665_66580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l665_66533

theorem solution_set_of_inequality : 
  {x : ℝ | (x - 2) / (x - 1) ≥ 2} = Set.Icc 0 1 \ {1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l665_66533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_3447_to_hundredth_l665_66536

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The statement that rounding 3.447 to the nearest hundredth equals 3.45 -/
theorem round_3447_to_hundredth :
  round_to_hundredth 3.447 = 3.45 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_3447_to_hundredth_l665_66536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_of_g_l665_66552

open Function Real

theorem no_zeros_of_g (f : ℝ → ℝ) (hf_diff : Differentiable ℝ f) 
  (hf_cont : Continuous f) 
  (hf_cond : ∀ x, x ≠ 0 → deriv f x + x⁻¹ * f x > 0) :
  ∀ x, x ≠ 0 → f x + x⁻¹ ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_zeros_of_g_l665_66552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_pi_6_custom_op_l665_66577

def custom_op (m n : ℝ) : ℝ := m^2 - m*n - n^2

theorem cos_sin_pi_6_custom_op : 
  custom_op (Real.cos (π/6)) (Real.sin (π/6)) = 1/2 - Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_pi_6_custom_op_l665_66577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_theorem_l665_66527

/-- The percentage increase required to regain the original salary after a reduction -/
noncomputable def salary_increase_percentage (original_salary : ℝ) : ℝ :=
  let reduced_salary := 0.85 * original_salary - 50
  (original_salary / reduced_salary - 1) * 100

/-- Theorem stating that the salary increase percentage is approximately 18.75% -/
theorem salary_increase_theorem (original_salary : ℝ) 
  (h1 : original_salary > 50) : 
  ∃ ε > 0, |salary_increase_percentage original_salary - 18.75| < ε := by
  sorry

#check salary_increase_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_theorem_l665_66527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_in_semicircle_l665_66578

/-- Two circles with diameters d and D (d < D) are placed inside a semicircle such that each circle
    touches both the arc and the diameter of the semicircle, as well as the other circle.
    A line is drawn through the centers of the circles, intersecting the extension of the diameter
    of the semicircle at point M. From point M, a tangent is drawn to the arc of the semicircle,
    touching it at point N. Then MN = (D*d)/(D-d). -/
theorem circle_in_semicircle (d D : ℝ) (h : 0 < d ∧ d < D) : 
  ∃ MN : ℝ, MN = (D * d) / (D - d) := by
  sorry

#check circle_in_semicircle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_in_semicircle_l665_66578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_of_circle_H_l665_66564

noncomputable def circle_G_radius : ℝ := 3

-- Define the radii of circles H and I in terms of a variable s
noncomputable def circle_I_radius (s : ℝ) : ℝ := s
noncomputable def circle_H_radius (s : ℝ) : ℝ := 4 * s

-- Define the distance between centers of G and H
noncomputable def GH_distance (s : ℝ) : ℝ := circle_G_radius - circle_H_radius s

-- Define the distance between centers of G and I
noncomputable def GI_distance (s : ℝ) : ℝ := circle_G_radius - circle_I_radius s

-- Define the distance between centers of H and I
noncomputable def HI_distance (s : ℝ) : ℝ := circle_H_radius s + circle_I_radius s

-- Theorem stating the radius of circle H
theorem radius_of_circle_H :
  ∃ s : ℝ, 
    s > 0 ∧ 
    circle_H_radius s = circle_I_radius s * 4 ∧
    GH_distance s > 0 ∧
    GI_distance s > 0 ∧
    HI_distance s = (GH_distance s).sqrt + (GI_distance s).sqrt ∧
    circle_H_radius s = 2 * Real.sqrt 117 - 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_of_circle_H_l665_66564
