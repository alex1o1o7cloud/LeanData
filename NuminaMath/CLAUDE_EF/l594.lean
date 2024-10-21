import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipses_same_eccentricity_l594_59441

/-- Definition of ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Definition of ellipse C₁ -/
def ellipse_C1 (x y : ℝ) : Prop := x^2 / 8 + y^2 / 6 = 1

/-- Eccentricity of an ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- C₁ passes through the point (2, -√3) -/
axiom C1_point : ellipse_C1 2 (-Real.sqrt 3)

/-- C₁'s focus lies on the x-axis -/
axiom C1_focus_on_x_axis : ∃ (f : ℝ), ∀ (y : ℝ), ellipse_C1 f y → y = 0

theorem ellipses_same_eccentricity :
  eccentricity 2 (Real.sqrt 3) = eccentricity (Real.sqrt 8) (Real.sqrt 6) ∧
  eccentricity 2 (Real.sqrt 3) = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipses_same_eccentricity_l594_59441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_cafeteria_survey_l594_59476

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  total_population : ℕ
  sample_size : ℕ
  range_start : ℕ
  range_end : ℕ

/-- Calculates the number of selected items in a given range for a systematic sample -/
def selected_in_range (s : SystematicSample) : ℕ :=
  (s.range_end - s.range_start + 1) / (s.total_population / s.sample_size)

/-- Theorem stating that for the given systematic sample, 5 students are selected from the range [61, 160] -/
theorem school_cafeteria_survey :
  let s : SystematicSample := {
    total_population := 1680,
    sample_size := 84,
    range_start := 61,
    range_end := 160
  }
  selected_in_range s = 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_cafeteria_survey_l594_59476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l594_59424

-- Define the propositions
def p (m : ℝ) : Prop := ∀ x y : ℝ, (x^2 / (m - 5) - y^2 / (m + 3) = 1) ↔ (-3 < m ∧ m < 5)

noncomputable def q : Prop := ∃ x : ℝ, Real.sin x - Real.cos x = 2

-- Theorem statement
theorem problem_statement :
  (∀ m : ℝ, ¬(p m)) ∧
  ¬q ∧
  ¬(∃ m : ℝ, p m ∨ q) ∧
  ¬q :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l594_59424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_square_side_length_l594_59443

/-- The sum of areas of an infinite series of squares, where each subsequent square
    is formed by joining the midpoints of the previous square's sides. -/
noncomputable def sumOfSquareAreas (s : ℝ) : ℝ := s^2 / (1 - 1/2)

/-- Theorem stating that if the sum of all squares' areas is 32 cm², 
    then the side length of the first square is 4 cm. -/
theorem first_square_side_length (s : ℝ) (h : sumOfSquareAreas s = 32) : s = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_square_side_length_l594_59443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l594_59480

noncomputable def f (x : ℝ) : ℝ := 1/2 - 1 / (2^x + 1)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ y, y ∈ Set.range f ↔ -1/2 < y ∧ y < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l594_59480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l594_59402

noncomputable def z : ℂ := (3 - Complex.I ^ 3) / (2 - Complex.I) + Complex.I

theorem imaginary_part_of_z : z.im = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l594_59402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l594_59495

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_ratio (a b : ArithmeticSequence) :
  (∀ k : ℕ, sum_n a k / sum_n b k = (2 * k : ℚ) / (3 * k + 1)) →
  a.a 6 / b.a 6 = 11 / 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l594_59495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_6270_l594_59448

theorem prime_divisors_of_6270 :
  (Finset.filter (fun p => Nat.Prime p ∧ 6270 % p = 0) (Finset.range 6271)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_6270_l594_59448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l594_59411

theorem min_value_expression (n : ℕ+) : 
  (n : ℝ) / 2 + 32 / (n : ℝ) ≥ 8 ∧ ((n : ℝ) / 2 + 32 / (n : ℝ) = 8 ↔ n = 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l594_59411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_theta_l594_59494

-- Define the curve
def curve (x y : ℝ) : Prop := Real.arcsin x = Real.arccos y

-- Define the ellipse
def ellipse (a b x y : ℝ) : Prop := a * x^2 + b * y^2 = 1

-- Define the circle
def circle_condition (x y : ℝ) : Prop := x^2 + y^2 = 2 / Real.sqrt 3

-- Define the relationship between a and b
def a_b_relation (a b : ℝ) : Prop := a = Real.sqrt (1 - b^2) ∧ 0 ≤ b ∧ b ≤ 1

-- Main theorem
theorem range_of_theta (a b : ℝ) :
  (∃ x y, curve x y ∧ a_b_relation a b) →
  (∀ x y, ellipse a b x y → circle_condition x y) →
  π / 6 ≤ Real.arccos a ∧ Real.arccos a ≤ π / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_theta_l594_59494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_increase_in_2000_l594_59473

def sales : ℕ → ℕ
  | 1994 => 30
  | 1995 => 36
  | 1996 => 45
  | 1997 => 50
  | 1998 => 65
  | 1999 => 70
  | 2000 => 88
  | 2001 => 90
  | 2002 => 85
  | 2003 => 75
  | _ => 0

def salesIncrease (year : ℕ) : ℤ :=
  (sales year : ℤ) - (sales (year - 1) : ℤ)

theorem largest_increase_in_2000 :
  ∀ y ∈ Finset.range 9,
    salesIncrease (1995 + y) ≤ salesIncrease 2000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_increase_in_2000_l594_59473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_and_circle_l594_59493

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -1

-- Define the circle
def tangent_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

-- Theorem statement
theorem parabola_directrix_and_circle :
  (∀ x y, parabola x y → 
    (directrix x ↔ x = -1) ∧ 
    (tangent_circle x y ↔ (x - 1)^2 + y^2 = 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_and_circle_l594_59493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_distances_imply_sqrt2_l594_59469

-- Define the set of valid x values
def ValidX : Set ℝ := {x : ℝ | x < 0 ∨ x > 0}

-- Define the distance functions
noncomputable def dist1 (x a b : ℝ) : ℝ := Real.sqrt ((x - a)^2 + (1/x - b)^2)
noncomputable def dist2 (x a b : ℝ) : ℝ := Real.sqrt ((x + a)^2 + (1/x + b)^2)

-- State the theorem
theorem constant_distances_imply_sqrt2 (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (c d : ℝ), ∀ x ∈ ValidX, dist1 x a b = c ∧ dist2 x a b = d) →
  a = Real.sqrt 2 ∧ b = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_distances_imply_sqrt2_l594_59469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_prize_probability_l594_59428

/-- The probability of getting a prize in a lottery with 10 prizes and 25 blanks is 2/7. -/
theorem lottery_prize_probability : 
  (10 : ℚ) / (10 + 25) = 2 / 7 :=
by
  -- Convert the fraction to decimals for comparison
  have h1 : (10 : ℚ) / (10 + 25) = 10 / 35 := by norm_num
  have h2 : (2 : ℚ) / 7 = 10 / 35 := by norm_num
  -- Show that both fractions are equal
  rw [h1, h2]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_prize_probability_l594_59428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l594_59482

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) :
  (1 / a + 6 * b) ^ (1/3) + (1 / b + 6 * c) ^ (1/3) + (1 / c + 6 * a) ^ (1/3) ≤ 1 / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l594_59482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_koschei_cant_escape_l594_59401

/-- Represents the side of the corridor a guard is leaning against -/
inductive Side
| West
| East

/-- Represents the four rooms in the corridor -/
inductive Room
| One
| Two
| Three
| Four

/-- Represents the state of the guards and Koschei's position -/
structure GameState where
  guard1 : Side
  guard2 : Side
  guard3 : Side
  koschei : Room

/-- Flips a guard's side -/
def flipSide (s : Side) : Side :=
  match s with
  | Side.West => Side.East
  | Side.East => Side.West

/-- Updates the game state when Koschei moves -/
def moveKoschei (state : GameState) (newRoom : Room) : GameState :=
  match state.koschei, newRoom with
  | Room.One, Room.Two => { state with guard1 := flipSide state.guard1, koschei := newRoom }
  | Room.Two, Room.Three => { state with guard1 := flipSide state.guard1, guard2 := flipSide state.guard2, koschei := newRoom }
  | Room.Three, Room.Four => { state with guard1 := flipSide state.guard1, guard2 := flipSide state.guard2, guard3 := flipSide state.guard3, koschei := newRoom }
  | Room.Two, Room.One => { state with guard1 := flipSide state.guard1, koschei := newRoom }
  | Room.Three, Room.Two => { state with guard1 := flipSide state.guard1, guard2 := flipSide state.guard2, koschei := newRoom }
  | Room.Four, Room.Three => { state with guard1 := flipSide state.guard1, guard2 := flipSide state.guard2, guard3 := flipSide state.guard3, koschei := newRoom }
  | _, _ => state  -- Invalid moves return the same state

/-- Checks if all guards are on the same side -/
def allGuardsSameSide (state : GameState) : Prop :=
  (state.guard1 = state.guard2) ∧ (state.guard2 = state.guard3)

/-- Theorem stating that there exists an initial configuration where Koschei can never escape -/
theorem koschei_cant_escape : ∃ (initialState : GameState),
  ∀ (moves : List Room),
    let finalState := moves.foldl moveKoschei initialState
    ¬(allGuardsSameSide finalState) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_koschei_cant_escape_l594_59401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_range_l594_59485

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 9 then Real.log x / Real.log 3 - 1
  else if x > 9 then 4 - Real.sqrt x
  else 0  -- This case should never occur in our problem

-- State the theorem
theorem abc_range (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (hf : f a = f b ∧ f b = f c) :
  81 < a * b * c ∧ a * b * c < 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_range_l594_59485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_ratios_l594_59400

def projection_matrix : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![1/14, -5/14, 0],
    ![-5/14, 24/14, 0],
    ![0, 0, 1]]

theorem projection_ratios (x y z : ℚ) (h : x ≠ 0) :
  projection_matrix.mulVec ![x, y, z] = ![x, y, z] →
  y / x = 13 / 5 ∧ z / x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_ratios_l594_59400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dandelion_color_change_l594_59442

/-- Represents the number of dandelions that turn white on a given day -/
def dandelions_turned_white (day : Nat) : Nat := sorry

/-- The total number of dandelions that were yellow two days ago -/
def total_yellow_dandelions : Nat := 25

/-- The number of dandelions that will turn white tomorrow -/
def will_turn_white_tomorrow : Nat := 9

/-- The number of dandelions that turned white yesterday -/
def turned_white_yesterday : Nat := dandelions_turned_white 1

/-- The number of dandelions that turned white today -/
def turned_white_today : Nat := dandelions_turned_white 2

theorem dandelion_color_change : 
  total_yellow_dandelions + will_turn_white_tomorrow = 
  turned_white_yesterday + turned_white_today + will_turn_white_tomorrow :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dandelion_color_change_l594_59442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_circle_l594_59415

/-- The line equation: 2x - y + a = 0 -/
def line_equation (x y a : ℝ) : Prop := 2 * x - y + a = 0

/-- The circle equation: x² + y² - 4x + 6y - 12 = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y - 12 = 0

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem intersection_line_circle (a : ℝ) :
  (a > -5) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    line_equation x₁ y₁ a ∧ 
    line_equation x₂ y₂ a ∧ 
    circle_equation x₁ y₁ ∧ 
    circle_equation x₂ y₂ ∧ 
    distance x₁ y₁ x₂ y₂ = 4 * Real.sqrt 5) →
  a = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_circle_l594_59415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l594_59431

def our_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 2) = 1 / (a n + 1)

theorem sequence_sum (a : ℕ → ℚ) (h : our_sequence a) (h100 : a 100 = a 96) :
  ∃ (x : ℚ), x * x = 5 ∧ (a 15 + a 16 = (4 + x) / 34 ∨ a 15 + a 16 = (4 - x) / 34) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_l594_59431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_interval_l594_59492

-- Define the function f(x) = x + cos(x)
noncomputable def f (x : ℝ) : ℝ := x + Real.cos x

-- State the theorem
theorem max_value_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧
  f x = (Real.pi / 2) ∧
  ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → f y ≤ f x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_interval_l594_59492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_2001_l594_59460

/-- The function f(k) = k^2 / 1.001^k -/
noncomputable def f (k : ℕ) : ℝ := (k^2 : ℝ) / (1.001^k)

/-- The maximum value of f(k) occurs at k = 2001 -/
theorem f_max_at_2001 : ∀ n : ℕ, f n ≤ f 2001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_at_2001_l594_59460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_ratio_theorem_l594_59471

/-- The ratio of the area of a regular octagon circumscribed about a circle 
    to the area of a regular octagon inscribed in the same circle -/
noncomputable def octagon_area_ratio : ℝ := 4 - 2 * Real.sqrt 2

/-- Theorem stating that the ratio of the areas of circumscribed and inscribed 
    regular octagons around a circle is equal to 4 - 2√2 -/
theorem octagon_area_ratio_theorem :
  let r : ℝ := 1  -- Radius of the circle (can be any positive real number)
  let inscribed_side : ℝ := 2 * r * Real.cos (π / 8)  -- Side length of inscribed octagon
  let circumscribed_side : ℝ := 2 * r  -- Side length of circumscribed octagon
  (circumscribed_side / inscribed_side) ^ 2 = octagon_area_ratio :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_ratio_theorem_l594_59471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_four_kg_l594_59459

-- Define the yield function W(x)
noncomputable def W (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then 5 * (x^2 + 3)
  else if 2 < x ∧ x ≤ 5 then 50 - 50 / (x + 1)
  else 0

-- Define the profit function f(x)
noncomputable def f (x : ℝ) : ℝ := 15 * W x - 30 * x

-- State the theorem
theorem max_profit_at_four_kg (x : ℝ) :
  0 ≤ x ∧ x ≤ 5 → f x ≤ 480 ∧ f 4 = 480 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_four_kg_l594_59459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_diff_inequality_no_negative_roots_l594_59456

-- Part 1
theorem sqrt_diff_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by sorry

-- Part 2
noncomputable def f (x : ℝ) : ℝ := Real.exp x + (x - 2) / (x + 1)

theorem no_negative_roots :
  ∀ x : ℝ, x < 0 → f x ≠ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_diff_inequality_no_negative_roots_l594_59456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l594_59450

def A : Set ℝ := {x : ℝ | x^2 ≥ 16}
def B (m : ℝ) : Set ℝ := {m}

theorem m_range (m : ℝ) (h : A ∪ B m = A) : m ∈ Set.Iic (-4) ∪ Set.Ici 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l594_59450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ratio_l594_59435

def my_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 5 ∧ ∀ n : ℕ, n ≥ 1 → a n * a (n + 1) = 2^n

theorem sequence_ratio (a : ℕ → ℝ) (h : my_sequence a) : a 7 / a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_ratio_l594_59435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_eq_expected_l594_59478

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The radius of the circle in feet -/
noncomputable def r : ℝ := 50

/-- The angle between adjacent points on the circle -/
noncomputable def θ : ℝ := 2 * Real.pi / n

/-- The chord length between two points separated by k positions -/
noncomputable def chord_length (k : ℕ) : ℝ := 2 * r * Real.sin (k * θ / 2)

/-- The total distance traveled by one point to all non-adjacent points and back -/
noncomputable def distance_per_point : ℝ := 2 * (chord_length 2 + chord_length 3 + chord_length 4)

/-- The total distance traveled by all points -/
noncomputable def total_distance : ℝ := n * distance_per_point

/-- The theorem stating the total distance traveled -/
theorem total_distance_eq_expected : total_distance = 2400 * Real.sqrt (2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_eq_expected_l594_59478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_subset_implies_m_range_l594_59453

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - (2*m-3)*x + m^2 - 3*m ≤ 0}

-- Theorem for the first question
theorem intersection_implies_m_value (m : ℝ) :
  A ∩ B m = Set.Icc 2 4 → m = 5 := by sorry

-- Theorem for the second question
theorem subset_implies_m_range (m : ℝ) :
  A ⊆ (Set.univ \ B m) → m < -2 ∨ m > 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_subset_implies_m_range_l594_59453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l594_59447

/-- An ellipse with center (6, 3), semi-major axis 6, and semi-minor axis 3 -/
structure Ellipse where
  center : ℝ × ℝ
  semi_major_axis : ℝ
  semi_minor_axis : ℝ
  h_center : center = (6, 3)
  h_semi_major : semi_major_axis = 6
  h_semi_minor : semi_minor_axis = 3
  h_axes : semi_major_axis > semi_minor_axis

/-- The distance between the foci of an ellipse -/
noncomputable def foci_distance (e : Ellipse) : ℝ :=
  2 * Real.sqrt (e.semi_major_axis^2 - e.semi_minor_axis^2)

/-- Theorem stating that the distance between the foci of the given ellipse is 6√3 -/
theorem ellipse_foci_distance (e : Ellipse) : foci_distance e = 6 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l594_59447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sqrt2_range_l594_59470

theorem log_sqrt2_range (a : ℝ) : 
  (0 < a → Real.log (Real.sqrt 2) / Real.log a < 1) ↔ (0 < a ∧ a < 1) ∨ a > Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sqrt2_range_l594_59470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l594_59497

-- Define the curves C₁ and C₂
noncomputable def C₁ (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 6

-- Define point P
def P : ℝ × ℝ := (1, 0)

-- Define the intersection points A and B
def isIntersection (p : ℝ × ℝ) : Prop := C₁ p.1 p.2 ∧ C₂ p.1 p.2

-- Define the distance function
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- Theorem statement
theorem intersection_product :
  ∃ (A B : ℝ × ℝ), isIntersection A ∧ isIntersection B ∧ A ≠ B →
  distance P A * distance P B = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l594_59497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l594_59416

-- Define the sets M and N
def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | Real.rpow 3 x > 1/3}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l594_59416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l594_59405

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := Real.sqrt 3 * x - y - 3 = 0

-- Define the slope product condition
def slope_product_condition (C : ℝ → ℝ → Prop) : Prop :=
  ∀ x₀ y₀ x₁ y₁, C x₀ y₀ → C x₁ y₁ → C (-x₁) (-y₁) →
    ((y₁ - y₀) / (x₁ - x₀)) * ((y₁ + y₀) / (x₁ + x₀)) = -1/4

-- Define the distance ratio condition
def distance_ratio_condition (C : ℝ → ℝ → Prop) (x₀ : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x₁ y₁ x₂ y₂,
    C x₁ y₁ → C x₂ y₂ → y₁ = k * (x₁ - 1) → y₂ = k * (x₂ - 1) →
    (x₀ - x₁) / (x₀ - x₂) = (x₁ - 1) / (1 - x₂)

theorem ellipse_theorem (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  (∃ c : ℝ, c > 0 ∧ line_l c 0) →
  slope_product_condition (ellipse_C a b) →
  (∃ C : ℝ → ℝ → Prop, C = ellipse_C 2 1) ∧
  (∃ x₀ : ℝ, x₀ = 4 ∧ distance_ratio_condition (ellipse_C 2 1) x₀) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l594_59405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_l_l594_59464

/-- A line passing through the origin -/
structure Line where
  slope : ℝ

/-- The angle between two lines in degrees -/
noncomputable def angle_between (l1 l2 : Line) : ℝ := sorry

/-- The slope angle of a line in degrees -/
noncomputable def slope_angle (l : Line) : ℝ := sorry

/-- The given line y = (√3/3)x + 1 -/
noncomputable def given_line : Line :=
  { slope := Real.sqrt 3 / 3 }

theorem slope_angle_of_l (l : Line) :
  (angle_between l given_line = 30) →
  (slope_angle l = 0 ∨ slope_angle l = 60) := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_l_l594_59464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l594_59483

-- Define the floor function (marked as noncomputable)
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the problem statement
theorem problem_statement (x y : ℝ) : 
  (y = 2 * (floor x) + 3) → 
  (y = 3 * (floor (x - 2)) + 5) → 
  (∃ (n : ℤ), x > ↑n ∧ x < ↑n + 1) →
  (15 < x + y ∧ x + y < 16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l594_59483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_speed_theorem_l594_59445

noncomputable section

-- Define the variables and constants
def swim_distance : ℝ := 2
def bike_distance : ℝ := 30
def run_distance : ℝ := 8
def total_time : ℝ := 200
def break_time : ℝ := 20

-- Define the speeds as functions of x
def run_speed (x : ℝ) : ℝ := x
def swim_speed (x : ℝ) : ℝ := x / 2
def bike_speed (x : ℝ) : ℝ := 3 * x + 2

-- Define a function for approximate equality
def approx_equal (a b : ℝ) : Prop := abs (a - b) < 0.01

-- Define the theorem
theorem triathlon_speed_theorem :
  ∃ x : ℝ, 
    (swim_distance / swim_speed x + 
     bike_distance / bike_speed x + 
     run_distance / run_speed x = total_time - break_time) ∧
    (approx_equal (run_speed x) 7.04) ∧
    (approx_equal (swim_speed x) 3.52) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_speed_theorem_l594_59445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l594_59429

/-- M(x, y) denotes the larger of x and y -/
noncomputable def M (x y : ℝ) : ℝ := max x y

/-- m(x, y) denotes the smaller of x and y -/
noncomputable def m (x y : ℝ) : ℝ := min x y

/-- Given distinct real numbers p, q, r, s, t with p < q < r < s < t,
    prove that M(m(M(p, q), r), M(s, m(t, p))) = s -/
theorem problem_statement (p q r s t : ℝ)
    (h1 : p < q) (h2 : q < r) (h3 : r < s) (h4 : s < t)
    (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
                  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
                  r ≠ s ∧ r ≠ t ∧
                  s ≠ t) :
  M (m (M p q) r) (M s (m t p)) = s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l594_59429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_not_periodic_l594_59421

/-- The function f(x) = cos x + cos(x√2) is not periodic -/
theorem cosine_sum_not_periodic :
  ¬ ∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), Real.cos x + Real.cos (x * Real.sqrt 2) = 
    Real.cos (x + T) + Real.cos ((x + T) * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_not_periodic_l594_59421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unequal_outcome_probability_l594_59418

theorem unequal_outcome_probability (n : ℕ) (p : ℚ) :
  n = 12 →
  p = 1 / 2 →
  (Finset.sum (Finset.range (n + 1)) (λ k ↦ if k = n / 2 then 0 else (n.choose k) * p^k * (1 - p)^(n - k))) = 793 / 1024 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unequal_outcome_probability_l594_59418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l594_59468

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x + 1

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := Real.exp x

-- Theorem statement
theorem tangent_line_at_zero :
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  (fun x y ↦ y = m * (x - x₀) + y₀) = (fun x y ↦ y = x + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l594_59468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_higher_prob_is_quarter_l594_59426

/-- Probability of a ball landing in bin k -/
noncomputable def prob (k : ℕ) : ℝ :=
  if k % 2 = 0 then (1 / 2) * (3 ^ (-k : ℤ)) else 3 ^ (-k : ℤ)

/-- The probability that a ball lands in any bin -/
noncomputable def totalProb : ℝ := ∑' k, prob k

/-- The probability that the red ball lands in a higher-numbered bin than the blue ball -/
noncomputable def redHigherProb : ℝ := (1 - totalProb) / 2

theorem red_higher_prob_is_quarter :
  redHigherProb = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_higher_prob_is_quarter_l594_59426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_triangle_properties_l594_59425

/-- An equilateral triangle with extensions --/
structure ExtendedEquilateralTriangle where
  a : ℝ  -- Side length of the original equilateral triangle
  x : ℝ  -- Length of the extensions
  h_positive : 0 < a  -- Assumption that side length is positive

/-- The side length of the new triangle formed by the extensions --/
noncomputable def sideLength (t : ExtendedEquilateralTriangle) : ℝ :=
  Real.sqrt (2 * t.x^2 + t.a^2 - 2 * t.a * t.x)

/-- Theorem stating that the new triangle is equilateral and its side length can be chosen --/
theorem extended_triangle_properties (t : ExtendedEquilateralTriangle) (l : ℝ) 
    (h_l : l ≥ t.a / 2) :
  -- The new triangle is equilateral
  ∃ (s : ℝ), s = sideLength t ∧ 
    -- The side length of the new triangle can be chosen
    ∃ (x : ℝ), t.x = x ∧ s = l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_triangle_properties_l594_59425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_theorem_l594_59457

-- Define the trapezoid
structure Trapezoid where
  R : ℝ
  leg_perpendicular : ℝ
  segment_ratio : Fin 3 → ℝ

-- Define the conditions
def trapezoid_conditions (t : Trapezoid) : Prop :=
  t.leg_perpendicular = 2 * t.R ∧
  t.segment_ratio 0 = 7 ∧
  t.segment_ratio 1 = 21 ∧
  t.segment_ratio 2 = 27

-- Define the area function
noncomputable def trapezoid_area (t : Trapezoid) : ℝ :=
  (100 * t.R^2) / (11 * Real.sqrt 21)

-- State the theorem
theorem trapezoid_area_theorem (t : Trapezoid) :
  trapezoid_conditions t → trapezoid_area t = (100 * t.R^2) / (11 * Real.sqrt 21) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_theorem_l594_59457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_to_F₂_l594_59436

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1

-- Define the foci
noncomputable def F₁ : ℝ × ℝ := sorry
noncomputable def F₂ : ℝ × ℝ := sorry

-- Define points A and B on the ellipse
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Assumptions
axiom A_on_ellipse : is_on_ellipse A.1 A.2
axiom B_on_ellipse : is_on_ellipse B.1 B.2
axiom line_through_F₁ : ∃ (t : ℝ), A = F₁ + t • (B - F₁)
axiom AB_distance : ‖A - B‖ = 6

-- Theorem to prove
theorem sum_distances_to_F₂ : ‖A - F₂‖ + ‖B - F₂‖ = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distances_to_F₂_l594_59436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_exp_graph_l594_59455

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the rotation transformation
def rotate_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- State the theorem
theorem rotated_exp_graph (x : ℝ) : 
  (rotate_180 (x, f x)).2 = -f (-x) := by
  -- Expand the definition of rotate_180
  simp [rotate_180, f]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_exp_graph_l594_59455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_box_height_is_six_l594_59461

/-- The height of rectangular boxes that maximally fill a wooden box. -/
noncomputable def rectangular_box_height : ℝ :=
  let wooden_box_volume : ℝ := 8 * 100 * 7 * 100 * 6 * 100  -- in cubic cm
  let rectangular_box_base_area : ℝ := 4 * 7  -- in square cm
  let max_boxes : ℝ := 2000000
  wooden_box_volume / (rectangular_box_base_area * max_boxes)

/-- Theorem stating that the height of the rectangular boxes is 6 cm. -/
theorem rectangular_box_height_is_six :
  rectangular_box_height = 6 := by
  -- Unfold the definition of rectangular_box_height
  unfold rectangular_box_height
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_box_height_is_six_l594_59461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l594_59409

theorem trigonometric_simplification (x : ℝ) :
  (Real.cos (2*x + π/2) * Real.sin (3*π/2 - 3*x) - Real.cos (2*x - 5*π) * Real.cos (3*x + 3*π/2)) /
  (Real.sin (5*π/2 - x) * Real.cos (4*x) + Real.sin x * Real.cos (5*π/2 + 4*x)) = Real.tan (5*x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l594_59409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_problem_l594_59433

theorem restaurant_bill_problem :
  ∃ (older younger : ℕ), 
    older > younger ∧ 
    older > 0 ∧ 
    younger > 0 ∧
    (5.50 : ℝ) + 0.55 * ((older : ℝ) + (younger : ℝ)) = 12.10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_restaurant_bill_problem_l594_59433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_five_boxes_3m_l594_59491

def box_volume (edge_length : ℝ) : ℝ := edge_length ^ 3

def total_volume (edge_length : ℝ) (num_boxes : ℕ) : ℝ :=
  (box_volume edge_length) * num_boxes

theorem total_volume_five_boxes_3m :
  total_volume 3 5 = 135 := by
  unfold total_volume
  unfold box_volume
  simp
  norm_num

#eval total_volume 3 5

/- Proof sketch:
1. Define the volume of a single box (cube) as edge_length^3
2. Define the total volume as the volume of one box multiplied by the number of boxes
3. Prove that for 5 boxes with edge length 3, the total volume is 135 cubic meters
4. Use Lean's simplification and numerical normalization to complete the proof
-/

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_volume_five_boxes_3m_l594_59491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_ellipse_l594_59462

-- Define the space we're working in
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

-- Define the fixed points A and B
variable (A B : E)

-- Define the set of points P satisfying the condition
def ellipse_set (A B : E) : Set E :=
  {P : E | ‖P - A‖ + ‖P - B‖ = 2 * ‖B - A‖ + 10}

-- Theorem statement
theorem is_ellipse (A B : E) : 
  ∃ (C : E) (a b : ℝ), a > b ∧ b > 0 ∧
    ellipse_set A B = {P : E | ‖P - C‖^2 / a^2 + ‖P - (C + B - A)‖^2 / b^2 = 1} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_ellipse_l594_59462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_must_be_88_l594_59498

def scores : List ℕ := [52, 61, 67, 72, 77, 88]

def is_integer (n : ℚ) : Prop := ∃ m : ℤ, n = m

def average_is_integer (entered_scores : List ℕ) : Prop :=
  ∀ k : ℕ, k ≤ entered_scores.length → 
    is_integer ((entered_scores.take k).sum / k)

theorem last_score_must_be_88 (perm : List ℕ) 
  (h_perm : perm.Perm scores) 
  (h_avg : average_is_integer perm) : 
  perm.getLast? = some 88 := by
  sorry

#check last_score_must_be_88

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_must_be_88_l594_59498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l594_59414

/-- Represents a parabola with equation y^2 = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a point on the parabola -/
structure PointOnParabola (para : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * para.p * x

/-- The focus of a parabola -/
noncomputable def focus (para : Parabola) : ℝ × ℝ := (para.p / 2, 0)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem to be proved -/
theorem parabola_directrix 
  (para : Parabola) 
  (M : PointOnParabola para) 
  (h_M_x : M.x = 4)
  (h_dist : distance (M.x, M.y) (focus para) = 6) :
  para.p = 4 ∧ -para.p / 2 = -2 := by
  sorry

#check parabola_directrix

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l594_59414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l594_59420

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < Real.pi) →
  (0 < B ∧ B < Real.pi) →
  (0 < C ∧ C < Real.pi) →
  A + B + C = Real.pi →
  -- Given conditions
  Real.sin C = 56 / 65 →
  Real.sin B = 12 / 13 →
  b = 3 →
  -- Conclusion
  c = 14 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l594_59420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_7digit_properties_l594_59423

def is_symmetric_digit (d : Nat) : Bool :=
  d = 0 ∨ d = 1 ∨ d = 8

def is_symmetric_pair (d1 d2 : Nat) : Bool :=
  (is_symmetric_digit d1 ∧ d1 = d2) ∨ (d1 = 6 ∧ d2 = 9) ∨ (d1 = 9 ∧ d2 = 6)

def is_symmetric_7digit (n : Nat) : Bool :=
  let digits := n.digits 10
  digits.length = 7 ∧
  is_symmetric_pair digits[0]! digits[6]! ∧
  is_symmetric_pair digits[1]! digits[5]! ∧
  is_symmetric_pair digits[2]! digits[4]! ∧
  is_symmetric_digit digits[3]!

def count_symmetric_7digit : Nat :=
  (List.range 9000000).filter is_symmetric_7digit |>.length

def count_symmetric_7digit_div4 : Nat :=
  (List.range 9000000).filter (λ n => is_symmetric_7digit n ∧ n % 4 = 0) |>.length

def sum_symmetric_7digit : Nat :=
  (List.range 9000000).filter is_symmetric_7digit |>.sum

theorem symmetric_7digit_properties :
  count_symmetric_7digit = 300 ∧
  count_symmetric_7digit_div4 = 75 ∧
  sum_symmetric_7digit = 1959460200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_7digit_properties_l594_59423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_belfried_tax_payment_l594_59410

/-- Calculates the special municipal payroll tax -/
noncomputable def municipal_payroll_tax (payroll : ℝ) : ℝ :=
  if payroll ≤ 200000 then 0
  else (payroll - 200000) * 0.002

/-- Theorem: Belfried Industries' tax payment -/
theorem belfried_tax_payment :
  municipal_payroll_tax 300000 = 200 := by
  -- Unfold the definition of municipal_payroll_tax
  unfold municipal_payroll_tax
  -- Simplify the if-then-else expression
  simp
  -- Perform the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_belfried_tax_payment_l594_59410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_inequality_l594_59444

theorem log_product_inequality (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : 0 < b) (hb' : b < 1) :
  Real.log (2 * a * b / (a + b)) / Real.log a * (Real.log (2 * a * b / (a + b)) / Real.log b) ≥ 1 ∧
  (Real.log (2 * a * b / (a + b)) / Real.log a * (Real.log (2 * a * b / (a + b)) / Real.log b) = 1 ↔ a = b) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_product_inequality_l594_59444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_proof_l594_59496

def NonNegativeIntegers : Set ℕ := {n : ℕ | n ≥ 0}

theorem constant_function_proof (f : ℕ → ℕ) 
  (h : ∀ x y : ℕ, x * f y + y * f x = (x + y) * f (x^2 + y^2)) :
  ∃ c : ℕ, ∀ x : ℕ, f x = c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_proof_l594_59496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_output_27_implies_input_5_5_l594_59408

theorem output_27_implies_input_5_5 (x : ℝ) : 
  let y := ⌊x⌋
  let z := (2 : ℝ)^y - y
  z = 27 → x = 5.5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_output_27_implies_input_5_5_l594_59408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_subset_complement_iff_m_range_l594_59403

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 8 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - (2*m - 3)*x + m^2 - 3*m ≤ 0}

-- Theorem for the first question
theorem intersection_implies_m_value :
  (∃ m : ℝ, A ∩ B m = Set.Icc 2 4) → (∃ m : ℝ, m = 5) :=
sorry

-- Theorem for the second question
theorem subset_complement_iff_m_range :
  ∀ m : ℝ, A ⊆ (B m)ᶜ ↔ m ∈ Set.Iic (-2) ∪ Set.Ioi 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_implies_m_value_subset_complement_iff_m_range_l594_59403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_volume_approx_l594_59432

/-- Regular pentagon side length -/
noncomputable def pentagonSide : ℝ := 1

/-- Angle of a regular pentagon in radians -/
noncomputable def pentagonAngle : ℝ := 2 * Real.pi / 5

/-- Radius of the circumcircle of the regular pentagon -/
noncomputable def pentagonRadius : ℝ := pentagonSide / (2 * Real.sin (pentagonAngle / 2))

/-- Diagonal of the regular pentagon -/
noncomputable def pentagonDiagonal : ℝ := pentagonSide * (1 + 2 * Real.sin (3 * pentagonAngle / 2))

/-- Side length of the inscribed cube -/
noncomputable def cubeSide : ℝ := pentagonDiagonal / (2 * Real.sqrt 2)

/-- Volume of the inscribed cube -/
noncomputable def cubeVolume : ℝ := cubeSide ^ 3

/-- Theorem stating the approximate volume of the inscribed cube -/
theorem inscribed_cube_volume_approx :
  |cubeVolume - 1.077| < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_volume_approx_l594_59432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_intersecting_lines_l594_59413

/-- A line passing through (0, 4) that intersects the hyperbola y^2 - 4x^2 = 16 at exactly one point -/
def IntersectingLine : Type := ℝ → ℝ

/-- The hyperbola y^2 - 4x^2 = 16 -/
def Hyperbola (x y : ℝ) : Prop := y^2 - 4*x^2 = 16

/-- A line passes through (0, 4) -/
def PassesThroughPoint (l : IntersectingLine) : Prop := l 0 = 4

/-- A line intersects the hyperbola at exactly one point -/
def IntersectsOnce (l : IntersectingLine) : Prop :=
  ∃! p : ℝ × ℝ, Hyperbola p.1 p.2 ∧ l p.1 = p.2

/-- There are exactly 3 lines passing through (0, 4) that intersect the hyperbola at one point -/
theorem three_intersecting_lines :
  ∃! (s : Finset IntersectingLine), (∀ l ∈ s, PassesThroughPoint l ∧ IntersectsOnce l) ∧ s.card = 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_intersecting_lines_l594_59413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_2n_formula_l594_59404

def sequence_a : ℕ → ℕ
  | 0 => 1  -- Add this case for 0
  | 1 => 1
  | n + 1 => if n % 2 = 1 then sequence_a n + 2 else 3 * sequence_a n

def sum_2n (n : ℕ) : ℕ :=
  (List.range (2 * n)).foldl (λ acc i => acc + sequence_a (i + 1)) 0

theorem sum_2n_formula (n : ℕ) :
  sum_2n n = 4 * 3^n - 4 * n - 4 := by
  sorry

#eval sum_2n 5  -- Example evaluation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_2n_formula_l594_59404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_democrat_vote_percentage_is_75_percent_l594_59465

/-- Represents the percentage of registered voters who are Democrats -/
noncomputable def democrat_percentage : ℝ := 0.6

/-- Represents the percentage of registered voters who are Republicans -/
noncomputable def republican_percentage : ℝ := 1 - democrat_percentage

/-- Represents the percentage of Republican voters expected to vote for candidate A -/
noncomputable def republican_vote_percentage : ℝ := 0.3

/-- Represents the total percentage of registered voters expected to vote for candidate A -/
noncomputable def total_vote_percentage : ℝ := 0.57

/-- Represents the percentage of Democrat voters expected to vote for candidate A -/
noncomputable def democrat_vote_percentage : ℝ := (total_vote_percentage - republican_percentage * republican_vote_percentage) / democrat_percentage

theorem democrat_vote_percentage_is_75_percent :
  democrat_vote_percentage = 0.75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_democrat_vote_percentage_is_75_percent_l594_59465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l594_59467

/-- The time taken for a train to cross a stationary point -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  train_length / (train_speed_kmh * 1000 / 3600)

/-- Theorem: A 160-meter long train moving at 36 km/h takes 16 seconds to cross a stationary point -/
theorem train_crossing_theorem :
  train_crossing_time 160 36 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l594_59467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_equals_two_l594_59452

-- Define the slopes of the two lines
noncomputable def slope1 : ℝ := -3
noncomputable def slope2 (m : ℝ) : ℝ := -6 / m

-- Define the condition for parallel lines
def parallel (m : ℝ) : Prop := slope1 = slope2 m

-- Theorem statement
theorem parallel_lines_m_equals_two :
  ∀ m : ℝ, parallel m → m = 2 :=
by
  intro m h
  -- The proof goes here
  sorry

#check parallel_lines_m_equals_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_equals_two_l594_59452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_over_b_l594_59479

open Real

theorem max_x_over_b (e₁ e₂ b : ℝ × ℝ) (x y : ℝ) :
  ‖e₁‖ = 1 →
  ‖e₂‖ = 1 →
  b = (x * e₁.1 + y * e₂.1, x * e₁.2 + y * e₂.2) →
  b ≠ (0, 0) →
  e₁.1 * e₂.1 + e₁.2 * e₂.2 = cos (π / 6) →
  (∀ x' y' : ℝ, |x'| / ‖(x' * e₁.1 + y' * e₂.1, x' * e₁.2 + y' * e₂.2)‖ ≤ 2) ∧
  (∃ x₀ y₀ : ℝ, |x₀| / ‖(x₀ * e₁.1 + y₀ * e₂.1, x₀ * e₁.2 + y₀ * e₂.2)‖ = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_over_b_l594_59479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_exponent_cancellation_l594_59490

theorem fraction_exponent_cancellation :
  (5/6 : ℚ)^4 * (5/6 : ℚ)^(-4 : ℤ) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_exponent_cancellation_l594_59490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_egg_distribution_l594_59488

theorem candy_egg_distribution (sofia mia pablo : ℕ) : 
  mia = 4 * sofia →
  pablo = 2 * mia →
  (1 : ℚ) / 24 = ((pablo + mia + sofia) / 3 - mia) / pablo := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_egg_distribution_l594_59488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_at_given_height_max_volume_l594_59438

/-- Represents a quadrangular pyramid with specific properties -/
structure QuadrangularPyramid where
  -- Base is a convex quadrilateral ABCD
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  -- Each lateral face forms a 45° angle with the base
  lateral_angle : ℝ
  -- Height of the pyramid
  height : ℝ
  -- Conditions
  ab_eq_bc : AB = 1
  cd_eq_da : CD = 2
  lateral_angle_eq : lateral_angle = π / 4

/-- Calculate the volume of the pyramid -/
noncomputable def volume (p : QuadrangularPyramid) : ℝ :=
  (p.AB * p.DA * p.height) / 3

/-- Theorem stating the volume when height is 9/5 -/
theorem volume_at_given_height (p : QuadrangularPyramid) 
  (h : p.height = 9/5) : volume p = 27/25 := by
  sorry

/-- Theorem stating the maximum volume and corresponding height -/
theorem max_volume (p : QuadrangularPyramid) :
  ∃ (h : ℝ), h = 2 ∧ volume { p with height := h } = 4/3 ∧
  ∀ (h' : ℝ), volume { p with height := h' } ≤ 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_at_given_height_max_volume_l594_59438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l594_59458

noncomputable def f (x : ℝ) : ℝ := (9*x^2 + 27*x - 64) / ((3*x - 4)*(x + 5)*(x-1))

theorem inequality_solution_set :
  {x : ℝ | f x < 4} = {x : ℝ | -5 < x ∧ x < -17/3} ∪ {x : ℝ | 1 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l594_59458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rock_band_airplane_refueling_cost_l594_59427

theorem rock_band_airplane_refueling_cost : 
  let num_small_planes : ℕ := 2
  let num_large_planes : ℕ := 2
  let small_tank_capacity : ℝ := 60
  let large_tank_capacity : ℝ := small_tank_capacity * 1.5
  let fuel_cost_per_liter : ℝ := 0.5
  let service_charge_per_plane : ℝ := 100
  let total_fuel_capacity : ℝ := num_small_planes * small_tank_capacity + num_large_planes * large_tank_capacity
  let total_fuel_cost : ℝ := total_fuel_capacity * fuel_cost_per_liter
  let total_service_charge : ℝ := (num_small_planes + num_large_planes) * service_charge_per_plane
  let total_cost : ℝ := total_fuel_cost + total_service_charge
  total_cost = 550 := by
    -- Proof goes here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rock_band_airplane_refueling_cost_l594_59427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_BAC_l594_59430

noncomputable def vector_AB : Fin 2 → ℝ := ![(-1 : ℝ), Real.sqrt 3]
noncomputable def vector_AC : Fin 2 → ℝ := ![(1 : ℝ), Real.sqrt 3]

theorem angle_BAC : 
  let dot_product := (vector_AB 0) * (vector_AC 0) + (vector_AB 1) * (vector_AC 1)
  let magnitude_AB := Real.sqrt ((vector_AB 0)^2 + (vector_AB 1)^2)
  let magnitude_AC := Real.sqrt ((vector_AC 0)^2 + (vector_AC 1)^2)
  let cos_angle := dot_product / (magnitude_AB * magnitude_AC)
  cos_angle = 1/2 ∧ Real.arccos cos_angle = π/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_BAC_l594_59430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hot_dogs_served_lunch_l594_59451

def hot_dogs_served_today : ℕ := 11
def hot_dogs_served_dinner : ℕ := 2

theorem hot_dogs_served_lunch : 
  hot_dogs_served_today - hot_dogs_served_dinner = 9 := by
  -- Proof goes here
  sorry

def hot_dogs_served_lunch_value : ℕ := hot_dogs_served_today - hot_dogs_served_dinner

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hot_dogs_served_lunch_l594_59451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirty_five_power_mn_l594_59446

theorem thirty_five_power_mn (m n : ℤ) (P Q : ℕ) 
  (h1 : P = (5 : ℕ)^(m.natAbs)) (h2 : Q = (7 : ℕ)^(n.natAbs)) : 
  (35 : ℕ)^(m.natAbs * n.natAbs) = P^(n.natAbs) * Q^(m.natAbs) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirty_five_power_mn_l594_59446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_with_right_angle_is_rectangle_l594_59454

-- Define a parallelogram
structure Parallelogram where
  -- Add necessary properties of a parallelogram
  parallelSides : Bool

-- Define a rectangle
structure Rectangle extends Parallelogram where
  -- Add necessary properties of a rectangle
  rightAngles : Bool

-- Define the theorem
theorem parallelogram_with_right_angle_is_rectangle 
  (P : Parallelogram) (has_right_angle : Bool) : Rectangle :=
  { parallelSides := P.parallelSides
    rightAngles := has_right_angle }

-- The proof would go here
#check parallelogram_with_right_angle_is_rectangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_with_right_angle_is_rectangle_l594_59454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_cd_product_l594_59412

/-- An equilateral triangle with vertices at (0,0), (c,7), and (d,19) -/
structure EquilateralTriangle where
  c : ℝ
  d : ℝ
  is_equilateral : (0, 0) ∈ (Set.univ : Set (ℝ × ℝ)) ∧ 
                   (c, 7) ∈ (Set.univ : Set (ℝ × ℝ)) ∧ 
                   (d, 19) ∈ (Set.univ : Set (ℝ × ℝ))

/-- The product of c and d in the equilateral triangle equals -806/9 -/
theorem equilateral_triangle_cd_product (t : EquilateralTriangle) : t.c * t.d = -806/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_cd_product_l594_59412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l594_59474

/-- The original proposition -/
def original_prop (x : ℝ) : Prop := x = 1 → |x| = 1

/-- The converse of the original proposition -/
def converse_prop (x : ℝ) : Prop := |x| = 1 → x = 1

/-- The inverse of the original proposition -/
def inverse_prop (x : ℝ) : Prop := x ≠ 1 → |x| ≠ 1

/-- The contrapositive of the original proposition -/
def contrapositive_prop (x : ℝ) : Prop := |x| ≠ 1 → x ≠ 1

/-- The set of all propositions -/
def prop_set : Set (ℝ → Prop) := {original_prop, converse_prop, inverse_prop, contrapositive_prop}

/-- The main theorem stating that exactly two of the propositions are true -/
theorem two_true_propositions : 
  ∃ (p q : ℝ → Prop), p ∈ prop_set ∧ 
                       q ∈ prop_set ∧ 
                       p ≠ q ∧ 
                       (∀ x, p x) ∧ 
                       (∀ x, q x) ∧
                       (∀ r, r ∈ prop_set → r ≠ p → r ≠ q → ¬(∀ x, r x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l594_59474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_parabola_properties_l594_59472

/-- Given a hyperbola with equation 16y^2 - 9x^2 = 144, prove its properties and a related parabola's equation -/
theorem hyperbola_and_parabola_properties :
  let hyperbola := {(x, y) : ℝ × ℝ | 16 * y^2 - 9 * x^2 = 144}
  let real_axis_length := 4
  let imaginary_axis_length := 3
  let eccentricity := 1.25
  let parabola := {(x, y) : ℝ × ℝ | x^2 = -8 * y}
  (real_axis_length = 4) ∧
  (imaginary_axis_length = 3) ∧
  (eccentricity = 1.25) ∧
  (∀ (x y : ℝ), (x, y) ∈ parabola ↔ x^2 = -8 * y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_parabola_properties_l594_59472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l594_59419

theorem count_integers_satisfying_inequality : 
  ∃ (S : Finset ℤ), (∀ x : ℤ, x ∈ S ↔ (x - 2)^2 ≤ 4) ∧ Finset.card S = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_inequality_l594_59419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gravel_cost_is_258_l594_59407

/-- Represents the dimensions and cost parameters of a lawn with intersecting roads -/
structure LawnWithRoads where
  length : ℚ
  width : ℚ
  road_width : ℚ
  gravel_cost_paise : ℚ

/-- Calculates the total cost of graveling the roads on the lawn -/
def total_gravel_cost (lawn : LawnWithRoads) : ℚ :=
  let road_area := lawn.length * lawn.road_width + lawn.width * lawn.road_width - lawn.road_width * lawn.road_width
  let cost_rupees := road_area * (lawn.gravel_cost_paise / 100)
  cost_rupees

/-- Theorem stating that the total cost of graveling for the given lawn is ₹258 -/
theorem gravel_cost_is_258 (lawn : LawnWithRoads) 
  (h1 : lawn.length = 55)
  (h2 : lawn.width = 35)
  (h3 : lawn.road_width = 4)
  (h4 : lawn.gravel_cost_paise = 75) :
  total_gravel_cost lawn = 258 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gravel_cost_is_258_l594_59407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_right_triangle_lambda_range_obtuse_l594_59440

open Real

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angleSum : A + B + C = π
  sidePos : a > 0 ∧ b > 0 ∧ c > 0

-- Define the given conditions
def condition1 (t : Triangle) : Prop :=
  (2 * t.a - t.c) * cos t.B = t.b * cos t.C

def condition2 (t : Triangle) (lambda : ℝ) : Prop :=
  sin t.A ^ 2 = sin t.B ^ 2 + sin t.C ^ 2 - lambda * sin t.B * sin t.C

-- Theorem statements
theorem angle_B_value (t : Triangle) (lambda : ℝ) 
  (h1 : condition1 t) (h2 : condition2 t lambda) : t.B = π / 3 :=
sorry

theorem right_triangle (t : Triangle) (lambda : ℝ) 
  (h1 : condition1 t) (h2 : condition2 t lambda) (h3 : lambda = Real.sqrt 3) : t.C = π / 2 :=
sorry

theorem lambda_range_obtuse (t : Triangle) (lambda : ℝ) 
  (h1 : condition1 t) (h2 : condition2 t lambda) 
  (h3 : t.B = π / 3) (h4 : t.C > π / 2) : 
  lambda ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioo (Real.sqrt 3 : ℝ) 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_value_right_triangle_lambda_range_obtuse_l594_59440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l594_59466

noncomputable def f (x : ℝ) := (x + 1) * Real.log x - (1/2) * x^2 + x - 1 / Real.exp x

theorem f_inequality (x : ℝ) (h : x > 0) :
  f x + (1/2) * x^2 - x > Real.log x - 1 - 2 * Real.exp (-2) := by
  sorry

#check f_inequality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l594_59466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_with_specific_hcf_lcm_l594_59481

/-- Given two positive integers with specific HCF and LCM properties, prove that one of them is 24 -/
theorem number_with_specific_hcf_lcm (A B : ℕ+) : 
  B = 156 →
  (Nat.gcd A B : ℚ) = 12 →
  (Nat.lcm A B : ℚ) = 312 →
  A = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_with_specific_hcf_lcm_l594_59481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l594_59477

/-- An arithmetic sequence with given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  second_term : a 2 = 1
  fifth_term : a 5 = -5

/-- The general term of the arithmetic sequence -/
def general_term (seq : ArithmeticSequence) : ℕ → ℚ :=
  λ n => -2 * n + 5

/-- The sum of the first n terms of the arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + (n * (n - 1) / 2) * (seq.a 2 - seq.a 1)

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = general_term seq n) ∧
  (∃ max_sum : ℚ, max_sum = 4 ∧ ∀ n, sum_n_terms seq n ≤ max_sum) :=
by
  sorry

#check arithmetic_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l594_59477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_greater_than_one_l594_59417

theorem sin_minus_cos_greater_than_one (α : ℝ) (h : π / 2 < α ∧ α < π) :
  Real.sin α - Real.cos α > 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_greater_than_one_l594_59417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_times_l594_59406

/-- Represents a runner in the race -/
structure Runner where
  name : String
  speed : Rat
  obstacles : List (Nat × Rat)

/-- Calculates the total time for a runner to complete the race -/
noncomputable def totalTime (r : Runner) (distance : Rat) (baseTime : Rat) : Rat :=
  let normalTime := distance * baseTime / r.speed
  let obstacleTime := r.obstacles.foldl (fun acc (mile, delay) => acc + delay) 0
  normalTime + obstacleTime

/-- Theorem stating the total times for each runner -/
theorem race_times (lexie celia nik : Runner) (h1 : lexie.speed = 1) 
    (h2 : celia.speed = 2) (h3 : nik.speed = 3/2) 
    (h4 : lexie.obstacles = [(5, 10), (25, 10)])
    (h5 : celia.obstacles = [(10, 5/2), (20, 5/2)]) 
    (h6 : nik.obstacles = [(15, 20/3)]) : 
    totalTime lexie 30 20 = 620 ∧ totalTime celia 30 20 = 305 ∧ totalTime nik 30 20 = 4079/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_times_l594_59406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_large_chips_proof_l594_59486

/-- The greatest possible number of large chips in a box of 60 chips, where the number of small chips
    exceeds the number of large chips by twice a prime number. -/
def max_large_chips : ℕ := 28

/-- Predicate to check if a distribution of chips is valid according to the problem conditions. -/
def is_valid_distribution (small large : ℕ) : Prop :=
  small + large = 60 ∧ 
  ∃ p : ℕ, Nat.Prime p ∧ small = large + 2 * p

/-- Proof that 28 is the maximum number of large chips satisfying the conditions. -/
theorem max_large_chips_proof :
  ∀ large : ℕ, (∃ small : ℕ, is_valid_distribution small large) → large ≤ max_large_chips :=
by
  intro large
  intro h
  sorry

#check max_large_chips
#check is_valid_distribution
#check max_large_chips_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_large_chips_proof_l594_59486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_in_fourth_quadrant_l594_59422

-- Define the logarithm function
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log_base a x
def g (a : ℝ) (x : ℝ) : ℝ := (1 - a) * x

-- Define what it means for a point to be in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- State the theorem
theorem intersection_in_fourth_quadrant (a : ℝ) (h : a > 1) :
  ∃ x y, x > 0 ∧ f a x = y ∧ g a x = y ∧ in_fourth_quadrant (x, y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_in_fourth_quadrant_l594_59422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_participants_l594_59487

/-- Represents the answer choices for each question -/
inductive Choice
| A
| B
| C

/-- Represents a participant's answers to all questions -/
def Participant := Fin 4 → Choice

/-- The condition that for any three participants, there is at least one question where their answers are all different -/
def validConfiguration (participants : Finset Participant) : Prop :=
  ∀ p1 p2 p3, p1 ∈ participants → p2 ∈ participants → p3 ∈ participants →
    p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 →
    ∃ q : Fin 4, p1 q ≠ p2 q ∧ p2 q ≠ p3 q ∧ p1 q ≠ p3 q

/-- The main theorem stating that the maximum number of participants is 9 -/
theorem max_participants :
  ∃ (participants : Finset Participant),
    validConfiguration participants ∧
    participants.card = 9 ∧
    ∀ (larger : Finset Participant),
      validConfiguration larger → larger.card ≤ 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_participants_l594_59487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_a_value_when_max_is_4_l594_59499

-- Define the function f(x) as noncomputable due to the use of Real.sqrt
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + Real.sqrt 3 * Real.sin (2 * x) + a

-- Theorem for the smallest positive period
theorem smallest_positive_period (a : ℝ) : 
  ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f a (x + T) = f a x) ∧ 
  (∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f a (x + T') = f a x) → T ≤ T') ∧
  T = Real.pi := by
  sorry

-- Theorem for the value of a when maximum is 4
theorem a_value_when_max_is_4 :
  ∃ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f a x ≤ 4) ∧
  (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f a x = 4) ∧
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_a_value_when_max_is_4_l594_59499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_curve_properties_l594_59437

/-- Line l in polar form -/
def line_l (ρ θ : ℝ) : Prop := ρ * (Real.cos θ + 2 * Real.sin θ) = 10

/-- Curve C in parametric form -/
def curve_C (x y θ : ℝ) : Prop := x = 3 * Real.cos θ ∧ y = 2 * Real.sin θ

/-- Standard form of line l -/
def line_l_standard (x y : ℝ) : Prop := x + 2 * y - 10 = 0

/-- Standard form of curve C -/
def curve_C_standard (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- Point on C with minimum distance to l -/
noncomputable def min_distance_point : ℝ × ℝ := (9/5, 8/5)

/-- Minimum distance between C and l -/
noncomputable def min_distance : ℝ := Real.sqrt 5

theorem line_and_curve_properties :
  (∀ ρ θ x y, line_l ρ θ → (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) → line_l_standard x y) ∧
  (∀ x y θ, curve_C x y θ → curve_C_standard x y) ∧
  (∃ θ, curve_C (min_distance_point.1) (min_distance_point.2) θ) ∧
  (∀ x y, curve_C x y (Real.arccos (x/3)) →
    Real.sqrt ((x + 2*y - 10)^2 / 5) ≥ min_distance) ∧
  (Real.sqrt ((min_distance_point.1 + 2*min_distance_point.2 - 10)^2 / 5) = min_distance) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_curve_properties_l594_59437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_harmonic_sum_exceeding_10_l594_59434

open BigOperators

def harmonic_sum (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, 1 / (k + 1 : ℚ)

def is_smallest_exceeding (n : ℕ) (threshold : ℚ) : Prop :=
  harmonic_sum n > threshold ∧ ∀ m < n, harmonic_sum m ≤ threshold

theorem smallest_harmonic_sum_exceeding_10 :
  is_smallest_exceeding 12320 10 := by
  sorry

#eval harmonic_sum 12320
#eval harmonic_sum 12319

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_harmonic_sum_exceeding_10_l594_59434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l594_59489

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  3 * x + 1 / (x - 1) ≥ 2 * Real.sqrt 3 + 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_function_l594_59489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_intersect_Q_l594_59475

def P : Set ℝ := {-4, -2, 0, 2, 4}
def Q : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem P_intersect_Q : P ∩ Q = {0, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_intersect_Q_l594_59475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l594_59439

open Real

theorem trigonometric_identities (α : ℝ) 
  (h : Real.tan α / (Real.tan α - 1) = -1) : 
  (Real.sin α - 2 * Real.cos α) / (Real.sin α + Real.cos α) = -1 ∧ 
  Real.sin α ^ 2 + Real.sin α * Real.cos α = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l594_59439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l594_59484

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x + x^(1/2 : ℝ) - 1

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) :
  f x > f (2*x - 4) ↔ 2 < x ∧ x < 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l594_59484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_largest_smallest_prime_factors_546_l594_59463

def largest_smallest_prime_factor_sum (n : ℕ) : ℕ :=
  let factors := n.factors
  (factors.minimum? |>.getD 1) + (factors.maximum? |>.getD 1)

theorem sum_largest_smallest_prime_factors_546 :
  largest_smallest_prime_factor_sum 546 = 15 := by
  rw [largest_smallest_prime_factor_sum]
  simp
  rfl

#eval largest_smallest_prime_factor_sum 546

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_largest_smallest_prime_factors_546_l594_59463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l594_59449

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * Real.log (x + 1)

-- State the theorem
theorem f_monotone_decreasing :
  ∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l594_59449
