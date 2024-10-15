import Mathlib

namespace NUMINAMATH_CALUDE_small_square_area_l3149_314978

-- Define the tile and its components
def TileArea : ℝ := 49
def HypotenuseLength : ℝ := 5
def NumTriangles : ℕ := 8

-- Theorem statement
theorem small_square_area :
  ∀ (small_square_area : ℝ),
    small_square_area = TileArea - NumTriangles * (HypotenuseLength^2 / 2) →
    small_square_area = 1 :=
by sorry

end NUMINAMATH_CALUDE_small_square_area_l3149_314978


namespace NUMINAMATH_CALUDE_smallest_n_for_20_colors_l3149_314966

/-- Represents a ball with a color -/
structure Ball :=
  (color : Nat)

/-- Represents a circular arrangement of balls -/
def CircularArrangement := List Ball

/-- Checks if a sequence of balls has at least k different colors -/
def hasAtLeastKColors (sequence : List Ball) (k : Nat) : Prop :=
  (sequence.map Ball.color).toFinset.card ≥ k

theorem smallest_n_for_20_colors 
  (total_balls : Nat) 
  (num_colors : Nat) 
  (balls_per_color : Nat) 
  (h1 : total_balls = 1000) 
  (h2 : num_colors = 40) 
  (h3 : balls_per_color = 25) 
  (h4 : total_balls = num_colors * balls_per_color) :
  ∃ (n : Nat), 
    (∀ (arrangement : CircularArrangement), 
      arrangement.length = total_balls → 
      ∃ (subsequence : List Ball), 
        subsequence.length = n ∧ 
        hasAtLeastKColors subsequence 20) ∧
    (∀ (m : Nat), m < n → 
      ∃ (arrangement : CircularArrangement), 
        arrangement.length = total_balls ∧ 
        ∀ (subsequence : List Ball), 
          subsequence.length = m → 
          ¬(hasAtLeastKColors subsequence 20)) ∧
    n = 352 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_20_colors_l3149_314966


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l3149_314932

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem tenth_term_of_sequence (a₁ d : ℝ) :
  arithmetic_sequence a₁ d 3 = 10 →
  arithmetic_sequence a₁ d 6 = 16 →
  arithmetic_sequence a₁ d 10 = 24 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l3149_314932


namespace NUMINAMATH_CALUDE_difference_m_n_l3149_314963

theorem difference_m_n (m n : ℕ+) (h : 10 * 2^(m : ℕ) = 2^(n : ℕ) + 2^((n : ℕ) + 2)) :
  (n : ℕ) - (m : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_m_n_l3149_314963


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l3149_314900

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x + 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  hyperbola_C A.1 A.2 ∧ hyperbola_C B.1 B.2 ∧ 
  line_l A.1 A.2 ∧ line_l B.1 B.2 ∧ 
  A ≠ B

-- Theorem statement
theorem hyperbola_theorem 
  (center : ℝ × ℝ) 
  (right_focus : ℝ × ℝ) 
  (right_vertex : ℝ × ℝ) 
  (A B : ℝ × ℝ) :
  center = (0, 0) →
  right_focus = (2, 0) →
  right_vertex = (Real.sqrt 3, 0) →
  intersection_points A B →
  (∀ x y, hyperbola_C x y ↔ x^2 / 3 - y^2 = 1) ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l3149_314900


namespace NUMINAMATH_CALUDE_forgotten_lawns_l3149_314957

/-- 
Given that:
- Roger earns $9 for each lawn he mows
- He had 14 lawns to mow
- He actually earned $54

Prove that the number of lawns Roger forgot to mow is equal to 14 minus the quotient of 54 and 9.
-/
theorem forgotten_lawns (earnings_per_lawn : ℕ) (total_lawns : ℕ) (actual_earnings : ℕ) :
  earnings_per_lawn = 9 →
  total_lawns = 14 →
  actual_earnings = 54 →
  total_lawns - (actual_earnings / earnings_per_lawn) = 8 :=
by sorry

end NUMINAMATH_CALUDE_forgotten_lawns_l3149_314957


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l3149_314945

theorem min_value_sum_squares (x y z : ℝ) (h : x + 2*y + 3*z = 6) :
  ∃ (min : ℝ), min = 18/7 ∧ x^2 + y^2 + z^2 ≥ min ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ + 2*y₀ + 3*z₀ = 6 ∧ x₀^2 + y₀^2 + z₀^2 = min :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l3149_314945


namespace NUMINAMATH_CALUDE_max_individual_score_l3149_314999

theorem max_individual_score (total_points : ℕ) (num_players : ℕ) (min_points : ℕ) 
  (h1 : total_points = 100)
  (h2 : num_players = 12)
  (h3 : min_points = 8)
  (h4 : ∀ i : ℕ, i < num_players → min_points ≤ (total_points / num_players)) :
  ∃ max_score : ℕ, max_score = 12 ∧ 
    ∀ player_score : ℕ, player_score ≤ max_score ∧
    (num_players - 1) * min_points + max_score = total_points :=
by sorry

end NUMINAMATH_CALUDE_max_individual_score_l3149_314999


namespace NUMINAMATH_CALUDE_product_of_arithmetic_sequences_l3149_314927

/-- An arithmetic sequence -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The product sequence of two arithmetic sequences -/
def product_seq (a b : ℕ → ℝ) (c : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, c n = a n * b n

theorem product_of_arithmetic_sequences
  (a b : ℕ → ℝ) (c : ℕ → ℝ)
  (ha : arithmetic_seq a)
  (hb : arithmetic_seq b)
  (hc : product_seq a b c)
  (h1 : c 1 = 1440)
  (h2 : c 2 = 1716)
  (h3 : c 3 = 1848) :
  c 8 = 348 := by
  sorry

end NUMINAMATH_CALUDE_product_of_arithmetic_sequences_l3149_314927


namespace NUMINAMATH_CALUDE_nonagon_diagonals_count_l3149_314948

/-- The number of distinct diagonals in a convex nonagon -/
def nonagon_diagonals : ℕ := 27

/-- A convex polygon with 9 sides -/
def nonagon : ℕ := 9

/-- Theorem stating that the number of distinct diagonals in a convex nonagon is 27 -/
theorem nonagon_diagonals_count : nonagon_diagonals = (nonagon * (nonagon - 3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_count_l3149_314948


namespace NUMINAMATH_CALUDE_quadratic_positive_function_m_range_l3149_314990

/-- A function is positive on a domain if there exists a subinterval where the function maps the interval to itself -/
def PositiveFunction (f : ℝ → ℝ) (D : Set ℝ) :=
  ∃ a b, a < b ∧ Set.Icc a b ⊆ D ∧ Set.Icc a b = f '' Set.Icc a b

/-- The quadratic function g(x) = x^2 - m -/
def g (m : ℝ) : ℝ → ℝ := fun x ↦ x^2 - m

theorem quadratic_positive_function_m_range :
  (∃ m, PositiveFunction (g m) (Set.Iio 0)) → ∃ m, m ∈ Set.Ioo (3/4) 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_positive_function_m_range_l3149_314990


namespace NUMINAMATH_CALUDE_parallel_condition_l3149_314911

/-- Two lines in the form of ax + by + c = 0 and dx + ey + f = 0 are parallel if and only if ae = bd -/
def are_parallel (a b c d e f : ℝ) : Prop := a * e = b * d

/-- The first line: ax + 2y - 1 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - 1 = 0

/-- The second line: x + (a + 1)y + 4 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

theorem parallel_condition (a : ℝ) :
  (a = -2 → are_parallel a 2 (-1) 1 (a + 1) 4) ∧
  (∃ b : ℝ, b ≠ -2 ∧ are_parallel b 2 (-1) 1 (b + 1) 4) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l3149_314911


namespace NUMINAMATH_CALUDE_intersection_trajectory_l3149_314901

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the endpoints of the major axis
def majorAxisEndpoints (A₁ A₂ : ℝ × ℝ) : Prop :=
  A₁ = (-3, 0) ∧ A₂ = (3, 0)

-- Define a chord perpendicular to the major axis
def perpendicularChord (P₁ P₂ : ℝ × ℝ) : Prop :=
  ellipse P₁.1 P₁.2 ∧ ellipse P₂.1 P₂.2 ∧ P₁.1 = P₂.1 ∧ P₁.2 = -P₂.2

-- Define the intersection point of A₁P₁ and A₂P₂
def intersectionPoint (Q : ℝ × ℝ) (A₁ A₂ P₁ P₂ : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ,
    Q = (1 - t₁) • A₁ + t₁ • P₁ ∧
    Q = (1 - t₂) • A₂ + t₂ • P₂

-- The theorem to be proved
theorem intersection_trajectory
  (A₁ A₂ P₁ P₂ Q : ℝ × ℝ)
  (h₁ : majorAxisEndpoints A₁ A₂)
  (h₂ : perpendicularChord P₁ P₂)
  (h₃ : intersectionPoint Q A₁ A₂ P₁ P₂) :
  Q.1^2 / 9 - Q.2^2 / 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_trajectory_l3149_314901


namespace NUMINAMATH_CALUDE_sqrt_180_simplification_l3149_314950

theorem sqrt_180_simplification : Real.sqrt 180 = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_180_simplification_l3149_314950


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3149_314967

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (2 - 3 * i) / (1 + 4 * i) = -10/17 - (11/17) * i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3149_314967


namespace NUMINAMATH_CALUDE_tom_typing_time_l3149_314953

/-- Calculates the time required to type a given number of pages -/
def typing_time (words_per_minute : ℕ) (words_per_page : ℕ) (num_pages : ℕ) : ℕ :=
  (words_per_page * num_pages) / words_per_minute

theorem tom_typing_time :
  typing_time 90 450 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_tom_typing_time_l3149_314953


namespace NUMINAMATH_CALUDE_cistern_emptying_time_l3149_314994

/-- Given a cistern with specific properties, prove the time it takes to empty -/
theorem cistern_emptying_time 
  (capacity : ℝ)
  (leak_empty_time : ℝ)
  (tap_rate : ℝ)
  (h1 : capacity = 480)
  (h2 : leak_empty_time = 20)
  (h3 : tap_rate = 4)
  : (capacity / (capacity / leak_empty_time - tap_rate) = 24) :=
by
  sorry

end NUMINAMATH_CALUDE_cistern_emptying_time_l3149_314994


namespace NUMINAMATH_CALUDE_modulus_of_complex_expression_l3149_314952

theorem modulus_of_complex_expression : 
  Complex.abs ((1 - 2 * Complex.I)^2 / Complex.I) = 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_expression_l3149_314952


namespace NUMINAMATH_CALUDE_virginia_eggs_remaining_l3149_314925

/-- Given Virginia starts with 96 eggs and Amy takes 3 eggs away, 
    prove that Virginia ends up with 93 eggs. -/
theorem virginia_eggs_remaining : 
  let initial_eggs : ℕ := 96
  let eggs_taken : ℕ := 3
  initial_eggs - eggs_taken = 93 := by sorry

end NUMINAMATH_CALUDE_virginia_eggs_remaining_l3149_314925


namespace NUMINAMATH_CALUDE_fifth_month_sale_is_6500_l3149_314910

/-- Calculates the sale in the fifth month given the sales for other months and the average -/
def fifth_month_sale (m1 m2 m3 m4 m6 avg : ℕ) : ℕ :=
  6 * avg - (m1 + m2 + m3 + m4 + m6)

/-- Theorem stating that the sale in the fifth month is 6500 -/
theorem fifth_month_sale_is_6500 :
  fifth_month_sale 6400 7000 6800 7200 5100 6500 = 6500 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_is_6500_l3149_314910


namespace NUMINAMATH_CALUDE_total_athletes_l3149_314972

/-- Given the ratio of players and the number of basketball players, 
    calculate the total number of athletes -/
theorem total_athletes (football baseball soccer basketball : ℕ) 
  (h_ratio : football = 10 ∧ baseball = 7 ∧ soccer = 5 ∧ basketball = 4)
  (h_basketball_players : basketball * 4 = 16) : 
  football * 4 + baseball * 4 + soccer * 4 + basketball * 4 = 104 := by
  sorry

#check total_athletes

end NUMINAMATH_CALUDE_total_athletes_l3149_314972


namespace NUMINAMATH_CALUDE_expand_polynomial_l3149_314970

theorem expand_polynomial (x : ℝ) : (x + 3) * (4 * x^2 - 8 * x + 5) = 4 * x^3 + 4 * x^2 - 19 * x + 15 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l3149_314970


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3149_314985

/-- The asymptote equations of the hyperbola x^2 - y^2/4 = 1 are y = 2x and y = -2x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 - y^2/4 = 1) → (∃ (k : ℝ), k = 2 ∨ k = -2) ∧ (y = k*x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3149_314985


namespace NUMINAMATH_CALUDE_square_sum_of_xy_l3149_314959

theorem square_sum_of_xy (x y : ℕ+) 
  (h1 : x * y + 2 * x + 2 * y = 152)
  (h2 : x ^ 2 * y + x * y ^ 2 = 1512) :
  x ^ 2 + y ^ 2 = 1136 ∨ x ^ 2 + y ^ 2 = 221 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_xy_l3149_314959


namespace NUMINAMATH_CALUDE_intersection_lines_l3149_314924

-- Define the fixed points M₁ and M₂
def M₁ : ℝ × ℝ := (26, 1)
def M₂ : ℝ × ℝ := (2, 1)

-- Define the point P
def P : ℝ × ℝ := (-2, 3)

-- Define the distance ratio condition
def distance_ratio (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  (((x - M₁.1)^2 + (y - M₁.2)^2) / ((x - M₂.1)^2 + (y - M₂.2)^2)) = 25

-- Define the trajectory of M
def trajectory (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 25

-- Define the chord length condition
def chord_length (l : ℝ → ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    trajectory x₁ y₁ ∧ trajectory x₂ y₂ ∧
    y₁ = l x₁ ∧ y₂ = l x₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 64

-- Theorem statement
theorem intersection_lines :
  ∀ (l : ℝ → ℝ),
    (∀ x, l x = -2 ∨ l x = (-5/12) * x + 23/6) ↔
    (∀ M, distance_ratio M → trajectory M.1 M.2) ∧
    chord_length l ∧
    l P.1 = P.2 :=
sorry

end NUMINAMATH_CALUDE_intersection_lines_l3149_314924


namespace NUMINAMATH_CALUDE_extremum_of_g_and_range_of_a_l3149_314965

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (x : ℝ) : ℝ := Real.exp x - a * x^2
def g (x : ℝ) : ℝ := Real.exp x - 2 * a * x

theorem extremum_of_g_and_range_of_a :
  (a > 0 → ∃ (x_min : ℝ), x_min = Real.log (2 * a) ∧ 
    (∀ y, g a y ≥ g a x_min) ∧ 
    g a x_min = 2 * a - 2 * a * Real.log (2 * a)) ∧
  ((∀ x ≥ 0, f a x ≥ x + (1 - x) * Real.exp x) → a ≤ 1) :=
sorry

end

end NUMINAMATH_CALUDE_extremum_of_g_and_range_of_a_l3149_314965


namespace NUMINAMATH_CALUDE_count_square_family_with_range_14_l3149_314995

/-- A function family is characterized by its analytic expression and range -/
structure FunctionFamily where
  expression : ℝ → ℝ
  range : Set ℝ

/-- Count the number of functions in a family with different domains -/
def countFunctionsInFamily (f : FunctionFamily) : ℕ :=
  sorry

/-- The specific function family we're interested in -/
def squareFamilyWithRange14 : FunctionFamily :=
  { expression := fun x ↦ x^2,
    range := {1, 4} }

/-- Theorem stating that the number of functions in our specific family is 9 -/
theorem count_square_family_with_range_14 :
  countFunctionsInFamily squareFamilyWithRange14 = 9 := by
  sorry

end NUMINAMATH_CALUDE_count_square_family_with_range_14_l3149_314995


namespace NUMINAMATH_CALUDE_fraction_ratio_equality_l3149_314912

theorem fraction_ratio_equality : ∃ x : ℚ, (5 / 34) / (7 / 48) = x / (1 / 13) ∧ x = 5 / 64 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ratio_equality_l3149_314912


namespace NUMINAMATH_CALUDE_equation_solution_l3149_314960

theorem equation_solution : ∃! x : ℝ, 9 / (5 + x / 0.75) = 1 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3149_314960


namespace NUMINAMATH_CALUDE_triangle_dot_product_l3149_314956

/-- Given a triangle ABC with |AB| = 4, |AC| = 1, and area = √3,
    prove that the dot product of AB and AC is ±2 -/
theorem triangle_dot_product (A B C : ℝ × ℝ) : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let AC := (C.1 - A.1, C.2 - A.2)
  (AB.1^2 + AB.2^2 = 16) →  -- |AB| = 4
  (AC.1^2 + AC.2^2 = 1) →   -- |AC| = 1
  (abs (AB.1 * AC.2 - AB.2 * AC.1) = 2 * Real.sqrt 3) →  -- Area = √3
  ((AB.1 * AC.1 + AB.2 * AC.2)^2 = 4) :=  -- Dot product squared = 4
by sorry

end NUMINAMATH_CALUDE_triangle_dot_product_l3149_314956


namespace NUMINAMATH_CALUDE_bank_account_withdrawal_l3149_314996

theorem bank_account_withdrawal (initial_balance deposit1 deposit2 final_balance_increase : ℕ) :
  initial_balance = 150 →
  deposit1 = 17 →
  deposit2 = 21 →
  final_balance_increase = 16 →
  ∃ withdrawal : ℕ, 
    initial_balance + deposit1 - withdrawal + deposit2 = initial_balance + final_balance_increase ∧
    withdrawal = 22 :=
by sorry

end NUMINAMATH_CALUDE_bank_account_withdrawal_l3149_314996


namespace NUMINAMATH_CALUDE_octagon_diagonals_l3149_314980

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l3149_314980


namespace NUMINAMATH_CALUDE_solution_characterization_l3149_314914

/-- The set of all solutions to the equation ab + bc + ca = 2(a + b + c) in natural numbers -/
def SolutionSet : Set (ℕ × ℕ × ℕ) :=
  {(2, 2, 2), (1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 4, 1), (4, 1, 2), (4, 2, 1)}

/-- The equation ab + bc + ca = 2(a + b + c) -/
def SatisfiesEquation (t : ℕ × ℕ × ℕ) : Prop :=
  let (a, b, c) := t
  a * b + b * c + c * a = 2 * (a + b + c)

theorem solution_characterization :
  ∀ t : ℕ × ℕ × ℕ, t ∈ SolutionSet ↔ SatisfiesEquation t :=
sorry

end NUMINAMATH_CALUDE_solution_characterization_l3149_314914


namespace NUMINAMATH_CALUDE_log_less_than_zero_implies_x_between_zero_and_one_l3149_314926

theorem log_less_than_zero_implies_x_between_zero_and_one (x : ℝ) :
  (∃ (y : ℝ), y = Real.log x ∧ y < 0) → 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_log_less_than_zero_implies_x_between_zero_and_one_l3149_314926


namespace NUMINAMATH_CALUDE_jose_bottle_caps_l3149_314916

/-- 
Given that Jose starts with some bottle caps, gets 2 more from Rebecca, 
and ends up with 9 bottle caps, prove that he started with 7 bottle caps.
-/
theorem jose_bottle_caps (x : ℕ) : x + 2 = 9 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_jose_bottle_caps_l3149_314916


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l3149_314937

/-- The distance from a point to the y-axis is equal to the absolute value of its x-coordinate -/
theorem distance_to_y_axis (P : ℝ × ℝ) : 
  let (x, y) := P
  abs x = Real.sqrt ((x - 0)^2 + (y - y)^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l3149_314937


namespace NUMINAMATH_CALUDE_cantor_set_max_operation_l3149_314902

theorem cantor_set_max_operation : 
  ∃ n : ℕ, (∀ k : ℕ, k > n → (2/3 : ℝ)^(k-1) * (1/3) < 1/60) ∧ 
           (2/3 : ℝ)^(n-1) * (1/3) ≥ 1/60 ∧ 
           n = 8 :=
sorry

end NUMINAMATH_CALUDE_cantor_set_max_operation_l3149_314902


namespace NUMINAMATH_CALUDE_power_of_four_l3149_314951

theorem power_of_four (k : ℕ) (h : 4^k = 5) : 4^(2*k + 2) = 400 := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_l3149_314951


namespace NUMINAMATH_CALUDE_sqrt_plus_one_iff_ax_plus_x_over_x_minus_one_l3149_314917

theorem sqrt_plus_one_iff_ax_plus_x_over_x_minus_one 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt a + 1 > b ↔ ∀ x > 1, a * x + x / (x - 1) > b := by
sorry

end NUMINAMATH_CALUDE_sqrt_plus_one_iff_ax_plus_x_over_x_minus_one_l3149_314917


namespace NUMINAMATH_CALUDE_lighthouse_lights_sum_l3149_314928

theorem lighthouse_lights_sum : 
  let n : ℕ := 7
  let a₁ : ℕ := 1
  let q : ℕ := 2
  let sum := (a₁ * (1 - q^n)) / (1 - q)
  sum = 127 := by
sorry

end NUMINAMATH_CALUDE_lighthouse_lights_sum_l3149_314928


namespace NUMINAMATH_CALUDE_inscribed_prism_properties_l3149_314987

/-- Regular triangular pyramid with inscribed regular triangular prism -/
structure PyramidWithPrism where
  pyramid_height : ℝ
  pyramid_base_side : ℝ
  prism_lateral_area : ℝ

/-- Possible solutions for the inscribed prism -/
structure PrismSolution where
  prism_height : ℝ
  lateral_area_ratio : ℝ

/-- Theorem stating the properties of the inscribed prism -/
theorem inscribed_prism_properties (p : PyramidWithPrism) 
  (h1 : p.pyramid_height = 15)
  (h2 : p.pyramid_base_side = 12)
  (h3 : p.prism_lateral_area = 120) :
  ∃ (s1 s2 : PrismSolution),
    (s1.prism_height = 10 ∧ s1.lateral_area_ratio = 1/9) ∧
    (s2.prism_height = 5 ∧ s2.lateral_area_ratio = 4/9) :=
sorry

end NUMINAMATH_CALUDE_inscribed_prism_properties_l3149_314987


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3149_314947

-- Define the repeating decimal 0.333...
def repeating_3 : ℚ := 1 / 3

-- Define the repeating decimal 0.0202...
def repeating_02 : ℚ := 2 / 99

-- Theorem statement
theorem sum_of_repeating_decimals : repeating_3 + repeating_02 = 35 / 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3149_314947


namespace NUMINAMATH_CALUDE_absolute_value_expression_l3149_314991

theorem absolute_value_expression : |-2| * (|-25| - |5|) = -40 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l3149_314991


namespace NUMINAMATH_CALUDE_zach_current_tickets_l3149_314962

def ferris_wheel_cost : ℕ := 2
def roller_coaster_cost : ℕ := 7
def log_ride_cost : ℕ := 1
def additional_tickets_needed : ℕ := 9

def total_cost : ℕ := ferris_wheel_cost + roller_coaster_cost + log_ride_cost

theorem zach_current_tickets : total_cost - additional_tickets_needed = 1 := by
  sorry

end NUMINAMATH_CALUDE_zach_current_tickets_l3149_314962


namespace NUMINAMATH_CALUDE_evaluate_expression_l3149_314961

theorem evaluate_expression : (64 : ℝ) ^ (0.125 : ℝ) * (64 : ℝ) ^ (0.375 : ℝ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3149_314961


namespace NUMINAMATH_CALUDE_remainder_123456789012_mod_180_l3149_314964

theorem remainder_123456789012_mod_180 : 123456789012 % 180 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_123456789012_mod_180_l3149_314964


namespace NUMINAMATH_CALUDE_sandwich_non_condiments_percentage_l3149_314941

theorem sandwich_non_condiments_percentage 
  (total_weight : ℝ) 
  (condiments_weight : ℝ) 
  (h1 : total_weight = 200) 
  (h2 : condiments_weight = 50) : 
  (total_weight - condiments_weight) / total_weight * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_non_condiments_percentage_l3149_314941


namespace NUMINAMATH_CALUDE_bens_initial_marbles_l3149_314997

theorem bens_initial_marbles (B : ℕ) : 
  (17 + B / 2 = B / 2 + 17) → B = 34 := by sorry

end NUMINAMATH_CALUDE_bens_initial_marbles_l3149_314997


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3149_314921

/-- Given a hyperbola and a parabola satisfying certain conditions, 
    prove that the hyperbola has a specific equation. -/
theorem hyperbola_equation 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (asymptote : b/a = Real.sqrt 3) 
  (focus_on_directrix : a^2 + b^2 = 36) : 
  a^2 = 9 ∧ b^2 = 27 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3149_314921


namespace NUMINAMATH_CALUDE_regular_polygon_reciprocal_sum_l3149_314918

/-- Given a regular polygon with n sides, where the reciprocal of the side length
    equals the sum of reciprocals of two specific diagonals, prove that n = 7. -/
theorem regular_polygon_reciprocal_sum (n : ℕ) (R : ℝ) (h_n : n ≥ 3) :
  (1 : ℝ) / (2 * R * Real.sin (π / n)) =
    1 / (2 * R * Real.sin (2 * π / n)) + 1 / (2 * R * Real.sin (3 * π / n)) →
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_reciprocal_sum_l3149_314918


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3149_314942

theorem chess_tournament_games (n : ℕ) (h : n = 20) : 
  (n * (n - 1)) / 2 = 190 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3149_314942


namespace NUMINAMATH_CALUDE_hypotenuse_length_16_l3149_314989

/-- A right triangle with one angle of 30 degrees -/
structure RightTriangle30 where
  /-- The length of the side opposite to the 30° angle -/
  short_side : ℝ
  /-- The short side is positive -/
  short_side_pos : 0 < short_side

/-- The length of the hypotenuse in a right triangle with a 30° angle -/
def hypotenuse (t : RightTriangle30) : ℝ := 2 * t.short_side

/-- Theorem: In a right triangle with a 30° angle, if the short side is 8, then the hypotenuse is 16 -/
theorem hypotenuse_length_16 (t : RightTriangle30) (h : t.short_side = 8) : 
  hypotenuse t = 16 := by sorry

end NUMINAMATH_CALUDE_hypotenuse_length_16_l3149_314989


namespace NUMINAMATH_CALUDE_isabel_camera_pictures_l3149_314906

/-- Represents the number of pictures in Isabel's photo upload scenario -/
structure IsabelPictures where
  phone : ℕ
  camera : ℕ
  albums : ℕ
  pics_per_album : ℕ

/-- The theorem stating the number of pictures Isabel uploaded from her camera -/
theorem isabel_camera_pictures (p : IsabelPictures) 
  (h1 : p.phone = 2)
  (h2 : p.albums = 3)
  (h3 : p.pics_per_album = 2)
  (h4 : p.albums * p.pics_per_album = p.phone + p.camera) :
  p.camera = 4 := by
  sorry

#check isabel_camera_pictures

end NUMINAMATH_CALUDE_isabel_camera_pictures_l3149_314906


namespace NUMINAMATH_CALUDE_solution_implies_a_value_l3149_314923

theorem solution_implies_a_value (a : ℝ) : 
  (∃ x : ℝ, x = 4 ∧ a * x - 3 = 4 * x + 1) → a = 5 := by
sorry

end NUMINAMATH_CALUDE_solution_implies_a_value_l3149_314923


namespace NUMINAMATH_CALUDE_tree_height_proof_l3149_314949

/-- Represents the height of a tree as a function of its breast diameter -/
def tree_height (x : ℝ) : ℝ := 25 * x + 15

theorem tree_height_proof :
  (tree_height 0.2 = 20) ∧
  (tree_height 0.28 = 22) ∧
  (tree_height 0.3 = 22.5) := by
  sorry

end NUMINAMATH_CALUDE_tree_height_proof_l3149_314949


namespace NUMINAMATH_CALUDE_soccer_ball_weight_l3149_314974

theorem soccer_ball_weight :
  ∀ (soccer_ball_weight bicycle_weight : ℝ),
    8 * soccer_ball_weight = 5 * bicycle_weight →
    4 * bicycle_weight = 120 →
    soccer_ball_weight = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_weight_l3149_314974


namespace NUMINAMATH_CALUDE_third_side_of_triangle_l3149_314954

theorem third_side_of_triangle (a b c : ℝ) : 
  a = 3 → b = 5 → c = 4 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) ∧ 
  (c < a + b ∧ a < b + c ∧ b < c + a) := by
  sorry

end NUMINAMATH_CALUDE_third_side_of_triangle_l3149_314954


namespace NUMINAMATH_CALUDE_larger_number_proof_l3149_314975

theorem larger_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 60) 
  (h2 : Nat.lcm a b = 60 * 11 * 15) : max a b = 900 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3149_314975


namespace NUMINAMATH_CALUDE_sum_x_coordinates_above_line_l3149_314955

def points : List (ℚ × ℚ) := [(2, 8), (5, 15), (10, 25), (15, 36), (19, 45), (22, 52), (25, 66)]

def isAboveLine (p : ℚ × ℚ) : Bool :=
  p.2 > 2 * p.1 + 5

def pointsAboveLine : List (ℚ × ℚ) :=
  points.filter isAboveLine

theorem sum_x_coordinates_above_line :
  (pointsAboveLine.map (·.1)).sum = 81 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_coordinates_above_line_l3149_314955


namespace NUMINAMATH_CALUDE_number_equation_l3149_314931

theorem number_equation (x : ℚ) : (x + 20 / 90) * 90 = 4520 ↔ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l3149_314931


namespace NUMINAMATH_CALUDE_shortest_distance_C1_C2_l3149_314982

/-- The curve C1 in Cartesian coordinates -/
def C1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 2

/-- The curve C2 as a line in Cartesian coordinates -/
def C2 (x y : ℝ) : Prop := x + y = 4

/-- The shortest distance between C1 and C2 -/
theorem shortest_distance_C1_C2 :
  ∃ (p q : ℝ × ℝ), C1 p.1 p.2 ∧ C2 q.1 q.2 ∧
    ∀ (p' q' : ℝ × ℝ), C1 p'.1 p'.2 → C2 q'.1 q'.2 →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) ∧
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_shortest_distance_C1_C2_l3149_314982


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l3149_314939

/-- The surface area of the circumscribed sphere of a rectangular parallelepiped
    with face diagonal lengths 2, √3, and √5 is 6π. -/
theorem circumscribed_sphere_surface_area 
  (x y z : ℝ) 
  (h1 : x^2 + y^2 = 4) 
  (h2 : y^2 + z^2 = 3) 
  (h3 : z^2 + x^2 = 5) : 
  4 * Real.pi * ((x^2 + y^2 + z^2) / 4) = 6 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l3149_314939


namespace NUMINAMATH_CALUDE_min_value_expression_l3149_314934

theorem min_value_expression (a b : ℝ) (h : a - b^2 = 4) :
  ∃ (m : ℝ), m = 5 ∧ ∀ (x y : ℝ), x - y^2 = 4 → x^2 - 3*y^2 + x - 15 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3149_314934


namespace NUMINAMATH_CALUDE_gadget_production_l3149_314973

/-- Represents the time (in hours) required for one worker to produce one gizmo -/
def gizmo_time : ℚ := sorry

/-- Represents the time (in hours) required for one worker to produce one gadget -/
def gadget_time : ℚ := sorry

/-- The number of gadgets produced by 30 workers in 4 hours -/
def n : ℕ := sorry

theorem gadget_production :
  -- In 1 hour, 80 workers produce 200 gizmos and 160 gadgets
  80 * (200 * gizmo_time + 160 * gadget_time) = 1 →
  -- In 2 hours, 40 workers produce 160 gizmos and 240 gadgets
  40 * (160 * gizmo_time + 240 * gadget_time) = 2 →
  -- In 4 hours, 30 workers produce 120 gizmos and n gadgets
  30 * (120 * gizmo_time + n * gadget_time) = 4 →
  -- The number of gadgets produced by 30 workers in 4 hours is 135680
  n = 135680 := by
  sorry

end NUMINAMATH_CALUDE_gadget_production_l3149_314973


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3149_314943

theorem right_triangle_hypotenuse (shorter_leg : ℝ) (longer_leg : ℝ) (area : ℝ) :
  shorter_leg > 0 →
  longer_leg = 3 * shorter_leg - 3 →
  area = (1 / 2) * shorter_leg * longer_leg →
  area = 84 →
  (shorter_leg ^ 2 + longer_leg ^ 2).sqrt = Real.sqrt 505 := by
  sorry

#check right_triangle_hypotenuse

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3149_314943


namespace NUMINAMATH_CALUDE_base8_perfect_square_c_not_unique_l3149_314929

/-- Represents a number in base 8 of the form ab5c -/
structure Base8Number where
  a : Nat
  b : Nat
  c : Nat
  a_nonzero : a ≠ 0
  b_valid : b < 8
  c_valid : c < 8

/-- Converts a Base8Number to its decimal representation -/
def toDecimal (n : Base8Number) : Nat :=
  512 * n.a + 64 * n.b + 40 + n.c

/-- Predicate to check if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

theorem base8_perfect_square_c_not_unique :
  ∃ (n1 n2 : Base8Number),
    n1.a = n2.a ∧ n1.b = n2.b ∧ n1.c ≠ n2.c ∧
    isPerfectSquare (toDecimal n1) ∧
    isPerfectSquare (toDecimal n2) := by
  sorry

end NUMINAMATH_CALUDE_base8_perfect_square_c_not_unique_l3149_314929


namespace NUMINAMATH_CALUDE_ping_pong_balls_l3149_314938

theorem ping_pong_balls (y w : ℕ) : 
  y = 2 * (w - 10) →
  w - 10 = 5 * (y - 9) →
  y = 10 ∧ w = 15 := by
sorry

end NUMINAMATH_CALUDE_ping_pong_balls_l3149_314938


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3149_314976

theorem min_value_expression (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  8 * a^4 + 12 * b^4 + 40 * c^4 + 18 * d^4 + 9 / (4 * a * b * c * d) ≥ 12 * Real.sqrt 2 :=
by sorry

theorem min_value_achievable :
  ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    8 * a^4 + 12 * b^4 + 40 * c^4 + 18 * d^4 + 9 / (4 * a * b * c * d) = 12 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l3149_314976


namespace NUMINAMATH_CALUDE_t_value_l3149_314969

theorem t_value (x y t : ℝ) (h1 : 2^x = t) (h2 : 5^y = t) (h3 : 1/x + 1/y = 2) (h4 : t ≠ 1) : t = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_t_value_l3149_314969


namespace NUMINAMATH_CALUDE_rectangle_length_l3149_314930

/-- The perimeter of a rectangle -/
def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: For a rectangle with perimeter 1200 and width 500, its length is 100 -/
theorem rectangle_length (p w : ℝ) (h1 : p = 1200) (h2 : w = 500) :
  ∃ l : ℝ, perimeter l w = p ∧ l = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l3149_314930


namespace NUMINAMATH_CALUDE_hamburger_combinations_l3149_314936

/-- The number of condiments available -/
def num_condiments : ℕ := 9

/-- The number of bun choices available -/
def num_bun_choices : ℕ := 2

/-- The number of meat patty choices available -/
def num_patty_choices : ℕ := 3

/-- The total number of different hamburger combinations -/
def total_hamburgers : ℕ := 2^num_condiments * num_bun_choices * num_patty_choices

theorem hamburger_combinations :
  total_hamburgers = 3072 :=
sorry

end NUMINAMATH_CALUDE_hamburger_combinations_l3149_314936


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3149_314981

/-- Given vectors a and b in ℝ², if a-b is perpendicular to ma+b, then m = 1/4 -/
theorem perpendicular_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
  (h1 : a = (2, 1))
  (h2 : b = (1, -1))
  (h3 : (a.1 - b.1, a.2 - b.2) • (m * a.1 + b.1, m * a.2 + b.2) = 0) :
  m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3149_314981


namespace NUMINAMATH_CALUDE_floor_properties_l3149_314998

theorem floor_properties (x : ℝ) : 
  (x - 1 < ⌊x⌋ ∧ ⌊x⌋ ≤ x) ∧ ⌊2*x⌋ - 2*⌊x⌋ ∈ ({0, 1} : Set ℤ) := by sorry

end NUMINAMATH_CALUDE_floor_properties_l3149_314998


namespace NUMINAMATH_CALUDE_transformation_correct_l3149_314913

def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]
def mirror_scale_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, -2]
def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -2; 2, 0]

theorem transformation_correct :
  mirror_scale_matrix * rotation_matrix = transformation_matrix :=
by sorry

end NUMINAMATH_CALUDE_transformation_correct_l3149_314913


namespace NUMINAMATH_CALUDE_inequality_solution_set_max_m_value_l3149_314904

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2|
def g (m : ℝ) (x : ℝ) : ℝ := -|x + 3| + m

-- Theorem for the solution set of the inequality
theorem inequality_solution_set (a : ℝ) :
  (∀ x, f x + a - 1 > 0 ↔ 
    (a = 1 ∧ x ≠ 2) ∨
    (a > 1) ∨
    (a < 1 ∧ (x < a + 1 ∨ x > 3 - a))) :=
sorry

-- Theorem for the maximum value of m
theorem max_m_value :
  ∀ m, (∀ x, f x > g m x) ↔ m < 5 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_max_m_value_l3149_314904


namespace NUMINAMATH_CALUDE_water_needed_for_solution_l3149_314944

theorem water_needed_for_solution (total_volume : ℝ) (water_ratio : ℝ) (desired_volume : ℝ) :
  water_ratio = 1/3 →
  desired_volume = 0.48 →
  water_ratio * desired_volume = 0.16 :=
by sorry

end NUMINAMATH_CALUDE_water_needed_for_solution_l3149_314944


namespace NUMINAMATH_CALUDE_square_area_ratio_l3149_314940

theorem square_area_ratio (side_C side_D : ℝ) (h1 : side_C = 45) (h2 : side_D = 60) :
  (side_C^2) / (side_D^2) = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3149_314940


namespace NUMINAMATH_CALUDE_collinear_points_iff_k_eq_neg_one_l3149_314983

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- The theorem stating the condition for collinearity of the given points -/
theorem collinear_points_iff_k_eq_neg_one (k : ℝ) :
  collinear ⟨3, 1⟩ ⟨6, 4⟩ ⟨10, k + 9⟩ ↔ k = -1 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_iff_k_eq_neg_one_l3149_314983


namespace NUMINAMATH_CALUDE_log_equation_solution_l3149_314905

theorem log_equation_solution (x : ℝ) :
  Real.log x / Real.log 8 = 1.75 → x = 32 * Real.sqrt (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3149_314905


namespace NUMINAMATH_CALUDE_share_investment_interest_rate_l3149_314908

/-- Calculates the interest rate for a share investment -/
theorem share_investment_interest_rate 
  (face_value : ℝ) 
  (dividend_rate : ℝ) 
  (market_value : ℝ) 
  (h1 : face_value = 52) 
  (h2 : dividend_rate = 0.09) 
  (h3 : market_value = 39) : 
  (dividend_rate * face_value) / market_value = 0.12 := by
  sorry

#check share_investment_interest_rate

end NUMINAMATH_CALUDE_share_investment_interest_rate_l3149_314908


namespace NUMINAMATH_CALUDE_marble_arrangement_l3149_314946

/-- Represents the color of a marble -/
inductive Color
| Blue
| Yellow

/-- Calculates the number of ways to arrange marbles -/
def arrange_marbles (blue : ℕ) (yellow : ℕ) : ℕ :=
  Nat.choose (yellow + blue - 1) (blue - 1)

/-- The main theorem -/
theorem marble_arrangement :
  let blue := 6
  let max_yellow := 17
  let arrangements := arrange_marbles blue max_yellow
  arrangements = 12376 ∧ arrangements % 1000 = 376 := by
  sorry


end NUMINAMATH_CALUDE_marble_arrangement_l3149_314946


namespace NUMINAMATH_CALUDE_total_instruments_is_19_l3149_314993

-- Define the number of instruments for Charlie
def charlie_flutes : ℕ := 1
def charlie_horns : ℕ := 2
def charlie_harps : ℕ := 1
def charlie_drums : ℕ := 1

-- Define the number of instruments for Carli
def carli_flutes : ℕ := 2 * charlie_flutes
def carli_horns : ℕ := charlie_horns / 2
def carli_harps : ℕ := 0
def carli_drums : ℕ := 3

-- Define the number of instruments for Nick
def nick_flutes : ℕ := charlie_flutes + carli_flutes
def nick_horns : ℕ := charlie_horns - carli_horns
def nick_harps : ℕ := 0
def nick_drums : ℕ := 4

-- Define the total number of instruments
def total_instruments : ℕ := 
  charlie_flutes + charlie_horns + charlie_harps + charlie_drums +
  carli_flutes + carli_horns + carli_harps + carli_drums +
  nick_flutes + nick_horns + nick_harps + nick_drums

-- Theorem statement
theorem total_instruments_is_19 : total_instruments = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_instruments_is_19_l3149_314993


namespace NUMINAMATH_CALUDE_computer_game_cost_l3149_314979

/-- The cost of the computer game Mr. Grey purchased, given the following conditions:
  * He bought 3 polo shirts for $26 each
  * He bought 2 necklaces for $83 each
  * He received a $12 rebate
  * The total cost after the rebate was $322
-/
theorem computer_game_cost : ℕ := by
  let polo_shirt_cost : ℕ := 26
  let polo_shirt_count : ℕ := 3
  let necklace_cost : ℕ := 83
  let necklace_count : ℕ := 2
  let rebate : ℕ := 12
  let total_cost_after_rebate : ℕ := 322

  have h1 : polo_shirt_cost * polo_shirt_count + necklace_cost * necklace_count + 90 = total_cost_after_rebate + rebate := by sorry

  exact 90

end NUMINAMATH_CALUDE_computer_game_cost_l3149_314979


namespace NUMINAMATH_CALUDE_extremum_at_one_implies_a_equals_four_l3149_314977

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_at_one_implies_a_equals_four (a b : ℝ) :
  f_derivative a b 1 = 0 → f a b 1 = 10 → a = 4 := by
  sorry

#check extremum_at_one_implies_a_equals_four

end NUMINAMATH_CALUDE_extremum_at_one_implies_a_equals_four_l3149_314977


namespace NUMINAMATH_CALUDE_cuboid_non_parallel_edges_l3149_314935

/-- Represents a cuboid with integer side lengths -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of edges not parallel to a given edge in a cuboid -/
def nonParallelEdges (c : Cuboid) : ℕ := sorry

/-- Theorem stating that a cuboid with side lengths 8, 6, and 4 has 8 edges not parallel to any given edge -/
theorem cuboid_non_parallel_edges :
  let c : Cuboid := { length := 8, width := 6, height := 4 }
  nonParallelEdges c = 8 := by sorry

end NUMINAMATH_CALUDE_cuboid_non_parallel_edges_l3149_314935


namespace NUMINAMATH_CALUDE_minimum_tents_l3149_314922

theorem minimum_tents (Y : ℕ) : (∃ X : ℕ, 
  X > 0 ∧ 
  10 * (X - 1) < (3 : ℚ) / 2 * Y ∧ (3 : ℚ) / 2 * Y < 10 * X ∧
  10 * (X + 2) < (8 : ℚ) / 5 * Y ∧ (8 : ℚ) / 5 * Y < 10 * (X + 3)) →
  Y ≥ 213 :=
by sorry

end NUMINAMATH_CALUDE_minimum_tents_l3149_314922


namespace NUMINAMATH_CALUDE_matthew_crackers_l3149_314992

def crackers_problem (total_crackers : ℕ) (crackers_per_friend : ℕ) : Prop :=
  total_crackers / crackers_per_friend = 4

theorem matthew_crackers : crackers_problem 8 2 := by
  sorry

end NUMINAMATH_CALUDE_matthew_crackers_l3149_314992


namespace NUMINAMATH_CALUDE_pizza_area_increase_l3149_314988

theorem pizza_area_increase (d₁ d₂ d₃ : ℝ) (h₁ : d₁ = 8) (h₂ : d₂ = 10) (h₃ : d₃ = 14) :
  let area (d : ℝ) := Real.pi * (d / 2)^2
  let percent_increase (a₁ a₂ : ℝ) := (a₂ - a₁) / a₁ * 100
  (percent_increase (area d₁) (area d₂) = 56.25) ∧
  (percent_increase (area d₂) (area d₃) = 96) := by
  sorry

end NUMINAMATH_CALUDE_pizza_area_increase_l3149_314988


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3149_314984

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 + a*x₁ + 3 = 0 ∧ 
   x₂^2 + a*x₂ + 3 = 0 ∧ 
   x₁^3 - 99/(2*x₂^2) = x₂^3 - 99/(2*x₁^2)) → 
  a = -6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3149_314984


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l3149_314907

theorem trig_expression_simplification (α : ℝ) : 
  (Real.tan (2 * π + α)) / (Real.tan (α + π) - Real.cos (-α) + Real.sin (π / 2 - α)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l3149_314907


namespace NUMINAMATH_CALUDE_race_result_l3149_314915

-- Define the type for athlete positions
inductive Position
| First
| Second
| Third
| Fourth

-- Define a function to represent the statements of athletes
def athleteStatement (pos : Position) : Prop :=
  match pos with
  | Position.First => pos = Position.First
  | Position.Second => pos ≠ Position.First
  | Position.Third => pos = Position.First
  | Position.Fourth => pos = Position.Fourth

-- Define the theorem
theorem race_result :
  ∃ (p₁ p₂ p₃ p₄ : Position),
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    (athleteStatement p₁ ∧ athleteStatement p₂ ∧ athleteStatement p₃ ∧ ¬athleteStatement p₄) ∧
    p₃ = Position.First :=
by sorry


end NUMINAMATH_CALUDE_race_result_l3149_314915


namespace NUMINAMATH_CALUDE_cereal_sugar_percentage_l3149_314933

/-- The percentage of sugar in cereal A -/
def sugar_a : ℝ := 10

/-- The ratio of cereal A to cereal B -/
def ratio : ℝ := 1

/-- The percentage of sugar in the final mixture -/
def sugar_mixture : ℝ := 6

/-- The percentage of sugar in cereal B -/
def sugar_b : ℝ := 2

theorem cereal_sugar_percentage :
  (sugar_a * ratio + sugar_b * ratio) / (ratio + ratio) = sugar_mixture :=
by sorry

end NUMINAMATH_CALUDE_cereal_sugar_percentage_l3149_314933


namespace NUMINAMATH_CALUDE_max_distance_theorem_l3149_314986

/-- Represents the characteristics of a motor boat on a river -/
structure RiverBoat where
  upstream_distance : ℝ  -- Distance the boat can travel upstream on a full tank
  downstream_distance : ℝ -- Distance the boat can travel downstream on a full tank

/-- Calculates the maximum round trip distance for a given boat -/
def max_round_trip_distance (boat : RiverBoat) : ℝ :=
  -- Implementation not provided, as per instructions
  sorry

/-- Theorem stating the maximum round trip distance for the given boat -/
theorem max_distance_theorem (boat : RiverBoat) 
  (h1 : boat.upstream_distance = 40)
  (h2 : boat.downstream_distance = 60) :
  max_round_trip_distance boat = 24 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_theorem_l3149_314986


namespace NUMINAMATH_CALUDE_subtracted_number_l3149_314909

theorem subtracted_number (x : ℤ) : 88 - x = 54 → x = 34 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_number_l3149_314909


namespace NUMINAMATH_CALUDE_intersection_P_Q_l3149_314919

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x * (x - 1) ≥ 0}
def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 3 * x^2 + 1}

-- Theorem statement
theorem intersection_P_Q : P ∩ Q = {x : ℝ | x > 1} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l3149_314919


namespace NUMINAMATH_CALUDE_triangle_properties_l3149_314920

/-- Given a triangle ABC with the following properties:
  * The area of the triangle is 3√15
  * b - c = 2, where b and c are sides of the triangle
  * cos A = -1/4, where A is an angle of the triangle
This theorem proves specific values for a, sin C, and cos(2A + π/6) -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (h_area : (1/2) * b * c * Real.sin A = 3 * Real.sqrt 15)
  (h_sides : b - c = 2)
  (h_cos_A : Real.cos A = -1/4) :
  a = 8 ∧ 
  Real.sin C = Real.sqrt 15 / 8 ∧
  Real.cos (2 * A + π/6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3149_314920


namespace NUMINAMATH_CALUDE_farm_legs_count_l3149_314971

/-- Calculates the total number of legs in a farm with ducks and horses -/
def total_legs (total_animals : ℕ) (num_ducks : ℕ) : ℕ :=
  let num_horses := total_animals - num_ducks
  let duck_legs := 2 * num_ducks
  let horse_legs := 4 * num_horses
  duck_legs + horse_legs

/-- Proves that in a farm with 11 animals, including 7 ducks and the rest horses, 
    the total number of legs is 30 -/
theorem farm_legs_count : total_legs 11 7 = 30 := by
  sorry

end NUMINAMATH_CALUDE_farm_legs_count_l3149_314971


namespace NUMINAMATH_CALUDE_tough_week_sales_800_l3149_314958

/-- The amount Haji's mother sells on a good week -/
def good_week_sales : ℝ := sorry

/-- The amount Haji's mother sells on a tough week -/
def tough_week_sales : ℝ := sorry

/-- The total amount Haji's mother makes in 5 good weeks and 3 tough weeks -/
def total_sales : ℝ := 10400

/-- Tough week sales are half of good week sales -/
axiom tough_week_half_good : tough_week_sales = good_week_sales / 2

/-- Total sales equation -/
axiom total_sales_equation : 5 * good_week_sales + 3 * tough_week_sales = total_sales

theorem tough_week_sales_800 : tough_week_sales = 800 := by
  sorry

end NUMINAMATH_CALUDE_tough_week_sales_800_l3149_314958


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3149_314968

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 3 - 1) :
  (1 - 1 / a) / ((a^2 - 1) / a) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3149_314968


namespace NUMINAMATH_CALUDE_sum_gcf_lcm_8_12_l3149_314903

theorem sum_gcf_lcm_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_gcf_lcm_8_12_l3149_314903
