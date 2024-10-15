import Mathlib

namespace NUMINAMATH_CALUDE_empty_box_weight_l2820_282069

def box_weight_problem (initial_weight : ℝ) (half_removed_weight : ℝ) : Prop :=
  ∃ (apple_weight : ℝ) (num_apples : ℕ) (box_weight : ℝ),
    initial_weight = box_weight + apple_weight * num_apples ∧
    half_removed_weight = box_weight + apple_weight * (num_apples / 2) ∧
    box_weight = 1

theorem empty_box_weight :
  box_weight_problem 9 5 := by
  sorry

end NUMINAMATH_CALUDE_empty_box_weight_l2820_282069


namespace NUMINAMATH_CALUDE_expected_potato_yield_l2820_282080

/-- Calculates the expected potato yield from a rectangular garden --/
theorem expected_potato_yield
  (length_steps : ℕ)
  (width_steps : ℕ)
  (step_length : ℝ)
  (yield_per_sqft : ℝ)
  (h1 : length_steps = 18)
  (h2 : width_steps = 25)
  (h3 : step_length = 3)
  (h4 : yield_per_sqft = 0.75)
  : ↑length_steps * step_length * (↑width_steps * step_length) * yield_per_sqft = 3037.5 := by
  sorry

end NUMINAMATH_CALUDE_expected_potato_yield_l2820_282080


namespace NUMINAMATH_CALUDE_derivative_symmetric_points_l2820_282034

/-- Given a function f(x) = ax^4 + bx^2 + c where f'(1) = 2, prove that f'(-1) = -2 -/
theorem derivative_symmetric_points 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^4 + b * x^2 + c)
  (h2 : deriv f 1 = 2) :
  deriv f (-1) = -2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_symmetric_points_l2820_282034


namespace NUMINAMATH_CALUDE_negative_a_squared_sum_l2820_282002

theorem negative_a_squared_sum (a : ℝ) : -3 * a^2 - 5 * a^2 = -8 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_squared_sum_l2820_282002


namespace NUMINAMATH_CALUDE_square_root_divided_by_15_equals_4_l2820_282025

theorem square_root_divided_by_15_equals_4 (x : ℝ) : 
  (Real.sqrt x) / 15 = 4 → x = 3600 := by
  sorry

end NUMINAMATH_CALUDE_square_root_divided_by_15_equals_4_l2820_282025


namespace NUMINAMATH_CALUDE_complex_division_simplification_l2820_282004

theorem complex_division_simplification :
  (3 + Complex.I) / (1 + Complex.I) = 2 - Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l2820_282004


namespace NUMINAMATH_CALUDE_coefficient_expansion_l2820_282058

theorem coefficient_expansion (a : ℝ) : 
  (∃ c : ℝ, c = 9 ∧ c = 1 + 4 * a) → a = 2 := by sorry

end NUMINAMATH_CALUDE_coefficient_expansion_l2820_282058


namespace NUMINAMATH_CALUDE_fred_grew_38_cantaloupes_l2820_282021

/-- The number of cantaloupes Tim grew -/
def tims_cantaloupes : ℕ := 44

/-- The total number of cantaloupes Fred and Tim grew together -/
def total_cantaloupes : ℕ := 82

/-- The number of cantaloupes Fred grew -/
def freds_cantaloupes : ℕ := total_cantaloupes - tims_cantaloupes

theorem fred_grew_38_cantaloupes : freds_cantaloupes = 38 := by
  sorry

end NUMINAMATH_CALUDE_fred_grew_38_cantaloupes_l2820_282021


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2820_282032

theorem complex_number_quadrant : 
  let z : ℂ := (1 + 2*I) / (1 - I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2820_282032


namespace NUMINAMATH_CALUDE_distance_A_to_B_l2820_282003

def point_A : Fin 3 → ℝ := ![2, 3, 5]
def point_B : Fin 3 → ℝ := ![3, 1, 7]

theorem distance_A_to_B :
  Real.sqrt ((point_B 0 - point_A 0)^2 + (point_B 1 - point_A 1)^2 + (point_B 2 - point_A 2)^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_A_to_B_l2820_282003


namespace NUMINAMATH_CALUDE_four_wheeler_wheels_l2820_282076

theorem four_wheeler_wheels (num_four_wheelers : ℕ) (wheels_per_four_wheeler : ℕ) : 
  num_four_wheelers = 17 → wheels_per_four_wheeler = 4 → num_four_wheelers * wheels_per_four_wheeler = 68 := by
  sorry

end NUMINAMATH_CALUDE_four_wheeler_wheels_l2820_282076


namespace NUMINAMATH_CALUDE_eggs_bought_l2820_282059

def initial_eggs : ℕ := 98
def final_eggs : ℕ := 106

theorem eggs_bought : final_eggs - initial_eggs = 8 := by
  sorry

end NUMINAMATH_CALUDE_eggs_bought_l2820_282059


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_value_l2820_282044

-- Define the curve
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x

-- Define the derivative of the curve
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem tangent_line_implies_a_value (a : ℝ) :
  (∀ x, x ≠ 0 → (f a x - f a 0) / (x - 0) ≤ 2) ∧
  (∀ x, x ≠ 0 → (f a x - f a 0) / (x - 0) ≥ 2) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_value_l2820_282044


namespace NUMINAMATH_CALUDE_savings_account_interest_rate_l2820_282000

theorem savings_account_interest_rate (initial_deposit : ℝ) (balance_after_first_year : ℝ) (total_increase_percentage : ℝ) : 
  initial_deposit = 5000 →
  balance_after_first_year = 5500 →
  total_increase_percentage = 21 →
  let total_balance := initial_deposit * (1 + total_increase_percentage / 100)
  let increase_second_year := total_balance - balance_after_first_year
  let percentage_increase_second_year := (increase_second_year / balance_after_first_year) * 100
  percentage_increase_second_year = 10 := by
sorry

end NUMINAMATH_CALUDE_savings_account_interest_rate_l2820_282000


namespace NUMINAMATH_CALUDE_lost_card_number_l2820_282073

theorem lost_card_number (n : ℕ) (h1 : n > 0) (h2 : (n * (n + 1)) / 2 - 101 ∈ Finset.range (n + 1)) : 
  (n * (n + 1)) / 2 - 101 = 4 := by
sorry

end NUMINAMATH_CALUDE_lost_card_number_l2820_282073


namespace NUMINAMATH_CALUDE_money_saved_monthly_payment_l2820_282029

/-- Calculates the money saved by paying monthly instead of weekly for a hotel stay. -/
theorem money_saved_monthly_payment (weekly_rate : ℕ) (monthly_rate : ℕ) (num_months : ℕ) 
  (h1 : weekly_rate = 280)
  (h2 : monthly_rate = 1000)
  (h3 : num_months = 3) :
  weekly_rate * 4 * num_months - monthly_rate * num_months = 360 := by
  sorry

#check money_saved_monthly_payment

end NUMINAMATH_CALUDE_money_saved_monthly_payment_l2820_282029


namespace NUMINAMATH_CALUDE_total_bread_slices_l2820_282060

/-- The number of sandwiches Ryan wants to make -/
def num_sandwiches : ℕ := 5

/-- The number of bread slices needed for each sandwich -/
def slices_per_sandwich : ℕ := 3

/-- Theorem: The total number of bread slices needed for Ryan's sandwiches is 15 -/
theorem total_bread_slices : num_sandwiches * slices_per_sandwich = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_bread_slices_l2820_282060


namespace NUMINAMATH_CALUDE_final_value_calculation_l2820_282043

theorem final_value_calculation : 
  let initial_value := 52
  let first_increase := initial_value * 1.20
  let second_decrease := first_increase * 0.90
  let final_increase := second_decrease * 1.15
  final_increase = 64.584 := by
sorry

end NUMINAMATH_CALUDE_final_value_calculation_l2820_282043


namespace NUMINAMATH_CALUDE_smallest_n_with_three_pairs_l2820_282047

/-- The function g(n) returns the number of distinct ordered pairs of positive integers (a, b) such that a^2 + b^2 + ab = n -/
def g (n : ℕ) : ℕ := (Finset.filter (fun p : ℕ × ℕ => p.1^2 + p.2^2 + p.1 * p.2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 48 is the smallest positive integer n for which g(n) = 3 -/
theorem smallest_n_with_three_pairs : (∀ m : ℕ, m > 0 ∧ m < 48 → g m ≠ 3) ∧ g 48 = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_with_three_pairs_l2820_282047


namespace NUMINAMATH_CALUDE_total_lockers_is_399_l2820_282010

/-- Represents the position of Minyoung's locker in the classroom -/
structure LockerPosition where
  front : ℕ
  back : ℕ
  left : ℕ
  right : ℕ

/-- Calculates the total number of lockers in the classroom based on Minyoung's locker position -/
def total_lockers (pos : LockerPosition) : ℕ :=
  (pos.front + pos.back - 1) * (pos.left + pos.right - 1)

/-- Theorem stating that the total number of lockers is 399 given Minyoung's locker position -/
theorem total_lockers_is_399 (pos : LockerPosition) 
  (h_front : pos.front = 8)
  (h_back : pos.back = 14)
  (h_left : pos.left = 7)
  (h_right : pos.right = 13) : 
  total_lockers pos = 399 := by
  sorry

#eval total_lockers ⟨8, 14, 7, 13⟩

end NUMINAMATH_CALUDE_total_lockers_is_399_l2820_282010


namespace NUMINAMATH_CALUDE_four_color_theorem_l2820_282092

/-- Represents a map as a planar graph -/
structure Map where
  vertices : Set Nat
  edges : Set (Nat × Nat)
  is_planar : Bool

/-- A coloring of a map -/
def Coloring (m : Map) := Nat → Fin 4

/-- Checks if a coloring is valid for a given map -/
def is_valid_coloring (m : Map) (c : Coloring m) : Prop :=
  ∀ (v₁ v₂ : Nat), (v₁, v₂) ∈ m.edges → c v₁ ≠ c v₂

/-- The Four Color Theorem -/
theorem four_color_theorem (m : Map) (h : m.is_planar = true) :
  ∃ (c : Coloring m), is_valid_coloring m c :=
sorry

end NUMINAMATH_CALUDE_four_color_theorem_l2820_282092


namespace NUMINAMATH_CALUDE_circle_center_problem_l2820_282009

/-- A circle tangent to two parallel lines with its center on a third line --/
theorem circle_center_problem (x y : ℝ) :
  (6 * x - 5 * y = 15) ∧ 
  (3 * x + 2 * y = 0) →
  x = 10 / 3 ∧ y = -5 := by
  sorry

#check circle_center_problem

end NUMINAMATH_CALUDE_circle_center_problem_l2820_282009


namespace NUMINAMATH_CALUDE_equidistant_implies_d_squared_l2820_282031

/-- A complex function g that scales by a complex number c+di -/
def g (c d : ℝ) (z : ℂ) : ℂ := (c + d * Complex.I) * z

/-- The property that g(z) is equidistant from z and the origin for all z -/
def equidistant (c d : ℝ) : Prop :=
  ∀ z : ℂ, Complex.abs (g c d z - z) = Complex.abs (g c d z)

theorem equidistant_implies_d_squared (c d : ℝ) 
  (h1 : equidistant c d) 
  (h2 : Complex.abs (c + d * Complex.I) = 5) : 
  d^2 = 99/4 := by sorry

end NUMINAMATH_CALUDE_equidistant_implies_d_squared_l2820_282031


namespace NUMINAMATH_CALUDE_baseball_league_games_l2820_282082

theorem baseball_league_games (n m : ℕ) : 
  (∃ (g₁ g₂ : Finset (Finset ℕ)), 
    (g₁.card = 4 ∧ g₂.card = 4) ∧ 
    (∀ t₁ ∈ g₁, ∀ t₂ ∈ g₁, t₁ ≠ t₂ → (∃ k : ℕ, k = n)) ∧
    (∀ t₁ ∈ g₁, ∀ t₂ ∈ g₂, (∃ k : ℕ, k = m)) ∧
    n > 2 * m ∧
    m > 4 ∧
    (∃ t ∈ g₁, 3 * n + 4 * m = 76)) →
  n = 48 := by sorry

end NUMINAMATH_CALUDE_baseball_league_games_l2820_282082


namespace NUMINAMATH_CALUDE_irrational_difference_representation_l2820_282041

theorem irrational_difference_representation (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  ∃ (α β : ℝ), Irrational α ∧ Irrational β ∧ 0 < α ∧ α < 1 ∧ 0 < β ∧ β < 1 ∧ x = α - β := by
  sorry

end NUMINAMATH_CALUDE_irrational_difference_representation_l2820_282041


namespace NUMINAMATH_CALUDE_fraction_nonnegative_l2820_282081

theorem fraction_nonnegative (x : ℝ) (h : x ≠ 3) : x^2 / (x - 3)^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_nonnegative_l2820_282081


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l2820_282083

theorem quadratic_always_positive (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + a > 0) ↔ (0 < a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l2820_282083


namespace NUMINAMATH_CALUDE_inequality_and_range_l2820_282024

theorem inequality_and_range (a b c m : ℝ) 
  (h1 : a + b + c + 2 - 2*m = 0)
  (h2 : a^2 + (1/4)*b^2 + (1/9)*c^2 + m - 1 = 0) :
  (a^2 + (1/4)*b^2 + (1/9)*c^2 ≥ (a + b + c)^2 / 14) ∧ 
  (-5/2 ≤ m ∧ m ≤ 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_and_range_l2820_282024


namespace NUMINAMATH_CALUDE_kyler_won_one_game_l2820_282046

/-- Represents a chess tournament between Peter, Emma, and Kyler -/
structure ChessTournament where
  total_games : ℕ
  peter_wins : ℕ
  peter_losses : ℕ
  emma_wins : ℕ
  emma_losses : ℕ
  kyler_losses : ℕ

/-- Calculates Kyler's wins in the chess tournament -/
def kyler_wins (t : ChessTournament) : ℕ :=
  t.total_games - (t.peter_wins + t.peter_losses + t.emma_wins + t.emma_losses + t.kyler_losses)

/-- Theorem stating that Kyler won 1 game in the given tournament conditions -/
theorem kyler_won_one_game (t : ChessTournament) 
  (h1 : t.total_games = 15)
  (h2 : t.peter_wins = 5)
  (h3 : t.peter_losses = 3)
  (h4 : t.emma_wins = 2)
  (h5 : t.emma_losses = 4)
  (h6 : t.kyler_losses = 4) :
  kyler_wins t = 1 := by
  sorry

end NUMINAMATH_CALUDE_kyler_won_one_game_l2820_282046


namespace NUMINAMATH_CALUDE_quartic_polynomial_unique_l2820_282068

/-- A monic quartic polynomial with real coefficients -/
def QuarticPolynomial (a b c d : ℝ) : ℝ → ℂ :=
  fun x ↦ (x^4 : ℂ) + a * (x^3 : ℂ) + b * (x^2 : ℂ) + c * (x : ℂ) + d

theorem quartic_polynomial_unique
  (q : ℝ → ℂ)
  (h_monic : ∀ x, q x = (x^4 : ℂ) + (a * x^3 : ℂ) + (b * x^2 : ℂ) + (c * x : ℂ) + d)
  (h_root : q (5 - 3*I) = 0)
  (h_constant : q 0 = -150) :
  q = QuarticPolynomial (-658/34) (19206/34) (-3822/17) (-150) :=
by sorry

end NUMINAMATH_CALUDE_quartic_polynomial_unique_l2820_282068


namespace NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l2820_282020

theorem rectangle_triangle_area_ratio :
  ∀ (L W : ℝ), L > 0 → W > 0 →
  (L * W) / ((1/2) * L * W) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_area_ratio_l2820_282020


namespace NUMINAMATH_CALUDE_binary_11101_equals_29_l2820_282072

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11101_equals_29 :
  binary_to_decimal [true, false, true, true, true] = 29 := by
  sorry

end NUMINAMATH_CALUDE_binary_11101_equals_29_l2820_282072


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2820_282013

def point : ℝ × ℝ := (8, -3)

theorem point_in_fourth_quadrant :
  let (x, y) := point
  x > 0 ∧ y < 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l2820_282013


namespace NUMINAMATH_CALUDE_math_english_time_difference_l2820_282053

/-- Represents an exam with a number of questions and a duration in hours -/
structure Exam where
  questions : ℕ
  duration : ℚ

/-- Calculates the time per question in minutes for a given exam -/
def timePerQuestion (e : Exam) : ℚ :=
  (e.duration * 60) / e.questions

theorem math_english_time_difference :
  let english : Exam := { questions := 30, duration := 1 }
  let math : Exam := { questions := 15, duration := 1.5 }
  timePerQuestion math - timePerQuestion english = 4 := by
  sorry

end NUMINAMATH_CALUDE_math_english_time_difference_l2820_282053


namespace NUMINAMATH_CALUDE_square_of_six_y_minus_four_l2820_282056

theorem square_of_six_y_minus_four (y : ℝ) (h : 3 * y^2 + 6 = 5 * y + 15) : 
  (6 * y - 4)^2 = 134 := by
  sorry

end NUMINAMATH_CALUDE_square_of_six_y_minus_four_l2820_282056


namespace NUMINAMATH_CALUDE_four_roots_iff_a_in_range_l2820_282042

-- Define the function f(x) = |x^2 + 3x|
def f (x : ℝ) : ℝ := |x^2 + 3*x|

-- Define the equation f(x) - a|x-1| = 0
def equation (a : ℝ) (x : ℝ) : Prop := f x - a * |x - 1| = 0

-- Define the property of having exactly 4 distinct real roots
def has_four_distinct_roots (a : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    equation a x₁ ∧ equation a x₂ ∧ equation a x₃ ∧ equation a x₄ ∧
    ∀ (x : ℝ), equation a x → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄

-- Theorem statement
theorem four_roots_iff_a_in_range :
  ∀ a : ℝ, has_four_distinct_roots a ↔ (a ∈ Set.Ioo 0 1 ∪ Set.Ioi 9) :=
sorry

end NUMINAMATH_CALUDE_four_roots_iff_a_in_range_l2820_282042


namespace NUMINAMATH_CALUDE_percent_greater_relative_to_sum_l2820_282067

/-- Given two real numbers M and N, this theorem states that the percentage
    by which M is greater than N, relative to their sum, is (100(M-N))/(M+N). -/
theorem percent_greater_relative_to_sum (M N : ℝ) :
  (M - N) / (M + N) * 100 = (100 * (M - N)) / (M + N) := by sorry

end NUMINAMATH_CALUDE_percent_greater_relative_to_sum_l2820_282067


namespace NUMINAMATH_CALUDE_function_decomposition_l2820_282015

-- Define a type for the domain that is symmetric with respect to the origin
structure SymmetricDomain where
  X : Type
  symm : X → X
  symm_involutive : ∀ x, symm (symm x) = x

-- Define a function on the symmetric domain
def Function (D : SymmetricDomain) := D.X → ℝ

-- Define an even function
def IsEven (D : SymmetricDomain) (f : Function D) : Prop :=
  ∀ x, f (D.symm x) = f x

-- Define an odd function
def IsOdd (D : SymmetricDomain) (f : Function D) : Prop :=
  ∀ x, f (D.symm x) = -f x

-- State the theorem
theorem function_decomposition (D : SymmetricDomain) (f : Function D) :
  ∃! (e o : Function D), (∀ x, f x = e x + o x) ∧ IsEven D e ∧ IsOdd D o := by
  sorry

end NUMINAMATH_CALUDE_function_decomposition_l2820_282015


namespace NUMINAMATH_CALUDE_most_accurate_reading_is_10_45_l2820_282057

/-- Represents a scientific weighing scale --/
structure ScientificScale where
  smallest_division : ℝ
  lower_bound : ℝ
  upper_bound : ℝ
  marker_position : ℝ

/-- Determines if a given reading is the most accurate for a scientific scale --/
def is_most_accurate_reading (s : ScientificScale) (reading : ℝ) : Prop :=
  s.lower_bound < reading ∧ 
  reading < s.upper_bound ∧ 
  reading % s.smallest_division = 0 ∧
  ∀ r, s.lower_bound < r ∧ r < s.upper_bound ∧ r % s.smallest_division = 0 → 
    |s.marker_position - reading| ≤ |s.marker_position - r|

/-- The theorem stating the most accurate reading for the given scale --/
theorem most_accurate_reading_is_10_45 (s : ScientificScale) 
  (h_division : s.smallest_division = 0.01)
  (h_lower : s.lower_bound = 10.41)
  (h_upper : s.upper_bound = 10.55)
  (h_marker : s.lower_bound < s.marker_position ∧ s.marker_position < (s.lower_bound + s.upper_bound) / 2) :
  is_most_accurate_reading s 10.45 :=
sorry

end NUMINAMATH_CALUDE_most_accurate_reading_is_10_45_l2820_282057


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2820_282037

theorem inequality_and_equality_condition (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  Real.sqrt 2 * (Real.sqrt (a * (a + b)^3) + b * Real.sqrt (a^2 + b^2)) ≤ 3 * (a^2 + b^2) ∧
  (Real.sqrt 2 * (Real.sqrt (a * (a + b)^3) + b * Real.sqrt (a^2 + b^2)) = 3 * (a^2 + b^2) ↔ a = b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2820_282037


namespace NUMINAMATH_CALUDE_percentage_of_male_employees_l2820_282091

theorem percentage_of_male_employees
  (total_employees : ℕ)
  (males_below_50 : ℕ)
  (h_total : total_employees = 5200)
  (h_below_50 : males_below_50 = 1170)
  (h_half_above_50 : males_below_50 = (total_employees * (percentage_males / 100) / 2)) :
  percentage_males = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_of_male_employees_l2820_282091


namespace NUMINAMATH_CALUDE_max_value_is_57_l2820_282026

/-- Represents a type of rock with its weight and value -/
structure Rock where
  weight : Nat
  value : Nat

/-- The problem setup -/
def rockTypes : List Rock := [
  { weight := 6, value := 18 },
  { weight := 3, value := 9 },
  { weight := 2, value := 3 }
]

/-- The maximum weight Carl can carry -/
def maxWeight : Nat := 20

/-- The minimum number of rocks available for each type -/
def minRocksPerType : Nat := 15

/-- A function to calculate the total value of a collection of rocks -/
def totalValue (rocks : List (Rock × Nat)) : Nat :=
  rocks.foldl (fun acc (rock, count) => acc + rock.value * count) 0

/-- A function to calculate the total weight of a collection of rocks -/
def totalWeight (rocks : List (Rock × Nat)) : Nat :=
  rocks.foldl (fun acc (rock, count) => acc + rock.weight * count) 0

/-- The main theorem stating that the maximum value Carl can carry is $57 -/
theorem max_value_is_57 :
  ∃ (rocks : List (Rock × Nat)),
    (∀ r ∈ rocks, r.1 ∈ rockTypes) ∧
    (∀ r ∈ rocks, r.2 ≤ minRocksPerType) ∧
    totalWeight rocks ≤ maxWeight ∧
    totalValue rocks = 57 ∧
    (∀ (other_rocks : List (Rock × Nat)),
      (∀ r ∈ other_rocks, r.1 ∈ rockTypes) →
      (∀ r ∈ other_rocks, r.2 ≤ minRocksPerType) →
      totalWeight other_rocks ≤ maxWeight →
      totalValue other_rocks ≤ 57) :=
by sorry


end NUMINAMATH_CALUDE_max_value_is_57_l2820_282026


namespace NUMINAMATH_CALUDE_shortest_ribbon_length_l2820_282035

theorem shortest_ribbon_length (a b c d : ℕ) (ha : a = 2) (hb : b = 5) (hc : c = 7) (hd : d = 11) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 770 :=
by sorry

end NUMINAMATH_CALUDE_shortest_ribbon_length_l2820_282035


namespace NUMINAMATH_CALUDE_evie_shell_collection_l2820_282063

theorem evie_shell_collection (daily_shells : ℕ) : 
  (6 * daily_shells - 2 = 58) → daily_shells = 10 := by
  sorry

end NUMINAMATH_CALUDE_evie_shell_collection_l2820_282063


namespace NUMINAMATH_CALUDE_power_equation_solution_l2820_282030

theorem power_equation_solution : ∃ x : ℕ, 2^4 + 3 = 5^2 - x ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2820_282030


namespace NUMINAMATH_CALUDE_number_is_composite_l2820_282070

theorem number_is_composite : ∃ (k : ℕ), k > 1 ∧ k ∣ (53 * 83 * 109 + 40 * 66 * 96) := by
  -- We claim that 149 divides the given number
  use 149
  constructor
  · -- 149 > 1
    norm_num
  · -- 149 divides the given number
    sorry


end NUMINAMATH_CALUDE_number_is_composite_l2820_282070


namespace NUMINAMATH_CALUDE_absolute_value_inequality_range_l2820_282095

theorem absolute_value_inequality_range :
  ∀ a : ℝ, (∀ x : ℝ, |x + 3| + |x - 1| ≥ a) ↔ a ≤ 4 :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_range_l2820_282095


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2820_282065

/-- Given a geometric sequence {a_n} where a_2020 = 8a_2017, prove that the common ratio q is 2. -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  a 2020 = 8 * a 2017 →         -- given condition
  q = 2 :=                      -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2820_282065


namespace NUMINAMATH_CALUDE_point_on_linear_graph_l2820_282005

/-- Given that the point (a, -1) lies on the graph of y = -2x + 1, prove that a = 1 -/
theorem point_on_linear_graph (a : ℝ) : 
  -1 = -2 * a + 1 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_linear_graph_l2820_282005


namespace NUMINAMATH_CALUDE_square_rectangle_perimeter_equality_l2820_282096

theorem square_rectangle_perimeter_equality :
  ∀ (square_side : ℝ) (rect_length rect_area : ℝ),
    square_side = 15 →
    rect_length = 18 →
    rect_area = 216 →
    4 * square_side = 2 * (rect_length + (rect_area / rect_length)) := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_perimeter_equality_l2820_282096


namespace NUMINAMATH_CALUDE_fraction_simplification_l2820_282049

theorem fraction_simplification : (3 : ℚ) / (2 - 3 / 4) = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2820_282049


namespace NUMINAMATH_CALUDE_smallest_area_right_triangle_l2820_282019

/-- The smallest possible area of a right triangle with two sides measuring 6 and 8 units is 24 square units. -/
theorem smallest_area_right_triangle : ℝ := by
  -- Let a and b be the two given sides of the right triangle
  let a : ℝ := 6
  let b : ℝ := 8
  
  -- Define the function to calculate the area of a right triangle
  let area (x y : ℝ) : ℝ := (1 / 2) * x * y
  
  -- State that the smallest area is 24
  let smallest_area : ℝ := 24
  
  sorry

end NUMINAMATH_CALUDE_smallest_area_right_triangle_l2820_282019


namespace NUMINAMATH_CALUDE_sum_abcd_equals_negative_ten_thirds_l2820_282038

theorem sum_abcd_equals_negative_ten_thirds 
  (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 6) : 
  a + b + c + d = -10/3 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_negative_ten_thirds_l2820_282038


namespace NUMINAMATH_CALUDE_set_membership_and_inclusion_l2820_282074

def A : Set ℤ := {x | ∃ m n : ℤ, x = m^2 - n^2}
def B : Set ℤ := {x | ∃ k : ℤ, x = 2*k + 1}

theorem set_membership_and_inclusion :
  (8 ∈ A ∧ 9 ∈ A ∧ 10 ∉ A) ∧ (∀ x : ℤ, x ∈ A → x ∈ B) := by sorry

end NUMINAMATH_CALUDE_set_membership_and_inclusion_l2820_282074


namespace NUMINAMATH_CALUDE_equality_implies_equation_l2820_282001

theorem equality_implies_equation (x y : ℝ) (h : x = y) : -1/3 * x + 1 = -1/3 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_equality_implies_equation_l2820_282001


namespace NUMINAMATH_CALUDE_mitch_hourly_rate_l2820_282084

/-- Mitch's hourly rate calculation --/
theorem mitch_hourly_rate :
  ∀ (weekday_hours_per_day : ℕ) 
    (weekend_hours_per_day : ℕ) 
    (weekday_count : ℕ) 
    (weekend_count : ℕ) 
    (weekend_multiplier : ℕ) 
    (weekly_earnings : ℕ),
  weekday_hours_per_day = 5 →
  weekend_hours_per_day = 3 →
  weekday_count = 5 →
  weekend_count = 2 →
  weekend_multiplier = 2 →
  weekly_earnings = 111 →
  (weekly_earnings : ℚ) / 
    (weekday_hours_per_day * weekday_count + 
     weekend_hours_per_day * weekend_count * weekend_multiplier) = 3 := by
  sorry

#check mitch_hourly_rate

end NUMINAMATH_CALUDE_mitch_hourly_rate_l2820_282084


namespace NUMINAMATH_CALUDE_expand_product_l2820_282036

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12*x + 27 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2820_282036


namespace NUMINAMATH_CALUDE_f_properties_l2820_282039

-- Define the function f(x) = lg |sin x|
noncomputable def f (x : ℝ) : ℝ := Real.log (|Real.sin x|)

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = f x) ∧                        -- f is even
  (∀ x, f (x + π) = f x) ∧                     -- f has period π
  (∀ x y, 0 < x ∧ x < y ∧ y < π/2 → f x < f y) -- f is monotonically increasing on (0, π/2)
  := by sorry

end NUMINAMATH_CALUDE_f_properties_l2820_282039


namespace NUMINAMATH_CALUDE_age_problem_l2820_282051

theorem age_problem (A B C : ℕ) : 
  (A + B + C) / 3 = 26 →
  (A + C) / 2 = 29 →
  B = 20 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l2820_282051


namespace NUMINAMATH_CALUDE_linear_equation_solution_l2820_282077

theorem linear_equation_solution (x y : ℝ) : 5 * x + y = 4 → y = 4 - 5 * x := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l2820_282077


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2820_282099

theorem geometric_sequence_problem (a b c r : ℤ) : 
  (b = a * r ∧ c = a * r^2) →  -- geometric sequence condition
  (r ≠ 0) →                   -- non-zero ratio
  (c = a + 56) →              -- given condition
  b = 21 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2820_282099


namespace NUMINAMATH_CALUDE_smallest_square_containing_circle_l2820_282022

theorem smallest_square_containing_circle (r : ℝ) (h : r = 7) : 
  (2 * r) ^ 2 = 196 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_containing_circle_l2820_282022


namespace NUMINAMATH_CALUDE_m_range_l2820_282086

theorem m_range : 
  (∀ x, (|x - m| < 1 ↔ 1/3 < x ∧ x < 1/2)) → 
  (-1/2 ≤ m ∧ m ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_m_range_l2820_282086


namespace NUMINAMATH_CALUDE_minimum_value_of_f_plus_f_prime_l2820_282016

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

-- Define the derivative of f
def f_prime (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

theorem minimum_value_of_f_plus_f_prime (a : ℝ) :
  (∃ x, f_prime a x = 0 ∧ x = 2) →
  (∀ m n, m ∈ Set.Icc (-1 : ℝ) 1 → n ∈ Set.Icc (-1 : ℝ) 1 →
    f a m + f_prime a n ≥ -13) ∧
  (∃ m n, m ∈ Set.Icc (-1 : ℝ) 1 ∧ n ∈ Set.Icc (-1 : ℝ) 1 ∧
    f a m + f_prime a n = -13) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_of_f_plus_f_prime_l2820_282016


namespace NUMINAMATH_CALUDE_expected_heads_is_60_l2820_282052

/-- The number of coins -/
def num_coins : ℕ := 64

/-- The maximum number of flips per coin -/
def max_flips : ℕ := 4

/-- The probability of getting heads on a single flip -/
def p_heads : ℚ := 1/2

/-- The probability of getting heads after up to four flips -/
def p_heads_total : ℚ := 1 - (1 - p_heads)^max_flips

/-- The expected number of coins showing heads after up to four flips -/
def expected_heads : ℚ := num_coins * p_heads_total

theorem expected_heads_is_60 : expected_heads = 60 := by
  sorry

end NUMINAMATH_CALUDE_expected_heads_is_60_l2820_282052


namespace NUMINAMATH_CALUDE_speed_conversion_l2820_282078

/-- Conversion factor from kilometers per hour to meters per second -/
def kmph_to_mps : ℚ := 5 / 18

/-- The given speed in kilometers per hour -/
def speed_kmph : ℚ := 216

/-- The speed in meters per second -/
def speed_mps : ℚ := speed_kmph * kmph_to_mps

theorem speed_conversion :
  speed_mps = 60 := by sorry

end NUMINAMATH_CALUDE_speed_conversion_l2820_282078


namespace NUMINAMATH_CALUDE_root_implies_t_value_l2820_282033

theorem root_implies_t_value (t : ℝ) : 
  (3 * (((-15 - Real.sqrt 145) / 6) ^ 2) + 15 * ((-15 - Real.sqrt 145) / 6) + t = 0) → 
  t = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_root_implies_t_value_l2820_282033


namespace NUMINAMATH_CALUDE_tom_theater_expenditure_l2820_282062

/-- Calculates Tom's expenditure for opening a theater --/
theorem tom_theater_expenditure
  (cost_per_sq_ft : ℝ)
  (space_per_seat : ℝ)
  (num_seats : ℕ)
  (construction_cost_multiplier : ℝ)
  (partner_contribution_percentage : ℝ)
  (h1 : cost_per_sq_ft = 5)
  (h2 : space_per_seat = 12)
  (h3 : num_seats = 500)
  (h4 : construction_cost_multiplier = 2)
  (h5 : partner_contribution_percentage = 0.4)
  : tom_expenditure = 54000 := by
  sorry

where
  tom_expenditure : ℝ :=
    let total_sq_ft := space_per_seat * num_seats
    let land_cost := cost_per_sq_ft * total_sq_ft
    let construction_cost := construction_cost_multiplier * land_cost
    let total_cost := land_cost + construction_cost
    (1 - partner_contribution_percentage) * total_cost

end NUMINAMATH_CALUDE_tom_theater_expenditure_l2820_282062


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_l2820_282040

theorem lcm_gcf_ratio : (Nat.lcm 256 162) / (Nat.gcd 256 162) = 10368 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_l2820_282040


namespace NUMINAMATH_CALUDE_complement_A_in_U_l2820_282055

def U : Set ℝ := {x | x^2 ≤ 4}
def A : Set ℝ := {x | |x + 1| ≤ 1}

theorem complement_A_in_U : 
  (U \ A) = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l2820_282055


namespace NUMINAMATH_CALUDE_max_value_problem_l2820_282066

theorem max_value_problem (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (a / (a + 1)) + (b / (b + 2)) ≤ (5 - 2 * Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_problem_l2820_282066


namespace NUMINAMATH_CALUDE_distance_A_B_min_value_expression_solutions_equation_max_product_mn_l2820_282064

-- Define the distance function on a number line
def distance (a b : ℝ) : ℝ := |a - b|

-- Statement 1
theorem distance_A_B : distance (-10) 8 = 18 := by sorry

-- Statement 2
theorem min_value_expression : 
  ∀ x : ℝ, |x - 3| + |x + 2| ≥ 5 := by sorry

-- Statement 3
theorem solutions_equation : 
  ∀ y : ℝ, |y - 3| + |y + 1| = 8 ↔ y = 5 ∨ y = -3 := by sorry

-- Statement 4
theorem max_product_mn : 
  ∀ m n : ℤ, (|m + 1| + |2 - m|) * (|n - 1| + |n + 3|) = 12 → 
  m * n ≤ 3 := by sorry

end NUMINAMATH_CALUDE_distance_A_B_min_value_expression_solutions_equation_max_product_mn_l2820_282064


namespace NUMINAMATH_CALUDE_length_BC_in_triangle_l2820_282054

/-- Parabola function -/
def parabola (x : ℝ) : ℝ := 2 * x^2

/-- Triangle ABC -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Theorem: Length of BC in triangle ABC -/
theorem length_BC_in_triangle (t : Triangle) : 
  (t.A.1 = 0 ∧ t.A.2 = 0) →  -- A is at origin
  (t.B.2 = parabola t.B.1) →  -- B is on parabola
  (t.C.2 = parabola t.C.1) →  -- C is on parabola
  (t.B.2 = t.C.2) →  -- BC is parallel to x-axis
  (1/2 * (t.C.1 - t.B.1) * t.B.2 = 128) →  -- Area of triangle is 128
  (t.C.1 - t.B.1 = 8) :=  -- Length of BC is 8
by sorry

end NUMINAMATH_CALUDE_length_BC_in_triangle_l2820_282054


namespace NUMINAMATH_CALUDE_expression_value_l2820_282028

theorem expression_value (x y z : ℤ) (hx : x = 3) (hy : y = 2) (hz : z = -1) :
  3 * x^2 - 4 * y + 5 * z = 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2820_282028


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l2820_282088

theorem cubic_sum_over_product (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 18)
  (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 21 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l2820_282088


namespace NUMINAMATH_CALUDE_handbag_discount_proof_l2820_282018

theorem handbag_discount_proof (initial_price : ℝ) (regular_discount : ℝ) (monday_discount : ℝ) :
  initial_price = 250 →
  regular_discount = 0.4 →
  monday_discount = 0.1 →
  let price_after_regular_discount := initial_price * (1 - regular_discount)
  let final_price := price_after_regular_discount * (1 - monday_discount)
  final_price = 135 :=
by sorry

end NUMINAMATH_CALUDE_handbag_discount_proof_l2820_282018


namespace NUMINAMATH_CALUDE_angle_CAD_is_15_degrees_l2820_282097

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Represents a square in 2D space -/
structure Square where
  B : Point2D
  C : Point2D
  D : Point2D
  E : Point2D

/-- Calculates the angle between three points in degrees -/
def angle (p1 p2 p3 : Point2D) : ℝ := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Checks if a quadrilateral is a square -/
def isSquare (s : Square) : Prop := sorry

/-- Theorem: In a coplanar configuration where ABC is an equilateral triangle 
    and BCDE is a square, the measure of angle CAD is 15 degrees -/
theorem angle_CAD_is_15_degrees 
  (A B C D E : Point2D) 
  (triangle : Triangle) 
  (square : Square) : 
  triangle.A = A ∧ triangle.B = B ∧ triangle.C = C ∧
  square.B = B ∧ square.C = C ∧ square.D = D ∧ square.E = E ∧
  isEquilateral triangle ∧ 
  isSquare square → 
  angle C A D = 15 := by
  sorry

end NUMINAMATH_CALUDE_angle_CAD_is_15_degrees_l2820_282097


namespace NUMINAMATH_CALUDE_x_squared_coefficient_l2820_282045

def expansion (x : ℝ) := (2*x + 1) * (x - 2)^3

theorem x_squared_coefficient : 
  (∃ a b c d : ℝ, expansion x = a*x^3 + b*x^2 + c*x + d) → 
  (∃ a c d : ℝ, expansion x = a*x^3 + 18*x^2 + c*x + d) :=
by sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_l2820_282045


namespace NUMINAMATH_CALUDE_negation_of_forall_inequality_negation_of_inequality_negation_of_proposition_l2820_282012

theorem negation_of_forall_inequality (P : ℝ → Prop) :
  (¬ ∀ x < 0, P x) ↔ (∃ x < 0, ¬ P x) := by sorry

theorem negation_of_inequality (x : ℝ) :
  ¬(1 - x > Real.exp x) ↔ (1 - x ≤ Real.exp x) := by sorry

theorem negation_of_proposition :
  (¬ ∀ x < 0, 1 - x > Real.exp x) ↔ (∃ x < 0, 1 - x ≤ Real.exp x) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_inequality_negation_of_inequality_negation_of_proposition_l2820_282012


namespace NUMINAMATH_CALUDE_robin_ate_twelve_cupcakes_l2820_282048

/-- The number of cupcakes Robin ate with chocolate sauce -/
def chocolate_cupcakes : ℕ := 4

/-- The number of cupcakes Robin ate with buttercream frosting -/
def buttercream_cupcakes : ℕ := 2 * chocolate_cupcakes

/-- The total number of cupcakes Robin ate -/
def total_cupcakes : ℕ := chocolate_cupcakes + buttercream_cupcakes

theorem robin_ate_twelve_cupcakes : total_cupcakes = 12 := by
  sorry

end NUMINAMATH_CALUDE_robin_ate_twelve_cupcakes_l2820_282048


namespace NUMINAMATH_CALUDE_spade_problem_l2820_282011

/-- Custom operation ⊙ for real numbers -/
def spade (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

/-- Theorem stating that 2 ⊙ (3 ⊙ 4) = 384 -/
theorem spade_problem : spade 2 (spade 3 4) = 384 := by
  sorry

end NUMINAMATH_CALUDE_spade_problem_l2820_282011


namespace NUMINAMATH_CALUDE_two_power_minus_three_power_eq_one_solutions_l2820_282007

theorem two_power_minus_three_power_eq_one_solutions :
  ∀ m n : ℕ, 2^m - 3^n = 1 ↔ (m = 1 ∧ n = 0) ∨ (m = 2 ∧ n = 1) := by
  sorry

end NUMINAMATH_CALUDE_two_power_minus_three_power_eq_one_solutions_l2820_282007


namespace NUMINAMATH_CALUDE_helen_cookies_proof_l2820_282061

/-- The number of cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 31

/-- The number of cookies Helen baked the day before yesterday -/
def cookies_day_before_yesterday : ℕ := 419

/-- The total number of cookies Helen baked until last night -/
def total_cookies : ℕ := cookies_yesterday + cookies_day_before_yesterday

theorem helen_cookies_proof : total_cookies = 450 := by
  sorry

end NUMINAMATH_CALUDE_helen_cookies_proof_l2820_282061


namespace NUMINAMATH_CALUDE_wall_width_proof_l2820_282089

theorem wall_width_proof (width height length volume : ℝ) : 
  height = 6 * width →
  length = 7 * height →
  volume = length * width * height →
  volume = 16128 →
  width = (384 : ℝ) ^ (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_wall_width_proof_l2820_282089


namespace NUMINAMATH_CALUDE_angle_DAE_is_10_degrees_l2820_282098

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def angle (A B C : ℝ × ℝ) : ℝ := sorry

theorem angle_DAE_is_10_degrees 
  (A B C D O E: ℝ × ℝ) 
  (triangle : Triangle A B C) 
  (h1 : angle A C B = 60)
  (h2 : angle C B A = 70)
  (h3 : D.1 = B.1 + (C.1 - B.1) * ((A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2)) / ((C.1 - B.1)^2 + (C.2 - B.2)^2))
  (h4 : D.2 = B.2 + (C.2 - B.2) * ((A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2)) / ((C.1 - B.1)^2 + (C.2 - B.2)^2))
  (h5 : ∃ r, Circle O r = {A, B, C})
  (h6 : E.1 = 2 * O.1 - A.1 ∧ E.2 = 2 * O.2 - A.2) :
  angle D A E = 10 := by
sorry


end NUMINAMATH_CALUDE_angle_DAE_is_10_degrees_l2820_282098


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2820_282017

theorem complex_number_quadrant (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2820_282017


namespace NUMINAMATH_CALUDE_andy_final_position_l2820_282090

/-- Represents a point in 2D space -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction -/
inductive Direction
  | East
  | North
  | West
  | South

/-- Represents the state of Andy the Ant -/
structure AntState where
  position : Point
  direction : Direction
  moveCount : Nat

/-- Performs a single move for Andy the Ant -/
def move (state : AntState) : AntState :=
  sorry

/-- Performs n moves for Andy the Ant -/
def moveN (n : Nat) (state : AntState) : AntState :=
  sorry

/-- The main theorem to prove -/
theorem andy_final_position :
  let initialState : AntState := {
    position := { x := -10, y := 10 },
    direction := Direction.East,
    moveCount := 0
  }
  let finalState := moveN 2030 initialState
  finalState.position = { x := -3054, y := 3053 } :=
sorry

end NUMINAMATH_CALUDE_andy_final_position_l2820_282090


namespace NUMINAMATH_CALUDE_employee_payment_l2820_282071

theorem employee_payment (total : ℝ) (x y : ℝ) (h1 : total = 528) (h2 : x = 1.2 * y) (h3 : total = x + y) : y = 240 := by
  sorry

end NUMINAMATH_CALUDE_employee_payment_l2820_282071


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2820_282087

theorem quadratic_real_roots (k m : ℝ) : 
  (∃ x : ℝ, x^2 + (2*k - 3*m)*x + (k^2 - 5*k*m + 6*m^2) = 0) ↔ k ≥ (15/8)*m :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2820_282087


namespace NUMINAMATH_CALUDE_tan_22_5_deg_identity_l2820_282085

theorem tan_22_5_deg_identity : 
  (Real.tan (22.5 * π / 180)) / (1 - (Real.tan (22.5 * π / 180))^2) = 1/2 := by sorry

end NUMINAMATH_CALUDE_tan_22_5_deg_identity_l2820_282085


namespace NUMINAMATH_CALUDE_sum_of_numbers_l2820_282094

theorem sum_of_numbers (a b : ℕ) (h : (a + b) * (a - b) = 1996) : a + b = 998 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l2820_282094


namespace NUMINAMATH_CALUDE_sales_difference_l2820_282050

-- Define the regular day sales quantities
def regular_croissants : ℕ := 10
def regular_muffins : ℕ := 10
def regular_sourdough : ℕ := 6
def regular_wholewheat : ℕ := 4

-- Define the Monday sales quantities
def monday_croissants : ℕ := 8
def monday_muffins : ℕ := 6
def monday_sourdough : ℕ := 15
def monday_wholewheat : ℕ := 10

-- Define the regular prices
def price_croissant : ℚ := 2.5
def price_muffin : ℚ := 1.75
def price_sourdough : ℚ := 4.25
def price_wholewheat : ℚ := 5

-- Define the discount rate
def discount_rate : ℚ := 0.1

-- Calculate the daily average sales
def daily_average : ℚ :=
  regular_croissants * price_croissant +
  regular_muffins * price_muffin +
  regular_sourdough * price_sourdough +
  regular_wholewheat * price_wholewheat

-- Calculate the Monday sales with discount
def monday_sales : ℚ :=
  monday_croissants * price_croissant * (1 - discount_rate) +
  monday_muffins * price_muffin * (1 - discount_rate) +
  monday_sourdough * price_sourdough * (1 - discount_rate) +
  monday_wholewheat * price_wholewheat * (1 - discount_rate)

-- State the theorem
theorem sales_difference : monday_sales - daily_average = 41.825 := by sorry

end NUMINAMATH_CALUDE_sales_difference_l2820_282050


namespace NUMINAMATH_CALUDE_total_pets_is_54_l2820_282075

/-- The number of pets owned by Teddy, Ben, and Dave -/
def total_pets : ℕ :=
  let teddy_dogs : ℕ := 7
  let teddy_cats : ℕ := 8
  let ben_extra_dogs : ℕ := 9
  let dave_extra_cats : ℕ := 13
  let dave_fewer_dogs : ℕ := 5

  let teddy_pets : ℕ := teddy_dogs + teddy_cats
  let ben_pets : ℕ := (teddy_dogs + ben_extra_dogs)
  let dave_pets : ℕ := (teddy_cats + dave_extra_cats) + (teddy_dogs - dave_fewer_dogs)

  teddy_pets + ben_pets + dave_pets

theorem total_pets_is_54 : total_pets = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_pets_is_54_l2820_282075


namespace NUMINAMATH_CALUDE_cars_per_salesperson_per_month_l2820_282093

/-- Proves that given 500 cars for sale, 10 sales professionals, and a 5-month period to sell all cars, each salesperson sells 10 cars per month. -/
theorem cars_per_salesperson_per_month 
  (total_cars : ℕ) 
  (sales_professionals : ℕ) 
  (months_to_sell : ℕ) 
  (h1 : total_cars = 500) 
  (h2 : sales_professionals = 10) 
  (h3 : months_to_sell = 5) :
  total_cars / (sales_professionals * months_to_sell) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_cars_per_salesperson_per_month_l2820_282093


namespace NUMINAMATH_CALUDE_right_triangle_existence_unique_non_right_triangle_l2820_282008

theorem right_triangle_existence (a b c : ℝ) : Bool :=
  a * a + b * b = c * c

theorem unique_non_right_triangle : 
  right_triangle_existence 3 4 5 = true ∧
  right_triangle_existence 1 1 (Real.sqrt 2) = true ∧
  right_triangle_existence 8 15 18 = false ∧
  right_triangle_existence 5 12 13 = true ∧
  right_triangle_existence 6 8 10 = true :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_existence_unique_non_right_triangle_l2820_282008


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_implies_m_eq_six_l2820_282079

/-- Represents a hyperbola with equation x²/m - y²/6 = 1 -/
structure Hyperbola (m : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / m - y^2 / 6 = 1

/-- Represents an asymptote of a hyperbola -/
structure Asymptote (m : ℝ) where
  slope : ℝ
  eq : ∀ (x y : ℝ), y = slope * x

/-- 
If a hyperbola with equation x²/m - y²/6 = 1 has an asymptote y = x,
then m = 6
-/
theorem hyperbola_asymptote_implies_m_eq_six (m : ℝ) 
  (h : Hyperbola m) 
  (a : Asymptote m) 
  (ha : a.slope = 1) : m = 6 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_implies_m_eq_six_l2820_282079


namespace NUMINAMATH_CALUDE_bus_passengers_l2820_282014

theorem bus_passengers (men women : ℕ) : 
  women = men / 3 →
  men - 24 = women + 12 →
  men + women = 72 :=
by sorry

end NUMINAMATH_CALUDE_bus_passengers_l2820_282014


namespace NUMINAMATH_CALUDE_lcm_factor_proof_l2820_282023

theorem lcm_factor_proof (A B : ℕ) (x : ℕ) (h1 : Nat.gcd A B = 23) (h2 : A = 391) 
  (h3 : Nat.lcm A B = 23 * 17 * x) : x = 17 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_proof_l2820_282023


namespace NUMINAMATH_CALUDE_final_price_after_discounts_l2820_282006

def original_price : ℝ := 15
def first_discount_rate : ℝ := 0.20
def second_discount_rate : ℝ := 0.25

theorem final_price_after_discounts :
  (original_price * (1 - first_discount_rate) * (1 - second_discount_rate)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_final_price_after_discounts_l2820_282006


namespace NUMINAMATH_CALUDE_amelia_wins_probability_l2820_282027

/-- Probability of Amelia's coin landing heads -/
def p_amelia : ℚ := 1/4

/-- Probability of Blaine's coin landing heads -/
def p_blaine : ℚ := 3/7

/-- Maximum number of rounds -/
def max_rounds : ℕ := 5

/-- The probability that Amelia wins the coin toss game -/
def amelia_wins_prob : ℚ := 223/784

/-- Theorem stating that the probability of Amelia winning is 223/784 -/
theorem amelia_wins_probability : 
  amelia_wins_prob = p_amelia * (1 - p_blaine) + 
    (1 - p_amelia) * (1 - p_blaine) * p_amelia * (1 - p_blaine) + 
    (1 - p_amelia) * (1 - p_blaine) * (1 - p_amelia) * (1 - p_blaine) * p_amelia := by
  sorry

#check amelia_wins_probability

end NUMINAMATH_CALUDE_amelia_wins_probability_l2820_282027
