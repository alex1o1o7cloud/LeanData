import Mathlib

namespace NUMINAMATH_CALUDE_coaches_average_age_l860_86035

theorem coaches_average_age 
  (total_members : ℕ) 
  (overall_average : ℕ) 
  (num_girls : ℕ) 
  (num_boys : ℕ) 
  (num_coaches : ℕ) 
  (girls_average : ℕ) 
  (boys_average : ℕ) 
  (h1 : total_members = 50)
  (h2 : overall_average = 18)
  (h3 : num_girls = 25)
  (h4 : num_boys = 20)
  (h5 : num_coaches = 5)
  (h6 : girls_average = 16)
  (h7 : boys_average = 17)
  (h8 : total_members = num_girls + num_boys + num_coaches) :
  (total_members * overall_average - num_girls * girls_average - num_boys * boys_average) / num_coaches = 32 := by
  sorry

end NUMINAMATH_CALUDE_coaches_average_age_l860_86035


namespace NUMINAMATH_CALUDE_intersection_point_parallel_through_point_perpendicular_with_y_intercept_l860_86018

-- Define the lines l₁ and l₂
def l₁ (m n x y : ℝ) : Prop := m * x + 8 * y + n = 0
def l₂ (m x y : ℝ) : Prop := 2 * x + m * y - 1 = 0

-- Scenario 1: l₁ and l₂ intersect at point P(m, 1)
theorem intersection_point (m n : ℝ) : 
  (l₁ m n m 1 ∧ l₂ m m 1) → (m = 1/3 ∧ n = -73/9) := by sorry

-- Scenario 2: l₁ is parallel to l₂ and passes through (3, -1)
theorem parallel_through_point (m n : ℝ) :
  (∀ x y : ℝ, l₁ m n x y ↔ l₂ m x y) ∧ l₁ m n 3 (-1) → 
  ((m = 4 ∧ n = -4) ∨ (m = -4 ∧ n = 20)) := by sorry

-- Scenario 3: l₁ is perpendicular to l₂ and y-intercept of l₁ is -1
theorem perpendicular_with_y_intercept (m n : ℝ) :
  (∀ x y : ℝ, l₁ m n x y → l₂ m x y → m * m = -1) ∧ l₁ m n 0 (-1) →
  (m = 0 ∧ n = 8) := by sorry

end NUMINAMATH_CALUDE_intersection_point_parallel_through_point_perpendicular_with_y_intercept_l860_86018


namespace NUMINAMATH_CALUDE_additional_plates_l860_86051

/-- The number of choices for each letter position in the original license plate system -/
def original_choices : Fin 3 → Nat
  | 0 => 5  -- First position
  | 1 => 3  -- Second position
  | 2 => 4  -- Third position

/-- The total number of possible license plates in the original system -/
def original_total : Nat := (original_choices 0) * (original_choices 1) * (original_choices 2)

/-- The number of choices for each letter position after adding one letter to each set -/
def new_choices : Fin 3 → Nat
  | i => (original_choices i) + 1

/-- The total number of possible license plates in the new system -/
def new_total : Nat := (new_choices 0) * (new_choices 1) * (new_choices 2)

/-- The theorem stating the number of additional license plates -/
theorem additional_plates : new_total - original_total = 60 := by
  sorry

end NUMINAMATH_CALUDE_additional_plates_l860_86051


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l860_86000

theorem solution_set_of_inequality (x : ℝ) :
  (x - 50) * (60 - x) > 0 ↔ x ∈ Set.Ioo 50 60 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l860_86000


namespace NUMINAMATH_CALUDE_bill_difference_l860_86002

/-- The number of $20 bills Mandy has -/
def mandy_twenty_bills : ℕ := 3

/-- The number of $50 bills Manny has -/
def manny_fifty_bills : ℕ := 2

/-- The value of a $20 bill -/
def twenty_bill_value : ℕ := 20

/-- The value of a $50 bill -/
def fifty_bill_value : ℕ := 50

/-- The value of a $10 bill -/
def ten_bill_value : ℕ := 10

/-- Theorem stating the difference in $10 bills between Manny and Mandy -/
theorem bill_difference :
  (manny_fifty_bills * fifty_bill_value) / ten_bill_value -
  (mandy_twenty_bills * twenty_bill_value) / ten_bill_value = 4 := by
  sorry

end NUMINAMATH_CALUDE_bill_difference_l860_86002


namespace NUMINAMATH_CALUDE_circle_intersection_symmetry_l860_86028

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (c1 c2 : Circle) (A B : ℝ × ℝ) : Prop :=
  -- The circles intersect at points A and B
  A ∈ {p : ℝ × ℝ | (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2} ∧
  A ∈ {p : ℝ × ℝ | (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2} ∧
  B ∈ {p : ℝ × ℝ | (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2} ∧
  B ∈ {p : ℝ × ℝ | (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2} ∧
  -- Centers of both circles are on the x-axis
  c1.center.2 = 0 ∧
  c2.center.2 = 0 ∧
  -- Coordinates of point A are (-3, 2)
  A = (-3, 2)

-- Theorem statement
theorem circle_intersection_symmetry (c1 c2 : Circle) (A B : ℝ × ℝ) :
  problem_setup c1 c2 A B → B = (-3, -2) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_symmetry_l860_86028


namespace NUMINAMATH_CALUDE_last_three_digits_of_5_pow_1999_l860_86091

def last_three_digits (n : ℕ) : ℕ := n % 1000

theorem last_three_digits_of_5_pow_1999 :
  last_three_digits (5^1999) = 125 := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_5_pow_1999_l860_86091


namespace NUMINAMATH_CALUDE_card_movement_strategy_exists_no_guaranteed_ace_strategy_l860_86064

/-- Represents a deck of cards arranged in a circle with one free spot -/
structure CircularDeck :=
  (cards : Fin 52 → Option (Fin 52))
  (free_spot : Fin 53)
  (initial_positions : Fin 52 → Fin 53)

/-- Represents a strategy for naming cards -/
def Strategy := ℕ → Fin 52

/-- Checks if a card is next to the free spot -/
def is_next_to_free_spot (deck : CircularDeck) (card : Fin 52) : Prop :=
  sorry

/-- Moves a card to the free spot if it's adjacent -/
def move_card (deck : CircularDeck) (card : Fin 52) : CircularDeck :=
  sorry

/-- Applies a strategy to a deck for a given number of steps -/
def apply_strategy (deck : CircularDeck) (strategy : Strategy) (steps : ℕ) : CircularDeck :=
  sorry

/-- Checks if all cards are not in their initial positions -/
def all_cards_moved (deck : CircularDeck) : Prop :=
  sorry

/-- Checks if the ace of spades is not next to the free spot -/
def ace_not_next_to_free (deck : CircularDeck) : Prop :=
  sorry

theorem card_movement_strategy_exists :
  ∃ (strategy : Strategy), ∀ (initial_deck : CircularDeck),
  ∃ (steps : ℕ), all_cards_moved (apply_strategy initial_deck strategy steps) :=
sorry

theorem no_guaranteed_ace_strategy :
  ¬∃ (strategy : Strategy), ∀ (initial_deck : CircularDeck),
  ∃ (steps : ℕ), ace_not_next_to_free (apply_strategy initial_deck strategy steps) :=
sorry

end NUMINAMATH_CALUDE_card_movement_strategy_exists_no_guaranteed_ace_strategy_l860_86064


namespace NUMINAMATH_CALUDE_factorization_of_a_squared_plus_5a_l860_86086

theorem factorization_of_a_squared_plus_5a (a : ℝ) : a^2 + 5*a = a*(a+5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_a_squared_plus_5a_l860_86086


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l860_86098

/-- Given points A, B, C, and O in a 2D plane, prove that the intersection point P
    of line segments AC and OB has coordinates (3, 3) -/
theorem intersection_point_coordinates :
  let A : Fin 2 → ℝ := ![4, 0]
  let B : Fin 2 → ℝ := ![4, 4]
  let C : Fin 2 → ℝ := ![2, 6]
  let O : Fin 2 → ℝ := ![0, 0]
  ∃ P : Fin 2 → ℝ,
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (fun i => t * (C i - A i) + A i)) ∧
    (∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ P = (fun i => s * (B i - O i) + O i)) ∧
    P = ![3, 3] :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_coordinates_l860_86098


namespace NUMINAMATH_CALUDE_consecutive_squares_sum_equality_l860_86032

theorem consecutive_squares_sum_equality :
  ∃ n : ℕ, n^2 + (n+1)^2 + (n+2)^2 = (n+3)^2 + (n+4)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_sum_equality_l860_86032


namespace NUMINAMATH_CALUDE_roots_polynomial_sum_l860_86057

theorem roots_polynomial_sum (p q : ℝ) : 
  p^2 - 6*p + 10 = 0 → q^2 - 6*q + 10 = 0 → p^4 + p^5*q^3 + p^3*q^5 + q^4 = 16056 :=
by sorry

end NUMINAMATH_CALUDE_roots_polynomial_sum_l860_86057


namespace NUMINAMATH_CALUDE_subtract_twice_l860_86065

theorem subtract_twice (a : ℝ) : a - 2*a = -a := by sorry

end NUMINAMATH_CALUDE_subtract_twice_l860_86065


namespace NUMINAMATH_CALUDE_wrench_can_turn_bolt_l860_86038

/-- Represents a wrench with a regular hexagonal shape -/
structure Wrench where
  sideLength : ℝ
  sideLength_pos : 0 < sideLength

/-- Represents a bolt with a square head -/
structure Bolt where
  sideLength : ℝ
  sideLength_pos : 0 < sideLength

/-- Condition for a wrench to turn a bolt -/
def canTurn (w : Wrench) (b : Bolt) : Prop :=
  Real.sqrt 3 / Real.sqrt 2 < b.sideLength / w.sideLength ∧ 
  b.sideLength / w.sideLength ≤ 3 - Real.sqrt 3

/-- Theorem stating the condition for a wrench to turn a bolt -/
theorem wrench_can_turn_bolt (w : Wrench) (b : Bolt) : 
  canTurn w b ↔ 
    (∃ (x : ℝ), b.sideLength = x * w.sideLength ∧ 
      Real.sqrt 3 / Real.sqrt 2 < x ∧ x ≤ 3 - Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_wrench_can_turn_bolt_l860_86038


namespace NUMINAMATH_CALUDE_lucy_bank_balance_l860_86074

theorem lucy_bank_balance (initial_balance deposit withdrawal : ℕ) :
  initial_balance = 65 →
  deposit = 15 →
  withdrawal = 4 →
  initial_balance + deposit - withdrawal = 76 := by
sorry

end NUMINAMATH_CALUDE_lucy_bank_balance_l860_86074


namespace NUMINAMATH_CALUDE_five_b_value_l860_86087

theorem five_b_value (a b : ℚ) (h1 : 3 * a + 4 * b = 0) (h2 : a = b - 3) : 5 * b = 45 / 7 := by
  sorry

end NUMINAMATH_CALUDE_five_b_value_l860_86087


namespace NUMINAMATH_CALUDE_complex_equation_imag_part_l860_86010

theorem complex_equation_imag_part :
  ∀ z : ℂ, z * (1 + Complex.I) = (3 : ℂ) + 2 * Complex.I →
  Complex.im z = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_imag_part_l860_86010


namespace NUMINAMATH_CALUDE_library_digital_format_l860_86014

-- Define the universe of books in the library
variable (Book : Type)

-- Define a predicate for a book being available in digital format
variable (isDigital : Book → Prop)

-- Define the theorem
theorem library_digital_format (h : ¬∀ (b : Book), isDigital b) :
  (∃ (b : Book), ¬isDigital b) ∧ (¬∀ (b : Book), isDigital b) := by
  sorry

end NUMINAMATH_CALUDE_library_digital_format_l860_86014


namespace NUMINAMATH_CALUDE_tan_nine_pi_fourth_l860_86039

theorem tan_nine_pi_fourth : Real.tan (9 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_nine_pi_fourth_l860_86039


namespace NUMINAMATH_CALUDE_consecutive_numbers_divisibility_l860_86041

theorem consecutive_numbers_divisibility (k : ℕ) :
  let r₁ := k % 2022
  let r₂ := (k + 1) % 2022
  let r₃ := (k + 2) % 2022
  Prime (r₁ + r₂ + r₃) →
  (k % 2022 = 0) ∨ ((k + 1) % 2022 = 0) ∨ ((k + 2) % 2022 = 0) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_divisibility_l860_86041


namespace NUMINAMATH_CALUDE_min_value_when_a_is_neg_three_a_range_when_inequality_holds_l860_86075

-- Define the function f
def f (a x : ℝ) : ℝ := |x - 1| + |x - a|

-- Theorem for part (1)
theorem min_value_when_a_is_neg_three :
  ∃ (min : ℝ), min = 4 ∧ ∀ x, f (-3) x ≥ min :=
sorry

-- Theorem for part (2)
theorem a_range_when_inequality_holds :
  (∀ x, f a x ≤ 2*a + 2*|x - 1|) → a ≥ 1/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_neg_three_a_range_when_inequality_holds_l860_86075


namespace NUMINAMATH_CALUDE_integer_average_l860_86054

theorem integer_average (k m r s t : ℕ) : 
  0 < k ∧ k < m ∧ m < r ∧ r < s ∧ s < t ∧ 
  t = 40 ∧ 
  r ≤ 23 ∧ 
  ∀ (k' m' r' s' t' : ℕ), 
    (0 < k' ∧ k' < m' ∧ m' < r' ∧ r' < s' ∧ s' < t' ∧ t' = 40) → r' ≤ r →
  (k + m + r + s + t) / 5 = 18 := by
sorry

end NUMINAMATH_CALUDE_integer_average_l860_86054


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l860_86033

theorem camping_trip_percentage 
  (total_students : ℕ) 
  (students_more_than_100 : ℕ) 
  (students_not_more_than_100 : ℕ) 
  (h1 : students_more_than_100 = (18 * total_students) / 100)
  (h2 : students_not_more_than_100 = (75 * (students_more_than_100 + students_not_more_than_100)) / 100) :
  (students_more_than_100 + students_not_more_than_100) * 100 / total_students = 72 :=
by sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l860_86033


namespace NUMINAMATH_CALUDE_appropriate_presentation_lengths_l860_86017

/-- Represents the duration of a presentation in minutes -/
def PresentationDuration : Set ℝ := { x | 20 ≤ x ∧ x ≤ 40 }

/-- The ideal speech rate in words per minute -/
def SpeechRate : ℝ := 120

/-- Calculates the range of appropriate word counts for a presentation -/
def AppropriateWordCount : Set ℕ :=
  { w | ∃ (d : ℝ), d ∈ PresentationDuration ∧ 
    (↑w : ℝ) ≥ 20 * SpeechRate ∧ (↑w : ℝ) ≤ 40 * SpeechRate }

/-- Theorem stating that 2700, 3900, and 4500 words are appropriate presentation lengths -/
theorem appropriate_presentation_lengths :
  2700 ∈ AppropriateWordCount ∧
  3900 ∈ AppropriateWordCount ∧
  4500 ∈ AppropriateWordCount :=
by sorry

end NUMINAMATH_CALUDE_appropriate_presentation_lengths_l860_86017


namespace NUMINAMATH_CALUDE_polynomial_roots_arithmetic_progression_l860_86034

/-- If a polynomial x^4 + jx^2 + kx + 256 has four distinct real roots in arithmetic progression, then j = -80 -/
theorem polynomial_roots_arithmetic_progression (j k : ℝ) : 
  (∃ (a b c d : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ 
    (∀ (x : ℝ), x^4 + j*x^2 + k*x + 256 = (x - a) * (x - b) * (x - c) * (x - d)) ∧
    (b - a = c - b) ∧ (c - b = d - c)) →
  j = -80 := by sorry

end NUMINAMATH_CALUDE_polynomial_roots_arithmetic_progression_l860_86034


namespace NUMINAMATH_CALUDE_additional_cost_proof_l860_86036

/-- Additional cost per international letter --/
def additional_cost_per_letter : ℚ := 55 / 100

/-- Number of letters --/
def num_letters : ℕ := 4

/-- Number of domestic letters --/
def num_domestic : ℕ := 2

/-- Number of international letters --/
def num_international : ℕ := 2

/-- Domestic postage rate per letter --/
def domestic_rate : ℚ := 108 / 100

/-- Weight of first international letter (in grams) --/
def weight_letter1 : ℕ := 25

/-- Weight of second international letter (in grams) --/
def weight_letter2 : ℕ := 45

/-- Rate for Country A for letters below 50 grams (per gram) --/
def rate_A_below50 : ℚ := 5 / 100

/-- Rate for Country B for letters below 50 grams (per gram) --/
def rate_B_below50 : ℚ := 4 / 100

/-- Total postage paid --/
def total_paid : ℚ := 630 / 100

theorem additional_cost_proof :
  let domestic_cost := num_domestic * domestic_rate
  let international_cost1 := weight_letter1 * rate_A_below50
  let international_cost2 := weight_letter2 * rate_B_below50
  let total_calculated := domestic_cost + international_cost1 + international_cost2
  let additional_total := total_paid - total_calculated
  additional_total / num_international = additional_cost_per_letter := by
  sorry

end NUMINAMATH_CALUDE_additional_cost_proof_l860_86036


namespace NUMINAMATH_CALUDE_base_seven_digits_of_1200_l860_86008

theorem base_seven_digits_of_1200 : ∃ n : ℕ, (7^(n-1) ≤ 1200 ∧ 1200 < 7^n) ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_digits_of_1200_l860_86008


namespace NUMINAMATH_CALUDE_gum_to_candy_ratio_l860_86022

/-- The cost of a candy bar in dollars -/
def candy_cost : ℚ := 3/2

/-- The total cost of the purchase in dollars -/
def total_cost : ℚ := 6

/-- The number of packs of gum purchased -/
def gum_packs : ℕ := 2

/-- The number of candy bars purchased -/
def candy_bars : ℕ := 3

theorem gum_to_candy_ratio :
  ∃ (gum_cost : ℚ), 
    gum_cost * gum_packs + candy_cost * candy_bars = total_cost ∧
    gum_cost / candy_cost = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_gum_to_candy_ratio_l860_86022


namespace NUMINAMATH_CALUDE_constant_term_expansion_l860_86068

theorem constant_term_expansion (x : ℝ) : 
  (∃ c : ℝ, c ≠ 0 ∧ ∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, 
    0 < |y - x| ∧ |y - x| < δ → |(1/y - y^3)^4 - c| < ε) → 
  c = -4 :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l860_86068


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l860_86090

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 4) * (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 8 / Real.sqrt 9) = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l860_86090


namespace NUMINAMATH_CALUDE_original_price_calculation_l860_86013

/-- Given an article sold for $35 with a 75% gain, prove that the original price was $20. -/
theorem original_price_calculation (sale_price : ℝ) (gain_percent : ℝ) 
  (h1 : sale_price = 35)
  (h2 : gain_percent = 75) :
  ∃ (original_price : ℝ), 
    sale_price = original_price * (1 + gain_percent / 100) ∧ 
    original_price = 20 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l860_86013


namespace NUMINAMATH_CALUDE_toy_distribution_l860_86070

/-- Given a number of pens and toys distributed among students, 
    where each student receives the same number of pens and toys, 
    prove that the number of toys is a multiple of the number of students. -/
theorem toy_distribution (num_pens : ℕ) (num_toys : ℕ) (num_students : ℕ) 
  (h1 : num_pens = 451)
  (h2 : num_students = 41)
  (h3 : num_pens % num_students = 0)
  (h4 : num_toys % num_students = 0) :
  ∃ k : ℕ, num_toys = num_students * k :=
sorry

end NUMINAMATH_CALUDE_toy_distribution_l860_86070


namespace NUMINAMATH_CALUDE_derivative_f_l860_86099

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.cos (1/3)) + (Real.sin (23*x))^2 / (23 * Real.cos (46*x))

-- State the theorem
theorem derivative_f :
  ∀ x : ℝ, deriv f x = Real.tan (46*x) / Real.cos (46*x) :=
by sorry

end NUMINAMATH_CALUDE_derivative_f_l860_86099


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_relation_l860_86078

-- Define the necessary structures
structure Line3D where
  -- Add necessary fields for a 3D line

structure Plane3D where
  -- Add necessary fields for a 3D plane

-- Define the relationships
def parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

def within (l : Line3D) (p : Plane3D) : Prop :=
  sorry

def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

def skew_lines (l1 l2 : Line3D) : Prop :=
  sorry

-- Theorem statement
theorem line_parallel_to_plane_relation (m n : Line3D) (α : Plane3D) 
    (h1 : parallel m α) (h2 : within n α) :
  parallel_lines m n ∨ skew_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_relation_l860_86078


namespace NUMINAMATH_CALUDE_manufacturing_plant_optimization_l860_86081

noncomputable def f (x : ℝ) : ℝ := 4 * (1 - x) * x^2

def domain (t : ℝ) (x : ℝ) : Prop := 0 < x ∧ x ≤ 2*t/(2*t+1)

theorem manufacturing_plant_optimization (t : ℝ) 
  (h1 : 0 < t) (h2 : t ≤ 2) :
  (f 0.5 = 0.5) ∧
  (∀ x, domain t x →
    (1 ≤ t → f x ≤ 16/27 ∧ (f x = 16/27 → x = 2/3)) ∧
    (t < 1 → f x ≤ 16*t^2/(2*t+1)^3 ∧ (f x = 16*t^2/(2*t+1)^3 → x = 2*t/(2*t+1)))) :=
by sorry

end NUMINAMATH_CALUDE_manufacturing_plant_optimization_l860_86081


namespace NUMINAMATH_CALUDE_least_marbles_ten_marbles_john_marbles_l860_86079

theorem least_marbles (m : ℕ) : m > 0 ∧ m % 7 = 3 ∧ m % 4 = 2 → m ≥ 10 := by
  sorry

theorem ten_marbles : 10 % 7 = 3 ∧ 10 % 4 = 2 := by
  sorry

theorem john_marbles : ∃ m : ℕ, m > 0 ∧ m % 7 = 3 ∧ m % 4 = 2 ∧ ∀ n : ℕ, (n > 0 ∧ n % 7 = 3 ∧ n % 4 = 2) → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_least_marbles_ten_marbles_john_marbles_l860_86079


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l860_86084

theorem complex_fraction_equals_i :
  let i : ℂ := Complex.I
  (1 + i) / (1 - i) = i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l860_86084


namespace NUMINAMATH_CALUDE_age_difference_l860_86023

theorem age_difference (man_age son_age : ℕ) : 
  man_age > son_age →
  son_age = 25 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 27 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l860_86023


namespace NUMINAMATH_CALUDE_max_sections_five_l860_86089

/-- The maximum number of sections created by n line segments in a rectangle -/
def max_sections (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => max_sections m + m + 1

/-- Theorem: The maximum number of sections created by 5 line segments in a rectangle is 16 -/
theorem max_sections_five : max_sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_sections_five_l860_86089


namespace NUMINAMATH_CALUDE_combined_age_of_siblings_l860_86063

-- Define the ages of the siblings
def aaron_age : ℕ := 15
def sister_age : ℕ := 3 * aaron_age
def henry_age : ℕ := 4 * sister_age
def alice_age : ℕ := aaron_age - 2

-- Theorem to prove
theorem combined_age_of_siblings : aaron_age + sister_age + henry_age + alice_age = 253 := by
  sorry

end NUMINAMATH_CALUDE_combined_age_of_siblings_l860_86063


namespace NUMINAMATH_CALUDE_tension_force_in_rod_system_l860_86095

/-- The tension force in a weightless rod system with a suspended weight. -/
theorem tension_force_in_rod_system (m g : ℝ) (T₀ T₁ T₂ : ℝ) : 
  m = 2 →
  g = 10 →
  T₂ = 1/4 * m * g →
  T₁ = 3/4 * m * g →
  T₀ * (1/4) + T₂ = T₁ * (1/2) →
  T₀ = 10 := by sorry

end NUMINAMATH_CALUDE_tension_force_in_rod_system_l860_86095


namespace NUMINAMATH_CALUDE_bruce_purchase_l860_86072

/-- Calculates the total amount Bruce paid for grapes and mangoes -/
def totalAmountPaid (grapeQuantity : ℕ) (grapeRate : ℕ) (mangoQuantity : ℕ) (mangoRate : ℕ) : ℕ :=
  grapeQuantity * grapeRate + mangoQuantity * mangoRate

/-- Proves that Bruce paid 985 for his purchase of grapes and mangoes -/
theorem bruce_purchase : totalAmountPaid 7 70 9 55 = 985 := by
  sorry

end NUMINAMATH_CALUDE_bruce_purchase_l860_86072


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l860_86062

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  a_1_eq : a 1 = 4
  a_7_sq_eq : (a 7) ^ 2 = (a 1) * (a 10)
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of the first n terms of the sequence -/
def S_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = -1/3 * n + 13/3) ∧
  (∃ n : ℕ, S_n seq n = 26 ∧ (n = 12 ∨ n = 13) ∧ ∀ m : ℕ, S_n seq m ≤ 26) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l860_86062


namespace NUMINAMATH_CALUDE_integral_equals_zero_l860_86042

theorem integral_equals_zero : 
  ∫ x in (0 : ℝ)..1, (4 * Real.sqrt (1 - x) - Real.sqrt (3 * x + 1)) / 
    ((Real.sqrt (3 * x + 1) + 4 * Real.sqrt (1 - x)) * (3 * x + 1)^2) = 0 := by sorry

end NUMINAMATH_CALUDE_integral_equals_zero_l860_86042


namespace NUMINAMATH_CALUDE_inequality_proof_l860_86026

theorem inequality_proof (s x y z : ℝ) 
  (hs : s > 0) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (h : s * x > z * y) : 
  ¬ (
    (x > z ∧ -x > -z ∧ s > z / x ∧ s < y / x) ∨
    (x > z ∧ -x > -z ∧ s > z / x) ∨
    (x > z ∧ -x > -z ∧ s < y / x) ∨
    (x > z ∧ s > z / x ∧ s < y / x) ∨
    (-x > -z ∧ s > z / x ∧ s < y / x) ∨
    (x > z ∧ -x > -z) ∨
    (x > z ∧ s > z / x) ∨
    (x > z ∧ s < y / x) ∨
    (-x > -z ∧ s > z / x) ∨
    (-x > -z ∧ s < y / x) ∨
    (s > z / x ∧ s < y / x) ∨
    (x > z) ∨
    (-x > -z) ∨
    (s > z / x) ∨
    (s < y / x)
  ) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l860_86026


namespace NUMINAMATH_CALUDE_triangle_area_l860_86096

-- Define the three lines
def line1 (x : ℝ) : ℝ := 2 * x + 1
def line2 (x : ℝ) : ℝ := -x + 4
def line3 (x : ℝ) : ℝ := -1

-- Define the vertices of the triangle
def vertex1 : ℝ × ℝ := (1, 3)
def vertex2 : ℝ × ℝ := (-1, -1)
def vertex3 : ℝ × ℝ := (5, -1)

-- Theorem statement
theorem triangle_area : 
  let vertices := [vertex1, vertex2, vertex3]
  let xs := vertices.map Prod.fst
  let ys := vertices.map Prod.snd
  abs ((xs[0] * (ys[1] - ys[2]) + xs[1] * (ys[2] - ys[0]) + xs[2] * (ys[0] - ys[1])) / 2) = 12 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_l860_86096


namespace NUMINAMATH_CALUDE_sport_preference_related_to_gender_l860_86048

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![40, 20],
    ![20, 30]]

-- Define the calculated K^2 value
def calculated_k_squared : ℝ := 7.82

-- Define the critical values and their corresponding probabilities
def critical_values : List (ℝ × ℝ) :=
  [(2.706, 0.10), (3.841, 0.05), (6.635, 0.01), (7.879, 0.005), (10.828, 0.001)]

-- Define the confidence level we want to prove
def target_confidence : ℝ := 0.99

-- Theorem statement
theorem sport_preference_related_to_gender :
  ∃ (lower_k upper_k : ℝ) (lower_p upper_p : ℝ),
    (lower_k, lower_p) ∈ critical_values ∧
    (upper_k, upper_p) ∈ critical_values ∧
    lower_k < calculated_k_squared ∧
    calculated_k_squared < upper_k ∧
    lower_p > 1 - target_confidence ∧
    upper_p < 1 - target_confidence :=
by sorry


end NUMINAMATH_CALUDE_sport_preference_related_to_gender_l860_86048


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l860_86021

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- State the theorem
theorem arithmetic_sequence_problem (a₁ d : ℝ) :
  d ≠ 0 →
  arithmetic_sequence a₁ d 5 = a₁^2 →
  a₁ * arithmetic_sequence a₁ d 21 = (arithmetic_sequence a₁ d 5)^2 →
  a₁ = 4 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l860_86021


namespace NUMINAMATH_CALUDE_lines_coincide_by_rotation_l860_86030

/-- Two lines that intersect can coincide by rotation -/
theorem lines_coincide_by_rotation (α c : ℝ) :
  ∃ (P : ℝ × ℝ), P.1 * Real.sin α = P.2 ∧ 
  ∃ (θ : ℝ), ∀ (x y : ℝ), 
    y = x * Real.sin α ↔ 
    (x - P.1) * Real.cos θ - (y - P.2) * Real.sin θ = 
    ((x - P.1) * Real.sin θ + (y - P.2) * Real.cos θ) * 2 + c :=
sorry

end NUMINAMATH_CALUDE_lines_coincide_by_rotation_l860_86030


namespace NUMINAMATH_CALUDE_jack_morning_emails_l860_86047

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := sorry

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 8

/-- The total number of emails Jack received in the morning and afternoon -/
def total_morning_afternoon : ℕ := 13

/-- Theorem stating that Jack received 5 emails in the morning -/
theorem jack_morning_emails : 
  morning_emails + afternoon_emails = total_morning_afternoon → 
  morning_emails = 5 := by sorry

end NUMINAMATH_CALUDE_jack_morning_emails_l860_86047


namespace NUMINAMATH_CALUDE_complex_power_six_l860_86082

theorem complex_power_six (i : ℂ) (h : i^2 = -1) : (1 + i)^6 = -8*i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_six_l860_86082


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l860_86077

theorem quadratic_roots_problem (a b m p r : ℝ) : 
  (∀ x, x^2 - m*x + 4 = 0 ↔ x = a ∨ x = b) →
  (∀ x, x^2 - p*x + r = 0 ↔ x = a + 2/b ∨ x = b + 2/a) →
  r = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l860_86077


namespace NUMINAMATH_CALUDE_sector_arc_length_and_area_l860_86073

/-- Given a sector with radius 2 and central angle π/6, prove that the arc length is π/3 and the area is π/3 -/
theorem sector_arc_length_and_area :
  let r : ℝ := 2
  let θ : ℝ := π / 6
  let arc_length : ℝ := r * θ
  let sector_area : ℝ := (1 / 2) * r * r * θ
  arc_length = π / 3 ∧ sector_area = π / 3 := by
sorry


end NUMINAMATH_CALUDE_sector_arc_length_and_area_l860_86073


namespace NUMINAMATH_CALUDE_theta_value_l860_86059

theorem theta_value (θ : Real) (h1 : 1 / Real.sin θ + 1 / Real.cos θ = 35 / 12) 
  (h2 : θ ∈ Set.Ioo 0 (Real.pi / 2)) :
  θ = Real.arcsin (3 / 5) ∨ θ = Real.arcsin (4 / 5) := by
  sorry

end NUMINAMATH_CALUDE_theta_value_l860_86059


namespace NUMINAMATH_CALUDE_jeff_average_skips_l860_86025

-- Define the number of rounds
def num_rounds : ℕ := 4

-- Define Sam's skips per round
def sam_skips : ℕ := 16

-- Define Jeff's skips for each round
def jeff_round1 : ℕ := sam_skips - 1
def jeff_round2 : ℕ := sam_skips - 3
def jeff_round3 : ℕ := sam_skips + 4
def jeff_round4 : ℕ := sam_skips / 2

-- Define Jeff's total skips
def jeff_total : ℕ := jeff_round1 + jeff_round2 + jeff_round3 + jeff_round4

-- Theorem to prove
theorem jeff_average_skips :
  jeff_total / num_rounds = 14 := by sorry

end NUMINAMATH_CALUDE_jeff_average_skips_l860_86025


namespace NUMINAMATH_CALUDE_pet_shop_grooming_l860_86040

/-- The pet shop grooming problem -/
theorem pet_shop_grooming (poodle_time terrier_time total_time : ℕ) 
  (terrier_count : ℕ) (poodle_count : ℕ) : 
  poodle_time = 30 →
  terrier_time = poodle_time / 2 →
  terrier_count = 8 →
  total_time = 210 →
  poodle_count * poodle_time + terrier_count * terrier_time = total_time →
  poodle_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_grooming_l860_86040


namespace NUMINAMATH_CALUDE_martha_cakes_l860_86019

/-- The number of cakes Martha needs to buy -/
def total_cakes (num_children : ℝ) (cakes_per_child : ℝ) : ℝ :=
  num_children * cakes_per_child

/-- Theorem: Martha needs to buy 54 cakes -/
theorem martha_cakes : total_cakes 3 18 = 54 := by
  sorry

end NUMINAMATH_CALUDE_martha_cakes_l860_86019


namespace NUMINAMATH_CALUDE_rectangle_length_l860_86067

/-- Given a rectangle with perimeter 30 cm and width 10 cm, prove its length is 5 cm -/
theorem rectangle_length (perimeter width : ℝ) (h1 : perimeter = 30) (h2 : width = 10) :
  2 * (width + (perimeter / 2 - width)) = perimeter → perimeter / 2 - width = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l860_86067


namespace NUMINAMATH_CALUDE_salary_change_percentage_l860_86029

theorem salary_change_percentage (initial_salary : ℝ) (h : initial_salary > 0) :
  let decreased_salary := initial_salary * (1 - 0.6)
  let final_salary := decreased_salary * (1 + 0.6)
  final_salary = initial_salary * 0.64 ∧ 
  (initial_salary - final_salary) / initial_salary = 0.36 :=
by sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l860_86029


namespace NUMINAMATH_CALUDE_f_nonnegative_iff_a_eq_one_f_greater_than_x_ln_x_minus_sin_x_l860_86061

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - x - a

-- Theorem 1: f(x) ≥ 0 if and only if a = 1
theorem f_nonnegative_iff_a_eq_one :
  (∀ x, f a x ≥ 0) ↔ a = 1 :=
sorry

-- Theorem 2: For a ≥ 1, f(x) > x ln x - sin x for all x > 0
theorem f_greater_than_x_ln_x_minus_sin_x
  (a : ℝ) (h : a ≥ 1) :
  ∀ x > 0, f a x > x * Real.log x - Real.sin x :=
sorry

end NUMINAMATH_CALUDE_f_nonnegative_iff_a_eq_one_f_greater_than_x_ln_x_minus_sin_x_l860_86061


namespace NUMINAMATH_CALUDE_order_of_abc_l860_86080

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem order_of_abc : a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l860_86080


namespace NUMINAMATH_CALUDE_total_nails_needed_l860_86009

def nails_per_plank : ℕ := 2
def number_of_planks : ℕ := 16

theorem total_nails_needed : nails_per_plank * number_of_planks = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_nails_needed_l860_86009


namespace NUMINAMATH_CALUDE_lucille_paint_cans_l860_86055

/-- Represents the dimensions of a wall -/
structure Wall where
  width : ℝ
  height : ℝ

/-- Calculates the area of a wall -/
def wallArea (w : Wall) : ℝ := w.width * w.height

/-- Represents the room to be painted -/
structure Room where
  wall1 : Wall
  wall2 : Wall
  wall3 : Wall
  wall4 : Wall

/-- Calculates the total area of all walls in the room -/
def totalArea (r : Room) : ℝ :=
  wallArea r.wall1 + wallArea r.wall2 + wallArea r.wall3 + wallArea r.wall4

/-- The coverage area of one can of paint -/
def paintCoverage : ℝ := 2

/-- Lucille's room configuration -/
def lucilleRoom : Room :=
  { wall1 := { width := 3, height := 2 }
  , wall2 := { width := 3, height := 2 }
  , wall3 := { width := 5, height := 2 }
  , wall4 := { width := 4, height := 2 } }

/-- Theorem: Lucille needs 15 cans of paint -/
theorem lucille_paint_cans : 
  ⌈(totalArea lucilleRoom) / paintCoverage⌉ = 15 := by sorry

end NUMINAMATH_CALUDE_lucille_paint_cans_l860_86055


namespace NUMINAMATH_CALUDE_income_calculation_l860_86027

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 8 = expenditure * 9 →  -- income and expenditure ratio is 9:8
  income = expenditure + savings → -- income equals expenditure plus savings
  savings = 4000 → -- savings are 4000
  income = 36000 := by -- prove that income is 36000
sorry

end NUMINAMATH_CALUDE_income_calculation_l860_86027


namespace NUMINAMATH_CALUDE_zeros_and_range_of_f_l860_86015

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + b - 1

theorem zeros_and_range_of_f (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, f 1 (-2) x = 0 ↔ x = 3 ∨ x = -1) ∧
  (∀ b : ℝ, (∃ x y : ℝ, x ≠ y ∧ f a b x = 0 ∧ f a b y = 0) ↔ 0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_zeros_and_range_of_f_l860_86015


namespace NUMINAMATH_CALUDE_expected_twos_is_one_third_l860_86012

/-- Represents a standard six-sided die -/
def StandardDie := Fin 6

/-- The probability of rolling a 2 on a standard die -/
def prob_two : ℚ := 1 / 6

/-- The probability of not rolling a 2 on a standard die -/
def prob_not_two : ℚ := 5 / 6

/-- The expected number of 2's when rolling two standard dice -/
def expected_twos : ℚ := 1 / 3

/-- Theorem: The expected number of 2's when rolling two standard dice is 1/3 -/
theorem expected_twos_is_one_third :
  expected_twos = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expected_twos_is_one_third_l860_86012


namespace NUMINAMATH_CALUDE_mark_spending_l860_86037

/-- Represents the grocery items Mark buys -/
inductive GroceryItem
  | Apple
  | Bread
  | Cheese
  | Cereal

/-- Represents Mark's grocery shopping trip -/
structure GroceryShopping where
  prices : GroceryItem → ℕ
  quantities : GroceryItem → ℕ
  appleBuyOneGetOneFree : Bool
  couponValue : ℕ
  couponThreshold : ℕ

def calculateTotalSpending (shopping : GroceryShopping) : ℕ :=
  sorry

theorem mark_spending (shopping : GroceryShopping) 
  (h1 : shopping.prices GroceryItem.Apple = 2)
  (h2 : shopping.prices GroceryItem.Bread = 3)
  (h3 : shopping.prices GroceryItem.Cheese = 6)
  (h4 : shopping.prices GroceryItem.Cereal = 5)
  (h5 : shopping.quantities GroceryItem.Apple = 4)
  (h6 : shopping.quantities GroceryItem.Bread = 5)
  (h7 : shopping.quantities GroceryItem.Cheese = 3)
  (h8 : shopping.quantities GroceryItem.Cereal = 4)
  (h9 : shopping.appleBuyOneGetOneFree = true)
  (h10 : shopping.couponValue = 10)
  (h11 : shopping.couponThreshold = 50)
  : calculateTotalSpending shopping = 47 := by
  sorry

end NUMINAMATH_CALUDE_mark_spending_l860_86037


namespace NUMINAMATH_CALUDE_zoo_revenue_example_l860_86083

/-- Calculates the total money made by a zoo over two days given the number of children and adults each day and the ticket prices. -/
def zoo_revenue (child_price adult_price : ℕ) (mon_children mon_adults tues_children tues_adults : ℕ) : ℕ :=
  (mon_children * child_price + mon_adults * adult_price) +
  (tues_children * child_price + tues_adults * adult_price)

/-- Theorem stating that the zoo made $61 in total for both days. -/
theorem zoo_revenue_example : zoo_revenue 3 4 7 5 4 2 = 61 := by
  sorry

end NUMINAMATH_CALUDE_zoo_revenue_example_l860_86083


namespace NUMINAMATH_CALUDE_equation_one_solutions_l860_86003

theorem equation_one_solutions :
  let f : ℝ → ℝ := λ x => x^2 - 2*x - 1
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l860_86003


namespace NUMINAMATH_CALUDE_cylinder_volume_increase_l860_86053

def R : ℝ := 10
def H : ℝ := 5

theorem cylinder_volume_increase (x : ℝ) : 
  π * (R + 2*x)^2 * H = π * R^2 * (H + 3*x) → x = 5 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_increase_l860_86053


namespace NUMINAMATH_CALUDE_five_fourths_of_eight_thirds_l860_86044

theorem five_fourths_of_eight_thirds (x : ℚ) : x = 8/3 → (5/4) * x = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_five_fourths_of_eight_thirds_l860_86044


namespace NUMINAMATH_CALUDE_function_properties_l860_86049

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * f y

theorem function_properties (f : ℝ → ℝ) 
  (h1 : functional_equation f) 
  (h2 : f (1/2) = 0) 
  (h3 : f 0 ≠ 0) : 
  (f 0 = 1) ∧ (∀ x : ℝ, f (1/2 + x) = -f (1/2 - x)) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l860_86049


namespace NUMINAMATH_CALUDE_max_vovochka_candies_l860_86058

/-- Represents the distribution of candies to classmates -/
def CandyDistribution := Fin 25 → ℕ

/-- The total number of candies -/
def totalCandies : ℕ := 200

/-- Checks if a candy distribution satisfies the condition that any 16 classmates have at least 100 candies -/
def isValidDistribution (d : CandyDistribution) : Prop :=
  ∀ (s : Finset (Fin 25)), s.card = 16 → (s.sum d) ≥ 100

/-- Calculates the number of candies Vovochka keeps for himself given a distribution -/
def vovochkaCandies (d : CandyDistribution) : ℕ :=
  totalCandies - (Finset.univ.sum d)

/-- Theorem stating that the maximum number of candies Vovochka can keep is 37 -/
theorem max_vovochka_candies :
  (∃ (d : CandyDistribution), isValidDistribution d ∧ vovochkaCandies d = 37) ∧
  (∀ (d : CandyDistribution), isValidDistribution d → vovochkaCandies d ≤ 37) :=
sorry

end NUMINAMATH_CALUDE_max_vovochka_candies_l860_86058


namespace NUMINAMATH_CALUDE_common_solution_condition_l860_86056

theorem common_solution_condition (a b : ℝ) : 
  (∃ x y : ℝ, 19 * x^2 + 19 * y^2 + a * x + b * y + 98 = 0 ∧ 
               98 * x^2 + 98 * y^2 + a * x + b * y + 19 = 0) ↔ 
  a^2 + b^2 ≥ 13689 := by
sorry

end NUMINAMATH_CALUDE_common_solution_condition_l860_86056


namespace NUMINAMATH_CALUDE_age_problem_l860_86076

theorem age_problem (a b c : ℕ) : 
  (4 * a + b = 3 * c) →
  (3 * c^3 = 4 * a^3 + b^3) →
  (Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1) →
  (a^2 + b^2 + c^2 = 35) :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l860_86076


namespace NUMINAMATH_CALUDE_distance_from_origin_l860_86060

/-- Given a point (x,y) satisfying certain conditions, prove that its distance from the origin is √(286 + 2√221) -/
theorem distance_from_origin (x y : ℝ) (h1 : y = 8) (h2 : x > 1) 
  (h3 : Real.sqrt ((x - 1)^2 + 2^2) = 15) : 
  Real.sqrt (x^2 + y^2) = Real.sqrt (286 + 2 * Real.sqrt 221) := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_l860_86060


namespace NUMINAMATH_CALUDE_position_of_81st_number_l860_86016

/-- Represents the triangular number pattern where each row has one more number than the previous row. -/
def TriangularPattern : Nat → Nat → Nat
  | row, pos => if pos ≤ row then (row * (row - 1)) / 2 + pos else 0

/-- The position of a number in the triangular pattern. -/
structure Position where
  row : Nat
  pos : Nat

/-- Finds the position of the nth number in the triangular pattern. -/
def findPosition (n : Nat) : Position :=
  let row := (Nat.sqrt (8 * n + 1) - 1) / 2 + 1
  let pos := n - (row * (row - 1)) / 2
  ⟨row, pos⟩

theorem position_of_81st_number :
  findPosition 81 = ⟨13, 3⟩ := by sorry

end NUMINAMATH_CALUDE_position_of_81st_number_l860_86016


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l860_86093

-- Define the function g(x)
def g (x : ℝ) : ℝ := 10 * x^4 - 17 * x^2 + 3

-- State the theorem
theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt 15 / 5 ∧
  (∀ x : ℝ, g x = 0 → x ≤ r) ∧
  g r = 0 := by
  sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l860_86093


namespace NUMINAMATH_CALUDE_max_omega_for_monotonic_sine_l860_86020

theorem max_omega_for_monotonic_sine (A ω : ℝ) (h_A : A > 0) (h_ω : ω > 0) :
  (∀ x ∈ Set.Icc (-3 * π / 4) (-π / 6),
    ∀ y ∈ Set.Icc (-3 * π / 4) (-π / 6),
    x < y → A * Real.sin (x + ω * π / 2) < A * Real.sin (y + ω * π / 2)) →
  ω ≤ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_omega_for_monotonic_sine_l860_86020


namespace NUMINAMATH_CALUDE_unique_intersection_l860_86011

-- Define the complex plane
variable (z : ℂ)

-- Define the equations
def equation1 (z : ℂ) : Prop := Complex.abs (z - 5) = 3 * Complex.abs (z + 5)
def equation2 (z : ℂ) (k : ℝ) : Prop := Complex.abs z = k

-- Define the intersection condition
def intersectsOnce (k : ℝ) : Prop :=
  ∃! z, equation1 z ∧ equation2 z k

-- Theorem statement
theorem unique_intersection :
  ∃! k, intersectsOnce k ∧ k = 12.5 := by sorry

end NUMINAMATH_CALUDE_unique_intersection_l860_86011


namespace NUMINAMATH_CALUDE_tan_22_5_decomposition_l860_86088

theorem tan_22_5_decomposition :
  ∃ (a b c d : ℕ+), 
    (Real.tan (22.5 * π / 180) = (a : ℝ).sqrt - (b : ℝ).sqrt + (c : ℝ).sqrt - (d : ℝ)) ∧
    a ≥ b ∧ b ≥ c ∧ c ≥ d ∧
    a + b + c + d = 3 := by sorry

end NUMINAMATH_CALUDE_tan_22_5_decomposition_l860_86088


namespace NUMINAMATH_CALUDE_circle_segment_area_l860_86024

theorem circle_segment_area (r chord_length intersection_dist : ℝ) 
  (hr : r = 45)
  (hc : chord_length = 84)
  (hi : intersection_dist = 15) : 
  ∃ (m n d : ℝ), 
    (m = 506.25 ∧ n = 1012.5 ∧ d = 1) ∧
    (∃ (area : ℝ), area = m * Real.pi - n * Real.sqrt d) :=
by sorry

end NUMINAMATH_CALUDE_circle_segment_area_l860_86024


namespace NUMINAMATH_CALUDE_line_passes_through_circle_center_l860_86066

/-- The line equation -/
def line_equation (x y m : ℝ) : Prop := x - 2*y + m = 0

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

/-- The center of the circle -/
def circle_center (x y : ℝ) : Prop := 
  circle_equation x y ∧ ∀ x' y', circle_equation x' y' → (x - x')^2 + (y - y')^2 ≤ (x' - x)^2 + (y' - y)^2

/-- The theorem statement -/
theorem line_passes_through_circle_center :
  ∃ m : ℝ, ∀ x y : ℝ, circle_center x y → line_equation x y m := by sorry

end NUMINAMATH_CALUDE_line_passes_through_circle_center_l860_86066


namespace NUMINAMATH_CALUDE_solve_for_y_l860_86005

theorem solve_for_y (x y : ℤ) (h1 : x^2 + x + 6 = y - 6) (h2 : x = -5) : y = 32 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l860_86005


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l860_86043

/-- The area of a circle with diameter 10 meters is 25π square meters. -/
theorem circle_area_with_diameter_10 :
  ∀ (circle_area : ℝ → ℝ) (pi : ℝ),
  (∀ r, circle_area r = pi * r^2) →
  circle_area 5 = 25 * pi := by
  sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l860_86043


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l860_86071

theorem system_of_equations_solution : ∀ x y : ℚ,
  (6 * x - 48 * y = 2) ∧ (3 * y - x = 4) →
  x^2 + y^2 = 442 / 25 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l860_86071


namespace NUMINAMATH_CALUDE_negation_of_proposition_P_l860_86050

theorem negation_of_proposition_P :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_P_l860_86050


namespace NUMINAMATH_CALUDE_sum_58_29_rounded_to_nearest_ten_l860_86004

/-- Rounds a number to the nearest multiple of 10 -/
def roundToNearestTen (x : ℤ) : ℤ :=
  10 * ((x + 5) / 10)

/-- The sum of 58 and 29 rounded to the nearest ten is 90 -/
theorem sum_58_29_rounded_to_nearest_ten :
  roundToNearestTen (58 + 29) = 90 := by
  sorry

end NUMINAMATH_CALUDE_sum_58_29_rounded_to_nearest_ten_l860_86004


namespace NUMINAMATH_CALUDE_daps_to_dips_l860_86031

/-- Representation of the currency conversion problem -/
structure Currency where
  daps : ℚ
  dops : ℚ
  dips : ℚ

/-- The conversion rates between currencies -/
def conversion_rates : Currency → Prop
  | c => c.daps * 4 = c.dops * 5 ∧ c.dops * 10 = c.dips * 4

/-- Theorem stating the equivalence of 125 daps to 50 dips -/
theorem daps_to_dips (c : Currency) (h : conversion_rates c) : 
  c.daps * 50 = c.dips * 125 := by
  sorry

end NUMINAMATH_CALUDE_daps_to_dips_l860_86031


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l860_86007

/-- 
Given a tetrahedron with:
- a, b: lengths of two opposite edges
- d: distance between edges a and b
- φ: angle between edges a and b
- V: volume of the tetrahedron

The volume V is equal to (1/6) * a * b * d * sin(φ)
-/
theorem tetrahedron_volume 
  (a b d φ V : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hd : d > 0) 
  (hφ : 0 < φ ∧ φ < π) 
  (hV : V > 0) :
  V = (1/6) * a * b * d * Real.sin φ :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l860_86007


namespace NUMINAMATH_CALUDE_book_cost_price_l860_86001

/-- Given a book sold for Rs 90 with a profit rate of 80%, prove that the cost price is Rs 50. -/
theorem book_cost_price (selling_price : ℝ) (profit_rate : ℝ) (h1 : selling_price = 90) (h2 : profit_rate = 80) :
  ∃ (cost_price : ℝ), cost_price = 50 ∧ profit_rate / 100 = (selling_price - cost_price) / cost_price :=
by sorry

end NUMINAMATH_CALUDE_book_cost_price_l860_86001


namespace NUMINAMATH_CALUDE_amelia_tuesday_distance_l860_86006

/-- The distance Amelia drove on Tuesday -/
def tuesday_distance (total_distance monday_distance remaining_distance : ℕ) : ℕ :=
  total_distance - (monday_distance + remaining_distance)

theorem amelia_tuesday_distance :
  tuesday_distance 8205 907 6716 = 582 := by
  sorry

end NUMINAMATH_CALUDE_amelia_tuesday_distance_l860_86006


namespace NUMINAMATH_CALUDE_travel_time_difference_l860_86097

def speed_A : ℝ := 60
def speed_B : ℝ := 45
def distance : ℝ := 360

theorem travel_time_difference :
  (distance / speed_B - distance / speed_A) * 60 = 120 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_difference_l860_86097


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l860_86092

theorem gcd_digits_bound (a b : ℕ) : 
  (1000000 ≤ a ∧ a < 10000000) →
  (1000000 ≤ b ∧ b < 10000000) →
  (10000000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 100000000000) →
  Nat.gcd a b < 10000 := by
sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l860_86092


namespace NUMINAMATH_CALUDE_assignment_satisfies_conditions_l860_86045

-- Define the set of people
inductive Person : Type
| Arthur : Person
| Burton : Person
| Congreve : Person
| Downs : Person
| Ewald : Person
| Flynn : Person

-- Define the set of positions
inductive Position : Type
| President : Position
| VicePresident : Position
| Secretary : Position
| Treasurer : Position

-- Define the assignment function
def assignment : Position → Person
| Position.President => Person.Flynn
| Position.VicePresident => Person.Ewald
| Position.Secretary => Person.Congreve
| Position.Treasurer => Person.Burton

-- Define the conditions
def arthur_condition (a : Position → Person) : Prop :=
  (a Position.VicePresident ≠ Person.Arthur) ∨ (a Position.President ≠ Person.Burton ∧ a Position.VicePresident ≠ Person.Burton ∧ a Position.Secretary ≠ Person.Burton ∧ a Position.Treasurer ≠ Person.Burton)

def burton_condition (a : Position → Person) : Prop :=
  a Position.VicePresident ≠ Person.Burton ∧ a Position.Secretary ≠ Person.Burton

def congreve_condition (a : Position → Person) : Prop :=
  (a Position.President ≠ Person.Burton ∧ a Position.VicePresident ≠ Person.Burton ∧ a Position.Secretary ≠ Person.Burton ∧ a Position.Treasurer ≠ Person.Burton) ∨
  (a Position.President = Person.Flynn ∨ a Position.VicePresident = Person.Flynn ∨ a Position.Secretary = Person.Flynn ∨ a Position.Treasurer = Person.Flynn)

def downs_condition (a : Position → Person) : Prop :=
  (a Position.President ≠ Person.Ewald ∧ a Position.VicePresident ≠ Person.Ewald ∧ a Position.Secretary ≠ Person.Ewald ∧ a Position.Treasurer ≠ Person.Ewald) ∧
  (a Position.President ≠ Person.Flynn ∧ a Position.VicePresident ≠ Person.Flynn ∧ a Position.Secretary ≠ Person.Flynn ∧ a Position.Treasurer ≠ Person.Flynn)

def ewald_condition (a : Position → Person) : Prop :=
  ¬(a Position.President = Person.Arthur ∧ (a Position.VicePresident = Person.Burton ∨ a Position.Secretary = Person.Burton ∨ a Position.Treasurer = Person.Burton)) ∧
  ¬(a Position.VicePresident = Person.Arthur ∧ (a Position.President = Person.Burton ∨ a Position.Secretary = Person.Burton ∨ a Position.Treasurer = Person.Burton)) ∧
  ¬(a Position.Secretary = Person.Arthur ∧ (a Position.President = Person.Burton ∨ a Position.VicePresident = Person.Burton ∨ a Position.Treasurer = Person.Burton)) ∧
  ¬(a Position.Treasurer = Person.Arthur ∧ (a Position.President = Person.Burton ∨ a Position.VicePresident = Person.Burton ∨ a Position.Secretary = Person.Burton))

def flynn_condition (a : Position → Person) : Prop :=
  (a Position.President = Person.Flynn) → (a Position.VicePresident ≠ Person.Congreve)

-- Theorem statement
theorem assignment_satisfies_conditions :
  arthur_condition assignment ∧
  burton_condition assignment ∧
  congreve_condition assignment ∧
  downs_condition assignment ∧
  ewald_condition assignment ∧
  flynn_condition assignment :=
sorry

end NUMINAMATH_CALUDE_assignment_satisfies_conditions_l860_86045


namespace NUMINAMATH_CALUDE_final_jellybean_count_l860_86085

def jellybean_count (initial : ℕ) (first_removal : ℕ) (addition : ℕ) (second_removal : ℕ) : ℕ :=
  initial - first_removal + addition - second_removal

theorem final_jellybean_count :
  jellybean_count 37 15 5 4 = 23 := by
  sorry

end NUMINAMATH_CALUDE_final_jellybean_count_l860_86085


namespace NUMINAMATH_CALUDE_investment_difference_proof_l860_86094

/-- Represents an investment scheme with an initial investment and a yield rate -/
structure Scheme where
  investment : ℝ
  yieldRate : ℝ

/-- Calculates the total amount in a scheme after a year -/
def totalAfterYear (s : Scheme) : ℝ :=
  s.investment + s.investment * s.yieldRate

/-- The difference in total amounts between two schemes after a year -/
def schemeDifference (s1 s2 : Scheme) : ℝ :=
  totalAfterYear s1 - totalAfterYear s2

theorem investment_difference_proof (schemeA schemeB : Scheme) 
  (h1 : schemeA.investment = 300)
  (h2 : schemeB.investment = 200)
  (h3 : schemeA.yieldRate = 0.3)
  (h4 : schemeB.yieldRate = 0.5) :
  schemeDifference schemeA schemeB = 90 := by
  sorry

end NUMINAMATH_CALUDE_investment_difference_proof_l860_86094


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l860_86046

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Theorem 1: B ⊆ A iff m ∈ (-∞, 3]
theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by sorry

-- Theorem 2: A ∩ B = ∅ iff m ∈ (-∞, 2) ∪ (4, +∞)
theorem disjoint_condition (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l860_86046


namespace NUMINAMATH_CALUDE_cos_beta_value_l860_86069

theorem cos_beta_value (α β : Real) (P : ℝ × ℝ) :
  P = (3, 4) →
  P.1 = 3 * Real.cos α ∧ P.2 = 3 * Real.sin α →
  Real.cos (α + β) = 1/3 →
  β ∈ Set.Ioo 0 Real.pi →
  Real.cos β = (3 + 8 * Real.sqrt 2) / 15 := by
  sorry

end NUMINAMATH_CALUDE_cos_beta_value_l860_86069


namespace NUMINAMATH_CALUDE_speed_calculation_l860_86052

theorem speed_calculation (distance : ℝ) (early_time : ℝ) (speed_reduction : ℝ) : 
  distance = 40 ∧ early_time = 4/60 ∧ speed_reduction = 5 →
  ∃ (v : ℝ), v > 0 ∧ 
    (distance / v = distance / (v - speed_reduction) - early_time) ↔ 
    v = 60 := by sorry

end NUMINAMATH_CALUDE_speed_calculation_l860_86052
