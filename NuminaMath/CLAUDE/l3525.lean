import Mathlib

namespace NUMINAMATH_CALUDE_eliminate_denominators_l3525_352573

theorem eliminate_denominators (x : ℝ) : 
  (2*x - 3) / 5 = 2*x / 3 - 3 ↔ 3*(2*x - 3) = 5*(2*x) - 3*15 := by
  sorry

end NUMINAMATH_CALUDE_eliminate_denominators_l3525_352573


namespace NUMINAMATH_CALUDE_seeds_in_fourth_pot_is_one_l3525_352537

/-- Given a total number of seeds, number of pots, and number of seeds per pot for the first three pots,
    calculate the number of seeds that will be planted in the fourth pot. -/
def seeds_in_fourth_pot (total_seeds : ℕ) (num_pots : ℕ) (seeds_per_pot : ℕ) : ℕ :=
  total_seeds - (seeds_per_pot * (num_pots - 1))

/-- Theorem stating that for the given problem, the number of seeds in the fourth pot is 1. -/
theorem seeds_in_fourth_pot_is_one :
  seeds_in_fourth_pot 10 4 3 = 1 := by
  sorry

#eval seeds_in_fourth_pot 10 4 3

end NUMINAMATH_CALUDE_seeds_in_fourth_pot_is_one_l3525_352537


namespace NUMINAMATH_CALUDE_grape_juice_percentage_l3525_352526

/-- Calculates the percentage of grape juice in a mixture after adding pure grape juice --/
theorem grape_juice_percentage
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_grape_juice : ℝ)
  (h1 : initial_volume = 30)
  (h2 : initial_concentration = 0.1)
  (h3 : added_grape_juice = 10) :
  let final_volume := initial_volume + added_grape_juice
  let initial_grape_juice := initial_volume * initial_concentration
  let final_grape_juice := initial_grape_juice + added_grape_juice
  final_grape_juice / final_volume = 0.325 := by
  sorry

end NUMINAMATH_CALUDE_grape_juice_percentage_l3525_352526


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_range_of_a_when_B_equals_A_l3525_352538

-- Define sets A and B
def A : Set ℝ := {x | (x + 2) * (x - 5) < 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}

-- Part 1
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B 2) = {x | -2 < x ∧ x ≤ 1 ∨ 3 ≤ x ∧ x < 5} := by sorry

-- Part 2
theorem range_of_a_when_B_equals_A :
  (∀ x, x ∈ B a ↔ x ∈ A) → -1 ≤ a ∧ a ≤ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_range_of_a_when_B_equals_A_l3525_352538


namespace NUMINAMATH_CALUDE_albert_books_multiple_l3525_352593

theorem albert_books_multiple (stu_books : ℕ) (total_books : ℕ) (x : ℚ) : 
  stu_books = 9 →
  total_books = 45 →
  total_books = stu_books + stu_books * x →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_albert_books_multiple_l3525_352593


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l3525_352511

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the theorem
theorem sum_of_coefficients_equals_one (a b : ℝ) : 
  (i^2 + a * i + b = 0) → (a + b = 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l3525_352511


namespace NUMINAMATH_CALUDE_A_intersect_B_is_empty_l3525_352587

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}
def B : Set ℝ := {x | x - 1 > 0}

-- Statement to prove
theorem A_intersect_B_is_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_is_empty_l3525_352587


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3525_352528

/-- A line that always passes through a fixed point regardless of the parameter m -/
def always_passes_through (m : ℝ) : Prop :=
  (m - 1) * (-3) + (2 * m - 3) * 1 + m = 0

/-- The theorem stating that the line passes through (-3, 1) for all real m -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, always_passes_through m :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3525_352528


namespace NUMINAMATH_CALUDE_sum_A_B_linear_combo_A_B_diff_A_B_specific_l3525_352524

-- Define A and B as functions of a and b
def A (a b : ℚ) : ℚ := 4 * a^2 * b - 3 * a * b + b^2
def B (a b : ℚ) : ℚ := a^2 - 3 * a^2 * b + 3 * a * b - b^2

-- Theorem 1: A + B = a² + a²b
theorem sum_A_B (a b : ℚ) : A a b + B a b = a^2 + a^2 * b := by sorry

-- Theorem 2: 3A + 4B = 4a² + 3ab - b²
theorem linear_combo_A_B (a b : ℚ) : 3 * A a b + 4 * B a b = 4 * a^2 + 3 * a * b - b^2 := by sorry

-- Theorem 3: A - B = -63/8 when a = 2 and b = -1/4
theorem diff_A_B_specific : A 2 (-1/4) - B 2 (-1/4) = -63/8 := by sorry

end NUMINAMATH_CALUDE_sum_A_B_linear_combo_A_B_diff_A_B_specific_l3525_352524


namespace NUMINAMATH_CALUDE_rugs_bought_is_twenty_l3525_352561

/-- Calculates the number of rugs bought given buying price, selling price, and total profit -/
def rugs_bought (buying_price selling_price total_profit : ℚ) : ℚ :=
  total_profit / (selling_price - buying_price)

/-- Theorem stating that the number of rugs bought is 20 -/
theorem rugs_bought_is_twenty :
  rugs_bought 40 60 400 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rugs_bought_is_twenty_l3525_352561


namespace NUMINAMATH_CALUDE_u_converges_to_L_least_k_for_bound_l3525_352599

def u : ℕ → ℚ
  | 0 => 1/8
  | n + 1 => 2 * u n - 2 * (u n)^2

def L : ℚ := 1/2

def converges_to (a : ℕ → ℚ) (l : ℚ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - l| < ε

theorem u_converges_to_L : converges_to u L := sorry

theorem least_k_for_bound :
  (∃ k, |u k - L| ≤ 1/2^10) ∧
  (∀ k < 4, |u k - L| > 1/2^10) ∧
  |u 4 - L| ≤ 1/2^10 := sorry

end NUMINAMATH_CALUDE_u_converges_to_L_least_k_for_bound_l3525_352599


namespace NUMINAMATH_CALUDE_ratio_problem_l3525_352557

/-- Given ratios A : B : C where A = 4x, B = 6x, C = 9x, and A = 50, prove the values of B and C and their average -/
theorem ratio_problem (x : ℚ) (A B C : ℚ) (h1 : A = 4 * x) (h2 : B = 6 * x) (h3 : C = 9 * x) (h4 : A = 50) :
  B = 75 ∧ C = 112.5 ∧ (B + C) / 2 = 93.75 := by
  sorry


end NUMINAMATH_CALUDE_ratio_problem_l3525_352557


namespace NUMINAMATH_CALUDE_national_day_2020_l3525_352521

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

-- Theorem statement
theorem national_day_2020 (national_day_2019 : DayOfWeek) 
  (h1 : national_day_2019 = DayOfWeek.Tuesday) 
  (h2 : advanceDay national_day_2019 2 = DayOfWeek.Thursday) : 
  advanceDay national_day_2019 2 = DayOfWeek.Thursday := by
  sorry

#check national_day_2020

end NUMINAMATH_CALUDE_national_day_2020_l3525_352521


namespace NUMINAMATH_CALUDE_jacket_price_after_discounts_l3525_352572

def initial_price : ℝ := 20
def first_discount : ℝ := 0.40
def second_discount : ℝ := 0.25

theorem jacket_price_after_discounts :
  let price_after_first := initial_price * (1 - first_discount)
  let final_price := price_after_first * (1 - second_discount)
  final_price = 9 := by sorry

end NUMINAMATH_CALUDE_jacket_price_after_discounts_l3525_352572


namespace NUMINAMATH_CALUDE_ring_stack_distance_l3525_352596

/-- Represents a stack of metallic rings -/
structure RingStack where
  topDiameter : ℕ
  smallestDiameter : ℕ
  thickness : ℕ

/-- Calculates the total vertical distance of a ring stack -/
def totalVerticalDistance (stack : RingStack) : ℕ :=
  let numRings := (stack.topDiameter - stack.smallestDiameter) / 2 + 1
  let sumDiameters := numRings * (stack.topDiameter + stack.smallestDiameter) / 2
  sumDiameters - numRings + 2 * stack.thickness

/-- Theorem stating the total vertical distance of the given ring stack -/
theorem ring_stack_distance :
  ∀ (stack : RingStack),
    stack.topDiameter = 22 ∧
    stack.smallestDiameter = 4 ∧
    stack.thickness = 1 →
    totalVerticalDistance stack = 122 := by
  sorry


end NUMINAMATH_CALUDE_ring_stack_distance_l3525_352596


namespace NUMINAMATH_CALUDE_thousandths_place_of_seven_thirty_seconds_l3525_352516

theorem thousandths_place_of_seven_thirty_seconds (n : ℕ) : 
  (7 : ℚ) / 32 = n / 1000 + (8 : ℚ) / 1000 + m / 10000 → n < 9 ∧ 0 ≤ m ∧ m < 10 :=
by sorry

end NUMINAMATH_CALUDE_thousandths_place_of_seven_thirty_seconds_l3525_352516


namespace NUMINAMATH_CALUDE_all_items_used_as_money_l3525_352518

structure MoneyItem where
  name : String
  used_as_money : Bool

def gold : MoneyItem := { name := "gold", used_as_money := true }
def stones : MoneyItem := { name := "stones", used_as_money := true }
def horses : MoneyItem := { name := "horses", used_as_money := true }
def dried_fish : MoneyItem := { name := "dried fish", used_as_money := true }
def mollusk_shells : MoneyItem := { name := "mollusk shells", used_as_money := true }

def money_items : List MoneyItem := [gold, stones, horses, dried_fish, mollusk_shells]

theorem all_items_used_as_money :
  (∀ item ∈ money_items, item.used_as_money = true) →
  (¬ ∃ item ∈ money_items, item.used_as_money = false) := by
  sorry

end NUMINAMATH_CALUDE_all_items_used_as_money_l3525_352518


namespace NUMINAMATH_CALUDE_percentage_seats_sold_l3525_352522

def stadium_capacity : ℕ := 60000
def fans_stayed_home : ℕ := 5000
def fans_attended : ℕ := 40000

theorem percentage_seats_sold :
  (fans_attended + fans_stayed_home) / stadium_capacity * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_seats_sold_l3525_352522


namespace NUMINAMATH_CALUDE_smallest_divisor_of_930_l3525_352581

theorem smallest_divisor_of_930 : ∃ (d : ℕ), d > 1 ∧ d ∣ 930 ∧ ∀ (k : ℕ), 1 < k ∧ k < d → ¬(k ∣ 930) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_of_930_l3525_352581


namespace NUMINAMATH_CALUDE_total_speech_time_l3525_352500

def speech_time (outline_time writing_time rewrite_time practice_time break1_time break2_time : ℝ) : ℝ :=
  outline_time + writing_time + rewrite_time + practice_time + break1_time + break2_time

theorem total_speech_time :
  let outline_time : ℝ := 30
  let break1_time : ℝ := 10
  let writing_time : ℝ := outline_time + 28
  let rewrite_time : ℝ := 15
  let break2_time : ℝ := 5
  let practice_time : ℝ := (writing_time + rewrite_time) / 2
  speech_time outline_time writing_time rewrite_time practice_time break1_time break2_time = 154.5 := by
  sorry

end NUMINAMATH_CALUDE_total_speech_time_l3525_352500


namespace NUMINAMATH_CALUDE_polynomial_with_arithmetic_progression_roots_l3525_352508

/-- A polynomial of the form x^4 + mx^2 + nx + 144 with four distinct real roots in arithmetic progression has m = -40 -/
theorem polynomial_with_arithmetic_progression_roots (m n : ℝ) : 
  (∃ (b d : ℝ) (h_distinct : d ≠ 0), 
    (∀ x : ℝ, x^4 + m*x^2 + n*x + 144 = (x - b)*(x - (b + d))*(x - (b + 2*d))*(x - (b + 3*d))) ∧
    (b ≠ b + d) ∧ (b + d ≠ b + 2*d) ∧ (b + 2*d ≠ b + 3*d)) →
  m = -40 := by
sorry

end NUMINAMATH_CALUDE_polynomial_with_arithmetic_progression_roots_l3525_352508


namespace NUMINAMATH_CALUDE_min_distance_MN_l3525_352539

def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

def point_on_line (x y : ℝ) : Prop := ∃ (m b : ℝ), y = m*x + b ∧ 1 = m*3 + b

def intersect_points (M N : ℝ × ℝ) : Prop :=
  point_on_line M.1 M.2 ∧ point_on_line N.1 N.2 ∧
  circle_equation M.1 M.2 ∧ circle_equation N.1 N.2

theorem min_distance_MN :
  ∀ (M N : ℝ × ℝ), intersect_points M N →
  ∃ (MN : ℝ × ℝ → ℝ × ℝ → ℝ),
    (∀ (A B : ℝ × ℝ), MN A B ≥ 0) ∧
    (MN M N ≥ 4) ∧
    (∃ (M' N' : ℝ × ℝ), intersect_points M' N' ∧ MN M' N' = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_MN_l3525_352539


namespace NUMINAMATH_CALUDE_max_missed_questions_to_pass_l3525_352543

theorem max_missed_questions_to_pass (total_questions : ℕ) (passing_percentage : ℚ) 
  (h1 : total_questions = 40)
  (h2 : passing_percentage = 75/100) : 
  ∃ (max_missed : ℕ), max_missed = 10 ∧ 
    (total_questions - max_missed : ℚ) / total_questions ≥ passing_percentage :=
by sorry

end NUMINAMATH_CALUDE_max_missed_questions_to_pass_l3525_352543


namespace NUMINAMATH_CALUDE_bus_passengers_l3525_352549

theorem bus_passengers (total : ℕ) (women_fraction : ℚ) (standing_men_fraction : ℚ) 
  (h1 : total = 48)
  (h2 : women_fraction = 2/3)
  (h3 : standing_men_fraction = 1/8) : 
  ↑total * (1 - women_fraction) * (1 - standing_men_fraction) = 14 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l3525_352549


namespace NUMINAMATH_CALUDE_remainder_of_n_squared_plus_2n_plus_3_l3525_352531

theorem remainder_of_n_squared_plus_2n_plus_3 (n : ℤ) (k : ℤ) (h : n = 100 * k - 1) :
  (n^2 + 2*n + 3) % 100 = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_n_squared_plus_2n_plus_3_l3525_352531


namespace NUMINAMATH_CALUDE_complement_union_equal_l3525_352513

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {0, 3, 4}
def B : Set Nat := {1, 3}

theorem complement_union_equal : (U \ A) ∪ B = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_union_equal_l3525_352513


namespace NUMINAMATH_CALUDE_tens_digit_of_sum_is_zero_l3525_352502

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 100) = (n % 10) - 1 ∧
  ((n / 10) % 10) = (n % 10) + 3

def reverse_number (n : ℕ) : ℕ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

theorem tens_digit_of_sum_is_zero (n : ℕ) (h : is_valid_number n) :
  ((n + reverse_number n) / 10) % 10 = 0 :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_sum_is_zero_l3525_352502


namespace NUMINAMATH_CALUDE_smallest_n_with_19_odd_digit_squares_l3525_352551

/-- A function that returns true if a number has an odd number of digits, false otherwise -/
def has_odd_digits (n : ℕ) : Bool :=
  sorry

/-- A function that counts how many numbers from 1 to n have squares with an odd number of digits -/
def count_odd_digit_squares (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that 44 is the smallest natural number N such that
    among the squares of integers from 1 to N, exactly 19 of them have an odd number of digits -/
theorem smallest_n_with_19_odd_digit_squares :
  ∀ n : ℕ, n < 44 → count_odd_digit_squares n < 19 ∧ count_odd_digit_squares 44 = 19 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_19_odd_digit_squares_l3525_352551


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3525_352595

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 96 → s^3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l3525_352595


namespace NUMINAMATH_CALUDE_marbles_probability_l3525_352527

def total_marbles : ℕ := 13
def black_marbles : ℕ := 4
def red_marbles : ℕ := 3
def green_marbles : ℕ := 6
def drawn_marbles : ℕ := 2

def prob_same_color : ℚ := 
  (black_marbles * (black_marbles - 1) + 
   red_marbles * (red_marbles - 1) + 
   green_marbles * (green_marbles - 1)) / 
  (total_marbles * (total_marbles - 1))

theorem marbles_probability : 
  prob_same_color = 4 / 13 :=
sorry

end NUMINAMATH_CALUDE_marbles_probability_l3525_352527


namespace NUMINAMATH_CALUDE_remainder_when_n_plus_3_and_n_plus_7_prime_l3525_352553

theorem remainder_when_n_plus_3_and_n_plus_7_prime (n : ℕ) 
  (h1 : Nat.Prime (n + 3)) 
  (h2 : Nat.Prime (n + 7)) : 
  n % 3 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_when_n_plus_3_and_n_plus_7_prime_l3525_352553


namespace NUMINAMATH_CALUDE_circles_intersection_sum_l3525_352509

/-- Given two circles intersecting at points (1, 3) and (m, 1), with their centers 
    on the line x - y + c/2 = 0, prove that m + c = 3 -/
theorem circles_intersection_sum (m c : ℝ) : 
  (∃ (circle1 circle2 : Set (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ circle1 ∩ circle2 ↔ ((x = 1 ∧ y = 3) ∨ (x = m ∧ y = 1))) ∧
    (∃ (x1 y1 x2 y2 : ℝ), 
      (x1, y1) ∈ circle1 ∧ (x2, y2) ∈ circle2 ∧
      x1 - y1 + c/2 = 0 ∧ x2 - y2 + c/2 = 0)) →
  m + c = 3 := by
sorry

end NUMINAMATH_CALUDE_circles_intersection_sum_l3525_352509


namespace NUMINAMATH_CALUDE_positions_after_307_moves_l3525_352536

/-- Represents the positions of the cat -/
inductive CatPosition
  | Top
  | TopRight
  | BottomRight
  | Bottom
  | BottomLeft
  | TopLeft

/-- Represents the positions of the mouse -/
inductive MousePosition
  | Top
  | TopRight
  | BetweenTopRightAndBottomRight
  | BottomRight
  | BetweenBottomRightAndBottom
  | Bottom
  | BottomLeft
  | BetweenBottomLeftAndTopLeft
  | TopLeft
  | BetweenTopLeftAndTop

/-- The number of hexagons in the larger hexagon -/
def numHexagons : Nat := 6

/-- The number of segments the mouse moves through -/
def numMouseSegments : Nat := 12

/-- Calculates the cat's position after a given number of moves -/
def catPositionAfterMoves (moves : Nat) : CatPosition :=
  match moves % numHexagons with
  | 0 => CatPosition.TopLeft
  | 1 => CatPosition.Top
  | 2 => CatPosition.TopRight
  | 3 => CatPosition.BottomRight
  | 4 => CatPosition.Bottom
  | 5 => CatPosition.BottomLeft
  | _ => CatPosition.Top  -- This case should never occur

/-- Calculates the mouse's position after a given number of moves -/
def mousePositionAfterMoves (moves : Nat) : MousePosition :=
  match moves % numMouseSegments with
  | 0 => MousePosition.Top
  | 1 => MousePosition.BetweenTopLeftAndTop
  | 2 => MousePosition.TopLeft
  | 3 => MousePosition.BetweenBottomLeftAndTopLeft
  | 4 => MousePosition.BottomLeft
  | 5 => MousePosition.Bottom
  | 6 => MousePosition.BetweenBottomRightAndBottom
  | 7 => MousePosition.BottomRight
  | 8 => MousePosition.BetweenTopRightAndBottomRight
  | 9 => MousePosition.TopRight
  | 10 => MousePosition.Top
  | 11 => MousePosition.BetweenTopLeftAndTop
  | _ => MousePosition.Top  -- This case should never occur

theorem positions_after_307_moves :
  catPositionAfterMoves 307 = CatPosition.Top ∧
  mousePositionAfterMoves 307 = MousePosition.BetweenBottomRightAndBottom :=
by sorry

end NUMINAMATH_CALUDE_positions_after_307_moves_l3525_352536


namespace NUMINAMATH_CALUDE_f_max_value_l3525_352545

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (x + Real.sin x) + Real.sin (x - Real.sin x) + (Real.pi / 2 - 2) * Real.sin (Real.sin x)

theorem f_max_value :
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (M = (Real.pi - 2) / Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l3525_352545


namespace NUMINAMATH_CALUDE_four_numbers_in_interval_l3525_352544

theorem four_numbers_in_interval (a b c d : Real) : 
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < π / 2 →
  ∃ x y, (x = a ∨ x = b ∨ x = c ∨ x = d) ∧
         (y = a ∨ y = b ∨ y = c ∨ y = d) ∧
         x ≠ y ∧
         |x - y| < π / 6 :=
by sorry

end NUMINAMATH_CALUDE_four_numbers_in_interval_l3525_352544


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3525_352564

theorem triangle_perimeter (a b c : ℝ) :
  |a - 2 * Real.sqrt 2| + Real.sqrt (b - 5) + (c - 3 * Real.sqrt 2)^2 = 0 →
  a + b + c = 5 + 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3525_352564


namespace NUMINAMATH_CALUDE_remainder_calculation_l3525_352517

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem remainder_calculation :
  rem (5/9 : ℚ) (-3/7 : ℚ) = -19/63 := by
  sorry

end NUMINAMATH_CALUDE_remainder_calculation_l3525_352517


namespace NUMINAMATH_CALUDE_video_game_pricing_l3525_352530

theorem video_game_pricing (total_games : ℕ) (non_working_games : ℕ) (total_earnings : ℕ) :
  total_games = 15 →
  non_working_games = 9 →
  total_earnings = 30 →
  (total_earnings : ℚ) / (total_games - non_working_games : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_video_game_pricing_l3525_352530


namespace NUMINAMATH_CALUDE_x_plus_reciprocal_two_implies_x_twelve_one_l3525_352503

theorem x_plus_reciprocal_two_implies_x_twelve_one (x : ℝ) (h : x + 1/x = 2) : x^12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_reciprocal_two_implies_x_twelve_one_l3525_352503


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l3525_352562

-- Equation 1
theorem equation_one_solution (x : ℚ) : 
  (3 / (2 * x - 2) + 1 / (1 - x) = 3) ↔ (x = 7/6) :=
sorry

-- Equation 2
theorem equation_two_no_solution :
  ¬∃ y : ℚ, y / (y - 1) - 2 / (y^2 - 1) = 1 :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l3525_352562


namespace NUMINAMATH_CALUDE_railway_stations_problem_l3525_352597

theorem railway_stations_problem (m n : ℕ) (h1 : n ≥ 1) :
  (m.choose 2 + n * m + n.choose 2) - m.choose 2 = 58 →
  ((m = 14 ∧ n = 2) ∨ (m = 29 ∧ n = 1)) := by
  sorry

end NUMINAMATH_CALUDE_railway_stations_problem_l3525_352597


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l3525_352586

theorem pure_imaginary_ratio (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : ∃ (t : ℝ), (3 - 4 * Complex.I) * (x + y * Complex.I) = t * Complex.I) : 
  x / y = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l3525_352586


namespace NUMINAMATH_CALUDE_proposition_implications_l3525_352510

theorem proposition_implications (p q : Prop) :
  ¬(¬p ∨ ¬q) → (p ∧ q) ∧ (p ∨ q) :=
by sorry

end NUMINAMATH_CALUDE_proposition_implications_l3525_352510


namespace NUMINAMATH_CALUDE_probability_of_red_is_one_fifth_l3525_352567

/-- Represents the contents of the bag -/
structure BagContents where
  red : ℕ
  white : ℕ
  black : ℕ

/-- Calculates the probability of drawing a red ball -/
def probabilityOfRed (bag : BagContents) : ℚ :=
  bag.red / (bag.red + bag.white + bag.black)

/-- Theorem stating that the probability of drawing a red ball is 1/5 -/
theorem probability_of_red_is_one_fifth (bag : BagContents) 
  (h1 : bag.red = 2) 
  (h2 : bag.white = 3) 
  (h3 : bag.black = 5) : 
  probabilityOfRed bag = 1/5 := by
  sorry

#check probability_of_red_is_one_fifth

end NUMINAMATH_CALUDE_probability_of_red_is_one_fifth_l3525_352567


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_value_l3525_352525

/-- A geometric sequence with first term 2 and satisfying a₃a₅ = 4a₆² has a₃ = 1 -/
theorem geometric_sequence_a3_value (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n, a n = 2 * r^(n-1))  -- {aₙ} is a geometric sequence
  → a 1 = 2                          -- a₁ = 2
  → a 3 * a 5 = 4 * (a 6)^2          -- a₃a₅ = 4a₆²
  → a 3 = 1                          -- a₃ = 1
:= by sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_value_l3525_352525


namespace NUMINAMATH_CALUDE_unique_root_condition_l3525_352519

/-- The equation x + 1 = √(px) has exactly one real root if and only if p = 4 or p ≤ 0. -/
theorem unique_root_condition (p : ℝ) : 
  (∃! x : ℝ, x + 1 = Real.sqrt (p * x)) ↔ p = 4 ∨ p ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_condition_l3525_352519


namespace NUMINAMATH_CALUDE_greatest_prime_divisor_digit_sum_l3525_352592

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem greatest_prime_divisor_digit_sum :
  ∃ (p : ℕ), is_prime p ∧ (32767 % p = 0) ∧
  (∀ q : ℕ, is_prime q → (32767 % q = 0) → q ≤ p) ∧
  sum_of_digits p = 14 := by
  sorry

end NUMINAMATH_CALUDE_greatest_prime_divisor_digit_sum_l3525_352592


namespace NUMINAMATH_CALUDE_original_polygon_sides_l3525_352552

theorem original_polygon_sides (n : ℕ) : 
  (∃ m : ℕ, (m - 2) * 180 = 1620 ∧ 
  (n = m + 1 ∨ n = m ∨ n = m - 1)) →
  (n = 10 ∨ n = 11 ∨ n = 12) :=
by sorry

end NUMINAMATH_CALUDE_original_polygon_sides_l3525_352552


namespace NUMINAMATH_CALUDE_product_mod_25_l3525_352571

theorem product_mod_25 :
  ∃ m : ℕ, 0 ≤ m ∧ m < 25 ∧ (105 * 77 * 132) % 25 = m ∧ m = 20 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_25_l3525_352571


namespace NUMINAMATH_CALUDE_area_of_rectangle_with_three_squares_l3525_352547

/-- Given three non-overlapping squares where one square has twice the side length of the other two,
    and the larger square has an area of 4 square inches, the area of the rectangle encompassing
    all three squares is 6 square inches. -/
theorem area_of_rectangle_with_three_squares (s : ℝ) : 
  s > 0 → (2 * s)^2 = 4 → 3 * s * 2 * s = 6 := by
  sorry

end NUMINAMATH_CALUDE_area_of_rectangle_with_three_squares_l3525_352547


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3525_352560

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≠ 0) ↔ (∃ x : ℝ, x^2 + x + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3525_352560


namespace NUMINAMATH_CALUDE_simplify_expression_l3525_352563

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 8) - (x + 6)*(3*x - 2) = 4*x - 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3525_352563


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3525_352540

/-- Proves that the cost price of an article is 40 given specific conditions --/
theorem cost_price_calculation (C M : ℝ) 
  (h1 : 0.95 * M = 1.25 * C)  -- Selling price after 5% discount equals 25% profit on cost
  (h2 : 0.95 * M = 50)        -- Selling price is 50
  : C = 40 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3525_352540


namespace NUMINAMATH_CALUDE_mary_final_book_count_l3525_352578

def calculate_final_books (initial_books : ℕ) (monthly_club_books : ℕ) (months : ℕ) 
  (bought_books : ℕ) (gift_books : ℕ) (removed_books : ℕ) : ℕ :=
  initial_books + monthly_club_books * months + bought_books + gift_books - removed_books

theorem mary_final_book_count : 
  calculate_final_books 72 1 12 7 5 15 = 81 := by sorry

end NUMINAMATH_CALUDE_mary_final_book_count_l3525_352578


namespace NUMINAMATH_CALUDE_alternate_interior_angles_parallel_l3525_352558

-- Define a structure for lines in a plane
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define a structure for angles
structure Angle :=
  (measure : ℝ)

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define a function to represent alternate interior angles
def alternate_interior_angles (l1 l2 : Line) (t : Line) : (Angle × Angle) :=
  sorry

-- Theorem statement
theorem alternate_interior_angles_parallel (l1 l2 t : Line) :
  let (angle1, angle2) := alternate_interior_angles l1 l2 t
  (angle1.measure = angle2.measure) → are_parallel l1 l2 :=
sorry

end NUMINAMATH_CALUDE_alternate_interior_angles_parallel_l3525_352558


namespace NUMINAMATH_CALUDE_lottery_probability_l3525_352535

theorem lottery_probability (winning_rate : ℚ) (num_tickets : ℕ) : 
  winning_rate = 1/3 → num_tickets = 3 → 
  (1 - (1 - winning_rate) ^ num_tickets) = 19/27 := by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_l3525_352535


namespace NUMINAMATH_CALUDE_clock_angle_at_3_40_l3525_352582

-- Define the clock and its properties
def clock_degrees : ℝ := 360
def minute_hand_speed : ℝ := 6
def hour_hand_speed : ℝ := 0.5
def time_elapsed : ℝ := 40  -- Minutes elapsed since 3:00

-- Define the positions of the hands at 3:40
def minute_hand_position : ℝ := time_elapsed * minute_hand_speed
def hour_hand_position : ℝ := 90 + time_elapsed * hour_hand_speed

-- Define the angle between the hands
def angle_between_hands : ℝ := |minute_hand_position - hour_hand_position|

-- Theorem statement
theorem clock_angle_at_3_40 : 
  min angle_between_hands (clock_degrees - angle_between_hands) = 130 :=
sorry

end NUMINAMATH_CALUDE_clock_angle_at_3_40_l3525_352582


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l3525_352541

theorem camping_trip_percentage :
  ∀ (total_percentage : ℝ) 
    (more_than_100 : ℝ) 
    (not_more_than_100 : ℝ),
  more_than_100 = 18 →
  total_percentage = more_than_100 + not_more_than_100 →
  total_percentage = 72 :=
by sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l3525_352541


namespace NUMINAMATH_CALUDE_sticker_distribution_l3525_352512

/-- The number of ways to distribute n identical objects among k distinct containers -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of stickers to be distributed -/
def num_stickers : ℕ := 11

/-- The number of sheets of paper -/
def num_sheets : ℕ := 5

theorem sticker_distribution :
  distribute num_stickers num_sheets = 1365 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l3525_352512


namespace NUMINAMATH_CALUDE_ratio_of_sums_l3525_352590

theorem ratio_of_sums (p q r u v w : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_sums_l3525_352590


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3525_352575

/-- The diagonal of a rectangle with length 40√3 cm and width 30√3 cm is 50√3 cm. -/
theorem rectangle_diagonal : 
  let length : ℝ := 40 * Real.sqrt 3
  let width : ℝ := 30 * Real.sqrt 3
  let diagonal : ℝ := Real.sqrt (length^2 + width^2)
  diagonal = 50 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l3525_352575


namespace NUMINAMATH_CALUDE_sales_volume_formula_max_profit_price_max_profit_value_profit_range_l3525_352505

/-- Represents the weekly sales volume as a function of price --/
def sales_volume (x : ℝ) : ℝ := -30 * x + 2100

/-- Represents the weekly profit as a function of price --/
def profit (x : ℝ) : ℝ := (x - 40) * (sales_volume x)

/-- The initial price in yuan --/
def initial_price : ℝ := 60

/-- The initial weekly sales in pieces --/
def initial_sales : ℝ := 300

/-- The cost price per piece in yuan --/
def cost_price : ℝ := 40

theorem sales_volume_formula (x : ℝ) : 
  sales_volume x = -30 * x + 2100 := by sorry

theorem max_profit_price : 
  ∃ (x : ℝ), ∀ (y : ℝ), profit x ≥ profit y ∧ x = 55 := by sorry

theorem max_profit_value : 
  profit 55 = 6750 := by sorry

theorem profit_range (x : ℝ) : 
  profit x ≥ 6480 ↔ 52 ≤ x ∧ x ≤ 58 := by sorry

end NUMINAMATH_CALUDE_sales_volume_formula_max_profit_price_max_profit_value_profit_range_l3525_352505


namespace NUMINAMATH_CALUDE_avg_problem_l3525_352570

/-- Average of two numbers -/
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

/-- Average of three numbers -/
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

/-- Theorem: The average of [2 4], [6 2], and [3 3] is 10/3 -/
theorem avg_problem : avg3 (avg2 2 4) (avg2 6 2) (avg2 3 3) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_avg_problem_l3525_352570


namespace NUMINAMATH_CALUDE_mary_initial_nickels_l3525_352515

/-- The number of nickels Mary initially had -/
def initial_nickels : ℕ := sorry

/-- The number of nickels Mary's dad gave her -/
def nickels_from_dad : ℕ := 5

/-- The total number of nickels Mary has now -/
def total_nickels : ℕ := 12

/-- Theorem stating that Mary initially had 7 nickels -/
theorem mary_initial_nickels : 
  initial_nickels = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_mary_initial_nickels_l3525_352515


namespace NUMINAMATH_CALUDE_parabola_through_point_l3525_352546

/-- A parabola passing through point (4, -2) has a standard equation of either x² = -8y or y² = x -/
theorem parabola_through_point (P : ℝ × ℝ) (h : P = (4, -2)) :
  (∃ (x y : ℝ), x^2 = -8*y ∧ P.1 = x ∧ P.2 = y) ∨
  (∃ (x y : ℝ), y^2 = x ∧ P.1 = x ∧ P.2 = y) := by
sorry

end NUMINAMATH_CALUDE_parabola_through_point_l3525_352546


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3525_352589

theorem cost_price_calculation (selling_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  selling_price = 12000 ∧ 
  discount_rate = 0.1 ∧ 
  profit_rate = 0.08 →
  (selling_price * (1 - discount_rate)) / (1 + profit_rate) = 10000 := by
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3525_352589


namespace NUMINAMATH_CALUDE_decade_cost_l3525_352529

/-- Vivian's annual car insurance cost in dollars -/
def annual_cost : ℕ := 2000

/-- Number of years in a decade -/
def decade : ℕ := 10

/-- Theorem: Vivian's total car insurance cost over a decade -/
theorem decade_cost : annual_cost * decade = 20000 := by
  sorry

end NUMINAMATH_CALUDE_decade_cost_l3525_352529


namespace NUMINAMATH_CALUDE_bakery_purchase_maximization_l3525_352542

/-- Represents the problem of maximizing purchases at a bakery --/
theorem bakery_purchase_maximization 
  (total_money : ℚ)
  (pastry_cost : ℚ)
  (coffee_cost : ℚ)
  (discount : ℚ)
  (discount_threshold : ℕ)
  (h1 : total_money = 50)
  (h2 : pastry_cost = 6)
  (h3 : coffee_cost = (3/2))
  (h4 : discount = (1/2))
  (h5 : discount_threshold = 5) :
  ∃ (pastries coffee : ℕ),
    (pastries > discount_threshold → 
      pastries * (pastry_cost - discount) + coffee * coffee_cost ≤ total_money) ∧
    (pastries ≤ discount_threshold → 
      pastries * pastry_cost + coffee * coffee_cost ≤ total_money) ∧
    pastries + coffee = 9 ∧
    ∀ (p c : ℕ), 
      ((p > discount_threshold → 
        p * (pastry_cost - discount) + c * coffee_cost ≤ total_money) ∧
      (p ≤ discount_threshold → 
        p * pastry_cost + c * coffee_cost ≤ total_money)) →
      p + c ≤ 9 := by
sorry


end NUMINAMATH_CALUDE_bakery_purchase_maximization_l3525_352542


namespace NUMINAMATH_CALUDE_parabola_vertex_l3525_352533

/-- The parabola is defined by the equation y = (x+2)^2 - 1 -/
def parabola (x y : ℝ) : Prop := y = (x + 2)^2 - 1

/-- The vertex of a parabola is the point where it reaches its maximum or minimum -/
def is_vertex (x y : ℝ) : Prop := 
  parabola x y ∧ ∀ x' y', parabola x' y' → y ≤ y'

/-- Theorem: The vertex of the parabola y = (x+2)^2 - 1 has coordinates (-2, -1) -/
theorem parabola_vertex : is_vertex (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3525_352533


namespace NUMINAMATH_CALUDE_expression_evaluation_l3525_352588

theorem expression_evaluation :
  let x : ℚ := 1/2
  let y : ℚ := -2
  ((x + 2*y)^2 - (x + y)*(x - y)) / (2*y) = -4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3525_352588


namespace NUMINAMATH_CALUDE_einstein_fundraising_goal_l3525_352506

def pizza_price : ℚ := 12
def fries_price : ℚ := 0.3
def soda_price : ℚ := 2

def pizza_sold : ℕ := 15
def fries_sold : ℕ := 40
def soda_sold : ℕ := 25

def additional_needed : ℚ := 258

def total_raised : ℚ := pizza_price * pizza_sold + fries_price * fries_sold + soda_price * soda_sold

theorem einstein_fundraising_goal :
  total_raised + additional_needed = 500 := by sorry

end NUMINAMATH_CALUDE_einstein_fundraising_goal_l3525_352506


namespace NUMINAMATH_CALUDE_circle_area_increase_l3525_352579

theorem circle_area_increase (r : ℝ) (hr : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
sorry

end NUMINAMATH_CALUDE_circle_area_increase_l3525_352579


namespace NUMINAMATH_CALUDE_count_special_numbers_eq_210_l3525_352569

/-- The number of ways to choose k elements from n elements without replacement and with order -/
def permutations (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The count of four-digit numbers with specific properties -/
def count_special_numbers : ℕ :=
  let digits := 10  -- 0 to 9
  let case1 := permutations 8 2 * permutations 2 2  -- for 0 and 8
  let case2 := permutations 7 1 * permutations 7 1 * permutations 2 2  -- for 1 and 9
  case1 + case2

theorem count_special_numbers_eq_210 :
  count_special_numbers = 210 := by sorry

end NUMINAMATH_CALUDE_count_special_numbers_eq_210_l3525_352569


namespace NUMINAMATH_CALUDE_prop_2_prop_4_l3525_352566

-- Proposition 2
theorem prop_2 (p q : Prop) : ¬(p ∨ q) → (¬p ∧ ¬q) := by sorry

-- Proposition 4
def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x + a)

theorem prop_4 (a : ℝ) : (∀ x, f a x = f a (-x)) → a = -1 := by sorry

end NUMINAMATH_CALUDE_prop_2_prop_4_l3525_352566


namespace NUMINAMATH_CALUDE_brownie_pieces_l3525_352554

theorem brownie_pieces (pan_length pan_width piece_length piece_width : ℕ) 
  (h1 : pan_length = 24)
  (h2 : pan_width = 30)
  (h3 : piece_length = 3)
  (h4 : piece_width = 4) :
  (pan_length * pan_width) / (piece_length * piece_width) = 60 := by
  sorry

#check brownie_pieces

end NUMINAMATH_CALUDE_brownie_pieces_l3525_352554


namespace NUMINAMATH_CALUDE_isosceles_triangle_with_sides_4_and_9_l3525_352574

/-- An isosceles triangle with side lengths a, b, and c, where at least two sides are equal. -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  isosceles : (a = b) ∨ (b = c) ∨ (a = c)
  triangle_inequality : a + b > c ∧ b + c > a ∧ a + c > b

/-- The theorem stating that in an isosceles triangle with two sides of lengths 4 and 9, the third side must be 9. -/
theorem isosceles_triangle_with_sides_4_and_9 :
  ∀ (t : IsoscelesTriangle), (t.a = 4 ∧ t.b = 9) ∨ (t.a = 9 ∧ t.b = 4) → t.c = 9 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_with_sides_4_and_9_l3525_352574


namespace NUMINAMATH_CALUDE_valid_quadrilateral_set_l3525_352555

/-- A function that checks if a set of four line segments can form a valid quadrilateral. -/
def is_valid_quadrilateral (a b c d : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a + b + c > d ∧ a + b + d > c ∧ a + c + d > b ∧ b + c + d > a

/-- Theorem stating that among the given sets, only (2,2,2) forms a valid quadrilateral with side length 5. -/
theorem valid_quadrilateral_set :
  ¬ is_valid_quadrilateral 1 1 1 5 ∧
  ¬ is_valid_quadrilateral 1 2 2 5 ∧
  ¬ is_valid_quadrilateral 1 1 7 5 ∧
  is_valid_quadrilateral 2 2 2 5 :=
by sorry

end NUMINAMATH_CALUDE_valid_quadrilateral_set_l3525_352555


namespace NUMINAMATH_CALUDE_complex_number_proof_l3525_352504

theorem complex_number_proof (a : ℝ) (h_a : a > 0) (z : ℂ) (h_z : z = a - Complex.I) 
  (h_real : (z + 2 / z).im = 0) :
  z = 1 - Complex.I ∧ 
  ∀ m : ℝ, (((m : ℂ) - z)^2).re < 0 ∧ (((m : ℂ) - z)^2).im > 0 ↔ 1 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_proof_l3525_352504


namespace NUMINAMATH_CALUDE_three_digit_sum_theorem_l3525_352568

def is_valid_digit_set (a b c : ℕ) : Prop :=
  a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
  (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

def sum_of_numbers (a b c : ℕ) : ℕ :=
  100 * (a + b + c) + 10 * (a + b + c) + (a + b + c)

theorem three_digit_sum_theorem :
  ∀ a b c : ℕ,
    is_valid_digit_set a b c →
    sum_of_numbers a b c = 1221 →
    ((a = 1 ∧ b = 1 ∧ c = 9) ∨
     (a = 2 ∧ b = 2 ∧ c = 7) ∨
     (a = 3 ∧ b = 3 ∧ c = 5) ∨
     (a = 4 ∧ b = 4 ∧ c = 3) ∨
     (a = 5 ∧ b = 5 ∧ c = 1)) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_sum_theorem_l3525_352568


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l3525_352594

theorem quadratic_root_difference (s t : ℝ) (hs : s > 0) (ht : t > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    x₁^2 + s*x₁ + t = 0 ∧
    x₂^2 + s*x₂ + t = 0 ∧
    |x₁ - x₂| = 2) →
  s = 2 * Real.sqrt (t + 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l3525_352594


namespace NUMINAMATH_CALUDE_missy_capacity_l3525_352550

/-- The number of insurance claims each agent can handle -/
structure AgentCapacity where
  jan : ℕ
  john : ℕ
  missy : ℕ

/-- The conditions given in the problem -/
def problem_conditions (c : AgentCapacity) : Prop :=
  c.jan = 20 ∧
  c.john = c.jan + (c.jan * 30 / 100) ∧
  c.missy = c.john + 15

/-- The theorem to prove -/
theorem missy_capacity (c : AgentCapacity) :
  problem_conditions c → c.missy = 41 := by
  sorry

end NUMINAMATH_CALUDE_missy_capacity_l3525_352550


namespace NUMINAMATH_CALUDE_stratified_sampling_high_group_l3525_352520

/-- Represents the number of students in each height group -/
structure HeightGroups where
  low : ℕ  -- [120, 130)
  mid : ℕ  -- [130, 140)
  high : ℕ -- [140, 150]

/-- Calculates the number of students to be selected from a group in stratified sampling -/
def stratifiedSample (totalPopulation : ℕ) (groupSize : ℕ) (sampleSize : ℕ) : ℕ :=
  (groupSize * sampleSize + totalPopulation - 1) / totalPopulation

/-- Proves that the number of students to be selected from the [140, 150] group is 3 -/
theorem stratified_sampling_high_group 
  (groups : HeightGroups)
  (h1 : groups.low + groups.mid + groups.high = 100)
  (h2 : groups.low = 20)
  (h3 : groups.mid = 50)
  (h4 : groups.high = 30)
  (totalSample : ℕ)
  (h5 : totalSample = 18) :
  stratifiedSample 100 groups.high totalSample = 3 := by
sorry

#eval stratifiedSample 100 30 18

end NUMINAMATH_CALUDE_stratified_sampling_high_group_l3525_352520


namespace NUMINAMATH_CALUDE_unique_a_divisibility_l3525_352532

theorem unique_a_divisibility (a : ℤ) (h1 : 0 < a) (h2 : a < 13) 
  (h3 : (13 : ℤ) ∣ (53^2017 + a)) : a = 12 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_divisibility_l3525_352532


namespace NUMINAMATH_CALUDE_football_progress_l3525_352534

/-- Calculates the overall progress in meters for a football team given their yard changes and the yard-to-meter conversion rate. -/
theorem football_progress (yard_to_meter : ℝ) (play1 play2 penalty play3 play4 : ℝ) :
  yard_to_meter = 0.9144 →
  play1 = -15 →
  play2 = 20 →
  penalty = -10 →
  play3 = 25 →
  play4 = -5 →
  (play1 + play2 + penalty + play3 + play4) * yard_to_meter = 13.716 := by
  sorry

end NUMINAMATH_CALUDE_football_progress_l3525_352534


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l3525_352580

theorem negative_fraction_comparison :
  -((4 : ℚ) / 5) < -((3 : ℚ) / 4) := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l3525_352580


namespace NUMINAMATH_CALUDE_chairs_to_remove_chair_adjustment_problem_l3525_352559

theorem chairs_to_remove (initial_chairs : ℕ) (chairs_per_row : ℕ) (expected_students : ℕ) : ℕ :=
  let min_chairs_needed := ((expected_students + chairs_per_row - 1) / chairs_per_row) * chairs_per_row
  initial_chairs - min_chairs_needed

theorem chair_adjustment_problem :
  chairs_to_remove 169 13 100 = 65 := by
  sorry

end NUMINAMATH_CALUDE_chairs_to_remove_chair_adjustment_problem_l3525_352559


namespace NUMINAMATH_CALUDE_radical_simplification_l3525_352548

theorem radical_simplification :
  Real.sqrt (4 - 2 * Real.sqrt 3) - Real.sqrt (4 + 2 * Real.sqrt 3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l3525_352548


namespace NUMINAMATH_CALUDE_prime_odd_sum_l3525_352585

theorem prime_odd_sum (a b : ℕ) : 
  Nat.Prime a → 
  Odd b → 
  a^2 + b = 2001 → 
  a + b = 1999 := by sorry

end NUMINAMATH_CALUDE_prime_odd_sum_l3525_352585


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l3525_352583

theorem last_two_digits_sum (n : ℕ) : (6^15 + 10^15) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l3525_352583


namespace NUMINAMATH_CALUDE_cos_neg_135_degrees_l3525_352514

theorem cos_neg_135_degrees :
  Real.cos ((-135 : ℝ) * π / 180) = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_neg_135_degrees_l3525_352514


namespace NUMINAMATH_CALUDE_log_equation_solution_l3525_352556

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 2 + Real.log x / Real.log 4 = 6 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3525_352556


namespace NUMINAMATH_CALUDE_folded_rectangle_EF_length_l3525_352577

/-- A rectangle ABCD with side lengths AB = 4 and BC = 8 is folded so that A and C coincide,
    forming a new shape ABEFD. This function calculates the length of EF. -/
def foldedRectangleEFLength (AB BC : ℝ) : ℝ :=
  4

/-- Theorem stating that for a rectangle ABCD with AB = 4 and BC = 8, when folded so that
    A and C coincide to form ABEFD, the length of EF is 4. -/
theorem folded_rectangle_EF_length :
  foldedRectangleEFLength 4 8 = 4 := by
  sorry

#check folded_rectangle_EF_length

end NUMINAMATH_CALUDE_folded_rectangle_EF_length_l3525_352577


namespace NUMINAMATH_CALUDE_difference_after_five_iterations_l3525_352523

def initial_sequence : List ℕ := [2, 0, 1, 9, 0]

def next_sequence (seq : List ℕ) : List ℕ :=
  let pairs := seq.zip (seq.rotateRight 1)
  pairs.map (fun (a, b) => a + b)

def iterate_sequence (seq : List ℕ) (n : ℕ) : List ℕ :=
  match n with
  | 0 => seq
  | n + 1 => iterate_sequence (next_sequence seq) n

def sum_between_zeros (seq : List ℕ) : ℕ :=
  let rotated := seq.dropWhile (· ≠ 0)
  (rotated.takeWhile (· ≠ 0)).sum

def sum_not_between_zeros (seq : List ℕ) : ℕ :=
  seq.sum - sum_between_zeros seq

theorem difference_after_five_iterations :
  let final_seq := iterate_sequence initial_sequence 5
  sum_not_between_zeros final_seq - sum_between_zeros final_seq = 1944 := by
  sorry

end NUMINAMATH_CALUDE_difference_after_five_iterations_l3525_352523


namespace NUMINAMATH_CALUDE_reading_assignment_l3525_352598

theorem reading_assignment (total_pages : ℕ) (pages_read : ℕ) (days_left : ℕ) : 
  total_pages = 408 →
  pages_read = 113 →
  days_left = 5 →
  (total_pages - pages_read) / days_left = 59 := by
  sorry

end NUMINAMATH_CALUDE_reading_assignment_l3525_352598


namespace NUMINAMATH_CALUDE_solution_pairs_l3525_352565

theorem solution_pairs (x y : ℝ) : 
  (|x + y| = 3 ∧ x * y = -10) → 
  ((x = 5 ∧ y = -2) ∨ (x = -2 ∧ y = 5) ∨ (x = 2 ∧ y = -5) ∨ (x = -5 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_l3525_352565


namespace NUMINAMATH_CALUDE_disinfectant_purchase_problem_l3525_352501

/-- The price difference between outdoor and indoor disinfectant -/
def price_difference : ℕ := 30

/-- The cost of 2 indoor and 3 outdoor disinfectant barrels -/
def sample_cost : ℕ := 340

/-- The total number of barrels to be purchased -/
def total_barrels : ℕ := 200

/-- The maximum total cost allowed -/
def max_cost : ℕ := 14000

/-- The price of indoor disinfectant -/
def indoor_price : ℕ := 50

/-- The price of outdoor disinfectant -/
def outdoor_price : ℕ := 80

/-- The minimum number of indoor disinfectant barrels to be purchased -/
def min_indoor_barrels : ℕ := 67

theorem disinfectant_purchase_problem :
  (outdoor_price = indoor_price + price_difference) ∧
  (2 * indoor_price + 3 * outdoor_price = sample_cost) ∧
  (∀ m : ℕ, m ≤ total_barrels →
    indoor_price * m + outdoor_price * (total_barrels - m) ≤ max_cost →
    m ≥ min_indoor_barrels) :=
by sorry

end NUMINAMATH_CALUDE_disinfectant_purchase_problem_l3525_352501


namespace NUMINAMATH_CALUDE_remove_all_triangles_no_triangles_remain_l3525_352591

/-- Represents a toothpick figure -/
structure ToothpickFigure where
  total_toothpicks : ℕ
  is_symmetric : Bool
  has_additional_rows : Bool

/-- Represents the number of toothpicks that must be removed to eliminate all triangles -/
def toothpicks_to_remove (figure : ToothpickFigure) : ℕ := 
  if figure.total_toothpicks = 40 ∧ figure.is_symmetric ∧ figure.has_additional_rows
  then 40
  else 0

/-- Theorem stating that for a specific toothpick figure, 40 toothpicks must be removed -/
theorem remove_all_triangles (figure : ToothpickFigure) :
  figure.total_toothpicks = 40 ∧ figure.is_symmetric ∧ figure.has_additional_rows →
  toothpicks_to_remove figure = 40 :=
by
  sorry

/-- Theorem stating that removing 40 toothpicks is sufficient to eliminate all triangles -/
theorem no_triangles_remain (figure : ToothpickFigure) :
  figure.total_toothpicks = 40 ∧ figure.is_symmetric ∧ figure.has_additional_rows →
  toothpicks_to_remove figure = 40 →
  ∀ remaining_triangles, remaining_triangles = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_remove_all_triangles_no_triangles_remain_l3525_352591


namespace NUMINAMATH_CALUDE_cookie_distribution_l3525_352584

theorem cookie_distribution (people : ℕ) (cookies_per_person : ℕ) (total_cookies : ℕ) :
  people = 4 →
  cookies_per_person = 22 →
  total_cookies = people * cookies_per_person →
  total_cookies = 88 := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_l3525_352584


namespace NUMINAMATH_CALUDE_square_difference_equality_l3525_352507

theorem square_difference_equality : 1005^2 - 995^2 - 1007^2 + 993^2 = -8000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l3525_352507


namespace NUMINAMATH_CALUDE_union_equals_reals_subset_of_complement_l3525_352576

-- Define the sets A and B
def A : Set ℝ := {x | x < 0 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 3 - 2*a}

-- Theorem for part (1)
theorem union_equals_reals (a : ℝ) : 
  A ∪ B a = Set.univ ↔ a ∈ Set.Iic 0 :=
sorry

-- Theorem for part (2)
theorem subset_of_complement (a : ℝ) :
  B a ⊆ -A ↔ a ∈ {x | x ≥ 1/2} :=
sorry

end NUMINAMATH_CALUDE_union_equals_reals_subset_of_complement_l3525_352576
