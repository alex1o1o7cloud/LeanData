import Mathlib

namespace blake_lollipops_l1064_106472

def problem (num_lollipops : ℕ) : Prop :=
  let num_chocolate_packs : ℕ := 6
  let lollipop_price : ℕ := 2
  let chocolate_pack_price : ℕ := 4 * lollipop_price
  let total_paid : ℕ := 6 * 10
  let change : ℕ := 4
  let total_spent : ℕ := total_paid - change
  let chocolate_cost : ℕ := num_chocolate_packs * chocolate_pack_price
  let lollipop_cost : ℕ := total_spent - chocolate_cost
  num_lollipops * lollipop_price = lollipop_cost

theorem blake_lollipops : ∃ (n : ℕ), problem n ∧ n = 4 := by
  sorry

end blake_lollipops_l1064_106472


namespace cos_four_minus_sin_four_equals_cos_double_l1064_106487

theorem cos_four_minus_sin_four_equals_cos_double (θ : ℝ) :
  Real.cos θ ^ 4 - Real.sin θ ^ 4 = Real.cos (2 * θ) := by
  sorry

end cos_four_minus_sin_four_equals_cos_double_l1064_106487


namespace cubic_factorization_l1064_106479

theorem cubic_factorization (m : ℝ) : m^3 - 16*m = m*(m+4)*(m-4) := by
  sorry

end cubic_factorization_l1064_106479


namespace inverse_of_proposition_l1064_106423

theorem inverse_of_proposition : 
  (∀ x : ℝ, x < 0 → x^2 > 0) → 
  (∀ x : ℝ, x^2 > 0 → x < 0) := by sorry

end inverse_of_proposition_l1064_106423


namespace max_sum_constraint_l1064_106448

theorem max_sum_constraint (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 16 * x * y * z = (x + y)^2 * (x + z)^2) :
  x + y + z ≤ 4 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
  16 * x' * y' * z' = (x' + y')^2 * (x' + z')^2 ∧ x' + y' + z' = 4 :=
by sorry

end max_sum_constraint_l1064_106448


namespace zero_sufficient_for_perpendicular_zero_not_necessary_for_perpendicular_zero_sufficient_not_necessary_for_perpendicular_l1064_106409

/-- Line l1 with equation x + ay - a = 0 -/
def line1 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + a * p.2 - a = 0}

/-- Line l2 with equation ax - (2a - 3)y - 1 = 0 -/
def line2 (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 - (2 * a - 3) * p.2 - 1 = 0}

/-- Two lines are perpendicular -/
def perpendicular (l1 l2 : Set (ℝ × ℝ)) : Prop :=
  ∃ (m1 m2 : ℝ), (∀ (p q : ℝ × ℝ), p ∈ l1 → q ∈ l1 → p ≠ q → (q.2 - p.2) = m1 * (q.1 - p.1)) ∧
                 (∀ (p q : ℝ × ℝ), p ∈ l2 → q ∈ l2 → p ≠ q → (q.2 - p.2) = m2 * (q.1 - p.1)) ∧
                 m1 * m2 = -1

/-- a=0 is a sufficient condition for perpendicularity -/
theorem zero_sufficient_for_perpendicular :
  perpendicular (line1 0) (line2 0) :=
sorry

/-- a=0 is not a necessary condition for perpendicularity -/
theorem zero_not_necessary_for_perpendicular :
  ∃ a : ℝ, a ≠ 0 ∧ perpendicular (line1 a) (line2 a) :=
sorry

/-- Main theorem: a=0 is sufficient but not necessary for perpendicularity -/
theorem zero_sufficient_not_necessary_for_perpendicular :
  (perpendicular (line1 0) (line2 0)) ∧
  (∃ a : ℝ, a ≠ 0 ∧ perpendicular (line1 a) (line2 a)) :=
sorry

end zero_sufficient_for_perpendicular_zero_not_necessary_for_perpendicular_zero_sufficient_not_necessary_for_perpendicular_l1064_106409


namespace complex_fraction_simplification_l1064_106405

theorem complex_fraction_simplification :
  (7 + 8 * Complex.I) / (3 - 4 * Complex.I) = -11/25 + 52/25 * Complex.I :=
by sorry

end complex_fraction_simplification_l1064_106405


namespace ellipse_chord_slope_l1064_106464

/-- The slope of a chord in an ellipse with midpoint (-2, 1) -/
theorem ellipse_chord_slope :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  (x₁^2 / 16 + y₁^2 / 9 = 1) →
  (x₂^2 / 16 + y₂^2 / 9 = 1) →
  (x₁ + x₂ = -4) →
  (y₁ + y₂ = 2) →
  ((y₂ - y₁) / (x₂ - x₁) = 9 / 8) :=
by sorry

end ellipse_chord_slope_l1064_106464


namespace characterization_of_special_numbers_l1064_106435

def is_power_of_two (n : ℕ+) : Prop :=
  ∃ k : ℕ, n = 2^k

def greatest_odd_divisor (n : ℕ+) : ℕ+ :=
  sorry

def smallest_odd_divisor (n : ℕ+) : ℕ+ :=
  sorry

def is_odd_prime (p : ℕ+) : Prop :=
  Nat.Prime p.val ∧ p.val % 2 = 1

theorem characterization_of_special_numbers (n : ℕ+) :
  ¬is_power_of_two n →
  (n = 3 * greatest_odd_divisor n + 5 * smallest_odd_divisor n ↔
    (∃ p : ℕ+, is_odd_prime p ∧ n = 8 * p) ∨ n = 60 ∨ n = 100) :=
  sorry

end characterization_of_special_numbers_l1064_106435


namespace quadratic_root_range_l1064_106428

/-- Represents a quadratic equation ax^2 + (a+2)x + 9a = 0 -/
def quadratic_equation (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 9 * a

theorem quadratic_root_range :
  ∀ a : ℝ,
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ < 1 ∧ 1 < x₂ ∧
    quadratic_equation a x₁ = 0 ∧ quadratic_equation a x₂ = 0) →
  -2/11 < a ∧ a < 0 :=
by sorry

end quadratic_root_range_l1064_106428


namespace numera_transaction_l1064_106440

/-- Represents a number in base s --/
def BaseS (digits : List Nat) (s : Nat) : Nat :=
  digits.foldr (fun d acc => d + s * acc) 0

/-- The transaction in the galaxy of Numera --/
theorem numera_transaction (s : Nat) : 
  s > 1 →  -- s must be greater than 1 to be a valid base
  BaseS [6, 3, 0] s + BaseS [2, 5, 0] s = BaseS [8, 8, 0] s →  -- cost of gadgets
  BaseS [4, 7, 0] s = BaseS [1, 0, 0, 0] s * 2 - BaseS [8, 8, 0] s →  -- change received
  s = 5 := by
  sorry

end numera_transaction_l1064_106440


namespace hex_to_binary_bits_l1064_106439

/-- The hexadecimal number A3F52 -/
def hex_number : ℕ := 0xA3F52

/-- The number of bits in the binary representation of a natural number -/
def num_bits (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.log2 n + 1

theorem hex_to_binary_bits :
  num_bits hex_number = 20 := by sorry

end hex_to_binary_bits_l1064_106439


namespace pool_width_is_twelve_l1064_106493

/-- Represents a rectangular swimming pool with a surrounding deck -/
structure PoolWithDeck where
  poolLength : ℝ
  poolWidth : ℝ
  deckWidth : ℝ
  totalArea : ℝ

/-- Theorem stating the width of the swimming pool given specific conditions -/
theorem pool_width_is_twelve (p : PoolWithDeck)
  (h1 : p.poolLength = 10)
  (h2 : p.deckWidth = 4)
  (h3 : p.totalArea = 360)
  (h4 : (p.poolLength + 2 * p.deckWidth) * (p.poolWidth + 2 * p.deckWidth) = p.totalArea) :
  p.poolWidth = 12 := by
  sorry

end pool_width_is_twelve_l1064_106493


namespace fraction_equality_l1064_106445

theorem fraction_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a / 2 = b / 3) :
  3 / b = 2 / a := by
  sorry

end fraction_equality_l1064_106445


namespace number_division_l1064_106473

theorem number_division (x : ℝ) (h : 5 * x = 100) : x / 10 = 2 := by
  sorry

end number_division_l1064_106473


namespace cube_divisors_count_l1064_106465

-- Define a natural number with exactly two prime divisors
def has_two_prime_divisors (n : ℕ) : Prop :=
  ∃ p q α β : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ n = p^α * q^β

-- Define the number of divisors function
noncomputable def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

-- State the theorem
theorem cube_divisors_count
  (n : ℕ)
  (h1 : has_two_prime_divisors n)
  (h2 : num_divisors (n^2) = 35) :
  num_divisors (n^3) = 70 := by
  sorry

end cube_divisors_count_l1064_106465


namespace purely_imaginary_condition_l1064_106447

theorem purely_imaginary_condition (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (∃ (b : ℝ), (1 - 2*Complex.I)*(a + Complex.I) = b*Complex.I) ↔ a = -2 :=
sorry

end purely_imaginary_condition_l1064_106447


namespace correct_average_calculation_l1064_106454

theorem correct_average_calculation (n : ℕ) (incorrect_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 ∧ incorrect_avg = 20 ∧ incorrect_num = 26 ∧ correct_num = 86 →
  (n : ℚ) * incorrect_avg + (correct_num - incorrect_num) = n * 26 := by
  sorry

end correct_average_calculation_l1064_106454


namespace not_p_false_sufficient_not_necessary_for_p_or_q_true_l1064_106425

theorem not_p_false_sufficient_not_necessary_for_p_or_q_true (p q : Prop) :
  (¬¬p → p ∨ q) ∧ ∃ (p q : Prop), (p ∨ q) ∧ ¬(¬¬p) := by sorry

end not_p_false_sufficient_not_necessary_for_p_or_q_true_l1064_106425


namespace parabola_coefficient_l1064_106443

/-- Proves that for a parabola y = ax^2 + bx + c with vertex at (q,q) and y-intercept at (0, -2q), 
    where q ≠ 0, the coefficient b equals 6/q. -/
theorem parabola_coefficient (a b c q : ℝ) (h1 : q ≠ 0) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (q, q) = ((-b) / (2 * a), a * ((-b) / (2 * a))^2 + b * ((-b) / (2 * a)) + c) →
  -2 * q = c →
  b = 6 / q :=
sorry

end parabola_coefficient_l1064_106443


namespace time_to_change_tires_l1064_106458

def minutes_to_wash_car : ℕ := 10
def minutes_to_change_oil : ℕ := 15
def cars_washed : ℕ := 9
def cars_oil_changed : ℕ := 6
def sets_of_tires_changed : ℕ := 2
def hours_worked : ℕ := 4

theorem time_to_change_tires :
  let total_minutes : ℕ := hours_worked * 60
  let washing_time : ℕ := cars_washed * minutes_to_wash_car
  let oil_change_time : ℕ := cars_oil_changed * minutes_to_change_oil
  let remaining_time : ℕ := total_minutes - (washing_time + oil_change_time)
  remaining_time / sets_of_tires_changed = 30 := by sorry

end time_to_change_tires_l1064_106458


namespace rationalize_denominator_l1064_106419

theorem rationalize_denominator :
  (1 : ℝ) / (Real.rpow 3 (1/3) + Real.rpow 27 (1/3)) = Real.rpow 9 (1/3) / 12 := by
  sorry

end rationalize_denominator_l1064_106419


namespace factor_expression_l1064_106470

theorem factor_expression (x : ℝ) : 12 * x^3 + 6 * x^2 = 6 * x^2 * (2 * x + 1) := by
  sorry

end factor_expression_l1064_106470


namespace seven_digit_divisible_by_11_l1064_106491

theorem seven_digit_divisible_by_11 : ∃ (a g : ℕ), ∃ (b c d e : ℕ),
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ g ∧ g ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  0 ≤ e ∧ e ≤ 9 ∧
  b + c + d + e = 18 ∧
  (a * 1000000 + b * 100000 + c * 10000 + d * 1000 + e * 100 + 7 * 10 + g) % 11 = 0 :=
by sorry

end seven_digit_divisible_by_11_l1064_106491


namespace only_B_is_random_event_l1064_106437

-- Define the type for a die roll
def DieRoll := Fin 6

-- Define the type for a pair of die rolls
def TwoDiceRoll := DieRoll × DieRoll

-- Define the sum of two dice
def diceSum (roll : TwoDiceRoll) : Nat := roll.1.val + roll.2.val + 2

-- Define the sample space
def Ω : Set TwoDiceRoll := Set.univ

-- Define the events
def A : Set TwoDiceRoll := {roll | diceSum roll = 1}
def B : Set TwoDiceRoll := {roll | diceSum roll = 6}
def C : Set TwoDiceRoll := {roll | diceSum roll > 12}
def D : Set TwoDiceRoll := {roll | diceSum roll < 13}

-- Theorem statement
theorem only_B_is_random_event :
  (A = ∅ ∧ B ≠ ∅ ∧ B ≠ Ω ∧ C = ∅ ∧ D = Ω) := by sorry

end only_B_is_random_event_l1064_106437


namespace equation_solution_l1064_106462

theorem equation_solution (m n k x : ℝ) 
  (hm : m ≠ 0) (hn : n ≠ 0) (hk : k ≠ 0) (hmn : m ≠ n) :
  (x + m)^2 - (x + n)^2 = k * (m - n)^2 → 
  x = ((k - 1) * (m + n) - 2 * k * n) / 2 := by
sorry

end equation_solution_l1064_106462


namespace triangle_angle_measure_l1064_106404

theorem triangle_angle_measure (a b c A B C : Real) : 
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  -- Given conditions
  (a = Real.sqrt 2) →
  (b = 2) →
  (B = π / 4) →
  -- Conclusion
  (A = π / 6) := by
sorry

end triangle_angle_measure_l1064_106404


namespace smallest_K_for_divisibility_l1064_106498

def repeatedDigit (d : ℕ) (K : ℕ) : ℕ :=
  d * (10^K - 1) / 9

theorem smallest_K_for_divisibility (K : ℕ) : 
  (∀ n : ℕ, n < K → ¬(198 ∣ repeatedDigit 2 n)) ∧ 
  (198 ∣ repeatedDigit 2 K) → 
  K = 18 := by
  sorry

end smallest_K_for_divisibility_l1064_106498


namespace bernardo_wins_l1064_106459

theorem bernardo_wins (N : ℕ) : N ≤ 999 ∧ 72 * N < 1000 ∧ 36 * N < 1000 ∧ ∀ m : ℕ, m < N → (72 * m ≥ 1000 ∨ 36 * m ≥ 1000) → N = 13 := by
  sorry

end bernardo_wins_l1064_106459


namespace distance_difference_l1064_106486

/-- Given distances between locations, prove the difference in total distances -/
theorem distance_difference (orchard_to_house house_to_pharmacy pharmacy_to_school : ℕ) 
  (h1 : orchard_to_house = 800)
  (h2 : house_to_pharmacy = 1300)
  (h3 : pharmacy_to_school = 1700) :
  (orchard_to_house + house_to_pharmacy) - pharmacy_to_school = 400 := by
  sorry

end distance_difference_l1064_106486


namespace decorations_used_l1064_106468

theorem decorations_used (boxes : Nat) (decorations_per_box : Nat) (given_away : Nat) : 
  boxes = 4 → decorations_per_box = 15 → given_away = 25 →
  boxes * decorations_per_box - given_away = 35 := by
  sorry

end decorations_used_l1064_106468


namespace sum_of_solutions_l1064_106477

theorem sum_of_solutions (a : ℝ) (h : a > 2) : 
  ∃ x₁ x₂ : ℝ, (Real.sqrt (a - Real.sqrt (a + x₁)) = x₁ + 1) ∧ 
              (Real.sqrt (a - Real.sqrt (a + x₂)) = x₂ + 1) ∧ 
              (x₁ + x₂ = -2) := by
  sorry

end sum_of_solutions_l1064_106477


namespace simplify_expression_l1064_106499

theorem simplify_expression (a : ℝ) (ha : a > 0) :
  a^2 / (Real.sqrt a * 3 * a^2) = 1 / Real.sqrt a :=
by sorry

end simplify_expression_l1064_106499


namespace complex_multiplication_l1064_106478

theorem complex_multiplication :
  let i : ℂ := Complex.I
  (1 - 2*i) * (2 + i) = 4 - 3*i := by sorry

end complex_multiplication_l1064_106478


namespace angle_half_in_third_quadrant_l1064_106474

open Real

-- Define the first quadrant
def FirstQuadrant (α : ℝ) : Prop := 0 < α ∧ α < π / 2

-- Define the third quadrant
def ThirdQuadrant (θ : ℝ) : Prop := π < θ ∧ θ < 3 * π / 2

-- State the theorem
theorem angle_half_in_third_quadrant (α : ℝ) 
  (h1 : FirstQuadrant α) 
  (h2 : |cos (α / 2)| = -cos (α / 2)) : 
  ThirdQuadrant (α / 2) := by
  sorry

end angle_half_in_third_quadrant_l1064_106474


namespace inverse_proportion_example_l1064_106429

/-- Two real numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_proportion_example :
  ∀ x y : ℝ → ℝ,
  InverselyProportional x y →
  x 5 = 40 →
  y 5 = 5 →
  y 20 = 20 →
  x 20 = 10 :=
by
  sorry

end inverse_proportion_example_l1064_106429


namespace last_two_digits_sum_l1064_106451

theorem last_two_digits_sum (n : ℕ) : n = 25 → (15^n + 5^n) % 100 = 0 := by
  sorry

end last_two_digits_sum_l1064_106451


namespace initial_average_calculation_l1064_106450

theorem initial_average_calculation (n : ℕ) (correct_sum incorrect_sum : ℝ) 
  (h1 : n = 10)
  (h2 : correct_sum / n = 18)
  (h3 : incorrect_sum = correct_sum - 46 + 26) :
  incorrect_sum / n = 16 := by
  sorry

end initial_average_calculation_l1064_106450


namespace mitzi_food_expense_l1064_106421

/-- Proves that the amount spent on food is $13 given the conditions of Mitzi's amusement park expenses --/
theorem mitzi_food_expense (
  total_brought : ℕ)
  (ticket_cost : ℕ)
  (tshirt_cost : ℕ)
  (money_left : ℕ)
  (h1 : total_brought = 75)
  (h2 : ticket_cost = 30)
  (h3 : tshirt_cost = 23)
  (h4 : money_left = 9)
  : total_brought - money_left - (ticket_cost + tshirt_cost) = 13 := by
  sorry

end mitzi_food_expense_l1064_106421


namespace percentage_changes_l1064_106430

/-- Given an initial value of 950, prove that increasing it by 80% and then
    decreasing the result by 65% yields 598.5. -/
theorem percentage_changes (initial : ℝ) (increase_percent : ℝ) (decrease_percent : ℝ) :
  initial = 950 →
  increase_percent = 80 →
  decrease_percent = 65 →
  (initial * (1 + increase_percent / 100)) * (1 - decrease_percent / 100) = 598.5 := by
  sorry

end percentage_changes_l1064_106430


namespace incorrect_exponent_operation_l1064_106436

theorem incorrect_exponent_operation (a : ℝ) (h : a ≠ 0 ∧ a ≠ 1) : (a^2)^3 ≠ a^5 := by
  sorry

end incorrect_exponent_operation_l1064_106436


namespace mikes_stamp_collection_last_page_l1064_106412

/-- Represents the stamp collection problem --/
structure StampCollection where
  initial_books : ℕ
  pages_per_book : ℕ
  initial_stamps_per_page : ℕ
  new_stamps_per_page : ℕ
  filled_books : ℕ
  filled_pages_in_last_book : ℕ

/-- Calculates the number of stamps on the last page after reorganization --/
def stamps_on_last_page (sc : StampCollection) : ℕ :=
  let total_stamps := sc.initial_books * sc.pages_per_book * sc.initial_stamps_per_page
  let filled_pages := sc.filled_books * sc.pages_per_book + sc.filled_pages_in_last_book
  let stamps_on_filled_pages := filled_pages * sc.new_stamps_per_page
  total_stamps - stamps_on_filled_pages

/-- Theorem stating that for Mike's stamp collection, the last page contains 9 stamps --/
theorem mikes_stamp_collection_last_page :
  let sc : StampCollection := {
    initial_books := 6,
    pages_per_book := 30,
    initial_stamps_per_page := 7,
    new_stamps_per_page := 9,
    filled_books := 3,
    filled_pages_in_last_book := 26
  }
  stamps_on_last_page sc = 9 := by
  sorry


end mikes_stamp_collection_last_page_l1064_106412


namespace tangent_condition_orthogonal_intersection_condition_l1064_106415

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0

-- Define the line equation
def line_eq (m x y : ℝ) : Prop := x + m*y = 3

-- Define the tangency condition
def is_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq m x y ∧
  ∀ (x' y' : ℝ), circle_eq x' y' ∧ line_eq m x' y' → (x', y') = (x, y)

-- Define the intersection condition
def intersects_at_orthogonal_points (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧
    line_eq m x₁ y₁ ∧ line_eq m x₂ y₂ ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧
    x₁ * x₂ + y₁ * y₂ = 0

-- Theorem statements
theorem tangent_condition :
  ∀ m : ℝ, is_tangent m ↔ m = 7/24 :=
sorry

theorem orthogonal_intersection_condition :
  ∀ m : ℝ, intersects_at_orthogonal_points m ↔ (m = 9 + 2*Real.sqrt 14 ∨ m = 9 - 2*Real.sqrt 14) :=
sorry

end tangent_condition_orthogonal_intersection_condition_l1064_106415


namespace exists_grid_with_more_than_20_components_l1064_106400

/-- Represents a diagonal in a cell --/
inductive Diagonal
| TopLeft
| TopRight

/-- Represents the grid --/
def Grid := Matrix (Fin 8) (Fin 8) Diagonal

/-- A function that counts the number of connected components in a grid --/
def countComponents (g : Grid) : ℕ := sorry

/-- Theorem stating that there exists a grid configuration with more than 20 components --/
theorem exists_grid_with_more_than_20_components :
  ∃ (g : Grid), countComponents g > 20 :=
sorry

end exists_grid_with_more_than_20_components_l1064_106400


namespace complex_magnitude_l1064_106480

theorem complex_magnitude (s : ℝ) (w : ℂ) (h1 : |s| < 4) (h2 : w + 4 / w = s) : Complex.abs w = 2 := by
  sorry

end complex_magnitude_l1064_106480


namespace ellipse_constant_product_l1064_106457

def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

def focus (x y : ℝ) : Prop := x = -1 ∧ y = 0

def min_distance (d : ℝ) : Prop := d = Real.sqrt 2 - 1

def point_M (x y : ℝ) : Prop := x = -5/4 ∧ y = 0

def line_intersects_ellipse (l : ℝ → ℝ → Prop) (a b : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ l x₁ y₁ ∧ l x₂ y₂ ∧ ellipse a b x₁ y₁ ∧ ellipse a b x₂ y₂

def product_MA_MB (xₐ yₐ xₘ yₘ xb yb : ℝ) : ℝ :=
  ((xₐ - xₘ)^2 + (yₐ - yₘ)^2) * ((xb - xₘ)^2 + (yb - yₘ)^2)

theorem ellipse_constant_product (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  ∀ l : ℝ → ℝ → Prop,
    (∃ x y, focus x y) →
    (∃ d, min_distance d) →
    (∃ xₘ yₘ, point_M xₘ yₘ) →
    line_intersects_ellipse l a b →
    (∃ xₐ yₐ xb yb xₘ yₘ,
      l xₐ yₐ ∧ l xb yb ∧ point_M xₘ yₘ ∧
      product_MA_MB xₐ yₐ xₘ yₘ xb yb = -7/16) :=
by sorry

end ellipse_constant_product_l1064_106457


namespace factorial_fraction_simplification_l1064_106420

theorem factorial_fraction_simplification : 
  (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 8 / 7 := by
  sorry

end factorial_fraction_simplification_l1064_106420


namespace sum_of_products_bound_l1064_106461

theorem sum_of_products_bound (a b c : ℝ) (h : a + b + c = 1) :
  0 ≤ a * b + a * c + b * c ∧ a * b + a * c + b * c ≤ 1/3 := by
sorry

end sum_of_products_bound_l1064_106461


namespace shooting_probabilities_l1064_106442

-- Define probabilities
def prob_A_hit : ℚ := 2/3
def prob_B_hit : ℚ := 3/4

-- Define number of shots
def num_shots : ℕ := 4

-- Define the probability of A missing at least once in 4 shots
def prob_A_miss_at_least_once : ℚ := 1 - prob_A_hit^num_shots

-- Define the probability of A hitting exactly 2 times in 4 shots
def prob_A_hit_exactly_two : ℚ := 
  (num_shots.choose 2 : ℚ) * prob_A_hit^2 * (1 - prob_A_hit)^(num_shots - 2)

-- Define the probability of B hitting exactly 3 times in 4 shots
def prob_B_hit_exactly_three : ℚ :=
  (num_shots.choose 3 : ℚ) * prob_B_hit^3 * (1 - prob_B_hit)^(num_shots - 3)

-- Define the probability of B stopping after exactly 5 shots
def prob_B_stop_after_five : ℚ := 
  prob_B_hit^2 * (1 - prob_B_hit) * (1 - prob_B_hit^2)

theorem shooting_probabilities :
  prob_A_miss_at_least_once = 65/81 ∧
  prob_A_hit_exactly_two * prob_B_hit_exactly_three = 1/8 ∧
  prob_B_stop_after_five = 45/1024 := by sorry

end shooting_probabilities_l1064_106442


namespace vector_sum_proof_l1064_106456

/-- Given two vectors a and b in ℝ², prove that their sum is (-1, 5) -/
theorem vector_sum_proof (a b : ℝ × ℝ) (ha : a = (2, 1)) (hb : b = (-3, 4)) :
  a + b = (-1, 5) := by
  sorry

end vector_sum_proof_l1064_106456


namespace parabola_hyperbola_intersection_l1064_106422

/-- Parabola C₁ with focus F and equation y² = 2px (p > 0) -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  eq : (x y : ℝ) → Prop

/-- Hyperbola C₂ with equation y²/4 - x²/3 = 1 -/
structure Hyperbola where
  eq : (x y : ℝ) → Prop

/-- Two points A and B in the first quadrant -/
structure IntersectionPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  first_quadrant : Prop

/-- Area of triangle FAB -/
def triangleArea (F A B : ℝ × ℝ) : ℝ := sorry

/-- Dot product of vectors FA and FB -/
def dotProduct (F A B : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem parabola_hyperbola_intersection
  (C₁ : Parabola)
  (C₂ : Hyperbola)
  (points : IntersectionPoints)
  (h₁ : C₁.p > 0)
  (h₂ : C₁.eq = fun x y ↦ y^2 = 2 * C₁.p * x)
  (h₃ : C₂.eq = fun x y ↦ y^2 / 4 - x^2 / 3 = 1)
  (h₄ : C₁.focus = (C₁.p / 2, 0))
  (h₅ : triangleArea C₁.focus points.A points.B = 2/3 * dotProduct C₁.focus points.A points.B) :
  C₁.p = 2 * Real.sqrt 3 := by
  sorry

end parabola_hyperbola_intersection_l1064_106422


namespace possible_values_of_e_l1064_106444

theorem possible_values_of_e :
  ∀ e : ℝ, |2 - e| = 5 → (e = 7 ∨ e = -3) :=
sorry

end possible_values_of_e_l1064_106444


namespace dog_reach_area_l1064_106433

/-- The area outside a regular hexagonal doghouse that a dog can reach when tethered to a vertex --/
theorem dog_reach_area (side_length : ℝ) (rope_length : ℝ) : 
  side_length = 2 → rope_length = 3 → 
  (area_outside_doghouse : ℝ) = (22 / 3) * Real.pi := by
  sorry

end dog_reach_area_l1064_106433


namespace mr_green_potato_yield_l1064_106492

/-- Calculates the expected potato yield from a rectangular garden --/
def expected_potato_yield (length_steps : ℕ) (width_steps : ℕ) (step_length : ℝ) 
  (usable_percentage : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  let length_feet := length_steps * step_length
  let width_feet := width_steps * step_length
  let total_area := length_feet * width_feet
  let usable_area := total_area * usable_percentage
  usable_area * yield_per_sqft

/-- Theorem stating the expected potato yield for Mr. Green's garden --/
theorem mr_green_potato_yield :
  expected_potato_yield 18 25 3 0.9 0.5 = 1822.5 := by
  sorry

end mr_green_potato_yield_l1064_106492


namespace seven_couples_handshakes_l1064_106449

/-- Represents a gathering of couples -/
structure Gathering where
  couples : ℕ
  deriving Repr

/-- Calculates the number of handshakes in a gathering under specific conditions -/
def handshakes (g : Gathering) : ℕ :=
  let total_people := 2 * g.couples
  let handshakes_per_person := total_people - 3  -- Excluding self, spouse, and one other
  (total_people * handshakes_per_person) / 2 - g.couples

/-- Theorem stating that in a gathering of 7 couples, 
    with the given handshake conditions, there are 77 handshakes -/
theorem seven_couples_handshakes :
  handshakes { couples := 7 } = 77 := by
  sorry

#eval handshakes { couples := 7 }

end seven_couples_handshakes_l1064_106449


namespace probability_divisible_by_20_l1064_106490

/-- The set of digits used to form the six-digit number -/
def digits : Finset Nat := {1, 2, 3, 4, 5, 8}

/-- The total number of possible six-digit arrangements -/
def total_arrangements : Nat := 720

/-- Predicate to check if a number is divisible by 20 -/
def is_divisible_by_20 (n : Nat) : Prop := n % 20 = 0

/-- The number of arrangements divisible by 20 -/
def arrangements_divisible_by_20 : Nat := 576

theorem probability_divisible_by_20 :
  (arrangements_divisible_by_20 : ℚ) / total_arrangements = 4 / 5 := by
  sorry

end probability_divisible_by_20_l1064_106490


namespace scientific_notation_87000000_l1064_106497

theorem scientific_notation_87000000 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 87000000 = a * (10 : ℝ) ^ n ∧ a = 8.7 ∧ n = 7 :=
by sorry

end scientific_notation_87000000_l1064_106497


namespace tournament_divisibility_l1064_106466

theorem tournament_divisibility (n : ℕ) 
  (h1 : ∃ (m : ℕ), (n * (n - 1) / 2 + 2 * n^2 - m) = 5 / 4 * (2 * n * (2 * n - 1) + m)) : 
  9 ∣ (3 * n) := by
sorry

end tournament_divisibility_l1064_106466


namespace vector_subtraction_l1064_106434

/-- Given two plane vectors a and b, prove that a - 2b equals the expected result -/
theorem vector_subtraction (a b : ℝ × ℝ) : 
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) := by sorry

end vector_subtraction_l1064_106434


namespace tv_cost_l1064_106460

def lindas_savings : ℚ := 960

theorem tv_cost (furniture_fraction : ℚ) (h1 : furniture_fraction = 3 / 4) :
  (1 - furniture_fraction) * lindas_savings = 240 := by
  sorry

end tv_cost_l1064_106460


namespace geometric_sequence_property_l1064_106463

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a) 
  (h_a4 : a 4 = 5) : 
  a 3 * a 5 = 25 := by
sorry

end geometric_sequence_property_l1064_106463


namespace max_xy_min_ratio_l1064_106438

theorem max_xy_min_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 4) : 
  (∀ a b, a > 0 → b > 0 → a + 2*b = 4 → a*b ≤ x*y) ∧ 
  (∀ a b, a > 0 → b > 0 → a + 2*b = 4 → y/x + 4/y ≤ a/b + 4/a) :=
sorry

end max_xy_min_ratio_l1064_106438


namespace greatest_integer_less_than_negative_fraction_l1064_106426

theorem greatest_integer_less_than_negative_fraction :
  ⌊-22/5⌋ = -5 := by sorry

end greatest_integer_less_than_negative_fraction_l1064_106426


namespace negation_of_universal_proposition_l1064_106410

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > 0 → x * Real.exp x > 0)) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ * Real.exp x₀ ≤ 0) := by
  sorry

end negation_of_universal_proposition_l1064_106410


namespace quadratic_solution_sum_l1064_106469

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 3 * x^2 - 5 * x + 20 = 0 ↔ x = a + b * I ∨ x = a - b * I) → 
  a + b^2 = 245/36 := by
  sorry

end quadratic_solution_sum_l1064_106469


namespace circle_C_properties_l1064_106484

/-- Definition of the circle C -/
def circle_C (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - b)^2 = 25}

/-- Theorem stating the properties of circle C and its tangent lines -/
theorem circle_C_properties :
  ∃ (a b : ℝ),
    (a + b + 1 = 0) ∧
    ((-2 - a)^2 + (0 - b)^2 = 25) ∧
    ((5 - a)^2 + (1 - b)^2 = 25) ∧
    (circle_C a b = {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 3)^2 = 25}) ∧
    (∀ (x y : ℝ), x = -3 → (x, y) ∈ circle_C a b → y = 0 ∨ y ≠ 0) ∧
    (∀ (x y : ℝ), y = (8/15) * (x + 3) → (x, y) ∈ circle_C a b → x = -3 ∨ x ≠ -3) :=
by
  sorry


end circle_C_properties_l1064_106484


namespace remainder_mod_five_l1064_106476

theorem remainder_mod_five : (9^6 + 8^8 + 7^9) % 5 = 4 := by
  sorry

end remainder_mod_five_l1064_106476


namespace bertolli_farm_corn_count_l1064_106482

theorem bertolli_farm_corn_count :
  ∀ (tomatoes onions corn : ℕ),
    tomatoes = 2073 →
    onions = 985 →
    tomatoes + corn - onions = 5200 →
    corn = 4039 :=
by
  sorry

end bertolli_farm_corn_count_l1064_106482


namespace prime_square_plus_2007p_minus_one_prime_l1064_106413

theorem prime_square_plus_2007p_minus_one_prime (p : ℕ) : 
  Prime p ∧ Prime (p^2 + 2007*p - 1) ↔ p = 3 := by
  sorry

end prime_square_plus_2007p_minus_one_prime_l1064_106413


namespace rental_cost_equality_l1064_106407

/-- The daily rate for Safety Rent-a-Car in dollars -/
def safety_daily_rate : ℝ := 21.95

/-- The per-mile rate for Safety Rent-a-Car in dollars -/
def safety_mile_rate : ℝ := 0.19

/-- The daily rate for City Rentals in dollars -/
def city_daily_rate : ℝ := 18.95

/-- The per-mile rate for City Rentals in dollars -/
def city_mile_rate : ℝ := 0.21

/-- The mileage at which the cost is the same for both companies -/
def equal_cost_mileage : ℝ := 150

theorem rental_cost_equality :
  safety_daily_rate + safety_mile_rate * equal_cost_mileage =
  city_daily_rate + city_mile_rate * equal_cost_mileage := by
  sorry

#check rental_cost_equality

end rental_cost_equality_l1064_106407


namespace pages_left_to_read_l1064_106455

theorem pages_left_to_read (total_pages read_pages : ℕ) 
  (h1 : total_pages = 563)
  (h2 : read_pages = 147) :
  total_pages - read_pages = 416 := by
sorry

end pages_left_to_read_l1064_106455


namespace gain_percent_calculation_l1064_106471

def cost_price : ℝ := 900
def selling_price : ℝ := 1170

theorem gain_percent_calculation : 
  (selling_price - cost_price) / cost_price * 100 = 30 := by
  sorry

end gain_percent_calculation_l1064_106471


namespace odd_function_property_l1064_106488

-- Define the function f on the interval [-1, 1]
def f : ℝ → ℝ := sorry

-- Define the property of being an odd function
def isOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_property :
  (∀ x ∈ Set.Icc (-1) 1, f x = f x) →  -- f is defined on [-1, 1]
  isOdd f →  -- f is an odd function
  (∀ x ∈ Set.Ioo 0 1, f x = x * (x - 1)) →  -- f(x) = x(x-1) for 0 < x ≤ 1
  (∀ x ∈ Set.Ioc (-1) 0, f x = -x^2 - x) :=  -- f(x) = -x^2 - x for -1 ≤ x < 0
by sorry

end odd_function_property_l1064_106488


namespace shape_e_not_in_square_pieces_l1064_106453

/-- Represents a shape in the diagram -/
structure Shape :=
  (id : String)

/-- Represents the set of shapes in the divided square -/
def SquarePieces : Finset Shape := sorry

/-- Represents the set of given shapes to check -/
def GivenShapes : Finset Shape := sorry

/-- Shape E is defined separately for the theorem -/
def ShapeE : Shape := { id := "E" }

theorem shape_e_not_in_square_pieces :
  ShapeE ∉ SquarePieces ∧
  ∀ s ∈ GivenShapes, s ≠ ShapeE → s ∈ SquarePieces :=
sorry

end shape_e_not_in_square_pieces_l1064_106453


namespace problem_solution_l1064_106401

theorem problem_solution (a b : ℝ) 
  (h1 : 5 + a = 6 - b) 
  (h2 : 3 + b = 8 + a) : 
  5 - a = 7 := by
sorry

end problem_solution_l1064_106401


namespace arithmetic_evaluation_l1064_106416

theorem arithmetic_evaluation : (7 - 6 * (-5)) - 4 * (-3) / (-2) = 31 := by
  sorry

end arithmetic_evaluation_l1064_106416


namespace existence_of_special_integer_l1064_106408

theorem existence_of_special_integer :
  ∃ (n : ℕ), n ≥ 2^2018 ∧
  ∀ (x y u v : ℕ), u > 1 → v > 1 → n ≠ x^u + y^v :=
by sorry

end existence_of_special_integer_l1064_106408


namespace absolute_value_plus_tan_sixty_degrees_l1064_106424

theorem absolute_value_plus_tan_sixty_degrees : 
  |(-2 + Real.sqrt 3)| + Real.tan (π / 3) = 2 := by
  sorry

end absolute_value_plus_tan_sixty_degrees_l1064_106424


namespace intersection_of_A_and_B_l1064_106427

def A : Set Char := {'a', 'b', 'c', 'd'}
def B : Set Char := {'b', 'c', 'd', 'e'}

theorem intersection_of_A_and_B :
  A ∩ B = {'b', 'c', 'd'} := by sorry

end intersection_of_A_and_B_l1064_106427


namespace expression_simplification_l1064_106414

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 3 + 1) :
  (1 - 1/m) / ((m^2 - 2*m + 1) / m) = Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_l1064_106414


namespace parallel_line_slope_l1064_106418

/-- The slope of a line parallel to 5x - 3y = 12 is 5/3 -/
theorem parallel_line_slope : 
  ∀ (m : ℚ), (∃ b : ℚ, ∀ x y : ℚ, 5 * x - 3 * y = 12 ↔ y = m * x + b) → m = 5 / 3 := by
  sorry

end parallel_line_slope_l1064_106418


namespace relationship_abc_l1064_106496

theorem relationship_abc (a b c : ℚ) : 
  (2 * a + a = 1) → (2 * b + b = 2) → (3 * c + c = 2) → a < c ∧ c < b := by
  sorry

end relationship_abc_l1064_106496


namespace equal_salary_at_5000_sales_l1064_106432

/-- Represents the monthly salary options for Juliet --/
structure SalaryOptions where
  flat_salary : ℝ
  base_salary : ℝ
  commission_rate : ℝ

/-- Calculates the total salary for the commission-based option --/
def commission_salary (options : SalaryOptions) (sales : ℝ) : ℝ :=
  options.base_salary + options.commission_rate * sales

/-- The specific salary options given in the problem --/
def juliet_options : SalaryOptions :=
  { flat_salary := 1800
    base_salary := 1600
    commission_rate := 0.04 }

/-- Theorem stating that the sales amount for equal salaries is $5000 --/
theorem equal_salary_at_5000_sales (options : SalaryOptions := juliet_options) :
  ∃ (sales : ℝ), sales = 5000 ∧ options.flat_salary = commission_salary options sales :=
by
  sorry

end equal_salary_at_5000_sales_l1064_106432


namespace geometric_sequence_properties_l1064_106481

/-- A geometric sequence with given first and fourth terms -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1/2 ∧ a 4 = -4 ∧ ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The common ratio of the geometric sequence -/
def common_ratio (a : ℕ → ℚ) : ℚ :=
  (a 2) / (a 1)

/-- Theorem: Properties of the geometric sequence -/
theorem geometric_sequence_properties (a : ℕ → ℚ) 
  (h : geometric_sequence a) : 
  common_ratio a = -2 ∧ ∀ n : ℕ, a n = 1/2 * (-2)^(n-1) := by
  sorry

end geometric_sequence_properties_l1064_106481


namespace inscribed_circle_radius_is_8_l1064_106485

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  -- The height of the triangle
  height : ℝ
  -- The ratio of base to side (4:3)
  baseToSideRatio : ℚ
  -- Assumption that the height is 20
  height_is_20 : height = 20
  -- Assumption that the base to side ratio is 4:3
  ratio_is_4_3 : baseToSideRatio = 4 / 3

/-- The radius of the inscribed circle in the isosceles triangle -/
def inscribedCircleRadius (t : IsoscelesTriangle) : ℝ := 8

/-- Theorem stating that the radius of the inscribed circle is 8 -/
theorem inscribed_circle_radius_is_8 (t : IsoscelesTriangle) :
  inscribedCircleRadius t = 8 := by sorry

end inscribed_circle_radius_is_8_l1064_106485


namespace page_number_divisibility_l1064_106402

theorem page_number_divisibility (n : ℕ) (k : ℕ) : 
  n ≥ 52 → 
  52 ≤ n → 
  n % 13 = 0 → 
  n % k = 0 → 
  ∀ m, m < n → (m % 13 = 0 → m % k = 0) → m < 52 →
  k = 4 := by
  sorry

end page_number_divisibility_l1064_106402


namespace angle_bisector_theorem_l1064_106441

-- Define the triangle
structure Triangle :=
  (a b c : ℝ)
  (distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (positive : a > 0 ∧ b > 0 ∧ c > 0)

-- Define the angle bisector property
def has_equal_angle_bisector_segments (t : Triangle) : Prop :=
  ∃ (d e f : ℝ), d > 0 ∧ e > 0 ∧ f > 0 ∧ d = e

-- Main theorem
theorem angle_bisector_theorem (t : Triangle) 
  (h : has_equal_angle_bisector_segments t) : 
  (t.a / (t.b + t.c) = t.b / (t.c + t.a) + t.c / (t.a + t.b)) ∧ 
  (Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c)) > Real.pi / 2) :=
sorry

end angle_bisector_theorem_l1064_106441


namespace function_minimum_value_l1064_106406

theorem function_minimum_value (x : ℝ) (h : x ≥ 5) :
  (x^2 - 4*x + 9) / (x - 4) ≥ 10 := by
  sorry

#check function_minimum_value

end function_minimum_value_l1064_106406


namespace gcd_factorial_problem_l1064_106495

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 5)) = 5040 := by
  sorry

end gcd_factorial_problem_l1064_106495


namespace quadratic_properties_l1064_106446

/-- A quadratic equation with roots 1 and -1 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  root_one : a + b + c = 0
  root_neg_one : a - b + c = 0

theorem quadratic_properties (eq : QuadraticEquation) :
  eq.a + eq.b + eq.c = 0 ∧ eq.b = 0 := by
  sorry

end quadratic_properties_l1064_106446


namespace chocolate_milk_probability_l1064_106452

theorem chocolate_milk_probability : 
  let n : ℕ := 7  -- total number of days
  let k : ℕ := 5  -- number of days with chocolate milk
  let p : ℚ := 3/4  -- probability of bottling chocolate milk on any given day
  Nat.choose n k * p^k * (1-p)^(n-k) = 5103/16384 := by
  sorry

end chocolate_milk_probability_l1064_106452


namespace equal_distribution_iff_even_total_l1064_106431

/-- Two piles of nuts with different numbers of nuts -/
structure NutPiles :=
  (pile1 : ℕ)
  (pile2 : ℕ)
  (different : pile1 ≠ pile2)

/-- The total number of nuts in both piles -/
def total_nuts (piles : NutPiles) : ℕ := piles.pile1 + piles.pile2

/-- A predicate indicating whether equal distribution is possible -/
def equal_distribution_possible (piles : NutPiles) : Prop :=
  ∃ (k : ℕ), piles.pile1 - k = piles.pile2 + k

/-- Theorem stating that equal distribution is possible if and only if the total number of nuts is even -/
theorem equal_distribution_iff_even_total (piles : NutPiles) :
  equal_distribution_possible piles ↔ Even (total_nuts piles) :=
sorry

end equal_distribution_iff_even_total_l1064_106431


namespace line_tangent_to_circumcircle_l1064_106467

/-- Represents a line in the form x = my + n -/
structure Line where
  m : ℝ
  n : ℝ
  h : n > 0

/-- Checks if a line passes through a given point -/
def Line.passesThrough (l : Line) (x y : ℝ) : Prop :=
  x = l.m * y + l.n

/-- Represents the feasible region with its circumcircle -/
structure FeasibleRegion where
  diameter : ℝ

/-- Main theorem -/
theorem line_tangent_to_circumcircle (l : Line) (fr : FeasibleRegion) :
  l.passesThrough 4 4 → fr.diameter = 8 → l.n = 4 := by sorry

end line_tangent_to_circumcircle_l1064_106467


namespace bike_rides_ratio_l1064_106417

/-- Proves that the ratio of John's bike rides to Billy's bike rides is 2:1 --/
theorem bike_rides_ratio : 
  ∀ (john_rides : ℕ),
  (17 : ℕ) + john_rides + (john_rides + 10) = 95 →
  (john_rides : ℚ) / 17 = 2 / 1 := by
  sorry

end bike_rides_ratio_l1064_106417


namespace integer_sum_problem_l1064_106411

theorem integer_sum_problem : 
  ∃ (a b : ℕ+), 
    (a.val * b.val + a.val + b.val = 143) ∧ 
    (Nat.gcd a.val b.val = 1) ∧ 
    (a.val < 30 ∧ b.val < 30) ∧ 
    (a.val + b.val = 23 ∨ a.val + b.val = 24 ∨ a.val + b.val = 28) := by
  sorry

end integer_sum_problem_l1064_106411


namespace range_of_k_l1064_106494

/-- Represents an ellipse equation -/
def is_ellipse (k : ℝ) : Prop :=
  2 * k - 1 > 0 ∧ k - 1 > 0

/-- Represents a hyperbola equation -/
def is_hyperbola (k : ℝ) : Prop :=
  (4 - k) * (k - 3) < 0

/-- The main theorem stating the range of k -/
theorem range_of_k :
  (∀ k : ℝ, (is_ellipse k ∨ is_hyperbola k) ∧ ¬(is_ellipse k ∧ is_hyperbola k)) →
  (∀ k : ℝ, k ≤ 1 ∨ (3 ≤ k ∧ k ≤ 4)) :=
sorry

end range_of_k_l1064_106494


namespace sphere_volume_implies_pi_l1064_106403

theorem sphere_volume_implies_pi (D : ℝ) (h : D > 0) :
  (D^3 / 2 + 1 / 21 * D^3 / 2 = π * D^3 / 6) → π = 22 / 7 := by
sorry

end sphere_volume_implies_pi_l1064_106403


namespace intersection_with_complement_l1064_106483

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set Nat := {2, 3, 5}

-- Define set B
def B : Set Nat := {1, 3, 4, 6}

-- Theorem statement
theorem intersection_with_complement : A ∩ (U \ B) = {2, 5} := by
  sorry

end intersection_with_complement_l1064_106483


namespace factorization_cubic_minus_linear_times_square_l1064_106475

theorem factorization_cubic_minus_linear_times_square (a b : ℝ) :
  a^3 - a*b^2 = a*(a+b)*(a-b) := by sorry

end factorization_cubic_minus_linear_times_square_l1064_106475


namespace equation_solution_l1064_106489

theorem equation_solution (z : ℝ) (hz : z ≠ 0) :
  (5 * z)^10 = (20 * z)^5 ↔ z = 4/5 := by
  sorry

end equation_solution_l1064_106489
