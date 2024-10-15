import Mathlib

namespace NUMINAMATH_CALUDE_parabola_points_order_l1883_188310

/-- Given a parabola y = 2(x-2)^2 + 1 and three points on it, 
    prove that the y-coordinates are in a specific order -/
theorem parabola_points_order (y₁ y₂ y₃ : ℝ) : 
  (y₁ = 2*(-3-2)^2 + 1) →  -- Point A(-3, y₁)
  (y₂ = 2*(3-2)^2 + 1) →   -- Point B(3, y₂)
  (y₃ = 2*(4-2)^2 + 1) →   -- Point C(4, y₃)
  y₂ < y₃ ∧ y₃ < y₁ := by
  sorry

end NUMINAMATH_CALUDE_parabola_points_order_l1883_188310


namespace NUMINAMATH_CALUDE_victors_final_amount_l1883_188320

/-- Calculates the final amount of money Victor has after transactions -/
def final_amount (initial : ℕ) (allowance : ℕ) (additional : ℕ) (expense : ℕ) : ℕ :=
  initial + allowance + additional - expense

/-- Theorem stating that Victor's final amount is $203 -/
theorem victors_final_amount :
  final_amount 145 88 30 60 = 203 := by
  sorry

end NUMINAMATH_CALUDE_victors_final_amount_l1883_188320


namespace NUMINAMATH_CALUDE_system_has_three_solutions_l1883_188390

/-- The system of equations has exactly 3 distinct real solutions -/
theorem system_has_three_solutions :
  ∃! (S : Set (ℝ × ℝ × ℝ × ℝ)), 
    (∀ (a b c d : ℝ), (a, b, c, d) ∈ S ↔ 
      (a = (b + c + d)^3 ∧
       b = (a + c + d)^3 ∧
       c = (a + b + d)^3 ∧
       d = (a + b + c)^3)) ∧
    S.ncard = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_has_three_solutions_l1883_188390


namespace NUMINAMATH_CALUDE_smallest_common_factor_l1883_188315

theorem smallest_common_factor (n : ℕ) : n > 0 ∧ 
  (∃ k : ℕ, k > 1 ∧ k ∣ (8*n - 3) ∧ k ∣ (6*n + 4)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, k > 1 ∧ k ∣ (8*m - 3) ∧ k ∣ (6*m + 4))) →
  n = 1 := by sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l1883_188315


namespace NUMINAMATH_CALUDE_parabola_translation_l1883_188323

/-- Represents a vertical translation of a function -/
def verticalTranslation (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := fun x ↦ f x + k

/-- Represents a horizontal translation of a function -/
def horizontalTranslation (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := fun x ↦ f (x + h)

/-- The original parabola function -/
def originalParabola : ℝ → ℝ := fun x ↦ x^2

/-- The resulting parabola after translation -/
def resultingParabola : ℝ → ℝ := fun x ↦ (x + 1)^2 + 3

theorem parabola_translation :
  verticalTranslation (horizontalTranslation originalParabola 1) 3 = resultingParabola := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l1883_188323


namespace NUMINAMATH_CALUDE_no_four_identical_digits_in_powers_of_two_l1883_188328

theorem no_four_identical_digits_in_powers_of_two :
  ∀ n : ℕ, ¬ ∃ a : ℕ, a < 10 ∧ (2^n : ℕ) % 10000 = a * 1111 :=
sorry

end NUMINAMATH_CALUDE_no_four_identical_digits_in_powers_of_two_l1883_188328


namespace NUMINAMATH_CALUDE_sum_of_different_geometric_not_geometric_l1883_188306

/-- Given two geometric sequences with different common ratios, their sum sequence is not a geometric sequence -/
theorem sum_of_different_geometric_not_geometric
  {α : Type*} [Field α]
  (a b : ℕ → α)
  (p q : α)
  (hp : p ≠ q)
  (ha : ∀ n, a (n + 1) = p * a n)
  (hb : ∀ n, b (n + 1) = q * b n)
  : ¬ (∃ r : α, ∀ n, (a (n + 1) + b (n + 1)) = r * (a n + b n)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_different_geometric_not_geometric_l1883_188306


namespace NUMINAMATH_CALUDE_other_communities_count_l1883_188312

/-- The number of students belonging to other communities in a school with given demographics -/
theorem other_communities_count (total : ℕ) (muslim hindu sikh buddhist christian jew : ℚ) 
  (h_total : total = 2500)
  (h_muslim : muslim = 28/100)
  (h_hindu : hindu = 26/100)
  (h_sikh : sikh = 12/100)
  (h_buddhist : buddhist = 10/100)
  (h_christian : christian = 6/100)
  (h_jew : jew = 4/100) :
  ↑total * (1 - (muslim + hindu + sikh + buddhist + christian + jew)) = 350 :=
by sorry

end NUMINAMATH_CALUDE_other_communities_count_l1883_188312


namespace NUMINAMATH_CALUDE_sequence_sum_l1883_188349

def geometric_sequence (a : ℕ → ℚ) (r : ℚ) :=
  ∀ n, a (n + 1) = r * a n

theorem sequence_sum (a : ℕ → ℚ) (r : ℚ) :
  geometric_sequence a r →
  a 0 = 16384 →
  a 5 = 16 →
  r = 1/4 →
  a 3 + a 4 = 320 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l1883_188349


namespace NUMINAMATH_CALUDE_managers_salary_solve_manager_salary_problem_l1883_188395

/-- Calculates the manager's salary given the number of employees, their average salary,
    and the increase in average salary when the manager's salary is added. -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) : ℚ :=
  let total_salary := num_employees * avg_salary
  let new_avg_salary := avg_salary + avg_increase
  let new_total_salary := (num_employees + 1) * new_avg_salary
  new_total_salary - total_salary

/-- Proves that the manager's salary is 3300 given the problem conditions. -/
theorem solve_manager_salary_problem :
  managers_salary 20 1200 100 = 3300 := by
  sorry

end NUMINAMATH_CALUDE_managers_salary_solve_manager_salary_problem_l1883_188395


namespace NUMINAMATH_CALUDE_committee_arrangement_count_l1883_188372

def num_women : ℕ := 7
def num_men : ℕ := 3
def num_rocking_chairs : ℕ := 7
def num_stools : ℕ := 3
def num_unique_chair : ℕ := 1
def total_seats : ℕ := num_women + num_men + num_unique_chair

def arrangement_count : ℕ := total_seats * (Nat.choose (total_seats - 1) num_stools)

theorem committee_arrangement_count :
  arrangement_count = 1320 :=
sorry

end NUMINAMATH_CALUDE_committee_arrangement_count_l1883_188372


namespace NUMINAMATH_CALUDE_manuscript_fee_calculation_l1883_188336

def tax_rate_1 : ℚ := 14 / 100
def tax_rate_2 : ℚ := 11 / 100
def tax_threshold_1 : ℕ := 800
def tax_threshold_2 : ℕ := 4000
def tax_paid : ℕ := 420

theorem manuscript_fee_calculation (fee : ℕ) : 
  (tax_threshold_1 < fee ∧ fee ≤ tax_threshold_2 ∧ 
   (fee - tax_threshold_1) * tax_rate_1 = tax_paid) → 
  fee = 3800 :=
by sorry

end NUMINAMATH_CALUDE_manuscript_fee_calculation_l1883_188336


namespace NUMINAMATH_CALUDE_equation_has_two_solutions_l1883_188361

-- Define the equation
def equation (x : ℝ) : Prop := Real.sqrt (9 - x) = x * Real.sqrt (9 - x)

-- Theorem statement
theorem equation_has_two_solutions :
  ∃ (a b : ℝ), a ≠ b ∧ equation a ∧ equation b ∧ ∀ (x : ℝ), equation x → (x = a ∨ x = b) := by
  sorry

end NUMINAMATH_CALUDE_equation_has_two_solutions_l1883_188361


namespace NUMINAMATH_CALUDE_sum_of_powers_l1883_188359

theorem sum_of_powers (w : ℂ) (hw : w^2 - w + 1 = 0) :
  w^97 + w^98 + w^99 + w^100 + w^101 + w^102 = -2 + 2*w := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1883_188359


namespace NUMINAMATH_CALUDE_tea_sale_prices_l1883_188337

structure Tea where
  name : String
  quantity : ℕ
  costPrice : ℚ
  profitPercentage : ℚ

def calculateSalePrice (tea : Tea) : ℚ :=
  tea.costPrice + tea.costPrice * (tea.profitPercentage / 100)

def teaA : Tea := ⟨"A", 120, 25, 45⟩
def teaB : Tea := ⟨"B", 60, 30, 35⟩
def teaC : Tea := ⟨"C", 40, 50, 25⟩
def teaD : Tea := ⟨"D", 30, 70, 20⟩

theorem tea_sale_prices :
  calculateSalePrice teaA = 36.25 ∧
  calculateSalePrice teaB = 40.5 ∧
  calculateSalePrice teaC = 62.5 ∧
  calculateSalePrice teaD = 84 := by
  sorry

end NUMINAMATH_CALUDE_tea_sale_prices_l1883_188337


namespace NUMINAMATH_CALUDE_roots_transformation_l1883_188339

theorem roots_transformation (a b c d : ℝ) : 
  (a^4 - 16*a - 2 = 0) ∧ 
  (b^4 - 16*b - 2 = 0) ∧ 
  (c^4 - 16*c - 2 = 0) ∧ 
  (d^4 - 16*d - 2 = 0) →
  ((a+b)/c^2)^4 - 16*((a+b)/c^2)^3 - 1/2 = 0 ∧
  ((a+c)/b^2)^4 - 16*((a+c)/b^2)^3 - 1/2 = 0 ∧
  ((b+c)/a^2)^4 - 16*((b+c)/a^2)^3 - 1/2 = 0 ∧
  ((b+d)/d^2)^4 - 16*((b+d)/d^2)^3 - 1/2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_transformation_l1883_188339


namespace NUMINAMATH_CALUDE_cab_driver_income_l1883_188368

theorem cab_driver_income (day2 day3 day4 day5 average : ℝ) 
  (h1 : day2 = 400)
  (h2 : day3 = 750)
  (h3 : day4 = 400)
  (h4 : day5 = 500)
  (h5 : average = 460)
  (h6 : average = (day1 + day2 + day3 + day4 + day5) / 5) :
  day1 = 250 := by
  sorry

#check cab_driver_income

end NUMINAMATH_CALUDE_cab_driver_income_l1883_188368


namespace NUMINAMATH_CALUDE_total_cost_equation_l1883_188369

/-- Represents the total cost of tickets for a school trip to Green World -/
def totalCost (x : ℕ) : ℕ :=
  40 * x + 60

/-- Theorem stating the relationship between the number of students and the total cost -/
theorem total_cost_equation (x : ℕ) (y : ℕ) :
  y = totalCost x ↔ y = 40 * x + 60 := by sorry

end NUMINAMATH_CALUDE_total_cost_equation_l1883_188369


namespace NUMINAMATH_CALUDE_no_two_digit_number_satisfies_conditions_l1883_188309

theorem no_two_digit_number_satisfies_conditions : ¬ ∃ n : ℕ,
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  n % 2 = 0 ∧         -- even
  n % 13 = 0 ∧        -- multiple of 13
  ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
  ∃ k : ℕ, a * b = k * k  -- product of digits is a perfect square
  := by sorry

end NUMINAMATH_CALUDE_no_two_digit_number_satisfies_conditions_l1883_188309


namespace NUMINAMATH_CALUDE_investment_interest_theorem_l1883_188374

/-- Calculates the total interest paid in an 18-month investment contract with specific interest rates and reinvestment -/
def total_interest (initial_investment : ℝ) : ℝ :=
  let interest_6m := initial_investment * 0.02
  let balance_10m := initial_investment + interest_6m
  let interest_10m := balance_10m * 0.03
  let balance_18m := balance_10m + interest_10m
  let interest_18m := balance_18m * 0.04
  interest_6m + interest_10m + interest_18m

/-- Theorem stating that the total interest paid in the given investment scenario is $926.24 -/
theorem investment_interest_theorem :
  total_interest 10000 = 926.24 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_theorem_l1883_188374


namespace NUMINAMATH_CALUDE_linear_system_solution_l1883_188300

theorem linear_system_solution (x y m : ℝ) : 
  x + 2*y = m → 
  2*x - 3*y = 4 → 
  x + y = 7 → 
  m = 9 := by
sorry

end NUMINAMATH_CALUDE_linear_system_solution_l1883_188300


namespace NUMINAMATH_CALUDE_count_positive_integer_solutions_l1883_188348

/-- The number of positive integer solutions for the equation x + y + z + t = 15 -/
theorem count_positive_integer_solutions : 
  (Finset.filter (fun (x : ℕ × ℕ × ℕ × ℕ) => x.1 + x.2.1 + x.2.2.1 + x.2.2.2 = 15) 
    (Finset.product (Finset.range 15) (Finset.product (Finset.range 15) 
      (Finset.product (Finset.range 15) (Finset.range 15))))).card = 364 := by
  sorry

#check count_positive_integer_solutions

end NUMINAMATH_CALUDE_count_positive_integer_solutions_l1883_188348


namespace NUMINAMATH_CALUDE_rectangle_area_l1883_188363

theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let perimeter := 2 * (l + b)
  perimeter = 112 → l * b = 588 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1883_188363


namespace NUMINAMATH_CALUDE_average_sum_abs_diff_l1883_188356

/-- A permutation of integers from 1 to 12 -/
def Permutation := Fin 12 → Fin 12

/-- The sum of absolute differences for a given permutation -/
def sumAbsDiff (p : Permutation) : ℚ :=
  |p 0 - p 1| + |p 2 - p 3| + |p 4 - p 5| + |p 6 - p 7| + |p 8 - p 9| + |p 10 - p 11|

/-- The set of all permutations of integers from 1 to 12 -/
def allPermutations : Finset Permutation := sorry

/-- The average value of sumAbsDiff over all permutations -/
def averageValue : ℚ := (allPermutations.sum sumAbsDiff) / allPermutations.card

theorem average_sum_abs_diff : averageValue = 143 / 33 := by sorry

end NUMINAMATH_CALUDE_average_sum_abs_diff_l1883_188356


namespace NUMINAMATH_CALUDE_probability_of_9_heads_in_12_flips_l1883_188394

def num_flips : ℕ := 12
def num_heads : ℕ := 9

theorem probability_of_9_heads_in_12_flips :
  (num_flips.choose num_heads : ℚ) / 2^num_flips = 55 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_9_heads_in_12_flips_l1883_188394


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l1883_188399

theorem opposite_of_negative_two : 
  ∀ x : ℤ, x + (-2) = 0 → x = 2 := by
sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l1883_188399


namespace NUMINAMATH_CALUDE_union_A_B_equals_open_interval_l1883_188366

-- Define set A
def A : Set ℝ := {x | Real.log (x - 1) < 0}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = 2^x - 1}

-- Theorem statement
theorem union_A_B_equals_open_interval :
  A ∪ B = Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_union_A_B_equals_open_interval_l1883_188366


namespace NUMINAMATH_CALUDE_initial_number_proof_l1883_188389

theorem initial_number_proof (x : ℕ) : x - 109 = 109 + 68 → x = 286 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l1883_188389


namespace NUMINAMATH_CALUDE_triangle_cinema_seats_l1883_188393

/-- Represents a triangular seating arrangement in a cinema --/
structure TriangularCinema where
  best_seat_number : ℕ
  total_seats : ℕ

/-- Checks if a given TriangularCinema configuration is valid --/
def is_valid_cinema (c : TriangularCinema) : Prop :=
  ∃ n : ℕ,
    -- The number of rows is 2n + 1
    (2 * n + 1) * ((2 * n + 1) + 1) / 2 = c.total_seats ∧
    -- The best seat is in the middle row
    (n + 1) * (n + 2) / 2 = c.best_seat_number

/-- Theorem stating the relationship between the best seat number and total seats --/
theorem triangle_cinema_seats (c : TriangularCinema) :
  c.best_seat_number = 265 → is_valid_cinema c → c.total_seats = 1035 := by
  sorry

#check triangle_cinema_seats

end NUMINAMATH_CALUDE_triangle_cinema_seats_l1883_188393


namespace NUMINAMATH_CALUDE_bart_notepad_spending_l1883_188352

/-- The amount of money Bart spent on notepads -/
def money_spent (cost_per_notepad : ℚ) (pages_per_notepad : ℕ) (total_pages : ℕ) : ℚ :=
  (total_pages / pages_per_notepad) * cost_per_notepad

/-- Theorem: Given the conditions, Bart spent $10 on notepads -/
theorem bart_notepad_spending :
  let cost_per_notepad : ℚ := 5/4  -- $1.25 represented as a rational number
  let pages_per_notepad : ℕ := 60
  let total_pages : ℕ := 480
  money_spent cost_per_notepad pages_per_notepad total_pages = 10 := by
  sorry


end NUMINAMATH_CALUDE_bart_notepad_spending_l1883_188352


namespace NUMINAMATH_CALUDE_particular_number_calculation_l1883_188347

theorem particular_number_calculation (x : ℝ) (h : 2.5 * x - 2.49 = 22.01) :
  (x / 2.5) + 2.49 + 22.01 = 28.42 := by
sorry

end NUMINAMATH_CALUDE_particular_number_calculation_l1883_188347


namespace NUMINAMATH_CALUDE_function_equation_solution_l1883_188367

/-- Given functions f and g satisfying the condition for all x and y, 
    prove that f and g have the specified forms. -/
theorem function_equation_solution 
  (f g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f y + g x - g y = Real.sin x + Real.cos y) :
  (∃ c : ℝ, (∀ x : ℝ, f x = (Real.sin x + Real.cos x) / 2 ∧ 
                       g x = (Real.sin x - Real.cos x) / 2 + c)) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l1883_188367


namespace NUMINAMATH_CALUDE_fence_painting_earnings_l1883_188380

/-- Calculate the total earnings from painting fences -/
theorem fence_painting_earnings
  (rate : ℝ)
  (num_fences : ℕ)
  (fence_length : ℝ)
  (h1 : rate = 0.20)
  (h2 : num_fences = 50)
  (h3 : fence_length = 500) :
  rate * (↑num_fences * fence_length) = 5000 := by
  sorry

end NUMINAMATH_CALUDE_fence_painting_earnings_l1883_188380


namespace NUMINAMATH_CALUDE_prob_no_consecutive_heads_is_9_64_l1883_188392

/-- The number of ways to arrange k heads in n + 1 positions without consecutive heads -/
def arrange_heads (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of sequences without two consecutive heads in 10 coin tosses -/
def total_favorable_sequences : ℕ :=
  arrange_heads 10 0 + arrange_heads 9 1 + arrange_heads 8 2 +
  arrange_heads 7 3 + arrange_heads 6 4 + arrange_heads 5 5

/-- The total number of possible outcomes when tossing a coin 10 times -/
def total_outcomes : ℕ := 2^10

/-- The probability of no two consecutive heads in 10 coin tosses -/
def prob_no_consecutive_heads : ℚ := total_favorable_sequences / total_outcomes

theorem prob_no_consecutive_heads_is_9_64 :
  prob_no_consecutive_heads = 9/64 := by sorry

end NUMINAMATH_CALUDE_prob_no_consecutive_heads_is_9_64_l1883_188392


namespace NUMINAMATH_CALUDE_range_of_a_l1883_188332

-- Define the inequality system
def inequality_system (a : ℝ) (x : ℝ) : Prop :=
  x > a ∧ x > 1

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {x | x > 1}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ x, inequality_system a x ↔ x ∈ solution_set a) →
  a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1883_188332


namespace NUMINAMATH_CALUDE_magic_square_sum_l1883_188355

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a b c d e : ℕ)
  (sum : ℕ)
  (row1_sum : a + 22 + b = sum)
  (row2_sum : 20 + c + e = sum)
  (row3_sum : 28 + d + 19 = sum)
  (col1_sum : a + 20 + 28 = sum)
  (col2_sum : 22 + c + d = sum)
  (col3_sum : b + e + 19 = sum)
  (diag1_sum : a + c + 19 = sum)
  (diag2_sum : 28 + c + b = sum)

/-- Theorem: In the given magic square, d + e = 70 -/
theorem magic_square_sum (ms : MagicSquare) : ms.d + ms.e = 70 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_sum_l1883_188355


namespace NUMINAMATH_CALUDE_keith_placed_scissors_l1883_188345

/-- The number of scissors Keith placed in the drawer -/
def scissors_placed (initial final : ℕ) : ℕ := final - initial

/-- Proof that Keith placed 22 scissors in the drawer -/
theorem keith_placed_scissors : scissors_placed 54 76 = 22 := by
  sorry

end NUMINAMATH_CALUDE_keith_placed_scissors_l1883_188345


namespace NUMINAMATH_CALUDE_athlete_team_division_l1883_188354

theorem athlete_team_division (n : ℕ) (k : ℕ) (total : ℕ) (specific : ℕ) :
  n = 10 →
  k = 5 →
  total = n →
  specific = 2 →
  (Nat.choose (n - specific) (k - 1)) = 70 := by
  sorry

end NUMINAMATH_CALUDE_athlete_team_division_l1883_188354


namespace NUMINAMATH_CALUDE_digit_move_equals_multiply_divide_l1883_188388

def N : ℕ := 2173913043478260869565

theorem digit_move_equals_multiply_divide :
  (N * 4) / 5 = (N % 10^22) * 10 + (N / 10^22) :=
by sorry

end NUMINAMATH_CALUDE_digit_move_equals_multiply_divide_l1883_188388


namespace NUMINAMATH_CALUDE_largest_equal_cost_number_l1883_188330

/-- Calculates the sum of digits in decimal representation -/
def sumDecimalDigits (n : Nat) : Nat :=
  sorry

/-- Calculates the sum of digits in binary representation -/
def sumBinaryDigits (n : Nat) : Nat :=
  sorry

/-- Checks if the costs are equal for both options -/
def equalCost (n : Nat) : Prop :=
  sumDecimalDigits n = sumBinaryDigits n

theorem largest_equal_cost_number :
  (∀ m : Nat, m < 500 → m > 404 → ¬(equalCost m)) ∧
  equalCost 404 :=
sorry

end NUMINAMATH_CALUDE_largest_equal_cost_number_l1883_188330


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l1883_188376

theorem max_sum_of_squares (m n : ℕ) : 
  m ∈ Finset.range 1982 ∧ 
  n ∈ Finset.range 1982 ∧ 
  (n^2 - m*n - m^2)^2 = 1 →
  m^2 + n^2 ≤ 3524578 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l1883_188376


namespace NUMINAMATH_CALUDE_test_score_combination_l1883_188304

theorem test_score_combination :
  ∀ (x y z : ℕ),
    x + y + z = 6 →
    8 * x + 2 * y = 20 →
    x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end NUMINAMATH_CALUDE_test_score_combination_l1883_188304


namespace NUMINAMATH_CALUDE_min_sum_reciprocal_distances_l1883_188326

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line passing through a point with a given angle -/
structure Line where
  point : Point
  angle : ℝ

/-- Curve in polar form -/
structure PolarCurve where
  equation : ℝ → ℝ

/-- Function to calculate the minimum sum of reciprocal distances -/
noncomputable def minSumReciprocalDistances (l : Line) (c : PolarCurve) : ℝ :=
  sorry

/-- Theorem stating the minimum value of the sum of reciprocal distances -/
theorem min_sum_reciprocal_distances :
  let p := Point.mk 1 2
  let l := Line.mk p α
  let c := PolarCurve.mk (fun θ ↦ 6 * Real.sin θ)
  minSumReciprocalDistances l c = 2 * Real.sqrt 7 / 7 := by sorry

end NUMINAMATH_CALUDE_min_sum_reciprocal_distances_l1883_188326


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1883_188313

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (Real.log x / Real.log 10)}
def B : Set ℝ := {x | 1 / x ≥ 1 / 2}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1883_188313


namespace NUMINAMATH_CALUDE_max_value_of_sqrt_sum_l1883_188387

theorem max_value_of_sqrt_sum (x : ℝ) (h : -9 ≤ x ∧ x ≤ 9) : 
  ∃ (max : ℝ), max = 6 ∧ 
  (∀ y : ℝ, -9 ≤ y ∧ y ≤ 9 → Real.sqrt (9 + y) + Real.sqrt (9 - y) ≤ max) ∧
  (∃ z : ℝ, -9 ≤ z ∧ z ≤ 9 ∧ Real.sqrt (9 + z) + Real.sqrt (9 - z) = max) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sqrt_sum_l1883_188387


namespace NUMINAMATH_CALUDE_equal_sides_implies_rhombus_rhombus_equal_diagonals_implies_square_equal_angles_implies_rectangle_l1883_188353

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define properties of quadrilaterals
def Quadrilateral.is_rhombus (q : Quadrilateral) : Prop :=
  sorry

def Quadrilateral.is_square (q : Quadrilateral) : Prop :=
  sorry

def Quadrilateral.is_rectangle (q : Quadrilateral) : Prop :=
  sorry

def Quadrilateral.has_equal_sides (q : Quadrilateral) : Prop :=
  sorry

def Quadrilateral.has_equal_diagonals (q : Quadrilateral) : Prop :=
  sorry

def Quadrilateral.has_equal_angles (q : Quadrilateral) : Prop :=
  sorry

-- Theorem statements
theorem equal_sides_implies_rhombus (q : Quadrilateral) :
  q.has_equal_sides → q.is_rhombus :=
sorry

theorem rhombus_equal_diagonals_implies_square (q : Quadrilateral) :
  q.is_rhombus → q.has_equal_diagonals → q.is_square :=
sorry

theorem equal_angles_implies_rectangle (q : Quadrilateral) :
  q.has_equal_angles → q.is_rectangle :=
sorry

end NUMINAMATH_CALUDE_equal_sides_implies_rhombus_rhombus_equal_diagonals_implies_square_equal_angles_implies_rectangle_l1883_188353


namespace NUMINAMATH_CALUDE_b_95_mod_64_l1883_188303

/-- The sequence b_n defined as 7^n + 9^n -/
def b (n : ℕ) : ℕ := 7^n + 9^n

/-- The theorem stating that b_95 ≡ 48 (mod 64) -/
theorem b_95_mod_64 : b 95 ≡ 48 [ZMOD 64] := by
  sorry

end NUMINAMATH_CALUDE_b_95_mod_64_l1883_188303


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1883_188325

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 2)
  (x_ge : x ≥ -1)
  (y_ge : y ≥ -3/2)
  (z_ge : z ≥ -2) :
  (∃ (x₀ y₀ z₀ : ℝ), x₀ + y₀ + z₀ = 2 ∧ 
   x₀ ≥ -1 ∧ y₀ ≥ -3/2 ∧ z₀ ≥ -2 ∧
   Real.sqrt (5 * x₀ + 5) + Real.sqrt (4 * y₀ + 6) + Real.sqrt (6 * z₀ + 10) = Real.sqrt 93) ∧
  (∀ (a b c : ℝ), a + b + c = 2 → 
   a ≥ -1 → b ≥ -3/2 → c ≥ -2 →
   Real.sqrt (5 * a + 5) + Real.sqrt (4 * b + 6) + Real.sqrt (6 * c + 10) ≤ Real.sqrt 93) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1883_188325


namespace NUMINAMATH_CALUDE_registration_cost_per_vehicle_l1883_188314

theorem registration_cost_per_vehicle 
  (num_dirt_bikes : ℕ) 
  (cost_per_dirt_bike : ℕ) 
  (num_off_road : ℕ) 
  (cost_per_off_road : ℕ) 
  (total_cost : ℕ) 
  (h1 : num_dirt_bikes = 3)
  (h2 : cost_per_dirt_bike = 150)
  (h3 : num_off_road = 4)
  (h4 : cost_per_off_road = 300)
  (h5 : total_cost = 1825) :
  (total_cost - (num_dirt_bikes * cost_per_dirt_bike + num_off_road * cost_per_off_road)) / (num_dirt_bikes + num_off_road) = 25 := by
    sorry

end NUMINAMATH_CALUDE_registration_cost_per_vehicle_l1883_188314


namespace NUMINAMATH_CALUDE_last_four_digits_are_user_number_l1883_188317

/-- Represents a mobile phone number -/
structure MobilePhoneNumber where
  digits : Fin 11 → Nat
  network_id : Fin 3 → Nat
  area_code : Fin 3 → Nat
  user_number : Fin 4 → Nat

/-- The structure of a mobile phone number -/
def mobile_number_structure (m : MobilePhoneNumber) : Prop :=
  (∀ i : Fin 3, m.network_id i = m.digits i) ∧
  (∀ i : Fin 3, m.area_code i = m.digits (i + 3)) ∧
  (∀ i : Fin 4, m.user_number i = m.digits (i + 7))

/-- Theorem stating that the last 4 digits of a mobile phone number represent the user number -/
theorem last_four_digits_are_user_number (m : MobilePhoneNumber) 
  (h : mobile_number_structure m) : 
  ∀ i : Fin 4, m.user_number i = m.digits (i + 7) := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_are_user_number_l1883_188317


namespace NUMINAMATH_CALUDE_min_value_expression_l1883_188341

theorem min_value_expression (x : ℝ) (h : x > 4) :
  (x + 5) / Real.sqrt (x - 4) ≥ 6 ∧ ∃ y : ℝ, y > 4 ∧ (y + 5) / Real.sqrt (y - 4) = 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1883_188341


namespace NUMINAMATH_CALUDE_expression_simplification_l1883_188384

theorem expression_simplification (a b : ℝ) (ha : a > 0) :
  a^(1/3) * b^(1/2) * (-3 * a^(1/2) * b^(1/3)) / ((1/3) * a^(1/6) * b^(5/6)) = -9 * a^(2/3) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1883_188384


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_chord_l1883_188351

/-- The line that intersects the unit circle -/
def intersecting_line (x y : ℝ) : Prop := x - y + 1 = 0

/-- The unit circle -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- The perpendicular bisector of the chord -/
def perpendicular_bisector (x y : ℝ) : Prop := x + y = 0

/-- Theorem: The perpendicular bisector of the chord formed by the intersection
    of the line x - y + 1 = 0 and the unit circle x^2 + y^2 = 1 
    has the equation x + y = 0 -/
theorem perpendicular_bisector_of_chord :
  ∀ (x y : ℝ), 
  intersecting_line x y → unit_circle x y →
  perpendicular_bisector x y :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_chord_l1883_188351


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l1883_188305

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x * Real.log 3 / Real.log 2 + y = Real.log 18 / Real.log 2

def equation2 (x y : ℝ) : Prop := (5 : ℝ)^x = 25^y

-- Theorem statement
theorem solution_satisfies_system :
  equation1 2 1 ∧ equation2 2 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l1883_188305


namespace NUMINAMATH_CALUDE_simplify_product_of_square_roots_l1883_188338

theorem simplify_product_of_square_roots (x : ℝ) (h : x > 0) :
  Real.sqrt (50 * x^3) * Real.sqrt (18 * x^2) * Real.sqrt (35 * x) = 30 * x^3 * Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_product_of_square_roots_l1883_188338


namespace NUMINAMATH_CALUDE_divisible_by_nine_l1883_188318

theorem divisible_by_nine (A : ℕ) : A < 10 → (83 * 1000 + A * 100 + 5) % 9 = 0 ↔ A = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l1883_188318


namespace NUMINAMATH_CALUDE_salt_mixture_percentage_l1883_188321

theorem salt_mixture_percentage : 
  let initial_volume : ℝ := 70
  let initial_concentration : ℝ := 0.20
  let added_volume : ℝ := 70
  let added_concentration : ℝ := 0.60
  let total_volume : ℝ := initial_volume + added_volume
  let final_concentration : ℝ := (initial_volume * initial_concentration + added_volume * added_concentration) / total_volume
  final_concentration = 0.40 := by sorry

end NUMINAMATH_CALUDE_salt_mixture_percentage_l1883_188321


namespace NUMINAMATH_CALUDE_no_preimage_range_l1883_188362

/-- The function f: ℝ → ℝ defined by f(x) = -x² + 2x -/
def f (x : ℝ) : ℝ := -x^2 + 2*x

/-- The theorem stating that k > 1 is the range of values for which f(x) = k has no solution -/
theorem no_preimage_range (k : ℝ) : 
  (∀ x, f x ≠ k) ↔ k > 1 := by
  sorry

#check no_preimage_range

end NUMINAMATH_CALUDE_no_preimage_range_l1883_188362


namespace NUMINAMATH_CALUDE_comet_orbit_equation_l1883_188346

/-- Represents the equation of an ellipse -/
structure EllipseEquation where
  a : ℝ
  b : ℝ
  (positive_a : 0 < a)
  (positive_b : 0 < b)

/-- Represents the orbital parameters of a comet -/
structure CometOrbit where
  perihelion : ℝ
  aphelion : ℝ
  (positive_perihelion : 0 < perihelion)
  (positive_aphelion : 0 < aphelion)
  (perihelion_less_than_aphelion : perihelion < aphelion)

/-- 
Given a comet's orbit with perihelion 2 AU and aphelion 6 AU from the Sun,
prove that its orbit equation is x²/16 + y²/12 = 1
-/
theorem comet_orbit_equation (orbit : CometOrbit) 
  (h_perihelion : orbit.perihelion = 2)
  (h_aphelion : orbit.aphelion = 6) : 
  ∃ (eq : EllipseEquation), eq.a = 4 ∧ eq.b = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_comet_orbit_equation_l1883_188346


namespace NUMINAMATH_CALUDE_condo_units_count_l1883_188324

/-- Represents a condo development with regular and penthouse floors. -/
structure Condo where
  total_floors : Nat
  penthouse_floors : Nat
  regular_units : Nat
  penthouse_units : Nat

/-- Calculates the total number of units in a condo. -/
def total_units (c : Condo) : Nat :=
  (c.total_floors - c.penthouse_floors) * c.regular_units + c.penthouse_floors * c.penthouse_units

/-- Theorem stating that a condo with the given specifications has 256 units. -/
theorem condo_units_count : 
  let c : Condo := {
    total_floors := 23,
    penthouse_floors := 2,
    regular_units := 12,
    penthouse_units := 2
  }
  total_units c = 256 := by
  sorry

#check condo_units_count

end NUMINAMATH_CALUDE_condo_units_count_l1883_188324


namespace NUMINAMATH_CALUDE_estimate_red_balls_l1883_188342

/-- Represents the number of balls in the bag -/
def total_balls : ℕ := 10

/-- Represents the total number of draws -/
def total_draws : ℕ := 1000

/-- Represents the number of times a red ball was drawn -/
def red_draws : ℕ := 200

/-- The estimated number of red balls in the bag -/
def estimated_red_balls : ℚ := (red_draws : ℚ) / total_draws * total_balls

theorem estimate_red_balls :
  estimated_red_balls = 2 := by sorry

end NUMINAMATH_CALUDE_estimate_red_balls_l1883_188342


namespace NUMINAMATH_CALUDE_ben_win_probability_l1883_188382

theorem ben_win_probability (lose_prob : ℚ) (h1 : lose_prob = 5/8) (h2 : ¬ ∃ tie_prob : ℚ, tie_prob ≠ 0) :
  1 - lose_prob = 3/8 :=
by sorry

end NUMINAMATH_CALUDE_ben_win_probability_l1883_188382


namespace NUMINAMATH_CALUDE_raine_steps_l1883_188381

/-- The number of steps Raine takes to walk to school -/
def steps_to_school : ℕ := 150

/-- The number of days Raine walks to and from school -/
def days : ℕ := 5

/-- The total number of steps Raine takes in five days -/
def total_steps : ℕ := 2 * steps_to_school * days

theorem raine_steps : total_steps = 1500 := by
  sorry

end NUMINAMATH_CALUDE_raine_steps_l1883_188381


namespace NUMINAMATH_CALUDE_tan_value_from_trig_equation_l1883_188333

theorem tan_value_from_trig_equation (α : Real) 
  (h : (Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 1/5) : 
  Real.tan α = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_trig_equation_l1883_188333


namespace NUMINAMATH_CALUDE_inequality_solution_l1883_188364

theorem inequality_solution (x : ℝ) : 
  (x * (x + 2)) / ((x - 5)^2) ≥ 15 ↔ 
  (x ≥ 5/2 ∧ x < 5) ∨ (x > 5 ∧ x ≤ 75/7) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1883_188364


namespace NUMINAMATH_CALUDE_student_grade_problem_l1883_188357

/-- Given a student's grades in three subjects, prove that if the second subject is 70%,
    the third subject is 90%, and the overall average is 70%, then the first subject must be 50%. -/
theorem student_grade_problem (grade1 grade2 grade3 : ℝ) : 
  grade2 = 70 → grade3 = 90 → (grade1 + grade2 + grade3) / 3 = 70 → grade1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_student_grade_problem_l1883_188357


namespace NUMINAMATH_CALUDE_population_growth_theorem_l1883_188398

/-- The annual population growth rate due to natural growth -/
def natural_growth_rate : ℝ := 0.06

/-- The overall population growth rate over 3 years -/
def total_growth_rate : ℝ := 0.157625

/-- The annual population decrease rate due to migration -/
def migration_decrease_rate : ℝ := 0.009434

theorem population_growth_theorem :
  ∃ (x : ℝ),
    (((1 + natural_growth_rate) * (1 - x))^3 = 1 + total_growth_rate) ∧
    (abs (x - migration_decrease_rate) < 0.00001) := by
  sorry

end NUMINAMATH_CALUDE_population_growth_theorem_l1883_188398


namespace NUMINAMATH_CALUDE_probability_four_old_balls_value_l1883_188343

def total_balls : ℕ := 12
def new_balls : ℕ := 9
def old_balls : ℕ := 3
def drawn_balls : ℕ := 3

def probability_four_old_balls : ℚ :=
  (Nat.choose old_balls 2 * Nat.choose new_balls 1) / Nat.choose total_balls drawn_balls

theorem probability_four_old_balls_value :
  probability_four_old_balls = 27 / 220 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_old_balls_value_l1883_188343


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_abs_x_gt_one_l1883_188302

theorem x_gt_one_sufficient_not_necessary_for_abs_x_gt_one :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧
  (∃ x : ℝ, |x| > 1 ∧ ¬(x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_abs_x_gt_one_l1883_188302


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l1883_188391

theorem quadratic_form_ratio (j : ℝ) : 
  ∃ (c p q : ℝ), 8 * j^2 - 6 * j + 20 = c * (j + p)^2 + q ∧ q / p = -151 / 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l1883_188391


namespace NUMINAMATH_CALUDE_prime_divisibility_l1883_188360

theorem prime_divisibility (p q : ℕ) : 
  Prime p → Prime q → (p * q) ∣ (2^p + 2^q) → p = 2 ∧ q = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_l1883_188360


namespace NUMINAMATH_CALUDE_fifth_bank_coins_l1883_188340

def coins_in_bank (n : ℕ) : ℕ := 72 + 9 * (n - 1)

theorem fifth_bank_coins :
  coins_in_bank 5 = 108 :=
by sorry

end NUMINAMATH_CALUDE_fifth_bank_coins_l1883_188340


namespace NUMINAMATH_CALUDE_last_digit_sum_powers_l1883_188385

theorem last_digit_sum_powers : (1993^2002 + 1995^2002) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_sum_powers_l1883_188385


namespace NUMINAMATH_CALUDE_fraction_ordering_l1883_188386

theorem fraction_ordering : 
  (21 : ℚ) / 17 < (18 : ℚ) / 13 ∧ (18 : ℚ) / 13 < (14 : ℚ) / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l1883_188386


namespace NUMINAMATH_CALUDE_min_students_with_both_l1883_188301

theorem min_students_with_both (n : ℕ) (glasses watches both : ℕ → ℕ) :
  (∀ m : ℕ, m ≥ n → glasses m = (3 * m) / 8) →
  (∀ m : ℕ, m ≥ n → watches m = (5 * m) / 6) →
  (∀ m : ℕ, m ≥ n → glasses m + watches m - both m = m) →
  (∃ m : ℕ, m ≥ n ∧ both m = 5 ∧ ∀ k, k < m → ¬(glasses k = (3 * k) / 8 ∧ watches k = (5 * k) / 6)) :=
sorry

end NUMINAMATH_CALUDE_min_students_with_both_l1883_188301


namespace NUMINAMATH_CALUDE_first_fifth_mile_charge_l1883_188308

/-- Represents the charge structure of a taxi company -/
structure TaxiCharge where
  first_fifth_mile : ℝ
  per_additional_fifth : ℝ

/-- Calculates the total charge for a given distance -/
def total_charge (c : TaxiCharge) (distance : ℝ) : ℝ :=
  c.first_fifth_mile + c.per_additional_fifth * (distance * 5 - 1)

/-- Theorem stating the charge for the first 1/5 mile -/
theorem first_fifth_mile_charge (c : TaxiCharge) :
  c.per_additional_fifth = 0.40 →
  total_charge c 8 = 18.10 →
  c.first_fifth_mile = 2.50 := by
sorry

end NUMINAMATH_CALUDE_first_fifth_mile_charge_l1883_188308


namespace NUMINAMATH_CALUDE_baseball_games_played_l1883_188373

theorem baseball_games_played (runs_1 runs_4 runs_5 : ℕ) (avg_runs : ℚ) : 
  runs_1 = 1 → runs_4 = 2 → runs_5 = 3 → avg_runs = 4 → 
  (runs_1 * 1 + runs_4 * 4 + runs_5 * 5 : ℚ) / (runs_1 + runs_4 + runs_5) = avg_runs → 
  runs_1 + runs_4 + runs_5 = 6 := by
sorry

end NUMINAMATH_CALUDE_baseball_games_played_l1883_188373


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1883_188335

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 3*y = 1) :
  (1/x + 1/y) ≥ 5 + 2*Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1883_188335


namespace NUMINAMATH_CALUDE_x29x_divisible_by_18_l1883_188350

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def four_digit_number (x : ℕ) : ℕ := 1000 * x + 290 + x

theorem x29x_divisible_by_18 :
  ∃! x : ℕ, is_single_digit x ∧ (four_digit_number x) % 18 = 0 ∧ x = 2 :=
sorry

end NUMINAMATH_CALUDE_x29x_divisible_by_18_l1883_188350


namespace NUMINAMATH_CALUDE_arithmetic_sum_property_l1883_188311

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sum_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 5 + a 7 = 2) →
  (a 4 + a 6 + a 8 = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_property_l1883_188311


namespace NUMINAMATH_CALUDE_trig_expression_equality_l1883_188377

theorem trig_expression_equality : 
  (Real.cos (27 * π / 180) - Real.sqrt 2 * Real.sin (18 * π / 180)) / Real.cos (63 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l1883_188377


namespace NUMINAMATH_CALUDE_smallest_dividend_l1883_188379

theorem smallest_dividend (q r : ℕ) (h1 : q = 12) (h2 : r = 3) :
  ∃ (a b : ℕ), a = b * q + r ∧ b > r ∧ ∀ (a' b' : ℕ), (a' = b' * q + r ∧ b' > r) → a ≤ a' :=
by sorry

end NUMINAMATH_CALUDE_smallest_dividend_l1883_188379


namespace NUMINAMATH_CALUDE_playful_not_brown_l1883_188307

structure Dog where
  playful : Prop
  brown : Prop
  knowsTricks : Prop
  canSwim : Prop

axiom all_playful_know_tricks : ∀ (d : Dog), d.playful → d.knowsTricks
axiom no_brown_can_swim : ∀ (d : Dog), d.brown → ¬d.canSwim
axiom cant_swim_dont_know_tricks : ∀ (d : Dog), ¬d.canSwim → ¬d.knowsTricks

theorem playful_not_brown : ∀ (d : Dog), d.playful → ¬d.brown := by
  sorry

end NUMINAMATH_CALUDE_playful_not_brown_l1883_188307


namespace NUMINAMATH_CALUDE_not_divisible_by_5_and_7_count_count_less_than_1000_l1883_188365

theorem not_divisible_by_5_and_7_count : Nat → Nat
  | n => (n + 1) - (n / 5 + n / 7 - n / 35)

theorem count_less_than_1000 :
  not_divisible_by_5_and_7_count 999 = 686 := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_5_and_7_count_count_less_than_1000_l1883_188365


namespace NUMINAMATH_CALUDE_bounded_fraction_exists_l1883_188378

theorem bounded_fraction_exists (C : ℝ) : ∃ C, ∀ k : ℤ, 
  |((k^8 - 2*k + 1) / (k^4 - 3))| < C :=
sorry

end NUMINAMATH_CALUDE_bounded_fraction_exists_l1883_188378


namespace NUMINAMATH_CALUDE_similar_triangles_problem_l1883_188319

/-- Represents a triangle with an area and a side length -/
structure Triangle where
  area : ℝ
  side : ℝ

/-- The problem statement -/
theorem similar_triangles_problem 
  (t1 t2 : Triangle)  -- Two triangles
  (h1 : t1.area > t2.area)  -- t1 is the larger triangle
  (h2 : t1.area - t2.area = 32)  -- Area difference is 32
  (h3 : ∃ k : ℕ, t1.area / t2.area = k^2)  -- Ratio of areas is square of an integer
  (h4 : ∃ n : ℕ, t2.area = n)  -- Smaller triangle area is an integer
  (h5 : t2.side = 4)  -- Side of smaller triangle is 4
  : t1.side = 12 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_problem_l1883_188319


namespace NUMINAMATH_CALUDE_parabola_hyperbola_tangency_l1883_188316

theorem parabola_hyperbola_tangency (n : ℝ) : 
  (∃ x y : ℝ, y = x^2 + 6 ∧ y^2 - n*x^2 = 4 ∧ 
    ∀ x' y' : ℝ, y' = x'^2 + 6 → y'^2 - n*x'^2 = 4 → (x', y') = (x, y)) →
  (n = 12 + 4*Real.sqrt 7 ∨ n = 12 - 4*Real.sqrt 7) :=
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_tangency_l1883_188316


namespace NUMINAMATH_CALUDE_race_outcomes_l1883_188397

/-- The number of contestants in the race -/
def num_contestants : ℕ := 6

/-- The number of podium positions (1st, 2nd, 3rd) -/
def podium_positions : ℕ := 3

/-- No ties are allowed in the race -/
axiom no_ties : True

/-- The number of different podium outcomes in the race -/
def podium_outcomes : ℕ := num_contestants * (num_contestants - 1) * (num_contestants - 2)

/-- Theorem: The number of different podium outcomes in the race is 120 -/
theorem race_outcomes : podium_outcomes = 120 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_l1883_188397


namespace NUMINAMATH_CALUDE_milk_bottle_recycling_l1883_188334

theorem milk_bottle_recycling (marcus_bottles john_bottles : ℕ) 
  (h1 : marcus_bottles = 25) 
  (h2 : john_bottles = 20) : 
  marcus_bottles + john_bottles = 45 := by
  sorry

end NUMINAMATH_CALUDE_milk_bottle_recycling_l1883_188334


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1883_188383

/-- A hyperbola with foci F₁ and F₂ -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A point on the hyperbola -/
def Point := ℝ × ℝ

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The angle between three points -/
def angle (p q r : ℝ × ℝ) : ℝ := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

theorem hyperbola_eccentricity (h : Hyperbola) (P : Point) 
  (h1 : distance P h.F₂ = 2 * distance P h.F₁)
  (h2 : angle P h.F₁ h.F₂ = Real.pi / 3) : 
  eccentricity h = (1 + Real.sqrt 13) / 2 := sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1883_188383


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l1883_188327

theorem roots_of_quadratic (x : ℝ) : x^2 = 2*x ↔ x = 0 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l1883_188327


namespace NUMINAMATH_CALUDE_square_sequence_formulas_l1883_188396

/-- The number of squares in the nth figure of the sequence -/
def num_squares (n : ℕ) : ℕ := 2 * n^2 - 2 * n + 1

/-- The first formula: (2n-1)^2 - 4 * (n(n-1)/2) -/
def formula_a (n : ℕ) : ℕ := (2 * n - 1)^2 - 2 * n * (n - 1)

/-- The third formula: 1 + (1 + 2 + ... + (n-1)) * 4 -/
def formula_c (n : ℕ) : ℕ := 1 + 2 * n * (n - 1)

/-- The fourth formula: (n-1)^2 + n^2 -/
def formula_d (n : ℕ) : ℕ := (n - 1)^2 + n^2

theorem square_sequence_formulas (n : ℕ) : 
  n > 0 → num_squares n = formula_a n ∧ num_squares n = formula_c n ∧ num_squares n = formula_d n :=
by sorry

end NUMINAMATH_CALUDE_square_sequence_formulas_l1883_188396


namespace NUMINAMATH_CALUDE_min_value_xy_l1883_188331

theorem min_value_xy (x y : ℝ) (hx : x > 1) (hy : y > 1) (h_log : Real.log x * Real.log y = Real.log 3) :
  ∀ z, x * y ≥ z → z ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_xy_l1883_188331


namespace NUMINAMATH_CALUDE_unique_solution_square_equation_l1883_188371

theorem unique_solution_square_equation :
  ∃! x : ℝ, (2010 + x)^2 = x^2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_square_equation_l1883_188371


namespace NUMINAMATH_CALUDE_log_equation_solution_l1883_188322

theorem log_equation_solution (p q : ℝ) (h : 0 < p) (h' : 0 < q) :
  Real.log p + 2 * Real.log q = Real.log (2 * p + q) → p = q / (q^2 - 2) :=
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1883_188322


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_difference_l1883_188358

theorem cube_root_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (x₁ ≠ x₂) ∧ 
  ((9 - x₁^2 / 4)^(1/3 : ℝ) = -3) ∧ 
  ((9 - x₂^2 / 4)^(1/3 : ℝ) = -3) ∧ 
  (abs (x₁ - x₂) = 24) := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_difference_l1883_188358


namespace NUMINAMATH_CALUDE_distance_PQ_is_25_l1883_188344

/-- The distance between point P and the intersection point Q of lines l₁ and l₂ is 25. -/
theorem distance_PQ_is_25 
  (P : ℝ × ℝ)
  (l₁ : Set (ℝ × ℝ))
  (l₂ : Set (ℝ × ℝ))
  (Q : ℝ × ℝ)
  (h₁ : P = (3, 2))
  (h₂ : ∀ (x y : ℝ), (x, y) ∈ l₁ ↔ ∃ t, x = 3 + 4/5 * t ∧ y = 2 + 3/5 * t)
  (h₃ : ∀ (x y : ℝ), (x, y) ∈ l₂ ↔ x - 2*y + 11 = 0)
  (h₄ : Q ∈ l₁ ∧ Q ∈ l₂) :
  dist P Q = 25 := by
  sorry

#check distance_PQ_is_25

end NUMINAMATH_CALUDE_distance_PQ_is_25_l1883_188344


namespace NUMINAMATH_CALUDE_dynaco_shares_sold_is_150_l1883_188329

/-- Represents the stock portfolio problem --/
structure StockProblem where
  microtron_price : ℕ
  dynaco_price : ℕ
  total_shares : ℕ
  average_price : ℕ

/-- Calculates the number of Dynaco shares sold --/
def dynaco_shares_sold (p : StockProblem) : ℕ :=
  (p.total_shares * p.average_price - p.microtron_price * p.total_shares) / (p.dynaco_price - p.microtron_price)

/-- Theorem stating that given the problem conditions, 150 Dynaco shares were sold --/
theorem dynaco_shares_sold_is_150 (p : StockProblem) 
  (h1 : p.microtron_price = 36)
  (h2 : p.dynaco_price = 44)
  (h3 : p.total_shares = 300)
  (h4 : p.average_price = 40) :
  dynaco_shares_sold p = 150 := by
  sorry

#eval dynaco_shares_sold { microtron_price := 36, dynaco_price := 44, total_shares := 300, average_price := 40 }

end NUMINAMATH_CALUDE_dynaco_shares_sold_is_150_l1883_188329


namespace NUMINAMATH_CALUDE_bird_speed_theorem_l1883_188375

theorem bird_speed_theorem (d t : ℝ) (h1 : d = 50 * (t + 1/12)) (h2 : d = 70 * (t - 1/12)) :
  let r := d / t
  ∃ ε > 0, abs (r - 58) < ε :=
sorry

end NUMINAMATH_CALUDE_bird_speed_theorem_l1883_188375


namespace NUMINAMATH_CALUDE_average_score_is_two_average_score_independent_of_class_size_l1883_188370

/-- Represents the score distribution for a test -/
structure ScoreDistribution where
  threePoints : Real
  twoPoints : Real
  onePoint : Real
  zeroPoints : Real

/-- Calculates the average score given a score distribution -/
def averageScore (dist : ScoreDistribution) : Real :=
  3 * dist.threePoints + 2 * dist.twoPoints + 1 * dist.onePoint + 0 * dist.zeroPoints

/-- Theorem: The average score is 2 for the given score distribution -/
theorem average_score_is_two (dist : ScoreDistribution) 
    (h1 : dist.threePoints = 0.3)
    (h2 : dist.twoPoints = 0.5)
    (h3 : dist.onePoint = 0.1)
    (h4 : dist.zeroPoints = 0.1)
    (h5 : dist.threePoints + dist.twoPoints + dist.onePoint + dist.zeroPoints = 1) :
    averageScore dist = 2 := by
  sorry

/-- Corollary: The average score is independent of the number of students -/
theorem average_score_independent_of_class_size (n : Nat) (dist : ScoreDistribution) 
    (h1 : dist.threePoints = 0.3)
    (h2 : dist.twoPoints = 0.5)
    (h3 : dist.onePoint = 0.1)
    (h4 : dist.zeroPoints = 0.1)
    (h5 : dist.threePoints + dist.twoPoints + dist.onePoint + dist.zeroPoints = 1) :
    averageScore dist = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_score_is_two_average_score_independent_of_class_size_l1883_188370
