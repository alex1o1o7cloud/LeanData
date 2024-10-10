import Mathlib

namespace simplify_trig_expression_l494_49464

theorem simplify_trig_expression :
  2 * Real.sqrt (1 + Real.sin 4) + Real.sqrt (2 + 2 * Real.cos 4) = 2 * Real.sin 2 := by
  sorry

end simplify_trig_expression_l494_49464


namespace line_parameterization_l494_49404

/-- Given a line y = 2x - 40 parameterized by (x, y) = (f(t), 20t - 14),
    prove that f(t) = 10t + 13 -/
theorem line_parameterization (f : ℝ → ℝ) : 
  (∀ t : ℝ, 20 * t - 14 = 2 * (f t) - 40) → 
  (∀ t : ℝ, f t = 10 * t + 13) := by
sorry

end line_parameterization_l494_49404


namespace distance_to_cemetery_l494_49454

/-- The distance from the school to the Martyrs' Cemetery in kilometers. -/
def distance : ℝ := 216

/-- The original scheduled time for the journey in minutes. -/
def scheduled_time : ℝ := 180

/-- The time saved in minutes when the bus increases speed by 1/5 after 1 hour. -/
def time_saved_1 : ℝ := 20

/-- The time saved in minutes when the bus increases speed by 1/3 after 72 km. -/
def time_saved_2 : ℝ := 30

/-- The distance traveled at original speed before increasing speed by 1/3. -/
def initial_distance : ℝ := 72

theorem distance_to_cemetery :
  (1 + 1/5) * (scheduled_time - 60 - time_saved_1) = scheduled_time - 60 ∧
  (1 + 1/3) * (scheduled_time - time_saved_2) = scheduled_time ∧
  distance = initial_distance / (1 - 2/3) := by
  sorry

end distance_to_cemetery_l494_49454


namespace sin_150_degrees_l494_49425

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end sin_150_degrees_l494_49425


namespace incompatible_inequalities_l494_49499

theorem incompatible_inequalities :
  ¬∃ (a b c d : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
    a + b < c + d ∧
    (a + b) * (c + d) < a * b + c * d ∧
    (a + b) * c * d < a * b * (c + d) := by
  sorry

end incompatible_inequalities_l494_49499


namespace percentage_equality_l494_49423

theorem percentage_equality (x : ℝ) (h : x > 0) :
  ∃ p : ℝ, p / 100 * (x + 20) = 0.3 * (0.6 * x) ∧ p = 1800 * x / (x + 20) := by
  sorry

end percentage_equality_l494_49423


namespace complement_of_A_in_U_l494_49491

-- Define the universal set U
def U : Set ℝ := {x : ℝ | -3 < x ∧ x < 3}

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 + x - 2 < 0}

-- Theorem statement
theorem complement_of_A_in_U : 
  (U \ A) = {x : ℝ | (-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x < 3)} := by
  sorry

end complement_of_A_in_U_l494_49491


namespace work_completion_time_l494_49448

/-- The time it takes for A and B to complete a work together, given their individual work rates -/
def time_to_complete_together (rate_b rate_a : ℚ) : ℚ :=
  1 / (rate_a + rate_b)

/-- The proposition that A and B complete the work in 4 days under the given conditions -/
theorem work_completion_time 
  (rate_b : ℚ) -- B's work rate
  (rate_a : ℚ) -- A's work rate
  (h1 : rate_b = 1 / 12) -- B completes the work in 12 days
  (h2 : rate_a = 2 * rate_b) -- A works twice as fast as B
  : time_to_complete_together rate_b rate_a = 4 := by
  sorry

#eval time_to_complete_together (1/12) (1/6)

end work_completion_time_l494_49448


namespace parabola_transformation_l494_49414

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := -x^2

/-- The transformed parabola function -/
def transformed_parabola (x : ℝ) : ℝ := -(x - 2)^2 - 3

/-- Theorem stating that the transformed parabola is equivalent to
    shifting the original parabola 2 units right and 3 units down -/
theorem parabola_transformation :
  ∀ x : ℝ, transformed_parabola x = original_parabola (x - 2) - 3 :=
by sorry

end parabola_transformation_l494_49414


namespace absolute_value_theorem_l494_49490

theorem absolute_value_theorem (a : ℝ) (h : a = -1) : |a + 3| = 2 := by
  sorry

end absolute_value_theorem_l494_49490


namespace stadium_problem_l494_49484

theorem stadium_problem (total_start : ℕ) (total_end : ℕ) 
  (h1 : total_start = 600)
  (h2 : total_end = 480) :
  ∃ (boys girls : ℕ),
    boys + girls = total_start ∧
    boys - boys / 4 + girls - girls / 8 = total_end ∧
    girls = 240 := by
  sorry

end stadium_problem_l494_49484


namespace task_completion_theorem_l494_49488

/-- Represents the number of workers and days to complete a task. -/
structure WorkerDays where
  workers : ℕ
  days : ℕ

/-- Represents the conditions of the problem. -/
structure TaskConditions where
  original : WorkerDays
  reduced : WorkerDays
  increased : WorkerDays

/-- The theorem to prove based on the given conditions. -/
theorem task_completion_theorem (conditions : TaskConditions) : 
  conditions.original.workers = 60 ∧ conditions.original.days = 10 :=
by
  have h1 : conditions.reduced.workers = conditions.original.workers - 20 := by sorry
  have h2 : conditions.reduced.days = conditions.original.days + 5 := by sorry
  have h3 : conditions.increased.workers = conditions.original.workers + 15 := by sorry
  have h4 : conditions.increased.days = conditions.original.days - 2 := by sorry
  have h5 : conditions.original.workers * conditions.original.days = 
            conditions.reduced.workers * conditions.reduced.days := by sorry
  have h6 : conditions.original.workers * conditions.original.days = 
            conditions.increased.workers * conditions.increased.days := by sorry
  sorry

end task_completion_theorem_l494_49488


namespace intersection_line_l494_49418

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := (x+4)^2 + (y+3)^2 = 8

-- Define the line
def line (x y : ℝ) : Prop := 4*x + 3*y + 13 = 0

-- Theorem statement
theorem intersection_line :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end intersection_line_l494_49418


namespace partner_investment_period_l494_49474

/-- Given two partners p and q with investment ratio 7:5 and profit ratio 7:10,
    where q invests for 14 months, prove that p invests for 7 months. -/
theorem partner_investment_period
  (investment_ratio : ℚ) -- Ratio of p's investment to q's investment
  (profit_ratio : ℚ) -- Ratio of p's profit to q's profit
  (q_period : ℕ) -- Investment period of partner q in months
  (h1 : investment_ratio = 7 / 5)
  (h2 : profit_ratio = 7 / 10)
  (h3 : q_period = 14) :
  ∃ (p_period : ℕ), p_period = 7 ∧ 
    (investment_ratio * p_period) / (q_period : ℚ) = profit_ratio :=
by sorry

end partner_investment_period_l494_49474


namespace fraction_subtraction_l494_49457

theorem fraction_subtraction : 7 - (2 / 5)^3 = 867 / 125 := by sorry

end fraction_subtraction_l494_49457


namespace midpoint_square_area_l494_49406

/-- Given a square with area 100, prove that a smaller square formed by 
    connecting the midpoints of the sides of the larger square has an area of 25. -/
theorem midpoint_square_area (large_square : Real × Real → Real × Real) 
  (h_area : (large_square (1, 1) - large_square (0, 0)).1 ^ 2 = 100) :
  let small_square := fun (t : Real × Real) => 
    ((large_square (t.1, t.2) + large_square (t.1 + 1, t.2 + 1)) : Real × Real) / 2
  (small_square (1, 1) - small_square (0, 0)).1 ^ 2 = 25 := by
  sorry


end midpoint_square_area_l494_49406


namespace absolute_value_inequality_solution_range_l494_49467

theorem absolute_value_inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, |x + 1| - |x - 2| ≥ a^2 - 4*a) ↔ a ∈ Set.Icc 1 3 := by
  sorry

end absolute_value_inequality_solution_range_l494_49467


namespace dance_attendance_problem_l494_49444

/-- Represents the number of different dance pairs at a school dance. -/
def total_pairs : ℕ := 430

/-- Represents the number of boys the first girl danced with. -/
def first_girl_partners : ℕ := 12

/-- Calculates the number of boys a girl danced with based on her position. -/
def partners_for_girl (girl_position : ℕ) : ℕ :=
  first_girl_partners + girl_position - 1

/-- Calculates the total number of dance pairs for a given number of girls. -/
def sum_of_pairs (num_girls : ℕ) : ℕ :=
  (num_girls * (2 * first_girl_partners + num_girls - 1)) / 2

/-- Represents the problem of finding the number of girls and boys at the dance. -/
theorem dance_attendance_problem :
  ∃ (num_girls num_boys : ℕ),
    num_girls > 0 ∧
    num_boys = partners_for_girl num_girls ∧
    sum_of_pairs num_girls = total_pairs ∧
    num_girls = 20 ∧
    num_boys = 31 := by
  sorry

end dance_attendance_problem_l494_49444


namespace only_two_is_sum_of_squares_among_repeating_twos_l494_49435

def is_repeating_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * (10^k - 1) / 9

def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a^2 + b^2

theorem only_two_is_sum_of_squares_among_repeating_twos :
  ∀ n : ℕ, is_repeating_two n → (is_sum_of_two_squares n ↔ n = 2) :=
by sorry

end only_two_is_sum_of_squares_among_repeating_twos_l494_49435


namespace percent_of_a_is_4b_l494_49434

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.8 * b) : 
  (4 * b) / a * 100 = 222.22222222222223 := by
  sorry

end percent_of_a_is_4b_l494_49434


namespace additional_investment_rate_barbata_investment_rate_l494_49432

/-- Calculates the interest rate of an additional investment given initial investment conditions and total annual income. -/
theorem additional_investment_rate (initial_investment : ℝ) (initial_rate : ℝ) (total_rate : ℝ) (total_income : ℝ) : ℝ :=
  let additional_investment := (total_income - initial_investment * total_rate) / (total_rate - initial_rate)
  let additional_income := total_income - initial_investment * initial_rate
  additional_income / additional_investment

/-- Proves that the interest rate of the additional investment is approximately 6.13% given the specified conditions. -/
theorem barbata_investment_rate : 
  let initial_investment : ℝ := 2200
  let initial_rate : ℝ := 0.05
  let total_rate : ℝ := 0.06
  let total_income : ℝ := 1099.9999999999998
  abs (additional_investment_rate initial_investment initial_rate total_rate total_income - 0.0613) < 0.0001 := by
  sorry

end additional_investment_rate_barbata_investment_rate_l494_49432


namespace tensor_inequality_range_l494_49413

def tensor (x y : ℝ) : ℝ := x * (1 - y)

theorem tensor_inequality_range (a : ℝ) : 
  (∀ x : ℝ, tensor (x - a) (x + a) < 1) ↔ a ∈ Set.Ioo (-1/2) (3/2) :=
sorry

end tensor_inequality_range_l494_49413


namespace last_two_digits_sum_l494_49415

theorem last_two_digits_sum (n : ℕ) : n = 23 →
  (7^n + 13^n) % 100 = 40 := by
  sorry

end last_two_digits_sum_l494_49415


namespace subtraction_preserves_inequality_l494_49462

theorem subtraction_preserves_inequality (a b : ℝ) : a > b → a - 1 > b - 1 := by
  sorry

end subtraction_preserves_inequality_l494_49462


namespace product_correction_l494_49439

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

theorem product_correction (a b : ℕ) : 
  (10 ≤ a ∧ a < 100) →  -- a is a two-digit number
  (reverse_digits a * b = 182) →  -- reversed a multiplied by b is 182
  (a * b = 533) :=  -- the correct product is 533
by sorry

end product_correction_l494_49439


namespace no_entangled_numbers_l494_49475

/-- A two-digit positive integer is entangled if it equals twice the sum of its nonzero tens digit and the cube of its units digit -/
def is_entangled (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ ∃ (a b : ℕ), a > 0 ∧ b < 10 ∧ n = 10 * a + b ∧ n = 2 * (a + b^3)

/-- There are no entangled two-digit positive integers -/
theorem no_entangled_numbers : ¬∃ (n : ℕ), is_entangled n := by
  sorry

end no_entangled_numbers_l494_49475


namespace money_left_after_sale_l494_49451

/-- Represents the total revenue from selling items in a store's inventory. -/
def total_revenue (
  category_a_items : ℕ)
  (category_b_items : ℕ)
  (category_c_items : ℕ)
  (category_a_price : ℚ)
  (category_b_price : ℚ)
  (category_c_price : ℚ)
  (category_a_discount : ℚ)
  (category_b_discount : ℚ)
  (category_c_discount : ℚ)
  (category_a_sold_percent : ℚ)
  (category_b_sold_percent : ℚ)
  (category_c_sold_percent : ℚ) : ℚ :=
  (category_a_items : ℚ) * category_a_price * (1 - category_a_discount) * category_a_sold_percent +
  (category_b_items : ℚ) * category_b_price * (1 - category_b_discount) * category_b_sold_percent +
  (category_c_items : ℚ) * category_c_price * (1 - category_c_discount) * category_c_sold_percent

/-- Theorem stating the amount of money left after the sale and paying creditors. -/
theorem money_left_after_sale : 
  total_revenue 1000 700 300 50 75 100 0.8 0.7 0.6 0.85 0.75 0.9 - 15000 = 16112.5 := by
  sorry

end money_left_after_sale_l494_49451


namespace basketball_team_subjects_l494_49455

theorem basketball_team_subjects (P C B : Finset Nat) : 
  (P ∪ C ∪ B).card = 18 →
  P.card = 10 →
  B.card = 7 →
  C.card = 5 →
  (P ∩ B).card = 3 →
  (B ∩ C).card = 2 →
  (P ∩ C).card = 1 →
  (P ∩ C ∩ B).card = 2 := by
sorry

end basketball_team_subjects_l494_49455


namespace sine_cosine_power_inequality_l494_49481

theorem sine_cosine_power_inequality (n m : ℕ) (hn : n > 0) (hm : m > 0) (hnm : n > m) :
  ∀ x : ℝ, 0 < x ∧ x < π / 2 →
    2 * |Real.sin x ^ n - Real.cos x ^ n| ≤ 3 * |Real.sin x ^ m - Real.cos x ^ m| :=
by sorry

end sine_cosine_power_inequality_l494_49481


namespace scramble_word_count_l494_49468

/-- The number of letters in the extended Kobish alphabet -/
def alphabet_size : ℕ := 21

/-- The maximum length of a word -/
def max_word_length : ℕ := 4

/-- Calculates the number of words of a given length that contain the letter B at least once -/
def words_with_b (length : ℕ) : ℕ :=
  alphabet_size^length - (alphabet_size - 1)^length

/-- The total number of valid words in the Scramble language -/
def total_valid_words : ℕ :=
  words_with_b 1 + words_with_b 2 + words_with_b 3 + words_with_b 4

theorem scramble_word_count : total_valid_words = 35784 := by
  sorry

end scramble_word_count_l494_49468


namespace smallest_multiple_l494_49438

theorem smallest_multiple (x : ℕ) : x = 432 ↔ 
  (x > 0 ∧ 500 * x % 864 = 0 ∧ ∀ y : ℕ, y > 0 → 500 * y % 864 = 0 → x ≤ y) := by
  sorry

end smallest_multiple_l494_49438


namespace rectangle_width_l494_49477

theorem rectangle_width (length area : ℚ) (h1 : length = 3/5) (h2 : area = 1/3) :
  area / length = 5/9 := by
  sorry

end rectangle_width_l494_49477


namespace subtract_fractions_l494_49405

theorem subtract_fractions : (2 : ℚ) / 3 - 5 / 12 = 1 / 4 := by
  sorry

end subtract_fractions_l494_49405


namespace last_two_pieces_l494_49449

def pieces : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_odd (n : Nat) : Bool := n % 2 = 1

def product_is_24 (s : Finset Nat) : Bool :=
  s.prod id = 24

def removal_process (s : Finset Nat) : Finset Nat :=
  let after_odd_removal := s.filter (fun n => ¬is_odd n)
  let after_product_removal := after_odd_removal.filter (fun n => ¬product_is_24 {n})
  after_product_removal

theorem last_two_pieces (s : Finset Nat) :
  s = pieces →
  (removal_process s = {2, 8} ∨ removal_process s = {6, 8}) :=
sorry

end last_two_pieces_l494_49449


namespace simple_interest_principal_l494_49459

/-- Simple interest calculation -/
theorem simple_interest_principal (rate : ℚ) (time : ℚ) (interest : ℚ) (principal : ℚ) : 
  rate = 6/100 →
  time = 4 →
  interest = 192 →
  principal * rate * time = interest →
  principal = 800 := by
sorry

end simple_interest_principal_l494_49459


namespace problem_solvers_equal_girls_l494_49422

/-- Given a class of students, prove that the number of students who solved a problem
    is equal to the total number of girls, given that the number of boys who solved
    the problem is equal to the number of girls who did not solve it. -/
theorem problem_solvers_equal_girls (total : ℕ) (boys girls : ℕ) 
    (boys_solved girls_solved : ℕ) : 
    boys + girls = total →
    boys_solved = girls - girls_solved →
    boys_solved + girls_solved = girls := by
  sorry

end problem_solvers_equal_girls_l494_49422


namespace first_product_of_98_l494_49409

/-- The first product of the digits of a two-digit number -/
def first_digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

/-- Theorem: The first product of the digits of 98 is 72 -/
theorem first_product_of_98 : first_digit_product 98 = 72 := by
  sorry

end first_product_of_98_l494_49409


namespace product_sum_theorem_l494_49482

theorem product_sum_theorem : ∃ (a b c : ℕ), 
  a ∈ ({1, 2, 4, 8, 16, 20} : Set ℕ) ∧ 
  b ∈ ({1, 2, 4, 8, 16, 20} : Set ℕ) ∧ 
  c ∈ ({1, 2, 4, 8, 16, 20} : Set ℕ) ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a * b * c = 80 ∧
  a + b + c = 25 := by
  sorry

end product_sum_theorem_l494_49482


namespace largest_number_with_13_matchsticks_has_digit_sum_9_l494_49443

/-- Represents the number of matchsticks needed to form each digit --/
def matchsticks_per_digit : Fin 10 → ℕ
| 0 => 6
| 1 => 2
| 2 => 5
| 3 => 5
| 4 => 4
| 5 => 5
| 6 => 6
| 7 => 3
| 8 => 7
| 9 => 6

/-- Represents a number as a list of digits --/
def Number := List (Fin 10)

/-- Calculates the sum of digits in a number --/
def sum_of_digits (n : Number) : ℕ :=
  n.foldl (fun acc d => acc + d.val) 0

/-- Calculates the total number of matchsticks used to form a number --/
def matchsticks_used (n : Number) : ℕ :=
  n.foldl (fun acc d => acc + matchsticks_per_digit d) 0

/-- Checks if a number is valid (uses exactly 13 matchsticks) --/
def is_valid_number (n : Number) : Prop :=
  matchsticks_used n = 13

/-- Compares two numbers lexicographically --/
def number_gt (a b : Number) : Prop :=
  match a, b with
  | [], [] => False
  | _ :: _, [] => True
  | [], _ :: _ => False
  | x :: xs, y :: ys => x > y ∨ (x = y ∧ number_gt xs ys)

/-- The main theorem to be proved --/
theorem largest_number_with_13_matchsticks_has_digit_sum_9 :
  ∃ (n : Number), is_valid_number n ∧
    (∀ (m : Number), is_valid_number m → number_gt n m ∨ n = m) ∧
    sum_of_digits n = 9 := by
  sorry

end largest_number_with_13_matchsticks_has_digit_sum_9_l494_49443


namespace farm_animal_count_l494_49452

/-- Represents the distribution of animals on a farm --/
structure FarmDistribution where
  chicken_coops : List Nat
  duck_coops : List Nat
  geese_coop : Nat
  quail_coop : Nat
  turkey_coops : List Nat
  cow_sheds : List Nat
  pig_sections : List Nat

/-- Calculates the total number of animals on the farm --/
def total_animals (farm : FarmDistribution) : Nat :=
  (farm.chicken_coops.sum + farm.duck_coops.sum + farm.geese_coop + 
   farm.quail_coop + farm.turkey_coops.sum + farm.cow_sheds.sum + 
   farm.pig_sections.sum)

/-- Theorem stating that the total number of animals on the farm is 431 --/
theorem farm_animal_count (farm : FarmDistribution) 
  (h1 : farm.chicken_coops = [60, 45, 55])
  (h2 : farm.duck_coops = [40, 35])
  (h3 : farm.geese_coop = 20)
  (h4 : farm.quail_coop = 50)
  (h5 : farm.turkey_coops = [10, 10])
  (h6 : farm.cow_sheds = [20, 10, 6])
  (h7 : farm.pig_sections = [15, 25, 30, 0]) :
  total_animals farm = 431 := by
  sorry

#eval total_animals {
  chicken_coops := [60, 45, 55],
  duck_coops := [40, 35],
  geese_coop := 20,
  quail_coop := 50,
  turkey_coops := [10, 10],
  cow_sheds := [20, 10, 6],
  pig_sections := [15, 25, 30, 0]
}

end farm_animal_count_l494_49452


namespace unique_number_between_cubes_l494_49410

theorem unique_number_between_cubes : ∃! (n : ℕ), 
  n > 0 ∧ 
  24 ∣ n ∧ 
  (9 : ℝ) < n^(1/3) ∧ 
  n^(1/3) < (9.1 : ℝ) ∧ 
  n = 744 := by sorry

end unique_number_between_cubes_l494_49410


namespace arithmetic_calculations_l494_49497

theorem arithmetic_calculations : 
  ((1 : ℝ) + 4 - (-7) + (-8) = 3) ∧ 
  (-8.9 - (-4.7) + 7.5 = 3.3) := by
sorry

end arithmetic_calculations_l494_49497


namespace shaniqua_haircuts_l494_49430

/-- Represents the pricing and earnings of a hairstylist --/
structure HairstylistEarnings where
  haircut_price : ℕ
  style_price : ℕ
  total_earnings : ℕ
  num_styles : ℕ

/-- Calculates the number of haircuts given the hairstylist's earnings information --/
def calculate_haircuts (e : HairstylistEarnings) : ℕ :=
  (e.total_earnings - e.style_price * e.num_styles) / e.haircut_price

/-- Theorem stating that given Shaniqua's earnings information, she gave 8 haircuts --/
theorem shaniqua_haircuts :
  let e : HairstylistEarnings := {
    haircut_price := 12,
    style_price := 25,
    total_earnings := 221,
    num_styles := 5
  }
  calculate_haircuts e = 8 := by sorry

end shaniqua_haircuts_l494_49430


namespace fraction_equality_l494_49436

theorem fraction_equality : (2 + 4 - 8 + 16 + 32 - 64 + 128) / (4 + 8 - 16 + 32 + 64 - 128 + 256) = 1 / 2 := by
  sorry

end fraction_equality_l494_49436


namespace divisor_proof_l494_49486

theorem divisor_proof (original : Nat) (added : Nat) (sum : Nat) (h1 : original = 859622) (h2 : added = 859560) (h3 : sum = original + added) :
  sum % added = 0 ∧ added ≠ 0 := by
  sorry

end divisor_proof_l494_49486


namespace marta_worked_19_hours_l494_49445

/-- Calculates the number of hours worked given total money collected, hourly wage, and total tips collected. -/
def hours_worked (total_money : ℕ) (hourly_wage : ℕ) (total_tips : ℕ) : ℕ :=
  (total_money - total_tips) / hourly_wage

theorem marta_worked_19_hours :
  hours_worked 240 10 50 = 19 := by
sorry

end marta_worked_19_hours_l494_49445


namespace clothing_selection_probability_l494_49446

/-- The number of shirts in the drawer -/
def num_shirts : ℕ := 6

/-- The number of pairs of shorts in the drawer -/
def num_shorts : ℕ := 3

/-- The number of pairs of socks in the drawer -/
def num_socks : ℕ := 8

/-- The total number of articles of clothing in the drawer -/
def total_items : ℕ := num_shirts + num_shorts + num_socks

/-- The number of articles of clothing to be randomly selected -/
def num_selected : ℕ := 4

/-- The probability of selecting at least one shirt, exactly one pair of shorts, and one pair of socks -/
theorem clothing_selection_probability : 
  (Nat.choose num_shorts 1 * Nat.choose num_socks 1 * 
   (Nat.choose num_shirts 2 + Nat.choose num_shirts 1)) / 
  Nat.choose total_items num_selected = 84 / 397 := by sorry

end clothing_selection_probability_l494_49446


namespace optimal_soap_cost_l494_49408

/-- Represents the discount percentage based on the number of bars purchased -/
def discount (bars : ℕ) : ℚ :=
  if bars ≥ 8 then 15/100
  else if bars ≥ 6 then 10/100
  else if bars ≥ 4 then 5/100
  else 0

/-- Calculates the cost of soap for a year -/
def soap_cost (price_per_bar : ℚ) (months_per_bar : ℕ) (months_in_year : ℕ) : ℚ :=
  let bars_needed := months_in_year / months_per_bar
  let total_cost := price_per_bar * bars_needed
  total_cost * (1 - discount bars_needed)

theorem optimal_soap_cost :
  soap_cost 8 2 12 = 432/10 :=
sorry

end optimal_soap_cost_l494_49408


namespace smallest_integer_theorem_l494_49480

def is_divisible (n m : ℕ) : Prop := m ∣ n

def smallest_integer_with_divisors (excluded : List ℕ) : ℕ :=
  let divisors := (List.range 31).filter (λ x => x ∉ excluded)
  divisors.foldl Nat.lcm 1

theorem smallest_integer_theorem :
  let n := smallest_integer_with_divisors [17, 19]
  (∀ k ∈ List.range 31, k ≠ 17 → k ≠ 19 → is_divisible n k) ∧
  (∀ m < n, ∃ k ∈ List.range 31, k ≠ 17 ∧ k ≠ 19 ∧ ¬is_divisible m k) ∧
  n = 122522400 := by
sorry

end smallest_integer_theorem_l494_49480


namespace average_wage_calculation_l494_49495

/-- Calculates the average wage per day paid by a contractor given the number of workers and their wages. -/
theorem average_wage_calculation
  (male_workers female_workers child_workers : ℕ)
  (male_wage female_wage child_wage : ℚ)
  (h_male : male_workers = 20)
  (h_female : female_workers = 15)
  (h_child : child_workers = 5)
  (h_male_wage : male_wage = 35)
  (h_female_wage : female_wage = 20)
  (h_child_wage : child_wage = 8) :
  (male_workers * male_wage + female_workers * female_wage + child_workers * child_wage) /
  (male_workers + female_workers + child_workers : ℚ) = 26 := by
  sorry

end average_wage_calculation_l494_49495


namespace toothpick_pattern_15th_stage_l494_49487

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem toothpick_pattern_15th_stage :
  arithmetic_sequence 5 3 15 = 47 := by
  sorry

end toothpick_pattern_15th_stage_l494_49487


namespace isosceles_right_triangle_c_coords_l494_49403

/-- An isosceles right triangle in 2D space -/
structure IsoscelesRightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  isIsosceles : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2
  isRight : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0

/-- Theorem: The coordinates of C in the given isosceles right triangle -/
theorem isosceles_right_triangle_c_coords :
  ∀ t : IsoscelesRightTriangle,
  t.A = (1, 0) → t.B = (3, 1) →
  t.C = (2, 3) ∨ t.C = (4, -1) := by
  sorry

end isosceles_right_triangle_c_coords_l494_49403


namespace gcd_count_for_product_360_l494_49427

theorem gcd_count_for_product_360 (a b : ℕ+) : 
  (Nat.gcd a b * Nat.lcm a b = 360) → 
  (∃ (S : Finset ℕ+), S.card = 11 ∧ (∀ x, x ∈ S ↔ ∃ c d : ℕ+, (Nat.gcd c d * Nat.lcm c d = 360) ∧ Nat.gcd c d = x)) :=
sorry

end gcd_count_for_product_360_l494_49427


namespace solve_system_l494_49494

theorem solve_system (x y : ℝ) 
  (eq1 : 2 * x - y = 5) 
  (eq2 : x + 2 * y = 5) : 
  x = 3 := by
sorry

end solve_system_l494_49494


namespace waiter_tip_problem_l494_49420

/-- Calculates the tip amount per customer given the total customers, non-tipping customers, and total tips. -/
def tip_per_customer (total_customers : ℕ) (non_tipping_customers : ℕ) (total_tips : ℚ) : ℚ :=
  total_tips / (total_customers - non_tipping_customers)

/-- Proves that given 10 total customers, 5 non-tipping customers, and $15 total tips, 
    the amount each tipping customer gave is $3. -/
theorem waiter_tip_problem :
  tip_per_customer 10 5 15 = 3 := by
  sorry

end waiter_tip_problem_l494_49420


namespace cyclic_quadrilateral_extreme_angles_l494_49440

/-- A cyclic quadrilateral with a specific angle ratio -/
structure CyclicQuadrilateral where
  -- Three consecutive angles with ratio 5:6:4
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  angle_ratio : a = 5 * (b / 6) ∧ c = 4 * (b / 6)
  -- Sum of opposite angles is 180°
  opposite_sum : a + d = 180 ∧ b + c = 180

/-- The largest and smallest angles in the cyclic quadrilateral -/
def extreme_angles (q : CyclicQuadrilateral) : ℝ × ℝ :=
  (108, 72)

theorem cyclic_quadrilateral_extreme_angles (q : CyclicQuadrilateral) :
  extreme_angles q = (108, 72) := by
  sorry

end cyclic_quadrilateral_extreme_angles_l494_49440


namespace hyperbola_iff_mn_neg_l494_49469

/-- Defines whether an equation represents a hyperbola -/
def is_hyperbola (m n : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / m + y^2 / n = 1 ∧ 
  ¬∃ (a b : ℝ), ∀ (x y : ℝ), x^2 / m + y^2 / n = 1 ↔ (x - a)^2 + (y - b)^2 = 1

/-- Proves that mn < 0 is necessary and sufficient for the equation to represent a hyperbola -/
theorem hyperbola_iff_mn_neg (m n : ℝ) :
  is_hyperbola m n ↔ m * n < 0 :=
sorry

end hyperbola_iff_mn_neg_l494_49469


namespace equation_solution_l494_49419

theorem equation_solution : 
  let f (x : ℂ) := (4 * x^3 + 4 * x^2 + 3 * x + 2) / (x - 2)
  let g (x : ℂ) := 4 * x^2 + 5 * x + 4
  let sol₁ : ℂ := (-9 + Complex.I * Real.sqrt 79) / 8
  let sol₂ : ℂ := (-9 - Complex.I * Real.sqrt 79) / 8
  (∀ x : ℂ, x ≠ 2 → f x = g x) → (f sol₁ = g sol₁ ∧ f sol₂ = g sol₂) := by
sorry

end equation_solution_l494_49419


namespace soap_box_length_proof_l494_49492

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (box : BoxDimensions) : ℝ :=
  box.length * box.width * box.height

theorem soap_box_length_proof 
  (carton : BoxDimensions) 
  (soap_box : BoxDimensions) 
  (max_boxes : ℕ) :
  carton.length = 25 ∧ 
  carton.width = 42 ∧ 
  carton.height = 60 ∧
  soap_box.width = 6 ∧ 
  soap_box.height = 10 ∧
  max_boxes = 150 ∧
  (max_boxes : ℝ) * boxVolume soap_box = boxVolume carton →
  soap_box.length = 7 := by
sorry

end soap_box_length_proof_l494_49492


namespace sum_of_A_and_B_l494_49472

theorem sum_of_A_and_B : ∀ (A B : ℚ), 3/7 = 6/A ∧ 6/A = B/21 → A + B = 23 := by
  sorry

end sum_of_A_and_B_l494_49472


namespace irregular_quadrilateral_tiles_plane_l494_49453

-- Define an irregular quadrilateral
structure IrregularQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ

-- Define a tiling of the plane
def PlaneTiling (Q : Type) := ℝ × ℝ → Q

-- Define the property of being a valid tiling (no gaps or overlaps)
def IsValidTiling (Q : Type) (tiling : PlaneTiling Q) : Prop := sorry

-- Theorem statement
theorem irregular_quadrilateral_tiles_plane (q : IrregularQuadrilateral) :
  ∃ (tiling : PlaneTiling IrregularQuadrilateral), IsValidTiling IrregularQuadrilateral tiling :=
sorry

end irregular_quadrilateral_tiles_plane_l494_49453


namespace six_students_three_competitions_l494_49478

/-- The number of ways to assign students to competitions -/
def registration_methods (num_students : ℕ) (num_competitions : ℕ) : ℕ :=
  num_competitions ^ num_students

/-- Theorem: The number of ways to assign 6 students to 3 competitions is 729 -/
theorem six_students_three_competitions :
  registration_methods 6 3 = 729 := by
  sorry

end six_students_three_competitions_l494_49478


namespace shorts_weight_l494_49442

/-- The maximum allowed weight for washing clothes -/
def max_weight : ℕ := 50

/-- The weight of a pair of socks in ounces -/
def sock_weight : ℕ := 2

/-- The weight of a pair of underwear in ounces -/
def underwear_weight : ℕ := 4

/-- The weight of a shirt in ounces -/
def shirt_weight : ℕ := 5

/-- The weight of a pair of pants in ounces -/
def pants_weight : ℕ := 10

/-- The number of pairs of pants Tony is washing -/
def num_pants : ℕ := 1

/-- The number of shirts Tony is washing -/
def num_shirts : ℕ := 2

/-- The number of pairs of socks Tony is washing -/
def num_socks : ℕ := 3

/-- The number of additional pairs of underwear Tony can add -/
def additional_underwear : ℕ := 4

/-- Theorem stating that the weight of a pair of shorts is 8 ounces -/
theorem shorts_weight :
  ∃ (shorts_weight : ℕ),
    shorts_weight = max_weight -
      (num_pants * pants_weight +
       num_shirts * shirt_weight +
       num_socks * sock_weight +
       additional_underwear * underwear_weight) :=
by sorry

end shorts_weight_l494_49442


namespace go_games_theorem_l494_49450

/-- The number of complete Go games that can be played simultaneously -/
def maxSimultaneousGames (totalBalls : ℕ) (ballsPerGame : ℕ) : ℕ :=
  totalBalls / ballsPerGame

theorem go_games_theorem :
  maxSimultaneousGames 901 53 = 17 := by
  sorry

end go_games_theorem_l494_49450


namespace overlapping_area_l494_49433

theorem overlapping_area (total_length : ℝ) (left_length right_length : ℝ) 
  (left_only_area right_only_area : ℝ) : 
  total_length = 16 →
  left_length = 9 →
  right_length = 7 →
  left_only_area = 27 →
  right_only_area = 18 →
  ∃ (overlap_area : ℝ), 
    overlap_area = 13.5 ∧
    left_length / right_length = (left_only_area + overlap_area) / (right_only_area + overlap_area) :=
by sorry

end overlapping_area_l494_49433


namespace base_height_ratio_l494_49421

/-- Represents a triangular field with specific properties -/
structure TriangularField where
  base : ℝ
  height : ℝ
  cultivation_cost : ℝ
  cost_per_hectare : ℝ
  base_multiple_of_height : ∃ k : ℝ, base = k * height
  total_cost : cultivation_cost = 333.18
  cost_rate : cost_per_hectare = 24.68
  base_value : base = 300
  height_value : height = 300

/-- Theorem stating that the ratio of base to height is 1:1 for the given triangular field -/
theorem base_height_ratio (field : TriangularField) : field.base / field.height = 1 := by
  sorry

#check base_height_ratio

end base_height_ratio_l494_49421


namespace max_value_theorem_l494_49417

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^2 + b^2 - Real.sqrt 3 * a * b = 1) : 
  Real.sqrt 3 * a^2 - a * b ≤ 2 + Real.sqrt 3 := by
sorry

end max_value_theorem_l494_49417


namespace factorization_proof_l494_49460

theorem factorization_proof (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) := by
  sorry

end factorization_proof_l494_49460


namespace dormitory_to_city_distance_dormitory_to_city_distance_proof_l494_49412

theorem dormitory_to_city_distance : ℝ → Prop :=
  fun total_distance =>
    (1/6 : ℝ) * total_distance +
    (1/4 : ℝ) * total_distance +
    (1/3 : ℝ) * total_distance +
    10 +
    (1/12 : ℝ) * total_distance = total_distance →
    total_distance = 60

-- The proof is omitted
theorem dormitory_to_city_distance_proof : dormitory_to_city_distance 60 := by
  sorry

end dormitory_to_city_distance_dormitory_to_city_distance_proof_l494_49412


namespace largest_four_digit_sum_20_l494_49479

/-- A four-digit number is a natural number between 1000 and 9999, inclusive. -/
def FourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

/-- Theorem stating that 9920 is the largest four-digit number whose digits sum to 20 -/
theorem largest_four_digit_sum_20 :
  FourDigitNumber 9920 ∧
  sumOfDigits 9920 = 20 ∧
  ∀ n : ℕ, FourDigitNumber n → sumOfDigits n = 20 → n ≤ 9920 :=
sorry

end largest_four_digit_sum_20_l494_49479


namespace arithmetic_sequence_middle_term_l494_49483

theorem arithmetic_sequence_middle_term (a : ℕ → ℝ) :
  (a 0 = 3^2) →
  (a 2 = 3^4) →
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →
  (a 1 = 45) :=
by sorry

end arithmetic_sequence_middle_term_l494_49483


namespace least_product_of_primes_above_30_l494_49437

theorem least_product_of_primes_above_30 :
  ∃ p q : ℕ, 
    Prime p ∧ Prime q ∧ 
    p > 30 ∧ q > 30 ∧ 
    p ≠ q ∧
    ∀ r s : ℕ, Prime r → Prime s → r > 30 → s > 30 → r ≠ s → p * q ≤ r * s :=
by sorry

end least_product_of_primes_above_30_l494_49437


namespace m_range_l494_49402

/-- Proposition p: m is a real number and m + 1 ≤ 0 -/
def p (m : ℝ) : Prop := m + 1 ≤ 0

/-- Proposition q: For all real x, x² + mx + 1 > 0 -/
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 > 0

/-- The range of m satisfying the given conditions -/
theorem m_range (m : ℝ) : 
  (p m ∧ q m → False) →  -- p ∧ q is false
  (p m ∨ q m) →          -- p ∨ q is true
  (m ≤ -2 ∨ (-1 < m ∧ m < 2)) := by
sorry

end m_range_l494_49402


namespace arithmetic_evaluation_l494_49441

theorem arithmetic_evaluation : (300 + 5 * 8) / (2^3 : ℝ) = 42.5 := by sorry

end arithmetic_evaluation_l494_49441


namespace interest_problem_l494_49431

/-- Given a principal amount and an interest rate, prove that they satisfy the conditions for simple and compound interest over 2 years -/
theorem interest_problem (P R : ℝ) : 
  (P * R * 2 / 100 = 20) →  -- Simple interest condition
  (P * ((1 + R/100)^2 - 1) = 22) →  -- Compound interest condition
  (P = 50 ∧ R = 20) := by
sorry

end interest_problem_l494_49431


namespace complex_equation_solution_l494_49465

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I) = 2 * Complex.I → z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l494_49465


namespace initial_daily_production_l494_49428

/-- The number of days the company worked after the initial 3 days -/
def additional_days : ℕ := 20

/-- The total number of parts produced -/
def total_parts : ℕ := 675

/-- The number of extra parts produced beyond the plan -/
def extra_parts : ℕ := 100

/-- The daily increase in parts production after the initial 3 days -/
def daily_increase : ℕ := 5

theorem initial_daily_production :
  ∃ (x : ℕ),
    x > 0 ∧
    3 * x + additional_days * (x + daily_increase) = total_parts + extra_parts ∧
    x = 29 := by
  sorry

end initial_daily_production_l494_49428


namespace path_count_l494_49447

/-- The number of paths on a grid from A to B satisfying specific conditions -/
def number_of_paths : ℕ :=
  Nat.choose 8 4

/-- Theorem stating that the number of paths is 70 -/
theorem path_count : number_of_paths = 70 := by
  sorry

end path_count_l494_49447


namespace bowlingPrizeOrders_l494_49463

/-- Represents the number of bowlers in the tournament -/
def numBowlers : ℕ := 7

/-- Represents the number of playoff matches -/
def numMatches : ℕ := 6

/-- The number of possible outcomes for each match -/
def outcomesPerMatch : ℕ := 2

/-- Calculates the total number of possible prize orders -/
def totalPossibleOrders : ℕ := outcomesPerMatch ^ numMatches

/-- Proves that the number of different possible prize orders is 64 -/
theorem bowlingPrizeOrders : totalPossibleOrders = 64 := by
  sorry

end bowlingPrizeOrders_l494_49463


namespace bucket_capacity_change_l494_49426

theorem bucket_capacity_change (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 200 →
  capacity_ratio = 4/5 →
  (original_buckets : ℚ) / capacity_ratio = 250 := by
sorry

end bucket_capacity_change_l494_49426


namespace right_triangle_identification_l494_49493

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_identification :
  ¬ is_right_triangle 2 3 4 ∧
  ¬ is_right_triangle (Real.sqrt 3) (Real.sqrt 4) (Real.sqrt 5) ∧
  is_right_triangle 3 4 5 ∧
  ¬ is_right_triangle 7 8 9 :=
by sorry

end right_triangle_identification_l494_49493


namespace line_through_point_l494_49476

/-- 
Given a line equation 2 - kx = -4y that passes through the point (3, -2),
prove that k = -2.
-/
theorem line_through_point (k : ℝ) : 
  (2 - k * 3 = -4 * (-2)) → k = -2 := by
  sorry

end line_through_point_l494_49476


namespace m_eq_2_sufficient_not_necessary_l494_49473

def A (m : ℝ) : Set ℝ := {1, m^2}
def B : Set ℝ := {2, 4}

theorem m_eq_2_sufficient_not_necessary :
  (∃ m : ℝ, (A m) ∩ B = {4} ∧ m ≠ 2) ∧
  (∀ m : ℝ, m = 2 → (A m) ∩ B = {4}) :=
sorry

end m_eq_2_sufficient_not_necessary_l494_49473


namespace tangent_sum_problem_l494_49498

theorem tangent_sum_problem (p q : ℝ) 
  (h1 : (Real.sin p / Real.cos q) + (Real.sin q / Real.cos p) = 2)
  (h2 : (Real.cos p / Real.sin q) + (Real.cos q / Real.sin p) = 3) :
  (Real.tan p / Real.tan q) + (Real.tan q / Real.tan p) = 8/5 := by
  sorry

end tangent_sum_problem_l494_49498


namespace correct_average_calculation_l494_49461

def total_numbers : ℕ := 20
def initial_average : ℚ := 35
def incorrect_numbers : List (ℚ × ℚ) := [(90, 45), (73, 36), (85, 42), (-45, -27), (64, 35)]

theorem correct_average_calculation :
  let incorrect_sum := initial_average * total_numbers
  let adjustment := (incorrect_numbers.map (λ (x : ℚ × ℚ) => x.1 - x.2)).sum
  let correct_sum := incorrect_sum + adjustment
  correct_sum / total_numbers = 41.8 := by sorry

end correct_average_calculation_l494_49461


namespace scientific_notation_of_small_number_l494_49470

theorem scientific_notation_of_small_number :
  ∃ (a : ℝ) (n : ℤ), 0.00000000005 = a * 10^n ∧ 1 ≤ a ∧ a < 10 :=
by
  -- The proof goes here
  sorry

end scientific_notation_of_small_number_l494_49470


namespace frustum_slant_height_is_9_l494_49471

/-- Represents a cone cut by a plane parallel to its base, forming a frustum -/
structure ConeFrustum where
  -- Ratio of top to bottom surface areas
  area_ratio : ℝ
  -- Slant height of the removed cone
  removed_slant_height : ℝ

/-- Calculates the slant height of the frustum -/
def slant_height_frustum (cf : ConeFrustum) : ℝ :=
  sorry

/-- Theorem stating the slant height of the frustum is 9 given the conditions -/
theorem frustum_slant_height_is_9 (cf : ConeFrustum) 
  (h1 : cf.area_ratio = 1 / 16)
  (h2 : cf.removed_slant_height = 3) : 
  slant_height_frustum cf = 9 :=
sorry

end frustum_slant_height_is_9_l494_49471


namespace f_composition_of_three_l494_49407

def f (x : ℝ) : ℝ := -3 * x + 5

theorem f_composition_of_three : f (f (f 3)) = -46 := by sorry

end f_composition_of_three_l494_49407


namespace quadratic_has_real_roots_l494_49485

/-- The quadratic equation x^2 - 4x + 2a = 0 has real roots when a = 1 -/
theorem quadratic_has_real_roots : ∃ (x : ℝ), x^2 - 4*x + 2 = 0 := by
  sorry

end quadratic_has_real_roots_l494_49485


namespace equal_interest_rate_equal_interest_l494_49401

/-- The rate at which a principal of 200 invested for 12 years produces the same
    interest as 400 invested for 5 years at 12% annual interest rate -/
theorem equal_interest_rate : ℝ :=
  let principal1 : ℝ := 200
  let time1 : ℝ := 12
  let principal2 : ℝ := 400
  let time2 : ℝ := 5
  let rate2 : ℝ := 12 / 100
  let interest2 : ℝ := principal2 * rate2 * time2
  10 / 100

/-- Proof that the calculated rate produces equal interest -/
theorem equal_interest (rate : ℝ) (h : rate = equal_interest_rate) :
  200 * rate * 12 = 400 * (12 / 100) * 5 := by
  sorry

#check equal_interest
#check equal_interest_rate

end equal_interest_rate_equal_interest_l494_49401


namespace marissa_initial_ribbon_l494_49424

/-- The amount of ribbon used per box in feet -/
def ribbon_per_box : ℝ := 0.7

/-- The number of boxes Marissa tied -/
def num_boxes : ℕ := 5

/-- The amount of ribbon left after tying all boxes in feet -/
def ribbon_left : ℝ := 1

/-- The initial amount of ribbon Marissa had in feet -/
def initial_ribbon : ℝ := ribbon_per_box * num_boxes + ribbon_left

theorem marissa_initial_ribbon :
  initial_ribbon = 4.5 := by sorry

end marissa_initial_ribbon_l494_49424


namespace direct_inverse_variation_l494_49458

theorem direct_inverse_variation (k : ℝ) (R X Y : ℝ → ℝ) :
  (∀ t, R t = k * X t / Y t) →  -- R varies directly as X and inversely as Y
  R 0 = 10 ∧ X 0 = 2 ∧ Y 0 = 4 →  -- Initial condition
  R 1 = 8 ∧ Y 1 = 5 →  -- New condition
  X 1 = 2 :=  -- Conclusion
by sorry

end direct_inverse_variation_l494_49458


namespace team_savings_is_36_dollars_l494_49429

-- Define the prices and team size
def regular_shirt_price : ℝ := 7.50
def regular_pants_price : ℝ := 15.00
def regular_socks_price : ℝ := 4.50
def discounted_shirt_price : ℝ := 6.75
def discounted_pants_price : ℝ := 13.50
def discounted_socks_price : ℝ := 3.75
def team_size : ℕ := 12

-- Define the total savings function
def total_savings : ℝ :=
  let regular_uniform_price := regular_shirt_price + regular_pants_price + regular_socks_price
  let discounted_uniform_price := discounted_shirt_price + discounted_pants_price + discounted_socks_price
  let savings_per_uniform := regular_uniform_price - discounted_uniform_price
  savings_per_uniform * team_size

-- Theorem statement
theorem team_savings_is_36_dollars : total_savings = 36 := by
  sorry

end team_savings_is_36_dollars_l494_49429


namespace yeast_growth_30_minutes_l494_49466

/-- The number of yeast cells after a given number of 5-minute intervals -/
def yeast_population (initial_population : ℕ) (intervals : ℕ) : ℕ :=
  initial_population * 2^intervals

/-- Theorem: After 30 minutes (6 intervals), the yeast population will be 3200 -/
theorem yeast_growth_30_minutes :
  yeast_population 50 6 = 3200 := by
  sorry

end yeast_growth_30_minutes_l494_49466


namespace multiply_469158_and_9999_l494_49496

theorem multiply_469158_and_9999 : 469158 * 9999 = 4691176842 := by
  sorry

end multiply_469158_and_9999_l494_49496


namespace smallest_square_coverage_l494_49456

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℕ := r.width * r.height

/-- Calculates the area of a square -/
def squareArea (s : Square) : ℕ := s.side * s.side

/-- Checks if a square can be exactly covered by a given number of rectangles -/
def canCoverSquare (s : Square) (r : Rectangle) (n : ℕ) : Prop :=
  squareArea s = n * rectangleArea r

/-- The theorem to be proved -/
theorem smallest_square_coverage :
  ∃ (s : Square) (n : ℕ),
    let r : Rectangle := ⟨2, 3⟩
    canCoverSquare s r n ∧
    (∀ (s' : Square) (n' : ℕ), canCoverSquare s' r n' → squareArea s ≤ squareArea s') ∧
    n = 6 :=
  sorry

end smallest_square_coverage_l494_49456


namespace divisor_count_l494_49416

def n : ℕ := 2028
def k : ℕ := 2004

theorem divisor_count (h : n = 2^2 * 3^2 * 13^2) : 
  (Finset.filter (fun d => (Finset.filter (fun x => x ∣ d) (Finset.range (n^k + 1))).card = n) 
   (Finset.filter (fun x => x ∣ n^k) (Finset.range (n^k + 1)))).card = 216 := by
  sorry

end divisor_count_l494_49416


namespace complement_to_set_l494_49489

def U : Set ℤ := {-1, 0, 1, 2, 4}

theorem complement_to_set (M : Set ℤ) (h : {x : ℤ | x ∈ U ∧ x ∉ M} = {-1, 1}) : 
  M = {0, 2, 4} := by
  sorry

end complement_to_set_l494_49489


namespace line_parallel_to_parallel_plane_l494_49411

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contained_in : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane 
  (m : Line) (α β : Plane) 
  (h1 : contained_in m α) 
  (h2 : parallel α β) : 
  line_parallel_to_plane m β :=
sorry

end line_parallel_to_parallel_plane_l494_49411


namespace total_insects_on_leaves_l494_49400

/-- The total number of insects on leaves with given conditions -/
def total_insects (
  num_leaves : ℕ
  ) (ladybugs_per_leaf : ℕ
  ) (ants_per_leaf : ℕ
  ) (caterpillars_per_third_leaf : ℕ
  ) : ℕ :=
  (num_leaves * ladybugs_per_leaf) +
  (num_leaves * ants_per_leaf) +
  (num_leaves / 3 * caterpillars_per_third_leaf)

/-- Theorem stating the total number of insects under given conditions -/
theorem total_insects_on_leaves :
  total_insects 84 139 97 53 = 21308 := by
  sorry

end total_insects_on_leaves_l494_49400
