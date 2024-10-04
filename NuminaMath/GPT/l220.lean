import Mathlib

namespace circle_condition_l220_220926

theorem circle_condition (f : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 4*x + 6*y + f = 0) ↔ f < 13 :=
by
  sorry

end circle_condition_l220_220926


namespace ordered_triple_exists_l220_220290

theorem ordered_triple_exists (a b c : ℝ) (h₁ : 4 < a) (h₂ : 4 < b) (h₃ : 4 < c)
  (h₄ : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) :
  (a, b, c) = (12, 10, 8) :=
sorry

end ordered_triple_exists_l220_220290


namespace equation_of_ellipse_equation_of_line_through_ellipse_l220_220858

variables (a b x y : ℝ)

-- Conditions for the ellipse
def ellipse (a b x y : ℝ) : Prop := (a > 0 ∧ b > 0) ∧ 
  (2 * a * (1/2) = 1 ∧ 2 = a ∧ b = sqrt 3) ∧
  ((x / 2) ^ 2 + (y / sqrt 3) ^ 2 = 1)

-- Prove that the equation of the ellipse is correct
theorem equation_of_ellipse (h : ellipse a b x y) : 
  ellipse a b x y → (a = 2 ∧ b = sqrt 3 ∧ x/2^2 + y/sqrt 3^2 = 1) :=
by {
  intros,
  exact h.right.left
}

-- Conditions for the line intersection problem
variables (M : ℝ × ℝ) (l : ℝ × ℝ → ℝ)

def line_through_point (l : ℝ × ℝ → ℝ) (M : ℝ × ℝ) : Prop := 
  l = λ p, p.2 - (1/2) * p.1 + 1 ∨ l = λ p, p.2 + (1/2) * p.1 - 1

def line_intersects_ellipse (l : ℝ × ℝ → ℝ) 
  (a b x y : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧
  (ellipse a b A.1 A.2 ∧ ellipse a b B.1 B.2 ∧ 
  l A = 0 ∧ l B = 0 ∧
  ((A.1, A.2) = (B.1, B.2) + (M.1, M.2) ∧ M.1 = 2 * B.1))

-- Prove that the equation of the line is correct
theorem equation_of_line_through_ellipse (h_ellipse : ellipse a b x y) 
  (h_line : line_through_point l M) 
  (h_intersection : line_intersects_ellipse l a b x y) : 
  (line_through_point l ⟨0, 1⟩) :=
by {
  sorry
}

end equation_of_ellipse_equation_of_line_through_ellipse_l220_220858


namespace quadratic_no_roots_c_positive_l220_220472

theorem quadratic_no_roots_c_positive
  (a b c : ℝ)
  (h_no_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0)
  (h_positive : a + b + c > 0) :
  c > 0 :=
sorry

end quadratic_no_roots_c_positive_l220_220472


namespace expression_value_l220_220539

theorem expression_value (a b c d m : ℚ) (h1 : a + b = 0) (h2 : a ≠ 0) (h3 : c * d = 1) (h4 : m = -5 ∨ m = 1) :
  |m| - (a / b) + ((a + b) / 2020) - (c * d) = 1 ∨ |m| - (a / b) + ((a + b) / 2020) - (c * d) = 5 :=
by sorry

end expression_value_l220_220539


namespace find_value_sum_l220_220691

noncomputable def f : ℝ → ℝ
  := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 3) = f x
axiom value_at_minus_one : f (-1) = 1

theorem find_value_sum :
  f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
by
  sorry

end find_value_sum_l220_220691


namespace range_of_a_l220_220713

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ 2 < a ∧ a ≤ 3 := by
  sorry

end range_of_a_l220_220713


namespace sequence_formula_l220_220702

theorem sequence_formula (a : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, 0 < n → a (n + 1) = a n / (1 + a n)) : 
  ∀ n : ℕ, 0 < n → a n = 1 / n := 
by 
  sorry

end sequence_formula_l220_220702


namespace problem_I_problem_II_l220_220114

-- Definitions
noncomputable def function_f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + (a - 2) * x - Real.log x

-- Problem (I)
theorem problem_I (a : ℝ) (h_min : ∀ x : ℝ, function_f a 1 ≤ function_f a x) :
  a = 1 ∧ (∀ x : ℝ, 0 < x ∧ x < 1 → (function_f a x < function_f a 1)) ∧ (∀ x : ℝ, x > 1 → (function_f a x > function_f a 1)) :=
sorry

-- Problem (II)
theorem problem_II (a x0 : ℝ) (h_a_gt_1 : a > 1) (h_x0_pos : 0 < x0) (h_x0_lt_1 : x0 < 1)
    (h_min : ∀ x : ℝ, function_f a (1/a) ≤ function_f a x) :
  ∀ x : ℝ, function_f a 0 > 0
:= sorry

end problem_I_problem_II_l220_220114


namespace power_division_l220_220957

-- Condition given
def sixty_four_is_power_of_eight : Prop := 64 = 8^2

-- Theorem to prove
theorem power_division : sixty_four_is_power_of_eight → 8^{15} / 64^6 = 512 := by
  intro h
  have h1 : 64^6 = (8^2)^6, from by rw [h]
  have h2 : (8^2)^6 = 8^{12}, from pow_mul 8 2 6
  rw [h1, h2]
  have h3 : 8^{15} / 8^{12} = 8^{15 - 12}, from pow_div 8 15 12
  rw [h3]
  have h4 : 8^{15 - 12} = 8^3, from by rw [sub_self_add]
  rw [h4]
  have h5 : 8^3 = 512, from by norm_num
  rw [h5]
  sorry

end power_division_l220_220957


namespace number_of_boys_took_exam_l220_220178

theorem number_of_boys_took_exam (T F : ℕ) (h_avg_all : 35 * T = 39 * 100 + 15 * F)
                                (h_total_boys : T = 100 + F) : T = 120 :=
sorry

end number_of_boys_took_exam_l220_220178


namespace smallest_n_with_digits_437_l220_220464

theorem smallest_n_with_digits_437 (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) (h_m_lt_n : m < n)
  (h_digits : ∃ k : ℕ, 1000 * m = 437 * n + k ∧ k < n) : n = 1809 :=
sorry

end smallest_n_with_digits_437_l220_220464


namespace dice_probability_l220_220353

theorem dice_probability :
  let one_digit_prob := 9 / 20
  let two_digit_prob := 11 / 20
  let number_of_dice := 5
  ∃ p : ℚ,
    (number_of_dice.choose 2) * (one_digit_prob ^ 2) * (two_digit_prob ^ 3) = p ∧
    p = 107811 / 320000 :=
by
  sorry

end dice_probability_l220_220353


namespace income_expenditure_ratio_l220_220771

theorem income_expenditure_ratio (I E S : ℝ) (hI : I = 10000) (hS : S = 2000) (hEq : S = I - E) : I / E = 5 / 4 :=
by {
  sorry
}

end income_expenditure_ratio_l220_220771


namespace selectFourPeopleProbProof_l220_220560

noncomputable def selectFourPeopleProbability
  (totalCouples : ℕ)
  (selectFemales : ℕ)
  (selectMales : ℕ)
  (M : ℕ)
  (binom : ℕ) : ℕ :=
M / binom

theorem selectFourPeopleProbProof
  (totalCouples : ℕ)
  (selectMales : ℕ)
  (selectFemales : ℕ)
  (M : ℕ) :
  totalCouples = 9 →
  selectMales = 2 →
  selectFemales = 2 →
  (∀ (M : ℕ), 0 ≤ M) →
  selectFourPeopleProbability totalCouples selectFemales selectMales M (Nat.choose 9 2) = M / (Nat.choose 9 2) :=
by
  intros h_total h_males h_females h_m
  rw [selectFourPeopleProbability]
  exact h_m M

end selectFourPeopleProbProof_l220_220560


namespace add_to_fraction_eq_l220_220784

theorem add_to_fraction_eq (n : ℤ) : (4 + n : ℤ) / (7 + n) = (2 : ℤ) / 3 → n = 2 := 
by {
  sorry
}

end add_to_fraction_eq_l220_220784


namespace points_description_l220_220833

noncomputable def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem points_description (x y : ℝ) : 
  (clubsuit x y = clubsuit y x) ↔ (x = 0) ∨ (y = 0) ∨ (x = y) ∨ (x + y = 0) := 
by 
  sorry

end points_description_l220_220833


namespace gcd_gx_x_eq_one_l220_220260

   variable (x : ℤ)
   variable (hx : ∃ k : ℤ, x = 34567 * k)

   def g (x : ℤ) : ℤ := (3 * x + 4) * (8 * x + 3) * (15 * x + 11) * (x + 15)

   theorem gcd_gx_x_eq_one : Int.gcd (g x) x = 1 :=
   by 
     sorry
   
end gcd_gx_x_eq_one_l220_220260


namespace double_number_no_common_digit_l220_220124

theorem double_number_no_common_digit (a b u v : ℕ) (x : ℕ) (hx : x = 10 * a + b / 2)
  (h1 : 10 * a + b = 2 * x)
  (h2 : x < 100 ∧ x >= 10)
  (h3 : u = a + b ∧ v = |a - b| ∧ u ≠ v)
  (h4 : a ≠ b) :
  x = 17 ∧ 2 * x = 34 :=
  sorry

end double_number_no_common_digit_l220_220124


namespace weighted_average_fish_caught_l220_220388

-- Define the daily catches for each person
def AangCatches := [5, 7, 9]
def SokkaCatches := [8, 5, 6]
def TophCatches := [10, 12, 8]
def ZukoCatches := [6, 7, 10]

-- Define the group catches
def GroupCatches := AangCatches ++ SokkaCatches ++ TophCatches ++ ZukoCatches

-- Calculate the total number of fish caught by the group
def TotalFishCaught := List.sum GroupCatches

-- Calculate the total number of days fished by the group
def TotalDaysFished := 4 * 3

-- Calculate the weighted average
def WeightedAverage := TotalFishCaught.toFloat / TotalDaysFished.toFloat

-- Proof statement
theorem weighted_average_fish_caught :
  WeightedAverage = 7.75 := by
  sorry

end weighted_average_fish_caught_l220_220388


namespace initial_friends_count_l220_220935

variable (F : ℕ)
variable (players_quit : ℕ)
variable (lives_per_player : ℕ)
variable (total_remaining_lives : ℕ)

theorem initial_friends_count
  (h1 : players_quit = 7)
  (h2 : lives_per_player = 8)
  (h3 : total_remaining_lives = 72) :
  F = 16 :=
by
  have h4 : 8 * (F - 7) = 72 := by sorry   -- Derived from given conditions
  have : 8 * F - 56 = 72 := by sorry        -- Simplify equation
  have : 8 * F = 128 := by sorry           -- Add 56 to both sides
  have : F = 16 := by sorry                -- Divide both sides by 8
  exact this                               -- Final result

end initial_friends_count_l220_220935


namespace least_n_condition_l220_220777

theorem least_n_condition (n : ℕ) (h1 : ∀ k : ℕ, 1 ≤ k → k ≤ n + 1 → (k ∣ n * (n - 1) → k ≠ n + 1)) : n = 4 :=
sorry

end least_n_condition_l220_220777


namespace brenda_has_8_dollars_l220_220200

-- Define the amounts of money each friend has
def emma_money : ℕ := 8
def daya_money : ℕ := emma_money + (25 * emma_money / 100) -- 25% more than Emma's money
def jeff_money : ℕ := 2 * daya_money / 5 -- Jeff has 2/5 of Daya's money
def brenda_money : ℕ := jeff_money + 4 -- Brenda has 4 more dollars than Jeff

-- The theorem stating the final question
theorem brenda_has_8_dollars : brenda_money = 8 :=
by
  sorry

end brenda_has_8_dollars_l220_220200


namespace amanda_quizzes_l220_220994

theorem amanda_quizzes (n : ℕ) (h1 : n > 0) (h2 : 92 * n + 97 = 93 * 5) : n = 4 :=
by
  sorry

end amanda_quizzes_l220_220994


namespace find_a_monotonic_intervals_exp_gt_xsquare_plus_one_l220_220870

-- Define the function f(x) and its derivative f'(x)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x - 1
noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a

-- Prove that a = 2 given the slope condition at x = 0
theorem find_a (a : ℝ) (h : f_prime 0 a = -1) : a = 2 :=
by sorry

-- Characteristics of the function f(x)
theorem monotonic_intervals (a : ℝ) (h : a = 2) :
  ∀ x : ℝ, (x ≤ Real.log 2 → f_prime x a ≤ 0) ∧ (x >= Real.log 2 → f_prime x a >= 0) :=
by sorry

-- Prove that e^x > x^2 + 1 when x > 0
theorem exp_gt_xsquare_plus_one (x : ℝ) (hx : x > 0) : Real.exp x > x^2 + 1 :=
by sorry

end find_a_monotonic_intervals_exp_gt_xsquare_plus_one_l220_220870


namespace ball_price_equation_l220_220132

structure BallPrices where
  (x : Real) -- price of each soccer ball in yuan
  (condition1 : ∀ (x : Real), (1500 / (x + 20) - 800 / x = 5))

/-- Prove that the equation follows from the given conditions. -/
theorem ball_price_equation (b : BallPrices) : 1500 / (b.x + 20) - 800 / b.x = 5 := 
by sorry

end ball_price_equation_l220_220132


namespace rectangle_area_l220_220470

-- Define the length and width of the rectangle based on given ratio
def length (k: ℝ) := 5 * k
def width (k: ℝ) := 2 * k

-- The perimeter condition
def perimeter (k: ℝ) := 2 * (length k) + 2 * (width k) = 280

-- The diagonal condition
def diagonal_condition (k: ℝ) := (width k) * Real.sqrt 2 = (length k) / 2

-- The area of the rectangle
def area (k: ℝ) := (length k) * (width k)

-- The main theorem to be proven
theorem rectangle_area : ∃ k: ℝ, perimeter k ∧ diagonal_condition k ∧ area k = 4000 :=
by
  sorry

end rectangle_area_l220_220470


namespace money_left_after_deductions_l220_220506

-- Define the weekly income
def weekly_income : ℕ := 500

-- Define the tax deduction as 10% of the weekly income
def tax : ℕ := (10 * weekly_income) / 100

-- Define the weekly water bill
def water_bill : ℕ := 55

-- Define the tithe as 10% of the weekly income
def tithe : ℕ := (10 * weekly_income) / 100

-- Define the total deductions
def total_deductions : ℕ := tax + water_bill + tithe

-- Define the money left
def money_left : ℕ := weekly_income - total_deductions

-- The statement to prove
theorem money_left_after_deductions : money_left = 345 := by
  sorry

end money_left_after_deductions_l220_220506


namespace number_of_possible_orders_l220_220878

def number_of_finishing_orders : ℕ := 4 * 3 * 2 * 1

theorem number_of_possible_orders : number_of_finishing_orders = 24 := 
by
  have h : number_of_finishing_orders = 24 := by norm_num
  exact h

end number_of_possible_orders_l220_220878


namespace scorpion_needs_10_millipedes_l220_220067

-- Define the number of segments required daily
def total_segments_needed : ℕ := 800

-- Define the segments already consumed by the scorpion
def segments_consumed : ℕ := 60 + 2 * (2 * 60)

-- Calculate the remaining segments needed
def remaining_segments_needed : ℕ := total_segments_needed - segments_consumed

-- Define the segments per millipede
def segments_per_millipede : ℕ := 50

-- Prove that the number of 50-segment millipedes to be eaten is 10
theorem scorpion_needs_10_millipedes 
  (h : remaining_segments_needed = 500) 
  (h2 : 500 / segments_per_millipede = 10) : 
  500 / segments_per_millipede = 10 := by
  sorry

end scorpion_needs_10_millipedes_l220_220067


namespace GCD_17_51_LCM_17_51_GCD_6_8_LCM_8_9_l220_220168

noncomputable def GCD (a b : ℕ) : ℕ := Nat.gcd a b
noncomputable def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem GCD_17_51 : GCD 17 51 = 17 := by
  sorry

theorem LCM_17_51 : LCM 17 51 = 51 := by
  sorry

theorem GCD_6_8 : GCD 6 8 = 2 := by
  sorry

theorem LCM_8_9 : LCM 8 9 = 72 := by
  sorry

end GCD_17_51_LCM_17_51_GCD_6_8_LCM_8_9_l220_220168


namespace range_of_a_for_inequality_l220_220723

theorem range_of_a_for_inequality :
  {a : ℝ // ∀ (x : ℝ), a * x^2 + 2 * a * x + 1 > 0} = {a : ℝ // 0 ≤ a ∧ a < 1} :=
sorry

end range_of_a_for_inequality_l220_220723


namespace second_group_children_is_16_l220_220072

def cases_purchased : ℕ := 13
def bottles_per_case : ℕ := 24
def camp_days : ℕ := 3
def first_group_children : ℕ := 14
def third_group_children : ℕ := 12
def bottles_per_child_per_day : ℕ := 3
def additional_bottles_needed : ℕ := 255

def fourth_group_children (x : ℕ) : ℕ := (14 + x + 12) / 2
def total_initial_bottles : ℕ := cases_purchased * bottles_per_case
def total_children (x : ℕ) : ℕ := 14 + x + 12 + fourth_group_children x 

def total_consumption (x : ℕ) : ℕ := (total_children x) * bottles_per_child_per_day * camp_days
def total_bottles_needed : ℕ := total_initial_bottles + additional_bottles_needed

theorem second_group_children_is_16 :
  ∃ x : ℕ, total_consumption x = total_bottles_needed ∧ x = 16 :=
by
  sorry

end second_group_children_is_16_l220_220072


namespace proof_op_l220_220910

def op (A B : ℕ) : ℕ := (A * B) / 2

theorem proof_op (a b c : ℕ) : op (op 4 6) 9 = 54 := by
  sorry

end proof_op_l220_220910


namespace greatest_sum_consecutive_integers_l220_220333

theorem greatest_sum_consecutive_integers (n : ℤ) (h : n * (n + 1) < 360) : n + (n + 1) ≤ 37 := by
  sorry

end greatest_sum_consecutive_integers_l220_220333


namespace compute_expression_l220_220830

theorem compute_expression : 12 * (1 / 7) * 14 * 2 = 48 := 
sorry

end compute_expression_l220_220830


namespace forty_percent_of_number_l220_220456

theorem forty_percent_of_number (N : ℝ) 
  (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 17) : 
  0.40 * N = 204 :=
sorry

end forty_percent_of_number_l220_220456


namespace figure_representation_l220_220361

theorem figure_representation (x y : ℝ) : 
  |x| + |y| ≤ 2 * Real.sqrt (x^2 + y^2) ∧ 2 * Real.sqrt (x^2 + y^2) ≤ 3 * max (|x|) (|y|) → 
  Figure2 :=
sorry

end figure_representation_l220_220361


namespace num_four_digit_palindromic_squares_l220_220642

theorem num_four_digit_palindromic_squares :
  let is_palindrome (n : ℕ) : Prop := (n.toString = n.toString.reverse)
  let four_digit_squares := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ ∃ m : ℕ, n = m * m ∧ 32 ≤ m ∧ m ≤ 99}
  let palindromic_four_digit_squares := {n : ℕ | n ∈ four_digit_squares ∧ is_palindrome n}
  palindromic_four_digit_squares.card = 1 :=
by
  sorry

end num_four_digit_palindromic_squares_l220_220642


namespace pirate_treasure_probability_l220_220365

theorem pirate_treasure_probability :
  let p_treasure_no_traps := 1 / 3
  let p_traps_no_treasure := 1 / 6
  let p_neither := 1 / 2
  let choose_4_out_of_8 := 70
  let p_4_treasure_no_traps := (1 / 3) ^ 4
  let p_4_neither := (1 / 2) ^ 4
  choose_4_out_of_8 * p_4_treasure_no_traps * p_4_neither = 35 / 648 :=
by
  sorry

end pirate_treasure_probability_l220_220365


namespace naomi_number_of_ways_to_1000_l220_220580

-- Define the initial condition and operations

def start : ℕ := 2

def add1 (n : ℕ) : ℕ := n + 1

def square (n : ℕ) : ℕ := n * n

-- Define a proposition that counts the number of ways to reach 1000 from 2 using these operations
def count_ways (start target : ℕ) : ℕ := sorry  -- We'll need a complex function to literally count the paths, but we'll abstract this here.

-- Theorem stating the number of ways to reach 1000
theorem naomi_number_of_ways_to_1000 : count_ways start 1000 = 128 := 
sorry

end naomi_number_of_ways_to_1000_l220_220580


namespace ball_bounce_height_l220_220493

theorem ball_bounce_height :
  ∃ (k : ℕ), 10 * (1 / 2) ^ k < 1 ∧ (∀ m < k, 10 * (1 / 2) ^ m ≥ 1) :=
sorry

end ball_bounce_height_l220_220493


namespace clubsuit_problem_l220_220669

def clubsuit (x y : ℤ) : ℤ :=
  (x^2 + y^2) * (x - y)

theorem clubsuit_problem : clubsuit 2 (clubsuit 3 4) = 16983 := 
by 
  sorry

end clubsuit_problem_l220_220669


namespace find_a7_l220_220698

theorem find_a7 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) (x : ℝ)
  (h : x^8 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 +
            a_4 * (x + 1)^4 + a_5 * (x + 1)^5 + a_6 * (x + 1)^6 +
            a_7 * (x + 1)^7 + a_8 * (x + 1)^8) : 
  a_7 = -8 := 
sorry

end find_a7_l220_220698


namespace jennifer_money_left_l220_220343

variable (initial_amount : ℝ) (spent_sandwich_rate : ℝ) (spent_museum_rate : ℝ) (spent_book_rate : ℝ)

def money_left := initial_amount - (spent_sandwich_rate * initial_amount + spent_museum_rate * initial_amount + spent_book_rate * initial_amount)

theorem jennifer_money_left (h_initial : initial_amount = 150)
  (h_sandwich_rate : spent_sandwich_rate = 1/5)
  (h_museum_rate : spent_museum_rate = 1/6)
  (h_book_rate : spent_book_rate = 1/2) :
  money_left initial_amount spent_sandwich_rate spent_museum_rate spent_book_rate = 20 :=
by
  sorry

end jennifer_money_left_l220_220343


namespace remainder_23_to_2047_mod_17_l220_220486

theorem remainder_23_to_2047_mod_17 :
  23^2047 % 17 = 11 := 
by {
  sorry
}

end remainder_23_to_2047_mod_17_l220_220486


namespace race_length_l220_220756

theorem race_length (cristina_speed nicky_speed : ℕ) (head_start total_time : ℕ) 
  (h1 : cristina_speed > nicky_speed) 
  (h2 : head_start = 12) (h3 : cristina_speed = 5) (h4 : nicky_speed = 3) 
  (h5 : total_time = 30) :
  let nicky_distance := nicky_speed * total_time,
      cristina_time := total_time - head_start,
      cristina_distance := cristina_speed * cristina_time in
  nicky_distance = cristina_distance 
  ∧ nicky_distance = 90 := 
by
  sorry

end race_length_l220_220756


namespace power_division_identity_l220_220961

theorem power_division_identity : (8 ^ 15) / (64 ^ 6) = 512 := by
  have h64 : 64 = 8 ^ 2 := by
    sorry
  have h_exp_rule : ∀ (a m n : ℕ), (a ^ m) ^ n = a ^ (m * n) := by
    sorry
  
  rw [h64]
  rw [h_exp_rule]
  sorry

end power_division_identity_l220_220961


namespace allison_craft_items_l220_220990

def glue_sticks (A B : Nat) : Prop := A = B + 8
def construction_paper (A B : Nat) : Prop := B = 6 * A

theorem allison_craft_items (Marie_glue_sticks Marie_paper_packs : Nat)
    (h1 : Marie_glue_sticks = 15)
    (h2 : Marie_paper_packs = 30) :
    ∃ (Allison_glue_sticks Allison_paper_packs total_items : Nat),
        glue_sticks Allison_glue_sticks Marie_glue_sticks ∧
        construction_paper Allison_paper_packs Marie_paper_packs ∧
        total_items = Allison_glue_sticks + Allison_paper_packs ∧
        total_items = 28 :=
by
    sorry

end allison_craft_items_l220_220990


namespace area_of_efgh_l220_220480

def small_rectangle_shorter_side : ℝ := 7
def small_rectangle_longer_side : ℝ := 3 * small_rectangle_shorter_side
def larger_rectangle_width : ℝ := small_rectangle_longer_side
def larger_rectangle_length : ℝ := small_rectangle_longer_side + small_rectangle_shorter_side

theorem area_of_efgh :
  larger_rectangle_length * larger_rectangle_width = 588 := by
  sorry

end area_of_efgh_l220_220480


namespace pow_div_l220_220953

theorem pow_div (x : ℕ) (a b c d : ℕ) (h1 : x^b = d) (h2 : x^(a*d) = c) : c / (d^b) = 512 := by
  sorry

end pow_div_l220_220953


namespace part1_min_value_part2_find_b_part3_range_b_div_a_l220_220718

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 - abs (a*x - b)

-- Part (1)
theorem part1_min_value : f 1 1 1 = -5/4 :=
by 
  sorry

-- Part (2)
theorem part2_find_b (b : ℝ) (h : b ≥ 2) (h_domain : ∀ x, 1 ≤ x ∧ x ≤ b) (h_range : ∀ y, 1 ≤ y ∧ y ≤ b) : 
  b = 2 :=
by 
  sorry

-- Part (3)
theorem part3_range_b_div_a (a b : ℝ) (h_distinct : (∃ x : ℝ, 0 < x ∧ x < 2 ∧ f x a b = 1 ∧ ∀ y : ℝ, 0 < y ∧ y < 2 ∧ f y a b = 1 ∧ x ≠ y)) : 
  1 < b / a ∧ b / a < 2 :=
by 
  sorry

end part1_min_value_part2_find_b_part3_range_b_div_a_l220_220718


namespace max_minus_min_l220_220864

noncomputable def f (x : ℝ) := if x > 0 then (x - 1) ^ 2 else (x + 1) ^ 2

theorem max_minus_min (n m : ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ (-1 / 2) → n ≤ f x ∧ f x ≤ m) →
  m - n = 1 :=
by { sorry }

end max_minus_min_l220_220864


namespace probability_no_shaded_square_l220_220204

theorem probability_no_shaded_square (n m : ℕ) (h1 : n = (2006.choose 2)) (h2 : m = 1003^2) : 
  (1 - (m / n) = (1002 / 2005)) := 
by
  -- Number of rectangles in one row
  have hn : n = 1003 * 2005 := h1
  -- Number of rectangles in one row containing a shaded square
  have hm : m = 1003 * 1003 := h2
  sorry

end probability_no_shaded_square_l220_220204


namespace books_per_shelf_l220_220548

theorem books_per_shelf (mystery_shelves picture_shelves total_books : ℕ)
    (h₁ : mystery_shelves = 5)
    (h₂ : picture_shelves = 3)
    (h₃ : total_books = 32) :
    (total_books / (mystery_shelves + picture_shelves) = 4) :=
by
    sorry

end books_per_shelf_l220_220548


namespace solve_for_y_l220_220027

theorem solve_for_y :
  ∀ (y : ℝ), (9 * y^2 + 49 * y^2 + 21/2 * y^2 = 1300) → y = 4.34 := 
by sorry

end solve_for_y_l220_220027


namespace minutkin_bedtime_l220_220911

def time_minutkin_goes_to_bed 
    (morning_time : ℕ) 
    (morning_turns : ℕ) 
    (night_turns : ℕ) 
    (morning_hours : ℕ) 
    (morning_minutes : ℕ)
    (hours_per_turn : ℕ) 
    (minutes_per_turn : ℕ) : Nat := 
    ((morning_hours * 60 + morning_minutes) - (night_turns * hours_per_turn * 60 + night_turns * minutes_per_turn)) % 1440 

theorem minutkin_bedtime : 
    time_minutkin_goes_to_bed 9 9 11 8 30 1 12 = 1290 :=
    sorry

end minutkin_bedtime_l220_220911


namespace h_inv_f_neg3_does_not_exist_real_l220_220380

noncomputable def h : ℝ → ℝ := sorry
noncomputable def f : ℝ → ℝ := sorry

theorem h_inv_f_neg3_does_not_exist_real (h_inv : ℝ → ℝ)
  (h_cond : ∀ (x : ℝ), f (h_inv (h x)) = 7 * x ^ 2 + 4) :
  ¬ ∃ x : ℝ, h_inv (f (-3)) = x :=
by 
  sorry

end h_inv_f_neg3_does_not_exist_real_l220_220380


namespace f_of_8_l220_220859

variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x : ℝ, f (-x) = -f (x)
axiom function_property : ∀ x : ℝ, f (x + 2) = -1 / f (x)

-- Statement to prove
theorem f_of_8 : f 8 = 0 :=
sorry

end f_of_8_l220_220859


namespace solve_nat_numbers_equation_l220_220019

theorem solve_nat_numbers_equation (n k l m : ℕ) (h_l : l > 1) 
  (h_eq : (1 + n^k)^l = 1 + n^m) : (n = 2) ∧ (k = 1) ∧ (l = 2) ∧ (m = 3) := 
by
  sorry

end solve_nat_numbers_equation_l220_220019


namespace egg_cartons_l220_220999

theorem egg_cartons (chickens eggs_per_chicken eggs_per_carton : ℕ) (h_chickens : chickens = 20) (h_eggs_per_chicken : eggs_per_chicken = 6) (h_eggs_per_carton : eggs_per_carton = 12) : 
  (chickens * eggs_per_chicken) / eggs_per_carton = 10 :=
by
  rw [h_chickens, h_eggs_per_chicken, h_eggs_per_carton] -- Replace the variables with the given values
  -- Calculate the number of eggs
  have h_eggs := 20 * 6
  -- Apply the number of eggs to find the number of cartons
  rw [show 20 * 6 = 120, from rfl, show 120 / 12 = 10, from rfl]
  sorry -- Placeholder for the detailed proof

end egg_cartons_l220_220999


namespace vertex_of_parabola_l220_220774

theorem vertex_of_parabola (a b : ℝ) (roots_condition : ∀ x, -x^2 + a * x + b ≤ 0 ↔ (x ≤ -3 ∨ x ≥ 5)) :
  ∃ v : ℝ × ℝ, v = (1, 16) :=
by
  sorry

end vertex_of_parabola_l220_220774


namespace nails_needed_l220_220693

-- Define the number of nails needed for each plank
def nails_per_plank : ℕ := 2

-- Define the number of planks used by John
def planks_used : ℕ := 16

-- The total number of nails needed.
theorem nails_needed : (nails_per_plank * planks_used) = 32 :=
by
  -- Our goal is to prove that nails_per_plank * planks_used = 32
  sorry

end nails_needed_l220_220693


namespace equation_of_perpendicular_line_l220_220466

theorem equation_of_perpendicular_line (a b c : ℝ) (p q : ℝ) (hx : a ≠ 0) (hy : b ≠ 0)
  (h_perpendicular : a * 2 + b * 1 = 0) (h_point : (-1) * a + 2 * b + c = 0)
  : a = 1 ∧ b = -2 ∧ c = -5 → (x:ℝ) * 1 + (y:ℝ) * (-2) + (-5) = 0 :=
by sorry

end equation_of_perpendicular_line_l220_220466


namespace fundamental_disagreement_l220_220602

-- Definitions based on conditions
def represents_materialism (s : String) : Prop :=
  s = "Without scenery, where does emotion come from?"

def represents_idealism (s : String) : Prop :=
  s = "Without emotion, where does scenery come from?"

-- Theorem statement
theorem fundamental_disagreement :
  ∀ (s1 s2 : String),
  (represents_materialism s1 ∧ represents_idealism s2) →
  (∃ disagreement : String,
    disagreement = "Acknowledging whether the essence of the world is material or consciousness") :=
by
  intros s1 s2 h
  existsi "Acknowledging whether the essence of the world is material or consciousness"
  sorry

end fundamental_disagreement_l220_220602


namespace C_converges_l220_220112

noncomputable def behavior_of_C (e R r : ℝ) (n : ℕ) : ℝ := e * (n^2) / (R + n * (r^2))

theorem C_converges (e R r : ℝ) (h₁ : 0 < r) : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |behavior_of_C e R r n - e / r^2| < ε := 
sorry

end C_converges_l220_220112


namespace product_of_coefficients_is_negative_integer_l220_220931

theorem product_of_coefficients_is_negative_integer
  (a b c : ℤ)
  (habc_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (discriminant_positive : (b * b - 4 * a * c) > 0)
  (product_cond : a * b * c = (c / a)) :
  ∃ k : ℤ, k < 0 ∧ k = a * b * c :=
by
  sorry

end product_of_coefficients_is_negative_integer_l220_220931


namespace sum_of_coefficients_l220_220603

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem sum_of_coefficients (a b c : ℝ) 
  (h1 : quadratic a b c 3 = 0) 
  (h2 : quadratic a b c 7 = 0)
  (h3 : ∃ x0, (∀ x, quadratic a b c x ≥ quadratic a b c x0) ∧ quadratic a b c x0 = 20) :
  a + b + c = -105 :=
by 
  sorry

end sum_of_coefficients_l220_220603


namespace perimeter_of_region_l220_220313

-- Define the condition
def area_of_region := 512 -- square centimeters
def number_of_squares := 8

-- Define the presumed perimeter
def presumed_perimeter := 144 -- the correct answer

-- Mathematical statement that needs proof
theorem perimeter_of_region (area_of_region: ℕ) (number_of_squares: ℕ) (presumed_perimeter: ℕ) : 
   area_of_region = 512 ∧ number_of_squares = 8 → presumed_perimeter = 144 :=
by 
  sorry

end perimeter_of_region_l220_220313


namespace inequality_holds_iff_l220_220621

theorem inequality_holds_iff (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → x^2 + (a - 4) * x + 4 > 0) ↔ a > 0 :=
by
  sorry

end inequality_holds_iff_l220_220621


namespace ratio_of_areas_of_concentric_circles_l220_220043

theorem ratio_of_areas_of_concentric_circles :
  (∀ (r1 r2 : ℝ), 
    r1 > 0 ∧ r2 > 0 ∧ 
    ((60 / 360) * 2 * Real.pi * r1 = (48 / 360) * 2 * Real.pi * r2)) →
    ((Real.pi * r1 ^ 2) / (Real.pi * r2 ^ 2) = (16 / 25)) :=
by
  intro h
  sorry

end ratio_of_areas_of_concentric_circles_l220_220043


namespace product_of_real_solutions_l220_220319

theorem product_of_real_solutions :
  (∀ x : ℝ, (x + 1) / (3 * x + 3) = (3 * x + 2) / (8 * x + 2)) →
  x = -1 ∨ x = -4 →
  (-1) * (-4) = 4 := 
sorry

end product_of_real_solutions_l220_220319


namespace solve_system_l220_220591

variable {R : Type*} [CommRing R]

-- Given conditions
variables (a b c x y z : R)

-- Assuming the given system of equations
axiom eq1 : x + a*y + a^2*z + a^3 = 0
axiom eq2 : x + b*y + b^2*z + b^3 = 0
axiom eq3 : x + c*y + c^2*z + c^3 = 0

-- The goal is to prove the mathematical equivalence
theorem solve_system : x = -a*b*c ∧ y = a*b + b*c + c*a ∧ z = -(a + b + c) :=
by
  sorry

end solve_system_l220_220591


namespace quarterly_insurance_payment_l220_220755

theorem quarterly_insurance_payment (annual_payment : ℕ) (quarters_in_year : ℕ) (quarterly_payment : ℕ) : 
  annual_payment = 1512 → quarters_in_year = 4 → quarterly_payment * quarters_in_year = annual_payment → quarterly_payment = 378 :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  sorry

end quarterly_insurance_payment_l220_220755


namespace totalCerealInThreeBoxes_l220_220479

def firstBox := 14
def secondBox := firstBox / 2
def thirdBox := secondBox + 5
def totalCereal := firstBox + secondBox + thirdBox

theorem totalCerealInThreeBoxes : totalCereal = 33 := 
by {
  sorry
}

end totalCerealInThreeBoxes_l220_220479


namespace compute_gf3_l220_220907

def f (x : ℝ) : ℝ := x^3 - 3
def g (x : ℝ) : ℝ := 2 * x^2 - x + 4

theorem compute_gf3 : g (f 3) = 1132 := 
by 
  sorry

end compute_gf3_l220_220907


namespace label_possible_iff_even_l220_220865

open Finset

variable {B : Type*} (A : Fin B → Finset B) (n : ℕ)

def satisfies_conditions (A : Fin (2 * n + 1) → Finset B) (n : ℕ) : Prop :=
  (∀ i, (A i).card = 2 * n) ∧
  (∀ i j, i < j → (A i ∩ A j).card = 1) ∧
  (∀ b ∈ ⋃ i, A i, (filter (λ i, b ∈ A i) (range (2 * n + 1))).card ≥ 2)

theorem label_possible_iff_even
  (h : ∃ (n : ℕ) (A : Fin (2 * n + 1) → Finset B), 
    satisfies_conditions A n) :
  ∀ {n : ℕ}, (∃ (A : Fin (2 * n + 1) → Finset B), satisfies_conditions A n) → 
  (∃ (f : B → Fin 2), ∀ i, (A i).filter (λ b, f b = 0).card = n) ↔ Even n :=
sorry

end label_possible_iff_even_l220_220865


namespace wage_of_one_man_l220_220064

/-- Proof that the wage of one man is Rs. 24 given the conditions. -/
theorem wage_of_one_man (M W_w B_w : ℕ) (H1 : 120 = 5 * M + W_w * 5 + B_w * 8) 
  (H2 : 5 * M = W_w * 5) (H3 : W_w * 5 = B_w * 8) : M = 24 :=
by
  sorry

end wage_of_one_man_l220_220064


namespace pastries_sold_value_l220_220228

-- Define the number of cakes sold and the relationship between cakes and pastries
def number_of_cakes_sold := 78
def pastries_sold (C : Nat) := C + 76

-- State the theorem we want to prove
theorem pastries_sold_value : pastries_sold number_of_cakes_sold = 154 := by
  sorry

end pastries_sold_value_l220_220228


namespace net_pay_rate_per_hour_l220_220978

-- Defining the given conditions
def travel_hours : ℕ := 3
def speed_mph : ℕ := 50
def fuel_efficiency : ℕ := 25 -- miles per gallon
def pay_rate_per_mile : ℚ := 0.60 -- dollars per mile
def gas_cost_per_gallon : ℚ := 2.50 -- dollars per gallon

-- Define the statement we want to prove
theorem net_pay_rate_per_hour : 
  (travel_hours * speed_mph * pay_rate_per_mile - 
  (travel_hours * speed_mph / fuel_efficiency) * gas_cost_per_gallon) / 
  travel_hours = 25 :=
by
  repeat {sorry}

end net_pay_rate_per_hour_l220_220978


namespace unique_four_digit_palindromic_square_l220_220653

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem unique_four_digit_palindromic_square :
  ∃! n : ℕ, is_palindrome n ∧ is_four_digit n ∧ ∃ m : ℕ, m^2 = n :=
sorry

end unique_four_digit_palindromic_square_l220_220653


namespace probability_no_shaded_square_l220_220203

theorem probability_no_shaded_square (n m : ℕ) (h1 : n = (2006.choose 2)) (h2 : m = 1003^2) : 
  (1 - (m / n) = (1002 / 2005)) := 
by
  -- Number of rectangles in one row
  have hn : n = 1003 * 2005 := h1
  -- Number of rectangles in one row containing a shaded square
  have hm : m = 1003 * 1003 := h2
  sorry

end probability_no_shaded_square_l220_220203


namespace range_of_a_for_inequality_l220_220724

theorem range_of_a_for_inequality :
  {a : ℝ // ∀ (x : ℝ), a * x^2 + 2 * a * x + 1 > 0} = {a : ℝ // 0 ≤ a ∧ a < 1} :=
sorry

end range_of_a_for_inequality_l220_220724


namespace range_of_k_l220_220314

theorem range_of_k (k : ℝ) : (2 > 0) ∧ (k > 0) ∧ (k < 2) ↔ (0 < k ∧ k < 2) :=
by
  sorry

end range_of_k_l220_220314


namespace initial_milk_amount_l220_220152

theorem initial_milk_amount (M : ℝ) (H1 : 0.05 * M = 0.02 * (M + 15)) : M = 10 :=
by
  sorry

end initial_milk_amount_l220_220152


namespace tan_neg_210_eq_neg_sqrt_3_div_3_l220_220971

theorem tan_neg_210_eq_neg_sqrt_3_div_3 : Real.tan (-210 * Real.pi / 180) = - (Real.sqrt 3 / 3) :=
by
  sorry

end tan_neg_210_eq_neg_sqrt_3_div_3_l220_220971


namespace yards_green_correct_l220_220326

-- Define the conditions
def total_yards_silk := 111421
def yards_pink := 49500

-- Define the question as a theorem statement
theorem yards_green_correct :
  (total_yards_silk - yards_pink = 61921) :=
by
  sorry

end yards_green_correct_l220_220326


namespace find_union_of_sets_l220_220877

-- Define the sets A and B in terms of a
def A (a : ℤ) : Set ℤ := { n | n = |a + 1| ∨ n = 3 ∨ n = 5 }
def B (a : ℤ) : Set ℤ := { n | n = 2 * a + 1 ∨ n = a^2 + 2 * a ∨ n = a^2 + 2 * a - 1 }

-- Given condition: A ∩ B = {2, 3}
def condition (a : ℤ) : Prop := A a ∩ B a = {2, 3}

-- The correct answer: A ∪ B = {-5, 2, 3, 5}
theorem find_union_of_sets (a : ℤ) (h : condition a) : A a ∪ B a = {-5, 2, 3, 5} :=
sorry

end find_union_of_sets_l220_220877


namespace four_digit_palindrome_square_count_l220_220648

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem four_digit_palindrome_square_count : 
  ∃! (n : ℕ), is_four_digit n ∧ is_palindrome n ∧ (∃ (m : ℕ), n = m * m) := by
  sorry

end four_digit_palindrome_square_count_l220_220648


namespace prob_X_ge_one_l220_220892

noncomputable def normal_distribution (mean variance : ℝ) : PMF ℝ :=
sorry -- PMF definition for normal distribution

noncomputable def X : PMF ℝ := normal_distribution (-1) (σ^2)

axiom prob_given : pmf.probability (set.Icc (-3) (-1)) X = 0.4

theorem prob_X_ge_one :
  pmf.probability (set.Ici 1) X = 0.1 :=
sorry

end prob_X_ge_one_l220_220892


namespace weight_of_tin_of_cookies_l220_220631

def weight_of_bag_of_chips := 20 -- in ounces
def weight_jasmine_carries := 336 -- converting 21 pounds to ounces
def bags_jasmine_buys := 6
def tins_multiplier := 4

theorem weight_of_tin_of_cookies 
  (weight_of_bag_of_chips : ℕ := weight_of_bag_of_chips)
  (weight_jasmine_carries : ℕ := weight_jasmine_carries)
  (bags_jasmine_buys : ℕ := bags_jasmine_buys)
  (tins_multiplier : ℕ := tins_multiplier) : 
  ℕ :=
  let total_weight_bags := bags_jasmine_buys * weight_of_bag_of_chips
  let total_weight_cookies := weight_jasmine_carries - total_weight_bags
  let num_of_tins := bags_jasmine_buys * tins_multiplier
  total_weight_cookies / num_of_tins

example : weight_of_tin_of_cookies weight_of_bag_of_chips weight_jasmine_carries bags_jasmine_buys tins_multiplier = 9 :=
by sorry

end weight_of_tin_of_cookies_l220_220631


namespace max_volume_cuboid_l220_220498

theorem max_volume_cuboid (x y z : ℕ) (h : 2 * (x * y + x * z + y * z) = 150) : x * y * z ≤ 125 :=
sorry

end max_volume_cuboid_l220_220498


namespace sweater_markup_percentage_l220_220061

-- The wholesale cost W and retail price R
variables (W R : ℝ)

-- The given condition
variable (h : 0.30 * R = 1.40 * W)

-- The theorem to prove
theorem sweater_markup_percentage (h : 0.30 * R = 1.40 * W) : (R - W) / W * 100 = 366.67 :=
by
  -- The solution steps would be placed here, if we were proving.
  sorry

end sweater_markup_percentage_l220_220061


namespace initial_men_count_l220_220596

theorem initial_men_count (M : ℕ) (A : ℕ) (H1 : 58 - (20 + 22) = 2 * M) : M = 8 :=
by
  sorry

end initial_men_count_l220_220596


namespace power_division_l220_220955

-- Condition given
def sixty_four_is_power_of_eight : Prop := 64 = 8^2

-- Theorem to prove
theorem power_division : sixty_four_is_power_of_eight → 8^{15} / 64^6 = 512 := by
  intro h
  have h1 : 64^6 = (8^2)^6, from by rw [h]
  have h2 : (8^2)^6 = 8^{12}, from pow_mul 8 2 6
  rw [h1, h2]
  have h3 : 8^{15} / 8^{12} = 8^{15 - 12}, from pow_div 8 15 12
  rw [h3]
  have h4 : 8^{15 - 12} = 8^3, from by rw [sub_self_add]
  rw [h4]
  have h5 : 8^3 = 512, from by norm_num
  rw [h5]
  sorry

end power_division_l220_220955


namespace therapy_charge_l220_220987

-- Define the charges
def first_hour_charge (S : ℝ) : ℝ := S + 50
def subsequent_hour_charge (S : ℝ) : ℝ := S

-- Define the total charge before service fee for 8 hours
def total_charge_8_hours_before_fee (F S : ℝ) : ℝ := F + 7 * S

-- Define the total charge including the service fee for 8 hours
def total_charge_8_hours (F S : ℝ) : ℝ := 1.10 * (F + 7 * S)

-- Define the total charge before service fee for 3 hours
def total_charge_3_hours_before_fee (F S : ℝ) : ℝ := F + 2 * S

-- Define the total charge including the service fee for 3 hours
def total_charge_3_hours (F S : ℝ) : ℝ := 1.10 * (F + 2 * S)

theorem therapy_charge (S F : ℝ) :
  (F = S + 50) → (1.10 * (F + 7 * S) = 900) → (1.10 * (F + 2 * S) = 371.87) :=
by {
  sorry
}

end therapy_charge_l220_220987


namespace inscribed_shapes_ratio_l220_220366

theorem inscribed_shapes_ratio {a b c : ℕ} (h : a^2 + b^2 = c^2) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) : 
  (∃ (x y : ℚ), x = b ∧ y = (a * b) / c ∧ x / y = 13 / 5) :=
by
  sorry

end inscribed_shapes_ratio_l220_220366


namespace number_difference_l220_220172

theorem number_difference (a b : ℕ) (h1 : a + b = 25650) (h2 : a % 100 = 0) (h3 : b = a / 100) :
  a - b = 25146 :=
sorry

end number_difference_l220_220172


namespace inequality_equivalence_l220_220679

theorem inequality_equivalence (x : ℝ) : 
  (x + 2) / (x - 1) ≥ 0 ↔ (x + 2) * (x - 1) ≥ 0 :=
sorry

end inequality_equivalence_l220_220679


namespace tom_initial_books_l220_220041

theorem tom_initial_books (B : ℕ) (h1 : B - 4 + 38 = 39) : B = 5 :=
by
  sorry

end tom_initial_books_l220_220041


namespace distance_between_A_and_B_is_750_l220_220758

def original_speed := 150 -- derived from the solution

def distance (S D : ℝ) :=
  (D / S) - (D / ((5 / 4) * S)) = 1 ∧
  ((D - 150) / S) - ((5 * (D - 150)) / (6 * S)) = 2 / 3

theorem distance_between_A_and_B_is_750 :
  ∃ D : ℝ, distance original_speed D ∧ D = 750 :=
by
  sorry

end distance_between_A_and_B_is_750_l220_220758


namespace geometric_progression_ratio_l220_220841

theorem geometric_progression_ratio (q : ℝ) (h : |q| < 1 ∧ ∀a : ℝ, a = 4 * (a * q / (1 - q) - a * q)) :
  q = 1 / 5 :=
by
  sorry

end geometric_progression_ratio_l220_220841


namespace list_scores_lowest_highest_l220_220303

variable (M Q S T : ℕ)

axiom Quay_thinks : Q = T
axiom Marty_thinks : M > T
axiom Shana_thinks : S < T
axiom Tana_thinks : T ≠ max M (max Q (max S T)) ∧ T ≠ min M (min Q (min S T))

theorem list_scores_lowest_highest : (S < T) ∧ (T = Q) ∧ (Q < M) ↔ (S < T) ∧ (T < M) :=
by
  sorry

end list_scores_lowest_highest_l220_220303


namespace same_profit_and_loss_selling_price_l220_220930

theorem same_profit_and_loss_selling_price (CP SP : ℝ) (h₁ : CP = 49) (h₂ : (CP - 42) = (SP - CP)) : SP = 56 :=
by 
  sorry

end same_profit_and_loss_selling_price_l220_220930


namespace find_f_prime_one_l220_220542

noncomputable def f (f'_1 : ℝ) (x : ℝ) := f'_1 * x^3 - 2 * x^2 + 3

theorem find_f_prime_one (f'_1 : ℝ) 
  (h_derivative : ∀ x : ℝ, deriv (f f'_1) x = 3 * f'_1 * x^2 - 4 * x)
  (h_value_at_1 : deriv (f f'_1) 1 = f'_1) :
  f'_1 = 2 :=
by 
  sorry

end find_f_prime_one_l220_220542


namespace license_plate_count_l220_220804

theorem license_plate_count :
  let digits := 10
  let letters := 26
  let positions := 6
  positions * digits^5 * letters^3 = 105456000 := by
  sorry

end license_plate_count_l220_220804


namespace max_sum_consecutive_integers_less_360_l220_220331

theorem max_sum_consecutive_integers_less_360 :
  ∃ n : ℤ, n * (n + 1) < 360 ∧ (n + (n + 1)) = 37 :=
by
  sorry

end max_sum_consecutive_integers_less_360_l220_220331


namespace largest_prime_17p_625_l220_220776

theorem largest_prime_17p_625 (p : ℕ) (h_prime : Nat.Prime p) (h_sqrt : ∃ q, 17 * p + 625 = q^2) : p = 67 :=
by
  sorry

end largest_prime_17p_625_l220_220776


namespace hexagon_tiling_colors_l220_220275

-- Problem Definition
theorem hexagon_tiling_colors (k l : ℕ) (hk : 0 < k ∨ 0 < l) : 
  ∃ n: ℕ, n = k^2 + k * l + l^2 :=
by
  sorry

end hexagon_tiling_colors_l220_220275


namespace range_of_a_l220_220721

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) := 
sorry

end range_of_a_l220_220721


namespace find_g_l220_220392

theorem find_g (x : ℝ) (g : ℝ → ℝ) :
  2 * x^5 - 4 * x^3 + 3 * x^2 + g x = 7 * x^4 - 5 * x^3 + x^2 - 9 * x + 2 →
  g x = -2 * x^5 + 7 * x^4 - x^3 - 2 * x^2 - 9 * x + 2 :=
by
  intro h
  sorry

end find_g_l220_220392


namespace max_apartment_size_l220_220670

theorem max_apartment_size (rate cost per_sqft : ℝ) (budget : ℝ) (h1 : rate = 1.20) (h2 : budget = 864) : cost = 720 :=
by
  sorry

end max_apartment_size_l220_220670


namespace prime_in_A_l220_220138

def is_in_A (n : ℕ) : Prop :=
  ∃ a b : ℤ, n = a^2 + 2 * b^2 ∧ b ≠ 0

theorem prime_in_A (p : ℕ) (hp : Nat.Prime p) (h : is_in_A (p^2)) : is_in_A p :=
by
  sorry

end prime_in_A_l220_220138


namespace num_even_three_digit_numbers_with_sum_of_tens_and_units_10_l220_220552

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def sum_of_tens_and_units_is_ten (n : ℕ) : Prop :=
  (n / 10 % 10) + (n % 10) = 10

theorem num_even_three_digit_numbers_with_sum_of_tens_and_units_10 : 
  ∃! (N : ℕ), (N = 36) ∧ 
               (∀ n : ℕ, is_three_digit n → is_even n → sum_of_tens_and_units_is_ten n →
                         n = 36) := 
sorry

end num_even_three_digit_numbers_with_sum_of_tens_and_units_10_l220_220552


namespace robert_arrival_time_l220_220461

def arrival_time (T : ℕ) : Prop :=
  ∃ D : ℕ, D = 10 * (12 - T) ∧ D = 15 * (13 - T)

theorem robert_arrival_time : arrival_time 15 :=
by
  sorry

end robert_arrival_time_l220_220461


namespace four_digit_perfect_square_palindrome_count_l220_220651

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_perfect_square (n : ℕ) : Prop :=
  let sqrt_n := Nat.sqrt n in sqrt_n * sqrt_n = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem four_digit_perfect_square_palindrome_count : 
  (finset.filter (λ n, is_palindrome n ∧ is_perfect_square n ∧ is_four_digit n) (finset.range 10000)).card = 1 :=
by {
  sorry
}

end four_digit_perfect_square_palindrome_count_l220_220651


namespace divisibility_of_n_l220_220007

def n : ℕ := (2^4 - 1) * (3^6 - 1) * (5^10 - 1) * (7^12 - 1)

theorem divisibility_of_n : 
    (5 ∣ n) ∧ (7 ∣ n) ∧ (11 ∣ n) ∧ (13 ∣ n) := 
by 
  sorry

end divisibility_of_n_l220_220007


namespace sum_arithmetic_sequence_l220_220750

open Nat

noncomputable def arithmetic_sum (a1 d n : ℕ) : ℝ :=
  (2 * a1 + (n - 1) * d) * n / 2

theorem sum_arithmetic_sequence (m n : ℕ) (h1 : m ≠ n) (h2 : m > 0) (h3 : n > 0)
    (S_m S_n : ℝ) (h4 : S_m = m / n) (h5 : S_n = n / m) 
    (a1 d : ℕ) (h6 : S_m = arithmetic_sum a1 d m) (h7 : S_n = arithmetic_sum a1 d n) 
    : arithmetic_sum a1 d (m + n) > 4 :=
by
  sorry

end sum_arithmetic_sequence_l220_220750


namespace find_second_number_l220_220686

theorem find_second_number (G N: ℕ) (h1: G = 101) (h2: 4351 % G = 8) (h3: N % G = 10) : N = 4359 :=
by 
  sorry

end find_second_number_l220_220686


namespace find_fraction_value_l220_220005

variable (a b : ℝ)

theorem find_fraction_value (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : (4 * a + b) / (a - 4 * b) = 3) :
  (a + 4 * b) / (4 * a - b) = 9 / 53 := 
  sorry

end find_fraction_value_l220_220005


namespace cuboid_edge_lengths_l220_220469

theorem cuboid_edge_lengths (
  a b c : ℕ
) (h_volume : a * b * c + a * b + b * c + c * a + a + b + c = 2000) :
  (a = 28 ∧ b = 22 ∧ c = 2) ∨ 
  (a = 28 ∧ b = 2 ∧ c = 22) ∨
  (a = 22 ∧ b = 28 ∧ c = 2) ∨
  (a = 22 ∧ b = 2 ∧ c = 28) ∨
  (a = 2 ∧ b = 28 ∧ c = 22) ∨
  (a = 2 ∧ b = 22 ∧ c = 28) :=
sorry

end cuboid_edge_lengths_l220_220469


namespace inscribed_circles_radii_sum_l220_220210

noncomputable def sum_of_radii (d : ℝ) (r1 r2 : ℝ) : Prop :=
  r1 + r2 = d / 2

theorem inscribed_circles_radii_sum (d : ℝ) (h : d = 23) (r1 r2 : ℝ) (h1 : r1 + r2 = d / 2) :
  r1 + r2 = 23 / 2 :=
by
  rw [h] at h1
  exact h1

end inscribed_circles_radii_sum_l220_220210


namespace four_digit_perfect_square_palindrome_count_l220_220650

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_perfect_square (n : ℕ) : Prop :=
  let sqrt_n := Nat.sqrt n in sqrt_n * sqrt_n = n

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem four_digit_perfect_square_palindrome_count : 
  (finset.filter (λ n, is_palindrome n ∧ is_perfect_square n ∧ is_four_digit n) (finset.range 10000)).card = 1 :=
by {
  sorry
}

end four_digit_perfect_square_palindrome_count_l220_220650


namespace pow_div_eq_l220_220948

theorem pow_div_eq : (8:ℕ) ^ 15 / (64:ℕ) ^ 6 = 512 := by
  have h1 : 64 = 8 ^ 2 := by sorry
  have h2 : (64:ℕ) ^ 6 = (8 ^ 2) ^ 6 := by sorry
  have h3 : (8 ^ 2) ^ 6 = 8 ^ 12 := by sorry
  have h4 : (8:ℕ) ^ 15 / (8 ^ 12) = 8 ^ (15 - 12) := by sorry
  have h5 : 8 ^ 3 = 512 := by sorry
  exact sorry

end pow_div_eq_l220_220948


namespace factorize_expression_1_factorize_expression_2_l220_220010

theorem factorize_expression_1 (m : ℤ) : 
  m^3 - 2 * m^2 - 4 * m + 8 = (m - 2)^2 * (m + 2) := 
sorry

theorem factorize_expression_2 (x y : ℤ) : 
  x^2 - 2 * x * y + y^2 - 9 = (x - y + 3) * (x - y - 3) :=
sorry

end factorize_expression_1_factorize_expression_2_l220_220010


namespace range_of_c_value_of_c_given_perimeter_l220_220862

variables (a b c : ℝ)

-- Question 1: Proving the range of values for c
theorem range_of_c (h1 : a + b = 3 * c - 2) (h2 : a - b = 2 * c - 6) :
  1 < c ∧ c < 6 :=
sorry

-- Question 2: Finding the value of c for a given perimeter
theorem value_of_c_given_perimeter (h1 : a + b = 3 * c - 2) (h2 : a - b = 2 * c - 6) (h3 : a + b + c = 18) :
  c = 5 :=
sorry

end range_of_c_value_of_c_given_perimeter_l220_220862


namespace inequality_y_lt_x_div_4_l220_220110

open Real

/-- Problem statement:
Given x ∈ (0, π / 6) and y ∈ (0, π / 6), and x * tan y = 2 * (1 - cos x),
prove that y < x / 4.
-/
theorem inequality_y_lt_x_div_4
  (x y : ℝ)
  (hx : 0 < x ∧ x < π / 6)
  (hy : 0 < y ∧ y < π / 6)
  (h : x * tan y = 2 * (1 - cos x)) :
  y < x / 4 := sorry

end inequality_y_lt_x_div_4_l220_220110


namespace intersection_of_A_and_B_l220_220717

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5}

-- State the theorem about the intersection of A and B
theorem intersection_of_A_and_B : A ∩ B = {2, 4} :=
  sorry

end intersection_of_A_and_B_l220_220717


namespace inequality_holds_l220_220109

variables {a b c : ℝ}

theorem inequality_holds (h1 : c < b) (h2 : b < a) (h3 : ac < 0) : ab > ac :=
sorry

end inequality_holds_l220_220109


namespace solve_tetrahedron_side_length_l220_220347

noncomputable def side_length_of_circumscribing_tetrahedron (r : ℝ) (tangent_spheres : ℕ) (radius_spheres_equal : ℝ) : ℝ := 
  if h : r = 1 ∧ tangent_spheres = 4 then
    2 + 2 * Real.sqrt 6
  else
    0

theorem solve_tetrahedron_side_length :
  side_length_of_circumscribing_tetrahedron 1 4 1 = 2 + 2 * Real.sqrt 6 :=
by
  sorry

end solve_tetrahedron_side_length_l220_220347


namespace problem1_problem2_l220_220084

open Real

theorem problem1: 
  ((25^(1/3) - 125^(1/2)) / 5^(1/4) = 5^(5/12) - 5^(5/4)) :=
sorry

theorem problem2 (a : ℝ) (h : 0 < a): 
  (a^2 / (a^(1/2) * a^(2/3)) = a^(5/6)) :=
sorry

end problem1_problem2_l220_220084


namespace members_left_for_treasurer_l220_220356

-- Conditions
def total_members : ℕ := 10
def probability_jarry_is_secretary_or_treasurer : ℝ := 0.2

-- Question and proof goal
theorem members_left_for_treasurer :
  ∃ (members_left : ℕ), 
    members_left = total_members - 2 :=
sorry

end members_left_for_treasurer_l220_220356


namespace more_than_half_millet_on_day_5_l220_220154

noncomputable def millet_amount (n : ℕ) : ℚ :=
  1 - (3 / 4)^n

theorem more_than_half_millet_on_day_5 : millet_amount 5 > 1 / 2 :=
by
  sorry

end more_than_half_millet_on_day_5_l220_220154


namespace complex_multiplication_l220_220512

theorem complex_multiplication {i : ℂ} (h : i^2 = -1) : i * (1 - i) = 1 + i := 
by 
  sorry

end complex_multiplication_l220_220512


namespace find_sequence_formula_l220_220536

variable (a : ℕ → ℝ)

noncomputable def sequence_formula := ∀ n : ℕ, a n = Real.sqrt n

lemma sequence_initial : a 1 = 1 :=
sorry

lemma sequence_recursive (n : ℕ) : a (n+1)^2 - a n^2 = 1 :=
sorry

theorem find_sequence_formula : sequence_formula a :=
sorry

end find_sequence_formula_l220_220536


namespace find_p_q_sum_l220_220813

-- Define the conditions
def is_fair_die (n : ℕ) := ∀ i, 1 ≤ i ∧ i ≤ n

def rolls_sum_to_prime (die1 : ℕ) (die2 : ℕ) (die3 : ℕ) (die4 : ℕ) : Prop :=
  Nat.Prime (die1 + die2 + die3 + die4)

-- State the theorem
theorem find_p_q_sum :
  let outcomes := [(a, b, c, d) | a <- fin 6, b <- fin 6, c <- fin 6, d <- fin 4],
      favorable_outcomes := [(a, b, c, d) ∈ outcomes | rolls_sum_to_prime a b c d],
      total_outcomes := 6^3 * 4,
      favorable_count := favorable_outcomes.length,
      p := favorable_count,
      q := total_outcomes - favorable_count 
      -- (p and q are relatively prime)
  in
  (Nat.coprime p q) → (p + q = 149) :=
by
  sorry

end find_p_q_sum_l220_220813


namespace solve_quadratic_and_cubic_eqns_l220_220939

-- Define the conditions as predicates
def eq1 (x : ℝ) : Prop := (x - 1)^2 = 4
def eq2 (x : ℝ) : Prop := (x - 2)^3 = -125

-- State the theorem
theorem solve_quadratic_and_cubic_eqns : 
  (∃ x : ℝ, eq1 x ∧ (x = 3 ∨ x = -1)) ∧ (∃ x : ℝ, eq2 x ∧ x = -3) :=
by
  sorry

end solve_quadratic_and_cubic_eqns_l220_220939


namespace marcia_oranges_l220_220752

noncomputable def averageCost
  (appleCost bananaCost orangeCost : ℝ) 
  (numApples numBananas numOranges : ℝ) : ℝ :=
  (numApples * appleCost + numBananas * bananaCost + numOranges * orangeCost) /
  (numApples + numBananas + numOranges)

theorem marcia_oranges : 
  ∀ (appleCost bananaCost orangeCost avgCost : ℝ) 
  (numApples numBananas numOranges : ℝ),
  appleCost = 2 → 
  bananaCost = 1 → 
  orangeCost = 3 → 
  numApples = 12 → 
  numBananas = 4 → 
  avgCost = 2 → 
  averageCost appleCost bananaCost orangeCost numApples numBananas numOranges = avgCost → 
  numOranges = 4 :=
by 
  intros appleCost bananaCost orangeCost avgCost numApples numBananas numOranges
         h1 h2 h3 h4 h5 h6 h7
  sorry

end marcia_oranges_l220_220752


namespace find_z_l220_220422

theorem find_z (x y z : ℝ) (h : 1 / (x + 1) + 1 / (y + 1) = 1 / z) :
  z = (x + 1) * (y + 1) / (x + y + 2) :=
sorry

end find_z_l220_220422


namespace foreign_stamps_count_l220_220129

-- Define the conditions
variables (total_stamps : ℕ) (more_than_10_years_old : ℕ) (both_foreign_and_old : ℕ) (neither_foreign_nor_old : ℕ)

theorem foreign_stamps_count 
  (h1 : total_stamps = 200)
  (h2 : more_than_10_years_old = 60)
  (h3 : both_foreign_and_old = 20)
  (h4 : neither_foreign_nor_old = 70) : 
  ∃ (foreign_stamps : ℕ), foreign_stamps = 90 :=
by
  -- let foreign_stamps be the variable representing the number of foreign stamps
  let foreign_stamps := total_stamps - neither_foreign_nor_old - more_than_10_years_old + both_foreign_and_old
  use foreign_stamps
  -- the proof will develop here to show that foreign_stamps = 90
  sorry

end foreign_stamps_count_l220_220129


namespace perpendicular_unit_vector_exists_l220_220854

theorem perpendicular_unit_vector_exists :
  ∃ (m n : ℝ), (2 * m + n = 0) ∧ (m^2 + n^2 = 1) ∧ (m = (Real.sqrt 5) / 5) ∧ (n = -(2 * (Real.sqrt 5)) / 5) :=
by
  sorry

end perpendicular_unit_vector_exists_l220_220854


namespace polygon_sides_from_interior_angles_l220_220035

theorem polygon_sides_from_interior_angles (S : ℕ) (h : S = 1260) : S = (9 - 2) * 180 :=
by
  sorry

end polygon_sides_from_interior_angles_l220_220035


namespace sum_of_roots_eq_zero_l220_220855

theorem sum_of_roots_eq_zero :
  ∀ (x : ℝ), x^2 - 7 * |x| + 6 = 0 → (∃ a b c d : ℝ, a + b + c + d = 0) :=
by
  sorry

end sum_of_roots_eq_zero_l220_220855


namespace negative_represents_departure_of_30_tons_l220_220134

theorem negative_represents_departure_of_30_tons (positive_neg_opposites : ∀ x:ℤ, -x = x * (-1))
  (arrival_represents_30 : ∀ x:ℤ, (x = 30) ↔ ("+30" represents arrival of 30 tons of grain)) :
  "-30" represents departure of 30 tons of grain :=
sorry

end negative_represents_departure_of_30_tons_l220_220134


namespace total_paths_from_X_to_Z_l220_220241

variable (X Y Z : Type)
variables (f : X → Y → Z)
variables (g : X → Z)

-- Conditions
def paths_X_to_Y : ℕ := 3
def paths_Y_to_Z : ℕ := 4
def direct_paths_X_to_Z : ℕ := 1

-- Proof problem statement
theorem total_paths_from_X_to_Z : paths_X_to_Y * paths_Y_to_Z + direct_paths_X_to_Z = 13 := sorry

end total_paths_from_X_to_Z_l220_220241


namespace dogs_with_flea_collars_l220_220997

-- Conditions
def T : ℕ := 80
def Tg : ℕ := 45
def B : ℕ := 6
def N : ℕ := 1

-- Goal: prove the number of dogs with flea collars is 40 given the above conditions
theorem dogs_with_flea_collars : ∃ F : ℕ, F = 40 ∧ T = Tg + F - B + N := 
by
  use 40
  sorry

end dogs_with_flea_collars_l220_220997


namespace range_of_a_l220_220544

theorem range_of_a
  (a : ℝ)
  (h1 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 ≥ a)
  (h2 : ∃ x0 : ℝ, x0^2 + 2*a*x0 + 2 - a = 0) :
  a ≤ -2 ∨ a = 1 :=
sorry

end range_of_a_l220_220544


namespace allison_craft_items_l220_220991

def glue_sticks (A B : Nat) : Prop := A = B + 8
def construction_paper (A B : Nat) : Prop := B = 6 * A

theorem allison_craft_items (Marie_glue_sticks Marie_paper_packs : Nat)
    (h1 : Marie_glue_sticks = 15)
    (h2 : Marie_paper_packs = 30) :
    ∃ (Allison_glue_sticks Allison_paper_packs total_items : Nat),
        glue_sticks Allison_glue_sticks Marie_glue_sticks ∧
        construction_paper Allison_paper_packs Marie_paper_packs ∧
        total_items = Allison_glue_sticks + Allison_paper_packs ∧
        total_items = 28 :=
by
    sorry

end allison_craft_items_l220_220991


namespace problem1_problem2_l220_220511

-- Problem 1: Proving the given equation under specified conditions
theorem problem1 (x y : ℝ) (h : x + y ≠ 0) : ((2 * x + 3 * y) / (x + y)) - ((x + 2 * y) / (x + y)) = 1 :=
sorry

-- Problem 2: Proving the given equation under specified conditions
theorem problem2 (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ 1) : ((a^2 - 1) / (a^2 - 4 * a + 4)) / ((a - 1) / (a - 2)) = (a + 1) / (a - 2) :=
sorry

end problem1_problem2_l220_220511


namespace income_is_12000_l220_220636

theorem income_is_12000 (P : ℝ) : (P * 1.02 = 12240) → (P = 12000) :=
by
  intro h
  sorry

end income_is_12000_l220_220636


namespace trig_second_quadrant_l220_220339

theorem trig_second_quadrant (α : ℝ) (h1 : α > π / 2) (h2 : α < π) :
  (|Real.sin α| / Real.sin α) - (|Real.cos α| / Real.cos α) = 2 :=
by
  sorry

end trig_second_quadrant_l220_220339


namespace true_statement_about_M_l220_220147

variable (U : Set ℕ) (M : Set ℕ)
axiom U_def : U = {1, 2, 3, 4, 5}
axiom complement_M_def : U \ M = {1, 3}

theorem true_statement_about_M : 2 ∈ M :=
by 
  rw [U_def, complement_M_def]
  sorry

end true_statement_about_M_l220_220147


namespace count_palindromic_four_digit_perfect_squares_l220_220640

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem count_palindromic_four_digit_perfect_squares : 
  (finset.card {n ∈ (finset.range 10000).filter (λ x, is_four_digit x ∧ is_perfect_square x ∧ is_palindrome x)} = 5) :=
by
  sorry

end count_palindromic_four_digit_perfect_squares_l220_220640


namespace scorpion_needs_10_millipedes_l220_220068

-- Define the number of segments required daily
def total_segments_needed : ℕ := 800

-- Define the segments already consumed by the scorpion
def segments_consumed : ℕ := 60 + 2 * (2 * 60)

-- Calculate the remaining segments needed
def remaining_segments_needed : ℕ := total_segments_needed - segments_consumed

-- Define the segments per millipede
def segments_per_millipede : ℕ := 50

-- Prove that the number of 50-segment millipedes to be eaten is 10
theorem scorpion_needs_10_millipedes 
  (h : remaining_segments_needed = 500) 
  (h2 : 500 / segments_per_millipede = 10) : 
  500 / segments_per_millipede = 10 := by
  sorry

end scorpion_needs_10_millipedes_l220_220068


namespace exists_b_gt_a_divides_l220_220157

theorem exists_b_gt_a_divides (a : ℕ) (h : 0 < a) :
  ∃ b : ℕ, b > a ∧ (1 + 2^a + 3^a) ∣ (1 + 2^b + 3^b) :=
sorry

end exists_b_gt_a_divides_l220_220157


namespace smallest_positive_integer_divisible_12_15_16_exists_l220_220252

theorem smallest_positive_integer_divisible_12_15_16_exists :
  ∃ x : ℕ, x > 0 ∧ 12 ∣ x ∧ 15 ∣ x ∧ 16 ∣ x ∧ x = 240 :=
by sorry

end smallest_positive_integer_divisible_12_15_16_exists_l220_220252


namespace solve_equation_l220_220463

theorem solve_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x^2 - 1 ≠ 0) : (x / (x - 1) = 2 / (x^2 - 1)) → (x = -2) :=
by
  intro h
  sorry

end solve_equation_l220_220463


namespace probability_three_girls_chosen_l220_220977

theorem probability_three_girls_chosen :
  let total_members := 15;
  let boys := 7;
  let girls := 8;
  let total_ways := Nat.choose total_members 3;
  let girls_ways := Nat.choose girls 3;
  total_ways = Nat.choose 15 3 ∧ girls_ways = Nat.choose 8 3 →
  (girls_ways : ℚ) / (total_ways : ℚ) = 8 / 65 := 
by  
  sorry

end probability_three_girls_chosen_l220_220977


namespace problem_inequality_l220_220014

theorem problem_inequality (a b c d : ℝ) (h1 : d ≥ 0) (h2 : a + b = 2) (h3 : c + d = 2) :
  (a^2 + c^2) * (a^2 + d^2) * (b^2 + c^2) * (b^2 + d^2) ≤ 25 :=
by sorry

end problem_inequality_l220_220014


namespace polygon_interior_angle_l220_220821

theorem polygon_interior_angle (n : ℕ) (hn : 3 * (180 - 180 * (n - 2) / n) + 180 = 180 * (n - 2) / n + 180) : n = 9 :=
by {
  sorry
}

end polygon_interior_angle_l220_220821


namespace smallest_positive_integer_l220_220690

theorem smallest_positive_integer :
  ∃ (n a b m : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ n = 153846 ∧
  (n = 10^m * a + b) ∧
  (7 * n = 2 * (10 * b + a)) :=
by
  sorry

end smallest_positive_integer_l220_220690


namespace find_value_l220_220559

theorem find_value (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a - 2 * b)^2 = 25 :=
by
  sorry

end find_value_l220_220559


namespace integral_f_eq_34_l220_220852

noncomputable def f (x : ℝ) := if x ∈ [0, 1] then (1 / Real.pi) * Real.sqrt (1 - x^2) else 2 - x

theorem integral_f_eq_34 :
  ∫ x in (0 : ℝ)..2, f x = 3 / 4 :=
by
  sorry

end integral_f_eq_34_l220_220852


namespace eight_pow_15_div_sixtyfour_pow_6_l220_220944

theorem eight_pow_15_div_sixtyfour_pow_6 :
  8^15 / 64^6 = 512 := by
  sorry

end eight_pow_15_div_sixtyfour_pow_6_l220_220944


namespace store_A_more_advantageous_l220_220796

theorem store_A_more_advantageous (x : ℕ) (h : x > 5) : 
  6000 + 4500 * (x - 1) < 4800 * x := 
by 
  sorry

end store_A_more_advantageous_l220_220796


namespace certain_number_l220_220972

-- Define the conditions as variables
variables {x : ℝ}

-- Define the proof problem
theorem certain_number (h : 0.15 * x = 0.025 * 450) : x = 75 :=
sorry

end certain_number_l220_220972


namespace intersection_of_sets_l220_220414

open Set

def A : Set ℝ := {x | (x - 2) / (x + 1) ≥ 0}

def B : Set ℝ := Ico 0 4  -- Ico stands for interval [0, 4)

theorem intersection_of_sets : A ∩ B = Ico 2 4 :=
by 
  sorry

end intersection_of_sets_l220_220414


namespace allison_total_craft_supplies_l220_220992

theorem allison_total_craft_supplies (M_glue : ℕ) (M_paper : ℕ) (A_glue_more : ℕ) (M_paper_ratio : ℕ) :
  M_glue = 15 → M_paper = 30 → A_glue_more = 8 → M_paper_ratio = 6 →
  let A_glue := M_glue + A_glue_more in
  let A_paper := M_paper / M_paper_ratio in
  A_glue + A_paper = 28 :=
by
  intros h1 h2 h3 h4
  let A_glue := 15 + 8
  let A_paper := 30 / 6
  sorry

end allison_total_craft_supplies_l220_220992


namespace total_rooms_count_l220_220294

noncomputable def apartment_area : ℕ := 160
noncomputable def living_room_area : ℕ := 60
noncomputable def other_room_area : ℕ := 20

theorem total_rooms_count (A : apartment_area = 160) (L : living_room_area = 60) (O : other_room_area = 20) :
  1 + (apartment_area - living_room_area) / other_room_area = 6 :=
by
  sorry

end total_rooms_count_l220_220294


namespace polynomial_exists_int_coeff_l220_220970

theorem polynomial_exists_int_coeff (n : ℕ) (hn : n > 1) : 
  ∃ P : Polynomial ℤ × Polynomial ℤ × Polynomial ℤ → Polynomial ℤ, 
  ∀ x : Polynomial ℤ, P ⟨x^n, x^(n+1), x + x^(n+2)⟩ = x :=
by sorry

end polynomial_exists_int_coeff_l220_220970


namespace chris_birthday_days_l220_220829

theorem chris_birthday_days (mod : ℕ → ℕ → ℕ) (day_of_week : ℕ → ℕ) :
  (mod 75 7 = 5) ∧ (mod 30 7 = 2) →
  (day_of_week 0 = 1) →
  (day_of_week 75 = 6) ∧ (day_of_week 30 = 3) := 
sorry

end chris_birthday_days_l220_220829


namespace repeating_decimal_to_fraction_l220_220523

noncomputable def repeating_decimal := 0.6 + 3 / 100

theorem repeating_decimal_to_fraction :
  repeating_decimal = 19 / 30 :=
  sorry

end repeating_decimal_to_fraction_l220_220523


namespace determine_a_of_parallel_lines_l220_220835

theorem determine_a_of_parallel_lines (a : ℝ) :
  (∀ x y : ℝ, 3 * y - 3 * a = 9 * x ↔ y = 3 * x + a) →
  (∀ x y : ℝ, y - 2 = (a - 3) * x ↔ y = (a - 3) * x + 2) →
  (∀ x y : ℝ, 3 * y - 3 * a = 9 * x → y - 2 = (a - 3) * x → 3 = a - 3) →
  a = 6 :=
by
  sorry

end determine_a_of_parallel_lines_l220_220835


namespace no_visit_days_in_2021_l220_220295

theorem no_visit_days_in_2021 (n1 n2 n3 n4 : ℕ) (days_in_year : ℕ) :
  n1 = 6 → n2 = 8 → n3 = 9 → n4 = 12 → days_in_year = 365 →
  let visits n := days_in_year / n in
  let lcm_visits a b := days_in_year / Nat.lcm a b in
  let lcm_visits_three a b c := days_in_year / Nat.lcm (Nat.lcm a b) c in
  let lcm_visits_four a b c d := days_in_year / Nat.lcm (Nat.lcm a b) (Nat.lcm c d) in
  let total_visits := visits n1 + visits n2 + visits n3 + visits n4 -
                      (lcm_visits n1 n2 + lcm_visits n1 n3 + lcm_visits n1 n4 + lcm_visits n2 n3 + lcm_visits n2 n4 + lcm_visits n3 n4) +
                      (lcm_visits_three n1 n2 n3 + lcm_visits_three n1 n2 n4 + lcm_visits_three n1 n3 n4 + lcm_visits_three n2 n3 n4) -
                      lcm_visits_four n1 n2 n3 n4 in
  days_in_year - total_visits = 280 :=
begin
  intros,
  sorry
end

end no_visit_days_in_2021_l220_220295


namespace exponential_inequality_l220_220459

-- Define the conditions
variables {n : ℤ} {x : ℝ}

theorem exponential_inequality 
  (h1 : n ≥ 2) 
  (h2 : |x| < 1) 
  : 2^n > (1 - x)^n + (1 + x)^n :=
sorry

end exponential_inequality_l220_220459


namespace airplane_average_speed_l220_220995

theorem airplane_average_speed :
  ∃ v : ℝ, 
  (1140 = 12 * (0.9 * v) + 26 * (1.2 * v)) ∧ 
  v = 27.14 := 
by
  sorry

end airplane_average_speed_l220_220995


namespace solve_equation_l220_220307

theorem solve_equation : ∃ x : ℝ, 3 * x + 2 * (x - 2) = 6 ↔ x = 2 :=
by
  sorry

end solve_equation_l220_220307


namespace susans_average_speed_l220_220592

theorem susans_average_speed :
  ∀ (total_distance first_leg_distance second_leg_distance : ℕ)
    (first_leg_speed second_leg_speed : ℕ)
    (total_time : ℚ),
    first_leg_distance = 40 →
    second_leg_distance = 20 →
    first_leg_speed = 15 →
    second_leg_speed = 60 →
    total_distance = first_leg_distance + second_leg_distance →
    total_time = (first_leg_distance / first_leg_speed : ℚ) + (second_leg_distance / second_leg_speed : ℚ) →
    total_distance / total_time = 20 :=
by
  sorry

end susans_average_speed_l220_220592


namespace allison_total_craft_supplies_l220_220993

theorem allison_total_craft_supplies (M_glue : ℕ) (M_paper : ℕ) (A_glue_more : ℕ) (M_paper_ratio : ℕ) :
  M_glue = 15 → M_paper = 30 → A_glue_more = 8 → M_paper_ratio = 6 →
  let A_glue := M_glue + A_glue_more in
  let A_paper := M_paper / M_paper_ratio in
  A_glue + A_paper = 28 :=
by
  intros h1 h2 h3 h4
  let A_glue := 15 + 8
  let A_paper := 30 / 6
  sorry

end allison_total_craft_supplies_l220_220993


namespace people_between_katya_and_polina_l220_220968

-- Definitions based on given conditions
def is_next_to (a b : ℕ) : Prop := (b = a + 1) ∨ (b = a - 1)
def position_alena : ℕ := 1
def position_lena : ℕ := 5
def position_sveta (pos_sveta : ℕ) : Prop := pos_sveta + 1 = position_lena
def position_katya (pos_katya : ℕ) : Prop := pos_katya = 3
def position_polina (pos_polina : ℕ) : Prop := (is_next_to position_alena pos_polina)

-- The question: prove the number of people between Katya and Polina is 0
theorem people_between_katya_and_polina : 
  ∃ (pos_katya pos_polina : ℕ),
    position_katya pos_katya ∧ 
    position_polina pos_polina ∧ 
    pos_polina + 1 = pos_katya ∧
    pos_katya = 3 ∧ pos_polina = 2 := 
sorry

end people_between_katya_and_polina_l220_220968


namespace number_of_sheets_l220_220180

theorem number_of_sheets (S E : ℕ) 
  (h1 : S - E = 40)
  (h2 : 5 * E = S) : 
  S = 50 := by 
  sorry

end number_of_sheets_l220_220180


namespace probability_no_shaded_square_l220_220202

theorem probability_no_shaded_square :
  let n := (2006 * 2005) / 2,
      m := 1003 * 1003 in
  (n - m) / n = 1002 / 2005 :=
by 
  sorry

end probability_no_shaded_square_l220_220202


namespace three_same_colored_balls_l220_220037

theorem three_same_colored_balls (balls : ℕ) (color_count : ℕ) (balls_per_color : ℕ) (h1 : balls = 60) (h2 : color_count = balls / balls_per_color) (h3 : balls_per_color = 6) :
  ∃ n, n = 21 ∧ (∀ picks : ℕ, picks ≥ n → ∃ c, ∃ k ≥ 3, k ≤ balls_per_color ∧ (c < color_count) ∧ (picks / c = k)) :=
sorry

end three_same_colored_balls_l220_220037


namespace complete_set_of_events_l220_220483

-- Define the range of numbers on a die
def die_range := {n : ℕ | 1 ≤ n ∧ n ≤ 6}

-- Define what an outcome is
def outcome := { p : ℕ × ℕ | p.1 ∈ die_range ∧ p.2 ∈ die_range }

-- The theorem stating the complete set of outcomes
theorem complete_set_of_events : outcome = { p : ℕ × ℕ | 1 ≤ p.1 ∧ p.1 ≤ 6 ∧ 1 ≤ p.2 ∧ p.2 ≤ 6 } :=
by sorry

end complete_set_of_events_l220_220483


namespace unwanted_texts_per_week_l220_220279

-- Define the conditions as constants
def messages_per_day_old : ℕ := 20
def messages_per_day_new : ℕ := 55
def days_per_week : ℕ := 7

-- Define the theorem stating the problem
theorem unwanted_texts_per_week (messages_per_day_old messages_per_day_new days_per_week 
  : ℕ) : (messages_per_day_new - messages_per_day_old) * days_per_week = 245 :=
by
  sorry

end unwanted_texts_per_week_l220_220279


namespace find_angle_D_l220_220734

-- Define the given angles and conditions
def angleA := 30
def angleB (D : ℝ) := 2 * D
def angleC (D : ℝ) := D + 40
def sum_of_angles (A B C D : ℝ) := A + B + C + D = 360

theorem find_angle_D (D : ℝ) (hA : angleA = 30) (hB : angleB D = 2 * D) (hC : angleC D = D + 40) (hSum : sum_of_angles angleA (angleB D) (angleC D) D):
  D = 72.5 :=
by
  -- Proof is omitted
  sorry

end find_angle_D_l220_220734


namespace exists_a_f_has_two_zeros_range_of_a_for_f_eq_g_l220_220860

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a + 2 * Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem exists_a_f_has_two_zeros (a : ℝ) :
  (0 < a ∧ a < 2) ∨ (-2 < a ∧ a < 0) → ∃ x₁ x₂ : ℝ, (0 ≤ x₁ ∧ x₁ ≤ 2 * Real.pi ∧ f x₁ a = 0) ∧
  (0 ≤ x₂ ∧ x₂ ≤ 2 * Real.pi ∧ f x₂ a = 0) ∧ x₁ ≠ x₂ := sorry

theorem range_of_a_for_f_eq_g :
  ∀ a : ℝ, a ∈ Set.Icc (-2 : ℝ) (3 : ℝ) →
  ∃ x₁ : ℝ, x₁ ∈ Set.Icc (0 : ℝ) (2 * Real.pi) ∧ f x₁ a = g 2 ∧
  ∃ x₂ : ℝ, x₂ ∈ Set.Icc (1 : ℝ) (2 : ℝ) ∧ f x₁ a = g x₂ := sorry

end exists_a_f_has_two_zeros_range_of_a_for_f_eq_g_l220_220860


namespace range_of_a_l220_220089

def my_Op (a b : ℝ) : ℝ := a - 2 * b

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, my_Op x 3 > 0 → my_Op x a > a) ↔ (∀ x : ℝ, x > 6 → x > 3 * a) → a ≤ 2 :=
by sorry

end range_of_a_l220_220089


namespace radius_of_circle_B_l220_220933

theorem radius_of_circle_B (diam_A : ℝ) (factor : ℝ) (r_A r_B : ℝ) 
  (h1 : diam_A = 80) 
  (h2 : r_A = diam_A / 2) 
  (h3 : r_A = factor * r_B) 
  (h4 : factor = 4) : r_B = 10 := 
by 
  sorry

end radius_of_circle_B_l220_220933


namespace complete_set_contains_all_rationals_l220_220664

theorem complete_set_contains_all_rationals (T : Set ℚ) (hT : ∀ (p q : ℚ), p / q ∈ T → p / (p + q) ∈ T ∧ q / (p + q) ∈ T) (r : ℚ) : 
  (r = 1 ∨ r = 1 / 2) → (∀ x : ℚ, 0 < x ∧ x < 1 → x ∈ T) :=
by
  sorry

end complete_set_contains_all_rationals_l220_220664


namespace strap_mask_probability_l220_220637

theorem strap_mask_probability 
  (p_regular_medical : ℝ)
  (p_surgical : ℝ)
  (p_strap_regular : ℝ)
  (p_strap_surgical : ℝ)
  (h_regular_medical : p_regular_medical = 0.8)
  (h_surgical : p_surgical = 0.2)
  (h_strap_regular : p_strap_regular = 0.1)
  (h_strap_surgical : p_strap_surgical = 0.2) :
  (p_regular_medical * p_strap_regular + p_surgical * p_strap_surgical) = 0.12 :=
by
  rw [h_regular_medical, h_surgical, h_strap_regular, h_strap_surgical]
  -- proof will go here
  sorry

end strap_mask_probability_l220_220637


namespace list_price_proof_l220_220668

theorem list_price_proof (x : ℝ) :
  (0.15 * (x - 15) = 0.25 * (x - 25)) → x = 40 :=
by
  sorry

end list_price_proof_l220_220668


namespace helga_tried_on_66_pairs_of_shoes_l220_220419

variables 
  (n1 n2 n3 n4 n5 n6 : ℕ)
  (h1 : n1 = 7)
  (h2 : n2 = n1 + 2)
  (h3 : n3 = 0)
  (h4 : n4 = 2 * (n1 + n2 + n3))
  (h5 : n5 = n2 - 3)
  (h6 : n6 = n1 + 5)
  (total : ℕ := n1 + n2 + n3 + n4 + n5 + n6)

theorem helga_tried_on_66_pairs_of_shoes : total = 66 :=
by sorry

end helga_tried_on_66_pairs_of_shoes_l220_220419


namespace sugar_total_more_than_two_l220_220509

noncomputable def x (p q : ℝ) : ℝ :=
p / q

noncomputable def y (p q : ℝ) : ℝ :=
q / p

theorem sugar_total_more_than_two (p q : ℝ) (hpq : p ≠ q) :
  x p q + y p q > 2 :=
by sorry

end sugar_total_more_than_two_l220_220509


namespace inclination_angle_of_y_axis_l220_220468

theorem inclination_angle_of_y_axis : 
  ∀ (l : ℝ), l = 90 :=
sorry

end inclination_angle_of_y_axis_l220_220468


namespace John_avg_speed_l220_220056

theorem John_avg_speed :
  ∀ (initial final : ℕ) (time : ℕ),
    initial = 27372 →
    final = 27472 →
    time = 4 →
    ((final - initial) / time) = 25 :=
by
  intros initial final time h_initial h_final h_time
  sorry

end John_avg_speed_l220_220056


namespace avg_page_count_per_essay_l220_220076

-- Definitions based on conditions
def students : ℕ := 15
def first_group_students : ℕ := 5
def second_group_students : ℕ := 5
def third_group_students : ℕ := 5
def pages_first_group : ℕ := 2
def pages_second_group : ℕ := 3
def pages_third_group : ℕ := 1
def total_pages := first_group_students * pages_first_group +
                   second_group_students * pages_second_group +
                   third_group_students * pages_third_group

-- Statement to prove
theorem avg_page_count_per_essay :
  total_pages / students = 2 :=
by
  sorry

end avg_page_count_per_essay_l220_220076


namespace min_value_expression_l220_220105

theorem min_value_expression {x y : ℝ} : 
  2 * x + y - 3 = 0 →
  x + 2 * y - 1 = 0 →
  (5 * x^2 + 8 * x * y + 5 * y^2 - 14 * x - 10 * y + 30) / (4 - x^2 - 10 * x * y - 25 * y^2) ^ (7 / 2) = 5 / 32 :=
by
  sorry

end min_value_expression_l220_220105


namespace unique_root_conditions_l220_220684

theorem unique_root_conditions (m : ℝ) (x y : ℝ) :
  (x^2 = 2 * abs x ∧ abs x - y - m = 1 - y^2) ↔ m = 3 / 4 := sorry

end unique_root_conditions_l220_220684


namespace multiply_fractions_l220_220828

theorem multiply_fractions :
  (2 / 3) * (5 / 7) * (8 / 9) = 80 / 189 :=
by sorry

end multiply_fractions_l220_220828


namespace sum_areas_of_square_and_rectangle_l220_220473

theorem sum_areas_of_square_and_rectangle (s w l : ℝ) 
  (h1 : s^2 + w * l = 130)
  (h2 : 4 * s - 2 * (w + l) = 20)
  (h3 : l = 2 * w) : 
  s^2 + 2 * w^2 = 118 :=
by
  -- Provide space for proof
  sorry

end sum_areas_of_square_and_rectangle_l220_220473


namespace avg_ballpoint_pens_per_day_l220_220057

theorem avg_ballpoint_pens_per_day (bundles_sold : ℕ) (pens_per_bundle : ℕ) (days : ℕ) (total_pens : ℕ) (avg_per_day : ℕ) 
  (h1 : bundles_sold = 15)
  (h2 : pens_per_bundle = 40)
  (h3 : days = 5)
  (h4 : total_pens = bundles_sold * pens_per_bundle)
  (h5 : avg_per_day = total_pens / days) :
  avg_per_day = 120 :=
by
  -- placeholder proof
  sorry

end avg_ballpoint_pens_per_day_l220_220057


namespace four_digit_perfect_square_palindrome_count_l220_220646

def is_palindrome (n : ℕ) : Prop :=
  let str := n.repr
  str = str.reverse

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n
  
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_four_digit_perfect_square_palindrome (n : ℕ) : Prop :=
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n

theorem four_digit_perfect_square_palindrome_count : 
  { n : ℕ | is_four_digit_perfect_square_palindrome n }.card = 4 := 
sorry

end four_digit_perfect_square_palindrome_count_l220_220646


namespace minimum_value_of_expression_l220_220103

noncomputable def expression (x y : ℝ) : ℝ :=
  (5 * x^2 + 8 * x * y + 5 * y^2 - 14 * x - 10 * y + 30) /
  (4 - x^2 - 10 * x * y - 25 * y^2) ^ (7 / 2)

theorem minimum_value_of_expression : 
  ∃ x y : ℝ, expression x y = 5/32 :=
by
  sorry

end minimum_value_of_expression_l220_220103


namespace smallest_k_l220_220843

theorem smallest_k (k : ℕ) : 
  (k > 0 ∧ (k*(k+1)*(2*k+1)/6) % 400 = 0) → k = 800 :=
by
  sorry

end smallest_k_l220_220843


namespace boats_distance_one_minute_before_collision_l220_220345

noncomputable def distance_between_boats_one_minute_before_collision
  (speed_boat1 : ℝ) (speed_boat2 : ℝ) (initial_distance : ℝ) : ℝ :=
  let relative_speed := speed_boat1 + speed_boat2
  let relative_speed_per_minute := relative_speed / 60
  let time_to_collide := initial_distance / relative_speed_per_minute
  let distance_one_minute_before := initial_distance - (relative_speed_per_minute * (time_to_collide - 1))
  distance_one_minute_before

theorem boats_distance_one_minute_before_collision :
  distance_between_boats_one_minute_before_collision 5 21 20 = 0.4333 :=
by
  -- Proof skipped
  sorry

end boats_distance_one_minute_before_collision_l220_220345


namespace roots_of_equation_l220_220969

theorem roots_of_equation (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∀ x : ℝ, a^2 * (x - b) / (a - b) * (x - c) / (a - c) + b^2 * (x - a) / (b - a) * (x - c) / (b - c) + c^2 * (x - a) / (c - a) * (x - b) / (c - b) = x^2 :=
by
  intros
  sorry

end roots_of_equation_l220_220969


namespace trajectory_eq_of_moving_point_Q_l220_220708

-- Define the conditions and the correct answer
theorem trajectory_eq_of_moving_point_Q 
(a b : ℝ) (h : a > b) (h_pos : b > 0)
(P Q : ℝ × ℝ)
(h_ellipse : (P.1^2) / (a^2) + (P.2^2) / (b^2) = 1)
(h_Q : Q = (P.1 * 2, P.2 * 2)) :
  (Q.1^2) / (4 * a^2) + (Q.2^2) / (4 * b^2) = 1 :=
by 
  sorry

end trajectory_eq_of_moving_point_Q_l220_220708


namespace find_oxygen_weight_l220_220687

-- Definitions of given conditions
def molecular_weight : ℝ := 68
def weight_hydrogen : ℝ := 1
def weight_chlorine : ℝ := 35.5

-- Definition of unknown atomic weight of oxygen
def weight_oxygen : ℝ := 15.75

-- Mathematical statement to prove
theorem find_oxygen_weight :
  weight_hydrogen + weight_chlorine + 2 * weight_oxygen = molecular_weight := by
sorry

end find_oxygen_weight_l220_220687


namespace quadrilateral_area_lt_one_l220_220561

theorem quadrilateral_area_lt_one 
  (a b c d : ℝ) 
  (h_a : a < 1) 
  (h_b : b < 1) 
  (h_c : c < 1) 
  (h_d : d < 1) 
  (h_pos_a : 0 ≤ a)
  (h_pos_b : 0 ≤ b)
  (h_pos_c : 0 ≤ c)
  (h_pos_d : 0 ≤ d) :
  ∃ (area : ℝ), area < 1 :=
by
  sorry

end quadrilateral_area_lt_one_l220_220561


namespace four_point_questions_l220_220058

theorem four_point_questions (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : y = 10 := 
sorry

end four_point_questions_l220_220058


namespace find_m_solve_inequality_l220_220714

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := m - |x - 2|

theorem find_m (m : ℝ) : (∀ x : ℝ, m - |x| ≥ 0 ↔ x ∈ [-1, 1]) → m = 1 :=
by
  sorry

theorem solve_inequality (x : ℝ) : |x + 1| + |x - 2| > 4 * 1 ↔ x < -3 / 2 ∨ x > 5 / 2 :=
by
  sorry

end find_m_solve_inequality_l220_220714


namespace triangle_right_l220_220477

theorem triangle_right (a b c : ℝ) (h₀ : a ≠ c) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : ∃ x₀ : ℝ, x₀ ≠ 0 ∧ x₀^2 + 2 * a * x₀ + b^2 = 0 ∧ x₀^2 + 2 * c * x₀ - b^2 = 0) :
  a^2 = b^2 + c^2 := 
sorry

end triangle_right_l220_220477


namespace part_I_part_II_l220_220115

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + ((2 * a^2) / x) + x

theorem part_I (a : ℝ) (h : a > 0) : 
  (∃ x : ℝ, x = 1 ∧ deriv (f a) x = -2) → a = 3 / 2 :=
sorry

theorem part_II (a : ℝ) (h : a = 3 / 2) : 
  (∀ x : ℝ, 0 < x ∧ x < 3 / 2 → deriv (f a) x < 0) ∧ 
  (∀ x : ℝ, x > 3 / 2 → deriv (f a) x > 0) :=
sorry

end part_I_part_II_l220_220115


namespace mike_total_investment_l220_220451

variable (T : ℝ)
variable (H1 : 0.09 * 1800 + 0.11 * (T - 1800) = 624)

theorem mike_total_investment : T = 6000 :=
by
  sorry

end mike_total_investment_l220_220451


namespace infinite_integer_solutions_l220_220008

theorem infinite_integer_solutions 
  (a b c k D x0 y0 : ℤ) 
  (hD_pos : D = b^2 - 4 * a * c) 
  (hD_non_square : (∀ n : ℤ, D ≠ n^2)) 
  (hk_nonzero : k ≠ 0) 
  (h_initial_sol : a * x0^2 + b * x0 * y0 + c * y0^2 = k) :
  ∃ (X Y : ℤ), a * X^2 + b * X * Y + c * Y^2 = k ∧
  (∀ (m : ℕ), ∃ (Xm Ym : ℤ), a * Xm^2 + b * Xm * Ym + c * Ym^2 = k ∧
  (Xm, Ym) ≠ (x0, y0)) :=
sorry

end infinite_integer_solutions_l220_220008


namespace unit_vector_perpendicular_to_a_l220_220853

theorem unit_vector_perpendicular_to_a :
  ∃ (m n : ℝ), 2 * m + n = 0 ∧ m^2 + n^2 = 1 ∧ m = sqrt 5 / 5 ∧ n = -2 * sqrt 5 / 5 :=
by
  sorry

end unit_vector_perpendicular_to_a_l220_220853


namespace mike_max_marks_l220_220344

theorem mike_max_marks (m : ℕ) (h : 30 * m = 237 * 10) : m = 790 := by
  sorry

end mike_max_marks_l220_220344


namespace boys_girls_relation_l220_220826

theorem boys_girls_relation (b g : ℕ) :
  (∃ b, 3 + (b - 1) * 2 = g) → b = (g - 1) / 2 :=
by
  intro h
  sorry

end boys_girls_relation_l220_220826


namespace no_real_roots_of_quadratic_l220_220421

theorem no_real_roots_of_quadratic (k : ℝ) (h : 12 - 3 * k < 0) : ∀ (x : ℝ), ¬ (x^2 + 4 * x + k = 0) := by
  sorry

end no_real_roots_of_quadratic_l220_220421


namespace negation_of_at_most_four_is_at_least_five_l220_220170

theorem negation_of_at_most_four_is_at_least_five :
  (∀ n : ℕ, n ≤ 4) ↔ (∃ n : ℕ, n ≥ 5) := 
sorry

end negation_of_at_most_four_is_at_least_five_l220_220170


namespace dealer_sold_BMWs_l220_220499

theorem dealer_sold_BMWs (total_cars : ℕ) (ford_pct toyota_pct nissan_pct bmw_pct : ℝ)
  (h_total_cars : total_cars = 300)
  (h_ford_pct : ford_pct = 0.1)
  (h_toyota_pct : toyota_pct = 0.2)
  (h_nissan_pct : nissan_pct = 0.3)
  (h_bmw_pct : bmw_pct = 1 - (ford_pct + toyota_pct + nissan_pct)) :
  total_cars * bmw_pct = 120 := by
  sorry

end dealer_sold_BMWs_l220_220499


namespace divisible_by_pow3_l220_220761

-- Define the digit sequence function
def num_with_digits (a n : Nat) : Nat :=
  a * ((10 ^ (3 ^ n) - 1) / 9)

-- Main theorem statement
theorem divisible_by_pow3 (a n : Nat) (h_pos : 0 < n) : (num_with_digits a n) % (3 ^ n) = 0 := 
by
  sorry

end divisible_by_pow3_l220_220761


namespace fraction_of_AD_eq_BC_l220_220457

theorem fraction_of_AD_eq_BC (x y : ℝ) (B C D A : ℝ) 
  (h1 : B < C) 
  (h2 : C < D)
  (h3 : D < A) 
  (hBD : B < D)
  (hCD : C < D)
  (hAD : A = D)
  (hAB : A - B = 3 * (D - B)) 
  (hAC : A - C = 7 * (D - C))
  (hx_eq : x = 2 * y) 
  (hADx : A - D = 4 * x)
  (hADy : A - D = 8 * y)
  : (C - B) = 1/8 * (A - D) := 
sorry

end fraction_of_AD_eq_BC_l220_220457


namespace log_difference_l220_220558

theorem log_difference {x y a : ℝ} (h : Real.log x - Real.log y = a) :
  Real.log ((x / 2)^3) - Real.log ((y / 2)^3) = 3 * a :=
by 
  sorry

end log_difference_l220_220558


namespace odd_operations_l220_220874

theorem odd_operations (a b : ℤ) (ha : ∃ k : ℤ, a = 2 * k + 1) (hb : ∃ j : ℤ, b = 2 * j + 1) :
  (∃ k : ℤ, (a * b) = 2 * k + 1) ∧ (∃ m : ℤ, a^2 = 2 * m + 1) :=
by {
  sorry
}

end odd_operations_l220_220874


namespace allison_total_supply_items_is_28_l220_220989

/-- Define the number of glue sticks Marie bought --/
def marie_glue_sticks : ℕ := 15
/-- Define the number of packs of construction paper Marie bought --/
def marie_construction_paper_packs : ℕ := 30
/-- Define the number of glue sticks Allison bought --/
def allison_glue_sticks : ℕ := marie_glue_sticks + 8
/-- Define the number of packs of construction paper Allison bought --/
def allison_construction_paper_packs : ℕ := marie_construction_paper_packs / 6
/-- Calculation of the total number of craft supply items Allison bought --/
def allison_total_supply_items : ℕ := allison_glue_sticks + allison_construction_paper_packs

/-- Prove that the total number of craft supply items Allison bought is equal to 28. --/
theorem allison_total_supply_items_is_28 : allison_total_supply_items = 28 :=
by sorry

end allison_total_supply_items_is_28_l220_220989


namespace impossible_to_obtain_one_l220_220701

theorem impossible_to_obtain_one (N : ℕ) (h : N % 3 = 0) : ¬(∃ k : ℕ, (∀ m : ℕ, (∃ q : ℕ, (N + 3 * m = 5 * q) ∧ (q = 1 → m + 1 ≤ k)))) :=
sorry

end impossible_to_obtain_one_l220_220701


namespace total_value_of_coins_is_correct_l220_220775

-- Definitions for the problem conditions
def number_of_dimes : ℕ := 22
def number_of_quarters : ℕ := 10
def value_of_dime : ℝ := 0.10
def value_of_quarter : ℝ := 0.25
def total_value_of_dimes : ℝ := number_of_dimes * value_of_dime
def total_value_of_quarters : ℝ := number_of_quarters * value_of_quarter
def total_value : ℝ := total_value_of_dimes + total_value_of_quarters

-- Theorem statement
theorem total_value_of_coins_is_correct : total_value = 4.70 := sorry

end total_value_of_coins_is_correct_l220_220775


namespace marching_band_members_l220_220316

theorem marching_band_members :
  ∃ (n : ℕ), 100 < n ∧ n < 200 ∧
             n % 4 = 1 ∧
             n % 5 = 2 ∧
             n % 7 = 3 :=
  by sorry

end marching_band_members_l220_220316


namespace correct_sentence_completion_l220_220085

-- Define the possible options
inductive Options
| A : Options  -- "However he was reminded frequently"
| B : Options  -- "No matter he was reminded frequently"
| C : Options  -- "However frequently he was reminded"
| D : Options  -- "No matter he was frequently reminded"

-- Define the correctness condition
def correct_option : Options := Options.C

-- Define the proof problem
theorem correct_sentence_completion (opt : Options) : opt = correct_option :=
by sorry

end correct_sentence_completion_l220_220085


namespace secant_length_problem_l220_220400

theorem secant_length_problem (tangent_length : ℝ) (internal_segment_length : ℝ) (external_segment_length : ℝ) 
    (h1 : tangent_length = 18) (h2 : internal_segment_length = 27) : external_segment_length = 9 :=
by
  sorry

end secant_length_problem_l220_220400


namespace normal_distribution_probability_l220_220258

noncomputable def xi : Type := sorry -- Define the random variable

def mean : ℝ := 2
def variance : ℝ := sorry -- placeholder for sigma^2
def P_xi_le_4 : ℝ := 0.84

-- Defining the problem: Prove that P(ξ ≤ 0) = 0.16 given the above conditions.
theorem normal_distribution_probability (xi : ℝ → ℝ) (h_norm : ∀ x, xi x = normalPDF mean variance x) :
  P_xi_le_4 = 0.84 → (xi 0) = 0.16 :=
by
  sorry

end normal_distribution_probability_l220_220258


namespace checkerboard_disc_coverage_l220_220359

/-- A circular disc with a diameter of 5 units is placed on a 10 x 10 checkerboard with each square having a side length of 1 unit such that the centers of both the disc and the checkerboard coincide.
    Prove that the number of checkerboard squares that are completely covered by the disc is 36. -/
theorem checkerboard_disc_coverage :
  let diameter : ℝ := 5
  let radius : ℝ := diameter / 2
  let side_length : ℝ := 1
  let board_size : ℕ := 10
  let disc_center : ℝ × ℝ := (board_size / 2, board_size / 2)
  ∃ (count : ℕ), count = 36 := 
  sorry

end checkerboard_disc_coverage_l220_220359


namespace proposition_induction_l220_220292

theorem proposition_induction {P : ℕ → Prop} (h : ∀ n, P n → P (n + 1)) (hn : ¬ P 7) : ¬ P 6 :=
by
  sorry

end proposition_induction_l220_220292


namespace residue_neg_437_mod_13_l220_220516

theorem residue_neg_437_mod_13 : (-437) % 13 = 5 :=
by
  sorry

end residue_neg_437_mod_13_l220_220516


namespace vasya_wins_l220_220618

-- Definition of the game and players
inductive Player
| Vasya : Player
| Petya : Player

-- Define the problem conditions
structure Game where
  initial_piles : ℕ := 1      -- Initially, there is one pile
  players_take_turns : Bool := true
  take_or_divide : Bool := true
  remove_last_wins : Bool := true
  vasya_first_but_cannot_take_initially : Bool := true

-- Define the function to determine the winner
def winner_of_game (g : Game) : Player :=
  if g.initial_piles = 1 ∧ g.vasya_first_but_cannot_take_initially then Player.Vasya else Player.Petya

-- Define the theorem stating Vasya will win given the game conditions
theorem vasya_wins : ∀ (g : Game), g = {
    initial_piles := 1,
    players_take_turns := true,
    take_or_divide := true,
    remove_last_wins := true,
    vasya_first_but_cannot_take_initially := true
} → winner_of_game g = Player.Vasya := by
  -- Insert proof here
  sorry

end vasya_wins_l220_220618


namespace pow_div_l220_220951

theorem pow_div (x : ℕ) (a b c d : ℕ) (h1 : x^b = d) (h2 : x^(a*d) = c) : c / (d^b) = 512 := by
  sorry

end pow_div_l220_220951


namespace solve_sin_cos_eq_l220_220590

namespace MathProof

open Real Int

-- Define the integer part (floor) of real functions
noncomputable def int_part (a : ℝ) : ℤ := int.floor a

-- Problem statement
theorem solve_sin_cos_eq (x : ℝ) :
  int_part (sin (2 * x)) - 2 * int_part (cos x) = 3 * int_part (sin (3 * x)) ↔
  ∃ n : ℤ, x ∈ (2 * π * n, π / 6 + 2 * π * n) ∪ (π / 6 + 2 * π * n, π / 4 + 2 * π * n) ∪ (π / 4 + 2 * π * n, π / 3 + 2 * π * n) := by
  sorry

end MathProof

end solve_sin_cos_eq_l220_220590


namespace sum_of_six_primes_even_l220_220929

/-- If A, B, and C are positive integers such that A, B, C, A-B, A+B, and A+B+C are all prime numbers, 
    and B is specifically the prime number 2,
    then the sum of these six primes is even. -/
theorem sum_of_six_primes_even (A B C : ℕ) (hA : Prime A) (hB : Prime B) (hC : Prime C) 
    (h1 : Prime (A - B)) (h2 : Prime (A + B)) (h3 : Prime (A + B + C)) (hB_eq_two : B = 2) : 
    Even (A + B + C + (A - B) + (A + B) + (A + B + C)) :=
by
  sorry

end sum_of_six_primes_even_l220_220929


namespace sum_quotient_reciprocal_eq_one_point_thirty_five_l220_220607

theorem sum_quotient_reciprocal_eq_one_point_thirty_five (x y : ℝ)
  (h1 : x + y = 45) (h2 : x * y = 500) : x / y + 1 / x + 1 / y = 1.35 := by
  -- Proof details would go here
  sorry

end sum_quotient_reciprocal_eq_one_point_thirty_five_l220_220607


namespace count_valid_ks_l220_220407

theorem count_valid_ks : 
  ∃ (ks : Finset ℕ), (∀ k ∈ ks, k > 0 ∧ k ≤ 50 ∧ 
    ∀ n : ℕ, n > 0 → 7 ∣ (2 * 3^(6 * n) + k * 2^(3 * n + 1) - 1)) ∧ ks.card = 7 :=
sorry

end count_valid_ks_l220_220407


namespace divide_estate_l220_220373

theorem divide_estate (total_estate : ℕ) (son_share : ℕ) (daughter_share : ℕ) (wife_share : ℕ) :
  total_estate = 210 →
  son_share = (4 / 7) * total_estate →
  daughter_share = (1 / 7) * total_estate →
  wife_share = (2 / 7) * total_estate →
  son_share + daughter_share + wife_share = total_estate :=
by
  intros
  sorry

end divide_estate_l220_220373


namespace polygon_with_interior_sum_1260_eq_nonagon_l220_220032

theorem polygon_with_interior_sum_1260_eq_nonagon :
  ∃ n : ℕ, (n-2) * 180 = 1260 ∧ n = 9 := by
  sorry

end polygon_with_interior_sum_1260_eq_nonagon_l220_220032


namespace sara_ate_16_apples_l220_220224

theorem sara_ate_16_apples (S : ℕ) (h1 : ∃ (A : ℕ), A = 4 * S ∧ S + A = 80) : S = 16 :=
by
  obtain ⟨A, h2, h3⟩ := h1
  sorry

end sara_ate_16_apples_l220_220224


namespace sum_first_five_terms_l220_220703

variable {a : ℕ → ℝ} -- Define the arithmetic sequence as a function

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

noncomputable def S_5 (a : ℕ → ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3 + a 4

theorem sum_first_five_terms (a : ℕ → ℝ) (h : is_arithmetic_sequence a) 
  (h2 : a 1 + a 3 = 6) : S_5 a = 15 :=
by
  -- skipping actual proof
  sorry

end sum_first_five_terms_l220_220703


namespace min_value_of_a_l220_220725

theorem min_value_of_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x^2 + 2*x*y ≤ a*(x^2 + y^2)) → (a ≥ (Real.sqrt 5 + 1) / 2) := 
sorry

end min_value_of_a_l220_220725


namespace fred_earned_63_dollars_l220_220904

-- Definitions for the conditions
def initial_money_fred : ℕ := 23
def initial_money_jason : ℕ := 46
def money_per_car : ℕ := 5
def money_per_lawn : ℕ := 10
def money_per_dog : ℕ := 3
def total_money_after_chores : ℕ := 86
def cars_washed : ℕ := 4
def lawns_mowed : ℕ := 3
def dogs_walked : ℕ := 7

-- The equivalent proof problem in Lean
theorem fred_earned_63_dollars :
  (initial_money_fred + (cars_washed * money_per_car) + 
      (lawns_mowed * money_per_lawn) + 
      (dogs_walked * money_per_dog) = total_money_after_chores) → 
  ((cars_washed * money_per_car) + 
      (lawns_mowed * money_per_lawn) + 
      (dogs_walked * money_per_dog) = 63) :=
by
  sorry

end fred_earned_63_dollars_l220_220904


namespace condition_is_sufficient_but_not_necessary_l220_220867

variable (P Q : Prop)

theorem condition_is_sufficient_but_not_necessary :
    (P → Q) ∧ ¬(Q → P) :=
sorry

end condition_is_sufficient_but_not_necessary_l220_220867


namespace smallest_five_digit_neg_int_congruent_to_one_mod_17_l220_220334

theorem smallest_five_digit_neg_int_congruent_to_one_mod_17 :
  ∃ (x : ℤ), x < -9999 ∧ x % 17 = 1 ∧ x = -10011 := by
  -- The proof would go here
  sorry

end smallest_five_digit_neg_int_congruent_to_one_mod_17_l220_220334


namespace ab_necessary_but_not_sufficient_l220_220288

theorem ab_necessary_but_not_sufficient (a b : ℝ) (i : ℂ) (hi : i^2 = -1) : 
  ab < 0 → ¬ (ab >= 0) ∧ (¬ (ab <= 0)) → (z = i * (a + b * i)) ∧ a > 0 ∧ -b > 0 := 
  sorry

end ab_necessary_but_not_sufficient_l220_220288


namespace race_order_count_l220_220880

-- Define the problem conditions
def participants : List String := ["Harry", "Ron", "Neville", "Hermione"]
def no_ties : Prop := True -- Since no ties are given directly, we denote this as always true for simplicity

-- Define the proof problem statement
theorem race_order_count (h_no_ties : no_ties) : participants.permutations.length = 24 := 
by
  -- Placeholder for proof
  sorry

end race_order_count_l220_220880


namespace isosceles_triangle_base_length_l220_220822

theorem isosceles_triangle_base_length :
  ∃ (x y : ℝ), 
    ((x + x / 2 = 15 ∧ y + x / 2 = 6) ∨ (x + x / 2 = 6 ∧ y + x / 2 = 15)) ∧ y = 1 :=
by
  sorry

end isosceles_triangle_base_length_l220_220822


namespace common_points_on_Euler_line_l220_220139

open EuclideanGeometry

variables {A B C A₁ B₁ C₁ A₂ B₂ C₂ A₃ B₃ C₃ X₁ X₂ : Point}

-- Let ABC be an acute scalene triangle
variables [Triangle ABC]
axiom acute.scalene (ABC : Triangle) : ScaleneTriangle ABC ∧ AcuteTriangle ABC

-- Let A₁, B₁, C₁ be the feet of the altitudes from A, B, C
axiom Altitudes.feet (A₁ B₁ C₁ : Point) (ABC : Triangle) : FeetsOfAltitudes A₁ B₁ C₁ ABC

-- A₂ is the intersection of the tangents to the circle ABC at B, C, and similarly for B₂, C₂
axiom TangentIntersection (A₂ B₂ C₂ : Point) (ABC : Triangle) : TangentIntersection A₂ B₂ C₂ ABC

-- A₂A₁ intersects the circle A₂B₂C₂ again at A₃ and similarly for B₃, C₃
axiom CircleIntersection (A₃ B₃ C₃ : Point) (A₂ B₂ C₂ A₁ B₁ C₁ : Point) : CircleIntersection A₃ B₃ C₃ A₂ B₂ C₂ A₁ B₁ C₁

-- Show that the circles AA₁A₃, BB₁B₃, and CC₁C₃ all have two common points, X₁ and X₂ which both lie on the Euler line of the triangle ABC
theorem common_points_on_Euler_line (ABC : Triangle) (A₁ B₁ C₁ A₂ B₂ C₂ A₃ B₃ C₃ X₁ X₂ : Point)
  (acute : ScaleneTriangle ABC ∧ AcuteTriangle ABC)
  (feet : FeetsOfAltitudes A₁ B₁ C₁ ABC)
  (tangent : TangentIntersection A₂ B₂ C₂ ABC)
  (circle_intersections : CircleIntersection A₃ B₃ C₃ A₂ B₂ C₂ A₁ B₁ C₁) :
  (Circle A A₁ A₃).intersection (Circle B B₁ B₃) = (Circle C C₁ C₃).intersection (Circle B B₁ B₃) ∧
  (X₁ ∈ EulerLine ABC) ∧ (X₂ ∈ EulerLine ABC) :=
sorry

end common_points_on_Euler_line_l220_220139


namespace distance_between_parallel_lines_l220_220324

theorem distance_between_parallel_lines 
  (d : ℝ) 
  (r : ℝ)
  (h1 : (42 * 21 + (d / 2) * 42 * (d / 2) = 42 * r^2))
  (h2 : (40 * 20 + (3 * d / 2) * 40 * (3 * d / 2) = 40 * r^2)) :
  d = 3 + 3 / 8 :=
  sorry

end distance_between_parallel_lines_l220_220324


namespace hyperbola_min_value_l220_220405

def hyperbola_condition : Prop :=
  ∀ (m : ℝ), ∀ (x y : ℝ), (4 * x + 3 * y + m = 0 → (x^2 / 9 - y^2 / 16 = 1) → false)

noncomputable def minimum_value : ℝ :=
  2 * Real.sqrt 37 - 6

theorem hyperbola_min_value :
  hyperbola_condition → minimum_value =  2 * Real.sqrt 37 - 6 :=
by
  intro h
  sorry

end hyperbola_min_value_l220_220405


namespace problem_solution_l220_220128

def grid_side : ℕ := 4
def square_size : ℝ := 2
def ellipse_major_axis : ℝ := 4
def ellipse_minor_axis : ℝ := 2
def circle_radius : ℝ := 1
def num_circles : ℕ := 3

noncomputable def grid_area : ℝ :=
  (grid_side * grid_side) * (square_size * square_size)

noncomputable def circle_area : ℝ :=
  num_circles * (Real.pi * (circle_radius ^ 2))

noncomputable def ellipse_area : ℝ :=
  Real.pi * (ellipse_major_axis / 2) * (ellipse_minor_axis / 2)

noncomputable def visible_shaded_area (A B : ℝ) : Prop :=
  grid_area = A - B * Real.pi

theorem problem_solution : ∃ A B, visible_shaded_area A B ∧ (A + B = 69) :=
by
  sorry

end problem_solution_l220_220128


namespace parallel_edges_octahedron_l220_220122

-- Definition of a regular octahedron's properties
structure regular_octahedron : Type :=
  (edges : ℕ) -- Number of edges in the octahedron

-- Constant to represent the regular octahedron with 12 edges.
def octahedron : regular_octahedron := { edges := 12 }

-- Definition to count unique pairs of parallel edges
def count_parallel_edge_pairs (o : regular_octahedron) : ℕ :=
  if o.edges = 12 then 12 else 0

-- Theorem to assert the number of pairs of parallel edges in a regular octahedron is 12
theorem parallel_edges_octahedron : count_parallel_edge_pairs octahedron = 12 :=
by
  -- Proof will be inserted here
  sorry

end parallel_edges_octahedron_l220_220122


namespace race_permutations_l220_220879

-- Define the problem conditions: four participants
def participants : Nat := 4

-- Define the factorial function for permutations
def factorial : Nat → Nat
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

-- Theorem: The number of different possible orders in which Harry, Ron, Neville, and Hermione can finish is 24
theorem race_permutations : factorial participants = 24 :=
by
  simp [participants, factorial]
  sorry

end race_permutations_l220_220879


namespace no_four_digit_perfect_square_palindromes_l220_220659

def is_palindrome (n : ℕ) : Prop :=
  (n.toString = n.toString.reverse)

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_four_digit_perfect_square_palindromes :
  ¬ ∃ n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
by introv H, cases H with n Hn sorry

end no_four_digit_perfect_square_palindromes_l220_220659


namespace fraction_milk_in_mug1_is_one_fourth_l220_220901

-- Condition definitions
def initial_tea_mug1 := 6 -- ounces
def initial_milk_mug2 := 6 -- ounces
def tea_transferred_mug1_to_mug2 := initial_tea_mug1 / 3
def tea_remaining_mug1 := initial_tea_mug1 - tea_transferred_mug1_to_mug2
def total_liquid_mug2 := initial_milk_mug2 + tea_transferred_mug1_to_mug2
def portion_transferred_back := total_liquid_mug2 / 4
def tea_ratio_mug2 := tea_transferred_mug1_to_mug2 / total_liquid_mug2
def milk_ratio_mug2 := initial_milk_mug2 / total_liquid_mug2
def tea_transferred_back := portion_transferred_back * tea_ratio_mug2
def milk_transferred_back := portion_transferred_back * milk_ratio_mug2
def final_tea_mug1 := tea_remaining_mug1 + tea_transferred_back
def final_milk_mug1 := milk_transferred_back
def final_total_liquid_mug1 := final_tea_mug1 + final_milk_mug1

-- Lean statement of the problem
theorem fraction_milk_in_mug1_is_one_fourth :
  final_milk_mug1 / final_total_liquid_mug1 = 1 / 4 :=
by
  sorry

end fraction_milk_in_mug1_is_one_fourth_l220_220901


namespace part_I_monotonicity_part_II_value_a_l220_220869

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / (x - 1)

def is_monotonic_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x < f y

def is_monotonic_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f y < f x

theorem part_I_monotonicity :
  (is_monotonic_increasing f {x | 2 < x}) ∧
  ((is_monotonic_decreasing f {x | x < 1}) ∧ (is_monotonic_decreasing f {x | 1 < x ∧ x < 2})) :=
by
  sorry

theorem part_II_value_a (a : ℝ) :
  (∀ x : ℝ, 2 ≤ x → (Real.exp x * (x - 2)) / ((x - 1)^2) ≥ a * (Real.exp x / (x - 1))) → a ∈ Set.Iic 0 :=
by
  sorry

end part_I_monotonicity_part_II_value_a_l220_220869


namespace expand_expression_l220_220522

theorem expand_expression (x : ℝ) : (x - 3) * (x + 6) = x^2 + 3 * x - 18 :=
by
  sorry

end expand_expression_l220_220522


namespace time_to_walk_l220_220038

variable (v l r w : ℝ)
variable (h1 : l = 15 * (v + r))
variable (h2 : l = 30 * (v + w))
variable (h3 : l = 20 * r)

theorem time_to_walk (h1 : l = 15 * (v + r)) (h2 : l = 30 * (v + w)) (h3 : l = 20 * r) : l / w = 60 := 
by sorry

end time_to_walk_l220_220038


namespace travis_return_probability_l220_220181

open Function

namespace CubeHopping

def Vertex := (Fin 2 × Fin 2 × Fin 2)

def adjacent (v1 v2 : Vertex) : Prop := 
  (v1.1 = v2.1 ∧ v1.2 = v2.2 ∧ v1.3 ≠ v2.3) ∨ 
  (v1.1 = v2.1 ∧ v1.2 ≠ v2.2 ∧ v1.3 = v2.3) ∨ 
  (v1.1 ≠ v2.1 ∧ v1.2 = v2.2 ∧ v1.3 = v2.3)

def probability_of_returning (start : Vertex) (moves : ℕ) : ℚ :=
-- Function to encapsulate the details of the probability calculation (to be filled in)
sorry

theorem travis_return_probability :
  probability_of_returning (0, 0, 0) 4 = 7 / 27 :=
sorry

end CubeHopping

end travis_return_probability_l220_220181


namespace probability_C_D_l220_220793

variable (P : String → ℚ)

axiom h₁ : P "A" = 1/4
axiom h₂ : P "B" = 1/3
axiom h₃ : P "A" + P "B" + P "C" + P "D" = 1

theorem probability_C_D : P "C" + P "D" = 5/12 := by
  sorry

end probability_C_D_l220_220793


namespace Nicki_runs_30_miles_per_week_in_second_half_l220_220452

/-
  Nicki ran 20 miles per week for the first half of the year.
  There are 26 weeks in each half of the year.
  She ran a total of 1300 miles for the year.
  Prove that Nicki ran 30 miles per week in the second half of the year.
-/

theorem Nicki_runs_30_miles_per_week_in_second_half (weekly_first_half : ℕ) (weeks_per_half : ℕ) (total_miles : ℕ) :
  weekly_first_half = 20 → weeks_per_half = 26 → total_miles = 1300 → 
  (total_miles - (weekly_first_half * weeks_per_half)) / weeks_per_half = 30 :=
by
  intros h1 h2 h3
  sorry

end Nicki_runs_30_miles_per_week_in_second_half_l220_220452


namespace circle_equation_unique_l220_220025

theorem circle_equation_unique {F D E : ℝ} : 
  (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧ 
  (∀ (x y : ℝ), (x = 1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧ 
  (∀ (x y : ℝ), (x = 4 ∧ y = 2) → x^2 + y^2 + D * x + E * y + F = 0) → 
  (x^2 + y^2 - 8 * x + 6 * y = 0) :=
by 
  sorry

end circle_equation_unique_l220_220025


namespace visitors_that_day_l220_220375

theorem visitors_that_day (total_visitors : ℕ) (previous_day_visitors : ℕ) 
  (h_total : total_visitors = 406) (h_previous : previous_day_visitors = 274) : 
  total_visitors - previous_day_visitors = 132 :=
by
  sorry

end visitors_that_day_l220_220375


namespace smaller_number_is_5_l220_220475

theorem smaller_number_is_5 (x y : ℤ) (h1 : x + y = 18) (h2 : x - y = 8) : y = 5 := by
  sorry

end smaller_number_is_5_l220_220475


namespace bowling_ball_weight_l220_220520

theorem bowling_ball_weight (b k : ℕ) (h1 : 8 * b = 4 * k) (h2 : 3 * k = 84) : b = 14 := by
  sorry

end bowling_ball_weight_l220_220520


namespace part1_l220_220062

theorem part1 (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) : 2 * x^2 + y^2 > x^2 + x * y := 
sorry

end part1_l220_220062


namespace outer_circle_radius_l220_220934

theorem outer_circle_radius (r R : ℝ) (hr : r = 4)
  (radius_increase : ∀ R, R' = 1.5 * R)
  (radius_decrease : ∀ r, r' = 0.75 * r)
  (area_increase : ∀ (A1 A2 : ℝ), A2 = 3.6 * A1)
  (initial_area : ∀ A1, A1 = π * R^2 - π * r^2)
  (new_area : ∀ A2 R' r', A2 = π * R'^2 - π * r'^2) :
  R = 6 := sorry

end outer_circle_radius_l220_220934


namespace total_number_of_lives_l220_220610

theorem total_number_of_lives (initial_players : ℕ) (additional_players : ℕ) (lives_per_player : ℕ) 
                              (h1 : initial_players = 7) (h2 : additional_players = 2) (h3 : lives_per_player = 7) : 
                              initial_players + additional_players * lives_per_player = 63 :=
by
  sorry

end total_number_of_lives_l220_220610


namespace alex_money_left_l220_220507

noncomputable def weekly_income := 500
def income_tax_rate := 0.10
def water_bill := 55
def tithe_rate := 0.10

theorem alex_money_left : (weekly_income - ((weekly_income * income_tax_rate) + (weekly_income * tithe_rate) + water_bill)) = 345 := 
by
  sorry

end alex_money_left_l220_220507


namespace price_of_horse_and_cow_l220_220437

theorem price_of_horse_and_cow (x y : ℝ) (h1 : 4 * x + 6 * y = 48) (h2 : 3 * x + 5 * y = 38) :
  (4 * x + 6 * y = 48) ∧ (3 * x + 5 * y = 38) := 
by
  exact ⟨h1, h2⟩

end price_of_horse_and_cow_l220_220437


namespace range_of_a_l220_220846

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x : ℝ, (2 * a + 1) * x + a - 2 > (2 * a + 1) * 0 + a - 2)
  (h2 : a - 2 < 0) : -1 / 2 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l220_220846


namespace solutions_eq_l220_220513

theorem solutions_eq :
  { (a, b, c) : ℕ × ℕ × ℕ | a * b + b * c + c * a = 2 * (a + b + c) } =
  { (2, 2, 2),
    (1, 2, 4), (1, 4, 2), 
    (2, 1, 4), (2, 4, 1),
    (4, 1, 2), (4, 2, 1) } :=
by sorry

end solutions_eq_l220_220513


namespace inequality_of_areas_l220_220913

variable {A B C D K L M N : Type}
variable [MetricSpace K] [MetricSpace L] [MetricSpace M] [MetricSpace N] [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variable {AKN BKL CLM DMN ABCD : ℝ}

-- Define the areas of the triangles and the quadrilateral
def S1 := area A K N
def S2 := area B K L
def S3 := area C L M
def S4 := area D M N
def S_ABCD := area A B C D

-- Ensure these points are well-defined, eg. all are part of respective quadrilateral sides
variable (K_on_AB : K ∈ line_segment A B)
variable (L_on_BC : L ∈ line_segment B C)
variable (M_on_CD : M ∈ line_segment C D)
variable (N_on_DA : N ∈ line_segment D A)

theorem inequality_of_areas
  (AKN_nonneg : 0 ≤ S1) 
  (BKL_nonneg : 0 ≤ S2) 
  (CLM_nonneg : 0 ≤ S3) 
  (DMN_nonneg : 0 ≤ S4)
  (ABCD_nonneg : 0 ≤ S_ABCD) :
  (∛S1 + ∛S2 + ∛S3 + ∛S4) ≤ 2 * ∛S_ABCD :=
by
  sorry

end inequality_of_areas_l220_220913


namespace area_ratio_of_concentric_circles_l220_220049

noncomputable theory

-- Define the given conditions
def C1 (r1 : ℝ) : ℝ := 2 * Real.pi * r1
def C2 (r2 : ℝ) : ℝ := 2 * Real.pi * r2
def arc_length (angle : ℝ) (circumference : ℝ) : ℝ := (angle / 360) * circumference

-- Lean statement for the math proof problem
theorem area_ratio_of_concentric_circles 
  (r1 r2 : ℝ) (h₁ : arc_length 60 (C1 r1) = arc_length 48 (C2 r2)) : 
  (Real.pi * r1^2) / (Real.pi * r2^2) = 16 / 25 :=
by
  sorry  -- Proof omitted

end area_ratio_of_concentric_circles_l220_220049


namespace difference_place_values_l220_220051

def place_value (digit : Char) (position : String) : Real :=
  match digit, position with
  | '1', "hundreds" => 100
  | '1', "tenths" => 0.1
  | _, _ => 0 -- for any other cases (not required in this problem)

theorem difference_place_values :
  (place_value '1' "hundreds" - place_value '1' "tenths" = 99.9) :=
by
  sorry

end difference_place_values_l220_220051


namespace x_intercept_of_line_l220_220897

open Real

theorem x_intercept_of_line : 
  ∃ x : ℝ, 
  (∃ m : ℝ, m = (3 - -5) / (10 - -6) ∧ (∀ y : ℝ, y = m * (x - 10) + 3)) ∧ 
  (∀ y : ℝ, y = 0 → x = 4) :=
sorry

end x_intercept_of_line_l220_220897


namespace convex_polygon_nonagon_l220_220820

theorem convex_polygon_nonagon (n : ℕ) (E I : ℝ) 
  (h1 : I = 3 * E + 180)
  (h2 : I + E = 180)
  (h3 : n * E = 360) : n = 9 :=
begin
  sorry
end

end convex_polygon_nonagon_l220_220820


namespace twelve_percent_greater_than_80_l220_220967

theorem twelve_percent_greater_than_80 (x : ℝ) (h : x = 80 + 0.12 * 80) : x = 89.6 :=
by
  sorry

end twelve_percent_greater_than_80_l220_220967


namespace value_of_a_m_minus_2n_l220_220886

variable (a : ℝ) (m n : ℝ)

theorem value_of_a_m_minus_2n (h1 : a^m = 8) (h2 : a^n = 4) : a^(m - 2 * n) = 1 / 2 :=
by
  sorry

end value_of_a_m_minus_2n_l220_220886


namespace trains_crossing_time_l220_220183

noncomputable def TrainA_length := 200  -- meters
noncomputable def TrainA_time := 15  -- seconds
noncomputable def TrainB_length := 300  -- meters
noncomputable def TrainB_time := 25  -- seconds

noncomputable def Speed (length : ℕ) (time : ℕ) := (length : ℝ) / (time : ℝ)

noncomputable def TrainA_speed := Speed TrainA_length TrainA_time
noncomputable def TrainB_speed := Speed TrainB_length TrainB_time

noncomputable def relative_speed := TrainA_speed + TrainB_speed
noncomputable def total_distance := (TrainA_length : ℝ) + (TrainB_length : ℝ)

noncomputable def crossing_time := total_distance / relative_speed

theorem trains_crossing_time :
  (crossing_time : ℝ) = 500 / 25.33 :=
sorry

end trains_crossing_time_l220_220183


namespace ratio_of_areas_of_concentric_circles_l220_220047

theorem ratio_of_areas_of_concentric_circles
  (C1 C2 : ℝ) -- circumferences of the smaller and larger circle
  (h : (1 / 6) * C1 = (2 / 15) * C2) -- condition given: 60-degree arc on the smaller circle equals 48-degree arc on the larger circle
  : (C1 / C2)^2 = (16 / 25) := by
  sorry

end ratio_of_areas_of_concentric_circles_l220_220047


namespace shorter_piece_length_l220_220207

theorem shorter_piece_length :
  ∃ (x : ℝ), x + 2 * x = 69 ∧ x = 23 :=
by
  sorry

end shorter_piece_length_l220_220207


namespace find_sum_of_coefficients_l220_220254

theorem find_sum_of_coefficients
  (a b c d : ℤ)
  (h1 : (x^2 + a * x + b) * (x^2 + c * x + d) = x^4 + x^3 - 2 * x^2 + 17 * x - 5) :
  a + b + c + d = 5 :=
by
  sorry

end find_sum_of_coefficients_l220_220254


namespace book_total_pages_l220_220192

-- Define the conditions given in the problem
def pages_per_night : ℕ := 12
def nights_to_finish : ℕ := 10

-- State that the total number of pages in the book is 120 given the conditions
theorem book_total_pages : (pages_per_night * nights_to_finish) = 120 :=
by sorry

end book_total_pages_l220_220192


namespace snack_cost_inequality_l220_220501

variables (S : ℝ)

def cost_water : ℝ := 0.50
def cost_fruit : ℝ := 0.25
def bundle_price : ℝ := 4.60
def special_price : ℝ := 2.00

theorem snack_cost_inequality (h : bundle_price = 4.60 ∧ special_price = 2.00 ∧
  cost_water = 0.50 ∧ cost_fruit = 0.25) : S < 15.40 / 16 := sorry

end snack_cost_inequality_l220_220501


namespace total_cars_parked_l220_220738

theorem total_cars_parked
  (area_a : ℕ) (util_a : ℕ)
  (area_b : ℕ) (util_b : ℕ)
  (area_c : ℕ) (util_c : ℕ)
  (area_d : ℕ) (util_d : ℕ)
  (space_per_car : ℕ) 
  (ha: area_a = 400 * 500)
  (hu_a: util_a = 80)
  (hb: area_b = 600 * 700)
  (hu_b: util_b = 75)
  (hc: area_c = 500 * 800)
  (hu_c: util_c = 65)
  (hd: area_d = 300 * 900)
  (hu_d: util_d = 70)
  (h_sp: space_per_car = 10) :
  (util_a * area_a / 100 / space_per_car + 
   util_b * area_b / 100 / space_per_car + 
   util_c * area_c / 100 / space_per_car + 
   util_d * area_d / 100 / space_per_car) = 92400 :=
by sorry

end total_cars_parked_l220_220738


namespace number_of_three_digit_numbers_is_48_l220_220612

-- Define the problem: the cards and their constraints
def card1 := (1, 2)
def card2 := (3, 4)
def card3 := (5, 6)

-- The condition given is that 6 cannot be used as 9

-- Define the function to compute the number of different three-digit numbers
def number_of_three_digit_numbers : Nat := 6 * 4 * 2

/- Prove that the number of different three-digit numbers that can be formed is 48 -/
theorem number_of_three_digit_numbers_is_48 : number_of_three_digit_numbers = 48 :=
by
  -- We skip the proof here
  sorry

end number_of_three_digit_numbers_is_48_l220_220612


namespace eq_x_minus_y_l220_220786

theorem eq_x_minus_y (x y : ℝ) : (x - y) * (x - y) = x^2 - 2 * x * y + y^2 :=
by
  sorry

end eq_x_minus_y_l220_220786


namespace chicken_problem_l220_220594

theorem chicken_problem (x y z : ℕ) :
  x + y + z = 100 ∧ 5 * x + 3 * y + z / 3 = 100 → 
  (x = 0 ∧ y = 25 ∧ z = 75) ∨ 
  (x = 12 ∧ y = 4 ∧ z = 84) ∨ 
  (x = 8 ∧ y = 11 ∧ z = 81) ∨ 
  (x = 4 ∧ y = 18 ∧ z = 78) := 
sorry

end chicken_problem_l220_220594


namespace Marta_max_piles_l220_220753

theorem Marta_max_piles (a b c : ℕ) (ha : a = 42) (hb : b = 60) (hc : c = 90) : 
  Nat.gcd (Nat.gcd a b) c = 6 := by
  rw [ha, hb, hc]
  have h : Nat.gcd (Nat.gcd 42 60) 90 = Nat.gcd 6 90 := by sorry
  exact h    

end Marta_max_piles_l220_220753


namespace total_tagged_numbers_l220_220381

theorem total_tagged_numbers {W X Y Z : ℕ} 
  (hW : W = 200)
  (hX : X = W / 2)
  (hY : Y = W + X)
  (hZ : Z = 400) :
  W + X + Y + Z = 1000 := by
  sorry

end total_tagged_numbers_l220_220381


namespace one_fourth_of_six_point_eight_eq_seventeen_tenths_l220_220097

theorem one_fourth_of_six_point_eight_eq_seventeen_tenths :  (6.8 / 4) = (17 / 10) :=
by
  -- Skipping proof
  sorry

end one_fourth_of_six_point_eight_eq_seventeen_tenths_l220_220097


namespace geometric_sequence_third_term_l220_220408

theorem geometric_sequence_third_term (a : ℕ → ℕ) (x : ℕ) (h1 : a 1 = 4) (h2 : a 2 = 6) (h3 : a 3 = x) (h_geom : ∀ n, a (n + 1) = a n * r) :
  x = 9 := 
sorry

end geometric_sequence_third_term_l220_220408


namespace ratio_sprite_to_coke_l220_220277

theorem ratio_sprite_to_coke (total_drink : ℕ) (coke_ounces : ℕ) (mountain_dew_parts : ℕ)
  (parts_coke : ℕ) (parts_mountain_dew : ℕ) (total_parts : ℕ) :
  total_drink = 18 →
  coke_ounces = 6 →
  parts_coke = 2 →
  parts_mountain_dew = 3 →
  total_parts = parts_coke + parts_mountain_dew + ((total_drink - coke_ounces - (parts_mountain_dew * (coke_ounces / parts_coke))) / (coke_ounces / parts_coke)) →
  (total_drink - coke_ounces - (parts_mountain_dew * (coke_ounces / parts_coke))) / coke_ounces = 1 / 2 :=
by sorry

end ratio_sprite_to_coke_l220_220277


namespace four_digit_palindromic_perfect_square_l220_220655

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

theorem four_digit_palindromic_perfect_square :
  {n : ℕ | is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n}.card = 1 :=
sorry

end four_digit_palindromic_perfect_square_l220_220655


namespace complex_number_in_first_quadrant_l220_220031

def is_in_first_quadrant (z : ℂ) : Prop :=
  0 < z.re ∧ 0 < z.im

theorem complex_number_in_first_quadrant (z : ℂ) (h : 0 < z.re ∧ 0 < z.im) : is_in_first_quadrant z :=
by sorry

end complex_number_in_first_quadrant_l220_220031


namespace sqrt_expression_l220_220674

theorem sqrt_expression : 2 * Real.sqrt 3 - (3 * Real.sqrt 2 + Real.sqrt 3) = Real.sqrt 3 - 3 * Real.sqrt 2 :=
by
  sorry

end sqrt_expression_l220_220674


namespace integer_points_between_A_B_l220_220216

/-- 
Prove that the number of integer coordinate points strictly between 
A(2, 3) and B(50, 80) on the line passing through A and B is c.
-/
theorem integer_points_between_A_B 
  (A B : ℤ × ℤ) (hA : A = (2, 3)) (hB : B = (50, 80)) 
  (c : ℕ) :
  ∃ (n : ℕ), n = c ∧ ∀ (x y : ℤ), (A.1 < x ∧ x < B.1) → (A.2 < y ∧ y < B.2) → 
              (y = ((A.2 - B.2) / (A.1 - B.1) * x + 3 - (A.2 - B.2) / (A.1 - B.1) * 2)) :=
by {
  sorry
}

end integer_points_between_A_B_l220_220216


namespace problem_solution_l220_220146

open Set

variable {U : Set ℕ} (M : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hC : U \ M = {1, 3})

theorem problem_solution : 2 ∈ M :=
by
  sorry

end problem_solution_l220_220146


namespace fraction_numerator_l220_220165

theorem fraction_numerator (x : ℚ) :
  (∃ n : ℚ, 4 * n - 4 = x ∧ x / (4 * n - 4) = 3 / 7) → x = 12 / 5 :=
by
  sorry

end fraction_numerator_l220_220165


namespace range_of_a_l220_220530

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^3 - a * x^2 - 4 * a * x + 4 * a^2 - 1 = 0 ∧ ∀ y : ℝ, 
  (y ≠ x → y^3 - a * y^2 - 4 * a * y + 4 * a^2 - 1 ≠ 0)) ↔ a < 3 / 4 := 
sorry

end range_of_a_l220_220530


namespace tunnel_length_l220_220789

def train_length : ℝ := 1.5
def exit_time_minutes : ℝ := 4
def speed_mph : ℝ := 45

theorem tunnel_length (d_train : ℝ := train_length)
                      (t_exit : ℝ := exit_time_minutes)
                      (v_mph : ℝ := speed_mph) :
  d_train + ((v_mph / 60) * t_exit - d_train) = 1.5 :=
by
  sorry

end tunnel_length_l220_220789


namespace incorrect_statement_among_ABCD_l220_220819

theorem incorrect_statement_among_ABCD :
  ¬ (-3 = Real.sqrt ((-3)^2)) :=
by
  sorry

end incorrect_statement_among_ABCD_l220_220819


namespace expected_value_of_draws_before_stopping_l220_220355

noncomputable def totalBalls := 10
noncomputable def redBalls := 2
noncomputable def whiteBalls := 8

noncomputable def prob_one_draw_white : ℚ := whiteBalls / totalBalls
noncomputable def prob_two_draws_white : ℚ := (redBalls / totalBalls) * (whiteBalls / (totalBalls - 1))
noncomputable def prob_three_draws_white : ℚ := (redBalls / (totalBalls - redBalls + 1)) * ((redBalls - 1) / (totalBalls - 1)) * (whiteBalls / (totalBalls - 2))

noncomputable def expected_draws_before_white : ℚ :=
  1 * prob_one_draw_white + 2 * prob_two_draws_white + 3 * prob_three_draws_white

theorem expected_value_of_draws_before_stopping : expected_draws_before_white = 11 / 9 := by
  sorry

end expected_value_of_draws_before_stopping_l220_220355


namespace find_number_l220_220890

theorem find_number:
  ∃ x : ℝ, (3/4 * x + 9 = 1/5 * (x - 8 * x^(1/3))) ∧ x = -27 :=
by
  sorry

end find_number_l220_220890


namespace part1_part2_i_part2_ii_l220_220432

def equation1 (x : ℝ) : Prop := 3 * x - 2 = 0
def equation2 (x : ℝ) : Prop := 2 * x - 3 = 0
def equation3 (x : ℝ) : Prop := x - (3 * x + 1) = -7

def inequality1 (x : ℝ) : Prop := -x + 2 > x - 5
def inequality2 (x : ℝ) : Prop := 3 * x - 1 > -x + 2

def sys_ineq (x m : ℝ) : Prop := x + m < 2 * x ∧ x - 2 < m

def equation4 (x : ℝ) : Prop := (2 * x - 1) / 3 = -3

theorem part1 : 
  ∀ (x : ℝ), inequality1 x → inequality2 x → equation2 x → equation3 x :=
by sorry

theorem part2_i :
  ∀ (m : ℝ), (∃ (x : ℝ), equation4 x ∧ sys_ineq x m) → -6 < m ∧ m < -4 :=
by sorry

theorem part2_ii :
  ∀ (m : ℝ), ¬ (sys_ineq 1 m ∧ sys_ineq 2 m) → m ≥ 2 ∨ m ≤ -1 :=
by sorry

end part1_part2_i_part2_ii_l220_220432


namespace haley_total_trees_l220_220549

-- Define the number of dead trees and remaining trees
def dead_trees : ℕ := 5
def remaining_trees : ℕ := 12

-- Prove the total number of trees Haley originally grew
theorem haley_total_trees :
  (dead_trees + remaining_trees) = 17 :=
by
  -- Providing the proof using sorry as placeholder
  sorry

end haley_total_trees_l220_220549


namespace zmod_field_l220_220289

theorem zmod_field (p : ℕ) [Fact (Nat.Prime p)] : Field (ZMod p) :=
sorry

end zmod_field_l220_220289


namespace probability_C_and_D_l220_220791

theorem probability_C_and_D (P_A P_B : ℚ) (H1 : P_A = 1/4) (H2 : P_B = 1/3) :
  P_C + P_D = 5/12 :=
by
  sorry

end probability_C_and_D_l220_220791


namespace simplify_expression_l220_220305

theorem simplify_expression (θ : ℝ) : 
  ((1 + Real.sin θ ^ 2) ^ 2 - Real.cos θ ^ 4) * ((1 + Real.cos θ ^ 2) ^ 2 - Real.sin θ ^ 4) = 4 * Real.sin (2 * θ) ^ 2 :=
by 
  sorry

end simplify_expression_l220_220305


namespace range_of_k_tan_alpha_l220_220416

noncomputable def f (x k : Real) : Real := Real.sin x + k

theorem range_of_k (k : Real) : 
  (∃ x : Real, f x k = 1) ↔ (0 ≤ k ∧ k ≤ 2) :=
sorry

theorem tan_alpha (α k : Real) (h : α ∈ Set.Ioo (0 : Real) Real.pi) (hf : f α k = 1 / 3 + k) : 
  Real.tan α = Real.sqrt 2 / 4 :=
sorry

end range_of_k_tan_alpha_l220_220416


namespace mango_rate_is_50_l220_220550

theorem mango_rate_is_50 (quantity_grapes kg_grapes_perkg quantity_mangoes total_paid cost_grapes cost_mangoes rate_mangoes : ℕ) 
  (h1 : quantity_grapes = 8) 
  (h2 : kg_grapes_perkg = 70) 
  (h3 : quantity_mangoes = 9) 
  (h4 : total_paid = 1010)
  (h5 : cost_grapes = quantity_grapes * kg_grapes_perkg)
  (h6 : cost_mangoes = total_paid - cost_grapes)
  (h7 : rate_mangoes = cost_mangoes / quantity_mangoes) : 
  rate_mangoes = 50 :=
by sorry

end mango_rate_is_50_l220_220550


namespace charitable_woman_l220_220497

theorem charitable_woman (initial_pennies : ℕ) 
  (farmer_share : ℕ) (beggar_share : ℕ) (boy_share : ℕ) (left_pennies : ℕ) 
  (h1 : initial_pennies = 42)
  (h2 : farmer_share = (initial_pennies / 2 + 1))
  (h3 : beggar_share = ((initial_pennies - farmer_share) / 2 + 2))
  (h4 : boy_share = ((initial_pennies - farmer_share - beggar_share) / 2 + 3))
  (h5 : left_pennies = initial_pennies - farmer_share - beggar_share - boy_share) : 
  left_pennies = 1 :=
by
  sorry

end charitable_woman_l220_220497


namespace teddy_bears_count_l220_220576

theorem teddy_bears_count (toys_count : ℕ) (toy_cost : ℕ) (total_money : ℕ) (teddy_bear_cost : ℕ) : 
  toys_count = 28 → 
  toy_cost = 10 → 
  total_money = 580 → 
  teddy_bear_cost = 15 →
  ((total_money - toys_count * toy_cost) / teddy_bear_cost) = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end teddy_bears_count_l220_220576


namespace sequence_term_l220_220730

theorem sequence_term (a : ℕ → ℕ) 
  (h1 : a 1 = 2009) 
  (h2 : a 2 = 2011) 
  (h3 : ∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = n + 1) 
  : a 1000 = 2342 := 
by 
  sorry

end sequence_term_l220_220730


namespace max_value_trig_formula_l220_220091

theorem max_value_trig_formula (x : ℝ) : ∃ (M : ℝ), M = 5 ∧ ∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ M := 
sorry

end max_value_trig_formula_l220_220091


namespace find_digit_to_make_divisible_by_seven_l220_220052

/-- 
  Given a number formed by concatenating 2023 digits of 6 with 2023 digits of 5.
  In a three-digit number 6*5, find the digit * to make this number divisible by 7.
  i.e., We must find the digit x such that the number 600 + 10x + 5 is divisible by 7.
-/
theorem find_digit_to_make_divisible_by_seven :
  ∃ x : ℕ, x < 10 ∧ (600 + 10 * x + 5) % 7 = 0 :=
sorry

end find_digit_to_make_divisible_by_seven_l220_220052


namespace point_not_in_first_quadrant_l220_220705

theorem point_not_in_first_quadrant (m n : ℝ) (h : m * n ≤ 0) : ¬ (m > 0 ∧ n > 0) :=
sorry

end point_not_in_first_quadrant_l220_220705


namespace italian_dressing_mixture_l220_220071

/-- A chef is using a mixture of two brands of Italian dressing. 
  The first brand contains 8% vinegar, and the second brand contains 13% vinegar.
  The chef wants to make 320 milliliters of a dressing that is 11% vinegar.
  This statement proves the amounts required for each brand of dressing. -/

theorem italian_dressing_mixture
  (x y : ℝ)
  (hx : x + y = 320)
  (hv : 0.08 * x + 0.13 * y = 0.11 * 320) :
  x = 128 ∧ y = 192 :=
sorry

end italian_dressing_mixture_l220_220071


namespace find_shortest_height_l220_220749

variable (T S P Q : ℝ)

theorem find_shortest_height (h1 : T = 77.75) (h2 : T = S + 9.5) (h3 : P = S + 5) (h4 : Q = P - 3) : S = 68.25 :=
  sorry

end find_shortest_height_l220_220749


namespace intersection_when_a_eq_1_range_for_A_inter_B_eq_A_l220_220287

noncomputable def f (x : ℝ) : ℝ := Real.log (3 - abs (x - 1))

def setA : Set ℝ := { x | 3 - abs (x - 1) > 0 }

def setB (a : ℝ) : Set ℝ := { x | x^2 - (a + 5) * x + 5 * a < 0 }

theorem intersection_when_a_eq_1 : (setA ∩ setB 1) = { x | 1 < x ∧ x < 4 } :=
by
  sorry

theorem range_for_A_inter_B_eq_A : { a | (setA ∩ setB a) = setA } = { a | a ≤ -2 } :=
by
  sorry

end intersection_when_a_eq_1_range_for_A_inter_B_eq_A_l220_220287


namespace problem_l220_220338

theorem problem (a b : ℤ) (ha : a = 4) (hb : b = -3) : -2 * a - b^2 + 2 * a * b = -41 := 
by
  -- Provide proof here
  sorry

end problem_l220_220338


namespace estimate_undetected_typos_l220_220328

variables (a b c : ℕ)
-- a, b, c ≥ 0 are non-negative integers representing discovered errors by proofreader A, B, and common errors respectively.

theorem estimate_undetected_typos (h : c ≤ a ∧ c ≤ b) :
  ∃ n : ℕ, n = a * b / c - a - b + c :=
sorry

end estimate_undetected_typos_l220_220328


namespace team_overall_progress_is_89_l220_220912

def yard_changes : List Int := [-5, 9, -12, 17, -15, 24, -7]

def overall_progress (changes : List Int) : Int :=
  changes.sum

theorem team_overall_progress_is_89 :
  overall_progress yard_changes = 89 :=
by
  sorry

end team_overall_progress_is_89_l220_220912


namespace shaded_region_area_l220_220273

def area_of_shaded_region (grid_height grid_width triangle_base triangle_height : ℝ) : ℝ :=
  let total_area := grid_height * grid_width
  let triangle_area := 0.5 * triangle_base * triangle_height
  total_area - triangle_area

theorem shaded_region_area :
  area_of_shaded_region 3 15 5 3 = 37.5 :=
by 
  sorry

end shaded_region_area_l220_220273


namespace max_value_S_n_l220_220263

theorem max_value_S_n 
  (a : ℕ → ℕ)
  (a1 : a 1 = 2)
  (S : ℕ → ℕ)
  (h : ∀ n, 6 * S n = 3 * a (n + 1) + 4 ^ n - 1) :
  ∃ n, S n = 10 := 
sorry

end max_value_S_n_l220_220263


namespace unique_solution_implies_relation_l220_220348

theorem unique_solution_implies_relation (a b : ℝ)
    (h : ∃! (x y : ℝ), y = x^2 + a * x + b ∧ x = y^2 + a * y + b) : 
    a^2 = 2 * (a + 2 * b) - 1 :=
by
  sorry

end unique_solution_implies_relation_l220_220348


namespace min_c_value_l220_220760

def y_eq_abs_sum (x a b c : ℝ) : ℝ := |x - a| + |x - b| + |x - c|
def y_eq_line (x : ℝ) : ℝ := -2 * x + 2023

theorem min_c_value (a b c : ℕ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (order : a ≤ b ∧ b < c)
  (unique_sol : ∃! x : ℝ, y_eq_abs_sum x a b c = y_eq_line x) :
  c = 2022 := sorry

end min_c_value_l220_220760


namespace min_value_of_expression_l220_220269

theorem min_value_of_expression {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : (a + b) * (a + c) = 4) : 2 * a + b + c ≥ 4 :=
sorry

end min_value_of_expression_l220_220269


namespace number_of_partitions_of_7_into_4_parts_l220_220268

theorem number_of_partitions_of_7_into_4_parts : 
  (finset.attach (finset.powerset (finset.range (7+1)))).filter (λ s, s.sum = 7 ∧ s.card ≤ 4)).card = 11 := 
by sorry

end number_of_partitions_of_7_into_4_parts_l220_220268


namespace mart_income_percentage_l220_220627

theorem mart_income_percentage 
  (J T M : ℝ)
  (h1 : M = 1.60 * T)
  (h2 : T = 0.60 * J) :
  M = 0.96 * J :=
sorry

end mart_income_percentage_l220_220627


namespace polynomial_relatively_prime_condition_l220_220682

open Polynomial

noncomputable def relatively_prime (a b : ℤ) : Prop := Int.gcd a b = 1

theorem polynomial_relatively_prime_condition (P : Polynomial ℤ) :
  (∀ a b : ℤ, relatively_prime a b → relatively_prime (P.eval a) (P.eval b)) ↔
  (∃ n : ℕ, P = Polynomial.C (↑(1 : ℤ)) * Polynomial.X ^ n ∨ P = Polynomial.C (↑(-1 : ℤ)) * Polynomial.X ^ n) :=
sorry

end polynomial_relatively_prime_condition_l220_220682


namespace no_such_integers_l220_220825

theorem no_such_integers (a b : ℤ) : 
  ¬ (∃ a b : ℤ, ∃ k₁ k₂ : ℤ, a^5 * b + 3 = k₁^3 ∧ a * b^5 + 3 = k₂^3) :=
by 
  sorry

end no_such_integers_l220_220825


namespace line_equation_l220_220527

theorem line_equation (x y : ℝ) (c : ℝ)
  (h1 : 2 * x - y + 3 = 0)
  (h2 : 4 * x + 3 * y + 1 = 0)
  (h3 : 3 * x + 2 * y + c = 0) :
  c = 1 := sorry

end line_equation_l220_220527


namespace race_permutations_l220_220881

theorem race_permutations (r1 r2 r3 r4 : Type) [decidable_eq r1] [decidable_eq r2] [decidable_eq r3] [decidable_eq r4] :
  fintype.card (finset.univ : finset {l : list r1 | l ~ [r1, r2, r3, r4]}) = 24 :=
by
  sorry

end race_permutations_l220_220881


namespace find_c_exactly_two_common_points_l220_220873

theorem find_c_exactly_two_common_points (c : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^3 - 3*x1 + c = 0) ∧ (x2^3 - 3*x2 + c = 0)) ↔ (c = -2 ∨ c = 2) := 
sorry

end find_c_exactly_two_common_points_l220_220873


namespace multiplication_addition_l220_220050

theorem multiplication_addition :
  23 * 37 + 16 = 867 :=
by
  sorry

end multiplication_addition_l220_220050


namespace inequality_abc_l220_220300

theorem inequality_abc (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a :=
by
  sorry

end inequality_abc_l220_220300


namespace complement_of_A_relative_to_I_l220_220716

def I : Set ℤ := {-2, -1, 0, 1, 2}

def A : Set ℤ := {x : ℤ | x^2 < 3}

def complement_I_A : Set ℤ := {x ∈ I | x ∉ A}

theorem complement_of_A_relative_to_I :
  complement_I_A = {-2, 2} := by
  sorry

end complement_of_A_relative_to_I_l220_220716


namespace area_diminished_by_64_percent_l220_220625

/-- Given a rectangular field where both the length and width are diminished by 40%, 
    prove that the area is diminished by 64%. -/
theorem area_diminished_by_64_percent (L W : ℝ) :
  let L' := 0.6 * L
  let W' := 0.6 * W
  let A := L * W
  let A' := L' * W'
  (A - A') / A * 100 = 64 :=
by
  sorry

end area_diminished_by_64_percent_l220_220625


namespace find_n_l220_220938

theorem find_n (a : ℝ) (x : ℝ) (y : ℝ) (h1 : 0 < a) (h2 : a * x + 0.6 * a * y = 5 / 10)
(h3 : 1.6 * a * x + 1.2 * a * y = 1 - 1 / 10) : 
∃ n : ℕ, n = 10 :=
by
  sorry

end find_n_l220_220938


namespace fraction_identity_l220_220557

theorem fraction_identity (a b : ℚ) (h : (a - 2 * b) / b = 3 / 5) : a / b = 13 / 5 :=
sorry

end fraction_identity_l220_220557


namespace pow_div_l220_220950

theorem pow_div (x : ℕ) (a b c d : ℕ) (h1 : x^b = d) (h2 : x^(a*d) = c) : c / (d^b) = 512 := by
  sorry

end pow_div_l220_220950


namespace james_fence_problem_l220_220900

theorem james_fence_problem (w : ℝ) (hw : 0 ≤ w) (h_area : w * (2 * w + 10) ≥ 120) : w = 5 :=
by
  sorry

end james_fence_problem_l220_220900


namespace f_1996x_eq_1996_f_x_l220_220909

theorem f_1996x_eq_1996_f_x (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x ^ 2 - f x * f y + f y ^ 2)) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x :=
by
  sorry

end f_1996x_eq_1996_f_x_l220_220909


namespace sum_of_squares_l220_220430

theorem sum_of_squares (n : ℕ) (h : n * (n + 1) * (n + 2) = 12 * (3 * n + 3)) :
  n^2 + (n + 1)^2 + (n + 2)^2 = 29 := 
sorry

end sum_of_squares_l220_220430


namespace flower_stones_per_bracelet_l220_220672

theorem flower_stones_per_bracelet (total_stones : ℝ) (bracelets : ℝ)  (H_total: total_stones = 88.0) (H_bracelets: bracelets = 8.0) :
  (total_stones / bracelets = 11.0) :=
by
  rw [H_total, H_bracelets]
  norm_num

end flower_stones_per_bracelet_l220_220672


namespace smallest_k_for_mutual_criticism_l220_220566

-- Define a predicate that checks if a given configuration of criticisms lead to mutual criticism
def mutual_criticism_exists (deputies : ℕ) (k : ℕ) : Prop :=
  k ≥ 8 -- This is derived from the problem where k = 8 is the smallest k ensuring a mutual criticism

theorem smallest_k_for_mutual_criticism:
  mutual_criticism_exists 15 8 :=
by
  -- This is the theorem statement with the conditions and correct answer. The proof is omitted.
  sorry

end smallest_k_for_mutual_criticism_l220_220566


namespace book_total_pages_l220_220191

-- Define the conditions given in the problem
def pages_per_night : ℕ := 12
def nights_to_finish : ℕ := 10

-- State that the total number of pages in the book is 120 given the conditions
theorem book_total_pages : (pages_per_night * nights_to_finish) = 120 :=
by sorry

end book_total_pages_l220_220191


namespace walter_hushpuppies_per_guest_l220_220940

variables (guests hushpuppies_per_batch time_per_batch total_time : ℕ)

def batches (total_time time_per_batch : ℕ) : ℕ :=
  total_time / time_per_batch

def total_hushpuppies (batches hushpuppies_per_batch : ℕ) : ℕ :=
  batches * hushpuppies_per_batch

def hushpuppies_per_guest (total_hushpuppies guests : ℕ) : ℕ :=
  total_hushpuppies / guests

theorem walter_hushpuppies_per_guest :
  ∀ (guests hushpuppies_per_batch time_per_batch total_time : ℕ),
    guests = 20 →
    hushpuppies_per_batch = 10 →
    time_per_batch = 8 →
    total_time = 80 →
    hushpuppies_per_guest (total_hushpuppies (batches total_time time_per_batch) hushpuppies_per_batch) guests = 5 :=
by 
  intros _ _ _ _ h_guests h_hpb h_tpb h_tt
  sorry

end walter_hushpuppies_per_guest_l220_220940


namespace donation_to_second_home_l220_220922

-- Definitions of the conditions
def total_donation := 700.00
def first_home_donation := 245.00
def third_home_donation := 230.00

-- Define the unknown donation to the second home
noncomputable def second_home_donation := total_donation - first_home_donation - third_home_donation

-- The theorem to prove
theorem donation_to_second_home :
  second_home_donation = 225.00 :=
by sorry

end donation_to_second_home_l220_220922


namespace construct_unit_segment_l220_220242

-- Definitions of the problem
variable (a b : ℝ)

-- Parabola definition
def parabola (x : ℝ) : ℝ := x^2 + a * x + b

-- Statement of the problem in Lean 4
theorem construct_unit_segment
  (h : ∃ x y : ℝ, parabola a b x = y) :
  ∃ (u v : ℝ), abs (u - v) = 1 :=
sorry

end construct_unit_segment_l220_220242


namespace smallest_n_divisibility_l220_220779

theorem smallest_n_divisibility (n : ℕ) (h : n = 5) : 
  ∃ k, (1 ≤ k ∧ k ≤ n + 1) ∧ (n^2 - n) % k = 0 ∧ ∃ i, (1 ≤ i ∧ i ≤ n + 1) ∧ (n^2 - n) % i ≠ 0 :=
by
  sorry

end smallest_n_divisibility_l220_220779


namespace correct_permutation_count_l220_220814

noncomputable def numValidPermutations (n : ℕ) : ℕ :=
  if n = 2018 then (1009.factorial) * (1010.factorial) else 0

theorem correct_permutation_count : numValidPermutations 2018 = (1009.factorial) * (1010.factorial) :=
by
  sorry

end correct_permutation_count_l220_220814


namespace two_digit_number_difference_perfect_square_l220_220770

theorem two_digit_number_difference_perfect_square (N : ℕ) (a b : ℕ)
  (h1 : N = 10 * a + b)
  (h2 : N % 100 = N)
  (h3 : 1 ≤ a ∧ a ≤ 9)
  (h4 : 0 ≤ b ∧ b ≤ 9)
  (h5 : (N - (10 * b + a : ℕ)) = 64) : 
  N = 90 := 
sorry

end two_digit_number_difference_perfect_square_l220_220770


namespace find_y_l220_220932

variable {x y : ℤ}
variables (h1 : y = 2 * x - 3) (h2 : x + y = 57)

theorem find_y : y = 37 :=
by {
    sorry
}

end find_y_l220_220932


namespace chair_cost_l220_220003

theorem chair_cost :
  (∃ (C : ℝ), 3 * C + 50 + 40 = 130 - 4) → 
  (∃ (C : ℝ), C = 12) :=
by
  sorry

end chair_cost_l220_220003


namespace sine_addition_l220_220247

noncomputable def sin_inv_45 := Real.arcsin (4 / 5)
noncomputable def tan_inv_12 := Real.arctan (1 / 2)

theorem sine_addition :
  Real.sin (sin_inv_45 + tan_inv_12) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sine_addition_l220_220247


namespace first_digit_l220_220125

-- Definitions and conditions
def isDivisibleBy (n m : ℕ) : Prop := m ∣ n

def number (x y : ℕ) : ℕ := 653 * 100 + x * 10 + y

-- Main theorem
theorem first_digit (x y : ℕ) (h₁ : isDivisibleBy (number x y) 80) (h₂ : x + y = 2) : x = 2 :=
sorry

end first_digit_l220_220125


namespace race_distance_l220_220757

variable (speed_cristina speed_nicky head_start time_nicky : ℝ)

theorem race_distance
  (h1 : speed_cristina = 5)
  (h2 : speed_nicky = 3)
  (h3 : head_start = 12)
  (h4 : time_nicky = 30) :
  let time_cristina := time_nicky - head_start
  let distance_nicky := speed_nicky * time_nicky
  let distance_cristina := speed_cristina * time_cristina
  distance_nicky = 90 ∧ distance_cristina = 90 :=
by
  sorry

end race_distance_l220_220757


namespace one_fourth_of_six_point_eight_eq_seventeen_tenths_l220_220098

theorem one_fourth_of_six_point_eight_eq_seventeen_tenths :  (6.8 / 4) = (17 / 10) :=
by
  -- Skipping proof
  sorry

end one_fourth_of_six_point_eight_eq_seventeen_tenths_l220_220098


namespace average_score_l220_220311

variable (score : Fin 5 → ℤ)
variable (actual_score : ℤ)
variable (rank : Fin 5)
variable (average : ℤ)

def students_scores_conditions := 
  score 0 = 10 ∧ score 1 = -5 ∧ score 2 = 0 ∧ score 3 = 8 ∧ score 4 = -3 ∧
  actual_score = 90 ∧ rank.val = 2

theorem average_score (h : students_scores_conditions score actual_score rank) :
  average = 92 :=
sorry

end average_score_l220_220311


namespace runners_adjacent_vertices_after_2013_l220_220323

def hexagon_run_probability (t : ℕ) : ℚ :=
  (2 / 3) + (1 / 3) * ((1 / 4) ^ t)

theorem runners_adjacent_vertices_after_2013 :
  hexagon_run_probability 2013 = (2 / 3) + (1 / 3) * ((1 / 4) ^ 2013) := 
by 
  sorry

end runners_adjacent_vertices_after_2013_l220_220323


namespace hudson_daily_burger_spending_l220_220227

-- Definitions based on conditions
def total_spent := 465
def days_in_december := 31

-- Definition of the question
def amount_spent_per_day := total_spent / days_in_december

-- The theorem to prove
theorem hudson_daily_burger_spending : amount_spent_per_day = 15 := by
  sorry

end hudson_daily_burger_spending_l220_220227


namespace margaret_time_correct_l220_220293

def margaret_time : ℕ :=
  let n := 7
  let r := 15
  (Nat.factorial n) / r

theorem margaret_time_correct : margaret_time = 336 := by
  sorry

end margaret_time_correct_l220_220293


namespace unique_9_tuple_satisfying_condition_l220_220515

theorem unique_9_tuple_satisfying_condition :
  ∃! (a : Fin 9 → ℕ), 
    (∀ i j k : Fin 9, i < j ∧ j < k →
      ∃ l : Fin 9, l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ a i + a j + a k + a l = 100) :=
sorry

end unique_9_tuple_satisfying_condition_l220_220515


namespace daragh_sisters_count_l220_220243

theorem daragh_sisters_count (initial_bears : ℕ) (favorite_bears : ℕ) (eden_initial_bears : ℕ) (eden_total_bears : ℕ) 
    (remaining_bears := initial_bears - favorite_bears)
    (eden_received_bears := eden_total_bears - eden_initial_bears)
    (bears_per_sister := eden_received_bears) :
    initial_bears = 20 → favorite_bears = 8 → eden_initial_bears = 10 → eden_total_bears = 14 → 
    remaining_bears / bears_per_sister = 3 := 
by
  sorry

end daragh_sisters_count_l220_220243


namespace exponentiation_problem_l220_220143

theorem exponentiation_problem 
(a b : ℝ) 
(h : a ^ b = 1 / 8) : a ^ (-3 * b) = 512 := 
sorry

end exponentiation_problem_l220_220143


namespace double_root_polynomial_l220_220221

theorem double_root_polynomial (b4 b3 b2 b1 : ℤ) (s : ℤ) :
  (Polynomial.eval s (Polynomial.C 1 * Polynomial.X^5 + Polynomial.C b4 * Polynomial.X^4 + Polynomial.C b3 * Polynomial.X^3 + Polynomial.C b2 * Polynomial.X^2 + Polynomial.C b1 * Polynomial.X + Polynomial.C 24) = 0)
  ∧ (Polynomial.eval s (Polynomial.derivative (Polynomial.C 1 * Polynomial.X^5 + Polynomial.C b4 * Polynomial.X^4 + Polynomial.C b3 * Polynomial.X^3 + Polynomial.C b2 * Polynomial.X^2 + Polynomial.C b1 * Polynomial.X + Polynomial.C 24)) = 0)
  → s = 1 ∨ s = -1 ∨ s = 2 ∨ s = -2 :=
by
  sorry

end double_root_polynomial_l220_220221


namespace value_of_M_l220_220555

theorem value_of_M (M : ℝ) (h : 0.25 * M = 0.35 * 1200) : M = 1680 := 
sorry

end value_of_M_l220_220555


namespace evan_45_l220_220741

theorem evan_45 (k n : ℤ) (h1 : n + (k * (2 * k - 1)) = 60) : 60 - n = 45 :=
by sorry

end evan_45_l220_220741


namespace polynomial_use_square_of_binomial_form_l220_220053

theorem polynomial_use_square_of_binomial_form (a b x y : ℝ) :
  (1 + x) * (x + 1) = (x + 1) ^ 2 ∧ 
  (2 * a + b) * (b - 2 * a) = b^2 - 4 * a^2 ∧ 
  (-a + b) * (a - b) = - (a - b)^2 ∧ 
  (x^2 - y) * (y^2 + x) ≠ (x + y)^2 :=
by 
  sorry

end polynomial_use_square_of_binomial_form_l220_220053


namespace profit_percentage_l220_220660

theorem profit_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 500) (h_selling : selling_price = 750) :
  ((selling_price - cost_price) / cost_price) * 100 = 50 :=
by
  sorry

end profit_percentage_l220_220660


namespace marble_distribution_l220_220179

theorem marble_distribution (x : ℚ) :
    (2 * x + 2) + (3 * x) + (x + 4) = 56 ↔ x = 25 / 3 := by
  sorry

end marble_distribution_l220_220179


namespace evaluate_neg64_to_7_over_3_l220_220838

theorem evaluate_neg64_to_7_over_3 (a : ℝ) (b : ℝ) (c : ℝ) 
  (h1 : a = -64) (h2 : b = (-4)) (h3 : c = (7/3)) :
  a ^ c = -65536 := 
by
  have h4 : (-64 : ℝ) = (-4) ^ 3 := by sorry
  have h5 : a = b^3 := by rw [h1, h2, h4]
  have h6 : a ^ c = (b^3) ^ (7/3) := by rw [←h5, h3]
  have h7 : (b^3)^c = b^(3*(7/3)) := by sorry
  have h8 : b^(3*(7/3)) = b^7 := by norm_num
  have h9 : b^7 = -65536 := by sorry
  rw [h6, h7, h8, h9]
  exact h9

end evaluate_neg64_to_7_over_3_l220_220838


namespace line_through_point_area_T_l220_220500

variable (a T : ℝ)

def triangle_line_equation (a T : ℝ) : Prop :=
  ∃ y x : ℝ, (a^2 * y + 2 * T * x - 2 * a * T = 0) ∧ (y = -((2 * T)/a^2) * x + (2 * T) / a) ∧ (x ≥ 0) ∧ (y ≥ 0)

theorem line_through_point_area_T (a T : ℝ) (h₁ : a > 0) (h₂ : T > 0) :
  triangle_line_equation a T :=
sorry

end line_through_point_area_T_l220_220500


namespace scorpion_millipedes_needed_l220_220066

theorem scorpion_millipedes_needed 
  (total_segments_required : ℕ)
  (eaten_millipede_1_segments : ℕ)
  (eaten_millipede_2_segments : ℕ)
  (segments_per_millipede : ℕ)
  (n_millipedes_needed : ℕ)
  (total_segments : total_segments_required = 800) 
  (segments_1 : eaten_millipede_1_segments = 60)
  (segments_2 : eaten_millipede_2_segments = 2 * 60)
  (needed_segments_calculation : 800 - (60 + 2 * (2 * 60)) = n_millipedes_needed * 50) 
  : n_millipedes_needed = 10 :=
by
  sorry

end scorpion_millipedes_needed_l220_220066


namespace difference_of_squares_l220_220476

theorem difference_of_squares (a b : ℕ) (h₁ : a + b = 60) (h₂ : a - b = 14) : a^2 - b^2 = 840 := by
  sorry

end difference_of_squares_l220_220476


namespace find_k_l220_220271

noncomputable def line1 (t : ℝ) (k : ℝ) : ℝ × ℝ := (1 - 2 * t, 2 + k * t)
noncomputable def line2 (s : ℝ) : ℝ × ℝ := (s, 1 - 2 * s)

def correct_k (k : ℝ) : Prop :=
  let slope1 := -k / 2
  let slope2 := -2
  slope1 * slope2 = -1

theorem find_k (k : ℝ) (h_perpendicular : correct_k k) : k = -1 :=
sorry

end find_k_l220_220271


namespace chord_square_l220_220236

/-- 
Circles with radii 3 and 6 are externally tangent and are internally tangent to a circle with radius 9. 
The circle with radius 9 has a chord that is a common external tangent of the other two circles. Prove that 
the square of the length of this chord is 72.
-/
theorem chord_square (O₁ O₂ O₃ : Type) 
  (r₁ r₂ r₃ : ℝ) 
  (O₁_tangent_O₂ : r₁ + r₂ = 9) 
  (O₃_tangent_O₁ : r₃ - r₁ = 6) 
  (O₃_tangent_O₂ : r₃ - r₂ = 3) 
  (tangent_chord : ℝ) : 
  tangent_chord^2 = 72 :=
by sorry

end chord_square_l220_220236


namespace prove_heron_formula_prove_S_squared_rrarc_l220_220762

variables {r r_a r_b r_c p a b c S : ℝ}

-- Problem 1: Prove Heron's Formula
theorem prove_heron_formula (h1 : r * p = r_a * (p - a))
                            (h2 : r * r_a = (p - b) * (p - c))
                            (h3 : r_b * r_c = p * (p - a)) :
  S^2 = p * (p - a) * (p - b) * (p - c) :=
sorry

-- Problem 2: Prove S^2 = r * r_a * r_b * r_c
theorem prove_S_squared_rrarc (h1 : r * p = r_a * (p - a))
                              (h2 : r * r_a = (p - b) * (p - c))
                              (h3 : r_b * r_c = p * (p - a)) :
  S^2 = r * r_a * r_b * r_c :=
sorry

end prove_heron_formula_prove_S_squared_rrarc_l220_220762


namespace average_of_all_5_numbers_is_20_l220_220164

def average_of_all_5_numbers
  (sum_3_numbers : ℕ)
  (avg_2_numbers : ℕ) : ℕ :=
(sum_3_numbers + 2 * avg_2_numbers) / 5

theorem average_of_all_5_numbers_is_20 :
  average_of_all_5_numbers 48 26 = 20 :=
by
  unfold average_of_all_5_numbers -- Expand the definition
  -- Sum of 5 numbers is 48 (sum of 3) + (2 * 26) (sum of other 2)
  -- Total sum is 48 + 52 = 100
  -- Average is 100 / 5 = 20
  norm_num -- Check the numeric calculation
  -- sorry

end average_of_all_5_numbers_is_20_l220_220164


namespace sum_units_tens_not_divisible_by_4_l220_220155

theorem sum_units_tens_not_divisible_by_4 :
  ∃ (n : ℕ), (n = 3674 ∨ n = 3684 ∨ n = 3694 ∨ n = 3704 ∨ n = 3714 ∨ n = 3722) ∧
  (¬ (∃ k, (n % 100) = 4 * k)) ∧
  ((n % 10) + (n / 10 % 10) = 11) :=
sorry

end sum_units_tens_not_divisible_by_4_l220_220155


namespace incorrect_statement_among_given_options_l220_220296

theorem incorrect_statement_among_given_options :
  (∀ (b h : ℝ), 3 * (b * h) = (3 * b) * h) ∧
  (∀ (b h : ℝ), 3 * (1 / 2 * b * h) = 1 / 2 * b * (3 * h)) ∧
  (∀ (π r : ℝ), 9 * (π * r * r) ≠ (π * (3 * r) * (3 * r))) ∧
  (∀ (a b : ℝ), (3 * a) / (2 * b) ≠ a / b) ∧
  (∀ (x : ℝ), x < 0 → 3 * x < x) →
  false :=
by
  sorry

end incorrect_statement_among_given_options_l220_220296


namespace frog_jump_problem_l220_220325

theorem frog_jump_problem (A B C : ℝ) (PA PB PC : ℝ) 
  (H1: PA' = (PB + PC) / 2)
  (H2: jump_distance_B = 60)
  (H3: jump_distance_B = 2 * abs ((PB - (PB + PC) / 2))) :
  third_jump_distance = 30 := sorry

end frog_jump_problem_l220_220325


namespace initial_weight_of_solution_Y_is_8_l220_220920

theorem initial_weight_of_solution_Y_is_8
  (W : ℝ)
  (hw1 : 0.25 * W = 0.20 * W + 0.4)
  (hw2 : W ≠ 0) : W = 8 :=
by
  sorry

end initial_weight_of_solution_Y_is_8_l220_220920


namespace upstream_distance_is_48_l220_220981

variables (distance_downstream time_downstream time_upstream speed_stream : ℝ)
variables (speed_boat distance_upstream : ℝ)

-- Given conditions
axiom h1 : distance_downstream = 84
axiom h2 : time_downstream = 2
axiom h3 : time_upstream = 2
axiom h4 : speed_stream = 9

-- Define the effective speeds
def speed_downstream (speed_boat speed_stream : ℝ) := speed_boat + speed_stream
def speed_upstream (speed_boat speed_stream : ℝ) := speed_boat - speed_stream

-- Equations based on travel times and distances
axiom eq1 : distance_downstream = (speed_downstream speed_boat speed_stream) * time_downstream
axiom eq2 : distance_upstream = (speed_upstream speed_boat speed_stream) * time_upstream

-- Theorem to prove the distance rowed upstream is 48 km
theorem upstream_distance_is_48 :
  distance_upstream = 48 :=
by
  sorry

end upstream_distance_is_48_l220_220981


namespace legos_set_cost_l220_220851

-- Definitions for the conditions
def cars_sold : ℕ := 3
def price_per_car : ℕ := 5
def total_earned : ℕ := 45

-- The statement to prove
theorem legos_set_cost :
  total_earned - (cars_sold * price_per_car) = 30 := by
  sorry

end legos_set_cost_l220_220851


namespace circle_equation_exists_l220_220101

-- Define the necessary conditions
def tangent_to_x_axis (r b : ℝ) : Prop :=
  r^2 = b^2

def center_on_line (a b : ℝ) : Prop :=
  3 * a - b = 0

def intersects_formula (a b r : ℝ) : Prop :=
  2 * r^2 = (a - b)^2 + 14

-- Main theorem combining the conditions and proving the circles' equations
theorem circle_equation_exists (a b r : ℝ) :
  tangent_to_x_axis r b →
  center_on_line a b →
  intersects_formula a b r →
  ((x - 1)^2 + (y - 3)^2 = 9 ∨ (x + 1)^2 + (y + 3)^2 = 9) :=
by
  intros h_tangent h_center h_intersects
  sorry

end circle_equation_exists_l220_220101


namespace simple_interest_rate_l220_220986

theorem simple_interest_rate (P R: ℝ) (T: ℝ) (H: T = 5) (H1: P * (1/6) = P * (R * T / 100)) : R = 10/3 :=
by {
  sorry
}

end simple_interest_rate_l220_220986


namespace eight_pow_15_div_sixtyfour_pow_6_l220_220945

theorem eight_pow_15_div_sixtyfour_pow_6 :
  8^15 / 64^6 = 512 := by
  sorry

end eight_pow_15_div_sixtyfour_pow_6_l220_220945


namespace find_x_l220_220310

variable (A B x : ℝ)
variable (h1 : A > 0) (h2 : B > 0)
variable (h3 : A = (x / 100) * B)

theorem find_x : x = 100 * (A / B) :=
by
  sorry

end find_x_l220_220310


namespace solve_system_for_x_l220_220423

theorem solve_system_for_x :
  ∃ x y : ℝ, (2 * x + y = 4) ∧ (x + 2 * y = 5) ∧ (x = 1) :=
by
  sorry

end solve_system_for_x_l220_220423


namespace determine_c_l220_220518

theorem determine_c (c : ℝ) :
  let vertex_x := -(-10 / (2 * 1))
  let vertex_y := c - ((-10)^2 / (4 * 1))
  ((5 - 0)^2 + (vertex_y - 0)^2 = 10^2)
  → (c = 25 + 5 * Real.sqrt 3 ∨ c = 25 - 5 * Real.sqrt 3) :=
by
  sorry

end determine_c_l220_220518


namespace powers_of_two_l220_220524

theorem powers_of_two (n : ℕ) (h : ∀ n, ∃ m, (2^n - 1) ∣ (m^2 + 9)) : ∃ s, n = 2^s :=
sorry

end powers_of_two_l220_220524


namespace find_points_per_enemy_l220_220623

def points_per_enemy (x : ℕ) : Prop :=
  let points_from_enemies := 6 * x
  let additional_points := 8
  let total_points := points_from_enemies + additional_points
  total_points = 62

theorem find_points_per_enemy (x : ℕ) (h : points_per_enemy x) : x = 9 :=
  by sorry

end find_points_per_enemy_l220_220623


namespace find_number_l220_220632

theorem find_number (x : ℝ) : (8 * x = 0.4 * 900) -> x = 45 :=
by
  sorry

end find_number_l220_220632


namespace exponentiation_rule_l220_220256

theorem exponentiation_rule (a m : ℕ) (h : (a^2)^m = a^6) : m = 3 :=
by
  sorry

end exponentiation_rule_l220_220256


namespace probability_no_shaded_square_l220_220201

theorem probability_no_shaded_square :
  let n := (2006 * 2005) / 2,
      m := 1003 * 1003 in
  (n - m) / n = 1002 / 2005 :=
by 
  sorry

end probability_no_shaded_square_l220_220201


namespace base_k_132_eq_30_l220_220790

theorem base_k_132_eq_30 (k : ℕ) (h : 1 * k^2 + 3 * k + 2 = 30) : k = 4 :=
sorry

end base_k_132_eq_30_l220_220790


namespace find_a_b_find_min_k_harmonic_ln_inequality_l220_220871

noncomputable def f (x : ℝ) (a b : ℝ) := a * x ^ 3 + b * x ^ 2

-- Definitions based on the problem conditions
axiom extreme_value_at_one (a b : ℝ) : f 1 a b = 1/6
axiom extreme_point_derivative (a b : ℝ) : (3 * a * 1 ^ 2 + 2 * b * 1) = 0

theorem find_a_b (a b : ℝ) : a = -1/3 ∧ b = 1/2 :=
sorry

-- Definitions for the inequality f'(x) ≤ k ln (x + 1)
noncomputable def f' (x : ℝ) (a b : ℝ) := 3 * a * x ^ 2 + 2 * b * x

axiom derivative_inequality (k : ℝ) (x : ℝ) (a b : ℝ) :
  x ∈ Set.Ici 0 → f' x a b ≤ k * Real.log (x + 1)

theorem find_min_k : 1 ≤ 1 :=
by
  -- Value derived from the problem's conclusion
  sorry

-- Series inequality problem
theorem harmonic_ln_inequality (n : ℕ) (hn : 0 < n) : 
  (∑ i in Finset.range (n + 1), 1 / (i + 1)) < Real.log (n + 1) + 2 :=
sorry

end find_a_b_find_min_k_harmonic_ln_inequality_l220_220871


namespace total_pages_read_l220_220137

theorem total_pages_read (J A C D : ℝ) 
  (hJ : J = 20)
  (hA : A = 2 * J + 2)
  (hC : C = J * A - 17)
  (hD : D = (C + J) / 2) :
  J + A + C + D = 1306.5 :=
by
  sorry

end total_pages_read_l220_220137


namespace new_car_distance_l220_220218

-- State the given conditions as a Lemma in Lean 4
theorem new_car_distance (d_old : ℝ) (d_new : ℝ) (h1 : d_old = 150) (h2 : d_new = d_old * 1.30) : d_new = 195 := 
by
  sorry

end new_car_distance_l220_220218


namespace chord_length_is_four_l220_220215

theorem chord_length_is_four :
  let line := {p : ℝ × ℝ | ∃ t : ℝ, p.1 = 2 + 2 * t ∧ p.2 = -t}
  let circle_center : ℝ × ℝ := (2, 0)
  let circle_radius : ℝ := 2
  let circle := {p : ℝ × ℝ | (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 = circle_radius^2}
  ∀ a b : ℝ × ℝ, a ∈ line → b ∈ line → a ∈ circle → b ∈ circle → a ≠ b → dist a b = 4 :=
by
  sorry

end chord_length_is_four_l220_220215


namespace promotional_savings_l220_220811

noncomputable def y (x : ℝ) : ℝ :=
if x ≤ 500 then x
else if x ≤ 1000 then 500 + 0.8 * (x - 500)
else 500 + 400 + 0.5 * (x - 1000)

theorem promotional_savings (payment : ℝ) (hx : y 2400 = 1600) : 2400 - payment = 800 :=
by sorry

end promotional_savings_l220_220811


namespace equal_real_roots_of_quadratic_l220_220720

theorem equal_real_roots_of_quadratic (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 3 = 0 ∧ 
               (∀ y : ℝ, 3 * y^2 - m * y + 3 = 0 → y = x)) → 
  m = 6 ∨ m = -6 :=
by
  sorry  -- proof to be filled in.

end equal_real_roots_of_quadratic_l220_220720


namespace simplify_expression_l220_220341

theorem simplify_expression :
  1 + 1 / (1 + 1 / (2 + 2)) = 9 / 5 := by
  sorry

end simplify_expression_l220_220341


namespace departs_if_arrives_l220_220133

theorem departs_if_arrives (grain_quantity : ℤ) (h : grain_quantity = 30) : -grain_quantity = -30 :=
by {
  have : -grain_quantity = -30,
  from congr_arg (λ x, -x) h,
  exact this
}

end departs_if_arrives_l220_220133


namespace coin_flip_probability_l220_220399

open ProbabilityTheory

theorem coin_flip_probability :
  let S := { outcomes | List.length (List.filter (λ x => x = true) outcomes) > List.length (List.filter (λ x => x = false) outcomes) }
  Pr (S : Set (List (Bool))) = 1 / 2 :=
by
  sorry

end coin_flip_probability_l220_220399


namespace divisible_by_11_l220_220255

theorem divisible_by_11 (n : ℤ) : (11 ∣ (n^2001 - n^4)) ↔ (n % 11 = 0 ∨ n % 11 = 1) :=
by
  sorry

end divisible_by_11_l220_220255


namespace pow_div_eq_l220_220949

theorem pow_div_eq : (8:ℕ) ^ 15 / (64:ℕ) ^ 6 = 512 := by
  have h1 : 64 = 8 ^ 2 := by sorry
  have h2 : (64:ℕ) ^ 6 = (8 ^ 2) ^ 6 := by sorry
  have h3 : (8 ^ 2) ^ 6 = 8 ^ 12 := by sorry
  have h4 : (8:ℕ) ^ 15 / (8 ^ 12) = 8 ^ (15 - 12) := by sorry
  have h5 : 8 ^ 3 = 512 := by sorry
  exact sorry

end pow_div_eq_l220_220949


namespace minimum_sum_l220_220514

theorem minimum_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
    (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) + ((a^2 * b) / (18 * b * c)) ≥ 4 / 9 :=
sorry

end minimum_sum_l220_220514


namespace arithmetic_sequence_a5_l220_220574

variables {a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ}

/-- The sum of the first 9 terms of an arithmetic sequence {a_n}, denoted by S_9, equals 45. 
    Prove that the 5th term, a_5, is 5. -/
theorem arithmetic_sequence_a5 {S_9 : ℤ} (h : S_9 = 45)
  (H : S_9 = (9 * (a_1 + a_9)) / 2) (H2 : a_1 + a_9 = 2 * a_5)
  : a_5 = 5 :=
by {
  /- Proof will be filled in later -/
  sorry
}

end arithmetic_sequence_a5_l220_220574


namespace smallest_positive_integer_div_conditions_l220_220782

theorem smallest_positive_integer_div_conditions :
  ∃ x : ℕ, x > 0 ∧ x % 5 = 2 ∧ x % 7 = 3 ∧ ∀ y : ℕ, (y > 0 ∧ y % 5 = 2 ∧ y % 7 = 3) → x ≤ y :=
  sorry

end smallest_positive_integer_div_conditions_l220_220782


namespace no_four_digit_perfect_square_palindromes_l220_220658

def is_palindrome (n : ℕ) : Prop :=
  (n.toString = n.toString.reverse)

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_four_digit_perfect_square_palindromes :
  ¬ ∃ n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
by introv H, cases H with n Hn sorry

end no_four_digit_perfect_square_palindromes_l220_220658


namespace diagonals_in_polygon_with_150_sides_l220_220231

-- (a) Definitions for conditions
def sides : ℕ := 150

def diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- (c) Statement of the problem in Lean 4
theorem diagonals_in_polygon_with_150_sides :
  diagonals sides = 11025 :=
by
  sorry

end diagonals_in_polygon_with_150_sides_l220_220231


namespace students_not_skating_nor_skiing_l220_220360

theorem students_not_skating_nor_skiing (total_students skating_students skiing_students both_students : ℕ)
  (h_total : total_students = 30)
  (h_skating : skating_students = 20)
  (h_skiing : skiing_students = 9)
  (h_both : both_students = 5) :
  total_students - (skating_students + skiing_students - both_students) = 6 :=
by
  sorry

end students_not_skating_nor_skiing_l220_220360


namespace all_elements_rational_l220_220004

open Set

def finite_set_in_interval (n : ℕ) : Set ℝ :=
  {x | ∃ i, i ∈ Finset.range (n + 1) ∧ (x = 0 ∨ x = 1 ∨ 0 < x ∧ x < 1)}

def unique_distance_condition (S : Set ℝ) : Prop :=
  ∀ d, d ≠ 1 → ∃ x_i x_j x_k x_l, x_i ∈ S ∧ x_j ∈ S ∧ x_k ∈ S ∧ x_l ∈ S ∧ 
        abs (x_i - x_j) = d ∧ abs (x_k - x_l) = d ∧ (x_i = x_k → x_j ≠ x_l)

theorem all_elements_rational
  (n : ℕ)
  (S : Set ℝ)
  (hS1 : ∀ x ∈ S, 0 ≤ x ∧ x ≤ 1)
  (hS2 : 0 ∈ S)
  (hS3 : 1 ∈ S)
  (hS4 : unique_distance_condition S) :
  ∀ x ∈ S, ∃ q : ℚ, (x : ℝ) = q := 
sorry

end all_elements_rational_l220_220004


namespace sandy_earnings_correct_l220_220586

def hourly_rate : ℕ := 15
def hours_worked_friday : ℕ := 10
def hours_worked_saturday : ℕ := 6
def hours_worked_sunday : ℕ := 14

def earnings_friday : ℕ := hours_worked_friday * hourly_rate
def earnings_saturday : ℕ := hours_worked_saturday * hourly_rate
def earnings_sunday : ℕ := hours_worked_sunday * hourly_rate

def total_earnings : ℕ := earnings_friday + earnings_saturday + earnings_sunday

theorem sandy_earnings_correct : total_earnings = 450 := by
  sorry

end sandy_earnings_correct_l220_220586


namespace son_present_age_l220_220635

variable (S F : ℕ)

-- Given conditions
def father_age := F = S + 34
def future_age_rel := F + 2 = 2 * (S + 2)

-- Theorem to prove the son's current age
theorem son_present_age (h₁ : father_age S F) (h₂ : future_age_rel S F) : S = 32 := by
  sorry

end son_present_age_l220_220635


namespace shorter_piece_length_l220_220488

noncomputable def total_length : ℝ := 140
noncomputable def ratio : ℝ := 2 / 5

theorem shorter_piece_length (x : ℝ) (y : ℝ) (h1 : x + y = total_length) (h2 : x = ratio * y) : x = 40 :=
by
  sorry

end shorter_piece_length_l220_220488


namespace cleaning_time_is_correct_l220_220671

-- Define the given conditions
def vacuuming_minutes_per_day : ℕ := 30
def vacuuming_days_per_week : ℕ := 3
def dusting_minutes_per_day : ℕ := 20
def dusting_days_per_week : ℕ := 2

-- Define the total cleaning time per week
def total_cleaning_time_per_week : ℕ :=
  (vacuuming_minutes_per_day * vacuuming_days_per_week) + (dusting_minutes_per_day * dusting_days_per_week)

-- State the theorem we want to prove
theorem cleaning_time_is_correct : total_cleaning_time_per_week = 130 := by
  sorry

end cleaning_time_is_correct_l220_220671


namespace statement_1_statement_2_statement_3_statement_4_l220_220848

variables (a b c x0 : ℝ)
noncomputable def P (x : ℝ) : ℝ := a*x^2 + b*x + c

-- Statement ①
theorem statement_1 (h : a - b + c = 0) : P a b c (-1) = 0 := sorry

-- Statement ②
theorem statement_2 (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a*x1^2 + c = 0 ∧ a*x2^2 + c = 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ P a b c x1 = 0 ∧ P a b c x2 = 0 := sorry

-- Statement ③
theorem statement_3 (h : P a b c c = 0) : a*c + b + 1 = 0 := sorry

-- Statement ④
theorem statement_4 (h : P a b c x0 = 0) : b^2 - 4*a*c = (2*a*x0 + b)^2 := sorry

end statement_1_statement_2_statement_3_statement_4_l220_220848


namespace count_palindromic_four_digit_perfect_squares_l220_220641

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem count_palindromic_four_digit_perfect_squares : 
  (finset.card {n ∈ (finset.range 10000).filter (λ x, is_four_digit x ∧ is_perfect_square x ∧ is_palindrome x)} = 5) :=
by
  sorry

end count_palindromic_four_digit_perfect_squares_l220_220641


namespace cos_330_eq_sqrt3_div_2_l220_220238

theorem cos_330_eq_sqrt3_div_2
    (h1 : ∀ θ : ℝ, Real.cos (2 * Real.pi - θ) = Real.cos θ)
    (h2 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2) :
    Real.cos (11 * Real.pi / 6) = Real.sqrt 3 / 2 :=
by
  -- Proof goes here
  sorry

end cos_330_eq_sqrt3_div_2_l220_220238


namespace simplify_expression_l220_220160

def i : Complex := Complex.I

theorem simplify_expression : 7 * (4 - 2 * i) + 4 * i * (7 - 2 * i) = 36 + 14 * i := by
  sorry

end simplify_expression_l220_220160


namespace angle_F_values_l220_220898

noncomputable def sin_f_values (D E : ℝ) (h1 : 2 * Real.sin D + 3 * Real.cos E = 3)
                                      (h2 : 3 * Real.sin E + 5 * Real.cos D = 4) : Set ℝ :=
{F | F = 14.5 ∨ F = 165.5}

theorem angle_F_values (D E : ℝ) (h1 : 2 * Real.sin D + 3 * Real.cos E = 3)
                                    (h2 : 3 * Real.sin E + 5 * Real.cos D = 4) :
  sin_f_values D E h1 h2 = {14.5, 165.5} :=
by simp [sin_f_values, h1, h2]; sorry

end angle_F_values_l220_220898


namespace bounded_region_area_is_1800_l220_220467

open Real

def region := {p : ℝ × ℝ | let (x, y) := p in 
  (x >= 0 → y^2 + 2*x*y + 60*x = 900) ∧ 
  (x < 0 → y^2 + 2*x*y - 60*x = 900)}

noncomputable def area_bounded_region : ℝ :=
  let vertices := [(0, 30), (0, -30), (30, -30), (-30, 30)] in
  let base := 30 - 0 in
  let height := 30 - (-30) in
  base * height

theorem bounded_region_area_is_1800 :
  ∃ (area : ℝ), area = area_bounded_region ∧ area = 1800 := by
  sorry

end bounded_region_area_is_1800_l220_220467


namespace stratified_sampling_example_l220_220798

theorem stratified_sampling_example 
  (N : ℕ) (S : ℕ) (D : ℕ) 
  (hN : N = 1000) (hS : S = 50) (hD : D = 200) : 
  D * (S : ℝ) / (N : ℝ) = 10 := 
by
  sorry

end stratified_sampling_example_l220_220798


namespace new_students_correct_l220_220434

variable 
  (students_start_year : Nat)
  (students_left : Nat)
  (students_end_year : Nat)

def new_students (students_start_year students_left students_end_year : Nat) : Nat :=
  students_end_year - (students_start_year - students_left)

theorem new_students_correct :
  ∀ (students_start_year students_left students_end_year : Nat),
  students_start_year = 10 →
  students_left = 4 →
  students_end_year = 48 →
  new_students students_start_year students_left students_end_year = 42 :=
by
  intros students_start_year students_left students_end_year h1 h2 h3
  rw [h1, h2, h3]
  unfold new_students
  norm_num

end new_students_correct_l220_220434


namespace tiles_needed_l220_220983

def tile_area : ℕ := 3 * 4
def floor_area : ℕ := 36 * 60

theorem tiles_needed : floor_area / tile_area = 180 := by
  sorry

end tiles_needed_l220_220983


namespace lcm_of_4_8_9_10_l220_220963

theorem lcm_of_4_8_9_10 : Nat.lcm (Nat.lcm 4 8) (Nat.lcm 9 10) = 360 := by
  sorry

end lcm_of_4_8_9_10_l220_220963


namespace evaluate_expression_l220_220335

theorem evaluate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) : -2 * a - b^2 + 2 * a * b = -41 := by
  sorry

end evaluate_expression_l220_220335


namespace arithmetic_sequence_common_difference_l220_220787

variable (a₁ d : ℝ)

def sum_odd := 5 * a₁ + 20 * d
def sum_even := 5 * a₁ + 25 * d

theorem arithmetic_sequence_common_difference 
  (h₁ : sum_odd a₁ d = 15) 
  (h₂ : sum_even a₁ d = 30) :
  d = 3 := 
by
  sorry

end arithmetic_sequence_common_difference_l220_220787


namespace bjorn_cannot_prevent_vakha_l220_220298

-- Define the primary settings and objects involved
def n_points : ℕ := 99
inductive Color
| red 
| blue 

structure GameState :=
  (turn : ℕ)
  (points : Fin n_points → Option Color)

-- Define the valid states of the game where turn must be within the range of points
def valid_state (s : GameState) : Prop :=
  s.turn ≤ n_points ∧ ∀ p, s.points p ≠ none

-- Define what it means for an equilateral triangle to be monochromatically colored
def monochromatic_equilateral_triangle (state : GameState) : Prop :=
  ∃ (p1 p2 p3 : Fin n_points), 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    (p1.val + (n_points/3) % n_points) = p2.val ∧
    (p2.val + (n_points/3) % n_points) = p3.val ∧
    (p3.val + (n_points/3) % n_points) = p1.val ∧
    (state.points p1 = state.points p2) ∧ 
    (state.points p2 = state.points p3)

-- Vakha's winning condition
def vakha_wins (state : GameState) : Prop := 
  monochromatic_equilateral_triangle state

-- Bjorn's winning condition prevents Vakha from winning
def bjorn_can_prevent_vakha (initial_state : GameState) : Prop :=
  ¬ vakha_wins initial_state

-- Main theorem stating Bjorn cannot prevent Vakha from winning
theorem bjorn_cannot_prevent_vakha : ∀ (initial_state : GameState),
  valid_state initial_state → ¬ bjorn_can_prevent_vakha initial_state :=
sorry

end bjorn_cannot_prevent_vakha_l220_220298


namespace quadratic_rewrite_l220_220882

theorem quadratic_rewrite (a b c x : ℤ) :
  (16 * x^2 - 40 * x - 72 = a^2 * x^2 + 2 * a * b * x + b^2 + c) →
  (a = 4 ∨ a = -4) →
  (2 * a * b = -40) →
  ab = -20 := by
sorry

end quadratic_rewrite_l220_220882


namespace concert_tickets_l220_220040

theorem concert_tickets (A C : ℕ) (h1 : C = 3 * A) (h2 : 7 * A + 3 * C = 6000) : A + C = 1500 :=
by {
  -- Proof omitted
  sorry
}

end concert_tickets_l220_220040


namespace complex_product_l220_220538

theorem complex_product (z1 z2 : ℂ) (h1 : Complex.abs z1 = 1) (h2 : Complex.abs z2 = 1) 
(h3 : z1 + z2 = -7/5 + (1/5) * Complex.I) : 
  z1 * z2 = 24/25 - (7/25) * Complex.I :=
by
  sorry

end complex_product_l220_220538


namespace trigonometric_identity_l220_220259

open Real

noncomputable def tan_condition (α : ℝ) : Prop := tan (-α) = 3

theorem trigonometric_identity (α : ℝ) (h : tan_condition α) : 
  (sin α)^2 - sin (2 * α) = 8 * cos (2 * α) :=
sorry

end trigonometric_identity_l220_220259


namespace problem_sol_l220_220510

-- Assume g is an invertible function
variable (g : ℝ → ℝ) (g_inv : ℝ → ℝ)
variable (h_invertible : ∀ y, g (g_inv y) = y ∧ g_inv (g y) = y)

-- Define p and q such that g(p) = 3 and g(q) = 5
variable (p q : ℝ)
variable (h1 : g p = 3) (h2 : g q = 5)

-- Goal to prove that p - q = 2
theorem problem_sol : p - q = 2 :=
by
  sorry

end problem_sol_l220_220510


namespace find_wrongly_written_height_l220_220597

def wrongly_written_height
  (n : ℕ)
  (avg_height_incorrect : ℝ)
  (actual_height : ℝ)
  (avg_height_correct : ℝ) : ℝ :=
  let total_height_incorrect := n * avg_height_incorrect
  let total_height_correct := n * avg_height_correct
  let height_difference := total_height_incorrect - total_height_correct
  actual_height + height_difference

theorem find_wrongly_written_height :
  wrongly_written_height 35 182 106 180 = 176 :=
by
  sorry

end find_wrongly_written_height_l220_220597


namespace A_completion_time_l220_220495

theorem A_completion_time :
  ∃ A : ℝ, (A > 0) ∧ (
    (2 * (1 / A + 1 / 10) + 3.0000000000000004 * (1 / 10) = 1) ↔ A = 4
  ) :=
by
  have B_workday := 10
  sorry -- proof would go here

end A_completion_time_l220_220495


namespace fish_ratio_l220_220156

variables (O R B : ℕ)
variables (h1 : O = B + 25)
variables (h2 : B = 75)
variables (h3 : (O + B + R) / 3 = 75)

theorem fish_ratio : R / O = 1 / 2 :=
sorry

end fish_ratio_l220_220156


namespace snowboard_price_after_discounts_l220_220011

theorem snowboard_price_after_discounts
  (original_price : ℝ) (friday_discount_rate : ℝ) (monday_discount_rate : ℝ) 
  (sales_tax_rate : ℝ) (price_after_all_adjustments : ℝ) :
  original_price = 200 →
  friday_discount_rate = 0.40 →
  monday_discount_rate = 0.20 →
  sales_tax_rate = 0.05 →
  price_after_all_adjustments = 100.80 :=
by
  intros
  sorry

end snowboard_price_after_discounts_l220_220011


namespace original_population_divisor_l220_220914

theorem original_population_divisor (a b c : ℕ) (ha : ∃ a, ∃ b, ∃ c, a^2 + 120 = b^2 ∧ b^2 + 80 = c^2) :
  7 ∣ a :=
by
  sorry

end original_population_divisor_l220_220914


namespace sin_cos_power_sum_l220_220905

theorem sin_cos_power_sum (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 4) : 
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 61 / 64 := 
by
  sorry

end sin_cos_power_sum_l220_220905


namespace problem_l220_220888

noncomputable def f : ℝ → ℝ := sorry

theorem problem (f_decreasing : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x)
                (h : ∀ x : ℝ, 0 < x → f (f x - 1 / Real.exp x) = 1 / Real.exp 1 + 1) :
  f (Real.log 2) = 3 / 2 :=
sorry

end problem_l220_220888


namespace cost_of_each_notebook_is_3_l220_220385

noncomputable def notebooks_cost (total_spent : ℕ) (backpack_cost : ℕ) (pens_cost : ℕ) (pencils_cost : ℕ) (num_notebooks : ℕ) : ℕ :=
  (total_spent - (backpack_cost + pens_cost + pencils_cost)) / num_notebooks

theorem cost_of_each_notebook_is_3 :
  notebooks_cost 32 15 1 1 5 = 3 :=
by
  sorry

end cost_of_each_notebook_is_3_l220_220385


namespace intersection_correct_l220_220876

def A (x : ℝ) : Prop := |x| > 4
def B (x : ℝ) : Prop := -2 < x ∧ x ≤ 6
def intersection (x : ℝ) : Prop := B x ∧ A x

theorem intersection_correct :
  ∀ x : ℝ, intersection x ↔ 4 < x ∧ x ≤ 6 := 
by
  sorry

end intersection_correct_l220_220876


namespace avg_velocity_2_to_2_1_l220_220605

def motion_eq (t : ℝ) : ℝ := 3 + t^2

theorem avg_velocity_2_to_2_1 : 
  (motion_eq 2.1 - motion_eq 2) / (2.1 - 2) = 4.1 :=
by
  sorry

end avg_velocity_2_to_2_1_l220_220605


namespace charity_delivered_100_plates_l220_220209

variables (cost_rice_per_plate cost_chicken_per_plate total_amount_spent : ℝ)
variable (P : ℝ)

-- Conditions provided
def rice_cost : ℝ := 0.10
def chicken_cost : ℝ := 0.40
def total_spent : ℝ := 50
def total_cost_per_plate : ℝ := rice_cost + chicken_cost

-- Lean 4 statement to prove:
theorem charity_delivered_100_plates :
  total_spent = 50 →
  total_cost_per_plate = rice_cost + chicken_cost →
  rice_cost = 0.10 →
  chicken_cost = 0.40 →
  P = total_spent / total_cost_per_plate →
  P = 100 :=
by
  sorry

end charity_delivered_100_plates_l220_220209


namespace price_increase_percentage_l220_220093

-- Define the problem conditions
def lowest_price := 12
def highest_price := 21

-- Formulate the goal as a theorem
theorem price_increase_percentage :
  ((highest_price - lowest_price) / lowest_price : ℚ) * 100 = 75 := by
  sorry

end price_increase_percentage_l220_220093


namespace arithmetic_sequence_15th_term_l220_220028

theorem arithmetic_sequence_15th_term (a1 a2 a3 : ℕ) (d : ℕ) (n : ℕ) (h1 : a1 = 3) (h2 : a2 = 14) (h3 : a3 = 25) (h4 : d = a2 - a1) (h5 : a2 - a1 = a3 - a2) (h6 : n = 15) :
  a1 + (n - 1) * d = 157 :=
by
  -- Proof goes here
  sorry

end arithmetic_sequence_15th_term_l220_220028


namespace scientific_notation_of_3395000_l220_220350

theorem scientific_notation_of_3395000 :
  3395000 = 3.395 * 10^6 :=
sorry

end scientific_notation_of_3395000_l220_220350


namespace a_sequence_arithmetic_sum_of_bn_l220_220000

   noncomputable def a (n : ℕ) : ℕ := 1 + n

   def S (n : ℕ) : ℕ := n * (n + 1) / 2

   def b (n : ℕ) : ℚ := 1 / S n

   def T (n : ℕ) : ℚ := (Finset.range n).sum b

   theorem a_sequence_arithmetic (n : ℕ) (a_n_positive : ∀ n, a n > 0)
     (a₁_is_one : a 0 = 1) :
     (a (n+1)) - a n = 1 := by
     sorry

   theorem sum_of_bn (n : ℕ) :
     T n = 2 * n / (n + 1) := by
     sorry
   
end a_sequence_arithmetic_sum_of_bn_l220_220000


namespace pencils_remaining_l220_220379

variable (initial_pencils : ℝ) (pencils_given : ℝ)

theorem pencils_remaining (h1 : initial_pencils = 56.0) 
                          (h2 : pencils_given = 9.5) 
                          : initial_pencils - pencils_given = 46.5 :=
by 
  sorry

end pencils_remaining_l220_220379


namespace GCD_180_252_315_l220_220329

theorem GCD_180_252_315 : Nat.gcd 180 (Nat.gcd 252 315) = 9 := by
  sorry

end GCD_180_252_315_l220_220329


namespace evaluate_expression_l220_220336

theorem evaluate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) : -2 * a - b^2 + 2 * a * b = -41 := by
  sorry

end evaluate_expression_l220_220336


namespace calculate_expression_l220_220744

theorem calculate_expression (p q r s : ℝ)
  (h1 : p + q + r + s = 10)
  (h2 : p^2 + q^2 + r^2 + s^2 = 26) :
  6 * (p^4 + q^4 + r^4 + s^4) - (p^3 + q^3 + r^3 + s^3) =
    6 * ((p-1)^4 + (q-1)^4 + (r-1)^4 + (s-1)^4) - ((p-1)^3 + (q-1)^3 + (r-1)^3 + (s-1)^3) :=
by {
  sorry
}

end calculate_expression_l220_220744


namespace neg_q_necessary_not_sufficient_for_neg_p_l220_220446

-- Proposition p: |x + 2| > 2
def p (x : ℝ) : Prop := abs (x + 2) > 2

-- Proposition q: 1 / (3 - x) > 1
def q (x : ℝ) : Prop := 1 / (3 - x) > 1

-- Negation of p and q
def neg_p (x : ℝ) : Prop := -4 ≤ x ∧ x ≤ 0
def neg_q (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 3

-- Theorem: negation of q is a necessary but not sufficient condition for negation of p
theorem neg_q_necessary_not_sufficient_for_neg_p :
  (∀ x : ℝ, neg_p x → neg_q x) ∧ (∃ x : ℝ, neg_q x ∧ ¬neg_p x) :=
by
  sorry

end neg_q_necessary_not_sufficient_for_neg_p_l220_220446


namespace book_pages_l220_220189

-- Define the conditions
def pages_per_night : ℕ := 12
def nights : ℕ := 10

-- Define the total pages in the book
def total_pages : ℕ := pages_per_night * nights

-- Prove that the total number of pages is 120
theorem book_pages : total_pages = 120 := by
  sorry

end book_pages_l220_220189


namespace prob_roll_678_before_less_than_5_l220_220212

open ProbabilityTheory

theorem prob_roll_678_before_less_than_5 :
  (fair_probability (of_real (1 / 960))) = (probability_of_condition (roll_until (< 5) => sequence_increasing [6, 7, 8])) :=
sorry

end prob_roll_678_before_less_than_5_l220_220212


namespace intersect_sets_l220_220572

open Set

theorem intersect_sets (A B : Set ℝ) (hA : A = {x | abs x < 3}) (hB : B = {x | 2^x > 1}) :
  A ∩ B = {x | 0 < x ∧ x < 3} := 
by
  sorry

end intersect_sets_l220_220572


namespace arithmetic_seq_S13_l220_220704

noncomputable def arithmetic_sequence_sum (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_seq_S13 (a_1 d : ℕ) (h : a_1 + 6 * d = 10) :
  arithmetic_sequence_sum a_1 d 13 = 130 :=
by
  sorry

end arithmetic_seq_S13_l220_220704


namespace find_p_for_natural_roots_l220_220095

-- The polynomial is given.
def cubic_polynomial (p x : ℝ) : ℝ := 5 * x^3 - 5 * (p + 1) * x^2 + (71 * p - 1) * x + 1

-- Problem statement to prove that p = 76 is the only real number such that
-- the cubic polynomial cubic_polynomial equals 66 * p has at least two natural number roots.
theorem find_p_for_natural_roots (p : ℝ) :
  (∃ (u v : ℕ), u ≠ v ∧ cubic_polynomial p u = 66 * p ∧ cubic_polynomial p v = 66 * p) ↔ p = 76 :=
by
  sorry

end find_p_for_natural_roots_l220_220095


namespace first_woman_hours_l220_220802

-- Definitions and conditions
variables (W k y t η : ℝ)
variables (work_rate : k * y * 45 = W)
variables (total_work : W = k * (t * ((y-1) * y) / 2 + y * η))
variables (first_vs_last : (y-1) * t + η = 5 * η)

-- The goal to prove
theorem first_woman_hours :
  (y - 1) * t + η = 75 := 
by
  sorry

end first_woman_hours_l220_220802


namespace find_x_given_conditions_l220_220694

theorem find_x_given_conditions (x y : ℚ) (h1 : x / y = 12 / 5) (h2 : y = 20) : x = 48 := by 
  sorry

end find_x_given_conditions_l220_220694


namespace product_of_two_numbers_l220_220169

theorem product_of_two_numbers (x y : ℕ) 
  (h1 : y = 15 * x) 
  (h2 : x + y = 400) : 
  x * y = 9375 :=
by
  sorry

end product_of_two_numbers_l220_220169


namespace geometric_sequence_fourth_term_l220_220291

theorem geometric_sequence_fourth_term (x : ℚ) (r : ℚ)
  (h1 : x ≠ 0)
  (h2 : x ≠ -1)
  (h3 : 3 * x + 3 = r * x)
  (h4 : 5 * x + 5 = r * (3 * x + 3)) :
  r^3 * (5 * x + 5) = -125 / 12 :=
by
  sorry

end geometric_sequence_fourth_term_l220_220291


namespace B_pow_101_l220_220281

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]
  ]

theorem B_pow_101 :
  B ^ 101 = ![
    ![0, 0, 1],
    ![1, 0, 0],
    ![0, 1, 0]
  ] :=
  sorry

end B_pow_101_l220_220281


namespace min_value_expression_l220_220104

theorem min_value_expression {x y : ℝ} : 
  2 * x + y - 3 = 0 →
  x + 2 * y - 1 = 0 →
  (5 * x^2 + 8 * x * y + 5 * y^2 - 14 * x - 10 * y + 30) / (4 - x^2 - 10 * x * y - 25 * y^2) ^ (7 / 2) = 5 / 32 :=
by
  sorry

end min_value_expression_l220_220104


namespace area_three_layers_l220_220039

def total_area_rugs : ℝ := 200
def floor_covered_area : ℝ := 140
def exactly_two_layers_area : ℝ := 24

theorem area_three_layers : (2 * (200 - 140 - 24) / 2 = 2 * 18) := 
by admit -- since we're instructed to skip the proof

end area_three_layers_l220_220039


namespace inequality_proof_l220_220613

theorem inequality_proof 
  {a b c : ℝ}
  (ha : 0 ≤ a)
  (hb : 0 ≤ b)
  (hc : 0 ≤ c)
  (h1 : a^2 ≤ b^2 + c^2)
  (h2 : b^2 ≤ c^2 + a^2)
  (h3 : c^2 ≤ a^2 + b^2) :
  (a + b + c) * (a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) ≥ 4 * (a^6 + b^6 + c^6) :=
sorry

end inequality_proof_l220_220613


namespace intersection_eq_l220_220118

def M : Set ℝ := {x | x < 3}
def N : Set ℝ := {x | x^2 - 6*x + 8 < 0}
def intersection : Set ℝ := {x | 2 < x ∧ x < 3}

theorem intersection_eq : M ∩ N = intersection := by
  sorry

end intersection_eq_l220_220118


namespace polygon_with_interior_sum_1260_eq_nonagon_l220_220033

theorem polygon_with_interior_sum_1260_eq_nonagon :
  ∃ n : ℕ, (n-2) * 180 = 1260 ∧ n = 9 := by
  sorry

end polygon_with_interior_sum_1260_eq_nonagon_l220_220033


namespace minimum_value_of_expression_l220_220102

noncomputable def expression (x y : ℝ) : ℝ :=
  (5 * x^2 + 8 * x * y + 5 * y^2 - 14 * x - 10 * y + 30) /
  (4 - x^2 - 10 * x * y - 25 * y^2) ^ (7 / 2)

theorem minimum_value_of_expression : 
  ∃ x y : ℝ, expression x y = 5/32 :=
by
  sorry

end minimum_value_of_expression_l220_220102


namespace find_x_y_sum_of_squares_l220_220681

theorem find_x_y_sum_of_squares :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ (xy + x + y = 47) ∧ (x^2 * y + x * y^2 = 506) ∧ (x^2 + y^2 = 101) :=
by {
  sorry
}

end find_x_y_sum_of_squares_l220_220681


namespace abs_eq_non_pos_2x_plus_4_l220_220783

-- Condition: |2x + 4| = 0
-- Conclusion: x = -2
theorem abs_eq_non_pos_2x_plus_4 (x : ℝ) : (|2 * x + 4| = 0) → x = -2 :=
by
  intro h
  -- Here lies the proof, but we use sorry to indicate the unchecked part.
  sorry

end abs_eq_non_pos_2x_plus_4_l220_220783


namespace power_division_l220_220954

-- Condition given
def sixty_four_is_power_of_eight : Prop := 64 = 8^2

-- Theorem to prove
theorem power_division : sixty_four_is_power_of_eight → 8^{15} / 64^6 = 512 := by
  intro h
  have h1 : 64^6 = (8^2)^6, from by rw [h]
  have h2 : (8^2)^6 = 8^{12}, from pow_mul 8 2 6
  rw [h1, h2]
  have h3 : 8^{15} / 8^{12} = 8^{15 - 12}, from pow_div 8 15 12
  rw [h3]
  have h4 : 8^{15 - 12} = 8^3, from by rw [sub_self_add]
  rw [h4]
  have h5 : 8^3 = 512, from by norm_num
  rw [h5]
  sorry

end power_division_l220_220954


namespace find_interest_rate_of_first_investment_l220_220751

noncomputable def total_interest : ℚ := 73
noncomputable def interest_rate_7_percent : ℚ := 0.07
noncomputable def invested_400 : ℚ := 400
noncomputable def interest_7_percent := invested_400 * interest_rate_7_percent
noncomputable def interest_first_investment := total_interest - interest_7_percent
noncomputable def invested_first : ℚ := invested_400 - 100
noncomputable def interest_first : ℚ := 45  -- calculated as total_interest - interest_7_percent

theorem find_interest_rate_of_first_investment (r : ℚ) :
  interest_first = invested_first * r * 1 → 
  r = 0.15 :=
by
  sorry

end find_interest_rate_of_first_investment_l220_220751


namespace valid_permutations_remainder_l220_220742

def countValidPermutations : Nat :=
  let total := (Finset.range 3).sum (fun j =>
    Nat.choose 3 (j + 2) * Nat.choose 5 j * Nat.choose 7 (j + 3))
  total % 1000

theorem valid_permutations_remainder :
  countValidPermutations = 60 := 
  sorry

end valid_permutations_remainder_l220_220742


namespace max_sum_consecutive_integers_less_360_l220_220330

theorem max_sum_consecutive_integers_less_360 :
  ∃ n : ℤ, n * (n + 1) < 360 ∧ (n + (n + 1)) = 37 :=
by
  sorry

end max_sum_consecutive_integers_less_360_l220_220330


namespace football_game_attendance_l220_220177

theorem football_game_attendance :
  let saturday : ℕ := 80
  let monday : ℕ := saturday - 20
  let friday : ℕ := saturday + monday
  let expected_total_audience : ℕ := 350
  let actual_total_audience : ℕ := expected_total_audience + 40
  let known_attendance : ℕ := saturday + monday + friday
  let wednesday : ℕ := actual_total_audience - known_attendance
  wednesday - monday = 50 :=
by
  let saturday : ℕ := 80
  let monday : ℕ := saturday - 20
  let friday : ℕ := saturday + monday
  let expected_total_audience : ℕ := 350
  let actual_total_audience : ℕ := expected_total_audience + 40
  let known_attendance : ℕ := saturday + monday + friday
  let wednesday : ℕ := actual_total_audience - known_attendance
  show wednesday - monday = 50
  sorry

end football_game_attendance_l220_220177


namespace pow_two_greater_than_square_l220_220158

theorem pow_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2 ^ n > n ^ 2 :=
  sorry

end pow_two_greater_than_square_l220_220158


namespace initial_pencils_sold_l220_220363

theorem initial_pencils_sold (x : ℕ) (P : ℝ)
  (h1 : 1 = 0.9 * (x * P))
  (h2 : 1 = 1.2 * (8.25 * P))
  : x = 11 :=
by sorry

end initial_pencils_sold_l220_220363


namespace original_sticker_price_l220_220149

theorem original_sticker_price (S : ℝ) (h1 : 0.80 * S - 120 = 0.65 * S - 10) : S = 733 := 
by
  sorry

end original_sticker_price_l220_220149


namespace intersection_complement_l220_220450

open Set

theorem intersection_complement (U A B : Set ℕ) (hU : U = {x | x ≤ 6}) (hA : A = {1, 3, 5}) (hB : B = {4, 5, 6}) :
  (U \ A) ∩ B = {4, 6} := by
  sorry

end intersection_complement_l220_220450


namespace find_solution_l220_220395

theorem find_solution (x y z : ℝ) :
  (x * (y^2 + z) = z * (z + x * y)) ∧ 
  (y * (z^2 + x) = x * (x + y * z)) ∧ 
  (z * (x^2 + y) = y * (y + x * z)) → 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end find_solution_l220_220395


namespace cos_beta_value_l220_220270

theorem cos_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2)
    (h1 : Real.sin α = 3/5) (h2 : Real.cos (α + β) = 5/13) : 
    Real.cos β = 56/65 := 
by
  sorry

end cos_beta_value_l220_220270


namespace discount_problem_l220_220795

theorem discount_problem (m : ℝ) (h : (200 * (1 - m / 100)^2 = 162)) : m = 10 :=
sorry

end discount_problem_l220_220795


namespace graph_with_6_vertices_contains_triangle_or_independent_set_l220_220304

def has_triangle_or_independent_set (G : SimpleGraph (Fin 6)) : Prop :=
  ∃ (A B C : Fin 6), G.Adj A B ∧ G.Adj B C ∧ G.Adj C A ∨ 
  ¬G.Adj A B ∧ ¬G.Adj B C ∧ ¬G.Adj C A

theorem graph_with_6_vertices_contains_triangle_or_independent_set (G : SimpleGraph (Fin 6)) :
  has_triangle_or_independent_set G :=
sorry

end graph_with_6_vertices_contains_triangle_or_independent_set_l220_220304


namespace tom_can_go_on_three_rides_l220_220327

def rides_possible (total_tickets : ℕ) (spent_tickets : ℕ) (tickets_per_ride : ℕ) : ℕ :=
  (total_tickets - spent_tickets) / tickets_per_ride

theorem tom_can_go_on_three_rides :
  rides_possible 40 28 4 = 3 :=
by
  -- proof goes here
  sorry

end tom_can_go_on_three_rides_l220_220327


namespace biased_coin_heads_divisible_by_3_l220_220354

open ProbabilityTheory

-- Define the probability of heads for the biased coin
def probability_of_heads : ℝ := 3 / 4

-- Define the number of tosses
def num_of_tosses : ℕ := 60

-- Define the event that number of heads is divisible by 3
def event_heads_div_by_3 (num_heads : ℕ) : Prop := num_heads % 3 = 0

-- Main theorem statement
theorem biased_coin_heads_divisible_by_3 :
  P (λ ω => event_heads_div_by_3 (sum (λ i:nat, if bernoulli probability_of_heads ω i then 1 else 0))) = 1 / 3 :=
sorry

end biased_coin_heads_divisible_by_3_l220_220354


namespace total_profit_l220_220816

theorem total_profit (A_investment : ℝ) (B_investment : ℝ) (C_investment : ℝ) 
                     (A_months : ℝ) (B_months : ℝ) (C_months : ℝ)
                     (C_share : ℝ) (A_profit_percentage : ℝ) : ℝ :=
  let A_capital_months := A_investment * A_months
  let B_capital_months := B_investment * B_months
  let C_capital_months := C_investment * C_months
  let total_capital_months := A_capital_months + B_capital_months + C_capital_months
  let P := (C_share * total_capital_months) / (C_capital_months * (1 - A_profit_percentage))
  P

example : total_profit 6500 8400 10000 6 5 3 1900 0.05 = 24667 := by
  sorry

end total_profit_l220_220816


namespace ratio_of_areas_of_concentric_circles_l220_220042

theorem ratio_of_areas_of_concentric_circles :
  (∀ (r1 r2 : ℝ), 
    r1 > 0 ∧ r2 > 0 ∧ 
    ((60 / 360) * 2 * Real.pi * r1 = (48 / 360) * 2 * Real.pi * r2)) →
    ((Real.pi * r1 ^ 2) / (Real.pi * r2 ^ 2) = (16 / 25)) :=
by
  intro h
  sorry

end ratio_of_areas_of_concentric_circles_l220_220042


namespace discriminant_of_quadratic_l220_220526

theorem discriminant_of_quadratic :
  let a := (5 : ℚ)
  let b := (5 + 1/5 : ℚ)
  let c := (1/5 : ℚ)
  let Δ := b^2 - 4 * a * c
  Δ = 576 / 25 :=
by
  sorry

end discriminant_of_quadratic_l220_220526


namespace no_such_ab_exists_l220_220519

theorem no_such_ab_exists : ¬ ∃ (a b : ℝ), ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 * Real.pi → (a * x + b)^2 - Real.cos x * (a * x + b) < (1 / 4) * (Real.sin x)^2 :=
by
  sorry

end no_such_ab_exists_l220_220519


namespace scientific_notation_to_decimal_l220_220598

theorem scientific_notation_to_decimal :
  5.2 * 10^(-5) = 0.000052 :=
sorry

end scientific_notation_to_decimal_l220_220598


namespace value_of_z_l220_220844

theorem value_of_z (z y : ℝ) (h1 : (12)^3 * z^3 / 432 = y) (h2 : y = 864) : z = 6 :=
by
  sorry

end value_of_z_l220_220844


namespace smallest_positive_integer_b_no_inverse_l220_220092

theorem smallest_positive_integer_b_no_inverse :
  ∃ b : ℕ, b > 0 ∧ gcd b 30 > 1 ∧ gcd b 42 > 1 ∧ b = 6 :=
by
  sorry

end smallest_positive_integer_b_no_inverse_l220_220092


namespace new_car_travel_distance_l220_220220

-- Define the distance traveled by the older car
def distance_older_car : ℝ := 150

-- Define the percentage increase
def percentage_increase : ℝ := 0.30

-- Define the condition for the newer car's travel distance
def distance_newer_car (d_old : ℝ) (perc_inc : ℝ) : ℝ :=
  d_old * (1 + perc_inc)

-- Prove the main statement
theorem new_car_travel_distance :
  distance_newer_car distance_older_car percentage_increase = 195 := by
  -- Skip the proof body as instructed
  sorry

end new_car_travel_distance_l220_220220


namespace complement_union_l220_220547

-- Definitions of sets A and B based on the conditions
def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x ≤ 0}

def B : Set ℝ := {x | x ≥ 1}

-- Theorem to prove the complement of the union of sets A and B within U
theorem complement_union (x : ℝ) : x ∉ (A ∪ B) ↔ (0 < x ∧ x < 1) := by
  sorry

end complement_union_l220_220547


namespace petrol_expenses_l220_220818

-- Definitions based on the conditions stated in the problem
def salary_saved (salary : ℝ) : ℝ := 0.10 * salary
def total_known_expenses : ℝ := 5000 + 1500 + 4500 + 2500 + 3940

-- Main theorem statement that needs to be proved
theorem petrol_expenses (salary : ℝ) (petrol : ℝ) :
  salary_saved salary = 2160 ∧ salary - 2160 = 19440 ∧ 
  5000 + 1500 + 4500 + 2500 + 3940 = total_known_expenses  →
  petrol = 2000 :=
sorry

end petrol_expenses_l220_220818


namespace lcm_4_8_9_10_l220_220964

theorem lcm_4_8_9_10 : Nat.lcm (Nat.lcm 4 8) (Nat.lcm 9 10) = 360 :=
by
  -- Definitions of the numbers (additional definitions from problem conditions)
  let four := 4 
  let eight := 8
  let nine := 9
  let ten := 10
  
  -- Prime factorizations:
  have h4 : Nat.prime_factors four = [2, 2],
    from rfl
  
  have h8 : Nat.prime_factors eight = [2, 2, 2],
    from rfl

  have h9 : Nat.prime_factors nine = [3, 3],
    from rfl

  have h10 : Nat.prime_factors ten = [2, 5],
    from rfl

  -- Least common multiple calculation
  let highest_2 := 2 ^ 3
  let highest_3 := 3 ^ 2
  let highest_5 := 5

  -- Multiply together
  let lcm := highest_2 * highest_3 * highest_5

  show Nat.lcm (Nat.lcm four eight) (Nat.lcm nine ten) = lcm
  sorry

end lcm_4_8_9_10_l220_220964


namespace sum_of_largest_and_smallest_l220_220176

theorem sum_of_largest_and_smallest (a b c : ℕ) (h1 : a = 10) (h2 : b = 11) (h3 : c = 12) :
  a + c = 22 :=
by
  sorry

end sum_of_largest_and_smallest_l220_220176


namespace foreign_exchange_decline_l220_220976

theorem foreign_exchange_decline (x : ℝ) (h1 : 200 * (1 - x)^2 = 98) : 
  200 * (1 - x)^2 = 98 :=
by
  sorry

end foreign_exchange_decline_l220_220976


namespace probability_one_boy_one_girl_l220_220917

-- Define the total number of students (5), the number of boys (3), and the number of girls (2).
def total_students : Nat := 5
def boys : Nat := 3
def girls : Nat := 2

-- Define the probability calculation in Lean.
noncomputable def select_2_students_prob : ℚ :=
  let total_combinations := Nat.choose total_students 2
  let favorable_combinations := Nat.choose boys 1 * Nat.choose girls 1
  favorable_combinations / total_combinations

-- The statement we need to prove is that this probability is 3/5
theorem probability_one_boy_one_girl : select_2_students_prob = 3 / 5 := sorry

end probability_one_boy_one_girl_l220_220917


namespace four_digit_palindromic_perfect_square_l220_220654

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

theorem four_digit_palindromic_perfect_square :
  {n : ℕ | is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n}.card = 1 :=
sorry

end four_digit_palindromic_perfect_square_l220_220654


namespace alice_is_10_years_older_l220_220376

-- Problem definitions
variables (A B : ℕ)

-- Conditions of the problem
def condition1 := A + 5 = 19
def condition2 := A + 6 = 2 * (B + 6)

-- Question to prove
theorem alice_is_10_years_older (h1 : condition1 A) (h2 : condition2 A B) : A - B = 10 := 
by
  sorry

end alice_is_10_years_older_l220_220376


namespace symmetry_implies_value_l220_220113

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 3)
noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := Real.cos (2 * x + φ)

theorem symmetry_implies_value :
  (∀ (x : ℝ), ∃ (k : ℤ), ω * x - Real.pi / 3 = k * Real.pi + Real.pi / 2) →
  (∀ (x : ℝ), ∃ (k : ℤ), 2 * x + φ = k * Real.pi) →
  0 < φ → φ < Real.pi →
  ω = 2 →
  φ = Real.pi / 6 →
  g (Real.pi / 3) φ = -Real.sqrt 3 / 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  exact sorry

end symmetry_implies_value_l220_220113


namespace largest_positive_real_root_l220_220677

theorem largest_positive_real_root (b2 b1 b0 : ℤ) (h2 : |b2| ≤ 3) (h1 : |b1| ≤ 3) (h0 : |b0| ≤ 3) :
  ∃ r : ℝ, (r > 0) ∧ (r^3 + (b2 : ℝ) * r^2 + (b1 : ℝ) * r + (b0 : ℝ) = 0) ∧ 3.5 < r ∧ r < 4.0 :=
sorry

end largest_positive_real_root_l220_220677


namespace largest_triangle_perimeter_maximizes_l220_220370

theorem largest_triangle_perimeter_maximizes 
  (y : ℤ) 
  (h1 : 3 ≤ y) 
  (h2 : y < 16) : 
  (7 + 9 + y) = 31 ↔ y = 15 := 
by 
  sorry

end largest_triangle_perimeter_maximizes_l220_220370


namespace arithmetic_sequence_fifth_term_l220_220186

theorem arithmetic_sequence_fifth_term:
  ∀ (a₁ aₙ : ℕ) (n : ℕ),
    n = 20 → a₁ = 2 → aₙ = 59 →
    ∃ d a₅, d = (59 - 2) / (20 - 1) ∧ a₅ = 2 + (5 - 1) * d ∧ a₅ = 14 :=
by
  sorry

end arithmetic_sequence_fifth_term_l220_220186


namespace longer_diagonal_rhombus_l220_220808

theorem longer_diagonal_rhombus (a b d1 d2 : ℝ) 
  (h1 : a = 35) 
  (h2 : d1 = 42) : 
  d2 = 56 := 
by 
  sorry

end longer_diagonal_rhombus_l220_220808


namespace card_area_after_one_inch_shortening_l220_220151

def initial_length := 5
def initial_width := 7
def new_area_shortened_side_two := 21
def shorter_side_reduction := 2
def longer_side_reduction := 1

theorem card_area_after_one_inch_shortening :
  (initial_length - shorter_side_reduction) * initial_width = new_area_shortened_side_two →
  initial_length * (initial_width - longer_side_reduction) = 30 :=
by
  intro h
  sorry

end card_area_after_one_inch_shortening_l220_220151


namespace value_of_a7_minus_a8_l220_220895

variable {a : ℕ → ℤ} (d a₁ : ℤ)

-- Definition that this is an arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℤ) (a₁ d : ℤ) : Prop :=
  ∀ n, a n = a₁ + (n - 1) * d

-- Given condition
def condition (a : ℕ → ℤ) : Prop :=
  a 2 + a 6 + a 8 + a 10 = 80

-- The goal to prove
theorem value_of_a7_minus_a8 (a : ℕ → ℤ) (h_arith : is_arithmetic_seq a a₁ d)
  (h_cond : condition a) : a 7 - a 8 = 8 :=
sorry

end value_of_a7_minus_a8_l220_220895


namespace intersection_of_sets_l220_220265

open Set Real

theorem intersection_of_sets :
  let M := {x : ℝ | x ≤ 4}
  let N := {x : ℝ | x > 0}
  M ∩ N = {x : ℝ | 0 < x ∧ x ≤ 4} :=
by
  sorry

end intersection_of_sets_l220_220265


namespace adjacent_zero_point_range_l220_220404

def f (x : ℝ) : ℝ := x - 1
def g (x : ℝ) (a : ℝ) : ℝ := x^2 - a*x - a + 3

theorem adjacent_zero_point_range (a : ℝ) :
  (∀ β, (∃ x, g x a = 0) → (|1 - β| ≤ 1 → (∃ x, f x = 0 → |x - β| ≤ 1))) →
  (2 ≤ a ∧ a ≤ 7 / 3) :=
sorry

end adjacent_zero_point_range_l220_220404


namespace power_mod_remainder_l220_220781

theorem power_mod_remainder (a b : ℕ) (h1 : a = 3) (h2 : b = 167) :
  (3^167) % 11 = 9 := by
  sorry

end power_mod_remainder_l220_220781


namespace parabola_focus_l220_220250

theorem parabola_focus (f : ℝ) :
  (∀ x : ℝ, 2*x^2 = x^2 + (2*x^2 - f)^2 - (2*x^2 - -f)^2) →
  f = -1/8 :=
by sorry

end parabola_focus_l220_220250


namespace infinite_sqrt_solution_l220_220244

noncomputable def infinite_sqrt (x : ℝ) : ℝ := Real.sqrt (20 + x)

theorem infinite_sqrt_solution : 
  ∃ x : ℝ, infinite_sqrt x = x ∧ x ≥ 0 ∧ x = 5 :=
by
  sorry

end infinite_sqrt_solution_l220_220244


namespace smallest_possible_N_l220_220144

theorem smallest_possible_N {p q r s t : ℕ} (hp: 0 < p) (hq: 0 < q) (hr: 0 < r) (hs: 0 < s) (ht: 0 < t) 
  (sum_eq: p + q + r + s + t = 3015) :
  ∃ N, N = max (p + q) (max (q + r) (max (r + s) (s + t))) ∧ N = 1508 := 
sorry

end smallest_possible_N_l220_220144


namespace determine_N_l220_220090

theorem determine_N (N : ℕ) :
    995 + 997 + 999 + 1001 + 1003 = 5005 - N → N = 5 := 
by 
  sorry

end determine_N_l220_220090


namespace ground_beef_sold_ratio_l220_220454

variable (beef_sold_Thursday : ℕ) (beef_sold_Saturday : ℕ) (avg_sold_per_day : ℕ) (days : ℕ)

theorem ground_beef_sold_ratio (h₁ : beef_sold_Thursday = 210)
                             (h₂ : beef_sold_Saturday = 150)
                             (h₃ : avg_sold_per_day = 260)
                             (h₄ : days = 3) :
  let total_sold := avg_sold_per_day * days
  let beef_sold_Friday := total_sold - beef_sold_Thursday - beef_sold_Saturday
  (beef_sold_Friday : ℕ) / (beef_sold_Thursday : ℕ) = 2 := by
  sorry

end ground_beef_sold_ratio_l220_220454


namespace neg_two_is_negative_rational_l220_220351

theorem neg_two_is_negative_rational : 
  (-2 : ℚ) < 0 ∧ ∃ (r : ℚ), r = -2 := 
by
  sorry

end neg_two_is_negative_rational_l220_220351


namespace negation_of_forall_ge_2_l220_220715

theorem negation_of_forall_ge_2 :
  (¬ ∀ x : ℝ, x ≥ 2) = (∃ x₀ : ℝ, x₀ < 2) :=
sorry

end negation_of_forall_ge_2_l220_220715


namespace raul_money_left_l220_220015

theorem raul_money_left (initial_money : ℕ) (cost_per_comic : ℕ) (number_of_comics : ℕ) (money_left : ℕ)
  (h1 : initial_money = 87)
  (h2 : cost_per_comic = 4)
  (h3 : number_of_comics = 8)
  (h4 : money_left = initial_money - (number_of_comics * cost_per_comic)) :
  money_left = 55 :=
by 
  rw [h1, h2, h3] at h4
  exact h4

end raul_money_left_l220_220015


namespace find_roots_of_star_eq_l220_220386

def star (a b : ℝ) : ℝ := a^2 - b^2

theorem find_roots_of_star_eq :
  (star (star 2 3) x = 9) ↔ (x = 4 ∨ x = -4) :=
by
  sorry

end find_roots_of_star_eq_l220_220386


namespace projection_inequality_l220_220534

-- Define the problem with given Cartesian coordinate system, finite set of points in space, and their orthogonal projections
variable (O_xyz : Type) -- Cartesian coordinate system
variable (S : Finset O_xyz) -- finite set of points in space
variable (S_x S_y S_z : Finset O_xyz) -- sets of orthogonal projections onto the planes

-- Define the orthogonal projections (left as a comment here since detailed implementation is not specified)
-- (In Lean, actual definitions of orthogonal projections would follow mathematical and geometric definitions)

-- State the theorem to be proved
theorem projection_inequality :
  (Finset.card S) ^ 2 ≤ (Finset.card S_x) * (Finset.card S_y) * (Finset.card S_z) := 
sorry

end projection_inequality_l220_220534


namespace eight_pow_15_div_sixtyfour_pow_6_l220_220943

theorem eight_pow_15_div_sixtyfour_pow_6 :
  8^15 / 64^6 = 512 := by
  sorry

end eight_pow_15_div_sixtyfour_pow_6_l220_220943


namespace blue_line_length_correct_l220_220840

def white_line_length : ℝ := 7.67
def difference_in_length : ℝ := 4.33
def blue_line_length : ℝ := 3.34

theorem blue_line_length_correct :
  white_line_length - difference_in_length = blue_line_length :=
by
  sorry

end blue_line_length_correct_l220_220840


namespace intersection_A_B_l220_220264

-- define the set A
def A : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ 3 * x - y = 7 }

-- define the set B
def B : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ 2 * x + y = 3 }

-- Prove the intersection
theorem intersection_A_B :
  A ∩ B = { (2, -1) } :=
by
  -- We will insert the proof here
  sorry

end intersection_A_B_l220_220264


namespace trajectory_of_circle_center_is_ellipse_l220_220402

theorem trajectory_of_circle_center_is_ellipse 
    (a b : ℝ) (θ : ℝ) 
    (h1 : a ≠ b)
    (h2 : 0 < a)
    (h3 : 0 < b)
    : ∃ (x y : ℝ), 
    (x, y) = (a * Real.cos θ, b * Real.sin θ) ∧ 
    (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end trajectory_of_circle_center_is_ellipse_l220_220402


namespace selling_price_of_article_l220_220378

theorem selling_price_of_article (CP : ℝ) (L_percent : ℝ) (SP : ℝ) 
  (h1 : CP = 600) 
  (h2 : L_percent = 50) 
  : SP = 300 := 
by
  sorry

end selling_price_of_article_l220_220378


namespace hancho_height_l220_220120

theorem hancho_height (Hansol_height : ℝ) (h1 : Hansol_height = 134.5) (ratio : ℝ) (h2 : ratio = 1.06) :
  Hansol_height * ratio = 142.57 := by
  sorry

end hancho_height_l220_220120


namespace middle_school_mentoring_l220_220130

theorem middle_school_mentoring (s n : ℕ) (h1 : s ≠ 0) (h2 : n ≠ 0) 
  (h3 : (n : ℚ) / 3 = (2 : ℚ) * (s : ℚ) / 5) : 
  (n / 3 + 2 * s / 5) / (n + s) = 4 / 11 := by
  sorry

end middle_school_mentoring_l220_220130


namespace periodic_even_function_value_l220_220427

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := (x + 1) * (x - a)

-- Conditions: 
-- 1. f(x) is even 
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- 2. f(x) is periodic with period 6
def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f x = f (x + p)

-- Main theorem
theorem periodic_even_function_value 
  (a : ℝ) 
  (f_def : ∀ x, -3 ≤ x ∧ x ≤ 3 → f x a = (x + 1) * (x - a))
  (h_even : is_even_function (f · a))
  (h_periodic : is_periodic_function (f · a) 6) : 
  f (-6) a = -1 := 
sorry

end periodic_even_function_value_l220_220427


namespace distance_between_A_B_is_16_l220_220896

-- The given conditions are translated as definitions
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 4

def curve (t : ℝ) : ℝ × ℝ := (t^2, t^3)

-- The theorem stating the proof problem
theorem distance_between_A_B_is_16 :
  let A : ℝ × ℝ := (4, 8)
  let B : ℝ × ℝ := (4, -8)
  let d : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  d = 16 :=
by
  sorry

end distance_between_A_B_is_16_l220_220896


namespace Mark_jump_rope_hours_l220_220577

theorem Mark_jump_rope_hours 
  (record : ℕ) 
  (jump_rate : ℕ) 
  (seconds_per_hour : ℕ) 
  (h_record : record = 54000) 
  (h_jump_rate : jump_rate = 3) 
  (h_seconds_per_hour : seconds_per_hour = 3600) 
  : (record / jump_rate) / seconds_per_hour = 5 := 
by
  sorry

end Mark_jump_rope_hours_l220_220577


namespace polygon_problem_l220_220111

theorem polygon_problem
  (sum_interior_angles : ℕ → ℝ)
  (sum_exterior_angles : ℝ)
  (condition : ∀ n, sum_interior_angles n = (3 * sum_exterior_angles) - 180) :
  (∃ n : ℕ, sum_interior_angles n = 180 * (n - 2) ∧ n = 7) ∧
  (∃ n : ℕ, n = 7 → (n * (n - 3) / 2) = 14) :=
by
  sorry

end polygon_problem_l220_220111


namespace least_number_to_divisible_l220_220346

theorem least_number_to_divisible (x : ℕ) : 
  (∃ x, (1049 + x) % 25 = 0) ∧ (∀ y, y < x → (1049 + y) % 25 ≠ 0) ↔ x = 1 :=
by
  sorry

end least_number_to_divisible_l220_220346


namespace value_of_b_l220_220026

theorem value_of_b (b x : ℝ) (h1 : 2 * x + 7 = 3) (h2 : b * x - 10 = -2) : b = -4 :=
by
  sorry

end value_of_b_l220_220026


namespace avg_page_count_per_essay_l220_220075

-- Definitions based on conditions
def students : ℕ := 15
def first_group_students : ℕ := 5
def second_group_students : ℕ := 5
def third_group_students : ℕ := 5
def pages_first_group : ℕ := 2
def pages_second_group : ℕ := 3
def pages_third_group : ℕ := 1
def total_pages := first_group_students * pages_first_group +
                   second_group_students * pages_second_group +
                   third_group_students * pages_third_group

-- Statement to prove
theorem avg_page_count_per_essay :
  total_pages / students = 2 :=
by
  sorry

end avg_page_count_per_essay_l220_220075


namespace total_family_members_l220_220903

variable (members_father_side : Nat) (percent_incr : Nat)
variable (members_mother_side := members_father_side + (members_father_side * percent_incr / 100))
variable (total_members := members_father_side + members_mother_side)

theorem total_family_members 
  (h1 : members_father_side = 10) 
  (h2 : percent_incr = 30) :
  total_members = 23 :=
by
  sorry

end total_family_members_l220_220903


namespace range_of_a_l220_220543

-- Define the negation of the original proposition as a function
def negated_prop (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + 4 * x + a > 0

-- State the theorem to be proven
theorem range_of_a (a : ℝ) (h : ¬∃ x : ℝ, a * x^2 + 4 * x + a ≤ 0) : a > 2 :=
  by
  -- Using the assumption to conclude the negated proposition holds
  let h_neg : negated_prop a := sorry
  
  -- Prove the range of a based on h_neg
  sorry

end range_of_a_l220_220543


namespace gcd_13924_27018_l220_220396

theorem gcd_13924_27018 : Int.gcd 13924 27018 = 2 := 
  by
    sorry

end gcd_13924_27018_l220_220396


namespace complex_square_eq_l220_220700

variables {a b : ℝ} {i : ℂ}

theorem complex_square_eq :
  a + i = 2 - b * i → (a + b * i) ^ 2 = 3 - 4 * i :=
by sorry

end complex_square_eq_l220_220700


namespace good_function_count_l220_220108

noncomputable def num_good_functions (n : ℕ) : ℕ :=
  if n < 2 then 0 else
    n * Nat.totient n

theorem good_function_count (n : ℕ) (h : n ≥ 2) :
  ∃ (f : ℤ → Fin (n + 1)), 
  (∀ k, 1 ≤ k ∧ k ≤ n - 1 → ∃ j, ∀ m, (f (m + j) : ℤ) ≡ (f (m + k) - f m : ℤ) [ZMOD (n + 1)]) → 
  num_good_functions n = n * Nat.totient n :=
sorry

end good_function_count_l220_220108


namespace ratio_of_areas_of_concentric_circles_l220_220046

theorem ratio_of_areas_of_concentric_circles
  (C1 C2 : ℝ) -- circumferences of the smaller and larger circle
  (h : (1 / 6) * C1 = (2 / 15) * C2) -- condition given: 60-degree arc on the smaller circle equals 48-degree arc on the larger circle
  : (C1 / C2)^2 = (16 / 25) := by
  sorry

end ratio_of_areas_of_concentric_circles_l220_220046


namespace color_divisors_with_conditions_l220_220571

/-- Define the primes, product of the first 100 primes, and set S -/
def first_100_primes : List Nat := sorry -- Assume we have the list of first 100 primes
def product_of_first_100_primes : Nat := first_100_primes.foldr (· * ·) 1
def S := {d : Nat | d > 1 ∧ ∃ m, product_of_first_100_primes = m * d}

/-- Statement of the problem in Lean 4 -/
theorem color_divisors_with_conditions :
  (∃ (k : Nat), (∀ (coloring : S → Fin k), 
    (∀ s1 s2 s3 : S, (s1 * s2 * s3 = product_of_first_100_primes) → (coloring s1 = coloring s2 ∨ coloring s1 = coloring s3 ∨ coloring s2 = coloring s3)) ∧
    (∀ c : Fin k, ∃ s : S, coloring s = c))) ↔ k = 100 := 
by
  sorry

end color_divisors_with_conditions_l220_220571


namespace no_purchase_count_l220_220352

def total_people : ℕ := 15
def people_bought_tvs : ℕ := 9
def people_bought_computers : ℕ := 7
def people_bought_both : ℕ := 3

theorem no_purchase_count : total_people - (people_bought_tvs - people_bought_both) - (people_bought_computers - people_bought_both) - people_bought_both = 2 := by
  sorry

end no_purchase_count_l220_220352


namespace max_value_expression_l220_220391

noncomputable def factorize_15000 := 2^3 * 3 * 5^4

theorem max_value_expression (x y : ℕ) (h1 : 6 * x^2 - 5 * x * y + y^2 = 0) (h2 : x ∣ factorize_15000) : 
  2 * x + 3 * y ≤ 60000 := sorry

end max_value_expression_l220_220391


namespace number_of_sections_l220_220609

def total_seats : ℕ := 270
def seats_per_section : ℕ := 30

theorem number_of_sections : total_seats / seats_per_section = 9 := 
by sorry

end number_of_sections_l220_220609


namespace find_n_of_sum_of_evens_l220_220629

-- Definitions based on conditions in part (a)
def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_of_evens_up_to (n : ℕ) : ℕ :=
  let k := (n - 1) / 2
  (k / 2) * (2 + (n - 1))

-- Problem statement in Lean
theorem find_n_of_sum_of_evens : 
  ∃ n : ℕ, is_odd n ∧ sum_of_evens_up_to n = 81 * 82 ∧ n = 163 :=
by
  sorry

end find_n_of_sum_of_evens_l220_220629


namespace caesar_cipher_WIN_shift_4_l220_220593

def alphabet := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def caesar_cipher (shift : ℕ) (msg : String) : String :=
  String.mk (msg.data.map (λ c => alphabet.get? ((alphabet.indexOf c + shift) % alphabet.length)).iget)

theorem caesar_cipher_WIN_shift_4 :
  caesar_cipher 4 "WIN" = "AMR" :=
by { unfold caesar_cipher, norm_num, sorry }

end caesar_cipher_WIN_shift_4_l220_220593


namespace exponent_multiplication_l220_220401

theorem exponent_multiplication (m n : ℕ) (h : m + n = 3) : 2^m * 2^n = 8 := 
by
  sorry

end exponent_multiplication_l220_220401


namespace power_division_l220_220956

-- Condition given
def sixty_four_is_power_of_eight : Prop := 64 = 8^2

-- Theorem to prove
theorem power_division : sixty_four_is_power_of_eight → 8^{15} / 64^6 = 512 := by
  intro h
  have h1 : 64^6 = (8^2)^6, from by rw [h]
  have h2 : (8^2)^6 = 8^{12}, from pow_mul 8 2 6
  rw [h1, h2]
  have h3 : 8^{15} / 8^{12} = 8^{15 - 12}, from pow_div 8 15 12
  rw [h3]
  have h4 : 8^{15 - 12} = 8^3, from by rw [sub_self_add]
  rw [h4]
  have h5 : 8^3 = 512, from by norm_num
  rw [h5]
  sorry

end power_division_l220_220956


namespace num_four_digit_palindromic_squares_l220_220643

theorem num_four_digit_palindromic_squares :
  let is_palindrome (n : ℕ) : Prop := (n.toString = n.toString.reverse)
  let four_digit_squares := {n : ℕ | 1000 ≤ n ∧ n ≤ 9999 ∧ ∃ m : ℕ, n = m * m ∧ 32 ≤ m ∧ m ≤ 99}
  let palindromic_four_digit_squares := {n : ℕ | n ∈ four_digit_squares ∧ is_palindrome n}
  palindromic_four_digit_squares.card = 1 :=
by
  sorry

end num_four_digit_palindromic_squares_l220_220643


namespace inequality_imply_positive_a_l220_220262

theorem inequality_imply_positive_a 
  (a b d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (h_d_pos : d > 0) 
  (h : a / b > -3 / (2 * d)) : a > 0 :=
sorry

end inequality_imply_positive_a_l220_220262


namespace no_digit_B_divisible_by_4_l220_220276

theorem no_digit_B_divisible_by_4 : 
  ∀ B : ℕ, B < 10 → ¬ (8 * 1000000 + B * 100000 + 4 * 10000 + 6 * 1000 + 3 * 100 + 5 * 10 + 1) % 4 = 0 :=
by
  intros B hB_lt_10
  sorry

end no_digit_B_divisible_by_4_l220_220276


namespace math_problem_l220_220394

noncomputable def is_solution (x : ℝ) : Prop :=
  1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 12

theorem math_problem :
  (is_solution ((7 + Real.sqrt 153) / 2)) ∧ (is_solution ((7 - Real.sqrt 153) / 2)) := 
by
  sorry

end math_problem_l220_220394


namespace find_smallest_d_l220_220766

noncomputable def smallest_possible_d (c d : ℕ) : ℕ :=
  if c - d = 8 ∧ Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16 then d else 0

-- Proving the smallest possible value of d given the conditions
theorem find_smallest_d :
  ∀ c d : ℕ, (0 < c) → (0 < d) → (c - d = 8) → 
  Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16 → d = 4 :=
by
  sorry

end find_smallest_d_l220_220766


namespace bob_clean_time_l220_220377

-- Definitions for the problem conditions
def alice_time : ℕ := 30
def bob_time := (1 / 3 : ℚ) * alice_time

-- The proof problem statement (only) in Lean 4
theorem bob_clean_time : bob_time = 10 := by
  sorry

end bob_clean_time_l220_220377


namespace function_properties_l220_220412

theorem function_properties (k : ℝ) :
  (∀ x : ℝ, x^2 + k * x + 25 ≠ 0 ∧ x^2 - (k - 6) * x + 16 ≠ 0) → 
  (-2 < k ∧ k < 10) :=
by
  intros h
  sorry

end function_properties_l220_220412


namespace decryption_ease_comparison_l220_220094

def unique_letters_of_thermometer : Finset Char := {'т', 'е', 'р', 'м', 'о'}
def unique_letters_of_remont : Finset Char := {'р', 'е', 'м', 'о', 'н', 'т'}
def easier_to_decrypt : Prop :=
  unique_letters_of_remont.card > unique_letters_of_thermometer.card

theorem decryption_ease_comparison : easier_to_decrypt :=
by
  -- We need to prove that |unique_letters_of_remont| > |unique_letters_of_thermometer|
  sorry

end decryption_ease_comparison_l220_220094


namespace probability_of_a_plus_b_gt_5_l220_220460

noncomputable def all_events : Finset (ℕ × ℕ) := 
  { (1, 2), (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4) }

noncomputable def successful_events : Finset (ℕ × ℕ) :=
  { (2, 4), (3, 3), (3, 4) }

theorem probability_of_a_plus_b_gt_5 : 
  (successful_events.card : ℚ) / (all_events.card : ℚ) = 1 / 3 := by
  sorry

end probability_of_a_plus_b_gt_5_l220_220460


namespace sara_ate_16_apples_l220_220223

theorem sara_ate_16_apples (S : ℕ) (h1 : ∃ (A : ℕ), A = 4 * S ∧ S + A = 80) : S = 16 :=
by
  obtain ⟨A, h2, h3⟩ := h1
  sorry

end sara_ate_16_apples_l220_220223


namespace problem_statement_l220_220676

noncomputable def a := Real.sqrt 3 + Real.sqrt 2
noncomputable def b := Real.sqrt 3 - Real.sqrt 2
noncomputable def expression := a^(2 * Real.log (Real.sqrt 5) / Real.log b)

theorem problem_statement : expression = 1 / 5 := by
  sorry

end problem_statement_l220_220676


namespace count_four_digit_palindrome_squares_l220_220639

-- Define what a palindrome number is
def is_palindrome (n : ℕ) : Prop :=
  let digits := to_string n
  digits = digits.reverse

-- Define four-digit perfect squares
def is_four_digit_perfect_square (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (∃ m : ℕ, n = m * m)

-- The theorem we want to prove
theorem count_four_digit_palindrome_squares :
  ∃! n, is_four_digit_perfect_square n ∧ is_palindrome n := {
  exists.intro 8100 (and.intro ⟨le_of_eq 8100, le_of_eq 8100, exists.intro 90 rfl⟩ rfl)
    (λ m h, and.elim h (λ h1 h2, 
      and.elim h1 (λ h1a h1b,
        exists.elim h1b (λ k hk,
          have hk1 := congr_arg (λ x, x * x) (eq.trans hk.symm (congr_arg has_sub m 90)),
          let h := (sub_eq_zero.mp hk1.symm.d_symm_trans)
          have m.char_zero := hk1 darge introspec Number.to_one_palindrome,
          rfl whether)))
      sorry -- the actual proof will go here
}

end count_four_digit_palindrome_squares_l220_220639


namespace coordinates_of_point_P_l220_220915

theorem coordinates_of_point_P :
  ∀ (P : ℝ × ℝ), (P.1, P.2) = -1 ∧ (P.2 = -Real.sqrt 3) :=
by
  sorry

end coordinates_of_point_P_l220_220915


namespace project_completion_rate_l220_220081

variables {a b c d e : ℕ} {f g : ℚ}  -- Assuming efficiency ratings can be represented by rational numbers.

theorem project_completion_rate (h : (a * f / c) = b / c) 
: (d * g / e) = bdge / ca := 
sorry

end project_completion_rate_l220_220081


namespace weight_of_five_single_beds_l220_220608

-- Define the problem conditions and the goal
theorem weight_of_five_single_beds :
  ∃ S D : ℝ, (2 * S + 4 * D = 100) ∧ (D = S + 10) → (5 * S = 50) :=
by
  sorry

end weight_of_five_single_beds_l220_220608


namespace price_of_table_l220_220985

-- Given the conditions:
def chair_table_eq1 (C T : ℝ) : Prop := 2 * C + T = 0.6 * (C + 2 * T)
def chair_table_eq2 (C T : ℝ) : Prop := C + T = 72

-- Prove that the price of one table is $63
theorem price_of_table (C T : ℝ) (h1 : chair_table_eq1 C T) (h2 : chair_table_eq2 C T) : T = 63 := by
  sorry

end price_of_table_l220_220985


namespace inequality_holds_l220_220584

theorem inequality_holds (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) * (4 * x + y + 2 * z) * (2 * x + y + 8 * z) ≥ (375 / 2) * x * y * z :=
by
  sorry

end inequality_holds_l220_220584


namespace sum_of_c_n_l220_220856

-- Define the sequence {b_n}
def b : ℕ → ℕ
| 0       => 1
| (n + 1) => 2 * b n + 3

-- Define the sequence {a_n}
def a (n : ℕ) : ℕ := 2 * n + 1

-- Define the sequence {c_n}
def c (n : ℕ) : ℚ := (a n) / (b n + 3)

-- Define the sum of the first n terms of {c_n}
def T (n : ℕ) : ℚ := (Finset.range n).sum (λ i => c i)

-- Theorem to prove
theorem sum_of_c_n : ∀ (n : ℕ), T n = (3 / 2 : ℚ) - ((2 * n + 3) / 2^(n + 1)) :=
by
  sorry

end sum_of_c_n_l220_220856


namespace arithmetic_sequence_seventh_term_l220_220474

theorem arithmetic_sequence_seventh_term (a d : ℚ) 
  (h1 : a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = 15)
  (h2 : a + 5 * d = 8) :
  a + 6 * d = 29 / 3 := 
sorry

end arithmetic_sequence_seventh_term_l220_220474


namespace one_fourth_of_6_8_is_fraction_l220_220100

theorem one_fourth_of_6_8_is_fraction :
  (6.8 / 4 : ℚ) = 17 / 10 :=
sorry

end one_fourth_of_6_8_is_fraction_l220_220100


namespace sqrt_log_equality_l220_220710

noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem sqrt_log_equality {x y : ℝ} (hx : x > 0) (hy : y > 0) :
    Real.sqrt (log4 x + 2 * log2 y) = Real.sqrt (log2 (x * y^2)) / Real.sqrt 2 :=
sorry

end sqrt_log_equality_l220_220710


namespace charcoal_drawings_count_l220_220611

/-- Thomas' drawings problem
  Thomas has 25 drawings in total.
  14 drawings with colored pencils.
  7 drawings with blending markers.
  The rest drawings are made with charcoal.
  We assert that the number of charcoal drawings is 4.
-/
theorem charcoal_drawings_count 
  (total_drawings : ℕ) 
  (colored_pencil_drawings : ℕ) 
  (marker_drawings : ℕ) :
  total_drawings = 25 →
  colored_pencil_drawings = 14 →
  marker_drawings = 7 →
  total_drawings - (colored_pencil_drawings + marker_drawings) = 4 := 
  by
    sorry

end charcoal_drawings_count_l220_220611


namespace photograph_perimeter_is_23_l220_220222

noncomputable def photograph_perimeter (w h m : ℝ) : ℝ :=
if (w + 4) * (h + 4) = m ∧ (w + 8) * (h + 8) = m + 94 then 2 * (w + h) else 0

theorem photograph_perimeter_is_23 (w h m : ℝ) 
    (h₁ : (w + 4) * (h + 4) = m) 
    (h₂ : (w + 8) * (h + 8) = m + 94) : 
    photograph_perimeter w h m = 23 := 
by 
  sorry

end photograph_perimeter_is_23_l220_220222


namespace scorpion_additional_millipedes_l220_220070

theorem scorpion_additional_millipedes :
  let total_segments := 800 in
  let segments_one_millipede := 60 in
  let segments_two_millipedes := 2 * (2 * segments_one_millipede) in
  let total_eaten := segments_one_millipede + segments_two_millipedes in
  let remaining_segments := total_segments - total_eaten in
  let segments_per_millipede := 50 in
  remaining_segments / segments_per_millipede = 10 :=
by {
  sorry
}

end scorpion_additional_millipedes_l220_220070


namespace japanese_turtle_crane_problem_l220_220312

theorem japanese_turtle_crane_problem (x y : ℕ) (h1 : x + y = 35) (h2 : 2 * x + 4 * y = 94) : x + y = 35 ∧ 2 * x + 4 * y = 94 :=
by
  sorry

end japanese_turtle_crane_problem_l220_220312


namespace revenue_effect_l220_220785

noncomputable def price_increase_factor : ℝ := 1.425
noncomputable def sales_decrease_factor : ℝ := 0.627

theorem revenue_effect (P Q R_new : ℝ) (h_price_increase : P ≠ 0) (h_sales_decrease : Q ≠ 0) :
  R_new = (P * price_increase_factor) * (Q * sales_decrease_factor) →
  ((R_new - P * Q) / (P * Q)) * 100 = -10.6825 :=
by
  sorry

end revenue_effect_l220_220785


namespace geometric_sequence_x_l220_220261

theorem geometric_sequence_x (x : ℝ) (h : 1 * 9 = x^2) : x = 3 ∨ x = -3 :=
by
  sorry

end geometric_sequence_x_l220_220261


namespace math_problem_l220_220711

theorem math_problem (x y : ℝ) 
  (h1 : 1/5 + x + y = 1) 
  (h2 : 1/5 * 1 + 2 * x + 3 * y = 11/5) : 
  (x = 2/5) ∧ 
  (y = 2/5) ∧ 
  (1/5 + x = 3/5) ∧ 
  ((1 - 11/5)^2 * (1/5) + (2 - 11/5)^2 * (2/5) + (3 - 11/5)^2 * (2/5) = 14/25) :=
by {
  sorry
}

end math_problem_l220_220711


namespace domain_f_l220_220315

open Real Set

noncomputable def f (x : ℝ) : ℝ := log (x + 1) + (x - 2) ^ 0

theorem domain_f :
  (∃ x : ℝ, f x = f x) ↔ (∀ x, (x > -1 ∧ x ≠ 2) ↔ (x ∈ Ioo (-1 : ℝ) 2 ∨ x ∈ Ioi 2)) :=
by
  sorry

end domain_f_l220_220315


namespace shrimp_appetizer_cost_l220_220619

-- Define the conditions
def shrimp_per_guest : ℕ := 5
def number_of_guests : ℕ := 40
def cost_per_pound : ℕ := 17
def shrimp_per_pound : ℕ := 20

-- Define the proof statement
theorem shrimp_appetizer_cost : 
  (shrimp_per_guest * number_of_guests / shrimp_per_pound) * cost_per_pound = 170 := 
by
  sorry

end shrimp_appetizer_cost_l220_220619


namespace tangent_line_at_zero_range_of_a_monotonic_decreasing_exp_sum_sin_lt_two_l220_220116

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * sin x + log (1 - x)

-- Problem 1: Prove the equation of the tangent line at x = 0 is y = 0 when a = 1.
theorem tangent_line_at_zero : 
  f 1 0 = 0 ∧ deriv (f 1) 0 = 0 := 
sorry

-- Problem 2: Prove the range of values for a where f(x) is monotonically decreasing on [0, 1) is (-∞, 1]
theorem range_of_a_monotonic_decreasing : 
  ∀ a, (∀ x ∈ Iio (1:ℝ), deriv (f a) x ≤ 0) ↔ a ≤ 1 :=
sorry

-- Problem 3: Prove e^(sum of the sines) is less than 2
theorem exp_sum_sin_lt_two (n : ℕ) (hn : 0 < n) :
  ∑ k in Finset.range n, sin (1 / ((k + 1)^2 : ℝ)) < log 2 :=
sorry


end tangent_line_at_zero_range_of_a_monotonic_decreasing_exp_sum_sin_lt_two_l220_220116


namespace standard_deviation_of_distribution_l220_220924

theorem standard_deviation_of_distribution (μ σ : ℝ) 
    (h₁ : μ = 15) (h₂ : μ - 2 * σ = 12) : σ = 1.5 := by
  sorry

end standard_deviation_of_distribution_l220_220924


namespace players_odd_sum_probability_l220_220389

theorem players_odd_sum_probability :
  let tiles := (1:ℕ) :: (2:ℕ) :: (3:ℕ) :: (4:ℕ) :: (5:ℕ) :: (6:ℕ) :: (7:ℕ) :: (8:ℕ) :: (9:ℕ) :: (10:ℕ) :: (11:ℕ) :: []
  let m := 1
  let n := 26
  m + n = 27 :=
by
  sorry

end players_odd_sum_probability_l220_220389


namespace find_number_l220_220973

theorem find_number (x : ℝ) (h : 0.05 * x = 12.75) : x = 255 :=
by
  sorry

end find_number_l220_220973


namespace number_of_negative_x_values_l220_220106

theorem number_of_negative_x_values : 
  (∃ (n : ℕ), ∀ (x : ℤ), x = n^2 - 196 ∧ x < 0) ∧ (n ≤ 13) :=
by 
  -- To formalize our problem we need quantifiers, inequalities and integer properties.
  sorry

end number_of_negative_x_values_l220_220106


namespace allison_total_supply_items_is_28_l220_220988

/-- Define the number of glue sticks Marie bought --/
def marie_glue_sticks : ℕ := 15
/-- Define the number of packs of construction paper Marie bought --/
def marie_construction_paper_packs : ℕ := 30
/-- Define the number of glue sticks Allison bought --/
def allison_glue_sticks : ℕ := marie_glue_sticks + 8
/-- Define the number of packs of construction paper Allison bought --/
def allison_construction_paper_packs : ℕ := marie_construction_paper_packs / 6
/-- Calculation of the total number of craft supply items Allison bought --/
def allison_total_supply_items : ℕ := allison_glue_sticks + allison_construction_paper_packs

/-- Prove that the total number of craft supply items Allison bought is equal to 28. --/
theorem allison_total_supply_items_is_28 : allison_total_supply_items = 28 :=
by sorry

end allison_total_supply_items_is_28_l220_220988


namespace prob_less_than_8_prob_at_least_7_l220_220410

def prob_9_or_above : ℝ := 0.56
def prob_8 : ℝ := 0.22
def prob_7 : ℝ := 0.12

theorem prob_less_than_8 : prob_7 + (1 - prob_9_or_above - prob_8) = 0.22 := 
sorry

theorem prob_at_least_7 : prob_9_or_above + prob_8 + prob_7 = 0.9 := 
sorry

end prob_less_than_8_prob_at_least_7_l220_220410


namespace triangle_angle_tangent_condition_l220_220565

theorem triangle_angle_tangent_condition
  (A B C : ℝ)
  (h1 : A + C = 2 * B)
  (h2 : Real.tan A * Real.tan C = 2 + Real.sqrt 3) :
  (A = Real.pi / 4 ∧ B = Real.pi / 3 ∧ C = 5 * Real.pi / 12) ∨
  (A = 5 * Real.pi / 12 ∧ B = Real.pi / 3 ∧ C = Real.pi / 4) :=
  sorry

end triangle_angle_tangent_condition_l220_220565


namespace find_a4_l220_220449

variable {α : Type*} [Field α] [Inhabited α]

-- Definitions of the geometric sequence conditions
def geometric_sequence_condition1 (a₁ q : α) : Prop :=
  a₁ * (1 + q) = -1

def geometric_sequence_condition2 (a₁ q : α) : Prop :=
  a₁ * (1 - q^2) = -3

-- Definition of the geometric sequence
def geometric_sequence (a₁ q : α) (n : ℕ) : α :=
  a₁ * q^n

-- The theorem to be proven
theorem find_a4 (a₁ q : α) (h₁ : geometric_sequence_condition1 a₁ q) (h₂ : geometric_sequence_condition2 a₁ q) :
  geometric_sequence a₁ q 3 = -8 :=
  sorry

end find_a4_l220_220449


namespace solve_natural_a_l220_220849

theorem solve_natural_a (a : ℕ) : 
  (∃ n : ℕ, a^2 + a + 1589 = n^2) ↔ (a = 43 ∨ a = 28 ∨ a = 316 ∨ a = 1588) :=
sorry

end solve_natural_a_l220_220849


namespace transformed_triangle_area_l220_220767

-- Define the function g and its properties
variable {R : Type*} [LinearOrderedField R]
variable (g : R → R)
variable (a b c : R)
variable (area_original : R)

-- Given conditions
-- The function g is defined such that the area of the triangle formed by 
-- points (a, g(a)), (b, g(b)), and (c, g(c)) is 24
axiom h₀ : {x | x = a ∨ x = b ∨ x = c} ⊆ Set.univ
axiom h₁ : area_original = 24

-- Define a function that computes the area of a triangle given three points
noncomputable def area_triangle (x1 y1 x2 y2 x3 y3 : R) : R := 
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem transformed_triangle_area (h₀ : {x | x = a ∨ x = b ∨ x = c} ⊆ Set.univ)
  (h₁ : area_triangle a (g a) b (g b) c (g c) = 24) :
  area_triangle (a / 3) (3 * g a) (b / 3) (3 * g b) (c / 3) (3 * g c) = 24 :=
sorry

end transformed_triangle_area_l220_220767


namespace parabola_directrix_l220_220166

theorem parabola_directrix (x y : ℝ) (h : y = 4 * x^2) : y = -1 / 16 :=
sorry

end parabola_directrix_l220_220166


namespace book_total_pages_l220_220208

theorem book_total_pages (num_chapters pages_per_chapter : ℕ) (h1 : num_chapters = 31) (h2 : pages_per_chapter = 61) :
  num_chapters * pages_per_chapter = 1891 := sorry

end book_total_pages_l220_220208


namespace simplify_fraction_l220_220161

variable (y b : ℚ)

theorem simplify_fraction : 
  (y+2) / 4 + (5 - 4*y + b) / 3 = (-13*y + 4*b + 26) / 12 := 
by
  sorry

end simplify_fraction_l220_220161


namespace largest_angle_in_triangle_l220_220894

theorem largest_angle_in_triangle (A B C : ℝ) (h₁ : A = 45) (h₂ : B / C = 4 / 5) (h₃ : A + B + C = 180) : 
  max A (max B C) = 75 :=
by
  -- Since no proof is needed, we mark it as sorry
  sorry

end largest_angle_in_triangle_l220_220894


namespace smallest_sector_angle_3_l220_220739

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_angles_is_360 (a : ℕ → ℕ) : Prop :=
  (Finset.range 15).sum a = 360

def smallest_possible_angle (a : ℕ → ℕ) (x : ℕ) : Prop :=
  ∀ i : ℕ, a i ≥ x

theorem smallest_sector_angle_3 :
  ∃ a : ℕ → ℕ,
    is_arithmetic_sequence a ∧
    sum_of_angles_is_360 a ∧
    smallest_possible_angle a 3 :=
sorry

end smallest_sector_angle_3_l220_220739


namespace chocolate_cost_is_75_l220_220503

def candy_bar_cost : ℕ := 25
def juice_pack_cost : ℕ := 50
def num_quarters : ℕ := 11
def total_cost_in_cents : ℕ := num_quarters * candy_bar_cost
def num_candy_bars : ℕ := 3
def num_pieces_of_chocolate : ℕ := 2

def chocolate_cost_in_cents (x : ℕ) : Prop :=
  (num_candy_bars * candy_bar_cost) + (num_pieces_of_chocolate * x) + juice_pack_cost = total_cost_in_cents

theorem chocolate_cost_is_75 : chocolate_cost_in_cents 75 :=
  sorry

end chocolate_cost_is_75_l220_220503


namespace find_denominator_l220_220127

theorem find_denominator (y : ℝ) (x : ℝ) (h₀ : y > 0) (h₁ : 9 * y / 20 + 3 * y / x = 0.75 * y) : x = 10 :=
sorry

end find_denominator_l220_220127


namespace frosting_needed_l220_220582

-- Definitions directly from the problem conditions
def cans_frosting_per_layer_cake := 1
def cans_frosting_per_single_cake := 0.5
def cans_frosting_per_dozen_cupcakes := 0.5
def cans_frosting_per_pan_brownies := 0.5

-- Quantities needed
def layer_cakes_needed := 3
def dozen_cupcakes_needed := 6
def single_cakes_needed := 12
def pans_brownies_needed := 18

-- Proposition to prove
theorem frosting_needed : 
  (cans_frosting_per_layer_cake * layer_cakes_needed) + 
  (cans_frosting_per_dozen_cupcakes * dozen_cupcakes_needed) + 
  (cans_frosting_per_single_cake * single_cakes_needed) + 
  (cans_frosting_per_pan_brownies * pans_brownies_needed) = 21 := 
by 
  sorry

end frosting_needed_l220_220582


namespace range_of_a_second_quadrant_l220_220733

theorem range_of_a_second_quadrant (a : ℝ) :
  (∀ (x y : ℝ), (x^2 + y^2 + 2 * a * x - 4 * a * y + 5 * a^2 - 4 = 0) → x < 0 ∧ y > 0) →
  a > 2 :=
sorry

end range_of_a_second_quadrant_l220_220733


namespace bhanu_income_l220_220229

theorem bhanu_income (I P : ℝ) (h1 : (P / 100) * I = 300) (h2 : (20 / 100) * (I - 300) = 140) : P = 30 := by
  sorry

end bhanu_income_l220_220229


namespace find_f_lg_lg_2_l220_220274

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * (Real.sin x) + 4

theorem find_f_lg_lg_2 (a b : ℝ) (m : ℝ) 
  (h1 : f a b (Real.logb 10 2) = 5) 
  (h2 : m = Real.logb 10 2) : 
  f a b (Real.logb 2 m) = 3 :=
sorry

end find_f_lg_lg_2_l220_220274


namespace distance_center_to_plane_l220_220706

noncomputable def sphere_center_to_plane_distance 
  (volume : ℝ) (AB AC : ℝ) (angleACB : ℝ) : ℝ :=
  let R := (3 * volume / 4 / Real.pi)^(1 / 3);
  let circumradius := AB / (2 * Real.sin (angleACB / 2));
  Real.sqrt (R^2 - circumradius^2)

theorem distance_center_to_plane 
  (volume : ℝ) (AB : ℝ) (angleACB : ℝ)
  (h_volume : volume = 500 * Real.pi / 3)
  (h_AB : AB = 4 * Real.sqrt 3)
  (h_angleACB : angleACB = Real.pi / 3) :
  sphere_center_to_plane_distance volume AB angleACB = 3 :=
by
  sorry

end distance_center_to_plane_l220_220706


namespace range_of_a_l220_220722

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) := 
sorry

end range_of_a_l220_220722


namespace parabola_directrix_l220_220167

theorem parabola_directrix (y : ℝ) (x : ℝ) (h : y = 8 * x^2) : 
  y = -1 / 32 :=
sorry

end parabola_directrix_l220_220167


namespace rate_of_work_l220_220487

theorem rate_of_work (A : ℝ) (h1: 0 < A) (h_eq : 1 / A + 1 / 6 = 1 / 2) : A = 3 := sorry

end rate_of_work_l220_220487


namespace friendly_enumeration_l220_220141

open Equiv.Perm

def friendly_permutation {n : ℕ} (α β : perm (Fin n)) (k : ℕ) : Prop :=
  ∀ i : Fin n, if i.val + 1 ≤ k then β i = α ⟨k - i.val, sorry⟩ else β i = α i

theorem friendly_enumeration (n : ℕ) (h : 2 ≤ n) : 
  ∃ (P : List (perm (Fin n))), 
    ∃ (m : ℕ), m = nat.factorial n ∧ 
    ∃ (P₀ : perm (Fin n)), 
      P ≠ [] ∧
      P.head = some P₀ ∧
      P.last = some P₀ ∧
      ∀ i (H: i < m), friendly_permutation (P.nthLe i (nat.lt_succ_self _)) (P.nthLe (i+1) (nat.lt_succ_self _)) (n-1) :=
sorry

end friendly_enumeration_l220_220141


namespace dog_food_bags_count_l220_220364

-- Define the constants based on the problem statement
def CatFoodBags := 327
def DogFoodMore := 273

-- Define the total number of dog food bags based on the given conditions
def DogFoodBags : ℤ := CatFoodBags + DogFoodMore

-- State the theorem we want to prove
theorem dog_food_bags_count : DogFoodBags = 600 := by
  sorry

end dog_food_bags_count_l220_220364


namespace average_page_count_per_essay_l220_220073

-- Conditions
def numberOfStudents := 15
def pagesFirstFive := 5 * 2
def pagesNextFive := 5 * 3
def pagesLastFive := 5 * 1

-- Total pages
def totalPages := pagesFirstFive + pagesNextFive + pagesLastFive

-- Proof problem statement
theorem average_page_count_per_essay : totalPages / numberOfStudents = 2 := by
  sorry

end average_page_count_per_essay_l220_220073


namespace geometric_sequence_general_term_l220_220528

theorem geometric_sequence_general_term :
  ∀ (n : ℕ), (n > 0) →
  (∃ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ (k : ℕ), k > 0 → a (k+1) = 2 * a k) ∧ a n = 2^(n-1)) :=
by
  sorry

end geometric_sequence_general_term_l220_220528


namespace initial_unread_messages_correct_l220_220150

-- Definitions based on conditions
def messages_read_per_day := 20
def messages_new_per_day := 6
def duration_in_days := 7
def effective_reading_rate := messages_read_per_day - messages_new_per_day

-- The initial number of unread messages
def initial_unread_messages := duration_in_days * effective_reading_rate

-- The theorem we want to prove
theorem initial_unread_messages_correct :
  initial_unread_messages = 98 :=
sorry

end initial_unread_messages_correct_l220_220150


namespace evaluate_expression_l220_220245

-- Given variables x and y are non-zero
variables (x y : ℝ)

-- Condition
axiom xy_nonzero : x * y ≠ 0

-- Statement of the proof
theorem evaluate_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 2) / x * (y^3 + 2) / y + (x^3 - 2) / y * (y^3 - 2) / x) = 2 * x * y * (x^2 * y^2) + 8 / (x * y) := 
by {
  sorry
}

end evaluate_expression_l220_220245


namespace find_root_of_polynomial_l220_220965

theorem find_root_of_polynomial (a c x : ℝ)
  (h1 : a + c = -3)
  (h2 : 64 * a + c = 60)
  (h3 : x = 2) :
  a * x^3 - 2 * x + c = 0 :=
by
  sorry

end find_root_of_polynomial_l220_220965


namespace arithmetic_progressions_count_l220_220297

theorem arithmetic_progressions_count (d : ℕ) (h_d : d = 2) (S : ℕ) (h_S : S = 200) : 
  ∃ n : ℕ, n = 6 := sorry

end arithmetic_progressions_count_l220_220297


namespace pow_sum_geq_pow_prod_l220_220301

theorem pow_sum_geq_pow_prod (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x^5 + y^5 ≥ x^4 * y + x * y^4 :=
 by sorry

end pow_sum_geq_pow_prod_l220_220301


namespace smallest_number_of_students_l220_220661

theorem smallest_number_of_students
    (g11 g10 g9 : Nat)
    (h_ratio1 : 4 * g9 = 3 * g11)
    (h_ratio2 : 6 * g10 = 5 * g11) :
  g11 + g10 + g9 = 31 :=
sorry

end smallest_number_of_students_l220_220661


namespace probability_of_point_in_spheres_l220_220665

noncomputable def radius_of_inscribed_sphere (R : ℝ) : ℝ := 2 * R / 3
noncomputable def radius_of_tangent_spheres (R : ℝ) : ℝ := 2 * R / 3

theorem probability_of_point_in_spheres
  (R : ℝ)  -- Radius of the circumscribed sphere
  (r : ℝ := radius_of_inscribed_sphere R)  -- Radius of the inscribed sphere
  (r_t : ℝ := radius_of_tangent_spheres R)  -- Radius of each tangent sphere
  (volume : ℝ := 4/3 * Real.pi * r^3)  -- Volume of each smaller sphere
  (total_small_volume : ℝ := 5 * volume)  -- Total volume of smaller spheres
  (circumsphere_volume : ℝ := 4/3 * Real.pi * (2 * R)^3)  -- Volume of the circumscribed sphere
  : 
  total_small_volume / circumsphere_volume = 5 / 27 :=
by
  sorry

end probability_of_point_in_spheres_l220_220665


namespace totalPlayers_l220_220788

def kabadiParticipants : ℕ := 50
def khoKhoParticipants : ℕ := 80
def soccerParticipants : ℕ := 30
def kabadiAndKhoKhoParticipants : ℕ := 15
def kabadiAndSoccerParticipants : ℕ := 10
def khoKhoAndSoccerParticipants : ℕ := 25
def allThreeParticipants : ℕ := 8

theorem totalPlayers : kabadiParticipants + khoKhoParticipants + soccerParticipants 
                       - kabadiAndKhoKhoParticipants - kabadiAndSoccerParticipants 
                       - khoKhoAndSoccerParticipants + allThreeParticipants = 118 :=
by 
  sorry

end totalPlayers_l220_220788


namespace negate_p_l220_220545

theorem negate_p (p : Prop) :
  (∃ x : ℝ, 0 < x ∧ 3^x < x^3) ↔ (¬ (∀ x : ℝ, 0 < x → 3^x ≥ x^3)) :=
by sorry

end negate_p_l220_220545


namespace center_square_number_l220_220728

def in_center_square (grid : Matrix (Fin 3) (Fin 3) ℕ) : ℕ := grid 1 1

theorem center_square_number
  (grid : Matrix (Fin 3) (Fin 3) ℕ)
  (consecutive_share_edge : ∀ (i j : Fin 3) (n : ℕ), 
                              (i < 2 ∨ j < 2) →
                              (∃ d, d ∈ [(-1,0), (1,0), (0,-1), (0,1)] ∧ 
                              grid (i + d.1) (j + d.2) = n + 1))
  (corner_sum_20 : grid 0 0 + grid 0 2 + grid 2 0 + grid 2 2 = 20)
  (diagonal_sum_15 : 
    (grid 0 0 + grid 1 1 + grid 2 2 = 15) 
    ∨ 
    (grid 0 2 + grid 1 1 + grid 2 0 = 15))
  : in_center_square grid = 5 := sorry

end center_square_number_l220_220728


namespace geometric_progression_ratio_l220_220226

theorem geometric_progression_ratio (r : ℕ) (h : 4 + 4 * r + 4 * r^2 + 4 * r^3 = 60) : r = 2 :=
by
  sorry

end geometric_progression_ratio_l220_220226


namespace a_minus_b_l220_220428

theorem a_minus_b (a b : ℚ) :
  (∀ x y, (x = 3 → y = 7) ∨ (x = 10 → y = 19) → y = a * x + b) →
  a - b = -(1/7) :=
by
  sorry

end a_minus_b_l220_220428


namespace labor_arrangement_count_l220_220634

theorem labor_arrangement_count (volunteers : ℕ) (choose_one_day : ℕ) (days : ℕ) 
    (h_volunteers : volunteers = 7) 
    (h_choose_one_day : choose_one_day = 3) 
    (h_days : days = 2) : 
    (Nat.choose volunteers choose_one_day) * (Nat.choose (volunteers - choose_one_day) choose_one_day) = 140 := 
by
  sorry

end labor_arrangement_count_l220_220634


namespace roses_problem_l220_220569

variable (R B C : ℕ)

theorem roses_problem
    (h1 : R = B + 10)
    (h2 : C = 10)
    (h3 : 16 - 6 = C)
    (h4 : B = R - C):
  R = B + 10 ∧ R - C = B := 
by 
  have hC: C = 10 := by linarith
  have hR: R = B + 10 := by linarith
  have hRC: R - C = B := by linarith
  exact ⟨hR, hRC⟩

end roses_problem_l220_220569


namespace sum_first_ten_terms_arithmetic_sequence_l220_220537

theorem sum_first_ten_terms_arithmetic_sequence (a d : ℝ) (S10 : ℝ) 
  (h1 : 0 < d) 
  (h2 : (a - d) + a + (a + d) = -6) 
  (h3 : (a - d) * a * (a + d) = 10) 
  (h4 : S10 = 5 * (2 * (a - d) + 9 * d)) :
  S10 = -20 + 35 * Real.sqrt 6.5 :=
by sorry

end sum_first_ten_terms_arithmetic_sequence_l220_220537


namespace num_students_third_section_l220_220810

-- Define the conditions
def num_students_first_section : ℕ := 65
def num_students_second_section : ℕ := 35
def num_students_fourth_section : ℕ := 42
def mean_marks_first_section : ℝ := 50
def mean_marks_second_section : ℝ := 60
def mean_marks_third_section : ℝ := 55
def mean_marks_fourth_section : ℝ := 45
def overall_average_marks : ℝ := 51.95

-- Theorem stating the number of students in the third section
theorem num_students_third_section
  (x : ℝ)
  (h : (num_students_first_section * mean_marks_first_section
       + num_students_second_section * mean_marks_second_section
       + x * mean_marks_third_section
       + num_students_fourth_section * mean_marks_fourth_section)
       = overall_average_marks * (num_students_first_section + num_students_second_section + x + num_students_fourth_section)) :
  x = 45 :=
by
  -- Proof will go here
  sorry

end num_students_third_section_l220_220810


namespace joe_average_score_l220_220443

theorem joe_average_score (A B C : ℕ) (lowest_score : ℕ) (final_average : ℕ) :
  lowest_score = 45 ∧ final_average = 65 ∧ (A + B + C) / 3 = final_average →
  (A + B + C + lowest_score) / 4 = 60 := by
  sorry

end joe_average_score_l220_220443


namespace average_page_count_per_essay_l220_220074

-- Conditions
def numberOfStudents := 15
def pagesFirstFive := 5 * 2
def pagesNextFive := 5 * 3
def pagesLastFive := 5 * 1

-- Total pages
def totalPages := pagesFirstFive + pagesNextFive + pagesLastFive

-- Proof problem statement
theorem average_page_count_per_essay : totalPages / numberOfStudents = 2 := by
  sorry

end average_page_count_per_essay_l220_220074


namespace sum_of_fractions_l220_220232

theorem sum_of_fractions :
  (1/15 + 2/15 + 3/15 + 4/15 + 5/15 + 6/15 + 7/15 + 8/15 + 9/15 + 46/15) = (91/15) := by
  sorry

end sum_of_fractions_l220_220232


namespace polynomial_evaluation_l220_220839

theorem polynomial_evaluation 
  (x : ℝ) 
  (h1 : x^2 - 3 * x - 10 = 0) 
  (h2 : x > 0) : 
  (x^4 - 3 * x^3 + 2 * x^2 + 5 * x - 7) = 318 :=
by
  sorry

end polynomial_evaluation_l220_220839


namespace simplify_fraction_l220_220919

theorem simplify_fraction :
  (4 / (Real.sqrt 108 + 2 * Real.sqrt 12 + 2 * Real.sqrt 27)) = (Real.sqrt 3 / 12) := 
by
  -- Proof goes here
  sorry

end simplify_fraction_l220_220919


namespace find_floors_l220_220667

theorem find_floors
  (a b : ℕ)
  (alexie_bathrooms_per_floor : ℕ := 3)
  (alexie_bedrooms_per_floor : ℕ := 2)
  (baptiste_bathrooms_per_floor : ℕ := 4)
  (baptiste_bedrooms_per_floor : ℕ := 3)
  (total_bathrooms : ℕ := 25)
  (total_bedrooms : ℕ := 18)
  (h1 : alexie_bathrooms_per_floor * a + baptiste_bathrooms_per_floor * b = total_bathrooms)
  (h2 : alexie_bedrooms_per_floor * a + baptiste_bedrooms_per_floor * b = total_bedrooms) :
  a = 3 ∧ b = 4 :=
by
  sorry

end find_floors_l220_220667


namespace evaluate_expression_l220_220235

theorem evaluate_expression :
  (- (3 / 4 : ℚ)) / 3 * (- (2 / 5 : ℚ)) = 1 / 10 := 
by
  -- Here is where the proof would go
  sorry

end evaluate_expression_l220_220235


namespace cotangent_product_identity_l220_220554

theorem cotangent_product_identity :
  (∏ i in Finset.range 45, (1 + Real.cot (i + 1 : ℝ) * Real.pi / 180)) = 2^23 :=
by
  sorry

end cotangent_product_identity_l220_220554


namespace inequality_holds_l220_220583

theorem inequality_holds (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  x^4 + y^4 + 2 / (x^2 * y^2) ≥ 4 := 
by
  sorry

end inequality_holds_l220_220583


namespace monthly_production_increase_l220_220984

/-- A salt manufacturing company produced 3000 tonnes in January and increased its
    production by some tonnes every month over the previous month until the end
    of the year. Given that the average daily production was 116.71232876712328 tonnes,
    determine the monthly production increase. -/
theorem monthly_production_increase :
  let initial_production := 3000
  let daily_average_production := 116.71232876712328
  let days_per_year := 365
  let total_yearly_production := daily_average_production * days_per_year
  let months_per_year := 12
  ∃ (x : ℝ), total_yearly_production = (months_per_year / 2) * (2 * initial_production + (months_per_year - 1) * x) → x = 100 :=
sorry

end monthly_production_increase_l220_220984


namespace percentage_returned_l220_220666

theorem percentage_returned (R : ℕ) (S : ℕ) (total : ℕ) (least_on_lot : ℕ) (max_rented : ℕ)
  (h1 : total = 20) (h2 : least_on_lot = 10) (h3 : max_rented = 20) (h4 : R = 20) (h5 : S ≥ 10) :
  (S / R) * 100 ≥ 50 := sorry

end percentage_returned_l220_220666


namespace smallest_possible_a_plus_b_l220_220541

theorem smallest_possible_a_plus_b :
  ∃ (a b : ℕ), (0 < a ∧ 0 < b) ∧ (2^10 * 7^3 = a^b) ∧ (a + b = 350753) :=
sorry

end smallest_possible_a_plus_b_l220_220541


namespace modified_full_house_probability_l220_220941

def total_choices : ℕ := Nat.choose 52 6

def ways_rank1 : ℕ := 13
def ways_3_cards : ℕ := Nat.choose 4 3
def ways_rank2 : ℕ := 12
def ways_2_cards : ℕ := Nat.choose 4 2
def ways_additional_card : ℕ := 11 * 4

def ways_modified_full_house : ℕ := ways_rank1 * ways_3_cards * ways_rank2 * ways_2_cards * ways_additional_card

def probability_modified_full_house : ℚ := ways_modified_full_house / total_choices

theorem modified_full_house_probability : probability_modified_full_house = 24 / 2977 := 
by sorry

end modified_full_house_probability_l220_220941


namespace car_and_bicycle_distances_l220_220121

noncomputable def train_speed : ℝ := 100 -- speed of the train in mph
noncomputable def car_speed : ℝ := (2 / 3) * train_speed -- speed of the car in mph
noncomputable def bicycle_speed : ℝ := (1 / 5) * train_speed -- speed of the bicycle in mph
noncomputable def travel_time_hours : ℝ := 30 / 60 -- travel time in hours, which is 0.5 hours

noncomputable def car_distance : ℝ := car_speed * travel_time_hours
noncomputable def bicycle_distance : ℝ := bicycle_speed * travel_time_hours

theorem car_and_bicycle_distances :
  car_distance = 100 / 3 ∧ bicycle_distance = 10 :=
by
  sorry

end car_and_bicycle_distances_l220_220121


namespace count_four_digit_palindrome_squares_l220_220638

-- Define what a palindrome number is
def is_palindrome (n : ℕ) : Prop :=
  let digits := to_string n
  digits = digits.reverse

-- Define four-digit perfect squares
def is_four_digit_perfect_square (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (∃ m : ℕ, n = m * m)

-- The theorem we want to prove
theorem count_four_digit_palindrome_squares :
  ∃! n, is_four_digit_perfect_square n ∧ is_palindrome n := {
  exists.intro 8100 (and.intro ⟨le_of_eq 8100, le_of_eq 8100, exists.intro 90 rfl⟩ rfl)
    (λ m h, and.elim h (λ h1 h2, 
      and.elim h1 (λ h1a h1b,
        exists.elim h1b (λ k hk,
          have hk1 := congr_arg (λ x, x * x) (eq.trans hk.symm (congr_arg has_sub m 90)),
          let h := (sub_eq_zero.mp hk1.symm.d_symm_trans)
          have m.char_zero := hk1 darge introspec Number.to_one_palindrome,
          rfl whether)))
      sorry -- the actual proof will go here
}

end count_four_digit_palindrome_squares_l220_220638


namespace pow_div_l220_220952

theorem pow_div (x : ℕ) (a b c d : ℕ) (h1 : x^b = d) (h2 : x^(a*d) = c) : c / (d^b) = 512 := by
  sorry

end pow_div_l220_220952


namespace max_sum_of_multiplication_table_l220_220773

theorem max_sum_of_multiplication_table :
  let numbers := [3, 5, 7, 11, 17, 19]
  let repeated_num := 19
  ∃ d e f, d ∈ numbers ∧ e ∈ numbers ∧ f ∈ numbers ∧ d ≠ e ∧ e ≠ f ∧ d ≠ f ∧
  3 * repeated_num * (d + e + f) = 1995 := 
by {
  sorry
}

end max_sum_of_multiplication_table_l220_220773


namespace total_baseball_cards_l220_220194

/-- 
Given that you have 5 friends and each friend gets 91 baseball cards, 
prove that the total number of baseball cards you have is 455.
-/
def baseball_cards (f c : Nat) (t : Nat) : Prop :=
  (t = f * c)

theorem total_baseball_cards:
  ∀ (f c t : Nat), f = 5 → c = 91 → t = 455 → baseball_cards f c t :=
by
  intros f c t hf hc ht
  sorry

end total_baseball_cards_l220_220194


namespace cos_double_angle_l220_220556

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l220_220556


namespace no_nat_nums_satisfy_gcd_lcm_condition_l220_220001

theorem no_nat_nums_satisfy_gcd_lcm_condition :
  ¬ ∃ (x y : ℕ), Nat.gcd x y + Nat.lcm x y = x + y + 2021 := 
sorry

end no_nat_nums_satisfy_gcd_lcm_condition_l220_220001


namespace solve_for_x_l220_220387

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 3 * x) :
    5 * y ^ 2 + 2 * y + 3 = 3 * (9 * x ^ 2 + y + 1) ↔ x = 0 ∨ x = 1 / 6 := 
by
  sorry

end solve_for_x_l220_220387


namespace sum_of_solutions_l220_220531

theorem sum_of_solutions (x : ℝ) :
  (∀ x, x^2 - 17 * x + 54 = 0) → 
  (∃ r s : ℝ, r ≠ s ∧ r + s = 17) :=
by
  sorry

end sum_of_solutions_l220_220531


namespace average_speed_palindrome_l220_220622

theorem average_speed_palindrome :
  ∀ (initial_odometer final_odometer : ℕ) (hours : ℕ),
  initial_odometer = 123321 →
  final_odometer = 124421 →
  hours = 4 →
  (final_odometer - initial_odometer) / hours = 275 :=
by
  intros initial_odometer final_odometer hours h1 h2 h3
  sorry

end average_speed_palindrome_l220_220622


namespace loss_percentage_is_11_percent_l220_220628

-- Definitions based on conditions
def costPrice : ℝ := 1500
def sellingPrice : ℝ := 1335

-- The statement to prove
theorem loss_percentage_is_11_percent :
  ((costPrice - sellingPrice) / costPrice) * 100 = 11 := by
  sorry

end loss_percentage_is_11_percent_l220_220628


namespace min_value_of_linear_expression_l220_220022

theorem min_value_of_linear_expression {x y : ℝ} (h1 : 2 * x - y ≥ 0) (h2 : x + y - 3 ≥ 0) (h3 : y - x ≥ 0) :
  ∃ z, z = 2 * x + y ∧ z = 4 := by
  sorry

end min_value_of_linear_expression_l220_220022


namespace solve_arcsin_arccos_eq_l220_220630

theorem solve_arcsin_arccos_eq (x : ℝ) :
  (arcsin (2 * x) + arcsin (1 - 2 * x) = arccos (2 * x)) ↔ (x = 0 ∨ x = 1 / 2 ∨ x = -1 / 2) :=
by
  sorry

end solve_arcsin_arccos_eq_l220_220630


namespace share_of_a_l220_220575

variables {a b c d : ℝ}
variables {total : ℝ}

-- Conditions
def condition1 (a b c d : ℝ) := a = (3/5) * (b + c + d)
def condition2 (a b c d : ℝ) := b = (2/3) * (a + c + d)
def condition3 (a b c d : ℝ) := c = (4/7) * (a + b + d)
def total_distributed (a b c d : ℝ) := a + b + c + d = 1200

-- Theorem to prove
theorem share_of_a (a b c d : ℝ) (h1 : condition1 a b c d) (h2 : condition2 a b c d) (h3 : condition3 a b c d) (h4 : total_distributed a b c d) : 
  a = 247.5 :=
sorry

end share_of_a_l220_220575


namespace pow_div_eq_l220_220946

theorem pow_div_eq : (8:ℕ) ^ 15 / (64:ℕ) ^ 6 = 512 := by
  have h1 : 64 = 8 ^ 2 := by sorry
  have h2 : (64:ℕ) ^ 6 = (8 ^ 2) ^ 6 := by sorry
  have h3 : (8 ^ 2) ^ 6 = 8 ^ 12 := by sorry
  have h4 : (8:ℕ) ^ 15 / (8 ^ 12) = 8 ^ (15 - 12) := by sorry
  have h5 : 8 ^ 3 = 512 := by sorry
  exact sorry

end pow_div_eq_l220_220946


namespace expression_evaluation_l220_220185

theorem expression_evaluation :
  (8 / 4 - 3^2 + 4 * 5) = 13 :=
by sorry

end expression_evaluation_l220_220185


namespace triangle_centroid_l220_220525

theorem triangle_centroid :
  let (x1, y1) := (2, 6)
  let (x2, y2) := (6, 2)
  let (x3, y3) := (4, 8)
  let centroid_x := (x1 + x2 + x3) / 3
  let centroid_y := (y1 + y2 + y3) / 3
  (centroid_x, centroid_y) = (4, 16 / 3) :=
by
  let x1 := 2
  let y1 := 6
  let x2 := 6
  let y2 := 2
  let x3 := 4
  let y3 := 8
  let centroid_x := (x1 + x2 + x3) / 3
  let centroid_y := (y1 + y2 + y3) / 3
  show (centroid_x, centroid_y) = (4, 16 / 3)
  sorry

end triangle_centroid_l220_220525


namespace maximum_a_pos_integer_greatest_possible_value_of_a_l220_220599

theorem maximum_a_pos_integer (a : ℕ) (h : ∃ x : ℤ, x^2 + (a * x : ℤ) = -20) : a ≤ 21 :=
by
  sorry

theorem greatest_possible_value_of_a : ∃ (a : ℕ), (∀ b : ℕ, (∃ x : ℤ, x^2 + (b * x : ℤ) = -20) → b ≤ 21) ∧ 21 = a :=
by
  sorry

end maximum_a_pos_integer_greatest_possible_value_of_a_l220_220599


namespace largest_triangle_perimeter_maximizes_l220_220369

theorem largest_triangle_perimeter_maximizes 
  (y : ℤ) 
  (h1 : 3 ≤ y) 
  (h2 : y < 16) : 
  (7 + 9 + y) = 31 ↔ y = 15 := 
by 
  sorry

end largest_triangle_perimeter_maximizes_l220_220369


namespace ratio_of_volumes_of_spheres_l220_220431

theorem ratio_of_volumes_of_spheres (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a / b = 1 / 2 ∧ b / c = 2 / 3) : a^3 / b^3 = 1 / 8 ∧ b^3 / c^3 = 8 / 27 :=
by
  sorry

end ratio_of_volumes_of_spheres_l220_220431


namespace gasoline_tank_capacity_l220_220979

theorem gasoline_tank_capacity :
  ∀ (x : ℕ), (5 / 6 * (x : ℚ) - 18 = 1 / 3 * (x : ℚ)) → x = 36 :=
by
  sorry

end gasoline_tank_capacity_l220_220979


namespace evaluate_expression_l220_220532

theorem evaluate_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 := 
by 
  sorry

end evaluate_expression_l220_220532


namespace longest_side_length_of_quadrilateral_l220_220832

-- Define the system of inequalities
def inFeasibleRegion (x y : ℝ) : Prop :=
  (x + 2 * y ≤ 4) ∧
  (3 * x + y ≥ 3) ∧
  (x ≥ 0) ∧
  (y ≥ 0)

-- The goal is to prove that the longest side length is 5
theorem longest_side_length_of_quadrilateral :
  ∃ a b c d : (ℝ × ℝ), inFeasibleRegion a.1 a.2 ∧
                  inFeasibleRegion b.1 b.2 ∧
                  inFeasibleRegion c.1 c.2 ∧
                  inFeasibleRegion d.1 d.2 ∧
                  -- For each side, specify the length condition (Euclidean distance)
                  max (dist a b) (max (dist b c) (max (dist c d) (dist d a))) = 5 :=
by sorry

end longest_side_length_of_quadrilateral_l220_220832


namespace successive_increases_eq_single_l220_220318

variable (P : ℝ)

def increase_by (initial : ℝ) (pct : ℝ) : ℝ := initial * (1 + pct)
def discount_by (initial : ℝ) (pct : ℝ) : ℝ := initial * (1 - pct)

theorem successive_increases_eq_single (P : ℝ) :
  increase_by (increase_by (discount_by (increase_by P 0.30) 0.10) 0.15) 0.20 = increase_by P 0.6146 :=
  sorry

end successive_increases_eq_single_l220_220318


namespace game_score_correct_answers_l220_220012

theorem game_score_correct_answers :
  ∃ x : ℕ, (∃ y : ℕ, x + y = 30 ∧ 7 * x - 12 * y = 77) ∧ x = 23 :=
by
  use 23
  sorry

end game_score_correct_answers_l220_220012


namespace find_a_l220_220088

def otimes (a b : ℝ) : ℝ := a - 2 * b

theorem find_a (a : ℝ) : (∀ x : ℝ, (otimes x 3 > 0 ∧ otimes x a > a) ↔ x > 6) → a ≤ 2 :=
begin
  intro h,
  sorry
end

end find_a_l220_220088


namespace find_a_l220_220448

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 9 * Real.log x

def monotonically_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f y ≤ f x

def valid_interval (a : ℝ) : Prop :=
  monotonically_decreasing f (Set.Icc (a-1) (a+1))

theorem find_a :
  {a : ℝ | valid_interval a} = {a : ℝ | 1 < a ∧ a ≤ 2} :=
by
  sorry

end find_a_l220_220448


namespace polynomial_roots_sum_l220_220743

theorem polynomial_roots_sum (p q : ℂ) (hp : p + q = 5) (hq : p * q = 7) : 
  p^3 + p^4 * q^2 + p^2 * q^4 + q^3 = 559 := 
by 
  sorry

end polynomial_roots_sum_l220_220743


namespace parallel_lines_eq_a_l220_220415

theorem parallel_lines_eq_a (a : ℝ) :
  (∀ x y : ℝ, x + 2 * a * y - 1 = 0 → (a + 1) * x - a * y = 0) → (a = -3/2 ∨ a = 0) :=
by sorry

end parallel_lines_eq_a_l220_220415


namespace scorpion_millipedes_needed_l220_220065

theorem scorpion_millipedes_needed 
  (total_segments_required : ℕ)
  (eaten_millipede_1_segments : ℕ)
  (eaten_millipede_2_segments : ℕ)
  (segments_per_millipede : ℕ)
  (n_millipedes_needed : ℕ)
  (total_segments : total_segments_required = 800) 
  (segments_1 : eaten_millipede_1_segments = 60)
  (segments_2 : eaten_millipede_2_segments = 2 * 60)
  (needed_segments_calculation : 800 - (60 + 2 * (2 * 60)) = n_millipedes_needed * 50) 
  : n_millipedes_needed = 10 :=
by
  sorry

end scorpion_millipedes_needed_l220_220065


namespace new_car_distance_l220_220217

-- State the given conditions as a Lemma in Lean 4
theorem new_car_distance (d_old : ℝ) (d_new : ℝ) (h1 : d_old = 150) (h2 : d_new = d_old * 1.30) : d_new = 195 := 
by
  sorry

end new_car_distance_l220_220217


namespace smallest_abs_diff_of_powers_l220_220398

open Nat

theorem smallest_abs_diff_of_powers :
  ∃ (m n : ℕ), abs (36 ^ m - 5 ^ n) = 11 :=
sorry

end smallest_abs_diff_of_powers_l220_220398


namespace total_cost_calculation_l220_220237

def total_transportation_cost (x : ℝ) : ℝ :=
  let cost_A_to_C := 20 * x
  let cost_A_to_D := 30 * (240 - x)
  let cost_B_to_C := 24 * (200 - x)
  let cost_B_to_D := 32 * (60 + x)
  cost_A_to_C + cost_A_to_D + cost_B_to_C + cost_B_to_D

theorem total_cost_calculation (x : ℝ) :
  total_transportation_cost x = 13920 - 2 * x := by
  sorry

end total_cost_calculation_l220_220237


namespace parallel_lines_m_values_l220_220266

theorem parallel_lines_m_values (m : ℝ) :
  (∀ (x y : ℝ), (m - 2) * x - y + 5 = 0) ∧ 
  (∀ (x y : ℝ), (m - 2) * x + (3 - m) * y + 2 = 0) → 
  (m = 2 ∨ m = 4) :=
sorry

end parallel_lines_m_values_l220_220266


namespace greatest_two_digit_number_l220_220962

theorem greatest_two_digit_number (x y : ℕ) (h1 : x < y) (h2 : x * y = 12) : 10 * x + y = 34 :=
sorry

end greatest_two_digit_number_l220_220962


namespace find_x_l220_220411

noncomputable def f (x : ℝ) : ℝ := 5 * x^3 - 3

theorem find_x (x : ℝ) : (f⁻¹ (-2) = x) → x = -43 := by
  sorry

end find_x_l220_220411


namespace inequality_solution_l220_220308

theorem inequality_solution 
  (x : ℝ) 
  (h1 : (x + 3) / 2 ≤ x + 2) 
  (h2 : 2 * (x + 4) > 4 * x + 2) : 
  -1 ≤ x ∧ x < 3 := sorry

end inequality_solution_l220_220308


namespace raul_money_left_l220_220016

theorem raul_money_left (initial_amount comics_cost comics_count: ℕ) (h1: initial_amount = 87) (h2: comics_cost = 4) (h3: comics_count = 8):
  initial_amount - (comics_cost * comics_count) = 55 :=
by
  rw [h1, h2, h3]
  norm_num

end raul_money_left_l220_220016


namespace find_value_of_f2_l220_220540

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

theorem find_value_of_f2 : f 2 = 101 / 99 :=
  sorry

end find_value_of_f2_l220_220540


namespace shadow_length_l220_220280

variable (H h d : ℝ) (h_pos : h > 0) (H_pos : H > 0) (H_neq_h : H ≠ h)

theorem shadow_length (x : ℝ) (hx : x = d * h / (H - h)) :
  x = d * h / (H - h) :=
sorry

end shadow_length_l220_220280


namespace largest_m_l220_220285

noncomputable def max_min_ab_bc_ca (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h1 : a + b + c = 9) (h2 : ab + bc + ca = 27) : ℝ :=
  min (a * b) (min (b * c) (c * a))

theorem largest_m (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : a + b + c = 9) (h2 : ab + bc + ca = 27) : max_min_ab_bc_ca a b c ha hb hc h1 h2 = 6.75 :=
by
  sorry

end largest_m_l220_220285


namespace dot_product_parallel_vectors_l220_220417

variable (x : ℝ)
def a := (1, 2 : ℝ × ℝ)
def b := (x, -4 : ℝ × ℝ)
def parallel (u v : ℝ × ℝ) : Prop := ∃ (λ : ℝ), v = (λ * u.1, λ * u.2)

theorem dot_product_parallel_vectors (h : parallel a b) : a.1 * b.1 + a.2 * b.2 = -10 := by
  sorry

end dot_product_parallel_vectors_l220_220417


namespace largest_of_seven_consecutive_l220_220171

theorem largest_of_seven_consecutive (n : ℕ) (h1 : (7 * n + 21 = 3020)) : (n + 6 = 434) :=
sorry

end largest_of_seven_consecutive_l220_220171


namespace find_x2017_l220_220406

-- Define that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

-- Define that f is increasing
def is_increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y
  
-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + n * d

-- Main theorem
theorem find_x2017
  (f : ℝ → ℝ) (x : ℕ → ℝ)
  (Hodd : is_odd_function f)
  (Hinc : is_increasing_function f)
  (Hseq : ∀ n, x (n + 1) = x n + 2)
  (H7_8 : f (x 7) + f (x 8) = 0) :
  x 2017 = 4019 := 
sorry

end find_x2017_l220_220406


namespace geometric_and_arithmetic_sequences_l220_220863

theorem geometric_and_arithmetic_sequences (a b c x y : ℝ) 
  (h1 : b^2 = a * c)
  (h2 : 2 * x = a + b)
  (h3 : 2 * y = b + c) :
  (a / x + c / y) = 2 := 
by 
  sorry

end geometric_and_arithmetic_sequences_l220_220863


namespace hockey_league_total_games_l220_220980

theorem hockey_league_total_games 
  (divisions : ℕ)
  (teams_per_division : ℕ)
  (intra_division_games : ℕ)
  (inter_division_games : ℕ) :
  divisions = 2 →
  teams_per_division = 6 →
  intra_division_games = 4 →
  inter_division_games = 2 →
  (divisions * ((teams_per_division * (teams_per_division - 1)) / 2) * intra_division_games) + 
  ((divisions / 2) * (divisions / 2) * teams_per_division * teams_per_division * inter_division_games) = 192 :=
by
  intros h_div h_teams h_intra h_inter
  sorry

end hockey_league_total_games_l220_220980


namespace fifth_friend_paid_13_l220_220845

noncomputable def fifth_friend_payment (a b c d e : ℝ) : Prop :=
a = (1/3) * (b + c + d + e) ∧
b = (1/4) * (a + c + d + e) ∧
c = (1/5) * (a + b + d + e) ∧
a + b + c + d + e = 120 ∧
e = 13

theorem fifth_friend_paid_13 : 
  ∃ (a b c d e : ℝ), fifth_friend_payment a b c d e := 
sorry

end fifth_friend_paid_13_l220_220845


namespace baby_guppies_l220_220225

theorem baby_guppies (x : ℕ) (h1 : 7 + x + 9 = 52) : x = 36 :=
by
  sorry

end baby_guppies_l220_220225


namespace proof_l220_220747

noncomputable def question (a b c : ℂ) : ℂ := (a^3 + b^3 + c^3) / (a * b * c)

theorem proof (a b c : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 15)
  (h5 : (a - b)^2 + (a - c)^2 + (b - c)^2 = 2 * a * b * c) :
  question a b c = 18 :=
by
  sorry

end proof_l220_220747


namespace temperature_at_night_is_minus_two_l220_220082

theorem temperature_at_night_is_minus_two (temperature_noon temperature_afternoon temperature_drop_by_night temperature_night : ℤ) : 
  temperature_noon = 5 → temperature_afternoon = 7 → temperature_drop_by_night = 9 → 
  temperature_night = temperature_afternoon - temperature_drop_by_night → 
  temperature_night = -2 := 
by
  intros h1 h2 h3 h4
  rw [h2, h3] at h4
  exact h4


end temperature_at_night_is_minus_two_l220_220082


namespace doughnut_price_l220_220420

theorem doughnut_price
  (K C B : ℕ)
  (h1: K = 4 * C + 5)
  (h2: K = 5 * C - 6)
  (h3: K = 2 * C + 3 * B) :
  B = 9 := 
sorry

end doughnut_price_l220_220420


namespace lemonade_glasses_from_fruit_l220_220902

noncomputable def lemons_per_glass : ℕ := 2
noncomputable def oranges_per_glass : ℕ := 1
noncomputable def total_lemons : ℕ := 18
noncomputable def total_oranges : ℕ := 10
noncomputable def grapefruits : ℕ := 6
noncomputable def lemons_per_grapefruit : ℕ := 2
noncomputable def oranges_per_grapefruit : ℕ := 1

theorem lemonade_glasses_from_fruit :
  (total_lemons / lemons_per_glass) = 9 →
  (total_oranges / oranges_per_glass) = 10 →
  min (total_lemons / lemons_per_glass) (total_oranges / oranges_per_glass) = 9 →
  (grapefruits * lemons_per_grapefruit = 12) →
  (grapefruits * oranges_per_grapefruit = 6) →
  (9 + grapefruits) = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end lemonade_glasses_from_fruit_l220_220902


namespace cone_base_radius_l220_220768

/-- A hemisphere of radius 3 rests on the base of a circular cone and is tangent to the cone's lateral surface along a circle. 
Given that the height of the cone is 9, prove that the base radius of the cone is 10.5. -/
theorem cone_base_radius
  (r_h : ℝ) (h : ℝ) (r : ℝ) 
  (hemisphere_tangent_cone : r_h = 3)
  (cone_height : h = 9)
  (tangent_circle_height : r - r_h = 3) :
  r = 10.5 := by
  sorry

end cone_base_radius_l220_220768


namespace cougar_ratio_l220_220083

theorem cougar_ratio (lions tigers total_cats cougars : ℕ) 
  (h_lions : lions = 12) 
  (h_tigers : tigers = 14) 
  (h_total : total_cats = 39) 
  (h_cougars : cougars = total_cats - (lions + tigers)) 
  : cougars * 2 = lions + tigers := 
by 
  rw [h_lions, h_tigers] 
  norm_num at * 
  sorry

end cougar_ratio_l220_220083


namespace min_rectangle_area_l220_220982

theorem min_rectangle_area : 
  ∃ (x y : ℕ), 2 * (x + y) = 80 ∧ x * y = 39 :=
by
  sorry

end min_rectangle_area_l220_220982


namespace largest_fraction_l220_220188

theorem largest_fraction :
  let A := (5 : ℚ) / 11
  let B := (7 : ℚ) / 15
  let C := (29 : ℚ) / 59
  let D := (200 : ℚ) / 399
  let E := (251 : ℚ) / 501
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end largest_fraction_l220_220188


namespace largest_possible_b_l220_220478

theorem largest_possible_b 
  (V : ℕ)
  (a b c : ℤ)
  (hV : V = 360)
  (h1 : 1 < c)
  (h2 : c < b)
  (h3 : b < a)
  (h4 : a * b * c = V) 
  : b = 12 := 
  sorry

end largest_possible_b_l220_220478


namespace abs_eq_5_iff_l220_220426

theorem abs_eq_5_iff (a : ℝ) : |a| = 5 ↔ a = 5 ∨ a = -5 :=
by
  sorry

end abs_eq_5_iff_l220_220426


namespace part1_part2_l220_220234

-- Part (1)
theorem part1 : -6 * -2 + -5 * 16 = -68 := by
  sorry

-- Part (2)
theorem part2 : -1^4 + (1 / 4) * (2 * -6 - (-4)^2) = -8 := by
  sorry

end part1_part2_l220_220234


namespace find_c_minus_a_l220_220891

variable (a b c d e : ℝ)

-- Conditions
axiom avg_ab : (a + b) / 2 = 40
axiom avg_bc : (b + c) / 2 = 60
axiom avg_de : (d + e) / 2 = 80
axiom geom_mean : (a * b * d) = (b * c * e)

theorem find_c_minus_a : c - a = 40 := by
  sorry

end find_c_minus_a_l220_220891


namespace pentagon_side_length_l220_220096

-- Define the side length of the equilateral triangle
def side_length_triangle : ℚ := 20 / 9

-- Define the perimeter of the equilateral triangle
def perimeter_triangle : ℚ := 3 * side_length_triangle

-- Define the side length of the regular pentagon
def side_length_pentagon : ℚ := 4 / 3

-- Prove that the side length of the regular pentagon has the same perimeter as the equilateral triangle
theorem pentagon_side_length (s : ℚ) (h1 : s = side_length_pentagon) :
  5 * s = perimeter_triangle :=
by
  -- Provide the solution
  sorry

end pentagon_side_length_l220_220096


namespace cats_not_liking_catnip_or_tuna_l220_220272

theorem cats_not_liking_catnip_or_tuna :
  ∀ (total_cats catnip_lovers tuna_lovers both_lovers : ℕ),
  total_cats = 80 →
  catnip_lovers = 15 →
  tuna_lovers = 60 →
  both_lovers = 10 →
  (total_cats - (catnip_lovers - both_lovers + both_lovers + tuna_lovers - both_lovers)) = 15 :=
by
  intros total_cats catnip_lovers tuna_lovers both_lovers ht hc ht hboth
  sorry

end cats_not_liking_catnip_or_tuna_l220_220272


namespace shoe_size_15_is_9point25_l220_220805

noncomputable def smallest_shoe_length (L : ℝ) := L
noncomputable def largest_shoe_length (L : ℝ) := L + 9 * (1/4 : ℝ)
noncomputable def length_ratio_condition (L : ℝ) := largest_shoe_length L = 1.30 * smallest_shoe_length L
noncomputable def shoe_length_size_15 (L : ℝ) := L + 7 * (1/4 : ℝ)

theorem shoe_size_15_is_9point25 : ∃ L : ℝ, length_ratio_condition L → shoe_length_size_15 L = 9.25 :=
by
  sorry

end shoe_size_15_is_9point25_l220_220805


namespace population_of_seventh_village_l220_220471

def village_populations : List ℕ := [803, 900, 1100, 1023, 945, 980]

def average_population : ℕ := 1000

theorem population_of_seventh_village 
  (h1 : List.length village_populations = 6)
  (h2 : 1000 * 7 = 7000)
  (h3 : village_populations.sum = 5751) : 
  7000 - village_populations.sum = 1249 := 
by {
  -- h1 ensures there's exactly 6 villages in the list
  -- h2 calculates the total population of 7 villages assuming the average population
  -- h3 calculates the sum of populations in the given list of 6 villages
  -- our goal is to show that 7000 - village_populations.sum = 1249
  -- this will be simplified in the proof
  sorry
}

end population_of_seventh_village_l220_220471


namespace second_hand_degrees_per_minute_l220_220579

theorem second_hand_degrees_per_minute (clock_gains_5_minutes_per_hour : true) :
  (360 / 60 = 6) := 
by
  sorry

end second_hand_degrees_per_minute_l220_220579


namespace exp_decreasing_range_l220_220126

theorem exp_decreasing_range (a : ℝ) :
  (∀ x : ℝ, (a-2) ^ x < (a-2) ^ (x - 1)) → 2 < a ∧ a < 3 :=
by
  sorry

end exp_decreasing_range_l220_220126


namespace range_of_a_l220_220707

open Set

variable (a : ℝ)

def P(a : ℝ) : Prop := ∀ x ∈ Icc (1 : ℝ) 2, x^2 - a ≥ 0

def Q(a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (hP : P a) (hQ : Q a) : a ≤ -2 ∨ a = 1 := sorry

end range_of_a_l220_220707


namespace KarenParagraphCount_l220_220444

theorem KarenParagraphCount :
  ∀ (num_essays num_short_ans num_paragraphs total_time essay_time short_ans_time paragraph_time : ℕ),
    (num_essays = 2) →
    (num_short_ans = 15) →
    (total_time = 240) →
    (essay_time = 60) →
    (short_ans_time = 3) →
    (paragraph_time = 15) →
    (total_time = num_essays * essay_time + num_short_ans * short_ans_time + num_paragraphs * paragraph_time) →
    num_paragraphs = 5 :=
by
  sorry

end KarenParagraphCount_l220_220444


namespace power_division_identity_l220_220959

theorem power_division_identity : (8 ^ 15) / (64 ^ 6) = 512 := by
  have h64 : 64 = 8 ^ 2 := by
    sorry
  have h_exp_rule : ∀ (a m n : ℕ), (a ^ m) ^ n = a ^ (m * n) := by
    sorry
  
  rw [h64]
  rw [h_exp_rule]
  sorry

end power_division_identity_l220_220959


namespace determinant_of_given_matrix_l220_220675

noncomputable def given_matrix : Matrix (Fin 4) (Fin 4) ℤ :=
![![1, -3, 3, 2], ![0, 5, -1, 0], ![4, -2, 1, 0], ![0, 0, 0, 6]]

theorem determinant_of_given_matrix :
  Matrix.det given_matrix = -270 := by
  sorry

end determinant_of_given_matrix_l220_220675


namespace tenth_term_arithmetic_sequence_l220_220517

def arithmetic_sequence (a1 d : ℚ) (n : ℕ) : ℚ :=
  a1 + (n - 1) * d

theorem tenth_term_arithmetic_sequence :
  arithmetic_sequence (1 / 2) (1 / 2) 10 = 5 :=
by
  sorry

end tenth_term_arithmetic_sequence_l220_220517


namespace find_crossed_out_digit_l220_220198

theorem find_crossed_out_digit (n : ℕ) (h_rev : ∀ (k : ℕ), k < n → k % 9 = 0) (remaining_sum : ℕ) 
  (crossed_sum : ℕ) (h_sum : remaining_sum + crossed_sum = 27) : 
  crossed_sum = 8 :=
by
  -- We can incorporate generating the value from digit sum here.
  sorry

end find_crossed_out_digit_l220_220198


namespace sandy_total_earnings_l220_220589

-- Define the conditions
def hourly_wage : ℕ := 15
def hours_friday : ℕ := 10
def hours_saturday : ℕ := 6
def hours_sunday : ℕ := 14

-- Define the total hours worked and total earnings
def total_hours := hours_friday + hours_saturday + hours_sunday
def total_earnings := total_hours * hourly_wage

-- State the theorem
theorem sandy_total_earnings : total_earnings = 450 := by
  sorry

end sandy_total_earnings_l220_220589


namespace right_triangle_perimeter_l220_220809

theorem right_triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) (perimeter : ℝ)
  (h1 : area = 180) 
  (h2 : leg1 = 30) 
  (h3 : (1 / 2) * leg1 * leg2 = area)
  (h4 : hypotenuse^2 = leg1^2 + leg2^2)
  (h5 : leg2 = 12) 
  (h6 : hypotenuse = 2 * Real.sqrt 261) :
  perimeter = 42 + 2 * Real.sqrt 261 :=
by
  sorry

end right_triangle_perimeter_l220_220809


namespace sharks_win_percentage_at_least_ninety_percent_l220_220923

theorem sharks_win_percentage_at_least_ninety_percent (N : ℕ) :
  let initial_games := 3
  let initial_shark_wins := 2
  let total_games := initial_games + N
  let total_shark_wins := initial_shark_wins + N
  total_shark_wins * 10 ≥ total_games * 9 ↔ N ≥ 7 :=
by
  intros
  sorry

end sharks_win_percentage_at_least_ninety_percent_l220_220923


namespace find_m_l220_220546

noncomputable def union_sets (A B : Set ℝ) : Set ℝ :=
  {x | x ∈ A ∨ x ∈ B}

theorem find_m :
  ∀ (m : ℝ),
    (A = {1, 2 ^ m}) →
    (B = {0, 2}) →
    (union_sets A B = {0, 1, 2, 8}) →
    m = 3 :=
by
  intros m hA hB hUnion
  sorry

end find_m_l220_220546


namespace arithmetic_sequence_product_l220_220286

theorem arithmetic_sequence_product {b : ℕ → ℤ} (d : ℤ) (h1 : ∀ n, b (n + 1) = b n + d)
    (h2 : b 5 * b 6 = 21) : b 4 * b 7 = -11 :=
  sorry

end arithmetic_sequence_product_l220_220286


namespace side_length_c_4_l220_220857

theorem side_length_c_4 (A : ℝ) (b S c : ℝ) 
  (hA : A = 120) (hb : b = 2) (hS : S = 2 * Real.sqrt 3) : 
  c = 4 :=
sorry

end side_length_c_4_l220_220857


namespace roots_and_a_of_polynomial_l220_220868

theorem roots_and_a_of_polynomial :
  ∀ (a : ℤ), 
  (∀ x : ℤ, x^4 - 16*x^3 + (81 - 2*a)*x^2 + (16*a - 142)*x + a^2 - 21*a + 68 = 0 → 
  (x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 7)) ↔ a = -4 :=
sorry

end roots_and_a_of_polynomial_l220_220868


namespace waiting_period_l220_220737

-- Variable declarations
variables (P : ℕ) (H : ℕ) (W : ℕ) (A : ℕ) (T : ℕ)
-- Condition declarations
variables (hp : P = 3) (hh : H = 5 * P) (ha : A = 3 * 7) (ht : T = 39)
-- Total time equation
variables (h_total : P + H + W + A = T)

-- Statement to prove
theorem waiting_period (hp : P = 3) (hh : H = 5 * P) (ha : A = 3 * 7) (ht : T = 39) (h_total : P + H + W + A = T) : 
  W = 3 :=
sorry

end waiting_period_l220_220737


namespace train_lengths_l220_220772

theorem train_lengths (L_A L_P L_B : ℕ) (speed_A_km_hr speed_B_km_hr : ℕ) (time_A_seconds : ℕ)
                      (h1 : L_P = L_A)
                      (h2 : speed_A_km_hr = 72)
                      (h3 : speed_B_km_hr = 80)
                      (h4 : time_A_seconds = 60)
                      (h5 : L_B = L_P / 2)
                      (h6 : L_A + L_P = (speed_A_km_hr * 1000 / 3600) * time_A_seconds) :
  L_A = 600 ∧ L_B = 300 :=
by
  sorry

end train_lengths_l220_220772


namespace laborer_monthly_income_l220_220163

variable (I : ℝ)

noncomputable def average_expenditure_six_months := 70 * 6
noncomputable def debt_condition := I * 6 < average_expenditure_six_months
noncomputable def expenditure_next_four_months := 60 * 4
noncomputable def total_income_next_four_months := expenditure_next_four_months + (average_expenditure_six_months - I * 6) + 30

theorem laborer_monthly_income (h1 : debt_condition I) (h2 : total_income_next_four_months I = I * 4) :
  I = 69 :=
by
  sorry

end laborer_monthly_income_l220_220163


namespace larger_triangle_perimeter_l220_220662

theorem larger_triangle_perimeter 
    (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
    (h1 : a = 6) (h2 : b = 8)
    (hypo_large : ∀ c : ℝ, c = 20) : 
    (2 * a + 2 * b + 20 = 48) :=
by {
  sorry
}

end larger_triangle_perimeter_l220_220662


namespace sum_squares_divisible_by_4_iff_even_l220_220740

theorem sum_squares_divisible_by_4_iff_even (a b c : ℕ) (ha : a % 2 = 0) (hb : b % 2 = 0) (hc : c % 2 = 0) : 
(a^2 + b^2 + c^2) % 4 = 0 ↔ 
  (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0) :=
sorry

end sum_squares_divisible_by_4_iff_even_l220_220740


namespace area_outside_smaller_squares_l220_220078

theorem area_outside_smaller_squares (side_large : ℕ) (side_small1 : ℕ) (side_small2 : ℕ)
  (no_overlap : Prop) (side_large_eq : side_large = 9)
  (side_small1_eq : side_small1 = 4)
  (side_small2_eq : side_small2 = 2) :
  (side_large * side_large - (side_small1 * side_small1 + side_small2 * side_small2)) = 61 :=
by
  sorry

end area_outside_smaller_squares_l220_220078


namespace number_of_integers_having_squares_less_than_10_million_l220_220553

theorem number_of_integers_having_squares_less_than_10_million : 
  ∃ n : ℕ, (n = 3162) ∧ (∀ k : ℕ, k ≤ 3162 → (k^2 < 10^7)) :=
by 
  sorry

end number_of_integers_having_squares_less_than_10_million_l220_220553


namespace solution_set_of_quadratic_inequality_l220_220727

theorem solution_set_of_quadratic_inequality (a b : ℝ)
  (h : ∀ x : ℝ, ax^2 + bx - 2 > 0 ↔ x ∈ Set.Ioo (-4 : ℝ) 1) :
  a + b = 2 := 
sorry

end solution_set_of_quadratic_inequality_l220_220727


namespace DE_minimal_length_in_triangle_l220_220440

noncomputable def min_length_DE (BC AC : ℝ) (angle_B : ℝ) : ℝ :=
  if BC = 5 ∧ AC = 12 ∧ angle_B = 13 then 2 * Real.sqrt 3 else sorry

theorem DE_minimal_length_in_triangle :
  min_length_DE 5 12 13 = 2 * Real.sqrt 3 :=
sorry

end DE_minimal_length_in_triangle_l220_220440


namespace radius_of_larger_circle_l220_220182

theorem radius_of_larger_circle (R1 R2 : ℝ) (α : ℝ) (h1 : α = 60) (h2 : R1 = 24) (h3 : R2 = 3 * R1) : 
  R2 = 72 := 
by
  sorry

end radius_of_larger_circle_l220_220182


namespace concatenated_natural_irrational_l220_220441

def concatenated_natural_decimal : ℝ := 0.1234567891011121314151617181920 -- and so on

theorem concatenated_natural_irrational :
  ¬ ∃ (p q : ℤ), q ≠ 0 ∧ concatenated_natural_decimal = p / q :=
sorry

end concatenated_natural_irrational_l220_220441


namespace find_starting_number_l220_220320

theorem find_starting_number (S : ℤ) (n : ℤ) (sum_eq : 10 = S) (consec_eq : S = (20 / 2) * (n + (n + 19))) : 
  n = -9 := 
by
  sorry

end find_starting_number_l220_220320


namespace Taylor_needs_14_jars_l220_220812

noncomputable def standard_jar_volume : ℕ := 60
noncomputable def big_container_volume : ℕ := 840

theorem Taylor_needs_14_jars : big_container_volume / standard_jar_volume = 14 :=
by sorry

end Taylor_needs_14_jars_l220_220812


namespace area_ratio_correct_l220_220079

noncomputable def area_ratio_of_ABC_and_GHJ : ℝ :=
  let side_length_ABC := 12
  let BD := 5
  let CE := 5
  let AF := 8
  let area_ABC := (Real.sqrt 3 / 4) * side_length_ABC ^ 2
  (1 / 74338) * area_ABC / area_ABC

theorem area_ratio_correct : area_ratio_of_ABC_and_GHJ = 1 / 74338 := by
  sorry

end area_ratio_correct_l220_220079


namespace find_c_l220_220600

theorem find_c (x : ℝ) (c : ℝ) (h1 : 3 * x + 5 = 4) (h2 : c * x + 6 = 3) : c = 9 :=
by
  sorry

end find_c_l220_220600


namespace smallest_integer_to_make_square_l220_220214

noncomputable def y : ℕ := 2^37 * 3^18 * 5^6 * 7^8

theorem smallest_integer_to_make_square : ∃ z : ℕ, z = 10 ∧ ∃ k : ℕ, (y * z) = k^2 :=
by
  sorry

end smallest_integer_to_make_square_l220_220214


namespace lines_through_three_distinct_points_l220_220884

theorem lines_through_three_distinct_points : 
  ∃ n : ℕ, n = 54 ∧ (∀ (i j k : ℕ), 1 ≤ i ∧ i ≤ 3 ∧ 1 ≤ j ∧ j ≤ 3 ∧ 1 ≤ k ∧ k ≤ 3 → 
  ∃ (a b c : ℤ), -- Direction vector (a, b, c)
  abs a ≤ 1 ∧ abs b ≤ 1 ∧ abs c ≤ 1 ∧
  ((i + a > 0 ∧ i + a ≤ 3) ∧ (j + b > 0 ∧ j + b ≤ 3) ∧ (k + c > 0 ∧ k + c ≤ 3) ∧
  (i + 2 * a > 0 ∧ i + 2 * a ≤ 3) ∧ (j + 2 * b > 0 ∧ j + 2 * b ≤ 3) ∧ (k + 2 * c > 0 ∧ k + 2 * c ≤ 3))) := 
sorry

end lines_through_three_distinct_points_l220_220884


namespace certain_number_divisibility_l220_220689

theorem certain_number_divisibility :
  ∃ k : ℕ, 3150 = 1050 * k :=
sorry

end certain_number_divisibility_l220_220689


namespace sequence_properties_l220_220409

theorem sequence_properties (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 1 = 1) →
  (∀ n : ℕ, a (n + 1) = 2 * a n + 3) →
  (∀ n : ℕ, S n = ∑ i in finset.range n, a (i + 1)) →
  (a 3 = 13) ∧
  (∀ n : ℕ, a (n + 1) + 3 = 2 * (a n + 3)) ∧
  (∀ n : ℕ, S n = 2^(n + 2) - 3 * n - 4) :=
by
  intros h1 h2 h3
  -- Proof omitted
  sorry

end sequence_properties_l220_220409


namespace g_value_at_8_l220_220745

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 6

theorem g_value_at_8 (g : ℝ → ℝ) (h1 : ∀ x : ℝ, g x = (1/216) * (x - (a^3)) * (x - (b^3)) * (x - (c^3))) 
  (h2 : g 0 = 1) 
  (h3 : ∀ a b c : ℝ, f (a) = 0 ∧ f (b) = 0 ∧ f (c) = 0) : 
  g 8 = 0 :=
sorry

end g_value_at_8_l220_220745


namespace cubical_block_weight_l220_220800

-- Given conditions
variables (s : ℝ) (volume_ratio : ℝ) (weight2 : ℝ)
variable (h : volume_ratio = 8)
variable (h_weight : weight2 = 40)

-- The problem statement
theorem cubical_block_weight (weight1 : ℝ) :
  volume_ratio * weight1 = weight2 → weight1 = 5 :=
by
  -- Assume volume ratio as 8, weight of the second cube as 40 pounds
  have h1 : volume_ratio = 8 := h
  have h2 : weight2 = 40 := h_weight
  -- sorry is here to indicate we are skipping the proof
  sorry

end cubical_block_weight_l220_220800


namespace negate_exists_l220_220928

theorem negate_exists : 
  (¬ ∃ x : ℝ, x^2 + x - 1 > 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≤ 0) :=
by sorry

end negate_exists_l220_220928


namespace problem_l220_220337

theorem problem (a b : ℤ) (ha : a = 4) (hb : b = -3) : -2 * a - b^2 + 2 * a * b = -41 := 
by
  -- Provide proof here
  sorry

end problem_l220_220337


namespace first_year_after_2022_with_digit_sum_5_l220_220439

def sum_of_digits (n : ℕ) : ℕ :=
  (toString n).foldl (λ acc c => acc + c.toNat - '0'.toNat) 0

theorem first_year_after_2022_with_digit_sum_5 :
  ∃ y : ℕ, y > 2022 ∧ sum_of_digits y = 5 ∧ ∀ z : ℕ, z > 2022 ∧ z < y → sum_of_digits z ≠ 5 :=
sorry

end first_year_after_2022_with_digit_sum_5_l220_220439


namespace max_possible_a_l220_220362

theorem max_possible_a :
  ∃ (a : ℚ), ∀ (m : ℚ), (1/3 < m ∧ m < a) →
    (∀ x : ℤ, 0 < x ∧ x ≤ 150 → (∀ y : ℤ, y ≠ m * x + 3)) ∧ a = 50/149 :=
by
  sorry

end max_possible_a_l220_220362


namespace doughnut_machine_completion_time_l220_220492

/-- The machine starts at 9:00 AM and by 12:00 PM it has finished one fourth of the day's job.
    Prove that the doughnut machine will complete the job at 9:00 PM. -/
theorem doughnut_machine_completion_time :
  ∀ (start finish : Time),
  start = (Time.mk 9 0 0) ∧
  finish = (Time.mk 12 0 0) ∧
  (finish - start).to_hours = 3 →
  start.to_hours + 12 = finish.to_hours + 9 :=
by
  intros start finish,
  intro h,
  cases h with start_eq h,
  cases h with finish_eq h,
  cases h,
  sorry

end doughnut_machine_completion_time_l220_220492


namespace weeding_planting_support_l220_220836

-- Definitions based on conditions
def initial_weeding := 31
def initial_planting := 18
def additional_support := 20

-- Let x be the number of people sent to support weeding.
variable (x : ℕ)

-- The equation to prove.
theorem weeding_planting_support :
  initial_weeding + x = 2 * (initial_planting + (additional_support - x)) :=
sorry

end weeding_planting_support_l220_220836


namespace shonda_kids_calculation_l220_220018

def number_of_kids (B E P F A : Nat) : Nat :=
  let T := B * E
  let total_people := T / P
  total_people - (F + A + 1)

theorem shonda_kids_calculation :
  (number_of_kids 15 12 9 10 7) = 2 :=
by
  unfold number_of_kids
  exact rfl

end shonda_kids_calculation_l220_220018


namespace four_digit_palindromic_perfect_square_count_l220_220645

/-- A definition of palindrome for a number. -/
def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_digits 10 in s = s.reverse

/-- A definition of four-digit numbers. -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

/-- A definition of perfect square. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The theorem statement to prove that there is exactly one four-digit perfect square that is a palindrome. -/
theorem four_digit_palindromic_perfect_square_count : ∃! n : ℕ, 
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end four_digit_palindromic_perfect_square_count_l220_220645


namespace ginger_size_l220_220823

theorem ginger_size (anna_size : ℕ) (becky_size : ℕ) (ginger_size : ℕ) 
  (h1 : anna_size = 2) 
  (h2 : becky_size = 3 * anna_size) 
  (h3 : ginger_size = 2 * becky_size - 4) : 
  ginger_size = 8 :=
by
  -- The proof is omitted, just the theorem statement is required.
  sorry

end ginger_size_l220_220823


namespace line_equation_and_inclination_l220_220086

variable (t : ℝ)
variable (x y : ℝ)
variable (α : ℝ)
variable (l : x = -3 + t ∧ y = 1 + sqrt 3 * t)

theorem line_equation_and_inclination 
  (H : l) : 
  (∃ a b c : ℝ, a = sqrt 3 ∧ b = -1 ∧ c = 3 * sqrt 3 + 1 ∧ a * x + b * y + c = 0) ∧
  α = Real.pi / 3 :=
by
  sorry

end line_equation_and_inclination_l220_220086


namespace hours_per_day_l220_220974

theorem hours_per_day 
  (H : ℕ)
  (h1 : 6 * 8 * H = 48 * H)
  (h2 : 4 * 3 * 8 = 96)
  (h3 : (48 * H) / 75 = 96 / 30) : 
  H = 5 :=
by
  sorry

end hours_per_day_l220_220974


namespace line_through_circles_l220_220119

theorem line_through_circles (D1 E1 D2 E2 : ℝ)
  (h1 : 2 * D1 - E1 + 2 = 0)
  (h2 : 2 * D2 - E2 + 2 = 0) :
  (2 * D1 - E1 + 2 = 0) ∧ (2 * D2 - E2 + 2 = 0) :=
by
  exact ⟨h1, h2⟩

end line_through_circles_l220_220119


namespace calc_x2015_l220_220906

noncomputable def f (x a : ℝ) : ℝ := x / (a * (x + 2))

theorem calc_x2015 (a x x_0 : ℝ) (x_seq : ℕ → ℝ)
  (h_unique: ∀ x, f x a = x → x = 0) 
  (h_a_val: a = 1 / 2)
  (h_f_x0: f x_0 a = 1 / 1008)
  (h_seq: ∀ n, x_seq (n + 1) = f (x_seq n) a)
  (h_x0_val: x_seq 0 = x_0):
  x_seq 2015 = 1 / 2015 :=
by
  sorry

end calc_x2015_l220_220906


namespace value_of_1_plus_i_cubed_l220_220199

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- The main statement to verify
theorem value_of_1_plus_i_cubed : (1 + i ^ 3) = (1 - i) :=
by {  
  -- Use given conditions here if needed
  sorry
}

end value_of_1_plus_i_cubed_l220_220199


namespace remainder_of_7_pow_12_mod_100_l220_220620

theorem remainder_of_7_pow_12_mod_100 : (7 ^ 12) % 100 = 1 := 
by sorry

end remainder_of_7_pow_12_mod_100_l220_220620


namespace simplify_expression_l220_220384

theorem simplify_expression (a b : ℝ) :
  ((3 * a^3 * b - 12 * a^2 * b^2 - 6 * a * b^3) / (-3 * a * b) - 4 * a * b) = (-a^2 + 2 * b^2) :=
by
  sorry

end simplify_expression_l220_220384


namespace Hillary_activities_LCM_l220_220998

theorem Hillary_activities_LCM :
  let swim := 6
  let run := 4
  let cycle := 16
  Nat.lcm (Nat.lcm swim run) cycle = 48 :=
by
  sorry

end Hillary_activities_LCM_l220_220998


namespace solve_a_b_l220_220393

theorem solve_a_b (a b : ℕ) (h₀ : 2 * a^2 = 3 * b^3) : ∃ k : ℕ, a = 18 * k^3 ∧ b = 6 * k^2 := 
sorry

end solve_a_b_l220_220393


namespace min_ratio_of_cylinder_cone_l220_220173

open Real

noncomputable def V1 (r : ℝ) : ℝ := 2 * π * r^3
noncomputable def V2 (R m r : ℝ) : ℝ := (1 / 3) * π * R^2 * m
noncomputable def geometric_constraint (R m r : ℝ) : Prop :=
  R / m = r / (sqrt ((m - r)^2 - r^2))

theorem min_ratio_of_cylinder_cone (r : ℝ) (hr : r > 0) : 
  ∃ R m, geometric_constraint R m r ∧ (V2 R m r) / (V1 r) = 4 / 3 := 
sorry

end min_ratio_of_cylinder_cone_l220_220173


namespace ratio_of_ages_l220_220278

theorem ratio_of_ages (joe_age_now james_age_now : ℕ) (h1 : joe_age_now = james_age_now + 10)
  (h2 : 2 * (joe_age_now + 8) = 3 * (james_age_now + 8)) : 
  (james_age_now + 8) / (joe_age_now + 8) = 2 / 3 := 
by
  sorry

end ratio_of_ages_l220_220278


namespace departure_of_30_tons_of_grain_l220_220135

-- Define positive as an arrival of grain.
def positive_arrival (x : ℤ) : Prop := x > 0

-- Define negative as a departure of grain.
def negative_departure (x : ℤ) : Prop := x < 0

-- The given conditions and question translated to a Lean statement.
theorem departure_of_30_tons_of_grain :
  (positive_arrival 30) → (negative_departure (-30)) :=
by
  intro pos30
  sorry

end departure_of_30_tons_of_grain_l220_220135


namespace g_of_2_l220_220029

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : x * g y = 2 * y * g x 
axiom g_of_10 : g 10 = 5

theorem g_of_2 : g 2 = 2 :=
by
    sorry

end g_of_2_l220_220029


namespace unread_pages_when_a_is_11_l220_220975

variable (a : ℕ)

def total_pages : ℕ := 250
def pages_per_day : ℕ := 15

def unread_pages_after_a_days (a : ℕ) : ℕ := total_pages - pages_per_day * a

theorem unread_pages_when_a_is_11 : unread_pages_after_a_days 11 = 85 :=
by
  sorry

end unread_pages_when_a_is_11_l220_220975


namespace least_n_condition_l220_220778

theorem least_n_condition (n : ℕ) (h1 : ∀ k : ℕ, 1 ≤ k → k ≤ n + 1 → (k ∣ n * (n - 1) → k ≠ n + 1)) : n = 4 :=
sorry

end least_n_condition_l220_220778


namespace henry_final_money_l220_220551

def initial_money : ℝ := 11.75
def received_from_relatives : ℝ := 18.50
def found_in_card : ℝ := 5.25
def spent_on_game : ℝ := 10.60
def donated_to_charity : ℝ := 3.15

theorem henry_final_money :
  initial_money + received_from_relatives + found_in_card - spent_on_game - donated_to_charity = 21.75 :=
by
  -- proof goes here
  sorry

end henry_final_money_l220_220551


namespace calculate_f_5_5_l220_220709

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom periodic_condition (x : ℝ) (h₂ : 2 ≤ x ∧ x ≤ 3) : f (x + 2) = -1 / f x
axiom defined_segment (x : ℝ) (h₂ : 2 ≤ x ∧ x ≤ 3) : f x = x

theorem calculate_f_5_5 : f 5.5 = 2.5 := sorry

end calculate_f_5_5_l220_220709


namespace monthly_rent_l220_220357

-- Definitions based on the given conditions
def length_ft : ℕ := 360
def width_ft : ℕ := 1210
def sq_feet_per_acre : ℕ := 43560
def cost_per_acre_per_month : ℕ := 60

-- Statement of the problem
theorem monthly_rent : (length_ft * width_ft / sq_feet_per_acre) * cost_per_acre_per_month = 600 := sorry

end monthly_rent_l220_220357


namespace initial_blue_balls_l220_220174

theorem initial_blue_balls (B : ℕ) (h1 : 25 - 5 = 20) (h2 : (B - 5) / 20 = 1 / 5) : B = 9 :=
by
  sorry

end initial_blue_balls_l220_220174


namespace unique_four_digit_palindromic_square_l220_220652

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem unique_four_digit_palindromic_square :
  ∃! n : ℕ, is_palindrome n ∧ is_four_digit n ∧ ∃ m : ℕ, m^2 = n :=
sorry

end unique_four_digit_palindromic_square_l220_220652


namespace tan_sum_identity_l220_220233

theorem tan_sum_identity :
  Real.tan (15 * Real.pi / 180) + Real.tan (30 * Real.pi / 180) + 
  Real.tan (15 * Real.pi / 180) * Real.tan (30 * Real.pi / 180) = 1 :=
by sorry

end tan_sum_identity_l220_220233


namespace ball_more_than_bat_l220_220023

theorem ball_more_than_bat :
  ∃ x y : ℕ, (2 * x + 3 * y = 1300) ∧ (3 * x + 2 * y = 1200) ∧ (y - x = 100) :=
by
  sorry

end ball_more_than_bat_l220_220023


namespace simplified_expression_result_l220_220850

theorem simplified_expression_result :
  ((2 + 3 + 6 + 7) / 3) + ((3 * 6 + 9) / 4) = 12.75 := 
by {
  sorry
}

end simplified_expression_result_l220_220850


namespace baseball_cards_total_l220_220193

theorem baseball_cards_total (num_friends : ℕ) (cards_per_friend : ℕ) (total_cards : ℕ)
  (h1 : num_friends = 5) (h2 : cards_per_friend = 91) :
  total_cards = 455 :=
by {
  rw [h1, h2],
  have : total_cards = 5 * 91,
  { sorry },
  rw this,
  exact rfl,
}

end baseball_cards_total_l220_220193


namespace sequence_with_limit_is_bounded_bounded_sequence_does_not_imply_limit_l220_220196

-- Part a) Prove that if a sequence has a limit, then it is bounded.
theorem sequence_with_limit_is_bounded (x : ℕ → ℝ) (x0 : ℝ) (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - x0| < ε) :
  ∃ C, ∀ n, |x n| ≤ C := by
  sorry

-- Part b) Is the converse statement true?
theorem bounded_sequence_does_not_imply_limit :
  ∃ (x : ℕ → ℝ), (∃ C, ∀ n, |x n| ≤ C) ∧ ¬(∃ x0, ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - x0| < ε) := by
  sorry

end sequence_with_limit_is_bounded_bounded_sequence_does_not_imply_limit_l220_220196


namespace largest_possible_perimeter_l220_220371

theorem largest_possible_perimeter (y : ℤ) (hy1 : 3 ≤ y) (hy2 : y < 16) : 7 + 9 + y ≤ 31 := 
by
  sorry

end largest_possible_perimeter_l220_220371


namespace sandy_earnings_correct_l220_220587

def hourly_rate : ℕ := 15
def hours_worked_friday : ℕ := 10
def hours_worked_saturday : ℕ := 6
def hours_worked_sunday : ℕ := 14

def earnings_friday : ℕ := hours_worked_friday * hourly_rate
def earnings_saturday : ℕ := hours_worked_saturday * hourly_rate
def earnings_sunday : ℕ := hours_worked_sunday * hourly_rate

def total_earnings : ℕ := earnings_friday + earnings_saturday + earnings_sunday

theorem sandy_earnings_correct : total_earnings = 450 := by
  sorry

end sandy_earnings_correct_l220_220587


namespace inequality_inequality_hold_l220_220283

theorem inequality_inequality_hold (k : ℕ) (x y z : ℝ) 
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) 
  (h_sum : x + y + z = 1) : 
  (x ^ (k + 2) / (x ^ (k + 1) + y ^ k + z ^ k) 
  + y ^ (k + 2) / (y ^ (k + 1) + z ^ k + x ^ k) 
  + z ^ (k + 2) / (z ^ (k + 1) + x ^ k + y ^ k)) 
  ≥ (1 / 7) :=
sorry

end inequality_inequality_hold_l220_220283


namespace taxi_speed_l220_220367

theorem taxi_speed (v : ℝ) (hA : ∀ v : ℝ, 3 * v = 6 * (v - 30)) : v = 60 :=
by
  sorry

end taxi_speed_l220_220367


namespace num_of_valid_m_vals_l220_220429

theorem num_of_valid_m_vals : 
  (∀ m x : ℤ, (x + m ≤ 4 ∧ (x / 2 - (x - 1) / 4 > 1 → x > 3 → ∃ (c : ℚ), (x + 1)/4 > 1 )) ∧
  (∃ (x : ℤ), (x + m ≤ 4 ∧ (x > 3) ∧ (m < 1 ∧ m > -4)) ∧ 
  ∃ a b : ℚ, x^2 + a * x + b = 0) → 
  (∃ (count m : ℤ), count = 2)) :=
sorry

end num_of_valid_m_vals_l220_220429


namespace marble_weight_l220_220581

theorem marble_weight (m d : ℝ) : (9 * m = 4 * d) → (3 * d = 36) → (m = 16 / 3) :=
by
  intro h1 h2
  sorry

end marble_weight_l220_220581


namespace remaining_budget_is_correct_l220_220803

def budget := 750
def flasks_cost := 200
def test_tubes_cost := (2 / 3) * flasks_cost
def safety_gear_cost := (1 / 2) * test_tubes_cost
def chemicals_cost := (3 / 4) * flasks_cost
def instruments_min_cost := 50

def total_spent := flasks_cost + test_tubes_cost + safety_gear_cost + chemicals_cost
def remaining_budget_before_instruments := budget - total_spent
def remaining_budget_after_instruments := remaining_budget_before_instruments - instruments_min_cost

theorem remaining_budget_is_correct :
  remaining_budget_after_instruments = 150 := by
  unfold remaining_budget_after_instruments remaining_budget_before_instruments total_spent flasks_cost test_tubes_cost safety_gear_cost chemicals_cost budget
  sorry

end remaining_budget_is_correct_l220_220803


namespace contains_all_integers_l220_220282

def is_closed_under_divisors (A : Set ℕ) : Prop :=
  ∀ {a b : ℕ}, b ∣ a → a ∈ A → b ∈ A

def contains_product_plus_one (A : Set ℕ) : Prop :=
  ∀ {a b : ℕ}, 1 < a → a < b → a ∈ A → b ∈ A → (1 + a * b) ∈ A

theorem contains_all_integers
  (A : Set ℕ)
  (h1 : is_closed_under_divisors A)
  (h2 : contains_product_plus_one A)
  (h3 : ∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ 1 < a ∧ 1 < b ∧ 1 < c) :
  ∀ n : ℕ, n > 0 → n ∈ A := 
  by 
    sorry

end contains_all_integers_l220_220282


namespace largest_possible_perimeter_l220_220372

theorem largest_possible_perimeter (y : ℤ) (hy1 : 3 ≤ y) (hy2 : y < 16) : 7 + 9 + y ≤ 31 := 
by
  sorry

end largest_possible_perimeter_l220_220372


namespace hallie_read_pages_third_day_more_than_second_day_l220_220418

theorem hallie_read_pages_third_day_more_than_second_day :
  ∀ (d1 d2 d3 d4 : ℕ),
  d1 = 63 →
  d2 = 2 * d1 →
  d4 = 29 →
  d1 + d2 + d3 + d4 = 354 →
  (d3 - d2) = 10 :=
by
  intros d1 d2 d3 d4 h1 h2 h4 h_sum
  sorry

end hallie_read_pages_third_day_more_than_second_day_l220_220418


namespace probability_reach_correct_l220_220806

noncomputable def probability_reach (n : ℕ) : ℚ :=
  (2/3) + (1/12) * (1 - (-1/3)^(n-1))

theorem probability_reach_correct (n : ℕ) (P_n : ℚ) :
  P_n = probability_reach n :=
by
  sorry

end probability_reach_correct_l220_220806


namespace part1_part2_l220_220712

-- Define the universal set U as real numbers ℝ
def U : Set ℝ := Set.univ

-- Define Set A
def A (a : ℝ) : Set ℝ := {x | abs (x - a) ≤ 1 }

-- Define Set B
def B : Set ℝ := {x | (4 - x) * (x - 1) ≤ 0 }

-- Part 1: Prove A ∪ B when a = 4
theorem part1 : A 4 ∪ B = {x : ℝ | x ≥ 3 ∨ x ≤ 1} :=
sorry

-- Part 2: Prove the range of values for a given A ∩ B = A
theorem part2 (a : ℝ) (h : A a ∩ B = A a) : a ≥ 5 ∨ a ≤ 0 :=
sorry

end part1_part2_l220_220712


namespace angle_DEF_EDF_proof_l220_220358

theorem angle_DEF_EDF_proof (angle_DOE : ℝ) (angle_EOD : ℝ) 
  (h1 : angle_DOE = 130) (h2 : angle_EOD = 90) :
  let angle_DEF := 45
  let angle_EDF := 45
  angle_DEF = 45 ∧ angle_EDF = 45 :=
by
  sorry

end angle_DEF_EDF_proof_l220_220358


namespace evaluate_expression_l220_220861

variable {a b c : ℝ}

theorem evaluate_expression
  (h : a / (35 - a) + b / (75 - b) + c / (85 - c) = 5) :
  7 / (35 - a) + 15 / (75 - b) + 17 / (85 - c) = 8 / 5 := by
  sorry

end evaluate_expression_l220_220861


namespace four_digit_palindromic_perfect_square_count_l220_220644

/-- A definition of palindrome for a number. -/
def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_digits 10 in s = s.reverse

/-- A definition of four-digit numbers. -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

/-- A definition of perfect square. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The theorem statement to prove that there is exactly one four-digit perfect square that is a palindrome. -/
theorem four_digit_palindromic_perfect_square_count : ∃! n : ℕ, 
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end four_digit_palindromic_perfect_square_count_l220_220644


namespace system_of_equations_has_solution_l220_220107

theorem system_of_equations_has_solution (m : ℝ) : 
  ∃ (x y : ℝ), y = 3 * m * x + 4 ∧ y = (3 * m - 1) * x + 3 :=
by
  sorry

end system_of_equations_has_solution_l220_220107


namespace reema_simple_interest_l220_220159

-- Definitions and conditions
def principal : ℕ := 1200
def rate_of_interest : ℕ := 6
def time_period : ℕ := rate_of_interest

-- Simple interest calculation
def calculate_simple_interest (P R T: ℕ) : ℕ :=
  (P * R * T) / 100

-- The theorem to prove that Reema paid Rs 432 as simple interest.
theorem reema_simple_interest : calculate_simple_interest principal rate_of_interest time_period = 432 := 
  sorry

end reema_simple_interest_l220_220159


namespace max_value_of_f_l220_220678

-- Define the function f(x) = 5x - x^2
def f (x : ℝ) : ℝ := 5 * x - x^2

-- The theorem we want to prove, stating the maximum value of f(x) is 6.25
theorem max_value_of_f : ∃ x, f x = 6.25 :=
by
  -- Placeholder proof, to be completed
  sorry

end max_value_of_f_l220_220678


namespace second_number_value_l220_220699

-- Definition of the problem conditions
variables (x y z : ℝ)
axiom h1 : z = 4.5 * y
axiom h2 : y = 2.5 * x
axiom h3 : (x + y + z) / 3 = 165

-- The goal is to prove y = 82.5 given the conditions h1, h2, and h3
theorem second_number_value : y = 82.5 :=
by
  sorry

end second_number_value_l220_220699


namespace same_terminal_side_angle_l220_220568

theorem same_terminal_side_angle (θ : ℤ) : θ = -390 → ∃ k : ℤ, 0 ≤ θ + k * 360 ∧ θ + k * 360 < 360 ∧ θ + k * 360 = 330 :=
  by
    sorry

end same_terminal_side_angle_l220_220568


namespace find_number_l220_220680

theorem find_number (x : ℝ) (h : 54 / 2 + 3 * x = 75) : x = 16 :=
by
  sorry

end find_number_l220_220680


namespace book_pages_l220_220190

-- Define the conditions
def pages_per_night : ℕ := 12
def nights : ℕ := 10

-- Define the total pages in the book
def total_pages : ℕ := pages_per_night * nights

-- Prove that the total number of pages is 120
theorem book_pages : total_pages = 120 := by
  sorry

end book_pages_l220_220190


namespace one_fourth_of_6_8_is_fraction_l220_220099

theorem one_fourth_of_6_8_is_fraction :
  (6.8 / 4 : ℚ) = 17 / 10 :=
sorry

end one_fourth_of_6_8_is_fraction_l220_220099


namespace solve_for_x_l220_220889

theorem solve_for_x (x : ℝ) (h : (2 * x + 7) / 6 = 13) : x = 35.5 :=
by
  -- Proof steps would go here
  sorry

end solve_for_x_l220_220889


namespace find_f_8_5_l220_220908

-- Conditions as definitions in Lean
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x
def segment_function (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 3 * x

-- The main theorem to prove
theorem find_f_8_5 (f : ℝ → ℝ) (h1 : even_function f) (h2 : periodic_function f 3) (h3 : segment_function f)
: f 8.5 = 1.5 :=
sorry

end find_f_8_5_l220_220908


namespace find_a_given_integer_roots_l220_220465

-- Given polynomial equation and the condition of integer roots
theorem find_a_given_integer_roots (a : ℤ) :
    (∃ x y : ℤ, x ≠ y ∧ (x^2 - (a+8)*x + 8*a - 1 = 0) ∧ (y^2 - (a+8)*y + 8*a - 1 = 0)) → 
    a = 8 := 
by
  sorry

end find_a_given_integer_roots_l220_220465


namespace inequality_solution_l220_220462

-- Define the condition for the denominator being positive
def denom_positive (x : ℝ) : Prop :=
  x^2 + 2*x + 7 > 0

-- Statement of the problem
theorem inequality_solution (x : ℝ) (h : denom_positive x) :
  (x + 6) / (x^2 + 2*x + 7) ≥ 0 ↔ x ≥ -6 :=
sorry

end inequality_solution_l220_220462


namespace relation_between_x_and_y_l220_220424

variable (t : ℝ)
variable (x : ℝ := t ^ (2 / (t - 1))) (y : ℝ := t ^ ((t + 1) / (t - 1)))

theorem relation_between_x_and_y (h1 : t > 0) (h2 : t ≠ 1) : y ^ (1 / x) = x ^ y :=
by sorry

end relation_between_x_and_y_l220_220424


namespace four_digit_perfect_square_palindrome_count_l220_220647

def is_palindrome (n : ℕ) : Prop :=
  let str := n.repr
  str = str.reverse

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n
  
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_four_digit_perfect_square_palindrome (n : ℕ) : Prop :=
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n

theorem four_digit_perfect_square_palindrome_count : 
  { n : ℕ | is_four_digit_perfect_square_palindrome n }.card = 4 := 
sorry

end four_digit_perfect_square_palindrome_count_l220_220647


namespace value_of_a_plus_b_l220_220036

theorem value_of_a_plus_b (a b c : ℤ) 
    (h1 : a + b + c = 11)
    (h2 : a + b - c = 19)
    : a + b = 15 := 
by
    -- Mathematical details skipped
    sorry

end value_of_a_plus_b_l220_220036


namespace geometric_sequence_max_product_l220_220567

theorem geometric_sequence_max_product
  (b : ℕ → ℝ) (q : ℝ) (b1 : ℝ)
  (h_b1_pos : b1 > 0)
  (h_q : 0 < q ∧ q < 1)
  (h_b : ∀ n, b (n + 1) = b n * q)
  (h_b7_gt_1 : b 7 > 1)
  (h_b8_lt_1 : b 8 < 1) :
  (∀ (n : ℕ), n = 7 → b 1 * b 2 * b 3 * b 4 * b 5 * b 6 * b 7 = b 1 * b 2 * b 3 * b 4 * b 5 * b 6 * b 7) :=
by {
  sorry
}

end geometric_sequence_max_product_l220_220567


namespace paths_from_A_to_B_no_revisits_l220_220883

noncomputable def numPaths : ℕ :=
  16

theorem paths_from_A_to_B_no_revisits : numPaths = 16 :=
by
  sorry

end paths_from_A_to_B_no_revisits_l220_220883


namespace odd_function_behavior_on_interval_l220_220726

theorem odd_function_behavior_on_interval
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ x₁ x₂, 1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 4 → f x₁ < f x₂)
  (h_max : ∀ x, 1 ≤ x → x ≤ 4 → f x ≤ 5) :
  (∀ x, -4 ≤ x → x ≤ -1 → f (-4) ≤ f x ∧ f x ≤ f (-1)) ∧ f (-4) = -5 :=
sorry

end odd_function_behavior_on_interval_l220_220726


namespace sum_due_is_363_l220_220060

/-
Conditions:
1. BD = 78
2. TD = 66
3. The formula: BD = TD + (TD^2 / PV)
This should imply that PV = 363 given the conditions.
-/

theorem sum_due_is_363 (BD TD PV : ℝ) (h1 : BD = 78) (h2 : TD = 66) (h3 : BD = TD + (TD^2 / PV)) : PV = 363 :=
by
  sorry

end sum_due_is_363_l220_220060


namespace _l220_220735

noncomputable def X := 0
noncomputable def Y := 1
noncomputable def Z := 2

noncomputable def angle_XYZ (X Y Z : ℝ) : ℝ := 90 -- Triangle XYZ where ∠X = 90°

noncomputable def length_YZ := 10 -- YZ = 10 units
noncomputable def length_XY := 6 -- XY = 6 units
noncomputable def length_XZ : ℝ := Real.sqrt (length_YZ^2 - length_XY^2) -- Pythagorean theorem to find XZ
noncomputable def cos_Z : ℝ := length_XZ / length_YZ -- cos Z = adjacent/hypotenuse

example : cos_Z = 0.8 :=
by {
  sorry
}

end _l220_220735


namespace brian_oranges_is_12_l220_220827

-- Define the number of oranges the person has
def person_oranges : Nat := 12

-- Define the number of oranges Brian has, which is zero fewer than the person's oranges
def brian_oranges : Nat := person_oranges - 0

-- The theorem stating that Brian has 12 oranges
theorem brian_oranges_is_12 : brian_oranges = 12 :=
by
  -- Proof is omitted
  sorry

end brian_oranges_is_12_l220_220827


namespace circle_areas_equal_l220_220123

theorem circle_areas_equal :
  let r1 := 15
  let d2 := 30
  let r2 := d2 / 2
  let A1 := Real.pi * r1^2
  let A2 := Real.pi * r2^2
  A1 = A2 :=
by
  sorry

end circle_areas_equal_l220_220123


namespace three_planes_max_division_l220_220614

-- Define the condition: three planes
variable (P1 P2 P3 : Plane)

-- Define the proof statement: three planes can divide the space into at most 8 parts
theorem three_planes_max_division : divides_space_at_most P1 P2 P3 8 :=
  sorry

end three_planes_max_division_l220_220614


namespace sum_of_squared_projections_l220_220403

theorem sum_of_squared_projections (a l m n : ℝ) (l_proj m_proj n_proj : ℝ)
  (h : l_proj = a * Real.cos θ)
  (h1 : m_proj = a * Real.cos (Real.pi / 3 - θ))
  (h2 : n_proj = a * Real.cos (Real.pi / 3 + θ)) :
  l_proj ^ 2 + m_proj ^ 2 + n_proj ^ 2 = 3 / 2 * a ^ 2 :=
by sorry

end sum_of_squared_projections_l220_220403


namespace remainder_equality_l220_220573

theorem remainder_equality
  (P P' K D R R' r r' : ℕ)
  (h1 : P > P')
  (h2 : P % K = 0)
  (h3 : P' % K = 0)
  (h4 : P % D = R)
  (h5 : P' % D = R')
  (h6 : (P * K - P') % D = r)
  (h7 : (R * K - R') % D = r') :
  r = r' :=
sorry

end remainder_equality_l220_220573


namespace percentage_wearing_blue_shirts_l220_220893

theorem percentage_wearing_blue_shirts (total_students : ℕ) (red_percentage green_percentage : ℕ) 
  (other_students : ℕ) (H1 : total_students = 900) (H2 : red_percentage = 28) 
  (H3 : green_percentage = 10) (H4 : other_students = 162) : 
  (44 : ℕ) = 100 - (red_percentage + green_percentage + (other_students * 100 / total_students)) :=
by
  sorry

end percentage_wearing_blue_shirts_l220_220893


namespace chair_arrangements_48_l220_220729

theorem chair_arrangements_48 :
  ∃ (n : ℕ), n = 8 ∧ (∀ (r c : ℕ), r * c = 48 → 2 ≤ r ∧ 2 ≤ c) := 
sorry

end chair_arrangements_48_l220_220729


namespace Paul_dig_days_alone_l220_220442

/-- Jake's daily work rate -/
def Jake_work_rate : ℚ := 1 / 16

/-- Hari's daily work rate -/
def Hari_work_rate : ℚ := 1 / 48

/-- Combined work rate of Jake, Paul, and Hari, when they work together they can dig the well in 8 days -/
def combined_work_rate (Paul_work_rate : ℚ) : Prop :=
  Jake_work_rate + Paul_work_rate + Hari_work_rate = 1 / 8

/-- Theorem stating that Paul can dig the well alone in 24 days -/
theorem Paul_dig_days_alone : ∃ (P : ℚ), combined_work_rate (1 / P) ∧ P = 24 :=
by
  use 24
  unfold combined_work_rate
  sorry

end Paul_dig_days_alone_l220_220442


namespace uniform_prob_correct_l220_220807

noncomputable def uniform_prob_within_interval 
  (α β γ δ : ℝ) 
  (h₁ : α ≤ β) 
  (h₂ : α ≤ γ) 
  (h₃ : γ < δ) 
  (h₄ : δ ≤ β) : ℝ :=
  (δ - γ) / (β - α)

theorem uniform_prob_correct 
  (α β γ δ : ℝ) 
  (hαβ : α ≤ β) 
  (hαγ : α ≤ γ) 
  (hγδ : γ < δ) 
  (hδβ : δ ≤ β) :
  uniform_prob_within_interval α β γ δ hαβ hαγ hγδ hδβ = (δ - γ) / (β - α) := sorry

end uniform_prob_correct_l220_220807


namespace probability_C_and_D_l220_220792

theorem probability_C_and_D (P_A P_B : ℚ) (H1 : P_A = 1/4) (H2 : P_B = 1/3) :
  P_C + P_D = 5/12 :=
by
  sorry

end probability_C_and_D_l220_220792


namespace balls_in_boxes_l220_220267

theorem balls_in_boxes : ∀ (balls boxes : ℕ), balls = 7 → boxes = 4 → 
  (number_of_unique_distributions balls boxes = 11) := by
  intros balls boxes hb hc
  subst hb
  subst hc
  sorry

end balls_in_boxes_l220_220267


namespace four_digit_palindrome_square_count_l220_220649

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem four_digit_palindrome_square_count : 
  ∃! (n : ℕ), is_four_digit n ∧ is_palindrome n ∧ (∃ (m : ℕ), n = m * m) := by
  sorry

end four_digit_palindrome_square_count_l220_220649


namespace races_needed_to_declare_winner_l220_220731

noncomputable def total_sprinters : ℕ := 275
noncomputable def sprinters_per_race : ℕ := 7
noncomputable def sprinters_advance : ℕ := 2
noncomputable def sprinters_eliminated : ℕ := 5

theorem races_needed_to_declare_winner :
  (total_sprinters - 1 + sprinters_eliminated) / sprinters_eliminated = 59 :=
by
  sorry

end races_needed_to_declare_winner_l220_220731


namespace solve_for_r_l220_220306

theorem solve_for_r (r : ℤ) : 24 - 5 = 3 * r + 7 → r = 4 :=
by
  intro h
  sorry

end solve_for_r_l220_220306


namespace complete_square_eq_l220_220966

theorem complete_square_eq (x : ℝ) : x^2 - 2 * x - 5 = 0 → (x - 1)^2 = 6 :=
by
  intro h
  have : x^2 - 2 * x = 5 := by linarith
  have : x^2 - 2 * x + 1 = 6 := by linarith
  exact eq_of_sub_eq_zero (by linarith)

end complete_square_eq_l220_220966


namespace palindromic_squares_count_l220_220656

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_palindromic_squares : ℕ :=
  (List.range' 32 (99 - 32 + 1)).count (λ n, 
    let sq := n * n in is_four_digit sq ∧ is_palindrome sq)

theorem palindromic_squares_count : count_palindromic_squares = 2 := by
  sorry

end palindromic_squares_count_l220_220656


namespace second_machine_time_l220_220502

theorem second_machine_time
  (machine1_rate : ℕ)
  (machine2_rate : ℕ)
  (combined_rate12 : ℕ)
  (combined_rate123 : ℕ)
  (rate3 : ℕ)
  (time3 : ℚ) :
  machine1_rate = 60 →
  machine2_rate = 120 →
  combined_rate12 = 200 →
  combined_rate123 = 600 →
  rate3 = 420 →
  time3 = 10 / 7 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_machine_time_l220_220502


namespace irreducible_fraction_l220_220918

theorem irreducible_fraction (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 :=
by
  sorry

end irreducible_fraction_l220_220918


namespace hoseoks_social_studies_score_l220_220719

theorem hoseoks_social_studies_score 
  (avg_three_subjects : ℕ) 
  (new_avg_with_social_studies : ℕ) 
  (total_score_three_subjects : ℕ) 
  (total_score_four_subjects : ℕ) 
  (S : ℕ)
  (h1 : avg_three_subjects = 89) 
  (h2 : new_avg_with_social_studies = 90) 
  (h3 : total_score_three_subjects = 3 * avg_three_subjects) 
  (h4 : total_score_four_subjects = 4 * new_avg_with_social_studies) :
  S = 93 :=
sorry

end hoseoks_social_studies_score_l220_220719


namespace smallest_b_l220_220017

theorem smallest_b (a b : ℝ) (h1 : 2 < a) (h2 : a < b) (h3 : a + b = 7) (h4 : 2 + a ≤ b) : b = 9 / 2 :=
by
  sorry

end smallest_b_l220_220017


namespace mark_remaining_money_l220_220578

theorem mark_remaining_money 
  (initial_money : ℕ) (num_books : ℕ) (cost_per_book : ℕ) (total_cost : ℕ) (remaining_money : ℕ) 
  (H1 : initial_money = 85)
  (H2 : num_books = 10)
  (H3 : cost_per_book = 5)
  (H4 : total_cost = num_books * cost_per_book)
  (H5 : remaining_money = initial_money - total_cost) : 
  remaining_money = 35 := 
by
  sorry

end mark_remaining_money_l220_220578


namespace emily_collected_8484_eggs_l220_220390

def number_of_baskets : ℕ := 303
def eggs_per_basket : ℕ := 28
def total_eggs : ℕ := number_of_baskets * eggs_per_basket

theorem emily_collected_8484_eggs : total_eggs = 8484 :=
by
  sorry

end emily_collected_8484_eggs_l220_220390


namespace palindromic_squares_count_l220_220657

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_palindromic_squares : ℕ :=
  (List.range' 32 (99 - 32 + 1)).count (λ n, 
    let sq := n * n in is_four_digit sq ∧ is_palindrome sq)

theorem palindromic_squares_count : count_palindromic_squares = 2 := by
  sorry

end palindromic_squares_count_l220_220657


namespace batsman_average_after_12th_innings_l220_220494

theorem batsman_average_after_12th_innings (A : ℕ) (total_runs_11 : ℕ) (total_runs_12 : ℕ ) : 
  total_runs_11 = 11 * A → 
  total_runs_12 = total_runs_11 + 55 → 
  (total_runs_12 / 12 = A + 1) → 
  (A + 1) = 44 := 
by
  intros h1 h2 h3
  sorry

end batsman_average_after_12th_innings_l220_220494


namespace probability_no_shaded_square_l220_220206

theorem probability_no_shaded_square :
  let num_rects := 1003 * 2005
  let num_rects_with_shaded := 1002^2
  let probability_no_shaded := 1 - (num_rects_with_shaded / num_rects)
  probability_no_shaded = 1 / 1003 := by
  -- The proof steps go here
  sorry

end probability_no_shaded_square_l220_220206


namespace prob_of_different_colors_l220_220309

def total_balls_A : ℕ := 4 + 5 + 6
def total_balls_B : ℕ := 7 + 6 + 2

noncomputable def prob_same_color : ℚ :=
  (4 / ↑total_balls_A * 7 / ↑total_balls_B) +
  (5 / ↑total_balls_A * 6 / ↑total_balls_B) +
  (6 / ↑total_balls_A * 2 / ↑total_balls_B)

noncomputable def prob_different_color : ℚ :=
  1 - prob_same_color

theorem prob_of_different_colors :
  prob_different_color = 31 / 45 :=
by
  sorry

end prob_of_different_colors_l220_220309


namespace greatest_sum_consecutive_integers_l220_220332

theorem greatest_sum_consecutive_integers (n : ℤ) (h : n * (n + 1) < 360) : n + (n + 1) ≤ 37 := by
  sorry

end greatest_sum_consecutive_integers_l220_220332


namespace power_division_identity_l220_220960

theorem power_division_identity : (8 ^ 15) / (64 ^ 6) = 512 := by
  have h64 : 64 = 8 ^ 2 := by
    sorry
  have h_exp_rule : ∀ (a m n : ℕ), (a ^ m) ^ n = a ^ (m * n) := by
    sorry
  
  rw [h64]
  rw [h_exp_rule]
  sorry

end power_division_identity_l220_220960


namespace value_of_m_l220_220927

theorem value_of_m (m : ℝ) (x : ℝ) (hx : x ≠ 0) :
  (∃ (k : ℝ), (2 * m - 1) * x ^ (m ^ 2) = k * x ^ n) → m = 1 :=
by
  sorry

end value_of_m_l220_220927


namespace n_cubed_plus_5n_divisible_by_6_l220_220458

theorem n_cubed_plus_5n_divisible_by_6 (n : ℕ) : ∃ k : ℤ, n^3 + 5 * n = 6 * k :=
by
  sorry

end n_cubed_plus_5n_divisible_by_6_l220_220458


namespace relationship_between_heights_is_correlated_l220_220606

theorem relationship_between_heights_is_correlated :
  (∃ r : ℕ, (r = 1 ∨ r = 2 ∨ r = 3 ∨ r = 4) ∧ r = 2) := by
  sorry

end relationship_between_heights_is_correlated_l220_220606


namespace incorrect_transformation_D_l220_220055

theorem incorrect_transformation_D (x y m : ℝ) (hxy: x = y) : m = 0 → ¬ (x / m = y / m) :=
by
  intro hm
  simp [hm]
  -- Lean's simp tactic simplifies known equalities
  -- The simp tactic will handle the contradiction case directly when m = 0.
  sorry

end incorrect_transformation_D_l220_220055


namespace phase_shift_of_sine_l220_220688

theorem phase_shift_of_sine :
  let B := 5
  let C := (3 * Real.pi) / 2
  let phase_shift := C / B
  phase_shift = (3 * Real.pi) / 10 := by
    sorry

end phase_shift_of_sine_l220_220688


namespace sqrt_two_irrational_l220_220485

theorem sqrt_two_irrational :
  ¬ ∃ (a b : ℕ), (a.gcd b = 1) ∧ (b ≠ 0) ∧ (a^2 = 2 * b^2) :=
sorry

end sqrt_two_irrational_l220_220485


namespace first_train_length_correct_l220_220937

noncomputable def length_of_first_train : ℝ :=
  let speed_first_train := 90 * 1000 / 3600  -- converting to m/s
  let speed_second_train := 72 * 1000 / 3600 -- converting to m/s
  let relative_speed := speed_first_train + speed_second_train
  let distance_apart := 630
  let length_second_train := 200
  let time_to_meet := 13.998880089592832
  let distance_covered := relative_speed * time_to_meet
  let total_distance := distance_apart
  let length_first_train := total_distance - length_second_train
  length_first_train

theorem first_train_length_correct :
  length_of_first_train = 430 :=
by
  -- Place for the proof steps
  sorry

end first_train_length_correct_l220_220937


namespace find_weight_first_dog_l220_220322

noncomputable def weight_first_dog (x : ℕ) (y : ℕ) : Prop :=
  (x + 31 + 35 + 33) / 4 = (x + 31 + 35 + 33 + y) / 5

theorem find_weight_first_dog (x : ℕ) : weight_first_dog x 31 → x = 25 := by
  sorry

end find_weight_first_dog_l220_220322


namespace range_of_alpha_l220_220013

variable {x : ℝ}

noncomputable def curve (x : ℝ) : ℝ := x^3 - x + 2

theorem range_of_alpha (x : ℝ) (α : ℝ) (h : α = Real.arctan (3*x^2 - 1)) :
  α ∈ Set.Ico 0 (Real.pi / 2) ∪ Set.Ico (3 * Real.pi / 4) Real.pi :=
sorry

end range_of_alpha_l220_220013


namespace non_formable_triangle_sticks_l220_220815

theorem non_formable_triangle_sticks 
  (sticks : Fin 8 → ℕ) 
  (h_no_triangle : ∀ (i j k : Fin 8), i < j → j < k → sticks i + sticks j ≤ sticks k) : 
  ∃ (max_length : ℕ), (max_length = sticks (Fin.mk 7 (by norm_num))) ∧ max_length = 21 := 
by 
  sorry

end non_formable_triangle_sticks_l220_220815


namespace inequality_sum_of_reciprocals_l220_220145

variable {a b c : ℝ}

theorem inequality_sum_of_reciprocals
  (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c)
  (hsum : a + b + c = 3) :
  (1 / (2 * a^2 + b^2 + c^2) + 1 / (2 * b^2 + c^2 + a^2) + 1 / (2 * c^2 + a^2 + b^2)) ≤ 3/4 :=
sorry

end inequality_sum_of_reciprocals_l220_220145


namespace price_difference_l220_220801

/-- Given an original price, two successive price increases, and special deal prices for a fixed number of items, 
    calculate the difference between the final retail price and the average special deal price. -/
theorem price_difference
  (original_price : ℝ) (first_increase_percent: ℝ) (second_increase_percent: ℝ)
  (special_deal_percent_1: ℝ) (num_items_1: ℕ) (special_deal_percent_2: ℝ) (num_items_2: ℕ)
  (final_retail_price : ℝ) (average_special_deal_price : ℝ) :
  original_price = 50 →
  first_increase_percent = 0.30 →
  second_increase_percent = 0.15 →
  special_deal_percent_1 = 0.70 →
  num_items_1 = 50 →
  special_deal_percent_2 = 0.85 →
  num_items_2 = 100 →
  final_retail_price = original_price * (1 + first_increase_percent) * (1 + second_increase_percent) →
  average_special_deal_price = 
    (num_items_1 * (special_deal_percent_1 * final_retail_price) + 
    num_items_2 * (special_deal_percent_2 * final_retail_price)) / 
    (num_items_1 + num_items_2) →
  final_retail_price - average_special_deal_price = 14.95 :=
by
  intros
  sorry

end price_difference_l220_220801


namespace xyz_inequality_l220_220746

theorem xyz_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 :=
sorry

end xyz_inequality_l220_220746


namespace opposite_sides_range_l220_220866

theorem opposite_sides_range (a : ℝ) :
  (3 * (-3) - 2 * (-1) - a) * (3 * 4 - 2 * (-6) - a) < 0 ↔ -7 < a ∧ a < 24 :=
by
  simp
  sorry

end opposite_sides_range_l220_220866


namespace intersecting_lines_implies_a_eq_c_l220_220349

theorem intersecting_lines_implies_a_eq_c
  (k b a c : ℝ)
  (h_kb : k ≠ b)
  (exists_point : ∃ (x y : ℝ), (y = k * x + k) ∧ (y = b * x + b) ∧ (y = a * x + c)) :
  a = c := 
sorry

end intersecting_lines_implies_a_eq_c_l220_220349


namespace cyclist_distance_l220_220211

theorem cyclist_distance
  (x t d : ℝ)
  (h1 : d = x * t)
  (h2 : d = (x + 1) * (3 * t / 4))
  (h3 : d = (x - 1) * (t + 3)) :
  d = 18 :=
by {
  sorry
}

end cyclist_distance_l220_220211


namespace root_relationship_l220_220872

theorem root_relationship (m n a b : ℝ) 
  (h_eq : ∀ x, 3 - (x - m) * (x - n) = 0 ↔ x = a ∨ x = b) : a < m ∧ m < n ∧ n < b :=
by
  sorry

end root_relationship_l220_220872


namespace georgie_ghost_ways_l220_220633

-- Define the total number of windows and locked windows
def total_windows : ℕ := 8
def locked_windows : ℕ := 2

-- Define the number of usable windows
def usable_windows : ℕ := total_windows - locked_windows

-- Define the theorem to prove the number of ways Georgie the Ghost can enter and exit
theorem georgie_ghost_ways :
  usable_windows * (usable_windows - 1) = 30 := by
  sorry

end georgie_ghost_ways_l220_220633


namespace fractions_are_integers_l220_220140

theorem fractions_are_integers (a b : ℕ) (h1 : 1 < a) (h2 : 1 < b) 
    (h3 : abs ((a : ℚ) / b - (a - 1) / (b - 1)) = 1) : 
    ∃ m n : ℤ, (a : ℚ) / b = m ∧ (a - 1) / (b - 1) = n := 
sorry

end fractions_are_integers_l220_220140


namespace problem_1_problem_2_problem_3_l220_220663

theorem problem_1 (avg_daily_production : ℕ) (deviation_wed : ℤ) :
  avg_daily_production = 3000 →
  deviation_wed = -15 →
  avg_daily_production + deviation_wed = 2985 :=
by intros; sorry

theorem problem_2 (avg_daily_production : ℕ) (deviation_sat : ℤ) (deviation_fri : ℤ) :
  avg_daily_production = 3000 →
  deviation_sat = 68 →
  deviation_fri = -20 →
  (avg_daily_production + deviation_sat) - (avg_daily_production + deviation_fri) = 88 :=
by intros; sorry

theorem problem_3 (planned_weekly_production : ℕ) (deviations : List ℤ) :
  planned_weekly_production = 21000 →
  deviations = [35, -12, -15, 30, -20, 68, -9] →
  planned_weekly_production + deviations.sum = 21077 :=
by intros; sorry

end problem_1_problem_2_problem_3_l220_220663


namespace bugs_meet_at_point_P_l220_220482

theorem bugs_meet_at_point_P (r1 r2 v1 v2 t : ℝ) (h1 : r1 = 7) (h2 : r2 = 3) (h3 : v1 = 4 * Real.pi) (h4 : v2 = 3 * Real.pi) :
  t = 14 :=
by
  repeat { sorry }

end bugs_meet_at_point_P_l220_220482


namespace johns_ratio_l220_220131

-- Definitions for initial counts
def initial_pink := 26
def initial_green := 15
def initial_yellow := 24
def initial_total := initial_pink + initial_green + initial_yellow

-- Definitions for Carl's and John's actions
def carl_pink_taken := 4
def john_pink_taken := 6
def remaining_pink := initial_pink - carl_pink_taken - john_pink_taken

-- Definition for remaining hard hats
def total_remaining := 43

-- Compute John's green hat withdrawal
def john_green_taken := (initial_total - carl_pink_taken - john_pink_taken) - total_remaining
def ratio := john_green_taken / john_pink_taken

theorem johns_ratio : ratio = 2 :=
by
  -- Proof details omitted
  sorry

end johns_ratio_l220_220131


namespace minimum_xy_l220_220697

theorem minimum_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y + 6 = x * y) : 
  x * y ≥ 18 :=
sorry

end minimum_xy_l220_220697


namespace chess_games_l220_220175

theorem chess_games (n : ℕ) (total_games : ℕ) (players : ℕ) (games_per_player : ℕ)
  (h1 : players = 9)
  (h2 : total_games = 36)
  (h3 : ∀ i : ℕ, i < players → games_per_player = players - 1)
  (h4 : 2 * total_games = players * games_per_player) :
  games_per_player = 1 :=
by
  rw [h1, h2] at h4
  sorry

end chess_games_l220_220175


namespace paula_go_kart_rides_l220_220299

theorem paula_go_kart_rides
  (g : ℕ)
  (ticket_cost_go_karts : ℕ := 4 * g)
  (ticket_cost_bumper_cars : ℕ := 20)
  (total_tickets : ℕ := 24) :
  ticket_cost_go_karts + ticket_cost_bumper_cars = total_tickets → g = 1 :=
by {
  sorry
}

end paula_go_kart_rides_l220_220299


namespace elena_hike_total_miles_l220_220445

theorem elena_hike_total_miles (x1 x2 x3 x4 x5 : ℕ)
  (h1 : x1 + x2 = 36)
  (h2 : x2 + x3 = 40)
  (h3 : x3 + x4 + x5 = 45)
  (h4 : x1 + x4 = 38) : 
  x1 + x2 + x3 + x4 + x5 = 81 := 
sorry

end elena_hike_total_miles_l220_220445


namespace compound_propositions_l220_220413

def divides (a b : Nat) : Prop := ∃ k : Nat, b = k * a

-- Define the propositions p and q
def p : Prop := divides 6 12
def q : Prop := divides 6 24

-- Prove the compound propositions
theorem compound_propositions :
  (p ∨ q) ∧ (p ∧ q) ∧ ¬¬p :=
by
  -- We are proving three statements:
  -- 1. "p or q" is true.
  -- 2. "p and q" is true.
  -- 3. "not p" is false (which is equivalent to "¬¬p" being true).
  -- The actual proof will be constructed here.
  sorry

end compound_propositions_l220_220413


namespace more_wrappers_than_bottle_caps_at_park_l220_220087

-- Define the number of bottle caps and wrappers found at the park.
def bottle_caps_found : ℕ := 11
def wrappers_found : ℕ := 28

-- State the theorem to prove the number of more wrappers than bottle caps found at the park is 17.
theorem more_wrappers_than_bottle_caps_at_park : wrappers_found - bottle_caps_found = 17 :=
by
  -- proof goes here
  sorry

end more_wrappers_than_bottle_caps_at_park_l220_220087


namespace unique_solution_n_l220_220253

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem unique_solution_n (h : ∀ n : ℕ, (n > 0) → n^3 = 8 * (sum_digits n)^3 + 6 * (sum_digits n) * n + 1 → n = 17) : 
  n = 17 := 
by
  sorry

end unique_solution_n_l220_220253


namespace day_after_7k_days_is_friday_day_before_7k_days_is_thursday_day_after_100_days_is_sunday_l220_220481

-- Definitions based on problem conditions and questions
def day_of_week_after (n : ℤ) (current_day : String) : String :=
  if n % 7 = 0 then current_day else
    if n % 7 = 1 then "Saturday" else
    if n % 7 = 2 then "Sunday" else
    if n % 7 = 3 then "Monday" else
    if n % 7 = 4 then "Tuesday" else
    if n % 7 = 5 then "Wednesday" else
    "Thursday"

def day_of_week_before (n : ℤ) (current_day : String) : String :=
  day_of_week_after (-n) current_day

-- Conditions
def today : String := "Friday"

-- Prove the following
theorem day_after_7k_days_is_friday (k : ℤ) : day_of_week_after (7 * k) today = "Friday" :=
by sorry

theorem day_before_7k_days_is_thursday (k : ℤ) : day_of_week_before (7 * k) today = "Thursday" :=
by sorry

theorem day_after_100_days_is_sunday : day_of_week_after 100 today = "Sunday" :=
by sorry

end day_after_7k_days_is_friday_day_before_7k_days_is_thursday_day_after_100_days_is_sunday_l220_220481


namespace scorpion_additional_millipedes_l220_220069

theorem scorpion_additional_millipedes :
  let total_segments := 800 in
  let segments_one_millipede := 60 in
  let segments_two_millipedes := 2 * (2 * segments_one_millipede) in
  let total_eaten := segments_one_millipede + segments_two_millipedes in
  let remaining_segments := total_segments - total_eaten in
  let segments_per_millipede := 50 in
  remaining_segments / segments_per_millipede = 10 :=
by {
  sorry
}

end scorpion_additional_millipedes_l220_220069


namespace range_of_a_l220_220748

def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 2) : a ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
by
  sorry

end range_of_a_l220_220748


namespace ship_length_is_correct_l220_220521

-- Define the variables
variables (L E S C : ℝ)

-- Define the given conditions
def condition1 (L E S C : ℝ) : Prop := 320 * E = L + 320 * (S - C)
def condition2 (L E S C : ℝ) : Prop := 80 * E = L - 80 * (S + C)

-- Mathematical statement to be proven
theorem ship_length_is_correct
  (L E S C : ℝ)
  (h1 : condition1 L E S C)
  (h2 : condition2 L E S C) :
  L = 26 * E + (2 / 3) * E :=
sorry

end ship_length_is_correct_l220_220521


namespace jill_arrives_earlier_by_30_minutes_l220_220899

theorem jill_arrives_earlier_by_30_minutes :
  ∀ (d : ℕ) (v_jill v_jack : ℕ),
  d = 2 →
  v_jill = 12 →
  v_jack = 3 →
  ((d / v_jack) * 60 - (d / v_jill) * 60) = 30 :=
by
  intros d v_jill v_jack hd hvjill hvjack
  sorry

end jill_arrives_earlier_by_30_minutes_l220_220899


namespace incorrect_transformation_l220_220054

theorem incorrect_transformation (x y m : ℕ) : 
  (x = y → x + 3 = y + 3) ∧
  (-2 * x = -2 * y → x = y) ∧
  (m ≠ 0 → (x = y ↔ (x / m = y / m))) ∧
  ¬(x = y → x / m = y / m) :=
by
  sorry

end incorrect_transformation_l220_220054


namespace horse_bags_problem_l220_220374

theorem horse_bags_problem (x y : ℤ) 
  (h1 : x - 1 = y + 1) : 
  x + 1 = 2 * (y - 1) :=
sorry

end horse_bags_problem_l220_220374


namespace wendy_pictures_in_one_album_l220_220484

theorem wendy_pictures_in_one_album 
  (total_pictures : ℕ) (pictures_per_album : ℕ) (num_other_albums : ℕ)
  (h_total : total_pictures = 45) (h_pictures_per_album : pictures_per_album = 2) 
  (h_num_other_albums : num_other_albums = 9) : 
  ∃ (pictures_in_one_album : ℕ), pictures_in_one_album = 27 :=
by {
  sorry
}

end wendy_pictures_in_one_album_l220_220484


namespace candy_problem_l220_220383

-- Define conditions and the statement
theorem candy_problem (K : ℕ) (h1 : 49 = K + 3 * K + 8 + 6 + 10 + 5) : K = 5 :=
sorry

end candy_problem_l220_220383


namespace James_total_area_l220_220736

theorem James_total_area :
  let initial_length := 13
  let initial_width := 18
  let increased_length := initial_length + 2
  let increased_width := initial_width + 2
  let single_room_area := increased_length * increased_width
  let four_rooms_area := 4 * single_room_area
  let larger_room_area := 2 * single_room_area
  let total_area := four_rooms_area + larger_room_area
  total_area = 1800 :=
by
  let initial_length := 13
  let initial_width := 18
  let increased_length := initial_length + 2
  let increased_width := initial_width + 2
  let single_room_area := increased_length * increased_width
  let four_rooms_area := 4 * single_room_area
  let larger_room_area := 2 * single_room_area
  let total_area := four_rooms_area + larger_room_area
  have h : total_area = 1800 := by sorry
  exact h

end James_total_area_l220_220736


namespace new_car_travel_distance_l220_220219

-- Define the distance traveled by the older car
def distance_older_car : ℝ := 150

-- Define the percentage increase
def percentage_increase : ℝ := 0.30

-- Define the condition for the newer car's travel distance
def distance_newer_car (d_old : ℝ) (perc_inc : ℝ) : ℝ :=
  d_old * (1 + perc_inc)

-- Prove the main statement
theorem new_car_travel_distance :
  distance_newer_car distance_older_car percentage_increase = 195 := by
  -- Skip the proof body as instructed
  sorry

end new_car_travel_distance_l220_220219


namespace units_digit_G100_l220_220239

def G (n : ℕ) := 3 * 2 ^ (2 ^ n) + 2

theorem units_digit_G100 : (G 100) % 10 = 0 :=
by
  sorry

end units_digit_G100_l220_220239


namespace calculate_moment_of_inertia_l220_220673

noncomputable def moment_of_inertia (a ρ₀ k : ℝ) : ℝ :=
  8 * (a ^ (9/2)) * ((ρ₀ / 7) + (k * a / 9))

theorem calculate_moment_of_inertia (a ρ₀ k : ℝ) 
  (h₀ : 0 ≤ a) :
  moment_of_inertia a ρ₀ k = 8 * a ^ (9/2) * ((ρ₀ / 7) + (k * a / 9)) :=
sorry

end calculate_moment_of_inertia_l220_220673


namespace sandy_total_earnings_l220_220588

-- Define the conditions
def hourly_wage : ℕ := 15
def hours_friday : ℕ := 10
def hours_saturday : ℕ := 6
def hours_sunday : ℕ := 14

-- Define the total hours worked and total earnings
def total_hours := hours_friday + hours_saturday + hours_sunday
def total_earnings := total_hours * hourly_wage

-- State the theorem
theorem sandy_total_earnings : total_earnings = 450 := by
  sorry

end sandy_total_earnings_l220_220588


namespace solve_system_l220_220921

theorem solve_system (x y z u : ℝ) :
  x^3 * y^2 * z = 2 ∧
  z^3 * u^2 * x = 32 ∧
  y^3 * z^2 * u = 8 ∧
  u^3 * x^2 * y = 8 →
  (x = 1 ∧ y = 1 ∧ z = 2 ∧ u = 2) ∨
  (x = 1 ∧ y = -1 ∧ z = 2 ∧ u = -2) ∨
  (x = -1 ∧ y = 1 ∧ z = -2 ∧ u = 2) ∨
  (x = -1 ∧ y = -1 ∧ z = -2 ∧ u = -2) :=
sorry

end solve_system_l220_220921


namespace rachel_homework_total_l220_220763

-- Definitions based on conditions
def math_homework : Nat := 8
def biology_homework : Nat := 3

-- Theorem based on the problem statement
theorem rachel_homework_total : math_homework + biology_homework = 11 := by
  -- typically, here you would provide a proof, but we use sorry to skip it
  sorry

end rachel_homework_total_l220_220763


namespace apple_price_33kg_l220_220080

theorem apple_price_33kg
  (l q : ℝ)
  (h1 : 10 * l = 3.62)
  (h2 : 30 * l + 6 * q = 12.48) :
  30 * l + 3 * q = 11.67 :=
by
  sorry

end apple_price_33kg_l220_220080


namespace repetend_of_4_div_17_l220_220251

theorem repetend_of_4_div_17 :
  ∃ (r : String), (∀ (n : ℕ), (∃ (k : ℕ), (0 < k) ∧ (∃ (q : ℤ), (4 : ℤ) * 10 ^ (n + 12 * k) / 17 % 10 ^ 12 = q)) ∧ r = "235294117647") :=
sorry

end repetend_of_4_div_17_l220_220251


namespace find_c_l220_220604

variable {a b c : ℝ} 
variable (h_perpendicular : (a / 3) * (-3 / b) = -1)
variable (h_intersect1 : 2 * a + 9 = c)
variable (h_intersect2 : 6 - 3 * b = -c)
variable (h_ab_equal : a = b)

theorem find_c : c = 39 := 
by
  sorry

end find_c_l220_220604


namespace pow_div_eq_l220_220947

theorem pow_div_eq : (8:ℕ) ^ 15 / (64:ℕ) ^ 6 = 512 := by
  have h1 : 64 = 8 ^ 2 := by sorry
  have h2 : (64:ℕ) ^ 6 = (8 ^ 2) ^ 6 := by sorry
  have h3 : (8 ^ 2) ^ 6 = 8 ^ 12 := by sorry
  have h4 : (8:ℕ) ^ 15 / (8 ^ 12) = 8 ^ (15 - 12) := by sorry
  have h5 : 8 ^ 3 = 512 := by sorry
  exact sorry

end pow_div_eq_l220_220947


namespace cyclist_wait_time_l220_220195

theorem cyclist_wait_time 
  (hiker_speed : ℝ) (cyclist_speed : ℝ) (wait_time : ℝ) (catch_up_time : ℝ) 
  (hiker_speed_eq : hiker_speed = 4) 
  (cyclist_speed_eq : cyclist_speed = 12) 
  (wait_time_eq : wait_time = 5 / 60) 
  (catch_up_time_eq : catch_up_time = (2 / 3) / (1 / 15)) 
  : catch_up_time * 60 = 10 := 
by 
  sorry

end cyclist_wait_time_l220_220195


namespace find_c_interval_l220_220249

theorem find_c_interval (c : ℚ) : 
  (c / 4 ≤ 3 + c ∧ 3 + c < -3 * (1 + c)) ↔ (-4 ≤ c ∧ c < -3 / 2) := 
by 
  sorry

end find_c_interval_l220_220249


namespace probability_at_least_4_girls_l220_220002

theorem probability_at_least_4_girls (h : ∀ i ∈ Finset.range 6, 0.5) : 
  (Finset.sum (Finset.range 3) 
    (λ k, Classical.choose (Nat.choose 6 (4 + k)))) / (2^6) = 11 / 32 :=
by
  sorry

end probability_at_least_4_girls_l220_220002


namespace power_division_identity_l220_220958

theorem power_division_identity : (8 ^ 15) / (64 ^ 6) = 512 := by
  have h64 : 64 = 8 ^ 2 := by
    sorry
  have h_exp_rule : ∀ (a m n : ℕ), (a ^ m) ^ n = a ^ (m * n) := by
    sorry
  
  rw [h64]
  rw [h_exp_rule]
  sorry

end power_division_identity_l220_220958


namespace sum_first_3n_terms_l220_220321

-- Geometric Sequence: Sum of first n terms Sn, first 2n terms S2n, first 3n terms S3n.
variables {n : ℕ} {S : ℕ → ℕ}

-- Conditions
def sum_first_n_terms (S : ℕ → ℕ) (n : ℕ) : Prop := S n = 48
def sum_first_2n_terms (S : ℕ → ℕ) (n : ℕ) : Prop := S (2 * n) = 60

-- Theorem to Prove
theorem sum_first_3n_terms {S : ℕ → ℕ} (h1 : sum_first_n_terms S n) (h2 : sum_first_2n_terms S n) :
  S (3 * n) = 63 :=
sorry

end sum_first_3n_terms_l220_220321


namespace probability_C_D_l220_220794

variable (P : String → ℚ)

axiom h₁ : P "A" = 1/4
axiom h₂ : P "B" = 1/3
axiom h₃ : P "A" + P "B" + P "C" + P "D" = 1

theorem probability_C_D : P "C" + P "D" = 5/12 := by
  sorry

end probability_C_D_l220_220794


namespace log_function_domain_l220_220024

theorem log_function_domain :
  { x : ℝ | x^2 - 2 * x - 3 > 0 } = { x | x > 3 } ∪ { x | x < -1 } :=
by {
  sorry
}

end log_function_domain_l220_220024


namespace baker_cakes_total_l220_220382

-- Conditions
def initial_cakes : ℕ := 121
def cakes_sold : ℕ := 105
def cakes_bought : ℕ := 170

-- Proof Problem
theorem baker_cakes_total :
  initial_cakes - cakes_sold + cakes_bought = 186 :=
by
  sorry

end baker_cakes_total_l220_220382


namespace polygon_n_sides_l220_220799

theorem polygon_n_sides (n : ℕ) (h : (n - 2) * 180 - x = 2000) : n = 14 :=
sorry

end polygon_n_sides_l220_220799


namespace room_length_l220_220925

theorem room_length (L : ℕ) (h : 72 * L + 918 = 2718) : L = 25 := by
  sorry

end room_length_l220_220925


namespace sarah_initial_trucks_l220_220197

theorem sarah_initial_trucks (trucks_given : ℕ) (trucks_left : ℕ) (initial_trucks : ℕ) :
  trucks_given = 13 → trucks_left = 38 → initial_trucks = trucks_left + trucks_given → initial_trucks = 51 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sarah_initial_trucks_l220_220197


namespace stratified_sampling_num_of_female_employees_l220_220797

theorem stratified_sampling_num_of_female_employees :
  ∃ (total_employees male_employees sample_size female_employees_to_draw : ℕ),
    total_employees = 750 ∧
    male_employees = 300 ∧
    sample_size = 45 ∧
    female_employees_to_draw = (total_employees - male_employees) * sample_size / total_employees ∧
    female_employees_to_draw = 27 :=
by
  sorry

end stratified_sampling_num_of_female_employees_l220_220797


namespace juice_difference_is_eight_l220_220030

-- Defining the initial conditions
def initial_large_barrel : ℕ := 10
def initial_small_barrel : ℕ := 8
def poured_juice : ℕ := 3

-- Defining the final amounts
def final_large_barrel : ℕ := initial_large_barrel + poured_juice
def final_small_barrel : ℕ := initial_small_barrel - poured_juice

-- The statement we need to prove
theorem juice_difference_is_eight :
  final_large_barrel - final_small_barrel = 8 :=
by
  -- Skipping the proof
  sorry

end juice_difference_is_eight_l220_220030


namespace point_A_final_position_supplement_of_beta_l220_220759

-- Define the initial and final position of point A on the number line
def initial_position := -5
def moved_position_right := initial_position + 4
def final_position := moved_position_right - 1

theorem point_A_final_position : final_position = -2 := 
by 
-- Proof can be added here
sorry

-- Define the angles and the relationship between them
def alpha := 40
def beta := 90 - alpha
def supplement_beta := 180 - beta

theorem supplement_of_beta : supplement_beta = 130 := 
by 
-- Proof can be added here
sorry

end point_A_final_position_supplement_of_beta_l220_220759


namespace multiples_of_15_between_17_and_158_l220_220885

theorem multiples_of_15_between_17_and_158 : 
  let first := 30
  let last := 150
  let step := 15
  Nat.succ ((last - first) / step) = 9 := 
by
  sorry

end multiples_of_15_between_17_and_158_l220_220885


namespace arithmetic_sequence_sum_l220_220732

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0) 
  (h_sum : a 3 + a 9 + a 15 + a 21 = 8) :
  a 1 + a 23 = 4 :=
sorry

end arithmetic_sequence_sum_l220_220732


namespace fourth_term_of_geometric_sequence_is_320_l220_220213

theorem fourth_term_of_geometric_sequence_is_320
  (a : ℕ) (r : ℕ)
  (h_a : a = 5)
  (h_fifth_term : a * r^4 = 1280) :
  a * r^3 = 320 := 
by
  sorry

end fourth_term_of_geometric_sequence_is_320_l220_220213


namespace largest_three_digit_divisible_and_prime_sum_l220_220397

theorem largest_three_digit_divisible_and_prime_sum :
  ∃ n : ℕ, 900 ≤ n ∧ n < 1000 ∧
           (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d ≠ 0 ∧ n % d = 0) ∧
           Prime (n / 100 + (n / 10) % 10 + n % 10) ∧
           n = 963 ∧
           ∀ m : ℕ, 900 ≤ m ∧ m < 1000 ∧
           (∀ d ∈ [m / 100, (m / 10) % 10, m % 10], d ≠ 0 ∧ m % d = 0) ∧
           Prime (m / 100 + (m / 10) % 10 + m % 10) →
           m ≤ 963 :=
by
  sorry

end largest_three_digit_divisible_and_prime_sum_l220_220397


namespace prob_shooting_A_first_l220_220936

-- Define the probabilities
def prob_A_hits : ℝ := 0.4
def prob_A_misses : ℝ := 0.6
def prob_B_hits : ℝ := 0.6
def prob_B_misses : ℝ := 0.4

-- Define the overall problem
theorem prob_shooting_A_first (k : ℕ) (ξ : ℕ) (hξ : ξ = k) :
  ((prob_A_misses * prob_B_misses)^(k-1)) * (1 - (prob_A_misses * prob_B_misses)) = 0.24^(k-1) * 0.76 :=
by
  -- Placeholder for proof
  sorry

end prob_shooting_A_first_l220_220936


namespace card_collection_average_l220_220496

theorem card_collection_average (n : ℕ) (h : (2 * n + 1) / 3 = 2017) : n = 3025 :=
by
  sorry

end card_collection_average_l220_220496


namespace sacksPerSectionDaily_l220_220601

variable (totalSacks : ℕ) (sections : ℕ) (sacksPerSection : ℕ)

-- Conditions from the problem
variables (h1 : totalSacks = 360) (h2 : sections = 8)

-- The theorem statement
theorem sacksPerSectionDaily : sacksPerSection = 45 :=
by
  have h3 : totalSacks / sections = 45 := by sorry
  have h4 : sacksPerSection = totalSacks / sections := by sorry
  exact Eq.trans h4 h3

end sacksPerSectionDaily_l220_220601


namespace generating_function_chebyshev_T_generating_function_chebyshev_U_l220_220685

noncomputable def generating_function_T (x z : ℂ) :=
  (∑ n in (Finset.range n).toFinset, (T n x) * (z ^ n))

noncomputable def generating_function_U (x z : ℂ) :=
  (∑ n in (Finset.range n).toFinset, (U n x) * (z ^ n))

theorem generating_function_chebyshev_T (x z : ℂ) :
  generating_function_T x z = (1 - x*z) / (1 - 2*x*z + z^2) :=
sorry

theorem generating_function_chebyshev_U (x z : ℂ) :
  generating_function_U x z = 1 / (1 - 2*x*z + z^2) :=
sorry

end generating_function_chebyshev_T_generating_function_chebyshev_U_l220_220685


namespace b_profit_l220_220491

noncomputable def profit_share (x t : ℝ) : ℝ :=
  let total_profit := 31500
  let a_investment := 3 * x
  let a_period := 2 * t
  let b_investment := x
  let b_period := t
  let profit_ratio_a := a_investment * a_period
  let profit_ratio_b := b_investment * b_period
  let total_ratio := profit_ratio_a + profit_ratio_b
  let b_share := profit_ratio_b / total_ratio
  b_share * total_profit

theorem b_profit (x t : ℝ) : profit_share x t = 4500 :=
by
  sorry

end b_profit_l220_220491


namespace projection_vector_satisfies_conditions_l220_220831

variable (v1 v2 : ℚ)

def line_l (t : ℚ) : ℚ × ℚ :=
(2 + 3 * t, 5 - 2 * t)

def line_m (s : ℚ) : ℚ × ℚ :=
(-2 + 3 * s, 7 - 2 * s)

theorem projection_vector_satisfies_conditions :
  3 * v1 + 2 * v2 = 6 ∧ 
  ∃ k : ℚ, v1 = k * 3 ∧ v2 = k * (-2) → 
  (v1, v2) = (18 / 5, -12 / 5) :=
by
  sorry

end projection_vector_satisfies_conditions_l220_220831


namespace ratio_of_smaller_circle_to_larger_circle_l220_220045

section circles

variables {Q : Type} (C1 C2 : ℝ) (angle1 : ℝ) (angle2 : ℝ)

def ratio_of_areas (C1 C2 : ℝ) : ℝ := (C1 / C2)^2

theorem ratio_of_smaller_circle_to_larger_circle
  (h1 : angle1 = 60)
  (h2 : angle2 = 48)
  (h3 : (angle1 / 360) * C1 = (angle2 / 360) * C2) :
  ratio_of_areas C1 C2 = 16 / 25 :=
by
  sorry

end circles

end ratio_of_smaller_circle_to_larger_circle_l220_220045


namespace negation_of_quadratic_statement_l220_220490

variable {x a b : ℝ}

theorem negation_of_quadratic_statement (h : x = a ∨ x = b) : x^2 - (a + b) * x + ab = 0 := sorry

end negation_of_quadratic_statement_l220_220490


namespace max_parts_divided_by_three_planes_l220_220615

theorem max_parts_divided_by_three_planes (parts_0_plane parts_1_plane parts_2_planes parts_3_planes: ℕ)
  (h0 : parts_0_plane = 1)
  (h1 : parts_1_plane = 2)
  (h2 : parts_2_planes = 4)
  (h3 : parts_3_planes = 8) :
  parts_3_planes = 8 :=
by
  sorry

end max_parts_divided_by_three_planes_l220_220615


namespace evaluate_neg64_to_7_over_3_l220_220837

theorem evaluate_neg64_to_7_over_3 (a : ℝ) (b : ℝ) (c : ℝ) 
  (h1 : a = -64) (h2 : b = (-4)) (h3 : c = (7/3)) :
  a ^ c = -65536 := 
by
  have h4 : (-64 : ℝ) = (-4) ^ 3 := by sorry
  have h5 : a = b^3 := by rw [h1, h2, h4]
  have h6 : a ^ c = (b^3) ^ (7/3) := by rw [←h5, h3]
  have h7 : (b^3)^c = b^(3*(7/3)) := by sorry
  have h8 : b^(3*(7/3)) = b^7 := by norm_num
  have h9 : b^7 = -65536 := by sorry
  rw [h6, h7, h8, h9]
  exact h9

end evaluate_neg64_to_7_over_3_l220_220837


namespace determine_h_l220_220834

noncomputable def h (x : ℝ) : ℝ :=
  -12*x^4 + 2*x^3 + 8*x^2 - 8*x + 3

theorem determine_h :
  (12*x^4 + 4*x^3 - 2*x + 3 + h x = 6*x^3 + 8*x^2 - 10*x + 6) ↔
  (h x = -12*x^4 + 2*x^3 + 8*x^2 - 8*x + 3) :=
by 
  sorry

end determine_h_l220_220834


namespace evaluate_expression_is_sixth_l220_220063

noncomputable def evaluate_expression := (1 / Real.log 3000^4 / Real.log 8) + (4 / Real.log 3000^4 / Real.log 9)

theorem evaluate_expression_is_sixth:
  evaluate_expression = 1 / 6 :=
  by
  sorry

end evaluate_expression_is_sixth_l220_220063


namespace quadruples_characterization_l220_220683

/-- Proving the characterization of quadruples (a, b, c, d) of non-negative integers 
such that ab = 2(1 + cd) and there exists a non-degenerate triangle with sides (a - c), 
(b - d), and (c + d). -/
theorem quadruples_characterization :
  ∀ (a b c d : ℕ), 
    ab = 2 * (1 + cd) ∧ 
    (a - c) + (b - d) > c + d ∧ 
    (a - c) + (c + d) > b - d ∧ 
    (b - d) + (c + d) > a - c ∧
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    (a = 1 ∧ b = 2 ∧ c = 0 ∧ d = 1) ∨ 
    (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 0) :=
by sorry

end quadruples_characterization_l220_220683


namespace angle_same_terminal_side_l220_220595

theorem angle_same_terminal_side (k : ℤ) : ∃ α : ℝ, α = k * 360 - 30 ∧ 0 ≤ α ∧ α < 360 → α = 330 :=
by
  sorry

end angle_same_terminal_side_l220_220595


namespace max_expression_value_l220_220529

theorem max_expression_value (a b c d e f g h k : ℤ) 
  (ha : a = 1 ∨ a = -1)
  (hb : b = 1 ∨ b = -1)
  (hc : c = 1 ∨ c = -1)
  (hd : d = 1 ∨ d = -1)
  (he : e = 1 ∨ e = -1)
  (hf : f = 1 ∨ f = -1)
  (hg : g = 1 ∨ g = -1)
  (hh : h = 1 ∨ h = -1)
  (hk : k = 1 ∨ k = -1) :
  a * e * k - a * f * h + b * f * g - b * d * k + c * d * h - c * e * g ≤ 4 :=
sorry

end max_expression_value_l220_220529


namespace problem_expression_l220_220533

theorem problem_expression (x y : ℝ) (h1 : x - y = 5) (h2 : x * y = 4) : x^2 + y^2 = 33 :=
by sorry

end problem_expression_l220_220533


namespace shaded_area_l220_220438

theorem shaded_area 
  (R r : ℝ) 
  (h_area_larger_circle : π * R ^ 2 = 100 * π) 
  (h_shaded_larger_fraction : 2 / 3 = (area_shaded_larger / (π * R ^ 2))) 
  (h_relationship_radius : r = R / 2) 
  (h_area_smaller_circle : π * r ^ 2 = 25 * π)
  (h_shaded_smaller_fraction : 1 / 3 = (area_shaded_smaller / (π * r ^ 2))) : 
  (area_shaded_larger + area_shaded_smaller = 75 * π) := 
sorry

end shaded_area_l220_220438


namespace counting_error_l220_220765

theorem counting_error
  (b g : ℕ)
  (initial_balloons := 5 * b + 4 * g)
  (popped_balloons := g + 2 * b)
  (remaining_balloons := initial_balloons - popped_balloons)
  (Dima_count := 100) :
  remaining_balloons ≠ Dima_count := by
  sorry

end counting_error_l220_220765


namespace min_C_over_D_l220_220425

theorem min_C_over_D (x C D : ℝ) (h1 : x^2 + 1/x^2 = C) (h2 : x + 1/x = D) (hC_pos : 0 < C) (hD_pos : 0 < D) : 
  (∃ m : ℝ, m = 2 * Real.sqrt 2 ∧ ∀ y : ℝ, y = C / D → y ≥ m) :=
  sorry

end min_C_over_D_l220_220425


namespace polygon_sides_from_interior_angles_l220_220034

theorem polygon_sides_from_interior_angles (S : ℕ) (h : S = 1260) : S = (9 - 2) * 180 :=
by
  sorry

end polygon_sides_from_interior_angles_l220_220034


namespace distributive_laws_fail_for_all_l220_220284

def has_op_hash (a b : ℝ) : ℝ := a + 2 * b

theorem distributive_laws_fail_for_all (x y z : ℝ) : 
  ¬ (∀ x y z, has_op_hash x (y + z) = has_op_hash x y + has_op_hash x z) ∧
  ¬ (∀ x y z, x + has_op_hash y z = has_op_hash (x + y) (x + z)) ∧
  ¬ (∀ x y z, has_op_hash x (has_op_hash y z) = has_op_hash (has_op_hash x y) (has_op_hash x z)) := 
sorry

end distributive_laws_fail_for_all_l220_220284


namespace johns_donation_l220_220626

theorem johns_donation (A J : ℝ) 
  (h1 : (75 / 1.5) = A) 
  (h2 : A * 2 = 100)
  (h3 : (100 + J) / 3 = 75) : 
  J = 125 :=
by 
  sorry

end johns_donation_l220_220626


namespace julia_cakes_remaining_l220_220436

/-- Formalizing the conditions of the problem --/
def cakes_per_day : ℕ := 5 - 1
def baking_days : ℕ := 6
def eaten_cakes_per_other_day : ℕ := 1
def total_days : ℕ := 6
def total_eaten_days : ℕ := total_days / 2

/-- The theorem to be proven --/
theorem julia_cakes_remaining : 
  let total_baked_cakes := cakes_per_day * baking_days in
  let total_eaten_cakes := eaten_cakes_per_other_day * total_eaten_days in
  total_baked_cakes - total_eaten_cakes = 21 := 
by
  sorry

end julia_cakes_remaining_l220_220436


namespace x_value_l220_220564

def x_is_75_percent_greater (x : ℝ) (y : ℝ) : Prop := x = y + 0.75 * y

theorem x_value (x : ℝ) : x_is_75_percent_greater x 150 → x = 262.5 :=
by
  intro h
  rw [x_is_75_percent_greater] at h
  sorry

end x_value_l220_220564


namespace solitaire_game_end_with_one_piece_l220_220455

theorem solitaire_game_end_with_one_piece (n : ℕ) : 
  ∃ (remaining_pieces : ℕ), 
  remaining_pieces = 1 ↔ n % 3 ≠ 0 :=
sorry

end solitaire_game_end_with_one_piece_l220_220455


namespace pyramid_height_correct_l220_220504

noncomputable def pyramid_height (a α : ℝ) : ℝ :=
  a / (Real.sqrt (2 * (Real.tan (α / 2))^2 - 2))

theorem pyramid_height_correct (a α : ℝ) (hα : α ≠ 0 ∧ α ≠ π) :
  ∃ m : ℝ, m = pyramid_height a α := 
by
  use a / (Real.sqrt (2 * (Real.tan (α / 2))^2 - 2))
  sorry

end pyramid_height_correct_l220_220504


namespace tunnel_length_is_correct_l220_220368

-- Define the conditions given in the problem
def length_of_train : ℕ := 90
def speed_of_train : ℕ := 160
def time_to_pass_tunnel : ℕ := 3

-- Define the length of the tunnel to be proven
def length_of_tunnel : ℕ := 480 - length_of_train

-- Define the statement to be proven
theorem tunnel_length_is_correct : length_of_tunnel = 390 := by
  sorry

end tunnel_length_is_correct_l220_220368


namespace expression_range_l220_220142

theorem expression_range (a b c d : ℝ) 
    (ha : 0 ≤ a) (ha' : a ≤ 2)
    (hb : 0 ≤ b) (hb' : b ≤ 2)
    (hc : 0 ≤ c) (hc' : c ≤ 2)
    (hd : 0 ≤ d) (hd' : d ≤ 2) :
  4 + 2 * Real.sqrt 2 ≤ 
    Real.sqrt (a^2 + (2-b)^2) 
    + Real.sqrt (b^2 + (2-c)^2) 
    + Real.sqrt (c^2 + (2-d)^2) 
    + Real.sqrt (d^2 + (2-a)^2) 
  ∧ Real.sqrt (a^2 + (2-b)^2) 
    + Real.sqrt (b^2 + (2-c)^2) 
    + Real.sqrt (c^2 + (2-d)^2) 
    + Real.sqrt (d^2 + (2-a)^2) ≤ 8 := 
sorry

end expression_range_l220_220142


namespace square_divisible_by_six_in_range_l220_220248

theorem square_divisible_by_six_in_range (x : ℕ) (h1 : ∃ n : ℕ, x = n^2)
  (h2 : 6 ∣ x) (h3 : 30 < x) (h4 : x < 150) : x = 36 ∨ x = 144 :=
by {
  sorry
}

end square_divisible_by_six_in_range_l220_220248


namespace costs_equal_at_60_guests_l220_220020

theorem costs_equal_at_60_guests :
  ∀ (x : ℕ),
  (800 + 30 * x = 500 + 35 * x) ↔ (x = 60) :=
by
  intro x
  split
  · intro h
    have : 800 - 500 = 35 * x - 30 * x,
    calc
      800 + 30 * x = 500 + 35 * x : h
    rw [add_comm, this]
    sorry
  · intro hx
    rw hx
    rfl

end costs_equal_at_60_guests_l220_220020


namespace simplify_expression_l220_220764

variable (x : ℝ)

theorem simplify_expression : (3 * x - 4) * (x + 9) - (x + 6) * (3 * x - 2) = 7 * x - 24 :=
by
  sorry

end simplify_expression_l220_220764


namespace minimum_height_l220_220570

theorem minimum_height (y : ℝ) (h : ℝ) (S : ℝ) (hS : S = 10 * y^2) (hS_min : S ≥ 150) (h_height : h = 2 * y) : h = 2 * Real.sqrt 15 :=
  sorry

end minimum_height_l220_220570


namespace ginger_size_l220_220824

theorem ginger_size (anna_size : ℕ) (becky_size : ℕ) (ginger_size : ℕ) 
  (h1 : anna_size = 2) 
  (h2 : becky_size = 3 * anna_size) 
  (h3 : ginger_size = 2 * becky_size - 4) : 
  ginger_size = 8 :=
by
  -- The proof is omitted, just the theorem statement is required.
  sorry

end ginger_size_l220_220824


namespace ratio_of_smaller_circle_to_larger_circle_l220_220044

section circles

variables {Q : Type} (C1 C2 : ℝ) (angle1 : ℝ) (angle2 : ℝ)

def ratio_of_areas (C1 C2 : ℝ) : ℝ := (C1 / C2)^2

theorem ratio_of_smaller_circle_to_larger_circle
  (h1 : angle1 = 60)
  (h2 : angle2 = 48)
  (h3 : (angle1 / 360) * C1 = (angle2 / 360) * C2) :
  ratio_of_areas C1 C2 = 16 / 25 :=
by
  sorry

end circles

end ratio_of_smaller_circle_to_larger_circle_l220_220044


namespace smallest_n_divisibility_l220_220780

theorem smallest_n_divisibility (n : ℕ) (h : n = 5) : 
  ∃ k, (1 ≤ k ∧ k ≤ n + 1) ∧ (n^2 - n) % k = 0 ∧ ∃ i, (1 ≤ i ∧ i ≤ n + 1) ∧ (n^2 - n) % i ≠ 0 :=
by
  sorry

end smallest_n_divisibility_l220_220780


namespace find_n_l220_220136

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 3 ∧ ∀ n : ℕ, a (n + 1) = -1 / (a n + 1)

theorem find_n (a : ℕ → ℚ) (h : seq a) : ∃ n : ℕ, a n = 3 ∧ n = 16 :=
by
  sorry

end find_n_l220_220136


namespace largest_divisor_of_product_of_seven_visible_numbers_l220_220996

theorem largest_divisor_of_product_of_seven_visible_numbers
  (die_faces : Finset ℕ) (h_die_faces : die_faces = {1, 2, 3, 4, 5, 6, 7, 8}) (hidden_face : ℕ)
  (h_hidden : hidden_face ∈ die_faces) :
  let Q := (die_faces.erase hidden_face).prod id
  in 48 ∣ Q :=
by
  sorry

end largest_divisor_of_product_of_seven_visible_numbers_l220_220996


namespace complement_set_l220_220433

def U : Set ℝ := Set.univ
def M : Set ℝ := {y | ∃ x : ℝ, 0 < x ∧ x < 1 ∧ y = Real.log x / Real.log 2}

theorem complement_set :
  Set.compl M = {y : ℝ | y ≥ 0} :=
by
  sorry

end complement_set_l220_220433


namespace base10_representation_of_n_l220_220317

theorem base10_representation_of_n (a b c n : ℕ) (ha : a > 0)
  (h14 : n = 14^2 * a + 14 * b + c)
  (h15 : n = 15^2 * a + 15 * c + b)
  (h6 : n = 6^3 * a + 6^2 * c + 6 * a + c) : n = 925 :=
by sorry

end base10_representation_of_n_l220_220317


namespace probability_of_picking_two_red_balls_l220_220624

open Nat

theorem probability_of_picking_two_red_balls :
  let total_balls := 4 + 3 + 2
  let total_ways := Combination 9 2
  let favorable_ways := Combination 4 2
  let probability := favorable_ways / total_ways
  probability = 1 / 6 := by
  let total_balls := 9
  let total_ways := Combination total_balls 2
  let favorable_ways := Combination 4 2
  have total_ways_is_36 : total_ways = 36 := by
    calculate the value via the combination formula
    sorry
  have favorable_ways_is_6 : favorable_ways = 6 := by
    calculate the value via the combination formula
    sorry
  let probability := favorable_ways / total_ways
  have probability_is_1_over_6 : probability = (1 / 6) := by
    calculate the final probability value
    sorry
  exact probability_is_1_over_6

end probability_of_picking_two_red_balls_l220_220624


namespace C_increases_as_n_increases_l220_220447

noncomputable def C (e R r n : ℝ) : ℝ := (e * n^2) / (R + n * r)

theorem C_increases_as_n_increases
  (e R r : ℝ) (e_pos : 0 < e) (R_pos : 0 < R) (r_pos : 0 < r) :
  ∀ n : ℝ, 0 < n → ∃ M : ℝ, ∀ N : ℝ, N > n → C e R r N > M :=
by
  sorry

end C_increases_as_n_increases_l220_220447


namespace find_a_l220_220562

theorem find_a (a : ℝ) (x : ℝ) : (a - 1) * x^|a| + 4 = 0 → |a| = 1 → a ≠ 1 → a = -1 :=
by
  intros
  sorry

end find_a_l220_220562


namespace probability_no_shaded_square_l220_220205

theorem probability_no_shaded_square :
  let num_rects := 1003 * 2005
  let num_rects_with_shaded := 1002^2
  let probability_no_shaded := 1 - (num_rects_with_shaded / num_rects)
  probability_no_shaded = 1 / 1003 := by
  -- The proof steps go here
  sorry

end probability_no_shaded_square_l220_220205


namespace speed_against_current_l220_220342

theorem speed_against_current (V_curr : ℝ) (V_man : ℝ) (V_curr_val : V_curr = 3.2) (V_man_with_curr : V_man = 15) :
    V_man - V_curr = 8.6 := 
by 
  rw [V_curr_val, V_man_with_curr]
  norm_num
  sorry

end speed_against_current_l220_220342


namespace equidistant_points_quadrants_l220_220117

open Real

theorem equidistant_points_quadrants : 
  ∀ x y : ℝ, 
    (4 * x + 6 * y = 24) → (|x| = |y|) → 
    ((0 < x ∧ 0 < y) ∨ (x < 0 ∧ 0 < y)) :=
by
  sorry

end equidistant_points_quadrants_l220_220117


namespace rice_in_each_container_l220_220059

-- Given conditions from the problem
def total_weight_pounds : ℚ := 25 / 4
def num_containers : ℕ := 4
def pounds_to_ounces : ℚ := 16

-- A theorem that each container has 25 ounces of rice given the conditions
theorem rice_in_each_container (h : total_weight_pounds * pounds_to_ounces / num_containers = 25) : True :=
  sorry

end rice_in_each_container_l220_220059


namespace sine_addition_l220_220246

noncomputable def sin_inv_45 := Real.arcsin (4 / 5)
noncomputable def tan_inv_12 := Real.arctan (1 / 2)

theorem sine_addition :
  Real.sin (sin_inv_45 + tan_inv_12) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end sine_addition_l220_220246


namespace inequality_abc_l220_220916

theorem inequality_abc 
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c) : 
  a^2 * b^2 + b^2 * c^2 + a^2 * c^2 ≥ a * b * c * (a + b + c) := 
by 
  sorry

end inequality_abc_l220_220916


namespace more_than_half_millet_on_day_5_l220_220153

/-- Setup: Initial conditions and recursive definition of millet quantity -/
def millet_amount_on_day (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, (3 / 4 : ℝ)^i / 4

/-- Proposition to prove: On the 5th day, just after placing the seeds, more than half the seeds are millet -/
theorem more_than_half_millet_on_day_5 :
  millet_amount_on_day 5 > 1 / 2 :=
begin
  sorry
end

end more_than_half_millet_on_day_5_l220_220153


namespace incorrect_option_D_l220_220585

variable (AB BC BO DO AO CO : ℝ)
variable (DAB : ℝ)
variable (ABCD_is_rectangle ABCD_is_rhombus ABCD_is_square: Prop)

def conditions_statement :=
  AB = BC ∧
  DAB = 90 ∧
  BO = DO ∧
  AO = CO ∧
  (ABCD_is_rectangle ↔ (AB = BC ∧ AB ≠ BC)) ∧
  (ABCD_is_rhombus ↔ AB = BC ∧ AB ≠ BC) ∧
  (ABCD_is_square ↔ ABCD_is_rectangle ∧ ABCD_is_rhombus)

theorem incorrect_option_D
  (h1: BO = DO)
  (h2: AO = CO)
  (h3: ABCD_is_rectangle)
  (h4: conditions_statement AB BC BO DO AO CO DAB ABCD_is_rectangle ABCD_is_rhombus ABCD_is_square):
  ¬ ABCD_is_square :=
by
  sorry
  -- Proof omitted

end incorrect_option_D_l220_220585


namespace y_increase_when_x_increases_by_9_units_l220_220754

-- Given condition as a definition: when x increases by 3 units, y increases by 7 units.
def x_increases_y_increases (x_increase y_increase : ℕ) : Prop := 
  (x_increase = 3) → (y_increase = 7)

-- Stating the problem: when x increases by 9 units, y increases by how many units?
theorem y_increase_when_x_increases_by_9_units : 
  ∀ (x_increase y_increase : ℕ), x_increases_y_increases x_increase y_increase → ((x_increase * 3 = 9) → (y_increase * 3 = 21)) :=
by
  intros x_increase y_increase cond h
  sorry

end y_increase_when_x_increases_by_9_units_l220_220754


namespace evaluate_expression_l220_220240

-- Define the operation * given by the table
def op (a b : ℕ) : ℕ :=
  match (a, b) with
  | (1,1) => 1 | (1,2) => 2 | (1,3) => 3 | (1,4) => 4
  | (2,1) => 2 | (2,2) => 4 | (2,3) => 1 | (2,4) => 3
  | (3,1) => 3 | (3,2) => 1 | (3,3) => 4 | (3,4) => 2
  | (4,1) => 4 | (4,2) => 3 | (4,3) => 2 | (4,4) => 1
  | _ => 0  -- default to handle cases outside the defined table

-- Define the theorem to prove $(2*4)*(1*3) = 4$
theorem evaluate_expression : op (op 2 4) (op 1 3) = 4 := by
  sorry

end evaluate_expression_l220_220240


namespace julia_cakes_remaining_l220_220435

namespace CakeProblem

def cakes_per_day : ℕ := 5 - 1
def days_baked : ℕ := 6
def total_cakes_baked : ℕ := cakes_per_day * days_baked
def days_clifford_eats : ℕ := days_baked / 2
def cakes_eaten_by_clifford : ℕ := days_clifford_eats

theorem julia_cakes_remaining : total_cakes_baked - cakes_eaten_by_clifford = 21 :=
by
  -- proof goes here
  sorry

end CakeProblem

end julia_cakes_remaining_l220_220435


namespace eight_pow_15_div_sixtyfour_pow_6_l220_220942

theorem eight_pow_15_div_sixtyfour_pow_6 :
  8^15 / 64^6 = 512 := by
  sorry

end eight_pow_15_div_sixtyfour_pow_6_l220_220942


namespace count_valid_n_l220_220692

theorem count_valid_n : 
  ∃ n_values : Finset ℤ, 
    (∀ n ∈ n_values, (n + 2 ≤ 6 * n - 8) ∧ (6 * n - 8 < 3 * n + 7)) ∧
    (n_values.card = 3) :=
by sorry

end count_valid_n_l220_220692


namespace air_quality_conditional_probability_l220_220505

theorem air_quality_conditional_probability :
  (P(A) = 0.8) → (P(A) ∧ P(B) = 0.6) → (P(B | A) = 0.75) := 
by
  intros h1 h2
  sorry

end air_quality_conditional_probability_l220_220505


namespace necessary_and_sufficient_condition_l220_220696

def line1 (x y : ℝ) : Prop := x - y - 1 = 0
def line2 (x y a : ℝ) : Prop := x + a * y - 2 = 0

def p (a : ℝ) : Prop := ∀ x y : ℝ, line1 x y → line2 x y a
def q (a : ℝ) : Prop := a = -1

theorem necessary_and_sufficient_condition (a : ℝ) : (p a) ↔ (q a) :=
by
  sorry

end necessary_and_sufficient_condition_l220_220696


namespace rectangle_length_width_difference_l220_220535

theorem rectangle_length_width_difference
  (x y : ℝ)
  (h1 : y = 1 / 3 * x)
  (h2 : 2 * x + 2 * y = 32)
  (h3 : Real.sqrt (x^2 + y^2) = 17) :
  abs (x - y) = 8 :=
sorry

end rectangle_length_width_difference_l220_220535


namespace race_outcomes_l220_220817

-- Definition of participants
inductive Participant
| Abe 
| Bobby
| Charles
| Devin
| Edwin
| Frank
deriving DecidableEq

open Participant

def num_participants : ℕ := 6

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Proving the number of different 1st-2nd-3rd outcomes
theorem race_outcomes : factorial 6 / factorial 3 = 120 := by
  sorry

end race_outcomes_l220_220817


namespace no_real_solutions_eq_l220_220769

theorem no_real_solutions_eq (x y : ℝ) :
  x^2 + y^2 - 2 * x + 4 * y + 6 ≠ 0 :=
sorry

end no_real_solutions_eq_l220_220769


namespace geometric_series_sum_eq_l220_220006

theorem geometric_series_sum_eq :
  let a := (5 : ℚ)
  let r := (-1/2 : ℚ)
  (∑' n : ℕ, a * r^n) = (10 / 3 : ℚ) :=
by
  sorry

end geometric_series_sum_eq_l220_220006


namespace find_matrix_l220_220842

theorem find_matrix (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : M^3 - 3 * M^2 + 2 * M = ![![8, 16], ![4, 8]]) : 
  M = ![![2, 4], ![1, 2]] :=
sorry

end find_matrix_l220_220842


namespace tim_coins_value_l220_220616

variable (d q : ℕ)

-- Given Conditions
def total_coins (d q : ℕ) : Prop := d + q = 18
def quarter_to_dime_relation (d q : ℕ) : Prop := q = d + 2

-- Prove the value of the coins
theorem tim_coins_value (d q : ℕ) (h1 : total_coins d q) (h2 : quarter_to_dime_relation d q) : 10 * d + 25 * q = 330 := by
  sorry

end tim_coins_value_l220_220616


namespace expr_value_l220_220187

theorem expr_value : (34 + 7)^2 - (7^2 + 34^2 + 7 * 34) = 238 := by
  sorry

end expr_value_l220_220187


namespace area_ratio_of_concentric_circles_l220_220048

noncomputable theory

-- Define the given conditions
def C1 (r1 : ℝ) : ℝ := 2 * Real.pi * r1
def C2 (r2 : ℝ) : ℝ := 2 * Real.pi * r2
def arc_length (angle : ℝ) (circumference : ℝ) : ℝ := (angle / 360) * circumference

-- Lean statement for the math proof problem
theorem area_ratio_of_concentric_circles 
  (r1 r2 : ℝ) (h₁ : arc_length 60 (C1 r1) = arc_length 48 (C2 r2)) : 
  (Real.pi * r1^2) / (Real.pi * r2^2) = 16 / 25 :=
by
  sorry  -- Proof omitted

end area_ratio_of_concentric_circles_l220_220048


namespace x1x2_lt_one_l220_220257

noncomputable section

open Real

def f (a : ℝ) (x : ℝ) : ℝ :=
  |exp x - exp 1| + exp x + a * x

theorem x1x2_lt_one (a x1 x2 : ℝ) 
  (ha : a < -exp 1) 
  (hzero1 : f a x1 = 0) 
  (hzero2 : f a x2 = 0) 
  (h_order : x1 < x2) : x1 * x2 < 1 := 
sorry

end x1x2_lt_one_l220_220257


namespace mod_remainder_l220_220340

theorem mod_remainder (n : ℤ) (h : n % 3 = 2) : (7 * n) % 3 = 2 := by
  sorry

end mod_remainder_l220_220340


namespace simplify_sqrt_neg2_squared_l220_220162

theorem simplify_sqrt_neg2_squared : 
  Real.sqrt ((-2 : ℝ)^2) = 2 := 
by
  sorry

end simplify_sqrt_neg2_squared_l220_220162


namespace constant_sequence_is_AP_and_GP_l220_220847

theorem constant_sequence_is_AP_and_GP (seq : ℕ → ℕ) (h : ∀ n, seq n = 7) :
  (∃ d, ∀ n, seq n = seq (n + 1) + d) ∧ (∃ r, ∀ n, seq (n + 1) = seq n * r) :=
by
  sorry

end constant_sequence_is_AP_and_GP_l220_220847


namespace extremum_condition_l220_220695

theorem extremum_condition (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^3 + a * x^2 + b * x + a^2)
  (h2 : f 1 = 10)
  (h3 : deriv f 1 = 0) :
  a + b = -7 :=
sorry

end extremum_condition_l220_220695


namespace form_eleven_form_twelve_form_thirteen_form_fourteen_form_fifteen_form_sixteen_form_seventeen_form_eighteen_form_nineteen_form_twenty_l220_220184

theorem form_eleven : 22 - (2 + (2 / 2)) = 11 := by
  sorry

theorem form_twelve : (2 * 2 * 2) - 2 / 2 = 12 := by
  sorry

theorem form_thirteen : (22 + 2 + 2) / 2 = 13 := by
  sorry

theorem form_fourteen : 2 * 2 * 2 * 2 - 2 = 14 := by
  sorry

theorem form_fifteen : (2 * 2)^2 - 2 / 2 = 15 := by
  sorry

theorem form_sixteen : (2 * 2)^2 * (2 / 2) = 16 := by
  sorry

theorem form_seventeen : (2 * 2)^2 + 2 / 2 = 17 := by
  sorry

theorem form_eighteen : 2 * 2 * 2 * 2 + 2 = 18 := by
  sorry

theorem form_nineteen : 22 - 2 - 2 / 2 = 19 := by
  sorry

theorem form_twenty : (22 - 2) * (2 / 2) = 20 := by
  sorry

end form_eleven_form_twelve_form_thirteen_form_fourteen_form_fifteen_form_sixteen_form_seventeen_form_eighteen_form_nineteen_form_twenty_l220_220184


namespace no_intersection_with_x_axis_l220_220563

open Real

theorem no_intersection_with_x_axis (m : ℝ) :
  (∀ x : ℝ, 3 ^ (-(|x - 1|)) + m ≠ 0) ↔ (m ≥ 0 ∨ m < -1) :=
by
  sorry

end no_intersection_with_x_axis_l220_220563


namespace nth_term_closed_form_arithmetic_sequence_l220_220875

open Nat

noncomputable def S (n : ℕ) : ℕ := 3 * n^2 + 4 * n
noncomputable def a (n : ℕ) : ℕ := if h : n > 0 then S n - S (n-1) else S n

theorem nth_term_closed_form (n : ℕ) (h : n > 0) : a n = 6 * n + 1 :=
by
  sorry

theorem arithmetic_sequence (n : ℕ) (h : n > 1) : a n - a (n - 1) = 6 :=
by
  sorry

end nth_term_closed_form_arithmetic_sequence_l220_220875


namespace scientific_notation_of_concentration_l220_220453

theorem scientific_notation_of_concentration :
  0.000042 = 4.2 * 10^(-5) :=
sorry

end scientific_notation_of_concentration_l220_220453


namespace infinite_pairs_m_n_l220_220302

theorem infinite_pairs_m_n :
  ∃ (f : ℕ → ℕ × ℕ), (∀ k, (f k).1 > 0 ∧ (f k).2 > 0 ∧ ((f k).1 ∣ (f k).2 ^ 2 + 1) ∧ ((f k).2 ∣ (f k).1 ^ 2 + 1)) :=
sorry

end infinite_pairs_m_n_l220_220302


namespace fewest_four_dollar_frisbees_l220_220489

theorem fewest_four_dollar_frisbees (x y : ℕ) (h1 : x + y = 64) (h2 : 3 * x + 4 * y = 196) : y = 4 :=
by
  sorry

end fewest_four_dollar_frisbees_l220_220489


namespace perpendicular_lines_l220_220009

def line1 (m : ℝ) (x y : ℝ) := m * x - 3 * y - 1 = 0
def line2 (m : ℝ) (x y : ℝ) := (3 * m - 2) * x - m * y + 2 = 0

theorem perpendicular_lines (m : ℝ) :
  (∀ x y : ℝ, line1 m x y) →
  (∀ x y : ℝ, line2 m x y) →
  (∀ x y : ℝ, (m / 3) * ((3 * m - 2) / m) = -1) →
  m = 0 ∨ m = -1/3 :=
by
  intros
  sorry

end perpendicular_lines_l220_220009


namespace find_initial_candies_l220_220617

-- Definitions for the conditions
def initial_candies (x : ℕ) : Prop :=
  (3 * x) % 4 = 0 ∧
  (x % 2) = 0 ∧
  ∃ (k : ℕ), 2 ≤ k ∧ k ≤ 6 ∧ (1 * x) / 2 - 20 - k = 4

-- Theorems we need to prove
theorem find_initial_candies (x : ℕ) (h : initial_candies x) : x = 52 ∨ x = 56 ∨ x = 60 :=
sorry

end find_initial_candies_l220_220617


namespace sin_60_eq_sqrt3_div_2_l220_220230

theorem sin_60_eq_sqrt3_div_2 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_60_eq_sqrt3_div_2_l220_220230


namespace equality_of_costs_l220_220021

theorem equality_of_costs (x : ℕ) :
  (800 + 30 * x = 500 + 35 * x) ↔ x = 60 := by
  sorry

end equality_of_costs_l220_220021


namespace canteen_distance_l220_220077

theorem canteen_distance (r G B : ℝ) (d_g d_b : ℝ) (h_g : G = 600) (h_b : B = 800) (h_dg_db : d_g = d_b) : 
  d_g = 781 :=
by
  -- Proof to be completed
  sorry

end canteen_distance_l220_220077


namespace barrels_are_1360_l220_220508

-- Defining the top layer dimensions and properties
def a : ℕ := 2
def b : ℕ := 1
def n : ℕ := 15

-- Defining the dimensions of the bottom layer based on given properties
def c : ℕ := a + n
def d : ℕ := b + n

-- Formula for the total number of barrels
def total_barrels : ℕ := n * ((2 * a + c) * b + (2 * c + a) * d + (d - b)) / 6

-- Theorem to prove
theorem barrels_are_1360 : total_barrels = 1360 :=
by
  sorry

end barrels_are_1360_l220_220508


namespace eval_composed_function_l220_220887

noncomputable def f (x : ℝ) := 3 * x^2 - 4
noncomputable def k (x : ℝ) := 5 * x^3 + 2

theorem eval_composed_function :
  f (k 2) = 5288 := 
by
  sorry

end eval_composed_function_l220_220887


namespace find_magnitude_a_l220_220148

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (1, 2 * m)
def vector_b (m : ℝ) : ℝ × ℝ := (m + 1, 1)
def vector_c (m : ℝ) : ℝ × ℝ := (2, m)
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem find_magnitude_a (m : ℝ) (h : dot_product (vector_add (vector_a m) (vector_c m)) (vector_b m) = 0) :
  magnitude (vector_a (-1 / 2)) = Real.sqrt 2 :=
by
  sorry

end find_magnitude_a_l220_220148
