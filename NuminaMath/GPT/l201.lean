import Mathlib

namespace average_weight_l201_20118

/-- 
Given the following conditions:
1. (A + B) / 2 = 40
2. (B + C) / 2 = 41
3. B = 27
Prove that the average weight of a, b, and c is 45 kg.
-/
theorem average_weight (A B C : ℝ) 
  (h1 : (A + B) / 2 = 40)
  (h2 : (B + C) / 2 = 41)
  (h3 : B = 27): 
  (A + B + C) / 3 = 45 :=
by
  sorry

end average_weight_l201_20118


namespace cards_ratio_l201_20116

variable (x : ℕ)

def partially_full_decks_cards := 3 * x
def full_decks_cards := 3 * 52
def total_cards_before := 200 + 34

theorem cards_ratio (h : 3 * x + full_decks_cards = total_cards_before) : x / 52 = 1 / 2 :=
by sorry

end cards_ratio_l201_20116


namespace distinct_ints_divisibility_l201_20103

theorem distinct_ints_divisibility
  (x y z : ℤ) 
  (h1 : x ≠ y) 
  (h2 : y ≠ z) 
  (h3 : z ≠ x) : 
  ∃ k : ℤ, (x - y) ^ 5 + (y - z) ^ 5 + (z - x) ^ 5 = 5 * (y - z) * (z - x) * (x - y) * k := 
by 
  sorry

end distinct_ints_divisibility_l201_20103


namespace weight_lift_equality_l201_20158

-- Definitions based on conditions
def total_weight_25_pounds_lifted_times := 750
def total_weight_20_pounds_lifted_per_time (n : ℝ) := 60 * n

-- Statement of the proof problem
theorem weight_lift_equality : ∃ n, total_weight_20_pounds_lifted_per_time n = total_weight_25_pounds_lifted_times :=
  sorry

end weight_lift_equality_l201_20158


namespace six_digit_number_l201_20109

noncomputable def number_of_digits (N : ℕ) : ℕ := sorry

theorem six_digit_number :
  ∀ (N : ℕ),
    (N % 2020 = 0) ∧
    (∀ a b : ℕ, (a ≠ b ∧ N / 10^a % 10 ≠ N / 10^b % 10)) ∧
    (∀ a b : ℕ, (a ≠ b) → ((N / 10^a % 10 = N / 10^b % 10) -> (N % 2020 ≠ 0))) →
    number_of_digits N = 6 :=
sorry

end six_digit_number_l201_20109


namespace island_length_l201_20102

theorem island_length (area width : ℝ) (h_area : area = 50) (h_width : width = 5) : 
  area / width = 10 := 
by
  sorry

end island_length_l201_20102


namespace total_games_played_l201_20114

def number_of_games (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem total_games_played :
  number_of_games 9 2 = 36 :=
by
  -- Proof to be filled in later
  sorry

end total_games_played_l201_20114


namespace f_zero_f_pos_f_decreasing_solve_inequality_l201_20162

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom f_mul_add (m n : ℝ) : f m * f n = f (m + n)
axiom f_pos_neg (x : ℝ) : x < 0 → 1 < f x

theorem f_zero : f 0 = 1 :=
sorry

theorem f_pos (x : ℝ) : 0 < x → 0 < f x ∧ f x < 1 :=
sorry

theorem f_decreasing (x₁ x₂ : ℝ) : x₁ < x₂ → f x₁ > f x₂ :=
sorry

theorem solve_inequality (a x : ℝ) :
  f (x^2 - 3 * a * x + 1) * f (-3 * x + 6 * a + 1) ≥ 1 ↔
  (a > 1/3 ∧ 2 ≤ x ∧ x ≤ 3 * a + 1) ∨
  (a = 1/3 ∧ x = 2) ∨
  (a < 1/3 ∧ 3 * a + 1 ≤ x ∧ x ≤ 2) :=
sorry

end f_zero_f_pos_f_decreasing_solve_inequality_l201_20162


namespace prop_sufficient_not_necessary_l201_20136

-- Let p and q be simple propositions.
variables (p q : Prop)

-- Define the statement to be proved: 
-- "either p or q is false" is a sufficient but not necessary condition 
-- for "not p is true".
theorem prop_sufficient_not_necessary (hpq : ¬(p ∧ q)) : ¬ p :=
sorry

end prop_sufficient_not_necessary_l201_20136


namespace largest_constant_C_l201_20105

theorem largest_constant_C :
  ∃ C, C = 2 / Real.sqrt 3 ∧ ∀ (x y z : ℝ), x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z) := sorry

end largest_constant_C_l201_20105


namespace least_number_subtracted_to_divisible_by_10_l201_20120

def least_subtract_to_divisible_by_10 (n : ℕ) : ℕ :=
  let last_digit := n % 10
  10 - last_digit

theorem least_number_subtracted_to_divisible_by_10 (n : ℕ) : (n = 427751) → ((n - least_subtract_to_divisible_by_10 n) % 10 = 0) :=
by
  intros h
  sorry

end least_number_subtracted_to_divisible_by_10_l201_20120


namespace find_a_l201_20188

noncomputable def parabola_eq (a b c : ℤ) (x : ℤ) : ℤ :=
  a * x^2 + b * x + c

theorem find_a (a b c : ℤ)
  (h_vertex : ∀ x, parabola_eq a b c x = a * (x - 2)^2 + 5) 
  (h_point : parabola_eq a b c 1 = 6) :
  a = 1 := 
by 
  sorry

end find_a_l201_20188


namespace number_of_students_per_normal_class_l201_20144

theorem number_of_students_per_normal_class (total_students : ℕ) (percentage_moving : ℕ) (grade_levels : ℕ) (adv_class_size : ℕ) (additional_classes : ℕ) 
  (h1 : total_students = 1590) 
  (h2 : percentage_moving = 40) 
  (h3 : grade_levels = 3) 
  (h4 : adv_class_size = 20) 
  (h5 : additional_classes = 6) : 
  (total_students * percentage_moving / 100 / grade_levels - adv_class_size) / additional_classes = 32 :=
by
  sorry

end number_of_students_per_normal_class_l201_20144


namespace S9_equals_27_l201_20180

variables {a : ℕ → ℤ} {S : ℕ → ℤ} {d : ℤ}

-- (Condition 1) The sequence is an arithmetic sequence: a_{n+1} = a_n + d
axiom arithmetic_seq : ∀ n : ℕ, a (n + 1) = a n + d

-- (Condition 2) The sum S_n is the sum of the first n terms of the sequence
axiom sum_first_n_terms : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- (Condition 3) Given a_1 = 2 * a_3 - 3
axiom given_condition : a 1 = 2 * a 3 - 3

-- Prove that S_9 = 27
theorem S9_equals_27 : S 9 = 27 :=
by
  sorry

end S9_equals_27_l201_20180


namespace discriminant_zero_no_harmonic_progression_l201_20126

theorem discriminant_zero_no_harmonic_progression (a b c : ℝ) 
    (h_disc : b^2 = 24 * a * c) : 
    ¬ (2 * (1 / b) = (1 / a) + (1 / c)) := 
sorry

end discriminant_zero_no_harmonic_progression_l201_20126


namespace calculate_total_customers_l201_20104

theorem calculate_total_customers 
    (num_no_tip : ℕ) 
    (total_tip_amount : ℕ) 
    (tip_per_customer : ℕ) 
    (number_tipped_customers : ℕ) 
    (number_total_customers : ℕ)
    (h1 : num_no_tip = 5) 
    (h2 : total_tip_amount = 15) 
    (h3 : tip_per_customer = 3) 
    (h4 : number_tipped_customers = total_tip_amount / tip_per_customer) :
    number_total_customers = number_tipped_customers + num_no_tip := 
by {
    sorry
}

end calculate_total_customers_l201_20104


namespace compute_fraction_l201_20192

theorem compute_fraction :
  ( (12^4 + 500) * (24^4 + 500) * (36^4 + 500) * (48^4 + 500) * (60^4 + 500) ) /
  ( (6^4 + 500) * (18^4 + 500) * (30^4 + 500) * (42^4 + 500) * (54^4 + 500) ) = -182 :=
by
  sorry

end compute_fraction_l201_20192


namespace sum_of_four_integers_l201_20195

noncomputable def originalSum (a b c d : ℤ) :=
  (a + b + c + d)

theorem sum_of_four_integers
  (a b c d : ℤ)
  (h1 : (a + b + c) / 3 + d = 8)
  (h2 : (a + b + d) / 3 + c = 12)
  (h3 : (a + c + d) / 3 + b = 32 / 3)
  (h4 : (b + c + d) / 3 + a = 28 / 3) :
  originalSum a b c d = 30 :=
sorry

end sum_of_four_integers_l201_20195


namespace problem_goal_l201_20111

-- Define the problem stating that there is a graph of points (x, y) satisfying the condition
def area_of_graph_satisfying_condition : Real :=
  let A := 2013
  -- Define the pairs (a, b) which are multiples of 2013
  let pairs := [(1, 2013), (3, 671), (11, 183), (33, 61)]
  -- Calculate the area of each region formed by pairs
  let area := pairs.length * 4
  area

-- Problem goal statement proving the area is equal to 16
theorem problem_goal : area_of_graph_satisfying_condition = 16 := by
  sorry

end problem_goal_l201_20111


namespace original_number_l201_20134

theorem original_number (x : ℝ) (h : x * 1.20 = 1080) : x = 900 :=
sorry

end original_number_l201_20134


namespace oxygen_part_weight_l201_20140

def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

def molecular_weight_N2O : ℝ := 2 * atomic_weight_N + atomic_weight_O
def given_molecular_weight : ℝ := 108

theorem oxygen_part_weight : molecular_weight_N2O = 44.02 → atomic_weight_O = 16.00 := by
  sorry

end oxygen_part_weight_l201_20140


namespace monotonically_increasing_sequence_b_bounds_l201_20152

theorem monotonically_increasing_sequence_b_bounds (b : ℝ) :
  (∀ n : ℕ, 0 < n → (n + 1)^2 + b * (n + 1) > n^2 + b * n) ↔ b > -3 :=
by
  sorry

end monotonically_increasing_sequence_b_bounds_l201_20152


namespace range_of_ab_l201_20151

noncomputable def range_ab : Set ℝ := 
  { x | 4 ≤ x ∧ x ≤ 112 / 9 }

theorem range_of_ab (a b : ℝ) 
  (q : ℝ) (h1 : q ∈ (Set.Icc (1/3) 2)) 
  (h2 : ∃ m : ℝ, ∃ nq : ℕ, 
    (m * q ^ nq) * m ^ (2 - nq) = 1 ∧ 
    (m + m * q ^ nq) = a ∧ 
    (m * q + m * q ^ 2) = b):
  ab = (q + 1/q + q^2 + 1/q^2) → 
  (ab ∈ range_ab) := 
by 
  sorry

end range_of_ab_l201_20151


namespace candies_count_l201_20146

theorem candies_count (x : ℚ) (h : x + 3 * x + 12 * x + 72 * x = 468) : x = 117 / 22 :=
by
  sorry

end candies_count_l201_20146


namespace premium_percentage_on_shares_l201_20166

theorem premium_percentage_on_shares
    (investment : ℕ)
    (share_price : ℕ)
    (premium_percentage : ℕ)
    (dividend_percentage : ℕ)
    (total_dividend : ℕ)
    (number_of_shares : ℕ)
    (investment_eq : investment = number_of_shares * (share_price + premium_percentage))
    (dividend_eq : total_dividend = number_of_shares * (share_price * dividend_percentage / 100))
    (investment_val : investment = 14400)
    (share_price_val : share_price = 100)
    (dividend_percentage_val : dividend_percentage = 5)
    (total_dividend_val : total_dividend = 600)
    (number_of_shares_val : number_of_shares = 600 / 5) :
    premium_percentage = 20 :=
by
  sorry

end premium_percentage_on_shares_l201_20166


namespace Arman_worked_last_week_l201_20110

variable (H : ℕ) -- hours worked last week
variable (wage_last_week wage_this_week : ℝ)
variable (hours_this_week worked_this_week two_weeks_earning : ℝ)
variable (worked_last_week : Prop)

-- Define assumptions based on the problem conditions
def condition1 : wage_last_week = 10 := by sorry
def condition2 : wage_this_week = 10.5 := by sorry
def condition3 : hours_this_week = 40 := by sorry
def condition4 : worked_this_week = wage_this_week * hours_this_week := by sorry
def condition5 : worked_this_week = 420 := by sorry -- 10.5 * 40
def condition6 : two_weeks_earning = wage_last_week * (H : ℝ) + worked_this_week := by sorry
def condition7 : two_weeks_earning = 770 := by sorry

-- Proof statement
theorem Arman_worked_last_week : worked_last_week := by
  have h1 : wage_last_week * (H : ℝ) + worked_this_week = two_weeks_earning := sorry
  have h2 : wage_last_week * (H : ℝ) + 420 = 770 := sorry
  have h3 : wage_last_week * (H : ℝ) = 350 := sorry
  have h4 : (10 : ℝ) * (H : ℝ) = 350 := sorry
  have h5 : H = 35 := sorry
  sorry

end Arman_worked_last_week_l201_20110


namespace inequality_solution_l201_20138

theorem inequality_solution (x : ℝ) :
  (2 * x^2 + x < 6) ↔ (-2 < x ∧ x < 3 / 2) :=
by
  sorry

end inequality_solution_l201_20138


namespace marks_in_english_l201_20153

theorem marks_in_english :
  let m := 35             -- Marks in Mathematics
  let p := 52             -- Marks in Physics
  let c := 47             -- Marks in Chemistry
  let b := 55             -- Marks in Biology
  let n := 5              -- Number of subjects
  let avg := 46.8         -- Average marks
  let total_marks := avg * n
  total_marks - (m + p + c + b) = 45 := sorry

end marks_in_english_l201_20153


namespace assisted_work_time_l201_20157

theorem assisted_work_time (a b c : ℝ) (ha : a = 1 / 11) (hb : b = 1 / 20) (hc : c = 1 / 55) :
  (1 / ((a + b) + (a + c) / 2)) = 8 :=
by
  sorry

end assisted_work_time_l201_20157


namespace range_of_a_l201_20130

noncomputable def has_two_distinct_real_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a^x₁ = x₁ ∧ a^x₂ = x₂

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : has_two_distinct_real_roots a) : 
  1 < a ∧ a < Real.exp (1 / Real.exp 1) :=
sorry

end range_of_a_l201_20130


namespace cost_of_headphones_l201_20121

-- Define the constants for the problem
def bus_ticket_cost : ℕ := 11
def drinks_and_snacks_cost : ℕ := 3
def wifi_cost_per_hour : ℕ := 2
def trip_hours : ℕ := 3
def earnings_per_hour : ℕ := 12
def total_earnings := earnings_per_hour * trip_hours
def total_expenses_without_headphones := bus_ticket_cost + drinks_and_snacks_cost + (wifi_cost_per_hour * trip_hours)

-- Prove the cost of headphones, H, is $16 
theorem cost_of_headphones : total_earnings = total_expenses_without_headphones + 16 := by
  -- setup the goal
  sorry

end cost_of_headphones_l201_20121


namespace total_boys_in_groups_l201_20139

-- Definitions of number of groups
def total_groups : ℕ := 35
def groups_with_1_boy : ℕ := 10
def groups_with_at_least_2_boys : ℕ := 19
def groups_with_3_boys_twice_groups_with_3_girls (groups_with_3_boys groups_with_3_girls : ℕ) : Prop :=
  groups_with_3_boys = 2 * groups_with_3_girls

theorem total_boys_in_groups :
  ∃ (groups_with_3_girls groups_with_3_boys groups_with_1_girl_2_boys : ℕ),
    groups_with_1_boy + groups_with_at_least_2_boys + groups_with_3_girls = total_groups
    ∧ groups_with_3_boys_twice_groups_with_3_girls groups_with_3_boys groups_with_3_girls
    ∧ groups_with_1_girl_2_boys + groups_with_3_boys = groups_with_at_least_2_boys
    ∧ (groups_with_1_boy * 1 + groups_with_1_girl_2_boys * 2 + groups_with_3_boys * 3) = 60 :=
sorry

end total_boys_in_groups_l201_20139


namespace highest_lowest_difference_l201_20164

variable (x1 x2 x3 x4 x5 x_max x_min : ℝ)

theorem highest_lowest_difference (h1 : x1 + x2 + x3 + x4 + x5 - x_max = 37.84)
                                  (h2 : x1 + x2 + x3 + x4 + x5 - x_min = 38.64):
                                  x_max - x_min = 0.8 := 
by
  sorry

end highest_lowest_difference_l201_20164


namespace inequality_solution_set_l201_20113

theorem inequality_solution_set 
  (a b : ℝ) 
  (h : ∀ x, x ∈ Set.Icc (-3) 1 → ax^2 + (a + b)*x + 2 > 0) : 
  a + b = -4/3 := 
sorry

end inequality_solution_set_l201_20113


namespace perfect_square_l201_20156

variables {n x k ℓ : ℕ}

theorem perfect_square (h1 : x^2 < n) (h2 : n < (x + 1)^2)
  (h3 : k = n - x^2) (h4 : ℓ = (x + 1)^2 - n) :
  ∃ m : ℕ, n - k * ℓ = m^2 :=
by
  sorry

end perfect_square_l201_20156


namespace value_of_f_at_1_over_16_l201_20129

noncomputable def f (x : ℝ) (α : ℝ) := x ^ α

theorem value_of_f_at_1_over_16 (α : ℝ) (h : f 4 α = 2) : f (1 / 16) α = 1 / 4 :=
by
  sorry

end value_of_f_at_1_over_16_l201_20129


namespace simon_project_score_l201_20135

-- Define the initial conditions
def num_students_before : Nat := 20
def num_students_total : Nat := 21
def avg_before : ℕ := 86
def avg_after : ℕ := 88

-- Calculate total score before Simon's addition
def total_score_before : ℕ := num_students_before * avg_before

-- Calculate total score after Simon's addition
def total_score_after : ℕ := num_students_total * avg_after

-- Definition to represent Simon's score
def simon_score : ℕ := total_score_after - total_score_before

-- Theorem that we need to prove
theorem simon_project_score : simon_score = 128 :=
by
  sorry

end simon_project_score_l201_20135


namespace average_hidden_primes_l201_20122

theorem average_hidden_primes (x y z : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) (hz : Nat.Prime z)
  (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h_sum : 44 + x = 59 + y ∧ 59 + y = 38 + z) :
  (x + y + z) / 3 = 14 := 
by
  sorry

end average_hidden_primes_l201_20122


namespace solve_fractional_equation_l201_20128

theorem solve_fractional_equation (x : ℝ) (h : x ≠ 3) : (2 * x) / (x - 3) = 1 ↔ x = -3 :=
by
  sorry

end solve_fractional_equation_l201_20128


namespace last_three_digits_of_7_pow_99_l201_20142

theorem last_three_digits_of_7_pow_99 : (7 ^ 99) % 1000 = 573 := 
by sorry

end last_three_digits_of_7_pow_99_l201_20142


namespace value_of_f_neg_a_l201_20176

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end value_of_f_neg_a_l201_20176


namespace cost_price_is_3000_l201_20115

variable (CP SP : ℝ)

-- Condition: selling price (SP) is 20% more than the cost price (CP)
def sellingPrice : ℝ := CP + 0.20 * CP

-- Condition: selling price (SP) is Rs. 3600
axiom selling_price_eq : SP = 3600

-- Given the above conditions, prove that the cost price (CP) is Rs. 3000
theorem cost_price_is_3000 (h : sellingPrice CP = SP) : CP = 3000 := by
  sorry

end cost_price_is_3000_l201_20115


namespace balance_balls_l201_20145

variable {R Y B W : ℕ}

theorem balance_balls (h1 : 4 * R = 8 * B) 
                      (h2 : 3 * Y = 9 * B) 
                      (h3 : 5 * B = 3 * W) : 
    (2 * R + 4 * Y + 3 * W) = 21 * B :=
by 
  sorry

end balance_balls_l201_20145


namespace infinite_danish_numbers_l201_20159

-- Definitions translated from problem conditions
def is_danish (n : ℕ) : Prop :=
  ∃ k, n = 3 * k ∨ n = 2 * 4 ^ k

theorem infinite_danish_numbers :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, is_danish n ∧ is_danish (2^n + n) := sorry

end infinite_danish_numbers_l201_20159


namespace range_of_m_l201_20133

def G (x y : ℤ) : ℤ :=
  if x ≥ y then x - y
  else y - x

theorem range_of_m (m : ℤ) :
  (∀ x, 0 < x → G x 1 > 4 → G (-1) x ≤ m) ↔ 9 ≤ m ∧ m < 10 :=
sorry

end range_of_m_l201_20133


namespace simplify_expression_l201_20198

variables {a b c : ℝ}
variable (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0)
variable (h₃ : b - 2 / c ≠ 0)

theorem simplify_expression :
  (a - 2 / b) / (b - 2 / c) = c / b :=
sorry

end simplify_expression_l201_20198


namespace tank_fraction_l201_20123

theorem tank_fraction (x : ℚ) (h₁ : 48 * x + 8 = 48 * (9 / 10)) : x = 2 / 5 :=
by
  sorry

end tank_fraction_l201_20123


namespace merchant_boxes_fulfill_order_l201_20175

theorem merchant_boxes_fulfill_order :
  ∃ (a b c d e : ℕ), 16 * a + 17 * b + 23 * c + 39 * d + 40 * e = 100 := sorry

end merchant_boxes_fulfill_order_l201_20175


namespace distance_from_Q_to_EG_l201_20173

noncomputable def distance_to_line : ℝ :=
  let E := (0, 5)
  let F := (5, 5)
  let G := (5, 0)
  let H := (0, 0)
  let N := (2.5, 0)
  let Q := (25 / 7, 10 / 7)
  let line_y := 5
  let distance := abs (line_y - Q.2)
  distance

theorem distance_from_Q_to_EG : distance_to_line = 25 / 7 :=
by
  sorry

end distance_from_Q_to_EG_l201_20173


namespace hyperbola_asymptotes_l201_20106

theorem hyperbola_asymptotes (x y : ℝ) : 
  (x^2 / 4 - y^2 = 1) → (y = x / 2 ∨ y = -x / 2) :=
by
  sorry

end hyperbola_asymptotes_l201_20106


namespace xyz_ineq_l201_20193

theorem xyz_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y + y * z + z * x = 1) : 
  x * y * z * (x + y + z) ≤ 1 / 3 := 
sorry

end xyz_ineq_l201_20193


namespace wonderland_cities_l201_20149

theorem wonderland_cities (V E B : ℕ) (hE : E = 45) (hB : B = 42) (h_connected : connected_graph) (h_simple : simple_graph) (h_bridges : count_bridges = 42) : V = 45 :=
sorry

end wonderland_cities_l201_20149


namespace bucket_water_l201_20100

theorem bucket_water (oz1 oz2 oz3 oz4 oz5 total1 total2: ℕ) 
  (h1 : oz1 = 11)
  (h2 : oz2 = 13)
  (h3 : oz3 = 12)
  (h4 : oz4 = 16)
  (h5 : oz5 = 10)
  (h_total : total1 = oz1 + oz2 + oz3 + oz4 + oz5)
  (h_second_bucket : total2 = 39)
  : total1 - total2 = 23 :=
sorry

end bucket_water_l201_20100


namespace sin_product_identity_sin_cos_fraction_identity_l201_20137

-- First Proof Problem: Proving that the product of sines equals the given value
theorem sin_product_identity :
  (Real.sin (Real.pi * 6 / 180) * 
   Real.sin (Real.pi * 42 / 180) * 
   Real.sin (Real.pi * 66 / 180) * 
   Real.sin (Real.pi * 78 / 180)) = 
  (Real.sqrt 5 - 1) / 32 := 
by 
  sorry

-- Second Proof Problem: Given sin alpha and alpha in the second quadrant, proving the given fraction value
theorem sin_cos_fraction_identity (α : Real) 
  (h1 : π/2 < α ∧ α < π)
  (h2 : Real.sin α = Real.sqrt 15 / 4) :
  (Real.sin (α + Real.pi / 4)) / 
  (Real.sin (2 * α) + Real.cos (2 * α) + 1) = 
  -Real.sqrt 2 :=
by 
  sorry

end sin_product_identity_sin_cos_fraction_identity_l201_20137


namespace complement_A_in_U_l201_20124

def U : Set ℕ := {2, 3, 4}
def A : Set ℕ := {2, 3}

theorem complement_A_in_U : (U \ A) = {4} :=
by 
  sorry

end complement_A_in_U_l201_20124


namespace semicircle_radius_l201_20189

-- Definition of the problem conditions
variables (a h : ℝ) -- base and height of the triangle
variable (R : ℝ)    -- radius of the semicircle

-- Statement of the proof problem
theorem semicircle_radius (h_pos : 0 < h) (a_pos : 0 < a) 
(semicircle_condition : ∀ R > 0, a * (h - R) = 2 * R * h) : R = a * h / (a + 2 * h) :=
sorry

end semicircle_radius_l201_20189


namespace root_quadratic_eq_k_value_l201_20182

theorem root_quadratic_eq_k_value (k : ℤ) :
  (∃ x : ℤ, x = 5 ∧ 2 * x ^ 2 + 3 * x - k = 0) → k = 65 :=
by
  sorry

end root_quadratic_eq_k_value_l201_20182


namespace object_distance_traveled_l201_20169

theorem object_distance_traveled
  (t : ℕ) (v_mph : ℝ) (mile_to_feet : ℕ)
  (h_t : t = 2)
  (h_v : v_mph = 68.18181818181819)
  (h_mile : mile_to_feet = 5280) :
  ∃ d : ℝ, d = 200 :=
by {
  sorry
}

end object_distance_traveled_l201_20169


namespace functional_equation_solution_l201_20185

theorem functional_equation_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f x * f y = f (x - y)) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) :=
sorry

end functional_equation_solution_l201_20185


namespace sum_first_8_terms_of_geom_seq_l201_20117

-- Definitions: the sequence a_n, common ratio q, and the fact that specific terms form an arithmetic sequence.
def geom_seq (a : ℕ → ℕ) (a1 : ℕ) (q : ℕ) := ∀ n, a n = a1 * q^(n-1)
def arith_seq (b c d : ℕ) := 2 * b + (c - 2 * b) = d

-- Conditions
variables {a : ℕ → ℕ} {a1 : ℕ} {q : ℕ}
variables (h1 : geom_seq a a1 q) (h2 : q = 2)
variables (h3 : arith_seq (2 * a 4) (a 6) 48)

-- Goal: sum of the first 8 terms of the sequence equals 255
def sum_geometric_sequence (a1 : ℕ) (q : ℕ) (n : ℕ) := a1 * (1 - q^n) / (1 - q)

theorem sum_first_8_terms_of_geom_seq : 
  sum_geometric_sequence a1 q 8 = 255 :=
by
  sorry

end sum_first_8_terms_of_geom_seq_l201_20117


namespace segment_length_is_13_l201_20101

def point := (ℝ × ℝ)

def p1 : point := (2, 3)
def p2 : point := (7, 15)

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem segment_length_is_13 : distance p1 p2 = 13 := by
  sorry

end segment_length_is_13_l201_20101


namespace bank_balance_after_two_years_l201_20186

-- Define the original amount deposited
def original_amount : ℝ := 5600

-- Define the interest rate
def interest_rate : ℝ := 0.07

-- Define the interest for each year based on the original amount
def interest_per_year : ℝ := original_amount * interest_rate

-- Define the total amount after two years
def total_amount_after_two_years : ℝ := original_amount + interest_per_year + interest_per_year

-- Define the target value
def target_value : ℝ := 6384

-- The theorem we aim to prove
theorem bank_balance_after_two_years : 
  total_amount_after_two_years = target_value := 
by
  -- Proof goes here
  sorry

end bank_balance_after_two_years_l201_20186


namespace depth_of_first_hole_l201_20154

-- Conditions as definitions in Lean 4
def number_of_workers_first_hole : Nat := 45
def hours_worked_first_hole : Nat := 8

def number_of_workers_second_hole : Nat := 110  -- 45 existing workers + 65 extra workers
def hours_worked_second_hole : Nat := 6
def depth_second_hole : Nat := 55

-- The key assumption that work done (W) is proportional to the depth of the hole (D)
theorem depth_of_first_hole :
  let work_first_hole := number_of_workers_first_hole * hours_worked_first_hole
  let work_second_hole := number_of_workers_second_hole * hours_worked_second_hole
  let depth_first_hole := (work_first_hole * depth_second_hole) / work_second_hole
  depth_first_hole = 30 := sorry

end depth_of_first_hole_l201_20154


namespace imaginary_part_of_z_l201_20190

open Complex

-- Definition of the complex number as per the problem statement
def z : ℂ := (2 - 3 * Complex.I) * Complex.I

-- The theorem stating that the imaginary part of the given complex number is 2
theorem imaginary_part_of_z : z.im = 2 :=
by
  sorry

end imaginary_part_of_z_l201_20190


namespace count_five_digit_numbers_ending_in_6_divisible_by_3_l201_20178

theorem count_five_digit_numbers_ending_in_6_divisible_by_3 : 
  (∃ (n : ℕ), n = 3000 ∧
  ∀ (x : ℕ), (x ≥ 10000 ∧ x ≤ 99999) ∧ (x % 10 = 6) ∧ (x % 3 = 0) ↔ 
  (∃ (k : ℕ), x = 10026 + k * 30 ∧ k < 3000)) :=
by
  -- Proof is omitted
  sorry

end count_five_digit_numbers_ending_in_6_divisible_by_3_l201_20178


namespace number_of_quadratic_PQ_equal_to_PR_l201_20160

noncomputable def P (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4)

def is_quadratic (Q : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, Q = λ x => a * x^2 + b * x + c

theorem number_of_quadratic_PQ_equal_to_PR :
  let possible_Qx_fwds := 4^4
  let non_quadratic_cases := 6
  possible_Qx_fwds - non_quadratic_cases = 250 :=
by
  sorry

end number_of_quadratic_PQ_equal_to_PR_l201_20160


namespace pharmacist_weights_exist_l201_20197

theorem pharmacist_weights_exist :
  ∃ (a b c : ℝ), a + b = 100 ∧ a + c = 101 ∧ b + c = 102 ∧ a < 90 ∧ b < 90 ∧ c < 90 :=
by
  sorry

end pharmacist_weights_exist_l201_20197


namespace arithmetic_geometric_sequence_problem_l201_20170

variable {n : ℕ}

def a (n : ℕ) : ℕ := 3 * n - 1
def b (n : ℕ) : ℕ := 2 ^ n
def S (n : ℕ) : ℕ := n * (2 + (2 + (n - 1) * (3 - 1))) / 2 -- sum of an arithmetic sequence
def T (n : ℕ) : ℕ := (3 * n - 4) * 2 ^ (n + 1) + 8

theorem arithmetic_geometric_sequence_problem :
  (a 1 = 2) ∧ (b 1 = 2) ∧ (a 4 + b 4 = 27) ∧ (S 4 - b 4 = 10) →
  (∀ n, T n = (3 * n - 4) * 2 ^ (n + 1) + 8) := sorry

end arithmetic_geometric_sequence_problem_l201_20170


namespace train_speed_is_64_98_kmph_l201_20107

noncomputable def train_length : ℝ := 200
noncomputable def bridge_length : ℝ := 180
noncomputable def passing_time : ℝ := 21.04615384615385
noncomputable def speed_in_kmph : ℝ := 3.6 * (train_length + bridge_length) / passing_time

theorem train_speed_is_64_98_kmph : abs (speed_in_kmph - 64.98) < 0.01 :=
by
  sorry

end train_speed_is_64_98_kmph_l201_20107


namespace geometric_progression_solution_l201_20191

theorem geometric_progression_solution (b4 b2 b6 : ℚ) (h1 : b4 - b2 = -45 / 32) (h2 : b6 - b4 = -45 / 512) :
  (∃ (b1 q : ℚ), b4 = b1 * q^3 ∧ b2 = b1 * q ∧ b6 = b1 * q^5 ∧ 
    ((b1 = 6 ∧ q = 1 / 4) ∨ (b1 = -6 ∧ q = -1 / 4))) :=
by
  sorry

end geometric_progression_solution_l201_20191


namespace sum_real_imag_parts_l201_20171

open Complex

theorem sum_real_imag_parts (z : ℂ) (i : ℂ) (i_property : i * i = -1) (z_eq : z * i = -1 + i) :
  (z.re + z.im = 2) :=
  sorry

end sum_real_imag_parts_l201_20171


namespace gcd_6724_13104_l201_20119

theorem gcd_6724_13104 : Int.gcd 6724 13104 = 8 := 
sorry

end gcd_6724_13104_l201_20119


namespace possible_value_of_b_l201_20165

-- Definition of the linear function
def linear_function (x : ℝ) (b : ℝ) : ℝ := -2 * x + b

-- Condition for the linear function to pass through the second, third, and fourth quadrants
def passes_second_third_fourth_quadrants (b : ℝ) : Prop :=
  b < 0

-- Lean 4 statement expressing the problem
theorem possible_value_of_b (b : ℝ) (h : passes_second_third_fourth_quadrants b) : b = -1 :=
  sorry

end possible_value_of_b_l201_20165


namespace auditorium_rows_l201_20148

noncomputable def rows_in_auditorium : Nat :=
  let class1 := 30
  let class2 := 26
  let condition1 := ∃ row : Nat, row < class1 ∧ ∀ students_per_row : Nat, students_per_row ≤ row 
  let condition2 := ∃ empty_rows : Nat, empty_rows ≥ 3 ∧ ∀ students : Nat, students = class2 - empty_rows
  29

theorem auditorium_rows (n : Nat) (class1 : Nat) (class2 : Nat) (c1 : class1 ≥ n) (c2 : class2 ≤ n - 3)
  : n = 29 :=
by
  sorry

end auditorium_rows_l201_20148


namespace angle_measure_l201_20174

theorem angle_measure (α : ℝ) 
  (h1 : 90 - α + (180 - α) = 180) : 
  α = 45 := 
by 
  sorry

end angle_measure_l201_20174


namespace taller_tree_height_is_108_l201_20196

variables (H : ℝ)

-- Conditions
def taller_tree_height := H
def shorter_tree_height := H - 18
def ratio_condition := (H - 18) / H = 5 / 6

-- Theorem to prove
theorem taller_tree_height_is_108 (hH : 0 < H) (h_ratio : ratio_condition H) : taller_tree_height H = 108 :=
sorry

end taller_tree_height_is_108_l201_20196


namespace scientific_notation_3050000_l201_20172

def scientific_notation (n : ℕ) : String :=
  "3.05 × 10^6"

theorem scientific_notation_3050000 :
  scientific_notation 3050000 = "3.05 × 10^6" :=
by
  sorry

end scientific_notation_3050000_l201_20172


namespace simplify_nested_fourth_roots_l201_20127

variable (M : ℝ)
variable (hM : M > 1)

theorem simplify_nested_fourth_roots : 
  (M^(1/4) * (M^(1/4) * (M^(1/4) * M)^(1/4))^(1/4))^(1/4) = M^(21/64) := by
  sorry

end simplify_nested_fourth_roots_l201_20127


namespace geometric_sequence_sufficient_not_necessary_l201_20179

theorem geometric_sequence_sufficient_not_necessary (a b c : ℝ) :
  (∃ r : ℝ, a = b * r ∧ b = c * r) → (b^2 = a * c) ∧ ¬ ( (b^2 = a * c) → (∃ r : ℝ, a = b * r ∧ b = c * r) ) :=
by
  sorry

end geometric_sequence_sufficient_not_necessary_l201_20179


namespace closest_total_population_of_cities_l201_20141

theorem closest_total_population_of_cities 
    (n_cities : ℕ) (avg_population_lower avg_population_upper : ℕ)
    (h_lower : avg_population_lower = 3800) (h_upper : avg_population_upper = 4200) :
  (25:ℕ) * (4000:ℕ) = 100000 :=
by
  sorry

end closest_total_population_of_cities_l201_20141


namespace norris_money_left_l201_20183

def sept_savings : ℕ := 29
def oct_savings : ℕ := 25
def nov_savings : ℕ := 31
def hugo_spent  : ℕ := 75
def total_savings : ℕ := sept_savings + oct_savings + nov_savings
def norris_left : ℕ := total_savings - hugo_spent

theorem norris_money_left : norris_left = 10 := by
  unfold norris_left total_savings sept_savings oct_savings nov_savings hugo_spent
  sorry

end norris_money_left_l201_20183


namespace product_of_roots_l201_20125

theorem product_of_roots (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
    (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 :=
by sorry

end product_of_roots_l201_20125


namespace geometric_progression_arcsin_sin_l201_20108

noncomputable def least_positive_t : ℝ :=
  9 + 4 * Real.sqrt 5

theorem geometric_progression_arcsin_sin 
  (α : ℝ) 
  (hα1: 0 < α) 
  (hα2: α < Real.pi / 2) 
  (t : ℝ) 
  (h : ∀ (a b c d : ℝ), 
    a = Real.arcsin (Real.sin α) ∧ 
    b = Real.arcsin (Real.sin (3 * α)) ∧ 
    c = Real.arcsin (Real.sin (5 * α)) ∧ 
    d = Real.arcsin (Real.sin (t * α)) → 
    b / a = c / b ∧ c / b = d / c) : 
  t = least_positive_t :=
sorry

end geometric_progression_arcsin_sin_l201_20108


namespace probability_individual_selected_l201_20199

theorem probability_individual_selected :
  ∀ (N M : ℕ) (m : ℕ), N = 100 → M = 5 → (m < N) →
  (probability_of_selecting_m : ℝ) =
  (1 / N * M) :=
by
  intros N M m hN hM hm
  sorry

end probability_individual_selected_l201_20199


namespace bottle_caps_total_l201_20167

-- Mathematical conditions
def x : ℕ := 18
def y : ℕ := 63

-- Statement to prove
theorem bottle_caps_total : x + y = 81 :=
by
  -- The proof is skipped as indicated by 'sorry'
  sorry

end bottle_caps_total_l201_20167


namespace baked_by_brier_correct_l201_20143

def baked_by_macadams : ℕ := 20
def baked_by_flannery : ℕ := 17
def total_baked : ℕ := 55

def baked_by_brier : ℕ := total_baked - (baked_by_macadams + baked_by_flannery)

-- Theorem statement
theorem baked_by_brier_correct : baked_by_brier = 18 := 
by
  -- proof will go here 
  sorry

end baked_by_brier_correct_l201_20143


namespace base12_mod_9_remainder_l201_20131

noncomputable def base12_to_base10 (n : ℕ) : ℕ :=
  1 * 12^3 + 7 * 12^2 + 3 * 12^1 + 2 * 12^0

theorem base12_mod_9_remainder : (base12_to_base10 1732) % 9 = 2 := by
  sorry

end base12_mod_9_remainder_l201_20131


namespace probability_factor_90_less_than_10_l201_20184

-- Definitions from conditions
def number_factors_90 : ℕ := 12
def factors_90_less_than_10 : ℕ := 6

-- The corresponding proof problem
theorem probability_factor_90_less_than_10 : 
  (factors_90_less_than_10 / number_factors_90 : ℚ) = 1 / 2 :=
by
  sorry  -- proof to be filled in

end probability_factor_90_less_than_10_l201_20184


namespace lateral_surface_area_of_rotated_square_l201_20155

noncomputable def lateralSurfaceAreaOfRotatedSquare (side_length : ℝ) : ℝ :=
  2 * Real.pi * side_length * side_length

theorem lateral_surface_area_of_rotated_square :
  lateralSurfaceAreaOfRotatedSquare 1 = 2 * Real.pi :=
by
  sorry

end lateral_surface_area_of_rotated_square_l201_20155


namespace cube_weight_l201_20181

theorem cube_weight (l1 l2 V1 V2 k : ℝ) (h1: l2 = 2 * l1) (h2: V1 = l1^3) (h3: V2 = (2 * l1)^3) (h4: w2 = 48) (h5: V2 * k = w2) (h6: V1 * k = w1):
  w1 = 6 :=
by
  sorry

end cube_weight_l201_20181


namespace find_two_digit_number_l201_20112

open Nat

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem find_two_digit_number :
  ∃ N : ℕ, 
    (10 ≤ N ∧ N < 100) ∧ 
    (N % 2 = 1) ∧ 
    (N % 9 = 0) ∧ 
    is_perfect_square ((N / 10) * (N % 10)) ∧ 
    N = 99 :=
by
  sorry

end find_two_digit_number_l201_20112


namespace cos_shifted_eq_l201_20132

noncomputable def cos_shifted (theta : ℝ) (h1 : Real.cos theta = -12 / 13) (h2 : theta ∈ Set.Ioo Real.pi (3 / 2 * Real.pi)) : Real :=
  Real.cos (theta + Real.pi / 4)

theorem cos_shifted_eq (theta : ℝ) (h1 : Real.cos theta = -12 / 13) (h2 : theta ∈ Set.Ioo Real.pi (3 / 2 * Real.pi)) :
  cos_shifted theta h1 h2 = -7 * Real.sqrt 2 / 26 := 
by
  sorry

end cos_shifted_eq_l201_20132


namespace range_of_m_l201_20194

noncomputable def prop_p (m : ℝ) : Prop :=
0 < m ∧ m < 1 / 3

noncomputable def prop_q (m : ℝ) : Prop :=
0 < m ∧ m < 15

theorem range_of_m (m : ℝ) : (prop_p m ∧ ¬ prop_q m) ∨ (¬ prop_p m ∧ prop_q m) ↔ 1 / 3 ≤ m ∧ m < 15 :=
sorry

end range_of_m_l201_20194


namespace ln_1_2_over_6_gt_e_l201_20163

theorem ln_1_2_over_6_gt_e :
  let x := 1.2
  let exp1 := x^6
  let exp2 := (1.44)^2 * 1.44
  let final_val := 2.0736 * 1.44
  final_val > 2.718 :=
by {
  sorry
}

end ln_1_2_over_6_gt_e_l201_20163


namespace marys_score_l201_20177

def score (c w : ℕ) : ℕ := 30 + 4 * c - w
def valid_score_range (s : ℕ) : Prop := s > 90 ∧ s ≤ 170

theorem marys_score : ∃ c w : ℕ, c + w ≤ 35 ∧ score c w = 170 ∧ 
  ∀ (s : ℕ), (valid_score_range s ∧ ∃ c' w', score c' w' = s ∧ c' + w' ≤ 35) → 
  (s = 170) :=
by
  sorry

end marys_score_l201_20177


namespace walking_speed_l201_20150

-- Define the constants and variables
def speed_there := 25 -- speed from village to post-office in kmph
def total_time := 5.8 -- total round trip time in hours
def distance := 20.0 -- distance to the post-office in km
 
-- Define the theorem that needs to be proved
theorem walking_speed :
  ∃ (speed_back : ℝ), speed_back = 4 := 
by
  sorry

end walking_speed_l201_20150


namespace grade_assignment_ways_l201_20161

theorem grade_assignment_ways : (4 ^ 12) = 16777216 :=
by
  -- mathematical proof
  sorry

end grade_assignment_ways_l201_20161


namespace find_diminished_value_l201_20187

theorem find_diminished_value :
  ∃ (x : ℕ), 1015 - x = Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 12 16) 18) 21) 28 :=
by
  use 7
  simp
  unfold Nat.lcm
  sorry

end find_diminished_value_l201_20187


namespace greatest_sum_of_other_two_roots_l201_20168

noncomputable def polynomial (x : ℝ) (k : ℝ) : ℝ :=
  x^3 - k * x^2 + 20 * x - 15

theorem greatest_sum_of_other_two_roots (k x1 x2 : ℝ) (h : polynomial 3 k = 0) (hx : x1 * x2 = 5)
  (h_prod_sum : 3 * x1 + 3 * x2 + x1 * x2 = 20) : x1 + x2 = 5 :=
by
  sorry

end greatest_sum_of_other_two_roots_l201_20168


namespace triangle_strike_interval_l201_20147

/-- Jacob strikes the cymbals every 7 beats and the triangle every t beats.
    Given both are struck at the same time every 14 beats, this proves t = 2. -/
theorem triangle_strike_interval :
  ∃ t : ℕ, t ≠ 7 ∧ (∀ n : ℕ, (7 * n % t = 0) → ∃ k : ℕ, 7 * n = 14 * k) ∧ t = 2 :=
by
  use 2
  sorry

end triangle_strike_interval_l201_20147
