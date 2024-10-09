import Mathlib

namespace cos_value_l1776_177697

-- Given condition
axiom sin_condition (α : ℝ) : Real.sin (Real.pi / 6 + α) = 2 / 3

-- The theorem we need to prove
theorem cos_value (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = 2 / 3) : 
  Real.cos (Real.pi / 3 - α) = 2 / 3 := 
by 
  sorry

end cos_value_l1776_177697


namespace solution_of_r_and_s_l1776_177617

theorem solution_of_r_and_s 
  (r s : ℝ)
  (hr : (r-5)*(r+5) = 25*r - 125)
  (hs : (s-5)*(s+5) = 25*s - 125)
  (h_distinct : r ≠ s)
  (h_order : r > s) : 
  r - s = 15 := by
sorry

end solution_of_r_and_s_l1776_177617


namespace concert_ratio_l1776_177616

theorem concert_ratio (a c : ℕ) (h1 : 30 * a + 15 * c = 2250) (h2 : a ≥ 1) (h3 : c ≥ 1) :
  a = 50 ∧ c = 50 ∧ a = c := 
sorry

end concert_ratio_l1776_177616


namespace intersection_point_in_AB_l1776_177602

def A (p : ℝ × ℝ) : Prop := p.snd = 2 * p.fst - 1
def B (p : ℝ × ℝ) : Prop := p.snd = p.fst + 3

theorem intersection_point_in_AB : (4, 7) ∈ {p : ℝ × ℝ | A p} ∩ {p : ℝ × ℝ | B p} :=
by
  sorry

end intersection_point_in_AB_l1776_177602


namespace rational_numbers_include_positives_and_negatives_l1776_177626

theorem rational_numbers_include_positives_and_negatives :
  ∃ (r : ℚ), r > 0 ∧ ∃ (r' : ℚ), r' < 0 :=
by
  sorry

end rational_numbers_include_positives_and_negatives_l1776_177626


namespace cash_after_brokerage_l1776_177639

theorem cash_after_brokerage (sale_amount : ℝ) (brokerage_rate : ℝ) :
  sale_amount = 109.25 → brokerage_rate = 0.0025 →
  (sale_amount - sale_amount * brokerage_rate) = 108.98 :=
by
  intros h1 h2
  sorry

end cash_after_brokerage_l1776_177639


namespace sum_of_squares_l1776_177695

theorem sum_of_squares (a b n : ℕ) (h : ∃ k : ℕ, a^2 + 2 * n * b^2 = k^2) : 
  ∃ e f : ℕ, a^2 + n * b^2 = e^2 + f^2 :=
by
  sorry

-- Theorem parameters and logical flow explained:

-- a, b, n : ℕ                  -- Natural number inputs
-- h : ∃ k : ℕ, a^2 + 2 * n * b^2 = k^2  -- Condition given in the problem that a^2 + 2nb^2 is a perfect square
-- Prove that there exist natural numbers e and f such that a^2 + nb^2 = e^2 + f^2

end sum_of_squares_l1776_177695


namespace dig_eq_conditions_l1776_177619

theorem dig_eq_conditions (n k : ℕ) 
  (h1 : 10^(k-1) ≤ n^n ∧ n^n < 10^k)
  (h2 : 10^(n-1) ≤ k^k ∧ k^k < 10^n) :
  (n = 1 ∧ k = 1) ∨ (n = 8 ∧ k = 8) ∨ (n = 9 ∧ k = 9) :=
by
  sorry

end dig_eq_conditions_l1776_177619


namespace angelaAgeInFiveYears_l1776_177656

namespace AgeProblem

variables (A B : ℕ) -- Define Angela's and Beth's current age as natural numbers.

-- Condition 1: Angela is four times as old as Beth.
axiom angelaAge : A = 4 * B

-- Condition 2: Five years ago, the sum of their ages was 45 years.
axiom ageSumFiveYearsAgo : (A - 5) + (B - 5) = 45

-- Theorem: Prove that Angela's age in 5 years will be 49.
theorem angelaAgeInFiveYears : A + 5 = 49 :=
by {
  -- proof goes here
  sorry
}

end AgeProblem

end angelaAgeInFiveYears_l1776_177656


namespace part_a_part_b_l1776_177699

def is_multiple_of_9 (n : ℕ) := n % 9 = 0
def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

theorem part_a : ∃ n : ℕ, is_multiple_of_9 n ∧ digit_sum n = 81 ∧ (n / 9) = 111111111 := 
sorry

theorem part_b : ∃ n1 n2 n3 n4 : ℕ,
  is_multiple_of_9 n1 ∧
  is_multiple_of_9 n2 ∧
  is_multiple_of_9 n3 ∧
  is_multiple_of_9 n4 ∧
  digit_sum n1 = 27 ∧ digit_sum n2 = 27 ∧ digit_sum n3 = 27 ∧ digit_sum n4 = 27 ∧
  (n1 / 9) + 1 = (n2 / 9) ∧ 
  (n2 / 9) + 1 = (n3 / 9) ∧ 
  (n3 / 9) + 1 = (n4 / 9) ∧ 
  (n4 / 9) < 1111 := 
sorry

end part_a_part_b_l1776_177699


namespace find_d_l1776_177613

theorem find_d (d : ℝ) (h : 3 * (2 - (π / 2)) = 6 + d * π) : d = -3 / 2 :=
by
  sorry

end find_d_l1776_177613


namespace solution_l1776_177625

def problem (a b : ℝ) : Prop :=
  ∀ (x : ℝ), (x + a) * (x - 3) = x^2 + 2 * x - b

theorem solution (a b : ℝ) (h : problem a b) : a - b = -10 :=
  sorry

end solution_l1776_177625


namespace number_of_arrangements_l1776_177610

theorem number_of_arrangements (V T : ℕ) (hV : V = 3) (hT : T = 4) :
  ∃ n : ℕ, n = 36 :=
by
  sorry

end number_of_arrangements_l1776_177610


namespace djibo_age_sum_years_ago_l1776_177674

theorem djibo_age_sum_years_ago (x : ℕ) (h₁: 17 - x + 28 - x = 35) : x = 5 :=
by
  -- proof is omitted as per instructions
  sorry

end djibo_age_sum_years_ago_l1776_177674


namespace num_other_adults_l1776_177652

-- Define the variables and conditions
def num_baskets : ℕ := 15
def eggs_per_basket : ℕ := 12
def eggs_per_person : ℕ := 9
def shonda_kids : ℕ := 2
def kids_friends : ℕ := 10
def num_participants : ℕ := (num_baskets * eggs_per_basket) / eggs_per_person

-- Prove the number of other adults at the Easter egg hunt
theorem num_other_adults : (num_participants - (shonda_kids + kids_friends + 1)) = 7 := by
  sorry

end num_other_adults_l1776_177652


namespace find_first_number_l1776_177650

/-- Given a sequence of 6 numbers b_1, b_2, ..., b_6 such that:
  1. For n ≥ 2, b_{2n} = b_{2n-1}^2
  2. For n ≥ 2, b_{2n+1} = (b_{2n} * b_{2n-1})^2
And the sequence ends as: b_4 = 16, b_5 = 256, and b_6 = 65536,
prove that the first number b_1 is 1/2. -/
theorem find_first_number : 
  ∃ b : ℕ → ℝ, b 6 = 65536 ∧ b 5 = 256 ∧ b 4 = 16 ∧ 
  (∀ n ≥ 2, b (2 * n) = (b (2 * n - 1)) ^ 2) ∧
  (∀ n ≥ 2, b (2 * n + 1) = (b (2 * n) * b (2 * n - 1)) ^ 2) ∧ 
  b 1 = 1/2 :=
by
  sorry

end find_first_number_l1776_177650


namespace consecutive_sum_is_10_l1776_177629

theorem consecutive_sum_is_10 (a : ℕ) (h : a + (a + 1) + (a + 2) + (a + 3) + (a + 4) = 50) : a + 2 = 10 :=
sorry

end consecutive_sum_is_10_l1776_177629


namespace burger_cost_proof_l1776_177609

variable {burger_cost fries_cost salad_cost total_cost : ℕ}
variable {quantity_of_fries : ℕ}

theorem burger_cost_proof (h_fries_cost : fries_cost = 2)
    (h_salad_cost : salad_cost = 3 * fries_cost)
    (h_quantity_of_fries : quantity_of_fries = 2)
    (h_total_cost : total_cost = 15)
    (h_equation : burger_cost + (quantity_of_fries * fries_cost) + salad_cost = total_cost) :
    burger_cost = 5 :=
by 
  sorry

end burger_cost_proof_l1776_177609


namespace soccer_ball_price_l1776_177692

theorem soccer_ball_price 
  (B S V : ℕ) 
  (h1 : (B + S + V) / 3 = 36)
  (h2 : B = V + 10)
  (h3 : S = V + 8) : 
  S = 38 := 
by 
  sorry

end soccer_ball_price_l1776_177692


namespace train_time_original_l1776_177666

theorem train_time_original (D : ℝ) (T : ℝ) 
  (h1 : D = 48 * T) 
  (h2 : D = 60 * (2/3)) : T = 5 / 6 := 
by
  sorry

end train_time_original_l1776_177666


namespace min_value_a_squared_plus_b_squared_l1776_177678

theorem min_value_a_squared_plus_b_squared :
  ∃ (a b : ℝ), (b = 3 * a - 6) → (a^2 + b^2 = 18 / 5) :=
by
  sorry

end min_value_a_squared_plus_b_squared_l1776_177678


namespace unique_ones_digits_divisible_by_8_l1776_177606

/-- Carla likes numbers that are divisible by 8.
    We want to show that there are 5 unique ones digits for such numbers. -/
theorem unique_ones_digits_divisible_by_8 : 
  (Finset.card 
    (Finset.image (fun n => n % 10) 
                  (Finset.filter (fun n => n % 8 = 0) (Finset.range 100)))) = 5 := 
by
  sorry

end unique_ones_digits_divisible_by_8_l1776_177606


namespace number_of_cows_brought_l1776_177696

/--
A certain number of cows and 10 goats are brought for Rs. 1500. 
If the average price of a goat is Rs. 70, and the average price of a cow is Rs. 400, 
then the number of cows brought is 2.
-/
theorem number_of_cows_brought : 
  ∃ c : ℕ, ∃ g : ℕ, g = 10 ∧ (70 * g + 400 * c = 1500) ∧ c = 2 :=
sorry

end number_of_cows_brought_l1776_177696


namespace number_of_small_cubes_l1776_177648

theorem number_of_small_cubes (X : ℕ) (h1 : ∃ k, k = 29 - X) (h2 : 4 * 4 * 4 = 64) (h3 : X + 8 * (29 - X) = 64) : X = 24 :=
by
  sorry

end number_of_small_cubes_l1776_177648


namespace stones_required_to_pave_hall_l1776_177673

noncomputable def hall_length_meters : ℝ := 36
noncomputable def hall_breadth_meters : ℝ := 15
noncomputable def stone_length_dms : ℝ := 4
noncomputable def stone_breadth_dms : ℝ := 5

theorem stones_required_to_pave_hall :
  let hall_length_dms := hall_length_meters * 10
  let hall_breadth_dms := hall_breadth_meters * 10
  let hall_area_dms_squared := hall_length_dms * hall_breadth_dms
  let stone_area_dms_squared := stone_length_dms * stone_breadth_dms
  let number_of_stones := hall_area_dms_squared / stone_area_dms_squared
  number_of_stones = 2700 :=
by
  sorry

end stones_required_to_pave_hall_l1776_177673


namespace max_ratio_1099_l1776_177691

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem max_ratio_1099 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 → (sum_of_digits n : ℚ) / n ≤ (sum_of_digits 1099 : ℚ) / 1099 :=
by
  intros n hn
  sorry

end max_ratio_1099_l1776_177691


namespace probability_neither_perfect_square_nor_cube_l1776_177686

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
noncomputable def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

theorem probability_neither_perfect_square_nor_cube :
  let total_numbers := 200
  let count_squares := (14 : ℕ)  -- Corresponds to the number of perfect squares
  let count_cubes := (5 : ℕ)  -- Corresponds to the number of perfect cubes
  let count_sixth_powers := (2 : ℕ)  -- Corresponds to the number of sixth powers
  let count_ineligible := count_squares + count_cubes - count_sixth_powers
  let count_eligible := total_numbers - count_ineligible
  (count_eligible : ℚ) / (total_numbers : ℚ) = 183 / 200 :=
by sorry

end probability_neither_perfect_square_nor_cube_l1776_177686


namespace pen_price_equation_l1776_177667

theorem pen_price_equation
  (x y : ℤ)
  (h1 : 100 * x - y = 100)
  (h2 : 2 * y - 100 * x = 200) : x = 4 :=
by
  sorry

end pen_price_equation_l1776_177667


namespace Bernardo_wins_with_smallest_M_l1776_177664

-- Define the operations
def Bernardo_op (n : ℕ) : ℕ := 3 * n
def Lucas_op (n : ℕ) : ℕ := n + 75

-- Define the game behavior
def game_sequence (M : ℕ) : List ℕ :=
  [M, Bernardo_op M, Lucas_op (Bernardo_op M), Bernardo_op (Lucas_op (Bernardo_op M)),
   Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M))),
   Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M)))),
   Lucas_op (Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M))))),
   Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op (Lucas_op (Bernardo_op M))))))]

-- Define winning condition
def Bernardo_wins (M : ℕ) : Prop :=
  let seq := game_sequence M
  seq.get! 5 < 1200 ∧ seq.get! 6 >= 1200

-- Sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

-- The final theorem statement
theorem Bernardo_wins_with_smallest_M :
  Bernardo_wins 9 ∧ (∀ M < 9, ¬Bernardo_wins M) ∧ sum_of_digits 9 = 9 :=
by
  sorry

end Bernardo_wins_with_smallest_M_l1776_177664


namespace simplify_fraction_90_150_l1776_177676

theorem simplify_fraction_90_150 :
  let num := 90
  let denom := 150
  let gcd := 30
  2 * 3^2 * 5 = num →
  2 * 3 * 5^2 = denom →
  (num / gcd) = 3 →
  (denom / gcd) = 5 →
  num / denom = (3 / 5) :=
by
  intros h1 h2 h3 h4
  sorry

end simplify_fraction_90_150_l1776_177676


namespace largest_lcm_value_l1776_177634

theorem largest_lcm_value :
  max (max (max (max (max (Nat.lcm 18 3) (Nat.lcm 18 6)) (Nat.lcm 18 9)) (Nat.lcm 18 12)) (Nat.lcm 18 15)) (Nat.lcm 18 18) = 90 := by
sorry

end largest_lcm_value_l1776_177634


namespace kathryn_remaining_money_l1776_177668

/-- Define the conditions --/
def rent := 1200
def salary := 5000
def food_and_travel_expenses := 2 * rent
def new_rent := rent / 2
def total_expenses := food_and_travel_expenses + new_rent
def remaining_money := salary - total_expenses

/-- Theorem to be proved --/
theorem kathryn_remaining_money : remaining_money = 2000 := by
  sorry

end kathryn_remaining_money_l1776_177668


namespace points_opposite_side_of_line_l1776_177679

theorem points_opposite_side_of_line :
  (∀ a : ℝ, ((2 * 2 - 3 * 1 + a) * (2 * 4 - 3 * 3 + a) < 0) ↔ -1 < a ∧ a < 1) :=
by sorry

end points_opposite_side_of_line_l1776_177679


namespace towel_percentage_decrease_l1776_177646

theorem towel_percentage_decrease
  (L B: ℝ)
  (original_area : ℝ := L * B)
  (new_length : ℝ := 0.70 * L)
  (new_breadth : ℝ := 0.75 * B)
  (new_area : ℝ := new_length * new_breadth) :
  ((original_area - new_area) / original_area) * 100 = 47.5 := 
by 
  sorry

end towel_percentage_decrease_l1776_177646


namespace oxen_eat_as_much_as_buffaloes_or_cows_l1776_177670

theorem oxen_eat_as_much_as_buffaloes_or_cows
  (B C O : ℝ)
  (h1 : 3 * B = 4 * C)
  (h2 : (15 * B + 8 * O + 24 * C) * 36 = (30 * B + 8 * O + 64 * C) * 18) :
  3 * B = 4 * O :=
by sorry

end oxen_eat_as_much_as_buffaloes_or_cows_l1776_177670


namespace quadratic_one_real_root_positive_m_l1776_177655

theorem quadratic_one_real_root_positive_m (m : ℝ) (h : m > 0) :
  (∀ x : ℝ, x^2 + 6 * m * x + 2 * m = 0 → ((6 * m)^2 - 4 * 1 * (2 * m) = 0)) → m = 2 / 9 :=
by
  sorry

end quadratic_one_real_root_positive_m_l1776_177655


namespace log_one_plus_x_sq_lt_x_sq_l1776_177684

theorem log_one_plus_x_sq_lt_x_sq {x : ℝ} (hx : 0 < x) : 
  Real.log (1 + x^2) < x^2 := 
sorry

end log_one_plus_x_sq_lt_x_sq_l1776_177684


namespace lesser_number_is_21_5_l1776_177669

theorem lesser_number_is_21_5
  (x y : ℝ)
  (h1 : x + y = 50)
  (h2 : x - y = 7) :
  y = 21.5 :=
by
  sorry

end lesser_number_is_21_5_l1776_177669


namespace edward_score_l1776_177614

theorem edward_score (total_points : ℕ) (friend_points : ℕ) 
  (h1 : total_points = 13) (h2 : friend_points = 6) : 
  ∃ edward_points : ℕ, edward_points = 7 :=
by
  sorry

end edward_score_l1776_177614


namespace factorize_expr_l1776_177636

theorem factorize_expr (a b : ℝ) : 2 * a^2 - a * b = a * (2 * a - b) := 
by
  sorry

end factorize_expr_l1776_177636


namespace domain_of_function_l1776_177642

variable (x : ℝ)

theorem domain_of_function :
  {x : ℝ | x + 3 ≥ 0 ∧ 2 - x ≠ 0} =
  {x : ℝ | x ≥ -3 ∧ x ≠ 2} :=
by
  sorry

end domain_of_function_l1776_177642


namespace cubic_identity_l1776_177633

variable {a b c : ℝ}

theorem cubic_identity (h1 : a + b + c = 13) (h2 : ab + ac + bc = 30) : a^3 + b^3 + c^3 - 3 * a * b * c = 1027 := 
by 
  sorry

end cubic_identity_l1776_177633


namespace binom_10_1_eq_10_l1776_177618

theorem binom_10_1_eq_10 : Nat.choose 10 1 = 10 := by
  sorry

end binom_10_1_eq_10_l1776_177618


namespace ratio_ac_l1776_177641

-- Definitions based on conditions
variables (a b c : ℕ)
variables (x y : ℕ)

-- Conditions
def ratio_ab := (a : ℚ) / (b : ℚ) = 2 / 3
def ratio_bc := (b : ℚ) / (c : ℚ) = 1 / 5

-- Theorem to prove the desired ratio
theorem ratio_ac (h1 : ratio_ab a b) (h2 : ratio_bc b c) : (a : ℚ) / (c : ℚ) = 2 / 15 :=
by
  sorry

end ratio_ac_l1776_177641


namespace peanuts_in_box_l1776_177603

theorem peanuts_in_box (initial_peanuts : ℕ) (added_peanuts : ℕ) (h1 : initial_peanuts = 4) (h2 : added_peanuts = 2) : initial_peanuts + added_peanuts = 6 := by
  sorry

end peanuts_in_box_l1776_177603


namespace slope_probability_l1776_177657

noncomputable def probability_of_slope_gte (x y : ℝ) (Q : ℝ × ℝ) : ℝ :=
  if y - 1 / 4 ≥ (2 / 3) * (x - 3 / 4) then 1 else 0

theorem slope_probability :
  let unit_square_area := 1  -- the area of the unit square
  let valid_area := (1 / 2) * (5 / 8) * (5 / 12) -- area of the triangle above the line
  valid_area / unit_square_area = 25 / 96 :=
sorry

end slope_probability_l1776_177657


namespace triangle_ratio_inequality_l1776_177627

/-- Given a triangle ABC, R is the radius of the circumscribed circle, 
    r is the radius of the inscribed circle, a is the length of the longest side,
    and h is the length of the shortest altitude. Prove that R / r > a / h. -/
theorem triangle_ratio_inequality
  (ABC : Triangle) (R r a h : ℝ)
  (hR : 2 * R ≥ a)
  (hr : 2 * r < h) :
  (R / r) > (a / h) :=
by
  -- sorry is used to skip the proof
  sorry

end triangle_ratio_inequality_l1776_177627


namespace sum_six_digit_odd_and_multiples_of_3_l1776_177622

-- Definitions based on conditions
def num_six_digit_odd_numbers : Nat := 9 * (10 ^ 4) * 5

def num_six_digit_multiples_of_3 : Nat := 900000 / 3

-- Proof statement
theorem sum_six_digit_odd_and_multiples_of_3 : 
  num_six_digit_odd_numbers + num_six_digit_multiples_of_3 = 750000 := 
by 
  sorry

end sum_six_digit_odd_and_multiples_of_3_l1776_177622


namespace integer_solutions_of_cubic_equation_l1776_177628

theorem integer_solutions_of_cubic_equation :
  ∀ (n m : ℤ),
    n ^ 6 + 3 * n ^ 5 + 3 * n ^ 4 + 2 * n ^ 3 + 3 * n ^ 2 + 3 * n + 1 = m ^ 3 ↔
    (n = 0 ∧ m = 1) ∨ (n = -1 ∧ m = 0) :=
by
  intro n m
  apply Iff.intro
  { intro h
    sorry }
  { intro h
    sorry }

end integer_solutions_of_cubic_equation_l1776_177628


namespace inequality_problem_l1776_177621

theorem inequality_problem (x y a b : ℝ) (h1 : x > y) (h2 : y > 1) (h3 : 0 < a) (h4 : a < b) (h5 : b < 1) : (a ^ x < b ^ y) :=
by 
  sorry

end inequality_problem_l1776_177621


namespace trapezium_top_width_l1776_177644

theorem trapezium_top_width (bottom_width : ℝ) (height : ℝ) (area : ℝ) (top_width : ℝ) 
  (h1 : bottom_width = 8) 
  (h2 : height = 50) 
  (h3 : area = 500) : top_width = 12 :=
by
  -- Definitions
  have h_formula : area = 1 / 2 * (top_width + bottom_width) * height := by sorry
  -- Applying given conditions to the formula
  rw [h1, h2, h3] at h_formula
  -- Solve for top_width
  sorry

end trapezium_top_width_l1776_177644


namespace find_b_l1776_177653

theorem find_b (b c x1 x2 : ℝ)
  (h_parabola_intersects_x_axis : (x1 ≠ x2) ∧ x1 * x2 = c ∧ x1 + x2 = -b ∧ x2 - x1 = 1)
  (h_parabola_intersects_y_axis : c ≠ 0)
  (h_length_ab : x2 - x1 = 1)
  (h_area_abc : (1 / 2) * (x2 - x1) * |c| = 1)
  : b = -3 :=
sorry

end find_b_l1776_177653


namespace inequality_nonempty_solution_set_l1776_177604

theorem inequality_nonempty_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x-3| + |x-4| < a) ↔ a > 1 :=
by
  sorry

end inequality_nonempty_solution_set_l1776_177604


namespace remainder_three_l1776_177672

theorem remainder_three (n : ℕ) (h1 : Nat.Prime (n + 3)) (h2 : Nat.Prime (n + 7)) : n % 3 = 1 :=
sorry

end remainder_three_l1776_177672


namespace pens_solution_exists_l1776_177654

-- Definition of the conditions
def pen_cost_eq (x y : ℕ) : Prop :=
  17 * x + 12 * y = 150

-- Proof problem statement that follows from the conditions
theorem pens_solution_exists :
  ∃ x y : ℕ, pen_cost_eq x y :=
by
  existsi (6 : ℕ)
  existsi (4 : ℕ)
  -- Normally the proof would go here, but as stated, we use sorry.
  sorry

end pens_solution_exists_l1776_177654


namespace hockey_pads_cost_l1776_177643

theorem hockey_pads_cost
  (initial_money : ℕ)
  (cost_hockey_skates : ℕ)
  (remaining_money : ℕ)
  (h : initial_money = 150)
  (h1 : cost_hockey_skates = initial_money / 2)
  (h2 : remaining_money = 25) :
  initial_money - cost_hockey_skates - 50 = remaining_money :=
by sorry

end hockey_pads_cost_l1776_177643


namespace angle_BAC_l1776_177640

theorem angle_BAC (A B C D : Type*) (AD BD CD : ℝ) (angle_BCA : ℝ) 
  (h_AD_BD : AD = BD) (h_BD_CD : BD = CD) (h_angle_BCA : angle_BCA = 40) :
  ∃ angle_BAC : ℝ, angle_BAC = 110 := 
sorry

end angle_BAC_l1776_177640


namespace no_solution_inequality_system_l1776_177661

theorem no_solution_inequality_system (m : ℝ) :
  (¬ ∃ x : ℝ, 2 * x - 1 < 3 ∧ x > m) ↔ m ≥ 2 :=
by
  sorry

end no_solution_inequality_system_l1776_177661


namespace michelle_travel_distance_l1776_177694

-- Define the conditions
def initial_fee : ℝ := 2
def charge_per_mile : ℝ := 2.5
def total_paid : ℝ := 12

-- Define the theorem to prove the distance Michelle traveled
theorem michelle_travel_distance : (total_paid - initial_fee) / charge_per_mile = 4 := by
  sorry

end michelle_travel_distance_l1776_177694


namespace total_clouds_counted_l1776_177677

def clouds_counted (carson_clouds : ℕ) (brother_factor : ℕ) : ℕ :=
  carson_clouds + (carson_clouds * brother_factor)

theorem total_clouds_counted (carson_clouds brother_factor total_clouds : ℕ) 
  (h₁ : carson_clouds = 6) (h₂ : brother_factor = 3) (h₃ : total_clouds = 24) :
  clouds_counted carson_clouds brother_factor = total_clouds :=
by
  sorry

end total_clouds_counted_l1776_177677


namespace quadratic_polynomial_inequality_l1776_177608

variable {a b c : ℝ}

theorem quadratic_polynomial_inequality (h1 : ∀ x : ℝ, a * x^2 + b * x + c < 0)
    (h2 : a < 0)
    (h3 : b^2 - 4 * a * c < 0) :
    b / a < c / a + 1 := 
by 
  sorry

end quadratic_polynomial_inequality_l1776_177608


namespace minimal_total_distance_l1776_177685

variable (A B : ℝ) -- Coordinates of houses A and B on a straight road
variable (h_dist : B - A = 50) -- The distance between A and B is 50 meters

-- Define a point X on the road
variable (X : ℝ)

-- Define the function that calculates the total distance from point X to A and B
def total_distance (A B X : ℝ) := abs (X - A) + abs (X - B)

-- The theorem stating that the total distance is minimized if X lies on the line segment AB
theorem minimal_total_distance : A ≤ X ∧ X ≤ B ↔ total_distance A B X = B - A :=
by
  sorry

end minimal_total_distance_l1776_177685


namespace total_cookies_l1776_177651

-- Definitions of the conditions
def cookies_in_bag : ℕ := 21
def bags_in_box : ℕ := 4
def boxes : ℕ := 2

-- Theorem stating the total number of cookies
theorem total_cookies : cookies_in_bag * bags_in_box * boxes = 168 := by
  sorry

end total_cookies_l1776_177651


namespace constants_A_B_C_l1776_177675

theorem constants_A_B_C (A B C : ℝ) (h₁ : ∀ x : ℝ, (x^2 + 5 * x - 6) / (x^4 + x^2) = A / x^2 + (B * x + C) / (x^2 + 1)) :
  A = -6 ∧ B = 0 ∧ C = 7 :=
by
  sorry

end constants_A_B_C_l1776_177675


namespace evaluate_f_neg3_l1776_177660

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 1

theorem evaluate_f_neg3 (a b c : ℝ) (h : f 3 a b c = 11) : f (-3) a b c = -9 := by
  sorry

end evaluate_f_neg3_l1776_177660


namespace max_side_length_l1776_177615

theorem max_side_length (a b c : ℕ) (h : a + b + c = 30) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_order : a ≤ b ∧ b ≤ c) (h_triangle_ineq : a + b > c) : c ≤ 14 := 
sorry

end max_side_length_l1776_177615


namespace diagonals_from_vertex_of_regular_polygon_l1776_177645

-- Definitions for the conditions in part a)
def exterior_angle (n : ℕ) : ℚ := 360 / n

-- Proof problem statement
theorem diagonals_from_vertex_of_regular_polygon
  (n : ℕ)
  (h1 : exterior_angle n = 36)
  : n - 3 = 7 :=
by sorry

end diagonals_from_vertex_of_regular_polygon_l1776_177645


namespace range_of_a_l1776_177620

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3 * a) ↔ (a ≤ -1 ∨ a ≥ 4) :=
by sorry

end range_of_a_l1776_177620


namespace apples_count_l1776_177681

theorem apples_count : (23 - 20 + 6 = 9) :=
by
  sorry

end apples_count_l1776_177681


namespace expected_interval_proof_l1776_177605

noncomputable def expected_interval_between_trains : ℝ := 3

theorem expected_interval_proof
  (northern_route_time southern_route_time : ℝ)
  (counter_clockwise_delay : ℝ)
  (home_to_work_less_than_work_to_home : ℝ) :
  northern_route_time = 17 →
  southern_route_time = 11 →
  counter_clockwise_delay = 75 / 60 →
  home_to_work_less_than_work_to_home = 1 →
  expected_interval_between_trains = 3 :=
by
  intros
  sorry

end expected_interval_proof_l1776_177605


namespace min_value_of_expression_l1776_177647

theorem min_value_of_expression (m n : ℝ) (h1 : m + 2 * n = 2) (h2 : m > 0) (h3 : n > 0) : 
  (1 / (m + 1) + 1 / (2 * n)) ≥ 4 / 3 :=
sorry

end min_value_of_expression_l1776_177647


namespace compare_x_y_l1776_177635

theorem compare_x_y (a b : ℝ) (h1 : a > b) (h2 : b > 1) (x y : ℝ)
  (hx : x = a + 1 / a) (hy : y = b + 1 / b) : x > y :=
by {
  sorry
}

end compare_x_y_l1776_177635


namespace bisection_method_correctness_l1776_177638

noncomputable def initial_interval_length : ℝ := 1
noncomputable def required_precision : ℝ := 0.01
noncomputable def minimum_bisections : ℕ := 7

theorem bisection_method_correctness :
  ∃ n : ℕ, (n ≥ minimum_bisections) ∧ (initial_interval_length / 2^n ≤ required_precision) :=
by
  sorry

end bisection_method_correctness_l1776_177638


namespace speed_of_slower_train_is_36_l1776_177600

-- Definitions used in the conditions
def length_of_train := 25 -- meters
def combined_length_of_trains := 2 * length_of_train -- meters
def time_to_pass := 18 -- seconds
def speed_of_faster_train := 46 -- km/hr
def conversion_factor := 1000 / 3600 -- to convert from km/hr to m/s

-- Prove that speed of the slower train is 36 km/hr
theorem speed_of_slower_train_is_36 :
  ∃ v : ℕ, v = 36 ∧ ((combined_length_of_trains : ℝ) = ((speed_of_faster_train - v) * conversion_factor * time_to_pass)) :=
sorry

end speed_of_slower_train_is_36_l1776_177600


namespace number_of_donuts_correct_l1776_177607

noncomputable def number_of_donuts_in_each_box :=
  let x : ℕ := 12
  let total_boxes : ℕ := 4
  let donuts_given_to_mom : ℕ := x
  let donuts_given_to_sister : ℕ := 6
  let donuts_left : ℕ := 30
  x

theorem number_of_donuts_correct :
  ∀ (x : ℕ),
  (total_boxes * x - donuts_given_to_mom - donuts_given_to_sister = donuts_left) → x = 12 :=
by
  sorry

end number_of_donuts_correct_l1776_177607


namespace train_crossing_signal_pole_l1776_177683

theorem train_crossing_signal_pole
  (length_train : ℕ)
  (same_length_platform : ℕ)
  (time_crossing_platform : ℕ)
  (h_train_platform : length_train = 420)
  (h_platform : same_length_platform = 420)
  (h_time_platform : time_crossing_platform = 60) : 
  (length_train / (length_train + same_length_platform / time_crossing_platform)) = 30 := 
by 
  sorry

end train_crossing_signal_pole_l1776_177683


namespace sin_B_triangle_area_l1776_177611

variable {a b c : ℝ}
variable {A B C : ℝ}

theorem sin_B (hC : C = 3 / 4 * Real.pi) (hSinA : Real.sin A = Real.sqrt 5 / 5) :
  Real.sin B = Real.sqrt 10 / 10 := by
  sorry

theorem triangle_area (hC : C = 3 / 4 * Real.pi) (hSinA : Real.sin A = Real.sqrt 5 / 5)
  (hDiff : c - a = 5 - Real.sqrt 10) (hSinB : Real.sin B = Real.sqrt 10 / 10) :
  1 / 2 * a * c * Real.sin B = 5 / 2 := by
  sorry

end sin_B_triangle_area_l1776_177611


namespace problem_statement_l1776_177665

theorem problem_statement (a b : ℝ) (h : a^2 + |b + 1| = 0) : (a + b)^2015 = -1 := by
  sorry

end problem_statement_l1776_177665


namespace even_digit_number_division_l1776_177624

theorem even_digit_number_division (N : ℕ) (n : ℕ) :
  (N % 2 = 0) ∧
  (∃ a b : ℕ, (∀ k : ℕ, N = a * 10^n + b → N = k * (a * b)) ∧
  ((N = (1000^(2*n - 1) + 1)^2 / 7) ∨
   (N = 12) ∨
   (N = (10^n + 2)^2 / 6) ∨
   (N = 1352) ∨
   (N = 15))) :=
sorry

end even_digit_number_division_l1776_177624


namespace setB_forms_right_triangle_l1776_177649

-- Define the sets of side lengths
def setA : (ℕ × ℕ × ℕ) := (2, 3, 4)
def setB : (ℕ × ℕ × ℕ) := (3, 4, 5)
def setC : (ℕ × ℕ × ℕ) := (5, 6, 7)
def setD : (ℕ × ℕ × ℕ) := (7, 8, 9)

-- Define the Pythagorean theorem condition
def isRightTriangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- The specific proof goal
theorem setB_forms_right_triangle : isRightTriangle 3 4 5 := by
  sorry

end setB_forms_right_triangle_l1776_177649


namespace trigonometric_identity_l1776_177663

theorem trigonometric_identity (α : ℝ) : 
  (1 + Real.cos (2 * α - 2 * Real.pi) + Real.cos (4 * α + 2 * Real.pi) - Real.cos (6 * α - Real.pi)) /
  (Real.cos (2 * Real.pi - 2 * α) + 2 * Real.cos (2 * α + Real.pi) ^ 2 - 1) = 
  2 * Real.cos (2 * α) :=
by sorry

end trigonometric_identity_l1776_177663


namespace avianna_blue_candles_l1776_177632

theorem avianna_blue_candles (r b : ℕ) (h1 : r = 45) (h2 : r/b = 5/3) : b = 27 :=
by sorry

end avianna_blue_candles_l1776_177632


namespace simplify_expr_l1776_177671

theorem simplify_expr (x : ℕ) (h : x = 2018) : x^2 + 2 * x - x * (x + 1) = x := by
  sorry

end simplify_expr_l1776_177671


namespace intersection_points_count_l1776_177687

open Real

theorem intersection_points_count :
  (∃ (x y : ℝ), ((x - ⌊x⌋)^2 + y^2 = x - ⌊x⌋) ∧ (y = 1/3 * x + 1)) →
  (∃ (n : ℕ), n = 8) :=
by
  -- proof goes here
  sorry

end intersection_points_count_l1776_177687


namespace fixed_point_of_shifted_exponential_l1776_177662

theorem fixed_point_of_shifted_exponential (a : ℝ) (H : a^0 = 1) : a^(3-3) + 3 = 4 :=
by
  sorry

end fixed_point_of_shifted_exponential_l1776_177662


namespace first_part_eq_19_l1776_177690

theorem first_part_eq_19 (x y : ℕ) (h1 : x + y = 36) (h2 : 8 * x + 3 * y = 203) : x = 19 :=
by sorry

end first_part_eq_19_l1776_177690


namespace division_problem_l1776_177658

theorem division_problem : 8900 / 6 / 4 = 1483.3333 :=
by sorry

end division_problem_l1776_177658


namespace kaleb_boxes_required_l1776_177698

/-- Kaleb's Games Packing Problem -/
theorem kaleb_boxes_required (initial_games sold_games box_capacity : ℕ) (h1 : initial_games = 76) (h2 : sold_games = 46) (h3 : box_capacity = 5) :
  ((initial_games - sold_games) / box_capacity) = 6 :=
by
  -- Skipping the proof
  sorry

end kaleb_boxes_required_l1776_177698


namespace six_digit_numbers_with_at_least_one_zero_correct_l1776_177688

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end six_digit_numbers_with_at_least_one_zero_correct_l1776_177688


namespace intersection_points_relation_l1776_177612

noncomputable def num_intersections (k : ℕ) : ℕ :=
  k * (k - 1) / 2

theorem intersection_points_relation (k : ℕ) :
  num_intersections (k + 1) = num_intersections k + k := by
sorry

end intersection_points_relation_l1776_177612


namespace first_box_weight_l1776_177680

theorem first_box_weight (X : ℕ) 
  (h1 : 11 + 5 + X = 18) : X = 2 := 
by
  sorry

end first_box_weight_l1776_177680


namespace log_prime_factor_inequality_l1776_177659

open Real

-- Define p(n) such that it returns the number of prime factors of n.
noncomputable def p (n: ℕ) : ℕ := sorry  -- This will be defined contextually for now

theorem log_prime_factor_inequality (n : ℕ) (hn : n > 0) : 
  log n ≥ (p n) * log 2 :=
by 
  sorry

end log_prime_factor_inequality_l1776_177659


namespace arcsin_one_eq_pi_div_two_l1776_177601

theorem arcsin_one_eq_pi_div_two : Real.arcsin 1 = Real.pi / 2 :=
by
  -- proof steps here
  sorry

end arcsin_one_eq_pi_div_two_l1776_177601


namespace original_students_count_l1776_177637

theorem original_students_count (N : ℕ) (T : ℕ) :
  (T = N * 85) →
  ((N - 5) * 90 = T - 300) →
  ((N - 8) * 95 = T - 465) →
  ((N - 15) * 100 = T - 955) →
  N = 30 :=
by
  intros h1 h2 h3 h4
  sorry

end original_students_count_l1776_177637


namespace minimum_shift_value_l1776_177630

noncomputable def f (x : ℝ) : ℝ := Real.cos x

theorem minimum_shift_value :
  ∃ m > 0, ∀ x, f (x + m) = Real.sin x ∧ m = 3 * Real.pi / 2 :=
by
  sorry

end minimum_shift_value_l1776_177630


namespace dilation_at_origin_neg3_l1776_177689

-- Define the dilation matrix centered at the origin with scale factor -3
def dilation_matrix (scale_factor : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![scale_factor, 0], ![0, scale_factor]]

-- The theorem stating that a dilation with scale factor -3 results in the specified matrix
theorem dilation_at_origin_neg3 :
  dilation_matrix (-3) = ![![(-3 : ℝ), 0], ![0, -3]] :=
sorry

end dilation_at_origin_neg3_l1776_177689


namespace sum_of_digits_of_valid_n_eq_seven_l1776_177682

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_valid_n (n : ℕ) : Prop :=
  (500 < n) ∧ (Nat.gcd 70 (n + 150) = 35) ∧ (Nat.gcd (n + 70) 150 = 50)

theorem sum_of_digits_of_valid_n_eq_seven :
  ∃ n : ℕ, is_valid_n n ∧ sum_of_digits n = 7 := by
  sorry

end sum_of_digits_of_valid_n_eq_seven_l1776_177682


namespace Sonja_oil_used_l1776_177623

theorem Sonja_oil_used :
  ∀ (oil peanuts total_weight : ℕ),
  (2 * oil + 8 * peanuts = 10) → (total_weight = 20) →
  ((20 / 10) * 2 = 4) :=
by 
  sorry

end Sonja_oil_used_l1776_177623


namespace cricketer_wickets_l1776_177693

noncomputable def initial_average (R W : ℝ) : ℝ := R / W

noncomputable def new_average (R W : ℝ) (additional_runs additional_wickets : ℝ) : ℝ :=
  (R + additional_runs) / (W + additional_wickets)

theorem cricketer_wickets (R W : ℝ) 
(h1 : initial_average R W = 12.4) 
(h2 : new_average R W 26 5 = 12.0) : 
  W = 85 :=
sorry

end cricketer_wickets_l1776_177693


namespace three_squares_sum_l1776_177631

theorem three_squares_sum (n : ℤ) (h : n > 5) : 
  3 * (n - 1)^2 + 32 = (n - 5)^2 + (n - 1)^2 + (n + 3)^2 :=
by sorry

end three_squares_sum_l1776_177631
