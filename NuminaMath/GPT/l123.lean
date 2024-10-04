import Mathlib

namespace pizza_fraction_after_six_trips_l123_123923

noncomputable def total_pizza_eaten (n : ℕ) : ℚ :=
if n = 0 then 0 
else (1 / 3) + (1 / 3 * ((1 / 2) ^ n - 1))*(1 / 2)

theorem pizza_fraction_after_six_trips :
  total_pizza_eaten 6 = 21 / 32 :=
by sorry

end pizza_fraction_after_six_trips_l123_123923


namespace value_of_a_minus_b_l123_123191

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 8) (h2 : |b| = 6) (h3 : |a + b| = a + b) : a - b = 2 ∨ a - b = 14 := 
sorry

end value_of_a_minus_b_l123_123191


namespace sum_50th_set_l123_123171

-- Definition of the sequence repeating pattern
def repeating_sequence : List (List Nat) :=
  [[1], [2, 2], [3, 3, 3], [4, 4, 4, 4]]

-- Definition to get the nth set in the repeating sequence
def nth_set (n : Nat) : List Nat :=
  repeating_sequence.get! ((n - 1) % 4)

-- Definition to sum the elements of a list
def sum_list (l : List Nat) : Nat :=
  l.sum

-- Proposition to prove that the sum of the 50th set is 4
theorem sum_50th_set : sum_list (nth_set 50) = 4 :=
by
  sorry

end sum_50th_set_l123_123171


namespace find_v₃_value_l123_123801

def f (x : ℕ) : ℕ := 7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

def v₃_expr (x : ℕ) : ℕ := (((7 * x + 6) * x + 5) * x + 4)

theorem find_v₃_value : v₃_expr 3 = 262 := by
  sorry

end find_v₃_value_l123_123801


namespace find_A_l123_123065

theorem find_A (A : ℤ) (h : A + 10 = 15) : A = 5 :=
sorry

end find_A_l123_123065


namespace karen_nuts_l123_123880

/-- Karen added 0.25 cup of walnuts to a batch of trail mix.
Later, she added 0.25 cup of almonds.
In all, Karen put 0.5 cups of nuts in the trail mix. -/
theorem karen_nuts (walnuts almonds : ℝ) 
  (h_walnuts : walnuts = 0.25) 
  (h_almonds : almonds = 0.25) : 
  walnuts + almonds = 0.5 := 
by
  sorry

end karen_nuts_l123_123880


namespace oil_in_Tank_C_is_982_l123_123137

-- Definitions of tank capacities and oil amounts
def capacity_A := 80
def capacity_B := 120
def capacity_C := 160
def capacity_D := 240

def total_oil_bought := 1387

def oil_in_A := 70
def oil_in_B := 95
def oil_in_D := capacity_D  -- Since Tank D is 100% full

-- Statement of the problem
theorem oil_in_Tank_C_is_982 :
  oil_in_A + oil_in_B + oil_in_D + (total_oil_bought - (oil_in_A + oil_in_B + oil_in_D)) = total_oil_bought :=
by
  sorry

end oil_in_Tank_C_is_982_l123_123137


namespace mean_of_two_numbers_l123_123100

theorem mean_of_two_numbers (a b : ℝ) (mean_twelve : ℝ) (mean_fourteen : ℝ) 
  (h1 : mean_twelve = 60) 
  (h2 : mean_fourteen = 75) 
  (sum_twelve : 12 * mean_twelve = 720) 
  (sum_fourteen : 14 * mean_fourteen = 1050) 
  : (a + b) / 2 = 165 :=
by
  sorry

end mean_of_two_numbers_l123_123100


namespace fractional_equation_m_value_l123_123063

theorem fractional_equation_m_value {x m : ℝ} (hx : 0 < x) (h : 3 / (x - 4) = 1 - (x + m) / (4 - x))
: m = -1 := sorry

end fractional_equation_m_value_l123_123063


namespace cos_240_degree_l123_123296

theorem cos_240_degree : Real.cos (240 * Real.pi / 180) = -1/2 :=
by
  sorry

end cos_240_degree_l123_123296


namespace cos_240_eq_neg_half_l123_123303

theorem cos_240_eq_neg_half : Real.cos (4 * Real.pi / 3) = -1/2 :=
by
  sorry

end cos_240_eq_neg_half_l123_123303


namespace count_prime_two_digit_sum_ten_is_three_l123_123542

-- Define a helper function to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to compute the sum of the digits of a two-digit number
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def sum_of_digits_is_ten (n : ℕ) : Prop := digit_sum n = 10

-- Prove the number of prime two-digit numbers whose digits sum to 10 is 3
theorem count_prime_two_digit_sum_ten_is_three :
  finset.filter (λ n, is_prime n ∧ is_two_digit n ∧ sum_of_digits_is_ten n) (finset.range 100)).card = 3 :=
by
  sorry

end count_prime_two_digit_sum_ten_is_three_l123_123542


namespace maggie_remaining_goldfish_l123_123888

theorem maggie_remaining_goldfish
  (total_goldfish : ℕ)
  (allowed_fraction : ℕ → ℕ)
  (caught_fraction : ℕ → ℕ)
  (halfsies : ℕ)
  (remaining_goldfish : ℕ)
  (h1 : total_goldfish = 100)
  (h2 : allowed_fraction total_goldfish = total_goldfish / 2)
  (h3 : caught_fraction (allowed_fraction total_goldfish) = (3 * allowed_fraction total_goldfish) / 5)
  (h4 : halfsies = allowed_fraction total_goldfish)
  (h5 : remaining_goldfish = halfsies - caught_fraction halfsies) :
  remaining_goldfish = 20 :=
sorry

end maggie_remaining_goldfish_l123_123888


namespace abc_equal_l123_123228

theorem abc_equal (a b c : ℝ) (h : a^2 + b^2 + c^2 - ab - bc - ac = 0) : a = b ∧ b = c :=
by
  sorry

end abc_equal_l123_123228


namespace rational_combination_zero_eqn_l123_123991

theorem rational_combination_zero_eqn (a b c : ℚ) (h : a + b * Real.sqrt 32 + c * Real.sqrt 34 = 0) :
  a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end rational_combination_zero_eqn_l123_123991


namespace gcd_poly_l123_123983

theorem gcd_poly (k : ℕ) : Nat.gcd ((4500 * k)^2 + 11 * (4500 * k) + 40) (4500 * k + 8) = 3 := by
  sorry

end gcd_poly_l123_123983


namespace correct_statement_is_C_l123_123478

-- Defining conditions
def statementA : Prop := "waiting_by_the_stump_for_a_hare_to_come_is_certain"
def statementB : Prop := "probability_of_0.0001_is_impossible"
def statementC : Prop := "drawing_red_ball_from_bag_with_only_5_red_balls_is_certain"
def statementD : Prop := "flipping_fair_coin_20_times_heads_up_must_be_10_times"

-- Theorem stating that statement C is the only correct one
theorem correct_statement_is_C :
  ¬statementA ∧ ¬statementB ∧ statementC ∧ ¬statementD :=
by
  sorry

end correct_statement_is_C_l123_123478


namespace graph_of_abs_g_l123_123906

noncomputable def g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -1 then x + 3
  else if -1 < x ∧ x ≤ 1 then -x^2 + 2
  else if 1 < x ∧ x ≤ 4 then x - 2
  else 0

noncomputable def abs_g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -3 then -(x + 3)
  else if -3 < x ∧ x ≤ -1 then x + 3
  else if -1 < x ∧ x ≤ 1 then -x^2 + 2
  else if 1 < x ∧ x ≤ 2 then -(x - 2)
  else if 2 < x ∧ x ≤ 4 then x - 2
  else 0

theorem graph_of_abs_g :
  ∀ x : ℝ, abs_g x = |g x| :=
by
  sorry

end graph_of_abs_g_l123_123906


namespace sum_of_consecutive_integers_l123_123793

theorem sum_of_consecutive_integers (x y : ℕ) (h1 : y = x + 1) (h2 : x * y = 812) : x + y = 57 :=
by
  -- proof skipped
  sorry

end sum_of_consecutive_integers_l123_123793


namespace number_of_diet_soda_bottles_l123_123687

theorem number_of_diet_soda_bottles (apples regular_soda total_bottles diet_soda : ℕ)
    (h_apples : apples = 36)
    (h_regular_soda : regular_soda = 80)
    (h_total_bottles : total_bottles = apples + 98)
    (h_diet_soda_eq : total_bottles = regular_soda + diet_soda) :
    diet_soda = 54 := by
  sorry

end number_of_diet_soda_bottles_l123_123687


namespace two_digit_primes_with_digit_sum_10_count_l123_123557

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end two_digit_primes_with_digit_sum_10_count_l123_123557


namespace matt_and_peter_worked_together_days_l123_123425

variables (W : ℝ) -- Represents total work
noncomputable def work_rate_peter := W / 35
noncomputable def work_rate_together := W / 20

theorem matt_and_peter_worked_together_days (x : ℝ) :
  (x / 20) + (14 / 35) = 1 → x = 12 :=
by {
  sorry
}

end matt_and_peter_worked_together_days_l123_123425


namespace number_of_valid_pairs_l123_123011

theorem number_of_valid_pairs (a b : ℕ) (cond1 : b > a) (cond2 : ∃ a b, b > a ∧ (2*a*b = 3*(a-4)*(b-4))) : 
    ∃ (pairs : set (ℕ × ℕ)), pairs = {(13, 108), (14, 60), (15, 44)} ∧ pairs.size = 3 :=
begin
  sorry
end

end number_of_valid_pairs_l123_123011


namespace first_negative_term_position_l123_123761

def a1 : ℤ := 1031
def d : ℤ := -3
def nth_term (n : ℕ) : ℤ := a1 + (n - 1 : ℤ) * d

theorem first_negative_term_position : ∃ n : ℕ, nth_term n < 0 ∧ n = 345 := 
by 
  -- Placeholder for proof
  sorry

end first_negative_term_position_l123_123761


namespace price_arun_paid_l123_123270

theorem price_arun_paid 
  (original_price : ℝ)
  (standard_concession_rate : ℝ) 
  (additional_concession_rate : ℝ)
  (reduced_price : ℝ)
  (final_price : ℝ) 
  (h1 : original_price = 2000)
  (h2 : standard_concession_rate = 0.30)
  (h3 : additional_concession_rate = 0.20)
  (h4 : reduced_price = original_price * (1 - standard_concession_rate))
  (h5 : final_price = reduced_price * (1 - additional_concession_rate)) :
  final_price = 1120 :=
by
  sorry

end price_arun_paid_l123_123270


namespace cone_lateral_surface_area_ratio_l123_123585

theorem cone_lateral_surface_area_ratio (r l S_lateral S_base : ℝ) (h1 : l = 3 * r)
  (h2 : S_lateral = π * r * l) (h3 : S_base = π * r^2) :
  S_lateral / S_base = 3 :=
by
  sorry

end cone_lateral_surface_area_ratio_l123_123585


namespace remainder_when_divided_by_x_minus_2_l123_123116

-- Define the polynomial function f(x)
def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 20*x^3 + x^2 - 47*x + 15

-- State the theorem to be proved with the given conditions
theorem remainder_when_divided_by_x_minus_2 :
  f 2 = -11 :=
by 
  -- Proof goes here
  sorry

end remainder_when_divided_by_x_minus_2_l123_123116


namespace triplet_sum_not_equal_two_l123_123252

theorem triplet_sum_not_equal_two :
  ¬((1.2 + -2.2 + 2) = 2) ∧ ¬((- 4 / 3 + - 2 / 3 + 3) = 2) :=
by
  sorry

end triplet_sum_not_equal_two_l123_123252


namespace number_of_ordered_pairs_is_three_l123_123010

-- Define the problem parameters and conditions
variables (a b : ℕ)
variable (b_gt_a : b > a)

-- Define the equation for the areas based on the problem conditions
def area_equation : Prop :=
  a * b = 3 * (a - 4) * (b - 4)

-- Main theorem statement
theorem number_of_ordered_pairs_is_three (h₁ : a > 0) (h₂ : b > 0) (h3: b_gt_a) (h4: area_equation a b) :
  ∃! (n : ℕ), n = 3 :=
begin
  sorry  -- Proof is omitted
end

end number_of_ordered_pairs_is_three_l123_123010


namespace simplify_expression_l123_123620

variable (x : ℝ)

theorem simplify_expression : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 :=
by
  sorry

end simplify_expression_l123_123620


namespace angle_C_side_c_area_of_triangle_l123_123874

open Real

variables (A B C a b c : Real)

noncomputable def acute_triangle (A B C a b c : ℝ) : Prop :=
  (A + B + C = π) ∧ (A > 0) ∧ (B > 0) ∧ (C > 0) ∧
  (A < π / 2) ∧ (B < π / 2) ∧ (C < π / 2) ∧
  (a^2 - 2 * sqrt 3 * a + 2 = 0) ∧
  (b^2 - 2 * sqrt 3 * b + 2 = 0) ∧
  (2 * sin (A + B) - sqrt 3 = 0)

noncomputable def length_side_c (a b : ℝ) : ℝ :=
  sqrt (a^2 + b^2 - 2 * a * b * cos (π / 3))

noncomputable def area_triangle (a b : ℝ) : ℝ := 
  (1 / 2) * a * b * sin (π / 3)

theorem angle_C (h : acute_triangle A B C a b c) : C = π / 3 :=
  sorry

theorem side_c (h : acute_triangle A B C a b c) : c = sqrt 6 :=
  sorry

theorem area_of_triangle (h : acute_triangle A B C a b c) : area_triangle a b = sqrt 3 / 2 :=
  sorry

end angle_C_side_c_area_of_triangle_l123_123874


namespace maximum_m_l123_123087

theorem maximum_m (a b c : ℝ)
  (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : a + b + c = 10)
  (h₅ : a * b + b * c + c * a = 25) :
  ∃ m, (m = min (a * b) (min (b * c) (c * a)) ∧ m = 25 / 9) :=
sorry

end maximum_m_l123_123087


namespace problem_q_value_l123_123572

theorem problem_q_value (p q : ℝ) (hpq : 1 < p ∧ p < q) 
  (h1 : 1 / p + 1 / q = 1) 
  (h2 : p * q = 8) : 
  q = 4 + 2 * real.sqrt 2 :=
by sorry

end problem_q_value_l123_123572


namespace cost_of_toilet_paper_roll_l123_123698

-- Definitions of the problem's conditions
def num_toilet_paper_rolls : Nat := 10
def num_paper_towel_rolls : Nat := 7
def num_tissue_boxes : Nat := 3

def cost_per_paper_towel : Real := 2
def cost_per_tissue_box : Real := 2

def total_cost : Real := 35

-- The function to prove
def cost_per_toilet_paper_roll (x : Real) :=
  num_toilet_paper_rolls * x + 
  num_paper_towel_rolls * cost_per_paper_towel + 
  num_tissue_boxes * cost_per_tissue_box = total_cost

-- Statement to prove
theorem cost_of_toilet_paper_roll : 
  cost_per_toilet_paper_roll 1.5 := 
by
  simp [num_toilet_paper_rolls, num_paper_towel_rolls, num_tissue_boxes, cost_per_paper_towel, cost_per_tissue_box, total_cost]
  sorry

end cost_of_toilet_paper_roll_l123_123698


namespace necessary_but_not_sufficient_condition_l123_123486

theorem necessary_but_not_sufficient_condition (x : ℝ) : (x > 2 → x > 1) ∧ ¬(x > 1 → x > 2) :=
by
  sorry

end necessary_but_not_sufficient_condition_l123_123486


namespace no_real_roots_of_ffx_eq_ninex_l123_123091

variable (a : ℝ)
noncomputable def f (x : ℝ) : ℝ :=
  x^2 * Real.log (4*(a+1)/a) / Real.log 2 +
  2 * x * Real.log (2 * a / (a + 1)) / Real.log 2 +
  Real.log ((a + 1)^2 / (4 * a^2)) / Real.log 2

theorem no_real_roots_of_ffx_eq_ninex (a : ℝ) (h_pos : ∀ x, 1 ≤ x → f a x > 0) :
  ¬ ∃ x, 1 ≤ x ∧ f a (f a x) = 9 * x :=
  sorry

end no_real_roots_of_ffx_eq_ninex_l123_123091


namespace division_proof_l123_123231

-- Defining the given conditions
def total_books := 1200
def first_div := 3
def second_div := 4
def final_books_per_category := 15

-- Calculating the number of books per each category after each division
def books_per_first_category := total_books / first_div
def books_per_second_group := books_per_first_category / second_div

-- Correcting the third division to ensure each part has 15 books
def third_div := books_per_second_group / final_books_per_category
def rounded_parts := (books_per_second_group : ℕ) / final_books_per_category -- Rounded to the nearest integer

-- The number of final parts must be correct to ensure the total final categories
def final_division := first_div * second_div * rounded_parts

-- Required proof statement
theorem division_proof : final_division = 84 ∧ books_per_second_group = final_books_per_category :=
by 
  sorry

end division_proof_l123_123231


namespace option_D_is_divisible_by_9_l123_123276

theorem option_D_is_divisible_by_9 (k : ℕ) (hk : k > 0) : 9 ∣ 3 * (2 + 7^k) := 
sorry

end option_D_is_divisible_by_9_l123_123276


namespace zero_a_and_b_l123_123934

theorem zero_a_and_b (a b : ℝ) (h : a^2 + |b| = 0) : a = 0 ∧ b = 0 :=
by
  sorry

end zero_a_and_b_l123_123934


namespace negation_of_exists_abs_lt_one_l123_123105

theorem negation_of_exists_abs_lt_one :
  (¬ ∃ x : ℝ, |x| < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) :=
by
  sorry

end negation_of_exists_abs_lt_one_l123_123105


namespace drum_filled_capacity_l123_123838

theorem drum_filled_capacity (C : ℝ) (h1 : 0 < C) :
    (4 / 5) * C + (1 / 2) * C = (13 / 10) * C :=
by
  sorry

end drum_filled_capacity_l123_123838


namespace trajectory_midpoints_parabola_l123_123444

theorem trajectory_midpoints_parabola {k : ℝ} (hk : k ≠ 0) :
  ∀ (x1 x2 y1 y2 : ℝ), 
    y1 = 2 * x1^2 → 
    y2 = 2 * x2^2 → 
    y2 - y1 = 2 * (x2 + x1) * (x2 - x1) → 
    x = (x1 + x2) / 2 → 
    k = (y2 - y1) / (x2 - x1) → 
    x = 1 / (4 * k) := 
sorry

end trajectory_midpoints_parabola_l123_123444


namespace train_length_l123_123504

theorem train_length :
  ∀ (t : ℝ) (v_man : ℝ) (v_train : ℝ),
  t = 41.9966402687785 →
  v_man = 3 →
  v_train = 63 →
  (v_train - v_man) * (5 / 18) * t = 699.94400447975 :=
by
  intros t v_man v_train ht hv_man hv_train
  -- Use the given conditions as definitions
  rw [ht, hv_man, hv_train]
  sorry

end train_length_l123_123504


namespace total_area_is_82_l123_123077

/-- Definition of the lengths of each segment as conditions -/
def length1 : ℤ := 7
def length2 : ℤ := 4
def length3 : ℤ := 5
def length4 : ℤ := 3
def length5 : ℤ := 2
def length6 : ℤ := 1

/-- Rectangle areas based on the given lengths -/
def area_A : ℤ := length1 * length2 -- 7 * 4
def area_B : ℤ := length3 * length2 -- 5 * 4
def area_C : ℤ := length1 * length4 -- 7 * 3
def area_D : ℤ := length3 * length5 -- 5 * 2
def area_E : ℤ := length4 * length6 -- 3 * 1

/-- The total area is the sum of all rectangle areas -/
def total_area : ℤ := area_A + area_B + area_C + area_D + area_E

/-- Theorem: The total area is 82 square units -/
theorem total_area_is_82 : total_area = 82 :=
by
  -- Proof left as an exercise
  sorry

end total_area_is_82_l123_123077


namespace min_disks_required_for_files_l123_123138

theorem min_disks_required_for_files :
  ∀ (number_of_files : ℕ)
    (files_0_9MB : ℕ)
    (files_0_6MB : ℕ)
    (disk_capacity_MB : ℝ)
    (file_size_0_9MB : ℝ)
    (file_size_0_6MB : ℝ)
    (file_size_0_45MB : ℝ),
  number_of_files = 40 →
  files_0_9MB = 5 →
  files_0_6MB = 15 →
  disk_capacity_MB = 1.44 →
  file_size_0_9MB = 0.9 →
  file_size_0_6MB = 0.6 →
  file_size_0_45MB = 0.45 →
  ∃ (min_disks : ℕ), min_disks = 16 :=
by
  sorry

end min_disks_required_for_files_l123_123138


namespace circle_passes_through_fixed_point_l123_123842

theorem circle_passes_through_fixed_point (a : ℝ) (ha : a ≠ 1) : 
  ∃ P : ℝ × ℝ, P = (1, 1) ∧ ∀ (x y : ℝ), (x^2 + y^2 - 2*a*x + 2*(a-2)*y + 2 = 0) → (x, y) = P :=
sorry

end circle_passes_through_fixed_point_l123_123842


namespace cos_240_eq_neg_half_l123_123290

theorem cos_240_eq_neg_half (h1 : Real.cos (180 * Real.pi / 180) = -1)
                            (h2 : Real.sin (180 * Real.pi / 180) = 0)
                            (h3 : Real.cos (60 * Real.pi / 180) = 1 / 2) :
  Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_240_eq_neg_half_l123_123290


namespace coordinates_of_point_P_l123_123195

open Real

def in_fourth_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 < 0

def distance_to_x_axis (P : ℝ × ℝ) : ℝ :=
  abs P.2

def distance_to_y_axis (P : ℝ × ℝ) : ℝ :=
  abs P.1

theorem coordinates_of_point_P (P : ℝ × ℝ) 
  (h1 : in_fourth_quadrant P) 
  (h2 : distance_to_x_axis P = 1) 
  (h3 : distance_to_y_axis P = 2) : 
  P = (2, -1) :=
by
  sorry

end coordinates_of_point_P_l123_123195


namespace sequence_a_n_l123_123210

noncomputable def a_n (n : ℕ) : ℚ :=
if n = 1 then 1 else (1 : ℚ) / (2 * n - 1)

theorem sequence_a_n (n : ℕ) (hn : n ≥ 1) : 
  (a_n 1 = 1) ∧ 
  (∀ n, a_n n ≠ 0) ∧ 
  (∀ n, n ≥ 2 → a_n n + 2 * a_n n * a_n (n - 1) - a_n (n - 1) = 0) →
  a_n n = 1 / (2 * n - 1) :=
by
  sorry

end sequence_a_n_l123_123210


namespace problem_statement_l123_123102

noncomputable def f : ℝ → ℝ := sorry

axiom func_condition : ∀ a b : ℝ, b^2 * f a = a^2 * f b
axiom f2_nonzero : f 2 ≠ 0

theorem problem_statement : (f 6 - f 3) / f 2 = 27 / 4 := 
by 
  sorry

end problem_statement_l123_123102


namespace max_three_kopecks_l123_123112

def is_coin_placement_correct (n1 n2 n3 : ℕ) : Prop :=
  -- Conditions for the placement to be valid
  ∀ (i j : ℕ), i < j → 
  ((j - i > 1 → n1 = 0) ∧ (j - i > 2 → n2 = 0) ∧ (j - i > 3 → n3 = 0))

theorem max_three_kopecks (n1 n2 n3 : ℕ) (h : n1 + n2 + n3 = 101) (placement_correct : is_coin_placement_correct n1 n2 n3) :
  n3 = 25 ∨ n3 = 26 :=
sorry

end max_three_kopecks_l123_123112


namespace probability_of_red_jelly_bean_l123_123003

-- Definitions based on conditions
def total_jelly_beans := 7 + 9 + 4 + 10
def red_jelly_beans := 7

-- Statement we want to prove
theorem probability_of_red_jelly_bean : (red_jelly_beans : ℚ) / total_jelly_beans = 7 / 30 :=
by
  -- Proof here
  sorry

end probability_of_red_jelly_bean_l123_123003


namespace tangent_line_at_point_is_x_minus_y_plus_1_eq_0_l123_123722

noncomputable def tangent_line (x : ℝ) : ℝ := x * Real.exp x + 1

theorem tangent_line_at_point_is_x_minus_y_plus_1_eq_0:
  tangent_line 0 = 1 →
  ∀ x y, y = tangent_line x → x - y + 1 = 0 → y = x * Real.exp x + 1 →
  x = 0 ∧ y = 1 → x - y + 1 = 0 :=
by
  intro h_point x y h_tangent h_eq h_coord
  sorry

end tangent_line_at_point_is_x_minus_y_plus_1_eq_0_l123_123722


namespace average_age_of_boys_l123_123623

def boys_age_proportions := (3, 5, 7)
def eldest_boy_age := 21

theorem average_age_of_boys : 
  ∃ (x : ℕ), 7 * x = eldest_boy_age ∧ (3 * x + 5 * x + 7 * x) / 3 = 15 :=
by
  sorry

end average_age_of_boys_l123_123623


namespace sum_of_two_numbers_l123_123451

theorem sum_of_two_numbers :
  (∃ x y : ℕ, y = 2 * x - 43 ∧ y = 31 ∧ x + y = 68) :=
sorry

end sum_of_two_numbers_l123_123451


namespace cos_240_eq_neg_half_l123_123289

theorem cos_240_eq_neg_half (h1 : Real.cos (180 * Real.pi / 180) = -1)
                            (h2 : Real.sin (180 * Real.pi / 180) = 0)
                            (h3 : Real.cos (60 * Real.pi / 180) = 1 / 2) :
  Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_240_eq_neg_half_l123_123289


namespace find_number_l123_123826

variable (x : ℝ)

theorem find_number 
  (h1 : 0.20 * x + 0.25 * 60 = 23) :
  x = 40 :=
sorry

end find_number_l123_123826


namespace find_min_positive_n_l123_123411

-- Assume the sequence {a_n} is given
variables {a : ℕ → ℤ}

-- Given conditions
-- a4 < 0 and a5 > |a4|
def condition1 (a : ℕ → ℤ) : Prop := a 4 < 0
def condition2 (a : ℕ → ℤ) : Prop := a 5 > abs (a 4)

-- Sum of the first n terms of the arithmetic sequence
def S (n : ℕ) (a : ℕ → ℤ) : ℤ := n * (a 1 + a n) / 2

-- The main theorem we need to prove
theorem find_min_positive_n (a : ℕ → ℤ) (h1 : condition1 a) (h2 : condition2 a) : ∃ n : ℕ, n = 8 ∧ S n a > 0 :=
by
  sorry

end find_min_positive_n_l123_123411


namespace area_of_circumcircle_l123_123404

-- Define the problem:
theorem area_of_circumcircle 
  (a b c : ℝ) 
  (A B C : Real) 
  (h_cosC : Real.cos C = (2 * Real.sqrt 2) / 3) 
  (h_bcosA_acoB : b * Real.cos A + a * Real.cos B = 2)
  (h_sides : c = 2):
  let sinC := Real.sqrt (1 - (2 * Real.sqrt 2 / 3)^2)
  let R := c / (2 * sinC)
  let area := Real.pi * R^2
  area = 9 * Real.pi / 5 :=
by 
  sorry

end area_of_circumcircle_l123_123404


namespace minimum_value_f_l123_123731

noncomputable def f (x y : ℝ) : ℝ :=
  (x^2 / (y - 2)^2) + (y^2 / (x - 2)^2)

theorem minimum_value_f (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  ∃ (a : ℝ), (∀ (b : ℝ), f x y >= b) ∧ a = 10 := sorry

end minimum_value_f_l123_123731


namespace percentage_of_local_arts_students_is_50_l123_123753

-- Definitions
def total_students_arts := 400
def total_students_science := 100
def total_students_commerce := 120
def percent_local_science := 25 / 100
def percent_local_commerce := 85 / 100
def total_locals := 327

-- Problem statement in Lean
theorem percentage_of_local_arts_students_is_50
  (x : ℕ) -- Percentage of local arts students as a natural number
  (h1 : percent_local_science * total_students_science = 25)
  (h2 : percent_local_commerce * total_students_commerce = 102)
  (h3 : (x / 100 : ℝ) * total_students_arts + 25 + 102 = total_locals) :
  x = 50 :=
sorry

end percentage_of_local_arts_students_is_50_l123_123753


namespace cos_240_eq_neg_half_l123_123355

open Real

theorem cos_240_eq_neg_half : cos (240 * π / 180) = -1/2 :=
by
  -- Step 1: Decompose the angle 240° = 180° + 60°
  have h1 : 240 * π / 180 = π + 60 * π / 180,
  { 
    norm_num, 
    field_simp, 
    linarith 
  },
  -- Step 2: Use the fact that the cosine of (π + θ) = - cos(θ)
  rw [h1, cos_add_pi],
  -- Step 3: Given that cos(60°) = 1/2
  have h2 : cos (60 * π / 180) = 1/2,
  {
    norm_num,
    exact Real.cos_pi_div_three (),
  },
  -- Conclude that cos(240°) = -1/2
  rw h2,
  norm_num

end cos_240_eq_neg_half_l123_123355


namespace derivative_y_l123_123932

noncomputable def u (x : ℝ) := 4 * x - 1 + Real.sqrt (16 * x ^ 2 - 8 * x + 2)
noncomputable def v (x : ℝ) := Real.sqrt (16 * x ^ 2 - 8 * x + 2) * Real.arctan (4 * x - 1)

noncomputable def y (x : ℝ) := Real.log (u x) - v x

theorem derivative_y (x : ℝ) :
  deriv y x = (4 * (1 - 4 * x)) / (Real.sqrt (16 * x ^ 2 - 8 * x + 2)) * Real.arctan (4 * x - 1) :=
by
  sorry

end derivative_y_l123_123932


namespace fraction_subtraction_simplified_l123_123970

theorem fraction_subtraction_simplified : (8 / 19 - 5 / 57) = (1 / 3) := by
  sorry

end fraction_subtraction_simplified_l123_123970


namespace car_speed_first_hour_l123_123913

-- Definitions based on the conditions in the problem
noncomputable def speed_second_hour := 30
noncomputable def average_speed := 45
noncomputable def total_time := 2

-- Assertion based on the problem's question and correct answer
theorem car_speed_first_hour: ∃ (x : ℕ), (average_speed * total_time) = (x + speed_second_hour) ∧ x = 60 :=
by
  sorry

end car_speed_first_hour_l123_123913


namespace recliner_price_drop_l123_123263

theorem recliner_price_drop
  (P : ℝ) (N : ℝ)
  (N' : ℝ := 1.8 * N)
  (G : ℝ := P * N)
  (G' : ℝ := 1.44 * G) :
  (P' : ℝ) → P' = 0.8 * P → (P - P') / P * 100 = 20 :=
by
  intros
  sorry

end recliner_price_drop_l123_123263


namespace percentage_income_spent_on_clothes_l123_123697

-- Define the assumptions
def monthly_income : ℝ := 90000
def household_expenses : ℝ := 0.5 * monthly_income
def medicine_expenses : ℝ := 0.15 * monthly_income
def savings : ℝ := 9000

-- Define the proof statement
theorem percentage_income_spent_on_clothes :
  ∃ (clothes_expenses : ℝ),
    clothes_expenses = monthly_income - household_expenses - medicine_expenses - savings ∧
    (clothes_expenses / monthly_income) * 100 = 25 := 
sorry

end percentage_income_spent_on_clothes_l123_123697


namespace num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l123_123549

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  d1 + d2 = s

def two_digit := ∀ n : ℕ, 10 ≤ n ∧ n < 100

theorem num_of_two_digit_prime_numbers_sum_of_digits_eq_10 : 
  (finset.filter (λ n, is_prime n ∧ digits_sum_to n 10 ∧ two_digit n) (finset.range 100)).card = 3 := by 
  sorry

end num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l123_123549


namespace main_theorem_l123_123417

-- Let x be a real number
variable {x : ℝ}

-- Define the given identity
def identity (M₁ M₂ : ℝ) : Prop :=
  ∀ x, (50 * x - 42) / (x^2 - 5 * x + 6) = M₁ / (x - 2) + M₂ / (x - 3)

-- The proposition to prove the numerical value of M₁M₂
def prove_M1M2_value : Prop :=
  ∀ (M₁ M₂ : ℝ), identity M₁ M₂ → M₁ * M₂ = -6264

theorem main_theorem : prove_M1M2_value :=
  sorry

end main_theorem_l123_123417


namespace goldfish_remaining_to_catch_l123_123891

-- Define the number of total goldfish in the aquarium
def total_goldfish : ℕ := 100

-- Define the number of goldfish Maggie is allowed to take home (half of total goldfish)
def allowed_to_take_home := total_goldfish / 2

-- Define the number of goldfish Maggie caught (3/5 of allowed_to_take_home)
def caught := (3 * allowed_to_take_home) / 5

-- Prove the number of goldfish Maggie remains with to catch
theorem goldfish_remaining_to_catch : allowed_to_take_home - caught = 20 := by
  -- Sorry is used to skip the proof
  sorry

end goldfish_remaining_to_catch_l123_123891


namespace area_transformation_l123_123957

variables {g : ℝ → ℝ}

theorem area_transformation (h : ∫ x in a..b, g x = 12) :
  ∫ x in c..d, 4 * g (2 * x + 3) = 48 :=
by
  sorry

end area_transformation_l123_123957


namespace cos_240_eq_neg_half_l123_123349

open Real

theorem cos_240_eq_neg_half : cos (240 * π / 180) = -1/2 :=
by
  -- Step 1: Decompose the angle 240° = 180° + 60°
  have h1 : 240 * π / 180 = π + 60 * π / 180,
  { 
    norm_num, 
    field_simp, 
    linarith 
  },
  -- Step 2: Use the fact that the cosine of (π + θ) = - cos(θ)
  rw [h1, cos_add_pi],
  -- Step 3: Given that cos(60°) = 1/2
  have h2 : cos (60 * π / 180) = 1/2,
  {
    norm_num,
    exact Real.cos_pi_div_three (),
  },
  -- Conclude that cos(240°) = -1/2
  rw h2,
  norm_num

end cos_240_eq_neg_half_l123_123349


namespace defective_percentage_is_0_05_l123_123507

-- Define the problem conditions as Lean definitions
def total_meters : ℕ := 4000
def defective_meters : ℕ := 2

-- Define the percentage calculation function
def percentage_defective (defective total : ℕ) : ℚ :=
  (defective : ℚ) / (total : ℚ) * 100

-- Rewrite the proof statement using these definitions
theorem defective_percentage_is_0_05 :
  percentage_defective defective_meters total_meters = 0.05 :=
by
  sorry

end defective_percentage_is_0_05_l123_123507


namespace cos_240_eq_neg_half_l123_123354

open Real

theorem cos_240_eq_neg_half : cos (240 * π / 180) = -1/2 :=
by
  -- Step 1: Decompose the angle 240° = 180° + 60°
  have h1 : 240 * π / 180 = π + 60 * π / 180,
  { 
    norm_num, 
    field_simp, 
    linarith 
  },
  -- Step 2: Use the fact that the cosine of (π + θ) = - cos(θ)
  rw [h1, cos_add_pi],
  -- Step 3: Given that cos(60°) = 1/2
  have h2 : cos (60 * π / 180) = 1/2,
  {
    norm_num,
    exact Real.cos_pi_div_three (),
  },
  -- Conclude that cos(240°) = -1/2
  rw h2,
  norm_num

end cos_240_eq_neg_half_l123_123354


namespace geometric_sequence_a2_a6_l123_123403

theorem geometric_sequence_a2_a6 (a : ℕ → ℝ) (r : ℝ) (h : ∀ n, a (n + 1) = r * a n) (h₄ : a 4 = 4) :
  a 2 * a 6 = 16 :=
sorry

end geometric_sequence_a2_a6_l123_123403


namespace profit_percentage_correct_l123_123147

-- Statement of the problem in Lean
theorem profit_percentage_correct (SP CP : ℝ) (hSP : SP = 400) (hCP : CP = 320) : 
  ((SP - CP) / CP) * 100 = 25 := by
  -- Proof goes here
  sorry

end profit_percentage_correct_l123_123147


namespace two_digit_prime_sum_to_ten_count_l123_123538

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Main statement to prove
theorem two_digit_prime_sum_to_ten_count :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
sorry

end two_digit_prime_sum_to_ten_count_l123_123538


namespace fraction_addition_l123_123805

theorem fraction_addition :
  (1 / 6) + (1 / 3) + (5 / 9) = 19 / 18 :=
by
  sorry

end fraction_addition_l123_123805


namespace distinct_arrangements_l123_123057

-- Define the conditions: 7 books, 3 are identical
def total_books : ℕ := 7
def identical_books : ℕ := 3

-- Statement that the number of distinct arrangements is 840
theorem distinct_arrangements : (Nat.factorial total_books) / (Nat.factorial identical_books) = 840 := 
by
  sorry

end distinct_arrangements_l123_123057


namespace solution_set_inequality_l123_123450

theorem solution_set_inequality (x : ℝ) : ((x - 1) * (x + 2) < 0) ↔ (-2 < x ∧ x < 1) := by
  sorry

end solution_set_inequality_l123_123450


namespace sufficient_not_necessary_l123_123818

-- Definitions based on the conditions
def f1 (x y : ℝ) : Prop := x^2 + y^2 = 0
def f2 (x y : ℝ) : Prop := x * y = 0

-- The theorem we need to prove
theorem sufficient_not_necessary (x y : ℝ) : f1 x y → f2 x y ∧ ¬ (f2 x y → f1 x y) := 
by sorry

end sufficient_not_necessary_l123_123818


namespace total_cost_of_square_park_l123_123502

-- Define the cost per side and number of sides
def cost_per_side : ℕ := 56
def sides_of_square : ℕ := 4

-- The total cost of fencing the park
def total_cost_of_fencing (cost_per_side : ℕ) (sides_of_square : ℕ) : ℕ := cost_per_side * sides_of_square

-- The statement we need to prove
theorem total_cost_of_square_park : total_cost_of_fencing cost_per_side sides_of_square = 224 :=
by sorry

end total_cost_of_square_park_l123_123502


namespace cos_240_eq_neg_half_l123_123301

theorem cos_240_eq_neg_half : Real.cos (4 * Real.pi / 3) = -1/2 :=
by
  sorry

end cos_240_eq_neg_half_l123_123301


namespace complete_square_l123_123118

theorem complete_square (x : ℝ) : (x^2 - 4*x + 2 = 0) → ((x - 2)^2 = 2) :=
by
  intro h
  sorry

end complete_square_l123_123118


namespace coffee_prices_purchase_ways_l123_123930

-- Define the cost equations for coffee A and B
def cost_equation1 (x y : ℕ) : Prop := 10 * x + 15 * y = 230
def cost_equation2 (x y : ℕ) : Prop := 25 * x + 25 * y = 450

-- Define what we need to prove for task 1
theorem coffee_prices (x y : ℕ) (h1 : cost_equation1 x y) (h2 : cost_equation2 x y) : x = 8 ∧ y = 10 := 
sorry

-- Define the condition for valid purchases of coffee A and B
def valid_purchase (m n : ℕ) : Prop := 8 * m + 10 * n = 200

-- Prove that there are 4 ways to purchase coffee A and B with 200 yuan
theorem purchase_ways : ∃ several : ℕ, several = 4 ∧ (∃ m n : ℕ, valid_purchase m n) := 
sorry

end coffee_prices_purchase_ways_l123_123930


namespace man_age_difference_l123_123946

theorem man_age_difference (S M : ℕ) (h1 : S = 22) (h2 : M + 2 = 2 * (S + 2)) : M - S = 24 :=
by
  sorry

end man_age_difference_l123_123946


namespace arrange_squares_l123_123600

theorem arrange_squares (n : ℕ) (h: n ≥ 5) : 
  ∃ (arrangement : list (ℕ × ℕ × ℕ × ℕ)), 
    (∀ (sq ∈ arrangement), sq.1 = sq.2 ∧ sq.3 = sq.4 ∧ sq.1 < sq.2 < ... < sq.n) ∧ 
    (∀ (sq1 sq2 ∈ arrangement), sq1 ≠ sq2 → 
       (sq1.1 = sq2.3 ∧ sq1.2 = sq2.4) ∨ 
       (sq1.3 = sq2.1 ∧ sq1.4 = sq2.2)) := 
sorry

end arrange_squares_l123_123600


namespace inequality_solution_l123_123975

theorem inequality_solution :
  {x : ℝ | (x^2 - 1) / (x - 3)^2 ≥ 0} = (Set.Iic (-1) ∪ Set.Ici 1) :=
by
  sorry

end inequality_solution_l123_123975


namespace circle_values_of_a_l123_123625

theorem circle_values_of_a (a : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 + 2*a*x + 2*a*y + 2*a^2 + a - 1 = 0) ↔ (a = -1 ∨ a = 0) :=
by
  sorry

end circle_values_of_a_l123_123625


namespace gcd_subtract_ten_l123_123920

theorem gcd_subtract_ten (a b : ℕ) (h₁ : a = 720) (h₂ : b = 90) : (Nat.gcd a b) - 10 = 80 := by
  sorry

end gcd_subtract_ten_l123_123920


namespace gcd_problem_l123_123468

-- Define the variables according to the conditions
def m : ℤ := 123^2 + 235^2 + 347^2
def n : ℤ := 122^2 + 234^2 + 348^2

-- Lean statement for the proof problem
theorem gcd_problem : Int.gcd m n = 1 := sorry

end gcd_problem_l123_123468


namespace c_sub_a_equals_90_l123_123782

variables (a b c : ℝ)

theorem c_sub_a_equals_90 (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 90) : c - a = 90 :=
by
  sorry

end c_sub_a_equals_90_l123_123782


namespace seconds_in_hours_3_5_l123_123392

theorem seconds_in_hours_3_5 : (60 * (60 * 3.5) = 12600) :=
by
  calc
    60 * (60 * 3.5) = 60 * 210  : by rw [mul_comm 60 3.5, mul_assoc, show 60 * 3.5 = 210, from rfl]
    ...              = 12600    : by rw [mul_comm 60 210, show 60 * 210 = 12600, from rfl]

end seconds_in_hours_3_5_l123_123392


namespace P_subset_M_l123_123215

def P : Set ℝ := {x | x^2 - 6 * x + 9 = 0}
def M : Set ℝ := {x | x > 1}

theorem P_subset_M : P ⊂ M := by sorry

end P_subset_M_l123_123215


namespace ellipse_foci_coordinates_l123_123721

theorem ellipse_foci_coordinates :
  (∀ x y : ℝ, x^2 / 9 + y^2 / 5 = 1 → (x = 2 ∧ y = 0) ∨ (x = -2 ∧ y = 0)) :=
by
  sorry

end ellipse_foci_coordinates_l123_123721


namespace profit_ratio_l123_123896

def praveen_initial_capital : ℝ := 3500
def hari_initial_capital : ℝ := 9000.000000000002
def total_months : ℕ := 12
def months_hari_invested : ℕ := total_months - 5

def effective_capital (initial_capital : ℝ) (months : ℕ) : ℝ :=
  initial_capital * months

theorem profit_ratio :
  effective_capital praveen_initial_capital total_months / effective_capital hari_initial_capital months_hari_invested 
  = 2 / 3 :=
by
  sorry

end profit_ratio_l123_123896


namespace find_unique_function_l123_123840

theorem find_unique_function (f : ℚ → ℚ)
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) :
  ∀ x : ℚ, f x = x + 1 :=
by
  sorry

end find_unique_function_l123_123840


namespace max_area_parallelogram_l123_123635

theorem max_area_parallelogram
    (P : ℝ)
    (a b : ℝ)
    (h1 : P = 60)
    (h2 : a = 3 * b)
    (h3 : P = 2 * a + 2 * b) :
    (a * b ≤ 168.75) :=
by
  -- We prove that given the conditions, the maximum area is 168.75 square units.
  sorry

end max_area_parallelogram_l123_123635


namespace range_of_a_l123_123051

open Real

theorem range_of_a (a : ℝ) :
  (∀ x > 0, ae^x + x + x * log x ≥ x^2) → a ≥ 1 / exp 2 :=
sorry

end range_of_a_l123_123051


namespace symmetric_point_l123_123526

-- Define the given conditions
def pointP : (ℤ × ℤ) := (3, -2)
def symmetry_line (y : ℤ) := (y = 1)

-- Prove the assertion that point Q is (3, 4)
theorem symmetric_point (x y1 y2 : ℤ) (hx: x = 3) (hy1: y1 = -2) (hy : symmetry_line 1) :
  (x, 2 * 1 - y1) = (3, 4) :=
by
  sorry

end symmetric_point_l123_123526


namespace union_complement_A_B_l123_123568

-- Definitions based on conditions
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | x < 6}
def C_R (S : Set ℝ) : Set ℝ := {x | ¬ (x ∈ S)}

-- The proof problem statement
theorem union_complement_A_B :
  (C_R B ∪ A = {x | 0 ≤ x}) :=
by 
  sorry

end union_complement_A_B_l123_123568


namespace mary_and_joan_marbles_l123_123423

theorem mary_and_joan_marbles : 9 + 3 = 12 :=
by
  rfl

end mary_and_joan_marbles_l123_123423


namespace employees_use_public_transport_l123_123682

-- Define the main assumptions based on the given conditions
def total_employees := 100
def drives_to_work := 0.60
def fraction_take_public_transport := 0.5

-- Define the problem statement
theorem employees_use_public_transport : 
  let drives := drives_to_work * total_employees in
  let doesnt_drive := total_employees - drives in
  let takes_public_transport := doesnt_drive * fraction_take_public_transport in
  takes_public_transport = 20 :=
by
  sorry

end employees_use_public_transport_l123_123682


namespace cos_240_eq_neg_half_l123_123310

theorem cos_240_eq_neg_half : ∀ (deg: ℝ), 
  deg = 240 → 
  (∀ a b : ℝ, 240 = a + b → a = 180 → b = 60 → 
    cos 240 = cos a * cos b - sin a * sin b) → 
  cos 180 = -1 →
  sin 180 = 0 →
  cos 60 = 1 / 2 →
  cos 240 = -1 / 2 :=
by 
  intros deg h_deg h_sum h_cos_180 h_sin_180 h_cos_60
  rw h_deg at h_sum
  have h_cos_identity := h_sum 180 60 rfl rfl rfl
  simp [h_cos_180, h_sin_180, h_cos_60] at h_cos_identity
  exact h_cos_identity

end cos_240_eq_neg_half_l123_123310


namespace system_sampling_arithmetic_sequence_l123_123145

theorem system_sampling_arithmetic_sequence :
  ∃ (seq : Fin 5 → ℕ), seq 0 = 8 ∧ seq 3 = 104 ∧ seq 1 = 40 ∧ seq 2 = 72 ∧ seq 4 = 136 ∧ 
    (∀ n m : Fin 5, 0 < n.val - m.val → seq n.val = seq m.val + 32 * (n.val - m.val)) :=
sorry

end system_sampling_arithmetic_sequence_l123_123145


namespace line_intersects_y_axis_at_point_intersection_at_y_axis_l123_123703

theorem line_intersects_y_axis_at_point :
  ∃ y, 5 * 0 - 7 * y = 35 := sorry

theorem intersection_at_y_axis :
  (∃ y, 5 * 0 - 7 * y = 35) → 0 - 7 * (-5) = 35 := sorry

end line_intersects_y_axis_at_point_intersection_at_y_axis_l123_123703


namespace intersection_A_C_U_B_l123_123771

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | Real.log x / Real.log 2 > 0}
def C_U_B : Set ℝ := {x | ¬ (Real.log x / Real.log 2 > 0)}

theorem intersection_A_C_U_B :
  A ∩ C_U_B = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_A_C_U_B_l123_123771


namespace fruit_vendor_l123_123733

theorem fruit_vendor (x y a b : ℕ) (C1 : 60 * x + 40 * y = 3100) (C2 : x + y = 60) 
                     (C3 : 15 * a + 20 * b = 600) (C4 : 3 * a + 4 * b = 120)
                     (C5 : 3 * a + 4 * b + 3 * (x - a) + 4 * (y - b) = 250) :
  (x = 35 ∧ y = 25) ∧ (820 - 12 * a - 16 * b = 340) ∧ (a + b = 52 ∨ a + b = 53) :=
by
  sorry

end fruit_vendor_l123_123733


namespace equilateral_if_acute_and_altitude_bisector_median_equal_l123_123777

theorem equilateral_if_acute_and_altitude_bisector_median_equal
  (A B C : Type) [triangle(ABC)]
  (acuteABC : ∀ (α β γ : ℝ), α < 90 ∧ β < 90 ∧ γ < 90) 
  (h_a : (altitude A B C)) 
  (l_b : (bisector B A C)) 
  (m_c : (median C A B))
  (h_a_eq_l_b : h_a = l_b) 
  (l_b_eq_m_c : l_b = m_c) :
  equilateral_triangle(ABC) :=
by 
  sorry

end equilateral_if_acute_and_altitude_bisector_median_equal_l123_123777


namespace simplify_expression_l123_123813

theorem simplify_expression : (0.4 * 0.5 + 0.3 * 0.2) = 0.26 := by
  sorry

end simplify_expression_l123_123813


namespace determinant_roots_l123_123089

theorem determinant_roots (s p q a b c : ℂ) 
  (h : ∀ x : ℂ, x^3 - s*x^2 + p*x + q = (x - a) * (x - b) * (x - c)) :
  (1 + a) * ((1 + b) * (1 + c) - 1) - ((1) * (1 + c) - 1) + ((1) - (1 + b)) = p + 3 * s :=
by {
  -- expanded determinant calculations
  sorry
}

end determinant_roots_l123_123089


namespace cos_240_is_neg_half_l123_123344

noncomputable def cos_240 : ℝ := Real.cos (240 * Real.pi / 180)

theorem cos_240_is_neg_half : cos_240 = -1/2 := by
  sorry

end cos_240_is_neg_half_l123_123344


namespace rectangle_ratio_constant_l123_123636

theorem rectangle_ratio_constant (length width : ℝ) (d k : ℝ)
  (h1 : length/width = 5/2)
  (h2 : 2 * (length + width) = 28)
  (h3 : d^2 = length^2 + width^2)
  (h4 : (length * width) = k * d^2) :
  k = (10/29) := by
  sorry

end rectangle_ratio_constant_l123_123636


namespace student_council_profit_l123_123110

def boxes : ℕ := 48
def erasers_per_box : ℕ := 24
def price_per_eraser : ℝ := 0.75

theorem student_council_profit :
  boxes * erasers_per_box * price_per_eraser = 864 := 
by
  sorry

end student_council_profit_l123_123110


namespace exists_x_y_l123_123611

theorem exists_x_y (n : ℕ) (hn : 0 < n) :
  ∃ x y : ℕ, n < x ∧ ¬ x ∣ y ∧ x^x ∣ y^y :=
by sorry

end exists_x_y_l123_123611


namespace miaCompletedAdditionalTasksOn6Days_l123_123968

def numDaysCompletingAdditionalTasks (n m : ℕ) : Prop :=
  n + m = 15 ∧ 4 * n + 7 * m = 78

theorem miaCompletedAdditionalTasksOn6Days (n m : ℕ): numDaysCompletingAdditionalTasks n m -> m = 6 :=
by
  intro h
  sorry

end miaCompletedAdditionalTasksOn6Days_l123_123968


namespace cosine_240_l123_123363

theorem cosine_240 (h1 : Real.cos 60 = 1 / 2) : Real.cos 240 = -1 / 2 :=
by
  have h2 : Real.cos 240 = -Real.cos 60 := by sorry
  rw [h2, h1]
  exact neg_div
  sorry

end cosine_240_l123_123363


namespace reggie_books_l123_123230

/-- 
Reggie's father gave him $48. Reggie bought some books, each of which cost $2, 
and now he has $38 left. How many books did Reggie buy?
-/
theorem reggie_books (initial_amount spent_amount remaining_amount book_cost books_bought : ℤ)
  (h_initial : initial_amount = 48)
  (h_remaining : remaining_amount = 38)
  (h_book_cost : book_cost = 2)
  (h_spent : spent_amount = initial_amount - remaining_amount)
  (h_books_bought : books_bought = spent_amount / book_cost) :
  books_bought = 5 :=
by
  sorry

end reggie_books_l123_123230


namespace unique_triangle_exists_l123_123125

theorem unique_triangle_exists : 
  (¬ (∀ (a b c : ℝ), a = 1 ∧ b = 2 ∧ c = 3 → a + b > c)) ∧
  (¬ (∀ (a b A : ℝ), a = 1 ∧ b = 2 ∧ A = 30 → ∃ (C : ℝ), C > 0)) ∧
  (¬ (∀ (a b A : ℝ), a = 1 ∧ b = 2 ∧ A = 100 → ∃ (C : ℝ), C > 0)) ∧
  (∀ (b c B : ℝ), b = 1 ∧ c = 1 ∧ B = 45 → ∃! (a c B : ℝ), b = 1 ∧ c = 1 ∧ B = 45) :=
by sorry

end unique_triangle_exists_l123_123125


namespace find_x_l123_123577

theorem find_x : 
  ∀ x : ℝ, (1 / (x + 4) + 1 / (x - 4) = 1 / (x - 4)) → x = 1 / 2 := 
by 
  sorry

end find_x_l123_123577


namespace rope_segment_equation_l123_123967

theorem rope_segment_equation (x : ℝ) (h1 : 2 - x > 0) :
  x^2 = 2 * (2 - x) :=
by
  sorry

end rope_segment_equation_l123_123967


namespace compare_logs_l123_123372

open Real

noncomputable def a := log 6 / log 3
noncomputable def b := 1 / log 5
noncomputable def c := log 14 / log 7

theorem compare_logs : a > b ∧ b > c := by
  sorry

end compare_logs_l123_123372


namespace wealth_ratio_l123_123965

theorem wealth_ratio 
  (P W : ℝ)
  (hP_pos : 0 < P)
  (hW_pos : 0 < W)
  (pop_A : ℝ := 0.30 * P)
  (wealth_A : ℝ := 0.40 * W)
  (pop_B : ℝ := 0.20 * P)
  (wealth_B : ℝ := 0.25 * W)
  (avg_wealth_A : ℝ := wealth_A / pop_A)
  (avg_wealth_B : ℝ := wealth_B / pop_B) :
  avg_wealth_A / avg_wealth_B = 16 / 15 :=
by
  sorry

end wealth_ratio_l123_123965


namespace min_value_sin_cos_expression_l123_123032

theorem min_value_sin_cos_expression : ∀ x : ℝ, 
  ∃ y : ℝ, y = (9 / 10) ∧ (y = infi (fun x => (sin x)^8 + (cos x)^8 + 1) / ((sin x)^6 + (cos x)^6 + 1)) :=
begin
  sorry
end

end min_value_sin_cos_expression_l123_123032


namespace tile_floor_with_polygons_l123_123114

theorem tile_floor_with_polygons (x y z: ℕ) (h1: 3 ≤ x) (h2: 3 ≤ y) (h3: 3 ≤ z) 
  (h_seamless: ((1 - (2 / (x: ℝ))) * 180 + (1 - (2 / (y: ℝ))) * 180 + (1 - (2 / (z: ℝ))) * 180 = 360)) :
  (1 / (x: ℝ) + 1 / (y: ℝ) + 1 / (z: ℝ) = 1 / 2) :=
by
  sorry

end tile_floor_with_polygons_l123_123114


namespace count_two_digit_primess_with_digit_sum_10_eq_3_l123_123540

def is_two_digit_prime_with_digit_sum_10 (n : ℕ) : Prop :=
  (10 ≤ n ∧ n < 100) ∧ Prime n ∧ (n / 10 + n % 10 = 10)

theorem count_two_digit_primess_with_digit_sum_10_eq_3 :
  (Finset.filter is_two_digit_prime_with_digit_sum_10 (Finset.range 100)).card = 3 := sorry

end count_two_digit_primess_with_digit_sum_10_eq_3_l123_123540


namespace sets_of_laces_needed_l123_123151

-- Define the conditions as constants
def teams := 4
def members_per_team := 10
def pairs_per_member := 2
def skates_per_pair := 2
def sets_of_laces_per_skate := 3

-- Formulate and state the theorem to be proven
theorem sets_of_laces_needed : 
  sets_of_laces_per_skate * (teams * members_per_team * (pairs_per_member * skates_per_pair)) = 480 :=
by sorry

end sets_of_laces_needed_l123_123151


namespace cos_240_eq_negative_half_l123_123324

theorem cos_240_eq_negative_half : Real.cos (240 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_240_eq_negative_half_l123_123324


namespace combined_bus_rides_length_l123_123919

theorem combined_bus_rides_length :
  let v := 0.62
  let z := 0.5
  let a := 0.72
  v + z + a = 1.84 :=
by
  let v := 0.62
  let z := 0.5
  let a := 0.72
  show v + z + a = 1.84
  sorry

end combined_bus_rides_length_l123_123919


namespace hash_hash_hash_72_eq_12_5_l123_123514

def hash (N : ℝ) : ℝ := 0.5 * N + 2

theorem hash_hash_hash_72_eq_12_5 : hash (hash (hash 72)) = 12.5 := 
by
  sorry

end hash_hash_hash_72_eq_12_5_l123_123514


namespace shortest_distance_between_circles_zero_l123_123282

noncomputable def center_radius_circle1 : (ℝ × ℝ) × ℝ :=
  let c1 := (3, -5)
  let r1 := Real.sqrt 20
  (c1, r1)

noncomputable def center_radius_circle2 : (ℝ × ℝ) × ℝ :=
  let c2 := (-4, 1)
  let r2 := Real.sqrt 1
  (c2, r2)

theorem shortest_distance_between_circles_zero :
  let c1 := center_radius_circle1.1
  let r1 := center_radius_circle1.2
  let c2 := center_radius_circle2.1
  let r2 := center_radius_circle2.2
  let dist := Real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2)
  dist < r1 + r2 → 0 = 0 :=
by
  intros
  -- Add appropriate steps for the proof (skipping by using sorry for now)
  sorry

end shortest_distance_between_circles_zero_l123_123282


namespace terms_of_sequence_are_equal_l123_123222

theorem terms_of_sequence_are_equal
    (n : ℤ)
    (h_n : n ≥ 2018)
    (a b : ℕ → ℕ)
    (h_a_distinct : ∀ i j, i ≠ j → a i ≠ a j)
    (h_b_distinct : ∀ i j, i ≠ j → b i ≠ b j)
    (h_a_bounds : ∀ i, a i ≤ 5 * n)
    (h_b_bounds : ∀ i, b i ≤ 5 * n)
    (h_arith_seq : ∀ i, (a (i + 1) * b i - a i * b (i + 1)) = (a 1 * b 0 - a 0 * b 1) * i) :
    ∀ i j, (a i * b j = a j * b i) := 
by 
  sorry

end terms_of_sequence_are_equal_l123_123222


namespace largest_n_for_factored_polynomial_l123_123726

theorem largest_n_for_factored_polynomial :
  ∃ (n : ℤ), (∀ (A B : ℤ), 3 * A * B = 108 → n = 3 * B + A) ∧ n = 325 :=
by 
  sorry

end largest_n_for_factored_polynomial_l123_123726


namespace certain_event_is_eventC_l123_123119

-- Definitions for the conditions:
def eventA := "A vehicle randomly arriving at an intersection encountering a red light"
def eventB := "The sun rising from the west in the morning"
def eventC := "Two out of 400 people sharing the same birthday"
def eventD := "Tossing a fair coin with the head facing up"

-- The proof goal: proving that event C is the certain event.
theorem certain_event_is_eventC : eventC = "Two out of 400 people sharing the same birthday" :=
sorry

end certain_event_is_eventC_l123_123119


namespace inequality_for_positive_reals_l123_123612

variable (a b c : ℝ)

theorem inequality_for_positive_reals (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := by
  sorry

end inequality_for_positive_reals_l123_123612


namespace cos_240_eq_neg_half_l123_123350

open Real

theorem cos_240_eq_neg_half : cos (240 * π / 180) = -1/2 :=
by
  -- Step 1: Decompose the angle 240° = 180° + 60°
  have h1 : 240 * π / 180 = π + 60 * π / 180,
  { 
    norm_num, 
    field_simp, 
    linarith 
  },
  -- Step 2: Use the fact that the cosine of (π + θ) = - cos(θ)
  rw [h1, cos_add_pi],
  -- Step 3: Given that cos(60°) = 1/2
  have h2 : cos (60 * π / 180) = 1/2,
  {
    norm_num,
    exact Real.cos_pi_div_three (),
  },
  -- Conclude that cos(240°) = -1/2
  rw h2,
  norm_num

end cos_240_eq_neg_half_l123_123350


namespace max_area_of_triangle_ABC_l123_123081

open Classical

theorem max_area_of_triangle_ABC (A B C : ℝ) :
  ( ∀ (α β γ a b c : ℝ), 
      α + β = 90 ∧ a + b + c = 12 ∧ 
      (α + β + γ = 180) ∧ (c = sqrt (a ^ 2 + b ^ 2)) ∧ 
      γ = 90 ∧ 
      (cos(α) / sin(β) + cos(β) / sin(α)) = 2 
      → 
      ∃ area : ℝ, 0 ≤ area ∧ area = 36 * (3 - 2 * sqrt 2)
  )

end max_area_of_triangle_ABC_l123_123081


namespace julia_drove_miles_l123_123681

theorem julia_drove_miles :
  ∀ (daily_rental_cost cost_per_mile total_paid : ℝ),
    daily_rental_cost = 29 →
    cost_per_mile = 0.08 →
    total_paid = 46.12 →
    total_paid - daily_rental_cost = cost_per_mile * 214 :=
by
  intros _ _ _ d_cost_eq cpm_eq tp_eq
  -- calculation and proof steps will be filled here
  sorry

end julia_drove_miles_l123_123681


namespace cos_beta_value_l123_123766

theorem cos_beta_value
  (α β : ℝ)
  (hαβ : 0 < α ∧ α < π ∧ 0 < β ∧ β < π)
  (h1 : Real.sin (α + β) = 5 / 13)
  (h2 : Real.tan (α / 2) = 1 / 2) :
  Real.cos β = -16 / 65 := 
by 
  sorry

end cos_beta_value_l123_123766


namespace valid_third_side_length_l123_123739

theorem valid_third_side_length : 4 < 6 ∧ 6 < 10 :=
by
  exact ⟨by norm_num, by norm_num⟩

end valid_third_side_length_l123_123739


namespace prove_absolute_value_subtract_power_l123_123183

noncomputable def smallest_absolute_value : ℝ := 0

theorem prove_absolute_value_subtract_power (b : ℝ) 
  (h1 : smallest_absolute_value = 0) 
  (h2 : b * b = 1) : 
  (|smallest_absolute_value - 2| - b ^ 2023 = 1) 
  ∨ (|smallest_absolute_value - 2| - b ^ 2023 = 3) :=
sorry

end prove_absolute_value_subtract_power_l123_123183


namespace SomuAge_l123_123814

theorem SomuAge (F S : ℕ) (h1 : S = F / 3) (h2 : S - 8 = (F - 8) / 5) : S = 16 :=
by 
  sorry

end SomuAge_l123_123814


namespace distance_inequality_of_centers_l123_123843

open EuclideanGeometry

variables {α β γ : ℝ} -- Angles of triangle ABC
variables {a b c : ℝ} -- Sides of triangle ABC
variables {A B C A_1 B_1 C_1 I H: Point} -- Points in the triangles

noncomputable def point_on_side (A B P : Point) : Prop :=
  ∃ r, 0 < r ∧ r < 1 ∧ A + r • (B - A) = P

noncomputable def angle_bisector (A B C P : Point) : Prop :=
  inner (B - A) (C - P) / ∥C - P∥ = inner (C - A) (B - P) / ∥B - P∥

noncomputable def incenter (ABC : Triangle) (I : Point) : Prop := 
  I = insphere_center ABC

noncomputable def orthocenter (A1 B1 C1 H : Triangle) (H : Point) : Prop :=
  H = orthocenter A1 B1 C1

theorem distance_inequality_of_centers 
  (h_acute : is_acute A B C)
  (h_pts sides: point_on_side B C A_1)
  (h_pt_B1: point_on_side C A B_1)
  (h_pt_C1: point_on_side A B C_1)
  (h_bisector_A1: angle_bisector A B C A_1)
  (h_bisector_B1: angle_bisector B C A B_1)
  (h_bisector_C1: angle_bisector C A B C_1)
  (h_incenter: incenter ⟨A, B, C⟩ I)
  (h_orthocenter: orthocenter ⟨A_1, B_1, C_1⟩ H):
  (distance A H) + (distance B H) + (distance C H) ≥
  (distance A I) + (distance B I) + (distance C I) := 
sorry

end distance_inequality_of_centers_l123_123843


namespace cube_surface_area_l123_123483

-- Define the edge length of the cube.
def edge_length (a : ℝ) : ℝ := 6 * a

-- Define the surface area of a cube given the edge length.
def surface_area (e : ℝ) : ℝ := 6 * (e * e)

-- The theorem to prove.
theorem cube_surface_area (a : ℝ) : surface_area (edge_length a) = 216 * (a * a) := 
  sorry

end cube_surface_area_l123_123483


namespace rational_solution_quadratic_l123_123517

theorem rational_solution_quadratic (m : ℕ) (h_pos : m > 0) : 
  (∃ (x : ℚ), x * x * m + 25 * x + m = 0) ↔ m = 10 ∨ m = 12 :=
by sorry

end rational_solution_quadratic_l123_123517


namespace range_of_a_l123_123788

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then x^2 - 2 * a * x - 2 else x + 36 / x - 6 * a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ f 2 a) ↔ (2 ≤ a ∧ a ≤ 5) :=
sorry

end range_of_a_l123_123788


namespace scenery_photos_correct_l123_123602

-- Define the problem conditions
def animal_photos := 10
def flower_photos := 3 * animal_photos
def photos_total := 45
def scenery_photos := flower_photos - 10

-- State the theorem
theorem scenery_photos_correct : scenery_photos = 20 ∧ animal_photos + flower_photos + scenery_photos = photos_total := by
  sorry

end scenery_photos_correct_l123_123602


namespace prob_winning_5_beans_prob_success_first_round_zero_beans_l123_123201

noncomputable def prob_first_round_success : ℚ := 3 / 4
noncomputable def prob_second_round_success : ℚ := 2 / 3
noncomputable def prob_third_round_success : ℚ := 1 / 2
noncomputable def prob_choose_proceed : ℚ := 1 / 2

-- Prove the probability of winning exactly 5 learning beans is 3/8
theorem prob_winning_5_beans : 
    prob_first_round_success * prob_choose_proceed = 3 / 8 := 
by sorry

-- Define events for successfully completing the first round but ending with zero beans
noncomputable def prob_A1 : ℚ := prob_first_round_success * prob_choose_proceed * (1 - prob_second_round_success)
noncomputable def prob_A2 : ℚ := prob_first_round_success * prob_choose_proceed * prob_second_round_success * prob_choose_proceed * (1 - prob_third_round_success)

-- Prove the probability of successfully completing the first round but ending with zero beans is 3/16
theorem prob_success_first_round_zero_beans : 
    prob_A1 + prob_A2 = 3 / 16 := 
by sorry

end prob_winning_5_beans_prob_success_first_round_zero_beans_l123_123201


namespace contrapositive_of_proposition_l123_123784

-- Proposition: If xy=0, then x=0
def proposition (x y : ℝ) : Prop := x * y = 0 → x = 0

-- Contrapositive: If x ≠ 0, then xy ≠ 0
def contrapositive (x y : ℝ) : Prop := x ≠ 0 → x * y ≠ 0

-- Proof that contrapositive of the given proposition holds
theorem contrapositive_of_proposition (x y : ℝ) : proposition x y ↔ contrapositive x y :=
by {
  sorry
}

end contrapositive_of_proposition_l123_123784


namespace clearance_sale_total_earnings_l123_123939

-- Define the variables used in the problem
def total_jackets := 214
def price_before_noon := 31.95
def price_after_noon := 18.95
def jackets_sold_after_noon := 133

-- Calculate the total earnings
def total_earnings_from_clearance_sale : Prop :=
  (133 * 18.95 + (214 - 133) * 31.95) = 5107.30

-- State the theorem to be proven
theorem clearance_sale_total_earnings : total_earnings_from_clearance_sale :=
  by sorry

end clearance_sale_total_earnings_l123_123939


namespace find_divisor_l123_123871

theorem find_divisor (D Q R Div : ℕ) (h1 : Q = 40) (h2 : R = 64) (h3 : Div = 2944) 
  (h4 : Div = (D * Q) + R) : D = 72 :=
by
  sorry

end find_divisor_l123_123871


namespace cos_240_eq_neg_half_l123_123306

theorem cos_240_eq_neg_half : Real.cos (4 * Real.pi / 3) = -1/2 :=
by
  sorry

end cos_240_eq_neg_half_l123_123306


namespace tenth_term_of_geometric_sequence_l123_123512

theorem tenth_term_of_geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) (tenth_term : ℚ) :
  a = 5 →
  r = 4 / 3 →
  n = 10 →
  tenth_term = a * r ^ (n - 1) →
  tenth_term = 1310720 / 19683 :=
by sorry

end tenth_term_of_geometric_sequence_l123_123512


namespace palabras_bookstore_workers_l123_123203

theorem palabras_bookstore_workers (W : ℕ) (h1 : W / 2 = (W / 2)) (h2 : W / 6 = (W / 6)) (h3 : 12 = 12) (h4 : W - (W / 2 + W / 6 - 12 + 1) = 35) : W = 210 := 
sorry

end palabras_bookstore_workers_l123_123203


namespace ball_bounce_height_l123_123004

theorem ball_bounce_height (a : ℝ) (r : ℝ) (threshold : ℝ) (k : ℕ) 
  (h_a : a = 20) (h_r : r = 1/2) (h_threshold : threshold = 0.5) :
  20 * (r^k) < threshold ↔ k = 5 :=
by sorry

end ball_bounce_height_l123_123004


namespace original_six_digit_number_l123_123787

theorem original_six_digit_number :
  ∃ a b c d e : ℕ, 
  (100000 + 10000 * a + 1000 * b + 100 * c + 10 * d + e = 142857) ∧ 
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + 1 = 64 * (100000 + 10000 * a + 1000 * b + 100 * c + 10 * d + e)) :=
by
  sorry

end original_six_digit_number_l123_123787


namespace matrix_not_invertible_x_l123_123714

theorem matrix_not_invertible_x (x : ℝ) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := ![![2 + x, 9], ![4 - x, 10]]
  A.det = 0 ↔ x = 16 / 19 := sorry

end matrix_not_invertible_x_l123_123714


namespace trajectory_of_P_eqn_l123_123734

noncomputable def point_A : ℝ × ℝ := (1, 0)

def curve_C (x : ℝ) : ℝ := x^2 - 2

def symmetric_point (Qx Qy Px Py : ℝ) : Prop :=
  Qx = 2 - Px ∧ Qy = -Py

theorem trajectory_of_P_eqn (Qx Qy Px Py : ℝ) (hQ_on_C : Qy = curve_C Qx)
  (h_symm : symmetric_point Qx Qy Px Py) :
  Py = -Px^2 + 4 * Px - 2 :=
by
  sorry

end trajectory_of_P_eqn_l123_123734


namespace problem_l123_123218

noncomputable def f : Polynomial ℝ := 3 * X^5 + 4 * X^4 - 12 * X^3 - 8 * X^2 + X + 4
noncomputable def d : Polynomial ℝ := X^2 - 2 * X + 1

theorem problem (q r : Polynomial ℝ) (hq : f = q * d + r) (hr_deg : r.degree < d.degree) :
  q.eval 1 + r.eval (-1) = -13 :=
sorry

end problem_l123_123218


namespace find_function_value_at_2_l123_123579

variables {f : ℕ → ℕ}

theorem find_function_value_at_2 (H : ∀ x : ℕ, Nat.succ (Nat.succ x * Nat.succ x + f x) = 12) : f 2 = 4 :=
by
  sorry

end find_function_value_at_2_l123_123579


namespace max_chords_l123_123181

noncomputable def max_closed_chords (n : ℕ) (h : n ≥ 3) : ℕ :=
  n

/-- Given an integer number n ≥ 3 and n distinct points on a circle, labeled 1 through n,
prove that the maximum number of closed chords [ij], i ≠ j, having pairwise non-empty intersections is n. -/
theorem max_chords {n : ℕ} (h : n ≥ 3) :
  max_closed_chords n h = n := 
sorry

end max_chords_l123_123181


namespace parabola_focus_l123_123524

theorem parabola_focus (x y : ℝ) (h : y = 4 * x^2) : (0, 1) = (0, 1) :=
by 
  -- key steps would go here
  sorry

end parabola_focus_l123_123524


namespace parabola_y1_gt_y2_l123_123877

variable {x1 x2 y1 y2 : ℝ}

theorem parabola_y1_gt_y2 
  (hx1 : -4 < x1 ∧ x1 < -2) 
  (hx2 : 0 < x2 ∧ x2 < 2) 
  (hy1 : y1 = x1^2) 
  (hy2 : y2 = x2^2) : 
  y1 > y2 :=
by 
  sorry

end parabola_y1_gt_y2_l123_123877


namespace gas_cycle_work_done_l123_123485

noncomputable def p0 : ℝ := 10^5
noncomputable def V0 : ℝ := 1

theorem gas_cycle_work_done :
  (3 * Real.pi * p0 * V0 = 942) :=
by
  have h1 : p0 = 10^5 := by rfl
  have h2 : V0 = 1 := by rfl
  sorry

end gas_cycle_work_done_l123_123485


namespace group_population_l123_123493

theorem group_population :
  ∀ (men women children : ℕ),
  (men = 2 * women) →
  (women = 3 * children) →
  (children = 30) →
  (men + women + children = 300) :=
by
  intros men women children h_men h_women h_children
  sorry

end group_population_l123_123493


namespace arithmetic_sequence_s9_l123_123075

noncomputable def arithmetic_sum (a1 d n : ℝ) : ℝ :=
  n * (2*a1 + (n - 1)*d) / 2

noncomputable def general_term (a1 d n : ℝ) : ℝ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_s9 (a1 d : ℝ)
  (h1 : general_term a1 d 3 + general_term a1 d 4 + general_term a1 d 8 = 25) :
  arithmetic_sum a1 d 9 = 75 :=
by sorry

end arithmetic_sequence_s9_l123_123075


namespace y_completion_days_l123_123816

theorem y_completion_days (d : ℕ) (h : (12 : ℚ) / d + 1 / 4 = 1) : d = 16 :=
by
  sorry

end y_completion_days_l123_123816


namespace equilateral_triangle_percentage_l123_123149

theorem equilateral_triangle_percentage (s : Real) :
  let area_square := s^2
  let area_triangle := (Real.sqrt 3 / 4) * s^2
  let total_area := area_square + area_triangle
  area_triangle / total_area * 100 = (4 * Real.sqrt 3 - 3) / 13 * 100 := by
  sorry

end equilateral_triangle_percentage_l123_123149


namespace cos_240_eq_neg_half_l123_123320

theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Sorry to skip the proof
  sorry

end cos_240_eq_neg_half_l123_123320


namespace quadruple_solution_l123_123029

theorem quadruple_solution (a b p n : ℕ) (hp: Nat.Prime p) (hp_pos: p > 0) (ha_pos: a > 0) (hb_pos: b > 0) (hn_pos: n > 0) :
    a^3 + b^3 = p^n →
    (∃ k, k ≥ 1 ∧ (
        (a = 2^(k-1) ∧ b = 2^(k-1) ∧ p = 2 ∧ n = 3*k-2) ∨ 
        (a = 2 * 3^(k-1) ∧ b = 3^(k-1) ∧ p = 3 ∧ n = 3*k-1) ∨ 
        (a = 3^(k-1) ∧ b = 2 * 3^(k-1) ∧ p = 3 ∧ n = 3*k-1)
    )) := 
sorry

end quadruple_solution_l123_123029


namespace price_reduction_daily_profit_l123_123938

theorem price_reduction_daily_profit
    (profit_per_item : ℕ)
    (avg_daily_sales : ℕ)
    (item_increase_per_unit_price_reduction : ℕ)
    (target_daily_profit : ℕ)
    (x : ℕ) :
    profit_per_item = 40 →
    avg_daily_sales = 20 →
    item_increase_per_unit_price_reduction = 2 →
    target_daily_profit = 1200 →

    ((profit_per_item - x) * (avg_daily_sales + item_increase_per_unit_price_reduction * x) = target_daily_profit) →
    x = 20 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end price_reduction_daily_profit_l123_123938


namespace geometric_sequence_sum_l123_123386

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_a2 : a 2 = 2)
  (h_a5 : a 5 = 1 / 4) :
  a 1 * a 2 + a 2 * a 3 + a 3 * a 4 + a 4 * a 5 + a 5 * a 6 = 341 / 32 :=
by sorry

end geometric_sequence_sum_l123_123386


namespace units_digit_of_G_1000_l123_123019

def G (n : ℕ) : ℕ := 3 ^ (3 ^ n) + 1

theorem units_digit_of_G_1000 : (G 1000) % 10 = 2 := 
  sorry

end units_digit_of_G_1000_l123_123019


namespace largest_value_fraction_l123_123749

theorem largest_value_fraction (x y : ℝ) (hx : 10 ≤ x ∧ x ≤ 20) (hy : 40 ≤ y ∧ y ≤ 60) :
  ∃ z, z = (x^2 / (2 * y)) ∧ z ≤ 5 :=
by
  sorry

end largest_value_fraction_l123_123749


namespace move_up_4_units_l123_123227

-- Define the given points M and N
def M : ℝ × ℝ := (-1, -1)
def N : ℝ × ℝ := (-1, 3)

-- State the theorem to be proved
theorem move_up_4_units (M N : ℝ × ℝ) :
  (M = (-1, -1)) → (N = (-1, 3)) → (N = (M.1, M.2 + 4)) :=
by
  intros hM hN
  rw [hM, hN]
  sorry

end move_up_4_units_l123_123227


namespace line_through_two_points_l123_123630

-- Define the points (2,5) and (0,3)
structure Point where
  x : ℝ
  y : ℝ

def P1 : Point := {x := 2, y := 5}
def P2 : Point := {x := 0, y := 3}

-- General form of a line equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the target line equation as x - y + 3 = 0
def targetLine : Line := {a := 1, b := -1, c := 3}

-- The proof statement to show that the general equation of the line passing through the points (2, 5) and (0, 3) is x - y + 3 = 0
theorem line_through_two_points : ∃ a b c, ∀ x y : ℝ, 
    (a * x + b * y + c = 0) ↔ 
    ((∀ {P : Point}, P = P1 → targetLine.a * P.x + targetLine.b * P.y + targetLine.c = 0) ∧ 
     (∀ {P : Point}, P = P2 → targetLine.a * P.x + targetLine.b * P.y + targetLine.c = 0)) :=
sorry

end line_through_two_points_l123_123630


namespace original_price_of_shirts_is_397_66_l123_123912

variable (P : ℝ)
variable (h : 0.95 * 0.9 * P = 340)

theorem original_price_of_shirts_is_397_66 : P = 397.66 :=
by
  have h_main : P = 340 / (0.95 * 0.9) := by linarith
  have : P ≈ 397.66 := by linarith
  exact h_main

end original_price_of_shirts_is_397_66_l123_123912


namespace kathryn_financial_statement_l123_123882

def kathryn_remaining_money (rent : ℕ) (salary : ℕ) (share_rent : ℕ → ℕ) (total_expenses : ℕ → ℕ) (remaining_money : ℕ → ℕ) : Prop :=
  rent = 1200 ∧
  salary = 5000 ∧
  share_rent rent = rent / 2 ∧
  ∀ rent_total, total_expenses (share_rent rent_total) = (share_rent rent_total) + 2 * rent_total ∧
  remaining_money salary total_expenses = salary - total_expenses (share_rent rent)

theorem kathryn_financial_statement : kathryn_remaining_money 1200 5000 (λ rent, rent / 2) (λ rent, rent / 2 + 2 * rent) (λ salary expenses, salary - expenses (λ rent, rent / 2)) :=
by {
  sorry
}

end kathryn_financial_statement_l123_123882


namespace base_seven_to_base_ten_l123_123115

theorem base_seven_to_base_ten (n : ℕ) (h : n = 54231) : 
  (1 * 7^0 + 3 * 7^1 + 2 * 7^2 + 4 * 7^3 + 5 * 7^4) = 13497 :=
by
  sorry

end base_seven_to_base_ten_l123_123115


namespace num_two_digit_primes_with_digit_sum_10_l123_123554

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ (n / 10 + n % 10 = 10) ∧ is_prime n}.to_finset.card = 3 :=
by sorry

end num_two_digit_primes_with_digit_sum_10_l123_123554


namespace bottles_count_l123_123135

-- Defining the conditions from the problem statement
def condition1 (x y : ℕ) : Prop := 3 * x + 4 * y = 108
def condition2 (x y : ℕ) : Prop := 2 * x + 3 * y = 76

-- The proof statement combining conditions and the solution
theorem bottles_count (x y : ℕ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 20 ∧ y = 12 :=
sorry

end bottles_count_l123_123135


namespace work_fraction_left_l123_123005

theorem work_fraction_left (A_days B_days : ℕ) (work_days : ℕ)
  (hA : A_days = 15) (hB : B_days = 20) (h_work : work_days = 3) :
  1 - (work_days * ((1 / A_days) + (1 / B_days))) = 13 / 20 :=
by
  rw [hA, hB, h_work]
  simp
  sorry

end work_fraction_left_l123_123005


namespace problems_per_hour_l123_123732

def num_math_problems : ℝ := 17.0
def num_spelling_problems : ℝ := 15.0
def total_hours : ℝ := 4.0

theorem problems_per_hour :
  (num_math_problems + num_spelling_problems) / total_hours = 8.0 := by
  sorry

end problems_per_hour_l123_123732


namespace cos_240_degree_l123_123292

theorem cos_240_degree : Real.cos (240 * Real.pi / 180) = -1/2 :=
by
  sorry

end cos_240_degree_l123_123292


namespace find_number_l123_123936

theorem find_number (x : ℝ) (h : 0.36 * x = 129.6) : x = 360 :=
by sorry

end find_number_l123_123936


namespace group_population_l123_123491

theorem group_population :
  ∀ (men women children : ℕ),
  (men = 2 * women) →
  (women = 3 * children) →
  (children = 30) →
  (men + women + children = 300) :=
by
  intros men women children h_men h_women h_children
  sorry

end group_population_l123_123491


namespace cos_240_is_neg_half_l123_123346

noncomputable def cos_240 : ℝ := Real.cos (240 * Real.pi / 180)

theorem cos_240_is_neg_half : cos_240 = -1/2 := by
  sorry

end cos_240_is_neg_half_l123_123346


namespace polynomial_expansion_l123_123027

variable (t : ℝ)

theorem polynomial_expansion :
  (3 * t^3 + 2 * t^2 - 4 * t + 3) * (-4 * t^3 + 3 * t - 5) = -12 * t^6 - 8 * t^5 + 25 * t^4 - 21 * t^3 - 22 * t^2 + 29 * t - 15 :=
by {
  sorry
}

end polynomial_expansion_l123_123027


namespace length_of_AE_l123_123140

theorem length_of_AE (AF CE ED : ℝ) (ABCD_area : ℝ) (hAF : AF = 30) (hCE : CE = 40) (hED : ED = 50) (hABCD_area : ABCD_area = 7200) : ∃ AE : ℝ, AE = 322.5 := sorry

end length_of_AE_l123_123140


namespace cos_240_eq_neg_half_l123_123316

theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Sorry to skip the proof
  sorry

end cos_240_eq_neg_half_l123_123316


namespace solve_equation_nat_numbers_l123_123900

theorem solve_equation_nat_numbers (a b c d e f g : ℕ) 
  (h : a * b * c * d * e * f * g = a + b + c + d + e + f + g) : 
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ ((f = 2 ∧ g = 7) ∨ (f = 7 ∧ g = 2))) ∨ 
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ ((f = 3 ∧ g = 4) ∨ (f = 4 ∧ g = 3))) :=
sorry

end solve_equation_nat_numbers_l123_123900


namespace num_candidates_appeared_each_state_l123_123589

-- Definitions
def candidates_appear : ℕ := 8000
def sel_pct_A : ℚ := 0.06
def sel_pct_B : ℚ := 0.07
def additional_selections_B : ℕ := 80

-- Proof Problem Statement
theorem num_candidates_appeared_each_state (x : ℕ) 
  (h1 : x = candidates_appear) 
  (h2 : sel_pct_A * ↑x = 0.06 * ↑x) 
  (h3 : sel_pct_B * ↑x = 0.07 * ↑x) 
  (h4 : sel_pct_B * ↑x = sel_pct_A * ↑x + additional_selections_B) : 
  x = candidates_appear := sorry

end num_candidates_appeared_each_state_l123_123589


namespace green_to_blue_ratio_l123_123513

-- Definition of the problem conditions
variable (G B R : ℕ)
variable (H1 : 2 * G = R)
variable (H2 : B = 80)
variable (H3 : R = 1280)

-- Theorem statement: the ratio of the green car's speed to the blue car's speed is 8:1
theorem green_to_blue_ratio (G B R : ℕ) (H1 : 2 * G = R) (H2 : B = 80) (H3 : R = 1280) :
  G / B = 8 :=
by
  sorry

end green_to_blue_ratio_l123_123513


namespace simplify_expression_l123_123438

theorem simplify_expression (a : ℚ) (h : a^2 - a - 7/2 = 0) : 
  a^2 - (a - (2 * a) / (a + 1)) / ((a^2 - 2 * a + 1) / (a^2 - 1)) = 7 / 2 := 
by
  sorry

end simplify_expression_l123_123438


namespace inequality_satisfaction_l123_123907

theorem inequality_satisfaction (x y : ℝ) : 
  y - x < Real.sqrt (x^2) ↔ (y < 0 ∨ y < 2 * x) := by 
sorry

end inequality_satisfaction_l123_123907


namespace shortest_distance_from_circle_to_line_l123_123268

theorem shortest_distance_from_circle_to_line :
  let circle := { p : ℝ × ℝ | (p.1 - 5)^2 + (p.2 - 3)^2 = 9 }
  let line := { p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 2 = 0 }
  ∀ (M : ℝ × ℝ), M ∈ circle → ∃ d : ℝ, d = 2 ∧ ∀ q ∈ line, dist M q = d := 
sorry

end shortest_distance_from_circle_to_line_l123_123268


namespace A_share_value_l123_123143

-- Define the shares using the common multiplier x
variable (x : ℝ)

-- Define the shares in terms of x
def A_share := 5 * x
def B_share := 2 * x
def C_share := 4 * x
def D_share := 3 * x

-- Given condition that C gets Rs. 500 more than D
def condition := C_share - D_share = 500

-- State the theorem to determine A's share given the conditions
theorem A_share_value (h : condition) : A_share = 2500 := by 
  sorry

end A_share_value_l123_123143


namespace length_of_tracks_l123_123148

theorem length_of_tracks (x y : ℕ) 
  (h1 : 6 * (x + 2 * y) = 5000)
  (h2 : 7 * (x + y) = 5000) : x = 5 * y :=
  sorry

end length_of_tracks_l123_123148


namespace floor_expression_equals_zero_l123_123017

theorem floor_expression_equals_zero
  (a b c : ℕ)
  (ha : a = 2010)
  (hb : b = 2007)
  (hc : c = 2008) :
  Int.floor ((a^3 : ℚ) / (b * c^2) - (c^3 : ℚ) / (b^2 * a)) = 0 := 
  sorry

end floor_expression_equals_zero_l123_123017


namespace solve_system_eqns_l123_123901

theorem solve_system_eqns (x y z : ℝ) :
  x^2 - 23 * y + 66 * z + 612 = 0 ∧
  y^2 + 62 * x - 20 * z + 296 = 0 ∧
  z^2 - 22 * x + 67 * y + 505 = 0 ↔
  x = -20 ∧ y = -22 ∧ z = -23 :=
by
  sorry

end solve_system_eqns_l123_123901


namespace correct_operation_l123_123121

theorem correct_operation : ∃ (a : ℝ), (3 + Real.sqrt 2 ≠ 3 * Real.sqrt 2) ∧ 
  ((a ^ 2) ^ 3 ≠ a ^ 5) ∧
  (Real.sqrt ((-7 : ℝ) ^ 2) ≠ -7) ∧
  (4 * a ^ 2 * a = 4 * a ^ 3) :=
by
  sorry

end correct_operation_l123_123121


namespace measure_exterior_angle_BAC_l123_123951

-- Define the interior angle of a regular nonagon
def nonagon_interior_angle := (180 * (9 - 2)) / 9

-- Define the exterior angle of the nonagon
def nonagon_exterior_angle := 360 - nonagon_interior_angle

-- The square's interior angle
def square_interior_angle := 90

-- The question to be proven
theorem measure_exterior_angle_BAC :
  nonagon_exterior_angle - square_interior_angle = 130 :=
  by
  sorry

end measure_exterior_angle_BAC_l123_123951


namespace drunk_drivers_traffic_class_l123_123755

-- Define the variables for drunk drivers and speeders
variable (d s : ℕ)

-- Define the given conditions as hypotheses
theorem drunk_drivers_traffic_class (h1 : d + s = 45) (h2 : s = 7 * d - 3) : d = 6 := by
  sorry

end drunk_drivers_traffic_class_l123_123755


namespace roof_length_width_diff_l123_123794

theorem roof_length_width_diff (w l : ℕ) (h1 : l = 4 * w) (h2 : 784 = l * w) : l - w = 42 := by
  sorry

end roof_length_width_diff_l123_123794


namespace sandy_shopping_l123_123617

theorem sandy_shopping (T : ℝ) (h : 0.70 * T = 217) : T = 310 := sorry

end sandy_shopping_l123_123617


namespace part1_part2_l123_123001

noncomputable def total_seating_arrangements : ℕ := 840
noncomputable def non_adjacent_4_people_arrangements : ℕ := 24
noncomputable def three_empty_adjacent_arrangements : ℕ := 120

theorem part1 : total_seating_arrangements - non_adjacent_4_people_arrangements = 816 := by
  sorry

theorem part2 : total_seating_arrangements - three_empty_adjacent_arrangements = 720 := by
  sorry

end part1_part2_l123_123001


namespace work_days_together_l123_123812

theorem work_days_together (p_rate q_rate : ℝ) (fraction_left : ℝ) (d : ℝ) 
  (h₁ : p_rate = 1/15) (h₂ : q_rate = 1/20) (h₃ : fraction_left = 8/15)
  (h₄ : (p_rate + q_rate) * d = 1 - fraction_left) : d = 4 :=
by
  sorry

end work_days_together_l123_123812


namespace solve_equation_1_solve_equation_2_l123_123234

theorem solve_equation_1 (x : ℝ) (h₁ : x - 4 = -5) : x = -1 :=
sorry

theorem solve_equation_2 (x : ℝ) (h₂ : (1/2) * x + 2 = 6) : x = 8 :=
sorry

end solve_equation_1_solve_equation_2_l123_123234


namespace tanks_fill_l123_123626

theorem tanks_fill
  (c : ℕ) -- capacity of each tank
  (h1 : 300 < c) -- first tank is filled with 300 liters, thus c > 300
  (h2 : 450 < c) -- second tank is filled with 450 liters, thus c > 450
  (h3 : (45 : ℝ) / 100 = (450 : ℝ) / c) -- second tank is 45% filled, thus 0.45 * c = 450
  (h4 : 300 + 450 < 2 * c) -- the two tanks have the same capacity, thus they must have enough capacity to be filled more than 750 liters
  : c - 300 + (c - 450) = 1250 :=
sorry

end tanks_fill_l123_123626


namespace marilyn_bananas_l123_123223

-- Defining the conditions
def boxes : ℕ := 8
def bananas_per_box : ℕ := 5

-- The statement that Marilyn has 40 bananas
theorem marilyn_bananas : boxes * bananas_per_box = 40 :=
by
  sorry

end marilyn_bananas_l123_123223


namespace compare_y_coordinates_l123_123563

theorem compare_y_coordinates (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁: (x₁ = -3) ∧ (y₁ = 2 * x₁ - 1)) 
  (h₂: (x₂ = -5) ∧ (y₂ = 2 * x₂ - 1)) : 
  y₁ > y₂ := 
by 
  sorry

end compare_y_coordinates_l123_123563


namespace series_sum_solution_l123_123086

noncomputable def series_sum (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a > b) (h₄ : b > c) : ℝ :=
  ∑' n : ℕ, (1 / ((n * c - (n - 1) * b) * ((n + 1) * c - n * b)))

theorem series_sum_solution (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a > b) (h₄ : b > c) :
  series_sum a b c h₀ h₁ h₂ h₃ h₄ = 1 / ((c - b) * c) := 
  sorry

end series_sum_solution_l123_123086


namespace cosine_240_l123_123361

theorem cosine_240 (h1 : Real.cos 60 = 1 / 2) : Real.cos 240 = -1 / 2 :=
by
  have h2 : Real.cos 240 = -Real.cos 60 := by sorry
  rw [h2, h1]
  exact neg_div
  sorry

end cosine_240_l123_123361


namespace circle_inscribed_radius_l123_123648

theorem circle_inscribed_radius (R α : ℝ) (hα : α < Real.pi) : 
  ∃ x : ℝ, x = R * (Real.sin (α / 4))^2 :=
sorry

end circle_inscribed_radius_l123_123648


namespace artifacts_per_wing_l123_123691

theorem artifacts_per_wing (P A w_wings p_wings a_wings : ℕ) (hp1 : w_wings = 8)
  (hp2 : A = 4 * P) (hp3 : p_wings = 3) (hp4 : (∃ L S : ℕ, L = 1 ∧ S = 12 ∧ P = 2 * S + L))
  (hp5 : a_wings = w_wings - p_wings) :
  A / a_wings = 20 :=
by
  sorry

end artifacts_per_wing_l123_123691


namespace max_area_l123_123211

theorem max_area (l w : ℝ) (h : l + 3 * w = 500) : l * w ≤ 62500 :=
by
  sorry

end max_area_l123_123211


namespace Jenny_minutes_of_sleep_l123_123594

def hours_of_sleep : ℕ := 8
def minutes_per_hour : ℕ := 60

theorem Jenny_minutes_of_sleep : hours_of_sleep * minutes_per_hour = 480 := by
  sorry

end Jenny_minutes_of_sleep_l123_123594


namespace inequality_holds_for_any_xyz_l123_123095

theorem inequality_holds_for_any_xyz (x y z : ℝ) : 
  x^4 + y^4 + z^2 + 1 ≥ 2 * x * (x * y^2 - x + z + 1) := 
by 
  sorry

end inequality_holds_for_any_xyz_l123_123095


namespace smallest_n_satisfying_equation_l123_123966

theorem smallest_n_satisfying_equation : ∃ (k : ℤ), (∃ (n : ℤ), n > 0 ∧ n % 2 = 1 ∧ (n ^ 3 + 2 * n ^ 2 = k ^ 2) ∧ ∀ m : ℤ, (m > 0 ∧ m < n ∧ m % 2 = 1) → ¬ (∃ j : ℤ, m ^ 3 + 2 * m ^ 2 = j ^ 2)) ∧ k % 2 = 1 :=
sorry

end smallest_n_satisfying_equation_l123_123966


namespace cosine_240_l123_123359

theorem cosine_240 (h1 : Real.cos 60 = 1 / 2) : Real.cos 240 = -1 / 2 :=
by
  have h2 : Real.cos 240 = -Real.cos 60 := by sorry
  rw [h2, h1]
  exact neg_div
  sorry

end cosine_240_l123_123359


namespace total_spending_l123_123159

theorem total_spending (Emma_spent : ℕ) (Elsa_spent : ℕ) (Elizabeth_spent : ℕ) : 
  Emma_spent = 58 →
  Elsa_spent = 2 * Emma_spent →
  Elizabeth_spent = 4 * Elsa_spent →
  Emma_spent + Elsa_spent + Elizabeth_spent = 638 := 
by
  intros h_Emma h_Elsa h_Elizabeth
  sorry

end total_spending_l123_123159


namespace total_money_spent_l123_123162

theorem total_money_spent (emma_spent : ℤ) (elsa_spent : ℤ) (elizabeth_spent : ℤ) 
(emma_condition : emma_spent = 58) 
(elsa_condition : elsa_spent = 2 * emma_spent) 
(elizabeth_condition : elizabeth_spent = 4 * elsa_spent) 
:
emma_spent + elsa_spent + elizabeth_spent = 638 :=
by
  rw [emma_condition, elsa_condition, elizabeth_condition]
  norm_num
  sorry

end total_money_spent_l123_123162


namespace largest_n_for_factored_polynomial_l123_123725

theorem largest_n_for_factored_polynomial :
  ∃ (n : ℤ), (∀ (A B : ℤ), 3 * A * B = 108 → n = 3 * B + A) ∧ n = 325 :=
by 
  sorry

end largest_n_for_factored_polynomial_l123_123725


namespace domain_of_f_l123_123665

noncomputable def f (x : ℝ) : ℝ := (x + 6) / Real.sqrt (x^2 - 5*x + 6)

theorem domain_of_f : 
  {x : ℝ | x^2 - 5*x + 6 > 0} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_f_l123_123665


namespace students_brought_two_plants_l123_123647

theorem students_brought_two_plants 
  (a1 a2 a3 a4 a5 : ℕ) (p1 p2 p3 p4 p5 : ℕ)
  (h1 : a1 + a2 + a3 + a4 + a5 = 20)
  (h2 : a1 * p1 + a2 * p2 + a3 * p3 + a4 * p4 + a5 * p5 = 30)
  (h3 : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
        p3 ≠ p4 ∧ p3 ≠ p5 ∧ p4 ≠ p5)
  : ∃ a : ℕ, a = 1 ∧ (∃ i : ℕ, p1 = 2 ∨ p2 = 2 ∨ p3 = 2 ∨ p4 = 2 ∨ p5 = 2) :=
sorry

end students_brought_two_plants_l123_123647


namespace lloyd_earnings_l123_123257

theorem lloyd_earnings:
  let regular_hours := 7.5
  let regular_rate := 4.50
  let overtime_multiplier := 2.0
  let hours_worked := 10.5
  let overtime_hours := hours_worked - regular_hours
  let overtime_rate := overtime_multiplier * regular_rate
  let regular_pay := regular_hours * regular_rate
  let overtime_pay := overtime_hours * overtime_rate
  let total_earnings := regular_pay + overtime_pay
  total_earnings = 60.75 :=
by
  sorry

end lloyd_earnings_l123_123257


namespace measure_of_angle_D_l123_123399

def angle_A := 95 -- Defined in step b)
def angle_B := angle_A
def angle_C := angle_A
def angle_D := angle_A + 50
def angle_E := angle_D
def angle_F := angle_D

theorem measure_of_angle_D (x : ℕ) (y : ℕ) :
  (angle_A = x) ∧ (angle_D = y) ∧ (y = x + 50) ∧ (3 * x + 3 * y = 720) → y = 145 :=
by
  intros
  sorry

end measure_of_angle_D_l123_123399


namespace union_prob_inconsistency_l123_123047

noncomputable def p_a : ℚ := 2/15
noncomputable def p_b : ℚ := 4/15
noncomputable def p_b_given_a : ℚ := 3

theorem union_prob_inconsistency : p_a + p_b - p_b_given_a * p_a = 0 → false := by
  sorry

end union_prob_inconsistency_l123_123047


namespace Jackie_apples_count_l123_123824

variable (Adam_apples Jackie_apples : ℕ)

-- Conditions
axiom Adam_has_14_apples : Adam_apples = 14
axiom Adam_has_5_more_than_Jackie : Adam_apples = Jackie_apples + 5

-- Theorem to prove
theorem Jackie_apples_count : Jackie_apples = 9 := by
  -- Use the conditions to derive the answer
  sorry

end Jackie_apples_count_l123_123824


namespace yield_percentage_is_correct_l123_123259

-- Defining the conditions and question
def market_value := 70
def face_value := 100
def dividend_percentage := 7
def annual_dividend := (dividend_percentage * face_value) / 100

-- Lean statement to prove the yield percentage
theorem yield_percentage_is_correct (market_value: ℕ) (annual_dividend: ℝ) : 
  ((annual_dividend / market_value) * 100) = 10 := 
by
  -- conditions from a)
  have market_value := 70
  have face_value := 100
  have dividend_percentage := 7
  have annual_dividend := (dividend_percentage * face_value) / 100
  
  -- proof will go here
  sorry

end yield_percentage_is_correct_l123_123259


namespace pipe_A_filling_time_l123_123895

theorem pipe_A_filling_time :
  ∃ (t : ℚ), 
  (∀ (t : ℚ), (t > 0) → (1 / t + 5 / t = 1 / 4.571428571428571) ↔ t = 27.42857142857143) := 
by
  -- definition of t and the corresponding conditions are directly derived from the problem
  sorry

end pipe_A_filling_time_l123_123895


namespace cos_240_is_neg_half_l123_123343

noncomputable def cos_240 : ℝ := Real.cos (240 * Real.pi / 180)

theorem cos_240_is_neg_half : cos_240 = -1/2 := by
  sorry

end cos_240_is_neg_half_l123_123343


namespace cos_240_eq_neg_half_l123_123308

theorem cos_240_eq_neg_half : ∀ (deg: ℝ), 
  deg = 240 → 
  (∀ a b : ℝ, 240 = a + b → a = 180 → b = 60 → 
    cos 240 = cos a * cos b - sin a * sin b) → 
  cos 180 = -1 →
  sin 180 = 0 →
  cos 60 = 1 / 2 →
  cos 240 = -1 / 2 :=
by 
  intros deg h_deg h_sum h_cos_180 h_sin_180 h_cos_60
  rw h_deg at h_sum
  have h_cos_identity := h_sum 180 60 rfl rfl rfl
  simp [h_cos_180, h_sin_180, h_cos_60] at h_cos_identity
  exact h_cos_identity

end cos_240_eq_neg_half_l123_123308


namespace group_population_l123_123492

theorem group_population :
  ∀ (men women children : ℕ),
  (men = 2 * women) →
  (women = 3 * children) →
  (children = 30) →
  (men + women + children = 300) :=
by
  intros men women children h_men h_women h_children
  sorry

end group_population_l123_123492


namespace stickers_given_to_sister_l123_123421

variable (initial bought birthday used left given : ℕ)

theorem stickers_given_to_sister :
  (initial = 20) →
  (bought = 12) →
  (birthday = 20) →
  (used = 8) →
  (left = 39) →
  (given = (initial + bought + birthday - used - left)) →
  given = 5 := by
  intros
  sorry

end stickers_given_to_sister_l123_123421


namespace roots_squared_sum_eq_13_l123_123768

/-- Let p and q be the roots of the quadratic equation x^2 - 5x + 6 = 0. Then the value of p^2 + q^2 is 13. -/
theorem roots_squared_sum_eq_13 (p q : ℝ) (h₁ : p + q = 5) (h₂ : p * q = 6) : p^2 + q^2 = 13 :=
by
  sorry

end roots_squared_sum_eq_13_l123_123768


namespace average_speed_l123_123716

theorem average_speed (initial final time : ℕ) (h_initial : initial = 2002) (h_final : final = 2332) (h_time : time = 11) : 
  (final - initial) / time = 30 := by
  sorry

end average_speed_l123_123716


namespace two_digit_prime_sum_digits_10_count_l123_123543

-- Two-digit number definition
def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Sum of digits equals 10
def digits_sum_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

-- Prime number definition
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Final Lean 4 statement
theorem two_digit_prime_sum_digits_10_count :
  { n : ℕ | two_digit n ∧ digits_sum_10 n ∧ is_prime n }.to_finset.card = 3 :=
by
  sorry

end two_digit_prime_sum_digits_10_count_l123_123543


namespace two_digit_prime_sum_to_ten_count_l123_123537

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Main statement to prove
theorem two_digit_prime_sum_to_ten_count :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
sorry

end two_digit_prime_sum_to_ten_count_l123_123537


namespace circle_radius_order_l123_123962

theorem circle_radius_order 
  (rA: ℝ) (rA_condition: rA = 2)
  (CB: ℝ) (CB_condition: CB = 10 * Real.pi)
  (AC: ℝ) (AC_condition: AC = 16 * Real.pi) :
  let rB := CB / (2 * Real.pi)
  let rC := Real.sqrt (AC / Real.pi)
  rA < rC ∧ rC < rB :=
by 
  sorry

end circle_radius_order_l123_123962


namespace parallel_lines_distance_sum_l123_123786

theorem parallel_lines_distance_sum (b c : ℝ) 
  (h1 : ∃ k : ℝ, 6 = 3 * k ∧ b = 4 * k) 
  (h2 : (abs ((c / 2) - 5) / (Real.sqrt (3^2 + 4^2))) = 3) : 
  b + c = 48 ∨ b + c = -12 := by
  sorry

end parallel_lines_distance_sum_l123_123786


namespace constant_term_expansion_l123_123582

theorem constant_term_expansion (a : ℝ) (h : (2 + a * x) * (1 + 1/x) ^ 5 = (2 + 5 * a)) : 2 + 5 * a = 12 → a = 2 :=
by
  intro h_eq
  have h_sum : 2 + 5 * a = 12 := h_eq
  sorry

end constant_term_expansion_l123_123582


namespace exists_solution_interval_inequality_l123_123164

theorem exists_solution_interval_inequality :
  ∀ x : ℝ, (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) ↔ 
  (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) > 1 / 5) := 
by
  sorry

end exists_solution_interval_inequality_l123_123164


namespace unique_k_solves_eq_l123_123790

theorem unique_k_solves_eq (k : ℕ) (hpos_k : k > 0) :
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = k * a * b) ↔ k = 2 :=
by
  sorry

end unique_k_solves_eq_l123_123790


namespace cos_240_degree_l123_123295

theorem cos_240_degree : Real.cos (240 * Real.pi / 180) = -1/2 :=
by
  sorry

end cos_240_degree_l123_123295


namespace roots_polynomial_l123_123415

noncomputable def roots_are (a b c : ℝ) : Prop :=
  a^3 - 18 * a^2 + 20 * a - 8 = 0 ∧ b^3 - 18 * b^2 + 20 * b - 8 = 0 ∧ c^3 - 18 * c^2 + 20 * c - 8 = 0

theorem roots_polynomial (a b c : ℝ) (h : roots_are a b c) : 
  (2 + a) * (2 + b) * (2 + c) = 128 :=
by
  sorry

end roots_polynomial_l123_123415


namespace gcd_a_b_eq_one_l123_123465

def a : ℕ := 123^2 + 235^2 + 347^2
def b : ℕ := 122^2 + 234^2 + 348^2

theorem gcd_a_b_eq_one : Nat.gcd a b = 1 :=
by
  sorry

end gcd_a_b_eq_one_l123_123465


namespace complement_of_M_in_U_l123_123172

def U := Set.univ (α := ℝ)
def M := {x : ℝ | x < -2 ∨ x > 8}
def compl_M := {x : ℝ | -2 ≤ x ∧ x ≤ 8}

theorem complement_of_M_in_U : compl_M = U \ M :=
by
  sorry

end complement_of_M_in_U_l123_123172


namespace cos_240_eq_neg_half_l123_123291

theorem cos_240_eq_neg_half (h1 : Real.cos (180 * Real.pi / 180) = -1)
                            (h2 : Real.sin (180 * Real.pi / 180) = 0)
                            (h3 : Real.cos (60 * Real.pi / 180) = 1 / 2) :
  Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_240_eq_neg_half_l123_123291


namespace two_digit_primes_with_digit_sum_10_count_l123_123558

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end two_digit_primes_with_digit_sum_10_count_l123_123558


namespace smallest_value_inequality_l123_123412

variable (a b c d : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)

theorem smallest_value_inequality :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 :=
sorry

end smallest_value_inequality_l123_123412


namespace friend_initial_marbles_l123_123436

theorem friend_initial_marbles (total_games : ℕ) (bids_per_game : ℕ) (games_lost : ℕ) (final_marbles : ℕ) 
  (h_games_eq : total_games = 9) (h_bids_eq : bids_per_game = 10) 
  (h_lost_eq : games_lost = 1) (h_final_eq : final_marbles = 90) : 
  ∃ initial_marbles : ℕ, initial_marbles = 20 := by
  sorry

end friend_initial_marbles_l123_123436


namespace christine_siri_total_money_l123_123016

-- Define the conditions
def christine_has_more_than_siri : ℝ := 20 -- Christine has 20 rs more than Siri
def christine_amount : ℝ := 20.5 -- Christine has 20.5 rs

-- Define the proof problem
theorem christine_siri_total_money :
  (∃ (siri_amount : ℝ), christine_amount = siri_amount + christine_has_more_than_siri) →
  ∃ total : ℝ, total = christine_amount + (christine_amount - christine_has_more_than_siri) ∧ total = 21 :=
by sorry

end christine_siri_total_money_l123_123016


namespace unique_solution_implies_a_eq_pm_b_l123_123370

theorem unique_solution_implies_a_eq_pm_b 
  (a b : ℝ) 
  (h_nonzero_a : a ≠ 0) 
  (h_nonzero_b : b ≠ 0) 
  (h_unique_solution : ∃! x : ℝ, a * (x - a) ^ 2 + b * (x - b) ^ 2 = 0) : 
  a = b ∨ a = -b :=
sorry

end unique_solution_implies_a_eq_pm_b_l123_123370


namespace rectangle_sides_l123_123847

theorem rectangle_sides (a b : ℝ) (h₁ : a < b) (h₂ : a * b = 2 * (a + b)) : a < 4 ∧ b > 4 :=
sorry

end rectangle_sides_l123_123847


namespace roots_sum_of_squares_l123_123219

noncomputable def proof_problem (p q r : ℝ) : Prop :=
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 598

theorem roots_sum_of_squares
  (p q r : ℝ)
  (h1 : p + q + r = 18)
  (h2 : p * q + q * r + r * p = 25)
  (h3 : p * q * r = 6) :
  proof_problem p q r :=
by {
  -- Solution steps here (omitted; not needed for the task)
  sorry
}

end roots_sum_of_squares_l123_123219


namespace correct_answer_l123_123773

def mary_initial_cards : ℝ := 18.0
def mary_bought_cards : ℝ := 40.0
def mary_left_cards : ℝ := 32.0
def mary_promised_cards (initial_cards : ℝ) (bought_cards : ℝ) (left_cards : ℝ) : ℝ :=
  initial_cards + bought_cards - left_cards

theorem correct_answer :
  mary_promised_cards mary_initial_cards mary_bought_cards mary_left_cards = 26.0 := by
  sorry

end correct_answer_l123_123773


namespace half_abs_diff_of_squares_l123_123655

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 20) (h2 : b = 15) :
  (1/2 : ℚ) * |(a:ℚ)^2 - (b:ℚ)^2| = 87.5 := by
  sorry

end half_abs_diff_of_squares_l123_123655


namespace find_original_number_l123_123993

def original_number_divide_multiply (x : ℝ) : Prop :=
  (x / 12) * 24 = x + 36

theorem find_original_number (x : ℝ) (h : original_number_divide_multiply x) : x = 36 :=
by
  sorry

end find_original_number_l123_123993


namespace minimize_average_comprehensive_cost_l123_123144

theorem minimize_average_comprehensive_cost :
  ∀ (f : ℕ → ℝ), (∀ (x : ℕ), x ≥ 10 → f x = 560 + 48 * x + 10800 / x) →
  ∃ x : ℕ, x = 15 ∧ ( ∀ y : ℕ, y ≥ 10 → f y ≥ f 15 ) :=
by
  sorry

end minimize_average_comprehensive_cost_l123_123144


namespace cone_volume_from_half_sector_l123_123500

theorem cone_volume_from_half_sector (r l : ℝ) (h : ℝ) 
    (h_r : r = 3) (h_l : l = 6) (h_h : h = 3 * Real.sqrt 3) : 
    (1 / 3) * Real.pi * r^2 * h = 9 * Real.pi * Real.sqrt 3 := 
by
  -- Sorry to skip the proof
  sorry

end cone_volume_from_half_sector_l123_123500


namespace find_line_eq_l123_123368

open Real

/-- Lean statement for the given proof problem -/
theorem find_line_eq (x y : ℝ) (l : ℝ → ℝ → Prop) :
  (∀ x y : ℝ, l x y ↔ 3 * x + 2 * y - 5 = 0) →
  (∀ x y : ℝ, (λ x y, 3 * x - 2 * y - 1 = 0) x y ↔ l x y) →
  (∃ k : ℝ , ∀ x y : ℝ, (λ x y, 2 * x + y + k = 0) x y ↔ l x y) → 
  ∃ k : ℝ, k = -3 :=
by
  sorry

end find_line_eq_l123_123368


namespace drunk_drivers_count_l123_123756

theorem drunk_drivers_count (D S : ℕ) (h1 : S = 7 * D - 3) (h2 : D + S = 45) : D = 6 :=
by
  sorry

end drunk_drivers_count_l123_123756


namespace correct_avg_and_mode_l123_123442

-- Define the conditions and correct answers
def avgIncorrect : ℚ := 13.5
def medianIncorrect : ℚ := 12
def modeCorrect : ℚ := 16
def totalNumbers : ℕ := 25
def incorrectNums : List ℚ := [33.5, 47.75, 58.5, 19/2]
def correctNums : List ℚ := [43.5, 56.25, 68.5, 21/2]

noncomputable def correctSum : ℚ := (avgIncorrect * totalNumbers) + (correctNums.sum - incorrectNums.sum)
noncomputable def correctAvg : ℚ := correctSum / totalNumbers

theorem correct_avg_and_mode :
  correctAvg = 367 / 25 ∧ modeCorrect = 16 :=
by
  sorry

end correct_avg_and_mode_l123_123442


namespace cos_240_eq_neg_half_l123_123339

/-- Prove that cos 240 degrees equals -1/2 --/
theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1/2 :=
  sorry

end cos_240_eq_neg_half_l123_123339


namespace central_angle_of_sector_l123_123796

noncomputable def central_angle (radius perimeter: ℝ) : ℝ :=
  ((perimeter - 2 * radius) / (2 * Real.pi * radius)) * 360

theorem central_angle_of_sector :
  central_angle 28 144 = 180.21 :=
by
  simp [central_angle]
  sorry

end central_angle_of_sector_l123_123796


namespace cos_240_eq_neg_half_l123_123336

/-- Prove that cos 240 degrees equals -1/2 --/
theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1/2 :=
  sorry

end cos_240_eq_neg_half_l123_123336


namespace cosine_240_l123_123360

theorem cosine_240 (h1 : Real.cos 60 = 1 / 2) : Real.cos 240 = -1 / 2 :=
by
  have h2 : Real.cos 240 = -Real.cos 60 := by sorry
  rw [h2, h1]
  exact neg_div
  sorry

end cosine_240_l123_123360


namespace at_least_5_limit_ups_needed_l123_123950

-- Let's denote the necessary conditions in Lean
variable (a : ℝ) -- the buying price of stock A

-- Initial price after 4 consecutive limit downs
def price_after_limit_downs (a : ℝ) : ℝ := a * (1 - 0.1) ^ 4

-- Condition of no loss after certain limit ups
def no_loss_after_limit_ups (a : ℝ) (x : ℕ) : Prop := 
  price_after_limit_downs a * (1 + 0.1)^x ≥ a
  
theorem at_least_5_limit_ups_needed (a : ℝ) : ∃ x, no_loss_after_limit_ups a x ∧ x ≥ 5 :=
by
  -- We are required to find such x and prove the condition, which has been shown in the mathematical solution
  sorry

end at_least_5_limit_ups_needed_l123_123950


namespace gcd_problem_l123_123467

-- Define the variables according to the conditions
def m : ℤ := 123^2 + 235^2 + 347^2
def n : ℤ := 122^2 + 234^2 + 348^2

-- Lean statement for the proof problem
theorem gcd_problem : Int.gcd m n = 1 := sorry

end gcd_problem_l123_123467


namespace radius_wheel_l123_123911

noncomputable def pi : ℝ := 3.14159

theorem radius_wheel (D : ℝ) (N : ℕ) (r : ℝ) (h1 : D = 760.57) (h2 : N = 500) :
  r = (D / N) / (2 * pi) :=
sorry

end radius_wheel_l123_123911


namespace verification_equation_3_conjecture_general_equation_l123_123427

theorem verification_equation_3 : 
  4 * Real.sqrt (4 / 15) = Real.sqrt (4 * (4 / 15)) :=
sorry

theorem conjecture :
  Real.sqrt (5 * (5 / 24)) = 5 * Real.sqrt (5 / 24) :=
sorry

theorem general_equation (n : ℕ) (h : 2 ≤ n) :
  n * Real.sqrt (n / (n^2 - 1)) = Real.sqrt (n + n / (n^2 - 1)) :=
sorry

end verification_equation_3_conjecture_general_equation_l123_123427


namespace probability_age_between_30_and_40_l123_123758

-- Assume total number of people in the group is 200
def total_people : ℕ := 200

-- Assume 80 people have an age of more than 40 years
def people_age_more_than_40 : ℕ := 80

-- Assume 70 people have an age between 30 and 40 years
def people_age_between_30_and_40 : ℕ := 70

-- Assume 30 people have an age between 20 and 30 years
def people_age_between_20_and_30 : ℕ := 30

-- Assume 20 people have an age of less than 20 years
def people_age_less_than_20 : ℕ := 20

-- The proof problem statement
theorem probability_age_between_30_and_40 :
  (people_age_between_30_and_40 : ℚ) / (total_people : ℚ) = 7 / 20 :=
by
  sorry

end probability_age_between_30_and_40_l123_123758


namespace exponential_comparisons_l123_123175

open Real

noncomputable def a : ℝ := 5 ^ (log 3.4 / log 2)
noncomputable def b : ℝ := 5 ^ (log 3.6 / (log 4))
noncomputable def c : ℝ := 5 ^ (log (10 / 3))

theorem exponential_comparisons :
  a > c ∧ c > b := by
  sorry

end exponential_comparisons_l123_123175


namespace goldfish_remaining_to_catch_l123_123890

-- Define the number of total goldfish in the aquarium
def total_goldfish : ℕ := 100

-- Define the number of goldfish Maggie is allowed to take home (half of total goldfish)
def allowed_to_take_home := total_goldfish / 2

-- Define the number of goldfish Maggie caught (3/5 of allowed_to_take_home)
def caught := (3 * allowed_to_take_home) / 5

-- Prove the number of goldfish Maggie remains with to catch
theorem goldfish_remaining_to_catch : allowed_to_take_home - caught = 20 := by
  -- Sorry is used to skip the proof
  sorry

end goldfish_remaining_to_catch_l123_123890


namespace gcd_117_182_evaluate_polynomial_l123_123130

-- Problem 1: Prove that GCD of 117 and 182 is 13
theorem gcd_117_182 : Int.gcd 117 182 = 13 := 
by
  sorry

-- Problem 2: Prove that evaluating the polynomial at x = -1 results in 12
noncomputable def f : ℤ → ℤ := λ x => 1 - 9 * x + 8 * x^2 - 4 * x^4 + 5 * x^5 + 3 * x^6

theorem evaluate_polynomial : f (-1) = 12 := 
by
  sorry

end gcd_117_182_evaluate_polynomial_l123_123130


namespace num_two_digit_primes_with_digit_sum_10_l123_123553

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ // 10 ≤ n ∧ n < 100 ∧ (n / 10 + n % 10 = 10) ∧ is_prime n}.to_finset.card = 3 :=
by sorry

end num_two_digit_primes_with_digit_sum_10_l123_123553


namespace length_breadth_difference_l123_123622

theorem length_breadth_difference (b l : ℕ) (h1 : b = 5) (h2 : l * b = 15 * b) : l - b = 10 :=
by
  sorry

end length_breadth_difference_l123_123622


namespace roots_relationship_l123_123559

theorem roots_relationship (a b c : ℝ) (α β : ℝ) 
  (h_eq : a * α^2 + b * α + c = 0)
  (h_triple : β = 3 * α)
  (h_vieta1 : α + β = -b / a)
  (h_vieta2 : α * β = c / a) : 
  3 * b^2 = 16 * a * c :=
sorry

end roots_relationship_l123_123559


namespace geometric_sequence_b_value_l123_123365

theorem geometric_sequence_b_value (b : ℝ) 
  (h1 : ∃ r : ℝ, 30 * r = b ∧ b * r = 9 / 4)
  (h2 : b > 0) : b = 3 * Real.sqrt 30 :=
by
  sorry

end geometric_sequence_b_value_l123_123365


namespace tabletop_qualification_l123_123006

theorem tabletop_qualification (length width diagonal : ℕ) :
  length = 60 → width = 32 → diagonal = 68 → (diagonal * diagonal = length * length + width * width) :=
by
  intros
  sorry

end tabletop_qualification_l123_123006


namespace archibald_percentage_wins_l123_123702

def archibald_wins : ℕ := 12
def brother_wins : ℕ := 18
def total_games_played : ℕ := archibald_wins + brother_wins

def percentage_archibald_wins : ℚ := (archibald_wins : ℚ) / (total_games_played : ℚ) * 100

theorem archibald_percentage_wins : percentage_archibald_wins = 40 := by
  sorry

end archibald_percentage_wins_l123_123702


namespace factorization_identity_l123_123165

variable (a b : ℝ)

theorem factorization_identity : 3 * a^2 + 6 * a * b = 3 * a * (a + 2 * b) := by
  sorry

end factorization_identity_l123_123165


namespace cos_240_eq_negative_half_l123_123329

theorem cos_240_eq_negative_half : Real.cos (240 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_240_eq_negative_half_l123_123329


namespace solution_eq_c_l123_123187

variables (x : ℝ) (a : ℝ) 

def p := ∃ x0 : ℝ, (0 < x0) ∧ (3^x0 + x0 = 2016)
def q := ∃ a : ℝ, (0 < a) ∧ (∀ x : ℝ, (|x| - a * x) = (|(x)| - a * (-x)))

theorem solution_eq_c : p ∧ ¬q :=
by {
  sorry -- proof placeholder
}

end solution_eq_c_l123_123187


namespace Danielle_has_6_rooms_l123_123022

axiom Danielle_rooms : ℕ
axiom Heidi_rooms : ℕ
axiom Grant_rooms : ℕ

axiom Heidi_has_3_times_Danielle : Heidi_rooms = 3 * Danielle_rooms
axiom Grant_has_1_9_Heidi : Grant_rooms = Heidi_rooms / 9
axiom Grant_has_2_rooms : Grant_rooms = 2

theorem Danielle_has_6_rooms : Danielle_rooms = 6 :=
by {
  -- proof steps would go here
  sorry
}

end Danielle_has_6_rooms_l123_123022


namespace sufficient_condition_for_m_ge_9_l123_123377

theorem sufficient_condition_for_m_ge_9
  (x m : ℝ)
  (p : |x - 4| ≤ 6)
  (q : x ≤ 1 + m)
  (h_sufficient : ∀ x, |x - 4| ≤ 6 → x ≤ 1 + m)
  (h_not_necessary : ∃ x, ¬(|x - 4| ≤ 6) ∧ x ≤ 1 + m) :
  m ≥ 9 := 
sorry

end sufficient_condition_for_m_ge_9_l123_123377


namespace limit_seq_converges_to_one_l123_123014

noncomputable def limit_seq (n : ℕ) : ℝ := 
  (real.sqrt (n^4 + 2) + real.sqrt (n - 2)) / (real.sqrt (n^4 + 2) + real.sqrt (n - 2))

theorem limit_seq_converges_to_one : 
  tendsto (λ n : ℕ, limit_seq n) at_top (𝓝 (1 : ℝ)) :=
begin
  sorry
end

end limit_seq_converges_to_one_l123_123014


namespace solve_system_eqns_l123_123902

theorem solve_system_eqns (x y z : ℝ) :
  x^2 - 23 * y + 66 * z + 612 = 0 ∧
  y^2 + 62 * x - 20 * z + 296 = 0 ∧
  z^2 - 22 * x + 67 * y + 505 = 0 ↔
  x = -20 ∧ y = -22 ∧ z = -23 :=
by
  sorry

end solve_system_eqns_l123_123902


namespace speed_of_stream_l123_123470

-- Definitions based on the conditions
def upstream_speed (c v : ℝ) : Prop := c - v = 4
def downstream_speed (c v : ℝ) : Prop := c + v = 12

-- Main theorem to prove
theorem speed_of_stream (c v : ℝ) (h1 : upstream_speed c v) (h2 : downstream_speed c v) : v = 4 :=
by
  sorry

end speed_of_stream_l123_123470


namespace cos_240_degree_l123_123298

theorem cos_240_degree : Real.cos (240 * Real.pi / 180) = -1/2 :=
by
  sorry

end cos_240_degree_l123_123298


namespace total_assignment_schemes_one_school_no_teachers_one_school_two_teachers_two_schools_no_teachers_l123_123454

section assignment_problems

-- Define the number of teachers and schools
def num_teachers : ℕ := 4
def num_schools : ℕ := 4

-- Problem 1: Total assignment schemes without any restrictions
theorem total_assignment_schemes : (num_schools ^ num_teachers) = 256 := by
  sorry

-- Problem 2: One school is not assigned any teachers
theorem one_school_no_teachers :
  (finset.card (finset.univ : finset (finset.univ.1 \ {1}.val)) * ((nat.choose num_teachers 2) * (3^3))) = 144 := by
  sorry

-- Problem 3: One specific school is assigned 2 teachers
theorem one_school_two_teachers :
  (nat.choose num_teachers 2 * 3^2) = 54 := by
  sorry

-- Problem 4: Exactly two schools are not assigned any teachers
theorem two_schools_no_teachers :
  ( nat.choose num_schools 2 * (nat.choose num_teachers 2 / nat.choose 2 2 + nat.choose num_teachers 1) ) * nat.choose 2 2 = 84 := by
  sorry

end assignment_problems

end total_assignment_schemes_one_school_no_teachers_one_school_two_teachers_two_schools_no_teachers_l123_123454


namespace perimeter_shaded_region_l123_123068

-- Definitions based on conditions
def circle_radius : ℝ := 10
def central_angle : ℝ := 300

-- Statement: Perimeter of the shaded region
theorem perimeter_shaded_region 
  : (10 : ℝ) + (10 : ℝ) + ((5 / 6) * (2 * Real.pi * 10)) = (20 : ℝ) + (50 / 3) * Real.pi :=
by
  sorry

end perimeter_shaded_region_l123_123068


namespace maximal_n_for_quadratic_factorization_l123_123724

theorem maximal_n_for_quadratic_factorization :
  ∃ n, n = 325 ∧ (∃ A B : ℤ, A * B = 108 ∧ n = 3 * B + A) :=
by
  use 325
  use 1, 108
  constructor
  · rfl
  constructor
  · norm_num
  · norm_num
  sorry

end maximal_n_for_quadratic_factorization_l123_123724


namespace cos_240_eq_neg_half_l123_123313

theorem cos_240_eq_neg_half : ∀ (deg: ℝ), 
  deg = 240 → 
  (∀ a b : ℝ, 240 = a + b → a = 180 → b = 60 → 
    cos 240 = cos a * cos b - sin a * sin b) → 
  cos 180 = -1 →
  sin 180 = 0 →
  cos 60 = 1 / 2 →
  cos 240 = -1 / 2 :=
by 
  intros deg h_deg h_sum h_cos_180 h_sin_180 h_cos_60
  rw h_deg at h_sum
  have h_cos_identity := h_sum 180 60 rfl rfl rfl
  simp [h_cos_180, h_sin_180, h_cos_60] at h_cos_identity
  exact h_cos_identity

end cos_240_eq_neg_half_l123_123313


namespace probability_of_non_yellow_is_19_over_28_l123_123487

def total_beans : ℕ := 4 + 5 + 9 + 10

def non_yellow_beans : ℕ := 4 + 5 + 10

def probability_not_yellow : ℚ := non_yellow_beans / total_beans

theorem probability_of_non_yellow_is_19_over_28 :
  probability_not_yellow = 19 / 28 := by
  sorry

end probability_of_non_yellow_is_19_over_28_l123_123487


namespace total_people_l123_123497

theorem total_people (M W C : ℕ) (h1 : M = 2 * W) (h2 : W = 3 * C) (h3 : C = 30) : M + W + C = 300 :=
by
  sorry

end total_people_l123_123497


namespace total_rent_payment_l123_123420

def weekly_rent : ℕ := 388
def number_of_weeks : ℕ := 1359

theorem total_rent_payment : weekly_rent * number_of_weeks = 526692 := 
  by 
  sorry

end total_rent_payment_l123_123420


namespace relationship_between_coefficients_l123_123837

theorem relationship_between_coefficients (a b c : ℚ) 
  (h : ∃ (α β : ℚ), β = 3 * α ∧ (α + β = -b / a) ∧ (α * β = c / a)) : 
  3 * b ^ 2 = 16 * a * c :=
sorry

end relationship_between_coefficients_l123_123837


namespace number_of_newspapers_l123_123213

theorem number_of_newspapers (total_reading_materials magazines_sold: ℕ) (h_total: total_reading_materials = 700) (h_magazines: magazines_sold = 425) : 
  ∃ newspapers_sold : ℕ, newspapers_sold + magazines_sold = total_reading_materials ∧ newspapers_sold = 275 :=
by
  sorry

end number_of_newspapers_l123_123213


namespace cos_240_eq_neg_half_l123_123286

theorem cos_240_eq_neg_half (h1 : Real.cos (180 * Real.pi / 180) = -1)
                            (h2 : Real.sin (180 * Real.pi / 180) = 0)
                            (h3 : Real.cos (60 * Real.pi / 180) = 1 / 2) :
  Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_240_eq_neg_half_l123_123286


namespace books_read_in_eight_hours_l123_123898

noncomputable def pages_per_hour : ℕ := 120
noncomputable def pages_per_book : ℕ := 360
noncomputable def total_reading_time : ℕ := 8

theorem books_read_in_eight_hours (h1 : pages_per_hour = 120) 
                                  (h2 : pages_per_book = 360) 
                                  (h3 : total_reading_time = 8) : 
                                  total_reading_time * pages_per_hour / pages_per_book = 2 := 
by sorry

end books_read_in_eight_hours_l123_123898


namespace slope_of_line_l123_123109

-- Defining the parametric equations of the line
def parametric_x (t : ℝ) : ℝ := 3 + 4 * t
def parametric_y (t : ℝ) : ℝ := 4 - 5 * t

-- Stating the problem in Lean: asserting the slope of the line
theorem slope_of_line : 
  (∃ (m : ℝ), ∀ t : ℝ, parametric_y t = m * parametric_x t + (4 - 3 * m)) 
  → (∃ m : ℝ, m = -5 / 4) :=
  by sorry

end slope_of_line_l123_123109


namespace inequality_proof_l123_123379

variable (a b : ℝ)

theorem inequality_proof (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 4) : 
  (1 / (a^2 + b^2) ≤ 1 / 8) :=
by
  sorry

end inequality_proof_l123_123379


namespace math_problem_l123_123472

theorem math_problem : 2357 + 3572 + 5723 + 2 * 7235 = 26122 :=
  by sorry

end math_problem_l123_123472


namespace xyz_value_l123_123182

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 40)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) + 2 * x * y * z = 20) 
  : x * y * z = 20 := 
by
  sorry

end xyz_value_l123_123182


namespace jasmine_total_cost_l123_123407

noncomputable def total_cost_jasmine
  (coffee_beans_amount : ℕ)
  (milk_amount : ℕ)
  (coffee_beans_cost : ℝ)
  (milk_cost : ℝ)
  (discount_combined : ℝ)
  (additional_discount_milk : ℝ)
  (tax_rate : ℝ) : ℝ :=
  let total_before_discounts := coffee_beans_amount * coffee_beans_cost + milk_amount * milk_cost
  let total_after_combined_discount := total_before_discounts - discount_combined * total_before_discounts
  let milk_cost_after_additional_discount := milk_amount * milk_cost - additional_discount_milk * (milk_amount * milk_cost)
  let total_after_all_discounts := coffee_beans_amount * coffee_beans_cost + milk_cost_after_additional_discount
  let tax := tax_rate * total_after_all_discounts
  total_after_all_discounts + tax

theorem jasmine_total_cost :
  total_cost_jasmine 4 2 2.50 3.50 0.10 0.05 0.08 = 17.98 :=
by
  unfold total_cost_jasmine
  sorry

end jasmine_total_cost_l123_123407


namespace intersection_sum_x_coordinates_mod_17_l123_123607

theorem intersection_sum_x_coordinates_mod_17 :
  ∃ x : ℤ, (∃ y₁ y₂ : ℤ, (y₁ ≡ 7 * x + 3 [ZMOD 17]) ∧ (y₂ ≡ 13 * x + 4 [ZMOD 17]))
       ∧ x ≡ 14 [ZMOD 17]  :=
by
  sorry

end intersection_sum_x_coordinates_mod_17_l123_123607


namespace cos_240_eq_neg_half_l123_123334

/-- Prove that cos 240 degrees equals -1/2 --/
theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1/2 :=
  sorry

end cos_240_eq_neg_half_l123_123334


namespace total_liters_needed_to_fill_two_tanks_l123_123628

theorem total_liters_needed_to_fill_two_tanks (capacity : ℕ) (liters_first_tank : ℕ) (liters_second_tank : ℕ) (percent_filled : ℕ) :
  liters_first_tank = 300 → 
  liters_second_tank = 450 → 
  percent_filled = 45 → 
  capacity = (liters_second_tank * 100) / percent_filled → 
  1000 - 300 = 700 → 
  1000 - 450 = 550 → 
  700 + 550 = 1250 :=
by sorry

end total_liters_needed_to_fill_two_tanks_l123_123628


namespace relationship_y1_y2_l123_123395

theorem relationship_y1_y2 (y1 y2 : ℝ) 
  (h1 : y1 = 3 / -1) 
  (h2 : y2 = 3 / -3) : 
  y1 < y2 :=
by
  sorry

end relationship_y1_y2_l123_123395


namespace calculate_P_AB_l123_123044

section Probability
-- Define the given probabilities
variables (P_B_given_A : ℚ) (P_A : ℚ)
-- Given conditions
def given_conditions := P_B_given_A = 3/10 ∧ P_A = 1/5

-- Prove that P(AB) = 3/50
theorem calculate_P_AB (h : given_conditions P_B_given_A P_A) : (P_A * P_B_given_A) = 3/50 :=
by
  rcases h with ⟨h1, h2⟩
  simp [h1, h2]
  -- Here we would include the steps leading to the conclusion; this part just states the theorem
  sorry

end Probability

end calculate_P_AB_l123_123044


namespace angie_and_diego_probability_l123_123700

theorem angie_and_diego_probability :
  let people := ["Angie", "Bridget", "Carlos", "Diego"]
  let arrangements := {arr : list String // arr.permutations}
  let seat_adjacent := ∀ p ∈ arrangements, p.nth 0 = some "Angie" → 
                       (p.nth 1 = some "Diego" ∨ p.nth 3 = some "Diego")
  ∃ favorable_counts = 
    list.count (λ a, seat_adjacent a) arrangements in
  favorable_counts / arrangements.size = 2 / 3 := sorry

end angie_and_diego_probability_l123_123700


namespace length_of_second_train_l123_123649

theorem length_of_second_train
  (length_first_train : ℝ)
  (speed_first_train : ℝ)
  (speed_second_train : ℝ)
  (crossing_time : ℝ)
  (total_distance : ℝ)
  (relative_speed_mps : ℝ)
  (length_second_train : ℝ) :
  length_first_train = 130 ∧ 
  speed_first_train = 60 ∧
  speed_second_train = 40 ∧
  crossing_time = 10.439164866810657 ∧
  relative_speed_mps = (speed_first_train + speed_second_train) * (5/18) ∧
  total_distance = relative_speed_mps * crossing_time ∧
  length_first_train + length_second_train = total_distance →
  length_second_train = 160 :=
by
  sorry

end length_of_second_train_l123_123649


namespace first_representation_second_representation_third_representation_l123_123672

theorem first_representation :
  1 + 2 + 3 + 4 + 5 + 6 + 7 + (8 * 9) = 100 := 
by 
  sorry

theorem second_representation:
  1 + 2 + 3 + 47 + (5 * 6) + 8 + 9 = 100 :=
by
  sorry

theorem third_representation:
  1 + 2 + 3 + 4 + 5 - 6 - 7 + 8 + 92 = 100 := 
by
  sorry

end first_representation_second_representation_third_representation_l123_123672


namespace ellipse_foci_on_y_axis_l123_123050

theorem ellipse_foci_on_y_axis (m : ℝ) : 
  (∃ (x y : ℝ), (x^2 / (m + 2)) - (y^2 / (m + 1)) = 1) ↔ (-2 < m ∧ m < -3/2) := 
by
  sorry

end ellipse_foci_on_y_axis_l123_123050


namespace evaluate_expression_l123_123026

theorem evaluate_expression : 8^3 + 4 * 8^2 + 6 * 8 + 3 = 1000 := by
  sorry

end evaluate_expression_l123_123026


namespace tangent_line_count_l123_123020

noncomputable def circles_tangent_lines (r1 r2 d : ℝ) : ℕ :=
if d = |r1 - r2| then 1 else 0 -- Define the function based on the problem statement

theorem tangent_line_count :
  circles_tangent_lines 4 5 3 = 1 := 
by
  -- Placeholder for the proof, which we are skipping as per instructions
  sorry

end tangent_line_count_l123_123020


namespace exists_a_squared_congruent_neg1_iff_p_mod_4_eq_1_l123_123433

theorem exists_a_squared_congruent_neg1_iff_p_mod_4_eq_1 (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 2 = 1) :
  (∃ a : ℤ, a^2 ≡ -1 [ZMOD p]) ↔ p % 4 = 1 :=
sorry

end exists_a_squared_congruent_neg1_iff_p_mod_4_eq_1_l123_123433


namespace arithmetic_sequence_sum_l123_123875

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- The sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (h_arith : is_arithmetic_sequence a d) (h_condition : a 2 + a 10 = 16) :
  a 4 + a 8 = 16 :=
sorry

end arithmetic_sequence_sum_l123_123875


namespace coordinates_of_point_P_l123_123196

open Real

def in_fourth_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 < 0

def distance_to_x_axis (P : ℝ × ℝ) : ℝ :=
  abs P.2

def distance_to_y_axis (P : ℝ × ℝ) : ℝ :=
  abs P.1

theorem coordinates_of_point_P (P : ℝ × ℝ) 
  (h1 : in_fourth_quadrant P) 
  (h2 : distance_to_x_axis P = 1) 
  (h3 : distance_to_y_axis P = 2) : 
  P = (2, -1) :=
by
  sorry

end coordinates_of_point_P_l123_123196


namespace fixed_monthly_fee_l123_123224

theorem fixed_monthly_fee :
  ∀ (x y : ℝ), 
  x + y = 20.00 → 
  x + 2 * y = 30.00 → 
  x + 3 * y = 40.00 → 
  x = 10.00 :=
by
  intros x y H1 H2 H3
  -- Proof can be filled out here
  sorry

end fixed_monthly_fee_l123_123224


namespace coefficient_of_squared_term_l123_123009

theorem coefficient_of_squared_term (a b c : ℝ) (h_eq : 5 * a^2 + 14 * b + 5 = 0) :
  a = 5 :=
sorry

end coefficient_of_squared_term_l123_123009


namespace find_q_l123_123718

def P (q x : ℝ) : ℝ := x^4 + 2 * q * x^3 - 3 * x^2 + 2 * q * x + 1

theorem find_q (q : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 < 0 ∧ x2 < 0 ∧ P q x1 = 0 ∧ P q x2 = 0) → q < 1 / 4 :=
by
  sorry

end find_q_l123_123718


namespace time_to_pass_platform_l123_123127

-- Definitions based on the conditions
def train_length : ℕ := 1500 -- (meters)
def tree_crossing_time : ℕ := 120 -- (seconds)
def platform_length : ℕ := 500 -- (meters)

-- Define the train's speed
def train_speed := train_length / tree_crossing_time

-- Define the total distance the train needs to cover to pass the platform
def total_distance := train_length + platform_length

-- The proof statement
theorem time_to_pass_platform : 
  total_distance / train_speed = 160 :=
by sorry

end time_to_pass_platform_l123_123127


namespace cosine_240_l123_123362

theorem cosine_240 (h1 : Real.cos 60 = 1 / 2) : Real.cos 240 = -1 / 2 :=
by
  have h2 : Real.cos 240 = -Real.cos 60 := by sorry
  rw [h2, h1]
  exact neg_div
  sorry

end cosine_240_l123_123362


namespace geometric_triangle_condition_right_geometric_triangle_condition_l123_123802

-- Definitions for the geometric progression
def geometric_sequence (a b c q : ℝ) : Prop :=
  b = a * q ∧ c = a * q^2

-- Conditions for forming a triangle
def forms_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Conditions for forming a right triangle using Pythagorean theorem
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem geometric_triangle_condition (a q : ℝ) (h1 : 1 ≤ q) (h2 : q < (1 + Real.sqrt 5) / 2) :
  ∃ (b c : ℝ), geometric_sequence a b c q ∧ forms_triangle a b c := 
sorry

theorem right_geometric_triangle_condition (a q : ℝ) :
  q = Real.sqrt ((1 + Real.sqrt 5) / 2) →
  ∃ (b c : ℝ), geometric_sequence a b c q ∧ right_triangle a b c :=
sorry

end geometric_triangle_condition_right_geometric_triangle_condition_l123_123802


namespace correlational_relationships_l123_123955

-- Definitions of relationships
def learning_attitude_and_academic_performance := "The relationship between a student's learning attitude and their academic performance"
def teacher_quality_and_student_performance := "The relationship between a teacher's teaching quality and students' academic performance"
def student_height_and_academic_performance := "The relationship between a student's height and their academic performance"
def family_economic_conditions_and_performance := "The relationship between family economic conditions and students' academic performance"

-- Definition of a correlational relationship
def correlational_relationship (relation : String) : Prop :=
  relation = learning_attitude_and_academic_performance ∨
  relation = teacher_quality_and_student_performance

-- Problem statement to prove
theorem correlational_relationships :
  correlational_relationship learning_attitude_and_academic_performance ∧ 
  correlational_relationship teacher_quality_and_student_performance :=
by
  -- Placeholder to indicate the proof is omitted
  sorry

end correlational_relationships_l123_123955


namespace find_stock_rate_l123_123720

theorem find_stock_rate (annual_income : ℝ) (investment_amount : ℝ) (R : ℝ) 
  (h1 : annual_income = 2000) (h2 : investment_amount = 6800) : 
  R = 2000 / 6800 :=
by
  sorry

end find_stock_rate_l123_123720


namespace sum_first_eight_terms_l123_123043

variables (a : ℕ → ℝ) (S : ℕ → ℝ)
variables (n : ℕ) (a_3 a_6 : ℝ)

-- Conditions
def arithmetic_sequence (a_3 a_6 : ℝ) : Prop := a_3 = 20 - a_6

def sum_terms (S : ℕ → ℝ) (a_3 a_6 : ℝ) : ℝ :=
  4 * (a_3 + a_6)

-- The proof goal
theorem sum_first_eight_terms (a_3 a_6 : ℝ) (h₁ : a_3 = 20 - a_6) : S 8 = 80 :=
by
  rw [arithmetic_sequence a_3 a_6] at h₁
  sorry

end sum_first_eight_terms_l123_123043


namespace problem_statement_l123_123394

theorem problem_statement (a b c : ℝ) (h1 : a - b = 2) (h2 : b - c = -3) : a - c = -1 := 
by
  sorry

end problem_statement_l123_123394


namespace total_people_in_group_l123_123495

theorem total_people_in_group (men women children : ℕ)
  (h1 : men = 2 * women)
  (h2 : women = 3 * children)
  (h3 : children = 30) :
  men + women + children = 300 :=
by
  sorry

end total_people_in_group_l123_123495


namespace length_of_train_l123_123273

variable (L V : ℝ)

def platform_crossing (L V : ℝ) := L + 350 = V * 39
def post_crossing (L V : ℝ) := L = V * 18

theorem length_of_train (h1 : platform_crossing L V) (h2 : post_crossing L V) : L = 300 :=
by
  sorry

end length_of_train_l123_123273


namespace cos_240_degree_l123_123299

theorem cos_240_degree : Real.cos (240 * Real.pi / 180) = -1/2 :=
by
  sorry

end cos_240_degree_l123_123299


namespace num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l123_123550

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  d1 + d2 = s

def two_digit := ∀ n : ℕ, 10 ≤ n ∧ n < 100

theorem num_of_two_digit_prime_numbers_sum_of_digits_eq_10 : 
  (finset.filter (λ n, is_prime n ∧ digits_sum_to n 10 ∧ two_digit n) (finset.range 100)).card = 3 := by 
  sorry

end num_of_two_digit_prime_numbers_sum_of_digits_eq_10_l123_123550


namespace brother_highlighter_spending_l123_123744

variables {total_money : ℝ} (sharpeners notebooks erasers highlighters : ℝ)

-- Conditions
def total_given := total_money = 100
def sharpeners_cost := 2 * 5
def notebooks_cost := 4 * 5
def erasers_cost := 10 * 4
def heaven_expenditure := sharpeners_cost + notebooks_cost
def brother_remaining := total_money - (heaven_expenditure + erasers_cost)
def brother_spent_on_highlighters := brother_remaining = 30

-- Statement
theorem brother_highlighter_spending (h1 : total_given) 
    (h2 : brother_spent_on_highlighters) : brother_remaining = 30 :=
sorry

end brother_highlighter_spending_l123_123744


namespace cos_240_is_neg_half_l123_123340

noncomputable def cos_240 : ℝ := Real.cos (240 * Real.pi / 180)

theorem cos_240_is_neg_half : cos_240 = -1/2 := by
  sorry

end cos_240_is_neg_half_l123_123340


namespace set_union_l123_123419

theorem set_union :
  let M := {x | x^2 + 2 * x - 3 = 0}
  let N := {-1, 2, 3}
  M ∪ N = {-1, 1, 2, -3, 3} :=
by
  sorry

end set_union_l123_123419


namespace cos_240_eq_negative_half_l123_123328

theorem cos_240_eq_negative_half : Real.cos (240 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_240_eq_negative_half_l123_123328


namespace equal_copper_content_alloy_l123_123833

theorem equal_copper_content_alloy (a b : ℝ) :
  ∃ x : ℝ, 0 < x ∧ x < 10 ∧
  (10 - x) * a + x * b = (15 - x) * b + x * a → x = 6 :=
by
  sorry

end equal_copper_content_alloy_l123_123833


namespace correct_result_l123_123476

-- Definitions to capture the problem conditions:
def cond1 (a b : ℤ) : Prop := 5 * a^2 * b - 2 * a^2 * b = 3 * a^2 * b
def cond2 (x : ℤ) : Prop := x^6 / x^2 = x^4
def cond3 (a b : ℤ) : Prop := (a - b)^2 = a^2 - b^2

-- Proof statement to verify the correct answer
theorem correct_result (x : ℤ) : (2 * x^2)^3 = 8 * x^6 :=
  by sorry

-- Note that cond1, cond2, and cond3 are intended to capture the erroneous conditions mentioned for completeness.

end correct_result_l123_123476


namespace percentage_hate_german_l123_123870

def percentage_hate_math : ℝ := 0.01
def percentage_hate_english : ℝ := 0.02
def percentage_hate_french : ℝ := 0.01
def percentage_hate_all_four : ℝ := 0.08

theorem percentage_hate_german : (0.08 - (0.01 + 0.02 + 0.01)) = 0.04 :=
by
  -- Proof goes here
  sorry

end percentage_hate_german_l123_123870


namespace length_of_CD_l123_123610

theorem length_of_CD (x y: ℝ) (h1: 5 * x = 3 * y) (u v: ℝ) (h2: u = x + 3) (h3: v = y - 3) (h4: 7 * u = 4 * v) : x + y = 264 :=
by
  sorry

end length_of_CD_l123_123610


namespace simplify_expression_l123_123619

variable (x : ℝ)

theorem simplify_expression : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 :=
by
  sorry

end simplify_expression_l123_123619


namespace trig_eqn_solution_l123_123817

open Real

theorem trig_eqn_solution (x : ℝ) (n : ℤ) :
  sin x ≠ 0 →
  cos x ≠ 0 →
  sin x + cos x ≥ 0 →
  (sqrt (1 + tan x) = sin x + cos x) →
  ∃ k : ℤ, (x = k * π + π / 4) ∨ (x = k * π - π / 4) ∨ (x = (2 * k * π + 3 * π / 4)) :=
by
  sorry

end trig_eqn_solution_l123_123817


namespace a2_a4_a6_a8_a10_a12_sum_l123_123574

theorem a2_a4_a6_a8_a10_a12_sum :
  ∀ (x : ℝ), 
    (1 + x + x^2)^6 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7 + a8 * x^8 + a9 * x^9 + a10 * x^10 + a11 * x^11 + a12 * x^12 →
    a2 + a4 + a6 + a8 + a10 + a12 = 364 :=
sorry

end a2_a4_a6_a8_a10_a12_sum_l123_123574


namespace total_people_l123_123499

theorem total_people (M W C : ℕ) (h1 : M = 2 * W) (h2 : W = 3 * C) (h3 : C = 30) : M + W + C = 300 :=
by
  sorry

end total_people_l123_123499


namespace original_number_of_men_l123_123675

theorem original_number_of_men (M : ℤ) (h1 : 8 * M = 5 * (M + 10)) : M = 17 := by
  -- Proof goes here
  sorry

end original_number_of_men_l123_123675


namespace slower_speed_l123_123694

theorem slower_speed (x : ℝ) :
  (5 * (24 / x) = 24 + 6) → x = 4 := 
by
  intro h
  sorry

end slower_speed_l123_123694


namespace pow_mod_remainder_l123_123804

theorem pow_mod_remainder :
  (2^2013 % 11) = 8 :=
sorry

end pow_mod_remainder_l123_123804


namespace gcd_expression_multiple_of_456_l123_123736

theorem gcd_expression_multiple_of_456 (a : ℤ) (h : ∃ k : ℤ, a = 456 * k) : 
  Int.gcd (3 * a^3 + a^2 + 4 * a + 57) a = 57 := by
  sorry

end gcd_expression_multiple_of_456_l123_123736


namespace cos_240_is_neg_half_l123_123341

noncomputable def cos_240 : ℝ := Real.cos (240 * Real.pi / 180)

theorem cos_240_is_neg_half : cos_240 = -1/2 := by
  sorry

end cos_240_is_neg_half_l123_123341


namespace norma_cards_left_l123_123426

def initial_cards : ℕ := 88
def lost_cards : ℕ := 70
def remaining_cards (initial lost : ℕ) : ℕ := initial - lost

theorem norma_cards_left : remaining_cards initial_cards lost_cards = 18 := by
  sorry

end norma_cards_left_l123_123426


namespace seven_times_one_fifth_cubed_l123_123707

theorem seven_times_one_fifth_cubed : 7 * (1 / 5) ^ 3 = 7 / 125 := 
by 
  sorry

end seven_times_one_fifth_cubed_l123_123707


namespace total_ranking_sequences_at_end_l123_123590

-- Define the teams
inductive Team
| E
| F
| G
| H

open Team

-- Conditions of the problem
def split_groups : (Team × Team) × (Team × Team) :=
  ((E, F), (G, H))

def saturday_matches : (Team × Team) × (Team × Team) :=
  ((E, F), (G, H))

-- Function to count total ranking sequences
noncomputable def total_ranking_sequences : ℕ := 4

-- Define the main theorem
theorem total_ranking_sequences_at_end : total_ranking_sequences = 4 :=
by
  sorry

end total_ranking_sequences_at_end_l123_123590


namespace perpendicular_line_and_plane_implication_l123_123388

variable (l m : Line)
variable (α β : Plane)

-- Given conditions
def line_perpendicular_to_plane (l : Line) (α : Plane) : Prop :=
sorry -- Assume this checks if line l is perpendicular to plane α

def line_in_plane (m : Line) (α : Plane) : Prop :=
sorry -- Assume this checks if line m is included in plane α

def line_perpendicular_to_line (l m : Line) : Prop :=
sorry -- Assume this checks if line l is perpendicular to line m

-- Lean statement for the proof problem
theorem perpendicular_line_and_plane_implication
  (h1 : line_perpendicular_to_plane l α)
  (h2 : line_in_plane m α) :
  line_perpendicular_to_line l m :=
sorry

end perpendicular_line_and_plane_implication_l123_123388


namespace sqrt_sum_eq_nine_l123_123575

theorem sqrt_sum_eq_nine (x : ℝ) (h : Real.sqrt (7 + x) + Real.sqrt (28 - x) = 9) :
  (7 + x) * (28 - x) = 529 :=
sorry

end sqrt_sum_eq_nine_l123_123575


namespace find_a_value_l123_123393

theorem find_a_value (a a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, (x + 1)^5 = a + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5) → 
  a = 32 :=
by
  sorry

end find_a_value_l123_123393


namespace count_prime_two_digit_sum_ten_is_three_l123_123541

-- Define a helper function to check if a number is a prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function to compute the sum of the digits of a two-digit number
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def sum_of_digits_is_ten (n : ℕ) : Prop := digit_sum n = 10

-- Prove the number of prime two-digit numbers whose digits sum to 10 is 3
theorem count_prime_two_digit_sum_ten_is_three :
  finset.filter (λ n, is_prime n ∧ is_two_digit n ∧ sum_of_digits_is_ten n) (finset.range 100)).card = 3 :=
by
  sorry

end count_prime_two_digit_sum_ten_is_three_l123_123541


namespace find_second_number_l123_123133

-- Define the two numbers A and B
variables (A B : ℝ)

-- Define the conditions
def condition1 := 0.20 * A = 0.30 * B + 80
def condition2 := A = 580

-- Define the goal
theorem find_second_number (h1 : condition1 A B) (h2 : condition2 A) : B = 120 :=
by sorry

end find_second_number_l123_123133


namespace expression_value_l123_123446

theorem expression_value (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) 
  (h₃ : (a^2 * b^2) / (a^4 - 2 * b^4) = 1) : 
  (a^2 - b^2) / (a^2 + b^2) = 1/3 := 
by
  sorry

end expression_value_l123_123446


namespace fraction_subtraction_equivalence_l123_123973

theorem fraction_subtraction_equivalence :
  (8 / 19) - (5 / 57) = 1 / 3 :=
by sorry

end fraction_subtraction_equivalence_l123_123973


namespace max_discount_l123_123271

variable (x : ℝ)

theorem max_discount (h1 : (1 + 0.8) * x = 360) : 360 - 1.2 * x = 120 := 
by
  sorry

end max_discount_l123_123271


namespace simplify_expression_l123_123864

theorem simplify_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^y * y^x) / (y^y * x^x) = (x / y) ^ (y - x) :=
by
  sorry

end simplify_expression_l123_123864


namespace classify_triangles_by_angles_l123_123646

-- Define the basic types and properties for triangles and their angle classifications
def acute_triangle (α β γ : ℝ) : Prop :=
  α < 90 ∧ β < 90 ∧ γ < 90

def right_triangle (α β γ : ℝ) : Prop :=
  α = 90 ∨ β = 90 ∨ γ = 90

def obtuse_triangle (α β γ : ℝ) : Prop :=
  α > 90 ∨ β > 90 ∨ γ > 90

-- Problem: Classify triangles by angles and prove that the correct classification is as per option A
theorem classify_triangles_by_angles :
  (∀ (α β γ : ℝ), acute_triangle α β γ ∨ right_triangle α β γ ∨ obtuse_triangle α β γ) :=
sorry

end classify_triangles_by_angles_l123_123646


namespace derivative_at_two_l123_123565

noncomputable def f (a : ℝ) (g : ℝ) (x : ℝ) : ℝ := a * x^3 + g * x^2 + 3

theorem derivative_at_two (a f_prime_2 : ℝ) (h_deriv_at_1 : deriv (f a f_prime_2) 1 = -5) :
  deriv (f a f_prime_2) 2 = -5 := by
  sorry

end derivative_at_two_l123_123565


namespace clinton_earnings_correct_l123_123905

-- Define the conditions as variables/constants
def num_students_Arlington : ℕ := 8
def days_Arlington : ℕ := 4

def num_students_Bradford : ℕ := 6
def days_Bradford : ℕ := 7

def num_students_Clinton : ℕ := 7
def days_Clinton : ℕ := 8

def total_compensation : ℝ := 1456

noncomputable def total_student_days : ℕ :=
  num_students_Arlington * days_Arlington + num_students_Bradford * days_Bradford + num_students_Clinton * days_Clinton

noncomputable def daily_wage : ℝ :=
  total_compensation / total_student_days

noncomputable def earnings_Clinton : ℝ :=
  daily_wage * (num_students_Clinton * days_Clinton)

theorem clinton_earnings_correct : earnings_Clinton = 627.2 := by 
  sorry

end clinton_earnings_correct_l123_123905


namespace find_x_value_l123_123256

-- Define the condition as a hypothesis
def condition (x : ℝ) : Prop := (x / 4) - x - (3 / 6) = 1

-- State the theorem
theorem find_x_value (x : ℝ) (h : condition x) : x = -2 := 
by sorry

end find_x_value_l123_123256


namespace subset_A_imp_range_a_disjoint_A_imp_range_a_l123_123569

-- Definition of sets A and B
def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a)*(x - 3*a) < 0}

-- Proof problem for Question 1
theorem subset_A_imp_range_a (a : ℝ) (h : A ⊆ B a) : 
  (4 / 3) ≤ a ∧ a ≤ 2 ∧ a ≠ 0 :=
sorry

-- Proof problem for Question 2
theorem disjoint_A_imp_range_a (a : ℝ) (h : A ∩ B a = ∅) : 
  a ≤ (2 / 3) ∨ a ≥ 4 :=
sorry

end subset_A_imp_range_a_disjoint_A_imp_range_a_l123_123569


namespace fewer_bees_than_flowers_l123_123458

theorem fewer_bees_than_flowers : 5 - 3 = 2 := by
  sorry

end fewer_bees_than_flowers_l123_123458


namespace cos_240_eq_negative_half_l123_123331

theorem cos_240_eq_negative_half : Real.cos (240 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_240_eq_negative_half_l123_123331


namespace number_one_fourth_less_than_25_percent_more_l123_123461

theorem number_one_fourth_less_than_25_percent_more (x : ℝ) :
  (3 / 4) * x = 1.25 * 80 → x = 133.33 :=
by
  intros h
  sorry

end number_one_fourth_less_than_25_percent_more_l123_123461


namespace tank_fill_fraction_l123_123061

theorem tank_fill_fraction (a b c : ℝ) (h1 : a=9) (h2 : b=54) (h3 : c=3/4) : (c * b + a) / b = 23 / 25 := 
by 
  sorry

end tank_fill_fraction_l123_123061


namespace count_two_digit_prime_with_digit_sum_10_l123_123530

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_two_digit_prime_with_digit_sum_10 : 
  (∃ n1 n2 n3 : ℕ, 
    (sum_of_digits n1 = 10 ∧ is_prime n1 ∧ 10 ≤ n1 ∧ n1 < 100) ∧
    (sum_of_digits n2 = 10 ∧ is_prime n2 ∧ 10 ≤ n2 ∧ n2 < 100) ∧
    (sum_of_digits n3 = 10 ∧ is_prime n3 ∧ 10 ≤ n3 ∧ n3 < 100) ∧
    n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) ∧
  ∀ n : ℕ, 
    (sum_of_digits n = 10 ∧ is_prime n ∧ 10 ≤ n ∧ n < 100)
    → (n = n1 ∨ n = n2 ∨ n = n3) :=
sorry

end count_two_digit_prime_with_digit_sum_10_l123_123530


namespace range_of_p_l123_123740

theorem range_of_p (p : ℝ) : 
  (∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → x^2 + p * x + 1 > 2 * x + p) → p > -1 := 
by
  sorry

end range_of_p_l123_123740


namespace sum_modified_midpoint_coordinates_l123_123015

theorem sum_modified_midpoint_coordinates :
  let p1 : (ℝ × ℝ) := (10, 3)
  let p2 : (ℝ × ℝ) := (-4, 7)
  let midpoint : (ℝ × ℝ) := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let modified_x := 2 * midpoint.1 
  (modified_x + midpoint.2) = 11 := by
  sorry

end sum_modified_midpoint_coordinates_l123_123015


namespace sector_area_correct_l123_123738

noncomputable def sector_area (θ r : ℝ) : ℝ :=
  (θ / (2 * Real.pi)) * (Real.pi * r^2)

theorem sector_area_correct : 
  sector_area (Real.pi / 3) 3 = (3 / 2) * Real.pi :=
by
  sorry

end sector_area_correct_l123_123738


namespace total_spent_on_concert_tickets_l123_123856

theorem total_spent_on_concert_tickets : 
  let price_per_ticket := 4
  let number_of_tickets := 3 + 5
  let discount_threshold := 5
  let discount_rate := 0.10
  let service_fee_per_ticket := 2
  let initial_cost := number_of_tickets * price_per_ticket
  let discount := if number_of_tickets > discount_threshold then discount_rate * initial_cost else 0
  let discounted_cost := initial_cost - discount
  let service_fee := number_of_tickets * service_fee_per_ticket
  let total_cost := discounted_cost + service_fee
  total_cost = 44.8 :=
by
  sorry

end total_spent_on_concert_tickets_l123_123856


namespace exists_multiple_with_equal_digit_sum_l123_123214

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_multiple_with_equal_digit_sum (k : ℕ) (h : k > 0) : 
  ∃ n : ℕ, (n % k = 0) ∧ (sum_of_digits n = sum_of_digits (n * n)) :=
sorry

end exists_multiple_with_equal_digit_sum_l123_123214


namespace evaluate_expression_l123_123820

theorem evaluate_expression : 
  70 + (5 * 12) / (180 / 3) = 71 :=
  by
  sorry

end evaluate_expression_l123_123820


namespace find_point_coordinates_l123_123074

open Real

-- Define circles C1 and C2
def circle_C1 (x y : ℝ) : Prop := (x + 4)^2 + (y - 2)^2 = 9
def circle_C2 (x y : ℝ) : Prop := (x - 5)^2 + (y - 6)^2 = 9

-- Define mutually perpendicular lines passing through point P
def line_l1 (P : ℝ × ℝ) (k : ℝ) (x y : ℝ) : Prop := y - P.2 = k * (x - P.1)
def line_l2 (P : ℝ × ℝ) (k : ℝ) (x y : ℝ) : Prop := y - P.2 = -1/k * (x - P.1)

-- Define the condition that chord lengths intercepted by lines on respective circles are equal
def equal_chord_lengths (P : ℝ × ℝ) (k : ℝ) : Prop :=
  abs (-4 * k - 2 + P.2 - k * P.1) / sqrt ((k^2) + 1) = abs (5 + 6 * k - k * P.2 - P.1) / sqrt ((k^2) + 1)

-- Main statement to be proved
theorem find_point_coordinates :
  ∃ (P : ℝ × ℝ), 
  circle_C1 (P.1) (P.2) ∧
  circle_C2 (P.1) (P.2) ∧
  (∀ k : ℝ, k ≠ 0 → equal_chord_lengths P k) ∧
  (P = (-3/2, 17/2) ∨ P = (5/2, -1/2)) :=
sorry

end find_point_coordinates_l123_123074


namespace cos_240_eq_neg_half_l123_123335

/-- Prove that cos 240 degrees equals -1/2 --/
theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1/2 :=
  sorry

end cos_240_eq_neg_half_l123_123335


namespace monotonically_increasing_sequence_b_bounds_l123_123963

theorem monotonically_increasing_sequence_b_bounds (b : ℝ) :
  (∀ n : ℕ, 0 < n → (n + 1)^2 + b * (n + 1) > n^2 + b * n) ↔ b > -3 :=
by
  sorry

end monotonically_increasing_sequence_b_bounds_l123_123963


namespace chocolates_sold_at_selling_price_l123_123866

theorem chocolates_sold_at_selling_price
  (C S : ℚ) (n : ℕ)
  (h1 : 44 * C = n * S)
  (h2 : S = 11/6 * C) :
  n = 24 :=
by
  -- Proof steps would be inserted here
  sorry

end chocolates_sold_at_selling_price_l123_123866


namespace diff_of_squares_525_475_l123_123669

theorem diff_of_squares_525_475 : 525^2 - 475^2 = 50000 := by
  sorry

end diff_of_squares_525_475_l123_123669


namespace max_value_neg7s_squared_plus_56s_plus_20_l123_123157

theorem max_value_neg7s_squared_plus_56s_plus_20 :
  ∃ s : ℝ, s = 4 ∧ ∀ t : ℝ, -7 * t^2 + 56 * t + 20 ≤ 132 := 
by
  sorry

end max_value_neg7s_squared_plus_56s_plus_20_l123_123157


namespace count_two_digit_primes_with_digit_sum_10_is_4_l123_123531

noncomputable def countTwoDigitPrimesWithDigitSum10 : ℕ :=
  let digitPairs := [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]
  let numbers := digitPairs.map (λ p => p.1 * 10 + p.2)
  numbers.count (λ n => Nat.Prime n)

theorem count_two_digit_primes_with_digit_sum_10_is_4 : countTwoDigitPrimesWithDigitSum10 = 4 :=
  sorry

end count_two_digit_primes_with_digit_sum_10_is_4_l123_123531


namespace distinct_permutations_mathematics_l123_123516

theorem distinct_permutations_mathematics : 
  let n := 11
  let freqM := 2
  let freqA := 2
  let freqT := 2
  (n.factorial / (freqM.factorial * freqA.factorial * freqT.factorial)) = 4989600 :=
by
  let n := 11
  let freqM := 2
  let freqA := 2
  let freqT := 2
  sorry

end distinct_permutations_mathematics_l123_123516


namespace count_multiples_of_4_l123_123860

/-- 
Prove that the number of multiples of 4 between 100 and 300 inclusive is 49.
-/
theorem count_multiples_of_4 : 
  ∃ n : ℕ, (∀ k : ℕ, 100 ≤ 4 * k ∧ 4 * k ≤ 300 ↔ k = 26 + n) ∧ n = 48 :=
by
  sorry

end count_multiples_of_4_l123_123860


namespace am_gm_inequality_l123_123765

theorem am_gm_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 1) :
  (1 - x) * (1 - y) * (1 - z) ≥ 8 * x * y * z :=
by sorry

end am_gm_inequality_l123_123765


namespace girls_count_l123_123069

variable (B G : ℕ)

theorem girls_count (h1: B = 387) (h2: G = (B + (54 * B) / 100)) : G = 596 := 
by 
  sorry

end girls_count_l123_123069


namespace find_remainder_l123_123987

theorem find_remainder (P Q R D D' Q' R' C : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R') :
  (P % (D * D')) = (D * R' + R + C) :=
sorry

end find_remainder_l123_123987


namespace red_large_toys_count_l123_123205

def percentage_red : ℝ := 0.25
def percentage_green : ℝ := 0.20
def percentage_blue : ℝ := 0.15
def percentage_yellow : ℝ := 0.25
def percentage_orange : ℝ := 0.15

def red_small : ℝ := 0.06
def red_medium : ℝ := 0.08
def red_large : ℝ := 0.07
def red_extra_large : ℝ := 0.04

def green_small : ℝ := 0.04
def green_medium : ℝ := 0.07
def green_large : ℝ := 0.05
def green_extra_large : ℝ := 0.04

def blue_small : ℝ := 0.06
def blue_medium : ℝ := 0.03
def blue_large : ℝ := 0.04
def blue_extra_large : ℝ := 0.02

def yellow_small : ℝ := 0.08
def yellow_medium : ℝ := 0.10
def yellow_large : ℝ := 0.05
def yellow_extra_large : ℝ := 0.02

def orange_small : ℝ := 0.09
def orange_medium : ℝ := 0.06
def orange_large : ℝ := 0.05
def orange_extra_large : ℝ := 0.05

def green_large_count : ℕ := 47

noncomputable def total_green_toys := green_large_count / green_large

noncomputable def total_toys := total_green_toys / percentage_green

noncomputable def red_large_toys := total_toys * red_large

theorem red_large_toys_count : red_large_toys = 329 := by
  sorry

end red_large_toys_count_l123_123205


namespace fraction_subtraction_equivalence_l123_123972

theorem fraction_subtraction_equivalence :
  (8 / 19) - (5 / 57) = 1 / 3 :=
by sorry

end fraction_subtraction_equivalence_l123_123972


namespace identify_quadratic_equation_l123_123251

def is_quadratic (eq : String) : Prop :=
  eq = "a * x^2 + b * x + c = 0"  /-
  This definition is a placeholder for checking if a 
  given equation is in the quadratic form. In practice,
  more advanced techniques like parsing and formally
  verifying the quadratic form would be used. -/

theorem identify_quadratic_equation :
  (is_quadratic "2 * x^2 - x - 3 = 0") :=
by
  sorry

end identify_quadratic_equation_l123_123251


namespace rounding_strategy_correct_l123_123701

-- Definitions of rounding functions
def round_down (n : ℕ) : ℕ := n - 1  -- Assuming n is a large integer, for simplicity
def round_up (n : ℕ) : ℕ := n + 1

-- Definitions for conditions
def cond1 (p q r : ℕ) : ℕ := round_down p / round_down q + round_down r
def cond2 (p q r : ℕ) : ℕ := round_up p / round_down q + round_down r
def cond3 (p q r : ℕ) : ℕ := round_down p / round_up q + round_down r
def cond4 (p q r : ℕ) : ℕ := round_down p / round_down q + round_up r
def cond5 (p q r : ℕ) : ℕ := round_up p / round_up q + round_down r

-- Theorem stating the correct condition
theorem rounding_strategy_correct (p q r : ℕ) (hp : 1 ≤ p) (hq : 1 ≤ q) (hr : 1 ≤ r) :
  cond3 p q r < p / q + r :=
sorry

end rounding_strategy_correct_l123_123701


namespace total_people_in_group_l123_123496

theorem total_people_in_group (men women children : ℕ)
  (h1 : men = 2 * women)
  (h2 : women = 3 * children)
  (h3 : children = 30) :
  men + women + children = 300 :=
by
  sorry

end total_people_in_group_l123_123496


namespace find_number_l123_123715

theorem find_number (x : ℝ) : 
  (3 * x / 5 - 220) * 4 + 40 = 360 → x = 500 :=
by
  intro h
  sorry

end find_number_l123_123715


namespace triangle_angle_area_l123_123066

theorem triangle_angle_area
  (A B C : ℝ) (a b c : ℝ)
  (h1 : c * Real.cos B = (2 * a - b) * Real.cos C)
  (h2 : C = Real.pi / 3)
  (h3 : c = 2)
  (h4 : a + b + c = 2 * Real.sqrt 3 + 2) :
  ∃ (area : ℝ), area = (2 * Real.sqrt 3) / 3 :=
by 
  -- Proof is omitted
  sorry

end triangle_angle_area_l123_123066


namespace carol_blocks_l123_123283

theorem carol_blocks (initial_blocks lost_blocks final_blocks : ℕ) 
  (h_initial : initial_blocks = 42) 
  (h_lost : lost_blocks = 25) : 
  final_blocks = initial_blocks - lost_blocks → final_blocks = 17 := by
  sorry

end carol_blocks_l123_123283


namespace sine_sum_zero_l123_123432

open Real 

theorem sine_sum_zero (α β γ : ℝ) :
  (sin α / (sin (α - β) * sin (α - γ))
  + sin β / (sin (β - α) * sin (β - γ))
  + sin γ / (sin (γ - α) * sin (γ - β)) = 0) :=
sorry

end sine_sum_zero_l123_123432


namespace simple_interest_sum_l123_123129

theorem simple_interest_sum :
  let P := 1750
  let CI := 4000 * ((1 + (10 / 100))^2) - 4000
  let SI := (1 / 2) * CI
  SI = (P * 8 * 3) / 100 
  :=
by
  -- Definitions
  let P := 1750
  let CI := 4000 * ((1 + 10 / 100)^2) - 4000
  let SI := (1 / 2) * CI
  
  -- Claim
  have : SI = (P * 8 * 3) / 100 := sorry

  exact this

end simple_interest_sum_l123_123129


namespace saving_20_days_cost_saving_20_days_saving_60_days_cost_saving_60_days_l123_123931

noncomputable def bread_saving (n_days : ℕ) : ℕ :=
  (1 / 2) * n_days

theorem saving_20_days :
  bread_saving 20 = 10 :=
by
  -- proof steps for bread_saving 20 = 10
  sorry

theorem cost_saving_20_days (cost_per_loaf : ℕ) :
  cost_per_loaf = 35 → (bread_saving 20 * cost_per_loaf) = 350 :=
by
  -- proof steps for cost_saving_20_days
  sorry

theorem saving_60_days :
  bread_saving 60 = 30 :=
by
  -- proof steps for bread_saving 60 = 30
  sorry

theorem cost_saving_60_days (cost_per_loaf : ℕ) :
  cost_per_loaf = 35 → (bread_saving 60 * cost_per_loaf) = 1050 :=
by
  -- proof steps for cost_saving_60_days
  sorry

end saving_20_days_cost_saving_20_days_saving_60_days_cost_saving_60_days_l123_123931


namespace complement_of_A_relative_to_U_l123_123216

def U := { x : ℝ | x < 3 }
def A := { x : ℝ | x < 1 }

def complement_U_A := { x : ℝ | 1 ≤ x ∧ x < 3 }

theorem complement_of_A_relative_to_U : (complement_U_A = { x : ℝ | x ∈ U ∧ x ∉ A }) :=
by
  sorry

end complement_of_A_relative_to_U_l123_123216


namespace greatest_x_value_l123_123976

theorem greatest_x_value :
  ∃ x : ℝ, (x ≠ 2 ∧ (x^2 - 5 * x - 14) / (x - 2) = 4 / (x + 4)) ∧ x = -2 ∧ 
           ∀ y, (y ≠ 2 ∧ (y^2 - 5 * y - 14) / (y - 2) = 4 / (y + 4)) → y ≤ x :=
by
  sorry

end greatest_x_value_l123_123976


namespace terminal_sides_positions_l123_123994

def in_third_quadrant (θ : ℝ) (k : ℤ) : Prop :=
  (180 + k * 360 : ℝ) < θ ∧ θ < (270 + k * 360 : ℝ)

theorem terminal_sides_positions (θ : ℝ) (k : ℤ) :
  in_third_quadrant θ k →
  ((2 * θ > 360 + 2 * k * 360 ∧ 2 * θ < 540 + 2 * k * 360) ∨
   (90 + k * 180 < θ / 2 ∧ θ / 2 < 135 + k * 180) ∨
   (2 * θ = 360 + 2 * k * 360) ∨ (2 * θ = 540 + 2 * k * 360) ∨ 
   (θ / 2 = 90 + k * 180) ∨ (θ / 2 = 135 + k * 180)) :=
by
  intro h
  sorry

end terminal_sides_positions_l123_123994


namespace S_13_eq_3510_l123_123964

def S (n : ℕ) : ℕ := n * (n + 2) * (n + 4) + n * (n + 2)

theorem S_13_eq_3510 : S 13 = 3510 :=
by
  sorry

end S_13_eq_3510_l123_123964


namespace triangle_angles_are_equal_l123_123217

theorem triangle_angles_are_equal
  (A B C : ℝ) (a b c : ℝ)
  (h1 : A + B + C = π)
  (h2 : A = B + (B - A))
  (h3 : B = C + (C - B))
  (h4 : 2 * (1 / b) = (1 / a) + (1 / c)) :
  A = π / 3 ∧ B = π / 3 ∧ C = π / 3 :=
sorry

end triangle_angles_are_equal_l123_123217


namespace nine_pow_div_eighty_one_pow_l123_123463

theorem nine_pow_div_eighty_one_pow (a b : ℕ) (h1 : a = 9^2) (h2 : b = a^4) :
  (9^10 / b = 81) := by
  sorry

end nine_pow_div_eighty_one_pow_l123_123463


namespace natalie_needs_10_bushes_l123_123024

-- Definitions based on the conditions
def bushes_to_containers (bushes : ℕ) := bushes * 10
def containers_to_zucchinis (containers : ℕ) := (containers * 3) / 4

-- The proof statement
theorem natalie_needs_10_bushes :
  ∃ bushes : ℕ, containers_to_zucchinis (bushes_to_containers bushes) ≥ 72 ∧ bushes = 10 :=
sorry

end natalie_needs_10_bushes_l123_123024


namespace man_age_difference_l123_123943

theorem man_age_difference (S M : ℕ) (h1 : S = 22) (h2 : M + 2 = 2 * (S + 2)) :
  M - S = 24 :=
by sorry

end man_age_difference_l123_123943


namespace personal_trainer_cost_proof_l123_123892

-- Define the conditions
def hourly_wage_before_raise : ℝ := 40
def raise_percentage : ℝ := 0.05
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 5
def old_bills_per_week : ℝ := 600
def leftover_money : ℝ := 980

-- Define the question
def new_hourly_wage : ℝ := hourly_wage_before_raise * (1 + raise_percentage)
def weekly_hours : ℕ := hours_per_day * days_per_week
def weekly_earnings : ℝ := new_hourly_wage * weekly_hours
def total_weekly_expenses : ℝ := weekly_earnings - leftover_money
def personal_trainer_cost_per_week : ℝ := total_weekly_expenses - old_bills_per_week

-- Theorem statement
theorem personal_trainer_cost_proof : personal_trainer_cost_per_week = 100 := 
by
  -- Proof to be filled
  sorry

end personal_trainer_cost_proof_l123_123892


namespace cost_price_per_meter_of_cloth_l123_123924

theorem cost_price_per_meter_of_cloth
  (meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) (total_profit : ℕ) (cost_price : ℕ)
  (meters_eq : meters = 80)
  (selling_price_eq : selling_price = 10000)
  (profit_per_meter_eq : profit_per_meter = 7)
  (total_profit_eq : total_profit = profit_per_meter * meters)
  (selling_price_calc : selling_price = cost_price + total_profit)
  (cost_price_calc : cost_price = selling_price - total_profit)
  : (selling_price - total_profit) / meters = 118 :=
by
  -- here we would provide the proof, but we skip it with sorry
  sorry

end cost_price_per_meter_of_cloth_l123_123924


namespace find_some_number_l123_123267

theorem find_some_number : 
  ∃ (some_number : ℝ), (∃ (n : ℝ), n = 54 ∧ (n / some_number) * (n / 162) = 1) → some_number = 18 :=
by
  sorry

end find_some_number_l123_123267


namespace evaluate_expression_l123_123025

theorem evaluate_expression : (4^150 * 9^152) / 6^301 = 27 / 2 := 
by 
  -- skipping the actual proof
  sorry

end evaluate_expression_l123_123025


namespace task_force_combinations_l123_123684

theorem task_force_combinations :
  (Nat.choose 10 4) * (Nat.choose 7 3) = 7350 :=
by
  sorry

end task_force_combinations_l123_123684


namespace brother_spent_on_highlighters_l123_123743

theorem brother_spent_on_highlighters : 
  let total_money := 100
  let cost_sharpener := 5
  let num_sharpeners := 2
  let cost_notebook := 5
  let num_notebooks := 4
  let cost_eraser := 4
  let num_erasers := 10
  let total_spent_sharpeners := num_sharpeners * cost_sharpener
  let total_spent_notebooks := num_notebooks * cost_notebook
  let total_spent_erasers := num_erasers * cost_eraser
  let total_spent := total_spent_sharpeners + total_spent_notebooks + total_spent_erasers
  let remaining_money := total_money - total_spent
  remaining_money = 30 :=
begin
  sorry
end

end brother_spent_on_highlighters_l123_123743


namespace domain_of_sqrt_and_fraction_l123_123443

def domain_of_function (x : ℝ) : Prop :=
  2 * x - 3 ≥ 0 ∧ x ≠ 3

theorem domain_of_sqrt_and_fraction :
  {x : ℝ | domain_of_function x} = {x : ℝ | x ≥ 3 / 2} \ {3} :=
by sorry

end domain_of_sqrt_and_fraction_l123_123443


namespace find_c_l123_123670

theorem find_c (c : ℝ) : 
  (∀ x : ℝ, x * (3 * x + 1) < c ↔ x ∈ Set.Ioo (-(7 / 3) : ℝ) (2 : ℝ)) → c = 14 :=
by
  intro h
  sorry

end find_c_l123_123670


namespace divides_quartic_sum_l123_123430

theorem divides_quartic_sum (a b c n : ℤ) (h1 : n ∣ (a + b + c)) (h2 : n ∣ (a^2 + b^2 + c^2)) : n ∣ (a^4 + b^4 + c^4) := 
sorry

end divides_quartic_sum_l123_123430


namespace cos_240_eq_neg_half_l123_123352

open Real

theorem cos_240_eq_neg_half : cos (240 * π / 180) = -1/2 :=
by
  -- Step 1: Decompose the angle 240° = 180° + 60°
  have h1 : 240 * π / 180 = π + 60 * π / 180,
  { 
    norm_num, 
    field_simp, 
    linarith 
  },
  -- Step 2: Use the fact that the cosine of (π + θ) = - cos(θ)
  rw [h1, cos_add_pi],
  -- Step 3: Given that cos(60°) = 1/2
  have h2 : cos (60 * π / 180) = 1/2,
  {
    norm_num,
    exact Real.cos_pi_div_three (),
  },
  -- Conclude that cos(240°) = -1/2
  rw h2,
  norm_num

end cos_240_eq_neg_half_l123_123352


namespace brother_highlighters_spent_l123_123745

-- Define the total money given by the father
def total_money : ℕ := 100

-- Define the amount Heaven spent (2 sharpeners + 4 notebooks at $5 each)
def heaven_spent : ℕ := 30

-- Define the amount Heaven's brother spent on erasers (10 erasers at $4 each)
def erasers_spent : ℕ := 40

-- Prove the amount Heaven's brother spent on highlighters
theorem brother_highlighters_spent : total_money - heaven_spent - erasers_spent == 30 :=
by
  sorry

end brother_highlighters_spent_l123_123745


namespace cos_240_eq_neg_half_l123_123333

/-- Prove that cos 240 degrees equals -1/2 --/
theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1/2 :=
  sorry

end cos_240_eq_neg_half_l123_123333


namespace count_two_digit_primes_with_digit_sum_10_l123_123545

theorem count_two_digit_primes_with_digit_sum_10 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n.digits 10).sum = 10 ∧ n.prime}.toFinset.card = 3 :=
by 
  sorry

end count_two_digit_primes_with_digit_sum_10_l123_123545


namespace original_number_is_17_l123_123250

theorem original_number_is_17 (x : ℤ) (h : (x + 6) % 23 = 0) : x = 17 :=
sorry

end original_number_is_17_l123_123250


namespace actual_average_height_correct_l123_123236

noncomputable def actual_average_height (n : ℕ) (average_height : ℝ) (wrong_height : ℝ) (actual_height : ℝ) : ℝ :=
  let total_height := average_height * n
  let difference := wrong_height - actual_height
  let correct_total_height := total_height - difference
  correct_total_height / n

theorem actual_average_height_correct :
  actual_average_height 35 184 166 106 = 182.29 :=
by
  sorry

end actual_average_height_correct_l123_123236


namespace artifacts_per_wing_l123_123690

theorem artifacts_per_wing (total_wings : ℕ) (painting_wings : ℕ) 
    (large_paintings : ℕ) (small_paintings_per_wing : ℕ) 
    (artifact_ratio : ℕ) 
    (h_total_wings : total_wings = 8) 
    (h_painting_wings : painting_wings = 3) 
    (h_large_paintings : large_paintings = 1) 
    (h_small_paintings_per_wing : small_paintings_per_wing = 12) 
    (h_artifact_ratio : artifact_ratio = 4) :
    let total_paintings := large_paintings + small_paintings_per_wing * 2 in
    let total_artifacts := artifact_ratio * total_paintings in
    let artifact_wings := total_wings - painting_wings in
    total_artifacts / artifact_wings = 20 :=
by
    sorry

end artifacts_per_wing_l123_123690


namespace man_age_difference_l123_123945

theorem man_age_difference (S M : ℕ) (h1 : S = 22) (h2 : M + 2 = 2 * (S + 2)) : M - S = 24 :=
by
  sorry

end man_age_difference_l123_123945


namespace actual_average_speed_l123_123685

variable {t : ℝ} (h₁ : t > 0) -- ensure that time is positive
variable {v : ℝ} 

theorem actual_average_speed (h₂ : v > 0)
  (h3 : v * t = (v + 12) * (3 / 4 * t)) : v = 36 :=
by
  sorry

end actual_average_speed_l123_123685


namespace cos_240_degree_l123_123297

theorem cos_240_degree : Real.cos (240 * Real.pi / 180) = -1/2 :=
by
  sorry

end cos_240_degree_l123_123297


namespace machines_produce_12x_boxes_in_expected_time_l123_123809

-- Definitions corresponding to the conditions
def rate_A (x : ℕ) := x / 10
def rate_B (x : ℕ) := 2 * x / 5
def rate_C (x : ℕ) := 3 * x / 8
def rate_D (x : ℕ) := x / 4

-- Total combined rate when working together
def combined_rate (x : ℕ) := rate_A x + rate_B x + rate_C x + rate_D x

-- The time taken to produce 12x boxes given their combined rate
def time_to_produce (x : ℕ) : ℕ := 12 * x / combined_rate x

-- Goal: Time taken should be 32/3 minutes
theorem machines_produce_12x_boxes_in_expected_time (x : ℕ) : time_to_produce x = 32 / 3 :=
sorry

end machines_produce_12x_boxes_in_expected_time_l123_123809


namespace num_two_digit_primes_with_digit_sum_10_l123_123536

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.to_finset.card = 3 := by
  sorry

end num_two_digit_primes_with_digit_sum_10_l123_123536


namespace normal_distribution_determined_l123_123208

namespace normal_distribution

-- Define the parameters for the normal distribution
variables (μ σ : ℝ)

-- The theorem statement: the parameters μ (mean) and σ (standard deviation) fully determine the normal distribution
theorem normal_distribution_determined (μ σ : ℝ) : 
  (∀ x, normal_pdf μ σ x = normal_pdf μ σ x) :=
sorry

end normal_distribution

end normal_distribution_determined_l123_123208


namespace exponent_problem_l123_123279

theorem exponent_problem : (-1 : ℝ)^2003 / (-1 : ℝ)^2004 = -1 := by
  sorry

end exponent_problem_l123_123279


namespace cos_240_eq_neg_half_l123_123332

/-- Prove that cos 240 degrees equals -1/2 --/
theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1/2 :=
  sorry

end cos_240_eq_neg_half_l123_123332


namespace count_multiples_of_4_l123_123859

/-- 
Prove that the number of multiples of 4 between 100 and 300 inclusive is 49.
-/
theorem count_multiples_of_4 : 
  ∃ n : ℕ, (∀ k : ℕ, 100 ≤ 4 * k ∧ 4 * k ≤ 300 ↔ k = 26 + n) ∧ n = 48 :=
by
  sorry

end count_multiples_of_4_l123_123859


namespace trivia_team_points_l123_123952

theorem trivia_team_points (total_members absent_members total_points : ℕ) 
    (h1 : total_members = 5) 
    (h2 : absent_members = 2) 
    (h3 : total_points = 18) 
    (h4 : total_members - absent_members = present_members) 
    (h5 : total_points = present_members * points_per_member) : 
    points_per_member = 6 :=
  sorry

end trivia_team_points_l123_123952


namespace find_roots_of_g_l123_123752

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 - a*x - b
noncomputable def g (x : ℝ) (a b : ℝ) : ℝ := b*x^2 - a*x - 1

theorem find_roots_of_g :
  (∀ a b : ℝ, f 2 a b = 0 ∧ f 3 a b = 0 → ∃ (x1 x2 : ℝ), g x1 a b = 0 ∧ g x2 a b = 0 ∧
    (x1 = -1/2 ∨ x1 = -1/3) ∧ (x2 = -1/2 ∨ x2 = -1/3) ∧ x1 ≠ x2) :=
by
  sorry

end find_roots_of_g_l123_123752


namespace annual_average_growth_rate_l123_123999

theorem annual_average_growth_rate (x : ℝ) :
  7200 * (1 + x)^2 = 8450 :=
sorry

end annual_average_growth_rate_l123_123999


namespace carlos_class_number_l123_123634

theorem carlos_class_number (b : ℕ) :
  (100 < b ∧ b < 200) ∧
  (b + 2) % 4 = 0 ∧
  (b + 3) % 5 = 0 ∧
  (b + 4) % 6 = 0 →
  b = 122 ∨ b = 182 :=
by
  -- The proof implementation goes here
  sorry

end carlos_class_number_l123_123634


namespace range_of_a_l123_123185

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ 1 → x * |x - a| - 2 < 0) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l123_123185


namespace no_divisibility_condition_by_all_others_l123_123780

theorem no_divisibility_condition_by_all_others 
  {p : ℕ → ℕ} 
  (h_distinct_odd_primes : ∀ i j, i ≠ j → Nat.Prime (p i) ∧ Nat.Prime (p j) ∧ p i ≠ p j ∧ p i % 2 = 1 ∧ p j % 2 = 1)
  (h_ordered : ∀ i j, i < j → p i < p j) :
  ¬ ∀ i j, i ≠ j → (∀ k ≠ i, k ≠ j → p k ∣ (p i ^ 8 - p j ^ 8)) :=
by
  sorry

end no_divisibility_condition_by_all_others_l123_123780


namespace calc_expression1_calc_expression2_l123_123153

-- Problem 1
theorem calc_expression1 (x y : ℝ) : (1/2 * x * y)^2 * 6 * x^2 * y = (3/2) * x^4 * y^3 := 
sorry

-- Problem 2
theorem calc_expression2 (a b : ℝ) : (2 * a + b)^2 = 4 * a^2 + 4 * a * b + b^2 := 
sorry

end calc_expression1_calc_expression2_l123_123153


namespace probability_both_tell_truth_l123_123481

theorem probability_both_tell_truth (pA pB : ℝ) (hA : pA = 0.80) (hB : pB = 0.60) : pA * pB = 0.48 :=
by
  subst hA
  subst hB
  sorry

end probability_both_tell_truth_l123_123481


namespace simplify_expression_l123_123618

variable (x : ℝ)

theorem simplify_expression : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 :=
by
  sorry

end simplify_expression_l123_123618


namespace positive_solution_of_x_l123_123570

theorem positive_solution_of_x :
  ∃ x y z : ℝ, (x * y = 6 - 2 * x - 3 * y) ∧ (y * z = 6 - 4 * y - 2 * z) ∧ (x * z = 30 - 4 * x - 3 * z) ∧ x > 0 ∧ x = 3 :=
by
  sorry

end positive_solution_of_x_l123_123570


namespace difference_of_roots_l123_123839

theorem difference_of_roots (r1 r2 : ℝ) 
    (h_eq : ∀ x : ℝ, x^2 - 9 * x + 4 = 0 ↔ x = r1 ∨ x = r2) : 
    abs (r1 - r2) = Real.sqrt 65 := 
sorry

end difference_of_roots_l123_123839


namespace range_of_f_l123_123384

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x

theorem range_of_f : set.range (λ x, f x) ∩ set.Icc (-2 : ℝ) 1 = set.Icc (-1 : ℝ) 3 :=
by
  sorry

end range_of_f_l123_123384


namespace georgia_coughs_5_times_per_minute_l123_123371

-- Definitions
def georgia_coughs_per_minute (G : ℕ) := true
def robert_coughs_per_minute (G : ℕ) := 2 * G
def total_coughs (G : ℕ) := 20 * (G + 2 * G) = 300

-- Theorem to prove
theorem georgia_coughs_5_times_per_minute (G : ℕ) 
  (h1 : georgia_coughs_per_minute G) 
  (h2 : robert_coughs_per_minute G = 2 * G) 
  (h3 : total_coughs G) : G = 5 := 
sorry

end georgia_coughs_5_times_per_minute_l123_123371


namespace maximum_regular_hours_is_40_l123_123947

-- Definitions based on conditions
def regular_pay_per_hour := 3
def overtime_pay_per_hour := 6
def total_payment_received := 168
def overtime_hours := 8
def overtime_earnings := overtime_hours * overtime_pay_per_hour
def regular_earnings := total_payment_received - overtime_earnings
def maximum_regular_hours := regular_earnings / regular_pay_per_hour

-- Lean theorem statement corresponding to the proof problem
theorem maximum_regular_hours_is_40 : maximum_regular_hours = 40 := by
  sorry

end maximum_regular_hours_is_40_l123_123947


namespace factorize_x4_y4_l123_123520

theorem factorize_x4_y4 (x y : ℝ) : x^4 - y^4 = (x^2 + y^2) * (x^2 - y^2) :=
by
  sorry

end factorize_x4_y4_l123_123520


namespace find_n_l123_123811

theorem find_n (n : ℝ) (h1 : ∀ m : ℝ, m = 4 → m^(m/2) = 4) : 
  n^(n/2) = 8 ↔ n = 2^Real.sqrt 6 :=
by
  sorry

end find_n_l123_123811


namespace graph_is_finite_distinct_points_l123_123391

def cost (n : ℕ) : ℕ := 18 * n + 3

theorem graph_is_finite_distinct_points : 
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 20 → 
  ∀ (m : ℕ), 1 ≤ m ∧ m ≤ 20 → 
  (cost n = cost m → n = m) ∧
  ∀ x : ℕ, ∃ n : ℕ, 1 ≤ n ∧ n ≤ 20 ∧ cost n = x :=
by
  sorry

end graph_is_finite_distinct_points_l123_123391


namespace rectangle_perimeter_l123_123926

variables (L W : ℕ)

-- conditions
def conditions : Prop :=
  L - 4 = W + 3 ∧
  (L - 4) * (W + 3) = L * W

-- prove the solution
theorem rectangle_perimeter (h : conditions L W) : 2 * L + 2 * W = 50 := sorry

end rectangle_perimeter_l123_123926


namespace sum_solutions_eq_16_l123_123410

theorem sum_solutions_eq_16 (x y : ℝ) 
  (h1 : |x - 5| = |y - 11|)
  (h2 : |x - 11| = 2 * |y - 5|)
  (h3 : x + y = 16) :
  x + y = 16 :=
by
  sorry

end sum_solutions_eq_16_l123_123410


namespace rachel_homework_difference_l123_123614

def total_difference (r m h s : ℕ) : ℕ :=
  (r - m) + (s - h)

theorem rachel_homework_difference :
    ∀ (r m h s : ℕ), r = 7 → m = 5 → h = 3 → s = 6 → total_difference r m h s = 5 :=
by
  intros r m h s hr hm hh hs
  rw [hr, hm, hh, hs]
  rfl

end rachel_homework_difference_l123_123614


namespace fraction_inequality_l123_123897

open Real

theorem fraction_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a < b + c) :
  a / (1 + a) < b / (1 + b) + c / (1 + c) :=
  sorry

end fraction_inequality_l123_123897


namespace determine_all_cards_l123_123266

noncomputable def min_cards_to_determine_positions : ℕ :=
  2

theorem determine_all_cards {k : ℕ} (h : k = min_cards_to_determine_positions) :
  ∀ (placed_cards : ℕ → ℕ × ℕ),
  (∀ n, 1 ≤ n ∧ n ≤ 300 → placed_cards n = placed_cards (n + 1) ∨ placed_cards n + (1, 0) = placed_cards (n + 1) ∨ placed_cards n + (0, 1) = placed_cards (n + 1))
  → k = 2 :=
by
  sorry

end determine_all_cards_l123_123266


namespace chenny_candies_l123_123831

def friends_count : ℕ := 7
def candies_per_friend : ℕ := 2
def candies_have : ℕ := 10

theorem chenny_candies : 
    (friends_count * candies_per_friend - candies_have) = 4 := by
    sorry

end chenny_candies_l123_123831


namespace exp_pos_for_all_x_l123_123633

theorem exp_pos_for_all_x (h : ¬ ∃ x_0 : ℝ, Real.exp x_0 ≤ 0) : ∀ x : ℝ, Real.exp x > 0 :=
by
  sorry

end exp_pos_for_all_x_l123_123633


namespace grace_mowing_hours_l123_123573

-- Definitions for conditions
def earnings_mowing (x : ℕ) : ℕ := 6 * x
def earnings_weeds : ℕ := 11 * 9
def earnings_mulch : ℕ := 9 * 10
def total_september_earnings (x : ℕ) : ℕ := earnings_mowing x + earnings_weeds + earnings_mulch

-- Proof statement (with the total earnings of 567 specified)
theorem grace_mowing_hours (x : ℕ) (h : total_september_earnings x = 567) : x = 63 := by
  sorry

end grace_mowing_hours_l123_123573


namespace max_quotient_l123_123046

theorem max_quotient (x y : ℝ) (h1 : -5 ≤ x) (h2 : x ≤ -3) (h3 : 3 ≤ y) (h4 : y ≤ 6) : 
  ∃ z, z = (x + y) / x ∧ ∀ w, w = (x + y) / x → w ≤ 0 :=
by
  sorry

end max_quotient_l123_123046


namespace find_remainder_mod_10_l123_123167

def inv_mod_10 (x : ℕ) : ℕ := 
  if x = 1 then 1 
  else if x = 3 then 7 
  else if x = 7 then 3 
  else if x = 9 then 9 
  else 0 -- invalid, not invertible

theorem find_remainder_mod_10 (a b c d : ℕ) 
  (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ d) (hd : d ≠ a) 
  (ha' : a < 10) (hb' : b < 10) (hc' : c < 10) (hd' : d < 10)
  (ha_inv : inv_mod_10 a ≠ 0) (hb_inv : inv_mod_10 b ≠ 0)
  (hc_inv : inv_mod_10 c ≠ 0) (hd_inv : inv_mod_10 d ≠ 0) :
  ((a * b * c + a * b * d + a * c * d + b * c * d) * (inv_mod_10 (a * b * c * d % 10))) % 10 = 0 :=
by
  sorry

end find_remainder_mod_10_l123_123167


namespace math_problem_l123_123861

variables {x y z a b c : ℝ}

theorem math_problem
  (h₁ : x / a + y / b + z / c = 4)
  (h₂ : a / x + b / y + c / z = 2) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 12 :=
sorry

end math_problem_l123_123861


namespace integer_solutions_l123_123995

theorem integer_solutions (m : ℤ) :
  (∃ x : ℤ, (m * x - 1) / (x - 1) = 2 + 1 / (1 - x)) → 
  (∃ x : ℝ, (m - 1) * x^2 + 2 * x + 1 / 2 = 0) →
  m = 3 :=
by
  sorry

end integer_solutions_l123_123995


namespace oliver_final_amount_is_54_04_l123_123202

noncomputable def final_amount : ℝ :=
  let initial := 33
  let feb_spent := 0.15 * initial
  let after_feb := initial - feb_spent
  let march_added := 32
  let after_march := after_feb + march_added
  let march_spent := 0.10 * after_march
  after_march - march_spent

theorem oliver_final_amount_is_54_04 : final_amount = 54.04 := by
  sorry

end oliver_final_amount_is_54_04_l123_123202


namespace not_all_prime_distinct_l123_123791

theorem not_all_prime_distinct (a1 a2 a3 : ℕ) (h1 : a1 ≠ a2) (h2 : a2 ≠ a3) (h3 : a1 ≠ a3)
  (h4 : 0 < a1) (h5 : 0 < a2) (h6 : 0 < a3)
  (h7 : a1 ∣ (a2 + a3 + a2 * a3)) (h8 : a2 ∣ (a3 + a1 + a3 * a1)) (h9 : a3 ∣ (a1 + a2 + a1 * a2)) :
  ¬ (Nat.Prime a1 ∧ Nat.Prime a2 ∧ Nat.Prime a3) :=
by
  sorry

end not_all_prime_distinct_l123_123791


namespace find_vector_p_l123_123056

noncomputable def vector_proj (v u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := v.1 * u.1 + v.2 * u.2
  let dot_u := u.1 * u.1 + u.2 * u.2
  let scale := dot_uv / dot_u
  (scale * u.1, scale * u.2)

theorem find_vector_p :
  ∃ p : ℝ × ℝ,
    vector_proj (5, -2) p = p ∧
    vector_proj (2, 6) p = p ∧
    p = (14 / 73, 214 / 73) :=
by
  sorry

end find_vector_p_l123_123056


namespace mark_collects_money_l123_123422

variable (households_per_day : Nat)
variable (days : Nat)
variable (pair_amount : Nat)
variable (half_factor : Nat)

theorem mark_collects_money
  (h1 : households_per_day = 20)
  (h2 : days = 5)
  (h3 : pair_amount = 40)
  (h4 : half_factor = 2) :
  (households_per_day * days / half_factor) * pair_amount = 2000 :=
by
  sorry

end mark_collects_money_l123_123422


namespace specific_value_eq_l123_123819

def specific_value (x : ℕ) : ℕ := 25 * x

theorem specific_value_eq : specific_value 27 = 675 := by
  sorry

end specific_value_eq_l123_123819


namespace cos_240_eq_neg_half_l123_123288

theorem cos_240_eq_neg_half (h1 : Real.cos (180 * Real.pi / 180) = -1)
                            (h2 : Real.sin (180 * Real.pi / 180) = 0)
                            (h3 : Real.cos (60 * Real.pi / 180) = 1 / 2) :
  Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_240_eq_neg_half_l123_123288


namespace question1_question2_l123_123054

def setA : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 3}

def setB (a : ℝ) : Set ℝ := {x : ℝ | abs (x - a) ≤ 1 }

def complementA : Set ℝ := {x : ℝ | x ≤ -1 ∨ x > 3}

theorem question1 : A = setA := sorry

theorem question2 (a : ℝ) : setB a ∩ complementA = setB a → a ∈ Set.union (Set.Iic (-2)) (Set.Ioi 4) := sorry

end question1_question2_l123_123054


namespace cos_240_eq_negative_half_l123_123326

theorem cos_240_eq_negative_half : Real.cos (240 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_240_eq_negative_half_l123_123326


namespace problem1_problem2_l123_123959

-- Problem 1
theorem problem1 (α : ℝ) (h : 2 * Real.sin α - Real.cos α = 0) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) + (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -10 / 3 :=
sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : Real.cos (π / 4 + x) = 3 / 5) :
  (Real.sin x ^ 3 + Real.sin x * Real.cos x ^ 2) / (1 - Real.tan x) = 7 * Real.sqrt 2 / 60 :=
sorry

end problem1_problem2_l123_123959


namespace similar_triangle_longest_side_length_l123_123237

-- Given conditions as definitions 
def originalTriangleSides (a b c : ℕ) : Prop := a = 8 ∧ b = 10 ∧ c = 12
def similarTrianglePerimeter (P : ℕ) : Prop := P = 150

-- Statement to be proved using the given conditions
theorem similar_triangle_longest_side_length (a b c P : ℕ) 
  (h1 : originalTriangleSides a b c) 
  (h2 : similarTrianglePerimeter P) : 
  ∃ x : ℕ, P = (a + b + c) * x ∧ 12 * x = 60 :=
by
  -- Proof would go here
  sorry

end similar_triangle_longest_side_length_l123_123237


namespace factor_expression_l123_123366

theorem factor_expression (x y a b : ℝ) : 
  ∃ f : ℝ, 3 * x * (a - b) - 9 * y * (b - a) = f * (x + 3 * y) ∧ f = 3 * (a - b) :=
by
  sorry

end factor_expression_l123_123366


namespace savings_calculation_l123_123501

noncomputable def calculate_savings (spent_price : ℝ) (saving_pct : ℝ) : ℝ :=
  let original_price := spent_price / (1 - (saving_pct / 100))
  original_price - spent_price

-- Define the spent price and saving percentage
def spent_price : ℝ := 20
def saving_pct : ℝ := 12.087912087912088

-- Statement to be proved
theorem savings_calculation : calculate_savings spent_price saving_pct = 2.75 :=
  sorry

end savings_calculation_l123_123501


namespace aaron_already_had_lids_l123_123275

-- Definitions for conditions
def number_of_boxes : ℕ := 3
def can_lids_per_box : ℕ := 13
def total_can_lids : ℕ := 53
def lids_from_boxes : ℕ := number_of_boxes * can_lids_per_box

-- The statement to be proven
theorem aaron_already_had_lids : total_can_lids - lids_from_boxes = 14 := 
by
  sorry

end aaron_already_had_lids_l123_123275


namespace sin_law_of_sines_l123_123200

theorem sin_law_of_sines (a b : ℝ) (sin_A sin_B : ℝ)
  (h1 : a = 3)
  (h2 : b = 4)
  (h3 : sin_A = 3 / 5) :
  sin_B = 4 / 5 := 
sorry

end sin_law_of_sines_l123_123200


namespace number_leaves_remainder_five_l123_123645

theorem number_leaves_remainder_five (k : ℕ) (n : ℕ) (least_num : ℕ) 
  (h₁ : least_num = 540)
  (h₂ : ∀ m, m % 12 = 5 → m ≥ least_num)
  (h₃ : n = 107) 
  : 540 % 107 = 5 :=
by sorry

end number_leaves_remainder_five_l123_123645


namespace calculate_Y_payment_l123_123113

theorem calculate_Y_payment (X Y : ℝ) (h1 : X + Y = 600) (h2 : X = 1.2 * Y) : Y = 600 / 2.2 :=
by
  sorry

end calculate_Y_payment_l123_123113


namespace promotional_rate_ratio_is_one_third_l123_123772

-- Define the conditions
def normal_monthly_charge : ℕ := 30
def extra_fee : ℕ := 15
def total_paid : ℕ := 175

-- Define the total data plan amount equation
def calculate_total (P : ℕ) : ℕ :=
  P + 2 * normal_monthly_charge + (normal_monthly_charge + extra_fee) + 2 * normal_monthly_charge

theorem promotional_rate_ratio_is_one_third (P : ℕ) (hP : calculate_total P = total_paid) :
  P * 3 = normal_monthly_charge :=
by sorry

end promotional_rate_ratio_is_one_third_l123_123772


namespace final_position_total_distance_l123_123696

-- Define the movements as a list
def movements : List Int := [-8, 7, -3, 9, -6, -4, 10]

-- Prove that the final position of the turtle is 5 meters north of the starting point
theorem final_position (movements : List Int) (h : movements = [-8, 7, -3, 9, -6, -4, 10]) : List.sum movements = 5 :=
by
  rw [h]
  sorry

-- Prove that the total distance crawled by the turtle is 47 meters
theorem total_distance (movements : List Int) (h : movements = [-8, 7, -3, 9, -6, -4, 10]) : List.sum (List.map Int.natAbs movements) = 47 :=
by
  rw [h]
  sorry

end final_position_total_distance_l123_123696


namespace cos_240_eq_neg_half_l123_123309

theorem cos_240_eq_neg_half : ∀ (deg: ℝ), 
  deg = 240 → 
  (∀ a b : ℝ, 240 = a + b → a = 180 → b = 60 → 
    cos 240 = cos a * cos b - sin a * sin b) → 
  cos 180 = -1 →
  sin 180 = 0 →
  cos 60 = 1 / 2 →
  cos 240 = -1 / 2 :=
by 
  intros deg h_deg h_sum h_cos_180 h_sin_180 h_cos_60
  rw h_deg at h_sum
  have h_cos_identity := h_sum 180 60 rfl rfl rfl
  simp [h_cos_180, h_sin_180, h_cos_60] at h_cos_identity
  exact h_cos_identity

end cos_240_eq_neg_half_l123_123309


namespace inequality_solution_l123_123637

theorem inequality_solution (x : ℝ) : 3 * x + 2 ≥ 5 ↔ x ≥ 1 :=
by sorry

end inequality_solution_l123_123637


namespace equation_verification_l123_123028

theorem equation_verification :
  (96 / 12 = 8) ∧ (45 - 37 = 8) := 
by
  -- We can add the necessary proofs later
  sorry

end equation_verification_l123_123028


namespace condition1_condition2_l123_123206

-- Define the point P
def P (m : ℝ) : ℝ × ℝ := (m + 1, 2 * m - 4)

-- Define the point A
def A : ℝ × ℝ := (-5, 2)

-- Condition 1: P lies on the x-axis
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

-- Condition 2: AP is parallel to the y-axis
def parallel_y_axis (a p : ℝ × ℝ) : Prop := a.1 = p.1

-- Prove the conditions
theorem condition1 (m : ℝ) (h : on_x_axis (P m)) : P m = (3, 0) :=
by
  sorry

theorem condition2 (m : ℝ) (h : parallel_y_axis A (P m)) : P m = (-5, -16) :=
by
  sorry

end condition1_condition2_l123_123206


namespace Energetics_factory_l123_123401

/-- In the country "Energetics," there are 150 factories, and some of them are connected by bus
routes that do not stop anywhere except at these factories. It turns out that any four factories
can be split into two pairs such that a bus runs between each pair of factories. Find the minimum
number of pairs of factories that can be connected by bus routes. -/
theorem Energetics_factory
  (factories : Finset ℕ) (routes : Finset (ℕ × ℕ))
  (h_factories : factories.card = 150)
  (h_routes : ∀ (X Y Z W : ℕ),
    {X, Y, Z, W} ⊆ factories →
    ∃ (X1 Y1 Z1 W1 : ℕ),
    (X1, Y1) ∈ routes ∧
    (Z1, W1) ∈ routes ∧
    (X1 = X ∨ X1 = Y ∨ X1 = Z ∨ X1 = W) ∧
    (Y1 = X ∨ Y1 = Y ∨ Y1 = Z ∨ Y1 = W) ∧
    (Z1 = X ∨ Z1 = Y ∨ Z1 = Z ∨ Z1 = W) ∧
    (W1 = X ∨ W1 = Y ∨ W1 = Z ∨ W1 = W)) :
  (2 * routes.card) ≥ 11025 := sorry

end Energetics_factory_l123_123401


namespace cos_240_eq_neg_half_l123_123305

theorem cos_240_eq_neg_half : Real.cos (4 * Real.pi / 3) = -1/2 :=
by
  sorry

end cos_240_eq_neg_half_l123_123305


namespace difference_rabbits_antelopes_l123_123084

variable (A R H W L : ℕ)
variable (x : ℕ)

def antelopes := 80
def rabbits := antelopes + x
def hyenas := (antelopes + rabbits) - 42
def wild_dogs := hyenas + 50
def leopards := rabbits / 2
def total_animals := 605

theorem difference_rabbits_antelopes
  (h1 : antelopes = 80)
  (h2 : rabbits = antelopes + x)
  (h3 : hyenas = (antelopes + rabbits) - 42)
  (h4 : wild_dogs = hyenas + 50)
  (h5 : leopards = rabbits / 2)
  (h6 : antelopes + rabbits + hyenas + wild_dogs + leopards = total_animals) : rabbits - antelopes = 70 := 
by
  -- Proof goes here
  sorry

end difference_rabbits_antelopes_l123_123084


namespace solve_trig_problem_l123_123174

theorem solve_trig_problem (α : ℝ) (h : Real.tan α = 1 / 3) :
  (Real.cos α)^2 - 2 * (Real.sin α)^2 / (Real.cos α)^2 = 7 / 9 := 
sorry

end solve_trig_problem_l123_123174


namespace isosceles_triangle_area_l123_123071

theorem isosceles_triangle_area (PQ PR QR : ℝ) (PS : ℝ) (h1 : PQ = PR)
  (h2 : QR = 10) (h3 : PS^2 + (QR / 2)^2 = PQ^2) : 
  (1/2) * QR * PS = 60 :=
by
  sorry

end isosceles_triangle_area_l123_123071


namespace seq_a_seq_b_l123_123041

theorem seq_a (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) :
  a 1 = 1 ∧ (∀ n, S (n + 1) = 3 * S n + 2) →
  (∀ n, a n = if n = 1 then 1 else 4 * 3 ^ (n - 2)) :=
by
  sorry

theorem seq_b (b : ℕ → ℕ) (a : ℕ → ℕ) (T : ℕ → ℕ) (n : ℕ) :
  (b n = 8 * n / (a (n + 1) - a n)) →
  (T n = 77 / 12 - (n / 2 + 3 / 4) * (1 / 3) ^ (n - 2)) :=
by
  sorry

end seq_a_seq_b_l123_123041


namespace category_D_cost_after_discount_is_correct_l123_123596

noncomputable def total_cost : ℝ := 2500
noncomputable def percentage_A : ℝ := 0.30
noncomputable def percentage_B : ℝ := 0.25
noncomputable def percentage_C : ℝ := 0.20
noncomputable def percentage_D : ℝ := 0.25
noncomputable def discount_A : ℝ := 0.03
noncomputable def discount_B : ℝ := 0.05
noncomputable def discount_C : ℝ := 0.07
noncomputable def discount_D : ℝ := 0.10

noncomputable def cost_before_discount_D : ℝ := total_cost * percentage_D
noncomputable def discount_amount_D : ℝ := cost_before_discount_D * discount_D
noncomputable def cost_after_discount_D : ℝ := cost_before_discount_D - discount_amount_D

theorem category_D_cost_after_discount_is_correct : cost_after_discount_D = 562.5 := 
by 
  sorry

end category_D_cost_after_discount_is_correct_l123_123596


namespace triangle_MOI_area_zero_l123_123586

noncomputable def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (6, 0)
def C : point := (0, 8)
def O : point := (3, 4)
def I : point := (2, 2)
def M : point := (1, 1)

def area_ΔMOI (M O I : point) : ℝ :=
1 / 2 * |(M.fst * (O.snd - I.snd) + O.fst * (I.snd - M.snd) + I.fst * (M.snd - O.snd))|

theorem triangle_MOI_area_zero
  (A B C O I M : point)
  (h1 : A = (0, 0))
  (h2 : B = (6, 0))
  (h3 : C = (0, 8))
  (h4 : O = (3, 4))
  (h5 : I = (2, 2))
  (h6 : M = (1, 1)) :
  area_ΔMOI M O I = 0 :=
sorry

end triangle_MOI_area_zero_l123_123586


namespace intersection_of_A_and_B_l123_123865

def A : Set ℝ := {x | x^2 - x = 0}
def B : Set ℝ := {y | y^2 + y = 0}

theorem intersection_of_A_and_B : A ∩ B = {0} :=
by
  sorry

end intersection_of_A_and_B_l123_123865


namespace vector_on_line_l123_123265

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (p q : V)

theorem vector_on_line (k : ℝ) (hpq : p ≠ q) :
  ∃ t : ℝ, k • p + (1/2 : ℝ) • q = p + t • (q - p) → k = 1/2 :=
by
  sorry

end vector_on_line_l123_123265


namespace geometric_sequence_seventh_term_l123_123941

theorem geometric_sequence_seventh_term (a r : ℕ) (h₁ : a = 6) (h₂ : a * r^4 = 486) : a * r^6 = 4374 :=
by
  -- The proof is not required, hence we use sorry.
  sorry

end geometric_sequence_seventh_term_l123_123941


namespace trig_identity_l123_123037

variable (α : ℝ)
variable (h : Real.sin α = 3 / 5)

theorem trig_identity : Real.sin (Real.pi / 2 + 2 * α) = 7 / 25 :=
by
  sorry

end trig_identity_l123_123037


namespace fewer_bees_than_flowers_l123_123455

theorem fewer_bees_than_flowers :
  (5 - 3 = 2) :=
by
  sorry

end fewer_bees_than_flowers_l123_123455


namespace smallest_int_rel_prime_150_l123_123527

theorem smallest_int_rel_prime_150 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 150 = 1 ∧ ∀ y : ℕ, (y > 1 ∧ Nat.gcd y 150 = 1) → x ≤ y :=
by
  sorry

end smallest_int_rel_prime_150_l123_123527


namespace negation_of_universal_prop_l123_123106

theorem negation_of_universal_prop:
  (¬ (∀ x : ℝ, x ^ 3 - x ≥ 0)) ↔ (∃ x : ℝ, x ^ 3 - x < 0) := 
by 
sorry

end negation_of_universal_prop_l123_123106


namespace batsman_average_increase_l123_123937

theorem batsman_average_increase (A : ℕ) 
    (h1 : 15 * A + 64 = 19 * 16) 
    (h2 : 19 - A = 3) : 
    19 - A = 3 := 
sorry

end batsman_average_increase_l123_123937


namespace negation_of_P_l123_123053

def P (x : ℝ) : Prop := x^2 + x - 1 < 0

theorem negation_of_P : (¬ ∀ x, P x) ↔ ∃ x : ℝ, x^2 + x - 1 ≥ 0 :=
by
  sorry

end negation_of_P_l123_123053


namespace range_of_a_l123_123413

noncomputable def f (x : ℝ) := Real.log (x + 1)
def A (x : ℝ) := (f (1 - 2 * x) > f x)
def B (a x : ℝ) := (a - 1 < x) ∧ (x < 2 * a^2)

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, A x ∧ B a x) ↔ (a < -1 / 2) ∨ (1 < a ∧ a < 4 / 3) :=
sorry

end range_of_a_l123_123413


namespace proportion_of_fathers_with_full_time_jobs_l123_123591

theorem proportion_of_fathers_with_full_time_jobs
  (P : ℕ) -- Total number of parents surveyed
  (mothers_proportion : ℝ := 0.4) -- Proportion of mothers in the survey
  (mothers_ftj_proportion : ℝ := 0.9) -- Proportion of mothers with full-time jobs
  (parents_no_ftj_proportion : ℝ := 0.19) -- Proportion of parents without full-time jobs
  (hfathers : ℝ := 0.6) -- Proportion of fathers in the survey
  (hfathers_ftj_proportion : ℝ) -- Proportion of fathers with full-time jobs
  : hfathers_ftj_proportion = 0.75 := 
by 
  sorry

end proportion_of_fathers_with_full_time_jobs_l123_123591


namespace two_digit_prime_sum_digits_10_count_l123_123544

-- Two-digit number definition
def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

-- Sum of digits equals 10
def digits_sum_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

-- Prime number definition
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Final Lean 4 statement
theorem two_digit_prime_sum_digits_10_count :
  { n : ℕ | two_digit n ∧ digits_sum_10 n ∧ is_prime n }.to_finset.card = 3 :=
by
  sorry

end two_digit_prime_sum_digits_10_count_l123_123544


namespace number_told_to_sasha_l123_123918

-- Defining concepts
def two_digit_number (a b : ℕ) : Prop := a < 10 ∧ b < 10 ∧ a * b ≥ 1

def product_of_digits (a b : ℕ) (P : ℕ) : Prop := P = a * b

def sum_of_digits (a b : ℕ) (S : ℕ) : Prop := S = a + b

def petya_guesses_in_three_attempts (P : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ), P = a * b ∧ P = c * d ∧ P = e * f ∧ 
  (a * b) ≠ (c * d) ∧ (a * b) ≠ (e * f) ∧ (c * d) ≠ (e * f)

def sasha_guesses_in_four_attempts (S : ℕ) : Prop :=
  ∃ (a b c d e f g h i j : ℕ), 
  S = a + b ∧ S = c + d ∧ S = e + f ∧ S = g + h ∧ S = i + j ∧
  (a + b) ≠ (c + d) ∧ (a + b) ≠ (e + f) ∧ (a + b) ≠ (g + h) ∧ (a + b) ≠ (i + j) ∧ 
  (c + d) ≠ (e + f) ∧ (c + d) ≠ (g + h) ∧ (c + d) ≠ (i + j) ∧ 
  (e + f) ≠ (g + h) ∧ (e + f) ≠ (i + j) ∧ 
  (g + h) ≠ (i + j)

theorem number_told_to_sasha : ∃ (S : ℕ), 
  ∀ (a b : ℕ), two_digit_number a b → 
  (product_of_digits a b (a * b) → petya_guesses_in_three_attempts (a * b)) → 
  (sum_of_digits a b S → sasha_guesses_in_four_attempts S) → S = 10 :=
by
  sorry

end number_told_to_sasha_l123_123918


namespace local_minimum_interval_l123_123052

-- Definitions of the function and its derivative
def y (x a : ℝ) : ℝ := x^3 - 2 * a * x + a
def y_prime (x a : ℝ) : ℝ := 3 * x^2 - 2 * a

-- The proof problem statement
theorem local_minimum_interval (a : ℝ) : 
  (0 < a ∧ a < 3 / 2) ↔ ∃ (x : ℝ), (0 < x ∧ x < 1) ∧ y_prime x a = 0 :=
sorry

end local_minimum_interval_l123_123052


namespace correct_operation_is_C_l123_123122

/--
Given the following statements:
1. \( a^3 \cdot a^2 = a^6 \)
2. \( (2a^3)^3 = 6a^9 \)
3. \( -6x^5 \div 2x^3 = -3x^2 \)
4. \( (-x-2)(x-2) = x^2 - 4 \)

Prove that the correct statement is \( -6x^5 \div 2x^3 = -3x^2 \) and the other statements are incorrect.
-/
theorem correct_operation_is_C (a x : ℝ) : 
  (a^3 * a^2 ≠ a^6) ∧
  ((2 * a^3)^3 ≠ 6 * a^9) ∧
  (-6 * x^5 / (2 * x^3) = -3 * x^2) ∧
  ((-x - 2) * (x - 2) ≠ x^2 - 4) := by
  sorry

end correct_operation_is_C_l123_123122


namespace coordinates_of_P_l123_123581

structure Point (α : Type) [LinearOrderedField α] :=
  (x : α)
  (y : α)

def in_fourth_quadrant {α : Type} [LinearOrderedField α] (P : Point α) : Prop :=
  P.x > 0 ∧ P.y < 0

def distance_to_axes_is_4 {α : Type} [LinearOrderedField α] (P : Point α) : Prop :=
  abs P.x = 4 ∧ abs P.y = 4

theorem coordinates_of_P {α : Type} [LinearOrderedField α] (P : Point α) :
  in_fourth_quadrant P ∧ distance_to_axes_is_4 P → P = ⟨4, -4⟩ :=
by
  sorry

end coordinates_of_P_l123_123581


namespace find_m_n_l123_123990

theorem find_m_n :
  ∀ (m n : ℤ), (∀ x : ℤ, (x - 4) * (x + 8) = x^2 + m * x + n) → 
  (m = 4 ∧ n = -32) :=
by
  intros m n h
  let x := 0
  sorry

end find_m_n_l123_123990


namespace find_B_l123_123760

-- Define the translation function for points in ℝ × ℝ.
def translate (p : ℝ × ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + v.1, p.2 + v.2)

-- Given conditions
def A : ℝ × ℝ := (2, 2)
def A' : ℝ × ℝ := (-2, -2)
def B : ℝ × ℝ := (-1, 1)

-- The vector v representing the translation from A to A'
def v : ℝ × ℝ := (A'.1 - A.1, A'.2 - A.2)

-- Proving the coordinates of B' after applying the same translation vector v to B
theorem find_B' : translate B v = (-5, -3) :=
by
  -- translation function needs to be instantiated with the correct values.
  -- Since this is just a Lean 4 statement, we'll not include the proof here and leave it as a sorry.
  sorry

end find_B_l123_123760


namespace cos_240_eq_negative_half_l123_123327

theorem cos_240_eq_negative_half : Real.cos (240 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_240_eq_negative_half_l123_123327


namespace correct_number_of_statements_l123_123762

theorem correct_number_of_statements (a b : ℤ) :
  (¬ (∃ h₁ : Even (a + 5 * b), ¬ Even (a - 7 * b)) ∧
   ∃ h₂ : a + b % 3 = 0, ¬ ((a % 3 = 0) ∧ (b % 3 = 0)) ∧
   ∃ h₃ : Prime (a + b), Prime (a - b)) →
   1 = 1 :=
by
  sorry

end correct_number_of_statements_l123_123762


namespace profit_percentage_is_ten_l123_123695

-- Define the cost price (CP) and selling price (SP) as constants
def CP : ℝ := 90.91
def SP : ℝ := 100

-- Define a theorem to prove the profit percentage is 10%
theorem profit_percentage_is_ten : ((SP - CP) / CP) * 100 = 10 := 
by 
  -- Skip the proof.
  sorry

end profit_percentage_is_ten_l123_123695


namespace cos_240_eq_neg_half_l123_123348

open Real

theorem cos_240_eq_neg_half : cos (240 * π / 180) = -1/2 :=
by
  -- Step 1: Decompose the angle 240° = 180° + 60°
  have h1 : 240 * π / 180 = π + 60 * π / 180,
  { 
    norm_num, 
    field_simp, 
    linarith 
  },
  -- Step 2: Use the fact that the cosine of (π + θ) = - cos(θ)
  rw [h1, cos_add_pi],
  -- Step 3: Given that cos(60°) = 1/2
  have h2 : cos (60 * π / 180) = 1/2,
  {
    norm_num,
    exact Real.cos_pi_div_three (),
  },
  -- Conclude that cos(240°) = -1/2
  rw h2,
  norm_num

end cos_240_eq_neg_half_l123_123348


namespace paula_remaining_money_l123_123609

theorem paula_remaining_money 
  (M : Int) (C_s : Int) (N_s : Int) (C_p : Int) (N_p : Int)
  (h1 : M = 250) 
  (h2 : C_s = 15) 
  (h3 : N_s = 5) 
  (h4 : C_p = 25) 
  (h5 : N_p = 3) : 
  M - (C_s * N_s + C_p * N_p) = 100 := 
by
  sorry

end paula_remaining_money_l123_123609


namespace average_weight_of_girls_l123_123925

theorem average_weight_of_girls :
  ∀ (total_students boys girls total_weight class_average_weight boys_average_weight girls_average_weight : ℝ),
  total_students = 25 →
  boys = 15 →
  girls = 10 →
  boys + girls = total_students →
  class_average_weight = 45 →
  boys_average_weight = 48 →
  total_weight = 1125 →
  girls_average_weight = (total_weight - (boys * boys_average_weight)) / girls →
  total_weight = class_average_weight * total_students →
  girls_average_weight = 40.5 :=
by
  intros total_students boys girls total_weight class_average_weight boys_average_weight girls_average_weight
  sorry

end average_weight_of_girls_l123_123925


namespace printer_z_time_l123_123429

theorem printer_z_time (T_X T_Y T_Z : ℝ) (hZX_Y : T_X = 2.25 * (T_Y + T_Z)) 
  (hX : T_X = 15) (hY : T_Y = 10) : T_Z = 20 :=
by
  rw [hX, hY] at hZX_Y
  sorry

end printer_z_time_l123_123429


namespace range_of_a_union_B_eq_A_range_of_a_inter_B_eq_empty_l123_123188

open Set

noncomputable def A (a : ℝ) : Set ℝ := { x : ℝ | a - 1 < x ∧ x < 2 * a + 1 }
def B : Set ℝ := { x : ℝ | 0 < x ∧ x < 1 }

theorem range_of_a_union_B_eq_A (a : ℝ) :
  (A a ∪ B) = A a ↔ (0 ≤ a ∧ a ≤ 1) := by
  sorry

theorem range_of_a_inter_B_eq_empty (a : ℝ) :
  (A a ∩ B) = ∅ ↔ (a ≤ - 1 / 2 ∨ 2 ≤ a) := by
  sorry

end range_of_a_union_B_eq_A_range_of_a_inter_B_eq_empty_l123_123188


namespace john_rental_weeks_l123_123083

noncomputable def camera_value : ℝ := 5000
noncomputable def rental_fee_rate : ℝ := 0.10
noncomputable def friend_payment_rate : ℝ := 0.40
noncomputable def john_total_payment : ℝ := 1200

theorem john_rental_weeks :
  let weekly_rental_fee := camera_value * rental_fee_rate
  let friend_payment := weekly_rental_fee * friend_payment_rate
  let john_weekly_payment := weekly_rental_fee - friend_payment
  let rental_weeks := john_total_payment / john_weekly_payment
  rental_weeks = 4 :=
by
  -- Place for proof steps
  sorry

end john_rental_weeks_l123_123083


namespace sequence_general_term_l123_123850

noncomputable def a₁ : ℕ → ℚ := sorry

variable (S : ℕ → ℚ)

axiom h₀ : a₁ 1 = -1
axiom h₁ : ∀ n : ℕ, a₁ (n + 1) = S n * S (n + 1)

theorem sequence_general_term (n : ℕ) : S n = -1 / n := by
  sorry

end sequence_general_term_l123_123850


namespace arithmetic_geometric_sequence_problem_l123_123048

variable {n : ℕ}

def a (n : ℕ) : ℕ := 3 * n - 1
def b (n : ℕ) : ℕ := 2 ^ n
def S (n : ℕ) : ℕ := n * (2 + (2 + (n - 1) * (3 - 1))) / 2 -- sum of an arithmetic sequence
def T (n : ℕ) : ℕ := (3 * n - 4) * 2 ^ (n + 1) + 8

theorem arithmetic_geometric_sequence_problem :
  (a 1 = 2) ∧ (b 1 = 2) ∧ (a 4 + b 4 = 27) ∧ (S 4 - b 4 = 10) →
  (∀ n, T n = (3 * n - 4) * 2 ^ (n + 1) + 8) := sorry

end arithmetic_geometric_sequence_problem_l123_123048


namespace no_solution_pos_integers_l123_123974

theorem no_solution_pos_integers (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a + b + c + d - 3 ≠ a * b + c * d := 
by
  sorry

end no_solution_pos_integers_l123_123974


namespace non_neg_int_solutions_l123_123712

theorem non_neg_int_solutions (n : ℕ) (a b : ℤ) :
  n^2 = a + b ∧ n^3 = a^2 + b^2 → n = 0 ∨ n = 1 ∨ n = 2 :=
by
  sorry

end non_neg_int_solutions_l123_123712


namespace field_day_difference_l123_123878

theorem field_day_difference :
  let girls_4th_first := 12
  let boys_4th_first := 13
  let girls_4th_second := 15
  let boys_4th_second := 11
  let girls_5th_first := 9
  let boys_5th_first := 13
  let girls_5th_second := 10
  let boys_5th_second := 11
  let total_girls := girls_4th_first + girls_4th_second + girls_5th_first + girls_5th_second
  let total_boys := boys_4th_first + boys_4th_second + boys_5th_first + boys_5th_second
  total_boys - total_girls = 2 :=
by
  let girls_4th_first := 12
  let boys_4th_first := 13
  let girls_4th_second := 15
  let boys_4th_second := 11
  let girls_5th_first := 9
  let boys_5th_first := 13
  let girls_5th_second := 10
  let boys_5th_second := 11
  let total_girls := girls_4th_first + girls_4th_second + girls_5th_first + girls_5th_second
  let total_boys := boys_4th_first + boys_4th_second + boys_5th_first + boys_5th_second
  have h1 : total_girls = 46 := rfl
  have h2 : total_boys = 48 := rfl
  have h3 : total_boys - total_girls = 2 := rfl
  exact h3

end field_day_difference_l123_123878


namespace fewer_bees_than_flowers_l123_123457

theorem fewer_bees_than_flowers : 5 - 3 = 2 := by
  sorry

end fewer_bees_than_flowers_l123_123457


namespace cos_240_eq_neg_half_l123_123315

theorem cos_240_eq_neg_half : ∀ (deg: ℝ), 
  deg = 240 → 
  (∀ a b : ℝ, 240 = a + b → a = 180 → b = 60 → 
    cos 240 = cos a * cos b - sin a * sin b) → 
  cos 180 = -1 →
  sin 180 = 0 →
  cos 60 = 1 / 2 →
  cos 240 = -1 / 2 :=
by 
  intros deg h_deg h_sum h_cos_180 h_sin_180 h_cos_60
  rw h_deg at h_sum
  have h_cos_identity := h_sum 180 60 rfl rfl rfl
  simp [h_cos_180, h_sin_180, h_cos_60] at h_cos_identity
  exact h_cos_identity

end cos_240_eq_neg_half_l123_123315


namespace A_eq_B_l123_123418

open Set

def A := {x | ∃ a : ℝ, x = 5 - 4 * a + a ^ 2}
def B := {y | ∃ b : ℝ, y = 4 * b ^ 2 + 4 * b + 2}

theorem A_eq_B : A = B := sorry

end A_eq_B_l123_123418


namespace sam_initial_money_l123_123779

theorem sam_initial_money :
  (9 * 7 + 16 = 79) :=
by
  sorry

end sam_initial_money_l123_123779


namespace no_real_roots_range_a_l123_123199

theorem no_real_roots_range_a (a : ℝ) : (¬∃ x : ℝ, 2 * x^2 + (a - 5) * x + 2 = 0) → 1 < a ∧ a < 9 :=
by
  sorry

end no_real_roots_range_a_l123_123199


namespace cos_240_eq_neg_half_l123_123287

theorem cos_240_eq_neg_half (h1 : Real.cos (180 * Real.pi / 180) = -1)
                            (h2 : Real.sin (180 * Real.pi / 180) = 0)
                            (h3 : Real.cos (60 * Real.pi / 180) = 1 / 2) :
  Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_240_eq_neg_half_l123_123287


namespace most_reasonable_sampling_method_l123_123072

-- Definitions for the conditions
def significant_difference_by_stage : Prop := 
  -- There is a significant difference in vision condition at different educational stages
  sorry

def no_significant_difference_by_gender : Prop :=
  -- There is no significant difference in vision condition between male and female students
  sorry

-- Theorem statement
theorem most_reasonable_sampling_method 
  (h1 : significant_difference_by_stage) 
  (h2 : no_significant_difference_by_gender) : 
  -- The most reasonable sampling method is stratified sampling by educational stage
  sorry :=
by
  -- Proof skipped
  sorry

end most_reasonable_sampling_method_l123_123072


namespace two_digit_primes_with_digit_sum_ten_l123_123533

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, 2 ≤ m → m * m ≤ n → n % m ≠ 0

theorem two_digit_primes_with_digit_sum_ten :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
by
  sorry

end two_digit_primes_with_digit_sum_ten_l123_123533


namespace billy_restaurant_bill_l123_123679

def adults : ℕ := 2
def children : ℕ := 5
def meal_cost : ℕ := 3

def total_people : ℕ := adults + children
def total_bill : ℕ := total_people * meal_cost

theorem billy_restaurant_bill : total_bill = 21 := 
by
  -- This is the placeholder for the proof.
  sorry

end billy_restaurant_bill_l123_123679


namespace intersects_negative_half_axis_range_l123_123852

noncomputable def f (m x : ℝ) : ℝ :=
  (m - 2) * x^2 - 4 * m * x + 2 * m - 6

theorem intersects_negative_half_axis_range (m : ℝ) :
  (1 ≤ m ∧ m < 2) ∨ (2 < m ∧ m < 3) ↔ (∃ x : ℝ, f m x < 0) :=
sorry

end intersects_negative_half_axis_range_l123_123852


namespace correct_fraction_statement_l123_123808

theorem correct_fraction_statement (x : ℝ) :
  (∀ a b : ℝ, (-a) / (-b) = a / b) ∧
  (¬ (∀ a : ℝ, a / 0 = 0)) ∧
  (∀ a b : ℝ, b ≠ 0 → (a * b) / (c * b) = a / c) → 
  ((∃ (a b : ℝ), a = 0 → a / b = 0) ∧ 
   (∀ (a b : ℝ), (a * k) / (b * k) = a / b) ∧ 
   (∀ (a b : ℝ), (-a) / (-b) = a / b) ∧ 
   (x < 1 → (|2 - x| + x) / 2 ≠ 0) 
  -> (∀ (a b : ℝ), (-a) / (-b) = a / b)) :=
by sorry

end correct_fraction_statement_l123_123808


namespace find_rabbits_l123_123067

theorem find_rabbits (heads rabbits chickens : ℕ) (h1 : rabbits + chickens = 40) (h2 : 4 * rabbits = 10 * 2 * chickens - 8) : rabbits = 33 :=
by
  -- We skip the proof here
  sorry

end find_rabbits_l123_123067


namespace value_of_y_when_x_is_zero_l123_123567

noncomputable def quadratic_y (h x : ℝ) : ℝ := -(x + h)^2

theorem value_of_y_when_x_is_zero :
  ∀ (h : ℝ), (∀ x, x < -3 → quadratic_y h x < quadratic_y h (-3)) →
            (∀ x, x > -3 → quadratic_y h x < quadratic_y h (-3)) →
            quadratic_y h 0 = -9 :=
by
  sorry

end value_of_y_when_x_is_zero_l123_123567


namespace time_to_traverse_nth_mile_l123_123693

theorem time_to_traverse_nth_mile (n : ℕ) (h : n ≥ 3) : ∃ t : ℕ, t = (n - 2)^2 :=
by
  -- Given:
  -- Speed varies inversely as the square of the number of miles already traveled.
  -- Speed is constant for each mile.
  -- The third mile is traversed in 4 hours.
  -- Show that:
  -- The time to traverse the nth mile is (n - 2)^2 hours.
  sorry

end time_to_traverse_nth_mile_l123_123693


namespace roots_of_quadratic_serve_as_eccentricities_l123_123242

theorem roots_of_quadratic_serve_as_eccentricities :
  ∀ (x1 x2 : ℝ), x1 * x2 = 1 ∧ x1 + x2 = 79 → (x1 > 1 ∧ x2 < 1) → 
  (x1 > 1 ∧ x2 < 1) ∧ x1 > 1 ∧ x2 < 1 :=
by
  sorry

end roots_of_quadratic_serve_as_eccentricities_l123_123242


namespace kathryn_remaining_money_l123_123881

variables (rent food_travel salary monthly_expenses remaining : ℝ)

-- Conditions
def rent_value := rent = 1200
def food_travel_expenses := food_travel = 2 * rent
def salary_value := salary = 5000
def shared_rent := monthly_expenses = rent / 2 + food_travel

-- Question and Answer
def money_remaining := remaining = salary - monthly_expenses

-- Theorem to prove
theorem kathryn_remaining_money (h1 : rent_value) (h2 : food_travel_expenses) (h3 : salary_value) (h4 : shared_rent) : money_remaining :=
sorry

end kathryn_remaining_money_l123_123881


namespace A_more_than_B_l123_123136

variable (A B C : ℝ)

-- Conditions
def condition1 : Prop := A = (1/3) * (B + C)
def condition2 : Prop := B = (2/7) * (A + C)
def condition3 : Prop := A + B + C = 1080

-- Conclusion
theorem A_more_than_B (A B C : ℝ) (h1 : condition1 A B C) (h2 : condition2 A B C) (h3 : condition3 A B C) :
  A - B = 30 :=
sorry

end A_more_than_B_l123_123136


namespace drawing_red_ball_is_certain_l123_123477

-- Conditions
def event_waiting_by_stump : Event := sorry -- the event cannot be quantified as certain
def event_prob_0_0001 : Event := sorry -- an event with a probability of 0.0001
def event_drawing_red_ball : Event := sorry -- drawing a red ball from a bag containing only 5 red balls
def event_flipping_coin_20_times : Event := sorry -- flipping a fair coin 20 times

-- Probabilities
axiom prob_event_drawing_red_ball : P event_drawing_red_ball = 1

-- Definition of certain event
def is_certain_event (e : Event) : Prop := P e = 1

-- Proof Statement (without proof body)
theorem drawing_red_ball_is_certain :
  is_certain_event event_drawing_red_ball :=
by {
  exact prob_event_drawing_red_ball
}

end drawing_red_ball_is_certain_l123_123477


namespace water_cost_function_solve_for_x_and_payments_l123_123398

def water_usage_A (x : ℕ) : ℕ := 5 * x
def water_usage_B (x : ℕ) : ℕ := 3 * x

def water_payment_A (x : ℕ) : ℕ :=
  if water_usage_A x <= 15 then 
    water_usage_A x * 2 
  else 
    15 * 2 + (water_usage_A x - 15) * 3

def water_payment_B (x : ℕ) : ℕ :=
  if water_usage_B x <= 15 then 
    water_usage_B x * 2 
  else 
    15 * 2 + (water_usage_B x - 15) * 3

def total_payment (x : ℕ) : ℕ := water_payment_A x + water_payment_B x

theorem water_cost_function (x : ℕ) : total_payment x =
  if 0 < x ∧ x ≤ 3 then 16 * x
  else if 3 < x ∧ x ≤ 5 then 21 * x - 15
  else if 5 < x then 24 * x - 30
  else 0 := sorry

theorem solve_for_x_and_payments (y : ℕ) : y = 114 → ∃ x, total_payment x = y ∧
  water_usage_A x = 30 ∧ water_payment_A x = 75 ∧
  water_usage_B x = 18 ∧ water_payment_B x = 39 := sorry

end water_cost_function_solve_for_x_and_payments_l123_123398


namespace bags_needed_l123_123097

-- Definitions for the condition
def total_sugar : ℝ := 35.5
def bag_capacity : ℝ := 0.5

-- Theorem statement to solve the problem
theorem bags_needed : total_sugar / bag_capacity = 71 := 
by 
  sorry

end bags_needed_l123_123097


namespace time_difference_l123_123640

/-- The time on a digital clock is 5:55. We need to calculate the number
of minutes that will pass before the clock next shows a time with all digits identical,
which is 11:11. -/
theorem time_difference : 
  let t1 := 5 * 60 + 55  -- Time 5:55 in minutes past midnight
  let t2 := 11 * 60 + 11 -- Time 11:11 in minutes past midnight
  in t2 - t1 = 316 := 
by 
  let t1 := 5 * 60 + 55 
  let t2 := 11 * 60 + 11 
  have h : t2 - t1 = 316 := by sorry
  exact h

end time_difference_l123_123640


namespace theater_ticket_lineup_l123_123631

theorem theater_ticket_lineup (eight_people : Fin 8)
  (two_standing_together : (Fin 8) × (Fin 8))
  (h : two_standing_together.1 ≠ two_standing_together.2) : 
  ∃ n : ℕ, n = 10080 := by
  let group_of_two := [two_standing_together.1, two_standing_together.2]
  have grouped_people := (8 - 1) -- 7! permutations of groups
  let arrange_two := 2 -- 2! permutations within the group
  have total_ways := (grouped_people.factors.prod * arrange_two.factors.prod)
  exact ⟨ total_ways, sorry ⟩

end theater_ticket_lineup_l123_123631


namespace abs_eq_two_implies_l123_123060

theorem abs_eq_two_implies (x : ℝ) (h : |x - 3| = 2) : x = 5 ∨ x = 1 := 
sorry

end abs_eq_two_implies_l123_123060


namespace weight_of_one_liter_ghee_brand_b_l123_123452

theorem weight_of_one_liter_ghee_brand_b (wa w_mix : ℕ) (vol_a vol_b : ℕ) (w_mix_total : ℕ) (wb : ℕ) :
  wa = 900 ∧ vol_a = 3 ∧ vol_b = 2 ∧ w_mix = 3360 →
  (vol_a * wa + vol_b * wb = w_mix →
  wb = 330) :=
by
  intros h_eq h_eq2
  obtain ⟨h_wa, h_vol_a, h_vol_b, h_w_mix⟩ := h_eq
  rw [h_wa, h_vol_a, h_vol_b, h_w_mix] at h_eq2
  sorry

end weight_of_one_liter_ghee_brand_b_l123_123452


namespace arithmetic_expression_evaluation_l123_123704

theorem arithmetic_expression_evaluation :
  1325 + (180 / 60) * 3 - 225 = 1109 :=
by
  sorry -- To be filled with the proof steps

end arithmetic_expression_evaluation_l123_123704


namespace max_sum_of_squares_l123_123085

theorem max_sum_of_squares (a b c d : ℝ) 
  (h1 : a + b = 17) 
  (h2 : ab + c + d = 85) 
  (h3 : ad + bc = 196) 
  (h4 : cd = 120) : 
  ∃ (a b c d : ℝ), a^2 + b^2 + c^2 + d^2 = 918 :=
by {
  sorry
}

end max_sum_of_squares_l123_123085


namespace position_after_steps_l123_123893

def equally_spaced_steps (total_distance num_steps distance_per_step steps_taken : ℕ) : Prop :=
  total_distance = num_steps * distance_per_step ∧ 
  ∀ k : ℕ, k ≤ num_steps → k * distance_per_step = distance_per_step * k

theorem position_after_steps (total_distance num_steps distance_per_step steps_taken : ℕ) 
  (h_eq : equally_spaced_steps total_distance num_steps distance_per_step steps_taken) 
  (h_total : total_distance = 32) (h_num : num_steps = 8) (h_steps : steps_taken = 6) : 
  steps_taken * (total_distance / num_steps) = 24 := 
by 
  sorry

end position_after_steps_l123_123893


namespace ice_cream_weekend_total_l123_123954

theorem ice_cream_weekend_total 
  (f : ℝ) (r : ℝ) (n : ℕ)
  (h_friday : f = 3.25)
  (h_saturday_reduction : r = 0.25)
  (h_num_people : n = 4)
  (h_saturday : (f - r * n) = 2.25)
  (h_sunday : 2 * ((f - r * n) / n) * n = 4.5) :
  f + (f - r * n) + (2 * ((f - r * n) / n) * n) = 10 := sorry

end ice_cream_weekend_total_l123_123954


namespace relationship_between_products_l123_123748

variable {a₁ a₂ b₁ b₂ : ℝ}

theorem relationship_between_products (h₁ : a₁ < a₂) (h₂ : b₁ < b₂) : a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := 
sorry

end relationship_between_products_l123_123748


namespace volleyball_team_lineup_l123_123775

theorem volleyball_team_lineup : 
  let team_members := 10
  let lineup_positions := 6
  10 * 9 * 8 * 7 * 6 * 5 = 151200 := by sorry

end volleyball_team_lineup_l123_123775


namespace range_of_a_l123_123798

theorem range_of_a (a : ℝ) (x : ℝ) : (x > a ∧ x > 1) → (x > 1) → (a ≤ 1) :=
by 
  intros hsol hx
  sorry

end range_of_a_l123_123798


namespace similar_triangle_longest_side_length_l123_123238

-- Given conditions as definitions 
def originalTriangleSides (a b c : ℕ) : Prop := a = 8 ∧ b = 10 ∧ c = 12
def similarTrianglePerimeter (P : ℕ) : Prop := P = 150

-- Statement to be proved using the given conditions
theorem similar_triangle_longest_side_length (a b c P : ℕ) 
  (h1 : originalTriangleSides a b c) 
  (h2 : similarTrianglePerimeter P) : 
  ∃ x : ℕ, P = (a + b + c) * x ∧ 12 * x = 60 :=
by
  -- Proof would go here
  sorry

end similar_triangle_longest_side_length_l123_123238


namespace f_zero_f_positive_all_f_increasing_f_range_l123_123023

universe u

noncomputable def f : ℝ → ℝ := sorry

axiom f_nonzero : f 0 ≠ 0
axiom f_positive : ∀ x : ℝ, 0 < x → f x > 1
axiom f_add_prop : ∀ a b : ℝ, f (a + b) = f a * f b

-- Problem 1: Prove that f(0) = 1
theorem f_zero : f 0 = 1 := sorry

-- Problem 2: Prove that for any x in ℝ, f(x) > 0
theorem f_positive_all (x : ℝ) : f x > 0 := sorry

-- Problem 3: Prove that f(x) is an increasing function on ℝ
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := sorry

-- Problem 4: Given f(x) * f(2x - x²) > 1, find the range of x
theorem f_range (x : ℝ) (h : f x * f (2*x - x^2) > 1) : 0 < x ∧ x < 3 := sorry

end f_zero_f_positive_all_f_increasing_f_range_l123_123023


namespace compute_expr_l123_123708

theorem compute_expr : 5^2 - 3 * 4 + 3^2 = 22 := by
  sorry

end compute_expr_l123_123708


namespace lines_parallel_iff_m_eq_1_l123_123853

-- Define the two lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := x + (1 + m) * y = 2 - m
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * m * x + 4 * y = -16

-- Parallel lines condition
def parallel_condition (m : ℝ) : Prop := (1 * 4 - 2 * m * (1 + m) = 0) ∧ (1 * 16 - 2 * m * (m - 2) ≠ 0)

-- The theorem to prove
theorem lines_parallel_iff_m_eq_1 (m : ℝ) : l1 m = l2 m → parallel_condition m → m = 1 :=
by 
  sorry

end lines_parallel_iff_m_eq_1_l123_123853


namespace salaries_proof_l123_123018

-- Define salaries as real numbers
variables (a b c d : ℝ)

-- Define assumptions
def conditions := 
  (a + b + c + d = 4000) ∧
  (0.05 * a + 0.15 * b = c) ∧ 
  (0.25 * d = 0.3 * b) ∧
  (b = 3 * c)

-- Define the solution as found
def solution :=
  (a = 2365.55) ∧
  (b = 645.15) ∧
  (c = 215.05) ∧
  (d = 774.18)

-- Prove that given the conditions, the solution holds
theorem salaries_proof : 
  (conditions a b c d) → (solution a b c d) := by
  sorry

end salaries_proof_l123_123018


namespace curve_C1_parametric_equiv_curve_C2_general_equiv_curve_C3_rectangular_equiv_max_distance_C2_to_C3_l123_123209

-- Definitions of the curves
def curve_C1 (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 1
def curve_C2_parametric (theta : ℝ) (x y : ℝ) : Prop := (x = 4 * Real.cos theta) ∧ (y = 3 * Real.sin theta)
def curve_C3_polar (rho theta : ℝ) : Prop := rho * (Real.cos theta - 2 * Real.sin theta) = 7

-- Proving the mathematical equivalence:
theorem curve_C1_parametric_equiv (t : ℝ) : ∃ x y, curve_C1 x y ∧ (x = 3 + Real.cos t) ∧ (y = 2 + Real.sin t) :=
by sorry

theorem curve_C2_general_equiv (x y : ℝ) : (∃ theta, curve_C2_parametric theta x y) ↔ (x^2 / 16 + y^2 / 9 = 1) :=
by sorry

theorem curve_C3_rectangular_equiv (x y : ℝ) : (∃ rho theta, x = rho * Real.cos theta ∧ y = rho * Real.sin theta ∧ curve_C3_polar rho theta) ↔ (x - 2 * y - 7 = 0) :=
by sorry

theorem max_distance_C2_to_C3 : ∃ (d : ℝ), d = (2 * Real.sqrt 65 + 7 * Real.sqrt 5) / 5 :=
by sorry

end curve_C1_parametric_equiv_curve_C2_general_equiv_curve_C3_rectangular_equiv_max_distance_C2_to_C3_l123_123209


namespace cosine_240_l123_123356

theorem cosine_240 (h1 : Real.cos 60 = 1 / 2) : Real.cos 240 = -1 / 2 :=
by
  have h2 : Real.cos 240 = -Real.cos 60 := by sorry
  rw [h2, h1]
  exact neg_div
  sorry

end cosine_240_l123_123356


namespace positively_correlated_variables_l123_123459

-- Define all conditions given in the problem
def weightOfCarVar1 : Type := ℝ
def avgDistPerLiter : Type := ℝ
def avgStudyTime : Type := ℝ
def avgAcademicPerformance : Type := ℝ
def dailySmokingAmount : Type := ℝ
def healthCondition : Type := ℝ
def sideLength : Type := ℝ
def areaOfSquare : Type := ℝ
def fuelConsumptionPerHundredKm : Type := ℝ

-- Define the relationship status between variables
def isPositivelyCorrelated (x y : Type) : Prop := sorry
def isFunctionallyRelated (x y : Type) : Prop := sorry

axiom weight_car_distance_neg : ¬ isPositivelyCorrelated weightOfCarVar1 avgDistPerLiter
axiom study_time_performance_pos : isPositivelyCorrelated avgStudyTime avgAcademicPerformance
axiom smoking_health_neg : ¬ isPositivelyCorrelated dailySmokingAmount healthCondition
axiom side_area_func : isFunctionallyRelated sideLength areaOfSquare
axiom car_weight_fuel_pos : isPositivelyCorrelated weightOfCarVar1 fuelConsumptionPerHundredKm

-- The proof statement to prove C is the correct answer
theorem positively_correlated_variables:
  isPositivelyCorrelated avgStudyTime avgAcademicPerformance ∧
  isPositivelyCorrelated weightOfCarVar1 fuelConsumptionPerHundredKm :=
by
  sorry

end positively_correlated_variables_l123_123459


namespace cos_240_is_neg_half_l123_123342

noncomputable def cos_240 : ℝ := Real.cos (240 * Real.pi / 180)

theorem cos_240_is_neg_half : cos_240 = -1/2 := by
  sorry

end cos_240_is_neg_half_l123_123342


namespace cosine_240_l123_123357

theorem cosine_240 (h1 : Real.cos 60 = 1 / 2) : Real.cos 240 = -1 / 2 :=
by
  have h2 : Real.cos 240 = -Real.cos 60 := by sorry
  rw [h2, h1]
  exact neg_div
  sorry

end cosine_240_l123_123357


namespace complementary_angles_ratio_4_to_1_smaller_angle_l123_123462

theorem complementary_angles_ratio_4_to_1_smaller_angle :
  ∃ (θ : ℝ), (4 * θ + θ = 90) ∧ (θ = 18) :=
by
  sorry

end complementary_angles_ratio_4_to_1_smaller_angle_l123_123462


namespace solve_system_eq_l123_123903

theorem solve_system_eq (x y z : ℤ) :
  (x^2 - 23 * y + 66 * z + 612 = 0) ∧ 
  (y^2 + 62 * x - 20 * z + 296 = 0) ∧ 
  (z^2 - 22 * x + 67 * y + 505 = 0) →
  (x = -20) ∧ (y = -22) ∧ (z = -23) :=
by {
  sorry
}

end solve_system_eq_l123_123903


namespace alpha_half_quadrant_l123_123562

open Real

theorem alpha_half_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * π - π / 2 < α ∧ α < 2 * k * π) :
  (∃ k1 : ℤ, (2 * k1 + 1) * π - π / 4 < α / 2 ∧ α / 2 < (2 * k1 + 1) * π) ∨
  (∃ k2 : ℤ, 2 * k2 * π - π / 4 < α / 2 ∧ α / 2 < 2 * k2 * π) :=
sorry

end alpha_half_quadrant_l123_123562


namespace seven_searchlights_shadow_length_l123_123096

noncomputable def searchlight_positioning (n : ℕ) (angle : ℝ) (shadow_length : ℝ) : Prop :=
  ∃ (positions : Fin n → ℝ × ℝ), ∀ i : Fin n, ∃ shadow : ℝ, shadow = shadow_length ∧
  (∀ j : Fin n, i ≠ j → ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi ∧
  θ - angle / 2 < θ ∧ θ + angle / 2 > θ → shadow = shadow_length)

theorem seven_searchlights_shadow_length :
  searchlight_positioning 7 (Real.pi / 2) 7000 :=
sorry

end seven_searchlights_shadow_length_l123_123096


namespace find_a2_plus_b2_l123_123439

theorem find_a2_plus_b2 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h: 8 * a^a * b^b = 27 * a^b * b^a) : a^2 + b^2 = 117 := by
  sorry

end find_a2_plus_b2_l123_123439


namespace parallelogram_opposite_sides_equal_l123_123448

-- Given definitions and properties of a parallelogram
structure Parallelogram (α : Type*) [Add α] [AddCommGroup α] [Module ℝ α] :=
(a b c d : α) 
(parallel_a : a + b = c + d)
(parallel_b : b + c = d + a)
(parallel_c : c + d = a + b)
(parallel_d : d + a = b + c)

open Parallelogram

-- Define problem statement to prove opposite sides are equal
theorem parallelogram_opposite_sides_equal {α : Type*} [Add α] [AddCommGroup α] [Module ℝ α] 
  (p : Parallelogram α) : 
  p.a = p.c ∧ p.b = p.d :=
sorry -- Proof goes here

end parallelogram_opposite_sides_equal_l123_123448


namespace b_n_plus_1_eq_2a_n_l123_123146

/-- Definition of binary sequences of length n that do not contain 0, 1, 0 -/
def a_n (n : ℕ) : ℕ := -- specify the actual counting function, placeholder below
  sorry

/-- Definition of binary sequences of length n that do not contain 0, 0, 1, 1 or 1, 1, 0, 0 -/
def b_n (n : ℕ) : ℕ := -- specify the actual counting function, placeholder below
  sorry

/-- Proof statement that for all positive integers n, b_{n+1} = 2a_n -/
theorem b_n_plus_1_eq_2a_n (n : ℕ) (hn : 0 < n) : b_n (n + 1) = 2 * a_n n :=
  sorry

end b_n_plus_1_eq_2a_n_l123_123146


namespace normal_level_shortage_l123_123253

theorem normal_level_shortage
  (T : ℝ) (Normal_level : ℝ)
  (h1 : 0.75 * T = 30)
  (h2 : 30 = 2 * Normal_level) :
  T - Normal_level = 25 := 
by
  sorry

end normal_level_shortage_l123_123253


namespace truck_travel_due_east_distance_l123_123274

theorem truck_travel_due_east_distance :
  ∀ (x : ℕ),
  (20 + 20)^2 + x^2 = 50^2 → x = 30 :=
by
  intro x
  sorry -- proof will be here

end truck_travel_due_east_distance_l123_123274


namespace problem_statement_l123_123767

theorem problem_statement (
  a b c d x y z t : ℝ
) (habcd : 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1) 
  (hxyz : 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z ∧ 1 ≤ t)
  (h_sum : a + b + c + d + x + y + z + t = 8) :
  a^2 + b^2 + c^2 + d^2 + x^2 + y^2 + z^2 + t^2 ≤ 28 := 
sorry

end problem_statement_l123_123767


namespace probability_at_least_one_woman_l123_123997

theorem probability_at_least_one_woman (m w n k : ℕ) (h_m : m = 7) (h_w : w = 3) (h_n : n = 10) (h_k : k = 3) :
  let total_people := m + w in
  let prob_no_woman := (m / total_people : ℝ) * ((m - 1) / (total_people - 1) : ℝ) * ((m - 2) / (total_people - 2) : ℝ) in
  let prob_at_least_one_woman := 1 - prob_no_woman in
  prob_no_woman = 7 / 24 ∧ prob_at_least_one_woman = 17 / 24 :=
by {
  sorry
}

end probability_at_least_one_woman_l123_123997


namespace energetics_minimum_bus_routes_l123_123400

theorem energetics_minimum_bus_routes :
  ∀ (factories : Finset ℕ) (f : ℕ → finset (ℕ × ℕ)),
  (\|factories| = 150) →
  (∀ (s : finset ℕ), (4 ≤ s.card → ∃ s₁ s₂ : finset ℕ, s₁.card = 2 ∧ s₂.card = 2 ∧ s₁ ∪ s₂ = s ∧ ∀ p ∈ s₁.product s₂, p.1 ≠ p.2 ∧ (p.1, p.2) ∈ f factories)) →
  ∀ (pairs : finset (ℕ × ℕ)),
  (∀ (p ∈ pairs, p.1 ≠ p.2 ∧ p.1 ∈ factories ∧ p.2 ∈ factories ∧ ∀ x ∈ factories, ∃! q ∈ pairs, q.1 = x ∨ q.2 = x)) →
  pairs.card = 11025 := 
by sorry

end energetics_minimum_bus_routes_l123_123400


namespace cos_alpha_value_l123_123173

theorem cos_alpha_value (α : ℝ) (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) : Real.cos α = 1 / 5 :=
sorry

end cos_alpha_value_l123_123173


namespace nonagon_perimeter_is_28_l123_123280

-- Definitions based on problem conditions
def numSides : Nat := 9
def lengthSides1 : Nat := 3
def lengthSides2 : Nat := 4
def numSidesOfLength1 : Nat := 8
def numSidesOfLength2 : Nat := 1

-- Theorem statement proving that the perimeter is 28 units
theorem nonagon_perimeter_is_28 : 
  numSides = numSidesOfLength1 + numSidesOfLength2 →
  8 * lengthSides1 + 1 * lengthSides2 = 28 :=
by
  intros
  sorry

end nonagon_perimeter_is_28_l123_123280


namespace trigonometric_identity_l123_123844

variable (θ : ℝ) (h : Real.tan θ = 2)

theorem trigonometric_identity : 
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4 / 5 := 
sorry

end trigonometric_identity_l123_123844


namespace man_gets_dividend_l123_123942

    -- Definitions based on conditions
    noncomputable def investment : ℝ := 14400
    noncomputable def premium_rate : ℝ := 0.20
    noncomputable def face_value : ℝ := 100
    noncomputable def dividend_rate : ℝ := 0.07

    -- Calculate the price per share with premium
    noncomputable def price_per_share : ℝ := face_value * (1 + premium_rate)

    -- Calculate the number of shares bought
    noncomputable def number_of_shares : ℝ := investment / price_per_share

    -- Calculate the dividend per share
    noncomputable def dividend_per_share : ℝ := face_value * dividend_rate

    -- Calculate the total dividend
    noncomputable def total_dividend : ℝ := dividend_per_share * number_of_shares

    -- The proof statement
    theorem man_gets_dividend : total_dividend = 840 := by
        sorry
    
end man_gets_dividend_l123_123942


namespace water_volume_correct_l123_123616

def total_initial_solution : ℚ := 0.08 + 0.04 + 0.02
def fraction_water_in_initial : ℚ := 0.04 / total_initial_solution
def desired_total_volume : ℚ := 0.84
def required_water_volume : ℚ := desired_total_volume * fraction_water_in_initial

theorem water_volume_correct : 
  required_water_volume = 0.24 :=
by
  -- The proof is omitted
  sorry

end water_volume_correct_l123_123616


namespace problem_equivalent_l123_123552

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

noncomputable def two_digit_prime_numbers_sum_10 : ℕ :=
  {n | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.card

theorem problem_equivalent : two_digit_prime_numbers_sum_10 = 3 :=
sorry

end problem_equivalent_l123_123552


namespace sampling_methods_correct_l123_123141

-- Define the conditions given in the problem.
def total_students := 200
def method_1_is_simple_random := true
def method_2_is_systematic := true

-- The proof problem statement, no proof is required.
theorem sampling_methods_correct :
  (method_1_is_simple_random = true) ∧
  (method_2_is_systematic = true) :=
by
  -- using conditions defined above, we state the theorem we need to prove
  sorry

end sampling_methods_correct_l123_123141


namespace sum_of_exponents_l123_123996

-- Definition of Like Terms
def like_terms (m n : ℕ) : Prop :=
  m = 3 ∧ n = 2

-- Theorem statement
theorem sum_of_exponents (m n : ℕ) (h : like_terms m n) : m + n = 5 :=
sorry

end sum_of_exponents_l123_123996


namespace cos_240_eq_neg_half_l123_123337

/-- Prove that cos 240 degrees equals -1/2 --/
theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1/2 :=
  sorry

end cos_240_eq_neg_half_l123_123337


namespace max_n_for_factorable_quadratic_l123_123729

theorem max_n_for_factorable_quadratic :
  ∃ n : ℤ, (∀ x : ℤ, ∃ A B : ℤ, (3*x^2 + n*x + 108) = (3*x + A)*( x + B) ∧ A*B = 108 ∧ n = A + 3*B) ∧ n = 325 :=
by
  sorry

end max_n_for_factorable_quadratic_l123_123729


namespace arithmetic_sequence_S_15_l123_123076

noncomputable def S (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  (n * (a 1 + a n)) / 2

variables {a : ℕ → ℤ}

theorem arithmetic_sequence_S_15 :
  (a 1 - a 4 - a 8 - a 12 + a 15 = 2) →
  (a 1 + a 15 = 2 * a 8) →
  (a 4 + a 12 = 2 * a 8) →
  S 15 a = -30 :=
by
  intros h1 h2 h3
  sorry

end arithmetic_sequence_S_15_l123_123076


namespace count_two_digit_primes_with_digit_sum_10_l123_123548

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter (λ n, digits_sum_to_10 n ∧ is_prime n) (Finset.filter is_two_digit (Finset.range 100))).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l123_123548


namespace gridiron_football_club_members_count_l123_123093

theorem gridiron_football_club_members_count :
  let sock_price := 6
  let tshirt_price := sock_price + 7
  let helmet_price := 2 * tshirt_price
  let total_cost_per_member := sock_price + tshirt_price + helmet_price
  let total_expenditure := 4680
  total_expenditure / total_cost_per_member = 104 :=
by
  let sock_price := 6
  let tshirt_price := sock_price + 7
  let helmet_price := 2 * tshirt_price
  let total_cost_per_member := sock_price + tshirt_price + helmet_price
  let total_expenditure := 4680
  sorry

end gridiron_football_club_members_count_l123_123093


namespace magnets_per_earring_l123_123435

theorem magnets_per_earring (M : ℕ) (h : 4 * (3 * M / 2) = 24) : M = 4 :=
by
  sorry

end magnets_per_earring_l123_123435


namespace total_peaches_l123_123453

theorem total_peaches (x : ℕ) (P : ℕ) 
(h1 : P = 6 * x + 57)
(h2 : 6 * x + 57 = 9 * x - 51) : 
  P = 273 :=
by
  sorry

end total_peaches_l123_123453


namespace drunk_drivers_count_l123_123757

theorem drunk_drivers_count (D S : ℕ) (h1 : S = 7 * D - 3) (h2 : D + S = 45) : D = 6 :=
by
  sorry

end drunk_drivers_count_l123_123757


namespace roots_of_quadratic_sum_cube_l123_123414

noncomputable def quadratic_roots (a b c : ℤ) (p q : ℤ) : Prop :=
  p^2 - b * p + c = 0 ∧ q^2 - b * q + c = 0

theorem roots_of_quadratic_sum_cube (p q : ℤ) :
  quadratic_roots 1 (-5) 6 p q →
  p^3 + p^4 * q^2 + p^2 * q^4 + q^3 = 503 :=
by
  sorry

end roots_of_quadratic_sum_cube_l123_123414


namespace inhabitable_fraction_of_mars_surface_l123_123868

theorem inhabitable_fraction_of_mars_surface :
  (3 / 5 : ℚ) * (2 / 3) = (2 / 5) :=
by
  sorry

end inhabitable_fraction_of_mars_surface_l123_123868


namespace smallest_positive_integer_x_l123_123979

theorem smallest_positive_integer_x :
  ∃ x : ℕ, 42 * x + 14 ≡ 4 [MOD 26] ∧ x ≡ 3 [MOD 5] ∧ x = 38 := 
by
  sorry

end smallest_positive_integer_x_l123_123979


namespace range_of_f_l123_123851

/-- Define the piecewise function f(x) -/
noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + 1 else Real.cos x

/-- Prove that the range of f(x) is [-1, ∞) -/
theorem range_of_f : Set.range f = Set.Ici (-1) :=
by sorry

end range_of_f_l123_123851


namespace min_value_x_plus_one_over_x_plus_two_l123_123981

theorem min_value_x_plus_one_over_x_plus_two (x : ℝ) (h : x > -2) : ∃ y : ℝ, y = x + 1 / (x + 2) ∧ y ≥ 0 := 
sorry

end min_value_x_plus_one_over_x_plus_two_l123_123981


namespace no_five_integer_solutions_divisibility_condition_l123_123832

variables (k : ℤ) 

-- Definition of equation
def equation (x y : ℤ) : Prop :=
  y^2 - k = x^3

-- Variables to capture the integer solutions
variables (x1 x2 x3 x4 x5 y1 : ℤ)

-- Prove that there do not exist five solutions satisfying the given forms
theorem no_five_integer_solutions :
  ¬(equation k x1 y1 ∧ 
    equation k x2 (y1 - 1) ∧ 
    equation k x3 (y1 - 2) ∧ 
    equation k x4 (y1 - 3) ∧ 
    equation k x5 (y1 - 4)) :=
sorry

-- Prove divisibility condition for the first four solutions
theorem divisibility_condition :
  (equation k x1 y1 ∧ 
   equation k x2 (y1 - 1) ∧ 
   equation k x3 (y1 - 2) ∧ 
   equation k x4 (y1 - 3)) → 
  63 ∣ (k - 17) :=
sorry

end no_five_integer_solutions_divisibility_condition_l123_123832


namespace area_EFGH_l123_123178

theorem area_EFGH (n : ℕ) (n_pos : 1 < n) (S_ABCD : ℝ) (h₁ : S_ABCD = 1) :
  ∃ S_EFGH : ℝ, S_EFGH = (n - 2) / n :=
by sorry

end area_EFGH_l123_123178


namespace maggie_remaining_goldfish_l123_123889

theorem maggie_remaining_goldfish
  (total_goldfish : ℕ)
  (allowed_fraction : ℕ → ℕ)
  (caught_fraction : ℕ → ℕ)
  (halfsies : ℕ)
  (remaining_goldfish : ℕ)
  (h1 : total_goldfish = 100)
  (h2 : allowed_fraction total_goldfish = total_goldfish / 2)
  (h3 : caught_fraction (allowed_fraction total_goldfish) = (3 * allowed_fraction total_goldfish) / 5)
  (h4 : halfsies = allowed_fraction total_goldfish)
  (h5 : remaining_goldfish = halfsies - caught_fraction halfsies) :
  remaining_goldfish = 20 :=
sorry

end maggie_remaining_goldfish_l123_123889


namespace maria_sold_in_first_hour_l123_123603

variable (x : ℕ)

-- Conditions
def sold_in_first_hour := x
def sold_in_second_hour := 2
def average_sold_in_two_hours := 6

-- Proof Goal
theorem maria_sold_in_first_hour :
  (sold_in_first_hour + sold_in_second_hour) / 2 = average_sold_in_two_hours → sold_in_first_hour = 10 :=
by
  sorry

end maria_sold_in_first_hour_l123_123603


namespace drunk_drivers_traffic_class_l123_123754

-- Define the variables for drunk drivers and speeders
variable (d s : ℕ)

-- Define the given conditions as hypotheses
theorem drunk_drivers_traffic_class (h1 : d + s = 45) (h2 : s = 7 * d - 3) : d = 6 := by
  sorry

end drunk_drivers_traffic_class_l123_123754


namespace range_of_f_when_a_neg_2_is_0_to_4_and_bounded_range_of_a_if_f_bounded_by_4_l123_123564

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 + a * Real.sin x - Real.cos x ^ 2

theorem range_of_f_when_a_neg_2_is_0_to_4_and_bounded :
  (∀ x : ℝ, 0 ≤ f (-2) x ∧ f (-2) x ≤ 4) :=
sorry

theorem range_of_a_if_f_bounded_by_4 :
  (∀ x : ℝ, abs (f a x) ≤ 4) → (-2 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_f_when_a_neg_2_is_0_to_4_and_bounded_range_of_a_if_f_bounded_by_4_l123_123564


namespace percent_of_number_l123_123132

theorem percent_of_number (x : ℝ) (hx : (120 / x) = (75 / 100)) : x = 160 := 
sorry

end percent_of_number_l123_123132


namespace proof_A_proof_C_l123_123807

theorem proof_A (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a * b ≤ ( (a + b) / 2) ^ 2 := 
sorry

theorem proof_C (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2) : 
  ∃ y, y = x * (4 - x^2).sqrt ∧ y ≤ 2 := 
sorry

end proof_A_proof_C_l123_123807


namespace cos_240_eq_neg_half_l123_123312

theorem cos_240_eq_neg_half : ∀ (deg: ℝ), 
  deg = 240 → 
  (∀ a b : ℝ, 240 = a + b → a = 180 → b = 60 → 
    cos 240 = cos a * cos b - sin a * sin b) → 
  cos 180 = -1 →
  sin 180 = 0 →
  cos 60 = 1 / 2 →
  cos 240 = -1 / 2 :=
by 
  intros deg h_deg h_sum h_cos_180 h_sin_180 h_cos_60
  rw h_deg at h_sum
  have h_cos_identity := h_sum 180 60 rfl rfl rfl
  simp [h_cos_180, h_sin_180, h_cos_60] at h_cos_identity
  exact h_cos_identity

end cos_240_eq_neg_half_l123_123312


namespace tens_digit_of_smallest_even_five_digit_number_l123_123785

def smallest_even_five_digit_number (digits : List ℕ) : ℕ :=
if h : 0 ∈ digits ∧ 3 ∈ digits ∧ 5 ∈ digits ∧ 6 ∈ digits ∧ 8 ∈ digits then
  35086
else
  0  -- this is just a placeholder to make the function total

theorem tens_digit_of_smallest_even_five_digit_number : 
  ∀ digits : List ℕ, 
    0 ∈ digits ∧ 
    3 ∈ digits ∧ 
    5 ∈ digits ∧ 
    6 ∈ digits ∧ 
    8 ∈ digits ∧ 
    digits.length = 5 → 
    (smallest_even_five_digit_number digits) / 10 % 10 = 8 :=
by
  intros digits h
  sorry

end tens_digit_of_smallest_even_five_digit_number_l123_123785


namespace modulus_of_z_is_five_l123_123090

def z : Complex := 3 + 4 * Complex.I

theorem modulus_of_z_is_five : Complex.abs z = 5 := by
  sorry

end modulus_of_z_is_five_l123_123090


namespace cos_240_eq_neg_half_l123_123338

/-- Prove that cos 240 degrees equals -1/2 --/
theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1/2 :=
  sorry

end cos_240_eq_neg_half_l123_123338


namespace cosine_240_l123_123358

theorem cosine_240 (h1 : Real.cos 60 = 1 / 2) : Real.cos 240 = -1 / 2 :=
by
  have h2 : Real.cos 240 = -Real.cos 60 := by sorry
  rw [h2, h1]
  exact neg_div
  sorry

end cosine_240_l123_123358


namespace find_m_l123_123447

-- Define the conditions
def function_is_decreasing (m : ℝ) : Prop := 
  (m^2 - m - 1 = 1) ∧ (1 - m < 0)

-- The proof problem: prove m = 2 given the conditions
theorem find_m (m : ℝ) (h : function_is_decreasing m) : m = 2 := 
by
  sorry -- Proof to be filled in

end find_m_l123_123447


namespace cube_pyramid_same_volume_height_l123_123139

theorem cube_pyramid_same_volume_height (h : ℝ) :
  let cube_edge : ℝ := 5
  let pyramid_base_edge : ℝ := 6
  let cube_volume : ℝ := cube_edge ^ 3
  let pyramid_volume : ℝ := (1 / 3) * (pyramid_base_edge ^ 2) * h
  cube_volume = pyramid_volume → h = 125 / 12 :=
by
  intros
  sorry

end cube_pyramid_same_volume_height_l123_123139


namespace largest_fraction_l123_123120

theorem largest_fraction :
  (∀ (a b : ℚ), a = 2 / 5 → b = 1 / 3 → a < b) ∧  
  (∀ (a c : ℚ), a = 2 / 5 → c = 7 / 15 → a < c) ∧ 
  (∀ (a d : ℚ), a = 2 / 5 → d = 5 / 12 → a < d) ∧ 
  (∀ (a e : ℚ), a = 2 / 5 → e = 3 / 8 → a < e) ∧ 
  (∀ (b c : ℚ), b = 1 / 3 → c = 7 / 15 → b < c) ∧
  (∀ (b d : ℚ), b = 1 / 3 → d = 5 / 12 → b < d) ∧ 
  (∀ (b e : ℚ), b = 1 / 3 → e = 3 / 8 → b < e) ∧ 
  (∀ (c d : ℚ), c = 7 / 15 → d = 5 / 12 → c > d) ∧
  (∀ (c e : ℚ), c = 7 / 15 → e = 3 / 8 → c > e) ∧
  (∀ (d e : ℚ), d = 5 / 12 → e = 3 / 8 → d > e) :=
sorry

end largest_fraction_l123_123120


namespace kathryn_remaining_money_l123_123883

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

end kathryn_remaining_money_l123_123883


namespace positive_difference_two_solutions_abs_eq_15_l123_123667

theorem positive_difference_two_solutions_abs_eq_15 :
  ∀ (x1 x2 : ℝ), (|x1 - 3| = 15) ∧ (|x2 - 3| = 15) ∧ (x1 > x2) → (x1 - x2 = 30) :=
by
  intros x1 x2 h
  sorry

end positive_difference_two_solutions_abs_eq_15_l123_123667


namespace count_two_digit_primes_with_digit_sum_10_l123_123555

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def is_prime (n : ℕ) : Prop := Nat.Prime n

def meets_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter meets_conditions (Finset.range 100)).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l123_123555


namespace find_natural_numbers_l123_123841

theorem find_natural_numbers :
  ∃ (x y : ℕ), 
    x * y - (x + y) = Nat.gcd x y + Nat.lcm x y ∧ 
    ((x = 6 ∧ y = 3) ∨ (x = 6 ∧ y = 4) ∨ (x = 3 ∧ y = 6) ∨ (x = 4 ∧ y = 6)) := 
by 
  sorry

end find_natural_numbers_l123_123841


namespace find_y_value_l123_123876

-- Define the angles in Lean
def angle1 (y : ℕ) : ℕ := 6 * y
def angle2 (y : ℕ) : ℕ := 7 * y
def angle3 (y : ℕ) : ℕ := 3 * y
def angle4 (y : ℕ) : ℕ := 2 * y

-- The condition that the sum of the angles is 360
def angles_sum_to_360 (y : ℕ) : Prop :=
  angle1 y + angle2 y + angle3 y + angle4 y = 360

-- The proof problem statement
theorem find_y_value (y : ℕ) (h : angles_sum_to_360 y) : y = 20 :=
sorry

end find_y_value_l123_123876


namespace sum_is_18_less_than_abs_sum_l123_123638

theorem sum_is_18_less_than_abs_sum : 
  (-5 + -4) = (|-5| + |-4| - 18) :=
by
  sorry

end sum_is_18_less_than_abs_sum_l123_123638


namespace line_tangent_to_ellipse_l123_123750

theorem line_tangent_to_ellipse (m : ℝ) :
  (∃! x : ℝ, x^2 + 4 * (m * x + 1)^2 = 1) → m^2 = 3 / 4 :=
by
  sorry

end line_tangent_to_ellipse_l123_123750


namespace solution_l123_123378

theorem solution (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 36) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) 
  : x * y * z = 8 := 
by sorry

end solution_l123_123378


namespace negate_proposition_l123_123107

theorem negate_proposition (x : ℝ) :
  (¬(x > 1 → x^2 > 1)) ↔ (x ≤ 1 → x^2 ≤ 1) :=
by
  sorry

end negate_proposition_l123_123107


namespace transport_capacity_rental_plans_l123_123474

-- Define the conditions
def condition1 (x y : ℝ) : Prop := 2 * x + 3 * y = 1800
def condition2 (x y : ℝ) : Prop := 3 * x + 4 * y = 2500
def transport_by_trucks (a b : ℕ) : Prop := 300 * a + 400 * b = 3100

theorem transport_capacity (x y : ℝ) (ha : condition1 x y) (hb : condition2 x y) :
  x = 300 ∧ y = 400 :=
sorry

theorem rental_plans :
  ∃ a b : ℕ, transport_by_trucks a b :=
sorry

end transport_capacity_rental_plans_l123_123474


namespace largest_lucky_number_l123_123742

theorem largest_lucky_number : 
  let a := 1
  let b := 4
  let lucky_number (x y : ℕ) := x + y + x * y
  let c1 := lucky_number a b
  let c2 := lucky_number b c1
  let c3 := lucky_number c1 c2
  c3 = 499 :=
by
  sorry

end largest_lucky_number_l123_123742


namespace arithmetic_sequence_check_l123_123862

theorem arithmetic_sequence_check 
  (a : ℕ → ℝ) 
  (d : ℝ)
  (h : ∀ n : ℕ, a (n+1) = a n + d) 
  : (∀ n : ℕ, (a n + 1) - (a (n - 1) + 1) = d) 
    ∧ (∀ n : ℕ, 2 * a (n + 1) - 2 * a n = 2 * d)
    ∧ (∀ n : ℕ, a (n + 1) - (a n + n) = d + 1) := 
by
  sorry

end arithmetic_sequence_check_l123_123862


namespace find_a8_l123_123376

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
def geom_sequence (a : ℕ → ℝ) (a1 q : ℝ) :=
  ∀ n, a n = a1 * q ^ n

def sum_geom_sequence (S a : ℕ → ℝ) (a1 q : ℝ) :=
  ∀ n, S n = a1 * (1 - q ^ (n + 1)) / (1 - q)

def arithmetic_sequence (S : ℕ → ℝ) :=
  S 9 = S 3 + S 6

def sum_a2_a5 (a : ℕ → ℝ) :=
  a 2 + a 5 = 4

theorem find_a8 (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 q : ℝ)
  (hgeom_seq : geom_sequence a a1 q)
  (hsum_geom_seq : sum_geom_sequence S a a1 q)
  (harith_seq : arithmetic_sequence S)
  (hsum_a2_a5 : sum_a2_a5 a) :
  a 8 = 2 :=
sorry

end find_a8_l123_123376


namespace monic_quadratic_polynomial_with_real_coefficients_l123_123369

noncomputable def quadratic_polynomial_with_root (r : ℂ) :=
  by pol :=
    let conj_r := complex.conj r
    let p := polynomial.X - polynomial.C r
    let q := polynomial.X - polynomial.C conj_r
    polynomial.monic ((p * q).map complex.ofReal)
    sorry  -- skipping the actual calculation

theorem monic_quadratic_polynomial_with_real_coefficients (r : ℂ) (hr : r = 3 + complex.I * real.sqrt 3) : 
  quadratic_polynomial_with_root r = polynomial.X^2 - 6 * polynomial.X + 12 :=
sorry  -- proof to be written

end monic_quadratic_polynomial_with_real_coefficients_l123_123369


namespace half_abs_diff_of_squares_l123_123654

theorem half_abs_diff_of_squares (a b : ℤ) (h1 : a = 20) (h2 : b = 15) :
  (1/2 : ℚ) * |(a:ℚ)^2 - (b:ℚ)^2| = 87.5 := by
  sorry

end half_abs_diff_of_squares_l123_123654


namespace equivalent_fraction_l123_123806

theorem equivalent_fraction (b : ℕ) (h : b = 2024) :
  (b^3 - 2 * b^2 * (b + 1) + 3 * b * (b + 1)^2 - (b + 1)^3 + 4) / (b * (b + 1)) = 2022 := by
  rw [h]
  sorry

end equivalent_fraction_l123_123806


namespace min_value_l123_123525

def f (x y : ℝ) : ℝ := x^2 + 4 * x * y + 5 * y^2 - 10 * x - 6 * y + 3

theorem min_value : ∃ x y : ℝ, (x + y = 2) ∧ (f x y = -(1/7)) :=
by
  sorry

end min_value_l123_123525


namespace cos_240_eq_neg_half_l123_123284

theorem cos_240_eq_neg_half (h1 : Real.cos (180 * Real.pi / 180) = -1)
                            (h2 : Real.sin (180 * Real.pi / 180) = 0)
                            (h3 : Real.cos (60 * Real.pi / 180) = 1 / 2) :
  Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_240_eq_neg_half_l123_123284


namespace probability_computation_l123_123258

noncomputable def probability_two_equal_three : ℚ :=
  let p_one_digit : ℚ := 3 / 4
  let p_two_digit : ℚ := 1 / 4
  let number_of_dice : ℕ := 5
  let ways_to_choose_two_digit := Nat.choose number_of_dice 2
  ways_to_choose_two_digit * (p_two_digit^2) * (p_one_digit^3)

theorem probability_computation :
  probability_two_equal_three = 135 / 512 :=
by
  sorry

end probability_computation_l123_123258


namespace gray_area_is_50pi_l123_123592

noncomputable section

-- Define the radii of the inner and outer circles
def R_inner : ℝ := 2.5
def R_outer : ℝ := 3 * R_inner

-- Area of circles
def A_inner : ℝ := Real.pi * R_inner^2
def A_outer : ℝ := Real.pi * R_outer^2

-- Define width of the gray region
def gray_width : ℝ := R_outer - R_inner

-- Gray area calculation
def A_gray : ℝ := A_outer - A_inner

-- The theorem stating the area of the gray region
theorem gray_area_is_50pi :
  gray_width = 5 → A_gray = 50 * Real.pi := by
  -- Here we assume the proof continues
  sorry

end gray_area_is_50pi_l123_123592


namespace distinct_divisors_in_set_l123_123764

theorem distinct_divisors_in_set (p : ℕ) (hp : Nat.Prime p) (hp5 : 5 < p) :
  ∃ (x y : ℕ), x ∈ {p - n^2 | n : ℕ} ∧ y ∈ {p - n^2 | n : ℕ} ∧ x ≠ y ∧ x ≠ 1 ∧ x ∣ y :=
by
  sorry

end distinct_divisors_in_set_l123_123764


namespace proof_problem_l123_123045

theorem proof_problem (x : ℝ) (a : ℝ) :
  (0 < x) → 
  (x + 1 / x ≥ 2) →
  (x + 4 / x^2 ≥ 3) →
  (x + 27 / x^3 ≥ 4) →
  a = 4^4 → 
  x + a / x^4 ≥ 5 :=
  sorry

end proof_problem_l123_123045


namespace shelves_in_room_l123_123953

theorem shelves_in_room
  (n_action_figures_per_shelf : ℕ)
  (total_action_figures : ℕ)
  (h1 : n_action_figures_per_shelf = 10)
  (h2 : total_action_figures = 80) :
  total_action_figures / n_action_figures_per_shelf = 8 := by
  sorry

end shelves_in_room_l123_123953


namespace cost_of_five_dozen_apples_l123_123150

theorem cost_of_five_dozen_apples 
  (cost_four_dozen : ℝ) 
  (cost_one_dozen : ℝ) 
  (cost_five_dozen : ℝ) 
  (h1 : cost_four_dozen = 31.20) 
  (h2 : cost_one_dozen = cost_four_dozen / 4) 
  (h3 : cost_five_dozen = 5 * cost_one_dozen)
  : cost_five_dozen = 39.00 :=
sorry

end cost_of_five_dozen_apples_l123_123150


namespace longest_side_similar_triangle_l123_123103

noncomputable def internal_angle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem longest_side_similar_triangle (a b c A : ℝ) (h₁ : a = 4) (h₂ : b = 6) (h₃ : c = 7) (h₄ : A = 132) :
  let k := Real.sqrt (132 / internal_angle 4 6 7)
  7 * k = 73.5 :=
by
  sorry

end longest_side_similar_triangle_l123_123103


namespace cos_240_degree_l123_123294

theorem cos_240_degree : Real.cos (240 * Real.pi / 180) = -1/2 :=
by
  sorry

end cos_240_degree_l123_123294


namespace find_slope_l123_123040

theorem find_slope 
  (k : ℝ)
  (y : ℝ -> ℝ)
  (P : ℝ × ℝ)
  (l : ℝ -> ℝ -> Prop)
  (A B F : ℝ × ℝ)
  (C : ℝ × ℝ -> Prop)
  (d : ℝ × ℝ -> ℝ × ℝ -> ℝ)
  (k_pos : P = (3, 0))
  (k_slope : ∀ x, y x = k * (x - 3))
  (k_int_hyperbola_A : C A)
  (k_int_hyperbola_B : C B)
  (k_focus : F = (2, 0))
  (k_sum_dist : d A F + d B F = 16) :
  k = 1 ∨ k = -1 :=
sorry

end find_slope_l123_123040


namespace remainder_of_P_div_D_is_25158_l123_123469

noncomputable def P (x : ℝ) := 4 * x^8 - 2 * x^6 + 5 * x^4 - x^3 + 3 * x - 15
def D (x : ℝ) := 2 * x - 6

theorem remainder_of_P_div_D_is_25158 : P 3 = 25158 := by
  sorry

end remainder_of_P_div_D_is_25158_l123_123469


namespace cos_240_degree_l123_123293

theorem cos_240_degree : Real.cos (240 * Real.pi / 180) = -1/2 :=
by
  sorry

end cos_240_degree_l123_123293


namespace correct_calculation_value_l123_123799

theorem correct_calculation_value (x : ℕ) (h : (x * 5) + 7 = 27) : (x + 5) * 7 = 63 :=
by
  -- The conditions are used directly in the definitions
  -- Given the condition (x * 5) + 7 = 27
  let h1 := h
  -- Solve for x and use x in the correct calculation
  sorry

end correct_calculation_value_l123_123799


namespace second_quadrant_necessary_not_sufficient_l123_123000

variable (α : ℝ) -- Assuming α is a real number for generality.

-- Define what it means for an angle to be in the second quadrant (90° < α < 180°).
def in_second_quadrant (α : ℝ) : Prop :=
  90 < α ∧ α < 180

-- Define what it means for an angle to be obtuse (90° < α ≤ 180°).
def is_obtuse (α : ℝ) : Prop :=
  90 < α ∧ α ≤ 180

-- State the theorem to prove: 
-- "The angle α is in the second quadrant" is a necessary but not sufficient condition for "α is an obtuse angle".
theorem second_quadrant_necessary_not_sufficient : 
  (∀ α, is_obtuse α → in_second_quadrant α) ∧ 
  (∃ α, in_second_quadrant α ∧ ¬is_obtuse α) :=
sorry

end second_quadrant_necessary_not_sufficient_l123_123000


namespace area_of_field_l123_123255

theorem area_of_field (b l : ℝ) (h1 : l = b + 30) (h2 : 2 * (l + b) = 540) : l * b = 18000 := 
by
  sorry

end area_of_field_l123_123255


namespace find_wanderer_in_8th_bar_l123_123872

noncomputable def wanderer_probability : ℚ := 1 / 3

theorem find_wanderer_in_8th_bar
    (total_bars : ℕ)
    (initial_prob_in_any_bar : ℚ)
    (prob_not_in_specific_bar : ℚ)
    (prob_not_in_first_seven : ℚ)
    (posterior_prob : ℚ)
    (h1 : total_bars = 8)
    (h2 : initial_prob_in_any_bar = 4 / 5)
    (h3 : prob_not_in_specific_bar = 1 - (initial_prob_in_any_bar / total_bars))
    (h4 : prob_not_in_first_seven = prob_not_in_specific_bar ^ 7)
    (h5 : posterior_prob = initial_prob_in_any_bar / prob_not_in_first_seven) :
    posterior_prob = wanderer_probability := 
sorry

end find_wanderer_in_8th_bar_l123_123872


namespace w_z_ratio_l123_123170

theorem w_z_ratio (w z : ℝ) (h : (1/w + 1/z) / (1/w - 1/z) = 2023) : (w + z) / (w - z) = -2023 :=
by sorry

end w_z_ratio_l123_123170


namespace complete_the_square_l123_123247

theorem complete_the_square (x : ℝ) : x^2 - 2 * x - 1 = 0 -> (x - 1)^2 = 2 := by
  sorry

end complete_the_square_l123_123247


namespace total_money_spent_l123_123163

theorem total_money_spent (emma_spent : ℤ) (elsa_spent : ℤ) (elizabeth_spent : ℤ) 
(emma_condition : emma_spent = 58) 
(elsa_condition : elsa_spent = 2 * emma_spent) 
(elizabeth_condition : elizabeth_spent = 4 * elsa_spent) 
:
emma_spent + elsa_spent + elizabeth_spent = 638 :=
by
  rw [emma_condition, elsa_condition, elizabeth_condition]
  norm_num
  sorry

end total_money_spent_l123_123163


namespace Z_in_third_quadrant_l123_123049

noncomputable def Z : ℂ := -2 * complex.I / (1 + 2 * complex.I)

def quadrant_iii (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem Z_in_third_quadrant : quadrant_iii Z :=
  sorry

end Z_in_third_quadrant_l123_123049


namespace roll_contains_25_coins_l123_123518

variable (coins_per_roll : ℕ)

def rolls_per_teller := 10
def number_of_tellers := 4
def total_coins := 1000

theorem roll_contains_25_coins : 
  (number_of_tellers * rolls_per_teller * coins_per_roll = total_coins) → 
  (coins_per_roll = 25) :=
by
  sorry

end roll_contains_25_coins_l123_123518


namespace solution_set_of_inequality_l123_123719

open Set

theorem solution_set_of_inequality :
  {x : ℝ | (x ≠ -2) ∧ (x ≠ -8) ∧ (2 / (x + 2) + 4 / (x + 8) ≥ 4 / 5)} =
  {x : ℝ | (-8 < x ∧ x < -2) ∨ (-2 < x ∧ x ≤ 4)} :=
by
  sorry

end solution_set_of_inequality_l123_123719


namespace solve_for_y_l123_123846

noncomputable def determinant3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h

noncomputable def determinant2x2 (a b c d : ℝ) : ℝ := 
  a*d - b*c

theorem solve_for_y (b y : ℝ) (h : b ≠ 0) :
  determinant3x3 (y + 2 * b) y y y (y + 2 * b) y y y (y + 2 * b) = 0 → 
  y = -b / 2 :=
by
  sorry

end solve_for_y_l123_123846


namespace half_absolute_difference_of_squares_l123_123663

-- Defining the variables a and b involved in the problem
def a : ℤ := 20
def b : ℤ := 15

-- Statement to prove the solution
theorem half_absolute_difference_of_squares : 
    (1 / 2 : ℚ) * (abs ((a ^ 2) - (b ^ 2))) = 87.5 :=
by
    -- Proof omitted
    sorry

end half_absolute_difference_of_squares_l123_123663


namespace washed_shirts_l123_123606

-- Definitions based on the conditions
def short_sleeve_shirts : ℕ := 39
def long_sleeve_shirts : ℕ := 47
def unwashed_shirts : ℕ := 66

-- The total number of shirts is the sum of short and long sleeve shirts
def total_shirts : ℕ := short_sleeve_shirts + long_sleeve_shirts

-- The problem to prove that Oliver washed 20 shirts
theorem washed_shirts :
  total_shirts - unwashed_shirts = 20 := 
sorry

end washed_shirts_l123_123606


namespace james_total_chore_time_l123_123406

theorem james_total_chore_time
  (V C L : ℝ)
  (hV : V = 3)
  (hC : C = 3 * V)
  (hL : L = C / 2) :
  V + C + L = 16.5 := by
  sorry

end james_total_chore_time_l123_123406


namespace inheritance_problem_l123_123506

variables (x1 x2 x3 x4 : ℕ)

theorem inheritance_problem
  (h1 : x1 + x2 + x3 + x4 = 1320)
  (h2 : x1 + x4 = x2 + x3)
  (h3 : x2 + x4 = 2 * (x1 + x3))
  (h4 : x3 + x4 = 3 * (x1 + x2)) :
  x1 = 55 ∧ x2 = 275 ∧ x3 = 385 ∧ x4 = 605 :=
by sorry

end inheritance_problem_l123_123506


namespace problem1_problem2_l123_123680

theorem problem1 (α : ℝ) (hα : Real.tan α = 2) :
    (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 := 
sorry

theorem problem2 (α : ℝ) (hα : Real.tan α = 2) :
    (Real.sin (↑(π/2) + α) * Real.cos (↑(5*π/2) - α) * Real.tan (↑(-π) + α)) / 
    (Real.tan (↑(7*π) - α) * Real.sin (↑π + α)) = Real.cos α := 
sorry

end problem1_problem2_l123_123680


namespace minimize_sum_of_legs_l123_123848

noncomputable def area_of_right_angle_triangle (a b : ℝ) : Prop :=
  1/2 * a * b = 50

theorem minimize_sum_of_legs (a b : ℝ) (h : area_of_right_angle_triangle a b) :
  a + b = 20 ↔ a = 10 ∧ b = 10 :=
by
  sorry

end minimize_sum_of_legs_l123_123848


namespace polynomial_divisibility_l123_123241

theorem polynomial_divisibility (C D : ℝ)
  (h : ∀ x, x^2 + x + 1 = 0 → x^102 + C * x + D = 0) :
  C + D = -1 := 
by 
  sorry

end polynomial_divisibility_l123_123241


namespace friends_popcorn_l123_123914

theorem friends_popcorn (pieces_per_serving : ℕ) (jared_count : ℕ) (total_servings : ℕ) (jared_friends : ℕ)
  (h1 : pieces_per_serving = 30)
  (h2 : jared_count = 90)
  (h3 : total_servings = 9)
  (h4 : jared_friends = 3) :
  (total_servings * pieces_per_serving - jared_count) / jared_friends = 60 := by
  sorry

end friends_popcorn_l123_123914


namespace probability_value_at_least_75_cents_l123_123134

theorem probability_value_at_least_75_cents :
  let coins := ({4, 5, 7, 3} : Finset ℕ)
  let pennies := 4
  let nickels := 5
  let dimes := 7
  let quarters := 3
  let total_coins := pennies + nickels + dimes + quarters
  let total_ways := Nat.choose total_coins 7
  let ways_case_1 := Nat.choose 16 4
  let ways_case_2 := 3 * Nat.choose 7 2 * Nat.choose 5 3
  let successful_ways := 1 * ways_case_1 + ways_case_2
  (successful_ways / total_ways : ℚ) = 2450 / 50388 :=
by {
  -- let the variables lean knows
  let coins := ({4, 5, 7, 3} : Finset ℕ),
  let pennies := 4,
  let nickels := 5,
  let dimes := 7,
  let quarters := 3,
  let total_coins := pennies + nickels + dimes + quarters,
  let total_ways := Nat.choose total_coins 7,
  let ways_case_1 := 1 * Nat.choose 16 4,
  let ways_case_2 := 3 * Nat.choose 7 2 * Nat.choose 5 3,
  let successful_ways := ways_case_1 + ways_case_2,
  -- calculate probability in rational number
  have : (successful_ways : ℚ) / (total_ways : ℚ) = 2450 / 50388,
  sorry
}

end probability_value_at_least_75_cents_l123_123134


namespace seq_sum_terms_l123_123179

def S (n : ℕ) : ℕ := 3^n - 2

def a (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 2 * 3^(n - 1)

theorem seq_sum_terms (n : ℕ) : 
  a n = if n = 1 then 1 else 2 * 3^(n-1) :=
sorry

end seq_sum_terms_l123_123179


namespace correct_propositions_l123_123108

noncomputable def sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ n > 1, a n + a (n+1) > 2 * a n

def prop1 (a : ℕ → ℝ) (h : sequence_condition a) : Prop :=
  a 2 > a 1 → ∀ n > 1, a n > a (n-1)

def prop4 (a : ℕ → ℝ) (h : sequence_condition a) : Prop :=
  ∃ d, ∀ n > 1, a n > a 1 + (n-1) * d

theorem correct_propositions {a : ℕ → ℝ}
  (h : sequence_condition a) :
  (prop1 a h) ∧ (prop4 a h) := 
sorry

end correct_propositions_l123_123108


namespace min_value_p_plus_q_l123_123441

-- Definitions related to the conditions.
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def satisfies_equations (a b p q : ℕ) : Prop :=
  20 * a + 17 * b = p ∧ 17 * a + 20 * b = q ∧ is_prime p ∧ is_prime q

def distinct_positive_integers (a b : ℕ) : Prop := a > 0 ∧ b > 0 ∧ a ≠ b

-- The main proof problem.
theorem min_value_p_plus_q (a b p q : ℕ) :
  distinct_positive_integers a b →
  satisfies_equations a b p q →
  p + q = 296 :=
by
  sorry

end min_value_p_plus_q_l123_123441


namespace side_lengths_are_10_and_50_l123_123909

-- Define variables used in the problem
variables {s t : ℕ}

-- Define the conditions
def condition1 (s t : ℕ) : Prop := 4 * s = 20 * t
def condition2 (s t : ℕ) : Prop := s + t = 60

-- Prove that given the conditions, the side lengths of the squares are 10 and 50
theorem side_lengths_are_10_and_50 (s t : ℕ) (h1 : condition1 s t) (h2 : condition2 s t) : (s = 50 ∧ t = 10) ∨ (s = 10 ∧ t = 50) :=
by sorry

end side_lengths_are_10_and_50_l123_123909


namespace framed_painting_ratio_l123_123822

theorem framed_painting_ratio (x : ℝ) (h : (15 + 2 * x) * (30 + 4 * x) = 900) : (15 + 2 * x) / (30 + 4 * x) = 1 / 2 :=
by
  sorry

end framed_painting_ratio_l123_123822


namespace find_altitude_to_hypotenuse_l123_123949

-- define the conditions
def area : ℝ := 540
def hypotenuse : ℝ := 36
def altitude : ℝ := 30

-- define the problem statement
theorem find_altitude_to_hypotenuse (A : ℝ) (c : ℝ) (h : ℝ) 
  (h_area : A = 540) (h_hypotenuse : c = 36) : h = 30 :=
by
  -- skipping the proof
  sorry

end find_altitude_to_hypotenuse_l123_123949


namespace arithmetic_sequence_99th_term_l123_123382

-- Define the problem with conditions and question
theorem arithmetic_sequence_99th_term (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : S 9 = 27) (h2 : a 10 = 8) :
  a 99 = 97 := 
sorry

end arithmetic_sequence_99th_term_l123_123382


namespace total_spending_l123_123158

theorem total_spending (Emma_spent : ℕ) (Elsa_spent : ℕ) (Elizabeth_spent : ℕ) : 
  Emma_spent = 58 →
  Elsa_spent = 2 * Emma_spent →
  Elizabeth_spent = 4 * Elsa_spent →
  Emma_spent + Elsa_spent + Elizabeth_spent = 638 := 
by
  intros h_Emma h_Elsa h_Elizabeth
  sorry

end total_spending_l123_123158


namespace employees_use_public_transportation_l123_123683

theorem employees_use_public_transportation
    (total_employees : ℕ)
    (drive_percentage : ℝ)
    (public_transportation_fraction : ℝ)
    (h1 : total_employees = 100)
    (h2 : drive_percentage = 0.60)
    (h3 : public_transportation_fraction = 0.50) :
    ((total_employees * (1 - drive_percentage)) * public_transportation_fraction) = 20 :=
by
    sorry

end employees_use_public_transportation_l123_123683


namespace S8_is_80_l123_123042

variable {a : ℕ → ℝ} -- sequence definition
variable {S : ℕ → ℝ} -- sum of sequence

-- Conditions
variable (h_seq : ∀ n, a (n + 1) = a n + d) -- arithmetic sequence
variable (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) -- sum of the first n terms
variable (h_cond : a 3 = 20 - a 6) -- given condition

theorem S8_is_80 (h_seq : ∀ n, a (n + 1) = a n + d) (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) (h_cond : a 3 = 20 - a 6) :
  S 8 = 80 :=
sorry

end S8_is_80_l123_123042


namespace half_abs_diff_squares_l123_123653

theorem half_abs_diff_squares (a b : ℤ) (ha : a = 20) (hb : b = 15) : 
  (|a^2 - b^2| / 2) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l123_123653


namespace watch_all_episodes_in_67_weeks_l123_123424

def total_episodes : ℕ := 201
def episodes_per_week : ℕ := 1 + 2

theorem watch_all_episodes_in_67_weeks :
  total_episodes / episodes_per_week = 67 := by 
  sorry

end watch_all_episodes_in_67_weeks_l123_123424


namespace systematic_sampling_first_segment_l123_123650

theorem systematic_sampling_first_segment:
  ∀ (total_students sample_size segment_size 
     drawn_16th drawn_first : ℕ),
  total_students = 160 →
  sample_size = 20 →
  segment_size = 8 →
  drawn_16th = 125 →
  drawn_16th = drawn_first + segment_size * (16 - 1) →
  drawn_first = 5 :=
by
  intros total_students sample_size segment_size drawn_16th drawn_first
         htots hsamp hseg hdrw16 heq
  sorry

end systematic_sampling_first_segment_l123_123650


namespace combined_area_l123_123503

noncomputable def diagonal : ℝ := 12 * Real.sqrt 2

noncomputable def side_of_square (d : ℝ) : ℝ := d / Real.sqrt 2

noncomputable def area_of_square (s : ℝ) : ℝ := s ^ 2

noncomputable def radius_of_circle (d : ℝ) : ℝ := d / 2

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r ^ 2

theorem combined_area (d : ℝ) (h : d = diagonal) :
  let s := side_of_square d
  let area_sq := area_of_square s
  let r := radius_of_circle d
  let area_circ := area_of_circle r
  area_sq + area_circ = 144 + 72 * Real.pi :=
by
  sorry

end combined_area_l123_123503


namespace least_n_factorial_multiple_of_840_l123_123062

theorem least_n_factorial_multiple_of_840 :
  ∃ (n : ℕ), n ≥ 7 ∧ (∃ (k : ℕ), (n.factorial = 840 * k)) :=
sorry

end least_n_factorial_multiple_of_840_l123_123062


namespace seven_y_minus_x_eq_three_l123_123475

-- Definitions for the conditions
variables (x y : ℤ)
variables (hx : x > 0)
variables (h1 : x = 11 * y + 4)
variables (h2 : 2 * x = 18 * y + 1)

-- The theorem we want to prove
theorem seven_y_minus_x_eq_three : 7 * y - x = 3 :=
by
  -- Placeholder for the proof.
  sorry

end seven_y_minus_x_eq_three_l123_123475


namespace cos_240_eq_neg_half_l123_123321

theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Sorry to skip the proof
  sorry

end cos_240_eq_neg_half_l123_123321


namespace negation_of_inequality_l123_123789

theorem negation_of_inequality :
  ¬ (∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 2 < 0 := 
sorry

end negation_of_inequality_l123_123789


namespace half_abs_diff_squares_l123_123658

theorem half_abs_diff_squares : 
  let a := 20 in
  let b := 15 in
  let square (x : ℕ) := x * x in
  let abs_diff := abs (square a - square b) in
  abs_diff / 2 = 87.5 := 
by
  let a := 20
  let b := 15
  let square (x : ℕ) := x * x
  let abs_diff := abs (square a - square b)
  have h1 : square a = 400 := by native_decide
  have h2 : square b = 225 := by native_decide
  have h3 : abs_diff = abs (400 - 225) := by simp [square, h1, h2]
  have h4 : abs_diff = 175 := by simp [h3]
  have h5 : abs_diff / 2 = 87.5 := by norm_num [h4]
  exact h5

end half_abs_diff_squares_l123_123658


namespace simplify_fraction_l123_123737

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (hx2 : x^2 - (1 / y) ≠ 0) (hy2 : y^2 - (1 / x) ≠ 0) :
  (x^2 - 1 / y) / (y^2 - 1 / x) = (x * (x^2 * y - 1)) / (y * (y^2 * x - 1)) :=
sorry

end simplify_fraction_l123_123737


namespace congruence_equiv_l123_123747

theorem congruence_equiv (x : ℤ) (h : 5 * x + 9 ≡ 3 [ZMOD 18]) : 3 * x + 14 ≡ 14 [ZMOD 18] :=
sorry

end congruence_equiv_l123_123747


namespace functional_equation_solution_l123_123515

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y, f (f (f x)) + f (f y) = f y + x) → (∀ x, f x = x) :=
by
  intros f h x
  -- Proof goes here
  sorry

end functional_equation_solution_l123_123515


namespace b_value_rational_polynomial_l123_123561

theorem b_value_rational_polynomial (a b : ℚ) :
  (Polynomial.aeval (2 + Real.sqrt 3) (Polynomial.C (-15) + Polynomial.C b * X + Polynomial.C a * X^2 + X^3 : Polynomial ℚ) = 0) →
  b = -44 :=
by
  sorry

end b_value_rational_polynomial_l123_123561


namespace value_of_f_2019_l123_123381

noncomputable def f : ℝ → ℝ := sorry

variables (x : ℝ)

-- Assumptions
axiom f_zero : f 0 = 2
axiom f_period : ∀ x : ℝ, f (x + 3) = -f x

-- The property to be proved
theorem value_of_f_2019 : f 2019 = -2 := sorry

end value_of_f_2019_l123_123381


namespace tracy_initial_candies_l123_123917

variable (x y : ℕ) (h1 : 2 ≤ y) (h2 : y ≤ 6)

theorem tracy_initial_candies :
  (x - (1/5 : ℚ) * x = (4/5 : ℚ) * x) ∧
  ((4/5 : ℚ) * x - (1/3 : ℚ) * (4/5 : ℚ) * x = (8/15 : ℚ) * x) ∧
  y - 10 * 2 + ((8/15 : ℚ) * x - 20) = 5 →
  x = 60 :=
by
  sorry

end tracy_initial_candies_l123_123917


namespace intersection_of_M_and_N_l123_123769

noncomputable def setM : Set ℝ := { x : ℝ | x^2 - 2 * x - 3 < 0 }
noncomputable def setN : Set ℝ := { x : ℝ | Real.log x / Real.log 2 < 1 }

theorem intersection_of_M_and_N : { x : ℝ | x ∈ setM ∧ x ∈ setN } = { x : ℝ | 0 < x ∧ x < 2 } :=
by
  sorry

end intersection_of_M_and_N_l123_123769


namespace find_special_integers_l123_123030

theorem find_special_integers 
  : ∃ n : ℕ, 100 ≤ n ∧ n ≤ 1997 ∧ (2^n + 2) % n = 0 ∧ (n = 66 ∨ n = 198 ∨ n = 398 ∨ n = 798) :=
by
  sorry

end find_special_integers_l123_123030


namespace lcm_first_eight_l123_123249

open Nat

-- Defines the set of the first eight positive integers
def first_eight : Finset ℕ := Finset.range 9

-- Prove that the least common multiple of the set {1, 2, 3, 4, 5, 6, 7, 8} is 840
theorem lcm_first_eight : first_eight.lcm id = 840 := sorry

end lcm_first_eight_l123_123249


namespace negative_value_option_D_l123_123123

theorem negative_value_option_D :
  (-7) * (-6) > 0 ∧
  (-7) - (-15) > 0 ∧
  0 * (-2) * (-3) = 0 ∧
  (-6) + (-4) < 0 :=
by
  sorry

end negative_value_option_D_l123_123123


namespace am_minus_gm_less_than_option_D_l123_123389

variable (c d : ℝ)
variable (hc_pos : 0 < c) (hd_pos : 0 < d) (hcd_lt : c < d)

noncomputable def am : ℝ := (c + d) / 2
noncomputable def gm : ℝ := Real.sqrt (c * d)

theorem am_minus_gm_less_than_option_D :
  (am c d - gm c d) < ((d - c) ^ 3 / (8 * c)) :=
sorry

end am_minus_gm_less_than_option_D_l123_123389


namespace ways_to_divide_five_people_into_two_rooms_l123_123528

theorem ways_to_divide_five_people_into_two_rooms : 
  (finset.card (finset.powerset len_univ).filter(λ s, s.card = 3)) = 10 :=
by
  sorry

end ways_to_divide_five_people_into_two_rooms_l123_123528


namespace inequality_proof_l123_123884

open Real

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / (x^2 + y * z) + 1 / (y^2 + z * x) + 1 / (z^2 + x * y)) ≤ 
  (1 / 2) * (1 / (x * y) + 1 / (y * z) + 1 / (z * x)) :=
by sorry

end inequality_proof_l123_123884


namespace find_roots_of_equation_l123_123521

theorem find_roots_of_equation
  (a b c d x : ℝ)
  (h1 : a + d = 2015)
  (h2 : b + c = 2015)
  (h3 : a ≠ c)
  (h4 : (x - a) * (x - b) = (x - c) * (x - d)) :
  x = 1007.5 :=
by
  sorry

end find_roots_of_equation_l123_123521


namespace alice_profit_l123_123827

-- Define the variables and conditions
def total_bracelets : ℕ := 52
def material_cost : ℝ := 3.00
def bracelets_given_away : ℕ := 8
def sale_price : ℝ := 0.25

-- Calculate the number of bracelets sold
def bracelets_sold : ℕ := total_bracelets - bracelets_given_away

-- Calculate the revenue from selling the bracelets
def revenue : ℝ := bracelets_sold * sale_price

-- Define the profit as revenue minus material cost
def profit : ℝ := revenue - material_cost

-- The statement to prove
theorem alice_profit : profit = 8.00 := 
by
  sorry

end alice_profit_l123_123827


namespace squares_arrangement_l123_123601

noncomputable def arrangement_possible (n : ℕ) (cond : n ≥ 5) : Prop :=
  ∃ (position : ℕ → ℕ × ℕ),
    (∀ i, 1 ≤ i ∧ i ≤ n → 
        ∃ j k, j ≠ k ∧ 
             dist (position i) (position j) = 1 ∧
             dist (position i) (position k) = 1)

theorem squares_arrangement (n : ℕ) (hn : n ≥ 5) :
  arrangement_possible n hn :=
  sorry

end squares_arrangement_l123_123601


namespace pipes_fill_tank_l123_123774

theorem pipes_fill_tank (T : ℝ) (h1 : T > 0)
  (h2 : (1/4 : ℝ) + 1/T - 1/20 = 1/2.5) : T = 5 := by
  sorry

end pipes_fill_tank_l123_123774


namespace minimum_value_is_14_div_27_l123_123034

noncomputable def minimum_value_expression (x : ℝ) : ℝ :=
  (Real.sin x)^8 + (Real.cos x)^8 + 1 / (Real.sin x)^6 + (Real.cos x)^6 + 1

theorem minimum_value_is_14_div_27 :
  ∃ x : ℝ, minimum_value_expression x = (14 / 27) :=
by
  sorry

end minimum_value_is_14_div_27_l123_123034


namespace polynomial_coeff_divisible_by_5_l123_123405

theorem polynomial_coeff_divisible_by_5 (a b c d : ℤ) 
  (h : ∀ (x : ℤ), (a * x^3 + b * x^2 + c * x + d) % 5 = 0) : 
  a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 ∧ d % 5 = 0 := 
by
  sorry

end polynomial_coeff_divisible_by_5_l123_123405


namespace number_of_children_coming_to_show_l123_123409

theorem number_of_children_coming_to_show :
  ∀ (cost_adult cost_child : ℕ) (number_adults total_cost : ℕ),
  cost_adult = 12 →
  cost_child = 10 →
  number_adults = 3 →
  total_cost = 66 →
  ∃ (c : ℕ), 3 = c := by
    sorry

end number_of_children_coming_to_show_l123_123409


namespace simplify_expression_l123_123899

theorem simplify_expression (x y : ℝ) (hx : x = 5) (hy : y = 2) :
  (10 * x * y^3) / (15 * x^2 * y^2) = 4 / 15 :=
by
  rw [hx, hy]
  -- here we would simplify but leave a hole
  sorry

end simplify_expression_l123_123899


namespace points_five_from_origin_l123_123269

theorem points_five_from_origin (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 := by
  sorry

end points_five_from_origin_l123_123269


namespace find_rate_of_interest_l123_123615

theorem find_rate_of_interest (P SI : ℝ) (r : ℝ) (hP : P = 1200) (hSI : SI = 108) (ht : r = r) :
  SI = P * r * r / 100 → r = 3 := by
  intros
  sorry

end find_rate_of_interest_l123_123615


namespace symmetry_axis_of_function_l123_123445

theorem symmetry_axis_of_function {x : ℝ} :
  (∃ k : ℤ, ∃ x : ℝ, (y = 2 * (Real.cos ((x / 2) + (Real.pi / 3))) ^ 2 - 1) ∧ (x + (2 * Real.pi) / 3 = k * Real.pi)) →
    x = (Real.pi / 3) ∧ 0 = y :=
sorry

end symmetry_axis_of_function_l123_123445


namespace horse_food_needed_l123_123508

theorem horse_food_needed
  (ratio_sheep_horses : ℕ := 6)
  (ratio_horses_sheep : ℕ := 7)
  (horse_food_per_day : ℕ := 230)
  (sheep_on_farm : ℕ := 48)
  (units : ℕ := sheep_on_farm / ratio_sheep_horses)
  (horses_on_farm : ℕ := units * ratio_horses_sheep) :
  horses_on_farm * horse_food_per_day = 12880 := by
  sorry

end horse_food_needed_l123_123508


namespace parity_of_expression_l123_123383

theorem parity_of_expression (a b c : ℤ) (h : (a + b + c) % 2 = 1) : (a^2 + b^2 - c^2 + 2*a*b) % 2 = 1 :=
by
sorry

end parity_of_expression_l123_123383


namespace tangent_line_equation_inequality_range_l123_123385

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_equation :
  let x := Real.exp 1
  ∀ e : ℝ, e = Real.exp 1 → 
  ∀ y : ℝ, y = f (Real.exp 1) → 
  ∀ a b : ℝ, (y = a * Real.exp 1 + b) ∧ (a = 2) ∧ (b = -e) := sorry

theorem inequality_range (x : ℝ) (hx : x > 0) :
  (f x - 1/2 ≤ (3/2) * x^2 + a * x) → ∀ a : ℝ, a ≥ -2 := sorry

end tangent_line_equation_inequality_range_l123_123385


namespace maximum_value_of_f_l123_123198

noncomputable def f : ℝ → ℝ :=
  fun x => -x^2 * (x^2 + 4*x + 4)

theorem maximum_value_of_f :
  ∀ x : ℝ, x ≠ 0 → x ≠ -2 → x ≠ 1 → x ≠ -3 → f x ≤ 0 ∧ f 0 = 0 :=
by
  sorry

end maximum_value_of_f_l123_123198


namespace b_present_age_l123_123482

/-- 
In 10 years, A will be twice as old as B was 10 years ago. 
A is currently 8 years older than B. 
Prove that B's current age is 38.
--/
theorem b_present_age (a b : ℕ) 
  (h1 : a + 10 = 2 * (b - 10)) 
  (h2 : a = b + 8) : 
  b = 38 := 
  sorry

end b_present_age_l123_123482


namespace inscribed_circle_circumference_l123_123156

theorem inscribed_circle_circumference (side_length : ℝ) (h : side_length = 10) : 
  ∃ C : ℝ, C = 2 * Real.pi * (side_length / 2) ∧ C = 10 * Real.pi := 
by 
  sorry

end inscribed_circle_circumference_l123_123156


namespace cos_240_eq_neg_half_l123_123285

theorem cos_240_eq_neg_half (h1 : Real.cos (180 * Real.pi / 180) = -1)
                            (h2 : Real.sin (180 * Real.pi / 180) = 0)
                            (h3 : Real.cos (60 * Real.pi / 180) = 1 / 2) :
  Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end cos_240_eq_neg_half_l123_123285


namespace Terrence_earns_l123_123879

theorem Terrence_earns :
  ∀ (J T E : ℝ), J + T + E = 90 ∧ J = T + 5 ∧ E = 25 → T = 30 :=
by
  intro J T E
  intro h
  obtain ⟨h₁, h₂, h₃⟩ := h
  sorry -- proof steps go here

end Terrence_earns_l123_123879


namespace second_school_more_students_l123_123460

theorem second_school_more_students (S1 S2 S3 : ℕ) 
  (hS3 : S3 = 200) 
  (hS1 : S1 = 2 * S2) 
  (h_total : S1 + S2 + S3 = 920) : 
  S2 - S3 = 40 :=
by
  sorry

end second_school_more_students_l123_123460


namespace half_abs_diff_squares_l123_123652

theorem half_abs_diff_squares (a b : ℤ) (ha : a = 20) (hb : b = 15) : 
  (|a^2 - b^2| / 2) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l123_123652


namespace range_of_k_l123_123886

theorem range_of_k :
  ∀ (a k : ℝ) (f : ℝ → ℝ),
    (∀ x, f x = if x ≥ 0 then k^2 * x + a^2 - k else x^2 + (a^2 + 4 * a) * x + (2 - a)^2) →
    (∀ x1 x2 : ℝ, x1 ≠ 0 → x2 ≠ 0 → x1 ≠ x2 → f x1 = f x2 → False) →
    -20 ≤ k ∧ k ≤ -4 :=
by
  sorry

end range_of_k_l123_123886


namespace number_divisible_by_23_and_29_l123_123233

theorem number_divisible_by_23_and_29 (a b c : ℕ) (ha : a < 10) (hb : b < 10) (hc : c < 10) :
  23 ∣ (200100 * a + 20010 * b + 2001 * c) ∧ 29 ∣ (200100 * a + 20010 * b + 2001 * c) :=
by
  sorry

end number_divisible_by_23_and_29_l123_123233


namespace maximal_n_for_quadratic_factorization_l123_123723

theorem maximal_n_for_quadratic_factorization :
  ∃ n, n = 325 ∧ (∃ A B : ℤ, A * B = 108 ∧ n = 3 * B + A) :=
by
  use 325
  use 1, 108
  constructor
  · rfl
  constructor
  · norm_num
  · norm_num
  sorry

end maximal_n_for_quadratic_factorization_l123_123723


namespace point_in_first_quadrant_l123_123207

theorem point_in_first_quadrant (x y : ℝ) (hx : x = 6) (hy : y = 2) : x > 0 ∧ y > 0 :=
by
  rw [hx, hy]
  exact ⟨by norm_num, by norm_num⟩

end point_in_first_quadrant_l123_123207


namespace gcd_pow_minus_one_l123_123035

theorem gcd_pow_minus_one (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  Nat.gcd (2^m - 1) (2^n - 1) = 2^(Nat.gcd m n) - 1 :=
by
  sorry

end gcd_pow_minus_one_l123_123035


namespace cos_240_eq_neg_half_l123_123318

theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Sorry to skip the proof
  sorry

end cos_240_eq_neg_half_l123_123318


namespace monkey2_peach_count_l123_123588

noncomputable def total_peaches : ℕ := 81
def monkey1_share (p : ℕ) : ℕ := (5 * p) / 6
def remaining_after_monkey1 (p : ℕ) : ℕ := p - monkey1_share p
def monkey2_share (p : ℕ) : ℕ := (5 * remaining_after_monkey1 p) / 9
def remaining_after_monkey2 (p : ℕ) : ℕ := remaining_after_monkey1 p - monkey2_share p
def monkey3_share (p : ℕ) : ℕ := remaining_after_monkey2 p

theorem monkey2_peach_count : monkey2_share total_peaches = 20 :=
by
  sorry

end monkey2_peach_count_l123_123588


namespace question1_question2_l123_123036

def vector_m (x : ℝ) : ℝ × ℝ := (Real.sin (x - Real.pi / 3), 1)
def vector_n (x : ℝ) : ℝ × ℝ := (Real.cos x, 1)

noncomputable def f (x : ℝ) := (vector_m x).1 * (vector_n x).1 + (vector_m x).2 * (vector_n x).2

theorem question1 {x : ℝ} (h : Real.sin (x - Real.pi / 3) = Real.cos x) : Real.tan x = 2 + Real.sqrt 3 :=
sorry

theorem question2 : ∀ x : ℝ, (0 ≤ x ∧ x ≤ Real.pi / 2) → f x ≤ (6 - Real.sqrt 3) / 4 :=
sorry

end question1_question2_l123_123036


namespace fishing_boat_should_go_out_to_sea_l123_123940

def good_weather_profit : ℤ := 6000
def bad_weather_loss : ℤ := -8000
def stay_at_port_loss : ℤ := -1000

def prob_good_weather : ℚ := 0.6
def prob_bad_weather : ℚ := 0.4

def expected_profit_going : ℚ :=  prob_good_weather * good_weather_profit + prob_bad_weather * bad_weather_loss
def expected_profit_staying : ℚ := stay_at_port_loss

theorem fishing_boat_should_go_out_to_sea : 
  expected_profit_going > expected_profit_staying :=
  sorry

end fishing_boat_should_go_out_to_sea_l123_123940


namespace expr_xn_add_inv_xn_l123_123038

theorem expr_xn_add_inv_xn {θ : ℝ} {x : ℂ} (h1 : 0 < θ) (h2 : θ < π) 
  (h3 : x + x⁻¹ = 2 * (Real.cos θ)) (n : ℤ) : x^n + (x^n)⁻¹ = 2 * (Real.cos (n * θ)) :=
sorry

end expr_xn_add_inv_xn_l123_123038


namespace solve_system_eq_l123_123904

theorem solve_system_eq (x y z : ℤ) :
  (x^2 - 23 * y + 66 * z + 612 = 0) ∧ 
  (y^2 + 62 * x - 20 * z + 296 = 0) ∧ 
  (z^2 - 22 * x + 67 * y + 505 = 0) →
  (x = -20) ∧ (y = -22) ∧ (z = -23) :=
by {
  sorry
}

end solve_system_eq_l123_123904


namespace cos_240_is_neg_half_l123_123345

noncomputable def cos_240 : ℝ := Real.cos (240 * Real.pi / 180)

theorem cos_240_is_neg_half : cos_240 = -1/2 := by
  sorry

end cos_240_is_neg_half_l123_123345


namespace percentage_fruits_in_good_condition_l123_123479

theorem percentage_fruits_in_good_condition (oranges bananas : ℕ) (rotten_oranges_pct rotten_bananas_pct : ℚ)
    (h_oranges : oranges = 600) (h_bananas : bananas = 400)
    (h_rotten_oranges_pct : rotten_oranges_pct = 0.15) (h_rotten_bananas_pct : rotten_bananas_pct = 0.06) :
    let rotten_oranges := (rotten_oranges_pct * oranges : ℚ)
    let rotten_bananas := (rotten_bananas_pct * bananas : ℚ)
    let total_rotten := rotten_oranges + rotten_bananas
    let total_fruits := (oranges + bananas : ℚ)
    let good_fruits := total_fruits - total_rotten
    let percentage_good_fruits := (good_fruits / total_fruits) * 100
    percentage_good_fruits = 88.6 :=
by
    sorry

end percentage_fruits_in_good_condition_l123_123479


namespace linear_equation_unique_l123_123671

theorem linear_equation_unique (x y : ℝ) : 
  (3 * x = 2 * y) ∧ 
  ¬(3 * x - 6 = x) ∧ 
  ¬(x - 1 / y = 0) ∧ 
  ¬(2 * x - 3 * y = x * y) :=
by
  sorry

end linear_equation_unique_l123_123671


namespace min_value_a_plus_b_l123_123958

theorem min_value_a_plus_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : a^2 ≥ 8 * b) (h4 : b^2 ≥ a) : a + b ≥ 6 := by
  sorry

end min_value_a_plus_b_l123_123958


namespace cost_of_coat_eq_l123_123916

-- Define the given conditions
def total_cost : ℕ := 110
def cost_of_shoes : ℕ := 30
def cost_per_jeans : ℕ := 20
def num_of_jeans : ℕ := 2

-- Define the cost calculation for the jeans
def cost_of_jeans : ℕ := num_of_jeans * cost_per_jeans

-- Define the known total cost (shoes and jeans)
def known_total_cost : ℕ := cost_of_shoes + cost_of_jeans

-- Prove James' coat cost
theorem cost_of_coat_eq :
  (total_cost - known_total_cost) = 40 :=
by
  sorry

end cost_of_coat_eq_l123_123916


namespace shooting_game_system_l123_123921

theorem shooting_game_system :
  ∃ x y : ℕ, (x + y = 20 ∧ 3 * x = y) :=
by
  sorry

end shooting_game_system_l123_123921


namespace total_initial_seashells_l123_123857

-- Definitions for the conditions
def Henry_seashells := 11
def Paul_seashells := 24

noncomputable def Leo_initial_seashells (total_seashells : ℕ) :=
  (total_seashells - (Henry_seashells + Paul_seashells)) * 4 / 3

theorem total_initial_seashells 
  (total_seashells_now : ℕ)
  (leo_shared_fraction : ℕ → ℕ)
  (h : total_seashells_now = 53) : 
  Henry_seashells + Paul_seashells + leo_shared_fraction 53 = 59 :=
by
  let L := Leo_initial_seashells 53
  have L_initial : L = 24 := by sorry
  exact sorry

end total_initial_seashells_l123_123857


namespace incorrect_statements_l123_123124

-- Defining the first condition
def condition1 : Prop :=
  let a_sq := 169
  let b_sq := 144
  let c_sq := a_sq - b_sq
  let c_ := Real.sqrt c_sq
  let focal_points := [(0, c_), (0, -c_)]
  ¬((c_, 0) ∈ focal_points) ∧ ¬((-c_, 0) ∈ focal_points)

-- Defining the second condition
def condition2 : Prop :=
  let m := 1  -- Example choice since m is unspecified
  let a_sq := m^2 + 1
  let b_sq := m^2
  let c_sq := a_sq - b_sq
  let c_ := Real.sqrt c_sq
  let focal_points := [(0, c_), (0, -c_)]
  (0, 1) ∈ focal_points ∧ (0, -1) ∈ focal_points

-- Defining the third condition
def condition3 : Prop :=
  let a1_sq := 16
  let b1_sq := 7
  let c1_sq := a1_sq - b1_sq
  let c1_ := Real.sqrt c1_sq
  let focal_points1 := [(c1_, 0), (-c1_, 0)]
  
  let m := 10  -- Example choice since m > 0 is unspecified
  let a2_sq := m - 5
  let b2_sq := m + 4
  let c2_sq := a2_sq - b2_sq
  let focal_points2 := [(0, Real.sqrt c2_sq), (0, -Real.sqrt c2_sq)]
  
  ¬ (focal_points1 = focal_points2)

-- Defining the fourth condition
def condition4 : Prop :=
  let B := (-3, 0)
  let C := (3, 0)
  let BC := (C.1 - B.1, C.2 - B.2)
  let BC_dist := Real.sqrt (BC.1^2 + BC.2^2)
  let A_locus_eq := ∀ (x y : ℝ), x^2 / 36 + y^2 / 27 = 1
  2 * BC_dist = 12

-- Proof verification
theorem incorrect_statements : Prop :=
  condition1 ∧ condition3

end incorrect_statements_l123_123124


namespace earth_surface_inhabitable_fraction_l123_123440

theorem earth_surface_inhabitable_fraction :
  (1 / 3 : ℝ) * (2 / 3 : ℝ) = 2 / 9 := 
by 
  sorry

end earth_surface_inhabitable_fraction_l123_123440


namespace matrix_power_application_l123_123597

variable (B : Matrix (Fin 2) (Fin 2) ℝ)
variable (v : Fin 2 → ℝ := ![4, -3])

theorem matrix_power_application :
  (B.mulVec v = ![8, -6]) →
  (B ^ 4).mulVec v = ![64, -48] :=
by
  intro h
  sorry

end matrix_power_application_l123_123597


namespace remi_water_consumption_proof_l123_123437

-- Definitions for the conditions
def daily_consumption (bottle_volume : ℕ) (refills_per_day : ℕ) : ℕ :=
  bottle_volume * refills_per_day

def total_spillage (spill1 : ℕ) (spill2 : ℕ) : ℕ :=
  spill1 + spill2

def total_consumption (daily : ℕ) (days : ℕ) (spill : ℕ) : ℕ :=
  (daily * days) - spill

-- Theorem proving the number of days d
theorem remi_water_consumption_proof (bottle_volume : ℕ) (refills_per_day : ℕ)
  (spill1 spill2 total_water : ℕ) (d : ℕ)
  (h1 : bottle_volume = 20) (h2 : refills_per_day = 3)
  (h3 : spill1 = 5) (h4 : spill2 = 8)
  (h5 : total_water = 407) :
  total_consumption (daily_consumption bottle_volume refills_per_day) d
    (total_spillage spill1 spill2) = total_water → d = 7 := 
by
  -- Assuming the hypotheses to show the equality
  intro h
  have daily := h1 ▸ h2 ▸ 20 * 3 -- ⇒ daily = 60
  have spillage := h3 ▸ h4 ▸ 5 + 8 -- ⇒ spillage = 13
  rw [daily_consumption, total_spillage, h5] at h
  rw [h1, h2, h3, h4] at h -- Substitute conditions in the hypothesis
  sorry -- place a placeholder for the actual proof

end remi_water_consumption_proof_l123_123437


namespace total_peanuts_is_388_l123_123408

def peanuts_total (jose kenya marcos : ℕ) : ℕ :=
  jose + kenya + marcos

theorem total_peanuts_is_388 :
  ∀ (jose kenya marcos : ℕ),
    (jose = 85) →
    (kenya = jose + 48) →
    (marcos = kenya + 37) →
    peanuts_total jose kenya marcos = 388 := 
by
  intros jose kenya marcos h_jose h_kenya h_marcos
  sorry

end total_peanuts_is_388_l123_123408


namespace piravena_trip_total_cost_l123_123094

-- Define the distances
def d_A_to_B : ℕ := 4000
def d_B_to_C : ℕ := 3000

-- Define the costs per kilometer
def bus_cost_per_km : ℝ := 0.15
def airplane_cost_per_km : ℝ := 0.12
def airplane_booking_fee : ℝ := 120

-- Define the individual costs and the total cost
def cost_A_to_B : ℝ := d_A_to_B * airplane_cost_per_km + airplane_booking_fee
def cost_B_to_C : ℝ := d_B_to_C * bus_cost_per_km
def total_cost : ℝ := cost_A_to_B + cost_B_to_C

-- Define the theorem we want to prove
theorem piravena_trip_total_cost :
  total_cost = 1050 := sorry

end piravena_trip_total_cost_l123_123094


namespace accurate_measurement_l123_123235

-- Define the properties of Dr. Sharadek's tape
structure SharadekTape where
  startsWithHalfCM : Bool -- indicates if the tape starts with a half-centimeter bracket
  potentialError : ℝ -- potential measurement error

-- Define the conditions as an instance of the structure
noncomputable def drSharadekTape : SharadekTape :=
  { startsWithHalfCM := true,
    potentialError := 0.5 }

-- Define a segment with a known precise measurement
structure Segment where
  length : ℝ

noncomputable def AB (N : ℕ) : Segment :=
  { length := N + 0.5 }

-- The theorem stating the correct answer under the given conditions
theorem accurate_measurement (N : ℕ) : 
  ∃ AB : Segment, AB.length = N + 0.5 :=
by
  existsi AB N
  exact rfl

end accurate_measurement_l123_123235


namespace incorrect_statement_C_l123_123566

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin x - Real.sin (2 * x) + Real.sqrt 3

theorem incorrect_statement_C :
  ¬ ∃ x, f x = -Real.sqrt 3 :=
by
  sorry

end incorrect_statement_C_l123_123566


namespace intersection_solution_l123_123079

-- Define lines
def line1 (x : ℝ) : ℝ := -x + 4
def line2 (x : ℝ) (m : ℝ) : ℝ := 2 * x + m

-- Define system of equations
def system1 (x y : ℝ) : Prop := x + y = 4
def system2 (x y m : ℝ) : Prop := 2 * x - y + m = 0

-- Proof statement
theorem intersection_solution (m : ℝ) (n : ℝ) :
  (system1 3 n) ∧ (system2 3 n m) ∧ (line1 3 = n) ∧ (line2 3 m = n) →
  (3, n) = (3, 1) :=
  by 
  -- The proof would go here
  sorry

end intersection_solution_l123_123079


namespace half_abs_diff_squares_l123_123657

theorem half_abs_diff_squares : (1 / 2) * |20^2 - 15^2| = 87.5 :=
by
  sorry

end half_abs_diff_squares_l123_123657


namespace value_of_f_g_of_5_l123_123580

def g (x : ℤ) : ℤ := 4 * x + 6
def f (x : ℤ) : ℤ := x^2 - 4 * x - 5

theorem value_of_f_g_of_5 : f (g 5) = 567 := by
  sorry

end value_of_f_g_of_5_l123_123580


namespace find_m_and_p_l123_123380

-- Definition of a point being on the parabola y^2 = 2px
def on_parabola (m : ℝ) (p : ℝ) : Prop :=
  (-3)^2 = 2 * p * m

-- Definition of the distance from the point (m, -3) to the focus being 5
def distance_to_focus (m : ℝ) (p : ℝ) : Prop :=
  m + p / 2 = 5

theorem find_m_and_p (m p : ℝ) (hp : 0 < p) : 
  (on_parabola m p) ∧ (distance_to_focus m p) → 
  (m = 1 / 2 ∧ p = 9) ∨ (m = 9 / 2 ∧ p = 1) :=
by
  sorry

end find_m_and_p_l123_123380


namespace carolyn_marbles_l123_123960

theorem carolyn_marbles (initial_marbles : ℕ) (shared_items : ℕ) (end_marbles: ℕ) : 
  initial_marbles = 47 → shared_items = 42 → end_marbles = initial_marbles - shared_items → end_marbles = 5 :=
by
  intros h₀ h₁ h₂
  rw [h₀, h₁] at h₂
  exact h₂

end carolyn_marbles_l123_123960


namespace max_d_n_is_one_l123_123243

open Int

/-- The sequence definition -/
def seq (n : ℕ) : ℤ := 100 + n^3

/-- The definition of d_n -/
def d_n (n : ℕ) : ℤ := gcd (seq n) (seq (n + 1))

/-- The theorem stating the maximum value of d_n for positive integers is 1 -/
theorem max_d_n_is_one : ∀ (n : ℕ), 1 ≤ n → d_n n = 1 := by
  sorry

end max_d_n_is_one_l123_123243


namespace square_units_digit_l123_123142

theorem square_units_digit (n : ℕ) (h : (n^2 / 10) % 10 = 7) : n^2 % 10 = 6 := 
sorry

end square_units_digit_l123_123142


namespace circle_equation_tangent_to_line_l123_123101

def circle_center : (ℝ × ℝ) := (3, -1)
def tangent_line (x y : ℝ) : Prop := 3 * x + 4 * y = 0

/-- The equation of the circle with center at (3, -1) and tangent to the line 3x + 4y = 0 is (x - 3)^2 + (y + 1)^2 = 1 -/
theorem circle_equation_tangent_to_line : 
  ∃ r, ∀ x y: ℝ, ((x - 3)^2 + (y + 1)^2 = r^2) ∧ (∀ (cx cy: ℝ), cx = 3 → cy = -1 → (tangent_line cx cy → r = 1)) :=
by
  sorry

end circle_equation_tangent_to_line_l123_123101


namespace power_mod_l123_123803

theorem power_mod (n m : ℕ) (hn : n = 13) (hm : m = 1000) : n ^ 21 % m = 413 :=
by
  rw [hn, hm]
  -- other steps of the proof would go here...
  sorry

end power_mod_l123_123803


namespace quadratic_has_two_distinct_real_roots_l123_123584

theorem quadratic_has_two_distinct_real_roots (m : ℝ) : 
  (∃ (x y : ℝ), x ≠ y ∧ x^2 - 2 * x + m = 0 ∧ y^2 - 2 * y + m = 0) ↔ m < 1 :=
sorry

end quadratic_has_two_distinct_real_roots_l123_123584


namespace simplify_expression_l123_123510

variable {R : Type*} [CommRing R] (x y : R)

theorem simplify_expression :
  (x - 2 * y) * (x + 2 * y) - x * (x - y) = -4 * y ^ 2 + x * y :=
by
  sorry

end simplify_expression_l123_123510


namespace solve_system_l123_123080

theorem solve_system :
  ∀ (x y m : ℝ), 
    (∀ P : ℝ × ℝ, P = (3,1) → 
      (P.2 = -P.1 + 4) ∧ (P.2 = 2 * P.1 + m)) → 
    x = 3 ∧ y = 1 ↔ (x + y - 4 = 0 ∧ 2*x - y + m = 0) :=
by
  intros x y m h
  split
  case mp =>
    intro hxy
    cases hxy
    use hxy_left, hxy_right
    have hP : (3,1) = (3 : ℝ, 1 : ℝ) := rfl
    specialize h (3,1) hP
    cases h with h1 h2
    rw [h1, h2],
    exact ⟨by simp, by linarith⟩
  case mpr =>
    intro hsys
    use 3, 1
    split
    case hp1 =>
      exact (by linarith : 3 + 1 - 4 = 0)
    case hp2 =>
      rw [← h.2 3 1 _ ⟨rfl, rfl⟩],
      simp,
    exact ⟨rfl, rfl⟩

end solve_system_l123_123080


namespace opposite_of_neg_one_third_l123_123792

noncomputable def a : ℚ := -1 / 3

theorem opposite_of_neg_one_third : -a = 1 / 3 := 
by 
sorry

end opposite_of_neg_one_third_l123_123792


namespace prove_tan_sum_is_neg_sqrt3_l123_123039

open Real

-- Given conditions as definitions
def condition1 (α β : ℝ) : Prop := 0 < α ∧ α < π ∧ 0 < β ∧ β < π
def condition2 (α β : ℝ) : Prop := sin α + sin β = sqrt 3 * (cos α + cos β)

-- The statement of the proof
theorem prove_tan_sum_is_neg_sqrt3 (α β : ℝ) (h1 : condition1 α β) (h2 : condition2 α β) :
  tan (α + β) = -sqrt 3 :=
sorry

end prove_tan_sum_is_neg_sqrt3_l123_123039


namespace expenditure_recorded_neg_20_l123_123194

-- Define the condition where income of 60 yuan is recorded as +60 yuan
def income_recorded (income : ℤ) : ℤ :=
  income

-- Define what expenditure is given the condition
def expenditure_recorded (expenditure : ℤ) : ℤ :=
  -expenditure

-- Prove that an expenditure of 20 yuan is recorded as -20 yuan
theorem expenditure_recorded_neg_20 :
  expenditure_recorded 20 = -20 :=
by
  sorry

end expenditure_recorded_neg_20_l123_123194


namespace cost_price_of_article_l123_123578

theorem cost_price_of_article (C SP1 SP2 G1 G2 : ℝ) 
  (h_SP1 : SP1 = 160) 
  (h_SP2 : SP2 = 220) 
  (h_gain_relation : G2 = 1.05 * G1) 
  (h_G1 : G1 = SP1 - C) 
  (h_G2 : G2 = SP2 - C) : C = 1040 :=
by
  sorry

end cost_price_of_article_l123_123578


namespace intersection_nonempty_implies_range_l123_123390

namespace ProofProblem

def M (x y : ℝ) : Prop := x + y + 1 ≥ Real.sqrt (2 * (x^2 + y^2))
def N (a x y : ℝ) : Prop := |x - a| + |y - 1| ≤ 1

theorem intersection_nonempty_implies_range (a : ℝ) :
  (∃ x y : ℝ, M x y ∧ N a x y) → (1 - Real.sqrt 6 ≤ a ∧ a ≤ 3 + Real.sqrt 10) :=
by
  sorry

end ProofProblem

end intersection_nonempty_implies_range_l123_123390


namespace half_abs_diff_squares_l123_123660

theorem half_abs_diff_squares (a b : ℤ) (h₁ : a = 20) (h₂ : b = 15) : 
  (|a^2 - b^2| / 2 : ℚ) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l123_123660


namespace eliza_irons_dress_in_20_minutes_l123_123519

def eliza_iron_time : Prop :=
∃ d : ℕ, 
  (d ≠ 0 ∧  -- To avoid division by zero
  8 + 180 / d = 17 ∧
  d = 20)

theorem eliza_irons_dress_in_20_minutes : eliza_iron_time :=
sorry

end eliza_irons_dress_in_20_minutes_l123_123519


namespace operation_preserves_remainder_l123_123008

theorem operation_preserves_remainder (N : ℤ) (k : ℤ) (m : ℤ) 
(f : ℤ → ℤ) (hN : N = 6 * k + 3) (hf : f N = 6 * m + 3) : f N % 6 = 3 :=
by
  sorry

end operation_preserves_remainder_l123_123008


namespace find_a_plus_d_l123_123810

theorem find_a_plus_d (a b c d : ℕ)
  (h1 : a + b = 14)
  (h2 : b + c = 9)
  (h3 : c + d = 3) : 
  a + d = 2 :=
by sorry

end find_a_plus_d_l123_123810


namespace min_value_ineq_l123_123887

theorem min_value_ineq (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) : 
  (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ 3 / 2 := 
sorry

end min_value_ineq_l123_123887


namespace denominator_is_five_l123_123254

-- Define the conditions
variables (n d : ℕ)
axiom h1 : d = n - 4
axiom h2 : n + 6 = 3 * d

-- The theorem that needs to be proven
theorem denominator_is_five : d = 5 :=
by
  sorry

end denominator_is_five_l123_123254


namespace median_a_sq_correct_sum_of_medians_sq_l123_123128

noncomputable def median_a_sq (a b c : ℝ) := (2 * b^2 + 2 * c^2 - a^2) / 4
noncomputable def median_b_sq (a b c : ℝ) := (2 * a^2 + 2 * c^2 - b^2) / 4
noncomputable def median_c_sq (a b c : ℝ) := (2 * a^2 + 2 * b^2 - c^2) / 4

theorem median_a_sq_correct (a b c : ℝ) : 
  median_a_sq a b c = (2 * b^2 + 2 * c^2 - a^2) / 4 :=
sorry

theorem sum_of_medians_sq (a b c : ℝ) :
  median_a_sq a b c + median_b_sq a b c + median_c_sq a b c = 
  3 * (a^2 + b^2 + c^2) / 4 :=
sorry

end median_a_sq_correct_sum_of_medians_sq_l123_123128


namespace rhombus_area_is_correct_l123_123522

def calculate_rhombus_area (d1 d2 : ℕ) : ℕ :=
  (d1 * d2) / 2

theorem rhombus_area_is_correct :
  calculate_rhombus_area (3 * 6) (3 * 4) = 108 := by
  sorry

end rhombus_area_is_correct_l123_123522


namespace series_problem_l123_123699

theorem series_problem (m : ℝ) :
  let a₁ := 9
  let a₂ := 3
  let b₁ := 9
  let b₂ := 3 + m
  let S₁ := a₁ / (1 - (a₂ / a₁))
  let S₂ := b₁ / (1 - (b₂ / b₁))
  S₂ = 3 * S₁ → m = 4 :=
by
  let a₁ := 9
  let a₂ := 3
  let b₁ := 9
  let b₂ := 3 + m
  let S₁ := a₁ / (1 - (a₂ / a₁))
  let S₂ := b₁ / (1 - (b₂ / b₁))
  have h_sum_equal : S₂ = 3 * S₁ := by assumption
  sorry

end series_problem_l123_123699


namespace felix_chopped_down_trees_l123_123367

theorem felix_chopped_down_trees
  (sharpening_cost : ℕ)
  (trees_per_sharpening : ℕ)
  (total_spent : ℕ)
  (times_sharpened : ℕ)
  (trees_chopped_down : ℕ)
  (h1 : sharpening_cost = 5)
  (h2 : trees_per_sharpening = 13)
  (h3 : total_spent = 35)
  (h4 : times_sharpened = total_spent / sharpening_cost)
  (h5 : trees_chopped_down = trees_per_sharpening * times_sharpened) :
  trees_chopped_down ≥ 91 :=
by
  sorry

end felix_chopped_down_trees_l123_123367


namespace SetC_not_right_angled_triangle_l123_123956

theorem SetC_not_right_angled_triangle :
  ¬ (7^2 + 24^2 = 26^2) :=
by 
  have h : 7^2 + 24^2 ≠ 26^2 := by decide
  exact h

end SetC_not_right_angled_triangle_l123_123956


namespace james_fraction_of_pizza_slices_l123_123593

theorem james_fraction_of_pizza_slices :
  (2 * 6 = 12) ∧ (8 / 12 = 2 / 3) :=
by
  sorry

end james_fraction_of_pizza_slices_l123_123593


namespace relationship_between_a_b_c_l123_123373

-- Given values
def a := Real.logb 2 5
def b := Real.logb 3 11
def c := 5 / 2

-- Theorem statement
theorem relationship_between_a_b_c : c > a ∧ a > b := by
  sorry

end relationship_between_a_b_c_l123_123373


namespace smallest_two_digit_prime_with_conditions_l123_123168

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ is_prime n

theorem smallest_two_digit_prime_with_conditions :
  ∃ p : ℕ, is_prime p ∧ 10 ≤ p ∧ p < 100 ∧ (p / 10 = 3) ∧ is_composite (((p % 10) * 10) + (p / 10) + 5) ∧ p = 31 :=
by
  sorry

end smallest_two_digit_prime_with_conditions_l123_123168


namespace investment_duration_l123_123012

noncomputable def log (x : ℝ) := Real.log x

theorem investment_duration 
  (P A : ℝ) 
  (r : ℝ) 
  (n : ℕ) 
  (t : ℝ) 
  (hP : P = 3000) 
  (hA : A = 3630) 
  (hr : r = 0.10) 
  (hn : n = 1) 
  (ht : A = P * (1 + r / n) ^ (n * t)) :
  t = 2 :=
by
  sorry

end investment_duration_l123_123012


namespace smallest_possible_value_of_AP_plus_BP_l123_123885

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

theorem smallest_possible_value_of_AP_plus_BP :
  let A := (1, 0)
  let B := (-3, 4)
  ∃ P : ℝ × ℝ, (P.2 ^ 2 = 4 * P.1) ∧
  (distance A P + distance B P = 12) :=
by
  -- proof steps would go here
  sorry

end smallest_possible_value_of_AP_plus_BP_l123_123885


namespace half_abs_diff_squares_l123_123661

theorem half_abs_diff_squares (a b : ℤ) (h₁ : a = 20) (h₂ : b = 15) : 
  (|a^2 - b^2| / 2 : ℚ) = 87.5 :=
by
  sorry

end half_abs_diff_squares_l123_123661


namespace find_number_l123_123815

theorem find_number (x : ℝ) (h : x = 0.16 * x + 21) : x = 25 :=
by
  sorry

end find_number_l123_123815


namespace eliza_received_12_almonds_l123_123855

theorem eliza_received_12_almonds (y : ℕ) (h1 : y - 8 = y / 3) : y = 12 :=
sorry

end eliza_received_12_almonds_l123_123855


namespace find_dividend_l123_123374

noncomputable def quotient : ℕ := 2015
noncomputable def remainder : ℕ := 0
noncomputable def divisor : ℕ := 105

theorem find_dividend : quotient * divisor + remainder = 20685 := by
  sorry

end find_dividend_l123_123374


namespace total_money_spent_l123_123161

variables (emma_spent : ℕ) (elsa_spent : ℕ) (elizabeth_spent : ℕ)
variables (total_spent : ℕ)

-- Conditions
def EmmaSpending : Prop := emma_spent = 58
def ElsaSpending : Prop := elsa_spent = 2 * emma_spent
def ElizabethSpending : Prop := elizabeth_spent = 4 * elsa_spent
def TotalSpending : Prop := total_spent = emma_spent + elsa_spent + elizabeth_spent

-- The theorem to prove
theorem total_money_spent 
  (h1 : EmmaSpending) 
  (h2 : ElsaSpending) 
  (h3 : ElizabethSpending) 
  (h4 : TotalSpending) : 
  total_spent = 638 := 
sorry

end total_money_spent_l123_123161


namespace certain_number_is_48_l123_123002

theorem certain_number_is_48 (x : ℕ) (h : x = 4) : 36 + 3 * x = 48 := by
  sorry

end certain_number_is_48_l123_123002


namespace max_four_digit_prime_product_l123_123763

theorem max_four_digit_prime_product :
  ∃ (x y : ℕ) (n : ℕ), x < 5 ∧ y < 5 ∧ x ≠ y ∧ Prime x ∧ Prime y ∧ Prime (10 * x + y) ∧ n = x * y * (10 * x + y) ∧ n = 138 :=
by
  sorry

end max_four_digit_prime_product_l123_123763


namespace expected_value_is_correct_l123_123686

noncomputable def expected_winnings : ℝ :=
  (1/12 : ℝ) * (9 + 8 + 7 + 6 + 5 + 1 + 2 + 3 + 4 + 5 + 6 + 7)

theorem expected_value_is_correct : expected_winnings = 5.25 := by
  sorry

end expected_value_is_correct_l123_123686


namespace man_speed_is_correct_l123_123272

noncomputable def speed_of_man (train_length : ℝ) (train_speed : ℝ) (cross_time : ℝ) : ℝ :=
  let train_speed_m_s := train_speed * (1000 / 3600)
  let relative_speed := train_length / cross_time
  let man_speed_m_s := relative_speed - train_speed_m_s
  man_speed_m_s * (3600 / 1000)

theorem man_speed_is_correct :
  speed_of_man 210 25 28 = 2 := by
  sorry

end man_speed_is_correct_l123_123272


namespace ribbon_segment_length_l123_123673

theorem ribbon_segment_length :
  ∀ (ribbon_length : ℚ) (segments : ℕ), ribbon_length = 4/5 → segments = 3 → 
  (ribbon_length / segments) = 4/15 :=
by
  intros ribbon_length segments h1 h2
  sorry

end ribbon_segment_length_l123_123673


namespace repeating_decimal_exceeds_decimal_representation_l123_123829

noncomputable def repeating_decimal : ℚ := 71 / 99
def decimal_representation : ℚ := 71 / 100

theorem repeating_decimal_exceeds_decimal_representation :
  repeating_decimal - decimal_representation = 71 / 9900 := by
  sorry

end repeating_decimal_exceeds_decimal_representation_l123_123829


namespace possible_values_count_l123_123982

theorem possible_values_count {x y z : ℤ} (h₁ : x = 5) (h₂ : y = -3) (h₃ : z = -1) :
  ∃ v, v = x - y - z ∧ (v = 7 ∨ v = 8 ∨ v = 9) :=
by
  sorry

end possible_values_count_l123_123982


namespace find_second_term_l123_123484

theorem find_second_term (A B : ℕ) (h1 : A / B = 3 / 4) (h2 : (A + 10) / (B + 10) = 4 / 5) : B = 40 :=
sorry

end find_second_term_l123_123484


namespace remaining_inventory_l123_123834

def initial_inventory : Int := 4500
def bottles_sold_mon : Int := 2445
def bottles_sold_tue : Int := 906
def bottles_sold_wed : Int := 215
def bottles_sold_thu : Int := 457
def bottles_sold_fri : Int := 312
def bottles_sold_sat : Int := 239
def bottles_sold_sun : Int := 188

def bottles_received_tue : Int := 350
def bottles_received_thu : Int := 750
def bottles_received_sat : Int := 981

def total_bottles_sold : Int := bottles_sold_mon + bottles_sold_tue + bottles_sold_wed + bottles_sold_thu + bottles_sold_fri + bottles_sold_sat + bottles_sold_sun
def total_bottles_received : Int := bottles_received_tue + bottles_received_thu + bottles_received_sat

theorem remaining_inventory (initial_inventory bottles_sold_mon bottles_sold_tue bottles_sold_wed bottles_sold_thu bottles_sold_fri bottles_sold_sat bottles_sold_sun bottles_received_tue bottles_received_thu bottles_received_sat total_bottles_sold total_bottles_received : Int) :
  initial_inventory - total_bottles_sold + total_bottles_received = 819 :=
by
  sorry

end remaining_inventory_l123_123834


namespace second_person_percentage_of_Deshaun_l123_123835

variable (days : ℕ) (books_read_by_Deshaun : ℕ) (pages_per_book : ℕ) (pages_per_day_by_second_person : ℕ)

theorem second_person_percentage_of_Deshaun :
  days = 80 →
  books_read_by_Deshaun = 60 →
  pages_per_book = 320 →
  pages_per_day_by_second_person = 180 →
  ((pages_per_day_by_second_person * days) / (books_read_by_Deshaun * pages_per_book) * 100) = 75 := 
by
  intros days_eq books_eq pages_eq second_pages_eq
  rw [days_eq, books_eq, pages_eq, second_pages_eq]
  simp
  sorry

end second_person_percentage_of_Deshaun_l123_123835


namespace solve_ticket_problem_l123_123915

def ticket_problem : Prop :=
  ∃ S N : ℕ, S + N = 2000 ∧ 9 * S + 11 * N = 20960 ∧ S = 520

theorem solve_ticket_problem : ticket_problem :=
sorry

end solve_ticket_problem_l123_123915


namespace min_expression_value_l123_123033

theorem min_expression_value (x : ℝ) : 
  (sin x)^8 + (cos x)^8 + 1) / ((sin x)^6 + (cos x)^6 + 1) >= 1/2 := 
sorry

end min_expression_value_l123_123033


namespace find_values_of_a_l123_123055

def P : Set ℝ := { x | x^2 + x - 6 = 0 }
def S (a : ℝ) : Set ℝ := { x | a * x + 1 = 0 }

theorem find_values_of_a (a : ℝ) : (S a ⊆ P) ↔ (a = 0 ∨ a = 1/3 ∨ a = -1/2) := by
  sorry

end find_values_of_a_l123_123055


namespace number_of_valid_pairs_l123_123823

-- Definition of the conditions according to step (a)
def perimeter (l w : ℕ) : Prop := 2 * (l + w) = 80
def integer_lengths (l w : ℕ) : Prop := true
def length_greater_than_width (l w : ℕ) : Prop := l > w

-- The mathematical proof problem according to step (c)
theorem number_of_valid_pairs : ∃ n : ℕ, 
  (∀ l w : ℕ, perimeter l w → integer_lengths l w → length_greater_than_width l w → ∃! pair : (ℕ × ℕ), pair = (l, w)) ∧
  n = 19 :=
by 
  sorry

end number_of_valid_pairs_l123_123823


namespace find_base_l123_123078

theorem find_base (r : ℕ) (h1 : 5 * r^2 + 3 * r + 4 + 3 * r^2 + 6 * r + 6 = r^3) : r = 10 :=
by
  sorry

end find_base_l123_123078


namespace parallelogram_opposite_sides_equal_l123_123449

-- Definition of a parallelogram and its properties
structure Parallelogram (P : Type*) :=
  (a b c d : P)
  (opposite_sides_parallel : ∀ {x y : P}, (x = a ∧ y = b) ∨ (x = b ∧ y = c) ∨ (x = c ∧ y = d) ∨ (x = d ∧ y = a) → (x = a ∧ y = d) → x ∥ y)
  (opposite_sides_equal : ∀ {x y : P}, (x = a ∧ y = c) ∨ (x = b ∧ y = d) → x = y)
  (opposite_angles_equal : true)  -- true signifies that it is given as a property in the solution
  (diagonals_bisect_each_other : true) -- true signifies that it is given as a property in the solution

-- Lean statement to prove: indicative that opposite sides are equal
theorem parallelogram_opposite_sides_equal (P: Type*) (parallelogram: Parallelogram P):
  ∃ a b c d : P, parallelogram.opposite_sides_equal :=
by
  -- skipping the proof
  sorry

end parallelogram_opposite_sides_equal_l123_123449


namespace largest_n_for_factoring_l123_123727

theorem largest_n_for_factoring :
  ∃ n, (∃ A B : ℤ, (3 * A = 3 * 108 + 1) ∧ (/3 * B * 108 = 2) ∧ 
  (3 * 36 + 3 = 111) ∧ (3 * 108 + A = n) )=
  (n = 325) := sorry
iddenLean_formatter.clonecreateAngular

end largest_n_for_factoring_l123_123727


namespace total_apples_correct_l123_123711

def craig_initial := 20.5
def judy_initial := 11.25
def dwayne_initial := 17.85
def eugene_to_craig := 7.15
def craig_to_dwayne := 3.5 / 2
def judy_to_sally := judy_initial / 2

def craig_final := craig_initial + eugene_to_craig - craig_to_dwayne
def dwayne_final := dwayne_initial + craig_to_dwayne
def judy_final := judy_initial - judy_to_sally
def sally_final := judy_to_sally

def total_apples := craig_final + judy_final + dwayne_final + sally_final

theorem total_apples_correct : total_apples = 56.75 := by
  -- skipping proof
  sorry

end total_apples_correct_l123_123711


namespace mary_spent_on_jacket_l123_123604

def shirt_cost : ℝ := 13.04
def total_cost : ℝ := 25.31
def jacket_cost : ℝ := total_cost - shirt_cost

theorem mary_spent_on_jacket :
  jacket_cost = 12.27 := by
  sorry

end mary_spent_on_jacket_l123_123604


namespace solve_inequality_l123_123621

theorem solve_inequality (a : ℝ) :
  (a < 1 / 2 ∧ ∀ x : ℝ, x^2 - x + a - a^2 < 0 ↔ a < x ∧ x < 1 - a) ∨
  (a > 1 / 2 ∧ ∀ x : ℝ, x^2 - x + a - a^2 < 0 ↔ 1 - a < x ∧ x < a) ∨
  (a = 1 / 2 ∧ ∀ x : ℝ, x^2 - x + a - a^2 < 0 ↔ false) :=
sorry

end solve_inequality_l123_123621


namespace artifacts_in_each_wing_l123_123692

theorem artifacts_in_each_wing (total_wings : ℕ) (artifact_factor : ℕ) (painting_wings : ℕ) 
  (large_painting : ℕ) (small_paintings_per_wing : ℕ) (remaining_artifact_wings : ℕ) 
  (total_paintings constant_total_paintings_expected : ℕ) (total_artifacts total_artifacts_expected : ℕ) 
  (artifacts_per_wing artifacts_per_wing_expected : ℕ) : 

  total_wings = 8 →
  artifact_factor = 4 →
  painting_wings = 3 →
  large_painting = 1 →
  small_paintings_per_wing = 12 →
  remaining_artifact_wings = total_wings - painting_wings →
  total_paintings = painting_wings * small_paintings_per_wing + large_painting →
  total_artifacts = total_paintings * artifact_factor →
  artifacts_per_wing = total_artifacts / remaining_artifact_wings →
  artifacts_per_wing = 20 :=

by
    intros htotal_wings hartifact_factor hpainting_wings hlarge_painting hsmall_paintings_per_wing hermaining_artifact_wings htotal_paintings htotal_artifacts hartifacts_per_wing,
    sorry

end artifacts_in_each_wing_l123_123692


namespace value_of_b_l123_123998

theorem value_of_b (y b : ℝ) (hy : y > 0) (h : (4 * y) / b + (3 * y) / 10 = 0.5 * y) : b = 20 :=
by
  -- Proof omitted for brevity
  sorry

end value_of_b_l123_123998


namespace officers_count_l123_123759

theorem officers_count (average_salary_all : ℝ) (average_salary_officers : ℝ) 
    (average_salary_non_officers : ℝ) (num_non_officers : ℝ) (total_salary : ℝ) : 
    average_salary_all = 120 → 
    average_salary_officers = 470 →  
    average_salary_non_officers = 110 → 
    num_non_officers = 525 → 
    total_salary = average_salary_all * (num_non_officers + O) → 
    total_salary = average_salary_officers * O + average_salary_non_officers * num_non_officers → 
    O = 15 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end officers_count_l123_123759


namespace partial_fraction_decomposition_product_l123_123977

theorem partial_fraction_decomposition_product :
  ∃ A B C : ℚ,
    (A + 2) * (A - 3) *
    (B - 2) * (B - 3) *
    (C - 2) * (C + 2) = x^2 - 12 ∧
    (A = -2) ∧
    (B = 2/5) ∧
    (C = 3/5) ∧
    (A * B * C = -12/25) :=
  sorry

end partial_fraction_decomposition_product_l123_123977


namespace time_difference_l123_123639

/-- The time on a digital clock is 5:55. We need to calculate the number
of minutes that will pass before the clock next shows a time with all digits identical,
which is 11:11. -/
theorem time_difference : 
  let t1 := 5 * 60 + 55  -- Time 5:55 in minutes past midnight
  let t2 := 11 * 60 + 11 -- Time 11:11 in minutes past midnight
  in t2 - t1 = 316 := 
by 
  let t1 := 5 * 60 + 55 
  let t2 := 11 * 60 + 11 
  have h : t2 - t1 = 316 := by sorry
  exact h

end time_difference_l123_123639


namespace sally_investment_l123_123232

theorem sally_investment (m : ℝ) (hmf : 0 ≤ m) 
  (total_investment : m + 7 * m = 200000) : 
  7 * m = 175000 :=
by
  -- Proof goes here
  sorry

end sally_investment_l123_123232


namespace range_g_a_values_l123_123980

noncomputable def g (x : ℝ) : ℝ := abs (x - 1) - abs (x - 2)

theorem range_g : ∀ x : ℝ, -1 ≤ g x ∧ g x ≤ 1 :=
sorry

theorem a_values (a : ℝ) : (∀ x : ℝ, g x < a^2 + a + 1) ↔ (a < -1 ∨ a > 1) :=
sorry

end range_g_a_values_l123_123980


namespace lunch_break_duration_l123_123894

theorem lunch_break_duration :
  ∃ L : ℝ, 
    ∀ (p h : ℝ),
      (9 - L) * (p + h) = 0.4 ∧
      (7 - L) * h = 0.3 ∧
      (12 - L) * p = 0.3 →
      L = 0.5 := by
  sorry

end lunch_break_duration_l123_123894


namespace hemisphere_surface_area_l123_123098

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (h : π * r^2 = 225 * π) : 
  2 * π * r^2 + π * r^2 = 675 * π := 
by
  sorry

end hemisphere_surface_area_l123_123098


namespace roses_picked_later_l123_123490

/-- Represents the initial number of roses the florist had. -/
def initial_roses : ℕ := 37

/-- Represents the number of roses the florist sold. -/
def sold_roses : ℕ := 16

/-- Represents the final number of roses the florist ended up with. -/
def final_roses : ℕ := 40

/-- Theorem which states the number of roses picked later is 19 given the conditions. -/
theorem roses_picked_later : (final_roses - (initial_roses - sold_roses)) = 19 :=
by
  -- proof steps are omitted, sorry as a placeholder
  sorry

end roses_picked_later_l123_123490


namespace pseudoprime_pow_minus_one_l123_123613

theorem pseudoprime_pow_minus_one (n : ℕ) (hpseudo : 2^n ≡ 2 [MOD n]) : 
  ∃ m : ℕ, 2^(2^n - 1) ≡ 1 [MOD (2^n - 1)] :=
by
  sorry

end pseudoprime_pow_minus_one_l123_123613


namespace max_value_sine_cosine_l123_123583

/-- If the maximum value of the function f(x) = 4 * sin x + a * cos x is 5, then a = ±3. -/
theorem max_value_sine_cosine (a : ℝ) :
  (∀ x : ℝ, 4 * Real.sin x + a * Real.cos x ≤ 5) →
  (∃ x : ℝ, 4 * Real.sin x + a * Real.cos x = 5) →
  a = 3 ∨ a = -3 :=
by
  sorry

end max_value_sine_cosine_l123_123583


namespace determine_OP_l123_123169

variables (a b c d q : ℝ)
variables (P : ℝ)
variables (h_ratio : (|a - P| / |P - d| = |b - P| / |P - c|))
variables (h_twice : P = 2 * q)

theorem determine_OP : P = 2 * q :=
sorry

end determine_OP_l123_123169


namespace tangent_line_to_circle_l123_123523

open Real

theorem tangent_line_to_circle (x y : ℝ) :
  ((x - 2) ^ 2 + (y + 1) ^ 2 = 9) ∧ ((x = -1) → (x = -1 ∧ y = 3) ∨ (y = (37 - 8*x) / 15)) :=
by {
  sorry
}

end tangent_line_to_circle_l123_123523


namespace minutes_before_noon_l123_123190

theorem minutes_before_noon (x : ℕ) (h1 : x = 40)
  (h2 : ∀ (t : ℕ), t = 180 - (x + 40) ∧ t = 3 * x) : x = 35 :=
by {
  sorry
}

end minutes_before_noon_l123_123190


namespace bailey_discount_l123_123152

noncomputable def discount_percentage (total_cost_without_discount amount_spent : ℝ) : ℝ :=
  ((total_cost_without_discount - amount_spent) / total_cost_without_discount) * 100

theorem bailey_discount :
  let guest_sets := 2
  let master_sets := 4
  let price_guest := 40
  let price_master := 50
  let amount_spent := 224
  let total_cost_without_discount := (guest_sets * price_guest) + (master_sets * price_master)
  discount_percentage total_cost_without_discount amount_spent = 20 := 
by
  sorry

end bailey_discount_l123_123152


namespace total_people_in_group_l123_123494

theorem total_people_in_group (men women children : ℕ)
  (h1 : men = 2 * women)
  (h2 : women = 3 * children)
  (h3 : children = 30) :
  men + women + children = 300 :=
by
  sorry

end total_people_in_group_l123_123494


namespace collinear_points_eq_sum_l123_123713

theorem collinear_points_eq_sum (a b : ℝ) :
  -- Collinearity conditions in ℝ³
  (∃ t1 t2 t3 t4 : ℝ,
    (2, a, b) = (a + t1 * (a - 2), 3 + t1 * (b - 3), b + t1 * (4 - b)) ∧
    (a, 3, b) = (a + t2 * (a - 2), 3 + t2 * (b - 3), b + t2 * (4 - b)) ∧
    (a, b, 4) = (a + t3 * (a - 2), 3 + t3 * (b - 3), b + t3 * (4 - b)) ∧
    (5, b, a) = (a + t4 * (a - 2), 3 + t4 * (b - 3), b + t4 * (4 - b))) →
  a + b = 9 :=
by
  sorry

end collinear_points_eq_sum_l123_123713


namespace max_n_for_factorable_quadratic_l123_123730

theorem max_n_for_factorable_quadratic :
  ∃ n : ℤ, (∀ x : ℤ, ∃ A B : ℤ, (3*x^2 + n*x + 108) = (3*x + A)*( x + B) ∧ A*B = 108 ∧ n = A + 3*B) ∧ n = 325 :=
by
  sorry

end max_n_for_factorable_quadratic_l123_123730


namespace product_less_by_nine_times_l123_123473

theorem product_less_by_nine_times (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : y < 10) : 
  (x * y) * 10 - x * y = 9 * (x * y) := 
by
  sorry

end product_less_by_nine_times_l123_123473


namespace consecutive_even_numbers_sum_is_3_l123_123244

-- Definitions from the conditions provided
def consecutive_even_numbers := [80, 82, 84]
def sum_of_numbers := 246

-- The problem is to prove that there are 3 consecutive even numbers summing up to 246
theorem consecutive_even_numbers_sum_is_3 :
  (consecutive_even_numbers.sum = sum_of_numbers) → consecutive_even_numbers.length = 3 :=
by
  sorry

end consecutive_even_numbers_sum_is_3_l123_123244


namespace probability_interval_chebyshev_l123_123908

theorem probability_interval_chebyshev (X : ℝ → ℝ) (a : ℝ) (σ² : ℝ) :
  (∀ x, 49.5 ≤ X x ≤ 50.5) →
  (a = 50) →
  (σ² = 0.1) →
  (∀ x, P(λ y, 49.5 ≤ y ∧ y ≤ 50.5) ≥ 0.6) :=
begin
  sorry
end

end probability_interval_chebyshev_l123_123908


namespace fewer_bees_than_flowers_l123_123456

theorem fewer_bees_than_flowers :
  (5 - 3 = 2) :=
by
  sorry

end fewer_bees_than_flowers_l123_123456


namespace cos_240_eq_negative_half_l123_123330

theorem cos_240_eq_negative_half : Real.cos (240 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_240_eq_negative_half_l123_123330


namespace arithmetic_geometric_proof_l123_123375

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a (n + 1) = a n + d

noncomputable def geometric_sequence (b : ℕ → ℤ) (r : ℤ) : Prop :=
∀ n, b (n + 1) = b n * r

theorem arithmetic_geometric_proof
  (a : ℕ → ℤ) (b : ℕ → ℤ) (d r : ℤ)
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_sequence b r)
  (h_cond1 : 3 * a 1 - a 8 * a 8 + 3 * a 15 = 0)
  (h_cond2 : a 8 = b 10):
  b 3 * b 17 = 36 :=
sorry

end arithmetic_geometric_proof_l123_123375


namespace two_digit_primes_with_digit_sum_ten_l123_123534

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, 2 ≤ m → m * m ≤ n → n % m ≠ 0

theorem two_digit_primes_with_digit_sum_ten :
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10}.card = 3 :=
by
  sorry

end two_digit_primes_with_digit_sum_ten_l123_123534


namespace total_liters_needed_to_fill_two_tanks_l123_123629

theorem total_liters_needed_to_fill_two_tanks (capacity : ℕ) (liters_first_tank : ℕ) (liters_second_tank : ℕ) (percent_filled : ℕ) :
  liters_first_tank = 300 → 
  liters_second_tank = 450 → 
  percent_filled = 45 → 
  capacity = (liters_second_tank * 100) / percent_filled → 
  1000 - 300 = 700 → 
  1000 - 450 = 550 → 
  700 + 550 = 1250 :=
by sorry

end total_liters_needed_to_fill_two_tanks_l123_123629


namespace nonagon_intersecting_lines_probability_l123_123364

open Classical
noncomputable theory

theorem nonagon_intersecting_lines_probability :
  let vertices := 9
  let total_lines := Nat.choose vertices 2
  let total_pairs := Nat.choose total_lines 2
  let total_intersecting_sets := Nat.choose vertices 4 - vertices
  let intersecting_line_pairs := total_intersecting_sets
  (total_intersecting_sets / total_pairs) = (13 / 70) :=
by
  sorry

end nonagon_intersecting_lines_probability_l123_123364


namespace longest_side_of_similar_triangle_l123_123239

theorem longest_side_of_similar_triangle (a b c : ℕ) (perimeter_similar : ℕ) (h₁ : a = 8) (h₂ : b = 10) (h₃ : c = 12) (h₄ : perimeter_similar = 150) : 
  ∃ x : ℕ, 12 * x = 60 :=
by {
  have side_sum := h₁.symm ▸ h₂.symm ▸ h₃.symm ▸ (8 + 10 + 12),  -- a + b + c = 8 + 10 + 12
  rw ←h₄ at side_sum,  -- replace 30 with 150
  use 5,               -- introduction of the ratio
  sorry                 -- steps to show the length of the longest side is 60
}

end longest_side_of_similar_triangle_l123_123239


namespace zero_sum_of_squares_eq_zero_l123_123059

theorem zero_sum_of_squares_eq_zero {a b : ℝ} (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
sorry

end zero_sum_of_squares_eq_zero_l123_123059


namespace proof_speed_of_man_in_still_water_l123_123689

def speed_of_man_in_still_water (v_m v_s : ℝ) : Prop :=
  50 / 4 = v_m + v_s ∧ 30 / 6 = v_m - v_s

theorem proof_speed_of_man_in_still_water (v_m v_s : ℝ) :
  speed_of_man_in_still_water v_m v_s → v_m = 8.75 :=
by
  intro h
  sorry

end proof_speed_of_man_in_still_water_l123_123689


namespace sum_of_purchases_l123_123212

variable (J : ℕ) (K : ℕ)

theorem sum_of_purchases :
  J = 230 →
  2 * J = K + 90 →
  J + K = 600 :=
by
  intros hJ hEq
  rw [hJ] at hEq
  sorry

end sum_of_purchases_l123_123212


namespace extreme_value_when_a_is_neg_one_range_of_a_for_f_non_positive_l123_123867

open Real

noncomputable def f (a x : ℝ) : ℝ := a * x * exp x - (x + 1) ^ 2

-- Question 1: Extreme value when a = -1
theorem extreme_value_when_a_is_neg_one : 
  f (-1) (-1) = 1 / exp 1 := sorry

-- Question 2: Range of a such that ∀ x ∈ [-1, 1], f(x) ≤ 0
theorem range_of_a_for_f_non_positive :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f a x ≤ 0) ↔ 0 ≤ a ∧ a ≤ 4 / exp 1 := sorry

end extreme_value_when_a_is_neg_one_range_of_a_for_f_non_positive_l123_123867


namespace device_works_probability_l123_123262

theorem device_works_probability (p_comp_damaged : ℝ) (two_components : Bool) :
  p_comp_damaged = 0.1 → two_components = true → (0.9 * 0.9 = 0.81) :=
by
  intros h1 h2
  sorry

end device_works_probability_l123_123262


namespace triangle_is_obtuse_l123_123397

noncomputable def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  B = 2 * A ∧ a = 1 ∧ b = 4 / 3 ∧ (a^2 + b^2 < c^2)

theorem triangle_is_obtuse (A B C : ℝ) (a b c : ℝ) (h : triangle_ABC A B C a b c) : 
  B > π / 2 :=
by
  sorry

end triangle_is_obtuse_l123_123397


namespace find_sum_of_x_and_y_l123_123229

theorem find_sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 8 * x - 4 * y - 20) : x + y = 2 := 
by
  sorry

end find_sum_of_x_and_y_l123_123229


namespace math_books_count_l123_123929

theorem math_books_count (M H : ℤ) (h1 : M + H = 90) (h2 : 4 * M + 5 * H = 397) : M = 53 :=
by
  sorry

end math_books_count_l123_123929


namespace find_green_pepper_weight_l123_123988

variable (weight_red_peppers : ℝ) (total_weight_peppers : ℝ)

theorem find_green_pepper_weight 
    (h1 : weight_red_peppers = 0.33) 
    (h2 : total_weight_peppers = 0.66) 
    : total_weight_peppers - weight_red_peppers = 0.33 := 
by sorry

end find_green_pepper_weight_l123_123988


namespace union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_l123_123741

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set ℕ := {4, 5, 6, 7, 8, 9}
def B : Set ℕ := {1, 2, 3, 4, 5, 6}

theorem union_of_A_and_B : A ∪ B = U := by
  sorry

theorem intersection_of_A_and_B : A ∩ B = {4, 5, 6} := by
  sorry

theorem complement_of_intersection : U \ (A ∩ B) = {1, 2, 3, 7, 8, 9} := by
  sorry

end union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_l123_123741


namespace sum_of_distances_eq_l123_123154

noncomputable def sum_of_distances_from_vertex_to_midpoints (A B C M N O : ℝ × ℝ) : ℝ :=
  let AM := Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2)
  let AN := Real.sqrt ((N.1 - A.1)^2 + (N.2 - A.2)^2)
  let AO := Real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2)
  AM + AN + AO

theorem sum_of_distances_eq (A B C M N O : ℝ × ℝ) (h1 : B = (3, 0)) (h2 : C = (3/2, (3 * Real.sqrt 3/2))) (h3 : M = (3/2, 0)) (h4 : N = (9/4, (3 * Real.sqrt 3/4))) (h5 : O = (3/4, (3 * Real.sqrt 3/4))) :
  sum_of_distances_from_vertex_to_midpoints A B C M N O = 3 + (9 / 2) * Real.sqrt 3 :=
by
  sorry

end sum_of_distances_eq_l123_123154


namespace total_population_calculation_l123_123873

theorem total_population_calculation :
  ∀ (total_lions total_leopards adult_lions adult_leopards : ℕ)
  (female_lions male_lions female_leopards male_leopards : ℕ)
  (adult_elephants baby_elephants total_elephants total_zebras : ℕ),
  total_lions = 200 →
  total_lions = 2 * total_leopards →
  adult_lions = 3 * total_lions / 4 →
  adult_leopards = 3 * total_leopards / 5 →
  female_lions = 3 * total_lions / 5 →
  male_lions = 2 * total_lions / 5 →
  female_leopards = 2 * total_leopards / 3 →
  male_leopards = total_leopards / 3 →
  adult_elephants = (adult_lions + adult_leopards) / 2 →
  baby_elephants = 100 →
  total_elephants = adult_elephants + baby_elephants →
  total_zebras = adult_elephants + total_leopards →
  total_lions + total_leopards + total_elephants + total_zebras = 710 :=
by sorry

end total_population_calculation_l123_123873


namespace largest_int_value_of_m_l123_123387

variable {x y m : ℤ}

theorem largest_int_value_of_m (h1 : x + 2 * y = 2 * m + 1)
                              (h2 : 2 * x + y = m + 2)
                              (h3 : x - y > 2) : m = -2 := 
sorry

end largest_int_value_of_m_l123_123387


namespace ratio_proof_l123_123688

theorem ratio_proof (x y z s : ℝ) (h1 : x < y) (h2 : y < z)
    (h3 : (x : ℝ) / y = y / z) (h4 : x + y + z = s) (h5 : x + y = z) :
    (x / y = (-1 + Real.sqrt 5) / 2) :=
by
  sorry

end ratio_proof_l123_123688


namespace avg_consecutive_integers_l123_123511

theorem avg_consecutive_integers (a : ℝ) (b : ℝ) 
  (h₁ : b = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5)) / 6) :
  (a + 5) = (b + (b + 1) + (b + 2) + (b + 3) + (b + 4) + (b + 5)) / 6 :=
by sorry

end avg_consecutive_integers_l123_123511


namespace cos_240_eq_negative_half_l123_123325

theorem cos_240_eq_negative_half : Real.cos (240 * Real.pi / 180) = -1 / 2 := by
  sorry

end cos_240_eq_negative_half_l123_123325


namespace total_people_l123_123498

theorem total_people (M W C : ℕ) (h1 : M = 2 * W) (h2 : W = 3 * C) (h3 : C = 30) : M + W + C = 300 :=
by
  sorry

end total_people_l123_123498


namespace eccentricities_ellipse_hyperbola_l123_123111

theorem eccentricities_ellipse_hyperbola :
  let a := 2
  let b := -5
  let c := 2
  let delta := b^2 - 4 * a * c
  let x1 := (-b + Real.sqrt delta) / (2 * a)
  let x2 := (-b - Real.sqrt delta) / (2 * a)
  (x1 > 1) ∧ (0 < x2) ∧ (x2 < 1) :=
sorry

end eccentricities_ellipse_hyperbola_l123_123111


namespace gcd_a_b_eq_one_l123_123466

def a : ℕ := 123^2 + 235^2 + 347^2
def b : ℕ := 122^2 + 234^2 + 348^2

theorem gcd_a_b_eq_one : Nat.gcd a b = 1 :=
by
  sorry

end gcd_a_b_eq_one_l123_123466


namespace length_of_purple_part_l123_123197

variables (P : ℝ) (black : ℝ) (blue : ℝ) (total_len : ℝ)

-- The conditions
def conditions := 
  black = 0.5 ∧ 
  blue = 2 ∧ 
  total_len = 4 ∧ 
  P + black + blue = total_len

-- The proof problem statement
theorem length_of_purple_part (h : conditions P 0.5 2 4) : P = 1.5 :=
sorry

end length_of_purple_part_l123_123197


namespace sector_area_l123_123396

theorem sector_area (r θ : ℝ) (h₁ : θ = 2) (h₂ : r * θ = 4) : (1 / 2) * r^2 * θ = 4 :=
by
  sorry

end sector_area_l123_123396


namespace minimum_f_l123_123985

def f (x : ℝ) : ℝ := |x - 2| + |5 - x|

theorem minimum_f : ∃ x, f x = 3 :=
by
  use 3
  unfold f
  sorry

end minimum_f_l123_123985


namespace integer_solutions_count_eq_11_l123_123858

theorem integer_solutions_count_eq_11 :
  ∃ (count : ℕ), (∀ n : ℤ, (n + 2) * (n - 5) + n ≤ 10 ↔ (n ≥ -4 ∧ n ≤ 6)) ∧ count = 11 :=
by
  sorry

end integer_solutions_count_eq_11_l123_123858


namespace longest_side_of_similar_triangle_l123_123240

theorem longest_side_of_similar_triangle (a b c : ℕ) (perimeter_similar : ℕ) (h₁ : a = 8) (h₂ : b = 10) (h₃ : c = 12) (h₄ : perimeter_similar = 150) : 
  ∃ x : ℕ, 12 * x = 60 :=
by {
  have side_sum := h₁.symm ▸ h₂.symm ▸ h₃.symm ▸ (8 + 10 + 12),  -- a + b + c = 8 + 10 + 12
  rw ←h₄ at side_sum,  -- replace 30 with 150
  use 5,               -- introduction of the ratio
  sorry                 -- steps to show the length of the longest side is 60
}

end longest_side_of_similar_triangle_l123_123240


namespace max_value_f_l123_123031

open Real

noncomputable def f (x : ℝ) := x * (1 - 2 * x)

theorem max_value_f : ∃ m : ℝ, (∀ x : ℝ, 0 < x ∧ x < (1 / 2) → f x ≤ m) ∧ (∃ x : ℝ, 0 < x ∧ x < (1 / 2) ∧ f x = m) :=
by
  unfold f
  -- Detailed proof with relevant approach goes here
  sorry

end max_value_f_l123_123031


namespace count_two_digit_primess_with_digit_sum_10_eq_3_l123_123539

def is_two_digit_prime_with_digit_sum_10 (n : ℕ) : Prop :=
  (10 ≤ n ∧ n < 100) ∧ Prime n ∧ (n / 10 + n % 10 = 10)

theorem count_two_digit_primess_with_digit_sum_10_eq_3 :
  (Finset.filter is_two_digit_prime_with_digit_sum_10 (Finset.range 100)).card = 3 := sorry

end count_two_digit_primess_with_digit_sum_10_eq_3_l123_123539


namespace largest_y_l123_123666

theorem largest_y : ∃ (y : ℤ), (y ≤ 3) ∧ (∀ (z : ℤ), (z > y) → ¬ (z / 4 + 6 / 7 < 7 / 4)) :=
by
  -- There exists an integer y such that y <= 3 and for all integers z greater than y, the inequality does not hold
  sorry

end largest_y_l123_123666


namespace necessary_but_not_sufficient_l123_123828

theorem necessary_but_not_sufficient (x y : ℝ) : 
  (x < 0 ∨ y < 0) → x + y < 0 :=
sorry

end necessary_but_not_sufficient_l123_123828


namespace probability_jerry_at_four_l123_123595

theorem probability_jerry_at_four :
  let total_flips := 8
  let coordinate := 4
  let total_possible_outcomes := 2 ^ total_flips
  let favorable_outcomes := Nat.choose total_flips (total_flips / 2 + coordinate / 2)
  let P := favorable_outcomes / total_possible_outcomes
  let a := 7
  let b := 64
  ∃ (a b : ℕ), Nat.gcd a b = 1 ∧ P = a / b ∧ a + b = 71
:= sorry

end probability_jerry_at_four_l123_123595


namespace count_two_digit_primes_with_digit_sum_10_l123_123547

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter (λ n, digits_sum_to_10 n ∧ is_prime n) (Finset.filter is_two_digit (Finset.range 100))).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l123_123547


namespace b_range_l123_123560

open Real

section
variable {f : ℝ → ℝ}

noncomputable def is_odd_function_on_ℝ_with_period_4 (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 4) = f x)

noncomputable def f_definition (x b : ℝ) : ℝ := ln (x^2 - x + b)
noncomputable def num_zeros_in_interval (f : ℝ → ℝ) (a b : ℝ) : ℕ := (∑ k in Icc a b, if f k = 0 then 1 else 0)

theorem b_range (h₁ : is_odd_function_on_ℝ_with_period_4 f)
                (h₂ : ∀ x ∈ Ioo 0 2, f x = f_definition x) 
                (h₃ : num_zeros_in_interval f (-2) 2 = 5) :
  (1 / 4 : ℝ) < b ∧ b ≤ 1 ∨ b = 5 / 4 :=
sorry
end

end b_range_l123_123560


namespace prod_roots_of_unity_l123_123155

open Complex
open BigOperators

theorem prod_roots_of_unity :
  (∏ j in Finset.range 12, ∏ k in Finset.range 15, (exp (2 * π * I * j / 13) - exp (2 * π * I * k / 17))) = 1 := 
sorry

end prod_roots_of_unity_l123_123155


namespace number_of_paper_cups_is_40_l123_123277

noncomputable def cost_paper_plate : ℝ := sorry
noncomputable def cost_paper_cup : ℝ := sorry
noncomputable def num_paper_cups_in_second_purchase : ℝ := sorry

-- Conditions
axiom first_condition : 100 * cost_paper_plate + 200 * cost_paper_cup = 7.50
axiom second_condition : 20 * cost_paper_plate + num_paper_cups_in_second_purchase * cost_paper_cup = 1.50

-- Goal
theorem number_of_paper_cups_is_40 : num_paper_cups_in_second_purchase = 40 := 
by 
  sorry

end number_of_paper_cups_is_40_l123_123277


namespace cos_240_eq_neg_half_l123_123307

theorem cos_240_eq_neg_half : Real.cos (4 * Real.pi / 3) = -1/2 :=
by
  sorry

end cos_240_eq_neg_half_l123_123307


namespace segment_order_l123_123948

def angle_sum_triangle (A B C : ℝ) : Prop := A + B + C = 180

def order_segments (angles_ABC angles_XYZ angles_ZWX : ℝ → ℝ → ℝ) : Prop :=
  let A := angles_ABC 55 60
  let B := angles_XYZ 95 70
  ∀ (XY YZ ZX WX WZ: ℝ), 
    YZ < ZX ∧ ZX < XY ∧ ZX < WZ ∧ WZ < WX

theorem segment_order:
  ∀ (A B C X Y Z W : Type)
  (XYZ_ang ZWX_ang : ℝ), 
  angle_sum_triangle 55 60 65 →
  angle_sum_triangle 95 70 15 →
  order_segments (angles_ABC) (angles_XYZ) (angles_ZWX)
:= sorry

end segment_order_l123_123948


namespace cone_slant_height_l123_123797

noncomputable def slant_height (r : ℝ) (CSA : ℝ) : ℝ := CSA / (Real.pi * r)

theorem cone_slant_height : slant_height 10 628.3185307179587 = 20 :=
by
  sorry

end cone_slant_height_l123_123797


namespace Sum_a2_a3_a7_l123_123770

-- Definitions from the conditions
variable {a : ℕ → ℝ} -- Define the arithmetic sequence as a function from natural numbers to real numbers
variable {S : ℕ → ℝ} -- Define the sum of the first n terms as a function from natural numbers to real numbers

-- Given conditions
axiom Sn_formula : ∀ n : ℕ, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))
axiom S7_eq_42 : S 7 = 42

theorem Sum_a2_a3_a7 :
  a 2 + a 3 + a 7 = 18 :=
sorry

end Sum_a2_a3_a7_l123_123770


namespace power_function_value_at_4_l123_123986

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := x^a

theorem power_function_value_at_4 :
  ∃ a : ℝ, power_function a 2 = (Real.sqrt 2) / 2 → power_function a 4 = 1 / 2 :=
by
  sorry

end power_function_value_at_4_l123_123986


namespace least_possible_value_of_smallest_integer_l123_123678

theorem least_possible_value_of_smallest_integer :
  ∀ (A B C D : ℕ), A < B → B < C → C < D → (A + B + C + D) / 4 = 70 → D = 90 → A ≥ 13 :=
by
  intros A B C D h₁ h₂ h₃ h₄ h₅
  sorry

end least_possible_value_of_smallest_integer_l123_123678


namespace fraction_eaten_correct_l123_123644

def initial_nuts : Nat := 30
def nuts_left : Nat := 5
def eaten_nuts : Nat := initial_nuts - nuts_left
def fraction_eaten : Rat := eaten_nuts / initial_nuts

theorem fraction_eaten_correct : fraction_eaten = 5 / 6 := by
  sorry

end fraction_eaten_correct_l123_123644


namespace sum_of_reciprocals_l123_123220

noncomputable def roots (p q r : ℂ) : Prop := 
  p ^ 3 - p + 1 = 0 ∧ q ^ 3 - q + 1 = 0 ∧ r ^ 3 - r + 1 = 0

theorem sum_of_reciprocals (p q r : ℂ) (h : roots p q r) : 
  (1 / (p + 2)) + (1 / (q + 2)) + (1 / (r + 2)) = - (10 / 13) := by 
  sorry

end sum_of_reciprocals_l123_123220


namespace polynomial_remainder_l123_123969

theorem polynomial_remainder (x : ℤ) :
  let poly := x^5 + 3*x^3 + 1
  let divisor := (x + 1)^2
  let remainder := 5*x + 9
  ∃ q : ℤ, poly = divisor * q + remainder := by
  sorry

end polynomial_remainder_l123_123969


namespace total_tosses_correct_l123_123402

def num_heads : Nat := 3
def num_tails : Nat := 7
def total_tosses : Nat := num_heads + num_tails

theorem total_tosses_correct : total_tosses = 10 := by
  sorry

end total_tosses_correct_l123_123402


namespace cos_240_eq_neg_half_l123_123323

theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Sorry to skip the proof
  sorry

end cos_240_eq_neg_half_l123_123323


namespace cos_240_is_neg_half_l123_123347

noncomputable def cos_240 : ℝ := Real.cos (240 * Real.pi / 180)

theorem cos_240_is_neg_half : cos_240 = -1/2 := by
  sorry

end cos_240_is_neg_half_l123_123347


namespace ab_bc_ca_negative_l123_123434

theorem ab_bc_ca_negative (a b c : ℝ) (h₁ : a + b + c = 0) (h₂ : abc > 0) : ab + bc + ca < 0 :=
sorry

end ab_bc_ca_negative_l123_123434


namespace arithmetic_sequence_problem_l123_123058

noncomputable def arithmetic_sequence_sum : ℕ → ℕ := sorry  -- Define S_n here

theorem arithmetic_sequence_problem (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : S 8 - S 3 = 10)
    (h2 : ∀ n, S (n + 1) = S n + a (n + 1)) (h3 : a 6 = 2) : S 11 = 22 :=
  sorry

end arithmetic_sequence_problem_l123_123058


namespace roots_purely_imaginary_l123_123709

open Complex

/-- 
  If m is a purely imaginary number, then the roots of the equation 
  8z^2 + 4i * z - m = 0 are purely imaginary.
-/
theorem roots_purely_imaginary (m : ℂ) (hm : m.im ≠ 0 ∧ m.re = 0) : 
  ∀ z : ℂ, 8 * z^2 + 4 * Complex.I * z - m = 0 → z.im ≠ 0 ∧ z.re = 0 :=
by
  sorry

end roots_purely_imaginary_l123_123709


namespace profit_distribution_l123_123092

theorem profit_distribution (investment_LiWei investment_WangGang profit total_investment : ℝ)
  (h1 : investment_LiWei = 16000)
  (h2 : investment_WangGang = 12000)
  (h3 : profit = 14000)
  (h4 : total_investment = investment_LiWei + investment_WangGang) :
  (profit * (investment_LiWei / total_investment) = 8000) ∧ 
  (profit * (investment_WangGang / total_investment) = 6000) :=
by
  sorry

end profit_distribution_l123_123092


namespace votes_cast_is_330_l123_123226

variable (T A F : ℝ)

theorem votes_cast_is_330
  (h1 : A = 0.40 * T)
  (h2 : F = A + 66)
  (h3 : T = F + A) :
  T = 330 :=
by
  sorry

end votes_cast_is_330_l123_123226


namespace probability_distinct_numbers_l123_123245

theorem probability_distinct_numbers 
  (total_balls : Finset (Fin 10))
  (red_balls : Finset (Fin 5))
  (black_balls : Finset (Fin 5))
  (numbered_balls : ∀ i, (i ∈ red_balls ∪ black_balls) → (∃ n : Fin 5, true))
  (h1 : total_balls.card = 10)
  (h2 : red_balls.card = 5)
  (h3 : black_balls.card = 5)
  (h4 : red_balls ∩ black_balls = ∅) :
  (∃ (p : ℚ), p = 8/21) :=
by
  sorry

end probability_distinct_numbers_l123_123245


namespace converse_and_inverse_l123_123021

-- Definitions
def is_circle (s : Type) : Prop := sorry
def has_no_corners (s : Type) : Prop := sorry

-- Converse Statement
def converse_false (s : Type) : Prop :=
  has_no_corners s → is_circle s → False

-- Inverse Statement
def inverse_true (s : Type) : Prop :=
  ¬ is_circle s → ¬ has_no_corners s

-- Main Proof Problem
theorem converse_and_inverse (s : Type) :
  (converse_false s) ∧ (inverse_true s) := sorry

end converse_and_inverse_l123_123021


namespace total_money_spent_l123_123160

variables (emma_spent : ℕ) (elsa_spent : ℕ) (elizabeth_spent : ℕ)
variables (total_spent : ℕ)

-- Conditions
def EmmaSpending : Prop := emma_spent = 58
def ElsaSpending : Prop := elsa_spent = 2 * emma_spent
def ElizabethSpending : Prop := elizabeth_spent = 4 * elsa_spent
def TotalSpending : Prop := total_spent = emma_spent + elsa_spent + elizabeth_spent

-- The theorem to prove
theorem total_money_spent 
  (h1 : EmmaSpending) 
  (h2 : ElsaSpending) 
  (h3 : ElizabethSpending) 
  (h4 : TotalSpending) : 
  total_spent = 638 := 
sorry

end total_money_spent_l123_123160


namespace length_of_PT_l123_123489

/--
Let circles O and P have radii 10 units and 3 units respectively, and they are externally tangent to each other at point Q. Segment TS is the common external tangent at points T and S, respectively. Prove that the length of PT is √69.
-/
theorem length_of_PT :
  ∀ (O P T S Q : Type) (rO rP : ℝ),  rO = 10 ∧ rP = 3 ∧
  ∃ (centerO centerP : O), ∃ (pointQ : Q), 
  (dist centerO centerP = 13) ∧ (dist pointQ centerP = 3) ∧  
  (dist centerO pointQ = 10) ∧
  -- TS is a common external tangent
  ∃ (pointT pointS: T), dist pointT pointT = 0 ∧ dist pointS pointS = 0 →
  dist pointT pointS = sqrt 69 := 
sorry

end length_of_PT_l123_123489


namespace instantaneous_velocity_at_3_l123_123632

def s (t : ℝ) : ℝ := 1 - t + t^2

def velocity_at_t (t : ℝ) : ℝ := (deriv s) t

theorem instantaneous_velocity_at_3 : velocity_at_t 3 = 5 := by
  sorry

end instantaneous_velocity_at_3_l123_123632


namespace intersect_P_Q_l123_123189

open Set

def P : Set ℤ := { x | (x - 3) * (x - 6) ≤ 0 }
def Q : Set ℤ := { 5, 7 }

theorem intersect_P_Q : P ∩ Q = {5} :=
sorry

end intersect_P_Q_l123_123189


namespace geo_seq_product_l123_123177

theorem geo_seq_product (a : ℕ → ℝ) (r : ℝ) (h_pos : ∀ n, 0 < a n) 
  (h_geom : ∀ n, a (n + 1) = a n * r) (h_a1a9 : a 1 * a 9 = 16) :
  a 2 * a 5 * a 8 = 64 :=
sorry

end geo_seq_product_l123_123177


namespace cos_240_eq_neg_half_l123_123304

theorem cos_240_eq_neg_half : Real.cos (4 * Real.pi / 3) = -1/2 :=
by
  sorry

end cos_240_eq_neg_half_l123_123304


namespace cos_240_eq_neg_half_l123_123351

open Real

theorem cos_240_eq_neg_half : cos (240 * π / 180) = -1/2 :=
by
  -- Step 1: Decompose the angle 240° = 180° + 60°
  have h1 : 240 * π / 180 = π + 60 * π / 180,
  { 
    norm_num, 
    field_simp, 
    linarith 
  },
  -- Step 2: Use the fact that the cosine of (π + θ) = - cos(θ)
  rw [h1, cos_add_pi],
  -- Step 3: Given that cos(60°) = 1/2
  have h2 : cos (60 * π / 180) = 1/2,
  {
    norm_num,
    exact Real.cos_pi_div_three (),
  },
  -- Conclude that cos(240°) = -1/2
  rw h2,
  norm_num

end cos_240_eq_neg_half_l123_123351


namespace cos_240_eq_neg_half_l123_123314

theorem cos_240_eq_neg_half : ∀ (deg: ℝ), 
  deg = 240 → 
  (∀ a b : ℝ, 240 = a + b → a = 180 → b = 60 → 
    cos 240 = cos a * cos b - sin a * sin b) → 
  cos 180 = -1 →
  sin 180 = 0 →
  cos 60 = 1 / 2 →
  cos 240 = -1 / 2 :=
by 
  intros deg h_deg h_sum h_cos_180 h_sin_180 h_cos_60
  rw h_deg at h_sum
  have h_cos_identity := h_sum 180 60 rfl rfl rfl
  simp [h_cos_180, h_sin_180, h_cos_60] at h_cos_identity
  exact h_cos_identity

end cos_240_eq_neg_half_l123_123314


namespace solve_f_neg_a_l123_123186

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem solve_f_neg_a (h : f a = 8) : f (-a) = -6 := by
  sorry

end solve_f_neg_a_l123_123186


namespace a_5_eq_14_l123_123849

def S (n : ℕ) : ℚ := (3 / 2) * n ^ 2 + (1 / 2) * n

def a (n : ℕ) : ℚ := S n - S (n - 1)

theorem a_5_eq_14 : a 5 = 14 := by {
  -- Proof steps go here
  sorry
}

end a_5_eq_14_l123_123849


namespace fraction_subtraction_simplified_l123_123971

theorem fraction_subtraction_simplified : (8 / 19 - 5 / 57) = (1 / 3) := by
  sorry

end fraction_subtraction_simplified_l123_123971


namespace find_c_l123_123576

theorem find_c (b c : ℝ) (h : (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c * x + 8)) : c = 17 / 3 := 
by
  -- Add the necessary assumptions and let Lean verify these assumptions.
  have b_eq : 3 * b = 8 := sorry
  have b_val : b = 8 / 3 := sorry
  have h_coeff : c = b + 3 := sorry
  exact h_coeff.trans (by rw [b_val]; norm_num)

end find_c_l123_123576


namespace tangent_lines_create_regions_l123_123643

theorem tangent_lines_create_regions (n : ℕ) (h : n = 26) : ∃ k, k = 68 :=
by
  have h1 : ∃ k, k = 68 := ⟨68, rfl⟩
  exact h1

end tangent_lines_create_regions_l123_123643


namespace cos_240_eq_neg_half_l123_123319

theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Sorry to skip the proof
  sorry

end cos_240_eq_neg_half_l123_123319


namespace problem_solution_l123_123166

theorem problem_solution (x : ℝ) : (∃ (x : ℝ), 5 < x ∧ x ≤ 6) ↔ (∃ (x : ℝ), (x - 3) / (x - 5) ≥ 3) :=
sorry

end problem_solution_l123_123166


namespace part1_part2_l123_123599

variable {f : ℝ → ℝ}

-- Condition 1: f is an odd function
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Condition 2: ∀ a b ∈ ℝ, (a + b ≠ 0) → (f(a) + f(b))/(a + b) > 0
def positiveQuotient (f : ℝ → ℝ) : Prop :=
  ∀ a b, a + b ≠ 0 → (f a + f b) / (a + b) > 0

-- Sub-problem (1): For any a, b ∈ ℝ, a > b ⟹ f(a) > f(b)
theorem part1 (h_odd : isOddFunction f) (h_posQuot : positiveQuotient f) (a b : ℝ) (h : a > b) : f a > f b :=
  sorry

-- Sub-problem (2): If f(9^x - 2 * 3^x) + f(2 * 9^x - k) > 0 for any x ∈ [0, ∞), then k < 1
theorem part2 (h_odd : isOddFunction f) (h_posQuot : positiveQuotient f) :
  (∀ x : ℝ, 0 ≤ x → f (9^x - 2 * 3^x) + f (2 * 9^x - k) > 0) → k < 1 :=
  sorry

end part1_part2_l123_123599


namespace spinner_probability_C_l123_123260

theorem spinner_probability_C 
  (P_A : ℚ) (P_B : ℚ) (P_C : ℚ) (P_D : ℚ)
  (hA : P_A = 1/3)
  (hB : P_B = 1/4)
  (hD : P_D = 1/6)
  (hSum : P_A + P_B + P_C + P_D = 1) :
  P_C = 1 / 4 := 
sorry

end spinner_probability_C_l123_123260


namespace cos_240_eq_neg_half_l123_123317

theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Sorry to skip the proof
  sorry

end cos_240_eq_neg_half_l123_123317


namespace integer_solutions_count_l123_123989

theorem integer_solutions_count :
  let cond1 (x : ℤ) := -4 * x ≥ 2 * x + 9
  let cond2 (x : ℤ) := -3 * x ≤ 15
  let cond3 (x : ℤ) := -5 * x ≥ x + 22
  ∃ s : Finset ℤ, 
    (∀ x ∈ s, cond1 x ∧ cond2 x ∧ cond3 x) ∧
    (∀ x, cond1 x ∧ cond2 x ∧ cond3 x → x ∈ s) ∧
    s.card = 2 :=
sorry

end integer_solutions_count_l123_123989


namespace x_eq_one_is_sufficient_but_not_necessary_for_x_squared_plus_x_minus_two_eq_zero_l123_123933

theorem x_eq_one_is_sufficient_but_not_necessary_for_x_squared_plus_x_minus_two_eq_zero :
  ∃ (x : ℝ), (x = 1) → (x^2 + x - 2 = 0) ∧ (¬ (∀ (y : ℝ), y^2 + y - 2 = 0 → y = 1)) := by
  sorry

end x_eq_one_is_sufficient_but_not_necessary_for_x_squared_plus_x_minus_two_eq_zero_l123_123933


namespace part_a_avg_area_difference_part_b_prob_same_area_part_c_expected_value_difference_l123_123922

-- Part (a)
theorem part_a_avg_area_difference : 
  let zahid_avg := (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) / 6
  let yana_avg := (21 / 6)^2
  zahid_avg - yana_avg = 35 / 12 := sorry

-- Part (b)
theorem part_b_prob_same_area :
  let prob_zahid_min n := (13 - 2 * n) / 36
  let prob_same_area := (1 / 36) * ((11 / 36) + (9 / 36) + (7 / 36) + (5 / 36) + (3 / 36) + (1 / 36))
  prob_same_area = 1 / 24 := sorry

-- Part (c)
theorem part_c_expected_value_difference :
  let yana_avg := 49 / 4
  let zahid_avg := (11 / 36 * 1^2 + 9 / 36 * 2^2 + 7 / 36 * 3^2 + 5 / 36 * 4^2 + 3 / 36 * 5^2 + 1 / 36 * 6^2)
  (yana_avg - zahid_avg) = 35 / 9 := sorry

end part_a_avg_area_difference_part_b_prob_same_area_part_c_expected_value_difference_l123_123922


namespace div_simplify_l123_123281

theorem div_simplify (a b : ℝ) (h : a ≠ 0) : (8 * a * b) / (2 * a) = 4 * b :=
by
  sorry

end div_simplify_l123_123281


namespace cos_240_eq_neg_half_l123_123353

open Real

theorem cos_240_eq_neg_half : cos (240 * π / 180) = -1/2 :=
by
  -- Step 1: Decompose the angle 240° = 180° + 60°
  have h1 : 240 * π / 180 = π + 60 * π / 180,
  { 
    norm_num, 
    field_simp, 
    linarith 
  },
  -- Step 2: Use the fact that the cosine of (π + θ) = - cos(θ)
  rw [h1, cos_add_pi],
  -- Step 3: Given that cos(60°) = 1/2
  have h2 : cos (60 * π / 180) = 1/2,
  {
    norm_num,
    exact Real.cos_pi_div_three (),
  },
  -- Conclude that cos(240°) = -1/2
  rw h2,
  norm_num

end cos_240_eq_neg_half_l123_123353


namespace total_surface_area_l123_123099

-- Define the context and given conditions
variable (r : ℝ) (π : ℝ)
variable (base_area : ℝ) (curved_surface_area : ℝ)

-- Assume that the area of the base (circle) of the hemisphere is given as 225π
def base_of_hemisphere_area (r : ℝ) (π : ℝ) :=
  π * r^2 = 225 * π

-- Derive the radius from the base area
def radius_from_base_area (r : ℝ) :=
  r = Real.sqrt (225)

-- Define the curved surface area of the hemisphere
def curved_surface_area_hemisphere (r : ℝ) (π : ℝ) :=
  curved_surface_area = (1 / 2) * (4 * π * r^2)

-- Provide the final calculation for the total surface area
def total_surface_area_hemisphere (curved_surface_area : ℝ) (base_area : ℝ) :=
  curved_surface_area + base_area = 675 * π

-- Main theorem that combines everything and matches the problem statement
theorem total_surface_area (r : ℝ) (π : ℝ) (base_area : ℝ) (curved_surface_area : ℝ) :
  base_of_hemisphere_area r π →
  radius_from_base_area r →
  curved_surface_area_hemisphere r π →
  total_surface_area_hemisphere curved_surface_area base_area :=
by
  intros h_base_radius h_radius h_curved_area
  sorry

end total_surface_area_l123_123099


namespace minimum_area_of_quadrilateral_l123_123073

theorem minimum_area_of_quadrilateral
  (ABCD : Type)
  (O : Type)
  (S_ABO : ℝ)
  (S_CDO : ℝ)
  (BC : ℝ)
  (cos_angle_ADC : ℝ)
  (h1 : S_ABO = 3 / 2)
  (h2 : S_CDO = 3 / 2)
  (h3 : BC = 3 * Real.sqrt 2)
  (h4 : cos_angle_ADC = 3 / Real.sqrt 10) :
  ∃ S_ABCD : ℝ, S_ABCD = 6 :=
sorry

end minimum_area_of_quadrilateral_l123_123073


namespace man_age_difference_l123_123944

theorem man_age_difference (S M : ℕ) (h1 : S = 22) (h2 : M + 2 = 2 * (S + 2)) :
  M - S = 24 :=
by sorry

end man_age_difference_l123_123944


namespace pythagorean_prime_divisibility_l123_123778

theorem pythagorean_prime_divisibility 
  (x y z : ℤ) (hx : prime x ∨ prime y ∨ prime z) 
  (hp : x > 5 ∧ y > 5 ∧ z > 5) 
  (h : x^2 + y^2 = z^2) : 
  (x ∣ 60) ∨ (y ∣ 60) ∨ (z ∣ 60) :=
by
  sorry

end pythagorean_prime_divisibility_l123_123778


namespace year_2023_ad_is_written_as_positive_2023_l123_123961

theorem year_2023_ad_is_written_as_positive_2023 :
  (∀ (year : Int), year = -500 → year = -500) → -- This represents the given condition that year 500 BC is -500
  (∀ (year : Int), year > 0) → -- This represents the condition that AD years are postive
  2023 = 2023 := -- The problem conclusion

by
  intros
  trivial -- The solution is quite trivial due to the conditions.

end year_2023_ad_is_written_as_positive_2023_l123_123961


namespace count_two_digit_primes_with_digit_sum_10_is_4_l123_123532

noncomputable def countTwoDigitPrimesWithDigitSum10 : ℕ :=
  let digitPairs := [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]
  let numbers := digitPairs.map (λ p => p.1 * 10 + p.2)
  numbers.count (λ n => Nat.Prime n)

theorem count_two_digit_primes_with_digit_sum_10_is_4 : countTwoDigitPrimesWithDigitSum10 = 4 :=
  sorry

end count_two_digit_primes_with_digit_sum_10_is_4_l123_123532


namespace isosceles_triangle_if_perpendiculars_intersect_at_single_point_l123_123431

noncomputable def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

theorem isosceles_triangle_if_perpendiculars_intersect_at_single_point
  (a b c : ℝ)
  (D E F P Q R H : Type)
  (intersection_point: P = Q ∧ Q = R ∧ P = R ∧ P = H) :
  is_isosceles_triangle a b c := 
sorry

end isosceles_triangle_if_perpendiculars_intersect_at_single_point_l123_123431


namespace sum_of_first_10_terms_l123_123180

noncomputable def sum_first_n_terms (a_1 d : ℕ) (n : ℕ) : ℕ :=
  (n / 2) * (2 * a_1 + (n - 1) * d)

theorem sum_of_first_10_terms (a : ℕ → ℕ) (a_2_a_4_sum : a 2 + a 4 = 4) (a_3_a_5_sum : a 3 + a 5 = 10) :
  sum_first_n_terms (a 1) (a 2 - a 1) 10 = 95 :=
  sorry

end sum_of_first_10_terms_l123_123180


namespace count_two_digit_primes_with_digit_sum_10_l123_123556

def sum_of_digits (n : ℕ) : ℕ := (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def is_prime (n : ℕ) : Prop := Nat.Prime n

def meets_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 10

theorem count_two_digit_primes_with_digit_sum_10 :
  (Finset.filter meets_conditions (Finset.range 100)).card = 3 :=
by
  sorry

end count_two_digit_primes_with_digit_sum_10_l123_123556


namespace intersection_A_B_l123_123735

-- Definitions of the sets A and B
def set_A : Set ℝ := { x | 3 ≤ x ∧ x ≤ 10 }
def set_B : Set ℝ := { x | 2 < x ∧ x < 7 }

-- Theorem statement to prove the intersection
theorem intersection_A_B : set_A ∩ set_B = { x | 3 ≤ x ∧ x < 7 } := by
  sorry

end intersection_A_B_l123_123735


namespace parallel_vectors_m_value_l123_123854

theorem parallel_vectors_m_value :
  ∀ (m : ℝ), (∀ k : ℝ, (1 : ℝ) = k * m ∧ (-2) = k * (-1)) -> m = (1 / 2) :=
by
  intros m h
  sorry

end parallel_vectors_m_value_l123_123854


namespace number_of_participants_with_5_points_l123_123070

-- Definitions for conditions
def num_participants : ℕ := 254

def points_for_victory : ℕ := 1

def additional_point_condition (winner_points loser_points : ℕ) : ℕ :=
  if winner_points < loser_points then 1 else 0

def points_for_loss : ℕ := 0

-- Theorem statement
theorem number_of_participants_with_5_points :
  ∃ num_students_with_5_points : ℕ, num_students_with_5_points = 56 := 
sorry

end number_of_participants_with_5_points_l123_123070


namespace proof_time_lent_to_C_l123_123264

theorem proof_time_lent_to_C :
  let P_B := 5000
  let R := 0.1
  let T_B := 2
  let Total_Interest := 2200
  let P_C := 3000
  let I_B := P_B * R * T_B
  let I_C := Total_Interest - I_B
  let T_C := I_C / (P_C * R)
  T_C = 4 :=
by
  sorry

end proof_time_lent_to_C_l123_123264


namespace total_ages_l123_123126

theorem total_ages (Xavier Yasmin : ℕ) (h1 : Xavier = 2 * Yasmin) (h2 : Xavier + 6 = 30) : Xavier + Yasmin = 36 :=
by
  sorry

end total_ages_l123_123126


namespace value_of_a_if_perpendicular_l123_123064

noncomputable def curve (x : ℝ) : ℝ := 2 * Real.sin x - 2 * Real.cos x

def point_of_tangency := (Real.pi / 2, 2)

def line (a : ℝ) (x y : ℝ) := x - a * y + 1 = 0

def derivative_at_point : ℝ := 2 * Real.cos (Real.pi / 2) + 2 * Real.sin (Real.pi / 2)

theorem value_of_a_if_perpendicular :
  ∀ (a : ℝ),
    derivative_at_point = -a → 
    a = -2 :=
by
  sorry

end value_of_a_if_perpendicular_l123_123064


namespace total_pencils_l123_123706

variable (C Y M D : ℕ)

-- Conditions
def cheryl_has_thrice_as_cyrus (h1 : C = 3 * Y) : Prop := true
def madeline_has_half_of_cheryl (h2 : M = 63 ∧ C = 2 * M) : Prop := true
def daniel_has_25_percent_of_total (h3 : D = (C + Y + M) / 4) : Prop := true

-- Total number of pencils for all four
theorem total_pencils (h1 : C = 3 * Y) (h2 : M = 63 ∧ C = 2 * M) (h3 : D = (C + Y + M) / 4) :
  C + Y + M + D = 289 :=
by { sorry }

end total_pencils_l123_123706


namespace some_number_value_l123_123192

theorem some_number_value (a : ℕ) (x : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * x * 49) : x = 9 := by
  sorry

end some_number_value_l123_123192


namespace contrapositive_of_sum_of_squares_l123_123783

theorem contrapositive_of_sum_of_squares
  (a b : ℝ)
  (h : a ≠ 0 ∨ b ≠ 0) :
  a^2 + b^2 ≠ 0 := 
sorry

end contrapositive_of_sum_of_squares_l123_123783


namespace how_many_pens_l123_123781

theorem how_many_pens
  (total_cost : ℝ)
  (num_pencils : ℕ)
  (avg_pencil_price : ℝ)
  (avg_pen_price : ℝ)
  (total_cost := 510)
  (num_pencils := 75)
  (avg_pencil_price := 2)
  (avg_pen_price := 12)
  : ∃ (num_pens : ℕ), num_pens = 30 :=
by
  sorry

end how_many_pens_l123_123781


namespace train_crossing_time_l123_123676

-- Definitions from conditions
def length_of_train : ℕ := 120
def length_of_bridge : ℕ := 150
def speed_kmph : ℕ := 36
def speed_mps : ℕ := speed_kmph * 1000 / 3600 -- Convert km/h to m/s
def total_distance : ℕ := length_of_train + length_of_bridge

-- Theorem statement
theorem train_crossing_time : total_distance / speed_mps = 27 := by
  sorry

end train_crossing_time_l123_123676


namespace part_I_section_I_part_I_section_II_part_II_section_I_part_II_section_II_l123_123131

-- Definition for problem I conditions and parts
def polynomial_expansion_I (x : ℝ) : ℝ := (2 * x - 1) ^ 10
noncomputable def coefficients (a : ℕ → ℝ) : Prop :=
  polynomial_expansion_I = (a 0) + (a 1) * (x - 1) + (a 2) * (x - 1) ^ 2 + 
                           (a 3) * (x - 1) ^ 3 + (a 4) * (x - 1) ^ 4 +
                           (a 5) * (x - 1) ^ 5 + (a 6) * (x - 1) ^ 6 +
                           (a 7) * (x - 1) ^ 7 + (a 8) * (x - 1) ^ 8 +
                           (a 9) * (x - 1) ^ 9 + (a 10) * (x - 1) ^ 10

theorem part_I_section_I (a : ℕ → ℝ) 
    (h : coefficients (λ i, a i)) : 
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 59049 := sorry

theorem part_I_section_II (a : ℕ → ℝ) 
    (h : coefficients (λ i, a i)) : 
  a 7 = 15360 := sorry

-- Definition for problem II conditions and parts
noncomputable def allocation_schemes_volunteers : ℕ := 
    fintype.card (finset.powerset_len 2 (finset.range 5)) * 
    finset.card (finset.permutations_of_multiset (finset.range 3 : multiset (fin 4)))

theorem part_II_section_I : allocation_schemes_volunteers = 240 := sorry

noncomputable def allocation_schemes_remaining_three_volunteers : ℕ := 
    (fintype.card (finset.powerset_len 2 (finset.range 4))) ^ 3 - 
    (fintype.card (finset.powerset_len 2 (finset.range 4))) - 
    (finset.card (finset.powerset_len 3 (finset.multiset_powerset (finset.range 4)))) * 
          ((fintype.card (finset.powerset_len 2 (finset.range 3))) ^ 3 - 
           (fintype.card (finset.powerset_len 2 (finset.range 3))))

theorem part_II_section_II : allocation_schemes_remaining_three_volunteers = 114 := sorry

end part_I_section_I_part_I_section_II_part_II_section_I_part_II_section_II_l123_123131


namespace count_two_digit_primes_with_digit_sum_10_l123_123546

theorem count_two_digit_primes_with_digit_sum_10 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ (n.digits 10).sum = 10 ∧ n.prime}.toFinset.card = 3 :=
by 
  sorry

end count_two_digit_primes_with_digit_sum_10_l123_123546


namespace height_of_picture_frame_l123_123825

-- Define the given conditions
def width : ℕ := 6
def perimeter : ℕ := 30
def perimeter_formula (w h : ℕ) : ℕ := 2 * (w + h)

-- Prove that the height of the picture frame is 9 inches
theorem height_of_picture_frame : ∃ height : ℕ, height = 9 ∧ perimeter_formula width height = perimeter :=
by
  -- Proof goes here
  sorry

end height_of_picture_frame_l123_123825


namespace compare_magnitudes_l123_123598

noncomputable def log_base_3_of_2 : ℝ := Real.log 2 / Real.log 3   -- def a
noncomputable def ln_2 : ℝ := Real.log 2                          -- def b
noncomputable def five_minus_pi : ℝ := 5 - Real.pi                -- def c

theorem compare_magnitudes :
  let a := log_base_3_of_2
  let b := ln_2
  let c := five_minus_pi
  c < a ∧ a < b :=
by
  sorry

end compare_magnitudes_l123_123598


namespace find_a_from_polynomial_factor_l123_123221

theorem find_a_from_polynomial_factor (a b : ℤ)
  (h: ∀ x : ℝ, x*x - x - 1 = 0 → a*x^5 + b*x^4 + 1 = 0) : a = 3 :=
sorry

end find_a_from_polynomial_factor_l123_123221


namespace solution_set_x2_f_x_positive_l123_123984

noncomputable def f : ℝ → ℝ := sorry
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_at_2 : f 2 = 0
axiom derivative_condition : ∀ x, x > 0 → ((x * (deriv f x) - f x) / x^2) > 0

theorem solution_set_x2_f_x_positive :
  {x : ℝ | x^2 * f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | x > 2} :=
sorry

end solution_set_x2_f_x_positive_l123_123984


namespace cos_240_eq_neg_half_l123_123322

theorem cos_240_eq_neg_half : Real.cos (240 * Real.pi / 180) = -1 / 2 :=
by
  -- Sorry to skip the proof
  sorry

end cos_240_eq_neg_half_l123_123322


namespace find_smallest_denominator_difference_l123_123088

theorem find_smallest_denominator_difference :
  ∃ (r s : ℕ), 
    r > 0 ∧ s > 0 ∧ 
    (5 : ℚ) / 11 < r / s ∧ r / s < (4 : ℚ) / 9 ∧ 
    ¬ ∃ t : ℕ, t < s ∧ (5 : ℚ) / 11 < r / t ∧ r / t < (4 : ℚ) / 9 ∧ 
    s - r = 11 := 
sorry

end find_smallest_denominator_difference_l123_123088


namespace students_with_B_l123_123587

theorem students_with_B (students_jacob : ℕ) (students_B_jacob : ℕ) (students_smith : ℕ) (ratio_same : (students_B_jacob / students_jacob : ℚ) = 2 / 5) : 
  ∃ y : ℕ, (y / students_smith : ℚ) = 2 / 5 ∧ y = 12 :=
by 
  use 12
  sorry

end students_with_B_l123_123587


namespace q_value_l123_123571

theorem q_value (p q : ℝ) (hpq1 : 1 < p) (hpql : p < q) (hq_condition : (1 / p) + (1 / q) = 1) (hpq2 : p * q = 8) : q = 4 + 2 * Real.sqrt 2 :=
  sorry

end q_value_l123_123571


namespace sixteenth_term_l123_123605

theorem sixteenth_term :
  (-1)^(16+1) * Real.sqrt (3 * (16 - 1)) = -3 * Real.sqrt 5 :=
by sorry

end sixteenth_term_l123_123605


namespace total_games_is_24_l123_123225

-- Definitions of conditions
def games_this_month : Nat := 9
def games_last_month : Nat := 8
def games_next_month : Nat := 7

-- Total games attended
def total_games_attended : Nat :=
  games_this_month + games_last_month + games_next_month

-- Problem statement
theorem total_games_is_24 : total_games_attended = 24 := by
  sorry

end total_games_is_24_l123_123225


namespace monotonic_increasing_interval_l123_123104

open Real

noncomputable def f (x : ℝ) : ℝ :=
  2 * sin x * cos x - 2 * cos x ^ 2 + 1

theorem monotonic_increasing_interval :
  ∀ k : ℤ, 
    (∀ x ∈ Ioo (k * π - π / 8) (k * π + 3 * π / 8), 
     f' x > 0) :=
by
  sorry

end monotonic_increasing_interval_l123_123104


namespace find_initial_amount_l123_123007

-- In this statement, we define the conditions and the goal based on the problem formulated above.
theorem find_initial_amount
  (P R : ℝ) -- P: Initial principal amount, R: Rate of interest in percentage
  (h1 : 956 = P * (1 + 3 * R / 100))
  (h2 : 1061 = P * (1 + 3 * (R + 4) / 100)) :
  P = 875 :=
by sorry

end find_initial_amount_l123_123007


namespace half_abs_diff_squares_l123_123656

theorem half_abs_diff_squares : (1 / 2) * |20^2 - 15^2| = 87.5 :=
by
  sorry

end half_abs_diff_squares_l123_123656


namespace tanks_fill_l123_123627

theorem tanks_fill
  (c : ℕ) -- capacity of each tank
  (h1 : 300 < c) -- first tank is filled with 300 liters, thus c > 300
  (h2 : 450 < c) -- second tank is filled with 450 liters, thus c > 450
  (h3 : (45 : ℝ) / 100 = (450 : ℝ) / c) -- second tank is 45% filled, thus 0.45 * c = 450
  (h4 : 300 + 450 < 2 * c) -- the two tanks have the same capacity, thus they must have enough capacity to be filled more than 750 liters
  : c - 300 + (c - 450) = 1250 :=
sorry

end tanks_fill_l123_123627


namespace inequality_proof_l123_123776

theorem inequality_proof (n : ℕ) (hn : n > 0) : (2 * n + 1) ^ n ≥ (2 * n) ^ n + (2 * n - 1) ^ n :=
by
  sorry

end inequality_proof_l123_123776


namespace initial_percentage_alcohol_l123_123821

variables (P : ℝ) (initial_volume : ℝ) (added_volume : ℝ) (total_volume : ℝ) (final_percentage : ℝ) (init_percentage : ℝ)

theorem initial_percentage_alcohol (h1 : initial_volume = 6)
                                  (h2 : added_volume = 3)
                                  (h3 : total_volume = initial_volume + added_volume)
                                  (h4 : final_percentage = 50)
                                  (h5 : init_percentage = 100 * (initial_volume * P / 100 + added_volume) / total_volume)
                                  : P = 25 :=
by {
  sorry
}

end initial_percentage_alcohol_l123_123821


namespace paper_clips_in_two_cases_l123_123488

theorem paper_clips_in_two_cases (c b : ℕ) : 
  2 * c * b * 200 = 2 * (c * b * 200) :=
by
  sorry

end paper_clips_in_two_cases_l123_123488


namespace remainder_x2023_l123_123978

theorem remainder_x2023 (x : ℤ) : 
  let dividend := x^2023 + 1
  let divisor := x^6 - x^4 + x^2 - 1
  let remainder := -x^7 + 1
  dividend % divisor = remainder :=
by
  sorry

end remainder_x2023_l123_123978


namespace gasoline_price_increase_l123_123910

theorem gasoline_price_increase
  (P Q : ℝ) -- Prices and quantities
  (x : ℝ) -- The percentage increase in price
  (h1 : (P * (1 + x / 100)) * (Q * 0.95) = P * Q * 1.14) -- Given condition
  : x = 20 := 
sorry

end gasoline_price_increase_l123_123910


namespace total_marbles_l123_123204

theorem total_marbles (r b y : ℕ) (h_ratio : 2 * b = 3 * r) (h_ratio_alt : 4 * b = 3 * y) (h_blue_marbles : b = 24) : r + b + y = 72 :=
by
  -- By assumption, b = 24
  have h1 : b = 24 := h_blue_marbles

  -- We have the ratios 2b = 3r and 4b = 3y
  have h2 : 2 * b = 3 * r := h_ratio
  have h3 : 4 * b = 3 * y := h_ratio_alt

  -- solved by given conditions 
  sorry

end total_marbles_l123_123204


namespace cos_240_eq_neg_half_l123_123311

theorem cos_240_eq_neg_half : ∀ (deg: ℝ), 
  deg = 240 → 
  (∀ a b : ℝ, 240 = a + b → a = 180 → b = 60 → 
    cos 240 = cos a * cos b - sin a * sin b) → 
  cos 180 = -1 →
  sin 180 = 0 →
  cos 60 = 1 / 2 →
  cos 240 = -1 / 2 :=
by 
  intros deg h_deg h_sum h_cos_180 h_sin_180 h_cos_60
  rw h_deg at h_sum
  have h_cos_identity := h_sum 180 60 rfl rfl rfl
  simp [h_cos_180, h_sin_180, h_cos_60] at h_cos_identity
  exact h_cos_identity

end cos_240_eq_neg_half_l123_123311


namespace tino_more_jellybeans_than_lee_l123_123246

-- Declare the conditions
variables (arnold_jellybeans lee_jellybeans tino_jellybeans : ℕ)
variables (arnold_jellybeans_half_lee : arnold_jellybeans = lee_jellybeans / 2)
variables (arnold_jellybean_count : arnold_jellybeans = 5)
variables (tino_jellybean_count : tino_jellybeans = 34)

-- The goal is to prove how many more jellybeans Tino has than Lee
theorem tino_more_jellybeans_than_lee : tino_jellybeans - lee_jellybeans = 24 :=
by
  sorry -- proof skipped

end tino_more_jellybeans_than_lee_l123_123246


namespace cos_240_eq_neg_half_l123_123300

theorem cos_240_eq_neg_half : Real.cos (4 * Real.pi / 3) = -1/2 :=
by
  sorry

end cos_240_eq_neg_half_l123_123300


namespace reeya_average_l123_123677

theorem reeya_average (s1 s2 s3 s4 s5 : ℕ) (h1 : s1 = 65) (h2 : s2 = 67) (h3 : s3 = 76) (h4 : s4 = 82) (h5 : s5 = 85) :
  (s1 + s2 + s3 + s4 + s5) / 5 = 75 := by
  sorry

end reeya_average_l123_123677


namespace Xiaoming_age_l123_123674

theorem Xiaoming_age (x : ℕ) (h1 : x = x) (h2 : x + 18 = 2 * (x + 6)) : x = 6 :=
sorry

end Xiaoming_age_l123_123674


namespace equal_costs_at_60_minutes_l123_123928

-- Define the base rates and the per minute rates for each company
def base_rate_united : ℝ := 9.00
def rate_per_minute_united : ℝ := 0.25
def base_rate_atlantic : ℝ := 12.00
def rate_per_minute_atlantic : ℝ := 0.20

-- Define the total cost functions
def cost_united (m : ℝ) : ℝ := base_rate_united + rate_per_minute_united * m
def cost_atlantic (m : ℝ) : ℝ := base_rate_atlantic + rate_per_minute_atlantic * m

-- State the theorem to be proved
theorem equal_costs_at_60_minutes : 
  ∃ (m : ℝ), cost_united m = cost_atlantic m ∧ m = 60 :=
by
  -- Pending proof
  sorry

end equal_costs_at_60_minutes_l123_123928


namespace identical_digits_time_l123_123642

theorem identical_digits_time (h : ∀ t, t = 355 -> ∃ u, u = 671 ∧ u - t = 316) : 
  ∃ u, u = 671 ∧ u - 355 = 316 := 
by sorry

end identical_digits_time_l123_123642


namespace triangle_inequality_proof_l123_123416

theorem triangle_inequality_proof (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 :=
sorry

end triangle_inequality_proof_l123_123416


namespace units_digit_sum_l123_123117

theorem units_digit_sum (n1 n2 : ℕ) (h1 : n1 % 10 = 1) (h2 : n2 % 10 = 3) : ((n1^3 + n2^3) % 10) = 8 := 
by
  sorry

end units_digit_sum_l123_123117


namespace tangent_circle_exists_l123_123800
open Set

-- Definitions of given point, line, and circle
variables {Point : Type*} {Line : Type*} {Circle : Type*} 
variables (M : Point) (l : Line) (S : Circle)
variables (center_S : Point) (radius_S : ℝ)

-- Conditions of the problem
variables (touches_line : Circle → Line → Prop) (touches_circle : Circle → Circle → Prop)
variables (passes_through : Circle → Point → Prop) (center_of : Circle → Point)
variables (radius_of : Circle → ℝ)

-- Existence theorem to prove
theorem tangent_circle_exists 
  (given_tangent_to_line : Circle → Line → Bool)
  (given_tangent_to_circle : Circle → Circle → Bool)
  (given_passes_through : Circle → Point → Bool):
  ∃ (Ω : Circle), 
    given_tangent_to_line Ω l ∧
    given_tangent_to_circle Ω S ∧
    given_passes_through Ω M :=
sorry

end tangent_circle_exists_l123_123800


namespace no_solution_iff_m_range_l123_123830

theorem no_solution_iff_m_range (m : ℝ) : 
  ¬ ∃ x : ℝ, |x-1| + |x-m| < 2*m ↔ (0 < m ∧ m < 1/3) := sorry

end no_solution_iff_m_range_l123_123830


namespace range_of_m_hyperbola_l123_123869

noncomputable def is_conic_hyperbola (expr : ℝ → ℝ → ℝ) : Prop :=
  ∃ f : ℝ, ∀ x y, expr x y = ((x - 2 * y + 3)^2 - f * (x^2 + y^2 + 2 * y + 1))

theorem range_of_m_hyperbola (m : ℝ) :
  is_conic_hyperbola (fun x y => m * (x^2 + y^2 + 2 * y + 1) - (x - 2 * y + 3)^2) → 5 < m :=
sorry

end range_of_m_hyperbola_l123_123869


namespace find_some_number_l123_123193

theorem find_some_number :
  ∃ (some_number : ℕ), let a := 105 in a^3 = 21 * 25 * some_number * 49 ∧ some_number = 5 :=
by
  sorry

end find_some_number_l123_123193


namespace problem_l123_123176

   def f (n : ℕ) : ℕ := sorry

   theorem problem (f : ℕ → ℕ) (h1 : ∀ n, f (f n) + f n = 2 * n + 3) (h2 : f 0 = 1) :
     f 2013 = 2014 :=
   sorry
   
end problem_l123_123176


namespace identical_digits_time_l123_123641

theorem identical_digits_time (h : ∀ t, t = 355 -> ∃ u, u = 671 ∧ u - t = 316) : 
  ∃ u, u = 671 ∧ u - 355 = 316 := 
by sorry

end identical_digits_time_l123_123641


namespace ninth_term_of_geometric_sequence_l123_123710

theorem ninth_term_of_geometric_sequence :
  let a1 := (5 : ℚ)
  let r := (3 / 4 : ℚ)
  (a1 * r^8) = (32805 / 65536 : ℚ) :=
by {
  sorry
}

end ninth_term_of_geometric_sequence_l123_123710


namespace parallel_line_through_point_l123_123464

theorem parallel_line_through_point (x y : ℝ) :
  (∃ (b : ℝ), (∀ (x : ℝ), y = 2 * x + b) ∧ y = 2 * 1 - 4) :=
sorry

end parallel_line_through_point_l123_123464


namespace triangle_inequalities_l123_123480

theorem triangle_inequalities (a b c : ℝ) (h : a < b + c) : b < a + c ∧ c < a + b := 
  sorry

end triangle_inequalities_l123_123480


namespace total_price_of_books_l123_123651

theorem total_price_of_books
  (total_books : ℕ)
  (math_books_cost : ℕ)
  (history_books_cost : ℕ)
  (math_books_bought : ℕ)
  (total_books_eq : total_books = 80)
  (math_books_cost_eq : math_books_cost = 4)
  (history_books_cost_eq : history_books_cost = 5)
  (math_books_bought_eq : math_books_bought = 10) :
  (math_books_bought * math_books_cost + (total_books - math_books_bought) * history_books_cost = 390) := 
by
  sorry

end total_price_of_books_l123_123651


namespace yellow_ball_kids_l123_123927

theorem yellow_ball_kids (total_kids white_ball_kids both_ball_kids : ℕ) :
  total_kids = 35 → white_ball_kids = 26 → both_ball_kids = 19 → 
  (total_kids = white_ball_kids + (total_kids - both_ball_kids)) → 
  (total_kids - (white_ball_kids - both_ball_kids)) = 28 :=
by
  sorry

end yellow_ball_kids_l123_123927


namespace circle_eq_tangent_x_axis_l123_123836

theorem circle_eq_tangent_x_axis (h k r : ℝ) (x y : ℝ)
  (h_center : h = -5)
  (k_center : k = 4)
  (tangent_x_axis : r = 4) :
  (x + 5)^2 + (y - 4)^2 = 16 :=
sorry

end circle_eq_tangent_x_axis_l123_123836


namespace units_digit_6_pow_4_l123_123471

-- Define the units digit function
def units_digit (n : ℕ) : ℕ := n % 10

-- Define the main theorem to prove
theorem units_digit_6_pow_4 : units_digit (6 ^ 4) = 6 := 
by
  sorry

end units_digit_6_pow_4_l123_123471


namespace marble_draw_l123_123261

/-- A container holds 30 red marbles, 25 green marbles, 23 yellow marbles,
15 blue marbles, 10 white marbles, and 7 black marbles. Prove that the
minimum number of marbles that must be drawn from the container without
replacement to ensure that at least 10 marbles of a single color are drawn
is 53. -/
theorem marble_draw (R G Y B W Bl : ℕ) (hR : R = 30) (hG : G = 25)
                               (hY : Y = 23) (hB : B = 15) (hW : W = 10)
                               (hBl : Bl = 7) : 
  ∃ (n : ℕ), n = 53 ∧ (∀ (x : ℕ), x ≠ n → 
  (x ≤ R → x ≤ G → x ≤ Y → x ≤ B → x ≤ W → x ≤ Bl → x < 10)) := 
by
  sorry

end marble_draw_l123_123261


namespace train_length_proof_l123_123505

-- Define speeds and time taken
def speed_train_kmph : ℝ := 63
def speed_man_kmph : ℝ := 3
def time_crossing_seconds : ℝ := 41.9966402687785

-- Speed conversion factor
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (5 / 18)

-- Relative speed in m/s
def relative_speed_mps : ℝ := kmph_to_mps (speed_train_kmph - speed_man_kmph)

-- Length of the train
def length_of_train : ℝ := relative_speed_mps * time_crossing_seconds

-- Proof stating the length of the train is approximately 699.94400447975 meters
theorem train_length_proof : abs (length_of_train - 699.94400447975) < 1e-6 := by
  sorry

end train_length_proof_l123_123505


namespace boys_and_girls_l123_123428

theorem boys_and_girls (B G : ℕ) (h1 : B + G = 30)
  (h2 : ∀ (i j : ℕ), i < B → j < B → i ≠ j → ∃ k, k < G ∧ ∀ l < B, l ≠ i → k ≠ l)
  (h3 : ∀ (i j : ℕ), i < G → j < G → i ≠ j → ∃ k, k < B ∧ ∀ l < G, l ≠ i → k ≠ l) :
  B = 15 ∧ G = 15 :=
by
  have hB : B ≤ G := sorry
  have hG : G ≤ B := sorry
  exact ⟨by linarith, by linarith⟩

end boys_and_girls_l123_123428


namespace solution_is_x_l123_123717

def find_x (x : ℝ) : Prop :=
  64 * (x + 1)^3 - 27 = 0

theorem solution_is_x : ∃ x : ℝ, find_x x ∧ x = -1 / 4 :=
by
  sorry

end solution_is_x_l123_123717


namespace smallest_integer_l123_123668

-- Given positive integer M such that
def satisfies_conditions (M : ℕ) : Prop :=
  M % 6 = 5 ∧
  M % 7 = 6 ∧
  M % 8 = 7 ∧
  M % 9 = 8 ∧
  M % 10 = 9 ∧
  M % 11 = 10 ∧
  M % 13 = 12

-- The main theorem to prove
theorem smallest_integer (M : ℕ) (h : satisfies_conditions M) : M = 360359 :=
  sorry

end smallest_integer_l123_123668


namespace dance_pairs_exist_l123_123013

variable {Boy Girl : Type} 

-- Define danced_with relation
variable (danced_with : Boy → Girl → Prop)

-- Given conditions
variable (H1 : ∀ (b : Boy), ∃ (g : Girl), ¬ danced_with b g)
variable (H2 : ∀ (g : Girl), ∃ (b : Boy), danced_with b g)

-- Proof that desired pairs exist
theorem dance_pairs_exist :
  ∃ (M1 M2 : Boy) (D1 D2 : Girl),
    danced_with M1 D1 ∧
    danced_with M2 D2 ∧
    ¬ danced_with M1 D2 ∧
    ¬ danced_with M2 D1 :=
sorry

end dance_pairs_exist_l123_123013


namespace maximum_area_triangle_l123_123082

theorem maximum_area_triangle 
  (A B C : ℝ)
  (h₁ : A + B + C = 180)
  (h₂ : ∀ A B : ℝ, ∀ a b : ℝ, (\frac{cos A}{sin B} + \frac{cos B}{sin A}) = 2)
  (h₃ : a + b + c = 12)
  : let area := \frac{1}{2} * a * b
    ∃ max_area : ℝ, max_area = 36 * (3 - 2 * sqrt 2)
    := sorry

end maximum_area_triangle_l123_123082


namespace bob_investment_correct_l123_123509

noncomputable def initial_investment_fundA : ℝ := 2000
noncomputable def interest_rate_fundA : ℝ := 0.12
noncomputable def initial_investment_fundB : ℝ := 1000
noncomputable def interest_rate_fundB : ℝ := 0.30
noncomputable def fundA_after_two_years := initial_investment_fundA * (1 + interest_rate_fundA)
noncomputable def fundB_after_two_years (B : ℝ) := B * (1 + interest_rate_fundB)^2
noncomputable def extra_value : ℝ := 549.9999999999998

theorem bob_investment_correct :
  fundA_after_two_years = fundB_after_two_years initial_investment_fundB + extra_value :=
by
  sorry

end bob_investment_correct_l123_123509


namespace half_absolute_difference_of_squares_l123_123662

-- Defining the variables a and b involved in the problem
def a : ℤ := 20
def b : ℤ := 15

-- Statement to prove the solution
theorem half_absolute_difference_of_squares : 
    (1 / 2 : ℚ) * (abs ((a ^ 2) - (b ^ 2))) = 87.5 :=
by
    -- Proof omitted
    sorry

end half_absolute_difference_of_squares_l123_123662


namespace min_value_binom_l123_123751

theorem min_value_binom
  (a b : ℕ → ℕ)
  (n : ℕ) (hn : 0 < n)
  (h1 : ∀ n, a n = 2^n)
  (h2 : ∀ n, b n = 4^n) :
  ∃ n, 2^n + (1 / 2^n) = 5 / 2 :=
sorry

end min_value_binom_l123_123751


namespace num_two_digit_primes_with_digit_sum_10_l123_123535

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def digits_sum_to_10 (n : ℕ) : Prop := (n / 10) + (n % 10) = 10

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem num_two_digit_primes_with_digit_sum_10 : 
  {n : ℕ | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.to_finset.card = 3 := by
  sorry

end num_two_digit_primes_with_digit_sum_10_l123_123535


namespace cos_240_eq_neg_half_l123_123302

theorem cos_240_eq_neg_half : Real.cos (4 * Real.pi / 3) = -1/2 :=
by
  sorry

end cos_240_eq_neg_half_l123_123302


namespace division_problem_l123_123278

theorem division_problem : 240 / (12 + 14 * 2) = 6 := by
  sorry

end division_problem_l123_123278


namespace symmetric_point_about_origin_l123_123624

theorem symmetric_point_about_origin (P Q : ℤ × ℤ) (h : P = (-2, -3)) : Q = (2, 3) :=
by
  sorry

end symmetric_point_about_origin_l123_123624


namespace hands_per_hoopit_l123_123608

-- Defining conditions
def num_hoopits := 7
def num_neglarts := 8
def total_toes := 164
def toes_per_hand_hoopit := 3
def toes_per_hand_neglart := 2
def hands_per_neglart := 5

-- The statement to prove
theorem hands_per_hoopit : 
  ∃ (H : ℕ), (H * toes_per_hand_hoopit * num_hoopits + hands_per_neglart * toes_per_hand_neglart * num_neglarts = total_toes) → H = 4 :=
sorry

end hands_per_hoopit_l123_123608


namespace find_number_l123_123935

theorem find_number (x : ℝ) (h : 0.30 * x = 90 + 120) : x = 700 :=
by 
  sorry

end find_number_l123_123935


namespace expression_evaluation_l123_123705

theorem expression_evaluation :
  (3 * Real.sqrt 12 - 2 * Real.sqrt (1 / 3) + Real.sqrt 48) / (2 * Real.sqrt 3) + (Real.sqrt (1 / 3))^2 = 5 :=
by
  sorry

end expression_evaluation_l123_123705


namespace evaluate_fraction_l123_123863

variable (a b x : ℝ)
variable (h1 : a ≠ b)
variable (h2 : b ≠ 0)
variable (h3 : x = a / b)

theorem evaluate_fraction :
  (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by
  sorry

end evaluate_fraction_l123_123863


namespace first_discount_percentage_l123_123795

theorem first_discount_percentage
  (P : ℝ)
  (initial_price final_price : ℝ)
  (second_discount : ℕ)
  (h1 : initial_price = 200)
  (h2 : final_price = 144)
  (h3 : second_discount = 10)
  (h4 : final_price = (P - (second_discount / 100) * P)) :
  (∃ x : ℝ, P = initial_price - (x / 100) * initial_price ∧ x = 20) :=
sorry

end first_discount_percentage_l123_123795


namespace half_abs_diff_squares_l123_123659

theorem half_abs_diff_squares : 
  let a := 20 in
  let b := 15 in
  let square (x : ℕ) := x * x in
  let abs_diff := abs (square a - square b) in
  abs_diff / 2 = 87.5 := 
by
  let a := 20
  let b := 15
  let square (x : ℕ) := x * x
  let abs_diff := abs (square a - square b)
  have h1 : square a = 400 := by native_decide
  have h2 : square b = 225 := by native_decide
  have h3 : abs_diff = abs (400 - 225) := by simp [square, h1, h2]
  have h4 : abs_diff = 175 := by simp [h3]
  have h5 : abs_diff / 2 = 87.5 := by norm_num [h4]
  exact h5

end half_abs_diff_squares_l123_123659


namespace find_n_l123_123184

theorem find_n (n : ℕ) (h₁ : n > 0) (h₂ : 3 * Nat.choose (n-1) (n-5) = 5 * (Nat.Perm (n-2) 2)^2) : n = 9 :=
by
  sorry

end find_n_l123_123184


namespace star_five_seven_l123_123845

def star (a b : ℕ) : ℕ := (a + b + 3) ^ 2

theorem star_five_seven : star 5 7 = 225 := by
  sorry

end star_five_seven_l123_123845


namespace range_of_c_l123_123746

variable {a c : ℝ}

theorem range_of_c (h : a ≥ 1 / 8) (sufficient_but_not_necessary : ∀ x > 0, 2 * x + a / x ≥ c) : c ≤ 1 := 
sorry

end range_of_c_l123_123746


namespace problem_equivalent_l123_123551

def digits_sum_to_10 (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

noncomputable def two_digit_prime_numbers_sum_10 : ℕ :=
  {n | is_two_digit n ∧ is_prime n ∧ digits_sum_to_10 n}.card

theorem problem_equivalent : two_digit_prime_numbers_sum_10 = 3 :=
sorry

end problem_equivalent_l123_123551


namespace audi_crossing_intersection_between_17_and_18_l123_123248

-- Given conditions:
-- Two cars, an Audi and a BMW, are moving along two intersecting roads at equal constant speeds.
-- At both 17:00 and 18:00, the BMW was twice as far from the intersection as the Audi.
-- Let the distance of Audi from the intersection at 17:00 be x and BMW's distance be 2x.
-- Both vehicles travel at a constant speed v.

noncomputable def car_position (initial_distance : ℝ) (velocity : ℝ) (time_elapsed : ℝ) : ℝ :=
  initial_distance + velocity * time_elapsed

theorem audi_crossing_intersection_between_17_and_18 (x v : ℝ) :
  ∃ t : ℝ, (t = 15 ∨ t = 45) ∧
    car_position x (-v) (t/60) = 0 ∧ car_position (2 * x) (-v) (t/60) = 2 * car_position x (-v) (1 - t/60) :=
sorry

end audi_crossing_intersection_between_17_and_18_l123_123248


namespace count_two_digit_prime_with_digit_sum_10_l123_123529

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_two_digit_prime_with_digit_sum_10 : 
  (∃ n1 n2 n3 : ℕ, 
    (sum_of_digits n1 = 10 ∧ is_prime n1 ∧ 10 ≤ n1 ∧ n1 < 100) ∧
    (sum_of_digits n2 = 10 ∧ is_prime n2 ∧ 10 ≤ n2 ∧ n2 < 100) ∧
    (sum_of_digits n3 = 10 ∧ is_prime n3 ∧ 10 ≤ n3 ∧ n3 < 100) ∧
    n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) ∧
  ∀ n : ℕ, 
    (sum_of_digits n = 10 ∧ is_prime n ∧ 10 ≤ n ∧ n < 100)
    → (n = n1 ∨ n = n2 ∨ n = n3) :=
sorry

end count_two_digit_prime_with_digit_sum_10_l123_123529


namespace largest_n_for_factoring_l123_123728

theorem largest_n_for_factoring :
  ∃ n, (∃ A B : ℤ, (3 * A = 3 * 108 + 1) ∧ (/3 * B * 108 = 2) ∧ 
  (3 * 36 + 3 = 111) ∧ (3 * 108 + A = n) )=
  (n = 325) := sorry
iddenLean_formatter.clonecreateAngular

end largest_n_for_factoring_l123_123728


namespace find_first_number_l123_123992

theorem find_first_number (n : ℝ) (h1 : n / 14.5 = 175) :
  n = 2537.5 :=
by 
  sorry

end find_first_number_l123_123992


namespace half_abs_diff_squares_l123_123664

theorem half_abs_diff_squares (a b : ℤ) (h₁ : a = 21) (h₂ : b = 17) :
  (|a^2 - b^2| / 2) = 76 :=
by 
  sorry

end half_abs_diff_squares_l123_123664
