import Mathlib

namespace perimeter_eq_20_l624_62404

-- Define the lengths of the sides
def horizontal_sides := [2, 3]
def vertical_sides := [2, 3, 3, 2]

-- Define the perimeter calculation
def perimeter := horizontal_sides.sum + vertical_sides.sum

theorem perimeter_eq_20 : perimeter = 20 :=
by
  -- We assert that the calculations do hold
  sorry

end perimeter_eq_20_l624_62404


namespace tourist_growth_rate_l624_62426

theorem tourist_growth_rate (F : ℝ) (x : ℝ) 
    (hMarch : F * 0.6 = 0.6 * F)
    (hApril : F * 0.6 * 0.5 = 0.3 * F)
    (hMay : 2 * F = 2 * F):
    (0.6 * 0.5 * (1 + x) = 2) :=
by
  sorry

end tourist_growth_rate_l624_62426


namespace part_a_l624_62434

theorem part_a (x : ℝ) (hx : x ≥ 1) : x^3 - 5 * x^2 + 8 * x - 4 ≥ 0 := 
  sorry

end part_a_l624_62434


namespace num_real_roots_of_abs_x_eq_l624_62431

theorem num_real_roots_of_abs_x_eq (k : ℝ) (hk : 6 < k ∧ k < 7) 
  : (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    (|x1| * x1 - 2 * x1 + 7 - k = 0) ∧ 
    (|x2| * x2 - 2 * x2 + 7 - k = 0) ∧
    (|x3| * x3 - 2 * x3 + 7 - k = 0)) ∧
  (¬ ∃ x4 : ℝ, x4 ≠ x1 ∧ x4 ≠ x2 ∧ x4 ≠ x3 ∧ |x4| * x4 - 2 * x4 + 7 - k = 0) :=
sorry

end num_real_roots_of_abs_x_eq_l624_62431


namespace sum_smallest_largest_2y_l624_62430

variable (a n y : ℤ)

noncomputable def is_even (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k
noncomputable def is_odd (x : ℤ) : Prop := ∃ k : ℤ, x = 2 * k + 1

theorem sum_smallest_largest_2y 
  (h1 : is_odd a) 
  (h2 : n % 2 = 0) 
  (h3 : y = a + n) : 
  a + (a + 2 * n) = 2 * y := 
by 
  sorry

end sum_smallest_largest_2y_l624_62430


namespace minimum_value_of_f_l624_62439

noncomputable def f (x : ℝ) := 2 * x + 18 / x

theorem minimum_value_of_f :
  ∃ x > 0, f x = 12 ∧ ∀ y > 0, f y ≥ 12 :=
by
  sorry

end minimum_value_of_f_l624_62439


namespace max_of_2xy_l624_62477

theorem max_of_2xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4) : 2 * x * y ≤ 8 :=
by
  sorry

end max_of_2xy_l624_62477


namespace min_value_of_function_l624_62428

noncomputable def func (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + Real.sin (2 * x)

theorem min_value_of_function : ∃ x : ℝ, func x = 1 - Real.sqrt 2 :=
by sorry

end min_value_of_function_l624_62428


namespace total_profit_calculation_l624_62400

-- Definitions based on conditions
def initial_investment_A := 5000
def initial_investment_B := 8000
def initial_investment_C := 9000
def initial_investment_D := 7000

def investment_A_after_4_months := initial_investment_A + 2000
def investment_B_after_4_months := initial_investment_B - 1000

def investment_C_after_6_months := initial_investment_C + 3000
def investment_D_after_6_months := initial_investment_D + 5000

def profit_A_percentage := 20
def profit_B_percentage := 30
def profit_C_percentage := 25
def profit_D_percentage := 25

def profit_C := 60000

-- Total profit is what we need to determine
def total_profit := 240000

-- The proof statement
theorem total_profit_calculation :
  total_profit = (profit_C * 100) / profit_C_percentage := 
by 
  sorry

end total_profit_calculation_l624_62400


namespace no_such_triplets_of_positive_reals_l624_62461

-- Define the conditions that the problem states.
def satisfies_conditions (a b c : ℝ) : Prop :=
  a = b + c ∧ b = c + a ∧ c = a + b

-- The main theorem to prove.
theorem no_such_triplets_of_positive_reals :
  ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) → satisfies_conditions a b c → false :=
by
  intro a b c
  intro ha hb hc
  intro habc
  sorry

end no_such_triplets_of_positive_reals_l624_62461


namespace probability_from_first_to_last_l624_62450

noncomputable def probability_path_possible (n : ℕ) : ℝ :=
  (2 ^ (n - 1) : ℝ) / (Nat.choose (2 * (n - 1)) (n - 1) : ℝ)

theorem probability_from_first_to_last (n : ℕ) (h : n > 1) :
  probability_path_possible n = 2 ^ (n - 1) / (Nat.choose (2 * (n - 1)) (n - 1)) := sorry

end probability_from_first_to_last_l624_62450


namespace largest_alpha_exists_l624_62479

theorem largest_alpha_exists : 
  ∃ α, (∀ m n : ℕ, 0 < m → 0 < n → (m:ℝ) / (n:ℝ) < Real.sqrt 7 → α / (n^2:ℝ) ≤ 7 - (m^2:ℝ) / (n^2:ℝ)) ∧ α = 3 :=
by
  sorry

end largest_alpha_exists_l624_62479


namespace num_rectangles_grid_l624_62449

theorem num_rectangles_grid (m n : ℕ) (hm : m = 5) (hn : n = 5) :
  let horiz_lines := m + 1
  let vert_lines := n + 1
  let num_ways_choose_2 (x : ℕ) := x * (x - 1) / 2
  num_ways_choose_2 horiz_lines * num_ways_choose_2 vert_lines = 225 :=
by
  sorry

end num_rectangles_grid_l624_62449


namespace find_f_at_one_l624_62437

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 4 * x ^ 2 - m * x + 5

theorem find_f_at_one :
  (∀ x : ℝ, x ≥ -2 → f x (-16) ≥ f (-2) (-16)) ∧
  (∀ x : ℝ, x ≤ -2 → f x (-16) ≤ f (-2) (-16)) →
  f 1 (-16) = 25 :=
sorry

end find_f_at_one_l624_62437


namespace remainder_when_divided_by_5_l624_62427

theorem remainder_when_divided_by_5 
  (k : ℕ)
  (h1 : k % 6 = 5)
  (h2 : k < 42)
  (h3 : k % 7 = 3) : 
  k % 5 = 2 := 
by 
  sorry

end remainder_when_divided_by_5_l624_62427


namespace simplify_and_evaluate_expression_l624_62489

theorem simplify_and_evaluate_expression (a b : ℤ) (h₁ : a = 1) (h₂ : b = -2) :
  (2 * a + b)^2 - 3 * a * (2 * a - b) = -12 :=
by
  rw [h₁, h₂]
  -- Now the expression to prove transforms to:
  -- (2 * 1 + (-2))^2 - 3 * 1 * (2 * 1 - (-2)) = -12
  -- Subsequent proof steps would follow simplification directly.
  sorry

end simplify_and_evaluate_expression_l624_62489


namespace q_one_eq_five_l624_62457

variable (q : ℝ → ℝ)
variable (h : q 1 = 5)

theorem q_one_eq_five : q 1 = 5 :=
by sorry

end q_one_eq_five_l624_62457


namespace total_cans_collected_l624_62429

theorem total_cans_collected (students_perez : ℕ) (half_perez_collected_20 : ℕ) (two_perez_collected_0 : ℕ) (remaining_perez_collected_8 : ℕ)
                             (students_johnson : ℕ) (third_johnson_collected_25 : ℕ) (three_johnson_collected_0 : ℕ) (remaining_johnson_collected_10 : ℕ)
                             (hp : students_perez = 28) (hc1 : half_perez_collected_20 = 28 / 2) (hc2 : two_perez_collected_0 = 2) (hc3 : remaining_perez_collected_8 = 12)
                             (hj : students_johnson = 30) (jc1 : third_johnson_collected_25 = 30 / 3) (jc2 : three_johnson_collected_0 = 3) (jc3 : remaining_johnson_collected_10 = 18) :
    (half_perez_collected_20 * 20 + two_perez_collected_0 * 0 + remaining_perez_collected_8 * 8
    + third_johnson_collected_25 * 25 + three_johnson_collected_0 * 0 + remaining_johnson_collected_10 * 10) = 806 :=
by
  sorry

end total_cans_collected_l624_62429


namespace increasing_interval_of_f_l624_62469

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem increasing_interval_of_f :
  ∀ x, x > 2 → ∀ y, y > x → f x < f y :=
sorry

end increasing_interval_of_f_l624_62469


namespace find_common_ratio_l624_62478

theorem find_common_ratio (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : 3 * S 3 = a 4 - 2)
  (h4 : 3 * S 2 = a 3 - 2)
  (h5 : ∀ n : ℕ, a (n+1) = q * a n) : q = 4 := sorry

end find_common_ratio_l624_62478


namespace fred_gave_cards_l624_62412

theorem fred_gave_cards (initial_cards : ℕ) (torn_cards : ℕ) 
  (bought_cards : ℕ) (total_cards : ℕ) (fred_cards : ℕ) : 
  initial_cards = 18 → torn_cards = 8 → bought_cards = 40 → total_cards = 84 →
  fred_cards = total_cards - (initial_cards - torn_cards + bought_cards) →
  fred_cards = 34 :=
by
  intros h_initial h_torn h_bought h_total h_fred
  sorry

end fred_gave_cards_l624_62412


namespace company_C_more_than_A_l624_62432

theorem company_C_more_than_A (A B C D: ℕ) (hA: A = 30) (hB: B = 2 * A)
    (hC: C = A + 10) (hD: D = C - 5) (total: A + B + C + D = 165) : C - A = 10 := 
by 
  sorry

end company_C_more_than_A_l624_62432


namespace frogs_meet_time_proven_l624_62453

-- Define the problem
def frogs_will_meet_at_time : Prop :=
  ∃ (meet_time : Nat),
    let initial_time := 12 * 60 -- 12:00 PM in minutes
    let initial_distance := 2015
    let green_frog_jump := 9
    let blue_frog_jump := 8 
    let combined_reduction := green_frog_jump + blue_frog_jump
    initial_distance % combined_reduction = 0 ∧
    meet_time == initial_time + (2 * (initial_distance / combined_reduction))

theorem frogs_meet_time_proven (h : frogs_will_meet_at_time) : meet_time = 15 * 60 + 56 :=
sorry

end frogs_meet_time_proven_l624_62453


namespace min_value_4a2_b2_plus_1_div_2a_minus_b_l624_62422

variable (a b : ℝ)

theorem min_value_4a2_b2_plus_1_div_2a_minus_b (h1 : 0 < a) (h2 : 0 < b)
  (h3 : a > b) (h4 : a * b = 1 / 2) : 
  ∃ c : ℝ, c = 2 * Real.sqrt 3 ∧ (∀ x y : ℝ, 0 < x → 0 < y → x > y → x * y = 1 / 2 → (4 * x^2 + y^2 + 1) / (2 * x - y) ≥ c) :=
sorry

end min_value_4a2_b2_plus_1_div_2a_minus_b_l624_62422


namespace weight_of_each_bag_of_planks_is_14_l624_62455

-- Definitions
def crate_capacity : Nat := 20
def num_crates : Nat := 15
def num_bags_nails : Nat := 4
def weight_bag_nails : Nat := 5
def num_bags_hammers : Nat := 12
def weight_bag_hammers : Nat := 5
def num_bags_planks : Nat := 10
def weight_to_leave_out : Nat := 80

-- Total weight calculations
def weight_nails := num_bags_nails * weight_bag_nails
def weight_hammers := num_bags_hammers * weight_bag_hammers
def total_weight_nails_hammers := weight_nails + weight_hammers
def total_crate_capacity := num_crates * crate_capacity
def weight_that_can_be_loaded := total_crate_capacity - weight_to_leave_out
def weight_available_for_planks := weight_that_can_be_loaded - total_weight_nails_hammers
def weight_each_bag_planks := weight_available_for_planks / num_bags_planks

-- Theorem statement
theorem weight_of_each_bag_of_planks_is_14 : weight_each_bag_planks = 14 :=
by {
  sorry
}

end weight_of_each_bag_of_planks_is_14_l624_62455


namespace Jenna_total_cost_l624_62407

theorem Jenna_total_cost :
  let skirt_material := 12 * 4
  let total_skirt_material := skirt_material * 3
  let sleeve_material := 5 * 2
  let bodice_material := 2
  let total_material := total_skirt_material + sleeve_material + bodice_material
  let cost_per_square_foot := 3
  let total_cost := total_material * cost_per_square_foot
  total_cost = 468 :=
by
  let skirt_material := 12 * 4
  let total_skirt_material := skirt_material * 3
  let sleeve_material := 5 * 2
  let bodice_material := 2
  let total_material := total_skirt_material + sleeve_material + bodice_material
  let cost_per_square_foot := 3
  let total_cost := total_material * cost_per_square_foot
  show total_cost = 468
  sorry

end Jenna_total_cost_l624_62407


namespace cos_five_pi_over_three_l624_62420

theorem cos_five_pi_over_three : Real.cos (5 * Real.pi / 3) = 1 / 2 := 
by 
  sorry

end cos_five_pi_over_three_l624_62420


namespace rewrite_neg_multiplication_as_exponent_l624_62463

theorem rewrite_neg_multiplication_as_exponent :
  -2 * 2 * 2 * 2 = - (2^4) :=
by
  sorry

end rewrite_neg_multiplication_as_exponent_l624_62463


namespace evaluate_expression_l624_62409

theorem evaluate_expression : -(16 / 4 * 11 - 70 + 5 * 11) = -29 := by
  sorry

end evaluate_expression_l624_62409


namespace find_a_l624_62458

theorem find_a (a : ℤ) (h_range : 0 ≤ a ∧ a < 13) (h_div : (51 ^ 2022 + a) % 13 = 0) : a = 12 := 
by
  sorry

end find_a_l624_62458


namespace profit_percentage_l624_62493

theorem profit_percentage (SP CP : ℕ) (h₁ : SP = 800) (h₂ : CP = 640) : (SP - CP) / CP * 100 = 25 :=
by 
  sorry

end profit_percentage_l624_62493


namespace units_digit_7_pow_2023_l624_62467

-- Define a function to compute the units digit
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- Main theorem to prove: the units digit of 7^2023 equals 3
theorem units_digit_7_pow_2023 : units_digit (7^2023) = 3 :=
by {
  -- It suffices to show that the units digits of powers of 7 repeat every 4 with the pattern [7, 9, 3, 1]
  sorry
}

end units_digit_7_pow_2023_l624_62467


namespace minimum_cost_l624_62466

theorem minimum_cost (price_pen_A price_pen_B price_notebook_A price_notebook_B : ℕ) 
  (discount_B : ℚ) (num_pens num_notebooks : ℕ)
  (h_price_pen : price_pen_A = 10) (h_price_notebook : price_notebook_A = 2)
  (h_discount : discount_B = 0.9) (h_num_pens : num_pens = 4) (h_num_notebooks : num_notebooks = 24) :
  ∃ (min_cost : ℕ), min_cost = 76 :=
by
  -- The conditions should be used here to construct the min_cost
  sorry

end minimum_cost_l624_62466


namespace schur_theorem_l624_62417

theorem schur_theorem {n : ℕ} (P : Fin n → Set ℕ) (h_partition : ∀ x : ℕ, ∃ i : Fin n, x ∈ P i) :
  ∃ (i : Fin n) (x y : ℕ), x ∈ P i ∧ y ∈ P i ∧ x + y ∈ P i :=
sorry

end schur_theorem_l624_62417


namespace inequality_sqrt_sum_leq_one_plus_sqrt_l624_62497

theorem inequality_sqrt_sum_leq_one_plus_sqrt (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  Real.sqrt (a * (1 - b) * (1 - c)) + Real.sqrt (b * (1 - a) * (1 - c)) + Real.sqrt (c * (1 - a) * (1 - b)) 
  ≤ 1 + Real.sqrt (a * b * c) :=
sorry

end inequality_sqrt_sum_leq_one_plus_sqrt_l624_62497


namespace number_of_intersections_l624_62460

def line_eq (x y : ℝ) : Prop := 4 * x + 9 * y = 12
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9

theorem number_of_intersections : 
  ∃ (p1 p2 : ℝ × ℝ), 
  (line_eq p1.1 p1.2 ∧ circle_eq p1.1 p1.2) ∧ 
  (line_eq p2.1 p2.2 ∧ circle_eq p2.1 p2.2) ∧ 
  p1 ≠ p2 ∧ 
  ∀ p : ℝ × ℝ, 
    (line_eq p.1 p.2 ∧ circle_eq p.1 p.2) → (p = p1 ∨ p = p2) :=
sorry

end number_of_intersections_l624_62460


namespace count_of_green_hats_l624_62459

-- Defining the total number of hats
def total_hats : ℕ := 85

-- Defining the costs of each hat type
def blue_cost : ℕ := 6
def green_cost : ℕ := 7
def red_cost : ℕ := 8

-- Defining the total cost
def total_cost : ℕ := 600

-- Defining the ratio as 3:2:1
def ratio_blue : ℕ := 3
def ratio_green : ℕ := 2
def ratio_red : ℕ := 1

-- Defining the multiplication factor
def x : ℕ := 14

-- Number of green hats based on the ratio
def G : ℕ := ratio_green * x

-- Proving that we bought 28 green hats
theorem count_of_green_hats : G = 28 := by
  -- proof steps intention: sorry to skip the proof
  sorry

end count_of_green_hats_l624_62459


namespace find_a_l624_62421

noncomputable def f (a x : ℝ) : ℝ :=
  (1 / 2 : ℝ) * a * x^3 - (3 / 2 : ℝ) * x^2 + (3 / 2 : ℝ) * a^2 * x

theorem find_a (a : ℝ) (h_max : ∀ x : ℝ, f a x ≤ f a 1) : a = -2 :=
sorry

end find_a_l624_62421


namespace part_a_part_b_l624_62415

-- Define the cost variables for chocolates, popsicles, and lollipops
variables (C P L : ℕ)

-- Given conditions
axiom cost_relation1 : 3 * C = 2 * P
axiom cost_relation2 : 2 * L = 5 * C

-- Part (a): Prove that Mário can buy 5 popsicles with the money for 3 lollipops
theorem part_a : 
  (3 * L) / P = 5 :=
by sorry

-- Part (b): Prove that Mário can buy 11 chocolates with the money for 3 chocolates, 2 popsicles, and 2 lollipops combined
theorem part_b : 
  (3 * C + 2 * P + 2 * L) / C = 11 :=
by sorry

end part_a_part_b_l624_62415


namespace students_both_l624_62408

noncomputable def students_total : ℕ := 32
noncomputable def students_go : ℕ := 18
noncomputable def students_chess : ℕ := 23

theorem students_both : students_go + students_chess - students_total = 9 := by
  sorry

end students_both_l624_62408


namespace Ken_bought_2_pounds_of_steak_l624_62484

theorem Ken_bought_2_pounds_of_steak (pound_cost total_paid change: ℝ) 
    (h1 : pound_cost = 7) 
    (h2 : total_paid = 20) 
    (h3 : change = 6) : 
    (total_paid - change) / pound_cost = 2 :=
by
  sorry

end Ken_bought_2_pounds_of_steak_l624_62484


namespace sample_size_eq_36_l624_62423

def total_population := 27 + 54 + 81
def ratio_elderly_total := 27 / total_population
def selected_elderly := 6
def sample_size := 36

theorem sample_size_eq_36 : 
  (selected_elderly : ℚ) / (sample_size : ℚ) = ratio_elderly_total → 
  sample_size = 36 := 
by 
sorry

end sample_size_eq_36_l624_62423


namespace matches_needed_eq_l624_62475

def count_matches (n : ℕ) : ℕ :=
  let total_triangles := n * n
  let internal_matches := 3 * total_triangles
  let external_matches := 4 * n
  internal_matches - external_matches + external_matches

theorem matches_needed_eq (n : ℕ) : count_matches 10 = 320 :=
by
  sorry

end matches_needed_eq_l624_62475


namespace picnic_attendance_l624_62472

theorem picnic_attendance (L x : ℕ) (h1 : L + x = 2015) (h2 : L - (x - 1) = 4) : x = 1006 := 
by
  sorry

end picnic_attendance_l624_62472


namespace choose_president_vice_president_and_committee_l624_62451

theorem choose_president_vice_president_and_committee :
  let num_ways : ℕ := 10 * 9 * (Nat.choose 8 2)
  num_ways = 2520 :=
by
  sorry

end choose_president_vice_president_and_committee_l624_62451


namespace square_lawn_side_length_l624_62494

theorem square_lawn_side_length (length width : ℕ) (h_length : length = 18) (h_width : width = 8) : 
  ∃ x : ℕ, x * x = length * width ∧ x = 12 := by
  -- Assume the necessary definitions and theorems to build the proof
  sorry

end square_lawn_side_length_l624_62494


namespace shorter_base_length_l624_62483

-- Let AB be the longer base of the trapezoid with length 24 cm
def AB : ℝ := 24

-- Let KT be the distance between midpoints of the diagonals with length 4 cm
def KT : ℝ := 4

-- Let CD be the shorter base of the trapezoid
variable (CD : ℝ)

-- The given condition is that KT is equal to half the difference of the lengths of the bases
axiom KT_eq : KT = (AB - CD) / 2

theorem shorter_base_length : CD = 16 := by
  sorry

end shorter_base_length_l624_62483


namespace number_increased_by_one_fourth_l624_62473

theorem number_increased_by_one_fourth (n : ℕ) (h : 25 * 80 / 100 = 20) (h1 : 80 - 20 = 60) :
  n + n / 4 = 60 ↔ n = 48 :=
by
  -- Conditions
  have h2 : 80 - 25 * 80 / 100 = 60 := by linarith [h, h1]
  have h3 : n + n / 4 = 60 := sorry
  -- Assertion (Proof to show is omitted)
  sorry

end number_increased_by_one_fourth_l624_62473


namespace total_revenue_correct_l624_62435

def price_per_book : ℝ := 25
def books_sold_monday : ℕ := 60
def discount_monday : ℝ := 0.10
def books_sold_tuesday : ℕ := 10
def discount_tuesday : ℝ := 0.0
def books_sold_wednesday : ℕ := 20
def discount_wednesday : ℝ := 0.05
def books_sold_thursday : ℕ := 44
def discount_thursday : ℝ := 0.15
def books_sold_friday : ℕ := 66
def discount_friday : ℝ := 0.20

def revenue (books_sold: ℕ) (discount: ℝ) : ℝ :=
  (1 - discount) * price_per_book * books_sold

theorem total_revenue_correct :
  revenue books_sold_monday discount_monday +
  revenue books_sold_tuesday discount_tuesday +
  revenue books_sold_wednesday discount_wednesday +
  revenue books_sold_thursday discount_thursday +
  revenue books_sold_friday discount_friday = 4330 := by 
sorry

end total_revenue_correct_l624_62435


namespace num_O_atoms_correct_l624_62481

-- Conditions
def atomic_weight_H : ℕ := 1
def atomic_weight_Cr : ℕ := 52
def atomic_weight_O : ℕ := 16
def num_H_atoms : ℕ := 2
def num_Cr_atoms : ℕ := 1
def molecular_weight : ℕ := 118

-- Calculations
def weight_H : ℕ := num_H_atoms * atomic_weight_H
def weight_Cr : ℕ := num_Cr_atoms * atomic_weight_Cr
def total_weight_H_Cr : ℕ := weight_H + weight_Cr
def weight_O : ℕ := molecular_weight - total_weight_H_Cr
def num_O_atoms : ℕ := weight_O / atomic_weight_O

-- Theorem to prove the number of Oxygen atoms is 4
theorem num_O_atoms_correct : num_O_atoms = 4 :=
by {
  sorry -- Proof not provided.
}

end num_O_atoms_correct_l624_62481


namespace range_of_m_l624_62490

/-- The range of the real number m such that the equation x^2/m + y^2/(2m - 1) = 1 represents an ellipse with foci on the x-axis is (1/2, 1). -/
theorem range_of_m (m : ℝ) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (∀ x y : ℝ, x^2 / m + y^2 / (2 * m - 1) = 1 → x^2 / a^2 + y^2 / b^2 = 1 ∧ b^2 < a^2))
  ↔ 1 / 2 < m ∧ m < 1 :=
sorry

end range_of_m_l624_62490


namespace find_term_ninth_term_l624_62456

variable (a_1 d a_k a_12 : ℤ)
variable (S_20 : ℤ := 200)

-- Definitions of the given conditions
def term_n (a_1 : ℤ) (d : ℤ) (n : ℤ) : ℤ := a_1 + (n - 1) * d

-- Problem Statement
theorem find_term_ninth_term :
  (∃ k, term_n a_1 d k + term_n a_1 d 12 = 20) ∧ 
  (S_20 = 10 * (2 * a_1 + 19 * d)) → 
  ∃ k, k = 9 :=
by sorry

end find_term_ninth_term_l624_62456


namespace find_MorkTaxRate_l624_62487

noncomputable def MorkIncome : ℝ := sorry
noncomputable def MorkTaxRate : ℝ := sorry 
noncomputable def MindyTaxRate : ℝ := 0.30 
noncomputable def MindyIncome : ℝ := 4 * MorkIncome 
noncomputable def combinedTaxRate : ℝ := 0.32 

theorem find_MorkTaxRate :
  (MorkTaxRate * MorkIncome + MindyTaxRate * MindyIncome) / (MorkIncome + MindyIncome) = combinedTaxRate →
  MorkTaxRate = 0.40 := sorry

end find_MorkTaxRate_l624_62487


namespace hoseoks_social_studies_score_l624_62443

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

end hoseoks_social_studies_score_l624_62443


namespace technicians_in_workshop_l624_62496

theorem technicians_in_workshop 
  (total_workers : ℕ) 
  (avg_salary_all : ℕ) 
  (avg_salary_tech : ℕ) 
  (avg_salary_rest : ℕ) 
  (total_salary : ℕ) 
  (T : ℕ) 
  (R : ℕ) 
  (h1 : total_workers = 14) 
  (h2 : avg_salary_all = 8000) 
  (h3 : avg_salary_tech = 10000) 
  (h4 : avg_salary_rest = 6000) 
  (h5 : total_salary = total_workers * avg_salary_all) 
  (h6 : T + R = 14)
  (h7 : total_salary = 112000) 
  (h8 : total_salary = avg_salary_tech * T + avg_salary_rest * R) :
  T = 7 := 
by {
  -- Proof goes here
  sorry
} 

end technicians_in_workshop_l624_62496


namespace bowling_ball_weight_l624_62485

theorem bowling_ball_weight (b k : ℕ) (h1 : 8 * b = 4 * k) (h2 : 3 * k = 84) : b = 14 := by
  sorry

end bowling_ball_weight_l624_62485


namespace hemisphere_surface_area_l624_62414

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (hπ : π = Real.pi) (h : π * r^2 = 3) :
    2 * π * r^2 + 3 = 9 :=
by
  sorry

end hemisphere_surface_area_l624_62414


namespace smallest_positive_period_intervals_of_monotonicity_max_min_values_l624_62413

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

-- Prove the smallest positive period
theorem smallest_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x := sorry

-- Prove the intervals of monotonicity
theorem intervals_of_monotonicity (k : ℤ) : 
  ∀ x y, (k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6) → 
         (k * Real.pi - Real.pi / 3 ≤ y ∧ y ≤ k * Real.pi + Real.pi / 6) → 
         (x < y → f x < f y) ∨ (y < x → f y < f x) := sorry

-- Prove the maximum and minimum values on [0, π/2]
theorem max_min_values : ∃ (max_val min_val : ℝ), max_val = 2 ∧ min_val = -1 ∧ 
  ∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x ≤ max_val ∧ f x ≥ min_val := sorry

end smallest_positive_period_intervals_of_monotonicity_max_min_values_l624_62413


namespace cost_of_4_stamps_l624_62416

theorem cost_of_4_stamps (cost_per_stamp : ℕ) (h : cost_per_stamp = 34) : 4 * cost_per_stamp = 136 :=
by
  sorry

end cost_of_4_stamps_l624_62416


namespace complement_U_M_correct_l624_62402

open Set

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {x | x^2 - 4 * x + 3 = 0}
def complement_U_M : Set ℕ := U \ M

theorem complement_U_M_correct : complement_U_M = {2, 4} :=
by
  -- Proof will be provided here
  sorry

end complement_U_M_correct_l624_62402


namespace nesbitts_inequality_l624_62425

theorem nesbitts_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c)) + (b / (c + a)) + (c / (a + b)) ≥ 3 / 2 :=
sorry

end nesbitts_inequality_l624_62425


namespace inequality_abc_sum_one_l624_62471

theorem inequality_abc_sum_one (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 1) :
  (a^2 + b^2 + c^2 + d) / (a + b + c)^3 +
  (b^2 + c^2 + d^2 + a) / (b + c + d)^3 +
  (c^2 + d^2 + a^2 + b) / (c + d + a)^3 +
  (d^2 + a^2 + b^2 + c) / (d + a + b)^3 > 4 := by
  sorry

end inequality_abc_sum_one_l624_62471


namespace questions_ratio_l624_62424

theorem questions_ratio (R A : ℕ) (H₁ : R + 6 + A = 24) :
  (R, 6, A) = (R, 6, A) :=
sorry

end questions_ratio_l624_62424


namespace boys_play_football_l624_62498

theorem boys_play_football (total_boys basketball_players neither_players both_players : ℕ)
    (h_total : total_boys = 22)
    (h_basketball : basketball_players = 13)
    (h_neither : neither_players = 3)
    (h_both : both_players = 18) : total_boys - neither_players - both_players + (both_players - basketball_players) = 19 :=
by
  sorry

end boys_play_football_l624_62498


namespace product_of_four_consecutive_integers_divisible_by_12_l624_62442

theorem product_of_four_consecutive_integers_divisible_by_12 (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3)) % 12 = 0 :=
sorry

end product_of_four_consecutive_integers_divisible_by_12_l624_62442


namespace find_p_q_sum_l624_62474

theorem find_p_q_sum (p q : ℝ) :
  (∀ x : ℝ, (x - 3) * (x - 5) = 0 → 3 * x ^ 2 - p * x + q = 0) →
  p = 24 ∧ q = 45 ∧ p + q = 69 :=
by
  intros h
  have h3 := h 3 (by ring)
  have h5 := h 5 (by ring)
  sorry

end find_p_q_sum_l624_62474


namespace ways_to_append_digit_divisible_by_3_l624_62486

-- Define a function that takes a digit and checks if it can make the number divisible by 3
def is_divisible_by_3 (n : ℕ) (d : ℕ) : Bool :=
  (n * 10 + d) % 3 == 0

-- Theorem stating that there are 4 ways to append a digit to make the number divisible by 3
theorem ways_to_append_digit_divisible_by_3 
  (n : ℕ) 
  (divisible_by_9_conditions : (n * 10 + 0) % 9 = 0 ∧ (n * 10 + 9) % 9 = 0) : 
  ∃ (ds : Finset ℕ), ds.card = 4 ∧ ∀ d ∈ ds, is_divisible_by_3 n d :=
  sorry

end ways_to_append_digit_divisible_by_3_l624_62486


namespace find_first_term_l624_62447

theorem find_first_term (S_n : ℕ → ℝ) (a d : ℝ) (n : ℕ) (h₁ : ∀ n > 0, S_n n = n * (2 * a + (n - 1) * d) / 2)
  (h₂ : d = 3) (h₃ : ∃ c, ∀ n > 0, S_n (3 * n) / S_n n = c) : a = 3 / 2 :=
by
  sorry

end find_first_term_l624_62447


namespace sufficient_but_not_necessary_condition_l624_62419

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := abs (x * (m * x + 2))

theorem sufficient_but_not_necessary_condition (m : ℝ) : 
  (∃ m0 : ℝ, m0 > 0 ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f m0 x1 ≤ f m0 x2)) ∧ 
  ¬ (∀ m : ℝ, (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f m x1 ≤ f m x2) → m > 0) :=
by sorry

end sufficient_but_not_necessary_condition_l624_62419


namespace simplify_expression_l624_62411

theorem simplify_expression (x : ℝ) (h : x = 1) : (x - 1)^2 + (x + 1) * (x - 1) - 2 * x^2 = -2 :=
by
  sorry

end simplify_expression_l624_62411


namespace inequality_div_half_l624_62410

theorem inequality_div_half (a b : ℝ) (h : a > b) : (a / 2 > b / 2) :=
sorry

end inequality_div_half_l624_62410


namespace roots_depend_on_k_l624_62418

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem roots_depend_on_k (k : ℝ) :
  let a := 1
  let b := -3
  let c := 2 - k
  discriminant a b c = 1 + 4 * k :=
by
  sorry

end roots_depend_on_k_l624_62418


namespace inequality_system_solution_range_l624_62452

theorem inequality_system_solution_range (x m : ℝ) :
  (∃ x : ℝ, (x + 1) / 2 < x / 3 + 1 ∧ x > 3 * m) → m < 1 :=
by
  sorry

end inequality_system_solution_range_l624_62452


namespace angle_ACB_is_25_l624_62491

theorem angle_ACB_is_25 (angle_ABD angle_BAC : ℝ) (is_supplementary : angle_ABD + (180 - angle_BAC) = 180) (angle_ABC_eq : angle_BAC = 95) (angle_ABD_eq : angle_ABD = 120) :
  180 - (angle_BAC + (180 - angle_ABD)) = 25 :=
by
  sorry

end angle_ACB_is_25_l624_62491


namespace find_a2_l624_62464

-- Define the geometric sequence and its properties
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Given conditions 
variables (a : ℕ → ℝ) (h_geom : is_geometric a)
variables (h_a1 : a 1 = 1/4)
variables (h_condition : a 3 * a 5 = 4 * (a 4 - 1))

-- The goal is to prove a 2 = 1/2
theorem find_a2 : a 2 = 1/2 :=
by
  sorry

end find_a2_l624_62464


namespace expression_eval_l624_62403

theorem expression_eval :
    (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * 
    (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) * 5040 = 
    (5^128 - 4^128) * 5040 := by
  sorry

end expression_eval_l624_62403


namespace mary_rental_hours_l624_62405

-- Definitions of the given conditions
def fixed_fee : ℝ := 17
def hourly_rate : ℝ := 7
def total_paid : ℝ := 80

-- Goal: Prove that the number of hours Mary paid for is 9
theorem mary_rental_hours : (total_paid - fixed_fee) / hourly_rate = 9 := 
by
  sorry

end mary_rental_hours_l624_62405


namespace find_W_l624_62438

noncomputable def volume_of_space (r_sphere r_cylinder h_cylinder : ℝ) : ℝ :=
  let V_sphere := (4 / 3) * Real.pi * r_sphere^3
  let V_cylinder := Real.pi * r_cylinder^2 * h_cylinder
  let V_cone := (1 / 3) * Real.pi * r_cylinder^2 * h_cylinder
  V_sphere - V_cylinder - V_cone

theorem find_W : volume_of_space 6 4 10 = (224 / 3) * Real.pi := by
  sorry

end find_W_l624_62438


namespace arithmetic_sequence_a8_l624_62482

variable (a : ℕ → ℝ)
variable (a2_eq : a 2 = 4)
variable (a6_eq : a 6 = 2)

theorem arithmetic_sequence_a8 :
  a 8 = 1 :=
sorry

end arithmetic_sequence_a8_l624_62482


namespace probability_of_prime_number_on_spinner_l624_62454

-- Definitions of conditions
def spinner_sections : List ℕ := [2, 3, 4, 5, 7, 9, 10, 11]
def total_sectors : ℕ := 8
def prime_count : ℕ := List.filter Nat.Prime spinner_sections |>.length

-- Statement of the theorem we want to prove
theorem probability_of_prime_number_on_spinner :
  (prime_count : ℚ) / total_sectors = 5 / 8 := by
  sorry

end probability_of_prime_number_on_spinner_l624_62454


namespace find_average_age_of_students_l624_62444

-- Given conditions
variables (n : ℕ) (T : ℕ) (A : ℕ)

-- 20 students in the class
def students : ℕ := 20

-- Teacher's age is 42 years
def teacher_age : ℕ := 42

-- When the teacher's age is included, the average age increases by 1
def average_age_increase (A : ℕ) := A + 1

-- Proof problem statement in Lean 4
theorem find_average_age_of_students (A : ℕ) :
  20 * A + 42 = 21 * (A + 1) → A = 21 :=
by
  -- Here should be the proof steps, added sorry to skip the proof
  sorry

end find_average_age_of_students_l624_62444


namespace find_a_l624_62470

theorem find_a (a : ℝ) (h : -2 * a + 1 = -1) : a = 1 :=
by sorry

end find_a_l624_62470


namespace find_ordered_pairs_l624_62480

theorem find_ordered_pairs (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : (a - b) ^ (a * b) = a ^ b * b ^ a) :
  (a, b) = (4, 2) := by
  sorry

end find_ordered_pairs_l624_62480


namespace number_of_elephants_l624_62406

theorem number_of_elephants (giraffes penguins total_animals elephants : ℕ)
  (h1 : giraffes = 5)
  (h2 : penguins = 2 * giraffes)
  (h3 : penguins = total_animals / 5)
  (h4 : elephants = total_animals * 4 / 100) :
  elephants = 2 := by
  -- The proof is omitted
  sorry

end number_of_elephants_l624_62406


namespace sum_integers_minus15_to_6_l624_62492

def sum_range (a b : ℤ) : ℤ :=
  (b - a + 1) * (a + b) / 2

theorem sum_integers_minus15_to_6 : sum_range (-15) (6) = -99 :=
  by
  -- Skipping the proof details
  sorry

end sum_integers_minus15_to_6_l624_62492


namespace min_expression_value_l624_62488

theorem min_expression_value (x y : ℝ) (hx : x > 2) (hy : y > 2) : 
  ∃ m : ℝ, (∀ x y : ℝ, x > 2 → y > 2 → (x^3 / (y - 2) + y^3 / (x - 2)) ≥ m) ∧ 
          (m = 64) :=
by
  sorry

end min_expression_value_l624_62488


namespace max_regular_hours_correct_l624_62462

-- Define the conditions
def regular_rate : ℝ := 16
def overtime_rate : ℝ := regular_rate + 0.75 * regular_rate
def total_hours_worked : ℝ := 57
def total_compensation : ℝ := 1116

-- Define the maximum regular hours per week
def max_regular_hours : ℝ := 40

-- Define the compensation equation
def compensation (H : ℝ) : ℝ :=
  regular_rate * H + overtime_rate * (total_hours_worked - H)

-- The theorem that needs to be proved
theorem max_regular_hours_correct :
  compensation max_regular_hours = total_compensation :=
by
  -- skolemize the proof
  sorry

end max_regular_hours_correct_l624_62462


namespace quadratic_expression_value_l624_62401

theorem quadratic_expression_value :
  ∀ x1 x2 : ℝ, (x1^2 - 4 * x1 - 2020 = 0) ∧ (x2^2 - 4 * x2 - 2020 = 0) →
  (x1^2 - 2 * x1 + 2 * x2 = 2028) :=
by
  intros x1 x2 h
  sorry

end quadratic_expression_value_l624_62401


namespace game_points_l624_62436

noncomputable def total_points (total_enemies : ℕ) (red_enemies : ℕ) (blue_enemies : ℕ) 
  (enemies_defeated : ℕ) (points_per_enemy : ℕ) (bonus_points : ℕ) 
  (hits_taken : ℕ) (points_lost_per_hit : ℕ) : ℕ :=
  (enemies_defeated * points_per_enemy + if enemies_defeated > 0 ∧ enemies_defeated < total_enemies then bonus_points else 0) - (hits_taken * points_lost_per_hit)

theorem game_points (h : total_points 6 3 3 4 3 5 2 2 = 13) : Prop := sorry

end game_points_l624_62436


namespace crows_and_trees_l624_62499

variable (x y : ℕ)

theorem crows_and_trees (h1 : x = 3 * y + 5) (h2 : x = 5 * (y - 1)) : 
  (x - 5) / 3 = y ∧ x / 5 = y - 1 :=
by
  sorry

end crows_and_trees_l624_62499


namespace total_number_of_students_l624_62445

namespace StudentRanking

def rank_from_right := 17
def rank_from_left := 5
def total_students (rank_from_right rank_from_left : ℕ) := rank_from_right + rank_from_left - 1

theorem total_number_of_students : total_students rank_from_right rank_from_left = 21 :=
by
  sorry

end StudentRanking

end total_number_of_students_l624_62445


namespace gcf_84_112_210_l624_62433

theorem gcf_84_112_210 : gcd (gcd 84 112) 210 = 14 := by sorry

end gcf_84_112_210_l624_62433


namespace calculation_of_cube_exponent_l624_62476

theorem calculation_of_cube_exponent (a : ℤ) : (-2 * a^3)^3 = -8 * a^9 := by
  sorry

end calculation_of_cube_exponent_l624_62476


namespace num_bases_ending_in_1_l624_62448

theorem num_bases_ending_in_1 : 
  (∃ bases : Finset ℕ, 
  ∀ b ∈ bases, 3 ≤ b ∧ b ≤ 10 ∧ (625 % b = 1) ∧ bases.card = 4) :=
sorry

end num_bases_ending_in_1_l624_62448


namespace combination_indices_l624_62468
open Nat

theorem combination_indices (x : ℕ) (h : choose 18 x = choose 18 (3 * x - 6)) : x = 3 ∨ x = 6 :=
by
  sorry

end combination_indices_l624_62468


namespace solve_inequality_l624_62441

theorem solve_inequality (x : ℝ) (h : x / 3 - 2 < 0) : x < 6 :=
sorry

end solve_inequality_l624_62441


namespace solve_trig_eq_l624_62446

theorem solve_trig_eq (x : ℝ) :
  (0.5 * (Real.cos (5 * x) + Real.cos (7 * x)) - Real.cos (2 * x) ^ 2 + Real.sin (3 * x) ^ 2 = 0) →
  (∃ k : ℤ, x = (Real.pi / 2) * (2 * k + 1) ∨ x = (2 * k * Real.pi / 11)) :=
sorry

end solve_trig_eq_l624_62446


namespace gcd_gx_x_multiple_of_18432_l624_62465

def g (x : ℕ) : ℕ := (3*x + 5) * (7*x + 2) * (13*x + 7) * (2*x + 10)

theorem gcd_gx_x_multiple_of_18432 (x : ℕ) (h : ∃ k : ℕ, x = 18432 * k) : Nat.gcd (g x) x = 28 :=
by
  sorry

end gcd_gx_x_multiple_of_18432_l624_62465


namespace Annika_three_times_Hans_in_future_l624_62440

theorem Annika_three_times_Hans_in_future
  (hans_age_now : Nat)
  (annika_age_now : Nat)
  (x : Nat)
  (hans_future_age : Nat)
  (annika_future_age : Nat)
  (H1 : hans_age_now = 8)
  (H2 : annika_age_now = 32)
  (H3 : hans_future_age = hans_age_now + x)
  (H4 : annika_future_age = annika_age_now + x)
  (H5 : annika_future_age = 3 * hans_future_age) :
  x = 4 := 
  by
  sorry

end Annika_three_times_Hans_in_future_l624_62440


namespace problem_l624_62495

def g (x : ℕ) : ℕ := x^2 + 1
def f (x : ℕ) : ℕ := 3 * x - 2

theorem problem : f (g 3) = 28 := by
  sorry

end problem_l624_62495
