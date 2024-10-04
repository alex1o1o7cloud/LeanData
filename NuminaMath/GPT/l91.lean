import Mathlib

namespace cos_inequality_y_zero_l91_91808

theorem cos_inequality_y_zero (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) (hy : 0 ≤ y ∧ y ≤ π / 2) :
  (∀ x ∈ Icc (0 : ℝ) (π / 2), cos (x + y) ≥ cos x * cos y) ↔ y = 0 := by
  sorry

end cos_inequality_y_zero_l91_91808


namespace num_of_loads_l91_91427

theorem num_of_loads (n : ℕ) (h1 : 7 * n = 42) : n = 6 :=
by
  sorry

end num_of_loads_l91_91427


namespace test_questions_l91_91611

theorem test_questions (x : ℕ) (h1 : x % 5 = 0) (h2 : 70 < 32 * 100 / x) (h3 : 32 * 100 / x < 77) : x = 45 := 
by sorry

end test_questions_l91_91611


namespace animal_count_l91_91479

theorem animal_count (dogs : ℕ) (cats : ℕ) (birds : ℕ) (fish : ℕ)
  (h1 : dogs = 6)
  (h2 : cats = dogs / 2)
  (h3 : birds = dogs * 2)
  (h4 : fish = dogs * 3) : 
  dogs + cats + birds + fish = 39 :=
by
  sorry

end animal_count_l91_91479


namespace even_and_nonneg_range_l91_91322

theorem even_and_nonneg_range : 
  (∀ x : ℝ, abs x = abs (-x) ∧ (abs x ≥ 0)) ∧ (∀ x : ℝ, x^2 + abs x = ( (-x)^2) + abs (-x) ∧ (x^2 + abs x ≥ 0)) := sorry

end even_and_nonneg_range_l91_91322


namespace basketball_game_half_points_l91_91937

noncomputable def eagles_geometric_sequence (a r : ℕ) (n : ℕ) : ℕ :=
  a * r ^ n

noncomputable def lions_arithmetic_sequence (b d : ℕ) (n : ℕ) : ℕ :=
  b + n * d

noncomputable def total_first_half_points (a r b d : ℕ) : ℕ :=
  eagles_geometric_sequence a r 0 + eagles_geometric_sequence a r 1 +
  lions_arithmetic_sequence b d 0 + lions_arithmetic_sequence b d 1

theorem basketball_game_half_points (a r b d : ℕ) (h1 : a + a * r = b + (b + d)) (h2 : a + a * r + a * r^2 + a * r^3 = b + (b + d) + (b + 2*d) + (b + 3*d)) :
  total_first_half_points a r b d = 8 :=
by sorry

end basketball_game_half_points_l91_91937


namespace total_handshakes_l91_91262

def total_people := 40
def group_x_people := 25
def group_x_known_others := 5
def group_y_people := 15
def handshakes_between_x_y := group_x_people * group_y_people
def handshakes_within_x := 25 * (25 - 1 - 5) / 2
def handshakes_within_y := (15 * (15 - 1)) / 2

theorem total_handshakes 
    (h1 : total_people = 40)
    (h2 : group_x_people = 25)
    (h3 : group_x_known_others = 5)
    (h4 : group_y_people = 15) :
    handshakes_between_x_y + handshakes_within_x + handshakes_within_y = 717 := 
by
  sorry

end total_handshakes_l91_91262


namespace smallest_area_of_right_triangle_l91_91100

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l91_91100


namespace sum_of_remainders_l91_91464

theorem sum_of_remainders {a b c d e : ℤ} (h1 : a % 13 = 3) (h2 : b % 13 = 5) (h3 : c % 13 = 7) (h4 : d % 13 = 9) (h5 : e % 13 = 11) : 
  ((a + b + c + d + e) % 13) = 9 :=
by
  sorry

end sum_of_remainders_l91_91464


namespace findNAndConstantTerm_l91_91244

def expansionHasFiveTerms (n : ℕ) : Prop :=
  (x + 2)^n has exactly 5 terms

def constantTerm (n : ℕ) : ℕ :=
  if 4 - r = 0 then (Nat.choose 4 4) * 2 ^ 4 else 0

theorem findNAndConstantTerm (n : ℕ) (h : expansionHasFiveTerms n) : n = 4 ∧ constantTerm n = 16 :=
by
  sorry

end findNAndConstantTerm_l91_91244


namespace find_smallest_n_l91_91364

theorem find_smallest_n (n : ℕ) : 
  (∃ n : ℕ, (n^2).digits.contains 7 ∧ ((n + 1)^2).digits.contains 7 ∧ (n + 2)!=n )

end find_smallest_n_l91_91364


namespace four_pq_plus_four_qp_l91_91524

theorem four_pq_plus_four_qp (p q : ℝ) (h : p / q - q / p = 21 / 10) : 
  4 * p / q + 4 * q / p = 16.8 :=
sorry

end four_pq_plus_four_qp_l91_91524


namespace total_stickers_at_end_of_week_l91_91869

-- Defining the initial and earned stickers as constants
def initial_stickers : ℕ := 39
def earned_stickers : ℕ := 22

-- Defining the goal as a proof statement
theorem total_stickers_at_end_of_week : initial_stickers + earned_stickers = 61 := 
by {
  sorry
}

end total_stickers_at_end_of_week_l91_91869


namespace find_starting_point_of_a_l91_91911

def point := ℝ × ℝ
def vector := ℝ × ℝ

def B : point := (1, 0)

def b : vector := (-3, -4)
def c : vector := (1, 1)

def a : vector := (3 * b.1 - 2 * c.1, 3 * b.2 - 2 * c.2)

theorem find_starting_point_of_a (hb : b = (-3, -4)) (hc : c = (1, 1)) (hB : B = (1, 0)) :
    let a := (3 * b.1 - 2 * c.1, 3 * b.2 - 2 * c.2)
    let start_A := (B.1 - a.1, B.2 - a.2)
    start_A = (12, 14) :=
by
  rw [hb, hc, hB]
  let a := (3 * (-3) - 2 * (1), 3 * (-4) - 2 * (1))
  let start_A := (1 - a.1, 0 - a.2)
  simp [a]
  sorry

end find_starting_point_of_a_l91_91911


namespace total_amount_correct_l91_91723

/-- Meghan has the following cash denominations: -/
def num_100_bills : ℕ := 2
def num_50_bills : ℕ := 5
def num_10_bills : ℕ := 10

/-- Value of each denomination: -/
def value_100_bill : ℕ := 100
def value_50_bill : ℕ := 50
def value_10_bill : ℕ := 10

/-- Meghan's total amount of money: -/
def total_amount : ℕ :=
  (num_100_bills * value_100_bill) +
  (num_50_bills * value_50_bill) +
  (num_10_bills * value_10_bill)

/-- The proof: -/
theorem total_amount_correct : total_amount = 550 :=
by
  -- sorry for now
  sorry

end total_amount_correct_l91_91723


namespace consistent_scale_l91_91281

-- Conditions definitions

def dist_gardensquare_newtonsville : ℕ := 3  -- in inches
def dist_newtonsville_madison : ℕ := 4  -- in inches
def speed_gardensquare_newtonsville : ℕ := 50  -- mph
def time_gardensquare_newtonsville : ℕ := 2  -- hours
def speed_newtonsville_madison : ℕ := 60  -- mph
def time_newtonsville_madison : ℕ := 3  -- hours

-- Actual distances calculated
def actual_distance_gardensquare_newtonsville : ℕ := speed_gardensquare_newtonsville * time_gardensquare_newtonsville
def actual_distance_newtonsville_madison : ℕ := speed_newtonsville_madison * time_newtonsville_madison

-- Prove the scale is consistent across the map
theorem consistent_scale :
  actual_distance_gardensquare_newtonsville / dist_gardensquare_newtonsville =
  actual_distance_newtonsville_madison / dist_newtonsville_madison :=
by
  sorry

end consistent_scale_l91_91281


namespace min_a_b_l91_91813

theorem min_a_b : 
  (∀ x : ℝ, 3 * a * (Real.sin x + Real.cos x) + 2 * b * Real.sin (2 * x) ≤ 3) →
  a + b = -2 →
  a = -4 / 5 :=
by
  sorry

end min_a_b_l91_91813


namespace sum_of_first_41_terms_is_94_l91_91843

def equal_product_sequence (a : ℕ → ℕ) (k : ℕ) : Prop := 
∀ (n : ℕ), a (n+1) * a (n+2) * a (n+3) = k

theorem sum_of_first_41_terms_is_94
  (a : ℕ → ℕ)
  (h1 : equal_product_sequence a 8)
  (h2 : a 1 = 1)
  (h3 : a 2 = 2) :
  (Finset.range 41).sum a = 94 :=
by
  sorry

end sum_of_first_41_terms_is_94_l91_91843


namespace additional_savings_if_purchase_together_l91_91641

theorem additional_savings_if_purchase_together :
  let price_per_window := 100
  let windows_each_offer := 4
  let free_each_offer := 1
  let dave_windows := 7
  let doug_windows := 8

  let cost_without_offer (windows : Nat) := windows * price_per_window
  let cost_with_offer (windows : Nat) := 
    if windows % (windows_each_offer + free_each_offer) = 0 then
      (windows / (windows_each_offer + free_each_offer)) * windows_each_offer * price_per_window
    else
      (windows / (windows_each_offer + free_each_offer)) * windows_each_offer * price_per_window 
      + (windows % (windows_each_offer + free_each_offer)) * price_per_window

  (cost_without_offer (dave_windows + doug_windows) 
  - cost_with_offer (dave_windows + doug_windows)) 
  - ((cost_without_offer dave_windows - cost_with_offer dave_windows)
  + (cost_without_offer doug_windows - cost_with_offer doug_windows)) = price_per_window := 
  sorry

end additional_savings_if_purchase_together_l91_91641


namespace probability_same_color_correct_l91_91913

def number_of_balls : ℕ := 16
def green_balls : ℕ := 8
def red_balls : ℕ := 5
def blue_balls : ℕ := 3

def probability_two_balls_same_color : ℚ :=
  ((green_balls / number_of_balls)^2 + (red_balls / number_of_balls)^2 + (blue_balls / number_of_balls)^2)

theorem probability_same_color_correct :
  probability_two_balls_same_color = 49 / 128 := sorry

end probability_same_color_correct_l91_91913


namespace can_lids_per_box_l91_91644

/-- Aaron initially has 14 can lids, and after adding can lids from 3 boxes,
he has a total of 53 can lids. How many can lids are in each box? -/
theorem can_lids_per_box (initial : ℕ) (total : ℕ) (boxes : ℕ) (h₀ : initial = 14) (h₁ : total = 53) (h₂ : boxes = 3) :
  (total - initial) / boxes = 13 :=
by
  sorry

end can_lids_per_box_l91_91644


namespace difference_of_squares_l91_91449

theorem difference_of_squares (x y : ℕ) (h₁ : x + y = 22) (h₂ : x * y = 120) (h₃ : x > y) : 
  x^2 - y^2 = 44 :=
sorry

end difference_of_squares_l91_91449


namespace tan_alpha_eq_neg_one_third_l91_91398

open Real

theorem tan_alpha_eq_neg_one_third
  (h : cos (π / 4 - α) / cos (π / 4 + α) = 1 / 2) :
  tan α = -1 / 3 :=
sorry

end tan_alpha_eq_neg_one_third_l91_91398


namespace graph_single_point_l91_91881

theorem graph_single_point (c : ℝ) : 
  (∃ x y : ℝ, ∀ (x' y' : ℝ), 4 * x'^2 + y'^2 + 16 * x' - 6 * y' + c = 0 → (x' = x ∧ y' = y)) → c = 7 := 
by
  sorry

end graph_single_point_l91_91881


namespace maximize_a_minus_b_plus_c_l91_91388

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.cos x + b * Real.cos (2 * x) + c * Real.cos (3 * x)

theorem maximize_a_minus_b_plus_c
  {a b c : ℝ}
  (h : ∀ x : ℝ, f a b c x ≥ -1) :
  a - b + c ≤ 1 :=
sorry

end maximize_a_minus_b_plus_c_l91_91388


namespace range_of_x_l91_91826

noncomputable def f (x : ℝ) : ℝ := 3 * x + Real.sin x

theorem range_of_x (x : ℝ) (h₀ : -1 < x ∧ x < 1) (h₁ : f 0 = 0) (h₂ : f (1 - x) + f (1 - x^2) < 0) :
  1 < x ∧ x < Real.sqrt 2 :=
by
  sorry

end range_of_x_l91_91826


namespace sequence_square_terms_l91_91687

theorem sequence_square_terms (k : ℤ) (y : ℕ → ℤ) 
  (h1 : y 1 = 1)
  (h2 : y 2 = 1)
  (h3 : ∀ n ≥ 1, y (n + 2) = (4 * k - 5) * y (n + 1) - y n + 4 - 2 * k) :
  (∀ n, ∃ m : ℤ, y n = m ^ 2) ↔ k = 3 :=
by sorry

end sequence_square_terms_l91_91687


namespace max_cookie_price_l91_91714

theorem max_cookie_price (k p : ℕ) :
  8 * k + 3 * p < 200 →
  4 * k + 5 * p > 150 →
  k ≤ 19 :=
sorry

end max_cookie_price_l91_91714


namespace hedge_cost_and_blocks_l91_91023

-- Define the costs of each type of block
def costA : Nat := 2
def costB : Nat := 3
def costC : Nat := 4

-- Define the number of each type of block per section
def blocksPerSectionA : Nat := 20
def blocksPerSectionB : Nat := 10
def blocksPerSectionC : Nat := 5

-- Define the number of sections
def sections : Nat := 8

-- Define the total cost calculation
def totalCost : Nat := sections * (blocksPerSectionA * costA + blocksPerSectionB * costB + blocksPerSectionC * costC)

-- Define the total number of each type of block used
def totalBlocksA : Nat := sections * blocksPerSectionA
def totalBlocksB : Nat := sections * blocksPerSectionB
def totalBlocksC : Nat := sections * blocksPerSectionC

-- State the theorem
theorem hedge_cost_and_blocks :
  totalCost = 720 ∧ totalBlocksA = 160 ∧ totalBlocksB = 80 ∧ totalBlocksC = 40 := by
  sorry

end hedge_cost_and_blocks_l91_91023


namespace function_value_range_l91_91450

noncomputable def f (x : ℝ) : ℝ := 9^x - 3^(x+1) + 2

theorem function_value_range :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → -1/4 ≤ f x ∧ f x ≤ 2 :=
by
  sorry

end function_value_range_l91_91450


namespace poem_mode_median_l91_91314

def poem_counts : List (ℕ × ℕ) := [(4, 3), (5, 4), (6, 4), (7, 5), (8, 7), (9, 5), (10, 1), (11, 1)]

def mode (counts : List (ℕ × ℕ)) : ℕ :=
  counts.foldr (λ p acc, if p.2 > acc.2 then p else acc) (0, 0) |>.fst

def median (counts : List (ℕ × ℕ)) : ℕ :=
  let sortedData := counts.flatMap (λ ⟨poem, num⟩, List.replicate num poem)
  let mid1 := sortedData.nth ((sortedData.length / 2) - 1)
  let mid2 := sortedData.nth (sortedData.length / 2)
  match mid1, mid2 with
  | some x, some y => (x + y) / 2
  | _, _ => 0

theorem poem_mode_median : mode poem_counts = 8 ∧ median poem_counts = 7 := by
  sorry

end poem_mode_median_l91_91314


namespace quadratic_two_distinct_real_roots_l91_91231

theorem quadratic_two_distinct_real_roots (k : ℝ) : ∃ x : ℝ, x^2 + 2 * x - k = 0 ∧ 
  (∀ x1 x2: ℝ, x1 ≠ x2 → x1^2 + 2 * x1 - k = 0 ∧ x2^2 + 2 * x2 - k = 0) ↔ k > -1 :=
by
  sorry

end quadratic_two_distinct_real_roots_l91_91231


namespace jessica_older_than_claire_l91_91984

-- Define the current age of Claire
def claire_current_age := 20 - 2

-- Define the current age of Jessica
def jessica_current_age := 24

-- Prove that Jessica is 6 years older than Claire
theorem jessica_older_than_claire : jessica_current_age - claire_current_age = 6 :=
by
  -- Definitions of the ages
  let claire_current_age := 18
  let jessica_current_age := 24

  -- Prove the age difference
  sorry

end jessica_older_than_claire_l91_91984


namespace circle_center_l91_91007

theorem circle_center (x y : ℝ) : (x - 2)^2 + (y + 1)^2 = 3 → (2, -1) = (2, -1) :=
by
  intro h
  -- Proof omitted
  sorry

end circle_center_l91_91007


namespace smallest_area_correct_l91_91151

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l91_91151


namespace sam_puppies_count_l91_91875

variable (initial_puppies : ℝ) (given_away_puppies : ℝ)

theorem sam_puppies_count (h1 : initial_puppies = 6.0) 
                          (h2 : given_away_puppies = 2.0) : 
                          initial_puppies - given_away_puppies = 4.0 :=
by simp [h1, h2]; sorry

end sam_puppies_count_l91_91875


namespace remaining_grandchild_share_l91_91255

theorem remaining_grandchild_share 
  (total : ℕ) 
  (half_share : ℕ) 
  (remaining : ℕ) 
  (n : ℕ) 
  (total_eq : total = 124600)
  (half_share_eq : half_share = total / 2)
  (remaining_eq : remaining = total - half_share)
  (n_eq : n = 10) 
  : remaining / n = 6230 := 
by sorry

end remaining_grandchild_share_l91_91255


namespace inequality_solution_l91_91056

theorem inequality_solution (x : ℝ) : 4 * x - 2 ≤ 3 * (x - 1) ↔ x ≤ -1 :=
by 
  sorry

end inequality_solution_l91_91056


namespace problem1_problem2_l91_91657

variable (m n x y : ℝ)

theorem problem1 : 4 * m * n^3 * (2 * m^2 - (3 / 4) * m * n^2) = 8 * m^3 * n^3 - 3 * m^2 * n^5 := sorry

theorem problem2 : (x - 6 * y^2) * (3 * x^3 + y) = 3 * x^4 + x * y - 18 * x^3 * y^2 - 6 * y^3 := sorry

end problem1_problem2_l91_91657


namespace credit_extended_by_automobile_finance_companies_l91_91926

def percentage_of_automobile_installment_credit : ℝ := 0.36
def total_consumer_installment_credit : ℝ := 416.66667
def fraction_extended_by_finance_companies : ℝ := 0.5

theorem credit_extended_by_automobile_finance_companies :
  fraction_extended_by_finance_companies * (percentage_of_automobile_installment_credit * total_consumer_installment_credit) = 75 :=
by
  sorry

end credit_extended_by_automobile_finance_companies_l91_91926


namespace smallest_area_correct_l91_91153

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l91_91153


namespace integer_solutions_range_l91_91227

def operation (p q : ℝ) : ℝ := p + q - p * q

theorem integer_solutions_range (m : ℝ) :
  (∃ (x1 x2 : ℤ), (operation 2 x1 > 0) ∧ (operation x1 3 ≤ m) ∧ (operation 2 x2 > 0) ∧ (operation x2 3 ≤ m) ∧ (x1 ≠ x2)) ↔ (3 ≤ m ∧ m < 5) :=
by sorry

end integer_solutions_range_l91_91227


namespace total_matches_won_l91_91699

-- Define the conditions
def matches_in_first_period (total: ℕ) (win_rate: ℚ) : ℕ := (total * win_rate).toNat
def matches_in_second_period (total: ℕ) (win_rate: ℚ) : ℕ := (total * win_rate).toNat

-- The main proof statement that we need to prove
theorem total_matches_won (total1 total2 : ℕ) (win_rate1 win_rate2 : ℚ) :
  matches_in_first_period total1 win_rate1 + matches_in_second_period total2 win_rate2 = 110 :=
by
  sorry

end total_matches_won_l91_91699


namespace average_of_five_numbers_l91_91416

noncomputable def average_of_two (x1 x2 : ℝ) := (x1 + x2) / 2
noncomputable def average_of_three (x3 x4 x5 : ℝ) := (x3 + x4 + x5) / 3
noncomputable def average_of_five (x1 x2 x3 x4 x5 : ℝ) := (x1 + x2 + x3 + x4 + x5) / 5

theorem average_of_five_numbers (x1 x2 x3 x4 x5 : ℝ)
    (h1 : average_of_two x1 x2 = 12)
    (h2 : average_of_three x3 x4 x5 = 7) :
    average_of_five x1 x2 x3 x4 x5 = 9 := by
  sorry

end average_of_five_numbers_l91_91416


namespace smallest_right_triangle_area_l91_91108

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l91_91108


namespace petya_max_margin_l91_91556

def max_margin_votes (total_votes P1 P2 V1 V2: ℕ) : ℕ := P1 + P2 - (V1 + V2)

theorem petya_max_margin 
  (P1 P2 V1 V2: ℕ)
  (H1: P1 = V1 + 9) 
  (H2: V2 = P2 + 9) 
  (H3: P1 + P2 + V1 + V2 = 27) 
  (H_win: P1 + P2 > V1 + V2) : 
  max_margin_votes 27 P1 P2 V1 V2 = 9 :=
by
  sorry

end petya_max_margin_l91_91556


namespace solve_for_x_l91_91182

theorem solve_for_x (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 40) : x = 32 :=
by
  sorry

end solve_for_x_l91_91182


namespace cost_of_one_book_l91_91414

theorem cost_of_one_book (s b c : ℕ) (h1 : s > 18) (h2 : b > 1) (h3 : c > b) (h4 : s * b * c = 3203) (h5 : s ≤ 36) : c = 11 :=
by
  sorry

end cost_of_one_book_l91_91414


namespace division_by_fraction_example_problem_l91_91199

theorem division_by_fraction (a b : ℝ) (hb : b ≠ 0) : 
  a / (1 / b) = a * b :=
by
  -- Proof goes here
  sorry

theorem example_problem : 12 / (1 / 6) = 72 :=
by
  have h : 6 ≠ 0 := by norm_num
  rw division_by_fraction 12 6 h
  norm_num

end division_by_fraction_example_problem_l91_91199


namespace intersection_result_complement_union_result_l91_91721

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | x > 0}

theorem intersection_result : A ∩ B = {x | 0 < x ∧ x < 2} :=
by
  sorry

theorem complement_union_result : (compl B) ∪ A = {x | x < 2} :=
by
  sorry

end intersection_result_complement_union_result_l91_91721


namespace abs_diff_eq_1point5_l91_91273

theorem abs_diff_eq_1point5 (x y : ℝ)
    (hx : (⌊x⌋ : ℝ) + (y - ⌊y⌋) = 3.7)
    (hy : (x - ⌊x⌋) + (⌊y⌋ : ℝ) = 4.2) :
        |x - y| = 1.5 :=
by
  sorry

end abs_diff_eq_1point5_l91_91273


namespace solve_system_l91_91032

theorem solve_system (a b c : ℝ)
  (h1 : b + c = 10 - 4 * a)
  (h2 : a + c = -16 - 4 * b)
  (h3 : a + b = 9 - 4 * c) :
  2 * a + 2 * b + 2 * c = 1 :=
by
  sorry

end solve_system_l91_91032


namespace lcm_28_72_l91_91945

theorem lcm_28_72 : Nat.lcm 28 72 = 504 := by
  sorry

end lcm_28_72_l91_91945


namespace current_inventory_l91_91783

noncomputable def initial_books : ℕ := 743
noncomputable def fiction_books : ℕ := 520
noncomputable def nonfiction_books : ℕ := 123
noncomputable def children_books : ℕ := 100

noncomputable def saturday_instore_sales : ℕ := 37
noncomputable def saturday_fiction_sales : ℕ := 15
noncomputable def saturday_nonfiction_sales : ℕ := 12
noncomputable def saturday_children_sales : ℕ := 10
noncomputable def saturday_online_sales : ℕ := 128

noncomputable def sunday_instore_multiplier : ℕ := 2
noncomputable def sunday_online_addition : ℕ := 34

noncomputable def new_shipment : ℕ := 160

noncomputable def current_books := 
  initial_books 
  - (saturday_instore_sales + saturday_online_sales)
  - (sunday_instore_multiplier * saturday_instore_sales + saturday_online_sales + sunday_online_addition)
  + new_shipment

theorem current_inventory : current_books = 502 := by
  sorry

end current_inventory_l91_91783


namespace find_cans_lids_l91_91646

-- Define the given conditions
def total_lids (x : ℕ) : ℕ := 14 + 3 * x

-- Define the proof problem
theorem find_cans_lids (x : ℕ) (h : total_lids x = 53) : x = 13 :=
sorry

end find_cans_lids_l91_91646


namespace total_animals_for_sale_l91_91477

theorem total_animals_for_sale (dogs cats birds fish : ℕ) 
  (h1 : dogs = 6)
  (h2 : cats = dogs / 2)
  (h3 : birds = dogs * 2)
  (h4 : fish = dogs * 3) :
  dogs + cats + birds + fish = 39 := 
by
  sorry

end total_animals_for_sale_l91_91477


namespace cube_expression_l91_91259

theorem cube_expression (a : ℝ) (h : (a + 1/a)^2 = 5) : a^3 + 1/a^3 = 2 * Real.sqrt 5 :=
by
  sorry

end cube_expression_l91_91259


namespace solve_system_of_equations_l91_91878

theorem solve_system_of_equations :
  ∃ (x y : ℕ), (x + 2 * y = 5) ∧ (3 * x + y = 5) ∧ (x = 1) ∧ (y = 2) :=
by {
  sorry
}

end solve_system_of_equations_l91_91878


namespace gasoline_used_by_car_l91_91470

noncomputable def total_gasoline_used (gasoline_per_km : ℝ) (duration_hours : ℝ) (speed_kmh : ℝ) : ℝ :=
  gasoline_per_km * duration_hours * speed_kmh

theorem gasoline_used_by_car :
  total_gasoline_used 0.14 (2 + 0.5) 93.6 = 32.76 := sorry

end gasoline_used_by_car_l91_91470


namespace delta_zeta_finish_time_l91_91616

noncomputable def delta_epsilon_zeta_proof_problem (D E Z : ℝ) (k : ℝ) : Prop :=
  (1 / D + 1 / E + 1 / Z = 1 / (D - 4)) ∧
  (1 / D + 1 / E + 1 / Z = 1 / (E - 3.5)) ∧
  (1 / E + 1 / Z = 2 / E) → 
  k = 2

-- Now we prepare the theorem statement
theorem delta_zeta_finish_time (D E Z k : ℝ) (h1 : 1 / D + 1 / E + 1 / Z = 1 / (D - 4))
                                (h2 : 1 / D + 1 / E + 1 / Z = 1 / (E - 3.5))
                                (h3 : 1 / E + 1 / Z = 2 / E) 
                                (h4 : E = 6) :
  k = 2 := 
sorry

end delta_zeta_finish_time_l91_91616


namespace smallest_area_of_right_triangle_l91_91162

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l91_91162


namespace smallest_area_right_triangle_l91_91125

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l91_91125


namespace total_unbroken_seashells_l91_91282

/-
Given:
On the first day, Tom found 7 seashells but 4 were broken.
On the second day, he found 12 seashells but 5 were broken.
On the third day, he found 15 seashells but 8 were broken.

We need to prove that Tom found 17 unbroken seashells in total over the three days.
-/

def first_day_total := 7
def first_day_broken := 4
def first_day_unbroken := first_day_total - first_day_broken

def second_day_total := 12
def second_day_broken := 5
def second_day_unbroken := second_day_total - second_day_broken

def third_day_total := 15
def third_day_broken := 8
def third_day_unbroken := third_day_total - third_day_broken

def total_unbroken := first_day_unbroken + second_day_unbroken + third_day_unbroken

theorem total_unbroken_seashells : total_unbroken = 17 := by
  sorry

end total_unbroken_seashells_l91_91282


namespace line_intersects_y_axis_l91_91475

-- Define the points
def P1 : ℝ × ℝ := (3, 18)
def P2 : ℝ × ℝ := (-9, -6)

-- State that the line passing through P1 and P2 intersects the y-axis at (0, 12)
theorem line_intersects_y_axis :
  ∃ y : ℝ, (∃ m b : ℝ, ∀ x : ℝ, y = m * x + b ∧ (m = (P2.2 - P1.2) / (P2.1 - P1.1)) ∧ (P1.2 = m * P1.1 + b) ∧ (x = 0) ∧ y = 12) :=
sorry

end line_intersects_y_axis_l91_91475


namespace inequality_condition_l91_91864

theorem inequality_condition 
  (a b c : ℝ) : 
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ (c > Real.sqrt (a^2 + b^2)) := 
sorry

end inequality_condition_l91_91864


namespace cost_of_set_of_paints_l91_91034

def classes : ℕ := 6
def folders_per_class : ℕ := 1
def pencils_per_class : ℕ := 3
def erasers_per_6_pencils : ℕ := 1
def folder_cost : ℕ := 6
def pencil_cost : ℕ := 2
def eraser_cost : ℕ := 1
def total_spent : ℕ := 80

theorem cost_of_set_of_paints : 
  let total_folder_cost := classes * folders_per_class * folder_cost,
      total_pencil_cost := classes * pencils_per_class * pencil_cost,
      total_pencils := classes * pencils_per_class,
      total_erasers := total_pencils / 6 * erasers_per_6_pencils,
      total_eraser_cost := total_erasers * eraser_cost,
      total_supplies_cost := total_folder_cost + total_pencil_cost + total_eraser_cost
  in total_spent - total_supplies_cost = 5 :=
by 
  sorry

end cost_of_set_of_paints_l91_91034


namespace map_to_actual_distance_ratio_l91_91437

def distance_in_meters : ℝ := 250
def distance_on_map_cm : ℝ := 5
def cm_per_meter : ℝ := 100

theorem map_to_actual_distance_ratio :
  distance_on_map_cm / (distance_in_meters * cm_per_meter) = 1 / 5000 :=
by
  sorry

end map_to_actual_distance_ratio_l91_91437


namespace polynomial_expansion_l91_91404

theorem polynomial_expansion :
  ∃ A B C D : ℝ, (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) 
  ∧ (A + B + C + D = 36) :=
by {
  sorry
}

end polynomial_expansion_l91_91404


namespace intersection_complement_l91_91829

open Set Real

noncomputable def U : Set ℝ := univ
noncomputable def A : Set ℝ := { y | y ≥ 0 }
noncomputable def B : Set ℝ := { y | y ≥ 1 }

theorem intersection_complement :
  A ∩ (U \ B) = Ico 0 1 :=
by
  sorry

end intersection_complement_l91_91829


namespace new_students_admitted_l91_91615

theorem new_students_admitted (orig_students : ℕ := 35) (increase_cost : ℕ := 42) (orig_expense : ℕ := 400) (dim_avg_expense : ℤ := 1) :
  ∃ (x : ℕ), x = 7 :=
by
  sorry

end new_students_admitted_l91_91615


namespace didi_total_fund_l91_91068

-- Define the conditions
def cakes : ℕ := 10
def slices_per_cake : ℕ := 8
def price_per_slice : ℕ := 1
def first_business_owner_donation_per_slice : ℚ := 0.5
def second_business_owner_donation_per_slice : ℚ := 0.25

-- Define the proof problem statement
theorem didi_total_fund (h1 : cakes * slices_per_cake = 80)
    (h2 : (80 : ℕ) * price_per_slice = 80)
    (h3 : (80 : ℕ) * first_business_owner_donation_per_slice = 40)
    (h4 : (80 : ℕ) * second_business_owner_donation_per_slice = 20) : 
    (80 : ℕ) + 40 + 20 = 140 := by
  -- The proof itself will be constructed here
  sorry

end didi_total_fund_l91_91068


namespace cube_volume_l91_91600

theorem cube_volume (A V : ℝ) (h : A = 16) : V = 64 :=
by
  -- Here, we would provide the proof, but for now, we end with sorry
  sorry

end cube_volume_l91_91600


namespace total_matches_won_l91_91700

-- Condition definitions
def matches1 := 100
def win_percentage1 := 0.5
def matches2 := 100
def win_percentage2 := 0.6

-- Theorem statement
theorem total_matches_won : matches1 * win_percentage1 + matches2 * win_percentage2 = 110 :=
by
  sorry

end total_matches_won_l91_91700


namespace number_exceeds_its_part_by_20_l91_91784

theorem number_exceeds_its_part_by_20 (x : ℝ) (h : x = (3/8) * x + 20) : x = 32 :=
sorry

end number_exceeds_its_part_by_20_l91_91784


namespace range_of_a_l91_91513

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 2 then |x - 2 * a| else x + 1 / (x - 2) + a

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, f a 2 ≤ f a x) : 1 ≤ a ∧ a ≤ 6 := 
sorry

end range_of_a_l91_91513


namespace percentage_increase_is_20_l91_91982

-- Defining the original cost and new cost
def original_cost := 200
def new_total_cost := 480

-- Doubling the capacity means doubling the original cost
def doubled_old_cost := 2 * original_cost

-- The increase in cost
def increase_cost := new_total_cost - doubled_old_cost

-- The percentage increase in cost
def percentage_increase := (increase_cost / doubled_old_cost) * 100

-- The theorem we need to prove
theorem percentage_increase_is_20 : percentage_increase = 20 :=
  by
  sorry

end percentage_increase_is_20_l91_91982


namespace product_of_B_coordinates_l91_91002

theorem product_of_B_coordinates :
  (∃ (x y : ℝ), (1 / 3 * x + 2 / 3 * 4 = 1 ∧ 1 / 3 * y + 2 / 3 * 2 = 7) ∧ x * y = -85) :=
by
  sorry

end product_of_B_coordinates_l91_91002


namespace complement_complement_l91_91691

theorem complement_complement (alpha : ℝ) (h : alpha = 35) : (90 - (90 - alpha)) = 35 := by
  -- proof goes here, but we write sorry to skip it
  sorry

end complement_complement_l91_91691


namespace area_of_lawn_l91_91919

theorem area_of_lawn 
  (park_length : ℝ) (park_width : ℝ) (road_width : ℝ) 
  (H1 : park_length = 60) (H2 : park_width = 40) (H3 : road_width = 3) : 
  (park_length * park_width - (park_length * road_width + park_width * road_width - road_width ^ 2)) = 2109 := 
by
  sorry

end area_of_lawn_l91_91919


namespace evaporate_water_l91_91531

theorem evaporate_water (M : ℝ) (W_i W_f x : ℝ) (d : ℝ)
  (h_initial_mass : M = 500)
  (h_initial_water_content : W_i = 0.85 * M)
  (h_final_water_content : W_f = 0.75 * (M - x))
  (h_desired_fraction : d = 0.75) :
  x = 200 := 
  sorry

end evaporate_water_l91_91531


namespace correct_conclusions_l91_91006

noncomputable def quadratic_solution_set (a b c : ℝ) : Prop :=
  ∀ x : ℝ, (-1 / 2 < x ∧ x < 3) ↔ (a * x^2 + b * x + c > 0)

theorem correct_conclusions (a b c : ℝ) (h : quadratic_solution_set a b c) : c > 0 ∧ 4 * a + 2 * b + c > 0 :=
  sorry

end correct_conclusions_l91_91006


namespace age_of_fourth_child_l91_91854

theorem age_of_fourth_child (c1 c2 c3 c4 : ℕ) (h1 : c1 = 15)
  (h2 : c2 = c1 - 1) (h3 : c3 = c2 - 4)
  (h4 : c4 = c3 - 2) : c4 = 8 :=
by
  sorry

end age_of_fourth_child_l91_91854


namespace arithmetic_sequence_Sn_l91_91303

noncomputable def S (n : ℕ) : ℕ := sorry -- S is the sequence function

theorem arithmetic_sequence_Sn {n : ℕ} (h1 : S n = 2) (h2 : S (3 * n) = 18) : S (4 * n) = 26 :=
  sorry

end arithmetic_sequence_Sn_l91_91303


namespace intersecting_circles_range_of_m_l91_91932

theorem intersecting_circles_range_of_m
  (x y m : ℝ)
  (C₁_eq : x^2 + y^2 - 2 * m * x + m^2 - 4 = 0)
  (C₂_eq : x^2 + y^2 + 2 * x - 4 * m * y + 4 * m^2 - 8 = 0)
  (intersect : ∃ x y : ℝ, (x^2 + y^2 - 2 * m * x + m^2 - 4 = 0) ∧ (x^2 + y^2 + 2 * x - 4 * m * y + 4 * m^2 - 8 = 0))
  : m ∈ Set.Ioo (-12/5) (-2/5) ∪ Set.Ioo (3/5) 2 := 
sorry

end intersecting_circles_range_of_m_l91_91932


namespace xy_computation_l91_91749

theorem xy_computation (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : 
  x * y = 21 := by
  sorry

end xy_computation_l91_91749


namespace smallest_right_triangle_area_l91_91119

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l91_91119


namespace find_M_for_same_asymptotes_l91_91605

theorem find_M_for_same_asymptotes :
  ∃ M : ℝ, ∀ x y : ℝ,
    (x^2 / 16 - y^2 / 25 = 1) →
    (y^2 / 50 - x^2 / M = 1) →
    (∀ x : ℝ, ∃ k : ℝ, y = k * x ↔ k = 5 / 4) →
    M = 32 :=
by
  sorry

end find_M_for_same_asymptotes_l91_91605


namespace smallest_right_triangle_area_l91_91118

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l91_91118


namespace smallest_area_right_triangle_l91_91137

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l91_91137


namespace division_by_repeating_decimal_l91_91073

-- Define the repeating decimal as a fraction
def repeating_decimal := 4 / 9

-- Prove the main theorem
theorem division_by_repeating_decimal : 8 / repeating_decimal = 18 :=
by
  -- lean implementation steps
  sorry

end division_by_repeating_decimal_l91_91073


namespace quadratic_eq_of_sum_and_product_l91_91245

theorem quadratic_eq_of_sum_and_product (a b c : ℝ) (h_sum : -b / a = 4) (h_product : c / a = 3) :
    ∀ (x : ℝ), a * x^2 + b * x + c = a * x^2 - 4 * a * x + 3 * a :=
by
  sorry

end quadratic_eq_of_sum_and_product_l91_91245


namespace toy_factory_days_per_week_l91_91776

theorem toy_factory_days_per_week (toys_per_week : ℕ) (toys_per_day : ℕ) (h₁ : toys_per_week = 4560) (h₂ : toys_per_day = 1140) : toys_per_week / toys_per_day = 4 := 
by {
  -- Proof to be provided
  sorry
}

end toy_factory_days_per_week_l91_91776


namespace unique_point_on_circle_conditions_l91_91237

noncomputable def point : Type := ℝ × ℝ

-- Define points A and B
def A : point := (-1, 4)
def B : point := (2, 1)

def PA_squared (P : point) : ℝ :=
  let (x, y) := P
  (x + 1) ^ 2 + (y - 4) ^ 2

def PB_squared (P : point) : ℝ :=
  let (x, y) := P
  (x - 2) ^ 2 + (y - 1) ^ 2

-- Define circle C
def on_circle (a : ℝ) (P : point) : Prop :=
  let (x, y) := P
  (x - a) ^ 2 + (y - 2) ^ 2 = 16

-- Define the condition PA² + 2PB² = 24
def condition (P : point) : Prop :=
  PA_squared P + 2 * PB_squared P = 24

-- The main theorem stating the possible values of a
theorem unique_point_on_circle_conditions :
  ∃ (a : ℝ), ∀ (P : point), on_circle a P → condition P → (a = -1 ∨ a = 3) :=
sorry

end unique_point_on_circle_conditions_l91_91237


namespace find_smallest_n_l91_91363

theorem find_smallest_n (n : ℕ) : 
  (∃ n : ℕ, (n^2).digits.contains 7 ∧ ((n + 1)^2).digits.contains 7 ∧ (n + 2)!=n )

end find_smallest_n_l91_91363


namespace algebraic_expression_value_l91_91965

theorem algebraic_expression_value (x : ℝ) (h : x^2 - 2*x - 2 = 0) : 3*x^2 - 6*x + 9 = 15 := by
  sorry

end algebraic_expression_value_l91_91965


namespace odd_events_probability_l91_91040

open ProbabilityTheory

variables {Ω : Type*} {n : ℕ}

-- Define our probability space and events
variable [ProbabilitySpace Ω]

-- Assuming that A_i are events, and P(A_i) = 1/(2 * i^2) for i = 2, 3, ..., n.
def A (i : ℕ) : Event Ω := sorry

-- Probability that A_i occurs
axiom prob_A (i : ℕ) (h2 : 2 ≤ i) : P (A i) = 1 / (2 * i^2)

-- Given conditions: Independent events
axiom A_is_independent : IndepEvents (λ i, A i) {i | 2 ≤ i ∧ i ≤ n}

-- The theorem we want to prove
theorem odd_events_probability (h1 : 2 ≤ n) : 
  (P (∑ i in Finset.range n, if nat.even i then 1 else 0 = 1)) = (n-1) / (4 * n) := 
sorry

end odd_events_probability_l91_91040


namespace cups_of_rice_in_afternoon_l91_91037

-- Definitions for conditions
def morning_cups : ℕ := 3
def evening_cups : ℕ := 5
def fat_per_cup : ℕ := 10
def weekly_total_fat : ℕ := 700

-- Theorem statement
theorem cups_of_rice_in_afternoon (morning_cups evening_cups fat_per_cup weekly_total_fat : ℕ) :
  (weekly_total_fat - (morning_cups + evening_cups) * fat_per_cup * 7) / fat_per_cup = 14 :=
by
  sorry

end cups_of_rice_in_afternoon_l91_91037


namespace find_angle_A_max_perimeter_incircle_l91_91243

-- Definition of the triangle and the conditions
variables {A B C : Real} {a b c : Real} 

-- The conditions given in the problem
def triangle_conditions (a b c A B C : Real) : Prop :=
  (b + c = a * (Real.cos C + Real.sqrt 3 * Real.sin C)) ∧
  A + B + C = Real.pi

-- Part 1: Prove the value of angle A
theorem find_angle_A (a b c A B C : Real) 
(h : triangle_conditions a b c A B C) : 
A = Real.pi / 3 := sorry

-- Part 2: Prove the maximum perimeter of the incircle when a=2
theorem max_perimeter_incircle (b c A B C : Real) 
(h : triangle_conditions 2 b c A B C) : 
2 * Real.pi * (Real.sqrt 3 / 6 * (b + c - 2)) ≤ (2 * Real.sqrt 3 / 3) * Real.pi := sorry

end find_angle_A_max_perimeter_incircle_l91_91243


namespace y_coordinate_of_C_l91_91726

theorem y_coordinate_of_C (h : ℝ) (H : ∀ (C : ℝ), C = h) :
  let A := (0, 0)
      B := (0, 5)
      C := (3, h)
      D := (6, 5)
      E := (6, 0)
  -- Assuming the area of the pentagon is 50
  let area_square_ABDE := 25
      area_triangle_BCD := 25
  -- Assuming the height of triangle BCD
  let height_triangle_BCD := h - 5
      base_triangle_BCD := 6
      area_BCD := (1/2) * base_triangle_BCD * height_triangle_BCD in
  area_square_ABDE + area_triangle_BCD = 50 →
  area_BCD = area_triangle_BCD →
  h = 40 / 3 :=
by intros h H A B C D E area_square_ABDE area_triangle_BCD height_triangle_BCD base_triangle_BCD area_BCD;
   sorry

end y_coordinate_of_C_l91_91726


namespace plan_A_is_cost_effective_l91_91805

-- Definitions of the costs considering the problem's conditions
def cost_plan_A (days_A : ℕ) (rate_A : ℕ) : ℕ := days_A * rate_A
def cost_plan_C (days_AB : ℕ) (rate_A : ℕ) (rate_B : ℕ) (remaining_B : ℕ) : ℕ :=
  (days_AB * (rate_A + rate_B)) + (remaining_B * rate_B)

-- Specification of the days and rates from the conditions
def days_A := 12
def rate_A := 10000
def rate_B := 6000
def days_AB := 3
def remaining_B := 13

-- Costs for each plan
def A_cost := cost_plan_A days_A rate_A
def C_cost := cost_plan_C days_AB rate_A rate_B remaining_B

-- Theorem stating that Plan A is more cost-effective
theorem plan_A_is_cost_effective : A_cost < C_cost := by
  unfold A_cost
  unfold C_cost
  sorry

end plan_A_is_cost_effective_l91_91805


namespace solve_abs_inequality_l91_91226

/-- Given the inequality 2 ≤ |x - 3| ≤ 8, we want to prove that the solution is [-5 ≤ x ≤ 1] ∪ [5 ≤ x ≤ 11] --/
theorem solve_abs_inequality (x : ℝ) :
  2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11) :=
sorry

end solve_abs_inequality_l91_91226


namespace solve_for_x_l91_91836

theorem solve_for_x (x y : ℝ) 
  (h1 : 3 * x - y = 7)
  (h2 : x + 3 * y = 7) :
  x = 2.8 :=
by
  sorry

end solve_for_x_l91_91836


namespace ball_count_difference_l91_91995

open Nat

theorem ball_count_difference :
  (total_balls = 145) →
  (soccer_balls = 20) →
  (basketballs > soccer_balls) →
  (tennis_balls = 2 * soccer_balls) →
  (baseballs = soccer_balls + 10) →
  (volleyballs = 30) →
  let accounted_balls := soccer_balls + tennis_balls + baseballs + volleyballs
  let basketballs := total_balls - accounted_balls
  (basketballs - soccer_balls = 5) :=
by
  intros
  let tennis_balls := 2 * soccer_balls
  let baseballs := soccer_balls + 10
  let accounted_balls := soccer_balls + tennis_balls + baseballs + volleyballs
  let basketballs := total_balls - accounted_balls
  exact sorry

end ball_count_difference_l91_91995


namespace smallest_area_of_right_triangle_l91_91087

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l91_91087


namespace option_D_correct_l91_91400

theorem option_D_correct (a b : ℝ) (h : a > b) : 3 * a > 3 * b :=
sorry

end option_D_correct_l91_91400


namespace total_cookies_l91_91928

variable (glenn_cookies : ℕ) (kenny_cookies : ℕ) (chris_cookies : ℕ)
hypothesis (h1 : glenn_cookies = 24)
hypothesis (h2 : glenn_cookies = 4 * kenny_cookies)
hypothesis (h3 : chris_cookies = kenny_cookies / 2)

theorem total_cookies : glenn_cookies + kenny_cookies + chris_cookies = 33 :=
by sorry

end total_cookies_l91_91928


namespace avg_median_of_subsets_of_range_l91_91026

open Finset

noncomputable def median {α : Type*} [LinearOrderedField α] (s : Finset α) : α :=
if h : s.card % 2 = 0 then 
  (nth s (s.card / 2)).get h + (nth s (s.pred_card / 2)).get h 
else 
  nth s (s.card / 2).get sorry

theorem avg_median_of_subsets_of_range :
  let S := range 2009 in
  let m (A : Finset ℕ) := median A in
  ∑ A in (powerset S).filter (λ A, A.card > 0), m A / (2^2008 - 1) = 2009 / 2 :=
by sorry

end avg_median_of_subsets_of_range_l91_91026


namespace reflect_parabola_y_axis_l91_91285

theorem reflect_parabola_y_axis (x y : ℝ) :
  (y = 2 * (x - 1)^2 - 4) → (y = 2 * (-x - 1)^2 - 4) :=
sorry

end reflect_parabola_y_axis_l91_91285


namespace hem_dress_time_l91_91422

theorem hem_dress_time
  (hem_length_feet : ℕ)
  (stitch_length_inches : ℝ)
  (stitches_per_minute : ℕ)
  (hem_length_inches : ℝ)
  (total_stitches : ℕ)
  (time_minutes : ℝ)
  (h1 : hem_length_feet = 3)
  (h2 : stitch_length_inches = 1 / 4)
  (h3 : stitches_per_minute = 24)
  (h4 : hem_length_inches = 12 * hem_length_feet)
  (h5 : total_stitches = hem_length_inches / stitch_length_inches)
  (h6 : time_minutes = total_stitches / stitches_per_minute) :
  time_minutes = 6 := 
sorry

end hem_dress_time_l91_91422


namespace smallest_n_squared_contains_7_l91_91367

-- Lean statement
theorem smallest_n_squared_contains_7 :
  ∃ n : ℕ, (n^2).toString.contains '7' ∧ ((n+1)^2).toString.contains '7' ∧
  ∀ m : ℕ, ((m < n) → ¬(m^2).toString.contains '7' ∨ ¬((m+1)^2).toString.contains '7') :=
begin
  sorry
end

end smallest_n_squared_contains_7_l91_91367


namespace Zhang_Laoshi_pens_l91_91882

theorem Zhang_Laoshi_pens (x : ℕ) (original_price new_price : ℝ)
  (discount : new_price = 0.75 * original_price)
  (more_pens : x * original_price = (x + 25) * new_price) :
  x = 75 :=
by
  sorry

end Zhang_Laoshi_pens_l91_91882


namespace max_value_4x_plus_3y_l91_91004

theorem max_value_4x_plus_3y :
  ∃ x y : ℝ, (x^2 + y^2 = 16 * x + 8 * y + 8) ∧ (∀ w, w = 4 * x + 3 * y → w ≤ 64) ∧ ∃ x y, 4 * x + 3 * y = 64 :=
sorry

end max_value_4x_plus_3y_l91_91004


namespace valid_four_digit_number_count_l91_91394

theorem valid_four_digit_number_count : 
  let first_digit_choices := 6 
  let last_digit_choices := 10 
  let middle_digits_valid_pairs := 9 * 9 - 18
  (first_digit_choices * middle_digits_valid_pairs * last_digit_choices = 3780) := by
  sorry

end valid_four_digit_number_count_l91_91394


namespace james_total_fish_catch_l91_91575

-- Definitions based on conditions
def poundsOfTrout : ℕ := 200
def poundsOfSalmon : ℕ := Nat.floor (1.5 * poundsOfTrout)
def poundsOfTuna : ℕ := 2 * poundsOfTrout

-- Proof statement
theorem james_total_fish_catch : poundsOfTrout + poundsOfSalmon + poundsOfTuna = 900 := by
  -- straightforward proof skipped for now
  sorry

end james_total_fish_catch_l91_91575


namespace measure_angle_BAO_l91_91543

open Real

-- Definitions for the problem setup
def is_diameter (CD : ℝ) (O : ℝ) : Prop := true -- Placeholder for the correct geometric definition
def lies_on_extension (A : ℝ) (D C : ℝ) : Prop := true -- Placeholder
def semicircle_angle (E D : ℝ) (θ : ℝ) : Prop := θ = 60
def intersection_with_semicircle (AE : ℝ) (B : ℝ) : Prop := true -- Placeholder

-- Conditions
variables {CD O A D C E B : ℝ}
variables (h1 : is_diameter CD O)
          (h2 : lies_on_extension A D C)
          (h3 : semicircle_angle E D 60)
          (h4 : intersection_with_semicircle A E B)
          (h5 : dist A B = dist O D)
          (h6 : ∠EOD = 60)

-- The statement to prove
theorem measure_angle_BAO : ∠BAO = 20 := by
  sorry

end measure_angle_BAO_l91_91543


namespace base_conversion_addition_correct_l91_91356

theorem base_conversion_addition_correct :
  let A := 10
  let C := 12
  let n13 := 3 * 13^2 + 7 * 13^1 + 6
  let n14 := 4 * 14^2 + A * 14^1 + C
  n13 + n14 = 1540 := by
    let A := 10
    let C := 12
    let n13 := 3 * 13^2 + 7 * 13^1 + 6
    let n14 := 4 * 14^2 + A * 14^1 + C
    let sum := n13 + n14
    have h1 : n13 = 604 := by sorry
    have h2 : n14 = 936 := by sorry
    have h3 : sum = 1540 := by sorry
    exact h3

end base_conversion_addition_correct_l91_91356


namespace polygons_ratio_four_three_l91_91447

theorem polygons_ratio_four_three : 
  ∃ (r k : ℕ), 3 ≤ r ∧ 3 ≤ k ∧ 
  (180 - (360 / r : ℝ)) / (180 - (360 / k : ℝ)) = 4 / 3 
  ∧ ((r, k) = (42,7) ∨ (r, k) = (18,6) ∨ (r, k) = (10,5) ∨ (r, k) = (6,4)) :=
sorry

end polygons_ratio_four_three_l91_91447


namespace smallest_right_triangle_area_l91_91109

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l91_91109


namespace distance_between_M_and_focus_l91_91521

theorem distance_between_M_and_focus
  (θ : ℝ)
  (x y : ℝ)
  (M : ℝ × ℝ := (1/2, 0))
  (F : ℝ × ℝ := (0, 1/2))
  (hx : x = 2 * Real.cos θ)
  (hy : y = 1 + Real.cos (2 * θ)) :
  Real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2) = Real.sqrt 2 / 2 :=
by
  sorry

end distance_between_M_and_focus_l91_91521


namespace james_total_catch_l91_91578

def pounds_of_trout : ℕ := 200
def pounds_of_salmon : ℕ := pounds_of_trout + (pounds_of_trout / 2)
def pounds_of_tuna : ℕ := 2 * pounds_of_salmon
def total_pounds_of_fish : ℕ := pounds_of_trout + pounds_of_salmon + pounds_of_tuna

theorem james_total_catch : total_pounds_of_fish = 1100 := by
  sorry

end james_total_catch_l91_91578


namespace inequality_min_m_l91_91250

theorem inequality_min_m (m : ℝ) (x : ℝ) (hx : 1 < x) : 
  x + m * Real.log x + 1 / Real.exp x ≥ Real.exp (m * Real.log x) :=
sorry

end inequality_min_m_l91_91250


namespace cubic_yards_to_cubic_feet_l91_91393

theorem cubic_yards_to_cubic_feet (yards_to_feet: 1 = 3): 6 * 27 = 162 := by
  -- We know from the setup that:
  -- 1 cubic yard = 27 cubic feet
  -- Hence,
  -- 6 cubic yards = 6 * 27 = 162 cubic feet
  sorry

end cubic_yards_to_cubic_feet_l91_91393


namespace truck_capacity_l91_91453

theorem truck_capacity (x y : ℝ)
  (h1 : 3 * x + 4 * y = 22)
  (h2 : 5 * x + 2 * y = 25) :
  4 * x + 3 * y = 23.5 :=
sorry

end truck_capacity_l91_91453


namespace time_ratio_A_to_B_l91_91483

theorem time_ratio_A_to_B (T_A T_B : ℝ) (hB : T_B = 36) (hTogether : 1 / T_A + 1 / T_B = 1 / 6) : T_A / T_B = 1 / 5 :=
by
  sorry

end time_ratio_A_to_B_l91_91483


namespace treasure_15_signs_l91_91793

def min_treasure_signs (signs_truthful: ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ k, signs_truthful k = 0 → (k ≠ n)) ∧ (∀ k, signs_truthful k > 0 → (k ≠ n)) ∧ 
  (∀ k, k < n → signs_truthful k ≠ 0) ∧ (∀ k, k > n → ¬ (signs_truthful k = 0))

theorem treasure_15_signs : 
  ∀ (signs_truthful : ℕ → ℕ)
  (count_1 : signs_truthful 15 = 15)
  (count_2 : signs_truthful 8 = 8)
  (count_3 : signs_truthful 4 = 4)
  (count_4 : signs_truthful 3 = 3)
  (all_false : ∀ k, signs_truthful k = 0 → ¬(∃ m, signs_truthful m = k)),
  min_treasure_signs signs_truthful 15 :=
by
  describe_theorem sorry

end treasure_15_signs_l91_91793


namespace ratio_of_divisors_l91_91862

def M : Nat := 75 * 75 * 140 * 343

noncomputable def sumOfOddDivisors (n : Nat) : Nat := 
  -- Function that computes the sum of all odd divisors of n. (placeholder)
  sorry

noncomputable def sumOfEvenDivisors (n : Nat) : Nat := 
  -- Function that computes the sum of all even divisors of n. (placeholder)
  sorry

theorem ratio_of_divisors :
  let sumOdd := sumOfOddDivisors M
  let sumEven := sumOfEvenDivisors M
  sumOdd / sumEven = 1 / 6 := 
by
  sorry

end ratio_of_divisors_l91_91862


namespace smallest_area_of_right_triangle_l91_91083

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l91_91083


namespace line_equation_parallel_l91_91220

theorem line_equation_parallel (x₁ y₁ m : ℝ) (h₁ : (x₁, y₁) = (1, -2)) (h₂ : m = 2) :
  ∃ a b c : ℝ, a * x₁ + b * y₁ + c = 0 ∧ a * 2 + b * 1 + c = 4 := by
sorry

end line_equation_parallel_l91_91220


namespace determine_denominator_of_fraction_l91_91321

theorem determine_denominator_of_fraction (x : ℝ) (h : 57 / x = 0.0114) : x = 5000 :=
by
  sorry

end determine_denominator_of_fraction_l91_91321


namespace sphere_volume_l91_91057

theorem sphere_volume (S : ℝ) (r : ℝ) (V : ℝ) (h₁ : S = 256 * Real.pi) (h₂ : S = 4 * Real.pi * r^2) : V = 2048 / 3 * Real.pi :=
by
  sorry

end sphere_volume_l91_91057


namespace kerosene_cost_is_024_l91_91542

-- Definitions from the conditions
def dozen_eggs_cost := 0.36 -- Cost of a dozen eggs is the same as 1 pound of rice which is $0.36
def pound_of_rice_cost := 0.36
def kerosene_cost := 8 * (0.36 / 12) -- Cost of kerosene is the cost of 8 eggs

-- Theorem to prove
theorem kerosene_cost_is_024 : kerosene_cost = 0.24 := by
  sorry

end kerosene_cost_is_024_l91_91542


namespace monica_cookies_left_l91_91429

theorem monica_cookies_left 
  (father_cookies : ℕ) 
  (mother_cookies : ℕ) 
  (brother_cookies : ℕ) 
  (sister_cookies : ℕ) 
  (aunt_cookies : ℕ) 
  (cousin_cookies : ℕ) 
  (total_cookies : ℕ)
  (father_cookies_eq : father_cookies = 12)
  (mother_cookies_eq : mother_cookies = father_cookies / 2)
  (brother_cookies_eq : brother_cookies = mother_cookies + 2)
  (sister_cookies_eq : sister_cookies = brother_cookies * 3)
  (aunt_cookies_eq : aunt_cookies = father_cookies * 2)
  (cousin_cookies_eq : cousin_cookies = aunt_cookies - 5)
  (total_cookies_eq : total_cookies = 120) : 
  total_cookies - (father_cookies + mother_cookies + brother_cookies + sister_cookies + aunt_cookies + cousin_cookies) = 27 :=
by
  sorry

end monica_cookies_left_l91_91429


namespace part1_part2_l91_91962

variables (a b c d m : Real) 

-- Condition: a and b are opposite numbers
def opposite_numbers (a b : Real) : Prop := a = -b

-- Condition: c and d are reciprocals
def reciprocals (c d : Real) : Prop := c = 1 / d

-- Condition: |m| = 3
def absolute_value_three (m : Real) : Prop := abs m = 3

-- Statement for part 1
theorem part1 (h1 : opposite_numbers a b) (h2 : reciprocals c d) (h3 : absolute_value_three m) :
  a + b = 0 ∧ c * d = 1 ∧ (m = 3 ∨ m = -3) :=
by
  sorry

-- Statement for part 2
theorem part2 (h1 : opposite_numbers a b) (h2 : reciprocals c d) (h3 : absolute_value_three m) (h4 : m < 0) :
  m^3 + c * d + (a + b) / m = -26 :=
by
  sorry

end part1_part2_l91_91962


namespace total_books_l91_91349

-- Define the given conditions
def books_per_shelf : ℕ := 8
def mystery_shelves : ℕ := 12
def picture_shelves : ℕ := 9

-- Define the number of books on each type of shelves
def total_mystery_books : ℕ := mystery_shelves * books_per_shelf
def total_picture_books : ℕ := picture_shelves * books_per_shelf

-- Define the statement to prove
theorem total_books : total_mystery_books + total_picture_books = 168 := by
  sorry

end total_books_l91_91349


namespace minimum_final_percentage_to_pass_l91_91501

-- Conditions
def problem_sets : ℝ := 100
def midterm_worth : ℝ := 100
def final_worth : ℝ := 300
def perfect_problem_sets_score : ℝ := 100
def midterm1_score : ℝ := 0.60 * midterm_worth
def midterm2_score : ℝ := 0.70 * midterm_worth
def midterm3_score : ℝ := 0.80 * midterm_worth
def passing_percentage : ℝ := 0.70

-- Derived Values
def total_points_available : ℝ := problem_sets + 3 * midterm_worth + final_worth
def required_points_to_pass : ℝ := passing_percentage * total_points_available
def total_points_before_final : ℝ := perfect_problem_sets_score + midterm1_score + midterm2_score + midterm3_score
def points_needed_from_final : ℝ := required_points_to_pass - total_points_before_final

-- Proof Statement
theorem minimum_final_percentage_to_pass : 
  ∃ (final_score : ℝ), (final_score / final_worth * 100) ≥ 60 :=
by
  -- Calculations for proof
  let required_final_percentage := (points_needed_from_final / final_worth) * 100
  -- We need to show that the required percentage is at least 60%
  have : required_final_percentage = 60 := sorry
  exact Exists.intro 180 sorry

end minimum_final_percentage_to_pass_l91_91501


namespace right_triangle_min_area_l91_91171

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l91_91171


namespace find_f_at_1_l91_91827

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem find_f_at_1 : f 1 = 2 := by
  sorry

end find_f_at_1_l91_91827


namespace find_measure_A_and_b_c_sum_l91_91264

open Real

noncomputable def triangle_abc (a b c A B C : ℝ) : Prop :=
  ∀ (A B C : ℝ),
  A + B + C = π ∧
  a = sin A ∧
  b = sin B ∧
  c = sin C ∧
  cos (A - C) - cos (A + C) = sqrt 3 * sin C

theorem find_measure_A_and_b_c_sum (a b c A B C : ℝ)
  (h_triangle : triangle_abc a b c A B C) 
  (h_area : (1/2) * b * c * (sin A) = (3 * sqrt 3) / 16) 
  (h_b_def : b = sin B) :
  A = π / 3 ∧ b + c = sqrt 3 := by
  sorry

end find_measure_A_and_b_c_sum_l91_91264


namespace prime_count_60_to_70_l91_91257

theorem prime_count_60_to_70 : ∃ primes : Finset ℕ, primes.card = 2 ∧ ∀ p ∈ primes, 60 < p ∧ p < 70 ∧ Nat.Prime p :=
by
  sorry

end prime_count_60_to_70_l91_91257


namespace brown_eggs_survived_l91_91588

-- Conditions
variables (B : ℕ)  -- Number of brown eggs that survived

-- States that Linda had three times as many white eggs as brown eggs before the fall
def white_eggs_eq_3_times_brown : Prop := 3 * B + B = 12

-- Theorem statement
theorem brown_eggs_survived (h : white_eggs_eq_3_times_brown B) : B = 3 :=
sorry

end brown_eggs_survived_l91_91588


namespace sum_consecutive_even_integers_l91_91760

theorem sum_consecutive_even_integers (n : ℕ) (h : 2 * n + 4 = 156) : 
  n + (n + 2) + (n + 4) = 234 := 
by
  sorry

end sum_consecutive_even_integers_l91_91760


namespace stock_price_end_of_third_year_l91_91664

def first_year_price (initial_price : ℝ) (first_year_increase : ℝ) : ℝ :=
  initial_price + (initial_price * first_year_increase)

def second_year_price (price_end_first : ℝ) (second_year_decrease : ℝ) : ℝ :=
  price_end_first - (price_end_first * second_year_decrease)

def third_year_price (price_end_second : ℝ) (third_year_increase : ℝ) : ℝ :=
  price_end_second + (price_end_second * third_year_increase)

theorem stock_price_end_of_third_year :
  ∀ (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) (third_year_increase : ℝ),
    initial_price = 150 →
    first_year_increase = 0.5 →
    second_year_decrease = 0.3 →
    third_year_increase = 0.2 →
    third_year_price (second_year_price (first_year_price initial_price first_year_increase) second_year_decrease) third_year_increase = 189 :=
by
  intros initial_price first_year_increase second_year_decrease third_year_increase
  sorry

end stock_price_end_of_third_year_l91_91664


namespace points_per_round_l91_91426

def total_points : ℕ := 78
def num_rounds : ℕ := 26

theorem points_per_round : total_points / num_rounds = 3 := by
  sorry

end points_per_round_l91_91426


namespace smallest_area_right_triangle_l91_91133

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l91_91133


namespace find_possible_values_of_a_l91_91678

noncomputable def P : Set ℝ := {x | x^2 + x - 6 = 0}
noncomputable def Q (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

theorem find_possible_values_of_a (a : ℝ) (h : Q a ⊆ P) :
  a = 0 ∨ a = -1/2 ∨ a = 1/3 := by
  sorry

end find_possible_values_of_a_l91_91678


namespace cylinder_radius_unique_l91_91980

theorem cylinder_radius_unique
  (r : ℝ) (h : ℝ) (V : ℝ) (y : ℝ)
  (h_eq : h = 2)
  (V_eq : V = 2 * Real.pi * r ^ 2)
  (y_eq_increase_radius : y = 2 * Real.pi * ((r + 6) ^ 2 - r ^ 2))
  (y_eq_increase_height : y = 6 * Real.pi * r ^ 2) :
  r = 6 :=
by
  sorry

end cylinder_radius_unique_l91_91980


namespace power_six_tens_digit_l91_91317

def tens_digit (x : ℕ) : ℕ := (x / 10) % 10

theorem power_six_tens_digit (n : ℕ) (hn : tens_digit (6^n) = 1) : n = 3 :=
sorry

end power_six_tens_digit_l91_91317


namespace sales_volume_maximum_profit_l91_91599

noncomputable def profit (x : ℝ) : ℝ := (x - 34) * (-2 * x + 296)

theorem sales_volume (x : ℝ) : 200 - 2 * (x - 48) = -2 * x + 296 := by
  sorry

theorem maximum_profit :
  (∀ x : ℝ, profit x ≤ profit 91) ∧ profit 91 = 6498 := by
  sorry

end sales_volume_maximum_profit_l91_91599


namespace number_of_n_l91_91507

theorem number_of_n (h1: n > 0) (h2: n ≤ 2000) (h3: ∃ m, 10 * n = m^2) : n = 14 :=
by sorry

end number_of_n_l91_91507


namespace smallest_area_correct_l91_91148

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l91_91148


namespace shifted_parabola_passes_through_point_l91_91886

theorem shifted_parabola_passes_through_point :
  let original_eq : ℝ → ℝ := λ x, -x^2 - 2*x + 3
  let transformed_eq : ℝ → ℝ := λ x, -x^2 + 2
  transformed_eq (-1) = 1 :=
by
  let original_eq : ℝ → ℝ := λ x, -x^2 - 2*x + 3
  let transformed_eq : ℝ → ℝ := λ x, -x^2 + 2
  sorry

end shifted_parabola_passes_through_point_l91_91886


namespace commute_time_variance_l91_91191

theorem commute_time_variance
  (x y : ℝ)
  (h1 : x + y = 20)
  (h2 : (x - 10)^2 + (y - 10)^2 = 8) :
  x^2 + y^2 = 208 :=
by
  sorry

end commute_time_variance_l91_91191


namespace last_two_digits_x_pow_y_add_y_pow_x_l91_91863

noncomputable def proof_problem (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1/x + 1/y = 2/13) : ℕ :=
  (x^y + y^x) % 100

theorem last_two_digits_x_pow_y_add_y_pow_x {x y : ℕ} (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1/x + 1/y = 2/13) : 
  proof_problem x y h1 h2 h3 h4 = 74 :=
sorry

end last_two_digits_x_pow_y_add_y_pow_x_l91_91863


namespace burn_rate_walking_l91_91791

def burn_rate_running : ℕ := 10
def total_calories : ℕ := 450
def total_time : ℕ := 60
def running_time : ℕ := 35

theorem burn_rate_walking :
  ∃ (W : ℕ), ((running_time * burn_rate_running) + ((total_time - running_time) * W) = total_calories) ∧ (W = 4) :=
by
  sorry

end burn_rate_walking_l91_91791


namespace sum_of_squares_of_rates_l91_91212

theorem sum_of_squares_of_rates (b j s : ℕ) 
  (h1 : 3 * b + 2 * j + 4 * s = 66) 
  (h2 : 3 * j + 2 * s + 4 * b = 96) : 
  b^2 + j^2 + s^2 = 612 := 
by 
  sorry

end sum_of_squares_of_rates_l91_91212


namespace complete_work_together_in_days_l91_91466

/-
p is 60% more efficient than q.
p can complete the work in 26 days.
Prove that p and q together will complete the work in approximately 18.57 days.
-/

noncomputable def work_together_days (p_efficiency q_efficiency : ℝ) (p_days : ℝ) : ℝ :=
  let p_work_rate := 1 / p_days
  let q_work_rate := q_efficiency / p_efficiency * p_work_rate
  let combined_work_rate := p_work_rate + q_work_rate
  1 / combined_work_rate

theorem complete_work_together_in_days :
  ∀ (p_efficiency q_efficiency p_days : ℝ),
  p_efficiency = 1 ∧ q_efficiency = 0.4 ∧ p_days = 26 →
  abs (work_together_days p_efficiency q_efficiency p_days - 18.57) < 0.01 := by
  intros p_efficiency q_efficiency p_days
  rintro ⟨heff_p, heff_q, hdays_p⟩
  simp [heff_p, heff_q, hdays_p, work_together_days]
  sorry

end complete_work_together_in_days_l91_91466


namespace A_and_D_independent_l91_91311

-- Define the probabilities of elementary events
def prob_A : ℚ := 1 / 6
def prob_B : ℚ := 1 / 6
def prob_C : ℚ := 5 / 36
def prob_D : ℚ := 1 / 6

-- Define the joint probability of A and D
def prob_A_and_D : ℚ := 1 / 36

-- Define the independence condition
def independent (P_X P_Y P_XY : ℚ) : Prop := P_XY = P_X * P_Y

-- Prove that events A and D are independent
theorem A_and_D_independent : 
  independent prob_A prob_D prob_A_and_D := by
  -- The proof is skipped
  sorry

end A_and_D_independent_l91_91311


namespace swimmer_speed_in_still_water_l91_91339

variable (distance : ℝ) (time : ℝ) (current_speed : ℝ) (swimmer_speed_still_water : ℝ)

-- Define the given conditions
def conditions := 
  distance = 8 ∧
  time = 5 ∧
  current_speed = 1.4 ∧
  (distance / time = swimmer_speed_still_water - current_speed)

-- The theorem we want to prove
theorem swimmer_speed_in_still_water : 
  conditions distance time current_speed swimmer_speed_still_water → 
  swimmer_speed_still_water = 3 := 
by 
  -- Skipping the actual proof
  sorry

end swimmer_speed_in_still_water_l91_91339


namespace smallest_area_correct_l91_91152

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l91_91152


namespace two_integer_solutions_iff_m_l91_91230

def op (p q : ℝ) : ℝ := p + q - p * q

theorem two_integer_solutions_iff_m (m : ℝ) :
  (∃ x1 x2 : ℤ, x1 ≠ x2 ∧ op 2 x1 > 0 ∧ op x1 3 ≤ m ∧ op 2 x2 > 0 ∧ op x2 3 ≤ m) ↔ 3 ≤ m ∧ m < 5 :=
by
  sorry

end two_integer_solutions_iff_m_l91_91230


namespace problem_statement_l91_91326

theorem problem_statement : (6^3 + 4^2) * 7^5 = 3897624 := by
  sorry

end problem_statement_l91_91326


namespace smallest_right_triangle_area_l91_91145

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l91_91145


namespace f_above_g_l91_91246

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (Real.exp x) / (x - m)
def g (x : ℝ) : ℝ := x^2 + x

theorem f_above_g (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1/2) : 
  ∀ x, m ≤ x ∧ x ≤ m + 1 → f x m > g x := 
sorry

end f_above_g_l91_91246


namespace positive_real_numbers_l91_91456

theorem positive_real_numbers
  (a b c : ℝ)
  (h1 : a + b + c > 0)
  (h2 : b * c + c * a + a * b > 0)
  (h3 : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 :=
by
  sorry

end positive_real_numbers_l91_91456


namespace gen_formula_arithmetic_seq_sum_maximizes_at_5_l91_91677

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a n = a 1 + (n - 1) * d

variables (an : ℕ → ℤ) (Sn : ℕ → ℤ)
variable (d : ℤ)

theorem gen_formula_arithmetic_seq (h1 : an 3 = 5) (h2 : an 10 = -9) :
  ∀ n, an n = 11 - 2 * n :=
sorry

theorem sum_maximizes_at_5 (h_seq : ∀ n, an n = 11 - 2 * n) :
  ∀ n, Sn n = (n * 10 - n^2) → (∃ n, ∀ k, Sn n ≥ Sn k) :=
sorry

end gen_formula_arithmetic_seq_sum_maximizes_at_5_l91_91677


namespace smallest_n_satisfying_conditions_l91_91370

def contains_digit (num : ℕ) (d : ℕ) : Prop :=
  d ∈ num.digits 10

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, contains_digit (n^2) 7 ∧ contains_digit ((n+1)^2) 7 ∧ ¬ contains_digit ((n+2)^2) 7 ∧ ∀ m : ℕ, m < n → ¬ (contains_digit (m^2) 7 ∧ contains_digit ((m+1)^2) 7 ∧ ¬ contains_digit ((m+2)^2) 7) := 
sorry

end smallest_n_satisfying_conditions_l91_91370


namespace animal_count_l91_91480

theorem animal_count (dogs : ℕ) (cats : ℕ) (birds : ℕ) (fish : ℕ)
  (h1 : dogs = 6)
  (h2 : cats = dogs / 2)
  (h3 : birds = dogs * 2)
  (h4 : fish = dogs * 3) : 
  dogs + cats + birds + fish = 39 :=
by
  sorry

end animal_count_l91_91480


namespace distinct_exponentiation_values_l91_91205

theorem distinct_exponentiation_values : 
  ∃ (standard other1 other2 other3 : ℕ), 
    standard ≠ other1 ∧ 
    standard ≠ other2 ∧ 
    standard ≠ other3 ∧ 
    other1 ≠ other2 ∧ 
    other1 ≠ other3 ∧ 
    other2 ≠ other3 := 
sorry

end distinct_exponentiation_values_l91_91205


namespace linear_regression_forecast_l91_91972

variable (x : ℝ) (y : ℝ)
variable (b : ℝ) (a : ℝ) (center_x : ℝ) (center_y : ℝ)

theorem linear_regression_forecast :
  b=-2 → center_x=4 → center_y=50 → (center_y = b * center_x + a) →
  (a = 58) → (x = 6) → y = b * x + a → y = 46 :=
by
  intros hb hcx hcy heq ha hx hy
  sorry

end linear_regression_forecast_l91_91972


namespace percentage_lower_grades_have_cars_l91_91893

-- Definitions for the conditions
def n_seniors : ℕ := 300
def p_car : ℚ := 0.50
def n_lower : ℕ := 900
def p_total : ℚ := 0.20

-- Definition for the number of students who have cars in the lower grades
def n_cars_lower : ℚ := 
  let total_students := n_seniors + n_lower
  let total_cars := p_total * total_students
  total_cars - (p_car * n_seniors)

-- Prove the percentage of freshmen, sophomores, and juniors who have cars
theorem percentage_lower_grades_have_cars : 
  (n_cars_lower / n_lower) * 100 = 10 := 
by sorry

end percentage_lower_grades_have_cars_l91_91893


namespace calculate_final_price_l91_91774

def original_price : ℝ := 120
def fixture_discount : ℝ := 0.20
def decor_discount : ℝ := 0.15

def discounted_price_after_first_discount (p : ℝ) (d : ℝ) : ℝ :=
  p * (1 - d)

def final_price (p : ℝ) (d1 : ℝ) (d2 : ℝ) : ℝ :=
  let price_after_first_discount := discounted_price_after_first_discount p d1
  price_after_first_discount * (1 - d2)

theorem calculate_final_price :
  final_price original_price fixture_discount decor_discount = 81.60 :=
by sorry

end calculate_final_price_l91_91774


namespace Ryan_has_28_marbles_l91_91931

theorem Ryan_has_28_marbles :
  ∃ R : ℕ, (12 + R) - (1/4 * (12 + R)) * 2 = 20 ∧ R = 28 :=
by
  sorry

end Ryan_has_28_marbles_l91_91931


namespace train_cross_signal_pole_in_18_seconds_l91_91634

noncomputable def train_length : ℝ := 300
noncomputable def platform_length : ℝ := 550
noncomputable def crossing_time_platform : ℝ := 51
noncomputable def signal_pole_crossing_time : ℝ := 18

theorem train_cross_signal_pole_in_18_seconds (t l_p t_p t_s : ℝ)
    (h1 : t = train_length)
    (h2 : l_p = platform_length)
    (h3 : t_p = crossing_time_platform)
    (h4 : t_s = signal_pole_crossing_time) : 
    (t + l_p) / t_p = train_length / signal_pole_crossing_time :=
by
  unfold train_length platform_length crossing_time_platform signal_pole_crossing_time at *
  -- proof will go here
  sorry

end train_cross_signal_pole_in_18_seconds_l91_91634


namespace g_of_10_l91_91443

noncomputable def g : ℕ → ℝ := sorry

axiom g_initial : g 1 = 2

axiom g_condition : ∀ (m n : ℕ), m ≥ n → g (m + n) + g (m - n) = 2 * g m + 3 * g n

theorem g_of_10 : g 10 = 496 :=
by
  sorry

end g_of_10_l91_91443


namespace rosa_called_last_week_l91_91833

noncomputable def total_pages_called : ℝ := 18.8
noncomputable def pages_called_this_week : ℝ := 8.6
noncomputable def pages_called_last_week : ℝ := total_pages_called - pages_called_this_week

theorem rosa_called_last_week :
  pages_called_last_week = 10.2 :=
by
  sorry

end rosa_called_last_week_l91_91833


namespace clive_change_l91_91200

theorem clive_change (total_money : ℝ) (num_olives_needed : ℕ) (olives_per_jar : ℕ) (cost_per_jar : ℝ)
  (h1 : total_money = 10)
  (h2 : num_olives_needed = 80)
  (h3 : olives_per_jar = 20)
  (h4 : cost_per_jar = 1.5) : total_money - (num_olives_needed / olives_per_jar) * cost_per_jar = 4 := by
  sorry

end clive_change_l91_91200


namespace count_solutions_congruence_l91_91679

theorem count_solutions_congruence (x : ℕ) (h1 : 0 < x ∧ x < 50) (h2 : x + 7 ≡ 45 [MOD 22]) : ∃ x1 x2, (x1 ≠ x2) ∧ (0 < x1 ∧ x1 < 50) ∧ (0 < x2 ∧ x2 < 50) ∧ (x1 + 7 ≡ 45 [MOD 22]) ∧ (x2 + 7 ≡ 45 [MOD 22]) ∧ (∀ y, (0 < y ∧ y < 50) ∧ (y + 7 ≡ 45 [MOD 22]) → (y = x1 ∨ y = x2)) :=
by {
  sorry
}

end count_solutions_congruence_l91_91679


namespace period_in_years_proof_l91_91376

-- Definitions
def marbles (P : ℕ) : ℕ := P

def remaining_marbles (M : ℕ) : ℕ := (M / 4)

def doubled_remaining_marbles (M : ℕ) : ℕ := 2 * (M / 4)

def age_in_five_years (current_age : ℕ) : ℕ := current_age + 5

-- Given Conditions
variables (P : ℕ) (current_age : ℕ) (H1 : marbles P = P) (H2 : current_age = 45)

-- Final Proof Goal
theorem period_in_years_proof (H3 : doubled_remaining_marbles P = age_in_five_years current_age) : P = 100 :=
sorry

end period_in_years_proof_l91_91376


namespace total_area_l91_91572

-- Defining basic dimensions as conditions
def left_vertical_length : ℕ := 7
def top_horizontal_length_left : ℕ := 5
def left_vertical_length_near_top : ℕ := 3
def top_horizontal_length_right_of_center : ℕ := 2
def right_vertical_length_near_center : ℕ := 3
def top_horizontal_length_far_right : ℕ := 2

-- Defining areas of partitioned rectangles
def area_bottom_left_rectangle : ℕ := 7 * 8
def area_middle_rectangle : ℕ := 5 * 3
def area_top_left_rectangle : ℕ := 2 * 8
def area_top_right_rectangle : ℕ := 2 * 7
def area_bottom_right_rectangle : ℕ := 4 * 4

-- Calculate the total area of the figure
theorem total_area : 
  area_bottom_left_rectangle + area_middle_rectangle + area_top_left_rectangle + area_top_right_rectangle + area_bottom_right_rectangle = 117 := by
  -- Proof steps will go here
  sorry

end total_area_l91_91572


namespace strategy2_is_better_final_cost_strategy2_correct_l91_91643

def initial_cost : ℝ := 12000

def strategy1_discount : ℝ := 
  let after_first_discount := initial_cost * 0.70
  let after_second_discount := after_first_discount * 0.85
  let after_third_discount := after_second_discount * 0.95
  after_third_discount

def strategy2_discount : ℝ := 
  let after_first_discount := initial_cost * 0.55
  let after_second_discount := after_first_discount * 0.90
  let after_third_discount := after_second_discount * 0.90
  let final_cost := after_third_discount + 150
  final_cost

theorem strategy2_is_better : strategy2_discount < strategy1_discount :=
by {
  sorry -- proof goes here
}

theorem final_cost_strategy2_correct : strategy2_discount = 5496 :=
by {
  sorry -- proof goes here
}

end strategy2_is_better_final_cost_strategy2_correct_l91_91643


namespace smallest_positive_int_linear_combination_l91_91905

theorem smallest_positive_int_linear_combination (m n : ℤ) :
  ∃ k : ℤ, 4509 * m + 27981 * n = k ∧ k > 0 ∧ k ≤ 4509 * m + 27981 * n → k = 3 :=
by
  sorry

end smallest_positive_int_linear_combination_l91_91905


namespace volleyball_team_selection_l91_91993

/-- A set representing players on the volleyball team -/
def players : Finset String := {
  "Missy", "Lauren", "Liz", -- triplets
  "Anna", "Mia",           -- twins
  "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10" -- other players
}

/-- The triplets -/
def triplets : Finset String := {"Missy", "Lauren", "Liz"}

/-- The twins -/
def twins : Finset String := {"Anna", "Mia"}

/-- The number of ways to choose 7 starters given the restrictions -/
theorem volleyball_team_selection : 
  let total_ways := (players.card.choose 7)
  let select_3_triplets := (players \ triplets).card.choose 4
  let select_2_twins := (players \ twins).card.choose 5
  let select_all_restriction := (players \ (triplets ∪ twins)).card.choose 2
  total_ways - select_3_triplets - select_2_twins + select_all_restriction = 9778 := by
  sorry

end volleyball_team_selection_l91_91993


namespace john_buys_360_packs_l91_91850

def John_buys_packs (classes students_per_class packs_per_student total_packs : ℕ) : Prop :=
  classes = 6 →
  students_per_class = 30 →
  packs_per_student = 2 →
  total_packs = (classes * students_per_class) * packs_per_student
  → total_packs = 360

theorem john_buys_360_packs : John_buys_packs 6 30 2 360 :=
by { intros, sorry }

end john_buys_360_packs_l91_91850


namespace smallest_area_of_right_triangle_l91_91116

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l91_91116


namespace polynomial_identity_sum_l91_91406

theorem polynomial_identity_sum (A B C D : ℤ) (h : (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) : 
  A + B + C + D = 36 := 
by 
  sorry

end polynomial_identity_sum_l91_91406


namespace basketball_game_proof_l91_91411

-- Definition of the conditions
def num_teams (x : ℕ) : Prop := ∃ n : ℕ, n = x

def games_played (x : ℕ) (total_games : ℕ) : Prop := total_games = 28

def game_combinations (x : ℕ) : ℕ := (x * (x - 1)) / 2

-- Proof statement using the conditions
theorem basketball_game_proof (x : ℕ) (h1 : num_teams x) (h2 : games_played x 28) : 
  game_combinations x = 28 := by
  sorry

end basketball_game_proof_l91_91411


namespace total_amount_received_correct_l91_91711

variable (total_won : ℝ) (fraction : ℝ) (students : ℕ)
variable (portion_per_student : ℝ := total_won * fraction)
variable (total_given : ℝ := portion_per_student * students)

theorem total_amount_received_correct :
  total_won = 555850 →
  fraction = 3 / 10000 →
  students = 500 →
  total_given = 833775 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_amount_received_correct_l91_91711


namespace find_z_l91_91446

theorem find_z
  (z : ℝ)
  (h : (1 : ℝ) • (2 : ℝ) + 4 • (-1 : ℝ) + z • (3 : ℝ) = 6) :
  z = 8 / 3 :=
by 
  sorry

end find_z_l91_91446


namespace square_plot_area_l91_91461

theorem square_plot_area (price_per_foot : ℝ) (total_cost : ℝ) (s : ℝ) (A : ℝ)
  (h1 : price_per_foot = 58)
  (h2 : total_cost = 1160)
  (h3 : total_cost = 4 * s * price_per_foot)
  (h4 : A = s * s) :
  A = 25 := by
  sorry

end square_plot_area_l91_91461


namespace tan_x_tan_y_relation_l91_91689

/-- If 
  (sin x / cos y) + (sin y / cos x) = 2 
  and 
  (cos x / sin y) + (cos y / sin x) = 3, 
  then 
  (tan x / tan y) + (tan y / tan x) = 16 / 3.
 -/
theorem tan_x_tan_y_relation (x y : ℝ)
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 16 / 3 :=
sorry

end tan_x_tan_y_relation_l91_91689


namespace trajectory_equation_no_such_point_l91_91824

-- Conditions for (I): The ratio of the distances is given
def ratio_condition (P : ℝ × ℝ) : Prop :=
  let M := (1, 0)
  let N := (4, 0)
  2 * Real.sqrt ((P.1 - M.1)^2 + P.2^2) = Real.sqrt ((P.1 - N.1)^2 + P.2^2)

-- Proof of (I): Find the trajectory equation of point P
theorem trajectory_equation : 
  ∀ P : ℝ × ℝ, ratio_condition P → P.1^2 + P.2^2 = 4 :=
by
  sorry

-- Conditions for (II): Given points A, B, C
def points_condition (P : ℝ × ℝ) : Prop :=
  let A := (-2, -2)
  let B := (-2, 6)
  let C := (-4, 2)
  (P.1 + 2)^2 + (P.2 + 2)^2 + 
  (P.1 + 2)^2 + (P.2 - 6)^2 + 
  (P.1 + 4)^2 + (P.2 - 2)^2 = 36

-- Proof of (II): Determine the non-existence of point P
theorem no_such_point (P : ℝ × ℝ) : 
  P.1^2 + P.2^2 = 4 → ¬ points_condition P :=
by
  sorry

end trajectory_equation_no_such_point_l91_91824


namespace contrapositive_l91_91045

theorem contrapositive (a b : ℕ) : (a = 0 → ab = 0) → (ab ≠ 0 → a ≠ 0) :=
by
  sorry

end contrapositive_l91_91045


namespace factorization_of_polynomial_l91_91742

theorem factorization_of_polynomial (x : ℝ) : 2 * x^2 - 12 * x + 18 = 2 * (x - 3)^2 := by
  sorry

end factorization_of_polynomial_l91_91742


namespace arithmetic_mean_of_4_and_16_l91_91821

-- Define the arithmetic mean condition
def is_arithmetic_mean (a b x : ℝ) : Prop :=
  x = (a + b) / 2

-- Theorem to prove that x = 10 if it is the mean of 4 and 16
theorem arithmetic_mean_of_4_and_16 (x : ℝ) (h : is_arithmetic_mean 4 16 x) : x = 10 :=
by
  sorry

end arithmetic_mean_of_4_and_16_l91_91821


namespace problem_am_hm_l91_91865

open Real

theorem problem_am_hm (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 2) :
  ∃ S : Set ℝ, (∀ s ∈ S, (2 ≤ s)) ∧ (∀ z, (2 ≤ z) → (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 2 ∧ z = 1/x + 1/y))
  ∧ (S = {z | 2 ≤ z}) := sorry

end problem_am_hm_l91_91865


namespace larger_number_is_37_point_435_l91_91744

theorem larger_number_is_37_point_435 (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 96) (h3 : x > y) : x = 37.435 :=
by
  sorry

end larger_number_is_37_point_435_l91_91744


namespace rectangle_clear_area_l91_91283

theorem rectangle_clear_area (EF FG : ℝ)
  (radius_E radius_F radius_G radius_H : ℝ) : 
  EF = 4 → FG = 6 → 
  radius_E = 2 → radius_F = 3 → radius_G = 1.5 → radius_H = 2.5 → 
  abs ((EF * FG) - (π * radius_E^2 / 4 + π * radius_F^2 / 4 + π * radius_G^2 / 4 + π * radius_H^2 / 4)) - 7.14 < 0.5 :=
by sorry

end rectangle_clear_area_l91_91283


namespace arithmetic_sequence_problem_l91_91706

theorem arithmetic_sequence_problem (a : Nat → Int) (d a1 : Int)
  (h1 : ∀ n, a n = a1 + (n - 1) * d) 
  (h2 : a 1 + 3 * a 8 = 1560) :
  2 * a 9 - a 10 = 507 :=
sorry

end arithmetic_sequence_problem_l91_91706


namespace main_theorem_l91_91238

def point (x y : ℝ) : Prop := true

def A : Prop := point (-3) 0
def B : Prop := point 3 0

def distance (p₁ p₂ : ℝ × ℝ) : ℝ := real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

def P (x y : ℝ) : Prop := distance (x, y) (-3, 0) = 2 * distance (x, y) (3, 0)

def curve_C (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 16

def l1 (x y : ℝ) : Prop := x + y + 3 = 0

def min_QM (Q M : ℝ × ℝ) : ℝ := distance Q M

-- Main theorem
theorem main_theorem (Q : ℝ × ℝ)
  (hQ : l1 Q.1 Q.2)
  (M : ℝ × ℝ)
  (hM : curve_C M.1 M.2)
  (h_intersect : ∀ l2 : ℝ × ℝ → Prop, l2 Q → curve_C M.1 M.2 → M ∈ l2 → (∀ B ≠ (M = B → ¬curve_C B.1 B.2))) :
  curve_C (5 : ℝ) 0 ∧ (∃ min_val, min_val = 4 ∧ min_val = min_QM Q M) :=
by sorry

end main_theorem_l91_91238


namespace smallest_area_correct_l91_91149

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l91_91149


namespace fraction_identity_l91_91815

noncomputable def calc_fractions (x y : ℝ) : ℝ :=
  (x + y) / (x - y)

theorem fraction_identity (x y : ℝ) (h : (1/x + 1/y) / (1/x - 1/y) = 1001) : calc_fractions x y = -1001 :=
by
  sorry

end fraction_identity_l91_91815


namespace val_4_at_6_l91_91696

def at_op (a b : ℤ) : ℤ := 2 * a - 4 * b

theorem val_4_at_6 : at_op 4 6 = -16 := by
  sorry

end val_4_at_6_l91_91696


namespace smallest_n_such_that_squares_contain_7_l91_91366

def contains_seven (n : ℕ) : Prop :=
  let digits := n.to_digits 10
  7 ∈ digits

theorem smallest_n_such_that_squares_contain_7 :
  ∃ n : ℕ, n >= 10 ∧ contains_seven (n^2) ∧ contains_seven ((n+1)^2) ∧ n = 26 :=
by 
  sorry

end smallest_n_such_that_squares_contain_7_l91_91366


namespace seeds_per_can_l91_91868

theorem seeds_per_can (total_seeds : ℕ) (num_cans : ℕ) (h1 : total_seeds = 54) (h2 : num_cans = 9) : total_seeds / num_cans = 6 :=
by {
  sorry
}

end seeds_per_can_l91_91868


namespace single_elimination_games_l91_91019

theorem single_elimination_games (n : ℕ) (h : n = 512) : (n - 1) = 511 :=
by
  sorry

end single_elimination_games_l91_91019


namespace max_area_perpendicular_l91_91765

theorem max_area_perpendicular (a b θ : ℝ) (ha : 0 < a) (hb : 0 < b) (hθ : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) : 
  ∃ θ_max, θ_max = Real.pi / 2 ∧ (∀ θ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → 
  (0 < Real.sin θ → (1 / 2) * a * b * Real.sin θ ≤ (1 / 2) * a * b * 1)) :=
sorry

end max_area_perpendicular_l91_91765


namespace petya_max_votes_difference_l91_91566

theorem petya_max_votes_difference :
  ∃ (P1 P2 V1 V2 : ℕ), 
    P1 = V1 + 9 ∧ 
    V2 = P2 + 9 ∧ 
    P1 + P2 + V1 + V2 = 27 ∧ 
    P1 + P2 > V1 + V2 ∧ 
    (P1 + P2) - (V1 + V2) = 9 := 
by
  sorry

end petya_max_votes_difference_l91_91566


namespace total_animals_for_sale_l91_91478

theorem total_animals_for_sale (dogs cats birds fish : ℕ) 
  (h1 : dogs = 6)
  (h2 : cats = dogs / 2)
  (h3 : birds = dogs * 2)
  (h4 : fish = dogs * 3) :
  dogs + cats + birds + fish = 39 := 
by
  sorry

end total_animals_for_sale_l91_91478


namespace line_equation_through_origin_and_circle_chord_length_l91_91631

theorem line_equation_through_origin_and_circle_chord_length 
  (x y : ℝ) 
  (h : x^2 + y^2 - 2 * x - 4 * y + 4 = 0) 
  (chord_length : ℝ) 
  (h_chord : chord_length = 2) 
  : 2 * x - y = 0 := 
sorry

end line_equation_through_origin_and_circle_chord_length_l91_91631


namespace smallest_area_right_triangle_l91_91088

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l91_91088


namespace find_constants_l91_91958

open Set

variable {α : Type*} [LinearOrderedField α]

def Set_1 : Set α := {x | x^2 - 3*x + 2 = 0}

def Set_2 (a : α) : Set α := {x | x^2 - a*x + (a-1) = 0}

def Set_3 (m : α) : Set α := {x | x^2 - m*x + 2 = 0}

theorem find_constants (a m : α) :
  (Set_1 ∪ Set_2 a = Set_1) ∧ (Set_1 ∩ Set_2 a = Set_3 m) → 
  a = 3 ∧ m = 3 :=
by sorry

end find_constants_l91_91958


namespace max_cookie_price_l91_91712

theorem max_cookie_price :
  ∃ k p : ℕ, 
    (8 * k + 3 * p < 200) ∧ 
    (4 * k + 5 * p > 150) ∧
    (∀ k' p' : ℕ, (8 * k' + 3 * p' < 200) ∧ (4 * k' + 5 * p' > 150) → k' ≤ 19) :=
sorry

end max_cookie_price_l91_91712


namespace smallest_area_of_right_triangle_l91_91117

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l91_91117


namespace cubic_expression_value_l91_91377

theorem cubic_expression_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2010 = 2011 :=
by sorry

end cubic_expression_value_l91_91377


namespace candy_peanut_butter_is_192_l91_91746

/-
   Define the conditions and the statement to be proved.
   The definitions follow directly from the problem's conditions.
-/
def candy_problem : Prop :=
  ∃ (peanut_butter_jar grape_jar banana_jar coconut_jar : ℕ),
    banana_jar = 43 ∧
    grape_jar = banana_jar + 5 ∧
    peanut_butter_jar = 4 * grape_jar ∧
    coconut_jar = 2 * banana_jar - 10 ∧
    peanut_butter_jar = 192
  -- The tuple (question, conditions, correct answer) is translated into this lemma

theorem candy_peanut_butter_is_192 : candy_problem :=
  by
    -- Skipping the actual proof as requested
    sorry

end candy_peanut_butter_is_192_l91_91746


namespace inequality_solution_set_l91_91302

theorem inequality_solution_set (x : ℝ) :
  x^2 * (x^2 + 2*x + 1) > 2*x * (x^2 + 2*x + 1) ↔
  ((x < -1) ∨ (-1 < x ∧ x < 0) ∨ (2 < x)) :=
sorry

end inequality_solution_set_l91_91302


namespace adam_has_10_apples_l91_91195

theorem adam_has_10_apples
  (Jackie_has_2_apples : ∀ Jackie_apples, Jackie_apples = 2)
  (Adam_has_8_more_apples : ∀ Adam_apples Jackie_apples, Adam_apples = Jackie_apples + 8)
  : ∀ Adam_apples, Adam_apples = 10 :=
by {
  sorry
}

end adam_has_10_apples_l91_91195


namespace manager_wage_l91_91925

variable (M D C : ℝ)

def condition1 : Prop := D = M / 2
def condition2 : Prop := C = 1.25 * D
def condition3 : Prop := C = M - 3.1875

theorem manager_wage (h1 : condition1 M D) (h2 : condition2 D C) (h3 : condition3 M C) : M = 8.5 :=
by
  sorry

end manager_wage_l91_91925


namespace M_subset_N_l91_91027

-- Define M and N using the given conditions
def M : Set ℝ := {α | ∃ (k : ℤ), α = k * 90} ∪ {α | ∃ (k : ℤ), α = k * 180 + 45}
def N : Set ℝ := {α | ∃ (k : ℤ), α = k * 45}

-- Prove that M is a subset of N
theorem M_subset_N : M ⊆ N :=
by
  sorry

end M_subset_N_l91_91027


namespace mary_regular_hours_l91_91279

theorem mary_regular_hours (x y : ℕ) :
  8 * x + 10 * y = 760 ∧ x + y = 80 → x = 20 :=
by
  intro h
  sorry

end mary_regular_hours_l91_91279


namespace road_length_l91_91486

theorem road_length (L : ℝ) (h1 : 300 = 200 + 100)
  (h2 : 50 * 100 = 2.5 / (L / 300))
  (h3 : 75 + 50 = 125)
  (h4 : (125 / 50) * (2.5 / 100) * 200 = L - 2.5) : L = 15 := 
by
  sorry

end road_length_l91_91486


namespace camel_cost_l91_91632

variables {C H O E G Z : ℕ} 

-- conditions
axiom h1 : 10 * C = 24 * H
axiom h2 : 16 * H = 4 * O
axiom h3 : 6 * O = 4 * E
axiom h4 : 3 * E = 15 * G
axiom h5 : 8 * G = 20 * Z
axiom h6 : 12 * E = 180000

-- goal
theorem camel_cost : C = 6000 :=
by sorry

end camel_cost_l91_91632


namespace comb_eq_comb_imp_n_eq_18_l91_91233

theorem comb_eq_comb_imp_n_eq_18 {n : ℕ} (h : Nat.choose n 14 = Nat.choose n 4) : n = 18 :=
sorry

end comb_eq_comb_imp_n_eq_18_l91_91233


namespace license_plate_count_l91_91258

theorem license_plate_count :
  let letters := 26
  let digits := 10
  let total_count := letters * (letters - 1) + letters
  total_count * digits = 6760 :=
by sorry

end license_plate_count_l91_91258


namespace find_four_real_numbers_l91_91810

theorem find_four_real_numbers (x1 x2 x3 x4 : ℝ) :
  (x1 + x2 * x3 * x4 = 2) ∧
  (x2 + x1 * x3 * x4 = 2) ∧
  (x3 + x1 * x2 * x4 = 2) ∧
  (x4 + x1 * x2 * x3 = 2) →
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = 3 ∧ x4 = -1) ∨
  (x1 = -1 ∧ x2 = 3 ∧ x3 = -1 ∧ x4 = -1) ∨
  (x1 = 3 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = -1) :=
sorry

end find_four_real_numbers_l91_91810


namespace even_numbers_with_specific_square_properties_l91_91028

theorem even_numbers_with_specific_square_properties (n : ℕ) :
  (10^13 ≤ n^2 ∧ n^2 < 10^14 ∧ (n^2 % 100) / 10 = 5) → 
  (2 ∣ n ∧ 273512 > 10^5) := 
sorry

end even_numbers_with_specific_square_properties_l91_91028


namespace inequality_solution_sets_l91_91730

variable (a x : ℝ)

theorem inequality_solution_sets:
    ({x | 12 * x^2 - a * x > a^2} =
        if a > 0 then {x | x < -a/4} ∪ {x | x > a/3}
        else if a = 0 then {x | x ≠ 0}
        else {x | x < a/3} ∪ {x | x > -a/4}) :=
by sorry

end inequality_solution_sets_l91_91730


namespace smallest_right_triangle_area_l91_91096

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l91_91096


namespace problem_statement_l91_91492

theorem problem_statement : 
  (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) + (Real.sqrt 2 - Real.sqrt 3) ^ 2 = 4 - 2 * Real.sqrt 6 := 
by 
  sorry

end problem_statement_l91_91492


namespace no_response_count_l91_91431

-- Define the conditions as constants
def total_guests : ℕ := 200
def yes_percentage : ℝ := 0.83
def no_percentage : ℝ := 0.09

-- Define the terms involved in the final calculation
def yes_respondents : ℕ := total_guests * yes_percentage
def no_respondents : ℕ := total_guests * no_percentage
def total_respondents : ℕ := yes_respondents + no_respondents
def non_respondents : ℕ := total_guests - total_respondents

-- State the theorem
theorem no_response_count : non_respondents = 16 := by
  sorry

end no_response_count_l91_91431


namespace largest_divisor_of_n_squared_l91_91538

theorem largest_divisor_of_n_squared (n : ℕ) (h_pos : 0 < n) (h_div : ∀ d, d ∣ n^2 → d = 900) : 900 ∣ n^2 :=
by sorry

end largest_divisor_of_n_squared_l91_91538


namespace initial_observations_l91_91602

theorem initial_observations (n : ℕ) (S : ℕ) 
  (h1 : S / n = 11)
  (h2 : ∃ (new_obs : ℕ), (S + new_obs) / (n + 1) = 10 ∧ new_obs = 4):
  n = 6 := 
sorry

end initial_observations_l91_91602


namespace symmetry_center_range_in_interval_l91_91526

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6) + 1

theorem symmetry_center (k : ℤ) :
  ∃ n : ℤ, ∃ x : ℝ, x = Real.pi / 12 + n * Real.pi / 2 ∧ f x = 1 := 
sorry

theorem range_in_interval :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → ∃ y : ℝ, f y ∈ Set.Icc 0 3 := 
sorry

end symmetry_center_range_in_interval_l91_91526


namespace residue_of_927_mod_37_l91_91210

-- Define the condition of the problem, which is the modulus and the number
def modulus : ℤ := 37
def number : ℤ := -927

-- Define the statement we need to prove: that the residue of -927 mod 37 is 35
theorem residue_of_927_mod_37 : (number % modulus + modulus) % modulus = 35 := by
  sorry

end residue_of_927_mod_37_l91_91210


namespace burn_time_for_structure_l91_91392

noncomputable def time_to_burn_structure (total_toothpicks : ℕ) (burn_time_per_toothpick : ℕ) (adjacent_corners : Bool) : ℕ :=
  if total_toothpicks = 38 ∧ burn_time_per_toothpick = 10 ∧ adjacent_corners = true then 65 else 0

theorem burn_time_for_structure :
  time_to_burn_structure 38 10 true = 65 :=
sorry

end burn_time_for_structure_l91_91392


namespace dividend_is_2160_l91_91971

theorem dividend_is_2160 (d q r : ℕ) (h₁ : d = 2016 + d) (h₂ : q = 15) (h₃ : r = 0) : d = 2160 :=
by
  sorry

end dividend_is_2160_l91_91971


namespace emily_small_gardens_l91_91670

theorem emily_small_gardens 
  (total_seeds : ℕ)
  (seeds_in_big_garden : ℕ)
  (seeds_per_small_garden : ℕ)
  (remaining_seeds := total_seeds - seeds_in_big_garden)
  (number_of_small_gardens := remaining_seeds / seeds_per_small_garden) :
  total_seeds = 41 → seeds_in_big_garden = 29 → seeds_per_small_garden = 4 → number_of_small_gardens = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end emily_small_gardens_l91_91670


namespace sales_fifth_month_l91_91189

theorem sales_fifth_month
  (a1 a2 a3 a4 a6 : ℕ)
  (h1 : a1 = 2435)
  (h2 : a2 = 2920)
  (h3 : a3 = 2855)
  (h4 : a4 = 3230)
  (h6 : a6 = 1000)
  (avg : ℕ)
  (h_avg : avg = 2500) :
  a1 + a2 + a3 + a4 + (15000 - 1000 - (a1 + a2 + a3 + a4)) + a6 = avg * 6 :=
by
  sorry

end sales_fifth_month_l91_91189


namespace powers_of_2_not_powers_of_4_below_1000000_equals_10_l91_91395

def num_powers_of_2_not_4 (n : ℕ) : ℕ :=
  let powers_of_2 := (List.range n).filter (fun k => (2^k < 1000000));
  let powers_of_4 := (List.range n).filter (fun k => (4^k < 1000000));
  powers_of_2.length - powers_of_4.length

theorem powers_of_2_not_powers_of_4_below_1000000_equals_10 : 
  num_powers_of_2_not_4 20 = 10 :=
by
  sorry

end powers_of_2_not_powers_of_4_below_1000000_equals_10_l91_91395


namespace triangle_right_triangle_l91_91839

variable {A B C : Real}  -- Define the angles A, B, and C

theorem triangle_right_triangle (sin_A sin_B sin_C : Real)
  (h : sin_A^2 + sin_B^2 = sin_C^2) 
  (triangle_cond : A + B + C = 180) : 
  (A = 90) ∨ (B = 90) ∨ (C = 90) := 
  sorry

end triangle_right_triangle_l91_91839


namespace range_of_slopes_of_line_AB_l91_91519

variables {x y : ℝ}

/-- (O is the coordinate origin),
    (the parabola y² = 4x),
    (points A and B in the first quadrant),
    (the product of the slopes of lines OA and OB being 1) -/
theorem range_of_slopes_of_line_AB
  (O : ℝ) 
  (A B : ℝ × ℝ)
  (hxA : 0 < A.fst)
  (hyA : 0 < A.snd)
  (hxB : 0 < B.fst)
  (hyB : 0 < B.snd)
  (hA_on_parabola : A.snd^2 = 4 * A.fst)
  (hB_on_parabola : B.snd^2 = 4 * B.fst)
  (h_product_slopes : (A.snd / A.fst) * (B.snd / B.fst) = 1) :
  (0 < (B.snd - A.snd) / (B.fst - A.fst) ∧ (B.snd - A.snd) / (B.fst - A.fst) < 1/2) := 
by
  sorry

end range_of_slopes_of_line_AB_l91_91519


namespace right_triangle_min_area_l91_91168

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l91_91168


namespace min_treasure_count_l91_91794

noncomputable def exists_truthful_sign : Prop :=
  ∃ (truthful: set ℕ), 
    truthful ⊆ {1, 2, 3, ..., 30} ∧ 
    (∀ t ∈ truthful, t = 15 ∨ t = 8 ∨ t = 4 ∨ t = 3) ∧
    (∀ t ∈ {1, 2, 3, ..., 30} \ truthful, 
       (if t = 15 then 15
        else if t = 8 then 8
        else if t = 4 then 4
        else if t = 3 then 3
        else 0) = 0)

theorem min_treasure_count : ∃ n, n = 15 ∧ exists_truthful_sign :=
sorry

end min_treasure_count_l91_91794


namespace tan_alpha_neg_one_third_l91_91397

theorem tan_alpha_neg_one_third
    (α : ℝ)
    (h : cos (↑(Real.pi / 4:ℝ) - α) / cos (↑(Real.pi / 4:ℝ) + α) = (1 / 2)) :
    tan α = -1 / 3 :=
sorry

end tan_alpha_neg_one_third_l91_91397


namespace cost_for_33_people_employees_for_14000_cost_l91_91188

-- Define the conditions for pricing
def price_per_ticket (x : Nat) : Int :=
  if x ≤ 30 then 400
  else max 280 (400 - 5 * (x - 30))

def total_cost (x : Nat) : Int :=
  x * price_per_ticket x

-- Problem Part 1: Proving the total cost for 33 people
theorem cost_for_33_people :
  total_cost 33 = 12705 :=
by
  sorry

-- Problem Part 2: Given a total cost of 14000, finding the number of employees
theorem employees_for_14000_cost :
  ∃ x : Nat, total_cost x = 14000 ∧ price_per_ticket x ≥ 280 :=
by
  sorry

end cost_for_33_people_employees_for_14000_cost_l91_91188


namespace slices_per_large_pizza_l91_91788

structure PizzaData where
  total_pizzas : Nat
  small_pizzas : Nat
  medium_pizzas : Nat
  slices_per_small : Nat
  slices_per_medium : Nat
  total_slices : Nat

def large_slices (data : PizzaData) : Nat := (data.total_slices - (data.small_pizzas * data.slices_per_small + data.medium_pizzas * data.slices_per_medium)) / (data.total_pizzas - data.small_pizzas - data.medium_pizzas)

def PizzaSlicingConditions := {data : PizzaData // 
  data.total_pizzas = 15 ∧
  data.small_pizzas = 4 ∧
  data.medium_pizzas = 5 ∧
  data.slices_per_small = 6 ∧
  data.slices_per_medium = 8 ∧
  data.total_slices = 136}

theorem slices_per_large_pizza (data : PizzaSlicingConditions) : large_slices data.val = 12 :=
by
  sorry

end slices_per_large_pizza_l91_91788


namespace units_digit_of_sum_of_sequence_l91_91178

theorem units_digit_of_sum_of_sequence :
  let sequence_sum := (1! + 1) + (2! + 2) + (3! + 3) + (4! + 4) + (5! + 5) + (6! + 6) +
                      (7! + 7) + (8! + 8) + (9! + 9) + (10! + 10)
  in sequence_sum % 10 = 8 :=
by {
  -- Proof is omitted
  sorry
}

end units_digit_of_sum_of_sequence_l91_91178


namespace divisible_by_6_l91_91768

theorem divisible_by_6 (n : ℕ) : 6 ∣ ((n - 1) * n * (n^3 + 1)) := sorry

end divisible_by_6_l91_91768


namespace magnitude_difference_l91_91831

noncomputable
def vector_a : ℝ × ℝ := (Real.cos (15 * Real.pi / 180), Real.sin (15 * Real.pi / 180))
noncomputable
def vector_b : ℝ × ℝ := (Real.cos (75 * Real.pi / 180), Real.sin (75 * Real.pi / 180))

noncomputable
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem magnitude_difference :
  magnitude (vector_a - (2 : ℝ) • vector_b) = Real.sqrt 3 :=
by
  sorry

end magnitude_difference_l91_91831


namespace fraction_of_products_inspected_jane_l91_91767

theorem fraction_of_products_inspected_jane 
  (P : ℝ) 
  (J : ℝ) 
  (John_rejection_rate : ℝ) 
  (Jane_rejection_rate : ℝ)
  (Total_rejection_rate : ℝ) 
  (hJohn : John_rejection_rate = 0.005) 
  (hJane : Jane_rejection_rate = 0.008) 
  (hTotal : Total_rejection_rate = 0.0075) 
  : J = 5 / 6 := by
{
  sorry
}

end fraction_of_products_inspected_jane_l91_91767


namespace gcd_of_g_y_l91_91240

def g (y : ℕ) : ℕ := (3 * y + 4) * (8 * y + 3) * (11 * y + 5) * (y + 11)

theorem gcd_of_g_y (y : ℕ) (hy : ∃ k, y = 30492 * k) : Nat.gcd (g y) y = 660 :=
by
  sorry

end gcd_of_g_y_l91_91240


namespace smallest_area_right_triangle_l91_91135

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l91_91135


namespace budget_remaining_l91_91782

noncomputable def solve_problem : Nat :=
  let total_budget := 325
  let cost_flasks := 150
  let cost_test_tubes := (2 / 3 : ℚ) * cost_flasks
  let cost_safety_gear := (1 / 2 : ℚ) * cost_test_tubes
  let total_expenses := cost_flasks + cost_test_tubes + cost_safety_gear
  total_budget - total_expenses

theorem budget_remaining : solve_problem = 25 := by
  sorry

end budget_remaining_l91_91782


namespace pentagon_area_50_l91_91725

def point := (ℝ × ℝ)

structure Pentagon :=
(A B C D E : point)

def area_rectangle (p1 p2 p3 p4 : point) : ℝ :=
let ⟨x1, y1⟩ := p1 in
let ⟨x2, y2⟩ := p2 in
let ⟨x3, y3⟩ := p3 in
let ⟨x4, y4⟩ := p4 in
abs((x3 - x1) * (y2 - y1))

def area_triangle (p1 p2 p3 : point) : ℝ :=
let ⟨x1, y1⟩ := p1 in
let ⟨x2, y2⟩ := p2 in
let ⟨x3, y3⟩ := p3 in
abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

def y_coordinate_C (pent: Pentagon) : ℝ :=
let ⟨_, _, pC, _, _⟩ := pent in
pC.2

theorem pentagon_area_50 (h : ℝ) :
  let A := (0, 0) in
  let B := (0, 5) in
  let C := (3, h) in
  let D := (6, 5) in
  let E := (6, 0) in
  let rect_area := area_rectangle A B D E in
  let tri_area := area_triangle B C D in
  rect_area + tri_area = 50 :=
by
  sorry

end pentagon_area_50_l91_91725


namespace repair_cost_total_l91_91853

def hourly_labor_cost : ℝ := 75
def labor_hours : ℝ := 16
def part_cost : ℝ := 1200
def labor_cost : ℝ := hourly_labor_cost * labor_hours
def total_cost : ℝ := labor_cost + part_cost

theorem repair_cost_total : total_cost = 2400 := 
by
  -- Proof omitted
  sorry

end repair_cost_total_l91_91853


namespace compare_xyz_l91_91382

open Real

noncomputable def x : ℝ := 6 * log 3 / log 64
noncomputable def y : ℝ := (1 / 3) * log 64 / log 3
noncomputable def z : ℝ := (3 / 2) * log 3 / log 8

theorem compare_xyz : x > y ∧ y > z := 
by {
  sorry
}

end compare_xyz_l91_91382


namespace probability_value_l91_91052

noncomputable def P (k : ℕ) (c : ℚ) : ℚ := c / (k * (k + 1))

theorem probability_value (c : ℚ) (h : P 1 c + P 2 c + P 3 c + P 4 c = 1) : P 1 c + P 2 c = 5 / 6 := 
by
  sorry

end probability_value_l91_91052


namespace mat_weavers_proof_l91_91328

def mat_weavers_rate
  (num_weavers_1 : ℕ) (num_mats_1 : ℕ) (num_days_1 : ℕ)
  (num_mats_2 : ℕ) (num_days_2 : ℕ) : ℕ :=
  let rate_per_weaver_per_day := num_mats_1 / (num_weavers_1 * num_days_1)
  let num_weavers_2 := num_mats_2 / (rate_per_weaver_per_day * num_days_2)
  num_weavers_2

theorem mat_weavers_proof :
  mat_weavers_rate 4 4 4 36 12 = 12 := by
  sorry

end mat_weavers_proof_l91_91328


namespace base8_subtraction_and_conversion_l91_91217

-- Define the base 8 numbers
def num1 : ℕ := 7463 -- 7463 in base 8
def num2 : ℕ := 3254 -- 3254 in base 8

-- Define the subtraction in base 8 and conversion to base 10
def result_base8 : ℕ := 4207 -- Expected result in base 8
def result_base10 : ℕ := 2183 -- Expected result in base 10

-- Helper function to convert from base 8 to base 10
def convert_base8_to_base10 (n : ℕ) : ℕ := 
  (n / 1000) * 8^3 + ((n / 100) % 10) * 8^2 + ((n / 10) % 10) * 8 + (n % 10)
 
-- Main theorem statement
theorem base8_subtraction_and_conversion :
  (num1 - num2 = result_base8) ∧ (convert_base8_to_base10 result_base8 = result_base10) :=
by
  sorry

end base8_subtraction_and_conversion_l91_91217


namespace jacket_price_is_48_l91_91915

-- Definitions according to the conditions
def jacket_problem (P S D : ℝ) : Prop :=
  S = P + 0.40 * S ∧
  D = 0.80 * S ∧
  16 = D - P

-- Statement of the theorem
theorem jacket_price_is_48 :
  ∃ P S D, jacket_problem P S D ∧ P = 48 :=
by
  sorry

end jacket_price_is_48_l91_91915


namespace day_of_week_50th_day_of_year_N_minus_1_l91_91979

def day_of_week (d : ℕ) (first_day : ℕ) : ℕ :=
  (first_day + d - 1) % 7

theorem day_of_week_50th_day_of_year_N_minus_1 
  (N : ℕ) 
  (day_250_N : ℕ) 
  (day_150_N_plus_1 : ℕ) 
  (h1 : day_250_N = 3)  -- 250th day of year N is Wednesday (3rd day of week, 0 = Sunday)
  (h2 : day_150_N_plus_1 = 3) -- 150th day of year N+1 is also Wednesday (3rd day of week, 0 = Sunday)
  : day_of_week 50 (day_of_week 1 ((day_of_week 1 day_250_N - 1 + 250) % 365 - 1 + 366)) = 6 := 
sorry

-- Explanation:
-- day_of_week function calculates the day of the week given the nth day of the year and the first day of the year.
-- Given conditions that 250th day of year N and 150th day of year N+1 are both Wednesdays (represented by 3 assuming Sunday = 0).
-- We need to derive that the 50th day of year N-1 is a Saturday (represented by 6 assuming Sunday = 0).

end day_of_week_50th_day_of_year_N_minus_1_l91_91979


namespace profit_sharing_l91_91789

theorem profit_sharing
  (A_investment B_investment C_investment total_profit : ℕ)
  (A_share : ℕ)
  (ratio_A ratio_B ratio_C : ℕ)
  (hA : A_investment = 6300)
  (hB : B_investment = 4200)
  (hC : C_investment = 10500)
  (hShare : A_share = 3810)
  (hRatio : ratio_A = 3 ∧ ratio_B = 2 ∧ ratio_C = 5)
  (hTotRatio : ratio_A + ratio_B + ratio_C = 10)
  (hShareCalc : A_share = (3/10) * total_profit) :
  total_profit = 12700 :=
sorry

end profit_sharing_l91_91789


namespace oranges_count_l91_91722

noncomputable def initial_oranges (O : ℕ) : Prop :=
  let apples := 14
  let blueberries := 6
  let remaining_fruits := 26
  13 + (O - 1) + 5 = remaining_fruits

theorem oranges_count (O : ℕ) (h : initial_oranges O) : O = 9 :=
by
  have eq : 13 + (O - 1) + 5 = 26 := h
  -- Simplify the equation to find O
  sorry

end oranges_count_l91_91722


namespace smallest_area_right_triangle_l91_91139

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l91_91139


namespace min_treasures_buried_l91_91796

-- Define the problem conditions
def Trees := ℕ
def Signs := ℕ

structure PalmTrees where
  total_trees : Trees
  trees_with_15_signs : Trees
  trees_with_8_signs : Trees
  trees_with_4_signs : Trees
  trees_with_3_signs : Trees

def condition (p: PalmTrees) : Prop :=
  p.total_trees = 30 ∧
  p.trees_with_15_signs = 15 ∧
  p.trees_with_8_signs = 8 ∧
  p.trees_with_4_signs = 4 ∧ 
  p.trees_with_3_signs = 3

def truthful_sign (buried_signs : Signs) (pt : PalmTrees) : Prop :=
  if buried_signs = 15 then pt.trees_with_15_signs = 0 else 
  if buried_signs = 8 then pt.trees_with_8_signs = 0 else 
  if buried_signs = 4 then pt.trees_with_4_signs = 0 else 
  if buried_signs = 3 then pt.trees_with_3_signs = 0 else 
  true

-- The theorem to prove
theorem min_treasures_buried (p : PalmTrees) (buried_signs : Signs) :
  condition p → truthful_sign buried_signs p → 
  buried_signs = 15 :=
by
  intros _ _
  sorry

end min_treasures_buried_l91_91796


namespace smallest_area_right_triangle_l91_91172

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l91_91172


namespace unique_function_l91_91499

theorem unique_function (f : ℝ → ℝ) 
  (H : ∀ (x y : ℝ), f (f x + 9 * y) = f y + 9 * x + 24 * y) : 
  ∀ x : ℝ, f x = 3 * x :=
by 
  sorry

end unique_function_l91_91499


namespace expression_value_l91_91662

theorem expression_value :
  2 - (-3) - 4 - (-5) - 6 - (-7) * 2 = 14 :=
by sorry

end expression_value_l91_91662


namespace smallest_area_right_triangle_l91_91155

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l91_91155


namespace petya_maximum_margin_l91_91560

def max_margin_votes (total_votes first_period_margin last_period_margin : ℕ) (petya_vasaya_margin : ℕ) :=
  ∀ (P1 P2 V1 V2 : ℕ),
    (P1 + P2 + V1 + V2 = total_votes) →
    (P1 = V1 + first_period_margin) →
    (V2 = P2 + last_period_margin) →
    (P1 + P2 > V1 + V2) →
    petya_vasaya_margin = P1 + P2 - (V1 + V2)

theorem petya_maximum_margin
  (total_votes first_period_margin last_period_margin : ℕ)
  (h_total_votes: total_votes = 27)
  (h_first_period_margin: first_period_margin = 9)
  (h_last_period_margin: last_period_margin = 9):
  ∃ (petya_vasaya_margin : ℕ), max_margin_votes total_votes first_period_margin last_period_margin petya_vasaya_margin ∧ petya_vasaya_margin = 9 :=
by {
    sorry
}

end petya_maximum_margin_l91_91560


namespace smallest_right_triangle_area_l91_91094

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l91_91094


namespace smallest_area_right_triangle_l91_91092

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l91_91092


namespace smallest_right_triangle_area_l91_91098

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l91_91098


namespace intersection_A_B_union_B_C_eq_B_iff_l91_91530

-- Definitions for the sets A, B, and C
def setA : Set ℝ := { x | x^2 - 3 * x < 0 }
def setB : Set ℝ := { x | (x + 2) * (4 - x) ≥ 0 }
def setC (a : ℝ) : Set ℝ := { x | a < x ∧ x ≤ a + 1 }

-- Proving that A ∩ B = { x | 0 < x < 3 }
theorem intersection_A_B : setA ∩ setB = { x : ℝ | 0 < x ∧ x < 3 } :=
sorry

-- Proving that B ∪ C = B implies the range of a is [-2, 3]
theorem union_B_C_eq_B_iff (a : ℝ) : (setB ∪ setC a = setB) ↔ (-2 ≤ a ∧ a ≤ 3) :=
sorry

end intersection_A_B_union_B_C_eq_B_iff_l91_91530


namespace smallest_prime_divisor_of_sum_l91_91757

theorem smallest_prime_divisor_of_sum (a b : ℕ) 
  (h₁ : a = 3 ^ 15) 
  (h₂ : b = 11 ^ 21) 
  (h₃ : odd a) 
  (h₄ : odd b) : 
  nat.prime_divisors (a + b) = [2] := 
by
  sorry

end smallest_prime_divisor_of_sum_l91_91757


namespace equal_share_of_candles_l91_91196

-- Define conditions
def ambika_candles : ℕ := 4
def aniyah_candles : ℕ := 6 * ambika_candles
def bree_candles : ℕ := 2 * aniyah_candles
def caleb_candles : ℕ := bree_candles + (bree_candles / 2)

-- Define the total candles and the equal share
def total_candles : ℕ := ambika_candles + aniyah_candles + bree_candles + caleb_candles
def each_share : ℕ := total_candles / 4

-- State the problem
theorem equal_share_of_candles : each_share = 37 := by
  sorry

end equal_share_of_candles_l91_91196


namespace regular_tetrahedron_height_eq_4r_l91_91823

noncomputable def equilateral_triangle_inscribed_circle_height (r : ℝ) : ℝ :=
3 * r

noncomputable def regular_tetrahedron_inscribed_sphere_height (r : ℝ) : ℝ :=
4 * r

theorem regular_tetrahedron_height_eq_4r (r : ℝ) :
  regular_tetrahedron_inscribed_sphere_height r = 4 * r :=
by
  unfold regular_tetrahedron_inscribed_sphere_height
  sorry

end regular_tetrahedron_height_eq_4r_l91_91823


namespace length_of_DC_l91_91845

noncomputable def AB : ℝ := 30
noncomputable def sine_A : ℝ := 4 / 5
noncomputable def sine_C : ℝ := 1 / 4
noncomputable def angle_ADB : ℝ := Real.pi / 2

theorem length_of_DC (h_AB : AB = 30) (h_sine_A : sine_A = 4 / 5) (h_sine_C : sine_C = 1 / 4) (h_angle_ADB : angle_ADB = Real.pi / 2) :
  ∃ DC : ℝ, DC = 24 * Real.sqrt 15 :=
by sorry

end length_of_DC_l91_91845


namespace kite_ratio_equality_l91_91267

-- Definitions for points, lines, and conditions in the geometric setup
variables {Point : Type*} [MetricSpace Point]

-- Assuming A, B, C, D, P, E, F, G, H, I, J are points
variable (A B C D P E F G H I J : Point)

-- Conditions based on the problem
variables (AB_eq_AD : dist A B = dist A D)
          (BC_eq_CD : dist B C = dist C D)
          (on_BD : P ∈ line B D)
          (line_PE_inter_AD : E ∈ line P E ∧ E ∈ line A D)
          (line_PF_inter_BC : F ∈ line P F ∧ F ∈ line B C)
          (line_PG_inter_AB : G ∈ line P G ∧ G ∈ line A B)
          (line_PH_inter_CD : H ∈ line P H ∧ H ∈ line C D)
          (GF_inter_BD_at_I : I ∈ line G F ∧ I ∈ line B D)
          (EH_inter_BD_at_J : J ∈ line E H ∧ J ∈ line B D)

-- The statement to prove
theorem kite_ratio_equality :
  dist P I / dist P B = dist P J / dist P D := sorry

end kite_ratio_equality_l91_91267


namespace integer_solutions_range_l91_91228

def operation (p q : ℝ) : ℝ := p + q - p * q

theorem integer_solutions_range (m : ℝ) :
  (∃ (x1 x2 : ℤ), (operation 2 x1 > 0) ∧ (operation x1 3 ≤ m) ∧ (operation 2 x2 > 0) ∧ (operation x2 3 ≤ m) ∧ (x1 ≠ x2)) ↔ (3 ≤ m ∧ m < 5) :=
by sorry

end integer_solutions_range_l91_91228


namespace clive_change_l91_91203

theorem clive_change (money : ℝ) (olives_needed : ℕ) (olives_per_jar : ℕ) (price_per_jar : ℝ) : 
  (money = 10) → 
  (olives_needed = 80) → 
  (olives_per_jar = 20) →
  (price_per_jar = 1.5) →
  money - (olives_needed / olives_per_jar) * price_per_jar = 4 := by
  sorry

end clive_change_l91_91203


namespace smallest_area_of_right_triangle_l91_91103

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l91_91103


namespace smallest_area_right_triangle_l91_91091

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l91_91091


namespace hemming_time_l91_91423

/-- Prove that the time it takes Jenna to hem her dress is 6 minutes given:
1. The dress's hem is 3 feet long.
2. Each stitch Jenna makes is 1/4 inch long.
3. Jenna makes 24 stitches per minute.
-/
theorem hemming_time (dress_length_feet : ℝ) (stitch_length_inches : ℝ) (stitches_per_minute : ℝ)
  (h1 : dress_length_feet = 3)
  (h2 : stitch_length_inches = 1/4)
  (h3 : stitches_per_minute = 24) : 
  let dress_length_inches := dress_length_feet * 12,
      total_stitches := dress_length_inches / stitch_length_inches,
      hemming_time := total_stitches / stitches_per_minute
  in hemming_time = 6 := 
sorry

end hemming_time_l91_91423


namespace arithmetic_sequence_tenth_term_l91_91951

/- 
  Define the arithmetic sequence in terms of its properties 
  and prove that the 10th term is 18.
-/

theorem arithmetic_sequence_tenth_term (a : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_a2 : a 2 = 2) (h_a5 : a 5 = 8) : a 10 = 18 := 
by 
  sorry

end arithmetic_sequence_tenth_term_l91_91951


namespace solve_equation_l91_91672

theorem solve_equation :
  {x : ℝ | (15 * x - x^2)/(x + 2) * (x + (15 - x)/(x + 2)) = 54 } = {12, -3, -3 + real.sqrt 33, -3 - real.sqrt 33} :=
by
  sorry

end solve_equation_l91_91672


namespace combined_distance_l91_91301

noncomputable def radius_wheel1 : ℝ := 22.4
noncomputable def revolutions_wheel1 : ℕ := 750

noncomputable def radius_wheel2 : ℝ := 15.8
noncomputable def revolutions_wheel2 : ℕ := 950

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

noncomputable def distance_covered (r : ℝ) (rev : ℕ) : ℝ := circumference r * rev

theorem combined_distance :
  distance_covered radius_wheel1 revolutions_wheel1 + distance_covered radius_wheel2 revolutions_wheel2 = 199896.96 := by
  sorry

end combined_distance_l91_91301


namespace initial_mixture_l91_91912

theorem initial_mixture (M : ℝ) (h1 : 0.20 * M + 20 = 0.36 * (M + 20)) : 
  M = 80 :=
by
  sorry

end initial_mixture_l91_91912


namespace Xiaoming_speed_l91_91180

theorem Xiaoming_speed (x xiaohong_speed_xiaoming_diff : ℝ) :
  (50 * (2 * x + 2) = 600) →
  (xiaohong_speed_xiaoming_diff = 2) →
  x + xiaohong_speed_xiaoming_diff = 7 :=
by
  intros h₁ h₂
  sorry

end Xiaoming_speed_l91_91180


namespace equivalent_single_discount_l91_91472

noncomputable def original_price : ℝ := 50
noncomputable def first_discount : ℝ := 0.30
noncomputable def second_discount : ℝ := 0.15
noncomputable def third_discount : ℝ := 0.10

theorem equivalent_single_discount :
  let discount_1 := original_price * (1 - first_discount)
  let discount_2 := discount_1 * (1 - second_discount)
  let discount_3 := discount_2 * (1 - third_discount)
  let final_price := discount_3
  (1 - (final_price / original_price)) = 0.4645 :=
by
  let discount_1 := original_price * (1 - first_discount)
  let discount_2 := discount_1 * (1 - second_discount)
  let discount_3 := discount_2 * (1 - third_discount)
  let final_price := discount_3
  sorry

end equivalent_single_discount_l91_91472


namespace smallest_n_satisfying_conditions_l91_91369

def contains_digit (num : ℕ) (d : ℕ) : Prop :=
  d ∈ num.digits 10

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, contains_digit (n^2) 7 ∧ contains_digit ((n+1)^2) 7 ∧ ¬ contains_digit ((n+2)^2) 7 ∧ ∀ m : ℕ, m < n → ¬ (contains_digit (m^2) 7 ∧ contains_digit ((m+1)^2) 7 ∧ ¬ contains_digit ((m+2)^2) 7) := 
sorry

end smallest_n_satisfying_conditions_l91_91369


namespace smallest_area_right_triangle_l91_91175

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l91_91175


namespace proof_a_eq_b_pow_n_l91_91607

theorem proof_a_eq_b_pow_n 
  (a b n : ℕ) 
  (h : ∀ k : ℕ, k ≠ b → b - k ∣ a - k^n) : a = b^n := 
by 
  sorry

end proof_a_eq_b_pow_n_l91_91607


namespace area_of_inscribed_square_l91_91695

theorem area_of_inscribed_square (D : ℝ) (h : D = 10) : 
  ∃ A : ℝ, A = 50 :=
by
  sorry

end area_of_inscribed_square_l91_91695


namespace two_integer_solutions_iff_m_l91_91229

def op (p q : ℝ) : ℝ := p + q - p * q

theorem two_integer_solutions_iff_m (m : ℝ) :
  (∃ x1 x2 : ℤ, x1 ≠ x2 ∧ op 2 x1 > 0 ∧ op x1 3 ≤ m ∧ op 2 x2 > 0 ∧ op x2 3 ≤ m) ↔ 3 ≤ m ∧ m < 5 :=
by
  sorry

end two_integer_solutions_iff_m_l91_91229


namespace petya_maximum_margin_l91_91557

def max_margin_votes (total_votes first_period_margin last_period_margin : ℕ) (petya_vasaya_margin : ℕ) :=
  ∀ (P1 P2 V1 V2 : ℕ),
    (P1 + P2 + V1 + V2 = total_votes) →
    (P1 = V1 + first_period_margin) →
    (V2 = P2 + last_period_margin) →
    (P1 + P2 > V1 + V2) →
    petya_vasaya_margin = P1 + P2 - (V1 + V2)

theorem petya_maximum_margin
  (total_votes first_period_margin last_period_margin : ℕ)
  (h_total_votes: total_votes = 27)
  (h_first_period_margin: first_period_margin = 9)
  (h_last_period_margin: last_period_margin = 9):
  ∃ (petya_vasaya_margin : ℕ), max_margin_votes total_votes first_period_margin last_period_margin petya_vasaya_margin ∧ petya_vasaya_margin = 9 :=
by {
    sorry
}

end petya_maximum_margin_l91_91557


namespace james_total_catch_l91_91577

def pounds_of_trout : ℕ := 200
def pounds_of_salmon : ℕ := pounds_of_trout + (pounds_of_trout / 2)
def pounds_of_tuna : ℕ := 2 * pounds_of_salmon
def total_pounds_of_fish : ℕ := pounds_of_trout + pounds_of_salmon + pounds_of_tuna

theorem james_total_catch : total_pounds_of_fish = 1100 := by
  sorry

end james_total_catch_l91_91577


namespace least_number_four_digits_divisible_by_15_25_40_75_l91_91910

noncomputable def least_four_digit_multiple : ℕ :=
  1200

theorem least_number_four_digits_divisible_by_15_25_40_75 :
  (∀ n, (n ∣ 15) ∧ (n ∣ 25) ∧ (n ∣ 40) ∧ (n ∣ 75)) → least_four_digit_multiple = 1200 :=
sorry

end least_number_four_digits_divisible_by_15_25_40_75_l91_91910


namespace find_x_range_l91_91838

-- Define the condition for the expression to be meaningful
def meaningful_expr (x : ℝ) : Prop := x - 3 ≥ 0

-- The range of values for x is equivalent to x being at least 3
theorem find_x_range (x : ℝ) : meaningful_expr x ↔ x ≥ 3 := by
  sorry

end find_x_range_l91_91838


namespace max_y_difference_l91_91934

theorem max_y_difference : (∃ x, (5 - 2 * x^2 + 2 * x^3 = 1 + x^2 + x^3)) ∧ 
                           (∀ y1 y2, y1 = 5 - 2 * (2^2) + 2 * (2^3) ∧ y2 = 5 - 2 * (1/2)^2 + 2 * (1/2)^3 → 
                           (y1 - y2 = 11.625)) := sorry

end max_y_difference_l91_91934


namespace tom_spend_l91_91618

def theater_cost (seat_count : ℕ) (sqft_per_seat : ℕ) (cost_per_sqft : ℕ) (construction_multiplier : ℕ) (partner_percentage : ℝ) : ℝ :=
  let total_sqft := seat_count * sqft_per_seat
  let land_cost := total_sqft * cost_per_sqft
  let construction_cost := construction_multiplier * land_cost
  let total_cost := land_cost + construction_cost
  let partner_contribution := partner_percentage * (total_cost : ℝ)
  total_cost - partner_contribution

theorem tom_spend (partner_percentage : ℝ) :
  theater_cost 500 12 5 2 partner_percentage = 54000 :=
sorry

end tom_spend_l91_91618


namespace clive_change_l91_91202

theorem clive_change (money : ℝ) (olives_needed : ℕ) (olives_per_jar : ℕ) (price_per_jar : ℝ) : 
  (money = 10) → 
  (olives_needed = 80) → 
  (olives_per_jar = 20) →
  (price_per_jar = 1.5) →
  money - (olives_needed / olives_per_jar) * price_per_jar = 4 := by
  sorry

end clive_change_l91_91202


namespace sparre_andersen_theorem_l91_91585

-- Define the problem context and constants
noncomputable def identically_distributed_random_variables {N : ℕ} : Type :=
  fin N → ℝ

-- Define sequences S_i using identically distributed random variables ξ_i
def S {N : ℕ} (ξ : identically_distributed_random_variables) (i : fin N) : ℝ :=
  if i = 0 then 0 else list.sum (list.of_fn (λ j : fin i, ξ j))

-- Function to count positive terms in a sequence
def N_n {N : ℕ} (ξ : identically_distributed_random_variables) (n : ℕ) : ℕ :=
  list.sum (list.of_fn (λ k : fin n, if S ξ k > 0 then 1 else 0))

-- The main theorem to be proven: Sparre Andersen's theorem
theorem sparre_andersen_theorem {N n k : ℕ} (ξ : identically_distributed_random_variables) (hkn : 0 ≤ k ∧ k ≤ n) :
  probability (λ (ξ : identically_distributed_random_variables), N_n ξ n = k) =
  probability (λ (ξ : identically_distributed_random_variables), N_n ξ k = k) * 
  probability (λ (ξ : identically_distributed_random_variables), N_n ξ (n - k) = 0) := by
  sorry

end sparre_andersen_theorem_l91_91585


namespace find_f_m_l91_91684

-- Definitions based on the conditions
def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 3

axiom condition (m a : ℝ) : f (-m) a = 1

-- The statement to be proven
theorem find_f_m (m a : ℝ) (hm : f (-m) a = 1) : f m a = 5 := 
by sorry

end find_f_m_l91_91684


namespace maddie_total_payment_l91_91428

def price_palettes : ℝ := 15
def num_palettes : ℕ := 3
def discount_palettes : ℝ := 0.20
def price_lipsticks : ℝ := 2.50
def num_lipsticks_bought : ℕ := 4
def num_lipsticks_pay : ℕ := 3
def price_hair_color : ℝ := 4
def num_hair_color : ℕ := 3
def discount_hair_color : ℝ := 0.10
def sales_tax_rate : ℝ := 0.08

def total_cost_palettes : ℝ := num_palettes * price_palettes
def total_cost_palettes_after_discount : ℝ := total_cost_palettes * (1 - discount_palettes)

def total_cost_lipsticks : ℝ := num_lipsticks_pay * price_lipsticks

def total_cost_hair_color : ℝ := num_hair_color * price_hair_color
def total_cost_hair_color_after_discount : ℝ := total_cost_hair_color * (1 - discount_hair_color)

def total_pre_tax : ℝ := total_cost_palettes_after_discount + total_cost_lipsticks + total_cost_hair_color_after_discount
def total_sales_tax : ℝ := total_pre_tax * sales_tax_rate
def total_cost : ℝ := total_pre_tax + total_sales_tax

theorem maddie_total_payment : total_cost = 58.64 := by
  sorry

end maddie_total_payment_l91_91428


namespace max_min_f_in_rectangle_l91_91222

def f (x y : ℝ) : ℝ := x^3 + y^3 + 6 * x * y

def in_rectangle (x y : ℝ) : Prop := 
  (-3 ≤ x ∧ x ≤ 1) ∧ (-3 ≤ y ∧ y ≤ 2)

theorem max_min_f_in_rectangle :
  ∃ (x_max y_max x_min y_min : ℝ),
    in_rectangle x_max y_max ∧ in_rectangle x_min y_min ∧
    (∀ x y, in_rectangle x y → f x y ≤ f x_max y_max) ∧
    (∀ x y, in_rectangle x y → f x_min y_min ≤ f x y) ∧
    f x_max y_max = 21 ∧ f x_min y_min = -55 :=
by
  sorry

end max_min_f_in_rectangle_l91_91222


namespace chess_team_girls_l91_91914

theorem chess_team_girls (B G : ℕ) (h1 : B + G = 26) (h2 : (G / 2) + B = 16) : G = 20 := by
  sorry

end chess_team_girls_l91_91914


namespace double_sum_evaluation_l91_91030

noncomputable def f (m n : ℕ) : ℕ := 3 * m + n + (m + n) ^ 2

theorem double_sum_evaluation :
  (∑ m:ℕ, ∑ n:ℕ, (2:ℝ) ^ -((f m n) : ℝ)) = (4 / 3 : ℝ) :=
begin
  sorry
end

end double_sum_evaluation_l91_91030


namespace min_value_l91_91954

-- Conditions
variables {x y : ℝ}
variable (hx : x > 0)
variable (hy : y > 0)
variable (hxy : x + y = 2)

-- Theorem
theorem min_value (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  ∃ x y, (x > 0) ∧ (y > 0) ∧ (x + y = 2) ∧ (1/x + 4/y = 9/2) := 
by
  sorry

end min_value_l91_91954


namespace find_certain_number_l91_91299

theorem find_certain_number (x : ℕ) 
  (h1 : (28 + x + 42 + 78 + 104) / 5 = 62) 
  (h2 : (48 + 62 + 98 + 124 + x) / 5 = 78) : 
  x = 58 := by
  sorry

end find_certain_number_l91_91299


namespace number_division_equals_value_l91_91017

theorem number_division_equals_value (x : ℝ) (h : x / 0.144 = 14.4 / 0.0144) : x = 144 :=
by
  sorry

end number_division_equals_value_l91_91017


namespace remainder_492381_div_6_l91_91459

theorem remainder_492381_div_6 : 492381 % 6 = 3 := 
by
  sorry

end remainder_492381_div_6_l91_91459


namespace guests_did_not_respond_l91_91430

theorem guests_did_not_respond (n : ℕ) (p_yes p_no : ℝ) (hn : n = 200)
    (hp_yes : p_yes = 0.83) (hp_no : p_no = 0.09) : 
    n - (n * p_yes + n * p_no) = 16 :=
by sorry

end guests_did_not_respond_l91_91430


namespace common_chord_of_circles_l91_91219

theorem common_chord_of_circles
  (x y : ℝ)
  (h1 : x^2 + y^2 + 2 * x = 0)
  (h2 : x^2 + y^2 - 4 * y = 0)
  : x + 2 * y = 0 := 
by
  -- Lean will check the logical consistency of the statement.
  sorry

end common_chord_of_circles_l91_91219


namespace probability_of_at_least_two_threes_l91_91817

def balls : finset ℕ := {1, 2, 3, 4}

def draws : fin 4 → ℕ := sorry /- Function from the index of draws to the ball number drawn, needs defining -/

def sum_condition (r : vector ℕ 4) : Prop := r.sum = 10

def at_least_two_threes (r : vector ℕ 4) : Prop := (r.filter (λ x => x = 3)).length ≥ 2

theorem probability_of_at_least_two_threes :
  (probability (at_least_two_threes) (sum_condition)) = 1 / 7 :=
sorry

end probability_of_at_least_two_threes_l91_91817


namespace smallest_prime_divisor_of_sum_l91_91758

theorem smallest_prime_divisor_of_sum : ∃ p : ℕ, Prime p ∧ p = 2 ∧ p ∣ (3 ^ 15 + 11 ^ 21) :=
by
  sorry

end smallest_prime_divisor_of_sum_l91_91758


namespace rotated_square_height_l91_91747

noncomputable def height_of_B (side_length : ℝ) (rotation_angle : ℝ) : ℝ :=
  let diagonal := side_length * Real.sqrt 2
  let vertical_component := diagonal * Real.sin rotation_angle
  vertical_component

theorem rotated_square_height :
  height_of_B 1 (Real.pi / 6) = Real.sqrt 2 / 2 :=
by
  sorry

end rotated_square_height_l91_91747


namespace cutoff_score_admission_l91_91786

theorem cutoff_score_admission (x : ℝ) 
  (h1 : (2 / 5) * (x + 15) + (3 / 5) * (x - 20) = 90) : x = 96 :=
sorry

end cutoff_score_admission_l91_91786


namespace right_triangle_perimeter_l91_91438

theorem right_triangle_perimeter
  (a b c : ℝ)
  (h_right: a^2 + b^2 = c^2)
  (h_area: (1/2) * a * b = (1/2) * c) :
  a + b + c = 2 * (Real.sqrt 2 + 1) :=
sorry

end right_triangle_perimeter_l91_91438


namespace sum_of_segments_AK_KB_eq_AB_l91_91197

-- Given conditions: length of segment AB is 9 cm
def length_AB : ℝ := 9

-- For any point K on segment AB, prove that AK + KB = AB
theorem sum_of_segments_AK_KB_eq_AB (K : ℝ) (h : 0 ≤ K ∧ K ≤ length_AB) : 
  K + (length_AB - K) = length_AB := by
  sorry

end sum_of_segments_AK_KB_eq_AB_l91_91197


namespace total_revenue_full_price_tickets_l91_91471

theorem total_revenue_full_price_tickets (f q : ℕ) (p : ℝ) :
  f + q = 170 ∧ f * p + q * (p / 4) = 2917 → f * p = 1748 := by
  sorry

end total_revenue_full_price_tickets_l91_91471


namespace smallest_base_b_l91_91901

theorem smallest_base_b (b : ℕ) (n : ℕ) : b > 3 ∧ 3 * b + 4 = n ^ 2 → b = 4 := 
by
  sorry

end smallest_base_b_l91_91901


namespace largest_mersenne_prime_is_127_l91_91075

noncomputable def largest_mersenne_prime_less_than_500 : ℕ :=
  127

theorem largest_mersenne_prime_is_127 :
  ∃ p : ℕ, Nat.Prime p ∧ (2^p - 1) = largest_mersenne_prime_less_than_500 ∧ 2^p - 1 < 500 := 
by 
  -- The largest Mersenne prime less than 500 is 127
  use 7
  sorry

end largest_mersenne_prime_is_127_l91_91075


namespace inequality_solution_set_non_empty_l91_91609

theorem inequality_solution_set_non_empty (a : ℝ) :
  (∃ x : ℝ, a * x > -1 ∧ x + a > 0) ↔ a > -1 :=
sorry

end inequality_solution_set_non_empty_l91_91609


namespace difference_of_squirrels_and_nuts_l91_91745

-- Definitions
def number_of_squirrels : ℕ := 4
def number_of_nuts : ℕ := 2

-- Theorem statement with conditions and conclusion
theorem difference_of_squirrels_and_nuts : number_of_squirrels - number_of_nuts = 2 := by
  sorry

end difference_of_squirrels_and_nuts_l91_91745


namespace exists_k_composite_l91_91872

theorem exists_k_composite (h : Nat) : ∃ k : ℕ, ∀ n : ℕ, 0 < n → ∃ p : ℕ, Prime p ∧ p ∣ (k * 2 ^ n + 1) :=
by
  sorry

end exists_k_composite_l91_91872


namespace Petya_victory_margin_l91_91571

theorem Petya_victory_margin :
  ∃ P1 P2 V1 V2 : ℕ, P1 = V1 + 9 ∧ V2 = P2 + 9 ∧ P1 + P2 + V1 + V2 = 27 ∧ 
  (P1 + P2) > (V1 + V2) ∧ (P1 + P2) - (V1 + V2) = 9 :=
begin
  sorry
end

end Petya_victory_margin_l91_91571


namespace square_side_length_equals_5_sqrt_pi_l91_91338

theorem square_side_length_equals_5_sqrt_pi :
  ∃ s : ℝ, ∃ r : ℝ, (r = 5) ∧ (s = 2 * r) ∧ (s ^ 2 = 25 * π) ∧ (s = 5 * Real.sqrt π) :=
by
  sorry

end square_side_length_equals_5_sqrt_pi_l91_91338


namespace rectangle_width_length_ratio_l91_91546

theorem rectangle_width_length_ratio (w l P : ℕ) (hP : P = 30) (hl : l = 10) (h_perimeter : P = 2*l + 2*w) :
  w / l = 1 / 2 :=
by
  sorry

end rectangle_width_length_ratio_l91_91546


namespace ROI_diff_after_2_years_is_10_l91_91214

variables (investment_Emma : ℝ) (investment_Briana : ℝ)
variables (yield_Emma : ℝ) (yield_Briana : ℝ)
variables (years : ℝ)

def annual_ROI_Emma (investment_Emma yield_Emma : ℝ) : ℝ :=
  yield_Emma * investment_Emma

def annual_ROI_Briana (investment_Briana yield_Briana : ℝ) : ℝ :=
  yield_Briana * investment_Briana

def total_ROI_Emma (investment_Emma yield_Emma years : ℝ) : ℝ :=
  annual_ROI_Emma investment_Emma yield_Emma * years

def total_ROI_Briana (investment_Briana yield_Briana years : ℝ) : ℝ :=
  annual_ROI_Briana investment_Briana yield_Briana * years

def ROI_difference (investment_Emma investment_Briana yield_Emma yield_Briana years : ℝ) : ℝ :=
  total_ROI_Briana investment_Briana yield_Briana years - total_ROI_Emma investment_Emma yield_Emma years

theorem ROI_diff_after_2_years_is_10 :
  ROI_difference 300 500 0.15 0.10 2 = 10 :=
by
  sorry

end ROI_diff_after_2_years_is_10_l91_91214


namespace printer_Y_time_l91_91627

theorem printer_Y_time (T_y : ℝ) : 
    (12 * (1 / (1 / T_y + 1 / 20)) = 1.8) → T_y = 10 := 
by 
sorry

end printer_Y_time_l91_91627


namespace evaluate_m_l91_91671

theorem evaluate_m :
  ∀ m : ℝ, (243:ℝ)^(1/5) = 3^m → m = 1 :=
by
  intro m
  sorry

end evaluate_m_l91_91671


namespace num_allocation_schemes_l91_91738

theorem num_allocation_schemes : 
  let C(n k : ℕ) := n.choose k
  let A(n k : ℕ) := nat.factorial n / nat.factorial (n - k)
  in (C 6 2 * C 4 2 * C 2 1 * C 1 1) / (A 2 2 * A 2 2) * A 4 4 = 1080 :=
by
  let C := λ (n k : ℕ), n.choose k
  let A := λ (n k : ℕ), nat.factorial n / nat.factorial (n - k)
  have h1 : C 6 2 = 15 := by sorry
  have h2 : C 4 2 = 6 := by sorry
  have h3 : C 2 1 = 2 := by sorry
  have h4 : C 1 1 = 1 := by sorry
  have h5 : A 2 2 = 2 := by sorry
  have h6 : A 4 4 = 24 := by sorry
  calc (C 6 2 * C 4 2 * C 2 1 * C 1 1) / (A 2 2 * A 2 2) * A 4 4 = (15 * 6 * 2 * 1) / (2 * 2) * 24 : by sorry
  ... = (180) / (4) * 24 : by sorry
  ... = 45 * 24 : by sorry
  ... = 1080 : by sorry

end num_allocation_schemes_l91_91738


namespace smallest_area_of_right_triangle_l91_91165

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l91_91165


namespace quadratic_cubic_expression_l91_91029

theorem quadratic_cubic_expression
  (r s : ℝ)
  (h_eq : ∀ x : ℝ, 3 * x^2 - 4 * x - 12 = 0 → x = r ∨ x = s) :
  (9 * r^3 - 9 * s^3) / (r - s) = 52 :=
by 
  sorry

end quadratic_cubic_expression_l91_91029


namespace range_of_a_l91_91371

theorem range_of_a (m : ℝ) (a : ℝ) (hx : ∃ x : ℝ, mx^2 + x - m - a = 0) : -1 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l91_91371


namespace positive_whole_numbers_with_fourth_root_less_than_six_l91_91256

theorem positive_whole_numbers_with_fourth_root_less_than_six :
  {n : ℕ | n > 0 ∧ (n : ℝ)^(1/4) < 6}.to_finset.card = 1295 :=
sorry

end positive_whole_numbers_with_fourth_root_less_than_six_l91_91256


namespace min_value_m_plus_n_l91_91041

theorem min_value_m_plus_n (m n : ℕ) (h : 108 * m = n^3) (hm : 0 < m) (hn : 0 < n) : m + n = 8 :=
sorry

end min_value_m_plus_n_l91_91041


namespace ratio_sqrt5_over_5_l91_91703

noncomputable def radius_ratio (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) : ℝ :=
a / b

theorem ratio_sqrt5_over_5 (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) :
  radius_ratio a b h = 1 / Real.sqrt 5 := 
sorry

end ratio_sqrt5_over_5_l91_91703


namespace product_identity_l91_91761

theorem product_identity :
  (1 + 1 / Nat.factorial 1) * (1 + 1 / Nat.factorial 2) * (1 + 1 / Nat.factorial 3) *
  (1 + 1 / Nat.factorial 4) * (1 + 1 / Nat.factorial 5) * (1 + 1 / Nat.factorial 6) *
  (1 + 1 / Nat.factorial 7) = 5041 / 5040 := sorry

end product_identity_l91_91761


namespace rooms_needed_l91_91192

/-
  We are given that there are 30 students and each hotel room accommodates 5 students.
  Prove that the number of rooms required to accommodate all students is 6.
-/
theorem rooms_needed (total_students : ℕ) (students_per_room : ℕ) (h1 : total_students = 30) (h2 : students_per_room = 5) : total_students / students_per_room = 6 := by
  -- proof
  sorry

end rooms_needed_l91_91192


namespace A_and_D_independent_l91_91309

-- Definitions of the events based on given conditions
def event_A (x₁ : ℕ) : Prop := x₁ = 1
def event_B (x₂ : ℕ) : Prop := x₂ = 2
def event_C (x₁ x₂ : ℕ) : Prop := x₁ + x₂ = 8
def event_D (x₁ x₂ : ℕ) : Prop := x₁ + x₂ = 7

-- Probabilities based on uniform distribution and replacement
def probability_event (event : ℕ → ℕ → Prop) : ℚ :=
  if h : ∃ x₁ : ℕ, ∃ x₂ : ℕ, x₁ ∈ finset.range 1 7 ∧ x₂ ∈ finset.range 1 7 ∧ event x₁ x₂
  then ((finset.card (finset.filter (λ x, event x.1 x.2)
                (finset.product (finset.range 1 7) (finset.range 1 7)))) : ℚ) / 36
  else 0

noncomputable def P_A : ℚ := 1 / 6
noncomputable def P_D : ℚ := 1 / 6
noncomputable def P_A_and_D : ℚ := 1 / 36

-- Independence condition (by definition): P(A ∩ D) = P(A) * P(D)
theorem A_and_D_independent :
  P_A_and_D = P_A * P_D := by
  sorry

end A_and_D_independent_l91_91309


namespace smallest_area_right_triangle_l91_91131

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l91_91131


namespace repeating_decimals_sum_l91_91941

-- Define the repeating decimals as rational numbers
def dec_0_3 : ℚ := 1 / 3
def dec_0_02 : ℚ := 2 / 99
def dec_0_0004 : ℚ := 4 / 9999

-- State the theorem that we need to prove
theorem repeating_decimals_sum :
  dec_0_3 + dec_0_02 + dec_0_0004 = 10581 / 29889 :=
by
  sorry

end repeating_decimals_sum_l91_91941


namespace min_ones_in_20_row_matrix_l91_91186

theorem min_ones_in_20_row_matrix :
  ∃ (n m : ℕ), n = 20 ∧ 
  (∀ i j : ℕ, i < j → column_unique i j) ∧ 
  (∀ i j k l : ℕ, column_pair_constraint i j k l) ∧ 
  m = 3820 :=
begin
  let rows := 20,
  let max_columns := 1 + (nat.choose 20 1) + (nat.choose 20 2) + (nat.choose 20 3),
  have columns_unique : ∀ (i j : ℕ), i < j → true := sorry,
  have pairwise_constraint : ∀ (i j k l : ℕ), true := sorry,
  let num_1s := (nat.choose 20 1 * 1) + (nat.choose 20 2 * 2) + (nat.choose 20 3 * 3),
  use [rows, num_1s],
  split,
  { refl, },
  split,
  { apply columns_unique, },
  split,
  { apply pairwise_constraint, },
  { refl, },
end

end min_ones_in_20_row_matrix_l91_91186


namespace solve_equation_l91_91729

noncomputable def unique_solution (x : ℝ) : Prop :=
  2 * x * Real.log x + x - 1 = 0 → x = 1

-- Statement of our theorem
theorem solve_equation (x : ℝ) (h : 0 < x) : unique_solution x := sorry

end solve_equation_l91_91729


namespace probability_of_selection_of_X_l91_91896

theorem probability_of_selection_of_X 
  (P_Y : ℝ)
  (P_X_and_Y : ℝ) :
  P_Y = 2 / 7 →
  P_X_and_Y = 0.05714285714285714 →
  ∃ P_X : ℝ, P_X = 0.2 :=
by
  intro hY hXY
  sorry

end probability_of_selection_of_X_l91_91896


namespace num_trucks_washed_l91_91855

theorem num_trucks_washed (total_revenue cars_revenue suvs_revenue truck_charge : ℕ) 
  (h_total : total_revenue = 100)
  (h_cars : cars_revenue = 7 * 5)
  (h_suvs : suvs_revenue = 5 * 7)
  (h_truck_charge : truck_charge = 6) : 
  ∃ T : ℕ, (total_revenue - suvs_revenue - cars_revenue) / truck_charge = T := 
by {
  use 5,
  sorry
}

end num_trucks_washed_l91_91855


namespace smallest_right_triangle_area_l91_91121

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l91_91121


namespace simplify_expression_l91_91433

variable (a b : ℝ)

theorem simplify_expression : a + (3 * a - 3 * b) - (a - 2 * b) = 3 * a - b := 
by 
  sorry

end simplify_expression_l91_91433


namespace increasing_interval_m_range_l91_91955

def y (x m : ℝ) : ℝ := x^2 + 2 * m * x + 10

theorem increasing_interval_m_range (m : ℝ) : (∀ x, 2 ≤ x → ∀ x', x' ≥ x → y x m ≤ y x' m) → (-2 : ℝ) ≤ m :=
sorry

end increasing_interval_m_range_l91_91955


namespace smallest_x_absolute_value_l91_91947

theorem smallest_x_absolute_value : ∃ x : ℤ, |x + 3| = 15 ∧ ∀ y : ℤ, |y + 3| = 15 → x ≤ y :=
sorry

end smallest_x_absolute_value_l91_91947


namespace parabola_constant_l91_91917

theorem parabola_constant (b c : ℝ)
  (h₁ : -20 = 2 * (-2)^2 + b * (-2) + c)
  (h₂ : 24 = 2 * 2^2 + b * 2 + c) : 
  c = -6 := 
by 
  sorry

end parabola_constant_l91_91917


namespace time_to_fill_pool_l91_91070

noncomputable def slower_pump_rate : ℝ := 1 / 12.5
noncomputable def faster_pump_rate : ℝ := 1.5 * slower_pump_rate
noncomputable def combined_rate : ℝ := slower_pump_rate + faster_pump_rate

theorem time_to_fill_pool : (1 / combined_rate) = 5 := 
by
  sorry

end time_to_fill_pool_l91_91070


namespace BKING_2023_reappears_at_20_l91_91297

-- Defining the basic conditions of the problem
def cycle_length_BKING : ℕ := 5
def cycle_length_2023 : ℕ := 4

-- Formulating the proof problem statement
theorem BKING_2023_reappears_at_20 :
  Nat.lcm cycle_length_BKING cycle_length_2023 = 20 :=
by
  sorry

end BKING_2023_reappears_at_20_l91_91297


namespace probability_of_drawing_ball_labeled_3_on_second_draw_l91_91011

theorem probability_of_drawing_ball_labeled_3_on_second_draw :
  let box1 := [1, 1, 2, 3],
      box2 := [1, 1, 3],
      box3 := [1, 1, 1, 2, 2] in
  (let p1 := 2 / 4 * 1 / 4 + 1 / 4 * 1 / 4 + 1 / 4 * 1 / 6 in
    p1 = 11 / 48) :=
  by {
  let box1 := [1, 1, 2, 3],
  let box2 := [1, 1, 3],
  let box3 := [1, 1, 1, 2, 2],
  let p1 := 2 / 4 * 1 / 4 + 1 / 4 * 1 / 4 + 1 / 4 * 1 / 6,
  show p1 = 11 / 48,
  sorry
}

end probability_of_drawing_ball_labeled_3_on_second_draw_l91_91011


namespace root_of_quadratic_l91_91697

theorem root_of_quadratic (a : ℝ) (ha : a ≠ 1) (hroot : (a-1) * 1^2 - a * 1 + a^2 = 0) : a = -1 := by
  sorry

end root_of_quadratic_l91_91697


namespace smallest_right_triangle_area_l91_91099

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l91_91099


namespace eq_iff_solution_l91_91232

theorem eq_iff_solution (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  x^y + y^x = x^x + y^y ↔ x = y :=
by sorry

end eq_iff_solution_l91_91232


namespace intersection_A_B_l91_91389

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 2 * x^2 - x - 1 < 0}
def B : Set ℝ := {x : ℝ | Real.log x / Real.log (1/2) < 3}

-- Define the intersection A ∩ B and state the theorem
theorem intersection_A_B : A ∩ B = {x : ℝ | 1/8 < x ∧ x < 1} := by
   sorry

end intersection_A_B_l91_91389


namespace complete_the_square_l91_91898

theorem complete_the_square (x : ℝ) : 
  (x^2 - 2 * x - 3 = 0) ↔ ((x - 1)^2 = 4) :=
by sorry

end complete_the_square_l91_91898


namespace platform_length_l91_91635

theorem platform_length (train_length : ℕ) (time_cross_platform : ℕ) (time_cross_pole : ℕ) (train_speed : ℕ) (L : ℕ)
  (h1 : train_length = 500) 
  (h2 : time_cross_platform = 65) 
  (h3 : time_cross_pole = 25) 
  (h4 : train_speed = train_length / time_cross_pole)
  (h5 : train_speed = (train_length + L) / time_cross_platform) :
  L = 800 := 
sorry

end platform_length_l91_91635


namespace smallest_area_of_right_triangle_l91_91112

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l91_91112


namespace initial_mean_of_observations_l91_91736

theorem initial_mean_of_observations (M : ℝ) (h1 : 50 * M + 30 = 50 * 40.66) : M = 40.06 := 
sorry

end initial_mean_of_observations_l91_91736


namespace proof_statement_l91_91451

open Classical

variable (Person : Type) (Nationality : Type) (Occupation : Type)

variable (A B C D : Person)
variable (UnitedKingdom UnitedStates Germany France : Nationality)
variable (Doctor Teacher : Occupation)

variable (nationality : Person → Nationality)
variable (occupation : Person → Occupation)
variable (can_swim : Person → Prop)
variable (play_sports_together : Person → Person → Prop)

noncomputable def proof :=
  (nationality A = UnitedKingdom ∧ nationality D = Germany)

axiom condition1 : occupation A = Doctor ∧ ∃ x : Person, nationality x = UnitedStates ∧ occupation x = Doctor
axiom condition2 : occupation B = Teacher ∧ ∃ x : Person, nationality x = Germany ∧ occupation x = Teacher 
axiom condition3 : can_swim C ∧ ∀ x : Person, nationality x = Germany → ¬ can_swim x
axiom condition4 : ∃ x : Person, nationality x = France ∧ play_sports_together A x

theorem proof_statement : 
  (nationality A = UnitedKingdom ∧ nationality D = Germany) :=
by {
  sorry
}

end proof_statement_l91_91451


namespace calculate_paint_area_l91_91330

def barn_length : ℕ := 12
def barn_width : ℕ := 15
def barn_height : ℕ := 6
def window_length : ℕ := 2
def window_width : ℕ := 2

def area_to_paint : ℕ := 796

theorem calculate_paint_area 
    (b_len : ℕ := barn_length) 
    (b_wid : ℕ := barn_width) 
    (b_hei : ℕ := barn_height) 
    (win_len : ℕ := window_length) 
    (win_wid : ℕ := window_width) : 
    b_len = 12 → 
    b_wid = 15 → 
    b_hei = 6 → 
    win_len = 2 → 
    win_wid = 2 →
    area_to_paint = 796 :=
by
  -- Here, the proof would be provided.
  -- This line is a placeholder (sorry) indicating that the proof is yet to be constructed.
  sorry

end calculate_paint_area_l91_91330


namespace smallest_area_right_triangle_l91_91126

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l91_91126


namespace right_triangle_min_area_l91_91167

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l91_91167


namespace find_c_l91_91822

theorem find_c (c : ℕ) (h : 111111222222 = c * (c + 1)) : c = 333333 :=
by
  -- proof goes here
  sorry

end find_c_l91_91822


namespace trash_picked_outside_l91_91069

theorem trash_picked_outside (T_tot : ℕ) (C1 C2 C3 C4 C5 C6 C7 C8 : ℕ)
  (hT_tot : T_tot = 1576)
  (hC1 : C1 = 124) (hC2 : C2 = 98) (hC3 : C3 = 176) (hC4 : C4 = 212)
  (hC5 : C5 = 89) (hC6 : C6 = 241) (hC7 : C7 = 121) (hC8 : C8 = 102) :
  T_tot - (C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8) = 413 :=
by sorry

end trash_picked_outside_l91_91069


namespace minimum_g_a_l91_91681

noncomputable def f (x a : ℝ) : ℝ := x ^ 2 + 2 * a * x + 3

noncomputable def g (a : ℝ) : ℝ := 3 * a ^ 2 + 2 * a

theorem minimum_g_a : ∀ a : ℝ, a ≤ -1 → g a = 3 * a ^ 2 + 2 * a → g a ≥ 1 := by
  sorry

end minimum_g_a_l91_91681


namespace right_triangle_of_medians_l91_91051

theorem right_triangle_of_medians
  (a b c m1 m2 m3 : ℝ)
  (h1 : 4 * m1^2 = 2 * (b^2 + c^2) - a^2)
  (h2 : 4 * m2^2 = 2 * (a^2 + c^2) - b^2)
  (h3 : 4 * m3^2 = 2 * (a^2 + b^2) - c^2)
  (h4 : m1^2 + m2^2 = 5 * m3^2) :
  c^2 = a^2 + b^2 :=
by
  sorry

end right_triangle_of_medians_l91_91051


namespace sequence_general_term_l91_91956

theorem sequence_general_term (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 3)
  (h3 : a 3 = 5)
  (h4 : a 4 = 7) :
  ∀ n, a n = 2 * n - 1 :=
by
  sorry

end sequence_general_term_l91_91956


namespace domain_of_f_x_minus_1_l91_91018

theorem domain_of_f_x_minus_1 (f : ℝ → ℝ) (h : ∀ x, x^2 + 1 ∈ Set.Icc 1 10 → x ∈ Set.Icc (-3 : ℝ) 2) :
  Set.Icc 2 (11 : ℝ) ⊆ {x : ℝ | x - 1 ∈ Set.Icc 1 10} :=
by
  sorry

end domain_of_f_x_minus_1_l91_91018


namespace triangle_height_l91_91290

theorem triangle_height (base height area : ℝ) 
(h_base : base = 3) (h_area : area = 9) 
(h_area_eq : area = (base * height) / 2) :
  height = 6 := 
by 
  sorry

end triangle_height_l91_91290


namespace calculate_f8_f4_l91_91989

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 5) = f x
axiom f_at_1 : f 1 = 1
axiom f_at_2 : f 2 = 3

theorem calculate_f8_f4 : f 8 - f 4 = -2 := by
  sorry

end calculate_f8_f4_l91_91989


namespace number_of_deleted_apps_l91_91498

def initial_apps := 16
def remaining_apps := 8

def deleted_apps : ℕ := initial_apps - remaining_apps

theorem number_of_deleted_apps : deleted_apps = 8 := 
by
  unfold deleted_apps initial_apps remaining_apps
  rfl

end number_of_deleted_apps_l91_91498


namespace third_month_sale_l91_91334

theorem third_month_sale (s1 s2 s4 s5 s6 avg_sale: ℕ) (h1: s1 = 5420) (h2: s2 = 5660) (h3: s4 = 6350) (h4: s5 = 6500) (h5: s6 = 8270) (h6: avg_sale = 6400) :
  ∃ s3: ℕ, s3 = 6200 :=
by
  sorry

end third_month_sale_l91_91334


namespace corrections_needed_l91_91835

-- Define the corrected statements
def corrected_statements : List String :=
  ["A = 50", "B = A", "x = 1", "y = 2", "z = 3", "INPUT“How old are you?”;x",
   "INPUT x", "PRINT“A+B=”;C", "PRINT“Good-bye!”"]

-- Define the function to check if the statement is correctly formatted
def is_corrected (statement : String) : Prop :=
  statement ∈ corrected_statements

-- Lean theorem statement to prove each original incorrect statement should be correctly formatted
theorem corrections_needed (s : String) (incorrect : s ∈ ["A = B = 50", "x = 1, y = 2, z = 3", 
  "INPUT“How old are you”x", "INPUT, x", "PRINT A+B=;C", "PRINT Good-bye!"]) :
  ∃ t : String, is_corrected t :=
by 
  sorry

end corrections_needed_l91_91835


namespace average_weighted_score_l91_91636

theorem average_weighted_score
  (score1 score2 score3 : ℕ)
  (weight1 weight2 weight3 : ℕ)
  (h_scores : score1 = 90 ∧ score2 = 85 ∧ score3 = 80)
  (h_weights : weight1 = 5 ∧ weight2 = 2 ∧ weight3 = 3) :
  (weight1 * score1 + weight2 * score2 + weight3 * score3) / (weight1 + weight2 + weight3) = 86 := 
by
  sorry

end average_weighted_score_l91_91636


namespace directrix_of_parabola_l91_91504

theorem directrix_of_parabola : ∀ (x : ℝ), y = (x^2 - 8*x + 12) / 16 → ∃ (d : ℝ), d = -1/2 := 
sorry

end directrix_of_parabola_l91_91504


namespace dishonest_shopkeeper_gain_l91_91181

-- Conditions: false weight used by shopkeeper
def false_weight : ℚ := 930
def true_weight : ℚ := 1000

-- Correct answer: gain percentage
def gain_percentage (false_weight true_weight : ℚ) : ℚ :=
  ((true_weight - false_weight) / false_weight) * 100

theorem dishonest_shopkeeper_gain :
  gain_percentage false_weight true_weight = 7.53 := by
  sorry

end dishonest_shopkeeper_gain_l91_91181


namespace smallest_right_triangle_area_l91_91144

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l91_91144


namespace equations_have_one_contact_point_l91_91211

theorem equations_have_one_contact_point (c : ℝ):
  (∃ x : ℝ, x^2 + 1 = 4 * x + c) ∧ (∀ x1 x2 : ℝ, (x1 ≠ x2) → ¬(x1^2 + 1 = 4 * x1 + c ∧ x2^2 + 1 = 4 * x2 + c)) ↔ c = -3 :=
by
  sorry

end equations_have_one_contact_point_l91_91211


namespace bus_passengers_l91_91614

theorem bus_passengers (initial : ℕ) (first_stop_on : ℕ) (other_stop_off : ℕ) (other_stop_on : ℕ) : 
  initial = 50 ∧ first_stop_on = 16 ∧ other_stop_off = 22 ∧ other_stop_on = 5 →
  initial + first_stop_on - other_stop_off + other_stop_on = 49 :=
by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h_more
  cases h_more with h3 h4
  rw [h1, h2, h3, h4]
  norm_num

end bus_passengers_l91_91614


namespace expression_is_minus_two_l91_91608

noncomputable def A : ℝ := (Real.sqrt 6 + Real.sqrt 2) * (Real.sqrt 3 - 2) * Real.sqrt (Real.sqrt 3 + 2)

theorem expression_is_minus_two : A = -2 := by
  sorry

end expression_is_minus_two_l91_91608


namespace find_ab_l91_91516

variable (a b : ℝ)

def point_symmetric_about_line (Px Py Qx Qy : ℝ) (m n c : ℝ) : Prop :=
  ∃ xM yM : ℝ,
  xM = (Px + Qx) / 2 ∧ yM = (Py + Qy) / 2 ∧
  m * xM + n * yM = c ∧
  (Py - Qy) / (Px - Qx) * (-n/m) = -1

theorem find_ab (H : point_symmetric_about_line (a + 2) (b + 2) (b - a) (-b) 4 3 11) :
  a = 4 ∧ b = 2 :=
sorry

end find_ab_l91_91516


namespace gcd_of_459_and_357_l91_91071

open EuclideanDomain

theorem gcd_of_459_and_357 : gcd 459 357 = 51 :=
sorry

end gcd_of_459_and_357_l91_91071


namespace farmer_plow_l91_91216

theorem farmer_plow (P : ℕ) (M : ℕ) (H1 : M = 12) (H2 : 8 * P + M * (8 - (55 / P)) = 30) (H3 : 55 % P = 0) : P = 10 :=
by
  sorry

end farmer_plow_l91_91216


namespace original_number_l91_91190

-- Define the original statement and conditions
theorem original_number (x : ℝ) (h : 3 * (2 * x + 9) = 81) : x = 9 := by
  -- Sorry placeholder stands for the proof steps
  sorry

end original_number_l91_91190


namespace determine_p_l91_91523

theorem determine_p (m : ℕ) (p : ℕ) (h1: m = 34) 
  (h2: (1 : ℝ)^ (m + 1) / 5^ (m + 1) * 1^18 / 4^18 = 1 / (2 * 10^ p)) : 
  p = 35 := by sorry

end determine_p_l91_91523


namespace minimum_treasure_buried_l91_91797

def palm_tree (n : Nat) := n < 30

def sign_condition (n : Nat) (k : Nat) : Prop :=
  if n = 15 then palm_tree n ∧ k = 15
  else if n = 8 then palm_tree n ∧ k = 8
  else if n = 4 then palm_tree n ∧ k = 4
  else if n = 3 then palm_tree n ∧ k = 3
  else False

def treasure_condition (n : Nat) (k : Nat) : Prop :=
  (n ≤ k) → ∀ x, palm_tree x → sign_condition x k → x ≠ n

theorem minimum_treasure_buried : ∃ k, k = 15 ∧ ∀ n, treasure_condition n k :=
by
  sorry

end minimum_treasure_buried_l91_91797


namespace floor_S_value_l91_91717

theorem floor_S_value
  (a b c d : ℝ)
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (h_ab_squared : a^2 + b^2 = 1458)
  (h_cd_squared : c^2 + d^2 = 1458)
  (h_ac_product : a * c = 1156)
  (h_bd_product : b * d = 1156) :
  (⌊a + b + c + d⌋ = 77) := 
sorry

end floor_S_value_l91_91717


namespace unknown_number_l91_91762

theorem unknown_number (x : ℝ) (h : 7^8 - 6/x + 9^3 + 3 + 12 = 95) : x = 1 / 960908.333 :=
sorry

end unknown_number_l91_91762


namespace original_ratio_l91_91319

theorem original_ratio (x y : ℕ) (h1 : y = 15) (h2 : x + 10 = y) : x / y = 1 / 3 :=
by
  sorry

end original_ratio_l91_91319


namespace vertex_of_parabola_l91_91441

theorem vertex_of_parabola :
  ∃ (h k : ℝ), (∀ x : ℝ, -2 * (x - h) ^ 2 + k = -2 * (x - 2) ^ 2 - 5) ∧ h = 2 ∧ k = -5 :=
by
  sorry

end vertex_of_parabola_l91_91441


namespace signup_ways_l91_91468

theorem signup_ways (students groups : ℕ) (h_students : students = 5) (h_groups : groups = 3) :
  (groups ^ students = 243) :=
by
  have calculation : 3 ^ 5 = 243 := by norm_num
  rwa [h_students, h_groups]

end signup_ways_l91_91468


namespace num_bad_oranges_l91_91840

theorem num_bad_oranges (G B : ℕ) (hG : G = 24) (ratio : G / B = 3) : B = 8 :=
by
  sorry

end num_bad_oranges_l91_91840


namespace smallest_area_of_right_triangle_l91_91105

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l91_91105


namespace james_total_fish_catch_l91_91576

-- Definitions based on conditions
def poundsOfTrout : ℕ := 200
def poundsOfSalmon : ℕ := Nat.floor (1.5 * poundsOfTrout)
def poundsOfTuna : ℕ := 2 * poundsOfTrout

-- Proof statement
theorem james_total_fish_catch : poundsOfTrout + poundsOfSalmon + poundsOfTuna = 900 := by
  -- straightforward proof skipped for now
  sorry

end james_total_fish_catch_l91_91576


namespace triangle_exterior_angle_bisectors_l91_91353

theorem triangle_exterior_angle_bisectors 
  (α β γ α1 β1 γ1 : ℝ) 
  (h₁ : α = (β / 2 + γ / 2)) 
  (h₂ : β = (γ / 2 + α / 2)) 
  (h₃ : γ = (α / 2 + β / 2)) :
  α = 180 - 2 * α1 ∧
  β = 180 - 2 * β1 ∧
  γ = 180 - 2 * γ1 := by
  sorry

end triangle_exterior_angle_bisectors_l91_91353


namespace tan_prod_eq_sqrt_seven_l91_91351

theorem tan_prod_eq_sqrt_seven : 
  let x := (Real.pi / 7) 
  let y := (2 * Real.pi / 7)
  let z := (3 * Real.pi / 7)
  Real.tan x * Real.tan y * Real.tan z = Real.sqrt 7 :=
by
  sorry

end tan_prod_eq_sqrt_seven_l91_91351


namespace sum_real_imaginary_part_l91_91860

noncomputable def imaginary_unit : ℂ := Complex.I

theorem sum_real_imaginary_part {z : ℂ} (h : z * imaginary_unit = 1 + imaginary_unit) :
  z.re + z.im = 2 := 
sorry

end sum_real_imaginary_part_l91_91860


namespace find_point_P_l91_91008

noncomputable def tangent_at (f : ℝ → ℝ) (x : ℝ) : ℝ := (deriv f) x

theorem find_point_P :
  ∃ (x₀ y₀ : ℝ), (y₀ = (1 / x₀)) 
  ∧ (0 < x₀)
  ∧ (tangent_at (fun x => x^2) 2 = 4)
  ∧ (tangent_at (fun x => (1 / x)) x₀ = -1 / 4) 
  ∧ (x₀ = 2)
  ∧ (y₀ = 1 / 2) :=
sorry

end find_point_P_l91_91008


namespace percent_defective_shipped_l91_91766

-- Conditions given in the problem
def percent_defective (percent_total_defective: ℝ) : Prop := percent_total_defective = 0.08
def percent_shipped_defective (percent_defective_shipped: ℝ) : Prop := percent_defective_shipped = 0.04

-- The main theorem we want to prove
theorem percent_defective_shipped (percent_total_defective percent_defective_shipped : ℝ) 
  (h1 : percent_defective percent_total_defective) (h2 : percent_shipped_defective percent_defective_shipped) : 
  (percent_total_defective * percent_defective_shipped * 100) = 0.32 :=
by
  sorry

end percent_defective_shipped_l91_91766


namespace tom_spending_is_correct_l91_91619

-- Conditions
def cost_per_square_foot : ℕ := 5
def square_feet_per_seat : ℕ := 12
def number_of_seats : ℕ := 500
def construction_multiplier : ℕ := 2
def partner_contribution_ratio : ℚ := 0.40

-- Calculate and verify Tom's spending
def total_square_footage := number_of_seats * square_feet_per_seat
def land_cost := total_square_footage * cost_per_square_foot
def construction_cost := construction_multiplier * land_cost
def total_cost := land_cost + construction_cost
def partner_contribution := partner_contribution_ratio * total_cost
def tom_spending := (1 - partner_contribution_ratio) * total_cost

theorem tom_spending_is_correct : tom_spending = 54000 := 
by 
    -- The theorems calculate specific values 
    sorry

end tom_spending_is_correct_l91_91619


namespace max_intersection_value_l91_91720

noncomputable def max_intersection_size (A B C : Finset ℕ) (h1 : (A.card = 2019) ∧ (B.card = 2019)) 
  (h2 : (2 ^ A.card + 2 ^ B.card + 2 ^ C.card = 2 ^ (A ∪ B ∪ C).card)) : ℕ :=
  if ((A.card = 2019) ∧ (B.card = 2019) ∧ (A ∩ B ∩ C).card = 2018)
  then (A ∩ B ∩ C).card 
  else 0

theorem max_intersection_value (A B C : Finset ℕ) (h1 : (A.card = 2019) ∧ (B.card = 2019)) 
  (h2 : (2 ^ A.card + 2 ^ B.card + 2 ^ C.card = 2 ^ (A ∪ B ∪ C).card)) :
  max_intersection_size A B C h1 h2 = 2018 :=
sorry

end max_intersection_value_l91_91720


namespace quadratic_radical_type_equivalence_l91_91923

def is_same_type_as_sqrt2 (x : ℝ) : Prop := ∃ k : ℚ, x = k * (Real.sqrt 2)

theorem quadratic_radical_type_equivalence (A B C D : ℝ) (hA : A = (Real.sqrt 8) / 7)
  (hB : B = Real.sqrt 3) (hC : C = Real.sqrt (1 / 3)) (hD : D = Real.sqrt 12) :
  is_same_type_as_sqrt2 A ∧ ¬ is_same_type_as_sqrt2 B ∧ ¬ is_same_type_as_sqrt2 C ∧ ¬ is_same_type_as_sqrt2 D :=
by
  sorry

end quadratic_radical_type_equivalence_l91_91923


namespace probability_of_selecting_one_male_and_one_female_l91_91063

noncomputable def probability_one_male_one_female : ℚ :=
  let total_ways := (Nat.choose 6 2) -- Total number of ways to select 2 out of 6
  let ways_one_male_one_female := (Nat.choose 3 1) * (Nat.choose 3 1) -- Ways to select 1 male and 1 female
  ways_one_male_one_female / total_ways

theorem probability_of_selecting_one_male_and_one_female :
  probability_one_male_one_female = 3 / 5 := by
  sorry

end probability_of_selecting_one_male_and_one_female_l91_91063


namespace find_salary_of_Thomas_l91_91294

-- Declare the variables representing the salaries of Raj, Roshan, and Thomas
variables (R S T : ℝ)

-- Given conditions as definitions
def avg_salary_Raj_Roshan : Prop := (R + S) / 2 = 4000
def avg_salary_Raj_Roshan_Thomas : Prop := (R + S + T) / 3 = 5000

-- Stating the theorem
theorem find_salary_of_Thomas
  (h1 : avg_salary_Raj_Roshan R S)
  (h2 : avg_salary_Raj_Roshan_Thomas R S T) : T = 7000 :=
by
  sorry

end find_salary_of_Thomas_l91_91294


namespace max_value_of_f_l91_91718

noncomputable def f (x a : ℝ) : ℝ := - (1/3) * x ^ 3 + (1/2) * x ^ 2 + 2 * a * x

theorem max_value_of_f (a : ℝ) (h0 : 0 < a) (h1 : a < 2)
  (h2 : ∀ x, 1 ≤ x → x ≤ 4 → f x a ≥ f 4 a)
  (h3 : f 4 a = -16 / 3) :
  f 2 a = 10 / 3 :=
sorry

end max_value_of_f_l91_91718


namespace cannot_be_combined_with_sqrt2_l91_91907

def can_be_combined (x y : ℝ) : Prop := ∃ k : ℝ, k * x = y

theorem cannot_be_combined_with_sqrt2 :
  let a := Real.sqrt (1 / 2)
  let b := Real.sqrt 8
  let c := Real.sqrt 12
  let d := -Real.sqrt 18
  ¬ can_be_combined c (Real.sqrt 2) := 
by
  sorry

end cannot_be_combined_with_sqrt2_l91_91907


namespace sausage_shop_period_l91_91289

theorem sausage_shop_period
  (strips_per_sandwich : ℕ)
  (time_per_sandwich : ℕ)
  (total_strips : ℕ)
  (h_strips : strips_per_sandwich = 4)
  (h_time : time_per_sandwich = 5)
  (h_total : total_strips = 48) :
  (total_strips / strips_per_sandwich) * time_per_sandwich = 60 := by
  sorry

end sausage_shop_period_l91_91289


namespace remove_five_magazines_l91_91327

theorem remove_five_magazines (magazines : Fin 10 → Set α) 
  (coffee_table : Set α) 
  (h_cover : (⋃ i, magazines i) = coffee_table) :
  ∃ ( S : Set α), S ⊆ coffee_table ∧ (∃ (removed : Finset (Fin 10)), removed.card = 5 ∧ 
    coffee_table \ (⋃ i ∈ removed, magazines i) ⊆ S ∧ (S = coffee_table \ (⋃ i ∈ removed, magazines i) ) ∧ 
    (⋃ i ∉ removed, magazines i) ∩ S = ∅) := 
sorry

end remove_five_magazines_l91_91327


namespace find_a2_l91_91675

open Classical

variable {a_n : ℕ → ℝ} {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n m : ℕ, a (n + m) = a n * q ^ m

theorem find_a2 (h1 : geometric_sequence a_n q)
                (h2 : a_n 7 = 1 / 4)
                (h3 : a_n 3 * a_n 5 = 4 * (a_n 4 - 1)) :
  a_n 2 = 8 :=
sorry

end find_a2_l91_91675


namespace problem_solution_l91_91001

variable (a : ℝ)
def ellipse_p (a : ℝ) : Prop := (0 < a) ∧ (a < 5)
def quadratic_q (a : ℝ) : Prop := (-3 ≤ a) ∧ (a ≤ 3)
def p_or_q (a : ℝ) : Prop := ((0 < a ∧ a < 5) ∨ ((-3 ≤ a) ∧ (a ≤ 3)))
def p_and_q (a : ℝ) : Prop := ((0 < a ∧ a < 5) ∧ ((-3 ≤ a) ∧ (a ≤ 3)))

theorem problem_solution (a : ℝ) :
  (ellipse_p a → 0 < a ∧ a < 5) ∧ 
  (¬(ellipse_p a) ∧ quadratic_q a → -3 ≤ a ∧ a ≤ 0) ∧
  (p_or_q a ∧ ¬(p_and_q a) → 3 < a ∧ a < 5 ∨ (-3 ≤ a ∧ a ≤ 0)) :=
  by
  sorry

end problem_solution_l91_91001


namespace fireflies_joined_l91_91992

theorem fireflies_joined (x : ℕ) : 
  let initial_fireflies := 3
  let flew_away := 2
  let remaining_fireflies := 9
  initial_fireflies + x - flew_away = remaining_fireflies → x = 8 := by
  sorry

end fireflies_joined_l91_91992


namespace smallest_right_triangle_area_l91_91120

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l91_91120


namespace smallest_area_of_right_triangle_l91_91082

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l91_91082


namespace value_of_a_27_l91_91000

def a_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 1) = a n + 2 * n

theorem value_of_a_27 (a : ℕ → ℕ) (h : a_sequence a) : a 27 = 702 :=
sorry

end value_of_a_27_l91_91000


namespace sequence_geometric_l91_91977

theorem sequence_geometric (a : ℕ → ℝ) (h : ∀ n, a n ≠ 0)
  (h_arith : 2 * a 2 = a 1 + a 3)
  (h_geom : a 3 ^ 2 = a 2 * a 4)
  (h_recip_arith : 2 / a 4 = 1 / a 3 + 1 / a 5) :
  a 3 ^ 2 = a 1 * a 5 :=
sorry

end sequence_geometric_l91_91977


namespace smallest_right_triangle_area_l91_91142

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l91_91142


namespace correct_statements_arithmetic_seq_l91_91674

/-- For an arithmetic sequence {a_n} with a1 > 0 and common difference d ≠ 0, 
    the correct statements among options A, B, C, and D are B and C. -/
theorem correct_statements_arithmetic_seq (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h_seq : ∀ n, a (n + 1) = a n + d) 
  (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2) (h_a1_pos : a 1 > 0) (h_d_ne_0 : d ≠ 0) : 
  (S 5 = S 9 → 
   S 7 = (10 * a 4) / 2) ∧ 
  (S 6 > S 7 → S 7 > S 8) := 
sorry

end correct_statements_arithmetic_seq_l91_91674


namespace largest_mersenne_prime_is_127_l91_91074

noncomputable def largest_mersenne_prime_less_than_500 : ℕ :=
  127

theorem largest_mersenne_prime_is_127 :
  ∃ p : ℕ, Nat.Prime p ∧ (2^p - 1) = largest_mersenne_prime_less_than_500 ∧ 2^p - 1 < 500 := 
by 
  -- The largest Mersenne prime less than 500 is 127
  use 7
  sorry

end largest_mersenne_prime_is_127_l91_91074


namespace repeating_decimal_equiv_fraction_l91_91072

theorem repeating_decimal_equiv_fraction :
  (0.1 ++ (list.repeat '4' 1) ++ (list.repeat '7' 1)).to_rat = 73 / 495 := sorry

end repeating_decimal_equiv_fraction_l91_91072


namespace smallest_n_such_that_squares_contain_7_l91_91365

def contains_seven (n : ℕ) : Prop :=
  let digits := n.to_digits 10
  7 ∈ digits

theorem smallest_n_such_that_squares_contain_7 :
  ∃ n : ℕ, n >= 10 ∧ contains_seven (n^2) ∧ contains_seven ((n+1)^2) ∧ n = 26 :=
by 
  sorry

end smallest_n_such_that_squares_contain_7_l91_91365


namespace total_revenue_calculation_l91_91897

variables (a b : ℕ) -- Assuming a and b are natural numbers representing the number of newspapers

-- Define the prices
def purchase_price_per_copy : ℝ := 0.4
def selling_price_per_copy : ℝ := 0.5
def return_price_per_copy : ℝ := 0.2

-- Define the revenue and cost calculations
def revenue_from_selling (b : ℕ) : ℝ := selling_price_per_copy * b
def revenue_from_returning (a b : ℕ) : ℝ := return_price_per_copy * (a - b)
def cost_of_purchasing (a : ℕ) : ℝ := purchase_price_per_copy * a

-- Define the total revenue
def total_revenue (a b : ℕ) : ℝ :=
  revenue_from_selling b + revenue_from_returning a b - cost_of_purchasing a

-- The theorem we need to prove
theorem total_revenue_calculation (a b : ℕ) :
  total_revenue a b = 0.3 * b - 0.2 * a :=
by
  sorry

end total_revenue_calculation_l91_91897


namespace petya_wins_max_margin_l91_91551

theorem petya_wins_max_margin {P1 P2 V1 V2 : ℕ} 
  (h1 : P1 = V1 + 9)
  (h2 : V2 = P2 + 9)
  (h3 : P1 + P2 + V1 + V2 = 27)
  (h4 : P1 + P2 > V1 + V2) :
  ∃ m : ℕ, m = 9 ∧ P1 + P2 - (V1 + V2) = m :=
by
  sorry

end petya_wins_max_margin_l91_91551


namespace how_many_years_younger_l91_91996

-- Define conditions
def age_ratio (sandy_age moll_age : ℕ) := sandy_age * 9 = moll_age * 7
def sandy_age := 70

-- Define the theorem to prove
theorem how_many_years_younger 
  (molly_age : ℕ) 
  (h1 : age_ratio sandy_age molly_age) 
  (h2 : sandy_age = 70) : molly_age - sandy_age = 20 := 
sorry

end how_many_years_younger_l91_91996


namespace probability_three_consecutive_heads_four_tosses_l91_91333

theorem probability_three_consecutive_heads_four_tosses :
  let total_outcomes := 16
  let favorable_outcomes := 2
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = 1 / 8 := by
    sorry

end probability_three_consecutive_heads_four_tosses_l91_91333


namespace estimate_2_sqrt_5_l91_91806

theorem estimate_2_sqrt_5: 4 < 2 * Real.sqrt 5 ∧ 2 * Real.sqrt 5 < 5 :=
by
  sorry

end estimate_2_sqrt_5_l91_91806


namespace angle_passing_through_point_l91_91891

-- Definition of the problem conditions
def is_terminal_side_of_angle (x y : ℝ) (α : ℝ) : Prop :=
  let r := Real.sqrt (x^2 + y^2);
  (x = Real.cos α * r) ∧ (y = Real.sin α * r)

-- Lean 4 statement of the problem
theorem angle_passing_through_point (α : ℝ) :
  is_terminal_side_of_angle 1 (-1) α → α = - (Real.pi / 4) :=
by sorry

end angle_passing_through_point_l91_91891


namespace solve_equation_l91_91673

theorem solve_equation : 
  ∀ x : ℝ, 
  (((15 * x - x^2) / (x + 2)) * (x + (15 - x) / (x + 2)) = 54) → (x = 9 ∨ x = -1) :=
by
  sorry

end solve_equation_l91_91673


namespace part1_f1_part1_f2_part1_f3_part2_f_gt_1_for_n_1_2_part2_f_lt_1_for_n_ge_3_l91_91235

noncomputable def a (n : ℕ) : ℚ := 1 / (n : ℚ)

noncomputable def S (n : ℕ) : ℚ := (Finset.range (n+1)).sum (λ k => a (k + 1))

noncomputable def f (n : ℕ) : ℚ :=
  if n = 1 then S 2
  else S (2 * n) - S (n - 1)

theorem part1_f1 : f 1 = 3 / 2 := by sorry

theorem part1_f2 : f 2 = 13 / 12 := by sorry

theorem part1_f3 : f 3 = 19 / 20 := by sorry

theorem part2_f_gt_1_for_n_1_2 (n : ℕ) (h₁ : n = 1 ∨ n = 2) : f n > 1 := by sorry

theorem part2_f_lt_1_for_n_ge_3 (n : ℕ) (h₁ : n ≥ 3) : f n < 1 := by sorry

end part1_f1_part1_f2_part1_f3_part2_f_gt_1_for_n_1_2_part2_f_lt_1_for_n_ge_3_l91_91235


namespace combined_population_lake_bright_and_sunshine_hills_l91_91948

theorem combined_population_lake_bright_and_sunshine_hills
  (p_toadon p_gordonia p_lake_bright p_riverbank p_sunshine_hills : ℕ)
  (h1 : p_toadon + p_gordonia + p_lake_bright + p_riverbank + p_sunshine_hills = 120000)
  (h2 : p_gordonia = 1 / 3 * 120000)
  (h3 : p_toadon = 3 / 4 * p_gordonia)
  (h4 : p_riverbank = p_toadon + 2 / 5 * p_toadon) :
  p_lake_bright + p_sunshine_hills = 8000 :=
by
  sorry

end combined_population_lake_bright_and_sunshine_hills_l91_91948


namespace area_enclosed_shape_l91_91410

open Function
open BigOperators

noncomputable def tangent_slope_at_origin : ℝ :=
  deriv (λ x : ℝ, exp (2 * x)) 0

theorem area_enclosed_shape : tangent_slope_at_origin = 2 → ∫ x in 0..2, (2 * x - x ^ 2) = 4 / 3 :=
by
  intro h_slope
  have h : tangent_slope_at_origin = 2 := h_slope
  rw [h]
  sorry

end area_enclosed_shape_l91_91410


namespace reserve_bird_percentage_l91_91969

theorem reserve_bird_percentage (total_birds hawks paddyfield_warbler_percentage kingfisher_percentage woodpecker_percentage owl_percentage : ℕ) 
  (h1 : total_birds = 5000)
  (h2 : hawks = 30 * total_birds / 100)
  (h3 : paddyfield_warbler_percentage = 40)
  (h4 : kingfisher_percentage = 25)
  (h5 : woodpecker_percentage = 15)
  (h6 : owl_percentage = 15) :
  let non_hawks := total_birds - hawks
  let paddyfield_warblers := paddyfield_warbler_percentage * non_hawks / 100
  let kingfishers := kingfisher_percentage * paddyfield_warblers / 100
  let woodpeckers := woodpecker_percentage * non_hawks / 100
  let owls := owl_percentage * non_hawks / 100
  let specified_non_hawks := paddyfield_warblers + kingfishers + woodpeckers + owls
  let unspecified_non_hawks := non_hawks - specified_non_hawks
  let percentage_unspecified := unspecified_non_hawks * 100 / total_birds
  percentage_unspecified = 14 := by
  sorry

end reserve_bird_percentage_l91_91969


namespace distinct_remainders_l91_91987

theorem distinct_remainders
  (a n : ℕ) (h_pos_a : 0 < a) (h_pos_n : 0 < n)
  (h_div : n ∣ a^n - 1) :
  ∀ i j : ℕ, i ∈ (Finset.range n).image (· + 1) →
            j ∈ (Finset.range n).image (· + 1) →
            (a^i + i) % n = (a^j + j) % n →
            i = j :=
by
  intros i j hi hj h
  sorry

end distinct_remainders_l91_91987


namespace find_y_l91_91360

-- Define the atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight of the compound C6HyO7
def molecular_weight : ℝ := 192

-- Define the contribution of Carbon and Oxygen
def contribution_C : ℝ := 6 * atomic_weight_C
def contribution_O : ℝ := 7 * atomic_weight_O

-- The proof statement
theorem find_y (y : ℕ) :
  molecular_weight = contribution_C + y * atomic_weight_H + contribution_O → y = 8 :=
by
  sorry

end find_y_l91_91360


namespace intersection_eq_neg1_l91_91830

open Set

noncomputable def setA : Set Int := {x : Int | x^2 - 1 ≤ 0}
def setB : Set Int := {x : Int | x^2 - x - 2 = 0}

theorem intersection_eq_neg1 : setA ∩ setB = {-1} := by
  sorry

end intersection_eq_neg1_l91_91830


namespace area_of_triangle_ABC_is_1_l91_91067

-- Define the vertices A, B, and C
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (2, 1)

-- Define the function to compute the area of the triangle given three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- The main theorem to prove that the area of triangle ABC is 1
theorem area_of_triangle_ABC_is_1 : triangle_area A B C = 1 := 
by
  sorry

end area_of_triangle_ABC_is_1_l91_91067


namespace jonessa_total_pay_l91_91883

theorem jonessa_total_pay (total_pay : ℝ) (take_home_pay : ℝ) (h1 : take_home_pay = 450) (h2 : 0.90 * total_pay = take_home_pay) : total_pay = 500 :=
by
  sorry

end jonessa_total_pay_l91_91883


namespace find_probability_l91_91043

noncomputable def normal_dist : ℝ → ℝ :=
  λ x, 1 / (5 * real.sqrt (2 * real.pi)) * real.exp (- (x - 100)^2 / (2 * 5^2))

theorem find_probability
  (ξ : ℝ)
  (h₁ : ∀ x, ξ = normal_dist x)
  (h₂ : ∃ ξ, P(ξ < 110) = 0.96) :
  P(90 < ξ ∧ ξ < 100) = 0.46 :=
sorry

end find_probability_l91_91043


namespace wood_rope_equations_l91_91544

theorem wood_rope_equations (x y : ℝ) (h1 : y - x = 4.5) (h2 : 0.5 * y = x - 1) :
  (y - x = 4.5) ∧ (0.5 * y = x - 1) :=
by
  sorry

end wood_rope_equations_l91_91544


namespace treasure_under_minimum_signs_l91_91792

theorem treasure_under_minimum_signs :
  (∃ (n : ℕ), (n ≤ 15) ∧ 
    (∀ i, i ∈ {15, 8, 4, 3} → 
      (if (i = n) then False else True))) :=
sorry

end treasure_under_minimum_signs_l91_91792


namespace unique_necklace_arrangements_l91_91265

-- Definitions
def num_beads : Nat := 7

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- The number of unique ways to arrange the beads on a necklace
-- considering rotations and reflections
theorem unique_necklace_arrangements : (factorial num_beads) / (num_beads * 2) = 360 := 
by
  sorry

end unique_necklace_arrangements_l91_91265


namespace petya_wins_max_margin_l91_91550

theorem petya_wins_max_margin {P1 P2 V1 V2 : ℕ} 
  (h1 : P1 = V1 + 9)
  (h2 : V2 = P2 + 9)
  (h3 : P1 + P2 + V1 + V2 = 27)
  (h4 : P1 + P2 > V1 + V2) :
  ∃ m : ℕ, m = 9 ∧ P1 + P2 - (V1 + V2) = m :=
by
  sorry

end petya_wins_max_margin_l91_91550


namespace simplify_expression_l91_91998

theorem simplify_expression :
  (512 : ℝ)^(1/4) * (343 : ℝ)^(1/2) = 28 * (14 : ℝ)^(1/4) := by
  sorry

end simplify_expression_l91_91998


namespace increasing_quadratic_l91_91936

noncomputable def f (a x : ℝ) : ℝ := 3 * x^2 - a * x + 4

theorem increasing_quadratic {a : ℝ} :
  (∀ x ≥ -5, 6 * x - a ≥ 0) ↔ a ≤ -30 :=
by
  sorry

end increasing_quadratic_l91_91936


namespace min_treasure_signs_buried_l91_91799

theorem min_treasure_signs_buried (
    total_trees signs_15 signs_8 signs_4 signs_3 : ℕ
    (h_total: total_trees = 30)
    (h_signs_15: signs_15 = 15)
    (h_signs_8: signs_8 = 8)
    (h_signs_4: signs_4 = 4)
    (h_signs_3: signs_3 = 3)
    (h_truthful: ∀ n, n ≠ signs_15 ∧ n ≠ signs_8 ∧ n ≠ signs_4 ∧ n ≠ signs_3 → true_sign n = false)
    -- true_sign n indicates if the sign on the tree stating "Exactly under n signs a treasure is buried" is true
) :
    ∃ n, n = 15 :=
by
  sorry

end min_treasure_signs_buried_l91_91799


namespace quadratic_roots_equal_l91_91950

theorem quadratic_roots_equal (m : ℝ) :
  (∃ x : ℝ, x^2 - 4 * x + m - 1 = 0 ∧ (∀ y : ℝ, y^2 - 4*y + m-1 = 0 → y = x)) ↔ (m = 5 ∧ (∀ x, x^2 - 4 * x + 4 = 0 ↔ x = 2)) :=
by
  sorry

end quadratic_roots_equal_l91_91950


namespace log_expression_equals_neg_one_l91_91350

open Real

-- Define the given logarithmic condition.
def log_condition := log 2 + log 5 = 1

-- Define the theorem to prove the given expression equals -1.
theorem log_expression_equals_neg_one : 
  log (5 / 2) + 2 * log 2 - (1 / 2)⁻¹ = -1 :=
by 
  have h1 : log 2 + log 5 = 1 := log_condition
  sorry

end log_expression_equals_neg_one_l91_91350


namespace false_conjunction_l91_91908

theorem false_conjunction (p q : Prop) (h : ¬(p ∧ q)) : ¬ (¬p ∧ ¬q) := sorry

end false_conjunction_l91_91908


namespace correct_system_of_equations_l91_91545

-- Definitions corresponding to the conditions
def wood_length (y x : ℝ) : Prop := y - x = 4.5
def rope_half_length (y x : ℝ) : Prop := (1 / 2) * y = x - 1

-- The final statement proving the system of equations
theorem correct_system_of_equations (y x : ℝ) :
  wood_length y x ∧ rope_half_length y x ↔ (y - x = 4.5 ∧ (1 / 2) * y = x - 1) :=
by
  split
  . intro h
    cases h with h1 h2
    exact ⟨h1, h2⟩
  . intro h
    cases h with h1 h2
    exact ⟨h1, h2⟩

end correct_system_of_equations_l91_91545


namespace geo_prog_sum_463_l91_91038

/-- Given a set of natural numbers forming an increasing geometric progression with an integer
common ratio where the sum equals 463, prove that these numbers must be {463}, {1, 462}, or {1, 21, 441}. -/
theorem geo_prog_sum_463 (n : ℕ) (b₁ q : ℕ) (s : Finset ℕ) (hgeo : ∀ i j, i < j → s.toList.get? i = some (b₁ * q^i) ∧ s.toList.get? j = some (b₁ * q^j))
  (hsum : s.sum id = 463) : 
  s = {463} ∨ s = {1, 462} ∨ s = {1, 21, 441} :=
sorry

end geo_prog_sum_463_l91_91038


namespace jason_text_messages_per_day_l91_91983

theorem jason_text_messages_per_day
  (monday_messages : ℕ)
  (tuesday_messages : ℕ)
  (total_messages : ℕ)
  (average_per_day : ℕ)
  (messages_wednesday_friday_per_day : ℕ) :
  monday_messages = 220 →
  tuesday_messages = monday_messages / 2 →
  average_per_day = 96 →
  total_messages = 5 * average_per_day →
  total_messages - (monday_messages + tuesday_messages) = 3 * messages_wednesday_friday_per_day →
  messages_wednesday_friday_per_day = 50 :=
by
  intros
  sorry

end jason_text_messages_per_day_l91_91983


namespace smallest_right_triangle_area_l91_91143

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l91_91143


namespace polynomial_identity_sum_l91_91407

theorem polynomial_identity_sum (A B C D : ℤ) (h : (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) : 
  A + B + C + D = 36 := 
by 
  sorry

end polynomial_identity_sum_l91_91407


namespace number_of_math_books_l91_91630

-- Definitions based on the conditions in the problem
def total_books (M H : ℕ) : Prop := M + H = 90
def total_cost (M H : ℕ) : Prop := 4 * M + 5 * H = 390

-- Proof statement
theorem number_of_math_books (M H : ℕ) (h1 : total_books M H) (h2 : total_cost M H) : M = 60 :=
  sorry

end number_of_math_books_l91_91630


namespace xy_value_l91_91752

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 := 
by sorry

end xy_value_l91_91752


namespace smallest_base_b_l91_91902

theorem smallest_base_b (b : ℕ) (n : ℕ) : b > 3 ∧ 3 * b + 4 = n ^ 2 → b = 4 := 
by
  sorry

end smallest_base_b_l91_91902


namespace mean_age_gauss_family_l91_91436

theorem mean_age_gauss_family :
  let ages := [7, 7, 7, 14, 15]
  let sum_ages := List.sum ages
  let number_of_children := List.length ages
  let mean_age := sum_ages / number_of_children
  mean_age = 10 :=
by
  sorry

end mean_age_gauss_family_l91_91436


namespace birthday_candles_l91_91347

theorem birthday_candles :
  ∀ (candles_Ambika : ℕ) (candles_Aniyah : ℕ),
  candles_Ambika = 4 →
  candles_Aniyah = 6 * candles_Ambika →
  (candles_Ambika + candles_Aniyah) / 2 = 14 :=
by
  intro candles_Ambika candles_Aniyah h1 h2
  rw [h1, h2]
  sorry

end birthday_candles_l91_91347


namespace bob_total_miles_l91_91927

def total_miles_day1 (T : ℝ) := 0.20 * T
def remaining_miles_day1 (T : ℝ) := T - total_miles_day1 T
def total_miles_day2 (T : ℝ) := 0.50 * remaining_miles_day1 T
def remaining_miles_day2 (T : ℝ) := remaining_miles_day1 T - total_miles_day2 T
def total_miles_day3 (T : ℝ) := 28

theorem bob_total_miles (T : ℝ) (h : total_miles_day3 T = remaining_miles_day2 T) : T = 70 :=
by
  sorry

end bob_total_miles_l91_91927


namespace probability_of_desired_roll_l91_91990

-- Definitions of six-sided dice rolls and probability results
def is_greater_than_four (n : ℕ) : Prop := n > 4
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5

-- Definitions of probabilities based on dice outcomes
def prob_greater_than_four : ℚ := 2 / 6
def prob_prime : ℚ := 3 / 6

-- Definition of joint probability for independent events
def joint_prob : ℚ := prob_greater_than_four * prob_prime

-- Theorem to prove
theorem probability_of_desired_roll : joint_prob = 1 / 6 := 
by
  sorry

end probability_of_desired_roll_l91_91990


namespace hired_waiters_l91_91654

theorem hired_waiters (W H : Nat) (hcooks : Nat := 9) 
                      (initial_ratio : 3 * W = 11 * hcooks)
                      (new_ratio : 9 = 5 * (W + H)) 
                      (original_waiters : W = 33) 
                      : H = 12 :=
by
  sorry

end hired_waiters_l91_91654


namespace infinite_grid_coloring_l91_91500

theorem infinite_grid_coloring (color : ℕ × ℕ → Fin 4)
  (h_coloring_condition : ∀ (i j : ℕ), color (i, j) ≠ color (i + 1, j) ∧
                                      color (i, j) ≠ color (i, j + 1) ∧
                                      color (i, j) ≠ color (i + 1, j + 1) ∧
                                      color (i + 1, j) ≠ color (i, j + 1)) :
  ∃ m : ℕ, ∃ a b : Fin 4, ∀ n : ℕ, color (m, n) = a ∨ color (m, n) = b :=
sorry

end infinite_grid_coloring_l91_91500


namespace xyz_sum_is_22_l91_91016

theorem xyz_sum_is_22 (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x * y = 24) (h2 : x * z = 48) (h3 : y * z = 72) : 
  x + y + z = 22 :=
sorry

end xyz_sum_is_22_l91_91016


namespace meaningful_expr_iff_x_ne_neg_5_l91_91179

theorem meaningful_expr_iff_x_ne_neg_5 (x : ℝ) : (x + 5 ≠ 0) ↔ (x ≠ -5) :=
by
  sorry

end meaningful_expr_iff_x_ne_neg_5_l91_91179


namespace cost_price_of_computer_table_l91_91445

theorem cost_price_of_computer_table (SP : ℝ) (CP : ℝ) (h : SP = CP * 1.24) (h_SP : SP = 8215) : CP = 6625 :=
by
  -- Start the proof block
  sorry -- Proof is not required as per the instructions

end cost_price_of_computer_table_l91_91445


namespace unique_solution_iff_t_eq_quarter_l91_91374

variable {x y t : ℝ}

theorem unique_solution_iff_t_eq_quarter : (∃! (x y : ℝ), (x ≥ y^2 + t * y ∧ y^2 + t * y ≥ x^2 + t)) ↔ t = 1 / 4 :=
by
  sorry

end unique_solution_iff_t_eq_quarter_l91_91374


namespace total_cookies_l91_91929

theorem total_cookies (chris kenny glenn : ℕ) 
  (h1 : chris = kenny / 2)
  (h2 : glenn = 4 * kenny)
  (h3 : glenn = 24) : 
  chris + kenny + glenn = 33 := 
by
  -- Focusing on defining the theorem statement correct without entering the proof steps.
  sorry

end total_cookies_l91_91929


namespace parabola_properties_l91_91811

theorem parabola_properties :
  let a := -2
  let b := 4
  let c := 8
  ∃ h k : ℝ, 
    (∀ x : ℝ, y = a * x^2 + b * x + c) ∧ 
    (h = 1) ∧ 
    (k = 10) ∧ 
    (a < 0) ∧ 
    (axisOfSymmetry = h) ∧ 
    (vertex = (h, k)) :=
by
  sorry

end parabola_properties_l91_91811


namespace average_score_difference_l91_91497

theorem average_score_difference {A B : ℝ} (hA : (19 * A + 125) / 20 = A + 5) (hB : (17 * B + 145) / 18 = B + 6) :
  (B + 6) - (A + 5) = 13 :=
  sorry

end average_score_difference_l91_91497


namespace div_by_six_l91_91871

theorem div_by_six (n : ℕ) : 6 ∣ (17^n - 11^n) :=
by
  sorry

end div_by_six_l91_91871


namespace clown_balloon_count_l91_91044

theorem clown_balloon_count (b1 b2 : ℕ) (h1 : b1 = 47) (h2 : b2 = 13) : b1 + b2 = 60 := by
  sorry

end clown_balloon_count_l91_91044


namespace smallest_n_squared_contains_7_l91_91368

-- Lean statement
theorem smallest_n_squared_contains_7 :
  ∃ n : ℕ, (n^2).toString.contains '7' ∧ ((n+1)^2).toString.contains '7' ∧
  ∀ m : ℕ, ((m < n) → ¬(m^2).toString.contains '7' ∨ ¬((m+1)^2).toString.contains '7') :=
begin
  sorry
end

end smallest_n_squared_contains_7_l91_91368


namespace french_fries_cost_is_correct_l91_91348

def burger_cost : ℝ := 5
def soft_drink_cost : ℝ := 3
def special_burger_meal_cost : ℝ := 9.5

def french_fries_cost : ℝ :=
  special_burger_meal_cost - (burger_cost + soft_drink_cost)

theorem french_fries_cost_is_correct :
  french_fries_cost = 1.5 :=
by
  unfold french_fries_cost
  unfold special_burger_meal_cost
  unfold burger_cost
  unfold soft_drink_cost
  sorry

end french_fries_cost_is_correct_l91_91348


namespace entire_hike_length_l91_91457

-- Definitions directly from the conditions in part a)
def tripp_backpack_weight : ℕ := 25
def charlotte_backpack_weight : ℕ := tripp_backpack_weight - 7
def miles_hiked_first_day : ℕ := 9
def miles_left_to_hike : ℕ := 27

-- Theorem proving the entire hike length
theorem entire_hike_length :
  miles_hiked_first_day + miles_left_to_hike = 36 :=
by
  sorry

end entire_hike_length_l91_91457


namespace students_received_B_l91_91412

theorem students_received_B (charles_ratio : ℚ) (dawsons_class : ℕ) 
  (h_charles_ratio : charles_ratio = 3 / 5) (h_dawsons_class : dawsons_class = 30) : 
  ∃ y : ℕ, (charles_ratio = y / dawsons_class) ∧ y = 18 := 
by 
  sorry

end students_received_B_l91_91412


namespace smallest_area_right_triangle_l91_91158

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l91_91158


namespace function_monotonicity_l91_91247

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem function_monotonicity :
  ∀ x₁ x₂, -Real.pi / 6 ≤ x₁ → x₁ ≤ x₂ → x₂ ≤ Real.pi / 3 → f x₁ ≤ f x₂ :=
by
  sorry

end function_monotonicity_l91_91247


namespace ellipse_eq_line_eq_l91_91803

-- Conditions for part (I)
def cond1 (a b : ℝ) : Prop := a > 0 ∧ b > 0 ∧ a > b
def pt_p_cond (PF1 PF2 : ℝ) : Prop := PF1 = 4 / 3 ∧ PF2 = 14 / 3 ∧ PF1^2 + PF2^2 = 1

-- Theorem for part (I)
theorem ellipse_eq (a b : ℝ) (PF1 PF2 : ℝ) (h₁ : cond1 a b) (h₂ : pt_p_cond PF1 PF2) : 
  (a = 3 ∧ b = 2 ∧ PF1 = 4 / 3 ∧ PF2 = 14 / 3) → 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

-- Conditions for part (II)
def center_circle (M : ℝ × ℝ) : Prop := M = (-2, 1)
def pts_symmetric (A B M : ℝ × ℝ) : Prop := A.1 + B.1 = 2 * M.1 ∧ A.2 + B.2 = 2 * M.2

-- Theorem for part (II)
theorem line_eq (A B M : ℝ × ℝ) (k : ℝ) (h₁ : center_circle M) (h₂ : pts_symmetric A B M) :
  k = 8 / 9 → (∀ x y : ℝ, 8 * x - 9 * y + 25 = 0) :=
sorry

end ellipse_eq_line_eq_l91_91803


namespace perimeter_after_adding_tiles_l91_91724

-- Definition of the initial configuration
def initial_perimeter := 16

-- Definition of the number of additional tiles
def additional_tiles := 3

-- Statement of the problem: to prove that the new perimeter is 22
theorem perimeter_after_adding_tiles : initial_perimeter + 2 * additional_tiles = 22 := 
by 
  -- The number initially added each side exposed would increase the perimeter incremented by 6
  -- You can also assume the boundary conditions for the shared sides reducing.
  sorry

end perimeter_after_adding_tiles_l91_91724


namespace sequence_divisibility_l91_91586

-- Define the sequence
def a (n : ℕ) : ℕ :=
  if n = 0 then 3 else 2 ^ n * n + 1

-- State the theorem
theorem sequence_divisibility (p : ℕ) (hp : Nat.Prime p) (hodd : p % 2 = 1) :
  ∃ m : ℕ, p ∣ a m ∧ p ∣ a (m + 1) := by
  sorry

end sequence_divisibility_l91_91586


namespace problem_l91_91960

theorem problem (a b : ℕ) (h1 : 2^4 + 2^4 = 2^a) (h2 : 3^5 + 3^5 + 3^5 = 3^b) : a + b = 11 :=
by {
  sorry
}

end problem_l91_91960


namespace total_revenue_correct_l91_91343

def price_per_book : ℝ := 25
def revenue_monday : ℝ := 60 * ((price_per_book * 0.9) * 1.05)
def revenue_tuesday : ℝ := 10 * (price_per_book * 1.03)
def revenue_wednesday : ℝ := 20 * ((price_per_book * 0.95) * 1.02)
def revenue_thursday : ℝ := 44 * ((price_per_book * 0.85) * 1.04)
def revenue_friday : ℝ := 66 * (price_per_book * 0.8)

def total_revenue : ℝ :=
  revenue_monday + revenue_tuesday + revenue_wednesday +
  revenue_thursday + revenue_friday

theorem total_revenue_correct :
  total_revenue = 4452.4 :=
by
  rw [total_revenue, revenue_monday, revenue_tuesday, revenue_wednesday, 
      revenue_thursday, revenue_friday]
  -- Verification steps would continue by calculating each term.
  sorry

end total_revenue_correct_l91_91343


namespace xy_value_l91_91751

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 := 
by sorry

end xy_value_l91_91751


namespace max_integers_sum_power_of_two_l91_91432

open Set

/-- Given a finite set of positive integers such that the sum of any two distinct elements is a power of two,
    the cardinality of the set is at most 2. -/
theorem max_integers_sum_power_of_two (S : Finset ℕ) (h_pos : ∀ x ∈ S, 0 < x)
  (h_sum : ∀ {a b : ℕ}, a ∈ S → b ∈ S → a ≠ b → ∃ n : ℕ, a + b = 2^n) : S.card ≤ 2 :=
sorry

end max_integers_sum_power_of_two_l91_91432


namespace right_triangle_min_area_l91_91166

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l91_91166


namespace eval_log32_4_l91_91939

noncomputable def log_base_change (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem eval_log32_4 : log_base_change 32 4 = 2 / 5 := 
by 
  sorry

end eval_log32_4_l91_91939


namespace probability_of_even_distinct_digits_l91_91487

noncomputable def probability_even_distinct_digits : ℚ :=
  let total_numbers := 9000
  let favorable_numbers := 2744
  favorable_numbers / total_numbers

theorem probability_of_even_distinct_digits : 
  probability_even_distinct_digits = 343 / 1125 :=
by
  sorry

end probability_of_even_distinct_digits_l91_91487


namespace megan_markers_l91_91280

def initial_markers : ℕ := 217
def roberts_gift : ℕ := 109
def sarah_took : ℕ := 35

def final_markers : ℕ := initial_markers + roberts_gift - sarah_took

theorem megan_markers : final_markers = 291 := by
  sorry

end megan_markers_l91_91280


namespace find_a_plus_b_l91_91859

theorem find_a_plus_b :
  ∃ (a b : ℝ), (∀ x : ℝ, (3 * (a * x + b) - 6) = 4 * x + 5) ∧ a + b = 5 :=
by 
  sorry

end find_a_plus_b_l91_91859


namespace solve_trigonometric_equation_l91_91013

theorem solve_trigonometric_equation :
  ∃ (S : Finset ℝ), (∀ X ∈ S, 0 < X ∧ X < 360 ∧ 1 + 2 * Real.sin (X * Real.pi / 180) - 4 * (Real.sin (X * Real.pi / 180))^2 - 8 * (Real.sin (X * Real.pi / 180))^3 = 0) ∧ S.card = 4 :=
by
  sorry

end solve_trigonometric_equation_l91_91013


namespace Corey_found_golf_balls_on_Saturday_l91_91208

def goal : ℕ := 48
def golf_balls_found_on_sunday : ℕ := 18
def golf_balls_needed : ℕ := 14
def golf_balls_found_on_saturday : ℕ := 16

theorem Corey_found_golf_balls_on_Saturday :
  (goal - golf_balls_found_on_sunday - golf_balls_needed) = golf_balls_found_on_saturday := 
by
  sorry

end Corey_found_golf_balls_on_Saturday_l91_91208


namespace probability_at_least_one_of_each_color_l91_91778

theorem probability_at_least_one_of_each_color
  (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) (red_balls : ℕ)
  (h_total : total_balls = 16)
  (h_black : black_balls = 8)
  (h_white : white_balls = 5)
  (h_red : red_balls = 3) :
  ((black_balls.choose 1) * (white_balls.choose 1) * (red_balls.choose 1) : ℚ) / total_balls.choose 3 = 3 / 14 :=
by
  sorry

end probability_at_least_one_of_each_color_l91_91778


namespace basketball_games_won_difference_l91_91638

theorem basketball_games_won_difference :
  ∀ (total_games games_won games_lost difference_won_lost : ℕ),
  total_games = 62 →
  games_won = 45 →
  games_lost = 17 →
  difference_won_lost = games_won - games_lost →
  difference_won_lost = 28 :=
by
  intros total_games games_won games_lost difference_won_lost
  intros h_total h_won h_lost h_diff
  rw [h_won, h_lost] at h_diff
  exact h_diff

end basketball_games_won_difference_l91_91638


namespace intersection_A_B_l91_91251

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }
def B : Set ℝ := { x | x < 2 }

theorem intersection_A_B : A ∩ B = { x | -1 ≤ x ∧ x < 2 } := 
by 
  sorry

end intersection_A_B_l91_91251


namespace zero_of_function_is_not_intersection_l91_91612

noncomputable def is_function_zero (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = 0

theorem zero_of_function_is_not_intersection (f : ℝ → ℝ) :
  ¬ (∀ x : ℝ, is_function_zero f x ↔ (f x = 0 ∧ x ∈ {x | f x = 0})) :=
by
  sorry

end zero_of_function_is_not_intersection_l91_91612


namespace four_digit_div_by_25_l91_91533

theorem four_digit_div_by_25 : 
  let count_a := 9 in  -- a ranges from 1 to 9
  let count_b := 10 in  -- b ranges from 0 to 9
  count_a * count_b = 90 := by
  sorry

end four_digit_div_by_25_l91_91533


namespace increasing_interval_f_l91_91737

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem increasing_interval_f :
  ∀ x, (2 < x) → (∃ ε > 0, ∀ δ > 0, δ < ε → f (x + δ) ≥ f x) :=
by
  sorry

end increasing_interval_f_l91_91737


namespace count_lineups_not_last_l91_91417

theorem count_lineups_not_last (n : ℕ) (htallest_not_last : n = 5) :
  ∃ (k : ℕ), k = 96 :=
by { sorry }

end count_lineups_not_last_l91_91417


namespace sin_alpha_is_neg_5_over_13_l91_91837

-- Definition of the problem conditions
variables (α : Real) (h1 : 0 < α) (h2 : α < 2 * Real.pi)
variable (quad4 : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi)
variable (h3 : Real.tan α = -5 / 12)

-- Proof statement
theorem sin_alpha_is_neg_5_over_13:
  Real.sin α = -5 / 13 :=
sorry

end sin_alpha_is_neg_5_over_13_l91_91837


namespace cowboy_shortest_distance_l91_91772

noncomputable def distance : ℝ :=
  let C := (0, 5)
  let B := (-10, 11)
  let C' := (0, -5)
  5 + Real.sqrt ((C'.1 - B.1)^2 + (C'.2 - B.2)^2)

theorem cowboy_shortest_distance :
  distance = 5 + Real.sqrt 356 :=
by
  sorry

end cowboy_shortest_distance_l91_91772


namespace dave_tickets_l91_91655

-- Definitions based on given conditions
def initial_tickets : ℕ := 25
def spent_tickets : ℕ := 22
def additional_tickets : ℕ := 15

-- Proof statement to demonstrate Dave would have 18 tickets
theorem dave_tickets : initial_tickets - spent_tickets + additional_tickets = 18 := by
  sorry

end dave_tickets_l91_91655


namespace a_n_formula_T_n_formula_l91_91384

variable (a : Nat → Int) (b : Nat → Int)
variable (S : Nat → Int) (T : Nat → Int)
variable (d a_1 : Int)

-- Conditions:
axiom a_seq_arith : ∀ n, a (n + 1) = a n + d
axiom S_arith : ∀ n, S n = n * (a 1 + a n) / 2
axiom S_10 : S 10 = 110
axiom geo_seq : (a 2) ^ 2 = a 1 * a 4
axiom b_def : ∀ n, b n = 1 / ((a n - 1) * (a n + 1))

-- Goals: 
-- 1. Find the general formula for the terms of sequence {a_n}
theorem a_n_formula : ∀ n, a n = 2 * n := sorry

-- 2. Find the sum of the first n terms T_n of the sequence {b_n} given b_n
theorem T_n_formula : ∀ n, T n = 1 / 2 - 1 / (4 * n + 2) := sorry

end a_n_formula_T_n_formula_l91_91384


namespace number_of_questions_is_45_l91_91610

-- Defining the conditions
def test_sections : ℕ := 5
def correct_answers : ℕ := 32
def min_percentage : ℝ := 0.70
def max_percentage : ℝ := 0.77
def question_range_min : ℝ := correct_answers / min_percentage
def question_range_max : ℝ := correct_answers / max_percentage
def multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

-- Statement to prove
theorem number_of_questions_is_45 (x : ℕ) (hx1 : 41 < x) (hx2 : x < 46) (hx3 : multiple_of_5 x) : x = 45 :=
by sorry

end number_of_questions_is_45_l91_91610


namespace lcm_gcf_ratio_240_630_l91_91080

theorem lcm_gcf_ratio_240_630 :
  let a := 240
  let b := 630
  Nat.lcm a b / Nat.gcd a b = 168 := by
  sorry

end lcm_gcf_ratio_240_630_l91_91080


namespace prime_p_satisfies_conditions_l91_91809

theorem prime_p_satisfies_conditions (p : ℕ) (hp1 : Nat.Prime p) (hp2 : p ≠ 2) (hp3 : p ≠ 7) :
  ∃ n : ℕ, n = 29 ∧ ∀ x y : ℕ, (1 ≤ x ∧ x ≤ 29) ∧ (1 ≤ y ∧ y ≤ 29) → (29 ∣ (y^2 - x^p - 26)) :=
sorry

end prime_p_satisfies_conditions_l91_91809


namespace negative_y_implies_negative_y_is_positive_l91_91402

theorem negative_y_implies_negative_y_is_positive (y : ℝ) (h : y < 0) : -y > 0 :=
sorry

end negative_y_implies_negative_y_is_positive_l91_91402


namespace cos_diff_alpha_beta_l91_91976

open Real

theorem cos_diff_alpha_beta (α β : ℝ) (h1 : sin α = 1 / 4) (h2 : β = -α) :
  cos (α - β) = 7 / 8 :=
by
  sorry

end cos_diff_alpha_beta_l91_91976


namespace last_four_digits_of_5_pow_2011_l91_91591

theorem last_four_digits_of_5_pow_2011 :
  (5 ^ 5) % 10000 = 3125 ∧
  (5 ^ 6) % 10000 = 5625 ∧
  (5 ^ 7) % 10000 = 8125 →
  (5 ^ 2011) % 10000 = 8125 :=
by
  sorry

end last_four_digits_of_5_pow_2011_l91_91591


namespace transformed_passes_through_l91_91887

def original_parabola (x : ℝ) : ℝ :=
  -x^2 - 2*x + 3

def transformed_parabola (x : ℝ) : ℝ :=
  -(x - 1)^2 + 2

theorem transformed_passes_through : transformed_parabola (-1) = 1 :=
  by sorry

end transformed_passes_through_l91_91887


namespace joeys_age_next_multiple_l91_91709

-- Definitions of the conditions and problem setup
def joey_age (chloe_age : ℕ) : ℕ := chloe_age + 2
def max_age : ℕ := 2
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Sum of digits function
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Main Lean statement
theorem joeys_age_next_multiple (chloe_age : ℕ) (H1 : is_prime chloe_age)
  (H2 : ∀ n : ℕ, (joey_age chloe_age + n) % (max_age + n) = 0)
  (H3 : ∀ i : ℕ, i < 11 → is_prime (chloe_age + i))
  : sum_of_digits (joey_age chloe_age + 1) = 5 :=
  sorry

end joeys_age_next_multiple_l91_91709


namespace problem_statement_l91_91401

theorem problem_statement (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 5) : b^2 - a^2 = -15 := by
  sorry

end problem_statement_l91_91401


namespace largest_mersenne_prime_less_than_500_l91_91077

theorem largest_mersenne_prime_less_than_500 : ∃ n : ℕ, (Nat.prime n ∧ (2^n - 1) = 127 ∧ ∀ m : ℕ, (Nat.prime m → (2^m - 1) < 500 → (2^m - 1) ≤ 127)) :=
sorry

end largest_mersenne_prime_less_than_500_l91_91077


namespace square_of_binomial_l91_91344

-- Define a condition that the given term is the square of a binomial.
theorem square_of_binomial (a b: ℝ) : (a + b) * (a + b) = (a + b) ^ 2 :=
by {
  -- The proof is omitted.
  sorry
}

end square_of_binomial_l91_91344


namespace range_of_function_l91_91053

theorem range_of_function : 
  ∀ y : ℝ, ∃ x : ℝ, y = 4^x + 2^x - 3 ↔ y > -3 :=
by
  sorry

end range_of_function_l91_91053


namespace exists_rectangle_in_inscribed_right_triangle_l91_91352

theorem exists_rectangle_in_inscribed_right_triangle :
  ∃ (L W : ℝ), 
    (45^2 / (1 + (5/2)^2) = L * L) ∧
    (2 * L = 45) ∧
    (2 * W = 45) ∧
    ((L = 25 ∧ W = 10) ∨ (L = 18.75 ∧ W = 7.5)) :=
by sorry

end exists_rectangle_in_inscribed_right_triangle_l91_91352


namespace average_grade_of_female_students_l91_91439

theorem average_grade_of_female_students
  (avg_all_students : ℝ)
  (avg_male_students : ℝ)
  (num_males : ℕ)
  (num_females : ℕ)
  (total_students := num_males + num_females)
  (total_score_all_students := avg_all_students * total_students)
  (total_score_male_students := avg_male_students * num_males) :
  avg_all_students = 90 →
  avg_male_students = 87 →
  num_males = 8 →
  num_females = 12 →
  ((total_score_all_students - total_score_male_students) / num_females) = 92 := by
  intros h_avg_all h_avg_male h_num_males h_num_females
  sorry

end average_grade_of_female_students_l91_91439


namespace a_in_M_l91_91276

def M : Set ℝ := { x | x ≤ 5 }
def a : ℝ := 2

theorem a_in_M : a ∈ M :=
by
  -- Proof omitted
  sorry

end a_in_M_l91_91276


namespace smallest_area_right_triangle_l91_91138

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l91_91138


namespace fifth_term_arithmetic_sequence_l91_91305

variable (a d : ℤ)

def arithmetic_sequence (n : ℤ) : ℤ :=
  a + (n - 1) * d

theorem fifth_term_arithmetic_sequence :
  arithmetic_sequence a d 20 = 12 →
  arithmetic_sequence a d 21 = 15 →
  arithmetic_sequence a d 5 = -33 :=
by
  intro h20 h21
  sorry

end fifth_term_arithmetic_sequence_l91_91305


namespace simplified_value_of_expression_l91_91460

theorem simplified_value_of_expression :
  (12 ^ 0.6) * (12 ^ 0.4) * (8 ^ 0.2) * (8 ^ 0.8) = 96 := 
by
  sorry

end simplified_value_of_expression_l91_91460


namespace emily_small_gardens_count_l91_91325

-- Definitions based on conditions
def initial_seeds : ℕ := 41
def seeds_planted_in_big_garden : ℕ := 29
def seeds_per_small_garden : ℕ := 4

-- Theorem statement
theorem emily_small_gardens_count (initial_seeds seeds_planted_in_big_garden seeds_per_small_garden : ℕ) :
  initial_seeds = 41 →
  seeds_planted_in_big_garden = 29 →
  seeds_per_small_garden = 4 →
  (initial_seeds - seeds_planted_in_big_garden) / seeds_per_small_garden = 3 :=
by
  intros
  sorry

end emily_small_gardens_count_l91_91325


namespace prob_first_red_light_third_intersection_l91_91193

noncomputable def red_light_at_third_intersection (p : ℝ) (h : p = 2/3) : ℝ :=
(1 - p) * (1 - (1/2)) * (1/2)

theorem prob_first_red_light_third_intersection (h : 2/3 = (2/3 : ℝ)) :
  red_light_at_third_intersection (2/3) h = 1/12 := sorry

end prob_first_red_light_third_intersection_l91_91193


namespace polar_curve_is_circle_l91_91046

theorem polar_curve_is_circle (θ ρ : ℝ) (h : 4 * Real.sin θ = 5 * ρ) : 
  ∃ c : ℝ×ℝ, ∀ (x y : ℝ), x^2 + y^2 = c.1^2 + c.2^2 :=
by
  sorry

end polar_curve_is_circle_l91_91046


namespace smallest_area_of_right_triangle_l91_91115

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l91_91115


namespace job_completion_time_l91_91183

theorem job_completion_time (r_p r_q r_r : ℚ) (h_p : r_p = 1 / 3) (h_q : r_q = 1 / 9) (h_r : r_r = 1 / 6) :
  let work := (r_p * 1) + (r_q * 2) + (r_r * 3)
  work >= 1 → 0 = 0 :=
by
  -- Definitions and given conditions
  assume r_p r_q r_r h_p h_q h_r,
  let work := (r_p * 1) + (r_q * 2) + (r_r * 3),
  assume : work >= 1,
  sorry

end job_completion_time_l91_91183


namespace tom_spending_is_correct_l91_91620

-- Conditions
def cost_per_square_foot : ℕ := 5
def square_feet_per_seat : ℕ := 12
def number_of_seats : ℕ := 500
def construction_multiplier : ℕ := 2
def partner_contribution_ratio : ℚ := 0.40

-- Calculate and verify Tom's spending
def total_square_footage := number_of_seats * square_feet_per_seat
def land_cost := total_square_footage * cost_per_square_foot
def construction_cost := construction_multiplier * land_cost
def total_cost := land_cost + construction_cost
def partner_contribution := partner_contribution_ratio * total_cost
def tom_spending := (1 - partner_contribution_ratio) * total_cost

theorem tom_spending_is_correct : tom_spending = 54000 := 
by 
    -- The theorems calculate specific values 
    sorry

end tom_spending_is_correct_l91_91620


namespace largest_red_points_l91_91975

noncomputable def maximum_red_points {n : ℕ} (points : Finset (ℝ × ℝ))
  (h_collinear : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p3 ∈ points → ¬ collinear p1 p2 p3)
  (coloring : points → Finset (ℝ × ℝ) × Finset (ℝ × ℝ))
  (h_coloring : ∀ t : Finset (ℝ × ℝ), t ⊆ points → t.card = 3 → 
    (∀ p ∈ t, p ∈ (coloring points).1) → ∃ b ∈ (coloring points).2, b ∈ convex_hull (t : Set (ℝ × ℝ))) : ℕ :=
1012

theorem largest_red_points {n : ℕ} (points : Finset (ℝ × ℝ)) :
  n = 2022 →
  (∀ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points → p2 ∈ points → p3 ∈ points → ¬ collinear p1 p2 p3) →
  ∃ coloring : points → Finset (ℝ × ℝ) × Finset (ℝ × ℝ), 
    (∀ t : Finset (ℝ × ℝ), t ⊆ points → t.card = 3 → 
      (∀ p ∈ t, p ∈ (coloring points).1) → ∃ b ∈ (coloring points).2, b ∈ convex_hull (t : Set (ℝ × ℝ))) →
  maximum_red_points points _ coloring _ = 1012 :=
by
  intros h_n h_collinear
  classical
  use (λ pts, (pt : ℝ × ℝ)
    let reds := {p ∈ pts | cond1 p} -- Example condition for red points
    let blues := pts \ reds
    (reds, blues)
  sorry

end largest_red_points_l91_91975


namespace beef_weight_loss_percentage_l91_91337

noncomputable def weight_after_processing : ℝ := 570
noncomputable def weight_before_processing : ℝ := 876.9230769230769

theorem beef_weight_loss_percentage :
  (weight_before_processing - weight_after_processing) / weight_before_processing * 100 = 35 :=
by
  sorry

end beef_weight_loss_percentage_l91_91337


namespace term_largest_binomial_coeff_constant_term_in_expansion_l91_91242

theorem term_largest_binomial_coeff {n : ℕ} (h : n = 8) :
  ∃ (k : ℕ) (coeff : ℤ), coeff * x ^ k = 1120 * x^4 :=
by
  sorry

theorem constant_term_in_expansion :
  ∃ (const : ℤ), const = 1280 :=
by
  sorry

end term_largest_binomial_coeff_constant_term_in_expansion_l91_91242


namespace value_of_fraction_l91_91263

variables {a_1 q : ℝ}

-- Define the conditions and the mathematical equivalent of the problem.
def geometric_sequence (a_1 q : ℝ) (h_pos : a_1 > 0 ∧ q > 0) :=
  2 * a_1 + a_1 * q = a_1 * q^2

theorem value_of_fraction (h_pos : a_1 > 0 ∧ q > 0) (h_geom : geometric_sequence a_1 q h_pos) :
  (a_1 * q^3 + a_1 * q^4) / (a_1 * q^2 + a_1 * q^3) = 2 :=
sorry

end value_of_fraction_l91_91263


namespace right_triangle_area_l91_91359

-- Define the lengths of the legs of the right triangle
def leg_length : ℝ := 1

-- State the theorem
theorem right_triangle_area (a b : ℝ) (h1 : a = leg_length) (h2 : b = leg_length) : 
  (1 / 2) * a * b = 1 / 2 :=
by
  rw [h1, h2]
  -- From the substitutions above, it simplifies to:
  sorry

end right_triangle_area_l91_91359


namespace imaginary_part_of_complex_division_l91_91953

theorem imaginary_part_of_complex_division : 
  let i := Complex.I
  let z := (1 - 2 * i) / (2 - i)
  Complex.im z = -3 / 5 :=
by
  sorry

end imaginary_part_of_complex_division_l91_91953


namespace log_equation_solution_l91_91209

theorem log_equation_solution (x : ℝ) (hx : 0 < x) :
  (Real.log x / Real.log 4) * (Real.log 8 / Real.log x) = Real.log 8 / Real.log 4 ↔ (x = 4 ∨ x = 8) :=
by
  sorry

end log_equation_solution_l91_91209


namespace orthocenter_on_line_AD_l91_91491

theorem orthocenter_on_line_AD
  (A B C D E H O1 O2 O3 : euclidean_space ℝ (fin 2))
  (hO1 : is_circumcenter A B E O1)
  (hO2 : is_circumcenter A D E O2)
  (hO3 : is_circumcenter C D E O3)
  (hH : is_orthocenter O1 O2 O3 H) : 
  collinear_ ℝ ({A, D, H}) :=
sorry

end orthocenter_on_line_AD_l91_91491


namespace chickens_bought_l91_91790

theorem chickens_bought (total_spent : ℤ) (egg_count : ℤ) (egg_price : ℤ) (chicken_price : ℤ) (egg_cost : ℤ := egg_count * egg_price) (chicken_spent : ℤ := total_spent - egg_cost) : total_spent = 88 → egg_count = 20 → egg_price = 2 → chicken_price = 8 → chicken_spent / chicken_price = 6 :=
by
  intros
  sorry

end chickens_bought_l91_91790


namespace ratio_boys_to_girls_l91_91540

theorem ratio_boys_to_girls (g b : ℕ) (h1 : g + b = 30) (h2 : b = g + 3) : 
  (b : ℚ) / g = 16 / 13 := 
by 
  sorry

end ratio_boys_to_girls_l91_91540


namespace Roselyn_initial_books_l91_91873

theorem Roselyn_initial_books :
  ∀ (books_given_to_Rebecca books_remaining books_given_to_Mara total_books_given initial_books : ℕ),
    books_given_to_Rebecca = 40 →
    books_remaining = 60 →
    books_given_to_Mara = 3 * books_given_to_Rebecca →
    total_books_given = books_given_to_Mara + books_given_to_Rebecca →
    initial_books = books_remaining + total_books_given →
    initial_books = 220 :=
by
  intros books_given_to_Rebecca books_remaining books_given_to_Mara total_books_given initial_books
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end Roselyn_initial_books_l91_91873


namespace petya_max_margin_l91_91552

def max_margin_votes (total_votes P1 P2 V1 V2: ℕ) : ℕ := P1 + P2 - (V1 + V2)

theorem petya_max_margin 
  (P1 P2 V1 V2: ℕ)
  (H1: P1 = V1 + 9) 
  (H2: V2 = P2 + 9) 
  (H3: P1 + P2 + V1 + V2 = 27) 
  (H_win: P1 + P2 > V1 + V2) : 
  max_margin_votes 27 P1 P2 V1 V2 = 9 :=
by
  sorry

end petya_max_margin_l91_91552


namespace james_fish_weight_l91_91580

theorem james_fish_weight :
  let trout := 200
  let salmon := trout + (trout * 0.5)
  let tuna := 2 * salmon
  trout + salmon + tuna = 1100 := 
by
  sorry

end james_fish_weight_l91_91580


namespace total_quantities_l91_91452

theorem total_quantities (N : ℕ) (S S₃ S₂ : ℕ)
  (h1 : S = 12 * N)
  (h2 : S₃ = 12)
  (h3 : S₂ = 48)
  (h4 : S = S₃ + S₂) :
  N = 5 :=
by
  sorry

end total_quantities_l91_91452


namespace smallest_area_right_triangle_l91_91141

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l91_91141


namespace element_of_M_l91_91828

def M : Set (ℕ × ℕ) := { (2, 3) }

theorem element_of_M : (2, 3) ∈ M :=
by
  sorry

end element_of_M_l91_91828


namespace smallest_area_right_triangle_l91_91156

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l91_91156


namespace two_circles_common_tangents_l91_91621

theorem two_circles_common_tangents (r : ℝ) (h_r : 0 < r) :
  ¬ ∃ (n : ℕ), n = 2 ∧
  (∀ (config : ℕ), 
    (config = 0 → n = 4) ∨
    (config = 1 → n = 0) ∨
    (config = 2 → n = 3) ∨
    (config = 3 → n = 1)) :=
by
  sorry

end two_circles_common_tangents_l91_91621


namespace finance_specialization_percentage_l91_91970

theorem finance_specialization_percentage (F : ℝ) :
  (76 - 43.333333333333336) = (90 - F) → 
  F = 57.333333333333336 :=
by
  sorry

end finance_specialization_percentage_l91_91970


namespace extremum_of_f_unique_solution_of_equation_l91_91525

noncomputable def f (x m : ℝ) : ℝ := (1/2) * x^2 - m * Real.log x

theorem extremum_of_f (m : ℝ) (h_pos : 0 < m) :
  ∃ x_min : ℝ, x_min = Real.sqrt m ∧
  ∀ x : ℝ, 0 < x → f x m ≥ f (Real.sqrt m) m :=
sorry

theorem unique_solution_of_equation (m : ℝ) (h_ge_one : 1 ≤ m) :
  ∃! x : ℝ, 0 < x ∧ f x m = x^2 - (m + 1) * x :=
sorry

#check extremum_of_f -- Ensure it can be checked
#check unique_solution_of_equation -- Ensure it can be checked

end extremum_of_f_unique_solution_of_equation_l91_91525


namespace min_value_frac_l91_91512

theorem min_value_frac (x y : ℝ) (h₁ : x + y = 1) (h₂ : x > 0) (h₃ : y > 0) : 
  ∃ c, (∀ (a b : ℝ), (a + b = 1) → (a > 0) → (b > 0) → (1/a + 4/b) ≥ c) ∧ c = 9 :=
by
  sorry

end min_value_frac_l91_91512


namespace smallest_area_right_triangle_l91_91128

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l91_91128


namespace degree_of_g_l91_91583

noncomputable def poly_degree (p : Polynomial ℝ) : ℕ :=
  Polynomial.natDegree p

theorem degree_of_g
  (f g : Polynomial ℝ)
  (h : Polynomial ℝ := f.comp g - g)
  (hf : poly_degree f = 3)
  (hh : poly_degree h = 8) :
  poly_degree g = 3 :=
sorry

end degree_of_g_l91_91583


namespace class_total_students_l91_91771

-- Definitions based on the conditions
def number_students_group : ℕ := 12
def frequency_group : ℚ := 0.25

-- Statement of the problem in Lean
theorem class_total_students (n : ℕ) (h : frequency_group = number_students_group / n) : n = 48 :=
by
  sorry

end class_total_students_l91_91771


namespace reflect_parabola_y_axis_l91_91284

theorem reflect_parabola_y_axis (x y : ℝ) :
  (y = 2 * (x - 1)^2 - 4) → (y = 2 * (-x - 1)^2 - 4) :=
sorry

end reflect_parabola_y_axis_l91_91284


namespace range_of_vector_magnitude_l91_91694

variable {V : Type} [NormedAddCommGroup V]

theorem range_of_vector_magnitude
  (A B C : V)
  (h_AB : ‖A - B‖ = 8)
  (h_AC : ‖A - C‖ = 5) :
  3 ≤ ‖B - C‖ ∧ ‖B - C‖ ≤ 13 :=
sorry

end range_of_vector_magnitude_l91_91694


namespace mandy_more_cinnamon_l91_91277

def cinnamon : ℝ := 0.67
def nutmeg : ℝ := 0.5

theorem mandy_more_cinnamon : cinnamon - nutmeg = 0.17 :=
by
  sorry

end mandy_more_cinnamon_l91_91277


namespace novelists_count_l91_91194

theorem novelists_count (n p : ℕ) (h1 : n / (n + p) = 5 / 8) (h2 : n + p = 24) : n = 15 :=
sorry

end novelists_count_l91_91194


namespace pos_solution_sum_l91_91734

theorem pos_solution_sum (c d : ℕ) (h_c_pos : 0 < c) (h_d_pos : 0 < d) :
  (∃ x : ℝ, x ^ 2 + 16 * x = 100 ∧ x = Real.sqrt c - d) → c + d = 172 :=
by
  intro h
  sorry

end pos_solution_sum_l91_91734


namespace oranges_and_apples_costs_l91_91770

theorem oranges_and_apples_costs :
  ∃ (x y : ℚ), 7 * x + 5 * y = 13 ∧ 3 * x + 4 * y = 8 ∧ 37 * x + 45 * y = 93 :=
by 
  sorry

end oranges_and_apples_costs_l91_91770


namespace distance_covered_l91_91589

-- Define the rate and time as constants
def rate : ℝ := 4 -- 4 miles per hour
def time : ℝ := 2 -- 2 hours

-- Theorem statement: Verify the distance covered
theorem distance_covered : rate * time = 8 := 
by
  sorry

end distance_covered_l91_91589


namespace find_value_l91_91380

theorem find_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2010 = 2011 := 
sorry

end find_value_l91_91380


namespace birthday_candles_l91_91346

theorem birthday_candles :
  ∀ (candles_Ambika : ℕ) (candles_Aniyah : ℕ),
  candles_Ambika = 4 →
  candles_Aniyah = 6 * 4 →
  (candles_Ambika + candles_Aniyah) / 2 = 14 :=
by
  intros candles_Ambika candles_Aniyah h_Ambika h_Aniyah
  rw [h_Ambika, h_Aniyah]
  norm_num

end birthday_candles_l91_91346


namespace tony_drive_time_l91_91895

noncomputable def time_to_first_friend (d₁ d₂ t₂ : ℝ) : ℝ :=
  let v := d₂ / t₂
  d₁ / v

theorem tony_drive_time (d₁ d₂ t₂ : ℝ) (h_d₁ : d₁ = 120) (h_d₂ : d₂ = 200) (h_t₂ : t₂ = 5) : 
    time_to_first_friend d₁ d₂ t₂ = 3 := by
  rw [h_d₁, h_d₂, h_t₂]
  -- Further simplification would follow here based on the proof steps, which we are omitting
  sorry

end tony_drive_time_l91_91895


namespace arithmetic_sequence_nth_term_l91_91885

theorem arithmetic_sequence_nth_term (x n : ℝ) 
  (h1 : 3*x - 4 = a1)
  (h2 : 7*x - 14 = a2)
  (h3 : 4*x + 6 = a3)
  (h4 : a_n = 3012) :
n = 392 :=
  sorry

end arithmetic_sequence_nth_term_l91_91885


namespace sum_of_squares_divisible_by_7_implies_product_divisible_by_49_l91_91574

theorem sum_of_squares_divisible_by_7_implies_product_divisible_by_49 (a b : ℕ) 
  (h : (a * a + b * b) % 7 = 0) : (a * b) % 49 = 0 :=
sorry

end sum_of_squares_divisible_by_7_implies_product_divisible_by_49_l91_91574


namespace Dylan_needs_two_trays_l91_91665

noncomputable def ice_cubes_glass : ℕ := 8
noncomputable def ice_cubes_pitcher : ℕ := 2 * ice_cubes_glass
noncomputable def tray_capacity : ℕ := 12
noncomputable def total_ice_cubes_used : ℕ := ice_cubes_glass + ice_cubes_pitcher
noncomputable def number_of_trays : ℕ := total_ice_cubes_used / tray_capacity

theorem Dylan_needs_two_trays : number_of_trays = 2 := by
  sorry

end Dylan_needs_two_trays_l91_91665


namespace find_number_l91_91629

theorem find_number (N : ℝ) (h : 0.4 * (3 / 5) * N = 36) : N = 150 :=
sorry

end find_number_l91_91629


namespace don_can_consume_more_rum_l91_91595

theorem don_can_consume_more_rum (rum_given_by_sally : ℕ) (multiplier : ℕ) (already_consumed : ℕ) :
    let max_consumption := multiplier * rum_given_by_sally in
    rum_given_by_sally = 10 →
    multiplier = 3 →
    already_consumed = 12 →
    max_consumption - (rum_given_by_sally + already_consumed) = 8 :=
by
  intros rum_given_by_sally multiplier already_consumed h1 h2 h3
  dsimp only
  rw [h1, h2, h3]
  norm_num
  sorry

end don_can_consume_more_rum_l91_91595


namespace equal_angles_count_l91_91894

-- Definitions corresponding to the problem conditions
def fast_clock_angle (t : ℝ) : ℝ := |30 * t - 5.5 * (t * 60)|
def slow_clock_angle (t : ℝ) : ℝ := |15 * t - 2.75 * (t * 60)|

theorem equal_angles_count :
  ∃ n : ℕ, n = 18 ∧ ∀ t : ℝ, 0 ≤ t ∧ t ≤ 12 →
  fast_clock_angle t = slow_clock_angle t ↔ n = 18 :=
sorry

end equal_angles_count_l91_91894


namespace arithmetic_sequence_sum_nine_l91_91680

variable {a : ℕ → ℤ} -- Define a_n sequence as a function from ℕ to ℤ

-- Define the conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ (d : ℤ), ∀ n m, a (n + m) = a n + m * d

def fifth_term_is_two (a : ℕ → ℤ) : Prop :=
  a 5 = 2

-- Lean statement to prove the sum of the first 9 terms
theorem arithmetic_sequence_sum_nine (a : ℕ → ℤ)
  (h1 : is_arithmetic_sequence a)
  (h2 : fifth_term_is_two a) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 2 * 9 :=
sorry

end arithmetic_sequence_sum_nine_l91_91680


namespace carpet_interior_length_l91_91187

/--
A carpet is designed using three different colors, forming three nested rectangles with different areas in an arithmetic progression. 
The innermost rectangle has a width of two feet. Each of the two colored borders is 2 feet wide on all sides.
Determine the length in feet of the innermost rectangle. 
-/
theorem carpet_interior_length 
  (x : ℕ) -- length of the innermost rectangle
  (hp : ∀ (a b c : ℕ), a = 2 * x ∧ b = (4 * x + 24) ∧ c = (4 * x + 56) → (b - a) = (c - b)) 
  : x = 4 :=
by
  sorry

end carpet_interior_length_l91_91187


namespace smallest_area_of_right_triangle_l91_91114

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l91_91114


namespace petya_max_votes_difference_l91_91565

theorem petya_max_votes_difference :
  ∃ (P1 P2 V1 V2 : ℕ), 
    P1 = V1 + 9 ∧ 
    V2 = P2 + 9 ∧ 
    P1 + P2 + V1 + V2 = 27 ∧ 
    P1 + P2 > V1 + V2 ∧ 
    (P1 + P2) - (V1 + V2) = 9 := 
by
  sorry

end petya_max_votes_difference_l91_91565


namespace ratio_of_green_to_yellow_l91_91323

-- Given conditions
def total_rows : ℕ := 6
def flowers_per_row : ℕ := 13
def yellow_flowers : ℕ := 12
def red_flowers : ℕ := 42
def total_flowers : ℕ := total_rows * flowers_per_row

-- Number of green flowers calculated from given condition
def green_flowers : ℕ := total_flowers - (yellow_flowers + red_flowers)

-- Our goal statement
theorem ratio_of_green_to_yellow :
  (green_flowers:ℚ) / yellow_flowers = 2 := 
by
  sorry

end ratio_of_green_to_yellow_l91_91323


namespace lcm_of_two_numbers_l91_91943

-- Define the numbers involved
def a : ℕ := 28
def b : ℕ := 72

-- Define the expected LCM result
def lcm_ab : ℕ := 504

-- State the problem as a theorem
theorem lcm_of_two_numbers : Nat.lcm a b = lcm_ab :=
by sorry

end lcm_of_two_numbers_l91_91943


namespace unique_intersection_y_eq_bx2_5x_2_y_eq_neg2x_neg2_iff_b_eq_49_div_16_l91_91661

theorem unique_intersection_y_eq_bx2_5x_2_y_eq_neg2x_neg2_iff_b_eq_49_div_16 
  (b : ℝ) : 
  (∃ (x : ℝ), bx^2 + 7*x + 4 = 0 ∧ ∀ (x' : ℝ), bx^2 + 7*x' + 4 ≠ 0) ↔ b = 49 / 16 :=
by
  sorry

end unique_intersection_y_eq_bx2_5x_2_y_eq_neg2x_neg2_iff_b_eq_49_div_16_l91_91661


namespace superior_sequences_count_l91_91676

noncomputable def number_of_superior_sequences (n : ℕ) : ℕ :=
  Nat.choose (2 * n + 1) (n + 1) * 2^n

theorem superior_sequences_count (n : ℕ) (h : 2 ≤ n) 
  (x : Fin (n + 1) → ℤ)
  (h1 : ∀ i, 0 ≤ i ∧ i ≤ n → |x i| ≤ n)
  (h2 : ∀ i j, 0 ≤ i ∧ i < j ∧ j ≤ n → x i ≠ x j)
  (h3 : ∀ (i j k : Nat), 0 ≤ i ∧ i < j ∧ j < k ∧ k ≤ n → 
    max (|x k - x i|) (|x k - x j|) = 
    (|x i - x j| + |x j - x k| + |x k - x i|) / 2) :
  number_of_superior_sequences n = Nat.choose (2 * n + 1) (n + 1) * 2^n :=
sorry

end superior_sequences_count_l91_91676


namespace no_nat_solutions_l91_91458
-- Import the Mathlib library

-- Lean statement for the proof problem
theorem no_nat_solutions (x : ℕ) : ¬ (19 * x^2 + 97 * x = 1997) :=
by {
  -- Solution omitted
  sorry
}

end no_nat_solutions_l91_91458


namespace smallest_area_of_right_triangle_l91_91164

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l91_91164


namespace price_of_each_shirt_is_15_30_l91_91846

theorem price_of_each_shirt_is_15_30:
  ∀ (shorts_price : ℝ) (num_shorts : ℕ) (shirt_num : ℕ) (total_paid : ℝ) (discount : ℝ),
  shorts_price = 15 →
  num_shorts = 3 →
  shirt_num = 5 →
  total_paid = 117 →
  discount = 0.10 →
  (total_paid - (num_shorts * shorts_price - discount * (num_shorts * shorts_price))) / shirt_num = 15.30 :=
by 
  sorry

end price_of_each_shirt_is_15_30_l91_91846


namespace compute_series_sum_l91_91493

noncomputable def term (n : ℕ) : ℝ := (5 * n - 2) / (3 ^ n)

theorem compute_series_sum : 
  ∑' n, term n = 11 / 4 := 
sorry

end compute_series_sum_l91_91493


namespace smallest_area_right_triangle_l91_91176

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l91_91176


namespace least_value_l91_91206

-- Define the quadratic function and its conditions
def quadratic_function (p q r : ℝ) (x : ℝ) : ℝ :=
  p * x^2 + q * x + r

-- Define the conditions for p, q, and r
def conditions (p q r : ℝ) : Prop :=
  p > 0 ∧ (q^2 - 4 * p * r < 0)

-- State the theorem that given the conditions the least value is (4pr - q^2) / 4p
theorem least_value (p q r : ℝ) (h : conditions p q r) :
  ∃ x : ℝ, (∀ y : ℝ, quadratic_function p q r y ≥ quadratic_function p q r x) ∧
  quadratic_function p q r x = (4 * p * r - q^2) / (4 * p) :=
sorry

end least_value_l91_91206


namespace max_single_player_salary_l91_91787

variable (n : ℕ) (m : ℕ) (p : ℕ) (s : ℕ)

theorem max_single_player_salary
  (h1 : n = 18)
  (h2 : ∀ i : ℕ, i < n → p ≥ 20000)
  (h3 : s = 800000)
  (h4 : n * 20000 ≤ s) :
  ∃ x : ℕ, x = 460000 :=
by
  sorry

end max_single_player_salary_l91_91787


namespace annie_serious_accident_probability_l91_91489

theorem annie_serious_accident_probability :
  (∀ temperature : ℝ, temperature < 32 → ∃ skid_chance_increase : ℝ, skid_chance_increase = 5 * ⌊ (32 - temperature) / 3 ⌋ / 100) →
  (∀ control_regain_chance : ℝ, control_regain_chance = 0.4) →
  (∀ control_loss_chance : ℝ, control_loss_chance = 1 - control_regain_chance) →
  (temperature = 8) →
  (serious_accident_probability = skid_chance_increase * control_loss_chance) →
  serious_accident_probability = 0.24 := by
  sorry

end annie_serious_accident_probability_l91_91489


namespace total_cookies_l91_91930

variable (ChrisCookies KennyCookies GlennCookies : ℕ)
variable (KennyHasCookies : GlennCookies = 4 * KennyCookies)
variable (ChrisHasCookies : ChrisCookies = KennyCookies / 2)
variable (GlennHas24Cookies : GlennCookies = 24)

theorem total_cookies : GlennCookies + KennyCookies + ChrisCookies = 33 := 
by
  have KennyCookiesEq : KennyCookies = 24 / 4 := by 
    rw [GlennHas24Cookies, mul_div_cancel_left, nat.mul_comm, nat.one_div, nat.div_self] ; trivial
  have ChrisCookiesEq : ChrisCookies = 6 / 2 := by 
    rw [KennyCookiesEq, ChrisHasCookies]
  rw [ChrisCookiesEq, KennyCookiesEq, GlennHas24Cookies]
  exact sorry

end total_cookies_l91_91930


namespace domain_of_f_eq_l91_91047

noncomputable def domain_of_f (x : ℝ) : Prop :=
  (x + 1 ≥ 0) ∧ (x ≠ 0)

theorem domain_of_f_eq :
  { x : ℝ | domain_of_f x} = { x : ℝ | -1 ≤ x ∧ x < 0 } ∪ { x : ℝ | 0 < x } :=
by
  sorry

end domain_of_f_eq_l91_91047


namespace smallest_area_of_right_triangle_l91_91086

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l91_91086


namespace negative_integer_solutions_l91_91050

theorem negative_integer_solutions (x : ℤ) : 3 * x + 1 ≥ -5 ↔ x = -2 ∨ x = -1 := 
by
  sorry

end negative_integer_solutions_l91_91050


namespace smallest_area_of_right_triangle_l91_91104

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l91_91104


namespace average_age_of_coaches_l91_91292

theorem average_age_of_coaches 
  (total_members : ℕ) (average_age_members : ℕ)
  (num_girls : ℕ) (average_age_girls : ℕ)
  (num_boys : ℕ) (average_age_boys : ℕ)
  (num_coaches : ℕ) :
  total_members = 30 →
  average_age_members = 20 →
  num_girls = 10 →
  average_age_girls = 18 →
  num_boys = 15 →
  average_age_boys = 19 →
  num_coaches = 5 →
  (600 - (num_girls * average_age_girls) - (num_boys * average_age_boys)) / num_coaches = 27 :=
by
  intros
  sorry

end average_age_of_coaches_l91_91292


namespace find_m_of_parallelepiped_volume_l91_91892

theorem find_m_of_parallelepiped_volume 
  {m : ℝ} 
  (h_pos : m > 0) 
  (h_vol : abs (3 * (m^2 - 9) - 2 * (4 * m - 15) + 2 * (12 - 5 * m)) = 20) : 
  m = (9 + Real.sqrt 249) / 6 :=
sorry

end find_m_of_parallelepiped_volume_l91_91892


namespace John_l91_91541

theorem John's_score_in_blackjack
  (Theodore_score : ℕ)
  (Zoey_cards : List ℕ)
  (winning_score : ℕ)
  (John_score : ℕ)
  (h1 : Theodore_score = 13)
  (h2 : Zoey_cards = [11, 3, 5])
  (h3 : winning_score = 19)
  (h4 : Zoey_cards.sum = winning_score)
  (h5 : winning_score ≠ Theodore_score) :
  John_score < 19 :=
by
  -- Here we would provide the proof if required
  sorry

end John_l91_91541


namespace count_four_digit_ints_divisible_by_25_l91_91534

def is_four_digit_int_of_form_ab25 (n : ℕ) : Prop :=
  ∃ a b, (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ n = 1000 * a + 100 * b + 25

theorem count_four_digit_ints_divisible_by_25 :
  {n : ℕ | is_four_digit_int_of_form_ab25 n}.to_finset.card = 90 :=
by
  sorry

end count_four_digit_ints_divisible_by_25_l91_91534


namespace find_d_l91_91966

theorem find_d (d : ℤ) (h : ∀ x : ℤ, 8 * x^3 + 23 * x^2 + d * x + 45 = 0 → 2 * x + 5 = 0) : 
  d = 163 := 
sorry

end find_d_l91_91966


namespace cement_total_l91_91391

-- Defining variables for the weights of cement
def weight_self : ℕ := 215
def weight_son : ℕ := 137

-- Defining the function that calculates the total weight of the cement
def total_weight (a b : ℕ) : ℕ := a + b

-- Theorem statement: Proving the total cement weight is 352 lbs
theorem cement_total : total_weight weight_self weight_son = 352 :=
by
  sorry

end cement_total_l91_91391


namespace condition_sufficient_but_not_necessary_l91_91014

variables (p q : Prop)

theorem condition_sufficient_but_not_necessary (hpq : ∀ q, (¬p → ¬q)) (hpns : ¬ (¬p → ¬q ↔ p → q)) : (p → q) ∧ ¬ (q → p) :=
by {
  sorry
}

end condition_sufficient_but_not_necessary_l91_91014


namespace growth_rate_l91_91733

variable (x : ℝ)

def initial_investment : ℝ := 500
def expected_investment : ℝ := 720

theorem growth_rate (x : ℝ) (h : 500 * (1 + x)^2 = 720) : x = 0.2 :=
by
  sorry

end growth_rate_l91_91733


namespace investment_at_6_percent_l91_91640

variables (x y : ℝ)

-- Conditions from the problem
def total_investment : Prop := x + y = 15000
def total_interest : Prop := 0.06 * x + 0.075 * y = 1023

-- Conclusion to prove
def invest_6_percent (x : ℝ) : Prop := x = 6800

theorem investment_at_6_percent (h1 : total_investment x y) (h2 : total_interest x y) : invest_6_percent x :=
by
  sorry

end investment_at_6_percent_l91_91640


namespace smallest_right_triangle_area_l91_91095

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l91_91095


namespace second_divisor_l91_91812

theorem second_divisor (x : ℕ) (k q : ℤ) : 
  (197 % 13 = 2) → 
  (x > 13) → 
  (197 % x = 5) → 
  x = 16 :=
by sorry

end second_divisor_l91_91812


namespace largest_A_l91_91373

def F (n a : ℕ) : ℕ :=
  let q := a / n
  let r := a % n
  q + r

theorem largest_A :
  ∃ n₁ n₂ n₃ n₄ n₅ n₆ : ℕ,
  (0 < n₁ ∧ 0 < n₂ ∧ 0 < n₃ ∧ 0 < n₄ ∧ 0 < n₅ ∧ 0 < n₆) ∧
  ∀ a, (1 ≤ a ∧ a ≤ 53590) -> 
    (F n₆ (F n₅ (F n₄ (F n₃ (F n₂ (F n₁ a))))) = 1) :=
sorry

end largest_A_l91_91373


namespace total_spent_is_correct_l91_91024

def trumpet : ℝ := 149.16
def music_tool : ℝ := 9.98
def song_book : ℝ := 4.14
def trumpet_maintenance_accessories : ℝ := 21.47
def valve_oil_original : ℝ := 8.20
def valve_oil_discount_rate : ℝ := 0.20
def valve_oil_discounted : ℝ := valve_oil_original * (1 - valve_oil_discount_rate)
def band_t_shirt : ℝ := 14.95
def sales_tax_rate : ℝ := 0.065

def total_before_tax : ℝ :=
  trumpet + music_tool + song_book + trumpet_maintenance_accessories + valve_oil_discounted + band_t_shirt

def sales_tax : ℝ := total_before_tax * sales_tax_rate

def total_amount_spent : ℝ := total_before_tax + sales_tax

theorem total_spent_is_correct : total_amount_spent = 219.67 := by
  sorry

end total_spent_is_correct_l91_91024


namespace students_remaining_after_third_stop_l91_91967

theorem students_remaining_after_third_stop
  (initial_students : ℕ)
  (third : ℚ) (stops : ℕ)
  (one_third_off : third = 1 / 3)
  (initial_students_eq : initial_students = 64)
  (stops_eq : stops = 3)
  : 64 * ((2 / 3) ^ 3) = 512 / 27 :=
by 
  sorry

end students_remaining_after_third_stop_l91_91967


namespace petya_maximum_margin_l91_91559

def max_margin_votes (total_votes first_period_margin last_period_margin : ℕ) (petya_vasaya_margin : ℕ) :=
  ∀ (P1 P2 V1 V2 : ℕ),
    (P1 + P2 + V1 + V2 = total_votes) →
    (P1 = V1 + first_period_margin) →
    (V2 = P2 + last_period_margin) →
    (P1 + P2 > V1 + V2) →
    petya_vasaya_margin = P1 + P2 - (V1 + V2)

theorem petya_maximum_margin
  (total_votes first_period_margin last_period_margin : ℕ)
  (h_total_votes: total_votes = 27)
  (h_first_period_margin: first_period_margin = 9)
  (h_last_period_margin: last_period_margin = 9):
  ∃ (petya_vasaya_margin : ℕ), max_margin_votes total_votes first_period_margin last_period_margin petya_vasaya_margin ∧ petya_vasaya_margin = 9 :=
by {
    sorry
}

end petya_maximum_margin_l91_91559


namespace vertex_of_parabola_l91_91442

theorem vertex_of_parabola :
  ∃ (h k : ℝ), (∀ x : ℝ, -2 * (x - h) ^ 2 + k = -2 * (x - 2) ^ 2 - 5) ∧ h = 2 ∧ k = -5 :=
by
  sorry

end vertex_of_parabola_l91_91442


namespace find_C_l91_91342

theorem find_C (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 350) : C = 50 := 
by
  sorry

end find_C_l91_91342


namespace polygon_interior_angle_sum_360_l91_91918

theorem polygon_interior_angle_sum_360 (n : ℕ) (h : (n-2) * 180 = 360) : n = 4 :=
sorry

end polygon_interior_angle_sum_360_l91_91918


namespace exists_root_in_interval_l91_91221

noncomputable def f (x : ℝ) : ℝ := 1 / (x - 1) - Real.log (x - 1) / Real.log 2

theorem exists_root_in_interval :
  ∃ c : ℝ, 2 < c ∧ c < 3 ∧ f c = 0 :=
by
  -- Proof goes here
  sorry

end exists_root_in_interval_l91_91221


namespace smallest_area_of_right_triangle_l91_91113

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l91_91113


namespace smallest_area_of_right_triangle_l91_91084

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l91_91084


namespace smallest_area_right_triangle_l91_91089

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l91_91089


namespace evaluate_expression_l91_91807

theorem evaluate_expression : (1 / (5^2)^4) * 5^15 = 5^7 :=
by
  sorry

end evaluate_expression_l91_91807


namespace john_must_deliver_1063_pizzas_l91_91710

-- Declare all the given conditions
def car_cost : ℕ := 8000
def maintenance_cost : ℕ := 500
def pizza_income (p : ℕ) : ℕ := 12 * p
def gas_cost (p : ℕ) : ℕ := 4 * p

-- Define the function that returns the net earnings
def net_earnings (p : ℕ) := pizza_income p - gas_cost p

-- Define the total expenses
def total_expenses : ℕ := car_cost + maintenance_cost

-- Define the minimum number of pizzas John must deliver
def minimum_pizzas (p : ℕ) : Prop := net_earnings p ≥ total_expenses

-- State the theorem that needs to be proved
theorem john_must_deliver_1063_pizzas : minimum_pizzas 1063 := by
  sorry

end john_must_deliver_1063_pizzas_l91_91710


namespace tangent_line_at_one_l91_91527

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x

theorem tangent_line_at_one (a b : ℝ) (h_tangent : ∀ x, f x = a * x + b) : 
  a + b = 1 := 
sorry

end tangent_line_at_one_l91_91527


namespace rectangle_length_l91_91769

theorem rectangle_length (b l : ℝ) 
  (h1 : l = 2 * b)
  (h2 : (l - 5) * (b + 5) = l * b + 75) : l = 40 := by
  sorry

end rectangle_length_l91_91769


namespace max_ratio_of_right_triangle_l91_91336

theorem max_ratio_of_right_triangle (a b c: ℝ) (h1: (1/2) * a * b = 30) (h2: a^2 + b^2 = c^2) : 
  (∀ x y z, (1/2 * x * y = 30) → (x^2 + y^2 = z^2) → 
  (x + y + z) / 30 ≤ (7.75 + 7.75 + 10.95) / 30) :=
by 
  sorry  -- The proof will show the maximum value is approximately 0.8817.

noncomputable def max_value := (7.75 + 7.75 + 10.95) / 30

end max_ratio_of_right_triangle_l91_91336


namespace smallest_area_of_right_triangle_l91_91161

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l91_91161


namespace garden_length_increase_l91_91473

variable (L W : ℝ)  -- Original length and width
variable (X : ℝ)    -- Percentage increase in length

theorem garden_length_increase :
  (1 + X / 100) * 0.8 = 1.1199999999999999 → X = 40 :=
by
  sorry

end garden_length_increase_l91_91473


namespace fraction_exponent_simplification_l91_91296

theorem fraction_exponent_simplification :
  (7^((1 : ℝ) / 4)) / (7^((1 : ℝ) / 6)) = 7^((1 : ℝ) / 12) :=
sorry

end fraction_exponent_simplification_l91_91296


namespace karen_kept_cookies_l91_91271

def total_cookies : ℕ := 50
def cookies_to_grandparents : ℕ := 8
def number_of_classmates : ℕ := 16
def cookies_per_classmate : ℕ := 2

theorem karen_kept_cookies (x : ℕ) 
  (H1 : x = total_cookies - (cookies_to_grandparents + number_of_classmates * cookies_per_classmate)) :
  x = 10 :=
by
  -- proof omitted
  sorry

end karen_kept_cookies_l91_91271


namespace pioneer_ages_l91_91315

def pioneer_data (Burov Gridnev Klimenko Kolya Petya Grisha : String) (Burov_age Gridnev_age Klimenko_age Petya_age Grisha_age : ℕ) :=
  Burov ≠ Kolya ∧
  Petya_age = 12 ∧
  Gridnev_age = Petya_age + 1 ∧
  Grisha_age = Petya_age + 1 ∧
  Burov_age = Grisha_age ∧
-- defining the names corresponding to conditions given in problem
  Burov = Grisha ∧ Gridnev = Kolya ∧ Klimenko = Petya 

theorem pioneer_ages (Burov Gridnev Klimenko Kolya Petya Grisha : String) (Burov_age Gridnev_age Klimenko_age Petya_age Grisha_age : ℕ)
  (h : pioneer_data Burov Gridnev Klimenko Kolya Petya Grisha Burov_age Gridnev_age Klimenko_age Petya_age Grisha_age) :
  (Burov, Burov_age) = (Grisha, 13) ∧ 
  (Gridnev, Gridnev_age) = (Kolya, 13) ∧ 
  (Klimenko, Klimenko_age) = (Petya, 12) :=
by
  sorry

end pioneer_ages_l91_91315


namespace exponentiation_problem_l91_91396

theorem exponentiation_problem (a b : ℤ) (h : 3 ^ a * 9 ^ b = (1 / 3 : ℚ)) : a + 2 * b = -1 :=
sorry

end exponentiation_problem_l91_91396


namespace smallest_right_triangle_area_l91_91110

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l91_91110


namespace total_pictures_480_l91_91042

noncomputable def total_pictures (pictures_per_album : ℕ) (num_albums : ℕ) : ℕ :=
  pictures_per_album * num_albums

theorem total_pictures_480 : total_pictures 20 24 = 480 :=
  by
    sorry

end total_pictures_480_l91_91042


namespace prob_one_boy_one_girl_l91_91653

-- Defining the probabilities of birth
def prob_boy := 2 / 3
def prob_girl := 1 / 3

-- Calculating the probability of all boys
def prob_all_boys := prob_boy ^ 4

-- Calculating the probability of all girls
def prob_all_girls := prob_girl ^ 4

-- Calculating the probability of having at least one boy and one girl
def prob_at_least_one_boy_and_one_girl := 1 - (prob_all_boys + prob_all_girls)

-- Proof statement
theorem prob_one_boy_one_girl : prob_at_least_one_boy_and_one_girl = 64 / 81 :=
by sorry

end prob_one_boy_one_girl_l91_91653


namespace arithmetic_mean_six_expressions_l91_91601

theorem arithmetic_mean_six_expressions (x : ℝ)
  (h : (x + 8 + 15 + 2 * x + 13 + 2 * x + 4 + 3 * x + 5) / 6 = 30) : x = 13.5 :=
by
  sorry

end arithmetic_mean_six_expressions_l91_91601


namespace cats_added_l91_91785

theorem cats_added (siamese_cats house_cats total_cats : ℕ) 
  (h1 : siamese_cats = 13) 
  (h2 : house_cats = 5) 
  (h3 : total_cats = 28) : 
  total_cats - (siamese_cats + house_cats) = 10 := 
by 
  sorry

end cats_added_l91_91785


namespace unique_solution_to_equation_l91_91598

theorem unique_solution_to_equation (x y z : ℤ) 
    (h : 5 * x^3 + 11 * y^3 + 13 * z^3 = 0) : x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end unique_solution_to_equation_l91_91598


namespace general_formula_of_sequence_l91_91021

open Nat

def a : ℕ → ℝ
| 0 => 1
| (n+1) => 3 * a n / (3 + a n)

theorem general_formula_of_sequence (n : ℕ) : a (n) = 3 / (n + 2) :=
by
  induction n with
  | zero => 
    simp [a]
  | succ n ih =>
    calc
      a (n + 1) = 3 * a n / (3 + a n) : by simp [a]
      ... = 3 * (3 / (n + 2)) / (3 + (3 / (n + 2))) : by rw [ih]
      ... = 3 * (3 / (n + 2)) / ((3 * (n + 2) + 3) / (n + 2)) : by field_simp
      ... = 3 * 3 / (3 * (n + 2) + 3) : by ring
      ... = 9 / (3 * (n + 2) + 3) : by ring
      ... = 3 / (n + 2 + 1) : by ring

end general_formula_of_sequence_l91_91021


namespace remaining_kids_l91_91065

def initial_kids : Float := 22.0
def kids_who_went_home : Float := 14.0

theorem remaining_kids : initial_kids - kids_who_went_home = 8.0 :=
by 
  sorry

end remaining_kids_l91_91065


namespace petya_wins_max_margin_l91_91548

theorem petya_wins_max_margin {P1 P2 V1 V2 : ℕ} 
  (h1 : P1 = V1 + 9)
  (h2 : V2 = P2 + 9)
  (h3 : P1 + P2 + V1 + V2 = 27)
  (h4 : P1 + P2 > V1 + V2) :
  ∃ m : ℕ, m = 9 ∧ P1 + P2 - (V1 + V2) = m :=
by
  sorry

end petya_wins_max_margin_l91_91548


namespace Petya_victory_margin_l91_91570

theorem Petya_victory_margin :
  ∃ P1 P2 V1 V2 : ℕ, P1 = V1 + 9 ∧ V2 = P2 + 9 ∧ P1 + P2 + V1 + V2 = 27 ∧ 
  (P1 + P2) > (V1 + V2) ∧ (P1 + P2) - (V1 + V2) = 9 :=
begin
  sorry
end

end Petya_victory_margin_l91_91570


namespace sum_of_exponents_l91_91465

-- Define the expression inside the radical
def radicand (a b c : ℝ) : ℝ := 40 * a^6 * b^3 * c^14

-- Define the simplified expression outside the radical
def simplified_expr (a b c : ℝ) : ℝ := (2 * a^2 * b * c^4)

-- State the theorem to prove the sum of the exponents of the variables outside the radical
theorem sum_of_exponents (a b c : ℝ) : 
  let exponents_sum := 2 + 1 + 4
  exponents_sum = 7 :=
by
  sorry

end sum_of_exponents_l91_91465


namespace don_can_have_more_rum_l91_91596

-- Definitions based on conditions:
def given_rum : ℕ := 10
def max_consumption_rate : ℕ := 3
def already_had : ℕ := 12

-- Maximum allowed consumption calculation:
def max_allowed_rum : ℕ := max_consumption_rate * given_rum

-- Remaining rum calculation:
def remaining_rum : ℕ := max_allowed_rum - already_had

-- Proof statement of the problem:
theorem don_can_have_more_rum : remaining_rum = 18 := by
  -- Let's compute directly:
  have h1 : max_allowed_rum = 30 := by
    simp [max_allowed_rum, max_consumption_rate, given_rum]

  have h2 : remaining_rum = 18 := by
    simp [remaining_rum, h1, already_had]

  exact h2

end don_can_have_more_rum_l91_91596


namespace min_value_of_expression_l91_91520

theorem min_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 
  x^2 + 4 * y^2 + 2 * x * y ≥ 3 / 4 :=
sorry

end min_value_of_expression_l91_91520


namespace kevin_final_cards_l91_91986

-- Define the initial conditions and problem
def initial_cards : ℕ := 20
def found_cards : ℕ := 47
def lost_cards_1 : ℕ := 7
def lost_cards_2 : ℕ := 12
def won_cards : ℕ := 15

-- Define the function to calculate the final count
def final_cards (initial found lost1 lost2 won : ℕ) : ℕ :=
  (initial + found - lost1 - lost2 + won)

-- Statement of the problem to be proven
theorem kevin_final_cards :
  final_cards initial_cards found_cards lost_cards_1 lost_cards_2 won_cards = 63 :=
by
  sorry

end kevin_final_cards_l91_91986


namespace odd_even_subsets_equal_sum_capacities_equal_sum_capacities_odd_subsets_l91_91587

open Finset

-- Condition Representation in Lean:
def Sn (n : ℕ) : Finset ℕ := range (n + 1)

def capacity (X : Finset ℕ) : ℕ := X.sum id

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_even (n : ℕ) : Prop := n % 2 = 0

def odd_subsets (n : ℕ) : Finset (Finset ℕ) := filter (λ X, is_odd (capacity X)) (powerset (Sn n))

def even_subsets (n : ℕ) : Finset (Finset ℕ) := filter (λ X, is_even (capacity X)) (powerset (Sn n))

-- Questions Representation in Lean:
-- 1. Prove that the number of odd subsets of Sn is equal to the number of even subsets of Sn.
theorem odd_even_subsets_equal (n : ℕ) : (odd_subsets n).card = (even_subsets n).card := sorry

-- 2. Prove that when n ≥ 3, the sum of the capacities of all odd subsets of Sn is equal to the sum of the capacities of all even subsets of Sn.
theorem sum_capacities_equal (n : ℕ) (h : n ≥ 3) : (odd_subsets n).sum capacity = (even_subsets n).sum capacity := sorry

-- 3. Find the sum of the capacities of all odd subsets of Sn when n ≥ 3.
theorem sum_capacities_odd_subsets (n : ℕ) (h : n ≥ 3) : (odd_subsets n).sum capacity = 2^(n-3) * n * (n + 1) := sorry

end odd_even_subsets_equal_sum_capacities_equal_sum_capacities_odd_subsets_l91_91587


namespace pool_half_capacity_at_6_hours_l91_91058

noncomputable def double_volume_every_hour (t : ℕ) : ℕ := 2 ^ t

theorem pool_half_capacity_at_6_hours (V : ℕ) (h : ∀ t : ℕ, V = double_volume_every_hour 8) : double_volume_every_hour 6 = V / 2 := by
  sorry

end pool_half_capacity_at_6_hours_l91_91058


namespace edward_initial_lives_l91_91354

def initialLives (lives_lost lives_left : Nat) : Nat :=
  lives_lost + lives_left

theorem edward_initial_lives (lost left : Nat) (H_lost : lost = 8) (H_left : left = 7) :
  initialLives lost left = 15 :=
by
  sorry

end edward_initial_lives_l91_91354


namespace lcm_28_72_l91_91946

theorem lcm_28_72 : Nat.lcm 28 72 = 504 := by
  sorry

end lcm_28_72_l91_91946


namespace find_a_value_l91_91705

theorem find_a_value 
  (a : ℝ) 
  (P : ℝ × ℝ) 
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 - 2 * a * x + 2 * y - 1 = 0)
  (M N : ℝ × ℝ)
  (tangent_condition : (N.snd - M.snd) / (N.fst - M.fst) + (M.fst + N.fst - 2) / (M.snd + N.snd) = 0) : 
  a = 3 ∨ a = -2 := 
sorry

end find_a_value_l91_91705


namespace scientific_notation_l91_91286

theorem scientific_notation : 899000 = 8.99 * 10^5 := 
by {
  -- We start by recognizing that we need to express 899,000 in scientific notation.
  -- Placing the decimal point after the first non-zero digit yields 8.99.
  -- Count the number of places moved (5 places to the left).
  -- Thus, 899,000 in scientific notation is 8.99 * 10^5.
  sorry
}

end scientific_notation_l91_91286


namespace determine_m_in_hexadecimal_conversion_l91_91409

theorem determine_m_in_hexadecimal_conversion :
  ∃ m : ℕ, 1 * 6^5 + 3 * 6^4 + m * 6^3 + 5 * 6^2 + 0 * 6^1 + 2 * 6^0 = 12710 ∧ m = 4 :=
by
  sorry

end determine_m_in_hexadecimal_conversion_l91_91409


namespace smallest_area_right_triangle_l91_91136

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l91_91136


namespace avg_salary_officers_correct_l91_91973

def total_employees := 465
def avg_salary_employees := 120
def non_officers := 450
def avg_salary_non_officers := 110
def officers := 15

theorem avg_salary_officers_correct : (15 * 420) = ((total_employees * avg_salary_employees) - (non_officers * avg_salary_non_officers)) := by
  sorry

end avg_salary_officers_correct_l91_91973


namespace petya_max_margin_l91_91555

def max_margin_votes (total_votes P1 P2 V1 V2: ℕ) : ℕ := P1 + P2 - (V1 + V2)

theorem petya_max_margin 
  (P1 P2 V1 V2: ℕ)
  (H1: P1 = V1 + 9) 
  (H2: V2 = P2 + 9) 
  (H3: P1 + P2 + V1 + V2 = 27) 
  (H_win: P1 + P2 > V1 + V2) : 
  max_margin_votes 27 P1 P2 V1 V2 = 9 :=
by
  sorry

end petya_max_margin_l91_91555


namespace prime_pair_perfect_square_l91_91218

theorem prime_pair_perfect_square (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h : ∃ a : ℕ, p^2 + p * q + q^2 = a^2) : (p = 3 ∧ q = 5) ∨ (p = 5 ∧ q = 3) := 
sorry

end prime_pair_perfect_square_l91_91218


namespace sum_of_x_and_y_l91_91403

theorem sum_of_x_and_y (x y : ℝ) 
  (h₁ : |x| + x + 5 * y = 2)
  (h₂ : |y| - y + x = 7) : 
  x + y = 3 := 
sorry

end sum_of_x_and_y_l91_91403


namespace quadratic_equation_reciprocal_integer_roots_l91_91223

noncomputable def quadratic_equation_conditions (a b c : ℝ) : Prop :=
  (∃ r : ℝ, (r * (1/r) = 1) ∧ (r + (1/r) = 4)) ∧ 
  (c = a) ∧ 
  (b = -4 * a)

theorem quadratic_equation_reciprocal_integer_roots (a b c : ℝ) (h1 : quadratic_equation_conditions a b c) : 
  c = a ∧ b = -4 * a :=
by
  obtain ⟨r, hr₁, hr₂⟩ := h1.1
  sorry

end quadratic_equation_reciprocal_integer_roots_l91_91223


namespace minimum_treasures_count_l91_91795

theorem minimum_treasures_count :
  ∃ (n : ℕ), n ≤ 30 ∧
    (
      (∀ (i : ℕ), (i < 15 → "Exactly under 15 signs a treasure is buried." → count_treasure i = 15) ∧
                  (i < 8 → "Exactly under 8 signs a treasure is buried." → count_treasure i = 8) ∧
                  (i < 4 → "Exactly under 4 signs a treasure is buried." → count_treasure i = 4) ∧
                  (i < 3 → "Exactly under 3 signs a treasure is buried." → count_treasure i = 3)
    ) ∧
    truthful (i : ℕ) → ¬ buried i → i )
    → n = 15 :=
sorry

end minimum_treasures_count_l91_91795


namespace range_of_a_neg_p_true_l91_91236

theorem range_of_a_neg_p_true :
  (∀ x : ℝ, x ∈ Set.Ioo (-2:ℝ) 0 → x^2 + (2*a - 1)*x + a ≠ 0) →
  ∀ a : ℝ, a ∈ Set.Icc 0 ((2 + Real.sqrt 3) / 2) :=
sorry

end range_of_a_neg_p_true_l91_91236


namespace remainder_is_37_l91_91463

theorem remainder_is_37
    (d q v r : ℕ)
    (h1 : d = 15968)
    (h2 : q = 89)
    (h3 : v = 179)
    (h4 : d = q * v + r) :
  r = 37 :=
sorry

end remainder_is_37_l91_91463


namespace Irja_wins_probability_l91_91268

noncomputable def probability_irja_wins : ℚ :=
  let X0 : ℚ := 4 / 7
  X0

theorem Irja_wins_probability :
  probability_irja_wins = 4 / 7 :=
sorry

end Irja_wins_probability_l91_91268


namespace hyperbola_asymptotes_l91_91009

theorem hyperbola_asymptotes
    (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 = Real.sqrt (1 + (b^2) / (a^2))) :
    (∀ x y : ℝ, (y = x * Real.sqrt 3) ∨ (y = -x * Real.sqrt 3)) :=
by
  sorry

end hyperbola_asymptotes_l91_91009


namespace albania_inequality_l91_91856

variable (a b c r R s : ℝ)
variable (h1 : a + b > c)
variable (h2 : b + c > a)
variable (h3 : c + a > b)
variable (h4 : r > 0)
variable (h5 : R > 0)
variable (h6 : s = (a + b + c) / 2)

theorem albania_inequality :
    1 / (a + b) + 1 / (a + c) + 1 / (b + c) ≤ r / (16 * R * s) + s / (16 * R * r) + 11 / (8 * s) :=
sorry

end albania_inequality_l91_91856


namespace possible_sets_B_l91_91957

def A : Set ℤ := {-1}

def isB (B : Set ℤ) : Prop :=
  A ∪ B = {-1, 3}

theorem possible_sets_B : ∀ B : Set ℤ, isB B → B = {3} ∨ B = {-1, 3} :=
by
  intros B hB
  sorry

end possible_sets_B_l91_91957


namespace largest_mersenne_prime_less_than_500_l91_91076

theorem largest_mersenne_prime_less_than_500 : ∃ n : ℕ, (Nat.prime n ∧ (2^n - 1) = 127 ∧ ∀ m : ℕ, (Nat.prime m → (2^m - 1) < 500 → (2^m - 1) ≤ 127)) :=
sorry

end largest_mersenne_prime_less_than_500_l91_91076


namespace maximum_m2_n2_l91_91935

theorem maximum_m2_n2 
  (m n : ℤ)
  (hm : 1 ≤ m ∧ m ≤ 1981) 
  (hn : 1 ≤ n ∧ n ≤ 1981)
  (h : (n^2 - m*n - m^2)^2 = 1) :
  m^2 + n^2 ≤ 3524578 :=
sorry

end maximum_m2_n2_l91_91935


namespace exponentiation_problem_l91_91890

theorem exponentiation_problem :
  (-0.125 ^ 2003) * (-8 ^ 2004) = -8 := 
sorry

end exponentiation_problem_l91_91890


namespace mandy_cinnamon_amount_correct_l91_91278

def mandy_cinnamon_amount (nutmeg : ℝ) (cinnamon : ℝ) : Prop :=
  cinnamon = nutmeg + 0.17

theorem mandy_cinnamon_amount_correct :
  mandy_cinnamon_amount 0.5 0.67 :=
by
  sorry

end mandy_cinnamon_amount_correct_l91_91278


namespace weighted_avg_M_B_eq_l91_91306

-- Define the weightages and the given weighted total marks equation
def weight_physics : ℝ := 1.5
def weight_chemistry : ℝ := 2
def weight_mathematics : ℝ := 1.25
def weight_biology : ℝ := 1.75
def weighted_total_M_B : ℝ := 250
def weighted_sum_M_B : ℝ := weight_mathematics + weight_biology

-- Theorem statement: Prove that the weighted average mark for mathematics and biology is 83.33
theorem weighted_avg_M_B_eq :
  (weighted_total_M_B / weighted_sum_M_B) = 83.33 :=
by
  sorry

end weighted_avg_M_B_eq_l91_91306


namespace product_of_squares_l91_91225

theorem product_of_squares (x : ℝ) (h : |5 * x| + 4 = 49) : x^2 * (if x = 9 then 9 else -9)^2 = 6561 :=
by
  sorry

end product_of_squares_l91_91225


namespace increase_in_votes_l91_91842

noncomputable def initial_vote_for (y : ℝ) : ℝ := 500 - y
noncomputable def revote_for (y : ℝ) : ℝ := (10 / 9) * y

theorem increase_in_votes {x x' y m : ℝ}
  (H1 : x + y = 500)
  (H2 : y - x = m)
  (H3 : x' - y = 2 * m)
  (H4 : x' + y = 500)
  (H5 : x' = (10 / 9) * y)
  (H6 : y = 282) :
  revote_for y - initial_vote_for y = 95 :=
by sorry

end increase_in_votes_l91_91842


namespace width_of_foil_covered_prism_l91_91313

theorem width_of_foil_covered_prism (L W H : ℝ) 
  (h1 : W = 2 * L)
  (h2 : W = 2 * H)
  (h3 : L * W * H = 128)
  (h4 : L = H) :
  W + 2 = 8 :=
sorry

end width_of_foil_covered_prism_l91_91313


namespace prob_allergic_prescribed_l91_91485

def P (a : Prop) : ℝ := sorry

axiom P_conditional (A B : Prop) : P B > 0 → P (A ∧ B) = P A * P (B ∧ A) / P B

def A : Prop := sorry -- represent the event that a patient is prescribed Undetenin
def B : Prop := sorry -- represent the event that a patient is allergic to Undetenin

axiom P_A : P A = 0.10
axiom P_B_given_A : P (B ∧ A) / P A = 0.02
axiom P_B : P B = 0.04

theorem prob_allergic_prescribed : P (A ∧ B) / P B = 0.05 :=
by
  have h1 : P (A ∧ B) / P A = 0.10 * 0.02 := sorry -- using definition of P_A and P_B_given_A
  have h2 : P (A ∧ B) = 0.002 := sorry -- calculating the numerator P(B and A)
  exact sorry -- use the axiom P_B to complete the theorem

end prob_allergic_prescribed_l91_91485


namespace lcm_of_two_numbers_l91_91944

-- Define the numbers involved
def a : ℕ := 28
def b : ℕ := 72

-- Define the expected LCM result
def lcm_ab : ℕ := 504

-- State the problem as a theorem
theorem lcm_of_two_numbers : Nat.lcm a b = lcm_ab :=
by sorry

end lcm_of_two_numbers_l91_91944


namespace area_ratio_of_regular_polygons_l91_91920

noncomputable def area_ratio (r : ℝ) : ℝ :=
  let A6 := (3 * Real.sqrt 3 / 2) * r^2
  let s8 := r * Real.sqrt (2 - Real.sqrt 2)
  let A8 := 2 * (1 + Real.sqrt 2) * (s8 ^ 2)
  A8 / A6

theorem area_ratio_of_regular_polygons (r : ℝ) :
  area_ratio r = 4 * (1 + Real.sqrt 2) * (2 - Real.sqrt 2) / (3 * Real.sqrt 3) :=
  sorry

end area_ratio_of_regular_polygons_l91_91920


namespace consistency_condition_l91_91502

variable 
  (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℝ)

theorem consistency_condition :
  (∃ x y : ℝ, 
    a1 * x + b1 * y = c1 ∧ 
    a2 * x + b2 * y = c2 ∧ 
    a3 * x + b3 * y = c3) ↔ 
  (a1 * (b2 * c3 - b3 * c2) + 
   a2 * (b3 * c1 - c3 * b1) + 
   a3 * (b1 * c2 - b2 * c1) = 0) :=
by sorry

end consistency_condition_l91_91502


namespace vectors_opposite_direction_l91_91254

noncomputable def a : ℝ × ℝ := (-2, 4)
noncomputable def b : ℝ × ℝ := (1, -2)

theorem vectors_opposite_direction : a = (-2 : ℝ) • b :=
by
  sorry

end vectors_opposite_direction_l91_91254


namespace range_of_a_l91_91517

def p (a : ℝ) : Prop := 0 < a ∧ a < 1
def q (a : ℝ) : Prop := a > 1 / 8

def resolution (a : ℝ) : Prop :=
(p a ∨ q a) ∧ ¬(p a ∧ q a) → (0 < a ∧ a ≤ 1 / 8) ∨ a ≥ 1

theorem range_of_a (a : ℝ) : resolution a := sorry

end range_of_a_l91_91517


namespace quotient_of_poly_div_l91_91900

theorem quotient_of_poly_div :
  (10 * X^4 - 5 * X^3 + 3 * X^2 + 11 * X - 6) / (5 * X^2 + 7) =
  2 * X^2 - X - (11 / 5) :=
sorry

end quotient_of_poly_div_l91_91900


namespace smallest_area_right_triangle_l91_91154

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l91_91154


namespace Sarah_score_l91_91997

theorem Sarah_score (G S : ℕ) (h1 : S = G + 60) (h2 : (S + G) / 2 = 108) : S = 138 :=
by
  sorry

end Sarah_score_l91_91997


namespace star_polygon_points_l91_91413

theorem star_polygon_points (p : ℕ) (ϕ : ℝ) :
  (∀ i : Fin p, ∃ Ci Di : ℝ, Ci = Di + 15) →
  (p * ϕ + p * (ϕ + 15) = 360) →
  p = 24 :=
by
  sorry

end star_polygon_points_l91_91413


namespace ratio_of_triangle_areas_l91_91316

theorem ratio_of_triangle_areas (a k : ℝ) (h_pos_a : 0 < a) (h_pos_k : 0 < k)
    (h_triangle_division : true) (h_square_area : ∃ s, s = a^2) (h_area_one_triangle : ∃ t, t = k * a^2) :
    ∃ r, r = (1 / (4 * k)) :=
by
  sorry

end ratio_of_triangle_areas_l91_91316


namespace trays_needed_to_fill_ice_cubes_l91_91668

-- Define the initial conditions
def ice_cubes_in_glass : Nat := 8
def multiplier_for_pitcher : Nat := 2
def spaces_per_tray : Nat := 12

-- Define the total ice cubes used
def total_ice_cubes_used : Nat := ice_cubes_in_glass + multiplier_for_pitcher * ice_cubes_in_glass

-- State the Lean theorem to be proven: The number of trays needed
theorem trays_needed_to_fill_ice_cubes : 
  total_ice_cubes_used / spaces_per_tray = 2 :=
  by 
  sorry

end trays_needed_to_fill_ice_cubes_l91_91668


namespace clive_change_l91_91201

theorem clive_change (total_money : ℝ) (num_olives_needed : ℕ) (olives_per_jar : ℕ) (cost_per_jar : ℝ)
  (h1 : total_money = 10)
  (h2 : num_olives_needed = 80)
  (h3 : olives_per_jar = 20)
  (h4 : cost_per_jar = 1.5) : total_money - (num_olives_needed / olives_per_jar) * cost_per_jar = 4 := by
  sorry

end clive_change_l91_91201


namespace total_kids_receive_macarons_l91_91590

theorem total_kids_receive_macarons :
  let mitch_good := 18
  let joshua := 26 -- 20 + 6
  let joshua_good := joshua - 3
  let miles := joshua * 2
  let miles_good := miles
  let renz := (3 * miles) / 4 - 1
  let renz_good := renz - 4
  let leah_good := 35 - 5
  let total_good := mitch_good + joshua_good + miles_good + renz_good + leah_good 
  let kids_with_3_macarons := 10
  let macaron_per_3 := kids_with_3_macarons * 3
  let remaining_macarons := total_good - macaron_per_3
  let kids_with_2_macarons := remaining_macarons / 2
  kids_with_3_macarons + kids_with_2_macarons = 73 :=
by 
  sorry

end total_kids_receive_macarons_l91_91590


namespace budget_remaining_l91_91781

noncomputable def solve_problem : Nat :=
  let total_budget := 325
  let cost_flasks := 150
  let cost_test_tubes := (2 / 3 : ℚ) * cost_flasks
  let cost_safety_gear := (1 / 2 : ℚ) * cost_test_tubes
  let total_expenses := cost_flasks + cost_test_tubes + cost_safety_gear
  total_budget - total_expenses

theorem budget_remaining : solve_problem = 25 := by
  sorry

end budget_remaining_l91_91781


namespace balls_left_correct_l91_91834

def initial_balls : ℕ := 10
def balls_removed : ℕ := 3
def balls_left : ℕ := initial_balls - balls_removed

theorem balls_left_correct : balls_left = 7 := 
by
  -- Proof omitted
  sorry

end balls_left_correct_l91_91834


namespace two_digit_sum_l91_91584

theorem two_digit_sum (x y n : ℕ) (hx : 10 ≤ x ∧ x < 100)
  (hy : 10 ≤ y ∧ y < 100) (h_rev : y = (x % 10) * 10 + x / 10)
  (h_diff_square : x^2 - y^2 = n^2) : x + y + n = 154 :=
sorry

end two_digit_sum_l91_91584


namespace Dylan_needs_two_trays_l91_91666

noncomputable def ice_cubes_glass : ℕ := 8
noncomputable def ice_cubes_pitcher : ℕ := 2 * ice_cubes_glass
noncomputable def tray_capacity : ℕ := 12
noncomputable def total_ice_cubes_used : ℕ := ice_cubes_glass + ice_cubes_pitcher
noncomputable def number_of_trays : ℕ := total_ice_cubes_used / tray_capacity

theorem Dylan_needs_two_trays : number_of_trays = 2 := by
  sorry

end Dylan_needs_two_trays_l91_91666


namespace smallest_area_right_triangle_l91_91132

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l91_91132


namespace smallest_area_of_right_triangle_l91_91101

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l91_91101


namespace min_workers_needed_to_make_profit_l91_91775

def wage_per_worker_per_hour := 20
def fixed_cost := 800
def units_per_worker_per_hour := 6
def price_per_unit := 4.5
def hours_per_workday := 9

theorem min_workers_needed_to_make_profit : ∃ (n : ℕ), 243 * n > 800 + 180 * n ∧ n ≥ 13 :=
by
  sorry

end min_workers_needed_to_make_profit_l91_91775


namespace tan_difference_l91_91399

theorem tan_difference (α β : ℝ) (h1 : Real.tan α = 3) (h2 : Real.tan β = 4 / 3) :
  Real.tan (α - β) = 1 / 3 :=
by
  sorry

end tan_difference_l91_91399


namespace gcd_expression_l91_91241

theorem gcd_expression (a : ℤ) (k : ℤ) (h1 : a = k * 1171) (h2 : k % 2 = 1) (prime_1171 : Prime 1171) : 
  Int.gcd (3 * a^2 + 35 * a + 77) (a + 15) = 1 :=
by
  sorry

end gcd_expression_l91_91241


namespace minimum_boxes_cost_300_muffins_l91_91637

theorem minimum_boxes_cost_300_muffins :
  ∃ (L_used M_used S_used : ℕ), 
    L_used + M_used + S_used = 28 ∧ 
    (L_used = 10 ∧ M_used = 15 ∧ S_used = 3) ∧ 
    (L_used * 15 + M_used * 9 + S_used * 5 = 300) ∧ 
    (L_used * 5 + M_used * 3 + S_used * 2 = 101) ∧ 
    (L_used ≤ 10 ∧ M_used ≤ 15 ∧ S_used ≤ 25) :=
by
  -- The proof is omitted (theorem statement only).
  sorry

end minimum_boxes_cost_300_muffins_l91_91637


namespace wash_cycle_time_l91_91312

-- Definitions for the conditions
def num_loads : Nat := 8
def dry_cycle_time_minutes : Nat := 60
def total_time_hours : Nat := 14
def total_time_minutes : Nat := total_time_hours * 60

-- The actual statement we need to prove
theorem wash_cycle_time (x : Nat) (h : num_loads * x + num_loads * dry_cycle_time_minutes = total_time_minutes) : x = 45 :=
by
  sorry

end wash_cycle_time_l91_91312


namespace john_buys_360_packs_l91_91849

def John_buys_packs (classes students_per_class packs_per_student total_packs : ℕ) : Prop :=
  classes = 6 →
  students_per_class = 30 →
  packs_per_student = 2 →
  total_packs = (classes * students_per_class) * packs_per_student
  → total_packs = 360

theorem john_buys_360_packs : John_buys_packs 6 30 2 360 :=
by { intros, sorry }

end john_buys_360_packs_l91_91849


namespace number_of_sick_animals_l91_91059

def total_animals := 26 + 40 + 34  -- Total number of animals at Stacy's farm
def sick_fraction := 1 / 2  -- Half of all animals get sick

-- Defining sick animals for each type
def sick_chickens := 26 * sick_fraction
def sick_piglets := 40 * sick_fraction
def sick_goats := 34 * sick_fraction

-- The main theorem to prove
theorem number_of_sick_animals :
  sick_chickens + sick_piglets + sick_goats = 50 :=
by
  -- Skeleton of the proof that is to be completed later
  sorry

end number_of_sick_animals_l91_91059


namespace susan_initial_amount_l91_91731

theorem susan_initial_amount :
  ∃ S: ℝ, (S - (1/5 * S + 1/4 * S + 120) = 1200) → S = 2400 :=
by
  sorry

end susan_initial_amount_l91_91731


namespace jerry_removed_figures_l91_91708

-- Definitions based on conditions
def initialFigures : ℕ := 3
def addedFigures : ℕ := 4
def currentFigures : ℕ := 6

-- Total figures after adding
def totalFigures := initialFigures + addedFigures

-- Proof statement defining how many figures were removed
theorem jerry_removed_figures : (totalFigures - currentFigures) = 1 := by
  sorry

end jerry_removed_figures_l91_91708


namespace shaded_area_l91_91490

theorem shaded_area (R : ℝ) (π : ℝ) (h1 : π * (R / 2)^2 * 2 = 1) : 
  (π * R^2 - (π * (R / 2)^2 * 2)) = 1 := 
by
  sorry

end shaded_area_l91_91490


namespace smallest_area_right_triangle_l91_91127

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l91_91127


namespace count_divisible_by_25_l91_91532

-- Define the conditions
def is_positive_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def ends_in_25 (n : ℕ) : Prop := n % 100 = 25

-- Define the main statement to prove
theorem count_divisible_by_25 : 
  (∃ (count : ℕ), count = 90 ∧
  ∀ n, is_positive_four_digit n ∧ ends_in_25 n → count = 90) :=
by {
  -- Outline the proof
  sorry
}

end count_divisible_by_25_l91_91532


namespace office_light_ratio_l91_91867

theorem office_light_ratio (bedroom_light: ℕ) (living_room_factor: ℕ) (total_energy: ℕ) 
  (time: ℕ) (ratio: ℕ) (office_light: ℕ) :
  bedroom_light = 6 →
  living_room_factor = 4 →
  total_energy = 96 →
  time = 2 →
  ratio = 3 →
  total_energy = (bedroom_light * time) + (office_light * time) + ((bedroom_light * living_room_factor) * time) →
  (office_light / bedroom_light) = ratio :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4] at h6
  -- The actual solution steps would go here
  sorry

end office_light_ratio_l91_91867


namespace arithmetic_sequence_problem_l91_91820

variable {a : ℕ → ℕ} -- Assuming a_n is a function from natural numbers to natural numbers

theorem arithmetic_sequence_problem (h1 : a 1 + a 2 = 10) (h2 : a 4 = a 3 + 2) :
  a 3 + a 4 = 18 :=
sorry

end arithmetic_sequence_problem_l91_91820


namespace petya_max_margin_l91_91554

def max_margin_votes (total_votes P1 P2 V1 V2: ℕ) : ℕ := P1 + P2 - (V1 + V2)

theorem petya_max_margin 
  (P1 P2 V1 V2: ℕ)
  (H1: P1 = V1 + 9) 
  (H2: V2 = P2 + 9) 
  (H3: P1 + P2 + V1 + V2 = 27) 
  (H_win: P1 + P2 > V1 + V2) : 
  max_margin_votes 27 P1 P2 V1 V2 = 9 :=
by
  sorry

end petya_max_margin_l91_91554


namespace election_votes_l91_91066

theorem election_votes (V : ℝ) (h1 : 0.56 * V - 0.44 * V = 288) : 0.56 * V = 1344 :=
by 
  sorry

end election_votes_l91_91066


namespace intervals_monotonicity_f_intervals_monotonicity_g_and_extremum_l91_91185

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem intervals_monotonicity_f :
  ∀ k : ℤ,
    (∀ x : ℝ, k * Real.pi ≤ x ∧ x ≤ k * Real.pi + Real.pi / 2 → f x = Real.cos (2 * x)) ∧
    (∀ x : ℝ, k * Real.pi + Real.pi / 2 ≤ x ∧ x ≤ k * Real.pi + Real.pi → f x = Real.cos (2 * x)) :=
sorry

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 6)

theorem intervals_monotonicity_g_and_extremum :
  ∀ x : ℝ,
    (-Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3 → g x = Real.cos (2 * (x + Real.pi / 6))) ∧
    (Real.pi / 3 ≤ x ∧ x ≤ 2 * Real.pi / 3 → g x = Real.cos (2 * (x + Real.pi / 6))) ∧
    (∀ x : ℝ, -Real.pi / 6 ≤ x ∧ x ≤ 2 * Real.pi / 3 → (g x ≤ 1 ∧ g x ≥ -1)) :=
sorry

end intervals_monotonicity_f_intervals_monotonicity_g_and_extremum_l91_91185


namespace independence_of_A_and_D_l91_91308

noncomputable def balls : Finset ℕ := {1, 2, 3, 4, 5, 6}
noncomputable def draw_one : ℕ := (1 : ℕ)
noncomputable def draw_two : ℕ := (2 : ℕ)

def event_A : ℕ → Prop := λ n, n = 1
def event_B : ℕ → Prop := λ n, n = 2
def event_C : ℕ × ℕ → Prop := λ pair, (pair.1 + pair.2) = 8
def event_D : ℕ × ℕ → Prop := λ pair, (pair.1 + pair.2) = 7

def prob (event : ℕ → Prop) : ℚ := 1 / 6
def joint_prob (event1 event2 : ℕ → Prop) : ℚ := (1 / 36)

theorem independence_of_A_and_D :
  joint_prob (λ n, event_A n) (λ n, event_D (draw_one, draw_two)) = prob event_A * prob (λ n, event_D (draw_one, draw_two)) :=
by
  sorry

end independence_of_A_and_D_l91_91308


namespace projection_of_c_onto_b_l91_91692

open Real

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := sqrt (b.1^2 + b.2^2)
  let scalar := dot_product / magnitude_b
  (scalar * b.1 / magnitude_b, scalar * b.2 / magnitude_b)

theorem projection_of_c_onto_b :
  let a := (2, 3)
  let b := (-4, 7)
  let c := (-a.1, -a.2)
  vector_projection c b = (-sqrt 65 / 5, -sqrt 65 / 5) :=
by sorry

end projection_of_c_onto_b_l91_91692


namespace find_a_l91_91988

noncomputable def a_b_c_complex (a b c : ℂ) : Prop :=
  a.re = a ∧ a + b + c = 4 ∧ a * b + b * c + c * a = 6 ∧ a * b * c = 8

theorem find_a (a b c : ℂ) (h : a_b_c_complex a b c) : a = 3 :=
by
  sorry

end find_a_l91_91988


namespace smallest_area_right_triangle_l91_91134

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l91_91134


namespace positive_integer_solutions_condition_l91_91514

theorem positive_integer_solutions_condition (a : ℕ) (A B : ℝ) :
  (∃ (x y z : ℕ), x^2 + y^2 + z^2 = (13 * a)^2 ∧
  x^2 * (A * x^2 + B * y^2) + y^2 * (A * y^2 + B * z^2) + z^2 * (A * z^2 + B * x^2) = (1/4) * (2 * A + B) * (13 * a)^4)
  ↔ A = (1 / 2) * B := 
sorry

end positive_integer_solutions_condition_l91_91514


namespace petya_max_votes_difference_l91_91563

theorem petya_max_votes_difference :
  ∃ (P1 P2 V1 V2 : ℕ), 
    P1 = V1 + 9 ∧ 
    V2 = P2 + 9 ∧ 
    P1 + P2 + V1 + V2 = 27 ∧ 
    P1 + P2 > V1 + V2 ∧ 
    (P1 + P2) - (V1 + V2) = 9 := 
by
  sorry

end petya_max_votes_difference_l91_91563


namespace possible_values_f2001_l91_91741

noncomputable def f : ℕ → ℝ := sorry

lemma functional_equation (a b d : ℕ) (h₁ : 1 < a) (h₂ : 1 < b) (h₃ : d = Nat.gcd a b) :
  f (a * b) = f d * (f (a / d) + f (b / d)) :=
sorry

theorem possible_values_f2001 :
  f 2001 = 0 ∨ f 2001 = 1 / 2 :=
sorry

end possible_values_f2001_l91_91741


namespace point_on_graph_l91_91764

def lies_on_graph (x y : ℝ) (f : ℝ → ℝ) : Prop :=
  y = f x

theorem point_on_graph :
  lies_on_graph (-2) 0 (λ x => (1 / 2) * x + 1) :=
by
  sorry

end point_on_graph_l91_91764


namespace isosceles_triangle_area_of_triangle_l91_91978

variables {A B C : ℝ} {a b c : ℝ}

-- Conditions
axiom triangle_sides (a b c : ℝ) (A B C : ℝ) : c = 2
axiom cosine_condition (a b c : ℝ) (A B C : ℝ) : b^2 - 2 * b * c * Real.cos A = a^2 - 2 * a * c * Real.cos B

-- Questions
theorem isosceles_triangle (a b c : ℝ) (A B C : ℝ)
  (h1 : c = 2) 
  (h2 : b^2 - 2 * b * c * Real.cos A = a^2 - 2 * a * c * Real.cos B) :
  a = b :=
sorry

theorem area_of_triangle (a b c : ℝ) (A B C : ℝ) 
  (h1 : c = 2) 
  (h2 : b^2 - 2 * b * c * Real.cos A = a^2 - 2 * a * c * Real.cos B)
  (h3 : 7 * Real.cos B = 2 * Real.cos C) 
  (h4 : a = b) :
  ∃ S : ℝ, S = Real.sqrt 15 :=
sorry

end isosceles_triangle_area_of_triangle_l91_91978


namespace percentage_return_on_investment_l91_91332

theorem percentage_return_on_investment (dividend_rate : ℝ) (face_value : ℝ) (purchase_price : ℝ) (return_percentage : ℝ) :
  dividend_rate = 0.125 → face_value = 40 → purchase_price = 20 → return_percentage = 25 :=
by
  intros h1 h2 h3
  sorry

end percentage_return_on_investment_l91_91332


namespace point_divides_segment_in_ratio_l91_91573

theorem point_divides_segment_in_ratio (A B C C1 A1 P : Type) 
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] 
  [AddCommGroup C1] [AddCommGroup A1] [AddCommGroup P]
  (h1 : AP / PA1 = 3 / 2)
  (h2 : CP / PC1 = 2 / 1) :
  AC1 / C1B = 2 / 3 :=
sorry

end point_divides_segment_in_ratio_l91_91573


namespace impossibility_exchange_l91_91981

theorem impossibility_exchange :
  ¬ ∃ (x y z : ℕ), (x + y + z = 10) ∧ (x + 3 * y + 5 * z = 25) := 
by
  sorry

end impossibility_exchange_l91_91981


namespace can_lids_per_box_l91_91645

/-- Aaron initially has 14 can lids, and after adding can lids from 3 boxes,
he has a total of 53 can lids. How many can lids are in each box? -/
theorem can_lids_per_box (initial : ℕ) (total : ℕ) (boxes : ℕ) (h₀ : initial = 14) (h₁ : total = 53) (h₂ : boxes = 3) :
  (total - initial) / boxes = 13 :=
by
  sorry

end can_lids_per_box_l91_91645


namespace sarees_original_price_l91_91743

theorem sarees_original_price (P : ℝ) (h : 0.90 * P * 0.95 = 342) : P = 400 :=
by
  sorry

end sarees_original_price_l91_91743


namespace min_treasures_buried_l91_91798

-- Definitions corresponding to conditions
def num_palm_trees : Nat := 30

def num_signs15 : Nat := 15
def num_signs8 : Nat := 8
def num_signs4 : Nat := 4
def num_signs3 : Nat := 3

def is_truthful (num_treasures num_signs : Nat) : Prop :=
  num_treasures ≠ num_signs

-- Theorem statement: The minimum number of signs under which the treasure can be buried
theorem min_treasures_buried (num_treasures : Nat) :
  (∀ (n : Nat), n = 15 ∨ n = 8 ∨ n = 4 ∨ n = 3 → is_truthful num_treasures n) →
  num_treasures = 15 :=
begin
  sorry
end

end min_treasures_buried_l91_91798


namespace pencil_price_units_l91_91261

noncomputable def price_pencil (base_price: ℕ) (extra_cost: ℕ): ℝ :=
  (base_price + extra_cost) / 10000.0

theorem pencil_price_units (base_price: ℕ) (extra_cost: ℕ) (h_base: base_price = 5000) (h_extra: extra_cost = 20) : 
  price_pencil base_price extra_cost = 0.5 := by
  sorry

end pencil_price_units_l91_91261


namespace arithmetic_sequence_a6_value_l91_91266

theorem arithmetic_sequence_a6_value
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 14) :
  a 6 = 2 :=
by
  sorry

end arithmetic_sequence_a6_value_l91_91266


namespace log32_eq_four_fifth_l91_91940

theorem log32_eq_four_fifth :
  log 32 4 = 2 / 5 := by
  sorry

end log32_eq_four_fifth_l91_91940


namespace smallest_right_triangle_area_l91_91122

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l91_91122


namespace jeff_current_cats_l91_91022

def initial_cats : ℕ := 20
def monday_found_kittens : ℕ := 2 + 3
def monday_stray_cats : ℕ := 4
def tuesday_injured_cats : ℕ := 1
def tuesday_health_issues_cats : ℕ := 2
def tuesday_family_cats : ℕ := 3
def wednesday_adopted_cats : ℕ := 4 * 2
def wednesday_pregnant_cats : ℕ := 2
def thursday_adopted_cats : ℕ := 3
def thursday_donated_cats : ℕ := 3
def friday_adopted_cats : ℕ := 2
def friday_found_cats : ℕ := 3

theorem jeff_current_cats : 
  initial_cats 
  + monday_found_kittens + monday_stray_cats 
  + (tuesday_injured_cats + tuesday_health_issues_cats + tuesday_family_cats)
  + (wednesday_pregnant_cats - wednesday_adopted_cats)
  + (thursday_donated_cats - thursday_adopted_cats)
  + (friday_found_cats - friday_adopted_cats) 
  = 30 := by
  sorry

end jeff_current_cats_l91_91022


namespace smallest_area_right_triangle_l91_91173

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l91_91173


namespace license_plate_combinations_l91_91012

-- Definitions of the conditions
def num_consonants : ℕ := 20
def num_vowels : ℕ := 6
def num_digits : ℕ := 10

-- The theorem statement
theorem license_plate_combinations : num_consonants * num_vowels * num_vowels * num_digits = 7200 := by
  sorry

end license_plate_combinations_l91_91012


namespace gcd_266_209_l91_91753

-- Definitions based on conditions
def a : ℕ := 266
def b : ℕ := 209

-- Theorem stating the GCD of a and b
theorem gcd_266_209 : Nat.gcd a b = 19 :=
by {
  -- Declare the specific integers as conditions
  let a := 266
  let b := 209
  -- Use the Euclidean algorithm (steps within the proof are not required)
  -- State that the conclusion is the GCD of a and b 
  sorry
}

end gcd_266_209_l91_91753


namespace real_number_c_l91_91889

theorem real_number_c (x1 x2 c : ℝ) (h_eqn : x1 + x2 = -1) (h_prod : x1 * x2 = c) (h_cond : x1^2 * x2 + x2^2 * x1 = 3) : c = -3 :=
by sorry

end real_number_c_l91_91889


namespace find_x2_plus_y2_l91_91690

theorem find_x2_plus_y2 (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 * y + x * y^2 + x + y = 99) : 
  x^2 + y^2 = 5745 / 169 := 
sorry

end find_x2_plus_y2_l91_91690


namespace bigger_part_of_dividing_56_l91_91329

theorem bigger_part_of_dividing_56 (x y : ℕ) (h₁ : x + y = 56) (h₂ : 10 * x + 22 * y = 780) : max x y = 38 :=
by
  sorry

end bigger_part_of_dividing_56_l91_91329


namespace max_non_similar_matrices_2020_l91_91802

open Matrix

noncomputable def max_non_similar_matrices (n : ℕ) : ℕ :=
  if n = 2020 then 673 else 0

variables {A : Matrix (Fin 2020) (Fin 2020) ℂ}

def is_adjugate (A A_adj : Matrix (Fin 2020) (Fin 2020) ℂ) : Prop := 
  A + A_adj = 1 ∧ A ⬝ A_adj = 1

theorem max_non_similar_matrices_2020 :
  ∀ (A : Matrix (Fin 2020) (Fin 2020) ℂ),
  (∃ A_adj, is_adjugate A A_adj) →
  max_non_similar_matrices 2020 = 673 :=
by
  sorry

end max_non_similar_matrices_2020_l91_91802


namespace map_distance_to_actual_distance_l91_91603

theorem map_distance_to_actual_distance
  (map_distance : ℝ)
  (scale_map_to_real : ℝ)
  (scale_real_distance : ℝ)
  (H_map_distance : map_distance = 18)
  (H_scale_map : scale_map_to_real = 0.5)
  (H_scale_real : scale_real_distance = 6) :
  (map_distance / scale_map_to_real) * scale_real_distance = 216 :=
by
  sorry

end map_distance_to_actual_distance_l91_91603


namespace Petya_victory_margin_l91_91567

theorem Petya_victory_margin :
  ∃ P1 P2 V1 V2 : ℕ, P1 = V1 + 9 ∧ V2 = P2 + 9 ∧ P1 + P2 + V1 + V2 = 27 ∧ 
  (P1 + P2) > (V1 + V2) ∧ (P1 + P2) - (V1 + V2) = 9 :=
begin
  sorry
end

end Petya_victory_margin_l91_91567


namespace jason_total_games_l91_91270

theorem jason_total_games :
  let jan_games := 11
  let feb_games := 17
  let mar_games := 16
  let apr_games := 20
  let may_games := 14
  let jun_games := 14
  let jul_games := 14
  jan_games + feb_games + mar_games + apr_games + may_games + jun_games + jul_games = 106 :=
by
  sorry

end jason_total_games_l91_91270


namespace arman_age_in_years_l91_91652

theorem arman_age_in_years (A S y : ℕ) (h1: A = 6 * S) (h2: S = 2 + 4) (h3: A + y = 40) : y = 4 :=
sorry

end arman_age_in_years_l91_91652


namespace smallest_area_of_right_triangle_l91_91102

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l91_91102


namespace range_equality_of_f_and_f_f_l91_91683

noncomputable def f (x a : ℝ) := x * Real.log x - x + 2 * a

theorem range_equality_of_f_and_f_f (a : ℝ) :
  (∀ x : ℝ, 0 < x → 1 < f x a) ∧ (∀ x : ℝ, 0 < x → f x a ≤ 1) →
  (∃ I : Set ℝ, (Set.range (λ x => f x a) = I) ∧ (Set.range (λ x => f (f x a) a) = I)) → 
  (1/2 < a ∧ a ≤ 1) :=
by 
  sorry

end range_equality_of_f_and_f_f_l91_91683


namespace value_of_expression_l91_91964

theorem value_of_expression (x : ℝ) (h : x^2 - 2*x - 2 = 0) : 3*x^2 - 6*x + 9 = 15 := 
by 
  have h₁ : x^2 - 2*x = 2 := by linarith
  calc
    3*x^2 - 6*x + 9 = 3*(x^2 - 2*x) + 9 : by ring
                ... = 3*2 + 9           : by rw [h₁]
                ... = 15                : by norm_num

end value_of_expression_l91_91964


namespace quadratic_root_value_l91_91535

theorem quadratic_root_value (a : ℝ) (h : a^2 + 2 * a - 3 = 0) : 2 * a^2 + 4 * a = 6 :=
by
  sorry

end quadratic_root_value_l91_91535


namespace cubic_expression_value_l91_91378

theorem cubic_expression_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2010 = 2011 :=
by sorry

end cubic_expression_value_l91_91378


namespace matthew_more_strawberries_than_betty_l91_91198

noncomputable def B : ℕ := 16

theorem matthew_more_strawberries_than_betty (M N : ℕ) 
  (h1 : M > B)
  (h2 : M = 2 * N) 
  (h3 : B + M + N = 70) : M - B = 20 :=
by
  sorry

end matthew_more_strawberries_than_betty_l91_91198


namespace find_k_l91_91298

theorem find_k 
  (x y: ℝ) 
  (h1: y = 5 * x + 3) 
  (h2: y = -2 * x - 25) 
  (h3: y = 3 * x + k) : 
  k = -5 :=
sorry

end find_k_l91_91298


namespace solution_l91_91362

theorem solution (t : ℝ) :
  let x := 3 * t
  let y := t
  let z := 0
  x^2 - 9 * y^2 = z^2 :=
by
  sorry

end solution_l91_91362


namespace parabola_properties_l91_91529

theorem parabola_properties (p : ℝ) (h1 : p > 0) :
  (∀ x y, y^2 = 2 * p * x ↔ y = 2 * x - 4) →
  (∀ A B : ℝ × ℝ, (sqrt ((fst B - fst A)^2 + (snd B - snd A)^2) = 3 * sqrt 5) →
  (p = 2) ∧
  let F := (1, 0) in
  let circumcircle_eq := (λ (x y : ℝ), (x - 13/2)^2 + (y + 1)^2 = 125/4) in
  ∀ P : ℝ × ℝ, (circumcircle_eq (fst P) (snd P)) →
  let center_to_AB_dist := 10 / sqrt 5 in
  let radius := 5 * sqrt 5 / 2 in
  (dist P (line AB) = center_to_AB_dist + radius = 9 * sqrt 5 / 2)) := sorry

end parabola_properties_l91_91529


namespace john_buys_packs_l91_91847

theorem john_buys_packs :
  let classes := 6
  let students_per_class := 30
  let packs_per_student := 2
  let total_students := classes * students_per_class
  let total_packs := total_students * packs_per_student
  total_packs = 360 :=
by
  let classes := 6
  let students_per_class := 30
  let packs_per_student := 2
  let total_students := classes * students_per_class
  let total_packs := total_students * packs_per_student
  show total_packs = 360
  sorry

end john_buys_packs_l91_91847


namespace find_Roe_speed_l91_91435

-- Definitions from the conditions
def Teena_speed : ℝ := 55
def time_in_hours : ℝ := 1.5
def initial_distance_difference : ℝ := 7.5
def final_distance_difference : ℝ := 15

-- Main theorem statement
theorem find_Roe_speed (R : ℝ) (h1 : R * time_in_hours + final_distance_difference = Teena_speed * time_in_hours - initial_distance_difference) :
  R = 40 :=
  sorry

end find_Roe_speed_l91_91435


namespace julia_average_speed_l91_91985

-- Define the conditions as constants
def total_distance : ℝ := 28
def total_time : ℝ := 4

-- Define the theorem stating Julia's average speed
theorem julia_average_speed : total_distance / total_time = 7 := by
  sorry

end julia_average_speed_l91_91985


namespace students_opted_both_math_science_l91_91415

def total_students : ℕ := 40
def not_opted_math : ℕ := 10
def not_opted_science : ℕ := 15
def not_opted_either : ℕ := 2

theorem students_opted_both_math_science :
  let T := total_students
  let M' := not_opted_math
  let S' := not_opted_science
  let E := not_opted_either
  let B := (T - M') + (T - S') - (T - E)
  B = 17 :=
by
  sorry

end students_opted_both_math_science_l91_91415


namespace find_x_for_equation_l91_91755

theorem find_x_for_equation : ∃ x : ℝ, (1 / 2) + ((2 / 3) * x + 4) - (8 / 16) = 4.25 ↔ x = 0.375 := 
by
  sorry

end find_x_for_equation_l91_91755


namespace find_h_at_2_l91_91537

noncomputable def h (x : ℝ) : ℝ := x^4 + 2 * x^3 - 12 * x^2 - 14 * x + 24

lemma poly_value_at_minus_2 : h (-2) = -4 := by
  sorry

lemma poly_value_at_1 : h 1 = -1 := by
  sorry

lemma poly_value_at_minus_4 : h (-4) = -16 := by
  sorry

lemma poly_value_at_3 : h 3 = -9 := by
  sorry

theorem find_h_at_2 : h 2 = -20 := by
  sorry

end find_h_at_2_l91_91537


namespace inverse_of_g_l91_91606

theorem inverse_of_g : 
  ∀ (g g_inv : ℝ → ℝ) (p q r s : ℝ),
  (∀ x, g x = (3 * x - 2) / (x + 4)) →
  (∀ x, g_inv x = (p * x + q) / (r * x + s)) →
  (∀ x, g (g_inv x) = x) →
  q / s = 2 / 3 :=
by
  intros g g_inv p q r s h_g h_g_inv h_g_ginv
  sorry

end inverse_of_g_l91_91606


namespace largest_possible_product_l91_91870

theorem largest_possible_product : 
  ∃ S1 S2 : Finset ℕ, 
  (S1 ∪ S2 = {1, 3, 4, 6, 7, 8, 9} ∧ S1 ∩ S2 = ∅ ∧ S1.prod id = S2.prod id) ∧ 
  (S1.prod id = 504 ∧ S2.prod id = 504) :=
by
  sorry

end largest_possible_product_l91_91870


namespace max_gold_coins_l91_91324

theorem max_gold_coins : ∃ n : ℕ, (∃ k : ℕ, n = 7 * k + 2) ∧ 50 < n ∧ n < 150 ∧ n = 149 :=
by
  sorry

end max_gold_coins_l91_91324


namespace cube_of_product_of_ab_l91_91387

theorem cube_of_product_of_ab (a b c : ℕ) (h1 : a * b * c = 180) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : (a * b) ^ 3 = 216 := 
sorry

end cube_of_product_of_ab_l91_91387


namespace Petya_victory_margin_l91_91568

theorem Petya_victory_margin :
  ∃ P1 P2 V1 V2 : ℕ, P1 = V1 + 9 ∧ V2 = P2 + 9 ∧ P1 + P2 + V1 + V2 = 27 ∧ 
  (P1 + P2) > (V1 + V2) ∧ (P1 + P2) - (V1 + V2) = 9 :=
begin
  sorry
end

end Petya_victory_margin_l91_91568


namespace triangle_area_l91_91825

theorem triangle_area (a b c : ℕ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10)
  (right_triangle : a^2 + b^2 = c^2) : (1 / 2 : ℝ) * (a * b) = 24 := by
  sorry

end triangle_area_l91_91825


namespace function_C_is_even_l91_91763

theorem function_C_is_even : ∀ x : ℝ, 2 * (-x)^2 - 1 = 2 * x^2 - 1 :=
by
  intro x
  sorry

end function_C_is_even_l91_91763


namespace sin_zero_range_valid_m_l91_91528

noncomputable def sin_zero_range (m : ℝ) : Prop :=
  ∀ f : ℝ → ℝ, 
    (∀ x : ℝ, f x = Real.sin (2 * x - Real.pi / 6) - m) →
    (∃ x1 x2 : ℝ, (0 ≤ x1 ∧ x1 ≤ Real.pi / 2) ∧ (0 ≤ x2 ∧ x2 ≤ Real.pi / 2) ∧ x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0)

theorem sin_zero_range_valid_m : 
  ∀ m : ℝ, sin_zero_range m ↔ (1 / 2 ≤ m ∧ m < 1) :=
sorry

end sin_zero_range_valid_m_l91_91528


namespace red_side_probability_l91_91331

theorem red_side_probability
  (num_cards : ℕ)
  (num_black_black : ℕ)
  (num_black_red : ℕ)
  (num_red_red : ℕ)
  (num_red_sides_total : ℕ)
  (num_red_sides_with_red_other_side : ℕ) :
  num_cards = 8 →
  num_black_black = 4 →
  num_black_red = 2 →
  num_red_red = 2 →
  num_red_sides_total = (num_red_red * 2 + num_black_red) →
  num_red_sides_with_red_other_side = (num_red_red * 2) →
  (num_red_sides_with_red_other_side / num_red_sides_total : ℝ) = 2 / 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end red_side_probability_l91_91331


namespace combinatorial_solution_l91_91518

theorem combinatorial_solution (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 14)
  (h3 : 0 ≤ 2 * x - 4) (h4 : 2 * x - 4 ≤ 14) : x = 4 ∨ x = 6 := by
  sorry

end combinatorial_solution_l91_91518


namespace thabo_total_books_l91_91884

noncomputable def total_books (H PNF PF : ℕ) : ℕ := H + PNF + PF

theorem thabo_total_books :
  ∀ (H PNF PF : ℕ),
    H = 30 →
    PNF = H + 20 →
    PF = 2 * PNF →
    total_books H PNF PF = 180 :=
by
  intros H PNF PF hH hPNF hPF
  sorry

end thabo_total_books_l91_91884


namespace domain_sqrt_quot_l91_91604

noncomputable def domain_of_function (f : ℝ → ℝ) : Set ℝ := {x : ℝ | f x ≠ 0}

theorem domain_sqrt_quot (x : ℝ) :
  (x + 1 ≥ 0) ∧ (x ≠ 0) ↔ (x ∈ {x : ℝ | -1 ≤ x ∧ x < 0} ∪ {x : ℝ | x > 0}) :=
by
  sorry

end domain_sqrt_quot_l91_91604


namespace sam_won_total_matches_l91_91698

/-- Sam's first 100 matches and he won 50% of them -/
def first_100_matches : ℕ := 100

/-- Sam won 50% of his first 100 matches -/
def win_rate_first : ℕ := 50

/-- Sam's next 100 matches and he won 60% of them -/
def next_100_matches : ℕ := 100

/-- Sam won 60% of his next 100 matches -/
def win_rate_next : ℕ := 60

/-- The total number of matches Sam won -/
def total_matches_won (first_100_matches: ℕ) (win_rate_first: ℕ) (next_100_matches: ℕ) (win_rate_next: ℕ) : ℕ :=
  (first_100_matches * win_rate_first) / 100 + (next_100_matches * win_rate_next) / 100

theorem sam_won_total_matches :
  total_matches_won first_100_matches win_rate_first next_100_matches win_rate_next = 110 :=
by
  sorry

end sam_won_total_matches_l91_91698


namespace smallest_area_right_triangle_l91_91090

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l91_91090


namespace Petya_victory_margin_l91_91569

theorem Petya_victory_margin :
  ∃ P1 P2 V1 V2 : ℕ, P1 = V1 + 9 ∧ V2 = P2 + 9 ∧ P1 + P2 + V1 + V2 = 27 ∧ 
  (P1 + P2) > (V1 + V2) ∧ (P1 + P2) - (V1 + V2) = 9 :=
begin
  sorry
end

end Petya_victory_margin_l91_91569


namespace smallest_area_right_triangle_l91_91129

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l91_91129


namespace length_of_smaller_cube_edge_is_5_l91_91639

-- Given conditions
def stacked_cube_composed_of_smaller_cubes (n: ℕ) (a: ℕ) : Prop := a * a * a = n

def volume_of_larger_cube (l: ℝ) (v: ℝ) : Prop := l ^ 3 = v

-- Problem statement: Prove that the length of one edge of the smaller cube is 5 cm
theorem length_of_smaller_cube_edge_is_5 :
  ∃ s: ℝ, stacked_cube_composed_of_smaller_cubes 8 2 ∧ volume_of_larger_cube (2*s) 1000 ∧ s = 5 :=
  sorry

end length_of_smaller_cube_edge_is_5_l91_91639


namespace all_terms_are_positive_integers_terms_product_square_l91_91207

def seq (x : ℕ → ℕ) : Prop :=
  x 1 = 1 ∧
  x 2 = 4 ∧
  ∀ n > 1, x n = Nat.sqrt (x (n - 1) * x (n + 1) + 1)

theorem all_terms_are_positive_integers (x : ℕ → ℕ) (h : seq x) : ∀ n, x n > 0 :=
sorry

theorem terms_product_square (x : ℕ → ℕ) (h : seq x) : ∀ n ≥ 1, ∃ k, 2 * x n * x (n + 1) + 1 = k ^ 2 :=
sorry

end all_terms_are_positive_integers_terms_product_square_l91_91207


namespace johns_total_packs_l91_91851

-- Defining the conditions
def classes : ℕ := 6
def students_per_class : ℕ := 30
def packs_per_student : ℕ := 2

-- Theorem statement
theorem johns_total_packs : 
  (classes * students_per_class * packs_per_student) = 360 :=
by
  -- The proof would go here
  sorry

end johns_total_packs_l91_91851


namespace tangent_line_eq_range_f_l91_91248

-- Given the function f(x) = 2x^3 - 9x^2 + 12x
def f(x : ℝ) : ℝ := 2 * x^3 - 9 * x^2 + 12 * x

-- (1) Prove that the equation of the tangent line to y = f(x) at (0, f(0)) is y = 12x
theorem tangent_line_eq : ∀ x, x = 0 → f x = 0 → (∃ m, m = 12 ∧ (∀ y, y = 12 * x)) :=
by
  sorry

-- (2) Prove that the range of f(x) on the interval [0, 3] is [0, 9]
theorem range_f : Set.Icc 0 9 = Set.image f (Set.Icc (0 : ℝ) 3) :=
by
  sorry

end tangent_line_eq_range_f_l91_91248


namespace remaining_budget_l91_91780

theorem remaining_budget
  (initial_budget : ℕ)
  (cost_flasks : ℕ)
  (cost_test_tubes : ℕ)
  (cost_safety_gear : ℕ)
  (h1 : initial_budget = 325)
  (h2 : cost_flasks = 150)
  (h3 : cost_test_tubes = (2 * cost_flasks) / 3)
  (h4 : cost_safety_gear = cost_test_tubes / 2) :
  initial_budget - (cost_flasks + cost_test_tubes + cost_safety_gear) = 25 := 
  by
  sorry

end remaining_budget_l91_91780


namespace min_value_x_squared_y_squared_z_squared_l91_91819

theorem min_value_x_squared_y_squared_z_squared
  (x y z : ℝ)
  (h : x + 2 * y + 3 * z = 6) :
  x^2 + y^2 + z^2 ≥ (18 / 7) :=
sorry

end min_value_x_squared_y_squared_z_squared_l91_91819


namespace third_order_central_moment_sum_l91_91716

open Probability

variables {Ω : Type*} {X X1 X2 : Ω → ℝ}
variable [MeasureSpace Ω]

def third_central_moment (X : Ω → ℝ) : ℝ :=
  moment_prop.central_moment X 3

theorem third_order_central_moment_sum (hX : X = λ ω, X1 ω + X2 ω) 
    (h_indep : independent X1 X2)
    (h_mu3_X1 : third_central_moment X1 = μ3_1)
    (h_mu3_X2 : third_central_moment X2 = μ3_2) :
  third_central_moment X = μ3_1 + μ3_2 :=
sorry

end third_order_central_moment_sum_l91_91716


namespace max_value_of_a1_l91_91003

theorem max_value_of_a1 (a1 a2 a3 a4 a5 a6 a7 : ℕ) (h_distinct : ∀ i j, i ≠ j → (i ≠ a1 → i ≠ a2 → i ≠ a3 → i ≠ a4 → i ≠ a5 → i ≠ a6 → i ≠ a7)) 
  (h_sum : a1 + a2 + a3 + a4 + a5 + a6 + a7 = 159) : a1 ≤ 19 :=
by
  sorry

end max_value_of_a1_l91_91003


namespace time_after_9876_seconds_l91_91269

-- Define the initial time in seconds
def initial_seconds : ℕ := 6 * 3600

-- Define the elapsed time in seconds
def elapsed_seconds : ℕ := 9876

-- Convert given time in seconds to hours, minutes, and seconds
def time_in_hms (total_seconds : ℕ) : (ℕ × ℕ × ℕ) :=
  let hours := total_seconds / 3600
  let minutes := (total_seconds % 3600) / 60
  let seconds := total_seconds % 60
  (hours, minutes, seconds)

-- Define the final time in 24-hour format (08:44:36)
def final_time : (ℕ × ℕ × ℕ) := (8, 44, 36)

-- The question's proof statement
theorem time_after_9876_seconds : 
  time_in_hms (initial_seconds + elapsed_seconds) = final_time :=
sorry

end time_after_9876_seconds_l91_91269


namespace bus_seats_needed_l91_91888

def members_playing_instruments : Prop :=
  let flute := 5
  let trumpet := 3 * flute
  let trombone := trumpet - 8
  let drum := trombone + 11
  let clarinet := 2 * flute
  let french_horn := trombone + 3
  let saxophone := (trumpet + trombone) / 2
  let piano := drum + 2
  let violin := french_horn - clarinet
  let guitar := 3 * flute
  let total_members := flute + trumpet + trombone + drum + clarinet + french_horn + saxophone + piano + violin + guitar
  total_members = 111

theorem bus_seats_needed : members_playing_instruments :=
by
  sorry

end bus_seats_needed_l91_91888


namespace Roselyn_initial_books_correct_l91_91874

variables (Roselyn_initial_books Mara_books Rebecca_books : ℕ)

-- Conditions
axiom A1 : Rebecca_books = 40
axiom A2 : Mara_books = 3 * Rebecca_books
axiom A3 : Roselyn_initial_books - (Rebecca_books + Mara_books) = 60

-- Proof statement
theorem Roselyn_initial_books_correct : Roselyn_initial_books = 220 :=
sorry

end Roselyn_initial_books_correct_l91_91874


namespace max_value_2ab_plus_2ac_sqrt3_l91_91719

variable (a b c : ℝ)
variable (h1 : a^2 + b^2 + c^2 = 1)
variable (h2 : 0 ≤ a)
variable (h3 : 0 ≤ b)
variable (h4 : 0 ≤ c)

theorem max_value_2ab_plus_2ac_sqrt3 : 2 * a * b + 2 * a * c * Real.sqrt 3 ≤ 1 := by
  sorry

end max_value_2ab_plus_2ac_sqrt3_l91_91719


namespace directrix_equation_l91_91505

def parabola_directrix (x : ℝ) : ℝ :=
  (x^2 - 8*x + 12) / 16

theorem directrix_equation :
  ∀ x, parabola_directrix x = y → y = -5/4 :=
sorry

end directrix_equation_l91_91505


namespace fish_offspring_base10_l91_91482

def convert_base_7_to_10 (n : ℕ) : ℕ :=
  let d2 := n / 49
  let r2 := n % 49
  let d1 := r2 / 7
  let d0 := r2 % 7
  d2 * 49 + d1 * 7 + d0

theorem fish_offspring_base10 :
  convert_base_7_to_10 265 = 145 :=
by
  sorry

end fish_offspring_base10_l91_91482


namespace problem_solution_l91_91818

variables {m n : ℝ}

theorem problem_solution (h1 : m^2 - n^2 = m * n) (h2 : m ≠ 0) (h3 : n ≠ 0) :
  (n / m) - (m / n) = -1 :=
sorry

end problem_solution_l91_91818


namespace john_spent_expected_amount_l91_91425

-- Define the original price of each pin
def original_price : ℝ := 20

-- Define the discount rate
def discount_rate : ℝ := 0.15

-- Define the number of pins
def number_of_pins : ℝ := 10

-- Define the sales tax rate
def tax_rate : ℝ := 0.08

-- Calculate the discount on each pin
def discount_per_pin : ℝ := discount_rate * original_price

-- Calculate the discounted price per pin
def discounted_price_per_pin : ℝ := original_price - discount_per_pin

-- Calculate the total discounted price for all pins
def total_discounted_price : ℝ := discounted_price_per_pin * number_of_pins

-- Calculate the sales tax on the total discounted price
def sales_tax : ℝ := tax_rate * total_discounted_price

-- Calculate the total amount spent including sales tax
def total_amount_spent : ℝ := total_discounted_price + sales_tax

-- The theorem that John spent $183.60 on pins including the sales tax
theorem john_spent_expected_amount : total_amount_spent = 183.60 :=
by
  sorry

end john_spent_expected_amount_l91_91425


namespace least_three_digit_eleven_heavy_l91_91341

def isElevenHeavy (n : ℕ) : Prop :=
  n % 11 > 6

theorem least_three_digit_eleven_heavy : ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ isElevenHeavy n ∧ ∀ m : ℕ, (m >= 100 ∧ m < 1000 ∧ isElevenHeavy m) → n ≤ m :=
sorry

end least_three_digit_eleven_heavy_l91_91341


namespace sasha_hometown_name_l91_91754

theorem sasha_hometown_name :
  ∃ (sasha_hometown : String), 
  (∃ (vadik_last_column : String), vadik_last_column = "ВКСАМО") →
  (∃ (sasha_transformed : String), sasha_transformed = "мТТЛАРАЕкис") →
  (∃ (sasha_starts_with : Char), sasha_starts_with = 'с') →
  sasha_hometown = "СТЕРЛИТАМАК" :=
by
  sorry

end sasha_hometown_name_l91_91754


namespace tile_difference_is_42_l91_91375

def original_blue_tiles : ℕ := 14
def original_green_tiles : ℕ := 8
def green_tiles_first_border : ℕ := 18
def green_tiles_second_border : ℕ := 30

theorem tile_difference_is_42 :
  (original_green_tiles + green_tiles_first_border + green_tiles_second_border) - original_blue_tiles = 42 :=
by
  sorry

end tile_difference_is_42_l91_91375


namespace part1_l91_91866

def f (x : ℝ) := x^2 - 2*x

theorem part1 (x : ℝ) :
  (|f x| + |x^2 + 2*x| ≥ 6*|x|) ↔ (x ≤ -3 ∨ 3 ≤ x ∨ x = 0) :=
sorry

end part1_l91_91866


namespace problem_statement_l91_91704

open BigOperators

-- Defining the arithmetic sequence
def a (n : ℕ) : ℕ := n - 1

-- Defining the sequence b_n
def b (n : ℕ) : ℕ :=
if n % 2 = 1 then
  a n + 1
else
  2 ^ a n

-- Defining T_2n as the sum of the first 2n terms of b
def T (n : ℕ) : ℕ :=
(∑ i in Finset.range n, b (2 * i + 1)) +
(∑ i in Finset.range n, b (2 * i + 2))

-- The theorem to be proven
theorem problem_statement (n : ℕ) : 
  a 2 * (a 4 + 1) = a 3 ^ 2 ∧
  T n = n^2 + (2^(2*n+1) - 2) / 3 :=
by
  sorry

end problem_statement_l91_91704


namespace smallest_area_correct_l91_91150

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l91_91150


namespace chord_length_of_larger_circle_tangent_to_smaller_circle_l91_91291

theorem chord_length_of_larger_circle_tangent_to_smaller_circle :
  ∀ (A B C : ℝ), B = 5 → π * (A ^ 2 - B ^ 2) = 50 * π → (C / 2) ^ 2 + B ^ 2 = A ^ 2 → C = 10 * Real.sqrt 2 :=
by
  intros A B C hB hArea hChord
  sorry

end chord_length_of_larger_circle_tangent_to_smaller_circle_l91_91291


namespace xy_fraction_equivalence_l91_91880

theorem xy_fraction_equivalence
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (x^2 + 4 * x * y) / (y^2 - 4 * x * y) = 3) :
  (x^2 - 4 * x * y) / (y^2 + 4 * x * y) = -1 :=
sorry

end xy_fraction_equivalence_l91_91880


namespace james_fish_weight_l91_91579

theorem james_fish_weight :
  let trout := 200
  let salmon := trout + (trout * 0.5)
  let tuna := 2 * salmon
  trout + salmon + tuna = 1100 := 
by
  sorry

end james_fish_weight_l91_91579


namespace proof_ellipse_equation_existence_lambda_l91_91515

open Real

noncomputable def ellipse := 
  {a b : ℝ // a = 2 ∧ b = sqrt 3 ∧ (∀ x y, (x/a)^2 + (y/b)^2 = 1) ∧ a > 0 ∧ b > 0}

theorem proof_ellipse_equation (a b : ℝ) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ) (R : ℝ × ℝ) (M N G : ℝ × ℝ) :
  P = (2 * sqrt 6 / 3, -1) →
  (F1, F2).1 = (0, a) →
  (F1, F2).2 = (0, -a) →
  abs (dist P F1 + dist P F2) = 4 →
  R = (4, 0) →
  M ∈ ({ x y | (x/a)^2 + (y/b)^2 = 1 }) →
  N ∈ ({ x y | (x/a)^2 + (y/b)^2 = 1 }) →
  G ∈ ({ x (-y) | (x/a)^2 + (y/b)^2 = 1 }) →
  by let equation_C := ellipse; 
     sorry

theorem existence_lambda (M N G F2 : ℝ × ℝ) :
  ∃ λ : ℝ, dist G F2 = λ * dist F2 N :=
sorry

end proof_ellipse_equation_existence_lambda_l91_91515


namespace defective_percentage_is_correct_l91_91924

noncomputable def percentage_defective (defective : ℕ) (total : ℝ) : ℝ := 
  (defective / total) * 100

theorem defective_percentage_is_correct : 
  percentage_defective 2 3333.3333333333335 = 0.06000600060006 :=
by
  sorry

end defective_percentage_is_correct_l91_91924


namespace cosine_neg_alpha_l91_91234

theorem cosine_neg_alpha (alpha : ℝ) (h : Real.sin (π/2 + alpha) = -3/5) : Real.cos (-alpha) = -3/5 :=
sorry

end cosine_neg_alpha_l91_91234


namespace friends_courses_l91_91974

-- Define the notions of students and their properties
structure Student :=
  (first_name : String)
  (last_name : String)
  (year : ℕ)

-- Define the specific conditions from the problem
def students : List Student := [
  ⟨"Peter", "Krylov", 1⟩,
  ⟨"Nikolay", "Ivanov", 2⟩,
  ⟨"Boris", "Karpov", 3⟩,
  ⟨"Vasily", "Orlov", 4⟩
]

-- The main statement of the problem
theorem friends_courses :
  ∀ (s : Student), s ∈ students →
    (s.first_name = "Peter" → s.last_name = "Krylov" ∧ s.year = 1) ∧
    (s.first_name = "Nikolay" → s.last_name = "Ivanov" ∧ s.year = 2) ∧
    (s.first_name = "Boris" → s.last_name = "Karpov" ∧ s.year = 3) ∧
    (s.first_name = "Vasily" → s.last_name = "Orlov" ∧ s.year = 4) :=
by
  sorry

end friends_courses_l91_91974


namespace determinant_of_B_l91_91031

noncomputable def B (b c : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := 
  ![![b, 2], 
    ![-3, c]]

theorem determinant_of_B (b c : ℝ) (h : B b c + 2 • (B b c)⁻¹ = 0) : 
  Matrix.det (B b c) = 4 := 
sorry

end determinant_of_B_l91_91031


namespace smallest_area_of_right_triangle_l91_91160

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l91_91160


namespace max_cookie_price_l91_91715

theorem max_cookie_price (k p : ℕ) :
  8 * k + 3 * p < 200 →
  4 * k + 5 * p > 150 →
  k ≤ 19 :=
sorry

end max_cookie_price_l91_91715


namespace machines_remain_closed_l91_91340

open Real

/-- A techno company has 14 machines of equal efficiency in its factory.
The annual manufacturing costs are Rs 42000 and establishment charges are Rs 12000.
The annual output of the company is Rs 70000. The annual output and manufacturing
costs are directly proportional to the number of machines. The shareholders get
12.5% profit, which is directly proportional to the annual output of the company.
If some machines remain closed throughout the year, then the percentage decrease
in the amount of profit of the shareholders is 12.5%. Prove that 2 machines remain
closed throughout the year. -/
theorem machines_remain_closed (machines total_cost est_charges output : ℝ)
    (shareholders_profit : ℝ)
    (machines_closed percentage_decrease : ℝ) :
  machines = 14 →
  total_cost = 42000 →
  est_charges = 12000 →
  output = 70000 →
  shareholders_profit = 0.125 →
  percentage_decrease = 0.125 →
  machines_closed = 2 :=
by
  sorry

end machines_remain_closed_l91_91340


namespace last_number_in_first_set_l91_91444

variables (x y : ℕ)

def mean (a b c d e : ℕ) : ℕ :=
  (a + b + c + d + e) / 5

theorem last_number_in_first_set :
  (mean 28 x 42 78 y = 90) ∧ (mean 128 255 511 1023 x = 423) → y = 104 :=
by 
  sorry

end last_number_in_first_set_l91_91444


namespace double_series_evaluation_l91_91215

theorem double_series_evaluation :
    (∑' m : ℕ, ∑' n : ℕ, (1 : ℝ) / (m * n * (m + n + 2))) = (3 / 2 : ℝ) :=
sorry

end double_series_evaluation_l91_91215


namespace find_cans_lids_l91_91647

-- Define the given conditions
def total_lids (x : ℕ) : ℕ := 14 + 3 * x

-- Define the proof problem
theorem find_cans_lids (x : ℕ) (h : total_lids x = 53) : x = 13 :=
sorry

end find_cans_lids_l91_91647


namespace smallest_area_right_triangle_l91_91157

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l91_91157


namespace reflect_point_x_axis_correct_l91_91020

-- Definition of the transformation reflecting a point across the x-axis
def reflect_x_axis (P : ℝ × ℝ) : ℝ × ℝ := (P.1, -P.2)

-- Define the original point coordinates
def P : ℝ × ℝ := (-2, 3)

-- The Lean proof statement
theorem reflect_point_x_axis_correct :
  reflect_x_axis P = (-2, -3) :=
sorry

end reflect_point_x_axis_correct_l91_91020


namespace carousel_revolutions_l91_91474

/-- Prove that the number of revolutions a horse 4 feet from the center needs to travel the same distance
as a horse 16 feet from the center making 40 revolutions is 160 revolutions. -/
theorem carousel_revolutions (r₁ : ℕ := 16) (revolutions₁ : ℕ := 40) (r₂ : ℕ := 4) :
  (revolutions₁ * (r₁ / r₂) = 160) :=
sorry

end carousel_revolutions_l91_91474


namespace scientific_notation_example_l91_91484

theorem scientific_notation_example :
  ∃ (a : ℝ) (b : ℤ), 1300000 = a * 10 ^ b ∧ a = 1.3 ∧ b = 6 :=
sorry

end scientific_notation_example_l91_91484


namespace petya_max_votes_difference_l91_91564

theorem petya_max_votes_difference :
  ∃ (P1 P2 V1 V2 : ℕ), 
    P1 = V1 + 9 ∧ 
    V2 = P2 + 9 ∧ 
    P1 + P2 + V1 + V2 = 27 ∧ 
    P1 + P2 > V1 + V2 ∧ 
    (P1 + P2) - (V1 + V2) = 9 := 
by
  sorry

end petya_max_votes_difference_l91_91564


namespace smallest_b_for_34b_perfect_square_is_4_l91_91904

theorem smallest_b_for_34b_perfect_square_is_4 :
  ∃ n : ℕ, ∀ b : ℤ, b > 3 → (3 * b + 4 = n * n → b = 4) :=
by
  existsi 4
  intros b hb
  intro h
  sorry

end smallest_b_for_34b_perfect_square_is_4_l91_91904


namespace tan_alpha_minus_beta_value_l91_91381

theorem tan_alpha_minus_beta_value (α β : Real) 
  (h1 : Real.sin α = 3 / 5) 
  (h2 : α ∈ Set.Ioo (π / 2) π) 
  (h3 : Real.tan (π - β) = 1 / 2) : 
  Real.tan (α - β) = -2 / 11 :=
by
  sorry

end tan_alpha_minus_beta_value_l91_91381


namespace calculation_identity_l91_91658

theorem calculation_identity :
  (3.14 - 1)^0 * (-1 / 4)^(-2) = 16 := by
  sorry

end calculation_identity_l91_91658


namespace skating_average_l91_91650

variable (minutesPerDay1 minutesPerDay2 : Nat)
variable (days1 days2 totalDays requiredAverage : Nat)

theorem skating_average :
  minutesPerDay1 = 80 →
  days1 = 6 →
  minutesPerDay2 = 100 →
  days2 = 2 →
  totalDays = 9 →
  requiredAverage = 95 →
  (minutesPerDay1 * days1 + minutesPerDay2 * days2 + x) / totalDays = requiredAverage →
  x = 175 :=
by
  intro h1 h2 h3 h4 h5 h6 h7
  sorry

end skating_average_l91_91650


namespace initial_alarm_time_was_l91_91622

def faster_watch_gain (rate : ℝ) (hours : ℝ) : ℝ := hours * rate

def absolute_time_difference (faster_time : ℝ) (correct_time : ℝ) : ℝ := faster_time - correct_time

theorem initial_alarm_time_was :
  ∀ (rate minutes time_difference : ℝ),
  rate = 2 →
  minutes = 12 →
  time_difference = minutes / rate →
  abs (4 - (4 - time_difference)) = 6 →
  (24 - 6) = 22 :=
by
  intros rate minutes time_difference hrate hminutes htime_diff htime
  sorry

end initial_alarm_time_was_l91_91622


namespace f_is_neither_odd_nor_even_l91_91539

-- Defining the function f(x)
def f (x : ℝ) : ℝ := x^2 + 6 * x

-- Defining the concept of an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

-- Defining the concept of an even function
def is_even (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = f x

-- The goal is to prove that f is neither odd nor even
theorem f_is_neither_odd_nor_even : ¬ is_odd f ∧ ¬ is_even f :=
by
  sorry

end f_is_neither_odd_nor_even_l91_91539


namespace equation1_solution_equation2_solution_l91_91287

theorem equation1_solution (x : ℝ) (h : 3 * x - 1 = x + 7) : x = 4 := by
  sorry

theorem equation2_solution (x : ℝ) (h : (x + 1) / 2 - 1 = (1 - 2 * x) / 3) : x = 5 / 7 := by
  sorry

end equation1_solution_equation2_solution_l91_91287


namespace sin_double_angle_identity_l91_91510

theorem sin_double_angle_identity (x : ℝ) (h : Real.sin (x + π/4) = -3/5) : Real.sin (2 * x) = -7/25 := 
by 
  sorry

end sin_double_angle_identity_l91_91510


namespace find_constant_l91_91735

noncomputable def expr (x C : ℝ) : ℝ :=
  (x - 1) * (x - 3) * (x - 4) * (x - 6) + C

theorem find_constant :
  (∀ x : ℝ, expr x (-0.5625) ≥ 1) → expr 3.5 (-0.5625) = 1 :=
by
  sorry

end find_constant_l91_91735


namespace expected_value_two_flips_l91_91469

open Probability

-- Define the biased coin probabilities
def biased_coin : Pmf bool :=
Pmf.ofProbFn (λ b, if b then 2/5 else 3/5)

-- Define the winnings for heads and tails
def winnings (b : bool) : ℚ :=
if b then 4 else -1

-- Expected value of winnings for a single flip
def E₁ : ℚ :=
Pmf.esum (biased_coin.bind (λ b, Pmf.return (winnings b)))

-- Expected value of winnings after two independent flips
def E₂ := 2 * E₁

-- Statement of the proof problem
theorem expected_value_two_flips : E₂ = 2 :=
by 
  rw [←mul_assoc]
  have hE₁ : E₁ = 1 := sorry
  rw [hE₁]
  norm_num

end expected_value_two_flips_l91_91469


namespace smallest_b_for_34b_perfect_square_is_4_l91_91903

theorem smallest_b_for_34b_perfect_square_is_4 :
  ∃ n : ℕ, ∀ b : ℤ, b > 3 → (3 * b + 4 = n * n → b = 4) :=
by
  existsi 4
  intros b hb
  intro h
  sorry

end smallest_b_for_34b_perfect_square_is_4_l91_91903


namespace min_stamps_value_l91_91033

theorem min_stamps_value (x y : ℕ) (hx : 5 * x + 7 * y = 74) : x + y = 12 :=
by
  sorry

end min_stamps_value_l91_91033


namespace petya_max_margin_l91_91553

def max_margin_votes (total_votes P1 P2 V1 V2: ℕ) : ℕ := P1 + P2 - (V1 + V2)

theorem petya_max_margin 
  (P1 P2 V1 V2: ℕ)
  (H1: P1 = V1 + 9) 
  (H2: V2 = P2 + 9) 
  (H3: P1 + P2 + V1 + V2 = 27) 
  (H_win: P1 + P2 > V1 + V2) : 
  max_margin_votes 27 P1 P2 V1 V2 = 9 :=
by
  sorry

end petya_max_margin_l91_91553


namespace customer_C_weight_l91_91642

def weights : List ℕ := [22, 25, 28, 31, 34, 36, 38, 40, 45]

-- Definitions for customer A and B such that customer A's total weight equals twice of customer B's total weight
variable {A B : List ℕ}

-- Condition on weights distribution
def valid_distribution (A B : List ℕ) : Prop :=
  (A.sum = 2 * B.sum) ∧ (A ++ B).sum + 38 = 299

-- Prove the weight of the bag received by customer C
theorem customer_C_weight :
  ∃ (C : ℕ), C ∈ weights ∧ C = 38 := by
  sorry

end customer_C_weight_l91_91642


namespace Faye_age_correct_l91_91663

def ages (C D E F G : ℕ) : Prop :=
  D = E - 2 ∧
  C = E + 3 ∧
  F = C - 1 ∧
  D = 16 ∧
  G = D - 5

theorem Faye_age_correct (C D E F G : ℕ) (h : ages C D E F G) : F = 20 :=
by {
  sorry
}

end Faye_age_correct_l91_91663


namespace avg_height_first_30_girls_l91_91293

theorem avg_height_first_30_girls (H : ℝ)
  (h1 : ∀ x : ℝ, 30 * x + 10 * 156 = 40 * 159) :
  H = 160 :=
by sorry

end avg_height_first_30_girls_l91_91293


namespace smallest_area_right_triangle_l91_91159

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l91_91159


namespace swimmer_speed_in_still_water_l91_91921

-- Define the conditions
def current_speed : ℝ := 2   -- Speed of the water current is 2 km/h
def swim_time : ℝ := 2.5     -- Time taken to swim against current is 2.5 hours
def distance : ℝ := 5        -- Distance swum against current is 5 km

-- Main theorem proving the swimmer's speed in still water
theorem swimmer_speed_in_still_water (v : ℝ) (h : v - current_speed = distance / swim_time) : v = 4 :=
by {
  -- Skipping the proof steps as per the requirements
  sorry
}

end swimmer_speed_in_still_water_l91_91921


namespace find_f_inv_128_l91_91963

noncomputable def f : ℕ → ℕ := sorry

axiom f_at_5 : f 5 = 2
axiom f_doubling : ∀ x : ℕ, f (2 * x) = 2 * f x

theorem find_f_inv_128 : f 320 = 128 :=
by sorry

end find_f_inv_128_l91_91963


namespace number_of_perfect_square_multiples_le_2000_l91_91508

theorem number_of_perfect_square_multiples_le_2000 :
  {n : ℕ | n ≤ 2000 ∧ ∃ k : ℕ, 10 * n = k^2}.finite.card = 14 := by
sorry

end number_of_perfect_square_multiples_le_2000_l91_91508


namespace vector_coordinates_l91_91418

theorem vector_coordinates (A B : ℝ × ℝ) (hA : A = (0, 1)) (hB : B = (-1, 2)) :
  B - A = (-1, 1) :=
sorry

end vector_coordinates_l91_91418


namespace solve_equation1_solve_equation2_l91_91434

theorem solve_equation1 (x : ℝ) (h1 : 2 * x - 9 = 4 * x) : x = -9 / 2 :=
by
  sorry

theorem solve_equation2 (x : ℝ) (h2 : 5 / 2 * x - 7 / 3 * x = 4 / 3 * 5 - 5) : x = 10 :=
by
  sorry

end solve_equation1_solve_equation2_l91_91434


namespace find_constants_u_v_l91_91503

theorem find_constants_u_v : 
  ∃ u v : ℝ, (∀ x : ℝ, 9 * x^2 - 36 * x - 81 = 0 ↔ (x + u)^2 = v) ∧ u + v = 7 :=
sorry

end find_constants_u_v_l91_91503


namespace subtract_value_is_34_l91_91968

theorem subtract_value_is_34 
    (x y : ℤ) 
    (h1 : (x - 5) / 7 = 7) 
    (h2 : (x - y) / 10 = 2) : 
    y = 34 := 
sorry

end subtract_value_is_34_l91_91968


namespace right_triangle_min_area_l91_91170

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l91_91170


namespace right_triangle_min_area_l91_91169

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l91_91169


namespace minimum_prime_product_l91_91455

noncomputable def is_prime : ℕ → Prop := sorry -- Assume the definition of prime

theorem minimum_prime_product (m n p : ℕ) 
  (hm : is_prime m) 
  (hn : is_prime n) 
  (hp : is_prime p) 
  (h_distinct : m ≠ n ∧ n ≠ p ∧ m ≠ p)
  (h_sum : m + n = p) : 
  m * n * p = 30 :=
sorry

end minimum_prime_product_l91_91455


namespace pn_value_2_pn_value_3_pn_minus_qn_geometric_pn_value_l91_91841

noncomputable def p (n : ℕ) : ℝ := sorry
noncomputable def q (n : ℕ) : ℝ := sorry

axiom pn_qn_recursion (n : ℕ) (hn : n ≥ 2) :
  p n = p (n - 1) * (1/6) + q (n - 1) * (5/6)
  ∧ q n = q (n - 1) * (1/6) + p (n - 1) * (5/6)

axiom initial_conditions : p 1 = 1 ∧ q 1 = 0

theorem pn_value_2 :
  p 2 = 1/6 := sorry

theorem pn_value_3 :
  p 3 = 26/36 := sorry

theorem pn_minus_qn_geometric (n : ℕ) (hn : n ≥ 2) :
  (p n - q n) = (-2/3)^(n-1) := sorry

theorem pn_value (n : ℕ) (hn : n ≥ 1) :
  p n = (1/2) * ((-2/3)^(n-1) + 1) := sorry

end pn_value_2_pn_value_3_pn_minus_qn_geometric_pn_value_l91_91841


namespace betty_initial_marbles_l91_91648

theorem betty_initial_marbles (B : ℝ) (h1 : 0.40 * B = 24) : B = 60 :=
by
  sorry

end betty_initial_marbles_l91_91648


namespace roots_real_and_equal_l91_91506

theorem roots_real_and_equal (a b c : ℝ) (h_eq : a = 1) (h_b : b = -4 * Real.sqrt 2) (h_c : c = 8) :
  ∃ x : ℝ, (a * x^2 + b * x + c = 0) ∧ (b^2 - 4 * a * c = 0) :=
by
  have h_a : a = 1 := h_eq;
  have h_b : b = -4 * Real.sqrt 2 := h_b;
  have h_c : c = 8 := h_c;
  sorry

end roots_real_and_equal_l91_91506


namespace willie_final_stickers_l91_91626

-- Definitions of initial stickers and given stickers
def willie_initial_stickers : ℝ := 36.0
def emily_gives : ℝ := 7.0

-- The statement to prove
theorem willie_final_stickers : willie_initial_stickers + emily_gives = 43.0 := by
  sorry

end willie_final_stickers_l91_91626


namespace marla_night_cost_is_correct_l91_91419

def lizard_value_bc := 8 -- 1 lizard is worth 8 bottle caps
def lizard_value_gw := 5 / 3 -- 3 lizards are worth 5 gallons of water
def horse_value_gw := 80 -- 1 horse is worth 80 gallons of water
def marla_daily_bc := 20 -- Marla can scavenge 20 bottle caps each day
def marla_days := 24 -- It takes Marla 24 days to collect the bottle caps

noncomputable def marla_night_cost_bc : ℕ :=
((marla_daily_bc * marla_days) - (horse_value_gw / lizard_value_gw * (3 * lizard_value_bc))) / marla_days

theorem marla_night_cost_is_correct :
  marla_night_cost_bc = 4 := by
  sorry

end marla_night_cost_is_correct_l91_91419


namespace translation_correct_l91_91300

def parabola1 (x : ℝ) : ℝ := -2 * (x + 2)^2 + 3
def parabola2 (x : ℝ) : ℝ := -2 * (x - 1)^2 - 1

theorem translation_correct :
  ∀ x : ℝ, parabola2 (x - 3) = parabola1 x - 4 :=
by
  sorry

end translation_correct_l91_91300


namespace pentagon_area_l91_91727

/-- This Lean statement represents the problem of finding the y-coordinate of vertex C
    in a pentagon with given vertex positions and specific area constraint. -/
theorem pentagon_area (y : ℝ) 
  (h_sym : true) -- The pentagon ABCDE has a vertical line of symmetry
  (h_A : (0, 0) = (0, 0)) -- A(0,0)
  (h_B : (0, 5) = (0, 5)) -- B(0, 5)
  (h_C : (3, y) = (3, y)) -- C(3, y)
  (h_D : (6, 5) = (6, 5)) -- D(6, 5)
  (h_E : (6, 0) = (6, 0)) -- E(6, 0)
  (h_area : 50 = 50) -- The total area of the pentagon is 50 square units
  : y = 35 / 3 :=
sorry

end pentagon_area_l91_91727


namespace irrational_b_eq_neg_one_l91_91274

theorem irrational_b_eq_neg_one
  (a : ℝ) (b : ℝ)
  (h_irrational : ¬ ∃ q : ℚ, a = (q : ℝ))
  (h_eq : ab + a - b = 1) :
  b = -1 :=
sorry

end irrational_b_eq_neg_one_l91_91274


namespace sqrt_5sq_4six_eq_320_l91_91204

theorem sqrt_5sq_4six_eq_320 : Real.sqrt (5^2 * 4^6) = 320 :=
by sorry

end sqrt_5sq_4six_eq_320_l91_91204


namespace question1_question2_l91_91777

open Finset 

noncomputable def choose_internal_specific_surj_exclude (total_internal : ℕ) (total_surgeon : ℕ) 
  (doctors_to_choose : ℕ) (specific_internal_must : ℕ) (specific_surgeon_exclude : ℕ) : ℕ :=
choose (total_internal + total_surgeon - (specific_internal_must + specific_surgeon_exclude)) 
       (doctors_to_choose - specific_internal_must)

theorem question1:
  let total_internal := 12
  let total_surgeon := 8
  let doctors_to_choose := 5
  let specific_internal_must := 1
  let specific_surgeon_exclude := 1 in
  choose_internal_specific_surj_exclude total_internal total_surgeon doctors_to_choose 
    specific_internal_must specific_surgeon_exclude = 3060 := by sorry

noncomputable def choose_both_included (total_internal : ℕ) (total_surgeon : ℕ) 
  (doctors_to_choose : ℕ) : ℕ :=
choose (total_internal + total_surgeon) doctors_to_choose - 
choose total_internal doctors_to_choose - 
choose total_surgeon doctors_to_choose

theorem question2:
  let total_internal := 12
  let total_surgeon := 8
  let doctors_to_choose := 5 in
  choose_both_included total_internal total_surgeon doctors_to_choose = 14656 := by sorry

end question1_question2_l91_91777


namespace digits_divisibility_property_l91_91275

-- Definition: Example function to sum the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (· + ·) 0

-- Theorem: Prove the correctness of the given mathematical problem
theorem digits_divisibility_property:
  ∀ n : ℕ, (n = 18 ∨ n = 27 ∨ n = 45 ∨ n = 63) →
  (sum_of_digits n % 9 = 0) → (n % 9 = 0) := by
  sorry

end digits_divisibility_property_l91_91275


namespace circular_patch_radius_l91_91773

theorem circular_patch_radius : 
  let r_cylinder := 3  -- radius of the container in cm
  let h_cylinder := 6  -- height of the container in cm
  let t_patch := 0.2   -- thickness of each patch in cm
  let V := π * r_cylinder^2 * h_cylinder -- Volume of the liquid

  let V_patch := V / 2                  -- Volume of each patch
  let r := 3 * Real.sqrt 15              -- the radius we want to prove

  r^2 * π * t_patch = V_patch           -- the volume equation for one patch
  →

  r = 3 * Real.sqrt 15 := 
by
  sorry

end circular_patch_radius_l91_91773


namespace prime_condition_composite_condition_l91_91594

theorem prime_condition (n : ℕ) (a : Fin n → ℕ) (h_distinct : Function.Injective a)
  (h_prime : Prime (2 * n - 1)) :
  ∃ i j : Fin n, i ≠ j ∧ ((a i + a j) / Nat.gcd (a i) (a j) ≥ 2 * n - 1) := 
sorry

theorem composite_condition (n : ℕ) (h_composite : ¬ Prime (2 * n - 1)) :
  ∃ a : Fin n → ℕ, Function.Injective a ∧ (∀ i j : Fin n, i ≠ j → ((a i + a j) / Nat.gcd (a i) (a j) < 2 * n - 1)) := 
sorry

end prime_condition_composite_condition_l91_91594


namespace min_value_a_plus_b_plus_c_l91_91408

-- Define the main conditions
variables {a b c : ℝ}
variables (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
variables (h_eq : a^2 + 2*a*b + 4*b*c + 2*c*a = 16)

-- Define the theorem
theorem min_value_a_plus_b_plus_c : 
  (∀ a b c : ℝ, (a > 0 ∧ b > 0 ∧ c > 0) ∧ (a^2 + 2*a*b + 4*b*c + 2*c*a = 16) → a + b + c ≥ 4) :=
sorry

end min_value_a_plus_b_plus_c_l91_91408


namespace apple_street_length_l91_91651

theorem apple_street_length :
  ∀ (n : ℕ) (d : ℕ), 
    (n = 15) → (d = 200) → 
    (∃ l : ℝ, (l = ((n + 1) * d) / 1000) ∧ l = 3.2) :=
by
  intros
  sorry

end apple_street_length_l91_91651


namespace problem_N_lowest_terms_l91_91814

theorem problem_N_lowest_terms :
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 2500 ∧ ∃ k : ℕ, k ∣ 128 ∧ (n + 11) % k = 0 ∧ (Nat.gcd (n^2 + 7) (n + 11)) > 1) →
  ∃ cnt : ℕ, cnt = 168 :=
by
  sorry

end problem_N_lowest_terms_l91_91814


namespace population_increase_l91_91739

theorem population_increase (birth_rate : ℝ) (death_rate : ℝ) (initial_population : ℝ) :
  initial_population = 1000 →
  birth_rate = 32 / 1000 →
  death_rate = 11 / 1000 →
  ((birth_rate - death_rate) / initial_population) * 100 = 2.1 :=
by
  sorry

end population_increase_l91_91739


namespace smallest_area_right_triangle_l91_91093

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l91_91093


namespace petya_wins_max_margin_l91_91547

theorem petya_wins_max_margin {P1 P2 V1 V2 : ℕ} 
  (h1 : P1 = V1 + 9)
  (h2 : V2 = P2 + 9)
  (h3 : P1 + P2 + V1 + V2 = 27)
  (h4 : P1 + P2 > V1 + V2) :
  ∃ m : ℕ, m = 9 ∧ P1 + P2 - (V1 + V2) = m :=
by
  sorry

end petya_wins_max_margin_l91_91547


namespace smallest_area_right_triangle_l91_91124

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l91_91124


namespace calories_burned_per_week_l91_91420

-- Definitions from conditions
def classes_per_week : ℕ := 3
def hours_per_class : ℚ := 1.5
def calories_per_minute : ℕ := 7

-- Prove the total calories burned per week
theorem calories_burned_per_week : 
  (classes_per_week * (hours_per_class * 60) * calories_per_minute) = 1890 := by
    sorry

end calories_burned_per_week_l91_91420


namespace hyperbola_y_relation_l91_91952

theorem hyperbola_y_relation {k y₁ y₂ : ℝ} 
  (A_on_hyperbola : y₁ = k / 2) 
  (B_on_hyperbola : y₂ = k / 3) 
  (k_positive : 0 < k) : 
  y₁ > y₂ := 
sorry

end hyperbola_y_relation_l91_91952


namespace circle_parabola_intersect_l91_91656

theorem circle_parabola_intersect (a : ℝ) :
  (∀ (x y : ℝ), x^2 + (y - 1)^2 = 1 ∧ y = a * x^2 → (x ≠ 0 ∨ y ≠ 0)) ↔ a > 1 / 2 :=
by
  sorry

end circle_parabola_intersect_l91_91656


namespace complex_problem_proof_l91_91682

open Complex

noncomputable def z : ℂ := (1 - I)^2 + 1 + 3 * I

theorem complex_problem_proof : z = 1 + I ∧ abs (z - 2 * I) = Real.sqrt 2 ∧ (∀ a b : ℝ, (z^2 + a + b = 1 - I) → (a = -3 ∧ b = 4)) := 
by
  have h1 : z = (1 - I)^2 + 1 + 3 * I := rfl
  have h2 : z = 1 + I := sorry
  have h3 : abs (z - 2 * I) = Real.sqrt 2 := sorry
  have h4 : (∀ a b : ℝ, (z^2 + a + b = 1 - I) → (a = -3 ∧ b = 4)) := sorry
  exact ⟨h2, h3, h4⟩

end complex_problem_proof_l91_91682


namespace find_ordered_pairs_l91_91358

theorem find_ordered_pairs :
  {p : ℕ × ℕ // 0 < p.1 ∧ 0 < p.2 ∧ (n = p.2 ∧ m = p.1 ∧ (n^3 + 1) % (m*n - 1) = 0)}
  = {(2, 1), (3, 1), (2, 2), (5, 2), (5, 3), (2, 5), (3, 5)} :=
by sorry

end find_ordered_pairs_l91_91358


namespace math_problem_l91_91685

noncomputable def f (x : ℝ) := |Real.exp x - 1|

theorem math_problem (x1 x2 : ℝ) (h1 : x1 < 0) (h2 : x2 > 0)
  (h3 : - Real.exp x1 * Real.exp x2 = -1) :
  (x1 + x2 = 0) ∧
  (0 < (Real.exp x2 + Real.exp x1 - 2) / (x2 - x1)) ∧
  (0 < Real.exp x1 ∧ Real.exp x1 < 1) :=
by
  sorry

end math_problem_l91_91685


namespace smallest_positive_period_monotonic_decreasing_interval_l91_91511

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) ^ 2 + 2 * Real.sin x * Real.cos x

theorem smallest_positive_period (T : ℝ) :
  (∀ x, f (x + T) = f x) ∧ T > 0 → T = Real.pi :=
by
  sorry

theorem monotonic_decreasing_interval :
  (∀ x, x ∈ Set.Icc (3 * Real.pi / 8) (7 * Real.pi / 8) → ∃ k : ℤ, 
     f (x + k * π) = f x ∧ f (x + k * π) ≤ f (x + (k + 1) * π)) :=
by
  sorry

end smallest_positive_period_monotonic_decreasing_interval_l91_91511


namespace solution_set_of_inequality_l91_91857

noncomputable def f : ℝ → ℝ
| x => if x < 2 then 2 * Real.exp (x - 1) else Real.log (x^2 - 1) / Real.log 3

theorem solution_set_of_inequality :
  {x : ℝ | f x > 2} = {x : ℝ | 1 < x ∧ x < 2} ∪ {x : ℝ | Real.sqrt 10 < x} :=
by
  sorry

end solution_set_of_inequality_l91_91857


namespace petya_maximum_margin_l91_91558

def max_margin_votes (total_votes first_period_margin last_period_margin : ℕ) (petya_vasaya_margin : ℕ) :=
  ∀ (P1 P2 V1 V2 : ℕ),
    (P1 + P2 + V1 + V2 = total_votes) →
    (P1 = V1 + first_period_margin) →
    (V2 = P2 + last_period_margin) →
    (P1 + P2 > V1 + V2) →
    petya_vasaya_margin = P1 + P2 - (V1 + V2)

theorem petya_maximum_margin
  (total_votes first_period_margin last_period_margin : ℕ)
  (h_total_votes: total_votes = 27)
  (h_first_period_margin: first_period_margin = 9)
  (h_last_period_margin: last_period_margin = 9):
  ∃ (petya_vasaya_margin : ℕ), max_margin_votes total_votes first_period_margin last_period_margin petya_vasaya_margin ∧ petya_vasaya_margin = 9 :=
by {
    sorry
}

end petya_maximum_margin_l91_91558


namespace smallest_right_triangle_area_l91_91097

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l91_91097


namespace smallest_number_l91_91756

theorem smallest_number (x : ℕ) (h1 : (x + 7) % 8 = 0) (h2 : (x + 7) % 11 = 0) (h3 : (x + 7) % 24 = 0) : x = 257 :=
sorry

end smallest_number_l91_91756


namespace smallest_right_triangle_area_l91_91106

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l91_91106


namespace find_a_l91_91049

theorem find_a (a : ℝ) : (∃ (p : ℝ × ℝ), p = (3, -9) ∧ (3 * a * p.1 + (2 * a + 1) * p.2 = 3 * a + 3)) → a = -1 :=
by
  sorry

end find_a_l91_91049


namespace hexagon_perimeter_sum_l91_91496

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

noncomputable def perimeter : ℝ := 
  distance 0 0 1 2 +
  distance 1 2 3 3 +
  distance 3 3 5 3 +
  distance 5 3 6 1 +
  distance 6 1 4 (-1) +
  distance 4 (-1) 0 0

theorem hexagon_perimeter_sum :
  perimeter = 3 * Real.sqrt 5 + 2 + 2 * Real.sqrt 2 + Real.sqrt 17 := 
sorry

end hexagon_perimeter_sum_l91_91496


namespace no_positive_numbers_satisfy_conditions_l91_91804

theorem no_positive_numbers_satisfy_conditions :
  ¬ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b + c = ab + ac + bc) ∧ (ab + ac + bc = abc) :=
by
  sorry

end no_positive_numbers_satisfy_conditions_l91_91804


namespace find_ice_cream_cost_l91_91649

def cost_of_ice_cream (total_paid cost_chapati cost_rice cost_vegetable : ℕ) (n_chapatis n_rice n_vegetables n_ice_cream : ℕ) : ℕ :=
  (total_paid - (n_chapatis * cost_chapati + n_rice * cost_rice + n_vegetables * cost_vegetable)) / n_ice_cream

theorem find_ice_cream_cost :
  let total_paid := 1051
  let cost_chapati := 6
  let cost_rice := 45
  let cost_vegetable := 70
  let n_chapatis := 16
  let n_rice := 5
  let n_vegetables := 7
  let n_ice_cream := 6
  cost_of_ice_cream total_paid cost_chapati cost_rice cost_vegetable n_chapatis n_rice n_vegetables n_ice_cream = 40 :=
by
  sorry

end find_ice_cream_cost_l91_91649


namespace hexagon_angles_l91_91702

theorem hexagon_angles (a e : ℝ) (h1 : a = e - 60) (h2 : 4 * a + 2 * e = 720) :
  e = 160 :=
by
  sorry

end hexagon_angles_l91_91702


namespace common_chord_eq_l91_91386

theorem common_chord_eq (x y : ℝ) :
  (x^2 + y^2 + 2*x + 8*y - 8 = 0) →
  (x^2 + y^2 - 4*x - 4*y - 2 = 0) →
  (x + 2*y - 1 = 0) :=
by
  intros h1 h2
  sorry

end common_chord_eq_l91_91386


namespace mary_change_l91_91991

def cost_of_berries : ℝ := 7.19
def cost_of_peaches : ℝ := 6.83
def amount_paid : ℝ := 20.00

theorem mary_change : amount_paid - (cost_of_berries + cost_of_peaches) = 5.98 := by
  sorry

end mary_change_l91_91991


namespace calculate_oplus_l91_91961

def op (X Y : ℕ) : ℕ :=
  (X + Y) / 2

theorem calculate_oplus : op (op 6 10) 14 = 11 := by
  sorry

end calculate_oplus_l91_91961


namespace petya_max_votes_difference_l91_91562

theorem petya_max_votes_difference :
  ∃ (P1 P2 V1 V2 : ℕ), 
    P1 = V1 + 9 ∧ 
    V2 = P2 + 9 ∧ 
    P1 + P2 + V1 + V2 = 27 ∧ 
    P1 + P2 > V1 + V2 ∧ 
    (P1 + P2) - (V1 + V2) = 9 := 
by
  sorry

end petya_max_votes_difference_l91_91562


namespace remaining_budget_l91_91779

theorem remaining_budget
  (initial_budget : ℕ)
  (cost_flasks : ℕ)
  (cost_test_tubes : ℕ)
  (cost_safety_gear : ℕ)
  (h1 : initial_budget = 325)
  (h2 : cost_flasks = 150)
  (h3 : cost_test_tubes = (2 * cost_flasks) / 3)
  (h4 : cost_safety_gear = cost_test_tubes / 2) :
  initial_budget - (cost_flasks + cost_test_tubes + cost_safety_gear) = 25 := 
  by
  sorry

end remaining_budget_l91_91779


namespace inequality_solution_l91_91055

theorem inequality_solution {x : ℝ} (h : |x + 3| - |x - 1| > 0) : x > -1 :=
sorry

end inequality_solution_l91_91055


namespace Katie_average_monthly_balance_l91_91581

def balances : List ℕ := [120, 240, 180, 180, 240]

def average (l : List ℕ) : ℕ := l.sum / l.length

theorem Katie_average_monthly_balance : average balances = 192 :=
by
  sorry

end Katie_average_monthly_balance_l91_91581


namespace A_and_D_mutual_independent_l91_91310

-- Probability theory definitions and assumptions.
noncomputable def prob_1_6 : ℚ := 1 / 6
noncomputable def prob_5_36 : ℚ := 5 / 36
noncomputable def prob_6_36 : ℚ := 6 / 36
noncomputable def prob_1_36 : ℚ := 1 / 36

-- Definitions of events with their corresponding probabilities.
def event_A (P : ℚ) : Prop := P = prob_1_6
def event_B (P : ℚ) : Prop := P = prob_1_6
def event_C (P : ℚ) : Prop := P = prob_5_36
def event_D (P : ℚ) : Prop := P = prob_6_36

-- Intersection probabilities:
def intersection_A_C (P : ℚ) : Prop := P = 0
def intersection_A_D (P : ℚ) : Prop := P = prob_1_36
def intersection_B_C (P : ℚ) : Prop := P = prob_1_36
def intersection_C_D (P : ℚ) : Prop := P = 0

-- Mutual independence definition.
def mutual_independent (P_X : ℚ) (P_Y : ℚ) (P_intersect : ℚ) : Prop :=
  P_X * P_Y = P_intersect

-- Theorem to prove:
theorem A_and_D_mutual_independent :
  event_A prob_1_6 →
  event_D prob_6_36 →
  intersection_A_D prob_1_36 →
  mutual_independent prob_1_6 prob_6_36 prob_1_36 := 
by 
  intros hA hD hAD
  rw [event_A, event_D, intersection_A_D] at hA hD hAD
  exact hA.symm ▸ hD.symm ▸ hAD.symm 

#check A_and_D_mutual_independent

end A_and_D_mutual_independent_l91_91310


namespace fraction_least_l91_91304

noncomputable def solve_fraction_least : Prop :=
  ∃ (x y : ℚ), x + y = 5/6 ∧ x * y = 1/8 ∧ (min x y = 1/6)
  
theorem fraction_least : solve_fraction_least :=
sorry

end fraction_least_l91_91304


namespace square_area_is_4802_l91_91732

-- Condition: the length of the diagonal of the square is 98 meters.
def diagonal (d : ℝ) := d = 98

-- Goal: Prove that the area of the square field is 4802 square meters.
theorem square_area_is_4802 (d : ℝ) (h : diagonal d) : ∃ (A : ℝ), A = 4802 := 
by sorry

end square_area_is_4802_l91_91732


namespace bananas_used_l91_91938

-- Define the conditions
def bananas_per_loaf := 4
def loaves_monday := 3
def loaves_tuesday := 2 * loaves_monday

-- Define the total bananas used
def bananas_monday := loaves_monday * bananas_per_loaf
def bananas_tuesday := loaves_tuesday * bananas_per_loaf
def total_bananas := bananas_monday + bananas_tuesday

-- Theorem statement to prove the total bananas used is 36
theorem bananas_used : total_bananas = 36 := by
  sorry

end bananas_used_l91_91938


namespace inequality_holds_l91_91625

theorem inequality_holds (x : ℝ) : x + 2 < x + 3 := 
by {
    sorry
}

end inequality_holds_l91_91625


namespace cost_of_paints_is_5_l91_91035

-- Define folders due to 6 classes
def folder_cost_per_item := 6
def num_classes := 6
def total_folder_cost : ℕ := folder_cost_per_item * num_classes

-- Define pencils due to the 6 classes and need per class
def pencil_cost_per_item := 2
def pencil_per_class := 3
def total_pencils : ℕ := pencil_per_class * num_classes
def total_pencil_cost : ℕ := pencil_cost_per_item * total_pencils

-- Define erasers needed based on pencils and their cost
def eraser_cost_per_item := 1
def pencils_per_eraser := 6
def total_erasers : ℕ := total_pencils / pencils_per_eraser
def total_eraser_cost : ℕ := eraser_cost_per_item * total_erasers

-- Total cost spent on folders, pencils, and erasers
def total_spent : ℕ := 80
def total_cost_supplies : ℕ := total_folder_cost + total_pencil_cost + total_eraser_cost

-- Cost of paints is the remaining amount when total cost is subtracted from total spent
def cost_of_paints : ℕ := total_spent - total_cost_supplies

-- The goal is to prove the cost of paints
theorem cost_of_paints_is_5 : cost_of_paints = 5 := by
  sorry

end cost_of_paints_is_5_l91_91035


namespace cube_less_than_three_times_square_l91_91899

theorem cube_less_than_three_times_square (x : ℤ) : x^3 < 3 * x^2 → x = 1 ∨ x = 2 :=
by
  sorry

end cube_less_than_three_times_square_l91_91899


namespace parallel_lines_a_value_l91_91253

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, 3 * x + 2 * a * y - 5 = 0 ↔ (3 * a - 1) * x - a * y - 2 = 0) →
  (a = 0 ∨ a = -1 / 6) :=
by
  sorry

end parallel_lines_a_value_l91_91253


namespace tom_spend_l91_91617

def theater_cost (seat_count : ℕ) (sqft_per_seat : ℕ) (cost_per_sqft : ℕ) (construction_multiplier : ℕ) (partner_percentage : ℝ) : ℝ :=
  let total_sqft := seat_count * sqft_per_seat
  let land_cost := total_sqft * cost_per_sqft
  let construction_cost := construction_multiplier * land_cost
  let total_cost := land_cost + construction_cost
  let partner_contribution := partner_percentage * (total_cost : ℝ)
  total_cost - partner_contribution

theorem tom_spend (partner_percentage : ℝ) :
  theater_cost 500 12 5 2 partner_percentage = 54000 :=
sorry

end tom_spend_l91_91617


namespace sin_x1_x2_value_l91_91005

open Real

theorem sin_x1_x2_value (m x1 x2 : ℝ) :
  (2 * sin (2 * x1) + cos (2 * x1) = m) →
  (2 * sin (2 * x2) + cos (2 * x2) = m) →
  (0 ≤ x1 ∧ x1 ≤ π / 2) →
  (0 ≤ x2 ∧ x2 ≤ π / 2) →
  sin (x1 + x2) = 2 * sqrt 5 / 5 := 
by
  sorry

end sin_x1_x2_value_l91_91005


namespace possible_values_of_a_l91_91686

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - a*x + 5 else a / x

theorem possible_values_of_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≥ f a y) ↔ (2 ≤ a ∧ a ≤ 3) :=
by
  sorry

end possible_values_of_a_l91_91686


namespace simplify_expression_l91_91999

variables {a b : ℝ}

theorem simplify_expression (a b : ℝ) : (2 * a^2 * b)^3 = 8 * a^6 * b^3 := 
by
  sorry

end simplify_expression_l91_91999


namespace largest_mersenne_prime_less_than_500_l91_91078

def mersenne_prime (n : ℕ) : ℕ := 2^n - 1

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem largest_mersenne_prime_less_than_500 :
  ∃ n, is_prime n ∧ mersenne_prime n < 500 ∧ ∀ m, is_prime m ∧ mersenne_prime m < 500 → mersenne_prime m ≤ mersenne_prime n :=
  sorry

end largest_mersenne_prime_less_than_500_l91_91078


namespace exist_sequences_l91_91660

def sequence_a (a : ℕ → ℤ) : Prop :=
  a 0 = 4 ∧ a 1 = 22 ∧ ∀ n ≥ 2, a n = 6 * a (n - 1) - a (n - 2)

theorem exist_sequences (a : ℕ → ℤ) (x y : ℕ → ℤ) :
  sequence_a a → (∀ n, 0 < x n ∧ 0 < y n) →
  (∀ n, a n = (y n ^ 2 + 7) / (x n - y n)) :=
by
  intro h_seq_a h_pos
  sorry

end exist_sequences_l91_91660


namespace probability_of_five_3s_is_099_l91_91357

-- Define conditions
def number_of_dice : ℕ := 15
def rolled_value : ℕ := 3
def probability_of_3 : ℚ := 1 / 8
def number_of_successes : ℕ := 5
def probability_of_not_3 : ℚ := 7 / 8

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability calculation
def probability_exactly_five_3s : ℚ :=
  binomial_coefficient number_of_dice number_of_successes *
  probability_of_3 ^ number_of_successes *
  probability_of_not_3 ^ (number_of_dice - number_of_successes)

theorem probability_of_five_3s_is_099 :
  probability_exactly_five_3s = 0.099 := by
  sorry -- Proof to be filled in later

end probability_of_five_3s_is_099_l91_91357


namespace number_of_sick_animals_l91_91060

def total_animals := 26 + 40 + 34  -- Total number of animals at Stacy's farm
def sick_fraction := 1 / 2  -- Half of all animals get sick

-- Defining sick animals for each type
def sick_chickens := 26 * sick_fraction
def sick_piglets := 40 * sick_fraction
def sick_goats := 34 * sick_fraction

-- The main theorem to prove
theorem number_of_sick_animals :
  sick_chickens + sick_piglets + sick_goats = 50 :=
by
  -- Skeleton of the proof that is to be completed later
  sorry

end number_of_sick_animals_l91_91060


namespace petya_wins_max_margin_l91_91549

theorem petya_wins_max_margin {P1 P2 V1 V2 : ℕ} 
  (h1 : P1 = V1 + 9)
  (h2 : V2 = P2 + 9)
  (h3 : P1 + P2 + V1 + V2 = 27)
  (h4 : P1 + P2 > V1 + V2) :
  ∃ m : ℕ, m = 9 ∧ P1 + P2 - (V1 + V2) = m :=
by
  sorry

end petya_wins_max_margin_l91_91549


namespace final_passenger_count_l91_91613

def total_passengers (initial : ℕ) (first_stop : ℕ) (off_bus : ℕ) (on_bus : ℕ) : ℕ :=
  (initial + first_stop) - off_bus + on_bus

theorem final_passenger_count :
  total_passengers 50 16 22 5 = 49 := by
  sorry

end final_passenger_count_l91_91613


namespace prop1_converse_prop1_inverse_prop1_contrapositive_prop2_converse_prop2_inverse_prop2_contrapositive_l91_91628

theorem prop1_converse (a b c : ℝ) (h : a * c^2 > b * c^2) : a > b := sorry

theorem prop1_inverse (a b c : ℝ) (h : a ≤ b) : a * c^2 ≤ b * c^2 := sorry

theorem prop1_contrapositive (a b c : ℝ) (h : a * c^2 ≤ b * c^2) : a ≤ b := sorry

theorem prop2_converse (a b c : ℝ) (f : ℝ → ℝ) (h : ∃x, f x = 0) : b^2 - 4 * a * c < 0 := sorry

theorem prop2_inverse (a b c : ℝ) (f : ℝ → ℝ) (h : b^2 - 4 * a * c ≥ 0) : ¬∃x, f x = 0 := sorry

theorem prop2_contrapositive (a b c : ℝ) (f : ℝ → ℝ) (h : ¬∃x, f x = 0) : b^2 - 4 * a * c ≥ 0 := sorry

end prop1_converse_prop1_inverse_prop1_contrapositive_prop2_converse_prop2_inverse_prop2_contrapositive_l91_91628


namespace connie_start_marbles_l91_91495

variable (marbles_total marbles_given marbles_left : ℕ)

theorem connie_start_marbles :
  marbles_given = 73 → marbles_left = 70 → marbles_total = marbles_given + marbles_left → marbles_total = 143 :=
by intros; sorry

end connie_start_marbles_l91_91495


namespace unique_b_for_quadratic_l91_91224

theorem unique_b_for_quadratic (c : ℝ) (h_c : c ≠ 0) : (∃! b : ℝ, b > 0 ∧ (2*b + 2/b)^2 - 4*c = 0) → c = 4 :=
by
  sorry

end unique_b_for_quadratic_l91_91224


namespace smallest_right_triangle_area_l91_91111

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l91_91111


namespace number_of_cows_l91_91748

def land_cost : ℕ := 30 * 20
def house_cost : ℕ := 120000
def chicken_cost : ℕ := 100 * 5
def installation_cost : ℕ := 6 * 100
def equipment_cost : ℕ := 6000
def total_cost : ℕ := 147700

theorem number_of_cows : 
  (total_cost - (land_cost + house_cost + chicken_cost + installation_cost + equipment_cost)) / 1000 = 20 := by
  sorry

end number_of_cows_l91_91748


namespace grid_minor_exists_l91_91372

theorem grid_minor_exists (r : ℤ) : ∃ k : ℤ, ∀ (G : Graph), has_treewidth_at_least G k → has_grid_minor G r r :=
by
  sorry

end grid_minor_exists_l91_91372


namespace smallest_right_triangle_area_l91_91123

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l91_91123


namespace math_proof_l91_91361

theorem math_proof :
  ∀ (x y z : ℚ), (2 * x - 3 * y - 2 * z = 0) →
                  (x + 3 * y - 28 * z = 0) →
                  (z ≠ 0) →
                  (x^2 + 3 * x * y * z) / (y^2 + z^2) = 280 / 37 :=
by
  intros x y z h1 h2 h3
  sorry

end math_proof_l91_91361


namespace calendar_reuse_initial_year_l91_91440

theorem calendar_reuse_initial_year (y k : ℕ)
    (h2064 : 2052 % 4 = 0)
    (h_y: y + 28 * k = 2052) :
    y = 1912 := by
  sorry

end calendar_reuse_initial_year_l91_91440


namespace james_calories_burned_per_week_l91_91421

theorem james_calories_burned_per_week :
  (let hours_per_class := 1.5
       minutes_per_hour := 60
       calories_per_minute := 7
       classes_per_week := 3
       minutes_per_class := hours_per_class * minutes_per_hour
       calories_per_class := minutes_per_class * calories_per_minute
       total_calories := calories_per_class * classes_per_week
   in total_calories) = 1890 := by
  sorry

end james_calories_burned_per_week_l91_91421


namespace maximize_expr_at_neg_5_l91_91462

-- Definition of the expression
def expr (x : ℝ) : ℝ := 1 - (x + 5) ^ 2

-- Prove that when x = -5, the expression has its maximum value
theorem maximize_expr_at_neg_5 : ∀ x : ℝ, expr x ≤ expr (-5) :=
by
  -- Placeholder for the proof
  sorry

end maximize_expr_at_neg_5_l91_91462


namespace sum_11_terms_l91_91385

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop := 
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n / 2) * (a 1 + a n)

def condition (a : ℕ → ℝ) : Prop :=
  a 5 + a 7 = 14

-- Proof Problem
theorem sum_11_terms (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : arithmetic_sequence a)
  (h_sum_formula : sum_first_n_terms S a)
  (h_condition : condition a) :
  S 11 = 77 := 
sorry

end sum_11_terms_l91_91385


namespace max_visible_sum_l91_91454

-- Definitions for the problem conditions

def numbers : List ℕ := [1, 3, 6, 12, 24, 48]

def num_faces (cubes : List ℕ) : Prop :=
  cubes.length = 18 -- since each of 3 cubes has 6 faces, we expect 18 numbers in total.

def is_valid_cube (cube : List ℕ) : Prop :=
  ∀ n ∈ cube, n ∈ numbers

def are_cubes (cubes : List (List ℕ)) : Prop :=
  cubes.length = 3 ∧ ∀ cube ∈ cubes, is_valid_cube cube ∧ cube.length = 6

-- The main theorem stating the maximum possible sum of the visible numbers
theorem max_visible_sum (cubes : List (List ℕ)) (h : are_cubes cubes) : ∃ s, s = 267 :=
by
  sorry

end max_visible_sum_l91_91454


namespace xyz_sum_is_22_l91_91015

theorem xyz_sum_is_22 (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x * y = 24) (h2 : x * z = 48) (h3 : y * z = 72) : 
  x + y + z = 22 :=
sorry

end xyz_sum_is_22_l91_91015


namespace calculate_LN_l91_91879

theorem calculate_LN (sinN : ℝ) (LM LN : ℝ) (h1 : sinN = 4 / 5) (h2 : LM = 20) : LN = 25 :=
by
  sorry

end calculate_LN_l91_91879


namespace cost_of_7_enchiladas_and_6_tacos_l91_91728

theorem cost_of_7_enchiladas_and_6_tacos (e t : ℝ) 
  (h₁ : 4 * e + 5 * t = 5.00) 
  (h₂ : 6 * e + 3 * t = 5.40) : 
  7 * e + 6 * t = 7.47 := 
sorry

end cost_of_7_enchiladas_and_6_tacos_l91_91728


namespace calculate_bankers_discount_l91_91740

noncomputable def present_worth : ℝ := 800
noncomputable def true_discount : ℝ := 36
noncomputable def face_value : ℝ := present_worth + true_discount
noncomputable def bankers_discount : ℝ := (face_value * true_discount) / (face_value - true_discount)

theorem calculate_bankers_discount :
  bankers_discount = 37.62 := 
sorry

end calculate_bankers_discount_l91_91740


namespace find_y_l91_91481

variables (y : ℝ)

def rectangle_vertices (A B C D : (ℝ × ℝ)) : Prop :=
  (A = (-2, y)) ∧ (B = (10, y)) ∧ (C = (-2, 1)) ∧ (D = (10, 1))

def rectangle_area (length height : ℝ) : Prop :=
  length * height = 108

def positive_value (x : ℝ) : Prop :=
  0 < x

theorem find_y (A B C D : (ℝ × ℝ)) (hV : rectangle_vertices y A B C D) (hA : rectangle_area 12 (y - 1)) (hP : positive_value y) :
  y = 10 :=
sorry

end find_y_l91_91481


namespace min_value_of_reciprocals_l91_91239

theorem min_value_of_reciprocals (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) : 
  ∃ x, x = (1 / a) + (1 / b) ∧ x ≥ 4 := 
sorry

end min_value_of_reciprocals_l91_91239


namespace total_new_people_last_year_l91_91025

-- Define the number of new people born and the number of people immigrated
def new_people_born : ℕ := 90171
def people_immigrated : ℕ := 16320

-- Prove that the total number of new people is 106491
theorem total_new_people_last_year : new_people_born + people_immigrated = 106491 := by
  sorry

end total_new_people_last_year_l91_91025


namespace find_total_students_l91_91633

variables (x X : ℕ)
variables (x_percent_students : ℕ) (total_students : ℕ)
variables (boys_fraction : ℝ)

-- Provided Conditions
axiom a1 : x_percent_students = 120
axiom a2 : boys_fraction = 0.30
axiom a3 : total_students = X

-- The theorem we need to prove
theorem find_total_students (a1 : 120 = x_percent_students) 
                            (a2 : boys_fraction = 0.30) 
                            (a3 : total_students = X) : 
  120 = (x / 100) * (boys_fraction * total_students) :=
sorry

end find_total_students_l91_91633


namespace exists_small_area_triangle_l91_91844

structure LatticePoint where
  x : Int
  y : Int

def isValidPoint (p : LatticePoint) : Prop := 
  |p.x| ≤ 2 ∧ |p.y| ≤ 2

def noThreeCollinear (points : List LatticePoint) : Prop := 
  ∀ (p1 p2 p3 : LatticePoint), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
  ((p2.x - p1.x) * (p3.y - p1.y) ≠ (p3.x - p1.x) * (p2.y - p1.y))

def triangleArea (p1 p2 p3 : LatticePoint) : ℝ :=
  0.5 * |(p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y) : ℝ)|

theorem exists_small_area_triangle
  (points : List LatticePoint)
  (h1 : ∀ p ∈ points, isValidPoint p)
  (h2 : noThreeCollinear points) :
  ∃ (p1 p2 p3 : LatticePoint), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ triangleArea p1 p2 p3 ≤ 2 :=
sorry

end exists_small_area_triangle_l91_91844


namespace john_buys_packs_l91_91848

theorem john_buys_packs :
  let classes := 6
  let students_per_class := 30
  let packs_per_student := 2
  let total_students := classes * students_per_class
  let total_packs := total_students * packs_per_student
  total_packs = 360 :=
by
  let classes := 6
  let students_per_class := 30
  let packs_per_student := 2
  let total_students := classes * students_per_class
  let total_packs := total_students * packs_per_student
  show total_packs = 360
  sorry

end john_buys_packs_l91_91848


namespace find_fg3_l91_91260

def f (x : ℝ) : ℝ := 2 * x - 5
def g (x : ℝ) : ℝ := x^2 + 1

theorem find_fg3 : f (g 3) = 15 :=
by
  sorry

end find_fg3_l91_91260


namespace mixed_alcohol_solution_l91_91877

theorem mixed_alcohol_solution 
    (vol_x : ℝ) (vol_y : ℝ) (conc_x : ℝ) (conc_y : ℝ) (target_conc : ℝ) (vol_y_given : vol_y = 750) 
    (conc_x_given : conc_x = 0.10) (conc_y_given : conc_y = 0.30) (target_conc_given : target_conc = 0.25) : 
    vol_x = 250 → 
    (conc_x * vol_x + conc_y * vol_y) / (vol_x + vol_y) = target_conc :=
by
  intros h_x
  rw [vol_y_given, conc_x_given, conc_y_given, target_conc_given, h_x]
  sorry

end mixed_alcohol_solution_l91_91877


namespace problem_1_problem_2_l91_91801

def simplify_calc : Prop :=
  125 * 3.2 * 25 = 10000

def solve_equation : Prop :=
  ∀ x: ℝ, 24 * (x - 12) = 16 * (x - 4) → x = 28

theorem problem_1 : simplify_calc :=
by
  sorry

theorem problem_2 : solve_equation :=
by
  sorry

end problem_1_problem_2_l91_91801


namespace food_needed_for_vacation_l91_91669

-- Define the conditions
def daily_food_per_dog := 250 -- in grams
def number_of_dogs := 4
def number_of_days := 14

-- Define the proof problem
theorem food_needed_for_vacation :
  (daily_food_per_dog * number_of_dogs * number_of_days / 1000) = 14 :=
by
  sorry

end food_needed_for_vacation_l91_91669


namespace find_number_l91_91081

theorem find_number (x : ℝ) : 60 + (x * 12) / (180 / 3) = 61 ↔ x = 5 := by
  sorry  -- proof can be filled in here when needed

end find_number_l91_91081


namespace runner_distance_l91_91701

theorem runner_distance (track_length race_length : ℕ) (A_speed B_speed C_speed : ℚ)
  (h1 : track_length = 400) (h2 : race_length = 800)
  (h3 : A_speed = 1) (h4 : B_speed = 8 / 7) (h5 : C_speed = 6 / 7) :
  ∃ distance_from_finish : ℚ, distance_from_finish = 200 :=
by {
  -- We are not required to provide the actual proof steps, just setting up the definitions and initial statements for the proof.
  sorry
}

end runner_distance_l91_91701


namespace math_proof_l91_91949

open BigOperators

noncomputable def problem_statement : Prop :=
  let floor_log_floor_sum := ∑ a in Finset.range 244, (Real.log a) / (Real.log 3).floor
  floor_log_floor_sum = 857

theorem math_proof : problem_statement := by
  -- problem and conditions
  sorry

end math_proof_l91_91949


namespace monotonic_invertible_function_l91_91942

theorem monotonic_invertible_function (f : ℝ → ℝ) (c : ℝ) (h_mono : ∀ x y, x < y → f x < f y) (h_inv : ∀ x, f (f⁻¹ x) = x) :
  (∀ x, f x + f⁻¹ x = 2 * x) ↔ ∀ x, f x = x + c :=
sorry

end monotonic_invertible_function_l91_91942


namespace cone_radius_l91_91054

theorem cone_radius
  (l : ℝ) (CSA : ℝ) (π : ℝ) (r : ℝ)
  (h_l : l = 15)
  (h_CSA : CSA = 141.3716694115407)
  (h_pi : π = Real.pi) :
  r = 3 :=
by
  sorry

end cone_radius_l91_91054


namespace xy_computation_l91_91750

theorem xy_computation (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : 
  x * y = 21 := by
  sorry

end xy_computation_l91_91750


namespace evaluate_expression_l91_91624

theorem evaluate_expression : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end evaluate_expression_l91_91624


namespace abs_diff_x_y_l91_91272

noncomputable def floor (z : ℝ) : ℤ := int.floor z
noncomputable def frac (z : ℝ) : ℝ := z - floor z

theorem abs_diff_x_y (x y : ℝ)
  (h1 : floor x + frac y = 3.7)
  (h2 : frac x + floor y = 4.2) :
  abs (x - y) = 1.5 :=
by
  sorry

end abs_diff_x_y_l91_91272


namespace inequality_solution_set_l91_91448

theorem inequality_solution_set (x : ℝ) : (x^2 + x) / (2*x - 1) ≤ 1 ↔ x < 1 / 2 := 
sorry

end inequality_solution_set_l91_91448


namespace cost_of_pack_of_socks_is_5_l91_91488

-- Conditions definitions
def shirt_price : ℝ := 12.00
def short_price : ℝ := 15.00
def trunks_price : ℝ := 14.00
def shirts_count : ℕ := 3
def shorts_count : ℕ := 2
def total_bill : ℝ := 102.00
def total_known_cost : ℝ := 3 * shirt_price + 2 * short_price + trunks_price

-- Definition of the problem statement
theorem cost_of_pack_of_socks_is_5 (S : ℝ) : total_bill = total_known_cost + S + 0.2 * (total_known_cost + S) → S = 5 := 
by
  sorry

end cost_of_pack_of_socks_is_5_l91_91488


namespace determine_a_l91_91467

theorem determine_a (a : ℝ) (x1 x2 : ℝ) :
  (x1 * x1 + (2 * a - 1) * x1 + a * a = 0) ∧
  (x2 * x2 + (2 * a - 1) * x2 + a * a = 0) ∧
  ((x1 + 2) * (x2 + 2) = 11) →
  a = -1 :=
by
  sorry

end determine_a_l91_91467


namespace sixty_five_percent_of_40_minus_four_fifths_of_25_l91_91959

theorem sixty_five_percent_of_40_minus_four_fifths_of_25 : 
  (0.65 * 40) - (0.8 * 25) = 6 := 
by
  sorry

end sixty_five_percent_of_40_minus_four_fifths_of_25_l91_91959


namespace n_fifth_plus_4n_mod_5_l91_91036

theorem n_fifth_plus_4n_mod_5 (n : ℕ) : (n^5 + 4 * n) % 5 = 0 := 
by
  sorry

end n_fifth_plus_4n_mod_5_l91_91036


namespace largest_fraction_l91_91906

theorem largest_fraction :
  let frac1 := (5 : ℚ) / 12
  let frac2 := (7 : ℚ) / 15
  let frac3 := (23 : ℚ) / 45
  let frac4 := (89 : ℚ) / 178
  let frac5 := (199 : ℚ) / 400
  frac3 > frac1 ∧ frac3 > frac2 ∧ frac3 > frac4 ∧ frac3 > frac5 :=
by
  let frac1 := (5 : ℚ) / 12
  let frac2 := (7 : ℚ) / 15
  let frac3 := (23 : ℚ) / 45
  let frac4 := (89 : ℚ) / 178
  let frac5 := (199 : ℚ) / 400
  sorry

end largest_fraction_l91_91906


namespace henry_books_l91_91832

def initial_books := 99
def boxes := 3
def books_per_box := 15
def room_books := 21
def coffee_table_books := 4
def kitchen_books := 18
def picked_books := 12

theorem henry_books :
  (initial_books - (boxes * books_per_box + room_books + coffee_table_books + kitchen_books) + picked_books) = 23 :=
by
  sorry

end henry_books_l91_91832


namespace trays_needed_to_fill_ice_cubes_l91_91667

-- Define the initial conditions
def ice_cubes_in_glass : Nat := 8
def multiplier_for_pitcher : Nat := 2
def spaces_per_tray : Nat := 12

-- Define the total ice cubes used
def total_ice_cubes_used : Nat := ice_cubes_in_glass + multiplier_for_pitcher * ice_cubes_in_glass

-- State the Lean theorem to be proven: The number of trays needed
theorem trays_needed_to_fill_ice_cubes : 
  total_ice_cubes_used / spaces_per_tray = 2 :=
  by 
  sorry

end trays_needed_to_fill_ice_cubes_l91_91667


namespace emily_sixth_score_needed_l91_91355

def emily_test_scores : List ℕ := [88, 92, 85, 90, 97]

def needed_sixth_score (scores : List ℕ) (target_mean : ℕ) : ℕ :=
  let current_sum := scores.sum
  let total_sum_needed := target_mean * (scores.length + 1)
  total_sum_needed - current_sum

theorem emily_sixth_score_needed :
  needed_sixth_score emily_test_scores 91 = 94 := by
  sorry

end emily_sixth_score_needed_l91_91355


namespace find_k_l91_91252

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (0, 1)

theorem find_k (k : ℝ) (h : dot_product (k * a.1, k * a.2 + b.2) (3 * a.1, 3 * a.2 - b.2) = 0) :
  k = 1 / 3 :=
by
  sorry

end find_k_l91_91252


namespace smallest_area_right_triangle_l91_91177

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l91_91177


namespace smallest_area_right_triangle_l91_91174

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l91_91174


namespace more_blue_marbles_l91_91064

theorem more_blue_marbles (r_boxes b_boxes marbles_per_box : ℕ) 
    (red_total_eq : r_boxes * marbles_per_box = 70) 
    (blue_total_eq : b_boxes * marbles_per_box = 126) 
    (r_boxes_eq : r_boxes = 5) 
    (b_boxes_eq : b_boxes = 9) 
    (marbles_per_box_eq : marbles_per_box = 14) : 
    126 - 70 = 56 := 
by 
  sorry

end more_blue_marbles_l91_91064


namespace intersect_sets_l91_91390

   variable (P : Set ℕ) (Q : Set ℕ)

   -- Definitions based on given conditions
   def P_def : Set ℕ := {1, 3, 5}
   def Q_def : Set ℕ := {x | 2 ≤ x ∧ x ≤ 5}

   -- Theorem statement in Lean 4
   theorem intersect_sets :
     P = P_def → Q = Q_def → P ∩ Q = {3, 5} :=
   by
     sorry
   
end intersect_sets_l91_91390


namespace jessica_routes_count_l91_91659

def line := Type

def valid_route_count (p q r s t u : line) : ℕ := 9 + 36 + 36

theorem jessica_routes_count (p q r s t u : line) :
  valid_route_count p q r s t u = 81 :=
by
  sorry

end jessica_routes_count_l91_91659


namespace shirt_cost_l91_91876

variables (S : ℝ)

theorem shirt_cost (h : 2 * S + (S + 3) + (1/2) * (2 * S + S + 3) = 36) : S = 7.88 :=
sorry

end shirt_cost_l91_91876


namespace problem1_l91_91184

   theorem problem1 : (Real.sqrt (9 / 4) + |2 - Real.sqrt 3| - (64 : ℝ) ^ (1 / 3) + 2⁻¹) = -Real.sqrt 3 :=
   by
     sorry
   
end problem1_l91_91184


namespace class_president_is_yi_l91_91922

variable (Students : Type)
variable (Jia Yi Bing StudyCommittee SportsCommittee ClassPresident : Students)
variable (age : Students → ℕ)

-- Conditions
axiom bing_older_than_study_committee : age Bing > age StudyCommittee
axiom jia_age_different_from_sports_committee : age Jia ≠ age SportsCommittee
axiom sports_committee_younger_than_yi : age SportsCommittee < age Yi

-- Prove that Yi is the class president
theorem class_president_is_yi : ClassPresident = Yi :=
sorry

end class_president_is_yi_l91_91922


namespace range_x_plus_y_l91_91522

theorem range_x_plus_y (x y : ℝ) (h : x^3 + y^3 = 2) : 0 < x + y ∧ x + y ≤ 2 :=
by {
  sorry
}

end range_x_plus_y_l91_91522


namespace inequality_solution_set_l91_91509

theorem inequality_solution_set :
  {x : ℝ | (x - 5) * (x + 1) > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 5} :=
by
  sorry

end inequality_solution_set_l91_91509


namespace books_sale_correct_l91_91994

variable (books_original books_left : ℕ)

def books_sold (books_original books_left : ℕ) : ℕ :=
  books_original - books_left

theorem books_sale_correct : books_sold 108 66 = 42 := by
  -- Since there is no need for the solution steps, we can assert the proof
  sorry

end books_sale_correct_l91_91994


namespace total_flight_time_l91_91335

theorem total_flight_time
  (distance : ℕ)
  (speed_out : ℕ)
  (speed_return : ℕ)
  (time_out : ℕ)
  (time_return : ℕ)
  (total_time : ℕ)
  (h1 : distance = 1500)
  (h2 : speed_out = 300)
  (h3 : speed_return = 500)
  (h4 : time_out = distance / speed_out)
  (h5 : time_return = distance / speed_return)
  (h6 : total_time = time_out + time_return) :
  total_time = 8 := 
  by {
    sorry
  }

end total_flight_time_l91_91335


namespace smallest_purple_marbles_l91_91424

theorem smallest_purple_marbles
  (n : ℕ)
  (h1 : n > 0)
  (h2 : n % 10 = 0)
  (h3 : 7 < (3 * n) / 10)
  (blue_marbles : ℕ := n / 2)
  (red_marbles : ℕ := n / 5)
  (green_marbles : ℕ := 7)
  (purple_marbles : ℕ := n - (blue_marbles + red_marbles + green_marbles)) :
  purple_marbles = 2 :=
by
  sorry

end smallest_purple_marbles_l91_91424


namespace smallest_area_right_triangle_l91_91130

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l91_91130


namespace half_of_animals_get_sick_l91_91061

theorem half_of_animals_get_sick : 
  let chickens := 26
  let piglets := 40
  let goats := 34
  let total_animals := chickens + piglets + goats
  let sick_animals := total_animals / 2
  sick_animals = 50 :=
by
  sorry

end half_of_animals_get_sick_l91_91061


namespace exists_k_tastrophic_function_l91_91582

noncomputable def k_tastrophic (f : ℕ+ → ℕ+) (k : ℕ) (n : ℕ+) : Prop :=
(f^[k] n) = n^k

theorem exists_k_tastrophic_function (k : ℕ) (h : k > 1) : ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, k_tastrophic f k n :=
by sorry

end exists_k_tastrophic_function_l91_91582


namespace factorization_of_polynomial_l91_91933

theorem factorization_of_polynomial : 
  ∀ (x : ℝ), 18 * x^3 + 9 * x^2 + 3 * x = 3 * x * (6 * x^2 + 3 * x + 1) :=
by sorry

end factorization_of_polynomial_l91_91933


namespace undefined_expression_l91_91816

theorem undefined_expression (b : ℝ) : b^2 - 9 = 0 ↔ b = 3 ∨ b = -3 := by
  sorry

end undefined_expression_l91_91816


namespace find_number_l91_91476

-- Define the conditions
def satisfies_condition (x : ℝ) : Prop := x * 4 * 25 = 812

-- The main theorem stating that the number satisfying the condition is 8.12
theorem find_number (x : ℝ) (h : satisfies_condition x) : x = 8.12 :=
by
  sorry

end find_number_l91_91476


namespace max_cookie_price_l91_91713

theorem max_cookie_price :
  ∃ k p : ℕ, 
    (8 * k + 3 * p < 200) ∧ 
    (4 * k + 5 * p > 150) ∧
    (∀ k' p' : ℕ, (8 * k' + 3 * p' < 200) ∧ (4 * k' + 5 * p' > 150) → k' ≤ 19) :=
sorry

end max_cookie_price_l91_91713


namespace smallest_right_triangle_area_l91_91147

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l91_91147


namespace antisymmetric_function_multiplication_cauchy_solution_l91_91597

variable (f : ℤ → ℤ)
variable (h : ∀ x y : ℤ, f (x + y) = f x + f y)

theorem antisymmetric : ∀ x : ℤ, f (-x) = -f x := by
  sorry

theorem function_multiplication : ∀ x y : ℤ, f (x * y) = x * f y := by
  sorry

theorem cauchy_solution : ∃ c : ℤ, ∀ x : ℤ, f x = c * x := by
  sorry

end antisymmetric_function_multiplication_cauchy_solution_l91_91597


namespace smallest_right_triangle_area_l91_91146

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l91_91146


namespace A_in_terms_of_B_l91_91858

-- Definitions based on conditions
def f (A B x : ℝ) : ℝ := A * x^2 - 3 * B^3
def g (B x : ℝ) : ℝ := B * x^2

-- Theorem statement
theorem A_in_terms_of_B (A B : ℝ) (hB : B ≠ 0) (h : f A B (g B 2) = 0) : A = 3 * B / 16 :=
by
  -- Proof omitted
  sorry

end A_in_terms_of_B_l91_91858


namespace height_radius_ratio_l91_91383

variables (R H V : ℝ) (π : ℝ) (A : ℝ)

-- Given conditions
def volume_condition : Prop := π * R^2 * H = V / 2
def surface_area : ℝ := 2 * π * R^2 + 2 * π * R * H

-- Statement to prove
theorem height_radius_ratio (h_volume : volume_condition R H V π) :
  H / R = 2 := 
sorry

end height_radius_ratio_l91_91383


namespace smallest_prime_divisor_sum_odd_powers_l91_91759

theorem smallest_prime_divisor_sum_odd_powers :
  (∃ p : ℕ, prime p ∧ p ∣ (3^15 + 11^21) ∧ p = 2) :=
by
  have h1 : 3^15 % 2 = 1 := by sorry
  have h2 : 11^21 % 2 = 1 := by sorry
  have h3 : (3^15 + 11^21) % 2 = 0 := by
    rw [← Nat.add_mod, h1, h2]
    exact Nat.mod_add_mod 1 1 2
  use 2
  constructor
  · exact Nat.prime_two
  · rw [Nat.dvd_iff_mod_eq_zero, h3] 
  · rfl

end smallest_prime_divisor_sum_odd_powers_l91_91759


namespace sum_series_eq_l91_91494

noncomputable def sum_series : ℕ → ℚ :=
λ n, ∑' n, (5 * n - 2) / 3^n

theorem sum_series_eq : sum_series 1 = 11 / 4 := 
sorry

end sum_series_eq_l91_91494


namespace polynomial_expansion_l91_91405

theorem polynomial_expansion :
  ∃ A B C D : ℝ, (∀ x : ℝ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) 
  ∧ (A + B + C + D = 36) :=
by {
  sorry
}

end polynomial_expansion_l91_91405


namespace oliver_shirts_not_washed_l91_91592

theorem oliver_shirts_not_washed :
  let short_sleeve_shirts := 39
  let long_sleeve_shirts := 47
  let total_shirts := short_sleeve_shirts + long_sleeve_shirts
  let washed_shirts := 20
  let not_washed_shirts := total_shirts - washed_shirts
  not_washed_shirts = 66 := by
  sorry

end oliver_shirts_not_washed_l91_91592


namespace find_m_l91_91249

-- Definition of the function as a direct proportion function with respect to x
def isDirectProportion (m : ℝ) : Prop :=
  m^2 - 8 = 1

-- Definition of the graph passing through the second and fourth quadrants
def passesThroughQuadrants (m : ℝ) : Prop :=
  m - 2 < 0

-- The theorem combining the conditions and proving the correct value of m
theorem find_m (m : ℝ) 
  (h1 : isDirectProportion m)
  (h2 : passesThroughQuadrants m) : 
  m = -3 :=
  sorry

end find_m_l91_91249


namespace remainder_division_l91_91916

theorem remainder_division (N : ℤ) (R1 : ℤ) (Q2 : ℤ) 
  (h1 : N = 44 * 432 + R1)
  (h2 : N = 38 * Q2 + 8) : 
  R1 = 0 := by
  sorry

end remainder_division_l91_91916


namespace smallest_area_right_triangle_l91_91140

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l91_91140


namespace largest_mersenne_prime_less_than_500_l91_91079

def mersenne_prime (n : ℕ) : ℕ := 2^n - 1

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem largest_mersenne_prime_less_than_500 :
  ∃ n, is_prime n ∧ mersenne_prime n < 500 ∧ ∀ m, is_prime m ∧ mersenne_prime m < 500 → mersenne_prime m ≤ mersenne_prime n :=
  sorry

end largest_mersenne_prime_less_than_500_l91_91079


namespace new_assistant_draw_time_l91_91909

-- Definitions based on conditions
def capacity : ℕ := 36
def halfway : ℕ := capacity / 2
def rate_top : ℕ := 1 / 6
def rate_bottom : ℕ := 1 / 4
def extra_time : ℕ := 24

-- The proof statement
theorem new_assistant_draw_time : 
  ∃ t : ℕ, ((capacity - (extra_time * rate_bottom * 1)) - halfway) = (t * rate_bottom * 1) ∧ t = 48 := by
sorry

end new_assistant_draw_time_l91_91909


namespace smallest_right_triangle_area_l91_91107

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l91_91107


namespace fish_in_aquarium_l91_91345

theorem fish_in_aquarium (initial_fish : ℕ) (added_fish : ℕ) (h1 : initial_fish = 10) (h2 : added_fish = 3) : initial_fish + added_fish = 13 := by
  sorry

end fish_in_aquarium_l91_91345


namespace max_xy_l91_91861

theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 3 * x + 8 * y = 48) : x * y ≤ 18 :=
sorry

end max_xy_l91_91861


namespace count_pos_int_multiple_6_lcm_gcd_l91_91688

open Nat

theorem count_pos_int_multiple_6_lcm_gcd : 
  let six_fact := fact 6
  let twelve_fact := fact 12
  ∃ (n : ℕ), n > 0 ∧ (n % 6 = 0) ∧ (Nat.lcm six_fact n = 6 * Nat.gcd twelve_fact n) ∧ (Finset.filter (λ n : ℕ, n > 0 ∧ (n % 6 = 0) ∧ (Nat.lcm six_fact n = 6 * Nat.gcd twelve_fact n)) (Finset.range 479001601)).card = 180 := 
  by sorry

end count_pos_int_multiple_6_lcm_gcd_l91_91688


namespace construct_using_five_twos_l91_91320

theorem construct_using_five_twos :
  (∃ (a b c d e f : ℕ), (22 * (a / b)) / c = 11 ∧
                        (22 / d) + (e / f) = 12 ∧
                        (22 + g + h) / i = 13 ∧
                        (2 * 2 * 2 * 2 - j) = 14 ∧
                        (22 / k) + (2 * 2) = 15) := by
  sorry

end construct_using_five_twos_l91_91320


namespace petya_maximum_margin_l91_91561

def max_margin_votes (total_votes first_period_margin last_period_margin : ℕ) (petya_vasaya_margin : ℕ) :=
  ∀ (P1 P2 V1 V2 : ℕ),
    (P1 + P2 + V1 + V2 = total_votes) →
    (P1 = V1 + first_period_margin) →
    (V2 = P2 + last_period_margin) →
    (P1 + P2 > V1 + V2) →
    petya_vasaya_margin = P1 + P2 - (V1 + V2)

theorem petya_maximum_margin
  (total_votes first_period_margin last_period_margin : ℕ)
  (h_total_votes: total_votes = 27)
  (h_first_period_margin: first_period_margin = 9)
  (h_last_period_margin: last_period_margin = 9):
  ∃ (petya_vasaya_margin : ℕ), max_margin_votes total_votes first_period_margin last_period_margin petya_vasaya_margin ∧ petya_vasaya_margin = 9 :=
by {
    sorry
}

end petya_maximum_margin_l91_91561


namespace closest_point_on_parabola_to_line_l91_91010

noncomputable def line := { P : ℝ × ℝ | 2 * P.1 - P.2 = 4 }
noncomputable def parabola := { P : ℝ × ℝ | P.2 = P.1^2 }

theorem closest_point_on_parabola_to_line : 
  ∃ P : ℝ × ℝ, P ∈ parabola ∧ 
  (∀ Q ∈ parabola, ∀ R ∈ line, dist P R ≤ dist Q R) ∧ 
  P = (1, 1) := 
sorry

end closest_point_on_parabola_to_line_l91_91010


namespace find_r_l91_91693

theorem find_r (r : ℝ) (h₁ : 0 < r) (h₂ : ∀ x y : ℝ, (x - y = r → x^2 + y^2 = r → False)) : r = 2 :=
sorry

end find_r_l91_91693


namespace smallest_area_of_right_triangle_l91_91085

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l91_91085


namespace find_value_l91_91379

theorem find_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2010 = 2011 := 
sorry

end find_value_l91_91379


namespace johns_total_packs_l91_91852

-- Defining the conditions
def classes : ℕ := 6
def students_per_class : ℕ := 30
def packs_per_student : ℕ := 2

-- Theorem statement
theorem johns_total_packs : 
  (classes * students_per_class * packs_per_student) = 360 :=
by
  -- The proof would go here
  sorry

end johns_total_packs_l91_91852


namespace remainder_when_divided_by_30_l91_91288

theorem remainder_when_divided_by_30 (y : ℤ)
  (h1 : 4 + y ≡ 9 [ZMOD 8])
  (h2 : 6 + y ≡ 8 [ZMOD 27])
  (h3 : 8 + y ≡ 27 [ZMOD 125]) :
  y ≡ 4 [ZMOD 30] :=
sorry

end remainder_when_divided_by_30_l91_91288


namespace ratio_ac_bd_l91_91536

theorem ratio_ac_bd (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 2 * c) (h3 : c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end ratio_ac_bd_l91_91536


namespace exponential_function_pass_through_point_l91_91048

theorem exponential_function_pass_through_point
  (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (a^(1 - 1) + 1 = 2) :=
by
  sorry

end exponential_function_pass_through_point_l91_91048


namespace x_mul_y_eq_4_l91_91039

theorem x_mul_y_eq_4 (x y z w : ℝ) (hw_pos : w > 0) 
  (h1 : x = w) (h2 : y = z) (h3 : w + w = z * w) 
  (h4 : y = w) (h5 : z = 3) (h6 : w + w = w * w) : 
  x * y = 4 := by
  sorry

end x_mul_y_eq_4_l91_91039


namespace jenny_investment_l91_91707

theorem jenny_investment :
  ∃ (m r : ℝ), m + r = 240000 ∧ r = 6 * m ∧ r = 205714.29 :=
by
  sorry

end jenny_investment_l91_91707


namespace younger_brother_silver_fraction_l91_91295

def frac_silver (x y : ℕ) : ℚ := (100 - x / 7 ) / y

theorem younger_brother_silver_fraction {x y : ℕ} 
    (cond1 : x / 5 + y / 7 = 100) 
    (cond2 : x / 7 + (100 - x / 7) = 100) : 
    frac_silver x y = 5 / 14 := 
sorry

end younger_brother_silver_fraction_l91_91295


namespace candy_bars_eaten_l91_91307

theorem candy_bars_eaten (calories_per_candy : ℕ) (total_calories : ℕ) (h1 : calories_per_candy = 31) (h2 : total_calories = 341) :
  total_calories / calories_per_candy = 11 :=
by
  sorry

end candy_bars_eaten_l91_91307


namespace evaluate_expression_l91_91623

theorem evaluate_expression : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end evaluate_expression_l91_91623


namespace centroid_calculation_correct_l91_91318

-- Define the vertices of the triangle
def P : ℝ × ℝ := (2, 3)
def Q : ℝ × ℝ := (-1, 4)
def R : ℝ × ℝ := (4, -2)

-- Define the coordinates of the centroid
noncomputable def S : ℝ × ℝ := ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3)

-- Prove that 7x + 2y = 15 for the centroid
theorem centroid_calculation_correct : 7 * S.1 + 2 * S.2 = 15 :=
by 
  -- Placeholder for the proof steps
  sorry

end centroid_calculation_correct_l91_91318


namespace ella_distance_from_start_l91_91213

noncomputable def compute_distance (m1 : ℝ) (f1 f2 m_to_f : ℝ) : ℝ :=
  let f1' := m1 * m_to_f
  let total_west := f1' + f2
  let distance_in_feet := Real.sqrt (f1^2 + total_west^2)
  distance_in_feet / m_to_f

theorem ella_distance_from_start :
  let starting_west := 10
  let first_north := 30
  let second_west := 40
  let meter_to_feet := 3.28084 
  compute_distance starting_west first_north second_west meter_to_feet = 24.01 := sorry

end ella_distance_from_start_l91_91213


namespace half_of_animals_get_sick_l91_91062

theorem half_of_animals_get_sick : 
  let chickens := 26
  let piglets := 40
  let goats := 34
  let total_animals := chickens + piglets + goats
  let sick_animals := total_animals / 2
  sick_animals = 50 :=
by
  sorry

end half_of_animals_get_sick_l91_91062


namespace total_time_watching_videos_l91_91800

theorem total_time_watching_videos 
  (cat_video_length : ℕ)
  (dog_video_length : ℕ)
  (gorilla_video_length : ℕ)
  (h1 : cat_video_length = 4)
  (h2 : dog_video_length = 2 * cat_video_length)
  (h3 : gorilla_video_length = 2 * (cat_video_length + dog_video_length)) :
  cat_video_length + dog_video_length + gorilla_video_length = 36 :=
  by
  sorry

end total_time_watching_videos_l91_91800


namespace sum_of_two_coprimes_l91_91593

theorem sum_of_two_coprimes (n : ℤ) (h : n ≥ 7) : 
  ∃ a b : ℤ, a + b = n ∧ Int.gcd a b = 1 ∧ a > 1 ∧ b > 1 :=
by
  sorry

end sum_of_two_coprimes_l91_91593


namespace smallest_area_of_right_triangle_l91_91163

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l91_91163
