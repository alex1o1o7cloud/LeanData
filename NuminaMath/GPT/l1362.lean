import Mathlib

namespace widget_cost_reduction_l1362_136228

theorem widget_cost_reduction:
  ∀ (C C_reduced : ℝ), 
  6 * C = 27.60 → 
  8 * C_reduced = 27.60 → 
  C - C_reduced = 1.15 := 
by
  intros C C_reduced h1 h2
  sorry

end widget_cost_reduction_l1362_136228


namespace rational_numbers_on_circle_l1362_136277

theorem rational_numbers_on_circle (a b c d e f : ℚ)
  (h1 : a = |b - c|)
  (h2 : b = d)
  (h3 : c = |d - e|)
  (h4 : d = |e - f|)
  (h5 : e = f)
  (h6 : a + b + c + d + e + f = 1) :
  [a, b, c, d, e, f] = [1/4, 1/4, 0, 1/4, 1/4, 0] :=
sorry

end rational_numbers_on_circle_l1362_136277


namespace johns_haircut_tip_percentage_l1362_136271

noncomputable def percent_of_tip (annual_spending : ℝ) (haircut_cost : ℝ) (haircut_frequency : ℕ) : ℝ := 
  ((annual_spending / haircut_frequency - haircut_cost) / haircut_cost) * 100

theorem johns_haircut_tip_percentage : 
  let hair_growth_rate : ℝ := 1.5
  let initial_length : ℝ := 6
  let max_length : ℝ := 9
  let haircut_cost : ℝ := 45
  let annual_spending : ℝ := 324
  let months_in_year : ℕ := 12
  let growth_period := 2 -- months it takes for hair to grow 3 inches
  let haircuts_per_year := months_in_year / growth_period -- number of haircuts per year
  percent_of_tip annual_spending haircut_cost haircuts_per_year = 20 := by
  sorry

end johns_haircut_tip_percentage_l1362_136271


namespace no_valid_solutions_l1362_136282

theorem no_valid_solutions (a b : ℝ) (h1 : ∀ x, (a * x + b) ^ 2 = 4 * x^2 + 4 * x + 4) : false :=
  by
  sorry

end no_valid_solutions_l1362_136282


namespace stream_speed_l1362_136200

theorem stream_speed (c v : ℝ) (h1 : c - v = 9) (h2 : c + v = 12) : v = 1.5 :=
by
  sorry

end stream_speed_l1362_136200


namespace set_intersection_complement_l1362_136210

open Set

noncomputable def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
noncomputable def B : Set ℝ := {x | 0 < x ∧ x < 4}

theorem set_intersection_complement :
  (compl A ∩ B) = {x | 0 < x ∧ x ≤ 3} :=
by
  sorry

end set_intersection_complement_l1362_136210


namespace remainder_of_s_minus_t_plus_t_minus_u_l1362_136267

theorem remainder_of_s_minus_t_plus_t_minus_u (s t u : ℕ) (hs : s % 12 = 4) (ht : t % 12 = 5) (hu : u % 12 = 7) (h_order : s > t ∧ t > u) :
  ((s - t) + (t - u)) % 12 = 9 :=
by sorry

end remainder_of_s_minus_t_plus_t_minus_u_l1362_136267


namespace chapter_page_difference_l1362_136203

/-- The first chapter of a book has 37 pages -/
def first_chapter_pages : Nat := 37

/-- The second chapter of a book has 80 pages -/
def second_chapter_pages : Nat := 80

/-- Prove the difference in the number of pages between the second and the first chapter is 43 -/
theorem chapter_page_difference : (second_chapter_pages - first_chapter_pages) = 43 := by
  sorry

end chapter_page_difference_l1362_136203


namespace no_real_roots_iff_l1362_136258

theorem no_real_roots_iff (k : ℝ) : (∀ x : ℝ, x^2 - 2*x - k ≠ 0) ↔ k < -1 :=
by
  sorry

end no_real_roots_iff_l1362_136258


namespace share_per_person_l1362_136265

-- Defining the total cost and number of people
def total_cost : ℝ := 12100
def num_people : ℝ := 11

-- The theorem stating that each person's share is $1,100.00
theorem share_per_person : total_cost / num_people = 1100 := by
  sorry

end share_per_person_l1362_136265


namespace certain_number_is_1862_l1362_136230

theorem certain_number_is_1862 (G N : ℕ) (hG: G = 4) (hN: ∃ k : ℕ, N = G * k + 6) (h1856: ∃ m : ℕ, 1856 = G * m + 4) : N = 1862 :=
by
  sorry

end certain_number_is_1862_l1362_136230


namespace simplify_expression_l1362_136284

theorem simplify_expression (y : ℝ) :
  4 * y - 8 * y^2 + 6 - (3 - 6 * y - 9 * y^2 + 2 * y^3) = -2 * y^3 + y^2 + 10 * y + 3 := 
by
  -- Proof goes here, but we just state sorry for now
  sorry

end simplify_expression_l1362_136284


namespace union_A_B_eq_univ_inter_compl_A_B_eq_interval_subset_B_range_of_a_l1362_136294

variable (A B C : Set ℝ)
variable (a : ℝ)

-- Condition definitions
def set_A : Set ℝ := {x | x ≤ 3 ∨ x ≥ 6}
def set_B : Set ℝ := {x | -2 < x ∧ x < 9}
def set_C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Proof statement (1)
theorem union_A_B_eq_univ (A B : Set ℝ) (h₁ : A = set_A) (h₂ : B = set_B) :
  A ∪ B = Set.univ := by sorry

theorem inter_compl_A_B_eq_interval (A B : Set ℝ) (h₁ : A = set_A) (h₂ : B = set_B) :
  (Set.univ \ A) ∩ B = {x | 3 < x ∧ x < 6} := by sorry

-- Proof statement (2)
theorem subset_B_range_of_a (a : ℝ) (h : set_C a ⊆ set_B) :
  -2 ≤ a ∧ a ≤ 8 := by sorry

end union_A_B_eq_univ_inter_compl_A_B_eq_interval_subset_B_range_of_a_l1362_136294


namespace hyperbola_intersection_l1362_136223

theorem hyperbola_intersection (b : ℝ) (h₁ : b > 0) :
  (b > 1) → (∀ x y : ℝ, ((x + 3 * y - 1 = 0) → ( ∃ x y : ℝ, (x^2 / 4 - y^2 / b^2 = 1) ∧ (x + 3 * y - 1 = 0))))
  :=
  sorry

end hyperbola_intersection_l1362_136223


namespace find_smallest_in_arithmetic_progression_l1362_136286

theorem find_smallest_in_arithmetic_progression (a d : ℝ)
  (h1 : (a-2*d)^3 + (a-d)^3 + a^3 + (a+d)^3 + (a+2*d)^3 = 0)
  (h2 : (a-2*d)^4 + (a-d)^4 + a^4 + (a+d)^4 + (a+2*d)^4 = 136) :
  (a - 2*d) = -2 * Real.sqrt 2 :=
sorry

end find_smallest_in_arithmetic_progression_l1362_136286


namespace intersection_of_sets_l1362_136261

open Set

variable {x : ℝ}

theorem intersection_of_sets : 
  let A := {x : ℝ | x^2 - 4*x + 3 < 0}
  let B := {x : ℝ | x > 2}
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_of_sets_l1362_136261


namespace farmer_price_per_dozen_l1362_136207

noncomputable def price_per_dozen 
(farmer_chickens : ℕ) 
(eggs_per_chicken : ℕ) 
(total_money_made : ℕ) 
(total_weeks : ℕ) 
(eggs_per_dozen : ℕ) 
: ℕ :=
total_money_made / (total_weeks * (farmer_chickens * eggs_per_chicken) / eggs_per_dozen)

theorem farmer_price_per_dozen 
  (farmer_chickens : ℕ) 
  (eggs_per_chicken : ℕ) 
  (total_money_made : ℕ) 
  (total_weeks : ℕ) 
  (eggs_per_dozen : ℕ) 
  (h_chickens : farmer_chickens = 46) 
  (h_eggs_per_chicken : eggs_per_chicken = 6) 
  (h_money : total_money_made = 552) 
  (h_weeks : total_weeks = 8) 
  (h_dozen : eggs_per_dozen = 12) 
: price_per_dozen farmer_chickens eggs_per_chicken total_money_made total_weeks eggs_per_dozen = 3 := 
by 
  rw [h_chickens, h_eggs_per_chicken, h_money, h_weeks, h_dozen]
  have : (552 : ℕ) / (8 * (46 * 6) / 12) = 3 := by norm_num
  exact this

end farmer_price_per_dozen_l1362_136207


namespace compute_expression_l1362_136215

theorem compute_expression (x : ℝ) (hx : x + 1 / x = 7) : 
  (x - 3)^2 + 36 / (x - 3)^2 = 12.375 := 
  sorry

end compute_expression_l1362_136215


namespace probability_of_A_winning_l1362_136279

-- Define the conditions
def p : ℝ := 0.6
def q : ℝ := 1 - p  -- probability of losing a set

-- Formulate the probabilities for each win scenario
def P_WW : ℝ := p * p
def P_LWW : ℝ := q * p * p
def P_WLW : ℝ := p * q * p

-- Calculate the total probability of winning the match
def total_probability : ℝ := P_WW + P_LWW + P_WLW

-- Prove that the total probability of A winning the match is 0.648
theorem probability_of_A_winning : total_probability = 0.648 :=
by
    -- Provide the calculation details
    sorry  -- replace with the actual proof steps if needed, otherwise keep sorry to skip the proof

end probability_of_A_winning_l1362_136279


namespace max_AB_CD_value_l1362_136280

def is_digit (x : ℕ) : Prop := x ≥ 1 ∧ x ≤ 9

noncomputable def max_AB_CD : ℕ :=
  let A := 9
  let B := 8
  let C := 7
  let D := 6
  (A + B) + (C + D)

theorem max_AB_CD_value :
  ∀ (A B C D : ℕ), 
    is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧ 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
    (A + B) + (C + D) ≤ max_AB_CD :=
by
  sorry

end max_AB_CD_value_l1362_136280


namespace evaluate_expression_l1362_136221

open Complex

theorem evaluate_expression (a b : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + a * b + b^2 = 0) :
  (a^6 + b^6) / (a + b)^6 = 18 :=
by
  sorry

end evaluate_expression_l1362_136221


namespace matrix_power_101_l1362_136281

noncomputable def matrix_B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem matrix_power_101 :
  (matrix_B ^ 101) = ![![0, 0, 1], ![1, 0, 0], ![0, 1, 0]] :=
  sorry

end matrix_power_101_l1362_136281


namespace distinct_convex_polygons_of_four_or_more_sides_l1362_136297

noncomputable def total_subsets (n : Nat) : Nat := 2^n

noncomputable def subsets_with_fewer_than_four_members (n : Nat) : Nat := 
  (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)

noncomputable def valid_subsets (n : Nat) : Nat := 
  total_subsets n - subsets_with_fewer_than_four_members n

theorem distinct_convex_polygons_of_four_or_more_sides (n : Nat) (h : n = 15) : valid_subsets n = 32192 := by
  sorry

end distinct_convex_polygons_of_four_or_more_sides_l1362_136297


namespace range_of_a_l1362_136298

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 3 → log (x - 1) + log (3 - x) = log (a - x)) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 1 < x₁ ∧ x₁ < 3 ∧ 1 < x₂ ∧ x₂ < 3) →
  3 < a ∧ a < 13 / 4 :=
by
  sorry

end range_of_a_l1362_136298


namespace greatest_possible_third_side_l1362_136274

theorem greatest_possible_third_side (t : ℕ) (h : 5 < t ∧ t < 15) : t = 14 :=
sorry

end greatest_possible_third_side_l1362_136274


namespace shaded_area_correct_l1362_136218

def first_rectangle_area (w l : ℕ) : ℕ := w * l
def second_rectangle_area (w l : ℕ) : ℕ := w * l
def overlap_triangle_area (b h : ℕ) : ℕ := (b * h) / 2
def total_shaded_area (area1 area2 overlap : ℕ) : ℕ := area1 + area2 - overlap

theorem shaded_area_correct :
  let w1 := 4
  let l1 := 12
  let w2 := 5
  let l2 := 10
  let b := 4
  let h := 5
  let area1 := first_rectangle_area w1 l1
  let area2 := second_rectangle_area w2 l2
  let overlap := overlap_triangle_area b h
  total_shaded_area area1 area2 overlap = 88 := 
by
  sorry

end shaded_area_correct_l1362_136218


namespace intersection_with_single_element_union_equals_A_l1362_136205

-- Definitions of the sets A and B
def A := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) := {x : ℝ | x^2 + 2 * (a + 1) * x + (a^2 - 5) = 0}

-- Statement for question (1)
theorem intersection_with_single_element (a : ℝ) (H : A = {1, 2} ∧ A ∩ B a = {2}) : a = -1 ∨ a = -3 :=
by
  sorry

-- Statement for question (2)
theorem union_equals_A (a : ℝ) (H1 : A = {1, 2}) (H2 : A ∪ B a = A) : (a ≥ -3 ∧ a ≤ -1) :=
by
  sorry

end intersection_with_single_element_union_equals_A_l1362_136205


namespace seq_problem_part1_seq_problem_part2_l1362_136238

def seq (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 2) = |a (n + 1) - a n|

theorem seq_problem_part1 (a : ℕ → ℤ) 
  (h1 : a 1 = 2)
  (h2 : a 2 = -1)
  (h_seq : seq a) :
  a 2008 = 0 := 
sorry

theorem seq_problem_part2 (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (h2 : a 2 = -1)
  (h_seq : seq a) :
  ∃ (M : ℤ), 
  (∀ n : ℕ, ∃ m : ℕ, m > n ∧ a m = 0) ∧ 
  (∀ n : ℕ, ∃ m : ℕ, m > n ∧ a m = M) := 
sorry

end seq_problem_part1_seq_problem_part2_l1362_136238


namespace find_m_eq_2_l1362_136208

theorem find_m_eq_2 (a m : ℝ) (h1 : a > 0) (h2 : -a * m^2 + 2 * a * m + 3 = 3) (h3 : m ≠ 0) : m = 2 :=
by
  sorry

end find_m_eq_2_l1362_136208


namespace intersection_point_exists_l1362_136269

def equation_1 (x y : ℝ) : Prop := 3 * x^2 - 12 * y^2 = 48
def line_eq (x y : ℝ) : Prop := y = - (1 / 3) * x + 5

theorem intersection_point_exists :
  ∃ (x y : ℝ), equation_1 x y ∧ line_eq x y ∧ x = 75 / 8 ∧ y = 15 / 8 :=
sorry

end intersection_point_exists_l1362_136269


namespace product_of_odd_primes_mod_32_l1362_136299

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def M : ℕ := odd_primes_less_than_32.foldr (· * ·) 1

theorem product_of_odd_primes_mod_32 : M % 32 = 17 :=
by
  -- the proof goes here
  sorry

end product_of_odd_primes_mod_32_l1362_136299


namespace pascal_row_10_sum_l1362_136273

-- Definition: sum of the numbers in Row n of Pascal's Triangle is 2^n
def pascal_row_sum (n : ℕ) : ℕ := 2^n

-- Theorem: sum of the numbers in Row 10 of Pascal's Triangle is 1024
theorem pascal_row_10_sum : pascal_row_sum 10 = 1024 :=
by
  sorry

end pascal_row_10_sum_l1362_136273


namespace no_integer_for_58th_power_64_digits_valid_replacement_for_64_digits_l1362_136259

theorem no_integer_for_58th_power_64_digits : ¬ ∃ n : ℤ, 10^63 ≤ n^58 ∧ n^58 < 10^64 :=
sorry

theorem valid_replacement_for_64_digits (k : ℕ) (hk : 1 ≤ k ∧ k ≤ 81) : 
  ¬ ∃ n : ℤ, 10^(k-1) ≤ n^58 ∧ n^58 < 10^k :=
sorry

end no_integer_for_58th_power_64_digits_valid_replacement_for_64_digits_l1362_136259


namespace shopkeeper_loss_amount_l1362_136235

theorem shopkeeper_loss_amount (total_stock_worth : ℝ)
                               (portion_sold_at_profit : ℝ)
                               (portion_sold_at_loss : ℝ)
                               (profit_percentage : ℝ)
                               (loss_percentage : ℝ) :
  total_stock_wworth = 14999.999999999996 →
  portion_sold_at_profit = 0.2 →
  portion_sold_at_loss = 0.8 →
  profit_percentage = 0.10 →
  loss_percentage = 0.05 →
  (total_stock_worth - ((portion_sold_at_profit * total_stock_worth * (1 + profit_percentage)) + 
                        (portion_sold_at_loss * total_stock_worth * (1 - loss_percentage)))) = 300 := 
by 
  sorry

end shopkeeper_loss_amount_l1362_136235


namespace pow_neg_one_diff_l1362_136276

theorem pow_neg_one_diff (n : ℤ) (h1 : n = 2010) (h2 : n + 1 = 2011) :
  (-1)^2010 - (-1)^2011 = 2 := 
by
  sorry

end pow_neg_one_diff_l1362_136276


namespace inequality_holds_if_and_only_if_l1362_136262

variable (x : ℝ) (b : ℝ)

theorem inequality_holds_if_and_only_if (hx : |x-5| + |x-3| + |x-2| < b) : b > 4 :=
sorry

end inequality_holds_if_and_only_if_l1362_136262


namespace m_necessary_not_sufficient_cond_l1362_136270

theorem m_necessary_not_sufficient_cond (m : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → x^3 - 3 * x + m = 0) → m ≤ 2 :=
sorry

end m_necessary_not_sufficient_cond_l1362_136270


namespace sum_sequence_correct_l1362_136295

def sequence_term (n : ℕ) : ℕ :=
  if n % 9 = 0 ∧ n % 32 = 0 then 7
  else if n % 7 = 0 ∧ n % 32 = 0 then 9
  else if n % 7 = 0 ∧ n % 9 = 0 then 32
  else 0

def sequence_sum (up_to : ℕ) : ℕ :=
  (Finset.range (up_to + 1)).sum sequence_term

theorem sum_sequence_correct : sequence_sum 2015 = 1106 := by
  sorry

end sum_sequence_correct_l1362_136295


namespace average_gas_mileage_round_trip_l1362_136257

theorem average_gas_mileage_round_trip :
  (300 / ((150 / 28) + (150 / 18))) = 22 := by
sorry

end average_gas_mileage_round_trip_l1362_136257


namespace odd_number_diff_of_squares_l1362_136224

theorem odd_number_diff_of_squares (k : ℕ) : ∃ n : ℕ, k = (n+1)^2 - n^2 ↔ ∃ m : ℕ, k = 2 * m + 1 := 
by 
  sorry

end odd_number_diff_of_squares_l1362_136224


namespace time_for_tom_to_finish_wall_l1362_136231

theorem time_for_tom_to_finish_wall (avery_rate tom_rate : ℝ) (combined_duration : ℝ) (remaining_wall : ℝ) :
  avery_rate = 1 / 2 ∧ tom_rate = 1 / 4 ∧ combined_duration = 1 ∧ remaining_wall = 1 / 4 →
  (remaining_wall / tom_rate) = 1 :=
by
  intros h
  -- Definitions from conditions
  let avery_rate := 1 / 2
  let tom_rate := 1 / 4
  let combined_duration := 1
  let remaining_wall := 1 / 4
  -- Question to be proven
  sorry

end time_for_tom_to_finish_wall_l1362_136231


namespace binom_12_9_eq_220_l1362_136242

noncomputable def binom (n k : ℕ) : ℕ :=
  n.choose k

theorem binom_12_9_eq_220 : binom 12 9 = 220 :=
sorry

end binom_12_9_eq_220_l1362_136242


namespace no_solution_to_inequalities_l1362_136254

theorem no_solution_to_inequalities : 
  ∀ x : ℝ, ¬ (4 * x - 3 < (x + 2)^2 ∧ (x + 2)^2 < 8 * x - 5) :=
by
  sorry

end no_solution_to_inequalities_l1362_136254


namespace find_x_minus_y_l1362_136240

def rotated_point (x y h k : ℝ) : ℝ × ℝ := (2 * h - x, 2 * k - y)

def reflected_point (x y : ℝ) : ℝ × ℝ := (y, x)

def transformed_point (x y : ℝ) : ℝ × ℝ :=
  reflected_point (rotated_point x y 2 3).1 (rotated_point x y 2 3).2

theorem find_x_minus_y (x y : ℝ) (h1 : transformed_point x y = (4, -1)) : x - y = 3 := 
by 
  sorry

end find_x_minus_y_l1362_136240


namespace gambler_largest_amount_proof_l1362_136212

noncomputable def largest_amount_received_back (initial_amount : ℝ) (value_25 : ℝ) (value_75 : ℝ) (value_250 : ℝ) 
                                               (total_lost_chips : ℝ) (coef_25_75_lost : ℝ) (coef_75_250_lost : ℝ) : ℝ :=
    initial_amount - (
    coef_25_75_lost * (total_lost_chips / (coef_25_75_lost + 1 + 1)) * value_25 +
    (total_lost_chips / (coef_25_75_lost + 1 + 1)) * value_75 +
    coef_75_250_lost * (total_lost_chips / (coef_25_75_lost + 1 + 1)) * value_250)

theorem gambler_largest_amount_proof :
    let initial_amount := 15000
    let value_25 := 25
    let value_75 := 75
    let value_250 := 250
    let total_lost_chips := 40
    let coef_25_75_lost := 2 -- number of lost $25 chips is twice the number of lost $75 chips
    let coef_75_250_lost := 2 -- number of lost $250 chips is twice the number of lost $75 chips
    largest_amount_received_back initial_amount value_25 value_75 value_250 total_lost_chips coef_25_75_lost coef_75_250_lost = 10000 :=
by {
    sorry
}

end gambler_largest_amount_proof_l1362_136212


namespace max_gcd_13n_plus_4_8n_plus_3_l1362_136296

theorem max_gcd_13n_plus_4_8n_plus_3 : 
  ∀ n : ℕ, n > 0 → ∃ d : ℕ, d = 7 ∧ ∀ k : ℕ, k = gcd (13 * n + 4) (8 * n + 3) → k ≤ d :=
by
  sorry

end max_gcd_13n_plus_4_8n_plus_3_l1362_136296


namespace boat_speed_in_still_water_l1362_136209

theorem boat_speed_in_still_water (b s : ℝ) 
  (h1 : b + s = 11) 
  (h2 : b - s = 3) : b = 7 :=
by
  sorry

end boat_speed_in_still_water_l1362_136209


namespace bus_carrying_capacity_l1362_136255

variables (C : ℝ)

theorem bus_carrying_capacity (h1 : ∀ x : ℝ, x = (3 / 5) * C) 
                              (h2 : ∀ y : ℝ, y = 50 - 18)
                              (h3 : ∀ z : ℝ, x + y = C) : C = 80 :=
by
  sorry

end bus_carrying_capacity_l1362_136255


namespace cost_price_of_article_l1362_136250

theorem cost_price_of_article (C MP SP : ℝ) (h1 : MP = 62.5) (h2 : SP = 0.95 * MP) (h3 : SP = 1.25 * C) :
  C = 47.5 :=
sorry

end cost_price_of_article_l1362_136250


namespace t_over_s_possible_values_l1362_136285

-- Define the initial conditions
variables (n : ℕ) (h : n ≥ 3)

-- The theorem statement
theorem t_over_s_possible_values (s t : ℕ) (h_s : s > 0) (h_t : t > 0) : 
  (∃ r : ℚ, r = t / s ∧ 1 ≤ r ∧ r < (n - 1)) :=
sorry

end t_over_s_possible_values_l1362_136285


namespace solve_MQ_above_A_l1362_136264

-- Definitions of the given conditions
def ABCD_side := 8
def MNPQ_length := 16
def MNPQ_width := 8
def area_outer_inner_ratio := 1 / 3

-- Definition to prove
def length_MQ_above_A := 8 / 3

-- The area calculations
def area_MNPQ := MNPQ_length * MNPQ_width
def area_ABCD := ABCD_side * ABCD_side
def area_outer := (area_outer_inner_ratio * area_MNPQ)
def MQ_above_A_calculated := area_outer / MNPQ_length

theorem solve_MQ_above_A :
  MQ_above_A_calculated = length_MQ_above_A := by sorry

end solve_MQ_above_A_l1362_136264


namespace solve_quadratic_eq_1_solve_quadratic_eq_2_l1362_136251

-- Proof for Equation 1
theorem solve_quadratic_eq_1 : ∀ x : ℝ, x^2 - 4 * x + 1 = 0 ↔ (x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) :=
by sorry

-- Proof for Equation 2
theorem solve_quadratic_eq_2 : ∀ x : ℝ, 5 * x - 2 = (2 - 5 * x) * (3 * x + 4) ↔ (x = 2 / 5 ∨ x = -5 / 3) :=
by sorry

end solve_quadratic_eq_1_solve_quadratic_eq_2_l1362_136251


namespace exists_integer_lt_sqrt_10_l1362_136283

theorem exists_integer_lt_sqrt_10 : ∃ k : ℤ, k < Real.sqrt 10 := by
  have h_sqrt_bounds : 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 := by
    -- Proof involving basic properties and calculations
    sorry
  exact ⟨3, h_sqrt_bounds.left⟩

end exists_integer_lt_sqrt_10_l1362_136283


namespace evaluate_f_difference_l1362_136213

def f (x : ℤ) : ℤ := x^6 + 3 * x^4 - 4 * x^3 + x^2 + 2 * x

theorem evaluate_f_difference : f 3 - f (-3) = -204 := by
  sorry

end evaluate_f_difference_l1362_136213


namespace original_triangle_area_l1362_136214

theorem original_triangle_area (area_of_new_triangle : ℝ) (side_length_ratio : ℝ) (quadrupled : side_length_ratio = 4) (new_area : area_of_new_triangle = 128) : 
  (area_of_new_triangle / side_length_ratio ^ 2) = 8 := by
  sorry

end original_triangle_area_l1362_136214


namespace time_to_cross_l1362_136244

noncomputable def length_first_train : ℝ := 210
noncomputable def speed_first_train : ℝ := 120 * 1000 / 3600 -- Convert to m/s
noncomputable def length_second_train : ℝ := 290.04
noncomputable def speed_second_train : ℝ := 80 * 1000 / 3600 -- Convert to m/s

noncomputable def relative_speed := speed_first_train + speed_second_train
noncomputable def total_length := length_first_train + length_second_train
noncomputable def crossing_time := total_length / relative_speed

theorem time_to_cross : crossing_time = 9 := by
  let length_first_train : ℝ := 210
  let speed_first_train : ℝ := 120 * 1000 / 3600 -- Convert to m/s
  let length_second_train : ℝ := 290.04
  let speed_second_train : ℝ := 80 * 1000 / 3600 -- Convert to m/s

  let relative_speed := speed_first_train + speed_second_train
  let total_length := length_first_train + length_second_train
  let crossing_time := total_length / relative_speed

  show crossing_time = 9
  sorry

end time_to_cross_l1362_136244


namespace knights_wins_33_l1362_136247

def sharks_wins : ℕ := sorry
def falcons_wins : ℕ := sorry
def knights_wins : ℕ := sorry
def wolves_wins : ℕ := sorry
def dragons_wins : ℕ := 38 -- Dragons won the most games

-- Condition 1: The Sharks won more games than the Falcons.
axiom sharks_won_more_than_falcons : sharks_wins > falcons_wins

-- Condition 2: The Knights won more games than the Wolves, but fewer than the Dragons.
axiom knights_won_more_than_wolves : knights_wins > wolves_wins
axiom knights_won_less_than_dragons : knights_wins < dragons_wins

-- Condition 3: The Wolves won more than 22 games.
axiom wolves_won_more_than_22 : wolves_wins > 22

-- The possible wins are 24, 27, 33, 36, and 38 and the dragons win 38 (already accounted in dragons_wins)

-- Prove that the Knights won 33 games.
theorem knights_wins_33 : knights_wins = 33 :=
sorry -- proof goes here

end knights_wins_33_l1362_136247


namespace percentage_big_bottles_sold_l1362_136222

-- Definitions of conditions
def total_small_bottles : ℕ := 6000
def total_big_bottles : ℕ := 14000
def small_bottles_sold_percentage : ℕ := 20
def total_bottles_remaining : ℕ := 15580

-- Theorem statement
theorem percentage_big_bottles_sold : 
  let small_bottles_sold := (small_bottles_sold_percentage * total_small_bottles) / 100
  let small_bottles_remaining := total_small_bottles - small_bottles_sold
  let big_bottles_remaining := total_bottles_remaining - small_bottles_remaining
  let big_bottles_sold := total_big_bottles - big_bottles_remaining
  (100 * big_bottles_sold) / total_big_bottles = 23 := 
by
  sorry

end percentage_big_bottles_sold_l1362_136222


namespace find_x_l1362_136293

theorem find_x (x : ℚ) (h : (3 + 1 / (2 + 1 / (3 + 3 / (4 + x)))) = 225 / 68) : 
  x = -50 / 19 := 
sorry

end find_x_l1362_136293


namespace log_six_two_l1362_136232

noncomputable def log_six (x : ℝ) : ℝ := Real.log x / Real.log 6

theorem log_six_two (a : ℝ) (h : log_six 3 = a) : log_six 2 = 1 - a :=
by
  sorry

end log_six_two_l1362_136232


namespace wall_area_l1362_136227

theorem wall_area (width : ℝ) (height : ℝ) (h1 : width = 2) (h2 : height = 4) : width * height = 8 := by
  sorry

end wall_area_l1362_136227


namespace hyperbola_range_of_k_l1362_136216

theorem hyperbola_range_of_k (x y k : ℝ) :
  (∃ x y : ℝ, (x^2 / (1 - 2 * k) - y^2 / (k - 2) = 1) ∧ (1 - 2 * k < 0) ∧ (k - 2 < 0)) →
  (1 / 2 < k ∧ k < 2) :=
by 
  sorry

end hyperbola_range_of_k_l1362_136216


namespace red_balls_approximation_l1362_136206

def total_balls : ℕ := 50
def red_ball_probability : ℚ := 7 / 10

theorem red_balls_approximation (r : ℕ)
  (h1 : total_balls = 50)
  (h2 : red_ball_probability = 0.7) :
  r = 35 := by
  sorry

end red_balls_approximation_l1362_136206


namespace cannot_be_expressed_as_difference_of_squares_can_be_expressed_as_difference_of_squares_2004_can_be_expressed_as_difference_of_squares_2005_can_be_expressed_as_difference_of_squares_2007_l1362_136220

theorem cannot_be_expressed_as_difference_of_squares (a b : ℤ) (h : 2006 = a^2 - b^2) : False := sorry

theorem can_be_expressed_as_difference_of_squares_2004 : ∃ (a b : ℤ), 2004 = a^2 - b^2 := by
  use 502, 500
  norm_num

theorem can_be_expressed_as_difference_of_squares_2005 : ∃ (a b : ℤ), 2005 = a^2 - b^2 := by
  use 1003, 1002
  norm_num

theorem can_be_expressed_as_difference_of_squares_2007 : ∃ (a b : ℤ), 2007 = a^2 - b^2 := by
  use 1004, 1003
  norm_num

end cannot_be_expressed_as_difference_of_squares_can_be_expressed_as_difference_of_squares_2004_can_be_expressed_as_difference_of_squares_2005_can_be_expressed_as_difference_of_squares_2007_l1362_136220


namespace counterexample_to_proposition_l1362_136211

theorem counterexample_to_proposition (a b : ℝ) (ha : a = 1) (hb : b = -1) :
  a > b ∧ ¬ (1 / a < 1 / b) :=
by
  sorry

end counterexample_to_proposition_l1362_136211


namespace cost_of_scooter_l1362_136226

-- Given conditions
variables (M T : ℕ)
axiom h1 : T = M + 4
axiom h2 : T = 15

-- Proof goal: The cost of the scooter is $26
theorem cost_of_scooter : M + T = 26 :=
by sorry

end cost_of_scooter_l1362_136226


namespace S_5_is_121_l1362_136236

-- Definitions of the sequence and its terms
def S : ℕ → ℕ := sorry  -- Define S_n
def a : ℕ → ℕ := sorry  -- Define a_n

-- Conditions
axiom S_2 : S 2 = 4
axiom recurrence_relation : ∀ n : ℕ, S (n + 1) = 1 + 2 * S n

-- Proof that S_5 = 121 given the conditions
theorem S_5_is_121 : S 5 = 121 := by
  sorry

end S_5_is_121_l1362_136236


namespace garden_area_increase_l1362_136246

-- Define the dimensions and perimeter of the rectangular garden
def length_rect : ℕ := 30
def width_rect : ℕ := 12
def area_rect : ℕ := length_rect * width_rect

def perimeter_rect : ℕ := 2 * (length_rect + width_rect)

-- Define the side length and area of the new square garden
def side_square : ℕ := perimeter_rect / 4
def area_square : ℕ := side_square * side_square

-- Define the increase in area
def increase_in_area : ℕ := area_square - area_rect

-- Prove the increase in area is 81 square feet
theorem garden_area_increase : increase_in_area = 81 := by
  sorry

end garden_area_increase_l1362_136246


namespace latte_cost_l1362_136234

theorem latte_cost (L : ℝ) 
  (latte_days : ℝ := 5)
  (iced_coffee_cost : ℝ := 2)
  (iced_coffee_days : ℝ := 3)
  (weeks_in_year : ℝ := 52)
  (spending_reduction : ℝ := 0.25)
  (savings : ℝ := 338) 
  (current_annual_spending : ℝ := 4 * savings)
  (weekly_spending : ℝ := latte_days * L + iced_coffee_days * iced_coffee_cost)
  (annual_spending_eq : weeks_in_year * weekly_spending = current_annual_spending) :
  L = 4 := 
sorry

end latte_cost_l1362_136234


namespace ratio_revenue_l1362_136219

variable (N D J : ℝ)

theorem ratio_revenue (h1 : J = N / 3) (h2 : D = 2.5 * (N + J) / 2) : N / D = 3 / 5 := by
  sorry

end ratio_revenue_l1362_136219


namespace possible_b4b7_products_l1362_136291

theorem possible_b4b7_products (b : ℕ → ℤ) (d : ℤ)
  (h_arith_sequence : ∀ n, b (n + 1) = b n + d)
  (h_increasing : ∀ n, b (n + 1) > b n)
  (h_product_21 : b 5 * b 6 = 21) :
  b 4 * b 7 = -779 ∨ b 4 * b 7 = 21 :=
by
  sorry

end possible_b4b7_products_l1362_136291


namespace triangle_area_PQR_l1362_136275

def point := (ℝ × ℝ)

def P : point := (2, 3)
def Q : point := (7, 3)
def R : point := (4, 10)

noncomputable def triangle_area (A B C : point) : ℝ :=
  (1/2) * ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_PQR : triangle_area P Q R = 17.5 :=
  sorry

end triangle_area_PQR_l1362_136275


namespace solve_for_x_l1362_136288

theorem solve_for_x : 
  ∀ x : ℚ, x + 5/6 = 7/18 - 2/9 → x = -2/3 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l1362_136288


namespace integer_solutions_to_inequality_l1362_136233

noncomputable def count_integer_solutions (n : ℕ) : ℕ :=
1 + 2 * n^2 + 2 * n

theorem integer_solutions_to_inequality (n : ℕ) :
  ∃ (count : ℕ), count = count_integer_solutions n ∧ 
  ∀ (x y : ℤ), |x| + |y| ≤ n → (∃ (k : ℕ), k = count) :=
by
  sorry

end integer_solutions_to_inequality_l1362_136233


namespace range_of_function_l1362_136202

theorem range_of_function : 
  ∀ y : ℝ, ∃ x : ℝ, y = 4^x + 2^x - 3 ↔ y > -3 :=
by
  sorry

end range_of_function_l1362_136202


namespace number_of_dogs_l1362_136289

variable {C D : ℕ}

def ratio_of_dogs_to_cats (D C : ℕ) : Prop := D = (15/7) * C

def ratio_after_additional_cats (D C : ℕ) : Prop :=
  D = 15 * (C + 8) / 11

theorem number_of_dogs (h1 : ratio_of_dogs_to_cats D C) (h2 : ratio_after_additional_cats D C) :
  D = 30 :=
by
  sorry

end number_of_dogs_l1362_136289


namespace conic_eccentricity_l1362_136256

theorem conic_eccentricity (m : ℝ) (h : 0 < -m) (h2 : (Real.sqrt (1 + (-1 / m))) = 2) : m = -1/3 := 
by
  -- Proof can be added here
  sorry

end conic_eccentricity_l1362_136256


namespace polynomial_sequence_finite_functions_l1362_136253

theorem polynomial_sequence_finite_functions (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) : 
  (1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1) := 
by
  sorry

end polynomial_sequence_finite_functions_l1362_136253


namespace prob_rain_next_day_given_today_rain_l1362_136272

variable (P_rain : ℝ) (P_rain_2_days : ℝ)
variable (p_given_rain : ℝ)

-- Given conditions
def condition_P_rain : Prop := P_rain = 1/3
def condition_P_rain_2_days : Prop := P_rain_2_days = 1/5

-- The question to prove
theorem prob_rain_next_day_given_today_rain (h1 : condition_P_rain P_rain) (h2 : condition_P_rain_2_days P_rain_2_days) :
  p_given_rain = 3/5 :=
by
  sorry

end prob_rain_next_day_given_today_rain_l1362_136272


namespace valeries_thank_you_cards_l1362_136287

variables (T R J B : ℕ)

theorem valeries_thank_you_cards :
  B = 2 →
  R = B + 3 →
  J = 2 * R →
  T + (B + 1) + R + J = 21 →
  T = 3 :=
by
  intros hB hR hJ hTotal
  sorry

end valeries_thank_you_cards_l1362_136287


namespace line_equation_l1362_136248

-- Definitions according to the conditions
def point_P := (3, 4)
def slope_angle_l := 90

-- Statement of the theorem to prove
theorem line_equation (l : ℝ → ℝ) (h1 : l point_P.1 = point_P.2) (h2 : slope_angle_l = 90) :
  ∃ k : ℝ, k = 3 ∧ ∀ x, l x = 3 - x :=
sorry

end line_equation_l1362_136248


namespace expected_winnings_is_350_l1362_136266

noncomputable def expected_winnings : ℝ :=
  (1 / 8) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

theorem expected_winnings_is_350 :
  expected_winnings = 3.5 :=
by sorry

end expected_winnings_is_350_l1362_136266


namespace sin_double_angle_pi_six_l1362_136263

theorem sin_double_angle_pi_six (α : ℝ)
  (h : 2 * Real.sin α = 1 + 2 * Real.sqrt 3 * Real.cos α) :
  Real.sin (2 * α - Real.pi / 6) = 7 / 8 :=
sorry

end sin_double_angle_pi_six_l1362_136263


namespace sum_of_squares_l1362_136245

theorem sum_of_squares (x y z : ℝ)
  (h1 : (x + y + z) / 3 = 10)
  (h2 : (xyz)^(1/3) = 6)
  (h3 : 3 / ((1/x) + (1/y) + (1/z)) = 4) : 
  x^2 + y^2 + z^2 = 576 := 
by
  sorry

end sum_of_squares_l1362_136245


namespace two_b_squared_eq_a_squared_plus_c_squared_l1362_136239

theorem two_b_squared_eq_a_squared_plus_c_squared (a b c : ℝ) (h : 1 / (a + b) + 1 / (b + c) = 2 / (c + a)) : 
  2 * b^2 = a^2 + c^2 := 
sorry

end two_b_squared_eq_a_squared_plus_c_squared_l1362_136239


namespace fraction_sent_afternoon_l1362_136252

-- Defining the problem conditions
def total_fliers : ℕ := 1000
def fliers_sent_morning : ℕ := total_fliers * 1/5
def fliers_left_afternoon : ℕ := total_fliers - fliers_sent_morning
def fliers_left_next_day : ℕ := 600
def fliers_sent_afternoon : ℕ := fliers_left_afternoon - fliers_left_next_day

-- Proving the fraction of fliers sent in the afternoon
theorem fraction_sent_afternoon : (fliers_sent_afternoon : ℚ) / fliers_left_afternoon = 1/4 :=
by
  -- proof goes here
  sorry

end fraction_sent_afternoon_l1362_136252


namespace julia_fascinating_last_digits_l1362_136249

theorem julia_fascinating_last_digits : ∃ n : ℕ, n = 10 ∧ (∀ x : ℕ, (∃ y : ℕ, x = 10 * y) → x % 10 < 10) :=
by
  sorry

end julia_fascinating_last_digits_l1362_136249


namespace circles_intersect_l1362_136278

theorem circles_intersect (t : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2 * t * x + t^2 - 4 = 0 ∧ x^2 + y^2 + 2 * x - 4 * t * y + 4 * t^2 - 8 = 0) ↔ 
  (-12 / 5 < t ∧ t < -2 / 5) ∨ (0 < t ∧ t < 2) :=
sorry

end circles_intersect_l1362_136278


namespace range_x_minus_y_l1362_136225

-- Definition of the curve in polar coordinates
def curve_polar (rho theta : ℝ) : Prop :=
  rho = 4 * Real.cos theta + 2 * Real.sin theta

-- Conversion to rectangular coordinates
noncomputable def curve_rectangular (x y : ℝ) : Prop :=
  x^2 + y^2 = 4 * x + 2 * y

-- The final Lean 4 statement
theorem range_x_minus_y (x y : ℝ) (h : curve_rectangular x y) :
  1 - Real.sqrt 10 ≤ x - y ∧ x - y ≤ 1 + Real.sqrt 10 :=
sorry

end range_x_minus_y_l1362_136225


namespace number_of_factors_n_l1362_136229

-- Defining the value of n with its prime factorization
def n : ℕ := 2^5 * 3^9 * 5^5

-- Theorem stating the number of natural-number factors of n
theorem number_of_factors_n : 
  (Nat.divisors n).card = 360 := by
  -- Proof is omitted
  sorry

end number_of_factors_n_l1362_136229


namespace plus_signs_count_l1362_136201

theorem plus_signs_count (total_symbols : ℕ) (n : ℕ) (m : ℕ)
    (h1 : total_symbols = 23)
    (h2 : ∀ (s : Finset ℕ), s.card = 10 → (∃ x ∈ s, x = n))
    (h3 : ∀ (s : Finset ℕ), s.card = 15 → (∃ x ∈ s, x = m)) :
    n = 14 := by
  sorry

end plus_signs_count_l1362_136201


namespace greatest_teams_l1362_136268

-- Define the number of girls and boys as constants
def numGirls : ℕ := 40
def numBoys : ℕ := 32

-- Define the greatest number of teams possible with equal number of girls and boys as teams.
theorem greatest_teams : Nat.gcd numGirls numBoys = 8 := sorry

end greatest_teams_l1362_136268


namespace cdf_from_pdf_l1362_136292

noncomputable def pdf (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if 0 < x ∧ x ≤ Real.pi / 2 then Real.cos x
  else 0

noncomputable def cdf (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if 0 < x ∧ x ≤ Real.pi / 2 then Real.sin x
  else 1

theorem cdf_from_pdf (x : ℝ) : 
  ∀ x : ℝ, cdf x = 
    if x ≤ 0 then 0
    else if 0 < x ∧ x ≤ Real.pi / 2 then Real.sin x
    else 1 :=
by
  sorry

end cdf_from_pdf_l1362_136292


namespace largest_square_factor_of_1800_l1362_136204

theorem largest_square_factor_of_1800 : 
  ∃ n, n^2 ∣ 1800 ∧ ∀ m, m^2 ∣ 1800 → m^2 ≤ n^2 :=
sorry

end largest_square_factor_of_1800_l1362_136204


namespace average_salary_of_all_employees_l1362_136290

theorem average_salary_of_all_employees 
    (avg_salary_officers : ℝ)
    (avg_salary_non_officers : ℝ)
    (num_officers : ℕ)
    (num_non_officers : ℕ)
    (h1 : avg_salary_officers = 450)
    (h2 : avg_salary_non_officers = 110)
    (h3 : num_officers = 15)
    (h4 : num_non_officers = 495) :
    (avg_salary_officers * num_officers + avg_salary_non_officers * num_non_officers)
    / (num_officers + num_non_officers) = 120 := by
  sorry

end average_salary_of_all_employees_l1362_136290


namespace sum_due_is_correct_l1362_136237

-- Definitions of the given conditions
def BD : ℝ := 78
def TD : ℝ := 66

-- Definition of the sum due (S)
noncomputable def S : ℝ := (TD^2) / (BD - TD) + TD

-- The theorem to be proved
theorem sum_due_is_correct : S = 429 := by
  sorry

end sum_due_is_correct_l1362_136237


namespace sum_of_abs_coeffs_l1362_136260

theorem sum_of_abs_coeffs (a : ℕ → ℤ) :
  (∀ x : ℤ, (1 - x)^5 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5) →
  |a 0| + |a 1| + |a 2| + |a 3| + |a 4| + |a 5| = 32 := 
by
  sorry

end sum_of_abs_coeffs_l1362_136260


namespace Polly_tweets_l1362_136241

theorem Polly_tweets :
  let HappyTweets := 18 * 50
  let HungryTweets := 4 * 35
  let WatchingReflectionTweets := 45 * 30
  let SadTweets := 6 * 20
  let PlayingWithToysTweets := 25 * 75
  HappyTweets + HungryTweets + WatchingReflectionTweets + SadTweets + PlayingWithToysTweets = 4385 :=
by
  sorry

end Polly_tweets_l1362_136241


namespace starting_number_l1362_136217

theorem starting_number (x : ℝ) (h : (x + 26) / 2 = 19) : x = 12 :=
by
  sorry

end starting_number_l1362_136217


namespace max_months_with_5_sundays_l1362_136243

theorem max_months_with_5_sundays (months : ℕ) (days_in_year : ℕ) (extra_sundays : ℕ) :
  months = 12 ∧ (days_in_year = 365 ∨ days_in_year = 366) ∧ extra_sundays = days_in_year % 7
  → ∃ max_months_with_5_sundays, max_months_with_5_sundays = 5 := 
by
  sorry

end max_months_with_5_sundays_l1362_136243
