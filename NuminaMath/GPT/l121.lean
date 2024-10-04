import Mathlib

namespace find_m_l121_121003

theorem find_m (m : ℝ) (x : ℝ) (h : 2*x + m = 1) (hx : x = -1) : m = 3 := 
by
  rw [hx] at h
  linarith

end find_m_l121_121003


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l121_121381

theorem number_of_subsets_with_at_least_four_adjacent_chairs (chairs : Finset ℕ) (h_chairs : chairs.card = 12)
  (h_circular : ∀ (s : Finset ℕ), s ⊆ chairs → (s.card ≥ 4 → ∃ t : Finset ℕ, t ⊆ chairs ∧ t.card = 4 ∧ ∀ i j ∈ t, abs (i - j) ≤ 1)) :
  ∃ (subsets : Finset (Finset ℕ)), (∀ s ∈ subsets, s ⊆ chairs ∧ s.card ≥ 4) ∧ subsets.card = 169 :=
sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l121_121381


namespace red_balls_in_bag_l121_121583

/-- Given the conditions of the ball distribution in the bag,
we need to prove the number of red balls is 9. -/
theorem red_balls_in_bag (total_balls white_balls green_balls yellow_balls purple_balls : ℕ)
  (prob_neither_red_nor_purple : ℝ) (h_total : total_balls = 100)
  (h_white : white_balls = 50) (h_green : green_balls = 30)
  (h_yellow : yellow_balls = 8) (h_purple : purple_balls = 3)
  (h_prob : prob_neither_red_nor_purple = 0.88) :
  ∃ R : ℕ, (total_balls = white_balls + green_balls + yellow_balls + purple_balls + R) ∧ R = 9 :=
by {
  sorry
}

end red_balls_in_bag_l121_121583


namespace range_of_m_l121_121833

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x + m ≠ 0
def q (m : ℝ) : Prop := ∃ y : ℝ, ∀ x : ℝ, (x^2)/(m-1) + y^2 = 1
def not_p (m : ℝ) : Prop := ¬ (p m)
def p_and_q (m : ℝ) : Prop := (p m) ∧ (q m)

theorem range_of_m (m : ℝ) : (¬ (not_p m) ∧ ¬ (p_and_q m)) → 1 < m ∧ m ≤ 2 :=
sorry

end range_of_m_l121_121833


namespace cosine_double_angle_l121_121685

theorem cosine_double_angle
  (θ : ℝ)
  (cos_θ : ℝ)
  (h : cos_θ = 3 / 5) :
  2 * cos_θ^2 - 1 = -7 / 25 :=
by
  sorry

end cosine_double_angle_l121_121685


namespace part_1_property_part_2_property_part_3_geometric_l121_121040

-- Defining properties
def prop1 (a : ℕ → ℕ) (i j m: ℕ) : Prop := i > j ∧ (a i)^2 / (a j) = a m
def prop2 (a : ℕ → ℕ) (n k l: ℕ) : Prop := n ≥ 3 ∧ k > l ∧ (a n) = (a k)^2 / (a l)

-- Part I: Sequence {a_n = n} check for property 1
theorem part_1_property (a : ℕ → ℕ) (h : ∀ n, a n = n) : ¬∃ i j m, prop1 a i j m := by
  sorry

-- Part II: Sequence {a_n = 2^(n-1)} check for property 1 and 2
theorem part_2_property (a : ℕ → ℕ) (h : ∀ n, a n = 2^(n-1)) : 
  (∀ i j, ∃ m, prop1 a i j m) ∧ (∀ n k l, prop2 a n k l) := by
  sorry

-- Part III: Increasing sequence that satisfies both properties is a geometric sequence
theorem part_3_geometric (a : ℕ → ℕ) (h_inc : ∀ n m, n < m → a n < a m) 
  (h_prop1 : ∀ i j, i > j → ∃ m, prop1 a i j m)
  (h_prop2 : ∀ n, n ≥ 3 → ∃ k l, k > l ∧ (a n) = (a k)^2 / (a l)) : 
  ∃ r, ∀ n, a (n + 1) = r * a n := by
  sorry

end part_1_property_part_2_property_part_3_geometric_l121_121040


namespace dot_product_to_linear_form_l121_121519

noncomputable def proof_problem (r a : ℝ × ℝ) (m : ℝ) : Prop :=
  let A := a.1
  let B := a.2
  let C := -m
  (r.1 * a.1 + r.2 * a.2 = m) → (A * r.1 + B * r.2 + C = 0)

-- The theorem statement
theorem dot_product_to_linear_form (r a : ℝ × ℝ) (m : ℝ) :
  proof_problem r a m :=
sorry

end dot_product_to_linear_form_l121_121519


namespace total_students_in_school_l121_121804

-- Definitions and conditions
def number_of_blind_students (B : ℕ) : Prop := ∃ B, 3 * B = 180
def number_of_other_disabilities (O : ℕ) (B : ℕ) : Prop := O = 2 * B
def total_students (T : ℕ) (D : ℕ) (B : ℕ) (O : ℕ) : Prop := T = D + B + O

theorem total_students_in_school : 
  ∃ (T B O : ℕ), number_of_blind_students B ∧ 
                 number_of_other_disabilities O B ∧ 
                 total_students T 180 B O ∧ 
                 T = 360 :=
by
  sorry

end total_students_in_school_l121_121804


namespace original_price_of_cycle_l121_121798

theorem original_price_of_cycle (SP : ℝ) (P : ℝ) (loss_percent : ℝ) 
  (h_loss : loss_percent = 18) 
  (h_SP : SP = 1148) 
  (h_eq : SP = (1 - loss_percent / 100) * P) : 
  P = 1400 := 
by 
  sorry

end original_price_of_cycle_l121_121798


namespace double_angle_cosine_l121_121677

theorem double_angle_cosine (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end double_angle_cosine_l121_121677


namespace hens_ratio_l121_121515

theorem hens_ratio
  (total_chickens : ℕ)
  (fraction_roosters : ℚ)
  (chickens_not_laying : ℕ)
  (h : total_chickens = 80)
  (fr : fraction_roosters = 1/4)
  (cnl : chickens_not_laying = 35) :
  (total_chickens * (1 - fraction_roosters) - chickens_not_laying) / (total_chickens * (1 - fraction_roosters)) = 5 / 12 :=
by
  sorry

end hens_ratio_l121_121515


namespace book_arrangement_ways_l121_121325

open Nat

theorem book_arrangement_ways : 
  let m := 4  -- Number of math books
  let h := 6  -- Number of history books
  -- Number of ways to place a math book on both ends:
  let ways_ends := m * (m - 1)  -- Choices for the left end and right end
  -- Number of ways to arrange the remaining books:
  let ways_entities := 2!  -- Arrangements of the remaining entities
  -- Number of ways to arrange history books within the block:
  let arrange_history := factorial h
  -- Total arrangements
  let total_ways := ways_ends * ways_entities * arrange_history
  total_ways = 17280 := sorry

end book_arrangement_ways_l121_121325


namespace solution_of_r_and_s_l121_121192

theorem solution_of_r_and_s 
  (r s : ℝ)
  (hr : (r-5)*(r+5) = 25*r - 125)
  (hs : (s-5)*(s+5) = 25*s - 125)
  (h_distinct : r ≠ s)
  (h_order : r > s) : 
  r - s = 15 := by
sorry

end solution_of_r_and_s_l121_121192


namespace gerald_remaining_pfennigs_l121_121462

-- Definitions of Gerald's initial money and the costs of items
def farthings : Nat := 54
def groats : Nat := 8
def florins : Nat := 17
def meat_pie_cost : Nat := 120
def sausage_roll_cost : Nat := 75

-- Conversion rates
def farthings_to_pfennigs (f : Nat) : Nat := f / 6
def groats_to_pfennigs (g : Nat) : Nat := g * 4
def florins_to_pfennigs (f : Nat) : Nat := f * 40

-- Total pfennigs Gerald has
def total_pfennigs : Nat :=
  farthings_to_pfennigs farthings + groats_to_pfennigs groats + florins_to_pfennigs florins

-- Total cost of both items
def total_cost : Nat := meat_pie_cost + sausage_roll_cost

-- Gerald's remaining pfennigs after purchase
def remaining_pfennigs : Nat := total_pfennigs - total_cost

theorem gerald_remaining_pfennigs :
  remaining_pfennigs = 526 :=
by
  sorry

end gerald_remaining_pfennigs_l121_121462


namespace barrels_in_one_ton_l121_121961

-- Definitions (conditions)
def barrel_weight : ℕ := 10 -- in kilograms
def ton_in_kilograms : ℕ := 1000

-- Theorem Statement
theorem barrels_in_one_ton : ton_in_kilograms / barrel_weight = 100 :=
by
  sorry

end barrels_in_one_ton_l121_121961


namespace solve_x_squared_eq_four_x_l121_121763

theorem solve_x_squared_eq_four_x : {x : ℝ | x^2 = 4*x} = {0, 4} := 
sorry

end solve_x_squared_eq_four_x_l121_121763


namespace magic_king_total_episodes_l121_121931

theorem magic_king_total_episodes
  (total_seasons : ℕ)
  (first_half_seasons : ℕ)
  (second_half_seasons : ℕ)
  (episodes_first_half : ℕ)
  (episodes_second_half : ℕ)
  (h1 : total_seasons = 10)
  (h2 : first_half_seasons = total_seasons / 2)
  (h3 : second_half_seasons = total_seasons / 2)
  (h4 : episodes_first_half = 20)
  (h5 : episodes_second_half = 25)
  : (first_half_seasons * episodes_first_half + second_half_seasons * episodes_second_half) = 225 :=
by
  sorry

end magic_king_total_episodes_l121_121931


namespace work_completion_time_l121_121246

theorem work_completion_time (x : ℝ) (a_work_rate b_work_rate combined_work_rate : ℝ) :
  a_work_rate = 1 / 15 ∧
  b_work_rate = 1 / 20 ∧
  combined_work_rate = 1 / 7.2 ∧
  a_work_rate + b_work_rate + (1 / x) = combined_work_rate → 
  x = 45 := by
  sorry

end work_completion_time_l121_121246


namespace katrina_cookies_left_l121_121344

theorem katrina_cookies_left (initial_cookies morning_cookies_sold lunch_cookies_sold afternoon_cookies_sold : ℕ)
  (h1 : initial_cookies = 120)
  (h2 : morning_cookies_sold = 36)
  (h3 : lunch_cookies_sold = 57)
  (h4 : afternoon_cookies_sold = 16) :
  initial_cookies - (morning_cookies_sold + lunch_cookies_sold + afternoon_cookies_sold) = 11 := 
by 
  sorry

end katrina_cookies_left_l121_121344


namespace repeating_decimal_addition_l121_121617

def repeating_decimal_45 := (45 / 99 : ℚ)
def repeating_decimal_36 := (36 / 99 : ℚ)

theorem repeating_decimal_addition :
  repeating_decimal_45 + repeating_decimal_36 = 9 / 11 :=
by
  sorry

end repeating_decimal_addition_l121_121617


namespace ratio_third_to_first_second_l121_121116

-- Define the times spent on each step
def time_first_step : ℕ := 30
def time_second_step : ℕ := time_first_step / 2
def time_total : ℕ := 90
def time_third_step : ℕ := time_total - (time_first_step + time_second_step)

-- Define the combined time for the first two steps
def time_combined_first_second : ℕ := time_first_step + time_second_step

-- The goal is to prove that the ratio of the time spent on the third step to the combined time spent on the first and second steps is 1:1
theorem ratio_third_to_first_second : time_third_step = time_combined_first_second :=
by
  -- Proof goes here
  sorry

end ratio_third_to_first_second_l121_121116


namespace cos_double_angle_l121_121666

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3 / 5) : Real.cos (2 * theta) = -7 / 25 :=
by
  sorry

end cos_double_angle_l121_121666


namespace fraction_of_defective_engines_l121_121072

theorem fraction_of_defective_engines
  (total_batches : ℕ)
  (engines_per_batch : ℕ)
  (non_defective_engines : ℕ)
  (H1 : total_batches = 5)
  (H2 : engines_per_batch = 80)
  (H3 : non_defective_engines = 300)
  : (total_batches * engines_per_batch - non_defective_engines) / (total_batches * engines_per_batch) = 1 / 4 :=
by
  -- Proof goes here.
  sorry

end fraction_of_defective_engines_l121_121072


namespace factor_expr_l121_121609

def expr1 (x : ℝ) := 16 * x^6 + 49 * x^4 - 9
def expr2 (x : ℝ) := 4 * x^6 - 14 * x^4 - 9

theorem factor_expr (x : ℝ) :
  (expr1 x - expr2 x) = 3 * x^4 * (4 * x^2 + 21) := 
by
  sorry

end factor_expr_l121_121609


namespace main_theorem_l121_121107

-- Definitions based on conditions
variables (A P H M E C : ℕ) 
-- Thickness of an algebra book
def x := 1
-- Thickness of a history book (twice that of algebra)
def history_thickness := 2 * x
-- Length of shelf filled by books
def z := A * x

-- Condition equations based on shelf length equivalences
def equation1 := A = P
def equation2 := 2 * H * x = M * x
def equation3 := E * x + C * history_thickness = z

-- Prove the relationship
theorem main_theorem : C = (M * (A - E)) / (2 * A * H) :=
by
  sorry

end main_theorem_l121_121107


namespace original_cost_of_remaining_shirt_l121_121457

theorem original_cost_of_remaining_shirt 
  (total_original_cost : ℝ) 
  (shirts_on_discount : ℕ) 
  (original_cost_per_discounted_shirt : ℝ) 
  (discount : ℝ) 
  (current_total_cost : ℝ) : 
  total_original_cost = 100 → 
  shirts_on_discount = 3 → 
  original_cost_per_discounted_shirt = 25 → 
  discount = 0.4 → 
  current_total_cost = 85 → 
  ∃ (remaining_shirts : ℕ) (original_cost_per_remaining_shirt : ℝ), 
    remaining_shirts = 2 ∧ 
    original_cost_per_remaining_shirt = 12.5 :=
by 
  sorry

end original_cost_of_remaining_shirt_l121_121457


namespace negation_equiv_l121_121214

theorem negation_equiv (x : ℝ) : ¬ (x^2 - 1 < 0) ↔ (x^2 - 1 ≥ 0) :=
by
  sorry

end negation_equiv_l121_121214


namespace cube_of_odd_number_minus_itself_divisible_by_24_l121_121898

theorem cube_of_odd_number_minus_itself_divisible_by_24 (n : ℤ) : 
  24 ∣ ((2 * n + 1) ^ 3 - (2 * n + 1)) :=
by
  sorry

end cube_of_odd_number_minus_itself_divisible_by_24_l121_121898


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l121_121383

theorem number_of_subsets_with_at_least_four_adjacent_chairs (chairs : Finset ℕ) (h_chairs : chairs.card = 12)
  (h_circular : ∀ (s : Finset ℕ), s ⊆ chairs → (s.card ≥ 4 → ∃ t : Finset ℕ, t ⊆ chairs ∧ t.card = 4 ∧ ∀ i j ∈ t, abs (i - j) ≤ 1)) :
  ∃ (subsets : Finset (Finset ℕ)), (∀ s ∈ subsets, s ⊆ chairs ∧ s.card ≥ 4) ∧ subsets.card = 169 :=
sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l121_121383


namespace ordered_pairs_1944_l121_121372

theorem ordered_pairs_1944 :
  ∃ n : ℕ, (∀ x y : ℕ, (x * y = 1944 ↔ x > 0 ∧ y > 0)) → n = 24 :=
by
  sorry

end ordered_pairs_1944_l121_121372


namespace average_difference_l121_121527

theorem average_difference (F1 L1 F2 L2 : ℤ) (H1 : F1 = 200) (H2 : L1 = 400) (H3 : F2 = 100) (H4 : L2 = 200) :
  (F1 + L1) / 2 - (F2 + L2) / 2 = 150 := 
by 
  sorry

end average_difference_l121_121527


namespace solution_of_r_and_s_l121_121193

theorem solution_of_r_and_s 
  (r s : ℝ)
  (hr : (r-5)*(r+5) = 25*r - 125)
  (hs : (s-5)*(s+5) = 25*s - 125)
  (h_distinct : r ≠ s)
  (h_order : r > s) : 
  r - s = 15 := by
sorry

end solution_of_r_and_s_l121_121193


namespace total_books_is_10_l121_121934

def total_books (B : ℕ) : Prop :=
  (2 / 5 : ℚ) * B + (3 / 10 : ℚ) * B + ((3 / 10 : ℚ) * B - 1) + 1 = B

theorem total_books_is_10 : total_books 10 := by
  sorry

end total_books_is_10_l121_121934


namespace total_dresses_l121_121983

theorem total_dresses (D M E : ℕ) (h1 : E = 16) (h2 : M = E / 2) (h3 : D = M + 12) : D + M + E = 44 :=
by
  sorry

end total_dresses_l121_121983


namespace cos_double_angle_l121_121668

variable (θ : Real)
variables (h : cos θ = 3 / 5)

theorem cos_double_angle :
  cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l121_121668


namespace angle_C_is_120_degrees_l121_121043

theorem angle_C_is_120_degrees (l m : ℝ) (A B C : ℝ) (hal : l = m) 
  (hA : A = 100) (hB : B = 140) : C = 120 := 
by 
  sorry

end angle_C_is_120_degrees_l121_121043


namespace find_fourth_number_l121_121933

theorem find_fourth_number (x : ℝ) (h : (3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5) = 800.0000000000001) : x = 0.3 :=
by
  sorry

end find_fourth_number_l121_121933


namespace eggs_removed_l121_121443

theorem eggs_removed (initial remaining : ℕ) (h1 : initial = 27) (h2 : remaining = 20) : initial - remaining = 7 :=
by
  sorry

end eggs_removed_l121_121443


namespace find_k_perpendicular_l121_121312

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (1, 1)
def vec_b : ℝ × ℝ := (2, -3)

-- Define a function for the vector k * a - 2 * b
def vec_expression (k : ℝ) : ℝ × ℝ :=
  (k * vec_a.1 - 2 * vec_b.1, k * vec_a.2 - 2 * vec_b.2)

-- Prove that if the dot product of vec_expression k and vec_a is zero, then k = -1
theorem find_k_perpendicular (k : ℝ) :
  ((vec_expression k).1 * vec_a.1 + (vec_expression k).2 * vec_a.2 = 0) → k = -1 :=
by
  sorry

end find_k_perpendicular_l121_121312


namespace min_value_l121_121169

theorem min_value (a b : ℝ) (h : a * b > 0) : (∃ x, x = a^2 + 4 * b^2 + 1 / (a * b) ∧ ∀ y, y = a^2 + 4 * b^2 + 1 / (a * b) → y ≥ 4) :=
sorry

end min_value_l121_121169


namespace speed_down_l121_121588

theorem speed_down {u avg_speed d v : ℝ} (hu : u = 18) (havg : avg_speed = 20.571428571428573) (hv : 2 * d / ((d / u) + (d / v)) = avg_speed) : v = 24 :=
by
  have h1 : 20.571428571428573 = 20.571428571428573 := rfl
  have h2 : 18 = 18 := rfl
  sorry

end speed_down_l121_121588


namespace units_digit_of_27_mul_36_l121_121621

theorem units_digit_of_27_mul_36 : (27 * 36) % 10 = 2 := by
  sorry

end units_digit_of_27_mul_36_l121_121621


namespace intersection_is_singleton_l121_121156

def M : Set ℕ := {1, 2}
def N : Set ℕ := {n | ∃ a ∈ M, n = 2 * a - 1}

theorem intersection_is_singleton : M ∩ N = {1} :=
by sorry

end intersection_is_singleton_l121_121156


namespace poodle_barked_24_times_l121_121974

-- Defining the conditions and question in Lean
def poodle_barks (terrier_barks_per_hush times_hushed: ℕ) : ℕ :=
  2 * terrier_barks_per_hush * times_hushed

theorem poodle_barked_24_times (terrier_barks_per_hush times_hushed: ℕ) :
  terrier_barks_per_hush = 2 → times_hushed = 6 → poodle_barks terrier_barks_per_hush times_hushed = 24 :=
by
  intros
  sorry

end poodle_barked_24_times_l121_121974


namespace smallest_in_sample_l121_121629

theorem smallest_in_sample:
  ∃ (m : ℕ) (δ : ℕ), m ≥ 0 ∧ δ > 0 ∧ δ * 5 = 80 ∧ 42 = δ * (42 / δ) + m ∧ m < δ ∧ (∀ i < 5, m + i * δ < 80) → m = 10 :=
by
  sorry

end smallest_in_sample_l121_121629


namespace candy_bar_calories_l121_121771

theorem candy_bar_calories :
  let calA := 150
  let calB := 200
  let calC := 250
  let countA := 2
  let countB := 3
  let countC := 4
  (countA * calA + countB * calB + countC * calC) = 1900 :=
by
  sorry

end candy_bar_calories_l121_121771


namespace minimum_tasks_for_18_points_l121_121418

def task_count (points : ℕ) : ℕ :=
  if points <= 9 then
    (points / 3) * 1
  else if points <= 15 then
    3 + (points - 9 + 2) / 3 * 2
  else
    3 + 4 + (points - 15 + 2) / 3 * 3

theorem minimum_tasks_for_18_points : task_count 18 = 10 := by
  sorry

end minimum_tasks_for_18_points_l121_121418


namespace solve_for_k_l121_121635

theorem solve_for_k (x y k : ℝ) (h1 : x = 2) (h2 : y = 3) (h3 : 5 * x - k * y - 7 = 0) : k = 1 :=
by
  sorry

end solve_for_k_l121_121635


namespace total_stuffed_animals_l121_121883

theorem total_stuffed_animals (M K T : ℕ) 
  (hM : M = 34) 
  (hK : K = 2 * M) 
  (hT : T = K + 5) : 
  M + K + T = 175 :=
by
  -- Adding sorry to complete the placeholder
  sorry

end total_stuffed_animals_l121_121883


namespace octal_to_decimal_7564_l121_121604

theorem octal_to_decimal_7564 : 7 * 8^3 + 5 * 8^2 + 6 * 8^1 + 4 * 8^0 = 3956 :=
by
  sorry 

end octal_to_decimal_7564_l121_121604


namespace sequence_term_four_l121_121713

theorem sequence_term_four (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 2) : a 4 = 7 :=
sorry

end sequence_term_four_l121_121713


namespace range_of_a_l121_121476

noncomputable def f (x a : ℝ) : ℝ := x^3 - a*x^2 + 4

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) → 3 < a :=
by
  sorry

end range_of_a_l121_121476


namespace boy_current_age_l121_121948

theorem boy_current_age (x : ℕ) (h : 5 ≤ x) (age_statement : x = 2 * (x - 5)) : x = 10 :=
by
  sorry

end boy_current_age_l121_121948


namespace janet_percentage_of_snowballs_l121_121182

-- Define the number of snowballs made by Janet
def janet_snowballs : ℕ := 50

-- Define the number of snowballs made by Janet's brother
def brother_snowballs : ℕ := 150

-- Define the total number of snowballs
def total_snowballs : ℕ := janet_snowballs + brother_snowballs

-- Define the percentage calculation function
def calculate_percentage (part whole : ℕ) : ℚ :=
  (part : ℚ) / (whole : ℚ) * 100

-- Proof statement
theorem janet_percentage_of_snowballs : calculate_percentage janet_snowballs total_snowballs = 25 := 
by
  sorry

end janet_percentage_of_snowballs_l121_121182


namespace cos_double_angle_l121_121670

variable (θ : Real)
variables (h : cos θ = 3 / 5)

theorem cos_double_angle :
  cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l121_121670


namespace inequality_350_l121_121358

theorem inequality_350 (a b c d : ℝ) : 
  (a - b) * (b - c) * (c - d) * (d - a) + (a - c)^2 * (b - d)^2 ≥ 0 :=
by
  sorry

end inequality_350_l121_121358


namespace katrina_cookies_left_l121_121342

/-- Katrina’s initial number of cookies. -/
def initial_cookies : ℕ := 120

/-- Cookies sold in the morning. 
    1 dozen is 12 cookies, so 3 dozen is 36 cookies. -/
def morning_sales : ℕ := 3 * 12

/-- Cookies sold during the lunch rush. -/
def lunch_sales : ℕ := 57

/-- Cookies sold in the afternoon. -/
def afternoon_sales : ℕ := 16

/-- Calculate the number of cookies left after all sales. -/
def cookies_left : ℕ :=
  initial_cookies - morning_sales - lunch_sales - afternoon_sales

/-- Prove that the number of cookies left for Katrina to take home is 11. -/
theorem katrina_cookies_left : cookies_left = 11 := by
  sorry

end katrina_cookies_left_l121_121342


namespace union_of_A_and_B_l121_121727

-- Define the sets A and B
def A : Set ℤ := {-1, 0}
def B : Set ℤ := {0, 1}

-- Prove that the union of A and B is {-1, 0, 1}
theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} :=
  by sorry

end union_of_A_and_B_l121_121727


namespace tangent_condition_sum_f_l121_121474

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

theorem tangent_condition (a : ℝ) (h : f a 1 = f a 1) (m : ℝ) : 
    (3 * a + 1 = (7 - (f a 1)) / 2) := 
    sorry

theorem sum_f (a : ℝ) (h : a = 3/7) : 
    f a (-4) + f a (-3) + f a (-2) + f a (-1) + f a 0 + 
    f a 1 + f a 2 + f a 3 + f a 4 = 9 := 
    sorry

end tangent_condition_sum_f_l121_121474


namespace largest_possible_three_day_success_ratio_l121_121813

noncomputable def beta_max_success_ratio : ℝ :=
  let (a : ℕ) := 33
  let (b : ℕ) := 50
  let (c : ℕ) := 225
  let (d : ℕ) := 300
  let (e : ℕ) := 100
  let (f : ℕ) := 200
  a / b + c / d + e / f

theorem largest_possible_three_day_success_ratio :
  beta_max_success_ratio = (358 / 600 : ℝ) :=
by
  sorry

end largest_possible_three_day_success_ratio_l121_121813


namespace baker_initial_cakes_cannot_be_determined_l121_121121

theorem baker_initial_cakes_cannot_be_determined (initial_pastries sold_cakes sold_pastries remaining_pastries : ℕ)
  (h1 : initial_pastries = 148)
  (h2 : sold_cakes = 15)
  (h3 : sold_pastries = 103)
  (h4 : remaining_pastries = 45)
  (h5 : sold_pastries + remaining_pastries = initial_pastries) :
  True :=
by
  sorry

end baker_initial_cakes_cannot_be_determined_l121_121121


namespace olivia_did_not_sell_4_bars_l121_121989

-- Define the constants and conditions
def price_per_bar : ℕ := 3
def total_bars : ℕ := 7
def money_made : ℕ := 9

-- Calculate the number of bars sold
def bars_sold : ℕ := money_made / price_per_bar

-- Calculate the number of bars not sold
def bars_not_sold : ℕ := total_bars - bars_sold

-- Theorem to prove the answer
theorem olivia_did_not_sell_4_bars : bars_not_sold = 4 := 
by 
  sorry

end olivia_did_not_sell_4_bars_l121_121989


namespace petrov_vasechkin_boards_l121_121734

theorem petrov_vasechkin_boards:
  ∃ n : ℕ, 
  (∃ x y : ℕ, 2 * x + 3 * y = 87 ∧ x + y = n) ∧ 
  (∃ u v : ℕ, 3 * u + 5 * v = 94 ∧ u + v = n) ∧ 
  n = 30 := 
sorry

end petrov_vasechkin_boards_l121_121734


namespace alyssa_total_spent_l121_121602

theorem alyssa_total_spent :
  let grapes := 12.08
  let cherries := 9.85
  grapes + cherries = 21.93 := by
  sorry

end alyssa_total_spent_l121_121602


namespace prob_m_gt_n_l121_121297

theorem prob_m_gt_n :
  let A : Finset ℕ := {2, 4, 6, 8, 10}
  let B : Finset ℕ := {1, 3, 5, 7, 9}
  (A.card = 5) → (B.card = 5) →
  (∃ (m : ℕ) (hm : m ∈ A) (n : ℕ) (hn : n ∈ B), m > n) →
  (15 / 25 : ℝ) = 0.6 :=
by
  intros A B hA hB hex
  sorry

end prob_m_gt_n_l121_121297


namespace Natasha_speed_over_limit_l121_121892

theorem Natasha_speed_over_limit (d : ℕ) (t : ℕ) (speed_limit : ℕ) 
    (h1 : d = 60) 
    (h2 : t = 1) 
    (h3 : speed_limit = 50) : (d / t - speed_limit = 10) :=
by
  -- Because d = 60, t = 1, and speed_limit = 50, we need to prove (60 / 1 - 50) = 10
  sorry

end Natasha_speed_over_limit_l121_121892


namespace cost_of_painting_new_room_l121_121913

theorem cost_of_painting_new_room
  (L B H : ℝ)    -- Dimensions of the original room
  (c : ℝ)        -- Cost to paint the original room
  (h₁ : c = 350) -- Given that the cost of painting the original room is Rs. 350
  (A : ℝ)        -- Area of the walls of the original room
  (h₂ : A = 2 * (L + B) * H) -- Given the area calculation for the original room
  (newA : ℝ)     -- Area of the walls of the new room
  (h₃ : newA = 18 * (L + B) * H) -- Given the area calculation for the new room
  : (350 / (2 * (L + B) * H)) * (18 * (L + B) * H) = 3150 :=
by
  sorry

end cost_of_painting_new_room_l121_121913


namespace graph_remains_connected_after_removing_one_color_l121_121774

/-- A 30-point graph where each pair is connected by an edge of one of four colors remains connected after removing all edges of some one color. -/
theorem graph_remains_connected_after_removing_one_color :
  ∃ C : Fin 4 → Prop,
    ∀ G : SimpleGraph (Fin 30),
    ((∀ v w : Fin 30, v ≠ w → ∃ c : Fin 4, G.edge v w c) →
     ∀ c : Fin 4, (∀ p q : Fin 30, p ≠ q → (∃ r : Fin 30, r ≠ p ∧ r ≠ q ∧ ¬G.edge p q c)) → 
     Connected (G ⊖ {E | E.2 = c → E ∉ G.edge})) :=
by sorry

end graph_remains_connected_after_removing_one_color_l121_121774


namespace school_girls_more_than_boys_l121_121859

def num_initial_girls := 632
def num_initial_boys := 410
def num_new_girls := 465
def num_total_girls := num_initial_girls + num_new_girls
def num_difference_girls_boys := num_total_girls - num_initial_boys

theorem school_girls_more_than_boys :
  num_difference_girls_boys = 687 :=
by
  sorry

end school_girls_more_than_boys_l121_121859


namespace monthly_salary_l121_121953

theorem monthly_salary (S : ℝ) (h1 : 0.20 * S + 1.20 * 0.80 * S = S) (h2 : S - 1.20 * 0.80 * S = 260) : S = 6500 :=
by
  sorry

end monthly_salary_l121_121953


namespace double_angle_cosine_l121_121678

theorem double_angle_cosine (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end double_angle_cosine_l121_121678


namespace smallest_positive_period_max_value_in_interval_min_value_in_interval_l121_121151

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + Real.pi / 6)

theorem smallest_positive_period : ∀ x, f (x + Real.pi) = f x := by
  sorry

theorem max_value_in_interval :
  ∃ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x = 2 := by
  sorry

theorem min_value_in_interval :
  ∃ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x = -1 := by
  sorry

end smallest_positive_period_max_value_in_interval_min_value_in_interval_l121_121151


namespace functional_solutions_l121_121450

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, x * f y + y * f x = (x + y) * (f x) * (f y)

theorem functional_solutions (f : ℝ → ℝ) (h : functional_equation f) : 
  (∀ x : ℝ, f x = 0) ∨ (∃ (a : ℝ), ∀ x : ℝ, (x ≠ 0 → f x = 1) ∧ (x = 0 → f x = a)) :=
  sorry

end functional_solutions_l121_121450


namespace vasya_tolya_badges_l121_121558

theorem vasya_tolya_badges (x y : ℤ)
    (h1 : y = x + 5) -- Vasya initially had 5 more badges than Tolya
    (h2 : (y - (6 * (y / 25) / 25) + (4 * x) / 25) = (x - (4 * x) / 5 + 6 * (y / 25) / 5 - 1)) : -- equation balancing after exchange
    x = 45 ∧ y = 50 := 
sorry

end vasya_tolya_badges_l121_121558


namespace network_connections_l121_121553

theorem network_connections (n : ℕ) (k : ℕ) :
  n = 30 → k = 4 → (n * k) / 2 = 60 := by
  intros h₁ h₂
  rw [h₁, h₂]
  sorry

end network_connections_l121_121553


namespace martha_cards_l121_121513

theorem martha_cards (start_cards : ℕ) : start_cards + 76 = 79 → start_cards = 3 :=
by
  sorry

end martha_cards_l121_121513


namespace scientific_notation_216000_l121_121748

theorem scientific_notation_216000 : 216000 = 2.16 * 10^5 :=
by
  -- proof will be provided here
  sorry

end scientific_notation_216000_l121_121748


namespace quadratic_has_two_distinct_real_roots_l121_121613

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  (x : ℝ) -> x^2 + m * x + 1 = 0 → (m < -2 ∨ m > 2) :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l121_121613


namespace milk_needed_for_cookies_l121_121776

-- Definition of the problem conditions
def cookies_per_milk_usage := 24
def milk_in_liters := 5
def liters_to_milliliters := 1000
def milk_for_6_cookies := 1250

-- Prove that 1250 milliliters of milk are needed to bake 6 cookies
theorem milk_needed_for_cookies
  (h1 : cookies_per_milk_usage = 24)
  (h2 : milk_in_liters = 5)
  (h3 : liters_to_milliliters = 1000) :
  milk_for_6_cookies = 1250 :=
by
  -- Proof is omitted with sorry
  sorry

end milk_needed_for_cookies_l121_121776


namespace distinct_arrangements_of_pebbles_in_octagon_l121_121865

noncomputable def number_of_distinct_arrangements : ℕ :=
  (Nat.factorial 8) / 16

theorem distinct_arrangements_of_pebbles_in_octagon : 
  number_of_distinct_arrangements = 2520 :=
by
  sorry

end distinct_arrangements_of_pebbles_in_octagon_l121_121865


namespace badge_counts_l121_121564

variable (x : ℝ) -- Number of badges Tolya initially had
variable (y : ℝ) -- Number of badges Vasya initially had 

-- Conditions
axiom initial_more_badges : y = x + 5
axiom exchange_badges_vasya : y - 0.24 * y + 0.2 * x = x - 1 + 0.24 * y
axiom exchange_badges_tolya : x - 0.2 * x + 0.24 * y = x - 0.2 * x + 0.24 * y - 0.24 * y + y

-- Proof goal
theorem badge_counts (x : ℝ) (y : ℝ) 
  (h1 : y = x + 5) 
  (h2 : y - 0.24 * y + 0.2 * x = x - 1 + 0.24 * y) 
  (h3 : x - 0.2 * x + 0.24 * y = x - 0.2 * x + 0.24 * y - 0.24 * y + y) : 
  x = 45 ∧ y = 50 :=
by
  sorry

end badge_counts_l121_121564


namespace h_at_0_l121_121511

noncomputable def h (x : ℝ) : ℝ := sorry -- the actual polynomial
-- Conditions for h(x)
axiom h_cond1 : h (-2) = -4
axiom h_cond2 : h (1) = -1
axiom h_cond3 : h (-3) = -9
axiom h_cond4 : h (3) = -9
axiom h_cond5 : h (5) = -25

-- Statement of the proof problem
theorem h_at_0 : h (0) = -90 := sorry

end h_at_0_l121_121511


namespace subsets_with_at_least_four_adjacent_chairs_l121_121390

theorem subsets_with_at_least_four_adjacent_chairs :
  let chairs := Finset.range 12 in
  ∑ n in chairs, if n ≥ 4 then (12.choose n) else 0 = 1622 :=
by
  sorry

end subsets_with_at_least_four_adjacent_chairs_l121_121390


namespace distance_from_origin_to_line_AB_is_sqrt6_div_3_l121_121301

open Real

structure Point where
  x : ℝ
  y : ℝ

def ellipse (p : Point) : Prop :=
  p.x^2 / 2 + p.y^2 = 1

def left_focus : Point := ⟨-1, 0⟩

def line_through_focus (t : ℝ) (p : Point) : Prop :=
  p.x = t * p.y - 1

def origin : Point := ⟨0, 0⟩

def perpendicular (A B : Point) : Prop :=
  A.x * B.x + A.y * B.y = 0

noncomputable def distance (O : Point) (A B : Point) : ℝ :=
  let a := A.y - B.y
  let b := B.x - A.x
  let c := A.x * B.y - A.y * B.x
  abs (a * O.x + b * O.y + c) / sqrt (a^2 + b^2)

theorem distance_from_origin_to_line_AB_is_sqrt6_div_3 
  (A B : Point)
  (hA_on_ellipse : ellipse A)
  (hB_on_ellipse : ellipse B)
  (h_line_through_focus : ∃ t : ℝ, line_through_focus t A ∧ line_through_focus t B)
  (h_perpendicular : perpendicular A B) :
  distance origin A B = sqrt 6 / 3 := sorry

end distance_from_origin_to_line_AB_is_sqrt6_div_3_l121_121301


namespace fifth_group_members_l121_121964

-- Define the number of members in the choir
def total_members : ℕ := 150 

-- Define the number of members in each group
def group1 : ℕ := 18 
def group2 : ℕ := 29 
def group3 : ℕ := 34 
def group4 : ℕ := 23 

-- Define the fifth group as the remaining members
def group5 : ℕ := total_members - (group1 + group2 + group3 + group4)

theorem fifth_group_members : group5 = 46 := sorry

end fifth_group_members_l121_121964


namespace cube_root_inequality_l121_121847

theorem cube_root_inequality (a b : ℝ) (h : a > b) : (a ^ (1/3)) > (b ^ (1/3)) :=
sorry

end cube_root_inequality_l121_121847


namespace enemies_left_undefeated_l121_121710

theorem enemies_left_undefeated (points_per_enemy points_earned total_enemies : ℕ) 
  (h1 : points_per_enemy = 3)
  (h2 : total_enemies = 6)
  (h3 : points_earned = 12) : 
  (total_enemies - points_earned / points_per_enemy) = 2 :=
by
  sorry

end enemies_left_undefeated_l121_121710


namespace necessary_but_not_sufficient_l121_121351

variable (p q : Prop)

theorem necessary_but_not_sufficient (h : ¬p → q) (h1 : ¬ (q → ¬p)) : ¬q → p := 
by
  sorry

end necessary_but_not_sufficient_l121_121351


namespace scientific_notation_of_0_0000205_l121_121449

noncomputable def scientific_notation (n : ℝ) : ℝ × ℤ := sorry

theorem scientific_notation_of_0_0000205 :
  scientific_notation 0.0000205 = (2.05, -5) :=
sorry

end scientific_notation_of_0_0000205_l121_121449


namespace div_expression_l121_121817

variable {α : Type*} [Field α]

theorem div_expression (a b c : α) : 4 * a^2 * b^2 * c / (-2 * a * b^2) = -2 * a * c := by
  sorry

end div_expression_l121_121817


namespace solve_for_x_l121_121139

theorem solve_for_x (x : ℝ) : 4 * x - 8 + 3 * x = 12 + 5 * x → x = 10 :=
by
  intro h
  sorry

end solve_for_x_l121_121139


namespace solve_equation_1_solve_equation_2_solve_equation_3_l121_121740

theorem solve_equation_1 (x : ℝ) : x^2 - 2 * x = 0 ↔ x = 0 ∨ x = 2 := 
sorry

theorem solve_equation_2 (x : ℝ) : (2 * x - 1)^2 = (3 - x)^2 ↔ x = -2 ∨ x = 4 / 3 := 
sorry

theorem solve_equation_3 (x : ℝ) : 3 * x * (x - 2) = x - 2 ↔ x = 2 ∨ x = 1 / 3 :=
sorry

end solve_equation_1_solve_equation_2_solve_equation_3_l121_121740


namespace fraction_simplification_l121_121278

theorem fraction_simplification (a b c : ℝ) :
  (4 * a^2 + 2 * c^2 - 4 * b^2 - 8 * b * c) / (3 * a^2 + 6 * a * c - 3 * c^2 - 6 * a * b) =
  (4 / 3) * ((a - 2 * b + c) * (a - c)) / ((a - b + c) * (a - b - c)) :=
by
  sorry

end fraction_simplification_l121_121278


namespace inequality_proof_l121_121648

theorem inequality_proof
  (x y z : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (hxyz : x * y * z = 1) :
  x^2 + y^2 + z^2 + x * y + y * z + z * x ≥ 2 * (Real.sqrt x + Real.sqrt y + Real.sqrt z) :=
by
  sorry

end inequality_proof_l121_121648


namespace cos_double_angle_l121_121654

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (2 * θ) = -7 / 25 := 
by
  sorry

end cos_double_angle_l121_121654


namespace find_n_l121_121488

noncomputable def condition (n : ℕ) : Prop :=
  (1/5)^n * (1/4)^18 = 1 / (2 * 10^35)

theorem find_n (n : ℕ) (h : condition n) : n = 35 :=
by
  sorry

end find_n_l121_121488


namespace radius_of_circumcircle_of_triangle_l121_121471

theorem radius_of_circumcircle_of_triangle (a b c : ℝ)
  (h : a^2 + b^2 + c^2 - 6*a - 8*b - 10*c + 50 = 0) :
  a = 3 ∧ b = 4 ∧ c = 5 ∧ (∃ (R : ℝ), R = 2.5) :=
by {
  sorry
}

end radius_of_circumcircle_of_triangle_l121_121471


namespace rooms_with_two_beds_l121_121814

variable (x y : ℕ)

theorem rooms_with_two_beds:
  x + y = 13 →
  2 * x + 3 * y = 31 →
  x = 8 :=
by
  intros h1 h2
  sorry

end rooms_with_two_beds_l121_121814


namespace combined_mpg_l121_121434

theorem combined_mpg (miles_alice : ℕ) (mpg_alice : ℕ) (miles_bob : ℕ) (mpg_bob : ℕ) :
  miles_alice = 120 ∧ mpg_alice = 30 ∧ miles_bob = 180 ∧ mpg_bob = 20 →
  (miles_alice + miles_bob) / ((miles_alice / mpg_alice) + (miles_bob / mpg_bob)) = 300 / 13 :=
by
  intros h
  sorry

end combined_mpg_l121_121434


namespace least_number_to_add_l121_121793

theorem least_number_to_add (x : ℕ) (h : 1056 % 23 = 21) : (1056 + x) % 23 = 0 ↔ x = 2 :=
by {
    sorry
}

end least_number_to_add_l121_121793


namespace increasing_C_l121_121640

theorem increasing_C (e R r : ℝ) (n : ℕ) (h₁ : 0 < e) (h₂ : 0 < R) (h₃ : 0 < r) (h₄ : 0 < n) :
    ∀ n1 n2 : ℕ, n1 < n2 → (e^2 * n1) / (R + n1 * r) < (e^2 * n2) / (R + n2 * r) :=
by
  sorry

end increasing_C_l121_121640


namespace internal_angles_and_area_of_grey_triangle_l121_121939

/-- Given three identical grey triangles, 
    three identical squares, and an equilateral 
    center triangle with area 2 cm^2,
    the internal angles of the grey triangles 
    are 120 degrees and 30 degrees, and the 
    total grey area is 6 cm^2. -/
theorem internal_angles_and_area_of_grey_triangle 
  (triangle_area : ℝ)
  (α β : ℝ)
  (grey_area : ℝ) :
  triangle_area = 2 →  
  α = 120 ∧ β = 30 ∧ grey_area = 6 :=
by
  sorry

end internal_angles_and_area_of_grey_triangle_l121_121939


namespace randy_piggy_bank_l121_121057

theorem randy_piggy_bank : 
  ∀ (initial_amount trips_per_month cost_per_trip months_per_year total_spent_left : ℕ),
  initial_amount = 200 →
  cost_per_trip = 2 →
  trips_per_month = 4 →
  months_per_year = 12 →
  total_spent_left = initial_amount - (cost_per_trip * trips_per_month * months_per_year) →
  total_spent_left = 104 :=
by
  intros initial_amount trips_per_month cost_per_trip months_per_year total_spent_left
  sorry

end randy_piggy_bank_l121_121057


namespace ava_first_coupon_day_l121_121119

theorem ava_first_coupon_day (first_coupon_day : ℕ) (coupon_interval : ℕ) 
    (closed_day : ℕ) (days_in_week : ℕ):
  first_coupon_day = 2 →  -- starting on Tuesday (considering Monday as 1)
  coupon_interval = 13 →
  closed_day = 7 →        -- Saturday is represented by 7
  days_in_week = 7 →
  ∀ n : ℕ, ((first_coupon_day + n * coupon_interval) % days_in_week) ≠ closed_day :=
by 
  -- Proof can be filled here.
  sorry

end ava_first_coupon_day_l121_121119


namespace pow_mod_3_225_l121_121231

theorem pow_mod_3_225 :
  (3 ^ 225) % 11 = 1 :=
by
  -- Given condition from problem:
  have h : 3 ^ 5 % 11 = 1 := by norm_num
  -- Proceed to prove based on this condition
  sorry

end pow_mod_3_225_l121_121231


namespace percentage_decrease_in_y_when_x_doubles_l121_121523

variable {k x y : ℝ}
variable (h_pos_x : 0 < x) (h_pos_y : 0 < y)
variable (inverse_proportional : x * y = k)

theorem percentage_decrease_in_y_when_x_doubles :
  (x' = 2 * x) →
  (y' = y / 2) →
  (100 * (y - y') / y) = 50 :=
by
  intro h1 h2
  simp [h1, h2]
  sorry

end percentage_decrease_in_y_when_x_doubles_l121_121523


namespace compute_cos_l121_121303

noncomputable def angle1 (A C B : ℝ) : Prop := A + C = 2 * B
noncomputable def angle2 (A C B : ℝ) : Prop := 1 / Real.cos A + 1 / Real.cos C = -Real.sqrt 2 / Real.cos B

theorem compute_cos (A B C : ℝ) (h1 : angle1 A C B) (h2 : angle2 A C B) : 
  Real.cos ((A - C) / 2) = Real.sqrt 2 / 2 :=
sorry

end compute_cos_l121_121303


namespace sequence_ratio_l121_121078

theorem sequence_ratio (S T a b : ℕ → ℚ) (h_sum_ratio : ∀ (n : ℕ), S n / T n = (7*n + 2) / (n + 3)) :
  a 7 / b 7 = 93 / 16 :=
by
  sorry

end sequence_ratio_l121_121078


namespace find_B_l121_121006

def A (a : ℝ) : Set ℝ := {3, Real.log a / Real.log 2}
def B (a b : ℝ) : Set ℝ := {a, b}

theorem find_B (a b : ℝ) (hA : A a = {3, 2}) (hB : B a b = {a, b}) (h : (A a) ∩ (B a b) = {2}) :
  B a b = {2, 4} :=
sorry

end find_B_l121_121006


namespace keiko_ephraim_same_heads_l121_121870

def keiko_outcomes := {HH, HT, TH, TT}
def ephraim_outcomes := {HH, HT, TH, TT}

def prob_outcome (outcome : ℕ) : ℚ := 1 / 4
def prob_same_heads := 3 / 8

theorem keiko_ephraim_same_heads :
  let outcomes := keiko_outcomes × ephraim_outcomes
  let favorable_outcomes := {(TT, TT), (HH, HH)} ∪ ({(HT, HT), (HT, TH), (TH, HT), (TH, TH)} ∩ {(HT, HT), (TH, HT)})
  let p := ∑ x in favorable_outcomes, prob_outcome x.1 * prob_outcome x.2 in
  p = prob_same_heads :=
by sorry

end keiko_ephraim_same_heads_l121_121870


namespace thirtieth_entry_satisfies_l121_121625

def r_9 (n : ℕ) : ℕ := n % 9

theorem thirtieth_entry_satisfies (n : ℕ) (h : ∃ k : ℕ, k < 30 ∧ ∀ m < 30, k ≠ m → 
    (r_9 (7 * n + 3) ≤ 4) ∧ 
    ((r_9 (7 * n + 3) ≤ 4) ↔ 
    (r_9 (7 * m + 3) > 4))) :
  n = 37 :=
sorry

end thirtieth_entry_satisfies_l121_121625


namespace range_of_a_l121_121840

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x - x^2 + (1 / 2) * a

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f a x ≤ 0) ↔ (0 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_a_l121_121840


namespace simplify_expression_l121_121901

theorem simplify_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) *
  (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 := 
by sorry

end simplify_expression_l121_121901


namespace fraction_meaningful_l121_121849

theorem fraction_meaningful (x : ℝ) : x - 3 ≠ 0 ↔ x ≠ 3 :=
by sorry

end fraction_meaningful_l121_121849


namespace chord_midpoint_line_eqn_l121_121862

-- Definitions of points and the ellipse condition
def P : ℝ × ℝ := (3, 2)

def is_midpoint (P E F : ℝ × ℝ) := 
  P.1 = (E.1 + F.1) / 2 ∧ P.2 = (E.2 + F.2) / 2

def ellipse (x y : ℝ) := 
  4 * x^2 + 9 * y^2 = 144

theorem chord_midpoint_line_eqn
  (E F : ℝ × ℝ) 
  (h1 : is_midpoint P E F)
  (h2 : ellipse E.1 E.2)
  (h3 : ellipse F.1 F.2):
  ∃ (m b : ℝ), (P.2 = m * P.1 + b) ∧ (2 * P.1 + 3 * P.2 - 12 = 0) :=
by 
  sorry

end chord_midpoint_line_eqn_l121_121862


namespace geometric_sequence_q_cubed_l121_121912

theorem geometric_sequence_q_cubed (q a_1 : ℝ) (h1 : q ≠ 0) (h2 : q ≠ 1) 
(h3 : 2 * (a_1 * (1 - q^9) / (1 - q)) = (a_1 * (1 - q^3) / (1 - q)) + (a_1 * (1 - q^6) / (1 - q))) : 
  q^3 = -1/2 := by
  sorry

end geometric_sequence_q_cubed_l121_121912


namespace range_of_y_l121_121705

theorem range_of_y (y : ℝ) (h1 : y < 0) (h2 : ⌈y⌉ * ⌊y⌋ = 72) : y ∈ set.Ioo (-9 : ℝ) (-8 : ℝ) :=
sorry

end range_of_y_l121_121705


namespace polygon_sides_l121_121769

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 3 * 360)
  (h2 : n ≥ 3) : 
  n = 8 := 
by
  sorry

end polygon_sides_l121_121769


namespace sum_arithmetic_sequence_l121_121467

theorem sum_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_arith_seq : ∀ n, S (n + 1) - S n = a n)
  (h_S2 : S 2 = 4) 
  (h_S4 : S 4 = 16) 
: a 5 + a 6 = 20 :=
sorry

end sum_arithmetic_sequence_l121_121467


namespace initial_value_l121_121779

theorem initial_value (x k : ℤ) (h : x + 294 = k * 456) : x = 162 :=
sorry

end initial_value_l121_121779


namespace cosine_double_angle_l121_121683

theorem cosine_double_angle
  (θ : ℝ)
  (cos_θ : ℝ)
  (h : cos_θ = 3 / 5) :
  2 * cos_θ^2 - 1 = -7 / 25 :=
by
  sorry

end cosine_double_angle_l121_121683


namespace length_of_first_platform_is_140_l121_121600

-- Definitions based on problem conditions
def train_length : ℝ := 190
def time_first_platform : ℝ := 15
def time_second_platform : ℝ := 20
def length_second_platform : ℝ := 250

-- Definition for the length of the first platform (what we're proving)
def length_first_platform (L : ℝ) : Prop :=
  (time_first_platform * (train_length + L) = time_second_platform * (train_length + length_second_platform))

-- Theorem: The length of the first platform is 140 meters
theorem length_of_first_platform_is_140 : length_first_platform 140 :=
  by sorry

end length_of_first_platform_is_140_l121_121600


namespace intercepts_sum_eq_eight_l121_121279

def parabola_x_y (x y : ℝ) := x = 3 * y^2 - 9 * y + 5

theorem intercepts_sum_eq_eight :
  ∃ (a b c : ℝ), parabola_x_y a 0 ∧ parabola_x_y 0 b ∧ parabola_x_y 0 c ∧ a + b + c = 8 :=
sorry

end intercepts_sum_eq_eight_l121_121279


namespace necessary_but_not_sufficient_l121_121794

theorem necessary_but_not_sufficient (x : ℝ) (h : x ≠ 1) : x^2 - 3 * x + 2 ≠ 0 :=
by
  intro h1
  -- Insert the proof here
  sorry

end necessary_but_not_sufficient_l121_121794


namespace contrapositive_x_squared_l121_121366

theorem contrapositive_x_squared :
  (∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1 → x^2 ≥ 1) := 
sorry

end contrapositive_x_squared_l121_121366


namespace tom_and_jerry_drank_80_ounces_l121_121555

theorem tom_and_jerry_drank_80_ounces
    (T J : ℝ) 
    (initial_T : T = 40)
    (initial_J : J = 2 * T)
    (T_drank J_drank : ℝ)
    (T_remaining J_remaining : ℝ)
    (T_after_pour J_after_pour : ℝ)
    (T_final J_final : ℝ)
    (H1 : T_drank = (2 / 3) * T)
    (H2 : J_drank = (2 / 3) * J)
    (H3 : T_remaining = T - T_drank)
    (H4 : J_remaining = J - J_drank)
    (H5 : T_after_pour = T_remaining + (1 / 4) * J_remaining)
    (H6 : J_after_pour = J_remaining - (1 / 4) * J_remaining)
    (H7 : T_final = T_after_pour - 5)
    (H8 : J_final = J_after_pour + 5)
    (H9 : T_final = J_final + 4)
    : T_drank + J_drank = 80 :=
by
  sorry

end tom_and_jerry_drank_80_ounces_l121_121555


namespace distinct_solutions_diff_l121_121188

theorem distinct_solutions_diff {r s : ℝ} 
  (h1 : ∀ a b : ℝ, (a ≠ b → ( ∃ x, x ≠ a ∧ x ≠ b ∧ (x = r ∨ x = s) ∧ (x-5)(x+5) = 25*x - 125) )) 
  (h2 : r > s) : r - s = 15 :=
by
  sorry

end distinct_solutions_diff_l121_121188


namespace sin_minus_pi_over_3_eq_neg_four_fifths_l121_121013

theorem sin_minus_pi_over_3_eq_neg_four_fifths
  (α : ℝ)
  (h : Real.cos (α + π / 6) = 4 / 5) :
  Real.sin (α - π / 3) = - (4 / 5) :=
by
  sorry

end sin_minus_pi_over_3_eq_neg_four_fifths_l121_121013


namespace expected_value_of_expression_is_50_l121_121730

def expected_value_single_digit : ℚ := (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) / 9

def expected_value_expression : ℚ :=
  (expected_value_single_digit + expected_value_single_digit + expected_value_single_digit +
   (expected_value_single_digit + expected_value_single_digit * expected_value_single_digit) +
   (expected_value_single_digit * expected_value_single_digit + expected_value_single_digit) +
   (expected_value_single_digit * expected_value_single_digit * expected_value_single_digit)) / 4

theorem expected_value_of_expression_is_50 :
  expected_value_expression = 50 := sorry

end expected_value_of_expression_is_50_l121_121730


namespace switches_connections_l121_121546

theorem switches_connections (n k : ℕ) (h1 : n = 30) (h2 : k = 4) : (n * k) / 2 = 60 :=
by
  sorry

end switches_connections_l121_121546


namespace circle_properties_l121_121528

noncomputable def pi : Real := 3.14
variable (C : Real) (diameter : Real) (radius : Real) (area : Real)

theorem circle_properties (h₀ : C = 31.4) :
  radius = C / (2 * pi) ∧
  diameter = 2 * radius ∧
  area = pi * radius^2 ∧
  radius = 5 ∧
  diameter = 10 ∧
  area = 78.5 :=
by
  sorry

end circle_properties_l121_121528


namespace Nero_speed_is_8_l121_121503

-- Defining the conditions
def Jerome_time := 6 -- in hours
def Nero_time := 3 -- in hours
def Jerome_speed := 4 -- in miles per hour

-- Calculation step
def Distance := Jerome_speed * Jerome_time

-- The theorem we need to prove (Nero's speed)
theorem Nero_speed_is_8 :
  (Distance / Nero_time) = 8 := by
  sorry

end Nero_speed_is_8_l121_121503


namespace number_of_students_l121_121208

theorem number_of_students (N T : ℕ) (h1 : T = 80 * N)
  (h2 : (T - 100) / (N - 5) = 90) : N = 35 := 
by 
  sorry

end number_of_students_l121_121208


namespace series_product_solution_l121_121611

theorem series_product_solution (y : ℚ) :
  ( (∑' n, (1 / 2) * (1 / 3) ^ n) * (∑' n, (1 / 3) * (-1 / 3) ^ n) ) = ∑' n, (1 / y) ^ (n + 1) → y = 19 / 3 :=
by
  sorry

end series_product_solution_l121_121611


namespace distinct_solutions_diff_l121_121189

theorem distinct_solutions_diff {r s : ℝ} 
  (h1 : ∀ a b : ℝ, (a ≠ b → ( ∃ x, x ≠ a ∧ x ≠ b ∧ (x = r ∨ x = s) ∧ (x-5)(x+5) = 25*x - 125) )) 
  (h2 : r > s) : r - s = 15 :=
by
  sorry

end distinct_solutions_diff_l121_121189


namespace magnitude_of_a_plus_b_in_range_l121_121158

noncomputable def a : ℝ × ℝ := (1, 1)
noncomputable def b (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
noncomputable def theta_domain : Set ℝ := {θ : ℝ | -Real.pi / 2 < θ ∧ θ < Real.pi / 2}

open Real

theorem magnitude_of_a_plus_b_in_range (θ : ℝ) (hθ : θ ∈ theta_domain) :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (cos θ, sin θ)
  1 < sqrt ((a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2) ∧ sqrt ((a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2) < (3 + 2 * sqrt 2) :=
sorry

end magnitude_of_a_plus_b_in_range_l121_121158


namespace exercise_l121_121234

-- Define a, b, and the identity
def a : ℕ := 45
def b : ℕ := 15

-- Theorem statement
theorem exercise :
  ((a + b) ^ 2 - (a ^ 2 + b ^ 2) = 1350) :=
by
  sorry

end exercise_l121_121234


namespace average_first_15_even_numbers_l121_121229

theorem average_first_15_even_numbers : 
  (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24 + 26 + 28 + 30) / 15 = 16 :=
by 
  sorry

end average_first_15_even_numbers_l121_121229


namespace tom_first_part_speed_l121_121226

theorem tom_first_part_speed 
  (total_distance : ℕ)
  (distance_first_part : ℕ)
  (speed_second_part : ℕ)
  (average_speed : ℕ)
  (total_time : ℕ)
  (distance_remaining : ℕ)
  (T2 : ℕ)
  (v : ℕ) :
  total_distance = 80 →
  distance_first_part = 30 →
  speed_second_part = 50 →
  average_speed = 40 →
  total_time = 2 →
  distance_remaining = 50 →
  T2 = 1 →
  total_time = distance_first_part / v + T2 →
  v = 30 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  -- Here, we need to prove that v = 30 given the above conditions.
  sorry

end tom_first_part_speed_l121_121226


namespace part_one_part_two_l121_121142

-- Given that tan α = 2, prove that the following expressions are correct:

theorem part_one (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (Real.pi - α) + Real.cos (α - Real.pi / 2) - Real.cos (3 * Real.pi + α)) / 
  (Real.cos (Real.pi / 2 + α) - Real.sin (2 * Real.pi + α) + 2 * Real.sin (α - Real.pi / 2)) = 
  -5 / 6 := 
by
  -- Proof skipped
  sorry

theorem part_two (α : ℝ) (h : Real.tan α = 2) :
  Real.cos (2 * α) + Real.sin α * Real.cos α = -1 / 5 := 
by
  -- Proof skipped
  sorry

end part_one_part_two_l121_121142


namespace biscuits_per_dog_l121_121516

-- Define constants for conditions
def total_biscuits : ℕ := 6
def number_of_dogs : ℕ := 2

-- Define the statement to prove
theorem biscuits_per_dog : total_biscuits / number_of_dogs = 3 := by
  -- Calculation here
  sorry

end biscuits_per_dog_l121_121516


namespace males_listen_l121_121377

theorem males_listen (males_dont_listen females_listen total_listen total_dont_listen : ℕ) 
  (h1 : males_dont_listen = 70)
  (h2 : females_listen = 75)
  (h3 : total_listen = 180)
  (h4 : total_dont_listen = 120) :
  ∃ m, m = 105 :=
by {
  sorry
}

end males_listen_l121_121377


namespace inequality_solution_set_l121_121826

theorem inequality_solution_set :
  {x : ℝ | (x^2 + 2*x + 2) / (x + 2) > 1} = {x : ℝ | (-2 < x ∧ x < -1) ∨ (0 < x)} :=
sorry

end inequality_solution_set_l121_121826


namespace fraction_equality_l121_121202

theorem fraction_equality
  (a b c d : ℝ) 
  (h1 : b ≠ c)
  (h2 : (a * c - b^2) / (a - 2 * b + c) = (b * d - c^2) / (b - 2 * c + d)) : 
  (a * c - b^2) / (a - 2 * b + c) = (a * d - b * c) / (a - b - c + d) ∧
  (b * d - c^2) / (b - 2 * c + d) = (a * d - b * c) / (a - b - c + d) := 
by
  sorry

end fraction_equality_l121_121202


namespace switch_connections_l121_121549

theorem switch_connections (n : ℕ) (d : ℕ) (h1 : n = 30) (h2 : d = 4) :
  (n * d) / 2 = 60 :=
by
  rw [h1, h2]
  norm_num
  sorry

end switch_connections_l121_121549


namespace tan_difference_l121_121167

theorem tan_difference (α β : ℝ) (hα : Real.tan α = 5) (hβ : Real.tan β = 3) : 
    Real.tan (α - β) = 1 / 8 := 
by 
  sorry

end tan_difference_l121_121167


namespace primes_pos_int_solutions_l121_121038

theorem primes_pos_int_solutions 
  (p : ℕ) [hp : Fact (Nat.Prime p)] (a b : ℕ) (h1 : ∃ k : ℤ, (4 * a + p : ℤ) + k * (4 * b + p : ℤ) = b * k * a)
  (h2 : ∃ m : ℤ, (a^2 : ℤ) + m * (b^2 : ℤ) = b * m * a) : a = b ∨ a = b * p :=
  sorry

end primes_pos_int_solutions_l121_121038


namespace find_a_l121_121320

theorem find_a 
  (a b c : ℚ) 
  (h1 : b = 4 * a) 
  (h2 : b = 15 - 4 * a - c) 
  (h3 : c = a + 2) : 
  a = 13 / 9 := 
by 
  sorry

end find_a_l121_121320


namespace sum_of_external_angles_of_octagon_staircase_slope_reduction_l121_121895

-- Problem A
theorem sum_of_external_angles_of_octagon : ∑ x in (fin 8).map (λ i, external_angles_octagon i), x = 360 :=
by sorry

-- Problem B
theorem staircase_slope_reduction :
  abs ((2.7 / real.sin (35 * real.pi / 180)) - (2.7 / real.sin (46 * real.pi / 180)) - 0.95) < 0.01 :=
by sorry

end sum_of_external_angles_of_octagon_staircase_slope_reduction_l121_121895


namespace real_estate_commission_l121_121568

theorem real_estate_commission (commission_rate commission selling_price : ℝ) 
  (h1 : commission_rate = 0.06) 
  (h2 : commission = 8880) : 
  selling_price = 148000 :=
by
  sorry

end real_estate_commission_l121_121568


namespace radius_of_circumscribed_circle_of_right_triangle_l121_121908

theorem radius_of_circumscribed_circle_of_right_triangle 
  (a b c : ℝ)
  (h_area : (1 / 2) * a * b = 10)
  (h_inradius : (a + b - c) / 2 = 1)
  (h_hypotenuse : c = Real.sqrt (a^2 + b^2)) :
  c / 2 = 4.5 := 
sorry

end radius_of_circumscribed_circle_of_right_triangle_l121_121908


namespace samantha_birth_year_l121_121367

theorem samantha_birth_year 
  (first_amc8 : ℕ)
  (amc8_annual : ∀ n : ℕ, n ≥ first_amc8)
  (seventh_amc8 : ℕ)
  (samantha_age : ℕ)
  (samantha_birth_year : ℕ)
  (move_year : ℕ)
  (h1 : first_amc8 = 1983)
  (h2 : seventh_amc8 = first_amc8 + 6)
  (h3 : seventh_amc8 = 1989)
  (h4 : samantha_age = 14)
  (h5 : samantha_birth_year = seventh_amc8 - samantha_age)
  (h6 : move_year = seventh_amc8 - 3) :
  samantha_birth_year = 1975 :=
sorry

end samantha_birth_year_l121_121367


namespace mike_total_spending_is_correct_l121_121891

-- Definitions for the costs of the items
def cost_marbles : ℝ := 9.05
def cost_football : ℝ := 4.95
def cost_baseball : ℝ := 6.52
def cost_toy_car : ℝ := 3.75
def cost_puzzle : ℝ := 8.99
def cost_stickers : ℝ := 1.25

-- Definitions for the discounts
def discount_puzzle : ℝ := 0.15
def discount_toy_car : ℝ := 0.10

-- Definition for the coupon
def coupon_amount : ℝ := 5.00

-- Total spent by Mike on toys
def total_spent : ℝ :=
  cost_marbles + 
  cost_football + 
  cost_baseball + 
  (cost_toy_car - cost_toy_car * discount_toy_car) + 
  (cost_puzzle - cost_puzzle * discount_puzzle) + 
  cost_stickers - 
  coupon_amount

-- Proof statement
theorem mike_total_spending_is_correct : 
  total_spent = 27.7865 :=
by
  sorry

end mike_total_spending_is_correct_l121_121891


namespace stuffed_animals_total_l121_121889

theorem stuffed_animals_total :
  let McKenna := 34
  let Kenley := 2 * McKenna
  let Tenly := Kenley + 5
  McKenna + Kenley + Tenly = 175 :=
by
  sorry

end stuffed_animals_total_l121_121889


namespace smallest_integer_ends_in_3_and_divisible_by_5_l121_121782

theorem smallest_integer_ends_in_3_and_divisible_by_5 : ∃ (n : ℕ), n > 0 ∧ (n % 10 = 3) ∧ (n % 5 = 0) ∧ n = 53 :=
by
  sorry

end smallest_integer_ends_in_3_and_divisible_by_5_l121_121782


namespace simplify_expression_l121_121293

theorem simplify_expression : ((1 + 2 + 3 + 4 + 5 + 6) / 3 + (3 * 5 + 12) / 4) = 13.75 :=
by
-- Proof steps would go here, but we replace them with 'sorry' for now.
sorry

end simplify_expression_l121_121293


namespace units_digit_3968_805_l121_121447

theorem units_digit_3968_805 : 
  (3968 ^ 805) % 10 = 8 := 
by
  -- Proof goes here
  sorry

end units_digit_3968_805_l121_121447


namespace banana_group_size_l121_121773

theorem banana_group_size (bananas groups : ℕ) (h1 : bananas = 407) (h2 : groups = 11) : bananas / groups = 37 :=
by sorry

end banana_group_size_l121_121773


namespace total_stuffed_animals_l121_121884

theorem total_stuffed_animals (M K T : ℕ) 
  (hM : M = 34) 
  (hK : K = 2 * M) 
  (hT : T = K + 5) : 
  M + K + T = 175 :=
by
  -- Adding sorry to complete the placeholder
  sorry

end total_stuffed_animals_l121_121884


namespace power_eval_l121_121992

theorem power_eval : (9^6 * 3^4) / (27^5) = 3 := by
  sorry

end power_eval_l121_121992


namespace sum_of_factors_of_1000_l121_121535

-- Define what it means for an integer to not contain the digit '0'
def no_zero_digits (n : ℕ) : Prop :=
∀ c ∈ (n.digits 10), c ≠ 0

-- Define the problem statement
theorem sum_of_factors_of_1000 :
  ∃ (a b : ℕ), a * b = 1000 ∧ no_zero_digits a ∧ no_zero_digits b ∧ (a + b = 133) :=
sorry

end sum_of_factors_of_1000_l121_121535


namespace room_length_calculation_l121_121172

-- Definitions of the problem conditions
def room_volume : ℝ := 10000
def room_width : ℝ := 10
def room_height : ℝ := 10

-- Statement to prove
theorem room_length_calculation : ∃ L : ℝ, L = room_volume / (room_width * room_height) ∧ L = 100 :=
by
  sorry

end room_length_calculation_l121_121172


namespace prime_divisor_condition_l121_121409

theorem prime_divisor_condition (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hdiv : q ∣ 2^p - 1) : p ∣ q - 1 :=
  sorry

end prime_divisor_condition_l121_121409


namespace joseph_power_cost_ratio_l121_121720

theorem joseph_power_cost_ratio
  (electric_oven_cost : ℝ)
  (total_cost : ℝ)
  (water_heater_cost : ℝ)
  (refrigerator_cost : ℝ)
  (H1 : electric_oven_cost = 500)
  (H2 : 2 * water_heater_cost = electric_oven_cost)
  (H3 : refrigerator_cost + water_heater_cost + electric_oven_cost = total_cost)
  (H4 : total_cost = 1500):
  (refrigerator_cost / water_heater_cost) = 3 := sorry

end joseph_power_cost_ratio_l121_121720


namespace hands_straight_line_time_l121_121122

noncomputable def time_when_hands_straight_line : List (ℕ × ℚ) :=
  let x₁ := 21 + 9 / 11
  let x₂ := 54 + 6 / 11
  [(4, x₁), (4, x₂)]

theorem hands_straight_line_time :
  time_when_hands_straight_line = [(4, 21 + 9 / 11), (4, 54 + 6 / 11)] :=
by
  sorry

end hands_straight_line_time_l121_121122


namespace find_y_l121_121145

theorem find_y (y : ℝ) (h : (y - 8) / (5 - (-3)) = -5 / 4) : y = -2 :=
by sorry

end find_y_l121_121145


namespace sum_of_transformed_numbers_l121_121770

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) : 3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end sum_of_transformed_numbers_l121_121770


namespace arithmetic_sequence_sum_S15_l121_121004

theorem arithmetic_sequence_sum_S15 (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hs5 : S 5 = 10) (hs10 : S 10 = 30) 
  (has : ∀ n, S n = n * (2 * a 1 + (n - 1) * a 2) / 2) : 
  S 15 = 60 := 
sorry

end arithmetic_sequence_sum_S15_l121_121004


namespace cos_double_angle_l121_121665

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3 / 5) : Real.cos (2 * theta) = -7 / 25 :=
by
  sorry

end cos_double_angle_l121_121665


namespace magic_king_episodes_proof_l121_121929

-- Let's state the condition in terms of the number of seasons and episodes:
def total_episodes (seasons: ℕ) (episodes_first_half: ℕ) (episodes_second_half: ℕ) : ℕ :=
  (seasons / 2) * episodes_first_half + (seasons / 2) * episodes_second_half

-- Define the conditions for the "Magic King" show
def magic_king_total_episodes : ℕ :=
  total_episodes 10 20 25

-- The statement of the problem - to prove that the total episodes is 225
theorem magic_king_episodes_proof : magic_king_total_episodes = 225 :=
by
  sorry

end magic_king_episodes_proof_l121_121929


namespace lower_bound_of_range_of_expression_l121_121458

theorem lower_bound_of_range_of_expression :
  ∃ L, (∀ n : ℤ, L < 4*n + 7 → 4*n + 7 < 100) ∧
  (∃! n_min n_max : ℤ, 4*n_min + 7 = L ∧ 4*n_max + 7 = 99 ∧ (n_max - n_min + 1 = 25)) :=
sorry

end lower_bound_of_range_of_expression_l121_121458


namespace number_of_subsets_with_four_adj_chairs_l121_121386

-- Definition of the problem constants
def n : ℕ := 12

-- Define the condition that our chairs are arranged in a circle with 12 chairs
def is_adjacent (i j : ℕ) : Prop :=
  (j = (i + 1) % n) ∨ (i = (j + 1) % n) 

-- Define what it means for a subset to have at least four adjacent chairs
def at_least_four_adjacent (s : Finset ℕ) : Prop :=
  ∃ i j k l, i ∈ s ∧ j ∈ s ∧ k ∈ s ∧ l ∈ s ∧ is_adjacent i j ∧ is_adjacent j k ∧ is_adjacent k l

-- The main theorem to prove
theorem number_of_subsets_with_four_adj_chairs : ∃ k, k = 1701 ∧ ∀ s : Finset ℕ, s.card ≤ n → at_least_four_adjacent s →
  (∃ t, t.card = 4 ∧ t ⊆ s ∧ at_least_four_adjacent t) := 
sorry

end number_of_subsets_with_four_adj_chairs_l121_121386


namespace percentage_of_y_in_relation_to_25_percent_of_x_l121_121055

variable (y x : ℕ) (p : ℕ)

-- Conditions
def condition1 : Prop := (y = (p * 25 * x) / 10000)
def condition2 : Prop := (y * x = 100 * 100)
def condition3 : Prop := (y = 125)

-- The proof goal
theorem percentage_of_y_in_relation_to_25_percent_of_x :
  condition1 y x p ∧ condition2 y x ∧ condition3 y → ((y * 100) / (25 * x / 100) = 625)
:= by
-- Here we would insert the proof steps, but they are omitted as per the requirements.
sorry

end percentage_of_y_in_relation_to_25_percent_of_x_l121_121055


namespace hexagon_perimeter_l121_121980

theorem hexagon_perimeter
  (A B C D E F : Type)  -- vertices of the hexagon
  (angle_A : ℝ) (angle_C : ℝ) (angle_E : ℝ)  -- nonadjacent angles
  (angle_B : ℝ) (angle_D : ℝ) (angle_F : ℝ)  -- adjacent angles
  (area_hexagon : ℝ)
  (side_length : ℝ)
  (h1 : angle_A = 120) (h2 : angle_C = 120) (h3 : angle_E = 120)
  (h4 : angle_B = 60) (h5 : angle_D = 60) (h6 : angle_F = 60)
  (h7 : area_hexagon = 24)
  (h8 : ∃ s, ∀ (u v : Type), side_length = s) :
  6 * side_length = 24 / (Real.sqrt 3 ^ (1/4)) :=
by
  sorry

end hexagon_perimeter_l121_121980


namespace minimum_shirts_for_saving_money_l121_121809

-- Define the costs for Acme and Gamma
def acme_cost (x : ℕ) : ℕ := 60 + 10 * x
def gamma_cost (x : ℕ) : ℕ := 15 * x

-- Prove that the minimum number of shirts x for which a customer saves money by using Acme is 13
theorem minimum_shirts_for_saving_money : ∃ (x : ℕ), 60 + 10 * x < 15 * x ∧ x = 13 := by
  sorry

end minimum_shirts_for_saving_money_l121_121809


namespace correct_ordering_of_f_values_l121_121100

variable {f : ℝ → ℝ}

theorem correct_ordering_of_f_values
  (hf_even : ∀ x : ℝ, f (-x) = f x)
  (hf_decreasing : ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ > f x₂) :
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end correct_ordering_of_f_values_l121_121100


namespace problem_statement_problem_statement_2_l121_121308

noncomputable def A (m : ℝ) : Set ℝ := {x | x > 2^m}
noncomputable def B : Set ℝ := {x | -4 < x - 4 ∧ x - 4 < 4}

theorem problem_statement (m : ℝ) (h1 : m = 2) :
  (A m ∪ B = {x | x > 0}) ∧ (A m ∩ B = {x | 4 < x ∧ x < 8}) :=
by sorry

theorem problem_statement_2 (m : ℝ) (h2 : A m ⊆ {x | x ≤ 0 ∨ 8 ≤ x}) :
  3 ≤ m :=
by sorry

end problem_statement_problem_statement_2_l121_121308


namespace f_inv_f_inv_15_l121_121364

def f (x : ℝ) : ℝ := 3 * x + 6

noncomputable def f_inv (x : ℝ) : ℝ := (x - 6) / 3

theorem f_inv_f_inv_15 : f_inv (f_inv 15) = -1 :=
by
  sorry

end f_inv_f_inv_15_l121_121364


namespace solution_set_f_less_x_plus_1_l121_121966

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_continuous : Continuous f
axiom f_at_1 : f 1 = 2
axiom f_derivative : ∀ x, deriv f x < 1

theorem solution_set_f_less_x_plus_1 : 
  ∀ x : ℝ, (f x < x + 1) ↔ (x > 1) :=
by
  sorry

end solution_set_f_less_x_plus_1_l121_121966


namespace train_speed_is_60_kmph_l121_121263

-- Define the conditions
def time_to_cross_pole_seconds : ℚ := 36
def length_of_train_meters : ℚ := 600

-- Define the conversion factors
def seconds_per_hour : ℚ := 3600
def meters_per_kilometer : ℚ := 1000

-- Convert the conditions to appropriate units
def time_to_cross_pole_hours : ℚ := time_to_cross_pole_seconds / seconds_per_hour
def length_of_train_kilometers : ℚ := length_of_train_meters / meters_per_kilometer

-- Prove that the speed of the train in km/hr is 60
theorem train_speed_is_60_kmph : 
  (length_of_train_kilometers / time_to_cross_pole_hours) = 60 := 
by
  sorry

end train_speed_is_60_kmph_l121_121263


namespace false_statement_l121_121532

noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^2 + Real.sqrt 3 * Real.sin x * Real.cos x

def p : Prop := ∃ x0 : ℝ, f x0 = -1
def q : Prop := ∀ x : ℝ, f (2 * Real.pi + x) = f x

theorem false_statement : ¬ (p ∧ q) := sorry

end false_statement_l121_121532


namespace circle_chairs_subsets_count_l121_121398

theorem circle_chairs_subsets_count :
  ∃ (n : ℕ), n = 12 → set.count (λ s : finset ℕ, s.card ≥ 4 ∧ ∀ i ∈ s, (i + 1) % 12 ∈ s) {s | s ⊆ finset.range 12} = 1712 := 
by
  sorry

end circle_chairs_subsets_count_l121_121398


namespace income_to_expenditure_ratio_l121_121210

theorem income_to_expenditure_ratio (I E S : ℝ) (hI : I = 10000) (hS : S = 4000) (hSavings : S = I - E) : I / E = 5 / 3 := by
  -- To prove: I / E = 5 / 3 given hI, hS, and hSavings
  sorry

end income_to_expenditure_ratio_l121_121210


namespace probability_six_distinct_numbers_l121_121083

theorem probability_six_distinct_numbers :
  let total_outcomes := 6^6
  let distinct_outcomes := Nat.factorial 6
  let probability := (distinct_outcomes:ℚ) / (total_outcomes:ℚ)
  probability = 5 / 324 :=
sorry

end probability_six_distinct_numbers_l121_121083


namespace geom_seq_m_equals_11_l121_121176

noncomputable def geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) :=
  ∀ (n : ℕ), a n = a1 * q ^ n

theorem geom_seq_m_equals_11 {a : ℕ → ℝ} {q : ℝ} (hq : q ≠ 1) 
  (h : geometric_sequence a 1 q) : 
  a 11 = a 1 * a 2 * a 3 * a 4 * a 5 := 
by sorry

end geom_seq_m_equals_11_l121_121176


namespace binomial_probability_l121_121632

open ProbabilityTheory

theorem binomial_probability (X : ℕ → ℕ) (hX : Binomial 6 (1/3) X) : 
  P (X = 2) = 80 / 243 := 
sorry

end binomial_probability_l121_121632


namespace binom_divisibility_l121_121737

theorem binom_divisibility (p : ℕ) (h₀ : Nat.Prime p) (h₁ : p % 2 = 1) : 
  (Nat.choose (2 * p - 1) (p - 1) - 1) % (p^2) = 0 := 
by 
  sorry

end binom_divisibility_l121_121737


namespace triangle_AC_5_sqrt_3_l121_121334

theorem triangle_AC_5_sqrt_3 
  (A B C : ℝ)
  (BC AC : ℝ)
  (h1 : 2 * Real.sin (A - B) + Real.cos (B + C) = 2)
  (h2 : BC = 5) :
  AC = 5 * Real.sqrt 3 :=
  sorry

end triangle_AC_5_sqrt_3_l121_121334


namespace magic_king_episodes_proof_l121_121928

-- Let's state the condition in terms of the number of seasons and episodes:
def total_episodes (seasons: ℕ) (episodes_first_half: ℕ) (episodes_second_half: ℕ) : ℕ :=
  (seasons / 2) * episodes_first_half + (seasons / 2) * episodes_second_half

-- Define the conditions for the "Magic King" show
def magic_king_total_episodes : ℕ :=
  total_episodes 10 20 25

-- The statement of the problem - to prove that the total episodes is 225
theorem magic_king_episodes_proof : magic_king_total_episodes = 225 :=
by
  sorry

end magic_king_episodes_proof_l121_121928


namespace age_ratio_l121_121869

/-- 
Axiom: Kareem's age is 42 and his son's age is 14. 
-/
axiom Kareem_age : ℕ
axiom Son_age : ℕ

/-- 
Conditions: 
  - Kareem's age after 10 years plus his son's age after 10 years equals 76.
  - Kareem's current age is 42.
  - His son's current age is 14.
-/
axiom age_condition : Kareem_age + 10 + Son_age + 10 = 76
axiom Kareem_current_age : Kareem_age = 42
axiom Son_current_age : Son_age = 14

/-- 
Theorem: The ratio of Kareem's age to his son's age is 3:1.
-/
theorem age_ratio : Kareem_age / Son_age = 3 / 1 := by {
  -- Proof skipped
  sorry 
}

end age_ratio_l121_121869


namespace problem_1_problem_2_l121_121307

-- Definition of sets A and B as in the problem's conditions
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | x > 2 ∨ x < -2}
def C (a : ℝ) : Set ℝ := {x | x ≤ a - 1}

-- Prove that A ∩ B is as described
theorem problem_1 : A ∩ B = {x | 2 < x ∧ x ≤ 5} := by
  sorry

-- Prove that a ≥ 6 given the conditions in the problem
theorem problem_2 (a : ℝ) : (A ⊆ C a) → a ≥ 6 := by
  sorry

end problem_1_problem_2_l121_121307


namespace total_chocolate_pieces_l121_121417

def total_chocolates (boxes : ℕ) (per_box : ℕ) : ℕ :=
  boxes * per_box

theorem total_chocolate_pieces :
  total_chocolates 6 500 = 3000 :=
by
  sorry

end total_chocolate_pieces_l121_121417


namespace median_of_64_consecutive_integers_l121_121373

theorem median_of_64_consecutive_integers (n : ℕ) (S : ℕ) (h1 : n = 64) (h2 : S = 8^4) :
  S / n = 64 :=
by
  -- to skip the proof
  sorry

end median_of_64_consecutive_integers_l121_121373


namespace find_tangent_point_and_slope_l121_121988

theorem find_tangent_point_and_slope :
  ∃ m n : ℝ, (m = 1 ∧ n = Real.exp 1 ∧ 
    (∀ x y : ℝ, y - n = (Real.exp m) * (x - m) → x = 0 ∧ y = 0) ∧ 
    (Real.exp m = Real.exp 1)) :=
sorry

end find_tangent_point_and_slope_l121_121988


namespace term_of_arithmetic_sequence_l121_121941

variable (a₁ : ℕ) (d : ℕ) (n : ℕ)

theorem term_of_arithmetic_sequence (h₁: a₁ = 2) (h₂: d = 5) (h₃: n = 50) :
    a₁ + (n - 1) * d = 247 := by
  sorry

end term_of_arithmetic_sequence_l121_121941


namespace smallest_positive_whole_number_divisible_by_first_five_primes_l121_121944

def is_prime (n : Nat) : Prop := Nat.Prime n

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

def smallest_positive_divisible (lst : List Nat) : Nat :=
  List.foldl (· * ·) 1 lst

theorem smallest_positive_whole_number_divisible_by_first_five_primes :
  smallest_positive_divisible first_five_primes = 2310 := by
  sorry

end smallest_positive_whole_number_divisible_by_first_five_primes_l121_121944


namespace total_nails_polished_l121_121970

-- Defining the number of girls
def num_girls : ℕ := 5

-- Defining the number of fingers and toes per person
def num_fingers_per_person : ℕ := 10
def num_toes_per_person : ℕ := 10

-- Defining the total number of nails per person
def nails_per_person : ℕ := num_fingers_per_person + num_toes_per_person

-- The theorem stating that the total number of nails polished for 5 girls is 100 nails
theorem total_nails_polished : num_girls * nails_per_person = 100 := by
  sorry

end total_nails_polished_l121_121970


namespace katrina_cookies_left_l121_121341

/-- Katrina’s initial number of cookies. -/
def initial_cookies : ℕ := 120

/-- Cookies sold in the morning. 
    1 dozen is 12 cookies, so 3 dozen is 36 cookies. -/
def morning_sales : ℕ := 3 * 12

/-- Cookies sold during the lunch rush. -/
def lunch_sales : ℕ := 57

/-- Cookies sold in the afternoon. -/
def afternoon_sales : ℕ := 16

/-- Calculate the number of cookies left after all sales. -/
def cookies_left : ℕ :=
  initial_cookies - morning_sales - lunch_sales - afternoon_sales

/-- Prove that the number of cookies left for Katrina to take home is 11. -/
theorem katrina_cookies_left : cookies_left = 11 := by
  sorry

end katrina_cookies_left_l121_121341


namespace percentage_reduction_is_20_percent_l121_121962

-- Defining the initial and final prices
def initial_price : ℝ := 25
def final_price : ℝ := 16

-- Defining the percentage reduction
def percentage_reduction (x : ℝ) := 1 - x

-- The equation representing the two reductions:
def equation (x : ℝ) := initial_price * (percentage_reduction x) * (percentage_reduction x)

theorem percentage_reduction_is_20_percent :
  ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ equation x = final_price ∧ x = 0.20 :=
by 
  sorry

end percentage_reduction_is_20_percent_l121_121962


namespace shortest_player_height_l121_121540

theorem shortest_player_height :
  ∀ (tallest_height difference : ℝ), 
    tallest_height = 77.75 ∧ difference = 9.5 → 
    tallest_height - difference = 68.25 :=
by
  intros tallest_height difference h
  cases h
  sorry

end shortest_player_height_l121_121540


namespace crayons_left_l121_121044

def initial_green_crayons : ℝ := 5
def initial_blue_crayons : ℝ := 8
def initial_yellow_crayons : ℝ := 7
def given_green_crayons : ℝ := 3.5
def given_blue_crayons : ℝ := 1.25
def given_yellow_crayons : ℝ := 2.75
def broken_yellow_crayons : ℝ := 0.5

theorem crayons_left (initial_green_crayons initial_blue_crayons initial_yellow_crayons given_green_crayons given_blue_crayons given_yellow_crayons broken_yellow_crayons : ℝ) :
  initial_green_crayons - given_green_crayons + 
  initial_blue_crayons - given_blue_crayons + 
  initial_yellow_crayons - given_yellow_crayons - broken_yellow_crayons = 12 :=
by
  sorry

end crayons_left_l121_121044


namespace expression_value_l121_121239

theorem expression_value (a b : ℕ) (ha : a = 45) (hb : b = 15) : (a + b)^2 - (a^2 + b^2) = 1350 := by 
  sorry

end expression_value_l121_121239


namespace june_time_to_bernard_l121_121185

theorem june_time_to_bernard (distance_Julia : ℝ) (time_Julia : ℝ) (distance_Bernard_June : ℝ) (time_Bernard : ℝ) (distance_June_Bernard : ℝ)
  (h1 : distance_Julia = 2) (h2 : time_Julia = 6) (h3 : distance_Bernard_June = 5) (h4 : time_Bernard = 15) (h5 : distance_June_Bernard = 7) :
  distance_June_Bernard / (distance_Julia / time_Julia) = 21 := by
    sorry

end june_time_to_bernard_l121_121185


namespace ivans_profit_l121_121506

def price_meat_per_kg : ℕ := 500
def kg_meat_sold : ℕ := 100
def price_eggs_per_dozen : ℕ := 50
def eggs_sold : ℕ := 20000
def annual_expenses : ℕ := 100000

def revenue_meat : ℕ := kg_meat_sold * price_meat_per_kg
def revenue_eggs : ℕ := eggs_sold * (price_eggs_per_dozen / 10)
def total_revenue : ℕ := revenue_meat + revenue_eggs

def profit : ℕ := total_revenue - annual_expenses

theorem ivans_profit : profit = 50000 := by
  sorry

end ivans_profit_l121_121506


namespace average_marks_l121_121526

-- Definitions
def Tatuya_score (Ivanna_score : ℕ) : ℕ := 2 * Ivanna_score
def Ivanna_score (Dorothy_score : ℕ) : ℕ := (3 * Dorothy_score) / 5
def Dorothy_score : ℕ := 90

-- Theorem statement
theorem average_marks :
  let Dorothy_score := Dorothy_score in
  let Ivanna_score := Ivanna_score Dorothy_score in
  let Tatuya_score := Tatuya_score Ivanna_score in
  (Dorothy_score + Ivanna_score + Tatuya_score) / 3 = 84 :=
by 
  -- Proof goes here
  sorry

end average_marks_l121_121526


namespace volume_of_rock_correct_l121_121801

-- Define the initial conditions
def tank_length := 30
def tank_width := 20
def water_depth := 8
def water_level_rise := 4

-- Define the volume function for the rise in water level
def calculate_volume_of_rise (length: ℕ) (width: ℕ) (rise: ℕ) : ℕ :=
  length * width * rise

-- Define the target volume of the rock
def volume_of_rock := 2400

-- The theorem statement that the volume of the rock is 2400 cm³
theorem volume_of_rock_correct :
  calculate_volume_of_rise tank_length tank_width water_level_rise = volume_of_rock :=
by 
  sorry

end volume_of_rock_correct_l121_121801


namespace abs_inequality_solution_l121_121741

theorem abs_inequality_solution :
  { x : ℝ | |x - 2| + |x + 3| < 6 } = { x | -7 / 2 < x ∧ x < 5 / 2 } :=
by
  sorry

end abs_inequality_solution_l121_121741


namespace find_c_l121_121489

noncomputable def condition1 (a b c : ℝ) : Prop :=
  a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)

noncomputable def condition2 (c : ℝ) : Prop :=
  6 * 15 * c = 1

theorem find_c (c : ℝ) (h1 : condition1 6 15 c) (h2 : condition2 c) : c = 11 := 
by
  sorry

end find_c_l121_121489


namespace stratified_sampling_grade_10_l121_121536

theorem stratified_sampling_grade_10 (x y z x_s : ℕ)
  (h1 : x = 2 * z)
  (h2 : y = 2 * z)
  (h3 : 45 * x / (x + y + z) = x_s) :
  x_s = 18 := by
  sorry

end stratified_sampling_grade_10_l121_121536


namespace walking_speed_l121_121799

theorem walking_speed 
  (D : ℝ) 
  (V_w : ℝ) 
  (h1 : D = V_w * 8) 
  (h2 : D = 36 * 2) : 
  V_w = 9 :=
by
  sorry

end walking_speed_l121_121799


namespace necessary_not_sufficient_condition_l121_121016

-- Define the necessary conditions for the equation to represent a hyperbola
def represents_hyperbola (k : ℝ) : Prop :=
  k > 5 ∨ k < -2

-- Define the condition for k
axiom k_in_real (k : ℝ) : Prop

-- The proof statement
theorem necessary_not_sufficient_condition (k : ℝ) (hk : k_in_real k) :
  (∃ (k_val : ℝ), k_val > 5 ∧ k = k_val) → represents_hyperbola k ∧ ¬ (represents_hyperbola k → k > 5) :=
by
  sorry

end necessary_not_sufficient_condition_l121_121016


namespace minimum_value_problem_l121_121878

theorem minimum_value_problem (a b c : ℝ) (hb : a > 0 ∧ b > 0 ∧ c > 0)
  (h : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 10) : 
  ∃ x, (x = 47) ∧ (a / b + b / c + c / a) * (b / a + c / b + a / c) ≥ x :=
by
  sorry

end minimum_value_problem_l121_121878


namespace octagon_reflected_arcs_area_l121_121262

theorem octagon_reflected_arcs_area :
  let s := 2
  let θ := 45
  let r := 2 / Real.sqrt (2 - Real.sqrt (2))
  let sector_area := θ / 360 * Real.pi * r^2
  let total_arc_area := 8 * sector_area
  let circle_area := Real.pi * r^2
  let bounded_region_area := 8 * (circle_area - 2 * Real.sqrt (2) * 1 / 2)
  bounded_region_area = (16 * Real.sqrt 2 / 3 - Real.pi)
:= sorry

end octagon_reflected_arcs_area_l121_121262


namespace f_10_equals_1_l121_121533

noncomputable def f : ℝ → ℝ 
| x => sorry 

axiom odd_f_x_minus_1 : ∀ x : ℝ, f (-x-1) = -f (x-1)
axiom even_f_x_plus_1 : ∀ x : ℝ, f (-x+1) = f (x+1)
axiom f_definition : ∀ x : ℝ, 0 ≤ x ∧ x < 1 → f x = 2^x

theorem f_10_equals_1 : f 10 = 1 :=
by
  sorry -- The actual proof goes here.

end f_10_equals_1_l121_121533


namespace passenger_waiting_time_probability_l121_121579

def bus_arrival_interval : ℕ := 5

def waiting_time_limit : ℕ := 3

/-- 
  Prove that for a bus arriving every 5 minutes,
  the probability that a passenger's waiting time 
  is no more than 3 minutes, given the passenger 
  arrives at a random time, is 3/5. 
--/
theorem passenger_waiting_time_probability 
  (bus_interval : ℕ) (time_limit : ℕ) 
  (random_arrival : ℝ) :
  bus_interval = 5 →
  time_limit = 3 →
  0 ≤ random_arrival ∧ random_arrival < bus_interval →
  (random_arrival ≤ time_limit) →
  (random_arrival / ↑bus_interval) = 3 / 5 :=
by
  sorry

end passenger_waiting_time_probability_l121_121579


namespace cos_double_angle_l121_121653

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (2 * θ) = -7 / 25 := 
by
  sorry

end cos_double_angle_l121_121653


namespace inverse_proportion_quad_l121_121020

theorem inverse_proportion_quad (k : ℝ) : (∀ x : ℝ, x > 0 → (k + 1) / x < 0) ∧ (∀ x : ℝ, x < 0 → (k + 1) / x > 0) ↔ k < -1 :=
by
  sorry

end inverse_proportion_quad_l121_121020


namespace greatest_value_NNM_l121_121453

theorem greatest_value_NNM :
  ∃ (M : ℕ), (M * M % 10 = M) ∧ (∃ (MM : ℕ), MM = 11 * M ∧ (MM * M = 396)) :=
by
  sorry

end greatest_value_NNM_l121_121453


namespace length_of_lunch_break_is_48_minutes_l121_121894

noncomputable def paula_and_assistants_lunch_break : ℝ := sorry

theorem length_of_lunch_break_is_48_minutes
  (p h L : ℝ)
  (h_monday : (9 - L) * (p + h) = 0.6)
  (h_tuesday : (7 - L) * h = 0.3)
  (h_wednesday : (10 - L) * p = 0.1) :
  L = 0.8 :=
sorry

end length_of_lunch_break_is_48_minutes_l121_121894


namespace first_operation_result_l121_121025

def pattern (x y : ℕ) : ℕ :=
  if (x, y) = (3, 7) then 27
  else if (x, y) = (4, 5) then 32
  else if (x, y) = (5, 8) then 60
  else if (x, y) = (6, 7) then 72
  else if (x, y) = (7, 8) then 98
  else 26

theorem first_operation_result : pattern 2 3 = 26 := by
  sorry

end first_operation_result_l121_121025


namespace parallel_vectors_condition_l121_121843

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ b = (k * a.1, k * a.2)

theorem parallel_vectors_condition (m : ℝ) :
  vectors_parallel (1, m + 1) (m, 2) ↔ m = -2 ∨ m = 1 := by
  sorry

end parallel_vectors_condition_l121_121843


namespace find_a_l121_121835

open Complex

theorem find_a (a : ℝ) (i : ℂ := Complex.I) (h : (a - i) ^ 2 = 2 * i) : a = -1 :=
sorry

end find_a_l121_121835


namespace smallest_integer_ends_3_divisible_5_l121_121781

theorem smallest_integer_ends_3_divisible_5 : ∃ n : ℕ, (53 = n ∧ (n % 10 = 3) ∧ (n % 5 = 0) ∧ ∀ m : ℕ, (m % 10 = 3) ∧ (m % 5 = 0) → 53 ≤ m) :=
by {
  sorry
}

end smallest_integer_ends_3_divisible_5_l121_121781


namespace unique_chair_arrangement_l121_121024

theorem unique_chair_arrangement (n : ℕ) (h : n = 49)
  (h1 : ∀ i j : ℕ, (n = i * j) → (i ≥ 2) ∧ (j ≥ 2)) :
  ∃! i j : ℕ, (n = i * j) ∧ (i ≥ 2) ∧ (j ≥ 2) :=
by
  sorry

end unique_chair_arrangement_l121_121024


namespace inequalities_quadrants_l121_121925

theorem inequalities_quadrants :
  (∀ x y : ℝ, y > 2 * x → y > 4 - x → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0)) := sorry

end inequalities_quadrants_l121_121925


namespace find_radius_l121_121127

theorem find_radius (QP QO r : ℝ) (hQP : QP = 420) (hQO : QO = 427) : r = 77 :=
by
  -- Given QP^2 + r^2 = QO^2
  have h : (QP ^ 2) + (r ^ 2) = (QO ^ 2) := sorry
  -- Calculate the squares
  have h1 : (420 ^ 2) = 176400 := sorry
  have h2 : (427 ^ 2) = 182329 := sorry
  -- r^2 = 182329 - 176400
  have h3 : r ^ 2 = 5929 := sorry
  -- Therefore, r = 77
  exact sorry

end find_radius_l121_121127


namespace find_judes_age_l121_121853

def jude_age (H : ℕ) (J : ℕ) : Prop :=
  H + 5 = 3 * (J + 5)

theorem find_judes_age : ∃ J : ℕ, jude_age 16 J ∧ J = 2 :=
by
  sorry

end find_judes_age_l121_121853


namespace tan_pi_over_12_eq_l121_121455

theorem tan_pi_over_12_eq : Real.tan (Real.pi / 12) = 2 - Real.sqrt 3 :=
by
  sorry

end tan_pi_over_12_eq_l121_121455


namespace find_last_week_rope_l121_121186

/-- 
Description: Mr. Sanchez bought 4 feet of rope less than he did the previous week. 
Given that he bought 96 inches in total, find how many feet he bought last week.
--/
theorem find_last_week_rope (F : ℕ) :
  12 * (F - 4) = 96 → F = 12 := by
  sorry

end find_last_week_rope_l121_121186


namespace total_value_of_item_l121_121086

theorem total_value_of_item (V : ℝ) (h1 : 0.07 * (V - 1000) = 87.50) :
  V = 2250 :=
by
  sorry

end total_value_of_item_l121_121086


namespace complement_correct_l121_121510

-- Define the universal set U
def U : Set ℤ := {x | -2 < x ∧ x ≤ 3}

-- Define the set A
def A : Set ℤ := {3}

-- Define the complement of A with respect to U
def complement_U_A : Set ℤ := {x | x ∈ U ∧ x ∉ A}

theorem complement_correct : complement_U_A = { -1, 0, 1, 2 } :=
by
  sorry

end complement_correct_l121_121510


namespace ratio_c_d_l121_121707

theorem ratio_c_d (x y c d : ℝ) (h1 : 4 * x - 2 * y = c)
  (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) : c / d = -1 / 3 := by
  sorry

end ratio_c_d_l121_121707


namespace find_functions_l121_121706

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def domain (f g : ℝ → ℝ) : Prop := ∀ x, x ≠ 1 → x ≠ -1 → true

theorem find_functions
  (f g : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_even : is_even g)
  (h_domain : domain f g)
  (h_eq : ∀ x, x ≠ 1 → x ≠ -1 → f x + g x = 1 / (x - 1)) :
  (∀ x, x ≠ 1 → x ≠ -1 → f x = x / (x^2 - 1)) ∧ 
  (∀ x, x ≠ 1 → x ≠ -1 → g x = 1 / (x^2 - 1)) := 
by
  sorry

end find_functions_l121_121706


namespace big_al_bananas_l121_121275

theorem big_al_bananas (total_bananas : ℕ) (a : ℕ)
  (h : total_bananas = 150)
  (h1 : a + (a + 7) + (a + 14) + (a + 21) + (a + 28) = total_bananas) :
  a + 14 = 30 :=
by
  -- Using the given conditions to prove the statement
  sorry

end big_al_bananas_l121_121275


namespace isaac_journey_time_l121_121050

def travel_time_total (speed : ℝ) (time1 : ℝ) (distance2 : ℝ) (rest_time : ℝ) (distance3 : ℝ) : ℝ :=
  let time2 := distance2 / speed
  let time3 := distance3 / speed
  time1 + time2 * 60 + rest_time + time3 * 60

theorem isaac_journey_time :
  travel_time_total 10 (30 : ℝ) 15 (30 : ℝ) 20 = 270 :=
by
  sorry

end isaac_journey_time_l121_121050


namespace second_number_l121_121452

theorem second_number (x : ℕ) (h1 : ∃ k : ℕ, 1428 = 129 * k + 9)
  (h2 : ∃ m : ℕ, x = 129 * m + 13) (h_gcd : ∀ (d : ℕ), d ∣ (1428 - 9 : ℕ) ∧ d ∣ (x - 13 : ℕ) → d ≤ 129) :
  x = 1561 :=
by
  sorry

end second_number_l121_121452


namespace total_dogs_on_farm_l121_121935

-- Definitions based on conditions from part a)
def num_dog_houses : ℕ := 5
def num_dogs_per_house : ℕ := 4

-- Statement to prove
theorem total_dogs_on_farm : num_dog_houses * num_dogs_per_house = 20 :=
by
  sorry

end total_dogs_on_farm_l121_121935


namespace shortest_player_height_l121_121539

theorem shortest_player_height :
  ∀ (tallest_height difference : ℝ), 
    tallest_height = 77.75 ∧ difference = 9.5 → 
    tallest_height - difference = 68.25 :=
by
  intros tallest_height difference h
  cases h
  sorry

end shortest_player_height_l121_121539


namespace kayak_rental_cost_l121_121081

variable (K : ℕ) -- the cost of a kayak rental per day
variable (x : ℕ) -- the number of kayaks rented

-- Conditions
def canoe_cost_per_day : ℕ := 11
def total_revenue : ℕ := 460
def canoes_more_than_kayaks : ℕ := 5

def ratio_condition : Prop := 4 * x = 3 * (x + 5)
def total_revenue_condition : Prop := canoe_cost_per_day * (x + 5) + K * x = total_revenue

-- Main statement
theorem kayak_rental_cost :
  ratio_condition x →
  total_revenue_condition K x →
  K = 16 := by sorry

end kayak_rental_cost_l121_121081


namespace circle_radius_l121_121252

theorem circle_radius (r x y : ℝ) (h1 : x = π * r^2) (h2 : y = 2 * π * r) (h3 : x + y = 180 * π) : r = 10 := 
by
  sorry

end circle_radius_l121_121252


namespace weight_of_empty_carton_l121_121541

theorem weight_of_empty_carton
    (half_full_carton_weight : ℕ)
    (full_carton_weight : ℕ)
    (h1 : half_full_carton_weight = 5)
    (h2 : full_carton_weight = 8) :
  full_carton_weight - 2 * (full_carton_weight - half_full_carton_weight) = 2 :=
by
  sorry

end weight_of_empty_carton_l121_121541


namespace diameter_of_circumscribed_circle_l121_121018

theorem diameter_of_circumscribed_circle (a : ℝ) (A : ℝ) (D : ℝ) 
  (h1 : a = 12) (h2 : A = 30) : D = 24 :=
by
  sorry

end diameter_of_circumscribed_circle_l121_121018


namespace circle_chords_integer_lengths_l121_121736

theorem circle_chords_integer_lengths (P O : ℝ) (d r : ℝ) (n : ℕ) : 
  dist P O = d → r = 20 → d = 12 → n = 9 := by
  sorry

end circle_chords_integer_lengths_l121_121736


namespace complex_fraction_simplify_l121_121015

variable (i : ℂ)
variable (h : i^2 = -1)

theorem complex_fraction_simplify :
  (1 - i) / ((1 + i) ^ 2) = -1/2 - i/2 :=
by
  sorry

end complex_fraction_simplify_l121_121015


namespace perfectSquareLastFourDigits_l121_121985

noncomputable def lastThreeDigitsForm (n : ℕ) : Prop :=
  ∃ a : ℕ, a ≤ 9 ∧ n % 1000 = a * 111

noncomputable def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem perfectSquareLastFourDigits (n : ℕ) :
  lastThreeDigitsForm n →
  isPerfectSquare n →
  (n % 10000 = 0 ∨ n % 10000 = 1444) :=
by {
  sorry
}

end perfectSquareLastFourDigits_l121_121985


namespace product_of_local_and_absolute_value_l121_121824

def localValue (n : ℕ) (digit : ℕ) : ℕ :=
  match n with
  | 564823 =>
    match digit with
    | 4 => 4000
    | _ => 0 -- only defining for digit 4 as per problem
  | _ => 0 -- only case for 564823 is considered

def absoluteValue (x : ℤ) : ℤ := if x < 0 then -x else x

theorem product_of_local_and_absolute_value:
  localValue 564823 4 * absoluteValue 4 = 16000 :=
by
  sorry

end product_of_local_and_absolute_value_l121_121824


namespace proof_problem_l121_121468

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x
def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a - x) = f (a + x)

theorem proof_problem (h_even : even_function f)
                      (h_period : ∀ x, f (x + 2) = -f x)
                      (h_incr : increasing_on f (-2) 0) :
                      periodic_function f 4 ∧ symmetric_about f 2 :=
by { sorry }

end proof_problem_l121_121468


namespace value_makes_expression_undefined_l121_121292

theorem value_makes_expression_undefined (a : ℝ) : 
    (a^2 - 9 * a + 20 = 0) ↔ (a = 4 ∨ a = 5) :=
by
  sorry

end value_makes_expression_undefined_l121_121292


namespace minimum_value_x2_minus_x1_range_of_a_l121_121152

noncomputable def f (x : ℝ) := Real.sin x + Real.exp x
noncomputable def g (x : ℝ) (a : ℝ) := a * x
noncomputable def F (x : ℝ) (a : ℝ) := f x - g x a

-- Question (I)
theorem minimum_value_x2_minus_x1 : ∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ 0 ≤ x₂ ∧ a = 1 / 3 ∧ f x₁ = g x₂ a → x₂ - x₁ = 3 := 
sorry

-- Question (II)
theorem range_of_a (a : ℝ) : (∀ x ≥ 0, F x a ≥ F (-x) a) ↔ a ≤ 2 :=
sorry

end minimum_value_x2_minus_x1_range_of_a_l121_121152


namespace quadratic_function_positive_l121_121442

theorem quadratic_function_positive (a m : ℝ) (h : a > 0) (h_fm : (m^2 + m + a) < 0) : (m + 1)^2 + (m + 1) + a > 0 :=
by sorry

end quadratic_function_positive_l121_121442


namespace range_of_a_l121_121146

variable (a : ℝ)

-- Definitions of propositions p and q
def p := ∀ x : ℝ, x^2 - 2*x - a ≥ 0
def q := ∃ x : ℝ, x^2 + x + 2*a - 1 ≤ 0

-- Lean 4 statement of the proof problem
theorem range_of_a : ¬ p a ∧ q a → -1 < a ∧ a ≤ 5/8 := by
  sorry

end range_of_a_l121_121146


namespace division_value_l121_121947

theorem division_value (n x : ℝ) (h₀ : n = 4.5) (h₁ : (n / x) * 12 = 9) : x = 6 :=
by
  sorry

end division_value_l121_121947


namespace equidistant_point_x_axis_l121_121402

theorem equidistant_point_x_axis (x : ℝ) (C D : ℝ × ℝ)
  (hC : C = (-3, 0))
  (hD : D = (0, 5))
  (heqdist : ∀ p : ℝ × ℝ, p.2 = 0 → 
    dist p C = dist p D) :
  x = 8 / 3 :=
by
  sorry

end equidistant_point_x_axis_l121_121402


namespace sample_size_is_correct_l121_121426

-- Define the school and selection conditions
def total_classes := 40
def students_per_class := 50

-- Given condition
def selected_students := 150

-- Theorem statement
theorem sample_size_is_correct : selected_students = 150 := 
by 
  sorry

end sample_size_is_correct_l121_121426


namespace scientific_notation_of_star_diameter_l121_121972

theorem scientific_notation_of_star_diameter:
    (∃ (c : ℝ) (n : ℕ), 1 ≤ c ∧ c < 10 ∧ 16600000000 = c * 10^n) → 
    16600000000 = 1.66 * 10^10 :=
by
  sorry

end scientific_notation_of_star_diameter_l121_121972


namespace katrina_cookies_left_l121_121340

/-- Katrina’s initial number of cookies. -/
def initial_cookies : ℕ := 120

/-- Cookies sold in the morning. 
    1 dozen is 12 cookies, so 3 dozen is 36 cookies. -/
def morning_sales : ℕ := 3 * 12

/-- Cookies sold during the lunch rush. -/
def lunch_sales : ℕ := 57

/-- Cookies sold in the afternoon. -/
def afternoon_sales : ℕ := 16

/-- Calculate the number of cookies left after all sales. -/
def cookies_left : ℕ :=
  initial_cookies - morning_sales - lunch_sales - afternoon_sales

/-- Prove that the number of cookies left for Katrina to take home is 11. -/
theorem katrina_cookies_left : cookies_left = 11 := by
  sorry

end katrina_cookies_left_l121_121340


namespace determine_ab_l121_121612

theorem determine_ab :
  ∃ a b : ℝ, 
  (3 + 8 * a = 2 - 3 * b) ∧ 
  (-1 - 6 * a = 4 * b) → 
  a = -1 / 14 ∧ b = -1 / 14 := 
by 
sorry

end determine_ab_l121_121612


namespace range_of_k_l121_121473

theorem range_of_k (k : ℝ) 
  (h1 : ∀ x y : ℝ, (x^2 / (k-3) + y^2 / (2-k) = 1) → (k-3 < 0) ∧ (2-k > 0)) : 
  k < 2 := by
  sorry

end range_of_k_l121_121473


namespace cos_double_angle_l121_121658

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 := sorry

end cos_double_angle_l121_121658


namespace wage_percent_change_l121_121592

-- Definitions based on given conditions
def initial_wage (W : ℝ) := W
def first_decrease (W : ℝ) := 0.60 * W
def first_increase (W : ℝ) := 0.78 * W
def second_decrease (W : ℝ) := 0.624 * W
def second_increase (W : ℝ) := 0.6864 * W

-- Lean theorem statement to prove overall percent change
theorem wage_percent_change : ∀ (W : ℝ), 
  ((second_increase (second_decrease (first_increase (first_decrease W))) - initial_wage W) / initial_wage W) * 100 = -31.36 :=
by sorry

end wage_percent_change_l121_121592


namespace find_z_l121_121630

noncomputable def z := {z : ℂ | ∃ i : ℂ, i^2 = -1 ∧ i * z = i - 1}

theorem find_z (i : ℂ) (hi : i^2 = -1) : ∃ z : ℂ, i * z = i - 1 ∧ z = 1 + i := by
  use 1 + i
  sorry

end find_z_l121_121630


namespace little_john_spent_on_sweets_l121_121729

theorem little_john_spent_on_sweets:
  let initial_amount := 10.10
  let amount_given_to_each_friend := 2.20
  let amount_left := 2.45
  let total_given_to_friends := 2 * amount_given_to_each_friend
  let amount_before_sweets := initial_amount - total_given_to_friends
  let amount_spent_on_sweets := amount_before_sweets - amount_left
  amount_spent_on_sweets = 3.25 :=
by
  sorry

end little_john_spent_on_sweets_l121_121729


namespace vertex_at_fixed_point_l121_121795

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 + 1

theorem vertex_at_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 1 = 2 :=
by
  sorry

end vertex_at_fixed_point_l121_121795


namespace game_24_set1_game_24_set2_l121_121198

-- Equivalent proof problem for set {3, 2, 6, 7}
theorem game_24_set1 (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 2) (h₃ : c = 6) (h₄ : d = 7) :
  ((d / b) * c + a) = 24 := by
  subst_vars
  sorry

-- Equivalent proof problem for set {3, 4, -6, 10}
theorem game_24_set2 (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = -6) (h₄ : d = 10) :
  ((b + c + d) * a) = 24 := by
  subst_vars
  sorry

end game_24_set1_game_24_set2_l121_121198


namespace range_of_a_l121_121708

theorem range_of_a (x : ℝ) (a : ℝ) (h₀ : x ∈ Set.Icc (-2 : ℝ) 3)
(h₁ : 2 * x - x ^ 2 ≥ a) : a ≤ 1 :=
sorry

end range_of_a_l121_121708


namespace probability_of_2_gold_no_danger_l121_121111

variable (caves : Finset Nat) (n : Nat)

-- Probability definitions
def P_gold_no_danger : ℚ := 1 / 5
def P_danger_no_gold : ℚ := 1 / 10
def P_neither : ℚ := 4 / 5

-- Probability calculation
def P_exactly_2_gold_none_danger : ℚ :=
  10 * (P_gold_no_danger) ^ 2 * (P_neither) ^ 3

theorem probability_of_2_gold_no_danger :
  (P_exactly_2_gold_none_danger) = 128 / 625 :=
sorry

end probability_of_2_gold_no_danger_l121_121111


namespace speed_of_first_half_of_journey_l121_121112

theorem speed_of_first_half_of_journey
  (total_time : ℝ)
  (speed_second_half : ℝ)
  (total_distance : ℝ)
  (first_half_distance : ℝ)
  (second_half_distance : ℝ)
  (time_second_half : ℝ)
  (time_first_half : ℝ)
  (speed_first_half : ℝ) :
  total_time = 15 →
  speed_second_half = 24 →
  total_distance = 336 →
  first_half_distance = total_distance / 2 →
  second_half_distance = total_distance / 2 →
  time_second_half = second_half_distance / speed_second_half →
  time_first_half = total_time - time_second_half →
  speed_first_half = first_half_distance / time_first_half →
  speed_first_half = 21 :=
by intros; sorry

end speed_of_first_half_of_journey_l121_121112


namespace S_10_is_65_l121_121349

variable (a_1 d : ℤ)
variable (S : ℤ → ℤ)

-- Define the arithmetic sequence conditions
def a_3 : ℤ := a_1 + 2 * d
def S_n (n : ℤ) : ℤ := n * a_1 + (n * (n - 1) / 2) * d

-- Given conditions
axiom a_3_is_4 : a_3 = 4
axiom S_9_minus_S_6_is_27 : S 9 - S 6 = 27

-- The target statement to be proven
theorem S_10_is_65 : S 10 = 65 :=
by
  sorry

end S_10_is_65_l121_121349


namespace exercise_l121_121236

-- Define a, b, and the identity
def a : ℕ := 45
def b : ℕ := 15

-- Theorem statement
theorem exercise :
  ((a + b) ^ 2 - (a ^ 2 + b ^ 2) = 1350) :=
by
  sorry

end exercise_l121_121236


namespace range_of_m_l121_121493

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, 0 < x → 0 < y → x < y → x^(m-1) > y^(m-1)) → m < 1 :=
by
  sorry

end range_of_m_l121_121493


namespace largest_value_l121_121243

theorem largest_value :
  let A := 1/2
  let B := 1/3 + 1/4
  let C := 1/4 + 1/5 + 1/6
  let D := 1/5 + 1/6 + 1/7 + 1/8
  let E := 1/6 + 1/7 + 1/8 + 1/9 + 1/10
  E > A ∧ E > B ∧ E > C ∧ E > D := by
sorry

end largest_value_l121_121243


namespace find_pair_l121_121820

theorem find_pair (a b : ℤ) :
  (∀ x : ℝ, (a * x^4 + b * x^3 + 20 * x^2 - 12 * x + 10) = (2 * x^2 + 3 * x - 4) * (c * x^2 + d * x + e)) → 
  (a = 2) ∧ (b = 27) :=
sorry

end find_pair_l121_121820


namespace pencils_undefined_l121_121545

-- Definitions for the conditions given in the problem
def initial_crayons : Nat := 41
def added_crayons : Nat := 12
def total_crayons : Nat := 53

-- Theorem stating the problem's required proof
theorem pencils_undefined (initial_crayons : Nat) (added_crayons : Nat) (total_crayons : Nat) : Prop :=
  initial_crayons = 41 ∧ added_crayons = 12 ∧ total_crayons = 53 → 
  ∃ (pencils : Nat), true
-- Since the number of pencils is unknown and no direct information is given, we represent it as an existential statement that pencils exist in some quantity, but we cannot determine their exact number based on given information.

end pencils_undefined_l121_121545


namespace equidistant_x_coordinate_l121_121401

open Real

-- Definitions for points C and D
def C : ℝ × ℝ := (-3, 0)
def D : ℝ × ℝ := (0, 5)

-- Definition for the distance function on the plane
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- The x-coordinate of the point that is equidistant from C and D
theorem equidistant_x_coordinate : ∃ x : ℝ, dist (x, 0) C = dist (x, 0) D ∧ x = 8/3 :=
by
  let x := 8/3
  have h1 : dist (x, 0) C = sqrt ((-3 - x)^2 + 0^2),
    simp only [C],
  have h2 : dist (x, 0) D = sqrt ((0 - x)^2 + (-5)^2),
    simp only [D],
  use x,
  split,
  {
    rw [dist, dist, h1, h2],
    sorry -- Proof steps omitted
  },
  {
    refl,
  }

end equidistant_x_coordinate_l121_121401


namespace total_football_games_l121_121059

theorem total_football_games (games_this_year : ℕ) (games_last_year : ℕ) (total_games : ℕ) : 
  games_this_year = 14 → games_last_year = 29 → total_games = games_this_year + games_last_year → total_games = 43 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end total_football_games_l121_121059


namespace math_equivalence_problem_l121_121556

theorem math_equivalence_problem :
  (2^2 + 92 * 3^2) * (4^2 + 92 * 5^2) = 1388^2 + 92 * 2^2 :=
by
  sorry

end math_equivalence_problem_l121_121556


namespace totalNutsInCar_l121_121355

-- Definitions based on the conditions
def busySquirrelNutsPerDay : Nat := 30
def busySquirrelDays : Nat := 35
def numberOfBusySquirrels : Nat := 2

def lazySquirrelNutsPerDay : Nat := 20
def lazySquirrelDays : Nat := 40
def numberOfLazySquirrels : Nat := 3

def sleepySquirrelNutsPerDay : Nat := 10
def sleepySquirrelDays : Nat := 45
def numberOfSleepySquirrels : Nat := 1

-- Calculate the total number of nuts stored by each type of squirrels
def totalNutsStoredByBusySquirrels : Nat := numberOfBusySquirrels * (busySquirrelNutsPerDay * busySquirrelDays)
def totalNutsStoredByLazySquirrels : Nat := numberOfLazySquirrels * (lazySquirrelNutsPerDay * lazySquirrelDays)
def totalNutsStoredBySleepySquirrel : Nat := numberOfSleepySquirrels * (sleepySquirrelNutsPerDay * sleepySquirrelDays)

-- The final theorem to prove
theorem totalNutsInCar : totalNutsStoredByBusySquirrels + totalNutsStoredByLazySquirrels + totalNutsStoredBySleepySquirrel = 4950 := by
  sorry

end totalNutsInCar_l121_121355


namespace common_root_divisibility_l121_121296

variables (a b c : ℤ)

theorem common_root_divisibility 
  (h1 : c ≠ b) 
  (h2 : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ (c - b) * x^2 + (c - a) * x + (a + b) = 0) 
  : 3 ∣ (a + b + 2 * c) :=
sorry

end common_root_divisibility_l121_121296


namespace nell_baseball_cards_l121_121893

theorem nell_baseball_cards 
  (ace_cards_now : ℕ) 
  (extra_baseball_cards : ℕ) 
  (B : ℕ) : 
  ace_cards_now = 55 →
  extra_baseball_cards = 123 →
  B = ace_cards_now + extra_baseball_cards →
  B = 178 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end nell_baseball_cards_l121_121893


namespace problem_solution_l121_121715

section
variables (a b : ℝ)

-- Definition of the \* operation
def star_op (a b : ℝ) : ℝ := (a + 1) * (b - 1)

-- Definition of a^{*2} as a \* a
def star_square (a : ℝ) : ℝ := star_op a a

-- Define the specific problem instance with x = 2
def problem_expr : ℝ := star_op 3 (star_square 2) - star_op 2 2 + 1

-- Theorem stating the correct answer
theorem problem_solution : problem_expr = 6 := by
  -- Proof steps, marked as 'sorry'
  sorry

end

end problem_solution_l121_121715


namespace scientific_notation_of_0_00003_l121_121249

theorem scientific_notation_of_0_00003 :
  0.00003 = 3 * 10^(-5) :=
sorry

end scientific_notation_of_0_00003_l121_121249


namespace stuffed_animals_total_l121_121890

theorem stuffed_animals_total :
  let McKenna := 34
  let Kenley := 2 * McKenna
  let Tenly := Kenley + 5
  McKenna + Kenley + Tenly = 175 :=
by
  sorry

end stuffed_animals_total_l121_121890


namespace cos_double_angle_l121_121690

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3/5) : Real.cos (2 * theta) = -7/25 :=
by
  sorry

end cos_double_angle_l121_121690


namespace exists_not_in_range_f_l121_121956

noncomputable def f : ℝ → ℕ :=
sorry

axiom functional_equation : ∀ (x y : ℝ), f (x + (1 / f y)) = f (y + (1 / f x))

theorem exists_not_in_range_f :
  ∃ n : ℕ, ∀ x : ℝ, f x ≠ n :=
sorry

end exists_not_in_range_f_l121_121956


namespace triangles_may_or_may_not_be_congruent_triangles_may_have_equal_areas_l121_121850

-- Given the conditions: two sides of one triangle are equal to two sides of another triangle.
-- And an angle opposite to one of these sides is equal to the angle opposite to the corresponding side.
variables {A B C D E F : Type}
variables {AB DE BC EF : ℝ} (h_AB_DE : AB = DE) (h_BC_EF : BC = EF)
variables {angle_A angle_D : ℝ} (h_angle_A_D : angle_A = angle_D)

-- Prove that the triangles may or may not be congruent
theorem triangles_may_or_may_not_be_congruent :
  ∃ (AB DE BC EF : ℝ) (angle_A angle_D : ℝ), AB = DE → BC = EF → angle_A = angle_D →
  (triangle_may_be_congruent_or_not : Prop) :=
sorry

-- Prove that the triangles may have equal areas
theorem triangles_may_have_equal_areas :
  ∃ (AB DE BC EF : ℝ) (angle_A angle_D : ℝ), AB = DE → BC = EF → angle_A = angle_D →
  (triangle_may_have_equal_areas : Prop) :=
sorry

end triangles_may_or_may_not_be_congruent_triangles_may_have_equal_areas_l121_121850


namespace find_angle_B_l121_121170

theorem find_angle_B 
  (A B C : ℝ)
  (h1 : B = A + 10)
  (h2 : C = B + 10)
  (h3 : A + B + C = 180) :
  B = 60 :=
sorry

end find_angle_B_l121_121170


namespace two_digit_number_solution_l121_121017

theorem two_digit_number_solution : ∃ (x y z : ℕ), (1 ≤ x ∧ x ≤ 9) ∧ (1 ≤ y ∧ y ≤ 9) ∧ 10 * x + y = 10 * 5 + 3 ∧ 10 * y + x = 10 * 3 + 5 ∧ 3 * z = 3 * 15 ∧ 2 * z = 2 * 15 := by
  sorry

end two_digit_number_solution_l121_121017


namespace find_integers_l121_121288

theorem find_integers (n : ℤ) : (6 ∣ (n - 4)) ∧ (10 ∣ (n - 8)) ↔ (n % 30 = 28) :=
by
  sorry

end find_integers_l121_121288


namespace parallel_lines_condition_l121_121645

theorem parallel_lines_condition (a : ℝ) : 
    (∀ x y : ℝ, 2 * x + a * y + 2 ≠ (a - 1) * x + y - 2) ↔ a = 2 := 
sorry

end parallel_lines_condition_l121_121645


namespace katrina_cookies_left_l121_121337

def initial_cookies : ℕ := 120
def morning_sales : ℕ := 3 * 12
def lunch_sales : ℕ := 57
def afternoon_sales : ℕ := 16
def total_sales : ℕ := morning_sales + lunch_sales + afternoon_sales
def cookies_left_to_take_home (initial: ℕ) (sold: ℕ) : ℕ := initial - sold

theorem katrina_cookies_left :
  cookies_left_to_take_home initial_cookies total_sales = 11 :=
by sorry

end katrina_cookies_left_l121_121337


namespace math_problem_l121_121244

theorem math_problem 
  (X : ℝ)
  (num1 : ℝ := 1 + 28/63)
  (num2 : ℝ := 8 + 7/16)
  (frac_sub1 : ℝ := 19/24 - 21/40)
  (frac_sub2 : ℝ := 1 + 28/63 - 17/21)
  (denom_calc : ℝ := 0.675 * 2.4 - 0.02) :
  0.125 * X / (frac_sub1 * num2) = (frac_sub2 * 0.7) / denom_calc → X = 5 := 
sorry

end math_problem_l121_121244


namespace solution_y_eq_2_l121_121487

theorem solution_y_eq_2 (y : ℝ) (h_pos : y > 0) (h_eq : y^6 = 64) : y = 2 :=
sorry

end solution_y_eq_2_l121_121487


namespace cos_double_angle_l121_121650

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (2 * θ) = -7 / 25 := 
by
  sorry

end cos_double_angle_l121_121650


namespace seashells_in_jar_at_end_of_month_l121_121136

noncomputable def seashells_in_week (initial: ℕ) (increment: ℕ) (week: ℕ) : ℕ :=
  initial + increment * week

theorem seashells_in_jar_at_end_of_month :
  seashells_in_week 50 20 0 +
  seashells_in_week 50 20 1 +
  seashells_in_week 50 20 2 +
  seashells_in_week 50 20 3 = 320 :=
sorry

end seashells_in_jar_at_end_of_month_l121_121136


namespace find_angle_B_find_cosine_sum_range_l121_121711

-- Define the acute triangle and given conditions
structure acute_triangle (A B C : ℝ) (a b c : ℝ) :=
(acute_A : 0 < A ∧ A < π / 2)
(acute_B : 0 < B ∧ B < π / 2)
(acute_C : 0 < C ∧ C < π / 2)
(sides : a > 0 ∧ b > 0 ∧ c > 0)
(angles_sum : A + B + C = π)
(given_condition : 2 * b * sin A = sqrt 3 * a)

-- Part Ⅰ: Proving the measure of angle B
theorem find_angle_B {A B C a b c : ℝ} (h : acute_triangle A B C a b c) : 
  B = π / 3 :=
sorry

-- Part Ⅱ: Proving the range of values for cos A + cos B + cos C
theorem find_cosine_sum_range {A B C a b c : ℝ} (h : acute_triangle A B C a b c) :
  (sqrt 3 + 1) / 2 < cos A + cos B + cos C ∧ cos A + cos B + cos C ≤ 3 / 2 :=
sorry

end find_angle_B_find_cosine_sum_range_l121_121711


namespace haley_initial_cupcakes_l121_121646

-- Define the conditions
def todd_eats : ℕ := 11
def packages : ℕ := 3
def cupcakes_per_package : ℕ := 3

-- Initial cupcakes calculation
def initial_cupcakes := packages * cupcakes_per_package + todd_eats

-- The theorem to prove
theorem haley_initial_cupcakes : initial_cupcakes = 20 :=
by
  -- Mathematical proof would go here,
  -- but we leave it as sorry for now.
  sorry

end haley_initial_cupcakes_l121_121646


namespace points_on_line_relation_l121_121001

theorem points_on_line_relation (b y1 y2 y3 : ℝ) 
  (h1 : y1 = -3 * (-2) + b) 
  (h2 : y2 = -3 * (-1) + b) 
  (h3 : y3 = -3 * 1 + b) : 
  y1 > y2 ∧ y2 > y3 :=
sorry

end points_on_line_relation_l121_121001


namespace cos_double_angle_l121_121699

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7/25 :=
by
  sorry

end cos_double_angle_l121_121699


namespace find_A_l121_121033

variables (a c : ℝ) (C A : ℝ)

-- Given conditions
def condition_1 : a = 4 * real.sqrt 3 := sorry
def condition_2 : c = 12 := sorry
def condition_3 : C = real.pi / 3 := sorry

theorem find_A : A = real.pi / 6 :=
by
  -- apply the given conditions
  have h1 : a = 4 * real.sqrt 3 := condition_1,
  have h2 : c = 12 := condition_2,
  have h3 : C = real.pi / 3 := condition_3,
  sorry

end find_A_l121_121033


namespace cos_double_angle_l121_121664

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3 / 5) : Real.cos (2 * theta) = -7 / 25 :=
by
  sorry

end cos_double_angle_l121_121664


namespace cosine_double_angle_l121_121684

theorem cosine_double_angle
  (θ : ℝ)
  (cos_θ : ℝ)
  (h : cos_θ = 3 / 5) :
  2 * cos_θ^2 - 1 = -7 / 25 :=
by
  sorry

end cosine_double_angle_l121_121684


namespace tickets_count_l121_121029

theorem tickets_count (x y: ℕ) (h : 3 * x + 5 * y = 78) : 
  ∃ n : ℕ , n = 6 :=
sorry

end tickets_count_l121_121029


namespace trigonometric_identity_l121_121124

theorem trigonometric_identity :
  Real.sin (17 * Real.pi / 180) * Real.sin (223 * Real.pi / 180) + 
  Real.sin (253 * Real.pi / 180) * Real.sin (313 * Real.pi / 180) = 1 / 2 := 
by
  sorry

end trigonometric_identity_l121_121124


namespace bryce_received_raisins_l121_121007

theorem bryce_received_raisins
  (C B : ℕ)
  (h1 : B = C + 8)
  (h2 : C = B / 3) :
  B = 12 :=
by sorry

end bryce_received_raisins_l121_121007


namespace car_speed_ratio_l121_121596

noncomputable def speed_ratio (t_round_trip t_leaves t_returns t_walk_start t_walk_end : ℕ) (meet_time : ℕ) : ℕ :=
  let one_way_time_car := t_round_trip / 2
  let total_car_time := t_returns - t_leaves
  let meeting_time_car := total_car_time / 2
  let remaining_time_to_factory := one_way_time_car - meeting_time_car
  let total_walk_time := t_walk_end - t_walk_start
  total_walk_time / remaining_time_to_factory

theorem car_speed_ratio :
  speed_ratio 60 120 160 60 140 80 = 8 :=
by
  sorry

end car_speed_ratio_l121_121596


namespace jaxon_toys_l121_121036

-- Definitions as per the conditions
def toys_jaxon : ℕ := sorry
def toys_gabriel : ℕ := 2 * toys_jaxon
def toys_jerry : ℕ := 2 * toys_jaxon + 8
def total_toys : ℕ := toys_jaxon + toys_gabriel + toys_jerry

-- Theorem to prove
theorem jaxon_toys : total_toys = 83 → toys_jaxon = 15 := sorry

end jaxon_toys_l121_121036


namespace trigonometry_expression_zero_l121_121171

variable {r : ℝ} {A B C : ℝ}
variable (a b c : ℝ) (sinA sinB sinC : ℝ)

-- The conditions from the problem
axiom Law_of_Sines_a : a = 2 * r * sinA
axiom Law_of_Sines_b : b = 2 * r * sinB
axiom Law_of_Sines_c : c = 2 * r * sinC

-- The theorem statement
theorem trigonometry_expression_zero :
  a * (sinC - sinB) + b * (sinA - sinC) + c * (sinB - sinA) = 0 :=
by
  -- Skipping the proof
  sorry

end trigonometry_expression_zero_l121_121171


namespace find_multiplier_l121_121240

theorem find_multiplier 
  (x : ℝ)
  (number : ℝ)
  (condition1 : 4 * number + x * number = 55)
  (condition2 : number = 5.0) :
  x = 7 :=
by
  sorry

end find_multiplier_l121_121240


namespace product_of_fractions_l121_121230

theorem product_of_fractions : (2 / 5) * (3 / 4) = 3 / 10 := 
  sorry

end product_of_fractions_l121_121230


namespace volume_of_cuboid_is_250_cm3_l121_121079

-- Define the edge length of the cube
def edge_length (a : ℕ) : ℕ := 5

-- Define the volume of a single cube
def cube_volume := (edge_length 5) ^ 3

-- Define the total volume of the cuboid formed by placing two such cubes in a line
def cuboid_volume := 2 * cube_volume

-- Theorem stating the volume of the cuboid formed
theorem volume_of_cuboid_is_250_cm3 : cuboid_volume = 250 := by
  sorry

end volume_of_cuboid_is_250_cm3_l121_121079


namespace obtain_x_squared_obtain_xy_l121_121336

theorem obtain_x_squared (x y : ℝ) (hx : x ≠ 1) (hx0 : 0 < x) (hy0 : 0 < y) :
  ∃ (k : ℝ), k = x^2 :=
by
  sorry

theorem obtain_xy (x y : ℝ) (hx0 : 0 < x) (hy0 : 0 < y) :
  ∃ (k : ℝ), k = x * y :=
by
  sorry

end obtain_x_squared_obtain_xy_l121_121336


namespace girl_attendance_l121_121273

theorem girl_attendance (g b : ℕ) (h1 : g + b = 1500) (h2 : (3 / 4 : ℚ) * g + (1 / 3 : ℚ) * b = 900) :
  (3 / 4 : ℚ) * g = 720 :=
by
  sorry

end girl_attendance_l121_121273


namespace cos_double_angle_l121_121702

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7/25 :=
by
  sorry

end cos_double_angle_l121_121702


namespace number_of_subsets_with_four_adj_chairs_l121_121384

-- Definition of the problem constants
def n : ℕ := 12

-- Define the condition that our chairs are arranged in a circle with 12 chairs
def is_adjacent (i j : ℕ) : Prop :=
  (j = (i + 1) % n) ∨ (i = (j + 1) % n) 

-- Define what it means for a subset to have at least four adjacent chairs
def at_least_four_adjacent (s : Finset ℕ) : Prop :=
  ∃ i j k l, i ∈ s ∧ j ∈ s ∧ k ∈ s ∧ l ∈ s ∧ is_adjacent i j ∧ is_adjacent j k ∧ is_adjacent k l

-- The main theorem to prove
theorem number_of_subsets_with_four_adj_chairs : ∃ k, k = 1701 ∧ ∀ s : Finset ℕ, s.card ≤ n → at_least_four_adjacent s →
  (∃ t, t.card = 4 ∧ t ⊆ s ∧ at_least_four_adjacent t) := 
sorry

end number_of_subsets_with_four_adj_chairs_l121_121384


namespace josephine_milk_containers_l121_121733

theorem josephine_milk_containers :
  3 * 2 + 2 * 0.75 + 5 * x = 10 → x = 0.5 :=
by
  intro h
  sorry

end josephine_milk_containers_l121_121733


namespace total_travel_time_in_minutes_l121_121048

def riding_rate : ℝ := 10 -- 10 miles per hour
def initial_riding_time : ℝ := 30 -- 30 minutes
def another_riding_distance : ℝ := 15 -- 15 miles
def resting_time : ℝ := 30 -- 30 minutes
def remaining_distance : ℝ := 20 -- 20 miles

theorem total_travel_time_in_minutes :
  initial_riding_time +
  (another_riding_distance / riding_rate * 60) +
  resting_time +
  (remaining_distance / riding_rate * 60) = 270 :=
by
  sorry

end total_travel_time_in_minutes_l121_121048


namespace bob_spending_over_limit_l121_121053

theorem bob_spending_over_limit : 
  ∀ (necklace_price book_price limit total_cost amount_over_limit : ℕ),
  necklace_price = 34 →
  book_price = necklace_price + 5 →
  limit = 70 →
  total_cost = necklace_price + book_price →
  amount_over_limit = total_cost - limit →
  amount_over_limit = 3 :=
by
  intros
  sorry

end bob_spending_over_limit_l121_121053


namespace tetrahedron_volume_distance_relation_l121_121425

theorem tetrahedron_volume_distance_relation
  (V : ℝ) (S1 S2 S3 S4 : ℝ) (H1 H2 H3 H4 : ℝ)
  (k : ℝ)
  (hS : (S1 / 1) = k) (hS2 : (S2 / 2) = k) (hS3 : (S3 / 3) = k) (hS4 : (S4 / 4) = k) :
  H1 + 2 * H2 + 3 * H3 + 4 * H4 = 3 * V / k :=
sorry

end tetrahedron_volume_distance_relation_l121_121425


namespace double_angle_cosine_l121_121674

theorem double_angle_cosine (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end double_angle_cosine_l121_121674


namespace problem_solution_l121_121491

-- Define the problem conditions and state the theorem
variable (a b : ℝ)
variable (h1 : a^2 - 4 * a + 3 = 0)
variable (h2 : b^2 - 4 * b + 3 = 0)
variable (h3 : a ≠ b)

theorem problem_solution : (a+1)*(b+1) = 8 := by
  sorry

end problem_solution_l121_121491


namespace sally_pokemon_cards_l121_121361

variable (X : ℤ)

theorem sally_pokemon_cards : X + 41 + 20 = 34 → X = -27 :=
by
  sorry

end sally_pokemon_cards_l121_121361


namespace sum_of_tesseract_elements_l121_121863

noncomputable def tesseract_edges : ℕ := 32
noncomputable def tesseract_vertices : ℕ := 16
noncomputable def tesseract_faces : ℕ := 24

theorem sum_of_tesseract_elements : tesseract_edges + tesseract_vertices + tesseract_faces = 72 := by
  -- proof here
  sorry

end sum_of_tesseract_elements_l121_121863


namespace cosine_double_angle_l121_121681

theorem cosine_double_angle
  (θ : ℝ)
  (cos_θ : ℝ)
  (h : cos_θ = 3 / 5) :
  2 * cos_θ^2 - 1 = -7 / 25 :=
by
  sorry

end cosine_double_angle_l121_121681


namespace republicans_in_house_l121_121071

theorem republicans_in_house (D R : ℕ) (h1 : D + R = 434) (h2 : R = D + 30) : R = 232 :=
by sorry

end republicans_in_house_l121_121071


namespace four_digit_number_is_2561_l121_121993

-- Define the problem domain based on given conditions
def unique_in_snowflake_and_directions (grid : Matrix (Fin 3) (Fin 6) ℕ) : Prop :=
  ∀ (i j : Fin 3), -- across all directions
    ∀ (x y : Fin 6), 
      (x ≠ y) → 
      (grid i x ≠ grid i y) -- uniqueness in i-direction
      ∧ (grid y x ≠ grid y y) -- uniqueness in j-direction

-- Assignment of numbers in the grid fulfilling the conditions
def grid : Matrix (Fin 3) (Fin 6) ℕ :=
![ ![2, 5, 2, 5, 1, 6], ![4, 3, 2, 6, 1, 1], ![6, 1, 4, 5, 3, 2] ]

-- Definition of the four-digit number
def ABCD : ℕ := grid 0 1 * 1000 + grid 0 2 * 100 + grid 0 3 * 10 + grid 0 4

-- The theorem to be proved
theorem four_digit_number_is_2561 :
  unique_in_snowflake_and_directions grid →
  ABCD = 2561 :=
sorry

end four_digit_number_is_2561_l121_121993


namespace product_of_roots_l121_121638

variable {x1 x2 : ℝ}

theorem product_of_roots (hx1 : x1 * Real.log x1 = 2006) (hx2 : x2 * Real.exp x2 = 2006) : x1 * x2 = 2006 :=
sorry

end product_of_roots_l121_121638


namespace scientific_notation_of_216000_l121_121745

theorem scientific_notation_of_216000 :
  216000 = 2.16 * 10^5 :=
sorry

end scientific_notation_of_216000_l121_121745


namespace odot_subtraction_l121_121445

-- Define the new operation
def odot (a b : ℚ) : ℚ := (a^3) / (b^2)

-- State the theorem
theorem odot_subtraction :
  ((odot (odot 2 4) 6) - (odot 2 (odot 4 6)) = -81 / 32) :=
by
  sorry

end odot_subtraction_l121_121445


namespace problem_1_l121_121957

theorem problem_1 (a b : ℝ) (h : b < a ∧ a < 0) : 
  (a + b < a * b) ∧ (¬ (abs a > abs b)) ∧ (¬ (1 / b > 1 / a ∧ 1 / a > 0)) ∧ (¬ (b / a + a / b > 2)) := sorry

end problem_1_l121_121957


namespace square_side_length_l121_121756

theorem square_side_length (x : ℝ) (h : 4 * x = 8 * Real.pi) : x = 6.28 := 
by {
  -- proof will go here
  sorry
}

end square_side_length_l121_121756


namespace jed_speed_l121_121855

theorem jed_speed
  (posted_speed_limit : ℕ := 50)
  (fine_per_mph_over_limit : ℕ := 16)
  (red_light_fine : ℕ := 75)
  (cellphone_fine : ℕ := 120)
  (parking_fine : ℕ := 50)
  (total_red_light_fines : ℕ := 2 * red_light_fine)
  (total_parking_fines : ℕ := 3 * parking_fine)
  (total_fine : ℕ := 1046)
  (non_speeding_fines : ℕ := total_red_light_fines + cellphone_fine + total_parking_fines)
  (speeding_fine : ℕ := total_fine - non_speeding_fines)
  (mph_over_limit : ℕ := speeding_fine / fine_per_mph_over_limit):
  (posted_speed_limit + mph_over_limit) = 89 :=
by
  sorry

end jed_speed_l121_121855


namespace tax_rate_computation_l121_121429

-- Define the inputs
def total_value : ℝ := 1720
def non_taxable_amount : ℝ := 600
def tax_paid : ℝ := 134.4

-- Define the derived taxable amount
def taxable_amount : ℝ := total_value - non_taxable_amount

-- Define the expected tax rate
def expected_tax_rate : ℝ := 0.12

-- State the theorem
theorem tax_rate_computation : 
  (tax_paid / taxable_amount * 100) = expected_tax_rate * 100 := 
by
  sorry

end tax_rate_computation_l121_121429


namespace trigonometric_identity_proof_l121_121294

variable (α : Real)

theorem trigonometric_identity_proof (h1 : Real.tan α = 4 / 3) (h2 : 0 < α ∧ α < Real.pi / 2) :
  Real.sin (Real.pi + α) + Real.cos (Real.pi - α) = -7 / 5 :=
by
  sorry

end trigonometric_identity_proof_l121_121294


namespace sister_granola_bars_l121_121845

-- Definitions based on conditions
def total_bars := 20
def chocolate_chip_bars := 8
def oat_honey_bars := 6
def peanut_butter_bars := 6

def greg_set_aside_chocolate := 3
def greg_set_aside_oat_honey := 2
def greg_set_aside_peanut_butter := 2

def final_chocolate_chip := chocolate_chip_bars - greg_set_aside_chocolate - 2  -- 2 traded away
def final_oat_honey := oat_honey_bars - greg_set_aside_oat_honey - 4           -- 4 traded away
def final_peanut_butter := peanut_butter_bars - greg_set_aside_peanut_butter

-- Final distribution to sisters
def older_sister_chocolate := 2.5 -- 2 whole bars + 1/2 bar
def younger_sister_peanut := 2.5  -- 2 whole bars + 1/2 bar

theorem sister_granola_bars :
  older_sister_chocolate = 2.5 ∧ younger_sister_peanut = 2.5 :=
by
  sorry

end sister_granola_bars_l121_121845


namespace price_increase_percentage_l121_121858

theorem price_increase_percentage (x : ℝ) :
  (0.9 * (1 + x / 100) * 0.9259259259259259 = 1) → x = 20 :=
by
  intros
  sorry

end price_increase_percentage_l121_121858


namespace num_boys_l121_121108

variable (B G : ℕ)

def ratio_boys_girls (B G : ℕ) : Prop := B = 7 * G
def total_students (B G : ℕ) : Prop := B + G = 48

theorem num_boys (B G : ℕ) (h1 : ratio_boys_girls B G) (h2 : total_students B G) : 
  B = 42 :=
by
  sorry

end num_boys_l121_121108


namespace scout_troop_profit_l121_121975

noncomputable def candy_profit (purchase_bars purchase_rate sell_bars sell_rate donation_fraction : ℕ) : ℕ :=
  let cost_price_per_bar := purchase_rate / purchase_bars
  let total_cost := purchase_bars * cost_price_per_bar
  let effective_cost := total_cost * donation_fraction
  let sell_price_per_bar := sell_rate / sell_bars
  let total_revenue := purchase_bars * sell_price_per_bar
  total_revenue - effective_cost

theorem scout_troop_profit :
  candy_profit 1200 3 4 3 1/2 = 700 := by
  sorry

end scout_troop_profit_l121_121975


namespace quadratic_root_k_eq_one_l121_121459

theorem quadratic_root_k_eq_one
  (k : ℝ)
  (h₀ : (k + 3) ≠ 0)
  (h₁ : ∃ x : ℝ, (x = 0) ∧ ((k + 3) * x^2 + 5 * x + k^2 + 2 * k - 3 = 0)) :
  k = 1 :=
by
  sorry

end quadratic_root_k_eq_one_l121_121459


namespace badge_exchange_proof_l121_121562

-- Definitions based on the conditions
def initial_badges_Tolya : ℝ := 45
def initial_badges_Vasya : ℝ := 50

def tollya_exchange_badges (badges_Tolya : ℝ) : ℝ := 0.2 * badges_Tolya
def tollya_receive_badges (badges_Vasya : ℝ) : ℝ := 0.24 * badges_Vasya
def vasya_exchange_badges (badges_Vasya: ℝ) : ℝ := 0.24 * badges_Vasya
def vasya_receive_badges (badges_Tolya: ℝ) : ℝ := 0.2 * badges_Tolya

-- Vasya ended up with one badge less than Tolya after the exchange
theorem badge_exchange_proof 
  (tolya_initial badges_Tolya_initial : ℝ)
  (badges_Vasya_initial: ℝ)
  (tollya_initial_has_24: tollya_receive_badges badges_Vasya_initial)
  (vasya_initial_has_20: vasya_receive_badges badges_Tolya_initial):
  (tollya_initial = initial_badges_Tolya) ∧ (vasya_initial = initial_badges_Vasya) :=
sorry

end badge_exchange_proof_l121_121562


namespace simplify_fraction_l121_121902

variable {x y : ℝ}

theorem simplify_fraction (hx : x ≠ 0) : (x * y) / (3 * x) = y / 3 := by
  sorry

end simplify_fraction_l121_121902


namespace cos_double_angle_l121_121652

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (2 * θ) = -7 / 25 := 
by
  sorry

end cos_double_angle_l121_121652


namespace average_growth_rate_le_half_sum_l121_121096

variable (p q x : ℝ)

theorem average_growth_rate_le_half_sum : 
  (1 + p) * (1 + q) = (1 + x) ^ 2 → x ≤ (p + q) / 2 :=
by
  intro h
  sorry

end average_growth_rate_le_half_sum_l121_121096


namespace blocks_differ_in_two_ways_exactly_l121_121584

theorem blocks_differ_in_two_ways_exactly 
  (materials : Finset String := {"plastic", "wood", "metal"})
  (sizes : Finset String := {"small", "medium", "large"})
  (colors : Finset String := {"blue", "green", "red", "yellow"})
  (shapes : Finset String := {"circle", "hexagon", "square", "triangle"})
  (target : String := "plastic medium red circle") :
  ∃ (n : ℕ), n = 37 := by
  sorry

end blocks_differ_in_two_ways_exactly_l121_121584


namespace standard_equation_of_ellipse_l121_121836

-- Define the conditions
def isEccentricity (e : ℝ) := e = (Real.sqrt 3) / 3
def segmentLength (L : ℝ) := L = (4 * Real.sqrt 3) / 3

-- Define properties
def is_ellipse (a b c : ℝ) := a > b ∧ b > 0 ∧ (a^2 = b^2 + c^2) ∧ (c = (Real.sqrt 3) / 3 * a)

-- The problem statement
theorem standard_equation_of_ellipse
(a b c : ℝ) (E L : ℝ)
(hE : isEccentricity E)
(hL : segmentLength L)
(h : is_ellipse a b c)
: (a = Real.sqrt 3) ∧ (c = 1) ∧ (b = Real.sqrt 2) ∧ (segmentLength L)
  → ( ∀ x y : ℝ, ((x^2 / 3) + (y^2 / 2) = 1) ) := by
  sorry

end standard_equation_of_ellipse_l121_121836


namespace ordered_pairs_satisfy_equation_l121_121315

theorem ordered_pairs_satisfy_equation :
  (∃ (a : ℝ) (b : ℤ), a > 0 ∧ 3 ≤ b ∧ b ≤ 203 ∧ (Real.log a / Real.log b) ^ 2021 = Real.log (a ^ 2021) / Real.log b) :=
sorry

end ordered_pairs_satisfy_equation_l121_121315


namespace min_r_minus_p_l121_121005

theorem min_r_minus_p : ∃ (p q r : ℕ), p * q * r = 362880 ∧ p < q ∧ q < r ∧ (∀ p' q' r' : ℕ, (p' * q' * r' = 362880 ∧ p' < q' ∧ q' < r') → r - p ≤ r' - p') ∧ r - p = 39 :=
by
  sorry

end min_r_minus_p_l121_121005


namespace negation_universal_exists_l121_121213

open Classical

theorem negation_universal_exists :
  (¬ ∀ x : ℝ, x > 0 → (x^2 - x + 3 > 0)) ↔ ∃ x : ℝ, x > 0 ∧ (x^2 - x + 3 ≤ 0) :=
by
  sorry

end negation_universal_exists_l121_121213


namespace number_of_cooks_l121_121274

variable (C W : ℕ)

-- Conditions
def initial_ratio := 3 * W = 8 * C
def new_ratio := 4 * C = W + 12

theorem number_of_cooks (h1 : initial_ratio W C) (h2 : new_ratio W C) : C = 9 := by
  sorry

end number_of_cooks_l121_121274


namespace fractional_inequality_solution_set_l121_121932

theorem fractional_inequality_solution_set (x : ℝ) :
  (x / (x + 1) < 0) ↔ (-1 < x) ∧ (x < 0) :=
sorry

end fractional_inequality_solution_set_l121_121932


namespace find_f_neg1_l121_121877

-- Definitions based on conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

variable {b : ℝ} (f : ℝ → ℝ)

axiom odd_f : odd_function f
axiom f_form : ∀ x, 0 ≤ x → f x = 2^(x + 1) + 2 * x + b
axiom b_value : b = -2

theorem find_f_neg1 : f (-1) = -4 :=
sorry

end find_f_neg1_l121_121877


namespace quadratic_inequality_solution_l121_121068

theorem quadratic_inequality_solution (a b : ℝ)
  (h : ∀ x : ℝ, -1/2 < x ∧ x < 1/3 → ax^2 + bx + 2 > 0) :
  a + b = -14 :=
sorry

end quadratic_inequality_solution_l121_121068


namespace intersection_of_sets_is_closed_interval_l121_121880

noncomputable def A := {x : ℝ | x ≤ 0 ∨ x ≥ 2}
noncomputable def B := {x : ℝ | x < 1}

theorem intersection_of_sets_is_closed_interval :
  A ∩ B = {x : ℝ | x ≤ 0} :=
sorry

end intersection_of_sets_is_closed_interval_l121_121880


namespace garage_sale_items_count_l121_121410

theorem garage_sale_items_count (n_high n_low: ℕ) :
  n_high = 17 ∧ n_low = 24 → total_items = 40 :=
by
  let n_high: ℕ := 17
  let n_low: ℕ := 24
  let total_items: ℕ := (n_high - 1) + (n_low - 1) + 1
  sorry

end garage_sale_items_count_l121_121410


namespace complete_square_l121_121363

theorem complete_square {x : ℝ} (h : x^2 + 10 * x - 3 = 0) : (x + 5)^2 = 28 :=
sorry

end complete_square_l121_121363


namespace evaluate_expression_l121_121521

theorem evaluate_expression (a : ℝ) (h : a = 2) : 
    (a / (a^2 - 1) - 1 / (a^2 - 1)) = 1 / 3 := by
  sorry

end evaluate_expression_l121_121521


namespace smallest_integer_ends_3_divisible_5_l121_121780

theorem smallest_integer_ends_3_divisible_5 : ∃ n : ℕ, (53 = n ∧ (n % 10 = 3) ∧ (n % 5 = 0) ∧ ∀ m : ℕ, (m % 10 = 3) ∧ (m % 5 = 0) → 53 ≤ m) :=
by {
  sorry
}

end smallest_integer_ends_3_divisible_5_l121_121780


namespace circle_equation_l121_121754

theorem circle_equation (x y : ℝ) :
  (∃ a < 0, (x - a)^2 + y^2 = 4 ∧ (0 - a)^2 + 0^2 = 4) ↔ (x + 2)^2 + y^2 = 4 := 
sorry

end circle_equation_l121_121754


namespace parallel_lines_a_l121_121480

theorem parallel_lines_a (a : ℝ) (x y : ℝ)
  (h1 : x + 2 * a * y - 1 = 0)
  (h2 : (a + 1) * x - a * y = 0)
  (h_parallel : ∀ (l1 l2 : ℝ → ℝ → Prop), l1 x y ∧ l2 x y → l1 = l2) :
  a = -3 / 2 ∨ a = 0 :=
sorry

end parallel_lines_a_l121_121480


namespace poly_coeff_sum_l121_121634

variable {a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ}

theorem poly_coeff_sum :
  (∀ x : ℝ, (2 * x - 1) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6) →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 + 6 * a_6 = 12 :=
by
  sorry

end poly_coeff_sum_l121_121634


namespace range_of_a_l121_121643

noncomputable def discriminant (a : ℝ) : ℝ :=
  (2 * a)^2 - 4 * 1 * 1

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + 2 * a * x + 1 < 0) ↔ (a < -1 ∨ a > 1) :=
by
  sorry

end range_of_a_l121_121643


namespace cos_double_angle_l121_121672

variable (θ : Real)
variables (h : cos θ = 3 / 5)

theorem cos_double_angle :
  cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l121_121672


namespace find_c_in_parabola_l121_121593

theorem find_c_in_parabola (b c : ℝ) (h₁ : 2 = (-1) ^ 2 + b * (-1) + c) (h₂ : 2 = 3 ^ 2 + b * 3 + c) : c = -1 :=
sorry

end find_c_in_parabola_l121_121593


namespace exponents_multiplication_l121_121786

variable (a : ℝ)

theorem exponents_multiplication : a^3 * a = a^4 := by
  sorry

end exponents_multiplication_l121_121786


namespace Jake_has_one_more_balloon_than_Allan_l121_121977

-- Defining the given values
def A : ℕ := 6
def J_initial : ℕ := 3
def J_buy : ℕ := 4
def J_total : ℕ := J_initial + J_buy

-- The theorem statement
theorem Jake_has_one_more_balloon_than_Allan : J_total - A = 1 := 
by
  sorry -- proof goes here

end Jake_has_one_more_balloon_than_Allan_l121_121977


namespace smallest_value_y_l121_121132

theorem smallest_value_y (y : ℝ) : (|y - 8| = 15) → y = -7 :=
by
  sorry

end smallest_value_y_l121_121132


namespace general_formula_for_a_n_a_n_greater_than_b_n_find_n_in_geometric_sequence_l121_121311

-- Defines the sequences and properties given in the problem
def sequences (a_n b_n S_n T_n : ℕ → ℕ) : Prop :=
  a_n 1 = 1 ∧ S_n 2 = 4 ∧ 
  (∀ n : ℕ, 3 * S_n (n + 1) = 2 * S_n n + S_n (n + 2) + a_n n)

-- (1) Prove the general formula for {a_n}
theorem general_formula_for_a_n
  (a_n b_n S_n T_n : ℕ → ℕ)
  (h : sequences a_n b_n S_n T_n) :
  ∀ n : ℕ, a_n n = 2 * n - 1 :=
sorry

-- (2) If {b_n} is an arithmetic sequence and ∀n ∈ ℕ, S_n > T_n, prove a_n > b_n
theorem a_n_greater_than_b_n
  (a_n b_n S_n T_n : ℕ → ℕ)
  (h : sequences a_n b_n S_n T_n)
  (arithmetic_b : ∃ d: ℕ, ∀ n: ℕ, b_n n = b_n 0 + n * d)
  (Sn_greater_Tn : ∀ (n : ℕ), S_n n > T_n n) :
  ∀ n : ℕ, a_n n > b_n n :=
sorry

-- (3) If {b_n} is a geometric sequence, find n such that (a_n + 2 * T_n) / (b_n + 2 * S_n) = a_k
theorem find_n_in_geometric_sequence
  (a_n b_n S_n T_n : ℕ → ℕ)
  (h : sequences a_n b_n S_n T_n)
  (geometric_b : ∃ r: ℕ, ∀ n: ℕ, b_n n = b_n 0 * r^n)
  (b1_eq_1 : b_n 1 = 1)
  (b2_eq_3 : b_n 2 = 3)
  (k : ℕ) :
  ∃ n : ℕ, (a_n n + 2 * T_n n) / (b_n n + 2 * S_n n) = a_n k := 
sorry

end general_formula_for_a_n_a_n_greater_than_b_n_find_n_in_geometric_sequence_l121_121311


namespace carter_stretching_legs_frequency_l121_121979

-- Given conditions
def tripDuration : ℤ := 14 * 60 -- in minutes
def foodStops : ℤ := 2
def gasStops : ℤ := 3
def pitStopDuration : ℤ := 20 -- in minutes
def totalTripDuration : ℤ := 18 * 60 -- in minutes

-- Prove that Carter stops to stretch his legs every 2 hours
theorem carter_stretching_legs_frequency :
  ∃ (stretchingStops : ℤ), (totalTripDuration - tripDuration = (foodStops + gasStops + stretchingStops) * pitStopDuration) ∧
    (stretchingStops * pitStopDuration = totalTripDuration - (tripDuration + (foodStops + gasStops) * pitStopDuration)) ∧
    (14 / stretchingStops = 2) :=
by sorry

end carter_stretching_legs_frequency_l121_121979


namespace decreased_price_correct_l121_121431

def actual_cost : ℝ := 250
def percentage_decrease : ℝ := 0.2

theorem decreased_price_correct : actual_cost - (percentage_decrease * actual_cost) = 200 :=
by
  sorry

end decreased_price_correct_l121_121431


namespace number_of_subsets_with_four_adj_chairs_l121_121385

-- Definition of the problem constants
def n : ℕ := 12

-- Define the condition that our chairs are arranged in a circle with 12 chairs
def is_adjacent (i j : ℕ) : Prop :=
  (j = (i + 1) % n) ∨ (i = (j + 1) % n) 

-- Define what it means for a subset to have at least four adjacent chairs
def at_least_four_adjacent (s : Finset ℕ) : Prop :=
  ∃ i j k l, i ∈ s ∧ j ∈ s ∧ k ∈ s ∧ l ∈ s ∧ is_adjacent i j ∧ is_adjacent j k ∧ is_adjacent k l

-- The main theorem to prove
theorem number_of_subsets_with_four_adj_chairs : ∃ k, k = 1701 ∧ ∀ s : Finset ℕ, s.card ≤ n → at_least_four_adjacent s →
  (∃ t, t.card = 4 ∧ t ⊆ s ∧ at_least_four_adjacent t) := 
sorry

end number_of_subsets_with_four_adj_chairs_l121_121385


namespace units_digit_of_27_mul_36_l121_121620

theorem units_digit_of_27_mul_36 : (27 * 36) % 10 = 2 := by
  sorry

end units_digit_of_27_mul_36_l121_121620


namespace base_of_500_in_decimal_is_7_l121_121996

theorem base_of_500_in_decimal_is_7 :
  ∃ b : ℕ, 5 ≤ b ∧ b ≤ 7 ∧
  ∀ n, (500 : ℕ).digits b = n.digits b ∧ 
  n.length = 4 ∧ (n.last % 2 = 1) :=
begin
  sorry
end

end base_of_500_in_decimal_is_7_l121_121996


namespace nearby_island_banana_production_l121_121960

theorem nearby_island_banana_production
  (x : ℕ)
  (h_prod: 10 * x + x = 99000) :
  x = 9000 :=
sorry

end nearby_island_banana_production_l121_121960


namespace ellipse_area_irrational_l121_121155

noncomputable def ellipse_area (a b : ℚ) : ℝ :=
  Real.pi * a * b

theorem ellipse_area_irrational 
  (a b : ℚ) : irrational (ellipse_area a b) :=
by {
  let A := ellipse_area a b,
  have h_rat : a * b ∈ ℚ := mul_rat (a) (b),
  have h_pi_irr : irrational Real.pi := Real.irrational_pi,
  exact irrational.mul_rat h_pi_irr h_rat,
  sorry
}

end ellipse_area_irrational_l121_121155


namespace g_4_minus_g_7_l121_121916

theorem g_4_minus_g_7 (g : ℝ → ℝ) (h_linear : ∀ x y : ℝ, g (x + y) = g x + g y)
  (h_diff : ∀ k : ℝ, g (k + 1) - g k = 5) : g 4 - g 7 = -15 :=
by
  sorry

end g_4_minus_g_7_l121_121916


namespace cos_double_angle_l121_121667

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3 / 5) : Real.cos (2 * theta) = -7 / 25 :=
by
  sorry

end cos_double_angle_l121_121667


namespace problem1_problem2_l121_121154

-- Definition of f(x)
def f (x : ℝ) : ℝ := abs (x - 1)

-- Definition of g(x)
def g (x t : ℝ) : ℝ := t * abs x - 2

-- Problem 1: Proof that f(x) > 2x + 1 implies x < 0
theorem problem1 (x : ℝ) : f x > 2 * x + 1 → x < 0 := by
  sorry

-- Problem 2: Proof that if f(x) ≥ g(x) for all x, then t ≤ 1
theorem problem2 (t : ℝ) : (∀ x : ℝ, f x ≥ g x t) → t ≤ 1 := by
  sorry

end problem1_problem2_l121_121154


namespace donna_smallest_n_l121_121448

theorem donna_smallest_n (n : ℕ) : 15 * n - 1 % 6 = 0 ↔ n % 6 = 5 := sorry

end donna_smallest_n_l121_121448


namespace triangle_side_lengths_triangle_circumradius_l121_121470

theorem triangle_side_lengths (a b c : ℝ) (h : a^2 + b^2 + c^2 - 6 * a - 8 * b - 10 * c + 50 = 0) :
  a = 3 ∧ b = 4 ∧ c = 5 :=
by sorry

theorem triangle_circumradius (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
  let R := c / 2 in R = 2.5 :=
by sorry

end triangle_side_lengths_triangle_circumradius_l121_121470


namespace katrina_cookies_left_l121_121338

def initial_cookies : ℕ := 120
def morning_sales : ℕ := 3 * 12
def lunch_sales : ℕ := 57
def afternoon_sales : ℕ := 16
def total_sales : ℕ := morning_sales + lunch_sales + afternoon_sales
def cookies_left_to_take_home (initial: ℕ) (sold: ℕ) : ℕ := initial - sold

theorem katrina_cookies_left :
  cookies_left_to_take_home initial_cookies total_sales = 11 :=
by sorry

end katrina_cookies_left_l121_121338


namespace evaluate_expression_l121_121163

theorem evaluate_expression (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y + 2 = 3 := by
  rw [hx, hy]
  sorry

end evaluate_expression_l121_121163


namespace max_value_of_f_l121_121918

noncomputable def f (x : ℝ) : ℝ := x * (4 - x)

theorem max_value_of_f : ∃ y, ∀ x ∈ Set.Ioo 0 4, f x ≤ y ∧ y = 4 :=
by
  sorry

end max_value_of_f_l121_121918


namespace probability_correct_l121_121856

noncomputable def probability_one_white_one_black
    (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) (draw_balls : ℕ) :=
if (total_balls = 4) ∧ (white_balls = 2) ∧ (black_balls = 2) ∧ (draw_balls = 2) then
  (2 * 2) / (Nat.choose total_balls draw_balls : ℚ)
else
  0

theorem probability_correct:
  probability_one_white_one_black 4 2 2 2 = 2 / 3 :=
by
  sorry

end probability_correct_l121_121856


namespace quadratic_rewrite_l121_121923

noncomputable def a : ℕ := 6
noncomputable def b : ℕ := 6
noncomputable def c : ℕ := 284
noncomputable def quadratic_coeffs_sum : ℕ := a + b + c

theorem quadratic_rewrite :
  (∃ a b c : ℕ, 6 * (x : ℕ) ^ 2 + 72 * x + 500 = a * (x + b) ^ 2 + c) →
  quadratic_coeffs_sum = 296 := by sorry

end quadratic_rewrite_l121_121923


namespace sequence_integers_l121_121725

variable {R : Type*} [CommRing R] {x y : R}

def a (n : ℕ) := ∑ k in Finset.range (n + 1), x ^ k * y ^ (n - k)

theorem sequence_integers
  (h : ∃ (m : ℕ), a m ∈ ℤ ∧ a (m + 1) ∈ ℤ ∧ a (m + 2) ∈ ℤ ∧ a (m + 3) ∈ ℤ) :
  ∀ n, a n ∈ ℤ := 
sorry

end sequence_integers_l121_121725


namespace rental_cost_equal_mileage_l121_121360

theorem rental_cost_equal_mileage :
  ∃ m : ℝ, 
    (21.95 + 0.19 * m = 18.95 + 0.21 * m) ∧ 
    m = 150 :=
by
  sorry

end rental_cost_equal_mileage_l121_121360


namespace tan_subtraction_l121_121164

theorem tan_subtraction (α β : ℝ) (h1 : Real.tan α = 5) (h2 : Real.tan β = 3) : 
  Real.tan (α - β) = 1 / 8 := 
by 
  sorry

end tan_subtraction_l121_121164


namespace percent_parrots_among_non_pelicans_l121_121027

theorem percent_parrots_among_non_pelicans 
  (parrots_percent pelicans_percent owls_percent sparrows_percent : ℝ) 
  (H1 : parrots_percent = 40) 
  (H2 : pelicans_percent = 20) 
  (H3 : owls_percent = 15) 
  (H4 : sparrows_percent = 100 - parrots_percent - pelicans_percent - owls_percent)
  (H5 : pelicans_percent / 100 < 1) :
  parrots_percent / (100 - pelicans_percent) * 100 = 50 :=
by sorry

end percent_parrots_among_non_pelicans_l121_121027


namespace charming_number_unique_l121_121282

def is_charming (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≥ 1 ∧ a ≤ 9 ∧ b ≤ 9 ∧ n = 10 * a + b ∧ n = 2 * a + b^3

theorem charming_number_unique : ∃! n, 10 ≤ n ∧ n ≤ 99 ∧ is_charming n := by
  sorry

end charming_number_unique_l121_121282


namespace cos_double_angle_l121_121701

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7/25 :=
by
  sorry

end cos_double_angle_l121_121701


namespace inequality_proof_l121_121352

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x ≥ y + z) : 
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 := 
sorry

end inequality_proof_l121_121352


namespace senior_ticket_cost_is_13_l121_121517

theorem senior_ticket_cost_is_13
    (adult_ticket_cost : ℕ)
    (child_ticket_cost : ℕ)
    (senior_ticket_cost : ℕ)
    (total_cost : ℕ)
    (num_adults : ℕ)
    (num_children : ℕ)
    (num_senior_citizens : ℕ)
    (age_child1 : ℕ)
    (age_child2 : ℕ)
    (age_child3 : ℕ) :
    adult_ticket_cost = 11 → 
    child_ticket_cost = 8 →
    total_cost = 64 →
    num_adults = 2 →
    num_children = 2 → -- children with discount tickets
    num_senior_citizens = 2 →
    age_child1 = 7 → 
    age_child2 = 10 → 
    age_child3 = 14 → -- this child does not get discount
    senior_ticket_cost * num_senior_citizens = total_cost - (num_adults * adult_ticket_cost + num_children * child_ticket_cost) →
    senior_ticket_cost = 13 :=
by
  intros
  sorry

end senior_ticket_cost_is_13_l121_121517


namespace loss_percentage_l121_121106

-- Definitions related to the problem
def CPA : Type := ℝ
def SPAB (CPA: ℝ) : ℝ := 1.30 * CPA
def SPBC (CPA: ℝ) : ℝ := 1.040000000000000036 * CPA

-- Theorem to prove the loss percentage when B sold the bicycle to C 
theorem loss_percentage (CPA : ℝ) (L : ℝ) (h1 : SPAB CPA * (1 - L) = SPBC CPA) : 
  L = 0.20 :=
by
  sorry

end loss_percentage_l121_121106


namespace number_of_molecules_correct_l121_121411

-- Define Avogadro's number
def avogadros_number : ℝ := 6.022 * 10^23

-- Define the given number of molecules
def given_number_of_molecules : ℝ := 3 * 10^26

-- State the problem
theorem number_of_molecules_correct :
  (number_of_molecules = given_number_of_molecules) :=
by
  sorry

end number_of_molecules_correct_l121_121411


namespace sequence_v_n_l121_121508

theorem sequence_v_n (v : ℕ → ℝ)
  (h_recurr : ∀ n, v (n+2) = 3 * v (n+1) - v n)
  (h_init1 : v 3 = 16)
  (h_init2 : v 6 = 211) : 
  v 5 = 81.125 :=
sorry

end sequence_v_n_l121_121508


namespace password_lock_probability_l121_121424

theorem password_lock_probability :
  (∃ (A B C : ℕ), 0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 
   (∃ password : ℕ × ℕ × ℕ, password = (A, B, C) ∧ 
   (∃ chosen_middle_digit : ℕ, 0 <= chosen_middle_digit ∧ chosen_middle_digit <= 9 ∧
   chosen_middle_digit = B))) →
  (1/10 : ℝ) = 0.1 :=
by 
  sorry

end password_lock_probability_l121_121424


namespace price_of_cheese_cookie_pack_l121_121582

theorem price_of_cheese_cookie_pack
    (cartons : ℕ) (boxes_per_carton : ℕ) (packs_per_box : ℕ) (total_cost : ℕ)
    (h_cartons : cartons = 12)
    (h_boxes_per_carton : boxes_per_carton = 12)
    (h_packs_per_box : packs_per_box = 10)
    (h_total_cost : total_cost = 1440) :
  (total_cost / (cartons * boxes_per_carton * packs_per_box) = 1) :=
by
  -- conditions are explicitly given in the theorem statement
  sorry

end price_of_cheese_cookie_pack_l121_121582


namespace cube_volume_l121_121971

theorem cube_volume (a : ℕ) (h1 : 9 * 12 * 3 = 324) (h2 : 108 * a^3 = 324) : a^3 = 27 :=
by {
  sorry
}

end cube_volume_l121_121971


namespace vector_CD_l121_121481

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D : V)
variables (a b c : V)
variable (h1 : B - A = a)
variable (h2 : B - C = b)
variable (h3 : D - A = c)

theorem vector_CD :
  D - C = -a + b + c :=
by
  -- Proof omitted
  sorry

end vector_CD_l121_121481


namespace infinite_divisors_l121_121520

theorem infinite_divisors (a : ℕ) : ∃ (a : ℕ) (a_seq : ℕ → ℕ), (∀ n : ℕ, (a_seq n)^2 ∣ 2^(a_seq n) + 3^(a_seq n)) :=
by
  sorry

end infinite_divisors_l121_121520


namespace six_digit_integers_count_l121_121010

theorem six_digit_integers_count : 
  let digits := [2, 2, 2, 5, 5, 9] in
  multiset.card (multiset.of_list (list.permutations digits).erase_dup) = 60 :=
by
  sorry

end six_digit_integers_count_l121_121010


namespace counting_five_digit_numbers_l121_121159

theorem counting_five_digit_numbers :
  ∃ (M : ℕ), 
    (∃ (b : ℕ), (∃ (y : ℕ), 10000 * b + y = 8 * y ∧ 10000 * b = 7 * y ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1429 ≤ y ∧ y ≤ 9996)) ∧ 
    (M = 1224) := 
by
  sorry

end counting_five_digit_numbers_l121_121159


namespace exist_prime_not_dividing_l121_121148

theorem exist_prime_not_dividing (p : ℕ) (hp : Prime p) : 
  ∃ q : ℕ, Prime q ∧ ∀ n : ℕ, 0 < n → ¬ (q ∣ n^p - p) := 
sorry

end exist_prime_not_dividing_l121_121148


namespace janet_percentage_of_snowballs_l121_121181

-- Define the number of snowballs made by Janet
def janet_snowballs : ℕ := 50

-- Define the number of snowballs made by Janet's brother
def brother_snowballs : ℕ := 150

-- Define the total number of snowballs
def total_snowballs : ℕ := janet_snowballs + brother_snowballs

-- Define the percentage calculation function
def calculate_percentage (part whole : ℕ) : ℚ :=
  (part : ℚ) / (whole : ℚ) * 100

-- Proof statement
theorem janet_percentage_of_snowballs : calculate_percentage janet_snowballs total_snowballs = 25 := 
by
  sorry

end janet_percentage_of_snowballs_l121_121181


namespace sum_of_possible_values_of_x_l121_121485

theorem sum_of_possible_values_of_x :
  ∀ x : ℝ, (x + 2) * (x - 3) = 20 → ∃ s, s = 1 :=
by
  sorry

end sum_of_possible_values_of_x_l121_121485


namespace number_of_ways_to_make_78_rubles_l121_121328

theorem number_of_ways_to_make_78_rubles : ∃ n, n = 5 ∧ ∃ x y : ℕ, 78 = 5 * x + 3 * y := sorry

end number_of_ways_to_make_78_rubles_l121_121328


namespace b101_mod_49_l121_121723

-- Definitions based on conditions
def b (n : ℕ) : ℕ := 5^n + 7^n

-- The formal statement of the proof problem
theorem b101_mod_49 : b 101 % 49 = 12 := by
  sorry

end b101_mod_49_l121_121723


namespace xiao_ming_water_usage_ge_8_l121_121405

def min_monthly_water_usage (x : ℝ) : Prop :=
  ∀ (c : ℝ), c ≥ 15 →
    (c = if x ≤ 5 then x * 1.8 else (5 * 1.8 + (x - 5) * 2)) →
      x ≥ 8

theorem xiao_ming_water_usage_ge_8 : ∃ x : ℝ, min_monthly_water_usage x :=
  sorry

end xiao_ming_water_usage_ge_8_l121_121405


namespace amount_left_after_expenses_l121_121066

namespace GirlScouts

def totalEarnings : ℝ := 30
def poolEntryCosts : ℝ :=
  5 * 3.5 + 3 * 2.0 + 2 * 1.0
def transportationCosts : ℝ :=
  6 * 1.5 + 4 * 0.75
def snackCosts : ℝ :=
  3 * 3.0 + 4 * 2.5 + 3 * 2.0
def totalExpenses : ℝ :=
  poolEntryCosts + transportationCosts + snackCosts
def amountLeft : ℝ :=
  totalEarnings - totalExpenses

theorem amount_left_after_expenses :
  amountLeft = -32.5 :=
by
  sorry

end GirlScouts

end amount_left_after_expenses_l121_121066


namespace find_AD_find_a_rhombus_l121_121844

variable (a : ℝ) (AB AD : ℝ)

-- Problem 1: Given AB = 2, find AD
theorem find_AD (h1 : AB = 2)
    (h_quad : ∀ x, x^2 - (a-4)*x + (a-1) = 0 → x = AB ∨ x = AD) : AD = 5 := sorry

-- Problem 2: Find the value of a such that ABCD is a rhombus
theorem find_a_rhombus (h_quad : ∀ x, x^2 - (a-4)*x + (a-1) = 0 → x = 2 → AB = AD → x = a ∨ AB = AD → x = 10) :
    a = 10 := sorry

end find_AD_find_a_rhombus_l121_121844


namespace range_of_a_l121_121022

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (e^x - a)^2 + x^2 - 2 * a * x + a^2 ≤ 1 / 2) ↔ a = 1 / 2 :=
by
  sorry

end range_of_a_l121_121022


namespace four_color_removal_l121_121775

open SimpleGraph

-- Define the graph as a complete graph on 30 vertices
def K_30 : SimpleGraph (Fin 30) := completeGraph (Fin 30)

-- Define the coloring function
variable (coloring : Fin 30 → Fin 30 → Fin 4)

-- A complete graph is connected
noncomputable def connected_graph : SimpleGraph (Fin 30) := {
  adj := λ v w, v ≠ w,
  symm := λ v w h, h.symm,
  loopless := λ v h, h rfl,
}

-- Given the conditions, prove the statement
theorem four_color_removal :
  ∃ c : Fin 4, ∀ {G' : SimpleGraph (Fin 30)},
    (G'.adj = λ v w, v ≠ w ∧ coloring v w ≠ c) → G'.connected :=
begin
  sorry
end

end four_color_removal_l121_121775


namespace expression_value_l121_121486

theorem expression_value (x y z : ℕ) (hx : x = 5) (hy : y = 4) (hz : z = 3) :
  ( (1 / (y : ℚ)) + (1 / (z : ℚ))) / (1 / (x : ℚ)) = 35 / 12 := by
  sorry

end expression_value_l121_121486


namespace ice_cream_vendor_l121_121606

theorem ice_cream_vendor (choco : ℕ) (mango : ℕ) (sold_choco : ℚ) (sold_mango : ℚ) 
  (h_choco : choco = 50) (h_mango : mango = 54) (h_sold_choco : sold_choco = 3/5) 
  (h_sold_mango : sold_mango = 2/3) : 
  choco - (choco * sold_choco) + mango - (mango * sold_mango) = 38 := 
by 
  sorry

end ice_cream_vendor_l121_121606


namespace chromatic_number_decrease_by_removal_l121_121900

theorem chromatic_number_decrease_by_removal 
  (n r : ℕ) (hn_pos : 0 < n) (hr_pos : 0 < r) :
  ∃ (N : ℕ), ∀ (G : SimpleGraph V) [fintype V] [decidable_rel G.adj]
    (hV_size : fintype.card V ≥ N)
    (hχG : χ G = n),
  ∃ (S : Finset V), S.card = r ∧ χ (G.delete_vertices S) ≥ n - 1 :=
by
  sorry

end chromatic_number_decrease_by_removal_l121_121900


namespace otimes_eq_abs_m_leq_m_l121_121819

noncomputable def otimes (x y : ℝ) : ℝ :=
if x ≤ y then x else y

theorem otimes_eq_abs_m_leq_m' :
  ∀ (m : ℝ), otimes (abs (m - 1)) m = abs (m - 1) → m ∈ Set.Ici (1 / 2) := 
by
  sorry

end otimes_eq_abs_m_leq_m_l121_121819


namespace larger_value_3a_plus_1_l121_121743

theorem larger_value_3a_plus_1 {a : ℝ} (h : 8 * a^2 + 6 * a + 2 = 0) : 3 * a + 1 ≤ 3 * (-1/4 : ℝ) + 1 := 
sorry

end larger_value_3a_plus_1_l121_121743


namespace part1_part2_l121_121876

variables (a b c : ℝ)

theorem part1 (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  ab + bc + ac ≤ 1 / 3 := sorry

theorem part2 (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  1 / a + 1 / b + 1 / c ≥ 9 := sorry

end part1_part2_l121_121876


namespace emma_final_amount_l121_121990

theorem emma_final_amount
  (initial_amount : ℕ)
  (furniture_cost : ℕ)
  (fraction_given_to_anna : ℚ)
  (amount_left : ℕ) :
  initial_amount = 2000 →
  furniture_cost = 400 →
  fraction_given_to_anna = 3 / 4 →
  amount_left = initial_amount - furniture_cost →
  amount_left - (fraction_given_to_anna * amount_left : ℚ) = 400 :=
by
  intros h_initial h_furniture h_fraction h_amount_left
  sorry

end emma_final_amount_l121_121990


namespace required_connections_l121_121551

theorem required_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) : (n * k) / 2 = 60 := by
  sorry

end required_connections_l121_121551


namespace arithmetic_sequence_general_term_sum_sequence_proof_l121_121299

theorem arithmetic_sequence_general_term (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (d : ℝ) (a1 : ℝ)
  (h1 : ∀ n, a_n n = a1 + (n - 1) * d)
  (h2 : d > 0)
  (h3 : a1 * (a1 + 3 * d) = 22)
  (h4 : 4 * a1 + 6 * d = 26) :
  ∀ n, a_n n = 3 * n - 1 := sorry

theorem sum_sequence_proof (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) (T_n : ℕ → ℝ)
  (h1 : ∀ n, a_n n = 3 * n - 1)
  (h2 : ∀ n, b_n n = 1 / (a_n n * a_n (n + 1)))
  (h3 : ∀ n, T_n n = (Finset.range n).sum b_n)
  (n : ℕ) :
  T_n n < 1 / 6 := sorry

end arithmetic_sequence_general_term_sum_sequence_proof_l121_121299


namespace rowing_speed_downstream_l121_121256

theorem rowing_speed_downstream (V_u V_s V_d : ℝ) (h1 : V_u = 10) (h2 : V_s = 15)
  (h3 : V_s = (V_u + V_d) / 2) : V_d = 20 := by
  sorry

end rowing_speed_downstream_l121_121256


namespace find_second_remainder_l121_121946

theorem find_second_remainder (k m n r : ℕ) 
  (h1 : n = 12 * k + 56) 
  (h2 : n = 34 * m + r) 
  (h3 : (22 + r) % 12 = 10) : 
  r = 10 :=
sorry

end find_second_remainder_l121_121946


namespace two_lines_in_3d_space_l121_121130

theorem two_lines_in_3d_space : 
  ∀ x y z : ℝ, x^2 + 2 * x * (y + z) + y^2 = z^2 + 2 * z * (y + x) + x^2 → 
  (∃ a : ℝ, y = -z ∧ x = 0) ∨ (∃ b : ℝ, z = - (2 / 3) * x) :=
  sorry

end two_lines_in_3d_space_l121_121130


namespace lottery_blanks_l121_121496

theorem lottery_blanks (P B : ℕ) (h₁ : P = 10) (h₂ : (P : ℝ) / (P + B) = 0.2857142857142857) : B = 25 := 
by
  sorry

end lottery_blanks_l121_121496


namespace remainder_of_3_pow_21_mod_11_l121_121567

theorem remainder_of_3_pow_21_mod_11 : (3^21 % 11) = 3 := 
by {
  sorry
}

end remainder_of_3_pow_21_mod_11_l121_121567


namespace severe_flood_probability_next_10_years_l121_121117

variable (A B C : Prop)
variable (P : Prop → ℝ)
variable (P_A : P A = 0.8)
variable (P_B : P B = 0.85)
variable (thirty_years_no_flood : ¬A)

theorem severe_flood_probability_next_10_years :
  P C = (P B - P A) / (1 - P A) := by
  sorry

end severe_flood_probability_next_10_years_l121_121117


namespace sin_double_angle_l121_121333

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Define the angle α such that its terminal side passes through point P
noncomputable def α : ℝ := sorry -- The exact definition of α is not needed for this statement

-- Define r as the distance from the origin to the point P
noncomputable def r : ℝ := Real.sqrt ((P.1 ^ 2) + (P.2 ^ 2))

-- Define sin(α) and cos(α)
noncomputable def sin_α : ℝ := P.2 / r
noncomputable def cos_α : ℝ := P.1 / r

-- The proof statement
theorem sin_double_angle : 2 * sin_α * cos_α = -4 / 5 := by
  sorry

end sin_double_angle_l121_121333


namespace behavior_of_f_in_interval_l121_121832

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 3 * m * x + 3

-- Define the property of even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- The theorem statement
theorem behavior_of_f_in_interval (m : ℝ) (hf_even : is_even_function (f m)) :
  m = 0 → (∀ x : ℝ, -4 < x ∧ x < 0 → f 0 x < f 0 (-x)) ∧ (∀ x : ℝ, 0 < x ∧ x < 2 → f 0 (-x) > f 0 x) :=
by 
  sorry

end behavior_of_f_in_interval_l121_121832


namespace abcd_eq_eleven_l121_121317

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
noncomputable def d : ℝ := sorry

-- Conditions on a, b, c, d
axiom cond_a : a = Real.sqrt (4 + Real.sqrt (5 + a))
axiom cond_b : b = Real.sqrt (4 - Real.sqrt (5 + b))
axiom cond_c : c = Real.sqrt (4 + Real.sqrt (5 - c))
axiom cond_d : d = Real.sqrt (4 - Real.sqrt (5 - d))

-- Theorem to prove
theorem abcd_eq_eleven : a * b * c * d = 11 :=
by
  sorry

end abcd_eq_eleven_l121_121317


namespace cos_double_angle_l121_121663

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3 / 5) : Real.cos (2 * theta) = -7 / 25 :=
by
  sorry

end cos_double_angle_l121_121663


namespace problem_statement_l121_121415

def number_of_combinations (n k : ℕ) : ℕ := Nat.choose n k

def successful_outcomes : ℕ :=
  (number_of_combinations 3 1) * (number_of_combinations 5 1) * (number_of_combinations 4 5) +
  (number_of_combinations 3 2) * (number_of_combinations 4 5)

def total_outcomes : ℕ := number_of_combinations 12 7

def probability_at_least_75_cents : ℚ :=
  successful_outcomes / total_outcomes

theorem problem_statement : probability_at_least_75_cents = 3 / 22 := by
  sorry

end problem_statement_l121_121415


namespace last_number_of_nth_row_sum_of_numbers_in_nth_row_position_of_2008_l121_121732

theorem last_number_of_nth_row (n : ℕ) : 
    let last_number := 2^n - 1
    last_number = 2^n - 1 := 
sorry

theorem sum_of_numbers_in_nth_row (n : ℕ) :
    let sum := (3 * 2^(n-3)) - 2^(n-2)
    sum = (3 * 2^(n-3)) - 2^(n-2) :=
sorry

theorem position_of_2008 : 
    let position := 985
    position = 985 :=
sorry

end last_number_of_nth_row_sum_of_numbers_in_nth_row_position_of_2008_l121_121732


namespace smallest_integer_ends_in_3_and_divisible_by_5_l121_121783

theorem smallest_integer_ends_in_3_and_divisible_by_5 : ∃ (n : ℕ), n > 0 ∧ (n % 10 = 3) ∧ (n % 5 = 0) ∧ n = 53 :=
by
  sorry

end smallest_integer_ends_in_3_and_divisible_by_5_l121_121783


namespace solve_for_y_l121_121062

theorem solve_for_y {y : ℝ} : (y - 5)^4 = 16 → y = 7 :=
by
  sorry

end solve_for_y_l121_121062


namespace f_divisible_by_g_l121_121577

noncomputable def f (a b c : ℚ) (x : ℚ) : ℚ := a * x^2 + b * x + c
noncomputable def g (d e : ℚ) (x : ℚ) : ℚ := d * x + e

theorem f_divisible_by_g (a b c d e : ℚ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d) (h_e : 0 < e) : 
  (∀ n : ℕ, 0 < n → ((f a b c n) / (g d e n)) = (f a b c n) / (g d e n) ∧ (f a b c n) % (g d e n) = 0) →
  ∀ x : ℚ, (f a b c x) % (g d e x) = 0 :=
by
  sorry

end f_divisible_by_g_l121_121577


namespace calculate_sum_of_triangles_l121_121903

def operation_triangle (a b c : Int) : Int :=
  a * b - c 

theorem calculate_sum_of_triangles :
  operation_triangle 3 4 5 + operation_triangle 1 2 4 + operation_triangle 2 5 6 = 9 :=
by 
  sorry

end calculate_sum_of_triangles_l121_121903


namespace value_of_x_l121_121318

theorem value_of_x (x : ℝ) (hx_pos : 0 < x) (hx_eq : x^2 = 1024) : x = 32 := 
by
  sorry

end value_of_x_l121_121318


namespace base_d_digit_difference_l121_121065

theorem base_d_digit_difference (A C d : ℕ) (h1 : d > 8)
  (h2 : d * A + C + (d * C + C) = 2 * d^2 + 3 * d + 2) :
  (A - C = d + 1) :=
sorry

end base_d_digit_difference_l121_121065


namespace cos_double_angle_l121_121659

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 := sorry

end cos_double_angle_l121_121659


namespace head_start_ratio_l121_121569

variable (Va Vb L H : ℕ)

-- Conditions
def speed_relation : Prop := Va = (4 * Vb) / 3

-- The head start fraction that makes A and B finish the race at the same time given the speed relation
theorem head_start_ratio (Va Vb L H : ℕ)
  (h1 : speed_relation Va Vb)
  (h2 : L > 0) : (H = L / 4) :=
sorry

end head_start_ratio_l121_121569


namespace min_value_of_sum_of_squares_l121_121999

theorem min_value_of_sum_of_squares (a b c : ℝ) (h : a + b + c = 1) : a^2 + b^2 + c^2 ≥ 1 / 3 :=
by
  sorry

end min_value_of_sum_of_squares_l121_121999


namespace extreme_value_0_at_minus_1_l121_121642

theorem extreme_value_0_at_minus_1 (m n : ℝ)
  (h1 : (-1) + 3 * m - n + m^2 = 0)
  (h2 : 3 - 6 * m + n = 0) :
  m + n = 11 :=
sorry

end extreme_value_0_at_minus_1_l121_121642


namespace infinite_series_sum_l121_121610

theorem infinite_series_sum :
  (∑' n : ℕ, (4 * (n + 1) - 3) / 3 ^ (n + 1)) = 13 / 8 :=
by sorry

end infinite_series_sum_l121_121610


namespace enumerate_set_l121_121760

open Set

def is_positive_integer (x : ℕ) : Prop := x > 0

theorem enumerate_set :
  { p : ℕ × ℕ | p.1 + p.2 = 4 ∧ is_positive_integer p.1 ∧ is_positive_integer p.2 } =
  { (1, 3), (2, 2), (3, 1) } := by 
sorry

end enumerate_set_l121_121760


namespace anna_reading_time_l121_121270

theorem anna_reading_time:
  (∀ n : ℕ, n ∈ (Finset.range 31).filter (λ x, ¬ ∃ k : ℕ, k * 3 + 3 = x + 1) → True) →
  (let chapters_read := (Finset.range 31).filter (λ x, ¬ (∃ k : ℕ, k * 3 + 3 = x + 1)).card,
  reading_time := chapters_read * 20,
  hours := reading_time / 60 in
  hours = 7) :=
by
  intros
  let chapters_read := (Finset.range 31).filter (λ x, ¬ ∃ k : ℕ, k * 3 + 3 = x + 1).card
  have h1 : chapters_read = 21 := by sorry
  let reading_time := chapters_read * 20
  have h2 : reading_time = 420 := by sorry
  let hours := reading_time / 60
  have h3 : hours = 7 := by sorry
  exact h3

end anna_reading_time_l121_121270


namespace stuffed_animals_total_l121_121888

theorem stuffed_animals_total :
  let McKenna := 34
  let Kenley := 2 * McKenna
  let Tenly := Kenley + 5
  McKenna + Kenley + Tenly = 175 :=
by
  sorry

end stuffed_animals_total_l121_121888


namespace flood_monitoring_technology_l121_121204

def geographicInformationTechnologies : Type := String

def RemoteSensing : geographicInformationTechnologies := "Remote Sensing"
def GlobalPositioningSystem : geographicInformationTechnologies := "Global Positioning System"
def GeographicInformationSystem : geographicInformationTechnologies := "Geographic Information System"
def DigitalEarth : geographicInformationTechnologies := "Digital Earth"

def effectiveFloodMonitoring (tech1 tech2 : geographicInformationTechnologies) : Prop :=
  (tech1 = RemoteSensing ∧ tech2 = GeographicInformationSystem) ∨ 
  (tech1 = GeographicInformationSystem ∧ tech2 = RemoteSensing)

theorem flood_monitoring_technology :
  effectiveFloodMonitoring RemoteSensing GeographicInformationSystem :=
by
  sorry

end flood_monitoring_technology_l121_121204


namespace withdraw_representation_l121_121787

-- Define the concept of depositing and withdrawing money.
def deposit (amount : ℕ) : ℤ := amount
def withdraw (amount : ℕ) : ℤ := - amount

-- Define the given condition: depositing $30,000 is represented as $+30,000.
def deposit_condition : deposit 30000 = 30000 := by rfl

-- The statement to be proved: withdrawing $40,000 is represented as $-40,000
theorem withdraw_representation (deposit_condition : deposit 30000 = 30000) : withdraw 40000 = -40000 :=
by
  sorry

end withdraw_representation_l121_121787


namespace solution_l121_121834

noncomputable def problem_statement : Prop :=
  ∃ (x y : ℝ),
    x - y = 1 ∧
    x^3 - y^3 = 2 ∧
    x^4 + y^4 = 23 / 9 ∧
    x^5 - y^5 = 29 / 9

theorem solution : problem_statement := sorry

end solution_l121_121834


namespace third_dog_average_daily_miles_l121_121439

/-- Bingo has three dogs. On average, they walk a total of 100 miles a week.

    The first dog walks an average of 2 miles a day.

    The second dog walks 1 mile if it is an odd day of the month and 3 miles if it is an even day of the month.

    Considering a 30-day month, the goal is to find the average daily miles of the third dog. -/
theorem third_dog_average_daily_miles :
  let total_dogs := 3
  let weekly_total_miles := 100
  let first_dog_daily_miles := 2
  let second_dog_odd_day_miles := 1
  let second_dog_even_day_miles := 3
  let days_in_month := 30
  let odd_days_in_month := 15
  let even_days_in_month := 15
  let weeks_in_month := days_in_month / 7
  let first_dog_monthly_miles := days_in_month * first_dog_daily_miles
  let second_dog_monthly_miles := (second_dog_odd_day_miles * odd_days_in_month) + (second_dog_even_day_miles * even_days_in_month)
  let third_dog_monthly_miles := (weekly_total_miles * weeks_in_month) - (first_dog_monthly_miles + second_dog_monthly_miles)
  let third_dog_daily_miles := third_dog_monthly_miles / days_in_month
  third_dog_daily_miles = 10.33 :=
by
  sorry

end third_dog_average_daily_miles_l121_121439


namespace area_triangle_ABC_l121_121175

noncomputable def point := ℝ × ℝ

structure Parallelogram (A B C D : point) : Prop :=
(parallel_AB_CD : ∃ m1 m2, m1 ≠ m2 ∧ (A.2 - B.2) / (A.1 - B.1) = m1 ∧ (C.2 - D.2) / (C.1 - D.1) = m2)
(equal_heights : ∃ h, (B.2 - A.2 = h) ∧ (C.2 - D.2 = h))
(area_parallelogram : (B.1 - A.1) * (B.2 - A.2) + (C.1 - D.1) * (C.2 - D.2) = 27)
(thrice_length : (C.1 - D.1) = 3 * (B.1 - A.1))

theorem area_triangle_ABC (A B C D : point) (h : Parallelogram A B C D) : 
  ∃ triangle_area : ℝ, triangle_area = 13.5 :=
by
  sorry

end area_triangle_ABC_l121_121175


namespace vanessa_score_l121_121857

theorem vanessa_score (total_points team_score other_players_avg_score: ℝ) : 
  total_points = 72 ∧ team_score = 7 ∧ other_players_avg_score = 4.5 → 
  ∃ vanessa_points: ℝ, vanessa_points = 40.5 :=
by
  sorry

end vanessa_score_l121_121857


namespace Tom_initial_investment_l121_121225

noncomputable def Jose_investment : ℝ := 45000
noncomputable def Jose_investment_time : ℕ := 10
noncomputable def total_profit : ℝ := 36000
noncomputable def Jose_share : ℝ := 20000
noncomputable def Tom_share : ℝ := total_profit - Jose_share
noncomputable def Tom_investment_time : ℕ := 12
noncomputable def proportion_Tom : ℝ := (4 : ℝ) / 5
noncomputable def Tom_expected_investment : ℝ := 6000

theorem Tom_initial_investment (T : ℝ) (h1 : Jose_investment = 45000)
                               (h2 : Jose_investment_time = 10)
                               (h3 : total_profit = 36000)
                               (h4 : Jose_share = 20000)
                               (h5 : Tom_investment_time = 12)
                               (h6 : Tom_share = 16000)
                               (h7 : proportion_Tom = (4 : ℝ) / 5)
                               : T = Tom_expected_investment :=
by
  sorry

end Tom_initial_investment_l121_121225


namespace cos_double_angle_l121_121692

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3/5) : cos (2*θ) = -7/25 := 
by 
  sorry

end cos_double_angle_l121_121692


namespace definite_integral_eval_l121_121821

theorem definite_integral_eval :
  ∫ x in (1:ℝ)..(3:ℝ), (2 * x - 1 / x ^ 2) = 22 / 3 :=
by
  sorry

end definite_integral_eval_l121_121821


namespace all_div_by_25_form_no_div_by_35_l121_121287

noncomputable def exists_div_by_25 (M : ℕ) : Prop :=
∃ (M N : ℕ) (n : ℕ), M = 6 * 10 ^ (n - 1) + N ∧ M = 25 * N ∧ 4 * N = 10 ^ (n - 1)

theorem all_div_by_25_form :
  ∀ M, exists_div_by_25 M → (∃ k : ℕ, M = 625 * 10 ^ k) :=
by
  intro M
  intro h
  sorry

noncomputable def not_exists_div_by_35 (M : ℕ) : Prop :=
∀ (M N : ℕ) (n : ℕ), M ≠ 6 * 10 ^ (n - 1) + N ∨ M ≠ 35 * N

theorem no_div_by_35 :
  ∀ M, not_exists_div_by_35 M :=
by
  intro M
  intro h
  sorry

end all_div_by_25_form_no_div_by_35_l121_121287


namespace max_min_rounded_value_l121_121601

theorem max_min_rounded_value (n : ℝ) (h : 3.75 ≤ n ∧ n < 3.85) : 
  (∀ n, 3.75 ≤ n ∧ n < 3.85 → n ≤ 3.84 ∧ n ≥ 3.75) :=
sorry

end max_min_rounded_value_l121_121601


namespace bob_spending_over_limit_l121_121052

theorem bob_spending_over_limit : 
  ∀ (necklace_price book_price limit total_cost amount_over_limit : ℕ),
  necklace_price = 34 →
  book_price = necklace_price + 5 →
  limit = 70 →
  total_cost = necklace_price + book_price →
  amount_over_limit = total_cost - limit →
  amount_over_limit = 3 :=
by
  intros
  sorry

end bob_spending_over_limit_l121_121052


namespace debra_probability_l121_121444

theorem debra_probability :
  let p_tail := 1 / 2
  let p_head := 1 / 2
  let p_THTHT := p_tail * p_head * p_tail * p_head * p_tail
  let P := (1 / 4) / (3 / 4)
  (p_THTHT * P) = 1 / 96 :=
by
  -- Definitions of p_tail, p_head, p_THTHT, and P
  let p_tail := 1 / 2
  let p_head := 1 / 2
  let p_THTHT := p_tail * p_head * p_tail * p_head * p_tail
  let P := (1 / 4) / (3 / 4)
  -- Placeholder for proof computation
  sorry

end debra_probability_l121_121444


namespace seashells_in_jar_at_end_of_month_l121_121137

noncomputable def seashells_in_week (initial: ℕ) (increment: ℕ) (week: ℕ) : ℕ :=
  initial + increment * week

theorem seashells_in_jar_at_end_of_month :
  seashells_in_week 50 20 0 +
  seashells_in_week 50 20 1 +
  seashells_in_week 50 20 2 +
  seashells_in_week 50 20 3 = 320 :=
sorry

end seashells_in_jar_at_end_of_month_l121_121137


namespace students_selected_juice_l121_121437

def fraction_of_students_choosing_juice (students_selected_juice_ratio students_selected_soda_ratio : ℚ) : ℚ :=
  students_selected_juice_ratio / students_selected_soda_ratio

def num_students_selecting (students_selected_soda : ℕ) (fraction_juice : ℚ) : ℚ :=
  fraction_juice * students_selected_soda

theorem students_selected_juice (students_selected_soda : ℕ) : students_selected_soda = 120 ∧
    (fraction_of_students_choosing_juice 0.15 0.75) = 1/5 →
    num_students_selecting students_selected_soda (fraction_of_students_choosing_juice 0.15 0.75) = 24 :=
by
  intros h
  sorry

end students_selected_juice_l121_121437


namespace mul_inv_mod_35_l121_121286

theorem mul_inv_mod_35 : (8 * 22) % 35 = 1 := 
  sorry

end mul_inv_mod_35_l121_121286


namespace geometric_sequence_iff_arithmetic_sequence_l121_121469

/-
  Suppose that {a_n} is an infinite geometric sequence with common ratio q, where q^2 ≠ 1.
  Also suppose that {b_n} is a sequence of positive natural numbers (ℕ).
  Prove that {a_{b_n}} forms a geometric sequence if and only if {b_n} forms an arithmetic sequence.
-/

theorem geometric_sequence_iff_arithmetic_sequence
  (a : ℕ → ℕ) (b : ℕ → ℕ) (q : ℝ)
  (h_geom_a : ∃ a1, ∀ n, a n = a1 * q ^ (n - 1))
  (h_q_squared_ne_one : q^2 ≠ 1)
  (h_bn_positive : ∀ n, 0 < b n) :
  (∃ a1, ∃ q', ∀ n, a (b n) = a1 * q' ^ n) ↔ (∃ d, ∀ n, b (n + 1) - b n = d) := 
sorry

end geometric_sequence_iff_arithmetic_sequence_l121_121469


namespace Rockets_won_38_games_l121_121827

-- Definitions for each team and their respective wins
variables (Sharks Dolphins Rockets Wolves Comets : ℕ)
variables (wins : Finset ℕ)
variables (shArks_won_more_than_Dolphins : Sharks > Dolphins)
variables (rockets_won_more_than_Wolves : Rockets > Wolves)
variables (rockets_won_fewer_than_Comets : Rockets < Comets)
variables (Wolves_won_more_than_25_games : Wolves > 25)
variables (possible_wins : wins = {28, 33, 38, 43})

-- Statement that the Rockets won 38 games given the conditions
theorem Rockets_won_38_games
  (shArks_won_more_than_Dolphins : Sharks > Dolphins)
  (rockets_won_more_than_Wolves : Rockets > Wolves)
  (rockets_won_fewer_than_Comets : Rockets < Comets)
  (Wolves_won_more_than_25_games : Wolves > 25)
  (possible_wins : wins = {28, 33, 38, 43}) :
  Rockets = 38 :=
sorry

end Rockets_won_38_games_l121_121827


namespace anna_reading_time_l121_121267

theorem anna_reading_time 
  (C : ℕ)
  (T_per_chapter : ℕ)
  (hC : C = 31) 
  (hT : T_per_chapter = 20) :
  (C - (C / 3)) * T_per_chapter / 60 = 7 := 
by 
  -- proof steps will go here
  sorry

end anna_reading_time_l121_121267


namespace saved_percentage_is_correct_l121_121811

def rent : ℝ := 5000
def milk : ℝ := 1500
def groceries : ℝ := 4500
def education : ℝ := 2500
def petrol : ℝ := 2000
def miscellaneous : ℝ := 5200
def amount_saved : ℝ := 2300

noncomputable def total_expenses : ℝ :=
  rent + milk + groceries + education + petrol + miscellaneous

noncomputable def total_salary : ℝ :=
  total_expenses + amount_saved

noncomputable def percentage_saved : ℝ :=
  (amount_saved / total_salary) * 100

theorem saved_percentage_is_correct :
  percentage_saved = 8.846 := by
  sorry

end saved_percentage_is_correct_l121_121811


namespace power_function_evaluation_l121_121838

noncomputable def f (α : ℝ) (x : ℝ) := x ^ α

theorem power_function_evaluation (α : ℝ) (h : f α 8 = 2) : f α (-1/8) = -1/2 :=
by
  sorry

end power_function_evaluation_l121_121838


namespace correct_choice_l121_121114

variable (a b : ℝ) (p q : Prop) (x : ℝ)

-- Proposition A: Incorrect because x > 3 is a sufficient condition for x > 2.
def propositionA : Prop := (∀ x : ℝ, x > 3 → x > 2) ∧ ¬ (∀ x : ℝ, x > 2 → x > 3)

-- Proposition B: Incorrect negation form.
def propositionB : Prop := ¬ (¬p → ¬q) ∧ (q → p)

-- Proposition C: Incorrect because it should be 1/a > 1/b given 0 < a < b.
def propositionC : Prop := (a > 0 ∧ b < 0) ∧ ¬ (1/a < 1/b)

-- Proposition D: Correct negation form.
def propositionD_negation_correct : Prop := 
  (¬ ∃ x : ℝ, x^2 = 1) = ( ∀ x : ℝ, x^2 ≠ 1)

theorem correct_choice : propositionD_negation_correct := by
  sorry

end correct_choice_l121_121114


namespace cricketer_average_after_22nd_inning_l121_121097

theorem cricketer_average_after_22nd_inning (A : ℚ) 
  (h1 : 21 * A + 134 = (A + 3.5) * 22)
  (h2 : 57 = A) :
  A + 3.5 = 60.5 :=
by
  exact sorry

end cricketer_average_after_22nd_inning_l121_121097


namespace union_A_B_eq_l121_121837

def A := { x : ℝ | real.log (x - 1) < 0 }
def B := { y : ℝ | ∃ x : ℝ, x ∈ A ∧ y = 2^x - 1 }

theorem union_A_B_eq : A ∪ B = {y : ℝ | 1 < y ∧ y < 3} :=
by sorry

end union_A_B_eq_l121_121837


namespace distinct_solutions_difference_l121_121190

theorem distinct_solutions_difference (r s : ℝ) (h1 : r ≠ s) (h2 : r > s)
  (h_eq : ∀ x : ℝ, (x - 5) * (x + 5) = 25 * x - 125 ↔ x = r ∨ x = s) : r - s = 15 :=
begin
  sorry
end

end distinct_solutions_difference_l121_121190


namespace units_digit_multiplication_l121_121623

-- Define a function to find the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Statement of the problem: Given the product 27 * 36, prove that the units digit is 2.
theorem units_digit_multiplication (a b : ℕ) (h1 : units_digit 27 = 7) (h2 : units_digit 36 = 6) :
  units_digit (27 * 36) = 2 :=
by
  have h3 : units_digit (7 * 6) = 2 := by sorry
  exact h3

end units_digit_multiplication_l121_121623


namespace binomial_coefficient_middle_term_l121_121860

theorem binomial_coefficient_middle_term :
  let n := 11
  let sum_odd := 1024
  sum_odd = 2^(n-1) →
  let binom_coef := Nat.choose n (n / 2 - 1)
  binom_coef = 462 :=
by
  intro n
  let n := 11
  intro sum_odd
  let sum_odd := 1024
  intro h
  let binom_coef := Nat.choose n (n / 2 - 1)
  have : binom_coef = 462 := sorry
  exact this

end binomial_coefficient_middle_term_l121_121860


namespace rectangle_area_288_l121_121261

/-- A rectangle contains eight circles arranged in a 2x4 grid. Each circle has a radius of 3 inches.
    We are asked to prove that the area of the rectangle is 288 square inches. --/
noncomputable def circle_radius : ℝ := 3
noncomputable def circles_per_width : ℕ := 2
noncomputable def circles_per_length : ℕ := 4
noncomputable def circle_diameter : ℝ := 2 * circle_radius
noncomputable def rectangle_width : ℝ := circles_per_width * circle_diameter
noncomputable def rectangle_length : ℝ := circles_per_length * circle_diameter
noncomputable def rectangle_area : ℝ := rectangle_length * rectangle_width

theorem rectangle_area_288 :
  rectangle_area = 288 :=
by
  -- Proof of the area will be filled in here.
  sorry

end rectangle_area_288_l121_121261


namespace cos_double_angle_l121_121661

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 := sorry

end cos_double_angle_l121_121661


namespace probability_of_passing_l121_121976

theorem probability_of_passing (pA pB pC : ℝ) (hA : pA = 0.8) (hB : pB = 0.6) (hC : pC = 0.5) :
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 0.96 := by
  sorry

end probability_of_passing_l121_121976


namespace present_ages_l121_121921

theorem present_ages
  (R D K : ℕ) (x : ℕ)
  (H1 : R = 4 * x)
  (H2 : D = 3 * x)
  (H3 : K = 5 * x)
  (H4 : R + 6 = 26)
  (H5 : (R + 8) + (D + 8) = K) :
  D = 15 ∧ K = 51 :=
sorry

end present_ages_l121_121921


namespace boat_license_combinations_l121_121597

theorem boat_license_combinations :
  let letters := ['A', 'M', 'S']
  let non_zero_digits := [1, 2, 3, 4, 5, 6, 7, 8, 9]
  let any_digit := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  3 * 9 * 10^4 = 270000 := 
by 
  sorry

end boat_license_combinations_l121_121597


namespace prove_AB_and_circle_symmetry_l121_121173

-- Definition of point A
def pointA : ℝ × ℝ := (4, -3)

-- Lengths relation |AB| = 2|OA|
def lengths_relation(u v : ℝ) : Prop :=
  u^2 + v^2 = 100

-- Orthogonality condition for AB and OA
def orthogonality_condition(u v : ℝ) : Prop :=
  4 * u - 3 * v = 0

-- Condition that ordinate of B is greater than 0
def ordinate_condition(v : ℝ) : Prop :=
  v - 3 > 0

-- Equation of the circle given in the problem
def given_circle_eqn(x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 1)^2 = 10

-- Symmetric circle equation to be proved
def symmetric_circle_eqn(x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 10

theorem prove_AB_and_circle_symmetry :
  (∃ u v : ℝ, lengths_relation u v ∧ orthogonality_condition u v ∧ ordinate_condition v ∧ u = 6 ∧ v = 8) ∧
  (∃ x y : ℝ, given_circle_eqn x y → symmetric_circle_eqn x y) :=
by
  sorry

end prove_AB_and_circle_symmetry_l121_121173


namespace base6_addition_l121_121940

/-- Adding two numbers in base 6 -/
theorem base6_addition : (3454 : ℕ) + (12345 : ℕ) = (142042 : ℕ) := by
  sorry

end base6_addition_l121_121940


namespace positive_divisors_of_x_l121_121115

theorem positive_divisors_of_x (x : ℕ) (h : ∀ d : ℕ, d ∣ x^3 → d = 1 ∨ d = x^3 ∨ d ∣ x^2) : (∀ d : ℕ, d ∣ x → d = 1 ∨ d = x ∨ d ∣ p) :=
by
  sorry

end positive_divisors_of_x_l121_121115


namespace sin_pi_minus_a_l121_121997

theorem sin_pi_minus_a (a : ℝ) (h_cos_a : Real.cos a = Real.sqrt 5 / 3) (h_range_a : a ∈ Set.Ioo (-Real.pi / 2) 0) : 
  Real.sin (Real.pi - a) = -2 / 3 :=
by sorry

end sin_pi_minus_a_l121_121997


namespace cos_double_angle_l121_121703

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7/25 :=
by
  sorry

end cos_double_angle_l121_121703


namespace find_side_length_l121_121757

theorem find_side_length (x : ℝ) : 
  (4 * x = 8 * Real.pi) → (x = Real.pi * 2) :=
by
  intro h
  calc
    x = (8 * Real.pi) / 4 : by linarith
    ... = 2 * Real.pi : by linarith

#eval Float.toString (2 * Float.pi)

end find_side_length_l121_121757


namespace sum_even_less_100_correct_l121_121123

-- Define the sequence of even, positive integers less than 100
def even_seq (n : ℕ) : Prop := n % 2 = 0 ∧ 0 < n ∧ n < 100

-- Sum of the first n positive integers
def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Sum of the even, positive integers less than 100
def sum_even_less_100 : ℕ := 2 * sum_n 49

theorem sum_even_less_100_correct : sum_even_less_100 = 2450 := by
  sorry

end sum_even_less_100_correct_l121_121123


namespace rate_of_grapes_l121_121314

theorem rate_of_grapes (G : ℝ) (H : 8 * G + 9 * 50 = 1010) : G = 70 := by
  sorry

end rate_of_grapes_l121_121314


namespace new_boarders_joined_l121_121924

theorem new_boarders_joined (boarders_initial day_students_initial boarders_final x : ℕ)
  (h1 : boarders_initial = 220)
  (h2 : (5:ℕ) * day_students_initial = (12:ℕ) * boarders_initial)
  (h3 : day_students_initial = 528)
  (h4 : (1:ℕ) * day_students_initial = (2:ℕ) * (boarders_initial + x)) :
  x = 44 := by
  sorry

end new_boarders_joined_l121_121924


namespace annual_growth_rate_l121_121030

theorem annual_growth_rate (x : ℝ) (h : 2000 * (1 + x) ^ 2 = 2880) : x = 0.2 :=
by sorry

end annual_growth_rate_l121_121030


namespace florist_picked_roses_l121_121968

def initial_roses : ℕ := 11
def sold_roses : ℕ := 2
def final_roses : ℕ := 41
def remaining_roses := initial_roses - sold_roses
def picked_roses := final_roses - remaining_roses

theorem florist_picked_roses : picked_roses = 32 :=
by
  -- This is where the proof would go, but we are leaving it empty on purpose
  sorry

end florist_picked_roses_l121_121968


namespace distinct_solutions_diff_l121_121195

theorem distinct_solutions_diff (r s : ℝ) (hr : r > s) 
  (h : ∀ x, (x - 5) * (x + 5) = 25 * x - 125 ↔ x = r ∨ x = s) : r - s = 15 :=
sorry

end distinct_solutions_diff_l121_121195


namespace katrina_cookies_left_l121_121339

def initial_cookies : ℕ := 120
def morning_sales : ℕ := 3 * 12
def lunch_sales : ℕ := 57
def afternoon_sales : ℕ := 16
def total_sales : ℕ := morning_sales + lunch_sales + afternoon_sales
def cookies_left_to_take_home (initial: ℕ) (sold: ℕ) : ℕ := initial - sold

theorem katrina_cookies_left :
  cookies_left_to_take_home initial_cookies total_sales = 11 :=
by sorry

end katrina_cookies_left_l121_121339


namespace chairs_adjacent_subsets_l121_121395

theorem chairs_adjacent_subsets (n : ℕ) (h_n : n = 12) :
  (∑ k in (range n.succ).filter (λ k, k ≥ 4), (nat.choose n k)) + 84 = 1704 :=
by sorry

end chairs_adjacent_subsets_l121_121395


namespace angle_degree_measure_l121_121942

theorem angle_degree_measure (x : ℝ) (h1 : (x + (90 - x) = 90)) (h2 : (x = 3 * (90 - x))) : x = 67.5 := by
  sorry

end angle_degree_measure_l121_121942


namespace magic_king_total_episodes_l121_121927

theorem magic_king_total_episodes :
  (∑ i in finset.range 5, 20) + (∑ j in finset.range 5, 25) = 225 :=
by sorry

end magic_king_total_episodes_l121_121927


namespace final_quantity_of_milk_l121_121091

-- Initially, a vessel is filled with 45 litres of pure milk
def initial_milk : Nat := 45

-- First operation: removing 9 litres of milk and replacing with water
def first_operation_milk(initial_milk : Nat) : Nat := initial_milk - 9
def first_operation_water : Nat := 9

-- Second operation: removing 9 litres of the mixture and replacing with water
def milk_fraction_mixture(milk : Nat) (total : Nat) : Rat := milk / total
def water_fraction_mixture(water : Nat) (total : Nat) : Rat := water / total

def second_operation_milk(milk : Nat) (total : Nat) (removed : Nat) : Rat := 
  milk - (milk_fraction_mixture milk total) * removed
def second_operation_water(water : Nat) (total : Nat) (removed : Nat) : Rat := 
  water - (water_fraction_mixture water total) * removed + removed

-- Prove the final quantity of milk
theorem final_quantity_of_milk : second_operation_milk 36 45 9 = 28.8 := by
  sorry

end final_quantity_of_milk_l121_121091


namespace cube_sphere_volume_ratio_l121_121098

theorem cube_sphere_volume_ratio (s : ℝ) (r : ℝ) (h : r = (Real.sqrt 3 * s) / 2):
  (s^3) / ((4 / 3) * Real.pi * r^3) = (2 * Real.sqrt 3) / Real.pi :=
by
  sorry

end cube_sphere_volume_ratio_l121_121098


namespace decimal_111_to_base_5_l121_121818

def decimal_to_base_5 (n : ℕ) : ℕ :=
  let rec loop (n : ℕ) (acc : ℕ) (place : ℕ) :=
    if n = 0 then acc
    else 
      let rem := n % 5
      let q := n / 5
      loop q (acc + rem * place) (place * 10)
  loop n 0 1

theorem decimal_111_to_base_5 : decimal_to_base_5 111 = 421 :=
  sorry

end decimal_111_to_base_5_l121_121818


namespace susan_strawberries_per_handful_l121_121744

-- Definitions of the given conditions
def total_picked := 75
def total_needed := 60
def strawberries_per_handful := 5

-- Derived conditions
def total_eaten := total_picked - total_needed
def number_of_handfuls := total_picked / strawberries_per_handful
def strawberries_eaten_per_handful := total_eaten / number_of_handfuls

-- The theorem we want to prove
theorem susan_strawberries_per_handful : strawberries_eaten_per_handful = 1 :=
by sorry

end susan_strawberries_per_handful_l121_121744


namespace paper_area_l121_121105

variable (L W : ℕ)

theorem paper_area (h1 : 2 * L + W = 34) (h2 : L + 2 * W = 38) : L * W = 140 := by
  sorry

end paper_area_l121_121105


namespace ratio_of_A_to_B_l121_121595

theorem ratio_of_A_to_B (v_A v_B : ℝ) (d_A d_B : ℝ) (h1 : d_A = 128) (h2 : d_B = 64) (h3 : d_A / v_A = d_B / v_B) : v_A / v_B = 2 := 
by
  sorry

end ratio_of_A_to_B_l121_121595


namespace problem_l121_121841

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

theorem problem (x : ℝ) :
  (∀ ε > 0, ∃ N : ℝ, ∀ m n : ℝ, m > N → n > N → Real.dist (f (m + ε)) (f m) < ε ∧
   Real.dist (f (n + ε)) (f n) < ε) ∧
  (Real.continuity_point f (-Real.pi / 6)) :=
begin
  split,
  { sorry }, -- Here's where the proof for periodicity would go
  { sorry }  -- Here's where the proof for the center of symmetry would go
end

#check @Real.sin_periodic
#check @Real.is_symm

end problem_l121_121841


namespace possible_values_of_p_l121_121846

theorem possible_values_of_p (p : ℕ) (a b : ℕ) (h_fact : (x : ℤ) → x^2 - 5 * x + p = (x - a) * (x - b))
  (h1 : a + b = 5) (h2 : 1 ≤ a ∧ a ≤ 4) (h3 : 1 ≤ b ∧ b ≤ 4) : 
  p = 4 ∨ p = 6 :=
sorry

end possible_values_of_p_l121_121846


namespace cos_double_angle_l121_121671

variable (θ : Real)
variables (h : cos θ = 3 / 5)

theorem cos_double_angle :
  cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l121_121671


namespace exercise_l121_121235

-- Define a, b, and the identity
def a : ℕ := 45
def b : ℕ := 15

-- Theorem statement
theorem exercise :
  ((a + b) ^ 2 - (a ^ 2 + b ^ 2) = 1350) :=
by
  sorry

end exercise_l121_121235


namespace distance_after_four_steps_l121_121973

theorem distance_after_four_steps (total_distance : ℝ) (steps : ℕ) (steps_taken : ℕ) :
   total_distance = 25 → steps = 7 → steps_taken = 4 → (steps_taken * (total_distance / steps) = 100 / 7) :=
by
    intro h1 h2 h3
    rw [h1, h2, h3]
    simp
    sorry

end distance_after_four_steps_l121_121973


namespace total_pins_cardboard_l121_121789

theorem total_pins_cardboard {length width pins : ℕ} (h_length : length = 34) (h_width : width = 14) (h_pins : pins = 35) :
  2 * pins * (length + width) / (length + width) = 140 :=
by
  sorry

end total_pins_cardboard_l121_121789


namespace first_day_more_than_300_l121_121199

def paperclips (n : ℕ) : ℕ := 4 * 3^n

theorem first_day_more_than_300 : ∃ n, paperclips n > 300 ∧ n = 4 := by
  sorry

end first_day_more_than_300_l121_121199


namespace range_of_y_l121_121704

theorem range_of_y (y : ℝ) (hy : y < 0) (hceil_floor : ⌈y⌉ * ⌊y⌋ = 72) : y ∈ Ioo (-9 : ℝ) (-8 : ℝ) :=
sorry

end range_of_y_l121_121704


namespace midpoint_product_zero_l121_121197

theorem midpoint_product_zero (x y : ℝ)
  (h_midpoint_x : (2 + x) / 2 = 4)
  (h_midpoint_y : (6 + y) / 2 = 3) :
  x * y = 0 :=
by
  sorry

end midpoint_product_zero_l121_121197


namespace diagonal_ratio_l121_121215

variable (a b : ℝ)
variable (d1 : ℝ) -- diagonal length of the first square
variable (r : ℝ := 1.5) -- ratio between perimeters

theorem diagonal_ratio (h : 4 * a / (4 * b) = r) (hd1 : d1 = a * Real.sqrt 2) : 
  (b * Real.sqrt 2) = (2/3) * d1 := 
sorry

end diagonal_ratio_l121_121215


namespace parabola_shift_right_l121_121907

theorem parabola_shift_right (x : ℝ) :
  let original_parabola := - (1 / 2) * x^2
  let shifted_parabola := - (1 / 2) * (x - 1)^2
  original_parabola = shifted_parabola :=
sorry

end parabola_shift_right_l121_121907


namespace cubes_difference_l121_121295

theorem cubes_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 25) : a^3 - b^3 = 99 := 
sorry

end cubes_difference_l121_121295


namespace long_sleeve_shirts_correct_l121_121608

def total_shirts : ℕ := 9
def short_sleeve_shirts : ℕ := 4
def long_sleeve_shirts : ℕ := total_shirts - short_sleeve_shirts

theorem long_sleeve_shirts_correct : long_sleeve_shirts = 5 := by
  sorry

end long_sleeve_shirts_correct_l121_121608


namespace stamps_max_l121_121021

theorem stamps_max (price_per_stamp : ℕ) (total_cents : ℕ) (h1 : price_per_stamp = 25) (h2 : total_cents = 5000) : 
  ∃ n : ℕ, (n * price_per_stamp ≤ total_cents) ∧ (∀ m : ℕ, (m > n) → (m * price_per_stamp > total_cents)) ∧ n = 200 := 
by
  sorry

end stamps_max_l121_121021


namespace n_squared_plus_n_is_even_l121_121060

theorem n_squared_plus_n_is_even (n : ℤ) : Even (n^2 + n) :=
by
  sorry

end n_squared_plus_n_is_even_l121_121060


namespace badges_exchange_l121_121560

theorem badges_exchange (Vasya_initial Tolya_initial : ℕ) 
    (h1 : Vasya_initial = Tolya_initial + 5)
    (h2 : Vasya_initial - 0.24 * Vasya_initial + 0.20 * Tolya_initial = Tolya_initial - 0.20 * Tolya_initial + 0.24 * Vasya_initial - 1) 
    : Vasya_initial = 50 ∧ Tolya_initial = 45 :=
by sorry

end badges_exchange_l121_121560


namespace watermelons_remaining_l121_121812

theorem watermelons_remaining :
  let initial_watermelons := 10 * 12
  let yesterdays_sale := 0.40 * initial_watermelons
  let remaining_after_yesterday := initial_watermelons - yesterdays_sale
  let todays_sale := (1 / 4) * remaining_after_yesterday
  let remaining_after_today := remaining_after_yesterday - todays_sale
  let tomorrows_sales := 1.5 * todays_sale
  let remaining_after_tomorrow := remaining_after_today - tomorrows_sales
  remaining_after_tomorrow = 27 :=
by
  sorry

end watermelons_remaining_l121_121812


namespace horse_drinking_water_l121_121035

-- Definitions and conditions

def initial_horses : ℕ := 3
def added_horses : ℕ := 5
def total_horses : ℕ := initial_horses + added_horses
def bathing_water_per_day : ℕ := 2
def total_water_28_days : ℕ := 1568
def days : ℕ := 28
def daily_water_total : ℕ := total_water_28_days / days

-- The statement looking to prove
theorem horse_drinking_water (D : ℕ) : 
  (total_horses * (D + bathing_water_per_day) = daily_water_total) → 
  D = 5 := 
by
  -- Add proof steps here
  sorry

end horse_drinking_water_l121_121035


namespace consequence_of_implication_l121_121414

-- Define the conditions
variable (A B : Prop)

-- State the theorem to prove
theorem consequence_of_implication (h : B → A) : A → B := 
  sorry

end consequence_of_implication_l121_121414


namespace find_intersection_find_range_of_a_l121_121644

-- Define the sets A and B
def A : Set ℝ := { x : ℝ | x < -2 ∨ (3 < x ∧ x < 4) }
def B : Set ℝ := { x : ℝ | -3 ≤ x ∧ x ≤ 5 }

-- Proof Problem 1: Prove the intersection A ∩ B
theorem find_intersection : (A ∩ B) = { x : ℝ | 3 < x ∧ x ≤ 5 } := by
  sorry

-- Define the set C and the condition B ∩ C = B
def C (a : ℝ) : Set ℝ := { x : ℝ | x ≥ a }
def condition (a : ℝ) : Prop := B ∩ C a = B

-- Proof Problem 2: Find the range of a
theorem find_range_of_a : ∀ a : ℝ, condition a → a ≤ -3 := by
  sorry

end find_intersection_find_range_of_a_l121_121644


namespace shelves_for_coloring_books_l121_121598

theorem shelves_for_coloring_books (initial_stock sold donated per_shelf remaining total_used needed_shelves : ℕ) 
    (h_initial : initial_stock = 150)
    (h_sold : sold = 55)
    (h_donated : donated = 30)
    (h_per_shelf : per_shelf = 12)
    (h_total_used : total_used = sold + donated)
    (h_remaining : remaining = initial_stock - total_used)
    (h_needed_shelves : (remaining + per_shelf - 1) / per_shelf = needed_shelves) :
    needed_shelves = 6 :=
by
  sorry

end shelves_for_coloring_books_l121_121598


namespace most_reasonable_sampling_method_l121_121031

-- Define the conditions
axiom significant_differences_in_educational_stages : Prop
axiom insignificant_differences_between_genders : Prop

-- Define the options
inductive SamplingMethod
| SimpleRandomSampling
| StratifiedSamplingByGender
| StratifiedSamplingByEducationalStage
| SystematicSampling

-- State the problem as a theorem
theorem most_reasonable_sampling_method
  (H1 : significant_differences_in_educational_stages)
  (H2 : insignificant_differences_between_genders) :
  SamplingMethod.StratifiedSamplingByEducationalStage = SamplingMethod.StratifiedSamplingByEducationalStage :=
by
  -- Proof is skipped
  sorry

end most_reasonable_sampling_method_l121_121031


namespace square_segment_ratio_l121_121714

theorem square_segment_ratio
  (A B C D E M P Q : ℝ × ℝ)
  (h_square: A = (0, 16) ∧ B = (16, 16) ∧ C = (16, 0) ∧ D = (0, 0))
  (h_E: E = (7, 0))
  (h_midpoint: M = ((0 + 7) / 2, (16 + 0) / 2))
  (h_bisector_P: P = (P.1, 16) ∧ (16 - 8 = (7 / 16) * (P.1 - 3.5)))
  (h_bisector_Q: Q = (Q.1, 0) ∧ (0 - 8 = (7 / 16) * (Q.1 - 3.5)))
  (h_PM: abs (16 - 8) = abs (P.2 - M.2))
  (h_MQ: abs (8 - 0) = abs (M.2 - Q.2)) :
  abs (P.2 - M.2) = abs (M.2 - Q.2) :=
sorry

end square_segment_ratio_l121_121714


namespace determine_f_l121_121477

noncomputable def f : ℝ → ℝ := sorry

axiom f_0 : f 0 = 1
axiom f_eq (x y : ℝ) : f (x * y + 1) = f x * f y - f y - x + 2

theorem determine_f (x : ℝ) : f x = x + 1 := by
  sorry

end determine_f_l121_121477


namespace cos_double_angle_l121_121687

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3/5) : Real.cos (2 * theta) = -7/25 :=
by
  sorry

end cos_double_angle_l121_121687


namespace lily_disproves_tom_claim_l121_121291

-- Define the cards and the claim
inductive Card
| A : Card
| R : Card
| Circle : Card
| Square : Card
| Triangle : Card

def has_consonant (c : Card) : Prop :=
  match c with
  | Card.R => true
  | _ => false

def has_triangle (c : Card) : Card → Prop :=
  fun c' =>
    match c with
    | Card.R => c' = Card.Triangle
    | _ => true

def tom_claim (c : Card) (c' : Card) : Prop :=
  has_consonant c → has_triangle c c'

-- Proof problem statement:
theorem lily_disproves_tom_claim (c : Card) (c' : Card) : c = Card.R → ¬ has_triangle c c' → ¬ tom_claim c c' :=
by
  intros
  sorry

end lily_disproves_tom_claim_l121_121291


namespace circle_symmetric_line_a_value_l121_121365

theorem circle_symmetric_line_a_value :
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0) →
  (∀ x y : ℝ, (x, y) = (-1, 2)) →
  (∀ x y : ℝ, ax + y + 1 = 0) →
  a = 3 :=
by
  sorry

end circle_symmetric_line_a_value_l121_121365


namespace ab_plus_cd_111_333_l121_121875

theorem ab_plus_cd_111_333 (a b c d : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a + b + d = 5) 
  (h3 : a + c + d = 20) 
  (h4 : b + c + d = 15) : 
  a * b + c * d = 111.333 := 
by
  sorry

end ab_plus_cd_111_333_l121_121875


namespace candy_bar_cost_l121_121719

def num_quarters := 4
def num_dimes := 3
def num_nickel := 1
def change_received := 4

def value_quarter := 25
def value_dime := 10
def value_nickel := 5

def total_paid := (num_quarters * value_quarter) + (num_dimes * value_dime) + (num_nickel * value_nickel)
def cost_candy_bar := total_paid - change_received

theorem candy_bar_cost : cost_candy_bar = 131 := by
  sorry

end candy_bar_cost_l121_121719


namespace gcd_9125_4277_l121_121082

theorem gcd_9125_4277 : Nat.gcd 9125 4277 = 1 :=
by
  -- proof by Euclidean algorithm steps
  sorry

end gcd_9125_4277_l121_121082


namespace games_played_by_player_3_l121_121074

theorem games_played_by_player_3 (games_1 games_2 : ℕ) (rotation_system : ℕ) :
  games_1 = 10 → games_2 = 21 →
  rotation_system = (games_2 - games_1) →
  rotation_system = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end games_played_by_player_3_l121_121074


namespace jason_pokemon_cards_l121_121716

-- Conditions
def initial_cards : ℕ := 13
def cards_given : ℕ := 9

-- Proof Statement
theorem jason_pokemon_cards (initial_cards cards_given : ℕ) : initial_cards - cards_given = 4 :=
by
  sorry

end jason_pokemon_cards_l121_121716


namespace sequence_periodicity_a5_a2019_l121_121802

theorem sequence_periodicity_a5_a2019 (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, 0 < n → a n * a (n + 2) = 3 * a (n + 1)) :
  a 5 * a 2019 = 27 :=
sorry

end sequence_periodicity_a5_a2019_l121_121802


namespace avg_ABC_l121_121509

variables (A B C : Set ℕ) -- Sets of people
variables (a b c : ℕ) -- Numbers of people in sets A, B, and C respectively
variables (sum_A sum_B sum_C : ℕ) -- Sums of the ages of people in sets A, B, and C respectively

-- Given conditions
axiom avg_A : sum_A / a = 30
axiom avg_B : sum_B / b = 20
axiom avg_C : sum_C / c = 45

axiom avg_AB : (sum_A + sum_B) / (a + b) = 25
axiom avg_AC : (sum_A + sum_C) / (a + c) = 40
axiom avg_BC : (sum_B + sum_C) / (b + c) = 32

theorem avg_ABC : (sum_A + sum_B + sum_C) / (a + b + c) = 35 :=
by
  sorry

end avg_ABC_l121_121509


namespace ice_creams_not_sold_l121_121605

theorem ice_creams_not_sold 
  (chocolate_ice_creams : ℕ)
  (mango_ice_creams : ℕ)
  (sold_chocolate : ℚ)
  (sold_mango : ℚ)
  (initial_chocolate : chocolate_ice_creams = 50)
  (initial_mango : mango_ice_creams = 54)
  (fraction_sold_chocolate : sold_chocolate = 3 / 5)
  (fraction_sold_mango : sold_mango = 2 / 3) :
  chocolate_ice_creams - (chocolate_ice_creams * fraction_sold_chocolate).toNat
  + mango_ice_creams - (mango_ice_creams * fraction_sold_mango).toNat = 38 := 
by {
  sorry
}

end ice_creams_not_sold_l121_121605


namespace overall_percent_change_l121_121589

theorem overall_percent_change (W : ℝ) : 
  (W * 0.6 * 1.3 * 0.8 * 1.1) / W = 0.624 :=
by {
  sorry
}

end overall_percent_change_l121_121589


namespace cos_double_angle_l121_121656

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 := sorry

end cos_double_angle_l121_121656


namespace max_profit_at_150_l121_121806

-- Define the conditions
def purchase_price : ℕ := 80
def total_items : ℕ := 1000
def selling_price_initial : ℕ := 100
def sales_volume_decrease : ℕ := 5

-- The profit function
def profit (x : ℕ) : ℤ :=
  (selling_price_initial + x) * (total_items - sales_volume_decrease * x) - purchase_price * total_items

-- The statement to prove: the selling price of 150 yuan/item maximizes the profit at 32500 yuan.
theorem max_profit_at_150 : profit 50 = 32500 := by
  sorry

end max_profit_at_150_l121_121806


namespace number_of_boys_in_school_l121_121069

theorem number_of_boys_in_school (x g : ℕ) (h1 : x + g = 400) (h2 : g = (x * 400) / 100) : x = 80 :=
by
  sorry

end number_of_boys_in_school_l121_121069


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l121_121387

theorem number_of_subsets_with_at_least_four_adjacent_chairs : 
  ∀ (n : ℕ), n = 12 → 
  (∃ (s : Finset (Finset (Fin n))), s.card = 1610 ∧ 
  (∀ (A : Finset (Fin n)), A ∈ s → (∃ (start : Fin n), ∀ i, i ∈ Finset.range 4 → A.contains (start + i % n)))) :=
by
  sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l121_121387


namespace artificial_scarcity_strategy_interview_strategy_l121_121064

-- Conditions for Part (a)
variable (high_end_goods : Type)      -- Type representing high-end goods
variable (manufacturers : high_end_goods → Prop)  -- Property of being a manufacturer
variable (sufficient_resources : high_end_goods → Prop) -- Sufficient resources to produce more
variable (demand : high_end_goods → ℤ)   -- Demand for the product
variable (supply : high_end_goods → ℤ)   -- Supply of the product 
variable (price : ℤ)   -- Price of the product

-- Theorem for Part (a)
theorem artificial_scarcity_strategy (H1 : ∀ g : high_end_goods, manufacturers g → sufficient_resources g → demand g = 3000 ∧ supply g = 200 ∧ price = 15000) :
  ∀ g : high_end_goods, manufacturers g → maintain_exclusivity g :=
sorry

-- Conditions for Part (b)
variable (interview_required : high_end_goods → Prop)   -- Interview requirement for purchase
variable (purchase_history : Prop)  -- Previous purchase history

-- Advantages and Disadvantages from Part (b)
def selective_clientele : Prop := sorry   -- Definition of selective clientele
def enhanced_exclusivity : Prop := sorry   -- Definition of enhanced exclusivity
def increased_transaction_costs : Prop := sorry   -- Definition of increased transaction costs

-- Theorem for Part (b)
theorem interview_strategy (H2 : ∀ g : high_end_goods, manufacturers g → interview_required g → purchase_history) :
  (selective_clientele ∧ enhanced_exclusivity) ∧ increased_transaction_costs :=
sorry

end artificial_scarcity_strategy_interview_strategy_l121_121064


namespace p1a_p1b_l121_121958

theorem p1a (m : ℕ) (hm : m > 1) : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 - y^2 = m^3 := by
  sorry  -- Proof is omitted

theorem p1b : ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^6 = y^2 + 127 ∧ x = 4 ∧ y = 63 := by
  sorry  -- Proof is omitted

end p1a_p1b_l121_121958


namespace equivalence_of_conditions_l121_121196

-- Conditions definitions
variables {Ω : Type*} {ξ ξₙ : Ω → ℝ}
variable {μ : MeasureTheory.Measure Ω}
variables [MeasureTheory.ProbabilityMeasure μ]
variable (hn : (λ n, MeasureTheory.CondDistrib ξ ξₙ μ.toOuterMeasure μ).ToSeq)
variable (finite_integral : ∀ n, MeasureTheory.Integrable ξₙ μ)

-- Define expectations and convergence
variable h₁ : ∀ n, 0 ≤ ξₙ n
variable h_limit : MeasureTheory.WeakConvergenceInDistribution ξₙ ξ

-- Goal: To prove the equivalence of the conditions
theorem equivalence_of_conditions :
  (tendsto (λ n, MeasureTheory.integral μ (ξₙ n)) at_top (𝓝 (MeasureTheory.integral μ ξ)) ∧ (MeasureTheory.integral μ ξ < ∞)) ↔
  (limsup (λ n, MeasureTheory.integral μ (ξₙ n)) at_top ≤ MeasureTheory.integral μ ξ ∧ (MeasureTheory.integral μ ξ < ∞)) ↔
  (measure_theory.uniform_integrable μ (λ n, ξₙ n) at_top) :=
sorry

end equivalence_of_conditions_l121_121196


namespace correct_operation_l121_121087

variable (a b : ℝ)

theorem correct_operation :
  ¬ (a^2 + a^3 = a^5) ∧
  ¬ ((a^2)^3 = a^5) ∧
  ¬ (a^2 * a^3 = a^6) ∧
  ((-a * b)^5 / (-a * b)^3 = a^2 * b^2) :=
by
  sorry

end correct_operation_l121_121087


namespace total_distance_walked_l121_121089

-- Define the conditions
def home_to_school : ℕ := 750
def half_distance : ℕ := home_to_school / 2
def return_home : ℕ := half_distance
def home_to_school_again : ℕ := home_to_school

-- Define the theorem statement
theorem total_distance_walked : 
  half_distance + return_home + home_to_school_again = 1500 := by
  sorry

end total_distance_walked_l121_121089


namespace find_x_squared_plus_y_squared_l121_121649

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = 10344 / 169 :=
by
  sorry

end find_x_squared_plus_y_squared_l121_121649


namespace not_factorable_l121_121899

-- Define the quartic polynomial P(x)
def P (x : ℤ) : ℤ := x^4 + 2 * x^2 + 2 * x + 2

-- Define the quadratic polynomials with integer coefficients
def Q₁ (a b x : ℤ) : ℤ := x^2 + a * x + b
def Q₂ (c d x : ℤ) : ℤ := x^2 + c * x + d

-- Define the condition for factorization, and the theorem to be proven
theorem not_factorable :
  ¬ ∃ (a b c d : ℤ), ∀ x : ℤ, P x = (Q₁ a b x) * (Q₂ c d x) := by
  sorry

end not_factorable_l121_121899


namespace find_number_l121_121160

theorem find_number :
  ∃ x : ℚ, x * (-1/2) = 1 ↔ x = -2 := 
sorry

end find_number_l121_121160


namespace problem_statement_l121_121554

noncomputable def correlation_coefficient (data : List (ℝ × ℝ)) : ℝ :=
let xs := data.map Prod.fst
let ys := data.map Prod.snd
let mean_x := (List.sum xs) / (xs.length : ℝ)
let mean_y := (List.sum ys) / (ys.length : ℝ)
let covariance := List.sum (data.map (λ (p : ℝ × ℝ), (p.1 - mean_x) * (p.2 - mean_y)))
let variance_x := List.sum (xs.map (λ x, (x - mean_x)^2))
let variance_y := List.sum (ys.map (λ y, (y - mean_y)^2))
covariance / (Real.sqrt variance_x * Real.sqrt variance_y)

noncomputable def regression_slope (data : List (ℝ × ℝ)) : ℝ :=
let xs := data.map Prod.fst
let ys := data.map Prod.snd
let mean_x := (List.sum xs) / (xs.length : ℝ)
let mean_y := (List.sum ys) / (ys.length : ℝ)
let numerator := List.sum (data.map (λ (p : ℝ × ℝ), (p.1 - mean_x) * (p.2 - mean_y)))
let denominator := List.sum (xs.map (λ x, (x - mean_x)^2))
numerator / denominator

noncomputable def regression_intercept (data : List (ℝ × ℝ)) : ℝ :=
let mean_x := (List.sum (data.map Prod.fst)) / (data.length : ℝ)
let mean_y := (List.sum (data.map Prod.snd)) / (data.length : ℝ)
mean_y - regression_slope data * mean_x

def predict_y (data : List (ℝ × ℝ)) (x : ℝ) : ℝ :=
regression_slope data * x + regression_intercept data

theorem problem_statement :
  let data := [(2, 300), (4, 400), (5, 400), (6, 400), (8, 500)] in
  correlation_coefficient data ≈ 0.95 ∧ predict_y data 15 = 700 :=
by { sorry }

end problem_statement_l121_121554


namespace expected_value_of_winnings_l121_121797

noncomputable def winnings (n : ℕ) : ℕ := 2 * n - 1

theorem expected_value_of_winnings : 
  (1 / 6 : ℚ) * ((winnings 1) + (winnings 2) + (winnings 3) + (winnings 4) + (winnings 5) + (winnings 6)) = 6 :=
by
  sorry

end expected_value_of_winnings_l121_121797


namespace intersection_of_sets_l121_121309

def setA : Set ℝ := {x | (x^2 - x - 2 < 0)}
def setB : Set ℝ := {y | ∃ x ≤ 0, y = 3^x}

theorem intersection_of_sets : (setA ∩ setB) = {z | 0 < z ∧ z ≤ 1} :=
sorry

end intersection_of_sets_l121_121309


namespace binomial_coefficients_sum_l121_121332

noncomputable def f (m n : ℕ) : ℕ :=
  (Nat.choose 6 m) * (Nat.choose 4 n)

theorem binomial_coefficients_sum : 
  f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 := 
by
  sorry

end binomial_coefficients_sum_l121_121332


namespace scientific_notation_of_216000_l121_121746

theorem scientific_notation_of_216000 :
  216000 = 2.16 * 10^5 :=
sorry

end scientific_notation_of_216000_l121_121746


namespace subsets_with_at_least_four_adjacent_chairs_l121_121379

/-- The number of subsets of a set of 12 chairs arranged in a circle that contain at least four adjacent chairs is 1776. -/
theorem subsets_with_at_least_four_adjacent_chairs (S : Finset (Fin 12)) :
  let n := 12 in
  ∃ F : Finset (Finset (Fin 12)), (∀ s ∈ F, ∃ l : List (Fin 12), (l.length ≥ 4 ∧ l.nodup ∧ ∀ i, i ∈ l → (i + 1) % n ∈ l)) ∧ F.card = 1776 := 
sorry

end subsets_with_at_least_four_adjacent_chairs_l121_121379


namespace converted_land_eqn_l121_121264

theorem converted_land_eqn (forest_land dry_land converted_dry_land : ℝ)
  (h1 : forest_land = 108)
  (h2 : dry_land = 54)
  (h3 : converted_dry_land = x) :
  (dry_land - converted_dry_land = 0.2 * (forest_land + converted_dry_land)) :=
by
  simp [h1, h2, h3]
  sorry

end converted_land_eqn_l121_121264


namespace number_of_integer_solutions_l121_121829

theorem number_of_integer_solutions :
  ∃ (n : ℕ), 
  (∀ (x y : ℤ), 2 * x + 3 * y = 7 ∧ 5 * x + n * y = n ^ 2) ∧
  (n = 8) := 
sorry

end number_of_integer_solutions_l121_121829


namespace john_free_throws_l121_121037

theorem john_free_throws 
  (hit_rate : ℝ) 
  (shots_per_foul : ℕ) 
  (fouls_per_game : ℕ) 
  (total_games : ℕ) 
  (percentage_played : ℝ) 
  : hit_rate = 0.7 → 
    shots_per_foul = 2 → 
    fouls_per_game = 5 → 
    total_games = 20 → 
    percentage_played = 0.8 → 
    ∃ (total_free_throws : ℕ), total_free_throws = 112 := 
by
  intros
  sorry

end john_free_throws_l121_121037


namespace possible_values_of_m_l121_121032

theorem possible_values_of_m
  (m : ℕ)
  (h1 : ∃ (m' : ℕ), m = m' ∧ 0 < m)            -- m is a positive integer
  (h2 : 2 * (m - 1) + 3 * (m + 2) > 4 * (m - 5))    -- AB + AC > BC
  (h3 : 2 * (m - 1) + 4 * (m + 5) > 3 * (m + 2))    -- AB + BC > AC
  (h4 : 3 * (m + 2) + 4 * (m + 5) > 2 * (m - 1))    -- AC + BC > AB
  (h5 : 3 * (m + 2) > 2 * (m - 1))                  -- AC > AB
  (h6 : 4 * (m + 5) > 3 * (m + 2))                  -- BC > AC
  : m ≥ 7 := 
sorry

end possible_values_of_m_l121_121032


namespace rectangle_area_diagonal_l121_121758

theorem rectangle_area_diagonal (r l w d : ℝ) (h_ratio : r = 5 / 2) (h_diag : d^2 = l^2 + w^2) : ∃ k : ℝ, (k = 10 / 29) ∧ (l / w = r) ∧ (l^2 + w^2 = d^2) :=
by
  sorry

end rectangle_area_diagonal_l121_121758


namespace smallest_prime_after_five_consecutive_nonprimes_l121_121242

theorem smallest_prime_after_five_consecutive_nonprimes :
  ∃ p : ℕ, Nat.Prime p ∧ 
          (∀ n : ℕ, n < p → ¬ (n ≥ 24 ∧ n < 29 ∧ ¬ Nat.Prime n)) ∧
          p = 29 :=
by
  sorry

end smallest_prime_after_five_consecutive_nonprimes_l121_121242


namespace janet_percentage_l121_121183

-- Define the number of snowballs made by Janet and her brother
def janet_snowballs : ℕ := 50
def brother_snowballs : ℕ := 150

-- Define the total number of snowballs
def total_snowballs : ℕ := janet_snowballs + brother_snowballs

-- Define the percentage function
def percentage (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- State the theorem to be proved
theorem janet_percentage : percentage janet_snowballs total_snowballs = 25 := by
  sorry

end janet_percentage_l121_121183


namespace switches_connections_l121_121547

theorem switches_connections (n k : ℕ) (h1 : n = 30) (h2 : k = 4) : (n * k) / 2 = 60 :=
by
  sorry

end switches_connections_l121_121547


namespace average_marks_l121_121524

theorem average_marks (D I T : ℕ) 
  (hD : D = 90)
  (hI : I = (3 * D) / 5)
  (hT : T = 2 * I) : 
  (D + I + T) / 3 = 84 :=
by
  sorry

end average_marks_l121_121524


namespace notebook_cost_l121_121738

theorem notebook_cost
  (initial_amount : ℝ)
  (notebook_count : ℕ)
  (pen_count : ℕ)
  (pen_cost : ℝ)
  (remaining_amount : ℝ)
  (total_spent : ℝ)
  (notebook_cost : ℝ) :
  initial_amount = 15 →
  notebook_count = 2 →
  pen_count = 2 →
  pen_cost = 1.5 →
  remaining_amount = 4 →
  total_spent = initial_amount - remaining_amount →
  total_spent = notebook_count * notebook_cost + pen_count * pen_cost →
  notebook_cost = 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end notebook_cost_l121_121738


namespace hypotenuse_length_l121_121578

theorem hypotenuse_length (a b c : ℝ) (h1 : a + b + c = 32) (h2 : a * b = 40) (h3 : a^2 + b^2 = c^2) : 
  c = 59 / 4 :=
by
  sorry

end hypotenuse_length_l121_121578


namespace desired_interest_rate_l121_121255

def face_value : Real := 52
def dividend_rate : Real := 0.09
def market_value : Real := 39

theorem desired_interest_rate : (dividend_rate * face_value / market_value) * 100 = 12 := by
  sorry

end desired_interest_rate_l121_121255


namespace tan_sum_pi_over_4_sin_cos_fraction_l121_121998

open Real

variable (α : ℝ)

axiom tan_α_eq_2 : tan α = 2

theorem tan_sum_pi_over_4 (α : ℝ) (h : tan α = 2) : tan (α + π / 4) = -3 :=
sorry

theorem sin_cos_fraction (α : ℝ) (h : tan α = 2) : (sin α + cos α) / (sin α - cos α) = 3 :=
sorry

end tan_sum_pi_over_4_sin_cos_fraction_l121_121998


namespace trioball_play_time_l121_121223

theorem trioball_play_time (total_duration : ℕ) (num_children : ℕ) (players_at_a_time : ℕ) 
  (equal_play_time : ℕ) (H1 : total_duration = 120) (H2 : num_children = 3) (H3 : players_at_a_time = 2)
  (H4 : equal_play_time = 240 / num_children)
  : equal_play_time = 80 := 
by 
  sorry

end trioball_play_time_l121_121223


namespace tileable_if_and_only_if_l121_121128

def is_tileable (n : ℕ) : Prop :=
  ∃ k : ℕ, 15 * n = 4 * k

theorem tileable_if_and_only_if (n : ℕ) :
  (n ≠ 1 ∧ n ≠ 2 ∧ n ≠ 4 ∧ n ≠ 7) ↔ is_tileable n :=
sorry

end tileable_if_and_only_if_l121_121128


namespace vasya_new_scoring_system_l121_121574

theorem vasya_new_scoring_system (a b c : ℕ) 
  (h1 : a + b + c = 52) 
  (h2 : a + b / 2 = 35) : a - c = 18 :=
by
  sorry

end vasya_new_scoring_system_l121_121574


namespace remainder_when_divided_by_15_l121_121785

theorem remainder_when_divided_by_15 (N : ℕ) (h1 : N % 60 = 49) : N % 15 = 4 :=
by
  sorry

end remainder_when_divided_by_15_l121_121785


namespace expression_eq_l121_121872

theorem expression_eq (x : ℝ) : 
    (x + 1)^4 + 4 * (x + 1)^3 + 6 * (x + 1)^2 + 4 * (x + 1) + 1 = (x + 2)^4 := 
  sorry

end expression_eq_l121_121872


namespace find_quotient_l121_121051

theorem find_quotient (dividend divisor remainder quotient : ℕ) 
  (h_dividend : dividend = 171) 
  (h_divisor : divisor = 21) 
  (h_remainder : remainder = 3) 
  (h_div_eq : dividend = divisor * quotient + remainder) :
  quotient = 8 :=
by sorry

end find_quotient_l121_121051


namespace solve_x_squared_eq_four_x_l121_121764

theorem solve_x_squared_eq_four_x : {x : ℝ | x^2 = 4*x} = {0, 4} := 
sorry

end solve_x_squared_eq_four_x_l121_121764


namespace new_average_doubled_marks_l121_121909

theorem new_average_doubled_marks (n : ℕ) (avg : ℕ) (h_n : n = 11) (h_avg : avg = 36) :
  (2 * avg * n) / n = 72 :=
by
  sorry

end new_average_doubled_marks_l121_121909


namespace cos_double_angle_l121_121651

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (2 * θ) = -7 / 25 := 
by
  sorry

end cos_double_angle_l121_121651


namespace derivative_at_zero_l121_121752

-- Given conditions
def f (x : ℝ) : ℝ := -2 * x^2 + 3

-- Theorem statement to prove
theorem derivative_at_zero : 
  deriv f 0 = 0 := 
by 
  sorry

end derivative_at_zero_l121_121752


namespace no_preimage_for_p_gt_1_l121_121917

def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem no_preimage_for_p_gt_1 (P : ℝ) (hP : P > 1) : ¬ ∃ x : ℝ, f x = P :=
sorry

end no_preimage_for_p_gt_1_l121_121917


namespace B_work_rate_l121_121251

theorem B_work_rate :
  let A := (1 : ℝ) / 8
  let C := (1 : ℝ) / 4.8
  (A + B + C = 1 / 2) → (B = 1 / 6) :=
by
  intro h
  let A : ℝ := 1 / 8
  let C : ℝ := 1 / 4.8
  let B : ℝ := 1 / 6
  sorry

end B_work_rate_l121_121251


namespace angle_in_third_quadrant_l121_121012

-- Definitions for quadrants
def in_fourth_quadrant (α : ℝ) : Prop := 270 < α ∧ α < 360
def in_third_quadrant (β : ℝ) : Prop := 180 < β ∧ β < 270

theorem angle_in_third_quadrant (α : ℝ) (h : in_fourth_quadrant α) : in_third_quadrant (180 - α) :=
by
  -- Proof goes here
  sorry

end angle_in_third_quadrant_l121_121012


namespace jude_age_today_l121_121851
-- Import the necessary libraries

-- Define the conditions as hypotheses and then state the required proof
theorem jude_age_today (heath_age_today : ℕ) (heath_age_in_5_years : ℕ) (jude_age_in_5_years : ℕ) 
  (H1 : heath_age_today = 16)
  (H2 : heath_age_in_5_years = heath_age_today + 5)
  (H3 : heath_age_in_5_years = 3 * jude_age_in_5_years) :
  jude_age_in_5_years - 5 = 2 :=
by
  -- Given conditions imply Jude's age today is 2. Proof is omitted.
  sorry

end jude_age_today_l121_121851


namespace odd_function_f_l121_121300

noncomputable def f : ℝ → ℝ
| x => if hx : x ≥ 0 then x * (1 - x) else x * (1 + x)

theorem odd_function_f {f : ℝ → ℝ}
  (h_odd : ∀ x : ℝ, f (-x) = - f x)
  (h_pos : ∀ x : ℝ, 0 ≤ x → f x = x * (1 - x)) :
  ∀ x : ℝ, x ≤ 0 → f x = x * (1 + x) := by
  intro x hx
  sorry

end odd_function_f_l121_121300


namespace average_expenditure_whole_week_l121_121790

theorem average_expenditure_whole_week (a b : ℕ) (h₁ : a = 3 * 350) (h₂ : b = 4 * 420) : 
  (a + b) / 7 = 390 :=
by 
  sorry

end average_expenditure_whole_week_l121_121790


namespace circle_properties_l121_121187

theorem circle_properties :
  ∃ p q s : ℝ, 
  (∀ x y : ℝ, x^2 + 16 * y + 89 = -y^2 - 12 * x ↔ (x + p)^2 + (y + q)^2 = s^2) ∧ 
  p + q + s = -14 + Real.sqrt 11 :=
by
  use -6, -8, Real.sqrt 11
  sorry

end circle_properties_l121_121187


namespace multiplication_verification_l121_121090

theorem multiplication_verification (x : ℕ) (h : 23 - x = 4) : 23 * x = 437 := by
  sorry

end multiplication_verification_l121_121090


namespace magic_king_total_episodes_l121_121926

theorem magic_king_total_episodes :
  (∑ i in finset.range 5, 20) + (∑ j in finset.range 5, 25) = 225 :=
by sorry

end magic_king_total_episodes_l121_121926


namespace base_seven_representation_l121_121995

theorem base_seven_representation 
  (k : ℕ) 
  (h1 : 4 ≤ k) 
  (h2 : k < 8) 
  (h3 : 500 / k^3 < k) 
  (h4 : 500 ≥ k^3) 
  : ∃ n m o p : ℕ, (500 = n * k^3 + m * k^2 + o * k + p) ∧ (p % 2 = 1) ∧ (n ≠ 0 ) :=
sorry

end base_seven_representation_l121_121995


namespace cos_double_angle_l121_121695

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3/5) : cos (2*θ) = -7/25 := 
by 
  sorry

end cos_double_angle_l121_121695


namespace double_angle_cosine_l121_121675

theorem double_angle_cosine (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end double_angle_cosine_l121_121675


namespace eval_expr_at_values_l121_121203

variable (x y : ℝ)

def expr := 2 * (3 * x^2 + x * y^2)- 3 * (2 * x * y^2 - x^2) - 10 * x^2

theorem eval_expr_at_values : x = -1 → y = 0.5 → expr x y = 0 :=
by
  intros hx hy
  rw [hx, hy]
  sorry

end eval_expr_at_values_l121_121203


namespace solve_for_x_l121_121362

variable {x y : ℝ}

theorem solve_for_x (h1 : y = (x^2 - 9) / (x - 3)) (h2 : y = 3 * x) : x = 3 / 2 := by
  sorry

end solve_for_x_l121_121362


namespace find_S11_l121_121839

variable (n : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d a1, ∀ n, a n = a1 + (n - 1) * d

axiom sum_of_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : ∀ n, S n = n * (a 1 + a n) / 2
axiom condition1 : is_arithmetic_sequence a
axiom condition2 : a 5 + a 7 = (a 6)^2

-- Proof (statement) that the sum of the first 11 terms is 22
theorem find_S11 : S 11 = 22 :=
  sorry

end find_S11_l121_121839


namespace dvds_left_l121_121982

-- Define the initial conditions
def owned_dvds : Nat := 13
def sold_dvds : Nat := 6

-- Define the goal
theorem dvds_left (owned_dvds : Nat) (sold_dvds : Nat) : owned_dvds - sold_dvds = 7 :=
by
  sorry

end dvds_left_l121_121982


namespace total_boxes_count_l121_121099

theorem total_boxes_count 
    (apples_per_crate : ℕ) (apples_crates : ℕ) 
    (oranges_per_crate : ℕ) (oranges_crates : ℕ) 
    (bananas_per_crate : ℕ) (bananas_crates : ℕ) 
    (rotten_apples_percentage : ℝ) (rotten_oranges_percentage : ℝ) (rotten_bananas_percentage : ℝ)
    (apples_per_box : ℕ) (oranges_per_box : ℕ) (bananas_per_box : ℕ) :
    apples_per_crate = 42 → apples_crates = 12 → 
    oranges_per_crate = 36 → oranges_crates = 15 → 
    bananas_per_crate = 30 → bananas_crates = 18 → 
    rotten_apples_percentage = 0.08 → rotten_oranges_percentage = 0.05 → rotten_bananas_percentage = 0.02 →
    apples_per_box = 10 → oranges_per_box = 12 → bananas_per_box = 15 →
    ∃ total_boxes : ℕ, total_boxes = 126 :=
by sorry

end total_boxes_count_l121_121099


namespace triangle_angle_A_l121_121034

theorem triangle_angle_A (a c : ℝ) (C A : ℝ) 
  (h1 : a = 4 * Real.sqrt 3)
  (h2 : c = 12)
  (h3 : C = Real.pi / 3)
  (h4 : a < c) :
  A = Real.pi / 6 :=
sorry

end triangle_angle_A_l121_121034


namespace Yoongi_has_smaller_number_l121_121721

def Jungkook_number : ℕ := 6 + 3
def Yoongi_number : ℕ := 4

theorem Yoongi_has_smaller_number : Yoongi_number < Jungkook_number :=
by
  exact sorry

end Yoongi_has_smaller_number_l121_121721


namespace moles_of_MgCO3_formed_l121_121289

theorem moles_of_MgCO3_formed 
  (moles_MgO : ℕ) (moles_CO2 : ℕ)
  (h_eq : moles_MgO = 3 ∧ moles_CO2 = 3)
  (balanced_eq : ∀ n : ℕ, n * MgO + n * CO2 = n * MgCO3) : 
  moles_MgCO3 = 3 :=
by
  sorry

end moles_of_MgCO3_formed_l121_121289


namespace point_on_parallel_line_with_P_l121_121603

-- Definitions
def is_on_parallel_line_x_axis (P Q : ℝ × ℝ) : Prop :=
  P.snd = Q.snd

theorem point_on_parallel_line_with_P :
  let P := (3, -2)
  let D := (-3, -2)
  is_on_parallel_line_x_axis P D :=
by
  sorry

end point_on_parallel_line_with_P_l121_121603


namespace total_cost_of_replacing_floor_l121_121077

-- Dimensions of the first rectangular section
def length1 : ℕ := 8
def width1 : ℕ := 7

-- Dimensions of the second rectangular section
def length2 : ℕ := 6
def width2 : ℕ := 4

-- Cost to remove the old flooring
def cost_removal : ℕ := 50

-- Cost of new flooring per square foot
def cost_per_sqft : ℝ := 1.25

-- Total cost to replace the floor in both sections of the L-shaped room
theorem total_cost_of_replacing_floor 
  (A1 : ℕ := length1 * width1)
  (A2 : ℕ := length2 * width2)
  (total_area : ℕ := A1 + A2)
  (cost_flooring : ℝ := total_area * cost_per_sqft)
  : cost_removal + cost_flooring = 150 :=
sorry

end total_cost_of_replacing_floor_l121_121077


namespace probability_one_white_one_black_two_touches_l121_121322

def probability_white_ball : ℚ := 7 / 10
def probability_black_ball : ℚ := 3 / 10

theorem probability_one_white_one_black_two_touches :
  (probability_white_ball * probability_black_ball) + (probability_black_ball * probability_white_ball) = (7 / 10) * (3 / 10) + (3 / 10) * (7 / 10) :=
by
  -- The proof is omitted here.
  sorry

end probability_one_white_one_black_two_touches_l121_121322


namespace total_sum_of_money_l121_121952

theorem total_sum_of_money (x : ℝ) (A B C : ℝ) 
  (hA : A = x) 
  (hB : B = 0.65 * x) 
  (hC : C = 0.40 * x) 
  (hC_share : C = 32) :
  A + B + C = 164 := 
  sorry

end total_sum_of_money_l121_121952


namespace no_hamiltonian_circuit_rhombic_dodecahedron_l121_121421

-- We define the graph of a rhombic dodecahedron.
def rhombic_dodecahedron : SimpleGraph ℕ := sorry

-- We state the theorem: the rhombic dodecahedron has no Hamiltonian circuit.
theorem no_hamiltonian_circuit_rhombic_dodecahedron :
  ¬(∃ p : List ℕ, rhombic_dodecahedron.IsHamiltonianCircuit p) :=
sorry

end no_hamiltonian_circuit_rhombic_dodecahedron_l121_121421


namespace contrapositive_proposition_l121_121922

theorem contrapositive_proposition
  (a b c d : ℝ) 
  (h : a + c ≠ b + d) : a ≠ b ∨ c ≠ d :=
sorry

end contrapositive_proposition_l121_121922


namespace find_judes_age_l121_121854

def jude_age (H : ℕ) (J : ℕ) : Prop :=
  H + 5 = 3 * (J + 5)

theorem find_judes_age : ∃ J : ℕ, jude_age 16 J ∧ J = 2 :=
by
  sorry

end find_judes_age_l121_121854


namespace geometric_sequence_value_l121_121371

theorem geometric_sequence_value (a : ℝ) (h_pos : 0 < a) 
    (h_geom1 : ∃ r, 25 * r = a)
    (h_geom2 : ∃ r, a * r = 7 / 9) : 
    a = 5 * Real.sqrt 7 / 3 :=
by
  sorry

end geometric_sequence_value_l121_121371


namespace ice_cream_eaten_on_friday_l121_121266

theorem ice_cream_eaten_on_friday
  (x : ℝ) -- the amount eaten on Friday night
  (saturday_night : ℝ) -- the amount eaten on Saturday night
  (total : ℝ) -- the total amount eaten
  
  (h1 : saturday_night = 0.25)
  (h2 : total = 3.5)
  (h3 : x + saturday_night = total) : x = 3.25 :=
by
  sorry

end ice_cream_eaten_on_friday_l121_121266


namespace total_travel_time_l121_121045

-- We define the conditions as assumptions in the Lean 4 statement.

def riding_rate := 10  -- miles per hour
def time_first_part_minutes := 30  -- initial 30 minutes in minutes
def additional_distance_1 := 15  -- another 15 miles
def rest_time_minutes := 30  -- resting for 30 minutes
def remaining_distance := 20  -- remaining distance of 20 miles

theorem total_travel_time : 
    let time_first_part := time_first_part_minutes
    let time_second_part := (additional_distance_1 : ℚ) / riding_rate * 60  -- convert hours to minutes
    let time_third_part := rest_time_minutes
    let time_fourth_part := (remaining_distance : ℚ) / riding_rate * 60  -- convert hours to minutes
    time_first_part + time_second_part + time_third_part + time_fourth_part = 270 :=
by
  sorry

end total_travel_time_l121_121045


namespace fraction_value_l121_121290

def x : ℚ := 4 / 7
def y : ℚ := 8 / 11

theorem fraction_value : (7 * x + 11 * y) / (49 * x * y) = 231 / 56 := by
  sorry

end fraction_value_l121_121290


namespace perpendicular_condition_l121_121633

def is_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem perpendicular_condition (a : ℝ) :
  is_perpendicular (a^2) (1/a) ↔ a = -1 :=
sorry

end perpendicular_condition_l121_121633


namespace science_homework_is_50_minutes_l121_121054

-- Define the times for each homework and project in minutes
def total_time : ℕ := 3 * 60  -- 3 hours converted to minutes
def math_homework : ℕ := 45
def english_homework : ℕ := 30
def history_homework : ℕ := 25
def special_project : ℕ := 30

-- Define a function to compute the time for science homework
def science_homework_time 
  (total_time : ℕ) 
  (math_time : ℕ) 
  (english_time : ℕ) 
  (history_time : ℕ) 
  (project_time : ℕ) : ℕ :=
  total_time - (math_time + english_time + history_time + project_time)

-- The theorem to prove the time Porche's science homework takes
theorem science_homework_is_50_minutes : 
  science_homework_time total_time math_homework english_homework history_homework special_project = 50 := 
sorry

end science_homework_is_50_minutes_l121_121054


namespace sum_even_integers_102_to_200_l121_121374

theorem sum_even_integers_102_to_200 :
  let S := (List.range' 102 (200 - 102 + 1)).filter (λ x => x % 2 = 0)
  List.sum S = 7550 := by
{
  sorry
}

end sum_even_integers_102_to_200_l121_121374


namespace badge_exchange_proof_l121_121561

-- Definitions based on the conditions
def initial_badges_Tolya : ℝ := 45
def initial_badges_Vasya : ℝ := 50

def tollya_exchange_badges (badges_Tolya : ℝ) : ℝ := 0.2 * badges_Tolya
def tollya_receive_badges (badges_Vasya : ℝ) : ℝ := 0.24 * badges_Vasya
def vasya_exchange_badges (badges_Vasya: ℝ) : ℝ := 0.24 * badges_Vasya
def vasya_receive_badges (badges_Tolya: ℝ) : ℝ := 0.2 * badges_Tolya

-- Vasya ended up with one badge less than Tolya after the exchange
theorem badge_exchange_proof 
  (tolya_initial badges_Tolya_initial : ℝ)
  (badges_Vasya_initial: ℝ)
  (tollya_initial_has_24: tollya_receive_badges badges_Vasya_initial)
  (vasya_initial_has_20: vasya_receive_badges badges_Tolya_initial):
  (tollya_initial = initial_badges_Tolya) ∧ (vasya_initial = initial_badges_Vasya) :=
sorry

end badge_exchange_proof_l121_121561


namespace constant_in_price_equation_l121_121216

theorem constant_in_price_equation (x y: ℕ) (h: y = 70 * x) : ∃ c, ∀ (x: ℕ), y = c * x ∧ c = 70 :=
  sorry

end constant_in_price_equation_l121_121216


namespace VivianMailApril_l121_121565

variable (piecesMailApril piecesMailMay piecesMailJune piecesMailJuly piecesMailAugust : ℕ)

-- Conditions
def condition_double_monthly (a b : ℕ) : Prop := b = 2 * a

axiom May : piecesMailMay = 10
axiom June : piecesMailJune = 20
axiom July : piecesMailJuly = 40
axiom August : piecesMailAugust = 80

axiom patternMay : condition_double_monthly piecesMailApril piecesMailMay
axiom patternJune : condition_double_monthly piecesMailMay piecesMailJune
axiom patternJuly : condition_double_monthly piecesMailJune piecesMailJuly
axiom patternAugust : condition_double_monthly piecesMailJuly piecesMailAugust

-- Statement to prove
theorem VivianMailApril :
  piecesMailApril = 5 :=
by
  sorry

end VivianMailApril_l121_121565


namespace sheets_in_set_l121_121718

-- Definitions of the conditions
def John_sheets_left (S E : ℕ) : Prop := S - E = 80
def Mary_sheets_used (S E : ℕ) : Prop := S = 4 * E

-- Theorems to prove the number of sheets
theorem sheets_in_set (S E : ℕ) (hJohn : John_sheets_left S E) (hMary : Mary_sheets_used S E) : S = 320 :=
by { 
  sorry 
}

end sheets_in_set_l121_121718


namespace flour_for_recipe_l121_121868

theorem flour_for_recipe (flour_needed shortening_have : ℚ)
  (flour_ratio shortening_ratio : ℚ) 
  (ratio : flour_ratio / shortening_ratio = 5)
  (shortening_used : shortening_ratio = 2 / 3) :
  flour_needed = 10 / 3 := 
by 
  sorry

end flour_for_recipe_l121_121868


namespace base_conversion_subtraction_l121_121284

/-- Definition of base conversion from base 7 and base 5 to base 10. -/
def convert_base_7_to_10 (n : Nat) : Nat :=
  match n with
  | 52103 => 5 * 7^4 + 2 * 7^3 + 1 * 7^2 + 0 * 7^1 + 3 * 7^0
  | _ => 0

def convert_base_5_to_10 (n : Nat) : Nat :=
  match n with
  | 43120 => 4 * 5^4 + 3 * 5^3 + 1 * 5^2 + 2 * 5^1 + 0 * 5^0
  | _ => 0

theorem base_conversion_subtraction : 
  convert_base_7_to_10 52103 - convert_base_5_to_10 43120 = 9833 :=
by
  -- The proof goes here
  sorry

end base_conversion_subtraction_l121_121284


namespace last_score_is_80_l121_121357

-- Define the list of scores
def scores : List ℕ := [71, 76, 80, 82, 91]

-- Define the total sum of the scores
def total_sum : ℕ := 400

-- Define the condition that the average after each score is an integer
def average_integer_condition (scores : List ℕ) (total_sum : ℕ) : Prop :=
  ∀ (sublist : List ℕ), sublist ≠ [] → sublist ⊆ scores → 
  (sublist.sum / sublist.length : ℕ) * sublist.length = sublist.sum

-- Define the proposition to prove that the last score entered must be 80
theorem last_score_is_80 : ∃ (last_score : ℕ), (last_score = 80) ∧
  average_integer_condition scores total_sum :=
sorry

end last_score_is_80_l121_121357


namespace average_age_of_team_l121_121911

theorem average_age_of_team
    (A : ℝ)
    (captain_age : ℝ)
    (wicket_keeper_age : ℝ)
    (bowlers_count : ℝ)
    (batsmen_count : ℝ)
    (team_members_count : ℝ)
    (avg_bowlers_age : ℝ)
    (avg_batsmen_age : ℝ)
    (total_age_team : ℝ) :
    captain_age = 28 →
    wicket_keeper_age = 31 →
    bowlers_count = 5 →
    batsmen_count = 4 →
    avg_bowlers_age = A - 2 →
    avg_batsmen_age = A + 3 →
    total_age_team = 28 + 31 + 5 * (A - 2) + 4 * (A + 3) →
    team_members_count * A = total_age_team →
    team_members_count = 11 →
    A = 30.5 :=
by
  intros
  sorry

end average_age_of_team_l121_121911


namespace algebraic_expression_value_l121_121494

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 2 * x + 7 = 6) : 4 * x^2 + 8 * x - 5 = -9 :=
by
  sorry

end algebraic_expression_value_l121_121494


namespace sum_of_a_and_b_l121_121514

noncomputable def a : ℕ :=
sorry

noncomputable def b : ℕ :=
sorry

theorem sum_of_a_and_b :
  (100 ≤ a ∧ a ≤ 999) ∧ (1000 ≤ b ∧ b ≤ 9999) ∧ (10000 * a + b = 7 * a * b) ->
  a + b = 1458 :=
by
  sorry

end sum_of_a_and_b_l121_121514


namespace acute_triangle_cosine_identity_l121_121177

variable {A B C O H B1 C1 : ℝ}

theorem acute_triangle_cosine_identity 
  (h1 : ∠BAC < π / 2)
  (h2 : ∠ABC < π / 2)
  (h3 : ∠ACB < π / 2)
  (hAB_AC : AB > AC)
  (circumcenter_O : circumcenter ∆ABC = O)
  (orthocenter_H : orthocenter ∆ABC = H)
  (BH_int_AC : line BH ∩ line AC = {B1})
  (CH_int_AB : line CH ∩ line AB = {C1})
  (OH_parallel_B1C1 : parallel OH B1C1) :
  cos(2 * ∠ABC) + cos(2 * ∠ACB) + 1 = 0 := 
sorry

end acute_triangle_cosine_identity_l121_121177


namespace expand_product_l121_121616

theorem expand_product (x : ℝ) : (x + 5) * (x - 16) = x^2 - 11 * x - 80 :=
by sorry

end expand_product_l121_121616


namespace tan_subtraction_l121_121165

theorem tan_subtraction (α β : ℝ) (h1 : Real.tan α = 5) (h2 : Real.tan β = 3) : 
  Real.tan (α - β) = 1 / 8 := 
by 
  sorry

end tan_subtraction_l121_121165


namespace unique_positive_integer_b_quadratic_solution_l121_121131

theorem unique_positive_integer_b_quadratic_solution (c : ℝ) :
  (∃! (b : ℕ), ∀ (x : ℝ), x^2 + (b^2 + (1 / b^2)) * x + c = 3) ↔ c = 5 :=
sorry

end unique_positive_integer_b_quadratic_solution_l121_121131


namespace cos_difference_l121_121144

theorem cos_difference (α β : ℝ) (h_α_acute : 0 < α ∧ α < π / 2)
                      (h_β_acute : 0 < β ∧ β < π / 2)
                      (h_cos_α : Real.cos α = 1 / 3)
                      (h_cos_sum : Real.cos (α + β) = -1 / 3) :
  Real.cos (α - β) = 23 / 27 := 
sorry

end cos_difference_l121_121144


namespace expression_value_l121_121237

theorem expression_value (a b : ℕ) (ha : a = 45) (hb : b = 15) : (a + b)^2 - (a^2 + b^2) = 1350 := by 
  sorry

end expression_value_l121_121237


namespace division_result_l121_121413

theorem division_result : (108 * 3 - (108 + 92)) / (92 * 7 - (45 * 3)) = 124 / 509 := 
by
  sorry

end division_result_l121_121413


namespace songs_downloaded_later_l121_121346

-- Definition that each song has a size of 5 MB
def song_size : ℕ := 5

-- Definition that the new songs will occupy 140 MB of memory space
def total_new_song_memory : ℕ := 140

-- Prove that the number of songs Kira downloaded later on that day is 28
theorem songs_downloaded_later (x : ℕ) (h : song_size * x = total_new_song_memory) : x = 28 :=
by
  sorry

end songs_downloaded_later_l121_121346


namespace probability_calculation_l121_121435

noncomputable def probability_at_least_seven_at_least_three_times : ℚ :=
  let p := 1 / 4
  let q := 3 / 4
  (4 * p^3 * q) + (p^4)

theorem probability_calculation :
  probability_at_least_seven_at_least_three_times = 13 / 256 :=
by sorry

end probability_calculation_l121_121435


namespace bottle_caps_sum_l121_121518

theorem bottle_caps_sum : 
  let starting_caps := 91
  let found_caps := 88
  starting_caps + found_caps = 179 :=
by
  sorry

end bottle_caps_sum_l121_121518


namespace greatest_p_meets_conditions_l121_121422

-- Define a four-digit number and its reversal being divisible by 63 and another condition of divisibility
def is_divisible_by (n m : ℕ) : Prop :=
  m % n = 0

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ a d => a * 10 + d) 0

def is_four_digit (n : ℕ) : Prop :=
  n >= 1000 ∧ n < 10000

def p := 9507

-- The main theorem we aim to prove.
theorem greatest_p_meets_conditions (p q : ℕ) 
  (h1 : is_four_digit p) 
  (h2 : is_four_digit q) 
  (h3 : reverse_digits p = q) 
  (h4 : is_divisible_by 63 p) 
  (h5 : is_divisible_by 63 q) 
  (h6 : is_divisible_by 9 p) : 
  p = 9507 :=
sorry

end greatest_p_meets_conditions_l121_121422


namespace x_plus_y_l121_121722

variables {e1 e2 : ℝ → ℝ → Prop} -- Represents the vectors as properties of reals
variables {x y : ℝ} -- Real numbers x and y

-- Assuming non-collinearity of e1 and e2 (This means e1 and e2 are independent)
axiom non_collinear : e1 ≠ e2 

-- Given condition translated into Lean
axiom main_equation : (3 * x - 4 * y = 6) ∧ (2 * x - 3 * y = 3)

-- Prove that x + y = 9
theorem x_plus_y : x + y = 9 := 
by
  sorry -- Proof will be provided here

end x_plus_y_l121_121722


namespace jason_pears_count_l121_121864

theorem jason_pears_count 
  (initial_pears : ℕ)
  (given_to_keith : ℕ)
  (received_from_mike : ℕ)
  (final_pears : ℕ)
  (h_initial : initial_pears = 46)
  (h_given : given_to_keith = 47)
  (h_received : received_from_mike = 12)
  (h_final : final_pears = 12) :
  initial_pears - given_to_keith + received_from_mike = final_pears :=
sorry

end jason_pears_count_l121_121864


namespace marshmallow_challenge_l121_121482

noncomputable def haley := 8
noncomputable def michael := 3 * haley
noncomputable def brandon := (1 / 2) * michael
noncomputable def sofia := 2 * (haley + brandon)
noncomputable def total := haley + michael + brandon + sofia

theorem marshmallow_challenge : total = 84 :=
by
  sorry

end marshmallow_challenge_l121_121482


namespace tangent_circle_line_l121_121162

theorem tangent_circle_line (r : ℝ) (h_pos : 0 < r) 
  (h_circle : ∀ x y : ℝ, x^2 + y^2 = r^2) 
  (h_line : ∀ x y : ℝ, x + y = r + 1) : 
  r = 1 + Real.sqrt 2 := 
by 
  sorry

end tangent_circle_line_l121_121162


namespace find_S11_l121_121174

variable {a : ℕ → ℚ} -- Define the arithmetic sequence as a function

-- Define conditions
def arithmetic_sequence (a : ℕ → ℚ) :=
∀ n m, a (n + m) = a n + a m

def S (n : ℕ) (a : ℕ → ℚ) : ℚ := (n / 2 : ℚ) * (a 1 + a n)

-- Define the problem statement to be proved
theorem find_S11 (h_arith : arithmetic_sequence a) (h_eq : a 3 + a 6 + a 9 = 54) : 
  S 11 a = 198 :=
sorry

end find_S11_l121_121174


namespace members_who_play_both_l121_121247

theorem members_who_play_both (N B T Neither : ℕ) (hN : N = 30) (hB : B = 16) (hT : T = 19) (hNeither : Neither = 2) : 
  B + T - (N - Neither) = 7 :=
by
  sorry

end members_who_play_both_l121_121247


namespace meteorological_forecast_l121_121905

theorem meteorological_forecast (prob_rain : ℝ) (h1 : prob_rain = 0.7) :
  (prob_rain = 0.7) → "There is a high probability of needing to carry rain gear when going out tomorrow." = "Correct" :=
by
  intro h
  sorry

end meteorological_forecast_l121_121905


namespace ned_weekly_sales_l121_121731

-- Define the conditions given in the problem
def normal_mouse_price : ℝ := 120
def normal_keyboard_price : ℝ := 80
def normal_scissor_price : ℝ := 30

def lt_hand_mouse_price := normal_mouse_price * 1.3
def lt_hand_keyboard_price := normal_keyboard_price * 1.2
def lt_hand_scissor_price := normal_scissor_price * 1.5

def lt_hand_mouse_daily_sales : ℝ := 25 * lt_hand_mouse_price
def lt_hand_keyboard_daily_sales : ℝ := 10 * lt_hand_keyboard_price
def lt_hand_scissor_daily_sales : ℝ := 15 * lt_hand_scissor_price

def total_daily_sales := lt_hand_mouse_daily_sales + lt_hand_keyboard_daily_sales + lt_hand_scissor_daily_sales
def days_open_per_week : ℝ := 4

def weekly_sales := total_daily_sales * days_open_per_week

-- The theorem to prove
theorem ned_weekly_sales : weekly_sales = 22140 := by
  -- The proof is omitted
  sorry

end ned_weekly_sales_l121_121731


namespace race_course_length_l121_121406

variable (v d : ℝ)

theorem race_course_length (h1 : 4 * v > 0) (h2 : ∀ t : ℝ, t > 0 → (d / (4 * v)) = ((d - 72) / v)) : d = 96 := by
  sorry

end race_course_length_l121_121406


namespace circle_chairs_subsets_count_l121_121396

theorem circle_chairs_subsets_count :
  ∃ (n : ℕ), n = 12 → set.count (λ s : finset ℕ, s.card ≥ 4 ∧ ∀ i ∈ s, (i + 1) % 12 ∈ s) {s | s ⊆ finset.range 12} = 1712 := 
by
  sorry

end circle_chairs_subsets_count_l121_121396


namespace Cauchy_solution_on_X_l121_121228

section CauchyEquation

variable (f : ℝ → ℝ)

def is_morphism (f : ℝ → ℝ) := ∀ x y : ℝ, f (x + y) = f x + f y

theorem Cauchy_solution_on_X :
  (∀ a b : ℤ, ∀ c d : ℤ, a + b * Real.sqrt 2 = c + d * Real.sqrt 2 → a = c ∧ b = d) →
  is_morphism f →
  ∃ x y : ℝ, ∀ a b : ℤ,
    f (a + b * Real.sqrt 2) = a * x + b * y :=
by
  intros h1 h2
  let x := f 1
  let y := f (Real.sqrt 2)
  exists x, y
  intros a b
  sorry

end CauchyEquation

end Cauchy_solution_on_X_l121_121228


namespace parking_lot_cars_l121_121497

theorem parking_lot_cars (total_capacity : ℕ) (num_levels : ℕ) (parked_cars : ℕ) 
  (h1 : total_capacity = 425) (h2 : num_levels = 5) (h3 : parked_cars = 23) : 
  (total_capacity / num_levels) - parked_cars = 62 :=
by
  sorry

end parking_lot_cars_l121_121497


namespace proof_subset_l121_121492

def set_A := {x : ℝ | x ≥ 0}

theorem proof_subset (B : Set ℝ) (h : set_A ∪ B = B) : set_A ⊆ B := 
by
  sorry

end proof_subset_l121_121492


namespace fixed_point_of_parabola_l121_121626

theorem fixed_point_of_parabola :
  ∀ (m : ℝ), ∃ (a b : ℝ), (∀ (x : ℝ), (a = -3 ∧ b = 81) → (y = 9*x^2 + m*x + 3*m) → (y = 81)) :=
by
  sorry

end fixed_point_of_parabola_l121_121626


namespace original_number_of_candies_l121_121432

theorem original_number_of_candies (x : ℝ) (h₀ : x * (0.7 ^ 3) = 40) : x = 117 :=
by 
  sorry

end original_number_of_candies_l121_121432


namespace division_dividend_l121_121495

/-- In a division sum, the quotient is 40, the divisor is 72, and the remainder is 64. We need to prove that the dividend is 2944. -/
theorem division_dividend : 
  let Q := 40
  let D := 72
  let R := 64
  (D * Q + R = 2944) :=
by
  sorry

end division_dividend_l121_121495


namespace calculate_f3_times_l121_121347

def f (n : ℕ) : ℕ :=
  if n ≤ 3 then n^2 + 1 else 4 * n + 2

theorem calculate_f3_times : f (f (f 3)) = 170 := by
  sorry

end calculate_f3_times_l121_121347


namespace contribution_amount_l121_121427

-- Definitions based on conditions
variable (x : ℝ)

-- Total amount needed
def total_needed := 200

-- Contributions from different families
def contribution_two_families := 2 * x
def contribution_eight_families := 8 * 10 -- 80
def contribution_ten_families := 10 * 5 -- 50
def total_contribution := contribution_two_families + contribution_eight_families + contribution_ten_families

-- Amount raised so far given they need 30 more to reach the target
def raised_so_far := total_needed - 30 -- 170

-- Statement to prove
theorem contribution_amount :
  total_contribution x = raised_so_far →
  x = 20 := by 
  sorry

end contribution_amount_l121_121427


namespace number_of_possible_plans_most_cost_effective_plan_l121_121937

-- Defining the conditions of the problem
def price_A := 12 -- Price of model A in million yuan
def price_B := 10 -- Price of model B in million yuan
def capacity_A := 240 -- Treatment capacity of model A in tons/month
def capacity_B := 200 -- Treatment capacity of model B in tons/month
def total_budget := 105 -- Total budget in million yuan
def min_treatment_volume := 2040 -- Minimum required treatment volume in tons/month
def total_units := 10 -- Total number of units to be purchased

def valid_purchase_plan (x y : ℕ) :=
  x + y = total_units ∧
  price_A * x + price_B * y ≤ total_budget ∧
  capacity_A * x + capacity_B * y ≥ min_treatment_volume

-- Stating the theorem for how many possible purchase plans exist
theorem number_of_possible_plans : 
  ∃ k : ℕ, k = 3 ∧
    (∀ (x y : ℕ), 
      valid_purchase_plan x y →
      x ∈ {0, 1, 2} ∧ y = total_units - x) :=
sorry

-- Stating the theorem for the most cost-effective plan
theorem most_cost_effective_plan :
  ∃ (x y : ℕ),
    valid_purchase_plan x y ∧
    price_A * x + price_B * y = 102 ∧
    x = 1 ∧ y = 9 :=
sorry

end number_of_possible_plans_most_cost_effective_plan_l121_121937


namespace ratio_of_andy_age_in_5_years_to_rahim_age_l121_121023

def rahim_age_now : ℕ := 6
def andy_age_now : ℕ := rahim_age_now + 1
def andy_age_in_5_years : ℕ := andy_age_now + 5
def ratio (a b : ℕ) : ℕ := a / b

theorem ratio_of_andy_age_in_5_years_to_rahim_age : ratio andy_age_in_5_years rahim_age_now = 2 := by
  sorry

end ratio_of_andy_age_in_5_years_to_rahim_age_l121_121023


namespace cos_double_angle_l121_121688

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3/5) : Real.cos (2 * theta) = -7/25 :=
by
  sorry

end cos_double_angle_l121_121688


namespace projectile_height_time_l121_121529

-- Define constants and the height function
def a : ℝ := -4.9
def b : ℝ := 29.75
def c : ℝ := -35
def y (t : ℝ) : ℝ := a * t^2 + b * t

-- Problem statement
theorem projectile_height_time (h : y t = 35) : ∃ t : ℝ, 0 < t ∧ abs (t - 1.598) < 0.001 := by
  -- Placeholder for actual proof
  sorry

end projectile_height_time_l121_121529


namespace required_connections_l121_121550

theorem required_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) : (n * k) / 2 = 60 := by
  sorry

end required_connections_l121_121550


namespace diving_classes_on_weekdays_l121_121904

theorem diving_classes_on_weekdays 
  (x : ℕ) 
  (weekend_classes_per_day : ℕ := 4)
  (people_per_class : ℕ := 5)
  (total_people_3_weeks : ℕ := 270)
  (weekend_days : ℕ := 2)
  (total_weeks : ℕ := 3)
  (weekend_total_classes : ℕ := weekend_classes_per_day * weekend_days * total_weeks) 
  (total_people_weekends : ℕ := weekend_total_classes * people_per_class) 
  (total_people_weekdays : ℕ := total_people_3_weeks - total_people_weekends)
  (weekday_classes_needed : ℕ := total_people_weekdays / people_per_class)
  (weekly_weekday_classes : ℕ := weekday_classes_needed / total_weeks)
  (h : weekly_weekday_classes = x)
  : x = 10 := sorry

end diving_classes_on_weekdays_l121_121904


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l121_121382

theorem number_of_subsets_with_at_least_four_adjacent_chairs (chairs : Finset ℕ) (h_chairs : chairs.card = 12)
  (h_circular : ∀ (s : Finset ℕ), s ⊆ chairs → (s.card ≥ 4 → ∃ t : Finset ℕ, t ⊆ chairs ∧ t.card = 4 ∧ ∀ i j ∈ t, abs (i - j) ≤ 1)) :
  ∃ (subsets : Finset (Finset ℕ)), (∀ s ∈ subsets, s ⊆ chairs ∧ s.card ≥ 4) ∧ subsets.card = 169 :=
sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l121_121382


namespace magnitude_of_2a_minus_b_l121_121157

/-- Definition of the vectors a and b --/
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, 3)

/-- Proposition stating the magnitude of 2a - b --/
theorem magnitude_of_2a_minus_b : 
  (Real.sqrt ((2 * a.1 - b.1) ^ 2 + (2 * a.2 - b.2) ^ 2)) = Real.sqrt 10 :=
by
  sorry

end magnitude_of_2a_minus_b_l121_121157


namespace network_connections_l121_121552

theorem network_connections (n : ℕ) (k : ℕ) :
  n = 30 → k = 4 → (n * k) / 2 = 60 := by
  intros h₁ h₂
  rw [h₁, h₂]
  sorry

end network_connections_l121_121552


namespace Frank_days_to_finish_book_l121_121141

theorem Frank_days_to_finish_book (pages_per_day : ℕ) (total_pages : ℕ) (h1 : pages_per_day = 22) (h2 : total_pages = 12518) : total_pages / pages_per_day = 569 := by
  sorry

end Frank_days_to_finish_book_l121_121141


namespace tan_difference_l121_121166

theorem tan_difference (α β : ℝ) (hα : Real.tan α = 5) (hβ : Real.tan β = 3) : 
    Real.tan (α - β) = 1 / 8 := 
by 
  sorry

end tan_difference_l121_121166


namespace tetradecagon_edge_length_correct_l121_121321

-- Define the parameters of the problem
def regular_tetradecagon_perimeter (n : ℕ := 14) : ℕ := 154

-- Define the length of one edge
def edge_length (P : ℕ) (n : ℕ) : ℕ := P / n

-- State the theorem
theorem tetradecagon_edge_length_correct :
  edge_length (regular_tetradecagon_perimeter 14) 14 = 11 := by
  sorry

end tetradecagon_edge_length_correct_l121_121321


namespace cerulean_survey_l121_121959

theorem cerulean_survey :
  let total_people := 120
  let kind_of_blue := 80
  let kind_and_green := 35
  let neither := 20
  total_people = kind_of_blue + (total_people - kind_of_blue - neither)
  → (kind_and_green + (total_people - kind_of_blue - kind_and_green - neither) + neither) = total_people
  → 55 = (kind_and_green + (total_people - kind_of_blue - kind_and_green - neither)) :=
by
  sorry

end cerulean_survey_l121_121959


namespace no_partition_of_six_consecutive_numbers_product_equal_l121_121501

theorem no_partition_of_six_consecutive_numbers_product_equal (n : ℕ) :
  ¬ ∃ (A B : Finset ℕ), 
    A ∪ B = Finset.range (n+6) ∧ 
    A ∩ B = ∅ ∧ 
    A.prod id = B.prod id :=
by
  sorry

end no_partition_of_six_consecutive_numbers_product_equal_l121_121501


namespace tims_drive_distance_l121_121133

theorem tims_drive_distance :
  let t1 := 120
  let t2 := 165
  ∃ y : ℕ, 
    (let speed_usual := y / t1
    let speed_reduced := speed_usual - 1 / 2
    let time_half_usual := (y / 2) / speed_usual
    let time_half_reduced := (y / 2) / speed_reduced
    in time_half_usual + time_half_reduced = t2) ∧ y = 140 := by
  sorry

end tims_drive_distance_l121_121133


namespace nth_permutation_2013_eq_3546127_l121_121472

-- Given the digits 1 through 7, there are 7! = 5040 permutations.
-- We want to prove that the 2013th permutation in ascending order is 3546127.

def digits : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def nth_permutation (n : ℕ) (digits : List ℕ) : List ℕ :=
  sorry

theorem nth_permutation_2013_eq_3546127 :
  nth_permutation 2013 digits = [3, 5, 4, 6, 1, 2, 7] :=
sorry

end nth_permutation_2013_eq_3546127_l121_121472


namespace factor_expression_l121_121285

theorem factor_expression (x : ℝ) :
  x^3 - 6 * x^2 + 9 * x = x * (x - 3)^2 := 
  sorry

end factor_expression_l121_121285


namespace infinite_integers_repr_l121_121056

theorem infinite_integers_repr : ∀ (k : ℕ), k > 1 →
  let a := (2 * k + 1) * k
  let b := (2 * k + 1) * k - 1
  let c := 2 * k + 1
  (a - 1) / b + (b - 1) / c + (c - 1) / a = k + 1 :=
by
  intros k hk
  let a := (2 * k + 1) * k
  let b := (2 * k + 1) * k - 1
  let c := 2 * k + 1
  sorry

end infinite_integers_repr_l121_121056


namespace possible_values_of_f_zero_l121_121209

theorem possible_values_of_f_zero (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = f x * f y) :
  f 0 = 0 ∨ f 0 = 1 :=
by
  sorry

end possible_values_of_f_zero_l121_121209


namespace chairs_adjacent_subsets_l121_121394

theorem chairs_adjacent_subsets (n : ℕ) (h_n : n = 12) :
  (∑ k in (range n.succ).filter (λ k, k ≥ 4), (nat.choose n k)) + 84 = 1704 :=
by sorry

end chairs_adjacent_subsets_l121_121394


namespace tan_double_angle_l121_121631

theorem tan_double_angle (α : ℝ) 
  (h : Real.tan α = 1 / 2) : Real.tan (2 * α) = 4 / 3 := 
by
  sorry

end tan_double_angle_l121_121631


namespace chairs_adjacent_subsets_l121_121393

theorem chairs_adjacent_subsets (n : ℕ) (h_n : n = 12) :
  (∑ k in (range n.succ).filter (λ k, k ≥ 4), (nat.choose n k)) + 84 = 1704 :=
by sorry

end chairs_adjacent_subsets_l121_121393


namespace switch_connections_l121_121548

theorem switch_connections (n : ℕ) (d : ℕ) (h1 : n = 30) (h2 : d = 4) :
  (n * d) / 2 = 60 :=
by
  rw [h1, h2]
  norm_num
  sorry

end switch_connections_l121_121548


namespace decreasing_function_range_l121_121150

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else -a * x

theorem decreasing_function_range (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (1 / 8 ≤ a ∧ a < 1 / 3) :=
by
  sorry

end decreasing_function_range_l121_121150


namespace no_common_points_l121_121129

theorem no_common_points : 
  ∀ (x y : ℝ), ¬(x^2 + y^2 = 9 ∧ x^2 + y^2 = 4) := 
by
  sorry

end no_common_points_l121_121129


namespace minimum_value_of_f_l121_121919

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3 * x + 9

-- State the theorem about the minimum value of the function
theorem minimum_value_of_f : ∃ x : ℝ, f x = 7 ∧ ∀ y : ℝ, f y ≥ 7 := sorry

end minimum_value_of_f_l121_121919


namespace wifi_cost_per_hour_l121_121438

-- Define the conditions as hypotheses
def ticket_cost : ℝ := 11
def snacks_cost : ℝ := 3
def headphones_cost : ℝ := 16
def hourly_income : ℝ := 12
def trip_duration : ℝ := 3
def total_expenses : ℝ := ticket_cost + snacks_cost + headphones_cost
def total_earnings : ℝ := hourly_income * trip_duration

-- Translate the proof problem to Lean 4 statement
theorem wifi_cost_per_hour: 
  (total_earnings - total_expenses) / trip_duration = 2 :=
by sorry

end wifi_cost_per_hour_l121_121438


namespace probability_of_both_selected_l121_121792

noncomputable def ramSelectionProbability : ℚ := 1 / 7
noncomputable def raviSelectionProbability : ℚ := 1 / 5

theorem probability_of_both_selected : 
  ramSelectionProbability * raviSelectionProbability = 1 / 35 :=
by sorry

end probability_of_both_selected_l121_121792


namespace determine_a_l121_121161

theorem determine_a (a x y : ℝ) (h : (a + 1) * x^(|a|) + y = -8) (h_linear : ∀ x y, (a + 1) * x^(|a|) + y = -8 → x ^ 1 = x): a = 1 :=
by 
  sorry

end determine_a_l121_121161


namespace cos_double_angle_l121_121700

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7/25 :=
by
  sorry

end cos_double_angle_l121_121700


namespace profit_percent_l121_121093

theorem profit_percent (P C : ℝ) (h : (2 / 3) * P = 0.88 * C) : P - C = 0.32 * C → (P - C) / C * 100 = 32 := by
  sorry

end profit_percent_l121_121093


namespace cos_diff_l121_121639

theorem cos_diff (α : ℝ) (h1 : Real.cos α = (Real.sqrt 2) / 10) (h2 : α > -π ∧ α < 0) :
  Real.cos (α - π / 4) = -3 / 5 :=
sorry

end cos_diff_l121_121639


namespace sand_removal_l121_121220

theorem sand_removal :
  let initial_weight := (8 / 3 : ℚ)
  let first_removal := (1 / 4 : ℚ)
  let second_removal := (5 / 6 : ℚ)
  initial_weight - (first_removal + second_removal) = (13 / 12 : ℚ) := by
  -- sorry is used here to skip the proof as instructed
  sorry

end sand_removal_l121_121220


namespace solve_quadratic_eq_l121_121761

theorem solve_quadratic_eq (x : ℝ) : x^2 = 4 * x → x = 0 ∨ x = 4 :=
by
  intro h
  sorry

end solve_quadratic_eq_l121_121761


namespace train_crosses_bridge_in_12_4_seconds_l121_121483

noncomputable def train_crossing_bridge_time (length_train : ℝ) (speed_train_kmph : ℝ) (length_bridge : ℝ) : ℝ :=
  let speed_train_mps := speed_train_kmph * (1000 / 3600)
  let total_distance := length_train + length_bridge
  total_distance / speed_train_mps

theorem train_crosses_bridge_in_12_4_seconds :
  train_crossing_bridge_time 110 72 138 = 12.4 :=
by
  sorry

end train_crosses_bridge_in_12_4_seconds_l121_121483


namespace distinct_solutions_diff_l121_121194

theorem distinct_solutions_diff (r s : ℝ) (hr : r > s) 
  (h : ∀ x, (x - 5) * (x + 5) = 25 * x - 125 ↔ x = r ∨ x = s) : r - s = 15 :=
sorry

end distinct_solutions_diff_l121_121194


namespace junior_score_l121_121323

theorem junior_score (total_students : ℕ) (juniors_percentage : ℝ) (seniors_percentage : ℝ)
  (class_average : ℝ) (senior_average : ℝ) (juniors_same_score : Prop) 
  (h1 : juniors_percentage = 0.2) (h2 : seniors_percentage = 0.8)
  (h3 : class_average = 85) (h4 : senior_average = 84) : 
  ∃ junior_score : ℝ, juniors_same_score → junior_score = 89 :=
by
  sorry

end junior_score_l121_121323


namespace maximum_cells_covered_at_least_five_times_l121_121075

theorem maximum_cells_covered_at_least_five_times :
  let areas := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let total_covered := List.sum areas
  let exact_coverage := 1 * 1 + 2 * 2 + 3 * 3 + 4 * 4
  let remaining_coverage := total_covered - exact_coverage
  let max_cells_covered_at_least_five := remaining_coverage / 5
  max_cells_covered_at_least_five = 5 :=
by
  sorry

end maximum_cells_covered_at_least_five_times_l121_121075


namespace proof_of_problem_l121_121399

theorem proof_of_problem (a b : ℝ) (h1 : a > b) (h2 : a * b = a / b) : b = 1 ∧ 0 < a :=
by
  sorry

end proof_of_problem_l121_121399


namespace sum_of_two_terms_is_term_iff_a_is_multiple_of_d_l121_121451

theorem sum_of_two_terms_is_term_iff_a_is_multiple_of_d
    (a d : ℤ) 
    (n k : ℕ) 
    (h : ∀ (p : ℕ), a + d * n + (a + d * k) = a + d * p)
    : ∃ m : ℤ, a = d * m :=
sorry

end sum_of_two_terms_is_term_iff_a_is_multiple_of_d_l121_121451


namespace soda_mineral_cost_l121_121607

theorem soda_mineral_cost
  (x y : ℝ)
  (h1 : 2 * x + y = 7)
  (h2 : 4 * x + 3 * y = 16) :
  10 * x + 10 * y = 45 :=
  sorry

end soda_mineral_cost_l121_121607


namespace anna_reading_time_l121_121271

theorem anna_reading_time
  (total_chapters : ℕ := 31)
  (reading_time_per_chapter : ℕ := 20)
  (hours_in_minutes : ℕ := 60) :
  let skipped_chapters := total_chapters / 3;
  let read_chapters := total_chapters - skipped_chapters;
  let total_reading_time_minutes := read_chapters * reading_time_per_chapter;
  let total_reading_time_hours := total_reading_time_minutes / hours_in_minutes;
  total_reading_time_hours = 7 :=
by
  sorry

end anna_reading_time_l121_121271


namespace badge_counts_l121_121563

variable (x : ℝ) -- Number of badges Tolya initially had
variable (y : ℝ) -- Number of badges Vasya initially had 

-- Conditions
axiom initial_more_badges : y = x + 5
axiom exchange_badges_vasya : y - 0.24 * y + 0.2 * x = x - 1 + 0.24 * y
axiom exchange_badges_tolya : x - 0.2 * x + 0.24 * y = x - 0.2 * x + 0.24 * y - 0.24 * y + y

-- Proof goal
theorem badge_counts (x : ℝ) (y : ℝ) 
  (h1 : y = x + 5) 
  (h2 : y - 0.24 * y + 0.2 * x = x - 1 + 0.24 * y) 
  (h3 : x - 0.2 * x + 0.24 * y = x - 0.2 * x + 0.24 * y - 0.24 * y + y) : 
  x = 45 ∧ y = 50 :=
by
  sorry

end badge_counts_l121_121563


namespace evaluate_expression_l121_121816

theorem evaluate_expression : 1 - (-2) * 2 - 3 - (-4) * 2 - 5 - (-6) * 2 = 17 := 
by
  sorry

end evaluate_expression_l121_121816


namespace r_s_t_u_bounds_l121_121298

theorem r_s_t_u_bounds (r s t u : ℝ) 
  (H1: 5 * r + 4 * s + 3 * t + 6 * u = 100)
  (H2: r ≥ s)
  (H3: s ≥ t)
  (H4: t ≥ u)
  (H5: u ≥ 0) :
  20 ≤ r + s + t + u ∧ r + s + t + u ≤ 25 := 
sorry

end r_s_t_u_bounds_l121_121298


namespace cos_double_angle_l121_121686

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3/5) : Real.cos (2 * theta) = -7/25 :=
by
  sorry

end cos_double_angle_l121_121686


namespace polygon_sides_l121_121767

theorem polygon_sides (n : ℕ) 
  (h1 : (n-2) * 180 = 3 * 360) 
  (h2 : n > 2) : n = 8 :=
sorry

end polygon_sides_l121_121767


namespace sequence_term_4th_l121_121479

theorem sequence_term_4th (a_n : ℕ → ℝ) (h : ∀ n, a_n n = 2 / (n^2 + n)) :
  ∃ n, a_n n = 1 / 10 ∧ n = 4 :=
by
  sorry

end sequence_term_4th_l121_121479


namespace joan_total_seashells_l121_121717

-- Definitions of the conditions
def joan_initial_seashells : ℕ := 79
def mike_additional_seashells : ℕ := 63

-- Definition of the proof problem statement
theorem joan_total_seashells : joan_initial_seashells + mike_additional_seashells = 142 :=
by
  -- Proof would go here
  sorry

end joan_total_seashells_l121_121717


namespace f_2202_minus_f_2022_l121_121879

-- Definitions and conditions
def f : ℕ+ → ℕ+ := sorry -- The exact function is provided through conditions and will be proven property-wise.

axiom f_increasing {a b : ℕ+} : a < b → f a < f b
axiom f_range (n : ℕ+) : ∃ m : ℕ+, f n = ⟨m, sorry⟩ -- ensuring f maps to ℕ+
axiom f_property (n : ℕ+) : f (f n) = 3 * n

-- Prove the statement
theorem f_2202_minus_f_2022 : f 2202 - f 2022 = 1638 :=
by sorry

end f_2202_minus_f_2022_l121_121879


namespace crayons_count_l121_121354

theorem crayons_count (l b f : ℕ) (h1 : l = b / 2) (h2 : b = 3 * f) (h3 : l = 27) : f = 18 :=
by
  sorry

end crayons_count_l121_121354


namespace units_digit_of_27_times_36_l121_121619

theorem units_digit_of_27_times_36 :
  let units_digit := fun (n : ℕ) => n % 10
  in units_digit (27 * 36) = 2 :=
by
  let units_digit := fun (n : ℕ) => n % 10
  have h27: units_digit 27 = 7 := by
    show 27 % 10 = 7
    sorry
  have h36: units_digit 36 = 6 := by
    show 36 % 10 = 6
    sorry
  have h42: units_digit (7 * 6) = 2 := by
    show 42 % 10 = 2
    sorry
  exact h42

end units_digit_of_27_times_36_l121_121619


namespace cos_double_angle_l121_121662

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3 / 5) : Real.cos (2 * theta) = -7 / 25 :=
by
  sorry

end cos_double_angle_l121_121662


namespace cos_double_angle_l121_121660

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 := sorry

end cos_double_angle_l121_121660


namespace subsets_with_at_least_four_adjacent_chairs_l121_121378

/-- The number of subsets of a set of 12 chairs arranged in a circle that contain at least four adjacent chairs is 1776. -/
theorem subsets_with_at_least_four_adjacent_chairs (S : Finset (Fin 12)) :
  let n := 12 in
  ∃ F : Finset (Finset (Fin 12)), (∀ s ∈ F, ∃ l : List (Fin 12), (l.length ≥ 4 ∧ l.nodup ∧ ∀ i, i ∈ l → (i + 1) % n ∈ l)) ∧ F.card = 1776 := 
sorry

end subsets_with_at_least_four_adjacent_chairs_l121_121378


namespace sum_of_money_l121_121542

-- Conditions
def mass_record_coin_kg : ℝ := 100  -- 100 kg
def mass_one_pound_coin_g : ℝ := 10  -- 10 g

-- Conversion factor
def kg_to_g : ℝ := 1000

-- Question: Prove the sum of money in £1 coins that weighs the same as the record-breaking coin is £10,000.
theorem sum_of_money 
  (mass_record_coin_g := mass_record_coin_kg * kg_to_g)
  (number_of_coins := mass_record_coin_g / mass_one_pound_coin_g) 
  (sum_of_money := number_of_coins) : 
  sum_of_money = 10000 :=
  sorry

end sum_of_money_l121_121542


namespace baker_made_cakes_l121_121120

theorem baker_made_cakes (sold_cakes left_cakes total_cakes : ℕ) (h1 : sold_cakes = 108) (h2 : left_cakes = 59) :
  total_cakes = sold_cakes + left_cakes → total_cakes = 167 := by
  intro h
  rw [h1, h2] at h
  exact h

-- The proof part is omitted since only the statement is required

end baker_made_cakes_l121_121120


namespace farmer_apples_l121_121530

theorem farmer_apples : 127 - 39 = 88 := by
  -- Skipping proof details
  sorry

end farmer_apples_l121_121530


namespace koschei_never_equal_l121_121101

-- Define the problem setup 
def coins_at_vertices (n1 n2 n3 n4 n5 n6 : ℕ) : Prop := 
  ∃ k : ℕ, n1 = k ∧ n2 = k ∧ n3 = k ∧ n4 = k ∧ n5 = k ∧ n6 = k

-- Define the operation condition
def operation_condition (n1 n2 n3 n4 n5 n6 : ℕ) : Prop :=
  ∃ x : ℕ, (n1 - x = x ∧ n2 + 6 * x = x) ∨ (n2 - x = x ∧ n3 + 6 * x = x) ∨ 
  (n3 - x = x ∧ n4 + 6 * x = x) ∨ (n4 - x = x ∧ n5 + 6 * x = x) ∨ 
  (n5 - x = x ∧ n6 + 6 * x = x) ∨ (n6 - x = x ∧ n1 + 6 * x = x)

-- The main theorem 
theorem koschei_never_equal (n1 n2 n3 n4 n5 n6 : ℕ) : 
  (∃ x : ℕ, coins_at_vertices n1 n2 n3 n4 n5 n6) → False :=
by
  sorry

end koschei_never_equal_l121_121101


namespace rubber_ball_radius_l121_121103

theorem rubber_ball_radius (r : ℝ) (radius_exposed_section : ℝ) (depth : ℝ) 
  (h1 : radius_exposed_section = 20) 
  (h2 : depth = 12) 
  (h3 : (r - depth)^2 + radius_exposed_section^2 = r^2) : 
  r = 22.67 :=
by
  sorry

end rubber_ball_radius_l121_121103


namespace sum_remainder_zero_l121_121949

theorem sum_remainder_zero
  (a b c : ℕ)
  (h₁ : a % 53 = 31)
  (h₂ : b % 53 = 15)
  (h₃ : c % 53 = 7) :
  (a + b + c) % 53 = 0 :=
by
  sorry

end sum_remainder_zero_l121_121949


namespace committee_count_is_252_l121_121327

/-- Definition of binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Problem statement: Number of ways to choose a 5-person committee from a club of 10 people is 252 -/
theorem committee_count_is_252 : binom 10 5 = 252 :=
by
  sorry

end committee_count_is_252_l121_121327


namespace compute_expression_l121_121277

theorem compute_expression : ((-5) * 3) - (7 * (-2)) + ((-4) * (-6)) = 23 := by
  sorry

end compute_expression_l121_121277


namespace domain_of_f_x_squared_l121_121153

theorem domain_of_f_x_squared {f : ℝ → ℝ} (h : ∀ x, -2 ≤ x ∧ x ≤ 3 → ∃ y, f (x + 1) = y) : 
  ∀ x, -2 ≤ x ∧ x ≤ 2 → ∃ y, f (x ^ 2) = y := 
by 
  sorry

end domain_of_f_x_squared_l121_121153


namespace gino_gave_away_l121_121463

theorem gino_gave_away (initial_sticks given_away left_sticks : ℝ) 
  (h1 : initial_sticks = 63.0) (h2 : left_sticks = 13.0) 
  (h3 : left_sticks = initial_sticks - given_away) : 
  given_away = 50.0 :=
by
  sorry

end gino_gave_away_l121_121463


namespace anna_reading_time_l121_121272

theorem anna_reading_time
  (total_chapters : ℕ := 31)
  (reading_time_per_chapter : ℕ := 20)
  (hours_in_minutes : ℕ := 60) :
  let skipped_chapters := total_chapters / 3;
  let read_chapters := total_chapters - skipped_chapters;
  let total_reading_time_minutes := read_chapters * reading_time_per_chapter;
  let total_reading_time_hours := total_reading_time_minutes / hours_in_minutes;
  total_reading_time_hours = 7 :=
by
  sorry

end anna_reading_time_l121_121272


namespace total_cars_produced_l121_121580

theorem total_cars_produced (cars_NA cars_EU : ℕ) (h1 : cars_NA = 3884) (h2 : cars_EU = 2871) : cars_NA + cars_EU = 6755 := by
  sorry

end total_cars_produced_l121_121580


namespace largest_and_smallest_value_of_expression_l121_121994

theorem largest_and_smallest_value_of_expression
  (w x y z : ℝ)
  (h1 : w + x + y + z = 0)
  (h2 : w^7 + x^7 + y^7 + z^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 :=
sorry

end largest_and_smallest_value_of_expression_l121_121994


namespace pears_count_l121_121206

theorem pears_count (A F P : ℕ)
  (hA : A = 12)
  (hF : F = 4 * 12 + 3)
  (hP : P = F - A) :
  P = 39 := by
  sorry

end pears_count_l121_121206


namespace triangle_solution_l121_121712

noncomputable theory
open Real

-- Definitions based on given conditions
def acute_triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π

variables {A B C a b c : ℝ}

-- The mathematical proof problem
theorem triangle_solution
  (h_acute : acute_triangle A B C)
  (h_side : a = sin A ∧ b = sin B ∧ c = sin C)
  (h_condition : 2 * b * sin A = sqrt 3 * a) :
  (B = π / 3) ∧ (frac (cos A + cos B + cos C) ∈ Icc (frac ((sqrt 3 + 1)/2)) (frac 3/2)) :=
by sorry

end triangle_solution_l121_121712


namespace angle_terminal_side_on_non_negative_y_axis_l121_121302

theorem angle_terminal_side_on_non_negative_y_axis (P : ℝ × ℝ) (α : ℝ) (hP : P = (0, 3)) :
  α = some_angle_with_terminal_side_on_non_negative_y_axis := by
  sorry

end angle_terminal_side_on_non_negative_y_axis_l121_121302


namespace correlation_index_l121_121500

-- Define the conditions given in the problem
def height_explains_weight_variation : Prop :=
  ∃ R : ℝ, R^2 = 0.64

-- State the main conjecture (actual proof omitted for simplicity)
theorem correlation_index (R : ℝ) (h : height_explains_weight_variation) : R^2 = 0.64 := by
  sorry

end correlation_index_l121_121500


namespace adults_had_meal_l121_121587

theorem adults_had_meal (A : ℕ) (h1 : 70 ≥ A) (h2 : ((70 - A) * 9) = (72 * 7)) : A = 14 := 
by
  sorry

end adults_had_meal_l121_121587


namespace total_amount_is_152_l121_121807

noncomputable def total_amount (p q r s t : ℝ) : ℝ := p + q + r + s + t

noncomputable def p_share (x : ℝ) : ℝ := 2 * x
noncomputable def q_share (x : ℝ) : ℝ := 1.75 * x
noncomputable def r_share (x : ℝ) : ℝ := 1.5 * x
noncomputable def s_share (x : ℝ) : ℝ := 1.25 * x
noncomputable def t_share (x : ℝ) : ℝ := 1.1 * x

theorem total_amount_is_152 (x : ℝ) (h1 : q_share x = 35) :
  total_amount (p_share x) (q_share x) (r_share x) (s_share x) (t_share x) = 152 := by
  sorry

end total_amount_is_152_l121_121807


namespace tangent_function_range_l121_121258

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/3) * x^3 - (a/2) * x^2 + 1
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x

theorem tangent_function_range {a : ℝ} :
  (∃ (m : ℝ), 4 * m^3 - 3 * a * m^2 + 6 = 0) ↔ a > 2 * Real.sqrt 33 :=
sorry -- proof omitted

end tangent_function_range_l121_121258


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l121_121389

theorem number_of_subsets_with_at_least_four_adjacent_chairs : 
  ∀ (n : ℕ), n = 12 → 
  (∃ (s : Finset (Finset (Fin n))), s.card = 1610 ∧ 
  (∀ (A : Finset (Fin n)), A ∈ s → (∃ (start : Fin n), ∀ i, i ∈ Finset.range 4 → A.contains (start + i % n)))) :=
by
  sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l121_121389


namespace find_angle_APB_l121_121499

open Real

noncomputable def angle_APB
    (PA_tangent_SAR: Bool) 
    (PB_tangent_RBT: Bool) 
    (SRT_straight: Bool)
    (arc_AS: ℝ)
    (arc_BT: ℝ) 
    : ℝ := 
  sorry

theorem find_angle_APB 
    (PA_tangent_SAR: Bool) 
    (PB_tangent_RBT: Bool) 
    (SRT_straight: Bool) 
    (arc_AS: ℝ := 45)
    (arc_BT: ℝ := 30) 
    :
    arc_AS = 45 → 
    arc_BT = 30 → 
    angle_APB PA_tangent_SAR PB_tangent_RBT SRT_straight arc_AS arc_BT = 75 :=
  sorry

end find_angle_APB_l121_121499


namespace coeff_x6_in_expansion_l121_121403

theorem coeff_x6_in_expansion : 
  (Polynomial.coeff ((1 - 3 * Polynomial.X ^ 3) ^ 7 : Polynomial ℤ) 6) = 189 :=
by
  sorry

end coeff_x6_in_expansion_l121_121403


namespace line_through_point_l121_121987

theorem line_through_point (k : ℝ) :
  (∃ k : ℝ, ∀ x y : ℝ, (x = 3) ∧ (y = -2) → (2 - 3 * k * x = -4 * y)) → k = -2/3 :=
by
  sorry

end line_through_point_l121_121987


namespace arithmetic_sequence_third_term_l121_121531

theorem arithmetic_sequence_third_term :
  ∀ (a d : ℤ), (a + 4 * d = 2) ∧ (a + 5 * d = 5) → (a + 2 * d = -4) :=
by sorry

end arithmetic_sequence_third_term_l121_121531


namespace part_one_part_two_l121_121306

-- Defining the function and its first derivative
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x
def f' (a b : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + b

-- Part (Ⅰ)
theorem part_one (a b : ℝ)
  (H1 : f' a b 3 = 24)
  (H2 : f' a b 1 = 0) :
  a = 1 ∧ b = -3 ∧ (∀ x, -1 ≤ x ∧ x ≤ 1 → f' 1 (-3) x ≤ 0) :=
sorry

-- Part (Ⅱ)
theorem part_two (b : ℝ)
  (H1 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → 3 * x^2 + b ≤ 0) :
  b ≤ -3 :=
sorry

end part_one_part_two_l121_121306


namespace jude_age_today_l121_121852
-- Import the necessary libraries

-- Define the conditions as hypotheses and then state the required proof
theorem jude_age_today (heath_age_today : ℕ) (heath_age_in_5_years : ℕ) (jude_age_in_5_years : ℕ) 
  (H1 : heath_age_today = 16)
  (H2 : heath_age_in_5_years = heath_age_today + 5)
  (H3 : heath_age_in_5_years = 3 * jude_age_in_5_years) :
  jude_age_in_5_years - 5 = 2 :=
by
  -- Given conditions imply Jude's age today is 2. Proof is omitted.
  sorry

end jude_age_today_l121_121852


namespace ratio_of_supply_to_demand_l121_121628

def supply : ℕ := 1800000
def demand : ℕ := 2400000

theorem ratio_of_supply_to_demand : (supply / demand : ℚ) = 3 / 4 := by
  sorry

end ratio_of_supply_to_demand_l121_121628


namespace cos_double_angle_l121_121693

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3/5) : cos (2*θ) = -7/25 := 
by 
  sorry

end cos_double_angle_l121_121693


namespace additional_cars_can_fit_l121_121498

-- Given definitions and conditions
variable (total_cars : Nat)
variable (levels : Nat)
variable (parked_cars : Nat)

-- Specific conditions for our problem
def total_cars := 425
def levels := 5
def parked_cars := 23

-- Goal statement
theorem additional_cars_can_fit : (total_cars / levels) - parked_cars = 62 := by
  sorry

end additional_cars_can_fit_l121_121498


namespace new_year_markup_l121_121800

variable (C : ℝ) -- original cost of the turtleneck sweater
variable (N : ℝ) -- New Year season markup in decimal form
variable (final_price : ℝ) -- final price in February

-- Conditions
def initial_markup (C : ℝ) := 1.20 * C
def after_new_year_markup (C : ℝ) (N : ℝ) := (1 + N) * initial_markup C
def discount_in_february (C : ℝ) (N : ℝ) := 0.94 * after_new_year_markup C N
def profit_in_february (C : ℝ) := 1.41 * C

-- Mathematically equivalent proof problem (statement only)
theorem new_year_markup :
  ∀ C : ℝ, ∀ N : ℝ,
    discount_in_february C N = profit_in_february C →
    N = 0.5 :=
by
  sorry

end new_year_markup_l121_121800


namespace discriminant_of_trinomial_l121_121201

theorem discriminant_of_trinomial (x1 x2 : ℝ) (h : x2 - x1 = 2) : (x2 - x1)^2 = 4 :=
by
  sorry

end discriminant_of_trinomial_l121_121201


namespace sum_even_102_to_200_l121_121375

def sum_even_integers (m n : ℕ) : ℕ :=
  sum (list.map (λ k, 2 * k) (list.range' m (n - m + 1)))

theorem sum_even_102_to_200 :
  sum_even_integers (102 / 2) (200 / 2) = 7550 := 
sorry

end sum_even_102_to_200_l121_121375


namespace multiply_powers_zero_exponent_distribute_term_divide_powers_l121_121978

-- 1. Prove a^{2} \cdot a^{3} = a^{5}
theorem multiply_powers (a : ℝ) : a^2 * a^3 = a^5 := 
sorry

-- 2. Prove (3.142 - π)^{0} = 1
theorem zero_exponent : (3.142 - Real.pi)^0 = 1 := 
sorry

-- 3. Prove 2a(a^{2} - 1) = 2a^{3} - 2a
theorem distribute_term (a : ℝ) : 2 * a * (a^2 - 1) = 2 * a^3 - 2 * a := 
sorry

-- 4. Prove (-m^{3})^{2} \div m^{4} = m^{2}
theorem divide_powers (m : ℝ) : ((-m^3)^2) / (m^4) = m^2 := 
sorry

end multiply_powers_zero_exponent_distribute_term_divide_powers_l121_121978


namespace jeans_price_difference_l121_121407

variable (x : Real)

theorem jeans_price_difference
  (hx : 0 < x) -- Assuming x > 0 for a positive cost
  (r := 1.40 * x)
  (c := 1.30 * r) :
  c = 1.82 * x :=
by
  sorry

end jeans_price_difference_l121_121407


namespace LittleJohnnyAnnualIncome_l121_121881

theorem LittleJohnnyAnnualIncome :
  ∀ (total_amount bank_amount bond_amount : ℝ) 
    (bank_interest bond_interest annual_income : ℝ),
    total_amount = 10000 →
    bank_amount = 6000 →
    bond_amount = 4000 →
    bank_interest = 0.05 →
    bond_interest = 0.09 →
    annual_income = bank_amount * bank_interest + bond_amount * bond_interest →
    annual_income = 660 :=
by
  intros total_amount bank_amount bond_amount bank_interest bond_interest annual_income 
  intros h_total_amount h_bank_amount h_bond_amount h_bank_interest h_bond_interest h_annual_income
  -- Proof is not required
  sorry

end LittleJohnnyAnnualIncome_l121_121881


namespace area_triangle_3_6_l121_121572

/-
Problem: Prove that the area of a triangle with base 3 meters and height 6 meters is 9 square meters.
Definitions: 
- base: The base of the triangle is 3 meters.
- height: The height of the triangle is 6 meters.
Conditions: 
- The area of a triangle formula.
Correct Answer: 9 square meters.
-/

def area_of_triangle (base height : ℕ) : ℕ := (base * height) / 2

theorem area_triangle_3_6 : area_of_triangle 3 6 = 9 := by
  sorry

end area_triangle_3_6_l121_121572


namespace seashells_at_end_of_month_l121_121134

-- Given conditions as definitions
def initial_seashells : ℕ := 50
def increase_per_week : ℕ := 20

-- Define function to calculate seashells in the nth week
def seashells_in_week (n : ℕ) : ℕ :=
  initial_seashells + n * increase_per_week

-- Lean statement to prove the number of seashells in the jar at the end of four weeks is 130
theorem seashells_at_end_of_month : seashells_in_week 4 = 130 :=
by
  sorry

end seashells_at_end_of_month_l121_121134


namespace partial_fraction_product_l121_121369

theorem partial_fraction_product (A B C : ℝ) :
  (∀ x : ℝ, x^3 - 3 * x^2 - 10 * x + 24 ≠ 0 →
            (x^2 - 25) / (x^3 - 3 * x^2 - 10 * x + 24) = A / (x - 2) + B / (x + 3) + C / (x - 4)) →
  A = 1 ∧ B = 1 ∧ C = 1 →
  A * B * C = 1 := by
  sorry

end partial_fraction_product_l121_121369


namespace find_x_l121_121014

open Real

theorem find_x (x : ℝ) (h : (x / 6) / 3 = 6 / (x / 3)) : x = 18 ∨ x = -18 :=
by
  sorry

end find_x_l121_121014


namespace original_number_of_workers_l121_121245

-- Definitions of the conditions given in the problem
def workers_days (W : ℕ) : ℕ := 35
def additional_workers : ℕ := 10
def reduced_days : ℕ := 10

-- The main theorem we need to prove
theorem original_number_of_workers (W : ℕ) (A : ℕ) 
  (h1 : W * workers_days W = (W + additional_workers) * (workers_days W - reduced_days)) :
  W = 25 :=
by
  sorry

end original_number_of_workers_l121_121245


namespace find_special_n_l121_121823

open Nat

theorem find_special_n (m : ℕ) (hm : m ≥ 3) :
  ∃ (n : ℕ), 
    (n = m^2 - 2) ∧ (∃ (k : ℕ), 1 ≤ k ∧ k < n ∧ 2 * (Nat.choose n k) = (Nat.choose n (k - 1) + Nat.choose n (k + 1))) :=
by
  sorry

end find_special_n_l121_121823


namespace product_of_two_numbers_l121_121777

theorem product_of_two_numbers (x y : ℝ) 
  (h1 : x - y = 1 * k) 
  (h2 : x + y = 2 * k) 
  (h3 : (x * y)^2 = 18 * k) : (x * y = 16) := 
by 
    sorry


end product_of_two_numbers_l121_121777


namespace notebooks_problem_l121_121400

variable (a b c : ℕ)

theorem notebooks_problem (h1 : a + 6 = b + c) (h2 : b + 10 = a + c) : c = 8 :=
  sorry

end notebooks_problem_l121_121400


namespace breadth_of_rectangular_plot_l121_121573

theorem breadth_of_rectangular_plot :
  ∃ b : ℝ, (∃ l : ℝ, l = 3 * b ∧ l * b = 867) ∧ b = 17 :=
by
  sorry

end breadth_of_rectangular_plot_l121_121573


namespace determinant_tan_matrix_l121_121283

theorem determinant_tan_matrix (B C : ℝ) (h : B + C = 3 * π / 4) :
  Matrix.det ![
    ![Real.tan (π / 4), 1, 1],
    ![1, Real.tan B, 1],
    ![1, 1, Real.tan C]
  ] = 1 :=
by
  sorry

end determinant_tan_matrix_l121_121283


namespace two_digit_numbers_condition_l121_121954

theorem two_digit_numbers_condition (a b : ℕ) (h1 : a ≠ 0) (h2 : 1 ≤ a ∧ a ≤ 9) (h3 : 0 ≤ b ∧ b ≤ 9) :
  (a + 1) * (b + 1) = 10 * a + b + 1 ↔ b = 9 := 
sorry

end two_digit_numbers_condition_l121_121954


namespace katrina_cookies_left_l121_121343

theorem katrina_cookies_left (initial_cookies morning_cookies_sold lunch_cookies_sold afternoon_cookies_sold : ℕ)
  (h1 : initial_cookies = 120)
  (h2 : morning_cookies_sold = 36)
  (h3 : lunch_cookies_sold = 57)
  (h4 : afternoon_cookies_sold = 16) :
  initial_cookies - (morning_cookies_sold + lunch_cookies_sold + afternoon_cookies_sold) = 11 := 
by 
  sorry

end katrina_cookies_left_l121_121343


namespace students_in_all_sections_is_six_l121_121772

-- Define the number of students in each section and the total.
variable (total_students : ℕ := 30)
variable (music_students : ℕ := 15)
variable (drama_students : ℕ := 18)
variable (dance_students : ℕ := 12)
variable (at_least_two_sections : ℕ := 14)

-- Define the number of students in all three sections.
def students_in_all_three_sections (total_students music_students drama_students dance_students at_least_two_sections : ℕ) : ℕ :=
  let a := 6 -- the result we want to prove
  a

-- The theorem proving that the number of students in all three sections is 6.
theorem students_in_all_sections_is_six :
  students_in_all_three_sections total_students music_students drama_students dance_students at_least_two_sections = 6 :=
by 
  sorry -- Proof is omitted

end students_in_all_sections_is_six_l121_121772


namespace equation_has_exactly_one_solution_l121_121460

theorem equation_has_exactly_one_solution (m : ℝ) : 
  (m ∈ { -1 } ∪ Set.Ioo (-1/2 : ℝ) (1/0) ) ↔ ∃ (x : ℝ), 2 * Real.sqrt (1 - m * (x + 2)) = x + 4 :=
sorry

end equation_has_exactly_one_solution_l121_121460


namespace marble_probability_is_correct_l121_121140

def marbles_probability
  (total_marbles: ℕ) 
  (red_marbles: ℕ) 
  (blue_marbles: ℕ) 
  (green_marbles: ℕ)
  (choose_marbles: ℕ) 
  (required_red: ℕ) 
  (required_blue: ℕ) 
  (required_green: ℕ): ℚ := sorry

-- Define conditions
def total_marbles := 7
def red_marbles := 3
def blue_marbles := 2
def green_marbles := 2
def choose_marbles := 4
def required_red := 2
def required_blue := 1
def required_green := 1

-- Proof statement
theorem marble_probability_is_correct : 
  marbles_probability total_marbles red_marbles blue_marbles green_marbles choose_marbles required_red required_blue required_green = (12 / 35 : ℚ) :=
sorry

end marble_probability_is_correct_l121_121140


namespace number_of_non_representable_l121_121348

theorem number_of_non_representable :
  ∀ (a b : ℕ), Nat.gcd a b = 1 →
  (∃ n : ℕ, ¬ ∃ x y : ℕ, n = a * x + b * y) :=
sorry

end number_of_non_representable_l121_121348


namespace sin_a_mul_sin_c_eq_sin_sq_b_zero_lt_B_le_pi_div_3_magnitude_BC_add_BA_l121_121861

open Real

namespace TriangleProofs

variables 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (BA BC : ℝ) 
  (h1 : sin B = sqrt 7 / 4) 
  (h2 : (cos A / sin A + cos C / sin C = 4 * sqrt 7 / 7)) 
  (h3 : BA * BC = 3 / 2)
  (h4 : a = b ∧ c = b)

-- 1. Prove that sin A * sin C = sin^2 B
theorem sin_a_mul_sin_c_eq_sin_sq_b : sin A * sin C = sin B ^ 2 := 
by sorry

-- 2. Prove that 0 < B ≤ π / 3
theorem zero_lt_B_le_pi_div_3 : 0 < B ∧ B ≤ π / 3 := 
by sorry

-- 3. Find the magnitude of the vector sum.
theorem magnitude_BC_add_BA : abs (BC + BA) = 2 * sqrt 2 := 
by sorry

end TriangleProofs

end sin_a_mul_sin_c_eq_sin_sq_b_zero_lt_B_le_pi_div_3_magnitude_BC_add_BA_l121_121861


namespace Pythagorean_triple_l121_121810

theorem Pythagorean_triple (n : ℕ) (hn : n % 2 = 1) (hn_geq : n ≥ 3) :
  n^2 + ((n^2 - 1) / 2)^2 = ((n^2 + 1) / 2)^2 := by
  sorry

end Pythagorean_triple_l121_121810


namespace double_angle_cosine_l121_121676

theorem double_angle_cosine (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end double_angle_cosine_l121_121676


namespace peg_arrangement_count_l121_121950

theorem peg_arrangement_count :
  let yellow_pegs := 6
  let red_pegs := 5
  let green_pegs := 4
  let blue_pegs := 3
  let orange_pegs := 2
  let rows := 6
  let columns := 5
  let factorial (n : ℕ) : ℕ := (list.range (n + 1)).foldl (*) 1 in
  (factorial yellow_pegs) * (factorial red_pegs) * (factorial green_pegs) *
  (factorial blue_pegs) * (factorial orange_pegs) = 86400 :=
by
  rw [factorial, list.range, list.foldl, ← nat.add_sub_of_le, ← list.range_succ, add_comm] 
  sorry

end peg_arrangement_count_l121_121950


namespace seventeen_number_selection_l121_121830

theorem seventeen_number_selection : ∃ (n : ℕ), (∀ s : Finset ℕ, (s ⊆ Finset.range 17) → (Finset.card s = n) → ∃ x y : ℕ, (x ∈ s) ∧ (y ∈ s) ∧ (x ≠ y) ∧ (x = 3 * y ∨ y = 3 * x)) ∧ (n = 13) :=
by
  sorry

end seventeen_number_selection_l121_121830


namespace time_period_simple_interest_l121_121963

theorem time_period_simple_interest 
  (P : ℝ) (R18 R12 : ℝ) (additional_interest : ℝ) (T : ℝ) :
  P = 2500 →
  R18 = 0.18 →
  R12 = 0.12 →
  additional_interest = 300 →
  P * R18 * T = P * R12 * T + additional_interest →
  T = 2 :=
by
  intros P_val R18_val R12_val add_int_val interest_eq
  rw [P_val, R18_val, R12_val, add_int_val] at interest_eq
  -- Continue the proof here
  sorry

end time_period_simple_interest_l121_121963


namespace integral_problem_l121_121615

noncomputable def integrand (x : ℝ) : ℝ := real.sqrt (1 - x^2) + x + x^3

theorem integral_problem :
  ∫ x in 0..1, integrand x = (Real.pi + 3) / 4 :=
by
  sorry

end integral_problem_l121_121615


namespace units_digit_multiplication_l121_121622

-- Define a function to find the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Statement of the problem: Given the product 27 * 36, prove that the units digit is 2.
theorem units_digit_multiplication (a b : ℕ) (h1 : units_digit 27 = 7) (h2 : units_digit 36 = 6) :
  units_digit (27 * 36) = 2 :=
by
  have h3 : units_digit (7 * 6) = 2 := by sorry
  exact h3

end units_digit_multiplication_l121_121622


namespace min_value_fract_ineq_l121_121636

theorem min_value_fract_ineq (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (1 / a + 9 / b) ≥ 16 := 
sorry

end min_value_fract_ineq_l121_121636


namespace number_of_correct_conclusions_l121_121914

-- Define the conditions given in the problem
def conclusion1 (x : ℝ) : Prop := x > 0 → x > Real.sin x
def conclusion2 (x : ℝ) : Prop := (x - Real.sin x = 0 → x = 0) → (x ≠ 0 → x - Real.sin x ≠ 0)
def conclusion3 (p q : Prop) : Prop := (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)
def conclusion4 : Prop := ¬(∀ x : ℝ, x - Real.log x > 0) = ∃ x : ℝ, x - Real.log x ≤ 0

-- Prove the number of correct conclusions is 3
theorem number_of_correct_conclusions : 
  (∃ x1 : ℝ, conclusion1 x1) ∧
  (∃ x1 : ℝ, conclusion2 x1) ∧
  (∃ p q : Prop, conclusion3 p q) ∧
  ¬conclusion4 →
  3 = 3 :=
by
  intros
  sorry

end number_of_correct_conclusions_l121_121914


namespace stuffed_animal_total_l121_121887

/-- McKenna has 34 stuffed animals. -/
def mckenna_stuffed_animals : ℕ := 34

/-- Kenley has twice as many stuffed animals as McKenna. -/
def kenley_stuffed_animals : ℕ := 2 * mckenna_stuffed_animals

/-- Tenly has 5 more stuffed animals than Kenley. -/
def tenly_stuffed_animals : ℕ := kenley_stuffed_animals + 5

/-- The total number of stuffed animals the three girls have. -/
def total_stuffed_animals : ℕ := mckenna_stuffed_animals + kenley_stuffed_animals + tenly_stuffed_animals

/-- Prove that the total number of stuffed animals is 175. -/
theorem stuffed_animal_total : total_stuffed_animals = 175 := by
  sorry

end stuffed_animal_total_l121_121887


namespace bc_money_l121_121113

variables (A B C : ℕ)

theorem bc_money (h1 : A + B + C = 400) (h2 : A + C = 300) (h3 : C = 50) : B + C = 150 :=
sorry

end bc_money_l121_121113


namespace line_equation_final_equation_l121_121465

theorem line_equation (k : ℝ) : 
  (∀ x y, y = k * (x - 1) + 1 ↔ 
  ∀ x y, y = k * ((x + 2) - 1) + 1 - 1) → 
  k = 1 / 2 :=
by
  sorry

theorem final_equation : 
  ∃ k : ℝ, k = 1 / 2 ∧ (∀ x y, y = k * (x - 1) + 1) → 
  ∀ x y, x - 2 * y + 1 = 0 :=
by
  sorry

end line_equation_final_equation_l121_121465


namespace triangle_sine_identity_triangle_cosine_identity_l121_121335

variables {A B C a b c : ℝ} (R : ℝ)

-- Law of Sines assumption
axiom law_of_sines : a = 2 * R * sin A ∧ b = 2 * R * sin B ∧ c = 2 * R * sin C

theorem triangle_sine_identity : 
    (a * sin ((B - C) / 2) / sin (A / 2) + 
     b * sin ((C - A) / 2) / sin (B / 2) + 
     c * sin ((A - B) / 2) / sin (C / 2)) = 0 :=
sorry 

theorem triangle_cosine_identity :
    (a * sin ((B - C) / 2) / cos (A / 2) + 
     b * sin ((C - A) / 2) / cos (B / 2) + 
     c * sin ((A - B) / 2) / cos (C / 2)) = 0 :=
sorry

end triangle_sine_identity_triangle_cosine_identity_l121_121335


namespace expression_numerator_l121_121019

theorem expression_numerator (p q : ℕ) (E : ℕ) 
  (h1 : p * 5 = q * 4)
  (h2 : (18 / 7) + (E / (2 * q + p)) = 3) : E = 6 := 
by 
  sorry

end expression_numerator_l121_121019


namespace power_mod_remainder_l121_121084

theorem power_mod_remainder :
  3 ^ 3021 % 13 = 1 :=
by
  sorry

end power_mod_remainder_l121_121084


namespace find_b_l121_121310

theorem find_b (b : ℝ) (y : ℝ) : (4 * 3 + 2 * y = b) ∧ (3 * 3 + 6 * y = 3 * b) → b = 27 :=
by
sorry

end find_b_l121_121310


namespace sum_of_products_l121_121070

theorem sum_of_products {a b c : ℝ}
  (h1 : a ^ 2 + b ^ 2 + c ^ 2 = 138)
  (h2 : a + b + c = 20) :
  a * b + b * c + c * a = 131 := 
by
  sorry

end sum_of_products_l121_121070


namespace initial_bacteria_count_l121_121750

theorem initial_bacteria_count (doubling_interval : ℕ) (initial_count four_minutes_final_count : ℕ)
  (h1 : doubling_interval = 30)
  (h2 : four_minutes_final_count = 524288)
  (h3 : ∀ t : ℕ, initial_count * 2 ^ (t / doubling_interval) = four_minutes_final_count) :
  initial_count = 2048 :=
sorry

end initial_bacteria_count_l121_121750


namespace soybeans_to_oil_l121_121576

theorem soybeans_to_oil 
    (kg_soybeans_to_tofu : ℝ)
    (kg_soybeans_to_oil : ℝ)
    (price_soybeans : ℝ)
    (price_tofu : ℝ)
    (price_oil : ℝ)
    (purchase_amount : ℝ)
    (sales_amount : ℝ)
    (amount_to_oil : ℝ)
    (used_soybeans_for_oil : ℝ) :
    kg_soybeans_to_tofu = 3 →
    kg_soybeans_to_oil = 6 →
    price_soybeans = 2 →
    price_tofu = 3 →
    price_oil = 15 →
    purchase_amount = 920 →
    sales_amount = 1800 →
    used_soybeans_for_oil = 360 →
    (6 * amount_to_oil) = 360 →
    15 * amount_to_oil + 3 * (460 - 6 * amount_to_oil) = 1800 :=
by sorry

end soybeans_to_oil_l121_121576


namespace nick_paths_from_origin_to_16_16_odd_direction_changes_l121_121011

theorem nick_paths_from_origin_to_16_16_odd_direction_changes :
  let total_paths := 2 * Nat.choose 30 15 in
  ∃ f : (Nat × Nat) → (List (Nat × Nat)), 
    (f (0, 0)).length = 32 ∧ 
    (∀ i, f (0, 0).nth i ≠ none → 
        (f (0, 0).nth i = some (f (0, 0).nth (i+1)).get_or_else (0, 0) ∧ 
        ∃ n, odd n ∧ n = (List.attach (f (0, 0)).filter (λ p, 
           (p.1.snd = 1 ∧ p.2.snd = 0) ∨ (p.1.snd = 0 ∧ p.2.snd = 1)).length)) :=
sorry

end nick_paths_from_origin_to_16_16_odd_direction_changes_l121_121011


namespace diff_of_squares_l121_121233

theorem diff_of_squares : (1001^2 - 999^2 = 4000) :=
by
  sorry

end diff_of_squares_l121_121233


namespace equation_solutions_count_l121_121350

theorem equation_solutions_count (n : ℕ) :
  (∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 2 * x + 3 * y + z + x^2 = n) →
  (n = 32 ∨ n = 33) :=
sorry

end equation_solutions_count_l121_121350


namespace distinct_solutions_difference_l121_121191

theorem distinct_solutions_difference (r s : ℝ) (h1 : r ≠ s) (h2 : r > s)
  (h_eq : ∀ x : ℝ, (x - 5) * (x + 5) = 25 * x - 125 ↔ x = r ∨ x = s) : r - s = 15 :=
begin
  sorry
end

end distinct_solutions_difference_l121_121191


namespace number_of_possible_plans_most_cost_effective_plan_l121_121938

-- Defining the conditions of the problem
def price_A := 12 -- Price of model A in million yuan
def price_B := 10 -- Price of model B in million yuan
def capacity_A := 240 -- Treatment capacity of model A in tons/month
def capacity_B := 200 -- Treatment capacity of model B in tons/month
def total_budget := 105 -- Total budget in million yuan
def min_treatment_volume := 2040 -- Minimum required treatment volume in tons/month
def total_units := 10 -- Total number of units to be purchased

def valid_purchase_plan (x y : ℕ) :=
  x + y = total_units ∧
  price_A * x + price_B * y ≤ total_budget ∧
  capacity_A * x + capacity_B * y ≥ min_treatment_volume

-- Stating the theorem for how many possible purchase plans exist
theorem number_of_possible_plans : 
  ∃ k : ℕ, k = 3 ∧
    (∀ (x y : ℕ), 
      valid_purchase_plan x y →
      x ∈ {0, 1, 2} ∧ y = total_units - x) :=
sorry

-- Stating the theorem for the most cost-effective plan
theorem most_cost_effective_plan :
  ∃ (x y : ℕ),
    valid_purchase_plan x y ∧
    price_A * x + price_B * y = 102 ∧
    x = 1 ∧ y = 9 :=
sorry

end number_of_possible_plans_most_cost_effective_plan_l121_121938


namespace cubic_polynomials_l121_121219

theorem cubic_polynomials (a b c a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
    (h1 : a - 1/b = r₁ ∧ b - 1/c = r₂ ∧ c - 1/a = r₃)
    (h2 : r₁ + r₂ + r₃ = 5)
    (h3 : r₁ * r₂ + r₂ * r₃ + r₃ * r₁ = -15)
    (h4 : r₁ * r₂ * r₃ = -3)
    (h5 : a₁ * b₁ * c₁ = 1 + Real.sqrt 2 ∨ a₁ * b₁ * c₁ = 1 - Real.sqrt 2)
    (h6 : a₂ * b₂ * c₂ = 1 + Real.sqrt 2 ∨ a₂ * b₂ * c₂ = 1 - Real.sqrt 2) :
    (-(a₁ * b₁ * c₁))^3 + (-(a₂ * b₂ * c₂))^3 = -14 := sorry

end cubic_polynomials_l121_121219


namespace union_A_B_intersection_complements_l121_121039
open Set

noncomputable def A : Set ℤ := {x | x^2 - 5 * x + 6 = 0}
noncomputable def B : Set ℤ := {x | x^2 - x - 2 = 0}
def U : Set ℤ := {x | abs x ≤ 3}

theorem union_A_B :
  A ∪ B = { -1, 2, 3 } :=
by sorry

theorem intersection_complements :
  (U \ A) ∩ (U \ B) = { -3, -2, 0, 1 } :=
by sorry

end union_A_B_intersection_complements_l121_121039


namespace smallest_portion_is_2_l121_121906

theorem smallest_portion_is_2 (a d : ℝ) (h1 : 5 * a = 120) (h2 : 3 * a + 3 * d = 7 * (2 * a - 3 * d)) : a - 2 * d = 2 :=
by sorry

end smallest_portion_is_2_l121_121906


namespace wheel_radius_increase_l121_121614

theorem wheel_radius_increase 
  (d₁ d₂ : ℝ) -- distances according to the odometer (600 and 580 miles)
  (r₀ : ℝ)   -- original radius (17 inches)
  (C₁: d₁ = 600)
  (C₂: d₂ = 580)
  (C₃: r₀ = 17) :
  ∃ Δr : ℝ, Δr = 0.57 :=
by
  sorry

end wheel_radius_increase_l121_121614


namespace remainder_of_sum_l121_121943

open Nat

theorem remainder_of_sum :
  (12345 + 12347 + 12349 + 12351 + 12353 + 12355 + 12357) % 16 = 9 :=
by 
  sorry

end remainder_of_sum_l121_121943


namespace union_A_B_interval_l121_121147

def setA (x : ℝ) : Prop := x ≥ -1
def setB (y : ℝ) : Prop := y ≥ 1

theorem union_A_B_interval :
  {x | setA x} ∪ {y | setB y} = {z : ℝ | z ≥ -1} :=
by
  sorry

end union_A_B_interval_l121_121147


namespace badges_exchange_l121_121559

theorem badges_exchange (Vasya_initial Tolya_initial : ℕ) 
    (h1 : Vasya_initial = Tolya_initial + 5)
    (h2 : Vasya_initial - 0.24 * Vasya_initial + 0.20 * Tolya_initial = Tolya_initial - 0.20 * Tolya_initial + 0.24 * Vasya_initial - 1) 
    : Vasya_initial = 50 ∧ Tolya_initial = 45 :=
by sorry

end badges_exchange_l121_121559


namespace profit_function_simplified_maximize_profit_l121_121796

-- Define the given conditions
def cost_per_product : ℝ := 3
def management_fee_per_product : ℝ := 3
def annual_sales_volume (x : ℝ) : ℝ := (12 - x) ^ 2

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - (cost_per_product + management_fee_per_product)) * annual_sales_volume x

-- Define the bounds for x
def x_bounds (x : ℝ) : Prop := 9 ≤ x ∧ x ≤ 11

-- Prove the profit function in simplified form
theorem profit_function_simplified (x : ℝ) (h : x_bounds x) :
    profit x = x ^ 3 - 30 * x ^ 2 + 288 * x - 864 :=
by
  sorry

-- Prove the maximum profit and the corresponding x value
theorem maximize_profit (x : ℝ) (h : x_bounds x) :
    (∀ y, (∃ x', x_bounds x' ∧ y = profit x') → y ≤ 27) ∧ profit 9 = 27 :=
by
  sorry

end profit_function_simplified_maximize_profit_l121_121796


namespace imaginary_part_of_z_l121_121041

open Complex

theorem imaginary_part_of_z (x y : ℝ) (h : (1 + I) * x + (1 - I) * y = 2) : 
  im ((x + I) / (y - I)) = 1 :=
by
  sorry

end imaginary_part_of_z_l121_121041


namespace volume_is_correct_l121_121456

def condition1 (x y z : ℝ) : Prop := abs (x + 2 * y + 3 * z) + abs (x + 2 * y - 3 * z) ≤ 18
def condition2 (x y z : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0
def region (x y z : ℝ) : Prop := condition1 x y z ∧ condition2 x y z

noncomputable def volume_of_region : ℝ :=
  60.75 -- the result obtained from the calculation steps

theorem volume_is_correct : ∀ (x y z : ℝ), region x y z → volume_of_region = 60.75 :=
by
  sorry

end volume_is_correct_l121_121456


namespace facemasks_per_box_l121_121313

theorem facemasks_per_box (x : ℝ) :
  (3 * x * 0.50) - 15 = 15 → x = 20 :=
by
  intros h
  sorry

end facemasks_per_box_l121_121313


namespace part_a_l121_121316

def f_X (X : Set (ℝ × ℝ)) (n : ℕ) : ℝ :=
  sorry  -- Placeholder for the largest possible area function

theorem part_a (X : Set (ℝ × ℝ)) (m n : ℕ) (h1 : m ≥ n) (h2 : n > 2) :
  f_X X m + f_X X n ≥ f_X X (m + 1) + f_X X (n - 1) :=
sorry

end part_a_l121_121316


namespace minimum_positive_temperature_announcement_l121_121118

-- Problem conditions translated into Lean
def num_interactions (x : ℕ) : ℕ := x * (x - 1)
def total_interactions := 132
def total_positive := 78
def total_negative := 54
def positive_temperature_count (x y : ℕ) : ℕ := y * (y - 1)
def negative_temperature_count (x y : ℕ) : ℕ := (x - y) * (x - 1 - y)
def minimum_positive_temperature (x y : ℕ) := 
  x = 12 → 
  total_interactions = total_positive + total_negative →
  total_positive + total_negative = num_interactions x →
  total_positive = positive_temperature_count x y →
  sorry -- proof goes here

theorem minimum_positive_temperature_announcement : ∃ y, 
  minimum_positive_temperature 12 y ∧ y = 3 :=
by {
  sorry -- proof goes here
}

end minimum_positive_temperature_announcement_l121_121118


namespace baking_trays_used_l121_121436

-- Let T be the number of baking trays Anna used.
variable (T : ℕ)

-- Condition: Each tray has 20 cupcakes.
def cupcakes_per_tray : ℕ := 20

-- Condition: Each cupcake was sold for $2.
def cupcake_price : ℕ := 2

-- Condition: Only 3/5 of the cupcakes were sold.
def fraction_sold : ℚ := 3 / 5

-- Condition: Anna earned $96 from sold cupcakes.
def earnings : ℕ := 96

-- Derived expressions:
def total_cupcakes (T : ℕ) : ℕ := cupcakes_per_tray * T

def sold_cupcakes (T : ℕ) : ℚ := fraction_sold * total_cupcakes T

def total_earnings (T : ℕ) : ℚ := cupcake_price * sold_cupcakes T

-- The statement to be proved: Given the conditions, the number of trays T must be 4.
theorem baking_trays_used (h : total_earnings T = earnings) : T = 4 := by
  sorry

end baking_trays_used_l121_121436


namespace sine_triangle_sides_l121_121871

variable {α β γ : ℝ}

-- Given conditions: α, β, γ are angles of a triangle.
def is_triangle_angles (α β γ : ℝ) : Prop :=
  α + β + γ = Real.pi ∧ 0 < α ∧ α < Real.pi ∧
  0 < β ∧ β < Real.pi ∧ 0 < γ ∧ γ < Real.pi

-- The proof statement: Prove that there exists a triangle with sides sin α, sin β, sin γ
theorem sine_triangle_sides (h : is_triangle_angles α β γ) :
  ∃ (x y z : ℝ), x = Real.sin α ∧ y = Real.sin β ∧ z = Real.sin γ ∧
  (x + y > z) ∧ (x + z > y) ∧ (y + z > x) := sorry

end sine_triangle_sides_l121_121871


namespace double_angle_cosine_l121_121679

theorem double_angle_cosine (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end double_angle_cosine_l121_121679


namespace time_to_pass_pole_l121_121110

def train_length : ℕ := 250
def platform_length : ℕ := 1250
def time_to_pass_platform : ℕ := 60

theorem time_to_pass_pole : 
  (train_length + platform_length) / time_to_pass_platform * train_length = 10 :=
by
  sorry

end time_to_pass_pole_l121_121110


namespace cos_double_angle_l121_121694

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3/5) : cos (2*θ) = -7/25 := 
by 
  sorry

end cos_double_angle_l121_121694


namespace anna_reading_time_l121_121269

theorem anna_reading_time:
  (∀ n : ℕ, n ∈ (Finset.range 31).filter (λ x, ¬ ∃ k : ℕ, k * 3 + 3 = x + 1) → True) →
  (let chapters_read := (Finset.range 31).filter (λ x, ¬ (∃ k : ℕ, k * 3 + 3 = x + 1)).card,
  reading_time := chapters_read * 20,
  hours := reading_time / 60 in
  hours = 7) :=
by
  intros
  let chapters_read := (Finset.range 31).filter (λ x, ¬ ∃ k : ℕ, k * 3 + 3 = x + 1).card
  have h1 : chapters_read = 21 := by sorry
  let reading_time := chapters_read * 20
  have h2 : reading_time = 420 := by sorry
  let hours := reading_time / 60
  have h3 : hours = 7 := by sorry
  exact h3

end anna_reading_time_l121_121269


namespace tangent_line_eq_l121_121428

noncomputable def equation_of_tangent_line (x y : ℝ) : Prop := 
  ∃ k : ℝ, (y = k * (x - 2) + 2) ∧ 2 * x + y - 6 = 0

theorem tangent_line_eq :
  ∀ (x y : ℝ), 
    (y = 2 / (x - 1)) ∧ (∃ (a b : ℝ), (a, b) = (1, 4)) ->
    equation_of_tangent_line x y :=
by
  sorry

end tangent_line_eq_l121_121428


namespace sum_even_integers_102_to_200_l121_121376

theorem sum_even_integers_102_to_200 : 
  (∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 1 102), 2 * k) = 2550 →
  (∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 100), 2 * k) + 1250 = 2550 → 
  (∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 100) + ∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 100), 2 * k) = 2550 → 
  (∑ k in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 100), 2 * k) = 1300 → 
  (∑ i in finset.filter (λ x, x % 2 = 0) (finset.Icc 51 200), 2 * i) = 1250 :=
begin
  sorry
end

end sum_even_integers_102_to_200_l121_121376


namespace alpha_nonneg_integer_l121_121788

theorem alpha_nonneg_integer (α : ℝ) 
  (h : ∀ n : ℕ, ∃ k : ℕ, n = k * α) : α ≥ 0 ∧ ∃ k : ℤ, α = k := 
sorry

end alpha_nonneg_integer_l121_121788


namespace weight_of_replaced_student_l121_121910

-- Define the conditions as hypotheses
variable (W : ℝ)
variable (h : W - 46 = 40)

-- Prove that W = 86
theorem weight_of_replaced_student : W = 86 :=
by
  -- We should conclude the proof; for now, we leave a placeholder
  sorry

end weight_of_replaced_student_l121_121910


namespace circle_chairs_subsets_count_l121_121397

theorem circle_chairs_subsets_count :
  ∃ (n : ℕ), n = 12 → set.count (λ s : finset ℕ, s.card ≥ 4 ∧ ∀ i ∈ s, (i + 1) % 12 ∈ s) {s | s ⊆ finset.range 12} = 1712 := 
by
  sorry

end circle_chairs_subsets_count_l121_121397


namespace integer_solution_exists_l121_121825

theorem integer_solution_exists (a : ℤ) : 
  (∃ k : ℤ, 2 * a^2 = 7 * k + 2) ↔ (a % 7 = 1 ∨ a % 7 = 6) :=
by sorry

end integer_solution_exists_l121_121825


namespace price_of_books_sold_at_lower_price_l121_121808

-- Define the conditions
variable (n m p q t : ℕ) (earnings price_high price_low : ℝ)

-- The given conditions
def total_books : ℕ := 10
def books_high_price : ℕ := 2 * total_books / 5 -- 2/5 of total books
def books_low_price : ℕ := total_books - books_high_price
def high_price : ℝ := 2.50
def total_earnings : ℝ := 22

-- The proposition to prove
theorem price_of_books_sold_at_lower_price
  (h_books_high_price : books_high_price = 4)
  (h_books_low_price : books_low_price = 6)
  (h_total_earnings : total_earnings = 22)
  (h_high_price : high_price = 2.50) :
  (price_low = 2) := 
-- Proof goes here 
sorry

end price_of_books_sold_at_lower_price_l121_121808


namespace total_walnut_trees_in_park_l121_121543

-- Define initial number of walnut trees in the park
def initial_walnut_trees : ℕ := 22

-- Define number of walnut trees planted by workers
def planted_walnut_trees : ℕ := 33

-- Prove the total number of walnut trees in the park
theorem total_walnut_trees_in_park : initial_walnut_trees + planted_walnut_trees = 55 := by
  sorry

end total_walnut_trees_in_park_l121_121543


namespace student_ticket_count_l121_121828

theorem student_ticket_count (S N : ℕ) (h1 : S + N = 821) (h2 : 2 * S + 3 * N = 1933) : S = 530 :=
sorry

end student_ticket_count_l121_121828


namespace rectangle_cut_into_square_l121_121594

theorem rectangle_cut_into_square (a b : ℝ) (h : a ≤ 4 * b) : 4 * b ≥ a := 
by 
  exact h

end rectangle_cut_into_square_l121_121594


namespace number_of_subsets_with_at_least_four_adjacent_chairs_l121_121388

theorem number_of_subsets_with_at_least_four_adjacent_chairs : 
  ∀ (n : ℕ), n = 12 → 
  (∃ (s : Finset (Finset (Fin n))), s.card = 1610 ∧ 
  (∀ (A : Finset (Fin n)), A ∈ s → (∃ (start : Fin n), ∀ i, i ∈ Finset.range 4 → A.contains (start + i % n)))) :=
by
  sorry

end number_of_subsets_with_at_least_four_adjacent_chairs_l121_121388


namespace latest_start_time_l121_121867

-- Define the times for each activity
def homework_time : ℕ := 30
def clean_room_time : ℕ := 30
def take_out_trash_time : ℕ := 5
def empty_dishwasher_time : ℕ := 10
def dinner_time : ℕ := 45

-- Define the total time required to finish everything in minutes
def total_time_needed : ℕ := homework_time + clean_room_time + take_out_trash_time + empty_dishwasher_time + dinner_time

-- Define the equivalent time in hours
def total_time_needed_hours : ℕ := total_time_needed / 60

-- Define movie start time and the time Justin gets home
def movie_start_time : ℕ := 20 -- (8 PM in 24-hour format)
def justin_home_time : ℕ := 17 -- (5 PM in 24-hour format)

-- Prove the latest time Justin can start his chores and homework
theorem latest_start_time : movie_start_time - total_time_needed_hours = 18 := by
  sorry

end latest_start_time_l121_121867


namespace selection_methods_count_l121_121461

theorem selection_methods_count
  (multiple_choice_questions : ℕ)
  (fill_in_the_blank_questions : ℕ)
  (h1 : multiple_choice_questions = 9)
  (h2 : fill_in_the_blank_questions = 3) :
  multiple_choice_questions + fill_in_the_blank_questions = 12 := by
  sorry

end selection_methods_count_l121_121461


namespace smallest_nine_ten_eleven_consecutive_sum_l121_121085

theorem smallest_nine_ten_eleven_consecutive_sum :
  ∃ n : ℕ, n > 0 ∧ (n % 9 = 0) ∧ (n % 10 = 5) ∧ (n % 11 = 0) ∧ n = 495 :=
by {
  sorry
}

end smallest_nine_ten_eleven_consecutive_sum_l121_121085


namespace cosine_double_angle_l121_121682

theorem cosine_double_angle
  (θ : ℝ)
  (cos_θ : ℝ)
  (h : cos_θ = 3 / 5) :
  2 * cos_θ^2 - 1 = -7 / 25 :=
by
  sorry

end cosine_double_angle_l121_121682


namespace lives_per_player_l121_121221

theorem lives_per_player (initial_players : ℕ) (additional_players : ℕ) (total_lives : ℕ) 
  (h1 : initial_players = 8) (h2 : additional_players = 2) (h3 : total_lives = 60) : 
  total_lives / (initial_players + additional_players) = 6 :=
by 
  sorry

end lives_per_player_l121_121221


namespace cos_double_angle_l121_121657

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 := sorry

end cos_double_angle_l121_121657


namespace part1_part2_l121_121842

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.exp (2 * x) - a * Real.exp x - x * Real.exp x

theorem part1 :
  (∀ x : ℝ, f a x ≥ 0) → a = 1 := sorry

theorem part2 (h : a = 1) :
  ∃ x₀ : ℝ, (∀ x : ℝ, f a x ≤ f a x₀) ∧
    (Real.log 2 / (2 * Real.exp 1) + 1 / (4 * Real.exp (2 * 1)) ≤ f a x₀ ∧
    f a x₀ < 1 / 4) := sorry

end part1_part2_l121_121842


namespace cos_double_angle_l121_121655

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3 / 5) : cos (2 * θ) = -7 / 25 := 
by
  sorry

end cos_double_angle_l121_121655


namespace no_valid_a_exists_l121_121076

theorem no_valid_a_exists (a : ℕ) (n : ℕ) (h1 : a > 1) (b := a * (10^n + 1)) :
  ¬ (∃ a : ℕ, b % (a^2) = 0) :=
by {
  sorry -- The actual proof is not required as per instructions.
}

end no_valid_a_exists_l121_121076


namespace merchant_installed_zucchini_l121_121969

theorem merchant_installed_zucchini (Z : ℕ) : 
  (15 + Z + 8) / 2 = 18 → Z = 13 :=
by
 sorry

end merchant_installed_zucchini_l121_121969


namespace scientific_notation_216000_l121_121747

theorem scientific_notation_216000 : 216000 = 2.16 * 10^5 :=
by
  -- proof will be provided here
  sorry

end scientific_notation_216000_l121_121747


namespace cos_double_angle_l121_121691

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3/5) : Real.cos (2 * theta) = -7/25 :=
by
  sorry

end cos_double_angle_l121_121691


namespace sum_of_coordinates_of_D_l121_121831

-- Definition of points M, C and D
structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨4, 7⟩
def C : Point := ⟨6, 2⟩

-- Conditions that M is the midpoint of segment CD
def isMidpoint (M C D : Point) : Prop :=
  ((C.x + D.x) / 2 = M.x) ∧
  ((C.y + D.y) / 2 = M.y)

-- Definition for the sum of the coordinates of a point
def sumOfCoordinates (P : Point) : ℝ :=
  P.x + P.y

-- The main theorem stating the sum of the coordinates of D is 14 given the conditions
theorem sum_of_coordinates_of_D :
  ∃ D : Point, isMidpoint M C D ∧ sumOfCoordinates D = 14 := 
sorry

end sum_of_coordinates_of_D_l121_121831


namespace sin_18_cos_36_eq_quarter_l121_121412

theorem sin_18_cos_36_eq_quarter : Real.sin (Real.pi / 10) * Real.cos (Real.pi / 5) = 1 / 4 :=
by
  sorry

end sin_18_cos_36_eq_quarter_l121_121412


namespace committee_count_is_252_l121_121326

/-- Definition of binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Problem statement: Number of ways to choose a 5-person committee from a club of 10 people is 252 -/
theorem committee_count_is_252 : binom 10 5 = 252 :=
by
  sorry

end committee_count_is_252_l121_121326


namespace quadratic_roots_l121_121217

theorem quadratic_roots (k : ℝ) :
  (∃ x : ℝ, x = 2 ∧ 4 * x ^ 2 - k * x + 6 = 0) →
  k = 11 ∧ (∃ x : ℝ, x ≠ 2 ∧ 4 * x ^ 2 - 11 * x + 6 = 0 ∧ x = 3 / 4) := 
by
  sorry

end quadratic_roots_l121_121217


namespace overall_percent_change_l121_121590

theorem overall_percent_change (W : ℝ) : 
  (W * 0.6 * 1.3 * 0.8 * 1.1) / W = 0.624 :=
by {
  sorry
}

end overall_percent_change_l121_121590


namespace sin_160_eq_sin_20_l121_121250

theorem sin_160_eq_sin_20 : Real.sin (160 * Real.pi / 180) = Real.sin (20 * Real.pi / 180) :=
by
  sorry

end sin_160_eq_sin_20_l121_121250


namespace vehicle_wax_initial_amount_l121_121502

theorem vehicle_wax_initial_amount
  (wax_car wax_suv wax_spilled wax_left original_amount : ℕ)
  (h_wax_car : wax_car = 3)
  (h_wax_suv : wax_suv = 4)
  (h_wax_spilled : wax_spilled = 2)
  (h_wax_left : wax_left = 2)
  (h_total_wax_used : wax_car + wax_suv = 7)
  (h_wax_before_waxing : wax_car + wax_suv + wax_spilled = 9) :
  original_amount = 11 := by
  sorry

end vehicle_wax_initial_amount_l121_121502


namespace gcd_result_is_two_l121_121726

theorem gcd_result_is_two
  (n m k j: ℕ) (hn : n > 0) (hm : m > 0) (hk : k > 0) (hj : j > 0) :
  Nat.gcd (Nat.gcd (16 * n) (20 * m)) (Nat.gcd (18 * k) (24 * j)) = 2 := 
by
  sorry

end gcd_result_is_two_l121_121726


namespace clock_hands_coincide_22_times_clock_hands_straight_angle_24_times_clock_hands_right_angle_48_times_l121_121647

theorem clock_hands_coincide_22_times
  (minute_hand_cycles_24_hours : ℕ := 24)
  (hour_hand_cycles_24_hours : ℕ := 2)
  (minute_hand_overtakes_hour_hand_per_12_hours : ℕ := 11) :
  2 * minute_hand_overtakes_hour_hand_per_12_hours = 22 :=
by
  -- Proof should be filled here
  sorry

theorem clock_hands_straight_angle_24_times
  (hours_in_day : ℕ := 24)
  (straight_angle_per_hour : ℕ := 1) :
  hours_in_day * straight_angle_per_hour = 24 :=
by
  -- Proof should be filled here
  sorry

theorem clock_hands_right_angle_48_times
  (hours_in_day : ℕ := 24)
  (right_angles_per_hour : ℕ := 2) :
  hours_in_day * right_angles_per_hour = 48 :=
by
  -- Proof should be filled here
  sorry

end clock_hands_coincide_22_times_clock_hands_straight_angle_24_times_clock_hands_right_angle_48_times_l121_121647


namespace determine_q_l121_121986

-- Define the polynomial p(x) and its square
def p (x : ℝ) : ℝ := x^2 + x + 1
def p_squared (x : ℝ) : ℝ := (x^2 + x + 1)^2

-- Define the identity condition
def identity_condition (x : ℝ) (q : ℝ → ℝ) : Prop := 
  p_squared x - 2 * p x * q x + (q x)^2 - 4 * p x + 3 * q x + 3 = 0

-- Ellaboration on the required solution
def correct_q (q : ℝ → ℝ) : Prop :=
  (∀ x, q x = x^2 + 2 * x) ∨ (∀ x, q x = x^2 - 1)

-- The theorem statement
theorem determine_q :
  ∀ q : ℝ → ℝ, (∀ x : ℝ, identity_condition x q) → correct_q q :=
by
  intros
  sorry

end determine_q_l121_121986


namespace inequality_proof_l121_121058

theorem inequality_proof (a b c : ℝ)
  (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |c * x^2 + b * x + a| ≤ 2 :=
by
  sorry

end inequality_proof_l121_121058


namespace find_n_l121_121126

theorem find_n (a : ℕ → ℝ) (d : ℝ) (n : ℕ) (S_odd : ℝ) (S_even : ℝ)
  (h1 : ∀ k, a (2 * k - 1) = a 0 + (2 * k - 2) * d)
  (h2 : ∀ k, a (2 * k) = a 1 + (2 * k - 1) * d)
  (h3 : 2 * n + 1 = n + (n + 1))
  (h4 : S_odd = (n + 1) * (a 0 + n * d))
  (h5 : S_even = n * (a 1 + (n - 1) * d))
  (h6 : S_odd = 4)
  (h7 : S_even = 3) : n = 3 :=
by
  sorry

end find_n_l121_121126


namespace negation_of_proposition_l121_121212

theorem negation_of_proposition :
  (¬ ∀ x > 0, x^2 + x ≥ 0) ↔ (∃ x > 0, x^2 + x < 0) :=
by 
  sorry

end negation_of_proposition_l121_121212


namespace total_students_at_year_end_l121_121028

def initial_students : ℝ := 10.0
def added_students : ℝ := 4.0
def new_students : ℝ := 42.0

theorem total_students_at_year_end : initial_students + added_students + new_students = 56.0 :=
by
  sorry

end total_students_at_year_end_l121_121028


namespace wage_percent_change_l121_121591

-- Definitions based on given conditions
def initial_wage (W : ℝ) := W
def first_decrease (W : ℝ) := 0.60 * W
def first_increase (W : ℝ) := 0.78 * W
def second_decrease (W : ℝ) := 0.624 * W
def second_increase (W : ℝ) := 0.6864 * W

-- Lean theorem statement to prove overall percent change
theorem wage_percent_change : ∀ (W : ℝ), 
  ((second_increase (second_decrease (first_increase (first_decrease W))) - initial_wage W) / initial_wage W) * 100 = -31.36 :=
by sorry

end wage_percent_change_l121_121591


namespace nero_speed_l121_121504

theorem nero_speed (jerome_speed : ℝ) (jerome_time : ℝ) (nero_time : ℝ) :
  jerome_speed = 4 → jerome_time = 6 → nero_time = 3 → 
  ∃ nero_speed : ℝ, nero_speed = 8 :=
by
  intros h1 h2 h3
  use jerome_speed * jerome_time / nero_time
  rw [h1, h2, h3]
  norm_num
  sorry

end nero_speed_l121_121504


namespace books_selection_l121_121484

theorem books_selection 
  (num_mystery : ℕ)
  (num_fantasy : ℕ)
  (num_biographies : ℕ)
  (Hmystery : num_mystery = 5)
  (Hfantasy : num_fantasy = 4)
  (Hbiographies : num_biographies = 6) :
  (num_mystery * num_fantasy * num_biographies = 120) :=
by
  -- Proof goes here
  sorry

end books_selection_l121_121484


namespace percentage_enclosed_by_hexagons_is_50_l121_121257

noncomputable def hexagon_area (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

noncomputable def square_area (s : ℝ) : ℝ :=
  s^2

noncomputable def total_tiling_unit_area (s : ℝ) : ℝ :=
  hexagon_area s + 3 * square_area s

noncomputable def percentage_enclosed_by_hexagons (s : ℝ) : ℝ :=
  (hexagon_area s / total_tiling_unit_area s) * 100

theorem percentage_enclosed_by_hexagons_is_50 (s : ℝ) : percentage_enclosed_by_hexagons s = 50 := by
  sorry

end percentage_enclosed_by_hexagons_is_50_l121_121257


namespace daughters_dress_probability_l121_121408

theorem daughters_dress_probability : (2 / 6 : ℚ) = 1 / 3 := 
        by
        calc
        (2 / 6 : ℚ) = 1 / 3 : by squeeze_simp
        -- here, squeeze_simp or norm_num would help eliminate the fraction.
        
#check daughters_dress_probability -- This should check type correctness

end daughters_dress_probability_l121_121408


namespace total_travel_time_l121_121046

-- We define the conditions as assumptions in the Lean 4 statement.

def riding_rate := 10  -- miles per hour
def time_first_part_minutes := 30  -- initial 30 minutes in minutes
def additional_distance_1 := 15  -- another 15 miles
def rest_time_minutes := 30  -- resting for 30 minutes
def remaining_distance := 20  -- remaining distance of 20 miles

theorem total_travel_time : 
    let time_first_part := time_first_part_minutes
    let time_second_part := (additional_distance_1 : ℚ) / riding_rate * 60  -- convert hours to minutes
    let time_third_part := rest_time_minutes
    let time_fourth_part := (remaining_distance : ℚ) / riding_rate * 60  -- convert hours to minutes
    time_first_part + time_second_part + time_third_part + time_fourth_part = 270 :=
by
  sorry

end total_travel_time_l121_121046


namespace seashells_at_end_of_month_l121_121135

-- Given conditions as definitions
def initial_seashells : ℕ := 50
def increase_per_week : ℕ := 20

-- Define function to calculate seashells in the nth week
def seashells_in_week (n : ℕ) : ℕ :=
  initial_seashells + n * increase_per_week

-- Lean statement to prove the number of seashells in the jar at the end of four weeks is 130
theorem seashells_at_end_of_month : seashells_in_week 4 = 130 :=
by
  sorry

end seashells_at_end_of_month_l121_121135


namespace polygon_sides_l121_121768

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 3 * 360)
  (h2 : n ≥ 3) : 
  n = 8 := 
by
  sorry

end polygon_sides_l121_121768


namespace ratio_of_a_to_b_l121_121755

theorem ratio_of_a_to_b (a y b : ℝ) (h1 : a = 0) (h2 : b = 2 * y) : a / b = 0 :=
by
  sorry

end ratio_of_a_to_b_l121_121755


namespace factor_of_lcm_l121_121368

theorem factor_of_lcm (A B hcf : ℕ) (h_gcd : Nat.gcd A B = hcf) (hcf_eq : hcf = 16) (A_eq : A = 224) :
  ∃ X : ℕ, X = 14 := by
  sorry

end factor_of_lcm_l121_121368


namespace percentage_chromium_first_alloy_l121_121329

theorem percentage_chromium_first_alloy 
  (x : ℝ) (w1 w2 : ℝ) (p2 p_new : ℝ) 
  (h1 : w1 = 10) 
  (h2 : w2 = 30) 
  (h3 : p2 = 0.08)
  (h4 : p_new = 0.09):
  ((x / 100) * w1 + p2 * w2) = p_new * (w1 + w2) → x = 12 :=
by
  sorry

end percentage_chromium_first_alloy_l121_121329


namespace sufficient_prime_logarithms_l121_121211

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

-- Statement of the properties of logarithms
axiom log_mul (b x y : ℝ) : log_b b (x * y) = log_b b x + log_b b y
axiom log_div (b x y : ℝ) : log_b b (x / y) = log_b b x - log_b b y
axiom log_pow (b x : ℝ) (n : ℝ) : log_b b (x ^ n) = n * log_b b x

-- Main theorem
theorem sufficient_prime_logarithms (b : ℝ) (hb : 1 < b) :
  (∀ p : ℕ, is_prime p → ∃ Lp : ℝ, log_b b p = Lp) →
  ∀ n : ℕ, n > 0 → ∃ Ln : ℝ, log_b b n = Ln :=
by
  sorry

end sufficient_prime_logarithms_l121_121211


namespace max_area_triang_ABC_l121_121709

noncomputable def max_area_triang (a b c : ℝ) (M : ℝ) (BM : ℝ := 2) (AM : ℝ := c - b) : ℝ :=
if M = (b + c) / 2 then 2 * Real.sqrt 3 else 0

theorem max_area_triang_ABC (a b c : ℝ) (M : ℝ) (BM : ℝ := 2) (AM : ℝ := c - b) (M_midpoint : M = (b + c) / 2) :
  max_area_triang a b c M BM AM = 2 * Real.sqrt 3 :=
by
  sorry

end max_area_triang_ABC_l121_121709


namespace weight_of_B_l121_121791

theorem weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 47) : B = 39 :=
by
  sorry

end weight_of_B_l121_121791


namespace polygon_sides_l121_121766

theorem polygon_sides (n : ℕ) 
  (h1 : (n-2) * 180 = 3 * 360) 
  (h2 : n > 2) : n = 8 :=
sorry

end polygon_sides_l121_121766


namespace balance_balls_l121_121200

variable (G B Y W P : ℝ)

-- Given conditions
def cond1 : 4 * G = 9 * B := sorry
def cond2 : 3 * Y = 8 * B := sorry
def cond3 : 7 * B = 5 * W := sorry
def cond4 : 4 * P = 10 * B := sorry

-- Theorem we need to prove
theorem balance_balls : 5 * G + 3 * Y + 3 * W + P = 26 * B :=
by
  -- skipping the proof
  sorry

end balance_balls_l121_121200


namespace expression_equals_one_l121_121359

variable {R : Type*} [Field R]
variables (x y z : R)

theorem expression_equals_one (h₁ : x ≠ y) (h₂ : x ≠ z) (h₃ : y ≠ z) :
    (x^2 / ((x - y) * (x - z)) + y^2 / ((y - x) * (y - z)) + z^2 / ((z - x) * (z - y))) = 1 :=
by sorry

end expression_equals_one_l121_121359


namespace maximize_profit_l121_121419

def cups_sold (p : ℝ) : ℝ :=
  150 - 4 * p

def revenue (p : ℝ) : ℝ :=
  p * cups_sold p

def cost : ℝ :=
  200

def profit (p : ℝ) : ℝ :=
  revenue p - cost

theorem maximize_profit (p : ℝ) (h : p ≤ 30) : p = 19 → profit p = 1206.25 :=
by
  sorry

end maximize_profit_l121_121419


namespace expression_value_l121_121238

theorem expression_value (a b : ℕ) (ha : a = 45) (hb : b = 15) : (a + b)^2 - (a^2 + b^2) = 1350 := by 
  sorry

end expression_value_l121_121238


namespace Bernardo_wins_probability_l121_121815

/-- Define the set from which Bernardo and Silvia pick numbers -/
def BernardoSet : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def SilviaSet : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- Define the number of ways to pick 3 numbers from each set -/
def choose_Bernardo := (BernardoSet.card.choose 3)
def choose_Silvia := (SilviaSet.card.choose 3)

/-- Probability calculation -/
noncomputable def probability_Bernardo_wins : ℚ := (choose_Bernardo : ℚ) / (choose_Bernardo + choose_Silvia)

theorem Bernardo_wins_probability :
    probability_Bernardo_wins = 7 / 9 :=
sorry

end Bernardo_wins_probability_l121_121815


namespace vasya_tolya_badges_l121_121557

theorem vasya_tolya_badges (x y : ℤ)
    (h1 : y = x + 5) -- Vasya initially had 5 more badges than Tolya
    (h2 : (y - (6 * (y / 25) / 25) + (4 * x) / 25) = (x - (4 * x) / 5 + 6 * (y / 25) / 5 - 1)) : -- equation balancing after exchange
    x = 45 ∧ y = 50 := 
sorry

end vasya_tolya_badges_l121_121557


namespace six_digit_numbers_count_l121_121009

theorem six_digit_numbers_count :
  let n := 6
  let m1 := 3
  let m2 := 2
  let m3 := 1
  (Nat.factorial n) / ((Nat.factorial m1) * (Nat.factorial m2) * (Nat.factorial m3)) = 60 := 
by
  let n := 6
  let m1 := 3
  let m2 := 2
  let m3 := 1
  calc
    (Nat.factorial n) / ((Nat.factorial m1) * (Nat.factorial m2) * (Nat.factorial m3))
      = 720 / (6 * 2 * 1) : by rw [Nat.factorial_six, Nat.factorial_three, Nat.factorial_two, Nat.factorial_one]
  ... = 720 / 12 : by norm_num
  ... = 60 : by norm_num

end six_digit_numbers_count_l121_121009


namespace difference_in_money_in_nickels_l121_121433

-- Define the given conditions
def alice_quarters (p : ℕ) : ℕ := 3 * p + 2
def bob_quarters (p : ℕ) : ℕ := 2 * p + 8

-- Define the difference in their money in nickels
def difference_in_nickels (p : ℕ) : ℕ := 5 * (p - 6)

-- The proof problem statement
theorem difference_in_money_in_nickels (p : ℕ) : 
  (5 * (alice_quarters p - bob_quarters p)) = difference_in_nickels p :=
by 
  sorry

end difference_in_money_in_nickels_l121_121433


namespace number_of_crosswalks_per_intersection_l121_121404

theorem number_of_crosswalks_per_intersection 
  (num_intersections : Nat) 
  (total_lines : Nat) 
  (lines_per_crosswalk : Nat) 
  (h1 : num_intersections = 5) 
  (h2 : total_lines = 400) 
  (h3 : lines_per_crosswalk = 20) :
  (total_lines / lines_per_crosswalk) / num_intersections = 4 :=
by
  -- Proof steps can be inserted here
  sorry

end number_of_crosswalks_per_intersection_l121_121404


namespace no_perfect_square_solution_l121_121512

theorem no_perfect_square_solution (n : ℕ) (x : ℕ) (hx : x < 10^n) :
  ¬ (∀ y, 0 ≤ y ∧ y ≤ 9 → ∃ z : ℤ, ∃ k : ℤ, 10^(n+1) * z + 10 * x + y = k^2) :=
sorry

end no_perfect_square_solution_l121_121512


namespace inequality_1_l121_121205

theorem inequality_1 (x : ℝ) : (x - 2) * (1 - 3 * x) > 2 → 1 < x ∧ x < 4 / 3 :=
by sorry

end inequality_1_l121_121205


namespace min_total_cost_l121_121281

-- Defining the variables involved
variables (x y z : ℝ)
variables (h : ℝ := 1) (V : ℝ := 4)
def base_cost (x y : ℝ) : ℝ := 200 * (x * y)
def side_cost (x y : ℝ) (h : ℝ) : ℝ := 100 * (2 * (x + y)) * h
def total_cost (x y h : ℝ) : ℝ := base_cost x y + side_cost x y h

-- The condition that volume is 4 m^3
theorem min_total_cost : 
  (∀ x y, x * y = V) → 
  ∃ x y, total_cost x y h = 1600 :=
by
  sorry

end min_total_cost_l121_121281


namespace soldiers_height_order_l121_121061

theorem soldiers_height_order {n : ℕ} (a b : Fin n → ℝ) 
  (ha : ∀ i j, i ≤ j → a i ≥ a j) 
  (hb : ∀ i j, i ≤ j → b i ≥ b j) 
  (h : ∀ i, a i ≤ b i) :
  ∀ i, a i ≤ b i :=
  by sorry

end soldiers_height_order_l121_121061


namespace magic_king_total_episodes_l121_121930

theorem magic_king_total_episodes
  (total_seasons : ℕ)
  (first_half_seasons : ℕ)
  (second_half_seasons : ℕ)
  (episodes_first_half : ℕ)
  (episodes_second_half : ℕ)
  (h1 : total_seasons = 10)
  (h2 : first_half_seasons = total_seasons / 2)
  (h3 : second_half_seasons = total_seasons / 2)
  (h4 : episodes_first_half = 20)
  (h5 : episodes_second_half = 25)
  : (first_half_seasons * episodes_first_half + second_half_seasons * episodes_second_half) = 225 :=
by
  sorry

end magic_king_total_episodes_l121_121930


namespace sequence_solution_l121_121478

theorem sequence_solution (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, (2*n - 1) * a (n + 1) = (2*n + 1) * a n) : 
∀ n : ℕ, a n = 2 * n - 1 := 
by
  sorry

end sequence_solution_l121_121478


namespace isaac_journey_time_l121_121049

def travel_time_total (speed : ℝ) (time1 : ℝ) (distance2 : ℝ) (rest_time : ℝ) (distance3 : ℝ) : ℝ :=
  let time2 := distance2 / speed
  let time3 := distance3 / speed
  time1 + time2 * 60 + rest_time + time3 * 60

theorem isaac_journey_time :
  travel_time_total 10 (30 : ℝ) 15 (30 : ℝ) 20 = 270 :=
by
  sorry

end isaac_journey_time_l121_121049


namespace can_combine_fig1_can_combine_fig2_l121_121094

-- Given areas for rectangle partitions
variables (S1 S2 S3 S4 : ℝ)
-- Condition: total area of black rectangles equals total area of white rectangles
variable (h1 : S1 + S2 = S3 + S4)

-- Proof problem for Figure 1
theorem can_combine_fig1 : ∃ A : ℝ, S1 + S2 = A ∧ S3 + S4 = A := by
  sorry

-- Proof problem for Figure 2
theorem can_combine_fig2 : ∃ B : ℝ, S1 + S2 = B ∧ S3 + S4 = B := by
  sorry

end can_combine_fig1_can_combine_fig2_l121_121094


namespace time_difference_halfway_point_l121_121092

theorem time_difference_halfway_point 
  (T_d : ℝ) 
  (T_s : ℝ := 2 * T_d) 
  (H_d : ℝ := T_d / 2) 
  (H_s : ℝ := T_s / 2) 
  (diff_time : ℝ := H_s - H_d) : 
  T_d = 35 →
  T_s = 2 * T_d →
  diff_time = 17.5 :=
by
  intros h1 h2
  sorry

end time_difference_halfway_point_l121_121092


namespace arithmetic_mean_solution_l121_121125

-- Define the Arithmetic Mean statement
theorem arithmetic_mean_solution (x : ℝ) (h : (x + 5 + 17 + 3 * x + 11 + 3 * x + 6) / 5 = 19) : 
  x = 8 :=
by
  sorry -- Proof is not required as per the instructions

end arithmetic_mean_solution_l121_121125


namespace subsets_with_at_least_four_adjacent_chairs_l121_121391

theorem subsets_with_at_least_four_adjacent_chairs :
  let chairs := Finset.range 12 in
  ∑ n in chairs, if n ≥ 4 then (12.choose n) else 0 = 1622 :=
by
  sorry

end subsets_with_at_least_four_adjacent_chairs_l121_121391


namespace find_larger_number_l121_121168

theorem find_larger_number 
  (x y : ℚ) 
  (h1 : 4 * y = 9 * x) 
  (h2 : y - x = 12) : 
  y = 108 / 5 := 
sorry

end find_larger_number_l121_121168


namespace dishonest_dealer_weight_l121_121586

noncomputable def dealer_weight_equiv (cost_price : ℝ) (selling_price : ℝ) (profit_percent : ℝ) : ℝ :=
  (1 - profit_percent / 100) * cost_price / selling_price

theorem dishonest_dealer_weight :
  dealer_weight_equiv 1 2 100 = 0.5 :=
by
  sorry

end dishonest_dealer_weight_l121_121586


namespace length_of_room_l121_121067

theorem length_of_room 
  (width : ℝ) (total_cost : ℝ) (rate_per_sq_meter : ℝ) 
  (h_width : width = 3.75) 
  (h_total_cost : total_cost = 16500) 
  (h_rate_per_sq_meter : rate_per_sq_meter = 800) : 
  ∃ length : ℝ, length = 5.5 :=
by
  sorry

end length_of_room_l121_121067


namespace number_of_red_balls_l121_121920

-- Conditions
variables (w r : ℕ)
variable (ratio_condition : 4 * r = 3 * w)
variable (white_balls : w = 8)

-- Prove the number of red balls
theorem number_of_red_balls : r = 6 :=
by
  sorry

end number_of_red_balls_l121_121920


namespace last_score_is_80_l121_121356

def is_integer (n : ℤ) (d : ℤ) : Prop :=
  d ≠ 0 ∧ ∃ k : ℤ, n = k * d

def average_condition (scores : List ℤ) : Prop :=
  (∀ i in List.range scores.length, is_integer (scores.take i |>.sum) (i + 1))

theorem last_score_is_80 (scores : List ℤ) (h : scores = [71, 76, 80, 82, 91]) :
  average_condition (insert_nth 80 scores) :=
by
  sorry

end last_score_is_80_l121_121356


namespace katrina_cookies_left_l121_121345

theorem katrina_cookies_left (initial_cookies morning_cookies_sold lunch_cookies_sold afternoon_cookies_sold : ℕ)
  (h1 : initial_cookies = 120)
  (h2 : morning_cookies_sold = 36)
  (h3 : lunch_cookies_sold = 57)
  (h4 : afternoon_cookies_sold = 16) :
  initial_cookies - (morning_cookies_sold + lunch_cookies_sold + afternoon_cookies_sold) = 11 := 
by 
  sorry

end katrina_cookies_left_l121_121345


namespace total_trees_planted_total_trees_when_a_100_l121_121224

-- Define the number of trees planted by each team based on 'a'
def trees_first_team (a : ℕ) : ℕ := a
def trees_second_team (a : ℕ) : ℕ := 2 * a + 8
def trees_third_team (a : ℕ) : ℕ := (2 * a + 8) / 2 - 6

-- Define the total number of trees
def total_trees (a : ℕ) : ℕ := 
  trees_first_team a + trees_second_team a + trees_third_team a

-- The main theorem
theorem total_trees_planted (a : ℕ) : total_trees a = 4 * a + 6 :=
by
  sorry

-- The specific calculation when a = 100
theorem total_trees_when_a_100 : total_trees 100 = 406 :=
by
  sorry

end total_trees_planted_total_trees_when_a_100_l121_121224


namespace ababab_divisible_by_13_l121_121088

theorem ababab_divisible_by_13 (a b : ℕ) (ha: a < 10) (hb: b < 10) : 
  13 ∣ (100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b) := 
by
  sorry

end ababab_divisible_by_13_l121_121088


namespace diameter_increase_l121_121955

theorem diameter_increase (A A' D D' : ℝ)
  (hA_increase: A' = 4 * A)
  (hA: A = π * (D / 2)^2)
  (hA': A' = π * (D' / 2)^2) :
  D' = 2 * D :=
by 
  sorry

end diameter_increase_l121_121955


namespace cube_odd_minus_itself_div_by_24_l121_121897

theorem cube_odd_minus_itself_div_by_24 (n : ℤ) : 
  (2 * n + 1)^3 - (2 * n + 1) ≡ 0 [MOD 24] := 
by 
  sorry

end cube_odd_minus_itself_div_by_24_l121_121897


namespace valid_license_plates_count_l121_121430

def validLicensePlates : Nat :=
  26 * 26 * 26 * 10 * 9 * 8

theorem valid_license_plates_count :
  validLicensePlates = 15818400 :=
by
  sorry

end valid_license_plates_count_l121_121430


namespace ellipse_problem_l121_121304

noncomputable def point_coordinates (x y b : ℝ) : Prop :=
  x = 1 ∧ y = 1 ∧ (4 * x^2 = 4) ∧ (4 * b^2 / (4 + b^2) = 1)

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  (Real.sqrt (a^2 - b^2)) / a

theorem ellipse_problem (b : ℝ) (h₁ : 4 * b^2 / (4 + b^2) = 1) :
  ∃ x y, point_coordinates x y b 
  ∧ eccentricity 2 b = Real.sqrt 6 / 3 := 
by 
  sorry

end ellipse_problem_l121_121304


namespace average_net_sales_per_month_l121_121207

def sales_jan : ℕ := 120
def sales_feb : ℕ := 80
def sales_mar : ℕ := 50
def sales_apr : ℕ := 130
def sales_may : ℕ := 90
def sales_jun : ℕ := 160

def monthly_expense : ℕ := 30
def num_months : ℕ := 6

def total_sales := sales_jan + sales_feb + sales_mar + sales_apr + sales_may + sales_jun
def total_expenses := monthly_expense * num_months
def net_total_sales := total_sales - total_expenses

theorem average_net_sales_per_month : net_total_sales / num_months = 75 :=
by {
  -- Lean code for proof here
  sorry
}

end average_net_sales_per_month_l121_121207


namespace calculate_first_worker_time_l121_121265

theorem calculate_first_worker_time
    (T : ℝ)
    (h : 1/T + 1/4 = 1/2.2222222222222223) :
    T = 5 := sorry

end calculate_first_worker_time_l121_121265


namespace rhombus_diagonal_length_l121_121753

theorem rhombus_diagonal_length (d1 d2 : ℝ) (area : ℝ) (h1 : area = 600) (h2 : d1 = 30) :
  d2 = 40 :=
by
  sorry

end rhombus_diagonal_length_l121_121753


namespace cos_double_angle_l121_121689

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3/5) : Real.cos (2 * theta) = -7/25 :=
by
  sorry

end cos_double_angle_l121_121689


namespace sum_of_undefined_values_l121_121945

theorem sum_of_undefined_values (y : ℝ) :
  (y^2 - 7 * y + 12 = 0) → y = 3 ∨ y = 4 → (3 + 4 = 7) :=
by
  intro hy
  intro hy'
  sorry

end sum_of_undefined_values_l121_121945


namespace number_of_intersections_l121_121454

theorem number_of_intersections (x y : ℝ) :
  (x^2 + y^2 = 16) ∧ (x = 4) → (x = 4 ∧ y = 0) :=
by {
  sorry
}

end number_of_intersections_l121_121454


namespace loss_per_metre_l121_121803

def total_metres : ℕ := 500
def selling_price : ℕ := 18000
def cost_price_per_metre : ℕ := 41

theorem loss_per_metre :
  (cost_price_per_metre * total_metres - selling_price) / total_metres = 5 :=
by sorry

end loss_per_metre_l121_121803


namespace max_at_zero_l121_121915

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem max_at_zero : ∃ x, (∀ y, f y ≤ f x) ∧ x = 0 :=
by 
  sorry

end max_at_zero_l121_121915


namespace cheese_cookie_price_l121_121581

theorem cheese_cookie_price
  (boxes_per_carton : ℕ)
  (packs_per_box : ℕ)
  (cost_per_dozen_cartons : ℕ) :
  boxes_per_carton = 12 →
  packs_per_box = 10 →
  cost_per_dozen_cartons = 1440 →
  cost_per_dozen_cartons / (boxes_per_carton * 12 * packs_per_box) = 1 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry -- Proof steps can be filled in, but not required for this task.

end cheese_cookie_price_l121_121581


namespace age_of_new_person_l121_121248

theorem age_of_new_person (T A : ℕ) (h1 : (T / 10 : ℤ) - 3 = (T - 40 + A) / 10) : A = 10 := 
sorry

end age_of_new_person_l121_121248


namespace circle_C2_equation_line_l_equation_l121_121464

-- Proof problem 1: Finding the equation of C2
theorem circle_C2_equation (C1_center_x C1_center_y : ℝ) (A_x A_y : ℝ) 
  (C2_center_x : ℝ) (C1_radius : ℝ) :
  C1_center_x = 6 ∧ C1_center_y = 7 ∧ C1_radius = 5 →
  A_x = 2 ∧ A_y = 4 →
  C2_center_x = 6 →
  (∀ y : ℝ, ((y - C1_center_y = C1_radius + (C1_radius + (y - C1_center_y)))) →
    (x - C2_center_x)^2 + (y - C2_center_y)^2 = 1) :=
sorry

-- Proof problem 2: Finding the equation of the line l
theorem line_l_equation (O_x O_y A_x A_y : ℝ) 
  (C1_center_x C1_center_y : ℝ) 
  (A_BC_dist : ℝ) :
  O_x = 0 ∧ O_y = 0 →
  A_x = 2 ∧ A_y = 4 →
  C1_center_x = 6 ∧ C1_center_y = 7 →
  A_BC_dist = 2 * (25^(1 / 2)) →
  ((2 : ℝ)*x - y + 5 = 0 ∨ (2 : ℝ)*x - y - 15 = 0) :=
sorry

end circle_C2_equation_line_l_equation_l121_121464


namespace molecular_weights_correct_l121_121441

-- Define atomic weights
def atomic_weight_Al : Float := 26.98
def atomic_weight_Cl : Float := 35.45
def atomic_weight_K : Float := 39.10

-- Define molecular weight calculations
def molecular_weight_AlCl3 : Float :=
  atomic_weight_Al + 3 * atomic_weight_Cl

def molecular_weight_KCl : Float :=
  atomic_weight_K + atomic_weight_Cl

-- Theorem statement to prove
theorem molecular_weights_correct :
  molecular_weight_AlCl3 = 133.33 ∧ molecular_weight_KCl = 74.55 :=
by
  -- This is where we would normally prove the equivalence
  sorry

end molecular_weights_correct_l121_121441


namespace tetrahedron_circumscribed_sphere_radius_l121_121324

open Real

theorem tetrahedron_circumscribed_sphere_radius :
  ∀ (A B C D : ℝ × ℝ × ℝ), 
    dist A B = 5 →
    dist C D = 5 →
    dist A C = sqrt 34 →
    dist B D = sqrt 34 →
    dist A D = sqrt 41 →
    dist B C = sqrt 41 →
    ∃ (R : ℝ), R = 5 * sqrt 2 / 2 :=
by
  intros A B C D hAB hCD hAC hBD hAD hBC
  sorry

end tetrahedron_circumscribed_sphere_radius_l121_121324


namespace class_gpa_l121_121534

theorem class_gpa (n : ℕ) (hn : n > 0) (gpa1 : ℝ := 30) (gpa2 : ℝ := 33) : 
    (gpa1 * (n:ℝ) + gpa2 * (2 * n : ℝ)) / (3 * n : ℝ) = 32 :=
by
  sorry

end class_gpa_l121_121534


namespace average_marks_l121_121525

-- Definitions
def tat_score (i_score : ℕ) : ℕ := 2 * i_score
def iva_score (d_score : ℕ) : ℕ := (3 / 5 : ℝ) * d_score

theorem average_marks :
  let D : ℕ := 90
  let I : ℕ := iva_score D
  let T : ℕ := tat_score I
  (D + I + T) / 3 = 84 :=
by {
  -- This is where the proof would go.
  -- Exact math proof steps are omitted as per the instructions.
  sorry
}

end average_marks_l121_121525


namespace geometric_sequence_product_l121_121000

theorem geometric_sequence_product 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * r)
  (h_pos : ∀ n, a n > 0)
  (h_log_sum : Real.log (a 3) + Real.log (a 8) + Real.log (a 13) = 6) :
  a 1 * a 15 = 10000 := 
sorry

end geometric_sequence_product_l121_121000


namespace flight_time_sum_l121_121866

theorem flight_time_sum (h m : ℕ)
  (Hdep : true)   -- Placeholder condition for the departure time being 3:45 PM
  (Hlay : 25 = 25)   -- Placeholder condition for the layover being 25 minutes
  (Harr : true)   -- Placeholder condition for the arrival time being 8:02 PM
  (HsameTZ : true)   -- Placeholder condition for the same time zone
  (H0m : 0 < m) 
  (Hm60 : m < 60)
  (Hfinal_time : (h, m) = (3, 52)) : 
  h + m = 55 := 
by {
  sorry
}

end flight_time_sum_l121_121866


namespace length_on_ninth_day_l121_121331

-- Define relevant variables and conditions.
variables (a1 d : ℕ)

-- Define conditions as hypotheses.
def problem_conditions : Prop :=
  (7 * a1 + 21 * d = 28) ∧ 
  (a1 + d + a1 + 4 * d + a1 + 7 * d = 15)

theorem length_on_ninth_day (h : problem_conditions a1 d) : (a1 + 8 * d = 9) :=
  sorry

end length_on_ninth_day_l121_121331


namespace find_arithmetic_sequence_l121_121936

theorem find_arithmetic_sequence (a d : ℝ) (h1 : (a - d) + a + (a + d) = 12) (h2 : (a - d) * a * (a + d) = 48) :
  (a = 4 ∧ d = 2) ∨ (a = 4 ∧ d = -2) :=
sorry

end find_arithmetic_sequence_l121_121936


namespace printing_company_proportion_l121_121260

theorem printing_company_proportion (x y : ℕ) :
  (28*x + 42*y) / (28*x) = 5/3 → x / y = 9 / 4 := by
  sorry

end printing_company_proportion_l121_121260


namespace equal_powers_eq_a_b_l121_121490

theorem equal_powers_eq_a_b 
  (a b : ℝ) 
  (ha_pos : 0 < a) 
  (hb_pos : 0 < b)
  (h_exp_eq : a^b = b^a)
  (h_a_lt_1 : a < 1) : 
  a = b :=
sorry

end equal_powers_eq_a_b_l121_121490


namespace proof_of_problem_statement_l121_121143

noncomputable def problem_statement : Prop :=
  ∀ (k : ℝ) (m : ℝ),
    (0 < m ∧ m < 3/2) → 
    (-3/(4 * m) = k) → 
    (k < -1/2)

theorem proof_of_problem_statement : problem_statement :=
  sorry

end proof_of_problem_statement_l121_121143


namespace numerical_puzzle_l121_121742

noncomputable def THETA (T : ℕ) (A : ℕ) : ℕ := 1000 * T + 100 * T + 10 * T + A
noncomputable def BETA (B : ℕ) (T : ℕ) (A : ℕ) : ℕ := 1000 * B + 100 * T + 10 * T + A
noncomputable def GAMMA (Γ : ℕ) (E : ℕ) (M : ℕ) (A : ℕ) : ℕ := 10000 * Γ + 1000 * E + 100 * M + 10 * M + A

theorem numerical_puzzle
  (T : ℕ) (B : ℕ) (E : ℕ) (M : ℕ) (Γ : ℕ) (A : ℕ)
  (h1 : A = 0)
  (h2 : Γ = 1)
  (h3 : T + T = M)
  (h4 : 2 * E = M)
  (h5 : T ≠ B)
  (h6 : B ≠ E)
  (h7 : E ≠ M)
  (h8 : M ≠ Γ)
  (h9 : Γ ≠ T)
  (h10 : Γ ≠ B)
  (h11 : THETA T A + BETA B T A = GAMMA Γ E M A) :
  THETA 4 0 + BETA 5 4 0 = GAMMA 1 9 8 0 :=
by {
  sorry
}

end numerical_puzzle_l121_121742


namespace distance_to_center_square_l121_121585

theorem distance_to_center_square (x y : ℝ) (h : x*x + y*y = 72) (h1 : x*x + (y + 8)*(y + 8) = 72) (h2 : (x + 4)*(x + 4) + y*y = 72) :
  x*x + y*y = 9 ∨ x*x + y*y = 185 :=
by
  sorry

end distance_to_center_square_l121_121585


namespace total_sum_of_money_is_71_l121_121218

noncomputable def totalCoins : ℕ := 334
noncomputable def coins20Paise : ℕ := 250
noncomputable def coins25Paise : ℕ := totalCoins - coins20Paise
noncomputable def value20Paise : ℕ := coins20Paise * 20
noncomputable def value25Paise : ℕ := coins25Paise * 25
noncomputable def totalValuePaise : ℕ := value20Paise + value25Paise
noncomputable def totalValueRupees : ℚ := totalValuePaise / 100

theorem total_sum_of_money_is_71 :
  totalValueRupees = 71 := by
  sorry

end total_sum_of_money_is_71_l121_121218


namespace manfred_average_paycheck_l121_121571

def average_paycheck : ℕ → ℕ → ℕ → ℕ := fun total_paychecks first_paychecks_value num_first_paychecks =>
  let remaining_paychecks_value := first_paychecks_value + 20
  let total_payment := (num_first_paychecks * first_paychecks_value) + ((total_paychecks - num_first_paychecks) * remaining_paychecks_value)
  let average_payment := total_payment / total_paychecks
  average_payment

theorem manfred_average_paycheck :
  average_paycheck 26 750 6 = 765 := by
  sorry

end manfred_average_paycheck_l121_121571


namespace rectangle_area_inscribed_circle_l121_121965

theorem rectangle_area_inscribed_circle (r : ℝ) (h : r = 7) (ratio : ℝ) (hratio : ratio = 3) : 
  (2 * r) * (ratio * (2 * r)) = 588 :=
by
  rw [h, hratio]
  sorry

end rectangle_area_inscribed_circle_l121_121965


namespace total_kids_played_l121_121507

def kids_played_week (monday tuesday wednesday thursday: ℕ): ℕ :=
  let friday := thursday + (thursday * 20 / 100)
  let saturday := friday - (friday * 30 / 100)
  let sunday := 2 * monday
  monday + tuesday + wednesday + thursday + friday + saturday + sunday

theorem total_kids_played : 
  kids_played_week 15 18 25 30 = 180 :=
by
  sorry

end total_kids_played_l121_121507


namespace shortest_player_height_l121_121538

-- let h_tall be the height of the tallest player
-- let h_short be the height of the shortest player
-- let diff be the height difference between the tallest and the shortest player

variable (h_tall h_short diff : ℝ)

-- conditions given in the problem
axiom tall_player_height : h_tall = 77.75
axiom height_difference : diff = 9.5
axiom height_relationship : h_tall = h_short + diff

-- the statement we need to prove
theorem shortest_player_height : h_short = 68.25 := by
  sorry

end shortest_player_height_l121_121538


namespace trader_sold_95_pens_l121_121109

theorem trader_sold_95_pens
  (C : ℝ)   -- cost price of one pen
  (N : ℝ)   -- number of pens sold
  (h1 : 19 * C = 0.20 * N * C):  -- condition: profit from selling N pens is equal to the cost of 19 pens, with 20% gain percentage
  N = 95 := by
-- You would place the proof here.
  sorry

end trader_sold_95_pens_l121_121109


namespace simplify_expression_correct_l121_121522

noncomputable def simplify_expression : ℝ :=
  2 * Real.sqrt (3 + Real.sqrt (5 - Real.sqrt (13 + Real.sqrt (48))))

theorem simplify_expression_correct : simplify_expression = (Real.sqrt 6) + (Real.sqrt 2) :=
  sorry

end simplify_expression_correct_l121_121522


namespace password_lock_probability_l121_121423

theorem password_lock_probability :
  (∃ (A B C : ℕ), 0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 
   (∃ password : ℕ × ℕ × ℕ, password = (A, B, C) ∧ 
   (∃ chosen_middle_digit : ℕ, 0 <= chosen_middle_digit ∧ chosen_middle_digit <= 9 ∧
   chosen_middle_digit = B))) →
  (1/10 : ℝ) = 0.1 :=
by 
  sorry

end password_lock_probability_l121_121423


namespace min_value_expression_l121_121874

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 2 / b) * (a + 2 / b - 1010) + (b + 2 / a) * (b + 2 / a - 1010) + 101010 = -404040 :=
sorry

end min_value_expression_l121_121874


namespace total_pieces_in_boxes_l121_121416

theorem total_pieces_in_boxes (num_boxes : ℕ) (pieces_per_box : ℕ) (total_pieces : ℕ)
    (h1 : num_boxes = 6) (h2 : pieces_per_box = 500) :
    num_boxes * pieces_per_box = total_pieces → total_pieces = 3000 :=
by
  intro h
  rw [h1, h2] at h
  rw h
  rfl

end total_pieces_in_boxes_l121_121416


namespace problem_log_inequality_l121_121728

noncomputable def f (x m : ℝ) := x - |x + 2| - |x - 3| - m

theorem problem (m : ℝ) (h1 : ∀ x : ℝ, (1 / m) - 4 ≥ f x m) :
  m > 0 :=
sorry

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_inequality (m : ℝ) (h2 : m > 0) :
  log_base (m + 1) (m + 2) > log_base (m + 2) (m + 3) :=
sorry

end problem_log_inequality_l121_121728


namespace hexagon_theorem_l121_121981

-- Define a structure for the hexagon with its sides
structure Hexagon :=
(side1 side2 side3 side4 side5 side6 : ℕ)

-- Define the conditions of the problem
def hexagon_conditions (h : Hexagon) : Prop :=
  h.side1 = 5 ∧ h.side2 = 6 ∧ h.side3 = 7 ∧
  (h.side1 + h.side2 + h.side3 + h.side4 + h.side5 + h.side6 = 38)

-- Define the proposition that we need to prove
def hexagon_proposition (h : Hexagon) : Prop :=
  (h.side3 = 7 ∨ h.side4 = 7 ∨ h.side5 = 7 ∨ h.side6 = 7) → 
  (h.side1 = 5 ∧ h.side2 = 6 ∧ h.side3 = 7 ∧ h.side4 = 7 ∧ h.side5 = 7 ∧ h.side6 = 7 → 3 = 3)

-- The proof statement combining conditions and the to-be-proven proposition
theorem hexagon_theorem (h : Hexagon) (hc : hexagon_conditions h) : hexagon_proposition h :=
by
  sorry -- No proof is required

end hexagon_theorem_l121_121981


namespace arithmetic_sequence_odd_function_always_positive_l121_121353

theorem arithmetic_sequence_odd_function_always_positive
    (f : ℝ → ℝ) (a : ℕ → ℝ)
    (h_odd : ∀ x, f (-x) = -f x)
    (h_monotone_geq_0 : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)
    (h_arith_seq : ∀ n, a (n + 1) = a n + (a 2 - a 1))
    (h_a3_neg : a 3 < 0) :
    f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) > 0 := by
    sorry

end arithmetic_sequence_odd_function_always_positive_l121_121353


namespace find_natrual_numbers_l121_121822

theorem find_natrual_numbers (k n : ℕ) (A B : Matrix (Fin n) (Fin n) ℤ) 
  (h1 : k ≥ 1) 
  (h2 : n ≥ 2) 
  (h3 : A ^ 3 = 0) 
  (h4 : A ^ k * B + B * A = 1) : 
  k = 1 ∧ Even n := 
sorry

end find_natrual_numbers_l121_121822


namespace ivan_sergeyevich_profit_l121_121505

def revenue_from_meat (meat_sold price_per_kg : ℝ) : ℝ :=
  meat_sold * price_per_kg

def revenue_from_eggs (eggs_sold price_per_dozen : ℝ) : ℝ :=
  eggs_sold * (price_per_dozen / 12)

def total_revenue (meat_revenue egg_revenue : ℝ) : ℝ :=
  meat_revenue + egg_revenue

def profit (total_revenue expenses : ℝ) : ℝ :=
  total_revenue - expenses

-- Given conditions
def meat_sold := 100
def price_per_kg := 500
def eggs_sold := 20000
def price_per_dozen := 50
def expenses := 100000

theorem ivan_sergeyevich_profit : 
  profit (total_revenue (revenue_from_meat meat_sold price_per_kg) (revenue_from_eggs eggs_sold price_per_dozen)) expenses = 50000 :=
by sorry

end ivan_sergeyevich_profit_l121_121505


namespace probability_between_bounds_l121_121002

noncomputable def normal_distribution (μ σ : ℝ) : OrElse := sorry

theorem probability_between_bounds 
  (σ : ℝ) (hσ : 0 < σ)
  (X : OrElse) (hX : X = normal_distribution 0 σ)
  (h : P(X > 2) = 0.023) :
  P(-2 ≤ X ∧ X ≤ 2) = 0.954 :=
sorry

end probability_between_bounds_l121_121002


namespace exists_x0_gt_0_f_x0_lt_0_implies_m_lt_neg_2_l121_121475

variable (m : ℝ)
def f (x : ℝ) : ℝ := x^2 + m*x + 1

theorem exists_x0_gt_0_f_x0_lt_0_implies_m_lt_neg_2 :
  (∃ x0 : ℝ, x0 > 0 ∧ f m x0 < 0) → m < -2 := by
  sorry

end exists_x0_gt_0_f_x0_lt_0_implies_m_lt_neg_2_l121_121475


namespace b_value_l121_121624

theorem b_value (x y b : ℝ) (h1 : x / (2 * y) = 3 / 2) (h2 : (7 * x + b * y) / (x - 2 * y) = 25) : b = 4 := 
by
  sorry

end b_value_l121_121624


namespace probability_at_least_one_l121_121026

-- Conditions
constant pA : ℝ := 0.8
constant pB : ℝ := 0.7

-- The problem we want to prove
theorem probability_at_least_one :
  let qA := 1 - pA in
  let qB := 1 - pB in
  let p_neither := qA * qB in
  1 - p_neither = 0.94 :=
by simp [pA, pB, (1 - pA) * (1 - pB)]

end probability_at_least_one_l121_121026


namespace units_digit_of_27_times_36_l121_121618

theorem units_digit_of_27_times_36 :
  let units_digit := fun (n : ℕ) => n % 10
  in units_digit (27 * 36) = 2 :=
by
  let units_digit := fun (n : ℕ) => n % 10
  have h27: units_digit 27 = 7 := by
    show 27 % 10 = 7
    sorry
  have h36: units_digit 36 = 6 := by
    show 36 % 10 = 6
    sorry
  have h42: units_digit (7 * 6) = 2 := by
    show 42 % 10 = 2
    sorry
  exact h42

end units_digit_of_27_times_36_l121_121618


namespace tissue_actual_diameter_l121_121241

theorem tissue_actual_diameter (magnification_factor : ℝ) (magnified_diameter : ℝ) 
(h1 : magnification_factor = 1000)
(h2 : magnified_diameter = 0.3) : 
  magnified_diameter / magnification_factor = 0.0003 :=
by sorry

end tissue_actual_diameter_l121_121241


namespace janet_percentage_l121_121184

-- Define the number of snowballs made by Janet and her brother
def janet_snowballs : ℕ := 50
def brother_snowballs : ℕ := 150

-- Define the total number of snowballs
def total_snowballs : ℕ := janet_snowballs + brother_snowballs

-- Define the percentage function
def percentage (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- State the theorem to be proved
theorem janet_percentage : percentage janet_snowballs total_snowballs = 25 := by
  sorry

end janet_percentage_l121_121184


namespace pieces_per_plant_yield_l121_121254

theorem pieces_per_plant_yield 
  (rows : ℕ) (plants_per_row : ℕ) (total_harvest : ℕ) 
  (h1 : rows = 30) (h2 : plants_per_row = 10) (h3 : total_harvest = 6000) : 
  (total_harvest / (rows * plants_per_row) = 20) :=
by
  -- Insert math proof here.
  sorry

end pieces_per_plant_yield_l121_121254


namespace max_distance_between_vertices_l121_121805

theorem max_distance_between_vertices (inner_perimeter outer_perimeter : ℕ) 
  (inner_perimeter_eq : inner_perimeter = 20) 
  (outer_perimeter_eq : outer_perimeter = 28) : 
  ∃ x y, x + y = 7 ∧ x^2 + y^2 = 25 ∧ (x^2 + (x + y)^2 = 65) :=
by
  sorry

end max_distance_between_vertices_l121_121805


namespace cos_double_angle_l121_121698

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7/25 :=
by
  sorry

end cos_double_angle_l121_121698


namespace right_triangle_distance_midpoint_l121_121330

noncomputable def distance_from_F_to_midpoint_DE
  (D E F : ℝ × ℝ)
  (right_triangle : ∃ A B C, A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
                    A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧ C.1^2 + C.2^2 = 1 ∧
                    D = A ∧ E = B ∧ F = C) 
  (DE : ℝ)
  (DF : ℝ)
  (EF : ℝ)
  : ℝ :=
  if hD : (D.1 - E.1)^2 + (D.2 - E.2)^2 = DE^2 then
    if hF : (D.1 - F.1)^2 + (D.2 - F.2)^2 = DF^2 then
      if hDE : DE = 15 then
        (15 / 2) --distance from F to midpoint of DE
      else
        0 -- This will never be executed since DE = 15 is a given condition
    else
      0 -- This will never be executed since DF = 9 is a given condition
  else
    0 -- This will never be executed since EF = 12 is a given condition

theorem right_triangle_distance_midpoint
  (D E F : ℝ × ℝ)
  (h_triangle : ∃ A B C, A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
                    A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧ C.1^2 + C.2^2 = 1 ∧
                    D = A ∧ E = B ∧ F = C)
  (hDE : (D.1 - E.1)^2 + (D.2 - E.2)^2 = 15^2)
  (hDF : (D.1 - F.1)^2 + (D.2 - F.2)^2 = 9^2)
  (hEF : (E.1 - F.1)^2 + (E.2 - F.2)^2 = 12^2) :
  distance_from_F_to_midpoint_DE D E F h_triangle 15 9 12 = 7.5 :=
by sorry

end right_triangle_distance_midpoint_l121_121330


namespace solve_quadratic_eq_l121_121762

theorem solve_quadratic_eq (x : ℝ) : x^2 = 4 * x → x = 0 ∨ x = 4 :=
by
  intro h
  sorry

end solve_quadratic_eq_l121_121762


namespace cos_double_angle_l121_121696

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3/5) : cos (2*θ) = -7/25 := 
by 
  sorry

end cos_double_angle_l121_121696


namespace sum_nat_numbers_from_1_to_5_l121_121784

theorem sum_nat_numbers_from_1_to_5 : (1 + 2 + 3 + 4 + 5 = 15) :=
by
  sorry

end sum_nat_numbers_from_1_to_5_l121_121784


namespace min_value_reciprocals_l121_121724

theorem min_value_reciprocals (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h_sum : x + y = 8) (h_prod : x * y = 12) : 
  (1/x + 1/y) = 2/3 :=
sorry

end min_value_reciprocals_l121_121724


namespace jail_time_calculation_l121_121778

-- Define conditions
def days_of_protest : ℕ := 30
def number_of_cities : ℕ := 21
def arrests_per_day : ℕ := 10
def pre_trial_days : ℕ := 4
def half_two_week_sentence_days : ℕ := 7 -- 1 week is half of 2 weeks

-- Define the calculation of the total combined weeks of jail time
def total_combined_weeks_jail_time : ℕ :=
  let total_arrests := arrests_per_day * number_of_cities * days_of_protest
  let total_days_jail_per_person := pre_trial_days + half_two_week_sentence_days
  let total_combined_days_jail_time := total_arrests * total_days_jail_per_person
  total_combined_days_jail_time / 7

-- Theorem statement
theorem jail_time_calculation : total_combined_weeks_jail_time = 9900 := by
  sorry

end jail_time_calculation_l121_121778


namespace shortest_player_height_l121_121537

-- let h_tall be the height of the tallest player
-- let h_short be the height of the shortest player
-- let diff be the height difference between the tallest and the shortest player

variable (h_tall h_short diff : ℝ)

-- conditions given in the problem
axiom tall_player_height : h_tall = 77.75
axiom height_difference : diff = 9.5
axiom height_relationship : h_tall = h_short + diff

-- the statement we need to prove
theorem shortest_player_height : h_short = 68.25 := by
  sorry

end shortest_player_height_l121_121537


namespace find_initial_tomatoes_l121_121420

-- Define the initial number of tomatoes
def initial_tomatoes (T : ℕ) : Prop :=
  T + 77 - 172 = 80

-- Theorem statement to prove the initial number of tomatoes is 175
theorem find_initial_tomatoes : ∃ T : ℕ, initial_tomatoes T ∧ T = 175 :=
sorry

end find_initial_tomatoes_l121_121420


namespace find_common_ratio_l121_121149

variable {α : Type*} [LinearOrderedField α] [NormedLinearOrderedField α]

def geometric_sequence (a : ℕ → α) (q : α) : Prop := ∀ n, a (n+1) = q * a n

def sum_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop := ∀ n, S n = (Finset.range n).sum a

theorem find_common_ratio
  (a : ℕ → α)
  (S : ℕ → α)
  (q : α)
  (pos_terms : ∀ n, 0 < a n)
  (geometric_seq : geometric_sequence a q)
  (sum_eq : sum_first_n_terms a S)
  (eqn : S 1 + 2 * S 5 = 3 * S 3) :
  q = (2:α)^(3 / 2) / 2^(3 / 2) :=
by
  sorry

end find_common_ratio_l121_121149


namespace fourth_triangle_exists_l121_121180

theorem fourth_triangle_exists (a b c d : ℝ)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)
  (h4 : a + b > d) (h5 : a + d > b) (h6 : b + d > a)
  (h7 : a + c > d) (h8 : a + d > c) (h9 : c + d > a) :
  b + c > d ∧ b + d > c ∧ c + d > b :=
by
  -- I skip the proof with "sorry"
  sorry

end fourth_triangle_exists_l121_121180


namespace rebecca_less_than_toby_l121_121073

-- Define the conditions
variable (x : ℕ) -- Thomas worked x hours
variable (tobyHours : ℕ := 2 * x - 10) -- Toby worked 10 hours less than twice what Thomas worked
variable (rebeccaHours : ℕ := 56) -- Rebecca worked 56 hours

-- Define the total hours worked in one week
axiom total_hours_worked : x + tobyHours + rebeccaHours = 157

-- The proof goal
theorem rebecca_less_than_toby : tobyHours - rebeccaHours = 8 := 
by
  -- (proof steps would go here)
  sorry

end rebecca_less_than_toby_l121_121073


namespace weight_of_pecans_l121_121253

theorem weight_of_pecans (total_weight_of_nuts almonds_weight pecans_weight : ℝ)
  (h1 : total_weight_of_nuts = 0.52)
  (h2 : almonds_weight = 0.14)
  (h3 : pecans_weight = total_weight_of_nuts - almonds_weight) :
  pecans_weight = 0.38 :=
  by
    sorry

end weight_of_pecans_l121_121253


namespace cosine_double_angle_l121_121680

theorem cosine_double_angle
  (θ : ℝ)
  (cos_θ : ℝ)
  (h : cos_θ = 3 / 5) :
  2 * cos_θ^2 - 1 = -7 / 25 :=
by
  sorry

end cosine_double_angle_l121_121680


namespace average_test_score_fifty_percent_l121_121319

-- Given conditions
def percent1 : ℝ := 15
def avg1 : ℝ := 100
def percent2 : ℝ := 50
def avg3 : ℝ := 63
def overall_average : ℝ := 76.05

-- Intermediate calculations based on given conditions
def total_percent : ℝ := 100
def percent3: ℝ := total_percent - percent1 - percent2
def sum_of_weights: ℝ := overall_average * total_percent

-- Expected average of the group that is 50% of the class
theorem average_test_score_fifty_percent (X: ℝ) :
  sum_of_weights = percent1 * avg1 + percent2 * X + percent3 * avg3 → X = 78 := by
  sorry

end average_test_score_fifty_percent_l121_121319


namespace minimum_value_of_f_on_interval_l121_121305

noncomputable def f (a x : ℝ) := Real.log x + a * x

theorem minimum_value_of_f_on_interval (a : ℝ) (h : a < 0) :
  ( ( -Real.log 2 ≤ a ∧ a < 0 ∧ ∀ x ∈ Set.Icc 1 2, f a x ≥ a ) ∧
    ( a < -Real.log 2 ∧ ∀ x ∈ Set.Icc 1 2, f a x ≥ (Real.log 2 + 2 * a) )
  ) :=
by
  sorry

end minimum_value_of_f_on_interval_l121_121305


namespace emma_final_amount_l121_121991

theorem emma_final_amount :
  ∀ (amount_from_bank spent_on_furniture remaining after_giving_friend: ℝ),
  amount_from_bank = 2000 →
  spent_on_furniture = 400 →
  remaining = amount_from_bank - spent_on_furniture →
  after_giving_friend = (3 / 4) * remaining →
  remaining - after_giving_friend = 400 :=
by
  intros amount_from_bank spent_on_furniture remaining after_giving_friend
  assume h1 h2 h3 h4
  sorry

end emma_final_amount_l121_121991


namespace sum_smallest_largest_even_integers_l121_121749

theorem sum_smallest_largest_even_integers (n : ℕ) (h_odd : n % 2 = 1) (b z : ℤ)
  (h_mean : z = b + n - 1) : (b + (b + 2 * (n - 1))) = 2 * z :=
by
  sorry

end sum_smallest_largest_even_integers_l121_121749


namespace problem1_solution_problem2_solution_l121_121063

theorem problem1_solution (x : ℝ) (h : 5 / (x - 1) = 1 / (2 * x + 1)) : x = -2 / 3 := sorry

theorem problem2_solution (x : ℝ) (h : 1 / (x - 2) + 2 = (1 - x) / (2 - x)) : false := sorry

end problem1_solution_problem2_solution_l121_121063


namespace subsets_with_at_least_four_adjacent_chairs_l121_121380

/-- The number of subsets of a set of 12 chairs arranged in a circle that contain at least four adjacent chairs is 1776. -/
theorem subsets_with_at_least_four_adjacent_chairs (S : Finset (Fin 12)) :
  let n := 12 in
  ∃ F : Finset (Finset (Fin 12)), (∀ s ∈ F, ∃ l : List (Fin 12), (l.length ≥ 4 ∧ l.nodup ∧ ∀ i, i ∈ l → (i + 1) % n ∈ l)) ∧ F.card = 1776 := 
sorry

end subsets_with_at_least_four_adjacent_chairs_l121_121380


namespace stuffed_animal_total_l121_121885

/-- McKenna has 34 stuffed animals. -/
def mckenna_stuffed_animals : ℕ := 34

/-- Kenley has twice as many stuffed animals as McKenna. -/
def kenley_stuffed_animals : ℕ := 2 * mckenna_stuffed_animals

/-- Tenly has 5 more stuffed animals than Kenley. -/
def tenly_stuffed_animals : ℕ := kenley_stuffed_animals + 5

/-- The total number of stuffed animals the three girls have. -/
def total_stuffed_animals : ℕ := mckenna_stuffed_animals + kenley_stuffed_animals + tenly_stuffed_animals

/-- Prove that the total number of stuffed animals is 175. -/
theorem stuffed_animal_total : total_stuffed_animals = 175 := by
  sorry

end stuffed_animal_total_l121_121885


namespace find_angle_A_l121_121178

-- Variables representing angles A and B
variables (A B : ℝ)

-- The conditions of the problem translated into Lean
def angle_relationship := A = 2 * B - 15
def angle_supplementary := A + B = 180

-- The theorem statement we need to prove
theorem find_angle_A (h1 : angle_relationship A B) (h2 : angle_supplementary A B) : A = 115 :=
by { sorry }

end find_angle_A_l121_121178


namespace total_stuffed_animals_l121_121882

theorem total_stuffed_animals (M K T : ℕ) 
  (hM : M = 34) 
  (hK : K = 2 * M) 
  (hT : T = K + 5) : 
  M + K + T = 175 :=
by
  -- Adding sorry to complete the placeholder
  sorry

end total_stuffed_animals_l121_121882


namespace tetrahedron_volume_l121_121896

theorem tetrahedron_volume (h_1 h_2 h_3 : ℝ) (V : ℝ)
  (h1_pos : 0 < h_1) (h2_pos : 0 < h_2) (h3_pos : 0 < h_3)
  (V_nonneg : 0 ≤ V) : 
  V ≥ (1 / 3) * h_1 * h_2 * h_3 := sorry

end tetrahedron_volume_l121_121896


namespace number_of_houses_built_l121_121222

def original_houses : ℕ := 20817
def current_houses : ℕ := 118558
def houses_built : ℕ := current_houses - original_houses

theorem number_of_houses_built :
  houses_built = 97741 := by
  sorry

end number_of_houses_built_l121_121222


namespace production_days_l121_121570

theorem production_days (n : ℕ) 
    (h1 : 70 * n + 90 = 75 * (n + 1)) : n = 3 := 
sorry

end production_days_l121_121570


namespace petrov_vasechkin_boards_l121_121735

theorem petrov_vasechkin_boards:
  ∃ n : ℕ, 
  (∃ x y : ℕ, 2 * x + 3 * y = 87 ∧ x + y = n) ∧ 
  (∃ u v : ℕ, 3 * u + 5 * v = 94 ∧ u + v = n) ∧ 
  n = 30 := 
sorry

end petrov_vasechkin_boards_l121_121735


namespace solve_inequality_l121_121627

theorem solve_inequality (x : Real) : 
  x^2 - 48 * x + 576 ≤ 16 ↔ 20 ≤ x ∧ x ≤ 28 :=
by
  sorry

end solve_inequality_l121_121627


namespace anna_reading_time_l121_121268

theorem anna_reading_time 
  (C : ℕ)
  (T_per_chapter : ℕ)
  (hC : C = 31) 
  (hT : T_per_chapter = 20) :
  (C - (C / 3)) * T_per_chapter / 60 = 7 := 
by 
  -- proof steps will go here
  sorry

end anna_reading_time_l121_121268


namespace cos_double_angle_l121_121669

variable (θ : Real)
variables (h : cos θ = 3 / 5)

theorem cos_double_angle :
  cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l121_121669


namespace male_athletes_sampled_l121_121599

-- Define the total number of athletes
def total_athletes : Nat := 98

-- Define the number of female athletes
def female_athletes : Nat := 42

-- Define the probability of being selected
def selection_probability : ℚ := 2 / 7

-- Calculate the number of male athletes
def male_athletes : Nat := total_athletes - female_athletes

-- State the theorem about the number of male athletes sampled
theorem male_athletes_sampled : male_athletes * selection_probability = 16 :=
by
  sorry

end male_athletes_sampled_l121_121599


namespace constant_term_binomial_l121_121751

theorem constant_term_binomial (n : ℕ) (h : n = 5) : ∃ (r : ℕ), r = 6 ∧ (Nat.choose (2 * n) r) = 210 := by
  sorry

end constant_term_binomial_l121_121751


namespace m_value_l121_121984

open Polynomial

noncomputable def f (m : ℚ) : Polynomial ℚ := X^4 - 5*X^2 + 4*X - C m

theorem m_value (m : ℚ) : (2 * X + 1) ∣ f m ↔ m = -51/16 := by sorry

end m_value_l121_121984


namespace ratio_of_speeds_l121_121227

theorem ratio_of_speeds (k r t V1 V2 : ℝ) (hk : 0 < k) (hr : 0 < r) (ht : 0 < t)
    (h1 : r * (V1 - V2) = k) (h2 : t * (V1 + V2) = k) :
    |r + t| / |r - t| = V1 / V2 :=
by
  sorry

end ratio_of_speeds_l121_121227


namespace abc_value_l121_121259

theorem abc_value (a b c : ℝ) 
  (h0 : (a * (0 : ℝ)^2 + b * (0 : ℝ) + c) = 7) 
  (h1 : (a * (1 : ℝ)^2 + b * (1 : ℝ) + c) = 4) : 
  a + b + 2 * c = 11 :=
by sorry

end abc_value_l121_121259


namespace square_difference_l121_121232

theorem square_difference :
  let a := 1001
  let b := 999
  a^2 - b^2 = 4000 :=
by
  let a := 1001
  let b := 999
  have h1 : a^2 - b^2 = (a + b) * (a - b), from sorry
  have h2 : a + b = 2000, by sorry
  have h3 : a - b = 2, by sorry
  show a^2 - b^2 = 4000, by sorry

end square_difference_l121_121232


namespace vector_at_t_neg3_l121_121466

theorem vector_at_t_neg3 :
  let a := (2, 3)
  let b := (12, -37)
  let d := ((b.1 - a.1) / 5, (b.2 - a.2) / 5)
  let line_param (t : ℝ) := (a.1 + t * d.1, a.2 + t * d.2)
  line_param (-3) = (-4, 27) := by
  -- Proof goes here
  sorry

end vector_at_t_neg3_l121_121466


namespace tank_capacity_l121_121967

theorem tank_capacity (x : ℝ) (h₁ : 0.25 * x = 60) (h₂ : 0.05 * x = 12) : x = 240 :=
sorry

end tank_capacity_l121_121967


namespace find_price_of_pants_l121_121179

theorem find_price_of_pants
  (price_jacket : ℕ)
  (num_jackets : ℕ)
  (price_shorts : ℕ)
  (num_shorts : ℕ)
  (num_pants : ℕ)
  (total_cost : ℕ)
  (h1 : price_jacket = 10)
  (h2 : num_jackets = 3)
  (h3 : price_shorts = 6)
  (h4 : num_shorts = 2)
  (h5 : num_pants = 4)
  (h6 : total_cost = 90)
  : (total_cost - (num_jackets * price_jacket + num_shorts * price_shorts)) / num_pants = 12 :=
by sorry

end find_price_of_pants_l121_121179


namespace arc_length_parametric_curve_l121_121276

noncomputable def curve_x (t : ℝ) : ℝ := Real.exp t * (Real.cos t + Real.sin t)
noncomputable def curve_y (t : ℝ) : ℝ := Real.exp t * (Real.cos t - Real.sin t)

theorem arc_length_parametric_curve : 
  ∫ (t : ℝ) in (π/2)..π, Real.sqrt ((Deriv curve_x t)^2 + (Deriv curve_y t)^2) = 2 * (Real.exp π - Real.exp (π/2)) :=
by
  sorry

end arc_length_parametric_curve_l121_121276


namespace cos_double_angle_l121_121673

variable (θ : Real)
variables (h : cos θ = 3 / 5)

theorem cos_double_angle :
  cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_l121_121673


namespace smallest_positive_period_maximum_f_B_l121_121641

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin (x + Real.pi / 3) - Real.sqrt 3 / 2

theorem smallest_positive_period (x : ℝ) : 
  (∀ T, (f (x + T) = f x) → (T ≥ 0) → T = Real.pi) := 
sorry

variable {a b c : ℝ}

lemma cos_law_cos_B (h : b^2 = a * c) : 
  ∀ (B : ℝ), (B > 0) ∧ (B < Real.pi) → 
  (1 / 2) ≤ Real.cos B ∧ Real.cos B < 1 := 
sorry

theorem maximum_f_B (h : b^2 = a * c) :
  ∀ (B : ℝ), (B > 0) ∧ (B < Real.pi) → 
  f B ≤ 1 := 
sorry

end smallest_positive_period_maximum_f_B_l121_121641


namespace monochromatic_rectangle_l121_121370

theorem monochromatic_rectangle (n : ℕ) (coloring : ℕ × ℕ → Fin n) :
  ∃ (a b c d : ℕ × ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
  (coloring a = coloring b) ∧ (coloring b = coloring c) ∧ (coloring c = coloring d) :=
sorry

end monochromatic_rectangle_l121_121370


namespace calculate_amount_after_two_years_l121_121138

noncomputable def amount_after_years (initial_value : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 + rate) ^ years

theorem calculate_amount_after_two_years :
  amount_after_years 51200 0.125 2 = 64800 :=
by
  sorry

end calculate_amount_after_two_years_l121_121138


namespace total_travel_time_in_minutes_l121_121047

def riding_rate : ℝ := 10 -- 10 miles per hour
def initial_riding_time : ℝ := 30 -- 30 minutes
def another_riding_distance : ℝ := 15 -- 15 miles
def resting_time : ℝ := 30 -- 30 minutes
def remaining_distance : ℝ := 20 -- 20 miles

theorem total_travel_time_in_minutes :
  initial_riding_time +
  (another_riding_distance / riding_rate * 60) +
  resting_time +
  (remaining_distance / riding_rate * 60) = 270 :=
by
  sorry

end total_travel_time_in_minutes_l121_121047


namespace stuffed_animal_total_l121_121886

/-- McKenna has 34 stuffed animals. -/
def mckenna_stuffed_animals : ℕ := 34

/-- Kenley has twice as many stuffed animals as McKenna. -/
def kenley_stuffed_animals : ℕ := 2 * mckenna_stuffed_animals

/-- Tenly has 5 more stuffed animals than Kenley. -/
def tenly_stuffed_animals : ℕ := kenley_stuffed_animals + 5

/-- The total number of stuffed animals the three girls have. -/
def total_stuffed_animals : ℕ := mckenna_stuffed_animals + kenley_stuffed_animals + tenly_stuffed_animals

/-- Prove that the total number of stuffed animals is 175. -/
theorem stuffed_animal_total : total_stuffed_animals = 175 := by
  sorry

end stuffed_animal_total_l121_121886


namespace rectangle_perimeter_l121_121575

noncomputable def perimeter (a b c : ℕ) : ℕ :=
  2 * (a + b)

theorem rectangle_perimeter (p q: ℕ) (rel_prime: Nat.gcd p q = 1) :
  ∃ (a b c: ℕ), p = 2 * (a + b) ∧ p + q = 52 ∧ a = 5 ∧ b = 12 ∧ c = 7 :=
by
  sorry

end rectangle_perimeter_l121_121575


namespace evaporation_fraction_l121_121095

theorem evaporation_fraction (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1)
  (h : (1 - x) * (3 / 4) = 1 / 6) : x = 7 / 9 :=
by
  sorry

end evaporation_fraction_l121_121095


namespace determine_constants_l121_121446

theorem determine_constants (P Q R : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
    (-2 * x^2 + 5 * x - 7) / (x^3 - x) = P / x + (Q * x + R) / (x^2 - 1)) ↔
    (P = 7 ∧ Q = -9 ∧ R = 5) :=
by
  sorry

end determine_constants_l121_121446


namespace bucket_fill_turns_l121_121440

-- Definitions
variables (Q : ℚ) (capacityP capacityQ capacityR drumCapacity turns : ℚ)

-- Conditions
def capacities : Prop :=
  capacityP = 3 * capacityQ ∧
  capacityR = (1/2) * capacityQ ∧
  drumCapacity = 80 * capacityP

theorem bucket_fill_turns (h : capacities Q capacityP capacityQ capacityR drumCapacity) : 
  let combined_capacity := 3 * capacityQ + capacityQ + (1/2) * capacityQ in
  turns = drumCapacity / combined_capacity → 
  turns = 54 := 
by 
  intros _ 
  sorry

end bucket_fill_turns_l121_121440


namespace evaluate_expression_l121_121739

theorem evaluate_expression (x : ℤ) (h1 : 0 ≤ x ∧ x ≤ 2) (h2 : x ≠ 1) (h3 : x ≠ 2) (h4 : x = 0) :
    ( ((4 - x) / (x - 1) - x) / ((x - 2) / (x - 1)) ) = -2 :=
by
    sorry

end evaluate_expression_l121_121739


namespace n_fraction_of_sum_l121_121951

theorem n_fraction_of_sum (l : List ℝ) (h1 : l.length = 21) (n : ℝ) (h2 : n ∈ l)
  (h3 : ∃ m, l.erase n = m ∧ m.length = 20 ∧ n = 4 * (m.sum / 20)) :
  n = (l.sum) / 6 :=
by
  sorry

end n_fraction_of_sum_l121_121951


namespace subsets_with_at_least_four_adjacent_chairs_l121_121392

theorem subsets_with_at_least_four_adjacent_chairs :
  let chairs := Finset.range 12 in
  ∑ n in chairs, if n ≥ 4 then (12.choose n) else 0 = 1622 :=
by
  sorry

end subsets_with_at_least_four_adjacent_chairs_l121_121392


namespace walnut_tree_total_count_l121_121544

theorem walnut_tree_total_count (current_trees new_trees : ℕ) (h_current : current_trees = 22) (h_new : new_trees = 33) : 
  current_trees + new_trees = 55 :=
by 
  rw [h_current, h_new]
  exact rfl

end walnut_tree_total_count_l121_121544


namespace greatest_consecutive_integers_sum_36_l121_121566

-- Definition of the sum of N consecutive integers starting from a
def sum_consecutive_integers (a N : ℤ) : ℤ :=
  (N * (2 * a + N - 1)) / 2

-- Problem statement in Lean 4
theorem greatest_consecutive_integers_sum_36 (N : ℤ) (h : sum_consecutive_integers (-35) 72 = 36) : N = 72 := by
  sorry

end greatest_consecutive_integers_sum_36_l121_121566


namespace calc_c_15_l121_121042

noncomputable def c : ℕ → ℝ
| 0 => 1 -- This case won't be used, setup for pattern match
| 1 => 3
| 2 => 5
| (n+3) => c (n+2) * c (n+1)

theorem calc_c_15 : c 15 = 3 ^ 235 :=
sorry

end calc_c_15_l121_121042


namespace quadratic_function_properties_l121_121280

def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 2

theorem quadratic_function_properties :
  (∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, f x ≤ f y) ∧
  (∃ x : ℝ, x = 1.5 ∧ ∀ y : ℝ, f x ≤ f y) :=
by
  sorry

end quadratic_function_properties_l121_121280


namespace ratio_first_to_second_l121_121765

theorem ratio_first_to_second (A B C : ℕ) (h1 : A + B + C = 98) (h2 : B = 30) (h3 : B / C = 5 / 8) : A / B = 2 / 3 :=
sorry

end ratio_first_to_second_l121_121765


namespace percentage_increase_in_radius_l121_121102

theorem percentage_increase_in_radius (r R : ℝ) (h : π * R^2 = π * r^2 + 1.25 * (π * r^2)) :
  R = 1.5 * r :=
by
  -- Proof goes here
  sorry

end percentage_increase_in_radius_l121_121102


namespace dice_sum_not_possible_l121_121080

theorem dice_sum_not_possible (a b c d : ℕ) (h₁ : 1 ≤ a ∧ a ≤ 6) (h₂ : 1 ≤ b ∧ b ≤ 6) 
(h₃ : 1 ≤ c ∧ c ≤ 6) (h₄ : 1 ≤ d ∧ d ≤ 6) (h_product : a * b * c * d = 216) : 
(a + b + c + d ≠ 15) ∧ (a + b + c + d ≠ 16) ∧ (a + b + c + d ≠ 18) :=
sorry

end dice_sum_not_possible_l121_121080


namespace selling_price_correct_l121_121104

noncomputable def cost_price : ℝ := 100
noncomputable def gain_percent : ℝ := 0.15
noncomputable def profit : ℝ := gain_percent * cost_price
noncomputable def selling_price : ℝ := cost_price + profit

theorem selling_price_correct : selling_price = 115 := by
  sorry

end selling_price_correct_l121_121104


namespace number_of_six_digit_integers_formed_with_repetition_l121_121008

theorem number_of_six_digit_integers_formed_with_repetition :
  ∃ n : ℕ, n = 60 ∧ nat.factorial 6 / (nat.factorial 3 * nat.factorial 2 * nat.factorial 1) = n :=
begin
  use 60,
  split,
  { refl },
  { sorry }
end

end number_of_six_digit_integers_formed_with_repetition_l121_121008


namespace cos_double_angle_l121_121697

theorem cos_double_angle (θ : ℝ) (h : cos θ = 3/5) : cos (2*θ) = -7/25 := 
by 
  sorry

end cos_double_angle_l121_121697


namespace three_number_relationship_l121_121759

theorem three_number_relationship :
  let a := (0.7 : ℝ) ^ 6
  let b := 6 ^ (0.7 : ℝ)
  let c := Real.log 6 / Real.log 0.7
  c < a ∧ a < b :=
sorry

end three_number_relationship_l121_121759


namespace range_of_x_add_y_l121_121873

noncomputable def floor_not_exceeding (z : ℝ) : ℤ := ⌊z⌋

theorem range_of_x_add_y (x y : ℝ) (h1 : y = 3 * floor_not_exceeding x + 4) 
    (h2 : y = 4 * floor_not_exceeding (x - 3) + 7) (h3 : ¬ ∃ n : ℤ, x = n) : 
    40 < x + y ∧ x + y < 41 :=
by 
  sorry 

end range_of_x_add_y_l121_121873


namespace unique_a_for_fx_eq_2ax_l121_121637

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * Real.log x

theorem unique_a_for_fx_eq_2ax (a : ℝ) (h : 0 < a) :
  (∀ x : ℝ, f x a = 2 * a * x → x = (a + Real.sqrt (a^2 + 4 * a)) / 2) →
  a = 1 / 2 :=
sorry

end unique_a_for_fx_eq_2ax_l121_121637


namespace solve_for_a_l121_121848

theorem solve_for_a (a : ℝ) : 
  (2 * a + 16 + 3 * a - 8) / 2 = 69 → a = 26 :=
by
  sorry

end solve_for_a_l121_121848
