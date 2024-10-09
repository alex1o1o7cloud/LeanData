import Mathlib

namespace george_purchased_two_large_pizzas_l2123_212390

noncomputable def small_slices := 4
noncomputable def large_slices := 8
noncomputable def small_pizzas_purchased := 3
noncomputable def george_slices := 3
noncomputable def bob_slices := george_slices + 1
noncomputable def susie_slices := bob_slices / 2
noncomputable def bill_slices := 3
noncomputable def fred_slices := 3
noncomputable def mark_slices := 3
noncomputable def leftover_slices := 10

noncomputable def total_slices_consumed := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

noncomputable def total_slices_before_eating := total_slices_consumed + leftover_slices

noncomputable def small_pizza_total_slices := small_pizzas_purchased * small_slices

noncomputable def large_pizza_total_slices := total_slices_before_eating - small_pizza_total_slices

noncomputable def large_pizzas_purchased := large_pizza_total_slices / large_slices

theorem george_purchased_two_large_pizzas : large_pizzas_purchased = 2 :=
sorry

end george_purchased_two_large_pizzas_l2123_212390


namespace evaluate_series_sum_l2123_212311

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l2123_212311


namespace maria_workday_ends_at_330_pm_l2123_212354

/-- 
Given:
1. Maria's workday is 8 hours long.
2. Her workday does not include her lunch break.
3. Maria starts work at 7:00 A.M.
4. She takes her lunch break at 11:30 A.M., lasting 30 minutes.
Prove that Maria's workday ends at 3:30 P.M.
-/
def maria_end_workday : Prop :=
  let start_time : Nat := 7 * 60 -- in minutes
  let lunch_start_time : Nat := 11 * 60 + 30 -- in minutes
  let lunch_duration : Nat := 30 -- in minutes
  let lunch_end_time : Nat := lunch_start_time + lunch_duration
  let total_work_minutes : Nat := 8 * 60
  let work_before_lunch : Nat := lunch_start_time - start_time
  let remaining_work : Nat := total_work_minutes - work_before_lunch
  let end_time : Nat := lunch_end_time + remaining_work
  end_time = 15 * 60 + 30

theorem maria_workday_ends_at_330_pm : maria_end_workday :=
  by
    sorry

end maria_workday_ends_at_330_pm_l2123_212354


namespace minimum_value_l2123_212367

noncomputable def min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1 / (a + 1) + 1 / (b + 1) = 1) :=
  a + 2 * b

theorem minimum_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 1 / (a + 1) + 1 / (b + 1) = 1) :
  min_value a b h₁ h₂ h₃ ≥ 2 * Real.sqrt 2 :=
sorry

end minimum_value_l2123_212367


namespace wyatt_bought_4_cartons_of_juice_l2123_212326

/-- 
Wyatt's mother gave him $74 to go to the store.
Wyatt bought 5 loaves of bread, each costing $5.
Each carton of orange juice cost $2.
Wyatt has $41 left.
We need to prove that Wyatt bought 4 cartons of orange juice.
-/
theorem wyatt_bought_4_cartons_of_juice (initial_money spent_money loaves_price juice_price loaves_qty money_left juice_qty : ℕ)
  (h1 : initial_money = 74)
  (h2 : money_left = 41)
  (h3 : loaves_price = 5)
  (h4 : juice_price = 2)
  (h5 : loaves_qty = 5)
  (h6 : spent_money = initial_money - money_left)
  (h7 : spent_money = loaves_qty * loaves_price + juice_qty * juice_price) :
  juice_qty = 4 :=
by
  -- the proof would go here
  sorry

end wyatt_bought_4_cartons_of_juice_l2123_212326


namespace ordered_triples_count_eq_4_l2123_212314

theorem ordered_triples_count_eq_4 :
  ∃ (S : Finset (ℝ × ℝ × ℝ)), 
    (∀ x y z : ℝ, (x, y, z) ∈ S ↔ (x ≠ 0) ∧ (y ≠ 0) ∧ (z ≠ 0) ∧ (xy + 1 = z) ∧ (yz + 1 = x) ∧ (zx + 1 = y)) ∧
    S.card = 4 :=
sorry

end ordered_triples_count_eq_4_l2123_212314


namespace sum_first_n_geometric_terms_l2123_212346

theorem sum_first_n_geometric_terms (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h1 : S 2 = 2) (h2 : S 6 = 4) :
  S 4 = 1 + Real.sqrt 5 :=
by
  sorry

end sum_first_n_geometric_terms_l2123_212346


namespace expression_positive_for_all_integers_l2123_212357

theorem expression_positive_for_all_integers (n : ℤ) : 6 * n^2 - 7 * n + 2 > 0 :=
by
  sorry

end expression_positive_for_all_integers_l2123_212357


namespace expression_divisible_512_l2123_212348

theorem expression_divisible_512 (n : ℤ) (h : n % 2 ≠ 0) : (n^12 - n^8 - n^4 + 1) % 512 = 0 := 
by 
  sorry

end expression_divisible_512_l2123_212348


namespace percentage_orange_juice_l2123_212377

-- Definitions based on conditions
def total_volume : ℝ := 120
def watermelon_percentage : ℝ := 0.60
def grape_juice_volume : ℝ := 30
def watermelon_juice_volume : ℝ := watermelon_percentage * total_volume
def combined_watermelon_grape_volume : ℝ := watermelon_juice_volume + grape_juice_volume
def orange_juice_volume : ℝ := total_volume - combined_watermelon_grape_volume

-- Lean 4 statement to prove the percentage of orange juice
theorem percentage_orange_juice : (orange_juice_volume / total_volume) * 100 = 15 := by
  -- sorry to skip the proof
  sorry

end percentage_orange_juice_l2123_212377


namespace cos_690_eq_sqrt3_div_2_l2123_212380

theorem cos_690_eq_sqrt3_div_2 : Real.cos (690 * Real.pi / 180) = Real.sqrt 3 / 2 := 
by
  sorry

end cos_690_eq_sqrt3_div_2_l2123_212380


namespace quadratic_inequality_solution_l2123_212340

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
by
  sorry

end quadratic_inequality_solution_l2123_212340


namespace probability_of_different_value_and_suit_l2123_212325

theorem probability_of_different_value_and_suit :
  let total_cards := 52
  let first_card_choices := 52
  let remaining_cards := 51
  let different_suits := 3
  let different_values := 12
  let favorable_outcomes := different_suits * different_values
  let total_outcomes := remaining_cards
  let probability := favorable_outcomes / total_outcomes
  probability = 12 / 17 := 
by
  sorry

end probability_of_different_value_and_suit_l2123_212325


namespace remainder_div_by_3_not_divisible_by_9_l2123_212375

theorem remainder_div_by_3 (x : ℕ) (h : x = 1493826) : x % 3 = 0 :=
by sorry

theorem not_divisible_by_9 (x : ℕ) (h : x = 1493826) : x % 9 ≠ 0 :=
by sorry

end remainder_div_by_3_not_divisible_by_9_l2123_212375


namespace fraction_done_by_B_l2123_212318

theorem fraction_done_by_B {A B : ℝ} (h : A = (2/5) * B) : (B / (A + B)) = (5/7) :=
by
  sorry

end fraction_done_by_B_l2123_212318


namespace find_pairs_l2123_212351

def Point := (ℤ × ℤ)

def P : Point := (1, 1)
def Q : Point := (4, 5)
def valid_pairs : List Point := [(4, 1), (7, 5), (10, 9), (1, 5), (4, 9)]

def area (P Q R : Point) : ℚ :=
  (1 / 2 : ℚ) * ((P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)).natAbs : ℚ)

theorem find_pairs :
  {pairs : List Point // ∀ (a b : ℤ), (0 ≤ a ∧ a ≤ 10 ∧ 0 ≤ b ∧ b ≤ 10 ∧ area P Q (a, b) = 6) ↔ (a, b) ∈ pairs} :=
  ⟨valid_pairs, by sorry⟩

end find_pairs_l2123_212351


namespace garden_plant_count_l2123_212309

theorem garden_plant_count :
  let rows := 52
  let columns := 15
  rows * columns = 780 := 
by
  sorry

end garden_plant_count_l2123_212309


namespace cory_packs_l2123_212338

theorem cory_packs (total_money_needed cost_per_pack : ℕ) (h1 : total_money_needed = 98) (h2 : cost_per_pack = 49) : total_money_needed / cost_per_pack = 2 :=
by 
  sorry

end cory_packs_l2123_212338


namespace find_a_extreme_value_l2123_212313

theorem find_a_extreme_value (a : ℝ) :
  (f : ℝ → ℝ := λ x => x^3 + a*x^2 + 3*x - 9) →
  (f' : ℝ → ℝ := λ x => 3*x^2 + 2*a*x + 3) →
  f' (-3) = 0 →
  a = 5 :=
by
  sorry

end find_a_extreme_value_l2123_212313


namespace square_of_nonnegative_not_positive_equiv_square_of_negative_nonpositive_l2123_212389

theorem square_of_nonnegative_not_positive_equiv_square_of_negative_nonpositive :
  (∀ n : ℝ, 0 ≤ n → n^2 ≤ 0 → False) ↔ (∀ m : ℝ, m < 0 → m^2 ≤ 0) := 
sorry

end square_of_nonnegative_not_positive_equiv_square_of_negative_nonpositive_l2123_212389


namespace negation_of_universal_proposition_l2123_212306

theorem negation_of_universal_proposition :
  (¬ ∀ (x : ℝ), x^3 - x^2 + 1 ≤ 0) ↔ ∃ (x₀ : ℝ), x₀^3 - x₀^2 + 1 > 0 :=
by {
  sorry
}

end negation_of_universal_proposition_l2123_212306


namespace nice_set_l2123_212322

def nice (P : Set (ℤ × ℤ)) : Prop :=
  ∀ (a b c d : ℤ), (a, b) ∈ P ∧ (c, d) ∈ P → (b, a) ∈ P ∧ (a + c, b - d) ∈ P

def is_solution (p q : ℤ) : Prop :=
  Int.gcd p q = 1 ∧ p % 2 ≠ q % 2

theorem nice_set (p q : ℤ) (P : Set (ℤ × ℤ)) :
  nice P → (p, q) ∈ P → is_solution p q → P = Set.univ := 
  sorry

end nice_set_l2123_212322


namespace katherine_has_4_apples_l2123_212342

variable (A P : ℕ)

theorem katherine_has_4_apples
  (h1 : P = 3 * A)
  (h2 : A + P = 16) :
  A = 4 := 
sorry

end katherine_has_4_apples_l2123_212342


namespace geometric_series_sum_l2123_212358

theorem geometric_series_sum (a r : ℚ) (ha : a = 1) (hr : r = 1/4) : 
  (∑' n:ℕ, a * r^n) = 4/3 :=
by
  rw [ha, hr]
  sorry

end geometric_series_sum_l2123_212358


namespace find_base_of_log_equation_l2123_212388

theorem find_base_of_log_equation :
  ∃ b : ℝ, (∀ x : ℝ, (9 : ℝ)^(x + 5) = (5 : ℝ)^x → x = Real.logb b ((9 : ℝ)^5)) ∧ b = 5 / 9 :=
by
  sorry

end find_base_of_log_equation_l2123_212388


namespace similar_triangles_y_value_l2123_212321

theorem similar_triangles_y_value :
  ∀ (y : ℚ),
    (12 : ℚ) / y = (9 : ℚ) / 6 → 
    y = 8 :=
by
  intros y h
  sorry

end similar_triangles_y_value_l2123_212321


namespace freq_count_of_third_group_l2123_212395

theorem freq_count_of_third_group
  (sample_size : ℕ) 
  (freq_third_group : ℝ) 
  (h1 : sample_size = 100) 
  (h2 : freq_third_group = 0.2) : 
  (sample_size * freq_third_group) = 20 :=
by 
  sorry

end freq_count_of_third_group_l2123_212395


namespace probability_of_event_3a_minus_1_gt_0_l2123_212341

noncomputable def probability_event : ℝ :=
if h : 0 <= 1 then (1 - 1/3) else 0

theorem probability_of_event_3a_minus_1_gt_0 (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) : 
  probability_event = 2 / 3 :=
by
  sorry

end probability_of_event_3a_minus_1_gt_0_l2123_212341


namespace decimal_to_fraction_l2123_212305

theorem decimal_to_fraction (x : ℚ) (h : x = 2.35) : x = 47 / 20 :=
by sorry

end decimal_to_fraction_l2123_212305


namespace problem_statement_l2123_212391

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : y - x > 1) :
  (1 - y) / x < 1 ∨ (1 + 3 * x) / y < 1 :=
sorry

end problem_statement_l2123_212391


namespace find_y_l2123_212381

theorem find_y (t : ℝ) (x : ℝ) (y : ℝ)
  (hx : x = 3 - 2 * t)
  (hy : y = 3 * t + 6)
  (hx_cond : x = -6) :
  y = 19.5 :=
by
  sorry

end find_y_l2123_212381


namespace remainder_sets_two_disjoint_subsets_l2123_212333

noncomputable def T : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

theorem remainder_sets_two_disjoint_subsets (m : ℕ)
  (h : m = (3^12 - 2 * 2^12 + 1) / 2) : m % 1000 = 625 := 
by {
  -- math proof is omitted
  sorry
}

end remainder_sets_two_disjoint_subsets_l2123_212333


namespace smaller_cube_volume_l2123_212343

theorem smaller_cube_volume
  (d : ℝ) (s : ℝ) (V : ℝ)
  (h1 : d = 12)  -- condition: diameter of the sphere equals the edge length of the larger cube
  (h2 : d = s * Real.sqrt 3)  -- condition: space diagonal of the smaller cube equals the diameter of the sphere
  (h3 : s = 12 / Real.sqrt 3)  -- condition: side length of the smaller cube
  (h4 : V = s^3)  -- condition: volume of the cube with side length s
  : V = 192 * Real.sqrt 3 :=  -- proving the volume of the smaller cube
sorry

end smaller_cube_volume_l2123_212343


namespace line_eq_45_deg_y_intercept_2_circle_eq_center_neg2_3_tangent_yaxis_l2123_212384

theorem line_eq_45_deg_y_intercept_2 :
  (∃ l : ℝ → ℝ, (l 0 = 2) ∧ (∀ x, l x = x + 2)) := sorry

theorem circle_eq_center_neg2_3_tangent_yaxis :
  (∃ c : ℝ × ℝ → ℝ, (c (-2, 3) = 0) ∧ (∀ x y, c (x, y) = (x + 2)^2 + (y - 3)^2 - 4)) := sorry

end line_eq_45_deg_y_intercept_2_circle_eq_center_neg2_3_tangent_yaxis_l2123_212384


namespace Joey_age_digit_sum_l2123_212349

structure Ages :=
  (joey_age : ℕ)
  (chloe_age : ℕ)
  (zoe_age : ℕ)

def is_multiple (a b : ℕ) : Prop :=
  ∃ k, a = k * b

def sum_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem Joey_age_digit_sum
  (C J Z : ℕ)
  (h1 : J = C + 1)
  (h2 : Z = 1)
  (h3 : ∃ n, C + n = (n + 1) * m)
  (m : ℕ) (hm : m = 9)
  (h4 : C - 1 = 36) :
  sum_digits (J + 37) = 12 :=
by
  sorry

end Joey_age_digit_sum_l2123_212349


namespace solution_of_inequality_system_l2123_212378

-- Definitions derived from the conditions in the problem
def inequality1 (x : ℝ) : Prop := 3 * x - 1 ≥ x + 1
def inequality2 (x : ℝ) : Prop := x + 4 > 4 * x - 2
def solution_set (x : ℝ) : Prop := 1 ≤ x ∧ x < 2

-- The Lean 4 statement for the proof problem
theorem solution_of_inequality_system (x : ℝ) : inequality1 x ∧ inequality2 x ↔ solution_set x := by
  sorry

end solution_of_inequality_system_l2123_212378


namespace minNumberOfRectangles_correct_l2123_212376

variable (k n : ℤ)

noncomputable def minNumberOfRectangles (k n : ℤ) : ℤ :=
  if 2 ≤ k ∧ k ≤ n ∧ n ≤ 2*k - 1 then
    if n = k ∨ n = 2*k - 1 then n else 2 * (n - k + 1)
  else 0 -- 0 if the conditions are not met

theorem minNumberOfRectangles_correct (k n : ℤ) (h : 2 ≤ k ∧ k ≤ n ∧ n ≤ 2*k - 1) : 
  minNumberOfRectangles k n = 
  if n = k ∨ n = 2*k - 1 then n else 2 * (n - k + 1) := 
by 
  -- Proof will go here
  sorry

end minNumberOfRectangles_correct_l2123_212376


namespace songs_performed_l2123_212394

variable (R L S M : ℕ)
variable (songs_total : ℕ)

def conditions := 
  R = 9 ∧ L = 6 ∧ (6 ≤ S ∧ S ≤ 9) ∧ (6 ≤ M ∧ M ≤ 9) ∧ songs_total = (R + L + S + M) / 3

theorem songs_performed (h : conditions R L S M songs_total) :
  songs_total = 9 ∨ songs_total = 10 ∨ songs_total = 11 :=
sorry

end songs_performed_l2123_212394


namespace green_apples_count_l2123_212356

def red_apples := 33
def students_took := 21
def extra_apples := 35

theorem green_apples_count : ∃ G : ℕ, red_apples + G - students_took = extra_apples ∧ G = 23 :=
by
  use 23
  have h1 : 33 + 23 - 21 = 35 := by norm_num
  exact ⟨h1, rfl⟩

end green_apples_count_l2123_212356


namespace words_on_each_page_l2123_212359

/-- Given a book with 150 pages, where each page has between 50 and 150 words, 
    and the total number of words in the book is congruent to 217 modulo 221, 
    prove that each page has 135 words. -/
theorem words_on_each_page (p : ℕ) (h1 : 50 ≤ p) (h2 : p ≤ 150) (h3 : 150 * p ≡ 217 [MOD 221]) : 
  p = 135 :=
by
  sorry

end words_on_each_page_l2123_212359


namespace pieces_per_sister_l2123_212350

-- Defining the initial conditions
def initial_cake_pieces : ℕ := 240
def percentage_eaten : ℕ := 60
def number_of_sisters : ℕ := 3

-- Defining the statements to be proved
theorem pieces_per_sister (initial_cake_pieces : ℕ) (percentage_eaten : ℕ) (number_of_sisters : ℕ) :
  let pieces_eaten := (percentage_eaten * initial_cake_pieces) / 100
  let remaining_pieces := initial_cake_pieces - pieces_eaten
  let pieces_per_sister := remaining_pieces / number_of_sisters
  pieces_per_sister = 32 :=
by 
  sorry

end pieces_per_sister_l2123_212350


namespace incorrect_conversion_D_l2123_212332

-- Definition of base conversions as conditions
def binary_to_decimal (b : String) : ℕ := -- Converts binary string to decimal number
  sorry

def octal_to_decimal (o : String) : ℕ := -- Converts octal string to decimal number
  sorry

def decimal_to_base_n (d : ℕ) (n : ℕ) : String := -- Converts decimal number to base-n string
  sorry

-- Given conditions
axiom cond1 : binary_to_decimal "101" = 5
axiom cond2 : octal_to_decimal "27" = 25 -- Note: "27"_base(8) is 2*8 + 7 = 23 in decimal; there's a typo in question's option.
axiom cond3 : decimal_to_base_n 119 6 = "315"
axiom cond4 : decimal_to_base_n 13 2 = "1101" -- Note: correcting from 62 to "1101"_base(2) which is 13

-- Prove the incorrect conversion between number systems
theorem incorrect_conversion_D : decimal_to_base_n 31 4 ≠ "62" :=
  sorry

end incorrect_conversion_D_l2123_212332


namespace pyramid_height_l2123_212337

theorem pyramid_height (h : ℝ) :
  let V_cube := 5^3
  let V_pyramid := (1/3) * 10^2 * h
  V_cube = V_pyramid → h = 3.75 :=
by
  let V_cube := 5^3
  let V_pyramid := (1/3) * 10^2 * h
  intros h_eq
  sorry

end pyramid_height_l2123_212337


namespace range_of_a_l2123_212316

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 3 < x ∧ x < 4 ∧ ax^2 - 4*a*x - 2 > 0) ↔ a < -2/3 :=
sorry

end range_of_a_l2123_212316


namespace symmetric_lines_a_b_l2123_212339

theorem symmetric_lines_a_b (x y a b : ℝ) (A : ℝ × ℝ) (hA : A = (1, 0))
  (h1 : x + 2 * y - 3 = 0)
  (h2 : a * x + 4 * y + b = 0)
  (h_slope : -1 / 2 = -a / 4)
  (h_point : a * 1 + 4 * 0 + b = 0) :
  a + b = 0 :=
sorry

end symmetric_lines_a_b_l2123_212339


namespace percentage_decrease_wages_l2123_212310

theorem percentage_decrease_wages (W : ℝ) (P : ℝ) : 
  (0.20 * W * (1 - P / 100)) = 0.70 * (0.20 * W) → 
  P = 30 :=
by
  sorry

end percentage_decrease_wages_l2123_212310


namespace intersection_M_N_l2123_212334

noncomputable def set_M : Set ℝ := {x | ∃ y, y = Real.sqrt (2 - x^2)}
noncomputable def set_N : Set ℝ := {y | ∃ x, y = x^2 - 1}

theorem intersection_M_N :
  (set_M ∩ set_N) = { x | -1 ≤ x ∧ x ≤ Real.sqrt 2 } := sorry

end intersection_M_N_l2123_212334


namespace coast_guard_overtake_smuggler_l2123_212392

noncomputable def time_of_overtake (initial_distance : ℝ) (initial_time : ℝ) 
                                   (smuggler_speed1 coast_guard_speed : ℝ) 
                                   (duration1 new_smuggler_speed : ℝ) : ℝ :=
  let distance_after_duration1 := initial_distance + (smuggler_speed1 * duration1) - (coast_guard_speed * duration1)
  let relative_speed_new := coast_guard_speed - new_smuggler_speed
  duration1 + (distance_after_duration1 / relative_speed_new)

theorem coast_guard_overtake_smuggler : 
  time_of_overtake 15 0 18 20 1 16 = 4.25 := by
  sorry

end coast_guard_overtake_smuggler_l2123_212392


namespace solution_set_inequality_l2123_212327

theorem solution_set_inequality (a : ℝ) :
  ∀ x : ℝ,
    (12 * x^2 - a * x > a^2) →
    ((a > 0 ∧ (x < -a / 4 ∨ x > a / 3)) ∨
     (a = 0 ∧ x ≠ 0) ∨
     (a < 0 ∧ (x > -a / 4 ∨ x < a / 3))) :=
by
  sorry

end solution_set_inequality_l2123_212327


namespace slower_pump_time_l2123_212362

theorem slower_pump_time (R : ℝ) (hours : ℝ) (combined_rate : ℝ) (faster_rate_adj : ℝ) (time_both : ℝ) :
  (combined_rate = R * (1 + faster_rate_adj)) →
  (faster_rate_adj = 1.5) →
  (time_both = 5) →
  (combined_rate * time_both = 1) →
  (hours = 1 / R) →
  hours = 12.5 :=
by
  sorry

end slower_pump_time_l2123_212362


namespace problem1_l2123_212363

open Real

theorem problem1 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  ∃ (m : ℝ), m = 9 / 2 ∧ ∀ (u v : ℝ), 0 < u → 0 < v → u + v = 1 → (1 / u + 4 / (1 + v)) ≥ m := 
sorry

end problem1_l2123_212363


namespace coefficient_x4_of_square_l2123_212344

theorem coefficient_x4_of_square (q : Polynomial ℝ) (hq : q = Polynomial.X^5 - 4 * Polynomial.X^2 + 3) :
  (Polynomial.coeff (q * q) 4 = 16) :=
by {
  sorry
}

end coefficient_x4_of_square_l2123_212344


namespace building_total_floors_l2123_212312

def earl_final_floor (start : ℕ) : ℕ :=
  start + 5 - 2 + 7

theorem building_total_floors (start : ℕ) (current : ℕ) (remaining : ℕ) (total : ℕ) :
  earl_final_floor start = current →
  remaining = 9 →
  total = current + remaining →
  start = 1 →
  total = 20 := by
sorry

end building_total_floors_l2123_212312


namespace final_fish_stock_l2123_212330

def initial_stock : ℤ := 200 
def sold_fish : ℤ := 50 
def fraction_spoiled : ℚ := 1/3 
def new_stock : ℤ := 200 

theorem final_fish_stock : 
    initial_stock - sold_fish - (fraction_spoiled * (initial_stock - sold_fish)) + new_stock = 300 := 
by 
  sorry

end final_fish_stock_l2123_212330


namespace triangle_inequality_part_a_l2123_212368

theorem triangle_inequality_part_a (a b c : ℝ) (h1 : a + b + c = 4) (h2 : a + b > c) (h3 : b + c > a) (h4 : c + a > b) :
  a^2 + b^2 + c^2 + a * b * c < 8 :=
sorry

end triangle_inequality_part_a_l2123_212368


namespace sum_of_cubes_of_consecutive_integers_l2123_212399

-- Define the given condition
def sum_of_squares_of_consecutive_integers (n : ℕ) : Prop :=
  (n - 1)^2 + n^2 + (n + 1)^2 = 7805

-- Define the statement we want to prove
theorem sum_of_cubes_of_consecutive_integers (n : ℕ) (h : sum_of_squares_of_consecutive_integers n) : 
  (n - 1)^3 + n^3 + (n + 1)^3 = 398259 :=
by
  sorry

end sum_of_cubes_of_consecutive_integers_l2123_212399


namespace compute_expression_l2123_212370

theorem compute_expression :
  24 * 42 + 58 * 24 + 12 * 24 = 2688 := by
  sorry

end compute_expression_l2123_212370


namespace envelopes_initial_count_l2123_212361

noncomputable def initialEnvelopes (given_per_friend : ℕ) (friends : ℕ) (left : ℕ) : ℕ :=
  given_per_friend * friends + left

theorem envelopes_initial_count
  (given_per_friend : ℕ) (friends : ℕ) (left : ℕ)
  (h_given_per_friend : given_per_friend = 3)
  (h_friends : friends = 5)
  (h_left : left = 22) :
  initialEnvelopes given_per_friend friends left = 37 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end envelopes_initial_count_l2123_212361


namespace happy_number_part1_happy_number_part2_happy_number_part3_l2123_212374

section HappyEquations

def is_happy_eq (a b c : ℤ) : Prop :=
  ∃ x1 x2 : ℤ, a ≠ 0 ∧ a * x1 * x1 + b * x1 + c = 0 ∧ a * x2 * x2 + b * x2 + c = 0

def happy_number (a b c : ℤ) : ℚ :=
  (4 * a * c - b ^ 2) / (4 * a)

def happy_to_each_other (a b c p q r : ℤ) : Prop :=
  let Fa : ℚ := happy_number a b c
  let Fb : ℚ := happy_number p q r
  |r * Fa - c * Fb| = 0

theorem happy_number_part1 :
  happy_number 1 (-2) (-3) = -4 :=
by sorry

theorem happy_number_part2 (m : ℤ) (h : 1 < m ∧ m < 6) :
  is_happy_eq 1 (2 * m - 1) (m ^ 2 - 2 * m - 3) →
  m = 3 ∧ happy_number 1 (2 * m - 1) (m ^ 2 - 2 * m - 3) = -25 / 4 :=
by sorry

theorem happy_number_part3 (m n : ℤ) :
  is_happy_eq 1 (-m) (m + 1) ∧ is_happy_eq 1 (-(n + 2)) (2 * n) →
  happy_to_each_other 1 (-m) (m + 1) 1 (-(n + 2)) (2 * n) →
  n = 0 ∨ n = 3 ∨ n = 3 / 2 :=
by sorry

end HappyEquations

end happy_number_part1_happy_number_part2_happy_number_part3_l2123_212374


namespace arithmetic_expression_equality_l2123_212397

theorem arithmetic_expression_equality :
  15 * 25 + 35 * 15 + 16 * 28 + 32 * 16 = 1860 := 
by 
  sorry

end arithmetic_expression_equality_l2123_212397


namespace weight_of_8_moles_CCl4_correct_l2123_212379

/-- The problem states that carbon tetrachloride (CCl4) is given, and we are to determine the weight of 8 moles of CCl4 based on its molar mass calculations. -/
noncomputable def weight_of_8_moles_CCl4 (molar_mass_C : ℝ) (molar_mass_Cl : ℝ) : ℝ :=
  let molar_mass_CCl4 := molar_mass_C + 4 * molar_mass_Cl
  8 * molar_mass_CCl4

/-- Given the molar masses of Carbon (C) and Chlorine (Cl), prove that the calculated weight of 8 moles of CCl4 matches the expected weight. -/
theorem weight_of_8_moles_CCl4_correct :
  let molar_mass_C := 12.01
  let molar_mass_Cl := 35.45
  weight_of_8_moles_CCl4 molar_mass_C molar_mass_Cl = 1230.48 := by
  sorry

end weight_of_8_moles_CCl4_correct_l2123_212379


namespace gcd_75_100_l2123_212382

def is_prime_power (n : ℕ) (p : ℕ) (k : ℕ) : Prop :=
  n = p ^ k

def prime_factors (n : ℕ) (fs : List (ℕ × ℕ)) : Prop :=
  ∀ p k, (p, k) ∈ fs ↔ is_prime_power n p k

theorem gcd_75_100 : gcd 75 100 = 25 :=
by sorry

end gcd_75_100_l2123_212382


namespace gold_weight_l2123_212383

theorem gold_weight:
  ∀ (G C A : ℕ), 
  C = 9 → 
  (A = (4 * G + C) / 5) → 
  A = 17 → 
  G = 19 :=
by
  intros G C A hc ha h17
  sorry

end gold_weight_l2123_212383


namespace average_speed_of_train_l2123_212366

-- Define conditions
def traveled_distance1 : ℝ := 240
def traveled_distance2 : ℝ := 450
def time_period1 : ℝ := 3
def time_period2 : ℝ := 5

-- Define total distance and total time based on the conditions
def total_distance : ℝ := traveled_distance1 + traveled_distance2
def total_time : ℝ := time_period1 + time_period2

-- Prove that the average speed is 86.25 km/h
theorem average_speed_of_train : total_distance / total_time = 86.25 := by
  -- Here should be the proof, but we put sorry since we only need the statement
  sorry

end average_speed_of_train_l2123_212366


namespace inequality_correct_l2123_212372

theorem inequality_correct (a b : ℝ) (ha : a < 0) (hb : b > 0) : (1/a) < (1/b) :=
sorry

end inequality_correct_l2123_212372


namespace original_cost_of_dress_l2123_212396

theorem original_cost_of_dress (x: ℝ) 
  (h1: x / 2 - 10 < x) 
  (h2: x - (x / 2 - 10) = 80) : 
  x = 140 :=
sorry

end original_cost_of_dress_l2123_212396


namespace fraction_comparison_l2123_212320

theorem fraction_comparison (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a / b > (a + 1) / (b + 1) :=
by sorry

end fraction_comparison_l2123_212320


namespace stars_substitution_correct_l2123_212329

-- Define x and y with given conditions
def ends_in_5 (n : ℕ) : Prop := n % 10 = 5
def product_ends_in_25 (x y : ℕ) : Prop := (x * y) % 100 = 25
def tens_digit_even (n : ℕ) : Prop := (n / 10) % 2 = 0
def valid_tens_digit (n : ℕ) : Prop := (n / 10) % 10 ≤ 3

theorem stars_substitution_correct :
  ∃ (x y : ℕ), ends_in_5 x ∧ ends_in_5 y ∧ product_ends_in_25 x y ∧ tens_digit_even x ∧ valid_tens_digit y ∧ x * y = 9125 :=
sorry

end stars_substitution_correct_l2123_212329


namespace nora_nuts_problem_l2123_212302

theorem nora_nuts_problem :
  ∃ n : ℕ, (∀ (a p c : ℕ), 30 * n = 18 * a ∧ 30 * n = 21 * p ∧ 30 * n = 16 * c) ∧ n = 34 :=
by
  -- Provided conditions and solution steps will go here.
  sorry

end nora_nuts_problem_l2123_212302


namespace escher_prints_consecutive_l2123_212336

noncomputable def probability_all_eschers_consecutive (n : ℕ) (m : ℕ) (k : ℕ) : ℚ :=
if h : m = n + 3 ∧ k = 4 then 1 / (n * (n + 1) * (n + 2)) else 0

theorem escher_prints_consecutive :
  probability_all_eschers_consecutive 10 12 4 = 1 / 1320 :=
  by sorry

end escher_prints_consecutive_l2123_212336


namespace find_theta_l2123_212355

def equilateral_triangle_angle : ℝ := 60
def square_angle : ℝ := 90
def pentagon_angle : ℝ := 108
def total_round_angle : ℝ := 360

theorem find_theta (θ : ℝ)
  (h_eq_tri : equilateral_triangle_angle = 60)
  (h_squ : square_angle = 90)
  (h_pen : pentagon_angle = 108)
  (h_round : total_round_angle = 360) :
  θ = total_round_angle - (equilateral_triangle_angle + square_angle + pentagon_angle) :=
sorry

end find_theta_l2123_212355


namespace find_a2023_l2123_212319

variable {a : ℕ → ℕ}
variable {x : ℕ}

def sequence_property (a: ℕ → ℕ) : Prop :=
  ∀ n, a n + a (n + 1) + a (n + 2) = 20

theorem find_a2023 (h1 : sequence_property a) 
                   (h2 : a 2 = 2 * x) 
                   (h3 : a 18 = 9 + x) 
                   (h4 : a 65 = 6 - x) : 
  a 2023 = 5 := 
by
  sorry

end find_a2023_l2123_212319


namespace survey_total_parents_l2123_212324

theorem survey_total_parents (P : ℝ)
  (h1 : 0.15 * P + 0.60 * P + 0.20 * 0.25 * P + 0.05 * P = P)
  (h2 : 0.05 * P = 6) : 
  P = 120 :=
sorry

end survey_total_parents_l2123_212324


namespace compute_permutation_eq_4_l2123_212303

def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem compute_permutation_eq_4 :
  (4 * permutation 8 4 + 2 * permutation 8 5) / (permutation 8 6 - permutation 9 5) * 1 = 4 :=
by
  sorry

end compute_permutation_eq_4_l2123_212303


namespace geometric_sequence_a9_l2123_212323

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

variable (a : ℕ → ℝ)
variable (q : ℝ)

theorem geometric_sequence_a9
  (h_seq : geometric_sequence a q)
  (h2 : a 1 * a 4 = -32)
  (h3 : a 2 + a 3 = 4)
  (hq : ∃ n : ℤ, q = ↑n) :
  a 8 = -256 := 
sorry

end geometric_sequence_a9_l2123_212323


namespace solution1_solution2_l2123_212371

-- Problem: Solving equations and finding their roots

-- Condition 1:
def equation1 (x : Real) : Prop := x^2 - 2 * x = -1

-- Condition 2:
def equation2 (x : Real) : Prop := (x + 3)^2 = 2 * x * (x + 3)

-- Correct answer 1
theorem solution1 : ∀ x : Real, equation1 x → x = 1 := 
by 
  sorry

-- Correct answer 2
theorem solution2 : ∀ x : Real, equation2 x → x = -3 ∨ x = 3 := 
by 
  sorry

end solution1_solution2_l2123_212371


namespace find_x_in_acute_triangle_l2123_212364

-- Definition of an acute triangle with given segment lengths due to altitudes
def acute_triangle_with_segments (A B C D E : Type) (BC AE BE : ℝ) (x : ℝ) : Prop :=
  BC = 4 + x ∧ AE = x ∧ BE = 8 ∧ (A ≠ B ∧ B ≠ C ∧ C ≠ A)

-- The theorem to prove
theorem find_x_in_acute_triangle (A B C D E : Type) (BC AE BE : ℝ) (x : ℝ) 
  (h : acute_triangle_with_segments A B C D E BC AE BE x) : 
  x = 4 :=
by
  -- As the focus is on the statement, we add sorry to skip the proof.
  sorry

end find_x_in_acute_triangle_l2123_212364


namespace roses_formula_l2123_212373

open Nat

def total_roses (n : ℕ) : ℕ := 
  (choose n 4) + (choose (n - 1) 2)

theorem roses_formula (n : ℕ) (h : n ≥ 4) : 
  total_roses n = (choose n 4) + (choose (n - 1) 2) := 
by
  sorry

end roses_formula_l2123_212373


namespace convert_binary_1101_to_decimal_l2123_212315

theorem convert_binary_1101_to_decimal : (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 13) :=
by sorry

end convert_binary_1101_to_decimal_l2123_212315


namespace black_dogs_count_l2123_212393

def number_of_brown_dogs := 20
def number_of_white_dogs := 10
def total_number_of_dogs := 45
def number_of_black_dogs := total_number_of_dogs - (number_of_brown_dogs + number_of_white_dogs)

theorem black_dogs_count : number_of_black_dogs = 15 := by
  sorry

end black_dogs_count_l2123_212393


namespace intersection_A_B_l2123_212352

-- Definitions of the sets A and B according to the problem conditions
def A : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}
def B : Set ℝ := {y | ∃ x : ℝ, y = Real.sqrt (-x^2 + 4 * x - 3)}

-- The proof problem statement
theorem intersection_A_B :
  A ∩ B = {y | 0 ≤ y ∧ y ≤ 1} :=
by
  sorry

end intersection_A_B_l2123_212352


namespace max_value_range_of_t_l2123_212317

theorem max_value_range_of_t (t x : ℝ) (h : t ≤ x ∧ x ≤ t + 2) 
: ∃ y : ℝ, y = -x^2 + 6 * x - 7 ∧ y = -(t - 3)^2 + 2 ↔ t ≥ 3 := 
by {
    sorry
}

end max_value_range_of_t_l2123_212317


namespace negative_represents_backward_l2123_212385

-- Definitions based on conditions
def forward (distance : Int) : Int := distance
def backward (distance : Int) : Int := -distance

-- The mathematical equivalent proof problem
theorem negative_represents_backward
  (distance : Int)
  (h : forward distance = 5) :
  backward distance = -5 :=
sorry

end negative_represents_backward_l2123_212385


namespace new_concentration_l2123_212386

def vessel1 := (3 : ℝ)  -- 3 litres
def conc1 := (0.25 : ℝ) -- 25% alcohol

def vessel2 := (5 : ℝ)  -- 5 litres
def conc2 := (0.40 : ℝ) -- 40% alcohol

def vessel3 := (7 : ℝ)  -- 7 litres
def conc3 := (0.60 : ℝ) -- 60% alcohol

def vessel4 := (4 : ℝ)  -- 4 litres
def conc4 := (0.15 : ℝ) -- 15% alcohol

def total_volume := (25 : ℝ) -- Total vessel capacity

noncomputable def alcohol_total : ℝ :=
  (vessel1 * conc1) + (vessel2 * conc2) + (vessel3 * conc3) + (vessel4 * conc4)

theorem new_concentration : (alcohol_total / total_volume = 0.302) :=
  sorry

end new_concentration_l2123_212386


namespace solve_inequality_l2123_212300

theorem solve_inequality (x : ℝ) :
  |(3 * x - 2) / (x ^ 2 - x - 2)| > 3 ↔ (x ∈ Set.Ioo (-1) (-2 / 3) ∪ Set.Ioo (1 / 3) 4) :=
by sorry

end solve_inequality_l2123_212300


namespace volume_and_surface_area_of_convex_body_l2123_212387

noncomputable def volume_of_convex_body (a b c : ℝ) : ℝ := 
  (a^2 + b^2 + c^2)^3 / (6 * a * b * c)

noncomputable def surface_area_of_convex_body (a b c : ℝ) : ℝ :=
  (a^2 + b^2 + c^2)^(5/2) / (a * b * c)

theorem volume_and_surface_area_of_convex_body (a b c d : ℝ)
  (h : d^2 = a^2 + b^2 + c^2) :
  volume_of_convex_body a b c = (a^2 + b^2 + c^2)^3 / (6 * a * b * c) ∧
  surface_area_of_convex_body a b c = (a^2 + b^2 + c^2)^(5/2) / (a * b * c) :=
by
  sorry

end volume_and_surface_area_of_convex_body_l2123_212387


namespace necessary_condition_x_squared_minus_x_lt_zero_l2123_212360

theorem necessary_condition_x_squared_minus_x_lt_zero (x : ℝ) :
  (x^2 - x < 0) → (-1 < x ∧ x < 1) ∧ ((-1 < x ∧ x < 1) → ¬ (x^2 - x < 0)) :=
by
  sorry

end necessary_condition_x_squared_minus_x_lt_zero_l2123_212360


namespace Kirill_is_69_l2123_212301

/-- Kirill is 14 centimeters shorter than his brother.
    Their sister's height is twice the height of Kirill.
    Their cousin's height is 3 centimeters more than the sister's height.
    Together, their heights equal 432 centimeters.
    We aim to prove that Kirill's height is 69 centimeters.
-/
def Kirill_height (K : ℕ) : Prop :=
  let brother_height := K + 14
  let sister_height := 2 * K
  let cousin_height := 2 * K + 3
  K + brother_height + sister_height + cousin_height = 432

theorem Kirill_is_69 {K : ℕ} (h : Kirill_height K) : K = 69 :=
by
  sorry

end Kirill_is_69_l2123_212301


namespace base_of_second_fraction_l2123_212398

theorem base_of_second_fraction (base : ℝ) (h1 : (1/2) ^ 16 * (1/base) ^ 8 = 1 / (18 ^ 16)): base = 81 :=
sorry

end base_of_second_fraction_l2123_212398


namespace compute_m_div_18_l2123_212347

noncomputable def ten_pow (n : ℕ) : ℕ := Nat.pow 10 n

def valid_digits (m : ℕ) : Prop :=
  ∀ d ∈ m.digits 10, d = 0 ∨ d = 8

def is_multiple_of_18 (m : ℕ) : Prop :=
  m % 18 = 0

theorem compute_m_div_18 :
  ∃ m, valid_digits m ∧ is_multiple_of_18 m ∧ m / 18 = 493827160 :=
by
  sorry

end compute_m_div_18_l2123_212347


namespace fourth_quarter_points_sum_l2123_212331

variable (a d b j : ℕ)

-- Conditions from the problem
def halftime_tied : Prop := 2 * a + d = 2 * b
def wildcats_won_by_four : Prop := 4 * a + 6 * d = 4 * b - 3 * j + 4

-- The proof goal to be established
theorem fourth_quarter_points_sum
  (h1 : halftime_tied a d b)
  (h2 : wildcats_won_by_four a d b j) :
  (a + 3 * d) + (b - 2 * j) = 28 :=
sorry

end fourth_quarter_points_sum_l2123_212331


namespace students_got_off_the_bus_l2123_212365

theorem students_got_off_the_bus
    (original_students : ℕ)
    (students_left : ℕ)
    (h_original : original_students = 10)
    (h_left : students_left = 7) :
    original_students - students_left = 3 :=
by {
  sorry
}

end students_got_off_the_bus_l2123_212365


namespace four_thirds_eq_36_l2123_212369

theorem four_thirds_eq_36 (x : ℝ) (h : (4 / 3) * x = 36) : x = 27 := by
  sorry

end four_thirds_eq_36_l2123_212369


namespace solve_gcd_problem_l2123_212307

def gcd_problem : Prop :=
  gcd 1337 382 = 191

theorem solve_gcd_problem : gcd_problem := 
by 
  sorry

end solve_gcd_problem_l2123_212307


namespace banker_gain_l2123_212353

theorem banker_gain :
  ∀ (t : ℝ) (r : ℝ) (TD : ℝ),
  t = 1 →
  r = 12 →
  TD = 65 →
  (TD * r * t) / (100 - (r * t)) = 8.86 :=
by
  intros t r TD ht hr hTD
  rw [ht, hr, hTD]
  sorry

end banker_gain_l2123_212353


namespace matrix_A_pow_50_l2123_212345

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![5, 1], ![-16, -3]]

theorem matrix_A_pow_50 :
  A ^ 50 = ![![201, 50], ![-800, -199]] :=
sorry

end matrix_A_pow_50_l2123_212345


namespace sufficiency_not_necessity_condition_l2123_212328

theorem sufficiency_not_necessity_condition (a : ℝ) (h : a > 1) : (a^2 > 1) ∧ ¬(∀ x : ℝ, x^2 > 1 → x > 1) :=
by
  sorry

end sufficiency_not_necessity_condition_l2123_212328


namespace sum_of_square_areas_l2123_212335

theorem sum_of_square_areas (a b : ℝ)
  (h1 : a + b = 14)
  (h2 : a - b = 2) :
  a^2 + b^2 = 100 := by
  sorry

end sum_of_square_areas_l2123_212335


namespace find_hours_l2123_212304

theorem find_hours (x : ℕ) (h : (14 + 10 + 13 + 9 + 12 + 11 + x) / 7 = 12) : x = 15 :=
by
  -- The proof is omitted
  sorry

end find_hours_l2123_212304


namespace smallest_possible_X_l2123_212308

-- Define conditions
def is_bin_digit (n : ℕ) : Prop := n = 0 ∨ n = 1

def only_bin_digits (T : ℕ) := ∀ d ∈ T.digits 10, is_bin_digit d

def divisible_by_15 (T : ℕ) : Prop := T % 15 = 0

def is_smallest_X (X : ℕ) : Prop :=
  ∀ T : ℕ, only_bin_digits T → divisible_by_15 T → T / 15 = X → (X = 74)

-- Final statement to prove
theorem smallest_possible_X : is_smallest_X 74 :=
  sorry

end smallest_possible_X_l2123_212308
