import Mathlib

namespace ratio_of_work_done_by_women_to_men_l2095_209506

theorem ratio_of_work_done_by_women_to_men 
  (total_work_men : ℕ := 15 * 21 * 8)
  (total_work_women : ℕ := 21 * 36 * 5) :
  (total_work_women : ℚ) / (total_work_men : ℚ) = 2 / 3 :=
by
  -- Proof goes here
  sorry

end ratio_of_work_done_by_women_to_men_l2095_209506


namespace number_of_possible_values_for_b_l2095_209562

theorem number_of_possible_values_for_b : 
  ∃ (n : ℕ), n = 10 ∧ ∀ (b : ℕ), (2 ≤ b) ∧ (b^2 ≤ 256) ∧ (256 < b^3) ↔ (7 ≤ b ∧ b ≤ 16) :=
by {
  sorry
}

end number_of_possible_values_for_b_l2095_209562


namespace negation_of_proposition_l2095_209507

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, 0 ≤ x ∧ (x^2 - 2*x - 3 = 0)) ↔ (∀ x : ℝ, 0 ≤ x → (x^2 - 2*x - 3 ≠ 0)) := 
by 
  sorry

end negation_of_proposition_l2095_209507


namespace difference_max_min_y_l2095_209572

theorem difference_max_min_y {total_students : ℕ} (initial_yes_pct initial_no_pct final_yes_pct final_no_pct : ℝ)
  (initial_conditions : initial_yes_pct = 0.4 ∧ initial_no_pct = 0.6)
  (final_conditions : final_yes_pct = 0.8 ∧ final_no_pct = 0.2) :
  ∃ (min_change max_change : ℝ), max_change - min_change = 0.2 := by
  sorry

end difference_max_min_y_l2095_209572


namespace total_expenditure_eq_fourteen_l2095_209523

variable (cost_barrette cost_comb : ℕ)
variable (kristine_barrettes kristine_combs crystal_barrettes crystal_combs : ℕ)

theorem total_expenditure_eq_fourteen 
  (h_cost_barrette : cost_barrette = 3)
  (h_cost_comb : cost_comb = 1)
  (h_kristine_barrettes : kristine_barrettes = 1)
  (h_kristine_combs : kristine_combs = 1)
  (h_crystal_barrettes : crystal_barrettes = 3)
  (h_crystal_combs : crystal_combs = 1) :
  (kristine_barrettes * cost_barrette + kristine_combs * cost_comb) +
  (crystal_barrettes * cost_barrette + crystal_combs * cost_comb) = 14 := 
by 
  sorry

end total_expenditure_eq_fourteen_l2095_209523


namespace value_of_x2_plus_9y2_l2095_209561

theorem value_of_x2_plus_9y2 {x y : ℝ} (h1 : x + 3 * y = 6) (h2 : x * y = -9) : x^2 + 9 * y^2 = 90 := 
sorry

end value_of_x2_plus_9y2_l2095_209561


namespace kitten_current_length_l2095_209585

theorem kitten_current_length (initial_length : ℕ) (double_after_2_weeks : ℕ → ℕ) (double_after_4_months : ℕ → ℕ)
  (h1 : initial_length = 4)
  (h2 : double_after_2_weeks initial_length = 2 * initial_length)
  (h3 : double_after_4_months (double_after_2_weeks initial_length) = 2 * (double_after_2_weeks initial_length)) :
  double_after_4_months (double_after_2_weeks initial_length) = 16 := 
by
  sorry

end kitten_current_length_l2095_209585


namespace exponents_of_equation_l2095_209534

theorem exponents_of_equation :
  ∃ (x y : ℕ), 2 * (3 ^ 8) ^ 2 * (2 ^ 3) ^ 2 * 3 = 2 ^ x * 3 ^ y ∧ x = 7 ∧ y = 17 :=
by
  use 7
  use 17
  sorry

end exponents_of_equation_l2095_209534


namespace lower_tap_used_earlier_l2095_209554

-- Define the conditions given in the problem
def capacity : ℕ := 36
def midway_capacity : ℕ := capacity / 2
def lower_tap_rate : ℕ := 4  -- minutes per litre
def upper_tap_rate : ℕ := 6  -- minutes per litre

def lower_tap_draw (minutes : ℕ) : ℕ := minutes / lower_tap_rate  -- litres drawn by lower tap
def beer_left_after_draw (initial_amount litres_drawn : ℕ) : ℕ := initial_amount - litres_drawn

-- Define the assistant's drawing condition
def assistant_draw_min : ℕ := 16
def assistant_draw_litres : ℕ := lower_tap_draw assistant_draw_min

-- Define proof statement
theorem lower_tap_used_earlier :
  let initial_amount := capacity
  let litres_when_midway := midway_capacity
  let litres_beer_left := beer_left_after_draw initial_amount assistant_draw_litres
  let additional_litres := litres_beer_left - litres_when_midway
  let time_earlier := additional_litres * upper_tap_rate
  time_earlier = 84 := 
by
  sorry

end lower_tap_used_earlier_l2095_209554


namespace tie_to_shirt_ratio_l2095_209560

-- Definitions for the conditions
def pants_cost : ℝ := 20
def shirt_cost : ℝ := 2 * pants_cost
def socks_cost : ℝ := 3
def r : ℝ := sorry -- This will be proved
def tie_cost : ℝ := r * shirt_cost
def uniform_cost : ℝ := pants_cost + shirt_cost + tie_cost + socks_cost

-- The total cost for five uniforms
def total_cost : ℝ := 5 * uniform_cost

-- The given total cost
def given_total_cost : ℝ := 355

-- The theorem to be proved
theorem tie_to_shirt_ratio :
  total_cost = given_total_cost → r = 1 / 5 := 
sorry

end tie_to_shirt_ratio_l2095_209560


namespace sum_of_digits_in_7_pow_1500_l2095_209570

-- Define the problem and conditions
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10
def sum_of_digits (n : ℕ) : ℕ := tens_digit n + units_digit n

theorem sum_of_digits_in_7_pow_1500 :
  sum_of_digits (7^1500) = 2 :=
by
  sorry

end sum_of_digits_in_7_pow_1500_l2095_209570


namespace ratio_of_wilted_roses_to_total_l2095_209576

-- Defining the conditions
def initial_roses := 24
def traded_roses := 12
def total_roses := initial_roses + traded_roses
def remaining_roses_after_second_night := 9
def roses_before_second_night := remaining_roses_after_second_night * 2
def wilted_roses_after_first_night := total_roses - roses_before_second_night
def ratio_wilted_to_total := wilted_roses_after_first_night / total_roses

-- Proving the ratio of wilted flowers to the total number of flowers after the first night is 1:2
theorem ratio_of_wilted_roses_to_total :
  ratio_wilted_to_total = (1/2) := by
  sorry

end ratio_of_wilted_roses_to_total_l2095_209576


namespace collinear_vector_l2095_209595

theorem collinear_vector (c R : ℝ) (A B : ℝ × ℝ) (hA: A.1 ^ 2 + A.2 ^ 2 = R ^ 2) (hB: B.1 ^ 2 + B.2 ^ 2 = R ^ 2) 
                         (h_line_A: 2 * A.1 + A.2 = c) (h_line_B: 2 * B.1 + B.2 = c) :
                         ∃ k : ℝ, (4, 2) = (k * (A.1 + B.1), k * (A.2 + B.2)) :=
sorry

end collinear_vector_l2095_209595


namespace average_number_of_stickers_per_album_is_correct_l2095_209524

def average_stickers_per_album (albums : List ℕ) (n : ℕ) : ℚ := (albums.sum : ℚ) / n

theorem average_number_of_stickers_per_album_is_correct :
  average_stickers_per_album [5, 7, 9, 14, 19, 12, 26, 18, 11, 15] 10 = 13.6 := 
by
  sorry

end average_number_of_stickers_per_album_is_correct_l2095_209524


namespace find_cosine_l2095_209590
open Real

noncomputable def alpha (α : ℝ) : Prop := 0 < α ∧ α < π / 2 ∧ sin α = 3 / 5

theorem find_cosine (α : ℝ) (h : alpha α) :
  cos (π - α / 2) = - (3 * sqrt 10) / 10 :=
by sorry

end find_cosine_l2095_209590


namespace problem_part1_problem_part2_l2095_209533

open Real

noncomputable def f (x : ℝ) := (sqrt 3) * sin x * cos x - (1 / 2) * cos (2 * x)

theorem problem_part1 : 
  (∀ x : ℝ, -1 ≤ f x) ∧ 
  (∃ T : ℝ, (T > 0) ∧ ∀ x : ℝ, f (x + T) = f x ∧ T = π) := 
sorry

theorem problem_part2 (C A B c : ℝ) :
  (f C = 1) → 
  (B = π / 6) → 
  (c = 2 * sqrt 3) → 
  ∃ b : ℝ, ∃ area : ℝ, b = 2 ∧ area = (1 / 2) * b * c * sin A ∧ area = 2 * sqrt 3 := 
sorry

end problem_part1_problem_part2_l2095_209533


namespace non_defective_probability_l2095_209514

theorem non_defective_probability :
  let p_B := 0.03
  let p_C := 0.01
  let p_def := p_B + p_C
  let p_non_def := 1 - p_def
  p_non_def = 0.96 :=
by
  let p_B := 0.03
  let p_C := 0.01
  let p_def := p_B + p_C
  let p_non_def := 1 - p_def
  sorry

end non_defective_probability_l2095_209514


namespace cost_price_USD_l2095_209509

-- Assume the conditions in Lean as given:
variable {C_USD : ℝ}

def condition1 (C_USD : ℝ) : Prop := 0.9 * C_USD + 200 = 1.04 * C_USD

theorem cost_price_USD (h : condition1 C_USD) : C_USD = 200 / 0.14 :=
by
  sorry

end cost_price_USD_l2095_209509


namespace find_p_l2095_209563

-- Define the coordinates as given in the problem
def Q : ℝ × ℝ := (0, 15)
def A : ℝ × ℝ := (3, 15)
def B : ℝ × ℝ := (15, 0)
def O : ℝ × ℝ := (0, 0)
def C (p : ℝ) : ℝ × ℝ := (0, p)

-- Defining the function to calculate area of triangle given three points
def area_of_triangle (P1 P2 P3 : ℝ × ℝ) : ℝ :=
  0.5 * abs (P1.fst * (P2.snd - P3.snd) + P2.fst * (P3.snd - P1.snd) + P3.fst * (P1.snd - P2.snd))

-- The statement we need to prove
theorem find_p :
  ∃ p : ℝ, area_of_triangle A B (C p) = 42 ∧ p = 11.75 :=
by
  sorry

end find_p_l2095_209563


namespace part1_part2_l2095_209564

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

-- Statement for part 1
theorem part1 (a b : ℝ) (h1 : f a b (-1) = 0) (h2 : f a b 3 = 0) (h3 : a ≠ 0) :
  a = -1 ∧ b = 4 :=
sorry

-- Statement for part 2
theorem part2 (a b : ℝ) (h1 : f a b 1 = 2) (h2 : a + b = 1) (h3 : a > 0) (h4 : b > 0) :
  (∀ x > 0, 1 / a + 4 / b ≥ 9) :=
sorry

end part1_part2_l2095_209564


namespace chocolate_bar_cost_l2095_209542

theorem chocolate_bar_cost (total_bars : ℕ) (sold_bars : ℕ) (total_money : ℕ) (cost : ℕ) 
  (h1 : total_bars = 13)
  (h2 : sold_bars = total_bars - 4)
  (h3 : total_money = 18)
  (h4 : total_money = sold_bars * cost) :
  cost = 2 :=
by sorry

end chocolate_bar_cost_l2095_209542


namespace average_income_l2095_209575

theorem average_income (income1 income2 income3 income4 income5 : ℝ)
    (h1 : income1 = 600) (h2 : income2 = 250) (h3 : income3 = 450) (h4 : income4 = 400) (h5 : income5 = 800) :
    (income1 + income2 + income3 + income4 + income5) / 5 = 500 := by
    sorry

end average_income_l2095_209575


namespace megan_files_in_folder_l2095_209555

theorem megan_files_in_folder :
  let initial_files := 93.0
  let added_files := 21.0
  let total_files := initial_files + added_files
  let total_folders := 14.25
  (total_files / total_folders) = 8.0 :=
by
  let initial_files := 93.0
  let added_files := 21.0
  let total_files := initial_files + added_files
  let total_folders := 14.25
  have h1 : total_files = initial_files + added_files := rfl
  have h2 : total_files = 114.0 := by sorry -- 93.0 + 21.0 = 114.0
  have h3 : total_files / total_folders = 8.0 := by sorry -- 114.0 / 14.25 = 8.0
  exact h3

end megan_files_in_folder_l2095_209555


namespace carol_remaining_distance_l2095_209535

def fuel_efficiency : ℕ := 25 -- miles per gallon
def gas_tank_capacity : ℕ := 18 -- gallons
def distance_to_home : ℕ := 350 -- miles

def total_distance_on_full_tank : ℕ := fuel_efficiency * gas_tank_capacity
def distance_after_home : ℕ := total_distance_on_full_tank - distance_to_home

theorem carol_remaining_distance :
  distance_after_home = 100 :=
sorry

end carol_remaining_distance_l2095_209535


namespace geom_series_sum_correct_l2095_209594

noncomputable def geometric_series_sum (b1 r : ℚ) (n : ℕ) : ℚ :=
b1 * (1 - r ^ n) / (1 - r)

theorem geom_series_sum_correct :
  geometric_series_sum (3/4) (3/4) 15 = 3177905751 / 1073741824 := by
sorry

end geom_series_sum_correct_l2095_209594


namespace probability_non_adjacent_sum_l2095_209503

-- Definitions and conditions from the problem
def total_trees := 13
def maple_trees := 4
def oak_trees := 3
def birch_trees := 6

-- Total possible arrangements of 13 trees
def total_arrangements := Nat.choose total_trees birch_trees

-- Number of ways to arrange birch trees with no two adjacent
def favorable_arrangements := Nat.choose (maple_trees + oak_trees + 1) birch_trees

-- Probability calculation
def probability_non_adjacent := (favorable_arrangements : ℚ) / (total_arrangements : ℚ)

-- This value should be simplified to form m/n in lowest terms
def fraction_part_m := 7
def fraction_part_n := 429

-- Verify m + n
def sum_m_n := fraction_part_m + fraction_part_n

-- Check that sum_m_n is equal to 436
theorem probability_non_adjacent_sum :
  sum_m_n = 436 := by {
    -- Placeholder proof
    sorry
}

end probability_non_adjacent_sum_l2095_209503


namespace number_of_BMWs_sold_l2095_209504

theorem number_of_BMWs_sold (total_cars : ℕ) (Audi_percent Toyota_percent Acura_percent Ford_percent : ℝ)
  (h_total_cars : total_cars = 250) 
  (h_percentages : Audi_percent = 0.10 ∧ Toyota_percent = 0.20 ∧ Acura_percent = 0.15 ∧ Ford_percent = 0.25) :
  ∃ (BMWs_sold : ℕ), BMWs_sold = 75 := 
by
  sorry

end number_of_BMWs_sold_l2095_209504


namespace no_snow_probability_l2095_209581

noncomputable def probability_of_no_snow (p_snow : ℚ) : ℚ :=
  1 - p_snow

theorem no_snow_probability : probability_of_no_snow (2/5) = 3/5 :=
  sorry

end no_snow_probability_l2095_209581


namespace domain_not_neg1_increasing_on_neg1_infty_min_max_on_3_5_l2095_209538

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

namespace f_props

theorem domain_not_neg1 : ∀ x : ℝ, x ≠ -1 ↔ x ∈ {y | y ≠ -1} :=
by simp [f]

theorem increasing_on_neg1_infty : ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → -1 < x2 → f x1 < f x2 :=
sorry

theorem min_max_on_3_5 : (∀ y : ℝ, y = f 3 → y = 5 / 4) ∧ (∀ y : ℝ, y = f 5 → y = 3 / 2) :=
sorry

end f_props

end domain_not_neg1_increasing_on_neg1_infty_min_max_on_3_5_l2095_209538


namespace profit_amount_calc_l2095_209505

-- Define the conditions as hypotheses
variables (SP : ℝ) (profit_percent : ℝ) (cost_price profit_amount : ℝ)

-- Given conditions
axiom selling_price : SP = 900
axiom profit_percentage : profit_percent = 50
axiom profit_formula : profit_amount = 0.5 * cost_price
axiom selling_price_formula : SP = cost_price + profit_amount

-- The theorem to be proven
theorem profit_amount_calc : profit_amount = 300 :=
by
  sorry

end profit_amount_calc_l2095_209505


namespace game_positions_l2095_209537

def spots := ["top-left", "top-right", "bottom-right", "bottom-left"]
def segments := ["top-left", "top-middle-left", "top-middle-right", "top-right", "right-top", "right-middle-top", "right-middle-bottom", "right-bottom", "bottom-right", "bottom-middle-right", "bottom-middle-left", "bottom-left", "left-top", "left-middle-top", "left-middle-bottom", "left-bottom"]

def cat_position_after_moves (n : Nat) : String :=
  spots.get! (n % 4)

def mouse_position_after_moves (n : Nat) : String :=
  segments.get! ((12 - (n % 12)) % 12)

theorem game_positions :
  cat_position_after_moves 359 = "bottom-right" ∧ 
  mouse_position_after_moves 359 = "left-middle-bottom" :=
by
  sorry

end game_positions_l2095_209537


namespace appropriate_sampling_method_l2095_209591

/--
Given there are 40 products in total, consisting of 10 first-class products,
25 second-class products, and 5 defective products, if we need to select
8 products for quality analysis, then the appropriate sampling method is
the stratified sampling method.
-/
theorem appropriate_sampling_method
  (total_products : ℕ)
  (first_class_products : ℕ)
  (second_class_products : ℕ)
  (defective_products : ℕ)
  (selected_products : ℕ)
  (stratified_sampling : ℕ → ℕ → ℕ → ℕ → Prop) :
  total_products = 40 →
  first_class_products = 10 →
  second_class_products = 25 →
  defective_products = 5 →
  selected_products = 8 →
  stratified_sampling total_products first_class_products second_class_products defective_products →
  stratified_sampling total_products first_class_products second_class_products defective_products :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end appropriate_sampling_method_l2095_209591


namespace total_hours_watched_l2095_209500

theorem total_hours_watched (Monday Tuesday Wednesday Thursday Friday : ℕ) (hMonday : Monday = 12) (hTuesday : Tuesday = 4) (hWednesday : Wednesday = 6) (hThursday : Thursday = (Monday + Tuesday + Wednesday) / 2) (hFriday : Friday = 19) :
  Monday + Tuesday + Wednesday + Thursday + Friday = 52 := by
  sorry

end total_hours_watched_l2095_209500


namespace correct_number_of_six_letter_words_l2095_209518

def number_of_six_letter_words (alphabet_size : ℕ) : ℕ :=
  alphabet_size ^ 4

theorem correct_number_of_six_letter_words :
  number_of_six_letter_words 26 = 456976 :=
by
  -- We write 'sorry' to omit the detailed proof.
  sorry

end correct_number_of_six_letter_words_l2095_209518


namespace max_volume_at_6_l2095_209513

noncomputable def volume (x : ℝ) : ℝ :=
  x * (36 - 2 * x)^2

theorem max_volume_at_6 :
  ∃ x : ℝ, (0 < x) ∧ (x < 18) ∧ 
  (∀ y : ℝ, (0 < y) ∧ (y < 18) → volume y ≤ volume 6) :=
by
  sorry

end max_volume_at_6_l2095_209513


namespace find_k_l2095_209532

theorem find_k : 
  ∃ x y k : ℝ, y = 7 * x - 2 ∧ y = -3 * x + 14 ∧ y = 4 * x + k ∧ k = 2.8 :=
by
  sorry

end find_k_l2095_209532


namespace combined_distance_is_12_l2095_209536

-- Define the distances the two ladies walked
def distance_second_lady : ℝ := 4
def distance_first_lady := 2 * distance_second_lady

-- Define the combined total distance
def combined_distance := distance_first_lady + distance_second_lady

-- Statement of the problem as a proof goal in Lean
theorem combined_distance_is_12 : combined_distance = 12 :=
by
  -- Definitions required for the proof
  let second := distance_second_lady
  let first := distance_first_lady
  let total := combined_distance
  
  -- Insert the necessary calculations and proof steps here
  -- Conclude with the desired result
  sorry

end combined_distance_is_12_l2095_209536


namespace positive_number_equals_seven_l2095_209512

theorem positive_number_equals_seven (x : ℝ) (h_pos : x > 0) (h_eq : x - 4 = 21 / x) : x = 7 :=
sorry

end positive_number_equals_seven_l2095_209512


namespace square_three_times_side_length_l2095_209551

theorem square_three_times_side_length (a : ℝ) : 
  ∃ s, s = a * Real.sqrt 3 ∧ s ^ 2 = 3 * a ^ 2 := 
by 
  sorry

end square_three_times_side_length_l2095_209551


namespace pencils_in_total_l2095_209515

theorem pencils_in_total
  (rows : ℕ) (pencils_per_row : ℕ) (total_pencils : ℕ)
  (h1 : rows = 14)
  (h2 : pencils_per_row = 11)
  (h3 : total_pencils = rows * pencils_per_row) :
  total_pencils = 154 :=
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end pencils_in_total_l2095_209515


namespace sphere_surface_area_from_volume_l2095_209571

theorem sphere_surface_area_from_volume 
  (V : ℝ) (h : V = 72 * Real.pi) :
  ∃ (A : ℝ), A = 36 * Real.pi * 2^(2/3) :=
by
  sorry

end sphere_surface_area_from_volume_l2095_209571


namespace solve_rational_equation_l2095_209510

theorem solve_rational_equation : 
  ∀ x : ℝ, x ≠ 1 -> (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1) → 
  (x = 6 ∨ x = -2) :=
by
  intro x
  intro h
  intro h_eq
  sorry

end solve_rational_equation_l2095_209510


namespace sum_of_constants_eq_17_l2095_209589

theorem sum_of_constants_eq_17
  (x y : ℝ)
  (a b c d : ℕ)
  (ha : a = 6)
  (hb : b = 2)
  (hc : c = 3)
  (hd : d = 3)
  (h1 : x + y = 4)
  (h2 : 3 * x * y = 4)
  (h3 : x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d) :
  a + b + c + d = 17 :=
sorry

end sum_of_constants_eq_17_l2095_209589


namespace toys_per_box_l2095_209531

theorem toys_per_box (number_of_boxes total_toys : ℕ) (h₁ : number_of_boxes = 4) (h₂ : total_toys = 32) :
  total_toys / number_of_boxes = 8 :=
by
  sorry

end toys_per_box_l2095_209531


namespace find_m_eccentricity_l2095_209527

theorem find_m_eccentricity :
  (∃ m : ℝ, (m > 0) ∧ (∃ c : ℝ, (c = 4 - m ∧ c = (1 / 2) * 2) ∨ (c = m - 4 ∧ c = (1 / 2) * 2)) ∧
  (m = 3 ∨ m = 16 / 3)) :=
sorry

end find_m_eccentricity_l2095_209527


namespace people_behind_yuna_l2095_209580

theorem people_behind_yuna (total_people : ℕ) (people_in_front : ℕ) (yuna : ℕ)
  (h1 : total_people = 7) (h2 : people_in_front = 2) (h3 : yuna = 1) :
  total_people - people_in_front - yuna = 4 :=
by
  sorry

end people_behind_yuna_l2095_209580


namespace weight_units_correct_l2095_209592

-- Definitions of weights
def weight_peanut_kernel := 1 -- gram
def weight_truck_capacity := 8 -- ton
def weight_xiao_ming := 30 -- kilogram
def weight_basketball := 580 -- gram

-- Proof that the weights have correct units
theorem weight_units_correct :
  (weight_peanut_kernel = 1 ∧ weight_truck_capacity = 8 ∧ weight_xiao_ming = 30 ∧ weight_basketball = 580) :=
by {
  sorry
}

end weight_units_correct_l2095_209592


namespace inscribed_circle_radius_is_correct_l2095_209546

noncomputable def radius_of_inscribed_circle (base height : ℝ) : ℝ := sorry

theorem inscribed_circle_radius_is_correct :
  radius_of_inscribed_circle 20 24 = 120 / 13 := sorry

end inscribed_circle_radius_is_correct_l2095_209546


namespace log2_monotone_l2095_209547

theorem log2_monotone (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a > b) ↔ (Real.log a / Real.log 2 > Real.log b / Real.log 2) :=
sorry

end log2_monotone_l2095_209547


namespace part1_part2_l2095_209593

noncomputable def A (x : ℝ) (k : ℝ) := -2 * x ^ 2 - (k - 1) * x + 1
noncomputable def B (x : ℝ) := -2 * (x ^ 2 - x + 2)

-- Part 1: If A is a quadratic binomial, then the value of k is 1
theorem part1 (x : ℝ) (k : ℝ) (h : ∀ x, A x k ≠ 0) : k = 1 :=
sorry

-- Part 2: When k = -1, C + 2A = B, then C = 2x^2 - 2x - 6
theorem part2 (x : ℝ) (C : ℝ → ℝ) (h1 : k = -1) (h2 : ∀ x, C x + 2 * A x k = B x) : (C x = 2 * x ^ 2 - 2 * x - 6) :=
sorry

end part1_part2_l2095_209593


namespace fewest_coach_handshakes_l2095_209543

theorem fewest_coach_handshakes (n k : ℕ) (h1 : (n * (n - 1)) / 2 + k = 281) : k = 5 :=
sorry

end fewest_coach_handshakes_l2095_209543


namespace sqrt_450_simplified_l2095_209528

theorem sqrt_450_simplified :
  (∀ {x : ℕ}, 9 = x * x) →
  (∀ {x : ℕ}, 25 = x * x) →
  (450 = 25 * 18) →
  (18 = 9 * 2) →
  Real.sqrt 450 = 15 * Real.sqrt 2 :=
by
  intros h9 h25 h450 h18
  sorry

end sqrt_450_simplified_l2095_209528


namespace percentage_assigned_exam_l2095_209557

-- Define the conditions of the problem
def total_students : ℕ := 100
def average_assigned : ℝ := 0.55
def average_makeup : ℝ := 0.95
def average_total : ℝ := 0.67

-- Define the proof problem statement
theorem percentage_assigned_exam :
  ∃ (x : ℝ), (x / total_students) * average_assigned + ((total_students - x) / total_students) * average_makeup = average_total ∧ x = 70 :=
by
  sorry

end percentage_assigned_exam_l2095_209557


namespace find_c_interval_l2095_209545

theorem find_c_interval (c : ℚ) : 
  (c / 4 ≤ 3 + c ∧ 3 + c < -3 * (1 + c)) ↔ (-4 ≤ c ∧ c < -3 / 2) := 
by 
  sorry

end find_c_interval_l2095_209545


namespace jill_total_watch_time_l2095_209540

theorem jill_total_watch_time :
  ∀ (length_first_show length_second_show total_watch_time : ℕ),
    length_first_show = 30 →
    length_second_show = 4 * length_first_show →
    total_watch_time = length_first_show + length_second_show →
    total_watch_time = 150 :=
by
  sorry

end jill_total_watch_time_l2095_209540


namespace perimeter_is_32_l2095_209508

-- Define the side lengths of the triangle
def a : ℕ := 13
def b : ℕ := 9
def c : ℕ := 10

-- Definition of the perimeter of the triangle
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- Theorem stating the perimeter is 32
theorem perimeter_is_32 : perimeter a b c = 32 :=
by
  sorry

end perimeter_is_32_l2095_209508


namespace ratio_of_friends_l2095_209597

theorem ratio_of_friends (friends_in_classes friends_in_clubs : ℕ) (thread_per_keychain total_thread : ℕ) 
  (h1 : thread_per_keychain = 12) (h2 : friends_in_classes = 6) (h3 : total_thread = 108)
  (keychains_total : total_thread / thread_per_keychain = 9) 
  (keychains_clubs : (total_thread / thread_per_keychain) - friends_in_classes = friends_in_clubs) :
  friends_in_clubs / friends_in_classes = 1 / 2 :=
by
  sorry

end ratio_of_friends_l2095_209597


namespace CorrectChoice_l2095_209599

open Classical

-- Define the integer n
variable (n : ℤ)

-- Define proposition p: 2n - 1 is always odd
def p : Prop := ∃ k : ℤ, 2 * k + 1 = 2 * n - 1

-- Define proposition q: 2n + 1 is always even
def q : Prop := ∃ k : ℤ, 2 * k = 2 * n + 1

-- The theorem we want to prove
theorem CorrectChoice : (p n ∨ q n) :=
by
  sorry

end CorrectChoice_l2095_209599


namespace cube_root_opposite_zero_l2095_209587

theorem cube_root_opposite_zero (x : ℝ) (h : x^(1/3) = -x) : x = 0 :=
sorry

end cube_root_opposite_zero_l2095_209587


namespace f_neg1_gt_f_1_l2095_209519

-- Definition of the function f and its properties.
variable {f : ℝ → ℝ}
variable (df : Differentiable ℝ f)
variable (eq_f : ∀ x : ℝ, f x = x^2 + 2 * x * f' 2)

-- The problem statement to prove f(-1) > f(1).
theorem f_neg1_gt_f_1 (h_deriv : ∀ x : ℝ, deriv f x = 2 * x - 8):
  f (-1) > f 1 :=
by
  sorry

end f_neg1_gt_f_1_l2095_209519


namespace even_function_expression_l2095_209578

theorem even_function_expression (f : ℝ → ℝ)
  (h₀ : ∀ x, x ≥ 0 → f x = x^2 - 3 * x + 4)
  (h_even : ∀ x, f x = f (-x)) :
  ∀ x, f x = if x < 0 then x^2 + 3 * x + 4 else x^2 - 3 * x + 4 :=
by {
  sorry
}

end even_function_expression_l2095_209578


namespace smallest_period_find_a_l2095_209501

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.cos x ^ 2 + Real.sqrt 3 * Real.sin (2 * x) + a

theorem smallest_period (a : ℝ) : 
  ∃ T > 0, ∀ x, f x a = f (x + T) a ∧ (∀ T' > 0, (∀ x, f x a = f (x + T') a) → T ≤ T') :=
by
  sorry

theorem find_a :
  ∃ a : ℝ, (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ 4) ∧ (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x a = 4) ∧ a = 1 :=
by
  sorry

end smallest_period_find_a_l2095_209501


namespace number_of_small_spheres_l2095_209541

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem number_of_small_spheres
  (d_large : ℝ) (d_small : ℝ)
  (h1 : d_large = 6) (h2 : d_small = 2) :
  let V_large := volume_of_sphere (d_large / 2)
  let V_small := volume_of_sphere (d_small / 2)
  V_large / V_small = 27 := 
by
  sorry

end number_of_small_spheres_l2095_209541


namespace rectangle_area_l2095_209552

-- Definitions
variables {height length : ℝ} (h : height = length / 2)
variables {area perimeter : ℝ} (a : area = perimeter)

-- Problem statement
theorem rectangle_area : ∃ h : ℝ, ∃ l : ℝ, ∃ area : ℝ, 
  (l = 2 * h) ∧ (area = l * h) ∧ (area = 2 * (l + h)) ∧ (area = 18) :=
sorry

end rectangle_area_l2095_209552


namespace border_area_correct_l2095_209583

-- Define the dimensions of the photograph
def photograph_height : ℕ := 12
def photograph_width : ℕ := 15

-- Define the width of the border
def border_width : ℕ := 3

-- Define the area of the photograph
def area_photograph : ℕ := photograph_height * photograph_width

-- Define the total dimensions including the frame
def total_height : ℕ := photograph_height + 2 * border_width
def total_width : ℕ := photograph_width + 2 * border_width

-- Define the area of the framed area
def area_framed : ℕ := total_height * total_width

-- Define the area of the border
def area_border : ℕ := area_framed - area_photograph

theorem border_area_correct : area_border = 198 := by
  sorry

end border_area_correct_l2095_209583


namespace num_7_digit_integers_correct_l2095_209574

-- Define the number of choices for each digit
def first_digit_choices : ℕ := 9
def other_digit_choices : ℕ := 10

-- Define the number of 7-digit positive integers
def num_7_digit_integers : ℕ := first_digit_choices * other_digit_choices^6

-- State the theorem to prove
theorem num_7_digit_integers_correct : num_7_digit_integers = 9000000 :=
by
  sorry

end num_7_digit_integers_correct_l2095_209574


namespace win_sector_area_l2095_209549

theorem win_sector_area (r : ℝ) (p_win : ℝ) (area_total : ℝ) 
  (h1 : r = 8)
  (h2 : p_win = 3 / 8)
  (h3 : area_total = π * r^2) :
  ∃ area_win, area_win = 24 * π ∧ area_win = p_win * area_total :=
by
  sorry

end win_sector_area_l2095_209549


namespace volume_ratio_john_emma_l2095_209579

theorem volume_ratio_john_emma (r_J h_J r_E h_E : ℝ) (diam_J diam_E : ℝ)
  (h_diam_J : diam_J = 8) (h_r_J : r_J = diam_J / 2) (h_h_J : h_J = 15)
  (h_diam_E : diam_E = 10) (h_r_E : r_E = diam_E / 2) (h_h_E : h_E = 12) :
  (π * r_J^2 * h_J) / (π * r_E^2 * h_E) = 4 / 5 := by
  sorry

end volume_ratio_john_emma_l2095_209579


namespace son_age_l2095_209553

variable (F S : ℕ)
variable (h₁ : F = 3 * S)
variable (h₂ : F - 8 = 4 * (S - 8))

theorem son_age : S = 24 := 
by 
  sorry

end son_age_l2095_209553


namespace find_sum_l2095_209598

variables (a b c d : ℕ)

axiom h1 : 6 * a + 2 * b = 3848
axiom h2 : 6 * c + 3 * d = 4410
axiom h3 : a + 3 * b + 2 * d = 3080

theorem find_sum : a + b + c + d = 1986 :=
by
  sorry

end find_sum_l2095_209598


namespace bob_second_third_lap_time_l2095_209544

theorem bob_second_third_lap_time :
  ∀ (lap_length : ℕ) (first_lap_time : ℕ) (average_speed : ℕ),
  lap_length = 400 →
  first_lap_time = 70 →
  average_speed = 5 →
  ∃ (second_third_lap_time : ℕ), second_third_lap_time = 85 :=
by
  intros lap_length first_lap_time average_speed lap_length_eq first_lap_time_eq average_speed_eq
  sorry

end bob_second_third_lap_time_l2095_209544


namespace least_number_divisible_increased_by_seven_l2095_209522

theorem least_number_divisible_increased_by_seven : 
  ∃ n : ℕ, (∀ k ∈ [24, 32, 36, 54], (n + 7) % k = 0) ∧ n = 857 := 
by
  sorry

end least_number_divisible_increased_by_seven_l2095_209522


namespace problem_l2095_209520

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

theorem problem (h₁ : f a b c 0 = f a b c 4) (h₂ : f a b c 4 > f a b c 1) : a > 0 ∧ 4 * a + b = 0 :=
by 
  sorry

end problem_l2095_209520


namespace ben_points_l2095_209548

theorem ben_points (B : ℕ) 
  (h1 : 42 = B + 21) : B = 21 := 
by 
-- Proof can be filled in here
sorry

end ben_points_l2095_209548


namespace probability_T_H_E_equal_L_A_V_A_l2095_209588

noncomputable def probability_condition : ℚ :=
  -- Number of total sample space (3^6)
  (3 ^ 6 : ℚ)

noncomputable def favorable_events_0 : ℚ :=
  -- Number of favorable outcomes where 𝑻 ⋅ 𝑯 ⋅ 𝑬 is 0 and 𝑳 ⋅ 𝑨 ⋅ 𝑽 ⋅ 𝑨 is 0
  26 * 19

noncomputable def favorable_events_1 : ℚ :=
  -- Number of favorable outcomes where 𝑻 ⋅ 𝑯 ⋅ 𝑬 is 1 and 𝑳 ⋅ 𝑨 ⋅ 𝑽 ⋅ 𝑨 is 1
  1

noncomputable def total_favorable_events : ℚ :=
  favorable_events_0 + favorable_events_1

theorem probability_T_H_E_equal_L_A_V_A :
  (total_favorable_events / probability_condition) = 55 / 81 :=
sorry

end probability_T_H_E_equal_L_A_V_A_l2095_209588


namespace union_A_B_equals_x_lt_3_l2095_209550

theorem union_A_B_equals_x_lt_3 :
  let A := { x : ℝ | 3 - x > 0 ∧ x + 2 > 0 }
  let B := { x : ℝ | 3 > 2*x - 1 }
  A ∪ B = { x : ℝ | x < 3 } :=
by
  sorry

end union_A_B_equals_x_lt_3_l2095_209550


namespace solve_problem_l2095_209530

-- Define the constants c and d
variables (c d : ℝ)

-- Define the conditions of the problem
def condition1 : Prop := 
  (∀ x : ℝ, (x + c) * (x + d) * (x + 15) = 0 ↔ x = -c ∨ x = -d ∨ x = -15) ∧
  -4 ≠ -c ∧ -4 ≠ -d ∧ -4 ≠ -15

def condition2 : Prop := 
  (∀ x : ℝ, (x + 3 * c) * (x + 4) * (x + 9) = 0 ↔ x = -4) ∧
  d ≠ -4 ∧ d ≠ -15

-- We need to prove this final result under the given conditions
theorem solve_problem (h1 : condition1 c d) (h2 : condition2 c d) : 100 * c + d = -291 := 
  sorry

end solve_problem_l2095_209530


namespace candy_problem_l2095_209516

theorem candy_problem
  (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999)
  (h3 : n + 7 ≡ 0 [MOD 9])
  (h4 : n - 9 ≡ 0 [MOD 6]) :
  n = 101 :=
sorry

end candy_problem_l2095_209516


namespace equation_of_parallel_line_l2095_209577

theorem equation_of_parallel_line (c : ℕ) :
  (∃ c, x + 2 * y + c = 0) ∧ (1 + 2 * 1 + c = 0) -> x + 2 * y - 3 = 0 :=
by 
  sorry

end equation_of_parallel_line_l2095_209577


namespace solve_x_l2095_209568

theorem solve_x (x : ℝ) (h : (x + 1) ^ 2 = 9) : x = 2 ∨ x = -4 :=
sorry

end solve_x_l2095_209568


namespace smallest_x_l2095_209573

noncomputable def f (x : ℝ) : ℝ :=
if 1 ≤ x ∧ x ≤ 4 then x^2 - 4 * x + 5 else sorry

theorem smallest_x (x : ℝ) (h₁ : ∀ x > 0, f (4 * x) = 4 * f x)
  (h₂ : ∀ x, (1 ≤ x ∧ x ≤ 4) → f x = x^2 - 4 * x + 5) :
  ∃ x₀, x₀ > 0 ∧ f x₀ = 1024 ∧ (∀ y, y > 0 ∧ f y = 1024 → y ≥ x₀) :=
sorry

end smallest_x_l2095_209573


namespace math_problem_l2095_209569

noncomputable def compute_value (c d : ℝ) : ℝ := 100 * c + d

-- Problem statement as a theorem
theorem math_problem
  (c d : ℝ)
  (H1 : ∀ x : ℝ, (x + c) * (x + d) * (x + 10) = 0 → x = -c ∨ x = -d ∨ x = -10)
  (H2 : ∀ x : ℝ, (x + 3 * c) * (x + 5) * (x + 8) = 0 → (x = -4 ∧ ∀ y : ℝ, y ≠ -4 → (y + d) * (y + 10) ≠ 0))
  (H3 : c ≠ 4 / 3 → 3 * c = d ∨ 3 * c = 10) :
  compute_value c d = 141.33 :=
by sorry

end math_problem_l2095_209569


namespace tan_double_angle_l2095_209566

theorem tan_double_angle (α β : ℝ) (h1 : Real.tan (α + β) = 7) (h2 : Real.tan (α - β) = 1) : 
  Real.tan (2 * α) = -4/3 :=
by
  sorry

end tan_double_angle_l2095_209566


namespace k_less_than_zero_l2095_209525

variable (k : ℝ)

def function_decreases (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂

theorem k_less_than_zero (h : function_decreases (λ x => k * x - 5)) : k < 0 :=
sorry

end k_less_than_zero_l2095_209525


namespace calculate_division_of_powers_l2095_209556

theorem calculate_division_of_powers (a : ℝ) : a^8 / a^2 = a^6 :=
by sorry

end calculate_division_of_powers_l2095_209556


namespace range_of_t_sum_of_squares_l2095_209526

-- Define the conditions and the problem statement in Lean

variables (a b c t x : ℝ)
variables (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)
variables (ineq1 : |x + 1| - |x - 2| ≥ |t - 1| + t)
variables (sum_pos : 2 * a + b + c = 2)

theorem range_of_t :
  (∃ x, |x + 1| - |x - 2| ≥ |t - 1| + t) → t ≤ 2 :=
sorry

theorem sum_of_squares :
  2 * a + b + c = 2 → 0 < a → 0 < b → 0 < c → a^2 + b^2 + c^2 ≥ 2 / 3 :=
sorry

end range_of_t_sum_of_squares_l2095_209526


namespace length_GH_l2095_209584

def length_AB : ℕ := 11
def length_FE : ℕ := 13
def length_CD : ℕ := 5

theorem length_GH : length_AB + length_CD + length_FE = 29 :=
by
  refine rfl -- This will unroll the constants and perform arithmetic

end length_GH_l2095_209584


namespace solve_for_x_l2095_209567

theorem solve_for_x (x y : ℝ) 
  (h1 : 3 * x - 2 * y = 8) 
  (h2 : 2 * x + 3 * y = 11) :
  x = 46 / 13 :=
by
  sorry

end solve_for_x_l2095_209567


namespace andrew_total_days_l2095_209559

noncomputable def hours_per_day : ℝ := 2.5
noncomputable def total_hours : ℝ := 7.5

theorem andrew_total_days : total_hours / hours_per_day = 3 := 
by 
  sorry

end andrew_total_days_l2095_209559


namespace find_b_and_area_l2095_209558

open Real

variables (a c : ℝ) (A b S : ℝ)

theorem find_b_and_area 
  (h1 : a = sqrt 7) 
  (h2 : c = 3) 
  (h3 : A = π / 3) :
  (b = 1 ∨ b = 2) ∧ (S = 3 * sqrt 3 / 4 ∨ S = 3 * sqrt 3 / 2) := 
by sorry

end find_b_and_area_l2095_209558


namespace problem_statement_l2095_209582

def g (x : ℝ) : ℝ :=
  x^2 - 5 * x

theorem problem_statement (x : ℝ) :
  (g (g x) = g x) ↔ (x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1) :=
by
  sorry

end problem_statement_l2095_209582


namespace find_k_collinear_l2095_209511

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-2, -1)
def c : ℝ × ℝ := (1, 2)

theorem find_k_collinear : ∃ k : ℝ, (1 - 2 * k, 3 - k) = (-k, k) * c ∧ k = -1/3 :=
by
  sorry

end find_k_collinear_l2095_209511


namespace percentage_of_mortality_l2095_209596

theorem percentage_of_mortality
  (P : ℝ) -- The population size could be represented as a real number
  (affected_fraction : ℝ) (dead_fraction : ℝ)
  (h1 : affected_fraction = 0.15) -- 15% of the population is affected
  (h2 : dead_fraction = 0.08) -- 8% of the affected population died
: (affected_fraction * dead_fraction) * 100 = 1.2 :=
by
  sorry

end percentage_of_mortality_l2095_209596


namespace gcd_323_391_l2095_209517

theorem gcd_323_391 : Nat.gcd 323 391 = 17 := 
by sorry

end gcd_323_391_l2095_209517


namespace solution_set_inequality_f_solution_range_a_l2095_209539

-- Define the function f 
def f (x : ℝ) := |x + 1| + |x - 3|

-- Statement for question 1
theorem solution_set_inequality_f (x : ℝ) : f x < 6 ↔ -2 < x ∧ x < 4 :=
sorry

-- Statement for question 2
theorem solution_range_a (a : ℝ) (h : ∃ x : ℝ, f x = |a - 2|) : a ≥ 6 ∨ a ≤ -2 :=
sorry

end solution_set_inequality_f_solution_range_a_l2095_209539


namespace simplify_expression_l2095_209565

theorem simplify_expression : 
  (1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 1))) = 
    ((Real.sqrt 3 - 2 * Real.sqrt 5 - 1) * (-16 - 2 * Real.sqrt 3)) / 244 := 
  sorry

end simplify_expression_l2095_209565


namespace expected_score_is_6_l2095_209529

-- Define the probabilities of making a shot
def p : ℝ := 0.5

-- Define the scores for each scenario
def score_first_shot : ℝ := 8
def score_second_shot : ℝ := 6
def score_third_shot : ℝ := 4
def score_no_shot : ℝ := 0

-- Compute the expected value
def expected_score : ℝ :=
  p * score_first_shot +
  (1 - p) * p * score_second_shot +
  (1 - p) * (1 - p) * p * score_third_shot +
  (1 - p) * (1 - p) * (1 - p) * score_no_shot

theorem expected_score_is_6 : expected_score = 6 := by
  sorry

end expected_score_is_6_l2095_209529


namespace negation_of_universal_statement_l2095_209502

theorem negation_of_universal_statement :
  ¬ (∀ x : ℝ, x^2 ≤ 1) ↔ ∃ x : ℝ, x^2 > 1 :=
by
  sorry

end negation_of_universal_statement_l2095_209502


namespace total_votes_l2095_209521

theorem total_votes (V : ℕ) 
  (h1 : V * 45 / 100 + V * 25 / 100 + V * 15 / 100 + 180 + 50 = V) : 
  V = 1533 := 
by
  sorry

end total_votes_l2095_209521


namespace sum_max_min_on_interval_l2095_209586

-- Defining the function f
def f (x : ℝ) : ℝ := x + 2

-- The proof statement
theorem sum_max_min_on_interval : 
  let M := max (f 0) (f 4)
  let N := min (f 0) (f 4)
  M + N = 8 := by
  -- Placeholder for proof
  sorry

end sum_max_min_on_interval_l2095_209586
