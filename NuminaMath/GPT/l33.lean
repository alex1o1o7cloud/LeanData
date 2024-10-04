import Mathlib

namespace magic_shop_change_l33_33451

theorem magic_shop_change :
  (∀ (cloak : Type), ∃ price_gold price_silver1 change_gold1 price_silver2 change_gold2, 
  price_silver1 = 20 ∧ change_gold1 = 4 ∧ 
  price_silver2 = 15 ∧ change_gold2 = 1 ∧ 
  price_gold = 14 ∧ 
  ∀ change_silver, 
    (20 - 4) * change_silver = 15 - 1 → -- Relation derived from the conditions
    (14 - (15 - 1) * change_silver / (20 - 4)) * change_silver = 10) := 
sorry

end magic_shop_change_l33_33451


namespace fair_coin_flip_difference_l33_33204

theorem fair_coin_flip_difference :
  let P1 := 5 * (1 / 2)^5;
  let P2 := (1 / 2)^5;
  P1 - P2 = 9 / 32 :=
by
  sorry

end fair_coin_flip_difference_l33_33204


namespace area_of_room_l33_33961

def length : ℝ := 12
def width : ℝ := 8

theorem area_of_room : length * width = 96 :=
by sorry

end area_of_room_l33_33961


namespace osborn_friday_time_l33_33489

-- Conditions
def time_monday : ℕ := 2
def time_tuesday : ℕ := 4
def time_wednesday : ℕ := 3
def time_thursday : ℕ := 4
def old_average_time_per_day : ℕ := 3
def school_days_per_week : ℕ := 5

-- Total time needed to match old average
def total_time_needed : ℕ := old_average_time_per_day * school_days_per_week

-- Total time spent from Monday to Thursday
def time_spent_mon_to_thu : ℕ := time_monday + time_tuesday + time_wednesday + time_thursday

-- Goal: Find time on Friday
def time_friday : ℕ := total_time_needed - time_spent_mon_to_thu

theorem osborn_friday_time : time_friday = 2 :=
by
  sorry

end osborn_friday_time_l33_33489


namespace initial_people_per_column_l33_33816

theorem initial_people_per_column (P x : ℕ) (h1 : P = 16 * x) (h2 : P = 48 * 10) : x = 30 :=
by 
  sorry

end initial_people_per_column_l33_33816


namespace purchase_combinations_correct_l33_33114

theorem purchase_combinations_correct : 
  let headphones : ℕ := 9
  let mice : ℕ := 13
  let keyboards : ℕ := 5
  let sets_keyboard_mouse : ℕ := 4
  let sets_headphones_mouse : ℕ := 5
in 
  (sets_keyboard_mouse * headphones) + 
  (sets_headphones_mouse * keyboards) + 
  (headphones * mice * keyboards) = 646 := 
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let sets_keyboard_mouse := 4
  let sets_headphones_mouse := 5
  sorry

end purchase_combinations_correct_l33_33114


namespace solve_ineq_for_a_eq_0_values_of_a_l33_33805

theorem solve_ineq_for_a_eq_0 :
  ∀ x : ℝ, (|x + 2| - 3 * |x|) ≥ 0 ↔ (-1/2 <= x ∧ x <= 1) := 
by
  sorry

theorem values_of_a :
  ∀ x a : ℝ, (|x + 2| - 3 * |x|) ≥ a → (a ≤ 2) := 
by
  sorry

end solve_ineq_for_a_eq_0_values_of_a_l33_33805


namespace range_of_a_l33_33093

theorem range_of_a (a : ℝ) :
  ¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 ≤ 0 → -1 < a ∧ a < 3 :=
by
  intro h
  sorry

end range_of_a_l33_33093


namespace fencing_cost_is_correct_l33_33499

def length : ℕ := 60
def cost_per_meter : ℕ := 27 -- using the closest integer value to 26.50
def breadth (l : ℕ) : ℕ := l - 20
def perimeter (l b : ℕ) : ℕ := 2 * l + 2 * b
def total_cost (P : ℕ) (c : ℕ) : ℕ := P * c

theorem fencing_cost_is_correct :
  total_cost (perimeter length (breadth length)) cost_per_meter = 5300 :=
  sorry

end fencing_cost_is_correct_l33_33499


namespace number_of_books_in_library_l33_33349

def number_of_bookcases : ℕ := 28
def shelves_per_bookcase : ℕ := 6
def books_per_shelf : ℕ := 19

theorem number_of_books_in_library : number_of_bookcases * shelves_per_bookcase * books_per_shelf = 3192 :=
by
  sorry

end number_of_books_in_library_l33_33349


namespace intersection_points_form_line_l33_33170

theorem intersection_points_form_line :
  ∀ (x y : ℝ), ((x * y = 12) ∧ ((x^2 / 16) + (y^2 / 36) = 1)) →
  ∃ (x1 x2 : ℝ) (y1 y2 : ℝ), (x, y) = (x1, y1) ∨ (x, y) = (x2, y2) ∧ (x2 - x1) * (y2 - y1) = x1 * y1 - x2 * y2 :=
by
  sorry

end intersection_points_form_line_l33_33170


namespace probability_10_coins_at_most_3_heads_l33_33562

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l33_33562


namespace computer_multiplications_in_30_minutes_l33_33614

def multiplications_per_second : ℕ := 20000
def seconds_per_minute : ℕ := 60
def minutes : ℕ := 30
def total_seconds : ℕ := minutes * seconds_per_minute
def expected_multiplications : ℕ := 36000000

theorem computer_multiplications_in_30_minutes :
  multiplications_per_second * total_seconds = expected_multiplications :=
by
  sorry

end computer_multiplications_in_30_minutes_l33_33614


namespace range_of_t_l33_33059

variable (f : ℝ → ℝ) (t : ℝ)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem range_of_t {f : ℝ → ℝ} {t : ℝ} 
  (Hodd : is_odd f) 
  (Hperiodic : ∀ x, f (x + 5 / 2) = -1 / f x) 
  (Hf1 : f 1 ≥ 1) 
  (Hf2014 : f 2014 = (t + 3) / (t - 3)) : 
  0 ≤ t ∧ t < 3 := by
  sorry

end range_of_t_l33_33059


namespace range_of_a_for_root_l33_33092

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - 2 * x + a

theorem range_of_a_for_root (a : ℝ) : (∃ x, f x a = 0) ↔ a ≤ 2 * Real.log 2 - 2 := sorry

end range_of_a_for_root_l33_33092


namespace average_speed_calculation_l33_33883

-- Define constants and conditions
def speed_swimming : ℝ := 1
def speed_running : ℝ := 6
def distance : ℝ := 1  -- We use a generic distance d = 1 (assuming normalized unit distance)

-- Proof statement
theorem average_speed_calculation :
  (2 * distance) / ((distance / speed_swimming) + (distance / speed_running)) = 12 / 7 :=
by
  sorry

end average_speed_calculation_l33_33883


namespace count_four_digit_numbers_divisible_by_5_ending_in_45_l33_33413

theorem count_four_digit_numbers_divisible_by_5_ending_in_45 : 
  ∃ n : ℕ, (∀ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 100 = 45 → (x % 5 = 0 ∧ x % 100 = 45)) 
  ∧ (n = 90) :=
by
  sorry

end count_four_digit_numbers_divisible_by_5_ending_in_45_l33_33413


namespace mean_of_two_fractions_l33_33876

theorem mean_of_two_fractions :
  ( (2 : ℚ) / 3 + (4 : ℚ) / 9 ) / 2 = 5 / 9 :=
by
  sorry

end mean_of_two_fractions_l33_33876


namespace attendance_changes_l33_33759

theorem attendance_changes :
  let m := 25  -- Monday attendance
  let t := 31  -- Tuesday attendance
  let w := 20  -- initial Wednesday attendance
  let th := 28  -- Thursday attendance
  let f := 22  -- Friday attendance
  let sa := 26  -- Saturday attendance
  let w_new := 30  -- corrected Wednesday attendance
  let initial_total := m + t + w + th + f + sa
  let new_total := m + t + w_new + th + f + sa
  let initial_mean := initial_total / 6
  let new_mean := new_total / 6
  let mean_increase := new_mean - initial_mean
  let initial_median := (25 + 26) / 2  -- median of [20, 22, 25, 26, 28, 31]
  let new_median := (26 + 28) / 2  -- median of [22, 25, 26, 28, 30, 31]
  let median_increase := new_median - initial_median
  mean_increase = 1.667 ∧ median_increase = 1.5 := by
sorry

end attendance_changes_l33_33759


namespace magic_shop_change_l33_33449

theorem magic_shop_change :
  (∀ (cloak : Type), ∃ price_gold price_silver1 change_gold1 price_silver2 change_gold2, 
  price_silver1 = 20 ∧ change_gold1 = 4 ∧ 
  price_silver2 = 15 ∧ change_gold2 = 1 ∧ 
  price_gold = 14 ∧ 
  ∀ change_silver, 
    (20 - 4) * change_silver = 15 - 1 → -- Relation derived from the conditions
    (14 - (15 - 1) * change_silver / (20 - 4)) * change_silver = 10) := 
sorry

end magic_shop_change_l33_33449


namespace shop_combinations_l33_33129

theorem shop_combinations :
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  have case1 : keyboard_and_mouse_sets * headphones = 4 * 9, by sorry
  have case2 : headphone_and_mouse_sets * keyboards = 5 * 5, by sorry
  have case3 : headphones * mice * keyboards = 9 * 13 * 5, by sorry
  show (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646,
  by rw [case1, case2, case3]; exact sorry

end shop_combinations_l33_33129


namespace max_participants_win_at_least_three_matches_l33_33612

theorem max_participants_win_at_least_three_matches (n : ℕ) (h : n = 200) : 
  ∃ k : ℕ, k = 66 ∧ ∀ m : ℕ, (k * 3 ≤ m) ∧ (m ≤ 199) → k ≤ m / 3 := 
by
  sorry

end max_participants_win_at_least_three_matches_l33_33612


namespace correct_statement_l33_33211

theorem correct_statement (x : ℝ) : 
  (∃ y : ℝ, y ≠ 0 ∧ y * x = 1 → x = 1 ∨ x = -1 ∨ x = 0) → false ∧
  (∃ y : ℝ, -y = y → y = 0 ∨ y = 1) → false ∧
  (abs x = x → x ≥ 0) → (x ^ 2 = 1 → x = 1 ∨ x = -1) :=
by
  sorry

end correct_statement_l33_33211


namespace cone_volume_from_half_sector_l33_33616

theorem cone_volume_from_half_sector (R : ℝ) (V : ℝ) : 
  R = 6 →
  V = (1/3) * Real.pi * (R / 2)^2 * (R * Real.sqrt 3) →
  V = 9 * Real.pi * Real.sqrt 3 := by sorry

end cone_volume_from_half_sector_l33_33616


namespace probability_of_at_most_3_heads_l33_33589

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l33_33589


namespace right_triangle_area_l33_33523

theorem right_triangle_area (leg1 hypotenuse : ℝ) (h1 : leg1 = 30) (h2 : hypotenuse = 34) (h3 : (leg1 ^ 2 + (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) ^ 2 = hypotenuse ^ 2)) :
  (1 / 2) * leg1 * (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) = 240 :=
by
  sorry

end right_triangle_area_l33_33523


namespace weight_of_each_dumbbell_l33_33977

-- Definitions based on conditions
def initial_dumbbells : Nat := 4
def added_dumbbells : Nat := 2
def total_dumbbells : Nat := initial_dumbbells + added_dumbbells -- 6
def total_weight : Nat := 120

-- Theorem statement
theorem weight_of_each_dumbbell (h : total_dumbbells = 6) (w : total_weight = 120) :
  total_weight / total_dumbbells = 20 :=
by
  -- Proof is to be written here
  sorry

end weight_of_each_dumbbell_l33_33977


namespace trajectory_of_point_l33_33896

theorem trajectory_of_point (x y k : ℝ) (hx : x ≠ 0) (hk : k ≠ 0) (h : |y| / |x| = k) : y = k * x ∨ y = -k * x :=
by
  sorry

end trajectory_of_point_l33_33896


namespace max_possible_value_of_e_n_l33_33966

noncomputable def b (n : ℕ) : ℚ := (8^n - 1) / 7

def e (n : ℕ) : ℕ := Int.gcd (b n).numerator (b (n + 1)).numerator

theorem max_possible_value_of_e_n : ∀ n : ℕ, e n = 1 :=
by sorry

end max_possible_value_of_e_n_l33_33966


namespace annual_average_growth_rate_l33_33041

theorem annual_average_growth_rate (x : ℝ) (h : x > 0): 
  100 * (1 + x)^2 = 169 :=
sorry

end annual_average_growth_rate_l33_33041


namespace probability_at_most_3_heads_10_flips_l33_33592

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l33_33592


namespace car_lease_annual_cost_l33_33353

/-- Tom decides to lease a car. He drives 50 miles on four specific days a week,
and 100 miles on the other three days. He pays $0.1 per mile and a weekly fee of $100.
Prove that the total annual cost he has to pay is $7800.
--/
theorem car_lease_annual_cost :
  let weekly_miles := 4 * 50 + 3 * 100
      weekly_mileage_cost := weekly_miles * 0.1
      weekly_total_cost := weekly_mileage_cost + 100
      annual_cost := weekly_total_cost * 52
  in annual_cost = 7800 :=
by
  let weekly_miles := 4 * 50 + 3 * 100
  let weekly_mileage_cost := weekly_miles * 0.1
  let weekly_total_cost := weekly_mileage_cost + 100
  let annual_cost := weekly_total_cost * 52
  sorry

end car_lease_annual_cost_l33_33353


namespace Billy_weight_is_159_l33_33628

def Carl_weight : ℕ := 145
def Brad_weight : ℕ := Carl_weight + 5
def Billy_weight : ℕ := Brad_weight + 9

theorem Billy_weight_is_159 : Billy_weight = 159 := by
  sorry

end Billy_weight_is_159_l33_33628


namespace Problem_l33_33393

theorem Problem (N : ℕ) (hn : N = 16) :
  (Nat.choose N 5) = 2002 := 
by 
  rw [hn] 
  sorry

end Problem_l33_33393


namespace proof1_proof2a_proof2b_l33_33680

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (S : ℝ)

-- Given conditions for Question 1
def condition1 := (a = 3 * Real.cos C ∧ b = 1)

-- Proof statement for Question 1
theorem proof1 : condition1 a b C → Real.tan C = 2 * Real.tan B :=
by sorry

-- Given conditions for Question 2a
def condition2a := (S = 1 / 2 * a * b * Real.sin C ∧ S = 1 / 2 * 3 * Real.cos C * 1 * Real.sin C)

-- Proof statement for Question 2a
theorem proof2a : condition2a a b C S → Real.cos (2 * B) = 3 / 5 :=
by sorry

-- Given conditions for Question 2b
def condition2b := (c = Real.sqrt 10 / 2)

-- Proof statement for Question 2b
theorem proof2b : condition1 a b C → condition2b c → Real.cos (2 * B) = 3 / 5 :=
by sorry

end proof1_proof2a_proof2b_l33_33680


namespace find_k_for_sum_of_cubes_l33_33385

theorem find_k_for_sum_of_cubes (k : ℝ) (r s : ℝ)
  (h1 : r + s = -2)
  (h2 : r * s = k / 3)
  (h3 : r^3 + s^3 = r + s) : k = 3 :=
by
  -- Sorry will be replaced by the actual proof
  sorry

end find_k_for_sum_of_cubes_l33_33385


namespace degrees_to_radians_216_l33_33768

theorem degrees_to_radians_216 : (216 / 180 : ℝ) * Real.pi = (6 / 5 : ℝ) * Real.pi := by
  sorry

end degrees_to_radians_216_l33_33768


namespace probability_at_most_3_heads_10_coins_l33_33604

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l33_33604


namespace measure_of_C_and_max_perimeter_l33_33432

noncomputable def triangle_C_and_perimeter (a b c A B C : ℝ) (hABC : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) (hc : c = Real.sqrt 3) : Prop :=
  (C = 2 * Real.pi / 3) ∧ (2 * Real.sin A + 2 * Real.sin B + c ≤ 2 + Real.sqrt 3)

-- Now the Lean theorem statement
theorem measure_of_C_and_max_perimeter (a b c A B C : ℝ) (hABC : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) (hc : c = Real.sqrt 3) :
  triangle_C_and_perimeter a b c A B C hABC hc :=
by 
  sorry

end measure_of_C_and_max_perimeter_l33_33432


namespace num_ways_to_buy_three_items_l33_33118

-- Defining the conditions based on the problem statement
def num_headphones : ℕ := 9
def num_mice : ℕ := 13
def num_keyboards : ℕ := 5
def num_kb_mouse_sets : ℕ := 4
def num_hp_mouse_sets : ℕ := 5

-- Defining the theorem statement
theorem num_ways_to_buy_three_items :
  (num_kb_mouse_sets * num_headphones) + 
  (num_hp_mouse_sets * num_keyboards) + 
  (num_headphones * num_mice * num_keyboards) = 646 := 
  by
  sorry

end num_ways_to_buy_three_items_l33_33118


namespace second_train_speed_l33_33875

noncomputable def speed_of_second_train (length1 length2 speed1 clearance_time : ℝ) : ℝ :=
  let total_distance := (length1 + length2) / 1000 -- convert meters to kilometers
  let time_in_hours := clearance_time / 3600 -- convert seconds to hours
  let relative_speed := total_distance / time_in_hours
  relative_speed - speed1

theorem second_train_speed : 
  speed_of_second_train 60 280 42 16.998640108791296 = 30.05 := 
by
  sorry

end second_train_speed_l33_33875


namespace find_strawberry_jelly_amount_l33_33710

noncomputable def strawberry_jelly (t b : ℕ) : ℕ := t - b

theorem find_strawberry_jelly_amount (h₁ : 6310 = 4518 + s) : s = 1792 := by
  sorry

end find_strawberry_jelly_amount_l33_33710


namespace other_root_l33_33267

theorem other_root (k : ℝ) : 
  5 * (2:ℝ)^2 + k * (2:ℝ) - 8 = 0 → 
  ∃ q : ℝ, 5 * q^2 + k * q - 8 = 0 ∧ q ≠ 2 ∧ q = -4/5 :=
by {
  sorry
}

end other_root_l33_33267


namespace cloak_change_l33_33453

theorem cloak_change (silver_for_cloak1 silver_for_cloak2 : ℕ) (gold_change1 gold_change2 : ℕ) (silver_change : ℕ) :
  silver_for_cloak1 = 20 →
  gold_change1 = 4 →
  silver_for_cloak2 = 15 →
  gold_change2 = 1 →
  ∃ silver_cost_of_cloak : ℕ, 
    silver_cost_of_cloak = (20 - 4) * (5 / 3) →
    silver_change = 10 →
    14 * (5 / 3) - 8 = silver_change :=
by 
  assume h1 h2 h3 h4,
  use 16, 
  sorry

end cloak_change_l33_33453


namespace solve_equation_l33_33847

def equation_holds (x : ℝ) : Prop := 
  (1 / (x + 10)) + (1 / (x + 8)) = (1 / (x + 11)) + (1 / (x + 7))

theorem solve_equation : equation_holds (-9) :=
by
  sorry

end solve_equation_l33_33847


namespace prove_prob_scoring_200_points_prove_prob_scoring_at_least_300_points_l33_33758

noncomputable def prob_scoring_200_points (P_A1 P_A2 P_A3 : ℝ) : ℝ :=
  P_A1 * P_A2 * (1 - P_A3) + (1 - P_A1) * P_A2 * P_A3

noncomputable def prob_scoring_at_least_300_points (P_A1 P_A2 P_A3 : ℝ) : ℝ :=
  P_A1 * (1 - P_A2) * P_A3 + (1 - P_A1) * P_A2 * P_A3 + P_A1 * P_A2 * P_A3

theorem prove_prob_scoring_200_points :
  prob_scoring_200_points 0.8 0.7 0.6 = 0.26 := sorry
  
theorem prove_prob_scoring_at_least_300_points :
  prob_scoring_at_least_300_points 0.8 0.7 0.6 = 0.564 := sorry

end prove_prob_scoring_200_points_prove_prob_scoring_at_least_300_points_l33_33758


namespace coin_flip_difference_l33_33196

/-- The positive difference between the probability of a fair coin landing heads up
exactly 4 times out of 5 flips and the probability of a fair coin landing heads up
5 times out of 5 flips is 1/8. -/
theorem coin_flip_difference :
  (5 * (1 / 2) ^ 5) - ((1 / 2) ^ 5) = (1 / 8) :=
by
  sorry

end coin_flip_difference_l33_33196


namespace inlet_rate_480_l33_33231

theorem inlet_rate_480 (capacity : ℕ) (T_outlet : ℕ) (T_outlet_inlet : ℕ) (R_i : ℕ) :
  capacity = 11520 →
  T_outlet = 8 →
  T_outlet_inlet = 12 →
  R_i = 480 :=
by
  intros
  sorry

end inlet_rate_480_l33_33231


namespace average_salary_all_workers_l33_33329

theorem average_salary_all_workers 
  (n : ℕ) (avg_salary_technicians avg_salary_rest total_avg_salary : ℝ)
  (h1 : n = 7) 
  (h2 : avg_salary_technicians = 8000) 
  (h3 : avg_salary_rest = 6000)
  (h4 : total_avg_salary = avg_salary_technicians) : 
  total_avg_salary = 8000 :=
by sorry

end average_salary_all_workers_l33_33329


namespace initial_books_in_bin_l33_33748

variable (X : ℕ)

theorem initial_books_in_bin (h1 : X - 3 + 10 = 11) : X = 4 :=
by
  sorry

end initial_books_in_bin_l33_33748


namespace nim_maximum_product_l33_33138

def nim_max_product (x y : ℕ) : ℕ :=
43 * 99 * x * y

theorem nim_maximum_product :
  ∃ x y : ℕ, (43 ≠ 0) ∧ (99 ≠ 0) ∧ (x ≠ 0) ∧ (y ≠ 0) ∧
  (43 + 99 + x + y = 0) ∧ (nim_max_product x y = 7704) :=
sorry

end nim_maximum_product_l33_33138


namespace probability_prime_or_odd_l33_33843

-- Define the set of balls and their corresponding numbers
def balls : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the set of prime numbers among the balls
def primes : Finset ℕ := {2, 3, 5, 7}

-- Define the set of odd numbers among the balls
def odds : Finset ℕ := {1, 3, 5, 7}

-- Define the set of numbers that are either prime or odd
def primes_or_odds := primes ∪ odds

-- Calculate the probability as the ratio of the size of primes_or_odds set to the size of balls set
def probability := (primes_or_odds.card : ℚ) / balls.card

-- Statement that the probability is 5/7
theorem probability_prime_or_odd : probability = 5 / 7 := by
  sorry

end probability_prime_or_odd_l33_33843


namespace number_of_ways_to_buy_three_items_l33_33133

def headphones : ℕ := 9
def mice : ℕ := 13
def keyboards : ℕ := 5
def keyboard_mouse_sets : ℕ := 4
def headphone_mouse_sets : ℕ := 5

theorem number_of_ways_to_buy_three_items : 
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 := 
by 
  sorry

end number_of_ways_to_buy_three_items_l33_33133


namespace positive_difference_between_probabilities_is_one_eighth_l33_33197

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem positive_difference_between_probabilities_is_one_eighth :
  let p : ℚ := 1 / 2
    n : ℕ := 5
    prob_4_heads := binomial_probability n 4 p
    prob_5_heads := binomial_probability n 5 p
  in |prob_4_heads - prob_5_heads| = 1 / 8 :=
by
  let p : ℚ := 1 / 2
  let n : ℕ := 5
  let prob_4_heads := binomial_probability n 4 p
  let prob_5_heads := binomial_probability n 5 p
  sorry

end positive_difference_between_probabilities_is_one_eighth_l33_33197


namespace total_pairs_sold_l33_33906

theorem total_pairs_sold (H S : ℕ) 
    (soft_lens_cost hard_lens_cost : ℕ)
    (total_sales : ℕ)
    (h1 : soft_lens_cost = 150)
    (h2 : hard_lens_cost = 85)
    (h3 : S = H + 5)
    (h4 : soft_lens_cost * S + hard_lens_cost * H = total_sales)
    (h5 : total_sales = 1455) :
    H + S = 11 := 
  sorry

end total_pairs_sold_l33_33906


namespace perimeter_of_shaded_shape_l33_33014

noncomputable def shaded_perimeter (x : ℝ) : ℝ := 
  let l := 18 - 2 * x
  3 * l

theorem perimeter_of_shaded_shape (x : ℝ) (hx : x > 0) (h_sectors : 2 * x + (18 - 2 * x) = 18) : 
  shaded_perimeter x = 54 := 
by
  rw [shaded_perimeter]
  rw [← h_sectors]
  simp
  sorry

end perimeter_of_shaded_shape_l33_33014


namespace range_of_m_l33_33099

variable {f : ℝ → ℝ}

def is_decreasing (f : ℝ → ℝ) := ∀ x y, x < y → f x > f y

theorem range_of_m (hf_dec : is_decreasing f) (hf_odd : ∀ x, f (-x) = -f x) 
  (h : ∀ m, f (m - 1) + f (2 * m - 1) > 0) : ∀ m, m < 2 / 3 :=
by
  sorry

end range_of_m_l33_33099


namespace probability_of_at_most_3_heads_l33_33569

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l33_33569


namespace rods_in_one_mile_l33_33401

/-- Definitions based on given conditions -/
def miles_to_furlongs := 8
def furlongs_to_rods := 40

/-- The theorem stating the number of rods in one mile -/
theorem rods_in_one_mile : (miles_to_furlongs * furlongs_to_rods) = 320 := 
  sorry

end rods_in_one_mile_l33_33401


namespace probability_at_most_three_heads_10_coins_l33_33600

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l33_33600


namespace width_of_first_tv_is_24_l33_33694

-- Define the conditions
def height_first_tv := 16
def cost_first_tv := 672
def width_new_tv := 48
def height_new_tv := 32
def cost_new_tv := 1152
def cost_per_sq_inch_diff := 1

-- Define the width of the first TV
def width_first_tv := 24

-- Define the areas
def area_first_tv (W : ℕ) := W * height_first_tv
def area_new_tv := width_new_tv * height_new_tv

-- Define the cost per square inch
def cost_per_sq_inch_first_tv (W : ℕ) := cost_first_tv / area_first_tv W
def cost_per_sq_inch_new_tv := cost_new_tv / area_new_tv

-- The proof statement
theorem width_of_first_tv_is_24 :
  cost_per_sq_inch_first_tv width_first_tv = cost_per_sq_inch_new_tv + cost_per_sq_inch_diff
  := by
    unfold cost_per_sq_inch_first_tv
    unfold area_first_tv
    unfold cost_per_sq_inch_new_tv
    unfold area_new_tv
    sorry -- proof to be filled in

end width_of_first_tv_is_24_l33_33694


namespace find_c_l33_33990

theorem find_c (c : ℝ) (h1 : 0 < c) (h2 : c < 6) (h3 : ((6 - c) / c) = 4 / 9) : c = 54 / 13 :=
sorry

end find_c_l33_33990


namespace probability_is_five_sevenths_l33_33844

noncomputable def probability_prime_or_odd : ℚ :=
  let balls := {1, 2, 3, 4, 5, 6, 7}
  let primes := {2, 3, 5, 7}
  let odds := {1, 3, 5, 7}
  let combined := primes ∪ odds
  let favorable_outcomes := combined.card
  let total_outcomes := balls.card
  favorable_outcomes / total_outcomes

theorem probability_is_five_sevenths :
  probability_prime_or_odd = 5 / 7 := 
sorry

end probability_is_five_sevenths_l33_33844


namespace longer_diagonal_eq_l33_33461

variable (a b : ℝ)
variable (h_cd : CD = a) (h_bc : BC = b) (h_diag : AC = a) (h_ad : AD = 2 * b)

theorem longer_diagonal_eq (CD BC AC AD BD : ℝ) (h_cd : CD = a)
  (h_bc : BC = b) (h_diag : AC = CD) (h_ad : AD = 2 * b) :
  BD = Real.sqrt (a^2 + 3 * b^2) :=
sorry

end longer_diagonal_eq_l33_33461


namespace belle_rawhide_bones_per_evening_l33_33912

theorem belle_rawhide_bones_per_evening 
  (cost_rawhide_bone : ℝ)
  (cost_dog_biscuit : ℝ)
  (num_dog_biscuits_per_evening : ℕ)
  (total_weekly_cost : ℝ)
  (days_per_week : ℕ)
  (rawhide_bones_per_evening : ℕ)
  (h1 : cost_rawhide_bone = 1)
  (h2 : cost_dog_biscuit = 0.25)
  (h3 : num_dog_biscuits_per_evening = 4)
  (h4 : total_weekly_cost = 21)
  (h5 : days_per_week = 7)
  (h6 : rawhide_bones_per_evening * cost_rawhide_bone * (days_per_week : ℝ) = total_weekly_cost - num_dog_biscuits_per_evening * cost_dog_biscuit * (days_per_week : ℝ)) :
  rawhide_bones_per_evening = 2 := 
sorry

end belle_rawhide_bones_per_evening_l33_33912


namespace total_number_of_games_l33_33012

theorem total_number_of_games (n : ℕ) (k : ℕ) (teams : Finset ℕ)
  (h_n : n = 8) (h_k : k = 2) (h_teams : teams.card = n) :
  (teams.card.choose k) = 28 :=
by
  sorry

end total_number_of_games_l33_33012


namespace distance_traveled_in_20_seconds_l33_33227

-- Define the initial distance, common difference, and total time
def initial_distance : ℕ := 8
def common_difference : ℕ := 9
def total_time : ℕ := 20

-- Define the nth term of the arithmetic sequence
def nth_term (n : ℕ) : ℕ := initial_distance + (n - 1) * common_difference

-- Define the sum of the first n terms of the arithmetic sequence
def sum_of_terms (n : ℕ) : ℕ := n * (initial_distance + nth_term n) / 2

-- The main theorem to be proven
theorem distance_traveled_in_20_seconds : sum_of_terms 20 = 1870 := 
by sorry

end distance_traveled_in_20_seconds_l33_33227


namespace rainfall_november_is_180_l33_33277

-- Defining the conditions
def daily_rainfall_first_15_days := 4 -- inches per day
def days_in_first_period := 15
def total_days_in_november := 30
def multiplier_for_second_period := 2

-- Calculation based on the problem's conditions
def total_rainfall_november := 
  (daily_rainfall_first_15_days * days_in_first_period) + 
  (multiplier_for_second_period * daily_rainfall_first_15_days * (total_days_in_november - days_in_first_period))

-- Prove that the total rainfall in November is 180 inches
theorem rainfall_november_is_180 : total_rainfall_november = 180 :=
by
  -- Proof steps (to be filled in)
  sorry

end rainfall_november_is_180_l33_33277


namespace arithmetic_sequence_sum_l33_33467

variable {a : ℕ → ℝ}

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

-- Definition of the fourth term condition
def a4_condition (a : ℕ → ℝ) : Prop :=
  a 4 = 2 - a 3

-- Definition of the sum of the first 6 terms
def sum_first_six_terms (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

-- Proof statement
theorem arithmetic_sequence_sum :
  is_arithmetic_sequence a →
  a4_condition a →
  sum_first_six_terms a = 6 :=
by
  sorry

end arithmetic_sequence_sum_l33_33467


namespace avg_of_multiples_of_4_is_even_l33_33210

theorem avg_of_multiples_of_4_is_even (m n : ℤ) (hm : m % 4 = 0) (hn : n % 4 = 0) :
  (m + n) / 2 % 2 = 0 := sorry

end avg_of_multiples_of_4_is_even_l33_33210


namespace graph_symmetric_about_point_l33_33862

noncomputable def tan_shifted_symmetric : Prop :=
  ∀ x : ℝ, tan (2 * (x + π / 6) + π / 6) = g(x)

theorem graph_symmetric_about_point :
  ∀ x : ℝ, tan (2 * x + π / 2) = g(x) → g(π / 4) = 0 :=
sorry

end graph_symmetric_about_point_l33_33862


namespace maximize_profit_of_store_l33_33222

-- Define the basic conditions
variable (basketball_cost volleyball_cost : ℕ)
variable (total_items budget : ℕ)
variable (price_increase_factor : ℚ)
variable (school_basketball_cost school_volleyball_difference school_volleyball_cost : ℕ)
variable (reduced_basketball_price_diff reduced_volleyball_price_diff : ℚ)

-- Definitions from conditions
def store_plans := total_items = 200
def budget_constraint := budget ≤ 5000
def cost_price_basketball := basketball_cost = 30
def cost_price_volleyball := volleyball_cost = 24
def selling_price_relation := price_increase_factor = 1.5
def school_basketball_cost_condition := school_basketball_cost = 1800
def school_volleyball_difference_condition := school_volleyball_difference = 10
def school_volleyball_cost_condition := school_volleyball_cost = 1500

-- First Problem: Finding selling price
def volleyball_selling_price (x : ℚ) := 
  school_volleyball_cost / x - school_basketball_cost / (price_increase_factor * x) = school_volleyball_difference

-- Second Problem: Maximizing profit
def profit (a : ℚ) (basketball_selling_price volleyball_selling_price : ℚ) :=
  let reduced_basketball_price := basketball_selling_price - reduced_basketball_price_diff
  let reduced_volleyball_price := volleyball_selling_price - reduced_volleyball_price_diff
  let profit_from_basketball := (reduced_basketball_price - basketball_cost) * a
  let profit_from_volleyball := (reduced_volleyball_price - volleyball_cost) * (200 - a)
  profit_from_basketball + profit_from_volleyball

def budget_constraint_for_max_profit (a : ℚ) := 
  let total_basketball_cost := basketball_cost * a
  let total_volleyball_cost := volleyball_cost * (200 - a)
  total_basketball_cost + total_volleyball_cost ≤ budget

theorem maximize_profit_of_store :
  ∃ volleyball_selling_price basketball_selling_price (a : ℚ),
    volleyball_selling_price volleyball_selling_price ∧ 
    budget_constraint_for_max_profit a ∧
    a = 33 ∧ (200 - a) = 167 := 
sorry

end maximize_profit_of_store_l33_33222


namespace quadratic_function_vertex_upwards_exists_l33_33030

theorem quadratic_function_vertex_upwards_exists :
  ∃ (a : ℝ), a > 0 ∧ ∃ (f : ℝ → ℝ), (∀ x, f x = a * (x - 1) * (x - 1) - 2) :=
by
  sorry

end quadratic_function_vertex_upwards_exists_l33_33030


namespace max_value_of_e_n_l33_33965

def b (n : ℕ) : ℕ := (8^n - 1) / 7
def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

theorem max_value_of_e_n : ∀ n : ℕ, e n = 1 := 
by
  sorry

end max_value_of_e_n_l33_33965


namespace chessboard_piece_arrangements_l33_33015

-- Define the problem in Lean
theorem chessboard_piece_arrangements (black_pos white_pos : ℕ)
  (black_pos_neq_white_pos : black_pos ≠ white_pos)
  (valid_position : black_pos < 64 ∧ white_pos < 64) :
  ¬(∀ (move : ℕ → ℕ → Prop), (move black_pos white_pos) → ∃! (p : ℕ × ℕ), move (p.fst) (p.snd)) :=
by sorry

end chessboard_piece_arrangements_l33_33015


namespace a2_value_l33_33077

open BigOperators
open Nat

def f (x : ℚ) : ℚ := ∑ i in (range 10).map (λ n, n + 1), (1 + x) ^ i

theorem a2_value : let a := (range 11).map (λ k, k * (k - 1) / 2)
                  in a.sum = 165 :=
by sorry

end a2_value_l33_33077


namespace prob_heads_at_most_3_out_of_10_flips_l33_33556

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l33_33556


namespace probability_of_at_most_3_heads_l33_33568

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l33_33568


namespace value_of_x_if_additive_inverses_l33_33264

theorem value_of_x_if_additive_inverses (x : ℝ) 
  (h : 4 * x - 1 + (3 * x - 6) = 0) : x = 1 := by
sorry

end value_of_x_if_additive_inverses_l33_33264


namespace min_cards_needed_l33_33488

/-- 
On a table, there are five types of number cards: 1, 3, 5, 7, and 9, with 30 cards of each type. 
Prove that the minimum number of cards required to ensure that the sum of the drawn card numbers 
can represent all integers from 1 to 200 is 26.
-/
theorem min_cards_needed : ∀ (cards_1 cards_3 cards_5 cards_7 cards_9 : ℕ), 
  cards_1 = 30 → cards_3 = 30 → cards_5 = 30 → cards_7 = 30 → cards_9 = 30 → 
  ∃ n, (n = 26) ∧ 
  (∀ k, 1 ≤ k ∧ k ≤ 200 → 
    ∃ a b c d e, 
      a ≤ cards_1 ∧ b ≤ cards_3 ∧ c ≤ cards_5 ∧ d ≤ cards_7 ∧ e ≤ cards_9 ∧ 
      k = a * 1 + b * 3 + c * 5 + d * 7 + e * 9) :=
by {
  sorry
}

end min_cards_needed_l33_33488


namespace find_a_8_l33_33466

-- Define the arithmetic sequence and its sum formula.
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n, a (n + 1) = a n + d

-- Define the sum of the first 'n' terms in the arithmetic sequence.
def sum_of_first_n_terms (S : ℕ → ℕ) (a : ℕ → ℕ) (n : ℕ) :=
  S n = n * (a 1 + a n) / 2

-- Given conditions
def S_15_eq_90 (S : ℕ → ℕ) : Prop := S 15 = 90

-- Prove that a_8 is 6
theorem find_a_8 (S : ℕ → ℕ) (a : ℕ → ℕ) (d : ℕ)
  (h1 : arithmetic_sequence a d) (h2 : sum_of_first_n_terms S a 15)
  (h3 : S_15_eq_90 S) : a 8 = 6 :=
sorry

end find_a_8_l33_33466


namespace greendale_points_l33_33313

theorem greendale_points : 
  let roosevelt_game1 := 30 
  let roosevelt_game2 := roosevelt_game1 / 2
  let roosevelt_game3 := roosevelt_game2 * 3
  let roosevelt_bonus := 50
  let greendale_diff := 10
  let roosevelt_total := roosevelt_game1 + roosevelt_game2 + roosevelt_game3 + roosevelt_bonus
  let greendale_total := roosevelt_total - greendale_diff
  greendale_total = 130 :=
by
  sorry

end greendale_points_l33_33313


namespace periodic_odd_function_l33_33942

theorem periodic_odd_function (f : ℝ → ℝ) (period : ℝ) (h_periodic : ∀ x, f (x + period) = f x) (h_odd : ∀ x, f (-x) = -f x) (h_value : f (-3) = 1) (α : ℝ) (h_tan : Real.tan α = 2) :
  f (20 * Real.sin α * Real.cos α) = -1 := 
sorry

end periodic_odd_function_l33_33942


namespace cloak_change_in_silver_l33_33457

theorem cloak_change_in_silver :
  (∀ c : ℤ, (20 = c + 4) → (15 = c + 1)) →
  (5 * g = 3) →
  14 * gold / exchange_rate = 10 := 
sorry

end cloak_change_in_silver_l33_33457


namespace probability_at_most_3_heads_l33_33558

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l33_33558


namespace find_ax_plus_a_negx_l33_33658

theorem find_ax_plus_a_negx
  (a : ℝ) (x : ℝ)
  (h₁ : a > 0)
  (h₂ : a^(x/2) + a^(-x/2) = 5) :
  a^x + a^(-x) = 23 :=
by
  sorry

end find_ax_plus_a_negx_l33_33658


namespace seq_a3_eq_1_l33_33728

theorem seq_a3_eq_1 (a : ℕ → ℤ) (h₁ : ∀ n ≥ 1, a (n + 1) = a n - 3) (h₂ : a 1 = 7) : a 3 = 1 :=
by
  sorry

end seq_a3_eq_1_l33_33728


namespace inscribed_circle_radius_right_triangle_l33_33308

theorem inscribed_circle_radius_right_triangle : 
  ∀ (DE EF DF : ℝ), 
    DE = 6 →
    EF = 8 →
    DF = 10 →
    ∃ (r : ℝ), r = 2 :=
by
  intros DE EF DF hDE hEF hDF
  sorry

end inscribed_circle_radius_right_triangle_l33_33308


namespace difference_white_black_l33_33350

def total_stones : ℕ := 928
def white_stones : ℕ := 713
def black_stones : ℕ := total_stones - white_stones

theorem difference_white_black :
  (white_stones - black_stones = 498) :=
by
  -- Leaving the proof for later
  sorry

end difference_white_black_l33_33350


namespace knights_round_table_l33_33881

theorem knights_round_table (n : ℕ) (h : ∃ (f e : ℕ), f = e ∧ f + e = n) : n % 4 = 0 :=
sorry

end knights_round_table_l33_33881


namespace speed_conversion_l33_33058

theorem speed_conversion (speed_m_s : ℚ) (conversion_factor : ℚ) :
  speed_m_s = 8 / 26 → conversion_factor = 3.6 →
  speed_m_s * conversion_factor = 1.1077 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end speed_conversion_l33_33058


namespace shop_combinations_l33_33128

theorem shop_combinations :
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  have case1 : keyboard_and_mouse_sets * headphones = 4 * 9, by sorry
  have case2 : headphone_and_mouse_sets * keyboards = 5 * 5, by sorry
  have case3 : headphones * mice * keyboards = 9 * 13 * 5, by sorry
  show (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646,
  by rw [case1, case2, case3]; exact sorry

end shop_combinations_l33_33128


namespace trig_identity_75_30_15_150_l33_33636

theorem trig_identity_75_30_15_150 :
  (Real.sin (75 * Real.pi / 180) * Real.cos (30 * Real.pi / 180) - 
   Real.sin (15 * Real.pi / 180) * Real.sin (150 * Real.pi / 180)) = 
  (Real.sqrt 2 / 2) :=
by
  -- Proof goes here
  sorry

end trig_identity_75_30_15_150_l33_33636


namespace purely_imaginary_complex_iff_l33_33664

theorem purely_imaginary_complex_iff (m : ℝ) :
  (m + 2 = 0) → (m = -2) :=
by
  sorry

end purely_imaginary_complex_iff_l33_33664


namespace compute_xy_l33_33874

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^2 - y^2 = 20) : x * y = -56 / 9 :=
by
  sorry

end compute_xy_l33_33874


namespace terminating_decimal_fraction_count_l33_33779

theorem terminating_decimal_fraction_count :
  (finset.filter (λ n : ℕ, 1 ≤ n ∧ n ≤ 180 ∧ 90 ∣ n) (finset.range 181)).card = 20 :=
by sorry

end terminating_decimal_fraction_count_l33_33779


namespace smallest_yellow_marbles_l33_33973

-- Definitions for given conditions
def total_marbles (n : ℕ): Prop := n > 0
def blue_marbles (n : ℕ) : ℕ := n / 4
def red_marbles (n : ℕ) : ℕ := n / 6
def green_marbles : ℕ := 7
def yellow_marbles (n : ℕ) : ℕ := n - (blue_marbles n + red_marbles n + green_marbles)

-- Lean statement that verifies the smallest number of yellow marbles is 0
theorem smallest_yellow_marbles (n : ℕ) (h : total_marbles n) : yellow_marbles n = 0 :=
  sorry

end smallest_yellow_marbles_l33_33973


namespace sum_of_15_terms_l33_33819

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sum_of_15_terms 
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_sum1 : a 1 + a 2 + a 3 = 1)
  (h_sum2 : a 4 + a 5 + a 6 = -2) :
  (a 1 + a 2 + a 3) + (a 4 + a 5 + a 6) + (a 7 + a 8 + a 9) +
  (a 10 + a 11 + a 12) + (a 13 + a 14 + a 15) = 11 :=
sorry

end sum_of_15_terms_l33_33819


namespace solve_system_of_equations_l33_33471

def proof_problem (a b c : ℚ) : Prop :=
  ((a - b = 2) ∧ (c = -5) ∧ (2 * a - 6 * b = 2)) → 
  (a = 5 / 2 ∧ b = 1 / 2 ∧ c = -5)

theorem solve_system_of_equations (a b c : ℚ) :
  proof_problem a b c :=
  by
    sorry

end solve_system_of_equations_l33_33471


namespace central_angle_of_regular_hexagon_l33_33331

theorem central_angle_of_regular_hexagon :
  ∀ (total_angle : ℝ) (sides : ℝ), total_angle = 360 → sides = 6 → total_angle / sides = 60 :=
by
  intros total_angle sides h_total_angle h_sides
  rw [h_total_angle, h_sides]
  norm_num

end central_angle_of_regular_hexagon_l33_33331


namespace pow_log_sqrt_l33_33740

theorem pow_log_sqrt (a b c : ℝ) (h1 : a = 81) (h2 : b = 500) (h3 : c = 3) :
  ((a ^ (Real.log b / Real.log c)) ^ (1 / 2)) = 250000 :=
by
  sorry

end pow_log_sqrt_l33_33740


namespace part_a_part_b_l33_33032

-- Part (a)
theorem part_a (ABC : Type) (M: ABC) (R_a R_b R_c r : ℝ):
  ∀ (ABC : Type) (A B C : ABC) (M : ABC), 
  R_a + R_b + R_c ≥ 6 * r := sorry

-- Part (b)
theorem part_b (ABC : Type) (M: ABC) (R_a R_b R_c r : ℝ):
  ∀ (ABC : Type) (A B C : ABC) (M : ABC), 
  R_a^2 + R_b^2 + R_c^2 ≥ 12 * r^2 := sorry

end part_a_part_b_l33_33032


namespace a4_value_l33_33430

variable (a : ℕ → ℤ)
variable (S : ℕ → ℤ)

-- Condition: The sum of the first n terms of the sequence {a_n} is S_n = n^2 - 1
axiom sum_of_sequence (n : ℕ) : S n = n^2 - 1

-- We need to prove that a_4 = 7
theorem a4_value : a 4 = S 4 - S 3 :=
by 
  -- Proof goes here
  sorry

end a4_value_l33_33430


namespace store_purchase_ways_l33_33109

theorem store_purchase_ways :
  ∃ (headphones sets_hm sets_km keyboards mice : ℕ), 
  headphones = 9 ∧
  mice = 13 ∧
  keyboards = 5 ∧
  sets_km = 4 ∧
  sets_hm = 5 ∧
  (sets_km * headphones + sets_hm * keyboards + headphones * mice * keyboards = 646) :=
by
  use 9, 5, 4, 5, 13
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  sorry

end store_purchase_ways_l33_33109


namespace probability_fully_lit_l33_33714

-- define the conditions of the problem
def characters : List String := ["K", "y", "o", "t", "o", " ", "G", "r", "a", "n", "d", " ", "H", "o", "t", "e", "l"]

-- define the length of the sequence
def length_sequence : ℕ := characters.length

-- theorem stating the probability of seeing the fully lit sign
theorem probability_fully_lit : (1 / length_sequence) = 1 / 5 :=
by
  -- The proof is omitted
  sorry

end probability_fully_lit_l33_33714


namespace man_is_older_by_24_l33_33895

-- Define the conditions as per the given problem
def present_age_son : ℕ := 22
def present_age_man (M : ℕ) : Prop := M + 2 = 2 * (present_age_son + 2)

-- State the problem: Prove that the man is 24 years older than his son
theorem man_is_older_by_24 (M : ℕ) (h : present_age_man M) : M - present_age_son = 24 := 
sorry

end man_is_older_by_24_l33_33895


namespace sally_bread_consumption_l33_33704

/-
Given:
    1) Sally eats 2 sandwiches on Saturday
    2) Sally eats 1 sandwich on Sunday
    3) Each sandwich uses 2 pieces of bread

Prove:
    Sally eats 6 pieces of bread across Saturday and Sunday
-/
theorem sally_bread_consumption (sandwiches_sat : Nat) (sandwiches_sun : Nat) (bread_per_sandwich : Nat)
    (H1 : sandwiches_sat = 2) (H2 : sandwiches_sun = 1) (H3 : bread_per_sandwich = 2) :
    2 * bread_per_sandwich + 1 * bread_per_sandwich = 6 := by
  sorry

end sally_bread_consumption_l33_33704


namespace sum_of_real_solutions_l33_33645

theorem sum_of_real_solutions:
  (∃ (s : ℝ), ∀ x : ℝ, 
    (x - 3) / (x^2 + 6 * x + 2) = (x - 6) / (x^2 - 12 * x) → 
    s = 106 / 9) :=
  sorry

end sum_of_real_solutions_l33_33645


namespace fuel_calculation_l33_33377

def total_fuel_needed (empty_fuel_per_mile people_fuel_per_mile bag_fuel_per_mile num_passengers num_crew bags_per_person miles : ℕ) : ℕ :=
  let total_people := num_passengers + num_crew
  let total_bags := total_people * bags_per_person
  let total_fuel_per_mile := empty_fuel_per_mile + people_fuel_per_mile * total_people + bag_fuel_per_mile * total_bags
  total_fuel_per_mile * miles

theorem fuel_calculation :
  total_fuel_needed 20 3 2 30 5 2 400 = 106000 :=
by
  sorry

end fuel_calculation_l33_33377


namespace never_sunday_l33_33437

theorem never_sunday (n : ℕ) (days_in_month : ℕ → ℕ) (is_leap_year : Bool) : 
  (∀ (month : ℕ), 1 ≤ month ∧ month ≤ 12 → (days_in_month month = 28 ∨ days_in_month month = 29 ∨ days_in_month month = 30 ∨ days_in_month month = 31) ∧
  (∃ (k : ℕ), k < 7 ∧ ∀ (d : ℕ), d < days_in_month month → (d % 7 = k ↔ n ≠ d))) → n = 31 := 
by
  sorry

end never_sunday_l33_33437


namespace probability_heads_at_most_3_of_10_coins_flipped_l33_33548

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l33_33548


namespace general_formula_sequence_sum_first_n_terms_l33_33886

-- Define the axioms or conditions of the arithmetic sequence
axiom a3_eq_7 : ∃ a1 d : ℝ, a1 + 2 * d = 7
axiom a5_plus_a7_eq_26 : ∃ a1 d : ℝ, (a1 + 4 * d) + (a1 + 6 * d) = 26

-- State the theorem for the general formula of the arithmetic sequence
theorem general_formula_sequence (a1 d : ℝ) (h3 : a1 + 2 * d = 7) (h5_7 : (a1 + 4 * d) + (a1 + 6 * d) = 26) :
  ∀ n : ℕ, a1 + (n - 1) * d = 2 * n + 1 :=
sorry

-- State the theorem for the sum of the first n terms of the arithmetic sequence
theorem sum_first_n_terms (a1 d : ℝ) (h3 : a1 + 2 * d = 7) (h5_7 : (a1 + 4 * d) + (a1 + 6 * d) = 26) :
  ∀ n : ℕ, n * (a1 + (n - 1) * d + a1) / 2 = (n^2 + 2 * n) :=
sorry

end general_formula_sequence_sum_first_n_terms_l33_33886


namespace min_marked_price_l33_33040

theorem min_marked_price 
  (x : ℝ) 
  (sets : ℝ) 
  (cost_per_set : ℝ) 
  (discount : ℝ) 
  (desired_profit : ℝ) 
  (purchase_cost : ℝ) 
  (total_revenue : ℝ) 
  (cost : ℝ)
  (h1 : sets = 40)
  (h2 : cost_per_set = 80)
  (h3 : discount = 0.9)
  (h4 : desired_profit = 4000)
  (h5 : cost = sets * cost_per_set)
  (h6 : total_revenue = sets * (discount * x))
  (h7 : total_revenue - cost ≥ desired_profit) : x ≥ 200 := by
  sorry

end min_marked_price_l33_33040


namespace probability_of_at_most_3_heads_l33_33570

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l33_33570


namespace whole_numbers_between_sqrt_18_and_sqrt_98_l33_33669

theorem whole_numbers_between_sqrt_18_and_sqrt_98 :
  let lower_bound := Real.sqrt 18
  let upper_bound := Real.sqrt 98
  let smallest_whole_num := 5
  let largest_whole_num := 9
  (largest_whole_num - smallest_whole_num + 1) = 5 :=
by
  -- Introduce variables
  let lower_bound := Real.sqrt 18
  let upper_bound := Real.sqrt 98
  let smallest_whole_num := 5
  let largest_whole_num := 9
  -- Sorry indicates the proof steps are skipped
  sorry

end whole_numbers_between_sqrt_18_and_sqrt_98_l33_33669


namespace cylinder_height_in_sphere_l33_33617

noncomputable def height_of_cylinder (r R : ℝ) : ℝ :=
  2 * Real.sqrt (R ^ 2 - r ^ 2)

theorem cylinder_height_in_sphere :
  height_of_cylinder 3 6 = 6 * Real.sqrt 3 :=
by
  sorry

end cylinder_height_in_sphere_l33_33617


namespace sum_of_money_proof_l33_33878

noncomputable def total_sum (A B C : ℝ) : ℝ := A + B + C

theorem sum_of_money_proof (A B C : ℝ) (h1 : B = 0.65 * A) (h2 : C = 0.40 * A) (h3 : C = 64) : total_sum A B C = 328 :=
by 
  sorry

end sum_of_money_proof_l33_33878


namespace bart_pages_bought_l33_33911

theorem bart_pages_bought (total_money : ℝ) (price_per_notepad : ℝ) (pages_per_notepad : ℕ)
  (h1 : total_money = 10) (h2 : price_per_notepad = 1.25) (h3 : pages_per_notepad = 60) :
  total_money / price_per_notepad * pages_per_notepad = 480 :=
by
  sorry

end bart_pages_bought_l33_33911


namespace CannotDetermineDraculaStatus_l33_33822

variable (Transylvanian_is_human : Prop)
variable (Dracula_is_alive : Prop)
variable (Statement : Transylvanian_is_human → Dracula_is_alive)

theorem CannotDetermineDraculaStatus : ¬ (∃ (H : Prop), H = Dracula_is_alive) :=
by
  sorry

end CannotDetermineDraculaStatus_l33_33822


namespace number_of_ways_to_buy_three_items_l33_33132

def headphones : ℕ := 9
def mice : ℕ := 13
def keyboards : ℕ := 5
def keyboard_mouse_sets : ℕ := 4
def headphone_mouse_sets : ℕ := 5

theorem number_of_ways_to_buy_three_items : 
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 := 
by 
  sorry

end number_of_ways_to_buy_three_items_l33_33132


namespace mixture_ratios_equal_quantities_l33_33880

-- Define the given conditions
def ratio_p_milk_water := (5, 4)
def ratio_q_milk_water := (2, 7)

-- Define what we're trying to prove: the ratio p : q such that the resulting mixture has equal milk and water
theorem mixture_ratios_equal_quantities 
  (P Q : ℝ) 
  (h1 : 5 * P + 2 * Q = 4 * P + 7 * Q) :
  P / Q = 5 :=
  sorry

end mixture_ratios_equal_quantities_l33_33880


namespace log_base_5_domain_correct_l33_33372

def log_base_5_domain : Set ℝ := {x : ℝ | x > 0}

theorem log_base_5_domain_correct : (∀ x : ℝ, x > 0 ↔ x ∈ log_base_5_domain) :=
by sorry

end log_base_5_domain_correct_l33_33372


namespace sequence_inequality_l33_33479
open Nat

variable (a : ℕ → ℝ)

noncomputable def conditions := 
  (a 1 ≥ 1) ∧ (∀ k : ℕ, a (k + 1) - a k ≥ 1)

theorem sequence_inequality (h : conditions a) : 
  ∀ n : ℕ, a (n + 1) ≥ n + 1 :=
sorry

end sequence_inequality_l33_33479


namespace blue_balls_count_l33_33220

def num_purple : Nat := 7
def num_yellow : Nat := 11
def min_tries : Nat := 19

theorem blue_balls_count (num_blue: Nat): num_blue = 1 :=
by
  have worst_case_picks := num_purple + num_yellow
  have h := min_tries
  sorry

end blue_balls_count_l33_33220


namespace marcia_wardrobe_cost_l33_33154

theorem marcia_wardrobe_cost :
  let skirt_price := 20
  let blouse_price := 15
  let pant_price := 30
  let num_skirts := 3
  let num_blouses := 5
  let num_pants := 2
  let pant_offer := buy_1_get_1_half
  let skirt_cost := num_skirts * skirt_price
  let blouse_cost := num_blouses * blouse_price
  let pant_full_price := pant_price
  let pant_half_price := pant_price / 2
  let pant_cost := pant_full_price + pant_half_price
  let total_cost := skirt_cost + blouse_cost + pant_cost
  total_cost = 180 :=
by
  sorry -- proof is omitted

end marcia_wardrobe_cost_l33_33154


namespace find_pairs_l33_33068

theorem find_pairs (m n: ℕ) (h: m > 0 ∧ n > 0 ∧ m + n - (3 * m * n) / (m + n) = 2011 / 3) : (m = 1144 ∧ n = 377) ∨ (m = 377 ∧ n = 1144) :=
by sorry

end find_pairs_l33_33068


namespace smallest_divisible_by_3_and_4_is_12_l33_33908

theorem smallest_divisible_by_3_and_4_is_12 
  (n : ℕ) 
  (h1 : ∃ k1 : ℕ, n = 3 * k1) 
  (h2 : ∃ k2 : ℕ, n = 4 * k2) 
  : n ≥ 12 := sorry

end smallest_divisible_by_3_and_4_is_12_l33_33908


namespace purchase_combinations_correct_l33_33113

theorem purchase_combinations_correct : 
  let headphones : ℕ := 9
  let mice : ℕ := 13
  let keyboards : ℕ := 5
  let sets_keyboard_mouse : ℕ := 4
  let sets_headphones_mouse : ℕ := 5
in 
  (sets_keyboard_mouse * headphones) + 
  (sets_headphones_mouse * keyboards) + 
  (headphones * mice * keyboards) = 646 := 
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let sets_keyboard_mouse := 4
  let sets_headphones_mouse := 5
  sorry

end purchase_combinations_correct_l33_33113


namespace g_neg_one_l33_33969

def g (d e f x : ℝ) : ℝ := d * x^9 - e * x^5 + f * x + 1

theorem g_neg_one {d e f : ℝ} (h : g d e f 1 = -1) : g d e f (-1) = 3 := by
  sorry

end g_neg_one_l33_33969


namespace solve_fraction_equation_l33_33853

theorem solve_fraction_equation (x : ℝ) :
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ↔ x = -9 :=
by {
  sorry
}

end solve_fraction_equation_l33_33853


namespace solve_for_x_l33_33983

theorem solve_for_x : 
  ∃ x : ℝ, (x^2 + 6 * x + 8 = -(x + 2) * (x + 6)) ∧ (x = -2 ∨ x = -5) :=
sorry

end solve_for_x_l33_33983


namespace surface_area_ratio_l33_33535

theorem surface_area_ratio (x : ℝ) (hx : x > 0) :
  let SA1 := 6 * (4 * x) ^ 2
  let SA2 := 6 * x ^ 2
  (SA1 / SA2) = 16 := by
  sorry

end surface_area_ratio_l33_33535


namespace ratio_QP_l33_33989

noncomputable def P : ℚ := 11 / 6
noncomputable def Q : ℚ := 5 / 2

theorem ratio_QP : Q / P = 15 / 11 := by 
  sorry

end ratio_QP_l33_33989


namespace find_range_of_a_l33_33832

noncomputable def range_of_a : Set ℝ :=
  {a | (∀ x : ℝ, x^2 - 2 * x > a) ∨ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0)}

theorem find_range_of_a :
  {a : ℝ | (∀ x : ℝ, x^2 - 2 * x > a) ∨ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0)} = 
  {a | (-2 < a ∧ a < -1) ∨ (1 ≤ a)} :=
by
  sorry

end find_range_of_a_l33_33832


namespace johns_average_speed_l33_33143

def start_time := 8 * 60 + 15  -- 8:15 a.m. in minutes
def end_time := 14 * 60 + 45   -- 2:45 p.m. in minutes
def break_start := 12 * 60     -- 12:00 p.m. in minutes
def break_duration := 30       -- 30 minutes
def total_distance := 240      -- Total distance in miles

def total_driving_time : ℕ := 
  (break_start - start_time) + (end_time - (break_start + break_duration))

def average_speed (distance : ℕ) (time : ℕ) : ℕ :=
  distance / (time / 60)  -- converting time from minutes to hours

theorem johns_average_speed :
  average_speed total_distance total_driving_time = 40 :=
by
  sorry

end johns_average_speed_l33_33143


namespace sally_bread_consumption_l33_33706

/-
Given:
    1) Sally eats 2 sandwiches on Saturday
    2) Sally eats 1 sandwich on Sunday
    3) Each sandwich uses 2 pieces of bread

Prove:
    Sally eats 6 pieces of bread across Saturday and Sunday
-/
theorem sally_bread_consumption (sandwiches_sat : Nat) (sandwiches_sun : Nat) (bread_per_sandwich : Nat)
    (H1 : sandwiches_sat = 2) (H2 : sandwiches_sun = 1) (H3 : bread_per_sandwich = 2) :
    2 * bread_per_sandwich + 1 * bread_per_sandwich = 6 := by
  sorry

end sally_bread_consumption_l33_33706


namespace sum_of_first_90_terms_l33_33429

def arithmetic_progression_sum (n : ℕ) (a d : ℚ) : ℚ :=
  (n : ℚ) / 2 * (2 * a + (n - 1) * d)

theorem sum_of_first_90_terms (a d : ℚ) :
  (arithmetic_progression_sum 15 a d = 150) →
  (arithmetic_progression_sum 75 a d = 75) →
  (arithmetic_progression_sum 90 a d = -112.5) :=
by
  sorry

end sum_of_first_90_terms_l33_33429


namespace multiplication_results_l33_33763

theorem multiplication_results
  (h1 : 25 * 4 = 100) :
  25 * 8 = 200 ∧ 25 * 12 = 300 ∧ 250 * 40 = 10000 ∧ 25 * 24 = 600 :=
by
  sorry

end multiplication_results_l33_33763


namespace sally_bread_consumption_l33_33707

theorem sally_bread_consumption :
  (2 * 2) + (1 * 2) = 6 :=
by
  sorry

end sally_bread_consumption_l33_33707


namespace average_marks_math_chem_l33_33996

theorem average_marks_math_chem (M P C : ℝ) (h1 : M + P = 60) (h2 : C = P + 20) : 
  (M + C) / 2 = 40 := 
by
  sorry

end average_marks_math_chem_l33_33996


namespace rachel_biology_homework_pages_l33_33978

-- Declare the known quantities
def math_pages : ℕ := 8
def total_math_biology_pages : ℕ := 11

-- Define biology_pages
def biology_pages : ℕ := total_math_biology_pages - math_pages

-- Assert the main theorem
theorem rachel_biology_homework_pages : biology_pages = 3 :=
by 
  -- Proof is omitted as instructed
  sorry

end rachel_biology_homework_pages_l33_33978


namespace find_x_l33_33801

theorem find_x {x y : ℝ} (hx : x ≠ 0) (hy : y ≠ 0)
    (h1 : x + 1/y = 10) (h2 : y + 1/x = 5/12) : x = 4 ∨ x = 6 :=
by
  sorry

end find_x_l33_33801


namespace q_investment_correct_l33_33214

-- Define the conditions
def profit_ratio := (4, 6)
def p_investment := 60000
def expected_q_investment := 90000

-- Define the theorem statement
theorem q_investment_correct (p_investment: ℕ) (q_investment: ℕ) (profit_ratio : ℕ × ℕ)
  (h_ratio: profit_ratio = (4, 6)) (hp_investment: p_investment = 60000) :
  q_investment = 90000 := by
  sorry

end q_investment_correct_l33_33214


namespace area_increase_is_50_l33_33897

def length := 13
def width := 10
def length_new := length + 2
def width_new := width + 2
def area_original := length * width
def area_new := length_new * width_new
def area_increase := area_new - area_original

theorem area_increase_is_50 : area_increase = 50 :=
by
  -- Here we will include the steps to prove the theorem if required
  sorry

end area_increase_is_50_l33_33897


namespace min_additional_matchsticks_needed_l33_33962

-- Define the number of matchsticks in a 3x7 grid
def matchsticks_in_3x7_grid : Nat := 4 * 7 + 3 * 8

-- Define the number of matchsticks in a 5x5 grid
def matchsticks_in_5x5_grid : Nat := 6 * 5 + 6 * 5

-- Define the minimum number of additional matchsticks required
def additional_matchsticks (matchsticks_in_3x7_grid matchsticks_in_5x5_grid : Nat) : Nat :=
  matchsticks_in_5x5_grid - matchsticks_in_3x7_grid

theorem min_additional_matchsticks_needed :
  additional_matchsticks matchsticks_in_3x7_grid matchsticks_in_5x5_grid = 8 :=
by 
  unfold additional_matchsticks matchsticks_in_3x7_grid matchsticks_in_5x5_grid
  sorry

end min_additional_matchsticks_needed_l33_33962


namespace terminating_decimals_l33_33775

theorem terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k : ℕ, k = 20 ∧ ∀ m : ℕ, 1 ≤ m ∧ m ≤ 180 → (decimal_terminates (m / 180) ↔ m % 9 = 0)) :=
by {
  sorry
}

def decimal_terminates (x : ℚ) : Prop :=
  sorry

end terminating_decimals_l33_33775


namespace complex_conjugate_identity_l33_33949

theorem complex_conjugate_identity (z : ℂ) (h : conj z * (1 + I) = 1 - I) : z = I :=
by sorry

end complex_conjugate_identity_l33_33949


namespace rectangle_to_square_l33_33703

theorem rectangle_to_square (a b : ℝ) (h1 : b / 2 < a) (h2 : a < b) :
  ∃ (r : ℝ), r = Real.sqrt (a * b) ∧ 
    (∃ (cut1 cut2 : ℝ × ℝ), 
      cut1.1 = 0 ∧ cut1.2 = a ∧
      cut2.1 = b - r ∧ cut2.2 = r - a ∧
      ∀ t, t = (a * b) - (r ^ 2)) := sorry

end rectangle_to_square_l33_33703


namespace percent_increase_share_price_l33_33879

theorem percent_increase_share_price (P : ℝ) 
  (h1 : ∃ P₁ : ℝ, P₁ = P + 0.25 * P)
  (h2 : ∃ P₂ : ℝ, P₂ = P + 0.80 * P)
  : ∃ percent_increase : ℝ, percent_increase = 44 := by
  sorry

end percent_increase_share_price_l33_33879


namespace sum_of_numbers_l33_33174

theorem sum_of_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a * b + b * c + c * a = 131) : 
  a + b + c = 22 := 
by 
  sorry

end sum_of_numbers_l33_33174


namespace positive_integer_solutions_l33_33838

theorem positive_integer_solutions:
  ∀ (x y : ℕ), (5 * x + y = 11) → (x > 0) → (y > 0) → (x = 1 ∧ y = 6) ∨ (x = 2 ∧ y = 1) :=
by
  sorry

end positive_integer_solutions_l33_33838


namespace smallest_number_greater_than_300_divided_by_25_has_remainder_24_l33_33613

theorem smallest_number_greater_than_300_divided_by_25_has_remainder_24 :
  ∃ x : ℕ, (x > 300) ∧ (x % 25 = 24) ∧ (x = 324) := by
  sorry

end smallest_number_greater_than_300_divided_by_25_has_remainder_24_l33_33613


namespace flowers_brought_at_dawn_l33_33891

theorem flowers_brought_at_dawn (F : ℕ) 
  (h1 : (3 / 5) * F = 180)
  (h2 :  (2 / 5) * F + (F - (3 / 5) * F) = 180) : 
  F = 300 := 
by
  sorry

end flowers_brought_at_dawn_l33_33891


namespace white_pawn_on_white_square_l33_33365

theorem white_pawn_on_white_square (w b N_b N_w : ℕ) (h1 : w > b) (h2 : N_b < N_w) : ∃ k : ℕ, k > 0 :=
by 
  -- Let's assume a contradiction
  -- The proof steps would be written here
  sorry

end white_pawn_on_white_square_l33_33365


namespace tom_annual_car_leasing_cost_l33_33352

theorem tom_annual_car_leasing_cost :
  let miles_mwf := 50 * 3  -- Miles driven on Monday, Wednesday, and Friday
  let miles_other_days := 100 * 4 -- Miles driven on the other days (Sunday, Tuesday, Thursday, Saturday)
  let weekly_miles := miles_mwf + miles_other_days -- Total miles driven per week

  let cost_per_mile := 0.1 -- Cost per mile
  let weekly_fee := 100 -- Weekly fee

  let weekly_cost := weekly_miles * cost_per_mile + weekly_fee -- Total weekly cost

  let weeks_per_year := 52
  let annual_cost := weekly_cost * weeks_per_year -- Annual cost

  annual_cost = 8060 :=
by
  sorry

end tom_annual_car_leasing_cost_l33_33352


namespace min_value_of_quadratic_expression_l33_33397

theorem min_value_of_quadratic_expression (x y z : ℝ) (h : x + y + z = 1) :
  ∃ (u : ℝ), (2 * x^2 + 3 * y^2 + z^2 = u) ∧ u = 6 / 11 :=
sorry

end min_value_of_quadratic_expression_l33_33397


namespace probability_at_most_three_heads_10_coins_l33_33601

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l33_33601


namespace fair_coin_difference_l33_33188

noncomputable def binom_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem fair_coin_difference :
  let p := (1 : ℚ) / 2 in
  let p_1 := binom_coeff 5 4 * p^4 * (1 - p) in
  let p_2 := p^5 in
  p_1 - p_2 = 1 / 8 :=
by
  sorry

end fair_coin_difference_l33_33188


namespace cube_strictly_increasing_l33_33084

theorem cube_strictly_increasing (a b : ℝ) (h : a > b) : a^3 > b^3 :=
sorry

end cube_strictly_increasing_l33_33084


namespace find_triples_tan_l33_33384

open Real

theorem find_triples_tan (x y z : ℝ) :
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) = 4 * x * y * z → 
  ∃ (A B C : ℝ), x = tan A ∧ y = tan B ∧ z = tan C :=
by
  sorry

end find_triples_tan_l33_33384


namespace probability_heads_at_most_3_l33_33576

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l33_33576


namespace largest_possible_expression_value_l33_33670

-- Definition of the conditions.
def distinct_digits (X Y Z : ℕ) : Prop := X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z ∧ X < 10 ∧ Y < 10 ∧ Z < 10

-- The main theorem statement.
theorem largest_possible_expression_value : ∀ (X Y Z : ℕ), distinct_digits X Y Z → 
  (100 * X + 10 * Y + Z - 10 * Z - Y - X) ≤ 900 :=
by
  sorry

end largest_possible_expression_value_l33_33670


namespace max_value_2cosx_3sinx_l33_33633

open Real 

theorem max_value_2cosx_3sinx : ∀ x : ℝ, 2 * cos x + 3 * sin x ≤ sqrt 13 :=
by sorry

end max_value_2cosx_3sinx_l33_33633


namespace min_value_x_plus_y_l33_33511

theorem min_value_x_plus_y (x y : ℤ) (det : 3 < x * y ∧ x * y < 5) : x + y = -5 :=
sorry

end min_value_x_plus_y_l33_33511


namespace annual_average_growth_rate_l33_33042

theorem annual_average_growth_rate (x : ℝ) (h : x > 0): 
  100 * (1 + x)^2 = 169 :=
sorry

end annual_average_growth_rate_l33_33042


namespace original_proposition_converse_proposition_false_inverse_proposition_false_contrapositive_proposition_true_l33_33037

-- Definition of the quadrilateral being a rhombus
def is_rhombus (quad : Type) : Prop := 
-- A quadrilateral is a rhombus if and only if all its sides are equal in length
sorry

-- Definition of the diagonals of quadrilateral being perpendicular
def diagonals_are_perpendicular (quad : Type) : Prop := 
-- The diagonals of a quadrilateral are perpendicular
sorry

-- Original proposition: If a quadrilateral is a rhombus, then its diagonals are perpendicular to each other
theorem original_proposition (quad : Type) : is_rhombus quad → diagonals_are_perpendicular quad :=
sorry

-- Converse proposition: If the diagonals of a quadrilateral are perpendicular to each other, then it is a rhombus, which is False
theorem converse_proposition_false (quad : Type) : diagonals_are_perpendicular quad → ¬ is_rhombus quad :=
sorry

-- Inverse proposition: If a quadrilateral is not a rhombus, then its diagonals are not perpendicular, which is False
theorem inverse_proposition_false (quad : Type) : ¬ is_rhombus quad → ¬ diagonals_are_perpendicular quad :=
sorry

-- Contrapositive proposition: If the diagonals of a quadrilateral are not perpendicular, then it is not a rhombus, which is True
theorem contrapositive_proposition_true (quad : Type) : ¬ diagonals_are_perpendicular quad → ¬ is_rhombus quad :=
sorry

end original_proposition_converse_proposition_false_inverse_proposition_false_contrapositive_proposition_true_l33_33037


namespace rectangle_length_reduction_l33_33341

theorem rectangle_length_reduction (L W : ℝ) (h : L > 0 ∧ W > 0) :
  let new_length := L * (1 - 10 / 100)
  let new_width := W * (10 / 9)
  (new_length * new_width = L * W) → 
  x = 10 := by sorry

end rectangle_length_reduction_l33_33341


namespace coin_flip_probability_difference_l33_33191

theorem coin_flip_probability_difference :
  let p1 := (nat.choose 5 4) * (1 / 2)^5
  let p2 := (1 / 2)^5
  (p1 - p2 = 1 / 8) :=
by
  let p1 := (nat.choose 5 4) * (1 / 2)^5
  let p2 := (1 / 2)^5
  sorry

end coin_flip_probability_difference_l33_33191


namespace mean_of_all_students_is_76_l33_33484

-- Definitions
def M : ℝ := 84
def A : ℝ := 70
def ratio_m_a : ℝ := 3 / 4

-- Theorem statement
theorem mean_of_all_students_is_76 (m a : ℝ) (hm : m = ratio_m_a * a) : 
  ((63 * a) + 70 * a) / ((3 / 4 * a) + a) = 76 := 
sorry

end mean_of_all_students_is_76_l33_33484


namespace trapezoid_third_largest_angle_l33_33224

theorem trapezoid_third_largest_angle (a d : ℝ)
  (h1 : 2 * a + 3 * d = 200)      -- Condition: 2a + 3d = 200°
  (h2 : a + d = 70) :             -- Condition: a + d = 70°
  a + 2 * d = 130 :=              -- Question: Prove a + 2d = 130°
by
  sorry

end trapezoid_third_largest_angle_l33_33224


namespace slips_prob_ratio_l33_33066

open Nat
open Rat

def slips := finset.range 10 + 1

def num_slips (n : nat) : nat :=
  if n ∈ finset.range 5 + 1 then 5 else if n ∈ finset.range 6 + 5 then 3 else 0

def total_ways : ℕ := nat.choose 50 4

def r : rat := (5 * nat.choose 5 4 : ℕ) / total_ways

def s : rat := 
  (finset.sum (finset.range 5 + 1) (λ c, (finset.sum (finset.range 6 + 5) (λ d, (c ≠ d) * (nat.choose 5 2 * nat.choose 3 2 : ℕ)))) : ℕ) / total_ways

theorem slips_prob_ratio : s / r = 30 :=
  sorry

end slips_prob_ratio_l33_33066


namespace probability_10_coins_at_most_3_heads_l33_33563

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l33_33563


namespace three_players_same_number_of_flips_l33_33735

noncomputable def biased_coin_outcome (heads_prob tails_prob : ℚ) (n : ℕ) : ℚ :=
  (tails_prob)^(n - 1) * heads_prob

theorem three_players_same_number_of_flips :
  let heads_prob : ℚ := 1 / 3
  let tails_prob : ℚ := 2 / 3
  let P (n : ℕ) : ℚ := (biased_coin_outcome heads_prob tails_prob n)^3
  (∑ n in filter (λ n, n ≥ 1) (range ∞), P n) = 1 / 19 :=
by sorry

end three_players_same_number_of_flips_l33_33735


namespace lukas_games_played_l33_33481

-- Define the given conditions
def average_points_per_game : ℕ := 12
def total_points_scored : ℕ := 60

-- Define Lukas' number of games
def number_of_games (total_points : ℕ) (average_points : ℕ) : ℕ :=
  total_points / average_points

-- Theorem and statement to prove
theorem lukas_games_played :
  number_of_games total_points_scored average_points_per_game = 5 :=
by
  sorry

end lukas_games_played_l33_33481


namespace alcohol_percentage_in_mixed_solution_l33_33364

theorem alcohol_percentage_in_mixed_solution :
  let vol1 := 8
  let perc1 := 0.25
  let vol2 := 2
  let perc2 := 0.12
  let total_alcohol := (vol1 * perc1) + (vol2 * perc2)
  let total_volume := vol1 + vol2
  (total_alcohol / total_volume) * 100 = 22.4 := by
  sorry

end alcohol_percentage_in_mixed_solution_l33_33364


namespace smallest_x_for_M_squared_l33_33930

theorem smallest_x_for_M_squared (M x : ℤ) (h1 : 540 = 2^2 * 3^3 * 5) (h2 : 540 * x = M^2) (h3 : x > 0) : x = 15 :=
sorry

end smallest_x_for_M_squared_l33_33930


namespace point_not_on_line_l33_33400

theorem point_not_on_line (a c : ℝ) (h : a * c > 0) : ¬(0 = 2500 * a + c) := by
  sorry

end point_not_on_line_l33_33400


namespace area_of_inscribed_rectangle_l33_33157

theorem area_of_inscribed_rectangle (h_triangle_altitude : 12 > 0)
  (h_segment_XZ : 15 > 0)
  (h_PQ_eq_one_third_PS : ∀ PQ PS : ℚ, PS = 3 * PQ) :
  ∃ PQ PS : ℚ, 
    (YM = 12) ∧
    (XZ = 15) ∧
    (PQ = (15 / 8 : ℚ)) ∧
    (PS = 3 * PQ) ∧ 
    ((PQ * PS) = (675 / 64 : ℚ)) :=
by
  -- Proof would go here.
  sorry

end area_of_inscribed_rectangle_l33_33157


namespace coin_flip_difference_l33_33195

/-- The positive difference between the probability of a fair coin landing heads up
exactly 4 times out of 5 flips and the probability of a fair coin landing heads up
5 times out of 5 flips is 1/8. -/
theorem coin_flip_difference :
  (5 * (1 / 2) ^ 5) - ((1 / 2) ^ 5) = (1 / 8) :=
by
  sorry

end coin_flip_difference_l33_33195


namespace sum_of_distinct_roots_eq_zero_l33_33477

theorem sum_of_distinct_roots_eq_zero
  (a b m n p : ℝ)
  (h1 : m ≠ n)
  (h2 : m ≠ p)
  (h3 : n ≠ p)
  (h_m : m^3 + a * m + b = 0)
  (h_n : n^3 + a * n + b = 0)
  (h_p : p^3 + a * p + b = 0) : 
  m + n + p = 0 :=
sorry

end sum_of_distinct_roots_eq_zero_l33_33477


namespace area_of_smallest_square_containing_circle_l33_33022

theorem area_of_smallest_square_containing_circle (r : ℝ) (h : r = 7) : ∃ a : ℝ, a = 196 := 
by
  use (2 * r) ^ 2
  rw h
  norm_num

end area_of_smallest_square_containing_circle_l33_33022


namespace annual_growth_rate_l33_33043

theorem annual_growth_rate (u_2021 u_2023 : ℝ) (x : ℝ) : 
    u_2021 = 1 ∧ u_2023 = 1.69 ∧ x > 0 → (u_2023 / u_2021) = (1 + x)^2 → x * 100 = 30 :=
by
  intros h1 h2
  sorry

end annual_growth_rate_l33_33043


namespace greendale_high_school_points_l33_33311

-- Define the conditions
def roosevelt_points_first_game : ℕ := 30
def roosevelt_points_second_game : ℕ := roosevelt_points_first_game / 2
def roosevelt_points_third_game : ℕ := roosevelt_points_second_game * 3
def roosevelt_total_points : ℕ := roosevelt_points_first_game + roosevelt_points_second_game + roosevelt_points_third_game
def roosevelt_total_with_bonus : ℕ := roosevelt_total_points + 50
def difference : ℕ := 10

-- Define the assertion for Greendale's points
def greendale_points : ℕ := roosevelt_total_with_bonus - difference

-- The theorem to be proved
theorem greendale_high_school_points : greendale_points = 130 :=
by
  -- Proof should be here, but we add sorry to skip it as per the instructions
  sorry

end greendale_high_school_points_l33_33311


namespace total_ways_to_buy_l33_33126

-- Definitions:
def h : ℕ := 9  -- number of headphones
def m : ℕ := 13 -- number of mice
def k : ℕ := 5  -- number of keyboards
def s_km : ℕ := 4 -- number of sets of keyboard and mouse
def s_hm : ℕ := 5 -- number of sets of headphones and mouse

-- Theorem stating the number of ways to buy three items is 646.
theorem total_ways_to_buy : (s_km * h) + (s_hm * k) + (h * m * k) = 646 := by
  -- Placeholder proof using sorry.
  sorry

end total_ways_to_buy_l33_33126


namespace count_four_digit_numbers_divisible_by_5_end_45_l33_33412

theorem count_four_digit_numbers_divisible_by_5_end_45 : 
  {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ n % 100 = 45 ∧ n % 5 = 0}.to_finset.card = 90 :=
by
  sorry

end count_four_digit_numbers_divisible_by_5_end_45_l33_33412


namespace max_integer_value_l33_33671

theorem max_integer_value (x : ℝ) : 
  ∃ m : ℤ, ∀ (x : ℝ), (3 * x^2 + 9 * x + 17) / (3 * x^2 + 9 * x + 7) ≤ m ∧ m = 41 :=
by sorry

end max_integer_value_l33_33671


namespace rectangle_ratio_l33_33179

/-- Conditions:
1. There are three identical squares and two rectangles forming a large square.
2. Each rectangle shares one side with a square and another side with the edge of the large square.
3. The side length of each square is 1 unit.
4. The total side length of the large square is 5 units.
Question:
What is the ratio of the length to the width of one of the rectangles? --/

theorem rectangle_ratio (sq_len : ℝ) (large_sq_len : ℝ) (side_ratio : ℝ) :
  sq_len = 1 ∧ large_sq_len = 5 ∧ 
  (∀ (rect_len rect_wid : ℝ), 3 * sq_len + 2 * rect_len = large_sq_len ∧ side_ratio = rect_len / rect_wid) →
  side_ratio = 1 / 2 :=
by
  sorry

end rectangle_ratio_l33_33179


namespace find_first_type_cookies_l33_33055

section CookiesProof

variable (x : ℕ)

-- Conditions
def box_first_type_cookies : ℕ := x
def box_second_type_cookies : ℕ := 20
def box_third_type_cookies : ℕ := 16
def boxes_first_type_sold : ℕ := 50
def boxes_second_type_sold : ℕ := 80
def boxes_third_type_sold : ℕ := 70
def total_cookies_sold : ℕ := 3320

-- Theorem to prove
theorem find_first_type_cookies 
  (h1 : 50 * x + 80 * box_second_type_cookies + 70 * box_third_type_cookies = total_cookies_sold) :
  x = 12 := by
    sorry

end CookiesProof

end find_first_type_cookies_l33_33055


namespace bottle_caps_per_person_l33_33829

noncomputable def initial_caps : Nat := 150
noncomputable def rebecca_caps : Nat := 42
noncomputable def alex_caps : Nat := 2 * rebecca_caps
noncomputable def total_caps : Nat := initial_caps + rebecca_caps + alex_caps
noncomputable def number_of_people : Nat := 6

theorem bottle_caps_per_person : total_caps / number_of_people = 46 := by
  sorry

end bottle_caps_per_person_l33_33829


namespace right_triangle_area_l33_33526

theorem right_triangle_area (leg1 leg2 hypotenuse : ℕ) (h_leg1 : leg1 = 30)
  (h_hypotenuse : hypotenuse = 34)
  (hypotenuse_sq : hypotenuse * hypotenuse = leg1 * leg1 + leg2 * leg2) :
  (1 / 2 : ℚ) * leg1 * leg2 = 240 := by
  sorry

end right_triangle_area_l33_33526


namespace oleg_bought_bar_for_60_rubles_l33_33696

theorem oleg_bought_bar_for_60_rubles (n : ℕ) (h₁ : 96 = n * (1 + n / 100)) : n = 60 :=
by {
  sorry
}

end oleg_bought_bar_for_60_rubles_l33_33696


namespace express_114_as_ones_and_threes_with_min_ten_ones_l33_33259

theorem express_114_as_ones_and_threes_with_min_ten_ones :
  ∃n: ℕ, n = 35 ∧ ∃ x y : ℕ, x + 3 * y = 114 ∧ x ≥ 10 := sorry

end express_114_as_ones_and_threes_with_min_ten_ones_l33_33259


namespace find_third_number_l33_33163

theorem find_third_number 
  (h1 : (14 + 32 + x) / 3 = (21 + 47 + 22) / 3 + 3) : x = 53 := by
  sorry

end find_third_number_l33_33163


namespace total_viewing_time_l33_33910

theorem total_viewing_time :
  let original_times := [4, 6, 7, 5, 9]
  let new_species_times := [3, 7, 8, 10]
  let total_breaks := 8
  let break_time_per_animal := 2
  let total_time := (original_times.sum + new_species_times.sum) + (total_breaks * break_time_per_animal)
  total_time = 75 :=
by
  sorry

end total_viewing_time_l33_33910


namespace solve_equation_l33_33850

theorem solve_equation (x : ℝ) :
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) → x = -9 :=
by 
  sorry

end solve_equation_l33_33850


namespace fencing_cost_is_correct_l33_33500

def length : ℕ := 60
def cost_per_meter : ℕ := 27 -- using the closest integer value to 26.50
def breadth (l : ℕ) : ℕ := l - 20
def perimeter (l b : ℕ) : ℕ := 2 * l + 2 * b
def total_cost (P : ℕ) (c : ℕ) : ℕ := P * c

theorem fencing_cost_is_correct :
  total_cost (perimeter length (breadth length)) cost_per_meter = 5300 :=
  sorry

end fencing_cost_is_correct_l33_33500


namespace greatest_value_of_q_sub_r_l33_33723

-- Definitions and conditions as given in the problem statement
def q : ℕ := 44
def r : ℕ := 15
def n : ℕ := 1027
def d : ℕ := 23

-- Statement of the theorem to be proved
theorem greatest_value_of_q_sub_r : 
  ∃ (q r : ℕ), n = d * q + r ∧ 1 ≤ q ∧ 1 ≤ r ∧ q - r = 29 := 
by
  use q
  use r
  have h1 : n = d * q + r := by norm_num
  have h2 : 1 ≤ q := by norm_num
  have h3 : 1 ≤ r := by norm_num
  have h4 : q - r = 29 := by norm_num
  tauto

end greatest_value_of_q_sub_r_l33_33723


namespace chord_length_l33_33943

theorem chord_length (ρ θ : ℝ) (p : ℝ) : 
  (∀ θ, ρ = 6 * Real.cos θ) ∧ (θ = Real.pi / 4) → 
  ∃ l : ℝ, l = 3 * Real.sqrt 2 :=
by
  sorry

end chord_length_l33_33943


namespace sam_mary_total_balloons_l33_33979

def Sam_initial_balloons : ℝ := 6.0
def Sam_gives : ℝ := 5.0
def Sam_remaining_balloons : ℝ := Sam_initial_balloons - Sam_gives

def Mary_balloons : ℝ := 7.0

def total_balloons : ℝ := Sam_remaining_balloons + Mary_balloons

theorem sam_mary_total_balloons : total_balloons = 8.0 :=
by
  sorry

end sam_mary_total_balloons_l33_33979


namespace intersection_height_correct_l33_33106

noncomputable def intersection_height 
  (height_pole_1 height_pole_2 distance : ℝ) : ℝ := 
  let slope_1 := -(height_pole_1 / distance)
  let slope_2 := height_pole_2 / distance
  let y_intercept_1 := height_pole_1
  let y_intercept_2 := 0
  let x_intersection := height_pole_1 / (slope_2 - slope_1)
  let y_intersection := slope_2 * x_intersection + y_intercept_2
  y_intersection

theorem intersection_height_correct 
  : intersection_height 30 90 150 = 22.5 := 
by sorry

end intersection_height_correct_l33_33106


namespace max_participants_win_at_least_three_matches_l33_33611

theorem max_participants_win_at_least_three_matches (n : ℕ) (h : n = 200) : 
  ∃ k : ℕ, k = 66 ∧ ∀ m : ℕ, (k * 3 ≤ m) ∧ (m ≤ 199) → k ≤ m / 3 := 
by
  sorry

end max_participants_win_at_least_three_matches_l33_33611


namespace probability_heads_at_most_3_l33_33572

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l33_33572


namespace measure_angle_4_l33_33634

theorem measure_angle_4 (m1 m2 m3 m5 m6 m4 : ℝ) 
  (h1 : m1 = 82) 
  (h2 : m2 = 34) 
  (h3 : m3 = 19) 
  (h4 : m5 = m6 + 10) 
  (h5 : m1 + m2 + m3 + m5 + m6 = 180)
  (h6 : m4 + m5 + m6 = 180) : 
  m4 = 135 :=
by
  -- Placeholder for the full proof, omitted due to instructions
  sorry

end measure_angle_4_l33_33634


namespace diamond_expression_evaluation_l33_33635

def diamond (a b : ℚ) : ℚ := a - (1 / b)

theorem diamond_expression_evaluation :
  ((diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4))) = -29 / 132 :=
by {
    sorry
}

end diamond_expression_evaluation_l33_33635


namespace find_z_l33_33089

variable {x y z w : ℝ}

theorem find_z (h : (1/x) + (1/y) = (1/z) + w) : z = (x * y) / (x + y - w * x * y) :=
by sorry

end find_z_l33_33089


namespace infinite_series_computation_l33_33631

theorem infinite_series_computation : 
  ∑' k : ℕ, (8^k) / ((2^k - 1) * (2^(k + 1) - 1)) = 4 :=
by
  sorry

end infinite_series_computation_l33_33631


namespace simplify_complex_fraction_l33_33846

-- Define the complex numbers involved
def numerator := 3 + 4 * Complex.I
def denominator := 5 - 2 * Complex.I

-- Define what we need to prove: the simplified form
theorem simplify_complex_fraction : 
    (numerator / denominator : Complex) = (7 / 29) + (26 / 29) * Complex.I := 
by
  -- Proof is omitted here
  sorry

end simplify_complex_fraction_l33_33846


namespace probability_10_coins_at_most_3_heads_l33_33565

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l33_33565


namespace phil_books_remaining_pages_l33_33306

/-- We define the initial number of books and the number of pages per book. -/
def initial_books : Nat := 10
def pages_per_book : Nat := 100
def lost_books : Nat := 2

/-- The goal is to find the total number of pages Phil has left after losing 2 books. -/
theorem phil_books_remaining_pages : (initial_books - lost_books) * pages_per_book = 800 := by 
  -- The proof will go here
  sorry

end phil_books_remaining_pages_l33_33306


namespace opposite_of_neg_five_l33_33503

theorem opposite_of_neg_five : ∃ (y : ℤ), -5 + y = 0 ∧ y = 5 :=
by
  use 5
  simp

end opposite_of_neg_five_l33_33503


namespace right_triangles_with_leg_2012_l33_33463

theorem right_triangles_with_leg_2012 :
  ∃ (a b c : ℕ), (a = 2012 ∧ (a^2 + b^2 = c^2 ∨ b^2 + a^2 = c^2)) ∧
  (b = 253005 ∧ c = 253013 ∨ b = 506016 ∧ c = 506020 ∨ b = 1012035 ∧ c = 1012037 ∨ b = 1509 ∧ c = 2515) :=
begin
  sorry
end

end right_triangles_with_leg_2012_l33_33463


namespace walt_total_interest_l33_33737

noncomputable def interest_8_percent (P_8 R_8 : ℝ) : ℝ :=
  P_8 * R_8

noncomputable def remaining_amount (P_total P_8 : ℝ) : ℝ :=
  P_total - P_8

noncomputable def interest_9_percent (P_9 R_9 : ℝ) : ℝ :=
  P_9 * R_9

noncomputable def total_interest (I_8 I_9 : ℝ) : ℝ :=
  I_8 + I_9

theorem walt_total_interest :
  let P_8 := 4000
  let R_8 := 0.08
  let P_total := 9000
  let R_9 := 0.09
  let I_8 := interest_8_percent P_8 R_8
  let P_9 := remaining_amount P_total P_8
  let I_9 := interest_9_percent P_9 R_9
  let I_total := total_interest I_8 I_9
  I_total = 770 := 
by
  sorry

end walt_total_interest_l33_33737


namespace island_length_l33_33620

theorem island_length (area width : ℝ) (h_area : area = 50) (h_width : width = 5) : 
  area / width = 10 := 
by
  sorry

end island_length_l33_33620


namespace cloak_change_l33_33454

theorem cloak_change (silver_for_cloak1 silver_for_cloak2 : ℕ) (gold_change1 gold_change2 : ℕ) (silver_change : ℕ) :
  silver_for_cloak1 = 20 →
  gold_change1 = 4 →
  silver_for_cloak2 = 15 →
  gold_change2 = 1 →
  ∃ silver_cost_of_cloak : ℕ, 
    silver_cost_of_cloak = (20 - 4) * (5 / 3) →
    silver_change = 10 →
    14 * (5 / 3) - 8 = silver_change :=
by 
  assume h1 h2 h3 h4,
  use 16, 
  sorry

end cloak_change_l33_33454


namespace find_number_l33_33931

theorem find_number (x : ℤ) (h : 5 + x * 5 = 15) : x = 2 :=
by
  sorry

end find_number_l33_33931


namespace decreases_as_x_increases_l33_33761

theorem decreases_as_x_increases (f : ℝ → ℝ) (h₁ : f = λ x, x^2 + 1) (h₂ : f = λ x, -x^2 + 1) 
  (h₃ : f = λ x, 2x + 1) (h₄ : f = λ x, -2x + 1) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) ↔ f = λ x, -2x + 1 :=
sorry

end decreases_as_x_increases_l33_33761


namespace purchase_combinations_correct_l33_33112

theorem purchase_combinations_correct : 
  let headphones : ℕ := 9
  let mice : ℕ := 13
  let keyboards : ℕ := 5
  let sets_keyboard_mouse : ℕ := 4
  let sets_headphones_mouse : ℕ := 5
in 
  (sets_keyboard_mouse * headphones) + 
  (sets_headphones_mouse * keyboards) + 
  (headphones * mice * keyboards) = 646 := 
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let sets_keyboard_mouse := 4
  let sets_headphones_mouse := 5
  sorry

end purchase_combinations_correct_l33_33112


namespace additional_boys_went_down_slide_l33_33873

theorem additional_boys_went_down_slide (initial_boys total_boys additional_boys : ℕ) (h1 : initial_boys = 22) (h2 : total_boys = 35) : additional_boys = 13 :=
by {
    -- Proof body will be here
    sorry
}

end additional_boys_went_down_slide_l33_33873


namespace total_cost_l33_33745

noncomputable def C1 : ℝ := 990 / 1.10
noncomputable def C2 : ℝ := 990 / 0.90

theorem total_cost (SP : ℝ) (profit_rate loss_rate : ℝ) : SP = 990 ∧ profit_rate = 0.10 ∧ loss_rate = 0.10 →
  C1 + C2 = 2000 :=
by
  intro h
  -- Show the sum of C1 and C2 equals 2000
  sorry

end total_cost_l33_33745


namespace chloe_paid_per_dozen_l33_33766

-- Definitions based on conditions
def half_dozen_sale_price : ℕ := 30
def profit : ℕ := 500
def dozens_sold : ℕ := 50
def full_dozen_sale_price := 2 * half_dozen_sale_price
def total_revenue := dozens_sold * full_dozen_sale_price
def total_cost := total_revenue - profit

-- Proof problem
theorem chloe_paid_per_dozen : (total_cost / dozens_sold) = 50 :=
by
  sorry

end chloe_paid_per_dozen_l33_33766


namespace simplify_and_evaluate_expression_l33_33320

theorem simplify_and_evaluate_expression :
  let x := -1
  let y := Real.sqrt 2
  (x + y) * (x - y) - (4 * x^3 * y - 8 * x * y^3) / (2 * x * y) = 5 :=
by
  let x := -1
  let y := Real.sqrt 2
  sorry

end simplify_and_evaluate_expression_l33_33320


namespace find_evening_tickets_l33_33343

noncomputable def matinee_price : ℕ := 5
noncomputable def evening_price : ℕ := 12
noncomputable def threeD_price : ℕ := 20
noncomputable def matinee_tickets : ℕ := 200
noncomputable def threeD_tickets : ℕ := 100
noncomputable def total_revenue : ℕ := 6600

theorem find_evening_tickets (E : ℕ) (hE : total_revenue = matinee_tickets * matinee_price + E * evening_price + threeD_tickets * threeD_price) :
  E = 300 :=
by
  sorry

end find_evening_tickets_l33_33343


namespace solution1_solution2_l33_33495

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

end solution1_solution2_l33_33495


namespace hyperbola_equation_constant_value_hyperbola_l33_33798

def hyperbola (x y : ℝ): Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x / a)^2 - (y / b)^2 = 1

def point_on_hyperbola (x y a b : ℝ): Prop :=
 (x / a)^2 - (y / b)^2 = 1

def eccentricity (a c : ℝ): ℝ := c / a 

noncomputable def constants (a b c : ℝ): Prop :=
  b = 2 * real.sqrt a ∧ c = 4

theorem hyperbola_equation (x y a b c : ℝ) (h1: a > 0) (h2: point_on_hyperbola 4 6 a b) (h3: eccentricity a c = 2) : 
  b = 2 * real.sqrt a ∧ c = 4 ∧ (x = 4 / 3 ∧ y = 0) :=
sorry

theorem constant_value_hyperbola (A B F2x F2y : ℝ) (h1: ∃ l, ∃ m : Prop, line_through l m F2x F2y A B ∧ F2x = 4 ∧ A ≠ V ∧ B ≠ V) :
  ∃ k : ℝ, k = 1 / 3 ∧ ∀ A F2 B,  abs ((1 / (dist (A F2x))) - (1 / (dist (B F2x)))) = k :=
sorry

end hyperbola_equation_constant_value_hyperbola_l33_33798


namespace fencing_cost_proof_l33_33501

theorem fencing_cost_proof (L : ℝ) (B : ℝ) (c : ℝ) (total_cost : ℝ)
  (hL : L = 60) (hL_B : L = B + 20) (hc : c = 26.50) : 
  total_cost = 5300 :=
by
  sorry

end fencing_cost_proof_l33_33501


namespace shirts_per_kid_l33_33141

-- Define given conditions
def n_buttons : Nat := 63
def buttons_per_shirt : Nat := 7
def n_kids : Nat := 3

-- The proof goal
theorem shirts_per_kid : (n_buttons / buttons_per_shirt) / n_kids = 3 := by
  sorry

end shirts_per_kid_l33_33141


namespace subtract_fractions_l33_33035

theorem subtract_fractions (p q : ℚ) (h₁ : 4 / p = 8) (h₂ : 4 / q = 18) : p - q = 5 / 18 := 
by 
  sorry

end subtract_fractions_l33_33035


namespace positive_difference_between_probabilities_is_one_eighth_l33_33199

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem positive_difference_between_probabilities_is_one_eighth :
  let p : ℚ := 1 / 2
    n : ℕ := 5
    prob_4_heads := binomial_probability n 4 p
    prob_5_heads := binomial_probability n 5 p
  in |prob_4_heads - prob_5_heads| = 1 / 8 :=
by
  let p : ℚ := 1 / 2
  let n : ℕ := 5
  let prob_4_heads := binomial_probability n 4 p
  let prob_5_heads := binomial_probability n 5 p
  sorry

end positive_difference_between_probabilities_is_one_eighth_l33_33199


namespace more_pens_than_pencils_l33_33228

-- Define the number of pencils (P) and pens (Pe)
def num_pencils : ℕ := 15 * 80

-- Define the number of pens (Pe) is more than twice the number of pencils (P)
def num_pens (Pe : ℕ) : Prop := Pe > 2 * num_pencils

-- State the total cost equation in terms of pens and pencils
def total_cost_eq (Pe : ℕ) : Prop := (5 * Pe + 4 * num_pencils = 18300)

-- Prove that the number of more pens than pencils is 1500
theorem more_pens_than_pencils (Pe : ℕ) (h1 : num_pens Pe) (h2 : total_cost_eq Pe) : (Pe - num_pencils = 1500) :=
by
  sorry

end more_pens_than_pencils_l33_33228


namespace all_possible_values_of_k_l33_33095

def is_partition_possible (k : ℕ) : Prop :=
  ∃ (A B : Finset ℕ), (A ∪ B = Finset.range (k + 1)) ∧ (A ∩ B = ∅) ∧ (A.sum id = 2 * B.sum id)

theorem all_possible_values_of_k (k : ℕ) : 
  is_partition_possible k → ∃ m : ℕ, k = 3 * m ∨ k = 3 * m - 1 :=
by
  intro h
  sorry

end all_possible_values_of_k_l33_33095


namespace probability_non_red_face_l33_33326

theorem probability_non_red_face :
  let red_faces := 5
  let yellow_faces := 3
  let blue_faces := 1
  let green_faces := 1
  let total_faces := 10
  let non_red_faces := yellow_faces + blue_faces + green_faces
  (non_red_faces / total_faces : ℚ) = 1 / 2 :=
by
  -- Using Lean's Rat library for rational numbers.
  let red_faces := 5
  let yellow_faces := 3
  let blue_faces := 1
  let green_faces := 1
  let total_faces := 10
  let non_red_faces := yellow_faces + blue_faces + green_faces
  show (non_red_faces / total_faces : ℚ) = 1 / 2
  sorry

end probability_non_red_face_l33_33326


namespace area_of_right_triangle_l33_33520

-- Given definitions
def leg_a : ℝ := 30
def hypotenuse_c : ℝ := 34

-- The theorem statement
theorem area_of_right_triangle : 
  ∀ (b : ℝ), b = real.sqrt (hypotenuse_c^2 - leg_a^2) → 
  let area := 1 / 2 * leg_a * b in
  area = 240 := 
by
  intro b
  intro h
  let area := 1 / 2 * leg_a * b
  sorry

end area_of_right_triangle_l33_33520


namespace fair_coin_difference_l33_33190

noncomputable def binom_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem fair_coin_difference :
  let p := (1 : ℚ) / 2 in
  let p_1 := binom_coeff 5 4 * p^4 * (1 - p) in
  let p_2 := p^5 in
  p_1 - p_2 = 1 / 8 :=
by
  sorry

end fair_coin_difference_l33_33190


namespace greendale_high_school_points_l33_33309

-- Define the conditions
def roosevelt_points_first_game : ℕ := 30
def roosevelt_points_second_game : ℕ := roosevelt_points_first_game / 2
def roosevelt_points_third_game : ℕ := roosevelt_points_second_game * 3
def roosevelt_total_points : ℕ := roosevelt_points_first_game + roosevelt_points_second_game + roosevelt_points_third_game
def roosevelt_total_with_bonus : ℕ := roosevelt_total_points + 50
def difference : ℕ := 10

-- Define the assertion for Greendale's points
def greendale_points : ℕ := roosevelt_total_with_bonus - difference

-- The theorem to be proved
theorem greendale_high_school_points : greendale_points = 130 :=
by
  -- Proof should be here, but we add sorry to skip it as per the instructions
  sorry

end greendale_high_school_points_l33_33309


namespace find_j_value_l33_33986

variable {R : Type*} [LinearOrderedField R]

-- Definitions based on conditions
def polynomial_has_four_distinct_real_roots_in_arithmetic_progression
(p : Polynomial R) : Prop :=
∃ a d : R, p.roots.toFinset = {a, a + d, a + 2*d, a + 3*d} ∧
a ≠ a + d ∧ a ≠ a + 2*d ∧ a ≠ a + 3*d ∧ a + d ≠ a + 2*d ∧
a + d ≠ a + 3*d ∧ a + 2*d ≠ a + 3*d

-- The main theorem statement
theorem find_j_value (k : R) 
  (h : polynomial_has_four_distinct_real_roots_in_arithmetic_progression 
  (Polynomial.X^4 + Polynomial.C j * Polynomial.X^2 + Polynomial.C k * Polynomial.X + Polynomial.C 900)) :
  j = -900 :=
sorry

end find_j_value_l33_33986


namespace product_of_real_parts_eq_neg_half_l33_33000

noncomputable def prod_real_parts_of_solutions (h : ℂ) (x : ℂ) :=
  let s1 := x - 2 + complex.sqrt 5 * complex.exp (complex.I * real.atan (3 / 4) / 2) in
  let s2 := x - 2 - complex.sqrt 5 * complex.exp (complex.I * real.atan (3 / 4) / 2) in
  re s1 * re s2

theorem product_of_real_parts_eq_neg_half (c1 c2 : ℂ) (h : c1^2 - 4 * c1 = 3 * complex.I)
  (h2 : c2^2 - 4 * c2 = 3 * complex.I) :
  re c1 * re c2 = -0.5 :=
by
  sorry

end product_of_real_parts_eq_neg_half_l33_33000


namespace no_convex_27gon_with_distinct_integer_angles_l33_33637

noncomputable def sum_of_interior_angles (n : ℕ) : ℕ :=
  (n - 2) * 180

def is_convex (n : ℕ) (angles : Fin n → ℕ) : Prop :=
  ∀ i, angles i < 180

def all_distinct (n : ℕ) (angles : Fin n → ℕ) : Prop :=
  ∀ i j, i ≠ j → angles i ≠ angles j

def sum_is_correct (n : ℕ) (angles : Fin n → ℕ) : Prop :=
  Finset.sum (Finset.univ : Finset (Fin n)) angles = sum_of_interior_angles n

theorem no_convex_27gon_with_distinct_integer_angles :
  ¬ ∃ (angles : Fin 27 → ℕ), is_convex 27 angles ∧ all_distinct 27 angles ∧ sum_is_correct 27 angles :=
by
  sorry

end no_convex_27gon_with_distinct_integer_angles_l33_33637


namespace hyogeun_weight_l33_33418

noncomputable def weights_are_correct : Prop :=
  ∃ H S G : ℝ, 
    H + S + G = 106.6 ∧
    G = S - 7.7 ∧
    S = H - 4.8 ∧
    H = 41.3

theorem hyogeun_weight : weights_are_correct :=
by
  sorry

end hyogeun_weight_l33_33418


namespace equivalence_statements_l33_33029

variables (P Q : Prop)

theorem equivalence_statements :
  (P → Q) ↔ ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) :=
by sorry

end equivalence_statements_l33_33029


namespace max_participants_won_at_least_three_matches_l33_33609

theorem max_participants_won_at_least_three_matches :
  ∀ (n : ℕ), n = 200 → ∃ k : ℕ, k ≤ 66 ∧ ∀ p : ℕ, (p ≥ k ∧ p > 66) → false := by
  sorry

end max_participants_won_at_least_three_matches_l33_33609


namespace lambda_range_l33_33800

theorem lambda_range (λ : ℝ) (a : ℕ → ℝ) (inc : ∀ n : ℕ, n > 0 → a n < a (n + 1)) :
  (∀ n : ℕ, n > 0 → a n = n^2 + λ * n) → λ > -3 :=
by
  intro h
  have h1 : a 1 = 1 + λ := by rw [h 1 (by norm_num)]
  have h2 : a 2 = 4 + 2 * λ := by rw [h 2 (by norm_num)]
  sorry

end lambda_range_l33_33800


namespace sqrt_infinite_nest_eq_two_l33_33262

theorem sqrt_infinite_nest_eq_two (y : ℝ) (h : y = Real.sqrt (2 + y)) : y = 2 := 
sorry

end sqrt_infinite_nest_eq_two_l33_33262


namespace find_z_l33_33950

theorem find_z (z : ℂ) (hz : conj z * (1 + complex.I) = 1 - complex.I) : 
  z = complex.I :=
by 
  sorry

end find_z_l33_33950


namespace measure_of_angle_F_l33_33182

theorem measure_of_angle_F (x : ℝ) (h1 : ∠D = ∠E) (h2 : ∠F = x + 40) (h3 : 2 * x + (x + 40) = 180) : 
  ∠F = 86.67 :=
by 
  sorry

end measure_of_angle_F_l33_33182


namespace pipe_fill_time_l33_33698

variable (t : ℝ)

theorem pipe_fill_time (h1 : 0 < t) (h2 : 0 < t / 5) (h3 : (1 / t) + (5 / t) = 1 / 5) : t = 30 :=
by
  sorry

end pipe_fill_time_l33_33698


namespace slices_per_pizza_l33_33185

theorem slices_per_pizza (num_pizzas num_slices : ℕ) (h1 : num_pizzas = 17) (h2 : num_slices = 68) :
  (num_slices / num_pizzas) = 4 :=
by
  sorry

end slices_per_pizza_l33_33185


namespace find_value_of_a_l33_33506

theorem find_value_of_a (a : ℝ) : 
  (let y := λ x : ℝ, a * x in  y 0 + y 1 = 3) ↔ a = 3 := 
by
  sorry

end find_value_of_a_l33_33506


namespace heavy_tailed_permutations_count_l33_33049

open Finset

-- Define the set of numbers and the condition defining heavy-tailed permutations
def is_heavy_tailed (perm : List ℕ) : Prop :=
  perm.length = 5 ∧
  {1, 2, 3, 5, 6}.subset perm.to_finset ∧
  (perm.headI + perm.tail.headI < perm.tail.tail.tailI.headI + perm.tail.tail.tailI.tail.headI)

-- Count the number of heavy-tailed permutations
def count_heavy_tailed_permutations : ℕ :=
  (univ : Finset (List ℕ)).filter is_heavy_tailed).card

-- Statement to prove in Lean 4
theorem heavy_tailed_permutations_count :
  count_heavy_tailed_permutations = 40 :=
by
  sorry

end heavy_tailed_permutations_count_l33_33049


namespace choose_18_4_eq_3060_l33_33914

/-- The number of ways to select 4 members from a group of 18 people (without regard to order). -/
theorem choose_18_4_eq_3060 : Nat.choose 18 4 = 3060 := 
by
  sorry

end choose_18_4_eq_3060_l33_33914


namespace cloak_change_14_gold_coins_l33_33445

def exchange_rate (silver gold : ℕ) : Prop :=
  ∃ c : ℕ, (20 - 4) * c = silver ∧ (15 - 1) * c = silver

def cloak_purchase (paid_gold received_silver : ℕ) : Prop :=
  let exchange_rate := (5 * 14) / 3 in
  received_silver = 2 * exchange_rate

theorem cloak_change_14_gold_coins :
  exchange_rate 16 3 →
  exchange_rate 14 1 →
  cloak_purchase 14 10 := sorry

end cloak_change_14_gold_coins_l33_33445


namespace max_value_M_l33_33071

theorem max_value_M : 
  ∃ t : ℝ, (t = (3 / (4 ^ (1 / 3)))) ∧ 
    (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → 
      a^3 + b^3 + c^3 - 3 * a * b * c ≥ t * (a * b^2 + b * c^2 + c * a^2 - 3 * a * b * c)) :=
sorry

end max_value_M_l33_33071


namespace find_m_l33_33692

def U : Set ℕ := {1, 2, 3, 4}
def compl_U_A : Set ℕ := {1, 4}

theorem find_m (m : ℕ) (A : Set ℕ) (hA : A = {x | x ^ 2 - 5 * x + m = 0 ∧ x ∈ U}) :
  compl_U_A = U \ A → m = 6 :=
by
  sorry

end find_m_l33_33692


namespace entrance_fee_increase_l33_33718

theorem entrance_fee_increase
  (entrance_fee_under_18 : ℕ)
  (rides_cost : ℕ)
  (num_rides : ℕ)
  (total_spent : ℕ)
  (total_cost_twins : ℕ)
  (total_ride_cost_twins : ℕ)
  (amount_spent_joe : ℕ)
  (total_ride_cost_joe : ℕ)
  (joe_entrance_fee : ℕ)
  (increase : ℕ)
  (percentage_increase : ℕ)
  (h1 : entrance_fee_under_18 = 5)
  (h2 : rides_cost = 50) -- representing $0.50 as 50 cents to maintain integer calculations
  (h3 : num_rides = 3)
  (h4 : total_spent = 2050) -- representing $20.5 as 2050 cents
  (h5 : total_cost_twins = 1300) -- combining entrance fees and cost of rides for the twins in cents
  (h6 : total_ride_cost_twins = 300) -- cost of rides for twins in cents
  (h7 : amount_spent_joe = 750) -- representing $7.5 as 750 cents
  (h8 : total_ride_cost_joe = 150) -- cost of rides for Joe in cents
  (h9 : joe_entrance_fee = 600) -- representing $6 as 600 cents
  (h10 : increase = 100) -- increase in entrance fee in cents
  (h11 : percentage_increase = 20) :
  percentage_increase = ((increase * 100) / entrance_fee_under_18) :=
sorry

end entrance_fee_increase_l33_33718


namespace evaluated_expression_l33_33770

noncomputable def evaluation_problem (x a y z c d : ℝ) : ℝ :=
  (2 * x^3 - 3 * a^4) / (y^2 + 4 * z^5) + c^4 - d^2

theorem evaluated_expression :
  evaluation_problem 0.66 0.1 0.66 0.1 0.066 0.1 = 1.309091916 :=
by
  sorry

end evaluated_expression_l33_33770


namespace cloak_change_14_gold_coins_l33_33448

def exchange_rate (silver gold : ℕ) : Prop :=
  ∃ c : ℕ, (20 - 4) * c = silver ∧ (15 - 1) * c = silver

def cloak_purchase (paid_gold received_silver : ℕ) : Prop :=
  let exchange_rate := (5 * 14) / 3 in
  received_silver = 2 * exchange_rate

theorem cloak_change_14_gold_coins :
  exchange_rate 16 3 →
  exchange_rate 14 1 →
  cloak_purchase 14 10 := sorry

end cloak_change_14_gold_coins_l33_33448


namespace sally_bread_consumption_l33_33705

/-
Given:
    1) Sally eats 2 sandwiches on Saturday
    2) Sally eats 1 sandwich on Sunday
    3) Each sandwich uses 2 pieces of bread

Prove:
    Sally eats 6 pieces of bread across Saturday and Sunday
-/
theorem sally_bread_consumption (sandwiches_sat : Nat) (sandwiches_sun : Nat) (bread_per_sandwich : Nat)
    (H1 : sandwiches_sat = 2) (H2 : sandwiches_sun = 1) (H3 : bread_per_sandwich = 2) :
    2 * bread_per_sandwich + 1 * bread_per_sandwich = 6 := by
  sorry

end sally_bread_consumption_l33_33705


namespace overall_gain_percentage_l33_33754

theorem overall_gain_percentage :
  let SP1 := 100
  let SP2 := 150
  let SP3 := 200
  let CP1 := SP1 / (1 + 0.20)
  let CP2 := SP2 / (1 + 0.15)
  let CP3 := SP3 / (1 - 0.05)
  let TCP := CP1 + CP2 + CP3
  let TSP := SP1 + SP2 + SP3
  let G := TSP - TCP
  let GP := (G / TCP) * 100
  GP = 6.06 := 
by {
  sorry
}

end overall_gain_percentage_l33_33754


namespace ferns_have_1260_leaves_l33_33288

def num_ferns : ℕ := 6
def fronds_per_fern : ℕ := 7
def leaves_per_frond : ℕ := 30
def total_leaves : ℕ := num_ferns * fronds_per_fern * leaves_per_frond

theorem ferns_have_1260_leaves : total_leaves = 1260 :=
by 
  -- proof goes here
  sorry

end ferns_have_1260_leaves_l33_33288


namespace num_ways_to_buy_three_items_l33_33119

-- Defining the conditions based on the problem statement
def num_headphones : ℕ := 9
def num_mice : ℕ := 13
def num_keyboards : ℕ := 5
def num_kb_mouse_sets : ℕ := 4
def num_hp_mouse_sets : ℕ := 5

-- Defining the theorem statement
theorem num_ways_to_buy_three_items :
  (num_kb_mouse_sets * num_headphones) + 
  (num_hp_mouse_sets * num_keyboards) + 
  (num_headphones * num_mice * num_keyboards) = 646 := 
  by
  sorry

end num_ways_to_buy_three_items_l33_33119


namespace total_trees_after_planting_l33_33998

def current_trees : ℕ := 7
def trees_planted_today : ℕ := 5
def trees_planted_tomorrow : ℕ := 4

theorem total_trees_after_planting : 
  current_trees + trees_planted_today + trees_planted_tomorrow = 16 :=
by
  sorry

end total_trees_after_planting_l33_33998


namespace solve_equation_l33_33848

def equation_holds (x : ℝ) : Prop := 
  (1 / (x + 10)) + (1 / (x + 8)) = (1 / (x + 11)) + (1 / (x + 7))

theorem solve_equation : equation_holds (-9) :=
by
  sorry

end solve_equation_l33_33848


namespace total_spent_on_clothing_l33_33683

def shorts_cost : ℝ := 15
def jacket_cost : ℝ := 14.82
def shirt_cost : ℝ := 12.51

theorem total_spent_on_clothing : shorts_cost + jacket_cost + shirt_cost = 42.33 := by
  -- Proof goes here.
  sorry

end total_spent_on_clothing_l33_33683


namespace find_angle_4_l33_33251

/-- Given angle conditions, prove that angle 4 is 22.5 degrees. -/
theorem find_angle_4 (angle : ℕ → ℝ) 
  (h1 : angle 1 + angle 2 = 180) 
  (h2 : angle 3 = angle 4) 
  (h3 : angle 1 = 85) 
  (h4 : angle 5 = 45) 
  (h5 : angle 1 + angle 5 + angle 6 = 180) : 
  angle 4 = 22.5 :=
sorry

end find_angle_4_l33_33251


namespace value_of_fraction_pow_l33_33406

theorem value_of_fraction_pow (a b : ℤ) 
  (h1 : ∀ x, (x^2 + (a + 1)*x + a*b) ≤ 0 ↔ -1 ≤ x ∧ x ≤ 4) : 
  ((1 / 2 : ℚ) ^ (a + 2*b) = 4) :=
sorry

end value_of_fraction_pow_l33_33406


namespace seven_people_different_rolls_l33_33845

def rolls_different (rolls : Fin 7 -> Fin 6) : Prop :=
  ∀ i : Fin 7, rolls i ≠ rolls ⟨(i + 1) % 7, sorry⟩

def probability_rolls_different : ℚ :=
  (625 : ℚ) / 2799

theorem seven_people_different_rolls (rolls : Fin 7 -> Fin 6) :
  (∃ rolls, rolls_different rolls) ->
  probability_rolls_different = 625 / 2799 :=
sorry

end seven_people_different_rolls_l33_33845


namespace smallest_integer_value_of_x_satisfying_eq_l33_33236

theorem smallest_integer_value_of_x_satisfying_eq (x : ℤ) (h : |x^2 - 5*x + 6| = 14) : 
  ∃ y : ℤ, (y = -1) ∧ ∀ z : ℤ, (|z^2 - 5*z + 6| = 14) → (y ≤ z) :=
sorry

end smallest_integer_value_of_x_satisfying_eq_l33_33236


namespace decimal_fraction_error_l33_33161

theorem decimal_fraction_error (A B C D E : ℕ) (hA : A < 100) 
    (h10B : 10 * B = A + C) (h10C : 10 * C = 6 * A + D) (h10D : 10 * D = 7 * A + E) 
    (hBCDE_lt_A : B < A ∧ C < A ∧ D < A ∧ E < A) : 
    false :=
sorry

end decimal_fraction_error_l33_33161


namespace tax_rate_correct_l33_33901

noncomputable def tax_rate (total_payroll : ℕ) (tax_free_payroll : ℕ) (tax_paid : ℕ) : ℚ :=
  if total_payroll > tax_free_payroll 
  then (tax_paid : ℚ) / (total_payroll - tax_free_payroll) * 100
  else 0

theorem tax_rate_correct :
  tax_rate 400000 200000 400 = 0.2 :=
by
  sorry

end tax_rate_correct_l33_33901


namespace racing_magic_circle_time_l33_33005

theorem racing_magic_circle_time
  (T : ℕ) -- Time taken by the racing magic to circle the track once
  (bull_rounds_per_hour : ℕ := 40) -- Rounds the Charging Bull makes in an hour
  (meet_time_minutes : ℕ := 6) -- Time in minutes to meet at starting point
  (charging_bull_seconds_per_round : ℕ := 3600 / bull_rounds_per_hour) -- Time in seconds per Charging Bull round
  (meet_time_seconds : ℕ := meet_time_minutes * 60) -- Time in seconds to meet at starting point
  (rounds_by_bull : ℕ := meet_time_seconds / charging_bull_seconds_per_round) -- Rounds completed by the Charging Bull to meet again
  (rounds_by_magic : ℕ := meet_time_seconds / T) -- Rounds completed by the Racing Magic to meet again
  (h1 : rounds_by_magic = 1) -- Racing Magic completes 1 round in the meet time
  : T = 360 := -- Racing Magic takes 360 seconds to circle the track once
  sorry

end racing_magic_circle_time_l33_33005


namespace angle_bisector_divides_longest_side_l33_33398

theorem angle_bisector_divides_longest_side :
  ∀ (a b c : ℕ) (p q : ℕ), a = 12 → b = 15 → c = 18 →
  p + q = c → p * b = q * a → p = 8 ∧ q = 10 :=
by
  intros a b c p q ha hb hc hpq hprop
  rw [ha, hb, hc] at *
  sorry

end angle_bisector_divides_longest_side_l33_33398


namespace probability_of_X_le_1_l33_33247

noncomputable def C (n k : ℕ) : ℚ := Nat.choose n k

noncomputable def P_X_le_1 := 
  (C 4 3 / C 6 3) + (C 4 2 * C 2 1 / C 6 3)

theorem probability_of_X_le_1 : P_X_le_1 = 4 / 5 := by
  sorry

end probability_of_X_le_1_l33_33247


namespace lisa_photos_last_weekend_l33_33298

def photos_of_animals : ℕ := 10
def photos_of_flowers : ℕ := 3 * photos_of_animals
def photos_of_scenery : ℕ := photos_of_flowers - 10
def total_photos_this_week : ℕ := photos_of_animals + photos_of_flowers + photos_of_scenery
def photos_last_weekend : ℕ := total_photos_this_week - 15

theorem lisa_photos_last_weekend : photos_last_weekend = 45 :=
by
  sorry

end lisa_photos_last_weekend_l33_33298


namespace points_on_parabola_l33_33792

theorem points_on_parabola (s : ℝ) : ∃ (u v : ℝ), u = 3^s - 4 ∧ v = 9^s - 7 * 3^s - 2 ∧ v = u^2 + u - 14 :=
by
  sorry

end points_on_parabola_l33_33792


namespace find_right_triangles_with_leg_2012_l33_33462

theorem find_right_triangles_with_leg_2012 :
  ∃ x y z : ℕ, x^2 + 2012^2 = y^2 ∧ x + y + z = 2012 + 253005 + 253013 ∨
  x^2 + 2012^2 = z^2 ∧ x + y + z = 2012 + 506016 + 506020 ∨
  x^2 + 2012^2 = y^2 ∧ x + y + z = 2012 + 1012035 + 1012037 ∨
  x^2 + 2012^2 = y^2 ∧ x + y + z = 2012 + 1509 + 2515 ∧
  y ≠ 2012 :=
begin
  sorry  -- The actual proof is omitted as specified.
end

end find_right_triangles_with_leg_2012_l33_33462


namespace hyperbola_solution_l33_33335

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

noncomputable def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 = 1

theorem hyperbola_solution : 
  (∀ x y, ellipse_eq x y → (x = 2 ∧ y = 1) → hyperbola_eq x y) :=
by
  intros x y h_ellipse h_point
  sorry

end hyperbola_solution_l33_33335


namespace factor_expression_l33_33925

theorem factor_expression (x y : ℝ) :
  66 * x^5 - 165 * x^9 + 99 * x^5 * y = 33 * x^5 * (2 - 5 * x^4 + 3 * y) :=
by
  sorry

end factor_expression_l33_33925


namespace constant_sum_of_distances_l33_33956

open Real

theorem constant_sum_of_distances (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
    (ellipse_condition : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → ∀ A B : ℝ × ℝ, A.2 > 0 ∧ B.2 > 0)
    (foci : (ℝ × ℝ) × (ℝ × ℝ) := ((-c, 0), (c, 0)))
    (points_AB : ∃ (A B : ℝ × ℝ), A.2 > 0 ∧ B.2 > 0 ∧ (A.1 - c)^2 / a^2 + A.2^2 / b^2 = 1 ∧ (B.1 - -c)^2 / a^2 + B.2^2 / b^2 = 1)
    (AF1_parallel_BF2 : ∀ (A B : ℝ × ℝ), (A.1 - -c) * (B.2 - 0) - (A.2 - 0) * (B.1 - c) = 0)
    (intersection_P: ∀ (A B : ℝ × ℝ), ∃ P : ℝ × ℝ, ((A.1 - c) * (B.2 - 0) = (A.2 - 0) * (P.1 - c)) ∧ ((B.1 - -c) * (A.2 - 0) = (B.2 - 0) * (P.1 - -c))) :
    ∃ k : ℝ, ∀ (P : ℝ × ℝ), dist P (foci.fst) + dist P (foci.snd) = k := 
sorry

end constant_sum_of_distances_l33_33956


namespace OH_squared_correct_l33_33475

noncomputable def OH_squared (O H : Point) (a b c R : ℝ) : ℝ :=
  9 * R^2 - (a^2 + b^2 + c^2)

theorem OH_squared_correct :
  ∀ (O H : Point) (a b c : ℝ) (R : ℝ),
    R = 7 →
    a^2 + b^2 + c^2 = 29 →
    OH_squared O H a b c R = 412 := by
  intros O H a b c R hR habc
  simp [OH_squared, hR, habc]
  sorry

end OH_squared_correct_l33_33475


namespace pythagorean_triple_solution_l33_33166

theorem pythagorean_triple_solution
  (x y z a b : ℕ)
  (h1 : x^2 + y^2 = z^2)
  (h2 : Nat.gcd x y = 1)
  (h3 : 2 ∣ y)
  (h4 : a > b)
  (h5 : b > 0)
  (h6 : (Nat.gcd a b = 1))
  (h7 : ((a % 2 = 1 ∧ b % 2 = 0) ∨ (a % 2 = 0 ∧ b % 2 = 1))) 
  : (x = a^2 - b^2 ∧ y = 2 * a * b ∧ z = a^2 + b^2) := 
sorry

end pythagorean_triple_solution_l33_33166


namespace Janet_earnings_l33_33286

/--
Janet works as an exterminator and also sells molten metal casts of fire ant nests on the Internet.
She gets paid $70 an hour for exterminator work and makes $20/pound on her ant nest sculptures.
Given that she does 20 hours of exterminator work and sells a 5-pound sculpture and a 7-pound sculpture,
prove that Janet's total earnings are $1640.
-/
theorem Janet_earnings :
  let hourly_rate_exterminator := 70
  let hours_worked := 20
  let rate_per_pound := 20
  let sculpture_one_weight := 5
  let sculpture_two_weight := 7

  let exterminator_earnings := hourly_rate_exterminator * hours_worked
  let total_sculpture_weight := sculpture_one_weight + sculpture_two_weight
  let sculpture_earnings := rate_per_pound * total_sculpture_weight

  let total_earnings := exterminator_earnings + sculpture_earnings
  total_earnings = 1640 := 
by
  sorry

end Janet_earnings_l33_33286


namespace value_of_A_l33_33342

theorem value_of_A 
  (H M A T E: ℤ)
  (H_value: H = 10)
  (MATH_value: M + A + T + H = 35)
  (TEAM_value: T + E + A + M = 42)
  (MEET_value: M + 2*E + T = 38) : 
  A = 21 := 
by 
  sorry

end value_of_A_l33_33342


namespace eval_power_81_11_over_4_l33_33241

theorem eval_power_81_11_over_4 : 81^(11/4) = 177147 := by
  sorry

end eval_power_81_11_over_4_l33_33241


namespace quadratic_roots_l33_33865

noncomputable def roots (p q : ℝ) :=
  if h : p = 1 then ⟨1, h, Classical.arbitrary (ℝ), by simp⟩ else
  if h : p = -2 then ⟨-2, -1, -1, by simp [show p = -2 from h]⟩ else
  false.elim (by simp_all)

theorem quadratic_roots (p q : ℝ) :
  (∃ x1 x2, (x1 + x2 = -p ∧ x1 * x2 = q) ∧ (x1 + 1) + (x2 + 1) = p^2 ∧ (x1 + 1) * (x2 + 1) = pq) →
  (p = 1 ∧ ∃ q, true) ∨ (p = -2 ∧ q = -1) :=
by
  intro h
  sorry

end quadratic_roots_l33_33865


namespace hexagon_perimeter_l33_33720

def side_length : ℕ := 10
def num_sides : ℕ := 6

theorem hexagon_perimeter : num_sides * side_length = 60 := by
  sorry

end hexagon_perimeter_l33_33720


namespace prob_not_equal_genders_l33_33302

noncomputable def probability_more_grandsons_or_granddaughters : ℚ :=
1 - ((Nat.choose 12 6 : ℚ) / (2^12))

theorem prob_not_equal_genders {n : ℕ} (hn : n = 12)
  (equal_prob : ∀ i, i < n → i ≥ 0 → (Nat.of_num 1 / Nat.of_num 2 : ℚ) = 1 / 2) :
  probability_more_grandsons_or_granddaughters = 793 / 1024 := by
  have hn_pos : 0 < n := by linarith
  have hij : _hid := sorry
  sorry

end prob_not_equal_genders_l33_33302


namespace remainder_division_l33_33209

-- Define the polynomial f(x) = x^51 + 51
def f (x : ℤ) : ℤ := x^51 + 51

-- State the theorem to be proven
theorem remainder_division : f (-1) = 50 :=
by
  -- proof goes here
  sorry

end remainder_division_l33_33209


namespace maximum_ratio_l33_33831

-- Defining the conditions
def two_digit_positive_integer (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Proving the main theorem
theorem maximum_ratio (x y : ℕ) (hx : two_digit_positive_integer x) (hy : two_digit_positive_integer y) (h_sum : x + y = 100) : 
  ∃ m, m = 9 ∧ ∀ r, r = x / y → r ≤ 9 := sorry

end maximum_ratio_l33_33831


namespace probability_heads_at_most_3_l33_33575

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l33_33575


namespace exists_h_l33_33167

noncomputable def F (x : ℝ) : ℝ := x^2 + 12 / x^2
noncomputable def G (x : ℝ) : ℝ := Real.sin (Real.pi * x^2)
noncomputable def H (x : ℝ) : ℝ := 1

theorem exists_h (h : ℝ → ℝ) (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 10) :
  |h x - x| < 1 / 3 :=
sorry

end exists_h_l33_33167


namespace regression_analysis_correct_l33_33280

-- Definition of the regression analysis context
def regression_analysis_variation (forecast_var : Type) (explanatory_var residual_var : Type) : Prop :=
  forecast_var = explanatory_var ∧ forecast_var = residual_var

-- The theorem to prove
theorem regression_analysis_correct :
  ∀ (forecast_var explanatory_var residual_var : Type),
  regression_analysis_variation forecast_var explanatory_var residual_var →
  (forecast_var = explanatory_var ∧ forecast_var = residual_var) :=
by
  intro forecast_var explanatory_var residual_var h
  exact h

end regression_analysis_correct_l33_33280


namespace rectangle_area_l33_33928

-- Define the width and length of the rectangle
def w : ℚ := 20 / 3
def l : ℚ := 2 * w

-- Define the perimeter constraint
def perimeter_condition : Prop := 2 * (l + w) = 40

-- Define the area of the rectangle
def area : ℚ := l * w

-- The theorem to prove
theorem rectangle_area : perimeter_condition → area = 800 / 9 :=
by
  intro h
  have hw : w = 20 / 3 := rfl
  have hl : l = 2 * w := rfl
  have hp : 2 * (l + w) = 40 := h
  sorry

end rectangle_area_l33_33928


namespace cos_five_pi_over_six_l33_33072

theorem cos_five_pi_over_six :
  Real.cos (5 * Real.pi / 6) = -(Real.sqrt 3 / 2) :=
sorry

end cos_five_pi_over_six_l33_33072


namespace shop_combinations_l33_33131

theorem shop_combinations :
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  have case1 : keyboard_and_mouse_sets * headphones = 4 * 9, by sorry
  have case2 : headphone_and_mouse_sets * keyboards = 5 * 5, by sorry
  have case3 : headphones * mice * keyboards = 9 * 13 * 5, by sorry
  show (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646,
  by rw [case1, case2, case3]; exact sorry

end shop_combinations_l33_33131


namespace retreat_center_probability_each_guest_gets_one_of_each_l33_33899

/-- Suppose there are four guests. Each guest should receive one nut roll, 
one cheese roll, one fruit roll, and one veggie roll. Once wrapped, 
the rolls became indistinguishable and rolls are picked randomly for each guest. 
The probability that each guest receives one roll of each type is \( \frac{1}{5320} \). -/
theorem retreat_center_probability_each_guest_gets_one_of_each :
  let p : ℚ := 1 / 5320 in
  p = (4 / 16) * (4 / 15) * (4 / 14) * (4 / 13) * (3 / 12) * (3 / 11) * (3 / 10) * (3 / 9) *
      (2 / 8) * (2 / 7) * (2 / 6) * (2 / 5) * 1 :=
sorry

end retreat_center_probability_each_guest_gets_one_of_each_l33_33899


namespace james_painted_area_l33_33960

-- Define the dimensions of the wall and windows
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window1_height : ℕ := 3
def window1_length : ℕ := 5
def window2_height : ℕ := 2
def window2_length : ℕ := 6

-- Calculate the areas
def wall_area : ℕ := wall_height * wall_length
def window1_area : ℕ := window1_height * window1_length
def window2_area : ℕ := window2_height * window2_length
def total_window_area : ℕ := window1_area + window2_area
def painted_area : ℕ := wall_area - total_window_area

theorem james_painted_area : painted_area = 123 :=
by
  -- The proof is omitted
  sorry

end james_painted_area_l33_33960


namespace medal_winners_combinations_l33_33274

theorem medal_winners_combinations:
  ∀ n k : ℕ, (n = 6) → (k = 3) → (n.choose k = 20) :=
by
  intros n k hn hk
  simp [hn, hk]
  -- We can continue the proof using additional math concepts if necessary.
  sorry

end medal_winners_combinations_l33_33274


namespace greendale_high_school_points_l33_33310

-- Define the conditions
def roosevelt_points_first_game : ℕ := 30
def roosevelt_points_second_game : ℕ := roosevelt_points_first_game / 2
def roosevelt_points_third_game : ℕ := roosevelt_points_second_game * 3
def roosevelt_total_points : ℕ := roosevelt_points_first_game + roosevelt_points_second_game + roosevelt_points_third_game
def roosevelt_total_with_bonus : ℕ := roosevelt_total_points + 50
def difference : ℕ := 10

-- Define the assertion for Greendale's points
def greendale_points : ℕ := roosevelt_total_with_bonus - difference

-- The theorem to be proved
theorem greendale_high_school_points : greendale_points = 130 :=
by
  -- Proof should be here, but we add sorry to skip it as per the instructions
  sorry

end greendale_high_school_points_l33_33310


namespace numeral_of_place_face_value_difference_l33_33165

theorem numeral_of_place_face_value_difference (P F : ℕ) (H : P - F = 63) (Hface : F = 7) : P = 70 :=
sorry

end numeral_of_place_face_value_difference_l33_33165


namespace quadratic_roots_transform_l33_33293

theorem quadratic_roots_transform {p q : ℝ} (h1 : 3 * p^2 + 5 * p - 7 = 0) (h2 : 3 * q^2 + 5 * q - 7 = 0) : (p - 2) * (q - 2) = 5 := 
by 
  sorry

end quadratic_roots_transform_l33_33293


namespace eleanor_distance_between_meetings_l33_33240

-- Conditions given in the problem
def track_length : ℕ := 720
def eric_time : ℕ := 4
def eleanor_time : ℕ := 5
def eric_speed : ℕ := track_length / eric_time
def eleanor_speed : ℕ := track_length / eleanor_time
def relative_speed : ℕ := eric_speed + eleanor_speed
def time_to_meet : ℚ := track_length / relative_speed

-- Proof task: prove that the distance Eleanor runs between consective meetings is 320 meters.
theorem eleanor_distance_between_meetings : eleanor_speed * time_to_meet = 320 := by
  sorry

end eleanor_distance_between_meetings_l33_33240


namespace valid_colorings_l33_33060

-- Define the coloring function and the condition
variable (f : ℕ → ℕ) -- f assigns a color (0, 1, or 2) to each natural number
variable (a b c : ℕ)
-- Colors are represented by 0, 1, or 2
variable (colors : Fin 3)

-- Define the condition to be checked
def valid_coloring : Prop :=
  ∀ a b c, 2000 * (a + b) = c → (f a = f b ∧ f b = f c) ∨ (f a ≠ f b ∧ f b ≠ f c ∧ f c ≠ f a)

-- Now define the two possible valid ways of coloring
def all_same_color : Prop :=
  ∃ color, ∀ n, f n = color

def every_third_different : Prop :=
  (∀ k : ℕ, f (3 * k) = 0 ∧ f (3 * k + 1) = 1 ∧ f (3 * k + 2) = 2)

-- Prove that these are the only two valid ways
theorem valid_colorings :
  valid_coloring f →
  all_same_color f ∨ every_third_different f :=
sorry

end valid_colorings_l33_33060


namespace probability_at_most_3_heads_l33_33578

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l33_33578


namespace simplify_expression_l33_33322

theorem simplify_expression :
  (∃ (a b c d : ℝ), 
   a = 14 * Real.sqrt 2 ∧ 
   b = 12 * Real.sqrt 2 ∧ 
   c = 8 * Real.sqrt 2 ∧ 
   d = 12 * Real.sqrt 2 ∧ 
   ((a / b) + (c / d) = 11 / 6)) :=
by 
  use 14 * Real.sqrt 2, 12 * Real.sqrt 2, 8 * Real.sqrt 2, 12 * Real.sqrt 2
  simp
  sorry

end simplify_expression_l33_33322


namespace area_of_smallest_square_l33_33023

-- Define a circle with a given radius
def radius : ℝ := 7

-- Define the diameter as twice the radius
def diameter : ℝ := 2 * radius

-- Define the side length of the smallest square that can contain the circle
def side_length : ℝ := diameter

-- Define the area of the square as the side length squared
def area_of_square : ℝ := side_length ^ 2

-- State the theorem: the area of the smallest square that contains a circle of radius 7 is 196
theorem area_of_smallest_square : area_of_square = 196 := by
    sorry

end area_of_smallest_square_l33_33023


namespace gcd_8251_6105_l33_33016

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 39 := by
  sorry

end gcd_8251_6105_l33_33016


namespace problem_statement_l33_33952

theorem problem_statement (a b c d : ℤ) (h1 : a - b = -3) (h2 : c + d = 2) : (b + c) - (a - d) = 5 :=
by
  -- Proof steps skipped.
  sorry

end problem_statement_l33_33952


namespace evaluate_f_g_at_3_l33_33148

def f (x : ℝ) : ℝ := x^2 + 2
def g (x : ℝ) : ℝ := 3 * x + 2

theorem evaluate_f_g_at_3 : f (g 3) = 123 := by
  sorry

end evaluate_f_g_at_3_l33_33148


namespace matrix_addition_is_correct_l33_33389

-- Definitions of matrices A and B according to given conditions
def A : Matrix (Fin 4) (Fin 4) ℤ :=  
  ![![ 3,  0,  1,  4],
    ![ 1,  2,  0,  0],
    ![ 5, -3,  2,  1],
    ![ 0,  0, -1,  3]]

def B : Matrix (Fin 4) (Fin 4) ℤ :=
  ![![-5, -7,  3,  2],
    ![ 4, -9,  5, -2],
    ![ 8,  2, -3,  0],
    ![ 1,  1, -2, -4]]

-- The expected result matrix from the addition of A and B
def C : Matrix (Fin 4) (Fin 4) ℤ :=
  ![![-2, -7,  4,  6],
    ![ 5, -7,  5, -2],
    ![13, -1, -1,  1],
    ![ 1,  1, -3, -1]]

-- The statement that A + B equals C
theorem matrix_addition_is_correct : A + B = C :=
by 
  -- Here we would provide the proof steps.
  sorry

end matrix_addition_is_correct_l33_33389


namespace greatest_k_for_200k_divides_100_factorial_l33_33746

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_k_for_200k_divides_100_factorial :
  let x := factorial 100
  let k_max := 12
  ∃ k : ℕ, y = 200 ^ k ∧ y ∣ x ∧ k = k_max :=
sorry

end greatest_k_for_200k_divides_100_factorial_l33_33746


namespace original_price_of_sarees_l33_33727

theorem original_price_of_sarees (P : ℝ) (h1 : 0.95 * 0.80 * P = 133) : P = 175 :=
sorry

end original_price_of_sarees_l33_33727


namespace pseudometric_d_pseudometric_rho_d_zero_not_imply_eq_rho_zero_not_imply_eq_l33_33158

-- Define the distance measures
def d (A B : Set α) [MeasurableSpace α] (P : Measure α) := P (A.symmDiff B)

def rho (A B : Set α) [MeasurableSpace α] (P : Measure α) := 
  if P (A ∪ B) ≠ 0 then P (A.symmDiff B) / P (A ∪ B) else 0

-- Define the pseudometric properties
theorem pseudometric_d (A B C : Set α) [MeasurableSpace α] (P : Measure α) :
  d A B P ≤ d A C P + d B C P :=
sorry

theorem pseudometric_rho (A B C : Set α) [MeasurableSpace α] (P : Measure α) 
  (hAB : P (A ∪ B) ≠ 0) (hAC : P (A ∪ C) ≠ 0) (hBC : P (B ∪ C) ≠ 0) :
  rho A B P ≤ rho A C P + rho B C P :=
sorry

-- Show that d(A, B) = 0 does not imply A = B
theorem d_zero_not_imply_eq (A B : Set α) [MeasurableSpace α] (P : Measure α) :
  d A B P = 0 → ¬ (A ≠ B) :=
sorry

-- Show that rho(A, B) = 0 does not imply A = B
theorem rho_zero_not_imply_eq (A B : Set α) [MeasurableSpace α] (P : Measure α) (hAB : P (A ∪ B) ≠ 0) :
  rho A B P = 0 → ¬ (A ≠ B) :=
sorry

end pseudometric_d_pseudometric_rho_d_zero_not_imply_eq_rho_zero_not_imply_eq_l33_33158


namespace competition_results_l33_33367

variables (x : ℝ) (freq1 freq3 freq4 freq5 freq2 : ℝ)

/-- Axiom: Given frequencies of groups and total frequency, determine the total number of participants and the probability of an excellent score -/
theorem competition_results :
  freq1 = 0.30 ∧
  freq3 = 0.15 ∧
  freq4 = 0.10 ∧
  freq5 = 0.05 ∧
  freq2 = 40 / x ∧
  (freq1 + freq2 + freq3 + freq4 + freq5 = 1) ∧
  (x * freq2 = 40) →
  x = 100 ∧ (freq4 + freq5 = 0.15) := sorry

end competition_results_l33_33367


namespace jill_spent_on_clothing_l33_33837

-- Define the total amount spent excluding taxes, T.
variable (T : ℝ)
-- Define the percentage of T Jill spent on clothing, C.
variable (C : ℝ)

-- Define the conditions based on the problem statement.
def jill_tax_conditions : Prop :=
  let food_percent := 0.20
  let other_items_percent := 0.30
  let clothing_tax := 0.04
  let food_tax := 0
  let other_tax := 0.10
  let total_tax := 0.05
  let food_amount := food_percent * T
  let other_items_amount := other_items_percent * T
  let clothing_amount := C * T
  let clothing_tax_amount := clothing_tax * clothing_amount
  let other_tax_amount := other_tax * other_items_amount
  let total_tax_amount := clothing_tax_amount + food_tax * food_amount + other_tax_amount
  C * T + food_percent * T + other_items_percent * T = T ∧
  total_tax_amount / T = total_tax

-- The goal is to prove that C = 0.50.
theorem jill_spent_on_clothing (h : jill_tax_conditions T C) : C = 0.50 :=
by
  sorry

end jill_spent_on_clothing_l33_33837


namespace probability_at_most_3_heads_l33_33581

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l33_33581


namespace initial_distance_between_A_and_B_l33_33038

theorem initial_distance_between_A_and_B
  (start_time : ℕ)        -- time in hours, 1 pm
  (meet_time : ℕ)         -- time in hours, 3 pm
  (speed_A : ℕ)           -- speed of A in km/hr
  (speed_B : ℕ)           -- speed of B in km/hr
  (time_walked : ℕ)       -- time walked in hours
  (distance_A : ℕ)        -- distance covered by A in km
  (distance_B : ℕ)        -- distance covered by B in km
  (initial_distance : ℕ)  -- initial distance between A and B

  (h1 : start_time = 1)
  (h2 : meet_time = 3)
  (h3 : speed_A = 5)
  (h4 : speed_B = 7)
  (h5 : time_walked = meet_time - start_time)
  (h6 : distance_A = speed_A * time_walked)
  (h7 : distance_B = speed_B * time_walked)
  (h8 : initial_distance = distance_A + distance_B) :

  initial_distance = 24 :=
by
  sorry

end initial_distance_between_A_and_B_l33_33038


namespace chess_champion_probability_l33_33156

theorem chess_champion_probability :
  let P_R := 0.6
  let P_S := 0.3
  let P_D := 0.1
  let P := 0.06 + 0.126 + 0.024 + 0.021 + 0.03 + 0.072 + 0.01
  1000 * P = 343 :=
by 
  let P_R := 0.6
  let P_S := 0.3
  let P_D := 0.1
  let P := 0.06 + 0.126 + 0.024 + 0.021 + 0.03 + 0.072 + 0.01
  show 1000 * P = 343
  sorry

end chess_champion_probability_l33_33156


namespace miniature_tank_height_l33_33924

-- Given conditions
def actual_tank_height : ℝ := 50
def actual_tank_volume : ℝ := 200000
def model_tank_volume : ℝ := 0.2

-- Theorem: Calculate the height of the miniature water tank
theorem miniature_tank_height :
  (model_tank_volume / actual_tank_volume) ^ (1/3 : ℝ) * actual_tank_height = 0.5 :=
by
  sorry

end miniature_tank_height_l33_33924


namespace cupboard_cost_price_l33_33036

theorem cupboard_cost_price (C SP NSP : ℝ) (h1 : SP = 0.84 * C) (h2 : NSP = 1.16 * C) (h3 : NSP = SP + 1200) : C = 3750 :=
by
  sorry

end cupboard_cost_price_l33_33036


namespace quadratic_solution_l33_33008

theorem quadratic_solution (x : ℝ) : x ^ 2 - 4 * x + 3 = 0 ↔ x = 1 ∨ x = 3 :=
by
  sorry

end quadratic_solution_l33_33008


namespace tree_planting_l33_33282

/-- The city plans to plant 500 thousand trees. The original plan 
was to plant x thousand trees per day. Due to volunteers, the actual number 
of trees planted per day exceeds the original plan by 30%. As a result, 
the task is completed 2 days ahead of schedule. Prove the equation. -/
theorem tree_planting
    (x : ℝ) 
    (hx : x > 0) : 
    (500 / x) - (500 / ((1 + 0.3) * x)) = 2 :=
sorry

end tree_planting_l33_33282


namespace cos2_alpha_add_sin2_alpha_eq_eight_over_five_l33_33677

theorem cos2_alpha_add_sin2_alpha_eq_eight_over_five (x y : ℝ) (r : ℝ) (α : ℝ) 
(hx : x = 2) 
(hy : y = 1)
(hr : r = Real.sqrt (x^2 + y^2))
(hcos : Real.cos α = x / r)
(hsin : Real.sin α = y / r) :
  Real.cos α ^ 2 + Real.sin (2 * α) = 8 / 5 :=
sorry

end cos2_alpha_add_sin2_alpha_eq_eight_over_five_l33_33677


namespace ellipse_equation_find_slope_l33_33079

theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_vs : b = 1) (h_fl : 2*sqrt(3) = 2*sqrt(a^2 - b^2)) : 
  ∀ x y : ℝ, (x / a)^2 + (y / b)^2 = 1 ↔ x^2 / 4 + y^2 = 1 :=
by sorry

theorem find_slope (k : ℝ) (P : ℝ × ℝ) (hP : P = (-2, 1)) (h_condition : ∃ B C : ℝ × ℝ, B ≠ C ∧ ∀ t : ℝ, (k*t + B.2)*(k*t + C.2) = -k^2*4*t^2 - 8k*t - 4k^2 - 2 - B.1 + 2 + C.1 + 4f - B.1 / (1-f) - C.1 / (1-f) = 2 ) :
  k = -4 :=
by sorry

end ellipse_equation_find_slope_l33_33079


namespace solve_equation_l33_33849

def equation_holds (x : ℝ) : Prop := 
  (1 / (x + 10)) + (1 / (x + 8)) = (1 / (x + 11)) + (1 / (x + 7))

theorem solve_equation : equation_holds (-9) :=
by
  sorry

end solve_equation_l33_33849


namespace probability_genuine_coins_given_weight_condition_l33_33858

/--
Given the following conditions:
- Ten counterfeit coins of equal weight are mixed with 20 genuine coins.
- The weight of a counterfeit coin is different from the weight of a genuine coin.
- Two pairs of coins are selected randomly without replacement from the 30 coins. 

Prove that the probability that all 4 selected coins are genuine, given that the combined weight
of the first pair is equal to the combined weight of the second pair, is 5440/5481.
-/
theorem probability_genuine_coins_given_weight_condition :
  let num_coins := 30
  let num_genuine := 20
  let num_counterfeit := 10
  let pairs_selected := 2
  let pairs_remaining := num_coins - pairs_selected * 2
  let P := (num_genuine / num_coins) * ((num_genuine - 1) / (num_coins - 1)) * ((num_genuine - 2) / pairs_remaining) * ((num_genuine - 3) / (pairs_remaining - 1))
  let event_A_given_B := P / (7 / 16)
  event_A_given_B = 5440 / 5481 := 
sorry

end probability_genuine_coins_given_weight_condition_l33_33858


namespace trajectory_eq_find_m_l33_33656

-- First problem: Trajectory equation
theorem trajectory_eq (P : ℝ × ℝ) (A B : ℝ × ℝ) :
  A = (1, 0) → B = (-1, 0) → 
  (dist P A) * (dist A B) = (dist P B) * (dist A B) → 
  P.snd ^ 2 = 4 * P.fst :=
by sorry

-- Second problem: Value of m
theorem find_m (P : ℝ × ℝ) (M N : ℝ × ℝ) (m : ℝ) :
  P.snd ^ 2 = 4 * P.fst → 
  M.snd = M.fst + m → 
  N.snd = N.fst + m →
  (M.fst - N.fst) * (M.snd - N.snd) + (N.snd - M.snd) * (N.fst - M.fst) = 0 →
  m ≠ 0 →
  m < 1 →
  m = -4 :=
by sorry

end trajectory_eq_find_m_l33_33656


namespace cost_of_45_lilies_l33_33233

-- Definitions of the given conditions
def cost_per_lily := 30 / 18
def lilies_18_bouquet_cost := 30
def number_of_lilies_in_bouquet := 45

-- Theorem stating the mathematical proof problem
theorem cost_of_45_lilies : cost_per_lily * number_of_lilies_in_bouquet = 75 := by
  -- The proof is omitted
  sorry

end cost_of_45_lilies_l33_33233


namespace total_time_last_two_videos_l33_33145

theorem total_time_last_two_videos
  (first_video_length : ℕ := 2 * 60)
  (second_video_length : ℕ := 4 * 60 + 30)
  (total_time : ℕ := 510) :
  ∃ t1 t2 : ℕ, t1 ≠ t2 ∧ t1 + t2 = total_time - first_video_length - second_video_length := by
  sorry

end total_time_last_two_videos_l33_33145


namespace photos_last_weekend_45_l33_33297

theorem photos_last_weekend_45 (photos_animals photos_flowers photos_scenery total_photos_this_weekend photos_last_weekend : ℕ)
  (h1 : photos_animals = 10)
  (h2 : photos_flowers = 3 * photos_animals)
  (h3 : photos_scenery = photos_flowers - 10)
  (h4 : total_photos_this_weekend = photos_animals + photos_flowers + photos_scenery)
  (h5 : photos_last_weekend = total_photos_this_weekend - 15) :
  photos_last_weekend = 45 :=
sorry

end photos_last_weekend_45_l33_33297


namespace integer_coordinates_midpoint_exists_l33_33057

theorem integer_coordinates_midpoint_exists (P : Fin 5 → ℤ × ℤ) :
  ∃ i j : Fin 5, i ≠ j ∧
    ∃ x y : ℤ, (2 * x = (P i).1 + (P j).1) ∧ (2 * y = (P i).2 + (P j).2) := sorry

end integer_coordinates_midpoint_exists_l33_33057


namespace cost_formula_l33_33716

-- Given Conditions
def flat_fee := 5  -- flat service fee in cents
def first_kg_cost := 12  -- cost for the first kilogram in cents
def additional_kg_cost := 5  -- cost for each additional kilogram in cents

-- Integer weight in kilograms
variable (P : ℕ)

-- Total cost calculation proof problem
theorem cost_formula : ∃ C, C = flat_fee + first_kg_cost + additional_kg_cost * (P - 1) → C = 5 * P + 12 :=
by
  sorry

end cost_formula_l33_33716


namespace number_added_l33_33178

theorem number_added (x y : ℝ) (h1 : x = 33) (h2 : x / 4 + y = 15) : y = 6.75 :=
by sorry

end number_added_l33_33178


namespace probability_heads_at_most_three_out_of_ten_coins_l33_33543

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l33_33543


namespace terminating_decimals_count_l33_33782

theorem terminating_decimals_count : 
  let N := 180 in
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0) in
  count_terminating_decimals.length = 60 :=
by
  let N := 180
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0)
  sorry

end terminating_decimals_count_l33_33782


namespace boys_at_reunion_l33_33890

theorem boys_at_reunion (n : ℕ) (h : n * (n - 1) = 56) : n = 8 :=
sorry

end boys_at_reunion_l33_33890


namespace stations_equation_l33_33667

theorem stations_equation (x : ℕ) (h : x * (x - 1) = 1482) : true :=
by
  sorry

end stations_equation_l33_33667


namespace mass_of_circle_is_one_l33_33009

variable (x y z : ℝ)

theorem mass_of_circle_is_one (h1 : 3 * y = 2 * x)
                              (h2 : 2 * y = x + 1)
                              (h3 : 5 * z = x + y)
                              (h4 : true) : z = 1 :=
sorry

end mass_of_circle_is_one_l33_33009


namespace ratio_of_A_to_B_l33_33918

theorem ratio_of_A_to_B (total_weight compound_A_weight compound_B_weight : ℝ)
  (h1 : total_weight = 108)
  (h2 : compound_B_weight = 90)
  (h3 : compound_A_weight = total_weight - compound_B_weight) :
  compound_A_weight / compound_B_weight = 1 / 5 :=
by
  sorry

end ratio_of_A_to_B_l33_33918


namespace symmetric_conic_transform_l33_33245

open Real

theorem symmetric_conic_transform (x y : ℝ) 
  (h1 : 2 * x^2 + 4 * x * y + 5 * y^2 - 22 = 0)
  (h2 : x - y + 1 = 0) : 
  5 * x^2 + 4 * x * y + 2 * y^2 + 6 * x - 19 = 0 := 
sorry

end symmetric_conic_transform_l33_33245


namespace biased_coin_prob_three_heads_l33_33741

def prob_heads := 1/3

theorem biased_coin_prob_three_heads : prob_heads^3 = 1/27 :=
by
  sorry

end biased_coin_prob_three_heads_l33_33741


namespace income_max_takehome_pay_l33_33436

theorem income_max_takehome_pay :
  ∃ x : ℝ, (∀ y : ℝ, 1000 * y - 5 * y^2 ≤ 1000 * x - 5 * x^2) ∧ x = 100 :=
by
  sorry

end income_max_takehome_pay_l33_33436


namespace proof_problem_l33_33803

-- Define the propositions p and q.
def p (a : ℝ) : Prop := a < -1/2 

def q (a b : ℝ) : Prop := a > b → (1 / (a + 1)) < (1 / (b + 1))

-- Define the final proof problem: proving that "p or q" is true.
theorem proof_problem (a b : ℝ) : (p a) ∨ (q a b) := by
  sorry

end proof_problem_l33_33803


namespace unique_solution_a_eq_4_l33_33105

theorem unique_solution_a_eq_4 (a : ℝ) (h : ∀ x1 x2 : ℝ, (a * x1^2 + a * x1 + 1 = 0 ∧ a * x2^2 + a * x2 + 1 = 0) → x1 = x2) : a = 4 :=
sorry

end unique_solution_a_eq_4_l33_33105


namespace solve_equation_l33_33851

theorem solve_equation (x : ℝ) :
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) → x = -9 :=
by 
  sorry

end solve_equation_l33_33851


namespace cube_root_of_sum_powers_l33_33207

noncomputable def cube_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem cube_root_of_sum_powers :
  cube_root (2^7 + 2^7 + 2^7) = 4 * cube_root 2 :=
by
  sorry

end cube_root_of_sum_powers_l33_33207


namespace find_number_l33_33388

theorem find_number :
  (∃ x : ℝ, x * (3 + Real.sqrt 5) = 1) ∧ (x = (3 - Real.sqrt 5) / 4) :=
sorry

end find_number_l33_33388


namespace fencing_cost_proof_l33_33502

theorem fencing_cost_proof (L : ℝ) (B : ℝ) (c : ℝ) (total_cost : ℝ)
  (hL : L = 60) (hL_B : L = B + 20) (hc : c = 26.50) : 
  total_cost = 5300 :=
by
  sorry

end fencing_cost_proof_l33_33502


namespace find_x_given_inverse_relationship_l33_33733

theorem find_x_given_inverse_relationship :
  ∀ (x y: ℝ), (0 < x ∧ 0 < y) ∧ ((x^3 * y = 64) ↔ (x = 2 ∧ y = 8)) ∧ (y = 500) →
  x = 2 / 5 :=
by
  intros x y h
  sorry

end find_x_given_inverse_relationship_l33_33733


namespace quadratic_distinct_positive_roots_l33_33920

theorem quadratic_distinct_positive_roots (a : ℝ) : 
  9 * (a - 2) > 0 → 
  a > 0 → 
  a^2 - 9 * a + 18 > 0 → 
  a ≠ 11 → 
  (2 < a ∧ a < 3) ∨ (6 < a ∧ a < 11) ∨ (11 < a) := 
by 
  intros h1 h2 h3 h4
  sorry

end quadratic_distinct_positive_roots_l33_33920


namespace triangle_BC_value_l33_33820

theorem triangle_BC_value (B C A : ℝ) (AB AC BC : ℝ) 
  (hB : B = 45) 
  (hAB : AB = 100)
  (hAC : AC = 100)
  (h_deg : A ≠ 0) :
  BC = 100 * Real.sqrt 2 := 
by 
  sorry

end triangle_BC_value_l33_33820


namespace probability_of_at_most_3_heads_out_of_10_l33_33583
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l33_33583


namespace integer_solutions_of_equation_l33_33773

theorem integer_solutions_of_equation :
  ∀ x y : ℤ, 2 * x^3 + x * y - 7 = 0 ↔ (x = -7 ∧ y = -99) ∨ (x = -1 ∧ y = -9) ∨ (x = 1 ∧ y = 5) ∨ (x = 7 ∧ y = -97) := by 
  sorry

end integer_solutions_of_equation_l33_33773


namespace Josh_wallet_total_l33_33684

theorem Josh_wallet_total (wallet_money : ℝ) (investment : ℝ) (increase_percentage : ℝ) (increase_amount : ℝ) (final_investment_value : ℝ) :
  wallet_money = 300 →
  investment = 2000 →
  increase_percentage = 0.30 →
  increase_amount = investment * increase_percentage →
  final_investment_value = investment + increase_amount →
  wallet_money + final_investment_value = 2900 :=
begin
  intros h1 h2 h3 h4 h5,
  rw [h1, h2, h3, h4, h5],
  sorry
end

end Josh_wallet_total_l33_33684


namespace outcome_transactions_l33_33301

-- Definition of initial property value and profit/loss percentages.
def property_value : ℝ := 15000
def profit_percentage : ℝ := 0.15
def loss_percentage : ℝ := 0.05

-- Calculate selling price after 15% profit.
def selling_price : ℝ := property_value * (1 + profit_percentage)

-- Calculate buying price after 5% loss based on the above selling price.
def buying_price : ℝ := selling_price * (1 - loss_percentage)

-- Calculate the net gain/loss.
def net_gain_or_loss : ℝ := selling_price - buying_price

-- Statement to be proved.
theorem outcome_transactions : net_gain_or_loss = 862.5 := by
  sorry

end outcome_transactions_l33_33301


namespace order_of_values_l33_33650

noncomputable def a : ℝ := (1 / 5) ^ 2
noncomputable def b : ℝ := 2 ^ (1 / 5)
noncomputable def c : ℝ := Real.log (1 / 5) / Real.log 2  -- change of base from log base 2 to natural log

theorem order_of_values : c < a ∧ a < b :=
by
  sorry

end order_of_values_l33_33650


namespace fifth_term_geometric_sequence_l33_33336

theorem fifth_term_geometric_sequence (x y : ℚ) (h1 : x = 2) (h2 : y = 1) :
    let a1 := x + y
    let a2 := x - y
    let a3 := x / y
    let a4 := x * y
    let r := (x - y)/(x + y)
    (a4 * r = (2 / 3)) :=
by
  -- Proof omitted
  sorry

end fifth_term_geometric_sequence_l33_33336


namespace total_money_l33_33230

theorem total_money (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 330) (h3 : C = 30) : 
  A + B + C = 500 :=
by
  sorry

end total_money_l33_33230


namespace range_of_a_l33_33402

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) + Real.exp (x + 2) - 2 * Real.exp 4
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x * x - 3 * a * Real.exp x
def A : Set ℝ := { x | f x = 0 }
def B (a : ℝ) : Set ℝ := { x | g x a = 0 }

theorem range_of_a (a : ℝ) :
  (∃ x₁ ∈ A, ∃ x₂ ∈ B a, |x₁ - x₂| < 1) →
  a ∈ Set.Ici (1 / (3 * Real.exp 1)) ∩ Set.Iic (4 / (3 * Real.exp 4)) :=
sorry

end range_of_a_l33_33402


namespace has_only_one_zero_point_l33_33091

noncomputable def f (x a : ℝ) := (x - 1) * Real.exp x + (a / 2) * x^2

theorem has_only_one_zero_point (a : ℝ) (h : -Real.exp 1 ≤ a ∧ a ≤ 0) :
  ∃! x : ℝ, f x a = 0 :=
sorry

end has_only_one_zero_point_l33_33091


namespace ellipse_parabola_intersection_l33_33802

theorem ellipse_parabola_intersection (a b k m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = Real.sqrt 2) (h4 : c^2 = a^2 - b^2)
    (h5 : (1 / 2) * 2 * a * 2 * b = 2 * Real.sqrt 3) (h6 : k ≠ 0) :
    (∃ (m: ℝ), (1 / 2) < m ∧ m < 2) :=
sorry

end ellipse_parabola_intersection_l33_33802


namespace trig_expression_value_l33_33421

theorem trig_expression_value {θ : Real} (h : Real.tan θ = 2) :
  (2 * Real.sin θ - Real.cos θ) / (Real.sin θ + 2 * Real.cos θ) = 3 / 4 := 
by
  sorry

end trig_expression_value_l33_33421


namespace cooks_number_l33_33909

variable (C W : ℕ)

theorem cooks_number (h1 : 10 * C = 3 * W) (h2 : 14 * C = 3 * (W + 12)) : C = 9 :=
by
  sorry

end cooks_number_l33_33909


namespace total_feet_l33_33048

theorem total_feet (H C F : ℕ) (h1 : H + C = 48) (h2 : H = 28) :
  F = 2 * H + 4 * C → F = 136 :=
by
  -- substitute H = 28 and perform the calculations
  sorry

end total_feet_l33_33048


namespace scale_length_discrepancy_l33_33226

theorem scale_length_discrepancy
  (scale_length_feet : ℝ)
  (parts : ℕ)
  (part_length_inches : ℝ)
  (ft_to_inch : ℝ := 12)
  (total_length_inches : ℝ := parts * part_length_inches)
  (scale_length_inches : ℝ := scale_length_feet * ft_to_inch) :
  scale_length_feet = 7 → 
  parts = 4 → 
  part_length_inches = 24 →
  total_length_inches - scale_length_inches = 12 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end scale_length_discrepancy_l33_33226


namespace find_other_root_l33_33659

theorem find_other_root (x y : ℚ) (h : 48 * x^2 - 77 * x + 21 = 0) (hx : x = 3 / 4) : y = 7 / 12 → 48 * y^2 - 77 * y + 21 = 0 := by
  sorry

end find_other_root_l33_33659


namespace product_of_all_possible_values_l33_33160

theorem product_of_all_possible_values (x : ℝ) (h : 2 * |x + 3| - 4 = 2) :
  ∃ (a b : ℝ), (x = a ∨ x = b) ∧ a * b = 0 :=
by
  sorry

end product_of_all_possible_values_l33_33160


namespace terminating_decimal_count_number_of_terminating_decimals_l33_33785

theorem terminating_decimal_count {n : ℕ} (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k, n = 9 * k ∧ 1 ≤ 9 * k ∧ 9 * k ≤ 180) ↔ n ∈ (finset.range 21).image (λ k, 9 * k) :=
by sorry

theorem number_of_terminating_decimals :
  (finset.range 21).card = 20 :=
by sorry

end terminating_decimal_count_number_of_terminating_decimals_l33_33785


namespace function_increasing_l33_33266

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) :=
  ∀ (x1 x2 : ℝ), x1 < x2 → f x1 < f x2

theorem function_increasing {f : ℝ → ℝ}
  (H : ∀ (x1 x2 : ℝ), x1 < x2 → f x1 < f x2) :
  is_monotonically_increasing f :=
by
  sorry

end function_increasing_l33_33266


namespace find_initial_alison_stamps_l33_33623

-- Define initial number of stamps Anna, Jeff, and Alison had
def initial_anna_stamps : ℕ := 37
def initial_jeff_stamps : ℕ := 31
def final_anna_stamps : ℕ := 50

-- Define the assumption that Alison gave Anna half of her stamps
def alison_gave_anna_half (a : ℕ) : Prop :=
  initial_anna_stamps + a / 2 = final_anna_stamps

-- Define the problem of finding the initial number of stamps Alison had
def alison_initial_stamps : ℕ := 26

theorem find_initial_alison_stamps :
  ∃ a : ℕ, alison_gave_anna_half a ∧ a = alison_initial_stamps :=
by
  sorry

end find_initial_alison_stamps_l33_33623


namespace find_x_l33_33050

noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3
noncomputable def log_base_5 (x : ℝ) : ℝ := Real.log x / Real.log 5
noncomputable def log_base_4 (x : ℝ) : ℝ := Real.log x / Real.log 4

noncomputable def right_triangular_prism_area (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_3 x
  let c := log_base_5 x
  (1/2) * a * b + 3 * c * (a + b + c)

noncomputable def right_triangular_prism_volume (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_3 x
  let c := log_base_5 x
  (1/2) * a * b * c

noncomputable def rectangular_prism_area (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_4 x
  2 * (a * b + a * a + b * a)

noncomputable def rectangular_prism_volume (x : ℝ) : ℝ := 
  let a := log_base_2 x
  let b := log_base_4 x
  a * b * a

theorem find_x (x : ℝ) (h : right_triangular_prism_area x + rectangular_prism_area x = right_triangular_prism_volume x + rectangular_prism_volume x) :
  x = 1152 := by
sorry

end find_x_l33_33050


namespace probability_heads_at_most_three_out_of_ten_coins_l33_33544

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l33_33544


namespace probability_at_most_three_heads_10_coins_l33_33597

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l33_33597


namespace can_buy_two_pies_different_in_both_l33_33272

structure Pie :=
  (filling : Type)
  (preparation : Type)

def apple : Type := unit
def cherry : Type := unit
def fried : Type := unit
def baked : Type := unit

def apple_fried : Pie := { filling := apple, preparation := fried }
def apple_baked : Pie := { filling := apple, preparation := baked }
def cherry_fried : Pie := { filling := cherry, preparation := fried }
def cherry_baked : Pie := { filling := cherry, preparation := baked }

def possible_pies : List Pie := [apple_fried, apple_baked, cherry_fried, cherry_baked]

theorem can_buy_two_pies_different_in_both 
  (available_pies : List Pie) 
  (h : available_pies.length ≥ 3) : 
  ∃ (p1 p2 : Pie), p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation := 
  sorry

end can_buy_two_pies_different_in_both_l33_33272


namespace cloak_change_14_gold_coins_l33_33446

def exchange_rate (silver gold : ℕ) : Prop :=
  ∃ c : ℕ, (20 - 4) * c = silver ∧ (15 - 1) * c = silver

def cloak_purchase (paid_gold received_silver : ℕ) : Prop :=
  let exchange_rate := (5 * 14) / 3 in
  received_silver = 2 * exchange_rate

theorem cloak_change_14_gold_coins :
  exchange_rate 16 3 →
  exchange_rate 14 1 →
  cloak_purchase 14 10 := sorry

end cloak_change_14_gold_coins_l33_33446


namespace a_7_eq_64_l33_33654

-- Define the problem conditions using variables in Lean
variable {a : ℕ → ℝ} -- defining the sequence as a function from natural numbers to reals
variable {q : ℝ}  -- common ratio

-- The sequence is geometric
axiom geom_seq (n : ℕ) : a (n + 1) = a n * q

-- Conditions given in the problem
axiom condition1 : a 1 + a 2 = 3
axiom condition2 : a 2 + a 3 = 6

-- Target statement to prove
theorem a_7_eq_64 : a 7 = 64 := 
sorry

end a_7_eq_64_l33_33654


namespace smallest_total_hot_dogs_l33_33031

def packs_hot_dogs := 12
def packs_buns := 9
def packs_mustard := 18
def packs_ketchup := 24

theorem smallest_total_hot_dogs : Nat.lcm (Nat.lcm (Nat.lcm packs_hot_dogs packs_buns) packs_mustard) packs_ketchup = 72 := by
  sorry

end smallest_total_hot_dogs_l33_33031


namespace gcd_8251_6105_l33_33020

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l33_33020


namespace least_positive_integer_l33_33187

theorem least_positive_integer (n : ℕ) : 
  (530 + n) % 4 = 0 → n = 2 :=
by {
  sorry
}

end least_positive_integer_l33_33187


namespace distance_traveled_by_center_of_ball_l33_33747

noncomputable def ball_diameter : ℝ := 6
noncomputable def ball_radius : ℝ := ball_diameter / 2
noncomputable def R1 : ℝ := 100
noncomputable def R2 : ℝ := 60
noncomputable def R3 : ℝ := 80
noncomputable def R4 : ℝ := 40

noncomputable def effective_radius_inner (R : ℝ) (r : ℝ) : ℝ := R - r
noncomputable def effective_radius_outer (R : ℝ) (r : ℝ) : ℝ := R + r

noncomputable def dist_travel_on_arc (R : ℝ) : ℝ := R * Real.pi

theorem distance_traveled_by_center_of_ball :
  dist_travel_on_arc (effective_radius_inner R1 ball_radius) +
  dist_travel_on_arc (effective_radius_outer R2 ball_radius) +
  dist_travel_on_arc (effective_radius_inner R3 ball_radius) +
  dist_travel_on_arc (effective_radius_outer R4 ball_radius) = 280 * Real.pi :=
by 
  -- Calculation steps can be filled in here but let's skip
  sorry

end distance_traveled_by_center_of_ball_l33_33747


namespace simplify_expression_l33_33159

theorem simplify_expression :
  ((5 * 10^7) / (2 * 10^2)) + (4 * 10^5) = 650000 := 
by
  sorry

end simplify_expression_l33_33159


namespace converse_false_l33_33333

variable {a b : ℝ}

theorem converse_false : (¬ (∀ a b : ℝ, (ab = 0 → a = 0))) :=
by
  sorry

end converse_false_l33_33333


namespace probability_of_at_most_3_heads_out_of_10_l33_33584
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l33_33584


namespace prob_heads_at_most_3_out_of_10_flips_l33_33555

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l33_33555


namespace compare_2_5_sqrt_6_l33_33056

theorem compare_2_5_sqrt_6 : 2.5 > Real.sqrt 6 := by
  sorry

end compare_2_5_sqrt_6_l33_33056


namespace coin_flip_probability_difference_l33_33193

theorem coin_flip_probability_difference :
  let p1 := (nat.choose 5 4) * (1 / 2)^5
  let p2 := (1 / 2)^5
  (p1 - p2 = 1 / 8) :=
by
  let p1 := (nat.choose 5 4) * (1 / 2)^5
  let p2 := (1 / 2)^5
  sorry

end coin_flip_probability_difference_l33_33193


namespace delivery_newspapers_15_houses_l33_33756

-- State the problem using Lean 4 syntax

noncomputable def delivery_sequences (n : ℕ) : ℕ :=
  if h : n < 3 then 2^n
  else if n = 3 then 6
  else delivery_sequences (n-1) + delivery_sequences (n-2) + delivery_sequences (n-3)

theorem delivery_newspapers_15_houses :
  delivery_sequences 15 = 849 :=
sorry

end delivery_newspapers_15_houses_l33_33756


namespace only_nice_number_is_three_l33_33290

def P (x : ℕ) : ℕ := x + 1
def Q (x : ℕ) : ℕ := x^2 + 1

def nice (n : ℕ) : Prop :=
  ∃ (xs ys : ℕ → ℕ), 
    xs 1 = 1 ∧ ys 1 = 3 ∧
    (∀ k, xs (k+1) = P (xs k) ∧ ys (k+1) = Q (ys k) ∨ xs (k+1) = Q (xs k) ∧ ys (k+1) = P (ys k)) ∧
    xs n = ys n

theorem only_nice_number_is_three (n : ℕ) : nice n ↔ n = 3 :=
by
  sorry

end only_nice_number_is_three_l33_33290


namespace parabola_vertex_l33_33715

-- Definition of the quadratic function representing the parabola
def parabola (x : ℝ) : ℝ := (3 * x - 1) ^ 2 + 2

-- Statement asserting the coordinates of the vertex of the given parabola
theorem parabola_vertex :
  ∃ h k : ℝ, ∀ x : ℝ, parabola x = 9 * (x - h) ^ 2 + k ∧ h = 1/3 ∧ k = 2 :=
by
  sorry

end parabola_vertex_l33_33715


namespace cloak_change_14_gold_coins_l33_33447

def exchange_rate (silver gold : ℕ) : Prop :=
  ∃ c : ℕ, (20 - 4) * c = silver ∧ (15 - 1) * c = silver

def cloak_purchase (paid_gold received_silver : ℕ) : Prop :=
  let exchange_rate := (5 * 14) / 3 in
  received_silver = 2 * exchange_rate

theorem cloak_change_14_gold_coins :
  exchange_rate 16 3 →
  exchange_rate 14 1 →
  cloak_purchase 14 10 := sorry

end cloak_change_14_gold_coins_l33_33447


namespace length_of_each_section_25_l33_33470

theorem length_of_each_section_25 (x : ℝ) 
  (h1 : ∃ x, x > 0)
  (h2 : 1000 / x = 15 / (1 / 2 * 3 / 4))
  : x = 25 := 
  sorry

end length_of_each_section_25_l33_33470


namespace x_cubed_inverse_cubed_l33_33661

theorem x_cubed_inverse_cubed (x : ℝ) (hx : x + 1/x = 3) : x^3 + 1/x^3 = 18 :=
by
  sorry

end x_cubed_inverse_cubed_l33_33661


namespace pies_differ_in_both_l33_33273

-- Defining types of pies
inductive Filling where
  | apple : Filling
  | cherry : Filling

inductive Preparation where
  | fried : Preparation
  | baked : Preparation

structure Pie where
  filling : Filling
  preparation : Preparation

-- The set of all possible pies
def allPies : Set Pie :=
  { ⟨Filling.apple, Preparation.fried⟩,
    ⟨Filling.apple, Preparation.baked⟩,
    ⟨Filling.cherry, Preparation.fried⟩,
    ⟨Filling.cherry, Preparation.baked⟩ }

-- Theorem stating that we can buy two pies that differ in both filling and preparation
theorem pies_differ_in_both (pies : Set Pie) (h : 3 ≤ pies.card) :
  ∃ (p1 p2 : Pie), p1 ∈ pies ∧ p2 ∈ pies ∧ p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation :=
  sorry

end pies_differ_in_both_l33_33273


namespace train_return_time_l33_33232

open Real

theorem train_return_time
  (C_small : Real := 1.5)
  (C_large : Real := 3)
  (speed : Real := 10)
  (initial_connection : String := "A to C")
  (switch_interval : Real := 1) :
  (126 = 2.1 * 60) :=
sorry

end train_return_time_l33_33232


namespace ratio_of_x_to_y_l33_33769

theorem ratio_of_x_to_y (x y : ℝ) (h : (8 * x - 5 * y) / (11 * x - 3 * y) = 4 / 7) : x / y = 23 / 12 := 
by
  sorry

end ratio_of_x_to_y_l33_33769


namespace highest_power_of_3_divides_l33_33024

def A_n (n : ℕ) : ℕ := (10^(3^n) - 1) / 9

theorem highest_power_of_3_divides (n : ℕ) : ∃ k : ℕ, A_n n = 3^n * k ∧ ¬ (3 * A_n n = 3^(n+1) * k)
:= by
  sorry

end highest_power_of_3_divides_l33_33024


namespace mason_ate_15_hotdogs_l33_33835

structure EatingContest where
  hotdogWeight : ℕ
  burgerWeight : ℕ
  pieWeight : ℕ
  noahBurgers : ℕ
  jacobPiesLess : ℕ
  masonHotdogsWeight : ℕ

theorem mason_ate_15_hotdogs (data : EatingContest)
    (h1 : data.hotdogWeight = 2)
    (h2 : data.burgerWeight = 5)
    (h3 : data.pieWeight = 10)
    (h4 : data.noahBurgers = 8)
    (h5 : data.jacobPiesLess = 3)
    (h6 : data.masonHotdogsWeight = 30) :
    (data.masonHotdogsWeight / data.hotdogWeight) = 15 :=
by
  sorry

end mason_ate_15_hotdogs_l33_33835


namespace coin_flip_probability_difference_l33_33192

theorem coin_flip_probability_difference :
  let p1 := (nat.choose 5 4) * (1 / 2)^5
  let p2 := (1 / 2)^5
  (p1 - p2 = 1 / 8) :=
by
  let p1 := (nat.choose 5 4) * (1 / 2)^5
  let p2 := (1 / 2)^5
  sorry

end coin_flip_probability_difference_l33_33192


namespace probability_at_most_three_heads_10_coins_l33_33599

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l33_33599


namespace stratified_sampling_l33_33900

theorem stratified_sampling (total_students : ℕ) (num_freshmen : ℕ)
                            (freshmen_sample : ℕ) (sample_size : ℕ)
                            (h1 : total_students = 1500)
                            (h2 : num_freshmen = 400)
                            (h3 : freshmen_sample = 12)
                            (h4 : (freshmen_sample : ℚ) / num_freshmen = sample_size / total_students) :
  sample_size = 45 :=
  by
  -- There would be some steps to prove this, but they are omitted.
  sorry

end stratified_sampling_l33_33900


namespace least_days_to_repay_twice_l33_33472

-- Define the initial conditions
def borrowed_amount : ℝ := 15
def daily_interest_rate : ℝ := 0.10
def interest_per_day : ℝ := borrowed_amount * daily_interest_rate
def total_amount_to_repay : ℝ := 2 * borrowed_amount

-- Define the condition we want to prove
theorem least_days_to_repay_twice : ∃ (x : ℕ), (borrowed_amount + interest_per_day * x) ≥ total_amount_to_repay ∧ x = 10 :=
by
  sorry

end least_days_to_repay_twice_l33_33472


namespace complex_power_identity_l33_33053

theorem complex_power_identity (i : ℂ) (h : i^2 = -1) : (1 + i)^4 = -4 :=
by
  sorry

end complex_power_identity_l33_33053


namespace Sam_balloons_correct_l33_33394

def Fred_balloons : Nat := 10
def Dan_balloons : Nat := 16
def Total_balloons : Nat := 72

def Sam_balloons : Nat := Total_balloons - Fred_balloons - Dan_balloons

theorem Sam_balloons_correct : Sam_balloons = 46 := by 
  have H : Sam_balloons = 72 - 10 - 16 := rfl
  simp at H
  exact H

end Sam_balloons_correct_l33_33394


namespace part1_double_root_equation_part2_value_m_squared_2m_2_part3_value_m_l33_33810

-- Part 1: Is x^2 - 3x + 2 = 0 a "double root equation"?
theorem part1_double_root_equation :
    ∃ (x₁ x₂ : ℝ), (x₁ ≠ x₂ ∧ x₁ * 2 = x₂) 
              ∧ (x^2 - 3 * x + 2 = 0) :=
sorry

-- Part 2: Given (x - 2)(x - m) = 0 is a "double root equation", find value of m^2 + 2m + 2.
theorem part2_value_m_squared_2m_2 (m : ℝ) :
    ∃ (v : ℝ), v = m^2 + 2 * m + 2 ∧ 
          (m = 1 ∨ m = 4) ∧
          (v = 5 ∨ v = 26) :=
sorry

-- Part 3: Determine m such that x^2 - (m-1)x + 32 = 0 is a "double root equation".
theorem part3_value_m (m : ℝ) :
    x^2 - (m - 1) * x + 32 = 0 ∧ 
    (m = 13 ∨ m = -11) :=
sorry

end part1_double_root_equation_part2_value_m_squared_2m_2_part3_value_m_l33_33810


namespace add_fractions_11_12_7_15_l33_33374

/-- A theorem stating that the sum of 11/12 and 7/15 is 83/60. -/
theorem add_fractions_11_12_7_15 : (11 / 12) + (7 / 15) = (83 / 60) := 
by
  sorry

end add_fractions_11_12_7_15_l33_33374


namespace probability_of_6_heads_in_10_flips_l33_33672

theorem probability_of_6_heads_in_10_flips :
  let total_outcomes := 2 ^ 10
  let favorable_outcomes := Nat.choose 10 6
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = 210 / 1024 :=
by
  sorry

end probability_of_6_heads_in_10_flips_l33_33672


namespace num_ways_to_buy_three_items_l33_33116

-- Defining the conditions based on the problem statement
def num_headphones : ℕ := 9
def num_mice : ℕ := 13
def num_keyboards : ℕ := 5
def num_kb_mouse_sets : ℕ := 4
def num_hp_mouse_sets : ℕ := 5

-- Defining the theorem statement
theorem num_ways_to_buy_three_items :
  (num_kb_mouse_sets * num_headphones) + 
  (num_hp_mouse_sets * num_keyboards) + 
  (num_headphones * num_mice * num_keyboards) = 646 := 
  by
  sorry

end num_ways_to_buy_three_items_l33_33116


namespace cube_with_holes_l33_33375

-- Definitions and conditions
def edge_length_cube : ℝ := 4
def side_length_hole : ℝ := 2
def depth_hole : ℝ := 1
def number_of_holes : ℕ := 6

-- Prove that the total surface area including inside surfaces is 144 square meters
def total_surface_area_including_inside_surfaces : ℝ :=
  let original_surface_area := 6 * (edge_length_cube ^ 2)
  let area_removed_per_hole := side_length_hole ^ 2
  let area_exposed_inside_per_hole := 2 * (side_length_hole * depth_hole) + area_removed_per_hole
  original_surface_area - number_of_holes * area_removed_per_hole + number_of_holes * area_exposed_inside_per_hole

-- Prove that the total volume of material removed is 24 cubic meters
def total_volume_removed : ℝ :=
  number_of_holes * (side_length_hole ^ 2 * depth_hole)

theorem cube_with_holes :
  total_surface_area_including_inside_surfaces = 144 ∧ total_volume_removed = 24 :=
by
  sorry

end cube_with_holes_l33_33375


namespace right_triangle_area_l33_33525

theorem right_triangle_area (leg1 hypotenuse : ℝ) (h1 : leg1 = 30) (h2 : hypotenuse = 34) (h3 : (leg1 ^ 2 + (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) ^ 2 = hypotenuse ^ 2)) :
  (1 / 2) * leg1 * (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) = 240 :=
by
  sorry

end right_triangle_area_l33_33525


namespace Barbara_spent_46_22_on_different_goods_l33_33627

theorem Barbara_spent_46_22_on_different_goods :
  let tuna_cost := (5 * 2) -- Total cost of tuna
  let water_cost := (4 * 1.5) -- Total cost of water
  let total_before_discount := 56 / 0.9 -- Total before discount, derived from the final amount paid after discount
  let total_tuna_water_cost := 10 + 6 -- Total cost of tuna and water together
  let different_goods_cost := total_before_discount - total_tuna_water_cost
  different_goods_cost = 46.22 := 
sorry

end Barbara_spent_46_22_on_different_goods_l33_33627


namespace arithmetic_sequence_100th_term_l33_33379

-- Define the first term and the common difference
def first_term : ℕ := 3
def common_difference : ℕ := 7

-- Define the formula for the nth term of an arithmetic sequence
def nth_term (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- Theorem: The 100th term of the arithmetic sequence is 696.
theorem arithmetic_sequence_100th_term :
  nth_term first_term common_difference 100 = 696 :=
  sorry

end arithmetic_sequence_100th_term_l33_33379


namespace probability_A_does_not_lose_l33_33438

theorem probability_A_does_not_lose (pA_wins p_draw : ℝ) (hA_wins : pA_wins = 0.4) (h_draw : p_draw = 0.2) :
  pA_wins + p_draw = 0.6 :=
by
  sorry

end probability_A_does_not_lose_l33_33438


namespace cos_shifted_alpha_l33_33796

theorem cos_shifted_alpha (α : ℝ) (h1 : Real.tan α = -3/4) (h2 : α ∈ Set.Ioc (3*Real.pi/2) (2*Real.pi)) :
  Real.cos (Real.pi/2 + α) = 3/5 :=
sorry

end cos_shifted_alpha_l33_33796


namespace gcd_8251_6105_l33_33021

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l33_33021


namespace find_c_l33_33337

noncomputable def f (x c : ℝ) := x * (x - c) ^ 2
noncomputable def f' (x c : ℝ) := 3 * x ^ 2 - 4 * c * x + c ^ 2
noncomputable def f'' (x c : ℝ) := 6 * x - 4 * c

theorem find_c (c : ℝ) : f' 2 c = 0 ∧ f'' 2 c < 0 → c = 6 :=
by {
  sorry
}

end find_c_l33_33337


namespace promoting_cashback_beneficial_for_bank_cashback_in_rubles_preferable_l33_33216

-- Definitions of conditions
def attracts_new_clients (bank_promotes_cashback : Prop) : Prop :=
  bank_promotes_cashback → ∃ (new_clients : Prop), new_clients

def promotes_partnerships (bank_promotes_cashback : Prop) : Prop :=
  bank_promotes_cashback → ∃ (partnerships : Prop), partnerships

def enhances_competitiveness (bank_promotes_cashback : Prop) : Prop :=
  bank_promotes_cashback → ∃ (competitiveness : Prop), competitiveness

def liquidity_advantage (cashback_rubles : Prop) : Prop :=
  cashback_rubles → ∃ (liquidity : Prop), liquidity

def no_expiry_concerns (cashback_rubles : Prop) : Prop :=
  cashback_rubles → ∃ (no_expiry : Prop), no_expiry

def no_partner_limitations (cashback_rubles : Prop) : Prop :=
  cashback_rubles → ∃ (partner_limitations : Prop), ¬partner_limitations

-- Lean statements for the proof problems
theorem promoting_cashback_beneficial_for_bank (bank_promotes_cashback : Prop) :
  attracts_new_clients bank_promotes_cashback ∧
  promotes_partnerships bank_promotes_cashback ∧ 
  enhances_competitiveness bank_promotes_cashback →
  bank_promotes_cashback := 
sorry

theorem cashback_in_rubles_preferable (cashback_rubles : Prop) :
  liquidity_advantage cashback_rubles ∧
  no_expiry_concerns cashback_rubles ∧
  no_partner_limitations cashback_rubles →
  cashback_rubles :=
sorry

end promoting_cashback_beneficial_for_bank_cashback_in_rubles_preferable_l33_33216


namespace buy_items_ways_l33_33120

theorem buy_items_ways (headphones keyboards mice keyboard_mouse_sets headphone_mouse_sets : ℕ) :
  headphones = 9 → keyboards = 5 → mice = 13 → keyboard_mouse_sets = 4 → headphone_mouse_sets = 5 →
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  intros h_eq k_eq m_eq kms_eq hms_eq
  have h_eq_gen : headphones = 9 := h_eq
  have k_eq_gen : keyboards = 5 := k_eq
  have m_eq_gen : mice = 13 := m_eq
  have kms_eq_gen : keyboard_mouse_sets = 4 := kms_eq
  have hms_eq_gen : headphone_mouse_sets = 5 := hms_eq
  sorry

end buy_items_ways_l33_33120


namespace probability_at_most_3_heads_l33_33557

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l33_33557


namespace right_triangle_area_l33_33528

theorem right_triangle_area (leg1 leg2 hypotenuse : ℕ) (h_leg1 : leg1 = 30)
  (h_hypotenuse : hypotenuse = 34)
  (hypotenuse_sq : hypotenuse * hypotenuse = leg1 * leg1 + leg2 * leg2) :
  (1 / 2 : ℚ) * leg1 * leg2 = 240 := by
  sorry

end right_triangle_area_l33_33528


namespace problem_a_problem_b_l33_33793

-- Definition for real roots condition in problem A
def has_real_roots (k : ℝ) : Prop :=
  let a := 1
  let b := -3
  let c := k
  b^2 - 4 * a * c ≥ 0

-- Problem A: Proving the range of k
theorem problem_a (k : ℝ) : has_real_roots k ↔ k ≤ 9 / 4 :=
by
  sorry

-- Definition for a quadratic equation having a given root
def has_root (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

-- Problem B: Proving the value of m given a common root condition
theorem problem_b (m : ℝ) : 
  (has_root 1 (-3) 2 1 ∧ has_root (m-1) 1 (m-3) 1) ↔ m = 3 / 2 :=
by
  sorry

end problem_a_problem_b_l33_33793


namespace inverse_proportion_function_has_m_value_l33_33339

theorem inverse_proportion_function_has_m_value
  (k : ℝ)
  (h1 : 2 * -3 = k)
  {m : ℝ}
  (h2 : 6 = k / m) :
  m = -1 :=
by
  sorry

end inverse_proportion_function_has_m_value_l33_33339


namespace find_valid_ns_l33_33642

theorem find_valid_ns (n : ℕ) (h1 : n > 1) (h2 : ∃ k : ℕ, k^2 = (n^2 + 7 * n + 136) / (n-1)) : n = 5 ∨ n = 37 :=
sorry

end find_valid_ns_l33_33642


namespace min_steps_for_humpty_l33_33098

theorem min_steps_for_humpty (x y : ℕ) (H : 47 * x - 37 * y = 1) : x + y = 59 :=
  sorry

end min_steps_for_humpty_l33_33098


namespace smallest_omega_l33_33003

theorem smallest_omega (ω : ℝ) (hω_pos : ω > 0) :
  (∃ k : ℤ, (2 / 3) * ω = 2 * k) -> ω = 3 :=
by
  sorry

end smallest_omega_l33_33003


namespace minimum_value_l33_33937

noncomputable def min_value_b_plus_4_over_a (a : ℝ) (b : ℝ) :=
  b + 4 / a

theorem minimum_value (a : ℝ) (b : ℝ) (h₁ : a > 0) 
  (h₂ : ∀ x : ℝ, x > 0 → (a * x - 2) * (x^2 + b * x - 5) ≥ 0) :
  min_value_b_plus_4_over_a a b = 2 * Real.sqrt 5 :=
sorry

end minimum_value_l33_33937


namespace larger_square_area_l33_33625

theorem larger_square_area 
    (s₁ s₂ s₃ s₄ : ℕ) 
    (H1 : s₁ = 20) 
    (H2 : s₂ = 10) 
    (H3 : s₃ = 18) 
    (H4 : s₄ = 12) :
    (s₃ + s₄) > (s₁ + s₂) :=
by
  sorry

end larger_square_area_l33_33625


namespace reflected_line_equation_l33_33861

-- Definitions based on given conditions
def incident_line (x : ℝ) : ℝ := 2 * x + 1
def reflection_line (x : ℝ) : ℝ := x

-- Statement of the mathematical problem
theorem reflected_line_equation :
  ∀ x y : ℝ, (incident_line x = y) → (reflection_line x = x) → y = (1/2) * x - (1/2) :=
sorry

end reflected_line_equation_l33_33861


namespace shop_combinations_l33_33130

theorem shop_combinations :
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let keyboard_and_mouse_sets := 4
  let headphone_and_mouse_sets := 5
  have case1 : keyboard_and_mouse_sets * headphones = 4 * 9, by sorry
  have case2 : headphone_and_mouse_sets * keyboards = 5 * 5, by sorry
  have case3 : headphones * mice * keyboards = 9 * 13 * 5, by sorry
  show (keyboard_and_mouse_sets * headphones) + (headphone_and_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646,
  by rw [case1, case2, case3]; exact sorry

end shop_combinations_l33_33130


namespace min_sum_a1_a2_l33_33508

-- Define the condition predicate for the sequence
def satisfies_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 2) = (a n + 2009) / (1 + a (n + 1))

-- State the main problem as a theorem in Lean 4
theorem min_sum_a1_a2 (a : ℕ → ℕ) (h_seq : satisfies_seq a) (h_pos : ∀ n, a n > 0) :
  a 1 * a 2 = 2009 → a 1 + a 2 = 90 :=
sorry

end min_sum_a1_a2_l33_33508


namespace olivia_spent_amount_l33_33177

noncomputable def initial_amount : ℕ := 100
noncomputable def collected_amount : ℕ := 148
noncomputable def final_amount : ℕ := 159

theorem olivia_spent_amount :
  initial_amount + collected_amount - final_amount = 89 :=
by
  sorry

end olivia_spent_amount_l33_33177


namespace exists_consecutive_composite_l33_33701

theorem exists_consecutive_composite (n : ℕ) : 
  ∃ (a : ℕ → ℕ), 
    (∀ i : ℕ, 2 ≤ i ∧ i ≤ n + 1 → a i = (n+1)! + i ∧ ∀ j : ℕ, 2 ≤ j → j ∣ a i ∧ j ∤ 1 ∧ j ∤ a i → a i ≠ j ∧ a i ≠ ((n+1)! + i) ∧ a i > 1) := 
sorry

end exists_consecutive_composite_l33_33701


namespace sun_city_population_correct_l33_33325

noncomputable def willowdale_population : Nat := 2000
noncomputable def roseville_population : Nat := 3 * willowdale_population - 500
noncomputable def sun_city_population : Nat := 2 * roseville_population + 1000

theorem sun_city_population_correct : sun_city_population = 12000 := by
  sorry

end sun_city_population_correct_l33_33325


namespace number_of_students_in_class_l33_33169

theorem number_of_students_in_class :
  ∃ a : ℤ, 100 ≤ a ∧ a ≤ 200 ∧ a % 4 = 1 ∧ a % 3 = 2 ∧ a % 7 = 3 ∧ a = 101 := 
sorry

end number_of_students_in_class_l33_33169


namespace smallest_of_three_consecutive_even_numbers_l33_33995

def sum_of_three_consecutive_even_numbers (n : ℕ) : Prop :=
  n + (n + 2) + (n + 4) = 162

theorem smallest_of_three_consecutive_even_numbers (n : ℕ) (h : sum_of_three_consecutive_even_numbers n) : n = 52 :=
by
  sorry

end smallest_of_three_consecutive_even_numbers_l33_33995


namespace optimal_tablet_combination_exists_l33_33872

/-- Define the daily vitamin requirement structure --/
structure Vitamins (A B C D : ℕ)

theorem optimal_tablet_combination_exists {x y : ℕ} :
  (∃ (x y : ℕ), 
    (3 * x ≥ 3) ∧ (x + y ≥ 9) ∧ (x + 3 * y ≥ 15) ∧ (2 * y ≥ 2) ∧
    (x + y = 9) ∧ 
    (20 * x + 60 * y = 3) ∧ 
    (x + 2 * y = 12) ∧ 
    (x = 6 ∧ y = 3)) := 
  by
  sorry

end optimal_tablet_combination_exists_l33_33872


namespace sam_letters_on_wednesday_l33_33841

/-- Sam's average letters per day. -/
def average_letters_per_day : ℕ := 5

/-- Number of days Sam wrote letters. -/
def number_of_days : ℕ := 2

/-- Letters Sam wrote on Tuesday. -/
def letters_on_tuesday : ℕ := 7

/-- Total letters Sam wrote in two days. -/
def total_letters : ℕ := average_letters_per_day * number_of_days

/-- Letters Sam wrote on Wednesday. -/
def letters_on_wednesday : ℕ := total_letters - letters_on_tuesday

theorem sam_letters_on_wednesday : letters_on_wednesday = 3 :=
by
  -- placeholder proof
  sorry

end sam_letters_on_wednesday_l33_33841


namespace triangle_sine_value_l33_33107

-- Define the triangle sides and angles
variables {a b c A B C : ℝ}

-- Main theorem stating the proof problem
theorem triangle_sine_value (h : a^2 = b^2 + c^2 - bc) :
  (a * Real.sin B) / b = Real.sqrt 3 / 2 := sorry

end triangle_sine_value_l33_33107


namespace inequality_solution_set_l33_33945

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 else -1

theorem inequality_solution_set :
  { x : ℝ | (x+1) * f x > 2 } = { x : ℝ | x < -3 } ∪ { x : ℝ | x > 1 } :=
by
  sorry

end inequality_solution_set_l33_33945


namespace Papi_Calot_plants_l33_33697

theorem Papi_Calot_plants :
  let initial_potatoes_plants := 10 * 25
  let initial_carrots_plants := 15 * 30
  let initial_onions_plants := 12 * 20
  let total_potato_plants := initial_potatoes_plants + 20
  let total_carrot_plants := initial_carrots_plants + 30
  let total_onion_plants := initial_onions_plants + 10
  total_potato_plants = 270 ∧
  total_carrot_plants = 480 ∧
  total_onion_plants = 250 := by
  sorry

end Papi_Calot_plants_l33_33697


namespace simplify_expression_l33_33493

theorem simplify_expression : 4 * (15 / 5) * (24 / -60) = - (24 / 5) := 
by
  sorry

end simplify_expression_l33_33493


namespace proof_problem_l33_33762

-- List of body weight increases in control group
def controlGroup : List ℝ := [15.2, 18.8, 20.2, 21.3, 22.5, 23.2, 25.8, 26.5, 27.5, 30.1,
                              32.6, 34.3, 34.8, 35.6, 35.6, 35.8, 36.2, 37.3, 40.5, 43.2]

-- List of body weight increases in experimental group
def experimentalGroup : List ℝ := [7.8, 9.2, 11.4, 12.4, 13.2, 15.5, 16.5, 18.0, 18.8, 19.2,
                                   19.8, 20.2, 21.6, 22.8, 23.6, 23.9, 25.1, 28.2, 32.3, 36.5]

-- Sample mean of the experimental group
noncomputable def sampleMeanExperimental : ℝ := (experimentalGroup.sum) / (experimentalGroup.length)

-- Median m of the combined list
def combinedGroup : List ℝ := (controlGroup ++ experimentalGroup).sort
noncomputable def median : ℝ := (combinedGroup.get! 19 + combinedGroup.get! 20) / 2

-- Contingency table counts
def controlLessThanM : ℕ := controlGroup.filter (λ x => x < median).length
def controlGreaterThanOrEqualM : ℕ := controlGroup.filter (λ x => x >= median).length
def experimentalLessThanM : ℕ := experimentalGroup.filter (λ x => x < median).length
def experimentalGreaterThanOrEqualM : ℕ := experimentalGroup.filter (λ x => x >= median).length

-- Chi-squared value K^2
noncomputable def K2 : ℝ :=
  let a := controlLessThanM
  let b := controlGreaterThanOrEqualM
  let c := experimentalLessThanM
  let d := experimentalGreaterThanOrEqualM
  let n := controlGroup.length + experimentalGroup.length
  n * (a * d - b * c) ^ 2 / ((a + b) * (c + d) * (a + c) * (b + d))

theorem proof_problem :
  sampleMeanExperimental = 19.8 ∧
  median = 23.4 ∧
  controlLessThanM = 6 ∧ controlGreaterThanOrEqualM = 14 ∧
  experimentalLessThanM = 14 ∧ experimentalGreaterThanOrEqualM = 6 ∧
  K2 = 6.4 :=
by
  unfold sampleMeanExperimental median controlLessThanM controlGreaterThanOrEqualM
         experimentalLessThanM experimentalGreaterThanOrEqualM K2
  simp only [*, List.length, List.sum, List.filter, List.sort, List.get!]
  -- Add necessary calculations to verify each part
  sorry

end proof_problem_l33_33762


namespace find_m_values_l33_33809

theorem find_m_values (m : ℕ) : (m - 3) ^ m = 1 ↔ m = 0 ∨ m = 2 ∨ m = 4 := sorry

end find_m_values_l33_33809


namespace small_pos_int_n_l33_33408

theorem small_pos_int_n (a : ℕ → ℕ) (n : ℕ) (a1_val : a 1 = 7)
  (recurrence: ∀ n, a (n + 1) = a n * (a n + 2)) :
  ∃ n : ℕ, a n > 2 ^ 4036 ∧ ∀ m : ℕ, (m < n) → a m ≤ 2 ^ 4036 :=
by
  sorry

end small_pos_int_n_l33_33408


namespace divisor_of_p_l33_33149

-- Define the necessary variables and assumptions
variables (p q r s : ℕ)

-- State the conditions
def conditions := gcd p q = 28 ∧ gcd q r = 45 ∧ gcd r s = 63 ∧ 80 < gcd s p ∧ gcd s p < 120 

-- State the proposition to prove: 11 divides p
theorem divisor_of_p (h : conditions p q r s) : 11 ∣ p := 
sorry

end divisor_of_p_l33_33149


namespace license_plates_count_l33_33668

def num_consonants : Nat := 20
def num_vowels : Nat := 6
def num_digits : Nat := 10
def num_symbols : Nat := 3

theorem license_plates_count : 
  num_consonants * num_vowels * num_consonants * num_digits * num_symbols = 72000 :=
by 
  sorry

end license_plates_count_l33_33668


namespace glass_sphere_wall_thickness_l33_33045

/-- Mathematically equivalent proof problem statement:
Given a hollow glass sphere with outer diameter 16 cm such that 3/8 of its surface remains dry,
and specific gravity of glass s = 2.523. The wall thickness of the sphere is equal to 0.8 cm. -/
theorem glass_sphere_wall_thickness 
  (outer_diameter : ℝ) (dry_surface_fraction : ℝ) (specific_gravity : ℝ) (required_thickness : ℝ) 
  (uniform_thickness : outer_diameter = 16)
  (dry_surface : dry_surface_fraction = 3 / 8)
  (s : specific_gravity = 2.523) :
  required_thickness = 0.8 :=
by
  sorry

end glass_sphere_wall_thickness_l33_33045


namespace find_g_inverse_sum_l33_33476

noncomputable def g (x : ℝ) : ℝ :=
if x ≤ 2 then x^2 - 2 * x + 2 else 3 - x

theorem find_g_inverse_sum :
  (∃ x, g x = -2 ∧ x = 5) ∧
  (∃ x, g x = 0 ∧ x = 3) ∧
  (∃ x, g x = 2 ∧ x = 0) ∧
  (5 + 3 + 0 = 8) := by
  sorry

end find_g_inverse_sum_l33_33476


namespace solve_fraction_equation_l33_33854

theorem solve_fraction_equation (x : ℝ) :
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ↔ x = -9 :=
by {
  sorry
}

end solve_fraction_equation_l33_33854


namespace distinct_four_digit_integers_with_one_repeating_digit_l33_33255

-- Define the set of odd digits
def odd_digits : Finset ℕ := {1, 3, 5, 7, 9}

-- Define the problem statement
theorem distinct_four_digit_integers_with_one_repeating_digit : 
  {n : ℕ // 1000 ≤ n ∧ n < 10000 ∧ (∀ d ∈ n.digits 10, d ∈ odd_digits) ∧ (∃ d ∈ odd_digits, n.digits 10.count d = 2)}.card = 360 := 
by
  sorry

end distinct_four_digit_integers_with_one_repeating_digit_l33_33255


namespace find_a_from_function_property_l33_33507

theorem find_a_from_function_property {a : ℝ} (h : ∀ (x : ℝ), (0 ≤ x → x ≤ 1 → ax ≤ 3) ∧ (0 ≤ x → x ≤ 1 → ax ≥ 3)) :
  a = 3 :=
sorry

end find_a_from_function_property_l33_33507


namespace triangle_equilateral_of_equal_angle_ratios_l33_33993

theorem triangle_equilateral_of_equal_angle_ratios
  (a b c : ℝ)
  (h₁ : a + b + c = 180)
  (h₂ : a = b)
  (h₃ : b = c) :
  a = 60 ∧ b = 60 ∧ c = 60 :=
by
  sorry

end triangle_equilateral_of_equal_angle_ratios_l33_33993


namespace num_terminating_decimals_l33_33789

theorem num_terminating_decimals (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℕ, k = 20) := by
  have h2 : ∀ n, (1 ≤ n ∧ n ≤ 180) → (∃ k, k = (180 / 9)) := sorry
  have h3 : 180 / 9 = 20 := by norm_num
  use 20
  exact h3

end num_terminating_decimals_l33_33789


namespace bucket_weight_l33_33039

theorem bucket_weight (x y p q : ℝ) 
  (h1 : x + (3 / 4) * y = p) 
  (h2 : x + (1 / 3) * y = q) :
  x + (5 / 6) * y = (6 * p - q) / 5 :=
sorry

end bucket_weight_l33_33039


namespace lisa_photos_last_weekend_l33_33299

def photos_of_animals : ℕ := 10
def photos_of_flowers : ℕ := 3 * photos_of_animals
def photos_of_scenery : ℕ := photos_of_flowers - 10
def total_photos_this_week : ℕ := photos_of_animals + photos_of_flowers + photos_of_scenery
def photos_last_weekend : ℕ := total_photos_this_week - 15

theorem lisa_photos_last_weekend : photos_last_weekend = 45 :=
by
  sorry

end lisa_photos_last_weekend_l33_33299


namespace probability_of_at_most_3_heads_out_of_10_l33_33586
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l33_33586


namespace quadratic_has_one_solution_l33_33237

theorem quadratic_has_one_solution (m : ℝ) : 3 * (49 / 12) - 7 * (49 / 12) + m = 0 → m = 49 / 12 :=
by
  sorry

end quadratic_has_one_solution_l33_33237


namespace probability_at_most_3_heads_l33_33561

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l33_33561


namespace adjacent_block_permutations_l33_33817

-- Define the set of digits
def digits : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Define the block of digits that must be adjacent
def block : List ℕ := [2, 5, 8]

-- Function to calculate permutations of a list (size n)
def fact (n : ℕ) : ℕ := Nat.factorial n

-- Calculate the total number of arrangements
def total_arrangements : ℕ := fact 8 * fact 3

-- The main theorem statement to be proved
theorem adjacent_block_permutations :
  total_arrangements = 241920 :=
by
  sorry

end adjacent_block_permutations_l33_33817


namespace r_daily_earnings_l33_33360

-- Given conditions as definitions
def daily_earnings (P Q R : ℕ) : Prop :=
(P + Q + R) * 9 = 1800 ∧ (P + R) * 5 = 600 ∧ (Q + R) * 7 = 910

-- Theorem statement corresponding to the problem
theorem r_daily_earnings : ∃ R : ℕ, ∀ P Q : ℕ, daily_earnings P Q R → R = 50 :=
by sorry

end r_daily_earnings_l33_33360


namespace puppies_per_cage_calculation_l33_33368

noncomputable def initial_puppies : ℝ := 18.0
noncomputable def additional_puppies : ℝ := 3.0
noncomputable def total_puppies : ℝ := initial_puppies + additional_puppies
noncomputable def total_cages : ℝ := 4.2
noncomputable def puppies_per_cage : ℝ := total_puppies / total_cages

theorem puppies_per_cage_calculation :
  puppies_per_cage = 5.0 :=
by
  sorry

end puppies_per_cage_calculation_l33_33368


namespace pascal_triangle_41st_number_42nd_row_l33_33186

open Nat

theorem pascal_triangle_41st_number_42nd_row :
  Nat.choose 42 40 = 861 := by
  sorry

end pascal_triangle_41st_number_42nd_row_l33_33186


namespace probability_heads_at_most_three_out_of_ten_coins_l33_33542

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l33_33542


namespace find_interval_for_inequality_l33_33383

open Set

theorem find_interval_for_inequality :
  {x : ℝ | (1 / (x^2 + 2) > 4 / x + 21 / 10)} = Ioo (-2 : ℝ) (0 : ℝ) := 
sorry

end find_interval_for_inequality_l33_33383


namespace average_wage_correct_l33_33888

def male_workers : ℕ := 20
def female_workers : ℕ := 15
def child_workers : ℕ := 5

def male_wage : ℕ := 25
def female_wage : ℕ := 20
def child_wage : ℕ := 8

def total_amount_paid_per_day : ℕ := 
  (male_workers * male_wage) + (female_workers * female_wage) + (child_workers * child_wage)

def total_number_of_workers : ℕ := 
  male_workers + female_workers + child_workers

def average_wage_per_day : ℕ := 
  total_amount_paid_per_day / total_number_of_workers

theorem average_wage_correct : 
  average_wage_per_day = 21 := by 
  sorry

end average_wage_correct_l33_33888


namespace sum_of_fractions_l33_33915

theorem sum_of_fractions:
  (7 / 12) + (11 / 15) = 79 / 60 :=
by
  sorry

end sum_of_fractions_l33_33915


namespace value_of_power_l33_33651

theorem value_of_power (a : ℝ) (m n k : ℕ) (h1 : a ^ m = 2) (h2 : a ^ n = 4) (h3 : a ^ k = 32) : 
  a ^ (3 * m + 2 * n - k) = 4 := 
by sorry

end value_of_power_l33_33651


namespace number_of_ways_to_choose_4_from_28_number_of_ways_to_choose_3_from_27_with_kolya_included_l33_33248

-- Part (a)
theorem number_of_ways_to_choose_4_from_28 :
  (Nat.choose 28 4) = 20475 :=
sorry

-- Part (b)
theorem number_of_ways_to_choose_3_from_27_with_kolya_included :
  (Nat.choose 27 3) = 2925 :=
sorry

end number_of_ways_to_choose_4_from_28_number_of_ways_to_choose_3_from_27_with_kolya_included_l33_33248


namespace find_y_value_l33_33496

theorem find_y_value (t : ℝ) (x : ℝ) (y : ℝ)
  (h1 : x = 3 - 2 * t)
  (h2 : y = 3 * t + 6)
  (h3 : x = -6)
  : y = 19.5 :=
by {
  sorry
}

end find_y_value_l33_33496


namespace combined_profit_percentage_correct_l33_33369

-- Definitions based on the conditions
noncomputable def profit_percentage_A := 30
noncomputable def discount_percentage_A := 10
noncomputable def profit_percentage_B := 24
noncomputable def discount_percentage_B := 15
noncomputable def profit_percentage_C := 40
noncomputable def discount_percentage_C := 20

-- Function to calculate selling price without discount
noncomputable def selling_price_without_discount (cost_price profit_percentage : ℝ) : ℝ :=
  cost_price * (1 + profit_percentage / 100)

-- Assume cost price for simplicity
noncomputable def cost_price : ℝ := 100

-- Calculations based on the conditions
noncomputable def selling_price_A := selling_price_without_discount cost_price profit_percentage_A
noncomputable def selling_price_B := selling_price_without_discount cost_price profit_percentage_B
noncomputable def selling_price_C := selling_price_without_discount cost_price profit_percentage_C

-- Calculate total cost price and the total selling price without any discount
noncomputable def total_cost_price := 3 * cost_price
noncomputable def total_selling_price_without_discount := selling_price_A + selling_price_B + selling_price_C

-- Combined profit
noncomputable def combined_profit := total_selling_price_without_discount - total_cost_price

-- Combined profit percentage
noncomputable def combined_profit_percentage := (combined_profit / total_cost_price) * 100

theorem combined_profit_percentage_correct :
  combined_profit_percentage = 31.33 :=
by
  sorry

end combined_profit_percentage_correct_l33_33369


namespace functional_relationship_inversely_proportional_l33_33751

-- Definitions based on conditions
def table_data : List (ℝ × ℝ) := [(100, 1.00), (200, 0.50), (400, 0.25), (500, 0.20)]

-- The main conjecture to be proved
theorem functional_relationship_inversely_proportional (y x : ℝ) (h : (x, y) ∈ table_data) : y = 100 / x :=
sorry

end functional_relationship_inversely_proportional_l33_33751


namespace comb_n_plus_1_2_l33_33513

theorem comb_n_plus_1_2 (n : ℕ) (h : 0 < n) : 
  (n + 1).choose 2 = (n + 1) * n / 2 :=
by sorry

end comb_n_plus_1_2_l33_33513


namespace smallest_integer_for_perfect_square_l33_33752

-- Given condition: y = 2^3 * 3^2 * 4^6 * 5^5 * 7^8 * 8^3 * 9^10 * 11^11
def y : ℕ := 2^3 * 3^2 * 4^6 * 5^5 * 7^8 * 8^3 * 9^10 * 11^11

-- The statement to prove
theorem smallest_integer_for_perfect_square (y : ℕ) : ∃ n : ℕ, n = 110 ∧ ∃ m : ℕ, (y * n) = m^2 := 
by {
  sorry
}

end smallest_integer_for_perfect_square_l33_33752


namespace num_ways_to_buy_three_items_l33_33117

-- Defining the conditions based on the problem statement
def num_headphones : ℕ := 9
def num_mice : ℕ := 13
def num_keyboards : ℕ := 5
def num_kb_mouse_sets : ℕ := 4
def num_hp_mouse_sets : ℕ := 5

-- Defining the theorem statement
theorem num_ways_to_buy_three_items :
  (num_kb_mouse_sets * num_headphones) + 
  (num_hp_mouse_sets * num_keyboards) + 
  (num_headphones * num_mice * num_keyboards) = 646 := 
  by
  sorry

end num_ways_to_buy_three_items_l33_33117


namespace marie_finishes_fourth_task_at_11_40_am_l33_33482

-- Define the given conditions
def start_time : ℕ := 7 * 60 -- start time in minutes from midnight (7:00 AM)
def second_task_end_time : ℕ := 9 * 60 + 20 -- end time of second task in minutes from midnight (9:20 AM)
def num_tasks : ℕ := 4 -- four tasks
def task_duration : ℕ := (second_task_end_time - start_time) / 2 -- duration of one task

-- Define the goal to prove: the end time of the fourth task
def fourth_task_finish_time : ℕ := second_task_end_time + 2 * task_duration

theorem marie_finishes_fourth_task_at_11_40_am : fourth_task_finish_time = 11 * 60 + 40 := by
  sorry

end marie_finishes_fourth_task_at_11_40_am_l33_33482


namespace probability_10_coins_at_most_3_heads_l33_33564

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l33_33564


namespace at_least_one_half_l33_33345

theorem at_least_one_half (x y z : ℝ) (h : x + y + z - 2 * (x * y + y * z + x * z) + 4 * x * y * z = 1 / 2) :
  x = 1 / 2 ∨ y = 1 / 2 ∨ z = 1 / 2 :=
by
  sorry

end at_least_one_half_l33_33345


namespace geometric_progression_complex_l33_33691

theorem geometric_progression_complex (a b c m : ℂ) (r : ℂ) (hr : r ≠ 0) 
    (h1 : a = r) (h2 : b = r^2) (h3 : c = r^3) 
    (h4 : a / (1 - b) = m) (h5 : b / (1 - c) = m) (h6 : c / (1 - a) = m) : 
    ∃ m : ℂ, ∀ a b c : ℂ, ∃ r : ℂ, a = r ∧ b = r^2 ∧ c = r^3 
    ∧ r ≠ 0 
    ∧ (a / (1 - b) = m) 
    ∧ (b / (1 - c) = m) 
    ∧ (c / (1 - a) = m) := 
sorry

end geometric_progression_complex_l33_33691


namespace fair_coin_flip_difference_l33_33205

theorem fair_coin_flip_difference :
  let P1 := 5 * (1 / 2)^5;
  let P2 := (1 / 2)^5;
  P1 - P2 = 9 / 32 :=
by
  sorry

end fair_coin_flip_difference_l33_33205


namespace max_planes_15_points_l33_33940

-- Define the total number of points
def total_points : ℕ := 15

-- Define the number of collinear points
def collinear_points : ℕ := 5

-- Compute the binomial coefficient C(n, k)
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Total number of planes formed by any 3 out of 15 points
def total_planes : ℕ := binom total_points 3

-- Number of degenerate planes formed by the collinear points
def degenerate_planes : ℕ := binom collinear_points 3

-- Maximum number of unique planes
def max_unique_planes : ℕ := total_planes - degenerate_planes

-- Lean theorem statement
theorem max_planes_15_points : max_unique_planes = 445 :=
by
  sorry

end max_planes_15_points_l33_33940


namespace blue_notes_per_red_note_l33_33142

-- Given conditions
def total_red_notes : ℕ := 5 * 6
def additional_blue_notes : ℕ := 10
def total_notes : ℕ := 100
def total_blue_notes := total_notes - total_red_notes

-- Proposition that needs to be proved
theorem blue_notes_per_red_note (x : ℕ) : total_red_notes * x + additional_blue_notes = total_blue_notes → x = 2 := by
  intro h
  sorry

end blue_notes_per_red_note_l33_33142


namespace union_sets_eq_l33_33833

-- Definitions of the sets M and N according to the conditions.
def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | Real.log x ≤ 0}

-- The theorem we want to prove
theorem union_sets_eq :
  (M ∪ N) = Set.Icc 0 1 :=
by
  sorry

end union_sets_eq_l33_33833


namespace total_toys_given_l33_33767

theorem total_toys_given (toys_for_boys : ℕ) (toys_for_girls : ℕ) (h1 : toys_for_boys = 134) (h2 : toys_for_girls = 269) : 
  toys_for_boys + toys_for_girls = 403 := 
by 
  sorry

end total_toys_given_l33_33767


namespace imaginary_part_of_complex_l33_33498

theorem imaginary_part_of_complex : ∀ z : ℂ, z = i^2 * (1 + i) → z.im = -1 :=
by
  intro z
  intro h
  sorry

end imaginary_part_of_complex_l33_33498


namespace area_of_curve_l33_33052

noncomputable def polar_curve (φ : Real) : Real :=
  (1 / 2) + Real.sin φ

noncomputable def area_enclosed_by_polar_curve : Real :=
  2 * ((1 / 2) * ∫ (φ : Real) in (-Real.pi / 2)..(Real.pi / 2), (polar_curve φ) ^ 2)

theorem area_of_curve : area_enclosed_by_polar_curve = (3 * Real.pi) / 4 :=
by
  sorry

end area_of_curve_l33_33052


namespace increasing_interval_l33_33168

-- Given function definition
def quad_func (x : ℝ) : ℝ := -x^2 + 1

-- Property to be proven: The function is increasing on the interval (-∞, 0]
theorem increasing_interval : ∀ x y : ℝ, x ≤ 0 → y ≤ 0 → x < y → quad_func x < quad_func y := by
  sorry

end increasing_interval_l33_33168


namespace max_acceptable_ages_l33_33859

noncomputable def acceptable_ages (avg_age std_dev : ℕ) : ℕ :=
  let lower_limit := avg_age - 2 * std_dev
  let upper_limit := avg_age + 2 * std_dev
  upper_limit - lower_limit + 1

theorem max_acceptable_ages : acceptable_ages 40 10 = 41 :=
by
  sorry

end max_acceptable_ages_l33_33859


namespace gcd_8251_6105_l33_33019

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l33_33019


namespace farm_horses_more_than_cows_l33_33486

variable (x : ℤ) -- number of cows initially, must be a positive integer

def initial_horses := 6 * x
def initial_cows := x
def horses_after_transaction := initial_horses - 30
def cows_after_transaction := initial_cows + 30

-- New ratio after transaction
def new_ratio := horses_after_transaction * 1 = 4 * cows_after_transaction

-- Prove that the farm owns 315 more horses than cows after transaction
theorem farm_horses_more_than_cows :
  new_ratio → horses_after_transaction - cows_after_transaction = 315 :=
by
  sorry

end farm_horses_more_than_cows_l33_33486


namespace real_values_x_l33_33063

theorem real_values_x (x y : ℝ) :
  (3 * y^2 + 5 * x * y + x + 7 = 0) →
  (5 * x + 6) * (5 * x - 14) ≥ 0 →
  x ≤ -6 / 5 ∨ x ≥ 14 / 5 :=
by
  sorry

end real_values_x_l33_33063


namespace nth_equation_l33_33305

theorem nth_equation (n : ℕ) (hn : n > 0) : 9 * n + (n - 1) = 10 * n - 1 :=
sorry

end nth_equation_l33_33305


namespace average_DE_l33_33073

theorem average_DE 
  (a b c d e : ℝ) 
  (avg_all : (a + b + c + d + e) / 5 = 80) 
  (avg_abc : (a + b + c) / 3 = 78) : 
  (d + e) / 2 = 83 := 
sorry

end average_DE_l33_33073


namespace right_triangle_area_l33_33518

/-- Given a right triangle with one leg of length 30 inches and a hypotenuse of 34 inches,
    the area of the triangle is 240 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 :=
by
  rw [h1, h2] at h3
  have hb : b = 16 := by
    rw [←h3]
    norm_num
  rw [h1, hb]
  norm_num
  sorry

end right_triangle_area_l33_33518


namespace monotonic_increasing_interval_l33_33721

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem monotonic_increasing_interval :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → ∀ (x1 x2 : ℝ), 0 ≤ x1 ∧ x1 ≤ 1 → 0 ≤ x2 ∧ x2 ≤ 1 → x1 ≤ x2 → sqrt (- x1 ^ 2 + 2 * x1) ≤ sqrt (- x2 ^ 2 + 2 * x2) :=
sorry

end monotonic_increasing_interval_l33_33721


namespace unique_solution_eqn_l33_33932

theorem unique_solution_eqn (a : ℝ) :
  (∃! x : ℝ, 3^(x^2 + 6 * a * x + 9 * a^2) = a * x^2 + 6 * a^2 * x + 9 * a^3 + a^2 - 4 * a + 4) ↔ (a = 1) :=
by
  sorry

end unique_solution_eqn_l33_33932


namespace solve_inequality_l33_33324

open Set

theorem solve_inequality (x : ℝ) :
  { x | (x^2 - 9) / (x^2 - 16) > 0 } = (Iio (-4)) ∪ (Ioi 4) :=
by
  sorry

end solve_inequality_l33_33324


namespace find_multiplier_l33_33101

variable {a b : ℝ} 

theorem find_multiplier (h1 : 3 * a = x * b) (h2 : a ≠ 0 ∧ b ≠ 0) (h3 : a / 4 = b / 3) : x = 4 :=
by
  sorry

end find_multiplier_l33_33101


namespace probability_heads_at_most_three_out_of_ten_coins_l33_33545

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l33_33545


namespace terminating_decimals_l33_33781

theorem terminating_decimals (n : ℤ) (h : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℤ, k = 20 ∧ ∀ n, 1 ≤ n ∧ n ≤ 180 → (∃ m, m * 180 = n * (2^2 * 5))) := by
  sorry

end terminating_decimals_l33_33781


namespace range_of_a_l33_33967

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x + 2
noncomputable def g (x : ℝ) : ℝ := (Real.exp 1 * Real.log x) / x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, (0 < x1 ∧ x1 ≤ 1 ∧ 0 < x2 ∧ x2 ≤ 1) → f a x1 ≥ g x2) →
  a ≥ -2 :=
sorry

end range_of_a_l33_33967


namespace original_price_four_pack_l33_33649

theorem original_price_four_pack (price_with_rush: ℝ) (increase_rate: ℝ) (num_packs: ℕ):
  price_with_rush = 13 → increase_rate = 0.30 → num_packs = 4 → num_packs * (price_with_rush / (1 + increase_rate)) = 40 :=
by
  intros h_price h_rate h_packs
  rw [h_price, h_rate, h_packs]
  sorry

end original_price_four_pack_l33_33649


namespace slope_of_tangent_line_at_zero_l33_33390

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem slope_of_tangent_line_at_zero : (deriv f 0) = 1 := 
by
  sorry

end slope_of_tangent_line_at_zero_l33_33390


namespace statements_correct_l33_33624

def best_decomposition(n : ℕ) : ℕ × ℕ :=
if h : ∃ (s t : ℕ), s * t = n ∧ s ≤ t ∧ ∀ p q, p * q = n → p ≤ q → |p - q| ≥ |s - t| 
then Classical.choose h else (1, n)

def F (n : ℕ) : ℚ :=
let ⟨p, q⟩ := best_decomposition n in
p / q

theorem statements_correct :
  F 2 = 1/2 ∧
  ¬ (F 24 = 3/8) ∧
  ¬ (F 27 = 3) ∧
  (∀ m : ℕ, F (m^2) = 1) :=
by
  -- Proof goes here
  sorry

end statements_correct_l33_33624


namespace probability_at_most_3_heads_10_coins_l33_33602

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l33_33602


namespace promote_cashback_beneficial_cashback_rubles_preferable_l33_33215

-- For the benefit of banks promoting cashback services
theorem promote_cashback_beneficial :
  ∀ (bank : Type) (services : Type), 
  (∃ (customers : set bank) (businesses : set services), 
    (∀ b ∈ businesses, ∃ d ∈ customers, promotes_cashback d b) ∧
    (∀ c ∈ customers, prefers_cashback c b)) :=
sorry

-- For the preference of customers on cashback in rubles
theorem cashback_rubles_preferable :
  ∀ (customer : Type) (cashback : Type),
  (∀ r ∈ cashback, rubles r → prefers_customer r cashback) :=
sorry

end promote_cashback_beneficial_cashback_rubles_preferable_l33_33215


namespace math_problem_l33_33423

theorem math_problem
  (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (h1 : p * q + r = 47)
  (h2 : q * r + p = 47)
  (h3 : r * p + q = 47) :
  p + q + r = 48 :=
sorry

end math_problem_l33_33423


namespace terminating_decimal_integers_count_l33_33777

/-- For how many integer values of n between 1 and 180 inclusive does the decimal 
representation of n / 180 terminate? We need to calculate the number of integers n such 
that n includes at least 9 as a factor. The answer is 20. -/
theorem terminating_decimal_integers_count :
  { n : ℤ | 1 ≤ n ∧ n ≤ 180 ∧ 9 ∣ n }.to_finset.card = 20 := 
by sorry

end terminating_decimal_integers_count_l33_33777


namespace circleII_area_l33_33054

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

theorem circleII_area (r₁ : ℝ) (h₁ : area_of_circle r₁ = 9) (h₂ : r₂ = 3 * 2 * r₁) : 
  area_of_circle r₂ = 324 :=
by
  sorry

end circleII_area_l33_33054


namespace count_four_digit_numbers_divisible_by_five_ending_45_l33_33414

-- Define the conditions as necessary in Lean
def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_divisible_by_five (n : ℕ) : Prop :=
  n % 5 = 0

def ends_with_45 (n : ℕ) : Prop :=
  n % 100 = 45

-- Statement that there exists 90 such four-digit numbers
theorem count_four_digit_numbers_divisible_by_five_ending_45 : 
  { n : ℕ // is_four_digit_number n ∧ is_divisible_by_five n ∧ ends_with_45 n }.card = 90 :=
sorry

end count_four_digit_numbers_divisible_by_five_ending_45_l33_33414


namespace largest_log_value_l33_33184

theorem largest_log_value :
  ∃ (x y z t : ℝ) (a b c : ℝ),
    x ≤ y ∧ y ≤ z ∧ z ≤ t ∧
    a = Real.log y / Real.log x ∧
    b = Real.log z / Real.log y ∧
    c = Real.log t / Real.log z ∧
    a = 15 ∧ b = 20 ∧ c = 21 ∧
    (∃ u v w, u = a * b ∧ v = b * c ∧ w = a * b * c ∧ w = 420) := sorry

end largest_log_value_l33_33184


namespace parabola_tangents_intersection_y_coord_l33_33294

theorem parabola_tangents_intersection_y_coord
  (a b : ℝ)
  (ha : A = (a, a^2 + 1))
  (hb : B = (b, b^2 + 1))
  (tangent_perpendicular : ∀ t1 t2 : ℝ, t1 * t2 = -1):
  ∃ y : ℝ, y = 3 / 4 :=
by
  sorry

end parabola_tangents_intersection_y_coord_l33_33294


namespace power_function_value_at_minus_two_l33_33662

-- Define the power function assumption and points
variable (f : ℝ → ℝ)
variable (hf : f (1 / 2) = 8)

-- Prove that the given condition implies the required result
theorem power_function_value_at_minus_two : f (-2) = -1 / 8 := 
by {
  -- proof to be filled here
  sorry
}

end power_function_value_at_minus_two_l33_33662


namespace annual_growth_rate_l33_33044

theorem annual_growth_rate (u_2021 u_2023 : ℝ) (x : ℝ) : 
    u_2021 = 1 ∧ u_2023 = 1.69 ∧ x > 0 → (u_2023 / u_2021) = (1 + x)^2 → x * 100 = 30 :=
by
  intros h1 h2
  sorry

end annual_growth_rate_l33_33044


namespace classrooms_student_hamster_difference_l33_33239

-- Define the problem conditions
def students_per_classroom := 22
def hamsters_per_classroom := 3
def number_of_classrooms := 5

-- Define the problem statement
theorem classrooms_student_hamster_difference :
  (students_per_classroom * number_of_classrooms) - 
  (hamsters_per_classroom * number_of_classrooms) = 95 :=
by
  sorry

end classrooms_student_hamster_difference_l33_33239


namespace area_of_right_triangle_l33_33521

-- Given definitions
def leg_a : ℝ := 30
def hypotenuse_c : ℝ := 34

-- The theorem statement
theorem area_of_right_triangle : 
  ∀ (b : ℝ), b = real.sqrt (hypotenuse_c^2 - leg_a^2) → 
  let area := 1 / 2 * leg_a * b in
  area = 240 := 
by
  intro b
  intro h
  let area := 1 / 2 * leg_a * b
  sorry

end area_of_right_triangle_l33_33521


namespace right_triangle_area_l33_33517

/-- Given a right triangle with one leg of length 30 inches and a hypotenuse of 34 inches,
    the area of the triangle is 240 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 :=
by
  rw [h1, h2] at h3
  have hb : b = 16 := by
    rw [←h3]
    norm_num
  rw [h1, hb]
  norm_num
  sorry

end right_triangle_area_l33_33517


namespace roshini_spent_on_sweets_l33_33300

variable (initial_amount friends_amount total_friends_amount sweets_amount : ℝ)

noncomputable def Roshini_conditions (initial_amount friends_amount total_friends_amount sweets_amount : ℝ) :=
  initial_amount = 10.50 ∧ friends_amount = 6.80 ∧ sweets_amount = 3.70 ∧ 2 * 3.40 = 6.80

theorem roshini_spent_on_sweets :
  ∀ (initial_amount friends_amount total_friends_amount sweets_amount : ℝ),
    Roshini_conditions initial_amount friends_amount total_friends_amount sweets_amount →
    initial_amount - friends_amount = sweets_amount :=
by
  intros initial_amount friends_amount total_friends_amount sweets_amount h
  cases h
  sorry

end roshini_spent_on_sweets_l33_33300


namespace race_distance_l33_33815

theorem race_distance {d x y z : ℝ} :
  (d / x = (d - 25) / y) →
  (d / y = (d - 15) / z) →
  (d / x = (d - 37) / z) →
  d = 125 :=
by
  intros h1 h2 h3
  -- Insert proof here
  sorry

end race_distance_l33_33815


namespace min_value_ineq_l33_33086

theorem min_value_ineq (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 3) : 
  (1 / a) + (4 / b) ≥ 3 :=
sorry

end min_value_ineq_l33_33086


namespace linear_function_no_second_quadrant_l33_33954

theorem linear_function_no_second_quadrant (k : ℝ) :
  (∀ x : ℝ, (y : ℝ) → y = k * x - k + 3 → ¬(x < 0 ∧ y > 0)) ↔ k ≥ 3 :=
sorry

end linear_function_no_second_quadrant_l33_33954


namespace max_sum_of_first_n_terms_l33_33655

variable {a : ℕ → ℝ} -- Define sequence a with index ℕ and real values
variable {d : ℝ}      -- Common difference for the arithmetic sequence

-- Conditions and question are formulated into the theorem statement
theorem max_sum_of_first_n_terms (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_diff_neg : d < 0)
  (h_a4_eq_a12 : (a 4)^2 = (a 12)^2) :
  n = 7 ∨ n = 8 := 
sorry

end max_sum_of_first_n_terms_l33_33655


namespace price_of_second_set_of_knives_l33_33144

def john_visits_houses_per_day : ℕ := 50
def percent_buying_per_day : ℝ := 0.20
def price_first_set : ℝ := 50
def weekly_sales : ℝ := 5000
def work_days_per_week : ℕ := 5

theorem price_of_second_set_of_knives
  (john_visits_houses_per_day : ℕ)
  (percent_buying_per_day : ℝ)
  (price_first_set : ℝ)
  (weekly_sales : ℝ)
  (work_days_per_week : ℕ) :
  0 < percent_buying_per_day ∧ percent_buying_per_day ≤ 1 ∧
  weekly_sales = 5000 ∧ 
  work_days_per_week = 5 ∧
  john_visits_houses_per_day = 50 ∧
  price_first_set = 50 → 
  (∃ price_second_set : ℝ, price_second_set = 150) :=
  sorry

end price_of_second_set_of_knives_l33_33144


namespace divide_and_add_l33_33877

variable (number : ℝ)

theorem divide_and_add (h : 4 * number = 166.08) : number / 4 + 0.48 = 10.86 := by
  -- assume the proof follows accurately
  sorry

end divide_and_add_l33_33877


namespace gnomes_and_ponies_l33_33317

theorem gnomes_and_ponies (g p : ℕ) (h1 : g + p = 15) (h2 : 2 * g + 4 * p = 36) : g = 12 ∧ p = 3 :=
by
  sorry

end gnomes_and_ponies_l33_33317


namespace smallest_percent_increase_from_2_to_3_l33_33646

def percent_increase (initial final : ℕ) : ℚ := 
  ((final - initial : ℕ) : ℚ) / (initial : ℕ) * 100

def value_at_question : ℕ → ℕ
| 1 => 100
| 2 => 200
| 3 => 300
| 4 => 500
| 5 => 1000
| 6 => 2000
| 7 => 4000
| 8 => 8000
| 9 => 16000
| 10 => 32000
| 11 => 64000
| 12 => 125000
| 13 => 250000
| 14 => 500000
| 15 => 1000000
| _ => 0  -- Default case for questions out of range

theorem smallest_percent_increase_from_2_to_3 :
  let p1 := percent_increase (value_at_question 1) (value_at_question 2)
  let p2 := percent_increase (value_at_question 2) (value_at_question 3)
  let p3 := percent_increase (value_at_question 3) (value_at_question 4)
  let p11 := percent_increase (value_at_question 11) (value_at_question 12)
  let p14 := percent_increase (value_at_question 14) (value_at_question 15)
  p2 < p1 ∧ p2 < p3 ∧ p2 < p11 ∧ p2 < p14 :=
by
  sorry

end smallest_percent_increase_from_2_to_3_l33_33646


namespace probability_of_target_hit_l33_33663

theorem probability_of_target_hit  :
  let A_hits := 0.9
  let B_hits := 0.8
  ∃ (P_A P_B : ℝ), 
  P_A = A_hits ∧ P_B = B_hits ∧ 
  (∀ events_independent : Prop, 
   events_independent → P_A * P_B = (0.1) * (0.2)) →
  1 - (0.1 * 0.2) = 0.98
:= 
  sorry

end probability_of_target_hit_l33_33663


namespace find_XY_sum_in_base10_l33_33641

def base8_addition_step1 (X : ℕ) : Prop :=
  X + 5 = 9

def base8_addition_step2 (Y X : ℕ) : Prop :=
  Y + 3 = X

theorem find_XY_sum_in_base10 (X Y : ℕ) (h1 : base8_addition_step1 X) (h2 : base8_addition_step2 Y X) :
  X + Y = 5 :=
by
  sorry

end find_XY_sum_in_base10_l33_33641


namespace minimize_material_used_l33_33180

theorem minimize_material_used (r h : ℝ) (V : ℝ) (S : ℝ) 
  (volume_formula : π * r^2 * h = V) (volume_given : V = 27 * π) :
  ∃ r, r = 3 :=
by
  sorry

end minimize_material_used_l33_33180


namespace carnival_rent_l33_33889

-- Define the daily popcorn earnings
def daily_popcorn : ℝ := 50
-- Define the multiplier for cotton candy earnings
def multiplier : ℝ := 3
-- Define the number of operational days
def days : ℕ := 5
-- Define the cost of ingredients
def ingredients_cost : ℝ := 75
-- Define the net earnings after expenses
def net_earnings : ℝ := 895
-- Define the total earnings from selling popcorn for all days
def total_popcorn_earnings : ℝ := daily_popcorn * days
-- Define the total earnings from selling cotton candy for all days
def total_cottoncandy_earnings : ℝ := (daily_popcorn * multiplier) * days
-- Define the total earnings before expenses
def total_earnings : ℝ := total_popcorn_earnings + total_cottoncandy_earnings
-- Define the amount remaining after paying the rent (which includes net earnings and ingredient cost)
def remaining_after_rent : ℝ := net_earnings + ingredients_cost
-- Define the rent
def rent : ℝ := total_earnings - remaining_after_rent

theorem carnival_rent : rent = 30 := by
  sorry

end carnival_rent_l33_33889


namespace determine_x_l33_33814

variable (n p : ℝ)

-- Definitions based on conditions
def x (n : ℝ) : ℝ := 4 * n
def percentage_condition (n p : ℝ) : Prop := 2 * n + 3 = (p / 100) * 25

-- Statement to be proven
theorem determine_x (h : percentage_condition n p) : x n = 4 * n := by
  sorry

end determine_x_l33_33814


namespace terminating_decimal_integers_count_l33_33776

/-- For how many integer values of n between 1 and 180 inclusive does the decimal 
representation of n / 180 terminate? We need to calculate the number of integers n such 
that n includes at least 9 as a factor. The answer is 20. -/
theorem terminating_decimal_integers_count :
  { n : ℤ | 1 ≤ n ∧ n ≤ 180 ∧ 9 ∣ n }.to_finset.card = 20 := 
by sorry

end terminating_decimal_integers_count_l33_33776


namespace reflection_proof_l33_33330

def original_center : (ℝ × ℝ) := (8, -3)
def reflection_line (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, -p.1)
def reflected_center : (ℝ × ℝ) := reflection_line original_center

theorem reflection_proof : reflected_center = (-3, -8) := by
  sorry

end reflection_proof_l33_33330


namespace max_value_exp_l33_33152

theorem max_value_exp (x y z : ℝ) (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) (h_constraint : x^2 + y^2 + z^2 = 1) :
  2 * x * y * Real.sqrt 8 + 7 * y * z + 5 * x * z ≤ 23.0219 :=
sorry

end max_value_exp_l33_33152


namespace probability_at_most_3_heads_10_coins_l33_33606

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l33_33606


namespace rectangle_perimeter_l33_33678

-- Definitions and assumptions
variables (outer_square_area inner_square_area : ℝ) (rectangles_identical : Prop)

-- Given conditions
def outer_square_area_condition : Prop := outer_square_area = 9
def inner_square_area_condition : Prop := inner_square_area = 1
def rectangles_identical_condition : Prop := rectangles_identical

-- The main theorem to prove
theorem rectangle_perimeter (h_outer : outer_square_area_condition outer_square_area)
                            (h_inner : inner_square_area_condition inner_square_area)
                            (h_rectangles : rectangles_identical_condition rectangles_identical) :
  ∃ perimeter : ℝ, perimeter = 6 :=
by
  sorry

end rectangle_perimeter_l33_33678


namespace graph_translation_properties_l33_33002

theorem graph_translation_properties (f g : ℝ → ℝ) (φ : ℝ) (hφ : |φ| < Real.pi / 2) :
  (∀ x, f x = Real.sin (2 * x + φ)) →
  (∀ x, g x = Real.sin (2 * (x + Real.pi / 12) + φ)) →
  (∀ x, g (-x) = g x) →
  (∀ x, x ∈ Ioo (0 : ℝ) (Real.pi / 2) → g x = Real.cos (2 * x) ∧ 
       (g x IS DECREASINGIN ON INTERVAL)) ∧
  (∀ x, g (Real.pi / 2 - x) = g (Real.pi / 2 + x)) :=
by
  sorry

end graph_translation_properties_l33_33002


namespace range_of_m_l33_33103

theorem range_of_m (m : ℝ) :
  (∀ x : ℕ, (x = 1 ∨ x = 2 ∨ x = 3) → (3 * x - m ≤ 0)) ↔ 9 ≤ m ∧ m < 12 :=
by
  sorry

end range_of_m_l33_33103


namespace sample_size_calculation_l33_33750

theorem sample_size_calculation :
  let workshop_A := 120
  let workshop_B := 80
  let workshop_C := 60
  let sample_from_C := 3
  let sampling_fraction := sample_from_C / workshop_C
  let sample_A := workshop_A * sampling_fraction
  let sample_B := workshop_B * sampling_fraction
  let sample_C := workshop_C * sampling_fraction
  let n := sample_A + sample_B + sample_C
  n = 13 :=
by
  sorry

end sample_size_calculation_l33_33750


namespace arithmetic_sequence_a2015_l33_33405

theorem arithmetic_sequence_a2015 :
  ∀ {a : ℕ → ℤ}, (a 1 = 2 ∧ a 5 = 6 ∧ (∀ n, a (n + 1) = a n + a 2 - a 1)) → a 2015 = 2016 :=
by
  sorry

end arithmetic_sequence_a2015_l33_33405


namespace minimize_perimeter_isosceles_l33_33391

noncomputable def inradius (A B C : ℝ) (r : ℝ) : Prop := sorry -- Define inradius

theorem minimize_perimeter_isosceles (A B C : ℝ) (r : ℝ) 
  (h1 : A + B + C = 180) -- Angles sum to 180 degrees
  (h2 : inradius A B C r) -- Given inradius
  (h3 : A = fixed_angle) -- Given fixed angle A
  : B = C :=
by sorry

end minimize_perimeter_isosceles_l33_33391


namespace luke_piles_coins_l33_33693

theorem luke_piles_coins (x : ℕ) (h_total_piles : 10 = 5 + 5) (h_total_coins : 10 * x = 30) :
  x = 3 :=
by
  sorry

end luke_piles_coins_l33_33693


namespace probability_even_and_div_by_three_l33_33304

-- Definitions of the conditions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_div_by_three (n : ℕ) : Prop := n % 3 = 0

-- Definition of die sides
def sides := {1, 2, 3, 4, 5, 6}

-- Probability computation
def probability_even_first_die : ℚ :=
  ↑(sides.filter is_even).card / ↑sides.card

def probability_div_by_three_second_die : ℚ :=
  ↑(sides.filter is_div_by_three).card / ↑sides.card

def combined_probability : ℚ :=
  probability_even_first_die * probability_div_by_three_second_die

-- The main hypothesis to prove
theorem probability_even_and_div_by_three :
  combined_probability = 1/6 :=
by
  sorry

end probability_even_and_div_by_three_l33_33304


namespace right_triangle_area_l33_33527

theorem right_triangle_area (leg1 leg2 hypotenuse : ℕ) (h_leg1 : leg1 = 30)
  (h_hypotenuse : hypotenuse = 34)
  (hypotenuse_sq : hypotenuse * hypotenuse = leg1 * leg1 + leg2 * leg2) :
  (1 / 2 : ℚ) * leg1 * leg2 = 240 := by
  sorry

end right_triangle_area_l33_33527


namespace john_total_fuel_usage_l33_33826

def city_fuel_rate := 6 -- liters per km for city traffic
def highway_fuel_rate := 4 -- liters per km for highway traffic

def trip1_city_distance := 50 -- km for Trip 1
def trip2_highway_distance := 35 -- km for Trip 2
def trip3_city_distance := 15 -- km for Trip 3 in city traffic
def trip3_highway_distance := 10 -- km for Trip 3 on highway

-- Define the total fuel consumption
def total_fuel_used : Nat :=
  (trip1_city_distance * city_fuel_rate) +
  (trip2_highway_distance * highway_fuel_rate) +
  (trip3_city_distance * city_fuel_rate) +
  (trip3_highway_distance * highway_fuel_rate)

theorem john_total_fuel_usage :
  total_fuel_used = 570 :=
by
  sorry

end john_total_fuel_usage_l33_33826


namespace maximum_q_minus_r_l33_33722

theorem maximum_q_minus_r : 
  ∀ q r : ℕ, (1027 = 23 * q + r) ∧ (q > 0) ∧ (r > 0) → q - r ≤ 29 := 
by
  sorry

end maximum_q_minus_r_l33_33722


namespace max_side_length_l33_33887

theorem max_side_length (a b c : ℕ) (h : a + b + c = 30) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_order : a ≤ b ∧ b ≤ c) (h_triangle_ineq : a + b > c) : c ≤ 14 := 
sorry

end max_side_length_l33_33887


namespace ratio_of_candy_bar_to_caramel_l33_33724

noncomputable def price_of_caramel : ℝ := 3
noncomputable def price_of_candy_bar (k : ℝ) : ℝ := k * price_of_caramel
noncomputable def price_of_cotton_candy (C : ℝ) : ℝ := 2 * C 

theorem ratio_of_candy_bar_to_caramel (k : ℝ) (C CC : ℝ) :
  C = price_of_candy_bar k →
  CC = price_of_cotton_candy C →
  6 * C + 3 * price_of_caramel + CC = 57 →
  C / price_of_caramel = 2 :=
by
  sorry

end ratio_of_candy_bar_to_caramel_l33_33724


namespace inequality_solution_set_l33_33173

theorem inequality_solution_set (x : ℝ) : (x-1)/(x+2) > 1 → x < -2 := sorry

end inequality_solution_set_l33_33173


namespace similarity_coefficient_l33_33229

theorem similarity_coefficient (α : ℝ) :
  (2 * Real.sin (α / 2) * (1 - Real.sin (α / 2)))
  = 2 * Real.sin (α / 2) * (1 - Real.sin (α / 2)) :=
sorry

end similarity_coefficient_l33_33229


namespace prob_heads_at_most_3_out_of_10_flips_l33_33554

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l33_33554


namespace sector_max_angle_l33_33941

variables (r l : ℝ)

theorem sector_max_angle (h : 2 * r + l = 40) : (l / r) = 2 :=
sorry

end sector_max_angle_l33_33941


namespace mike_gave_12_pears_l33_33682

variable (P M K N : ℕ)

def initial_pears := 46
def pears_given_to_keith := 47
def pears_left := 11

theorem mike_gave_12_pears (M : ℕ) : 
  initial_pears - pears_given_to_keith + M = pears_left → M = 12 :=
by
  intro h
  sorry

end mike_gave_12_pears_l33_33682


namespace find_m_value_l33_33478

theorem find_m_value (m : ℚ) :
  (m * 2 / 3 + m * 4 / 9 + m * 8 / 27 = 1) → m = 27 / 38 :=
by 
  intro h
  sorry

end find_m_value_l33_33478


namespace trajectory_sufficient_not_necessary_l33_33719

-- Define for any point P if its trajectory is y = |x|
def trajectory (P : ℝ × ℝ) : Prop :=
  P.2 = abs P.1

-- Define for any point P if its distances to the coordinate axes are equal
def equal_distances (P : ℝ × ℝ) : Prop :=
  abs P.1 = abs P.2

-- The main statement: prove that the trajectory is a sufficient but not necessary condition for equal_distances
theorem trajectory_sufficient_not_necessary (P : ℝ × ℝ) :
  trajectory P → equal_distances P ∧ ¬(equal_distances P → trajectory P) := 
sorry

end trajectory_sufficient_not_necessary_l33_33719


namespace largest_angle_of_scalene_triangle_l33_33279

-- Define the problem statement in Lean
theorem largest_angle_of_scalene_triangle (x : ℝ) (hx : x = 30) : 3 * x = 90 :=
by {
  -- Given that the smallest angle is x and x = 30 degrees
  sorry
}

end largest_angle_of_scalene_triangle_l33_33279


namespace photos_last_weekend_45_l33_33296

theorem photos_last_weekend_45 (photos_animals photos_flowers photos_scenery total_photos_this_weekend photos_last_weekend : ℕ)
  (h1 : photos_animals = 10)
  (h2 : photos_flowers = 3 * photos_animals)
  (h3 : photos_scenery = photos_flowers - 10)
  (h4 : total_photos_this_weekend = photos_animals + photos_flowers + photos_scenery)
  (h5 : photos_last_weekend = total_photos_this_weekend - 15) :
  photos_last_weekend = 45 :=
sorry

end photos_last_weekend_45_l33_33296


namespace probability_of_at_most_3_heads_l33_33587

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l33_33587


namespace sum_of_constants_eq_17_l33_33176

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

end sum_of_constants_eq_17_l33_33176


namespace min_x_squared_plus_y_squared_l33_33744

theorem min_x_squared_plus_y_squared (x y : ℝ) (h : (x + 5) * (y - 5) = 0) :
  x^2 + y^2 ≥ 50 :=
by
  sorry

end min_x_squared_plus_y_squared_l33_33744


namespace right_triangle_area_l33_33524

theorem right_triangle_area (leg1 hypotenuse : ℝ) (h1 : leg1 = 30) (h2 : hypotenuse = 34) (h3 : (leg1 ^ 2 + (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) ^ 2 = hypotenuse ^ 2)) :
  (1 / 2) * leg1 * (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) = 240 :=
by
  sorry

end right_triangle_area_l33_33524


namespace correct_answer_l33_33075

theorem correct_answer (m n : ℤ) (h : 3 * m * n + 3 * m = n + 2) : 3 * m + n = -2 := 
by
  sorry

end correct_answer_l33_33075


namespace cubic_inequality_l33_33857

theorem cubic_inequality (x : ℝ) : x^3 - 4*x^2 + 4*x < 0 ↔ x < 0 :=
by
  sorry

end cubic_inequality_l33_33857


namespace find_coefficients_l33_33480

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem find_coefficients 
  (A B Q C P : V) 
  (hQ : Q = (5 / 7 : ℝ) • A + (2 / 7 : ℝ) • B)
  (hC : C = A + 2 • B)
  (hP : P = Q + C) : 
  ∃ s v : ℝ, P = s • A + v • B ∧ s = 12 / 7 ∧ v = 16 / 7 :=
by
  sorry

end find_coefficients_l33_33480


namespace smallest_clock_equiv_to_square_greater_than_10_l33_33975

def clock_equiv (h k : ℕ) : Prop :=
  (h % 12) = (k % 12)

theorem smallest_clock_equiv_to_square_greater_than_10 : ∃ h > 10, clock_equiv h (h * h) ∧ ∀ h' > 10, clock_equiv h' (h' * h') → h ≤ h' :=
by
  sorry

end smallest_clock_equiv_to_square_greater_than_10_l33_33975


namespace number_of_ways_to_buy_three_items_l33_33135

def headphones : ℕ := 9
def mice : ℕ := 13
def keyboards : ℕ := 5
def keyboard_mouse_sets : ℕ := 4
def headphone_mouse_sets : ℕ := 5

theorem number_of_ways_to_buy_three_items : 
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 := 
by 
  sorry

end number_of_ways_to_buy_three_items_l33_33135


namespace distinct_three_digit_numbers_count_l33_33258

theorem distinct_three_digit_numbers_count : 
  (∃ digits : Finset ℕ, 
    digits = {1, 2, 3, 4, 5} ∧ 
    ∀ n ∈ digits, n ∈ {1, 2, 3, 4, 5} ∧ 
    (digits.card = 5)) → 
  card (finset.univ.image (λ p : Finset (ℕ × ℕ × ℕ), 
      {x | x.fst ∈ digits ∧ x.snd ∈ digits ∧ x.snd.snd ∈ digits ∧ x.fst ≠ x.snd ∧ x.snd ≠ x.snd.snd ∧ x.fst ≠ x.snd.snd})) = 60 := by
  sorry

end distinct_three_digit_numbers_count_l33_33258


namespace part1_part2_l33_33363

namespace Problem

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem part1 (m : ℝ) : (B m ⊆ A) → (m ≤ 3) :=
by
  intro h
  sorry

theorem part2 (m : ℝ) : (A ∩ B m = ∅) → (m < 2 ∨ 4 < m) :=
by
  intro h
  sorry

end Problem

end part1_part2_l33_33363


namespace sandra_stickers_l33_33980

theorem sandra_stickers :
  ∃ N : ℕ, N > 1 ∧ (N % 3 = 1) ∧ (N % 5 = 1) ∧ (N % 11 = 1) ∧ N = 166 :=
by {
  sorry
}

end sandra_stickers_l33_33980


namespace cost_of_double_burger_l33_33917

-- Definitions based on conditions
def total_cost : ℝ := 64.50
def total_burgers : ℕ := 50
def single_burger_cost : ℝ := 1.00
def double_burgers : ℕ := 29

-- Proof goal
theorem cost_of_double_burger : (total_cost - single_burger_cost * (total_burgers - double_burgers)) / double_burgers = 1.50 :=
by
  sorry

end cost_of_double_burger_l33_33917


namespace minimum_value_l33_33797

-- Define the geometric sequence and its conditions
variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (positive : ∀ n, 0 < a n)
variable (geometric_seq : ∀ n, a (n+1) = q * a n)
variable (condition1 : a 6 = a 5 + 2 * a 4)
variable (m n : ℕ)
variable (condition2 : ∀ m n, sqrt (a m * a n) = 2 * a 1 → a m = a n)

-- Prove that the minimum value of 1/m + 9/n is 4
theorem minimum_value : m + n = 4 → (∀ x y : ℝ, (0 < x ∧ 0 < y) → (1 / x + 9 / y) ≥ 4) :=
sorry

end minimum_value_l33_33797


namespace terminating_decimals_count_l33_33783

theorem terminating_decimals_count : 
  let N := 180 in
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0) in
  count_terminating_decimals.length = 60 :=
by
  let N := 180
  let count_terminating_decimals := (List.range' 1 N).filter (λ n, n % 3 = 0)
  sorry

end terminating_decimals_count_l33_33783


namespace pairs_of_natural_numbers_l33_33061

theorem pairs_of_natural_numbers (a b : ℕ) (h₁ : b ∣ a + 1) (h₂ : a ∣ b + 1) :
    (a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3) ∨ (a = 2 ∧ b = 1) ∨ (a = 3 ∧ b = 2) :=
by {
  sorry
}

end pairs_of_natural_numbers_l33_33061


namespace triangle_value_a_l33_33102

theorem triangle_value_a (a : ℕ) (h1: a + 2 > 6) (h2: a + 6 > 2) (h3: 2 + 6 > a) : a = 7 :=
sorry

end triangle_value_a_l33_33102


namespace prob_heads_at_most_3_out_of_10_flips_l33_33552

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l33_33552


namespace axis_of_symmetry_parabola_eq_l33_33643

theorem axis_of_symmetry_parabola_eq : ∀ (x y p : ℝ), 
  y = -2 * x^2 → 
  (x^2 = -2 * p * y) → 
  (p = 1/4) →  
  (y = p / 2) → 
  y = 1 / 8 := by 
  intros x y p h1 h2 h3 h4
  sorry

end axis_of_symmetry_parabola_eq_l33_33643


namespace prove_m_eq_n_l33_33687

variable (m n : ℕ)

noncomputable def p := m + n + 1

theorem prove_m_eq_n 
  (is_prime : Prime p) 
  (divides : p ∣ 2 * (m^2 + n^2) - 1) : 
  m = n :=
by
  sorry

end prove_m_eq_n_l33_33687


namespace problem_l33_33674

theorem problem (h : (0.00027 : ℝ) = 27 / 100000) : (10^5 - 10^3) * 0.00027 = 26.73 := by
  sorry

end problem_l33_33674


namespace sufficient_but_not_necessary_l33_33660

theorem sufficient_but_not_necessary (x : ℝ) : (x = -1 → x^2 - 5 * x - 6 = 0) ∧ (∃ y : ℝ, y ≠ -1 ∧ y^2 - 5 * y - 6 = 0) :=
by
  sorry

end sufficient_but_not_necessary_l33_33660


namespace range_of_a_l33_33396

def p (a : ℝ) : Prop := ∀ k : ℝ, ∃ x y : ℝ, (y = k * x + 1) ∧ (x^2 + (y^2) / a = 1)
def q (a : ℝ) : Prop := ∃ x0 : ℝ, 4^x0 - 2^x0 - a ≤ 0

theorem range_of_a (a : ℝ) : ¬(p a ∧ q a) ∧ (p a ∨ q a) → -1/4 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l33_33396


namespace second_number_added_is_5_l33_33868

theorem second_number_added_is_5
  (x : ℕ) (h₁ : x = 3)
  (y : ℕ)
  (h₂ : (x + 1) * (x + 13) = (x + y) * (x + y)) :
  y = 5 :=
sorry

end second_number_added_is_5_l33_33868


namespace figure_F10_squares_l33_33632

def num_squares (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 1 + 3 * (n - 1) * n

theorem figure_F10_squares : num_squares 10 = 271 :=
by sorry

end figure_F10_squares_l33_33632


namespace train_crossing_time_l33_33615

noncomputable def time_to_cross_platform
  (speed_kmph : ℝ)
  (length_train : ℝ)
  (length_platform : ℝ) : ℝ :=
  let speed_ms := speed_kmph / 3.6
  let total_distance := length_train + length_platform
  total_distance / speed_ms

theorem train_crossing_time
  (speed_kmph : ℝ)
  (length_train : ℝ)
  (length_platform : ℝ)
  (h_speed : speed_kmph = 72)
  (h_train_length : length_train = 280.0416)
  (h_platform_length : length_platform = 240) :
  time_to_cross_platform speed_kmph length_train length_platform = 26.00208 := by
  sorry

end train_crossing_time_l33_33615


namespace probability_at_most_3_heads_l33_33580

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l33_33580


namespace triangle_inequality_a2_lt_ab_ac_l33_33076

theorem triangle_inequality_a2_lt_ab_ac {a b c : ℝ} (h1 : a < b + c) (h2 : 0 < a) : a^2 < a * b + a * c := 
sorry

end triangle_inequality_a2_lt_ab_ac_l33_33076


namespace john_bought_3_reels_l33_33825

theorem john_bought_3_reels (reel_length section_length : ℕ) (n_sections : ℕ)
  (h1 : reel_length = 100) (h2 : section_length = 10) (h3 : n_sections = 30) :
  n_sections * section_length / reel_length = 3 :=
by
  sorry

end john_bought_3_reels_l33_33825


namespace functional_equation_solution_l33_33217

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, (f x + y) * (f (x - y) + 1) = f (f (x * f (x + 1)) - y * f (y - 1))) → (∀ x : ℝ, f x = x) :=
by
  intros f h x
  -- Proof would go here
  sorry

end functional_equation_solution_l33_33217


namespace exercise_l33_33491

theorem exercise (n : ℕ) : 900 ∣ (6 ^ (2 * (n + 1)) - 2 ^ (n + 3) * 3 ^ (n + 2) + 36) :=
sorry

end exercise_l33_33491


namespace length_of_one_pencil_l33_33140

theorem length_of_one_pencil (l : ℕ) (h1 : 2 * l = 24) : l = 12 :=
by {
  sorry
}

end length_of_one_pencil_l33_33140


namespace ticket_price_divisor_l33_33894

def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

def GCD (a b : ℕ) := Nat.gcd a b

theorem ticket_price_divisor :
  let total7 := 70
  let total8 := 98
  let y := 4
  is_divisor (GCD total7 total8) y :=
by
  sorry

end ticket_price_divisor_l33_33894


namespace ellipse_eq_find_k_l33_33080

open Real

-- Part I: Prove the equation of the ellipse
theorem ellipse_eq (a b : ℝ) (h_a : a = 2) (h_b : b = 1) :
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 = 1 :=
by
  sorry

-- Part II: Prove the value of k
theorem find_k (P : ℝ × ℝ) (hP : P = (-2, 1)) (MN_length : ℝ) (hMN : MN_length = 2) 
  (k : ℝ) (a b : ℝ) (h_a : a = 2) (h_b : b = 1) :
  ∃ k : ℝ, k = -4 :=
by
  sorry

end ellipse_eq_find_k_l33_33080


namespace buy_items_ways_l33_33123

theorem buy_items_ways (headphones keyboards mice keyboard_mouse_sets headphone_mouse_sets : ℕ) :
  headphones = 9 → keyboards = 5 → mice = 13 → keyboard_mouse_sets = 4 → headphone_mouse_sets = 5 →
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  intros h_eq k_eq m_eq kms_eq hms_eq
  have h_eq_gen : headphones = 9 := h_eq
  have k_eq_gen : keyboards = 5 := k_eq
  have m_eq_gen : mice = 13 := m_eq
  have kms_eq_gen : keyboard_mouse_sets = 4 := kms_eq
  have hms_eq_gen : headphone_mouse_sets = 5 := hms_eq
  sorry

end buy_items_ways_l33_33123


namespace find_m_value_l33_33407

theorem find_m_value (x: ℝ) (m: ℝ) (hx: x > 2) (hm: m > 0) (h_min: ∀ y, (y = x + m / (x - 2)) → y ≥ 6) : m = 4 := 
sorry

end find_m_value_l33_33407


namespace evaluation_at_x_4_l33_33494

noncomputable def simplified_expression (x : ℝ) :=
  (x - 1 - (3 / (x + 1))) / ((x^2 + 2 * x) / (x + 1))

theorem evaluation_at_x_4 : simplified_expression 4 = 1 / 2 :=
by
  sorry

end evaluation_at_x_4_l33_33494


namespace probability_10_coins_at_most_3_heads_l33_33566

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l33_33566


namespace problem_1_problem_2_l33_33362

open Real

theorem problem_1
  (a b m n : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hm : m > 0)
  (hn : n > 0) :
  (m ^ 2 / a + n ^ 2 / b) ≥ ((m + n) ^ 2 / (a + b)) :=
sorry

theorem problem_2
  (x : ℝ)
  (hx1 : 0 < x)
  (hx2 : x < 1 / 2) :
  (2 / x + 9 / (1 - 2 * x)) ≥ 25 ∧ (2 / x + 9 / (1 - 2 * x)) = 25 ↔ x = 1 / 5 :=
sorry

end problem_1_problem_2_l33_33362


namespace derivative_at_zero_l33_33090

-- Define the function f(x)
def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

-- State the theorem with the given conditions and expected result
theorem derivative_at_zero :
  (deriv f 0 = -120) :=
by
  sorry

end derivative_at_zero_l33_33090


namespace sufficient_condition_parallel_planes_l33_33246

-- Definitions for lines and planes
variable {Line Plane : Type}
variable {m n : Line}
variable {α β : Plane}

-- Relations between lines and planes
variable (parallel_line : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Condition for sufficient condition for α parallel β
theorem sufficient_condition_parallel_planes
  (h1 : parallel_line m n)
  (h2 : perpendicular_line_plane m α)
  (h3 : perpendicular_line_plane n β) :
  parallel_plane α β :=
sorry

end sufficient_condition_parallel_planes_l33_33246


namespace right_triangles_with_leg_2012_l33_33464

theorem right_triangles_with_leg_2012 :
  ∀ (a b c : ℕ), a = 2012 ∧ a ^ 2 + b ^ 2 = c ^ 2 → 
  (b = 253005 ∧ c = 253013) ∨ 
  (b = 506016 ∧ c = 506020) ∨ 
  (b = 1012035 ∧ c = 1012037) ∨ 
  (b = 1509 ∧ c = 2515) :=
by
  intros
  sorry

end right_triangles_with_leg_2012_l33_33464


namespace arithmetic_sequence_n_equals_8_l33_33078

theorem arithmetic_sequence_n_equals_8
  (a : ℕ → ℝ) 
  (h_arith : ∀ n m, a (n + 1) - a n = a (m + 1) - a m) 
  (h2 : a 2 + a 5 = 18)
  (h3 : a 3 * a 4 = 32)
  (h_n : ∃ n, a n = 128) :
  ∃ n, a n = 128 ∧ n = 8 := 
sorry

end arithmetic_sequence_n_equals_8_l33_33078


namespace car_owners_without_motorcycle_or_bicycle_l33_33869

noncomputable def total_adults := 500
noncomputable def car_owners := 400
noncomputable def motorcycle_owners := 200
noncomputable def bicycle_owners := 150
noncomputable def car_motorcycle_owners := 100
noncomputable def motorcycle_bicycle_owners := 50
noncomputable def car_bicycle_owners := 30

theorem car_owners_without_motorcycle_or_bicycle :
  car_owners - car_motorcycle_owners - car_bicycle_owners = 270 := by
  sorry

end car_owners_without_motorcycle_or_bicycle_l33_33869


namespace problem_statement_l33_33409

noncomputable def verify_ratio (x y c d : ℝ) (h1 : 4 * x - 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) : Prop :=
  c / d = -1/3

theorem problem_statement (x y c d : ℝ) (h1 : 4 * x - 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) : verify_ratio x y c d h1 h2 h3 :=
  sorry

end problem_statement_l33_33409


namespace value_of_x_if_additive_inverses_l33_33263

theorem value_of_x_if_additive_inverses (x : ℝ) 
  (h : 4 * x - 1 + (3 * x - 6) = 0) : x = 1 := by
sorry

end value_of_x_if_additive_inverses_l33_33263


namespace original_price_lamp_l33_33223

theorem original_price_lamp
  (P : ℝ)
  (discount_rate : ℝ)
  (discounted_price : ℝ)
  (discount_is_20_perc : discount_rate = 0.20)
  (new_price_is_96 : discounted_price = 96)
  (price_after_discount : discounted_price = P * (1 - discount_rate)) :
  P = 120 :=
by
  sorry

end original_price_lamp_l33_33223


namespace probability_at_most_3_heads_10_flips_l33_33593

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l33_33593


namespace compare_squares_l33_33939

theorem compare_squares (a b : ℝ) : a^2 + b^2 ≥ ab + a + b - 1 :=
by
  sorry

end compare_squares_l33_33939


namespace range_of_a_l33_33096

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (x ∈ {x : ℝ | x ≥ 3 ∨ x ≤ -1} ∩ {x : ℝ | x ≤ a} ↔ x ∈ {x : ℝ | x ≤ a})) ↔ a ≤ -1 :=
by sorry

end range_of_a_l33_33096


namespace terminating_decimal_fraction_count_l33_33778

theorem terminating_decimal_fraction_count :
  (finset.filter (λ n : ℕ, 1 ≤ n ∧ n ≤ 180 ∧ 90 ∣ n) (finset.range 181)).card = 20 :=
by sorry

end terminating_decimal_fraction_count_l33_33778


namespace combined_salaries_of_A_B_C_D_l33_33504

theorem combined_salaries_of_A_B_C_D (salaryE : ℕ) (avg_salary : ℕ) (num_people : ℕ)
    (h1 : salaryE = 9000) (h2 : avg_salary = 8800) (h3 : num_people = 5) :
    (avg_salary * num_people) - salaryE = 35000 :=
by
  sorry

end combined_salaries_of_A_B_C_D_l33_33504


namespace trivia_team_l33_33997

theorem trivia_team (total_students groups students_per_group students_not_picked : ℕ) (h1 : total_students = 65)
  (h2 : groups = 8) (h3 : students_per_group = 6) (h4 : students_not_picked = total_students - groups * students_per_group) :
  students_not_picked = 17 :=
sorry

end trivia_team_l33_33997


namespace number_line_point_B_l33_33666

theorem number_line_point_B (A B : ℝ) (AB : ℝ) (h1 : AB = 4 * Real.sqrt 2) (h2 : A = 3 * Real.sqrt 2) :
  B = -Real.sqrt 2 ∨ B = 7 * Real.sqrt 2 :=
sorry

end number_line_point_B_l33_33666


namespace base8_subtraction_correct_l33_33067

theorem base8_subtraction_correct : (453 - 326 : ℕ) = 125 :=
by sorry

end base8_subtraction_correct_l33_33067


namespace part1_part2_l33_33665

open Real

noncomputable def f (x a : ℝ) : ℝ := x * exp (a * x) + x * cos x + 1

theorem part1 (x : ℝ) (hx : 0 ≤ x) : cos x ≥ 1 - (1 / 2) * x^2 := 
sorry

theorem part2 (a x : ℝ) (ha : 1 ≤ a) (hx : 0 ≤ x) : f x a ≥ (1 + sin x)^2 := 
sorry

end part1_part2_l33_33665


namespace gcd_lcm_identity_l33_33688

variables {n m k : ℕ}

/-- Given positive integers n, m, and k such that n divides lcm(m, k) 
    and m divides lcm(n, k), we prove that n * gcd(m, k) = m * gcd(n, k). -/
theorem gcd_lcm_identity (n_pos : 0 < n) (m_pos : 0 < m) (k_pos : 0 < k) 
  (h1 : n ∣ Nat.lcm m k) (h2 : m ∣ Nat.lcm n k) :
  n * Nat.gcd m k = m * Nat.gcd n k :=
sorry

end gcd_lcm_identity_l33_33688


namespace one_fourth_of_eight_point_four_l33_33051

theorem one_fourth_of_eight_point_four : (8.4 / 4) = (21 / 10) :=
by
  -- The expected proof would go here
  sorry

end one_fourth_of_eight_point_four_l33_33051


namespace total_ways_to_buy_l33_33124

-- Definitions:
def h : ℕ := 9  -- number of headphones
def m : ℕ := 13 -- number of mice
def k : ℕ := 5  -- number of keyboards
def s_km : ℕ := 4 -- number of sets of keyboard and mouse
def s_hm : ℕ := 5 -- number of sets of headphones and mouse

-- Theorem stating the number of ways to buy three items is 646.
theorem total_ways_to_buy : (s_km * h) + (s_hm * k) + (h * m * k) = 646 := by
  -- Placeholder proof using sorry.
  sorry

end total_ways_to_buy_l33_33124


namespace right_triangle_area_l33_33519

/-- Given a right triangle with one leg of length 30 inches and a hypotenuse of 34 inches,
    the area of the triangle is 240 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 :=
by
  rw [h1, h2] at h3
  have hb : b = 16 := by
    rw [←h3]
    norm_num
  rw [h1, hb]
  norm_num
  sorry

end right_triangle_area_l33_33519


namespace fair_coin_flip_difference_l33_33203

theorem fair_coin_flip_difference :
  let P1 := 5 * (1 / 2)^5;
  let P2 := (1 / 2)^5;
  P1 - P2 = 9 / 32 :=
by
  sorry

end fair_coin_flip_difference_l33_33203


namespace cloak_change_in_silver_l33_33459

theorem cloak_change_in_silver :
  (∀ c : ℤ, (20 = c + 4) → (15 = c + 1)) →
  (5 * g = 3) →
  14 * gold / exchange_rate = 10 := 
sorry

end cloak_change_in_silver_l33_33459


namespace only_n_divides_2_to_n_minus_1_l33_33772

theorem only_n_divides_2_to_n_minus_1 (n : ℕ) (h1 : n > 0) : n ∣ (2^n - 1) ↔ n = 1 :=
by
  sorry

end only_n_divides_2_to_n_minus_1_l33_33772


namespace externally_tangent_internally_tangent_common_chord_and_length_l33_33410

-- Definitions of Circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y - 1 = 0
def circle2 (x y : ℝ) (m : ℝ) : Prop := x^2 + y^2 - 10*x - 12*y + m = 0

-- Proof problem 1: Externally tangent
theorem externally_tangent (m : ℝ) : (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) → m = 25 + 10 * Real.sqrt 11 :=
sorry

-- Proof problem 2: Internally tangent
theorem internally_tangent (m : ℝ) : (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) → m = 25 - 10 * Real.sqrt 11 :=
sorry

-- Proof problem 3: Common chord and length when m = 45
theorem common_chord_and_length :
  (∃ x y : ℝ, circle2 x y 45) →
  (∃ l : ℝ, l = 4 * Real.sqrt 7 ∧ ∀ x y : ℝ, (circle1 x y ∧ circle2 x y 45) → (4*x + 3*y - 23 = 0)) :=
sorry

end externally_tangent_internally_tangent_common_chord_and_length_l33_33410


namespace carmen_more_miles_l33_33334

-- Definitions for the conditions
def carmen_distance : ℕ := 90
def daniel_distance : ℕ := 75

-- The theorem statement
theorem carmen_more_miles : carmen_distance - daniel_distance = 15 :=
by
  sorry

end carmen_more_miles_l33_33334


namespace max_M_min_N_l33_33653

noncomputable def M (x y : ℝ) : ℝ := x / (2 * x + y) + y / (x + 2 * y)
noncomputable def N (x y : ℝ) : ℝ := x / (x + 2 * y) + y / (2 * x + y)

theorem max_M_min_N (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∃ t : ℝ, (∀ x y, 0 < x → 0 < y → M x y ≤ t) ∧ (∀ x y, 0 < x → 0 < y → N x y ≥ t) ∧ t = 2 / 3) :=
sorry

end max_M_min_N_l33_33653


namespace point_of_tangency_of_circles_l33_33512

/--
Given two circles defined by the following equations:
1. \( x^2 - 2x + y^2 - 10y + 17 = 0 \)
2. \( x^2 - 8x + y^2 - 10y + 49 = 0 \)
Prove that the coordinates of the point of tangency of these circles are \( (2.5, 5) \).
-/
theorem point_of_tangency_of_circles :
  (∃ x y : ℝ, (x^2 - 2*x + y^2 - 10*y + 17 = 0) ∧ (x = 2.5) ∧ (y = 5)) ∧ 
  (∃ x' y' : ℝ, (x'^2 - 8*x' + y'^2 - 10*y' + 49 = 0) ∧ (x' = 2.5) ∧ (y' = 5)) :=
sorry

end point_of_tangency_of_circles_l33_33512


namespace circle_cut_by_parabolas_l33_33681

theorem circle_cut_by_parabolas (n : ℕ) (h : n = 10) : 
  2 * n ^ 2 + 1 = 201 :=
by
  sorry

end circle_cut_by_parabolas_l33_33681


namespace preston_charges_5_dollars_l33_33699

def cost_per_sandwich (x : Real) : Prop :=
  let number_of_sandwiches := 18
  let delivery_fee := 20
  let tip_percentage := 0.10
  let total_received := 121
  let total_cost := number_of_sandwiches * x + delivery_fee
  let tip := tip_percentage * total_cost
  let final_amount := total_cost + tip
  final_amount = total_received

theorem preston_charges_5_dollars :
  ∀ x : Real, cost_per_sandwich x → x = 5 :=
by
  intros x h
  sorry

end preston_charges_5_dollars_l33_33699


namespace find_x_from_arithmetic_mean_l33_33712

theorem find_x_from_arithmetic_mean (x : ℝ) 
  (h : (x + 10 + 18 + 3 * x + 16 + (x + 5) + (3 * x + 6)) / 6 = 25) : 
  x = 95 / 8 := by
  sorry

end find_x_from_arithmetic_mean_l33_33712


namespace cuberoot_sum_eq_l33_33208

theorem cuberoot_sum_eq (a : ℝ) (h : a = 2^7 + 2^7 + 2^7) : 
  real.cbrt a = 4 * real.cbrt 6 :=
by {
  sorry
}

end cuberoot_sum_eq_l33_33208


namespace exists_n_consecutive_composite_l33_33700

theorem exists_n_consecutive_composite (n : ℕ) :
  ∃ (seq : Fin (n+1) → ℕ), (∀ i, seq i > 1 ∧ ¬ Nat.Prime (seq i)) ∧
                          ∀ k, seq k = (n+1)! + (2 + k) := 
by
  sorry

end exists_n_consecutive_composite_l33_33700


namespace tina_wins_more_than_losses_l33_33871

theorem tina_wins_more_than_losses 
  (initial_wins : ℕ)
  (additional_wins : ℕ)
  (first_loss : ℕ)
  (doubled_wins : ℕ)
  (second_loss : ℕ)
  (total_wins : ℕ)
  (total_losses : ℕ)
  (final_difference : ℕ) :
  initial_wins = 10 →
  additional_wins = 5 →
  first_loss = 1 →
  doubled_wins = 30 →
  second_loss = 1 →
  total_wins = initial_wins + additional_wins + doubled_wins →
  total_losses = first_loss + second_loss →
  final_difference = total_wins - total_losses →
  final_difference = 43 :=
by
  sorry

end tina_wins_more_than_losses_l33_33871


namespace solve_for_a_l33_33946

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a * x

theorem solve_for_a : 
  (∃ a : ℝ, f (f 0 a) a = 4 * a) → (a = 2) :=
by
  sorry

end solve_for_a_l33_33946


namespace negation_of_statement_l33_33344

theorem negation_of_statement :
  ¬(∀ x : ℝ, ∃ n : ℕ, 0 < n ∧ n > x^2) ↔ (∃ x : ℝ, ∀ n : ℕ, 0 < n → n < x^2) := by
sorry

end negation_of_statement_l33_33344


namespace x_add_y_add_one_is_composite_l33_33864

theorem x_add_y_add_one_is_composite (x y : ℕ) (hx : x > 1) (hy : y > 1) 
  (k : ℕ) (h : x^2 + x * y - y = k^2) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (x + y + 1 = a * b) :=
by
  sorry

end x_add_y_add_one_is_composite_l33_33864


namespace maximum_value_expression_l33_33151

noncomputable def max_expr_value (x θ : ℝ) (h1 : 0 < x) (h2 : 0 ≤ θ) (h3 : θ ≤ π) : ℝ :=
  2 * Real.sqrt 2 - 2

theorem maximum_value_expression :
  ∀ (x θ : ℝ), 0 < x → 0 ≤ θ → θ ≤ π →
  ( ∃ (M : ℝ), (∀ (x' θ' : ℝ), 0 < x' → 0 ≤ θ' → θ' ≤ π → 
    (x' ^ 2 + 2 - Real.sqrt (x' ^ 4 + 4 * (Real.sin θ') ^ 2)) / x' ≤ M)
     ∧ M = 2 * Real.sqrt 2 - 2) :=
by 
  intros x θ h1 h2 h3
  use max_expr_value x θ h1 h2 h3
  split
  · sorry  -- Prove that the expression (x^2 + 2 - sqrt(x^4 + 4*sin^2 θ)) / x has an upper bound M
  · refl   -- Prove M = 2 * sqrt 2 - 2

end maximum_value_expression_l33_33151


namespace women_science_majors_is_30_percent_l33_33439

noncomputable def percentage_women_science_majors (ns_percent : ℝ) (m_percent : ℝ) (m_sci_percent : ℝ) : ℝ :=
  let w_percent := 1 - m_percent
  let m_sci_total := m_percent * m_sci_percent
  let total_sci := 1 - ns_percent
  let w_sci_total := total_sci - m_sci_total
  (w_sci_total / w_percent) * 100

theorem women_science_majors_is_30_percent :
  percentage_women_science_majors 0.60 0.40 0.55 = 30 := by
  sorry

end women_science_majors_is_30_percent_l33_33439


namespace terminating_decimals_count_l33_33787

theorem terminating_decimals_count :
  ∃ (count : ℕ), count = 60 ∧
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 180 →
  (∃ m : ℕ, m * 180 = n * (2^2 * 5)) → 60 = count := sorry

end terminating_decimals_count_l33_33787


namespace parcel_total_weight_l33_33711

theorem parcel_total_weight (x y z : ℝ) 
  (h1 : x + y = 132) 
  (h2 : y + z = 146) 
  (h3 : z + x = 140) : 
  x + y + z = 209 :=
by
  sorry

end parcel_total_weight_l33_33711


namespace probability_at_most_3_heads_l33_33579

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l33_33579


namespace total_books_correct_l33_33473

-- Define the number of books each person has
def booksKeith : Nat := 20
def booksJason : Nat := 21
def booksMegan : Nat := 15

-- Define the total number of books they have together
def totalBooks : Nat := booksKeith + booksJason + booksMegan

-- Prove that the total number of books is 56
theorem total_books_correct : totalBooks = 56 := by
  sorry

end total_books_correct_l33_33473


namespace factor_polynomial_l33_33970

theorem factor_polynomial (n : ℕ) (hn : 2 ≤ n) 
  (a : ℝ) (b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ k : ℤ, n < 2 * k + 1 ∧ 2 * k + 1 < 3 * n ∧ 
  a = (-(2 * Real.cos ((2 * k + 1 : ℝ) * Real.pi / (2 * n)))) ^ (2 * n / (2 * n - 1)) ∧ 
  b = (2 * Real.cos ((2 * k + 1 : ℝ) * Real.pi / (2 * n))) ^ (2 / (2 * n - 1)) := sorry

end factor_polynomial_l33_33970


namespace chang_apple_problem_l33_33630

theorem chang_apple_problem 
  (A : ℝ)
  (h1 : 0.50 * A * 0.50 + 0.25 * A * 0.10 + 0.15 * A * 0.30 + 0.10 * A * 0.20 = 80)
  : A = 235 := 
sorry

end chang_apple_problem_l33_33630


namespace circle_radius_9_l33_33647

theorem circle_radius_9 (k : ℝ) : 
  (∀ x y : ℝ, 2 * x^2 + 20 * x + 3 * y^2 + 18 * y - k = 81) → 
  (k = 94) :=
by
  sorry

end circle_radius_9_l33_33647


namespace probability_of_at_most_3_heads_l33_33571

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l33_33571


namespace silver_coins_change_l33_33441

-- Define the conditions
def condition1 : ℕ × ℕ := (20, 4) -- (20 silver coins, 4 gold coins change)
def condition2 : ℕ × ℕ := (15, 1) -- (15 silver coins, 1 gold coin change)
def cost_of_cloak_in_gold_coins : ℕ := 14

-- Define the theorem to be proven
theorem silver_coins_change (s1 g1 s2 g2 cloak_g : ℕ) (h1 : (s1, g1) = condition1) (h2 : (s2, g2) = condition2) :
  ∃ silver : ℕ, (silver = 10) :=
by {
  sorry
}

end silver_coins_change_l33_33441


namespace josh_total_money_l33_33685

-- Define the initial conditions
def initial_wallet : ℝ := 300
def initial_investment : ℝ := 2000
def stock_increase_rate : ℝ := 0.30

-- The expected total amount Josh will have after selling his stocks
def expected_total_amount : ℝ := 2900

-- Define the problem: that the total money in Josh's wallet after selling all stocks equals $2900
theorem josh_total_money :
  let increased_value := initial_investment * stock_increase_rate
  let new_investment := initial_investment + increased_value
  let total_money := new_investment + initial_wallet
  total_money = expected_total_amount :=
by
  sorry

end josh_total_money_l33_33685


namespace rainfall_november_is_180_l33_33278

-- Defining the conditions
def daily_rainfall_first_15_days := 4 -- inches per day
def days_in_first_period := 15
def total_days_in_november := 30
def multiplier_for_second_period := 2

-- Calculation based on the problem's conditions
def total_rainfall_november := 
  (daily_rainfall_first_15_days * days_in_first_period) + 
  (multiplier_for_second_period * daily_rainfall_first_15_days * (total_days_in_november - days_in_first_period))

-- Prove that the total rainfall in November is 180 inches
theorem rainfall_november_is_180 : total_rainfall_november = 180 :=
by
  -- Proof steps (to be filled in)
  sorry

end rainfall_november_is_180_l33_33278


namespace max_m_value_l33_33087

theorem max_m_value (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) : 
  (∀ a b : ℝ, a > 0 → b > 0 → ((2 / a) + (1 / b) ≥ (m / (2 * a + b)))) → m ≤ 9 :=
sorry

end max_m_value_l33_33087


namespace terminating_decimal_count_l33_33790

theorem terminating_decimal_count : 
  (Finset.filter (λ n, n % 3 = 0) (Finset.range 181)).card = 60 := by
  sorry

end terminating_decimal_count_l33_33790


namespace compare_probabilities_l33_33046

noncomputable def box_bad_coin_prob_method_one : ℝ := 1 - (0.99 ^ 10)
noncomputable def box_bad_coin_prob_method_two : ℝ := 1 - ((49 / 50) ^ 5)

theorem compare_probabilities : box_bad_coin_prob_method_one < box_bad_coin_prob_method_two := by
  sorry

end compare_probabilities_l33_33046


namespace inequality_holds_for_all_reals_l33_33065

theorem inequality_holds_for_all_reals (x : ℝ) : 
  7 / 20 + |3 * x - 2 / 5| ≥ 1 / 4 :=
sorry

end inequality_holds_for_all_reals_l33_33065


namespace confirm_per_capita_volumes_l33_33539

noncomputable def west_volume_per_capita := 21428
noncomputable def non_west_volume_per_capita := 26848.55
noncomputable def russia_volume_per_capita := 302790.13

theorem confirm_per_capita_volumes :
  west_volume_per_capita = 21428 ∧
  non_west_volume_per_capita = 26848.55 ∧
  russia_volume_per_capita = 302790.13 :=
by
  split; sorry
  split; sorry
  sorry

end confirm_per_capita_volumes_l33_33539


namespace largest_among_a_b_c_l33_33395

theorem largest_among_a_b_c (x : ℝ) (h0 : 0 < x) (h1 : x < 1)
  (a : ℝ := 2 * Real.sqrt x) 
  (b : ℝ := 1 + x) 
  (c : ℝ := 1 / (1 - x)) : c > b ∧ b > a := by
  sorry

end largest_among_a_b_c_l33_33395


namespace calculate_rolls_of_toilet_paper_l33_33640

-- Definitions based on the problem conditions
def seconds_per_egg := 15
def minutes_per_roll := 30
def total_cleaning_minutes := 225
def number_of_eggs := 60
def time_per_minute := 60

-- Calculation of the time spent on eggs in minutes
def egg_cleaning_minutes := (number_of_eggs * seconds_per_egg) / time_per_minute

-- Total cleaning time minus time spent on eggs
def remaining_cleaning_minutes := total_cleaning_minutes - egg_cleaning_minutes

-- Verify the number of rolls of toilet paper cleaned up
def rolls_of_toilet_paper := remaining_cleaning_minutes / minutes_per_roll

-- Theorem statement to be proved
theorem calculate_rolls_of_toilet_paper : rolls_of_toilet_paper = 7 := by
  sorry

end calculate_rolls_of_toilet_paper_l33_33640


namespace cow_calf_ratio_l33_33047

theorem cow_calf_ratio (cost_cow cost_calf : ℕ) (h_cow : cost_cow = 880) (h_calf : cost_calf = 110) :
  cost_cow / cost_calf = 8 :=
by {
  sorry
}

end cow_calf_ratio_l33_33047


namespace prove_difference_l33_33348

theorem prove_difference (x y : ℝ) (h1 : x + y = 500) (h2 : x * y = 22000) : y - x = -402.5 :=
sorry

end prove_difference_l33_33348


namespace largest_integral_x_l33_33386

theorem largest_integral_x (x y : ℤ) (h1 : (1 : ℚ)/4 < x/7) (h2 : x/7 < (2 : ℚ)/3) (h3 : x + y = 10) : x = 4 :=
by
  sorry

end largest_integral_x_l33_33386


namespace remainder_of_sum_div_18_l33_33025

theorem remainder_of_sum_div_18 :
  let nums := [11065, 11067, 11069, 11071, 11073, 11075, 11077, 11079, 11081]
  let residues := [1, 3, 5, 7, 9, 11, 13, 15, 17]
  (nums.sum % 18) = 9 := by
    sorry

end remainder_of_sum_div_18_l33_33025


namespace greendale_points_l33_33314

theorem greendale_points : 
  let roosevelt_game1 := 30 
  let roosevelt_game2 := roosevelt_game1 / 2
  let roosevelt_game3 := roosevelt_game2 * 3
  let roosevelt_bonus := 50
  let greendale_diff := 10
  let roosevelt_total := roosevelt_game1 + roosevelt_game2 + roosevelt_game3 + roosevelt_bonus
  let greendale_total := roosevelt_total - greendale_diff
  greendale_total = 130 :=
by
  sorry

end greendale_points_l33_33314


namespace range_of_a_l33_33811

def tangent_perpendicular_to_y_axis (a : ℝ) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ (3 * a * x^2 + 1 / x = 0)

theorem range_of_a : {a : ℝ | tangent_perpendicular_to_y_axis a} = {a : ℝ | a < 0} :=
by
  sorry

end range_of_a_l33_33811


namespace find_number_of_people_l33_33892

def number_of_people (total_shoes : Nat) (shoes_per_person : Nat) : Nat :=
  total_shoes / shoes_per_person

theorem find_number_of_people :
  number_of_people 20 2 = 10 := 
by
  sorry

end find_number_of_people_l33_33892


namespace probability_at_most_3_heads_10_coins_l33_33603

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l33_33603


namespace buy_items_ways_l33_33122

theorem buy_items_ways (headphones keyboards mice keyboard_mouse_sets headphone_mouse_sets : ℕ) :
  headphones = 9 → keyboards = 5 → mice = 13 → keyboard_mouse_sets = 4 → headphone_mouse_sets = 5 →
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  intros h_eq k_eq m_eq kms_eq hms_eq
  have h_eq_gen : headphones = 9 := h_eq
  have k_eq_gen : keyboards = 5 := k_eq
  have m_eq_gen : mice = 13 := m_eq
  have kms_eq_gen : keyboard_mouse_sets = 4 := kms_eq
  have hms_eq_gen : headphone_mouse_sets = 5 := hms_eq
  sorry

end buy_items_ways_l33_33122


namespace bret_spends_77_dollars_l33_33629

def num_people : ℕ := 4
def main_meal_cost : ℝ := 12.0
def num_appetizers : ℕ := 2
def appetizer_cost : ℝ := 6.0
def tip_rate : ℝ := 0.20
def rush_order_fee : ℝ := 5.0

def total_cost (num_people : ℕ) (main_meal_cost : ℝ) (num_appetizers : ℕ) (appetizer_cost : ℝ) (tip_rate : ℝ) (rush_order_fee : ℝ) : ℝ :=
  let main_meal_total := num_people * main_meal_cost
  let appetizer_total := num_appetizers * appetizer_cost
  let subtotal := main_meal_total + appetizer_total
  let tip := tip_rate * subtotal
  subtotal + tip + rush_order_fee

theorem bret_spends_77_dollars :
  total_cost num_people main_meal_cost num_appetizers appetizer_cost tip_rate rush_order_fee = 77.0 :=
by
  sorry

end bret_spends_77_dollars_l33_33629


namespace weight_of_pants_l33_33736

def weight_socks := 2
def weight_underwear := 4
def weight_shirt := 5
def weight_shorts := 8
def total_allowed := 50

def weight_total (num_shirts num_shorts num_socks num_underwear : Nat) :=
  num_shirts * weight_shirt + num_shorts * weight_shorts + num_socks * weight_socks + num_underwear * weight_underwear

def items_in_wash := weight_total 2 1 3 4

theorem weight_of_pants :
  let weight_pants := total_allowed - items_in_wash
  weight_pants = 10 :=
by
  sorry

end weight_of_pants_l33_33736


namespace solve_problem_l33_33088

open Complex

noncomputable def problem_statement (a : ℝ) : Prop :=
  abs ((a : ℂ) + I) / abs I = 2
  
theorem solve_problem {a : ℝ} : problem_statement a → a = Real.sqrt 3 :=
by
  sorry

end solve_problem_l33_33088


namespace problem_solution_l33_33146

theorem problem_solution (A B : ℝ) (h : ∀ x, x ≠ 3 → (A / (x - 3)) + B * (x + 2) = (-4 * x^2 + 14 * x + 38) / (x - 3)) : 
  A + B = 46 :=
sorry

end problem_solution_l33_33146


namespace probability_not_buy_l33_33725

-- Define the given probability of Sam buying a new book
def P_buy : ℚ := 5 / 8

-- Theorem statement: The probability that Sam will not buy a new book is 3 / 8
theorem probability_not_buy : 1 - P_buy = 3 / 8 :=
by
  -- Proof omitted
  sorry

end probability_not_buy_l33_33725


namespace range_g_l33_33104

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + x + 1
noncomputable def g (a x : ℝ) : ℝ := x^2 + a * x + 1

theorem range_g (a : ℝ) (h : Set.range (λ x => f a x) = Set.univ) : Set.range (λ x => g a x) = { y : ℝ | 1 ≤ y } := by
  sorry

end range_g_l33_33104


namespace grocery_store_distance_l33_33327

theorem grocery_store_distance 
    (park_house : ℕ) (park_store : ℕ) (total_distance : ℕ) (grocery_store_house: ℕ) :
    park_house = 5 ∧ park_store = 3 ∧ total_distance = 16 → grocery_store_house = 8 :=
by 
    sorry

end grocery_store_distance_l33_33327


namespace cube_strictly_increasing_l33_33085

theorem cube_strictly_increasing (a b : ℝ) (h : a > b) : a^3 > b^3 :=
sorry

end cube_strictly_increasing_l33_33085


namespace probability_heads_at_most_3_of_10_coins_flipped_l33_33549

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l33_33549


namespace probability_heads_at_most_3_of_10_coins_flipped_l33_33551

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l33_33551


namespace constant_term_is_35_l33_33074

noncomputable def constant_term_in_binomial_expansion (n : ℕ) : ℕ :=
  (Nat.choose n 3)

theorem constant_term_is_35 :
  ∃ (n : ℕ), 
    (2 * Nat.choose n 2 = Nat.choose n 1 + Nat.choose n 3) ∧
    constant_term_in_binomial_expansion n = 35 :=
by {
  use 7,
  simp only [constant_term_in_binomial_expansion, Nat.factorial, Nat.choose, mul_eq_mul_left_iff],
  split,
  {
    norm_num,
    apply Eq.symm,
    exact Nat.choose_succ_succ_eq n 2 3 
  },
  { norm_num }
}

end constant_term_is_35_l33_33074


namespace store_purchase_ways_l33_33110

theorem store_purchase_ways :
  ∃ (headphones sets_hm sets_km keyboards mice : ℕ), 
  headphones = 9 ∧
  mice = 13 ∧
  keyboards = 5 ∧
  sets_km = 4 ∧
  sets_hm = 5 ∧
  (sets_km * headphones + sets_hm * keyboards + headphones * mice * keyboards = 646) :=
by
  use 9, 5, 4, 5, 13
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  sorry

end store_purchase_ways_l33_33110


namespace line_point_t_l33_33771

theorem line_point_t (t : ℝ) : 
  (∃ t, (0, 3) = (0, 3) ∧ (-8, 0) = (-8, 0) ∧ (5 - 3) / t = 3 / 8) → (t = 16 / 3) :=
by
  sorry

end line_point_t_l33_33771


namespace remainder_of_large_number_l33_33357

theorem remainder_of_large_number :
  (1235678901 % 101) = 1 :=
by
  have h1: (10^8 % 101) = 1 := sorry
  have h2: (10^6 % 101) = 1 := sorry
  have h3: (10^4 % 101) = 1 := sorry
  have h4: (10^2 % 101) = 1 := sorry
  have large_number_decomposition: 1235678901 = 12 * 10^8 + 35 * 10^6 + 67 * 10^4 + 89 * 10^2 + 1 := sorry
  -- Proof using the decomposition and modulo properties
  sorry

end remainder_of_large_number_l33_33357


namespace profit_without_discount_l33_33533

theorem profit_without_discount (CP SP_discount SP_without_discount : ℝ) (profit_discount profit_without_discount percent_discount : ℝ)
  (h1 : CP = 100) 
  (h2 : percent_discount = 0.05) 
  (h3 : profit_discount = 0.425) 
  (h4 : SP_discount = CP + profit_discount * CP) 
  (h5 : SP_discount = 142.5)
  (h6 : SP_without_discount = SP_discount / (1 - percent_discount)) : 
  profit_without_discount = ((SP_without_discount - CP) / CP) * 100 := 
by
  sorry

end profit_without_discount_l33_33533


namespace ray_two_digit_number_l33_33882

theorem ray_two_digit_number (a b n : ℕ) (ha : 1 ≤ a) (hb : b ≤ 9) (hn : n = 10 * a + b) (h1 : n = 4 * (a + b) + 3) (h2 : n + 18 = 10 * b + a) : n = 35 := by
  sorry

end ray_two_digit_number_l33_33882


namespace eliminate_y_substitution_l33_33794

theorem eliminate_y_substitution (x y : ℝ) (h1 : y = x - 5) (h2 : 3 * x - y = 8) : 3 * x - x + 5 = 8 := 
by
  sorry

end eliminate_y_substitution_l33_33794


namespace domain_of_sqrt_function_l33_33860

theorem domain_of_sqrt_function (f : ℝ → ℝ) (x : ℝ) 
  (h1 : ∀ x, (1 / (Real.log x) - 2) ≥ 0) 
  (h2 : ∀ x, Real.log x ≠ 0) : 
  (1 < x ∧ x ≤ Real.sqrt 10) ↔ (∀ x, 0 < Real.log x ∧ Real.log x ≤ 1 / 2) := 
  sorry

end domain_of_sqrt_function_l33_33860


namespace probability_of_at_most_3_heads_l33_33590

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l33_33590


namespace unique_four_letter_sequence_l33_33639

def alphabet_value (c : Char) : ℕ :=
  if 'A' <= c ∧ c <= 'Z' then (c.toNat - 'A'.toNat + 1) else 0

def sequence_product (s : String) : ℕ :=
  s.foldl (λ acc c => acc * alphabet_value c) 1

theorem unique_four_letter_sequence (s : String) :
  sequence_product "WXYZ" = sequence_product s → s = "WXYZ" :=
by
  sorry

end unique_four_letter_sequence_l33_33639


namespace keesha_total_cost_is_correct_l33_33686

noncomputable def hair_cost : ℝ := 
  let cost := 50.0 
  let discount := cost * 0.10 
  let discounted_cost := cost - discount 
  let tip := discounted_cost * 0.20 
  discounted_cost + tip

noncomputable def nails_cost : ℝ := 
  let manicure_cost := 30.0 
  let pedicure_cost := 35.0 * 0.50 
  let total_without_tip := manicure_cost + pedicure_cost 
  let tip := total_without_tip * 0.20 
  total_without_tip + tip

noncomputable def makeup_cost : ℝ := 
  let cost := 40.0 
  let tax := cost * 0.07 
  let total_without_tip := cost + tax 
  let tip := total_without_tip * 0.20 
  total_without_tip + tip

noncomputable def facial_cost : ℝ := 
  let cost := 60.0 
  let discount := cost * 0.15 
  let discounted_cost := cost - discount 
  let tip := discounted_cost * 0.20 
  discounted_cost + tip

noncomputable def total_cost : ℝ := 
  hair_cost + nails_cost + makeup_cost + facial_cost

theorem keesha_total_cost_is_correct : total_cost = 223.56 := by
  sorry

end keesha_total_cost_is_correct_l33_33686


namespace simplify_polynomial_l33_33321

theorem simplify_polynomial :
  (3 * x ^ 5 - 2 * x ^ 3 + 5 * x ^ 2 - 8 * x + 6) + (7 * x ^ 4 + x ^ 3 - 3 * x ^ 2 + x - 9) =
  3 * x ^ 5 + 7 * x ^ 4 - x ^ 3 + 2 * x ^ 2 - 7 * x - 3 :=
by
  sorry

end simplify_polynomial_l33_33321


namespace solve_quadratic_eq_l33_33982

theorem solve_quadratic_eq (x : ℂ) : 
  x^2 + 6 * x + 8 = -(x + 2) * (x + 6) ↔ (x = -3 + complex.I ∨ x = -3 - complex.I) := 
by
  sorry 

end solve_quadratic_eq_l33_33982


namespace simplify_and_evaluate_l33_33319

variable (x y : ℝ)
variable (condition_x : x = 1/3)
variable (condition_y : y = -6)

theorem simplify_and_evaluate :
  3 * x^2 * y - (6 * x * y^2 - 2 * (x * y + (3/2) * x^2 * y)) + 2 * (3 * x * y^2 - x * y) = -4 :=
by
  rw [condition_x, condition_y]
  sorry

end simplify_and_evaluate_l33_33319


namespace probability_heads_at_most_3_l33_33573

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l33_33573


namespace swimming_time_l33_33755

theorem swimming_time (c t : ℝ) 
  (h1 : 10.5 + c ≠ 0)
  (h2 : 10.5 - c ≠ 0)
  (h3 : t = 45 / (10.5 + c))
  (h4 : t = 18 / (10.5 - c)) :
  t = 3 := 
by
  sorry

end swimming_time_l33_33755


namespace initially_calculated_average_height_l33_33985

/-- Suppose the average height of 20 students was initially calculated incorrectly. Later, it was found that one student's height 
was incorrectly recorded as 151 cm instead of 136 cm. Given the actual average height of the students is 174.25 cm, prove that the 
initially calculated average height was 173.5 cm. -/
theorem initially_calculated_average_height
  (initial_avg actual_avg : ℝ)
  (num_students : ℕ)
  (incorrect_height correct_height : ℝ)
  (h_avg : actual_avg = 174.25)
  (h_students : num_students = 20)
  (h_incorrect : incorrect_height = 151)
  (h_correct : correct_height = 136)
  (h_total_actual : num_students * actual_avg = num_students * initial_avg + incorrect_height - correct_height) :
  initial_avg = 173.5 :=
by
  sorry

end initially_calculated_average_height_l33_33985


namespace probability_at_most_3_heads_10_flips_l33_33594

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l33_33594


namespace perpendicular_lines_b_value_l33_33497

theorem perpendicular_lines_b_value 
  (b : ℝ) 
  (line1 : ∀ x y : ℝ, x + 3 * y + 5 = 0 → True) 
  (line2 : ∀ x y : ℝ, b * x + 3 * y + 5 = 0 → True)
  (perpendicular_condition : (-1 / 3) * (-b / 3) = -1) : 
  b = -9 := 
sorry

end perpendicular_lines_b_value_l33_33497


namespace to_ellipse_area_correct_l33_33675

/-
Define the endpoints of the major axis of the ellipse.
-/
def p1 : ℝ × ℝ := (-5, 3)
def p2 : ℝ × ℝ := (15, 3)

/-
Define the point through which the ellipse passes.
-/
def pointOnEllipse : ℝ × ℝ := (10, 10)

/-
Calculate the midpoint as the center of the ellipse.
-/
def center : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

/-
Calculate the semi-major axis length.
-/
def a : ℝ := (Real.abs (p2.1 - p1.1)) / 2

/-
Calculate b^2 from the given ellipse equation and point.
-/
def bSquare : ℝ := (49 * 4) / 3

/-
The area of the ellipse is given by the formula π * a * b, where b = sqrt(bSquare).
-/
noncomputable def area : ℝ := Real.pi * a * Real.sqrt bSquare

/-
Theorem to prove that the calculated area is equal to the given correct answer.
-/
theorem ellipse_area_correct : area = (140 * Real.pi / 3) * Real.sqrt 3 := sorry

end to_ellipse_area_correct_l33_33675


namespace probability_heads_at_most_3_l33_33574

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l33_33574


namespace sum_of_digits_l33_33465

theorem sum_of_digits :
  ∃ (E M V Y : ℕ), 
    (E ≠ M ∧ E ≠ V ∧ E ≠ Y ∧ M ≠ V ∧ M ≠ Y ∧ V ≠ Y) ∧
    (10 * Y + E) * (10 * M + E) = 111 * V ∧ 
    1 ≤ V ∧ V ≤ 9 ∧ 
    E + M + V + Y = 21 :=
by 
  sorry

end sum_of_digits_l33_33465


namespace cafeteria_can_make_7_pies_l33_33884

theorem cafeteria_can_make_7_pies (initial_apples handed_out apples_per_pie : ℕ)
  (h1 : initial_apples = 86)
  (h2 : handed_out = 30)
  (h3 : apples_per_pie = 8) :
  ((initial_apples - handed_out) / apples_per_pie) = 7 := 
by
  sorry

end cafeteria_can_make_7_pies_l33_33884


namespace line_and_circle_separate_l33_33806

theorem line_and_circle_separate
  (θ : ℝ) (hθ : ¬ ∃ k : ℤ, θ = k * Real.pi) :
  ¬ ∃ (x y : ℝ), (x^2 + y^2 = 1 / 2) ∧ (x * Real.cos θ + y - 1 = 0) :=
by
  sorry

end line_and_circle_separate_l33_33806


namespace zoey_holidays_in_a_year_l33_33743

-- Given conditions as definitions
def holidays_per_month : ℕ := 2
def months_in_a_year : ℕ := 12

-- Definition of the total holidays in a year
def total_holidays_in_year : ℕ := holidays_per_month * months_in_a_year

-- Proof statement
theorem zoey_holidays_in_a_year : total_holidays_in_year = 24 := 
by
  sorry

end zoey_holidays_in_a_year_l33_33743


namespace right_triangle_area_l33_33514

theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 240 :=
by
  sorry

end right_triangle_area_l33_33514


namespace abs_sum_condition_l33_33100

theorem abs_sum_condition (a b : ℝ) (h1 : |a| = 7) (h2 : |b| = 3) (h3 : a * b > 0) : a + b = 10 ∨ a + b = -10 :=
by { sorry }

end abs_sum_condition_l33_33100


namespace prob_heads_at_most_3_out_of_10_flips_l33_33553

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l33_33553


namespace monotonic_intervals_range_of_values_l33_33947

-- Part (1): Monotonic intervals of the function
theorem monotonic_intervals (a : ℝ) (h_a : a = 0) :
  (∀ x, 0 < x ∧ x < 1 → (1 + Real.log x) / x > 0) ∧ (∀ x, 1 < x → (1 + Real.log x) / x < 0) :=
by
  sorry

-- Part (2): Range of values for \(a\)
theorem range_of_values (a : ℝ) (h_f : ∀ x, 0 < x → (1 + Real.log x) / x - a ≤ 0) : 
  1 ≤ a :=
by
  sorry

end monotonic_intervals_range_of_values_l33_33947


namespace quadratic_real_roots_range_l33_33268

theorem quadratic_real_roots_range (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 3 * x - 9 / 4 = 0) →
  (k >= -1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_real_roots_range_l33_33268


namespace simplify_expression_l33_33981

theorem simplify_expression (x y : ℝ) : (5 - 4 * y) - (6 + 5 * y - 2 * x) = -1 - 9 * y + 2 * x := by
  sorry

end simplify_expression_l33_33981


namespace anna_discontinued_coaching_on_2nd_august_l33_33907

theorem anna_discontinued_coaching_on_2nd_august
  (coaching_days : ℕ) (non_leap_year : ℕ) (first_day : ℕ) 
  (days_in_january : ℕ) (days_in_february : ℕ) (days_in_march : ℕ) 
  (days_in_april : ℕ) (days_in_may : ℕ) (days_in_june : ℕ) 
  (days_in_july : ℕ) (days_in_august : ℕ)
  (not_leap_year : non_leap_year = 365)
  (first_day_of_year : first_day = 1)
  (january_days : days_in_january = 31)
  (february_days : days_in_february = 28)
  (march_days : days_in_march = 31)
  (april_days : days_in_april = 30)
  (may_days : days_in_may = 31)
  (june_days : days_in_june = 30)
  (july_days : days_in_july = 31)
  (august_days : days_in_august = 31)
  (total_coaching_days : coaching_days = 245) :
  ∃ day, day = 2 ∧ month = "August" := 
sorry

end anna_discontinued_coaching_on_2nd_august_l33_33907


namespace solve_fraction_equation_l33_33855

theorem solve_fraction_equation (x : ℝ) :
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ↔ x = -9 :=
by {
  sorry
}

end solve_fraction_equation_l33_33855


namespace min_value_expression_l33_33738

theorem min_value_expression (x : ℝ) : 
  ∃ y : ℝ, (y = (x+2)*(x+3)*(x+4)*(x+5) + 3033) ∧ y ≥ 3032 ∧ 
  (∀ z : ℝ, (z = (x+2)*(x+3)*(x+4)*(x+5) + 3033) → z ≥ 3032) := 
sorry

end min_value_expression_l33_33738


namespace ratio_of_roots_l33_33923

theorem ratio_of_roots (c : ℝ) :
  (∃ (x1 x2 : ℝ), 5 * x1^2 - 2 * x1 + c = 0 ∧ 5 * x2^2 - 2 * x2 + c = 0 ∧ x1 / x2 = -3 / 5) → c = -3 :=
by
  sorry

end ratio_of_roots_l33_33923


namespace fraction_subtraction_property_l33_33218

variable (a b c d : ℚ)

theorem fraction_subtraction_property :
  (a / b - c / d) = ((a - c) / (b + d)) → (a / c) = (b / d) ^ 2 := 
by
  sorry

end fraction_subtraction_property_l33_33218


namespace polygon_area_correct_l33_33487

def AreaOfPolygon : Real := 37.5

def polygonVertices : List (Real × Real) :=
  [(0, 0), (5, 0), (5, 5), (0, 5), (5, 10), (0, 10), (0, 0)]

theorem polygon_area_correct :
  (∃ (A : Real) (verts : List (Real × Real)),
    verts = polygonVertices ∧ A = AreaOfPolygon ∧ 
    A = 37.5) := by
  sorry

end polygon_area_correct_l33_33487


namespace sum_of_digits_l33_33795

theorem sum_of_digits (A B C D : ℕ) (H1: A < B) (H2: B < C) (H3: C < D)
  (H4: A > 0) (H5: B > 0) (H6: C > 0) (H7: D > 0)
  (H8: 1000 * A + 100 * B + 10 * C + D + 1000 * D + 100 * C + 10 * B + A = 11990) : 
  (A, B, C, D) = (1, 9, 9, 9) :=
sorry

end sum_of_digits_l33_33795


namespace greatest_value_q_minus_r_l33_33537

theorem greatest_value_q_minus_r {x y : ℕ} (hx : x < 10) (hy : y < 10) (hqr : 9 * (x - y) < 70) :
  9 * (x - y) = 63 :=
sorry

end greatest_value_q_minus_r_l33_33537


namespace minimum_dot_product_l33_33137

-- Define point coordinates
structure Point where
  x : ℝ
  y : ℝ

-- Define points A, B, C, D according to the given problem statement
def A : Point := ⟨0, 0⟩
def B : Point := ⟨1, 0⟩
def C : Point := ⟨1, 2⟩
def D : Point := ⟨0, 2⟩

-- Define the condition for points E and F on the sides BC and CD respectively.
def isOnBC (E : Point) : Prop := E.x = 1 ∧ 0 ≤ E.y ∧ E.y ≤ 2
def isOnCD (F : Point) : Prop := F.y = 2 ∧ 0 ≤ F.x ∧ F.x ≤ 1

-- Define the distance constraint for |EF| = 1
def distEF (E F : Point) : Prop :=
  (F.x - E.x)^2 + (F.y - E.y)^2 = 1

-- Define the dot product between vectors AE and AF
def dotProductAEAF (E F : Point) : ℝ :=
  2 * E.y + F.x

-- Main theorem to prove the minimum dot product value
theorem minimum_dot_product (E F : Point) (hE : isOnBC E) (hF : isOnCD F) (hDistEF : distEF E F) :
  dotProductAEAF E F = 5 - Real.sqrt 5 :=
  sorry

end minimum_dot_product_l33_33137


namespace largest_number_2013_l33_33732

theorem largest_number_2013 (x y : ℕ) (h1 : x + y = 2013)
    (h2 : y = 5 * (x / 100 + 1)) : max x y = 1913 := by
  sorry

end largest_number_2013_l33_33732


namespace balls_in_boxes_distinguished_boxes_l33_33260

theorem balls_in_boxes_distinguished_boxes :
  ∃ (n : ℕ), n = nat.choose 9 2 ∧ n = 36 :=
by {
  -- just write the statement and use 'sorry' to skip proof
  use 36,
  split,
  { unfold nat.choose,
    -- calculation details omitted
    sorry, },
  { refl, },
}

end balls_in_boxes_distinguished_boxes_l33_33260


namespace abs_diff_m_n_l33_33689

variable (m n : ℝ)

theorem abs_diff_m_n (h1 : m * n = 6) (h2 : m + n = 7) (h3 : m^2 - n^2 = 13) : |m - n| = 13 / 7 :=
by
  sorry

end abs_diff_m_n_l33_33689


namespace small_mold_radius_l33_33893

theorem small_mold_radius (r : ℝ) (n : ℝ) (s : ℝ) :
    r = 2 ∧ n = 8 ∧ (1 / 2) * (2 / 3) * Real.pi * r^3 = (8 * (2 / 3) * Real.pi * s^3) → s = 1 :=
by
  sorry

end small_mold_radius_l33_33893


namespace green_block_weight_l33_33963

theorem green_block_weight
    (y : ℝ)
    (g : ℝ)
    (h1 : y = 0.6)
    (h2 : y = g + 0.2) :
    g = 0.4 :=
by
  sorry

end green_block_weight_l33_33963


namespace probability_at_most_three_heads_10_coins_l33_33598

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l33_33598


namespace red_fraction_after_doubling_l33_33434

theorem red_fraction_after_doubling (x : ℕ) (h : (3/5 : ℚ) * x = (3/5 : ℚ) * x) : 
  let blue_marbles := (3/5 : ℚ) * x
      red_marbles := x - blue_marbles
      new_red_marbles := 2 * red_marbles 
      total_marbles := blue_marbles + new_red_marbles 
  in new_red_marbles / total_marbles = (4/7 : ℚ) :=
by 
  sorry

end red_fraction_after_doubling_l33_33434


namespace find_efg_correct_l33_33171

noncomputable def find_efg (M : ℕ) : ℕ :=
  let efgh := M % 10000
  let e := efgh / 1000
  let efg := efgh / 10
  if (M^2 % 10000 = efgh) ∧ (e ≠ 0) ∧ ((M % 32 = 0 ∧ (M - 1) % 125 = 0) ∨ (M % 125 = 0 ∧ (M - 1) % 32 = 0))
  then efg
  else 0
  
theorem find_efg_correct {M : ℕ} (h_conditions: (M^2 % 10000 = M % 10000) ∧ (M % 32 = 0 ∧ (M - 1) % 125 = 0 ∨ M % 125 = 0 ∧ (M-1) % 32 = 0) ∧ ((M % 10000 / 1000) ≠ 0)) :
  find_efg M = 362 :=
by
  sorry

end find_efg_correct_l33_33171


namespace probability_at_most_3_heads_10_flips_l33_33595

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l33_33595


namespace find_cost_price_per_item_min_items_type_A_l33_33371

-- Definitions based on the conditions
def cost_A (x : ℝ) (y : ℝ) : Prop := 4 * x + 10 = 5 * y
def cost_B (x : ℝ) (y : ℝ) : Prop := 20 * x + 10 * y = 160

-- Proving the cost price per item of goods A and B
theorem find_cost_price_per_item : ∃ x y : ℝ, cost_A x y ∧ cost_B x y ∧ x = 5 ∧ y = 6 :=
by
  -- This is where the proof would go
  sorry

-- Additional conditions for part (2)
def profit_condition (a : ℕ) : Prop :=
  10 * (a - 30) + 8 * (200 - (a - 30)) - 5 * a - 6 * (200 - a) ≥ 640

-- Proving the minimum number of items of type A purchased
theorem min_items_type_A : ∃ a : ℕ, profit_condition a ∧ a ≥ 100 :=
by
  -- This is where the proof would go
  sorry

end find_cost_price_per_item_min_items_type_A_l33_33371


namespace slope_of_tangent_at_point_l33_33172

theorem slope_of_tangent_at_point (x : ℝ) (y : ℝ) (h_curve : y = x^3)
    (h_slope : 3*x^2 = 3) : (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) := 
sorry

end slope_of_tangent_at_point_l33_33172


namespace carlos_goal_l33_33765

def july_books : ℕ := 28
def august_books : ℕ := 30
def june_books : ℕ := 42

theorem carlos_goal (goal : ℕ) :
  goal = june_books + july_books + august_books := by
  sorry

end carlos_goal_l33_33765


namespace range_of_a_l33_33807

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 5}

-- Define the theorem to be proved
theorem range_of_a (a : ℝ) (h₁ : A a ⊆ B) (h₂ : 2 - a < 2 + a) : 0 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l33_33807


namespace count_four_digit_numbers_divisible_by_5_ending_in_45_l33_33411

theorem count_four_digit_numbers_divisible_by_5_ending_in_45 : 
  ∃ (n : ℕ), n = 90 ∧ ∀ (x : ℕ), (1000 ≤ x ∧ x < 10000) ∧ (x % 100 = 45) → count x = n :=
sorry

end count_four_digit_numbers_divisible_by_5_ending_in_45_l33_33411


namespace range_of_a_l33_33428

noncomputable def f (a x : ℝ) : ℝ := Real.log (x^2 - a * x - 3)

def monotonic_increasing (a : ℝ) : Prop :=
  ∀ x > 1, 2 * x - a > 0

def positive_argument (a : ℝ) : Prop :=
  ∀ x > 1, x^2 - a * x - 3 > 0

theorem range_of_a :
  {a : ℝ | monotonic_increasing a ∧ positive_argument a} = {a : ℝ | a ≤ -2} :=
sorry

end range_of_a_l33_33428


namespace mother_l33_33955

def problem_conditions (D M : ℤ) : Prop :=
  (2 * D + M = 70) ∧ (D + 2 * M = 95)

theorem mother's_age_is_40 (D M : ℤ) (h : problem_conditions D M) : M = 40 :=
by sorry

end mother_l33_33955


namespace classA_win_championship_expectation_of_X_is_0_9_l33_33679

noncomputable def classA_win_championship_probability : ℝ :=
  let pA := 0.4
  let pB := 0.5
  let pC := 0.8
  pA * pB * pC + (1 - pA) * pB * pC + pA * (1 - pB) * pC + pA * pB * (1 - pC)

theorem classA_win_championship : classA_win_championship_probability = 0.6 := 
sorry

def distribution_of_X : list (ℤ × ℝ) :=
  [ (-3, 0.4 * 0.5 * 0.8),
    (0, (1 - 0.4) * 0.5 * 0.8 + 0.4 * (1 - 0.5) * 0.2 + 0.4 * 0.5 * (1 - 0.8)),
    (3, (1 - 0.4) * 0.5 * 0.2 + (1 - 0.4) * 0.5 * 0.8 + 0.4 * (1 - 0.5) * 0.2),
    (6, (1 - 0.4) * (1 - 0.5) * 0.2) ]

noncomputable def expectation_of_X : ℝ :=
  ∑ (x, px) in distribution_of_X, x * px

theorem expectation_of_X_is_0_9 : expectation_of_X = 0.9 :=
sorry

end classA_win_championship_expectation_of_X_is_0_9_l33_33679


namespace total_spots_l33_33690

-- Define the variables
variables (R C G S B : ℕ)

-- State the problem conditions
def conditions : Prop :=
  R = 46 ∧
  C = R / 2 - 5 ∧
  G = 5 * C ∧
  S = 3 * R ∧
  B = 2 * (G + S)

-- State the proof problem
theorem total_spots : conditions R C G S B → G + C + S + B = 702 :=
by
  intro h
  obtain ⟨hR, hC, hG, hS, hB⟩ := h
  -- The proof steps would go here
  sorry

end total_spots_l33_33690


namespace minimum_harmonic_sum_l33_33292

theorem minimum_harmonic_sum
  (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sum : a + b + c = 2) :
  (1 / a + 1 / b + 1 / c) ≥ 9 / 2 :=
by
  sorry

end minimum_harmonic_sum_l33_33292


namespace projection_problem_l33_33147

noncomputable def vector_proj (w v : ℝ × ℝ) : ℝ × ℝ := sorry -- assume this definition

variables (v w : ℝ × ℝ)

-- Given condition
axiom proj_v : vector_proj w v = ⟨4, 3⟩

-- Proof Statement
theorem projection_problem :
  vector_proj w (7 • v + 2 • w) = ⟨28, 21⟩ + 2 • w :=
sorry

end projection_problem_l33_33147


namespace count_valid_numbers_l33_33416

-- Define what it means to be a four-digit number that ends in 45
def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ n % 100 = 45 ∧ n % 5 = 0

-- Define the set of valid two-digit prefixes
def valid_prefixes : set ℕ := {ab | 10 ≤ ab ∧ ab ≤ 99}

-- Define the set of four-digit numbers that end in 45 and are divisible by 5
def valid_numbers : set ℕ := {n | ∃ ab : ℕ, ab ∈ valid_prefixes ∧ n = ab * 100 + 45}

-- State the theorem
theorem count_valid_numbers : (finset.card (finset.filter is_valid_number (finset.range 10000)) = 90) :=
sorry

end count_valid_numbers_l33_33416


namespace total_juice_drunk_l33_33316

noncomputable def juiceConsumption (samDrink benDrink : ℕ) (samConsRatio benConsRatio : ℚ) : ℚ :=
  let samConsumed := samConsRatio * samDrink
  let samRemaining := samDrink - samConsumed
  let benConsumed := benConsRatio * benDrink
  let benRemaining := benDrink - benConsumed
  let benToSam := (1 / 2) * benRemaining + 1
  let samTotal := samConsumed + benToSam
  let benTotal := benConsumed - benToSam
  samTotal + benTotal

theorem total_juice_drunk : juiceConsumption 12 20 (2 / 3 : ℚ) (2 / 3 : ℚ) = 32 :=
sorry

end total_juice_drunk_l33_33316


namespace gcd_8251_6105_l33_33018

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 39 := by
  sorry

end gcd_8251_6105_l33_33018


namespace candies_bought_friday_l33_33238

-- Definitions based on the given conditions
def candies_bought_tuesday : ℕ := 3
def candies_bought_thursday : ℕ := 5
def candies_left (c : ℕ) : Prop := c = 4
def candies_eaten (c : ℕ) : Prop := c = 6

-- Theorem to prove the number of candies bought on Friday
theorem candies_bought_friday (c_left c_eaten : ℕ) (h_left : candies_left c_left) (h_eaten : candies_eaten c_eaten) : 
  (10 - (candies_bought_tuesday + candies_bought_thursday) = 2) :=
  by
    sorry

end candies_bought_friday_l33_33238


namespace max_participants_won_at_least_three_matches_l33_33610

theorem max_participants_won_at_least_three_matches :
  ∀ (n : ℕ), n = 200 → ∃ k : ℕ, k ≤ 66 ∧ ∀ p : ℕ, (p ≥ k ∧ p > 66) → false := by
  sorry

end max_participants_won_at_least_three_matches_l33_33610


namespace estimated_fish_in_pond_l33_33136

theorem estimated_fish_in_pond :
  ∀ (number_marked_first_catch total_second_catch number_marked_second_catch : ℕ),
    number_marked_first_catch = 100 →
    total_second_catch = 108 →
    number_marked_second_catch = 9 →
    ∃ est_total_fish : ℕ, (number_marked_second_catch / total_second_catch : ℝ) = (number_marked_first_catch / est_total_fish : ℝ) ∧ est_total_fish = 1200 := 
by
  intros number_marked_first_catch total_second_catch number_marked_second_catch
  sorry

end estimated_fish_in_pond_l33_33136


namespace fifth_eqn_nth_eqn_l33_33695

theorem fifth_eqn : 10 * 12 + 1 = 121 :=
by
  sorry

theorem nth_eqn (n : ℕ) : 2 * n * (2 * n + 2) + 1 = (2 * n + 1)^2 :=
by
  sorry

end fifth_eqn_nth_eqn_l33_33695


namespace Jonah_paid_commensurate_l33_33828

def price_per_pineapple (P : ℝ) :=
  let number_of_pineapples := 6
  let rings_per_pineapple := 12
  let total_rings := number_of_pineapples * rings_per_pineapple
  let price_per_4_rings := 5
  let price_per_ring := price_per_4_rings / 4
  let total_revenue := total_rings * price_per_ring
  let profit := 72
  total_revenue - number_of_pineapples * P = profit

theorem Jonah_paid_commensurate {P : ℝ} (h : price_per_pineapple P) :
  P = 3 :=
  sorry

end Jonah_paid_commensurate_l33_33828


namespace mryak_bryak_problem_l33_33870

variable (m b : ℚ)

theorem mryak_bryak_problem
  (h1 : 3 * m = 5 * b + 10)
  (h2 : 6 * m = 8 * b + 31) :
  7 * m - 9 * b = 38 := sorry

end mryak_bryak_problem_l33_33870


namespace g_interval_l33_33968

noncomputable def g (a b c : ℝ) : ℝ :=
  a / (a + b) + b / (b + c) + c / (c + a)

theorem g_interval (a b c : ℝ) (ha : 0 < a) (hb: 0 < b) (hc : 0 < c) :
  1 < g a b c ∧ g a b c < 2 :=
sorry

end g_interval_l33_33968


namespace cloak_change_in_silver_l33_33460

theorem cloak_change_in_silver :
  (∀ c : ℤ, (20 = c + 4) → (15 = c + 1)) →
  (5 * g = 3) →
  14 * gold / exchange_rate = 10 := 
sorry

end cloak_change_in_silver_l33_33460


namespace arithmetic_sequence_product_l33_33346

theorem arithmetic_sequence_product (a : ℕ → ℚ) (d : ℚ) (a7 : a 7 = 20) (diff : d = 2) : 
  let a1 := a 1,
      a2 := a 2
  in a1 * a2 = 80 :=
by
  sorry

end arithmetic_sequence_product_l33_33346


namespace solution_cos_eq_l33_33927

open Real

theorem solution_cos_eq (x : ℝ) :
  (cos x)^2 + (cos (2 * x))^2 + (cos (3 * x))^2 = 1 ↔
  (∃ k : ℤ, x = k * π / 2 + π / 4) ∨ (∃ k : ℤ, x = k * π / 3 + π / 6) :=
by sorry

end solution_cos_eq_l33_33927


namespace store_purchase_ways_l33_33108

theorem store_purchase_ways :
  ∃ (headphones sets_hm sets_km keyboards mice : ℕ), 
  headphones = 9 ∧
  mice = 13 ∧
  keyboards = 5 ∧
  sets_km = 4 ∧
  sets_hm = 5 ∧
  (sets_km * headphones + sets_hm * keyboards + headphones * mice * keyboards = 646) :=
by
  use 9, 5, 4, 5, 13
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  sorry

end store_purchase_ways_l33_33108


namespace smallest_positive_integer_n_l33_33529

theorem smallest_positive_integer_n :
  ∃ n : ℕ, (n > 0 ∧ 17 * n % 7 = 2) ∧ ∀ m : ℕ, (m > 0 ∧ 17 * m % 7 = 2) → n ≤ m := 
sorry

end smallest_positive_integer_n_l33_33529


namespace find_number_of_shorts_l33_33621

def price_of_shorts : ℕ := 7
def price_of_shoes : ℕ := 20
def total_spent : ℕ := 75

-- We represent the price of 4 tops as a variable
variable (T : ℕ)

theorem find_number_of_shorts (S : ℕ) (h : 7 * S + 4 * T + 20 = 75) : S = 7 :=
by
  sorry

end find_number_of_shorts_l33_33621


namespace terminating_decimal_count_l33_33791

theorem terminating_decimal_count : 
  (Finset.filter (λ n, n % 3 = 0) (Finset.range 181)).card = 60 := by
  sorry

end terminating_decimal_count_l33_33791


namespace fill_in_the_blank_l33_33242

theorem fill_in_the_blank (x : ℕ) (h : (x - x) + x * x + x / x = 50) : x = 7 :=
sorry

end fill_in_the_blank_l33_33242


namespace least_subtracted_number_l33_33026

def is_sum_of_digits_at_odd_places (n : ℕ) : ℕ :=
  (n / 100000) % 10 + (n / 1000) % 10 + (n / 10) % 10

def is_sum_of_digits_at_even_places (n : ℕ) : ℕ :=
  (n / 10000) % 10 + (n / 100) % 10 + (n % 10)

def diff_digits_odd_even (n : ℕ) : ℕ :=
  is_sum_of_digits_at_odd_places n - is_sum_of_digits_at_even_places n

theorem least_subtracted_number :
  ∃ x : ℕ, (427398 - x) % 11 = 0 ∧ x = 7 :=
by
  sorry

end least_subtracted_number_l33_33026


namespace j_at_4_l33_33162

noncomputable def h (x : ℚ) : ℚ := 5 / (3 - x)

noncomputable def h_inv (x : ℚ) : ℚ := (3 * x - 5) / x

noncomputable def j (x : ℚ) : ℚ := (1 / h_inv x) + 7

theorem j_at_4 : j 4 = 53 / 7 :=
by
  -- Proof steps would be inserted here.
  sorry

end j_at_4_l33_33162


namespace new_energy_vehicles_l33_33212

-- Given conditions
def conditions (a b : ℕ) : Prop :=
  3 * a + 2 * b = 95 ∧ 4 * a + 1 * b = 110

-- Given prices
def purchase_prices : Prop :=
  ∃ a b, conditions a b ∧ a = 25 ∧ b = 10

-- Total value condition for different purchasing plans
def purchase_plans (m n : ℕ) : Prop :=
  25 * m + 10 * n = 250 ∧ m > 0 ∧ n > 0

-- Number of different purchasing plans
def num_purchase_plans : Prop :=
  ∃ num_plans, num_plans = 4

-- Profit calculation for a given plan
def profit (m n : ℕ) : ℕ :=
  12 * m + 8 * n

-- Maximum profit condition
def max_profit : Prop :=
  ∃ max_profit, max_profit = 184 ∧ ∀ (m n : ℕ), purchase_plans m n → profit m n ≤ 184

-- Main theorem
theorem new_energy_vehicles : purchase_prices ∧ num_purchase_plans ∧ max_profit :=
  sorry

end new_energy_vehicles_l33_33212


namespace cloak_change_l33_33456

theorem cloak_change (silver_for_cloak1 silver_for_cloak2 : ℕ) (gold_change1 gold_change2 : ℕ) (silver_change : ℕ) :
  silver_for_cloak1 = 20 →
  gold_change1 = 4 →
  silver_for_cloak2 = 15 →
  gold_change2 = 1 →
  ∃ silver_cost_of_cloak : ℕ, 
    silver_cost_of_cloak = (20 - 4) * (5 / 3) →
    silver_change = 10 →
    14 * (5 / 3) - 8 = silver_change :=
by 
  assume h1 h2 h3 h4,
  use 16, 
  sorry

end cloak_change_l33_33456


namespace roots_reciprocal_sum_l33_33509

theorem roots_reciprocal_sum (x₁ x₂ : ℝ) 
    (h_roots : x₁ * x₁ + x₁ - 2 = 0 ∧ x₂ * x₂ + x₂ - 2 = 0):
    x₁ ≠ x₂ → (1 / x₁ + 1 / x₂ = 1 / 2) :=
by
  intro h_neq
  sorry

end roots_reciprocal_sum_l33_33509


namespace solve_eq_2_pow_x_plus_3_pow_y_eq_z_sq_l33_33387

theorem solve_eq_2_pow_x_plus_3_pow_y_eq_z_sq (x y z : ℕ) :
  ((x = 3 ∧ y = 0 ∧ z = 3) ∨ (x = 0 ∧ y = 1 ∧ z = 2) ∨ (x = 4 ∧ y = 2 ∧ z = 5)) →
  2^x + 3^y = z^2 :=
by
  sorry

end solve_eq_2_pow_x_plus_3_pow_y_eq_z_sq_l33_33387


namespace battery_life_ge_6_l33_33867

-- Defining the service life of the battery as a normal random variable with mean 4
-- and variance sigma^2, where sigma > 0.

noncomputable theory
open ProbabilityTheory MeasureTheory

variable (σ : ℝ) (hσ : σ > 0)

-- X is a normally distributed random variable N(4, σ^2)
def X : ProbabilitySpace ℝ := Normal 4 σ

-- The probability that service life is at least 2 years is given as 0.9
axiom hx2 : ℙ {x | X x ≥ 2} = 0.9

-- Prove that the probability the battery lasts at least 6 years is 0.1
theorem battery_life_ge_6 : ℙ {x | X x ≥ 6} = 0.1 :=
sorry

end battery_life_ge_6_l33_33867


namespace combined_profit_percentage_correct_l33_33370

-- Definitions based on the conditions
noncomputable def profit_percentage_A := 30
noncomputable def discount_percentage_A := 10
noncomputable def profit_percentage_B := 24
noncomputable def discount_percentage_B := 15
noncomputable def profit_percentage_C := 40
noncomputable def discount_percentage_C := 20

-- Function to calculate selling price without discount
noncomputable def selling_price_without_discount (cost_price profit_percentage : ℝ) : ℝ :=
  cost_price * (1 + profit_percentage / 100)

-- Assume cost price for simplicity
noncomputable def cost_price : ℝ := 100

-- Calculations based on the conditions
noncomputable def selling_price_A := selling_price_without_discount cost_price profit_percentage_A
noncomputable def selling_price_B := selling_price_without_discount cost_price profit_percentage_B
noncomputable def selling_price_C := selling_price_without_discount cost_price profit_percentage_C

-- Calculate total cost price and the total selling price without any discount
noncomputable def total_cost_price := 3 * cost_price
noncomputable def total_selling_price_without_discount := selling_price_A + selling_price_B + selling_price_C

-- Combined profit
noncomputable def combined_profit := total_selling_price_without_discount - total_cost_price

-- Combined profit percentage
noncomputable def combined_profit_percentage := (combined_profit / total_cost_price) * 100

theorem combined_profit_percentage_correct :
  combined_profit_percentage = 31.33 :=
by
  sorry

end combined_profit_percentage_correct_l33_33370


namespace bananas_count_l33_33175

theorem bananas_count
    (total_fruit : ℕ)
    (apples_ratio : ℕ)
    (persimmons_ratio : ℕ)
    (apples_and_persimmons : apples_ratio * bananas + persimmons_ratio * bananas = total_fruit)
    (apples_ratio_val : apples_ratio = 4)
    (persimmons_ratio_val : persimmons_ratio = 3)
    (total_fruit_value : total_fruit = 210) :
    bananas = 30 :=
by
  sorry

end bananas_count_l33_33175


namespace silver_coins_change_l33_33443

-- Define the conditions
def condition1 : ℕ × ℕ := (20, 4) -- (20 silver coins, 4 gold coins change)
def condition2 : ℕ × ℕ := (15, 1) -- (15 silver coins, 1 gold coin change)
def cost_of_cloak_in_gold_coins : ℕ := 14

-- Define the theorem to be proven
theorem silver_coins_change (s1 g1 s2 g2 cloak_g : ℕ) (h1 : (s1, g1) = condition1) (h2 : (s2, g2) = condition2) :
  ∃ silver : ℕ, (silver = 10) :=
by {
  sorry
}

end silver_coins_change_l33_33443


namespace janet_total_earnings_l33_33283

-- Definitions based on conditions from step a)
def hourly_wage := 70
def hours_worked := 20
def rate_per_pound := 20
def weight_sculpture1 := 5
def weight_sculpture2 := 7

-- Statement for the proof problem
theorem janet_total_earnings : 
  let earnings_from_extermination := hourly_wage * hours_worked
  let earnings_from_sculpture1 := rate_per_pound * weight_sculpture1
  let earnings_from_sculpture2 := rate_per_pound * weight_sculpture2
  earnings_from_extermination + earnings_from_sculpture1 + earnings_from_sculpture2 = 1640 := 
by
  sorry

end janet_total_earnings_l33_33283


namespace area_of_triangle_ABC_eq_3_l33_33799

variable {n : ℕ}

def arithmetic_seq (a_1 d : ℤ) : ℕ → ℤ
| 0     => 0
| (n+1) => a_1 + n * d

def sum_arithmetic_seq (a_1 d : ℤ) : ℕ → ℤ
| 0     => 0
| (n+1) => (n + 1) * a_1 + (n * (n + 1) / 2) * d

def f (n : ℕ) : ℤ := sum_arithmetic_seq 4 6 n

def point_A (n : ℕ) : ℤ × ℤ := (n, f n)
def point_B (n : ℕ) : ℤ × ℤ := (n + 1, f (n + 1))
def point_C (n : ℕ) : ℤ × ℤ := (n + 2, f (n + 2))

def area_of_triangle (A B C : ℤ × ℤ) : ℤ :=
  (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)).natAbs / 2

theorem area_of_triangle_ABC_eq_3 : 
  ∀ (n : ℕ), area_of_triangle (point_A n) (point_B n) (point_C n) = 3 := 
sorry

end area_of_triangle_ABC_eq_3_l33_33799


namespace length_of_AB_area_of_ΔABF1_l33_33944

theorem length_of_AB (A B : ℝ × ℝ) (x1 x2 y1 y2 : ℝ) :
  (y1 = x1 - 2) ∧ (y2 = x2 - 2) ∧ ((x1 = 0) ∧ (y1 = -2)) ∧ ((x2 = 8/3) ∧ (y2 = 2/3)) →
  |((x1 - x2)^2 + (y1 - y2)^2)^(1/2)| = (8 / 3) * (2)^(1/2) :=
by sorry

theorem area_of_ΔABF1 (A B F1 : ℝ × ℝ) (x1 x2 y1 y2 : ℝ) :
  (F1 = (0, -2)) ∧ ((y1 = x1 - 2) ∧ (y2 = x2 - 2) ∧ ((x1 = 0) ∧ (y1 = -2)) ∧ ((x2 = 8/3) ∧ (y2 = 2/3))) →
  (1/2) * (((x1 - x2)^2 + (y1 - y2)^2)^(1/2)) * (|(-2-2)/((2)^(1/2))|) = 16 / 3 :=
by sorry

end length_of_AB_area_of_ΔABF1_l33_33944


namespace remaining_lemons_proof_l33_33315

-- Definitions for initial conditions
def initial_lemons_first_tree   := 15
def initial_lemons_second_tree  := 20
def initial_lemons_third_tree   := 25

def sally_picked_first_tree     := 7
def mary_picked_second_tree     := 9
def tom_picked_first_tree       := 12

def lemons_fell_each_tree       := 4
def animals_eaten_per_tree      := lemons_fell_each_tree / 2

-- Definitions for intermediate calculations
def remaining_lemons_first_tree_full := initial_lemons_first_tree - sally_picked_first_tree - tom_picked_first_tree
def remaining_lemons_first_tree      := if remaining_lemons_first_tree_full < 0 then 0 else remaining_lemons_first_tree_full

def remaining_lemons_second_tree := initial_lemons_second_tree - mary_picked_second_tree

def mary_picked_third_tree := (remaining_lemons_second_tree : ℚ) / 2
def remaining_lemons_third_tree_full := (initial_lemons_third_tree : ℚ) - mary_picked_third_tree
def remaining_lemons_third_tree      := Nat.floor remaining_lemons_third_tree_full

-- Adjusting for fallen and eaten lemons
def final_remaining_lemons_first_tree_full := remaining_lemons_first_tree - lemons_fell_each_tree + animals_eaten_per_tree
def final_remaining_lemons_first_tree      := if final_remaining_lemons_first_tree_full < 0 then 0 else final_remaining_lemons_first_tree_full

def final_remaining_lemons_second_tree     := remaining_lemons_second_tree - lemons_fell_each_tree + animals_eaten_per_tree

def final_remaining_lemons_third_tree_full := remaining_lemons_third_tree - lemons_fell_each_tree + animals_eaten_per_tree
def final_remaining_lemons_third_tree      := if final_remaining_lemons_third_tree_full < 0 then 0 else final_remaining_lemons_third_tree_full

-- Lean 4 statement to prove the equivalence
theorem remaining_lemons_proof :
  final_remaining_lemons_first_tree = 0 ∧
  final_remaining_lemons_second_tree = 9 ∧
  final_remaining_lemons_third_tree = 18 :=
by
  -- The proof is omitted as per the requirement
  sorry

end remaining_lemons_proof_l33_33315


namespace remainder_when_divided_by_7_l33_33028

theorem remainder_when_divided_by_7 (k : ℕ) (h1 : k % 5 = 2) (h2 : k % 6 = 5) : k % 7 = 3 :=
sorry

end remainder_when_divided_by_7_l33_33028


namespace cloak_change_l33_33455

theorem cloak_change (silver_for_cloak1 silver_for_cloak2 : ℕ) (gold_change1 gold_change2 : ℕ) (silver_change : ℕ) :
  silver_for_cloak1 = 20 →
  gold_change1 = 4 →
  silver_for_cloak2 = 15 →
  gold_change2 = 1 →
  ∃ silver_cost_of_cloak : ℕ, 
    silver_cost_of_cloak = (20 - 4) * (5 / 3) →
    silver_change = 10 →
    14 * (5 / 3) - 8 = silver_change :=
by 
  assume h1 h2 h3 h4,
  use 16, 
  sorry

end cloak_change_l33_33455


namespace jeffery_fish_count_l33_33823

variable (J R Y : ℕ)

theorem jeffery_fish_count :
  (R = 3 * J) → (Y = 2 * R) → (J + R + Y = 100) → (Y = 60) :=
by
  intros hR hY hTotal
  have h1 : R = 3 * J := hR
  have h2 : Y = 2 * R := hY
  rw [h1, h2] at hTotal
  sorry

end jeffery_fish_count_l33_33823


namespace num_terminating_decimals_l33_33788

theorem num_terminating_decimals (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℕ, k = 20) := by
  have h2 : ∀ n, (1 ≤ n ∧ n ≤ 180) → (∃ k, k = (180 / 9)) := sorry
  have h3 : 180 / 9 = 20 := by norm_num
  use 20
  exact h3

end num_terminating_decimals_l33_33788


namespace pages_left_proof_l33_33307

-- Definitions based on the given conditions
def initial_books : ℕ := 10
def pages_per_book : ℕ := 100
def lost_books : ℕ := 2

-- Calculate the number of books left
def books_left : ℕ := initial_books - lost_books

-- Calculate the total number of pages left
def total_pages_left : ℕ := books_left * pages_per_book

-- Theorem statement representing the proof problem
theorem pages_left_proof : total_pages_left = 800 := by
  -- Call the necessary built-ins and perform the calculation steps within the proof.
  simp [total_pages_left, books_left, initial_books, lost_books, pages_per_book]
  sorry -- Proof steps will go here, or use a by computation or decidability tactics


end pages_left_proof_l33_33307


namespace root_conditions_l33_33812

theorem root_conditions (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ |x1^2 - 5 * x1| = a ∧ |x2^2 - 5 * x2| = a) ↔ (a = 0 ∨ a > 25 / 4) := 
by 
  sorry

end root_conditions_l33_33812


namespace distinct_three_digit_numbers_count_l33_33257

theorem distinct_three_digit_numbers_count :
  ∃ (numbers : Finset (Fin 1000)), (∀ n ∈ numbers, (n / 100) < 5 ∧ (n / 10 % 10) < 5 ∧ (n % 10) < 5 ∧ 
  (n / 100) ≠ (n / 10 % 10) ∧ (n / 100) ≠ (n % 10) ∧ (n / 10 % 10) ≠ (n % 10)) ∧ numbers.card = 60 := 
sorry

end distinct_three_digit_numbers_count_l33_33257


namespace largest_number_2013_l33_33731

theorem largest_number_2013 (x y : ℕ) (h1 : x + y = 2013)
    (h2 : y = 5 * (x / 100 + 1)) : max x y = 1913 := by
  sorry

end largest_number_2013_l33_33731


namespace conditional_probability_B_given_A_l33_33440

open Probability

noncomputable def problem := 
  let μ : ℝ := 100
  let σ : ℝ := 10
  let X : MeasureTheory.Measure ℝ := MeasureTheory.Measure.normal μ σ
  let A : Set ℝ := {x | 80 < x ∧ x ≤ 100}
  let B : Set ℝ := {x | 70 < x ∧ x ≤ 90}
  P(X|A).prob B = 27 / 95

theorem conditional_probability_B_given_A : problem :=
  begin
    sorry
  end

end conditional_probability_B_given_A_l33_33440


namespace modulus_of_complex_z_l33_33252

open Complex

theorem modulus_of_complex_z (z : ℂ) (h : z * (2 - 3 * I) = 6 + 4 * I) : 
  Complex.abs z = 2 * Real.sqrt 313 / 13 :=
by
  sorry

end modulus_of_complex_z_l33_33252


namespace find_x_l33_33992

def operation_eur (x y : ℕ) : ℕ := 3 * x * y

theorem find_x (y x : ℕ) (h1 : y = 3) (h2 : operation_eur y (operation_eur x 5) = 540) : x = 4 :=
by
  sorry

end find_x_l33_33992


namespace value_range_of_quadratic_function_l33_33011

def quadratic_function (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem value_range_of_quadratic_function :
  (∀ x : ℝ, 1 < x ∧ x ≤ 4 → -1 < quadratic_function x ∧ quadratic_function x ≤ 3) :=
sorry

end value_range_of_quadratic_function_l33_33011


namespace coplanar_vectors_set_B_l33_33422

variables {V : Type*} [AddCommGroup V] [Module ℝ V] 
variables (a b c : V)

theorem coplanar_vectors_set_B
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ (k₁ k₂ : ℝ), 
    k₁ • (2 • a + b) + k₂ • (a + b + c) = 7 • a + 5 • b + 3 • c :=
by { sorry }

end coplanar_vectors_set_B_l33_33422


namespace probability_of_at_most_3_heads_l33_33591

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l33_33591


namespace minimum_value_x_plus_4y_l33_33652

variable {x y : ℝ}

-- Define the conditions
axiom hx : x > 0
axiom hy : y > 0
axiom hxy : x + y = 2 * x * y

-- Define the proof statement
theorem minimum_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2 * x * y) : x + 4 * y = 9 / 2 := sorry

end minimum_value_x_plus_4y_l33_33652


namespace inequality_solution_l33_33007

theorem inequality_solution (x : ℝ) :
  (2 / (x - 3) ≤ 5) ↔ (x < 3 ∨ x ≥ 17 / 5) := 
sorry

end inequality_solution_l33_33007


namespace part1_part2_part3_l33_33281

theorem part1 {k b : ℝ} (h₀ : k ≠ 0) (h₁ : b = 1) (h₂ : k + b = 2) : 
  ∀ x : ℝ, y = k * x + b → y = x + 1 :=
by sorry

theorem part2 : ∃ (C : ℝ × ℝ), 
  C.1 = 3 ∧ C.2 = 4 ∧ y = x + 1 :=
by sorry

theorem part3 {n k : ℝ} (h₀ : k ≠ 0) 
  (h₁ : ∀ x : ℝ, x < 3 → (2 / 3) * x + n > x + 1 ∧ (2 / 3) * x + n < 4) 
  (h₂ : ∀ x : ℝ, y = (2 / 3) * x + n → y = 4 ∧ x = 3) :
  n = 2 :=
by sorry

end part1_part2_part3_l33_33281


namespace no_integers_satisfy_eq_l33_33839

theorem no_integers_satisfy_eq (m n : ℤ) : m^2 ≠ n^5 - 4 := 
by {
  sorry
}

end no_integers_satisfy_eq_l33_33839


namespace equation_solution_l33_33622

theorem equation_solution :
  ∃ a b c d : ℤ, a > 0 ∧ (∀ x : ℝ, (64 * x^2 + 96 * x - 36) = (a * x + b)^2 + d) ∧ c = -36 ∧ a + b + c + d = -94 :=
by sorry

end equation_solution_l33_33622


namespace number_of_roses_cut_l33_33351

-- Let's define the initial and final conditions
def initial_roses : ℕ := 6
def final_roses : ℕ := 16

-- Define the number of roses Mary cut from her garden
def roses_cut := final_roses - initial_roses

-- Now, we state the theorem we aim to prove
theorem number_of_roses_cut : roses_cut = 10 :=
by
  -- Proof goes here
  sorry

end number_of_roses_cut_l33_33351


namespace buy_items_ways_l33_33121

theorem buy_items_ways (headphones keyboards mice keyboard_mouse_sets headphone_mouse_sets : ℕ) :
  headphones = 9 → keyboards = 5 → mice = 13 → keyboard_mouse_sets = 4 → headphone_mouse_sets = 5 →
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 :=
by
  intros h_eq k_eq m_eq kms_eq hms_eq
  have h_eq_gen : headphones = 9 := h_eq
  have k_eq_gen : keyboards = 5 := k_eq
  have m_eq_gen : mice = 13 := m_eq
  have kms_eq_gen : keyboard_mouse_sets = 4 := kms_eq
  have hms_eq_gen : headphone_mouse_sets = 5 := hms_eq
  sorry

end buy_items_ways_l33_33121


namespace ratio_in_two_years_l33_33753

def son_age : ℕ := 22
def man_age : ℕ := son_age + 24

theorem ratio_in_two_years :
  (man_age + 2) / (son_age + 2) = 2 := 
sorry

end ratio_in_two_years_l33_33753


namespace terminating_decimals_l33_33780

theorem terminating_decimals (n : ℤ) (h : 1 ≤ n ∧ n ≤ 180) :
  (∃ k : ℤ, k = 20 ∧ ∀ n, 1 ≤ n ∧ n ≤ 180 → (∃ m, m * 180 = n * (2^2 * 5))) := by
  sorry

end terminating_decimals_l33_33780


namespace total_ways_to_buy_l33_33125

-- Definitions:
def h : ℕ := 9  -- number of headphones
def m : ℕ := 13 -- number of mice
def k : ℕ := 5  -- number of keyboards
def s_km : ℕ := 4 -- number of sets of keyboard and mouse
def s_hm : ℕ := 5 -- number of sets of headphones and mouse

-- Theorem stating the number of ways to buy three items is 646.
theorem total_ways_to_buy : (s_km * h) + (s_hm * k) + (h * m * k) = 646 := by
  -- Placeholder proof using sorry.
  sorry

end total_ways_to_buy_l33_33125


namespace pies_differ_in_both_l33_33271

-- Definitions of pie types
inductive Filling
| apple
| cherry

inductive PreparationMethod
| fried
| baked

structure Pie where
  filling : Filling
  method : PreparationMethod

-- The set of all possible pie types
def pies : Set Pie := {
  {filling := Filling.apple, method := PreparationMethod.fried},
  {filling := Filling.cherry, method := PreparationMethod.fried},
  {filling := Filling.apple, method := PreparationMethod.baked},
  {filling := Filling.cherry, method := PreparationMethod.baked}
}

-- The statement to prove: If there are at least three types of pies available, then there exist two pies that differ in both filling and preparation method.
theorem pies_differ_in_both :
  ∃ (p1 p2 : Pie), p1 ∈ pies ∧ p2 ∈ pies ∧ p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.method ≠ p2.method :=
begin
  sorry
end

end pies_differ_in_both_l33_33271


namespace value_to_subtract_l33_33813

theorem value_to_subtract (N x : ℕ) 
  (h1 : (N - x) / 7 = 7) 
  (h2 : (N - 2) / 13 = 4) : x = 5 :=
by 
  sorry

end value_to_subtract_l33_33813


namespace sum_of_edges_96_l33_33510

noncomputable def volume (a r : ℝ) : ℝ := 
  (a / r) * a * (a * r)

noncomputable def surface_area (a r : ℝ) : ℝ := 
  2 * ((a^2) / r + a^2 + a^2 * r)

noncomputable def sum_of_edges (a r : ℝ) : ℝ := 
  4 * ((a / r) + a + (a * r))

theorem sum_of_edges_96 :
  (∃ (a r : ℝ), volume a r = 512 ∧ surface_area a r = 384 ∧ sum_of_edges a r = 96) :=
by
  have a := 8
  have r := 1
  have h_volume : volume a r = 512 := sorry
  have h_surface_area : surface_area a r = 384 := sorry
  have h_sum_of_edges : sum_of_edges a r = 96 := sorry
  exact ⟨a, r, h_volume, h_surface_area, h_sum_of_edges⟩

end sum_of_edges_96_l33_33510


namespace no_common_points_l33_33804

noncomputable def f (a x : ℝ) : ℝ := x^2 - a * x
noncomputable def g (a b x : ℝ) : ℝ := b + a * Real.log (x - 1)
noncomputable def h (a x : ℝ) : ℝ := x^2 - a * x - a * Real.log (x - 1)
noncomputable def G (a : ℝ) : ℝ := -a^2 / 4 + 1 - a * Real.log (a / 2)

theorem no_common_points (a b : ℝ) (h1 : 1 ≤ a) :
  (∀ x > 1, f a x ≠ g a b x) ↔ b < 3 / 4 + Real.log 2 :=
by
  sorry

end no_common_points_l33_33804


namespace number_of_numbers_l33_33713

theorem number_of_numbers 
  (avg : ℚ) (avg1 : ℚ) (avg2 : ℚ) (avg3 : ℚ)
  (h_avg : avg = 4.60) 
  (h_avg1 : avg1 = 3.4) 
  (h_avg2 : avg2 = 3.8) 
  (h_avg3 : avg3 = 6.6) 
  (h_sum_eq : 2 * avg1 + 2 * avg2 + 2 * avg3 = 27.6) : 
  (27.6 / avg = 6) := 
  by sorry

end number_of_numbers_l33_33713


namespace union_M_N_intersection_M_complement_N_l33_33948

open Set

variable (U : Set ℝ) (M N : Set ℝ)

-- Define the universal set
def is_universal_set (U : Set ℝ) : Prop :=
  U = univ

-- Define the set M
def is_set_M (M : Set ℝ) : Prop :=
  M = {x | ∃ y, y = (x - 2).sqrt}  -- or equivalently x ≥ 2

-- Define the set N
def is_set_N (N : Set ℝ) : Prop :=
  N = {x | x < 1 ∨ x > 3}

-- Define the complement of N in U
def complement_set_N (U N : Set ℝ) : Set ℝ :=
  U \ N

-- Prove M ∪ N = {x | x < 1 ∨ x ≥ 2}
theorem union_M_N (U : Set ℝ) (M N : Set ℝ) (hU : is_universal_set U) (hM : is_set_M M) (hN : is_set_N N) :
  M ∪ N = {x | x < 1 ∨ x ≥ 2} :=
  sorry

-- Prove M ∩ (complement of N in U) = {x | 2 ≤ x ≤ 3}
theorem intersection_M_complement_N (U : Set ℝ) (M N : Set ℝ) (hU : is_universal_set U) (hM : is_set_M M) (hN : is_set_N N) :
  M ∩ (complement_set_N U N) = {x | 2 ≤ x ∧ x ≤ 3} :=
  sorry

end union_M_N_intersection_M_complement_N_l33_33948


namespace sam_walked_distance_when_meeting_l33_33933

variable (D_s D_f : ℝ)
variable (t : ℝ)

theorem sam_walked_distance_when_meeting
  (h1 : 55 = D_f + D_s)
  (h2 : D_f = 6 * t)
  (h3 : D_s = 5 * t) :
  D_s = 25 :=
by 
  -- This is where the proof would go
  sorry

end sam_walked_distance_when_meeting_l33_33933


namespace distinct_ways_to_place_digits_l33_33427

theorem distinct_ways_to_place_digits : 
    ∃ n : ℕ, 
    n = 120 ∧ 
    n = nat.factorial 5 := 
by
  sorry

end distinct_ways_to_place_digits_l33_33427


namespace infinite_geometric_series_sum_l33_33064

-- Definitions of first term and common ratio
def a : ℝ := 1 / 5
def r : ℝ := 1 / 2

-- Proposition to prove that the sum of the geometric series is 2/5
theorem infinite_geometric_series_sum : (a / (1 - r)) = (2 / 5) :=
by 
  -- Sorry is used here as a placeholder for the proof.
  sorry

end infinite_geometric_series_sum_l33_33064


namespace product_of_five_integers_l33_33392

theorem product_of_five_integers (E F G H I : ℚ)
  (h1 : E + F + G + H + I = 110)
  (h2 : E / 2 = F / 3 ∧ F / 3 = G * 4 ∧ G * 4 = H * 2 ∧ H * 2 = I - 5) :
  E * F * G * H * I = 623400000 / 371293 := by
  sorry

end product_of_five_integers_l33_33392


namespace trapezium_other_side_length_l33_33183

theorem trapezium_other_side_length 
  (side1 : ℝ) (perpendicular_distance : ℝ) (area : ℝ) (side1_val : side1 = 5) 
  (perpendicular_distance_val : perpendicular_distance = 6) (area_val : area = 27) : 
  ∃ other_side : ℝ, other_side = 4 :=
by
  sorry

end trapezium_other_side_length_l33_33183


namespace no_intersection_abs_functions_l33_33417

open Real

theorem no_intersection_abs_functions : 
  ∀ f g : ℝ → ℝ, 
  (∀ x, f x = |2 * x + 5|) → 
  (∀ x, g x = -|3 * x - 2|) → 
  (∀ y, ∀ x1 x2, f x1 = y ∧ g x2 = y → y = 0 ∧ x1 = -5/2 ∧ x2 = 2/3 → (x1 ≠ x2)) → 
  (∃ x, f x = g x) → 
  false := 
  by
    intro f g hf hg h
    sorry

end no_intersection_abs_functions_l33_33417


namespace Sally_bought_20_pokemon_cards_l33_33840

theorem Sally_bought_20_pokemon_cards
  (initial_cards : ℕ)
  (cards_from_dan : ℕ)
  (total_cards : ℕ)
  (bought_cards : ℕ)
  (h1 : initial_cards = 27)
  (h2 : cards_from_dan = 41)
  (h3 : total_cards = 88)
  (h4 : total_cards = initial_cards + cards_from_dan + bought_cards) :
  bought_cards = 20 := 
by
  sorry

end Sally_bought_20_pokemon_cards_l33_33840


namespace probability_at_most_3_heads_l33_33577

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l33_33577


namespace two_pies_differ_l33_33269

theorem two_pies_differ (F_A F_C B_A B_C : Bool) :
  (F_A ∨ F_C ∨ B_A ∨ B_C) →
  (F_A ∧ F_C ∧ B_A ∧ B_C) ∧ 
  (∀ a b, (a ≠ b) → (a.filling ≠ b.filling ∧ a.preparation ≠ b.preparation)) :=
by
  intros H1 H2
  sorry

end two_pies_differ_l33_33269


namespace decimal_digits_of_fraction_l33_33739

noncomputable def fraction : ℚ := 987654321 / (2 ^ 30 * 5 ^ 2)

theorem decimal_digits_of_fraction :
  ∃ n ≥ 30, fraction = (987654321 / 10^2) / 2^28 := sorry

end decimal_digits_of_fraction_l33_33739


namespace probability_heads_at_most_3_of_10_coins_flipped_l33_33547

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l33_33547


namespace smallest_n_l33_33380

/--
Each of \( 2020 \) boxes in a line contains 2 red marbles, 
and for \( 1 \le k \le 2020 \), the box in the \( k \)-th 
position also contains \( k \) white marbles. 

Let \( Q(n) \) be the probability that James stops after 
drawing exactly \( n \) marbles. Prove that the smallest 
value of \( n \) for which \( Q(n) < \frac{1}{2020} \) 
is 31.
-/
theorem smallest_n (Q : ℕ → ℚ) (hQ : ∀ n, Q n = (2 : ℚ) / ((n + 1) * (n + 2)))
  : ∃ n, Q n < 1/2020 ∧ ∀ m < n, Q m ≥ 1/2020 := by
  sorry

end smallest_n_l33_33380


namespace probability_same_color_two_balls_cheryl_draws_l33_33013

theorem probability_same_color_two_balls_cheryl_draws 
  (balls : Finset (Fin 6))
  (colors : balls → Fin 3)
  (c1 c2 c3 : Finset (Fin 6))
  (disjoint_c1_c2 : Disjoint c1 c2)
  (disjoint_c1_c3 : Disjoint c1 c3)
  (disjoint_c2_c3 : Disjoint c2 c3)
  (hc1 : c1.card = 2)
  (hc2 : c2.card = 2)
  (hc3 : c3.card = 2)
  (hc_union : c1 ∪ c2 ∪ c3 = balls)
  :
  (↑((c3.filter (λ i, colors i = colors (c3.choose (c3.card-1)))).card) / (c3.card.choose 2 : ℝ) = (1 / 5 : ℝ)) :=
by sorry

end probability_same_color_two_balls_cheryl_draws_l33_33013


namespace find_c_k_l33_33006

noncomputable def common_difference (a : ℕ → ℕ) : ℕ := sorry
noncomputable def common_ratio (b : ℕ → ℕ) : ℕ := sorry
noncomputable def arith_seq (d : ℕ) (n : ℕ) : ℕ := 1 + (n - 1) * d
noncomputable def geom_seq (r : ℕ) (n : ℕ) : ℕ := r^(n - 1)
noncomputable def combined_seq (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) : ℕ := a n + b n

variable (k : ℕ) (d : ℕ) (r : ℕ)

-- Conditions
axiom arith_condition : common_difference (arith_seq d) = d
axiom geom_condition : common_ratio (geom_seq r) = r
axiom combined_k_minus_1 : combined_seq (arith_seq d) (geom_seq r) (k - 1) = 50
axiom combined_k_plus_1 : combined_seq (arith_seq d) (geom_seq r) (k + 1) = 1500

-- Prove that c_k = 2406
theorem find_c_k : combined_seq (arith_seq d) (geom_seq r) k = 2406 := by
  sorry

end find_c_k_l33_33006


namespace marble_arrangement_l33_33468

theorem marble_arrangement :
  let blue := 6
  let yellow := 16
  let total := blue + yellow
  let arrangements := Nat.choose total blue
  arrangements % 1000 = 8 :=
by
  let blue := 6
  let yellow := 16
  let total := blue + yellow
  let arrangements := Nat.choose total blue
  have h1 : arrangements = 8008 := by sorry
  have h2 : 8008 % 1000 = 8 := by norm_num
  rw [h1, h2]
  rfl

end marble_arrangement_l33_33468


namespace lines_intersect_and_find_point_l33_33818

theorem lines_intersect_and_find_point (n : ℝ)
  (h₁ : ∀ t : ℝ, ∃ (x y z : ℝ), x / 2 = t ∧ y / -3 = t ∧ z / n = t)
  (h₂ : ∀ t : ℝ, ∃ (x y z : ℝ), (x + 1) / 3 = t ∧ (y + 5) / 2 = t ∧ z / 1 = t) :
  n = 1 ∧ (∃ (x y z : ℝ), x = 2 ∧ y = -3 ∧ z = 1) :=
sorry

end lines_intersect_and_find_point_l33_33818


namespace fair_coin_flip_probability_difference_l33_33201

noncomputable def choose : ℕ → ℕ → ℕ
| _, 0 => 1
| 0, _ => 0
| n + 1, r + 1 => choose n r + choose n (r + 1)

theorem fair_coin_flip_probability_difference :
  let p := 1 / 2
  let p_heads_4_out_of_5 := choose 5 4 * p^4 * p
  let p_heads_5_out_of_5 := p^5
  p_heads_4_out_of_5 - p_heads_5_out_of_5 = 1 / 8 :=
by
  sorry

end fair_coin_flip_probability_difference_l33_33201


namespace Janet_earnings_l33_33285

/--
Janet works as an exterminator and also sells molten metal casts of fire ant nests on the Internet.
She gets paid $70 an hour for exterminator work and makes $20/pound on her ant nest sculptures.
Given that she does 20 hours of exterminator work and sells a 5-pound sculpture and a 7-pound sculpture,
prove that Janet's total earnings are $1640.
-/
theorem Janet_earnings :
  let hourly_rate_exterminator := 70
  let hours_worked := 20
  let rate_per_pound := 20
  let sculpture_one_weight := 5
  let sculpture_two_weight := 7

  let exterminator_earnings := hourly_rate_exterminator * hours_worked
  let total_sculpture_weight := sculpture_one_weight + sculpture_two_weight
  let sculpture_earnings := rate_per_pound * total_sculpture_weight

  let total_earnings := exterminator_earnings + sculpture_earnings
  total_earnings = 1640 := 
by
  sorry

end Janet_earnings_l33_33285


namespace betty_books_l33_33373

variable (B : ℝ)
variable (h : B + (5/4) * B = 45)

theorem betty_books : B = 20 := by
  sorry

end betty_books_l33_33373


namespace cost_increase_per_scrap_rate_l33_33994

theorem cost_increase_per_scrap_rate (x : ℝ) :
  ∀ x Δx, y = 56 + 8 * x → Δx = 1 → y + Δy = 56 + 8 * (x + Δx) → Δy = 8 :=
by
  sorry

end cost_increase_per_scrap_rate_l33_33994


namespace avg_and_variance_decrease_l33_33340

noncomputable def original_heights : List ℝ := [180, 184, 188, 190, 192, 194]
noncomputable def new_heights : List ℝ := [180, 184, 188, 190, 192, 188]

noncomputable def avg (heights : List ℝ) : ℝ :=
  heights.sum / heights.length

noncomputable def variance (heights : List ℝ) (mean : ℝ) : ℝ :=
  (heights.map (λ h => (h - mean) ^ 2)).sum / heights.length

theorem avg_and_variance_decrease :
  let original_mean := avg original_heights
  let new_mean := avg new_heights
  let original_variance := variance original_heights original_mean
  let new_variance := variance new_heights new_mean
  new_mean < original_mean ∧ new_variance < original_variance :=
by
  sorry

end avg_and_variance_decrease_l33_33340


namespace fraction_equality_l33_33742

theorem fraction_equality (a b c : ℝ) (hc : c ≠ 0) (h : a / c = b / c) : a = b := 
by
  sorry

end fraction_equality_l33_33742


namespace total_ways_to_buy_l33_33127

-- Definitions:
def h : ℕ := 9  -- number of headphones
def m : ℕ := 13 -- number of mice
def k : ℕ := 5  -- number of keyboards
def s_km : ℕ := 4 -- number of sets of keyboard and mouse
def s_hm : ℕ := 5 -- number of sets of headphones and mouse

-- Theorem stating the number of ways to buy three items is 646.
theorem total_ways_to_buy : (s_km * h) + (s_hm * k) + (h * m * k) = 646 := by
  -- Placeholder proof using sorry.
  sorry

end total_ways_to_buy_l33_33127


namespace sqrt_expression_eq_seven_div_two_l33_33234

theorem sqrt_expression_eq_seven_div_two :
  (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1 / 2) * Real.sqrt 12 / Real.sqrt 24) = 7 / 2 :=
by
  sorry

end sqrt_expression_eq_seven_div_two_l33_33234


namespace sally_bread_consumption_l33_33709

theorem sally_bread_consumption :
  (2 * 2) + (1 * 2) = 6 :=
by
  sorry

end sally_bread_consumption_l33_33709


namespace fair_coin_flip_probability_difference_l33_33202

noncomputable def choose : ℕ → ℕ → ℕ
| _, 0 => 1
| 0, _ => 0
| n + 1, r + 1 => choose n r + choose n (r + 1)

theorem fair_coin_flip_probability_difference :
  let p := 1 / 2
  let p_heads_4_out_of_5 := choose 5 4 * p^4 * p
  let p_heads_5_out_of_5 := p^5
  p_heads_4_out_of_5 - p_heads_5_out_of_5 = 1 / 8 :=
by
  sorry

end fair_coin_flip_probability_difference_l33_33202


namespace different_numerators_count_l33_33830

-- Define the set of all rational numbers r of the form 0.ab repeated
def T : Set ℚ := { r | ∃ (a b : ℤ), (0 ≤ a) ∧ (a ≤ 9) ∧ (0 ≤ b) ∧ (b ≤ 9) ∧
                                      r = (10 * a + b) / 99 ∧ 0 < r ∧ r < 1 }

-- Define the predicate that checks if two integers are coprime
def coprime (m n : ℤ) : Prop := Int.gcd m n = 1

-- Define the top-level Lean theorem
theorem different_numerators_count :
  ∃ n : ℕ, n = 60 ∧ ∀ r ∈ T, ∃ p q : ℤ, coprime p q ∧ r = p / q ∧ q = 99 → p ∉ {1, 2, ..., 99} := sorry

end different_numerators_count_l33_33830


namespace terminating_decimals_l33_33774

theorem terminating_decimals (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k : ℕ, k = 20 ∧ ∀ m : ℕ, 1 ≤ m ∧ m ≤ 180 → (decimal_terminates (m / 180) ↔ m % 9 = 0)) :=
by {
  sorry
}

def decimal_terminates (x : ℚ) : Prop :=
  sorry

end terminating_decimals_l33_33774


namespace C_is_20_years_younger_l33_33010

variable (A B C : ℕ)

-- Conditions from the problem
axiom age_condition : A + B = B + C + 20

-- Theorem representing the proof problem
theorem C_is_20_years_younger : A = C + 20 := sorry

end C_is_20_years_younger_l33_33010


namespace rectangular_prism_faces_l33_33898

theorem rectangular_prism_faces (n : ℕ) (h1 : ∀ z : ℕ, z > 0 → z^3 = 2 * n^3) 
  (h2 : n > 0) :
  (∃ f : ℕ, f = (1 / 6 : ℚ) * (6 * 2 * n^3) ∧ 
    f = 10 * n^2) ↔ n = 5 := by
sorry

end rectangular_prism_faces_l33_33898


namespace symmetric_circle_equation_l33_33062

noncomputable def equation_of_symmetric_circle (C₁ : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, C₁ x y ↔ x^2 + y^2 - 4 * x - 8 * y + 19 = 0

theorem symmetric_circle_equation :
  ∀ (C₁ : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop),
  equation_of_symmetric_circle C₁ l →
  (∀ x y, l x y ↔ x + 2 * y - 5 = 0) →
  ∃ C₂ : ℝ → ℝ → Prop, (∀ x y, C₂ x y ↔ x^2 + y^2 = 1) :=
by
  intros C₁ l hC₁ hₗ
  sorry

end symmetric_circle_equation_l33_33062


namespace silver_coins_change_l33_33442

-- Define the conditions
def condition1 : ℕ × ℕ := (20, 4) -- (20 silver coins, 4 gold coins change)
def condition2 : ℕ × ℕ := (15, 1) -- (15 silver coins, 1 gold coin change)
def cost_of_cloak_in_gold_coins : ℕ := 14

-- Define the theorem to be proven
theorem silver_coins_change (s1 g1 s2 g2 cloak_g : ℕ) (h1 : (s1, g1) = condition1) (h2 : (s2, g2) = condition2) :
  ∃ silver : ℕ, (silver = 10) :=
by {
  sorry
}

end silver_coins_change_l33_33442


namespace toy_store_bears_shelves_l33_33904

theorem toy_store_bears_shelves (initial_stock shipment bears_per_shelf total_bears number_of_shelves : ℕ)
  (h1 : initial_stock = 17)
  (h2 : shipment = 10)
  (h3 : bears_per_shelf = 9)
  (h4 : total_bears = initial_stock + shipment)
  (h5 : number_of_shelves = total_bears / bears_per_shelf) :
  number_of_shelves = 3 :=
by
  sorry

end toy_store_bears_shelves_l33_33904


namespace janet_total_earnings_l33_33284

-- Definitions based on conditions from step a)
def hourly_wage := 70
def hours_worked := 20
def rate_per_pound := 20
def weight_sculpture1 := 5
def weight_sculpture2 := 7

-- Statement for the proof problem
theorem janet_total_earnings : 
  let earnings_from_extermination := hourly_wage * hours_worked
  let earnings_from_sculpture1 := rate_per_pound * weight_sculpture1
  let earnings_from_sculpture2 := rate_per_pound * weight_sculpture2
  earnings_from_extermination + earnings_from_sculpture1 + earnings_from_sculpture2 = 1640 := 
by
  sorry

end janet_total_earnings_l33_33284


namespace right_triangle_area_l33_33516

theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 240 :=
by
  sorry

end right_triangle_area_l33_33516


namespace gcd_228_1995_l33_33354

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := 
by
  sorry

end gcd_228_1995_l33_33354


namespace cousin_typing_time_l33_33974

theorem cousin_typing_time (speed_ratio : ℕ) (my_time_hours : ℕ) (minutes_per_hour : ℕ) (my_time_minutes : ℕ) :
  speed_ratio = 4 →
  my_time_hours = 3 →
  minutes_per_hour = 60 →
  my_time_minutes = my_time_hours * minutes_per_hour →
  ∃ (cousin_time : ℕ), cousin_time = my_time_minutes / speed_ratio := by
  sorry

end cousin_typing_time_l33_33974


namespace compound_interest_rate_l33_33244

theorem compound_interest_rate : 
  let P := 14800
  let interest := 4265.73
  let A := 19065.73
  let t := 2
  let n := 1
  let r := 0.13514
  (P : ℝ) * (1 + r)^t = A :=
by
-- Here we will provide the steps of the proof
sorry

end compound_interest_rate_l33_33244


namespace total_cases_after_three_weeks_l33_33836

-- Definitions and conditions directly from the problem
def week1_cases : ℕ := 5000
def week2_cases : ℕ := week1_cases / 2
def week3_cases : ℕ := week2_cases + 2000
def total_cases : ℕ := week1_cases + week2_cases + week3_cases

-- The theorem to prove
theorem total_cases_after_three_weeks :
  total_cases = 12000 := 
by
  -- Sorry allows us to skip the actual proof
  sorry

end total_cases_after_three_weeks_l33_33836


namespace ratio_eq_23_over_28_l33_33206

theorem ratio_eq_23_over_28 (a b : ℚ) (h : (12 * a - 5 * b) / (14 * a - 3 * b) = 4 / 7) : 
  a / b = 23 / 28 := 
sorry

end ratio_eq_23_over_28_l33_33206


namespace solution_l33_33381

noncomputable def problem_statement (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

theorem solution (f : ℝ → ℝ) (h : problem_statement f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b * x^2 := by
  sorry

end solution_l33_33381


namespace inequality_holds_equality_condition_l33_33291

variables {x y z : ℝ}
-- Assuming positive real numbers and the given condition
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom pos_z : 0 < z
axiom h : x * y + y * z + z * x = x + y + z

theorem inequality_holds : 
  (1 / (x^2 + y + 1)) + (1 / (y^2 + z + 1)) + (1 / (z^2 + x + 1)) ≤ 1 :=
by
  sorry

theorem equality_condition : 
  (1 / (x^2 + y + 1)) + (1 / (y^2 + z + 1)) + (1 / (z^2 + x + 1)) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end inequality_holds_equality_condition_l33_33291


namespace valid_choice_count_l33_33764

def is_valid_base_7_digit (n : ℕ) : Prop := n < 7
def is_valid_base_8_digit (n : ℕ) : Prop := n < 8
def to_base_10_base_7 (c3 c2 c1 c0 : ℕ) : ℕ := 2401 * c3 + 343 * c2 + 49 * c1 + 7 * c0
def to_base_10_base_8 (d3 d2 d1 d0 : ℕ) : ℕ := 4096 * d3 + 512 * d2 + 64 * d1 + 8 * d0
def is_four_digit_number (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

theorem valid_choice_count :
  ∃ (N : ℕ), is_four_digit_number N →
  ∀ (c3 c2 c1 c0 d3 d2 d1 d0 : ℕ),
    is_valid_base_7_digit c3 → is_valid_base_7_digit c2 → is_valid_base_7_digit c1 → is_valid_base_7_digit c0 →
    is_valid_base_8_digit d3 → is_valid_base_8_digit d2 → is_valid_base_8_digit d1 → is_valid_base_8_digit d0 →
    to_base_10_base_7 c3 c2 c1 c0 = N →
    to_base_10_base_8 d3 d2 d1 d0 = N →
    (to_base_10_base_7 c3 c2 c1 c0 + to_base_10_base_8 d3 d2 d1 d0) % 1000 = (2 * N) % 1000 → N = 20 :=
sorry

end valid_choice_count_l33_33764


namespace min_value_expr_l33_33153

theorem min_value_expr (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ k : ℝ, k = 6 ∧ (∃ a b c : ℝ,
                  0 < a ∧
                  0 < b ∧
                  0 < c ∧
                  (k = (a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a)) :=
sorry

end min_value_expr_l33_33153


namespace cloak_change_in_silver_l33_33458

theorem cloak_change_in_silver :
  (∀ c : ℤ, (20 = c + 4) → (15 = c + 1)) →
  (5 * g = 3) →
  14 * gold / exchange_rate = 10 := 
sorry

end cloak_change_in_silver_l33_33458


namespace proof_a_square_plus_a_plus_one_l33_33419

theorem proof_a_square_plus_a_plus_one (a : ℝ) (h : 2 * (5 - a) * (6 + a) = 100) : a^2 + a + 1 = -19 := 
by 
  sorry

end proof_a_square_plus_a_plus_one_l33_33419


namespace prime_in_A_l33_33289

open Nat

def is_in_A (x : ℕ) : Prop :=
  ∃ (a b : ℤ), x = a^2 + 2*b^2 ∧ a * b ≠ 0

theorem prime_in_A (p : ℕ) [Fact (Nat.Prime p)] (h : is_in_A (p^2)) : is_in_A p :=
  sorry

end prime_in_A_l33_33289


namespace pencil_length_l33_33034

theorem pencil_length :
  let purple := 1.5
  let black := 0.5
  let blue := 2
  purple + black + blue = 4 := by sorry

end pencil_length_l33_33034


namespace div_equiv_l33_33001

theorem div_equiv : (0.75 / 25) = (7.5 / 250) :=
by
  sorry

end div_equiv_l33_33001


namespace scientific_notation_of_4370000_l33_33638

theorem scientific_notation_of_4370000 :
  4370000 = 4.37 * 10^6 :=
sorry

end scientific_notation_of_4370000_l33_33638


namespace probability_not_equal_genders_l33_33303

noncomputable def probability_more_grandsons_or_more_granddaughters : ℚ :=
  let total_ways := 2 ^ 12
  let equal_distribution_ways := (Nat.choose 12 6)
  let probability_equal := (equal_distribution_ways : ℚ) / (total_ways : ℚ)
  1 - probability_equal

theorem probability_not_equal_genders (n : ℕ) (p : ℚ) (hp : p = 1 / 2) (hn : n = 12) :
  probability_more_grandsons_or_more_granddaughters = 793 / 1024 :=
by
  sorry

end probability_not_equal_genders_l33_33303


namespace wombat_clawing_l33_33378

variable (W : ℕ)
variable (R : ℕ := 1)

theorem wombat_clawing :
    (9 * W + 3 * R = 39) → (W = 4) :=
by 
  sorry

end wombat_clawing_l33_33378


namespace distinct_ways_to_place_digits_l33_33426

theorem distinct_ways_to_place_digits :
  let digits := {1, 2, 3, 4}
  let boxes := 5
  let empty_box := 1
  -- There are 5! permutations of the list [0, 1, 2, 3, 4]
  let total_digits := insert 0 digits
  -- Resulting in 120 ways to place these digits in 5 boxes
  nat.factorial boxes = 120 :=
by 
  sorry

end distinct_ways_to_place_digits_l33_33426


namespace inequalities_count_three_l33_33155

theorem inequalities_count_three
  (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : x^2 < a^2) (h2 : y^3 < b^3) :
  (x^2 + y^2 < a^2 + b^2) ∧ ¬(x^2 - y^2 < a^2 - b^2) ∧ (x^2 * y^3 < a^2 * b^3) ∧ (x^2 / y^3 < a^2 / b^3) := 
sorry

end inequalities_count_three_l33_33155


namespace find_c_l33_33644

theorem find_c (x y c : ℝ) (h : x = 5 * y) (h2 : 7 * x + 4 * y = 13 * c) : c = 3 * y :=
by
  sorry

end find_c_l33_33644


namespace mean_equivalence_l33_33991

theorem mean_equivalence {x : ℚ} :
  (8 + 15 + 21) / 3 = (18 + x) / 2 → x = 34 / 3 :=
by
  sorry

end mean_equivalence_l33_33991


namespace necessary_but_not_sufficient_l33_33938

theorem necessary_but_not_sufficient (a : ℝ) : (a < 2 → a^2 < 2 * a) ∧ (a^2 < 2 * a → 0 < a ∧ a < 2) := sorry

end necessary_but_not_sufficient_l33_33938


namespace cube_inequality_of_greater_l33_33083

variable (a b : ℝ)

theorem cube_inequality_of_greater (h : a > b) : a^3 > b^3 :=
sorry

end cube_inequality_of_greater_l33_33083


namespace right_triangle_cos_B_l33_33676

theorem right_triangle_cos_B (A B C : ℝ) (hC : C = 90) (hSinA : Real.sin A = 2 / 3) :
  Real.cos B = 2 / 3 :=
sorry

end right_triangle_cos_B_l33_33676


namespace solve_for_n_l33_33531

theorem solve_for_n (n : ℤ) (h : n + (n + 1) + (n + 2) + (n + 3) = 34) : n = 7 :=
by
  sorry

end solve_for_n_l33_33531


namespace root_polynomial_value_l33_33403

theorem root_polynomial_value (m : ℝ) (h : m^2 + 3 * m - 2022 = 0) : m^3 + 4 * m^2 - 2019 * m - 2023 = -1 :=
  sorry

end root_polynomial_value_l33_33403


namespace daily_sales_volume_selling_price_for_profit_l33_33972

noncomputable def cost_price : ℝ := 40
noncomputable def initial_selling_price : ℝ := 60
noncomputable def initial_sales_volume : ℝ := 20
noncomputable def price_decrease_per_increase : ℝ := 5
noncomputable def volume_increase_per_decrease : ℝ := 10

theorem daily_sales_volume (p : ℝ) (v : ℝ) :
  v = initial_sales_volume + ((initial_selling_price - p) / price_decrease_per_increase) * volume_increase_per_decrease :=
sorry

theorem selling_price_for_profit (p : ℝ) (profit : ℝ) :
  profit = (p - cost_price) * (initial_sales_volume + ((initial_selling_price - p) / price_decrease_per_increase) * volume_increase_per_decrease) → p = 54 :=
sorry

end daily_sales_volume_selling_price_for_profit_l33_33972


namespace travel_time_K_l33_33885

/-
Given that:
1. K's speed is x miles per hour.
2. M's speed is x - 1 miles per hour.
3. K takes 1 hour less than M to travel 60 miles (i.e., 60/x hours).
Prove that K's time to travel 60 miles is 6 hours.
-/
theorem travel_time_K (x : ℝ)
  (h1 : x > 0)
  (h2 : x ≠ 1)
  (h3 : 60 / (x - 1) - 60 / x = 1) :
  60 / x = 6 :=
sorry

end travel_time_K_l33_33885


namespace gcd_8251_6105_l33_33017

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 39 := by
  sorry

end gcd_8251_6105_l33_33017


namespace magic_shop_change_l33_33450

theorem magic_shop_change :
  (∀ (cloak : Type), ∃ price_gold price_silver1 change_gold1 price_silver2 change_gold2, 
  price_silver1 = 20 ∧ change_gold1 = 4 ∧ 
  price_silver2 = 15 ∧ change_gold2 = 1 ∧ 
  price_gold = 14 ∧ 
  ∀ change_silver, 
    (20 - 4) * change_silver = 15 - 1 → -- Relation derived from the conditions
    (14 - (15 - 1) * change_silver / (20 - 4)) * change_silver = 10) := 
sorry

end magic_shop_change_l33_33450


namespace purchase_combinations_correct_l33_33115

theorem purchase_combinations_correct : 
  let headphones : ℕ := 9
  let mice : ℕ := 13
  let keyboards : ℕ := 5
  let sets_keyboard_mouse : ℕ := 4
  let sets_headphones_mouse : ℕ := 5
in 
  (sets_keyboard_mouse * headphones) + 
  (sets_headphones_mouse * keyboards) + 
  (headphones * mice * keyboards) = 646 := 
by
  let headphones := 9
  let mice := 13
  let keyboards := 5
  let sets_keyboard_mouse := 4
  let sets_headphones_mouse := 5
  sorry

end purchase_combinations_correct_l33_33115


namespace distinct_ways_to_place_digits_l33_33425

theorem distinct_ways_to_place_digits :
  let n := 4 -- number of digits
  let k := 5 -- number of boxes
  (k * (n!)) = 120 := by
  sorry

end distinct_ways_to_place_digits_l33_33425


namespace probability_of_at_most_3_heads_l33_33567

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l33_33567


namespace min_value_n_constant_term_l33_33356

-- Define the problem statement
theorem min_value_n_constant_term (n r : ℕ) (h : 2 * n = 5 * r) : n = 5 :=
by sorry

end min_value_n_constant_term_l33_33356


namespace sam_walked_distance_l33_33534

theorem sam_walked_distance
  (distance_apart : ℝ) (fred_speed : ℝ) (sam_speed : ℝ) (t : ℝ)
  (H1 : distance_apart = 35) (H2 : fred_speed = 2) (H3 : sam_speed = 5)
  (H4 : 2 * t + 5 * t = distance_apart) :
  5 * t = 25 :=
by
  -- Lean proof goes here
  sorry

end sam_walked_distance_l33_33534


namespace no_integer_roots_l33_33070

theorem no_integer_roots : ∀ x : ℤ, x^3 - 4 * x^2 - 11 * x + 20 ≠ 0 := 
by
  sorry

end no_integer_roots_l33_33070


namespace probability_of_at_most_3_heads_l33_33588

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l33_33588


namespace sum_of_possible_values_of_x_l33_33505

theorem sum_of_possible_values_of_x :
  let sq_side := (x - 4)
  let rect_length := (x - 5)
  let rect_width := (x + 6)
  let sq_area := (sq_side)^2
  let rect_area := rect_length * rect_width
  (3 * (sq_area) = rect_area) → ∃ (x1 x2 : ℝ), (3 * (x1 - 4) ^ 2 = (x1 - 5) * (x1 + 6)) ∧ (3 * (x2 - 4) ^ 2 = (x2 - 5) * (x2 + 6)) ∧ (x1 + x2 = 12.5) := 
by
  sorry

end sum_of_possible_values_of_x_l33_33505


namespace arithmetic_sum_l33_33916

theorem arithmetic_sum (a₁ an n : ℕ) (h₁ : a₁ = 5) (h₂ : an = 32) (h₃ : n = 10) :
  (n * (a₁ + an)) / 2 = 185 :=
by
  sorry

end arithmetic_sum_l33_33916


namespace general_term_l33_33254

open Nat

noncomputable def S (n : ℕ) : ℕ := 2 * n^2 - 3 * n

theorem general_term (n : ℕ) (hn : n > 0) : (S n - S (n - 1)) = 4 * n - 5 := by
  sorry

end general_term_l33_33254


namespace average_rate_of_change_correct_l33_33164

def f (x : ℝ) : ℝ := 2 * x + 1

theorem average_rate_of_change_correct :
  (f 2 - f 1) / (2 - 1) = 2 :=
by
  sorry

end average_rate_of_change_correct_l33_33164


namespace minimum_value_of_expr_l33_33648

def expr (x : ℝ) : ℝ :=
  1 + (Real.cos ((π * Real.sin (2 * x)) / Real.sqrt 3))^2 + 
      (Real.sin (2 * Real.sqrt 3 * π * Real.cos x))^2

theorem minimum_value_of_expr :
  ∀ x, x ∈ { x | ∃ k : ℤ, x = π * k + π / 6 ∨ x = π * k - π / 6} ∧
  expr x = 1 :=
begin
  sorry
end

end minimum_value_of_expr_l33_33648


namespace probability_at_most_3_heads_l33_33560

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l33_33560


namespace measure_of_angle_F_l33_33181

theorem measure_of_angle_F {D E F : ℝ}
  (isosceles : D = E)
  (angle_F_condition : F = D + 40)
  (sum_of_angles : D + E + F = 180) :
  F = 260 / 3 :=
by
  sorry

end measure_of_angle_F_l33_33181


namespace store_purchase_ways_l33_33111

theorem store_purchase_ways :
  ∃ (headphones sets_hm sets_km keyboards mice : ℕ), 
  headphones = 9 ∧
  mice = 13 ∧
  keyboards = 5 ∧
  sets_km = 4 ∧
  sets_hm = 5 ∧
  (sets_km * headphones + sets_hm * keyboards + headphones * mice * keyboards = 646) :=
by
  use 9, 5, 4, 5, 13
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  split; norm_num
  sorry

end store_purchase_ways_l33_33111


namespace sin_square_range_l33_33934

def range_sin_square_values (α β : ℝ) : Prop :=
  3 * (Real.sin α) ^ 2 - 2 * Real.sin α + 2 * (Real.sin β) ^ 2 = 0

theorem sin_square_range (α β : ℝ) (h : range_sin_square_values α β) :
  0 ≤ (Real.sin α) ^ 2 + (Real.sin β) ^ 2 ∧ 
  (Real.sin α) ^ 2 + (Real.sin β) ^ 2 ≤ 4 / 9 :=
sorry

end sin_square_range_l33_33934


namespace darwin_spending_fraction_l33_33922

theorem darwin_spending_fraction {x : ℝ} (h1 : 600 - 600 * x - (1 / 4) * (600 - 600 * x) = 300) :
  x = 1 / 3 :=
sorry

end darwin_spending_fraction_l33_33922


namespace vans_for_field_trip_l33_33483

-- Definitions based on conditions
def students := 25
def adults := 5
def van_capacity := 5

-- Calculate total number of people
def total_people := students + adults

-- Calculate number of vans needed
def vans_needed := total_people / van_capacity

-- Theorem statement
theorem vans_for_field_trip : vans_needed = 6 := by
  -- Proof would go here
  sorry

end vans_for_field_trip_l33_33483


namespace area_of_parallelogram_l33_33328

variable (b : ℕ)
variable (h : ℕ)
variable (A : ℕ)

-- Condition: The height is twice the base.
def height_twice_base := h = 2 * b

-- Condition: The base is 9.
def base_is_9 := b = 9

-- Condition: The area of the parallelogram is base times height.
def area_formula := A = b * h

-- Question: Prove that the area of the parallelogram is 162.
theorem area_of_parallelogram 
  (h_twice : height_twice_base h b) 
  (b_val : base_is_9 b) 
  (area_form : area_formula A b h): A = 162 := 
sorry

end area_of_parallelogram_l33_33328


namespace fraction_of_married_men_l33_33626

theorem fraction_of_married_men (num_women : ℕ) (num_single_women : ℕ) (num_married_women : ℕ)
  (num_married_men : ℕ) (total_people : ℕ) 
  (h1 : num_single_women = num_women / 4) 
  (h2 : num_married_women = num_women - num_single_women)
  (h3 : num_married_men = num_married_women) 
  (h4 : total_people = num_women + num_married_men) :
  (num_married_men : ℚ) / (total_people : ℚ) = 3 / 7 := 
by 
  sorry

end fraction_of_married_men_l33_33626


namespace sequence_2019_value_l33_33866

theorem sequence_2019_value :
  ∃ a : ℕ → ℤ, (∀ n ≥ 4, a n = a (n-1) * a (n-3)) ∧ a 1 = 1 ∧ a 2 = 1 ∧ a 3 = -1 ∧ a 2019 = -1 :=
by
  sorry

end sequence_2019_value_l33_33866


namespace linear_eq_must_be_one_l33_33265

theorem linear_eq_must_be_one (m : ℝ) : (∀ x y : ℝ, (m + 1) * x + 3 * y ^ m = 5 → (m = 1)) :=
by
  intros x y h
  sorry

end linear_eq_must_be_one_l33_33265


namespace inequality_solution_l33_33323

theorem inequality_solution (x : ℝ) : 
  (0 < (x + 2) / ((x - 3)^3)) ↔ (x < -2 ∨ x > 3)  :=
by
  sorry

end inequality_solution_l33_33323


namespace change_in_expression_is_correct_l33_33431

def change_in_expression (x a : ℝ) : ℝ :=
  if increases : true then (x + a)^2 - 3 - (x^2 - 3)
  else (x - a)^2 - 3 - (x^2 - 3)

theorem change_in_expression_is_correct (x a : ℝ) :
  a > 0 → change_in_expression x a = 2 * a * x + a^2 ∨ change_in_expression x a = -(2 * a * x) + a^2 :=
by
  sorry

end change_in_expression_is_correct_l33_33431


namespace parkway_girls_not_playing_soccer_l33_33536

theorem parkway_girls_not_playing_soccer (total_students boys soccer_students : ℕ) 
    (percent_boys_playing_soccer : ℕ) 
    (h1 : total_students = 420)
    (h2 : boys = 312)
    (h3 : soccer_students = 250)
    (h4 : percent_boys_playing_soccer = 86) :
   (total_students - boys - (soccer_students - soccer_students * percent_boys_playing_soccer / 100)) = 73 :=
by sorry

end parkway_girls_not_playing_soccer_l33_33536


namespace length_AD_l33_33541

theorem length_AD (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 13) (h3 : x * (13 - x) = 36) : x = 4 ∨ x = 9 :=
by sorry

end length_AD_l33_33541


namespace find_phi_l33_33757

theorem find_phi :
  ∀ φ : ℝ, 0 < φ ∧ φ < 90 → 
    (∃θ : ℝ, θ = 144 ∧ θ = 2 * φ ∧ (144 - θ) = 72) → φ = 81 :=
by
  intros φ h1 h2
  sorry

end find_phi_l33_33757


namespace exists_100_digit_number_divisible_by_sum_of_digits_l33_33926

-- Definitions
def is_100_digit_number (n : ℕ) : Prop :=
  10^99 ≤ n ∧ n < 10^100

def no_zero_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_divisible_by_sum_of_digits (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

-- Main theorem statement
theorem exists_100_digit_number_divisible_by_sum_of_digits :
  ∃ n : ℕ, is_100_digit_number n ∧ no_zero_digits n ∧ is_divisible_by_sum_of_digits n :=
sorry

end exists_100_digit_number_divisible_by_sum_of_digits_l33_33926


namespace factorial_expression_simplification_l33_33913

theorem factorial_expression_simplification : (3 * (Nat.factorial 5) + 15 * (Nat.factorial 4)) / (Nat.factorial 6) = 1 := by
  sorry

end factorial_expression_simplification_l33_33913


namespace correct_option_is_A_l33_33532

-- Define the options as terms
def optionA (x : ℝ) := (1/2) * x - 5 * x = 18
def optionB (x : ℝ) := (1/2) * x > 5 * x - 1
def optionC (y : ℝ) := 8 * y - 4
def optionD := 5 - 2 = 3

-- Define a function to check if an option is an equation
def is_equation (option : Prop) : Prop :=
  ∃ (x : ℝ), option = ((1/2) * x - 5 * x = 18)

-- Prove that optionA is the equation
theorem correct_option_is_A : is_equation (optionA x) :=
by
  sorry

end correct_option_is_A_l33_33532


namespace regular_hexagon_interior_angle_deg_l33_33717

theorem regular_hexagon_interior_angle_deg (n : ℕ) (h1 : n = 6) :
  let sum_of_interior_angles : ℕ := (n - 2) * 180
  let each_angle : ℕ := sum_of_interior_angles / n
  each_angle = 120 := by
  sorry

end regular_hexagon_interior_angle_deg_l33_33717


namespace susie_remaining_money_l33_33984

noncomputable def calculate_remaining_money : Float :=
  let weekday_hours := 4.0
  let weekday_rate := 12.0
  let weekdays := 5.0
  let weekend_hours := 2.5
  let weekend_rate := 15.0
  let weekends := 2.0
  let total_weekday_earnings := weekday_hours * weekday_rate * weekdays
  let total_weekend_earnings := weekend_hours * weekend_rate * weekends
  let total_earnings := total_weekday_earnings + total_weekend_earnings
  let spent_makeup := 3 / 8 * total_earnings
  let remaining_after_makeup := total_earnings - spent_makeup
  let spent_skincare := 2 / 5 * remaining_after_makeup
  let remaining_after_skincare := remaining_after_makeup - spent_skincare
  let spent_cellphone := 1 / 6 * remaining_after_skincare
  let final_remaining := remaining_after_skincare - spent_cellphone
  final_remaining

theorem susie_remaining_money : calculate_remaining_money = 98.4375 := by
  sorry

end susie_remaining_money_l33_33984


namespace functional_equation_solution_l33_33382

theorem functional_equation_solution (f : ℤ → ℤ) :
  (∀ m n : ℤ, f (f (m + n)) = f m + f n) ↔
  (∃ a : ℤ, ∀ n : ℤ, f n = n + a ∨ f n = 0) :=
sorry

end functional_equation_solution_l33_33382


namespace inequality_range_l33_33424

theorem inequality_range (y : ℝ) (b : ℝ) (hb : 0 < b) : (|y-5| + 2 * |y-2| > b) ↔ (b < 3) := 
sorry

end inequality_range_l33_33424


namespace fraction_red_marbles_after_doubling_l33_33435

theorem fraction_red_marbles_after_doubling (x : ℕ) (h : x > 0) :
  let blue_fraction : ℚ := 3 / 5
  let red_fraction := 1 - blue_fraction
  let initial_blue_marbles := blue_fraction * x
  let initial_red_marbles := red_fraction * x
  let new_red_marbles := 2 * initial_red_marbles
  let new_total_marbles := initial_blue_marbles + new_red_marbles
  let new_red_fraction := new_red_marbles / new_total_marbles
  new_red_fraction = 4 / 7 :=
sorry

end fraction_red_marbles_after_doubling_l33_33435


namespace solve_problem_for_m_n_l33_33069

theorem solve_problem_for_m_n (m n : ℕ) (h₀ : m > 0) (h₁ : n > 0) (h₂ : m * (n + m) = n * (n - m)) :
  ((∃ h : ℕ, m = (2 * h + 1) * h ∧ n = (2 * h + 1) * (h + 1)) ∨ 
   (∃ h : ℕ, h > 0 ∧ m = 2 * h * (4 * h^2 - 1) ∧ n = 2 * h * (4 * h^2 + 1))) := 
sorry

end solve_problem_for_m_n_l33_33069


namespace smallest_part_of_division_l33_33261

theorem smallest_part_of_division (x : ℝ) (h : 2 * x + (1/2) * x + (1/4) * x = 105) : 
  (1/4) * x = 10.5 :=
sorry

end smallest_part_of_division_l33_33261


namespace log_product_eq_two_l33_33376

open Real

theorem log_product_eq_two
  : log 5 / log 3 * log 6 / log 5 * log 9 / log 6 = 2 := by
  sorry

end log_product_eq_two_l33_33376


namespace central_angle_regular_hexagon_l33_33332

theorem central_angle_regular_hexagon :
  ∀ (circle_degrees hexagon_sides : ℕ), circle_degrees = 360 ∧ hexagon_sides = 6 → circle_degrees / hexagon_sides = 60 :=
by {
  intros circle_degrees hexagon_sides h,
  cases h with h_circle h_hexagon,
  rw [h_circle, h_hexagon],
  norm_num,
  done
}

end central_angle_regular_hexagon_l33_33332


namespace tax_free_amount_is_600_l33_33903

variable (X : ℝ) -- X is the tax-free amount

-- Given conditions
variable (total_value : ℝ := 1720)
variable (tax_paid : ℝ := 89.6)
variable (tax_rate : ℝ := 0.08)

-- Proof problem
theorem tax_free_amount_is_600
  (h1 : 0.08 * (total_value - X) = tax_paid) :
  X = 600 :=
by
  sorry

end tax_free_amount_is_600_l33_33903


namespace value_of_b_l33_33347

-- Define the variables and conditions
variables (a b c : ℚ)
axiom h1 : a + b + c = 150
axiom h2 : a + 10 = b - 3
axiom h3 : b - 3 = 4 * c 

-- The statement we want to prove
theorem value_of_b : b = 655 / 9 := 
by 
  -- We start with assumptions h1, h2, and h3
  sorry

end value_of_b_l33_33347


namespace sum_of_ages_of_sarahs_friends_l33_33842

noncomputable def sum_of_ages (a b c : ℕ) : ℕ := a + b + c

theorem sum_of_ages_of_sarahs_friends (a b c : ℕ) (h_distinct : ∀ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_single_digits : ∀ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10)
  (h_product_36 : ∃ (x y : ℕ), x * y = 36 ∧ x ≠ y)
  (h_factor_36 : ∀ (x y z : ℕ), x ∣ 36 ∧ y ∣ 36 ∧ z ∣ 36) :
  ∃ (a b c : ℕ), sum_of_ages a b c = 16 := 
sorry

end sum_of_ages_of_sarahs_friends_l33_33842


namespace sandwich_price_l33_33530

-- Definitions based on conditions
def price_of_soda : ℝ := 0.87
def total_cost : ℝ := 6.46
def num_soda : ℝ := 4
def num_sandwich : ℝ := 2

-- The key equation based on conditions
def total_cost_equation (S : ℝ) : Prop := 
  num_sandwich * S + num_soda * price_of_soda = total_cost

theorem sandwich_price :
  ∃ S : ℝ, total_cost_equation S ∧ S = 1.49 :=
by
  sorry

end sandwich_price_l33_33530


namespace solve_inequality_l33_33856

theorem solve_inequality (x : ℝ) :
  (x - 2) / (x + 5) ≤ 1 / 2 ↔ x ∈ Set.Ioc (-5 : ℝ) 9 :=
by
  sorry

end solve_inequality_l33_33856


namespace number_of_ways_to_buy_three_items_l33_33134

def headphones : ℕ := 9
def mice : ℕ := 13
def keyboards : ℕ := 5
def keyboard_mouse_sets : ℕ := 4
def headphone_mouse_sets : ℕ := 5

theorem number_of_ways_to_buy_three_items : 
  (keyboard_mouse_sets * headphones) + (headphone_mouse_sets * keyboards) + (headphones * mice * keyboards) = 646 := 
by 
  sorry

end number_of_ways_to_buy_three_items_l33_33134


namespace product_of_two_numbers_l33_33726

theorem product_of_two_numbers (x y : ℝ) (h1 : x^2 + y^2 = 289) (h2 : x + y = 23) : x * y = 120 :=
sorry

end product_of_two_numbers_l33_33726


namespace average_primes_30_50_l33_33033

/-- The theorem statement for proving the average of all prime numbers between 30 and 50 is 39.8 -/
theorem average_primes_30_50 : (31 + 37 + 41 + 43 + 47) / 5 = 39.8 :=
  by
  sorry

end average_primes_30_50_l33_33033


namespace coin_flip_difference_l33_33194

/-- The positive difference between the probability of a fair coin landing heads up
exactly 4 times out of 5 flips and the probability of a fair coin landing heads up
5 times out of 5 flips is 1/8. -/
theorem coin_flip_difference :
  (5 * (1 / 2) ^ 5) - ((1 / 2) ^ 5) = (1 / 8) :=
by
  sorry

end coin_flip_difference_l33_33194


namespace composite_sum_of_powers_l33_33657

theorem composite_sum_of_powers (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h : a * b = c * d) : 
  ∃ x y : ℕ, 1 < x ∧ 1 < y ∧ a^2016 + b^2016 + c^2016 + d^2016 = x * y :=
by sorry

end composite_sum_of_powers_l33_33657


namespace finite_solutions_to_equation_l33_33318

theorem finite_solutions_to_equation :
  ∃ (n : ℕ), ∀ (a b c : ℕ), (a > 0 ∧ b > 0 ∧ c > 0) ∧ (1 / (a:ℚ) + 1 / (b:ℚ) + 1 / (c:ℚ) = 1 / 1983) → 
  (a ≤ n ∧ b ≤ n ∧ c ≤ n) :=
sorry

end finite_solutions_to_equation_l33_33318


namespace volume_of_box_l33_33359

theorem volume_of_box (l w h : ℝ) (h1 : l * w = 24) (h2 : w * h = 16) (h3 : l * h = 6) :
  l * w * h = 48 :=
by
  sorry

end volume_of_box_l33_33359


namespace value_of_a_l33_33729

theorem value_of_a
  (h : { x : ℝ | -3 < x ∧ x < -1 ∨ 2 < x } = { x : ℝ | (x+a)/(x^2 + 4*x + 3) > 0}) : a = -2 :=
by sorry

end value_of_a_l33_33729


namespace probability_at_most_3_heads_10_coins_l33_33605

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l33_33605


namespace solve_for_x_l33_33420

theorem solve_for_x (x y : ℚ) (h1 : 3 * x - y = 7) (h2 : x + 3 * y = 2) : x = 23 / 10 :=
by
  -- Proof is omitted
  sorry

end solve_for_x_l33_33420


namespace mass_percentage_of_Cl_in_compound_l33_33929

theorem mass_percentage_of_Cl_in_compound (mass_percentage_Cl : ℝ) (h : mass_percentage_Cl = 92.11) : mass_percentage_Cl = 92.11 :=
sorry

end mass_percentage_of_Cl_in_compound_l33_33929


namespace distinct_three_digit_numbers_l33_33256

theorem distinct_three_digit_numbers : 
  let digits := {1, 2, 3, 4, 5} in
  ∃ (count : ℕ), count = 60 := by
  sorry

end distinct_three_digit_numbers_l33_33256


namespace sin_double_angle_neg_one_l33_33808

theorem sin_double_angle_neg_one (α : ℝ) (a b : ℝ × ℝ) (h₁ : a = (1, Real.cos α)) (h₂ : b = (Real.sin α, 1)) (h₃ : a.1 * b.1 + a.2 * b.2 = 0) :
  Real.sin (2 * α) = -1 :=
sorry

end sin_double_angle_neg_one_l33_33808


namespace average_speed_l33_33988

def initial_odometer_reading : ℕ := 20
def final_odometer_reading : ℕ := 200
def travel_duration : ℕ := 6

theorem average_speed :
  (final_odometer_reading - initial_odometer_reading) / travel_duration = 30 := by
  sorry

end average_speed_l33_33988


namespace greendale_points_l33_33312

theorem greendale_points : 
  let roosevelt_game1 := 30 
  let roosevelt_game2 := roosevelt_game1 / 2
  let roosevelt_game3 := roosevelt_game2 * 3
  let roosevelt_bonus := 50
  let greendale_diff := 10
  let roosevelt_total := roosevelt_game1 + roosevelt_game2 + roosevelt_game3 + roosevelt_bonus
  let greendale_total := roosevelt_total - greendale_diff
  greendale_total = 130 :=
by
  sorry

end greendale_points_l33_33312


namespace find_m_n_calculate_expression_l33_33935

-- Define the polynomials A and B
def A (m x : ℝ) := 5 * x^2 - m * x + 1
def B (n x : ℝ) := 2 * x^2 - 2 * x - n

-- The conditions
variable (x : ℝ) (m n : ℝ)
def no_linear_or_constant_terms (m : ℝ) (n : ℝ) : Prop :=
  ∀ x : ℝ, 3 * x^2 + (2 - m) * x + (1 + n) = 3 * x^2

-- The target theorem
theorem find_m_n 
  (h : no_linear_or_constant_terms m n) : 
  m = 2 ∧ n = -1 := sorry

-- Calculate the expression when m = 2 and n = -1
theorem calculate_expression
  (hm : m = 2)
  (hn : n = -1) : 
  m^2 + n^2 - 2 * m * n = 9 := sorry

end find_m_n_calculate_expression_l33_33935


namespace probability_four_vertices_same_plane_proof_l33_33249

noncomputable def probability_four_vertices_same_plane : ℚ := 
  let total_ways := Nat.choose 8 4
  let favorable_ways := 12
  favorable_ways / total_ways

theorem probability_four_vertices_same_plane_proof : 
  probability_four_vertices_same_plane = 6 / 35 :=
by
  -- include necessary definitions and calculations for the actual proof
  sorry

end probability_four_vertices_same_plane_proof_l33_33249


namespace expression_result_l33_33004

-- We denote k as a natural number representing the number of digits in A, B, C, and D.
variable (k : ℕ)

-- Definitions of the numbers A, B, C, D, and E based on the problem statement.
def A : ℕ := 3 * ((10 ^ (k - 1) - 1) / 9)
def B : ℕ := 4 * ((10 ^ (k - 1) - 1) / 9)
def C : ℕ := 6 * ((10 ^ (k - 1) - 1) / 9)
def D : ℕ := 7 * ((10 ^ (k - 1) - 1) / 9)
def E : ℕ := 5 * ((10 ^ (2 * k) - 1) / 9)

-- The statement we want to prove.
theorem expression_result :
  E - A * D - B * C + 1 = (10 ^ (k + 1) - 1) / 9 :=
by
  sorry

end expression_result_l33_33004


namespace alice_has_ball_after_three_turns_l33_33905

def probability_Alice_has_ball (turns: ℕ) : ℚ :=
  match turns with
  | 0 => 1 -- Alice starts with the ball
  | _ => sorry -- We would typically calculate this by recursion or another approach.

theorem alice_has_ball_after_three_turns :
  probability_Alice_has_ball 3 = 11 / 27 :=
by
  sorry

end alice_has_ball_after_three_turns_l33_33905


namespace odd_and_even_inter_empty_l33_33540

-- Define the set of odd numbers
def odd_numbers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}

-- Define the set of even numbers
def even_numbers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

-- The theorem stating that the intersection of odd numbers and even numbers is empty
theorem odd_and_even_inter_empty : odd_numbers ∩ even_numbers = ∅ :=
by
  -- placeholder for the proof
  sorry

end odd_and_even_inter_empty_l33_33540


namespace total_artworks_created_l33_33834

theorem total_artworks_created
  (students_group1 : ℕ := 24) (students_group2 : ℕ := 12)
  (kits_total : ℕ := 48)
  (kits_per_3_students : ℕ := 3) (kits_per_2_students : ℕ := 2)
  (artwork_types : ℕ := 3)
  (paintings_group1_1 : ℕ := 12 * 2) (drawings_group1_1 : ℕ := 12 * 4) (sculptures_group1_1 : ℕ := 12 * 1)
  (paintings_group1_2 : ℕ := 12 * 1) (drawings_group1_2 : ℕ := 12 * 5) (sculptures_group1_2 : ℕ := 12 * 3)
  (paintings_group2_1 : ℕ := 4 * 3) (drawings_group2_1 : ℕ := 4 * 6) (sculptures_group2_1 : ℕ := 4 * 3)
  (paintings_group2_2 : ℕ := 8 * 4) (drawings_group2_2 : ℕ := 8 * 7) (sculptures_group2_2 : ℕ := 8 * 1)
  : (paintings_group1_1 + paintings_group1_2 + paintings_group2_1 + paintings_group2_2) +
    (drawings_group1_1 + drawings_group1_2 + drawings_group2_1 + drawings_group2_2) +
    (sculptures_group1_1 + sculptures_group1_2 + sculptures_group2_1 + sculptures_group2_2) = 336 :=
by sorry

end total_artworks_created_l33_33834


namespace right_triangle_area_l33_33515

theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 240 :=
by
  sorry

end right_triangle_area_l33_33515


namespace fair_coin_flip_probability_difference_l33_33200

noncomputable def choose : ℕ → ℕ → ℕ
| _, 0 => 1
| 0, _ => 0
| n + 1, r + 1 => choose n r + choose n (r + 1)

theorem fair_coin_flip_probability_difference :
  let p := 1 / 2
  let p_heads_4_out_of_5 := choose 5 4 * p^4 * p
  let p_heads_5_out_of_5 := p^5
  p_heads_4_out_of_5 - p_heads_5_out_of_5 = 1 / 8 :=
by
  sorry

end fair_coin_flip_probability_difference_l33_33200


namespace mean_of_all_students_l33_33485

variable (M A m a : ℕ)
variable (M_val : M = 84)
variable (A_val : A = 70)
variable (ratio : m = 3 * a / 4)

theorem mean_of_all_students (M A m a : ℕ) (M_val : M = 84) (A_val : A = 70) (ratio : m = 3 * a / 4) :
    (63 * a + 70 * a) / (7 * a / 4) = 76 := by
  sorry

end mean_of_all_students_l33_33485


namespace probability_of_at_most_3_heads_out_of_10_l33_33585
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l33_33585


namespace table_tennis_championship_max_winners_l33_33607

theorem table_tennis_championship_max_winners 
  (n : ℕ) 
  (knockout_tournament : ∀ (players : ℕ), players = 200) 
  (eliminations_per_match : ℕ := 1) 
  (matches_needed_to_win_3_times : ℕ := 3) : 
  ∃ k : ℕ, k = 66 ∧ k * matches_needed_to_win_3_times ≤ (n - 1) :=
begin
  sorry
end

end table_tennis_championship_max_winners_l33_33607


namespace find_center_of_circle_l33_33749

-- Condition 1: The circle is tangent to the lines 3x - 4y = 12 and 3x - 4y = -48
def tangent_line1 (x y : ℝ) : Prop := 3 * x - 4 * y = 12
def tangent_line2 (x y : ℝ) : Prop := 3 * x - 4 * y = -48

-- Condition 2: The center of the circle lies on the line x - 2y = 0
def center_line (x y : ℝ) : Prop := x - 2 * y = 0

-- The center of the circle
def circle_center (x y : ℝ) : Prop := 
  tangent_line1 x y ∧ tangent_line2 x y ∧ center_line x y

-- Statement to prove
theorem find_center_of_circle : 
  circle_center (-18) (-9) := 
sorry

end find_center_of_circle_l33_33749


namespace school_fitness_event_participants_l33_33139

theorem school_fitness_event_participants :
  let p0 := 500 -- initial number of participants in 2000
  let r1 := 0.3 -- increase rate in 2001
  let r2 := 0.4 -- increase rate in 2002
  let r3 := 0.5 -- increase rate in 2003
  let p1 := p0 * (1 + r1) -- participants in 2001
  let p2 := p1 * (1 + r2) -- participants in 2002
  let p3 := p2 * (1 + r3) -- participants in 2003
  p3 = 1365 -- prove that number of participants in 2003 is 1365
:= sorry

end school_fitness_event_participants_l33_33139


namespace Dan_gave_Sara_limes_l33_33921

theorem Dan_gave_Sara_limes : 
  ∀ (original_limes now_limes given_limes : ℕ),
  original_limes = 9 →
  now_limes = 5 →
  given_limes = original_limes - now_limes →
  given_limes = 4 :=
by
  intros original_limes now_limes given_limes h1 h2 h3
  sorry

end Dan_gave_Sara_limes_l33_33921


namespace solution_set_inequality_l33_33971

def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2

theorem solution_set_inequality (a x : ℝ) (h : Set.Ioo (-1 : ℝ) (2 : ℝ) = {x | |f a x| < 6}) : 
    {x | f a x ≤ 1} = {x | x ≥ 1 / 4} :=
sorry

end solution_set_inequality_l33_33971


namespace exist_two_pies_differing_in_both_l33_33270

-- Define enumeration types for fillings and preparation methods
inductive Filling
| apple
| cherry

inductive Preparation
| fried
| baked

-- Define pie type with filling and preparation
structure Pie where
  filling : Filling
  preparation : Preparation

-- Define the types of pies available
def pie1 : Pie := { filling := Filling.apple, preparation := Preparation.fried }
def pie2 : Pie := { filling := Filling.cherry, preparation := Preparation.fried }
def pie3 : Pie := { filling := Filling.apple, preparation := Preparation.baked }
def pie4 : Pie := { filling := Filling.cherry, preparation := Preparation.baked }

-- Define the list of available pies
def availablePies : List Pie := [pie1, pie2, pie3, pie4]

theorem exist_two_pies_differing_in_both (pies : List Pie) (h : pies.length ≥ 3) :
  ∃ p1 p2 : Pie, p1 ≠ p2 ∧ p1.filling ≠ p2.filling ∧ p1.preparation ≠ p2.preparation :=
by
  -- Proof content to be filled in
  sorry

end exist_two_pies_differing_in_both_l33_33270


namespace b_squared_gt_4ac_l33_33469

theorem b_squared_gt_4ac (a b c : ℝ) (h : (a + b + c) * c < 0) : b^2 > 4 * a * c :=
by
  sorry

end b_squared_gt_4ac_l33_33469


namespace total_rainfall_November_l33_33276

def rain_first_15_days : ℕ := 4

def days_first_15 : ℕ := 15

def rain_last_15_days : ℕ := 2 * rain_first_15_days

def days_last_15 : ℕ := 15

def total_rainfall : ℕ := 
  (rain_first_15_days * days_first_15) + (rain_last_15_days * days_last_15)

theorem total_rainfall_November : total_rainfall = 180 := by
  sorry

end total_rainfall_November_l33_33276


namespace decreasing_y_as_x_increases_l33_33760

theorem decreasing_y_as_x_increases :
  (∀ x1 x2, x1 < x2 → (-2 * x1 + 1) > (-2 * x2 + 1)) ∧
  ¬ (∀ x1 x2, x1 < x2 → (x1^2 + 1) > (x2^2 + 1)) ∧
  ¬ (∀ x1 x2, x1 < x2 → (-x1^2 + 1) > (-x2^2 + 1)) ∧
  ¬ (∀ x1 x2, x1 < x2 → (2 * x1 + 1) > (2 * x2 + 1)) :=
by
  sorry

end decreasing_y_as_x_increases_l33_33760


namespace gas_volume_ranking_l33_33538

theorem gas_volume_ranking (Russia_V: ℝ) (Non_West_V: ℝ) (West_V: ℝ)
  (h_russia: Russia_V = 302790.13)
  (h_non_west: Non_West_V = 26848.55)
  (h_west: West_V = 21428): Russia_V > Non_West_V ∧ Non_West_V > West_V :=
by
  have h1: Russia_V = 302790.13 := h_russia
  have h2: Non_West_V = 26848.55 := h_non_west
  have h3: West_V = 21428 := h_west
  sorry


end gas_volume_ranking_l33_33538


namespace solution_set_f_div_x_lt_zero_l33_33366

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_div_x_lt_zero :
  (∀ x, f (2 + (2 - x)) = f x) ∧
  (∀ x1 x2 : ℝ, x1 < 2 ∧ x2 < 2 ∧ x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) < 0) ∧
  f 4 = 0 →
  { x : ℝ | f x / x < 0 } = { x | x < 0 } ∪ { x | 0 < x ∧ x < 4 } :=
sorry

end solution_set_f_div_x_lt_zero_l33_33366


namespace ferns_have_1260_leaves_l33_33287

def num_ferns : ℕ := 6
def fronds_per_fern : ℕ := 7
def leaves_per_frond : ℕ := 30
def total_leaves : ℕ := num_ferns * fronds_per_fern * leaves_per_frond

theorem ferns_have_1260_leaves : total_leaves = 1260 :=
by 
  -- proof goes here
  sorry

end ferns_have_1260_leaves_l33_33287


namespace min_value_y_l33_33936

theorem min_value_y (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / a + 4 / b = 2) : 4 * a + b ≥ 8 :=
sorry

end min_value_y_l33_33936


namespace terminating_decimals_count_l33_33786

theorem terminating_decimals_count :
  ∃ (count : ℕ), count = 60 ∧
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 180 →
  (∃ m : ℕ, m * 180 = n * (2^2 * 5)) → 60 = count := sorry

end terminating_decimals_count_l33_33786


namespace g_800_eq_768_l33_33987

noncomputable def g : ℕ → ℕ := sorry

axiom g_condition1 (n : ℕ) : g (g n) = 2 * n
axiom g_condition2 (n : ℕ) : g (4 * n + 3) = 4 * n + 1

theorem g_800_eq_768 : g 800 = 768 := by
  sorry

end g_800_eq_768_l33_33987


namespace solve_system_of_equations_l33_33213

-- Given conditions
variables {a b c k x y z : ℝ}
variables (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
variables (eq1 : a * x + b * y + c * z = k)
variables (eq2 : a^2 * x + b^2 * y + c^2 * z = k^2)
variables (eq3 : a^3 * x + b^3 * y + c^3 * z = k^3)

-- Statement to be proved
theorem solve_system_of_equations :
  x = k * (k - c) * (k - b) / (a * (a - c) * (a - b)) ∧
  y = k * (k - c) * (k - a) / (b * (b - c) * (b - a)) ∧
  z = k * (k - a) * (k - b) / (c * (c - a) * (c - b)) :=
sorry

end solve_system_of_equations_l33_33213


namespace inequality_solution_set_min_value_of_x_plus_y_l33_33253

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (2 * a + 1) * x + 2

theorem inequality_solution_set (a : ℝ) :
  (if a < 0 then (∀ x : ℝ, f a x > 0 ↔ (1/a < x ∧ x < 2))
   else if a = 0 then (∀ x : ℝ, f a x > 0 ↔ x < 2)
   else if 0 < a ∧ a < 1/2 then (∀ x : ℝ, f a x > 0 ↔ (x < 2 ∨ 1/a < x))
   else if a = 1/2 then (∀ x : ℝ, f a x > 0 ↔ (x ≠ 2))
   else if a > 1/2 then (∀ x : ℝ, f a x > 0 ↔ (x < 1/a ∨ x > 2))
   else false) := 
sorry

theorem min_value_of_x_plus_y (a : ℝ) (h : 0 < a) (x y : ℝ) (hx : y ≥ f a (|x|)) :
  x + y ≥ -a - (1/a) := 
sorry

end inequality_solution_set_min_value_of_x_plus_y_l33_33253


namespace prob_4_vertices_same_plane_of_cube_l33_33250

/-- A cube contains 8 vertices. We need to calculate the probability that 4 chosen vertices lie on the same plane. -/
theorem prob_4_vertices_same_plane_of_cube : 
  let total_ways := (Nat.choose 8 4)
  let favorable_ways := 12
  in (favorable_ways / total_ways : ℚ) = 6 / 35 :=
by
  sorry

end prob_4_vertices_same_plane_of_cube_l33_33250


namespace fruits_given_away_l33_33824

-- Definitions based on the conditions
def initial_pears := 10
def initial_oranges := 20
def initial_apples := 2 * initial_pears
def initial_fruits := initial_pears + initial_oranges + initial_apples
def fruits_left := 44

-- Theorem to prove the total number of fruits given to her sister
theorem fruits_given_away : initial_fruits - fruits_left = 6 := by
  sorry

end fruits_given_away_l33_33824


namespace compute_expression_l33_33919

theorem compute_expression : 10 * (3 / 27) * 36 = 40 := 
by 
  sorry

end compute_expression_l33_33919


namespace complement_of_A_in_U_l33_33097

-- Define the universal set U and set A
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {3, 4, 5}

-- Define the complement of A in U
theorem complement_of_A_in_U : (U \ A) = {1, 2} :=
by
  sorry

end complement_of_A_in_U_l33_33097


namespace selling_prices_maximize_profit_l33_33221

-- Definitions for the conditions
def total_items : ℕ := 200
def budget : ℤ := 5000
def cost_basketball : ℤ := 30
def cost_volleyball : ℤ := 24
def selling_price_ratio : ℚ := 3 / 2
def school_purchase_basketballs_value : ℤ := 1800
def school_purchase_volleyballs_value : ℤ := 1500
def basketballs_fewer_than_volleyballs : ℤ := 10

-- Part 1: Proof of selling prices
theorem selling_prices (x : ℚ) :
  (school_purchase_volleyballs_value / x - school_purchase_basketballs_value / (x * selling_price_ratio) = basketballs_fewer_than_volleyballs)
  → ∃ (basketball_price volleyball_price : ℚ), basketball_price = 45 ∧ volleyball_price = 30 :=
by
  sorry

-- Part 2: Proof of maximizing profit
theorem maximize_profit (a : ℕ) :
  (cost_basketball * a + cost_volleyball * (total_items - a) ≤ budget)
  → ∃ optimal_a : ℕ, (optimal_a = 33 ∧ total_items - optimal_a = 167) :=
by
  sorry

end selling_prices_maximize_profit_l33_33221


namespace population_increase_20th_century_l33_33361

theorem population_increase_20th_century (P : ℕ) :
  let population_mid_century := 3 * P
  let population_end_century := 12 * P
  (population_end_century - P) / P * 100 = 1100 :=
by
  sorry

end population_increase_20th_century_l33_33361


namespace tan_shift_symmetric_l33_33863

theorem tan_shift_symmetric :
  let f (x : ℝ) := Real.tan (2 * x + Real.pi / 6)
  let g (x : ℝ) := f (x + Real.pi / 6)
  g (Real.pi / 4) = 0 ∧ ∀ x, g (Real.pi / 2 - x) = -g (Real.pi / 2 + x) :=
by
  sorry

end tan_shift_symmetric_l33_33863


namespace quadratic_intersects_x_axis_only_once_l33_33094

theorem quadratic_intersects_x_axis_only_once (a : ℝ) :
  (∀ x : ℝ, (a * x^2 - a * x + 3 * x + 1 = 0) → a = 1 ∨ a = 9) :=
sorry

end quadratic_intersects_x_axis_only_once_l33_33094


namespace abcd_zero_l33_33702

theorem abcd_zero (a b c d : ℝ) (h1 : a + b + c + d = 0) (h2 : ab + ac + bc + bd + ad + cd = 0) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
sorry

end abcd_zero_l33_33702


namespace no_real_roots_of_x_squared_plus_5_l33_33235

theorem no_real_roots_of_x_squared_plus_5 : ¬ ∃ (x : ℝ), x^2 + 5 = 0 :=
by
  sorry

end no_real_roots_of_x_squared_plus_5_l33_33235


namespace positive_difference_between_probabilities_is_one_eighth_l33_33198

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem positive_difference_between_probabilities_is_one_eighth :
  let p : ℚ := 1 / 2
    n : ℕ := 5
    prob_4_heads := binomial_probability n 4 p
    prob_5_heads := binomial_probability n 5 p
  in |prob_4_heads - prob_5_heads| = 1 / 8 :=
by
  let p : ℚ := 1 / 2
  let n : ℕ := 5
  let prob_4_heads := binomial_probability n 4 p
  let prob_5_heads := binomial_probability n 5 p
  sorry

end positive_difference_between_probabilities_is_one_eighth_l33_33198


namespace alex_wins_if_picks_two_l33_33619

theorem alex_wins_if_picks_two (matches_left : ℕ) (alex_picks bob_picks : ℕ) :
  matches_left = 30 →
  1 ≤ alex_picks ∧ alex_picks ≤ 6 →
  1 ≤ bob_picks ∧ bob_picks ≤ 6 →
  alex_picks = 2 →
  (∀ n, (n % 7 ≠ 0) → ¬ (∃ k, matches_left - k ≤ 0 ∧ (matches_left - k) % 7 = 0)) :=
by sorry

end alex_wins_if_picks_two_l33_33619


namespace blackboard_problem_l33_33295

theorem blackboard_problem (n : ℕ) (h_pos : 0 < n) :
  ∃ x, (∀ (t : ℕ), t < n - 1 → ∃ a b : ℕ, a + b + 2 * (t + 1) = n + 1 ∧ a > 0 ∧ b > 0) → 
  x ≥ 2 ^ ((4 * n ^ 2 - 4) / 3) :=
by
  sorry

end blackboard_problem_l33_33295


namespace sum_of_squares_of_consecutive_integers_is_perfect_square_l33_33492

theorem sum_of_squares_of_consecutive_integers_is_perfect_square (x : ℤ) :
  ∃ k : ℤ, k ^ 2 = x ^ 2 + (x + 1) ^ 2 + (x ^ 2 * (x + 1) ^ 2) :=
by
  use (x^2 + x + 1)
  sorry

end sum_of_squares_of_consecutive_integers_is_perfect_square_l33_33492


namespace total_rainfall_November_l33_33275

def rain_first_15_days : ℕ := 4

def days_first_15 : ℕ := 15

def rain_last_15_days : ℕ := 2 * rain_first_15_days

def days_last_15 : ℕ := 15

def total_rainfall : ℕ := 
  (rain_first_15_days * days_first_15) + (rain_last_15_days * days_last_15)

theorem total_rainfall_November : total_rainfall = 180 := by
  sorry

end total_rainfall_November_l33_33275


namespace rent_for_each_room_l33_33618

theorem rent_for_each_room (x : ℝ) (ha : 4800 / x = 4200 / (x - 30)) (hx : x = 240) :
  x = 240 ∧ (x - 30) = 210 :=
by
  sorry

end rent_for_each_room_l33_33618


namespace fair_coin_difference_l33_33189

noncomputable def binom_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem fair_coin_difference :
  let p := (1 : ℚ) / 2 in
  let p_1 := binom_coeff 5 4 * p^4 * (1 - p) in
  let p_2 := p^5 in
  p_1 - p_2 = 1 / 8 :=
by
  sorry

end fair_coin_difference_l33_33189


namespace probability_at_most_3_heads_l33_33559

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l33_33559


namespace volume_of_box_l33_33358

theorem volume_of_box (l w h : ℝ) (h1 : l * w = 24) (h2 : w * h = 16) (h3 : l * h = 6) :
  l * w * h = 48 :=
by
  sorry

end volume_of_box_l33_33358


namespace probability_heads_at_most_3_of_10_coins_flipped_l33_33550

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l33_33550


namespace probability_at_most_3_heads_10_flips_l33_33596

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l33_33596


namespace find_complex_z_l33_33951

theorem find_complex_z (z : ℂ) (h : conj z * (1 + I) = 1 - I) : z = I :=
sorry

end find_complex_z_l33_33951


namespace cube_inequality_of_greater_l33_33082

variable (a b : ℝ)

theorem cube_inequality_of_greater (h : a > b) : a^3 > b^3 :=
sorry

end cube_inequality_of_greater_l33_33082


namespace relationship_between_p_and_q_l33_33150

variables {x y : ℝ}

def p (x y : ℝ) := (x^2 + y^2) * (x - y)
def q (x y : ℝ) := (x^2 - y^2) * (x + y)

theorem relationship_between_p_and_q (h1 : x < y) (h2 : y < 0) : p x y > q x y := 
  by sorry

end relationship_between_p_and_q_l33_33150


namespace area_perimeter_quadratic_l33_33338

theorem area_perimeter_quadratic (a x y : ℝ) (h1 : x = 4 * a) (h2 : y = a^2) : y = (x / 4)^2 :=
by sorry

end area_perimeter_quadratic_l33_33338


namespace no_triangle_sum_of_any_two_angles_lt_120_no_triangle_sum_of_any_two_angles_gt_120_l33_33959

theorem no_triangle_sum_of_any_two_angles_lt_120 (α β γ : ℝ) (h : α + β + γ = 180) :
  ¬ (α + β < 120 ∧ β + γ < 120 ∧ γ + α < 120) :=
by
  sorry

theorem no_triangle_sum_of_any_two_angles_gt_120 (α β γ : ℝ) (h : α + β + γ = 180) :
  ¬ (α + β > 120 ∧ β + γ > 120 ∧ γ + α > 120) :=
by
  sorry

end no_triangle_sum_of_any_two_angles_lt_120_no_triangle_sum_of_any_two_angles_gt_120_l33_33959


namespace sum_of_remainders_l33_33027

theorem sum_of_remainders (n : ℤ) (h : n % 15 = 7) : 
  (n % 3) + (n % 5) = 3 := 
by
  -- the proof will go here
  sorry

end sum_of_remainders_l33_33027


namespace john_hourly_wage_with_bonus_l33_33827

structure JohnJob where
  daily_wage : ℕ
  work_hours : ℕ
  bonus_amount : ℕ
  extra_hours : ℕ

def total_daily_wage (job : JohnJob) : ℕ :=
  job.daily_wage + job.bonus_amount

def total_work_hours (job : JohnJob) : ℕ :=
  job.work_hours + job.extra_hours

def hourly_wage (job : JohnJob) : ℕ :=
  total_daily_wage job / total_work_hours job

noncomputable def johns_job : JohnJob :=
  { daily_wage := 80, work_hours := 8, bonus_amount := 20, extra_hours := 2 }

theorem john_hourly_wage_with_bonus :
  hourly_wage johns_job = 10 :=
by
  sorry

end john_hourly_wage_with_bonus_l33_33827


namespace terminating_decimal_count_number_of_terminating_decimals_l33_33784

theorem terminating_decimal_count {n : ℕ} (h1 : 1 ≤ n) (h2 : n ≤ 180) :
  (∃ k, n = 9 * k ∧ 1 ≤ 9 * k ∧ 9 * k ≤ 180) ↔ n ∈ (finset.range 21).image (λ k, 9 * k) :=
by sorry

theorem number_of_terminating_decimals :
  (finset.range 21).card = 20 :=
by sorry

end terminating_decimal_count_number_of_terminating_decimals_l33_33784


namespace line_eq_circle_eq_l33_33081

section
  variable (A B : ℝ × ℝ)
  variable (A_eq : A = (4, 6))
  variable (B_eq : B = (-2, 4))

  theorem line_eq : ∃ (a b c : ℝ), (a, b, c) = (1, -3, 14) ∧ ∀ x y, (y - 6) = ((4 - 6) / (-2 - 4)) * (x - 4) → a * x + b * y + c = 0 :=
  sorry

  theorem circle_eq : ∃ (h k r : ℝ), (h, k, r) = (1, 5, 10) ∧ ∀ x y, (x - 1)^2 + (y - 5)^2 = 10 :=
  sorry
end

end line_eq_circle_eq_l33_33081


namespace bead_problem_l33_33821

theorem bead_problem 
  (x y : ℕ) 
  (hx : 19 * x + 17 * y = 2017): 
  (x + y = 107) ∨ (x + y = 109) ∨ (x + y = 111) ∨ (x + y = 113) ∨ (x + y = 115) ∨ (x + y = 117) := 
sorry

end bead_problem_l33_33821


namespace silver_coins_change_l33_33444

-- Define the conditions
def condition1 : ℕ × ℕ := (20, 4) -- (20 silver coins, 4 gold coins change)
def condition2 : ℕ × ℕ := (15, 1) -- (15 silver coins, 1 gold coin change)
def cost_of_cloak_in_gold_coins : ℕ := 14

-- Define the theorem to be proven
theorem silver_coins_change (s1 g1 s2 g2 cloak_g : ℕ) (h1 : (s1, g1) = condition1) (h2 : (s2, g2) = condition2) :
  ∃ silver : ℕ, (silver = 10) :=
by {
  sorry
}

end silver_coins_change_l33_33444


namespace appropriate_sampling_method_l33_33734

-- Definitions and conditions
def total_products : ℕ := 40
def first_class_products : ℕ := 10
def second_class_products : ℕ := 25
def defective_products : ℕ := 5
def samples_needed : ℕ := 8

-- Theorem statement
theorem appropriate_sampling_method : 
  (first_class_products + second_class_products + defective_products = total_products) ∧ 
  (2 ≤ first_class_products ∧ 2 ≤ second_class_products ∧ 1 ≤ defective_products) → 
  "Stratified Sampling" = "The appropriate sampling method for quality analysis" :=
  sorry

end appropriate_sampling_method_l33_33734


namespace table_tennis_championship_max_winners_l33_33608

theorem table_tennis_championship_max_winners 
  (n : ℕ) 
  (knockout_tournament : ∀ (players : ℕ), players = 200) 
  (eliminations_per_match : ℕ := 1) 
  (matches_needed_to_win_3_times : ℕ := 3) : 
  ∃ k : ℕ, k = 66 ∧ k * matches_needed_to_win_3_times ≤ (n - 1) :=
begin
  sorry
end

end table_tennis_championship_max_winners_l33_33608


namespace largest_allowed_set_size_correct_l33_33474

noncomputable def largest_allowed_set_size (N : ℕ) : ℕ :=
  N - Nat.floor (N / 4)

def is_allowed (S : Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → (a ∣ b → b ∣ c → False)

theorem largest_allowed_set_size_correct (N : ℕ) (hN : 0 < N) : 
  ∃ S : Finset ℕ, is_allowed S ∧ S.card = largest_allowed_set_size N := sorry

end largest_allowed_set_size_correct_l33_33474


namespace sally_bread_consumption_l33_33708

theorem sally_bread_consumption :
  (2 * 2) + (1 * 2) = 6 :=
by
  sorry

end sally_bread_consumption_l33_33708


namespace distance_between_points_on_parabola_l33_33404

theorem distance_between_points_on_parabola :
  ∀ (x1 x2 y1 y2 : ℝ), 
    (y1^2 = 4 * x1) → (y2^2 = 4 * x2) → (x2 = x1 + 2) → (|y2 - y1| = 4 * Real.sqrt x2 - 4 * Real.sqrt x1) →
    Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 8 :=
by
  intros x1 x2 y1 y2 h1 h2 h3 h4
  sorry

end distance_between_points_on_parabola_l33_33404


namespace num_ways_to_designated_face_l33_33225

-- Define the structure of the dodecahedron
inductive Face
| Top
| Bottom
| TopRing (n : ℕ)   -- n ranges from 1 to 5
| BottomRing (n : ℕ)  -- n ranges from 1 to 5
deriving Repr, DecidableEq

-- Define adjacency relations on Faces (simplified)
def adjacent : Face → Face → Prop
| Face.Top, Face.TopRing n          => true
| Face.TopRing n, Face.TopRing m    => (m = (n % 5) + 1) ∨ (m = ((n + 3) % 5) + 1)
| Face.TopRing n, Face.BottomRing m => true
| Face.BottomRing n, Face.BottomRing m => true
| _, _ => false

-- Predicate for specific face on the bottom ring
def designated_bottom_face (f : Face) : Prop :=
  match f with
  | Face.BottomRing 1 => true
  | _ => false

-- Define the number of ways to move from top to the designated bottom face
noncomputable def num_ways : ℕ :=
  5 + 10

-- Lean statement that represents our equivalent proof problem
theorem num_ways_to_designated_face :
  num_ways = 15 := by
  sorry

end num_ways_to_designated_face_l33_33225


namespace circle_O₁_equation_sum_of_squares_constant_l33_33399

-- Given conditions
def circle_O (x y : ℝ) := x^2 + y^2 = 25
def center_O₁ (m : ℝ) : ℝ × ℝ := (m, 0) 
def intersect_point := (3, 4)
def is_intersection (x y : ℝ) := circle_O x y ∧ (x - intersect_point.1)^2 + (y - intersect_point.2)^2 = 0
def line_passing_P (k : ℝ) (x y : ℝ) := y - intersect_point.2 = k * (x - intersect_point.1)
def point_on_circle (circle : ℝ × ℝ → Prop) (x y : ℝ) := circle (x, y)
def distance_squared (A B : ℝ × ℝ) := (A.1 - B.1)^2 + (A.2 - B.2)^2

-- Problem statements
theorem circle_O₁_equation (k : ℝ) (m : ℝ) (x y : ℝ) (h : k = 1) (h_intersect: is_intersection 3 4)
  (h_BP_distance : distance_squared (3, 4) (x, y) = (7 * Real.sqrt 2)^2) : 
  (x - 14)^2 + y^2 = 137 := sorry

theorem sum_of_squares_constant (k m : ℝ) (h : k ≠ 0) (h_perpendicular : line_passing_P (-1/k) 3 4)
  (A B C D : ℝ × ℝ) (h_AB_distance : distance_squared A B = 4 * m^2 / (1 + k^2)) 
  (h_CD_distance : distance_squared C D = 4 * m^2 * k^2 / (1 + k^2)) : 
  distance_squared A B + distance_squared C D = 4 * m^2 := sorry

end circle_O₁_equation_sum_of_squares_constant_l33_33399


namespace average_incorrect_answers_is_correct_l33_33433

-- Definitions
def total_items : ℕ := 60
def liza_correct_answers : ℕ := (90 * total_items) / 100
def rose_correct_answers : ℕ := liza_correct_answers + 2
def max_correct_answers : ℕ := liza_correct_answers - 5

def liza_incorrect_answers : ℕ := total_items - liza_correct_answers
def rose_incorrect_answers : ℕ := total_items - rose_correct_answers
def max_incorrect_answers : ℕ := total_items - max_correct_answers

def average_incorrect_answers : ℚ :=
  (liza_incorrect_answers + rose_incorrect_answers + max_incorrect_answers) / 3

-- Theorem statement
theorem average_incorrect_answers_is_correct : average_incorrect_answers = 7 := by
  -- Proof goes here
  sorry

end average_incorrect_answers_is_correct_l33_33433


namespace one_room_cheaper_by_l33_33976

-- Define the initial prices of the apartments
variables (a b : ℝ)

-- Define the increase rates and the new prices
def new_price_one_room := 1.21 * a
def new_price_two_room := 1.11 * b
def new_total_price := 1.15 * (a + b)

-- The main theorem encapsulating the problem
theorem one_room_cheaper_by : a + b ≠ 0 → 1.21 * a + 1.11 * b = 1.15 * (a + b) → b / a = 1.5 :=
by
  intro h_non_zero h_prices
  -- we assume the main theorem is true to structure the goal state
  sorry

end one_room_cheaper_by_l33_33976


namespace solve_fractional_equation_l33_33730

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 3) : (2 / (x - 3) = 3 / x) → x = 9 :=
by
  sorry

end solve_fractional_equation_l33_33730


namespace magic_shop_change_l33_33452

theorem magic_shop_change :
  (∀ (cloak : Type), ∃ price_gold price_silver1 change_gold1 price_silver2 change_gold2, 
  price_silver1 = 20 ∧ change_gold1 = 4 ∧ 
  price_silver2 = 15 ∧ change_gold2 = 1 ∧ 
  price_gold = 14 ∧ 
  ∀ change_silver, 
    (20 - 4) * change_silver = 15 - 1 → -- Relation derived from the conditions
    (14 - (15 - 1) * change_silver / (20 - 4)) * change_silver = 10) := 
sorry

end magic_shop_change_l33_33452


namespace unique_function_l33_33243

def functional_equation (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (-f x - f y) = 1 - x - y

theorem unique_function :
  ∀ f : ℤ → ℤ, (functional_equation f) → (∀ x : ℤ, f x = x - 1) :=
by
  intros f h
  sorry

end unique_function_l33_33243


namespace gcd_228_1995_l33_33355

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := 
by
  sorry

end gcd_228_1995_l33_33355


namespace pradeep_marks_l33_33490

-- Conditions as definitions
def passing_percentage : ℝ := 0.35
def max_marks : ℕ := 600
def fail_difference : ℕ := 25

def passing_marks (total_marks : ℕ) (percentage : ℝ) : ℝ :=
  percentage * total_marks

def obtained_marks (passing_marks : ℝ) (difference : ℕ) : ℝ :=
  passing_marks - difference

-- Theorem statement
theorem pradeep_marks : obtained_marks (passing_marks max_marks passing_percentage) fail_difference = 185 := by
  sorry

end pradeep_marks_l33_33490


namespace total_candies_l33_33958

-- Define variables and conditions
variables (x y z : ℕ)
axiom h1 : x = y / 2
axiom h2 : x + z = 24
axiom h3 : y + z = 34

-- The statement to be proved
theorem total_candies : x + y + z = 44 :=
by
  sorry

end total_candies_l33_33958


namespace principal_equivalence_l33_33902

-- Define the conditions
def SI : ℝ := 4020.75
def R : ℝ := 9
def T : ℝ := 5

-- Define the principal calculation
noncomputable def P := SI / (R * T / 100)

-- Prove that the principal P equals 8935
theorem principal_equivalence : P = 8935 := by
  sorry

end principal_equivalence_l33_33902


namespace count_four_digit_numbers_divisible_by_5_and_ending_with_45_l33_33415

def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def ends_with_45 (n : ℕ) : Prop := (n % 100) = 45 
def divisible_by_5 (n : ℕ) : Prop := (n % 5) = 0

theorem count_four_digit_numbers_divisible_by_5_and_ending_with_45 : 
  {n : ℕ | is_four_digit_number n ∧ ends_with_45 n ∧ divisible_by_5 n}.to_finset.card = 90 :=
sorry

end count_four_digit_numbers_divisible_by_5_and_ending_with_45_l33_33415


namespace added_water_correct_l33_33219

theorem added_water_correct (initial_fullness : ℝ) (final_fullness : ℝ) (capacity : ℝ) (initial_amount : ℝ) (final_amount : ℝ) (added_water : ℝ) :
    initial_fullness = 0.30 →
    final_fullness = 3/4 →
    capacity = 60 →
    initial_amount = initial_fullness * capacity →
    final_amount = final_fullness * capacity →
    added_water = final_amount - initial_amount →
    added_water = 27 :=
by
  intros
  -- Insert the proof here
  sorry

end added_water_correct_l33_33219


namespace solve_equation_l33_33852

theorem solve_equation (x : ℝ) :
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) → x = -9 :=
by 
  sorry

end solve_equation_l33_33852


namespace probability_heads_at_most_three_out_of_ten_coins_l33_33546

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l33_33546


namespace find_value_l33_33964

open Classical

variables (a b c : ℝ)

-- Assume a, b, c are roots of the polynomial x^3 - 24x^2 + 50x - 42
def is_root (x : ℝ) : Prop := x^3 - 24*x^2 + 50*x - 42 = 0

-- Vieta's formulas for the given polynomial
axiom h1 : is_root a
axiom h2 : is_root b
axiom h3 : is_root c
axiom h4 : a + b + c = 24
axiom h5 : a * b + b * c + c * a = 50
axiom h6 : a * b * c = 42

-- We want to prove the given expression equals 476/43
theorem find_value : 
  (a/(1/a + b*c) + b/(1/b + c*a) + c/(1/c + a*b) = 476/43) :=
sorry

end find_value_l33_33964


namespace coefficient_sum_of_squares_is_23456_l33_33953

theorem coefficient_sum_of_squares_is_23456 
  (p q r s t u : ℤ)
  (h : ∀ x : ℤ, 1728 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) :
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 23456 := 
by
  sorry

end coefficient_sum_of_squares_is_23456_l33_33953


namespace probability_of_at_most_3_heads_out_of_10_l33_33582
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l33_33582


namespace first_point_x_coord_l33_33957

variables (m n : ℝ)

theorem first_point_x_coord (h1 : m = 2 * n + 5) (h2 : m + 5 = 2 * (n + 2.5) + 5) : 
  m = 2 * n + 5 :=
by 
  sorry

end first_point_x_coord_l33_33957


namespace area_of_right_triangle_l33_33522

-- Given definitions
def leg_a : ℝ := 30
def hypotenuse_c : ℝ := 34

-- The theorem statement
theorem area_of_right_triangle : 
  ∀ (b : ℝ), b = real.sqrt (hypotenuse_c^2 - leg_a^2) → 
  let area := 1 / 2 * leg_a * b in
  area = 240 := 
by
  intro b
  intro h
  let area := 1 / 2 * leg_a * b
  sorry

end area_of_right_triangle_l33_33522


namespace ab_value_l33_33673

-- Defining the conditions as Lean assumptions
theorem ab_value (a b c : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) (h3 : a + b + c = 10) : a * b = 9 :=
by
  sorry

end ab_value_l33_33673


namespace rectangle_dimensions_exist_l33_33999

theorem rectangle_dimensions_exist :
  ∃ (a b c d : ℕ), (a * b + c * d = 81) ∧ (2 * (a + b) = 2 * 2 * (c + d) ∨ 2 * (c + d) = 2 * 2 * (a + b)) :=
by sorry

end rectangle_dimensions_exist_l33_33999
