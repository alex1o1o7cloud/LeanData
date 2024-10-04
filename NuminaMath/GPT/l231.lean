import Mathlib

namespace hyperbola_sufficient_condition_l231_231544

-- Define the condition for the equation to represent a hyperbola
def represents_hyperbola (k : ℝ) : Prop :=
  (3 - k) * (k - 1) < 0

-- Lean 4 statement to prove that k > 3 is a sufficient condition for the given equation
theorem hyperbola_sufficient_condition (k : ℝ) (h : k > 3) :
  represents_hyperbola k :=
sorry

end hyperbola_sufficient_condition_l231_231544


namespace largest_package_markers_l231_231837

def Alex_markers : ℕ := 36
def Becca_markers : ℕ := 45
def Charlie_markers : ℕ := 60

theorem largest_package_markers (d : ℕ) :
  d ∣ Alex_markers ∧ d ∣ Becca_markers ∧ d ∣ Charlie_markers → d ≤ 3 :=
by
  sorry

end largest_package_markers_l231_231837


namespace largest_whole_number_l231_231946

theorem largest_whole_number (x : ℕ) (h1 : 9 * x < 150) : x ≤ 16 :=
by sorry

end largest_whole_number_l231_231946


namespace prove_a1_geq_2k_l231_231672

variable (n k : ℕ) (a : ℕ → ℕ)
variable (h1: ∀ i, 1 ≤ i → i ≤ n → 1 < a i)
variable (h2: ∀ i j, 1 ≤ i → i < j → j ≤ n → ¬ (a i ∣ a j))
variable (h3: 3^k < 2*n ∧ 2*n < 3^(k + 1))

theorem prove_a1_geq_2k : a 1 ≥ 2^k :=
by
  sorry

end prove_a1_geq_2k_l231_231672


namespace greatest_number_l231_231262

-- Define the base conversions
def octal_to_decimal (n : Nat) : Nat := 3 * 8^1 + 2
def quintal_to_decimal (n : Nat) : Nat := 1 * 5^2 + 1 * 5^1 + 1
def binary_to_decimal (n : Nat) : Nat := 1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0
def senary_to_decimal (n : Nat) : Nat := 5 * 6^1 + 4

theorem greatest_number :
  max (max (octal_to_decimal 32) (quintal_to_decimal 111)) (max (binary_to_decimal 101010) (senary_to_decimal 54))
  = binary_to_decimal 101010 := by sorry

end greatest_number_l231_231262


namespace least_x_l231_231243

noncomputable def is_odd_prime (n : ℕ) : Prop :=
  n > 1 ∧ Prime n ∧ n % 2 = 1

theorem least_x (x p : ℕ) (hp : Prime p) (hx : x > 0) (hodd_prime : is_odd_prime (x / (12 * p))) : x = 72 := 
  sorry

end least_x_l231_231243


namespace sum_series_equals_4_div_9_l231_231476

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l231_231476


namespace tabs_per_window_l231_231905

def totalTabs (browsers windowsPerBrowser tabsOpened : Nat) : Nat :=
  tabsOpened / (browsers * windowsPerBrowser)

theorem tabs_per_window : totalTabs 2 3 60 = 10 := by
  sorry

end tabs_per_window_l231_231905


namespace infinite_series_sum_l231_231529

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l231_231529


namespace sum_of_factors_36_l231_231651

theorem sum_of_factors_36 : (∑ i in {d ∣ 36 | i}, i) = 91 := by
  sorry

end sum_of_factors_36_l231_231651


namespace prime_condition_l231_231868

theorem prime_condition (p : ℕ) [Fact (Nat.Prime p)] :
  (∀ (a : ℕ), (1 < a ∧ a < p / 2) → (∃ (b : ℕ), (p / 2 < b ∧ b < p) ∧ p ∣ (a * b - 1))) ↔ (p = 5 ∨ p = 7 ∨ p = 13) := by
  sorry

end prime_condition_l231_231868


namespace probability_first_queen_second_diamond_l231_231053

def is_queen (card : ℕ) : Prop :=
card = 12 ∨ card = 25 ∨ card = 38 ∨ card = 51

def is_diamond (card : ℕ) : Prop :=
card % 13 = 0

def first_card_queen (first_card : ℕ) (cards : finset ℕ) : Prop :=
is_queen first_card

def second_card_diamond (second_card : ℕ) (remaining_cards : finset ℕ) : Prop :=
is_diamond second_card

theorem probability_first_queen_second_diamond (cards : finset ℕ) (h : cards.card = 52) :
  ((\sum x in cards.filter is_queen, 1 / 52) * (\sum y in cards.filter is_diamond, 1 / 51)) = (1 / 52) :=
sorry

end probability_first_queen_second_diamond_l231_231053


namespace total_surface_area_l231_231841

noncomputable def calculate_surface_area
  (radius : ℝ) (reflective : Bool) : ℝ :=
  let base_area := (radius^2 * Real.pi)
  let curved_surface_area := (4 * Real.pi * (radius^2)) / 2
  let effective_surface_area := if reflective then 2 * curved_surface_area else curved_surface_area
  effective_surface_area

theorem total_surface_area (r : ℝ) (h₁_reflective : Bool) (h₂_reflective : Bool) :
  r = 8 →
  h₁_reflective = false →
  h₂_reflective = true →
  (calculate_surface_area r h₁_reflective + calculate_surface_area r h₂_reflective) = 384 * Real.pi := 
by
  sorry

end total_surface_area_l231_231841


namespace order_of_y1_y2_y3_l231_231299

/-
Given three points A(-3, y1), B(3, y2), and C(4, y3) all lie on the parabola y = 2*(x - 2)^2 + 1,
prove that y2 < y3 < y1.
-/
theorem order_of_y1_y2_y3 :
  let y1 := 2 * (-3 - 2)^2 + 1
  let y2 := 2 * (3 - 2)^2 + 1
  let y3 := 2 * (4 - 2)^2 + 1
  y2 < y3 ∧ y3 < y1 :=
by
  let y1 := 2 * (-3 - 2)^2 + 1
  let y2 := 2 * (3 - 2)^2 + 1
  let y3 := 2 * (4 - 2)^2 + 1
  sorry

end order_of_y1_y2_y3_l231_231299


namespace triangle_rectangle_ratio_l231_231979

-- Definitions of the perimeter conditions and the relationship between length and width of the rectangle.
def equilateral_triangle_side_length (t : ℕ) : Prop :=
  3 * t = 24

def rectangle_dimensions (l w : ℕ) : Prop :=
  2 * l + 2 * w = 24 ∧ l = 2 * w

-- The main theorem stating the desired ratio.
theorem triangle_rectangle_ratio (t l w : ℕ) 
  (ht : equilateral_triangle_side_length t) (hlw : rectangle_dimensions l w) : t / w = 2 :=
by
  sorry

end triangle_rectangle_ratio_l231_231979


namespace analytical_expression_of_odd_function_l231_231553

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 2 * x + 3 
else if x = 0 then 0 
else -x^2 - 2 * x - 3

theorem analytical_expression_of_odd_function :
  ∀ x : ℝ, f x =
    if x > 0 then x^2 - 2 * x + 3 
    else if x = 0 then 0 
    else -x^2 - 2 * x - 3 :=
by
  sorry

end analytical_expression_of_odd_function_l231_231553


namespace leakage_empty_time_l231_231669

variables (a : ℝ) (h1 : a > 0) -- Assuming a is positive for the purposes of the problem

theorem leakage_empty_time (h : 7 * a > 0) : (7 * a) / 6 = 7 * a / 6 :=
by
  sorry

end leakage_empty_time_l231_231669


namespace octahedral_dice_sum_16_probability_l231_231793

theorem octahedral_dice_sum_16_probability : 
  let outcomes := (1:8) × (1:8)
  let successful_outcome := (8, 8) in
  (∑ outcome in outcomes, if outcome.1 + outcome.2 = 16 then 1 else 0) / (8 * 8) = 1 / 64 :=
sorry

end octahedral_dice_sum_16_probability_l231_231793


namespace consecutive_natural_numbers_sum_l231_231796

theorem consecutive_natural_numbers_sum :
  (∃ (n : ℕ), 0 < n → n ≤ 4 ∧ (n-1) + n + (n+1) ≤ 12) → 
  (∃ n_sets : ℕ, n_sets = 4) :=
by
  sorry

end consecutive_natural_numbers_sum_l231_231796


namespace average_of_11_results_l231_231953

theorem average_of_11_results 
  (S1: ℝ) (S2: ℝ) (fifth_result: ℝ) -- Define the variables
  (h1: S1 / 5 = 49)                -- sum of the first 5 results
  (h2: S2 / 7 = 52)                -- sum of the last 7 results
  (h3: fifth_result = 147)         -- the fifth result 
  : (S1 + S2 - fifth_result) / 11 = 42 := -- statement of the problem
by
  sorry

end average_of_11_results_l231_231953


namespace red_balls_in_box_l231_231172

theorem red_balls_in_box {n : ℕ} (h : n = 6) (p : (∃ (r : ℕ), r / 6 = 1 / 3)) : ∃ r, r = 2 :=
by
  sorry

end red_balls_in_box_l231_231172


namespace ratio_of_triangle_side_to_rectangle_width_l231_231982

variables (t w l : ℕ)

-- Condition 1: The perimeter of the equilateral triangle is 24 inches
def triangle_perimeter := 3 * t = 24

-- Condition 2: The perimeter of the rectangle is 24 inches
def rectangle_perimeter := 2 * l + 2 * w = 24

-- Condition 3: The length of the rectangle is twice its width
def length_double_width := l = 2 * w

-- The ratio of the side length of the triangle to the width of the rectangle is 2
theorem ratio_of_triangle_side_to_rectangle_width
    (h_triangle : triangle_perimeter t)
    (h_rectangle : rectangle_perimeter l w)
    (h_length_width : length_double_width l w) :
    t / w = 2 :=
by
    sorry

end ratio_of_triangle_side_to_rectangle_width_l231_231982


namespace number_of_partition_chains_l231_231828

theorem number_of_partition_chains (n : ℕ) (h : n > 0) : 
  ∃ (num_chains : ℕ), num_chains = (n! * (n-1)!) / 2^(n-1) :=
by 
  exists (\(n! * (n-1)!) / 2^(n-1));
  sorry

end number_of_partition_chains_l231_231828


namespace hiker_final_distance_l231_231397

-- Definitions of the movements
def northward_movement : ℤ := 20
def southward_movement : ℤ := 8
def westward_movement : ℤ := 15
def eastward_movement : ℤ := 10

-- Definitions of the net movements
def net_north_south_movement : ℤ := northward_movement - southward_movement
def net_east_west_movement : ℤ := westward_movement - eastward_movement

-- The proof statement
theorem hiker_final_distance : 
  (net_north_south_movement^2 + net_east_west_movement^2) = 13^2 := by 
    sorry

end hiker_final_distance_l231_231397


namespace num_male_rabbits_l231_231899

/-- 
There are 12 white rabbits and 9 black rabbits. 
There are 8 female rabbits. 
Prove that the number of male rabbits is 13.
-/
theorem num_male_rabbits (white_rabbits : ℕ) (black_rabbits : ℕ) (female_rabbits: ℕ) 
  (h_white : white_rabbits = 12) (h_black : black_rabbits = 9) (h_female : female_rabbits = 8) :
  (white_rabbits + black_rabbits - female_rabbits = 13) :=
by
  sorry

end num_male_rabbits_l231_231899


namespace unique_pair_fraction_l231_231921

theorem unique_pair_fraction (p : ℕ) (hprime : Nat.Prime p) (hodd : p % 2 = 1) :
  ∃! (n m : ℕ), (n ≠ m) ∧ (2 / (p : ℚ) = 1 / (n : ℚ) + 1 / (m : ℚ)) ∧
  ((n = (p + 1) / 2 ∧ m = p * (p + 1) / 2) ∨ (m = (p + 1) / 2 ∧ n = p * (p + 1) / 2)) := sorry

end unique_pair_fraction_l231_231921


namespace conditions_iff_positive_l231_231142

theorem conditions_iff_positive (a b : ℝ) (h₁ : a + b > 0) (h₂ : ab > 0) : 
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ ab > 0) :=
sorry

end conditions_iff_positive_l231_231142


namespace intersection_of_A_and_B_l231_231874

def setA : Set ℝ := { x | x^2 - 3 * x - 4 < 0 }
def setB : Set ℝ := {-4, 1, 3, 5}
def resultSet : Set ℝ := {1, 3}

theorem intersection_of_A_and_B :
  setA ∩ setB = resultSet := 
by
  sorry

end intersection_of_A_and_B_l231_231874


namespace length_AB_is_4_l231_231145

section HyperbolaProof

/-- Define the hyperbola -/
def hyperbola (x y : ℝ) : Prop :=
  (x^2 / 16) - (y^2 / 8) = 1

/-- Define the line l given by x = 2√6 -/
def line_l (x : ℝ) : Prop :=
  x = 2 * Real.sqrt 6

/-- Define the condition for intersection points -/
def intersect_points (x y : ℝ) : Prop :=
  hyperbola x y ∧ line_l x

/-- Prove the length of the line segment AB is 4 -/
theorem length_AB_is_4 :
  ∀ y : ℝ, intersect_points (2 * Real.sqrt 6) y → |y| = 2 → length_AB = 4 :=
sorry

end HyperbolaProof

end length_AB_is_4_l231_231145


namespace trig_identity_l231_231714

theorem trig_identity (α : ℝ) (h : Real.tan α = 4) : 
  (1 + Real.cos (2 * α) + 8 * Real.sin α ^ 2) / Real.sin (2 * α) = 65 / 4 :=
by
  sorry

end trig_identity_l231_231714


namespace range_of_a_l231_231297

variable {a : ℝ}

def A (a : ℝ) : Set ℝ := { x | (x - 2) * (x - (a + 1)) < 0 }
def B (a : ℝ) : Set ℝ := { x | (x - 2 * a) / (x - (a^2 + 1)) < 0 }

theorem range_of_a (a : ℝ) : B a ⊆ A a ↔ (a = -1 / 2) ∨ (2 ≤ a ∧ a ≤ 3) := by
  sorry

end range_of_a_l231_231297


namespace B_k_largest_at_45_l231_231120

def B_k (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.1)^k

theorem B_k_largest_at_45 : ∀ k : ℕ, k = 45 → ∀ m : ℕ, m ≠ 45 → B_k 45 > B_k m :=
by
  intro k h_k m h_m
  sorry

end B_k_largest_at_45_l231_231120


namespace max_n_satisfying_inequality_l231_231898

theorem max_n_satisfying_inequality : 
  ∃ (n : ℤ), 303 * n^3 ≤ 380000 ∧ ∀ m : ℤ, m > n → 303 * m^3 > 380000 := sorry

end max_n_satisfying_inequality_l231_231898


namespace pto_shirts_total_cost_l231_231367

theorem pto_shirts_total_cost :
  let cost_Kindergartners : ℝ := 101 * 5.80
  let cost_FirstGraders : ℝ := 113 * 5.00
  let cost_SecondGraders : ℝ := 107 * 5.60
  let cost_ThirdGraders : ℝ := 108 * 5.25
  cost_Kindergartners + cost_FirstGraders + cost_SecondGraders + cost_ThirdGraders = 2317.00 := by
  sorry

end pto_shirts_total_cost_l231_231367


namespace Linda_purchase_cost_l231_231598

def price_peanuts : ℝ := sorry
def price_berries : ℝ := sorry
def price_coconut : ℝ := sorry
def price_dates : ℝ := sorry

theorem Linda_purchase_cost:
  ∃ (p b c d : ℝ), 
    (p + b + c + d = 30) ∧ 
    (3 * p = d) ∧
    ((p + b) / 2 = c) ∧
    (b + c = 65 / 9) :=
sorry

end Linda_purchase_cost_l231_231598


namespace susan_annual_percentage_increase_l231_231757

theorem susan_annual_percentage_increase :
  let initial_jerry := 14400
  let initial_susan := 6250
  let jerry_first_year := initial_jerry * (6 / 5 : ℝ)
  let jerry_second_year := jerry_first_year * (9 / 10 : ℝ)
  let jerry_third_year := jerry_second_year * (6 / 5 : ℝ)
  jerry_third_year = 18662.40 →
  (initial_susan : ℝ) * (1 + r)^3 = 18662.40 →
  r = 0.44 :=
by {
  sorry
}

end susan_annual_percentage_increase_l231_231757


namespace cherries_per_quart_of_syrup_l231_231755

-- Definitions based on conditions
def time_to_pick_cherries : ℚ := 2
def cherries_picked_in_time : ℚ := 300
def time_to_make_syrup : ℚ := 3
def total_time_for_all_syrup : ℚ := 33
def total_quarts : ℚ := 9

-- Derivation of how many cherries are needed per quart
theorem cherries_per_quart_of_syrup : 
  (cherries_picked_in_time / time_to_pick_cherries) * (total_time_for_all_syrup - total_quarts * time_to_make_syrup) / total_quarts = 100 :=
by
  repeat { sorry }

end cherries_per_quart_of_syrup_l231_231755


namespace stratified_sampling_red_balls_l231_231001

theorem stratified_sampling_red_balls (total_balls red_balls sample_size : ℕ) (h_total : total_balls = 100) (h_red : red_balls = 20) (h_sample : sample_size = 10) :
  (sample_size * (red_balls / total_balls)) = 2 := by
  sorry

end stratified_sampling_red_balls_l231_231001


namespace find_k_l231_231748

theorem find_k (k x : ℝ) (h1 : x + k - 4 = 0) (h2 : x = 2) : k = 2 :=
by
  sorry

end find_k_l231_231748


namespace sum_of_consecutive_integers_largest_set_of_consecutive_positive_integers_l231_231375

theorem sum_of_consecutive_integers (n : ℕ) (a : ℕ) (h : n ≥ 1) (h_sum : n * (2 * a + n - 1) = 56) : n ≤ 7 := 
by
  sorry

theorem largest_set_of_consecutive_positive_integers : ∃ n a, n ≥ 1 ∧ n * (2 * a + n - 1) = 56 ∧ n = 7 := 
by
  use 7, 1
  repeat {split}
  sorry

end sum_of_consecutive_integers_largest_set_of_consecutive_positive_integers_l231_231375


namespace older_friend_is_38_l231_231941

-- Define the conditions
def younger_friend_age (x : ℕ) : Prop := 
  ∃ (y : ℕ), (y = x + 2 ∧ x + y = 74)

-- Define the age of the older friend
def older_friend_age (x : ℕ) : ℕ := x + 2

-- State the theorem
theorem older_friend_is_38 : ∃ x, younger_friend_age x ∧ older_friend_age x = 38 :=
by
  sorry

end older_friend_is_38_l231_231941


namespace increased_speed_l231_231787

theorem increased_speed (S : ℝ) : 
  (∀ (usual_speed : ℝ) (usual_time : ℝ) (distance : ℝ), 
    usual_speed = 20 ∧ distance = 100 ∧ usual_speed * usual_time = distance ∧ S * (usual_time - 1) = distance) → 
  S = 25 :=
by
  intros h1
  sorry

end increased_speed_l231_231787


namespace sum_series_equals_4_div_9_l231_231477

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l231_231477


namespace f_1997_leq_666_l231_231191

noncomputable def f : ℕ+ → ℕ := sorry

axiom f_mn_inequality : ∀ (m n : ℕ+), f (m + n) ≥ f m + f n
axiom f_two : f 2 = 0
axiom f_three_pos : f 3 > 0
axiom f_9999 : f 9999 = 3333

theorem f_1997_leq_666 : f 1997 ≤ 666 := sorry

end f_1997_leq_666_l231_231191


namespace groceries_delivered_amount_l231_231407

noncomputable def alex_saved_up : ℝ := 14500
noncomputable def car_cost : ℝ := 14600
noncomputable def charge_per_trip : ℝ := 1.5
noncomputable def percentage_charge : ℝ := 0.05
noncomputable def number_of_trips : ℕ := 40

theorem groceries_delivered_amount :
  ∃ G : ℝ, charge_per_trip * number_of_trips + percentage_charge * G = car_cost - alex_saved_up ∧ G = 800 :=
by {
  use 800,
  rw [mul_comm (800 : ℝ), mul_assoc],
  norm_num,
  exact add_comm 60 (40 : ℝ),
  sorry
}

end groceries_delivered_amount_l231_231407


namespace series_sum_eq_four_ninths_l231_231496

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l231_231496


namespace abc_min_value_l231_231187

open Real

theorem abc_min_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_sum : a + b + c = 1) (h_bound : a ≤ b ∧ b ≤ c ∧ c ≤ 3 * a) :
  3 * a * a * (1 - 4 * a) = (9/343) := 
sorry

end abc_min_value_l231_231187


namespace johnny_hours_second_job_l231_231180

theorem johnny_hours_second_job (x : ℕ) (h_eq : 5 * (69 + 10 * x) = 445) : x = 2 :=
by 
  -- The proof will go here, but we skip it as per the instructions
  sorry

end johnny_hours_second_job_l231_231180


namespace find_a_l231_231147

theorem find_a (a : ℝ) (f : ℝ → ℝ) (h : f = fun x => a * x ^ 3 - 3 * x) (h1 : f (-1) = 4) : a = -1 :=
by
  sorry

end find_a_l231_231147


namespace min_value_fraction_l231_231857

theorem min_value_fraction : ∃ (x : ℝ), (∀ y : ℝ, (y^2 + 9) / (Real.sqrt (y^2 + 5)) ≥ (9 * Real.sqrt 5) / 5)
  := sorry

end min_value_fraction_l231_231857


namespace infinite_series_sum_l231_231530

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l231_231530


namespace probability_queen_then_diamond_l231_231054

-- Define a standard deck of 52 cards
def deck := List.range 52

-- Define a function to check if a card is a Queen
def is_queen (card : ℕ) : Prop :=
card % 13 = 10

-- Define a function to check if a card is a Diamond. Here assuming index for diamond starts at 0 and ends at 12
def is_diamond (card : ℕ) : Prop :=
card / 13 = 0

-- The main theorem statement
theorem probability_queen_then_diamond : 
  let prob := 1 / 52 * 12 / 51 + 3 / 52 * 13 / 51
  prob = 52 / 221 :=
by
  sorry

end probability_queen_then_diamond_l231_231054


namespace eval_F_at_4_f_5_l231_231572

def f (a : ℤ) : ℤ := 3 * a - 6
def F (a : ℤ) (b : ℤ) : ℤ := 2 * b ^ 2 + 3 * a

theorem eval_F_at_4_f_5 : F 4 (f 5) = 174 := by
  sorry

end eval_F_at_4_f_5_l231_231572


namespace order_of_y1_y2_y3_l231_231300

/-
Given three points A(-3, y1), B(3, y2), and C(4, y3) all lie on the parabola y = 2*(x - 2)^2 + 1,
prove that y2 < y3 < y1.
-/
theorem order_of_y1_y2_y3 :
  let y1 := 2 * (-3 - 2)^2 + 1
  let y2 := 2 * (3 - 2)^2 + 1
  let y3 := 2 * (4 - 2)^2 + 1
  y2 < y3 ∧ y3 < y1 :=
by
  let y1 := 2 * (-3 - 2)^2 + 1
  let y2 := 2 * (3 - 2)^2 + 1
  let y3 := 2 * (4 - 2)^2 + 1
  sorry

end order_of_y1_y2_y3_l231_231300


namespace triangle_inequality_inequality_l231_231195

variable {a b c : ℝ}
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (triangle_ineq : a + b > c)

theorem triangle_inequality_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) (triangle_ineq : a + b > c) :
  a^3 + b^3 + 3 * a * b * c > c^3 :=
sorry

end triangle_inequality_inequality_l231_231195


namespace value_of_P_dot_Q_l231_231347

def P : Set ℝ := {x | Real.log x / Real.log 2 < 1}
def Q : Set ℝ := {x | abs (x - 2) < 1}
def P_dot_Q (P Q : Set ℝ) : Set ℝ := {x | x ∈ P ∧ x ∉ Q}

theorem value_of_P_dot_Q : P_dot_Q P Q = {x | 0 < x ∧ x ≤ 1} := by
  sorry

end value_of_P_dot_Q_l231_231347


namespace expression_equals_one_l231_231110

theorem expression_equals_one : 
  (Real.sqrt 6 / Real.sqrt 2) + abs (1 - Real.sqrt 3) - Real.sqrt 12 + (1 / 2)⁻¹ = 1 := 
by sorry

end expression_equals_one_l231_231110


namespace crayons_total_l231_231415

theorem crayons_total (Billy_crayons : ℝ) (Jane_crayons : ℝ)
  (h1 : Billy_crayons = 62.0) (h2 : Jane_crayons = 52.0) :
  Billy_crayons + Jane_crayons = 114.0 := 
by
  sorry

end crayons_total_l231_231415


namespace power_mod_l231_231542

theorem power_mod (a : ℕ) : 5 ^ 2023 % 17 = 2 := by
  sorry

end power_mod_l231_231542


namespace am_gm_inequality_l231_231924

theorem am_gm_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) : 
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * (a * b * c)^2 :=
by
  sorry

end am_gm_inequality_l231_231924


namespace at_most_two_greater_than_one_l231_231344

theorem at_most_two_greater_than_one (a b c : ℝ) (h : a * b * c = 1) :
  ¬ (2 * a - 1 / b > 1 ∧ 2 * b - 1 / c > 1 ∧ 2 * c - 1 / a > 1) :=
by
  sorry

end at_most_two_greater_than_one_l231_231344


namespace second_day_speed_faster_l231_231679

def first_day_distance := 18
def first_day_speed := 3
def first_day_time := first_day_distance / first_day_speed
def second_day_time := first_day_time - 1
def third_day_speed := 5
def third_day_time := 3
def third_day_distance := third_day_speed * third_day_time
def total_distance := 53

theorem second_day_speed_faster :
  ∃ r2, (first_day_distance + (second_day_time * r2) + third_day_distance = total_distance) → (r2 - first_day_speed = 1) :=
by
  sorry

end second_day_speed_faster_l231_231679


namespace simpsons_paradox_example_l231_231383

theorem simpsons_paradox_example :
  ∃ n1 n2 a1 a2 b1 b2,
    n1 = 10 ∧ a1 = 3 ∧ b1 = 2 ∧
    n2 = 90 ∧ a2 = 45 ∧ b2 = 488 ∧
    ((a1 : ℝ) / n1 > (b1 : ℝ) / n1) ∧
    ((a2 : ℝ) / n2 > (b2 : ℝ) / n2) ∧
    ((a1 + a2 : ℝ) / (n1 + n2) < (b1 + b2 : ℝ) / (n1 + n2)) :=
by
  use 10, 90, 3, 45, 2, 488
  simp
  sorry

end simpsons_paradox_example_l231_231383


namespace largest_angle_of_consecutive_odd_int_angles_is_125_l231_231045

-- Definitions for a convex hexagon with six consecutive odd integer interior angles
def is_consecutive_odd_integers (xs : List ℕ) : Prop :=
  ∀ n, 0 ≤ n ∧ n < 5 → xs.get! n + 2 = xs.get! (n + 1)

def hexagon_angles_sum_720 (xs : List ℕ) : Prop :=
  xs.length = 6 ∧ xs.sum = 720

-- Main theorem statement
theorem largest_angle_of_consecutive_odd_int_angles_is_125 (xs : List ℕ) 
(h1 : is_consecutive_odd_integers xs) 
(h2 : hexagon_angles_sum_720 xs) : 
  xs.maximum = 125 := 
sorry

end largest_angle_of_consecutive_odd_int_angles_is_125_l231_231045


namespace larger_number_of_two_l231_231807

theorem larger_number_of_two (x y : ℝ) (h1 : x - y = 3) (h2 : x + y = 29) (h3 : x * y > 200) : x = 16 :=
by sorry

end larger_number_of_two_l231_231807


namespace mul_102_102_l231_231117

theorem mul_102_102 : 102 * 102 = 10404 := by
  sorry

end mul_102_102_l231_231117


namespace find_t_given_V_S_l231_231566

variables (g V V0 S S0 a t : ℝ)

theorem find_t_given_V_S :
  (V = g * (t - a) + V0) →
  (S = (1 / 2) * g * (t - a) ^ 2 + V0 * (t - a) + S0) →
  t = a + (V - V0) / g :=
by
  intros h1 h2
  sorry

end find_t_given_V_S_l231_231566


namespace route_y_slower_by_2_4_minutes_l231_231021
noncomputable def time_route_x : ℝ := (7 : ℝ) / (35 : ℝ)
noncomputable def time_downtown_y : ℝ := (1 : ℝ) / (10 : ℝ)
noncomputable def time_other_y : ℝ := (7 : ℝ) / (50 : ℝ)
noncomputable def time_route_y : ℝ := time_downtown_y + time_other_y

theorem route_y_slower_by_2_4_minutes :
  ((time_route_y - time_route_x) * 60) = 2.4 :=
by
  -- Provide the required proof here
  sorry

end route_y_slower_by_2_4_minutes_l231_231021


namespace find_larger_number_of_two_l231_231225

theorem find_larger_number_of_two (A B : ℕ) (hcf lcm : ℕ) (factor1 factor2 : ℕ)
  (h_hcf : hcf = 23)
  (h_factor1 : factor1 = 13)
  (h_factor2 : factor2 = 16)
  (h_lcm : lcm = hcf * factor1 * factor2)
  (h_A : A = hcf * m ∧ m = factor1)
  (h_B : B = hcf * n ∧ n = factor2):
  max A B = 368 := by
  sorry

end find_larger_number_of_two_l231_231225


namespace num_points_common_to_graphs_l231_231993

theorem num_points_common_to_graphs :
  (∃ (x y : ℝ), (2 * x - y + 3 = 0 ∧ x + y - 3 = 0)) ∧
  (∃ (x y : ℝ), (2 * x - y + 3 = 0 ∧ 3 * x - 4 * y + 8 = 0)) ∧
  (∃ (x y : ℝ), (4 * x + y - 5 = 0 ∧ x + y - 3 = 0)) ∧
  (∃ (x y : ℝ), (4 * x + y - 5 = 0 ∧ 3 * x - 4 * y + 8 = 0)) ∧
  ∀ (x y : ℝ), ((2 * x - y + 3 = 0 ∨ 4 * x + y - 5 = 0) ∧ (x + y - 3 = 0 ∨ 3 * x - 4 * y + 8 = 0)) →
  ∃ (p1 p2 p3 p4 : ℝ × ℝ), 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 :=
sorry

end num_points_common_to_graphs_l231_231993


namespace gas_cost_per_gallon_l231_231700

-- Define the conditions as Lean definitions
def miles_per_gallon : ℕ := 32
def total_miles : ℕ := 336
def total_cost : ℕ := 42

-- Prove the cost of gas per gallon, which is $4 per gallon
theorem gas_cost_per_gallon : total_cost / (total_miles / miles_per_gallon) = 4 :=
by
  sorry

end gas_cost_per_gallon_l231_231700


namespace a_range_l231_231555

noncomputable def f (x : ℝ) : ℝ :=
  4 * Real.log x - (1 / 2) * x^2 + 3 * x

def is_monotonic_on_interval (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a (a + 1), 4 / x - x + 3 > 0

theorem a_range (a : ℝ) :
  is_monotonic_on_interval a → (0 < a ∧ a ≤ 3) :=
by 
  sorry

end a_range_l231_231555


namespace intersection_A_B_l231_231886

-- Define set A and set B based on the conditions
def set_A : Set ℝ := {x : ℝ | x^2 - 3 * x - 4 < 0}
def set_B : Set ℝ := {-4, 1, 3, 5}

-- State the theorem to prove the intersection of A and B
theorem intersection_A_B : set_A ∩ set_B = {1, 3} :=
by sorry

end intersection_A_B_l231_231886


namespace sum_series_l231_231471

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l231_231471


namespace base_5_division_quotient_l231_231531

theorem base_5_division_quotient :
  base_to_nat 5 [2, 4, 3, 4, 2] / base_to_nat 5 [2, 3] = base_to_nat 5 [4, 3] :=
sorry

end base_5_division_quotient_l231_231531


namespace sin_690_eq_neg_0_5_l231_231666

theorem sin_690_eq_neg_0_5 : Real.sin (690 * Real.pi / 180) = -0.5 := by
  sorry

end sin_690_eq_neg_0_5_l231_231666


namespace larger_number_l231_231226

noncomputable def hcf (a b : ℕ) := Nat.gcd a b
noncomputable def lcm (a b : ℕ) := (a * b) / (hcf a b)

theorem larger_number
  (A B : ℕ)
  (h : hcf A B = 60)
  (h1 : ∃ (m n : ℕ), lcm A B = 60 * 11 * 15 ∧ A = 60 * m ∧ B = 60 * n ∧ Nat.coprime m n ∧ m * n = 11 * 15) :
  max A B = 900 :=
by
  sorry -- proof omitted

end larger_number_l231_231226


namespace sum_series_l231_231469

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l231_231469


namespace shell_highest_point_time_l231_231258

theorem shell_highest_point_time (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : a * 7^2 + b * 7 + c = a * 14^2 + b * 14 + c) :
  (-b / (2 * a)) = 10.5 :=
by
  -- The proof is omitted as per the instructions
  sorry

end shell_highest_point_time_l231_231258


namespace N_is_perfect_square_l231_231182

def N (n : ℕ) : ℕ :=
  (10^(2*n+1) - 1) / 9 * 10 + 
  2 * (10^(n+1) - 1) / 9 + 25

theorem N_is_perfect_square (n : ℕ) : ∃ k, k^2 = N n :=
  sorry

end N_is_perfect_square_l231_231182


namespace series_sum_eval_l231_231283

noncomputable def series_sum (n : ℕ) : ℚ :=
  ∑ k in Finset.range (n + 1), 1 / ((4 * k - 3) * (4 * k + 1) : ℚ)

theorem series_sum_eval (n : ℕ) : series_sum n = n / (4 * n + 1) :=
  sorry

end series_sum_eval_l231_231283


namespace average_speed_of_trip_l231_231073

theorem average_speed_of_trip :
  let distance_local := 60
  let speed_local := 20
  let distance_highway := 120
  let speed_highway := 60
  let total_distance := distance_local + distance_highway
  let time_local := distance_local / speed_local
  let time_highway := distance_highway / speed_highway
  let total_time := time_local + time_highway
  let average_speed := total_distance / total_time
  average_speed = 36 := 
by 
  sorry

end average_speed_of_trip_l231_231073


namespace boat_speed_in_still_water_l231_231417

variables (V_b V_c V_w : ℝ)

-- Conditions from the problem
def speed_upstream (V_b V_c V_w : ℝ) : ℝ := V_b - V_c - V_w
def water_current_range (V_c : ℝ) : Prop := 2 ≤ V_c ∧ V_c ≤ 4
def wind_resistance_range (V_w : ℝ) : Prop := -1 ≤ V_w ∧ V_w ≤ 1
def upstream_speed : Prop := speed_upstream V_b 4 (2 - (-1)) + (2 - -1) = 4

-- Statement of the proof problem
theorem boat_speed_in_still_water :
  (∀ V_c V_w, water_current_range V_c → wind_resistance_range V_w → speed_upstream V_b V_c V_w = 4) → V_b = 7 :=
by
  sorry

end boat_speed_in_still_water_l231_231417


namespace four_digit_numbers_no_5s_8s_l231_231896

def count_valid_four_digit_numbers : Nat :=
  let thousand_place := 7  -- choices: 1, 2, 3, 4, 6, 7, 9
  let other_places := 8  -- choices: 0, 1, 2, 3, 4, 6, 7, 9
  thousand_place * other_places * other_places * other_places

theorem four_digit_numbers_no_5s_8s : count_valid_four_digit_numbers = 3584 :=
by
  rfl

end four_digit_numbers_no_5s_8s_l231_231896


namespace probability_first_queen_second_diamond_l231_231051

def is_queen (card : ℕ) : Prop :=
card = 12 ∨ card = 25 ∨ card = 38 ∨ card = 51

def is_diamond (card : ℕ) : Prop :=
card % 13 = 0

def first_card_queen (first_card : ℕ) (cards : finset ℕ) : Prop :=
is_queen first_card

def second_card_diamond (second_card : ℕ) (remaining_cards : finset ℕ) : Prop :=
is_diamond second_card

theorem probability_first_queen_second_diamond (cards : finset ℕ) (h : cards.card = 52) :
  ((\sum x in cards.filter is_queen, 1 / 52) * (\sum y in cards.filter is_diamond, 1 / 51)) = (1 / 52) :=
sorry

end probability_first_queen_second_diamond_l231_231051


namespace hcf_of_two_numbers_l231_231229

theorem hcf_of_two_numbers (A B : ℕ) (h1 : A * B = 4107) (h2 : A = 111) : (Nat.gcd A B) = 37 :=
by
  -- Given conditions
  have h3 : B = 37 := by
    -- Deduce B from given conditions
    sorry
  -- Prove hcf (gcd) is 37
  sorry

end hcf_of_two_numbers_l231_231229


namespace cost_of_one_lesson_l231_231211

-- Define the conditions
def total_cost_for_lessons : ℝ := 360
def total_hours_of_lessons : ℝ := 18
def duration_of_one_lesson : ℝ := 1.5

-- Define the theorem statement
theorem cost_of_one_lesson :
  (total_cost_for_lessons / total_hours_of_lessons) * duration_of_one_lesson = 30 := by
  -- Proof goes here
  sorry

end cost_of_one_lesson_l231_231211


namespace cubes_sum_formula_l231_231190

theorem cubes_sum_formula (a b : ℝ) (h1 : a + b = 7) (h2 : a * b = 5) : a^3 + b^3 = 238 := 
by 
  sorry

end cubes_sum_formula_l231_231190


namespace vessel_capacity_proof_l231_231835

variable (V1_capacity : ℕ) (V2_capacity : ℕ) (total_mixture : ℕ) (final_vessel_capacity : ℕ)
variable (A1_percentage : ℕ) (A2_percentage : ℕ)

theorem vessel_capacity_proof
  (h1 : V1_capacity = 2)
  (h2 : A1_percentage = 35)
  (h3 : V2_capacity = 6)
  (h4 : A2_percentage = 50)
  (h5 : total_mixture = 8)
  (h6 : final_vessel_capacity = 10)
  : final_vessel_capacity = 10 := 
by
  sorry

end vessel_capacity_proof_l231_231835


namespace toy_store_shelves_l231_231833

theorem toy_store_shelves (initial_bears : ℕ) (shipment_bears : ℕ) (bears_per_shelf : ℕ)
                          (h_initial : initial_bears = 5) (h_shipment : shipment_bears = 7) 
                          (h_per_shelf : bears_per_shelf = 6) : 
                          (initial_bears + shipment_bears) / bears_per_shelf = 2 :=
by
  sorry

end toy_store_shelves_l231_231833


namespace no_real_pairs_arithmetic_prog_l231_231848

theorem no_real_pairs_arithmetic_prog :
  ¬ ∃ a b : ℝ, (a = (1 / 2) * (8 + b)) ∧ (a + a * b = 2 * b) := by
sorry

end no_real_pairs_arithmetic_prog_l231_231848


namespace find_sin_2alpha_l231_231888

theorem find_sin_2alpha (α : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 4) Real.pi) 
  (h2 : 3 * Real.cos (2 * α) = 4 * Real.sin (Real.pi / 4 - α)) : 
  Real.sin (2 * α) = -1 / 9 :=
sorry

end find_sin_2alpha_l231_231888


namespace triangle_area_change_l231_231162

theorem triangle_area_change (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  let A_original := (B * H) / 2
  let H_new := H * 0.60
  let B_new := B * 1.40
  let A_new := (B_new * H_new) / 2
  (A_new = A_original * 0.84) :=
by
  sorry

end triangle_area_change_l231_231162


namespace interior_angle_regular_octagon_exterior_angle_regular_octagon_l231_231813

-- Definitions
def sumInteriorAngles (n : ℕ) : ℕ := 180 * (n - 2)
def oneInteriorAngle (n : ℕ) (sumInterior : ℕ) : ℕ := sumInterior / n
def sumExteriorAngles : ℕ := 360
def oneExteriorAngle (n : ℕ) (sumExterior : ℕ) : ℕ := sumExterior / n

-- Theorem statements
theorem interior_angle_regular_octagon : oneInteriorAngle 8 (sumInteriorAngles 8) = 135 := by sorry

theorem exterior_angle_regular_octagon : oneExteriorAngle 8 sumExteriorAngles = 45 := by sorry

end interior_angle_regular_octagon_exterior_angle_regular_octagon_l231_231813


namespace team_count_l231_231622

theorem team_count (girls boys : ℕ) (g : girls = 4) (b : boys = 6) :
  (Nat.choose 4 2) * (Nat.choose 6 2) = 90 := by
  sorry

end team_count_l231_231622


namespace distribution_of_items_l231_231414

open Finset

theorem distribution_of_items :
  ∃ (count : ℕ), count = 52 ∧
  ∃ (ways : Finset (Multiset (Multiset ℕ))), ways.card = 52 ∧
  (∀ (items : Finset ℕ) (bags : Finset (Finset (Finset ℕ))),
    items.card = 5 →
    bags.card = 4 →
    (∃ way : Multiset (Multiset ℕ), way ∈ ways) →
    bags ⊆ ways.toFinset) :=
begin
  sorry
end

end distribution_of_items_l231_231414


namespace Lisa_total_spoons_l231_231203

def total_spoons (children : ℕ) (spoons_per_child : ℕ) (decorative_spoons : ℕ) (large_spoons : ℕ) (teaspoons : ℕ) : ℕ := 
  (children * spoons_per_child) + decorative_spoons + (large_spoons + teaspoons)

theorem Lisa_total_spoons :
  (total_spoons 4 3 2 10 15) = 39 :=
by
  sorry

end Lisa_total_spoons_l231_231203


namespace sum_of_series_l231_231463

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l231_231463


namespace geometric_sequence_sum_l231_231843

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_pos : ∀ n, a n > 0)
  (h1 : a 1 + a 3 = 3)
  (h2 : a 4 + a 6 = 6):
  a 1 * a 3 + a 2 * a 4 + a 3 * a 5 + a 4 * a 6 + a 5 * a 7 = 62 :=
sorry

end geometric_sequence_sum_l231_231843


namespace ratio_of_work_done_by_women_to_men_l231_231674

theorem ratio_of_work_done_by_women_to_men 
  (total_work_men : ℕ := 15 * 21 * 8)
  (total_work_women : ℕ := 21 * 36 * 5) :
  (total_work_women : ℚ) / (total_work_men : ℚ) = 2 / 3 :=
by
  -- Proof goes here
  sorry

end ratio_of_work_done_by_women_to_men_l231_231674


namespace mark_notebooks_at_126_percent_l231_231253

variable (L : ℝ) (C : ℝ) (M : ℝ) (S : ℝ)

def merchant_condition1 := C = 0.85 * L
def merchant_condition2 := C = 0.75 * S
def merchant_condition3 := S = 0.9 * M

theorem mark_notebooks_at_126_percent :
    merchant_condition1 L C →
    merchant_condition2 C S →
    merchant_condition3 S M →
    M = 1.259 * L := by
  intros h1 h2 h3
  sorry

end mark_notebooks_at_126_percent_l231_231253


namespace cylindrical_to_rectangular_conversion_l231_231991

theorem cylindrical_to_rectangular_conversion 
  (r θ z : ℝ) 
  (h1 : r = 10) 
  (h2 : θ = Real.pi / 3) 
  (h3 : z = -2) :
  (r * Real.cos θ, r * Real.sin θ, z) = (5, 5 * Real.sqrt 3, -2) :=
by
  sorry

end cylindrical_to_rectangular_conversion_l231_231991


namespace expected_visible_people_l231_231825

noncomputable def E_X_n (n : ℕ) : ℝ :=
  match n with
  | 0       => 0   -- optional: edge case for n = 0 (0 people, 0 visible)
  | 1       => 1
  | (n + 1) => E_X_n n + 1 / (n + 1)

theorem expected_visible_people (n : ℕ) : E_X_n n = 1 + (∑ i in Finset.range n, 1 / (i + 1)) :=
by
  sorry

end expected_visible_people_l231_231825


namespace perimeter_of_quadrilateral_eq_fifty_l231_231806

theorem perimeter_of_quadrilateral_eq_fifty
  (a b : ℝ)
  (h1 : a = 10)
  (h2 : b = 15)
  (h3 : ∀ (p q r s : ℝ), p + q = r + s) : 
  2 * a + 2 * b = 50 := 
by
  sorry

end perimeter_of_quadrilateral_eq_fifty_l231_231806


namespace num_possible_radii_l231_231986

theorem num_possible_radii:
  ∃ (S : Finset ℕ), 
  (∀ r ∈ S, r < 60 ∧ (2 * r * π ∣ 120 * π)) ∧ 
  S.card = 11 := 
sorry

end num_possible_radii_l231_231986


namespace problem1_problem2_l231_231080

-- Define the sets of balls and boxes
inductive Ball
| ball1 | ball2 | ball3 | ball4

inductive Box
| boxA | boxB | boxC

-- Define the arrangements for the first problem
def arrangements_condition1 (arrangement : Ball → Box) : Prop :=
  (arrangement Ball.ball3 = Box.boxB) ∧
  (∃ b1 b2 b3 : Box, b1 ≠ b2 ∧ b2 ≠ b3 ∧ b3 ≠ b1 ∧ 
    ∃ (f : Ball → Box), 
      (f Ball.ball1 = b1) ∧ (f Ball.ball2 = b2) ∧ (f Ball.ball3 = Box.boxB) ∧ (f Ball.ball4 = b3))

-- Define the proof statement for the first problem
theorem problem1 : ∃ n : ℕ, (∀ arrangement : Ball → Box, arrangements_condition1 arrangement → n = 7) :=
sorry

-- Define the arrangements for the second problem
def arrangements_condition2 (arrangement : Ball → Box) : Prop :=
  (arrangement Ball.ball1 ≠ Box.boxA) ∧
  (arrangement Ball.ball2 ≠ Box.boxB)

-- Define the proof statement for the second problem
theorem problem2 : ∃ n : ℕ, (∀ arrangement : Ball → Box, arrangements_condition2 arrangement → n = 36) :=
sorry

end problem1_problem2_l231_231080


namespace Queen_High_School_teachers_needed_l231_231263

def students : ℕ := 1500
def classes_per_student : ℕ := 6
def students_per_class : ℕ := 25
def classes_per_teacher : ℕ := 5

theorem Queen_High_School_teachers_needed : 
  (students * classes_per_student) / students_per_class / classes_per_teacher = 72 :=
by 
  sorry

end Queen_High_School_teachers_needed_l231_231263


namespace equal_vectors_implies_collinear_l231_231704

-- Definitions for vectors and their properties
variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def collinear (u v : V) : Prop := ∃ (a : ℝ), v = a • u 

def equal_vectors (u v : V) : Prop := u = v

theorem equal_vectors_implies_collinear (u v : V)
  (h : equal_vectors u v) : collinear u v :=
by sorry

end equal_vectors_implies_collinear_l231_231704


namespace sarah_cupcakes_l231_231705

theorem sarah_cupcakes (c k d : ℕ) (h1 : c + k = 6) (h2 : 90 * c + 40 * k = 100 * d) : c = 4 ∨ c = 6 :=
by {
  sorry -- Proof is omitted as requested.
}

end sarah_cupcakes_l231_231705


namespace balance_balls_l231_231214

open Real

variables (G B Y W : ℝ)

-- Conditions
def condition1 := (4 * G = 8 * B)
def condition2 := (3 * Y = 6 * B)
def condition3 := (8 * B = 6 * W)

-- Theorem statement
theorem balance_balls 
  (h1 : condition1 G B) 
  (h2 : condition2 Y B) 
  (h3 : condition3 B W) :
  ∃ (B_needed : ℝ), B_needed = 5 * G + 3 * Y + 4 * W ∧ B_needed = 64 / 3 * B :=
sorry

end balance_balls_l231_231214


namespace checkerboard_probability_l231_231602

def checkerboard_size : ℕ := 10

def total_squares (n : ℕ) : ℕ := n * n

def perimeter_squares (n : ℕ) : ℕ := 4 * n - 4

def inner_squares (n : ℕ) : ℕ := total_squares n - perimeter_squares n

def probability_not_touching_edge (n : ℕ) : ℚ := inner_squares n / total_squares n

theorem checkerboard_probability :
  probability_not_touching_edge checkerboard_size = 16 / 25 := by
  sorry

end checkerboard_probability_l231_231602


namespace gary_profit_l231_231129

theorem gary_profit :
  let total_flour := 8 -- pounds
  let cost_flour := 4 -- dollars
  let large_cakes_flour := 5 -- pounds
  let small_cakes_flour := 3 -- pounds
  let flour_per_large_cake := 0.75 -- pounds per large cake
  let flour_per_small_cake := 0.25 -- pounds per small cake
  let cost_additional_large := 1.5 -- dollars per large cake
  let cost_additional_small := 0.75 -- dollars per small cake
  let cost_baking_equipment := 10 -- dollars
  let revenue_per_large := 6.5 -- dollars per large cake
  let revenue_per_small := 2.5 -- dollars per small cake
  let num_large_cakes := 6 -- (from calculation: ⌊5 / 0.75⌋)
  let num_small_cakes := 12 -- (from calculation: 3 / 0.25)
  let cost_additional_ingredients := num_large_cakes * cost_additional_large + num_small_cakes * cost_additional_small
  let total_revenue := num_large_cakes * revenue_per_large + num_small_cakes * revenue_per_small
  let total_cost := cost_flour + cost_baking_equipment + cost_additional_ingredients
  let profit := total_revenue - total_cost
  profit = 37 := by
  sorry

end gary_profit_l231_231129


namespace tan_2beta_l231_231547

theorem tan_2beta {α β : ℝ} 
  (h₁ : Real.tan (α + β) = 2) 
  (h₂ : Real.tan (α - β) = 3) : 
  Real.tan (2 * β) = -1 / 7 :=
by 
  sorry

end tan_2beta_l231_231547


namespace lars_bakes_for_six_hours_l231_231910

variable (h : ℕ)

-- Conditions
def bakes_loaves : ℕ := 10 * h
def bakes_baguettes : ℕ := 15 * h
def total_breads : ℕ := bakes_loaves h + bakes_baguettes h

-- Proof goal
theorem lars_bakes_for_six_hours (h : ℕ) (H : total_breads h = 150) : h = 6 :=
sorry

end lars_bakes_for_six_hours_l231_231910


namespace lisa_need_add_pure_juice_l231_231207

theorem lisa_need_add_pure_juice
  (x : ℝ) 
  (total_volume : ℝ := 2)
  (initial_pure_juice_fraction : ℝ := 0.10)
  (desired_pure_juice_fraction : ℝ := 0.25) 
  (added_pure_juice : ℝ := x) 
  (initial_pure_juice_amount : ℝ := total_volume * initial_pure_juice_fraction)
  (final_pure_juice_amount : ℝ := initial_pure_juice_amount + added_pure_juice)
  (final_volume : ℝ := total_volume + added_pure_juice) :
  (final_pure_juice_amount / final_volume) = desired_pure_juice_fraction → x = 0.4 :=
by
  intro h
  sorry

end lisa_need_add_pure_juice_l231_231207


namespace intersection_complement_l231_231355

open Set

-- Defining sets A, B and universal set U
def A : Set ℕ := {1, 2, 3, 5, 7}
def B : Set ℕ := {x | 1 < x ∧ x ≤ 6}
def U : Set ℕ := A ∪ B

-- Statement of the proof problem
theorem intersection_complement :
  A ∩ (U \ B) = {1, 7} :=
by
  sorry

end intersection_complement_l231_231355


namespace tournament_byes_and_games_l231_231751

/-- In a single-elimination tournament with 300 players initially registered,
- if the number of players in each subsequent round must be a power of 2,
- then 44 players must receive a bye in the first round, and 255 total games
- must be played to determine the champion. -/
theorem tournament_byes_and_games :
  let initial_players := 300
  let pow2_players := 256
  44 = initial_players - pow2_players ∧
  255 = pow2_players - 1 :=
by
  let initial_players := 300
  let pow2_players := 256
  have h_byes : 44 = initial_players - pow2_players := by sorry
  have h_games : 255 = pow2_players - 1 := by sorry
  exact ⟨h_byes, h_games⟩

end tournament_byes_and_games_l231_231751


namespace smallest_positive_integer_n_l231_231849

theorem smallest_positive_integer_n (n : ℕ) (h : n > 0) : 3^n ≡ n^3 [MOD 5] ↔ n = 3 :=
sorry

end smallest_positive_integer_n_l231_231849


namespace remainder_product_mod_eq_l231_231738

theorem remainder_product_mod_eq (n : ℤ) :
  ((12 - 2 * n) * (n + 5)) % 11 = (-2 * n^2 + 2 * n + 5) % 11 := by
  sorry

end remainder_product_mod_eq_l231_231738


namespace retirement_percentage_l231_231334

-- Define the conditions
def gross_pay : ℝ := 1120
def tax_deduction : ℝ := 100
def net_paycheck : ℝ := 740

-- Define the total deduction
def total_deduction : ℝ := gross_pay - net_paycheck
def retirement_deduction : ℝ := total_deduction - tax_deduction

-- Define the theorem to prove
theorem retirement_percentage :
  (retirement_deduction / gross_pay) * 100 = 25 :=
by
  sorry

end retirement_percentage_l231_231334


namespace simultaneous_equations_solution_l231_231990

theorem simultaneous_equations_solution :
  ∃ x y : ℚ, (3 * x + 4 * y = 20) ∧ (9 * x - 8 * y = 36) ∧ (x = 76 / 15) ∧ (y = 18 / 15) :=
by
  sorry

end simultaneous_equations_solution_l231_231990


namespace find_m_l231_231121

theorem find_m (m : ℕ) (h : m * (Nat.factorial m) + 2 * (Nat.factorial m) = 5040) : m = 5 :=
by
  sorry

end find_m_l231_231121


namespace probability_first_queen_second_diamond_l231_231062

/-- 
Given that two cards are dealt at random from a standard deck of 52 cards,
the probability that the first card is a Queen and the second card is a Diamonds suit is 1/52.
-/
theorem probability_first_queen_second_diamond : 
  let total_cards := 52
  let probability (num events : ℕ) := (events : ℝ) / (num + 0 : ℝ)
  let prob_first_queen := probability total_cards 4
  let prob_second_diamond_if_first_queen_diamond := probability (total_cards - 1) 12
  let prob_second_diamond_if_first_other_queen := probability (total_cards - 1) 13
  let prob_first_queen_diamond := probability total_cards 1
  let prob_first_queen_other := probability total_cards 3
  let joint_prob_first_queen_diamond := prob_first_queen_diamond * prob_second_diamond_if_first_queen_diamond
  let joint_prob_first_queen_other := prob_first_queen_other * prob_second_diamond_if_first_other_queen
  let total_probability := joint_prob_first_queen_diamond + joint_prob_first_queen_other
  total_probability = probability total_cards 1 := 
by
  -- Proof goes here
  sorry

end probability_first_queen_second_diamond_l231_231062


namespace f_odd_f_periodic_f_def_on_interval_problem_solution_l231_231104

noncomputable def f : ℝ → ℝ := 
sorry

theorem f_odd (x : ℝ) : f (-x) = -f x := 
sorry

theorem f_periodic (x : ℝ) : f (x + 4) = f x := 
sorry

theorem f_def_on_interval (x : ℝ) (h : -2 < x ∧ x < 0) : f x = 2 ^ x :=
sorry

theorem problem_solution : f 2015 - f 2014 = 1 / 2 :=
sorry

end f_odd_f_periodic_f_def_on_interval_problem_solution_l231_231104


namespace cos_minus_sin_eq_neg_one_fifth_l231_231551

theorem cos_minus_sin_eq_neg_one_fifth
  (α : ℝ)
  (h1 : Real.sin (2 * α) = 24 / 25)
  (h2 : π < α ∧ α < 5 * π / 4) :
  Real.cos α - Real.sin α = -1 / 5 := sorry

end cos_minus_sin_eq_neg_one_fifth_l231_231551


namespace fraction_product_l231_231421

theorem fraction_product : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end fraction_product_l231_231421


namespace evaluate_series_sum_l231_231436

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l231_231436


namespace solution_set_l231_231846

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f x = f (-x)
axiom monotone_decreasing_f : ∀ {a b : ℝ}, 0 ≤ a → a ≤ b → f b ≤ f a
axiom f_half_eq_zero : f (1 / 2) = 0

theorem solution_set :
  { x : ℝ | f (Real.log x / Real.log (1 / 4)) < 0 } = 
  { x : ℝ | 0 < x ∧ x < 1 / 2 } ∪ { x : ℝ | 2 < x } :=
by
  sorry

end solution_set_l231_231846


namespace intersection_of_sets_l231_231878
-- Define the sets and the proof statement
theorem intersection_of_sets : 
  let A := { x : ℝ | x^2 - 3 * x - 4 < 0 }
  let B := {-4, 1, 3, 5}
  A ∩ B = {1, 3} :=
by
  sorry

end intersection_of_sets_l231_231878


namespace sum_series_eq_4_div_9_l231_231455

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l231_231455


namespace expected_visible_eq_sum_l231_231823

noncomputable def expected_visible (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, 1/(k+1 : ℚ)) + 1

theorem expected_visible_eq_sum (n : ℕ) :
  expected_visible n = (Finset.range n).sum (λ k, 1/(k+1 : ℚ)) + 1 :=
by
  sorry

end expected_visible_eq_sum_l231_231823


namespace relationship_above_l231_231350

noncomputable def a : ℝ := Real.log 5 / Real.log 2
noncomputable def b : ℝ := Real.log 15 / (2 * Real.log 2)
noncomputable def c : ℝ := Real.sqrt 2

theorem relationship_above (ha : a = Real.log 5 / Real.log 2) 
                           (hb : b = Real.log 15 / (2 * Real.log 2))
                           (hc : c = Real.sqrt 2) : a > b ∧ b > c :=
by
  sorry

end relationship_above_l231_231350


namespace min_students_l231_231325

variable (L : ℕ) (H : ℕ) (M : ℕ) (e : ℕ)

def find_min_students : Prop :=
  H = 2 * L ∧ 
  M = L + H ∧ 
  e = L + M + H ∧ 
  e = 6 * L ∧ 
  L ≥ 1

theorem min_students (L : ℕ) (H : ℕ) (M : ℕ) (e : ℕ) : find_min_students L H M e → e = 6 := 
by 
  intro h 
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  sorry

end min_students_l231_231325


namespace soccer_tournament_probability_l231_231216

noncomputable def prob_teamA_more_points : ℚ :=
  (163 : ℚ) / 256

theorem soccer_tournament_probability :
  m + n = 419 ∧ prob_teamA_more_points = 163 / 256 := sorry

end soccer_tournament_probability_l231_231216


namespace shaded_region_area_correct_l231_231400

noncomputable def hexagon_side : ℝ := 4
noncomputable def major_axis : ℝ := 4
noncomputable def minor_axis : ℝ := 2

noncomputable def hexagon_area := (3 * Real.sqrt 3 / 2) * hexagon_side^2

noncomputable def semi_ellipse_area : ℝ :=
  (1 / 2) * Real.pi * major_axis * minor_axis

noncomputable def total_semi_ellipse_area := 4 * semi_ellipse_area 

noncomputable def shaded_region_area := hexagon_area - total_semi_ellipse_area

theorem shaded_region_area_correct : shaded_region_area = 48 * Real.sqrt 3 - 16 * Real.pi :=
by
  sorry

end shaded_region_area_correct_l231_231400


namespace fractional_product_l231_231424

theorem fractional_product :
  ((3/4) * (4/5) * (5/6) * (6/7) * (7/8)) = 3/8 :=
by
  sorry

end fractional_product_l231_231424


namespace infinite_series_sum_l231_231519

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l231_231519


namespace sum_series_equals_4_div_9_l231_231480

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l231_231480


namespace temperature_max_time_l231_231362

theorem temperature_max_time (t : ℝ) (h : 0 ≤ t) : 
  (-t^2 + 10 * t + 60 = 85) → t = 15 := 
sorry

end temperature_max_time_l231_231362


namespace inequality_proof_l231_231189

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (b * c) + b / (a * c) + c / (a * b) ≥ 2 / a + 2 / b - 2 / c := 
  sorry

end inequality_proof_l231_231189


namespace number_of_valid_pairs_l231_231988

theorem number_of_valid_pairs :
  (∃ (count : ℕ), count = 280 ∧
    (∃ (m n : ℕ),
      1 ≤ m ∧ m ≤ 2899 ∧
      5^n < 2^m ∧ 2^m < 2^(m+3) ∧ 2^(m+3) < 5^(n+1))) :=
sorry

end number_of_valid_pairs_l231_231988


namespace lisa_total_spoons_l231_231200

def number_of_baby_spoons (num_children num_spoons_per_child : Nat) : Nat :=
  num_children * num_spoons_per_child

def number_of_decorative_spoons : Nat := 2

def number_of_old_spoons (baby_spoons decorative_spoons : Nat) : Nat :=
  baby_spoons + decorative_spoons
  
def number_of_new_spoons (large_spoons teaspoons : Nat) : Nat :=
  large_spoons + teaspoons

def total_number_of_spoons (old_spoons new_spoons : Nat) : Nat :=
  old_spoons + new_spoons

theorem lisa_total_spoons
  (children : Nat)
  (spoons_per_child : Nat)
  (large_spoons : Nat)
  (teaspoons : Nat)
  (children_eq : children = 4)
  (spoons_per_child_eq : spoons_per_child = 3)
  (large_spoons_eq : large_spoons = 10)
  (teaspoons_eq : teaspoons = 15)
  : total_number_of_spoons (number_of_old_spoons (number_of_baby_spoons children spoons_per_child) number_of_decorative_spoons) (number_of_new_spoons large_spoons teaspoons) = 39 :=
by
  sorry

end lisa_total_spoons_l231_231200


namespace gcd_A_B_l231_231352

def A : ℤ := 1989^1990 - 1988^1990
def B : ℤ := 1989^1989 - 1988^1989

theorem gcd_A_B : Int.gcd A B = 1 := 
by
  -- Conditions
  have h1 : A = 1989^1990 - 1988^1990 := rfl
  have h2 : B = 1989^1989 - 1988^1989 := rfl
  -- Conclusion
  sorry

end gcd_A_B_l231_231352


namespace gcd_of_1230_and_990_l231_231534

theorem gcd_of_1230_and_990 : Nat.gcd 1230 990 = 30 :=
by
  sorry

end gcd_of_1230_and_990_l231_231534


namespace probability_of_three_cards_l231_231382

-- Conditions
def deck_size : ℕ := 52
def spades : ℕ := 13
def spades_face_cards : ℕ := 3
def face_cards : ℕ := 12
def diamonds : ℕ := 13

-- Probability of drawing specific cards
def prob_first_spade_non_face : ℚ := 10 / 52
def prob_second_face_given_first_spade_non_face : ℚ := 12 / 51
def prob_third_diamond_given_first_two : ℚ := 13 / 50

def prob_first_spade_face : ℚ := 3 / 52
def prob_second_face_given_first_spade_face : ℚ := 9 / 51

-- Final probability
def final_probability := 
  (prob_first_spade_non_face * prob_second_face_given_first_spade_non_face * prob_third_diamond_given_first_two) +
  (prob_first_spade_face * prob_second_face_given_first_spade_face * prob_third_diamond_given_first_two)

theorem probability_of_three_cards :
  final_probability = 1911 / 132600 := 
by
  sorry

end probability_of_three_cards_l231_231382


namespace find_pairs_of_square_numbers_l231_231861

theorem find_pairs_of_square_numbers (a b k : ℕ) (hk : k ≥ 2) 
  (h_eq : (a * a + b * b) = k * k * (a * b + 1)) : 
  (a = k ∧ b = k * k * k) ∨ (b = k ∧ a = k * k * k) :=
by
  sorry

end find_pairs_of_square_numbers_l231_231861


namespace number_of_true_propositions_is_one_l231_231976

theorem number_of_true_propositions_is_one :
  (¬ ∀ x : ℝ, x^4 > x^2) ∧
  (¬ (∀ (p q : Prop), ¬ (p ∧ q) → (¬ p ∧ ¬ q))) ∧
  (¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0) →
  1 = 1 :=
by
  sorry

end number_of_true_propositions_is_one_l231_231976


namespace smallest_integer_cubing_y_eq_350_l231_231099

def y : ℕ := 2^3 * 3^5 * 4^5 * 5^4 * 6^3 * 7^5 * 8^2

theorem smallest_integer_cubing_y_eq_350 : ∃ z : ℕ, z * y = (2^23) * (3^9) * (5^6) * (7^6) → z = 350 :=
by
  sorry

end smallest_integer_cubing_y_eq_350_l231_231099


namespace prob_B_second_shot_prob_A_ith_shot_expected_A_shots_l231_231776

-- Define probabilities and parameters
def pA : ℝ := 0.6
def pB : ℝ := 0.8

-- Define the probability of selecting the first shooter as 0.5 for each player
def first_shot_prob : ℝ := 0.5

-- Proof that the probability that player B takes the second shot is 0.6
theorem prob_B_second_shot : (first_shot_prob * (1 - pA) + first_shot_prob * pB) = 0.6 := 
by sorry

-- Define the recursive probability for player A taking the nth shot
noncomputable def P (n : ℕ) : ℝ :=
if n = 0 then 0.5
else 0.4 * (P (n - 1)) + 0.2

-- Proof that the probability that player A takes the i-th shot is given by the formula
theorem prob_A_ith_shot (i : ℕ) : P i = (1 / 3) + (1 / 6) * ((2 / 5) ^ (i - 1)) :=
by sorry

-- Define the expected number of times player A shoots in the first n shots based on provided P formula
noncomputable def E_Y (n : ℕ) : ℝ :=
(sum i in finset.range n, P i)

-- Proof that the expected number of times player A shoots in the first n shots is given by the formula
theorem expected_A_shots (n : ℕ) : E_Y n = (5 / 18) * (1 - (2 / 5) ^ n) + (n / 3) :=
by sorry

end prob_B_second_shot_prob_A_ith_shot_expected_A_shots_l231_231776


namespace probability_red_purple_not_same_bed_l231_231173

def colors : Set String := {"red", "yellow", "white", "purple"}

def C (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_red_purple_not_same_bed : 
  let total_ways := C 4 2
  let unwanted_ways := 2
  let desired_ways := total_ways - unwanted_ways
  let probability := (desired_ways : ℚ) / total_ways
  probability = 2 / 3 := by
  sorry

end probability_red_purple_not_same_bed_l231_231173


namespace fraction_equality_l231_231185

theorem fraction_equality (a b : ℝ) (h : a ≠ b) (h_eq : a / b + (a + 5 * b) / (b + 5 * a) = 2) : a / b = 3 / 5 :=
by
  sorry

end fraction_equality_l231_231185


namespace cookies_taken_in_four_days_l231_231802

def initial_cookies : ℕ := 70
def cookies_left : ℕ := 28
def days_in_week : ℕ := 7
def days_taken : ℕ := 4
def daily_cookies_taken (total_cookies_taken : ℕ) : ℕ := total_cookies_taken / days_in_week
def total_cookies_taken : ℕ := initial_cookies - cookies_left

theorem cookies_taken_in_four_days :
  daily_cookies_taken total_cookies_taken * days_taken = 24 := by
  sorry

end cookies_taken_in_four_days_l231_231802


namespace part_i_part_ii_l231_231150

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - 2 * a) + abs (x - a)

-- Part I: Prove solution to the inequality.
theorem part_i (x : ℝ) : f x 1 > 3 ↔ x ∈ {x | x < 0} ∪ {x | x > 3} :=
sorry

-- Part II: Prove the inequality for general a and b with condition for equality.
theorem part_ii (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  f b a ≥ f a a ∧ ((2 * a - b = 0 ∨ b - a = 0) ∨ (2 * a - b > 0 ∧ b - a > 0) ∨ (2 * a - b < 0 ∧ b - a < 0)) ↔ f b a = f a a :=
sorry

end part_i_part_ii_l231_231150


namespace appropriate_speech_length_l231_231340

def speech_length_min := 20
def speech_length_max := 40
def speech_rate := 120

theorem appropriate_speech_length 
  (min_words := speech_length_min * speech_rate) 
  (max_words := speech_length_max * speech_rate) : 
  ∀ n : ℕ, n >= min_words ∧ n <= max_words ↔ (n = 2500 ∨ n = 3800 ∨ n = 4600) := 
by 
  sorry

end appropriate_speech_length_l231_231340


namespace ratio_of_numbers_l231_231318

theorem ratio_of_numbers (a b : ℝ) (h1 : 0 < b) (h2 : b < a) 
  (h3 : 2 * ((a + b) / 2) = Real.sqrt (10 * a * b)) : abs (a / b - 8) < 1 :=
by
  sorry

end ratio_of_numbers_l231_231318


namespace eggs_per_hen_per_day_l231_231926

theorem eggs_per_hen_per_day
  (hens : ℕ) (days : ℕ) (neighborTaken : ℕ) (dropped : ℕ) (finalEggs : ℕ) (E : ℕ) 
  (h1 : hens = 3) 
  (h2 : days = 7) 
  (h3 : neighborTaken = 12) 
  (h4 : dropped = 5) 
  (h5 : finalEggs = 46) 
  (totalEggs : ℕ := hens * E * days) 
  (afterNeighbor : ℕ := totalEggs - neighborTaken) 
  (beforeDropping : ℕ := finalEggs + dropped) : 
  totalEggs = beforeDropping + neighborTaken → E = 3 := sorry

end eggs_per_hen_per_day_l231_231926


namespace remaining_oil_quantity_check_remaining_oil_quantity_l231_231167

def initial_oil_quantity : Real := 40
def outflow_rate : Real := 0.2

theorem remaining_oil_quantity (t : Real) : Real :=
  initial_oil_quantity - outflow_rate * t

theorem check_remaining_oil_quantity (t : Real) : remaining_oil_quantity t = 40 - 0.2 * t := 
by 
  sorry

end remaining_oil_quantity_check_remaining_oil_quantity_l231_231167


namespace find_sample_size_l231_231085

def ratio_A : ℚ := 2
def ratio_B : ℚ := 3
def ratio_C : ℚ := 5
def total_ratio : ℚ := ratio_A + ratio_B + ratio_C
def proportion_B : ℚ := ratio_B / total_ratio
def units_B_sampled : ℚ := 24

theorem find_sample_size : ∃ n : ℚ, proportion_B = units_B_sampled / n ∧ n = 80 :=
by
  sorry

end find_sample_size_l231_231085


namespace fraction_problem_l231_231737

theorem fraction_problem (m n p q : ℚ) 
  (h1 : m / n = 20) 
  (h2 : p / n = 5) 
  (h3 : p / q = 1 / 15) : 
  m / q = 4 / 15 :=
sorry

end fraction_problem_l231_231737


namespace sum_k_over_4k_l231_231445

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l231_231445


namespace P_plus_Q_is_expected_l231_231560

-- defining the set P
def P : Set ℝ := { x | x ^ 2 - 3 * x - 4 ≤ 0 }

-- defining the set Q
def Q : Set ℝ := { x | x ^ 2 - 2 * x - 15 > 0 }

-- defining the set P + Q
def P_plus_Q : Set ℝ := { x | (x ∈ P ∨ x ∈ Q) ∧ ¬(x ∈ P ∧ x ∈ Q) }

-- the expected result
def expected_P_plus_Q : Set ℝ := { x | x < -3 } ∪ { x | -1 ≤ x ∧ x ≤ 4 } ∪ { x | x > 5 }

-- theorem stating that P + Q equals the expected result
theorem P_plus_Q_is_expected : P_plus_Q = expected_P_plus_Q := by
  sorry

end P_plus_Q_is_expected_l231_231560


namespace shaded_area_l231_231698

-- Defining the conditions
def small_square_side := 4
def large_square_side := 12
def half_large_square_side := large_square_side / 2

-- DG is calculated as (12 / 16) * small_square_side = 3
def DG := (large_square_side / (half_large_square_side + small_square_side)) * small_square_side

-- Calculating area of triangle DGF
def area_triangle_DGF := (DG * small_square_side) / 2

-- Area of the smaller square
def area_small_square := small_square_side * small_square_side

-- Area of the shaded region
def area_shaded_region := area_small_square - area_triangle_DGF

-- The theorem stating the question
theorem shaded_area : area_shaded_region = 10 := by
  sorry

end shaded_area_l231_231698


namespace no_int_coeffs_l231_231030

def P (a b c d : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem no_int_coeffs (a b c d : ℤ) : 
  ¬ (P a b c d 19 = 1 ∧ P a b c d 62 = 2) :=
by sorry

end no_int_coeffs_l231_231030


namespace sin_double_angle_l231_231570

open Real

theorem sin_double_angle (θ : ℝ) (h : cos (π / 4 - θ) = 1 / 2) : sin (2 * θ) = -1 / 2 := 
by 
  sorry

end sin_double_angle_l231_231570


namespace total_pages_l231_231759

def Johnny_word_count : ℕ := 195
def Madeline_word_count : ℕ := 2 * Johnny_word_count
def Timothy_word_count : ℕ := Madeline_word_count + 50
def Samantha_word_count : ℕ := 3 * Madeline_word_count
def Ryan_word_count : ℕ := Johnny_word_count + 100
def Words_per_page : ℕ := 235

def pages_needed (words : ℕ) : ℕ :=
  if words % Words_per_page = 0 then words / Words_per_page else words / Words_per_page + 1

theorem total_pages :
  pages_needed Johnny_word_count +
  pages_needed Madeline_word_count +
  pages_needed Timothy_word_count +
  pages_needed Samantha_word_count +
  pages_needed Ryan_word_count = 12 :=
  by sorry

end total_pages_l231_231759


namespace remainder_of_sum_of_binomials_l231_231624

open Nat

theorem remainder_of_sum_of_binomials (hprime : Prime 2023) :
    (∑ k in Finset.range 65, binomial 2020 k) % 2023 = 1089 := sorry

end remainder_of_sum_of_binomials_l231_231624


namespace general_term_formula_l231_231869

theorem general_term_formula (a : ℕ → ℝ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → (n+1) * a (n+1) - n * a n^2 + (n+1) * a n * a (n+1) - n * a n = 0) :
  ∀ n : ℕ, 0 < n → a n = 1 / n :=
by
  sorry

end general_term_formula_l231_231869


namespace sum_k_over_4k_l231_231446

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l231_231446


namespace camping_trip_percentage_l231_231076

theorem camping_trip_percentage (t : ℕ) (h1 : 22 / 100 * t > 0) (h2 : 75 / 100 * (22 / 100 * t) ≤ t) :
  (88 / 100 * t) = t :=
by
  sorry

end camping_trip_percentage_l231_231076


namespace smallest_positive_integer_cube_ends_in_632_l231_231127

theorem smallest_positive_integer_cube_ends_in_632 :
  ∃ n : ℕ, (n > 0) ∧ (n^3 % 1000 = 632) ∧ ∀ m : ℕ, (m > 0) ∧ (m^3 % 1000 = 632) → n ≤ m := 
sorry

end smallest_positive_integer_cube_ends_in_632_l231_231127


namespace g_50_l231_231621

noncomputable def g : ℝ → ℝ :=
sorry

axiom functional_equation (x y : ℝ) : g (x * y) = x * g y
axiom g_2 : g 2 = 10

theorem g_50 : g 50 = 250 :=
sorry

end g_50_l231_231621


namespace remainder_of_5_pow_2023_mod_17_l231_231540

theorem remainder_of_5_pow_2023_mod_17 :
  5^2023 % 17 = 11 :=
by
  have h1 : 5^2 % 17 = 8 := by sorry
  have h2 : 5^4 % 17 = 13 := by sorry
  have h3 : 5^8 % 17 = -1 := by sorry
  have h4 : 5^16 % 17 = 1 := by sorry
  have h5 : 2023 = 16 * 126 + 7 := by sorry
  sorry

end remainder_of_5_pow_2023_mod_17_l231_231540


namespace boys_in_parkway_l231_231328

theorem boys_in_parkway (total_students : ℕ) (students_playing_soccer : ℕ) (percentage_boys_playing_soccer : ℝ)
                        (girls_not_playing_soccer : ℕ) :
                        total_students = 420 ∧ students_playing_soccer = 250 ∧ percentage_boys_playing_soccer = 0.86 
                        ∧ girls_not_playing_soccer = 73 → 
                        ∃ total_boys : ℕ, total_boys = 312 :=
by
  -- Proof omitted
  sorry

end boys_in_parkway_l231_231328


namespace monthly_income_calculation_l231_231822

variable (deposit : ℝ)
variable (percentage : ℝ)
variable (monthly_income : ℝ)

theorem monthly_income_calculation 
    (h1 : deposit = 3800) 
    (h2 : percentage = 0.32) 
    (h3 : deposit = percentage * monthly_income) : 
    monthly_income = 11875 :=
by
  sorry

end monthly_income_calculation_l231_231822


namespace intersection_of_A_and_B_l231_231871

def setA : Set ℝ := { x | x^2 - 3 * x - 4 < 0 }
def setB : Set ℝ := {-4, 1, 3, 5}
def resultSet : Set ℝ := {1, 3}

theorem intersection_of_A_and_B :
  setA ∩ setB = resultSet := 
by
  sorry

end intersection_of_A_and_B_l231_231871


namespace sum_factors_36_l231_231652

theorem sum_factors_36 : ∑ x in {1, 2, 3, 4, 6, 9, 12, 18, 36}, x = 91 := by
  sorry

end sum_factors_36_l231_231652


namespace find_k_l231_231152

open Real

def vector := ℝ × ℝ

def dot_product (v1 v2 : vector) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def orthogonal (v1 v2 : vector) : Prop := dot_product v1 v2 = 0

theorem find_k (k : ℝ) :
  let a : vector := (2, 3)
  let b : vector := (1, 4)
  let c : vector := (k, 3)
  orthogonal (a.1 + b.1, a.2 + b.2) c → k = -7 :=
by
  intros
  sorry

end find_k_l231_231152


namespace sequence_property_l231_231139

noncomputable def U : ℕ → ℕ
| 0       => 0  -- This definition is added to ensure U 1 corresponds to U_1 = 1
| (n + 1) => U n + (n + 1)

theorem sequence_property (n : ℕ) : U n + U (n + 1) = (n + 1) * (n + 1) :=
  sorry

end sequence_property_l231_231139


namespace sum_of_series_l231_231466

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l231_231466


namespace three_digit_integers_211_421_l231_231066

def is_one_more_than_multiple_of (n k : ℕ) : Prop :=
  ∃ m : ℕ, n = m * k + 1

theorem three_digit_integers_211_421
  (n : ℕ) (h1 : (100 ≤ n) ∧ (n ≤ 999))
  (h2 : is_one_more_than_multiple_of n 2)
  (h3 : is_one_more_than_multiple_of n 3)
  (h4 : is_one_more_than_multiple_of n 5)
  (h5 : is_one_more_than_multiple_of n 7) :
  n = 211 ∨ n = 421 :=
sorry

end three_digit_integers_211_421_l231_231066


namespace hyperbola_eccentricity_l231_231556

variable (x y a b c : ℝ)

-- Conditions given in the problem
def hyperbola (x y a b : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ ((x^2 / a^2) - (y^2 / b^2) = 1)

def asymptote_through_point (a b : ℝ) (point_x point_y : ℝ) : Prop :=
  (point_y / point_x = b / a)

def eccentricity (a b : ℝ) : ℝ :=
  sqrt (1 + (b^2 / a^2))

-- Given that the point (sqrt(2), sqrt(6)) is on asymptote
def point := (sqrt 2, sqrt 6)

-- The main statement to prove the eccentricity is 2
theorem hyperbola_eccentricity :
  ∀ a b : ℝ, hyperbola x y a b ∧ asymptote_through_point a b (fst point) (snd point) → 
  eccentricity a b = 2 :=
by
  sorry

end hyperbola_eccentricity_l231_231556


namespace volume_of_larger_cube_is_343_l231_231091

-- We will define the conditions first
def smaller_cube_side_length : ℤ := 1
def number_of_smaller_cubes : ℤ := 343
def volume_small_cube (l : ℤ) : ℤ := l^3
def diff_surface_area (l L : ℤ) : ℤ := (number_of_smaller_cubes * 6 * l^2) - (6 * L^2)

-- Main statement to prove the volume of the larger cube
theorem volume_of_larger_cube_is_343 :
  ∃ L, volume_small_cube smaller_cube_side_length * number_of_smaller_cubes = L^3 ∧
        diff_surface_area smaller_cube_side_length L = 1764 ∧
        volume_small_cube L = 343 :=
by
  sorry

end volume_of_larger_cube_is_343_l231_231091


namespace intersection_A_B_l231_231881

open Set

variable (x : ℝ)

def A : Set ℝ := {x | x^2 - 3 * x - 4 < 0}
def B : Set ℝ := {-4, 1, 3, 5}

theorem intersection_A_B :
  A ∩ B = { 1, 3 } :=
sorry

end intersection_A_B_l231_231881


namespace shaded_area_is_correct_l231_231972

noncomputable def square_shaded_area (side : ℝ) (beta : ℝ) (cos_beta : ℝ) : ℝ :=
  if (0 < beta) ∧ (beta < 90) ∧ (cos_beta = 3 / 5) ∧ (side = 2) then 3 / 10 
  else 0

theorem shaded_area_is_correct :
  square_shaded_area 2 beta (3 / 5) = 3 / 10 :=
by
  sorry

end shaded_area_is_correct_l231_231972


namespace distribute_items_5_in_4_identical_bags_l231_231412

theorem distribute_items_5_in_4_identical_bags : 
  let items := 5 in 
  let bags := 4 in 
  number_of_ways_to_distribute items bags = 36 := 
by sorry

end distribute_items_5_in_4_identical_bags_l231_231412


namespace quadratic_inequality_m_range_l231_231557

theorem quadratic_inequality_m_range (m : ℝ) : (∀ x : ℝ, m * x^2 + 2 * m * x - 8 ≥ 0) ↔ (m ≠ 0) :=
by
  sorry

end quadratic_inequality_m_range_l231_231557


namespace perpendicular_dot_product_zero_l231_231895

variables (a : ℝ)
def m := (a, 2)
def n := (1, 1 - a)

theorem perpendicular_dot_product_zero : (m a).1 * (n a).1 + (m a).2 * (n a).2 = 0 → a = 2 :=
by sorry

end perpendicular_dot_product_zero_l231_231895


namespace cookies_with_five_cups_l231_231342

-- Define the initial condition: Lee can make 24 cookies with 3 cups of flour
def cookies_per_cup := 24 / 3

-- Theorem stating Lee can make 40 cookies with 5 cups of flour
theorem cookies_with_five_cups : 5 * cookies_per_cup = 40 :=
by
  sorry

end cookies_with_five_cups_l231_231342


namespace remainder_1425_1427_1429_mod_12_l231_231815

theorem remainder_1425_1427_1429_mod_12 :
  (1425 * 1427 * 1429) % 12 = 11 :=
by
  sorry

end remainder_1425_1427_1429_mod_12_l231_231815


namespace theater_rows_25_l231_231683

theorem theater_rows_25 (n : ℕ) (x : ℕ) (k : ℕ) (h : n = 1000) (h1 : k > 16) (h2 : (2 * x + k) * (k + 1) = 2000) : (k + 1) = 25 :=
by
  -- The proof goes here, which we omit for the problem statement.
  sorry

end theater_rows_25_l231_231683


namespace union_of_A_and_B_l231_231298

def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {1, 2}

theorem union_of_A_and_B : A ∪ B = {x | x ≤ 2} := sorry

end union_of_A_and_B_l231_231298


namespace parabola_directrix_eq_l231_231944

theorem parabola_directrix_eq (p : ℝ) (h : y^2 = 2 * x ∧ p = 1) : x = -p / 2 := by
  sorry

end parabola_directrix_eq_l231_231944


namespace Lisa_total_spoons_l231_231202

def total_spoons (children : ℕ) (spoons_per_child : ℕ) (decorative_spoons : ℕ) (large_spoons : ℕ) (teaspoons : ℕ) : ℕ := 
  (children * spoons_per_child) + decorative_spoons + (large_spoons + teaspoons)

theorem Lisa_total_spoons :
  (total_spoons 4 3 2 10 15) = 39 :=
by
  sorry

end Lisa_total_spoons_l231_231202


namespace ella_stamps_value_l231_231116

theorem ella_stamps_value :
  let total_stamps := 18
  let value_of_6_stamps := 18
  let consistent_value_per_stamp := value_of_6_stamps / 6
  total_stamps * consistent_value_per_stamp = 54 := by
  sorry

end ella_stamps_value_l231_231116


namespace solve_inequality_l231_231610

theorem solve_inequality (a : ℝ) : (6 * x^2 + a * x - a^2 < 0) ↔
  ((a > 0) ∧ (-a / 2 < x ∧ x < a / 3)) ∨
  ((a < 0) ∧ (a / 3 < x ∧ x < -a / 2)) ∨
  ((a = 0) ∧ false) :=
by 
  sorry

end solve_inequality_l231_231610


namespace sin_double_theta_l231_231569

-- Given condition
def given_condition (θ : ℝ) : Prop :=
  Real.cos (Real.pi / 4 - θ) = 1 / 2

-- The statement we want to prove: sin(2θ) = -1/2
theorem sin_double_theta (θ : ℝ) (h : given_condition θ) : Real.sin (2 * θ) = -1 / 2 :=
sorry

end sin_double_theta_l231_231569


namespace percentage_increase_l231_231433

variables (A B C D E : ℝ)
variables (A_inc B_inc C_inc D_inc E_inc : ℝ)

-- Conditions
def conditions (A_inc B_inc C_inc D_inc E_inc : ℝ) :=
  A_inc = 0.1 * A ∧
  B_inc = (1/15) * B ∧
  C_inc = 0.05 * C ∧
  D_inc = 0.04 * D ∧
  E_inc = (1/30) * E ∧
  B = 1.5 * A ∧
  C = 2 * A ∧
  D = 2.5 * A ∧
  E = 3 * A

-- Theorem to prove
theorem percentage_increase (A B C D E : ℝ) (A_inc B_inc C_inc D_inc E_inc : ℝ) :
  conditions A B C D E A_inc B_inc C_inc D_inc E_inc →
  (A_inc + B_inc + C_inc + D_inc + E_inc) / (A + B + C + D + E) = 0.05 :=
by
  sorry

end percentage_increase_l231_231433


namespace factor_difference_of_squares_l231_231856

theorem factor_difference_of_squares (y : ℝ) : 25 - 16 * y ^ 2 = (5 - 4 * y) * (5 + 4 * y) :=
by
  sorry

end factor_difference_of_squares_l231_231856


namespace sequence_formula_l231_231041

-- Defining the sequence and the conditions
def bounded_seq (a : ℕ → ℝ) : Prop :=
  ∃ C > 0, ∀ n, |a n| ≤ C

-- Statement of the problem in Lean
theorem sequence_formula (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = 3 * a n - 4) →
  bounded_seq a →
  ∀ n : ℕ, a n = 2 :=
by
  intros h1 h2
  sorry

end sequence_formula_l231_231041


namespace union_of_A_and_B_l231_231356

open Set -- to use set notation and operations

def A : Set ℝ := { x | -1/2 < x ∧ x < 2 }

def B : Set ℝ := { x | x^2 ≤ 1 }

theorem union_of_A_and_B :
  A ∪ B = Ico (-1:ℝ) 2 := 
by
  -- proof steps would go here, but we skip these with sorry.
  sorry

end union_of_A_and_B_l231_231356


namespace sum_series_eq_4_div_9_l231_231503

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l231_231503


namespace tiffany_lives_after_game_l231_231673

/-- Tiffany's initial number of lives -/
def initial_lives : ℕ := 43

/-- Lives Tiffany loses in the hard part of the game -/
def lost_lives : ℕ := 14

/-- Lives Tiffany gains in the next level -/
def gained_lives : ℕ := 27

/-- Calculate the total lives Tiffany has after losing and gaining lives -/
def total_lives : ℕ := (initial_lives - lost_lives) + gained_lives

-- Prove that the total number of lives Tiffany has is 56
theorem tiffany_lives_after_game : total_lives = 56 := by
  -- This is where the proof would go
  sorry

end tiffany_lives_after_game_l231_231673


namespace M_subset_N_cond_l231_231548

theorem M_subset_N_cond (a : ℝ) (h : 0 < a) :
  (∀ p : ℝ × ℝ, p ∈ {p : ℝ × ℝ | p.fst^2 + p.snd^2 = a^2} → p ∈ {p : ℝ × ℝ | |p.fst + p.snd| + |p.fst - p.snd| ≤ 2}) ↔ (0 < a ∧ a ≤ 1) :=
sorry

end M_subset_N_cond_l231_231548


namespace average_speed_second_bus_l231_231231

theorem average_speed_second_bus (x : ℝ) (h1 : x > 0) :
  (12 / x) - (12 / (1.2 * x)) = 3 / 60 :=
by
  sorry

end average_speed_second_bus_l231_231231


namespace rth_term_arithmetic_progression_l231_231128

-- Define the sum of the first n terms of the arithmetic progression
def S (n : ℕ) : ℕ := 5 * n + 4 * n^3

-- Define the r-th term of the arithmetic progression
def a (r : ℕ) : ℕ := S r - S (r - 1)

-- Theorem stating the r-th term of the arithmetic progression
theorem rth_term_arithmetic_progression (r : ℕ) : a r = 12 * r^2 - 12 * r + 9 := by
  sorry

end rth_term_arithmetic_progression_l231_231128


namespace sum_of_series_l231_231461

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l231_231461


namespace sum_of_factors_36_l231_231645

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by
  -- The definition of the set of factors of 36
  have factors_36 : Set Nat := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  -- Summing the factors
  have sum_factors_36 := (∑ i in factors_36, i)
  show sum_factors_36 = 91
  sorry

end sum_of_factors_36_l231_231645


namespace ratio_of_speeds_l231_231901

theorem ratio_of_speeds (v_A v_B : ℝ) (h1 : 500 / v_A = 400 / v_B) : v_A / v_B = 5 / 4 :=
by
  sorry

end ratio_of_speeds_l231_231901


namespace total_hunts_is_21_l231_231584

-- Define the initial conditions
def Sam_hunts : Nat := 6
def Rob_hunts : Nat := Sam_hunts / 2
def Rob_Sam_total_hunt : Nat := Sam_hunts + Rob_hunts
def Mark_hunts : Nat := Rob_Sam_total_hunt / 3
def Peter_hunts : Nat := Mark_hunts * 3

-- The main theorem to prove
theorem total_hunts_is_21 : Sam_hunts + Rob_hunts + Mark_hunts + Peter_hunts = 21 :=
by
  sorry

end total_hunts_is_21_l231_231584


namespace series_sum_eq_four_ninths_l231_231492

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l231_231492


namespace sum_infinite_series_l231_231513

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l231_231513


namespace find_value_of_expression_l231_231303

theorem find_value_of_expression (a : ℝ) (h : a^2 + 3 * a - 1 = 0) : 2 * a^2 + 6 * a + 2021 = 2023 := 
by
  sorry

end find_value_of_expression_l231_231303


namespace number_of_parallelograms_l231_231775

theorem number_of_parallelograms : 
  (∀ b d k : ℕ, k > 1 → k * b * d = 500000 → (b * d > 0 ∧ y = x ∧ y = k * x)) → 
  (∃ N : ℕ, N = 720) :=
sorry

end number_of_parallelograms_l231_231775


namespace distinct_solutions_abs_eq_l231_231309

theorem distinct_solutions_abs_eq : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (|2 * x1 - 14| = |x1 + 4| ∧ |2 * x2 - 14| = |x2 + 4|) ∧ (∀ x, |2 * x - 14| = |x + 4| → (x = x1 ∨ x = x2)) :=
by {
  sorry
}

end distinct_solutions_abs_eq_l231_231309


namespace gcd_1230_990_l231_231537

theorem gcd_1230_990 : Int.gcd 1230 990 = 30 := by
  sorry

end gcd_1230_990_l231_231537


namespace intersection_point_exists_l231_231321

theorem intersection_point_exists
  (m n a b : ℝ)
  (h1 : m * a + 2 * m * b = 5)
  (h2 : n * a - 2 * n * b = 7)
  : (∃ x y : ℝ, 
    (y = (5 / (2 * m)) - (1 / 2) * x) ∧ 
    (y = (1 / 2) * x - (7 / (2 * n))) ∧
    (x = a) ∧ (y = b)) :=
sorry

end intersection_point_exists_l231_231321


namespace total_ice_cream_sales_l231_231025

theorem total_ice_cream_sales (tuesday_sales : ℕ) (h1 : tuesday_sales = 12000)
    (wednesday_sales : ℕ) (h2 : wednesday_sales = 2 * tuesday_sales) :
    tuesday_sales + wednesday_sales = 36000 := by
  -- This is the proof statement
  sorry

end total_ice_cream_sales_l231_231025


namespace isabella_hair_growth_l231_231332

theorem isabella_hair_growth :
  ∀ (initial final : ℤ), initial = 18 → final = 24 → final - initial = 6 :=
by
  intros initial final h_initial h_final
  rw [h_initial, h_final]
  exact rfl
-- sorry

end isabella_hair_growth_l231_231332


namespace boat_travel_time_downstream_l231_231668

-- Define the given conditions and statement to prove
theorem boat_travel_time_downstream (B : ℝ) (C : ℝ) (Us : ℝ) (Ds : ℝ) :
  (C = B / 4) ∧ (Us = B - C) ∧ (Ds = B + C) ∧ (Us = 3) ∧ (15 / Us = 5) ∧ (15 / Ds = 3) :=
by
  -- Provide the proof here; currently using sorry to skip the proof
  sorry

end boat_travel_time_downstream_l231_231668


namespace charlie_cost_per_gb_l231_231213

noncomputable def total_data_usage (w1 w2 w3 w4 : ℕ) : ℕ := w1 + w2 + w3 + w4

noncomputable def data_over_limit (total_data usage_limit: ℕ) : ℕ :=
  if total_data > usage_limit then total_data - usage_limit else 0

noncomputable def cost_per_gb (extra_cost data_over_limit: ℕ) : ℕ :=
  if data_over_limit > 0 then extra_cost / data_over_limit else 0

theorem charlie_cost_per_gb :
  let D := 8
  let w1 := 2
  let w2 := 3
  let w3 := 5
  let w4 := 10
  let C := 120
  let total_data := total_data_usage w1 w2 w3 w4
  let data_over := data_over_limit total_data D
  C / data_over = 10 := by
  -- Sorry to skip the proof
  sorry

end charlie_cost_per_gb_l231_231213


namespace infinite_series_sum_l231_231523

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l231_231523


namespace sum_factors_36_l231_231637

theorem sum_factors_36 : (∑ n in {d | d ∣ 36}.to_finset, n) = 91 :=
by
  sorry

end sum_factors_36_l231_231637


namespace find_c_l231_231858

theorem find_c (c : ℝ) 
    (h : ∀ x, (x - 4) ∣ (c * x^3 + 16 * x^2 - 5 * c * x + 40)) : 
    c = -74 / 11 :=
by
  sorry

end find_c_l231_231858


namespace find_AB_l231_231395

theorem find_AB 
  (A B C Q N : Point)
  (h_AQ_QC : AQ / QC = 5 / 2)
  (h_CN_NB : CN / NB = 5 / 2)
  (h_QN : QN = 5 * Real.sqrt 2) : 
  AB = 7 * Real.sqrt 5 :=
sorry

end find_AB_l231_231395


namespace sum_of_factors_36_l231_231639

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := by
  sorry

end sum_of_factors_36_l231_231639


namespace find_train_speed_l231_231970

variable (bridge_length train_length train_crossing_time : ℕ)

def speed_of_train (bridge_length train_length train_crossing_time : ℕ) : ℕ :=
  (bridge_length + train_length) / train_crossing_time

theorem find_train_speed
  (bridge_length : ℕ) (train_length : ℕ) (train_crossing_time : ℕ)
  (h_bridge_length : bridge_length = 180)
  (h_train_length : train_length = 120)
  (h_train_crossing_time : train_crossing_time = 20) :
  speed_of_train bridge_length train_length train_crossing_time = 15 := by
  sorry

end find_train_speed_l231_231970


namespace opposite_of_8_is_neg_8_l231_231975

theorem opposite_of_8_is_neg_8 : - (8 : ℤ) = -8 :=
by
  sorry

end opposite_of_8_is_neg_8_l231_231975


namespace probability_first_queen_second_diamond_l231_231052

def is_queen (card : ℕ) : Prop :=
card = 12 ∨ card = 25 ∨ card = 38 ∨ card = 51

def is_diamond (card : ℕ) : Prop :=
card % 13 = 0

def first_card_queen (first_card : ℕ) (cards : finset ℕ) : Prop :=
is_queen first_card

def second_card_diamond (second_card : ℕ) (remaining_cards : finset ℕ) : Prop :=
is_diamond second_card

theorem probability_first_queen_second_diamond (cards : finset ℕ) (h : cards.card = 52) :
  ((\sum x in cards.filter is_queen, 1 / 52) * (\sum y in cards.filter is_diamond, 1 / 51)) = (1 / 52) :=
sorry

end probability_first_queen_second_diamond_l231_231052


namespace algebra_correct_option_B_l231_231817

theorem algebra_correct_option_B (a b c : ℝ) (h : b * (c^2 + 1) ≠ 0) : 
  (a * (c^2 + 1)) / (b * (c^2 + 1)) = a / b := 
by
  -- Skipping the proof to focus on the statement
  sorry

end algebra_correct_option_B_l231_231817


namespace parabola_expression_l231_231710

theorem parabola_expression:
  (∀ x : ℝ, y = a * (x + 3) * (x - 1)) →
  a * (0 + 3) * (0 - 1) = 2 →
  a = -2 / 3 →
  (∀ x : ℝ, y = -2 / 3 * x^2 - 4 / 3 * x + 2) :=
by
  sorry

end parabola_expression_l231_231710


namespace series_sum_eq_four_ninths_l231_231497

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l231_231497


namespace commission_amount_l231_231770

theorem commission_amount 
  (new_avg_commission : ℤ) (increase_in_avg : ℤ) (sales_count : ℤ) 
  (total_commission_before : ℤ) (total_commission_after : ℤ) : 
  new_avg_commission = 400 → increase_in_avg = 150 → sales_count = 6 → 
  total_commission_before = (sales_count - 1) * (new_avg_commission - increase_in_avg) → 
  total_commission_after = sales_count * new_avg_commission → 
  total_commission_after - total_commission_before = 1150 :=
by 
  sorry

end commission_amount_l231_231770


namespace range_of_a_l231_231749

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x - a) * (1 - x - a) < 1) → -1/2 < a ∧ a < 3/2 :=
by
  sorry

end range_of_a_l231_231749


namespace sum_k_over_4k_l231_231443

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l231_231443


namespace series_sum_eq_four_ninths_l231_231493

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l231_231493


namespace hall_volume_l231_231820

theorem hall_volume (length breadth : ℝ) (h : ℝ)
  (h_length : length = 15) (h_breadth : breadth = 12)
  (h_area : 2 * (length * breadth) = 2 * (breadth * h) + 2 * (length * h)) :
  length * breadth * h = 8004 := 
by
  -- Proof not required
  sorry

end hall_volume_l231_231820


namespace find_prob_xi_less_78_l231_231795

noncomputable def math_test_score_distribution (σ : ℝ) : ProbabilityDist ℝ :=
  NormalDist.mk 85 σ

axiom given_probabilities : ∀ (σ : ℝ),
  P(83 < ↥(math_test_score_distribution σ)) (↥(math_test_score_distribution σ) < 87) = 0.3 ∧
  P(78 < ↥(math_test_score_distribution σ)) (↥(math_test_score_distribution σ) < 83) = 0.13

theorem find_prob_xi_less_78 {σ : ℝ} :
  P(↥(math_test_score_distribution σ) < 78) = 0.22 :=
sorry

end find_prob_xi_less_78_l231_231795


namespace total_hunts_l231_231587

-- Conditions
def Sam_hunts : ℕ := 6
def Rob_hunts := Sam_hunts / 2
def combined_Rob_Sam_hunts := Rob_hunts + Sam_hunts
def Mark_hunts := combined_Rob_Sam_hunts / 3
def Peter_hunts := 3 * Mark_hunts

-- Question and proof statement
theorem total_hunts : Sam_hunts + Rob_hunts + Mark_hunts + Peter_hunts = 21 := by
  sorry

end total_hunts_l231_231587


namespace min_value_fraction_l231_231889

variable {a b : ℝ}

theorem min_value_fraction (h₁ : a + b = 1) (ha : a > 0) (hb : b > 0) : 
  (1 / a + 4 / b) ≥ 9 :=
sorry

end min_value_fraction_l231_231889


namespace slip_3_5_in_F_l231_231006

def slips := [1.5, 2, 2, 2.5, 2.5, 3, 3, 3, 3.5, 3.5, 4, 4, 4.5, 5, 5.5]

def cup_sum (x : List ℝ) := List.sum x

def slips_dist (A B C D E F : List ℝ) : Prop :=
  cup_sum A + cup_sum B + cup_sum C + cup_sum D + cup_sum E + cup_sum F = 50 ∧ 
  cup_sum A = 6 ∧ cup_sum B = 8 ∧ cup_sum C = 10 ∧ cup_sum D = 12 ∧ cup_sum E = 14 ∧ cup_sum F = 16 ∧
  2.5 ∈ B ∧ 2.5 ∈ D ∧ 4 ∈ C

def contains_slip (c : List ℝ) (v : ℝ) : Prop := v ∈ c

theorem slip_3_5_in_F (A B C D E F : List ℝ) (h : slips_dist A B C D E F) : 
  contains_slip F 3.5 :=
sorry

end slip_3_5_in_F_l231_231006


namespace least_comic_books_l231_231009

theorem least_comic_books (n : ℕ) (h1 : n % 7 = 3) (h2 : n % 4 = 1) : n = 17 :=
sorry

end least_comic_books_l231_231009


namespace bob_questions_three_hours_l231_231693

theorem bob_questions_three_hours : 
  let first_hour := 13
  let second_hour := first_hour * 2
  let third_hour := second_hour * 2
  first_hour + second_hour + third_hour = 91 :=
by
  sorry

end bob_questions_three_hours_l231_231693


namespace fewer_onions_grown_l231_231269

def num_tomatoes := 2073
def num_cobs_of_corn := 4112
def num_onions := 985

theorem fewer_onions_grown : num_tomatoes + num_cobs_of_corn - num_onions = 5200 := by
  sorry

end fewer_onions_grown_l231_231269


namespace term_sequence_l231_231246

theorem term_sequence (n : ℕ) (h : (-1:ℤ) ^ (n + 1) * n * (n + 1) = -20) : n = 4 :=
sorry

end term_sequence_l231_231246


namespace total_students_l231_231046

-- Definitions extracted from the conditions 
def ratio_boys_girls := 8 / 5
def number_of_boys := 128

-- Theorem to prove the total number of students
theorem total_students : 
  (128 + (5 / 8) * 128 = 208) ∧ ((128 : ℝ) * (13 / 8) = 208) :=
by
  sorry

end total_students_l231_231046


namespace sum_series_eq_four_ninths_l231_231488

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l231_231488


namespace remainder_when_dividing_698_by_13_is_9_l231_231067

theorem remainder_when_dividing_698_by_13_is_9 :
  ∃ k m : ℤ, 242 = k * 13 + 8 ∧
             698 = m * 13 + 9 ∧
             (k + m) * 13 + 4 = 940 :=
by {
  sorry
}

end remainder_when_dividing_698_by_13_is_9_l231_231067


namespace last_digit_of_large_prime_l231_231836

theorem last_digit_of_large_prime :
  let n := 2^859433 - 1
  let last_digit := n % 10
  last_digit = 1 :=
by
  sorry

end last_digit_of_large_prime_l231_231836


namespace evaluate_expression_l231_231118

theorem evaluate_expression : (1 / (1 - 1 / (3 + 1 / 4))) = (13 / 9) :=
by
  sorry

end evaluate_expression_l231_231118


namespace fraction_product_l231_231422

theorem fraction_product : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end fraction_product_l231_231422


namespace lisa_total_spoons_l231_231206

def num_children := 4
def spoons_per_child := 3
def decorative_spoons := 2
def large_spoons := 10
def teaspoons := 15

def total_spoons := num_children * spoons_per_child + decorative_spoons + large_spoons + teaspoons

theorem lisa_total_spoons : total_spoons = 39 := by
  sorry

end lisa_total_spoons_l231_231206


namespace solutionTriangle_l231_231324

noncomputable def solveTriangle (a b : ℝ) (B : ℝ) : (ℝ × ℝ × ℝ) :=
  let A := 30
  let C := 30
  let c := 2
  (A, C, c)

theorem solutionTriangle :
  solveTriangle 2 (2 * Real.sqrt 3) 120 = (30, 30, 2) :=
by
  sorry

end solutionTriangle_l231_231324


namespace pump_without_leak_time_l231_231094

theorem pump_without_leak_time :
  ∃ T : ℝ, (1/T - 1/5.999999999999999 = 1/3) ∧ T = 2 :=
by 
  sorry

end pump_without_leak_time_l231_231094


namespace range_of_x_l231_231141

noncomputable 
def proposition_p (x : ℝ) : Prop := 6 - 3 * x ≥ 0

noncomputable 
def proposition_q (x : ℝ) : Prop := 1 / (x + 1) < 0

theorem range_of_x (x : ℝ) : proposition_p x ∧ ¬proposition_q x → x ∈ Set.Icc (-1 : ℝ) (2 : ℝ) := by
  sorry

end range_of_x_l231_231141


namespace coat_price_calculation_l231_231087

noncomputable def effective_price (initial_price : ℝ) 
  (reduction1 reduction2 reduction3 : ℝ) 
  (tax1 tax2 tax3 : ℝ) : ℝ :=
  let price_after_first_month := initial_price * (1 - reduction1 / 100) * (1 + tax1 / 100)
  let price_after_second_month := price_after_first_month * (1 - reduction2 / 100) * (1 + tax2 / 100)
  let price_after_third_month := price_after_second_month * (1 - reduction3 / 100) * (1 + tax3 / 100)
  price_after_third_month

noncomputable def total_percent_reduction (initial_price final_price : ℝ) : ℝ :=
  (initial_price - final_price) / initial_price * 100

theorem coat_price_calculation :
  let original_price := 500
  let price_final := effective_price original_price 10 15 20 5 8 6
  let reduction_percentage := total_percent_reduction original_price price_final
  price_final = 367.824 ∧ reduction_percentage = 26.44 :=
by
  sorry

end coat_price_calculation_l231_231087


namespace sum_of_factors_36_l231_231664

theorem sum_of_factors_36 : 
  (∑ i in {d ∈ (finset.range 37) | 36 % d = 0}.to_finset, i) = 91 := 
by
  sorry

end sum_of_factors_36_l231_231664


namespace find_train_probability_l231_231335

-- Define the time range and parameters
def start_time : ℕ := 120
def end_time : ℕ := 240
def wait_time : ℕ := 30

-- Define the conditions
def is_in_range (t : ℕ) : Prop := start_time ≤ t ∧ t ≤ end_time

-- Define the probability function
def probability_of_finding_train : ℚ :=
  let area_triangle : ℚ := (1 / 2) * 30 * 30
  let area_parallelogram : ℚ := 90 * 30
  let shaded_area : ℚ := area_triangle + area_parallelogram
  let total_area : ℚ := (end_time - start_time) * (end_time - start_time)
  shaded_area / total_area

-- The theorem to prove
theorem find_train_probability :
  probability_of_finding_train = 7 / 32 :=
by
  sorry

end find_train_probability_l231_231335


namespace sum_series_eq_4_div_9_l231_231504

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l231_231504


namespace count_silver_coins_l231_231678

theorem count_silver_coins 
  (gold_value : ℕ)
  (silver_value : ℕ)
  (num_gold_coins : ℕ)
  (cash : ℕ)
  (total_money : ℕ) :
  gold_value = 50 →
  silver_value = 25 →
  num_gold_coins = 3 →
  cash = 30 →
  total_money = 305 →
  ∃ S : ℕ, num_gold_coins * gold_value + S * silver_value + cash = total_money ∧ S = 5 := 
by
  sorry

end count_silver_coins_l231_231678


namespace evaluate_expression_l231_231996

theorem evaluate_expression :
  (-3)^4 - (-3)^3 + (-3)^2 - 3^2 + 3^3 - 3^4 = 9 :=
by
  sorry

end evaluate_expression_l231_231996


namespace find_x_l231_231345

theorem find_x (p : ℕ) (hprime : Nat.Prime p) (hgt5 : p > 5) (x : ℕ) (hx : x ≠ 0) :
    (∀ n : ℕ, 0 < n → (5 * p + x) ∣ (5 * p ^ n + x ^ n)) ↔ x = p := by
  sorry

end find_x_l231_231345


namespace inspection_arrangements_l231_231832

-- Definitions based on conditions
def liberal_arts_classes : ℕ := 2
def science_classes : ℕ := 3
def num_students (classes : ℕ) : ℕ := classes

-- Main theorem statement
theorem inspection_arrangements (liberal_arts_classes science_classes : ℕ)
  (h1: liberal_arts_classes = 2) (h2: science_classes = 3) : 
  num_students liberal_arts_classes * num_students science_classes = 24 :=
by {
  -- Given there are 2 liberal arts classes and 3 science classes,
  -- there are exactly 24 ways to arrange the inspections as per the conditions provided.
  sorry
}

end inspection_arrangements_l231_231832


namespace probability_queen_then_diamond_l231_231055

-- Define a standard deck of 52 cards
def deck := List.range 52

-- Define a function to check if a card is a Queen
def is_queen (card : ℕ) : Prop :=
card % 13 = 10

-- Define a function to check if a card is a Diamond. Here assuming index for diamond starts at 0 and ends at 12
def is_diamond (card : ℕ) : Prop :=
card / 13 = 0

-- The main theorem statement
theorem probability_queen_then_diamond : 
  let prob := 1 / 52 * 12 / 51 + 3 / 52 * 13 / 51
  prob = 52 / 221 :=
by
  sorry

end probability_queen_then_diamond_l231_231055


namespace range_of_m_cond_l231_231576

noncomputable def quadratic_inequality (x m : ℝ) : Prop :=
  x^2 + m * x + 2 * m - 3 ≥ 0

theorem range_of_m_cond (m : ℝ) (h1 : 2 ≤ m) (h2 : m ≤ 6) (x : ℝ) :
  quadratic_inequality x m :=
sorry

end range_of_m_cond_l231_231576


namespace min_value_expression_l231_231354

theorem min_value_expression (k x y z : ℝ) (hk : 0 < k) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ x_min y_min z_min : ℝ, (0 < x_min) ∧ (0 < y_min) ∧ (0 < z_min) ∧
  (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z →
    k * (4 * z / (2 * x + y) + 4 * x / (y + 2 * z) + y / (x + z))
    ≥ 3 * k) ∧
  k * (4 * z_min / (2 * x_min + y_min) + 4 * x_min / (y_min + 2 * z_min) + y_min / (x_min + z_min)) = 3 * k :=
by sorry

end min_value_expression_l231_231354


namespace range_of_m_l231_231943

theorem range_of_m (x y : ℝ) (m : ℝ) (h1 : x^2 + y^2 = 9) (h2 : |x| + |y| ≥ m) :
    m ≤ 3 / 2 := 
sorry

end range_of_m_l231_231943


namespace one_way_ticket_cost_l231_231398

theorem one_way_ticket_cost (x : ℝ) (h : 50 / 26 < x) : x >= 2 :=
by sorry

end one_way_ticket_cost_l231_231398


namespace sum_series_eq_4_div_9_l231_231454

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l231_231454


namespace hemisphere_surface_area_l231_231233

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (hπ : π = Real.pi) (h : π * r^2 = 3) :
    2 * π * r^2 + 3 = 9 :=
by
  sorry

end hemisphere_surface_area_l231_231233


namespace vector_subtraction_magnitude_l231_231308

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Given conditions
def condition1 : Real := 3 -- |a|
def condition2 : Real := 2 -- |b|
def condition3 : Real := 4 -- |a + b|

-- Proving the statement
theorem vector_subtraction_magnitude (h1 : ‖a‖ = condition1) (h2 : ‖b‖ = condition2) (h3 : ‖a + b‖ = condition3) :
  ‖a - b‖ = Real.sqrt 10 :=
by
  sorry

end vector_subtraction_magnitude_l231_231308


namespace maximum_value_of_z_l231_231304

theorem maximum_value_of_z :
  ∃ x y : ℝ, (x - y ≥ 0) ∧ (x + y ≤ 2) ∧ (y ≥ 0) ∧ (∀ u v : ℝ, (u - v ≥ 0) ∧ (u + v ≤ 2) ∧ (v ≥ 0) → 3 * u - v ≤ 6) :=
by
  sorry

end maximum_value_of_z_l231_231304


namespace infinite_series_sum_l231_231521

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l231_231521


namespace fewer_onions_correct_l231_231273

-- Define the quantities
def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

-- Calculate the total number of tomatoes and corn
def tomatoes_and_corn : ℕ := tomatoes + corn

-- Calculate the number of fewer onions
def fewer_onions : ℕ := tomatoes_and_corn - onions

-- State the theorem and provide the proof
theorem fewer_onions_correct : fewer_onions = 5200 :=
by
  -- The statement is proved directly by the calculations above
  -- Providing the actual proof is not necessary as per the guidelines
  sorry

end fewer_onions_correct_l231_231273


namespace lineup_condition1_lineup_condition2_lineup_condition3_l231_231379

-- Define the problem for Condition 1
theorem lineup_condition1 (total_positions : ℕ) (middle_positions : ℕ) (positions_to_choose : ℕ) (ways_middle : ℕ) (ways_remaining : ℕ) 
  (total_ways : ℕ) :
  total_positions = 7 → middle_positions = 5 → positions_to_choose = 2 →
  ways_middle * ways_remaining = total_ways → ways_middle = (middle_positions !)/(positions_to_choose ! * (middle_positions - positions_to_choose) !) →
  ways_remaining = (middle_positions !)/((middle_positions - way_remaining)!) → 
  total_ways = 2400 :=
by {
  sorry
}

-- Define the problem for Condition 2
theorem lineup_condition2 (boys : ℕ) (girls : ℕ) (ways_boys : ℕ) (ways_girls : ℕ) (ways_units : ℕ) (total_ways : ℕ) :
  boys = 3 → girls = 4 → 
  ways_boys = boys ! → ways_girls = girls ! →
  ways_units = 2 ! →
  total_ways = ways_boys * ways_girls * ways_units →
  total_ways = 288 :=
by {
  sorry
}

-- Define the problem for Condition 3
theorem lineup_condition3 (girls : ℕ) (positions_for_boys : ℕ) (ways_girls : ℕ) (ways_boys : ℕ) (total_ways : ℕ) :
  girls = 4 → positions_for_boys = girls + 1 →
  ways_girls = girls ! → ways_boys = (positions_for_boys !)/((positions_for_boys - 3)!) →
  total_ways = ways_girls * ways_boys →
  total_ways = 1440 :=
by {
  sorry
}

end lineup_condition1_lineup_condition2_lineup_condition3_l231_231379


namespace value_of_a_sub_b_l231_231574

theorem value_of_a_sub_b (a b : ℝ) (h1 : abs a = 8) (h2 : abs b = 5) (h3 : a > 0) (h4 : b < 0) : a - b = 13 := 
  sorry

end value_of_a_sub_b_l231_231574


namespace ratio_of_B_to_C_l231_231951

theorem ratio_of_B_to_C (A B C : ℕ) (h1 : A + B + C = 98) (h2 : (A : ℚ) / B = 2 / 3) (h3 : B = 30) : ((B : ℚ) / C) = 5 / 8 :=
by
  sorry

end ratio_of_B_to_C_l231_231951


namespace value_of_k_l231_231288

theorem value_of_k (k : ℕ) (h : 24 / k = 4) : k = 6 := by
  sorry

end value_of_k_l231_231288


namespace matrix_determinant_l231_231346

variables (a b c : ℝ^3)

noncomputable def E : ℝ := Matrix.det ![![2 * a, 3 * b, 4 * c]].transpose

theorem matrix_determinant (a b c : ℝ^3) :
  Matrix.det ![![2 * a + 3 * b, 3 * b + 4 * c, 4 * c + 2 * a]].transpose = 48 * E a b c :=
sorry

end matrix_determinant_l231_231346


namespace sum_of_factors_36_l231_231661

def is_factor (n m : ℕ) : Prop :=
  m > 0 ∧ n % m = 0

def factors (n : ℕ) : List ℕ :=
  List.filter (is_factor n) (List.range (n + 1))

theorem sum_of_factors_36 : List.sum (factors 36) = 91 := 
  sorry

end sum_of_factors_36_l231_231661


namespace first_digit_base_5_of_2197_l231_231633

theorem first_digit_base_5_of_2197 : 
  ∃ k : ℕ, 2197 = k * 625 + r ∧ k = 3 ∧ r < 625 :=
by
  -- existence of k and r follows from the division algorithm
  -- sorry is used to indicate the part of the proof that needs to be filled in
  sorry

end first_digit_base_5_of_2197_l231_231633


namespace infinite_series_sum_l231_231528

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l231_231528


namespace age_of_older_friend_l231_231938

theorem age_of_older_friend (a b : ℕ) (h1 : a - b = 2) (h2 : a + b = 74) : a = 38 :=
by
  sorry

end age_of_older_friend_l231_231938


namespace complements_intersection_l231_231019

open Set

noncomputable def U : Set ℕ := { x | x ≤ 5 }
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 4}

theorem complements_intersection :
  (U \ A) ∩ (U \ B) = {0, 5} :=
by
  sorry

end complements_intersection_l231_231019


namespace ratio_of_areas_l231_231215

-- Definition of sides and given condition
variables {a b c d : ℝ}
-- Given condition in the problem.
axiom condition : a / c = 3 / 5 ∧ b / d = 3 / 5

-- Statement of the theorem to be proved in Lean 4
theorem ratio_of_areas (h : a / c = 3 / 5) (h' : b / d = 3 / 5) : (a * b) / (c * d) = 9 / 25 :=
by sorry

end ratio_of_areas_l231_231215


namespace perfect_squares_diff_two_consecutive_l231_231312

theorem perfect_squares_diff_two_consecutive (n : ℕ) (h : n = 20000) :
  {a : ℕ | ∃ b : ℕ, b * 2 + 1 < n ∧ a^2 = b * 2 + 1}.card = 71 :=
by
  sorry

end perfect_squares_diff_two_consecutive_l231_231312


namespace cube_red_face_probability_l231_231361

theorem cube_red_face_probability :
  let faces_total := 6
  let red_faces := 3
  let probability_red := red_faces / faces_total
  probability_red = 1 / 2 :=
by
  sorry

end cube_red_face_probability_l231_231361


namespace calculate_r_l231_231961

def a := 0.24 * 450
def b := 0.62 * 250
def c := 0.37 * 720
def d := 0.38 * 100
def sum_bc := b + c
def diff := sum_bc - a
def r := diff / d

theorem calculate_r : r = 8.25 := by
  sorry

end calculate_r_l231_231961


namespace arithmetic_example_l231_231842

theorem arithmetic_example : (2468 * 629) / (1234 * 37) = 34 :=
by
  sorry

end arithmetic_example_l231_231842


namespace millie_initial_bracelets_l231_231771

theorem millie_initial_bracelets (n : ℕ) (h1 : n - 2 = 7) : n = 9 :=
sorry

end millie_initial_bracelets_l231_231771


namespace original_price_of_computer_l231_231746

theorem original_price_of_computer :
  ∃ (P : ℝ), (1.30 * P = 377) ∧ (2 * P = 580) ∧ (P = 290) :=
by
  existsi (290 : ℝ)
  sorry

end original_price_of_computer_l231_231746


namespace range_of_a_l231_231136

theorem range_of_a (a x : ℝ) (p : 0.5 ≤ x ∧ x ≤ 1) (q : (x - a) * (x - a - 1) > 0) :
  (0 ≤ a ∧ a ≤ 0.5) :=
by 
  sorry

end range_of_a_l231_231136


namespace infinite_series_sum_l231_231525

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l231_231525


namespace laila_scores_possible_values_l231_231341

theorem laila_scores_possible_values :
  ∃ (num_y_values : ℕ), num_y_values = 4 ∧ 
  (∀ (x y : ℤ), 0 ≤ x ∧ x ≤ 100 ∧
                 0 ≤ y ∧ y ≤ 100 ∧
                 4 * x + y = 410 ∧
                 y > x → 
                 (y = 86 ∨ y = 90 ∨ y = 94 ∨ y = 98)
  ) :=
  ⟨4, by sorry⟩

end laila_scores_possible_values_l231_231341


namespace two_A_minus_B_l231_231176

theorem two_A_minus_B (A B : ℝ) 
  (h1 : Real.tan (A - B - Real.pi) = 1 / 2) 
  (h2 : Real.tan (3 * Real.pi - B) = 1 / 7) : 
  2 * A - B = -3 * Real.pi / 4 :=
sorry

end two_A_minus_B_l231_231176


namespace pen_sales_average_l231_231603

theorem pen_sales_average :
  ∃ d : ℕ, (48 = (96 + 44 * d) / (d + 1)) → d = 12 :=
by
  sorry

end pen_sales_average_l231_231603


namespace sin_negative_angle_periodic_l231_231416

theorem sin_negative_angle_periodic :
  sin (-17 * Real.pi / 3) = (Real.sqrt 3) / 2 :=
by
  sorry

end sin_negative_angle_periodic_l231_231416


namespace f_2019_equals_neg2_l231_231143

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)
variable (h_period : ∀ x : ℝ, f (x + 4) = f x)
variable (h_defined : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2)

theorem f_2019_equals_neg2 : f 2019 = -2 :=
by 
  sorry

end f_2019_equals_neg2_l231_231143


namespace max_good_pairs_1_to_30_l231_231254

def is_good_pair (a b : ℕ) : Prop := a % b = 0 ∨ b % a = 0

def max_good_pairs_in_range (n : ℕ) : ℕ :=
  if n = 30 then 13 else 0

theorem max_good_pairs_1_to_30 : max_good_pairs_in_range 30 = 13 :=
by
  sorry

end max_good_pairs_1_to_30_l231_231254


namespace find_positive_integer_x_l231_231285

theorem find_positive_integer_x :
  ∃ x : ℕ, x > 0 ∧ (5 * x + 1) / (x - 1) > 2 * x + 2 ∧
  ∀ y : ℕ, y > 0 ∧ (5 * y + 1) / (y - 1) > 2 * x + 2 → y = 2 :=
sorry

end find_positive_integer_x_l231_231285


namespace library_visitors_equation_l231_231601

-- Variables representing conditions
variable (x : ℝ)  -- Monthly average growth rate
variable (first_month_visitors : ℝ) -- Visitors in the first month
variable (total_visitors_by_third_month : ℝ) -- Total visitors by the third month

-- Setting specific values for conditions
def first_month_visitors := 600
def total_visitors_by_third_month := 2850

-- The Lean statement that the specified equation holds
theorem library_visitors_equation :
  first_month_visitors + first_month_visitors * (1 + x) + first_month_visitors * (1 + x)^2 = total_visitors_by_third_month :=
sorry

end library_visitors_equation_l231_231601


namespace ratio_xyz_l231_231994

theorem ratio_xyz (a x y z : ℝ) : 
  5 * x + 4 * y - 6 * z = a ∧
  4 * x - 5 * y + 7 * z = 27 * a ∧
  6 * x + 5 * y - 4 * z = 18 * a →
  (x :ℝ) / (y :ℝ) = 3 / 4 ∧
  (y :ℝ) / (z :ℝ) = 4 / 5 :=
by
  sorry

end ratio_xyz_l231_231994


namespace smallest_five_consecutive_even_sum_320_l231_231376

theorem smallest_five_consecutive_even_sum_320 : ∃ (a b c d e : ℤ), a + b + c + d + e = 320 ∧ (∀ i j : ℤ, (i = a ∨ i = b ∨ i = c ∨ i = d ∨ i = e) → (j = a ∨ j = b ∨ j = c ∨ j = d ∨ j = e) → (i = j + 2 ∨ i = j - 2 ∨ i = j)) ∧ (a ≤ b ∧ a ≤ c ∧ a ≤ d ∧ a ≤ e) ∧ a = 60 :=
by
  sorry

end smallest_five_consecutive_even_sum_320_l231_231376


namespace fractional_product_l231_231426

theorem fractional_product :
  ((3/4) * (4/5) * (5/6) * (6/7) * (7/8)) = 3/8 :=
by
  sorry

end fractional_product_l231_231426


namespace sum_of_two_numbers_l231_231223

theorem sum_of_two_numbers :
  ∀ (A B : ℚ), (A - B = 8) → (1 / 4 * (A + B) = 6) → (A = 16) → (A + B = 24) :=
by
  intros A B h1 h2 h3
  sorry

end sum_of_two_numbers_l231_231223


namespace simplify_expression_l231_231706

theorem simplify_expression : 
    1 - 1 / (1 + Real.sqrt (2 + Real.sqrt 3)) + 1 / (1 - Real.sqrt (2 - Real.sqrt 3)) 
    = 1 + (Real.sqrt (2 - Real.sqrt 3) + Real.sqrt (2 + Real.sqrt 3)) / (-1 - Real.sqrt 3) := 
by
  sorry

end simplify_expression_l231_231706


namespace intersection_A_B_l231_231882

open Set

variable (x : ℝ)

def A : Set ℝ := {x | x^2 - 3 * x - 4 < 0}
def B : Set ℝ := {-4, 1, 3, 5}

theorem intersection_A_B :
  A ∩ B = { 1, 3 } :=
sorry

end intersection_A_B_l231_231882


namespace wizard_elixir_combinations_l231_231684

theorem wizard_elixir_combinations :
  let herbs := 4
  let crystals := 6
  let invalid_combinations := 3
  herbs * crystals - invalid_combinations = 21 := 
by
  sorry

end wizard_elixir_combinations_l231_231684


namespace infinite_series_sum_l231_231526

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l231_231526


namespace min_value_square_distance_l231_231718

theorem min_value_square_distance (x y : ℝ) (h : x^2 + y^2 - 4*x + 2 = 0) : 
  ∃ c, (∀ x y : ℝ, x^2 + y^2 - 4*x + 2 = 0 → x^2 + (y - 2)^2 ≥ c) ∧ c = 2 :=
sorry

end min_value_square_distance_l231_231718


namespace max_four_color_rectangles_l231_231866

def color := Fin 4
def grid := Fin 100 × Fin 100
def colored_grid := grid → color

def count_four_color_rectangles (g : colored_grid) : ℕ := sorry

theorem max_four_color_rectangles (g : colored_grid) :
  count_four_color_rectangles g ≤ 9375000 := sorry

end max_four_color_rectangles_l231_231866


namespace num_divisible_by_both_digits_l231_231113

theorem num_divisible_by_both_digits : 
  ∃ n, n = 14 ∧ ∀ (d : ℕ), (d ≥ 10 ∧ d < 100) → 
      (∀ a b, (d = 10 * a + b) → d % a = 0 ∧ d % b = 0 → (a = b ∨ a * 2 = b ∨ a * 5 = b)) :=
sorry

end num_divisible_by_both_digits_l231_231113


namespace find_a_value_l231_231830

theorem find_a_value :
  (∀ (x y : ℝ), (x = 1.5 → y = 8 → x * y = 12) ∧ 
               (x = 2 → y = 6 → x * y = 12) ∧ 
               (x = 3 → y = 4 → x * y = 12)) →
  ∃ (a : ℝ), (5 * a = 12 ∧ a = 2.4) :=
by
  sorry

end find_a_value_l231_231830


namespace pears_to_peaches_l231_231008

-- Define the weights of pears and peaches
variables (pear peach : ℝ) 

-- Given conditions: 9 pears weigh the same as 6 peaches
axiom weight_ratio : 9 * pear = 6 * peach

-- Theorem to prove: 36 pears weigh the same as 24 peaches
theorem pears_to_peaches (h : 9 * pear = 6 * peach) : 36 * pear = 24 * peach :=
by
  sorry

end pears_to_peaches_l231_231008


namespace sum_series_eq_4_div_9_l231_231499

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l231_231499


namespace sum_series_l231_231473

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l231_231473


namespace nathalie_total_coins_l231_231773

theorem nathalie_total_coins
  (quarters dimes nickels : ℕ)
  (ratio_condition : quarters = 9 * nickels ∧ dimes = 3 * nickels)
  (value_condition : 25 * quarters + 10 * dimes + 5 * nickels = 1820) :
  quarters + dimes + nickels = 91 :=
by
  sorry

end nathalie_total_coins_l231_231773


namespace kiana_siblings_ages_l231_231760

/-- Kiana has two twin brothers, one is twice as old as the other, 
and their ages along with Kiana's age multiply to 72. Prove that 
the sum of their ages is 13. -/
theorem kiana_siblings_ages
  (y : ℕ) (K : ℕ) (h1 : 2 * y * K = 72) :
  y + 2 * y + K = 13 := 
sorry

end kiana_siblings_ages_l231_231760


namespace sum_max_min_ratio_l231_231699

theorem sum_max_min_ratio (x y : ℝ) 
  (h_ellipse : 3 * x^2 + 2 * x * y + 4 * y^2 - 14 * x - 24 * y + 47 = 0) 
  : (∃ m_max m_min : ℝ, (∀ (x y : ℝ), 3 * x^2 + 2 * x * y + 4 * y^2 - 14 * x - 24 * y + 47 = 0 → y = m_max * x ∨ y = m_min * x) ∧ (m_max + m_min = 37 / 22)) :=
sorry

end sum_max_min_ratio_l231_231699


namespace isosceles_triangle_perimeter_l231_231887

theorem isosceles_triangle_perimeter (m x₁ x₂ : ℝ) (h₁ : 1^2 + m * 1 + 5 = 0) 
  (hx : x₁^2 + m * x₁ + 5 = 0 ∧ x₂^2 + m * x₂ + 5 = 0)
  (isosceles : (x₁ = x₂ ∨ x₁ = 1 ∨ x₂ = 1)) : 
  ∃ (P : ℝ), P = 11 :=
by 
  -- Here, you'd prove that under these conditions, the perimeter must be 11.
  sorry

end isosceles_triangle_perimeter_l231_231887


namespace no_nontrivial_sum_periodic_functions_l231_231677

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ ∀ x, f (x + p) = f x

def is_nontrivial_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := 
  periodic f p ∧ ∃ x y, x ≠ y ∧ f x ≠ f y

theorem no_nontrivial_sum_periodic_functions (g h : ℝ → ℝ) :
  is_nontrivial_periodic_function g 1 →
  is_nontrivial_periodic_function h π →
  ¬ ∃ T > 0, ∀ x, (g + h) (x + T) = (g + h) x :=
sorry

end no_nontrivial_sum_periodic_functions_l231_231677


namespace unique_pair_fraction_l231_231922

theorem unique_pair_fraction (p : ℕ) (hprime : Nat.Prime p) (hodd : p % 2 = 1) :
  ∃! (n m : ℕ), (n ≠ m) ∧ (2 / (p : ℚ) = 1 / (n : ℚ) + 1 / (m : ℚ)) ∧
  ((n = (p + 1) / 2 ∧ m = p * (p + 1) / 2) ∨ (m = (p + 1) / 2 ∧ n = p * (p + 1) / 2)) := sorry

end unique_pair_fraction_l231_231922


namespace locus_of_center_l231_231533

-- Define point A
def PointA : ℝ × ℝ := (-2, 0)

-- Define the tangent line
def TangentLine : ℝ := 2

-- The condition to prove the locus equation
theorem locus_of_center (x₀ y₀ : ℝ) :
  (∃ r : ℝ, abs (x₀ - TangentLine) = r ∧ (x₀ + 2)^2 + y₀^2 = r^2) →
  y₀^2 = -8 * x₀ := by
  sorry

end locus_of_center_l231_231533


namespace sum_k_over_4k_l231_231447

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l231_231447


namespace sum_of_series_l231_231464

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l231_231464


namespace total_hunts_l231_231586

-- Conditions
def Sam_hunts : ℕ := 6
def Rob_hunts := Sam_hunts / 2
def combined_Rob_Sam_hunts := Rob_hunts + Sam_hunts
def Mark_hunts := combined_Rob_Sam_hunts / 3
def Peter_hunts := 3 * Mark_hunts

-- Question and proof statement
theorem total_hunts : Sam_hunts + Rob_hunts + Mark_hunts + Peter_hunts = 21 := by
  sorry

end total_hunts_l231_231586


namespace calories_difference_l231_231108

theorem calories_difference
  (calories_squirrel : ℕ := 300)
  (squirrels_per_hour : ℕ := 6)
  (calories_rabbit : ℕ := 800)
  (rabbits_per_hour : ℕ := 2) :
  ((squirrels_per_hour * calories_squirrel) - (rabbits_per_hour * calories_rabbit)) = 200 :=
by
  sorry

end calories_difference_l231_231108


namespace problem1_problem2_problem3_problem4_l231_231277

theorem problem1 : (-3 + 8 - 7 - 15) = -17 := 
sorry

theorem problem2 : (23 - 6 * (-3) + 2 * (-4)) = 33 := 
sorry

theorem problem3 : (-8 / (4 / 5) * (-2 / 3)) = 20 / 3 := 
sorry

theorem problem4 : (-2^2 - 9 * (-1 / 3)^2 + abs (-4)) = -1 := 
sorry

end problem1_problem2_problem3_problem4_l231_231277


namespace total_chocolate_bars_in_large_box_l231_231962

-- Define the given conditions
def small_boxes : ℕ := 16
def chocolate_bars_per_box : ℕ := 25

-- State the proof problem
theorem total_chocolate_bars_in_large_box :
  small_boxes * chocolate_bars_per_box = 400 :=
by
  -- The proof is omitted
  sorry

end total_chocolate_bars_in_large_box_l231_231962


namespace infinite_series_sum_l231_231517

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l231_231517


namespace determine_weights_l231_231218

-- Definitions
variable {W : Type} [AddCommGroup W] [OrderedAddCommMonoid W]
variable (w : Fin 20 → W) -- List of weights for 20 people
variable (s : W) -- Total sum of weights
variable (lower upper : W) -- Lower and upper weight limits

-- Conditions
def weight_constraints : Prop :=
  (∀ i, lower ≤ w i ∧ w i ≤ upper) ∧ (Finset.univ.sum w = s)

-- Problem statement
theorem determine_weights (w : Fin 20 → ℝ) :
  weight_constraints w 60 90 3040 →
  ∃ w : Fin 20 → ℝ, weight_constraints w 60 90 3040 := by
  sorry

end determine_weights_l231_231218


namespace area_of_triangle_l231_231353

-- Definitions and conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 4) = 1
def focal_distance (a b c : ℝ) : Prop := c = real.sqrt (a^2 - b^2)
def in_ratio (d1 d2 : ℝ) : Prop := d1 = 2 * d2
def point_P_on_ellipse (x y : ℝ) := ellipse x y
def distance_sum (PF1 PF2 : ℝ) (a : ℝ) : Prop := PF1 + PF2 = 2 * a

-- Given parameters
def a := 3
def b := 2
def c := real.sqrt (a^2 - b^2)
def F1 := (-c, 0)
def F2 := (c, 0)

-- Main statement to prove
theorem area_of_triangle
    (x y : ℝ)
    (H1 : ellipse x y)
    (PF1 PF2 : ℝ)
    (H2 : in_ratio PF1 PF2)
    (H3 : distance_sum PF1 PF2 a) :
  let base := 4 -- PF1
  let height := 2 -- PF2
  in (1 / 2) * base * height = 4 := by
  -- You can include the definitions here if necessary for clarity but not the proof itself.
  sorry

end area_of_triangle_l231_231353


namespace sum_infinite_series_l231_231511

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l231_231511


namespace boxes_difference_l231_231071

theorem boxes_difference (white_balls red_balls balls_per_box : ℕ)
  (h_white : white_balls = 30)
  (h_red : red_balls = 18)
  (h_box : balls_per_box = 6) :
  (white_balls / balls_per_box) - (red_balls / balls_per_box) = 2 :=
by 
  sorry

end boxes_difference_l231_231071


namespace vector_x_solution_l231_231323

theorem vector_x_solution (x : ℝ) (a b c : ℝ × ℝ)
  (ha : a = (-2,0))
  (hb : b = (2,1))
  (hc : c = (x,1))
  (collinear : ∃ k : ℝ, 3 • a + b = k • c) :
  x = -4 :=
by
  sorry

end vector_x_solution_l231_231323


namespace possible_real_values_l231_231754

theorem possible_real_values (y: ℝ) (h_mean: (8 + 1 + 7 + 1 + y + 1 + 9) / 7 = (27 + y) / 7)
  (h_mode: 1 = 1) 
  (h_median: (y < 1 → 1 = 1) ∧ (1 < y ∧ y < 7 → y = y) ∧ (y ≥ 7 → 7 = 7)) 
  (h_ap: ∃a b c: ℝ, a = 1 ∧ b = 7 ∧ c = (27 + y) / 7 ∧ (b - a = 7 - 1) ∧ (c - b = (27 + y) / 7 - 7)): y = 64 := 
by {
  sorry
}

end possible_real_values_l231_231754


namespace sum_series_l231_231474

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l231_231474


namespace repeating_decimal_to_fraction_l231_231997

theorem repeating_decimal_to_fraction :
  (2 + (35 / 99 : ℚ)) = (233 / 99) := 
sorry

end repeating_decimal_to_fraction_l231_231997


namespace intersection_A_B_l231_231884

-- Define set A and set B based on the conditions
def set_A : Set ℝ := {x : ℝ | x^2 - 3 * x - 4 < 0}
def set_B : Set ℝ := {-4, 1, 3, 5}

-- State the theorem to prove the intersection of A and B
theorem intersection_A_B : set_A ∩ set_B = {1, 3} :=
by sorry

end intersection_A_B_l231_231884


namespace product_of_three_equal_numbers_l231_231368

theorem product_of_three_equal_numbers
    (a b : ℕ) (x : ℕ)
    (h1 : a = 12)
    (h2 : b = 22)
    (h_mean : (a + b + 3 * x) / 5 = 20) :
    x * x * x = 10648 := by
  sorry

end product_of_three_equal_numbers_l231_231368


namespace share_y_is_18_l231_231403

-- Definitions from conditions
def total_amount := 70
def ratio_x := 100
def ratio_y := 45
def ratio_z := 30
def total_ratio := ratio_x + ratio_y + ratio_z
def part_value := total_amount / total_ratio
def share_y := ratio_y * part_value

-- Statement to be proved
theorem share_y_is_18 : share_y = 18 :=
by
  -- Placeholder for the proof
  sorry

end share_y_is_18_l231_231403


namespace max_a_plus_2b_l231_231148

theorem max_a_plus_2b (a b : ℝ) (h : a^2 + 2 * b^2 = 1) : a + 2 * b ≤ Real.sqrt 3 := 
sorry

end max_a_plus_2b_l231_231148


namespace alice_total_pints_wednesday_l231_231023

-- Define pints of ice cream Alice bought each day
def pints_sunday : ℕ := 4
def pints_monday : ℕ := 3 * pints_sunday
def pints_tuesday : ℕ := (1 / 3 : ℝ) * pints_monday.toReal
def pints_returned_wednesday : ℝ := (1 / 2 : ℝ) * pints_tuesday
def pints_on_wednesday : ℝ := pints_sunday.toReal + pints_monday.toReal + pints_tuesday - pints_returned_wednesday

theorem alice_total_pints_wednesday : pints_on_wednesday = 18 := by
  sorry

end alice_total_pints_wednesday_l231_231023


namespace sum_infinite_series_l231_231508

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l231_231508


namespace evaluate_series_sum_l231_231439

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l231_231439


namespace find_acute_angle_l231_231567

theorem find_acute_angle (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) 
    (h3 : Real.sin α = 1 - Real.sqrt 3 * Real.tan (π / 18) * Real.sin α) : 
    α = π / 3 * 5 / 9 :=
by
  sorry

end find_acute_angle_l231_231567


namespace total_animals_hunted_l231_231581

theorem total_animals_hunted :
  let sam_hunts := 6
  let rob_hunts := sam_hunts / 2
  let total_sam_rob := sam_hunts + rob_hunts
  let mark_hunts := total_sam_rob / 3
  let peter_hunts := mark_hunts * 3
  sam_hunts + rob_hunts + mark_hunts + peter_hunts = 21 :=
by
  sorry

end total_animals_hunted_l231_231581


namespace fraction_product_l231_231418

theorem fraction_product :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := 
by
  -- Detailed proof steps would go here
  sorry

end fraction_product_l231_231418


namespace total_hunts_l231_231588

-- Conditions
def Sam_hunts : ℕ := 6
def Rob_hunts := Sam_hunts / 2
def combined_Rob_Sam_hunts := Rob_hunts + Sam_hunts
def Mark_hunts := combined_Rob_Sam_hunts / 3
def Peter_hunts := 3 * Mark_hunts

-- Question and proof statement
theorem total_hunts : Sam_hunts + Rob_hunts + Mark_hunts + Peter_hunts = 21 := by
  sorry

end total_hunts_l231_231588


namespace sum_series_eq_4_div_9_l231_231457

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l231_231457


namespace probability_of_a_b_c_l231_231385

noncomputable def probability_condition : ℚ :=
  5 / 6 * 5 / 6 * 7 / 8

theorem probability_of_a_b_c : 
  let a_outcome := 6
  let b_outcome := 6
  let c_outcome := 8
  (1 / a_outcome) * (1 / b_outcome) * (1 / c_outcome) = probability_condition :=
sorry

end probability_of_a_b_c_l231_231385


namespace factorize_expression_l231_231240

variables (a b x : ℝ)

theorem factorize_expression :
    5 * a * (x^2 - 1) - 5 * b * (x^2 - 1) = 5 * (x + 1) * (x - 1) * (a - b) := 
by
  sorry

end factorize_expression_l231_231240


namespace fraction_problem_l231_231734

-- Definitions translated from conditions
variables (m n p q : ℚ)
axiom h1 : m / n = 20
axiom h2 : p / n = 5
axiom h3 : p / q = 1 / 15

-- Statement to prove
theorem fraction_problem : m / q = 4 / 15 :=
by
  sorry

end fraction_problem_l231_231734


namespace is_hexagonal_number_2016_l231_231155

theorem is_hexagonal_number_2016 :
  ∃ (n : ℕ), 2 * n^2 - n = 2016 :=
sorry

end is_hexagonal_number_2016_l231_231155


namespace chord_length_l231_231676

theorem chord_length {r : ℝ} (h : r = 15) : 
  ∃ (CD : ℝ), CD = 26 * Real.sqrt 3 :=
by
  sorry

end chord_length_l231_231676


namespace gran_age_indeterminate_l231_231154

theorem gran_age_indeterminate
(gran_age : ℤ) -- Let Gran's age be denoted by gran_age
(guess1 : ℤ := 75) -- The first grandchild guessed 75
(guess2 : ℤ := 78) -- The second grandchild guessed 78
(guess3 : ℤ := 81) -- The third grandchild guessed 81
-- One guess is mistaken by 1 year
(h1 : (abs (gran_age - guess1) = 1) ∨ (abs (gran_age - guess2) = 1) ∨ (abs (gran_age - guess3) = 1))
-- Another guess is mistaken by 2 years
(h2 : (abs (gran_age - guess1) = 2) ∨ (abs (gran_age - guess2) = 2) ∨ (abs (gran_age - guess3) = 2))
-- Another guess is mistaken by 4 years
(h3 : (abs (gran_age - guess1) = 4) ∨ (abs (gran_age - guess2) = 4) ∨ (abs (gran_age - guess3) = 4)) :
  False := sorry

end gran_age_indeterminate_l231_231154


namespace sum_series_eq_four_ninths_l231_231483

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l231_231483


namespace manager_salary_l231_231617

def avg_salary_employees := 1500
def num_employees := 20
def avg_salary_increase := 600
def num_total_people := num_employees + 1

def total_salary_employees := num_employees * avg_salary_employees
def new_avg_salary := avg_salary_employees + avg_salary_increase
def total_salary_with_manager := num_total_people * new_avg_salary

theorem manager_salary : total_salary_with_manager - total_salary_employees = 14100 :=
by
  sorry

end manager_salary_l231_231617


namespace min_value_part1_l231_231965

open Real

theorem min_value_part1 (x : ℝ) (h : x > 1) : (x + 4 / (x - 1)) ≥ 5 :=
by {
  sorry
}

end min_value_part1_l231_231965


namespace gcd_1230_990_l231_231536

theorem gcd_1230_990 : Int.gcd 1230 990 = 30 := by
  sorry

end gcd_1230_990_l231_231536


namespace sum_first_40_terms_l231_231590

-- Given: The sum of the first 10 terms of a geometric sequence is 9
axiom S_10 : ℕ → ℕ
axiom sum_S_10 : S_10 10 = 9 

-- Given: The sum of the terms from the 11th to the 20th is 36
axiom S_20 : ℕ → ℕ
axiom sum_S_20 : S_20 20 - S_10 10 = 36

-- Let Sn be the sum of the first n terms in the geometric sequence
def Sn (n : ℕ) : ℕ := sorry

-- Prove: The sum of the first 40 terms is 144
theorem sum_first_40_terms : Sn 40 = 144 := sorry

end sum_first_40_terms_l231_231590


namespace flag_arrangement_division_l231_231236

noncomputable def flag_arrangement_modulo : ℕ :=
  let num_blue_flags := 9
  let num_red_flags := 8
  let num_slots := num_blue_flags + 1
  let initial_arrangements := (num_slots.choose num_red_flags) * (num_blue_flags + 1)
  let invalid_cases := (num_blue_flags.choose num_red_flags) * 2
  let M := initial_arrangements - invalid_cases
  M % 1000

theorem flag_arrangement_division (M : ℕ) (num_blue_flags num_red_flags : ℕ) :
  num_blue_flags = 9 → num_red_flags = 8 → M = flag_arrangement_modulo → M % 1000 = 432 :=
by
  intros _ _ hM
  rw [hM]
  trivial

end flag_arrangement_division_l231_231236


namespace set_subset_condition_l231_231305

theorem set_subset_condition (a : ℝ) :
  (∀ x, (1 < a * x ∧ a * x < 2) → (-1 < x ∧ x < 1)) → (|a| ≥ 2 ∨ a = 0) :=
by
  intro h
  sorry

end set_subset_condition_l231_231305


namespace minimum_points_to_guarantee_highest_score_l231_231327

theorem minimum_points_to_guarantee_highest_score :
  ∃ (score1 score2 score3 : ℕ), 
   (score1 = 7 ∨ score1 = 4 ∨ score1 = 2) ∧ (score2 = 7 ∨ score2 = 4 ∨ score2 = 2) ∧
   (score3 = 7 ∨ score3 = 4 ∨ score3 = 2) ∧ 
   (∀ (score4 : ℕ), 
     (score4 = 7 ∨ score4 = 4 ∨ score4 = 2) → 
     (score1 + score2 + score3 + score4 < 25)) → 
  score1 + score2 + score3 + 7 ≥ 25 :=
   sorry

end minimum_points_to_guarantee_highest_score_l231_231327


namespace sum_series_eq_4_div_9_l231_231453

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l231_231453


namespace children_count_l231_231799

-- Define the total number of passengers on the airplane
def total_passengers : ℕ := 240

-- Define the ratio of men to women
def men_to_women_ratio : ℕ × ℕ := (3, 2)

-- Define the percentage of passengers who are either men or women
def percent_men_women : ℕ := 60

-- Define the number of children on the airplane
def number_of_children (total : ℕ) (percent : ℕ) : ℕ := 
  (total * (100 - percent)) / 100

theorem children_count :
  number_of_children total_passengers percent_men_women = 96 := by
  sorry

end children_count_l231_231799


namespace other_group_land_l231_231100

def total_land : ℕ := 900
def remaining_land : ℕ := 385
def lizzies_group_land : ℕ := 250

theorem other_group_land :
  total_land - remaining_land - lizzies_group_land = 265 :=
by
  sorry

end other_group_land_l231_231100


namespace find_unknown_rate_l231_231963

variable (x : ℕ)

theorem find_unknown_rate
    (c3 : ℕ := 3 * 100)
    (c5 : ℕ := 5 * 150)
    (n : ℕ := 10)
    (avg_price : ℕ := 160) 
    (h : c3 + c5 + 2 * x = avg_price * n) :
    x = 275 := 
by
  -- Proof goes here.
  sorry

end find_unknown_rate_l231_231963


namespace sin_double_theta_l231_231568

-- Given condition
def given_condition (θ : ℝ) : Prop :=
  Real.cos (Real.pi / 4 - θ) = 1 / 2

-- The statement we want to prove: sin(2θ) = -1/2
theorem sin_double_theta (θ : ℝ) (h : given_condition θ) : Real.sin (2 * θ) = -1 / 2 :=
sorry

end sin_double_theta_l231_231568


namespace total_ice_cream_sales_l231_231026

theorem total_ice_cream_sales (tuesday_sales : ℕ) (h1 : tuesday_sales = 12000)
    (wednesday_sales : ℕ) (h2 : wednesday_sales = 2 * tuesday_sales) :
    tuesday_sales + wednesday_sales = 36000 := by
  -- This is the proof statement
  sorry

end total_ice_cream_sales_l231_231026


namespace circles_intersect_l231_231959

noncomputable def circle1 := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
noncomputable def circle2 := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 9}

theorem circles_intersect :
  ∃ p : ℝ × ℝ, p ∈ circle1 ∧ p ∈ circle2 :=
sorry

end circles_intersect_l231_231959


namespace slope_of_asymptotes_l231_231559

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (x - 2)^2 / 144 - (y + 3)^2 / 81 = 1

-- The theorem stating the slope of the asymptotes
theorem slope_of_asymptotes : ∀ x y : ℝ, hyperbola x y → (∃ m : ℝ, m = 3 / 4) :=
by
  sorry

end slope_of_asymptotes_l231_231559


namespace coordinates_of_D_l231_231783

-- Definitions of the points and translation conditions
def A : (ℝ × ℝ) := (-1, 4)
def B : (ℝ × ℝ) := (-4, -1)
def C : (ℝ × ℝ) := (4, 7)

theorem coordinates_of_D :
  ∃ (D : ℝ × ℝ), D = (1, 2) ∧
  ∀ (translate : ℝ × ℝ), translate = (C.1 - A.1, C.2 - A.2) → 
  D = (B.1 + translate.1, B.2 + translate.2) :=
by
  sorry

end coordinates_of_D_l231_231783


namespace total_nails_l231_231631

-- Definitions based on the conditions
def Violet_nails : ℕ := 27
def Tickletoe_nails : ℕ := (27 - 3) / 2

-- Theorem to prove the total number of nails
theorem total_nails : Violet_nails + Tickletoe_nails = 39 := by
  sorry

end total_nails_l231_231631


namespace sum_factors_36_l231_231649

theorem sum_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by 
  -- Sum the whole-number factors of 36
  sorry

end sum_factors_36_l231_231649


namespace cloves_needed_l231_231393

theorem cloves_needed (cv_fp : 3 / 2 = 1.5) (cw_fp : 3 / 3 = 1) (vc_fp : 3 / 8 = 0.375) : 
  let cloves_for_vampires := 45
  let cloves_for_wights := 12
  let cloves_for_bats := 15
  30 * (3 / 2) + 12 * (3 / 3) + 40 * (3 / 8) = 72 := by
  sorry

end cloves_needed_l231_231393


namespace probability_of_70th_percentile_is_25_over_56_l231_231330

-- Define the weights of the students
def weights : List ℕ := [90, 100, 110, 120, 140, 150, 150, 160]

-- Define the number of students to select
def n_selected_students : ℕ := 3

-- Define the percentile value
def percentile_value : ℕ := 70

-- Define the corresponding weight for the 70th percentile
def percentile_weight : ℕ := 150

-- Define the combination function
noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability calculation
noncomputable def probability_70th_percentile : ℚ :=
  let total_ways := C 8 3
  let favorable_ways := (C 2 2) * (C 5 1) + (C 2 1) * (C 5 2)
  favorable_ways / total_ways

-- Define the theorem to prove the probability
theorem probability_of_70th_percentile_is_25_over_56 :
  probability_70th_percentile = 25 / 56 := by
  sorry

end probability_of_70th_percentile_is_25_over_56_l231_231330


namespace max_value_in_sample_is_10_l231_231002

noncomputable def sample_size : ℕ := 5
noncomputable def sample_mean : ℝ := 7
noncomputable def sample_variance : ℝ := 4

theorem max_value_in_sample_is_10
  (samples : Fin sample_size → ℝ)
  (h1 : (∑ i, samples i) / sample_size = sample_mean)
  (h2 : (∑ i, (samples i - sample_mean) ^ 2) / sample_size = sample_variance)
  (h3 : ∀ i j, i ≠ j → samples i ≠ samples j) : 
  ∃ i, samples i = 10 :=
sorry

end max_value_in_sample_is_10_l231_231002


namespace parabola_y_values_order_l231_231301

theorem parabola_y_values_order :
  let y1 := 2 * (-3 - 2) ^ 2 + 1
  let y2 := 2 * (3 - 2) ^ 2 + 1
  let y3 := 2 * (4 - 2) ^ 2 + 1
  y2 < y3 ∧ y3 < y1 :=
by
  let y1 := 2 * (-3 - 2) ^ 2 + 1
  let y2 := 2 * (3 - 2) ^ 2 + 1
  let y3 := 2 * (4 - 2) ^ 2 + 1
  -- The proof is omitted
  sorry

end parabola_y_values_order_l231_231301


namespace circle_radius_triple_area_l231_231616

/-- Given the area of a circle is tripled when its radius r is increased by n, prove that 
    r = n * (sqrt(3) - 1) / 2 -/
theorem circle_radius_triple_area (r n : ℝ) (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) :
  r = n * (Real.sqrt 3 - 1) / 2 :=
sorry

end circle_radius_triple_area_l231_231616


namespace calculate_fg1_l231_231573

def f (x : ℝ) : ℝ := 4 - 3 * x
def g (x : ℝ) : ℝ := x^3 + 1

theorem calculate_fg1 : f (g 1) = -2 :=
by
  sorry

end calculate_fg1_l231_231573


namespace solution_ratio_l231_231956

-- Describe the problem conditions
variable (a b : ℝ) -- amounts of solutions A and B

-- conditions
def proportion_A : ℝ := 0.20 -- Alcohol concentration in solution A
def proportion_B : ℝ := 0.60 -- Alcohol concentration in solution B
def final_proportion : ℝ := 0.40 -- Final alcohol concentration

-- Lean statement
theorem solution_ratio (h : 0.20 * a + 0.60 * b = 0.40 * (a + b)) : a = b := by
  sorry

end solution_ratio_l231_231956


namespace solution_l231_231114

noncomputable def problem : Prop := 
  - (Real.sin (133 * Real.pi / 180)) * (Real.cos (197 * Real.pi / 180)) -
  (Real.cos (47 * Real.pi / 180)) * (Real.cos (73 * Real.pi / 180)) = 1 / 2

theorem solution : problem :=
by
  sorry

end solution_l231_231114


namespace cookies_taken_in_four_days_l231_231803

def initial_cookies : ℕ := 70
def cookies_left : ℕ := 28
def days_in_week : ℕ := 7
def days_taken : ℕ := 4
def daily_cookies_taken (total_cookies_taken : ℕ) : ℕ := total_cookies_taken / days_in_week
def total_cookies_taken : ℕ := initial_cookies - cookies_left

theorem cookies_taken_in_four_days :
  daily_cookies_taken total_cookies_taken * days_taken = 24 := by
  sorry

end cookies_taken_in_four_days_l231_231803


namespace rice_less_than_beans_by_30_l231_231208

noncomputable def GB : ℝ := 60
noncomputable def S : ℝ := 50

theorem rice_less_than_beans_by_30 (R : ℝ) (x : ℝ) (h1 : R = 60 - x) (h2 : (2/3) * R + (4/5) * S + GB = 120) : 60 - R = 30 :=
by 
  -- Proof steps would go here, but they are not required for this task.
  sorry

end rice_less_than_beans_by_30_l231_231208


namespace Lisa_total_spoons_l231_231201

def total_spoons (children : ℕ) (spoons_per_child : ℕ) (decorative_spoons : ℕ) (large_spoons : ℕ) (teaspoons : ℕ) : ℕ := 
  (children * spoons_per_child) + decorative_spoons + (large_spoons + teaspoons)

theorem Lisa_total_spoons :
  (total_spoons 4 3 2 10 15) = 39 :=
by
  sorry

end Lisa_total_spoons_l231_231201


namespace abs_neg_two_l231_231219

theorem abs_neg_two : abs (-2) = 2 := by
  sorry

end abs_neg_two_l231_231219


namespace find_larger_number_l231_231605

theorem find_larger_number :
  ∃ (x y : ℝ), (y = x + 10) ∧ (x = y / 2) ∧ (x + y = 34) → y = 20 :=
by
  sorry

end find_larger_number_l231_231605


namespace triangle_rectangle_ratio_l231_231978

-- Definitions of the perimeter conditions and the relationship between length and width of the rectangle.
def equilateral_triangle_side_length (t : ℕ) : Prop :=
  3 * t = 24

def rectangle_dimensions (l w : ℕ) : Prop :=
  2 * l + 2 * w = 24 ∧ l = 2 * w

-- The main theorem stating the desired ratio.
theorem triangle_rectangle_ratio (t l w : ℕ) 
  (ht : equilateral_triangle_side_length t) (hlw : rectangle_dimensions l w) : t / w = 2 :=
by
  sorry

end triangle_rectangle_ratio_l231_231978


namespace sum_series_eq_4_div_9_l231_231501

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l231_231501


namespace div_5_implies_one_div_5_l231_231930

theorem div_5_implies_one_div_5 (a b : ℕ) (h : 5 ∣ (a * b)) : (5 ∣ a) ∨ (5 ∣ b) :=
by 
  sorry

end div_5_implies_one_div_5_l231_231930


namespace unique_pair_exists_l231_231915

theorem unique_pair_exists (p : ℕ) (hp : p.prime ) (hodd : p % 2 = 1) : 
  ∃ m n : ℕ, m ≠ n ∧ (2 : ℚ) / (p : ℚ) = 1 / (n : ℚ) + 1 / (m : ℚ) ∧ 
             (n = (p + 1) / 2) ∧ (m = (p * (p + 1)) / 2) :=
by
  sorry

end unique_pair_exists_l231_231915


namespace geometric_series_arithmetic_sequence_l231_231750

noncomputable def geometric_seq_ratio (a : ℕ → ℝ) (q : ℝ) : Prop := 
∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_series_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_seq_ratio a q)
  (h_pos : ∀ n, a n > 0)
  (h_arith : a 1 = (a 0 + 2 * a 1) / 2) :
  a 5 / a 3 = 3 + 2 * Real.sqrt 2 :=
sorry

end geometric_series_arithmetic_sequence_l231_231750


namespace fred_current_money_l231_231762

-- Conditions
def initial_amount_fred : ℕ := 19
def earned_amount_fred : ℕ := 21

-- Question and Proof
theorem fred_current_money : initial_amount_fred + earned_amount_fred = 40 :=
by sorry

end fred_current_money_l231_231762


namespace sum_series_eq_4_div_9_l231_231505

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l231_231505


namespace unique_nat_pair_l231_231918

theorem unique_nat_pair (p : ℕ) (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ (n m : ℕ), n ≠ m ∧ (2 / p : ℚ) = (1 / n + 1 / m : ℚ) ∧ ∀ (n' m' : ℕ), 
  n' ≠ m' ∧ (2 / p : ℚ) = (1 / n' + 1 / m' : ℚ) → (n', m') = (n, m) ∨ (n', m') = (m, n) :=
by
  sorry

end unique_nat_pair_l231_231918


namespace sequence_sum_l231_231716

theorem sequence_sum (a : ℕ → ℝ)
  (h₀ : ∀ n : ℕ, 0 < a n)
  (h₁ : a 1 = 1)
  (h₂ : ∀ n : ℕ, a (n + 2) = 1 + 1 / a n)
  (h₃ : a 2014 = a 2016) :
  a 13 + a 2016 = 21 / 13 + (1 + Real.sqrt 5) / 2 :=
sorry

end sequence_sum_l231_231716


namespace ab_equiv_l231_231546

theorem ab_equiv (a b : ℝ) (hb : b ≠ 0) (h : (a - b) / b = 3 / 7) : a / b = 10 / 7 :=
by
  sorry

end ab_equiv_l231_231546


namespace proposition_B_l231_231130

-- Definitions of planes and lines
variable {Plane : Type}
variable {Line : Type}
variable (α β : Plane)
variable (m n : Line)

-- Definitions of parallel and perpendicular relationships
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (_perpendicular : Line → Line → Prop)

-- Theorem statement
theorem proposition_B (h1 : perpendicular m α) (h2 : parallel n α) : _perpendicular m n :=
sorry

end proposition_B_l231_231130


namespace train_speed_l231_231098

/--A train leaves Delhi at 9 a.m. at a speed of 30 kmph.
Another train leaves at 3 p.m. on the same day and in the same direction.
The two trains meet 720 km away from Delhi.
Prove that the speed of the second train is 120 kmph.-/
theorem train_speed
  (speed_first_train speed_first_kmph : 30 = 30)
  (leave_first_train : Nat)
  (leave_first_9am : 9 = 9)
  (leave_second_train : Nat)
  (leave_second_3pm : 3 = 3)
  (distance_meeting_km : Nat)
  (distance_meeting_720km : 720 = 720) :
  ∃ speed_second_train, speed_second_train = 120 := 
sorry

end train_speed_l231_231098


namespace fraction_equal_l231_231730

variable {m n p q : ℚ}

-- Define the conditions
def condition1 := (m / n = 20)
def condition2 := (p / n = 5)
def condition3 := (p / q = 1 / 15)

-- State the theorem
theorem fraction_equal (h1 : condition1) (h2 : condition2) (h3 : condition3) : (m / q = 4 / 15) :=
  sorry

end fraction_equal_l231_231730


namespace find_nine_boxes_of_same_variety_l231_231973

theorem find_nine_boxes_of_same_variety (boxes : ℕ) (A B C : ℕ) (h_total : boxes = 25) (h_one_variety : boxes = A + B + C) 
  (hA : A ≤ 25) (hB : B ≤ 25) (hC : C ≤ 25) :
  (A ≥ 9) ∨ (B ≥ 9) ∨ (C ≥ 9) :=
sorry

end find_nine_boxes_of_same_variety_l231_231973


namespace division_criterion_based_on_stroke_l231_231106

-- Definition of a drawable figure with a single stroke
def drawable_in_one_stroke (figure : Type) : Prop := sorry -- exact conditions can be detailed with figure representation

-- Example figures for the groups (types can be extended based on actual representation)
def Group1 := {fig1 : Type // drawable_in_one_stroke fig1}
def Group2 := {fig2 : Type // ¬drawable_in_one_stroke fig2}

-- Problem Statement:
theorem division_criterion_based_on_stroke (fig : Type) :
  (drawable_in_one_stroke fig ∨ ¬drawable_in_one_stroke fig) := by
  -- We state that every figure belongs to either Group1 or Group2
  sorry

end division_criterion_based_on_stroke_l231_231106


namespace intersection_with_complement_l231_231558

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {0, 2, 4}

theorem intersection_with_complement (hU : U = {0, 1, 2, 3, 4})
                                     (hA : A = {0, 1, 2, 3})
                                     (hB : B = {0, 2, 4}) :
  A ∩ (U \ B) = {1, 3} :=
by sorry

end intersection_with_complement_l231_231558


namespace total_hunts_is_21_l231_231583

-- Define the initial conditions
def Sam_hunts : Nat := 6
def Rob_hunts : Nat := Sam_hunts / 2
def Rob_Sam_total_hunt : Nat := Sam_hunts + Rob_hunts
def Mark_hunts : Nat := Rob_Sam_total_hunt / 3
def Peter_hunts : Nat := Mark_hunts * 3

-- The main theorem to prove
theorem total_hunts_is_21 : Sam_hunts + Rob_hunts + Mark_hunts + Peter_hunts = 21 :=
by
  sorry

end total_hunts_is_21_l231_231583


namespace probability_four_ones_in_five_rolls_l231_231741

-- Define the probability of rolling a 1 on a fair six-sided die
def prob_one_roll_one : ℚ := 1 / 6

-- Define the probability of not rolling a 1 on a fair six-sided die
def prob_one_roll_not_one : ℚ := 5 / 6

-- Define the number of successes needed, here 4 ones in 5 rolls
def num_successes : ℕ := 4

-- Define the total number of trials, here 5 rolls
def num_trials : ℕ := 5

-- Binomial probability calculation for 4 successes in 5 trials with probability of success prob_one_roll_one
def binomial_prob (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_four_ones_in_five_rolls : binomial_prob num_trials num_successes prob_one_roll_one = 25 / 7776 := 
by
  sorry

end probability_four_ones_in_five_rolls_l231_231741


namespace sum_series_equals_4_div_9_l231_231478

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l231_231478


namespace simplify_expression_l231_231245

theorem simplify_expression : (4 + 3) + (8 - 3 - 1) = 11 := by
  sorry

end simplify_expression_l231_231245


namespace total_boys_in_school_l231_231170

-- Define the total percentage of boys belonging to other communities
def percentage_other_communities := 100 - (44 + 28 + 10)

-- Total number of boys in the school, represented by a variable B
def total_boys (B : ℕ) : Prop :=
0.18 * (B : ℝ) = 117

-- The theorem states that the total number of boys B is 650
theorem total_boys_in_school : ∃ B : ℕ, total_boys B ∧ B = 650 :=
sorry

end total_boys_in_school_l231_231170


namespace calculate_myOp_l231_231701

-- Define the operation
def myOp (x y : ℝ) : ℝ := x^3 - y

-- Given condition for h as a real number
variable (h : ℝ)

-- The theorem we need to prove
theorem calculate_myOp : myOp (2 * h) (myOp (2 * h) (2 * h)) = 2 * h := by
  sorry

end calculate_myOp_l231_231701


namespace appropriate_sampling_method_is_stratified_l231_231251

-- Definition of the problem conditions
def total_students := 500 + 500
def male_students := 500
def female_students := 500
def survey_sample_size := 100

-- The goal is to show that given these conditions, the appropriate sampling method is Stratified sampling method.
theorem appropriate_sampling_method_is_stratified :
  total_students = 1000 ∧
  male_students = 500 ∧
  female_students = 500 ∧
  survey_sample_size = 100 →
  sampling_method = "Stratified" :=
by
  intros h
  sorry

end appropriate_sampling_method_is_stratified_l231_231251


namespace part1_part2_l231_231134

noncomputable def f (x : ℝ) : ℝ := abs (x + 20) - abs (16 - x)

theorem part1 (x : ℝ) : f x ≥ 0 ↔ x ≥ -2 := 
by sorry

theorem part2 (m : ℝ) (x_exists : ∃ x : ℝ, f x ≥ m) : m ≤ 36 := 
by sorry

end part1_part2_l231_231134


namespace intersection_is_3_l231_231727

open Set -- Open the Set namespace to use set notation

theorem intersection_is_3 {A B : Set ℤ} (hA : A = {1, 3}) (hB : B = {-1, 2, 3}) :
  A ∩ B = {3} :=
by {
-- Proof goes here
  sorry
}

end intersection_is_3_l231_231727


namespace amount_over_budget_l231_231107

-- Define the prices of each item
def cost_necklace_A : ℕ := 34
def cost_necklace_B : ℕ := 42
def cost_necklace_C : ℕ := 50
def cost_first_book := cost_necklace_A + 20
def cost_second_book := cost_necklace_C - 10

-- Define Bob's budget
def budget : ℕ := 100

-- Define the total cost
def total_cost := cost_necklace_A + cost_necklace_B + cost_necklace_C + cost_first_book + cost_second_book

-- Prove the amount over budget
theorem amount_over_budget : total_cost - budget = 120 := by
  sorry

end amount_over_budget_l231_231107


namespace correct_propositions_l231_231554

variable (P1 P2 P3 P4 : Prop)

-- Proposition 1: The negation of ∀ x ∈ ℝ, cos(x) > 0 is ∃ x ∈ ℝ such that cos(x) ≤ 0. 
def prop1 : Prop := 
  (¬ (∀ x : ℝ, Real.cos x > 0)) ↔ (∃ x : ℝ, Real.cos x ≤ 0)

-- Proposition 2: If 0 < a < 1, then the equation x^2 + a^x - 3 = 0 has only one real root.
def prop2 : Prop := 
  ∀ a : ℝ, (0 < a ∧ a < 1) → (∃! x : ℝ, x^2 + a^x - 3 = 0)

-- Proposition 3: For any real number x, if f(-x) = f(x) and f'(x) > 0 when x > 0, then f'(x) < 0 when x < 0.
def prop3 (f : ℝ → ℝ) : Prop := 
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, x > 0 → deriv f x > 0) →
  (∀ x : ℝ, x < 0 → deriv f x < 0)

-- Proposition 4: For a rectangle with area S and perimeter l, the pair of real numbers (6, 8) is a valid (S, l) pair.
def prop4 : Prop :=
  ∃ (a b : ℝ), (a * b = 6) ∧ (2 * (a + b) = 8)

theorem correct_propositions (P1_def : prop1)
                            (P3_def : ∀ f : ℝ → ℝ, prop3 f) :
                          P1 ∧ P3 :=
by
  sorry

end correct_propositions_l231_231554


namespace fuel_tank_oil_quantity_l231_231169

theorem fuel_tank_oil_quantity (t : ℝ) (Q : ℝ) : (Q = 40 - 0.2 * t) :=
begin
  sorry
end

end fuel_tank_oil_quantity_l231_231169


namespace value_of_k_l231_231897

theorem value_of_k :
  ∀ (x k : ℝ), (x + 6) * (x - 5) = x^2 + k * x - 30 → k = 1 :=
by
  intros x k h
  sorry

end value_of_k_l231_231897


namespace regular_octagon_angle_l231_231814

theorem regular_octagon_angle (n : ℕ) (h₁ : n = 8) :
  let interior_angle := 135
  let exterior_angle := 45
  interior_angle = 135 ∧ exterior_angle = 45 :=
by
  let interior_sum := 180 * (n - 2)
  have h₂ : interior_sum = 1080 := by
    rw [h₁, Nat.sub_self, Nat.mul_one]
  let int_angle := interior_sum / n
  have h₃ : int_angle = 135 := by
    rw [h₂, h₁]
    norm_num
  let ext_angle := 180 - int_angle
  have h₄ : ext_angle = 45 := by
    rw [h₃]
    norm_num
  split
  · exact h₃
  · exact h₄
  sorry -- Finalizing the proof.

end regular_octagon_angle_l231_231814


namespace probability_sum_16_is_1_over_64_l231_231792

-- Define the problem setup
def octahedral_faces : Finset ℕ := Finset.range 9 \ {0} -- Faces labeled 1 through 8

-- Define the event for the sum of 16
def event_sum_16 : Finset (ℕ × ℕ) :=
  Finset.filter (λ p, p.1 + p.2 = 16) (octahedral_faces.product octahedral_faces)

-- Total outcomes with two octahedral dice
def total_outcomes : ℕ := octahedral_faces.card * octahedral_faces.card

-- Probability of rolling a sum of 16
def probability_sum_16 : ℚ := (event_sum_16.card : ℚ) / total_outcomes

theorem probability_sum_16_is_1_over_64 :
  probability_sum_16 = 1 / 64 := by
  sorry

end probability_sum_16_is_1_over_64_l231_231792


namespace infinite_series_sum_l231_231515

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l231_231515


namespace unusual_digits_exists_l231_231088

def is_unusual (n : ℕ) : Prop :=
  let len := n.digits.count;
  let high_power := 10 ^ len;
  (n^3 % high_power = n) ∧ (n^2 % high_power ≠ n)

theorem unusual_digits_exists :
  ∃ n1 n2 : ℕ, (n1 ≥ 10^99 ∧ n1 < 10^100 ∧ is_unusual n1) ∧ 
             (n2 ≥ 10^99 ∧ n2 < 10^100 ∧ is_unusual n2) ∧
             (n1 ≠ n2) :=
by
  let n1 := 10^100 - 1;
  let n2 := (10^100 / 2) - 1;
  use n1, n2;
  sorry

end unusual_digits_exists_l231_231088


namespace percentage_profit_l231_231578

theorem percentage_profit 
  (C S : ℝ) 
  (h : 29 * C = 24 * S) : 
  ((S - C) / C) * 100 = 20.83 := 
by
  sorry

end percentage_profit_l231_231578


namespace collinear_points_d_value_l231_231049

theorem collinear_points_d_value (a b c d : ℚ)
  (h1 : b = a)
  (h2 : c = -(a+1)/2)
  (collinear : (4 * d * (4 * a + 5) + a + 1 = 0)) :
  d = 9/20 :=
by {
  sorry
}

end collinear_points_d_value_l231_231049


namespace function_value_at_2018_l231_231552

theorem function_value_at_2018 (f : ℝ → ℝ)
  (h1 : f 4 = 2 - Real.sqrt 3)
  (h2 : ∀ x, f (x + 2) = 1 / (- f x)) :
  f 2018 = -2 - Real.sqrt 3 :=
by
  sorry

end function_value_at_2018_l231_231552


namespace sum_series_l231_231468

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l231_231468


namespace inequality_solution_l231_231047

theorem inequality_solution (x : ℝ) : (x^2 - x - 2 < 0) ↔ (-1 < x ∧ x < 2) :=
by
  sorry

end inequality_solution_l231_231047


namespace correct_calculation_l231_231068

theorem correct_calculation (a b : ℝ) : (a * b) ^ 2 = a ^ 2 * b ^ 2 := by
  sorry

end correct_calculation_l231_231068


namespace Cara_skate_distance_l231_231629

-- Definitions corresponding to the conditions
def distance_CD : ℝ := 150
def speed_Cara : ℝ := 10
def speed_Dan : ℝ := 6
def angle_Cara_CD : ℝ := 45

-- main theorem based on the problem and given conditions
theorem Cara_skate_distance : ∃ t : ℝ, distance_CD = 150 ∧ speed_Cara = 10 ∧ speed_Dan = 6
                            ∧ angle_Cara_CD = 45 
                            ∧ 10 * t = 253.5 :=
by
  sorry

end Cara_skate_distance_l231_231629


namespace car_speed_624km_in_2_2_5_hours_l231_231249

theorem car_speed_624km_in_2_2_5_hours : 
  ∀ (distance time_in_hours : ℝ), distance = 624 → time_in_hours = 2 + (2/5) → distance / time_in_hours = 260 :=
by
  intros distance time_in_hours h_dist h_time
  sorry

end car_speed_624km_in_2_2_5_hours_l231_231249


namespace sum_series_eq_4_div_9_l231_231502

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l231_231502


namespace radius_first_field_l231_231377

theorem radius_first_field (r_2 : ℝ) (h_r2 : r_2 = 10) (h_area : ∃ A_2, ∃ A_1, A_1 = 0.09 * A_2 ∧ A_2 = π * r_2^2) : ∃ r_1 : ℝ, r_1 = 3 :=
by
  sorry

end radius_first_field_l231_231377


namespace fraction_problem_l231_231620

theorem fraction_problem (a : ℕ) (h1 : (a:ℚ)/(a + 27) = 865/1000) : a = 173 := 
by
  sorry

end fraction_problem_l231_231620


namespace problem1_div_expr_problem2_div_expr_l231_231406

-- Problem 1
theorem problem1_div_expr : (1 / 30) / ((2 / 3) - (1 / 10) + (1 / 6) - (2 / 5)) = 1 / 10 :=
by 
  -- sorry is added to mark the spot for the proof
  sorry

-- Problem 2
theorem problem2_div_expr : (-1 / 20) / (-(1 / 4) - (2 / 5) + (9 / 10) - (3 / 2)) = 1 / 25 :=
by 
  -- sorry is added to mark the spot for the proof
  sorry

end problem1_div_expr_problem2_div_expr_l231_231406


namespace cloves_of_garlic_needed_l231_231392

def cloves_needed_for_vampires (vampires : ℕ) : ℕ :=
  (vampires * 3) / 2

def cloves_needed_for_wights (wights : ℕ) : ℕ :=
  (wights * 3) / 3

def cloves_needed_for_vampire_bats (vampire_bats : ℕ) : ℕ :=
  (vampire_bats * 3) / 8

theorem cloves_of_garlic_needed (vampires wights vampire_bats : ℕ) :
  cloves_needed_for_vampires 30 + cloves_needed_for_wights 12 + 
  cloves_needed_for_vampire_bats 40 = 72 :=
by
  sorry

end cloves_of_garlic_needed_l231_231392


namespace width_of_room_l231_231794

theorem width_of_room (length room_area cost paving_rate : ℝ) 
  (H_length : length = 5.5) 
  (H_cost : cost = 17600)
  (H_paving_rate : paving_rate = 800)
  (H_area : room_area = cost / paving_rate) :
  room_area = length * 4 :=
by
  -- sorry to skip proof
  sorry

end width_of_room_l231_231794


namespace cost_of_largest_pot_l231_231671

theorem cost_of_largest_pot
    (x : ℝ)
    (hx : 6 * x + (0.1 + 0.2 + 0.3 + 0.4 + 0.5) = 8.25) :
    (x + 0.5) = 1.625 :=
sorry

end cost_of_largest_pot_l231_231671


namespace sum_infinite_series_l231_231514

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l231_231514


namespace seashells_initial_count_l231_231101

theorem seashells_initial_count (S : ℕ)
  (h1 : S - 70 = 2 * 55) : S = 180 :=
by
  sorry

end seashells_initial_count_l231_231101


namespace total_nails_needed_l231_231010

-- Definitions based on problem conditions
def nails_per_plank : ℕ := 2
def planks_needed : ℕ := 2

-- Theorem statement: Prove that the total number of nails John needs is 4.
theorem total_nails_needed : nails_per_plank * planks_needed = 4 := by
  sorry

end total_nails_needed_l231_231010


namespace sum_series_l231_231467

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l231_231467


namespace system_of_equations_solution_l231_231712

theorem system_of_equations_solution
  (x y z : ℤ)
  (h1 : x + y + z = 12)
  (h2 : 8 * x + 5 * y + 3 * z = 60) :
  (x = 0 ∧ y = 12 ∧ z = 0) ∨
  (x = 2 ∧ y = 7 ∧ z = 3) ∨
  (x = 4 ∧ y = 2 ∧ z = 6) :=
sorry

end system_of_equations_solution_l231_231712


namespace probability_first_queen_second_diamond_l231_231061

/-- 
Given that two cards are dealt at random from a standard deck of 52 cards,
the probability that the first card is a Queen and the second card is a Diamonds suit is 1/52.
-/
theorem probability_first_queen_second_diamond : 
  let total_cards := 52
  let probability (num events : ℕ) := (events : ℝ) / (num + 0 : ℝ)
  let prob_first_queen := probability total_cards 4
  let prob_second_diamond_if_first_queen_diamond := probability (total_cards - 1) 12
  let prob_second_diamond_if_first_other_queen := probability (total_cards - 1) 13
  let prob_first_queen_diamond := probability total_cards 1
  let prob_first_queen_other := probability total_cards 3
  let joint_prob_first_queen_diamond := prob_first_queen_diamond * prob_second_diamond_if_first_queen_diamond
  let joint_prob_first_queen_other := prob_first_queen_other * prob_second_diamond_if_first_other_queen
  let total_probability := joint_prob_first_queen_diamond + joint_prob_first_queen_other
  total_probability = probability total_cards 1 := 
by
  -- Proof goes here
  sorry

end probability_first_queen_second_diamond_l231_231061


namespace possible_values_for_a_l231_231018

def A : Set ℝ := {x | x^2 + 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + a * x + 4 = 0}

theorem possible_values_for_a (a : ℝ) : (B a).Nonempty ∧ B a ⊆ A ↔ a = 4 :=
sorry

end possible_values_for_a_l231_231018


namespace inscribed_quadrilateral_inradius_l231_231952

noncomputable def calculate_inradius (a b c d: ℝ) (A: ℝ) : ℝ := (A / ((a + c + b + d) / 2))

theorem inscribed_quadrilateral_inradius {a b c d: ℝ} (h1: a + c = 10) (h2: b + d = 10) (h3: a + b + c + d = 20) (hA: 12 = 12):
  calculate_inradius a b c d 12 = 6 / 5 :=
by
  sorry

end inscribed_quadrilateral_inradius_l231_231952


namespace investment_amount_l231_231728

noncomputable def PV (FV : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  FV / (1 + r) ^ n

theorem investment_amount (FV : ℝ) (r : ℝ) (n : ℕ) (PV : ℝ) : FV = 1000000 ∧ r = 0.08 ∧ n = 20 → PV = 1000000 / (1 + 0.08)^20 :=
by
  intros
  sorry

end investment_amount_l231_231728


namespace evaluate_series_sum_l231_231437

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l231_231437


namespace mark_vs_jenny_bottle_cap_distance_l231_231907

theorem mark_vs_jenny_bottle_cap_distance :
  let jenny_initial := 18
  let jenny_bounce := jenny_initial * (1 / 3)
  let jenny_total := jenny_initial + jenny_bounce
  let mark_initial := 15
  let mark_bounce := mark_initial * 2
  let mark_total := mark_initial + mark_bounce
  mark_total - jenny_total = 21 :=
by
  let jenny_initial := 18
  let jenny_bounce := jenny_initial * (1 / 3)
  let jenny_total := jenny_initial + jenny_bounce
  let mark_initial := 15
  let mark_bounce := mark_initial * 2
  let mark_total := mark_initial + mark_bounce
  calc
    mark_total - jenny_total = (mark_initial + mark_bounce) - (jenny_initial + jenny_bounce) : by sorry
                          ... = (15 + 30) - (18 + 6) : by sorry
                          ... = 45 - 24 : by sorry
                          ... = 21 : by sorry

end mark_vs_jenny_bottle_cap_distance_l231_231907


namespace calc_expr_solve_fractional_eq_l231_231077

-- Problem 1: Calculate the expression
theorem calc_expr : (-2)^2 - (64:ℝ)^(1/3) + (-3)^0 - (1/3)^0 = 0 := 
by 
  -- provide intermediate steps (not necessary as per the question requirements)
  sorry

-- Problem 2: Solve the fractional equation
theorem solve_fractional_eq (x : ℝ) (h : x ≠ -1) : 
  (x / (x + 1) = 5 / (2 * x + 2) - 1) ↔ x = 3 / 4 := 
by 
  -- provide intermediate steps (not necessary as per the question requirements)
  sorry

end calc_expr_solve_fractional_eq_l231_231077


namespace number_of_black_boxcars_l231_231607

def red_boxcars : Nat := 3
def blue_boxcars : Nat := 4
def black_boxcar_capacity : Nat := 4000
def boxcar_total_capacity : Nat := 132000

def blue_boxcar_capacity : Nat := 2 * black_boxcar_capacity
def red_boxcar_capacity : Nat := 3 * blue_boxcar_capacity

def red_boxcar_total_capacity : Nat := red_boxcars * red_boxcar_capacity
def blue_boxcar_total_capacity : Nat := blue_boxcars * blue_boxcar_capacity

def other_total_capacity : Nat := red_boxcar_total_capacity + blue_boxcar_total_capacity
def remaining_capacity : Nat := boxcar_total_capacity - other_total_capacity
def expected_black_boxcars : Nat := remaining_capacity / black_boxcar_capacity

theorem number_of_black_boxcars :
  expected_black_boxcars = 7 := by
  sorry

end number_of_black_boxcars_l231_231607


namespace sum_of_factors_36_l231_231638

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := by
  sorry

end sum_of_factors_36_l231_231638


namespace square_area_divided_into_equal_rectangles_l231_231097

theorem square_area_divided_into_equal_rectangles (w : ℝ) (a : ℝ) (h : 5 = w) :
  (∃ s : ℝ, s * s = a ∧ s * s / 5 = a / 5) ↔ a = 400 :=
by
  sorry

end square_area_divided_into_equal_rectangles_l231_231097


namespace ratio_37m48s_2h13m15s_l231_231960

-- Define the total seconds for 37 minutes and 48 seconds
def t1 := 37 * 60 + 48

-- Define the total seconds for 2 hours, 13 minutes, and 15 seconds
def t2 := 2 * 3600 + 13 * 60 + 15

-- Prove the ratio t1 / t2 = 2268 / 7995
theorem ratio_37m48s_2h13m15s : t1 / t2 = 2268 / 7995 := 
by sorry

end ratio_37m48s_2h13m15s_l231_231960


namespace problem_solution_l231_231040

theorem problem_solution (x : ℝ) : (3 / (x - 3) = 4 / (x - 4)) ↔ (x = 0) := 
by
  sorry

end problem_solution_l231_231040


namespace speed_of_man_proof_l231_231968

noncomputable def speed_of_man (train_length : ℝ) (crossing_time : ℝ) (train_speed_kph : ℝ) : ℝ :=
  let train_speed_mps := (train_speed_kph * 1000) / 3600
  let relative_speed := train_length / crossing_time
  train_speed_mps - relative_speed

theorem speed_of_man_proof 
  (train_length : ℝ := 600) 
  (crossing_time : ℝ := 35.99712023038157) 
  (train_speed_kph : ℝ := 64) :
  speed_of_man train_length crossing_time train_speed_kph = 1.10977777777778 :=
by
  -- Proof goes here
  sorry

end speed_of_man_proof_l231_231968


namespace rationalize_denominator_l231_231931

-- Definitions based on given conditions
def numerator : ℝ := 45
def denominator : ℝ := Real.sqrt 45
def original_expression : ℝ := numerator / denominator

-- The goal is proving that the original expression equals to the simplified form
theorem rationalize_denominator :
  original_expression = 3 * Real.sqrt 5 :=
by
  -- Place the incomplete proof here, skipped with sorry
  sorry

end rationalize_denominator_l231_231931


namespace negation_of_forall_l231_231373

theorem negation_of_forall (h : ¬ ∀ x > 0, Real.exp x > x + 1) : ∃ x > 0, Real.exp x < x + 1 :=
sorry

end negation_of_forall_l231_231373


namespace part1_part2_l231_231592

theorem part1 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a + 2 * c = b * Real.cos C + Real.sqrt 3 * b * Real.sin C) : 
  B = 2 * Real.pi / 3 := 
sorry

theorem part2 
  (a b c : ℝ) 
  (A C : ℝ) 
  (h1 : a + 2 * c = b * Real.cos C + Real.sqrt 3 * b * Real.sin C)
  (h2 : b = 3) : 
  6 < (a + b + c) ∧ (a + b + c) ≤ 3 + 2 * Real.sqrt 3 :=
sorry

end part1_part2_l231_231592


namespace solution_set_of_fx_eq_zero_l231_231717

noncomputable def f (x : ℝ) : ℝ :=
if hx : x = 0 then 0 else if 0 < x then Real.log x / Real.log 2 else - (Real.log (-x) / Real.log 2)

lemma f_is_odd : ∀ x : ℝ, f (-x) = - f x :=
by sorry

lemma f_is_log_for_positive : ∀ x : ℝ, 0 < x → f x = Real.log x / Real.log 2 :=
by sorry

theorem solution_set_of_fx_eq_zero :
  {x : ℝ | f x = 0} = {-1, 0, 1} :=
by sorry

end solution_set_of_fx_eq_zero_l231_231717


namespace triangle_sum_correct_l231_231615

def triangle_op (a b c : ℕ) : ℕ :=
  a * b / c

theorem triangle_sum_correct :
  triangle_op 4 8 2 + triangle_op 5 10 5 = 26 :=
by
  sorry

end triangle_sum_correct_l231_231615


namespace fraction_product_l231_231420

theorem fraction_product :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := 
by
  -- Detailed proof steps would go here
  sorry

end fraction_product_l231_231420


namespace remaining_download_time_l231_231769

-- Define the relevant quantities
def total_size : ℝ := 1250
def downloaded : ℝ := 310
def download_speed : ℝ := 2.5

-- State the theorem
theorem remaining_download_time : (total_size - downloaded) / download_speed = 376 := by
  -- Proof will be filled in here
  sorry

end remaining_download_time_l231_231769


namespace ophelia_average_pay_l231_231929

theorem ophelia_average_pay : ∀ (n : ℕ), 
  (51 + 100 * (n - 1)) / n = 93 ↔ n = 7 :=
by
  sorry

end ophelia_average_pay_l231_231929


namespace exists_integer_K_l231_231282

theorem exists_integer_K (Z : ℕ) (K : ℕ) : 
  1000 < Z ∧ Z < 2000 ∧ Z = K^4 → 
  ∃ K, K = 6 := 
by
  sorry

end exists_integer_K_l231_231282


namespace candy_bars_weeks_l231_231761

theorem candy_bars_weeks (buy_per_week : ℕ) (eat_per_4_weeks : ℕ) (saved_candies : ℕ) (weeks_passed : ℕ) :
  (buy_per_week = 2) →
  (eat_per_4_weeks = 1) →
  (saved_candies = 28) →
  (weeks_passed = 4 * (saved_candies / (4 * buy_per_week - eat_per_4_weeks))) →
  weeks_passed = 16 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end candy_bars_weeks_l231_231761


namespace remaining_slices_correct_l231_231932

def pies : Nat := 2
def slices_per_pie : Nat := 8
def slices_total : Nat := pies * slices_per_pie
def slices_rebecca_initial : Nat := 1 * pies
def slices_remaining_after_rebecca : Nat := slices_total - slices_rebecca_initial
def slices_family_friends : Nat := 7
def slices_remaining_after_family_friends : Nat := slices_remaining_after_rebecca - slices_family_friends
def slices_rebecca_husband_last : Nat := 2
def slices_remaining : Nat := slices_remaining_after_family_friends - slices_rebecca_husband_last

theorem remaining_slices_correct : slices_remaining = 5 := 
by sorry

end remaining_slices_correct_l231_231932


namespace necessary_condition_for_ellipse_l231_231372

theorem necessary_condition_for_ellipse (m : ℝ) : 
  (5 - m > 0) → (m + 3 > 0) → (5 - m ≠ m + 3) → (-3 < m ∧ m < 5 ∧ m ≠ 1) :=
by sorry

end necessary_condition_for_ellipse_l231_231372


namespace point_B_coordinates_l231_231306

-- Defining the vector a
def vec_a : ℝ × ℝ := (1, 0)

-- Defining the point A
def A : ℝ × ℝ := (4, 4)

-- Definition of the line y = 2x
def on_line (P : ℝ × ℝ) : Prop := P.2 = 2 * P.1

-- Defining a vector as being parallel to another vector
def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

-- Lean statement for the proof
theorem point_B_coordinates (B : ℝ × ℝ) (h1 : on_line B) (h2 : parallel (B.1 - 4, B.2 - 4) vec_a) :
  B = (2, 4) :=
sorry

end point_B_coordinates_l231_231306


namespace tv_price_change_l231_231388

theorem tv_price_change (P : ℝ) :
  let decrease := 0.20
  let increase := 0.45
  let new_price := P * (1 - decrease)
  let final_price := new_price * (1 + increase)
  final_price - P = 0.16 * P := 
by
  sorry

end tv_price_change_l231_231388


namespace math_study_time_l231_231854

-- Conditions
def science_time : ℕ := 25
def total_time : ℕ := 60

-- Theorem statement
theorem math_study_time :
  total_time - science_time = 35 := by
  -- Proof placeholder
  sorry

end math_study_time_l231_231854


namespace students_neither_l231_231364

-- Define the given conditions
def total_students : Nat := 460
def football_players : Nat := 325
def cricket_players : Nat := 175
def both_players : Nat := 90

-- Define the Lean statement for the proof problem
theorem students_neither (total_students football_players cricket_players both_players : Nat) (h1 : total_students = 460)
  (h2 : football_players = 325) (h3 : cricket_players = 175) (h4 : both_players = 90) :
  total_students - (football_players + cricket_players - both_players) = 50 := by
  sorry

end students_neither_l231_231364


namespace domain_ln_x_minus_1_l231_231942

def domain_of_log_function (x : ℝ) : Prop := x > 1

theorem domain_ln_x_minus_1 (x : ℝ) : domain_of_log_function x ↔ x > 1 :=
by {
  sorry
}

end domain_ln_x_minus_1_l231_231942


namespace sum_of_ages_is_22_l231_231934

noncomputable def Ashley_Age := 8
def Mary_Age (M : ℕ) := 7 * Ashley_Age = 4 * M

theorem sum_of_ages_is_22 (M : ℕ) (h : Mary_Age M):
  Ashley_Age + M = 22 :=
by
  -- skipping proof details
  sorry

end sum_of_ages_is_22_l231_231934


namespace least_possible_value_of_a_plus_b_l231_231349

theorem least_possible_value_of_a_plus_b : ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧
  Nat.gcd (a + b) 330 = 1 ∧
  b ∣ a^a ∧ 
  ∀ k : ℕ, b^3 ∣ a^a → (k ∣ a → k = 1) ∧
  a + b = 392 :=
by
  sorry

end least_possible_value_of_a_plus_b_l231_231349


namespace coat_price_reduction_l231_231086

noncomputable def reduced_price_with_tax (price : ℝ) (reduction : ℝ) (tax : ℝ) : ℝ :=
  let reduced_price := price * (1 - reduction)
  let taxed_price := reduced_price * (1 + tax)
  taxed_price

theorem coat_price_reduction : 
  let initial_price : ℝ := 500
  let first_month_reduction : ℝ := 0.1
  let first_month_tax : ℝ := 0.05
  let second_month_reduction : ℝ := 0.15
  let second_month_tax : ℝ := 0.08
  let third_month_reduction : ℝ := 0.2
  let third_month_tax : ℝ := 0.06
  let price_after_first_month := reduced_price_with_tax initial_price first_month_reduction first_month_tax
  let price_after_second_month := reduced_price_with_tax price_after_first_month second_month_reduction second_month_tax
  let price_after_third_month := reduced_price_with_tax price_after_second_month third_month_reduction third_month_tax
  let total_percent_reduction := (initial_price - price_after_third_month) / initial_price * 100
  price_after_third_month ≈ 367.824 ∧ total_percent_reduction ≈ 26.44 := 
  by
    sorry

end coat_price_reduction_l231_231086


namespace solve_for_x_l231_231427

variable {R : Type*} [Field R]

def matrix_3x3 (x : R) : Matrix (Fin 3) (Fin 3) R :=
  !![3,  1, -1;
     4,  x,  2;
     1,  3,  6]

theorem solve_for_x (x : R) :
  matrix_3x3 x.det = 0 → x = 52 / 19 := by
  sorry

end solve_for_x_l231_231427


namespace seq_50th_term_eq_327_l231_231122

theorem seq_50th_term_eq_327 : 
  let n := 50
  let binary_representation : List Nat := [1, 1, 0, 0, 1, 0] -- 50 in binary
  let powers_of_3 := [5, 4, 1] -- Positions of 1s in the binary representation 
  let term := List.sum (powers_of_3.map (λ k => 3^k))
  term = 327 := by
  sorry

end seq_50th_term_eq_327_l231_231122


namespace average_percent_decrease_is_35_percent_l231_231763

-- Given conditions
def last_week_small_price_per_pack := 7 / 3
def this_week_small_price_per_pack := 5 / 4
def last_week_large_price_per_pack := 8 / 2
def this_week_large_price_per_pack := 9 / 3

-- Calculate percent decrease for small packs
def small_pack_percent_decrease := ((last_week_small_price_per_pack - this_week_small_price_per_pack) / last_week_small_price_per_pack) * 100

-- Calculate percent decrease for large packs
def large_pack_percent_decrease := ((last_week_large_price_per_pack - this_week_large_price_per_pack) / last_week_large_price_per_pack) * 100

-- Calculate average percent decrease
def average_percent_decrease := (small_pack_percent_decrease + large_pack_percent_decrease) / 2

theorem average_percent_decrease_is_35_percent : average_percent_decrease = 35 := by
  sorry

end average_percent_decrease_is_35_percent_l231_231763


namespace cash_after_brokerage_l231_231039

theorem cash_after_brokerage (sale_amount : ℝ) (brokerage_rate : ℝ) :
  sale_amount = 109.25 → brokerage_rate = 0.0025 →
  (sale_amount - sale_amount * brokerage_rate) = 108.98 :=
by
  intros h1 h2
  sorry

end cash_after_brokerage_l231_231039


namespace probability_all_black_l231_231248

open ProbabilityTheory

-- Conditions
def probability (p : ℝ) : ℝ := if 0 ≤ p ∧ p ≤ 1 then p else 0

def initial_state (x : ℕ × ℕ) : ℝ :=
  let black := probability (1 / 3)
  let white := probability (1 / 3)
  let red := probability (1 / 3)
  if x.1 < 4 ∧ x.2 < 4 then black + white + red else 0

def rotated_state (x : ℕ × ℕ) : ℝ :=
  if x.1 < 4 ∧ x.2 < 4 then
    let pos := (x.2, 3 - x.1)
    let is_red := (initial_state pos = probability (1 / 3)) -- red
    let is_black := (initial_state x = probability (1 / 3)) -- black
    if is_red ∧ is_black then probability (1 / 3) -- black
    else initial_state x
  else 0

def final_grid_black (x : ℕ × ℕ) : ℝ :=
  if x.1 < 4 ∧ x.2 < 4 then
    let pos := (x.2, 3 - x.1)
    let res := initial_state pos + (probability (1 / 9)) -- considering the rotation affecting black
    res
  else 0

-- Goal
theorem probability_all_black : 
  (Π x : ℕ × ℕ, x.1 < 4 ∧ x.2 < 4 → final_grid_black x = probability (4 / 9)) 
  → (probability (4 / 9) ^ 16) = (4 / 9)^16 :=
by sorry

end probability_all_black_l231_231248


namespace percentage_error_in_area_l231_231670

theorem percentage_error_in_area (s : ℝ) (h_s_pos: s > 0) :
  let measured_side := 1.01 * s
  let actual_area := s ^ 2
  let measured_area := measured_side ^ 2
  let error_in_area := measured_area - actual_area
  (error_in_area / actual_area) * 100 = 2.01 :=
by
  sorry

end percentage_error_in_area_l231_231670


namespace part1_part2_l231_231004

theorem part1 (x y : ℝ) (h1 : (1, 0) = (x, y)) (h2 : (0, 2) = (x, y)): 
    ∃ k b : ℝ, k = -2 ∧ b = 2 ∧ y = k * x + b := 
by 
  sorry

theorem part2 (m n : ℝ) (h : n = -2 * m + 2) (hm : -2 < m ∧ m ≤ 3):
    -4 ≤ n ∧ n < 6 := 
by 
  sorry

end part1_part2_l231_231004


namespace total_questions_l231_231928

theorem total_questions (qmc : ℕ) (qtotal : ℕ) (h1 : 10 = qmc) (h2 : qmc = (20 / 100) * qtotal) : qtotal = 50 :=
sorry

end total_questions_l231_231928


namespace max_tickets_jane_can_buy_l231_231859

-- Define ticket prices and Jane's budget
def ticket_price := 15
def discounted_price := 12
def discount_threshold := 5
def jane_budget := 150

-- Prove that the maximum number of tickets Jane can buy is 11
theorem max_tickets_jane_can_buy : 
  ∃ (n : ℕ), n ≤ 11 ∧ (if n ≤ discount_threshold then ticket_price * n ≤ jane_budget else (ticket_price * discount_threshold + discounted_price * (n - discount_threshold)) ≤ jane_budget)
  ∧ ∀ m : ℕ, (if m ≤ 11 then (if m ≤ discount_threshold then ticket_price * m ≤ jane_budget else (ticket_price * discount_threshold + discounted_price * (m - discount_threshold)) ≤ jane_budget) else false)  → m ≤ 11 := 
by
  sorry

end max_tickets_jane_can_buy_l231_231859


namespace number_of_tables_l231_231927

-- Defining the given parameters
def linen_cost : ℕ := 25
def place_setting_cost : ℕ := 10
def rose_cost : ℕ := 5
def lily_cost : ℕ := 4
def num_place_settings : ℕ := 4
def num_roses : ℕ := 10
def num_lilies : ℕ := 15
def total_decoration_cost : ℕ := 3500

-- Defining the cost per table
def cost_per_table : ℕ := linen_cost + (num_place_settings * place_setting_cost) + (num_roses * rose_cost) + (num_lilies * lily_cost)

-- Proof problem statement: Proving number of tables is 20
theorem number_of_tables : (total_decoration_cost / cost_per_table) = 20 :=
by
  sorry

end number_of_tables_l231_231927


namespace yogurt_combinations_l231_231405

-- Define the conditions from a)
def num_flavors : ℕ := 5
def num_toppings : ℕ := 8
def num_sizes : ℕ := 3

-- Define the problem in a theorem statement
theorem yogurt_combinations : num_flavors * ((num_toppings * (num_toppings - 1)) / 2) * num_sizes = 420 :=
by
  -- sorry is used here to skip the proof
  sorry

end yogurt_combinations_l231_231405


namespace g_neg6_eq_neg28_l231_231351

-- Define the given function g
def g (x : ℝ) : ℝ := 2 * x^7 - 3 * x^3 + 4 * x - 8

-- State the main theorem to prove g(-6) = -28 under the given conditions
theorem g_neg6_eq_neg28 (h1 : g 6 = 12) : g (-6) = -28 :=
by
  sorry

end g_neg6_eq_neg28_l231_231351


namespace solution_part_1_solution_part_2_l231_231831

def cost_price_of_badges (x y : ℕ) : Prop :=
  (x - y = 4) ∧ (6 * x = 10 * y)

theorem solution_part_1 (x y : ℕ) :
  cost_price_of_badges x y → x = 10 ∧ y = 6 :=
by
  sorry

def maximizing_profit (m : ℕ) (w : ℕ) : Prop :=
  (10 * m + 6 * (400 - m) ≤ 2800) ∧ (w = m + 800)

theorem solution_part_2 (m : ℕ) :
  maximizing_profit m 900 → m = 100 :=
by
  sorry


end solution_part_1_solution_part_2_l231_231831


namespace shaded_area_of_modified_design_l231_231326

noncomputable def radius_of_circles (side_length : ℝ) (grid_size : ℕ) : ℝ :=
  (side_length / grid_size) / 2

noncomputable def area_of_circle (radius : ℝ) : ℝ :=
  Real.pi * radius^2

noncomputable def area_of_square (side_length : ℝ) : ℝ :=
  side_length^2

noncomputable def shaded_area (side_length : ℝ) (grid_size : ℕ) : ℝ :=
  let r := radius_of_circles side_length grid_size
  let total_circle_area := 9 * area_of_circle r
  area_of_square side_length - total_circle_area

theorem shaded_area_of_modified_design :
  shaded_area 24 3 = (576 - 144 * Real.pi) :=
by
  sorry

end shaded_area_of_modified_design_l231_231326


namespace DogHeight_is_24_l231_231111

-- Define the given conditions as Lean definitions (variables and equations)
variable (CarterHeight DogHeight BettyHeight : ℝ)

-- Assume the conditions given in the problem
axiom h1 : CarterHeight = 2 * DogHeight
axiom h2 : BettyHeight + 12 = CarterHeight
axiom h3 : BettyHeight = 36

-- State the proposition (the height of Carter's dog)
theorem DogHeight_is_24 : DogHeight = 24 :=
by
  -- Proof goes here
  sorry

end DogHeight_is_24_l231_231111


namespace intersection_of_sets_l231_231876
-- Define the sets and the proof statement
theorem intersection_of_sets : 
  let A := { x : ℝ | x^2 - 3 * x - 4 < 0 }
  let B := {-4, 1, 3, 5}
  A ∩ B = {1, 3} :=
by
  sorry

end intersection_of_sets_l231_231876


namespace perpendicular_vectors_m_value_l231_231153

theorem perpendicular_vectors_m_value
  (a : ℝ × ℝ := (1, 2))
  (b : ℝ × ℝ)
  (h_perpendicular : (a.1 * b.1 + a.2 * b.2) = 0) :
  b = (-2, 1) :=
by
  sorry

end perpendicular_vectors_m_value_l231_231153


namespace infinite_series_sum_l231_231520

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l231_231520


namespace gaussian_solutions_count_l231_231788

noncomputable def solve_gaussian (x : ℝ) : ℕ :=
  if h : x^2 = 2 * (⌊x⌋ : ℝ) + 1 then 
    1 
  else
    0

theorem gaussian_solutions_count :
  ∀ x : ℝ, solve_gaussian x = 2 :=
sorry

end gaussian_solutions_count_l231_231788


namespace sum_of_integers_l231_231577

theorem sum_of_integers : (∀ (x y : ℤ), x = -4 ∧ y = -5 ∧ x - y = 1 → x + y = -9) := 
by 
  intros x y
  sorry

end sum_of_integers_l231_231577


namespace intersection_M_N_l231_231747

def M : Set ℝ := {x : ℝ | -4 < x ∧ x < 4}
def N : Set ℝ := {x : ℝ | x ≥ -1 / 3}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 / 3 ≤ x ∧ x < 4} :=
sorry

end intersection_M_N_l231_231747


namespace total_birds_l231_231048

theorem total_birds (g d : Nat) (h₁ : g = 58) (h₂ : d = 37) : g + d = 95 :=
by
  sorry

end total_birds_l231_231048


namespace find_first_number_l231_231257

variable (a : ℕ → ℤ)

axiom recurrence_rel : ∀ (n : ℕ), n ≥ 4 → a n = a (n - 1) + a (n - 2) + a (n - 3)
axiom a8_val : a 8 = 29
axiom a9_val : a 9 = 56
axiom a10_val : a 10 = 108

theorem find_first_number : a 1 = 32 :=
sorry

end find_first_number_l231_231257


namespace P_not_77_for_all_integers_l231_231777

def P (x y : ℤ) : ℤ := x^5 - 4 * x^4 * y - 5 * y^2 * x^3 + 20 * y^3 * x^2 + 4 * y^4 * x - 16 * y^5

theorem P_not_77_for_all_integers (x y : ℤ) : P x y ≠ 77 :=
sorry

end P_not_77_for_all_integers_l231_231777


namespace combine_material_points_l231_231900

variables {K K₁ K₂ : Type} {m m₁ m₂ : ℝ}

-- Assume some properties and operations for type K
noncomputable def add_material_points (K₁ K₂ : K × ℝ) : K × ℝ :=
(K₁.1, K₁.2 + K₂.2)

theorem combine_material_points (K₁ K₂ : K × ℝ) :
  (add_material_points K₁ K₂) = (K₁.1, K₁.2 + K₂.2) :=
sorry

end combine_material_points_l231_231900


namespace fewer_onions_grown_l231_231268

def num_tomatoes := 2073
def num_cobs_of_corn := 4112
def num_onions := 985

theorem fewer_onions_grown : num_tomatoes + num_cobs_of_corn - num_onions = 5200 := by
  sorry

end fewer_onions_grown_l231_231268


namespace P_ne_77_for_integers_l231_231780

def P (x y : ℤ) : ℤ :=
  x^5 - 4 * x^4 * y - 5 * y^2 * x^3 + 20 * y^3 * x^2 + 4 * y^4 * x - 16 * y^5

theorem P_ne_77_for_integers (x y : ℤ) : P x y ≠ 77 :=
by
  sorry

end P_ne_77_for_integers_l231_231780


namespace inequality_reciprocal_l231_231295

theorem inequality_reciprocal (a b : ℝ) (h₀ : a < b) (h₁ : b < 0) : (1 / a) > (1 / b) :=
sorry

end inequality_reciprocal_l231_231295


namespace stratified_sampling_l231_231256

-- Definitions
def total_staff : ℕ := 150
def senior_titles : ℕ := 45
def intermediate_titles : ℕ := 90
def clerks : ℕ := 15
def sample_size : ℕ := 10

-- Ratios for stratified sampling
def senior_sample : ℕ := (senior_titles * sample_size) / total_staff
def intermediate_sample : ℕ := (intermediate_titles * sample_size) / total_staff
def clerks_sample : ℕ := (clerks * sample_size) / total_staff

-- Theorem statement
theorem stratified_sampling :
  senior_sample = 3 ∧ intermediate_sample = 6 ∧ clerks_sample = 1 :=
by
  sorry

end stratified_sampling_l231_231256


namespace sum_series_eq_4_div_9_l231_231458

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l231_231458


namespace maximize_probability_l231_231810

def numbers_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def pairs_summing_to_12 (l : List Int) : List (Int × Int) :=
  List.filter (fun (p : Int × Int) => p.1 + p.2 = 12) (List.product l l)

def distinct_pairs (pairs : List (Int × Int)) : List (Int × Int) :=
  List.filter (fun (p : Int × Int) => p.1 ≠ p.2) pairs

def valid_pairs (l : List Int) : List (Int × Int) :=
  distinct_pairs (pairs_summing_to_12 l)

def count_valid_pairs (l : List Int) : Nat :=
  List.length (valid_pairs l)

def remove_and_check (x : Int) : List Int :=
  List.erase numbers_list x

theorem maximize_probability :
  ∀ x : Int, count_valid_pairs (remove_and_check 6) ≥ count_valid_pairs (remove_and_check x) :=
sorry

end maximize_probability_l231_231810


namespace prove_ab_leq_one_l231_231370

theorem prove_ab_leq_one (a b : ℝ) (h : (a + b + a) * (a + b + b) = 9) : ab ≤ 1 := 
by
  sorry

end prove_ab_leq_one_l231_231370


namespace find_a2018_l231_231138

-- Given Conditions
def initial_condition (a : ℕ → ℤ) : Prop :=
  a 1 = -1

def absolute_difference (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → abs (a n - a (n-1)) = 2^(n-1)

def subseq_decreasing (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (2*n-1) > a (2*(n+1)-1)

def subseq_increasing (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (2*n) < a (2*(n+1))

-- Theorem to Prove
theorem find_a2018 (a : ℕ → ℤ)
  (h1 : initial_condition a)
  (h2 : absolute_difference a)
  (h3 : subseq_decreasing a)
  (h4 : subseq_increasing a) :
  a 2018 = (2^2018 - 1) / 3 :=
sorry

end find_a2018_l231_231138


namespace unique_pair_odd_prime_l231_231912

theorem unique_pair_odd_prime (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃! (n m : ℕ), n ≠ m ∧ (2 / p : ℚ) = (1 / n) + (1 / m) ∧ 
  n = (p + 1) / 2 ∧ m = (p * (p + 1)) / 2 :=
by
  sorry

end unique_pair_odd_prime_l231_231912


namespace min_buses_needed_l231_231682

theorem min_buses_needed (students : ℕ) (cap1 cap2 : ℕ) (h_students : students = 530) (h_cap1 : cap1 = 40) (h_cap2 : cap2 = 45) :
  min (Nat.ceil (students / cap1)) (Nat.ceil (students / cap2)) = 12 :=
  sorry

end min_buses_needed_l231_231682


namespace economical_speed_l231_231789

variable (a k : ℝ)
variable (ha : 0 < a) (hk : 0 < k)

theorem economical_speed (v : ℝ) : 
  v = (a / (2 * k))^(1/3) :=
sorry

end economical_speed_l231_231789


namespace solve_equation1_solve_equation2_l231_231786

-- Define the first equation (x-3)^2 + 2x(x-3) = 0
def equation1 (x : ℝ) : Prop := (x - 3)^2 + 2 * x * (x - 3) = 0

-- Define the second equation x^2 - 4x + 1 = 0
def equation2 (x : ℝ) : Prop := x^2 - 4 * x + 1 = 0

-- Theorem stating the solutions for the first equation
theorem solve_equation1 : ∀ (x : ℝ), equation1 x ↔ x = 3 ∨ x = 1 :=
by
  intro x
  sorry  -- Proof is omitted

-- Theorem stating the solutions for the second equation
theorem solve_equation2 : ∀ (x : ℝ), equation2 x ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 :=
by
  intro x
  sorry  -- Proof is omitted

end solve_equation1_solve_equation2_l231_231786


namespace find_cost_price_l231_231821

theorem find_cost_price (SP PP : ℝ) (hSP : SP = 600) (hPP : PP = 25) : 
  ∃ CP : ℝ, CP = 480 := 
by
  sorry

end find_cost_price_l231_231821


namespace find_m_l231_231713

open Set

def U : Set ℕ := {0, 1, 2, 3}
def A (m : ℤ) : Set ℕ := {x ∈ U | x^2 + m * x = 0}
def complement_A (m : ℤ) : Set ℕ := {1, 2}

theorem find_m (m : ℤ) (hA : complement_A m = U \ A m) : m = -3 :=
by
  sorry

end find_m_l231_231713


namespace sum_of_factors_36_l231_231650

theorem sum_of_factors_36 : (∑ i in {d ∣ 36 | i}, i) = 91 := by
  sorry

end sum_of_factors_36_l231_231650


namespace initially_calculated_average_weight_l231_231221

-- Define the conditions
def num_boys : ℕ := 20
def correct_average_weight : ℝ := 58.7
def misread_weight : ℝ := 56
def correct_weight : ℝ := 62
def weight_difference : ℝ := correct_weight - misread_weight

-- State the goal
theorem initially_calculated_average_weight :
  let correct_total_weight := correct_average_weight * num_boys
  let initial_total_weight := correct_total_weight - weight_difference
  let initially_calculated_weight := initial_total_weight / num_boys
  initially_calculated_weight = 58.4 :=
by
  sorry

end initially_calculated_average_weight_l231_231221


namespace sum_of_series_l231_231462

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l231_231462


namespace sum_series_equals_4_div_9_l231_231479

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l231_231479


namespace ratio_pen_pencil_l231_231358

theorem ratio_pen_pencil (P : ℝ) (pencil_cost total_cost : ℝ) 
  (hc1 : pencil_cost = 8) 
  (hc2 : total_cost = 12)
  (hc3 : P + pencil_cost = total_cost) : 
  P / pencil_cost = 1 / 2 :=
by 
  sorry

end ratio_pen_pencil_l231_231358


namespace push_mower_cuts_one_acre_per_hour_l231_231756

noncomputable def acres_per_hour_push_mower : ℕ :=
  let total_acres := 8
  let fraction_riding := 3 / 4
  let riding_mower_rate := 2
  let mowing_hours := 5
  let acres_riding := fraction_riding * total_acres
  let time_riding_mower := acres_riding / riding_mower_rate
  let remaining_hours := mowing_hours - time_riding_mower
  let remaining_acres := total_acres - acres_riding
  remaining_acres / remaining_hours

theorem push_mower_cuts_one_acre_per_hour :
  acres_per_hour_push_mower = 1 := 
by 
  -- Detailed proof steps would go here.
  sorry

end push_mower_cuts_one_acre_per_hour_l231_231756


namespace fill_table_with_numbers_l231_231366

-- Define the main theorem based on the conditions and question.
theorem fill_table_with_numbers (numbers : Finset ℤ) (table : ℕ → ℕ → ℤ)
  (h_numbers_card : numbers.card = 100)
  (h_sum_1x3_horizontal : ∀ i j, (table i j + table i (j + 1) + table i (j + 2) ∈ numbers))
  (h_sum_1x3_vertical : ∀ i j, (table i j + table (i + 1) j + table (i + 2) j ∈ numbers)):
  ∃ (t : ℕ → ℕ → ℤ), (∀ k, 1 ≤ k ∧ k ≤ 6 → ∃ i j, t i j = k) :=
sorry

end fill_table_with_numbers_l231_231366


namespace bob_questions_created_l231_231691

theorem bob_questions_created :
  let q1 := 13
  let q2 := 2 * q1
  let q3 := 2 * q2
  q1 + q2 + q3 = 91 :=
by
  sorry

end bob_questions_created_l231_231691


namespace unique_pair_fraction_l231_231923

theorem unique_pair_fraction (p : ℕ) (hprime : Nat.Prime p) (hodd : p % 2 = 1) :
  ∃! (n m : ℕ), (n ≠ m) ∧ (2 / (p : ℚ) = 1 / (n : ℚ) + 1 / (m : ℚ)) ∧
  ((n = (p + 1) / 2 ∧ m = p * (p + 1) / 2) ∨ (m = (p + 1) / 2 ∧ n = p * (p + 1) / 2)) := sorry

end unique_pair_fraction_l231_231923


namespace complex_sum_l231_231192

-- Define the given condition as a hypothesis
variables {z : ℂ} (h : z^2 + z + 1 = 0)

-- Define the statement to prove
theorem complex_sum (h : z^2 + z + 1 = 0) : z^96 + z^97 + z^98 + z^99 + z^100 + z^101 = 0 :=
sorry

end complex_sum_l231_231192


namespace correct_sentence_is_D_l231_231239

-- Define the sentences as strings
def sentence_A : String :=
  "Between any two adjacent integers on the number line, an infinite number of fractions can be inserted to fill the gaps on the number line; mathematicians once thought that with this approach, the entire number line was finally filled."

def sentence_B : String :=
  "With zero as the center, all integers are arranged from right to left at equal distances, and then connected with a horizontal line; this is what we call the 'number line'."

def sentence_C : String :=
  "The vast collection of books in the Beijing Library contains an enormous amount of information, but it is still finite, whereas the number pi contains infinite information, which is awe-inspiring."

def sentence_D : String :=
  "Pi is fundamentally the exact ratio of a circle's circumference to its diameter, but the infinite sequence it produces has the greatest uncertainty; we cannot help but be amazed and shaken by the marvel and mystery of nature."

-- Define the problem statement
theorem correct_sentence_is_D :
  sentence_D ≠ "" := by
  sorry

end correct_sentence_is_D_l231_231239


namespace specified_time_is_30_total_constuction_cost_is_180000_l231_231250

noncomputable def specified_time (x : ℕ) :=
  let teamA_rate := 1 / (x:ℝ)
  let teamB_rate := 2 / (3 * (x:ℝ))
  (teamA_rate + teamB_rate) * 15 + 5 * teamA_rate = 1

theorem specified_time_is_30 : specified_time 30 :=
  by 
    sorry

noncomputable def total_constuction_cost (x : ℕ) (costA : ℕ) (costB : ℕ) :=
  let teamA_rate := 1 / (x:ℝ)
  let teamB_rate := 2 / (3 * (x:ℝ))
  let total_time := 1 / (teamA_rate + teamB_rate)
  total_time * (costA + costB)

theorem total_constuction_cost_is_180000 : total_constuction_cost 30 6500 3500 = 180000 :=
  by 
    sorry

end specified_time_is_30_total_constuction_cost_is_180000_l231_231250


namespace average_speed_of_train_l231_231404

theorem average_speed_of_train (d1 d2: ℝ) (t1 t2: ℝ) (h_d1: d1 = 250) (h_d2: d2 = 350) (h_t1: t1 = 2) (h_t2: t2 = 4) :
  (d1 + d2) / (t1 + t2) = 100 := by
  sorry

end average_speed_of_train_l231_231404


namespace unique_pair_odd_prime_l231_231914

theorem unique_pair_odd_prime (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃! (n m : ℕ), n ≠ m ∧ (2 / p : ℚ) = (1 / n) + (1 / m) ∧ 
  n = (p + 1) / 2 ∧ m = (p * (p + 1)) / 2 :=
by
  sorry

end unique_pair_odd_prime_l231_231914


namespace labor_budget_constraint_l231_231237

-- Define the conditions
def wage_per_carpenter : ℕ := 50
def wage_per_mason : ℕ := 40
def labor_budget : ℕ := 2000
def num_carpenters (x : ℕ) := x
def num_masons (y : ℕ) := y

-- The proof statement
theorem labor_budget_constraint (x y : ℕ) 
    (hx : wage_per_carpenter * num_carpenters x + wage_per_mason * num_masons y ≤ labor_budget) : 
    5 * x + 4 * y ≤ 200 := 
by sorry

end labor_budget_constraint_l231_231237


namespace intersection_of_A_and_B_l231_231872

def setA : Set ℝ := { x | x^2 - 3 * x - 4 < 0 }
def setB : Set ℝ := {-4, 1, 3, 5}
def resultSet : Set ℝ := {1, 3}

theorem intersection_of_A_and_B :
  setA ∩ setB = resultSet := 
by
  sorry

end intersection_of_A_and_B_l231_231872


namespace four_digit_sum_10_divisible_by_9_is_0_l231_231310

theorem four_digit_sum_10_divisible_by_9_is_0 : 
  ∀ (N : ℕ), (1000 * ((N / 1000) % 10) + 100 * ((N / 100) % 10) + 10 * ((N / 10) % 10) + (N % 10) = 10) ∧ (N % 9 = 0) → false :=
by
  sorry

end four_digit_sum_10_divisible_by_9_is_0_l231_231310


namespace calculate_radius_l231_231974

noncomputable def radius_of_wheel (D : ℝ) (N : ℕ) (π : ℝ) : ℝ :=
  D / (2 * π * N)

theorem calculate_radius : 
  radius_of_wheel 4224 3000 Real.pi = 0.224 :=
by
  sorry

end calculate_radius_l231_231974


namespace fewer_onions_grown_l231_231270

def num_tomatoes := 2073
def num_cobs_of_corn := 4112
def num_onions := 985

theorem fewer_onions_grown : num_tomatoes + num_cobs_of_corn - num_onions = 5200 := by
  sorry

end fewer_onions_grown_l231_231270


namespace sum_of_factors_36_l231_231640

def is_factor (n d : ℕ) : Prop := d ∣ n

def sum_of_factors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d => is_factor n d).sum

theorem sum_of_factors_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_of_factors_36_l231_231640


namespace tan_add_pi_over_4_l231_231864

variable {α : ℝ}

theorem tan_add_pi_over_4 (h : Real.tan (α - Real.pi / 4) = 1 / 4) : Real.tan (α + Real.pi / 4) = -4 :=
sorry

end tan_add_pi_over_4_l231_231864


namespace intersection_A_B_l231_231885

-- Define set A and set B based on the conditions
def set_A : Set ℝ := {x : ℝ | x^2 - 3 * x - 4 < 0}
def set_B : Set ℝ := {-4, 1, 3, 5}

-- State the theorem to prove the intersection of A and B
theorem intersection_A_B : set_A ∩ set_B = {1, 3} :=
by sorry

end intersection_A_B_l231_231885


namespace sum_series_eq_4_div_9_l231_231456

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l231_231456


namespace original_set_cardinality_l231_231600

-- Definitions based on conditions
def is_reversed_error (n : ℕ) : Prop :=
  ∃ (A B C : ℕ), 100 * A + 10 * B + C = n ∧ 100 * C + 10 * B + A = n + 198 ∧ C - A = 2

-- The theorem to prove
theorem original_set_cardinality : ∃ n : ℕ, is_reversed_error n ∧ n = 10 := by
  sorry

end original_set_cardinality_l231_231600


namespace polygon_number_of_sides_l231_231399

-- Definitions based on conditions
def sum_of_exterior_angles (n : ℕ) : ℕ := 360
def exterior_angle (angle : ℕ) : ℕ := 30

-- The theorem statement
theorem polygon_number_of_sides (n : ℕ) (angle : ℕ) 
  (h1 : sum_of_exterior_angles n = 360)
  (h2 : exterior_angle angle = 30) : 
  n = 12 := 
by
  sorry

end polygon_number_of_sides_l231_231399


namespace graph_passes_through_quadrants_l231_231224

def linear_function (x : ℝ) : ℝ := -5 * x + 5

theorem graph_passes_through_quadrants :
  (∃ x y : ℝ, linear_function x = y ∧ x > 0 ∧ y > 0) ∧  -- Quadrant I
  (∃ x y : ℝ, linear_function x = y ∧ x < 0 ∧ y > 0) ∧  -- Quadrant II
  (∃ x y : ℝ, linear_function x = y ∧ x > 0 ∧ y < 0)    -- Quadrant IV
  :=
by
  sorry

end graph_passes_through_quadrants_l231_231224


namespace size_of_coffee_cup_l231_231022

-- Define the conditions and the final proof statement
variable (C : ℝ) (h1 : (1/4) * C) (h2 : (1/2) * C) (remaining_after_cold : (1/4) * C - 1 = 2)

theorem size_of_coffee_cup : C = 6 := by
  -- Here the proof would go, but we omit it with sorry
  sorry

end size_of_coffee_cup_l231_231022


namespace fractions_sum_to_decimal_l231_231389

theorem fractions_sum_to_decimal :
  (2 / 10) + (4 / 100) + (6 / 1000) = 0.246 :=
by 
  sorry

end fractions_sum_to_decimal_l231_231389


namespace possible_values_of_t_l231_231016

theorem possible_values_of_t
  (theta : ℝ) 
  (x y t : ℝ) :
  x = Real.cos theta →
  y = Real.sin theta →
  t = (Real.sin theta) ^ 2 + (Real.cos theta) ^ 2 →
  x^2 + y^2 = 1 →
  t = 1 := by
  sorry

end possible_values_of_t_l231_231016


namespace wall_area_160_l231_231255

noncomputable def wall_area (small_tile_area : ℝ) (fraction_small : ℝ) : ℝ :=
  small_tile_area / fraction_small

theorem wall_area_160 (small_tile_area : ℝ) (fraction_small : ℝ) (h1 : small_tile_area = 80) (h2 : fraction_small = 1 / 2) :
  wall_area small_tile_area fraction_small = 160 :=
by
  rw [wall_area, h1, h2]
  norm_num

end wall_area_160_l231_231255


namespace loss_of_50_denoted_as_minus_50_l231_231753

def is_profit (x : Int) : Prop :=
  x > 0

def is_loss (x : Int) : Prop :=
  x < 0

theorem loss_of_50_denoted_as_minus_50 : is_loss (-50) :=
  by
    -- proof steps would go here
    sorry

end loss_of_50_denoted_as_minus_50_l231_231753


namespace george_initial_amount_l231_231291

-- Definitions as per conditions
def cost_of_shirt : ℕ := 24
def cost_of_socks : ℕ := 11
def amount_left : ℕ := 65

-- Goal: Prove that the initial amount of money George had is 100
theorem george_initial_amount : (cost_of_shirt + cost_of_socks + amount_left) = 100 := 
by sorry

end george_initial_amount_l231_231291


namespace triangle_sides_inequality_triangle_sides_equality_condition_l231_231011

theorem triangle_sides_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (triangle_cond : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
sorry

theorem triangle_sides_equality_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (triangle_cond : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c := 
sorry

end triangle_sides_inequality_triangle_sides_equality_condition_l231_231011


namespace green_tea_price_decrease_l231_231222

def percentage_change (old_price new_price : ℚ) : ℚ :=
  ((new_price - old_price) / old_price) * 100

theorem green_tea_price_decrease
  (C : ℚ)
  (h1 : C > 0)
  (july_coffee_price : ℚ := 2 * C)
  (mixture_price : ℚ := 3.45)
  (july_green_tea_price : ℚ := 0.3)
  (old_green_tea_price : ℚ := C)
  (equal_mixture : ℚ := (1.5 * july_green_tea_price) + (1.5 * july_coffee_price)) :
  mixture_price = equal_mixture →
  percentage_change old_green_tea_price july_green_tea_price = -70 :=
by
  sorry

end green_tea_price_decrease_l231_231222


namespace cone_height_of_semicircular_sheet_l231_231064

theorem cone_height_of_semicircular_sheet (R h : ℝ) (h_cond: h = R) : h = R :=
by
  exact h_cond

end cone_height_of_semicircular_sheet_l231_231064


namespace parabola_has_one_x_intercept_l231_231281

-- Define the equation of the parabola
def parabola (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 3

-- State the theorem that proves the number of x-intercepts
theorem parabola_has_one_x_intercept : ∃! x, ∃ y : ℝ, parabola y = x ∧ y = 0 :=
by
  -- Proof goes here, but it's omitted
  sorry

end parabola_has_one_x_intercept_l231_231281


namespace series_sum_eq_four_ninths_l231_231491

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l231_231491


namespace arithmetic_seq_a7_l231_231174

theorem arithmetic_seq_a7 (a : ℕ → ℕ) (h1 : a 2 = 2) (h2 : a 4 + a 5 = 12) : a 7 = 10 :=
by
  sorry

end arithmetic_seq_a7_l231_231174


namespace john_total_distance_l231_231909

theorem john_total_distance :
  let speed := 55 -- John's speed in mph
  let time1 := 2 -- Time before lunch in hours
  let time2 := 3 -- Time after lunch in hours
  let distance1 := speed * time1 -- Distance before lunch
  let distance2 := speed * time2 -- Distance after lunch
  let total_distance := distance1 + distance2 -- Total distance

  total_distance = 275 :=
by
  sorry

end john_total_distance_l231_231909


namespace lisa_total_spoons_l231_231199

def number_of_baby_spoons (num_children num_spoons_per_child : Nat) : Nat :=
  num_children * num_spoons_per_child

def number_of_decorative_spoons : Nat := 2

def number_of_old_spoons (baby_spoons decorative_spoons : Nat) : Nat :=
  baby_spoons + decorative_spoons
  
def number_of_new_spoons (large_spoons teaspoons : Nat) : Nat :=
  large_spoons + teaspoons

def total_number_of_spoons (old_spoons new_spoons : Nat) : Nat :=
  old_spoons + new_spoons

theorem lisa_total_spoons
  (children : Nat)
  (spoons_per_child : Nat)
  (large_spoons : Nat)
  (teaspoons : Nat)
  (children_eq : children = 4)
  (spoons_per_child_eq : spoons_per_child = 3)
  (large_spoons_eq : large_spoons = 10)
  (teaspoons_eq : teaspoons = 15)
  : total_number_of_spoons (number_of_old_spoons (number_of_baby_spoons children spoons_per_child) number_of_decorative_spoons) (number_of_new_spoons large_spoons teaspoons) = 39 :=
by
  sorry

end lisa_total_spoons_l231_231199


namespace value_of_a_l231_231146

theorem value_of_a (a : ℝ) : (-2)^2 + 3*(-2) + a = 0 → a = 2 :=
by {
  sorry
}

end value_of_a_l231_231146


namespace hyperbola_foci_coordinates_l231_231937

theorem hyperbola_foci_coordinates :
  ∀ x y : ℝ, (x^2 / 4) - (y^2 / 12) = 1 → (x, y) = (4, 0) ∨ (x, y) = (-4, 0) :=
by
  -- We assume the given equation of the hyperbola
  intro x y h
  -- sorry is used to skip the actual proof steps
  sorry

end hyperbola_foci_coordinates_l231_231937


namespace fewer_onions_than_tomatoes_and_corn_l231_231265

def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

theorem fewer_onions_than_tomatoes_and_corn :
  (tomatoes + corn - onions) = 5200 :=
by
  sorry

end fewer_onions_than_tomatoes_and_corn_l231_231265


namespace repeating_decimal_to_fraction_l231_231999

theorem repeating_decimal_to_fraction : (2.353535... : Rational) = 233/99 :=
by
  sorry

end repeating_decimal_to_fraction_l231_231999


namespace sum_series_eq_four_ninths_l231_231487

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l231_231487


namespace impossible_transformation_l231_231331

def f (x : ℝ) := x^2 + 5 * x + 4
def g (x : ℝ) := x^2 + 10 * x + 8

theorem impossible_transformation :
  (∀ x, f (x) = x^2 + 5 * x + 4) →
  (∀ x, g (x) = x^2 + 10 * x + 8) →
  (¬ ∃ t : ℝ → ℝ → ℝ, (∀ x, t (f x) x = g x)) :=
by
  sorry

end impossible_transformation_l231_231331


namespace cost_per_meter_of_fencing_l231_231378

/-- The sides of the rectangular field -/
def sides_ratio (length width : ℕ) : Prop := 3 * width = 4 * length

/-- The area of the rectangular field -/
def area (length width area : ℕ) : Prop := length * width = area

/-- The cost per meter of fencing -/
def cost_per_meter (total_cost perimeter : ℕ) : ℕ := total_cost * 100 / perimeter

/-- Prove that the cost per meter of fencing the field in paise is 25 given:
 1) The sides of a rectangular field are in the ratio 3:4.
 2) The area of the field is 8112 sq. m.
 3) The total cost of fencing the field is 91 rupees. -/
theorem cost_per_meter_of_fencing
  (length width perimeter : ℕ) 
  (h1 : sides_ratio length width)
  (h2 : area length width 8112)
  (h3 : perimeter = 2 * (length + width))
  (total_cost : ℕ)
  (h4 : total_cost = 91) :
  cost_per_meter total_cost perimeter = 25 :=
by
  sorry

end cost_per_meter_of_fencing_l231_231378


namespace older_friend_is_38_l231_231940

-- Define the conditions
def younger_friend_age (x : ℕ) : Prop := 
  ∃ (y : ℕ), (y = x + 2 ∧ x + y = 74)

-- Define the age of the older friend
def older_friend_age (x : ℕ) : ℕ := x + 2

-- State the theorem
theorem older_friend_is_38 : ∃ x, younger_friend_age x ∧ older_friend_age x = 38 :=
by
  sorry

end older_friend_is_38_l231_231940


namespace min_value_of_function_l231_231137

theorem min_value_of_function (x : ℝ) (h : x > 2) : (x + 1 / (x - 2)) ≥ 4 :=
  sorry

end min_value_of_function_l231_231137


namespace prove_a_eq_1_l231_231193

variables {a b c d k m : ℕ}
variables (h_odd_a : a%2 = 1) 
          (h_odd_b : b%2 = 1) 
          (h_odd_c : c%2 = 1) 
          (h_odd_d : d%2 = 1)
          (h_a_pos : 0 < a) 
          (h_ineq1 : a < b) 
          (h_ineq2 : b < c) 
          (h_ineq3 : c < d)
          (h_eqn1 : a * d = b * c)
          (h_eqn2 : a + d = 2^k) 
          (h_eqn3 : b + c = 2^m)

theorem prove_a_eq_1 
  (h_odd_a : a%2 = 1) 
  (h_odd_b : b%2 = 1) 
  (h_odd_c : c%2 = 1) 
  (h_odd_d : d%2 = 1)
  (h_a_pos : 0 < a) 
  (h_ineq1 : a < b) 
  (h_ineq2 : b < c) 
  (h_ineq3 : c < d)
  (h_eqn1 : a * d = b * c)
  (h_eqn2 : a + d = 2^k) 
  (h_eqn3 : b + c = 2^m) :
  a = 1 := by
  sorry

end prove_a_eq_1_l231_231193


namespace area_ratio_trapezoids_l231_231628

/-- 
Given trapezoid ABCD formed by three congruent isosceles triangles DAO, AOB, and OBC, 
where AD = AO = OB = BC = 13 and AB = DO = OC = 15, 
with points X and Y as midpoints of AD and BC respectively, 
prove that the simplified ratio of the areas of trapezoid ABYX to trapezoid XYCD is 1:1, 
and hence the sum p+q of the ratio p:q is 2.
 -/
theorem area_ratio_trapezoids (AD AO OB BC AB DO OC : ℝ) (X Y : ℝ) (ratio p q : ℕ) 
  (hcong : AD = AO ∧ AO = OB ∧ OB = BC ∧ BC = 13) 
  (hparallel : DO = OC ∧ OC = AB ∧ AB = 15)
  (hmidX : X = (AD / 2)) (hmidY : Y = (BC / 2))
  (hxy : X = Y)
  (hratio : ratio = (1:1))
  : (p + q = 2) := 
sorry

end area_ratio_trapezoids_l231_231628


namespace evaluate_series_sum_l231_231438

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l231_231438


namespace lisa_total_spoons_l231_231205

def num_children := 4
def spoons_per_child := 3
def decorative_spoons := 2
def large_spoons := 10
def teaspoons := 15

def total_spoons := num_children * spoons_per_child + decorative_spoons + large_spoons + teaspoons

theorem lisa_total_spoons : total_spoons = 39 := by
  sorry

end lisa_total_spoons_l231_231205


namespace ice_cream_sales_l231_231027

theorem ice_cream_sales : 
  let tuesday_sales := 12000
  let wednesday_sales := 2 * tuesday_sales
  let total_sales := tuesday_sales + wednesday_sales
  total_sales = 36000 := 
by 
  sorry

end ice_cream_sales_l231_231027


namespace series_sum_eq_four_ninths_l231_231495

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l231_231495


namespace cone_base_circumference_l231_231252

theorem cone_base_circumference (r : ℝ) (theta : ℝ) (h_r : r = 6) (h_theta : theta = 240) :
  (2 / 3) * (2 * Real.pi * r) = 8 * Real.pi :=
by
  have circle_circumference : ℝ := 2 * Real.pi * r
  sorry

end cone_base_circumference_l231_231252


namespace inverse_proposition_of_square_positive_l231_231042

theorem inverse_proposition_of_square_positive :
  (∀ x : ℝ, x < 0 → x^2 > 0) →
  (∀ x : ℝ, ¬ (x^2 > 0) → ¬ (x < 0)) :=
by
  intro h
  intros x h₁
  sorry

end inverse_proposition_of_square_positive_l231_231042


namespace remainder_of_power_mod_l231_231539

theorem remainder_of_power_mod :
  (5^2023) % 17 = 15 :=
begin
  sorry
end

end remainder_of_power_mod_l231_231539


namespace gcd_n_cube_plus_16_n_plus_3_l231_231860

theorem gcd_n_cube_plus_16_n_plus_3 (n : ℕ) (h : n > 2^3) : Nat.gcd (n^3 + 16) (n + 3) = 1 := 
sorry

end gcd_n_cube_plus_16_n_plus_3_l231_231860


namespace sum_max_min_ratio_ellipse_l231_231278

theorem sum_max_min_ratio_ellipse :
  ∃ (a b : ℝ), (∀ (x y : ℝ), 3*x^2 + 2*x*y + 4*y^2 - 18*x - 28*y + 50 = 0 → (y/x = a ∨ y/x = b)) ∧ a + b = 13 :=
by
  sorry

end sum_max_min_ratio_ellipse_l231_231278


namespace range_of_largest_root_l231_231989

theorem range_of_largest_root :
  ∀ (a_2 a_1 a_0 : ℝ), 
  (|a_2| ≤ 1 ∧ |a_1| ≤ 1 ∧ |a_0| ≤ 1) ∧ (a_2 + a_1 + a_0 = 0) →
  (∃ s > 1, ∀ x > 0, x^3 + 3*a_2*x^2 + 5*a_1*x + a_0 = 0 → x ≤ s) ∧
  (s < 2) :=
by sorry

end range_of_largest_root_l231_231989


namespace solve_quadratic_equation_l231_231124

noncomputable def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

theorem solve_quadratic_equation :
  let x1 := -2
      x2 := 11 in
  quadratic_eq 1 (-9) (-22) x1 ∧ quadratic_eq 1 (-9) (-22) x2 ∧ x1 < x2 :=
by
  sorry

end solve_quadratic_equation_l231_231124


namespace total_loss_is_correct_l231_231688

variable (A P : ℝ)
variable (Ashok_loss Pyarelal_loss : ℝ)

-- Condition 1: Ashok's capital is 1/9 of Pyarelal's capital
def ashokCapital (A P : ℝ) : Prop :=
  A = (1 / 9) * P

-- Condition 2: Pyarelal's loss was Rs 1800
def pyarelalLoss (Pyarelal_loss : ℝ) : Prop :=
  Pyarelal_loss = 1800

-- Question: What was the total loss in the business?
def totalLoss (Ashok_loss Pyarelal_loss : ℝ) : ℝ :=
  Ashok_loss + Pyarelal_loss

-- The mathematically equivalent proof problem statement
theorem total_loss_is_correct (P A : ℝ) (Ashok_loss Pyarelal_loss : ℝ)
  (h1 : ashokCapital A P)
  (h2 : pyarelalLoss Pyarelal_loss)
  (h3 : Ashok_loss = (1 / 9) * Pyarelal_loss) :
  totalLoss Ashok_loss Pyarelal_loss = 2000 := by
  sorry

end total_loss_is_correct_l231_231688


namespace find_n_l231_231319

noncomputable def parabola_focus : ℝ × ℝ :=
  (2, 0)

noncomputable def hyperbola_focus (n : ℝ) : ℝ × ℝ :=
  (Real.sqrt (3 + n), 0)

theorem find_n (n : ℝ) : hyperbola_focus n = parabola_focus → n = 1 :=
by
  sorry

end find_n_l231_231319


namespace fewer_onions_correct_l231_231272

-- Define the quantities
def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

-- Calculate the total number of tomatoes and corn
def tomatoes_and_corn : ℕ := tomatoes + corn

-- Calculate the number of fewer onions
def fewer_onions : ℕ := tomatoes_and_corn - onions

-- State the theorem and provide the proof
theorem fewer_onions_correct : fewer_onions = 5200 :=
by
  -- The statement is proved directly by the calculations above
  -- Providing the actual proof is not necessary as per the guidelines
  sorry

end fewer_onions_correct_l231_231272


namespace sum_series_eq_four_ninths_l231_231486

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l231_231486


namespace diet_cola_cost_l231_231599

theorem diet_cola_cost (T C : ℝ) 
  (h1 : T + 6 + C = 2 * T)
  (h2 : (T + 6 + C) + T = 24) : C = 2 := 
sorry

end diet_cola_cost_l231_231599


namespace problem_l231_231012

variable (a b c : ℝ)

def a_def : a = Real.log (1 / 2) := sorry
def b_def : b = Real.exp (1 / Real.exp 1) := sorry
def c_def : c = Real.exp (-2) := sorry

theorem problem (ha : a = Real.log (1 / 2)) 
               (hb : b = Real.exp (1 / Real.exp 1)) 
               (hc : c = Real.exp (-2)) : 
               a < c ∧ c < b := 
by
  rw [ha, hb, hc]
  sorry

end problem_l231_231012


namespace sum_of_factors_36_l231_231641

def is_factor (n d : ℕ) : Prop := d ∣ n

def sum_of_factors (n : ℕ) : ℕ := 
  (List.range (n + 1)).filter (λ d => is_factor n d).sum

theorem sum_of_factors_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_of_factors_36_l231_231641


namespace brandon_skittles_loss_l231_231695

theorem brandon_skittles_loss (original final : ℕ) (H1 : original = 96) (H2 : final = 87) : original - final = 9 :=
by sorry

end brandon_skittles_loss_l231_231695


namespace first_player_winning_strategy_l231_231238
noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem first_player_winning_strategy (x1 y1 : ℕ)
    (h1 : x1 > 0) (h2 : y1 > 0) :
    (x1 / y1 = 1) ∨ 
    (x1 / y1 > golden_ratio) ∨ 
    (x1 / y1 < 1 / golden_ratio) :=
sorry

end first_player_winning_strategy_l231_231238


namespace pie_slices_remaining_l231_231933

theorem pie_slices_remaining :
  let total_slices := 2 * 8 in
  let rebecca_slices := 1 + 1 in
  let remaining_after_rebecca := total_slices - rebecca_slices in
  let family_friends_slices := 0.5 * remaining_after_rebecca in
  let remaining_after_family_friends := remaining_after_rebecca - family_friends_slices in
  let sunday_evening_slices := 1 + 1 in
  let final_remaining_slices := remaining_after_family_friends - sunday_evening_slices in
  final_remaining_slices = 5 :=
by
  sorry

end pie_slices_remaining_l231_231933


namespace sin_cos_double_angle_identity_l231_231131

theorem sin_cos_double_angle_identity (α : ℝ) 
  (h1 : Real.sin α = 1/3) 
  (h2 : α ∈ Set.Ioc (π/2) π) : 
  Real.sin (2*α) + Real.cos (2*α) = (7 - 4 * Real.sqrt 2) / 9 := 
by
  sorry

end sin_cos_double_angle_identity_l231_231131


namespace sum_of_factors_of_36_l231_231643

-- Define the number in question
def n : ℕ := 36

-- Define the set of whole-number factors of n
def factors_of_n : Finset ℕ := {1, 2, 3, 4, 6, 9, 12, 18, 36}

-- Define the sum of the factors of n
def sum_of_factors : ℕ := factors_of_n.sum (λ x, x)

-- The theorem to be proven
theorem sum_of_factors_of_36 : sum_of_factors = 91 := 
by {
  sorry
}

end sum_of_factors_of_36_l231_231643


namespace sum_of_factors_36_eq_91_l231_231658

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l231_231658


namespace germinated_seeds_l231_231247

noncomputable def germination_deviation_bound (n : ℕ) (p : ℝ) (P : ℝ) : Prop :=
  let q := 1 - p in
  ∃ ε : ℝ, ε ≈ 0.034 ∧
  P (|((m : ℝ) / n : ℝ) - p| < ε) = P

theorem germinated_seeds :
  germination_deviation_bound 600 0.9 0.995 :=
sorry

end germinated_seeds_l231_231247


namespace probability_of_line_intersecting_both_squares_l231_231594

open MeasureTheory

theorem probability_of_line_intersecting_both_squares :
  let R := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1} in
  let P := uniformMeasure R in
  ∫ p in R, indicator (set_of (λ p : ℝ × ℝ, ∃ b : ℝ, p.2 = 1 / 2 * p.1 + b ∧ -1 / 2 ≤ b ∧ b ≤ 1 / 2 ∧ 0 ≤ 1 / 2 + b ∧ 1 / 2 + b ≤ 1)) p ∂P = 3 / 4 := sorry

end probability_of_line_intersecting_both_squares_l231_231594


namespace paperclips_exceed_200_at_friday_l231_231209

def paperclips_on_day (n : ℕ) : ℕ :=
  3 * 4^n

theorem paperclips_exceed_200_at_friday : 
  ∃ n : ℕ, n = 4 ∧ paperclips_on_day n > 200 :=
by
  sorry

end paperclips_exceed_200_at_friday_l231_231209


namespace product_of_all_n_satisfying_quadratic_l231_231126

theorem product_of_all_n_satisfying_quadratic :
  (∃ n : ℕ, n^2 - 40 * n + 399 = 3) ∧
  (∀ p : ℕ, Prime p → ((∃ n : ℕ, n^2 - 40 * n + 399 = p) → p = 3)) →
  ∃ n1 n2 : ℕ, (n1^2 - 40 * n1 + 399 = 3) ∧ (n2^2 - 40 * n2 + 399 = 3) ∧ n1 ≠ n2 ∧ (n1 * n2 = 396) :=
by
  sorry

end product_of_all_n_satisfying_quadratic_l231_231126


namespace choir_members_unique_l231_231227

theorem choir_members_unique (n : ℕ) :
  (n % 10 = 6) ∧ 
  (n % 11 = 6) ∧ 
  (150 ≤ n) ∧ 
  (n ≤ 300) → 
  n = 226 := 
by
  sorry

end choir_members_unique_l231_231227


namespace part_I_part_II_l231_231865

variable {a b : ℝ}

theorem part_I (h1 : a * b ≠ 0) (h2 : a * b > 0) :
  b / a + a / b ≥ 2 :=
sorry

theorem part_II (h1 : a * b ≠ 0) (h3 : a * b < 0) :
  abs (b / a + a / b) ≥ 2 :=
sorry

end part_I_part_II_l231_231865


namespace sum_of_roots_of_quadratic_eq_l231_231816

theorem sum_of_roots_of_quadratic_eq (x : ℝ) (hx : x^2 = 8 * x + 15) :
  ∃ S : ℝ, S = 8 :=
by
  sorry

end sum_of_roots_of_quadratic_eq_l231_231816


namespace P_ne_77_for_integers_l231_231779

def P (x y : ℤ) : ℤ :=
  x^5 - 4 * x^4 * y - 5 * y^2 * x^3 + 20 * y^3 * x^2 + 4 * y^4 * x - 16 * y^5

theorem P_ne_77_for_integers (x y : ℤ) : P x y ≠ 77 :=
by
  sorry

end P_ne_77_for_integers_l231_231779


namespace infinite_series_sum_l231_231518

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l231_231518


namespace adult_ticket_cost_l231_231630

def num_total_tickets : ℕ := 510
def cost_senior_ticket : ℕ := 15
def total_receipts : ℤ := 8748
def num_senior_tickets : ℕ := 327
def num_adult_tickets : ℕ := num_total_tickets - num_senior_tickets
def revenue_senior : ℤ := num_senior_tickets * cost_senior_ticket
def revenue_adult (cost_adult_ticket : ℤ) : ℤ := num_adult_tickets * cost_adult_ticket

theorem adult_ticket_cost : 
  ∃ (cost_adult_ticket : ℤ), 
    revenue_adult cost_adult_ticket + revenue_senior = total_receipts ∧ 
    cost_adult_ticket = 21 :=
by
  sorry

end adult_ticket_cost_l231_231630


namespace power_mod_l231_231543

theorem power_mod (a : ℕ) : 5 ^ 2023 % 17 = 2 := by
  sorry

end power_mod_l231_231543


namespace age_of_older_friend_l231_231939

theorem age_of_older_friend (a b : ℕ) (h1 : a - b = 2) (h2 : a + b = 74) : a = 38 :=
by
  sorry

end age_of_older_friend_l231_231939


namespace mia_days_not_worked_l231_231210

theorem mia_days_not_worked :
  ∃ (y : ℤ), (∃ (x : ℤ), 
  x + y = 30 ∧ 80 * x - 40 * y = 1600) ∧ y = 20 :=
by
  sorry

end mia_days_not_worked_l231_231210


namespace calculation_result_l231_231384

theorem calculation_result : 3 * 11 + 3 * 12 + 3 * 15 + 11 = 125 := 
by
  sorry

end calculation_result_l231_231384


namespace triangle_angle_y_l231_231065

theorem triangle_angle_y (y : ℝ) (h1 : 2 * y + (y + 10) + 4 * y = 180) : 
  y = 170 / 7 := 
by
  sorry

end triangle_angle_y_l231_231065


namespace students_playing_both_football_and_cricket_l231_231774

theorem students_playing_both_football_and_cricket
  (total_students : ℕ)
  (students_playing_football : ℕ)
  (students_playing_cricket : ℕ)
  (students_neither_football_nor_cricket : ℕ) :
  total_students = 250 →
  students_playing_football = 160 →
  students_playing_cricket = 90 →
  students_neither_football_nor_cricket = 50 →
  (students_playing_football + students_playing_cricket - (total_students - students_neither_football_nor_cricket)) = 50 :=
by
  intros h_total h_football h_cricket h_neither
  sorry

end students_playing_both_football_and_cricket_l231_231774


namespace race_total_people_l231_231000

theorem race_total_people (b t : ℕ) 
(h1 : b = t + 15) 
(h2 : 3 * t = 2 * b + 15) : 
b + t = 105 := 
sorry

end race_total_people_l231_231000


namespace ratio_simplified_l231_231862

theorem ratio_simplified (total finished : ℕ) (h_total : total = 15) (h_finished : finished = 6) :
  (total - finished) / (Nat.gcd (total - finished) finished) = 3 ∧ finished / (Nat.gcd (total - finished) finished) = 2 := by
  sorry

end ratio_simplified_l231_231862


namespace rearrangement_inequality_l231_231867

theorem rearrangement_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a / (b + c) + b / (a + c) + c / (a + b)) ≥ (3 / 2) ∧ (a = b ∧ b = c ∧ c = a ↔ (a / (b + c) + b / (a + c) + c / (a + b) = 3 / 2)) :=
by 
  -- Proof omitted
  sorry

end rearrangement_inequality_l231_231867


namespace proof_problem_l231_231902

noncomputable def red_balls : ℕ := 5
noncomputable def black_balls : ℕ := 2
noncomputable def total_balls : ℕ := red_balls + black_balls
noncomputable def draws : ℕ := 3

noncomputable def prob_red_ball := red_balls / total_balls
noncomputable def prob_black_ball := black_balls / total_balls

noncomputable def E_X : ℚ := (1/7) + 2*(4/7) + 3*(2/7)
noncomputable def E_Y : ℚ := 2*(1/7) + 1*(4/7) + 0*(2/7)
noncomputable def E_xi : ℚ := 3 * (5/7)

noncomputable def D_X : ℚ := (1 - 15/7) ^ 2 * (1/7) + (2 - 15/7) ^ 2 * (4/7) + (3 - 15/7) ^ 2 * (2/7)
noncomputable def D_Y : ℚ := (2 - 6/7) ^ 2 * (1/7) + (1 - 6/7) ^ 2 * (4/7) + (0 - 6/7) ^ 2 * (2/7)
noncomputable def D_xi : ℚ := 3 * (5/7) * (1 - 5/7)

theorem proof_problem :
  (E_X / E_Y = 5 / 2) ∧ 
  (D_X ≤ D_Y) ∧ 
  (E_X = E_xi) ∧ 
  (D_X < D_xi) :=
by {
  sorry
}

end proof_problem_l231_231902


namespace part_a_part_b_part_c_part_d_part_e_l231_231029

variable (n : ℤ)

theorem part_a : (n^3 - n) % 3 = 0 :=
  sorry

theorem part_b : (n^5 - n) % 5 = 0 :=
  sorry

theorem part_c : (n^7 - n) % 7 = 0 :=
  sorry

theorem part_d : (n^11 - n) % 11 = 0 :=
  sorry

theorem part_e : (n^13 - n) % 13 = 0 :=
  sorry

end part_a_part_b_part_c_part_d_part_e_l231_231029


namespace expected_visible_people_l231_231824

open BigOperators

def X (n : ℕ) : ℕ := -- Define the random variable X_n for the number of visible people, this needs a formal definition

noncomputable def harmonic_sum (n : ℕ) : ℚ := ∑ i in Finset.range (n + 1), (1:ℚ) / i.succ -- Harmonic sum

theorem expected_visible_people (n : ℕ) : 
  ∃ (E : ℕ → ℚ), E n = harmonic_sum n := by
  sorry

end expected_visible_people_l231_231824


namespace total_animals_hunted_l231_231582

theorem total_animals_hunted :
  let sam_hunts := 6
  let rob_hunts := sam_hunts / 2
  let total_sam_rob := sam_hunts + rob_hunts
  let mark_hunts := total_sam_rob / 3
  let peter_hunts := mark_hunts * 3
  sam_hunts + rob_hunts + mark_hunts + peter_hunts = 21 :=
by
  sorry

end total_animals_hunted_l231_231582


namespace alpha_plus_beta_l231_231863

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π)
variable (hβ : 0 < β ∧ β < π)
variable (h1 : Real.sin (α - β) = 3 / 4)
variable (h2 : Real.tan α / Real.tan β = -5)

theorem alpha_plus_beta (h3 : α + β = 5 * π / 6) : α + β = 5 * π / 6 :=
by
  sorry

end alpha_plus_beta_l231_231863


namespace trajectory_is_parabola_l231_231614

theorem trajectory_is_parabola (C : ℝ × ℝ) (M : ℝ × ℝ) (l : ℝ → ℝ)
  (hM : M = (0, 3)) (hl : ∀ y, l y = -3)
  (h : dist C M = |C.2 + 3|) : C.1^2 = 12 * C.2 := by
  sorry

end trajectory_is_parabola_l231_231614


namespace more_customers_after_lunch_rush_l231_231260

-- Definitions for conditions
def initial_customers : ℝ := 29.0
def added_customers : ℝ := 20.0
def total_customers : ℝ := 83.0

-- The number of additional customers that came in after the lunch rush
def additional_customers (initial additional total : ℝ) : ℝ :=
  total - (initial + additional)

-- Statement to prove
theorem more_customers_after_lunch_rush :
  additional_customers initial_customers added_customers total_customers = 34.0 :=
by
  sorry

end more_customers_after_lunch_rush_l231_231260


namespace unique_pair_exists_l231_231916

theorem unique_pair_exists (p : ℕ) (hp : p.prime ) (hodd : p % 2 = 1) : 
  ∃ m n : ℕ, m ≠ n ∧ (2 : ℚ) / (p : ℚ) = 1 / (n : ℚ) + 1 / (m : ℚ) ∧ 
             (n = (p + 1) / 2) ∧ (m = (p * (p + 1)) / 2) :=
by
  sorry

end unique_pair_exists_l231_231916


namespace log_product_l231_231276

theorem log_product : (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 2 := by
  sorry

end log_product_l231_231276


namespace sum_factors_36_l231_231653

theorem sum_factors_36 : ∑ x in {1, 2, 3, 4, 6, 9, 12, 18, 36}, x = 91 := by
  sorry

end sum_factors_36_l231_231653


namespace distinct_collections_of_letters_l231_231360

open Finset

noncomputable def GEOGRAPHY : Multiset Char := ['G', 'E', 'O', 'G', 'R', 'A', 'P', 'H', 'Y']

def vowels : Finset Char := {'E', 'O', 'A'}
def consonants : Finset Char := {'G', 'R', 'P', 'H', 'Y'}

theorem distinct_collections_of_letters :
  (count_combinations GEOGRAPHY 2 vowels) * (count_combinations GEOGRAPHY 3 consonants) = 68 :=
by
  sorry

end distinct_collections_of_letters_l231_231360


namespace sin_cos_inequality_l231_231765

theorem sin_cos_inequality (α : ℝ) 
  (h1 : 0 ≤ α) (h2 : α < 2 * Real.pi) 
  (h3 : Real.sin α > Real.sqrt 3 * Real.cos α) : 
  (Real.pi / 3 < α ∧ α < 4 * Real.pi / 3) :=
sorry

end sin_cos_inequality_l231_231765


namespace find_m_value_l231_231092

theorem find_m_value (m : ℚ) :
  (m - 10) / -10 = (5 - m) / -8 → m = 65 / 9 :=
by
  sorry

end find_m_value_l231_231092


namespace how_many_years_older_is_a_than_b_l231_231074

variable (a b c : ℕ)

theorem how_many_years_older_is_a_than_b
  (hb : b = 4)
  (hc : c = b / 2)
  (h_ages_sum : a + b + c = 12) :
  a - b = 2 := by
  sorry

end how_many_years_older_is_a_than_b_l231_231074


namespace initial_number_of_men_l231_231220

variable (M : ℕ) (A : ℕ)
variable (change_in_age: ℕ := 16)
variable (age_increment: ℕ := 2)

theorem initial_number_of_men :
  ((A + age_increment) * M = A * M + change_in_age) → M = 8 :=
by
  intros h_1
  sorry

end initial_number_of_men_l231_231220


namespace cloves_of_garlic_needed_l231_231391

def cloves_needed_for_vampires (vampires : ℕ) : ℕ :=
  (vampires * 3) / 2

def cloves_needed_for_wights (wights : ℕ) : ℕ :=
  (wights * 3) / 3

def cloves_needed_for_vampire_bats (vampire_bats : ℕ) : ℕ :=
  (vampire_bats * 3) / 8

theorem cloves_of_garlic_needed (vampires wights vampire_bats : ℕ) :
  cloves_needed_for_vampires 30 + cloves_needed_for_wights 12 + 
  cloves_needed_for_vampire_bats 40 = 72 :=
by
  sorry

end cloves_of_garlic_needed_l231_231391


namespace larger_solution_quadratic_l231_231711

theorem larger_solution_quadratic :
  ∃ x : ℝ, x^2 - 13 * x + 30 = 0 ∧ (∀ y : ℝ, y^2 - 13 * y + 30 = 0 → y ≤ x) ∧ x = 10 := 
by
  sorry

end larger_solution_quadratic_l231_231711


namespace infinite_series_sum_l231_231524

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l231_231524


namespace greatest_possible_value_l231_231313

theorem greatest_possible_value (x : ℝ) (hx : x^3 + (1 / x^3) = 9) : x + (1 / x) = 3 := by
  sorry

end greatest_possible_value_l231_231313


namespace range_of_k_l231_231745

noncomputable def quadratic_has_real_roots (k : ℝ): Prop :=
  ∃ x : ℝ, k * x^2 - 2 * x - 1 = 0

theorem range_of_k (k : ℝ) : quadratic_has_real_roots k ↔ k ≥ -1 :=
by
  sorry

end range_of_k_l231_231745


namespace total_screens_sold_is_45000_l231_231690

-- Define the number of screens sold in each month based on X
variables (X : ℕ)

-- Conditions given in the problem
def screens_in_January := X
def screens_in_February := 2 * X
def screens_in_March := (screens_in_January X + screens_in_February X) / 2
def screens_in_April := min (2 * screens_in_March X) 20000

-- Given that April sales were 18000
axiom apr_sales_18000 : screens_in_April X = 18000

-- Total sales is the sum of sales from January to April
def total_sales := screens_in_January X + screens_in_February X + screens_in_March X + 18000

-- Prove that total sales is 45000
theorem total_screens_sold_is_45000 : total_sales X = 45000 :=
by sorry

end total_screens_sold_is_45000_l231_231690


namespace students_chose_apples_l231_231329

theorem students_chose_apples (total students choosing_bananas : ℕ) (h1 : students_choosing_bananas = 168) 
  (h2 : 3 * total = 4 * students_choosing_bananas) : (total / 4) = 56 :=
  by
  sorry

end students_chose_apples_l231_231329


namespace max_projection_area_tetrahedron_l231_231995

-- Define the side length of the tetrahedron
variable (a : ℝ)

-- Define a theorem stating the maximum projection area of a tetrahedron
theorem max_projection_area_tetrahedron (h : a > 0) : 
  ∃ A, A = (a^2 / 2) :=
by
  -- Proof is omitted
  sorry

end max_projection_area_tetrahedron_l231_231995


namespace choir_members_l231_231369

theorem choir_members (k m n : ℕ) (h1 : n = k^2 + 11) (h2 : n = m * (m + 5)) : n ≤ 325 :=
by
  sorry -- A proof would go here, showing that n = 325 meets the criteria

end choir_members_l231_231369


namespace sum_of_factors_36_eq_91_l231_231659

/-- A helper definition to list all whole-number factors of 36 -/
def factors_of_36 : List ℕ := [1, 2, 3, 4, 6, 9, 12, 18, 36]

/-- The sum_of_factors function computes the sum of factors of a given number -/
def sum_of_factors (n : ℕ) (factors : List ℕ) : ℕ :=
  factors.foldl (· + ·) 0

/-- The main theorem stating that the sum of the whole-number factors of 36 is 91 -/
theorem sum_of_factors_36_eq_91 : sum_of_factors 36 factors_of_36 = 91 := by
  sorry

end sum_of_factors_36_eq_91_l231_231659


namespace sum_series_equals_4_div_9_l231_231475

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l231_231475


namespace remainder_division_l231_231307

variable (P D K Q R R'_q R'_r : ℕ)

theorem remainder_division (h1 : P = Q * D + R) (h2 : R = R'_q * K + R'_r) (h3 : K < D) : 
  P % (D * K) = R'_r :=
sorry

end remainder_division_l231_231307


namespace evaluate_series_sum_l231_231442

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l231_231442


namespace angle_bisector_slope_l231_231286

theorem angle_bisector_slope :
  ∀ m1 m2 : ℝ, m1 = 2 → m2 = 4 → (∃ k : ℝ, k = (6 - Real.sqrt 21) / (-7) → k = (-6 + Real.sqrt 21) / 7) :=
by
  sorry

end angle_bisector_slope_l231_231286


namespace fraction_problem_l231_231732

-- Definitions translated from conditions
variables (m n p q : ℚ)
axiom h1 : m / n = 20
axiom h2 : p / n = 5
axiom h3 : p / q = 1 / 15

-- Statement to prove
theorem fraction_problem : m / q = 4 / 15 :=
by
  sorry

end fraction_problem_l231_231732


namespace sum_factors_36_l231_231636

theorem sum_factors_36 : (∑ n in {d | d ∣ 36}.to_finset, n) = 91 :=
by
  sorry

end sum_factors_36_l231_231636


namespace max_strong_boys_l231_231575

theorem max_strong_boys (n : ℕ) (h : n = 100) (a b : Fin n → ℕ) 
  (ha : ∀ i j : Fin n, i < j → a i > a j) 
  (hb : ∀ i j : Fin n, i < j → b i < b j) : 
  ∃ k : ℕ, k = n := 
sorry

end max_strong_boys_l231_231575


namespace pages_copyable_l231_231333

-- Define the conditions
def cents_per_dollar : ℕ := 100
def dollars_available : ℕ := 25
def cost_per_page : ℕ := 3

-- Define the total cents available
def total_cents : ℕ := dollars_available * cents_per_dollar

-- Define the expected number of full pages
def expected_pages : ℕ := 833

theorem pages_copyable :
  (total_cents : ℕ) / cost_per_page = expected_pages := sorry

end pages_copyable_l231_231333


namespace sum_abc_l231_231314

theorem sum_abc (a b c : ℝ) 
  (h : (a - 6)^2 + (b - 3)^2 + (c - 2)^2 = 0) : 
  a + b + c = 11 := 
by 
  sorry

end sum_abc_l231_231314


namespace giants_need_to_win_more_games_l231_231037

/-- The Giants baseball team is trying to make their league playoff.
They have played 20 games and won 12 of them. To make the playoffs, they need to win 2/3 of 
their games over the season. If there are 10 games left, how many do they have to win to
make the playoffs? 
-/
theorem giants_need_to_win_more_games (played won needed_won total remaining required_wins additional_wins : ℕ)
    (h1 : played = 20)
    (h2 : won = 12)
    (h3 : remaining = 10)
    (h4 : total = played + remaining)
    (h5 : total = 30)
    (h6 : required_wins = 2 * total / 3)
    (h7 : additional_wins = required_wins - won) :
    additional_wins = 8 := 
    by
      -- sorry should be used if the proof steps were required.
sorry

end giants_need_to_win_more_games_l231_231037


namespace cloves_needed_l231_231394

theorem cloves_needed (cv_fp : 3 / 2 = 1.5) (cw_fp : 3 / 3 = 1) (vc_fp : 3 / 8 = 0.375) : 
  let cloves_for_vampires := 45
  let cloves_for_wights := 12
  let cloves_for_bats := 15
  30 * (3 / 2) + 12 * (3 / 3) + 40 * (3 / 8) = 72 := by
  sorry

end cloves_needed_l231_231394


namespace unique_nat_pair_l231_231919

theorem unique_nat_pair (p : ℕ) (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ (n m : ℕ), n ≠ m ∧ (2 / p : ℚ) = (1 / n + 1 / m : ℚ) ∧ ∀ (n' m' : ℕ), 
  n' ≠ m' ∧ (2 / p : ℚ) = (1 / n' + 1 / m' : ℚ) → (n', m') = (n, m) ∨ (n', m') = (m, n) :=
by
  sorry

end unique_nat_pair_l231_231919


namespace bob_questions_created_l231_231692

theorem bob_questions_created :
  let q1 := 13
  let q2 := 2 * q1
  let q3 := 2 * q2
  q1 + q2 + q3 = 91 :=
by
  sorry

end bob_questions_created_l231_231692


namespace sum_of_squares_l231_231293

theorem sum_of_squares (a b c : ℝ) :
  a + b + c = 4 → ab + ac + bc = 4 → a^2 + b^2 + c^2 = 8 :=
by
  sorry

end sum_of_squares_l231_231293


namespace sum_k_over_4k_l231_231449

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l231_231449


namespace maximize_area_center_coordinates_l231_231619

theorem maximize_area_center_coordinates (k : ℝ) :
  (∃ r : ℝ, r^2 = 1 - (3/4) * k^2 ∧ r ≥ 0) →
  ((k = 0) → ∃ a b : ℝ, (a = 0 ∧ b = -1)) :=
by
  sorry

end maximize_area_center_coordinates_l231_231619


namespace cyrus_pages_proof_l231_231992

def pages_remaining (total_pages: ℝ) (day1: ℝ) (day2: ℝ) (day3: ℝ) (day4: ℝ) (day5: ℝ) : ℝ :=
  total_pages - (day1 + day2 + day3 + day4 + day5)

theorem cyrus_pages_proof :
  let total_pages := 750
  let day1 := 30
  let day2 := 1.5 * day1
  let day3 := day2 / 2
  let day4 := 2.5 * day3
  let day5 := 15
  pages_remaining total_pages day1 day2 day3 day4 day5 = 581.25 :=
by 
  sorry

end cyrus_pages_proof_l231_231992


namespace trapezoid_problem_l231_231363

theorem trapezoid_problem (b h x : ℝ) 
  (h1 : x = (12500 / (x - 75)) - 75)
  (h_cond : (b + 75) / (b + 25) = 3 / 2)
  (b_solution : b = 75) :
  (⌊(x^2 / 100)⌋ : ℤ) = 181 :=
by
  -- The statement only requires us to assert the proof goal
  sorry

end trapezoid_problem_l231_231363


namespace intersection_A_B_l231_231879

open Set

variable (x : ℝ)

def A : Set ℝ := {x | x^2 - 3 * x - 4 < 0}
def B : Set ℝ := {-4, 1, 3, 5}

theorem intersection_A_B :
  A ∩ B = { 1, 3 } :=
sorry

end intersection_A_B_l231_231879


namespace neither_necessary_nor_sufficient_l231_231720

theorem neither_necessary_nor_sufficient (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) :
  ¬(∀ a b, (a > b → (1 / a < 1 / b)) ∧ ((1 / a < 1 / b) → a > b)) := sorry

end neither_necessary_nor_sufficient_l231_231720


namespace initial_lychees_count_l231_231925

theorem initial_lychees_count (L : ℕ) (h1 : L / 2 = 2 * 100 * 5 / 5 * 5) : L = 500 :=
by sorry

end initial_lychees_count_l231_231925


namespace negation_of_homework_submission_l231_231043

variable {S : Type} -- S is the set of all students in this class
variable (H : S → Prop) -- H(x) means "student x has submitted the homework"

theorem negation_of_homework_submission :
  (¬ ∀ x, H x) ↔ (∃ x, ¬ H x) :=
by
  sorry

end negation_of_homework_submission_l231_231043


namespace cafeteria_apples_l231_231618

theorem cafeteria_apples (handed_out: ℕ) (pies: ℕ) (apples_per_pie: ℕ) 
(h1: handed_out = 27) (h2: pies = 5) (h3: apples_per_pie = 4) : handed_out + pies * apples_per_pie = 47 :=
by
  -- The proof will be provided here if needed
  sorry

end cafeteria_apples_l231_231618


namespace number_machine_output_l231_231093

def number_machine (n : ℕ) : ℕ :=
  let step1 := n * 3
  let step2 := step1 + 20
  let step3 := step2 / 2
  let step4 := step3 ^ 2
  let step5 := step4 - 45
  step5

theorem number_machine_output : number_machine 90 = 20980 := by
  sorry

end number_machine_output_l231_231093


namespace percentage_increase_equiv_l231_231244

theorem percentage_increase_equiv {P : ℝ} : 
  (P * (1 + 0.08) * (1 + 0.08)) = (P * 1.1664) :=
by
  sorry

end percentage_increase_equiv_l231_231244


namespace num_valid_n_l231_231015

theorem num_valid_n (n q r : ℤ) (h₁ : 10000 ≤ n) (h₂ : n ≤ 99999)
  (h₃ : n = 50 * q + r) (h₄ : 200 ≤ q) (h₅ : q ≤ 1999)
  (h₆ : 0 ≤ r) (h₇ : r < 50) :
  (∃ (count : ℤ), count = 14400) := by
  sorry

end num_valid_n_l231_231015


namespace intersection_of_A_and_B_l231_231873

def setA : Set ℝ := { x | x^2 - 3 * x - 4 < 0 }
def setB : Set ℝ := {-4, 1, 3, 5}
def resultSet : Set ℝ := {1, 3}

theorem intersection_of_A_and_B :
  setA ∩ setB = resultSet := 
by
  sorry

end intersection_of_A_and_B_l231_231873


namespace count_p_shape_points_l231_231072

-- Define the problem conditions
def side_length : ℕ := 10
def point_interval : ℕ := 1
def num_sides : ℕ := 3
def correction_corners : ℕ := 2

-- Define the total expected points
def total_expected_points : ℕ := 31

-- Proof statement
theorem count_p_shape_points :
  ((side_length / point_interval + 1) * num_sides - correction_corners) = total_expected_points := by
  sorry

end count_p_shape_points_l231_231072


namespace prize_winner_is_B_l231_231851

-- Define the possible entries winning the prize
inductive Prize
| A
| B
| C
| D

open Prize

-- Define each student's predictions
def A_pred (prize : Prize) : Prop := prize = C ∨ prize = D
def B_pred (prize : Prize) : Prop := prize = B
def C_pred (prize : Prize) : Prop := prize ≠ A ∧ prize ≠ D
def D_pred (prize : Prize) : Prop := prize = C

-- Define the main theorem to prove
theorem prize_winner_is_B (prize : Prize) :
  (A_pred prize ∧ B_pred prize ∧ ¬C_pred prize ∧ ¬D_pred prize) ∨
  (A_pred prize ∧ ¬B_pred prize ∧ C_pred prize ∧ ¬D_pred prize) ∨
  (¬A_pred prize ∧ B_pred prize ∧ C_pred prize ∧ ¬D_pred prize) ∨
  (¬A_pred prize ∧ ¬B_pred prize ∧ C_pred prize ∧ D_pred prize) →
  prize = B :=
sorry

end prize_winner_is_B_l231_231851


namespace Jose_share_land_l231_231336

theorem Jose_share_land (total_land : ℕ) (num_siblings : ℕ) (total_parts : ℕ) (share_per_person : ℕ) :
  total_land = 20000 → num_siblings = 4 → total_parts = (1 + num_siblings) → share_per_person = (total_land / total_parts) → 
  share_per_person = 4000 :=
by
  sorry

end Jose_share_land_l231_231336


namespace probability_one_card_each_l231_231261

-- Define the total number of cards
def total_cards := 12

-- Define the number of cards from Adrian
def adrian_cards := 7

-- Define the number of cards from Bella
def bella_cards := 5

-- Calculate the probability of one card from each cousin when selecting two cards without replacement
theorem probability_one_card_each :
  (adrian_cards / total_cards) * (bella_cards / (total_cards - 1)) +
  (bella_cards / total_cards) * (adrian_cards / (total_cards - 1)) =
  35 / 66 := sorry

end probability_one_card_each_l231_231261


namespace value_of_b_l231_231158

theorem value_of_b (b : ℚ) (h : b + b / 4 = 3) : b = 12 / 5 := by
  sorry

end value_of_b_l231_231158


namespace ratio_current_to_past_l231_231112

-- Conditions
def current_posters : ℕ := 22
def posters_after_summer (p : ℕ) : ℕ := p + 6
def posters_two_years_ago : ℕ := 14

-- Proof problem statement
theorem ratio_current_to_past (h₁ : current_posters = 22) (h₂ : posters_two_years_ago = 14) : 
  (current_posters / Nat.gcd current_posters posters_two_years_ago) = 11 ∧ 
  (posters_two_years_ago / Nat.gcd current_posters posters_two_years_ago) = 7 :=
by
  sorry

end ratio_current_to_past_l231_231112


namespace students_not_enrolled_in_course_l231_231166

def total_students : ℕ := 150
def french_students : ℕ := 61
def german_students : ℕ := 32
def spanish_students : ℕ := 45
def french_and_german : ℕ := 15
def french_and_spanish : ℕ := 12
def german_and_spanish : ℕ := 10
def all_three_courses : ℕ := 5

theorem students_not_enrolled_in_course : total_students - 
    (french_students + german_students + spanish_students - 
     french_and_german - french_and_spanish - german_and_spanish + 
     all_three_courses) = 44 := by
  sorry

end students_not_enrolled_in_course_l231_231166


namespace quadratic_inequality_solution_l231_231626

theorem quadratic_inequality_solution 
  (a b : ℝ) 
  (h1 : (∀ x : ℝ, x^2 + a * x + b > 0 → (x < 3 ∨ x > 1))) :
  ∀ x : ℝ, a * x + b < 0 → x > 3 / 4 := 
by 
  sorry

end quadratic_inequality_solution_l231_231626


namespace series_sum_eq_four_ninths_l231_231494

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l231_231494


namespace ratio_eval_l231_231119

universe u

def a : ℕ := 121
def b : ℕ := 123
def c : ℕ := 122

theorem ratio_eval : (2 ^ a * 3 ^ b) / (6 ^ c) = (3 / 2) := by
  sorry

end ratio_eval_l231_231119


namespace sum_of_factors_36_l231_231665

theorem sum_of_factors_36 : 
  (∑ i in {d ∈ (finset.range 37) | 36 % d = 0}.to_finset, i) = 91 := 
by
  sorry

end sum_of_factors_36_l231_231665


namespace number_of_marked_points_l231_231608

theorem number_of_marked_points
  (a1 a2 b1 b2 : ℕ)
  (hA : a1 * a2 = 50)
  (hB : b1 * b2 = 56)
  (h_sum : a1 + a2 = b1 + b2) :
  a1 + a2 + 1 = 16 :=
sorry

end number_of_marked_points_l231_231608


namespace cost_price_600_l231_231971

variable (CP SP : ℝ)

theorem cost_price_600 
  (h1 : SP = 1.08 * CP) 
  (h2 : SP = 648) : 
  CP = 600 := 
by
  sorry

end cost_price_600_l231_231971


namespace sum_of_factors_36_l231_231660

def is_factor (n m : ℕ) : Prop :=
  m > 0 ∧ n % m = 0

def factors (n : ℕ) : List ℕ :=
  List.filter (is_factor n) (List.range (n + 1))

theorem sum_of_factors_36 : List.sum (factors 36) = 91 := 
  sorry

end sum_of_factors_36_l231_231660


namespace Priyanka_chocolates_l231_231987

variable (N S So P Sa T : ℕ)

theorem Priyanka_chocolates :
  (N + S = 10) →
  (So + P = 15) →
  (Sa + T = 10) →
  (N = 4) →
  ((S = 2 * y) ∨ (P = 2 * So)) →
  P = 10 :=
by
  sorry

end Priyanka_chocolates_l231_231987


namespace m_plus_n_eq_123_l231_231063

/- Define the smallest prime number -/
def m : ℕ := 2

/- Define the largest integer less than 150 with exactly three positive divisors -/
def n : ℕ := 121

/- Prove that the sum of m and n is 123 -/
theorem m_plus_n_eq_123 : m + n = 123 := by
  -- By definition, m is 2 and n is 121
  -- So, their sum is 123
  rfl

end m_plus_n_eq_123_l231_231063


namespace max_cubes_fit_l231_231635

theorem max_cubes_fit (L S : ℕ) (hL : L = 10) (hS : S = 2) : (L * L * L) / (S * S * S) = 125 := by
  sorry

end max_cubes_fit_l231_231635


namespace carla_initial_marbles_l231_231696

theorem carla_initial_marbles
  (marbles_bought : ℕ)
  (total_marbles_now : ℕ)
  (h1 : marbles_bought = 134)
  (h2 : total_marbles_now = 187) :
  total_marbles_now - marbles_bought = 53 :=
by
  sorry

end carla_initial_marbles_l231_231696


namespace sum_factors_36_l231_231648

theorem sum_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by 
  -- Sum the whole-number factors of 36
  sorry

end sum_factors_36_l231_231648


namespace alice_pints_wednesday_l231_231024

-- Initial conditions
def pints_sunday : ℕ := 4
def pints_monday : ℕ := 3 * pints_sunday
def pints_tuesday : ℕ := pints_monday / 3
def total_pints_before_return : ℕ := pints_sunday + pints_monday + pints_tuesday
def pints_returned_wednesday : ℕ := pints_tuesday / 2
def pints_wednesday : ℕ := total_pints_before_return - pints_returned_wednesday

-- The proof statement
theorem alice_pints_wednesday : pints_wednesday = 18 :=
by
  sorry

end alice_pints_wednesday_l231_231024


namespace solve_for_n_l231_231161

theorem solve_for_n (n : ℕ) (h : 2 * n - 5 = 1) : n = 3 :=
by
  sorry

end solve_for_n_l231_231161


namespace g_at_3_l231_231597

def g (x : ℝ) : ℝ := x^3 - 2 * x^2 + x

theorem g_at_3 : g 3 = 12 := by
  sorry

end g_at_3_l231_231597


namespace fraction_equal_l231_231729

variable {m n p q : ℚ}

-- Define the conditions
def condition1 := (m / n = 20)
def condition2 := (p / n = 5)
def condition3 := (p / q = 1 / 15)

-- State the theorem
theorem fraction_equal (h1 : condition1) (h2 : condition2) (h3 : condition3) : (m / q = 4 / 15) :=
  sorry

end fraction_equal_l231_231729


namespace fish_count_l231_231381

theorem fish_count (T : ℕ) :
  (T > 10 ∧ T ≤ 18) ∧ ((T > 18 ∧ T > 15 ∧ ¬(T > 10)) ∨ (¬(T > 18) ∧ T > 15 ∧ T > 10) ∨ (T > 18 ∧ ¬(T > 15) ∧ T > 10)) →
  T = 16 ∨ T = 17 ∨ T = 18 :=
sorry

end fish_count_l231_231381


namespace million_to_scientific_notation_l231_231274

theorem million_to_scientific_notation (population_henan : ℝ) (h : population_henan = 98.83 * 10^6) :
  population_henan = 9.883 * 10^7 :=
by sorry

end million_to_scientific_notation_l231_231274


namespace annual_interest_rate_is_correct_l231_231179

theorem annual_interest_rate_is_correct :
  ∃ r : ℝ, r = 0.0583 ∧
  (200 * (1 + r)^2 = 224) :=
by
  sorry

end annual_interest_rate_is_correct_l231_231179


namespace non_athletic_parents_l231_231627

-- Define the conditions
variables (total_students athletic_dads athletic_moms both_athletic : ℕ)

-- Assume the given conditions
axiom h1 : total_students = 45
axiom h2 : athletic_dads = 17
axiom h3 : athletic_moms = 20
axiom h4 : both_athletic = 11

-- Statement to be proven
theorem non_athletic_parents : total_students - (athletic_dads - both_athletic + athletic_moms - both_athletic + both_athletic) = 19 :=
by {
  -- We intentionally skip the proof here
  sorry
}

end non_athletic_parents_l231_231627


namespace myrtle_eggs_after_collection_l231_231212

def henA_eggs_per_day : ℕ := 3
def henB_eggs_per_day : ℕ := 4
def henC_eggs_per_day : ℕ := 2
def henD_eggs_per_day : ℕ := 5
def henE_eggs_per_day : ℕ := 3

def days_gone : ℕ := 12
def eggs_taken_by_neighbor : ℕ := 32

def eggs_dropped_day1 : ℕ := 3
def eggs_dropped_day2 : ℕ := 5
def eggs_dropped_day3 : ℕ := 2

theorem myrtle_eggs_after_collection :
  let total_eggs :=
    (henA_eggs_per_day * days_gone) +
    (henB_eggs_per_day * days_gone) +
    (henC_eggs_per_day * days_gone) +
    (henD_eggs_per_day * days_gone) +
    (henE_eggs_per_day * days_gone)
  let remaining_eggs_after_neighbor := total_eggs - eggs_taken_by_neighbor
  let total_dropped_eggs := eggs_dropped_day1 + eggs_dropped_day2 + eggs_dropped_day3
  let eggs_after_drops := remaining_eggs_after_neighbor - total_dropped_eggs
  eggs_after_drops = 162 := 
by 
  sorry

end myrtle_eggs_after_collection_l231_231212


namespace g_is_zero_l231_231343

noncomputable def g (x : Real) : Real := 
  Real.sqrt (Real.cos x ^ 4 + 4 * Real.sin x ^ 2) - 
  Real.sqrt (Real.sin x ^ 4 + 4 * Real.cos x ^ 2)

theorem g_is_zero : ∀ x : Real, g x = 0 := by
  sorry

end g_is_zero_l231_231343


namespace cost_of_each_pair_of_shorts_l231_231612

variable (C : ℝ)
variable (h_discount : 3 * C - 2.7 * C = 3)

theorem cost_of_each_pair_of_shorts : C = 10 :=
by 
  sorry

end cost_of_each_pair_of_shorts_l231_231612


namespace find_length_of_PB_l231_231348

theorem find_length_of_PB
  (PA : ℝ) -- Define PA
  (h_PA : PA = 4) -- Condition PA = 4
  (PB : ℝ) -- Define PB
  (PT : ℝ) -- Define PT
  (h_PT : PT = PB - 2 * PA) -- Condition PT = PB - 2 * PA
  (h_power_of_a_point : PA * PB = PT^2) -- Condition PA * PB = PT^2
  : PB = 16 :=
sorry

end find_length_of_PB_l231_231348


namespace sin_double_angle_l231_231571

open Real

theorem sin_double_angle (θ : ℝ) (h : cos (π / 4 - θ) = 1 / 2) : sin (2 * θ) = -1 / 2 := 
by 
  sorry

end sin_double_angle_l231_231571


namespace average_payment_debt_l231_231969

theorem average_payment_debt :
  let total_payments := 65
  let first_20_payment := 410
  let increment := 65
  let remaining_payment := first_20_payment + increment
  let first_20_total := 20 * first_20_payment
  let remaining_total := 45 * remaining_payment
  let total_paid := first_20_total + remaining_total
  let average_payment := total_paid / total_payments
  average_payment = 455 := by sorry

end average_payment_debt_l231_231969


namespace intersection_of_sets_l231_231151

noncomputable def set_A := {x : ℝ | Real.log x ≥ 0}
noncomputable def set_B := {x : ℝ | x^2 < 9}

theorem intersection_of_sets :
  set_A ∩ set_B = {x : ℝ | 1 ≤ x ∧ x < 3} :=
by {
  sorry
}

end intersection_of_sets_l231_231151


namespace lisa_total_spoons_l231_231204

def num_children := 4
def spoons_per_child := 3
def decorative_spoons := 2
def large_spoons := 10
def teaspoons := 15

def total_spoons := num_children * spoons_per_child + decorative_spoons + large_spoons + teaspoons

theorem lisa_total_spoons : total_spoons = 39 := by
  sorry

end lisa_total_spoons_l231_231204


namespace card_prob_queen_diamond_l231_231058

theorem card_prob_queen_diamond : 
  let total_cards := 52
  let queen_first_prob := 4 / 52
  let remaining_cards_after_queen := 51
  let diamond_after_queen_of_diamonds_prob := 12 / 51
  let diamond_after_non_diamond_queen_prob := 13 / 51
  let case1_prob := (1 / total_cards) * diamond_after_queen_of_diamonds_prob
  let case2_prob := (3 / total_cards) * diamond_after_non_diamond_queen_prob
  let total_prob := case1_prob + case2_prob in
  total_prob = 18 / 221 :=
by
  let total_cards := 52
  let queen_first_prob := 4 / 52
  let remaining_cards_after_queen := 51
  let diamond_after_queen_of_diamonds_prob := 12 / 51
  let diamond_after_non_diamond_queen_prob := 13 / 51
  let case1_prob := (1 / total_cards) * diamond_after_queen_of_diamonds_prob
  let case2_prob := (3 / total_cards) * diamond_after_non_diamond_queen_prob
  let total_prob := case1_prob + case2_prob
  sorry

end card_prob_queen_diamond_l231_231058


namespace negation_of_p_l231_231892

open Real

def p : Prop := ∃ x : ℝ, sin x < (1 / 2) * x

theorem negation_of_p : ¬p ↔ ∀ x : ℝ, sin x ≥ (1 / 2) * x := 
by
  sorry

end negation_of_p_l231_231892


namespace domain_of_sqrt_function_l231_231160

theorem domain_of_sqrt_function (m : ℝ) : (∀ x : ℝ, mx^2 + mx + 1 ≥ 0) ↔ 0 ≤ m ∧ m ≤ 4 := sorry

end domain_of_sqrt_function_l231_231160


namespace min_containers_needed_l231_231396

def container_capacity : ℕ := 500
def required_tea : ℕ := 5000

theorem min_containers_needed (n : ℕ) : n * container_capacity ≥ required_tea → n = 10 :=
sorry

end min_containers_needed_l231_231396


namespace find_x_l231_231081

noncomputable def eq_num (x : ℝ) : Prop :=
  9 - 3 / (1 / 3) + x = 3

theorem find_x : ∃ x : ℝ, eq_num x ∧ x = 3 := 
by
  sorry

end find_x_l231_231081


namespace calculator_transform_implication_l231_231083

noncomputable def transform (x n S : ℕ) : Prop :=
  (S > x^n + 1)

theorem calculator_transform_implication (x n S : ℕ) (hx : 0 < x) (hn : 0 < n) (hS : 0 < S) 
  (h_transform: transform x n S) : S > x^n + x - 1 := by
  sorry

end calculator_transform_implication_l231_231083


namespace sufficient_but_not_necessary_condition_l231_231742

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem sufficient_but_not_necessary_condition {x y : ℝ} :
  (floor x = floor y) → (abs (x - y) < 1) ∧ (¬ (abs (x - y) < 1) → (floor x ≠ floor y)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l231_231742


namespace sum_series_eq_4_div_9_l231_231506

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l231_231506


namespace omega_value_l231_231724

noncomputable def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6)

theorem omega_value (ω x₁ x₂ : ℝ) (h_ω : ω > 0) (h_x1 : f ω x₁ = -2) (h_x2 : f ω x₂ = 0) (h_min : |x₁ - x₂| = Real.pi) :
  ω = 1 / 2 := 
by 
  sorry

end omega_value_l231_231724


namespace repeating_decimal_as_fraction_l231_231998

theorem repeating_decimal_as_fraction : 
  let x := 2.353535... in
  x = 233 / 99 ∧ Nat.gcd 233 99 = 1 :=
by
  sorry

end repeating_decimal_as_fraction_l231_231998


namespace sixteen_is_sixtyfour_percent_l231_231785

theorem sixteen_is_sixtyfour_percent (x : ℝ) (h : 16 / x = 64 / 100) : x = 25 :=
by sorry

end sixteen_is_sixtyfour_percent_l231_231785


namespace greatest_x_value_l231_231634

theorem greatest_x_value : 
  (∃ x : ℝ, 2 * x^2 + 7 * x + 3 = 5 ∧ ∀ y : ℝ, (2 * y^2 + 7 * y + 3 = 5) → y ≤ x) → x = 1 / 2 :=
by
  sorry

end greatest_x_value_l231_231634


namespace ducks_and_geese_meeting_l231_231752

theorem ducks_and_geese_meeting:
  ∀ x : ℕ, ( ∀ ducks_speed : ℚ, ducks_speed = (1/7) ) → 
         ( ∀ geese_speed : ℚ, geese_speed = (1/9) ) → 
         (ducks_speed * x + geese_speed * x = 1) :=
by
  sorry

end ducks_and_geese_meeting_l231_231752


namespace problem_solution_l231_231284

theorem problem_solution (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ 3) :
    (8.17 * real.sqrt (3 * x - x ^ 2) < 4 - x) :=
sorry

end problem_solution_l231_231284


namespace sum_series_eq_four_ninths_l231_231490

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l231_231490


namespace place_value_ratio_56439_2071_l231_231175

theorem place_value_ratio_56439_2071 :
  let n := 56439.2071
  let digit_6_place_value := 1000
  let digit_2_place_value := 0.1
  digit_6_place_value / digit_2_place_value = 10000 :=
by
  sorry

end place_value_ratio_56439_2071_l231_231175


namespace P_not_77_for_all_integers_l231_231778

def P (x y : ℤ) : ℤ := x^5 - 4 * x^4 * y - 5 * y^2 * x^3 + 20 * y^3 * x^2 + 4 * y^4 * x - 16 * y^5

theorem P_not_77_for_all_integers (x y : ℤ) : P x y ≠ 77 :=
sorry

end P_not_77_for_all_integers_l231_231778


namespace inversely_proportional_y_l231_231188

theorem inversely_proportional_y (k : ℚ) (x y : ℚ) (hx_neg_10 : x = -10) (hy_5 : y = 5) (hprop : y * x = k) (hx_neg_4 : x = -4) : 
  y = 25 / 2 := 
by
  sorry

end inversely_proportional_y_l231_231188


namespace middle_number_is_40_l231_231967

theorem middle_number_is_40 (A B C : ℕ) (h1 : C = 56) (h2 : C - A = 32) (h3 : B / C = 5 / 7) : B = 40 :=
  sorry

end middle_number_is_40_l231_231967


namespace land_per_person_l231_231338

noncomputable def total_land_area : ℕ := 20000
noncomputable def num_people_sharing : ℕ := 5

theorem land_per_person (Jose_land : ℕ) (h : Jose_land = total_land_area / num_people_sharing) :
  Jose_land = 4000 :=
by
  sorry

end land_per_person_l231_231338


namespace probability_queen_then_diamond_l231_231056

-- Define a standard deck of 52 cards
def deck := List.range 52

-- Define a function to check if a card is a Queen
def is_queen (card : ℕ) : Prop :=
card % 13 = 10

-- Define a function to check if a card is a Diamond. Here assuming index for diamond starts at 0 and ends at 12
def is_diamond (card : ℕ) : Prop :=
card / 13 = 0

-- The main theorem statement
theorem probability_queen_then_diamond : 
  let prob := 1 / 52 * 12 / 51 + 3 / 52 * 13 / 51
  prob = 52 / 221 :=
by
  sorry

end probability_queen_then_diamond_l231_231056


namespace compute_value_l231_231697

theorem compute_value (a b c : ℕ) (h : a = 262 ∧ b = 258 ∧ c = 150) : 
  (a^2 - b^2) + c = 2230 := 
by
  sorry

end compute_value_l231_231697


namespace count_possible_P_l231_231005

-- Define the distinct digits with initial conditions
def digits : Type := {n // n ≥ 0 ∧ n ≤ 9}

-- Define the parameters P, Q, R, S as distinct digits
variables (P Q R S : digits)

-- Define the condition that P, Q, R, S are distinct.
def distinct (P Q R S : digits) : Prop := 
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S

-- Assertion conditions based on a valid subtraction layout
def valid_subtraction (P Q R S : digits) : Prop :=
  Q.val - P.val = S.val ∧ (P.val - R.val = P.val) ∧ (P.val - Q.val = S.val)

-- Prove that there are exactly 9 possible values for P.
theorem count_possible_P : ∃ n : ℕ, n = 9 ∧ ∀ P Q R S : digits, distinct P Q R S → valid_subtraction P Q R S → n = 9 :=
by sorry

end count_possible_P_l231_231005


namespace card_prob_queen_diamond_l231_231057

theorem card_prob_queen_diamond : 
  let total_cards := 52
  let queen_first_prob := 4 / 52
  let remaining_cards_after_queen := 51
  let diamond_after_queen_of_diamonds_prob := 12 / 51
  let diamond_after_non_diamond_queen_prob := 13 / 51
  let case1_prob := (1 / total_cards) * diamond_after_queen_of_diamonds_prob
  let case2_prob := (3 / total_cards) * diamond_after_non_diamond_queen_prob
  let total_prob := case1_prob + case2_prob in
  total_prob = 18 / 221 :=
by
  let total_cards := 52
  let queen_first_prob := 4 / 52
  let remaining_cards_after_queen := 51
  let diamond_after_queen_of_diamonds_prob := 12 / 51
  let diamond_after_non_diamond_queen_prob := 13 / 51
  let case1_prob := (1 / total_cards) * diamond_after_queen_of_diamonds_prob
  let case2_prob := (3 / total_cards) * diamond_after_non_diamond_queen_prob
  let total_prob := case1_prob + case2_prob
  sorry

end card_prob_queen_diamond_l231_231057


namespace verify_value_of_sum_l231_231235

noncomputable def value_of_sum (a b c d e f : ℕ) (values : Finset ℕ) : ℕ :=
if h : a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values ∧ e ∈ values ∧ f ∈ values ∧
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
        c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
        d ≠ e ∧ d ≠ f ∧
        e ≠ f ∧
        a + b = c ∧
        b + c = d ∧
        c + e = f
then a + c + f
else 0

theorem verify_value_of_sum :
  ∃ (a b c d e f : ℕ) (values : Finset ℕ),
  values = {4, 12, 15, 27, 31, 39} ∧
  a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values ∧ e ∈ values ∧ f ∈ values ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a + b = c ∧
  b + c = d ∧
  c + e = f ∧
  value_of_sum a b c d e f values = 73 :=
by
  sorry

end verify_value_of_sum_l231_231235


namespace sufficient_but_not_necessary_condition_l231_231550

-- Step d: Lean 4 statement
theorem sufficient_but_not_necessary_condition 
  (m n : ℕ) (e : ℚ) (h₁ : m = 5) (h₂ : n = 4) (h₃ : e = 3 / 5)
  (ellipse_eq : ∀ x y : ℝ, x^2 / m^2 + y^2 / n^2 = 1) :
  (m = 5 ∧ n = 4) → (e = 3 / 5) ∧ (¬(e = 3 / 5 → m = 5 ∧ n = 4)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l231_231550


namespace polynomial_roots_l231_231708

open Polynomial

theorem polynomial_roots :
  (roots (X^3 - 2 * X^2 - 5 * X + 6)).toFinset = {1, -2, 3} :=
sorry

end polynomial_roots_l231_231708


namespace remainder_of_5_pow_2023_mod_17_l231_231541

theorem remainder_of_5_pow_2023_mod_17 :
  5^2023 % 17 = 11 :=
by
  have h1 : 5^2 % 17 = 8 := by sorry
  have h2 : 5^4 % 17 = 13 := by sorry
  have h3 : 5^8 % 17 = -1 := by sorry
  have h4 : 5^16 % 17 = 1 := by sorry
  have h5 : 2023 = 16 * 126 + 7 := by sorry
  sorry

end remainder_of_5_pow_2023_mod_17_l231_231541


namespace solve_seating_problem_l231_231798

-- Define the conditions of the problem
def valid_seating_arrangements (n : ℕ) : Prop :=
  (∃ (x y : ℕ), x < y ∧ x + 1 < y ∧ y < n ∧ 
    (n ≥ 5 ∧ y - x - 1 > 0)) ∧
  (∃! (x' y' : ℕ), x' < y' ∧ x' + 1 < y' ∧ y' < n ∧ 
    (n ≥ 5 ∧ y' - x' - 1 > 0))

-- State the theorem
theorem solve_seating_problem : ∃ n : ℕ, valid_seating_arrangements n ∧ n = 5 :=
by
  sorry

end solve_seating_problem_l231_231798


namespace partI_solution_set_l231_231725

def f (x : ℝ) (a : ℝ) : ℝ := abs (x + a) - abs (x - a^2 - a)

theorem partI_solution_set (x : ℝ) : 
  (f x 1 ≤ 1) ↔ (x ≤ -1) :=
sorry

end partI_solution_set_l231_231725


namespace trajectory_of_center_of_moving_circle_l231_231945

noncomputable def center_trajectory (x y : ℝ) : Prop :=
  0 < y ∧ y ≤ 1 ∧ x^2 = 4 * (y - 1)

theorem trajectory_of_center_of_moving_circle (x y : ℝ) :
  0 ≤ y ∧ y ≤ 2 ∧ x^2 + y^2 = 4 ∧ 0 < y → center_trajectory x y :=
by
  sorry

end trajectory_of_center_of_moving_circle_l231_231945


namespace intersection_A_B_l231_231883

-- Define set A and set B based on the conditions
def set_A : Set ℝ := {x : ℝ | x^2 - 3 * x - 4 < 0}
def set_B : Set ℝ := {-4, 1, 3, 5}

-- State the theorem to prove the intersection of A and B
theorem intersection_A_B : set_A ∩ set_B = {1, 3} :=
by sorry

end intersection_A_B_l231_231883


namespace sum_series_l231_231472

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l231_231472


namespace weight_difference_l231_231936

theorem weight_difference (W_A W_B W_C W_D W_E : ℝ)
  (h_avg_ABC : (W_A + W_B + W_C) / 3 = 80)
  (h_WA : W_A = 95)
  (h_avg_ABCD : (W_A + W_B + W_C + W_D) / 4 = 82)
  (h_avg_BCDE : (W_B + W_C + W_D + W_E) / 4 = 81) :
  W_E - W_D = 3 :=
by
  sorry

end weight_difference_l231_231936


namespace problem_l231_231140

open Real

def p (x : ℝ) : Prop := 2*x^2 + 2*x + 1/2 < 0

def q (x y : ℝ) : Prop := (x^2)/4 - (y^2)/12 = 1 ∧ x ≥ 2

def x0_condition (x0 : ℝ) : Prop := sin x0 - cos x0 = sqrt 2

theorem problem (h1 : ∀ x : ℝ, ¬ p x)
               (h2 : ∃ x y : ℝ, q x y)
               (h3 : ∃ x0 : ℝ, x0_condition x0) :
               ∀ x : ℝ, ¬ ¬ p x := 
sorry

end problem_l231_231140


namespace sum_series_equals_4_div_9_l231_231482

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l231_231482


namespace batsman_average_after_17th_inning_l231_231082

theorem batsman_average_after_17th_inning
  (A : ℕ)  -- average after the 16th inning
  (h1 : 16 * A + 300 = 17 * (A + 10)) :
  A + 10 = 140 :=
by
  sorry

end batsman_average_after_17th_inning_l231_231082


namespace profit_per_unit_and_minimum_units_l231_231359

noncomputable def conditions (x y m : ℝ) : Prop :=
  2 * x + 7 * y = 41 ∧
  x + 3 * y = 18 ∧
  0.5 * m + 0.3 * (30 - m) ≥ 13.1

theorem profit_per_unit_and_minimum_units (x y m : ℝ) :
  conditions x y m → x = 3 ∧ y = 5 ∧ m ≥ 21 :=
by
  sorry

end profit_per_unit_and_minimum_units_l231_231359


namespace unique_pair_odd_prime_l231_231913

theorem unique_pair_odd_prime (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃! (n m : ℕ), n ≠ m ∧ (2 / p : ℚ) = (1 / n) + (1 / m) ∧ 
  n = (p + 1) / 2 ∧ m = (p * (p + 1)) / 2 :=
by
  sorry

end unique_pair_odd_prime_l231_231913


namespace fixed_point_for_all_parabolas_l231_231428

theorem fixed_point_for_all_parabolas : ∃ (x y : ℝ), (∀ t : ℝ, y = 4 * x^2 + 2 * t * x - 3 * t) ∧ x = 1 ∧ y = 4 :=
by 
  sorry

end fixed_point_for_all_parabolas_l231_231428


namespace value_of_x_plus_y_l231_231964

-- Define the sum of integers from 50 to 60
def sum_integers_50_to_60 : ℤ := List.sum (List.range' 50 (60 - 50 + 1))

-- Calculate the number of even integers from 50 to 60
def count_even_integers_50_to_60 : ℤ := List.length (List.filter (λ n => n % 2 = 0) (List.range' 50 (60 - 50 + 1)))

-- Define x and y based on the given conditions
def x : ℤ := sum_integers_50_to_60
def y : ℤ := count_even_integers_50_to_60

-- The main theorem to prove
theorem value_of_x_plus_y : x + y = 611 := by
  -- Placeholder for the proof
  sorry

end value_of_x_plus_y_l231_231964


namespace smallest_y_l231_231316

theorem smallest_y (y : ℕ) : 
    (y % 5 = 4) ∧ 
    (y % 7 = 6) ∧ 
    (y % 8 = 7) → 
    y = 279 :=
sorry

end smallest_y_l231_231316


namespace roots_quartic_sum_l231_231013

theorem roots_quartic_sum (c d : ℝ) (h1 : c + d = 3) (h2 : c * d = 1) (hc : Polynomial.eval c (Polynomial.C (-1) + Polynomial.X ^ 4 - 6 * Polynomial.X ^ 3 - 4 * Polynomial.X) = 0) (hd : Polynomial.eval d (Polynomial.C (-1) + Polynomial.X ^ 4 - 6 * Polynomial.X ^ 3 - 4 * Polynomial.X) = 0) :
  c * d + c + d = 4 :=
by
  sorry

end roots_quartic_sum_l231_231013


namespace triangle_rectangle_ratio_l231_231977

-- Definitions of the perimeter conditions and the relationship between length and width of the rectangle.
def equilateral_triangle_side_length (t : ℕ) : Prop :=
  3 * t = 24

def rectangle_dimensions (l w : ℕ) : Prop :=
  2 * l + 2 * w = 24 ∧ l = 2 * w

-- The main theorem stating the desired ratio.
theorem triangle_rectangle_ratio (t l w : ℕ) 
  (ht : equilateral_triangle_side_length t) (hlw : rectangle_dimensions l w) : t / w = 2 :=
by
  sorry

end triangle_rectangle_ratio_l231_231977


namespace incorrect_option_c_l231_231790

theorem incorrect_option_c (R : ℝ) : 
  let cylinder_lateral_area := 4 * π * R^2
  let sphere_surface_area := 4 * π * R^2
  cylinder_lateral_area = sphere_surface_area :=
  sorry

end incorrect_option_c_l231_231790


namespace taco_cost_l231_231070

theorem taco_cost (T E : ℝ) (h1 : 2 * T + 3 * E = 7.80) (h2 : 3 * T + 5 * E = 12.70) : T = 0.90 := 
by 
  sorry

end taco_cost_l231_231070


namespace gcd_of_1230_and_990_l231_231535

theorem gcd_of_1230_and_990 : Nat.gcd 1230 990 = 30 :=
by
  sorry

end gcd_of_1230_and_990_l231_231535


namespace sum_infinite_series_l231_231512

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l231_231512


namespace jellybeans_count_l231_231908

noncomputable def jellybeans_initial (y: ℝ) (n: ℕ) : ℝ :=
  y / (0.7 ^ n)

theorem jellybeans_count (y x: ℝ) (n: ℕ) (h: y = 24) (h2: n = 3) :
  x = 70 :=
by
  apply sorry

end jellybeans_count_l231_231908


namespace bicycle_cost_l231_231075

theorem bicycle_cost (CP_A SP_B SP_C : ℝ) (h1 : SP_B = CP_A * 1.20) (h2 : SP_C = SP_B * 1.25) (h3 : SP_C = 225) : CP_A = 150 :=
by
  sorry

end bicycle_cost_l231_231075


namespace Jose_share_land_l231_231337

theorem Jose_share_land (total_land : ℕ) (num_siblings : ℕ) (total_parts : ℕ) (share_per_person : ℕ) :
  total_land = 20000 → num_siblings = 4 → total_parts = (1 + num_siblings) → share_per_person = (total_land / total_parts) → 
  share_per_person = 4000 :=
by
  sorry

end Jose_share_land_l231_231337


namespace groceries_value_l231_231408

-- Conditions
def alex_saved : ℝ := 14500
def car_cost : ℝ := 14600
def trip_charge : ℝ := 1.5
def earn_percentage : ℝ := 0.05
def num_trips : ℝ := 40

-- Proof Statement
theorem groceries_value
  (alex_saved : ℝ)
  (car_cost : ℝ)
  (trip_charge : ℝ)
  (earn_percentage : ℝ)
  (num_trips : ℝ)
  (h_saved : alex_saved = 14500)
  (h_car_cost : car_cost = 14600)
  (h_trip_charge : trip_charge = 1.5)
  (h_earn_percentage : earn_percentage = 0.05)
  (h_num_trips : num_trips = 40) :

  let needed_savings := car_cost - alex_saved in
  let earnings_from_trips := num_trips * trip_charge in
  let earnings_from_groceries := needed_savings - earnings_from_trips in
  let total_value_of_groceries := earnings_from_groceries / earn_percentage in
  total_value_of_groceries = 800 := by {
    sorry
  }

end groceries_value_l231_231408


namespace nonnegative_solution_positive_solution_l231_231596

/-- For k > 7, there exist non-negative integers x and y such that 5*x + 3*y = k. -/
theorem nonnegative_solution (k : ℤ) (hk : k > 7) : ∃ x y : ℕ, 5 * x + 3 * y = k :=
sorry

/-- For k > 15, there exist positive integers x and y such that 5*x + 3*y = k. -/
theorem positive_solution (k : ℤ) (hk : k > 15) : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 5 * x + 3 * y = k :=
sorry

end nonnegative_solution_positive_solution_l231_231596


namespace find_c_for_equal_real_roots_l231_231579

theorem find_c_for_equal_real_roots
  (c : ℝ)
  (h : ∀ x : ℝ, x^2 + 6 * x + c = 0 → x = -3) : c = 9 :=
sorry

end find_c_for_equal_real_roots_l231_231579


namespace probability_first_queen_second_diamond_l231_231060

/-- 
Given that two cards are dealt at random from a standard deck of 52 cards,
the probability that the first card is a Queen and the second card is a Diamonds suit is 1/52.
-/
theorem probability_first_queen_second_diamond : 
  let total_cards := 52
  let probability (num events : ℕ) := (events : ℝ) / (num + 0 : ℝ)
  let prob_first_queen := probability total_cards 4
  let prob_second_diamond_if_first_queen_diamond := probability (total_cards - 1) 12
  let prob_second_diamond_if_first_other_queen := probability (total_cards - 1) 13
  let prob_first_queen_diamond := probability total_cards 1
  let prob_first_queen_other := probability total_cards 3
  let joint_prob_first_queen_diamond := prob_first_queen_diamond * prob_second_diamond_if_first_queen_diamond
  let joint_prob_first_queen_other := prob_first_queen_other * prob_second_diamond_if_first_other_queen
  let total_probability := joint_prob_first_queen_diamond + joint_prob_first_queen_other
  total_probability = probability total_cards 1 := 
by
  -- Proof goes here
  sorry

end probability_first_queen_second_diamond_l231_231060


namespace second_number_is_correct_l231_231966

theorem second_number_is_correct (x : Real) (h : 108^2 + x^2 = 19928) : x = Real.sqrt 8264 :=
by
  sorry

end second_number_is_correct_l231_231966


namespace scientific_notation_proof_l231_231036

-- Given number is 657,000
def number : ℕ := 657000

-- Scientific notation of the given number
def scientific_notation (n : ℕ) : Prop :=
    n = 657000 ∧ (6.57 : ℝ) * (10 : ℝ)^5 = 657000

theorem scientific_notation_proof : scientific_notation number :=
by 
  sorry

end scientific_notation_proof_l231_231036


namespace meaningful_fraction_l231_231230

theorem meaningful_fraction {x : ℝ} : (x - 2) ≠ 0 ↔ x ≠ 2 :=
by
  sorry

end meaningful_fraction_l231_231230


namespace function_characterization_l231_231707
noncomputable def f : ℕ → ℕ := sorry

theorem function_characterization (h : ∀ m n : ℕ, m^2 + f n ∣ m * f m + n) : 
  ∀ n : ℕ, f n = n :=
by
  intro n
  sorry

end function_characterization_l231_231707


namespace no_solution_exists_l231_231115

def product_of_digits (x : ℕ) : ℕ :=
  if x < 10 then x else (x / 10) * (x % 10)

theorem no_solution_exists :
  ¬ ∃ x : ℕ, product_of_digits x = x^2 - 10 * x - 22 :=
by
  sorry

end no_solution_exists_l231_231115


namespace series_sum_eq_four_ninths_l231_231498

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, k / (4 : ℝ) ^ (k + 1)

theorem series_sum_eq_four_ninths : sum_series = 4 / 9 := 
sorry

end series_sum_eq_four_ninths_l231_231498


namespace election_ratio_l231_231903

theorem election_ratio (X Y : ℝ) 
  (h : 0.74 * X + 0.5000000000000002 * Y = 0.66 * (X + Y)) : 
  X / Y = 2 :=
by sorry

end election_ratio_l231_231903


namespace actual_time_is_1240pm_l231_231280

def kitchen_and_cellphone_start (t : ℕ) : Prop := t = 8 * 60  -- 8:00 AM in minutes
def kitchen_clock_after_breakfast (t : ℕ) : Prop := t = 8 * 60 + 30  -- 8:30 AM in minutes
def cellphone_after_breakfast (t : ℕ) : Prop := t = 8 * 60 + 20  -- 8:20 AM in minutes
def kitchen_clock_at_3pm (t : ℕ) : Prop := t = 15 * 60  -- 3:00 PM in minutes

theorem actual_time_is_1240pm : 
  (kitchen_and_cellphone_start 480) ∧ 
  (kitchen_clock_after_breakfast 510) ∧ 
  (cellphone_after_breakfast 500) ∧
  (kitchen_clock_at_3pm 900) → 
  real_time_at_kitchen_clock_time_3pm = 12 * 60 + 40 :=
by
  sorry

end actual_time_is_1240pm_l231_231280


namespace find_exact_speed_l231_231772

variable (d t v : ℝ)

-- Conditions as Lean definitions
def distance_eq1 : d = 50 * (t - 1/12) := sorry
def distance_eq2 : d = 70 * (t + 1/12) := sorry
def travel_time : t = 1/2 := sorry -- deduced travel time from the equations and given conditions
def correct_speed : v = 42 := sorry -- Mr. Bird needs to drive at 42 mph to be exactly on time

-- Lean 4 statement proving the required speed is 42 mph
theorem find_exact_speed : v = d / t :=
  by
    sorry

end find_exact_speed_l231_231772


namespace explicit_form_of_function_l231_231726

theorem explicit_form_of_function (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x * f x + f x * f y + y - 1) = f (x * f x + x * y) + y - 1) : 
  ∀ x : ℝ, f x = x :=
by
  sorry

end explicit_form_of_function_l231_231726


namespace jo_climb_stairs_ways_l231_231855

def f : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| (n + 3) => f (n + 2) + f (n + 1) + f n

theorem jo_climb_stairs_ways : f 8 = 81 :=
by
    sorry

end jo_climb_stairs_ways_l231_231855


namespace intersection_A_B_l231_231880

open Set

variable (x : ℝ)

def A : Set ℝ := {x | x^2 - 3 * x - 4 < 0}
def B : Set ℝ := {-4, 1, 3, 5}

theorem intersection_A_B :
  A ∩ B = { 1, 3 } :=
sorry

end intersection_A_B_l231_231880


namespace partial_fraction_sum_zero_l231_231109

theorem partial_fraction_sum_zero (A B C D E F : ℚ) :
  (∀ x : ℚ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -4 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
      A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
sorry

end partial_fraction_sum_zero_l231_231109


namespace ratio_of_triangle_side_to_rectangle_width_l231_231981

variables (t w l : ℕ)

-- Condition 1: The perimeter of the equilateral triangle is 24 inches
def triangle_perimeter := 3 * t = 24

-- Condition 2: The perimeter of the rectangle is 24 inches
def rectangle_perimeter := 2 * l + 2 * w = 24

-- Condition 3: The length of the rectangle is twice its width
def length_double_width := l = 2 * w

-- The ratio of the side length of the triangle to the width of the rectangle is 2
theorem ratio_of_triangle_side_to_rectangle_width
    (h_triangle : triangle_perimeter t)
    (h_rectangle : rectangle_perimeter l w)
    (h_length_width : length_double_width l w) :
    t / w = 2 :=
by
    sorry

end ratio_of_triangle_side_to_rectangle_width_l231_231981


namespace negation_exists_zero_product_l231_231623

variable {R : Type} [LinearOrderedField R]

variable (f g : R → R)

theorem negation_exists_zero_product :
  (¬ ∃ x : R, f x * g x = 0) ↔ ∀ x : R, f x ≠ 0 ∧ g x ≠ 0 :=
by
  sorry

end negation_exists_zero_product_l231_231623


namespace find_a_l231_231797

-- Define the curve y = x^2 + x
def curve (x : ℝ) : ℝ := x^2 + x

-- Line equation ax - y + 1 = 0
def line (a : ℝ) (x y : ℝ) : Prop := a * x - y + 1 = 0

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, line a x y → y = x^2 + x) ∧
  (deriv curve 1 = 2 * 1 + 1) →
  (2 * 1 + 1 = -1 / a) →
  a = -1 / 3 :=
by
  sorry

end find_a_l231_231797


namespace find_values_of_real_numbers_l231_231549

theorem find_values_of_real_numbers (x y : ℝ)
  (h : 2 * x - 1 + (y + 1) * Complex.I = x - y - (x + y) * Complex.I) :
  x = 3 ∧ y = -2 :=
sorry

end find_values_of_real_numbers_l231_231549


namespace roots_diff_l231_231709

theorem roots_diff (m : ℝ) : 
  (∃ α β : ℝ, 2 * α * α - m * α - 8 = 0 ∧ 
              2 * β * β - m * β - 8 = 0 ∧ 
              α ≠ β ∧ 
              α - β = m - 1) ↔ (m = 6 ∨ m = -10 / 3) :=
by
  sorry

end roots_diff_l231_231709


namespace evaluate_series_sum_l231_231440

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l231_231440


namespace no_product_equal_remainder_l231_231984

theorem no_product_equal_remainder (n : ℤ) : 
  ¬ (n = (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) ∨
     (n + 1) = n * (n + 2) * (n + 3) * (n + 4) * (n + 5) ∨
     (n + 2) = n * (n + 1) * (n + 3) * (n + 4) * (n + 5) ∨
     (n + 3) = n * (n + 1) * (n + 2) * (n + 4) * (n + 5) ∨
     (n + 4) = n * (n + 1) * (n + 2) * (n + 3) * (n + 5) ∨
     (n + 5) = n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by sorry

end no_product_equal_remainder_l231_231984


namespace oil_quantity_relationship_l231_231168

variable (Q : ℝ) (t : ℝ)

-- Initial quantity of oil in the tank
def initial_quantity := 40

-- Flow rate of oil out of the tank
def flow_rate := 0.2

-- Function relationship between remaining oil quantity Q and time t
theorem oil_quantity_relationship : Q = initial_quantity - flow_rate * t :=
sorry

end oil_quantity_relationship_l231_231168


namespace scientific_notation_correct_l231_231033

theorem scientific_notation_correct : 657000 = 6.57 * 10^5 :=
by
  sorry

end scientific_notation_correct_l231_231033


namespace recurring_division_l231_231811

def recurring_to_fraction (recurring: ℝ) (part: ℝ): ℝ :=
  part * recurring

theorem recurring_division (recurring: ℝ) (part1 part2: ℝ):
  recurring_to_fraction recurring part1 = 0.63 →
  recurring_to_fraction recurring part2 = 0.18 →
  recurring ≠ 0 →
  (0.63:ℝ)/0.18 = (7:ℝ)/2 :=
by
  intros h1 h2 h3
  rw [recurring_to_fraction] at h1 h2
  sorry

end recurring_division_l231_231811


namespace sharks_at_newport_l231_231279

theorem sharks_at_newport :
  ∃ (x : ℕ), (∃ (y : ℕ), y = 4 * x ∧ x + y = 110) ∧ x = 22 :=
by {
  sorry
}

end sharks_at_newport_l231_231279


namespace ice_cream_sales_l231_231028

theorem ice_cream_sales : 
  let tuesday_sales := 12000
  let wednesday_sales := 2 * tuesday_sales
  let total_sales := tuesday_sales + wednesday_sales
  total_sales = 36000 := 
by 
  sorry

end ice_cream_sales_l231_231028


namespace number_of_TVs_in_shop_c_l231_231954

theorem number_of_TVs_in_shop_c 
  (a b d e : ℕ) 
  (avg : ℕ) 
  (num_shops : ℕ) 
  (total_TVs_in_other_shops : ℕ) 
  (total_TVs : ℕ) 
  (sum_shops : a + b + d + e = total_TVs_in_other_shops) 
  (avg_sets : avg = total_TVs / num_shops) 
  (number_shops : num_shops = 5)
  (avg_value : avg = 48)
  (T_a : a = 20) 
  (T_b : b = 30) 
  (T_d : d = 80) 
  (T_e : e = 50) 
  : (total_TVs - total_TVs_in_other_shops = 60) := 
by 
  sorry

end number_of_TVs_in_shop_c_l231_231954


namespace distance_PF_l231_231078

-- Definitions for the given conditions
structure Rectangle :=
  (EF GH: ℝ)
  (interior_point : ℝ × ℝ)
  (PE : ℝ)
  (PH : ℝ)
  (PG : ℝ)

-- The theorem to prove PF equals 12 under the given conditions
theorem distance_PF 
  (r : Rectangle)
  (hPE : r.PE = 5)
  (hPH : r.PH = 12)
  (hPG : r.PG = 13) :
  ∃ PF, PF = 12 := 
sorry

end distance_PF_l231_231078


namespace b11_eq_4_l231_231595

variables {a : ℕ → ℤ} {b : ℕ → ℤ} {d r : ℤ} {a1 : ℤ}

-- Define non-zero arithmetic sequence {a_n} with common difference d
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define geometric sequence {b_n} with common ratio r
def is_geometric_sequence (b : ℕ → ℤ) (r : ℤ) : Prop :=
  ∀ n, b (n + 1) = b n * r

-- The given conditions
axiom a1_minus_a7_sq_plus_a13_eq_zero : a 1 - (a 7) ^ 2 + a 13 = 0
axiom b7_eq_a7 : b 7 = a 7

-- The problem statement to prove: b 11 = 4
theorem b11_eq_4
  (arith_seq : is_arithmetic_sequence a d)
  (geom_seq : is_geometric_sequence b r)
  (a1_non_zero : a1 ≠ 0) :
  b 11 = 4 :=
sorry

end b11_eq_4_l231_231595


namespace central_angle_of_cone_l231_231947

theorem central_angle_of_cone (A : ℝ) (l : ℝ) (r : ℝ) (θ : ℝ)
  (hA : A = (1 / 2) * 2 * Real.pi * r)
  (hl : l = 1)
  (ha : A = (3 / 8) * Real.pi) :
  θ = (3 / 4) * Real.pi :=
by
  sorry

end central_angle_of_cone_l231_231947


namespace half_sum_of_squares_of_even_or_odd_l231_231781

theorem half_sum_of_squares_of_even_or_odd (n1 n2 : ℤ) (a b : ℤ) :
  (n1 % 2 = 0 ∧ n2 % 2 = 0 ∧ n1 = 2*a ∧ n2 = 2*b ∨
   n1 % 2 = 1 ∧ n2 % 2 = 1 ∧ n1 = 2*a + 1 ∧ n2 = 2*b + 1) →
  ∃ x y : ℤ, (n1^2 + n2^2) / 2 = x^2 + y^2 :=
by
  intro h
  sorry

end half_sum_of_squares_of_even_or_odd_l231_231781


namespace sum_series_eq_4_div_9_l231_231452

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l231_231452


namespace sum_infinite_series_l231_231510

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l231_231510


namespace sum_of_series_l231_231460

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l231_231460


namespace product_has_trailing_zeros_l231_231228

theorem product_has_trailing_zeros (a b : ℕ) (h1 : a = 350) (h2 : b = 60) :
  ∃ (n : ℕ), (10^n ∣ a * b) ∧ n = 3 :=
by
  sorry

end product_has_trailing_zeros_l231_231228


namespace complete_sets_characterization_l231_231766

-- Definition of a complete set
def complete_set (A : Set ℕ) : Prop :=
  ∀ {a b : ℕ}, (a + b ∈ A) → (a * b ∈ A)

-- Theorem stating that the complete sets of natural numbers are exactly
-- {1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}, ℕ.
theorem complete_sets_characterization :
  ∀ (A : Set ℕ), complete_set A ↔ (A = {1} ∨ A = {1, 2} ∨ A = {1, 2, 3} ∨ A = {1, 2, 3, 4} ∨ A = Set.univ) :=
sorry

end complete_sets_characterization_l231_231766


namespace fraction_product_l231_231423

theorem fraction_product : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 :=
by
  sorry

end fraction_product_l231_231423


namespace sharp_sharp_sharp_20_l231_231845

def sharp (N : ℝ) : ℝ := (0.5 * N)^2 + 1

theorem sharp_sharp_sharp_20 : sharp (sharp (sharp 20)) = 1627102.64 :=
by
  sorry

end sharp_sharp_sharp_20_l231_231845


namespace intersection_of_sets_l231_231877
-- Define the sets and the proof statement
theorem intersection_of_sets : 
  let A := { x : ℝ | x^2 - 3 * x - 4 < 0 }
  let B := {-4, 1, 3, 5}
  A ∩ B = {1, 3} :=
by
  sorry

end intersection_of_sets_l231_231877


namespace summer_camp_activity_l231_231171

theorem summer_camp_activity :
  ∃ (a b c d e f : ℕ), 
  a + b + c + d + 3 * e + 4 * f = 12 ∧ 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧
  f = 1 := by
  sorry

end summer_camp_activity_l231_231171


namespace solve_for_x_l231_231850

theorem solve_for_x : ∃ x : ℚ, 5 * (2 * x - 3) = 3 * (3 - 4 * x) + 15 ∧ x = (39 : ℚ) / 22 :=
by
  use (39 : ℚ) / 22
  sorry

end solve_for_x_l231_231850


namespace blue_marbles_in_bag_l231_231829

theorem blue_marbles_in_bag
  (total_marbles : ℕ)
  (red_marbles : ℕ)
  (prob_red_white : ℚ)
  (number_red_marbles: red_marbles = 9) 
  (total_marbles_eq: total_marbles = 30) 
  (prob_red_white_eq: prob_red_white = 5/6): 
  ∃ (blue_marbles : ℕ), blue_marbles = 5 :=
by
  have W := 16        -- This is from (9 + W)/30 = 5/6 which gives W = 16
  let B := total_marbles - red_marbles - W
  use B
  have h : B = 30 - 9 - 16 := by
    -- Remaining calculations
    sorry
  exact h

end blue_marbles_in_bag_l231_231829


namespace sum_of_series_l231_231465

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l231_231465


namespace sum_of_angles_is_90_l231_231380

variables (α β γ : ℝ)
-- Given angles marked on squared paper, which imply certain geometric properties
axiom angle_properties : α + β + γ = 90

theorem sum_of_angles_is_90 : α + β + γ = 90 := 
by
  apply angle_properties

end sum_of_angles_is_90_l231_231380


namespace probability_one_each_l231_231844

-- Define the counts of letters
def total_letters : ℕ := 11
def cybil_count : ℕ := 5
def ronda_count : ℕ := 5
def andy_initial_count : ℕ := 1

-- Define the probability calculation
def probability_one_from_cybil_and_one_from_ronda : ℚ :=
  (cybil_count / total_letters) * (ronda_count / (total_letters - 1)) +
  (ronda_count / total_letters) * (cybil_count / (total_letters - 1))

theorem probability_one_each (total_letters cybil_count ronda_count andy_initial_count : ℕ) :
  probability_one_from_cybil_and_one_from_ronda = 5 / 11 := sorry

end probability_one_each_l231_231844


namespace line_no_intersect_parabola_range_l231_231183

def parabola_eq (x : ℝ) : ℝ := x^2 + 4

def line_eq (m x : ℝ) : ℝ := m * (x - 10) + 6

theorem line_no_intersect_parabola_range (r s m : ℝ) :
  (m^2 - 40 * m + 8 = 0) →
  r < s →
  (∀ x, parabola_eq x ≠ line_eq m x) →
  r + s = 40 :=
by
  sorry

end line_no_intersect_parabola_range_l231_231183


namespace fewer_onions_correct_l231_231271

-- Define the quantities
def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

-- Calculate the total number of tomatoes and corn
def tomatoes_and_corn : ℕ := tomatoes + corn

-- Calculate the number of fewer onions
def fewer_onions : ℕ := tomatoes_and_corn - onions

-- State the theorem and provide the proof
theorem fewer_onions_correct : fewer_onions = 5200 :=
by
  -- The statement is proved directly by the calculations above
  -- Providing the actual proof is not necessary as per the guidelines
  sorry

end fewer_onions_correct_l231_231271


namespace sum_series_eq_four_ninths_l231_231485

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l231_231485


namespace Bomi_change_l231_231840

def candy_cost : ℕ := 350
def chocolate_cost : ℕ := 500
def total_paid : ℕ := 1000
def total_cost := candy_cost + chocolate_cost
def change := total_paid - total_cost

theorem Bomi_change : change = 150 :=
by
  -- Here we would normally provide the proof steps.
  sorry

end Bomi_change_l231_231840


namespace problem1_problem2_problem3_problem4_l231_231985

section
  variable (a b c d : Int)

  theorem problem1 : -27 + (-32) + (-8) + 72 = 5 := by
    sorry

  theorem problem2 : -4 - 2 * 32 + (-2 * 32) = -132 := by
    sorry

  theorem problem3 : (-48 : Int) / (-2 : Int)^3 - (-25 : Int) * (-4 : Int) + (-2 : Int)^3 = -102 := by
    sorry

  theorem problem4 : (-3 : Int)^2 - (3 / 2)^3 * (2 / 9) - 6 / (-(2 / 3))^3 = -12 := by
    sorry
end

end problem1_problem2_problem3_problem4_l231_231985


namespace could_not_be_diagonal_lengths_l231_231667

-- Definitions of the diagonal conditions
def diagonal_condition (s : List ℕ) : Prop :=
  match s with
  | [x, y, z] => x^2 + y^2 > z^2 ∧ x^2 + z^2 > y^2 ∧ y^2 + z^2 > x^2
  | _ => false

-- Statement of the problem
theorem could_not_be_diagonal_lengths : 
  ¬ diagonal_condition [5, 6, 8] :=
by 
  sorry

end could_not_be_diagonal_lengths_l231_231667


namespace extremum_condition_l231_231184

noncomputable def y (a x : ℝ) : ℝ := Real.exp (a * x) + 3 * x

theorem extremum_condition (a : ℝ) :
  (∃ x : ℝ, (y a x = 0) ∧ ∀ x' > x, y a x' < y a x) → a < -3 :=
by
  sorry

end extremum_condition_l231_231184


namespace correct_calculation_l231_231069

theorem correct_calculation (x y a b : ℝ) :
  (3*x + 3*y ≠ 6*x*y) ∧
  (x + x ≠ x^2) ∧
  (-9*y^2 + 16*y^2 ≠ 7) ∧
  (9*a^2*b - 9*a^2*b = 0) :=
by
  sorry

end correct_calculation_l231_231069


namespace evaluate_series_sum_l231_231441

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l231_231441


namespace sin_neg_600_eq_sqrt_3_div_2_l231_231430

theorem sin_neg_600_eq_sqrt_3_div_2 :
  Real.sin (-(600 * Real.pi / 180)) = Real.sqrt 3 / 2 :=
sorry

end sin_neg_600_eq_sqrt_3_div_2_l231_231430


namespace unique_pair_exists_l231_231917

theorem unique_pair_exists (p : ℕ) (hp : p.prime ) (hodd : p % 2 = 1) : 
  ∃ m n : ℕ, m ≠ n ∧ (2 : ℚ) / (p : ℚ) = 1 / (n : ℚ) + 1 / (m : ℚ) ∧ 
             (n = (p + 1) / 2) ∧ (m = (p * (p + 1)) / 2) :=
by
  sorry

end unique_pair_exists_l231_231917


namespace option_C_l231_231102

theorem option_C (a b c : ℝ) (h₀ : a > b) (h₁ : b > c) (h₂ : c > 0) :
  (b + c) / (a + c) > b / a :=
sorry

end option_C_l231_231102


namespace mortgage_loan_amount_l231_231564

/-- Given the initial payment is 1,800,000 rubles and it represents 30% of the property cost C, 
    prove that the mortgage loan amount is 4,200,000 rubles. -/
theorem mortgage_loan_amount (C : ℝ) (h : 0.3 * C = 1800000) : C - 1800000 = 4200000 :=
by
  sorry

end mortgage_loan_amount_l231_231564


namespace complex_third_quadrant_l231_231135

-- Define the imaginary unit i.
def i : ℂ := Complex.I 

-- Define the complex number z = i * (1 + i).
def z : ℂ := i * (1 + i)

-- Prove that z lies in the third quadrant.
theorem complex_third_quadrant : z.re < 0 ∧ z.im < 0 := 
by
  sorry

end complex_third_quadrant_l231_231135


namespace triangle_inequality_inequality_l231_231194

variable {a b c : ℝ}
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (triangle_ineq : a + b > c)

theorem triangle_inequality_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) (triangle_ineq : a + b > c) :
  a^3 + b^3 + 3 * a * b * c > c^3 :=
sorry

end triangle_inequality_inequality_l231_231194


namespace daily_salmon_l231_231853

-- Definitions of the daily consumption of trout and total fish
def daily_trout : ℝ := 0.2
def daily_total_fish : ℝ := 0.6

-- Theorem statement that the daily consumption of salmon is 0.4 buckets
theorem daily_salmon : daily_total_fish - daily_trout = 0.4 := 
by
  -- Skipping the proof, as required
  sorry

end daily_salmon_l231_231853


namespace infinite_series_sum_l231_231522

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l231_231522


namespace least_possible_value_l231_231958

noncomputable def min_value_expression : ℝ :=
  let f (x : ℝ) := (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2164
  have : (∀ x, f x ≥ 2161.75) := sorry,
  infi f

theorem least_possible_value : ∃ x : ℝ, (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2164 = 2161.75 ∧ 
  (∀ x : ℝ, (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2164 ≥ 2161.75) :=
sorry

end least_possible_value_l231_231958


namespace correct_number_of_statements_l231_231371

-- Definitions based on the problem's conditions
def condition_1 : Prop :=
  ∀ (n : ℕ) (a b c d e : ℚ), n = 5 ∧ ∃ x y z, (x = a ∧ y = b ∧ z = c) ∧ (x < 0 ∧ y < 0 ∧ z < 0 ∧ d ≥ 0 ∧ e ≥ 0) →
  (a * b * c * d * e < 0 ∨ a * b * c * d * e = 0)

def condition_2 : Prop := 
  ∀ m : ℝ, |m| + m = 0 → m ≤ 0

def condition_3 : Prop := 
  ∀ a b : ℝ, (1 / a < 1 / b) → ¬ (a < b ∨ b < a)

def condition_4 : Prop := 
  ∀ a : ℝ, ∃ max_val, max_val = 5 ∧ 5 - |a - 5| ≤ max_val

-- Main theorem to state the correct number of true statements
theorem correct_number_of_statements : 
  (condition_2 ∧ condition_4) ∧
  ¬condition_1 ∧ 
  ¬condition_3 :=
by
  sorry

end correct_number_of_statements_l231_231371


namespace infinite_series_sum_l231_231516

theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ) * (1 / 4) ^ k = 4 / 9 := by
  sorry

end infinite_series_sum_l231_231516


namespace total_children_in_circle_l231_231784

theorem total_children_in_circle 
  (n : ℕ)  -- number of children
  (h_even : Even n)   -- condition: the circle is made up of an even number of children
  (h_pos : n > 0) -- condition: there are some children
  (h_opposite : (15 % n + 15 % n) % n = 0)  -- condition: the 15th child clockwise from Child A is facing Child A (implies opposite)
  : n = 30 := 
sorry

end total_children_in_circle_l231_231784


namespace find_square_divisible_by_four_l231_231532

/-- There exists an x such that x is a square number, x is divisible by four, 
and 39 < x < 80, and that x = 64 is such a number. --/
theorem find_square_divisible_by_four : ∃ (x : ℕ), (∃ (n : ℕ), x = n^2) ∧ (x % 4 = 0) ∧ (39 < x ∧ x < 80) ∧ x = 64 :=
  sorry

end find_square_divisible_by_four_l231_231532


namespace prob1_prob2_odd_prob2_monotonic_prob3_l231_231721

variable (a : ℝ) (f : ℝ → ℝ)
variable (hf : ∀ x : ℝ, f (log a x) = a / (a^2 - 1) * (x - 1 / x))
variable (ha : 0 < a ∧ a < 1)

-- Problem 1: Prove the expression for f(x)
theorem prob1 (x : ℝ) : f x = a / (a^2 - 1) * (a^x - a^(-x)) := sorry

-- Problem 2: Prove oddness and monotonicity of f(x)
theorem prob2_odd : ∀ x, f (-x) = -f x := sorry
theorem prob2_monotonic : ∀ x₁ x₂ : ℝ, (x₁ < x₂) → (f x₁ < f x₂) := sorry

-- Problem 3: Determine the range of k
theorem prob3 (k : ℝ) : (∀ t : ℝ, 1 ≤ t ∧ t ≤ 3 → f (3 * t^2 - 1) + f (4 * t - k) > 0) → (k < 6) := sorry

end prob1_prob2_odd_prob2_monotonic_prob3_l231_231721


namespace simplify_expression_l231_231957

theorem simplify_expression :
  (4 + 2 + 6) / 3 - (2 + 1) / 3 = 3 := by
  sorry

end simplify_expression_l231_231957


namespace complex_number_sum_equals_one_l231_231186

variable {a b c d : ℝ}
variable {ω : ℂ}

theorem complex_number_sum_equals_one
  (ha : a ≠ -1) 
  (hb : b ≠ -1) 
  (hc : c ≠ -1) 
  (hd : d ≠ -1) 
  (hω : ω^4 = 1) 
  (hω_ne : ω ≠ 1)
  (h_eq : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 4 / ω)
  : (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 1 :=
by sorry

end complex_number_sum_equals_one_l231_231186


namespace sum_factors_36_eq_91_l231_231663

theorem sum_factors_36_eq_91 : 
  (∑ (i : ℕ) in ({1, 2, 3, 4, 6, 9, 12, 18, 36} : Finset ℕ), i) = 91 :=
by
  sorry

end sum_factors_36_eq_91_l231_231663


namespace parabola_y_values_order_l231_231302

theorem parabola_y_values_order :
  let y1 := 2 * (-3 - 2) ^ 2 + 1
  let y2 := 2 * (3 - 2) ^ 2 + 1
  let y3 := 2 * (4 - 2) ^ 2 + 1
  y2 < y3 ∧ y3 < y1 :=
by
  let y1 := 2 * (-3 - 2) ^ 2 + 1
  let y2 := 2 * (3 - 2) ^ 2 + 1
  let y3 := 2 * (4 - 2) ^ 2 + 1
  -- The proof is omitted
  sorry

end parabola_y_values_order_l231_231302


namespace original_fraction_is_two_thirds_l231_231740

theorem original_fraction_is_two_thirds
  (x y : ℕ)
  (h1 : x / (y + 1) = 1 / 2)
  (h2 : (x + 1) / y = 1) :
  x / y = 2 / 3 := by
  sorry

end original_fraction_is_two_thirds_l231_231740


namespace nonzero_rational_pow_zero_l231_231812

theorem nonzero_rational_pow_zero 
  (num : ℤ) (denom : ℤ) (hnum : num = -1241376497) (hdenom : denom = 294158749357) (h_nonzero: num ≠ 0 ∧ denom ≠ 0) :
  (num / denom : ℚ) ^ 0 = 1 := 
by 
  sorry

end nonzero_rational_pow_zero_l231_231812


namespace total_amount_spent_l231_231156

-- Define the prices of the CDs
def price_life_journey : ℕ := 100
def price_day_life : ℕ := 50
def price_when_rescind : ℕ := 85

-- Define the discounted price for The Life Journey CD
def discount_life_journey : ℕ := 20 -- 20% discount equivalent to $20
def discounted_price_life_journey : ℕ := price_life_journey - discount_life_journey

-- Define the number of CDs bought
def num_life_journey : ℕ := 3
def num_day_life : ℕ := 4
def num_when_rescind : ℕ := 2

-- Define the function to calculate money spent on each type with offers in consideration
def cost_life_journey : ℕ := num_life_journey * discounted_price_life_journey
def cost_day_life : ℕ := (num_day_life / 2) * price_day_life -- Buy one get one free offer
def cost_when_rescind : ℕ := num_when_rescind * price_when_rescind

-- Calculate the total cost
def total_cost := cost_life_journey + cost_day_life + cost_when_rescind

-- Define Lean theorem to prove the total cost
theorem total_amount_spent : total_cost = 510 :=
  by
    -- Skipping the actual proof as the prompt specifies
    sorry

end total_amount_spent_l231_231156


namespace trig_values_same_terminal_side_l231_231234

-- Statement: The trigonometric function values of angles with the same terminal side are equal.
theorem trig_values_same_terminal_side (θ₁ θ₂ : ℝ) (h : ∃ k : ℤ, θ₂ = θ₁ + 2 * k * π) :
  (∀ f : ℝ -> ℝ, f θ₁ = f θ₂) :=
by
  sorry

end trig_values_same_terminal_side_l231_231234


namespace total_animals_hunted_l231_231580

theorem total_animals_hunted :
  let sam_hunts := 6
  let rob_hunts := sam_hunts / 2
  let total_sam_rob := sam_hunts + rob_hunts
  let mark_hunts := total_sam_rob / 3
  let peter_hunts := mark_hunts * 3
  sam_hunts + rob_hunts + mark_hunts + peter_hunts = 21 :=
by
  sorry

end total_animals_hunted_l231_231580


namespace solve_inequality_l231_231611

theorem solve_inequality (x : ℝ) : 
  (-9 * x^2 + 6 * x + 15 > 0) ↔ (x > -1 ∧ x < 5/3) := 
sorry

end solve_inequality_l231_231611


namespace appropriate_presentation_length_l231_231050

-- Definitions and conditions
def ideal_speaking_rate : ℕ := 160
def min_minutes : ℕ := 20
def max_minutes : ℕ := 40
def appropriate_words_range (words : ℕ) : Prop :=
  words ≥ (min_minutes * ideal_speaking_rate) ∧ words ≤ (max_minutes * ideal_speaking_rate)

-- Statement to prove
theorem appropriate_presentation_length : appropriate_words_range 5000 :=
by sorry

end appropriate_presentation_length_l231_231050


namespace sum_of_series_l231_231459

theorem sum_of_series : 
  (∑ k in (Finset.range 100000).filter (λ n, n > 0), k / 4^k) = 4/9 :=
by sorry

end sum_of_series_l231_231459


namespace pump_rates_l231_231805

theorem pump_rates (x y z : ℝ)
(h1 : x + y + z = 14)
(h2 : z = x + 3)
(h3 : y = 11 - 2 * x)
(h4 : 9 / x = (28 - 2 * y) / z)
: x = 3 ∧ y = 5 ∧ z = 6 :=
by
  sorry

end pump_rates_l231_231805


namespace find_angle_B_l231_231911

noncomputable def angle_B (A B C a b c : ℝ): Prop := 
  a * Real.cos B - b * Real.cos A = b ∧ 
  C = Real.pi / 5

theorem find_angle_B (a b c A B C : ℝ) (h : angle_B A B C a b c) : 
  B = 4 * Real.pi / 15 :=
by
  sorry

end find_angle_B_l231_231911


namespace alloy_mixture_l231_231686

theorem alloy_mixture (x y : ℝ) 
  (h1 : x + y = 1000)
  (h2 : 0.25 * x + 0.50 * y = 450) : 
  x = 200 ∧ y = 800 :=
by
  -- Proof will follow here
  sorry

end alloy_mixture_l231_231686


namespace range_of_a_for_inequality_l231_231133

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ p q : ℝ, (0 < p ∧ p < 1) → (0 < q ∧ q < 1) → p ≠ q → (f a p - f a q) / (p - q) > 1) ↔ 3 ≤ a :=
sorry

end range_of_a_for_inequality_l231_231133


namespace num_students_in_class_l231_231165

-- Define the conditions
variables (S : ℕ) (num_boys : ℕ) (num_boys_under_6ft : ℕ)

-- Assume the conditions given in the problem
axiom two_thirds_boys : num_boys = (2 * S) / 3
axiom three_fourths_under_6ft : num_boys_under_6ft = (3 * num_boys) / 4
axiom nineteen_boys_under_6ft : num_boys_under_6ft = 19

-- The statement we want to prove
theorem num_students_in_class : S = 38 :=
by
  -- Proof omitted (insert proof here)
  sorry

end num_students_in_class_l231_231165


namespace intersection_of_sets_l231_231875
-- Define the sets and the proof statement
theorem intersection_of_sets : 
  let A := { x : ℝ | x^2 - 3 * x - 4 < 0 }
  let B := {-4, 1, 3, 5}
  A ∩ B = {1, 3} :=
by
  sorry

end intersection_of_sets_l231_231875


namespace pool_filling_time_l231_231007

theorem pool_filling_time (rate_jim rate_sue rate_tony : ℝ) (h1 : rate_jim = 1 / 30) (h2 : rate_sue = 1 / 45) (h3 : rate_tony = 1 / 90) : 
     1 / (rate_jim + rate_sue + rate_tony) = 15 := by
  sorry

end pool_filling_time_l231_231007


namespace min_weighings_to_order_four_stones_l231_231105

theorem min_weighings_to_order_four_stones : ∀ (A B C D : ℝ), 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D → ∃ n, n = 5 :=
by sorry

end min_weighings_to_order_four_stones_l231_231105


namespace peach_bun_weight_l231_231838

theorem peach_bun_weight (O triangle : ℕ) 
  (h1 : O = 2 * triangle + 40) 
  (h2 : O + 80 = triangle + 200) : 
  O + triangle = 280 := 
by 
  sorry

end peach_bun_weight_l231_231838


namespace expected_visible_people_l231_231826

-- Definition of expectation of X_n as the sum of the harmonic series.
theorem expected_visible_people (n : ℕ) : 
  (∑ i in finset.range (n) + 1), 1 / (i + 1) = (∑ i in finset.range n, 1 / (i + 1)) :=
by
  sorry

end expected_visible_people_l231_231826


namespace square_tile_area_l231_231434

-- Definition and statement of the problem
theorem square_tile_area (side_length : ℝ) (h : side_length = 7) : 
  (side_length * side_length) = 49 :=
by
  sorry

end square_tile_area_l231_231434


namespace same_color_points_distance_2004_l231_231949

noncomputable def exists_same_color_points_at_distance_2004 (color : ℝ × ℝ → ℕ) : Prop :=
  ∃ (p q : ℝ × ℝ), (p ≠ q) ∧ (color p = color q) ∧ (dist p q = 2004)

/-- The plane is colored in two colors. Prove that there exist two points of the same color at a distance of 2004 meters. -/
theorem same_color_points_distance_2004 {color : ℝ × ℝ → ℕ}
  (hcolor : ∀ p, color p = 1 ∨ color p = 2) :
  exists_same_color_points_at_distance_2004 color :=
sorry

end same_color_points_distance_2004_l231_231949


namespace volume_cube_box_for_pyramid_l231_231357

theorem volume_cube_box_for_pyramid (h_pyramid : height_of_pyramid = 18) 
  (base_side_pyramid : side_of_square_base = 15) : 
  volume_of_box = 18^3 :=
by
  sorry

end volume_cube_box_for_pyramid_l231_231357


namespace determine_c_l231_231744

noncomputable def ab5c_decimal (a b c : ℕ) : ℕ :=
  729 * a + 81 * b + 45 + c

theorem determine_c (a b c : ℕ) (h₁ : a ≠ 0) (h₂ : ∃ k : ℕ, ab5c_decimal a b c = k^2) :
  c = 0 ∨ c = 7 :=
by
  sorry

end determine_c_l231_231744


namespace ratio_of_art_to_math_books_l231_231181

-- The conditions provided
def total_budget : ℝ := 500
def price_math_book : ℝ := 20
def num_math_books : ℕ := 4
def num_art_books : ℕ := num_math_books
def price_art_book : ℝ := 20
def num_science_books : ℕ := num_math_books + 6
def price_science_book : ℝ := 10
def cost_music_books : ℝ := 160

-- Desired proof statement
theorem ratio_of_art_to_math_books : num_art_books / num_math_books = 1 :=
by
  sorry

end ratio_of_art_to_math_books_l231_231181


namespace division_of_203_by_single_digit_l231_231390

theorem division_of_203_by_single_digit (d : ℕ) (h : 1 ≤ d ∧ d < 10) : 
  ∃ q : ℕ, q = 203 / d ∧ (10 ≤ q ∧ q < 100 ∨ 100 ≤ q ∧ q < 1000) := 
by
  sorry

end division_of_203_by_single_digit_l231_231390


namespace find_a_b_c_eq_32_l231_231287

variables {a b c : ℤ}

theorem find_a_b_c_eq_32
  (h1 : ∃ a b : ℤ, x^2 + 19 * x + 88 = (x + a) * (x + b))
  (h2 : ∃ b c : ℤ, x^2 - 21 * x + 108 = (x - b) * (x - c)) :
  a + b + c = 32 :=
sorry

end find_a_b_c_eq_32_l231_231287


namespace infinite_perfect_squares_in_arithmetic_sequence_l231_231604

theorem infinite_perfect_squares_in_arithmetic_sequence 
  (a d : ℕ) 
  (h_exists_perfect_square : ∃ (n₀ k : ℕ), a + n₀ * d = k^2) 
  : ∃ (S : ℕ → ℕ), (∀ n, ∃ t, S n = a + t * d ∧ ∃ k, S n = k^2) ∧ (∀ m n, S m = S n → m = n) :=
sorry

end infinite_perfect_squares_in_arithmetic_sequence_l231_231604


namespace fraction_problem_l231_231736

theorem fraction_problem (m n p q : ℚ) 
  (h1 : m / n = 20) 
  (h2 : p / n = 5) 
  (h3 : p / q = 1 / 15) : 
  m / q = 4 / 15 :=
sorry

end fraction_problem_l231_231736


namespace fraction_problem_l231_231735

theorem fraction_problem (m n p q : ℚ) 
  (h1 : m / n = 20) 
  (h2 : p / n = 5) 
  (h3 : p / q = 1 / 15) : 
  m / q = 4 / 15 :=
sorry

end fraction_problem_l231_231735


namespace evaluate_series_sum_l231_231435

noncomputable def geometric_series_sum : ℝ :=
  ∑' k, (k + 1 : ℝ) / (2^(k+1))

theorem evaluate_series_sum:
  (∑' k, ((k + 1 : ℝ) / (4^(k + 1)))) = (4 / 9) := 
sorry

end evaluate_series_sum_l231_231435


namespace cosine_angle_between_vectors_l231_231562

noncomputable def vector_cosine (a b : ℝ × ℝ) : ℝ :=
  let dot_product := (a.1 * b.1 + a.2 * b.2)
  let magnitude_a := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)
  let magnitude_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dot_product / (magnitude_a * magnitude_b)

theorem cosine_angle_between_vectors : ∀ (k : ℝ), 
  let a := (3, 1)
  let b := (1, 3)
  let c := (k, -2)
  (3 - k) / 3 = 1 →
  vector_cosine a c = Real.sqrt 5 / 5 := by
  intros
  sorry

end cosine_angle_between_vectors_l231_231562


namespace solve_inequality_l231_231609

-- Define the inequality as a function
def inequality_holds (x : ℝ) : Prop :=
  (2 * x + 3) / (x + 4) > (4 * x + 5) / (3 * x + 10)

-- Define the solution set as intervals excluding the points
def solution_set (x : ℝ) : Prop :=
  x < -5 / 2 ∨ x > -2

theorem solve_inequality (x : ℝ) : inequality_holds x ↔ solution_set x :=
by sorry

end solve_inequality_l231_231609


namespace locus_of_midpoint_of_chord_l231_231144

theorem locus_of_midpoint_of_chord 
  (A B C : ℝ) (h_arith_seq : A - 2 * B + C = 0) 
  (h_passing_through : ∀ t : ℝ,  t*A + -2*B + C = 0) :
  ∀ (x y : ℝ), 
    (Ax + By + C = 0) → 
    (h_on_parabola : y = -2 * x ^ 2) 
    → y + 1 = -(2 * x - 1) ^ 2 :=
sorry

end locus_of_midpoint_of_chord_l231_231144


namespace investment_period_two_years_l231_231103

theorem investment_period_two_years
  (P : ℝ) (r : ℝ) (A : ℝ) (n : ℕ) (hP : P = 6000) (hr : r = 0.10) (hA : A = 7260) (hn : n = 1) : 
  ∃ t : ℝ, t = 2 ∧ A = P * (1 + r / n) ^ (n * t) :=
by
  sorry

end investment_period_two_years_l231_231103


namespace f_monotonicity_l231_231847

noncomputable def f : ℝ → ℝ := sorry -- Definition of the function f(x)

axiom f_symm (x : ℝ) : f (1 - x) = f x

axiom f_derivative (x : ℝ) : (x - 1 / 2) * (deriv f x) > 0

theorem f_monotonicity (x1 x2 : ℝ) (h1 : x1 < x2) (h2 : x1 + x2 > 1) : f x1 < f x2 :=
sorry

end f_monotonicity_l231_231847


namespace hyperbola_range_of_m_l231_231003

theorem hyperbola_range_of_m (m : ℝ) : (∃ f : ℝ → ℝ → ℝ, ∀ x y: ℝ, f x y = (x^2 / (4 - m) - y^2 / (2 + m))) → (4 - m) * (2 + m) > 0 → -2 < m ∧ m < 4 :=
by
  intros h_eq h_cond
  sorry

end hyperbola_range_of_m_l231_231003


namespace graphs_differ_l231_231818

theorem graphs_differ (x : ℝ) :
  (∀ (y : ℝ), y = x + 3 ↔ y ≠ (x^2 - 1) / (x - 1) ∧
              y ≠ (x^2 - 1) / (x - 1) ∧
              ∀ (y : ℝ), y = (x^2 - 1) / (x - 1) ↔ ∀ (z : ℝ), y ≠ x + 3 ∧ y ≠ x + 1) := sorry

end graphs_differ_l231_231818


namespace negate_proposition_l231_231374

theorem negate_proposition : (¬ ∀ x : ℝ, x^2 + 2*x + 1 > 0) ↔ ∃ x : ℝ, x^2 + 2*x + 1 ≤ 0 := by
  sorry

end negate_proposition_l231_231374


namespace jessica_mother_age_l231_231758

theorem jessica_mother_age
  (mother_age_when_died : ℕ)
  (jessica_age_when_died : ℕ)
  (jessica_current_age : ℕ)
  (years_since_mother_died : ℕ)
  (half_age_condition : jessica_age_when_died = mother_age_when_died / 2)
  (current_age_condition : jessica_current_age = 40)
  (years_since_death_condition : years_since_mother_died = 10)
  (age_at_death_condition : jessica_age_when_died = jessica_current_age - years_since_mother_died) :
  mother_age_when_died + years_since_mother_died = 70 :=
by {
  sorry
}

end jessica_mother_age_l231_231758


namespace gcd_282_470_l231_231808

theorem gcd_282_470 : Nat.gcd 282 470 = 94 :=
by
  sorry

end gcd_282_470_l231_231808


namespace election_at_least_one_past_officer_l231_231839

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem election_at_least_one_past_officer : 
  let total_candidates := 16
  let past_officers := 7
  let officer_positions := 5
  choose total_candidates officer_positions - choose (total_candidates - past_officers) officer_positions = 4242 :=
by
  sorry

end election_at_least_one_past_officer_l231_231839


namespace minimum_value_sum_l231_231294

theorem minimum_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a / (b + 3 * c) + b / (8 * c + 4 * a) + 9 * c / (3 * a + 2 * b)) ≥ 47 / 48 :=
by sorry

end minimum_value_sum_l231_231294


namespace trees_per_day_l231_231782

def blocks_per_tree := 3
def total_blocks := 30
def days := 5

theorem trees_per_day : (total_blocks / days) / blocks_per_tree = 2 := by
  sorry

end trees_per_day_l231_231782


namespace total_students_l231_231852

-- Define the conditions
def students_in_front : Nat := 7
def position_from_back : Nat := 6

-- Define the proof problem
theorem total_students : (students_in_front + 1 + (position_from_back - 1)) = 13 := by
  -- Proof steps will go here (use sorry to skip for now)
  sorry

end total_students_l231_231852


namespace sum_of_two_rationals_negative_l231_231322

theorem sum_of_two_rationals_negative (a b : ℚ) (h : a + b < 0) : a < 0 ∨ b < 0 := sorry

end sum_of_two_rationals_negative_l231_231322


namespace train_length_l231_231834

theorem train_length :
  (∃ L : ℕ, (L / 15) = (L + 800) / 45) → L = 400 :=
by
  sorry

end train_length_l231_231834


namespace card_prob_queen_diamond_l231_231059

theorem card_prob_queen_diamond : 
  let total_cards := 52
  let queen_first_prob := 4 / 52
  let remaining_cards_after_queen := 51
  let diamond_after_queen_of_diamonds_prob := 12 / 51
  let diamond_after_non_diamond_queen_prob := 13 / 51
  let case1_prob := (1 / total_cards) * diamond_after_queen_of_diamonds_prob
  let case2_prob := (3 / total_cards) * diamond_after_non_diamond_queen_prob
  let total_prob := case1_prob + case2_prob in
  total_prob = 18 / 221 :=
by
  let total_cards := 52
  let queen_first_prob := 4 / 52
  let remaining_cards_after_queen := 51
  let diamond_after_queen_of_diamonds_prob := 12 / 51
  let diamond_after_non_diamond_queen_prob := 13 / 51
  let case1_prob := (1 / total_cards) * diamond_after_queen_of_diamonds_prob
  let case2_prob := (3 / total_cards) * diamond_after_non_diamond_queen_prob
  let total_prob := case1_prob + case2_prob
  sorry

end card_prob_queen_diamond_l231_231059


namespace female_managers_count_l231_231689

-- Definitions based on conditions
def total_employees : Nat := 250
def female_employees : Nat := 90
def total_managers : Nat := 40
def male_associates : Nat := 160

-- Statement to prove
theorem female_managers_count : (total_managers = 40) :=
by
  sorry

end female_managers_count_l231_231689


namespace lindsey_squat_weight_l231_231767

theorem lindsey_squat_weight :
  let bandA := 7
  let bandB := 5
  let bandC := 3
  let leg_weight := 10
  let dumbbell := 15
  let total_weight := (2 * bandA) + (2 * bandB) + (2 * bandC) + (2 * leg_weight) + dumbbell
  total_weight = 65 :=
by
  sorry

end lindsey_squat_weight_l231_231767


namespace remainder_of_expression_l231_231315

theorem remainder_of_expression (n : ℤ) : (10 + n^2) % 7 = (3 + n^2) % 7 := 
by {
  sorry
}

end remainder_of_expression_l231_231315


namespace john_paid_percentage_l231_231950

theorem john_paid_percentage (SRP WP : ℝ) (h1 : SRP = 1.40 * WP) (h2 : ∀ P, P = (1 / 3) * SRP) : ((1 / 3) * SRP / SRP * 100) = 33.33 :=
by
  sorry

end john_paid_percentage_l231_231950


namespace perfect_square_of_division_l231_231870

theorem perfect_square_of_division (a b : ℤ) (ha : 0 < a) (hb : 0 < b) 
  (h : (a * b + 1) ∣ (a^2 + b^2)) : ∃ k : ℤ, 0 < k ∧ k^2 = (a^2 + b^2) / (a * b + 1) :=
by
  sorry

end perfect_square_of_division_l231_231870


namespace age_difference_l231_231387

variable (A B C : ℕ)

def age_relationship (B C : ℕ) : Prop :=
  B = 2 * C

def total_ages (A B C : ℕ) : Prop :=
  A + B + C = 72

theorem age_difference (B : ℕ) (hB : B = 28) (h1 : age_relationship B C) (h2 : total_ages A B C) :
  A - B = 2 :=
sorry

end age_difference_l231_231387


namespace response_rate_increase_approx_l231_231242

theorem response_rate_increase_approx :
  let original_customers := 80
  let original_respondents := 7
  let redesigned_customers := 63
  let redesigned_respondents := 9
  let original_response_rate := (original_respondents : ℝ) / original_customers * 100
  let redesigned_response_rate := (redesigned_respondents : ℝ) / redesigned_customers * 100
  let percentage_increase := (redesigned_response_rate - original_response_rate) / original_response_rate * 100
  abs (percentage_increase - 63.24) < 0.01 := by
  sorry

end response_rate_increase_approx_l231_231242


namespace speed_of_mrs_a_l231_231020

theorem speed_of_mrs_a
  (distance_between : ℝ)
  (speed_mr_a : ℝ)
  (speed_bee : ℝ)
  (distance_bee_travelled : ℝ)
  (time_bee : ℝ)
  (remaining_distance : ℝ)
  (speed_mrs_a : ℝ) :
  distance_between = 120 ∧
  speed_mr_a = 30 ∧
  speed_bee = 60 ∧
  distance_bee_travelled = 180 ∧
  time_bee = distance_bee_travelled / speed_bee ∧
  remaining_distance = distance_between - (speed_mr_a * time_bee) ∧
  speed_mrs_a = remaining_distance / time_bee →
  speed_mrs_a = 10 := by
  sorry

end speed_of_mrs_a_l231_231020


namespace fraction_problem_l231_231733

-- Definitions translated from conditions
variables (m n p q : ℚ)
axiom h1 : m / n = 20
axiom h2 : p / n = 5
axiom h3 : p / q = 1 / 15

-- Statement to prove
theorem fraction_problem : m / q = 4 / 15 :=
by
  sorry

end fraction_problem_l231_231733


namespace none_of_these_l231_231893

variables (a b c d e f : Prop)

-- Given conditions
axiom condition1 : a > b → c > d
axiom condition2 : c < d → e > f

-- Invalid conclusions
theorem none_of_these :
  ¬(a < b → e > f) ∧
  ¬(e > f → a < b) ∧
  ¬(e < f → a > b) ∧
  ¬(a > b → e < f) := sorry

end none_of_these_l231_231893


namespace sum_k_over_4k_l231_231444

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l231_231444


namespace scientific_notation_of_1500_l231_231038

theorem scientific_notation_of_1500 :
  (1500 : ℝ) = 1.5 * 10^3 :=
sorry

end scientific_notation_of_1500_l231_231038


namespace percent_of_dollar_in_pocket_l231_231809

theorem percent_of_dollar_in_pocket :
  let nickel := 5
  let dime := 10
  let quarter := 25
  let half_dollar := 50
  (nickel + 2 * dime + quarter + half_dollar = 100) →
  (100 / 100 * 100 = 100) :=
by
  intros
  sorry

end percent_of_dollar_in_pocket_l231_231809


namespace walk_time_is_correct_l231_231241

noncomputable def time_to_walk_one_block := 
  let blocks := 18
  let bike_time_per_block := 20 -- seconds
  let additional_walk_time := 12 * 60 -- 12 minutes in seconds
  let walk_time := blocks * bike_time_per_block + additional_walk_time
  walk_time / blocks

theorem walk_time_is_correct : 
  let W := time_to_walk_one_block
  W = 60 := by
    sorry -- proof goes here

end walk_time_is_correct_l231_231241


namespace cookies_taken_in_four_days_l231_231801

-- Define the initial conditions
def initial_cookies : ℕ := 70
def remaining_cookies : ℕ := 28
def days_in_week : ℕ := 7
def days_of_interest : ℕ := 4

-- Define the total cookies taken out in a week
def cookies_taken_week := initial_cookies - remaining_cookies

-- Define the cookies taken out each day
def cookies_taken_per_day := cookies_taken_week / days_in_week

-- Final statement to show the number of cookies taken out in four days
theorem cookies_taken_in_four_days : cookies_taken_per_day * days_of_interest = 24 := by
  sorry -- The proof steps will be here.

end cookies_taken_in_four_days_l231_231801


namespace sum_k_over_4k_l231_231448

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l231_231448


namespace find_n_from_A_k_l231_231014

theorem find_n_from_A_k (n : ℕ) (A : ℕ → ℕ) (h1 : A 1 = Int.natAbs (n + 1))
  (h2 : ∀ k : ℕ, k > 0 → A k = Int.natAbs (n + (2 * k - 1)))
  (h3 : A 100 = 2005) : n = 1806 :=
sorry

end find_n_from_A_k_l231_231014


namespace num_ways_to_distribute_items_l231_231413

theorem num_ways_to_distribute_items : 
  let items := 5
  let bags := 4
  let distinct_items := 5
  let identical_bags := 4
  (number_of_ways_to_distribute_items_in_4_identical_bags distinct_items identical_bags = 36) := sorry

end num_ways_to_distribute_items_l231_231413


namespace probability_one_out_of_three_l231_231259

def probability_passing_exactly_one (p : ℚ) (n k : ℕ) :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_one_out_of_three :
  probability_passing_exactly_one (1/3) 3 1 = 4/9 :=
by sorry

end probability_one_out_of_three_l231_231259


namespace hyperbola_eccentricity_sqrt_5_l231_231722

theorem hyperbola_eccentricity_sqrt_5
  (a b : ℝ)
  (h1 : ∀ x y : ℝ, ((x^2 / a^2) - (y^2 / b^2) = 1))
  (h2 : ∀ x : ℝ, (y = x^2 + 1))
  (h3 : ∃ x y : ℝ, ((x^2 / a^2) - (y^2 / b^2) = 1) ∧ (y = x^2 + 1) ∧ 
        ∀ x' y' : ℝ, ((x'^2 / a^2) - (y'^2 / b^2) = 1) ∧ (y' = x^2 + 1) → (x, y) = (x', y')) :
  (∃ e : ℝ, e = sqrt 5) :=
by
  sorry

end hyperbola_eccentricity_sqrt_5_l231_231722


namespace unique_solution_inequality_l231_231163

theorem unique_solution_inequality (a : ℝ) : 
  (∀ x : ℝ, -3 ≤ x^2 - 2 * a * x + a ∧ x^2 - 2 * a * x + a ≤ -2 → ∃! x : ℝ, x^2 - 2 * a * x + a = -2) ↔ (a = 2 ∨ a = -1) :=
sorry

end unique_solution_inequality_l231_231163


namespace sum_of_factors_of_36_l231_231642

-- Define the number in question
def n : ℕ := 36

-- Define the set of whole-number factors of n
def factors_of_n : Finset ℕ := {1, 2, 3, 4, 6, 9, 12, 18, 36}

-- Define the sum of the factors of n
def sum_of_factors : ℕ := factors_of_n.sum (λ x, x)

-- The theorem to be proven
theorem sum_of_factors_of_36 : sum_of_factors = 91 := 
by {
  sorry
}

end sum_of_factors_of_36_l231_231642


namespace sum_of_factors_36_eq_91_l231_231655

-- Define what it means to be a whole-number factor of 36
def is_factor_of_36 (n : ℕ) : Prop := n ∣ 36

-- Define the sum of all factors of 36
noncomputable def sum_factors_of_36 : ℕ := 
  ∑ n in Finset.filter is_factor_of_36 (Finset.range 37), n

-- State the theorem
theorem sum_of_factors_36_eq_91 : 
  sum_factors_of_36 = 91 := 
by 
  sorry

end sum_of_factors_36_eq_91_l231_231655


namespace ice_cream_children_count_ice_cream_girls_count_l231_231365

-- Proof Problem for part (a)
theorem ice_cream_children_count (n : ℕ) (h : 3 * n = 24) : n = 8 := sorry

-- Proof Problem for part (b)
theorem ice_cream_girls_count (x y : ℕ) (h : x + y = 8) 
  (hx_even : x % 2 = 0) (hy_even : y % 2 = 0) (hx_pos : x > 0) (hxy : x < y) : y = 6 := sorry

end ice_cream_children_count_ice_cream_girls_count_l231_231365


namespace monotonic_intervals_max_value_of_k_l231_231890

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 2
noncomputable def f_prime (x a : ℝ) : ℝ := Real.exp x - a

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ a < f x₂ a) ∧
  (a > 0 → ∀ x₁ x₂ : ℝ,
    x₁ < x₂ → (x₁ < Real.log a → f x₁ a > f x₂ a) ∧ (x₁ > Real.log a → f x₁ a < f x₂ a)) :=
sorry

theorem max_value_of_k (x : ℝ) (k : ℤ) (a : ℝ) (h_a : a = 1)
  (h : ∀ x > 0, (x - k) * f_prime x a + x + 1 > 0) :
  k ≤ 2 :=
sorry

end monotonic_intervals_max_value_of_k_l231_231890


namespace unique_nat_pair_l231_231920

theorem unique_nat_pair (p : ℕ) (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ (n m : ℕ), n ≠ m ∧ (2 / p : ℚ) = (1 / n + 1 / m : ℚ) ∧ ∀ (n' m' : ℕ), 
  n' ≠ m' ∧ (2 / p : ℚ) = (1 / n' + 1 / m' : ℚ) → (n', m') = (n, m) ∨ (n', m') = (m, n) :=
by
  sorry

end unique_nat_pair_l231_231920


namespace sequences_count_l231_231125

theorem sequences_count (a_n b_n c_n : ℕ → ℕ) :
  (a_n 1 = 1) ∧ (b_n 1 = 1) ∧ (c_n 1 = 1) ∧ 
  (∀ n : ℕ, a_n (n + 1) = a_n n + b_n n) ∧ 
  (∀ n : ℕ, b_n (n + 1) = a_n n + b_n n + c_n n) ∧ 
  (∀ n : ℕ, c_n (n + 1) = b_n n + c_n n) → 
  ∀ n : ℕ, a_n n + b_n n + c_n n = 
            (1/2 * ((1 + Real.sqrt 2)^(n+1) + (1 - Real.sqrt 2)^(n+1))) :=
by
  intro h
  sorry

end sequences_count_l231_231125


namespace total_cost_is_correct_l231_231632

noncomputable def total_cost : ℝ :=
  let palm_fern_cost := 15.00
  let creeping_jenny_cost := 4.00
  let geranium_cost := 3.50
  let elephant_ear_cost := 7.00
  let purple_fountain_grass_cost := 6.00
  let pots := 6
  let sales_tax := 0.07
  let cost_one_pot := palm_fern_cost 
                   + 4 * creeping_jenny_cost 
                   + 4 * geranium_cost 
                   + 2 * elephant_ear_cost 
                   + 3 * purple_fountain_grass_cost
  let total_pots_cost := pots * cost_one_pot
  let tax := total_pots_cost * sales_tax
  total_pots_cost + tax

theorem total_cost_is_correct : total_cost = 494.34 :=
by
  -- This is where the proof would go, but we are adding sorry to skip the proof
  sorry

end total_cost_is_correct_l231_231632


namespace gillian_more_than_three_times_sandi_l231_231032

-- Definitions of the conditions
def sandi_initial : ℕ := 600
def sandi_spent : ℕ := sandi_initial / 2
def gillian_spent : ℕ := 1050
def three_times_sandi_spent : ℕ := 3 * sandi_spent

-- Theorem statement with the proof to be added
theorem gillian_more_than_three_times_sandi :
  gillian_spent - three_times_sandi_spent = 150 := 
sorry

end gillian_more_than_three_times_sandi_l231_231032


namespace yoongi_calculation_l231_231386

theorem yoongi_calculation (x : ℝ) (h : x - 5 = 30) : x / 7 = 5 :=
by
  sorry

end yoongi_calculation_l231_231386


namespace elderly_teachers_in_sample_l231_231625

-- Definitions based on the conditions
def numYoungTeachersSampled : ℕ := 320
def ratioYoungToElderly : ℚ := 16 / 9

-- The theorem that needs to be proved
theorem elderly_teachers_in_sample :
  ∃ numElderlyTeachersSampled : ℕ, 
    numYoungTeachersSampled * (9 / 16) = numElderlyTeachersSampled := 
by
  use 180
  sorry

end elderly_teachers_in_sample_l231_231625


namespace initial_books_l231_231178

theorem initial_books (B : ℕ) (h : B + 5 = 7) : B = 2 :=
by sorry

end initial_books_l231_231178


namespace sum_of_factors_36_l231_231644

theorem sum_of_factors_36 : (∑ i in {1, 2, 3, 4, 6, 9, 12, 18, 36}, i) = 91 := 
by
  -- The definition of the set of factors of 36
  have factors_36 : Set Nat := {1, 2, 3, 4, 6, 9, 12, 18, 36}
  -- Summing the factors
  have sum_factors_36 := (∑ i in factors_36, i)
  show sum_factors_36 = 91
  sorry

end sum_of_factors_36_l231_231644


namespace equilateral_triangle_properties_l231_231589

noncomputable def equilateral_triangle_perimeter (a : ℝ) : ℝ :=
3 * a

noncomputable def equilateral_triangle_bisector_length (a : ℝ) : ℝ :=
(a * Real.sqrt 3) / 2

theorem equilateral_triangle_properties (a : ℝ) (h : a = 10) :
  equilateral_triangle_perimeter a = 30 ∧
  equilateral_triangle_bisector_length a = 5 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_properties_l231_231589


namespace vector_subtraction_l231_231561

-- Define the given vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, 1)

-- State the theorem that the vector subtraction b - a equals (2, -1)
theorem vector_subtraction : b - a = (2, -1) :=
by
  -- Proof is omitted and replaced with sorry
  sorry

end vector_subtraction_l231_231561


namespace set_intersection_complement_l231_231894

-- Definitions corresponding to conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | x > 1}

-- Statement to prove
theorem set_intersection_complement : A ∩ (U \ B) = {x | -1 < x ∧ x ≤ 1} := by
  sorry

end set_intersection_complement_l231_231894


namespace general_term_formula_not_arithmetic_sequence_l231_231296

noncomputable def geometric_sequence (n : ℕ) : ℕ := 2^n

theorem general_term_formula :
  ∀ (a : ℕ → ℕ),
    (∀ n, a n = geometric_sequence n) →
    (∃ (q : ℕ),
      ∀ n, a n = 2^n) :=
by
  sorry

theorem not_arithmetic_sequence :
  ∀ (a : ℕ → ℕ),
    (∀ n, a n = geometric_sequence n) →
    ¬(∃ m n p : ℕ, m < n ∧ n < p ∧ (2 * a n = a m + a p)) :=
by
  sorry

end general_term_formula_not_arithmetic_sequence_l231_231296


namespace ratio_of_cookies_l231_231768

-- Definitions based on the conditions
def initial_cookies : ℕ := 19
def cookies_to_friend : ℕ := 5
def cookies_left : ℕ := 5
def cookies_eaten : ℕ := 2

-- Calculating the number of cookies left after giving cookies to the friend
def cookies_after_giving_to_friend := initial_cookies - cookies_to_friend

-- Maria gave to her family the remaining cookies minus the cookies she has left and she has eaten.
def cookies_given_to_family := cookies_after_giving_to_friend - cookies_eaten - cookies_left

-- The ratio to be proven 1:2, which is mathematically 1/2
theorem ratio_of_cookies : (cookies_given_to_family : ℚ) / (cookies_after_giving_to_friend : ℚ) = 1 / 2 := by
  sorry

end ratio_of_cookies_l231_231768


namespace effective_annual_interest_rate_is_correct_l231_231613

noncomputable def quarterly_interest_rate : ℝ := 0.02

noncomputable def annual_interest_rate (quarterly_rate : ℝ) : ℝ :=
  ((1 + quarterly_rate) ^ 4 - 1) * 100

theorem effective_annual_interest_rate_is_correct :
  annual_interest_rate quarterly_interest_rate = 8.24 :=
by
  sorry

end effective_annual_interest_rate_is_correct_l231_231613


namespace arithmetic_sequence_common_difference_l231_231149

theorem arithmetic_sequence_common_difference {a : ℕ → ℝ} (h₁ : a 1 = 2) (h₂ : a 2 + a 4 = a 6) : ∃ d : ℝ, d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l231_231149


namespace range_of_a_max_value_of_z_l231_231196

variable (a b : ℝ)

-- Definition of the assumptions
def condition1 := (2 * a + b = 9)
def condition2 := (|9 - b| + |a| < 3)
def condition3 := (a > 0)
def condition4 := (b > 0)
def z := a^2 * b

-- Statement for problem (i)
theorem range_of_a (h1 : condition1 a b) (h2 : condition2 a b) : -1 < a ∧ a < 1 := sorry

-- Statement for problem (ii)
theorem max_value_of_z (h1 : condition1 a b) (h2 : condition3 a) (h3 : condition4 b) : 
  z a b = 27 := sorry

end range_of_a_max_value_of_z_l231_231196


namespace fractional_product_l231_231425

theorem fractional_product :
  ((3/4) * (4/5) * (5/6) * (6/7) * (7/8)) = 3/8 :=
by
  sorry

end fractional_product_l231_231425


namespace university_math_students_l231_231264

theorem university_math_students
  (total_students : ℕ)
  (math_only : ℕ)
  (stats_only : ℕ)
  (both_courses : ℕ)
  (H1 : total_students = 75)
  (H2 : math_only + stats_only + both_courses = total_students)
  (H3 : math_only = 2 * (stats_only + both_courses))
  (H4 : both_courses = 9) :
  math_only + both_courses = 53 :=
by
  sorry

end university_math_students_l231_231264


namespace janessa_initial_cards_l231_231906

theorem janessa_initial_cards (X : ℕ)  :
  (X + 45 = 49) →
  X = 4 :=
by
  intro h
  sorry

end janessa_initial_cards_l231_231906


namespace lisa_total_spoons_l231_231198

def number_of_baby_spoons (num_children num_spoons_per_child : Nat) : Nat :=
  num_children * num_spoons_per_child

def number_of_decorative_spoons : Nat := 2

def number_of_old_spoons (baby_spoons decorative_spoons : Nat) : Nat :=
  baby_spoons + decorative_spoons
  
def number_of_new_spoons (large_spoons teaspoons : Nat) : Nat :=
  large_spoons + teaspoons

def total_number_of_spoons (old_spoons new_spoons : Nat) : Nat :=
  old_spoons + new_spoons

theorem lisa_total_spoons
  (children : Nat)
  (spoons_per_child : Nat)
  (large_spoons : Nat)
  (teaspoons : Nat)
  (children_eq : children = 4)
  (spoons_per_child_eq : spoons_per_child = 3)
  (large_spoons_eq : large_spoons = 10)
  (teaspoons_eq : teaspoons = 15)
  : total_number_of_spoons (number_of_old_spoons (number_of_baby_spoons children spoons_per_child) number_of_decorative_spoons) (number_of_new_spoons large_spoons teaspoons) = 39 :=
by
  sorry

end lisa_total_spoons_l231_231198


namespace circle_in_square_radius_l231_231402

noncomputable def radius_of_circle_in_quadrilateral 
  (side_length : ℝ) (M_is_midpoint : Prop) 
  (circle_touches_AM_CD_DA : Prop) : ℝ :=
3 - real.sqrt 5

theorem circle_in_square_radius (ABCD_square : Prop)
  (side_length_eq : side_length = 2)
  (M_midpoint_of_BC : M_is_midpoint)
  (circle_touches_sides : circle_touches_AM_CD_DA)
  : radius_of_circle_in_quadrilateral side_length M_is_midpoint circle_touches_AM_CD_DA 
  = 3 - real.sqrt 5 := 
sorry

end circle_in_square_radius_l231_231402


namespace negative_number_unique_l231_231685

theorem negative_number_unique (a b c d : ℚ) (h₁ : a = 1) (h₂ : b = 0) (h₃ : c = 1/2) (h₄ : d = -2) :
  ∃! x : ℚ, x < 0 ∧ (x = a ∨ x = b ∨ x = c ∨ x = d) :=
by 
  sorry

end negative_number_unique_l231_231685


namespace scientific_notation_proof_l231_231035

-- Given number is 657,000
def number : ℕ := 657000

-- Scientific notation of the given number
def scientific_notation (n : ℕ) : Prop :=
    n = 657000 ∧ (6.57 : ℝ) * (10 : ℝ)^5 = 657000

theorem scientific_notation_proof : scientific_notation number :=
by 
  sorry

end scientific_notation_proof_l231_231035


namespace perfect_squares_as_difference_l231_231311

theorem perfect_squares_as_difference (N : ℕ) (hN : N = 20000) : 
  (∃ (n : ℕ), n = 71 ∧ 
    ∀ m < N, 
      (∃ a b : ℤ, 
        a^2 = m ∧
        b^2 = m + ((b + 1)^2 - b^2) - 1 ∧ 
        (b + 1)^2 - b^2 = 2 * b + 1)) :=
by 
  sorry

end perfect_squares_as_difference_l231_231311


namespace sum_series_l231_231470

theorem sum_series : ∑' (k : ℕ) (k > 0), k / (4 : ℝ) ^ k = 4 / 9 :=
by
  sorry

end sum_series_l231_231470


namespace solve_quadratic_equation_l231_231123

noncomputable def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

theorem solve_quadratic_equation :
  let x1 := -2
      x2 := 11 in
  quadratic_eq 1 (-9) (-22) x1 ∧ quadratic_eq 1 (-9) (-22) x2 ∧ x1 < x2 :=
by
  sorry

end solve_quadratic_equation_l231_231123


namespace shaded_region_area_correct_l231_231591

noncomputable def area_shaded_region : ℝ := 
  let side_length := 2
  let radius := 1
  let area_square := side_length^2
  let area_circle := Real.pi * radius^2
  area_square - area_circle

theorem shaded_region_area_correct : area_shaded_region = 4 - Real.pi :=
  by
    sorry

end shaded_region_area_correct_l231_231591


namespace distance_to_larger_cross_section_l231_231955

theorem distance_to_larger_cross_section
    (A B : ℝ)
    (a b : ℝ)
    (d : ℝ)
    (h : ℝ)
    (h_eq : h = 30):
  A = 300 * Real.sqrt 2 → 
  B = 675 * Real.sqrt 2 → 
  a = Real.sqrt (A / B) → 
  b = d / (1 - a) → 
  d = 10 → 
  b = h :=
by
  sorry

end distance_to_larger_cross_section_l231_231955


namespace sum_of_money_l231_231593

theorem sum_of_money (jimin_100_won : ℕ) (jimin_50_won : ℕ) (seokjin_100_won : ℕ) (seokjin_10_won : ℕ) 
  (h1 : jimin_100_won = 5) (h2 : jimin_50_won = 1) (h3 : seokjin_100_won = 2) (h4 : seokjin_10_won = 7) :
  jimin_100_won * 100 + jimin_50_won * 50 + seokjin_100_won * 100 + seokjin_10_won * 10 = 820 :=
by
  sorry

end sum_of_money_l231_231593


namespace f_is_even_f_monotonic_increase_range_of_a_for_solutions_l231_231197

-- Define the function f(x) = x^2 - 2a|x|
def f (a x : ℝ) : ℝ := x^2 - 2 * a * |x|

-- Given a > 0
variable (a : ℝ) (ha : a > 0)

-- 1. Prove that f(x) is an even function.
theorem f_is_even : ∀ x : ℝ, f a x = f a (-x) := sorry

-- 2. Prove the interval of monotonic increase for f(x) when x > 0 is [a, +∞).
theorem f_monotonic_increase (x : ℝ) (hx : x > 0) : a ≤ x → ∃ c : ℝ, x ≤ c := sorry

-- 3. Prove the range of values for a for which the equation f(x) = -1 has solutions is a ≥ 1.
theorem range_of_a_for_solutions : (∃ x : ℝ, f a x = -1) ↔ 1 ≤ a := sorry

end f_is_even_f_monotonic_increase_range_of_a_for_solutions_l231_231197


namespace land_per_person_l231_231339

noncomputable def total_land_area : ℕ := 20000
noncomputable def num_people_sharing : ℕ := 5

theorem land_per_person (Jose_land : ℕ) (h : Jose_land = total_land_area / num_people_sharing) :
  Jose_land = 4000 :=
by
  sorry

end land_per_person_l231_231339


namespace FC_value_l231_231292

variables (DC CB AB AD ED FC CA BD : ℝ)

-- Set the conditions as variables
variable (h_DC : DC = 10)
variable (h_CB : CB = 12)
variable (h_AB : AB = (1/3) * AD)
variable (h_ED : ED = (2/3) * AD)
variable (h_BD : BD = 22)
variable (BD_eq : BD = DC + CB)
variable (CA_eq : CA = CB + AB)

-- Define the relationship for the final result
def find_FC (DC CB AB AD ED FC CA BD : ℝ) := FC = (ED * CA) / AD

-- The main statement to be proven
theorem FC_value : 
  find_FC DC CB AB (33 : ℝ) (22 : ℝ) FC (23 : ℝ) (22 : ℝ) → 
  FC = (506/33) :=
by 
  intros h
  sorry

end FC_value_l231_231292


namespace exists_projectile_time_l231_231791

noncomputable def projectile_time := 
  ∃ t1 t2 : ℝ, (-4.9 * t1^2 + 31 * t1 - 40 = 0) ∧ ((abs (t1 - 1.8051) < 0.001) ∨ (abs (t2 - 4.5319) < 0.001))

theorem exists_projectile_time : projectile_time := 
sorry

end exists_projectile_time_l231_231791


namespace unusual_numbers_exist_l231_231089

noncomputable def n1 : ℕ := 10 ^ 100 - 1
noncomputable def n2 : ℕ := 10 ^ 100 / 2 - 1

theorem unusual_numbers_exist : 
  (n1 ^ 3 % 10 ^ 100 = n1 ∧ n1 ^ 2 % 10 ^ 100 ≠ n1) ∧ 
  (n2 ^ 3 % 10 ^ 100 = n2 ∧ n2 ^ 2 % 10 ^ 100 ≠ n2) :=
by
  sorry

end unusual_numbers_exist_l231_231089


namespace unique_p_value_l231_231289

theorem unique_p_value (p : Nat) (h₁ : Nat.Prime (p+10)) (h₂ : Nat.Prime (p+14)) : p = 3 := by
  sorry

end unique_p_value_l231_231289


namespace proportion_solution_l231_231739

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 5 / 8) : x = 1.2 := 
by 
suffices h₀ : x = 6 / 5 by sorry
suffices h₁ : 6 / 5 = 1.2 by sorry
-- Proof steps go here
sorry

end proportion_solution_l231_231739


namespace tan_sum_identity_l231_231719

theorem tan_sum_identity (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) : 
  Real.tan α + (1 / Real.tan α) = 3 :=
by
  sorry

end tan_sum_identity_l231_231719


namespace fewer_onions_than_tomatoes_and_corn_l231_231267

def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

theorem fewer_onions_than_tomatoes_and_corn :
  (tomatoes + corn - onions) = 5200 :=
by
  sorry

end fewer_onions_than_tomatoes_and_corn_l231_231267


namespace product_of_a_and_c_l231_231804

theorem product_of_a_and_c (a b c : ℝ) (h1 : a + b + c = 100) (h2 : a - b = 20) (h3 : b - c = 30) : a * c = 378.07 :=
by
  sorry

end product_of_a_and_c_l231_231804


namespace fraction_shaded_l231_231096

theorem fraction_shaded (s r : ℝ) (h : s^2 = 3 * r^2) :
    (1/2 * π * r^2) / (1/4 * π * s^2) = 2/3 := 
  sorry

end fraction_shaded_l231_231096


namespace david_and_maria_ages_l231_231819

theorem david_and_maria_ages 
  (D Y M : ℕ)
  (h1 : Y = D + 7)
  (h2 : Y = 2 * D)
  (h3 : M = D + 4)
  (h4 : M = Y / 2)
  : D = 7 ∧ M = 11 := by
  sorry

end david_and_maria_ages_l231_231819


namespace hemisphere_surface_area_l231_231232

-- Define the condition of the problem
def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2
def base_area_of_hemisphere : ℝ := 3

-- The proof problem statement
theorem hemisphere_surface_area : 
  ∃ (r : ℝ), (Real.pi * r^2 = 3) → (2 * Real.pi * r^2 + Real.pi * r^2 = 9) := 
by 
  sorry

end hemisphere_surface_area_l231_231232


namespace sum_factors_of_36_l231_231646

def is_factor (n d : ℕ) : Prop := d ∣ n

def factors (n : ℕ) : List ℕ := list.filter (λ d, is_factor n d) (List.range (n + 1))

def sum_of_factors (n : ℕ) : ℕ := (factors n).sum

theorem sum_factors_of_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_factors_of_36_l231_231646


namespace sum_series_eq_four_ninths_l231_231489

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l231_231489


namespace dasha_ate_one_bowl_l231_231948

-- Define the quantities for Masha, Dasha, Glasha, and Natasha
variables (M D G N : ℕ)

-- Given conditions
def conditions : Prop :=
  (M + D + G + N = 16) ∧
  (G + N = 9) ∧
  (M > D) ∧
  (M > G) ∧
  (M > N)

-- The problem statement rewritten in Lean: Prove that given the conditions, Dasha ate 1 bowl.
theorem dasha_ate_one_bowl (h : conditions M D G N) : D = 1 :=
sorry

end dasha_ate_one_bowl_l231_231948


namespace no_real_solution_l231_231703

theorem no_real_solution :
  ∀ x : ℝ, ((x - 4 * x + 15)^2 + 3)^2 + 1 ≠ -|x|^2 :=
by
  intro x
  sorry

end no_real_solution_l231_231703


namespace probability_A_selected_B_not_selected_l231_231275

def comb (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_A_selected_B_not_selected :
  let totalWays := comb 5 2,
      favorableWays := comb 3 1
  in (favorableWays : ℚ) / totalWays = 3 / 10 :=
by
  let totalWays := comb 5 2
  let favorableWays := comb 3 1
  have totalWays_eq : totalWays = 10 := by sorry
  have favorableWays_eq : favorableWays = 3 := by sorry
  calc
    (favorableWays : ℚ) / totalWays
        = (3 : ℚ) / 10 : by rw [favorableWays_eq, totalWays_eq]
        ... = 3 / 10   : by norm_num

end probability_A_selected_B_not_selected_l231_231275


namespace find_number_l231_231743

-- We define n, x, y as real numbers
variables (n x y : ℝ)

-- Define the conditions as hypotheses
def condition1 : Prop := n * (x - y) = 4
def condition2 : Prop := 6 * x - 3 * y = 12

-- Define the theorem we need to prove: If the conditions hold, then n = 2
theorem find_number (h1 : condition1 n x y) (h2 : condition2 x y) : n = 2 := 
sorry

end find_number_l231_231743


namespace infinite_series_sum_l231_231527

noncomputable def sum_series : ℝ := ∑' k : ℕ, (k + 1) / (4^ (k + 1))

theorem infinite_series_sum : sum_series = 4 / 9 :=
by
  sorry

end infinite_series_sum_l231_231527


namespace determine_s_l231_231431

noncomputable def quadratic_root_conjugate (p s : ℝ) : Prop :=
  let root1 := 4 + 3 * Complex.i in
  let root2 := 4 - 3 * Complex.i in
  let sum_roots := root1 + root2 in
  let prod_roots := root1 * root2 in
  let quadratic_eq := ∀ x, 3 * x^2 + p * x + s = 0 in
  sum_roots = 8 ∧ prod_roots = 25 ∧ s = 75

theorem determine_s (p s : ℝ) (h : quadratic_root_conjugate p s) : s = 75 :=
by
  sorry

end determine_s_l231_231431


namespace rational_solution_exists_l231_231764

theorem rational_solution_exists (a b c : ℤ) (x₀ y₀ z₀ : ℤ) (h₀ : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h₁ : a * x₀^2 + b * y₀^2 + c * z₀^2 = 0) (h₂ : x₀ ≠ 0 ∨ y₀ ≠ 0 ∨ z₀ ≠ 0) : 
  ∃ (x y z : ℚ), a * x^2 + b * y^2 + c * z^2 = 1 := 
sorry

end rational_solution_exists_l231_231764


namespace expected_number_of_visible_people_l231_231827

noncomputable def expected_visible_people (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, 1 / (i + 1)

theorem expected_number_of_visible_people (n : ℕ) :
  expected_visible_people n = ∑ i in Finset.range n, 1 / (i + 1) := 
by
  -- Proof is omitted as per instructions
  sorry

end expected_number_of_visible_people_l231_231827


namespace fraction_product_l231_231419

theorem fraction_product :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := 
by
  -- Detailed proof steps would go here
  sorry

end fraction_product_l231_231419


namespace sum_infinite_series_l231_231507

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l231_231507


namespace binomial_sum_of_coefficients_l231_231157

-- Given condition: for the third term in the expansion, the binomial coefficient is 15
def binomial_coefficient_condition (n : ℕ) := Nat.choose n 2 = 15

-- The goal: the sum of the coefficients of all terms in the expansion is 1/64
theorem binomial_sum_of_coefficients (n : ℕ) (h : binomial_coefficient_condition n) :
  (1:ℚ) / (2 : ℚ)^6 = 1 / 64 :=
by 
  have h₁ : n = 6 := by sorry -- Solve for n using the given condition.
  sorry -- Prove the sum of coefficients when x is 1.

end binomial_sum_of_coefficients_l231_231157


namespace fraction_equal_l231_231731

variable {m n p q : ℚ}

-- Define the conditions
def condition1 := (m / n = 20)
def condition2 := (p / n = 5)
def condition3 := (p / q = 1 / 15)

-- State the theorem
theorem fraction_equal (h1 : condition1) (h2 : condition2) (h3 : condition3) : (m / q = 4 / 15) :=
  sorry

end fraction_equal_l231_231731


namespace exists_two_unusual_numbers_l231_231090

noncomputable def is_unusual (n : ℕ) : Prop :=
  (n ^ 3 % 10 ^ 100 = n) ∧ (n ^ 2 % 10 ^ 100 ≠ n)

theorem exists_two_unusual_numbers :
  ∃ n1 n2 : ℕ, (is_unusual n1) ∧ (is_unusual n2) ∧ (n1 ≠ n2) ∧ (n1 >= 10 ^ 99) ∧ (n1 < 10 ^ 100) ∧ (n2 >= 10 ^ 99) ∧ (n2 < 10 ^ 100) :=
begin
  sorry
end

end exists_two_unusual_numbers_l231_231090


namespace function_strictly_decreasing_l231_231429

open Real

theorem function_strictly_decreasing :
  ∀ x : ℝ, 0 < x ∧ x ≤ 1 / 2 → deriv (λ x, 1 / x + 2 * log x) x < 0 :=
by
  -- Proof is omitted
  sorry

end function_strictly_decreasing_l231_231429


namespace sum_of_factors_of_36_l231_231657

open List

def factors (n : ℕ) : List ℕ :=
  List.filter (λ x, n % x = 0) (List.range (n + 1))

def sumFactors (n : ℕ) : ℕ :=
  (factors n).sum

theorem sum_of_factors_of_36 : sumFactors 36 = 91 := by
  sorry

end sum_of_factors_of_36_l231_231657


namespace remainder_of_power_mod_l231_231538

theorem remainder_of_power_mod :
  (5^2023) % 17 = 15 :=
begin
  sorry
end

end remainder_of_power_mod_l231_231538


namespace regular_polygon_inscribed_circle_area_l231_231401

theorem regular_polygon_inscribed_circle_area
  (n : ℕ) (R : ℝ) (hR : R ≠ 0) (h_area : (1 / 2 : ℝ) * n * R^2 * Real.sin (2 * Real.pi / n) = 4 * R^2) :
  n = 20 :=
by 
  sorry

end regular_polygon_inscribed_circle_area_l231_231401


namespace tickets_distribution_l231_231675

theorem tickets_distribution (people tickets : ℕ) (h_people : people = 9) (h_tickets : tickets = 24)
  (h_each_gets_at_least_one : ∀ (i : ℕ), i < people → (1 : ℕ) ≤ 1) :
  ∃ (count : ℕ), count ≥ 4 ∧ ∃ (f : ℕ → ℕ), (∀ i, i < people → 1 ≤ f i ∧ f i ≤ tickets) ∧ (∀ i < people, ∃ j < people, f i = f j) :=
  sorry

end tickets_distribution_l231_231675


namespace arrange_polynomial_descending_l231_231411

variable (a b : ℤ)

def polynomial := -a + 3 * a^5 * b^3 + 5 * a^3 * b^5 - 9 + 4 * a^2 * b^2 

def rearranged_polynomial := 3 * a^5 * b^3 + 5 * a^3 * b^5 + 4 * a^2 * b^2 - a - 9

theorem arrange_polynomial_descending :
  polynomial a b = rearranged_polynomial a b :=
sorry

end arrange_polynomial_descending_l231_231411


namespace sum_series_eq_four_ninths_l231_231484

open BigOperators

noncomputable def S : ℝ := ∑' (n : ℕ), (n + 1 : ℝ) / 4 ^ (n + 1)

theorem sum_series_eq_four_ninths : S = 4 / 9 :=
by
  sorry

end sum_series_eq_four_ninths_l231_231484


namespace scientific_notation_correct_l231_231034

theorem scientific_notation_correct : 657000 = 6.57 * 10^5 :=
by
  sorry

end scientific_notation_correct_l231_231034


namespace sum_series_eq_4_div_9_l231_231451

theorem sum_series_eq_4_div_9 : ∑' k : ℕ, (k + 1) / 4^(k + 1) = 4 / 9 := 
sorry

end sum_series_eq_4_div_9_l231_231451


namespace ratio_of_triangle_side_to_rectangle_width_l231_231980

variables (t w l : ℕ)

-- Condition 1: The perimeter of the equilateral triangle is 24 inches
def triangle_perimeter := 3 * t = 24

-- Condition 2: The perimeter of the rectangle is 24 inches
def rectangle_perimeter := 2 * l + 2 * w = 24

-- Condition 3: The length of the rectangle is twice its width
def length_double_width := l = 2 * w

-- The ratio of the side length of the triangle to the width of the rectangle is 2
theorem ratio_of_triangle_side_to_rectangle_width
    (h_triangle : triangle_perimeter t)
    (h_rectangle : rectangle_perimeter l w)
    (h_length_width : length_double_width l w) :
    t / w = 2 :=
by
    sorry

end ratio_of_triangle_side_to_rectangle_width_l231_231980


namespace kite_perimeter_l231_231680

-- Given the kite's diagonals, shorter sides, and longer sides
def diagonals : ℕ × ℕ := (12, 30)
def shorter_sides : ℕ := 10
def longer_sides : ℕ := 15

-- Problem statement: Prove that the perimeter is 50 inches
theorem kite_perimeter (diag1 diag2 short_len long_len : ℕ) 
                       (h_diag : diag1 = 12 ∧ diag2 = 30)
                       (h_short : short_len = 10)
                       (h_long : long_len = 15) : 
                       2 * short_len + 2 * long_len = 50 :=
by
  -- We provide no proof, only the statement
  sorry

end kite_perimeter_l231_231680


namespace sum_of_factors_of_36_l231_231656

open List

def factors (n : ℕ) : List ℕ :=
  List.filter (λ x, n % x = 0) (List.range (n + 1))

def sumFactors (n : ℕ) : ℕ :=
  (factors n).sum

theorem sum_of_factors_of_36 : sumFactors 36 = 91 := by
  sorry

end sum_of_factors_of_36_l231_231656


namespace problem_statement_l231_231702

def op (x y : ℝ) : ℝ := (x + 3) * (y - 1)

theorem problem_statement (a : ℝ) : (∀ x : ℝ, op (x - a) (x + a) > -16) ↔ -2 < a ∧ a < 6 :=
by
  sorry

end problem_statement_l231_231702


namespace solution_set_inequality_l231_231132

theorem solution_set_inequality (a : ℕ) (h : ∀ x : ℝ, (a-2) * x > (a-2) → x < 1) : a = 0 ∨ a = 1 :=
by
  sorry

end solution_set_inequality_l231_231132


namespace sum_factors_of_36_l231_231647

def is_factor (n d : ℕ) : Prop := d ∣ n

def factors (n : ℕ) : List ℕ := list.filter (λ d, is_factor n d) (List.range (n + 1))

def sum_of_factors (n : ℕ) : ℕ := (factors n).sum

theorem sum_factors_of_36 : sum_of_factors 36 = 91 :=
by 
  sorry

end sum_factors_of_36_l231_231647


namespace max_A_value_l231_231545

-- Variables
variables {x1 x2 x3 y1 y2 y3 z1 z2 z3 : ℝ}

-- Assumptions
axiom pos_x1 : 0 < x1
axiom pos_x2 : 0 < x2
axiom pos_x3 : 0 < x3
axiom pos_y1 : 0 < y1
axiom pos_y2 : 0 < y2
axiom pos_y3 : 0 < y3
axiom pos_z1 : 0 < z1
axiom pos_z2 : 0 < z2
axiom pos_z3 : 0 < z3

-- Statement
theorem max_A_value :
  ∃ A : ℝ, 
    (∀ x1 x2 x3 y1 y2 y3 z1 z2 z3, 
    (0 < x1) → (0 < x2) → (0 < x3) →
    (0 < y1) → (0 < y2) → (0 < y3) →
    (0 < z1) → (0 < z2) → (0 < z3) →
    (x1^3 + x2^3 + x3^3 + 1) * (y1^3 + y2^3 + y3^3 + 1) * (z1^3 + z2^3 + z3^3 + 1) ≥
    A * (x1 + y1 + z1) * (x2 + y2 + z2) * (x3 + y3 + z3)) ∧ 
    A = 9/2 := 
by 
  exists 9/2 
  sorry

end max_A_value_l231_231545


namespace sum_series_eq_4_div_9_l231_231500

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l231_231500


namespace sum_infinite_series_l231_231509

theorem sum_infinite_series : 
  ∑' k : ℕ, (k + 1) / 4 ^ (k + 1) = 4 / 9 :=
by
  -- proof goes here
  sorry

end sum_infinite_series_l231_231509


namespace sum_series_equals_4_div_9_l231_231481

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, (k + 1 : ℝ) / 4^(k + 1)

theorem sum_series_equals_4_div_9 : sum_series = 4 / 9 := by
  sorry

end sum_series_equals_4_div_9_l231_231481


namespace groceries_delivered_l231_231409

variables (S C P g T G : ℝ)
theorem groceries_delivered (hS : S = 14500) (hC : C = 14600) (hP : P = 1.5) (hg : g = 0.05) (hT : T = 40) :
  G = 800 :=
by {
  sorry
}

end groceries_delivered_l231_231409


namespace prove_a_lt_one_l231_231891

/-- Given the function f defined as -2 * ln x + 1 / 2 * (x^2 + 1) - a * x,
    where a > 0, if f(x) ≥ 0 holds in the interval (1, ∞)
    and f(x) = 0 has a unique solution, then a < 1. -/
theorem prove_a_lt_one (f : ℝ → ℝ) (a : ℝ) 
    (h1 : ∀ x, f x = -2 * Real.log x + 1 / 2 * (x^2 + 1) - a * x)
    (h2 : a > 0)
    (h3 : ∀ x, x > 1 → f x ≥ 0)
    (h4 : ∃! x, f x = 0) : 
    a < 1 :=
by
  sorry

end prove_a_lt_one_l231_231891


namespace cookies_taken_in_four_days_l231_231800

-- Define the initial conditions
def initial_cookies : ℕ := 70
def remaining_cookies : ℕ := 28
def days_in_week : ℕ := 7
def days_of_interest : ℕ := 4

-- Define the total cookies taken out in a week
def cookies_taken_week := initial_cookies - remaining_cookies

-- Define the cookies taken out each day
def cookies_taken_per_day := cookies_taken_week / days_in_week

-- Final statement to show the number of cookies taken out in four days
theorem cookies_taken_in_four_days : cookies_taken_per_day * days_of_interest = 24 := by
  sorry -- The proof steps will be here.

end cookies_taken_in_four_days_l231_231800


namespace sum_k_over_4k_l231_231450

theorem sum_k_over_4k :
  ∑' (k : ℕ) (hk : k > 0), (k : ℝ) / 4^k = 4 / 9 :=
by sorry

end sum_k_over_4k_l231_231450


namespace expected_rice_yield_l231_231320

theorem expected_rice_yield (x : ℝ) (y : ℝ) (h : y = 5 * x + 250) (hx : x = 80) : y = 650 :=
by
  sorry

end expected_rice_yield_l231_231320


namespace program_output_for_six_l231_231031

-- Define the factorial function
def factorial : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

-- The theorem we want to prove
theorem program_output_for_six : factorial 6 = 720 := by
  sorry

end program_output_for_six_l231_231031


namespace percentage_problem_l231_231159

variable (x : ℝ)

theorem percentage_problem (h : 0.4 * x = 160) : 240 / x = 0.6 :=
by sorry

end percentage_problem_l231_231159


namespace shirts_per_minute_l231_231687

theorem shirts_per_minute (S : ℕ) 
  (h1 : 12 * S + 14 = 156) : S = 11 := 
by
  sorry

end shirts_per_minute_l231_231687


namespace mean_temperature_is_correct_l231_231044

-- Defining the list of temperatures
def temperatures : List ℝ := [75, 74, 76, 77, 80, 81, 83, 85, 83, 85]

-- Lean statement asserting the mean temperature is 79.9
theorem mean_temperature_is_correct : temperatures.sum / (temperatures.length: ℝ) = 79.9 := 
by
  sorry

end mean_temperature_is_correct_l231_231044


namespace bob_questions_three_hours_l231_231694

theorem bob_questions_three_hours : 
  let first_hour := 13
  let second_hour := first_hour * 2
  let third_hour := second_hour * 2
  first_hour + second_hour + third_hour = 91 :=
by
  sorry

end bob_questions_three_hours_l231_231694


namespace fewer_onions_than_tomatoes_and_corn_l231_231266

def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

theorem fewer_onions_than_tomatoes_and_corn :
  (tomatoes + corn - onions) = 5200 :=
by
  sorry

end fewer_onions_than_tomatoes_and_corn_l231_231266


namespace range_of_a_l231_231017

variable {x a : ℝ}

def p (a : ℝ) (x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2*x - 8 > 0

theorem range_of_a (ha : a < 0) 
  (H : (∀ x, ¬ p a x → q x) ∧ ∃ x, q x ∧ ¬ p a x ∧ ¬ q x) : a ≤ -4 := 
sorry

end range_of_a_l231_231017


namespace cube_root_two_irrational_l231_231079

theorem cube_root_two_irrational : ∛2 ∉ ℚ :=
sorry

end cube_root_two_irrational_l231_231079


namespace sum_factors_36_eq_91_l231_231662

theorem sum_factors_36_eq_91 : 
  (∑ (i : ℕ) in ({1, 2, 3, 4, 6, 9, 12, 18, 36} : Finset ℕ), i) = 91 :=
by
  sorry

end sum_factors_36_eq_91_l231_231662


namespace domain_range_sum_l231_231432

theorem domain_range_sum (m n : ℝ) 
  (h1 : ∀ x, m ≤ x ∧ x ≤ n → 3 * m ≤ -x ^ 2 + 2 * x ∧ -x ^ 2 + 2 * x ≤ 3 * n)
  (h2 : -m ^ 2 + 2 * m = 3 * m)
  (h3 : -n ^ 2 + 2 * n = 3 * n) :
  m = -1 ∧ n = 0 ∧ m + n = -1 := 
by 
  sorry

end domain_range_sum_l231_231432


namespace arithmetic_sqrt_of_25_l231_231935

theorem arithmetic_sqrt_of_25 : ∃ (x : ℝ), x^2 = 25 ∧ x = 5 :=
by 
  sorry

end arithmetic_sqrt_of_25_l231_231935


namespace eraser_crayon_difference_l231_231606

def initial_crayons : Nat := 601
def initial_erasers : Nat := 406
def final_crayons : Nat := 336
def final_erasers : Nat := initial_erasers

theorem eraser_crayon_difference :
  final_erasers - final_crayons = 70 :=
by
  sorry

end eraser_crayon_difference_l231_231606


namespace bobs_fruit_drink_cost_l231_231983

theorem bobs_fruit_drink_cost
  (cost_soda : ℕ)
  (cost_hamburger : ℕ)
  (cost_sandwiches : ℕ)
  (bob_total_spent same_amount : ℕ)
  (andy_spent_eq : same_amount = cost_soda + 2 * cost_hamburger)
  (andy_bob_spent_eq : same_amount = bob_total_spent)
  (bob_sandwich_cost_eq : cost_sandwiches = 3)
  (andy_spent_eq_total : cost_soda = 1)
  (andy_burger_cost : cost_hamburger = 2)
  : bob_total_spent - cost_sandwiches = 2 :=
by
  sorry

end bobs_fruit_drink_cost_l231_231983


namespace quadratic_polynomial_solution_is_zero_l231_231563

-- Definitions based on given conditions
variables (a b c r s : ℝ)
variables (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
variables (h2 : a ≠ b ∧ a ≠ c ∧ b ≠ c)
variables (h3 : r + s = -b / a)
variables (h4 : r * s = c / a)

-- Proposition matching the equivalent proof problem
theorem quadratic_polynomial_solution_is_zero :
  ¬ ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
  (∃ r s : ℝ, (r + s = -b / a) ∧ (r * s = c / a) ∧ (c = r * s ∨ b = r * s ∨ a = r * s) ∧
  (a = r ∨ a = s)) :=
sorry

end quadratic_polynomial_solution_is_zero_l231_231563


namespace intersection_complement_l231_231565

-- Definitions and conditions
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 3}
def B : Set ℕ := {1, 3, 4}
def C_U (B : Set ℕ) : Set ℕ := {x ∈ U | x ∉ B}

-- Theorem statement
theorem intersection_complement :
  (C_U B) ∩ A = {0, 2} := 
by
  -- Proof is not required, so we use sorry
  sorry

end intersection_complement_l231_231565


namespace negate_at_most_two_l231_231410

def atMost (n : Nat) : Prop := ∃ k : Nat, k ≤ n
def atLeast (n : Nat) : Prop := ∃ k : Nat, k ≥ n

theorem negate_at_most_two : ¬ atMost 2 ↔ atLeast 3 := by
  sorry

end negate_at_most_two_l231_231410


namespace value_of_expression_l231_231317

variable (x y : ℝ)

theorem value_of_expression 
  (h1 : x + Real.sqrt (x * y) + y = 9)
  (h2 : x^2 + x * y + y^2 = 27) :
  x - Real.sqrt (x * y) + y = 3 :=
sorry

end value_of_expression_l231_231317


namespace triangle_is_isosceles_l231_231164

theorem triangle_is_isosceles (A B C : ℝ)
  (h : Real.log (Real.sin A) - Real.log (Real.cos B) - Real.log (Real.sin C) = Real.log 2) :
  ∃ a b c : ℝ, a = b ∨ b = c ∨ a = c := 
sorry

end triangle_is_isosceles_l231_231164


namespace arrangement_count_l231_231681

theorem arrangement_count (n_classes : ℕ) (n_factories : ℕ) (classes_per_factory : Finset (Finset ℕ)) :
  n_classes = 5 → n_factories = 4 → 
  (∀ f ∈ classes_per_factory, Finset.card f ≥ 1) → 
  Finset.card classes_per_factory = n_factories →
  ∑ f in classes_per_factory, Finset.card f = n_classes →
  ∃ count : ℕ, count = 240 :=
begin
  intros h1 h2 h3 h4 h5,
  use 240,
  sorry
end

end arrangement_count_l231_231681


namespace ken_ride_time_l231_231904

variables (x y k t : ℝ)

-- Condition 1: It takes Ken 80 seconds to walk down an escalator when it is not moving.
def condition1 : Prop := 80 * x = y

-- Condition 2: It takes Ken 40 seconds to walk down an escalator when it is moving with a 10-second delay.
def condition2 : Prop := 50 * (x + k) = y

-- Condition 3: There is a 10-second delay before the escalator starts moving.
def condition3 : Prop := t = y / k + 10

-- Related Speed
def condition4 : Prop := k = 0.6 * x

-- Proposition: The time Ken takes to ride the escalator down without walking, including the delay, is 143 seconds.
theorem ken_ride_time {x y k t : ℝ} (h1 : condition1 x y) (h2 : condition2 x y k) (h3 : condition3 y k t) (h4 : condition4 x k) :
  t = 143 :=
by sorry

end ken_ride_time_l231_231904


namespace max_rect_area_l231_231095

theorem max_rect_area (l w : ℤ) (h1 : 2 * l + 2 * w = 40) (h2 : 0 < l) (h3 : 0 < w) : 
  l * w ≤ 100 :=
by sorry

end max_rect_area_l231_231095


namespace car_daily_rental_cost_l231_231084

theorem car_daily_rental_cost 
  (x : ℝ)
  (cost_per_mile : ℝ)
  (budget : ℝ)
  (miles : ℕ)
  (h1 : cost_per_mile = 0.18)
  (h2 : budget = 75)
  (h3 : miles = 250)
  (h4 : x + (miles * cost_per_mile) = budget) : 
  x = 30 := 
sorry

end car_daily_rental_cost_l231_231084


namespace total_hunts_is_21_l231_231585

-- Define the initial conditions
def Sam_hunts : Nat := 6
def Rob_hunts : Nat := Sam_hunts / 2
def Rob_Sam_total_hunt : Nat := Sam_hunts + Rob_hunts
def Mark_hunts : Nat := Rob_Sam_total_hunt / 3
def Peter_hunts : Nat := Mark_hunts * 3

-- The main theorem to prove
theorem total_hunts_is_21 : Sam_hunts + Rob_hunts + Mark_hunts + Peter_hunts = 21 :=
by
  sorry

end total_hunts_is_21_l231_231585


namespace math_proof_problem_l231_231715

-- Definitions for conditions:
def condition1 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 3 / 2) = -f x
def condition2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x - 3 / 4) = -f (- (x - 3 / 4))

-- Statements to prove:
def statement1 (f : ℝ → ℝ) : Prop := ∃ p, p ≠ 0 ∧ ∀ x, f (x + p) = f x
def statement2 (f : ℝ → ℝ) : Prop := ∀ x, f (-(3 / 4) - x) = f (-(3 / 4) + x)
def statement3 (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def statement4 (f : ℝ → ℝ) : Prop := ¬(∀ x y : ℝ, x < y → f x ≤ f y)

theorem math_proof_problem (f : ℝ → ℝ) (h1 : condition1 f) (h2 : condition2 f) :
  statement1 f ∧ statement2 f ∧ statement3 f ∧ statement4 f :=
by
  sorry

end math_proof_problem_l231_231715


namespace sum_of_factors_36_eq_91_l231_231654

-- Define what it means to be a whole-number factor of 36
def is_factor_of_36 (n : ℕ) : Prop := n ∣ 36

-- Define the sum of all factors of 36
noncomputable def sum_factors_of_36 : ℕ := 
  ∑ n in Finset.filter is_factor_of_36 (Finset.range 37), n

-- State the theorem
theorem sum_of_factors_36_eq_91 : 
  sum_factors_of_36 = 91 := 
by 
  sorry

end sum_of_factors_36_eq_91_l231_231654


namespace jason_fishes_on_day_12_l231_231177

def initial_fish_count : ℕ := 10

def fish_on_day (n : ℕ) : ℕ :=
  if n = 0 then initial_fish_count else
  (match n with
  | 1 => 10 * 3
  | 2 => 30 * 3
  | 3 => 90 * 3
  | 4 => 270 * 3 * 3 / 5 -- removes fish according to rule
  | 5 => (270 * 3 * 3 / 5) * 3
  | 6 => ((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7 -- removes fish according to rule
  | 7 => (((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3
  | 8 => ((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25
  | 9 => (((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3
  | 10 => ((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) - (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) / 2)
  | 11 => (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) - (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) / 2)) * 3
  | 12 => (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) - (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) / 2)) * 3 + (3 * (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) - (((((((270 * 3 * 3 / 5) * 3 * 3) * 4 / 7) * 3) + 25) * 3) / 2)) * 3) + 5
  | _ => 0
  )
 
theorem jason_fishes_on_day_12 : fish_on_day 12 = 1220045 := 
  by sorry

end jason_fishes_on_day_12_l231_231177


namespace smallest_b_l231_231217

theorem smallest_b
  (a b : ℕ)
  (h_pos : 0 < b)
  (h_diff : a - b = 8)
  (h_gcd : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) :
  b = 4 := sorry

end smallest_b_l231_231217


namespace PQR_product_l231_231290

def PQR_condition (P Q R S : ℕ) : Prop :=
  P + Q + R + S = 100 ∧
  ∃ x : ℕ, P = x - 4 ∧ Q = x + 4 ∧ R = x / 4 ∧ S = 4 * x

theorem PQR_product (P Q R S : ℕ) (h : PQR_condition P Q R S) : P * Q * R * S = 61440 :=
by 
  sorry

end PQR_product_l231_231290


namespace hyperbola_eccentricity_sqrt5_l231_231723

noncomputable def eccentricity_of_hyperbola (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b/a)^2)

theorem hyperbola_eccentricity_sqrt5
  (a b : ℝ)
  (h : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (y = x^2 + 1) → (x, y) = (1, 2)) :
  eccentricity_of_hyperbola a b = Real.sqrt 5 :=
by sorry

end hyperbola_eccentricity_sqrt5_l231_231723
