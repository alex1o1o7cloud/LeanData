import Mathlib

namespace find_opposite_of_neg_half_l2038_203888

-- Define the given number
def given_num : ℚ := -1/2

-- Define what it means to find the opposite of a number
def opposite (x : ℚ) : ℚ := -x

-- State the theorem
theorem find_opposite_of_neg_half : opposite given_num = 1/2 :=
by
  -- Proof is omitted for now
  sorry

end find_opposite_of_neg_half_l2038_203888


namespace farmer_steven_total_days_l2038_203856

theorem farmer_steven_total_days 
(plow_acres_per_day : ℕ)
(mow_acres_per_day : ℕ)
(farmland_acres : ℕ)
(grassland_acres : ℕ)
(h_plow : plow_acres_per_day = 10)
(h_mow : mow_acres_per_day = 12)
(h_farmland : farmland_acres = 55)
(h_grassland : grassland_acres = 30) :
((farmland_acres / plow_acres_per_day) + (grassland_acres / mow_acres_per_day) = 8) := by
  sorry

end farmer_steven_total_days_l2038_203856


namespace man_speed_with_the_stream_l2038_203896

def speed_with_the_stream (V_m V_s : ℝ) : Prop :=
  V_m + V_s = 2

theorem man_speed_with_the_stream (V_m V_s : ℝ) (h1 : V_m - V_s = 2) (h2 : V_m = 2) : speed_with_the_stream V_m V_s :=
by
  sorry

end man_speed_with_the_stream_l2038_203896


namespace mica_should_have_28_26_euros_l2038_203828

namespace GroceryShopping

def pasta_cost : ℝ := 3 * 1.70
def ground_beef_cost : ℝ := 0.5 * 8.20
def pasta_sauce_base_cost : ℝ := 3 * 2.30
def pasta_sauce_discount : ℝ := pasta_sauce_base_cost * 0.10
def pasta_sauce_discounted_cost : ℝ := pasta_sauce_base_cost - pasta_sauce_discount
def quesadillas_cost : ℝ := 11.50

def total_cost_before_vat : ℝ :=
  pasta_cost + ground_beef_cost + pasta_sauce_discounted_cost + quesadillas_cost

def vat : ℝ := total_cost_before_vat * 0.05

def total_cost_including_vat : ℝ := total_cost_before_vat + vat

theorem mica_should_have_28_26_euros :
  total_cost_including_vat = 28.26 := by
  -- This is the statement without the proof. 
  sorry

end GroceryShopping

end mica_should_have_28_26_euros_l2038_203828


namespace probability_of_three_given_sum_seven_l2038_203870

theorem probability_of_three_given_sum_seven : 
  (∃ (dice1 dice2 : ℕ), (1 ≤ dice1 ∧ dice1 ≤ 6 ∧ 1 ≤ dice2 ∧ dice2 ≤ 6) ∧ (dice1 + dice2 = 7) 
    ∧ (dice1 = 3 ∨ dice2 = 3)) →
  (∃ (dice1 dice2 : ℕ), (1 ≤ dice1 ∧ dice1 ≤ 6 ∧ 1 ≤ dice2 ∧ dice2 ≤ 6) ∧ (dice1 + dice2 = 7)) →
  ∃ (p : ℚ), p = 1/3 :=
by 
  sorry

end probability_of_three_given_sum_seven_l2038_203870


namespace two_digit_number_with_tens_5_l2038_203891

-- Definitions and conditions
variable (A : Nat)

-- Problem statement as a Lean theorem
theorem two_digit_number_with_tens_5 (hA : A < 10) : (10 * 5 + A) = 50 + A := by
  sorry

end two_digit_number_with_tens_5_l2038_203891


namespace sum_of_consecutive_integers_eq_pow_of_two_l2038_203861

theorem sum_of_consecutive_integers_eq_pow_of_two (n : ℕ) : 
  (∀ a b : ℕ, a < b → 2 * n ≠ (a + b) * (b - a + 1)) ↔ ∃ k : ℕ, n = 2 ^ k := 
sorry

end sum_of_consecutive_integers_eq_pow_of_two_l2038_203861


namespace symmetric_to_origin_l2038_203815

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem symmetric_to_origin (p : ℝ × ℝ) (h : p = (3, -1)) : symmetric_point p = (-3, 1) :=
by
  -- This is just the statement; the proof is not provided.
  sorry

end symmetric_to_origin_l2038_203815


namespace record_expenditure_l2038_203853

theorem record_expenditure (income_recording : ℤ) (expenditure_amount : ℤ) (h : income_recording = 20) : -expenditure_amount = -50 :=
by sorry

end record_expenditure_l2038_203853


namespace find_digit_A_l2038_203842

theorem find_digit_A :
  ∃ A : ℕ, 
    2 * 10^6 + A * 10^5 + 9 * 10^4 + 9 * 10^3 + 5 * 10^2 + 6 * 10^1 + 1 = (3 * (523 + A)) ^ 2 
    ∧ A = 4 :=
by
  sorry

end find_digit_A_l2038_203842


namespace nth_permutation_2013_eq_3546127_l2038_203847

-- Given the digits 1 through 7, there are 7! = 5040 permutations.
-- We want to prove that the 2013th permutation in ascending order is 3546127.

def digits : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def nth_permutation (n : ℕ) (digits : List ℕ) : List ℕ :=
  sorry

theorem nth_permutation_2013_eq_3546127 :
  nth_permutation 2013 digits = [3, 5, 4, 6, 1, 2, 7] :=
sorry

end nth_permutation_2013_eq_3546127_l2038_203847


namespace handshake_problem_l2038_203801

theorem handshake_problem (n : ℕ) (h : n * (n - 1) / 2 = 1770) : n = 60 :=
sorry

end handshake_problem_l2038_203801


namespace initial_customers_correct_l2038_203886

def initial_customers (remaining : ℕ) (left : ℕ) : ℕ := remaining + left

theorem initial_customers_correct :
  initial_customers 12 9 = 21 :=
by
  sorry

end initial_customers_correct_l2038_203886


namespace jungkook_seokjin_books_l2038_203825

/-- Given the number of books Jungkook and Seokjin originally had and the number of books they 
   bought, prove that Jungkook has 7 more books than Seokjin. -/
theorem jungkook_seokjin_books
  (jungkook_initial : ℕ)
  (seokjin_initial : ℕ)
  (jungkook_bought : ℕ)
  (seokjin_bought : ℕ)
  (h1 : jungkook_initial = 28)
  (h2 : seokjin_initial = 28)
  (h3 : jungkook_bought = 18)
  (h4 : seokjin_bought = 11) :
  (jungkook_initial + jungkook_bought) - (seokjin_initial + seokjin_bought) = 7 :=
by
  sorry

end jungkook_seokjin_books_l2038_203825


namespace largest_n_in_base10_l2038_203808

-- Definitions corresponding to the problem conditions
def n_eq_base8_expr (A B C : ℕ) : ℕ := 64 * A + 8 * B + C
def n_eq_base12_expr (A B C : ℕ) : ℕ := 144 * C + 12 * B + A

-- Problem statement translated into Lean
theorem largest_n_in_base10 (n A B C : ℕ) (h1 : n = n_eq_base8_expr A B C) 
    (h2 : n = n_eq_base12_expr A B C) (hA : A < 8) (hB : B < 8) (hC : C < 12) (h_pos: n > 0) : 
    n ≤ 509 :=
sorry

end largest_n_in_base10_l2038_203808


namespace days_in_month_find_days_in_month_l2038_203819

noncomputable def computers_per_thirty_minutes : ℕ := 225 / 100 -- representing 2.25
def monthly_computers : ℕ := 3024
def hours_per_day : ℕ := 24

theorem days_in_month (computers_per_hour : ℕ) (daily_production : ℕ) : ℕ :=
  let computers_per_hour := (2 * computers_per_thirty_minutes)
  let daily_production := (computers_per_hour * hours_per_day)
  (monthly_computers / daily_production)

theorem find_days_in_month :
  days_in_month (2 * computers_per_thirty_minutes) ((2 * computers_per_thirty_minutes) * hours_per_day) = 28 :=
by
  sorry

end days_in_month_find_days_in_month_l2038_203819


namespace time_to_fill_bucket_completely_l2038_203871

-- Define the conditions given in the problem
def time_to_fill_two_thirds (time_filled: ℕ) : ℕ := 90

-- Define what we need to prove
theorem time_to_fill_bucket_completely (time_filled: ℕ) : 
  time_to_fill_two_thirds time_filled = 90 → time_filled = 135 :=
by
  sorry

end time_to_fill_bucket_completely_l2038_203871


namespace greatest_GCD_of_product_7200_l2038_203885

theorem greatest_GCD_of_product_7200 :
  ∃ (a b : ℕ), a * b = 7200 ∧ ∀ d, (d ∣ a ∧ d ∣ b) → d ≤ 60 :=
by
  sorry

end greatest_GCD_of_product_7200_l2038_203885


namespace bowling_ball_weight_l2038_203821

-- Definitions for the conditions
def kayak_weight : ℕ := 36
def total_weight_of_two_kayaks := 2 * kayak_weight
def total_weight_of_nine_bowling_balls (ball_weight : ℕ) := 9 * ball_weight  

theorem bowling_ball_weight (w : ℕ) (h1 : total_weight_of_two_kayaks = total_weight_of_nine_bowling_balls w) : w = 8 :=
by
  -- Proof goes here
  sorry

end bowling_ball_weight_l2038_203821


namespace number_properties_l2038_203845

def number : ℕ := 52300600

def position_of_2 : ℕ := 10^6

def value_of_2 : ℕ := 20000000

def position_of_5 : ℕ := 10^7

def value_of_5 : ℕ := 50000000

def read_number : String := "five hundred twenty-three million six hundred"

theorem number_properties : 
  position_of_2 = (10^6) ∧ value_of_2 = 20000000 ∧ 
  position_of_5 = (10^7) ∧ value_of_5 = 50000000 ∧ 
  read_number = "five hundred twenty-three million six hundred" :=
by sorry

end number_properties_l2038_203845


namespace f_0_plus_f_1_l2038_203865

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom f_neg1 : f (-1) = 2

theorem f_0_plus_f_1 : f 0 + f 1 = -2 :=
by
  sorry

end f_0_plus_f_1_l2038_203865


namespace megatech_budget_allocation_l2038_203852

theorem megatech_budget_allocation :
  let microphotonics := 14
  let food_additives := 10
  let gmo := 24
  let industrial_lubricants := 8
  let basic_astrophysics := 25
  microphotonics + food_additives + gmo + industrial_lubricants + basic_astrophysics = 81 →
  100 - 81 = 19 :=
by
  intros
  -- We are given the sums already, so directly calculate the remaining percentage.
  sorry

end megatech_budget_allocation_l2038_203852


namespace addition_problem_l2038_203824

theorem addition_problem (m n p q : ℕ) (Hm : m = 2) (Hn : 2 + n + 7 + 5 = 20) (Hp : 1 + 6 + p + 8 = 24) (Hq : 3 + 2 + q = 12) (Hpositives : 0 < m ∧ 0 < n ∧ 0 < p ∧ 0 < q) :
  m + n + p + q = 24 :=
sorry

end addition_problem_l2038_203824


namespace ratio_avg_speed_round_trip_l2038_203839

def speed_boat := 20
def speed_current := 4
def distance := 2

theorem ratio_avg_speed_round_trip :
  let downstream_speed := speed_boat + speed_current
  let upstream_speed := speed_boat - speed_current
  let time_down := distance / downstream_speed
  let time_up := distance / upstream_speed
  let total_time := time_down + time_up
  let total_distance := distance + distance
  let avg_speed := total_distance / total_time
  avg_speed / speed_boat = 24 / 25 :=
by sorry

end ratio_avg_speed_round_trip_l2038_203839


namespace cost_of_each_orange_l2038_203840

theorem cost_of_each_orange (calories_per_orange : ℝ) (total_money : ℝ) (calories_needed : ℝ) (money_left : ℝ) :
  calories_per_orange = 80 → 
  total_money = 10 → 
  calories_needed = 400 → 
  money_left = 4 → 
  (total_money - money_left) / (calories_needed / calories_per_orange) = 1.2 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end cost_of_each_orange_l2038_203840


namespace composite_probability_l2038_203816

/--
Given that a number selected at random from the first 50 natural numbers,
where 1 is neither prime nor composite,
the probability of selecting a composite number is 34/49.
-/
theorem composite_probability :
  let total_numbers := 50
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  let num_primes := primes.length
  let num_composites := total_numbers - num_primes - 1
  let probability_composite := (num_composites : ℚ) / (total_numbers - 1)
  probability_composite = 34 / 49 :=
by {
  sorry
}

end composite_probability_l2038_203816


namespace x_cube_plus_y_cube_l2038_203876

theorem x_cube_plus_y_cube (x y : ℝ) (h₁ : x + y = 1) (h₂ : x^2 + y^2 = 3) : x^3 + y^3 = 4 :=
sorry

end x_cube_plus_y_cube_l2038_203876


namespace find_k_eq_neg_four_thirds_l2038_203820

-- Definitions based on conditions
def hash_p (k : ℚ) (p : ℚ) : ℚ := k * p + 20

-- Using the initial condition
def triple_hash_18 (k : ℚ) : ℚ :=
  let hp := hash_p k 18
  let hhp := hash_p k hp
  hash_p k hhp

-- The Lean statement for the desired proof
theorem find_k_eq_neg_four_thirds (k : ℚ) (h : triple_hash_18 k = -4) : k = -4 / 3 :=
sorry

end find_k_eq_neg_four_thirds_l2038_203820


namespace walter_bus_time_l2038_203803

/--
Walter wakes up at 6:30 a.m., leaves for the bus at 7:30 a.m., attends 7 classes that each last 45 minutes,
enjoys a 40-minute lunch, and spends 2.5 hours of additional time at school for activities.
He takes the bus home and arrives at 4:30 p.m.
Prove that Walter spends 35 minutes on the bus.
-/
theorem walter_bus_time : 
  let total_time_away := 9 * 60 -- in minutes
  let class_time := 7 * 45 -- in minutes
  let lunch_time := 40 -- in minutes
  let additional_school_time := 2.5 * 60 -- in minutes
  total_time_away - (class_time + lunch_time + additional_school_time) = 35 := 
by
  sorry

end walter_bus_time_l2038_203803


namespace balls_in_boxes_l2038_203899

theorem balls_in_boxes : 
  ∀ (balls boxes : ℕ), (balls = 6) → (boxes = 3) → 
  (∃ ways : ℕ, ways = 7) :=
by
  sorry

end balls_in_boxes_l2038_203899


namespace compare_abc_l2038_203878

noncomputable def a : ℝ := Real.log 10 / Real.log 5
noncomputable def b : ℝ := Real.log 12 / Real.log 6
noncomputable def c : ℝ := Real.log 14 / Real.log 7

theorem compare_abc : a > b ∧ b > c := by
  sorry

end compare_abc_l2038_203878


namespace hexagon_midpoints_equilateral_l2038_203844

noncomputable def inscribed_hexagon_midpoints_equilateral (r : ℝ) (h : ℝ) 
  (hex : ∀ (A B C D E F : ℝ) (O : ℝ), 
    true) : Prop :=
  ∀ (M N P : ℝ), 
    true

theorem hexagon_midpoints_equilateral (r : ℝ) (h : ℝ) 
  (hex : ∀ (A B C D E F : ℝ) (O : ℝ), 
    true) : 
  inscribed_hexagon_midpoints_equilateral r h hex :=
sorry

end hexagon_midpoints_equilateral_l2038_203844


namespace arithmetic_sequence_problem_l2038_203881

theorem arithmetic_sequence_problem :
  let sum_first_sequence := (100 / 2) * (2501 + 2600)
  let sum_second_sequence := (100 / 2) * (401 + 500)
  let sum_third_sequence := (50 / 2) * (401 + 450)
  sum_first_sequence - sum_second_sequence - sum_third_sequence = 188725 :=
by
  let sum_first_sequence := (100 / 2) * (2501 + 2600)
  let sum_second_sequence := (100 / 2) * (401 + 500)
  let sum_third_sequence := (50 / 2) * (401 + 450)
  sorry

end arithmetic_sequence_problem_l2038_203881


namespace circle_area_polar_eq_l2038_203811

theorem circle_area_polar_eq (r θ : ℝ) :
    (r = 3 * Real.cos θ - 4 * Real.sin θ) → (π * (5 / 2) ^ 2 = 25 * π / 4) :=
by 
  sorry

end circle_area_polar_eq_l2038_203811


namespace mary_total_spent_l2038_203877

def store1_shirt : ℝ := 13.04
def store1_jacket : ℝ := 12.27
def store2_shoes : ℝ := 44.15
def store2_dress : ℝ := 25.50
def hat_price : ℝ := 9.99
def discount : ℝ := 0.10
def store4_handbag : ℝ := 30.93
def store4_scarf : ℝ := 7.42
def sunglasses_price : ℝ := 20.75
def sales_tax : ℝ := 0.05

def store1_total : ℝ := store1_shirt + store1_jacket
def store2_total : ℝ := store2_shoes + store2_dress
def store3_total : ℝ := 
  let hat_cost := hat_price * 2
  let discount_amt := hat_cost * discount
  hat_cost - discount_amt
def store4_total : ℝ := store4_handbag + store4_scarf
def store5_total : ℝ := 
  let tax := sunglasses_price * sales_tax
  sunglasses_price + tax

def total_spent : ℝ := store1_total + store2_total + store3_total + store4_total + store5_total

theorem mary_total_spent : total_spent = 173.08 := sorry

end mary_total_spent_l2038_203877


namespace seashells_total_l2038_203889

theorem seashells_total :
  let monday := 5
  let tuesday := 7 - 3
  let wednesday := (2 * monday) / 2
  let thursday := 3 * 7
  monday + tuesday + wednesday + thursday = 35 :=
by
  sorry

end seashells_total_l2038_203889


namespace lily_coffee_budget_l2038_203884

variable (initial_amount celery_price cereal_original_price bread_price milk_original_price potato_price : ℕ)
variable (cereal_discount milk_discount number_of_potatoes : ℕ)

theorem lily_coffee_budget 
  (h_initial_amount : initial_amount = 60)
  (h_celery_price : celery_price = 5)
  (h_cereal_original_price : cereal_original_price = 12)
  (h_bread_price : bread_price = 8)
  (h_milk_original_price : milk_original_price = 10)
  (h_potato_price : potato_price = 1)
  (h_number_of_potatoes : number_of_potatoes = 6)
  (h_cereal_discount : cereal_discount = 50)
  (h_milk_discount : milk_discount = 10) :
  initial_amount - (celery_price + (cereal_original_price * cereal_discount / 100) + bread_price + (milk_original_price - (milk_original_price * milk_discount / 100)) + (potato_price * number_of_potatoes)) = 26 :=
by
  sorry

end lily_coffee_budget_l2038_203884


namespace compute_fraction_equation_l2038_203806

theorem compute_fraction_equation :
  (8 * (2 / 3: ℚ)^4 + 2 = 290 / 81) :=
sorry

end compute_fraction_equation_l2038_203806


namespace calories_per_serving_l2038_203800

theorem calories_per_serving (x : ℕ) (total_calories bread_calories servings : ℕ)
    (h1: total_calories = 500) (h2: bread_calories = 100) (h3: servings = 2)
    (h4: total_calories = bread_calories + (servings * x)) :
    x = 200 :=
by
  sorry

end calories_per_serving_l2038_203800


namespace polynomial_root_expression_l2038_203805

theorem polynomial_root_expression (a b : ℂ) 
  (h₁ : a + b = 5) (h₂ : a * b = 6) : 
  a^4 + a^5 * b^3 + a^3 * b^5 + b^4 = 2905 := by
  sorry

end polynomial_root_expression_l2038_203805


namespace integer_sum_19_l2038_203813

variable (p q r s : ℤ)

theorem integer_sum_19 (h1 : p - q + r = 4) 
                       (h2 : q - r + s = 5) 
                       (h3 : r - s + p = 7) 
                       (h4 : s - p + q = 3) :
                       p + q + r + s = 19 :=
by
  sorry

end integer_sum_19_l2038_203813


namespace ratio_of_boys_to_girls_l2038_203846

-- Define the given conditions and provable statement
theorem ratio_of_boys_to_girls (S G : ℕ) (h : (2/3 : ℚ) * G = (1/5 : ℚ) * S) : (S - G) * 3 = 7 * G :=
by
  -- This is a placeholder for solving the proof
  sorry

end ratio_of_boys_to_girls_l2038_203846


namespace distance_from_mountains_l2038_203893

/-- Given distances and scales from the problem description -/
def distance_between_mountains_map : ℤ := 312 -- in inches
def actual_distance_between_mountains : ℤ := 136 -- in km
def scale_A : ℤ := 1 -- 1 inch represents 1 km
def scale_B : ℤ := 2 -- 1 inch represents 2 km
def distance_from_mountain_A_map : ℤ := 25 -- in inches
def distance_from_mountain_B_map : ℤ := 40 -- in inches

/-- Prove the actual distances from Ram's camp to the mountains -/
theorem distance_from_mountains (dA dB : ℤ) :
  (dA = distance_from_mountain_A_map * scale_A) ∧ 
  (dB = distance_from_mountain_B_map * scale_B) :=
by {
  sorry -- Proof placeholder
}

end distance_from_mountains_l2038_203893


namespace decaf_percentage_correct_l2038_203875

def initial_stock : ℝ := 400
def initial_decaf_percent : ℝ := 0.20
def additional_stock : ℝ := 100
def additional_decaf_percent : ℝ := 0.70

theorem decaf_percentage_correct :
  ((initial_decaf_percent * initial_stock + additional_decaf_percent * additional_stock) / (initial_stock + additional_stock)) * 100 = 30 :=
by
  sorry

end decaf_percentage_correct_l2038_203875


namespace find_integer_a_l2038_203897

theorem find_integer_a (a : ℤ) : (∃ x : ℕ, a * x = 3) ↔ a = 1 ∨ a = 3 :=
by
  sorry

end find_integer_a_l2038_203897


namespace arithmetic_progression_contains_sixth_power_l2038_203895

theorem arithmetic_progression_contains_sixth_power (a b : ℕ) (h_ap_pos : ∀ t : ℕ, a + b * t > 0)
  (h_contains_square : ∃ n : ℕ, ∃ t : ℕ, a + b * t = n^2)
  (h_contains_cube : ∃ m : ℕ, ∃ t : ℕ, a + b * t = m^3) :
  ∃ k : ℕ, ∃ t : ℕ, a + b * t = k^6 :=
sorry

end arithmetic_progression_contains_sixth_power_l2038_203895


namespace age_proof_l2038_203818

noncomputable def father_age_current := 33
noncomputable def xiaolin_age_current := 3

def father_age (X : ℕ) := 11 * X
def future_father_age (F : ℕ) := F + 7
def future_xiaolin_age (X : ℕ) := X + 7

theorem age_proof (F X : ℕ) (h1 : F = father_age X) 
  (h2 : future_father_age F = 4 * future_xiaolin_age X) : 
  F = father_age_current ∧ X = xiaolin_age_current :=
by 
  sorry

end age_proof_l2038_203818


namespace fraction_div_addition_l2038_203860

theorem fraction_div_addition : ( (3 / 7 : ℚ) / 4) + (1 / 28) = (1 / 7) :=
  sorry

end fraction_div_addition_l2038_203860


namespace arithmetic_sequence_a5_l2038_203814

theorem arithmetic_sequence_a5
  (a : ℕ → ℤ) -- a is the arithmetic sequence function
  (S : ℕ → ℤ) -- S is the sum of the first n terms of the sequence
  (h1 : S 5 = 2 * S 4) -- Condition S_5 = 2S_4
  (h2 : a 2 + a 4 = 8) -- Condition a_2 + a_4 = 8
  (hS : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2) -- Definition of S_n
  (ha : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) -- Definition of a_n
: a 5 = 10 := 
by
  -- proof
  sorry

end arithmetic_sequence_a5_l2038_203814


namespace sandbox_side_length_l2038_203855

theorem sandbox_side_length (side_length : ℝ) (sand_sq_inches_per_pound : ℝ := 80 / 30) (total_sand_pounds : ℝ := 600) :
  (side_length ^ 2 = total_sand_pounds * sand_sq_inches_per_pound) → side_length = 40 := 
by
  sorry

end sandbox_side_length_l2038_203855


namespace quadrant_606_quadrant_minus_950_same_terminal_side_quadrant_minus_97_l2038_203892

theorem quadrant_606 (θ : ℝ) : θ = 606 → (180 < (θ % 360) ∧ (θ % 360) < 270) := by
  sorry

theorem quadrant_minus_950 (θ : ℝ) : θ = -950 → (90 < (θ % 360) ∧ (θ % 360) < 180) := by
  sorry

theorem same_terminal_side (α k : ℤ) : (α = -457 + k * 360) ↔ (∃ n : ℤ, α = -457 + n * 360) := by
  sorry

theorem quadrant_minus_97 (θ : ℝ) : θ = -97 → (180 < (θ % 360) ∧ (θ % 360) < 270) := by
  sorry

end quadrant_606_quadrant_minus_950_same_terminal_side_quadrant_minus_97_l2038_203892


namespace correct_statement_l2038_203843

theorem correct_statement :
  (Real.sqrt (9 / 16) = 3 / 4) :=
by
  sorry

end correct_statement_l2038_203843


namespace problem_statement_l2038_203804

-- Definitions for given conditions
variables (a b m n x : ℤ)

-- Assuming conditions: a = -b, mn = 1, and |x| = 2
axiom opp_num : a = -b
axiom recip : m * n = 1
axiom abs_x : |x| = 2

-- Problem statement to prove
theorem problem_statement :
  -2 * m * n + (a + b) / 2023 + x * x = 2 :=
by 
  sorry

end problem_statement_l2038_203804


namespace find_natural_numbers_l2038_203866

theorem find_natural_numbers (n k : ℕ) (h : 2^n - 5^k = 7) : n = 5 ∧ k = 2 :=
by
  sorry

end find_natural_numbers_l2038_203866


namespace units_digit_7_pow_2050_l2038_203859

theorem units_digit_7_pow_2050 : (7 ^ 2050) % 10 = 9 := 
by 
  sorry

end units_digit_7_pow_2050_l2038_203859


namespace total_days_of_work_l2038_203835

theorem total_days_of_work (r1 r2 r3 r4 : ℝ) (h1 : r1 = 1 / 12) (h2 : r2 = 1 / 8) (h3 : r3 = 1 / 24) (h4 : r4 = 1 / 16) : 
  (1 / (r1 + r2 + r3 + r4) = 3.2) :=
by 
  sorry

end total_days_of_work_l2038_203835


namespace glass_sphere_wall_thickness_l2038_203809

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

end glass_sphere_wall_thickness_l2038_203809


namespace sibling_age_difference_l2038_203898

theorem sibling_age_difference 
  (x : ℕ) 
  (h : 3 * x + 2 * x + 1 * x = 90) : 
  3 * x - x = 30 := 
by 
  sorry

end sibling_age_difference_l2038_203898


namespace smallest_possible_a_plus_b_l2038_203812

theorem smallest_possible_a_plus_b :
  ∃ (a b : ℕ), (0 < a ∧ 0 < b) ∧ (2^10 * 7^3 = a^b) ∧ (a + b = 350753) :=
sorry

end smallest_possible_a_plus_b_l2038_203812


namespace triangle_cosine_condition_l2038_203854

variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to angles A, B, and C

-- Definitions according to the problem conditions
def law_of_sines (a b : ℝ) (A B : ℝ) : Prop :=
  a / Real.sin A = b / Real.sin B

theorem triangle_cosine_condition (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : law_of_sines a b A B)
  (h1 : a > b) : Real.cos (2 * A) < Real.cos (2 * B) ↔ a > b :=
by
  sorry

end triangle_cosine_condition_l2038_203854


namespace number_composition_l2038_203807

theorem number_composition :
  5 * 100000 + 6 * 100 + 3 * 10 + 6 * 0.01 = 500630.06 := 
by 
  sorry

end number_composition_l2038_203807


namespace integral_f_x_l2038_203851

theorem integral_f_x (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2 * ∫ t in (0 : ℝ)..1, f t) : 
  ∫ t in (0 : ℝ)..1, f t = -1 / 3 := by
  sorry

end integral_f_x_l2038_203851


namespace quadratic_equation_roots_l2038_203874

theorem quadratic_equation_roots (m n : ℝ) 
  (h_sum : m + n = -3) 
  (h_prod : m * n = 1) 
  (h_equation : m^2 + 3 * m + 1 = 0) :
  (3 * m + 1) / (m^3 * n) = -1 := 
by sorry

end quadratic_equation_roots_l2038_203874


namespace notecard_calculation_l2038_203823

theorem notecard_calculation (N E : ℕ) (h₁ : N - E = 80) (h₂ : N = 3 * E) : N = 120 :=
sorry

end notecard_calculation_l2038_203823


namespace balls_in_boxes_l2038_203848

theorem balls_in_boxes : 
  (number_of_ways : ℕ) = 52 :=
by
  let number_of_balls := 5
  let number_of_boxes := 4
  let balls_indistinguishable := true
  let boxes_distinguishable := true
  let max_balls_per_box := 3
  
  -- Proof omitted
  sorry

end balls_in_boxes_l2038_203848


namespace range_of_g_l2038_203857

noncomputable def g (x : ℝ) : ℤ :=
if x > -3 then
  ⌈1 / ((x + 3)^2)⌉
else
  ⌊1 / ((x + 3)^2)⌋

theorem range_of_g :
  ∀ y : ℤ, (∃ x : ℝ, g x = y) ↔ (∃ n : ℕ, y = n + 1) :=
by sorry

end range_of_g_l2038_203857


namespace m_le_three_l2038_203849

-- Definitions
def setA (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5
def setB (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

-- Theorem statement
theorem m_le_three (m : ℝ) : (∀ x : ℝ, setB m x → setA x) → m ≤ 3 := by
  sorry

end m_le_three_l2038_203849


namespace point_on_circle_l2038_203833

theorem point_on_circle 
    (P : ℝ × ℝ) 
    (h_l1 : 2 * P.1 - 3 * P.2 + 4 = 0)
    (h_l2 : 3 * P.1 - 2 * P.2 + 1 = 0) 
    (h_circle : (P.1 - 2) ^ 2 + (P.2 - 4) ^ 2 = 5) : 
    (P.1 - 2) ^ 2 + (P.2 - 4) ^ 2 = 5 :=
by
  sorry

end point_on_circle_l2038_203833


namespace perimeter_of_region_l2038_203862

-- Define the condition
def area_of_region := 512 -- square centimeters
def number_of_squares := 8

-- Define the presumed perimeter
def presumed_perimeter := 144 -- the correct answer

-- Mathematical statement that needs proof
theorem perimeter_of_region (area_of_region: ℕ) (number_of_squares: ℕ) (presumed_perimeter: ℕ) : 
   area_of_region = 512 ∧ number_of_squares = 8 → presumed_perimeter = 144 :=
by 
  sorry

end perimeter_of_region_l2038_203862


namespace find_common_ratio_l2038_203850

def first_term : ℚ := 4 / 7
def second_term : ℚ := 12 / 7

theorem find_common_ratio (r : ℚ) : second_term = first_term * r → r = 3 :=
by
  sorry

end find_common_ratio_l2038_203850


namespace smallest_value_of_x_l2038_203830

theorem smallest_value_of_x (x : ℝ) (hx : |3 * x + 7| = 26) : x = -11 :=
sorry

end smallest_value_of_x_l2038_203830


namespace circle_equation_equivalence_l2038_203890

theorem circle_equation_equivalence 
    (x y : ℝ) : 
    x^2 + y^2 - 2 * x - 5 = 0 ↔ (x - 1)^2 + y^2 = 6 :=
sorry

end circle_equation_equivalence_l2038_203890


namespace calculate_expression_l2038_203831

theorem calculate_expression :
  (-3)^4 + (-3)^3 + 3^3 + 3^4 = 162 :=
by
  -- Since all necessary conditions are listed in the problem statement, we honor this structure
  -- The following steps are required logically but are not presently necessary for detailed proof means.
  sorry

end calculate_expression_l2038_203831


namespace ratio_rect_prism_l2038_203826

namespace ProofProblem

variables (w l h : ℕ)
def rect_prism (w l h : ℕ) : Prop := w * l * h = 128

theorem ratio_rect_prism (h1 : rect_prism w l h) :
  (w : ℕ) ≠ 0 ∧ (l : ℕ) ≠ 0 ∧ (h : ℕ) ≠ 0 ∧ 
  (∃ k, w = k ∧ l = k ∧ h = 2 * k) :=
sorry

end ProofProblem

end ratio_rect_prism_l2038_203826


namespace trig_equation_solution_l2038_203810

open Real

theorem trig_equation_solution (x : ℝ) (k n : ℤ) :
  (sin (2 * x)) ^ 4 + (sin (2 * x)) ^ 3 * (cos (2 * x)) -
  8 * (sin (2 * x)) * (cos (2 * x)) ^ 3 - 8 * (cos (2 * x)) ^ 4 = 0 ↔
  (∃ k : ℤ, x = -π / 8 + (π * k) / 2) ∨ 
  (∃ n : ℤ, x = (1 / 2) * arctan 2 + (π * n) / 2) := sorry

end trig_equation_solution_l2038_203810


namespace smallest_k_l2038_203838

theorem smallest_k (n k : ℕ) (h1: 2000 < n) (h2: n < 3000)
  (h3: ∀ i, 2 ≤ i → i ≤ k → n % i = i - 1) :
  k = 9 :=
by
  sorry

end smallest_k_l2038_203838


namespace num_valid_sequences_10_transformations_l2038_203802

/-- Define the transformations: 
    L: 90° counterclockwise rotation,
    R: 90° clockwise rotation,
    H: reflection across the x-axis,
    V: reflection across the y-axis. -/
inductive Transformation
| L | R | H | V

/-- Define a function to get the number of valid sequences of transformations
    that bring the vertices E, F, G, H back to their original positions.-/
def countValidSequences : ℕ :=
  56

/-- The theorem to prove that the number of valid sequences
    of 10 transformations resulting in the identity transformation is 56. -/
theorem num_valid_sequences_10_transformations : 
  countValidSequences = 56 :=
sorry

end num_valid_sequences_10_transformations_l2038_203802


namespace unique_positive_solution_eq_15_l2038_203882

theorem unique_positive_solution_eq_15 
  (x : ℝ) 
  (h1 : x > 0) 
  (h2 : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end unique_positive_solution_eq_15_l2038_203882


namespace division_value_l2038_203867

theorem division_value (x : ℚ) (h : (5 / 2) / x = 5 / 14) : x = 7 :=
sorry

end division_value_l2038_203867


namespace zero_in_interval_l2038_203829

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - x^(1/3)

theorem zero_in_interval : ∃ x ∈ Set.Ioo (1/3 : ℝ) (1/2 : ℝ), f x = 0 :=
by
  -- The correct statement only
  sorry

end zero_in_interval_l2038_203829


namespace other_cube_side_length_l2038_203858

theorem other_cube_side_length (s_1 s_2 : ℝ) (h1 : s_1 = 1) (h2 : 6 * s_2^2 / 6 = 36) : s_2 = 6 :=
by
  sorry

end other_cube_side_length_l2038_203858


namespace sum_of_squares_inequality_l2038_203879

theorem sum_of_squares_inequality (a b c : ℝ) : 
  a^2 + b^2 + c^2 ≥ ab + bc + ca :=
by
  sorry

end sum_of_squares_inequality_l2038_203879


namespace base9_digit_divisible_by_13_l2038_203872

theorem base9_digit_divisible_by_13 :
    ∃ (d : ℕ), (0 ≤ d ∧ d ≤ 8) ∧ (13 ∣ (2 * 9^4 + d * 9^3 + 6 * 9^2 + d * 9 + 4)) :=
by
  sorry

end base9_digit_divisible_by_13_l2038_203872


namespace simplify_and_find_ratio_l2038_203817

theorem simplify_and_find_ratio (k : ℤ) : (∃ (c d : ℤ), (∀ x y : ℤ, c = 1 ∧ d = 2 ∧ x = c ∧ y = d → ((6 * k + 12) / 6 = k + 2) ∧ (c / d = 1 / 2))) :=
by
  use 1
  use 2
  sorry

end simplify_and_find_ratio_l2038_203817


namespace find_a_l2038_203868

variables (a b c : ℝ) (A B C : ℝ) (sin : ℝ → ℝ)
variables (sqrt_three_two sqrt_two_two : ℝ)

-- Assume that A = 60 degrees, B = 45 degrees, and b = sqrt(6)
def angle_A : A = π / 3 := by
  sorry

def angle_B : B = π / 4 := by
  sorry

def side_b : b = Real.sqrt 6 := by
  sorry

def sin_60 : sin (π / 3) = sqrt_three_two := by
  sorry

def sin_45 : sin (π / 4) = sqrt_two_two := by
  sorry

-- Prove that a = 3 based on the given conditions
theorem find_a (sin_rule : a / sin A = b / sin B)
  (sin_60_def : sqrt_three_two = Real.sqrt 3 / 2)
  (sin_45_def : sqrt_two_two = Real.sqrt 2 / 2) : a = 3 := by
  sorry

end find_a_l2038_203868


namespace find_a_value_l2038_203832

theorem find_a_value
    (a : ℝ)
    (line : ∀ (x y : ℝ), 3 * x + y + a = 0)
    (circle : ∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y = 0) :
    a = 1 := sorry

end find_a_value_l2038_203832


namespace bookmark_position_second_book_l2038_203822

-- Definitions for the conditions
def pages_per_book := 250
def cover_thickness_ratio := 10
def total_books := 2
def distance_bookmarks_factor := 1 / 3

-- Derived constants
def cover_thickness := cover_thickness_ratio * pages_per_book
def total_pages := (pages_per_book * total_books) + (cover_thickness * total_books * 2)
def distance_between_bookmarks := total_pages * distance_bookmarks_factor
def midpoint_pages_within_book := (pages_per_book / 2) + cover_thickness

-- Definitions for bookmarks positions
def first_bookmark_position := midpoint_pages_within_book
def remaining_pages_after_first_bookmark := distance_between_bookmarks - midpoint_pages_within_book
def second_bookmark_position := remaining_pages_after_first_bookmark - cover_thickness

-- Theorem stating the goal
theorem bookmark_position_second_book :
  35 ≤ second_bookmark_position ∧ second_bookmark_position < 36 :=
sorry

end bookmark_position_second_book_l2038_203822


namespace boston_trip_distance_l2038_203883

theorem boston_trip_distance :
  ∃ d : ℕ, 40 * d = 440 :=
by
  sorry

end boston_trip_distance_l2038_203883


namespace find_g8_l2038_203894

variable (g : ℝ → ℝ)

theorem find_g8 (h1 : ∀ x y : ℝ, g (x + y) = g x + g y) (h2 : g 7 = 8) : g 8 = 64 / 7 :=
sorry

end find_g8_l2038_203894


namespace sqrt_sum_equality_l2038_203827

open Real

theorem sqrt_sum_equality :
  (sqrt (18 - 8 * sqrt 2) + sqrt (18 + 8 * sqrt 2) = 8) :=
sorry

end sqrt_sum_equality_l2038_203827


namespace new_number_formed_l2038_203880

theorem new_number_formed (t u : ℕ) (ht : t < 10) (hu : u < 10) : 3 * 100 + (10 * t + u) = 300 + 10 * t + u := 
by {
  sorry
}

end new_number_formed_l2038_203880


namespace three_wheels_possible_two_wheels_not_possible_l2038_203864

-- Define the conditions as hypotheses
def wheels_spokes (total_spokes_visible : ℕ) (max_spokes_per_wheel : ℕ) (wheels : ℕ) : Prop :=
  total_spokes_visible >= wheels * max_spokes_per_wheel ∧ wheels ≥ 1

-- Prove if a) three wheels is a possible solution
theorem three_wheels_possible : ∃ wheels, wheels = 3 ∧ wheels_spokes 7 3 wheels := by
  sorry

-- Prove if b) two wheels is not a possible solution
theorem two_wheels_not_possible : ¬ ∃ wheels, wheels = 2 ∧ wheels_spokes 7 3 wheels := by
  sorry

end three_wheels_possible_two_wheels_not_possible_l2038_203864


namespace minimum_study_tools_l2038_203863

theorem minimum_study_tools (n : Nat) : n^3 ≥ 366 → n ≥ 8 := by
  intros h
  sorry

end minimum_study_tools_l2038_203863


namespace total_bricks_in_wall_l2038_203837

theorem total_bricks_in_wall :
  let bottom_row_bricks := 18
  let rows := [bottom_row_bricks, bottom_row_bricks - 1, bottom_row_bricks - 2, bottom_row_bricks - 3, bottom_row_bricks - 4]
  (rows.sum = 80) := 
by
  let bottom_row_bricks := 18
  let rows := [bottom_row_bricks, bottom_row_bricks - 1, bottom_row_bricks - 2, bottom_row_bricks - 3, bottom_row_bricks - 4]
  sorry

end total_bricks_in_wall_l2038_203837


namespace no_four_points_with_equal_tangents_l2038_203869

theorem no_four_points_with_equal_tangents :
  ∀ (A B C D : ℝ × ℝ),
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧
    A ≠ C ∧ B ≠ D →
    ¬ (∀ (P Q : ℝ × ℝ), (P = A ∧ Q = B) ∨ (P = C ∧ Q = D) →
      ∃ (M : ℝ × ℝ) (r : ℝ), M ≠ P ∧ M ≠ Q ∧
      (dist A M = dist C M ∧ dist B M = dist D M ∧
       dist P M > r ∧ dist Q M > r)) :=
by sorry

end no_four_points_with_equal_tangents_l2038_203869


namespace min_dist_l2038_203836

open Complex

theorem min_dist (z w : ℂ) (hz : abs (z - (2 - 5 * I)) = 2) (hw : abs (w - (-3 + 4 * I)) = 4) :
  ∃ d, d = abs (z - w) ∧ d ≥ (Real.sqrt 106 - 6) := sorry

end min_dist_l2038_203836


namespace positive_two_digit_integers_remainder_4_div_9_l2038_203841

theorem positive_two_digit_integers_remainder_4_div_9 : ∃ (n : ℕ), 
  (10 ≤ 9 * n + 4) ∧ (9 * n + 4 < 100) ∧ (∃ (k : ℕ), 1 ≤ k ∧ k ≤ 10 ∧ ∀ m, 1 ≤ m ∧ m ≤ 10 → n = k) :=
by
  sorry

end positive_two_digit_integers_remainder_4_div_9_l2038_203841


namespace value_of_x_in_terms_of_z_l2038_203887

variable {z : ℝ} {x y : ℝ}
  
theorem value_of_x_in_terms_of_z (h1 : y = z + 50) (h2 : x = 0.70 * y) : x = 0.70 * z + 35 := 
  sorry

end value_of_x_in_terms_of_z_l2038_203887


namespace tan_sum_eq_l2038_203834

theorem tan_sum_eq (α : ℝ) (h : Real.tan (α + Real.pi / 4) = 2) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = -1/2 :=
by sorry

end tan_sum_eq_l2038_203834


namespace total_number_of_components_l2038_203873

-- Definitions based on the conditions in the problem
def number_of_B_components := 300
def number_of_C_components := 200
def sample_size := 45
def number_of_A_components_drawn := 20
def number_of_C_components_drawn := 10

-- The statement to be proved
theorem total_number_of_components :
  (number_of_A_components_drawn * (number_of_B_components + number_of_C_components) / sample_size) 
  + number_of_B_components 
  + number_of_C_components 
  = 900 := 
by 
  sorry

end total_number_of_components_l2038_203873
