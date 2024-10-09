import Mathlib

namespace weight_of_b_l85_8592

theorem weight_of_b (a b c : ℝ) (h1 : a + b + c = 126) (h2 : a + b = 80) (h3 : b + c = 86) : b = 40 :=
sorry

end weight_of_b_l85_8592


namespace length_of_bridge_l85_8599

-- Define the problem conditions
def length_train : ℝ := 110 -- Length of the train in meters
def speed_kmph : ℝ := 60 -- Speed of the train in kmph

-- Convert speed from kmph to m/s
noncomputable def speed_mps : ℝ := speed_kmph * 1000 / 3600

-- Define the time taken to cross the bridge
def time_seconds : ℝ := 16.7986561075114

-- Define the total distance covered
noncomputable def total_distance : ℝ := speed_mps * time_seconds

-- Prove the length of the bridge
theorem length_of_bridge : total_distance - length_train = 170 := 
by
  -- Proof will be here
  sorry

end length_of_bridge_l85_8599


namespace smallest_number_of_ones_l85_8571

-- Definitions inferred from the problem conditions
def N := (10^100 - 1) / 3
def M_k (k : ℕ) := (10^k - 1) / 9

theorem smallest_number_of_ones (k : ℕ) : M_k k % N = 0 → k = 300 :=
by {
  sorry
}

end smallest_number_of_ones_l85_8571


namespace simplify_and_evaluate_l85_8584

theorem simplify_and_evaluate (a b : ℝ) (h_eqn : a^2 + b^2 - 2 * a + 4 * b = -5) :
  (a - 2 * b) * (a^2 + 2 * a * b + 4 * b^2) - a * (a - 5 * b) * (a + 3 * b) = 120 :=
sorry

end simplify_and_evaluate_l85_8584


namespace fraction_fliers_afternoon_l85_8541

theorem fraction_fliers_afternoon :
  ∀ (initial_fliers remaining_fliers next_day_fliers : ℕ),
    initial_fliers = 2500 →
    next_day_fliers = 1500 →
    remaining_fliers = initial_fliers - initial_fliers / 5 →
    (remaining_fliers - next_day_fliers) / remaining_fliers = 1 / 4 :=
by
  intros initial_fliers remaining_fliers next_day_fliers
  sorry

end fraction_fliers_afternoon_l85_8541


namespace total_carriages_proof_l85_8585

noncomputable def total_carriages (E N' F N : ℕ) : ℕ :=
  E + N + N' + F

theorem total_carriages_proof
  (E N N' F : ℕ)
  (h1 : E = 130)
  (h2 : E = N + 20)
  (h3 : N' = 100)
  (h4 : F = N' + 20) :
  total_carriages E N' F N = 460 := by
  sorry

end total_carriages_proof_l85_8585


namespace range_of_x_for_sqrt_l85_8517

-- Define the condition under which the expression inside the square root is non-negative.
def sqrt_condition (x : ℝ) : Prop :=
  x - 7 ≥ 0

-- Main theorem to prove the range of values for x
theorem range_of_x_for_sqrt (x : ℝ) : sqrt_condition x ↔ x ≥ 7 :=
by
  -- Proof steps go here (omitted as per instructions)
  sorry

end range_of_x_for_sqrt_l85_8517


namespace kylie_stamps_l85_8588

theorem kylie_stamps (K N : ℕ) (h1 : N = K + 44) (h2 : K + N = 112) : K = 34 :=
by
  sorry

end kylie_stamps_l85_8588


namespace prime_cannot_be_sum_of_three_squares_l85_8553

theorem prime_cannot_be_sum_of_three_squares (p : ℕ) (hp : Nat.Prime p) (hmod : p % 8 = 7) :
  ¬∃ a b c : ℤ, p = a^2 + b^2 + c^2 :=
by
  sorry

end prime_cannot_be_sum_of_three_squares_l85_8553


namespace perimeter_of_rectangular_field_l85_8564

theorem perimeter_of_rectangular_field (L B : ℝ) 
    (h1 : B = 0.60 * L) 
    (h2 : L * B = 37500) : 
    2 * L + 2 * B = 800 :=
by 
  -- proof goes here
  sorry

end perimeter_of_rectangular_field_l85_8564


namespace range_of_f_l85_8581

/-- Define the piecewise function f(x) -/
noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + 1 else Real.cos x

/-- Prove that the range of f(x) is [-1, ∞) -/
theorem range_of_f : Set.range f = Set.Ici (-1) :=
by sorry

end range_of_f_l85_8581


namespace person_a_work_days_l85_8503

theorem person_a_work_days (x : ℝ) :
  (2 * (1 / x + 1 / 45) = 1 / 9) → (x = 30) :=
by
  sorry

end person_a_work_days_l85_8503


namespace probability_other_side_red_given_seen_red_l85_8518

-- Definition of conditions
def total_cards := 9
def black_black_cards := 5
def black_red_cards := 2
def red_red_cards := 2
def red_sides := (2 * red_red_cards) + black_red_cards -- Total number of red sides
def favorable_red_red_sides := 2 * red_red_cards      -- Number of red sides on fully red cards

-- The required probability
def probability_other_side_red_given_red : ℚ := sorry

-- The main statement to prove
theorem probability_other_side_red_given_seen_red :
  probability_other_side_red_given_red = 2/3 :=
sorry

end probability_other_side_red_given_seen_red_l85_8518


namespace largest_number_of_HCF_LCM_l85_8534

theorem largest_number_of_HCF_LCM (HCF : ℕ) (k1 k2 : ℕ) (n1 n2 : ℕ) 
  (hHCF : HCF = 50)
  (hk1 : k1 = 11) 
  (hk2 : k2 = 12) 
  (hn1 : n1 = HCF * k1) 
  (hn2 : n2 = HCF * k2) :
  max n1 n2 = 600 := by
  sorry

end largest_number_of_HCF_LCM_l85_8534


namespace intersection_of_A_and_B_l85_8546

def A : Set ℝ := { x | x < 3 }
def B : Set ℝ := { x | Real.log (x - 1) / Real.log 3 > 0 }

theorem intersection_of_A_and_B :
  (A ∩ B) = { x | 2 < x ∧ x < 3 } :=
sorry

end intersection_of_A_and_B_l85_8546


namespace tom_candies_left_is_ten_l85_8547

-- Define initial conditions
def initial_candies: ℕ := 2
def friend_gave_candies: ℕ := 7
def bought_candies: ℕ := 10

-- Define total candies before sharing
def total_candies := initial_candies + friend_gave_candies + bought_candies

-- Define the number of candies Tom gives to his sister
def candies_given := total_candies / 2

-- Define the number of candies Tom has left
def candies_left := total_candies - candies_given

-- Prove the final number of candies left
theorem tom_candies_left_is_ten : candies_left = 10 :=
by
  -- The proof is left as an exercise
  sorry

end tom_candies_left_is_ten_l85_8547


namespace junghyeon_stickers_l85_8519

def total_stickers : ℕ := 25
def junghyeon_sticker_count (yejin_stickers : ℕ) : ℕ := 2 * yejin_stickers + 1

theorem junghyeon_stickers (yejin_stickers : ℕ) (h : yejin_stickers + junghyeon_sticker_count yejin_stickers = total_stickers) : 
  junghyeon_sticker_count yejin_stickers = 17 :=
  by
  sorry

end junghyeon_stickers_l85_8519


namespace ticket_price_profit_condition_maximize_profit_at_7_point_5_l85_8587

-- Define the ticket price increase and the total profit function
def ticket_price (x : ℝ) := (10 + x) * (500 - 20 * x)

-- Prove that the function equals 6000 at x = 10 and x = 25
theorem ticket_price_profit_condition (x : ℝ) :
  ticket_price x = 6000 ↔ (x = 10 ∨ x = 25) :=
by sorry

-- Prove that m = 7.5 maximizes the profit
def profit (m : ℝ) := -20 * m^2 + 300 * m + 5000

theorem maximize_profit_at_7_point_5 (m : ℝ) :
  m = 7.5 ↔ (∀ m, profit 7.5 ≥ profit m) :=
by sorry

end ticket_price_profit_condition_maximize_profit_at_7_point_5_l85_8587


namespace perpendicular_lines_slope_condition_l85_8526

theorem perpendicular_lines_slope_condition (k : ℝ) :
  (∀ x y : ℝ, y = k * x - 1 ↔ x + 2 * y + 3 = 0) → k = 2 :=
by
  sorry

end perpendicular_lines_slope_condition_l85_8526


namespace percentage_expression_l85_8573

variable {A B : ℝ} (hA : A > 0) (hB : B > 0)

theorem percentage_expression (h : A = (x / 100) * B) : x = 100 * (A / B) :=
sorry

end percentage_expression_l85_8573


namespace horizontal_distance_l85_8545

def curve (x : ℝ) := x^3 - x^2 - x - 6

def P_condition (x : ℝ) := curve x = 10
def Q_condition1 (x : ℝ) := curve x = 2
def Q_condition2 (x : ℝ) := curve x = -2

theorem horizontal_distance (x_P x_Q: ℝ) (hP: P_condition x_P) (hQ1: Q_condition1 x_Q ∨ Q_condition2 x_Q) :
  |x_P - x_Q| = 3 := sorry

end horizontal_distance_l85_8545


namespace hotel_charge_comparison_l85_8529

theorem hotel_charge_comparison (R G P : ℝ) 
  (h1 : P = R - 0.70 * R)
  (h2 : P = G - 0.10 * G) :
  ((R - G) / G) * 100 = 170 :=
by
  sorry

end hotel_charge_comparison_l85_8529


namespace simplify_fraction_l85_8565

theorem simplify_fraction (a : ℝ) (h : a ≠ 2) : (3 - a) / (a - 2) + 1 = 1 / (a - 2) :=
by
  -- proof goes here
  sorry

end simplify_fraction_l85_8565


namespace lindy_distance_l85_8593

theorem lindy_distance
  (d : ℝ) (v_j : ℝ) (v_c : ℝ) (v_l : ℝ) (t : ℝ)
  (h1 : d = 270)
  (h2 : v_j = 4)
  (h3 : v_c = 5)
  (h4 : v_l = 8)
  (h_time : t = d / (v_j + v_c)) :
  v_l * t = 240 := by
  sorry

end lindy_distance_l85_8593


namespace percent_commute_l85_8558

variable (x : ℝ)

theorem percent_commute (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 :=
by
  sorry

end percent_commute_l85_8558


namespace machine_working_time_l85_8591

def shirts_per_minute : ℕ := 3
def total_shirts_made : ℕ := 6

theorem machine_working_time : 
  (total_shirts_made / shirts_per_minute) = 2 :=
by
  sorry

end machine_working_time_l85_8591


namespace count_even_numbers_l85_8524

theorem count_even_numbers : 
  ∃ n : ℕ, n = 199 ∧ ∀ m : ℕ, (302 ≤ m ∧ m < 700 ∧ m % 2 = 0) → 
    151 ≤ ((m - 300) / 2) ∧ ((m - 300) / 2) ≤ 349 :=
sorry

end count_even_numbers_l85_8524


namespace intersect_x_axis_unique_l85_8580

theorem intersect_x_axis_unique (a : ℝ) : (∀ x, (ax^2 + (3 - a) * x + 1) = 0 → x = 0) ↔ (a = 0 ∨ a = 1 ∨ a = 9) := by
  sorry

end intersect_x_axis_unique_l85_8580


namespace largest_value_in_interval_l85_8539

theorem largest_value_in_interval (x : ℝ) (h : 0 < x ∧ x < 1) : 
  (∀ y ∈ ({x, x^3, 3*x, x^(1/3), 1/x} : Set ℝ), y ≤ 1/x) :=
sorry

end largest_value_in_interval_l85_8539


namespace ratio_squirrels_to_raccoons_l85_8554

def animals_total : ℕ := 84
def raccoons : ℕ := 12
def squirrels : ℕ := animals_total - raccoons

theorem ratio_squirrels_to_raccoons : (squirrels : ℚ) / raccoons = 6 :=
by
  sorry

end ratio_squirrels_to_raccoons_l85_8554


namespace domain_of_function_l85_8583

theorem domain_of_function :
  ∀ x : ℝ, (1 / (1 - x) ≥ 0 ∧ 1 - x ≠ 0) ↔ (x < 1) :=
by
  sorry

end domain_of_function_l85_8583


namespace sum_m_n_l85_8500

-- Define the conditions and the result

def probabilityOfNo3x3RedSquare : ℚ :=
  65408 / 65536

def gcd_65408_65536 := Nat.gcd 65408 65536

def simplifiedProbability : ℚ :=
  probabilityOfNo3x3RedSquare / gcd_65408_65536

def m : ℕ :=
  511

def n : ℕ :=
  512

theorem sum_m_n : m + n = 1023 := by
  sorry

end sum_m_n_l85_8500


namespace white_trees_count_l85_8531

noncomputable def calculate_white_trees (total_trees pink_percent red_trees : ℕ) : ℕ :=
  total_trees - (total_trees * pink_percent / 100 + red_trees)

theorem white_trees_count 
  (h1 : total_trees = 42)
  (h2 : pink_percent = 100 / 3)
  (h3 : red_trees = 2) :
  calculate_white_trees total_trees pink_percent red_trees = 26 :=
by
  -- proof will go here
  sorry

end white_trees_count_l85_8531


namespace car_speed_first_hour_l85_8508

-- Definitions based on the conditions in the problem
noncomputable def speed_second_hour := 30
noncomputable def average_speed := 45
noncomputable def total_time := 2

-- Assertion based on the problem's question and correct answer
theorem car_speed_first_hour: ∃ (x : ℕ), (average_speed * total_time) = (x + speed_second_hour) ∧ x = 60 :=
by
  sorry

end car_speed_first_hour_l85_8508


namespace gcd_product_eq_gcd_l85_8569

theorem gcd_product_eq_gcd {a b c : ℤ} (hab : Int.gcd a b = 1) : Int.gcd a (b * c) = Int.gcd a c := 
by 
  sorry

end gcd_product_eq_gcd_l85_8569


namespace measure_of_angle_C_sin_A_plus_sin_B_l85_8527

-- Problem 1
theorem measure_of_angle_C (a b c : ℝ) (h1 : a^2 + b^2 - c^2 = 8) (h2 : (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) : C = Real.pi / 3 := 
sorry

-- Problem 2
theorem sin_A_plus_sin_B (a b c A B C : ℝ) (h1 : a^2 + b^2 - c^2 = 8) (h2 : (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) (h3 : c = 2 * Real.sqrt 3) : Real.sin A + Real.sin B = 3 / 2 := 
sorry

end measure_of_angle_C_sin_A_plus_sin_B_l85_8527


namespace relationship_a_b_l85_8596

noncomputable def e : ℝ := Real.exp 1

theorem relationship_a_b
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : e^a + 2 * a = e^b + 3 * b) :
  a > b :=
sorry

end relationship_a_b_l85_8596


namespace ken_height_l85_8586

theorem ken_height 
  (height_ivan : ℝ) (height_jackie : ℝ) (height_ken : ℝ)
  (h1 : height_ivan = 175) (h2 : height_jackie = 175)
  (h_avg : (height_ivan + height_jackie + height_ken) / 3 = (height_ivan + height_jackie) / 2 * 1.04) :
  height_ken = 196 := 
sorry

end ken_height_l85_8586


namespace is_possible_to_finish_7th_l85_8577

theorem is_possible_to_finish_7th 
  (num_teams : ℕ)
  (wins_ASTC : ℕ)
  (losses_ASTC : ℕ)
  (points_per_win : ℕ)
  (points_per_draw : ℕ) 
  (total_points : ℕ)
  (rank_ASTC : ℕ)
  (points_ASTC : ℕ)
  (points_needed_by_top_6 : ℕ → ℕ)
  (points_8th_and_9th : ℕ) :
  num_teams = 9 ∧ wins_ASTC = 5 ∧ losses_ASTC = 3 ∧ points_per_win = 3 ∧ points_per_draw = 1 ∧ 
  total_points = 108 ∧ rank_ASTC = 7 ∧ points_ASTC = 15 ∧ points_needed_by_top_6 7 = 105 ∧ points_8th_and_9th ≤ 3 →
  ∃ (top_7_points : ℕ), 
  top_7_points = 105 ∧ (top_7_points + points_8th_and_9th) = total_points := 
sorry

end is_possible_to_finish_7th_l85_8577


namespace log_product_l85_8595

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_product (x y : ℝ) (hx : 0 < x) (hy : 1 < y) :
  log_base (y^3) x * log_base (x^4) (y^3) * log_base (y^5) (x^2) * log_base (x^2) (y^5) * log_base (y^3) (x^4) =
  (1/3) * log_base y x :=
by
  sorry

end log_product_l85_8595


namespace remy_used_25_gallons_l85_8521

noncomputable def RomanGallons : ℕ := 8

noncomputable def RemyGallons (R : ℕ) : ℕ := 3 * R + 1

theorem remy_used_25_gallons (R : ℕ) (h1 : RemyGallons R = 1 + 3 * R) (h2 : R + RemyGallons R = 33) : RemyGallons R = 25 := by
  sorry

end remy_used_25_gallons_l85_8521


namespace roots_of_polynomial_l85_8540

-- Define the polynomial
def P (x : ℝ) : ℝ := x^3 - 7 * x^2 + 14 * x - 8

-- Prove that the roots of P are {1, 2, 4}
theorem roots_of_polynomial :
  ∃ (S : Set ℝ), S = {1, 2, 4} ∧ ∀ x, P x = 0 ↔ x ∈ S :=
by
  sorry

end roots_of_polynomial_l85_8540


namespace sin_15_deg_eq_l85_8511

theorem sin_15_deg_eq : 
  Real.sin (15 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := 
by
  -- conditions
  have h1 : Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  have h2 : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  have h3 : Real.sin (30 * Real.pi / 180) = 1 / 2 := by sorry
  have h4 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := by sorry
  
  -- proof
  sorry

end sin_15_deg_eq_l85_8511


namespace product_of_last_two_digits_l85_8537

theorem product_of_last_two_digits (A B : ℕ) (hn1 : 10 * A + B ≡ 0 [MOD 5]) (hn2 : A + B = 16) : A * B = 30 :=
sorry

end product_of_last_two_digits_l85_8537


namespace field_length_to_width_ratio_l85_8523
-- Import the math library

-- Define the problem conditions and proof goal statement
theorem field_length_to_width_ratio (w : ℝ) (l : ℝ) (area_pond : ℝ) (area_field : ℝ) 
    (h_length : l = 16) (h_area_pond : area_pond = 64) 
    (h_area_relation : area_pond = (1/2) * area_field)
    (h_field_area : area_field = l * w) : l / w = 2 :=
by 
  -- Leaving the proof as an exercise
  sorry

end field_length_to_width_ratio_l85_8523


namespace snow_first_day_eq_six_l85_8522

variable (snow_first_day snow_second_day snow_fourth_day snow_fifth_day : ℤ)

theorem snow_first_day_eq_six
  (h1 : snow_second_day = snow_first_day + 8)
  (h2 : snow_fourth_day = snow_second_day - 2)
  (h3 : snow_fifth_day = snow_fourth_day + 2 * snow_first_day)
  (h4 : snow_fifth_day = 24) :
  snow_first_day = 6 := by
  sorry

end snow_first_day_eq_six_l85_8522


namespace max_squares_on_checkerboard_l85_8502

theorem max_squares_on_checkerboard (n : ℕ) (h1 : n = 7) (h2 : ∀ s : ℕ, s = 2) : ∃ max_squares : ℕ, max_squares = 18 := sorry

end max_squares_on_checkerboard_l85_8502


namespace rearrange_distinct_sums_mod_4028_l85_8574

theorem rearrange_distinct_sums_mod_4028 
  (x : Fin 2014 → ℤ) (y : Fin 2014 → ℤ) 
  (hx : ∀ i j : Fin 2014, i ≠ j → x i % 2014 ≠ x j % 2014)
  (hy : ∀ i j : Fin 2014, i ≠ j → y i % 2014 ≠ y j % 2014) :
  ∃ σ : Fin 2014 → Fin 2014, Function.Bijective σ ∧ 
  ∀ i j : Fin 2014, i ≠ j → ( x i + y (σ i) ) % 4028 ≠ ( x j + y (σ j) ) % 4028 
:= by
  sorry

end rearrange_distinct_sums_mod_4028_l85_8574


namespace evaluate_expression_l85_8535

noncomputable def greatest_integer (x : Real) : Int := ⌊x⌋

theorem evaluate_expression (y : Real) (h : y = 7.2) :
  greatest_integer 6.5 * greatest_integer (2 / 3)
  + greatest_integer 2 * y
  + greatest_integer 8.4 - 6.0 = 16.4 := by
  simp [greatest_integer, h]
  sorry

end evaluate_expression_l85_8535


namespace cube_sphere_volume_relation_l85_8549

theorem cube_sphere_volume_relation (n : ℕ) (h : 2 < n)
  (h_volume : n^3 - (n^3 * pi / 6) = (n^3 * pi / 3)) : n = 8 :=
sorry

end cube_sphere_volume_relation_l85_8549


namespace problem_correctness_l85_8513

variable (f : ℝ → ℝ)
variable (h₀ : ∀ x, f x > 0)
variable (h₁ : ∀ a b, f a * f b = f (a + b))

theorem problem_correctness :
  (f 0 = 1) ∧
  (∀ a, f (-a) = 1 / f a) ∧
  (∀ a, f a = (f (3 * a)) ^ (1 / 3)) :=
by 
  -- Using the hypotheses provided
  sorry

end problem_correctness_l85_8513


namespace functional_eq_solution_l85_8548

open Real

theorem functional_eq_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = f (x - y) + 4 * x * y) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x^2 + c := 
sorry

end functional_eq_solution_l85_8548


namespace simplify_fraction_l85_8594

variable (x y : ℝ)

theorem simplify_fraction :
  (2 * x + y) / 4 + (5 * y - 4 * x) / 6 - y / 12 = (-x + 6 * y) / 6 :=
by
  sorry

end simplify_fraction_l85_8594


namespace colorings_without_two_corners_l85_8536

def valid_colorings (n: ℕ) (exclude_cells : Finset (Fin n × Fin n)) : ℕ := sorry

theorem colorings_without_two_corners :
  valid_colorings 5 ∅ = 120 →
  valid_colorings 5 {(0, 0)} = 96 →
  valid_colorings 5 {(0, 0), (4, 4)} = 78 :=
by {
  sorry
}

end colorings_without_two_corners_l85_8536


namespace count_perfect_cube_or_fourth_power_lt_1000_l85_8563

theorem count_perfect_cube_or_fourth_power_lt_1000 :
  ∃ n, n = 14 ∧ (∀ x, (0 < x ∧ x < 1000 ∧ (∃ k, x = k^3 ∨ x = k^4)) ↔ ∃ i, i < n) :=
by sorry

end count_perfect_cube_or_fourth_power_lt_1000_l85_8563


namespace calvin_overall_score_l85_8557

theorem calvin_overall_score :
  let test1_pct := 0.6
  let test1_total := 15
  let test2_pct := 0.85
  let test2_total := 20
  let test3_pct := 0.75
  let test3_total := 40
  let total_problems := 75

  let correct_test1 := test1_pct * test1_total
  let correct_test2 := test2_pct * test2_total
  let correct_test3 := test3_pct * test3_total
  let total_correct := correct_test1 + correct_test2 + correct_test3

  let overall_percentage := (total_correct / total_problems) * 100
  overall_percentage.round = 75 :=
sorry

end calvin_overall_score_l85_8557


namespace opposite_of_one_fourth_l85_8590

/-- The opposite of the fraction 1/4 is -1/4 --/
theorem opposite_of_one_fourth : - (1 / 4) = -1 / 4 :=
by
  sorry

end opposite_of_one_fourth_l85_8590


namespace find_values_l85_8510

theorem find_values (h t u : ℕ) 
  (h0 : u = h - 5) 
  (h1 : (h * 100 + t * 10 + u) - (h * 100 + u * 10 + t) = 96)
  (hu : h < 10 ∧ t < 10 ∧ u < 10) :
  h = 5 ∧ t = 9 ∧ u = 0 :=
by 
  sorry

end find_values_l85_8510


namespace word_count_in_language_l85_8562

theorem word_count_in_language :
  let vowels := 3
  let consonants := 5
  let num_syllables := (vowels * consonants) + (consonants * vowels)
  let num_words := num_syllables * num_syllables
  num_words = 900 :=
by
  let vowels := 3
  let consonants := 5
  let num_syllables := (vowels * consonants) + (consonants * vowels)
  let num_words := num_syllables * num_syllables
  have : num_words = 900 := sorry
  exact this

end word_count_in_language_l85_8562


namespace find_B_coords_l85_8516

-- Define point A and vector a
def A : (ℝ × ℝ) := (1, -3)
def a : (ℝ × ℝ) := (3, 4)

-- Assume B is at coordinates (m, n) and AB = 2a
def B : (ℝ × ℝ) := (7, 5)
def AB : (ℝ × ℝ) := (B.1 - A.1, B.2 - A.2)

-- Prove point B has the correct coordinates
theorem find_B_coords : AB = (2 * a.1, 2 * a.2) → B = (7, 5) :=
by
  intro h
  sorry

end find_B_coords_l85_8516


namespace set_difference_M_N_l85_8582

def setM : Set ℝ := { x | -1 < x ∧ x < 1 }
def setN : Set ℝ := { x | x / (x - 1) ≤ 0 }

theorem set_difference_M_N :
  setM \ setN = { x | -1 < x ∧ x < 0 } := sorry

end set_difference_M_N_l85_8582


namespace prime_p_in_range_l85_8501

theorem prime_p_in_range (p : ℕ) (prime_p : Nat.Prime p) 
    (h : ∃ a b : ℤ, a * b = -530 * p ∧ a + b = p) : 43 < p ∧ p ≤ 53 := 
sorry

end prime_p_in_range_l85_8501


namespace lcm_of_36_and_100_l85_8578

theorem lcm_of_36_and_100 : Nat.lcm 36 100 = 900 :=
by
  -- The proof is omitted
  sorry

end lcm_of_36_and_100_l85_8578


namespace milk_left_in_storage_l85_8589

-- Define initial and rate conditions
def initialMilk : ℕ := 30000
def pumpedRate : ℕ := 2880
def pumpedHours : ℕ := 4
def addedRate : ℕ := 1500
def addedHours : ℕ := 7

-- The proof problem: Prove the final amount in storage tank == 28980 gallons
theorem milk_left_in_storage : 
  initialMilk - (pumpedRate * pumpedHours) + (addedRate * addedHours) = 28980 := 
sorry

end milk_left_in_storage_l85_8589


namespace simplify_polynomials_l85_8533

-- Define the polynomials
def poly1 (q : ℝ) : ℝ := 5 * q^4 + 3 * q^3 - 7 * q + 8
def poly2 (q : ℝ) : ℝ := 6 - 9 * q^3 + 4 * q - 3 * q^4

-- The goal is to prove that the sum of poly1 and poly2 simplifies correctly
theorem simplify_polynomials (q : ℝ) : 
  poly1 q + poly2 q = 2 * q^4 - 6 * q^3 - 3 * q + 14 := 
by 
  sorry

end simplify_polynomials_l85_8533


namespace hilt_has_2_pennies_l85_8512

-- Define the total value of coins each person has without considering Mrs. Hilt's pennies
def dimes : ℕ := 2
def nickels : ℕ := 2
def hilt_base_amount : ℕ := dimes * 10 + nickels * 5 -- 30 cents

def jacob_pennies : ℕ := 4
def jacob_nickels : ℕ := 1
def jacob_dimes : ℕ := 1
def jacob_amount : ℕ := jacob_pennies * 1 + jacob_nickels * 5 + jacob_dimes * 10 -- 19 cents

def difference : ℕ := 13
def hilt_pennies : ℕ := 2 -- The solution's correct answer

theorem hilt_has_2_pennies : hilt_base_amount - jacob_amount + hilt_pennies = difference := by sorry

end hilt_has_2_pennies_l85_8512


namespace volume_ratio_of_frustum_l85_8514

theorem volume_ratio_of_frustum
  (h_s h : ℝ)
  (A_s A : ℝ)
  (V_s V : ℝ)
  (ratio_lateral_area : ℝ)
  (ratio_height : ℝ)
  (ratio_base_area : ℝ)
  (H_lateral_area: ratio_lateral_area = 9 / 16)
  (H_height: ratio_height = 3 / 5)
  (H_base_area: ratio_base_area = 9 / 25)
  (H_volume_small: V_s = 1 / 3 * h_s * A_s)
  (H_volume_total: V = 1 / 3 * h * A - 1 / 3 * h_s * A_s) :
  V_s / V = 27 / 98 :=
by
  sorry

end volume_ratio_of_frustum_l85_8514


namespace units_digit_27_3_sub_17_3_l85_8597

theorem units_digit_27_3_sub_17_3 : 
  (27 ^ 3 - 17 ^ 3) % 10 = 0 :=
sorry

end units_digit_27_3_sub_17_3_l85_8597


namespace correct_scientific_notation_representation_l85_8576

-- Defining the given number of visitors in millions
def visitors_in_millions : Float := 8.0327
-- Converting this number to an integer and expressing in scientific notation
def rounded_scientific_notation (num : Float) : String :=
  if num == 8.0327 then "8.0 × 10^6" else "incorrect"

-- The mathematical proof statement
theorem correct_scientific_notation_representation :
  rounded_scientific_notation visitors_in_millions = "8.0 × 10^6" :=
by
  sorry

end correct_scientific_notation_representation_l85_8576


namespace distance_between_closest_points_correct_l85_8515

noncomputable def circle_1_center : ℝ × ℝ := (3, 3)
noncomputable def circle_2_center : ℝ × ℝ := (20, 12)
noncomputable def circle_1_radius : ℝ := circle_1_center.2
noncomputable def circle_2_radius : ℝ := circle_2_center.2
noncomputable def distance_between_centers : ℝ := Real.sqrt ((20 - 3)^2 + (12 - 3)^2)
noncomputable def distance_between_closest_points : ℝ := distance_between_centers - (circle_1_radius + circle_2_radius)

theorem distance_between_closest_points_correct :
  distance_between_closest_points = Real.sqrt 370 - 15 :=
sorry

end distance_between_closest_points_correct_l85_8515


namespace travel_speed_l85_8598

theorem travel_speed (distance : ℝ) (time : ℝ) (h_distance : distance = 195) (h_time : time = 3) : 
  distance / time = 65 :=
by 
  rw [h_distance, h_time]
  norm_num

end travel_speed_l85_8598


namespace cannot_be_value_of_omega_l85_8559

theorem cannot_be_value_of_omega (ω : ℤ) (φ : ℝ) (k n : ℤ) 
  (h1 : 0 < ω) 
  (h2 : |φ| < π / 2)
  (h3 : ω * (π / 12) + φ = k * π + π / 2)
  (h4 : -ω * (π / 6) + φ = n * π) : 
  ∀ m : ℤ, ω ≠ 4 * m := 
sorry

end cannot_be_value_of_omega_l85_8559


namespace joan_dimes_l85_8551

theorem joan_dimes (initial_dimes spent_dimes remaining_dimes : ℕ) 
    (h1 : initial_dimes = 5) (h2 : spent_dimes = 2) 
    (h3 : remaining_dimes = initial_dimes - spent_dimes) : 
    remaining_dimes = 3 := 
sorry

end joan_dimes_l85_8551


namespace fraction_eval_l85_8520

theorem fraction_eval :
  (3 / 7 + 5 / 8) / (5 / 12 + 1 / 4) = 177 / 112 :=
by
  sorry

end fraction_eval_l85_8520


namespace regular_polygon_sides_l85_8567

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l85_8567


namespace miracle_tree_fruit_count_l85_8530

theorem miracle_tree_fruit_count :
  ∃ (apples oranges pears : ℕ), 
  apples + oranges + pears = 30 ∧
  apples = 6 ∧ oranges = 9 ∧ pears = 15 := by
  sorry

end miracle_tree_fruit_count_l85_8530


namespace budget_for_supplies_l85_8561

-- Conditions as definitions
def percentage_transportation := 20
def percentage_research_development := 9
def percentage_utilities := 5
def percentage_equipment := 4
def degrees_salaries := 216
def total_degrees := 360
def total_percentage := 100

-- Mathematical problem: Prove the percentage spent on supplies
theorem budget_for_supplies :
  (total_percentage - (percentage_transportation +
                       percentage_research_development +
                       percentage_utilities +
                       percentage_equipment) - 
   ((degrees_salaries * total_percentage) / total_degrees)) = 2 := by
  sorry

end budget_for_supplies_l85_8561


namespace largest_five_digit_number_divisible_by_5_l85_8560

theorem largest_five_digit_number_divisible_by_5 : 
  ∃ n, (n % 5 = 0) ∧ (99990 ≤ n) ∧ (n ≤ 99995) ∧ (∀ m, (m % 5 = 0) → (99990 ≤ m) → (m ≤ 99995) → m ≤ n) :=
by
  -- The proof is omitted as per the instructions
  sorry

end largest_five_digit_number_divisible_by_5_l85_8560


namespace cube_inequality_of_greater_l85_8542

theorem cube_inequality_of_greater {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cube_inequality_of_greater_l85_8542


namespace circle_equation_with_diameter_endpoints_l85_8575

theorem circle_equation_with_diameter_endpoints (A B : ℝ × ℝ) (x y : ℝ) :
  A = (1, 4) → B = (3, -2) → (x-2)^2 + (y-1)^2 = 10 :=
by
  sorry

end circle_equation_with_diameter_endpoints_l85_8575


namespace sum_symmetry_l85_8555

def f (x : ℝ) : ℝ :=
  x^2 * (1 - x)^2

theorem sum_symmetry :
  f (1/7) - f (2/7) + f (3/7) - f (4/7) + f (5/7) - f (6/7) = 0 :=
by
  sorry

end sum_symmetry_l85_8555


namespace passengers_initial_count_l85_8532

-- Let's define the initial number of passengers
variable (P : ℕ)

-- Given conditions:
def final_passengers (initial additional left : ℕ) : ℕ := initial + additional - left

-- The theorem statement to prove P = 28 given the conditions
theorem passengers_initial_count
  (final_count : ℕ)
  (h1 : final_count = 26)
  (h2 : final_passengers P 7 9 = final_count) 
  : P = 28 :=
by
  sorry

end passengers_initial_count_l85_8532


namespace total_right_handed_players_l85_8572

theorem total_right_handed_players
  (total_players throwers mp_players non_throwers L R : ℕ)
  (ratio_L_R : 2 * R = 3 * L)
  (total_eq : total_players = 120)
  (throwers_eq : throwers = 60)
  (mp_eq : mp_players = 20)
  (non_throwers_eq : non_throwers = total_players - throwers - mp_players)
  (non_thrower_sum_eq : L + R = non_throwers) :
  (throwers + mp_players + R = 104) :=
by
  sorry

end total_right_handed_players_l85_8572


namespace existential_proposition_l85_8506

theorem existential_proposition :
  (∃ x y : ℝ, x + y > 1) ∧ (∀ P : Prop, (∃ x y : ℝ, x + y > 1 → P) → P) :=
sorry

end existential_proposition_l85_8506


namespace area_of_backyard_eq_400_l85_8579

-- Define the conditions
def length_condition (l : ℕ) : Prop := 25 * l = 1000
def perimeter_condition (l w : ℕ) : Prop := 20 * (l + w) = 1000

-- State the theorem
theorem area_of_backyard_eq_400 (l w : ℕ) (h_length : length_condition l) (h_perimeter : perimeter_condition l w) : l * w = 400 :=
  sorry

end area_of_backyard_eq_400_l85_8579


namespace mark_total_eggs_in_a_week_l85_8538

-- Define the given conditions
def first_store_eggs_per_day := 5 * 12 -- 5 dozen eggs per day
def second_store_eggs_per_day := 30
def third_store_eggs_per_odd_day := 25 * 12 -- 25 dozen eggs per odd day
def third_store_eggs_per_even_day := 15 * 12 -- 15 dozen eggs per even day
def days_per_week := 7
def odd_days_per_week := 4
def even_days_per_week := 3

-- Lean theorem statement to prove the total eggs supplied in a week
theorem mark_total_eggs_in_a_week : 
    first_store_eggs_per_day * days_per_week + 
    second_store_eggs_per_day * days_per_week + 
    third_store_eggs_per_odd_day * odd_days_per_week + 
    third_store_eggs_per_even_day * even_days_per_week =
    2370 := 
    sorry  -- Placeholder for the actual proof

end mark_total_eggs_in_a_week_l85_8538


namespace sled_dog_race_l85_8550

theorem sled_dog_race (d t : ℕ) (h1 : d + t = 315) (h2 : (1.2 : ℚ) * d + t = (1 / 2 : ℚ) * (2 * d + 3 * t)) :
  d = 225 ∧ t = 90 :=
sorry

end sled_dog_race_l85_8550


namespace cos_210_eq_neg_sqrt3_over_2_l85_8566

theorem cos_210_eq_neg_sqrt3_over_2 :
  Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by sorry

end cos_210_eq_neg_sqrt3_over_2_l85_8566


namespace sum_of_possible_values_l85_8505

theorem sum_of_possible_values (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 2) :
  (x - 2) * (y - 2) = 6 ∨ (x - 2) * (y - 2) = 9 →
  (if (x - 2) * (y - 2) = 6 then 6 else 0) + (if (x - 2) * (y - 2) = 9 then 9 else 0) = 15 :=
by
  sorry

end sum_of_possible_values_l85_8505


namespace total_tickets_sold_l85_8525

-- Define the parameters and conditions
def VIP_ticket_price : ℝ := 45.00
def general_ticket_price : ℝ := 20.00
def total_revenue : ℝ := 7500.00
def tickets_difference : ℕ := 276

-- Define the total number of tickets sold
def total_number_of_tickets (V G : ℕ) : ℕ := V + G

-- The theorem to be proved
theorem total_tickets_sold (V G : ℕ) 
  (h1 : VIP_ticket_price * V + general_ticket_price * G = total_revenue)
  (h2 : V = G - tickets_difference) : 
  total_number_of_tickets V G = 336 :=
by
  sorry

end total_tickets_sold_l85_8525


namespace Elaine_rent_increase_l85_8509

noncomputable def Elaine_rent_percent (E: ℝ) : ℝ :=
  let last_year_rent := 0.20 * E
  let this_year_earnings := 1.25 * E
  let this_year_rent := 0.30 * this_year_earnings
  let ratio := (this_year_rent / last_year_rent) * 100
  ratio

theorem Elaine_rent_increase (E : ℝ) : Elaine_rent_percent E = 187.5 :=
by 
  -- The proof would go here.
  sorry

end Elaine_rent_increase_l85_8509


namespace part1_part2_l85_8568

open Set Real

def M (x : ℝ) : Prop := x^2 - 3 * x - 18 ≤ 0
def N (x : ℝ) (a : ℝ) : Prop := 1 - a ≤ x ∧ x ≤ 2 * a + 1

theorem part1 (a : ℝ) (h : a = 3) : (Icc (-2 : ℝ) 6 = {x | M x ∧ N x a}) ∧ (compl {x | N x a} = Iic (-2) ∪ Ioi 7) :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x, M x ∧ N x a ↔ N x a) → a ≤ 5 / 2 :=
by
  sorry

end part1_part2_l85_8568


namespace lottery_probability_l85_8504

theorem lottery_probability (p: ℝ) :
  (∀ n, 1 ≤ n ∧ n ≤ 15 → p = 2/3) →
  (true → p = 0.6666666666666666) →
  p = 2/3 :=
by
  intros h h'
  sorry

end lottery_probability_l85_8504


namespace Rams_monthly_salary_l85_8507

variable (R S A : ℝ)
variable (annual_salary : ℝ)
variable (monthly_salary_conversion : annual_salary / 12 = A)
variable (ram_shyam_condition : 0.10 * R = 0.08 * S)
variable (shyam_abhinav_condition : S = 2 * A)
variable (abhinav_annual_salary : annual_salary = 192000)

theorem Rams_monthly_salary 
  (annual_salary : ℝ)
  (ram_shyam_condition : 0.10 * R = 0.08 * S)
  (shyam_abhinav_condition : S = 2 * A)
  (abhinav_annual_salary : annual_salary = 192000)
  (monthly_salary_conversion: annual_salary / 12 = A): 
  R = 25600 := by
  sorry

end Rams_monthly_salary_l85_8507


namespace round_robin_total_points_l85_8544

theorem round_robin_total_points :
  let points_per_match := 2
  let total_matches := 3
  (total_matches * points_per_match) = 6 :=
by
  sorry

end round_robin_total_points_l85_8544


namespace range_of_a_l85_8528

theorem range_of_a (a x y : ℝ) (h1: x + 3 * y = 2 + a) (h2: 3 * x + y = -4 * a) (h3: x + y > 2) : a < -2 :=
sorry

end range_of_a_l85_8528


namespace shortest_remaining_side_length_l85_8556

noncomputable def triangle_has_right_angle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem shortest_remaining_side_length {a b : ℝ} (ha : a = 5) (hb : b = 12) (h_right_angle : ∃ c, triangle_has_right_angle a b c) :
  ∃ c, c = 5 :=
by 
  sorry

end shortest_remaining_side_length_l85_8556


namespace negation_exists_gt_one_l85_8552

theorem negation_exists_gt_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) :=
sorry

end negation_exists_gt_one_l85_8552


namespace complement_intersection_l85_8543

open Set

variable (U M N : Set ℕ)
variable (H₁ : U = {1, 2, 3, 4, 5, 6})
variable (H₂ : M = {1, 2, 3, 5})
variable (H₃ : N = {1, 3, 4, 6})

theorem complement_intersection :
  (U \ (M ∩ N)) = {2, 4, 5, 6} :=
by
  sorry

end complement_intersection_l85_8543


namespace walmart_knives_eq_three_l85_8570

variable (k : ℕ)

-- Walmart multitool
def walmart_tools : ℕ := 1 + k + 2

-- Target multitool (with twice as many knives as Walmart)
def target_tools : ℕ := 1 + 2 * k + 3 + 1

-- The condition that Target multitool has 5 more tools compared to Walmart
theorem walmart_knives_eq_three (h : target_tools k = walmart_tools k + 5) : k = 3 :=
by
  sorry

end walmart_knives_eq_three_l85_8570
