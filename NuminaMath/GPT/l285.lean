import Mathlib

namespace NUMINAMATH_GPT_tree_height_when_planted_l285_28545

def initial_height (current_height : ℕ) (growth_rate : ℕ) (current_age : ℕ) (initial_age : ℕ) : ℕ :=
  current_height - (current_age - initial_age) * growth_rate

theorem tree_height_when_planted :
  initial_height 23 3 7 1 = 5 :=
by
  sorry

end NUMINAMATH_GPT_tree_height_when_planted_l285_28545


namespace NUMINAMATH_GPT_smaller_rectangle_dimensions_l285_28594

theorem smaller_rectangle_dimensions (side_length : ℝ) (L W : ℝ) 
  (h1 : side_length = 10) 
  (h2 : L + 2 * L = side_length) 
  (h3 : W = L) : 
  L = 10 / 3 ∧ W = 10 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_smaller_rectangle_dimensions_l285_28594


namespace NUMINAMATH_GPT_integer_solutions_exist_l285_28577

theorem integer_solutions_exist (m n : ℤ) :
  ∃ (w x y z : ℤ), 
  (w + x + 2 * y + 2 * z = m) ∧ 
  (2 * w - 2 * x + y - z = n) := sorry

end NUMINAMATH_GPT_integer_solutions_exist_l285_28577


namespace NUMINAMATH_GPT_max_volume_cuboid_l285_28525

theorem max_volume_cuboid (x y z : ℕ) (h : 2 * (x * y + x * z + y * z) = 150) : x * y * z ≤ 125 :=
sorry

end NUMINAMATH_GPT_max_volume_cuboid_l285_28525


namespace NUMINAMATH_GPT_correct_microorganism_dilution_statement_l285_28582

def microorganism_dilution_conditions (A B C D : Prop) : Prop :=
  (A ↔ ∀ (dilutions : ℕ) (n : ℕ), 1000 ≤ dilutions ∧ dilutions ≤ 10000000) ∧
  (B ↔ ∀ (dilutions : ℕ) (actinomycetes : ℕ), dilutions = 1000 ∨ dilutions = 10000 ∨ dilutions = 100000) ∧
  (C ↔ ∀ (dilutions : ℕ) (fungi : ℕ), dilutions = 1000 ∨ dilutions = 10000 ∨ dilutions = 100000) ∧
  (D ↔ ∀ (dilutions : ℕ) (bacteria_first_time : ℕ), 10 ≤ dilutions ∧ dilutions ≤ 10000000)

theorem correct_microorganism_dilution_statement (A B C D : Prop)
  (h : microorganism_dilution_conditions A B C D) : D :=
sorry

end NUMINAMATH_GPT_correct_microorganism_dilution_statement_l285_28582


namespace NUMINAMATH_GPT_proposition_p_q_true_l285_28590

def represents_hyperbola (m : ℝ) : Prop := (1 - m) * (m + 2) < 0

def represents_ellipse (m : ℝ) : Prop := (2 * m > 2 - m) ∧ (2 - m > 0)

theorem proposition_p_q_true (m : ℝ) :
  represents_hyperbola m ∧ represents_ellipse m → (1 < m ∧ m < 2) :=
by
  sorry

end NUMINAMATH_GPT_proposition_p_q_true_l285_28590


namespace NUMINAMATH_GPT_range_of_m_l285_28530

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x > 0 → 9^x - m * 3^x + m + 1 > 0) → m < 2 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l285_28530


namespace NUMINAMATH_GPT_rows_with_exactly_10_people_l285_28574

theorem rows_with_exactly_10_people (x : ℕ) (total_people : ℕ) (row_nine_seat : ℕ) (row_ten_seat : ℕ) 
    (H1 : row_nine_seat = 9) (H2 : row_ten_seat = 10) 
    (H3 : total_people = 55) 
    (H4 : total_people = x * row_ten_seat + (6 - x) * row_nine_seat) 
    : x = 1 :=
by
  sorry

end NUMINAMATH_GPT_rows_with_exactly_10_people_l285_28574


namespace NUMINAMATH_GPT_percent_non_bikers_play_basketball_l285_28536

noncomputable def total_children (N : ℕ) : ℕ := N
def basketball_players (N : ℕ) : ℕ := 7 * N / 10
def bikers (N : ℕ) : ℕ := 4 * N / 10
def basketball_bikers (N : ℕ) : ℕ := 3 * basketball_players N / 10
def basketball_non_bikers (N : ℕ) : ℕ := basketball_players N - basketball_bikers N
def non_bikers (N : ℕ) : ℕ := N - bikers N

theorem percent_non_bikers_play_basketball (N : ℕ) :
  (basketball_non_bikers N * 100 / non_bikers N) = 82 :=
by sorry

end NUMINAMATH_GPT_percent_non_bikers_play_basketball_l285_28536


namespace NUMINAMATH_GPT_scientific_notation_of_463_4_billion_l285_28506

theorem scientific_notation_of_463_4_billion :
  (463.4 * 10^9) = (4.634 * 10^11) := by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_463_4_billion_l285_28506


namespace NUMINAMATH_GPT_printer_to_enhanced_ratio_l285_28570

def B : ℕ := 2125
def P : ℕ := 2500 - B
def E : ℕ := B + 500
def total_price := E + P

theorem printer_to_enhanced_ratio :
  (P : ℚ) / total_price = 1 / 8 := 
by {
  -- skipping the proof
  sorry
}

end NUMINAMATH_GPT_printer_to_enhanced_ratio_l285_28570


namespace NUMINAMATH_GPT_addition_correctness_l285_28538

theorem addition_correctness : 1.25 + 47.863 = 49.113 :=
by 
  sorry

end NUMINAMATH_GPT_addition_correctness_l285_28538


namespace NUMINAMATH_GPT_average_goals_is_92_l285_28557

-- Definitions based on conditions
def layla_goals : ℕ := 104
def kristin_fewer_goals : ℕ := 24
def kristin_goals : ℕ := layla_goals - kristin_fewer_goals
def combined_goals : ℕ := layla_goals + kristin_goals
def average_goals : ℕ := combined_goals / 2

-- Theorem
theorem average_goals_is_92 : average_goals = 92 := 
  sorry

end NUMINAMATH_GPT_average_goals_is_92_l285_28557


namespace NUMINAMATH_GPT_sum_of_consecutive_pages_with_product_15300_l285_28576

theorem sum_of_consecutive_pages_with_product_15300 : 
  ∃ n : ℕ, n * (n + 1) = 15300 ∧ n + (n + 1) = 247 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_pages_with_product_15300_l285_28576


namespace NUMINAMATH_GPT_container_weight_l285_28597

-- Definition of the problem conditions
def weight_of_copper_bar : ℕ := 90
def weight_of_steel_bar := weight_of_copper_bar + 20
def weight_of_tin_bar := weight_of_steel_bar / 2

-- Formal statement to be proven
theorem container_weight (n : ℕ) (h1 : weight_of_steel_bar = 2 * weight_of_tin_bar)
  (h2 : weight_of_steel_bar = weight_of_copper_bar + 20)
  (h3 : weight_of_copper_bar = 90) :
  20 * (weight_of_copper_bar + weight_of_steel_bar + weight_of_tin_bar) = 5100 := 
by sorry

end NUMINAMATH_GPT_container_weight_l285_28597


namespace NUMINAMATH_GPT_school_A_original_students_l285_28551

theorem school_A_original_students 
  (x y : ℕ) 
  (h1 : x + y = 864) 
  (h2 : x - 32 = y + 80) : 
  x = 488 := 
by 
  sorry

end NUMINAMATH_GPT_school_A_original_students_l285_28551


namespace NUMINAMATH_GPT_cost_per_liter_of_fuel_l285_28587

-- Definitions and conditions
def fuel_capacity : ℕ := 150
def initial_fuel : ℕ := 38
def change_received : ℕ := 14
def initial_money : ℕ := 350

-- Proof problem
theorem cost_per_liter_of_fuel :
  (initial_money - change_received) / (fuel_capacity - initial_fuel) = 3 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_liter_of_fuel_l285_28587


namespace NUMINAMATH_GPT_probability_of_winning_l285_28556

open Nat

theorem probability_of_winning (h : True) : 
  let num_cards := 3
  let num_books := 5
  (1 - (Nat.choose num_cards 2 * 2^num_books - num_cards) / num_cards^num_books) = 50 / 81 := sorry

end NUMINAMATH_GPT_probability_of_winning_l285_28556


namespace NUMINAMATH_GPT_multiply_negatives_l285_28527

theorem multiply_negatives : (- (1 / 2)) * (- 2) = 1 :=
by
  sorry

end NUMINAMATH_GPT_multiply_negatives_l285_28527


namespace NUMINAMATH_GPT_fruit_problem_l285_28593

theorem fruit_problem
  (A B C : ℕ)
  (hA : A = 4) 
  (hB : B = 6) 
  (hC : C = 12) :
  ∃ x : ℕ, 1 = x / 2 := 
by
  sorry

end NUMINAMATH_GPT_fruit_problem_l285_28593


namespace NUMINAMATH_GPT_inequality_k_l285_28509

variable {R : Type} [LinearOrderedField R] [Nontrivial R]

theorem inequality_k (x y z : R) (k : ℕ) (h : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) 
  (hineq : (1/x) + (1/y) + (1/z) ≥ x + y + z) :
  (1/x^k) + (1/y^k) + (1/z^k) ≥ x^k + y^k + z^k :=
sorry

end NUMINAMATH_GPT_inequality_k_l285_28509


namespace NUMINAMATH_GPT_Bryce_received_raisins_l285_28533

theorem Bryce_received_raisins :
  ∃ x : ℕ, (∀ y : ℕ, x = y + 6) ∧ (∀ z : ℕ, z = x / 2) → x = 12 :=
by
  sorry

end NUMINAMATH_GPT_Bryce_received_raisins_l285_28533


namespace NUMINAMATH_GPT_food_per_puppy_meal_l285_28584

-- Definitions for conditions
def mom_daily_food : ℝ := 1.5 * 3
def num_puppies : ℕ := 5
def total_food_needed : ℝ := 57
def num_days : ℕ := 6

-- Total food for the mom dog over the given period
def total_mom_food : ℝ := mom_daily_food * num_days

-- Total food for the puppies over the given period
def total_puppy_food : ℝ := total_food_needed - total_mom_food

-- Total number of puppy meals over the given period
def total_puppy_meals : ℕ := (num_puppies * 2) * num_days

theorem food_per_puppy_meal :
  total_puppy_food / total_puppy_meals = 0.5 :=
  sorry

end NUMINAMATH_GPT_food_per_puppy_meal_l285_28584


namespace NUMINAMATH_GPT_regular_polygon_sides_l285_28598

theorem regular_polygon_sides (n : ℕ) (h : n > 2) (h_angle : 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l285_28598


namespace NUMINAMATH_GPT_transformed_equation_correct_l285_28555
-- Import the necessary library

-- Define the original equation of the parabola
def original_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the translation functions for the transformations
def translate_right (x : ℝ) : ℝ := x - 1
def translate_down (y : ℝ) : ℝ := y - 3

-- Define the transformed parabola equation
def transformed_parabola (x : ℝ) : ℝ := -2 * (translate_right x)^2 |> translate_down

-- The theorem stating the transformed equation
theorem transformed_equation_correct :
  ∀ x, transformed_parabola x = -2 * (x - 1)^2 - 3 :=
by { sorry }

end NUMINAMATH_GPT_transformed_equation_correct_l285_28555


namespace NUMINAMATH_GPT_min_value_expression_l285_28520

theorem min_value_expression (a b : ℝ) : 
  4 + (a + b)^2 ≥ 4 ∧ (4 + (a + b)^2 = 4 ↔ a + b = 0) := by
sorry

end NUMINAMATH_GPT_min_value_expression_l285_28520


namespace NUMINAMATH_GPT_sine_product_inequality_l285_28549

theorem sine_product_inequality :
  (1 / 8 : ℝ) < (Real.sin (20 * Real.pi / 180) * Real.sin (50 * Real.pi / 180) * Real.sin (70 * Real.pi / 180)) ∧
                (Real.sin (20 * Real.pi / 180) * Real.sin (50 * Real.pi / 180) * Real.sin (70 * Real.pi / 180)) < (1 / 4 : ℝ) :=
sorry

end NUMINAMATH_GPT_sine_product_inequality_l285_28549


namespace NUMINAMATH_GPT_friends_area_is_greater_by_14_point_4_times_l285_28544

theorem friends_area_is_greater_by_14_point_4_times :
  let tommy_length := 2 * 200
  let tommy_width := 3 * 150
  let tommy_area := tommy_length * tommy_width
  let friend_block_area := 180 * 180
  let friend_area := 80 * friend_block_area
  friend_area / tommy_area = 14.4 :=
by
  let tommy_length := 2 * 200
  let tommy_width := 3 * 150
  let tommy_area := tommy_length * tommy_width
  let friend_block_area := 180 * 180
  let friend_area := 80 * friend_block_area
  sorry

end NUMINAMATH_GPT_friends_area_is_greater_by_14_point_4_times_l285_28544


namespace NUMINAMATH_GPT_smaller_number_is_5_l285_28529

theorem smaller_number_is_5 (x y : ℤ) (h1 : x + y = 18) (h2 : x - y = 8) : y = 5 := by
  sorry

end NUMINAMATH_GPT_smaller_number_is_5_l285_28529


namespace NUMINAMATH_GPT_hostel_provisions_l285_28566

theorem hostel_provisions (x : ℕ) :
  (250 * x = 200 * 60) → x = 48 :=
by
  sorry

end NUMINAMATH_GPT_hostel_provisions_l285_28566


namespace NUMINAMATH_GPT_wire_cut_example_l285_28534

theorem wire_cut_example (total_length piece_ratio : ℝ) (h1 : total_length = 28) (h2 : piece_ratio = 2.00001 / 5) :
  ∃ (shorter_piece : ℝ), shorter_piece + piece_ratio * shorter_piece = total_length ∧ shorter_piece = 20 :=
by
  sorry

end NUMINAMATH_GPT_wire_cut_example_l285_28534


namespace NUMINAMATH_GPT_hens_count_l285_28522

theorem hens_count (H C : ℕ) (h_heads : H + C = 60) (h_feet : 2 * H + 4 * C = 200) : H = 20 :=
by
  sorry

end NUMINAMATH_GPT_hens_count_l285_28522


namespace NUMINAMATH_GPT_triangle_angle_sum_l285_28567

theorem triangle_angle_sum (a b : ℝ) (ha : a = 40) (hb : b = 60) : ∃ x : ℝ, x = 180 - (a + b) :=
by
  use 80
  sorry

end NUMINAMATH_GPT_triangle_angle_sum_l285_28567


namespace NUMINAMATH_GPT_brick_width_correct_l285_28528

theorem brick_width_correct
  (courtyard_length_m : ℕ) (courtyard_width_m : ℕ) (brick_length_cm : ℕ) (num_bricks : ℕ)
  (total_area_cm : ℕ) (brick_width_cm : ℕ) :
  courtyard_length_m = 25 →
  courtyard_width_m = 16 →
  brick_length_cm = 20 →
  num_bricks = 20000 →
  total_area_cm = courtyard_length_m * 100 * courtyard_width_m * 100 →
  total_area_cm = num_bricks * brick_length_cm * brick_width_cm →
  brick_width_cm = 10 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_brick_width_correct_l285_28528


namespace NUMINAMATH_GPT_train_length_l285_28575

theorem train_length (speed_kmph : ℝ) (time_seconds : ℝ) (speed_mps : ℝ) (length_train : ℝ) : 
  speed_kmph = 90 → 
  time_seconds = 6 → 
  speed_mps = (speed_kmph * 1000 / 3600) →
  length_train = (speed_mps * time_seconds) → 
  length_train = 150 :=
by
  intros h_speed h_time h_speed_mps h_length
  sorry

end NUMINAMATH_GPT_train_length_l285_28575


namespace NUMINAMATH_GPT_problem1_problem2_l285_28583

-- Problem 1: Prove that the solution to f(x) <= 0 for a = -2 is [1, +∞)
theorem problem1 (x : ℝ) : (|x + 2| - 2 * x - 1 ≤ 0) ↔ (1 ≤ x) := sorry

-- Problem 2: Prove that the range of m such that there exists x ∈ ℝ satisfying f(x) + |x + 2| ≤ m for a = 1 is m ≥ 0
theorem problem2 (m : ℝ) : (∃ x : ℝ, |x - 1| - 2 * x - 1 + |x + 2| ≤ m) ↔ (0 ≤ m) := sorry

end NUMINAMATH_GPT_problem1_problem2_l285_28583


namespace NUMINAMATH_GPT_inequality_proof_l285_28541

theorem inequality_proof
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a^2 + b^2 + c^2 = 1) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ (2 * (a^3 + b^3 + c^3)) / (a * b * c) + 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l285_28541


namespace NUMINAMATH_GPT_analytical_expression_range_of_t_l285_28592

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x

theorem analytical_expression (x : ℝ) :
  (f (x + 1) - f x = 2 * x - 2) ∧ (f 1 = -2) :=
by
  sorry

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f x > 0 ∧ f (x + t) < 0 → x = 1) ↔ (-2 <= t ∧ t < -1) :=
by
  sorry

end NUMINAMATH_GPT_analytical_expression_range_of_t_l285_28592


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_squares_l285_28504

open Nat

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h_prod : a * b = 5) : 
  (1 : ℚ) / (a * a) + (1 : ℚ) / (b * b) = 26 / 25 :=
by
  -- proof steps skipping with sorry
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_squares_l285_28504


namespace NUMINAMATH_GPT_candies_per_person_l285_28580

def clowns : ℕ := 4
def children : ℕ := 30
def initial_candies : ℕ := 700
def candies_left : ℕ := 20

def total_people : ℕ := clowns + children
def candies_sold : ℕ := initial_candies - candies_left

theorem candies_per_person : candies_sold / total_people = 20 := by
  sorry

end NUMINAMATH_GPT_candies_per_person_l285_28580


namespace NUMINAMATH_GPT_pool_min_cost_l285_28578

noncomputable def CostMinimization (x : ℝ) : ℝ :=
  150 * 1600 + 720 * (x + 1600 / x)

theorem pool_min_cost :
  ∃ (x : ℝ), x = 40 ∧ CostMinimization x = 297600 :=
by
  sorry

end NUMINAMATH_GPT_pool_min_cost_l285_28578


namespace NUMINAMATH_GPT_trig_identity_l285_28585

theorem trig_identity
  (x : ℝ)
  (h : Real.tan (π / 4 + x) = 2014) :
  1 / Real.cos (2 * x) + Real.tan (2 * x) = 2014 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l285_28585


namespace NUMINAMATH_GPT_max_marks_l285_28507

theorem max_marks (M : ℝ) (h : 0.92 * M = 460) : M = 500 :=
by
  sorry

end NUMINAMATH_GPT_max_marks_l285_28507


namespace NUMINAMATH_GPT_average_snowfall_per_minute_l285_28568

def total_snowfall := 550
def days_in_december := 31
def hours_per_day := 24
def minutes_per_hour := 60

theorem average_snowfall_per_minute :
  (total_snowfall : ℝ) / (days_in_december * hours_per_day * minutes_per_hour) = 550 / (31 * 24 * 60) :=
by
  sorry

end NUMINAMATH_GPT_average_snowfall_per_minute_l285_28568


namespace NUMINAMATH_GPT_sum_n_div_n4_add_16_eq_9_div_320_l285_28505

theorem sum_n_div_n4_add_16_eq_9_div_320 :
  ∑' n : ℕ, n / (n^4 + 16) = 9 / 320 :=
sorry

end NUMINAMATH_GPT_sum_n_div_n4_add_16_eq_9_div_320_l285_28505


namespace NUMINAMATH_GPT_area_union_after_rotation_l285_28521

-- Define the sides of the triangle
def PQ : ℝ := 11
def QR : ℝ := 13
def PR : ℝ := 12

-- Define the condition that H is the centroid of the triangle PQR
def centroid (P Q R H : ℝ × ℝ) : Prop := sorry -- This definition would require geometric relationships.

-- Statement to prove the area of the union of PQR and P'Q'R' after 180° rotation about H.
theorem area_union_after_rotation (P Q R H : ℝ × ℝ) (hPQ : dist P Q = PQ) (hQR : dist Q R = QR) (hPR : dist P R = PR) (hH : centroid P Q R H) : 
  let s := (PQ + QR + PR) / 2
  let area_PQR := Real.sqrt (s * (s - PQ) * (s - QR) * (s - PR))
  2 * area_PQR = 12 * Real.sqrt 105 :=
sorry

end NUMINAMATH_GPT_area_union_after_rotation_l285_28521


namespace NUMINAMATH_GPT_pigeons_count_l285_28512

theorem pigeons_count :
  let initial_pigeons := 1
  let additional_pigeons := 1
  (initial_pigeons + additional_pigeons) = 2 :=
by
  sorry

end NUMINAMATH_GPT_pigeons_count_l285_28512


namespace NUMINAMATH_GPT_discounted_price_correct_l285_28572

def discounted_price (P : ℝ) : ℝ :=
  P * 0.80 * 0.90 * 0.95

theorem discounted_price_correct :
  discounted_price 9502.923976608186 = 6498.40 :=
by
  sorry

end NUMINAMATH_GPT_discounted_price_correct_l285_28572


namespace NUMINAMATH_GPT_coprime_divides_product_l285_28565

theorem coprime_divides_product {a b n : ℕ} (h1 : Nat.gcd a b = 1) (h2 : a ∣ n) (h3 : b ∣ n) : ab ∣ n :=
by
  sorry

end NUMINAMATH_GPT_coprime_divides_product_l285_28565


namespace NUMINAMATH_GPT_benny_books_l285_28516

variable (B : ℕ) -- the number of books Benny had initially

theorem benny_books (h : B - 10 + 33 = 47) : B = 24 :=
sorry

end NUMINAMATH_GPT_benny_books_l285_28516


namespace NUMINAMATH_GPT_sum_of_given_numbers_l285_28535

theorem sum_of_given_numbers : 30 + 80000 + 700 + 60 = 80790 :=
  by
    sorry

end NUMINAMATH_GPT_sum_of_given_numbers_l285_28535


namespace NUMINAMATH_GPT_find_f_neg_2_l285_28543

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x: ℝ, f (-x) = -f x

-- Problem statement
theorem find_f_neg_2 (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_fx_pos : ∀ x : ℝ, x > 0 → f x = 2 * x ^ 2 - 7) : 
  f (-2) = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg_2_l285_28543


namespace NUMINAMATH_GPT_total_points_scored_l285_28510

-- Definitions based on the conditions
def three_point_shots := 13
def two_point_shots := 20
def free_throws := 5
def missed_free_throws := 2
def points_per_three_point_shot := 3
def points_per_two_point_shot := 2
def points_per_free_throw := 1
def penalty_per_missed_free_throw := 1

-- Main statement proving the total points James scored
theorem total_points_scored :
  three_point_shots * points_per_three_point_shot +
  two_point_shots * points_per_two_point_shot +
  free_throws * points_per_free_throw -
  missed_free_throws * penalty_per_missed_free_throw = 82 :=
by
  sorry

end NUMINAMATH_GPT_total_points_scored_l285_28510


namespace NUMINAMATH_GPT_line_equation_through_point_l285_28564

theorem line_equation_through_point 
  (x y : ℝ)
  (h1 : (5, 2) ∈ {p : ℝ × ℝ | p.2 = p.1 * (2 / 5)})
  (h2 : (5, 2) ∈ {p : ℝ × ℝ | p.1 / 6 + p.2 / 12 = 1}) 
  (h3 : (5,2) ∈ {p : ℝ × ℝ | 2 * p.1 = p.2 }) :
  (2 * x + y - 12 = 0 ∨ 
   2 * x - 5 * y = 0) := 
sorry

end NUMINAMATH_GPT_line_equation_through_point_l285_28564


namespace NUMINAMATH_GPT_find_m_l285_28524

theorem find_m (m : ℝ) (h1 : m > 0) (h2 : ∃ s : ℝ, (s = (m + 1 - 4) / (2 - m)) ∧ s = Real.sqrt 5) :
  m = (10 - Real.sqrt 5) / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l285_28524


namespace NUMINAMATH_GPT_Charles_speed_with_music_l285_28502

theorem Charles_speed_with_music (S : ℝ) (h1 : 40 / 60 + 30 / 60 = 70 / 60) (h2 : S * (40 / 60) + 4 * (30 / 60) = 6) : S = 8 :=
by
  sorry

end NUMINAMATH_GPT_Charles_speed_with_music_l285_28502


namespace NUMINAMATH_GPT_quadratic_form_proof_l285_28531

theorem quadratic_form_proof (k : ℝ) (a b c : ℝ) (h1 : 8*k^2 - 16*k + 28 = a * (k + b)^2 + c) (h2 : a = 8) (h3 : b = -1) (h4 : c = 20) : c / b = -20 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_form_proof_l285_28531


namespace NUMINAMATH_GPT_area_square_given_diagonal_l285_28560

theorem area_square_given_diagonal (d : ℝ) (h : d = 16) : (∃ A : ℝ, A = 128) :=
by 
  sorry

end NUMINAMATH_GPT_area_square_given_diagonal_l285_28560


namespace NUMINAMATH_GPT_sub_two_three_l285_28558

theorem sub_two_three : 2 - 3 = -1 := 
by 
  sorry

end NUMINAMATH_GPT_sub_two_three_l285_28558


namespace NUMINAMATH_GPT_evaluate_expression_l285_28501

theorem evaluate_expression : 3 * (3 * (3 * (3 + 2) + 2) + 2) + 2 = 161 := sorry

end NUMINAMATH_GPT_evaluate_expression_l285_28501


namespace NUMINAMATH_GPT_number_difference_l285_28559

theorem number_difference (x y : ℕ) (h₁ : x + y = 41402) (h₂ : ∃ k : ℕ, x = 100 * k) (h₃ : y = x / 100) : x - y = 40590 :=
sorry

end NUMINAMATH_GPT_number_difference_l285_28559


namespace NUMINAMATH_GPT_wall_height_proof_l285_28561

-- The dimensions of the brick in meters
def brick_length : ℝ := 0.30
def brick_width : ℝ := 0.12
def brick_height : ℝ := 0.10

-- The dimensions of the wall in meters
def wall_length : ℝ := 6
def wall_width : ℝ := 4

-- The number of bricks needed
def number_of_bricks : ℝ := 1366.6666666666667

-- The height of the wall in meters
def wall_height : ℝ := 0.205

-- The volume of one brick
def volume_of_one_brick : ℝ := brick_length * brick_width * brick_height

-- The total volume of all bricks needed
def total_volume_of_bricks : ℝ := number_of_bricks * volume_of_one_brick

-- The volume of the wall
def volume_of_wall : ℝ := wall_length * wall_width * wall_height

-- Proof that the height of the wall is 0.205 meters
theorem wall_height_proof : volume_of_wall = total_volume_of_bricks :=
by
  -- use definitions to evaluate the equality
  sorry

end NUMINAMATH_GPT_wall_height_proof_l285_28561


namespace NUMINAMATH_GPT_convert_rectangular_to_polar_l285_28552

theorem convert_rectangular_to_polar (x y : ℝ) (h₁ : x = -2) (h₂ : y = -2) : 
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ (r, θ) = (2 * Real.sqrt 2, 5 * Real.pi / 4) := by
  sorry

end NUMINAMATH_GPT_convert_rectangular_to_polar_l285_28552


namespace NUMINAMATH_GPT_common_tangents_l285_28547

theorem common_tangents (r1 r2 d : ℝ) (h_r1 : r1 = 10) (h_r2 : r2 = 4) : 
  ∀ (n : ℕ), (n = 1) → ¬ (∃ (d : ℝ), 
    (6 < d ∧ d < 14 ∧ n = 2) ∨ 
    (d = 14 ∧ n = 3) ∨ 
    (d < 6 ∧ n = 0) ∨ 
    (d > 14 ∧ n = 4)) :=
by
  intro n h
  sorry

end NUMINAMATH_GPT_common_tangents_l285_28547


namespace NUMINAMATH_GPT_derivative_of_ln_2x_l285_28596

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x)

theorem derivative_of_ln_2x (x : ℝ) : deriv f x = 1 / x :=
  sorry

end NUMINAMATH_GPT_derivative_of_ln_2x_l285_28596


namespace NUMINAMATH_GPT_general_term_formula_l285_28537
-- Import the Mathlib library 

-- Define the conditions as given in the problem
/-- 
Define the sequence that represents the numerators. 
This is an arithmetic sequence of odd numbers starting from 1.
-/
def numerator (n : ℕ) : ℕ := 2 * n + 1

/-- 
Define the sequence that represents the denominators. 
This is a geometric sequence with the first term being 2 and common ratio being 2.
-/
def denominator (n : ℕ) : ℕ := 2^(n+1)

-- State the main theorem that we need to prove
theorem general_term_formula (n : ℕ) : (numerator n) / (denominator n) = (2 * n + 1) / 2^(n+1) :=
sorry

end NUMINAMATH_GPT_general_term_formula_l285_28537


namespace NUMINAMATH_GPT_solution_set_of_inequality_l285_28562

theorem solution_set_of_inequality (x : ℝ) : (x * |x - 1| > 0) ↔ (0 < x ∧ x < 1 ∨ 1 < x) := 
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l285_28562


namespace NUMINAMATH_GPT_solve_positive_integer_x_l285_28526

theorem solve_positive_integer_x : ∃ (x : ℕ), 4 * x^2 - 16 * x - 60 = 0 ∧ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_positive_integer_x_l285_28526


namespace NUMINAMATH_GPT_intersection_distance_squared_l285_28508

-- Definitions for the circles
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 1)^2 + (y - 4)^2 = 9

-- Statement to prove
theorem intersection_distance_squared : 
  ∃ C D : ℝ × ℝ, circle1 C.1 C.2 ∧ circle2 C.1 C.2 ∧ circle1 D.1 D.2 ∧ circle2 D.1 D.2 ∧ 
  (C ≠ D) ∧ ((C.1 - D.1)^2 + (C.2 - D.2)^2 = 224 / 9) :=
sorry

end NUMINAMATH_GPT_intersection_distance_squared_l285_28508


namespace NUMINAMATH_GPT_piggy_bank_dimes_l285_28595

theorem piggy_bank_dimes (q d : ℕ) 
  (h1 : q + d = 100) 
  (h2 : 25 * q + 10 * d = 1975) : 
  d = 35 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_piggy_bank_dimes_l285_28595


namespace NUMINAMATH_GPT_probability_even_sum_97_l285_28500

-- You don't need to include numbers since they are directly available in Lean's library
-- This will help to ensure broader compatibility and avoid namespace issues

theorem probability_even_sum_97 (m n : ℕ) (hmn : Nat.gcd m n = 1) 
  (hprob : (224 : ℚ) / 455 = m / n) : 
  m + n = 97 :=
sorry

end NUMINAMATH_GPT_probability_even_sum_97_l285_28500


namespace NUMINAMATH_GPT_coefficient_x5_in_product_l285_28554

noncomputable def P : Polynomial ℤ := 
  Polynomial.C 1 * Polynomial.X ^ 6 +
  Polynomial.C (-2) * Polynomial.X ^ 5 +
  Polynomial.C 3 * Polynomial.X ^ 4 +
  Polynomial.C (-4) * Polynomial.X ^ 3 +
  Polynomial.C 5 * Polynomial.X ^ 2 +
  Polynomial.C (-6) * Polynomial.X +
  Polynomial.C 7

noncomputable def Q : Polynomial ℤ := 
  Polynomial.C 3 * Polynomial.X ^ 4 +
  Polynomial.C (-4) * Polynomial.X ^ 3 +
  Polynomial.C 5 * Polynomial.X ^ 2 +
  Polynomial.C 6 * Polynomial.X +
  Polynomial.C (-8)

theorem coefficient_x5_in_product (p q : Polynomial ℤ) :
  (p * q).coeff 5 = 2 :=
by
  have P := 
    Polynomial.C 1 * Polynomial.X ^ 6 +
    Polynomial.C (-2) * Polynomial.X ^ 5 +
    Polynomial.C 3 * Polynomial.X ^ 4 +
    Polynomial.C (-4) * Polynomial.X ^ 3 +
    Polynomial.C 5 * Polynomial.X ^ 2 +
    Polynomial.C (-6) * Polynomial.X +
    Polynomial.C 7
  have Q := 
    Polynomial.C 3 * Polynomial.X ^ 4 +
    Polynomial.C (-4) * Polynomial.X ^ 3 +
    Polynomial.C 5 * Polynomial.X ^ 2 +
    Polynomial.C 6 * Polynomial.X +
    Polynomial.C (-8)

  sorry

end NUMINAMATH_GPT_coefficient_x5_in_product_l285_28554


namespace NUMINAMATH_GPT_find_S16_l285_28571

-- Definitions
def geom_seq (a : ℕ → ℝ) : Prop := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def sum_of_geom_seq (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, S n = a 0 * (1 - (a 1 / a 0)^n) / (1 - (a 1 / a 0))

-- Problem conditions
variables {a : ℕ → ℝ} {S : ℕ → ℝ}

axiom geom_seq_a : geom_seq a
axiom S4_eq : S 4 = 4
axiom S8_eq : S 8 = 12

-- Theorem
theorem find_S16 : S 16 = 60 :=
  sorry

end NUMINAMATH_GPT_find_S16_l285_28571


namespace NUMINAMATH_GPT_judys_school_week_l285_28591

theorem judys_school_week
  (pencils_used : ℕ)
  (packs_cost : ℕ)
  (total_cost : ℕ)
  (days_period : ℕ)
  (pencils_per_pack : ℕ)
  (pencils_in_school_days : ℕ)
  (total_pencil_use : ℕ) :
  (total_cost / packs_cost * pencils_per_pack = total_pencil_use) →
  (total_pencil_use / days_period = pencils_used) →
  (pencils_in_school_days / pencils_used = 5) :=
sorry

end NUMINAMATH_GPT_judys_school_week_l285_28591


namespace NUMINAMATH_GPT_number_of_possible_flags_l285_28519

def colors : List String := ["purple", "gold"]

noncomputable def num_choices_per_stripe (colors : List String) : Nat := 
  colors.length

theorem number_of_possible_flags :
  (num_choices_per_stripe colors) ^ 3 = 8 := 
by
  -- Proof
  sorry

end NUMINAMATH_GPT_number_of_possible_flags_l285_28519


namespace NUMINAMATH_GPT_uki_cupcakes_per_day_l285_28523

-- Define the conditions
def price_cupcake : ℝ := 1.50
def price_cookie : ℝ := 2
def price_biscuit : ℝ := 1
def daily_cookies : ℝ := 10
def daily_biscuits : ℝ := 20
def total_earnings : ℝ := 350
def days : ℝ := 5

-- Define the number of cupcakes baked per day
def cupcakes_per_day (x : ℝ) : Prop :=
  let earnings_cupcakes := price_cupcake * x * days
  let earnings_cookies := price_cookie * daily_cookies * days
  let earnings_biscuits := price_biscuit * daily_biscuits * days
  earnings_cupcakes + earnings_cookies + earnings_biscuits = total_earnings

-- The statement to be proven
theorem uki_cupcakes_per_day : cupcakes_per_day 20 :=
by 
  sorry

end NUMINAMATH_GPT_uki_cupcakes_per_day_l285_28523


namespace NUMINAMATH_GPT_ceiling_fraction_evaluation_l285_28517

theorem ceiling_fraction_evaluation :
  (Int.ceil ((19 : ℚ) / 8 - Int.ceil ((45 : ℚ) / 19)) / Int.ceil ((45 : ℚ) / 8 + Int.ceil ((8 * 19 : ℚ) / 45))) = 0 :=
by
  sorry

end NUMINAMATH_GPT_ceiling_fraction_evaluation_l285_28517


namespace NUMINAMATH_GPT_find_three_digit_number_l285_28588

def digits_to_num (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

theorem find_three_digit_number (a b c : ℕ) (h1 : 8 * a + 5 * b + c = 100) (h2 : a + b + c = 20) :
  digits_to_num a b c = 866 :=
by 
  sorry

end NUMINAMATH_GPT_find_three_digit_number_l285_28588


namespace NUMINAMATH_GPT_decreasing_on_interval_l285_28569

variable {x m n : ℝ}

def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := |x^2 - 2 * m * x + n|

theorem decreasing_on_interval
  (h : ∀ x, f x m n = |x^2 - 2 * m * x + n|)
  (h_cond : m^2 - n ≤ 0) :
  ∀ x y, x ≤ y → y ≤ m → f y m n ≤ f x m n :=
sorry

end NUMINAMATH_GPT_decreasing_on_interval_l285_28569


namespace NUMINAMATH_GPT_probabilities_inequalities_l285_28579

variables (M N : Prop) (P : Prop → ℝ)

axiom P_pos_M : P M > 0
axiom P_pos_N : P N > 0
axiom P_cond_N_M : P (N ∧ M) / P M > P (N ∧ ¬M) / P (¬M)

theorem probabilities_inequalities :
    (P (N ∧ M) / P M > P (N ∧ ¬M) / P (¬M)) ∧
    (P (N ∧ M) > P N * P M) ∧
    (P (M ∧ N) / P N > P (M ∧ ¬N) / P (¬N)) :=
by
    sorry

end NUMINAMATH_GPT_probabilities_inequalities_l285_28579


namespace NUMINAMATH_GPT_minimum_surface_area_of_cube_l285_28546

noncomputable def brick_length := 25
noncomputable def brick_width := 15
noncomputable def brick_height := 5
noncomputable def side_length := Nat.lcm brick_width brick_length
noncomputable def surface_area := 6 * side_length * side_length

theorem minimum_surface_area_of_cube : surface_area = 33750 := 
by
  sorry

end NUMINAMATH_GPT_minimum_surface_area_of_cube_l285_28546


namespace NUMINAMATH_GPT_marked_price_percentage_fixed_l285_28511

-- Definitions based on the conditions
def discount_percentage : ℝ := 0.18461538461538467
def profit_percentage : ℝ := 0.06

-- The final theorem statement
theorem marked_price_percentage_fixed (CP MP SP : ℝ) 
  (h1 : SP = CP * (1 + profit_percentage))  
  (h2 : SP = MP * (1 - discount_percentage)) :
  (MP / CP - 1) * 100 = 30 := 
sorry

end NUMINAMATH_GPT_marked_price_percentage_fixed_l285_28511


namespace NUMINAMATH_GPT_other_divisor_l285_28503

theorem other_divisor (x : ℕ) (h1 : 266 % 33 = 2) (h2 : 266 % x = 2) : x = 132 :=
sorry

end NUMINAMATH_GPT_other_divisor_l285_28503


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l285_28513

theorem arithmetic_sequence_sum (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) (m : ℕ) 
  (h1 : S_n m = 0) (h2 : S_n (m - 1) = -2) (h3 : S_n (m + 1) = 3) :
  m = 5 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l285_28513


namespace NUMINAMATH_GPT_cost_of_item_D_is_30_usd_l285_28514

noncomputable def cost_of_item_D_in_usd (total_spent items_ABC_spent tax_paid service_fee_rate exchange_rate : ℝ) : ℝ :=
  let total_spent_with_fee := total_spent * (1 + service_fee_rate)
  let item_D_cost_FC := total_spent_with_fee - items_ABC_spent
  item_D_cost_FC * exchange_rate

theorem cost_of_item_D_is_30_usd
  (total_spent : ℝ)
  (items_ABC_spent : ℝ)
  (tax_paid : ℝ)
  (service_fee_rate : ℝ)
  (exchange_rate : ℝ)
  (h_total_spent : total_spent = 500)
  (h_items_ABC_spent : items_ABC_spent = 450)
  (h_tax_paid : tax_paid = 60)
  (h_service_fee_rate : service_fee_rate = 0.02)
  (h_exchange_rate : exchange_rate = 0.5) :
  cost_of_item_D_in_usd total_spent items_ABC_spent tax_paid service_fee_rate exchange_rate = 30 :=
by
  have h1 : total_spent * (1 + service_fee_rate) = 500 * 1.02 := sorry
  have h2 : 500 * 1.02 - 450 = 60 := sorry
  have h3 : 60 * 0.5 = 30 := sorry
  sorry

end NUMINAMATH_GPT_cost_of_item_D_is_30_usd_l285_28514


namespace NUMINAMATH_GPT_part_a_l285_28518

open Complex

theorem part_a (z : ℂ) (hz : abs z = 1) :
  (abs (z + 1) - Real.sqrt 2) * (abs (z - 1) - Real.sqrt 2) ≤ 0 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_part_a_l285_28518


namespace NUMINAMATH_GPT_locus_of_P_is_single_ray_l285_28573
  
noncomputable def M : ℝ × ℝ := (1, 0)
noncomputable def N : ℝ × ℝ := (3, 0)

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2))

theorem locus_of_P_is_single_ray (P : ℝ × ℝ) (h : distance P M - distance P N = 2) : 
∃ α : ℝ, P = (3 + α * (P.1 - 3), α * P.2) :=
sorry

end NUMINAMATH_GPT_locus_of_P_is_single_ray_l285_28573


namespace NUMINAMATH_GPT_probability_not_snowing_l285_28550

  -- Define the probability that it will snow tomorrow
  def P_snowing : ℚ := 2 / 5

  -- Define the probability that it will not snow tomorrow
  def P_not_snowing : ℚ := 1 - P_snowing

  -- Theorem stating the required proof
  theorem probability_not_snowing : P_not_snowing = 3 / 5 :=
  by 
    -- Proof would go here
    sorry
  
end NUMINAMATH_GPT_probability_not_snowing_l285_28550


namespace NUMINAMATH_GPT_proportional_segments_l285_28599

-- Define the tetrahedron and points
structure Tetrahedron :=
(A B C D O A1 B1 C1 : ℝ)

-- Define the conditions of the problem
variables {tetra : Tetrahedron}

-- Define the segments and their relationships
axiom segments_parallel (DA : ℝ) (DB : ℝ) (DC : ℝ)
  (OA1 : ℝ) (OB1 : ℝ) (OC1 : ℝ) :
  OA1 / DA + OB1 / DB + OC1 / DC = 1

-- The theorem to prove, which follows directly from the given axiom 
theorem proportional_segments (DA DB DC : ℝ)
  (OA1 : ℝ) (OB1 : ℝ) (OC1 : ℝ) :
  OA1 / DA + OB1 / DB + OC1 / DC = 1 :=
segments_parallel DA DB DC OA1 OB1 OC1

end NUMINAMATH_GPT_proportional_segments_l285_28599


namespace NUMINAMATH_GPT_cost_of_car_l285_28540

theorem cost_of_car (initial_payment : ℕ) (num_installments : ℕ) (installment_amount : ℕ) : 
  initial_payment = 3000 →
  num_installments = 6 →
  installment_amount = 2500 →
  initial_payment + num_installments * installment_amount = 18000 :=
by
  intros h_initial h_num h_installment
  sorry

end NUMINAMATH_GPT_cost_of_car_l285_28540


namespace NUMINAMATH_GPT_parallel_lines_iff_determinant_zero_l285_28542

theorem parallel_lines_iff_determinant_zero (a1 b1 c1 a2 b2 c2 : ℝ) :
  (a1 * b2 - a2 * b1 = 0) ↔ ((a1 * c2 - a2 * c1 = 0) → (b1 * c2 - b2 * c1 = 0)) := 
sorry

end NUMINAMATH_GPT_parallel_lines_iff_determinant_zero_l285_28542


namespace NUMINAMATH_GPT_first_three_digits_of_decimal_part_of_10_pow_1001_plus_1_pow_9_div_8_l285_28589

noncomputable def first_three_digits_of_decimal_part (x : ℝ) : ℕ :=
  -- here we would have the actual definition
  sorry

theorem first_three_digits_of_decimal_part_of_10_pow_1001_plus_1_pow_9_div_8 :
  first_three_digits_of_decimal_part ((10^1001 + 1)^((9:ℝ) / 8)) = 125 :=
sorry

end NUMINAMATH_GPT_first_three_digits_of_decimal_part_of_10_pow_1001_plus_1_pow_9_div_8_l285_28589


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l285_28532

-- Define the polynomial function f(x)
def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 20*x^3 + x^2 - 47*x + 15

-- State the theorem to be proved with the given conditions
theorem remainder_when_divided_by_x_minus_2 :
  f 2 = -11 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l285_28532


namespace NUMINAMATH_GPT_expectation_fish_l285_28563

noncomputable def fish_distribution : ℕ → ℚ → ℚ → ℚ → ℚ :=
  fun N a b c => (a / b) * (1 - (c / (a + b + c) ^ N))

def x_distribution : ℚ := 0.18
def y_distribution : ℚ := 0.02
def other_distribution : ℚ := 0.80
def total_fish : ℕ := 10

theorem expectation_fish :
  fish_distribution total_fish x_distribution y_distribution other_distribution = 1.6461 :=
  by
    sorry

end NUMINAMATH_GPT_expectation_fish_l285_28563


namespace NUMINAMATH_GPT_product_mod_7_l285_28515

theorem product_mod_7 :
  (2009 % 7 = 4) ∧ (2010 % 7 = 5) ∧ (2011 % 7 = 6) ∧ (2012 % 7 = 0) →
  (2009 * 2010 * 2011 * 2012) % 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_product_mod_7_l285_28515


namespace NUMINAMATH_GPT_find_integer_K_l285_28548

-- Definitions based on the conditions
def is_valid_K (K Z : ℤ) : Prop :=
  Z = K^4 ∧ 3000 < Z ∧ Z < 4000 ∧ K > 1 ∧ ∃ (z : ℤ), K^4 = z^3

theorem find_integer_K :
  ∃ (K : ℤ), is_valid_K K 2401 :=
by
  sorry

end NUMINAMATH_GPT_find_integer_K_l285_28548


namespace NUMINAMATH_GPT_find_a_l285_28586

noncomputable def geometric_sequence_solution (a : ℝ) : Prop :=
  (a + 1) ^ 2 = (1 / (a - 1)) * (a ^ 2 - 1)

theorem find_a (a : ℝ) : geometric_sequence_solution a → a = 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l285_28586


namespace NUMINAMATH_GPT_student_test_ratio_l285_28581

theorem student_test_ratio :
  ∀ (total_questions correct_responses : ℕ),
  total_questions = 100 →
  correct_responses = 93 →
  (total_questions - correct_responses) / correct_responses = 7 / 93 :=
by
  intros total_questions correct_responses h_total_questions h_correct_responses
  sorry

end NUMINAMATH_GPT_student_test_ratio_l285_28581


namespace NUMINAMATH_GPT_robert_coin_arrangement_l285_28553

noncomputable def num_arrangements (gold : ℕ) (silver : ℕ) : ℕ :=
  if gold + silver = 8 ∧ gold = 5 ∧ silver = 3 then 504 else 0

theorem robert_coin_arrangement :
  num_arrangements 5 3 = 504 := 
sorry

end NUMINAMATH_GPT_robert_coin_arrangement_l285_28553


namespace NUMINAMATH_GPT_central_angle_of_region_l285_28539

theorem central_angle_of_region (A : ℝ) (θ : ℝ) (h : (1:ℝ) / 8 = (θ / 360) * A / A) : θ = 45 :=
by
  sorry

end NUMINAMATH_GPT_central_angle_of_region_l285_28539
