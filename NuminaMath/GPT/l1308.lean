import Mathlib

namespace NUMINAMATH_GPT_find_N_l1308_130810

theorem find_N (a b c N : ℝ) (h1 : a + b + c = 120) (h2 : a - 10 = N) 
               (h3 : b + 10 = N) (h4 : 7 * c = N): N = 56 :=
by
  sorry

end NUMINAMATH_GPT_find_N_l1308_130810


namespace NUMINAMATH_GPT_functional_equation_solution_l1308_130886

theorem functional_equation_solution (f : ℤ → ℤ)
  (h : ∀ m n : ℤ, f (f (m + n)) = f m + f n) :
  (∃ a : ℤ, ∀ n : ℤ, f n = n + a) ∨ (∀ n : ℤ, f n = 0) := by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l1308_130886


namespace NUMINAMATH_GPT_opposite_of_neg2023_l1308_130813

theorem opposite_of_neg2023 : ∀ (x : Int), -2023 + x = 0 → x = 2023 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_opposite_of_neg2023_l1308_130813


namespace NUMINAMATH_GPT_train_length_proof_l1308_130884

noncomputable def train_speed_kmh : ℝ := 50
noncomputable def crossing_time_s : ℝ := 9
noncomputable def length_of_train_m : ℝ := 125

theorem train_length_proof:
  ∀ (speed_kmh: ℝ) (time_s: ℝ), 
  speed_kmh = train_speed_kmh →
  time_s = crossing_time_s →
  (speed_kmh * (1000 / 3600) * time_s) = length_of_train_m :=
by intros speed_kmh time_s h_speed_kmh h_time_s
   -- Proof omitted
   sorry

end NUMINAMATH_GPT_train_length_proof_l1308_130884


namespace NUMINAMATH_GPT_driver_travel_distance_per_week_l1308_130882

noncomputable def daily_distance := 30 * 3 + 25 * 4 + 40 * 2

noncomputable def total_weekly_distance := daily_distance * 6 + 35 * 5

theorem driver_travel_distance_per_week : total_weekly_distance = 1795 := by
  simp [daily_distance, total_weekly_distance]
  done

end NUMINAMATH_GPT_driver_travel_distance_per_week_l1308_130882


namespace NUMINAMATH_GPT_solution_set_inequality_l1308_130821

theorem solution_set_inequality (x : ℝ) : (x + 1) * (2 - x) < 0 ↔ x > 2 ∨ x < -1 :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_l1308_130821


namespace NUMINAMATH_GPT_darkest_cell_product_l1308_130811

theorem darkest_cell_product (a b c d : ℕ)
  (h1 : a > 1) (h2 : b > 1) (h3 : c = a * b)
  (h4 : d = c * (9 * 5) * (9 * 11)) :
  d = 245025 :=
by
  sorry

end NUMINAMATH_GPT_darkest_cell_product_l1308_130811


namespace NUMINAMATH_GPT_cory_prime_sum_l1308_130832

def primes_between_30_and_60 : List ℕ := [31, 37, 41, 43, 47, 53, 59]

theorem cory_prime_sum :
  let smallest := 31
  let largest := 59
  let median := 43
  smallest ∈ primes_between_30_and_60 ∧
  largest ∈ primes_between_30_and_60 ∧
  median ∈ primes_between_30_and_60 ∧
  primes_between_30_and_60 = [31, 37, 41, 43, 47, 53, 59] → 
  smallest + largest + median = 133 := 
by
  intros; sorry

end NUMINAMATH_GPT_cory_prime_sum_l1308_130832


namespace NUMINAMATH_GPT_maximum_term_of_sequence_l1308_130889

noncomputable def a (n : ℕ) : ℝ := n * (3 / 4)^n

theorem maximum_term_of_sequence : ∃ n : ℕ, a n = a 3 ∧ ∀ m : ℕ, a m ≤ a 3 :=
by sorry

end NUMINAMATH_GPT_maximum_term_of_sequence_l1308_130889


namespace NUMINAMATH_GPT_area_of_rhombus_perimeter_of_rhombus_l1308_130800

-- Definitions and conditions for the area of the rhombus
def d1 : ℕ := 18
def d2 : ℕ := 16

-- Definition for the side length of the rhombus
def side_length : ℕ := 10

-- Statement for the area of the rhombus
theorem area_of_rhombus : (d1 * d2) / 2 = 144 := by
  sorry

-- Statement for the perimeter of the rhombus
theorem perimeter_of_rhombus : 4 * side_length = 40 := by
  sorry

end NUMINAMATH_GPT_area_of_rhombus_perimeter_of_rhombus_l1308_130800


namespace NUMINAMATH_GPT_basketball_not_table_tennis_l1308_130872

-- Definitions and conditions
def total_students := 30
def like_basketball := 15
def like_table_tennis := 10
def do_not_like_either := 8
def like_both (x : ℕ) := x

-- Theorem statement
theorem basketball_not_table_tennis (x : ℕ) (H : (like_basketball - x) + (like_table_tennis - x) + x + do_not_like_either = total_students) : (like_basketball - x) = 12 :=
by
  sorry

end NUMINAMATH_GPT_basketball_not_table_tennis_l1308_130872


namespace NUMINAMATH_GPT_decompose_two_over_eleven_decompose_two_over_n_l1308_130829

-- Problem 1: Decompose 2/11
theorem decompose_two_over_eleven : (2 : ℚ) / 11 = (1 / 6) + (1 / 66) :=
  sorry

-- Problem 2: General form for 2/n for odd n >= 5
theorem decompose_two_over_n (n : ℕ) (hn : n ≥ 5) (odd_n : n % 2 = 1) :
  (2 : ℚ) / n = (1 / ((n + 1) / 2)) + (1 / (n * (n + 1) / 2)) :=
  sorry

end NUMINAMATH_GPT_decompose_two_over_eleven_decompose_two_over_n_l1308_130829


namespace NUMINAMATH_GPT_log_monotonic_increasing_l1308_130875

noncomputable def f (a x : ℝ) := Real.log x / Real.log a

theorem log_monotonic_increasing (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : 1 < a) :
  f a (a + 1) > f a 2 := 
by
  -- Here the actual proof will be added.
  sorry

end NUMINAMATH_GPT_log_monotonic_increasing_l1308_130875


namespace NUMINAMATH_GPT_dots_not_visible_l1308_130812

def total_dots (n_dice : ℕ) : ℕ := n_dice * 21

def sum_visible_dots (visible : List ℕ) : ℕ := visible.foldl (· + ·) 0

theorem dots_not_visible (visible : List ℕ) (h : visible = [1, 1, 2, 3, 4, 5, 5, 6]) :
  total_dots 4 - sum_visible_dots visible = 57 :=
by
  rw [total_dots, sum_visible_dots]
  simp
  sorry

end NUMINAMATH_GPT_dots_not_visible_l1308_130812


namespace NUMINAMATH_GPT_percentage_problem_l1308_130822

variable (x : ℝ)

theorem percentage_problem (h : 0.4 * x = 160) : 240 / x = 0.6 :=
by sorry

end NUMINAMATH_GPT_percentage_problem_l1308_130822


namespace NUMINAMATH_GPT_distance_on_dirt_road_l1308_130888

theorem distance_on_dirt_road :
  ∀ (initial_gap distance_gap_on_city dirt_road_distance : ℝ),
  initial_gap = 2 → 
  distance_gap_on_city = initial_gap - ((initial_gap - (40 * (1 / 30)))) → 
  dirt_road_distance = distance_gap_on_city * (40 / 60) * (70 / 40) * (30 / 70) →
  dirt_road_distance = 1 :=
by
  intros initial_gap distance_gap_on_city dirt_road_distance h1 h2 h3
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_distance_on_dirt_road_l1308_130888


namespace NUMINAMATH_GPT_total_animals_for_sale_l1308_130808

theorem total_animals_for_sale (dogs cats birds fish : ℕ) 
  (h1 : dogs = 6)
  (h2 : cats = dogs / 2)
  (h3 : birds = dogs * 2)
  (h4 : fish = dogs * 3) :
  dogs + cats + birds + fish = 39 := 
by
  sorry

end NUMINAMATH_GPT_total_animals_for_sale_l1308_130808


namespace NUMINAMATH_GPT_sum_on_simple_interest_is_1750_l1308_130807

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r)^t - P

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

theorem sum_on_simple_interest_is_1750 :
  let P_ci := 4000
  let r_ci := 0.10
  let t_ci := 2
  let r_si := 0.08
  let t_si := 3
  let CI := compound_interest P_ci r_ci t_ci
  let SI := CI / 2
  let P_si := SI / (r_si * t_si)
  P_si = 1750 :=
by
  sorry

end NUMINAMATH_GPT_sum_on_simple_interest_is_1750_l1308_130807


namespace NUMINAMATH_GPT_determine_c_l1308_130862

theorem determine_c (c : ℝ) (r : ℝ) (h1 : 2 * r^2 - 8 * r - c = 0) (h2 : r ≠ 0) (h3 : 2 * (r + 5.5)^2 + 5 * (r + 5.5) = c) :
  c = 12 :=
sorry

end NUMINAMATH_GPT_determine_c_l1308_130862


namespace NUMINAMATH_GPT_arithmetic_operations_result_eq_one_over_2016_l1308_130826

theorem arithmetic_operations_result_eq_one_over_2016 :
  (∃ op1 op2 : ℚ → ℚ → ℚ, op1 (1/8) (op2 (1/9) (1/28)) = 1/2016) :=
sorry

end NUMINAMATH_GPT_arithmetic_operations_result_eq_one_over_2016_l1308_130826


namespace NUMINAMATH_GPT_unique_a_for_system_solution_l1308_130870

-- Define the variables
variables (a b x y : ℝ)

-- Define the system of equations
def system_has_solution (a b : ℝ) : Prop :=
  ∃ x y : ℝ, 2^(b * x) + (a + 1) * b * y^2 = a^2 ∧ (a-1) * x^3 + y^3 = 1

-- Main theorem statement
theorem unique_a_for_system_solution :
  a = -1 ↔ ∀ b : ℝ, system_has_solution a b :=
sorry

end NUMINAMATH_GPT_unique_a_for_system_solution_l1308_130870


namespace NUMINAMATH_GPT_exponent_property_l1308_130855

theorem exponent_property (a b : ℕ) : (a * b^2)^3 = a^3 * b^6 :=
by sorry

end NUMINAMATH_GPT_exponent_property_l1308_130855


namespace NUMINAMATH_GPT_license_plate_count_l1308_130865

theorem license_plate_count :
  let letters := 26
  let digits := 10
  let second_char_options := letters - 1 + digits
  let third_char_options := digits - 1
  letters * second_char_options * third_char_options = 8190 :=
by
  sorry

end NUMINAMATH_GPT_license_plate_count_l1308_130865


namespace NUMINAMATH_GPT_age_sum_l1308_130835

-- Defining the ages of Henry and Jill
def Henry_age : ℕ := 20
def Jill_age : ℕ := 13

-- The statement we need to prove
theorem age_sum : Henry_age + Jill_age = 33 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_age_sum_l1308_130835


namespace NUMINAMATH_GPT_angle_relationship_l1308_130805

-- Define the angles and the relationship
def larger_angle : ℝ := 99
def smaller_angle : ℝ := 81

-- State the problem as a theorem
theorem angle_relationship : larger_angle - smaller_angle = 18 := 
by
  -- The proof would be here
  sorry

end NUMINAMATH_GPT_angle_relationship_l1308_130805


namespace NUMINAMATH_GPT_popsicle_count_l1308_130848

-- Define the number of each type of popsicles
def num_grape_popsicles : Nat := 2
def num_cherry_popsicles : Nat := 13
def num_banana_popsicles : Nat := 2

-- Prove the total number of popsicles
theorem popsicle_count : num_grape_popsicles + num_cherry_popsicles + num_banana_popsicles = 17 := by
  sorry

end NUMINAMATH_GPT_popsicle_count_l1308_130848


namespace NUMINAMATH_GPT_vanya_speed_increased_by_4_l1308_130858

variable (v : ℝ)
variable (h1 : (v + 2) / v = 2.5)

theorem vanya_speed_increased_by_4 (h1 : (v + 2) / v = 2.5) : (v + 4) / v = 4 := 
sorry

end NUMINAMATH_GPT_vanya_speed_increased_by_4_l1308_130858


namespace NUMINAMATH_GPT_first_guinea_pig_food_l1308_130814

theorem first_guinea_pig_food (x : ℕ) (h1 : ∃ x : ℕ, R = x + 2 * x + (2 * x + 3)) (hp : 13 = x + 2 * x + (2 * x + 3)) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_first_guinea_pig_food_l1308_130814


namespace NUMINAMATH_GPT_auditorium_total_chairs_l1308_130883

theorem auditorium_total_chairs 
  (n : ℕ)
  (h1 : 2 + 5 - 1 = n)   -- n is the number of rows which is equal to 6
  (h2 : 3 + 4 - 1 = n)   -- n is the number of chairs per row which is also equal to 6
  : n * n = 36 :=        -- the total number of chairs is 36
by
  sorry

end NUMINAMATH_GPT_auditorium_total_chairs_l1308_130883


namespace NUMINAMATH_GPT_inequality_positives_l1308_130896

theorem inequality_positives (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a * (a + 1)) / (b + 1) + (b * (b + 1)) / (a + 1) ≥ a + b :=
sorry

end NUMINAMATH_GPT_inequality_positives_l1308_130896


namespace NUMINAMATH_GPT_complement_intersect_A_B_range_of_a_l1308_130845

-- Definitions for sets A and B
def setA : Set ℝ := {x | -2 < x ∧ x < 0}
def setB : Set ℝ := {x | ∃ y, y = Real.sqrt (x + 1)}

-- First statement to prove
theorem complement_intersect_A_B : (setAᶜ ∩ setB) = {x | x ≥ 0} :=
  sorry

-- Definition for set C
def setC (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a + 1}

-- Second statement to prove
theorem range_of_a (a : ℝ) : (setC a ⊆ setA) ↔ (a ≤ -1) ∨ (-1 ≤ a ∧ a ≤ -1 / 2) :=
  sorry

end NUMINAMATH_GPT_complement_intersect_A_B_range_of_a_l1308_130845


namespace NUMINAMATH_GPT_sin_300_eq_neg_sqrt3_div_2_l1308_130894

/-- Prove that the sine of 300 degrees is equal to -√3/2. -/
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
sorry

end NUMINAMATH_GPT_sin_300_eq_neg_sqrt3_div_2_l1308_130894


namespace NUMINAMATH_GPT_number_of_blue_fish_l1308_130843

def total_fish : ℕ := 22
def goldfish : ℕ := 15
def blue_fish : ℕ := total_fish - goldfish

theorem number_of_blue_fish : blue_fish = 7 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_number_of_blue_fish_l1308_130843


namespace NUMINAMATH_GPT_no_very_convex_function_exists_l1308_130854

-- Definition of very convex function
def very_convex (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x + f y) / 2 ≥ f ((x + y) / 2) + |x - y|

-- Theorem stating the non-existence of very convex functions
theorem no_very_convex_function_exists : ¬∃ f : ℝ → ℝ, very_convex f :=
by {
  sorry
}

end NUMINAMATH_GPT_no_very_convex_function_exists_l1308_130854


namespace NUMINAMATH_GPT_range_of_m_l1308_130881

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 - m * x + 1 > 0) ↔ (-2 < m ∧ m < 2) :=
  sorry

end NUMINAMATH_GPT_range_of_m_l1308_130881


namespace NUMINAMATH_GPT_ordered_pairs_bound_l1308_130874

variable (m n : ℕ) (a b : ℕ → ℝ)

theorem ordered_pairs_bound
  (h_m : m ≥ n)
  (h_n : n ≥ 2022)
  : (∃ (pairs : Finset (ℕ × ℕ)), 
      (∀ i j, (i, j) ∈ pairs → 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n ∧ |a i + b j - (i * j)| ≤ m) ∧
      pairs.card ≤ 3 * n * Real.sqrt (m * Real.log (n))) := 
  sorry

end NUMINAMATH_GPT_ordered_pairs_bound_l1308_130874


namespace NUMINAMATH_GPT_equations_have_different_graphs_l1308_130873

theorem equations_have_different_graphs :
  ¬(∀ x : ℝ, (2 * (x - 3)) / (x + 3) = 2 * (x - 3) ∧ 
              (x + 3) * ((2 * x^2 - 18) / (x + 3)) = 2 * x^2 - 18 ∧
              (2 * x - 3) = (2 * (x - 3)) ∧ 
              (2 * x - 3) = (2 * x - 3)) :=
by
  sorry

end NUMINAMATH_GPT_equations_have_different_graphs_l1308_130873


namespace NUMINAMATH_GPT_proof_problem_l1308_130816

theorem proof_problem (a b : ℝ) (n : ℕ) 
  (P1 P2 : ℝ × ℝ)
  (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_n_gt_1 : n > 1)
  (h_P1_on_curve : P1.1 ^ n = a * P1.2 ^ n + b)
  (h_P2_on_curve : P2.1 ^ n = a * P2.2 ^ n + b)
  (h_y1_lt_y2 : P1.2 < P2.2)
  (A : ℝ) (h_A : A = (1/2) * |P1.1 * P2.2 - P2.1 * P1.2|) :
  b * P2.2 > 2 * n * P1.2 ^ (n - 1) * a ^ (1 - (1 / n)) * A :=
sorry

end NUMINAMATH_GPT_proof_problem_l1308_130816


namespace NUMINAMATH_GPT_x_y_difference_is_perfect_square_l1308_130866

theorem x_y_difference_is_perfect_square (x y : ℕ) (h : 3 * x^2 + x = 4 * y^2 + y) : ∃ k : ℕ, k^2 = x - y :=
by {sorry}

end NUMINAMATH_GPT_x_y_difference_is_perfect_square_l1308_130866


namespace NUMINAMATH_GPT_units_digit_of_17_pow_3_mul_24_l1308_130877

def unit_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_17_pow_3_mul_24 :
  unit_digit (17^3 * 24) = 2 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_17_pow_3_mul_24_l1308_130877


namespace NUMINAMATH_GPT_minimum_hexagon_perimeter_l1308_130867

-- Define the conditions given in the problem
def small_equilateral_triangle (side_length : ℝ) (triangle_count : ℕ) :=
  triangle_count = 57 ∧ side_length = 1

def hexagon_with_conditions (angle_condition : ℝ → Prop) :=
  ∀ θ, angle_condition θ → θ ≤ 180 ∧ θ > 0

-- State the main problem as a theorem
theorem minimum_hexagon_perimeter : ∀ n : ℕ, ∃ p : ℕ,
  (small_equilateral_triangle 1 57) → 
  (∃ angle_condition, hexagon_with_conditions angle_condition) →
  (n = 57) →
  p = 19 :=
by
  sorry

end NUMINAMATH_GPT_minimum_hexagon_perimeter_l1308_130867


namespace NUMINAMATH_GPT_find_k_l1308_130868

theorem find_k 
  (A B X Y : ℝ × ℝ)
  (hA : A = (-3, 0))
  (hB : B = (0, -3))
  (hX : X = (0, 9))
  (Yx : Y.1 = 15)
  (hXY_parallel : (Y.2 - X.2) / (Y.1 - X.1) = (B.2 - A.2) / (B.1 - A.1)) :
  Y.2 = -6 := by
  -- proofs are omitted as per the requirements
  sorry

end NUMINAMATH_GPT_find_k_l1308_130868


namespace NUMINAMATH_GPT_compound_interest_amount_l1308_130887

theorem compound_interest_amount:
  let SI := (5250 * 4 * 2) / 100
  let CI := 2 * SI
  let P := 420 / 0.21 
  CI = P * ((1 + 0.1) ^ 2 - 1) →
  SI = 210 →
  CI = 420 →
  P = 2000 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_amount_l1308_130887


namespace NUMINAMATH_GPT_clean_per_hour_l1308_130890

-- Definitions of the conditions
def total_pieces : ℕ := 80
def start_time : ℕ := 8
def end_time : ℕ := 12
def total_hours : ℕ := end_time - start_time

-- Proof statement
theorem clean_per_hour : total_pieces / total_hours = 20 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_clean_per_hour_l1308_130890


namespace NUMINAMATH_GPT_exponentiation_property_l1308_130864

variable (a : ℝ)

theorem exponentiation_property : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_GPT_exponentiation_property_l1308_130864


namespace NUMINAMATH_GPT_find_ivans_number_l1308_130837

theorem find_ivans_number :
  ∃ (a b c d e f g h i j k l : ℕ),
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    10 ≤ c ∧ c < 100 ∧
    10 ≤ d ∧ d < 100 ∧
    1000 ≤ e ∧ e < 10000 ∧
    (a * 10^10 + b * 10^8 + c * 10^6 + d * 10^4 + e) = 132040530321 := sorry

end NUMINAMATH_GPT_find_ivans_number_l1308_130837


namespace NUMINAMATH_GPT_chocolate_chips_per_cookie_l1308_130859

theorem chocolate_chips_per_cookie
  (num_batches : ℕ)
  (cookies_per_batch : ℕ)
  (num_people : ℕ)
  (chocolate_chips_per_person : ℕ) :
  (num_batches = 3) →
  (cookies_per_batch = 12) →
  (num_people = 4) →
  (chocolate_chips_per_person = 18) →
  (chocolate_chips_per_person / (num_batches * cookies_per_batch / num_people) = 2) :=
by
  sorry

end NUMINAMATH_GPT_chocolate_chips_per_cookie_l1308_130859


namespace NUMINAMATH_GPT_jim_catches_bob_in_20_minutes_l1308_130817

theorem jim_catches_bob_in_20_minutes
  (bob_speed : ℝ)
  (jim_speed : ℝ)
  (bob_head_start : ℝ)
  (bob_speed_mph : bob_speed = 6)
  (jim_speed_mph : jim_speed = 9)
  (bob_headstart_miles : bob_head_start = 1) :
  ∃ (m : ℝ), m = 20 := 
by
  sorry

end NUMINAMATH_GPT_jim_catches_bob_in_20_minutes_l1308_130817


namespace NUMINAMATH_GPT_unique_solution_l1308_130827

noncomputable def is_solution (f : ℝ → ℝ) : Prop :=
    (∀ x, x ≥ 1 → f x ≤ 2 * (x + 1)) ∧
    (∀ x, x ≥ 1 → f (x + 1) = (1 / x) * ((f x)^2 - 1))

theorem unique_solution (f : ℝ → ℝ) :
    is_solution f → (∀ x, x ≥ 1 → f x = x + 1) := 
sorry

end NUMINAMATH_GPT_unique_solution_l1308_130827


namespace NUMINAMATH_GPT_expected_value_twelve_sided_die_l1308_130878

theorem expected_value_twelve_sided_die : 
  (1 / 12 * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)) = 6.5 := by
  sorry

end NUMINAMATH_GPT_expected_value_twelve_sided_die_l1308_130878


namespace NUMINAMATH_GPT_train_crossing_time_l1308_130852

def train_length : ℕ := 100  -- length of the train in meters
def bridge_length : ℕ := 180  -- length of the bridge in meters
def train_speed_kmph : ℕ := 36  -- speed of the train in kmph

theorem train_crossing_time 
  (TL : ℕ := train_length) 
  (BL : ℕ := bridge_length) 
  (TSK : ℕ := train_speed_kmph) : 
  (TL + BL) / ((TSK * 1000) / 3600) = 28 := by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l1308_130852


namespace NUMINAMATH_GPT_total_amount_paid_l1308_130857

-- Define the quantities and rates as constants
def quantity_grapes : ℕ := 8
def rate_grapes : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 55

-- Define the cost functions
def cost_grapes (q : ℕ) (r : ℕ) : ℕ := q * r
def cost_mangoes (q : ℕ) (r : ℕ) : ℕ := q * r

-- Define the total cost function
def total_cost (c1 : ℕ) (c2 : ℕ) : ℕ := c1 + c2

-- State the proof problem
theorem total_amount_paid :
  total_cost (cost_grapes quantity_grapes rate_grapes) (cost_mangoes quantity_mangoes rate_mangoes) = 1055 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l1308_130857


namespace NUMINAMATH_GPT_initial_capacity_l1308_130815

theorem initial_capacity (x : ℝ) (h1 : 0.9 * x = 198) : x = 220 :=
by
  sorry

end NUMINAMATH_GPT_initial_capacity_l1308_130815


namespace NUMINAMATH_GPT_oven_clock_actual_time_l1308_130876

theorem oven_clock_actual_time :
  ∀ (h : ℕ), (oven_time : h = 10) →
  (oven_gains : ℕ) = 8 →
  (initial_time : ℕ) = 18 →          
  (initial_wall_time : ℕ) = 18 →
  (wall_time_after_one_hour : ℕ) = 19 →
  (oven_time_after_one_hour : ℕ) = 19 + 8/60 →
  ℕ := sorry

end NUMINAMATH_GPT_oven_clock_actual_time_l1308_130876


namespace NUMINAMATH_GPT_min_third_side_of_right_triangle_l1308_130840

theorem min_third_side_of_right_triangle (a b : ℕ) (h1 : a = 4) (h2 : b = 5) :
  ∃ c : ℕ, (min c (4 + 5 - 3) - (4 - 3)) = 3 :=
sorry

end NUMINAMATH_GPT_min_third_side_of_right_triangle_l1308_130840


namespace NUMINAMATH_GPT_shift_down_two_units_l1308_130850

def original_function (x : ℝ) : ℝ := 2 * x + 1

def shifted_function (x : ℝ) : ℝ := original_function x - 2

theorem shift_down_two_units :
  ∀ x : ℝ, shifted_function x = 2 * x - 1 :=
by 
  intros x
  simp [shifted_function, original_function]
  sorry

end NUMINAMATH_GPT_shift_down_two_units_l1308_130850


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l1308_130885

theorem isosceles_triangle_base_length :
  ∀ (p_equilateral p_isosceles side_equilateral : ℕ), 
  p_equilateral = 60 → 
  side_equilateral = p_equilateral / 3 →
  p_isosceles = 55 →
  ∀ (base_isosceles : ℕ),
  side_equilateral + side_equilateral + base_isosceles = p_isosceles →
  base_isosceles = 15 :=
by
  intros p_equilateral p_isosceles side_equilateral h1 h2 h3 base_isosceles h4
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l1308_130885


namespace NUMINAMATH_GPT_min_attempts_to_pair_keys_suitcases_l1308_130819

theorem min_attempts_to_pair_keys_suitcases (n : ℕ) : ∃ p : ℕ, (∀ (keyOpen : Fin n → Fin n), ∃ f : (Fin n × Fin n) → Bool, ∀ (i j : Fin n), i ≠ j → (keyOpen i = j ↔ f (i, j) = tt)) ∧ p = Nat.choose n 2 := by
  sorry

end NUMINAMATH_GPT_min_attempts_to_pair_keys_suitcases_l1308_130819


namespace NUMINAMATH_GPT_parallel_vectors_implies_x_l1308_130891

-- a definition of the vectors a and b
def vector_a : ℝ × ℝ := (2, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (1, x)

-- a definition for vector addition
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- a definition for scalar multiplication
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- a definition for vector subtraction
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

-- the theorem statement
theorem parallel_vectors_implies_x (x : ℝ) (h : 
  vector_add vector_a (vector_b x) = ⟨3, 1 + x⟩ ∧
  vector_sub (scalar_mul 2 vector_a) (vector_b x) = ⟨3, 2 - x⟩ ∧
  ∃ k : ℝ, vector_add vector_a (vector_b x) = scalar_mul k (vector_sub (scalar_mul 2 vector_a) (vector_b x))
  ) : x = 1 / 2 :=
sorry

end NUMINAMATH_GPT_parallel_vectors_implies_x_l1308_130891


namespace NUMINAMATH_GPT_not_less_than_x3_y5_for_x2y_l1308_130833

theorem not_less_than_x3_y5_for_x2y (x y : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) : x^2 * y ≥ x^3 + y^5 :=
sorry

end NUMINAMATH_GPT_not_less_than_x3_y5_for_x2y_l1308_130833


namespace NUMINAMATH_GPT_find_the_number_l1308_130851

theorem find_the_number (n : ℤ) 
    (h : 45 - (28 - (n - (15 - 18))) = 57) :
    n = 37 := 
sorry

end NUMINAMATH_GPT_find_the_number_l1308_130851


namespace NUMINAMATH_GPT_transformed_line_l1308_130809

-- Define the original line equation
def original_line (x y : ℝ) : Prop := (x - 2 * y = 2)

-- Define the transformation
def transformation (x y x' y' : ℝ) : Prop :=
  (x' = x) ∧ (y' = 2 * y)

-- Prove that the transformed line equation holds
theorem transformed_line (x y x' y' : ℝ) (h₁ : original_line x y) (h₂ : transformation x y x' y') :
  x' - y' = 2 :=
sorry

end NUMINAMATH_GPT_transformed_line_l1308_130809


namespace NUMINAMATH_GPT_cornelia_travel_countries_l1308_130893

theorem cornelia_travel_countries (europe south_america asia half_remaining : ℕ) 
  (h1 : europe = 20)
  (h2 : south_america = 10)
  (h3 : asia = 6)
  (h4 : asia = half_remaining / 2) : 
  europe + south_america + half_remaining = 42 :=
by
  sorry

end NUMINAMATH_GPT_cornelia_travel_countries_l1308_130893


namespace NUMINAMATH_GPT_factorize_m_factorize_x_factorize_xy_l1308_130871

theorem factorize_m (m : ℝ) : m^2 + 7 * m - 18 = (m - 2) * (m + 9) := 
sorry

theorem factorize_x (x : ℝ) : x^2 - 2 * x - 8 = (x + 2) * (x - 4) :=
sorry

theorem factorize_xy (x y : ℝ) : (x * y)^2 - 7 * (x * y) + 10 = (x * y - 2) * (x * y - 5) := 
sorry

end NUMINAMATH_GPT_factorize_m_factorize_x_factorize_xy_l1308_130871


namespace NUMINAMATH_GPT_curve_cartesian_equation_max_value_3x_plus_4y_l1308_130849

noncomputable def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ := (rho * Real.cos theta, rho * Real.sin theta)

theorem curve_cartesian_equation :
  (∀ (rho theta : ℝ), rho^2 = 36 / (4 * (Real.cos theta)^2 + 9 * (Real.sin theta)^2)) →
  ∀ x y : ℝ, (∃ theta : ℝ, x = 3 * Real.cos theta ∧ y = 2 * Real.sin theta) → (x^2) / 9 + (y^2) / 4 = 1 :=
sorry

theorem max_value_3x_plus_4y :
  (∀ (rho theta : ℝ), rho^2 = 36 / (4 * (Real.cos theta)^2 + 9 * (Real.sin theta)^2)) →
  ∃ x y : ℝ, (∃ theta : ℝ, x = 3 * Real.cos theta ∧ y = 2 * Real.sin theta) ∧ (∀ ϴ : ℝ, 3 * (3 * Real.cos ϴ) + 4 * (2 * Real.sin ϴ) ≤ Real.sqrt 145) :=
sorry

end NUMINAMATH_GPT_curve_cartesian_equation_max_value_3x_plus_4y_l1308_130849


namespace NUMINAMATH_GPT_towel_bleach_decrease_l1308_130830

theorem towel_bleach_decrease (L B L' B' A A' : ℝ)
    (hB : B' = 0.6 * B)
    (hA : A' = 0.42 * A)
    (hA_def : A = L * B)
    (hA'_def : A' = L' * B') :
    L' = 0.7 * L :=
by
  sorry

end NUMINAMATH_GPT_towel_bleach_decrease_l1308_130830


namespace NUMINAMATH_GPT_smallest_n_for_multiples_of_7_l1308_130879

theorem smallest_n_for_multiples_of_7 (x y : ℤ) (h1 : x ≡ 4 [ZMOD 7]) (h2 : y ≡ 5 [ZMOD 7]) :
  ∃ n : ℕ, 0 < n ∧ (x^2 + x * y + y^2 + n ≡ 0 [ZMOD 7]) ∧ ∀ m : ℕ, 0 < m ∧ (x^2 + x * y + y^2 + m ≡ 0 [ZMOD 7]) → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_multiples_of_7_l1308_130879


namespace NUMINAMATH_GPT_factor_exp_l1308_130820

theorem factor_exp (k : ℕ) : 3^1999 - 3^1998 - 3^1997 + 3^1996 = k * 3^1996 → k = 16 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_factor_exp_l1308_130820


namespace NUMINAMATH_GPT_solve_fraction_equation_l1308_130847

theorem solve_fraction_equation (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 1) :
  (3 / (x - 1) - 2 / x = 0) ↔ x = -2 := by
sorry

end NUMINAMATH_GPT_solve_fraction_equation_l1308_130847


namespace NUMINAMATH_GPT_cone_to_cylinder_water_height_l1308_130863

theorem cone_to_cylinder_water_height :
  let r_cone := 15 -- radius of the cone
  let h_cone := 24 -- height of the cone
  let r_cylinder := 18 -- radius of the cylinder
  let V_cone := (1 / 3: ℝ) * Real.pi * r_cone^2 * h_cone -- volume of the cone
  let h_cylinder := V_cone / (Real.pi * r_cylinder^2) -- height of the water in the cylinder
  h_cylinder = 8.33 := by
  sorry

end NUMINAMATH_GPT_cone_to_cylinder_water_height_l1308_130863


namespace NUMINAMATH_GPT_perimeter_of_polygon_is_15_l1308_130895

-- Definitions for the problem conditions
def side_length_of_square : ℕ := 5
def fraction_of_square_occupied (n : ℕ) : ℚ := 3 / 4

-- Problem statement: Prove that the perimeter of the polygon is 15 units
theorem perimeter_of_polygon_is_15 :
  4 * side_length_of_square * (fraction_of_square_occupied side_length_of_square) = 15 := 
by
  sorry

end NUMINAMATH_GPT_perimeter_of_polygon_is_15_l1308_130895


namespace NUMINAMATH_GPT_rectangle_area_l1308_130853

variable (w l A P : ℝ)
variable (h1 : l = w + 6)
variable (h2 : A = w * l)
variable (h3 : P = 2 * (w + l))
variable (h4 : A = 2 * P)
variable (h5 : w = 3)

theorem rectangle_area
  (w l A P : ℝ)
  (h1 : l = w + 6)
  (h2 : A = w * l)
  (h3 : P = 2 * (w + l))
  (h4 : A = 2 * P)
  (h5 : w = 3) :
  A = 27 := 
sorry

end NUMINAMATH_GPT_rectangle_area_l1308_130853


namespace NUMINAMATH_GPT_apples_for_juice_is_correct_l1308_130802

noncomputable def apples_per_year : ℝ := 8 -- 8 million tons
noncomputable def percentage_mixed : ℝ := 0.30 -- 30%
noncomputable def remaining_apples := apples_per_year * (1 - percentage_mixed) -- Apples after mixed
noncomputable def percentage_for_juice : ℝ := 0.60 -- 60%
noncomputable def apples_for_juice := remaining_apples * percentage_for_juice -- Apples for juice

theorem apples_for_juice_is_correct :
  apples_for_juice = 3.36 :=
by
  sorry

end NUMINAMATH_GPT_apples_for_juice_is_correct_l1308_130802


namespace NUMINAMATH_GPT_rationalize_value_of_a2_minus_2a_value_of_2a3_minus_4a2_minus_1_l1308_130861

variable (a : ℂ)

theorem rationalize (h : a = 1 / (Real.sqrt 2 - 1)) : a = Real.sqrt 2 + 1 := by
  sorry

theorem value_of_a2_minus_2a (h : a = Real.sqrt 2 + 1) : a ^ 2 - 2 * a = 1 := by
  sorry

theorem value_of_2a3_minus_4a2_minus_1 (h : a = Real.sqrt 2 + 1) : 2 * a ^ 3 - 4 * a ^ 2 - 1 = 2 * Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_GPT_rationalize_value_of_a2_minus_2a_value_of_2a3_minus_4a2_minus_1_l1308_130861


namespace NUMINAMATH_GPT_special_number_is_square_l1308_130892

-- Define the special number format
def special_number (n : ℕ) : ℕ :=
  3 * (10^n - 1)/9 + 4

theorem special_number_is_square (n : ℕ) :
  ∃ k : ℕ, k * k = special_number n := by
  sorry

end NUMINAMATH_GPT_special_number_is_square_l1308_130892


namespace NUMINAMATH_GPT_find_original_price_l1308_130856

variable (P : ℝ)

def final_price (discounted_price : ℝ) (discount_rate : ℝ) (original_price : ℝ) : Prop :=
  discounted_price = (1 - discount_rate) * original_price

theorem find_original_price (h1 : final_price 120 0.4 P) : P = 200 := 
by
  sorry

end NUMINAMATH_GPT_find_original_price_l1308_130856


namespace NUMINAMATH_GPT_magnitude_correct_l1308_130806

open Real

noncomputable def magnitude_of_vector_addition
  (a b : ℝ × ℝ)
  (theta : ℝ)
  (ha : a = (1, 1))
  (hb : ‖b‖ = 2)
  (h_angle : theta = π / 4) : ℝ :=
  ‖3 • a + b‖

theorem magnitude_correct 
  (a b : ℝ × ℝ)
  (theta : ℝ)
  (ha : a = (1, 1))
  (hb : ‖b‖ = 2)
  (h_angle : theta = π / 4) :
  magnitude_of_vector_addition a b theta ha hb h_angle = sqrt 34 :=
sorry

end NUMINAMATH_GPT_magnitude_correct_l1308_130806


namespace NUMINAMATH_GPT_first_two_cards_black_prob_l1308_130860

noncomputable def probability_first_two_black : ℚ :=
  let total_cards := 52
  let black_cards := 26
  let first_draw_prob := black_cards / total_cards
  let second_draw_prob := (black_cards - 1) / (total_cards - 1)
  first_draw_prob * second_draw_prob

theorem first_two_cards_black_prob :
  probability_first_two_black = 25 / 102 :=
by
  sorry

end NUMINAMATH_GPT_first_two_cards_black_prob_l1308_130860


namespace NUMINAMATH_GPT_Watson_class_student_count_l1308_130898

def num_kindergartners : ℕ := 14
def num_first_graders : ℕ := 24
def num_second_graders : ℕ := 4

def total_students : ℕ := num_kindergartners + num_first_graders + num_second_graders

theorem Watson_class_student_count : total_students = 42 := 
by
    sorry

end NUMINAMATH_GPT_Watson_class_student_count_l1308_130898


namespace NUMINAMATH_GPT_intervals_of_monotonicity_of_f_l1308_130838

noncomputable def f (a b c d : ℝ) (x : ℝ) := a * x^3 + b * x^2 + c * x + d

theorem intervals_of_monotonicity_of_f (a b c d : ℝ)
  (h1 : ∃ P : ℝ × ℝ, P.1 = 0 ∧ d = P.2 ∧ (12 * P.1 - P.2 - 4 = 0))
  (h2 : ∃ x : ℝ, x = 2 ∧ (f a b c d x = 0) ∧ (∃ x : ℝ, x = 0 ∧ (3 * a * x^2 + 2 * b * x + c = 12))) 
  : ( ∃ a b c d : ℝ , (f a b c d) = (2 * x^3 - 9 * x^2 + 12 * x -4)) := 
  sorry

end NUMINAMATH_GPT_intervals_of_monotonicity_of_f_l1308_130838


namespace NUMINAMATH_GPT_unsatisfactory_tests_l1308_130834

theorem unsatisfactory_tests {n k : ℕ} (h1 : n < 50) 
  (h2 : n % 7 = 0) 
  (h3 : n % 3 = 0) 
  (h4 : n % 2 = 0)
  (h5 : n = 7 * (n / 7) + 3 * (n / 3) + 2 * (n / 2) + k) : 
  k = 1 := 
by 
  sorry

end NUMINAMATH_GPT_unsatisfactory_tests_l1308_130834


namespace NUMINAMATH_GPT_avg_children_in_families_with_children_l1308_130836

noncomputable def avg_children_with_children (total_families : ℕ) (avg_children : ℝ) (childless_families : ℕ) : ℝ :=
  let total_children := total_families * avg_children
  let families_with_children := total_families - childless_families
  total_children / families_with_children

theorem avg_children_in_families_with_children :
  avg_children_with_children 15 3 3 = 3.8 := by
  sorry

end NUMINAMATH_GPT_avg_children_in_families_with_children_l1308_130836


namespace NUMINAMATH_GPT_birgit_time_to_travel_8km_l1308_130880

theorem birgit_time_to_travel_8km
  (total_hours : ℝ)
  (total_distance : ℝ)
  (speed_difference : ℝ)
  (distance_to_travel : ℝ)
  (total_minutes := total_hours * 60)
  (average_speed := total_minutes / total_distance)
  (birgit_speed := average_speed - speed_difference) :
  total_hours = 3.5 →
  total_distance = 21 →
  speed_difference = 4 →
  distance_to_travel = 8 →
  (birgit_speed * distance_to_travel) = 48 :=
by
  sorry

end NUMINAMATH_GPT_birgit_time_to_travel_8km_l1308_130880


namespace NUMINAMATH_GPT_find_f_neg_a_l1308_130803

noncomputable def f (x : ℝ) : ℝ := x^3 * Real.cos x + 1

variable (a : ℝ)

-- Given condition
axiom h_fa : f a = 11

-- Statement to prove
theorem find_f_neg_a : f (-a) = -9 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg_a_l1308_130803


namespace NUMINAMATH_GPT_find_point_A_equidistant_l1308_130801

theorem find_point_A_equidistant :
  ∃ (x : ℝ), (∃ A : ℝ × ℝ × ℝ, A = (x, 0, 0)) ∧
              (∃ B : ℝ × ℝ × ℝ, B = (4, 0, 5)) ∧
              (∃ C : ℝ × ℝ × ℝ, C = (5, 4, 2)) ∧
              (dist (x, 0, 0) (4, 0, 5) = dist (x, 0, 0) (5, 4, 2)) ∧ 
              (x = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_point_A_equidistant_l1308_130801


namespace NUMINAMATH_GPT_function_properties_l1308_130899

-- Define the function and conditions
def f : ℝ → ℝ := sorry

axiom condition1 (x : ℝ) : f (10 + x) = f (10 - x)
axiom condition2 (x : ℝ) : f (20 - x) = -f (20 + x)

-- Lean statement to encapsulate the question and expected result
theorem function_properties (x : ℝ) : (f (-x) = -f x) ∧ (f (x + 40) = f x) :=
sorry

end NUMINAMATH_GPT_function_properties_l1308_130899


namespace NUMINAMATH_GPT_find_PA_PB_sum_2sqrt6_l1308_130831

noncomputable def polar_equation (ρ θ : ℝ) : Prop :=
  ρ - 2 * Real.cos θ - 6 * Real.sin θ + 1 / ρ = 0

noncomputable def parametric_line (t x y : ℝ) : Prop :=
  x = 3 + 1 / 2 * t ∧ y = 3 + Real.sqrt 3 / 2 * t

def point_P (x y : ℝ) : Prop :=
  x = 3 ∧ y = 3

theorem find_PA_PB_sum_2sqrt6 :
  (∃ ρ θ t₁ t₂, polar_equation ρ θ ∧ parametric_line t₁ 3 3 ∧ parametric_line t₂ 3 3 ∧
  point_P 3 3 ∧ |t₁| + |t₂| = 2 * Real.sqrt 6) := sorry

end NUMINAMATH_GPT_find_PA_PB_sum_2sqrt6_l1308_130831


namespace NUMINAMATH_GPT_pedro_plums_l1308_130823

theorem pedro_plums :
  ∃ P Q : ℕ, P + Q = 32 ∧ 2 * P + Q = 52 ∧ P = 20 :=
by
  sorry

end NUMINAMATH_GPT_pedro_plums_l1308_130823


namespace NUMINAMATH_GPT_max_value_2ab_2bc_2cd_2da_l1308_130839

theorem max_value_2ab_2bc_2cd_2da {a b c d : ℕ} :
  (a = 2 ∨ a = 3 ∨ a = 5 ∨ a = 7) ∧
  (b = 2 ∨ b = 3 ∨ b = 5 ∨ b = 7) ∧
  (c = 2 ∨ c = 3 ∨ c = 5 ∨ c = 7) ∧
  (d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7) ∧
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧
  (b ≠ c) ∧ (b ≠ d) ∧
  (c ≠ d)
  → 2 * (a * b + b * c + c * d + d * a) ≤ 144 :=
by
  sorry

end NUMINAMATH_GPT_max_value_2ab_2bc_2cd_2da_l1308_130839


namespace NUMINAMATH_GPT_average_price_blankets_l1308_130841

theorem average_price_blankets :
  let cost_blankets1 := 3 * 100
  let cost_blankets2 := 5 * 150
  let cost_blankets3 := 550
  let total_cost := cost_blankets1 + cost_blankets2 + cost_blankets3
  let total_blankets := 3 + 5 + 2
  total_cost / total_blankets = 160 :=
by
  sorry

end NUMINAMATH_GPT_average_price_blankets_l1308_130841


namespace NUMINAMATH_GPT_simplify_and_compute_l1308_130869

theorem simplify_and_compute (x : ℝ) (h : x = 4) : (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end NUMINAMATH_GPT_simplify_and_compute_l1308_130869


namespace NUMINAMATH_GPT_find_non_integer_solution_l1308_130818

noncomputable def q (x y : ℝ) (b : Fin 10 → ℝ) : ℝ :=
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 +
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3

theorem find_non_integer_solution (b : Fin 10 → ℝ)
  (h0 : q 0 0 b = 0)
  (h1 : q 1 0 b = 0)
  (h2 : q (-1) 0 b = 0)
  (h3 : q 0 1 b = 0)
  (h4 : q 0 (-1) b = 0)
  (h5 : q 1 1 b = 0)
  (h6 : q 1 (-1) b = 0)
  (h7 : q (-1) 1 b = 0)
  (h8 : q (-1) (-1) b = 0) :
  ∃ r s : ℝ, q r s b = 0 ∧ ¬ (∃ n : ℤ, r = n) ∧ ¬ (∃ n : ℤ, s = n) :=
sorry

end NUMINAMATH_GPT_find_non_integer_solution_l1308_130818


namespace NUMINAMATH_GPT_correct_operation_l1308_130846

theorem correct_operation :
  (2 * a - a ≠ 2) ∧ ((a - 1) * (a - 1) ≠ a ^ 2 - 1) ∧ (a ^ 6 / a ^ 3 ≠ a ^ 2) ∧ ((-2 * a ^ 3) ^ 2 = 4 * a ^ 6) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l1308_130846


namespace NUMINAMATH_GPT_germination_rate_sunflower_l1308_130828

variable (s_d s_s f_d f_s p : ℕ) (g_d g_f : ℚ)

-- Define the conditions
def conditions :=
  s_d = 25 ∧ s_s = 25 ∧ g_d = 0.60 ∧ g_f = 0.80 ∧ p = 28 ∧ f_d = 12 ∧ f_s = 16

-- Define the statement to be proved
theorem germination_rate_sunflower (h : conditions s_d s_s f_d f_s p g_d g_f) : 
  (f_s / (g_f * (s_s : ℚ))) > 0.0 ∧ (f_s / (g_f * (s_s : ℚ)) * 100) = 80 := 
by
  sorry

end NUMINAMATH_GPT_germination_rate_sunflower_l1308_130828


namespace NUMINAMATH_GPT_quadrilateral_area_l1308_130824

theorem quadrilateral_area (d h1 h2 : ℝ) (hd : d = 40) (hh1 : h1 = 9) (hh2 : h2 = 6) :
  1 / 2 * d * h1 + 1 / 2 * d * h2 = 300 := 
by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_l1308_130824


namespace NUMINAMATH_GPT_product_xyz_is_minus_one_l1308_130804

-- Definitions of the variables and equations
variables (x y z : ℝ)

-- Assumptions based on the given conditions
def condition1 : Prop := x + (1 / y) = 2
def condition2 : Prop := y + (1 / z) = 2
def condition3 : Prop := z + (1 / x) = 2

-- The theorem stating the conclusion to be proved
theorem product_xyz_is_minus_one (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 z x) : x * y * z = -1 :=
by sorry

end NUMINAMATH_GPT_product_xyz_is_minus_one_l1308_130804


namespace NUMINAMATH_GPT_equilateral_triangle_roots_l1308_130842

theorem equilateral_triangle_roots (p q : ℂ) (z1 z2 : ℂ) (h1 : z2 = Complex.exp (2 * Real.pi * Complex.I / 3) * z1)
  (h2 : 0 + p * z1 + q = 0) (h3 : p = -z1 - z2) (h4 : q = z1 * z2) : (p^2 / q) = 1 :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_roots_l1308_130842


namespace NUMINAMATH_GPT_paving_cost_l1308_130897

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate : ℝ := 1000
def area : ℝ := length * width
def cost : ℝ := area * rate

theorem paving_cost :
  cost = 20625 := by sorry

end NUMINAMATH_GPT_paving_cost_l1308_130897


namespace NUMINAMATH_GPT_probability_first_ge_second_l1308_130825

-- Define the number of faces
def faces : ℕ := 10

-- Define the total number of outcomes excluding the duplicates
def total_outcomes : ℕ := faces * faces - faces

-- Calculate the number of favorable outcomes
def favorable_outcomes : ℕ := 10 + 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1

-- Define the probability as a fraction
def probability : ℚ := favorable_outcomes / total_outcomes

-- The statement we want to prove
theorem probability_first_ge_second :
  probability = 11 / 18 :=
sorry

end NUMINAMATH_GPT_probability_first_ge_second_l1308_130825


namespace NUMINAMATH_GPT_train_length_l1308_130844

theorem train_length (v : ℝ) (t : ℝ) (conversion_factor : ℝ) : v = 45 → t = 16 → conversion_factor = 1000 / 3600 → (v * (conversion_factor) * t) = 200 :=
  by
  intros hv ht hcf
  rw [hv, ht, hcf]
  -- Proof steps skipped
  sorry

end NUMINAMATH_GPT_train_length_l1308_130844
