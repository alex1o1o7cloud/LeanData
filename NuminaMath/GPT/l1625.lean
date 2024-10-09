import Mathlib

namespace singer_arrangements_l1625_162557

theorem singer_arrangements (s1 s2 : Type) [Fintype s1] [Fintype s2] 
  (h1 : Fintype.card s1 = 4) (h2 : Fintype.card s2 = 1) :
  ∃ n : ℕ, n = 18 :=
by
  sorry

end singer_arrangements_l1625_162557


namespace polygon_sides_l1625_162518

theorem polygon_sides (n : ℕ) (h1 : n ≥ 3)
  (h2 : ∃ (theta theta' : ℝ), theta = (n - 2) * 180 / n ∧ theta' = (n + 7) * 180 / (n + 9) ∧ theta' = theta + 9) : n = 15 :=
sorry

end polygon_sides_l1625_162518


namespace stewart_farm_food_l1625_162546

variable (S H : ℕ) (HorseFoodPerHorsePerDay : Nat) (TotalSheep : Nat)

theorem stewart_farm_food (ratio_sheep_horses : 6 * H = 7 * S) 
  (total_sheep_count : S = 48) 
  (horse_food : HorseFoodPerHorsePerDay = 230) : 
  HorseFoodPerHorsePerDay * (7 * 48 / 6) = 12880 :=
by
  sorry

end stewart_farm_food_l1625_162546


namespace x_100_equals_2_power_397_l1625_162502

-- Define the sequences
noncomputable def a_n (n : ℕ) : ℕ := 2^(n-1)
noncomputable def b_n (n : ℕ) : ℕ := 5*n - 3

-- Define the merged sequence x_n
noncomputable def x_n (k : ℕ) : ℕ := 2^(4*k - 3)

-- Prove x_100 is 2^397
theorem x_100_equals_2_power_397 : x_n 100 = 2^397 := by
  unfold x_n
  show 2^(4*100 - 3) = 2^397
  rfl

end x_100_equals_2_power_397_l1625_162502


namespace horner_method_v1_l1625_162536

def polynomial (x : ℝ) : ℝ := 4 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

theorem horner_method_v1 (x : ℝ) (h : x = 5) : 
  ((4 * x + 2) * x + 3.5) = 22 := by
  rw [h]
  norm_num
  sorry

end horner_method_v1_l1625_162536


namespace find_b_l1625_162532

-- Given conditions
def p (x : ℝ) : ℝ := 2 * x - 7
def q (x : ℝ) (b : ℝ) : ℝ := 3 * x - b

-- Assertion we need to prove
theorem find_b (b : ℝ) (h : p (q 3 b) = 3) : b = 4 := 
by
  sorry

end find_b_l1625_162532


namespace students_taking_neither_580_l1625_162511

noncomputable def numberOfStudentsTakingNeither (total students_m students_a students_d students_ma students_md students_ad students_mad : ℕ) : ℕ :=
  let total_taking_at_least_one := (students_m + students_a + students_d) 
                                - (students_ma + students_md + students_ad) 
                                + students_mad
  total - total_taking_at_least_one

theorem students_taking_neither_580 :
  let total := 800
  let students_m := 140
  let students_a := 90
  let students_d := 75
  let students_ma := 50
  let students_md := 30
  let students_ad := 25
  let students_mad := 20
  numberOfStudentsTakingNeither total students_m students_a students_d students_ma students_md students_ad students_mad = 580 :=
by
  sorry

end students_taking_neither_580_l1625_162511


namespace range_of_a_l1625_162531

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ((1 - a) * x > 1 - a) → (x < 1)) → (1 < a) :=
by sorry

end range_of_a_l1625_162531


namespace total_lunch_bill_l1625_162533

def cost_of_hotdog : ℝ := 5.36
def cost_of_salad : ℝ := 5.10

theorem total_lunch_bill : cost_of_hotdog + cost_of_salad = 10.46 := 
by
  sorry

end total_lunch_bill_l1625_162533


namespace non_allergic_children_l1625_162590

theorem non_allergic_children (T : ℕ) (h1 : T / 2 = n) (h2 : ∀ m : ℕ, 10 = m) (h3 : ∀ k : ℕ, 10 = k) :
  10 = 10 :=
by
  sorry

end non_allergic_children_l1625_162590


namespace p_over_q_at_neg1_l1625_162585

-- Definitions of p(x) and q(x) based on given conditions
noncomputable def q (x : ℝ) := (x + 3) * (x - 2)
noncomputable def p (x : ℝ) := 2 * x

-- Define the main function y = p(x) / q(x)
noncomputable def y (x : ℝ) := p x / q x

-- Statement to prove the value of p(-1) / q(-1)
theorem p_over_q_at_neg1 : y (-1) = (1 : ℝ) / 3 :=
by
  sorry

end p_over_q_at_neg1_l1625_162585


namespace area_of_original_triangle_l1625_162534

theorem area_of_original_triangle (a : Real) (S_intuitive : Real) : 
  a = 2 -> S_intuitive = (Real.sqrt 3) -> (S_intuitive / (Real.sqrt 2 / 4)) = 2 * Real.sqrt 6 := 
by
  sorry

end area_of_original_triangle_l1625_162534


namespace handshake_problem_l1625_162551

theorem handshake_problem (x y : ℕ) 
  (H : (x * (x - 1)) / 2 + y = 159) : 
  x = 18 ∧ y = 6 := 
sorry

end handshake_problem_l1625_162551


namespace sum_of_solutions_l1625_162542

theorem sum_of_solutions (x : ℝ) :
  (∀ x, x^2 - 17 * x + 54 = 0) → 
  (∃ r s : ℝ, r ≠ s ∧ r + s = 17) :=
by
  sorry

end sum_of_solutions_l1625_162542


namespace global_phone_company_customers_l1625_162565

theorem global_phone_company_customers :
  (total_customers = 25000) →
  (us_percentage = 0.20) →
  (canada_percentage = 0.12) →
  (australia_percentage = 0.15) →
  (uk_percentage = 0.08) →
  (india_percentage = 0.05) →
  (us_customers = total_customers * us_percentage) →
  (canada_customers = total_customers * canada_percentage) →
  (australia_customers = total_customers * australia_percentage) →
  (uk_customers = total_customers * uk_percentage) →
  (india_customers = total_customers * india_percentage) →
  (mentioned_countries_customers = us_customers + canada_customers + australia_customers + uk_customers + india_customers) →
  (other_countries_customers = total_customers - mentioned_countries_customers) →
  (other_countries_customers = 10000) ∧ (us_customers / other_countries_customers = 1 / 2) :=
by
  -- The further proof steps would go here if needed
  sorry

end global_phone_company_customers_l1625_162565


namespace total_cost_is_734_l1625_162512

-- Define the cost of each ice cream flavor
def cost_vanilla : ℕ := 99
def cost_chocolate : ℕ := 129
def cost_strawberry : ℕ := 149

-- Define the amount of each flavor Mrs. Hilt buys
def num_vanilla : ℕ := 2
def num_chocolate : ℕ := 3
def num_strawberry : ℕ := 1

-- Calculate the total cost in cents
def total_cost : ℕ :=
  (num_vanilla * cost_vanilla) +
  (num_chocolate * cost_chocolate) +
  (num_strawberry * cost_strawberry)

-- Statement of the proof problem
theorem total_cost_is_734 : total_cost = 734 :=
by
  sorry

end total_cost_is_734_l1625_162512


namespace no_second_quadrant_l1625_162524

theorem no_second_quadrant (k : ℝ) :
  (∀ x : ℝ, (x < 0 → 3 * x + k - 2 ≤ 0)) → k ≤ 2 :=
by
  intro h
  sorry

end no_second_quadrant_l1625_162524


namespace find_a_l1625_162501

noncomputable def f (x a : ℝ) : ℝ := x * (Real.exp x + a * Real.exp (-x))

theorem find_a (a : ℝ) : (∀ x : ℝ, f x a = -f (-x) a) → a = 1 :=
by
  sorry

end find_a_l1625_162501


namespace graph_single_point_c_eq_7_l1625_162573

theorem graph_single_point_c_eq_7 (x y : ℝ) (c : ℝ) :
  (∃ p : ℝ × ℝ, ∀ x y : ℝ, 3 * x^2 + 4 * y^2 + 6 * x - 8 * y + c = 0 ↔ (x, y) = p) →
  c = 7 :=
by
  sorry

end graph_single_point_c_eq_7_l1625_162573


namespace siblings_gmat_scores_l1625_162574

-- Define the problem conditions
variables (x y z : ℝ)

theorem siblings_gmat_scores (h1 : x - y = 1/3) (h2 : z = (x + y) / 2) : 
  y = x - 1/3 ∧ z = x - 1/6 :=
by
  sorry

end siblings_gmat_scores_l1625_162574


namespace cos_A_value_l1625_162505

theorem cos_A_value (A B C : ℝ) 
  (A_internal : A + B + C = Real.pi) 
  (cos_B : Real.cos B = 1 / 2)
  (sin_C : Real.sin C = 3 / 5) : 
  Real.cos A = (3 * Real.sqrt 3 - 4) / 10 := 
by
  sorry

end cos_A_value_l1625_162505


namespace least_integer_of_sum_in_ratio_l1625_162562

theorem least_integer_of_sum_in_ratio (a b c : ℕ) (h1 : a + b + c = 90) (h2 : a * 3 = b * 2) (h3 : a * 5 = c * 2) : a = 18 :=
by
  sorry

end least_integer_of_sum_in_ratio_l1625_162562


namespace min_side_length_of_square_l1625_162566

theorem min_side_length_of_square (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ s : ℝ, s = 
    if a < (Real.sqrt 2 + 1) * b then 
      a 
    else 
      (Real.sqrt 2 / 2) * (a + b) := 
sorry

end min_side_length_of_square_l1625_162566


namespace calf_probability_l1625_162564

theorem calf_probability 
  (P_B1 : ℝ := 0.6)  -- Proportion of calves from the first farm
  (P_B2 : ℝ := 0.3)  -- Proportion of calves from the second farm
  (P_B3 : ℝ := 0.1)  -- Proportion of calves from the third farm
  (P_B1_A : ℝ := 0.15)  -- Conditional probability of a calf weighing more than 300 kg given it is from the first farm
  (P_B2_A : ℝ := 0.25)  -- Conditional probability of a calf weighing more than 300 kg given it is from the second farm
  (P_B3_A : ℝ := 0.35)  -- Conditional probability of a calf weighing more than 300 kg given it is from the third farm)
  (P_A : ℝ := P_B1 * P_B1_A + P_B2 * P_B2_A + P_B3 * P_B3_A) : 
  P_B3 * P_B3_A / P_A = 0.175 := 
by
  sorry

end calf_probability_l1625_162564


namespace value_of_f_at_3_l1625_162529

def f (x : ℚ) : ℚ := (2 * x + 3) / (4 * x - 5)

theorem value_of_f_at_3 : f 3 = 9 / 7 := by
  sorry

end value_of_f_at_3_l1625_162529


namespace cubic_roots_identity_l1625_162597

theorem cubic_roots_identity (x1 x2 p q : ℝ) 
  (h1 : x1^2 + p * x1 + q = 0) 
  (h2 : x2^2 + p * x2 + q = 0) :
  (x1^3 + x2^3 = 3 * p * q - p^3) ∧ 
  (x1^3 - x2^3 = (p^2 - q) * Real.sqrt (p^2 - 4 * q) ∨ 
   x1^3 - x2^3 = -(p^2 - q) * Real.sqrt (p^2 - 4 * q)) :=
by
  sorry

end cubic_roots_identity_l1625_162597


namespace swimming_speed_l1625_162509

theorem swimming_speed (s v : ℝ) (h_s : s = 4) (h_time : 1 / (v - s) = 2 * (1 / (v + s))) : v = 12 := 
by
  sorry

end swimming_speed_l1625_162509


namespace james_savings_l1625_162537

-- Define the conditions
def cost_vest : ℝ := 250
def weight_plates_pounds : ℕ := 200
def cost_per_pound : ℝ := 1.2
def original_weight_vest_cost : ℝ := 700
def discount : ℝ := 100

-- Define the derived quantities based on conditions
def cost_weight_plates : ℝ := weight_plates_pounds * cost_per_pound
def total_cost_setup : ℝ := cost_vest + cost_weight_plates
def discounted_weight_vest_cost : ℝ := original_weight_vest_cost - discount
def savings : ℝ := discounted_weight_vest_cost - total_cost_setup

-- The statement to prove the savings
theorem james_savings : savings = 110 := by
  sorry

end james_savings_l1625_162537


namespace concave_number_count_l1625_162588

def is_concave_number (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  n >= 100 ∧ n < 1000 ∧ tens < hundreds ∧ tens < units

theorem concave_number_count : ∃ n : ℕ, 
  (∀ m < 1000, is_concave_number m → m = n) ∧ n = 240 :=
by
  sorry

end concave_number_count_l1625_162588


namespace cups_per_serving_l1625_162553

theorem cups_per_serving (total_cups servings : ℝ) (h1 : total_cups = 36) (h2 : servings = 18.0) :
  total_cups / servings = 2 :=
by 
  sorry

end cups_per_serving_l1625_162553


namespace rectangle_area_l1625_162523

theorem rectangle_area (b l : ℕ) (P : ℕ) (h1 : l = 3 * b) (h2 : P = 64) (h3 : P = 2 * (l + b)) :
  l * b = 192 :=
by
  sorry

end rectangle_area_l1625_162523


namespace rectangle_area_error_percent_l1625_162578

theorem rectangle_area_error_percent 
  (L W : ℝ)
  (hL: L > 0)
  (hW: W > 0) :
  let original_area := L * W
  let measured_length := 1.06 * L
  let measured_width := 0.95 * W
  let measured_area := measured_length * measured_width
  let error := measured_area - original_area
  let error_percent := (error / original_area) * 100
  error_percent = 0.7 := by
  let original_area := L * W
  let measured_length := 1.06 * L
  let measured_width := 0.95 * W
  let measured_area := measured_length * measured_width
  let error := measured_area - original_area
  let error_percent := (error / original_area) * 100
  sorry

end rectangle_area_error_percent_l1625_162578


namespace rhombus_region_area_l1625_162539

noncomputable def region_area (s : ℝ) (angleB : ℝ) : ℝ :=
  let h := (s / 2) * (Real.sin (angleB / 2))
  let area_triangle := (1 / 2) * (s / 2) * h
  3 * area_triangle

theorem rhombus_region_area : region_area 3 150 = 0.87345 := by
    sorry

end rhombus_region_area_l1625_162539


namespace maximize_profit_l1625_162582

-- Definitions from the conditions
def cost_price : ℝ := 16
def initial_selling_price : ℝ := 20
def initial_sales_volume : ℝ := 80
def price_decrease_per_step : ℝ := 0.5
def sales_increase_per_step : ℝ := 20

def functional_relationship (x : ℝ) : ℝ := -40 * x + 880

-- The main theorem we need to prove
theorem maximize_profit :
  (∀ x, 16 ≤ x → x ≤ 20 → functional_relationship x = -40 * x + 880) ∧
  (∃ x, 16 ≤ x ∧ x ≤ 20 ∧ (∀ y, 16 ≤ y → y ≤ 20 → 
    ((-40 * x + 880) * (x - cost_price) ≥ (-40 * y + 880) * (y - cost_price)) ∧
    (-40 * x + 880) * (x - cost_price) = 360 ∧ x = 19)) :=
by
  sorry

end maximize_profit_l1625_162582


namespace paint_needed_for_new_statues_l1625_162567

-- Conditions
def pint_for_original : ℕ := 1
def original_height : ℕ := 8
def num_statues : ℕ := 320
def new_height : ℕ := 2
def scale_ratio : ℚ := (new_height : ℚ) / (original_height : ℚ)
def area_ratio : ℚ := scale_ratio ^ 2

-- Correct Answer
def total_paint_needed : ℕ := 20

-- Theorem to be proved
theorem paint_needed_for_new_statues :
  pint_for_original * num_statues * area_ratio = total_paint_needed := 
by
  sorry

end paint_needed_for_new_statues_l1625_162567


namespace number_of_ways_to_assign_roles_l1625_162586

theorem number_of_ways_to_assign_roles :
  let men := 6
  let women := 5
  let male_roles := 3
  let female_roles := 2
  let either_gender_roles := 1
  let total_men := men - male_roles
  let total_women := women - female_roles
  (men.choose male_roles) * (women.choose female_roles) * (total_men + total_women).choose either_gender_roles = 14400 := by 
sorry

end number_of_ways_to_assign_roles_l1625_162586


namespace blocks_used_for_fenced_area_l1625_162591

theorem blocks_used_for_fenced_area
  (initial_blocks : ℕ) (building_blocks : ℕ) (farmhouse_blocks : ℕ) (remaining_blocks : ℕ) :
  initial_blocks = 344 →
  building_blocks = 80 →
  farmhouse_blocks = 123 →
  remaining_blocks = 84 →
  initial_blocks - building_blocks - farmhouse_blocks - remaining_blocks = 57 :=
by
  intros h1 h2 h3 h4
  sorry

end blocks_used_for_fenced_area_l1625_162591


namespace nikka_us_stamp_percentage_l1625_162504

/-- 
Prove that 20% of Nikka's stamp collection are US stamps given the following conditions:
1. Nikka has a total of 100 stamps.
2. 35 of those stamps are Chinese.
3. 45 of those stamps are Japanese.
-/
theorem nikka_us_stamp_percentage
  (total_stamps : ℕ)
  (chinese_stamps : ℕ)
  (japanese_stamps : ℕ)
  (h1 : total_stamps = 100)
  (h2 : chinese_stamps = 35)
  (h3 : japanese_stamps = 45) :
  ((total_stamps - (chinese_stamps + japanese_stamps)) / total_stamps) * 100 = 20 := 
by
  sorry

end nikka_us_stamp_percentage_l1625_162504


namespace gain_percentage_l1625_162515

theorem gain_percentage (CP SP : ℕ) (h_sell : SP = 10 * CP) : 
  (10 * CP / 25 * CP) * 100 = 40 := by
  sorry

end gain_percentage_l1625_162515


namespace jellybean_ratio_l1625_162592

theorem jellybean_ratio (L Tino Arnold : ℕ) (h1 : Tino = L + 24) (h2 : Arnold = 5) (h3 : Tino = 34) :
  Arnold / L = 1 / 2 :=
by
  sorry

end jellybean_ratio_l1625_162592


namespace basic_computer_price_l1625_162508

theorem basic_computer_price (C P : ℝ)
  (h1 : C + P = 2500)
  (h2 : P = (C + 500 + P) / 3)
  : C = 1500 :=
sorry

end basic_computer_price_l1625_162508


namespace chord_constant_l1625_162587

theorem chord_constant (
    d : ℝ
) : (∃ t : ℝ, (∀ A B : ℝ × ℝ,
    A.2 = A.1^3 ∧ B.2 = B.1^3 ∧ d = 1/2 ∧
    (C : ℝ × ℝ) = (0, d) ∧ 
    (∀ (AC BC: ℝ),
        AC = dist A C ∧
        BC = dist B C ∧
        t = (1 / (AC^2) + 1 / (BC^2))
    )) → t = 4) := 
sorry

end chord_constant_l1625_162587


namespace number_of_valid_numbers_l1625_162522

-- Define a function that checks if a number is composed of digits from the set {1, 2, 3}
def composed_of_123 (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ n.digits 10 → d = 1 ∨ d = 2 ∨ d = 3

-- Define a predicate for a number being less than 200,000
def less_than_200000 (n : ℕ) : Prop := n < 200000

-- Define a predicate for a number being divisible by 3
def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- The main theorem statement
theorem number_of_valid_numbers : ∃ (count : ℕ), count = 202 ∧ 
  (∀ (n : ℕ), less_than_200000 n → composed_of_123 n → divisible_by_3 n → n < count) :=
sorry

end number_of_valid_numbers_l1625_162522


namespace sum_gcd_lcm_is_244_l1625_162570

-- Definitions of the constants
def a : ℕ := 12
def b : ℕ := 80

-- Main theorem statement
theorem sum_gcd_lcm_is_244 : Nat.gcd a b + Nat.lcm a b = 244 := by
  sorry

end sum_gcd_lcm_is_244_l1625_162570


namespace first_shaded_complete_cycle_seat_190_l1625_162568

theorem first_shaded_complete_cycle_seat_190 : 
  ∀ (n : ℕ), (n ≥ 1) → 
  ∃ m : ℕ, 
    ((m ≥ n) ∧ 
    (∀ i : ℕ, (1 ≤ i ∧ i ≤ 12) → 
    ∃ k : ℕ, (k ≤ m ∧ (k * (k + 1) / 2) % 12 = (i - 1) % 12))) ↔ 
  ∃ m : ℕ, (m = 19 ∧ 190 = (m * (m + 1)) / 2) :=
by
  sorry

end first_shaded_complete_cycle_seat_190_l1625_162568


namespace pile_splitting_l1625_162540

theorem pile_splitting (single_stone_piles : ℕ) :
  ∃ (final_heap_size : ℕ), 
    (∀ heap_size ≤ single_stone_piles, heap_size > 0 → (heap_size * 2) ≥ heap_size) ∧ (final_heap_size = single_stone_piles) :=
by
  sorry

end pile_splitting_l1625_162540


namespace average_of_first_20_even_numbers_not_divisible_by_3_or_5_l1625_162507

def first_20_valid_even_numbers : List ℕ :=
  [2, 4, 8, 14, 16, 22, 26, 28, 32, 34, 38, 44, 46, 52, 56, 58, 62, 64, 68, 74]

-- Check the sum of these numbers
def sum_first_20_valid_even_numbers : ℕ :=
  first_20_valid_even_numbers.sum

-- Define average calculation
def average_first_20_valid_even_numbers : ℕ :=
  sum_first_20_valid_even_numbers / 20

theorem average_of_first_20_even_numbers_not_divisible_by_3_or_5 :
  average_first_20_valid_even_numbers = 35 :=
by
  sorry

end average_of_first_20_even_numbers_not_divisible_by_3_or_5_l1625_162507


namespace angle_in_third_quadrant_l1625_162535

theorem angle_in_third_quadrant (α : ℝ) (k : ℤ) :
  (2 * ↑k * Real.pi + Real.pi < α ∧ α < 2 * ↑k * Real.pi + 3 * Real.pi / 2) →
  (∃ (m : ℤ), (0 < α / 3 + m * 2 * Real.pi ∧ α / 3 + m * 2 * Real.pi < Real.pi ∨
                π < α / 3 + m * 2 * Real.pi ∧ α / 3 + m * 2 * Real.pi < 3 * Real.pi / 2 ∨ 
                -π < α / 3 + m * 2 * Real.pi ∧ α / 3 + m * 2 * Real.pi < 0)) :=
by
  sorry

end angle_in_third_quadrant_l1625_162535


namespace trapezoid_area_l1625_162543

theorem trapezoid_area (x : ℝ) :
  let base1 := 4 * x
  let base2 := 6 * x
  let height := x
  (base1 + base2) / 2 * height = 5 * x^2 :=
by
  sorry

end trapezoid_area_l1625_162543


namespace problem_1_l1625_162526

theorem problem_1
  (α : ℝ)
  (h : Real.tan α = -1/2) :
  1 / (Real.sin α ^ 2 - Real.sin α * Real.cos α - 2 * Real.cos α ^ 2) = -1 := 
sorry

end problem_1_l1625_162526


namespace expand_binomials_l1625_162516

variable (x y : ℝ)

theorem expand_binomials: (2 * x - 5) * (3 * y + 15) = 6 * x * y + 30 * x - 15 * y - 75 :=
by sorry

end expand_binomials_l1625_162516


namespace isosceles_trapezoid_side_length_l1625_162545

theorem isosceles_trapezoid_side_length (A b1 b2 h half_diff s : ℝ) (h0 : A = 44) (h1 : b1 = 8) (h2 : b2 = 14) 
    (h3 : A = 0.5 * (b1 + b2) * h)
    (h4 : h = 4) 
    (h5 : half_diff = (b2 - b1) / 2) 
    (h6 : half_diff = 3)
    (h7 : s^2 = h^2 + half_diff^2)
    (h8 : s = 5) : 
    s = 5 :=
by 
    apply h8

end isosceles_trapezoid_side_length_l1625_162545


namespace coins_from_brother_l1625_162561

-- Defining the conditions as variables
variables (piggy_bank_coins : ℕ) (father_coins : ℕ) (given_to_Laura : ℕ) (left_coins : ℕ)

-- Setting the conditions
def conditions : Prop :=
  piggy_bank_coins = 15 ∧
  father_coins = 8 ∧
  given_to_Laura = 21 ∧
  left_coins = 15

-- The main theorem statement
theorem coins_from_brother (B : ℕ) :
  conditions piggy_bank_coins father_coins given_to_Laura left_coins →
  piggy_bank_coins + B + father_coins - given_to_Laura = left_coins →
  B = 13 :=
by
  sorry

end coins_from_brother_l1625_162561


namespace asymptote_equations_l1625_162541

open Real

noncomputable def hyperbola_asymptotes (a b : ℝ) (e : ℝ) (x y : ℝ) :=
  (a > 0) ∧ (b > 0) ∧ (e = sqrt 3) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

theorem asymptote_equations (a b : ℝ) (ha : a > 0) (hb : b > 0) (he : sqrt (a^2 + b^2) / a = sqrt 3) :
  ∀ (x : ℝ), ∃ (y : ℝ), y = sqrt 2 * x ∨ y = -sqrt 2 * x :=
sorry

end asymptote_equations_l1625_162541


namespace problem1_problem2_l1625_162521

variable (α : ℝ)

axiom tan_alpha_condition : Real.tan (Real.pi + α) = -1/2

-- Problem 1 Statement
theorem problem1 
  (tan_alpha_condition : Real.tan (Real.pi + α) = -1/2) : 
  (2 * Real.cos (Real.pi - α) - 3 * Real.sin (Real.pi + α)) / 
  (4 * Real.cos (α - 2 * Real.pi) + Real.cos (3 * Real.pi / 2 - α)) = -7/9 := 
sorry

-- Problem 2 Statement
theorem problem2
  (tan_alpha_condition : Real.tan (Real.pi + α) = -1/2) :
  Real.sin α ^ 2 - 2 * Real.sin α * Real.cos α + 4 * Real.cos α ^ 2 = 21/5 := 
sorry

end problem1_problem2_l1625_162521


namespace equation_of_line_passing_through_A_equation_of_circle_l1625_162554

variable {α β γ : ℝ}
variable {a b c u v w : ℝ}
variable (A : ℝ × ℝ × ℝ) -- Barycentric coordinates of point A

-- Statement for the equation of a line passing through point A in barycentric coordinates
theorem equation_of_line_passing_through_A (A : ℝ × ℝ × ℝ) : 
  ∃ (u v w : ℝ), u * α + v * β + w * γ = 0 := by
  sorry

-- Statement for the equation of a circle in barycentric coordinates
theorem equation_of_circle {u v w : ℝ} :
  -a^2 * β * γ - b^2 * γ * α - c^2 * α * β +
  (u * α + v * β + w * γ) * (α + β + γ) = 0 := by
  sorry

end equation_of_line_passing_through_A_equation_of_circle_l1625_162554


namespace total_slices_l1625_162547

def pizzas : ℕ := 2
def slices_per_pizza : ℕ := 8

theorem total_slices : pizzas * slices_per_pizza = 16 :=
by
  sorry

end total_slices_l1625_162547


namespace four_letters_three_mailboxes_l1625_162560

theorem four_letters_three_mailboxes : (3 ^ 4) = 81 :=
  by sorry

end four_letters_three_mailboxes_l1625_162560


namespace student_solved_correctly_l1625_162589

-- Problem conditions as definitions
def sums_attempted : Nat := 96

def sums_correct (x : Nat) : Prop :=
  let sums_wrong := 3 * x
  x + sums_wrong = sums_attempted

-- Lean statement to prove
theorem student_solved_correctly (x : Nat) (h : sums_correct x) : x = 24 :=
  sorry

end student_solved_correctly_l1625_162589


namespace range_of_a_l1625_162519

noncomputable def odd_function_periodic_real (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ -- odd function condition
  (∀ x, f (x + 5) = f x) ∧ -- periodic function condition
  (f 1 < -1) ∧ -- given condition
  (f 4 = Real.log a / Real.log 2) -- condition using log base 2

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (h : odd_function_periodic_real f a) : a > 2 :=
by sorry 

end range_of_a_l1625_162519


namespace reflect_point_x_axis_correct_l1625_162559

-- Definition of the transformation reflecting a point across the x-axis
def reflect_x_axis (P : ℝ × ℝ) : ℝ × ℝ := (P.1, -P.2)

-- Define the original point coordinates
def P : ℝ × ℝ := (-2, 3)

-- The Lean proof statement
theorem reflect_point_x_axis_correct :
  reflect_x_axis P = (-2, -3) :=
sorry

end reflect_point_x_axis_correct_l1625_162559


namespace hall_length_width_difference_l1625_162583

theorem hall_length_width_difference (L W : ℝ) 
  (h1 : W = 1 / 2 * L)
  (h2 : L * W = 450) :
  L - W = 15 :=
sorry

end hall_length_width_difference_l1625_162583


namespace circumscribed_radius_of_triangle_ABC_l1625_162528

variable (A B C R : ℝ) (a b c : ℝ)

noncomputable def triangle_ABC (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ B = 2 * A ∧ C = 3 * A

noncomputable def side_length (A a : ℝ) : Prop :=
  a = 6

noncomputable def circumscribed_radius (A B C a R : ℝ) : Prop :=
  2 * R = a / (Real.sin (Real.pi * A / 180))

theorem circumscribed_radius_of_triangle_ABC:
  triangle_ABC A B C →
  side_length A a →
  circumscribed_radius A B C a R →
  R = 6 :=
by
  intros
  sorry

end circumscribed_radius_of_triangle_ABC_l1625_162528


namespace triangle_AB_eq_3_halves_CK_l1625_162595

/-- Mathematically equivalent problem:
In an acute triangle ABC, rectangle ACGH is constructed with AC as one side, and CG : AC = 2:1.
A square BCEF is constructed with BC as one side. The height CD from A to B intersects GE at point K.
Prove that AB = 3/2 * CK. -/
theorem triangle_AB_eq_3_halves_CK
  (A B C H G E K : Type)
  (triangle_ABC_acute : ∀(A B C : Type), True) 
  (rectangle_ACGH : ∀(A C G H : Type), True) 
  (square_BCEF : ∀(B C E F : Type), True)
  (H_C_G_collinear : ∀(H C G : Type), True)
  (HCG_ratio : ∀ (AC CG : ℝ), CG / AC = 2 / 1)
  (BC_side : ∀ (BC : ℝ), BC = 1)
  (height_CD_intersection : ∀ (A B C D E G : Type), True)
  (intersection_point_K : ∀ (C D G E K : Type), True) :
  ∃ (AB CK : ℝ), AB = 3 / 2 * CK :=
by sorry

end triangle_AB_eq_3_halves_CK_l1625_162595


namespace max_value_of_expression_l1625_162527

def real_numbers (m n : ℝ) := m > 0 ∧ n < 0 ∧ (1 / m + 1 / n = 1)

theorem max_value_of_expression (m n : ℝ) (h : real_numbers m n) : 4 * m + n ≤ 1 :=
  sorry

end max_value_of_expression_l1625_162527


namespace moles_of_water_from_reaction_l1625_162506

def moles_of_water_formed (nh4cl_moles : ℕ) (naoh_moles : ℕ) : ℕ :=
  nh4cl_moles -- Because 1:1 ratio of reactants producing water

theorem moles_of_water_from_reaction :
  moles_of_water_formed 3 3 = 3 := by
  -- Use the condition of the 1:1 reaction ratio derivable from the problem's setup.
  sorry

end moles_of_water_from_reaction_l1625_162506


namespace deepak_present_age_l1625_162598

/-- Let Rahul and Deepak's current ages be 4x and 3x respectively
  Given that:
  1. The ratio between Rahul and Deepak's ages is 4:3
  2. After 6 years, Rahul's age will be 26 years
  Prove that Deepak's present age is 15 years.
-/
theorem deepak_present_age (x : ℕ) (hx : 4 * x + 6 = 26) : 3 * x = 15 :=
by
  sorry

end deepak_present_age_l1625_162598


namespace rajesh_monthly_savings_l1625_162594

theorem rajesh_monthly_savings
  (salary : ℝ)
  (percentage_food : ℝ)
  (percentage_medicines : ℝ)
  (percentage_savings : ℝ)
  (amount_food : ℝ := percentage_food * salary)
  (amount_medicines : ℝ := percentage_medicines * salary)
  (remaining_amount : ℝ := salary - (amount_food + amount_medicines))
  (save_amount : ℝ := percentage_savings * remaining_amount)
  (H_salary : salary = 15000)
  (H_percentage_food : percentage_food = 0.40)
  (H_percentage_medicines : percentage_medicines = 0.20)
  (H_percentage_savings : percentage_savings = 0.60) :
  save_amount = 3600 :=
by
  sorry

end rajesh_monthly_savings_l1625_162594


namespace parabola_line_non_intersect_l1625_162555

def P (x : ℝ) : ℝ := x^2 + 3 * x + 1
def Q : ℝ × ℝ := (10, 50)

def line_through_Q_with_slope (m x : ℝ) : ℝ := m * (x - Q.1) + Q.2

theorem parabola_line_non_intersect (r s : ℝ) (h : ∀ m, (r < m ∧ m < s) ↔ (∀ x, 
  x^2 + (3 - m) * x + (10 * m - 49) ≠ 0)) : r + s = 46 := 
sorry

end parabola_line_non_intersect_l1625_162555


namespace find_m_value_l1625_162538

theorem find_m_value : ∃ m : ℤ, 81 - 6 = 25 + m ∧ m = 50 :=
by
  sorry

end find_m_value_l1625_162538


namespace value_of_a_l1625_162581

theorem value_of_a (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2*x + m ≤ 0) → m > 1 :=
sorry

end value_of_a_l1625_162581


namespace calculate_power_expr_l1625_162593

theorem calculate_power_expr :
  let a := (-8 : ℝ)
  let b := (0.125 : ℝ)
  a^2023 * b^2024 = -0.125 :=
by
  sorry

end calculate_power_expr_l1625_162593


namespace problem1_problem2_problem3_l1625_162549

noncomputable def f (x : ℝ) : ℝ := if x >= 0 then x^2 - 2 * x else (abs x)^2 - 2 * abs x

-- Define the condition that f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Problem 1: Prove the minimum value of f(x) is -1.
theorem problem1 (h_even : even_function f) : ∃ x : ℝ, f x = -1 :=
by
  sorry

-- Problem 2: Prove the solution set of f(x) > 0 is (-∞, -2) ∪ (2, +∞).
theorem problem2 (h_even : even_function f) : 
  { x : ℝ | f x > 0 } = { x : ℝ | x < -2 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

-- Problem 3: Prove there exists a real number x such that f(x+2) + f(-x) = 0.
theorem problem3 (h_even : even_function f) : ∃ x : ℝ, f (x + 2) + f (-x) = 0 :=
by
  sorry

end problem1_problem2_problem3_l1625_162549


namespace arithmetic_seq_sum_ratio_l1625_162520

theorem arithmetic_seq_sum_ratio (a1 d : ℝ) (S : ℕ → ℝ) 
  (hSn : ∀ n, S n = n * a1 + d * (n * (n - 1) / 2))
  (h_ratio : S 3 / S 6 = 1 / 3) :
  S 9 / S 6 = 2 :=
by
  sorry

end arithmetic_seq_sum_ratio_l1625_162520


namespace f_2007_2007_l1625_162503

def f (n : ℕ) : ℕ :=
  n.digits 10 |>.map (fun d => d * d) |>.sum

def f_k : ℕ → ℕ → ℕ
| 0, n => n
| (k+1), n => f (f_k k n)

theorem f_2007_2007 : f_k 2007 2007 = 145 :=
by
  sorry -- Proof omitted

end f_2007_2007_l1625_162503


namespace solve_for_y_l1625_162514

variable {b c y : Real}

theorem solve_for_y (h : b > c) (h_eq : y^2 + c^2 = (b - y)^2) : y = (b^2 - c^2) / (2 * b) := 
sorry

end solve_for_y_l1625_162514


namespace sphere_center_x_axis_eq_l1625_162552

theorem sphere_center_x_axis_eq (a : ℝ) (R : ℝ) (x y z : ℝ) :
  (x - a) ^ 2 + y ^ 2 + z ^ 2 = R ^ 2 → (0 - a) ^ 2 + (0 - 0) ^ 2 + (0 - 0) ^ 2 = R ^ 2 →
  a = R →
  (x ^ 2 - 2 * a * x + y ^ 2 + z ^ 2 = 0) :=
by
  sorry

end sphere_center_x_axis_eq_l1625_162552


namespace balcony_more_than_orchestra_l1625_162500

-- Conditions
def total_tickets (O B : ℕ) : Prop := O + B = 340
def total_cost (O B : ℕ) : Prop := 12 * O + 8 * B = 3320

-- The statement we need to prove based on the conditions
theorem balcony_more_than_orchestra (O B : ℕ) (h1 : total_tickets O B) (h2 : total_cost O B) :
  B - O = 40 :=
sorry

end balcony_more_than_orchestra_l1625_162500


namespace ages_of_Mel_and_Lexi_l1625_162580

theorem ages_of_Mel_and_Lexi (M L K : ℤ)
  (h1 : M = K - 3)
  (h2 : L = M + 2)
  (h3 : K = 60) :
  M = 57 ∧ L = 59 :=
  by
    -- Proof steps are omitted.
    sorry

end ages_of_Mel_and_Lexi_l1625_162580


namespace white_tulips_multiple_of_seven_l1625_162599

/-- Let R be the number of red tulips, which is given as 91. 
    We also know that the greatest number of identical bouquets that can be made without 
    leaving any flowers out is 7.
    Prove that the number of white tulips W is a multiple of 7. -/
theorem white_tulips_multiple_of_seven (R : ℕ) (g : ℕ) (W : ℕ) (hR : R = 91) (hg : g = 7) :
  ∃ w : ℕ, W = 7 * w :=
by
  sorry

end white_tulips_multiple_of_seven_l1625_162599


namespace ab_inequality_l1625_162548

theorem ab_inequality
  {a b : ℝ}
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (a_b_sum : a + b = 2) :
  ∀ n : ℕ, 2 ≤ n → (a^n + 1) * (b^n + 1) ≥ 4 :=
by
  sorry

end ab_inequality_l1625_162548


namespace total_flour_needed_l1625_162577

theorem total_flour_needed (Katie_flour : ℕ) (Sheila_flour : ℕ) 
  (h1 : Katie_flour = 3) 
  (h2 : Sheila_flour = Katie_flour + 2) : 
  Katie_flour + Sheila_flour = 8 := 
  by 
  sorry

end total_flour_needed_l1625_162577


namespace side_length_of_smaller_square_l1625_162544

theorem side_length_of_smaller_square (s : ℝ) (A1 A2 : ℝ) (h1 : 5 * 5 = A1 + A2) (h2 : 2 * A2 = A1 + 25)  : s = 5 * Real.sqrt 3 / 3 :=
by
  sorry

end side_length_of_smaller_square_l1625_162544


namespace pythagorean_triangle_divisible_by_5_l1625_162571

theorem pythagorean_triangle_divisible_by_5 {a b c : ℕ} (h : a^2 + b^2 = c^2) : 
  5 ∣ a ∨ 5 ∣ b ∨ 5 ∣ c := 
by
  sorry

end pythagorean_triangle_divisible_by_5_l1625_162571


namespace friend_balloon_count_l1625_162563

theorem friend_balloon_count (you_balloons friend_balloons : ℕ) (h1 : you_balloons = 7) (h2 : you_balloons = friend_balloons + 2) : friend_balloons = 5 :=
by
  sorry

end friend_balloon_count_l1625_162563


namespace spinner_probability_l1625_162575

theorem spinner_probability (P_D P_E : ℝ) (hD : P_D = 2/5) (hE : P_E = 1/5) 
  (hTotal : P_D + P_E + P_F = 1) : P_F = 2/5 :=
by
  sorry

end spinner_probability_l1625_162575


namespace find_number_added_l1625_162517

theorem find_number_added (x : ℕ) : (1250 / 50) + x = 7525 ↔ x = 7500 := by
  sorry

end find_number_added_l1625_162517


namespace problem_I_problem_II_l1625_162596

-- Problem (I)
theorem problem_I (a : ℝ) (h : ∀ x : ℝ, x^2 - 3 * a * x + 9 > 0) : -2 ≤ a ∧ a ≤ 2 :=
sorry

-- Problem (II)
theorem problem_II (m : ℝ) 
  (h₁ : ∀ x : ℝ, x^2 + 2 * x - 8 < 0 → x - m > 0)
  (h₂ : ∃ x : ℝ, x^2 + 2 * x - 8 < 0) : m ≤ -4 :=
sorry

end problem_I_problem_II_l1625_162596


namespace min_value_of_x_plus_y_l1625_162576

theorem min_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy: 0 < y) (h: 9 * x + y = x * y) : x + y ≥ 16 := 
sorry

end min_value_of_x_plus_y_l1625_162576


namespace probability_closer_to_6_l1625_162556

theorem probability_closer_to_6 :
  let interval : Set ℝ := Set.Icc 0 6
  let subinterval : Set ℝ := Set.Icc 3 6
  let length_interval := 6
  let length_subinterval := 3
  (length_subinterval / length_interval) = 0.5 := by
    sorry

end probability_closer_to_6_l1625_162556


namespace minimum_value_2_only_in_option_b_l1625_162584

noncomputable def option_a (x : ℝ) : ℝ := x + 1 / x
noncomputable def option_b (x : ℝ) : ℝ := 3^x + 3^(-x)
noncomputable def option_c (x : ℝ) : ℝ := (Real.log x) + 1 / (Real.log x)
noncomputable def option_d (x : ℝ) : ℝ := (Real.sin x) + 1 / (Real.sin x)

theorem minimum_value_2_only_in_option_b :
  (∀ x > 0, option_a x ≠ 2) ∧
  (∃ x, option_b x = 2) ∧
  (∀ x (h: 0 < x) (h' : x < 1), option_c x ≠ 2) ∧
  (∀ x (h: 0 < x) (h' : x < π / 2), option_d x ≠ 2) :=
by
  sorry

end minimum_value_2_only_in_option_b_l1625_162584


namespace line_shift_upwards_l1625_162569

theorem line_shift_upwards (x y : ℝ) (h : y = -2 * x) : y + 3 = -2 * x + 3 :=
by sorry

end line_shift_upwards_l1625_162569


namespace hydrogen_atoms_in_compound_l1625_162550

theorem hydrogen_atoms_in_compound : 
  ∀ (C O H : ℕ) (molecular_weight : ℕ), 
  C = 1 → 
  O = 3 → 
  molecular_weight = 62 → 
  (12 * C + 16 * O + H = molecular_weight) → 
  H = 2 := 
by
  intros C O H molecular_weight hc ho hmw hcalc
  sorry

end hydrogen_atoms_in_compound_l1625_162550


namespace range_of_a_l1625_162513

variable (a x : ℝ)

def p (a x : ℝ) : Prop := a - 4 < x ∧ x < a + 4

def q (x : ℝ) : Prop := (x - 2) * (x - 3) > 0

theorem range_of_a (h : ∀ (x : ℝ), p a x → q x) : a <= -2 ∨ a >= 7 := 
by sorry

end range_of_a_l1625_162513


namespace grade_representation_l1625_162530

theorem grade_representation :
  (8, 1) = (8, 1) :=
by
  sorry

end grade_representation_l1625_162530


namespace problem1_problem2_problem3_problem4_problem5_problem6_l1625_162558

-- First problem: \(\frac{1}{3} + \left(-\frac{1}{2}\right) = -\frac{1}{6}\)
theorem problem1 : (1 / 3 : ℚ) + (-1 / 2) = -1 / 6 := by sorry

-- Second problem: \(-2 - \left(-9\right) = 7\)
theorem problem2 : (-2 : ℚ) - (-9) = 7 := by sorry

-- Third problem: \(\frac{15}{16} - \left(-7\frac{1}{16}\right) = 8\)
theorem problem3 : (15 / 16 : ℚ) - (-(7 + 1 / 16)) = 8 := by sorry

-- Fourth problem: \(-\left|-4\frac{2}{7}\right| - \left|+1\frac{5}{7}\right| = -6\)
theorem problem4 : -|(-4 - 2 / 7 : ℚ)| - |(1 + 5 / 7)| = -6 := by sorry

-- Fifth problem: \(6 + \left(-12\right) + 8.3 + \left(-7.5\right) = -5.2\)
theorem problem5 : (6 : ℚ) + (-12) + (83 / 10) + (-75 / 10) = -52 / 10 := by sorry

-- Sixth problem: \(\left(-\frac{1}{8}\right) + 3.25 + 2\frac{3}{5} + \left(-5.875\right) + 1.15 = 1\)
theorem problem6 : (-1 / 8 : ℚ) + 3 + 1 / 4 + 2 + 3 / 5 + (-5 - 875 / 1000) + 1 + 15 / 100 = 1 := by sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l1625_162558


namespace white_pairs_coincide_l1625_162572

theorem white_pairs_coincide 
  (red_half : ℕ) (blue_half : ℕ) (white_half : ℕ)
  (red_pairs : ℕ) (blue_pairs : ℕ) (red_white_pairs : ℕ) :
  red_half = 2 → blue_half = 4 → white_half = 6 →
  red_pairs = 1 → blue_pairs = 2 → red_white_pairs = 2 →
  2 * (red_half - red_pairs + blue_half - 2 * blue_pairs + 
       white_half - 2 * red_white_pairs) = 4 :=
by
  intros 
    h_red_half h_blue_half h_white_half 
    h_red_pairs h_blue_pairs h_red_white_pairs
  rw [h_red_half, h_blue_half, h_white_half, 
      h_red_pairs, h_blue_pairs, h_red_white_pairs]
  sorry

end white_pairs_coincide_l1625_162572


namespace cos_2alpha_plus_pi_over_3_l1625_162510

open Real

theorem cos_2alpha_plus_pi_over_3 
  (alpha : ℝ) 
  (h1 : cos (alpha - π / 12) = 3 / 5) 
  (h2 : 0 < alpha ∧ alpha < π / 2) : 
  cos (2 * alpha + π / 3) = -24 / 25 := 
sorry

end cos_2alpha_plus_pi_over_3_l1625_162510


namespace major_axis_length_l1625_162579

-- Definitions of the given conditions
structure Ellipse :=
  (focus1 focus2 : ℝ × ℝ)
  (tangent_to_x_axis : Bool)

noncomputable def length_of_major_axis (E : Ellipse) : ℝ :=
  let (x1, y1) := E.focus1
  let (x2, y2) := E.focus2
  Real.sqrt ((x2 - x1) ^ 2 + (y2 + y1) ^ 2)

-- The theorem we want to prove given the conditions
theorem major_axis_length (E : Ellipse)
  (h1 : E.focus1 = (9, 20))
  (h2 : E.focus2 = (49, 55))
  (h3 : E.tangent_to_x_axis = true):
  length_of_major_axis E = 85 :=
by
  sorry

end major_axis_length_l1625_162579


namespace arithmetic_seq_a7_l1625_162525

structure arith_seq (a : ℕ → ℤ) : Prop :=
  (step : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d)

theorem arithmetic_seq_a7
  {a : ℕ → ℤ}
  (h_seq : arith_seq a)
  (h1 : a 1 = 2)
  (h2 : a 3 + a 5 = 10)
  : a 7 = 8 :=
by
  sorry

end arithmetic_seq_a7_l1625_162525
