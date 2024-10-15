import Mathlib

namespace NUMINAMATH_GPT_evaluate_expression_at_minus3_l1048_104833

theorem evaluate_expression_at_minus3:
  (∀ x, x = -3 → (3 + x * (3 + x) - 3^2 + x) / (x - 3 + x^2 - x) = -3/2) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_minus3_l1048_104833


namespace NUMINAMATH_GPT_problem1_problem2_l1048_104866

-- Problem 1: Prove the range of k for any real number x
theorem problem1 (k : ℝ) (x : ℝ) (h : (k*x^2 + k*x + 4) / (x^2 + x + 1) > 1) :
  1 ≤ k ∧ k < 13 :=
sorry

-- Problem 2: Prove the range of k for any x in the interval (0, 1]
theorem problem2 (k : ℝ) (x : ℝ) (hx : 0 < x) (hx1 : x ≤ 1) (h : (k*x^2 + k*x + 4) / (x^2 + x + 1) > 1) :
  k > -1/2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1048_104866


namespace NUMINAMATH_GPT_construct_triangle_given_side_and_medians_l1048_104880

theorem construct_triangle_given_side_and_medians
  (AB : ℝ) (m_a m_b : ℝ)
  (h1 : AB > 0) (h2 : m_a > 0) (h3 : m_b > 0) :
  ∃ (A B C : ℝ × ℝ),
    (∃ G : ℝ × ℝ, 
      dist A B = AB ∧ 
      dist A G = (2 / 3) * m_a ∧
      dist B G = (2 / 3) * m_b ∧ 
      dist G (midpoint ℝ A C) = m_b / 3 ∧ 
      dist G (midpoint ℝ B C) = m_a / 3) :=
sorry

end NUMINAMATH_GPT_construct_triangle_given_side_and_medians_l1048_104880


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l1048_104899

open Real

noncomputable def ellipse_eccentricity : ℝ :=
  let a : ℝ := 4
  let b : ℝ := 2 * sqrt 3
  let c : ℝ := sqrt (a^2 - b^2)
  c / a

theorem eccentricity_of_ellipse (a b : ℝ) (ha : a = 4) (hb : b = 2 * sqrt 3) (h_eq : ∀ A B : ℝ, |A - B| = b^2 / 2 → |A - 2 * sqrt 3| + |B - 2 * sqrt 3| ≤ 10) :
  ellipse_eccentricity = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l1048_104899


namespace NUMINAMATH_GPT_initial_students_per_group_l1048_104864

-- Define the conditions
variables {x : ℕ} (h : 3 * x - 2 = 22)

-- Lean 4 statement of the proof problem
theorem initial_students_per_group (x : ℕ) (h : 3 * x - 2 = 22) : x = 8 :=
sorry

end NUMINAMATH_GPT_initial_students_per_group_l1048_104864


namespace NUMINAMATH_GPT_heather_ends_up_with_45_blocks_l1048_104883

-- Conditions
def initialBlocks (Heather : Type) : ℕ := 86
def sharedBlocks (Heather : Type) : ℕ := 41

-- The theorem to prove
theorem heather_ends_up_with_45_blocks (Heather : Type) :
  (initialBlocks Heather) - (sharedBlocks Heather) = 45 :=
by
  sorry

end NUMINAMATH_GPT_heather_ends_up_with_45_blocks_l1048_104883


namespace NUMINAMATH_GPT_no_solutions_Y_l1048_104828

theorem no_solutions_Y (Y : ℕ) : 2 * Y + Y + 3 * Y = 14 ↔ false :=
by 
  sorry

end NUMINAMATH_GPT_no_solutions_Y_l1048_104828


namespace NUMINAMATH_GPT_shortest_chord_through_point_on_circle_l1048_104836

theorem shortest_chord_through_point_on_circle :
  ∀ (M : ℝ × ℝ) (x y : ℝ),
    M = (3, 0) →
    x^2 + y^2 - 8 * x - 2 * y + 10 = 0 →
    ∃ (a b c : ℝ), a * x + b * y + c = 0 ∧ a = 1 ∧ b = 1 ∧ c = -3 :=
by
  sorry

end NUMINAMATH_GPT_shortest_chord_through_point_on_circle_l1048_104836


namespace NUMINAMATH_GPT_items_per_baggie_l1048_104853

def num_pretzels : ℕ := 64
def num_suckers : ℕ := 32
def num_kids : ℕ := 16
def num_goldfish : ℕ := 4 * num_pretzels
def total_items : ℕ := num_pretzels + num_goldfish + num_suckers

theorem items_per_baggie : total_items / num_kids = 22 :=
by
  -- Calculation proof
  sorry

end NUMINAMATH_GPT_items_per_baggie_l1048_104853


namespace NUMINAMATH_GPT_always_composite_for_x64_l1048_104881

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = n

theorem always_composite_for_x64 (n : ℕ) : is_composite (n^4 + 64) :=
by
  sorry

end NUMINAMATH_GPT_always_composite_for_x64_l1048_104881


namespace NUMINAMATH_GPT_truck_boxes_per_trip_l1048_104861

theorem truck_boxes_per_trip (total_boxes trips : ℕ) (h1 : total_boxes = 871) (h2 : trips = 218) : total_boxes / trips = 4 := by
  sorry

end NUMINAMATH_GPT_truck_boxes_per_trip_l1048_104861


namespace NUMINAMATH_GPT_total_cost_correct_l1048_104824

noncomputable def totalCost : ℝ :=
  let fuel_efficiences := [15, 12, 14, 10, 13, 15]
  let distances := [10, 6, 7, 5, 3, 9]
  let gas_prices := [3.5, 3.6, 3.4, 3.55, 3.55, 3.5]
  let gas_used := distances.zip fuel_efficiences |>.map (λ p => (p.1 : ℝ) / p.2)
  let costs := gas_prices.zip gas_used |>.map (λ p => p.1 * p.2)
  costs.sum

theorem total_cost_correct : abs (totalCost - 10.52884) < 0.01 := by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l1048_104824


namespace NUMINAMATH_GPT_probability_none_A_B_C_l1048_104845

-- Define the probabilities as given conditions
def P_A : ℝ := 0.25
def P_B : ℝ := 0.40
def P_C : ℝ := 0.35
def P_AB : ℝ := 0.20
def P_AC : ℝ := 0.15
def P_BC : ℝ := 0.25
def P_ABC : ℝ := 0.10

-- Prove that the probability that none of the events A, B, C occur simultaneously is 0.50
theorem probability_none_A_B_C : 1 - (P_A + P_B + P_C - P_AB - P_AC - P_BC + P_ABC) = 0.50 :=
by
  sorry

end NUMINAMATH_GPT_probability_none_A_B_C_l1048_104845


namespace NUMINAMATH_GPT_fill_up_mini_vans_l1048_104870

/--
In a fuel station, the service costs $2.20 per vehicle and every liter of fuel costs $0.70.
Assume that mini-vans have a tank size of 65 liters, and trucks have a tank size of 143 liters.
Given that 2 trucks were filled up and the total cost was $347.7,
prove the number of mini-vans filled up is 3.
-/
theorem fill_up_mini_vans (m : ℝ) (t : ℝ) 
    (service_cost_per_vehicle fuel_cost_per_liter : ℝ)
    (van_tank_size truck_tank_size total_cost : ℝ):
    service_cost_per_vehicle = 2.20 →
    fuel_cost_per_liter = 0.70 →
    van_tank_size = 65 →
    truck_tank_size = 143 →
    t = 2 →
    total_cost = 347.7 →
    (service_cost_per_vehicle * m + service_cost_per_vehicle * t) + (fuel_cost_per_liter * van_tank_size * m) + (fuel_cost_per_liter * truck_tank_size * t) = total_cost →
    m = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_fill_up_mini_vans_l1048_104870


namespace NUMINAMATH_GPT_water_in_bowl_after_adding_4_cups_l1048_104825

def total_capacity_bowl := 20 -- Capacity of the bowl in cups

def initially_half_full (C : ℕ) : Prop :=
C = total_capacity_bowl / 2

def after_adding_4_cups (initial : ℕ) : ℕ :=
initial + 4

def seventy_percent_full (C : ℕ) : ℕ :=
7 * C / 10

theorem water_in_bowl_after_adding_4_cups :
  ∀ (C initial after_adding) (h1 : initially_half_full initial)
  (h2 : after_adding = after_adding_4_cups initial)
  (h3 : after_adding = seventy_percent_full C),
  after_adding = 14 := 
by
  intros C initial after_adding h1 h2 h3
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_water_in_bowl_after_adding_4_cups_l1048_104825


namespace NUMINAMATH_GPT_original_number_is_17_l1048_104815

theorem original_number_is_17 (x : ℤ) (h : (x + 6) % 23 = 0) : x = 17 :=
sorry

end NUMINAMATH_GPT_original_number_is_17_l1048_104815


namespace NUMINAMATH_GPT_unique_x_condition_l1048_104819

theorem unique_x_condition (x : ℝ) : 
  (1 ≤ x ∧ x < 2) ∧ (∀ n : ℕ, 0 < n → (⌊2^n * x⌋ % 4 = 1 ∨ ⌊2^n * x⌋ % 4 = 2)) ↔ x = 4/3 := 
by 
  sorry

end NUMINAMATH_GPT_unique_x_condition_l1048_104819


namespace NUMINAMATH_GPT_smallest_pos_integer_for_frac_reducible_l1048_104872

theorem smallest_pos_integer_for_frac_reducible :
  ∃ n : ℕ, n > 0 ∧ ∃ d > 1, d ∣ (n - 17) ∧ d ∣ (6 * n + 8) ∧ n = 127 :=
by
  sorry

end NUMINAMATH_GPT_smallest_pos_integer_for_frac_reducible_l1048_104872


namespace NUMINAMATH_GPT_papaya_cost_is_one_l1048_104898

theorem papaya_cost_is_one (lemons_cost : ℕ) (mangos_cost : ℕ) (total_fruits : ℕ) (total_cost_paid : ℕ) :
    (lemons_cost = 2) → (mangos_cost = 4) → (total_fruits = 12) → (total_cost_paid = 21) → 
    let discounts := total_fruits / 4
    let lemons_bought := 6
    let mangos_bought := 2
    let papayas_bought := 4
    let total_discount := discounts
    let total_cost_before_discount := lemons_bought * lemons_cost + mangos_bought * mangos_cost + papayas_bought * P
    total_cost_before_discount - total_discount = total_cost_paid → 
    P = 1 := 
by 
  intros h1 h2 h3 h4 
  let discounts := total_fruits / 4
  let lemons_bought := 6
  let mangos_bought := 2
  let papayas_bought := 4
  let total_discount := discounts
  let total_cost_before_discount := lemons_bought * lemons_cost + mangos_bought * mangos_cost + papayas_bought * P
  sorry

end NUMINAMATH_GPT_papaya_cost_is_one_l1048_104898


namespace NUMINAMATH_GPT_calculate_subtraction_l1048_104878

theorem calculate_subtraction :
  ∀ (x : ℕ), (49 = 50 - 1) → (49^2 = 50^2 - 99)
  := by
  intros x h
  sorry

end NUMINAMATH_GPT_calculate_subtraction_l1048_104878


namespace NUMINAMATH_GPT_zhang_shan_sales_prediction_l1048_104806

theorem zhang_shan_sales_prediction (x : ℝ) (y : ℝ) (h : x = 34) (reg_eq : y = 2 * x + 60) : y = 128 :=
by
  sorry

end NUMINAMATH_GPT_zhang_shan_sales_prediction_l1048_104806


namespace NUMINAMATH_GPT_master_li_speeding_l1048_104841

theorem master_li_speeding (distance : ℝ) (time : ℝ) (speed_limit : ℝ) (average_speed : ℝ)
  (h_distance : distance = 165)
  (h_time : time = 2)
  (h_speed_limit : speed_limit = 80)
  (h_average_speed : average_speed = distance / time)
  (h_speeding : average_speed > speed_limit) :
  True :=
sorry

end NUMINAMATH_GPT_master_li_speeding_l1048_104841


namespace NUMINAMATH_GPT_find_valid_pairs_l1048_104816

open Nat

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ m : ℕ, 2 ≤ m → m ≤ p / 2 → ¬(m ∣ p)

def valid_pair (n p : ℕ) : Prop :=
  is_prime p ∧ 0 < n ∧ n ≤ 2 * p ∧ n ^ (p - 1) ∣ (p - 1) ^ n + 1

theorem find_valid_pairs (n p : ℕ) : valid_pair n p ↔ (n = 1 ∧ is_prime p) ∨ (n, p) = (2, 2) ∨ (n, p) = (3, 3) := by
  sorry

end NUMINAMATH_GPT_find_valid_pairs_l1048_104816


namespace NUMINAMATH_GPT_speed_of_current_l1048_104888

theorem speed_of_current (v_w v_c : ℝ) (h_downstream : 125 = (v_w + v_c) * 10)
                         (h_upstream : 60 = (v_w - v_c) * 10) :
  v_c = 3.25 :=
by {
  sorry
}

end NUMINAMATH_GPT_speed_of_current_l1048_104888


namespace NUMINAMATH_GPT_line_through_points_l1048_104838

theorem line_through_points 
  (A1 B1 A2 B2 : ℝ) 
  (h₁ : A1 * -7 + B1 * 9 = 1) 
  (h₂ : A2 * -7 + B2 * 9 = 1) :
  ∃ (k : ℝ), ∀ (x y : ℝ), (A1, B1) ≠ (A2, B2) → y = k * x + (B1 - k * A1) → -7 * x + 9 * y = 1 :=
sorry

end NUMINAMATH_GPT_line_through_points_l1048_104838


namespace NUMINAMATH_GPT_tg_arccos_le_cos_arctg_l1048_104867

theorem tg_arccos_le_cos_arctg (x : ℝ) (h₀ : -1 ≤ x ∧ x ≤ 1) :
  (Real.tan (Real.arccos x) ≤ Real.cos (Real.arctan x)) → 
  (x ∈ Set.Icc (-1:ℝ) 0 ∨ x ∈ Set.Icc (Real.sqrt ((Real.sqrt 5 - 1) / 2)) 1) :=
by
  sorry

end NUMINAMATH_GPT_tg_arccos_le_cos_arctg_l1048_104867


namespace NUMINAMATH_GPT_height_of_trapezium_l1048_104894

-- Define the lengths of the parallel sides
def length_side1 : ℝ := 10
def length_side2 : ℝ := 18

-- Define the given area of the trapezium
def area_trapezium : ℝ := 210

-- The distance between the parallel sides (height) we want to prove
def height_between_sides : ℝ := 15

-- State the problem as a theorem in Lean: prove that the height is correct
theorem height_of_trapezium :
  (1 / 2) * (length_side1 + length_side2) * height_between_sides = area_trapezium :=
by
  sorry

end NUMINAMATH_GPT_height_of_trapezium_l1048_104894


namespace NUMINAMATH_GPT_maximize_profit_at_six_l1048_104800

-- Defining the functions (conditions)
def y1 (x : ℝ) : ℝ := 17 * x^2
def y2 (x : ℝ) : ℝ := 2 * x^3 - x^2
def profit (x : ℝ) : ℝ := y1 x - y2 x

-- The condition x > 0
def x_pos (x : ℝ) : Prop := x > 0

-- Proving the maximum profit is achieved at x = 6 (question == answer)
theorem maximize_profit_at_six : ∀ x > 0, (∀ y > 0, y = profit x → x = 6) :=
by 
  intros x hx y hy
  sorry

end NUMINAMATH_GPT_maximize_profit_at_six_l1048_104800


namespace NUMINAMATH_GPT_apples_on_tree_now_l1048_104877

-- Definitions based on conditions
def initial_apples : ℕ := 11
def apples_picked : ℕ := 7
def new_apples : ℕ := 2

-- Theorem statement proving the final number of apples on the tree
theorem apples_on_tree_now : initial_apples - apples_picked + new_apples = 6 := 
by 
  sorry

end NUMINAMATH_GPT_apples_on_tree_now_l1048_104877


namespace NUMINAMATH_GPT_similar_triangle_perimeter_l1048_104862

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

def is_isosceles (T : Triangle) : Prop :=
  T.a = T.b ∨ T.a = T.c ∨ T.b = T.c

def similar_triangles (T1 T2 : Triangle) : Prop :=
  T1.a / T2.a = T1.b / T2.b ∧ T1.b / T2.b = T1.c / T2.c ∧ T1.a / T2.a = T1.c / T2.c

noncomputable def perimeter (T : Triangle) : ℝ :=
  T.a + T.b + T.c

theorem similar_triangle_perimeter
  (T1 T2 : Triangle)
  (T1_isosceles : is_isosceles T1)
  (T1_sides : T1.a = 7 ∧ T1.b = 7 ∧ T1.c = 12)
  (T2_similar : similar_triangles T1 T2)
  (T2_longest_side : T2.c = 30) :
  perimeter T2 = 65 :=
by
  sorry

end NUMINAMATH_GPT_similar_triangle_perimeter_l1048_104862


namespace NUMINAMATH_GPT_round_24_6375_to_nearest_tenth_l1048_104803

def round_to_nearest_tenth (n : ℚ) : ℚ :=
  let tenths := (n * 10).floor / 10
  let hundredths := (n * 100).floor % 10
  if hundredths < 5 then tenths else (tenths + 0.1)

theorem round_24_6375_to_nearest_tenth :
  round_to_nearest_tenth 24.6375 = 24.6 :=
by
  sorry

end NUMINAMATH_GPT_round_24_6375_to_nearest_tenth_l1048_104803


namespace NUMINAMATH_GPT_saturn_moon_approximation_l1048_104895

theorem saturn_moon_approximation : (1.2 * 10^5) * 10 = 1.2 * 10^6 := 
by sorry

end NUMINAMATH_GPT_saturn_moon_approximation_l1048_104895


namespace NUMINAMATH_GPT_possible_values_of_g_l1048_104844

noncomputable def g (a b c : ℝ) : ℝ :=
  a / (a + b) + b / (b + c) + c / (c + a)

theorem possible_values_of_g (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 < g a b c ∧ g a b c < 2 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_g_l1048_104844


namespace NUMINAMATH_GPT_find_a_l1048_104834

-- Define the sets A and B and their union
variables (a : ℕ)
def A : Set ℕ := {0, 2, a}
def B : Set ℕ := {1, a^2}
def C : Set ℕ := {0, 1, 2, 3, 9}

-- Define the condition and prove that it implies a = 3
theorem find_a (h : A a ∪ B a = C) : a = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_a_l1048_104834


namespace NUMINAMATH_GPT_cost_price_of_article_l1048_104827

theorem cost_price_of_article (SP CP : ℝ) (h1 : SP = 150) (h2 : SP = CP + (1 / 4) * CP) : CP = 120 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_article_l1048_104827


namespace NUMINAMATH_GPT_minimum_jellybeans_l1048_104822

theorem minimum_jellybeans (n : ℕ) : n ≥ 150 ∧ n % 15 = 14 → n = 164 :=
by sorry

end NUMINAMATH_GPT_minimum_jellybeans_l1048_104822


namespace NUMINAMATH_GPT_smallest_fraction_l1048_104848

theorem smallest_fraction 
  (x y z t : ℝ) 
  (h1 : 1 < x) 
  (h2 : x < y) 
  (h3 : y < z) 
  (h4 : z < t) : 
  (min (min (min (min ((x + y) / (z + t)) ((x + t) / (y + z))) ((y + z) / (x + t))) ((y + t) / (x + z))) ((z + t) / (x + y))) = (x + y) / (z + t) :=
by {
    sorry
}

end NUMINAMATH_GPT_smallest_fraction_l1048_104848


namespace NUMINAMATH_GPT_smallest_sum_of_consecutive_integers_gt_420_l1048_104852

theorem smallest_sum_of_consecutive_integers_gt_420 : 
  ∃ n : ℕ, (n * (n + 1) > 420) ∧ (n + (n + 1) = 43) := sorry

end NUMINAMATH_GPT_smallest_sum_of_consecutive_integers_gt_420_l1048_104852


namespace NUMINAMATH_GPT_electricity_consumption_scientific_notation_l1048_104818

def electricity_consumption (x : Float) : String := 
  let scientific_notation := "3.64 × 10^4"
  scientific_notation

theorem electricity_consumption_scientific_notation :
  electricity_consumption 36400 = "3.64 × 10^4" :=
by 
  sorry

end NUMINAMATH_GPT_electricity_consumption_scientific_notation_l1048_104818


namespace NUMINAMATH_GPT_area_of_shape_l1048_104809

theorem area_of_shape (x y : ℝ) (α : ℝ) (P : ℝ × ℝ) :
  (x - 2 * Real.cos α)^2 + (y - 2 * Real.sin α)^2 = 16 →
  ∃ A : ℝ, A = 32 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_of_shape_l1048_104809


namespace NUMINAMATH_GPT_rectangular_solid_edges_sum_l1048_104823

theorem rectangular_solid_edges_sum 
  (a b c : ℝ) 
  (h1 : a * b * c = 8)
  (h2 : 2 * (a * b + b * c + c * a) = 32)
  (h3 : b^2 = a * c) : 
  4 * (a + b + c) = 32 := 
  sorry

end NUMINAMATH_GPT_rectangular_solid_edges_sum_l1048_104823


namespace NUMINAMATH_GPT_even_mult_expressions_divisible_by_8_l1048_104859

theorem even_mult_expressions_divisible_by_8 {a : ℤ} (h : ∃ k : ℤ, a = 2 * k) :
  (8 ∣ a * (a^2 + 20)) ∧ (8 ∣ a * (a^2 - 20)) ∧ (8 ∣ a * (a^2 - 4)) := by
  sorry

end NUMINAMATH_GPT_even_mult_expressions_divisible_by_8_l1048_104859


namespace NUMINAMATH_GPT_determine_f_3_2016_l1048_104814

noncomputable def f : ℕ → ℕ → ℕ
| 0, y       => y + 1
| (x + 1), 0 => f x 1
| (x + 1), (y + 1) => f x (f (x + 1) y)

theorem determine_f_3_2016 : f 3 2016 = 2 ^ 2019 - 3 := by
  sorry

end NUMINAMATH_GPT_determine_f_3_2016_l1048_104814


namespace NUMINAMATH_GPT_alex_plays_with_friends_l1048_104857

-- Define the players in the game
variables (A B V G D : Prop)

-- Define the conditions
axiom h1 : A → (B ∧ ¬V)
axiom h2 : B → (G ∨ D)
axiom h3 : ¬V → (¬B ∧ ¬D)
axiom h4 : ¬A → (B ∧ ¬G)

theorem alex_plays_with_friends : 
    (A ∧ V ∧ D) ∨ (¬A ∧ B ∧ ¬G) ∨ (B ∧ ¬V ∧ D) := 
by {
    -- Here would go the proof steps combining the axioms and conditions logically
    sorry
}

end NUMINAMATH_GPT_alex_plays_with_friends_l1048_104857


namespace NUMINAMATH_GPT_conic_section_is_parabola_l1048_104871

theorem conic_section_is_parabola (x y : ℝ) : y^4 - 16 * x^2 = 2 * y^2 - 64 → ((y^2 - 1)^2 = 16 * x^2 - 63) ∧ (∃ k : ℝ, y^2 = 4 * k * x + 1) :=
sorry

end NUMINAMATH_GPT_conic_section_is_parabola_l1048_104871


namespace NUMINAMATH_GPT_values_satisfying_ggx_eq_gx_l1048_104886

def g (x : ℝ) : ℝ := x^2 - 4 * x

theorem values_satisfying_ggx_eq_gx (x : ℝ) :
  g (g x) = g x ↔ x = 0 ∨ x = 1 ∨ x = 3 ∨ x = 4 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_values_satisfying_ggx_eq_gx_l1048_104886


namespace NUMINAMATH_GPT_park_area_l1048_104843

-- Definitions for the conditions
def length (breadth : ℕ) : ℕ := 4 * breadth
def perimeter (length breadth : ℕ) : ℕ := 2 * (length + breadth)

-- Formal statement of the proof problem
theorem park_area (breadth : ℕ) (h1 : perimeter (length breadth) breadth = 1600) : 
  let len := length breadth
  len * breadth = 102400 := 
by 
  sorry

end NUMINAMATH_GPT_park_area_l1048_104843


namespace NUMINAMATH_GPT_find_missing_number_l1048_104817

def average (l : List ℕ) : ℚ := l.sum / l.length

theorem find_missing_number : 
  ∃ x : ℕ, 
    average [744, 745, 747, 748, 749, 752, 752, 753, 755, x] = 750 :=
sorry

end NUMINAMATH_GPT_find_missing_number_l1048_104817


namespace NUMINAMATH_GPT_neg_p_equiv_l1048_104839

theorem neg_p_equiv (p : Prop) : 
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end NUMINAMATH_GPT_neg_p_equiv_l1048_104839


namespace NUMINAMATH_GPT_automotive_test_l1048_104805

theorem automotive_test (D T1 T2 T3 T_total : ℕ) (H1 : 3 * D = 180) 
  (H2 : T1 = D / 4) (H3 : T2 = D / 5) (H4 : T3 = D / 6)
  (H5 : T_total = T1 + T2 + T3) : T_total = 37 :=
  sorry

end NUMINAMATH_GPT_automotive_test_l1048_104805


namespace NUMINAMATH_GPT_all_numbers_equal_l1048_104826

theorem all_numbers_equal 
  (x : Fin 2007 → ℝ)
  (h : ∀ (I : Finset (Fin 2007)), I.card = 7 → ∃ (J : Finset (Fin 2007)), J.card = 11 ∧ 
  (1 / 7 : ℝ) * I.sum x = (1 / 11 : ℝ) * J.sum x) :
  ∃ c : ℝ, ∀ i : Fin 2007, x i = c :=
by sorry

end NUMINAMATH_GPT_all_numbers_equal_l1048_104826


namespace NUMINAMATH_GPT_repeated_three_digit_divisible_l1048_104884

theorem repeated_three_digit_divisible (μ : ℕ) (h : 100 ≤ μ ∧ μ < 1000) :
  ∃ k : ℕ, (1000 * μ + μ) = k * 7 * 11 * 13 := by
sorry

end NUMINAMATH_GPT_repeated_three_digit_divisible_l1048_104884


namespace NUMINAMATH_GPT_valid_parametrizations_l1048_104882

-- Define the line as a function
def line (x : ℝ) : ℝ := -2 * x + 7

-- Define vectors and their properties
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def on_line (v : Vector2D) : Prop :=
  v.y = line v.x

def direction_vector (v1 v2 : Vector2D) : Vector2D :=
  ⟨v2.x - v1.x, v2.y - v1.y⟩

def is_multiple (v1 v2 : Vector2D) : Prop :=
  ∃ k : ℝ, v2.x = k * v1.x ∧ v2.y = k * v1.y

-- Define the given parameterizations
def param_A (t : ℝ) : Vector2D := ⟨0 + t * 5, 7 + t * 10⟩
def param_B (t : ℝ) : Vector2D := ⟨2 + t * 1, 3 + t * -2⟩
def param_C (t : ℝ) : Vector2D := ⟨7 + t * 4, 0 + t * -8⟩
def param_D (t : ℝ) : Vector2D := ⟨-1 + t * 2, 9 + t * 4⟩
def param_E (t : ℝ) : Vector2D := ⟨3 + t * 2, 1 + t * 0⟩

-- Define the theorem
theorem valid_parametrizations :
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨0, 7⟩ (param_A t)) ∧ on_line (param_A t) → False) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨2, 3⟩ (param_B t)) ∧ on_line (param_B t)) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨7, 0⟩ (param_C t)) ∧ on_line (param_C t)) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨-1, 9⟩ (param_D t)) ∧ on_line (param_D t) → False) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨3, 1⟩ (param_E t)) ∧ on_line (param_E t) → False) :=
by
  sorry

end NUMINAMATH_GPT_valid_parametrizations_l1048_104882


namespace NUMINAMATH_GPT_lcm_of_54_96_120_150_l1048_104813

theorem lcm_of_54_96_120_150 : Nat.lcm 54 (Nat.lcm 96 (Nat.lcm 120 150)) = 21600 := by
  sorry

end NUMINAMATH_GPT_lcm_of_54_96_120_150_l1048_104813


namespace NUMINAMATH_GPT_square_floor_tile_count_l1048_104835

/-
A square floor is tiled with congruent square tiles.
The tiles on the two diagonals of the floor are black.
If there are 101 black tiles, then the total number of tiles is 2601.
-/
theorem square_floor_tile_count  
  (s : ℕ) 
  (hs_odd : s % 2 = 1)  -- s is odd
  (h_black_tile_count : 2 * s - 1 = 101) 
  : s^2 = 2601 := 
by 
  sorry

end NUMINAMATH_GPT_square_floor_tile_count_l1048_104835


namespace NUMINAMATH_GPT_angle_between_generatrix_and_base_of_cone_l1048_104821

theorem angle_between_generatrix_and_base_of_cone (r R H : ℝ) (α : ℝ)
  (h_cylinder_height : H = 2 * R)
  (h_total_surface_area : 2 * Real.pi * r * H + 2 * Real.pi * r^2 = Real.pi * R^2) :
  α = Real.arctan (2 * (4 + Real.sqrt 6) / 5) :=
sorry

end NUMINAMATH_GPT_angle_between_generatrix_and_base_of_cone_l1048_104821


namespace NUMINAMATH_GPT_find_prime_numbers_of_form_p_p_plus_1_l1048_104830

def has_at_most_19_digits (n : ℕ) : Prop := n < 10^19

theorem find_prime_numbers_of_form_p_p_plus_1 :
  {n : ℕ | ∃ p : ℕ, n = p^p + 1 ∧ has_at_most_19_digits n ∧ Nat.Prime n} = {2, 5, 257} :=
by
  sorry

end NUMINAMATH_GPT_find_prime_numbers_of_form_p_p_plus_1_l1048_104830


namespace NUMINAMATH_GPT_number_of_positive_integers_l1048_104846

theorem number_of_positive_integers (n : ℕ) (hpos : 0 < n) (h : 24 - 6 * n ≥ 12) : n = 1 ∨ n = 2 :=
sorry

end NUMINAMATH_GPT_number_of_positive_integers_l1048_104846


namespace NUMINAMATH_GPT_determinant_identity_l1048_104896

variable (x y z w : ℝ)
variable (h1 : x * w - y * z = -3)

theorem determinant_identity :
  (x + z) * w - (y + w) * z = -3 :=
by sorry

end NUMINAMATH_GPT_determinant_identity_l1048_104896


namespace NUMINAMATH_GPT_fill_time_without_leak_l1048_104887

theorem fill_time_without_leak (F L : ℝ)
  (h1 : (F - L) * 12 = 1)
  (h2 : L * 24 = 1) :
  1 / F = 8 := 
sorry

end NUMINAMATH_GPT_fill_time_without_leak_l1048_104887


namespace NUMINAMATH_GPT_fraction_meaningful_iff_l1048_104890

theorem fraction_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_iff_l1048_104890


namespace NUMINAMATH_GPT_problem_1_problem_2_l1048_104873

theorem problem_1 (a b c: ℝ) (h1: a > 0) (h2: b > 0) :
  a^3 + b^3 ≥ a^2 * b + a * b^2 :=
by
  sorry

theorem problem_2 (a b c: ℝ) (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: a + b + c = 1) :
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1048_104873


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1048_104889

theorem sufficient_not_necessary_condition (x : ℝ) : (|x - 1/2| < 1/2) → (x^3 < 1) ∧ ¬(x^3 < 1) → (|x - 1/2| < 1/2) :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1048_104889


namespace NUMINAMATH_GPT_cos_sq_half_diff_eq_csquared_over_a2_b2_l1048_104807

theorem cos_sq_half_diff_eq_csquared_over_a2_b2
  (a b c α β : ℝ)
  (h1 : a^2 + b^2 ≠ 0)
  (h2 : a * (Real.cos α) + b * (Real.sin α) = c)
  (h3 : a * (Real.cos β) + b * (Real.sin β) = c)
  (h4 : ∀ k : ℤ, α ≠ β + 2 * k * Real.pi) :
  Real.cos (α - β) / 2 = c^2 / (a^2 + b^2) :=
by
  sorry

end NUMINAMATH_GPT_cos_sq_half_diff_eq_csquared_over_a2_b2_l1048_104807


namespace NUMINAMATH_GPT_sphere_radius_five_times_surface_area_l1048_104802

theorem sphere_radius_five_times_surface_area (R : ℝ) (h₁ : (4 * π * R^3 / 3) = 5 * (4 * π * R^2)) : R = 15 :=
sorry

end NUMINAMATH_GPT_sphere_radius_five_times_surface_area_l1048_104802


namespace NUMINAMATH_GPT_carlotta_performance_time_l1048_104851

theorem carlotta_performance_time :
  ∀ (s p t : ℕ),  -- s for singing, p for practicing, t for tantrums
  (∀ (n : ℕ), p = 3 * n ∧ t = 5 * n) →
  s = 6 →
  (s + p + t) = 54 :=
by 
  intros s p t h1 h2
  rcases h1 1 with ⟨h3, h4⟩
  sorry

end NUMINAMATH_GPT_carlotta_performance_time_l1048_104851


namespace NUMINAMATH_GPT_greatest_possible_sum_example_sum_case_l1048_104808

/-- For integers x and y such that x^2 + y^2 = 50, the greatest possible value of x + y is 10. -/
theorem greatest_possible_sum (x y : ℤ) (h : x^2 + y^2 = 50) : x + y ≤ 10 :=
sorry

-- Auxiliary theorem to state that 10 can be achieved
theorem example_sum_case : ∃ (x y : ℤ), x^2 + y^2 = 50 ∧ x + y = 10 :=
sorry

end NUMINAMATH_GPT_greatest_possible_sum_example_sum_case_l1048_104808


namespace NUMINAMATH_GPT_lisa_eggs_total_l1048_104842

def children_mon_tue := 4 * 2 * 2
def husband_mon_tue := 3 * 2 
def lisa_mon_tue := 2 * 2
def total_mon_tue := children_mon_tue + husband_mon_tue + lisa_mon_tue

def children_wed := 4 * 3
def husband_wed := 4
def lisa_wed := 3
def total_wed := children_wed + husband_wed + lisa_wed

def children_thu := 4 * 1
def husband_thu := 2
def lisa_thu := 1
def total_thu := children_thu + husband_thu + lisa_thu

def children_fri := 4 * 2
def husband_fri := 3
def lisa_fri := 2
def total_fri := children_fri + husband_fri + lisa_fri

def total_week := total_mon_tue + total_wed + total_thu + total_fri

def weeks_per_year := 52
def yearly_eggs := total_week * weeks_per_year

def children_holidays := 4 * 2 * 8
def husband_holidays := 2 * 8
def lisa_holidays := 2 * 8
def total_holidays := children_holidays + husband_holidays + lisa_holidays

def total_annual_eggs := yearly_eggs + total_holidays

theorem lisa_eggs_total : total_annual_eggs = 3476 := by
  sorry

end NUMINAMATH_GPT_lisa_eggs_total_l1048_104842


namespace NUMINAMATH_GPT_remainder_of_product_mod_17_l1048_104837

theorem remainder_of_product_mod_17 :
  (2005 * 2006 * 2007 * 2008 * 2009) % 17 = 0 :=
sorry

end NUMINAMATH_GPT_remainder_of_product_mod_17_l1048_104837


namespace NUMINAMATH_GPT_power_function_value_at_3_l1048_104858

theorem power_function_value_at_3
  (f : ℝ → ℝ)
  (h1 : ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α)
  (h2 : f 2 = 1 / 4) :
  f 3 = 1 / 9 := 
sorry

end NUMINAMATH_GPT_power_function_value_at_3_l1048_104858


namespace NUMINAMATH_GPT_max_area_of_rectangle_l1048_104812

theorem max_area_of_rectangle (x y : ℝ) (h : 2 * (x + y) = 60) : x * y ≤ 225 :=
by sorry

end NUMINAMATH_GPT_max_area_of_rectangle_l1048_104812


namespace NUMINAMATH_GPT_part_one_part_two_l1048_104897

noncomputable def f (x a : ℝ) : ℝ :=
  Real.log (1 + x) + a * Real.cos x

noncomputable def g (x : ℝ) : ℝ :=
  f x 2 - 1 / (1 + x)

theorem part_one (a : ℝ) : 
  (∀ x, f x a = Real.log (1 + x) + a * Real.cos x) ∧ 
  f 0 a = 2 ∧ 
  (∀ x, x + f (0:ℝ) a = x + 2) → 
  a = 2 := 
sorry

theorem part_two : 
  (∀ x, g x = Real.log (1 + x) + 2 * Real.cos x - 1 / (1 + x)) →
  (∃ y, -1 < y ∧ y < (Real.pi / 2) ∧ g y = 0) ∧ 
  (∀ x, -1 < x ∧ x < (Real.pi / 2) → g x ≠ 0) →
  (∃! y, -1 < y ∧ y < (Real.pi / 2) ∧ g y = 0) :=
sorry

end NUMINAMATH_GPT_part_one_part_two_l1048_104897


namespace NUMINAMATH_GPT_number_is_450064_l1048_104891

theorem number_is_450064 : (45 * 10000 + 64) = 450064 :=
by
  sorry

end NUMINAMATH_GPT_number_is_450064_l1048_104891


namespace NUMINAMATH_GPT_natalie_list_count_l1048_104879

theorem natalie_list_count : ∀ n : ℕ, (15 ≤ n ∧ n ≤ 225) → ((225 - 15 + 1) = 211) :=
by
  intros n h
  sorry

end NUMINAMATH_GPT_natalie_list_count_l1048_104879


namespace NUMINAMATH_GPT_fifth_term_power_of_five_sequence_l1048_104850

theorem fifth_term_power_of_five_sequence : 5^0 + 5^1 + 5^2 + 5^3 + 5^4 = 781 := 
by
sorry

end NUMINAMATH_GPT_fifth_term_power_of_five_sequence_l1048_104850


namespace NUMINAMATH_GPT_total_animals_made_it_to_shore_l1048_104832

def boat (total_sheep total_cows total_dogs sheep_drowned cows_drowned dogs_saved : Nat) : Prop :=
  cows_drowned = sheep_drowned * 2 ∧
  dogs_saved = total_dogs ∧
  total_sheep + total_cows + total_dogs - sheep_drowned - cows_drowned = 35

theorem total_animals_made_it_to_shore :
  boat 20 10 14 3 6 14 :=
by
  sorry

end NUMINAMATH_GPT_total_animals_made_it_to_shore_l1048_104832


namespace NUMINAMATH_GPT_trash_cans_street_count_l1048_104855

theorem trash_cans_street_count (S B : ℕ) (h1 : B = 2 * S) (h2 : S + B = 42) : S = 14 :=
by
  sorry

end NUMINAMATH_GPT_trash_cans_street_count_l1048_104855


namespace NUMINAMATH_GPT_sampling_methods_correct_l1048_104875

-- Definitions of the conditions:
def is_simple_random_sampling (method : String) : Prop := 
  method = "random selection of 24 students by the student council"

def is_systematic_sampling (method : String) : Prop := 
  method = "selection of students numbered from 001 to 240 whose student number ends in 3"

-- The equivalent math proof problem:
theorem sampling_methods_correct :
  is_simple_random_sampling "random selection of 24 students by the student council" ∧
  is_systematic_sampling "selection of students numbered from 001 to 240 whose student number ends in 3" :=
by
  sorry

end NUMINAMATH_GPT_sampling_methods_correct_l1048_104875


namespace NUMINAMATH_GPT_rectangle_clear_area_l1048_104863

theorem rectangle_clear_area (EF FG : ℝ)
  (radius_E radius_F radius_G radius_H : ℝ) : 
  EF = 4 → FG = 6 → 
  radius_E = 2 → radius_F = 3 → radius_G = 1.5 → radius_H = 2.5 → 
  abs ((EF * FG) - (π * radius_E^2 / 4 + π * radius_F^2 / 4 + π * radius_G^2 / 4 + π * radius_H^2 / 4)) - 7.14 < 0.5 :=
by sorry

end NUMINAMATH_GPT_rectangle_clear_area_l1048_104863


namespace NUMINAMATH_GPT_min_value_of_expr_l1048_104860

theorem min_value_of_expr (x : ℝ) (h : x > 2) : ∃ y, (y = x + 4 / (x - 2)) ∧ y ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expr_l1048_104860


namespace NUMINAMATH_GPT_ravi_work_alone_days_l1048_104820

theorem ravi_work_alone_days (R : ℝ) (h1 : 1 / 75 + 1 / R = 1 / 30) : R = 50 :=
sorry

end NUMINAMATH_GPT_ravi_work_alone_days_l1048_104820


namespace NUMINAMATH_GPT_intersection_of_sets_l1048_104856

noncomputable def setA : Set ℝ := { x | |x - 2| ≤ 3 }
noncomputable def setB : Set ℝ := { y | ∃ x : ℝ, y = 1 - x^2 }

theorem intersection_of_sets :
  setA ∩ setB = { z : ℝ | z ∈ [-1, 1] } :=
sorry

end NUMINAMATH_GPT_intersection_of_sets_l1048_104856


namespace NUMINAMATH_GPT_odd_function_values_l1048_104847

noncomputable def f (a b x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_values (a b : ℝ) :
  (∀ x : ℝ, f a b (-x) = -f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_odd_function_values_l1048_104847


namespace NUMINAMATH_GPT_ranking_sequences_l1048_104840

theorem ranking_sequences
    (A D B E C : Type)
    (h_no_ties : ∀ (X Y : Type), X ≠ Y)
    (h_games : (W1 = A ∨ W1 = D) ∧ (W2 = B ∨ W2 = E) ∧ (W3 = W1 ∨ W3 = C)) :
  ∃! (n : ℕ), n = 48 := 
sorry

end NUMINAMATH_GPT_ranking_sequences_l1048_104840


namespace NUMINAMATH_GPT_kittens_count_l1048_104831

def cats_taken_in : ℕ := 12
def cats_initial : ℕ := cats_taken_in / 2
def cats_post_adoption : ℕ := cats_taken_in + cats_initial - 3
def cats_now : ℕ := 19

theorem kittens_count :
  ∃ k : ℕ, cats_post_adoption + k - 1 = cats_now :=
by
  use 5
  sorry

end NUMINAMATH_GPT_kittens_count_l1048_104831


namespace NUMINAMATH_GPT_copper_tin_ratio_l1048_104869

theorem copper_tin_ratio 
    (w1 w2 w_new : ℝ) 
    (r1_copper r1_tin r2_copper r2_tin : ℝ) 
    (r_new_copper r_new_tin : ℝ)
    (pure_copper : ℝ)
    (h1 : w1 = 10)
    (h2 : w2 = 16)
    (h3 : r1_copper = 4 / 5 * w1)
    (h4 : r1_tin = 1 / 5 * w1)
    (h5 : r2_copper = 1 / 4 * w2)
    (h6 : r2_tin = 3 / 4 * w2)
    (h7 : r_new_copper = r1_copper + r2_copper + pure_copper)
    (h8 : r_new_tin = r1_tin + r2_tin)
    (h9 : w_new = 35)
    (h10 : r_new_copper + r_new_tin + pure_copper = w_new)
    (h11 : pure_copper = 9) :
    r_new_copper / r_new_tin = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_copper_tin_ratio_l1048_104869


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1048_104804

theorem necessary_but_not_sufficient_condition (x : ℝ) : |x - 1| < 2 → -3 < x ∧ x < 3 :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1048_104804


namespace NUMINAMATH_GPT_reflect_parabola_x_axis_l1048_104810

theorem reflect_parabola_x_axis (x : ℝ) (a b c : ℝ) :
  (∀ y : ℝ, y = x^2 + x - 2 → -y = x^2 + x - 2) →
  (∀ y : ℝ, -y = x^2 + x - 2 → y = -x^2 - x + 2) :=
by
  intros h₁ h₂
  intro y
  sorry

end NUMINAMATH_GPT_reflect_parabola_x_axis_l1048_104810


namespace NUMINAMATH_GPT_find_k_l1048_104829

-- Conditions
def t : ℕ := 6
def is_nonzero_digit (n : ℕ) : Prop := n > 0 ∧ n < 10

-- Given these conditions, we need to prove that k = 9
theorem find_k (k t : ℕ) (h1 : t = 6) (h2 : is_nonzero_digit k) (h3 : is_nonzero_digit t) :
    (8 * 10^2 + k * 10 + 8) + (k * 10^2 + 8 * 10 + 8) - 16 * t * 10^0 * 6 = (9 * 10 + 8) + (9 * 10^2 + 8 * 10 + 8) - (16 * 6 * 10^1 + 6) → k = 9 := 
sorry

end NUMINAMATH_GPT_find_k_l1048_104829


namespace NUMINAMATH_GPT_problem_condition_l1048_104854

noncomputable def f (x b : ℝ) := Real.exp x * (x - b)
noncomputable def f_prime (x b : ℝ) := Real.exp x * (x - b + 1)
noncomputable def g (x : ℝ) := (x^2 + 2*x) / (x + 1)

theorem problem_condition (b : ℝ) :
  (∃ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x b + x * f_prime x b > 0) → b < 8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_condition_l1048_104854


namespace NUMINAMATH_GPT_exists_a_not_divisible_l1048_104811

theorem exists_a_not_divisible (p : ℕ) (hp_prime : Prime p) (hp_ge_5 : p ≥ 5) :
  ∃ a : ℕ, 1 ≤ a ∧ a ≤ p - 2 ∧ (¬ (p^2 ∣ (a^(p-1) - 1)) ∧ ¬ (p^2 ∣ ((a+1)^(p-1) - 1))) :=
  sorry

end NUMINAMATH_GPT_exists_a_not_divisible_l1048_104811


namespace NUMINAMATH_GPT_lawnmower_blade_cost_l1048_104849

theorem lawnmower_blade_cost (x : ℕ) : 4 * x + 7 = 39 → x = 8 :=
by
  sorry

end NUMINAMATH_GPT_lawnmower_blade_cost_l1048_104849


namespace NUMINAMATH_GPT_proof_problem_l1048_104801

noncomputable def a : ℚ := 2 / 3
noncomputable def b : ℚ := - 3 / 2
noncomputable def n : ℕ := 2023

theorem proof_problem :
  (a ^ n) * (b ^ n) = -1 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1048_104801


namespace NUMINAMATH_GPT_final_price_hat_final_price_tie_l1048_104865

theorem final_price_hat (initial_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) 
    (h_initial : initial_price = 20) 
    (h_first : first_discount = 0.25) 
    (h_second : second_discount = 0.20) : 
    initial_price * (1 - first_discount) * (1 - second_discount) = 12 := 
by 
  rw [h_initial, h_first, h_second]
  norm_num

theorem final_price_tie (initial_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) 
    (t_initial : initial_price = 15) 
    (t_first : first_discount = 0.10) 
    (t_second : second_discount = 0.30) : 
    initial_price * (1 - first_discount) * (1 - second_discount) = 9.45 := 
by 
  rw [t_initial, t_first, t_second]
  norm_num

end NUMINAMATH_GPT_final_price_hat_final_price_tie_l1048_104865


namespace NUMINAMATH_GPT_union_A_B_l1048_104868

open Set

def A := {x : ℝ | x * (x - 2) < 3}
def B := {x : ℝ | 5 / (x + 1) ≥ 1}
def U := {x : ℝ | -1 < x ∧ x ≤ 4}

theorem union_A_B : A ∪ B = U := 
sorry

end NUMINAMATH_GPT_union_A_B_l1048_104868


namespace NUMINAMATH_GPT_ratio_of_flour_to_eggs_l1048_104893

theorem ratio_of_flour_to_eggs (F E : ℕ) (h1 : E = 60) (h2 : F + E = 90) : F / 30 = 1 ∧ E / 30 = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_flour_to_eggs_l1048_104893


namespace NUMINAMATH_GPT_problem_1_1_eval_l1048_104874

noncomputable def E (a b c : ℝ) : ℝ :=
  let A := (1/a - 1/(b+c))/(1/a + 1/(b+c))
  let B := 1 + (b^2 + c^2 - a^2)/(2*b*c)
  let C := (a - b - c)/(a * b * c)
  (A * B) / C

theorem problem_1_1_eval :
  E 0.02 (-11.05) 1.07 = 0.1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_problem_1_1_eval_l1048_104874


namespace NUMINAMATH_GPT_lcm_first_ten_numbers_l1048_104876

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end NUMINAMATH_GPT_lcm_first_ten_numbers_l1048_104876


namespace NUMINAMATH_GPT_two_positive_numbers_inequality_three_positive_numbers_am_gm_l1048_104885

theorem two_positive_numbers_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  x^3 + y^3 ≥ x^2 * y + x * y^2 ∧ (x = y ↔ x^3 + y^3 = x^2 * y + x * y^2) := by
sorry

theorem three_positive_numbers_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) / 3 ≥ (a * b * c)^(1/3) ∧ (a = b ∧ b = c ↔ (a + b + c) / 3 = (a * b * c)^(1/3)) := by
sorry

end NUMINAMATH_GPT_two_positive_numbers_inequality_three_positive_numbers_am_gm_l1048_104885


namespace NUMINAMATH_GPT_arithmetic_progression_correct_l1048_104892

noncomputable def nth_term_arithmetic_progression (n : ℕ) : ℝ :=
  4.2 * n + 9.3

def recursive_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  a 1 = 13.5 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = a n + 4.2

theorem arithmetic_progression_correct (n : ℕ) :
  (nth_term_arithmetic_progression n = 4.2 * n + 9.3) ∧
  ∀ (a : ℕ → ℝ), recursive_arithmetic_progression a → a n = 4.2 * n + 9.3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_correct_l1048_104892
