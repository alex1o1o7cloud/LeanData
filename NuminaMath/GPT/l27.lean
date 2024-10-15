import Mathlib

namespace NUMINAMATH_GPT_find_d_and_r_l27_2728

theorem find_d_and_r (d r : ℤ)
  (h1 : 1210 % d = r)
  (h2 : 1690 % d = r)
  (h3 : 2670 % d = r) :
  d - 4 * r = -20 := sorry

end NUMINAMATH_GPT_find_d_and_r_l27_2728


namespace NUMINAMATH_GPT_minimum_value_of_expression_l27_2771

open Real

noncomputable def f (x y z : ℝ) : ℝ := (x + 2 * y) / (x * y * z)

theorem minimum_value_of_expression :
  ∀ (x y z : ℝ),
    x > 0 → y > 0 → z > 0 →
    x + y + z = 1 →
    x = 2 * y →
    f x y z = 8 :=
by
  intro x y z x_pos y_pos z_pos h_sum h_xy
  sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l27_2771


namespace NUMINAMATH_GPT_find_m_l27_2796

theorem find_m (m : ℝ) :
  (∃ x a : ℝ, |x - 1| - |x + m| ≥ a ∧ a ≤ 5) ↔ (m = 4 ∨ m = -6) :=
by
  sorry

end NUMINAMATH_GPT_find_m_l27_2796


namespace NUMINAMATH_GPT_jeremy_money_ratio_l27_2778

theorem jeremy_money_ratio :
  let cost_computer := 3000
  let cost_accessories := 0.10 * cost_computer
  let money_left := 2700
  let total_spent := cost_computer + cost_accessories
  let money_before_purchase := total_spent + money_left
  (money_before_purchase / cost_computer) = 2 := by
  sorry

end NUMINAMATH_GPT_jeremy_money_ratio_l27_2778


namespace NUMINAMATH_GPT_translate_parabola_upwards_l27_2775

theorem translate_parabola_upwards (x y : ℝ) (h : y = x^2) : y + 1 = x^2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_translate_parabola_upwards_l27_2775


namespace NUMINAMATH_GPT_cyclist_wait_time_l27_2729

theorem cyclist_wait_time
  (hiker_speed : ℝ)
  (hiker_speed_pos : hiker_speed = 4)
  (cyclist_speed : ℝ)
  (cyclist_speed_pos : cyclist_speed = 24)
  (waiting_time_minutes : ℝ)
  (waiting_time_minutes_pos : waiting_time_minutes = 5) :
  (waiting_time_minutes / 60) * cyclist_speed = 2 →
  (2 / hiker_speed) * 60 = 30 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cyclist_wait_time_l27_2729


namespace NUMINAMATH_GPT_area_inequality_l27_2746

open Real

variables (AB CD AD BC S : ℝ) (alpha beta : ℝ)
variables (α_pos : 0 < α ∧ α < π) (β_pos : 0 < β ∧ β < π)
variables (S_pos : 0 < S) (H1 : ConvexQuadrilateral AB CD AD BC S)

theorem area_inequality :
  AB * CD * sin α + AD * BC * sin β ≤ 2 * S ∧ 2 * S ≤ AB * CD + AD * BC :=
sorry

end NUMINAMATH_GPT_area_inequality_l27_2746


namespace NUMINAMATH_GPT_total_students_in_faculty_l27_2715

theorem total_students_in_faculty :
  (let sec_year_num := 230
   let sec_year_auto := 423
   let both_subj := 134
   let sec_year_total := 0.80
   let at_least_one_subj := sec_year_num + sec_year_auto - both_subj
   ∃ (T : ℝ), sec_year_total * T = at_least_one_subj ∧ T = 649) := by
  sorry

end NUMINAMATH_GPT_total_students_in_faculty_l27_2715


namespace NUMINAMATH_GPT_root_product_minus_sums_l27_2710

variable {b c : ℝ}

theorem root_product_minus_sums
  (h1 : 3 * b^2 + 5 * b - 2 = 0)
  (h2 : 3 * c^2 + 5 * c - 2 = 0)
  : (b - 1) * (c - 1) = 2 := 
by
  sorry

end NUMINAMATH_GPT_root_product_minus_sums_l27_2710


namespace NUMINAMATH_GPT_pinedale_bus_speed_l27_2767

theorem pinedale_bus_speed 
  (stops_every_minutes : ℕ)
  (num_stops : ℕ)
  (distance_km : ℕ)
  (time_per_stop_minutes : stops_every_minutes = 5)
  (dest_stops : num_stops = 8)
  (dest_distance : distance_km = 40) 
  : (distance_km / (num_stops * stops_every_minutes / 60)) = 60 := 
by
  sorry

end NUMINAMATH_GPT_pinedale_bus_speed_l27_2767


namespace NUMINAMATH_GPT_find_m_n_l27_2722

theorem find_m_n (m n : ℕ) (hmn : m + 6 < n + 4)
  (median_cond : ((m + 2 + m + 6 + n + 4 + n + 5) / 7) = n + 2)
  (mean_cond : ((m + (m + 2) + (m + 6) + (n + 4) + (n + 5) + (2 * n - 1) + (2 * n + 2)) / 7) = n + 2) :
  m + n = 10 :=
sorry

end NUMINAMATH_GPT_find_m_n_l27_2722


namespace NUMINAMATH_GPT_binary_arithmetic_l27_2754

theorem binary_arithmetic 
  : (0b10110 + 0b1011 - 0b11100 + 0b11101 = 0b100010) :=
by
  sorry

end NUMINAMATH_GPT_binary_arithmetic_l27_2754


namespace NUMINAMATH_GPT_evaluate_expression_l27_2700

theorem evaluate_expression : (-(18 / 3 * 12 - 80 + 4 * 12)) ^ 2 = 1600 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l27_2700


namespace NUMINAMATH_GPT_correct_operation_l27_2720

theorem correct_operation :
  (∀ (a : ℤ), 3 * a + 2 * a ≠ 5 * a ^ 2) ∧
  (∀ (a : ℤ), a ^ 6 / a ^ 2 ≠ a ^ 3) ∧
  (∀ (a : ℤ), (-3 * a ^ 3) ^ 2 = 9 * a ^ 6) ∧
  (∀ (a : ℤ), (a + 2) ^ 2 ≠ a ^ 2 + 4) := 
by
  sorry

end NUMINAMATH_GPT_correct_operation_l27_2720


namespace NUMINAMATH_GPT_min_tip_percentage_l27_2779

noncomputable def meal_cost : ℝ := 37.25
noncomputable def total_paid : ℝ := 40.975
noncomputable def tip_percentage (P : ℝ) : Prop := P > 0 ∧ P < 15 ∧ (meal_cost + (P/100) * meal_cost = total_paid)

theorem min_tip_percentage : ∃ P : ℝ, tip_percentage P ∧ P = 10 := by
  sorry

end NUMINAMATH_GPT_min_tip_percentage_l27_2779


namespace NUMINAMATH_GPT_intersection_sums_l27_2748

theorem intersection_sums (x1 x2 x3 y1 y2 y3 : ℝ) (h1 : y1 = x1^3 - 6 * x1 + 4)
  (h2 : y2 = x2^3 - 6 * x2 + 4) (h3 : y3 = x3^3 - 6 * x3 + 4)
  (h4 : x1 + 3 * y1 = 3) (h5 : x2 + 3 * y2 = 3) (h6 : x3 + 3 * y3 = 3) :
  x1 + x2 + x3 = 0 ∧ y1 + y2 + y3 = 3 := 
by
  sorry

end NUMINAMATH_GPT_intersection_sums_l27_2748


namespace NUMINAMATH_GPT_price_of_ice_cream_l27_2705

theorem price_of_ice_cream (x : ℝ) :
  (225 * x + 125 * 0.52 = 200) → (x = 0.60) :=
sorry

end NUMINAMATH_GPT_price_of_ice_cream_l27_2705


namespace NUMINAMATH_GPT_leah_coins_value_l27_2785

theorem leah_coins_value : 
  ∃ (p n : ℕ), p + n = 15 ∧ n + 1 = p ∧ 5 * n + 1 * p = 43 := 
by
  sorry

end NUMINAMATH_GPT_leah_coins_value_l27_2785


namespace NUMINAMATH_GPT_solve_inequality_l27_2707

theorem solve_inequality {x : ℝ} : (x^2 - 5 * x + 6 ≤ 0) → (2 ≤ x ∧ x ≤ 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_inequality_l27_2707


namespace NUMINAMATH_GPT_triangle_transform_same_l27_2784

def Point := ℝ × ℝ

def reflect_x (p : Point) : Point :=
(p.1, -p.2)

def rotate_180 (p : Point) : Point :=
(-p.1, -p.2)

def reflect_y (p : Point) : Point :=
(-p.1, p.2)

def transform (p : Point) : Point :=
reflect_y (rotate_180 (reflect_x p))

theorem triangle_transform_same (A B C : Point) :
A = (2, 1) → B = (4, 1) → C = (2, 3) →
(transform A = (2, 1) ∧ transform B = (4, 1) ∧ transform C = (2, 3)) :=
by
  intros
  sorry

end NUMINAMATH_GPT_triangle_transform_same_l27_2784


namespace NUMINAMATH_GPT_sum_of_squares_and_product_l27_2766

theorem sum_of_squares_and_product (x y : ℤ) 
  (h1 : x^2 + y^2 = 290) 
  (h2 : x * y = 96) :
  x + y = 22 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_and_product_l27_2766


namespace NUMINAMATH_GPT_largest_possible_value_of_b_l27_2732

theorem largest_possible_value_of_b (b : ℚ) (h : (3 * b + 4) * (b - 2) = 9 * b) : b ≤ 4 :=
sorry

end NUMINAMATH_GPT_largest_possible_value_of_b_l27_2732


namespace NUMINAMATH_GPT_simplify_fraction_l27_2790

theorem simplify_fraction (a b : ℝ) :
  ( (3 * b) / (2 * a^2) )^3 = 27 * b^3 / (8 * a^6) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l27_2790


namespace NUMINAMATH_GPT_volume_calculation_l27_2724

noncomputable def enclosedVolume : Real :=
  let f (x y z : Real) : Real := x^2016 + y^2016 + z^2
  let V : Real := 360
  V

theorem volume_calculation : enclosedVolume = 360 :=
by
  sorry

end NUMINAMATH_GPT_volume_calculation_l27_2724


namespace NUMINAMATH_GPT_total_expenditure_of_7_people_l27_2769

theorem total_expenditure_of_7_people :
  ∃ A : ℝ, 
    (6 * 11 + (A + 6) = 7 * A) ∧
    (6 * 11 = 66) ∧
    (∃ total : ℝ, total = 6 * 11 + (A + 6) ∧ total = 84) :=
by 
  sorry

end NUMINAMATH_GPT_total_expenditure_of_7_people_l27_2769


namespace NUMINAMATH_GPT_min_value_expression_l27_2727

-- Define the given problem conditions and statement
theorem min_value_expression :
  ∀ (x y : ℝ), 0 < x → 0 < y → 6 ≤ (y / x) + (16 * x / (2 * x + y)) :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l27_2727


namespace NUMINAMATH_GPT_larger_cookie_raisins_l27_2713

theorem larger_cookie_raisins : ∃ n r, 5 ≤ n ∧ n ≤ 10 ∧ (n - 1) * r + (r + 1) = 100 ∧ r + 1 = 12 :=
by
  sorry

end NUMINAMATH_GPT_larger_cookie_raisins_l27_2713


namespace NUMINAMATH_GPT_items_in_bags_l27_2719

def calculateWaysToPlaceItems (n_items : ℕ) (n_bags : ℕ) : ℕ :=
  sorry

theorem items_in_bags :
  calculateWaysToPlaceItems 5 3 = 41 :=
by sorry

end NUMINAMATH_GPT_items_in_bags_l27_2719


namespace NUMINAMATH_GPT_smallest_angle_product_l27_2701

-- Define an isosceles triangle with angle at B being the smallest angle
def isosceles_triangle (α : ℝ) : Prop :=
  α < 90 ∧ α = 180 / 7

-- Proof that the smallest angle multiplied by 6006 is 154440
theorem smallest_angle_product : 
  isosceles_triangle α → (180 / 7) * 6006 = 154440 :=
by
  intros
  sorry

end NUMINAMATH_GPT_smallest_angle_product_l27_2701


namespace NUMINAMATH_GPT_unit_prices_min_total_cost_l27_2743

-- Part (1): Proving the unit prices of ingredients A and B.
theorem unit_prices (x y : ℝ)
    (h₁ : x + y = 68)
    (h₂ : 5 * x + 3 * y = 280) :
    x = 38 ∧ y = 30 :=
by
  -- Sorry, proof not provided
  sorry

-- Part (2): Proving the minimum cost calculation.
theorem min_total_cost (m : ℝ)
    (h₁ : m + (36 - m) = 36)
    (h₂ : m ≥ 2 * (36 - m)) :
    (38 * m + 30 * (36 - m)) = 1272 :=
by
  -- Sorry, proof not provided
  sorry

end NUMINAMATH_GPT_unit_prices_min_total_cost_l27_2743


namespace NUMINAMATH_GPT_divisor_in_second_division_l27_2757

theorem divisor_in_second_division 
  (n : ℤ) 
  (h1 : (68 : ℤ) * 269 = n) 
  (d q : ℤ) 
  (h2 : n = d * q + 1) 
  (h3 : Prime 18291):
  d = 18291 := by
  sorry

end NUMINAMATH_GPT_divisor_in_second_division_l27_2757


namespace NUMINAMATH_GPT_find_y_l27_2764

-- Definitions of the given conditions
def is_straight_line (A B : Point) : Prop := 
  ∃ C D, A ≠ C ∧ B ≠ D

def angle (A B C : Point) : ℝ := sorry -- Assume angle is a function providing the angle in degrees

-- The proof problem statement
theorem find_y
  (A B C D X Y Z : Point)
  (hAB : is_straight_line A B)
  (hCD : is_straight_line C D)
  (hAXB : angle A X B = 180) 
  (hYXZ : angle Y X Z = 70)
  (hCYX : angle C Y X = 110) :
  angle X Y Z = 40 :=
sorry

end NUMINAMATH_GPT_find_y_l27_2764


namespace NUMINAMATH_GPT_time_spent_on_road_l27_2777

theorem time_spent_on_road (Total_time_hours Stop1_minutes Stop2_minutes Stop3_minutes : ℕ) 
  (h1: Total_time_hours = 13) 
  (h2: Stop1_minutes = 25) 
  (h3: Stop2_minutes = 10) 
  (h4: Stop3_minutes = 25) : 
  Total_time_hours - (Stop1_minutes + Stop2_minutes + Stop3_minutes) / 60 = 12 :=
by
  sorry

end NUMINAMATH_GPT_time_spent_on_road_l27_2777


namespace NUMINAMATH_GPT_scorpion_needs_10_millipedes_l27_2745

-- Define the number of segments required daily
def total_segments_needed : ℕ := 800

-- Define the segments already consumed by the scorpion
def segments_consumed : ℕ := 60 + 2 * (2 * 60)

-- Calculate the remaining segments needed
def remaining_segments_needed : ℕ := total_segments_needed - segments_consumed

-- Define the segments per millipede
def segments_per_millipede : ℕ := 50

-- Prove that the number of 50-segment millipedes to be eaten is 10
theorem scorpion_needs_10_millipedes 
  (h : remaining_segments_needed = 500) 
  (h2 : 500 / segments_per_millipede = 10) : 
  500 / segments_per_millipede = 10 := by
  sorry

end NUMINAMATH_GPT_scorpion_needs_10_millipedes_l27_2745


namespace NUMINAMATH_GPT_monotonic_has_at_most_one_solution_l27_2742

def monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y ∨ f y ≤ f x

theorem monotonic_has_at_most_one_solution (f : ℝ → ℝ) (c : ℝ) 
  (hf : monotonic f) : ∃! x : ℝ, f x = c :=
sorry

end NUMINAMATH_GPT_monotonic_has_at_most_one_solution_l27_2742


namespace NUMINAMATH_GPT_complex_is_1_sub_sqrt3i_l27_2726

open Complex

theorem complex_is_1_sub_sqrt3i (z : ℂ) (h : z * (1 + Real.sqrt 3 * I) = abs (1 + Real.sqrt 3 * I)) : z = 1 - Real.sqrt 3 * I :=
sorry

end NUMINAMATH_GPT_complex_is_1_sub_sqrt3i_l27_2726


namespace NUMINAMATH_GPT_school_sports_event_l27_2725

theorem school_sports_event (x y z : ℤ) (hx : x > y) (hy : y > z) (hz : z > 0)
  (points_A points_B points_E : ℤ) (ha : points_A = 22) (hb : points_B = 9) 
  (he : points_E = 9) (vault_winner_B : True) :
  ∃ n : ℕ, n = 5 ∧ second_place_grenade_throwing_team = 8^B :=
by
  sorry

end NUMINAMATH_GPT_school_sports_event_l27_2725


namespace NUMINAMATH_GPT_hcf_of_two_numbers_l27_2792

theorem hcf_of_two_numbers (A B H L : ℕ) (h1 : A * B = 1800) (h2 : L = 200) (h3 : A * B = H * L) : H = 9 :=
by
  sorry

end NUMINAMATH_GPT_hcf_of_two_numbers_l27_2792


namespace NUMINAMATH_GPT_find_functions_l27_2795

noncomputable def pair_of_functions_condition (f g : ℝ → ℝ) : Prop :=
∀ x y : ℝ, g (f (x + y)) = f x + (2 * x + y) * g y

theorem find_functions (f g : ℝ → ℝ) :
  pair_of_functions_condition f g →
  (∃ c d : ℝ, ∀ x : ℝ, f x = c * (x + d)) :=
sorry

end NUMINAMATH_GPT_find_functions_l27_2795


namespace NUMINAMATH_GPT_range_of_a_if_ineq_has_empty_solution_l27_2751

theorem range_of_a_if_ineq_has_empty_solution (a : ℝ) :
  (∀ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0) → -2 ≤ a ∧ a < 6/5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_if_ineq_has_empty_solution_l27_2751


namespace NUMINAMATH_GPT_relationship_between_D_and_A_l27_2717

variable {A B C D : Prop}

theorem relationship_between_D_and_A
  (h1 : A → B)
  (h2 : B → C)
  (h3 : D ↔ C) :
  (A → D) ∧ ¬(D → A) :=
by
sorry

end NUMINAMATH_GPT_relationship_between_D_and_A_l27_2717


namespace NUMINAMATH_GPT_number_of_integers_in_original_list_l27_2760

theorem number_of_integers_in_original_list :
  ∃ n m : ℕ, (m + 2) * (n + 1) = m * n + 15 ∧
             (m + 1) * (n + 2) = m * n + 16 ∧
             n = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_integers_in_original_list_l27_2760


namespace NUMINAMATH_GPT_alex_score_l27_2797

theorem alex_score 
    (n : ℕ) -- number of students
    (avg_19 : ℕ) -- average score of first 19 students
    (avg_20 : ℕ) -- average score of all 20 students
    (h_n : n = 20) -- number of students is 20
    (h_avg_19 : avg_19 = 75) -- average score of first 19 students is 75
    (h_avg_20 : avg_20 = 76) -- average score of all 20 students is 76
  : ∃ alex_score : ℕ, alex_score = 95 := 
by
    sorry

end NUMINAMATH_GPT_alex_score_l27_2797


namespace NUMINAMATH_GPT_ball_hits_ground_l27_2786

theorem ball_hits_ground (t : ℝ) :
  (∃ t, -(16 * t^2) + 32 * t + 30 = 0 ∧ t = 1 + (Real.sqrt 46) / 4) :=
sorry

end NUMINAMATH_GPT_ball_hits_ground_l27_2786


namespace NUMINAMATH_GPT_cost_of_graveling_per_sq_meter_l27_2721

theorem cost_of_graveling_per_sq_meter
    (length_lawn : ℝ) (breadth_lawn : ℝ)
    (width_road : ℝ) (total_cost_gravel : ℝ)
    (length_road_area : ℝ) (breadth_road_area : ℝ) (intersection_area : ℝ)
    (total_graveled_area : ℝ) (cost_per_sq_meter : ℝ) :
    length_lawn = 55 →
    breadth_lawn = 35 →
    width_road = 4 →
    total_cost_gravel = 258 →
    length_road_area = length_lawn * width_road →
    intersection_area = width_road * width_road →
    breadth_road_area = breadth_lawn * width_road - intersection_area →
    total_graveled_area = length_road_area + breadth_road_area →
    cost_per_sq_meter = total_cost_gravel / total_graveled_area →
    cost_per_sq_meter = 0.75 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cost_of_graveling_per_sq_meter_l27_2721


namespace NUMINAMATH_GPT_sum_of_digits_9x_l27_2733

theorem sum_of_digits_9x (a b c d e : ℕ) (x : ℕ) :
  (1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e ≤ 9) →
  x = 10000 * a + 1000 * b + 100 * c + 10 * d + e →
  (b - a) + (c - b) + (d - c) + (e - d) + (10 - e) = 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_9x_l27_2733


namespace NUMINAMATH_GPT_trucks_in_yard_l27_2770

/-- The number of trucks in the yard is 23, given the conditions. -/
theorem trucks_in_yard (T : ℕ) (H1 : ∃ n : ℕ, n > 0)
  (H2 : ∃ k : ℕ, k = 5 * T)
  (H3 : T + 5 * T = 140) : T = 23 :=
sorry

end NUMINAMATH_GPT_trucks_in_yard_l27_2770


namespace NUMINAMATH_GPT_number_of_zeros_l27_2763

noncomputable def g (x : ℝ) : ℝ := Real.cos (Real.log x)

theorem number_of_zeros (n : ℕ) : (1 < x ∧ x < Real.exp Real.pi) → (∃! x : ℝ, g x = 0 ∧ 1 < x ∧ x < Real.exp Real.pi) → n = 1 :=
sorry

end NUMINAMATH_GPT_number_of_zeros_l27_2763


namespace NUMINAMATH_GPT_negative_exponent_example_l27_2736

theorem negative_exponent_example : 3^(-2 : ℤ) = (1 : ℚ) / (3^2) :=
by sorry

end NUMINAMATH_GPT_negative_exponent_example_l27_2736


namespace NUMINAMATH_GPT_valid_passwords_count_l27_2712

-- Define the total number of unrestricted passwords
def total_passwords : ℕ := 10000

-- Define the number of restricted passwords (ending with 6, 3, 9)
def restricted_passwords : ℕ := 10

-- Define the total number of valid passwords
def valid_passwords := total_passwords - restricted_passwords

theorem valid_passwords_count : valid_passwords = 9990 := 
by 
  sorry

end NUMINAMATH_GPT_valid_passwords_count_l27_2712


namespace NUMINAMATH_GPT_carla_needs_24_cans_l27_2759

variable (cans_chilis : ℕ) (cans_beans : ℕ) (tomato_multiplier : ℕ) (batch_factor : ℕ)

def cans_tomatoes (cans_beans : ℕ) (tomato_multiplier : ℕ) : ℕ :=
  cans_beans * tomato_multiplier

def normal_batch_cans (cans_chilis : ℕ) (cans_beans : ℕ) (tomato_cans : ℕ) : ℕ :=
  cans_chilis + cans_beans + tomato_cans

def total_cans (normal_cans : ℕ) (batch_factor : ℕ) : ℕ :=
  normal_cans * batch_factor

theorem carla_needs_24_cans : 
  cans_chilis = 1 → 
  cans_beans = 2 → 
  tomato_multiplier = 3 / 2 → 
  batch_factor = 4 → 
  total_cans (normal_batch_cans cans_chilis cans_beans (cans_tomatoes cans_beans tomato_multiplier)) batch_factor = 24 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_carla_needs_24_cans_l27_2759


namespace NUMINAMATH_GPT_problem_statement_l27_2749

theorem problem_statement (n m : ℕ) (hn : n ≠ 0) (hm : m ≠ 0) : 
  (n * 5^n)^n = m * 5^9 ↔ n = 3 ∧ m = 27 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l27_2749


namespace NUMINAMATH_GPT_probability_of_adjacent_vertices_in_dodecagon_l27_2714

def probability_at_least_two_adjacent_vertices (n : ℕ) : ℚ :=
  if n = 12 then 24 / 55 else 0  -- Only considering the dodecagon case

theorem probability_of_adjacent_vertices_in_dodecagon :
  probability_at_least_two_adjacent_vertices 12 = 24 / 55 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_adjacent_vertices_in_dodecagon_l27_2714


namespace NUMINAMATH_GPT_rearrange_digits_2552_l27_2703

theorem rearrange_digits_2552 : 
    let digits := [2, 5, 5, 2]
    let factorial := fun n => Nat.factorial n
    let permutations := (factorial 4) / (factorial 2 * factorial 2)
    permutations = 6 :=
by
  sorry

end NUMINAMATH_GPT_rearrange_digits_2552_l27_2703


namespace NUMINAMATH_GPT_selling_price_eq_l27_2787

theorem selling_price_eq (cp sp L : ℕ) (h_cp: cp = 47) (h_L : L = cp - 40) (h_profit_loss_eq : sp - cp = L) :
  sp = 54 :=
by
  sorry

end NUMINAMATH_GPT_selling_price_eq_l27_2787


namespace NUMINAMATH_GPT_initial_dolphins_l27_2702

variable (D : ℕ)

theorem initial_dolphins (h1 : 3 * D + D = 260) : D = 65 :=
by
  sorry

end NUMINAMATH_GPT_initial_dolphins_l27_2702


namespace NUMINAMATH_GPT_positional_relationship_of_circles_l27_2799

theorem positional_relationship_of_circles 
  (m n : ℝ)
  (h1 : ∃ (x y : ℝ), x^2 - 10 * x + n = 0 ∧ y^2 - 10 * y + n = 0 ∧ x = 2 ∧ y = m) :
  n = 2 * m ∧ m = 8 → 16 > 2 + 8 :=
by
  sorry

end NUMINAMATH_GPT_positional_relationship_of_circles_l27_2799


namespace NUMINAMATH_GPT_value_of_z_sub_y_add_x_l27_2788

-- Represent 312 in base 3
def base3_representation : List ℕ := [1, 0, 1, 2, 1, 0] -- 312 in base 3 is 101210

-- Define x, y, z
def x : ℕ := (base3_representation.count 0)
def y : ℕ := (base3_representation.count 1)
def z : ℕ := (base3_representation.count 2)

-- Proposition to be proved
theorem value_of_z_sub_y_add_x : z - y + x = 2 := by
  sorry

end NUMINAMATH_GPT_value_of_z_sub_y_add_x_l27_2788


namespace NUMINAMATH_GPT_number_of_students_l27_2730

-- Define the conditions
variable (n : ℕ) (jayden_rank_best jayden_rank_worst : ℕ)
variable (h1 : jayden_rank_best = 100)
variable (h2 : jayden_rank_worst = 100)

-- Define the question
theorem number_of_students (h1 : jayden_rank_best = 100) (h2 : jayden_rank_worst = 100) : n = 199 := 
  sorry

end NUMINAMATH_GPT_number_of_students_l27_2730


namespace NUMINAMATH_GPT_find_coordinates_of_C_l27_2747

structure Point where
  x : ℝ
  y : ℝ

def parallelogram (A B C D : Point) : Prop :=
  (B.x - A.x = C.x - D.x ∧ B.y - A.y = C.y - D.y) ∧
  (D.x - A.x = C.x - B.x ∧ D.y - A.y = C.y - B.y)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨7, 3⟩
def D : Point := ⟨3, 7⟩
def C : Point := ⟨8, 7⟩

theorem find_coordinates_of_C :
  parallelogram A B C D → C = ⟨8, 7⟩ :=
by
  intro h
  have h₁ := h.1.1
  have h₂ := h.1.2
  have h₃ := h.2.1
  have h₄ := h.2.2
  sorry

end NUMINAMATH_GPT_find_coordinates_of_C_l27_2747


namespace NUMINAMATH_GPT_expected_balls_original_positions_l27_2765

noncomputable def expected_original_positions : ℝ :=
  8 * ((3/4:ℝ)^3)

theorem expected_balls_original_positions :
  expected_original_positions = 3.375 := by
  sorry

end NUMINAMATH_GPT_expected_balls_original_positions_l27_2765


namespace NUMINAMATH_GPT_polyhedron_volume_formula_l27_2781

noncomputable def polyhedron_volume (H S1 S2 S3 : ℝ) : ℝ :=
  (1 / 6) * H * (S1 + S2 + 4 * S3)

theorem polyhedron_volume_formula 
  (H S1 S2 S3 : ℝ)
  (bases_parallel_planes : Prop)
  (lateral_faces_trapezoids_parallelograms_or_triangles : Prop)
  (H_distance : Prop) 
  (S1_area_base : Prop) 
  (S2_area_base : Prop) 
  (S3_area_cross_section : Prop) : 
  polyhedron_volume H S1 S2 S3 = (1 / 6) * H * (S1 + S2 + 4 * S3) :=
sorry

end NUMINAMATH_GPT_polyhedron_volume_formula_l27_2781


namespace NUMINAMATH_GPT_percentage_of_women_attended_picnic_l27_2762

variable (E : ℝ) -- Total number of employees
variable (M : ℝ) -- The number of men
variable (W : ℝ) -- The number of women
variable (P : ℝ) -- Percentage of women who attended the picnic

-- Conditions
variable (h1 : M = 0.30 * E)
variable (h2 : W = E - M)
variable (h3 : 0.20 * M = 0.20 * 0.30 * E)
variable (h4 : 0.34 * E = 0.20 * 0.30 * E + P * (E - 0.30 * E))

-- Goal
theorem percentage_of_women_attended_picnic : P = 0.40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_women_attended_picnic_l27_2762


namespace NUMINAMATH_GPT_marble_problem_l27_2772

theorem marble_problem (a : ℚ) :
  (a + 2 * a + 3 * 2 * a + 5 * (3 * 2 * a) + 2 * (5 * (3 * 2 * a)) = 212) ↔
  (a = 212 / 99) :=
by
  sorry

end NUMINAMATH_GPT_marble_problem_l27_2772


namespace NUMINAMATH_GPT_find_base_b_l27_2704

theorem find_base_b (b : ℕ) :
  (2 * b^2 + 4 * b + 3) + (1 * b^2 + 5 * b + 6) = (4 * b^2 + 1 * b + 1) →
  7 < b →
  b = 10 :=
by
  intro h₁ h₂
  sorry

end NUMINAMATH_GPT_find_base_b_l27_2704


namespace NUMINAMATH_GPT_inv_f_zero_l27_2791

noncomputable def f (a b x : Real) : Real := 1 / (2 * a * x + 3 * b)

theorem inv_f_zero (a b : Real) (ha : a ≠ 0) (hb : b ≠ 0) : f a b (1 / (3 * b)) = 0 :=
by 
  sorry

end NUMINAMATH_GPT_inv_f_zero_l27_2791


namespace NUMINAMATH_GPT_find_value_of_a_l27_2780

variables (a : ℚ)

-- Definitions based on the conditions
def Brian_has_mar_bles : ℚ := 3 * a
def Caden_original_mar_bles : ℚ := 4 * Brian_has_mar_bles a
def Daryl_original_mar_bles : ℚ := 2 * Caden_original_mar_bles a
def Caden_after_give_10 : ℚ := Caden_original_mar_bles a - 10
def Daryl_after_receive_10 : ℚ := Daryl_original_mar_bles a + 10

-- Together Caden and Daryl now have 190 marbles
def together_mar_bles : ℚ := Caden_after_give_10 a + Daryl_after_receive_10 a

theorem find_value_of_a : together_mar_bles a = 190 → a = 95 / 18 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_a_l27_2780


namespace NUMINAMATH_GPT_find_x_y_l27_2758

theorem find_x_y 
  (x y : ℝ) 
  (h1 : (15 + 30 + x + y) / 4 = 25) 
  (h2 : x = y + 10) :
  x = 32.5 ∧ y = 22.5 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_y_l27_2758


namespace NUMINAMATH_GPT_carla_total_marbles_l27_2752

def initial_marbles : ℝ := 187.0
def bought_marbles : ℝ := 134.0

theorem carla_total_marbles : initial_marbles + bought_marbles = 321.0 := by
  sorry

end NUMINAMATH_GPT_carla_total_marbles_l27_2752


namespace NUMINAMATH_GPT_sam_total_cans_l27_2744

theorem sam_total_cans (bags_sat : ℕ) (bags_sun : ℕ) (cans_per_bag : ℕ) 
  (h_sat : bags_sat = 3) (h_sun : bags_sun = 4) (h_cans : cans_per_bag = 9) : 
  (bags_sat + bags_sun) * cans_per_bag = 63 := 
by
  sorry

end NUMINAMATH_GPT_sam_total_cans_l27_2744


namespace NUMINAMATH_GPT_tank_fish_count_l27_2783

theorem tank_fish_count (total_fish blue_fish : ℕ) 
  (h1 : blue_fish = total_fish / 3)
  (h2 : 10 * 2 = blue_fish) : 
  total_fish = 60 :=
sorry

end NUMINAMATH_GPT_tank_fish_count_l27_2783


namespace NUMINAMATH_GPT_arithmetic_mean_reciprocals_first_four_primes_l27_2739

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_reciprocals_first_four_primes_l27_2739


namespace NUMINAMATH_GPT_g_f_neg4_eq_12_l27_2755

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^2 - 8

-- Define the assumption that g(f(4)) = 12
axiom g : ℝ → ℝ
axiom g_f4 : g (f 4) = 12

-- The theorem to prove that g(f(-4)) = 12
theorem g_f_neg4_eq_12 : g (f (-4)) = 12 :=
sorry -- proof placeholder

end NUMINAMATH_GPT_g_f_neg4_eq_12_l27_2755


namespace NUMINAMATH_GPT_total_animal_legs_l27_2708

def number_of_dogs : ℕ := 2
def number_of_chickens : ℕ := 1
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2

theorem total_animal_legs : number_of_dogs * legs_per_dog + number_of_chickens * legs_per_chicken = 10 :=
by
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_total_animal_legs_l27_2708


namespace NUMINAMATH_GPT_tiling_remainder_is_888_l27_2734

noncomputable def boardTilingWithThreeColors (n : ℕ) : ℕ :=
  if n = 8 then
    4 * (21 * (3^3 - 3*2^3 + 3) +
         35 * (3^4 - 4*2^4 + 6) +
         35 * (3^5 - 5*2^5 + 10) +
         21 * (3^6 - 6*2^6 + 15) +
         7 * (3^7 - 7*2^7 + 21) +
         1 * (3^8 - 8*2^8 + 28))
  else
    0

theorem tiling_remainder_is_888 :
  boardTilingWithThreeColors 8 % 1000 = 888 :=
by
  sorry

end NUMINAMATH_GPT_tiling_remainder_is_888_l27_2734


namespace NUMINAMATH_GPT_total_matches_played_l27_2723

theorem total_matches_played (home_wins : ℕ) (rival_wins : ℕ) (draws : ℕ) (home_wins_eq : home_wins = 3) (rival_wins_eq : rival_wins = 2 * home_wins) (draws_eq : draws = 4) (no_losses : ∀ (t : ℕ), t = 0) :
  home_wins + rival_wins + 2 * draws = 17 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_matches_played_l27_2723


namespace NUMINAMATH_GPT_largest_common_number_in_arithmetic_sequences_l27_2709

theorem largest_common_number_in_arithmetic_sequences (n : ℕ) :
  (∃ a1 a2 : ℕ, a1 = 5 + 8 * n ∧ a2 = 3 + 9 * n ∧ a1 = a2 ∧ 1 ≤ a1 ∧ a1 ≤ 150) →
  (a1 = 93) :=
by
  sorry

end NUMINAMATH_GPT_largest_common_number_in_arithmetic_sequences_l27_2709


namespace NUMINAMATH_GPT_fourth_power_sum_l27_2718

variable (a b c : ℝ)

theorem fourth_power_sum (h1 : a + b + c = 2) 
                         (h2 : a^2 + b^2 + c^2 = 3) 
                         (h3 : a^3 + b^3 + c^3 = 4) : 
                         a^4 + b^4 + c^4 = 41 / 6 := 
by 
  sorry

end NUMINAMATH_GPT_fourth_power_sum_l27_2718


namespace NUMINAMATH_GPT_rectangle_circumference_15pi_l27_2782

noncomputable def rectangle_diagonal (a b : ℝ) : ℝ := 
  Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def circumference_of_circle (d : ℝ) : ℝ := 
  Real.pi * d
  
theorem rectangle_circumference_15pi :
  let a := 9
  let b := 12
  let diagonal := rectangle_diagonal a b
  circumference_of_circle diagonal = 15 * Real.pi :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_circumference_15pi_l27_2782


namespace NUMINAMATH_GPT_geometric_sequence_value_l27_2741

theorem geometric_sequence_value (a : ℕ → ℝ) (h : ∀ n, a n > 0)
  (h_geometric : ∀ n, a (n+2) = a (n+1) * (a (n+1) / a n)) :
  a 3 * a 5 = 4 → a 4 = 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_value_l27_2741


namespace NUMINAMATH_GPT_lowest_fraction_combine_two_slowest_l27_2753

def rate_a (hours : ℕ) : ℚ := 1 / 4
def rate_b (hours : ℕ) : ℚ := 1 / 5
def rate_c (hours : ℕ) : ℚ := 1 / 8

theorem lowest_fraction_combine_two_slowest : 
  (rate_b 1 + rate_c 1) = 13 / 40 :=
by sorry

end NUMINAMATH_GPT_lowest_fraction_combine_two_slowest_l27_2753


namespace NUMINAMATH_GPT_not_all_sets_of_10_segments_form_triangle_l27_2735

theorem not_all_sets_of_10_segments_form_triangle :
  ¬ ∀ (segments : Fin 10 → ℝ), ∃ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (segments a + segments b > segments c) ∧
    (segments a + segments c > segments b) ∧
    (segments b + segments c > segments a) :=
by
  sorry

end NUMINAMATH_GPT_not_all_sets_of_10_segments_form_triangle_l27_2735


namespace NUMINAMATH_GPT_bananas_each_child_l27_2761

theorem bananas_each_child (x : ℕ) (B : ℕ) 
  (h1 : 660 * x = B)
  (h2 : 330 * (x + 2) = B) : 
  x = 2 := 
by 
  sorry

end NUMINAMATH_GPT_bananas_each_child_l27_2761


namespace NUMINAMATH_GPT_probability_two_red_faces_eq_three_eighths_l27_2740

def cube_probability : ℚ :=
  let total_cubes := 64 -- Total number of smaller cubes
  let two_red_faces_cubes := 24 -- Number of smaller cubes with exactly two red faces
  two_red_faces_cubes / total_cubes

theorem probability_two_red_faces_eq_three_eighths :
  cube_probability = 3 / 8 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_probability_two_red_faces_eq_three_eighths_l27_2740


namespace NUMINAMATH_GPT_age_problem_lean4_l27_2716

/-
Conditions:
1. Mr. Bernard's age in eight years will be 60.
2. Luke's age in eight years will be 28.
3. Sarah's age in eight years will be 48.
4. The sum of their ages in eight years will be 136.

Question (translated to proof problem):
Prove that 10 years less than the average age of all three of them is approximately 35.33.

The Lean 4 statement below formalizes this:
-/

theorem age_problem_lean4 :
  let bernard_age := 60
  let luke_age := 28
  let sarah_age := 48
  let total_age := bernard_age + luke_age + sarah_age
  total_age = 136 → ((total_age / 3.0) - 10.0 = 35.33) :=
by
  intros
  sorry

end NUMINAMATH_GPT_age_problem_lean4_l27_2716


namespace NUMINAMATH_GPT_proof_problem_l27_2738

theorem proof_problem 
  (A a B b : ℝ) 
  (h1 : |A - 3 * a| ≤ 1 - a) 
  (h2 : |B - 3 * b| ≤ 1 - b) 
  (h3 : 0 < a) 
  (h4 : 0 < b) :
  (|((A * B) / 3) - 3 * (a * b)|) - 3 * (a * b) ≤ 1 - (a * b) :=
sorry

end NUMINAMATH_GPT_proof_problem_l27_2738


namespace NUMINAMATH_GPT_pieces_in_each_package_l27_2711

-- Definitions from conditions
def num_packages : ℕ := 5
def extra_pieces : ℕ := 6
def total_pieces : ℕ := 41

-- Statement to prove
theorem pieces_in_each_package : ∃ x : ℕ, num_packages * x + extra_pieces = total_pieces ∧ x = 7 :=
by
  -- Begin the proof with the given setup
  sorry

end NUMINAMATH_GPT_pieces_in_each_package_l27_2711


namespace NUMINAMATH_GPT_sequence_formula_l27_2776

theorem sequence_formula (x : ℕ → ℤ) :
  x 1 = 1 →
  x 2 = -1 →
  (∀ n, n ≥ 2 → x (n-1) + x (n+1) = 2 * x n) →
  ∀ n, x n = -2 * n + 3 :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l27_2776


namespace NUMINAMATH_GPT_algebraic_expression_value_l27_2794

theorem algebraic_expression_value (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 4 * x^2 + 6 * x - 9 = -7 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l27_2794


namespace NUMINAMATH_GPT_smallest_of_three_integers_l27_2706

theorem smallest_of_three_integers (a b c : ℤ) (h1 : a * b * c = 32) (h2 : a + b + c = 3) : min (min a b) c = -4 := 
sorry

end NUMINAMATH_GPT_smallest_of_three_integers_l27_2706


namespace NUMINAMATH_GPT_second_pipe_filling_time_l27_2789

theorem second_pipe_filling_time (T : ℝ) :
  (∃ T : ℝ, (1 / 8 + 1 / T = 1 / 4.8) ∧ T = 12) :=
by
  sorry

end NUMINAMATH_GPT_second_pipe_filling_time_l27_2789


namespace NUMINAMATH_GPT_neg_p_sufficient_not_necessary_q_l27_2774

-- Definitions from the given conditions
def p (a : ℝ) : Prop := a ≥ 1
def q (a : ℝ) : Prop := a ≤ 2

-- The theorem stating the mathematical equivalence
theorem neg_p_sufficient_not_necessary_q (a : ℝ) : (¬ p a → q a) ∧ ¬ (q a → ¬ p a) := 
by sorry

end NUMINAMATH_GPT_neg_p_sufficient_not_necessary_q_l27_2774


namespace NUMINAMATH_GPT_negation_proposition_l27_2756

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^2 - x - 1 < 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≥ 0) :=
by 
  sorry

end NUMINAMATH_GPT_negation_proposition_l27_2756


namespace NUMINAMATH_GPT_both_questions_correct_l27_2773

-- Define variables as constants
def nA : ℝ := 0.85  -- 85%
def nB : ℝ := 0.70  -- 70%
def nAB : ℝ := 0.60 -- 60%

theorem both_questions_correct:
  nAB = 0.60 := by
  sorry

end NUMINAMATH_GPT_both_questions_correct_l27_2773


namespace NUMINAMATH_GPT_students_in_class_l27_2793

theorem students_in_class (n m f r u : ℕ) (cond1 : 20 < n ∧ n < 30)
  (cond2 : f = 2 * m) (cond3 : n = m + f)
  (cond4 : r = 3 * u - 1) (cond5 : r + u = n) :
  n = 27 :=
sorry

end NUMINAMATH_GPT_students_in_class_l27_2793


namespace NUMINAMATH_GPT_arrange_cubes_bound_l27_2750

def num_ways_to_arrange_cubes_into_solids (n : ℕ) : ℕ := sorry

theorem arrange_cubes_bound (n : ℕ) (h : n = (2015^100)) :
  10^14 < num_ways_to_arrange_cubes_into_solids n ∧
  num_ways_to_arrange_cubes_into_solids n < 10^15 := sorry

end NUMINAMATH_GPT_arrange_cubes_bound_l27_2750


namespace NUMINAMATH_GPT_mari_vs_kendra_l27_2768

-- Variable Definitions
variables (K M S : ℕ)  -- Number of buttons Kendra, Mari, and Sue made
variables (h1: 2*S = K) -- Sue made half as many as Kendra
variables (h2: S = 6)   -- Sue made 6 buttons
variables (h3: M = 64)  -- Mari made 64 buttons

-- Theorem Statement
theorem mari_vs_kendra (K M S : ℕ) (h1 : 2 * S = K) (h2 : S = 6) (h3 : M = 64) :
  M = 5 * K + 4 :=
sorry

end NUMINAMATH_GPT_mari_vs_kendra_l27_2768


namespace NUMINAMATH_GPT_tetrahedron_perpendicular_distances_inequalities_l27_2798

section Tetrahedron

variables {R : Type*} [LinearOrderedField R]

variables {S_A S_B S_C S_D V d_A d_B d_C d_D h_A h_B h_C h_D : R}

/-- Given areas and perpendicular distances of a tetrahedron, prove inequalities involving these parameters. -/
theorem tetrahedron_perpendicular_distances_inequalities 
  (h1 : S_A * d_A + S_B * d_B + S_C * d_C + S_D * d_D = 3 * V) : 
  (min h_A (min h_B (min h_C h_D)) ≤ d_A + d_B + d_C + d_D) ∧ 
  (d_A + d_B + d_C + d_D ≤ max h_A (max h_B (max h_C h_D))) ∧ 
  (d_A * d_B * d_C * d_D ≤ 81 * V ^ 4 / (256 * S_A * S_B * S_C * S_D)) :=
sorry

end Tetrahedron

end NUMINAMATH_GPT_tetrahedron_perpendicular_distances_inequalities_l27_2798


namespace NUMINAMATH_GPT_triangle_area_correct_l27_2737
noncomputable def area_of_triangle_intercepts : ℝ :=
  let f (x : ℝ) : ℝ := (x - 3) ^ 2 * (x + 2)
  let x1 := 3
  let x2 := -2
  let y_intercept := f 0
  let base := x1 - x2
  let height := y_intercept
  1 / 2 * base * height

theorem triangle_area_correct :
  area_of_triangle_intercepts = 45 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_correct_l27_2737


namespace NUMINAMATH_GPT_smallest_three_digit_multiple_of_13_l27_2731

theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, (100 ≤ n) ∧ (n < 1000) ∧ (n % 13 = 0) ∧ (∀ k : ℕ, (100 ≤ k) ∧ (k < 1000) ∧ (k % 13 = 0) → n ≤ k) → n = 104 :=
by
  sorry

end NUMINAMATH_GPT_smallest_three_digit_multiple_of_13_l27_2731
