import Mathlib

namespace find_num_candies_bought_l2244_224479

-- Conditions
def cost_per_candy := 80
def sell_price_per_candy := 100
def num_sold := 48
def profit := 800

-- Question equivalence
theorem find_num_candies_bought (x : ℕ) 
  (hc : cost_per_candy = 80)
  (hs : sell_price_per_candy = 100)
  (hn : num_sold = 48)
  (hp : profit = 800) :
  48 * 100 - 80 * x = 800 → x = 50 :=
  by
  sorry

end find_num_candies_bought_l2244_224479


namespace right_triangle_perimeter_l2244_224424

theorem right_triangle_perimeter (n : ℕ) (hn : Nat.Prime n) (x y : ℕ) 
  (h1 : y^2 = x^2 + n^2) : n + x + y = n + n^2 := by
  sorry

end right_triangle_perimeter_l2244_224424


namespace six_digit_mod_27_l2244_224498

theorem six_digit_mod_27 (X : ℕ) (hX : 100000 ≤ X ∧ X < 1000000) (Y : ℕ) (hY : ∃ a b : ℕ, 100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000 ∧ X = 1000 * a + b ∧ Y = 1000 * b + a) :
  X % 27 = Y % 27 := 
by
  sorry

end six_digit_mod_27_l2244_224498


namespace paint_houses_l2244_224423

theorem paint_houses (time_per_house : ℕ) (hour_to_minute : ℕ) (hours_available : ℕ) 
  (h1 : time_per_house = 20) (h2 : hour_to_minute = 60) (h3 : hours_available = 3) :
  (hours_available * hour_to_minute) / time_per_house = 9 :=
by
  sorry

end paint_houses_l2244_224423


namespace cryptarithm_problem_l2244_224484

theorem cryptarithm_problem (F E D : ℤ) (h1 : F - E = D - 1) (h2 : D + E + F = 16) (h3 : F - E = D) : 
    F - E = 5 :=
by sorry

end cryptarithm_problem_l2244_224484


namespace evaluate_sum_l2244_224429

theorem evaluate_sum (a b c : ℝ) 
  (h : (a / (36 - a) + b / (49 - b) + c / (81 - c) = 9)) :
  (6 / (36 - a) + 7 / (49 - b) + 9 / (81 - c) = 5.047) :=
by
  sorry

end evaluate_sum_l2244_224429


namespace scout_troop_profit_calc_l2244_224406

theorem scout_troop_profit_calc
  (candy_bars : ℕ := 1200)
  (purchase_rate : ℚ := 3/6)
  (sell_rate : ℚ := 2/3) :
  (candy_bars * sell_rate - candy_bars * purchase_rate) = 200 :=
by
  sorry

end scout_troop_profit_calc_l2244_224406


namespace find_x_l2244_224432

variable (m k x Km2 mk : ℚ)

def valid_conditions (m k : ℚ) : Prop :=
  m > 2 * k ∧ k > 0

def initial_acid (m : ℚ) : ℚ :=
  (m*m)/100

def diluted_acid (m k x : ℚ) : ℚ :=
  ((2*m) - k) * (m + x) / 100

theorem find_x (m k : ℚ) (h : valid_conditions m k):
  ∃ x : ℚ, (m^2 = diluted_acid m k x) ∧ x = (k * m - m^2) / (2 * m - k) :=
sorry

end find_x_l2244_224432


namespace intersection_point_of_lines_l2244_224464

theorem intersection_point_of_lines :
  (∃ x y : ℝ, y = x ∧ y = -x + 2 ∧ (x = 1 ∧ y = 1)) :=
sorry

end intersection_point_of_lines_l2244_224464


namespace geometric_sequence_common_ratio_l2244_224458

theorem geometric_sequence_common_ratio 
  (a1 q : ℝ) 
  (h : (a1 * (1 - q^3) / (1 - q)) + 3 * (a1 * (1 - q^2) / (1 - q)) = 0) : 
  q = -1 :=
sorry

end geometric_sequence_common_ratio_l2244_224458


namespace aardvark_total_distance_l2244_224495

noncomputable def total_distance (r_small r_large : ℝ) : ℝ :=
  let small_circumference := 2 * Real.pi * r_small
  let large_circumference := 2 * Real.pi * r_large
  let half_small_circumference := small_circumference / 2
  let half_large_circumference := large_circumference / 2
  let radial_distance := r_large - r_small
  let total_radial_distance := radial_distance + r_large
  half_small_circumference + radial_distance + half_large_circumference + total_radial_distance

theorem aardvark_total_distance :
  total_distance 15 30 = 45 * Real.pi + 45 :=
by
  sorry

end aardvark_total_distance_l2244_224495


namespace product_of_fractions_l2244_224410

-- Definitions from the conditions
def a : ℚ := 2 / 3 
def b : ℚ := 3 / 5
def c : ℚ := 4 / 7
def d : ℚ := 5 / 9

-- Statement of the proof problem
theorem product_of_fractions : a * b * c * d = 8 / 63 := 
by
  sorry

end product_of_fractions_l2244_224410


namespace resulting_polygon_sides_l2244_224445

theorem resulting_polygon_sides 
    (triangle_sides : ℕ := 3) 
    (square_sides : ℕ := 4) 
    (pentagon_sides : ℕ := 5) 
    (heptagon_sides : ℕ := 7) 
    (hexagon_sides : ℕ := 6) 
    (octagon_sides : ℕ := 8) 
    (shared_sides : ℕ := 1) :
    (2 * shared_sides + 4 * (shared_sides + 1)) = 16 := by 
  sorry

end resulting_polygon_sides_l2244_224445


namespace milk_needed_for_cookies_l2244_224419

-- Define the given conditions
def liters_to_cups (liters : ℕ) : ℕ := liters * 4

def milk_per_cookies (cups cookies : ℕ) : ℚ := cups / cookies

-- Define the problem statement
theorem milk_needed_for_cookies (h1 : milk_per_cookies 20 30 = milk_per_cookies x 12) : x = 8 :=
sorry

end milk_needed_for_cookies_l2244_224419


namespace matrix_power_50_l2244_224408

def P : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 2],
  ![-4, -3]
]

theorem matrix_power_50 :
  P ^ 50 = ![
    ![1, 0],
    ![0, 1]
  ] :=
sorry

end matrix_power_50_l2244_224408


namespace eval_expression_l2244_224444

theorem eval_expression : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 2023) / (2023 * 2024) = -4044 :=
by 
  sorry

end eval_expression_l2244_224444


namespace temperature_representation_l2244_224472

def represents_zero_degrees_celsius (t₁ : ℝ) : Prop := t₁ = 10

theorem temperature_representation (t₁ t₂ : ℝ) (h₀ : represents_zero_degrees_celsius t₁) 
    (h₁ : t₂ > t₁):
    t₂ = 17 :=
by
  -- Proof is omitted here
  sorry

end temperature_representation_l2244_224472


namespace max_abs_sum_on_ellipse_l2244_224440

theorem max_abs_sum_on_ellipse :
  ∀ (x y : ℝ), (x^2 / 4) + (y^2 / 9) = 1 → |x| + |y| ≤ 5 :=
by sorry

end max_abs_sum_on_ellipse_l2244_224440


namespace initial_percentage_increase_l2244_224433

theorem initial_percentage_increase (W R : ℝ) (P : ℝ) 
  (h1 : R = W * (1 + P / 100)) 
  (h2 : R * 0.75 = W * 1.3500000000000001) : P = 80 := 
by
  sorry

end initial_percentage_increase_l2244_224433


namespace parabola_focus_l2244_224450

theorem parabola_focus (x y p : ℝ) (h_eq : y = 2 * x^2) (h_standard_form : x^2 = (1 / 2) * y) (h_p : p = 1 / 4) : 
    (0, p / 2) = (0, 1 / 8) := by
    sorry

end parabola_focus_l2244_224450


namespace equation_B_is_quadratic_l2244_224471

theorem equation_B_is_quadratic : ∀ y : ℝ, ∃ A B C : ℝ, (5 * y ^ 2 - 5 * y = 0) ∧ A ≠ 0 :=
by
  sorry

end equation_B_is_quadratic_l2244_224471


namespace shepherd_boys_equation_l2244_224467

theorem shepherd_boys_equation (x : ℕ) :
  6 * x + 14 = 8 * x - 2 :=
by sorry

end shepherd_boys_equation_l2244_224467


namespace green_duck_percentage_l2244_224499

theorem green_duck_percentage (G_small G_large : ℝ) (D_small D_large : ℕ)
    (H1 : G_small = 0.20) (H2 : D_small = 20)
    (H3 : G_large = 0.15) (H4 : D_large = 80) : 
    ((G_small * D_small + G_large * D_large) / (D_small + D_large)) * 100 = 16 := 
by
  sorry

end green_duck_percentage_l2244_224499


namespace fruit_display_total_l2244_224451

-- Define the number of bananas
def bananas : ℕ := 5

-- Define the number of oranges, twice the number of bananas
def oranges : ℕ := 2 * bananas

-- Define the number of apples, twice the number of oranges
def apples : ℕ := 2 * oranges

-- Define the total number of fruits
def total_fruits : ℕ := bananas + oranges + apples

-- Theorem statement: Prove that the total number of fruits is 35
theorem fruit_display_total : total_fruits = 35 := 
by
  -- proof will be skipped, but all necessary conditions and definitions are included to ensure the statement is valid
  sorry

end fruit_display_total_l2244_224451


namespace stratified_sampling_grade12_l2244_224404

theorem stratified_sampling_grade12 (total_students grade12_students sample_size : ℕ) 
  (h_total : total_students = 2000) 
  (h_grade12 : grade12_students = 700) 
  (h_sample : sample_size = 400) : 
  (sample_size * grade12_students) / total_students = 140 := 
by 
  sorry

end stratified_sampling_grade12_l2244_224404


namespace train_length_l2244_224491

theorem train_length
  (speed_kmph : ℝ)
  (platform_length : ℝ)
  (crossing_time : ℝ)
  (conversion_factor : ℝ)
  (speed_mps : ℝ)
  (distance_covered : ℝ)
  (train_length : ℝ) :
  speed_kmph = 72 →
  platform_length = 240 →
  crossing_time = 26 →
  conversion_factor = 5 / 18 →
  speed_mps = speed_kmph * conversion_factor →
  distance_covered = speed_mps * crossing_time →
  train_length = distance_covered - platform_length →
  train_length = 280 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end train_length_l2244_224491


namespace proof_a_plus_2b_equal_7_l2244_224474

theorem proof_a_plus_2b_equal_7 (a b : ℕ) (h1 : 82 * 1000 + a * 10 + 7 + 6 * b = 190) (h2 : 1 ≤ a) (h3 : a < 10) (h4 : 1 ≤ b) (h5 : b < 10) : 
  a + 2 * b = 7 :=
by sorry

end proof_a_plus_2b_equal_7_l2244_224474


namespace john_avg_speed_l2244_224427

/-- John's average speed problem -/
theorem john_avg_speed (d : ℕ) (total_time : ℕ) (time1 : ℕ) (speed1 : ℕ) 
  (time2 : ℕ) (speed2 : ℕ) (time3 : ℕ) (x : ℕ) :
  d = 144 ∧ total_time = 120 ∧ time1 = 40 ∧ speed1 = 64 
  ∧ time2 = 40 ∧ speed2 = 70 ∧ time3 = 40 
  → (d = time1 * speed1 + time2 * speed2 + time3 * x / 60)
  → x = 82 := 
by
  intros h1 h2
  sorry

end john_avg_speed_l2244_224427


namespace monthly_growth_rate_l2244_224415

theorem monthly_growth_rate (x : ℝ)
  (turnover_may : ℝ := 1)
  (turnover_july : ℝ := 1.21)
  (growth_rate_condition : (1 + x) ^ 2 = 1.21) :
  x = 0.1 :=
sorry

end monthly_growth_rate_l2244_224415


namespace complex_point_second_quadrant_l2244_224485

theorem complex_point_second_quadrant (i : ℂ) (h1 : i^4 = 1) :
  ∃ (z : ℂ), z = ((i^(2014))/(1 + i) * i) ∧ z.re < 0 ∧ z.im > 0 :=
by
  sorry

end complex_point_second_quadrant_l2244_224485


namespace proportion_difference_l2244_224402

theorem proportion_difference : (0.80 * 40) - ((4 / 5) * 20) = 16 := 
by 
  sorry

end proportion_difference_l2244_224402


namespace expression_evaluation_l2244_224455

theorem expression_evaluation : (3 * 15) + 47 - 27 * (2^3) / 4 = 38 := by
  sorry

end expression_evaluation_l2244_224455


namespace distance_from_point_to_focus_l2244_224428

theorem distance_from_point_to_focus (x0 : ℝ) (h1 : (2 * Real.sqrt 3)^2 = 4 * x0) :
    x0 + 1 = 4 := by
  sorry

end distance_from_point_to_focus_l2244_224428


namespace find_x_l2244_224462

-- Define the conditions as hypotheses
def problem_statement (x : ℤ) : Prop :=
  (3 * x > 30) ∧ (x ≥ 10) ∧ (x > 5) ∧ 
  (x = 9)

-- Define the theorem statement
theorem find_x : ∃ x : ℤ, problem_statement x :=
by
  -- Sorry to skip proof as instructed
  sorry

end find_x_l2244_224462


namespace value_of_item_l2244_224421

theorem value_of_item (a b m p : ℕ) (h : a ≠ b) (eq_capitals : a * x + m = b * x + p) : 
  x = (p - m) / (a - b) :=
by
  sorry

end value_of_item_l2244_224421


namespace pascal_fifth_element_row_20_l2244_224463

theorem pascal_fifth_element_row_20 :
  (Nat.choose 20 4) = 4845 := by
  sorry

end pascal_fifth_element_row_20_l2244_224463


namespace smallest_digit_not_in_units_place_of_odd_l2244_224420

theorem smallest_digit_not_in_units_place_of_odd :
  ∀ d : ℕ, (d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5 ∧ d ≠ 7 ∧ d ≠ 9) → d = 0 → d = 0 :=
by intros d h1 h2; exact h2

end smallest_digit_not_in_units_place_of_odd_l2244_224420


namespace riding_owners_ratio_l2244_224469

theorem riding_owners_ratio :
  ∃ (R W : ℕ), (R + W = 16) ∧ (4 * R + 6 * W = 80) ∧ (R : ℚ) / 16 = 1/2 :=
by
  sorry

end riding_owners_ratio_l2244_224469


namespace abc_inequality_l2244_224438

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b + b * c + a * c)^2 ≥ 3 * a * b * c * (a + b + c) :=
by sorry

end abc_inequality_l2244_224438


namespace average_speed_triathlon_l2244_224470

theorem average_speed_triathlon :
  let swimming_distance := 1.5
  let biking_distance := 3
  let running_distance := 2
  let swimming_speed := 2
  let biking_speed := 25
  let running_speed := 8

  let t_s := swimming_distance / swimming_speed
  let t_b := biking_distance / biking_speed
  let t_r := running_distance / running_speed
  let total_time := t_s + t_b + t_r

  let total_distance := swimming_distance + biking_distance + running_distance
  let average_speed := total_distance / total_time

  average_speed = 5.8 :=
  by
    sorry

end average_speed_triathlon_l2244_224470


namespace equation_has_at_most_one_real_root_l2244_224460

def has_inverse (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, ∀ x, g (f x) = x

theorem equation_has_at_most_one_real_root (f : ℝ → ℝ) (a : ℝ) (h : has_inverse f) :
  ∀ x1 x2 : ℝ, f x1 = a ∧ f x2 = a → x1 = x2 :=
by sorry

end equation_has_at_most_one_real_root_l2244_224460


namespace average_sale_l2244_224418

-- Defining the monthly sales as constants
def sale_month1 : ℝ := 6435
def sale_month2 : ℝ := 6927
def sale_month3 : ℝ := 6855
def sale_month4 : ℝ := 7230
def sale_month5 : ℝ := 6562
def sale_month6 : ℝ := 7391

-- The final theorem statement to prove the average sale
theorem average_sale : (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) / 6 = 6900 := 
by 
  sorry

end average_sale_l2244_224418


namespace cistern_length_l2244_224439

theorem cistern_length (w d A : ℝ) (h : d = 1.25 ∧ w = 4 ∧ A = 68.5) :
  ∃ L : ℝ, (L * w) + (2 * L * d) + (2 * w * d) = A ∧ L = 9 :=
by
  obtain ⟨h_d, h_w, h_A⟩ := h
  use 9
  simp [h_d, h_w, h_A]
  norm_num
  sorry

end cistern_length_l2244_224439


namespace willie_cream_from_farm_l2244_224481

variable (total_needed amount_to_buy amount_from_farm : ℕ)

theorem willie_cream_from_farm :
  total_needed = 300 → amount_to_buy = 151 → amount_from_farm = total_needed - amount_to_buy → amount_from_farm = 149 := by
  intros
  sorry

end willie_cream_from_farm_l2244_224481


namespace union_when_m_is_one_range_of_m_condition_1_range_of_m_condition_2_l2244_224477

open Set

noncomputable def A := {x : ℝ | -2 < x ∧ x < 2}
noncomputable def B (m : ℝ) := {x : ℝ | (m - 2) ≤ x ∧ x ≤ (2 * m + 1)}

-- Part (1):
theorem union_when_m_is_one :
  A ∪ B 1 = {x : ℝ | -2 < x ∧ x ≤ 3} := sorry

-- Part (2):
theorem range_of_m_condition_1 :
  ∀ m : ℝ, A ∩ B m = ∅ ↔ m ∈ Iic (-3/2) ∪ Ici 4 := sorry

theorem range_of_m_condition_2 :
  ∀ m : ℝ, A ∪ B m = A ↔ m ∈ Iio (-3) ∪ Ioo 0 (1/2) := sorry

end union_when_m_is_one_range_of_m_condition_1_range_of_m_condition_2_l2244_224477


namespace candy_bars_per_bag_l2244_224483

def total_candy_bars : ℕ := 15
def number_of_bags : ℕ := 5

theorem candy_bars_per_bag : total_candy_bars / number_of_bags = 3 :=
by
  sorry

end candy_bars_per_bag_l2244_224483


namespace expand_expression_l2244_224436

theorem expand_expression : ∀ (x : ℝ), (20 * x - 25) * 3 * x = 60 * x^2 - 75 * x := 
by
  intro x
  sorry

end expand_expression_l2244_224436


namespace marion_paperclips_correct_l2244_224416

def yun_initial_paperclips := 30
def yun_remaining_paperclips (x : ℕ) : ℕ := (2 * x) / 5
def marion_paperclips (x y : ℕ) : ℕ := (4 * (yun_remaining_paperclips x)) / 3 + y
def y := 7

theorem marion_paperclips_correct : marion_paperclips yun_initial_paperclips y = 23 := by
  sorry

end marion_paperclips_correct_l2244_224416


namespace club_committee_probability_l2244_224454

noncomputable def probability_at_least_two_boys_and_two_girls (total_members boys girls committee_size : ℕ) : ℚ :=
  let total_ways := Nat.choose total_members committee_size
  let ways_fewer_than_two_boys := (Nat.choose girls committee_size) + (boys * Nat.choose girls (committee_size - 1))
  let ways_fewer_than_two_girls := (Nat.choose boys committee_size) + (girls * Nat.choose boys (committee_size - 1))
  let ways_invalid := ways_fewer_than_two_boys + ways_fewer_than_two_girls
  (total_ways - ways_invalid) / total_ways

theorem club_committee_probability :
  probability_at_least_two_boys_and_two_girls 30 12 18 6 = 457215 / 593775 :=
by
  sorry

end club_committee_probability_l2244_224454


namespace find_pairs_l2244_224411

theorem find_pairs (a b q r : ℕ) (h1 : a * b = q * (a + b) + r)
  (h2 : q^2 + r = 2011) (h3 : 0 ≤ r ∧ r < a + b) : 
  (∃ t : ℕ, 1 ≤ t ∧ t ≤ 45 ∧ (a = t ∧ b = t + 2012 ∨ a = t + 2012 ∧ b = t)) :=
by
  sorry

end find_pairs_l2244_224411


namespace average_math_score_first_year_students_l2244_224456

theorem average_math_score_first_year_students 
  (total_male_students : ℕ) (total_female_students : ℕ)
  (sample_size : ℕ) (avg_score_male : ℕ) (avg_score_female : ℕ)
  (male_sample_size female_sample_size : ℕ)
  (weighted_avg : ℚ) :
  total_male_students = 300 → 
  total_female_students = 200 →
  sample_size = 60 → 
  avg_score_male = 110 →
  avg_score_female = 100 →
  male_sample_size = (3 * sample_size) / 5 →
  female_sample_size = (2 * sample_size) / 5 →
  weighted_avg = (male_sample_size * avg_score_male + female_sample_size * avg_score_female : ℕ) / sample_size → 
  weighted_avg = 106 := 
by
  sorry

end average_math_score_first_year_students_l2244_224456


namespace ratio_of_x_to_y_l2244_224488

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + 3 * y) = 1 / 2) : x / y = 7 / 4 :=
sorry

end ratio_of_x_to_y_l2244_224488


namespace function_relationship_value_of_x_l2244_224497

variable {x y : ℝ}

-- Given conditions:
-- Condition 1: y is inversely proportional to x
def inversely_proportional (p : ℝ) (q : ℝ) (k : ℝ) : Prop := p = k / q

-- Condition 2: y(2) = -3
def specific_value (x_val y_val : ℝ) : Prop := y_val = -3 ∧ x_val = 2

-- Questions rephrased as Lean theorems:

-- The function relationship between y and x is y = -6 / x
theorem function_relationship (k : ℝ) (hx : x ≠ 0) 
  (h_inv_prop: inversely_proportional y x k) (h_spec : specific_value 2 (-3)) : k = -6 :=
by
  sorry

-- When y = 2, x = -3
theorem value_of_x (hx : x ≠ 0) (hy : y = 2)
  (h_inv_prop : inversely_proportional y x (-6)) : x = -3 :=
by
  sorry

end function_relationship_value_of_x_l2244_224497


namespace largest_four_digit_divisible_by_14_l2244_224493

theorem largest_four_digit_divisible_by_14 :
  ∃ (A : ℕ), A = 9898 ∧ 
  (∃ a b : ℕ, A = 1010 * a + 101 * b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) ∧
  (A % 14 = 0) ∧
  (A = (d1 * 100 + d2 * 10 + d1) * 101)
  :=
sorry

end largest_four_digit_divisible_by_14_l2244_224493


namespace race_track_width_l2244_224468

noncomputable def width_of_race_track (C_inner : ℝ) (r_outer : ℝ) : ℝ :=
  let r_inner := C_inner / (2 * Real.pi)
  r_outer - r_inner

theorem race_track_width : 
  width_of_race_track 880 165.0563499208679 = 25.0492072460867 :=
by
  sorry

end race_track_width_l2244_224468


namespace Ivanka_more_months_l2244_224447

variable (I : ℕ) (W : ℕ)

theorem Ivanka_more_months (hW : W = 18) (hI_W : I + W = 39) : I - W = 3 :=
by
  sorry

end Ivanka_more_months_l2244_224447


namespace cauchy_schwarz_inequality_l2244_224452

theorem cauchy_schwarz_inequality
  (x1 y1 z1 x2 y2 z2 : ℝ) :
  (x1 * x2 + y1 * y2 + z1 * z2) ^ 2 ≤ (x1 ^ 2 + y1 ^ 2 + z1 ^ 2) * (x2 ^ 2 + y2 ^ 2 + z2 ^ 2) := 
sorry

end cauchy_schwarz_inequality_l2244_224452


namespace probability_neither_red_nor_purple_l2244_224465

theorem probability_neither_red_nor_purple :
  (100 - (47 + 3)) / 100 = 0.5 :=
by sorry

end probability_neither_red_nor_purple_l2244_224465


namespace fraction_paint_remaining_l2244_224425

theorem fraction_paint_remaining :
  let original_paint := 1
  let first_day_usage := original_paint / 4
  let paint_remaining_after_first_day := original_paint - first_day_usage
  let second_day_usage := paint_remaining_after_first_day / 2
  let paint_remaining_after_second_day := paint_remaining_after_first_day - second_day_usage
  let third_day_usage := paint_remaining_after_second_day / 3
  let paint_remaining_after_third_day := paint_remaining_after_second_day - third_day_usage
  paint_remaining_after_third_day = original_paint / 4 := 
by
  sorry

end fraction_paint_remaining_l2244_224425


namespace bananas_first_day_l2244_224486

theorem bananas_first_day (x : ℕ) (h : x + (x + 6) + (x + 12) + (x + 18) + (x + 24) = 100) : x = 8 := by
  sorry

end bananas_first_day_l2244_224486


namespace cash_sales_is_48_l2244_224490

variable (total_sales : ℝ) (credit_fraction : ℝ) (cash_sales : ℝ)

-- Conditions: Total sales were $80, 2/5 of the total sales were credit sales
def problem_conditions := total_sales = 80 ∧ credit_fraction = 2/5 ∧ cash_sales = (1 - credit_fraction) * total_sales

-- Question: Prove that the amount of cash sales Mr. Brandon made is $48.
theorem cash_sales_is_48 (h : problem_conditions total_sales credit_fraction cash_sales) : 
  cash_sales = 48 :=
by
  sorry

end cash_sales_is_48_l2244_224490


namespace peaches_in_each_basket_l2244_224422

variable (R : ℕ)

theorem peaches_in_each_basket (h : 6 * R = 96) : R = 16 :=
by
  sorry

end peaches_in_each_basket_l2244_224422


namespace sum_of_areas_l2244_224480

theorem sum_of_areas :
  (∑' n : ℕ, Real.pi * (1 / 9 ^ n)) = (9 * Real.pi) / 8 :=
by
  sorry

end sum_of_areas_l2244_224480


namespace gcd_triangular_number_l2244_224434

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem gcd_triangular_number (n : ℕ) (h : n > 2) :
  ∃ k, n = 12 * k + 2 → gcd (6 * triangular_number n) (n - 2) = 12 :=
  sorry

end gcd_triangular_number_l2244_224434


namespace rectangle_area_l2244_224446

theorem rectangle_area (a b : ℝ) (x : ℝ) 
  (h1 : x^2 + (x / 2)^2 = (a + b)^2) 
  (h2 : x > 0) : 
  x * (x / 2) = (2 * (a + b)^2) / 5 := 
by 
  sorry

end rectangle_area_l2244_224446


namespace trigonometric_inequality_l2244_224442

theorem trigonometric_inequality (a b x : ℝ) :
  (Real.sin x + a * Real.cos x) * (Real.sin x + b * Real.cos x) ≤ 1 + ( (a + b) / 2 )^2 :=
by
  sorry

end trigonometric_inequality_l2244_224442


namespace ratio_of_investments_l2244_224459

theorem ratio_of_investments {A B C : ℝ} (x y z k : ℝ)
  (h1 : B - A = 100)
  (h2 : A + B + C = 2900)
  (h3 : A = 6 * k)
  (h4 : B = 5 * k)
  (h5 : C = 4 * k) : 
  (x / y = 6 / 5) ∧ (y / z = 5 / 4) ∧ (x / z = 6 / 4) :=
by
  sorry

end ratio_of_investments_l2244_224459


namespace P_in_first_quadrant_l2244_224405

def point_in_first_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 > 0

theorem P_in_first_quadrant (k : ℝ) (h : k > 0) : point_in_first_quadrant (3, k) :=
by
  sorry

end P_in_first_quadrant_l2244_224405


namespace triangle_inequality_l2244_224400

variables {a b c x y z : ℝ}

theorem triangle_inequality 
  (h1 : ∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  (h2 : x + y + z = 0) :
  a^2 * y * z + b^2 * z * x + c^2 * x * y ≤ 0 :=
sorry

end triangle_inequality_l2244_224400


namespace literature_club_students_neither_english_nor_french_l2244_224461

theorem literature_club_students_neither_english_nor_french
  (total_students english_students french_students both_students : ℕ)
  (h1 : total_students = 120)
  (h2 : english_students = 72)
  (h3 : french_students = 52)
  (h4 : both_students = 12) :
  (total_students - ((english_students - both_students) + (french_students - both_students) + both_students) = 8) :=
by
  sorry

end literature_club_students_neither_english_nor_french_l2244_224461


namespace foil_covered_prism_width_l2244_224487

theorem foil_covered_prism_width
    (l w h : ℕ)
    (inner_volume : l * w * h = 128)
    (width_length_relation : w = 2 * l)
    (width_height_relation : w = 2 * h) :
    (w + 2) = 10 := 
sorry

end foil_covered_prism_width_l2244_224487


namespace negation_of_universal_proposition_l2244_224473

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0 :=
by sorry

end negation_of_universal_proposition_l2244_224473


namespace limit_of_nested_radical_l2244_224475

theorem limit_of_nested_radical :
  ∃ F : ℝ, F = 43 ∧ F = Real.sqrt (86 + 41 * F) :=
sorry

end limit_of_nested_radical_l2244_224475


namespace books_on_shelves_l2244_224494

-- Definitions based on the problem conditions.
def bookshelves : ℕ := 1250
def books_per_shelf : ℕ := 45
def total_books : ℕ := 56250

-- Theorem statement
theorem books_on_shelves : bookshelves * books_per_shelf = total_books := 
by
  sorry

end books_on_shelves_l2244_224494


namespace determine_x_l2244_224448

/-
  Determine \( x \) when \( y = 19 \)
  given the ratio of \( 5x - 3 \) to \( y + 10 \) is constant,
  and when \( x = 3 \), \( y = 4 \).
-/

theorem determine_x (x y k : ℚ) (h1 : ∀ x y, (5 * x - 3) / (y + 10) = k)
  (h2 : 5 * 3 - 3 / (4 + 10) = k) : x = 39 / 7 :=
sorry

end determine_x_l2244_224448


namespace cos_monotonic_increasing_interval_l2244_224407

open Real

noncomputable def monotonic_increasing_interval (k : ℤ) : Set ℝ :=
  {x : ℝ | k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6}

theorem cos_monotonic_increasing_interval (k : ℤ) :
  ∀ x : ℝ,
    (∃ y, y = cos (π / 3 - 2 * x)) →
    (monotonic_increasing_interval k x) :=
by
  sorry

end cos_monotonic_increasing_interval_l2244_224407


namespace cubic_difference_l2244_224426

theorem cubic_difference (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a^3 - b^3 = 108 :=
sorry

end cubic_difference_l2244_224426


namespace product_three_numbers_l2244_224466

theorem product_three_numbers 
  (a b c : ℝ)
  (h1 : a + b + c = 30)
  (h2 : a = 3 * (b + c))
  (h3 : b = 5 * c) : 
  a * b * c = 176 := 
by
  sorry

end product_three_numbers_l2244_224466


namespace min_ν_of_cubic_eq_has_3_positive_real_roots_l2244_224431

open Real

noncomputable def cubic_eq (x θ : ℝ) : ℝ :=
  x^3 * sin θ - (sin θ + 2) * x^2 + 6 * x - 4

noncomputable def ν (θ : ℝ) : ℝ :=
  (9 * sin θ ^ 2 - 4 * sin θ + 3) / 
  ((1 - cos θ) * (2 * cos θ - 6 * sin θ - 3 * sin (2 * θ) + 2))

theorem min_ν_of_cubic_eq_has_3_positive_real_roots :
  (∀ x:ℝ, cubic_eq x θ = 0 → 0 < x) →
  ν θ = 621 / 8 :=
sorry

end min_ν_of_cubic_eq_has_3_positive_real_roots_l2244_224431


namespace uniform_prob_correct_l2244_224496

noncomputable def uniform_prob_within_interval 
  (α β γ δ : ℝ) 
  (h₁ : α ≤ β) 
  (h₂ : α ≤ γ) 
  (h₃ : γ < δ) 
  (h₄ : δ ≤ β) : ℝ :=
  (δ - γ) / (β - α)

theorem uniform_prob_correct 
  (α β γ δ : ℝ) 
  (hαβ : α ≤ β) 
  (hαγ : α ≤ γ) 
  (hγδ : γ < δ) 
  (hδβ : δ ≤ β) :
  uniform_prob_within_interval α β γ δ hαβ hαγ hγδ hδβ = (δ - γ) / (β - α) := sorry

end uniform_prob_correct_l2244_224496


namespace simplify_expression_l2244_224413

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b ^ 2 + 2 * b) - 2 * b ^ 2 = 9 * b ^ 3 + 4 * b ^ 2 :=
by
  sorry

end simplify_expression_l2244_224413


namespace cos_of_angle_l2244_224430

theorem cos_of_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.cos (3 * Real.pi / 2 + 2 * θ) = 3 / 5 := 
by
  sorry

end cos_of_angle_l2244_224430


namespace total_cost_of_horse_and_saddle_l2244_224489

noncomputable def saddle_cost : ℝ := 1000
noncomputable def horse_cost : ℝ := 4 * saddle_cost
noncomputable def total_cost : ℝ := saddle_cost + horse_cost

theorem total_cost_of_horse_and_saddle :
    total_cost = 5000 := by
  sorry

end total_cost_of_horse_and_saddle_l2244_224489


namespace unique_solution_l2244_224403

noncomputable def valid_solutions (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + (2 - a) * x + 1 = 0 ∧ -1 < x ∧ x ≤ 3 ∧ x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2

theorem unique_solution (a : ℝ) :
  (valid_solutions a) ↔ (a = 4.5 ∨ (a < 0) ∨ (a > 16 / 3)) := 
sorry

end unique_solution_l2244_224403


namespace sum_of_reciprocal_AP_l2244_224478

theorem sum_of_reciprocal_AP (a1 a2 a3 : ℝ) (d : ℝ)
  (h1 : a1 + a2 + a3 = 11/18)
  (h2 : 1/a1 + 1/a2 + 1/a3 = 18)
  (h3 : 1/a2 = 1/a1 + d)
  (h4 : 1/a3 = 1/a1 + 2*d) :
  (a1 = 1/9 ∧ a2 = 1/6 ∧ a3 = 1/3) ∨ (a1 = 1/3 ∧ a2 = 1/6 ∧ a3 = 1/9) :=
sorry

end sum_of_reciprocal_AP_l2244_224478


namespace gcd_mn_eq_one_l2244_224449

def m : ℤ := 123^2 + 235^2 - 347^2
def n : ℤ := 122^2 + 234^2 - 348^2

theorem gcd_mn_eq_one : Int.gcd m n = 1 := 
by
  sorry

end gcd_mn_eq_one_l2244_224449


namespace average_percentage_decrease_l2244_224453

theorem average_percentage_decrease :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 100 * (1 - x)^2 = 81 ∧ x = 0.1 :=
by
  sorry

end average_percentage_decrease_l2244_224453


namespace molly_age_is_63_l2244_224435

variable (Sandy_age Molly_age : ℕ)

theorem molly_age_is_63 (h1 : Sandy_age = 49) (h2 : Sandy_age / Molly_age = 7 / 9) : Molly_age = 63 :=
by
  sorry

end molly_age_is_63_l2244_224435


namespace find_term_number_l2244_224482

variable {α : ℝ} (b : ℕ → ℝ) (q : ℝ)

namespace GeometricProgression

noncomputable def geometric_progression (b : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ (n : ℕ), b (n + 1) = b n * q

noncomputable def satisfies_conditions (α : ℝ) (b : ℕ → ℝ) : Prop :=
  b 25 = 2 * Real.tan α ∧ b 31 = 2 * Real.sin α

theorem find_term_number (α : ℝ) (b : ℕ → ℝ) (q : ℝ) (hb : geometric_progression b q) (hc : satisfies_conditions α b) :
  ∃ n, b n = Real.sin (2 * α) ∧ n = 37 :=
sorry

end GeometricProgression

end find_term_number_l2244_224482


namespace Petya_can_determine_weight_l2244_224412

theorem Petya_can_determine_weight (n : ℕ) (distinct_weights : Fin n → ℕ) 
  (device : (Fin 10 → Fin n) → ℕ) (ten_thousand_weights : n = 10000)
  (no_two_same : (∀ i j : Fin n, i ≠ j → distinct_weights i ≠ distinct_weights j)) :
  ∃ i : Fin n, ∃ w : ℕ, distinct_weights i = w :=
by
  sorry

end Petya_can_determine_weight_l2244_224412


namespace sum_of_digits_of_B_is_7_l2244_224476

theorem sum_of_digits_of_B_is_7 : 
  let A := 16 ^ 16
  let sum_digits (n : ℕ) : ℕ := n.digits 10 |>.sum
  let S := sum_digits
  let B := S (S A)
  sum_digits B = 7 :=
sorry

end sum_of_digits_of_B_is_7_l2244_224476


namespace probability_correct_l2244_224401

noncomputable def probability_one_white_one_black
    (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) (draw_balls : ℕ) :=
if (total_balls = 4) ∧ (white_balls = 2) ∧ (black_balls = 2) ∧ (draw_balls = 2) then
  (2 * 2) / (Nat.choose total_balls draw_balls : ℚ)
else
  0

theorem probability_correct:
  probability_one_white_one_black 4 2 2 2 = 2 / 3 :=
by
  sorry

end probability_correct_l2244_224401


namespace multiple_rate_is_correct_l2244_224414

-- Define Lloyd's standard working hours per day
def regular_hours_per_day : ℝ := 7.5

-- Define Lloyd's standard hourly rate
def regular_rate : ℝ := 3.5

-- Define the total hours worked on a specific day
def total_hours_worked : ℝ := 10.5

-- Define the total earnings for that specific day
def total_earnings : ℝ := 42.0

-- Define the function to calculate the multiple of the regular rate for excess hours
noncomputable def multiple_of_regular_rate (r_hours : ℝ) (r_rate : ℝ) (t_hours : ℝ) (t_earnings : ℝ) : ℝ :=
  let regular_earnings := r_hours * r_rate
  let excess_hours := t_hours - r_hours
  let excess_earnings := t_earnings - regular_earnings
  (excess_earnings / excess_hours) / r_rate

-- The statement to be proved
theorem multiple_rate_is_correct : 
  multiple_of_regular_rate regular_hours_per_day regular_rate total_hours_worked total_earnings = 1.5 :=
by
  sorry

end multiple_rate_is_correct_l2244_224414


namespace time_to_count_60_envelopes_is_40_time_to_count_90_envelopes_is_10_l2244_224441

noncomputable def time_to_count_envelopes (num_envelopes : ℕ) : ℕ :=
(num_envelopes / 10) * 10

theorem time_to_count_60_envelopes_is_40 :
  time_to_count_envelopes 60 = 40 := 
sorry

theorem time_to_count_90_envelopes_is_10 :
  time_to_count_envelopes 90 = 10 := 
sorry

end time_to_count_60_envelopes_is_40_time_to_count_90_envelopes_is_10_l2244_224441


namespace four_digit_number_sum_eq_4983_l2244_224417

def reverse_number (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n % 1000) / 100
  let d3 := (n % 100) / 10
  let d4 := n % 10
  1000 * d4 + 100 * d3 + 10 * d2 + d1

theorem four_digit_number_sum_eq_4983 (n : ℕ) :
  n + reverse_number n = 4983 ↔ n = 1992 ∨ n = 2991 :=
by sorry

end four_digit_number_sum_eq_4983_l2244_224417


namespace center_of_circle_l2244_224409

-- Definition of the main condition: the given circle equation
def circle_equation (x y : ℝ) := x^2 + y^2 = 10 * x - 4 * y + 14

-- Statement to prove: that x + y = 3 when (x, y) is the center of the circle described by circle_equation
theorem center_of_circle {x y : ℝ} (h : circle_equation x y) : x + y = 3 := 
by 
  sorry

end center_of_circle_l2244_224409


namespace total_number_of_marbles_is_1050_l2244_224492

def total_marbles : Nat :=
  let marbles_in_second_bowl := 600
  let marbles_in_first_bowl := (3 * marbles_in_second_bowl) / 4
  marbles_in_first_bowl + marbles_in_second_bowl

theorem total_number_of_marbles_is_1050 : total_marbles = 1050 := by
  sorry

end total_number_of_marbles_is_1050_l2244_224492


namespace max_value_f1_on_interval_range_of_a_g_increasing_l2244_224443

noncomputable def f1 (x : ℝ) : ℝ := 2 * x^2 + x + 2

theorem max_value_f1_on_interval : 
  (∀ x, x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) → f1 x ≤ 5) ∧ 
  (∃ x, x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) ∧ f1 x = 5) :=
sorry

noncomputable def f2 (a x : ℝ) : ℝ := a * x^2 + (a - 1) * x + a

theorem range_of_a (a : ℝ) : 
  (∀ x, x ∈ Set.Icc (1 : ℝ) (2 : ℝ) → f2 a x / x ≥ 2) → a ≥ 1 :=
sorry

noncomputable def g (a x : ℝ) : ℝ := a * x^2 + (a - 1) * x + a + (1 - (a-1) * x^2) / x

theorem g_increasing (a : ℝ) : 
  (∀ x1 x2, (2 < x1 ∧ x1 < x2 ∧ x2 < 3) → g a x1 < g a x2) → a ≥ 1 / 16 :=
sorry

end max_value_f1_on_interval_range_of_a_g_increasing_l2244_224443


namespace fraction_available_on_third_day_l2244_224437

noncomputable def liters_used_on_first_day (initial_amount : ℕ) : ℕ :=
  (initial_amount / 2)

noncomputable def liters_added_on_second_day : ℕ :=
  1

noncomputable def original_solution : ℕ :=
  4

noncomputable def remaining_solution_after_first_day : ℕ :=
  original_solution - liters_used_on_first_day original_solution

noncomputable def remaining_solution_after_second_day : ℕ :=
  remaining_solution_after_first_day + liters_added_on_second_day

noncomputable def fraction_of_original_solution : ℚ :=
  remaining_solution_after_second_day / original_solution

theorem fraction_available_on_third_day : fraction_of_original_solution = 3 / 4 :=
by
  sorry

end fraction_available_on_third_day_l2244_224437


namespace original_price_of_computer_l2244_224457

theorem original_price_of_computer (P : ℝ) (h1 : 1.20 * P = 351) (h2 : 2 * P = 585) : P = 292.5 :=
by
  sorry

end original_price_of_computer_l2244_224457
