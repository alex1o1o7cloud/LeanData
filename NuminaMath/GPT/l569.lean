import Mathlib

namespace area_ratio_l569_56903

theorem area_ratio (l b r : ℝ) (h1 : l = 2 * b) (h2 : 6 * b = 2 * π * r) :
  (l * b) / (π * r ^ 2) = 2 * π / 9 :=
by {
  sorry
}

end area_ratio_l569_56903


namespace hunter_3_proposal_l569_56929

theorem hunter_3_proposal {hunter1_coins hunter2_coins hunter3_coins : ℕ} :
  hunter3_coins = 99 ∧ hunter1_coins = 1 ∧ (hunter1_coins + hunter3_coins + hunter2_coins = 100) :=
  sorry

end hunter_3_proposal_l569_56929


namespace more_math_than_reading_l569_56925

def pages_reading := 4
def pages_math := 7

theorem more_math_than_reading : pages_math - pages_reading = 3 :=
by
  sorry

end more_math_than_reading_l569_56925


namespace sequence_from_625_to_629_l569_56915

def arrows_repeating_pattern (n : ℕ) : ℕ := n % 5

theorem sequence_from_625_to_629 :
  arrows_repeating_pattern 625 = 0 ∧ arrows_repeating_pattern 629 = 4 →
  ∃ (seq : ℕ → ℕ), 
    (seq 0 = arrows_repeating_pattern 625) ∧
    (seq 1 = arrows_repeating_pattern (625 + 1)) ∧
    (seq 2 = arrows_repeating_pattern (625 + 2)) ∧
    (seq 3 = arrows_repeating_pattern (625 + 3)) ∧
    (seq 4 = arrows_repeating_pattern 629) := 
sorry

end sequence_from_625_to_629_l569_56915


namespace roy_consumes_tablets_in_225_minutes_l569_56969

variables 
  (total_tablets : ℕ) 
  (time_per_tablet : ℕ)

def total_time_to_consume_all_tablets 
  (total_tablets : ℕ) 
  (time_per_tablet : ℕ) : ℕ :=
  (total_tablets - 1) * time_per_tablet

theorem roy_consumes_tablets_in_225_minutes 
  (h1 : total_tablets = 10) 
  (h2 : time_per_tablet = 25) : 
  total_time_to_consume_all_tablets total_tablets time_per_tablet = 225 :=
by
  -- Proof goes here
  sorry

end roy_consumes_tablets_in_225_minutes_l569_56969


namespace average_speed_car_y_l569_56997

-- Defining the constants based on the problem conditions
def speedX : ℝ := 35
def timeDifference : ℝ := 1.2  -- This is 72 minutes converted to hours
def distanceFromStartOfY : ℝ := 294

-- Defining the main statement
theorem average_speed_car_y : 
  ( ∀ timeX timeY distanceX distanceY : ℝ, 
      timeX = timeY + timeDifference ∧
      distanceX = speedX * timeX ∧
      distanceY = distanceFromStartOfY ∧
      distanceX = distanceFromStartOfY + speedX * timeDifference
  → distanceY / timeX = 30.625) :=
sorry

end average_speed_car_y_l569_56997


namespace cost_price_of_article_l569_56970

theorem cost_price_of_article (x : ℝ) (h : 57 - x = x - 43) : x = 50 := 
by 
  sorry

end cost_price_of_article_l569_56970


namespace add_fifteen_sub_fifteen_l569_56906

theorem add_fifteen (n : ℕ) (m : ℕ) : n + m = 195 :=
by {
  sorry  -- placeholder for the actual proof
}

theorem sub_fifteen (n : ℕ) (m : ℕ) : n - m = 165 :=
by {
  sorry  -- placeholder for the actual proof
}

-- Let's instantiate these theorems with the specific values from the problem:
noncomputable def verify_addition : 180 + 15 = 195 :=
by exact add_fifteen 180 15

noncomputable def verify_subtraction : 180 - 15 = 165 :=
by exact sub_fifteen 180 15

end add_fifteen_sub_fifteen_l569_56906


namespace intersect_not_A_B_l569_56945

open Set

-- Define the universal set U
def U := ℝ

-- Define set A
def A := {x : ℝ | x ≤ 3}

-- Define set B
def B := {x : ℝ | x ≤ 6}

-- Define the complement of A in U
def not_A := {x : ℝ | x > 3}

-- The proof problem
theorem intersect_not_A_B :
  (not_A ∩ B) = {x : ℝ | 3 < x ∧ x ≤ 6} :=
sorry

end intersect_not_A_B_l569_56945


namespace outermost_diameter_l569_56904

def radius_of_fountain := 6 -- derived from the information that 12/2 = 6
def width_of_garden := 9
def width_of_inner_walking_path := 3
def width_of_outer_walking_path := 7

theorem outermost_diameter :
  2 * (radius_of_fountain + width_of_garden + width_of_inner_walking_path + width_of_outer_walking_path) = 50 :=
by
  sorry

end outermost_diameter_l569_56904


namespace cos_60_eq_half_l569_56962

theorem cos_60_eq_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_60_eq_half_l569_56962


namespace division_addition_example_l569_56917

theorem division_addition_example : 12 / (1 / 6) + 3 = 75 := by
  sorry

end division_addition_example_l569_56917


namespace train_length_l569_56949

/-- Proof problem: 
  Given the speed of a train is 52 km/hr and it crosses a 280-meter long platform in 18 seconds,
  prove that the length of the train is 259.92 meters.
-/
theorem train_length (speed_kmh : ℕ) (platform_length : ℕ) (time_sec : ℕ) (speed_mps : ℝ) 
  (distance_covered : ℝ) (train_length : ℝ) :
  speed_kmh = 52 → platform_length = 280 → time_sec = 18 → 
  speed_mps = (speed_kmh * 1000) / 3600 → distance_covered = speed_mps * time_sec →
  train_length = distance_covered - platform_length →
  train_length = 259.92 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end train_length_l569_56949


namespace sandy_fingernails_length_l569_56908

/-- 
Sandy, who just turned 12 this month, has a goal for tying the world record for longest fingernails, 
which is 26 inches. Her fingernails grow at a rate of one-tenth of an inch per month. 
She will be 32 when she achieves the world record. 
Prove that her fingernails are currently 2 inches long.
-/
theorem sandy_fingernails_length 
  (current_age : ℕ) (world_record_length : ℝ) (growth_rate : ℝ) (years_to_achieve : ℕ) : 
  current_age = 12 → 
  world_record_length = 26 → 
  growth_rate = 0.1 → 
  years_to_achieve = 20 →
  (world_record_length - growth_rate * 12 * years_to_achieve) = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end sandy_fingernails_length_l569_56908


namespace candy_store_problem_l569_56939

variable (S : ℝ)
variable (not_caught_percentage : ℝ) (sample_percentage : ℝ)
variable (caught_percentage : ℝ := 1 - not_caught_percentage)

theorem candy_store_problem
  (h1 : not_caught_percentage = 0.15)
  (h2 : sample_percentage = 25.88235294117647) :
  caught_percentage * sample_percentage = 22 := by
  sorry

end candy_store_problem_l569_56939


namespace car_travel_distance_l569_56919

theorem car_travel_distance :
  ∃ S : ℝ, 
    (S > 0) ∧ 
    (∃ v1 v2 t1 t2 t3 t4 : ℝ, 
      (S / 2 = v1 * t1) ∧ (26.25 = v2 * t2) ∧ 
      (S / 2 = v2 * t3) ∧ (31.2 = v1 * t4) ∧ 
      (∃ k : ℝ, k = (S - 31.2) / (v1 + v2) ∧ k > 0 ∧ 
        (S = 58))) := sorry

end car_travel_distance_l569_56919


namespace determine_base_l569_56938

theorem determine_base (b : ℕ) (h : (3 * b + 1)^2 = b^3 + 2 * b + 1) : b = 10 :=
by
  sorry

end determine_base_l569_56938


namespace f_half_l569_56983

theorem f_half (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 - 2 * x) = 1 / (x ^ 2)) :
  f (1 / 2) = 16 :=
sorry

end f_half_l569_56983


namespace perimeter_percent_increase_l569_56944

noncomputable def side_increase (s₁ s₂_ratio s₃_ratio s₄_ratio s₅_ratio : ℝ) : ℝ :=
  let s₂ := s₂_ratio * s₁
  let s₃ := s₃_ratio * s₂
  let s₄ := s₄_ratio * s₃
  let s₅ := s₅_ratio * s₄
  s₅

theorem perimeter_percent_increase (s₁ : ℝ) (s₂_ratio s₃_ratio s₄_ratio s₅_ratio : ℝ) (P₁ := 3 * s₁)
    (P₅ := 3 * side_increase s₁ s₂_ratio s₃_ratio s₄_ratio s₅_ratio) :
    s₁ = 4 → s₂_ratio = 1.5 → s₃_ratio = 1.3 → s₄_ratio = 1.5 → s₅_ratio = 1.3 →
    P₅ = 45.63 →
    ((P₅ - P₁) / P₁) * 100 = 280.3 :=
by
  intros
  -- proof goes here
  sorry

end perimeter_percent_increase_l569_56944


namespace find_x_l569_56987

theorem find_x (n : ℕ) (h_odd : n % 2 = 1)
  (h_three_primes : ∃ (p1 p2 p3 : ℕ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ 
    11 = p1 ∧ (7 ^ n + 1) = p1 * p2 * p3) :
  (7 ^ n + 1) = 16808 :=
by
  sorry

end find_x_l569_56987


namespace sequence_value_2009_l569_56947

theorem sequence_value_2009 
  (a : ℕ → ℝ)
  (h_recur : ∀ n ≥ 2, a n = a (n - 1) * a (n + 1))
  (h_a1 : a 1 = 1 + Real.sqrt 3)
  (h_a1776 : a 1776 = 4 + Real.sqrt 3) :
  a 2009 = (3 / 2) + (3 * Real.sqrt 3 / 2) := 
sorry

end sequence_value_2009_l569_56947


namespace age_ratio_l569_56995

theorem age_ratio (S M : ℕ) (h₁ : M = S + 35) (h₂ : S = 33) : 
  (M + 2) / (S + 2) = 2 :=
by
  -- proof goes here
  sorry

end age_ratio_l569_56995


namespace exists_multiple_with_digits_0_or_1_l569_56968

theorem exists_multiple_with_digits_0_or_1 (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, (k % n = 0) ∧ (∀ digit ∈ k.digits 10, digit = 0 ∨ digit = 1) ∧ (k.digits 10).length ≤ n :=
sorry

end exists_multiple_with_digits_0_or_1_l569_56968


namespace correct_answer_l569_56985

-- Define the problem conditions and question
def equation (y : ℤ) : Prop := y + 2 = -3

-- Prove that the correct answer is y = -5
theorem correct_answer : ∀ y : ℤ, equation y → y = -5 :=
by
  intros y h
  unfold equation at h
  linarith

end correct_answer_l569_56985


namespace carbon_copies_after_folding_l569_56901

-- Define the initial condition of sheets and carbon papers
def initial_sheets : ℕ := 3
def initial_carbons : ℕ := 2

-- Define the condition of folding the paper
def fold_paper (sheets carbons : ℕ) : ℕ := sheets * 2

-- Statement of the problem
theorem carbon_copies_after_folding : (fold_paper initial_sheets initial_carbons - initial_sheets + initial_carbons) = 4 :=
by
  sorry

end carbon_copies_after_folding_l569_56901


namespace p_has_49_l569_56923

theorem p_has_49 (P : ℝ) (h : P = (2/7) * P + 35) : P = 49 :=
by
  sorry

end p_has_49_l569_56923


namespace tangent_line_at_zero_l569_56905

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem tangent_line_at_zero : ∀ x : ℝ, x = 0 → Real.exp x * Real.sin x = 0 ∧ (Real.exp x * (Real.sin x + Real.cos x)) = 1 → (∀ y, y = x) :=
  by
    sorry

end tangent_line_at_zero_l569_56905


namespace max_regions_with_6_chords_l569_56910

-- Definition stating the number of regions created by k chords
def regions_by_chords (k : ℕ) : ℕ :=
  1 + (k * (k + 1)) / 2

-- Lean statement for the proof problem
theorem max_regions_with_6_chords : regions_by_chords 6 = 22 :=
  by sorry

end max_regions_with_6_chords_l569_56910


namespace price_per_maple_tree_l569_56921

theorem price_per_maple_tree 
  (cabin_price : ℕ) (initial_cash : ℕ) (remaining_cash : ℕ)
  (num_cypress : ℕ) (price_cypress : ℕ)
  (num_pine : ℕ) (price_pine : ℕ)
  (num_maple : ℕ) 
  (total_raised_from_trees : ℕ) :
  cabin_price = 129000 ∧ 
  initial_cash = 150 ∧ 
  remaining_cash = 350 ∧ 
  num_cypress = 20 ∧ 
  price_cypress = 100 ∧ 
  num_pine = 600 ∧ 
  price_pine = 200 ∧ 
  num_maple = 24 ∧ 
  total_raised_from_trees = 129350 - initial_cash → 
  (price_maple : ℕ) = 300 :=
by 
  sorry

end price_per_maple_tree_l569_56921


namespace potato_bag_weight_l569_56963

theorem potato_bag_weight (w : ℕ) (h₁ : w = 36) : w = 36 :=
by
  sorry

end potato_bag_weight_l569_56963


namespace number_of_subsets_B_l569_56991

def A : Set ℕ := {1, 3}
def C : Set ℕ := {1, 3, 5}

theorem number_of_subsets_B : ∃ (n : ℕ), (∀ B : Set ℕ, A ∪ B = C → n = 4) :=
sorry

end number_of_subsets_B_l569_56991


namespace sugar_already_put_in_l569_56993

-- Definitions based on conditions
def required_sugar : ℕ := 13
def additional_sugar_needed : ℕ := 11

-- Theorem to be proven
theorem sugar_already_put_in :
  required_sugar - additional_sugar_needed = 2 := by
  sorry

end sugar_already_put_in_l569_56993


namespace integer_pairs_perfect_squares_l569_56965

theorem integer_pairs_perfect_squares (a b : ℤ) :
  (∃ k : ℤ, (a, b) = (k^2, 0) ∨ (a, b) = (0, k^2) ∨ (a, b) = (k, 1-k) ∨ (a, b) = (-6, -5) ∨ (a, b) = (-5, -6) ∨ (a, b) = (-4, -4))
  ↔ 
  (∃ x1 x2 : ℤ, a^2 + 4*b = x1^2 ∧ b^2 + 4*a = x2^2) :=
sorry

end integer_pairs_perfect_squares_l569_56965


namespace base_height_ratio_l569_56959

-- Define the conditions
def cultivation_cost : ℝ := 333.18
def rate_per_hectare : ℝ := 24.68
def base_of_field : ℝ := 300
def height_of_field : ℝ := 300

-- Prove the ratio of base to height is 1
theorem base_height_ratio (b h : ℝ) (cost rate : ℝ)
  (h1 : cost = 333.18) (h2 : rate = 24.68) 
  (h3 : b = 300) (h4 : h = 300) : b / h = 1 :=
by
  sorry

end base_height_ratio_l569_56959


namespace a_11_is_12_l569_56973

-- Definitions and conditions
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d
def a_2 (a : ℕ → ℝ) := a 2 = 3
def a_6 (a : ℕ → ℝ) := a 6 = 7

-- The statement to prove
theorem a_11_is_12 (a : ℕ → ℝ) (h_arith : arithmetic_sequence a) (h_a2 : a_2 a) (h_a6 : a_6 a) : a 11 = 12 :=
  sorry

end a_11_is_12_l569_56973


namespace maria_min_score_fourth_quarter_l569_56979

theorem maria_min_score_fourth_quarter (x : ℝ) :
  (82 + 77 + 78 + x) / 4 ≥ 85 ↔ x ≥ 103 :=
by
  sorry

end maria_min_score_fourth_quarter_l569_56979


namespace male_percentage_l569_56972

theorem male_percentage (total_employees : ℕ)
  (males_below_50 : ℕ)
  (percentage_males_at_least_50 : ℝ)
  (male_percentage : ℝ) :
  total_employees = 2200 →
  males_below_50 = 616 →
  percentage_males_at_least_50 = 0.3 → 
  male_percentage = 40 :=
by
  sorry

end male_percentage_l569_56972


namespace triangle_inequality_l569_56955

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  (a + b) / (a + b + c) > 1 / 2 :=
sorry

end triangle_inequality_l569_56955


namespace mow_lawn_time_l569_56943

noncomputable def time_to_mow (lawn_length lawn_width: ℝ) 
(swat_width overlap width_conversion: ℝ) (speed: ℝ) : ℝ :=
(lawn_length * lawn_width) / (((swat_width - overlap) / width_conversion) * lawn_length * speed)

theorem mow_lawn_time : 
  time_to_mow 120 180 30 6 12 6000 = 1.8 := 
by
  -- Given:
  -- Lawn dimensions: 120 feet by 180 feet
  -- Mower swath: 30 inches with 6 inches overlap
  -- Walking speed: 6000 feet per hour
  -- Conversion factor: 12 inches = 1 foot
  sorry

end mow_lawn_time_l569_56943


namespace smallest_N_for_circular_table_l569_56961

/--
  Given a circular table with 60 chairs, prove that the smallest number of people, N,
  such that any additional person must sit next to someone already seated is 20.
-/
theorem smallest_N_for_circular_table (N : ℕ) (h : N = 20) : 
  ∀ (next_seated : ℕ), next_seated ≤ N → (∃ i : ℕ, i < N ∧ next_seated = i + 1 ∨ next_seated = i - 1) :=
by
  sorry

end smallest_N_for_circular_table_l569_56961


namespace possible_triangle_perimeters_l569_56953

theorem possible_triangle_perimeters :
  {p | ∃ (a b c : ℝ), ((a = 3 ∨ a = 6) ∧ (b = 3 ∨ b = 6) ∧ (c = 3 ∨ c = 6)) ∧
                        (a + b > c) ∧ (b + c > a) ∧ (c + a > b) ∧
                        p = a + b + c} = {9, 15, 18} :=
by
  sorry

end possible_triangle_perimeters_l569_56953


namespace triangle_area_is_24_l569_56982

-- Define the vertices
def vertex1 : ℝ × ℝ := (3, 2)
def vertex2 : ℝ × ℝ := (3, -4)
def vertex3 : ℝ × ℝ := (11, -4)

-- Define a function to calculate the area of a triangle given its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1))

-- Prove the area of the triangle with the given vertices is 24 square units
theorem triangle_area_is_24 : triangle_area vertex1 vertex2 vertex3 = 24 := by
  sorry

end triangle_area_is_24_l569_56982


namespace height_of_tank_A_l569_56994

theorem height_of_tank_A (C_A C_B h_B : ℝ) (capacity_ratio : ℝ) :
  C_A = 8 → C_B = 10 → h_B = 8 → capacity_ratio = 0.4800000000000001 →
  ∃ h_A : ℝ, h_A = 6 := by
  intros hCA hCB hHB hCR
  sorry

end height_of_tank_A_l569_56994


namespace cost_of_mozzarella_cheese_l569_56902

-- Define the problem conditions as Lean definitions
def blendCostPerKg : ℝ := 696.05
def romanoCostPerKg : ℝ := 887.75
def weightMozzarella : ℝ := 19
def weightRomano : ℝ := 18.999999999999986  -- Practically the same as 19 in context
def totalWeight : ℝ := weightMozzarella + weightRomano

-- Define the expected result for the cost per kilogram of mozzarella cheese
def expectedMozzarellaCostPerKg : ℝ := 504.40

-- Theorem statement to verify the cost of mozzarella cheese
theorem cost_of_mozzarella_cheese :
  weightMozzarella * (expectedMozzarellaCostPerKg : ℝ) + weightRomano * romanoCostPerKg = totalWeight * blendCostPerKg := by
  sorry

end cost_of_mozzarella_cheese_l569_56902


namespace find_primes_l569_56964

theorem find_primes (p : ℕ) (hp : Nat.Prime p) :
  (∃ a b c k : ℤ, a^2 + b^2 + c^2 = p ∧ a^4 + b^4 + c^4 = k * p) ↔ (p = 2 ∨ p = 3) :=
by
  sorry

end find_primes_l569_56964


namespace parallel_lines_slope_eq_l569_56978

theorem parallel_lines_slope_eq (a : ℝ) : (∀ x y : ℝ, 3 * y - 4 * a = 8 * x) ∧ (∀ x y : ℝ, y - 2 = (a + 4) * x) → a = -4 / 3 :=
by
  sorry

end parallel_lines_slope_eq_l569_56978


namespace gcd_1080_920_is_40_l569_56999

theorem gcd_1080_920_is_40 : Nat.gcd 1080 920 = 40 :=
by
  sorry

end gcd_1080_920_is_40_l569_56999


namespace value_a_squared_plus_b_squared_l569_56928

-- Defining the problem with the given conditions
theorem value_a_squared_plus_b_squared (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 6) : a^2 + b^2 = 21 :=
by
  sorry

end value_a_squared_plus_b_squared_l569_56928


namespace trig_relationship_l569_56926

theorem trig_relationship : 
  let a := Real.sin (145 * Real.pi / 180)
  let b := Real.cos (52 * Real.pi / 180)
  let c := Real.tan (47 * Real.pi / 180)
  a < b ∧ b < c :=
by 
  sorry

end trig_relationship_l569_56926


namespace fixed_point_graph_l569_56988

theorem fixed_point_graph (a : ℝ) (h_pos : 0 < a) (h_neq_one : a ≠ 1) : ∃ x y : ℝ, (x = 2 ∧ y = 2 ∧ y = a^(x-2) + 1) :=
by
  use 2
  use 2
  sorry

end fixed_point_graph_l569_56988


namespace numerator_of_first_fraction_l569_56966

theorem numerator_of_first_fraction (y : ℝ) (h : y > 0) (x : ℝ) 
  (h_eq : (x / y) * y + (3 * y) / 10 = 0.35 * y) : x = 32 := 
by
  sorry

end numerator_of_first_fraction_l569_56966


namespace conditional_probability_l569_56927

-- Given probabilities:
def p_a : ℚ := 5/23
def p_b : ℚ := 7/23
def p_c : ℚ := 1/23
def p_a_and_b : ℚ := 2/23
def p_a_and_c : ℚ := 1/23
def p_b_and_c : ℚ := 1/23
def p_a_and_b_and_c : ℚ := 1/23

-- Theorem statement to prove:
theorem conditional_probability : p_a_and_b_and_c / p_a_and_c = 1 :=
by
  sorry

end conditional_probability_l569_56927


namespace gcd_markers_l569_56992

variable (n1 n2 n3 : ℕ)

-- Let the markers Mary, Luis, and Ali bought be represented by n1, n2, and n3
def MaryMarkers : ℕ := 36
def LuisMarkers : ℕ := 45
def AliMarkers : ℕ := 75

theorem gcd_markers : Nat.gcd (Nat.gcd MaryMarkers LuisMarkers) AliMarkers = 3 := by
  sorry

end gcd_markers_l569_56992


namespace original_sugar_amount_l569_56956

theorem original_sugar_amount (f : ℕ) (s t r : ℕ) (h1 : f = 5) (h2 : r = 10) (h3 : t = 14) (h4 : f * 2 = r):
  s = t / 2 := sorry

end original_sugar_amount_l569_56956


namespace find_the_number_l569_56958

-- Define the number we are trying to find
variable (x : ℝ)

-- Define the main condition from the problem
def main_condition : Prop := 0.7 * x - 40 = 30

-- Formalize the goal to prove
theorem find_the_number (h : main_condition x) : x = 100 :=
by
  -- Placeholder for the proof
  sorry

end find_the_number_l569_56958


namespace locus_of_point_parabola_l569_56967

/-- If the distance from point P to the point F (4, 0) is one unit less than its distance to the line x + 5 = 0, then the equation of the locus of point P is y^2 = 16x. -/
theorem locus_of_point_parabola :
  ∀ P : ℝ × ℝ, dist P (4, 0) + 1 = abs (P.1 + 5) → P.2^2 = 16 * P.1 :=
by
  sorry

end locus_of_point_parabola_l569_56967


namespace art_club_artworks_l569_56974

-- Define the conditions
def students := 25
def artworks_per_student_per_quarter := 3
def quarters_per_year := 4
def years := 3

-- Calculate total artworks
theorem art_club_artworks : 
  students * artworks_per_student_per_quarter * quarters_per_year * years = 900 :=
by
  sorry

end art_club_artworks_l569_56974


namespace max_sum_of_abc_l569_56913

theorem max_sum_of_abc (A B C : ℕ) (h₁ : A ≠ B) (h₂ : B ≠ C) (h₃ : A ≠ C) (h₄ : A * B * C = 2310) : 
  A + B + C ≤ 52 :=
sorry

end max_sum_of_abc_l569_56913


namespace Kaylee_total_boxes_needed_l569_56936

-- Defining the conditions
def lemon_biscuits := 12
def chocolate_biscuits := 5
def oatmeal_biscuits := 4
def still_needed := 12

-- Defining the total boxes sold so far
def total_sold := lemon_biscuits + chocolate_biscuits + oatmeal_biscuits

-- Defining the total number of boxes that need to be sold in total
def total_needed := total_sold + still_needed

-- Lean statement to prove the required total number of boxes
theorem Kaylee_total_boxes_needed : total_needed = 33 :=
by
  sorry

end Kaylee_total_boxes_needed_l569_56936


namespace value_of_a_l569_56952

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → |a * x + 1| ≤ 3) ↔ a = 2 :=
by
  sorry

end value_of_a_l569_56952


namespace last_four_digits_of_5_pow_2011_l569_56911

theorem last_four_digits_of_5_pow_2011 : (5^2011 % 10000) = 8125 := by
  sorry

end last_four_digits_of_5_pow_2011_l569_56911


namespace roots_sum_eq_product_l569_56932

theorem roots_sum_eq_product (m : ℝ) :
  (∀ x : ℝ, 2 * (x - 1) * (x - 3 * m) = x * (m - 4)) →
  (∀ a b : ℝ, 2 * a * b = 2 * (5 * m + 6) / -2 ∧ 2 * a * b = 6 * m / 2) →
  m = -2 / 3 :=
by
  sorry

end roots_sum_eq_product_l569_56932


namespace people_at_table_l569_56948

theorem people_at_table (n : ℕ)
  (h1 : ∃ (d : ℕ), d > 0 ∧ forall i : ℕ, 1 ≤ i ∧ i < n → (i + d) % n ≠ (31 % n))
  (h2 : ((31 - 7) % n) = ((31 - 14) % n)) :
  n = 41 := 
sorry

end people_at_table_l569_56948


namespace examination_duration_in_hours_l569_56912

theorem examination_duration_in_hours 
  (total_questions : ℕ)
  (type_A_questions : ℕ)
  (time_for_A_problems : ℝ) 
  (time_ratio_A_to_B : ℝ)
  (total_time_for_A : ℝ) 
  (total_time : ℝ) :
  total_questions = 200 → 
  type_A_questions = 15 → 
  time_ratio_A_to_B = 2 → 
  total_time_for_A = 25.116279069767444 →
  total_time = (total_time_for_A + 185 * (25.116279069767444 / 15 / 2)) → 
  total_time / 60 = 3 :=
by sorry

end examination_duration_in_hours_l569_56912


namespace jamie_hours_each_time_l569_56914

theorem jamie_hours_each_time (hours_per_week := 2) (weeks := 6) (rate := 10) (total_earned := 360) : 
  ∃ (h : ℕ), h = 3 ∧ (hours_per_week * weeks * rate * h = total_earned) := 
by
  sorry

end jamie_hours_each_time_l569_56914


namespace number_of_pen_refills_l569_56941

-- Conditions
variable (k : ℕ) (x : ℕ) (hk : k > 0) (hx : (4 + k) * x = 6)

-- Question and conclusion as a theorem statement
theorem number_of_pen_refills (hk : k > 0) (hx : (4 + k) * x = 6) : 2 * x = 2 :=
sorry

end number_of_pen_refills_l569_56941


namespace solve_system_l569_56980

theorem solve_system :
    (∃ x y z : ℝ, 5 * x^2 + 3 * y^2 + 3 * x * y + 2 * x * z - y * z - 10 * y + 5 = 0 ∧
                49 * x^2 + 65 * y^2 + 49 * z^2 - 14 * x * y - 98 * x * z + 14 * y * z - 182 * x - 102 * y + 182 * z + 233 =0
                ∧ ((x = 0 ∧ y = 1 ∧ z = -2)
                   ∨ (x = 2/7 ∧ y = 1 ∧ z = -12/7))) :=
by
  sorry

end solve_system_l569_56980


namespace greatest_b_max_b_value_l569_56900

theorem greatest_b (b y : ℤ) (h : b > 0) (hy : y^2 + b*y = -21) : b ≤ 22 :=
sorry

theorem max_b_value : ∃ b : ℤ, (∀ y : ℤ, y^2 + b*y = -21 → b > 0) ∧ (b = 22) :=
sorry

end greatest_b_max_b_value_l569_56900


namespace first_batch_price_is_50_max_number_of_type_a_tools_l569_56922

-- Define the conditions
def first_batch_cost : Nat := 2000
def second_batch_cost : Nat := 2200
def price_increase : Nat := 5
def max_total_cost : Nat := 2500
def type_b_cost : Nat := 40
def total_third_batch : Nat := 50

-- First batch price per tool
theorem first_batch_price_is_50 (x : Nat) (h1 : first_batch_cost * (x + price_increase) = second_batch_cost * x) :
  x = 50 :=
sorry

-- Second batch price per tool & maximum type A tools in third batch
theorem max_number_of_type_a_tools (y : Nat)
  (h2 : 55 * y + type_b_cost * (total_third_batch - y) ≤ max_total_cost) :
  y ≤ 33 :=
sorry

end first_batch_price_is_50_max_number_of_type_a_tools_l569_56922


namespace function_symmetry_implies_even_l569_56996

theorem function_symmetry_implies_even (f : ℝ → ℝ) (h1 : ∃ x, f x ≠ 0)
  (h2 : ∀ x y, f x = y ↔ -f (-x) = -y) : ∀ x, f x = f (-x) :=
by
  sorry

end function_symmetry_implies_even_l569_56996


namespace enrollment_increase_1991_to_1992_l569_56954

theorem enrollment_increase_1991_to_1992 (E E_1992 E_1993 : ℝ)
    (h1 : E_1993 = 1.26 * E)
    (h2 : E_1993 = 1.05 * E_1992) :
    ((E_1992 - E) / E) * 100 = 20 :=
by
  sorry

end enrollment_increase_1991_to_1992_l569_56954


namespace variance_of_data_set_l569_56918

theorem variance_of_data_set :
  let data_set := [2, 3, 4, 5, 6]
  let mean := (2 + 3 + 4 + 5 + 6) / 5
  let variance := (1 / 5 : Real) * ((2 - mean)^2 + (3 - mean)^2 + (4 - mean)^2 + (5 - mean)^2 + (6 - mean)^2)
  variance = 2 :=
by
  sorry

end variance_of_data_set_l569_56918


namespace alpha_beta_square_l569_56960

-- Statement of the problem in Lean 4
theorem alpha_beta_square :
  ∀ (α β : ℝ), (α ≠ β ∧ ∀ x : ℝ, x^2 - 2 * x - 1 = 0 → (x = α ∨ x = β)) → (α - β)^2 = 8 := 
by
  intros α β h
  sorry

end alpha_beta_square_l569_56960


namespace line_through_points_l569_56931

/-- The line passing through points A(1, 1) and B(2, 3) satisfies the equation 2x - y - 1 = 0. -/
theorem line_through_points (x y : ℝ) :
  (∃ k : ℝ, k * (y - 1) = 2 * (x - 1)) → 2 * x - y - 1 = 0 :=
by
  sorry

end line_through_points_l569_56931


namespace sum_interior_angles_polygon_l569_56981

theorem sum_interior_angles_polygon (n : ℕ) (h : 180 * (n - 2) = 1440) :
  180 * ((n + 3) - 2) = 1980 := by
  sorry

end sum_interior_angles_polygon_l569_56981


namespace determine_correct_path_l569_56933

variable (A B C : Type)
variable (truthful : A → Prop)
variable (whimsical : A → Prop)
variable (answers : A → Prop)
variable (path_correct : A → Prop)

-- Conditions
axiom two_truthful_one_whimsical (x y z : A) : (truthful x ∧ truthful y ∧ whimsical z) ∨ 
                                                (truthful x ∧ truthful z ∧ whimsical y) ∨ 
                                                (truthful y ∧ truthful z ∧ whimsical x)

axiom traveler_aware : ∀ x y : A, truthful x → ¬ truthful y
axiom siblings : A → B → C → Prop
axiom ask_sibling : A → B → C → Prop

-- Conditions formalized
axiom ask_about_truthfulness (x y : A) : answers x → (truthful y ↔ ¬truthful y)

theorem determine_correct_path (x y z : A) :
  (truthful x ∧ ¬truthful y ∧ path_correct x) ∨
  (¬truthful x ∧ truthful y ∧ path_correct y) ∨
  (¬truthful x ∧ ¬truthful y ∧ truthful z ∧ path_correct z) :=
sorry

end determine_correct_path_l569_56933


namespace complement_of_A_in_U_l569_56998

noncomputable def U : Set ℤ := {x : ℤ | x^2 ≤ 2*x + 3}
def A : Set ℤ := {0, 1, 2}

theorem complement_of_A_in_U : (U \ A) = {-1, 3} :=
by
  sorry

end complement_of_A_in_U_l569_56998


namespace equivalent_discount_l569_56907

theorem equivalent_discount (original_price : ℝ) (d1 d2 single_discount : ℝ) :
  original_price = 50 →
  d1 = 0.15 →
  d2 = 0.10 →
  single_discount = 0.235 →
  original_price * (1 - d1) * (1 - d2) = original_price * (1 - single_discount) :=
by
  intros
  sorry

end equivalent_discount_l569_56907


namespace range_half_diff_l569_56946

theorem range_half_diff (α β : ℝ) (h1 : -π/2 ≤ α) (h2 : α < β) (h3 : β ≤ π/2) : 
    -π/2 ≤ (α - β) / 2 ∧ (α - β) / 2 < 0 := 
    sorry

end range_half_diff_l569_56946


namespace anes_age_l569_56920

theorem anes_age (w w_d : ℤ) (n : ℤ) 
  (h1 : 1436 ≤ w ∧ w < 1445)
  (h2 : 1606 ≤ w_d ∧ w_d < 1615)
  (h3 : w_d = w + n * 40) : 
  n = 4 :=
sorry

end anes_age_l569_56920


namespace range_of_p_l569_56942

def p (x : ℝ) : ℝ := (x^3 + 3)^2

theorem range_of_p :
  (∀ y, ∃ x ∈ Set.Ici (-1 : ℝ), p x = y) ↔ y ∈ Set.Ici (4 : ℝ) :=
by
  sorry

end range_of_p_l569_56942


namespace one_fourth_of_eight_point_four_l569_56986

theorem one_fourth_of_eight_point_four : (8.4 / 4) = (21 / 10) :=
by
  -- The expected proof would go here
  sorry

end one_fourth_of_eight_point_four_l569_56986


namespace least_five_digit_perfect_square_cube_l569_56909

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l569_56909


namespace product_of_fractions_l569_56934

-- Define the fractions as ratios.
def fraction1 : ℚ := 2 / 5
def fraction2 : ℚ := 7 / 10

-- State the theorem that proves the product of the fractions is equal to the simplified result.
theorem product_of_fractions : fraction1 * fraction2 = 7 / 25 :=
by
  -- Skip the proof.
  sorry

end product_of_fractions_l569_56934


namespace min_value_2a_3b_equality_case_l569_56989

theorem min_value_2a_3b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 / a + 3 / b = 1) : 
  2 * a + 3 * b ≥ 25 :=
sorry

theorem equality_case (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 / a + 3 / b = 1) :
  (a = 5) ∧ (b = 5) → 2 * a + 3 * b = 25 :=
sorry

end min_value_2a_3b_equality_case_l569_56989


namespace find_four_digit_numbers_l569_56930

def isFourDigitNumber (n : ℕ) : Prop := (1000 ≤ n) ∧ (n < 10000)

noncomputable def solveABCD (AB CD : ℕ) : ℕ := 100 * AB + CD

theorem find_four_digit_numbers :
  ∀ (AB CD : ℕ),
    isFourDigitNumber (solveABCD AB CD) →
    solveABCD AB CD = AB * CD + AB ^ 2 →
      solveABCD AB CD = 1296 ∨ solveABCD AB CD = 3468 :=
by
  intros AB CD h1 h2
  sorry

end find_four_digit_numbers_l569_56930


namespace circle_area_l569_56957

theorem circle_area (x y : ℝ) : 
  x^2 + y^2 - 18 * x + 8 * y = -72 → 
  ∃ r : ℝ, r = 5 ∧ π * r ^ 2 = 25 * π := 
by
  sorry

end circle_area_l569_56957


namespace circle1_standard_form_circle2_standard_form_l569_56937

-- Define the first circle equation and its corresponding answer in standard form
theorem circle1_standard_form :
  ∀ x y : ℝ, (x^2 + y^2 + 2*x + 4*y - 4 = 0) ↔ ((x + 1)^2 + (y + 2)^2 = 9) :=
by
  intro x y
  sorry

-- Define the second circle equation and its corresponding answer in standard form
theorem circle2_standard_form :
  ∀ x y : ℝ, (3*x^2 + 3*y^2 + 6*x + 3*y - 15 = 0) ↔ ((x + 1)^2 + (y + 1/2)^2 = 25/4) :=
by
  intro x y
  sorry

end circle1_standard_form_circle2_standard_form_l569_56937


namespace intersection_point_x_value_l569_56951

theorem intersection_point_x_value :
  ∃ x y : ℚ, (y = 3 * x - 22) ∧ (3 * x + y = 100) ∧ (x = 20 + 1 / 3) := by
  sorry

end intersection_point_x_value_l569_56951


namespace intersection_and_complement_find_m_l569_56916

-- Define the sets A, B, C
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}
def C (m : ℝ) : Set ℝ := {x | m+1 ≤ x ∧ x ≤ 3*m}

-- State the first proof problem: intersection A ∩ B and complement of B
theorem intersection_and_complement (x : ℝ) : 
  (x ∈ (A ∩ B) ↔ (2 ≤ x ∧ x ≤ 3)) ∧ 
  (x ∈ (compl B) ↔ (x < 1 ∨ x > 4)) :=
by 
  sorry

-- State the second proof problem: find m satisfying A ∪ C(m) = A
theorem find_m (m : ℝ) (x : ℝ) : 
  (∀ x, (x ∈ A ∪ C m) ↔ (x ∈ A)) ↔ (m = 1) :=
by 
  sorry

end intersection_and_complement_find_m_l569_56916


namespace total_short_trees_l569_56990

def short_trees_initial := 41
def short_trees_planted := 57

theorem total_short_trees : short_trees_initial + short_trees_planted = 98 := by
  sorry

end total_short_trees_l569_56990


namespace range_of_f_l569_56950

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^x else -Real.log x / Real.log 2

theorem range_of_f : Set.Iic 2 = Set.range f :=
  by sorry

end range_of_f_l569_56950


namespace tim_investment_l569_56977

noncomputable def initial_investment_required 
  (A : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  A / ((1 + r / n) ^ (n * t))

theorem tim_investment :
  initial_investment_required 100000 0.10 2 3 = 74622 :=
by
  sorry

end tim_investment_l569_56977


namespace john_outside_doors_count_l569_56940

theorem john_outside_doors_count 
  (bedroom_doors : ℕ := 3) 
  (cost_outside_door : ℕ := 20) 
  (total_cost : ℕ := 70) 
  (cost_bedroom_door := cost_outside_door / 2) 
  (total_bedroom_cost := bedroom_doors * cost_bedroom_door) 
  (outside_doors := (total_cost - total_bedroom_cost) / cost_outside_door) : 
  outside_doors = 2 :=
by
  sorry

end john_outside_doors_count_l569_56940


namespace max_possible_value_of_k_l569_56984

noncomputable def max_knights_saying_less : Nat :=
  let n := 2015
  let k := n - 2
  k

theorem max_possible_value_of_k : max_knights_saying_less = 2013 :=
by
  sorry

end max_possible_value_of_k_l569_56984


namespace discount_percentage_l569_56924

theorem discount_percentage (original_price new_price : ℕ) (h₁ : original_price = 120) (h₂ : new_price = 96) : 
  ((original_price - new_price) * 100 / original_price) = 20 := 
by
  -- sorry is used here to skip the proof
  sorry

end discount_percentage_l569_56924


namespace ArithmeticSequenceSum_l569_56975

theorem ArithmeticSequenceSum (a : ℕ → ℕ) (d : ℕ) 
  (h1 : a 1 + a 2 = 10) 
  (h2 : a 4 = a 3 + 2)
  (h3 : ∀ n : ℕ, a n = a 1 + (n - 1) * d) :
  a 3 + a 4 = 18 :=
by
  sorry

end ArithmeticSequenceSum_l569_56975


namespace find_a4_l569_56971

variable (a_1 d : ℝ)

def a_n (n : ℕ) : ℝ :=
  a_1 + (n - 1) * d

axiom condition1 : (a_n a_1 d 2 + a_n a_1 d 6) / 2 = 5 * Real.sqrt 3
axiom condition2 : (a_n a_1 d 3 + a_n a_1 d 7) / 2 = 7 * Real.sqrt 3

theorem find_a4 : a_n a_1 d 4 = 5 * Real.sqrt 3 :=
by
  -- Proof should go here, but we insert "sorry" to mark it as incomplete.
  sorry

end find_a4_l569_56971


namespace solution_set_of_system_of_inequalities_l569_56935

theorem solution_set_of_system_of_inequalities :
  {x : ℝ | |x| - 1 < 0 ∧ x^2 - 3 * x < 0} = {x : ℝ | 0 < x ∧ x < 1} :=
sorry

end solution_set_of_system_of_inequalities_l569_56935


namespace find_AM_l569_56976

-- Definitions (conditions)
variables {A M B : ℝ}
variable  (collinear : A ≤ M ∧ M ≤ B ∨ B ≤ M ∧ M ≤ A ∨ A ≤ B ∧ B ≤ M)
          (h1 : abs (M - A) = 2 * abs (M - B)) 
          (h2 : abs (A - B) = 6)

-- Proof problem statement
theorem find_AM : (abs (M - A) = 4) ∨ (abs (M - A) = 12) :=
by 
  sorry

end find_AM_l569_56976
