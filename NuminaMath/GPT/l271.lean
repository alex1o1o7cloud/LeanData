import Mathlib

namespace NUMINAMATH_GPT_slower_train_pass_time_l271_27128

noncomputable def relative_speed_km_per_hr (v1 v2 : ℕ) : ℕ :=
v1 + v2

noncomputable def relative_speed_m_per_s (v_km_per_hr : ℕ) : ℝ :=
(v_km_per_hr * 5) / 18

noncomputable def time_to_pass (distance_m : ℕ) (speed_m_per_s : ℝ) : ℝ :=
distance_m / speed_m_per_s

theorem slower_train_pass_time
  (length_train1 length_train2 : ℕ)
  (speed_train1_km_per_hr speed_train2_km_per_hr : ℕ)
  (distance_to_cover : ℕ)
  (h1 : length_train1 = 800)
  (h2 : length_train2 = 600)
  (h3 : speed_train1_km_per_hr = 85)
  (h4 : speed_train2_km_per_hr = 65)
  (h5 : distance_to_cover = length_train2) :
  time_to_pass distance_to_cover (relative_speed_m_per_s (relative_speed_km_per_hr speed_train1_km_per_hr speed_train2_km_per_hr)) = 14.4 := 
sorry

end NUMINAMATH_GPT_slower_train_pass_time_l271_27128


namespace NUMINAMATH_GPT_abs_div_inequality_l271_27179

theorem abs_div_inequality (x : ℝ) : 
  (|-((x+1)/x)| > (x+1)/x) ↔ (-1 < x ∧ x < 0) :=
sorry

end NUMINAMATH_GPT_abs_div_inequality_l271_27179


namespace NUMINAMATH_GPT_range_of_y_l271_27107

theorem range_of_y (m n k y : ℝ)
  (h₁ : 0 ≤ m)
  (h₂ : 0 ≤ n)
  (h₃ : 0 ≤ k)
  (h₄ : m - k + 1 = 1)
  (h₅ : 2 * k + n = 1)
  (h₆ : y = 2 * k^2 - 8 * k + 6)
  : 5 / 2 ≤ y ∧ y ≤ 6 :=
by
  sorry

end NUMINAMATH_GPT_range_of_y_l271_27107


namespace NUMINAMATH_GPT_total_amount_spent_l271_27166
-- Since we need broader imports, we include the whole Mathlib library

-- Definition of the prices of each CD and the quantity purchased
def price_the_life_journey : ℕ := 100
def price_a_day_a_life : ℕ := 50
def price_when_you_rescind : ℕ := 85
def quantity_purchased : ℕ := 3

-- Tactic to calculate the total amount spent
theorem total_amount_spent : (price_the_life_journey * quantity_purchased) + 
                             (price_a_day_a_life * quantity_purchased) + 
                             (price_when_you_rescind * quantity_purchased) 
                             = 705 := by
  sorry

end NUMINAMATH_GPT_total_amount_spent_l271_27166


namespace NUMINAMATH_GPT_probability_of_selecting_one_second_class_product_l271_27154

def total_products : ℕ := 100
def first_class_products : ℕ := 90
def second_class_products : ℕ := 10
def selected_products : ℕ := 3
def exactly_one_second_class_probability : ℚ :=
  (Nat.choose first_class_products 2 * Nat.choose second_class_products 1) / Nat.choose total_products selected_products

theorem probability_of_selecting_one_second_class_product :
  exactly_one_second_class_probability = 0.25 := 
  sorry

end NUMINAMATH_GPT_probability_of_selecting_one_second_class_product_l271_27154


namespace NUMINAMATH_GPT_lulu_final_cash_l271_27148

-- Definitions of the problem conditions
def initial_amount : ℕ := 65
def spent_on_ice_cream : ℕ := 5
def spent_on_tshirt (remaining : ℕ) : ℕ := remaining / 2
def deposit_in_bank (remaining : ℕ) : ℕ := remaining / 5

-- The proof problem statement
theorem lulu_final_cash :
  ∃ final_cash : ℕ,
    final_cash = initial_amount - spent_on_ice_cream - spent_on_tshirt (initial_amount - spent_on_ice_cream) - 
                      deposit_in_bank (spent_on_tshirt (initial_amount - spent_on_ice_cream)) ∧
    final_cash = 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_lulu_final_cash_l271_27148


namespace NUMINAMATH_GPT_blue_pens_count_l271_27122

-- Definitions based on the conditions
def total_pens (B R : ℕ) : Prop := B + R = 82
def more_blue_pens (B R : ℕ) : Prop := B = R + 6

-- The theorem to prove
theorem blue_pens_count (B R : ℕ) (h1 : total_pens B R) (h2 : more_blue_pens B R) : B = 44 :=
by {
  -- This is where the proof steps would normally go.
  sorry
}

end NUMINAMATH_GPT_blue_pens_count_l271_27122


namespace NUMINAMATH_GPT_rhombus_perimeter_l271_27191

-- Define the conditions for the rhombus
variable (d1 d2 : ℝ) (a b s : ℝ)

-- State the condition that the diagonals of a rhombus measure 24 cm and 10 cm
def diagonal_condition := (d1 = 24) ∧ (d2 = 10)

-- State the Pythagorean theorem for the lengths of half-diagonals
def pythagorean_theorem := a^2 + b^2 = s^2

-- State the relationship of diagonals bisecting each other at right angles
def bisect_condition := (a = d1 / 2) ∧ (b = d2 / 2)

-- State the definition of the perimeter for a rhombus
def perimeter (s : ℝ) : ℝ := 4 * s

-- The theorem we want to prove
theorem rhombus_perimeter : diagonal_condition d1 d2 →
                            bisect_condition d1 d2 a b →
                            pythagorean_theorem a b s →
                            perimeter s = 52 :=
by
  intros h1 h2 h3
  -- Proof would go here, but it is omitted
  sorry

end NUMINAMATH_GPT_rhombus_perimeter_l271_27191


namespace NUMINAMATH_GPT_number_of_non_Speedsters_l271_27108

theorem number_of_non_Speedsters (V : ℝ) (h0 : (4 / 15) * V = 12) : (2 / 3) * V = 30 :=
by
  -- The conditions are such that:
  -- V is the total number of vehicles.
  -- (4 / 15) * V = 12 means 4/5 of 1/3 of the total vehicles are convertibles.
  -- We need to prove that 2/3 of the vehicles are not Speedsters.
  sorry

end NUMINAMATH_GPT_number_of_non_Speedsters_l271_27108


namespace NUMINAMATH_GPT_proof_question_1_l271_27165

noncomputable def question_1 (x : ℝ) : ℝ :=
  (Real.sin (2 * x) + 2 * (Real.sin x)^2) / (1 - Real.tan x)

theorem proof_question_1 :
  ∀ x : ℝ, (Real.cos (π / 4 + x) = 3 / 5) →
  (17 * π / 12 < x ∧ x < 7 * π / 4) →
  question_1 x = -9 / 20 :=
by
  intros x h1 h2
  sorry

end NUMINAMATH_GPT_proof_question_1_l271_27165


namespace NUMINAMATH_GPT_man_speed_against_current_l271_27101

theorem man_speed_against_current:
  ∀ (V_current : ℝ) (V_still : ℝ) (current_speed : ℝ),
    V_current = V_still + current_speed →
    V_current = 16 →
    current_speed = 3.2 →
    V_still - current_speed = 9.6 :=
by
  intros V_current V_still current_speed h1 h2 h3
  sorry

end NUMINAMATH_GPT_man_speed_against_current_l271_27101


namespace NUMINAMATH_GPT_graph_of_equation_is_two_lines_l271_27114

theorem graph_of_equation_is_two_lines :
  ∀ (x y : ℝ), (2 * x - y)^2 = 4 * x^2 - y^2 ↔ (y = 0 ∨ y = 2 * x) :=
by
  sorry

end NUMINAMATH_GPT_graph_of_equation_is_two_lines_l271_27114


namespace NUMINAMATH_GPT_value_of_expression_l271_27186

theorem value_of_expression : 1 + 2 + 3 - 4 + 5 + 6 + 7 - 8 + 9 + 10 + 11 - 12 = 30 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l271_27186


namespace NUMINAMATH_GPT_largest_digit_M_l271_27173

-- Define the conditions as Lean types
def digit_sum_divisible_by_3 (M : ℕ) := (4 + 5 + 6 + 7 + M) % 3 = 0
def even_digit (M : ℕ) := M % 2 = 0

-- Define the problem statement in Lean
theorem largest_digit_M (M : ℕ) (h : even_digit M ∧ digit_sum_divisible_by_3 M) : M ≤ 8 ∧ (∀ N : ℕ, even_digit N ∧ digit_sum_divisible_by_3 N → N ≤ M) :=
sorry

end NUMINAMATH_GPT_largest_digit_M_l271_27173


namespace NUMINAMATH_GPT_avg_height_and_variance_correct_l271_27168

noncomputable def avg_height_and_variance
  (total_students : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (boys_avg_height : ℕ)
  (boys_variance : ℕ)
  (girls_avg_height : ℕ)
  (girls_variance : ℕ) : (ℕ × ℕ) := 
  let total_students := 300
  let boys := 180
  let girls := 120
  let boys_avg_height := 170
  let boys_variance := 14
  let girls_avg_height := 160
  let girls_variance := 24
  let avg_height := (boys * boys_avg_height + girls * girls_avg_height) / total_students 
  let variance := (boys * (boys_variance + (boys_avg_height - avg_height) ^ 2) 
                    + girls * (girls_variance + (girls_avg_height - avg_height) ^ 2)) / total_students
  (avg_height, variance)

theorem avg_height_and_variance_correct:
   avg_height_and_variance 300 180 120 170 14 160 24 = (166, 42) := 
  by {
    sorry
  }

end NUMINAMATH_GPT_avg_height_and_variance_correct_l271_27168


namespace NUMINAMATH_GPT_power_equality_l271_27126

theorem power_equality (x : ℝ) (n : ℕ) (h : x^(2 * n) = 3) : x^(4 * n) = 9 := 
by 
  sorry

end NUMINAMATH_GPT_power_equality_l271_27126


namespace NUMINAMATH_GPT_swim_club_member_count_l271_27115

theorem swim_club_member_count :
  let total_members := 60
  let passed_percentage := 0.30
  let passed_members := total_members * passed_percentage
  let not_passed_members := total_members - passed_members
  let preparatory_course_members := 12
  not_passed_members - preparatory_course_members = 30 :=
by
  sorry

end NUMINAMATH_GPT_swim_club_member_count_l271_27115


namespace NUMINAMATH_GPT_least_bulbs_needed_l271_27139

/-- Tulip bulbs come in packs of 15, and daffodil bulbs come in packs of 16.
  Rita wants to buy the same number of tulip and daffodil bulbs. 
  The goal is to prove that the least number of bulbs she needs to buy is 240, i.e.,
  the least common multiple of 15 and 16 is 240. -/
theorem least_bulbs_needed : Nat.lcm 15 16 = 240 := 
by
  sorry

end NUMINAMATH_GPT_least_bulbs_needed_l271_27139


namespace NUMINAMATH_GPT_company_a_percentage_l271_27143

theorem company_a_percentage (total_profits: ℝ) (p_b: ℝ) (profit_b: ℝ) (profit_a: ℝ) :
  p_b = 0.40 →
  profit_b = 60000 →
  profit_a = 90000 →
  total_profits = profit_b / p_b →
  (profit_a / total_profits) * 100 = 60 :=
by
  intros h_pb h_profit_b h_profit_a h_total_profits
  sorry

end NUMINAMATH_GPT_company_a_percentage_l271_27143


namespace NUMINAMATH_GPT_eccentricity_range_l271_27102

def hyperbola (a b x y : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)
def right_branch_hyperbola_P (a b c x y : ℝ) : Prop := hyperbola a b x y ∧ (c = a) ∧ (2 * c = a)

theorem eccentricity_range {a b c : ℝ} (h: hyperbola a b c c) (h1 : 2 * a = 2 * c) (h2 : c = a) :
  1 < (c / a) ∧ (c / a) ≤ (Real.sqrt 10 / 2 : ℝ) := by
  sorry

end NUMINAMATH_GPT_eccentricity_range_l271_27102


namespace NUMINAMATH_GPT_find_real_numbers_l271_27138

theorem find_real_numbers (x : ℝ) :
  (x^3 - x^2 = (x^2 - x)^2) ↔ (x = 0 ∨ x = 1 ∨ x = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_real_numbers_l271_27138


namespace NUMINAMATH_GPT_not_sum_six_odd_squares_l271_27169

-- Definition stating that a number is odd.
def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

-- Given that the square of any odd number is 1 modulo 8.
lemma odd_square_mod_eight (n : ℕ) (h : is_odd n) : (n^2) % 8 = 1 :=
sorry

-- Main theorem stating that 1986 cannot be the sum of six squares of odd numbers.
theorem not_sum_six_odd_squares : ¬ ∃ n1 n2 n3 n4 n5 n6 : ℕ, 
    is_odd n1 ∧ is_odd n2 ∧ is_odd n3 ∧ is_odd n4 ∧ is_odd n5 ∧ is_odd n6 ∧
    n1^2 + n2^2 + n3^2 + n4^2 + n5^2 + n6^2 = 1986 :=
sorry

end NUMINAMATH_GPT_not_sum_six_odd_squares_l271_27169


namespace NUMINAMATH_GPT_shortest_part_is_15_l271_27100

namespace ProofProblem

def rope_length : ℕ := 60
def ratio_part1 : ℕ := 3
def ratio_part2 : ℕ := 4
def ratio_part3 : ℕ := 5

def total_parts := ratio_part1 + ratio_part2 + ratio_part3
def length_per_part := rope_length / total_parts
def shortest_part_length := ratio_part1 * length_per_part

theorem shortest_part_is_15 :
  shortest_part_length = 15 := by
  sorry

end ProofProblem

end NUMINAMATH_GPT_shortest_part_is_15_l271_27100


namespace NUMINAMATH_GPT_units_digit_sum_base8_l271_27181

theorem units_digit_sum_base8 : 
  let n1 := 53 
  let n2 := 64 
  let sum_base8 := n1 + n2 
  (sum_base8 % 8) = 7 := 
by 
  sorry

end NUMINAMATH_GPT_units_digit_sum_base8_l271_27181


namespace NUMINAMATH_GPT_kernels_popped_in_first_bag_l271_27171

theorem kernels_popped_in_first_bag :
  ∀ (x : ℕ), 
    (total_kernels : ℕ := 75 + 50 + 100) →
    (total_popped : ℕ := x + 42 + 82) →
    (average_percentage_popped : ℚ := 82) →
    ((total_popped : ℚ) / total_kernels) * 100 = average_percentage_popped →
    x = 61 :=
by
  sorry

end NUMINAMATH_GPT_kernels_popped_in_first_bag_l271_27171


namespace NUMINAMATH_GPT_count_similar_divisors_l271_27156

def is_integrally_similar_divisible (a b c : ℕ) : Prop :=
  ∃ x y z : ℕ, a * c = b * z ∧
  x ≤ y ∧ y ≤ z ∧
  b = 2023 ∧ a * c = 2023^2

theorem count_similar_divisors (b : ℕ) (hb : b = 2023) :
  ∃ (n : ℕ), n = 7 ∧ 
    (∀ (a c : ℕ), a ≤ b ∧ b ≤ c → is_integrally_similar_divisible a b c) :=
by
  sorry

end NUMINAMATH_GPT_count_similar_divisors_l271_27156


namespace NUMINAMATH_GPT_campers_difference_l271_27112

theorem campers_difference 
       (total : ℕ)
       (campers_two_weeks_ago : ℕ) 
       (campers_last_week : ℕ) 
       (diff: ℕ)
       (h_total : total = 150)
       (h_two_weeks_ago : campers_two_weeks_ago = 40) 
       (h_last_week : campers_last_week = 80) : 
       diff = campers_two_weeks_ago - (total - campers_two_weeks_ago - campers_last_week) :=
by
  sorry

end NUMINAMATH_GPT_campers_difference_l271_27112


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l271_27196

open Real

-- For part (1)
theorem part1_solution_set (x a : ℝ) (h : a = 3) : |2 * x - a| + a ≤ 6 ↔ 0 ≤ x ∧ x ≤ 3 := 
by {
  sorry
}

-- For part (2)
theorem part2_range_of_a (f g : ℝ → ℝ) (hf : ∀ x, f x = |2 * x - a| + a) (hg : ∀ x, g x = |2 * x - 3|) :
  (∀ x, f x + g x ≥ 5) ↔ a ≥ 11 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l271_27196


namespace NUMINAMATH_GPT_radius_of_circle_is_4_l271_27147

noncomputable def circle_radius
  (a : ℝ) 
  (radius : ℝ) 
  (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x + 9 = 0 ∧ (-a, 0) = (5, 0) ∧ radius = 4

theorem radius_of_circle_is_4 
  (a x y : ℝ) 
  (radius : ℝ) 
  (h : circle_radius a radius x y) : 
  radius = 4 :=
by 
  sorry

end NUMINAMATH_GPT_radius_of_circle_is_4_l271_27147


namespace NUMINAMATH_GPT_cricket_run_rate_l271_27161

theorem cricket_run_rate (run_rate_first_10_overs : ℝ) (target : ℝ) (overs_first_phase : ℕ) (overs_remaining : ℕ) :
  run_rate_first_10_overs = 4.6 → target = 282 → overs_first_phase = 10 → overs_remaining = 40 →
  (target - run_rate_first_10_overs * overs_first_phase) / overs_remaining = 5.9 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cricket_run_rate_l271_27161


namespace NUMINAMATH_GPT_EllenBreadMakingTime_l271_27120

-- Definitions based on the given problem
def RisingTimeTypeA : ℕ → ℝ := λ n => n * 4
def BakingTimeTypeA : ℕ → ℝ := λ n => n * 2.5
def RisingTimeTypeB : ℕ → ℝ := λ n => n * 3.5
def BakingTimeTypeB : ℕ → ℝ := λ n => n * 3

def TotalTime (nA nB : ℕ) : ℝ :=
  (RisingTimeTypeA nA + BakingTimeTypeA nA) +
  (RisingTimeTypeB nB + BakingTimeTypeB nB)

theorem EllenBreadMakingTime :
  TotalTime 3 2 = 32.5 := by
  sorry

end NUMINAMATH_GPT_EllenBreadMakingTime_l271_27120


namespace NUMINAMATH_GPT_shaded_square_area_l271_27159

noncomputable def Pythagorean_area (a b c : ℕ) (area_a area_b area_c : ℕ) : Prop :=
  area_a = a^2 ∧ area_b = b^2 ∧ area_c = c^2 ∧ a^2 + b^2 = c^2

theorem shaded_square_area 
  (area1 area2 area3 : ℕ)
  (area_unmarked : ℕ)
  (h1 : area1 = 5)
  (h2 : area2 = 8)
  (h3 : area3 = 32)
  (h_unmarked: area_unmarked = area2 + area3)
  (h_shaded : area1 + area_unmarked = 45) :
  area1 + area_unmarked = 45 :=
by
  exact h_shaded

end NUMINAMATH_GPT_shaded_square_area_l271_27159


namespace NUMINAMATH_GPT_train_speed_l271_27141

theorem train_speed (v : ℝ) : (∃ t : ℝ, 2 * v + t * v = 285 ∧ t = 285 / 38) → v = 30 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l271_27141


namespace NUMINAMATH_GPT_complement_union_l271_27137

def U : Set ℤ := {x | -3 < x ∧ x ≤ 4}
def A : Set ℤ := {-2, -1, 3}
def B : Set ℤ := {1, 2, 3}

def C (U : Set ℤ) (S : Set ℤ) : Set ℤ := {x | x ∈ U ∧ x ∉ S}

theorem complement_union (A B : Set ℤ) (U : Set ℤ) :
  C U (A ∪ B) = {0, 4} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_l271_27137


namespace NUMINAMATH_GPT_whale_ninth_hour_consumption_l271_27152

-- Define the arithmetic sequence conditions
def first_hour_consumption : ℕ := 10
def common_difference : ℕ := 5

-- Define the total consumption over 12 hours
def total_consumption := 12 * (first_hour_consumption + (first_hour_consumption + 11 * common_difference)) / 2

-- Prove the ninth hour (which is the 8th term) consumption
theorem whale_ninth_hour_consumption :
  total_consumption = 450 →
  first_hour_consumption + 8 * common_difference = 50 := 
by
  intros h
  sorry
  

end NUMINAMATH_GPT_whale_ninth_hour_consumption_l271_27152


namespace NUMINAMATH_GPT_symmetric_line_eq_l271_27160

theorem symmetric_line_eq (x y : ℝ) (h : 2 * x - y = 0) : 2 * x + y = 0 :=
sorry

end NUMINAMATH_GPT_symmetric_line_eq_l271_27160


namespace NUMINAMATH_GPT_age_of_b_l271_27106

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 42) : b = 16 :=
by
  sorry

end NUMINAMATH_GPT_age_of_b_l271_27106


namespace NUMINAMATH_GPT_solve_quadratic_eq_l271_27193

theorem solve_quadratic_eq (x : ℝ) :
  (3 * (2 * x + 1) = (2 * x + 1)^2) →
  (x = -1/2 ∨ x = 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l271_27193


namespace NUMINAMATH_GPT_crossing_time_approx_11_16_seconds_l271_27105

noncomputable def length_train_1 : ℝ := 140 -- length of the first train in meters
noncomputable def length_train_2 : ℝ := 170 -- length of the second train in meters
noncomputable def speed_train_1_km_hr : ℝ := 60 -- speed of the first train in km/hr
noncomputable def speed_train_2_km_hr : ℝ := 40 -- speed of the second train in km/hr

noncomputable def speed_conversion_factor : ℝ := 5 / 18 -- conversion factor from km/hr to m/s

-- convert speeds from km/hr to m/s
noncomputable def speed_train_1_m_s : ℝ := speed_train_1_km_hr * speed_conversion_factor
noncomputable def speed_train_2_m_s : ℝ := speed_train_2_km_hr * speed_conversion_factor

-- calculate relative speed in m/s (since they are moving in opposite directions)
noncomputable def relative_speed_m_s : ℝ := speed_train_1_m_s + speed_train_2_m_s

-- total distance to be covered
noncomputable def total_distance : ℝ := length_train_1 + length_train_2

-- calculate the time to cross each other
noncomputable def crossing_time : ℝ := total_distance / relative_speed_m_s

theorem crossing_time_approx_11_16_seconds : abs (crossing_time - 11.16) < 0.01 := by
    sorry

end NUMINAMATH_GPT_crossing_time_approx_11_16_seconds_l271_27105


namespace NUMINAMATH_GPT_lowest_price_for_butter_l271_27175

def cost_single_package : ℝ := 7.0
def cost_8oz_package : ℝ := 4.0
def cost_4oz_package : ℝ := 2.0
def discount : ℝ := 0.5

theorem lowest_price_for_butter : 
  min cost_single_package (cost_8oz_package + 2 * (cost_4oz_package * discount)) = 6.0 :=
by
  sorry

end NUMINAMATH_GPT_lowest_price_for_butter_l271_27175


namespace NUMINAMATH_GPT_number_of_girls_l271_27135

theorem number_of_girls 
  (B G : ℕ) 
  (h1 : B + G = 480) 
  (h2 : 5 * B = 3 * G) :
  G = 300 := 
sorry

end NUMINAMATH_GPT_number_of_girls_l271_27135


namespace NUMINAMATH_GPT_find_a9_a10_l271_27136

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * r

theorem find_a9_a10 (a : ℕ → ℝ) (r : ℝ)
  (h_geom : geometric_sequence a r)
  (h1 : a 1 + a 2 = 1)
  (h3 : a 3 + a 4 = 2) :
  a 9 + a 10 = 16 := 
sorry

end NUMINAMATH_GPT_find_a9_a10_l271_27136


namespace NUMINAMATH_GPT_grid_square_count_l271_27127

theorem grid_square_count :
  let width := 6
  let height := 6
  let num_1x1 := (width - 1) * (height - 1)
  let num_2x2 := (width - 2) * (height - 2)
  let num_3x3 := (width - 3) * (height - 3)
  let num_4x4 := (width - 4) * (height - 4)
  num_1x1 + num_2x2 + num_3x3 + num_4x4 = 54 :=
by
  sorry

end NUMINAMATH_GPT_grid_square_count_l271_27127


namespace NUMINAMATH_GPT_method_is_systematic_sampling_l271_27163

-- Define the conditions
def rows : ℕ := 25
def seats_per_row : ℕ := 20
def filled_auditorium : Prop := True
def seat_numbered_15_sampled : Prop := True
def interval : ℕ := 20

-- Define the concept of systematic sampling
def systematic_sampling (rows seats_per_row interval : ℕ) : Prop :=
  (rows > 0 ∧ seats_per_row > 0 ∧ interval > 0 ∧ (interval = seats_per_row))

-- State the problem in terms of proving that the sampling method is systematic
theorem method_is_systematic_sampling :
  filled_auditorium → seat_numbered_15_sampled → systematic_sampling rows seats_per_row interval :=
by
  intros h1 h2
  -- Assume that the proof goes here
  sorry

end NUMINAMATH_GPT_method_is_systematic_sampling_l271_27163


namespace NUMINAMATH_GPT_odd_and_monotonically_decreasing_l271_27198

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≥ f y

theorem odd_and_monotonically_decreasing :
  is_odd (fun x : ℝ => -x^3) ∧ is_monotonically_decreasing (fun x : ℝ => -x^3) :=
by
  sorry

end NUMINAMATH_GPT_odd_and_monotonically_decreasing_l271_27198


namespace NUMINAMATH_GPT_calculate_expression_l271_27134

theorem calculate_expression : 16^4 * 8^2 / 4^12 = (1 : ℚ) / 4 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l271_27134


namespace NUMINAMATH_GPT_two_circles_tangent_internally_l271_27131

-- Define radii and distance between centers
def R : ℝ := 7
def r : ℝ := 4
def distance_centers : ℝ := 3

-- Statement of the problem
theorem two_circles_tangent_internally :
  distance_centers = R - r → 
  -- Positional relationship: tangent internally
  (distance_centers = abs (R - r)) :=
sorry

end NUMINAMATH_GPT_two_circles_tangent_internally_l271_27131


namespace NUMINAMATH_GPT_number_of_blue_spotted_fish_l271_27117

theorem number_of_blue_spotted_fish : 
  ∀ (fish_total : ℕ) (one_third_blue : ℕ) (half_spotted : ℕ),
    fish_total = 30 →
    one_third_blue = fish_total / 3 →
    half_spotted = one_third_blue / 2 →
    half_spotted = 5 := 
by
  intros fish_total one_third_blue half_spotted ht htb hhs
  sorry

end NUMINAMATH_GPT_number_of_blue_spotted_fish_l271_27117


namespace NUMINAMATH_GPT_tan_theta_value_l271_27194

theorem tan_theta_value (θ k : ℝ) 
  (h1 : Real.sin θ = (k + 1) / (k - 3)) 
  (h2 : Real.cos θ = (k - 1) / (k - 3)) 
  (h3 : (Real.sin θ ≠ 0) ∧ (Real.cos θ ≠ 0)) : 
  Real.tan θ = 3 / 4 := 
sorry

end NUMINAMATH_GPT_tan_theta_value_l271_27194


namespace NUMINAMATH_GPT_percentage_given_to_close_friends_l271_27116

-- Definitions
def total_boxes : ℕ := 20
def pens_per_box : ℕ := 5
def total_pens : ℕ := total_boxes * pens_per_box
def pens_left_after_classmates : ℕ := 45

-- Proposition
theorem percentage_given_to_close_friends (P : ℝ) :
  total_boxes = 20 → pens_per_box = 5 → pens_left_after_classmates = 45 →
  (3 / 4) * (100 - P) = (pens_left_after_classmates : ℝ) →
  P = 40 :=
by
  intros h_total_boxes h_pens_per_box h_pens_left_after h_eq
  sorry

end NUMINAMATH_GPT_percentage_given_to_close_friends_l271_27116


namespace NUMINAMATH_GPT_positive_integer_triplets_l271_27124

theorem positive_integer_triplets (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_lcm : a + b + c = Nat.lcm a (Nat.lcm b c)) :
  (∃ k, k ≥ 1 ∧ a = k ∧ b = 2 * k ∧ c = 3 * k) :=
sorry

end NUMINAMATH_GPT_positive_integer_triplets_l271_27124


namespace NUMINAMATH_GPT_total_sum_of_money_is_71_l271_27121

noncomputable def totalCoins : ℕ := 334
noncomputable def coins20Paise : ℕ := 250
noncomputable def coins25Paise : ℕ := totalCoins - coins20Paise
noncomputable def value20Paise : ℕ := coins20Paise * 20
noncomputable def value25Paise : ℕ := coins25Paise * 25
noncomputable def totalValuePaise : ℕ := value20Paise + value25Paise
noncomputable def totalValueRupees : ℚ := totalValuePaise / 100

theorem total_sum_of_money_is_71 :
  totalValueRupees = 71 := by
  sorry

end NUMINAMATH_GPT_total_sum_of_money_is_71_l271_27121


namespace NUMINAMATH_GPT_conic_curve_eccentricity_l271_27174

theorem conic_curve_eccentricity (m : ℝ) 
    (h1 : ∃ k, k ≠ 0 ∧ 1 * k = m ∧ m * k = 4)
    (h2 : m = -2) : ∃ e : ℝ, e = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_conic_curve_eccentricity_l271_27174


namespace NUMINAMATH_GPT_find_a_l271_27199

variable (a b c : ℚ)

theorem find_a (h1 : a + b + c = 150) (h2 : a - 3 = b + 4) (h3 : b + 4 = 4 * c) : 
  a = 631 / 9 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l271_27199


namespace NUMINAMATH_GPT_simplify_and_evaluate_l271_27132

noncomputable section

def x := Real.sqrt 3 + 1

theorem simplify_and_evaluate :
  (x / (x^2 - 1) / (1 - (1 / (x + 1)))) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l271_27132


namespace NUMINAMATH_GPT_female_participation_fraction_l271_27187

noncomputable def fraction_of_females (males_last_year : ℕ) (females_last_year : ℕ) : ℚ :=
  let males_this_year := (1.10 * males_last_year : ℚ)
  let females_this_year := (1.25 * females_last_year : ℚ)
  females_this_year / (males_this_year + females_this_year)

theorem female_participation_fraction
  (males_last_year : ℕ) (participation_increase : ℚ)
  (males_increase : ℚ) (females_increase : ℚ)
  (h_males_last_year : males_last_year = 30)
  (h_participation_increase : participation_increase = 1.15)
  (h_males_increase : males_increase = 1.10)
  (h_females_increase : females_increase = 1.25)
  (h_females_last_year : females_last_year = 15) :
  fraction_of_females males_last_year females_last_year = 19 / 52 := by
  sorry

end NUMINAMATH_GPT_female_participation_fraction_l271_27187


namespace NUMINAMATH_GPT_area_of_X_part_l271_27130

theorem area_of_X_part :
    (∃ s : ℝ, s^2 = 2520 ∧ 
     (∃ E F G H : ℝ, E = F ∧ F = G ∧ G = H ∧ 
         E = s / 4 ∧ F = s / 4 ∧ G = s / 4 ∧ H = s / 4) ∧ 
     2520 * 11 / 24 = 1155) :=
by
  sorry

end NUMINAMATH_GPT_area_of_X_part_l271_27130


namespace NUMINAMATH_GPT_sqrt_170569_sqrt_175561_l271_27182

theorem sqrt_170569 : Nat.sqrt 170569 = 413 := 
by 
  sorry 

theorem sqrt_175561 : Nat.sqrt 175561 = 419 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_170569_sqrt_175561_l271_27182


namespace NUMINAMATH_GPT_correct_options_l271_27113

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function : Prop := ∀ x : ℝ, f x = f (-x)
def function_definition : Prop := ∀ x : ℝ, (0 < x) → f x = x^2 + x

-- Statements to be proved
def option_A : Prop := f (-1) = 2
def option_B_incorrect : Prop := ¬ (∀ x : ℝ, (f x ≥ f 0) ↔ x ≥ 0) -- Reformulated as not a correct statement
def option_C : Prop := ∀ x : ℝ, x < 0 → f x = x^2 - x
def option_D : Prop := ∀ x : ℝ, (0 < x ∧ x < 2) ↔ f (x - 1) < 2

-- Prove that the correct statements are A, C, and D
theorem correct_options (h_even : is_even_function f) (h_def : function_definition f) :
  option_A f ∧ option_C f ∧ option_D f := by
  sorry

end NUMINAMATH_GPT_correct_options_l271_27113


namespace NUMINAMATH_GPT_xyz_squared_l271_27142

theorem xyz_squared (x y z p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0)
  (hxy : x + y = p) (hyz : y + z = q) (hzx : z + x = r) :
  x^2 + y^2 + z^2 = (p^2 + q^2 + r^2 - p * q - q * r - r * p) / 2 :=
by
  sorry

end NUMINAMATH_GPT_xyz_squared_l271_27142


namespace NUMINAMATH_GPT_min_value_of_reciprocal_sum_l271_27146

open Real

theorem min_value_of_reciprocal_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : x + y = 12) (h4 : x * y = 20) : (1 / x + 1 / y) = 3 / 5 :=
sorry

end NUMINAMATH_GPT_min_value_of_reciprocal_sum_l271_27146


namespace NUMINAMATH_GPT_minimum_equilateral_triangles_l271_27178

theorem minimum_equilateral_triangles (side_small : ℝ) (side_large : ℝ)
  (h_small : side_small = 1) (h_large : side_large = 15) :
  225 = (side_large / side_small)^2 :=
by
  -- Proof is skipped.
  sorry

end NUMINAMATH_GPT_minimum_equilateral_triangles_l271_27178


namespace NUMINAMATH_GPT_axis_center_symmetry_sine_shifted_l271_27172
  noncomputable def axis_of_symmetry (k : ℤ) : ℝ := 3 * Real.pi / 4 + k * Real.pi

  noncomputable def center_of_symmetry (k : ℤ) : ℝ × ℝ := (Real.pi / 4 + k * Real.pi, 0)

  theorem axis_center_symmetry_sine_shifted :
    ∀ (k : ℤ),
    ∃ x y : ℝ,
      (x = axis_of_symmetry k) ∧ (y = 0) ∧ (y, 0) = center_of_symmetry k := 
  sorry
  
end NUMINAMATH_GPT_axis_center_symmetry_sine_shifted_l271_27172


namespace NUMINAMATH_GPT_vector_dot_product_l271_27158

variables (a b : ℝ × ℝ)
variables (h1 : a = (1, 2))
variables (h2 : a - (1 / 5) • b = (-2, 1))

theorem vector_dot_product : (a.1 * b.1 + a.2 * b.2) = 25 :=
by
  sorry

end NUMINAMATH_GPT_vector_dot_product_l271_27158


namespace NUMINAMATH_GPT_factor_expression_l271_27197

theorem factor_expression (b : ℤ) : 52 * b ^ 2 + 208 * b = 52 * b * (b + 4) := 
by {
  sorry
}

end NUMINAMATH_GPT_factor_expression_l271_27197


namespace NUMINAMATH_GPT_complementary_angles_difference_l271_27140

def complementary_angles (θ1 θ2 : ℝ) : Prop :=
  θ1 + θ2 = 90

theorem complementary_angles_difference:
  ∀ (θ1 θ2 : ℝ), 
  (θ1 / θ2 = 4 / 5) → 
  complementary_angles θ1 θ2 → 
  abs (θ2 - θ1) = 10 :=
by
  sorry

end NUMINAMATH_GPT_complementary_angles_difference_l271_27140


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l271_27129

theorem boat_speed_in_still_water (b s : ℕ) (h1 : b + s = 21) (h2 : b - s = 9) : b = 15 := by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l271_27129


namespace NUMINAMATH_GPT_investment_difference_l271_27190

theorem investment_difference (x y z : ℕ) 
  (h1 : x + (x + y) + (x + 2 * y) = 9000)
  (h2 : (z / 9000) = (800 / 1800)) 
  (h3 : z = x + 2 * y) :
  y = 1000 := 
by
  -- omitted proof steps
  sorry

end NUMINAMATH_GPT_investment_difference_l271_27190


namespace NUMINAMATH_GPT_ratio_A_to_B_l271_27167

noncomputable def A_annual_income : ℝ := 436800.0000000001
noncomputable def B_increase_rate : ℝ := 0.12
noncomputable def C_monthly_income : ℝ := 13000

noncomputable def A_monthly_income : ℝ := A_annual_income / 12
noncomputable def B_monthly_income : ℝ := C_monthly_income + (B_increase_rate * C_monthly_income)

theorem ratio_A_to_B :
  ((A_monthly_income / 80) : ℝ) = 455 ∧
  ((B_monthly_income / 80) : ℝ) = 182 :=
by
  sorry

end NUMINAMATH_GPT_ratio_A_to_B_l271_27167


namespace NUMINAMATH_GPT_total_wages_l271_27164

theorem total_wages (A_days B_days : ℝ) (A_wages : ℝ) (W : ℝ) 
  (h1 : A_days = 10)
  (h2 : B_days = 15)
  (h3 : A_wages = 2100) :
  W = 3500 :=
by sorry

end NUMINAMATH_GPT_total_wages_l271_27164


namespace NUMINAMATH_GPT_sabina_loan_l271_27155

-- Define the conditions
def tuition_per_year : ℕ := 30000
def living_expenses_per_year : ℕ := 12000
def duration : ℕ := 4
def sabina_savings : ℕ := 10000
def grant_first_two_years_percent : ℕ := 40
def grant_last_two_years_percent : ℕ := 30
def scholarship_percent : ℕ := 20

-- Calculate total tuition for 4 years
def total_tuition : ℕ := tuition_per_year * duration

-- Calculate total living expenses for 4 years
def total_living_expenses : ℕ := living_expenses_per_year * duration

-- Calculate total cost
def total_cost : ℕ := total_tuition + total_living_expenses

-- Calculate grant coverage
def grant_first_two_years : ℕ := (grant_first_two_years_percent * tuition_per_year / 100) * 2
def grant_last_two_years : ℕ := (grant_last_two_years_percent * tuition_per_year / 100) * 2
def total_grant_coverage : ℕ := grant_first_two_years + grant_last_two_years

-- Calculate scholarship savings
def annual_scholarship_savings : ℕ := living_expenses_per_year * scholarship_percent / 100
def total_scholarship_savings : ℕ := annual_scholarship_savings * (duration - 1)

-- Calculate total reductions
def total_reductions : ℕ := total_grant_coverage + total_scholarship_savings + sabina_savings

-- Calculate the total loan needed
def total_loan_needed : ℕ := total_cost - total_reductions

theorem sabina_loan : total_loan_needed = 108800 := by
  sorry

end NUMINAMATH_GPT_sabina_loan_l271_27155


namespace NUMINAMATH_GPT_Norm_photo_count_l271_27103

variables (L M N : ℕ)

-- Conditions from the problem
def cond1 : Prop := L = N - 60
def cond2 : Prop := N = 2 * L + 10

-- Given the conditions, prove N = 110
theorem Norm_photo_count (h1 : cond1 L N) (h2 : cond2 L N) : N = 110 :=
by
  sorry

end NUMINAMATH_GPT_Norm_photo_count_l271_27103


namespace NUMINAMATH_GPT_sum_of_a_and_b_is_two_l271_27170

variable (a b : ℝ)
variable (h_a_nonzero : a ≠ 0)
variable (h_fn_passes_through_point : (a * 1^2 + b * 1 - 1) = 1)

theorem sum_of_a_and_b_is_two : a + b = 2 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_a_and_b_is_two_l271_27170


namespace NUMINAMATH_GPT_rectangle_area_l271_27195

theorem rectangle_area (L B : ℕ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 226) : L * B = 3060 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l271_27195


namespace NUMINAMATH_GPT_third_candidate_more_votes_than_john_l271_27119

-- Define the given conditions
def total_votes : ℕ := 1150
def john_votes : ℕ := 150
def remaining_votes : ℕ := total_votes - john_votes
def james_votes : ℕ := (7 * remaining_votes) / 10
def john_and_james_votes : ℕ := john_votes + james_votes
def third_candidate_votes : ℕ := total_votes - john_and_james_votes

-- Stating the problem to prove
theorem third_candidate_more_votes_than_john : third_candidate_votes - john_votes = 150 := 
by
  sorry

end NUMINAMATH_GPT_third_candidate_more_votes_than_john_l271_27119


namespace NUMINAMATH_GPT_min_distance_sum_l271_27189

open Real EuclideanGeometry

-- Define the parabola y^2 = 4x
noncomputable def parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1 

-- Define the fixed point M
def M : ℝ × ℝ := (2, 3)

-- Define the line l: x = -1
def line_l (P : ℝ × ℝ) : ℝ := abs (P.1 + 1)

-- Define the distance from point P to point M
def distance_to_M (P : ℝ × ℝ) : ℝ := dist P M

-- Define the distance from point P to line l
def distance_to_line (P : ℝ × ℝ) := line_l P 

-- Define the sum of distances
def sum_of_distances (P : ℝ × ℝ) : ℝ := distance_to_M P + distance_to_line P

-- Prove the minimum value of the sum of distances
theorem min_distance_sum : ∃ P, parabola P ∧ sum_of_distances P = sqrt 10 := sorry

end NUMINAMATH_GPT_min_distance_sum_l271_27189


namespace NUMINAMATH_GPT_problem_statement_l271_27157

theorem problem_statement (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (D : ℕ) (M : ℕ) (h_gcd : D = Nat.gcd (Nat.gcd a b) c) (h_lcm : M = Nat.lcm (Nat.lcm a b) c) :
  ((D * M = a * b * c) ∧ ((Nat.gcd a b = 1) ∧ (Nat.gcd b c = 1) ∧ (Nat.gcd a c = 1) → (D * M = a * b * c))) :=
by sorry

end NUMINAMATH_GPT_problem_statement_l271_27157


namespace NUMINAMATH_GPT_min_value_of_expr_l271_27177

theorem min_value_of_expr (a : ℝ) (ha : a > 1) : a + a^2 / (a - 1) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_expr_l271_27177


namespace NUMINAMATH_GPT_general_term_formula_l271_27162

noncomputable def Sn (n : ℕ) (a : ℝ) : ℝ := 3^n + a
noncomputable def an (n : ℕ) : ℝ := 2 * 3^(n-1)

theorem general_term_formula {a : ℝ} (n : ℕ) (h : Sn n a = 3^n + a) :
  Sn n a - Sn (n-1) a = an n :=
sorry

end NUMINAMATH_GPT_general_term_formula_l271_27162


namespace NUMINAMATH_GPT_number_of_women_is_24_l271_27150

-- Define the variables and conditions
variables (x : ℕ) (men_initial : ℕ) (women_initial : ℕ) (men_current : ℕ) (women_current : ℕ)

-- representing the initial ratio and the changes
def initial_conditions : Prop :=
  men_initial = 4 * x ∧ women_initial = 5 * x ∧
  men_current = men_initial + 2 ∧ women_current = 2 * (women_initial - 3)

-- representing the current number of men
def current_men_condition : Prop := men_current = 14

-- The proof we need to generate
theorem number_of_women_is_24 (x : ℕ) (men_initial women_initial men_current women_current : ℕ)
  (h1 : initial_conditions x men_initial women_initial men_current women_current)
  (h2 : current_men_condition men_current) : women_current = 24 :=
by
  -- proof steps here
  sorry

end NUMINAMATH_GPT_number_of_women_is_24_l271_27150


namespace NUMINAMATH_GPT_simplify_eval_l271_27188

variable (x : ℝ)
def expr := 8 * x^2 - (x - 2) * (3 * x + 1) - 2 * (x + 1) * (x - 1)

theorem simplify_eval (h : x = -2) : expr x = 6 := by
  sorry

end NUMINAMATH_GPT_simplify_eval_l271_27188


namespace NUMINAMATH_GPT_weight_in_kilograms_l271_27151

-- Definitions based on conditions
def weight_of_one_bag : ℕ := 250
def number_of_bags : ℕ := 8

-- Converting grams to kilograms (1000 grams = 1 kilogram)
def grams_to_kilograms (grams : ℕ) : ℕ := grams / 1000

-- Total weight in grams
def total_weight_in_grams : ℕ := weight_of_one_bag * number_of_bags

-- Proof that the total weight in kilograms is 2
theorem weight_in_kilograms : grams_to_kilograms total_weight_in_grams = 2 :=
by
  sorry

end NUMINAMATH_GPT_weight_in_kilograms_l271_27151


namespace NUMINAMATH_GPT_ordinary_eq_from_param_eq_l271_27184

theorem ordinary_eq_from_param_eq (α : ℝ) :
  (∃ (x y : ℝ), x = 3 * Real.cos α + 1 ∧ y = - Real.cos α → x + 3 * y - 1 = 0 ∧ (-2 ≤ x ∧ x ≤ 4)) := 
sorry

end NUMINAMATH_GPT_ordinary_eq_from_param_eq_l271_27184


namespace NUMINAMATH_GPT_simplify_fraction_product_l271_27180

theorem simplify_fraction_product :
  4 * (18 / 5) * (35 / -63) * (8 / 14) = - (32 / 7) :=
by sorry

end NUMINAMATH_GPT_simplify_fraction_product_l271_27180


namespace NUMINAMATH_GPT_history_books_count_l271_27183

theorem history_books_count :
  ∃ (total_books reading_books math_books science_books history_books : ℕ),
    total_books = 10 ∧
    reading_books = (2 * total_books) / 5 ∧
    math_books = (3 * total_books) / 10 ∧
    science_books = math_books - 1 ∧
    history_books = total_books - (reading_books + math_books + science_books) ∧
    history_books = 1 :=
by
  sorry

end NUMINAMATH_GPT_history_books_count_l271_27183


namespace NUMINAMATH_GPT_pie_eating_contest_l271_27153

def pies_eaten (Adam Bill Sierra Taylor: ℕ) : ℕ :=
  Adam + Bill + Sierra + Taylor

theorem pie_eating_contest (Bill : ℕ) 
  (Adam_eq_Bill_plus_3 : ∀ B: ℕ, Adam = B + 3)
  (Sierra_eq_2times_Bill : ∀ B: ℕ, Sierra = 2 * B)
  (Sierra_eq_12 : Sierra = 12)
  (Taylor_eq_avg : ∀ A B S: ℕ, Taylor = (A + B + S) / 3)
  : pies_eaten Adam Bill Sierra Taylor = 36 := sorry

end NUMINAMATH_GPT_pie_eating_contest_l271_27153


namespace NUMINAMATH_GPT_chess_tournament_l271_27123

theorem chess_tournament (n k : ℕ) (S : ℕ) (m : ℕ) 
  (h1 : S ≤ k * n) 
  (h2 : S ≥ m * n) 
  : m ≤ k := 
by 
  sorry

end NUMINAMATH_GPT_chess_tournament_l271_27123


namespace NUMINAMATH_GPT_trig_expr_correct_l271_27149

noncomputable def trig_expr : ℝ := Real.sin (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) - 
                                   Real.cos (160 * Real.pi / 180) * Real.sin (170 * Real.pi / 180)

theorem trig_expr_correct : trig_expr = 1 / 2 := 
  sorry

end NUMINAMATH_GPT_trig_expr_correct_l271_27149


namespace NUMINAMATH_GPT_integer_solutions_l271_27109

theorem integer_solutions (x : ℝ) (n : ℤ)
  (h1 : ⌊x⌋ = n) :
  3 * x - 2 * n + 4 = 0 ↔
  x = -4 ∨ x = (-14:ℚ)/3 ∨ x = (-16:ℚ)/3 :=
by sorry

end NUMINAMATH_GPT_integer_solutions_l271_27109


namespace NUMINAMATH_GPT_translation_4_units_upwards_l271_27185

theorem translation_4_units_upwards (M N : ℝ × ℝ) (hx : M.1 = N.1) (hy_diff : N.2 - M.2 = 4) :
  N = (M.1, M.2 + 4) :=
by
  sorry

end NUMINAMATH_GPT_translation_4_units_upwards_l271_27185


namespace NUMINAMATH_GPT_number_of_integer_values_l271_27111

theorem number_of_integer_values (x : ℤ) (h : ⌊Real.sqrt x⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end NUMINAMATH_GPT_number_of_integer_values_l271_27111


namespace NUMINAMATH_GPT_rockham_soccer_league_l271_27145

theorem rockham_soccer_league (cost_socks : ℕ) (cost_tshirt : ℕ) (custom_fee : ℕ) (total_cost : ℕ) :
  cost_socks = 6 →
  cost_tshirt = cost_socks + 7 →
  custom_fee = 200 →
  total_cost = 2892 →
  ∃ members : ℕ, total_cost - custom_fee = members * (2 * (cost_socks + cost_tshirt)) ∧ members = 70 :=
by
  intros
  sorry

end NUMINAMATH_GPT_rockham_soccer_league_l271_27145


namespace NUMINAMATH_GPT_identity_n1_n2_product_l271_27110

theorem identity_n1_n2_product :
  (∃ (N1 N2 : ℤ),
    (∀ x : ℚ, (35 * x - 29) / (x^2 - 3 * x + 2) = N1 / (x - 1) + N2 / (x - 2)) ∧
    N1 * N2 = -246) :=
sorry

end NUMINAMATH_GPT_identity_n1_n2_product_l271_27110


namespace NUMINAMATH_GPT_bubble_gum_cost_l271_27133

theorem bubble_gum_cost (n_pieces : ℕ) (total_cost : ℕ) (cost_per_piece : ℕ) 
  (h1 : n_pieces = 136) (h2 : total_cost = 2448) : cost_per_piece = 18 :=
by
  sorry

end NUMINAMATH_GPT_bubble_gum_cost_l271_27133


namespace NUMINAMATH_GPT_euler_quadrilateral_theorem_l271_27118

theorem euler_quadrilateral_theorem (A1 A2 A3 A4 P Q : ℝ) 
  (midpoint_P : P = (A1 + A3) / 2)
  (midpoint_Q : Q = (A2 + A4) / 2) 
  (length_A1A2 length_A2A3 length_A3A4 length_A4A1 length_A1A3 length_A2A4 length_PQ : ℝ)
  (h1 : length_A1A2 = A1A2) (h2 : length_A2A3 = A2A3)
  (h3 : length_A3A4 = A3A4) (h4 : length_A4A1 = A4A1)
  (h5 : length_A1A3 = A1A3) (h6 : length_A2A4 = A2A4)
  (h7 : length_PQ = PQ) :
  length_A1A2^2 + length_A2A3^2 + length_A3A4^2 + length_A4A1^2 = 
  length_A1A3^2 + length_A2A4^2 + 4 * length_PQ^2 := sorry

end NUMINAMATH_GPT_euler_quadrilateral_theorem_l271_27118


namespace NUMINAMATH_GPT_infinite_series_converges_l271_27125

theorem infinite_series_converges :
  (∑' n : ℕ, if n > 0 then (3 * n - 2) / (n * (n + 1) * (n + 3)) else 0) = 7 / 6 :=
by
  sorry

end NUMINAMATH_GPT_infinite_series_converges_l271_27125


namespace NUMINAMATH_GPT_algebraic_expression_value_l271_27176

theorem algebraic_expression_value (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2 * x^2 - 7 = -6 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l271_27176


namespace NUMINAMATH_GPT_find_quotient_l271_27104

theorem find_quotient
  (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ)
  (h1 : dividend = 131) (h2 : divisor = 14) (h3 : remainder = 5)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_quotient_l271_27104


namespace NUMINAMATH_GPT_factorize_polynomial_l271_27192

def p (a b : ℝ) : ℝ := a^2 - b^2 + 2 * a + 1

theorem factorize_polynomial (a b : ℝ) : 
  p a b = (a + 1 + b) * (a + 1 - b) :=
by
  sorry

end NUMINAMATH_GPT_factorize_polynomial_l271_27192


namespace NUMINAMATH_GPT_smallest_x_with_18_factors_and_factors_18_24_l271_27144

theorem smallest_x_with_18_factors_and_factors_18_24 :
  ∃ (x : ℕ), (∃ (a b : ℕ), x = 2^a * 3^b ∧ 18 ∣ x ∧ 24 ∣ x ∧ (a + 1) * (b + 1) = 18) ∧
    (∀ y, (∃ (c d : ℕ), y = 2^c * 3^d ∧ 18 ∣ y ∧ 24 ∣ y ∧ (c + 1) * (d + 1) = 18) → x ≤ y) :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_with_18_factors_and_factors_18_24_l271_27144
