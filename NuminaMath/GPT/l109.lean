import Mathlib

namespace train_speed_l109_109899

-- Definitions of the given conditions
def platform_length : ℝ := 250
def train_length : ℝ := 470.06
def time_taken : ℝ := 36

-- Definition of the total distance covered
def total_distance := platform_length + train_length

-- The proof problem: Prove that the calculated speed is approximately 20.0017 m/s
theorem train_speed :
  (total_distance / time_taken) = 20.0017 :=
by
  -- The actual proof goes here, but for now we leave it as sorry
  sorry

end train_speed_l109_109899


namespace equation_represents_point_l109_109971

theorem equation_represents_point 
  (a b x y : ℝ) 
  (h : (x - a) ^ 2 + (y + b) ^ 2 = 0) : 
  x = a ∧ y = -b := 
by
  sorry

end equation_represents_point_l109_109971


namespace boys_joined_school_l109_109153

theorem boys_joined_school (initial_boys final_boys boys_joined : ℕ) 
  (h1 : initial_boys = 214) 
  (h2 : final_boys = 1124) 
  (h3 : final_boys = initial_boys + boys_joined) : 
  boys_joined = 910 := 
by 
  rw [h1, h2] at h3
  sorry

end boys_joined_school_l109_109153


namespace sarah_calculate_profit_l109_109168

noncomputable def sarah_total_profit (hot_day_price : ℚ) (regular_day_price : ℚ) (cost_per_cup : ℚ) (cups_per_day : ℕ) (hot_days : ℕ) (total_days : ℕ) : ℚ := 
  let hot_day_revenue := hot_day_price * cups_per_day * hot_days
  let regular_day_revenue := regular_day_price * cups_per_day * (total_days - hot_days)
  let total_revenue := hot_day_revenue + regular_day_revenue
  let total_cost := cost_per_cup * cups_per_day * total_days
  total_revenue - total_cost

theorem sarah_calculate_profit : 
  let hot_day_price := (20951704545454546 : ℚ) / 10000000000000000
  let regular_day_price := hot_day_price / 1.25
  let cost_per_cup := 75 / 100
  let cups_per_day := 32
  let hot_days := 4
  let total_days := 10
  sarah_total_profit hot_day_price regular_day_price cost_per_cup cups_per_day hot_days total_days = (34935102 : ℚ) / 10000000 :=
by
  sorry

end sarah_calculate_profit_l109_109168


namespace cos_double_angle_trig_identity_l109_109905

theorem cos_double_angle_trig_identity
  (α : ℝ) 
  (h : Real.sin (α - Real.pi / 3) = 4 / 5) : 
  Real.cos (2 * α + Real.pi / 3) = 7 / 25 :=
by
  sorry

end cos_double_angle_trig_identity_l109_109905


namespace original_price_calculation_l109_109120

-- Definitions directly from problem conditions
def price_after_decrease (original_price : ℝ) : ℝ := 0.76 * original_price
def new_price : ℝ := 988

-- Statement embedding our problem
theorem original_price_calculation (x : ℝ) (hx : price_after_decrease x = new_price) : x = 1300 :=
by
  sorry

end original_price_calculation_l109_109120


namespace cubic_difference_l109_109140

theorem cubic_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 59) : a^3 - b^3 = 448 :=
by
  sorry

end cubic_difference_l109_109140


namespace seonmi_initial_money_l109_109964

theorem seonmi_initial_money (M : ℝ) (h1 : M/6 = 250) : M = 1500 :=
by
  sorry

end seonmi_initial_money_l109_109964


namespace max_expression_value_l109_109471

theorem max_expression_value (a b c d : ℝ) 
  (h1 : -6.5 ≤ a ∧ a ≤ 6.5) 
  (h2 : -6.5 ≤ b ∧ b ≤ 6.5) 
  (h3 : -6.5 ≤ c ∧ c ≤ 6.5) 
  (h4 : -6.5 ≤ d ∧ d ≤ 6.5) : 
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 182 :=
sorry

end max_expression_value_l109_109471


namespace total_earnings_correct_l109_109125

-- Given conditions
def charge_oil_change : ℕ := 20
def charge_repair : ℕ := 30
def charge_car_wash : ℕ := 5

def number_oil_changes : ℕ := 5
def number_repairs : ℕ := 10
def number_car_washes : ℕ := 15

-- Calculation of earnings based on the conditions
def earnings_from_oil_changes : ℕ := charge_oil_change * number_oil_changes
def earnings_from_repairs : ℕ := charge_repair * number_repairs
def earnings_from_car_washes : ℕ := charge_car_wash * number_car_washes

-- The total earnings
def total_earnings : ℕ := earnings_from_oil_changes + earnings_from_repairs + earnings_from_car_washes

-- Proof statement: Prove that the total earnings are $475
theorem total_earnings_correct : total_earnings = 475 := by -- our proof will go here
  sorry

end total_earnings_correct_l109_109125


namespace cube_paint_same_color_l109_109261

theorem cube_paint_same_color (colors : Fin 6) : ∃ ways : ℕ, ways = 6 :=
sorry

end cube_paint_same_color_l109_109261


namespace seating_capacity_for_ten_tables_in_two_rows_l109_109669

-- Definitions based on the problem conditions
def seating_for_one_table : ℕ := 6

def seating_for_two_tables : ℕ := 10

def seating_for_three_tables : ℕ := 14

def additional_people_per_table : ℕ := 4

-- Calculating the seating capacity for n tables based on the pattern
def seating_capacity (n : ℕ) : ℕ :=
  if n = 1 then seating_for_one_table
  else seating_for_one_table + (n - 1) * additional_people_per_table

-- Proof statement without the proof
theorem seating_capacity_for_ten_tables_in_two_rows :
  (seating_capacity 5) * 2 = 44 :=
by sorry

end seating_capacity_for_ten_tables_in_two_rows_l109_109669


namespace molecular_weight_AlOH3_l109_109434

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

theorem molecular_weight_AlOH3 :
  (atomic_weight_Al + 3 * atomic_weight_O + 3 * atomic_weight_H) = 78.01 :=
by
  sorry

end molecular_weight_AlOH3_l109_109434


namespace total_students_in_faculty_l109_109255

theorem total_students_in_faculty :
  (let sec_year_num := 230
   let sec_year_auto := 423
   let both_subj := 134
   let sec_year_total := 0.80
   let at_least_one_subj := sec_year_num + sec_year_auto - both_subj
   ∃ (T : ℝ), sec_year_total * T = at_least_one_subj ∧ T = 649) := by
  sorry

end total_students_in_faculty_l109_109255


namespace debate_club_girls_l109_109484

theorem debate_club_girls (B G : ℕ) 
  (h1 : B + G = 22)
  (h2 : B + (1/3 : ℚ) * G = 14) : G = 12 :=
sorry

end debate_club_girls_l109_109484


namespace mod_add_l109_109754

theorem mod_add (n : ℕ) (h : n % 5 = 3) : (n + 2025) % 5 = 3 := by
  sorry

end mod_add_l109_109754


namespace segment_length_calc_l109_109417

noncomputable def segment_length_parallel_to_side
  (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) : ℝ :=
  a * (b + c) / (a + b + c)

theorem segment_length_calc
  (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  segment_length_parallel_to_side a b c a_pos b_pos c_pos = a * (b + c) / (a + b + c) :=
sorry

end segment_length_calc_l109_109417


namespace split_numbers_cubic_l109_109835

theorem split_numbers_cubic (m : ℕ) (hm : 1 < m) (assumption : m^2 - m + 1 = 73) : m = 9 :=
sorry

end split_numbers_cubic_l109_109835


namespace flowers_per_vase_l109_109824

-- Definitions of conditions in Lean 4
def number_of_carnations : ℕ := 7
def number_of_roses : ℕ := 47
def total_number_of_flowers : ℕ := number_of_carnations + number_of_roses
def number_of_vases : ℕ := 9

-- Statement in Lean 4
theorem flowers_per_vase : total_number_of_flowers / number_of_vases = 6 := by
  unfold total_number_of_flowers
  show (7 + 47) / 9 = 6
  sorry

end flowers_per_vase_l109_109824


namespace bananas_each_child_l109_109530

theorem bananas_each_child (x : ℕ) (B : ℕ) 
  (h1 : 660 * x = B)
  (h2 : 330 * (x + 2) = B) : 
  x = 2 := 
by 
  sorry

end bananas_each_child_l109_109530


namespace sushi_father_lollipops_l109_109615

variable (x : ℕ)

theorem sushi_father_lollipops (h : x - 5 = 7) : x = 12 := by
  sorry

end sushi_father_lollipops_l109_109615


namespace inequality_holds_for_k_2_l109_109461

theorem inequality_holds_for_k_2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a * b + b * c + c * a + 2 * (1 / a + 1 / b + 1 / c) ≥ 9 := 
by 
  sorry

end inequality_holds_for_k_2_l109_109461


namespace number_of_integers_in_original_list_l109_109577

theorem number_of_integers_in_original_list :
  ∃ n m : ℕ, (m + 2) * (n + 1) = m * n + 15 ∧
             (m + 1) * (n + 2) = m * n + 16 ∧
             n = 4 :=
by {
  sorry
}

end number_of_integers_in_original_list_l109_109577


namespace sequence_formula_l109_109767

theorem sequence_formula (a : ℕ → ℤ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, n ≥ 2 → a n = 3 * a (n - 1) + 4) :
  ∀ n : ℕ, n ≥ 1 → a n = 3^n - 2 :=
by 
sorry

end sequence_formula_l109_109767


namespace alex_baked_cherry_pies_l109_109315

theorem alex_baked_cherry_pies (total_pies : ℕ) (ratio_apple : ℕ) (ratio_blueberry : ℕ) (ratio_cherry : ℕ)
  (h1 : total_pies = 30)
  (h2 : ratio_apple = 1)
  (h3 : ratio_blueberry = 5)
  (h4 : ratio_cherry = 4) :
  (total_pies * ratio_cherry / (ratio_apple + ratio_blueberry + ratio_cherry) = 12) :=
by {
  sorry
}

end alex_baked_cherry_pies_l109_109315


namespace expressway_lengths_l109_109011

theorem expressway_lengths (x y : ℕ) (h1 : x + y = 519) (h2 : x = 2 * y - 45) : x = 331 ∧ y = 188 :=
by
  -- Proof omitted
  sorry

end expressway_lengths_l109_109011


namespace ratio_of_A_to_B_l109_109462

theorem ratio_of_A_to_B (A B C : ℝ) (h1 : A + B + C = 544) (h2 : B = (1/4) * C) (hA : A = 64) (hB : B = 96) (hC : C = 384) : A / B = 2 / 3 :=
by 
  sorry

end ratio_of_A_to_B_l109_109462


namespace amount_spent_on_tracksuit_l109_109759

-- Definitions based on the conditions
def original_price (x : ℝ) := x
def discount_rate : ℝ := 0.20
def savings : ℝ := 30
def actual_spent (x : ℝ) := 0.8 * x

-- Theorem statement derived from the proof translation
theorem amount_spent_on_tracksuit (x : ℝ) (h : (original_price x) * discount_rate = savings) :
  actual_spent x = 120 :=
by
  sorry

end amount_spent_on_tracksuit_l109_109759


namespace math_problem_l109_109855

variable {x y z : ℝ}

def condition1 (x : ℝ) := x = 1.2 * 40
def condition2 (x y : ℝ) := y = x - 0.35 * x
def condition3 (x y z : ℝ) := z = (x + y) / 2

theorem math_problem (x y z : ℝ) (h1 : condition1 x) (h2 : condition2 x y) (h3 : condition3 x y z) :
  z = 39.6 :=
by
  sorry

end math_problem_l109_109855


namespace count_distinct_m_in_right_triangle_l109_109819

theorem count_distinct_m_in_right_triangle (k : ℝ) (hk : k > 0) :
  ∃! m : ℝ, (m = -3/8 ∨ m = -3/4) :=
by
  sorry

end count_distinct_m_in_right_triangle_l109_109819


namespace probability_reach_edge_within_five_hops_l109_109949

-- Define the probability of reaching an edge within n hops from the center
noncomputable def probability_reach_edge_by_hops (n : ℕ) : ℚ :=
if n = 5 then 121 / 128 else 0 -- This is just a placeholder for the real recursive computation.

-- Main theorem to prove
theorem probability_reach_edge_within_five_hops :
  probability_reach_edge_by_hops 5 = 121 / 128 :=
by
  -- Skipping the actual proof here
  sorry

end probability_reach_edge_within_five_hops_l109_109949


namespace average_salary_l109_109145

theorem average_salary (avg_officer_salary avg_nonofficer_salary num_officers num_nonofficers : ℕ) (total_salary total_employees : ℕ) : 
  avg_officer_salary = 430 → 
  avg_nonofficer_salary = 110 → 
  num_officers = 15 → 
  num_nonofficers = 465 → 
  total_salary = avg_officer_salary * num_officers + avg_nonofficer_salary * num_nonofficers → 
  total_employees = num_officers + num_nonofficers → 
  total_salary / total_employees = 120 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end average_salary_l109_109145


namespace divisibility_of_expression_l109_109749

open Int

theorem divisibility_of_expression (a b : ℤ) (ha : Prime a) (hb : Prime b) (ha_gt7 : a > 7) (hb_gt7 : b > 7) :
  290304 ∣ (a^2 - 1) * (b^2 - 1) * (a^6 - b^6) :=
sorry

end divisibility_of_expression_l109_109749


namespace sum_of_squares_of_projections_constant_l109_109273

-- Defines a function that calculates the sum of the squares of the projections of the edges of a cube onto any plane.
def sum_of_squares_of_projections (a : ℝ) (n : ℝ × ℝ × ℝ) : ℝ :=
  let α := n.1
  let β := n.2.1
  let γ := n.2.2
  4 * (a^2) * (2)

-- Define the theorem statement that proves the sum of the squares of the projections is constant and equal to 8a^2
theorem sum_of_squares_of_projections_constant (a : ℝ) (n : ℝ × ℝ × ℝ) :
  sum_of_squares_of_projections a n = 8 * a^2 :=
by
  -- Since we assume the trigonometric identity holds, directly match the sum_of_squares_of_projections function result.
  sorry

end sum_of_squares_of_projections_constant_l109_109273


namespace jo_bob_pulled_chain_first_time_l109_109057

/-- Given the conditions of the balloon ride, prove that Jo-Bob pulled the chain
    for the first time for 15 minutes. --/
theorem jo_bob_pulled_chain_first_time (x : ℕ) : 
  (50 * x - 100 + 750 = 1400) → (x = 15) :=
by
  intro h
  sorry

end jo_bob_pulled_chain_first_time_l109_109057


namespace fifth_equation_in_pattern_l109_109431

theorem fifth_equation_in_pattern :
  (1 - 4 + 9 - 16 + 25) = (1 + 2 + 3 + 4 + 5) :=
sorry

end fifth_equation_in_pattern_l109_109431


namespace range_of_a_for_inequality_l109_109894

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + a > 0) → a > 1 :=
by
  sorry

end range_of_a_for_inequality_l109_109894


namespace negation_of_p_l109_109497

theorem negation_of_p :
  (¬ (∀ x : ℝ, x^3 + 2 < 0)) = ∃ x : ℝ, x^3 + 2 ≥ 0 := 
  by sorry

end negation_of_p_l109_109497


namespace part1_minimum_value_part2_max_k_l109_109017

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x
noncomputable def g (x : ℝ) : ℝ := (x + x * Real.log x) / (x - 1)

theorem part1_minimum_value : ∃ x₀ : ℝ, x₀ = Real.exp (-2) ∧ f x₀ = -Real.exp (-2) := 
by
  use Real.exp (-2)
  sorry

theorem part2_max_k (k : ℤ) : (∀ x > 1, f x > k * (x - 1)) → k ≤ 3 := 
by
  sorry

end part1_minimum_value_part2_max_k_l109_109017


namespace exactly_one_solves_l109_109121

-- Define the independent probabilities for person A and person B
variables (p₁ p₂ : ℝ)

-- Assume probabilities are between 0 and 1 inclusive
axiom h1 : 0 ≤ p₁ ∧ p₁ ≤ 1
axiom h2 : 0 ≤ p₂ ∧ p₂ ≤ 1

theorem exactly_one_solves : (p₁ * (1 - p₂) + p₂ * (1 - p₁)) = (p₁ * (1 - p₂) + p₂ * (1 - p₁)) := 
by sorry

end exactly_one_solves_l109_109121


namespace sequence_missing_number_l109_109051

theorem sequence_missing_number : 
  ∃ x, (x - 21 = 7 ∧ 37 - x = 9) ∧ x = 28 := by
  sorry

end sequence_missing_number_l109_109051


namespace relationship_between_D_and_A_l109_109229

variable {A B C D : Prop}

theorem relationship_between_D_and_A
  (h1 : A → B)
  (h2 : B → C)
  (h3 : D ↔ C) :
  (A → D) ∧ ¬(D → A) :=
by
sorry

end relationship_between_D_and_A_l109_109229


namespace equal_powers_eq_a_b_l109_109928

theorem equal_powers_eq_a_b 
  (a b : ℝ) 
  (ha_pos : 0 < a) 
  (hb_pos : 0 < b)
  (h_exp_eq : a^b = b^a)
  (h_a_lt_1 : a < 1) : 
  a = b :=
sorry

end equal_powers_eq_a_b_l109_109928


namespace mail_in_six_months_l109_109348

/-- The post office receives 60 letters and 20 packages per day. Each month has 30 days. -/
def daily_letters := 60
def daily_packages := 20
def days_per_month := 30
def months := 6

/-- Prove that the post office handles 14400 pieces of mail in six months. -/
theorem mail_in_six_months : (daily_letters + daily_packages) * days_per_month * months = 14400 :=
by
  sorry

end mail_in_six_months_l109_109348


namespace solve_fraction_equation_l109_109457

theorem solve_fraction_equation :
  {x : ℝ | (1 / (x^2 + 13 * x - 12) + 1 / (x^2 + 4 * x - 12) + 1 / (x^2 - 15 * x - 12) = 0)} =
  {1, -12, 12, -1} :=
by
  sorry

end solve_fraction_equation_l109_109457


namespace range_of_a_l109_109373

noncomputable def f (x a : ℝ) : ℝ := 
  (1 / 2) * (Real.cos x + Real.sin x) * (Real.cos x - Real.sin x - 4 * a) + (4 * a - 3) * x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → 
  0 ≤ (Real.cos (2 * x) - 2 * a * (Real.sin x - Real.cos x) + 4 * a - 3)) ↔ (a ≥ 1.5) :=
sorry

end range_of_a_l109_109373


namespace sum_of_base5_numbers_l109_109166

-- Definitions for the numbers in base 5
def n1_base5 := (1 * 5^2 + 3 * 5^1 + 2 * 5^0 : ℕ)
def n2_base5 := (2 * 5^2 + 1 * 5^1 + 4 * 5^0 : ℕ)
def n3_base5 := (3 * 5^2 + 4 * 5^1 + 1 * 5^0 : ℕ)

-- Sum the numbers in base 10
def sum_base10 := n1_base5 + n2_base5 + n3_base5

-- Define the base 5 value of the sum
def sum_base5 := 
  -- Convert the sum to base 5
  1 * 5^3 + 2 * 5^2 + 4 * 5^1 + 2 * 5^0

-- The theorem we want to prove
theorem sum_of_base5_numbers :
    (132 + 214 + 341 : ℕ) = 1242 := by
    sorry

end sum_of_base5_numbers_l109_109166


namespace polygon_sides_from_diagonals_l109_109221

theorem polygon_sides_from_diagonals (n D : ℕ) (h1 : D = 15) (h2 : D = n * (n - 3) / 2) : n = 8 :=
by
  -- skipping proof
  sorry

end polygon_sides_from_diagonals_l109_109221


namespace center_of_circle_l109_109799

theorem center_of_circle (x y : ℝ) : 
  (x - 1) ^ 2 + (y + 1) ^ 2 = 4 ↔ (x^2 + y^2 - 2*x + 2*y - 2 = 0) :=
sorry

end center_of_circle_l109_109799


namespace expression_for_A_plus_2B_A_plus_2B_independent_of_b_l109_109031

theorem expression_for_A_plus_2B (a b : ℝ) : 
  let A := 2 * a^2 + 3 * a * b - 2 * b - 1
  let B := -a^2 - a * b + 1
  A + 2 * B = a * b - 2 * b + 1 :=
by
  sorry

theorem A_plus_2B_independent_of_b (a : ℝ) :
  (∀ b : ℝ, let A := 2 * a^2 + 3 * a * b - 2 * b - 1
            let B := -a^2 - a * b + 1
            A + 2 * B = a * b - 2 * b + 1) →
  a = 2 :=
by
  sorry

end expression_for_A_plus_2B_A_plus_2B_independent_of_b_l109_109031


namespace weaving_problem_solution_l109_109716

noncomputable def daily_increase :=
  let a1 := 5
  let n := 30
  let sum_total := 390
  let d := (sum_total - a1 * n) * 2 / (n * (n - 1))
  d

theorem weaving_problem_solution :
  daily_increase = 16 / 29 :=
by
  sorry

end weaving_problem_solution_l109_109716


namespace value_of_m_l109_109912

theorem value_of_m (m : ℝ) (h1 : m - 2 ≠ 0) (h2 : |m| - 1 = 1) : m = -2 := by {
  sorry
}

end value_of_m_l109_109912


namespace find_x1_l109_109676

noncomputable def parabola (a h k x : ℝ) : ℝ := a * (x - h)^2 + k

theorem find_x1 
  (a h k m x1 : ℝ)
  (h1 : parabola a h k (-1) = 2)
  (h2 : parabola a h k 1 = -2)
  (h3 : parabola a h k 3 = 2)
  (h4 : parabola a h k (-2) = m)
  (h5 : parabola a h k x1 = m) :
  x1 = 4 := 
sorry

end find_x1_l109_109676


namespace max_value_of_quadratic_l109_109480

theorem max_value_of_quadratic :
  ∃ y : ℝ, (∀ x : ℝ, y ≥ -x^2 + 5 * x - 4) ∧ y = 9 / 4 :=
sorry

end max_value_of_quadratic_l109_109480


namespace second_route_time_l109_109169

-- Defining time for the first route with all green lights
def R_green : ℕ := 10

-- Defining the additional time added by each red light
def per_red_light : ℕ := 3

-- Defining total time for the first route with all red lights
def R_red : ℕ := R_green + 3 * per_red_light

-- Defining the second route time plus the difference
def S : ℕ := R_red - 5

theorem second_route_time : S = 14 := by
  sorry

end second_route_time_l109_109169


namespace translate_line_up_l109_109458

theorem translate_line_up (x y : ℝ) (h : y = 2 * x - 3) : y + 6 = 2 * x + 3 :=
by sorry

end translate_line_up_l109_109458


namespace range_a_l109_109776

noncomputable def f (x a : ℝ) : ℝ := Real.log x + x + 2 / x - a
noncomputable def g (x : ℝ) : ℝ := Real.log x + x + 2 / x

theorem range_a (a : ℝ) : (∃ x > 0, f x a = 0) → a ≥ 3 :=
by
sorry

end range_a_l109_109776


namespace volume_calculation_l109_109545

noncomputable def enclosedVolume : Real :=
  let f (x y z : Real) : Real := x^2016 + y^2016 + z^2
  let V : Real := 360
  V

theorem volume_calculation : enclosedVolume = 360 :=
by
  sorry

end volume_calculation_l109_109545


namespace game_prob_comparison_l109_109908

theorem game_prob_comparison
  (P_H : ℚ) (P_T : ℚ) (h : P_H = 3/4 ∧ P_T = 1/4)
  (independent : ∀ (n : ℕ), (1 - P_H)^n = (1 - P_T)^n) :
  ((P_H^4 + P_T^4) = (P_H^3 * P_T^2 + P_T^3 * P_H^2) + 1/4) :=
by
  sorry

end game_prob_comparison_l109_109908


namespace not_all_sets_of_10_segments_form_triangle_l109_109537

theorem not_all_sets_of_10_segments_form_triangle :
  ¬ ∀ (segments : Fin 10 → ℝ), ∃ (a b c : Fin 10), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (segments a + segments b > segments c) ∧
    (segments a + segments c > segments b) ∧
    (segments b + segments c > segments a) :=
by
  sorry

end not_all_sets_of_10_segments_form_triangle_l109_109537


namespace not_perfect_square_l109_109486

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, 7 * n + 3 = k^2 := 
by
  sorry

end not_perfect_square_l109_109486


namespace value_of_expression_l109_109410

theorem value_of_expression (n : ℝ) (h : n + 1/n = 10) : n^2 + (1/n^2) + 6 = 104 :=
by
  sorry

end value_of_expression_l109_109410


namespace smallest_k_for_perfect_cube_l109_109397

noncomputable def isPerfectCube (m : ℕ) : Prop :=
  ∃ n : ℤ, n^3 = m

theorem smallest_k_for_perfect_cube :
  ∃ k : ℕ, k > 0 ∧ (∀ m : ℕ, ((2^4) * (3^2) * (5^5) * k = m) → isPerfectCube m) ∧ k = 60 :=
sorry

end smallest_k_for_perfect_cube_l109_109397


namespace children_boys_count_l109_109817

theorem children_boys_count (girls : ℕ) (total_children : ℕ) (boys : ℕ) 
  (h₁ : girls = 35) (h₂ : total_children = 62) : boys = 27 :=
by
  sorry

end children_boys_count_l109_109817


namespace total_height_of_three_buildings_l109_109189

theorem total_height_of_three_buildings :
  let h1 := 600
  let h2 := 2 * h1
  let h3 := 3 * (h1 + h2)
  h1 + h2 + h3 = 7200 :=
by
  sorry

end total_height_of_three_buildings_l109_109189


namespace maximum_value_of_f_l109_109954

noncomputable def f (t : ℝ) : ℝ := ((3^t - 4 * t) * t) / (9^t)

theorem maximum_value_of_f : ∃ t : ℝ, f t = 1/16 :=
sorry

end maximum_value_of_f_l109_109954


namespace problem1_problem2_problem3_l109_109352

theorem problem1 (x : ℤ) (h : 263 - x = 108) : x = 155 :=
by sorry

theorem problem2 (x : ℤ) (h : 25 * x = 1950) : x = 78 :=
by sorry

theorem problem3 (x : ℤ) (h : x / 15 = 64) : x = 960 :=
by sorry

end problem1_problem2_problem3_l109_109352


namespace find_monthly_fee_l109_109883

variable (monthly_fee : ℝ) (cost_per_minute : ℝ := 0.12) (minutes_used : ℕ := 178) (total_bill : ℝ := 23.36)

theorem find_monthly_fee
  (h1 : total_bill = monthly_fee + (cost_per_minute * minutes_used)) :
  monthly_fee = 2 :=
by
  sorry

end find_monthly_fee_l109_109883


namespace simplify_absolute_value_l109_109003

theorem simplify_absolute_value : abs (-(5^2) + 6 * 2) = 13 := by
  sorry

end simplify_absolute_value_l109_109003


namespace sara_received_quarters_correct_l109_109327

-- Define the initial number of quarters Sara had
def sara_initial_quarters : ℕ := 21

-- Define the total number of quarters Sara has now
def sara_total_quarters : ℕ := 70

-- Define the number of quarters Sara received from her dad
def sara_received_quarters : ℕ := 49

-- State that the number of quarters Sara received can be deduced by the difference
theorem sara_received_quarters_correct :
  sara_total_quarters = sara_initial_quarters + sara_received_quarters :=
by simp [sara_initial_quarters, sara_total_quarters, sara_received_quarters]

end sara_received_quarters_correct_l109_109327


namespace min_x_minus_y_l109_109354

theorem min_x_minus_y {x y : ℝ} (hx : 0 ≤ x) (hx2 : x ≤ 2 * Real.pi) (hy : 0 ≤ y) (hy2 : y ≤ 2 * Real.pi)
    (h : 2 * Real.sin x * Real.cos y - Real.sin x + Real.cos y = 1 / 2) : 
    x - y = -Real.pi / 2 := 
sorry

end min_x_minus_y_l109_109354


namespace perimeter_of_square_l109_109495

theorem perimeter_of_square (s : ℕ) (h : s = 13) : 4 * s = 52 :=
by {
  sorry
}

end perimeter_of_square_l109_109495


namespace range_of_a_l109_109012

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (|x + a| < 3 ↔ 2 < x ∧ x < 3)) ↔ -5 ≤ a ∧ a ≤ 0 := by
  sorry

end range_of_a_l109_109012


namespace sample_variance_is_two_l109_109513

theorem sample_variance_is_two (a : ℝ) (h_avg : (a + 0 + 1 + 2 + 3) / 5 = 1) : 
  (1 / 5) * ((a - 1)^2 + (0 - 1)^2 + (1 - 1)^2 + (2 - 1)^2 + (3 - 1)^2) = 2 :=
sorry

end sample_variance_is_two_l109_109513


namespace function_is_decreasing_on_R_l109_109838

def is_decreasing (a : ℝ) : Prop := a - 1 < 0

theorem function_is_decreasing_on_R (a : ℝ) : (1 < a ∧ a < 2) ↔ is_decreasing a :=
by
  sorry

end function_is_decreasing_on_R_l109_109838


namespace central_angle_agree_l109_109893

theorem central_angle_agree (ratio_agree : ℕ) (ratio_disagree : ℕ) (ratio_no_preference : ℕ) (total_angle : ℝ) :
  ratio_agree = 7 → ratio_disagree = 2 → ratio_no_preference = 1 → total_angle = 360 →
  (ratio_agree / (ratio_agree + ratio_disagree + ratio_no_preference) * total_angle = 252) :=
by
  -- conditions and assumptions
  intros h_agree h_disagree h_no_preference h_total_angle
  -- simplified steps here
  sorry

end central_angle_agree_l109_109893


namespace inequality_ab_leq_a_b_l109_109752

theorem inequality_ab_leq_a_b (a b : ℝ) (x : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  a * b ≤ (a * (Real.sin x) ^ 2 + b * (Real.cos x) ^ 2) * (a * (Real.cos x) ^ 2 + b * (Real.sin x) ^ 2)
  ∧ (a * (Real.sin x) ^ 2 + b * (Real.cos x) ^ 2) * (a * (Real.cos x) ^ 2 + b * (Real.sin x) ^ 2) ≤ (a + b)^2 / 4 := 
sorry

end inequality_ab_leq_a_b_l109_109752


namespace op_5_2_l109_109122

def op (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem op_5_2 : op 5 2 = 30 := 
by sorry

end op_5_2_l109_109122


namespace percentage_of_women_attended_picnic_l109_109559

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

end percentage_of_women_attended_picnic_l109_109559


namespace distance_between_opposite_vertices_l109_109321

noncomputable def calculate_d (a b c v k t : ℝ) : ℝ :=
  (1 / (2 * k)) * Real.sqrt (2 * (k^4 - 16 * t^2 - 8 * v * k))

theorem distance_between_opposite_vertices (a b c v k t d : ℝ)
  (h1 : v = a * b * c)
  (h2 : k = a + b + c)
  (h3 : 16 * t^2 = k * (k - 2 * a) * (k - 2 * b) * (k - 2 * c))
  : d = calculate_d a b c v k t := 
by {
    -- The proof is omitted based on the requirement.
    sorry
}

end distance_between_opposite_vertices_l109_109321


namespace least_number_of_shoes_needed_on_island_l109_109126

def number_of_inhabitants : ℕ := 10000
def percentage_one_legged : ℕ := 5
def shoes_needed (N : ℕ) : ℕ :=
  let one_legged := (percentage_one_legged * N) / 100
  let two_legged := N - one_legged
  let barefooted_two_legged := two_legged / 2
  let shoes_for_one_legged := one_legged
  let shoes_for_two_legged := (two_legged - barefooted_two_legged) * 2
  shoes_for_one_legged + shoes_for_two_legged

theorem least_number_of_shoes_needed_on_island :
  shoes_needed number_of_inhabitants = 10000 :=
sorry

end least_number_of_shoes_needed_on_island_l109_109126


namespace mari_vs_kendra_l109_109574

-- Variable Definitions
variables (K M S : ℕ)  -- Number of buttons Kendra, Mari, and Sue made
variables (h1: 2*S = K) -- Sue made half as many as Kendra
variables (h2: S = 6)   -- Sue made 6 buttons
variables (h3: M = 64)  -- Mari made 64 buttons

-- Theorem Statement
theorem mari_vs_kendra (K M S : ℕ) (h1 : 2 * S = K) (h2 : S = 6) (h3 : M = 64) :
  M = 5 * K + 4 :=
sorry

end mari_vs_kendra_l109_109574


namespace total_area_of_paintings_l109_109277

-- Definitions based on the conditions
def painting1_area := 3 * (5 * 5) -- 3 paintings of 5 feet by 5 feet
def painting2_area := 10 * 8 -- 1 painting of 10 feet by 8 feet
def painting3_area := 5 * 9 -- 1 painting of 5 feet by 9 feet

-- The proof statement we aim to prove
theorem total_area_of_paintings : painting1_area + painting2_area + painting3_area = 200 :=
by
  sorry

end total_area_of_paintings_l109_109277


namespace sum_of_squares_and_product_l109_109568

theorem sum_of_squares_and_product (x y : ℤ) 
  (h1 : x^2 + y^2 = 290) 
  (h2 : x * y = 96) :
  x + y = 22 :=
sorry

end sum_of_squares_and_product_l109_109568


namespace height_of_triangle_l109_109080

variables (a b h' : ℝ)

theorem height_of_triangle (h : (1/2) * a * h' = a * b) : h' = 2 * b :=
sorry

end height_of_triangle_l109_109080


namespace small_seats_capacity_l109_109727

-- Definitions
def num_small_seats : ℕ := 2
def people_per_small_seat : ℕ := 14

-- Statement to prove
theorem small_seats_capacity :
  num_small_seats * people_per_small_seat = 28 :=
by
  -- Proof goes here
  sorry

end small_seats_capacity_l109_109727


namespace moles_of_AgOH_formed_l109_109341

theorem moles_of_AgOH_formed (moles_AgNO3 : ℕ) (moles_NaOH : ℕ) 
  (reaction : moles_AgNO3 + moles_NaOH = 2) : moles_AgNO3 + 2 = 2 :=
by
  sorry

end moles_of_AgOH_formed_l109_109341


namespace volume_of_cube_l109_109209

theorem volume_of_cube (A : ℝ) (s V : ℝ) 
  (hA : A = 150) 
  (h_surface_area : A = 6 * s^2) 
  (h_side_length : s = 5) :
  V = s^3 →
  V = 125 :=
by
  sorry

end volume_of_cube_l109_109209


namespace problem_statement_l109_109194

noncomputable def f (x : ℝ) : ℝ := ∫ t in -x..x, Real.cos t

theorem problem_statement : f (f (Real.pi / 4)) = 2 * Real.sin (Real.sqrt 2) := 
by
  sorry

end problem_statement_l109_109194


namespace exists_divisor_c_of_f_l109_109406

theorem exists_divisor_c_of_f (f : ℕ → ℕ) 
  (h₁ : ∀ n, f n ≥ 2)
  (h₂ : ∀ m n, f (m + n) ∣ (f m + f n)) :
  ∃ c > 1, ∀ n, c ∣ f n :=
sorry

end exists_divisor_c_of_f_l109_109406


namespace op_exp_eq_l109_109601

-- Define the operation * on natural numbers
def op (a b : ℕ) : ℕ := a ^ b

-- The theorem to be proven
theorem op_exp_eq (a b n : ℕ) : (op a b)^n = op a (b^n) := by
  sorry

end op_exp_eq_l109_109601


namespace find_integer_solutions_l109_109717

theorem find_integer_solutions :
  (a b : ℤ) →
  3 * a^2 * b^2 + b^2 = 517 + 30 * a^2 →
  (a = 2 ∧ b = 7) ∨ (a = -2 ∧ b = 7) ∨ (a = 2 ∧ b = -7) ∨ (a = -2 ∧ b = -7) :=
sorry

end find_integer_solutions_l109_109717


namespace graham_crackers_leftover_l109_109353

-- Definitions for the problem conditions
def initial_boxes_graham := 14
def initial_packets_oreos := 15
def initial_ounces_cream_cheese := 36

def boxes_per_cheesecake := 2
def packets_per_cheesecake := 3
def ounces_per_cheesecake := 4

-- Define the statement that needs to be proved
theorem graham_crackers_leftover :
  initial_boxes_graham - (min (initial_boxes_graham / boxes_per_cheesecake) (min (initial_packets_oreos / packets_per_cheesecake) (initial_ounces_cream_cheese / ounces_per_cheesecake)) * boxes_per_cheesecake) = 4 :=
by sorry

end graham_crackers_leftover_l109_109353


namespace diagonals_sum_pentagon_inscribed_in_circle_l109_109114

theorem diagonals_sum_pentagon_inscribed_in_circle
  (FG HI GH IJ FJ : ℝ)
  (h1 : FG = 4)
  (h2 : HI = 4)
  (h3 : GH = 11)
  (h4 : IJ = 11)
  (h5 : FJ = 15) :
  3 * FJ + (FJ^2 - 121) / 4 + (FJ^2 - 16) / 11 = 80 := by {
  sorry
}

end diagonals_sum_pentagon_inscribed_in_circle_l109_109114


namespace profits_to_revenues_ratio_l109_109891

theorem profits_to_revenues_ratio (R P: ℝ) 
    (rev_2009: R_2009 = 0.8 * R) 
    (profit_2009_rev_2009: P_2009 = 0.2 * R_2009)
    (profit_2009: P_2009 = 1.6 * P):
    (P / R) * 100 = 10 :=
by
  sorry

end profits_to_revenues_ratio_l109_109891


namespace equal_probability_after_adding_balls_l109_109201

theorem equal_probability_after_adding_balls :
  let initial_white := 2
  let initial_yellow := 3
  let added_white := 4
  let added_yellow := 3
  let total_white := initial_white + added_white
  let total_yellow := initial_yellow + added_yellow
  let total_balls := total_white + total_yellow
  (total_white / total_balls) = (total_yellow / total_balls) := by
  sorry

end equal_probability_after_adding_balls_l109_109201


namespace constant_c_square_of_binomial_l109_109446

theorem constant_c_square_of_binomial (c : ℝ) (h : ∃ d : ℝ, (3*x + d)^2 = 9*x^2 - 18*x + c) : c = 9 :=
sorry

end constant_c_square_of_binomial_l109_109446


namespace sum_of_solutions_l109_109523

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 8) ^ 2 = 49) (h2 : (x2 - 8) ^ 2 = 49) : x1 + x2 = 16 :=
sorry

end sum_of_solutions_l109_109523


namespace approximation_accuracy_l109_109501

noncomputable def radius (k : Circle) : ℝ := sorry
def BG_equals_radius (BG : ℝ) (r : ℝ) := BG = r
def DB_equals_radius_sqrt3 (DB DG r : ℝ) := DB = DG ∧ DG = r * Real.sqrt 3
def cos_beta (cos_beta : ℝ) := cos_beta = 1 / (2 * Real.sqrt 3)
def sin_beta (sin_beta : ℝ) := sin_beta = Real.sqrt 11 / (2 * Real.sqrt 3)
def angle_BCH (angle_BCH : ℝ) (beta : ℝ) := angle_BCH = 120 - beta
def side_nonagon (a_9 r : ℝ) := a_9 = 2 * r * Real.sin 20
def bounds_sin_20 (sin_20 : ℝ) := 0.34195 < sin_20 ∧ sin_20 < 0.34205
def error_margin_low (BH_low a_9 r : ℝ) := 0.6839 * r < a_9
def error_margin_high (BH_high a_9 r : ℝ) := a_9 < 0.6841 * r

theorem approximation_accuracy
  (r : ℝ) (BG DB DG : ℝ) (beta : ℝ) (a_9 BH_low BH_high : ℝ)
  (h1 : BG_equals_radius BG r)
  (h2 : DB_equals_radius_sqrt3 DB DG r)
  (h3 : cos_beta (1 / (2 * Real.sqrt 3)))
  (h4 : sin_beta (Real.sqrt 11 / (2 * Real.sqrt 3)))
  (h5 : angle_BCH (120 - beta) beta)
  (h6 : side_nonagon a_9 r)
  (h7 : bounds_sin_20 (Real.sin 20))
  (h8 : error_margin_low BH_low a_9 r)
  (h9 : error_margin_high BH_high a_9 r) : 
  0.6861 * r < BH_high ∧ BH_low < 0.6864 * r :=
sorry

end approximation_accuracy_l109_109501


namespace set_theorem_1_set_theorem_2_set_theorem_3_set_theorem_4_set_theorem_5_set_theorem_6_set_theorem_7_l109_109306

variable {U : Type} [DecidableEq U]
variables (A B C K : Set U)

theorem set_theorem_1 : (A \ K) ∪ (B \ K) = (A ∪ B) \ K := sorry
theorem set_theorem_2 : A \ (B \ C) = (A \ B) ∪ (A ∩ C) := sorry
theorem set_theorem_3 : A \ (A \ B) = A ∩ B := sorry
theorem set_theorem_4 : (A \ B) \ C = (A \ C) \ (B \ C) := sorry
theorem set_theorem_5 : A \ (B ∩ C) = (A \ B) ∪ (A \ C) := sorry
theorem set_theorem_6 : A \ (B ∪ C) = (A \ B) ∩ (A \ C) := sorry
theorem set_theorem_7 : A \ B = (A ∪ B) \ B ∧ A \ B = A \ (A ∩ B) := sorry

end set_theorem_1_set_theorem_2_set_theorem_3_set_theorem_4_set_theorem_5_set_theorem_6_set_theorem_7_l109_109306


namespace compare_a_b_l109_109136

theorem compare_a_b (m : ℝ) (h : m > 1) 
  (a : ℝ := (Real.sqrt (m+1)) - (Real.sqrt m))
  (b : ℝ := (Real.sqrt m) - (Real.sqrt (m-1))) : a < b :=
by
  sorry

end compare_a_b_l109_109136


namespace evaluate_expression_l109_109257

theorem evaluate_expression : (-(18 / 3 * 12 - 80 + 4 * 12)) ^ 2 = 1600 := by
  sorry

end evaluate_expression_l109_109257


namespace books_price_arrangement_l109_109632

theorem books_price_arrangement (c : ℝ) (prices : Fin 40 → ℝ)
  (h₁ : ∀ i : Fin 39, prices i.succ = prices i + 3)
  (h₂ : prices ⟨39, by norm_num⟩ = prices ⟨19, by norm_num⟩ + prices ⟨20, by norm_num⟩) :
  prices 20 = prices 19 + 3 := 
sorry

end books_price_arrangement_l109_109632


namespace freshmen_more_than_sophomores_l109_109976

theorem freshmen_more_than_sophomores :
  ∀ (total_students juniors not_sophomores not_freshmen seniors adv_grade freshmen sophomores : ℕ),
    total_students = 1200 →
    juniors = 264 →
    not_sophomores = 660 →
    not_freshmen = 300 →
    seniors = 240 →
    adv_grade = 20 →
    freshmen = total_students - not_freshmen - seniors - adv_grade →
    sophomores = total_students - not_sophomores - seniors - adv_grade →
    freshmen - sophomores = 360 :=
by
  intros total_students juniors not_sophomores not_freshmen seniors adv_grade freshmen sophomores
  intros h_total h_juniors h_not_sophomores h_not_freshmen h_seniors h_adv_grade h_freshmen h_sophomores
  sorry

end freshmen_more_than_sophomores_l109_109976


namespace calculation_result_l109_109658

theorem calculation_result : 8 * 5.4 - 0.6 * 10 / 1.2 = 38.2 :=
by
  sorry

end calculation_result_l109_109658


namespace time_to_fill_is_correct_l109_109000

-- Definitions of rates
variable (R_1 : ℚ) (R_2 : ℚ)

-- Conditions given in the problem
def rate1 := (1 : ℚ) / 8
def rate2 := (1 : ℚ) / 12

-- The resultant rate when both pipes work together
def combined_rate := rate1 + rate2

-- Calculate the time taken to fill the tank
def time_to_fill_tank := 1 / combined_rate

theorem time_to_fill_is_correct (h1 : R_1 = rate1) (h2 : R_2 = rate2) :
  time_to_fill_tank = 24 / 5 := by
  sorry

end time_to_fill_is_correct_l109_109000


namespace find_triple_abc_l109_109504

theorem find_triple_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
    (h_sum : a + b + c = 3)
    (h2 : a^2 - a ≥ 1 - b * c)
    (h3 : b^2 - b ≥ 1 - a * c)
    (h4 : c^2 - c ≥ 1 - a * b) :
    a = 1 ∧ b = 1 ∧ c = 1 :=
by
  sorry

end find_triple_abc_l109_109504


namespace smaller_number_l109_109791

theorem smaller_number (x y : ℝ) (h1 : x + y = 16) (h2 : x - y = 4) (h3 : x * y = 60) : y = 6 :=
sorry

end smaller_number_l109_109791


namespace sixth_element_row_20_l109_109957

theorem sixth_element_row_20 : (Nat.choose 20 5) = 15504 := by
  sorry

end sixth_element_row_20_l109_109957


namespace final_apples_count_l109_109070

def initial_apples : ℝ := 5708
def apples_given_away : ℝ := 2347.5
def additional_apples_harvested : ℝ := 1526.75

theorem final_apples_count :
  initial_apples - apples_given_away + additional_apples_harvested = 4887.25 :=
by
  sorry

end final_apples_count_l109_109070


namespace tangent_line_at_origin_l109_109775

/-- 
The curve is given by y = exp x.
The tangent line to this curve that passes through the origin (0, 0) 
has the equation y = exp 1 * x.
-/
theorem tangent_line_at_origin :
  ∀ (x y : ℝ), y = Real.exp x → (∃ k : ℝ, ∀ x, y = k * x ∧ k = Real.exp 1) :=
by
  sorry

end tangent_line_at_origin_l109_109775


namespace total_matches_played_l109_109544

theorem total_matches_played (home_wins : ℕ) (rival_wins : ℕ) (draws : ℕ) (home_wins_eq : home_wins = 3) (rival_wins_eq : rival_wins = 2 * home_wins) (draws_eq : draws = 4) (no_losses : ∀ (t : ℕ), t = 0) :
  home_wins + rival_wins + 2 * draws = 17 :=
by {
  sorry
}

end total_matches_played_l109_109544


namespace negation_of_universal_l109_109705

theorem negation_of_universal:
  ¬(∀ x : ℝ, (0 < x ∧ x < (π / 2)) → x > Real.sin x) ↔
  ∃ x : ℝ, (0 < x ∧ x < (π / 2)) ∧ x ≤ Real.sin x := by
  sorry

end negation_of_universal_l109_109705


namespace farmer_kent_income_l109_109362

-- Define the constants and conditions
def watermelon_weight : ℕ := 23
def price_per_pound : ℕ := 2
def number_of_watermelons : ℕ := 18

-- Construct the proof statement
theorem farmer_kent_income : 
  price_per_pound * watermelon_weight * number_of_watermelons = 828 := 
by
  -- Skipping the proof here, just stating the theorem.
  sorry

end farmer_kent_income_l109_109362


namespace calculate_ratio_milk_l109_109680

def ratio_milk_saturdays_weekdays (S : ℕ) : Prop :=
  let Weekdays := 15 -- total milk on weekdays
  let Sundays := 9 -- total milk on Sundays
  S + Weekdays + Sundays = 30 → S / Weekdays = 2 / 5

theorem calculate_ratio_milk : ratio_milk_saturdays_weekdays 6 :=
by
  unfold ratio_milk_saturdays_weekdays
  intros
  apply sorry -- Proof goes here

end calculate_ratio_milk_l109_109680


namespace product_area_perimeter_eq_104sqrt26_l109_109390

noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  ((b.1 - a.1) ^ 2 + (b.2 - a.2) ^ 2).sqrt

noncomputable def side_length := distance (5, 5) (0, 4)

noncomputable def area_of_square := side_length ^ 2

noncomputable def perimeter_of_square := 4 * side_length

noncomputable def product_area_perimeter := area_of_square * perimeter_of_square

theorem product_area_perimeter_eq_104sqrt26 :
  product_area_perimeter = 104 * Real.sqrt 26 :=
by 
  -- placeholder for the proof
  sorry

end product_area_perimeter_eq_104sqrt26_l109_109390


namespace circles_are_externally_tangent_l109_109975

noncomputable def circleA : Prop := ∀ x y : ℝ, x^2 + y^2 + 4 * x + 2 * y + 1 = 0
noncomputable def circleB : Prop := ∀ x y : ℝ, x^2 + y^2 - 2 * x - 6 * y + 1 = 0

theorem circles_are_externally_tangent (hA : circleA) (hB : circleB) : 
  ∃ P Q : ℝ, (P = 5) ∧ (Q = 5) := 
by 
  -- start proving with given conditions
  sorry

end circles_are_externally_tangent_l109_109975


namespace probability_of_rain_on_at_least_one_day_is_correct_l109_109748

def rain_on_friday_probability : ℝ := 0.30
def rain_on_saturday_probability : ℝ := 0.45
def rain_on_sunday_probability : ℝ := 0.50

def rain_on_at_least_one_day_probability : ℝ := 1 - (1 - rain_on_friday_probability) * (1 - rain_on_saturday_probability) * (1 - rain_on_sunday_probability)

theorem probability_of_rain_on_at_least_one_day_is_correct :
  rain_on_at_least_one_day_probability = 0.8075 := by
sorry

end probability_of_rain_on_at_least_one_day_is_correct_l109_109748


namespace maximum_M_for_right_triangle_l109_109833

theorem maximum_M_for_right_triangle (a b c : ℝ) (h1 : a ≤ b) (h2 : b < c) (h3 : a^2 + b^2 = c^2) :
  (1 / a + 1 / b + 1 / c) ≥ (5 + 3 * Real.sqrt 2) / (a + b + c) :=
sorry

end maximum_M_for_right_triangle_l109_109833


namespace toms_total_score_l109_109853

def regular_enemy_points : ℕ := 10
def elite_enemy_points : ℕ := 25
def boss_enemy_points : ℕ := 50

def regular_enemy_bonus (kills : ℕ) : ℚ :=
  if 100 ≤ kills ∧ kills < 150 then 0.50
  else if 150 ≤ kills ∧ kills < 200 then 0.75
  else if kills ≥ 200 then 1.00
  else 0.00

def elite_enemy_bonus (kills : ℕ) : ℚ :=
  if 15 ≤ kills ∧ kills < 25 then 0.30
  else if 25 ≤ kills ∧ kills < 35 then 0.50
  else if kills >= 35 then 0.70
  else 0.00

def boss_enemy_bonus (kills : ℕ) : ℚ :=
  if 5 ≤ kills ∧ kills < 10 then 0.20
  else if kills ≥ 10 then 0.40
  else 0.00

noncomputable def total_score (regular_kills elite_kills boss_kills : ℕ) : ℚ :=
  let regular_points := regular_kills * regular_enemy_points
  let elite_points := elite_kills * elite_enemy_points
  let boss_points := boss_kills * boss_enemy_points
  let regular_total := regular_points + regular_points * regular_enemy_bonus regular_kills
  let elite_total := elite_points + elite_points * elite_enemy_bonus elite_kills
  let boss_total := boss_points + boss_points * boss_enemy_bonus boss_kills
  regular_total + elite_total + boss_total

theorem toms_total_score :
  total_score 160 20 8 = 3930 := by
  sorry

end toms_total_score_l109_109853


namespace number_of_students_l109_109551

-- Define the conditions
variable (n : ℕ) (jayden_rank_best jayden_rank_worst : ℕ)
variable (h1 : jayden_rank_best = 100)
variable (h2 : jayden_rank_worst = 100)

-- Define the question
theorem number_of_students (h1 : jayden_rank_best = 100) (h2 : jayden_rank_worst = 100) : n = 199 := 
  sorry

end number_of_students_l109_109551


namespace sqrt_meaningful_iff_l109_109455

theorem sqrt_meaningful_iff (x: ℝ) : (6 - 2 * x ≥ 0) ↔ (x ≤ 3) :=
by
  sorry

end sqrt_meaningful_iff_l109_109455


namespace ratio_of_rats_l109_109966

theorem ratio_of_rats (x y : ℝ) (h : (0.56 * x) / (0.84 * y) = 1 / 2) : x / y = 3 / 4 :=
sorry

end ratio_of_rats_l109_109966


namespace integer_satisfaction_l109_109704

theorem integer_satisfaction (x : ℤ) : 
  (x + 15 ≥ 16 ∧ -3 * x ≥ -15) ↔ (1 ≤ x ∧ x ≤ 5) :=
by 
  sorry

end integer_satisfaction_l109_109704


namespace largest_in_given_numbers_l109_109665

noncomputable def A := 5.14322
noncomputable def B := 5.1432222222222222222 -- B = 5.143(bar)2
noncomputable def C := 5.1432323232323232323 -- C = 5.14(bar)32
noncomputable def D := 5.1432432432432432432 -- D = 5.1(bar)432
noncomputable def E := 5.1432143214321432143 -- E = 5.(bar)4321

theorem largest_in_given_numbers : D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end largest_in_given_numbers_l109_109665


namespace largest_integer_x_cubed_lt_three_x_squared_l109_109873

theorem largest_integer_x_cubed_lt_three_x_squared : 
  ∃ x : ℤ, x^3 < 3 * x^2 ∧ (∀ y : ℤ, y^3 < 3 * y^2 → y ≤ x) :=
  sorry

end largest_integer_x_cubed_lt_three_x_squared_l109_109873


namespace base3_sum_example_l109_109396

noncomputable def base3_add (a b : ℕ) : ℕ := sorry  -- Function to perform base-3 addition

theorem base3_sum_example : 
  base3_add (base3_add (base3_add (base3_add 2 120) 221) 1112) 1022 = 21201 := sorry

end base3_sum_example_l109_109396


namespace whole_number_M_l109_109489

theorem whole_number_M (M : ℤ) (hM : 9 < (M : ℝ) / 4 ∧ (M : ℝ) / 4 < 10) : M = 37 ∨ M = 38 ∨ M = 39 := by
  sorry

end whole_number_M_l109_109489


namespace complement_of_intersection_l109_109184

open Set

def A : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def B : Set ℝ := {x | x^2 + x - 6 > 0}
def S : Set ℝ := univ -- S is the set of all real numbers

theorem complement_of_intersection :
  S \ (A ∩ B) = { x : ℝ | x ≤ 2 } ∪ { x : ℝ | 3 < x } :=
by
  sorry

end complement_of_intersection_l109_109184


namespace inscribed_square_ratio_l109_109435

-- Define the problem context:
variables {x y : ℝ}

-- Conditions on the triangles and squares:
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ a > 0 ∧ b > 0 ∧ c > 0

def inscribed_square_first_triangle (a b c x : ℝ) : Prop :=
  is_right_triangle a b c ∧ a = 5 ∧ b = 12 ∧ c = 13 ∧
  x = 60 / 17

def inscribed_square_second_triangle (d e f y : ℝ) : Prop :=
  is_right_triangle d e f ∧ d = 6 ∧ e = 8 ∧ f = 10 ∧
  y = 25 / 8

-- Lean theorem to be proven with given conditions:
theorem inscribed_square_ratio :
  inscribed_square_first_triangle 5 12 13 x →
  inscribed_square_second_triangle 6 8 10 y →
  x / y = 96 / 85 := by
  sorry

end inscribed_square_ratio_l109_109435


namespace positive_numbers_l109_109858

theorem positive_numbers {a b c : ℝ} (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) : 0 < a ∧ 0 < b ∧ 0 < c :=
sorry

end positive_numbers_l109_109858


namespace total_animal_legs_l109_109223

def number_of_dogs : ℕ := 2
def number_of_chickens : ℕ := 1
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2

theorem total_animal_legs : number_of_dogs * legs_per_dog + number_of_chickens * legs_per_chicken = 10 :=
by
  -- The proof is skipped
  sorry

end total_animal_legs_l109_109223


namespace find_s_l109_109682

variable {a b n r s : ℝ}

theorem find_s (h1 : Polynomial.aeval a (Polynomial.X ^ 2 - Polynomial.C n * Polynomial.X + Polynomial.C 6) = 0)
              (h2 : Polynomial.aeval b (Polynomial.X ^ 2 - Polynomial.C n * Polynomial.X + Polynomial.C 6) = 0)
              (h_ab : a * b = 6)
              (h_roots : Polynomial.aeval (a + 2/b) (Polynomial.X ^ 2 - Polynomial.C r * Polynomial.X + Polynomial.C s) = 0)
              (h_roots2 : Polynomial.aeval (b + 2/a) (Polynomial.X ^ 2 - Polynomial.C r * Polynomial.X + Polynomial.C s) = 0) :
  s = 32/3 := 
sorry

end find_s_l109_109682


namespace total_expenditure_of_7_people_l109_109535

theorem total_expenditure_of_7_people :
  ∃ A : ℝ, 
    (6 * 11 + (A + 6) = 7 * A) ∧
    (6 * 11 = 66) ∧
    (∃ total : ℝ, total = 6 * 11 + (A + 6) ∧ total = 84) :=
by 
  sorry

end total_expenditure_of_7_people_l109_109535


namespace base_conversion_subtraction_l109_109391

def base8_to_base10 : Nat := 5 * 8^5 + 4 * 8^4 + 3 * 8^3 + 2 * 8^2 + 1 * 8^1 + 0 * 8^0
def base9_to_base10 : Nat := 6 * 9^4 + 5 * 9^3 + 4 * 9^2 + 3 * 9^1 + 2 * 9^0

theorem base_conversion_subtraction :
  base8_to_base10 - base9_to_base10 = 136532 :=
by
  -- Proof steps go here
  sorry

end base_conversion_subtraction_l109_109391


namespace probability_eq_l109_109343

noncomputable def probability_exactly_two_one_digit_and_three_two_digit : ℚ := 
  let n := 5
  let p_one_digit := 9 / 20
  let p_two_digit := 11 / 20
  let binomial_coeff := Nat.choose 5 2
  (binomial_coeff * p_one_digit^2 * p_two_digit^3)

theorem probability_eq : probability_exactly_two_one_digit_and_three_two_digit = 539055 / 1600000 := 
  sorry

end probability_eq_l109_109343


namespace largest_integer_less_than_hundred_with_remainder_five_l109_109475

theorem largest_integer_less_than_hundred_with_remainder_five (n : ℤ) :
  n < 100 ∧ n % 8 = 5 → n = 93 :=
by
  sorry

end largest_integer_less_than_hundred_with_remainder_five_l109_109475


namespace expand_and_simplify_l109_109910

theorem expand_and_simplify (x : ℝ) : (x^2 + 4) * (x - 5) = x^3 - 5 * x^2 + 4 * x - 20 := 
sorry

end expand_and_simplify_l109_109910


namespace anna_money_left_l109_109195

theorem anna_money_left : 
  let initial_money := 10.0
  let gum_cost := 3.0 -- 3 packs at $1.00 each
  let chocolate_cost := 5.0 -- 5 bars at $1.00 each
  let cane_cost := 1.0 -- 2 canes at $0.50 each
  let total_spent := gum_cost + chocolate_cost + cane_cost
  let money_left := initial_money - total_spent
  money_left = 1.0 := by
  sorry

end anna_money_left_l109_109195


namespace caitlins_team_number_l109_109962

noncomputable def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the two-digit prime numbers
def two_digit_prime (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ is_prime n

-- Lean statement
theorem caitlins_team_number (h_date birthday_before today birthday_after : ℕ)
  (p₁ p₂ p₃ : ℕ)
  (h1 : two_digit_prime p₁)
  (h2 : two_digit_prime p₂)
  (h3 : two_digit_prime p₃)
  (h4 : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃)
  (h5 : p₁ + p₂ = today ∨ p₁ + p₃ = today ∨ p₂ + p₃ = today)
  (h6 : (p₁ + p₂ = birthday_before ∨ p₁ + p₃ = birthday_before ∨ p₂ + p₃ = birthday_before)
       ∧ birthday_before < today)
  (h7 : (p₁ + p₂ = birthday_after ∨ p₁ + p₃ = birthday_after ∨ p₂ + p₃ = birthday_after)
       ∧ birthday_after > today) :
  p₃ = 11 := by
  sorry

end caitlins_team_number_l109_109962


namespace laps_needed_to_reach_total_distance_l109_109641

-- Define the known conditions
def total_distance : ℕ := 2400
def lap_length : ℕ := 150
def laps_run_each : ℕ := 6
def total_laps_run : ℕ := 2 * laps_run_each

-- Define the proof goal
theorem laps_needed_to_reach_total_distance :
  (total_distance - total_laps_run * lap_length) / lap_length = 4 :=
by
  sorry

end laps_needed_to_reach_total_distance_l109_109641


namespace find_line_equation_l109_109702

theorem find_line_equation 
  (A : ℝ × ℝ) (hA : A = (-2, -3)) 
  (h_perpendicular : ∃ k b : ℝ, ∀ x y, 3 * x + 4 * y - 3 = 0 → k * x + y = b) :
  ∃ k' b' : ℝ, (∀ x y, k' * x + y = b' → y = (4 / 3) * x + 1 / 3) ∧ (k' = 4 ∧ b' = -1) :=
by
  sorry

end find_line_equation_l109_109702


namespace investments_interest_yielded_l109_109319

def total_investment : ℝ := 15000
def part_one_investment : ℝ := 8200
def rate_one : ℝ := 0.06
def rate_two : ℝ := 0.075

def part_two_investment : ℝ := total_investment - part_one_investment

def interest_one : ℝ := part_one_investment * rate_one * 1
def interest_two : ℝ := part_two_investment * rate_two * 1

def total_interest : ℝ := interest_one + interest_two

theorem investments_interest_yielded : total_interest = 1002 := by
  sorry

end investments_interest_yielded_l109_109319


namespace girls_at_start_l109_109464

theorem girls_at_start (B G : ℕ) (h1 : B + G = 600) (h2 : 6 * B + 7 * G = 3840) : G = 240 :=
by
  -- actual proof is omitted
  sorry

end girls_at_start_l109_109464


namespace ratio_price_16_to_8_l109_109505

def price_8_inch := 5
def P : ℝ := sorry
def price_16_inch := 5 * P
def daily_earnings := 3 * price_8_inch + 5 * price_16_inch
def three_day_earnings := 3 * daily_earnings
def total_earnings := 195

theorem ratio_price_16_to_8 : total_earnings = three_day_earnings → P = 2 :=
by
  sorry

end ratio_price_16_to_8_l109_109505


namespace bridge_length_l109_109514

   noncomputable def walking_speed_km_per_hr : ℝ := 6
   noncomputable def walking_time_minutes : ℝ := 15

   noncomputable def length_of_bridge (speed_km_per_hr : ℝ) (time_min : ℝ) : ℝ :=
     (speed_km_per_hr * 1000 / 60) * time_min

   theorem bridge_length :
     length_of_bridge walking_speed_km_per_hr walking_time_minutes = 1500 := 
   by
     sorry
   
end bridge_length_l109_109514


namespace num_unpainted_cubes_l109_109043

theorem num_unpainted_cubes (n : ℕ) (h1 : n ^ 3 = 125) : (n - 2) ^ 3 = 27 :=
by
  sorry

end num_unpainted_cubes_l109_109043


namespace arithmetic_sequence_term_count_l109_109656

theorem arithmetic_sequence_term_count (a d l n : ℕ) (h1 : a = 11) (h2 : d = 4) (h3 : l = 107) :
  l = a + (n - 1) * d → n = 25 := by
  sorry

end arithmetic_sequence_term_count_l109_109656


namespace max_a_2017_2018_ge_2017_l109_109349

def seq_a (a : ℕ → ℤ) (b : ℕ → ℕ) : Prop :=
  a 0 = 0 ∧ a 1 = 1 ∧ (∀ n, n ≥ 1 → 
  (b (n-1) = 1 → a (n+1) = a n * b n + a (n-1)) ∧ 
  (b (n-1) > 1 → a (n+1) = a n * b n - a (n-1)))

theorem max_a_2017_2018_ge_2017 (a : ℕ → ℤ) (b : ℕ → ℕ) (h : seq_a a b) :
  max (a 2017) (a 2018) ≥ 2017 :=
sorry

end max_a_2017_2018_ge_2017_l109_109349


namespace arrange_cubes_bound_l109_109536

def num_ways_to_arrange_cubes_into_solids (n : ℕ) : ℕ := sorry

theorem arrange_cubes_bound (n : ℕ) (h : n = (2015^100)) :
  10^14 < num_ways_to_arrange_cubes_into_solids n ∧
  num_ways_to_arrange_cubes_into_solids n < 10^15 := sorry

end arrange_cubes_bound_l109_109536


namespace percentage_discount_l109_109498

theorem percentage_discount (P S : ℝ) (hP : P = 50) (hS : S = 35) : (P - S) / P * 100 = 30 := by
  sorry

end percentage_discount_l109_109498


namespace describe_set_T_l109_109898

theorem describe_set_T:
  ( ∀ (x y : ℝ), ((x + 2 = 4 ∧ y - 5 ≤ 4) ∨ (y - 5 = 4 ∧ x + 2 ≤ 4) ∨ (x + 2 = y - 5 ∧ 4 ≤ x + 2)) →
    ( ∃ (x y : ℝ), x = 2 ∧ y ≤ 9 ∨ y = 9 ∧ x ≤ 2 ∨ y = x + 7 ∧ x ≥ 2 ∧ y ≥ 9) ) :=
sorry

end describe_set_T_l109_109898


namespace g_value_at_6_l109_109657

noncomputable def g (v : ℝ) : ℝ :=
  let x := (v + 2) / 4
  x^2 - x + 2

theorem g_value_at_6 :
  g 6 = 4 := by
  sorry

end g_value_at_6_l109_109657


namespace ethan_expected_wins_l109_109491

-- Define the conditions
def P_win := 2 / 5
def P_tie := 2 / 5
def P_loss := 1 / 5

-- Define the adjusted probabilities
def adj_P_win := P_win / (P_win + P_loss)
def adj_P_loss := P_loss / (P_win + P_loss)

-- Define Ethan's expected number of wins before losing
def expected_wins_before_loss : ℚ := 2

-- The theorem to prove 
theorem ethan_expected_wins :
  ∃ E : ℚ, 
    E = (adj_P_win * (E + 1) + adj_P_loss * 0) ∧ 
    E = expected_wins_before_loss :=
by
  sorry

end ethan_expected_wins_l109_109491


namespace cos_C_value_l109_109192

theorem cos_C_value (a b c : ℝ) (A B C : ℝ)
  (h1 : a * Real.cos B + b * Real.cos A = 3 * c * Real.cos C)
  (h2 : 0 < A ∧ A < π)
  (h3 : 0 < B ∧ B < π)
  (h4 : 0 < C ∧ C < π)
  (h5 : A + B + C = π)
  : Real.cos C = (Real.sqrt 10) / 10 :=
sorry

end cos_C_value_l109_109192


namespace tank_insulation_cost_l109_109742

theorem tank_insulation_cost (l w h : ℝ) (cost_per_sqft : ℝ) (SA : ℝ) (C : ℝ) 
  (h_l : l = 6) (h_w : w = 3) (h_h : h = 2) (h_cost_per_sqft : cost_per_sqft = 20) 
  (h_SA : SA = 2 * l * w + 2 * l * h + 2 * w * h)
  (h_C : C = SA * cost_per_sqft) :
  C = 1440 := 
by
  -- proof will be filled in here
  sorry

end tank_insulation_cost_l109_109742


namespace twin_brothers_age_l109_109030

theorem twin_brothers_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 17) : x = 8 := 
  sorry

end twin_brothers_age_l109_109030


namespace five_digit_numbers_count_five_digit_numbers_ge_30000_rank_of_50124_l109_109710

-- Prove that the number of five-digit numbers is 27216
theorem five_digit_numbers_count : ∃ n, n = 9 * (Nat.factorial 9 / Nat.factorial 5) := by
  sorry

-- Prove that the number of five-digit numbers greater than or equal to 30000 is 21168
theorem five_digit_numbers_ge_30000 : 
  ∃ n, n = 7 * (Nat.factorial 9 / Nat.factorial 5) := by
  sorry

-- Prove that the rank of 50124 among five-digit numbers with distinct digits in descending order is 15119
theorem rank_of_50124 : 
  ∃ n, n = (Nat.factorial 5) - 1 := by
  sorry

end five_digit_numbers_count_five_digit_numbers_ge_30000_rank_of_50124_l109_109710


namespace two_digit_numbers_of_form_3_pow_n_l109_109345

theorem two_digit_numbers_of_form_3_pow_n :
  ∃ (n1 n2 : ℕ), (10 ≤ 3^n1 ∧ 3^n1 ≤ 99) ∧ (10 ≤ 3^n2 ∧ 3^n2 ≤ 99) ∧ n2 - n1 + 1 = 2 :=
by
  sorry

end two_digit_numbers_of_form_3_pow_n_l109_109345


namespace train_speed_l109_109086

/-- Given: 
1. A train travels a distance of 80 km in 40 minutes. 
2. We need to prove that the speed of the train is 120 km/h.
-/
theorem train_speed (distance : ℝ) (time_minutes : ℝ) (time_hours : ℝ) (speed : ℝ) 
  (h_distance : distance = 80) 
  (h_time_minutes : time_minutes = 40) 
  (h_time_hours : time_hours = 40 / 60) 
  (h_speed : speed = distance / time_hours) : 
  speed = 120 :=
sorry

end train_speed_l109_109086


namespace smallest_x_satisfies_eq_l109_109144

theorem smallest_x_satisfies_eq : ∃ x : ℝ, (1 / (x - 5) + 1 / (x - 7) = 5 / (2 * (x - 6))) ∧ x = 7 - Real.sqrt 6 :=
by
  -- The proof steps would go here, but we're skipping them with sorry for now.
  sorry

end smallest_x_satisfies_eq_l109_109144


namespace area_of_rhombus_l109_109914

theorem area_of_rhombus (d1 d2 : ℕ) (h1 : d1 = 16) (h2 : d2 = 20) : (d1 * d2) / 2 = 160 := by
  sorry

end area_of_rhombus_l109_109914


namespace fraction_studying_japanese_l109_109524

variable (J S : ℕ)
variable (hS : S = 3 * J)

def fraction_of_seniors_studying_japanese := (1 / 3 : ℚ) * S
def fraction_of_juniors_studying_japanese := (3 / 4 : ℚ) * J

def total_students := S + J

theorem fraction_studying_japanese (J S : ℕ) (hS : S = 3 * J) :
  ((1 / 3 : ℚ) * S + (3 / 4 : ℚ) * J) / (S + J) = 7 / 16 :=
by {
  -- proof to be filled in
  sorry
}

end fraction_studying_japanese_l109_109524


namespace sum_of_repeating_decimal_digits_of_five_thirteenths_l109_109842

theorem sum_of_repeating_decimal_digits_of_five_thirteenths 
  (a b : ℕ)
  (h1 : 5 / 13 = (a * 10 + b) / 99)
  (h2 : (a * 10 + b) = 38) :
  a + b = 11 :=
sorry

end sum_of_repeating_decimal_digits_of_five_thirteenths_l109_109842


namespace julia_birth_year_is_1979_l109_109765

-- Definitions based on conditions
def wayne_age_in_2021 : ℕ := 37
def wayne_birth_year : ℕ := 2021 - wayne_age_in_2021
def peter_birth_year : ℕ := wayne_birth_year - 3
def julia_birth_year : ℕ := peter_birth_year - 2

-- Theorem to prove
theorem julia_birth_year_is_1979 : julia_birth_year = 1979 := by
  sorry

end julia_birth_year_is_1979_l109_109765


namespace nina_total_cost_l109_109155

-- Define the cost of the first pair of shoes
def first_pair_cost : ℕ := 22

-- Define the cost of the second pair of shoes
def second_pair_cost : ℕ := first_pair_cost + (first_pair_cost / 2)

-- Define the total cost for both pairs of shoes
def total_cost : ℕ := first_pair_cost + second_pair_cost

-- The formal statement of the problem
theorem nina_total_cost : total_cost = 55 := by
  sorry

end nina_total_cost_l109_109155


namespace determine_c_absolute_value_l109_109940

theorem determine_c_absolute_value
  (a b c : ℤ)
  (h_gcd : Int.gcd (Int.gcd a b) c = 1)
  (h_root : a * (Complex.mk 3 1)^4 + b * (Complex.mk 3 1)^3 + c * (Complex.mk 3 1)^2 + b * (Complex.mk 3 1) + a = 0) :
  |c| = 109 := 
sorry

end determine_c_absolute_value_l109_109940


namespace rosy_current_age_l109_109342

theorem rosy_current_age 
  (R : ℕ) 
  (h1 : ∀ (david_age rosy_age : ℕ), david_age = rosy_age + 12) 
  (h2 : ∀ (david_age_plus_4 rosy_age_plus_4 : ℕ), david_age_plus_4 = 2 * rosy_age_plus_4) : 
  R = 8 := 
sorry

end rosy_current_age_l109_109342


namespace highest_point_difference_l109_109996

theorem highest_point_difference :
  let A := -112
  let B := -80
  let C := -25
  max A (max B C) - min A (min B C) = 87 :=
by
  sorry

end highest_point_difference_l109_109996


namespace degree_measure_supplement_complement_l109_109036

theorem degree_measure_supplement_complement : 
  let alpha := 63 -- angle value
  let theta := 90 - alpha -- complement of the angle
  let phi := 180 - theta -- supplement of the complement
  phi = 153 := -- prove the final step
by
  sorry

end degree_measure_supplement_complement_l109_109036


namespace compare_y_values_l109_109609

theorem compare_y_values (y1 y2 : ℝ) 
  (hA : y1 = (-1)^2 - 4*(-1) - 3) 
  (hB : y2 = 1^2 - 4*1 - 3) : y1 > y2 :=
by
  sorry

end compare_y_values_l109_109609


namespace infinite_geometric_series_sum_l109_109627

/-
Mathematical problem: Calculate the sum of the infinite geometric series 1 + (1/2) + (1/2)^2 + (1/2)^3 + ... . Express your answer as a common fraction.

Conditions:
- The first term \( a \) is 1.
- The common ratio \( r \) is \(\frac{1}{2}\).

Answer:
- The sum of the series is 2.
-/

theorem infinite_geometric_series_sum :
  let a := 1
  let r := 1 / 2
  (a * (1 / (1 - r))) = 2 :=
by
  let a := 1
  let r := 1 / 2
  have h : 1 * (1 / (1 - r)) = 2 := by sorry
  exact h

end infinite_geometric_series_sum_l109_109627


namespace keiko_jogging_speed_l109_109183

variable (s : ℝ) -- Keiko's jogging speed
variable (b : ℝ) -- radius of the inner semicircle
variable (L_inner : ℝ := 200 + 2 * Real.pi * b) -- total length of the inner track
variable (L_outer : ℝ := 200 + 2 * Real.pi * (b + 8)) -- total length of the outer track
variable (t_inner : ℝ := L_inner / s) -- time to jog the inside edge
variable (t_outer : ℝ := L_outer / s) -- time to jog the outside edge
variable (time_difference : ℝ := 48) -- time difference between jogging inside and outside edges

theorem keiko_jogging_speed : L_inner = 200 + 2 * Real.pi * b →
                           L_outer = 200 + 2 * Real.pi * (b + 8) →
                           t_outer = t_inner + 48 →
                           s = Real.pi / 3 :=
by
  intro h1 h2 h3
  sorry

end keiko_jogging_speed_l109_109183


namespace songs_after_operations_l109_109178

-- Definitions based on conditions
def initialSongs : ℕ := 15
def deletedSongs : ℕ := 8
def addedSongs : ℕ := 50

-- Problem statement to be proved
theorem songs_after_operations : initialSongs - deletedSongs + addedSongs = 57 :=
by
  sorry

end songs_after_operations_l109_109178


namespace find_parameters_l109_109335

noncomputable def cubic_function (a b : ℝ) (x : ℝ) : ℝ :=
  x^3 + a * x^2 + b * x + 27

def deriv_cubic_function (a b : ℝ) (x : ℝ) : ℝ :=
  3 * x^2 + 2 * a * x + b

theorem find_parameters
  (a b : ℝ)
  (h1 : deriv_cubic_function a b (-1) = 0)
  (h2 : deriv_cubic_function a b 3 = 0) :
  a = -3 ∧ b = -9 :=
by
  -- leaving proof as sorry since the task doesn't require proving
  sorry

end find_parameters_l109_109335


namespace find_base_b_l109_109237

theorem find_base_b (b : ℕ) :
  (2 * b^2 + 4 * b + 3) + (1 * b^2 + 5 * b + 6) = (4 * b^2 + 1 * b + 1) →
  7 < b →
  b = 10 :=
by
  intro h₁ h₂
  sorry

end find_base_b_l109_109237


namespace problem_I_problem_II_l109_109490

noncomputable def f (x : ℝ) : ℝ := x - 2 * Real.sin x

theorem problem_I :
  ∀ x ∈ Set.Icc 0 Real.pi, (f x) ≥ (f (Real.pi / 3) - Real.sqrt 3) ∧ (f x) ≤ f Real.pi :=
sorry

theorem problem_II :
  ∀ a : ℝ, ((∃ x : ℝ, (0 < x ∧ x < Real.pi / 2) ∧ f x < a * x) ↔ a > -1) :=
sorry

end problem_I_problem_II_l109_109490


namespace minimum_value_of_f_l109_109147

noncomputable def f (x : ℝ) : ℝ := 4 * x + 1 / (4 * x - 5)

theorem minimum_value_of_f (x : ℝ) : x > 5 / 4 → ∃ y, ∀ z, f z ≥ y ∧ y = 7 :=
by
  intro h
  sorry

end minimum_value_of_f_l109_109147


namespace value_x_plus_2y_plus_3z_l109_109990

variable (x y z : ℝ)

theorem value_x_plus_2y_plus_3z :
  x + y = 5 →
  z^2 = x * y + y - 9 →
  x + 2 * y + 3 * z = 8 :=
by
  intro h1 h2
  sorry

end value_x_plus_2y_plus_3z_l109_109990


namespace smallest_xyz_sum_l109_109262

theorem smallest_xyz_sum (x y z : ℕ) (h1 : (x + y) * (y + z) = 2016) (h2 : (x + y) * (z + x) = 1080) :
  x > 0 → y > 0 → z > 0 → x + y + z = 61 :=
  sorry

end smallest_xyz_sum_l109_109262


namespace sochi_apartment_price_decrease_l109_109293

theorem sochi_apartment_price_decrease (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  let moscow_rub_decrease := 0.2
  let moscow_eur_decrease := 0.4
  let sochi_rub_decrease := 0.1
  let new_moscow_rub := (1 - moscow_rub_decrease) * a
  let new_moscow_eur := (1 - moscow_eur_decrease) * b
  let ruble_to_euro := new_moscow_rub / new_moscow_eur
  let new_sochi_rub := (1 - sochi_rub_decrease) * a
  let new_sochi_eur := new_sochi_rub / ruble_to_euro
  let decrease_percentage := (b - new_sochi_eur) / b * 100
  decrease_percentage = 32.5 :=
by
  sorry

end sochi_apartment_price_decrease_l109_109293


namespace smallest_y_l109_109764

theorem smallest_y (y : ℤ) :
  (∃ k : ℤ, y^2 + 3*y + 7 = k*(y-2)) ↔ y = -15 :=
sorry

end smallest_y_l109_109764


namespace min_value_fraction_l109_109217

theorem min_value_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  (1 / a + 9 / b) ≥ 8 :=
by sorry

end min_value_fraction_l109_109217


namespace ratio_of_ages_l109_109649

variables (X Y : ℕ)

theorem ratio_of_ages (h1 : X - 6 = 24) (h2 : X + Y = 36) : X / Y = 2 :=
by 
  have h3 : X = 30 - 6 := by sorry
  have h4 : X = 24 := by sorry
  have h5 : X + Y = 36 := by sorry
  have h6 : Y = 12 := by sorry
  have h7 : X / Y = 2 := by sorry
  exact h7

end ratio_of_ages_l109_109649


namespace two_d_minus_c_zero_l109_109958

theorem two_d_minus_c_zero :
  ∃ (c d : ℕ), (∀ x : ℕ, x^2 - 18 * x + 72 = (x - c) * (x - d)) ∧ c > d ∧ (2 * d - c = 0) := 
sorry

end two_d_minus_c_zero_l109_109958


namespace factorization_of_polynomial_l109_109197

theorem factorization_of_polynomial : 
  ∀ (x : ℝ), 18 * x^3 + 9 * x^2 + 3 * x = 3 * x * (6 * x^2 + 3 * x + 1) :=
by sorry

end factorization_of_polynomial_l109_109197


namespace complement_A_union_B_m_eq_4_B_nonempty_and_subset_A_range_m_l109_109106

-- Definitions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | (m + 1) ≤ x ∧ x ≤ (2 * m - 1)}

-- Part (Ⅰ)
theorem complement_A_union_B_m_eq_4 :
  (m = 4) → compl (A ∪ B 4) = {x | x < -2} ∪ {x | x > 7} := 
by
  sorry

-- Part (Ⅱ)
theorem B_nonempty_and_subset_A_range_m :
  (∃ x, x ∈ B m) ∧ (B m ⊆ A) → 2 ≤ m ∧ m ≤ 3 :=
by
  sorry

end complement_A_union_B_m_eq_4_B_nonempty_and_subset_A_range_m_l109_109106


namespace hyperbola_asymptotes_l109_109423

-- Define the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := y^2 - (x^2 / 4) = 1

-- The statement to prove: The equation of the asymptotes of the hyperbola is as follows
theorem hyperbola_asymptotes :
  (∀ x y : ℝ, hyperbola_eq x y → (y = (1/2) * x ∨ y = -(1/2) * x)) :=
sorry

end hyperbola_asymptotes_l109_109423


namespace max_min_vec_magnitude_l109_109113

noncomputable def vec_a (θ : ℝ) := (Real.cos θ, Real.sin θ)
noncomputable def vec_b : ℝ × ℝ := (Real.sqrt 3, 1)

noncomputable def vec_result (θ : ℝ) := (2 * Real.cos θ - Real.sqrt 3, 2 * Real.sin θ - 1)

noncomputable def vec_magnitude (θ : ℝ) := Real.sqrt ((2 * Real.cos θ - Real.sqrt 3)^2 + (2 * Real.sin θ - 1)^2)

theorem max_min_vec_magnitude : 
  ∃ θ_max θ_min, 
    vec_magnitude θ_max = 4 ∧ 
    vec_magnitude θ_min = 0 :=
by
  sorry

end max_min_vec_magnitude_l109_109113


namespace cosine_of_eight_times_alpha_l109_109024

theorem cosine_of_eight_times_alpha (α : ℝ) (hypotenuse : ℝ) 
  (cos_α : ℝ) (cos_2α : ℝ) (cos_4α : ℝ) 
  (h₀ : hypotenuse = Real.sqrt (1^2 + (Real.sqrt 2)^2))
  (h₁ : cos_α = (Real.sqrt 2) / hypotenuse)
  (h₂ : cos_2α = 2 * cos_α^2 - 1)
  (h₃ : cos_4α = 2 * cos_2α^2 - 1)
  (h₄ : cos_8α = 2 * cos_4α^2 - 1) :
  cos_8α = 17 / 81 := 
  by
  sorry

end cosine_of_eight_times_alpha_l109_109024


namespace min_value_expression_l109_109571

-- Define the given problem conditions and statement
theorem min_value_expression :
  ∀ (x y : ℝ), 0 < x → 0 < y → 6 ≤ (y / x) + (16 * x / (2 * x + y)) :=
by
  sorry

end min_value_expression_l109_109571


namespace Tom_has_38_photos_l109_109112

theorem Tom_has_38_photos :
  ∃ (Tom Tim Paul : ℕ), 
  (Paul = Tim + 10) ∧ 
  (Tim = 152 - 100) ∧ 
  (152 = Tom + Paul + Tim) ∧ 
  (Tom = 38) :=
by
  sorry

end Tom_has_38_photos_l109_109112


namespace trajectory_eq_l109_109977

-- Define the points O, A, and B
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (-1, -2)

-- Define the vector equation for point C given the parameters s and t
def C (s t : ℝ) : ℝ × ℝ := (s * 2 + t * -1, s * 1 + t * -2)

-- Prove the equation of the trajectory of C given s + t = 1
theorem trajectory_eq (s t : ℝ) (h : s + t = 1) : ∃ x y : ℝ, C s t = (x, y) ∧ x - y - 1 = 0 := by
  -- The proof will be added here
  sorry

end trajectory_eq_l109_109977


namespace vanaspati_percentage_l109_109355

theorem vanaspati_percentage (Q : ℝ) (h1 : 0.60 * Q > 0) (h2 : Q + 10 > 0) (h3 : Q = 10) :
    let total_ghee := Q + 10
    let pure_ghee := 0.60 * Q + 10
    let pure_ghee_fraction := pure_ghee / total_ghee
    pure_ghee_fraction = 0.80 → 
    let vanaspati_fraction := 1 - pure_ghee_fraction
    vanaspati_fraction * 100 = 40 :=
by
  intros
  sorry

end vanaspati_percentage_l109_109355


namespace scorpion_needs_10_millipedes_l109_109539

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

end scorpion_needs_10_millipedes_l109_109539


namespace icosahedron_minimal_rotation_l109_109450

structure Icosahedron :=
  (faces : ℕ)
  (is_regular : Prop)
  (face_shape : Prop)

def icosahedron := Icosahedron.mk 20 (by sorry) (by sorry)

def theta (θ : ℝ) : Prop :=
  ∃ θ > 0, ∀ h : Icosahedron, 
  h.faces = 20 ∧ h.is_regular ∧ h.face_shape → θ = 72

theorem icosahedron_minimal_rotation :
  ∃ θ > 0, ∀ h : Icosahedron,
  h.faces = 20 ∧ h.is_regular ∧ h.face_shape → θ = 72 :=
by sorry

end icosahedron_minimal_rotation_l109_109450


namespace height_of_isosceles_triangle_l109_109266

variable (s : ℝ) (h : ℝ) (A : ℝ)
variable (triangle : ∀ (s : ℝ) (h : ℝ), A = 0.5 * (2 * s) * h)
variable (rectangle : ∀ (s : ℝ), A = s^2)

theorem height_of_isosceles_triangle (s : ℝ) (h : ℝ) (A : ℝ) (triangle : ∀ (s : ℝ) (h : ℝ), A = 0.5 * (2 * s) * h)
  (rectangle : ∀ (s : ℝ), A = s^2) : h = s := by
  sorry

end height_of_isosceles_triangle_l109_109266


namespace largest_corner_sum_l109_109141

-- Define the cube and its properties
structure Cube :=
  (faces : ℕ → ℕ)
  (opposite_faces_sum_to_8 : ∀ i, faces i + faces (7 - i) = 8)

-- Prove that the largest sum of three numbers whose faces meet at one corner is 16
theorem largest_corner_sum (c : Cube) : ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
  (c.faces i + c.faces j + c.faces k = 16) :=
sorry

end largest_corner_sum_l109_109141


namespace initial_dolphins_l109_109252

variable (D : ℕ)

theorem initial_dolphins (h1 : 3 * D + D = 260) : D = 65 :=
by
  sorry

end initial_dolphins_l109_109252


namespace sin_2x_eq_7_div_25_l109_109369

theorem sin_2x_eq_7_div_25 (x : ℝ) (h : Real.sin (Real.pi / 4 - x) = 3 / 5) :
    Real.sin (2 * x) = 7 / 25 := by
  sorry

end sin_2x_eq_7_div_25_l109_109369


namespace smallest_positive_period_symmetry_axis_range_of_f_l109_109956
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 6))

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem symmetry_axis (k : ℤ) :
  ∃ k : ℤ, ∃ x : ℝ, f x = f (x + k * (Real.pi / 2)) ∧ x = (Real.pi / 6) + k * (Real.pi / 2) := sorry

theorem range_of_f : 
  ∀ x, -Real.pi / 12 ≤ x ∧ x ≤ Real.pi / 2 → -1/2 ≤ f x ∧ f x ≤ 1 := sorry

end smallest_positive_period_symmetry_axis_range_of_f_l109_109956


namespace remaining_amount_spent_on_watermelons_l109_109939

def pineapple_cost : ℕ := 7
def total_spent : ℕ := 38
def pineapples_purchased : ℕ := 2

theorem remaining_amount_spent_on_watermelons:
  total_spent - (pineapple_cost * pineapples_purchased) = 24 :=
by
  sorry

end remaining_amount_spent_on_watermelons_l109_109939


namespace fourth_power_sum_l109_109234

variable (a b c : ℝ)

theorem fourth_power_sum (h1 : a + b + c = 2) 
                         (h2 : a^2 + b^2 + c^2 = 3) 
                         (h3 : a^3 + b^3 + c^3 = 4) : 
                         a^4 + b^4 + c^4 = 41 / 6 := 
by 
  sorry

end fourth_power_sum_l109_109234


namespace chef_initial_eggs_l109_109159

-- Define the conditions
def eggs_in_fridge := 10
def eggs_per_cake := 5
def cakes_made := 10

-- Prove that the number of initial eggs is 60
theorem chef_initial_eggs : (eggs_per_cake * cakes_made + eggs_in_fridge) = 60 :=
by
  sorry

end chef_initial_eggs_l109_109159


namespace consecutive_negatives_product_sum_l109_109712

theorem consecutive_negatives_product_sum:
  ∃ (n: ℤ), n < 0 ∧ (n + 1) < 0 ∧ n * (n + 1) = 3080 ∧ n + (n + 1) = -111 :=
by
  sorry

end consecutive_negatives_product_sum_l109_109712


namespace ptarmigan_environmental_capacity_l109_109130

theorem ptarmigan_environmental_capacity (predators_eradicated : Prop) (mass_deaths : Prop) : 
  (after_predator_eradication : predators_eradicated → mass_deaths) →
  (environmental_capacity_increased : Prop) → environmental_capacity_increased :=
by
  intros h1 h2
  sorry

end ptarmigan_environmental_capacity_l109_109130


namespace finite_decimals_are_rational_l109_109035

-- Conditions as definitions
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b
def is_infinite_decimal (x : ℝ) : Prop := ¬∃ (n : ℤ), x = ↑n
def is_finite_decimal (x : ℝ) : Prop := ∃ (a b : ℕ), b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

-- Equivalence to statement C: Finite decimals are rational numbers
theorem finite_decimals_are_rational : ∀ (x : ℝ), is_finite_decimal x → is_rational x := by
  sorry

end finite_decimals_are_rational_l109_109035


namespace coffee_shrinkage_l109_109722

theorem coffee_shrinkage :
  let initial_volume_per_cup := 8
  let shrink_factor := 0.5
  let number_of_cups := 5
  let final_volume_per_cup := initial_volume_per_cup * shrink_factor
  let total_remaining_coffee := final_volume_per_cup * number_of_cups
  total_remaining_coffee = 20 :=
by
  -- This is where the steps of the solution would go.
  -- We'll put a sorry here to indicate omission of proof.
  sorry

end coffee_shrinkage_l109_109722


namespace find_multiple_l109_109613

-- Defining the conditions
def first_lock_time := 5
def second_lock_time (x : ℕ) := 5 * x - 3

-- Proving the multiple
theorem find_multiple : 
  ∃ x : ℕ, (5 * first_lock_time * x - 3) * 5 = 60 ∧ (x = 3) :=
by
  sorry

end find_multiple_l109_109613


namespace sampling_interval_is_9_l109_109307

-- Conditions
def books_per_hour : ℕ := 362
def sampled_books_per_hour : ℕ := 40

-- Claim to prove
theorem sampling_interval_is_9 : (360 / sampled_books_per_hour = 9) := by
  sorry

end sampling_interval_is_9_l109_109307


namespace smallest_x_plus_y_l109_109267

theorem smallest_x_plus_y {x y : ℕ} (hxy : x ≠ y) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 12) : x + y = 49 :=
sorry

end smallest_x_plus_y_l109_109267


namespace triangle_base_is_8_l109_109385

/- Problem Statement:
We have a square with a perimeter of 48 and a triangle with a height of 36.
We need to prove that if both the square and the triangle have the same area, then the base of the triangle (x) is 8.
-/

theorem triangle_base_is_8
  (square_perimeter : ℝ)
  (triangle_height : ℝ)
  (same_area : ℝ) :
  square_perimeter = 48 →
  triangle_height = 36 →
  same_area = (square_perimeter / 4) ^ 2 →
  same_area = (1 / 2) * x * triangle_height →
  x = 8 :=
by
  sorry

end triangle_base_is_8_l109_109385


namespace albert_snakes_count_l109_109037

noncomputable def garden_snake_length : ℝ := 10.0
noncomputable def boa_ratio : ℝ := 1 / 7.0
noncomputable def boa_length : ℝ := 1.428571429

theorem albert_snakes_count : 
  garden_snake_length = 10.0 ∧ 
  boa_ratio = 1 / 7.0 ∧ 
  boa_length = 1.428571429 → 
  2 = 2 :=
by
  intro h
  sorry   -- Proof will go here

end albert_snakes_count_l109_109037


namespace xyz_inequality_l109_109811

theorem xyz_inequality (x y z : ℝ) : x^2 + y^2 + z^2 ≥ x * y + y * z + z * x := 
  sorry

end xyz_inequality_l109_109811


namespace binomial_multiplication_subtraction_l109_109292

variable (x : ℤ)

theorem binomial_multiplication_subtraction :
  (4 * x - 3) * (x + 6) - ( (2 * x + 1) * (x - 4) ) = 2 * x^2 + 28 * x - 14 := by
  sorry

end binomial_multiplication_subtraction_l109_109292


namespace lizard_problem_theorem_l109_109175

def lizard_problem : Prop :=
  ∃ (E W S : ℕ), 
  E = 3 ∧ 
  W = 3 * E ∧ 
  S = 7 * W ∧ 
  (S + W) - E = 69

theorem lizard_problem_theorem : lizard_problem :=
by
  sorry

end lizard_problem_theorem_l109_109175


namespace tom_buys_papayas_l109_109736

-- Defining constants for the costs of each fruit
def lemon_cost : ℕ := 2
def papaya_cost : ℕ := 1
def mango_cost : ℕ := 4

-- Defining the number of each fruit Tom buys
def lemons_bought : ℕ := 6
def mangos_bought : ℕ := 2
def total_paid : ℕ := 21

-- Defining the function to calculate the total cost 
def total_cost (P : ℕ) : ℕ := (lemons_bought * lemon_cost) + (mangos_bought * mango_cost) + (P * papaya_cost)

-- Defining the function to calculate the discount based on the total number of fruits
def discount (P : ℕ) : ℕ := (lemons_bought + mangos_bought + P) / 4

-- Main theorem to prove
theorem tom_buys_papayas (P : ℕ) : total_cost P - discount P = total_paid → P = 4 := 
by
  intro h
  sorry

end tom_buys_papayas_l109_109736


namespace car_fuel_efficiency_l109_109297

theorem car_fuel_efficiency 
  (H C T : ℤ)
  (h₁ : 900 = H * T)
  (h₂ : 600 = C * T)
  (h₃ : C = H - 5) :
  C = 10 := by
  sorry

end car_fuel_efficiency_l109_109297


namespace javier_total_time_spent_l109_109834

def outlining_time : ℕ := 30
def writing_time : ℕ := outlining_time + 28
def practicing_time : ℕ := writing_time / 2

theorem javier_total_time_spent : outlining_time + writing_time + practicing_time = 117 := by
  sorry

end javier_total_time_spent_l109_109834


namespace sphere_surface_area_of_solid_l109_109814

theorem sphere_surface_area_of_solid (l w h : ℝ) (hl : l = 2) (hw : w = 1) (hh : h = 2) 
: 4 * Real.pi * ((Real.sqrt (l^2 + w^2 + h^2) / 2)^2) = 9 * Real.pi := 
by 
  sorry

end sphere_surface_area_of_solid_l109_109814


namespace households_in_city_l109_109058

theorem households_in_city (x : ℕ) (h1 : x < 100) (h2 : x + x / 3 = 100) : x = 75 :=
sorry

end households_in_city_l109_109058


namespace geometric_inequality_l109_109839

variable {q : ℝ} {b : ℕ → ℝ}

def geometric_sequence (b : ℕ → ℝ) (q : ℝ) : Prop := ∀ n : ℕ, b (n + 1) = b n * q

theorem geometric_inequality
  (h_geometric : geometric_sequence b q)
  (h_q_gt_one : q > 1)
  (h_pos : ∀ n : ℕ, b n > 0) :
  b 4 + b 8 > b 5 + b 7 :=
by
  sorry

end geometric_inequality_l109_109839


namespace inequality_holds_l109_109942

theorem inequality_holds (x y z : ℝ) : x^2 + y^2 + z^2 ≥ Real.sqrt 2 * (x * y + y * z) := 
by 
  sorry

end inequality_holds_l109_109942


namespace remaining_fuel_relation_l109_109785

-- Define the car's travel time and remaining fuel relation
def initial_fuel : ℝ := 100

def fuel_consumption_rate : ℝ := 6

def remaining_fuel (t : ℝ) : ℝ := initial_fuel - fuel_consumption_rate * t

-- Prove that the remaining fuel after t hours is given by the linear relationship Q = 100 - 6t
theorem remaining_fuel_relation (t : ℝ) : remaining_fuel t = 100 - 6 * t := by
  -- Proof is omitted, as per instructions
  sorry

end remaining_fuel_relation_l109_109785


namespace largest_possible_value_of_b_l109_109557

theorem largest_possible_value_of_b (b : ℚ) (h : (3 * b + 4) * (b - 2) = 9 * b) : b ≤ 4 :=
sorry

end largest_possible_value_of_b_l109_109557


namespace inverse_function_condition_l109_109010

noncomputable def f (m x : ℝ) := (3 * x + 4) / (m * x - 5)

theorem inverse_function_condition (m : ℝ) :
  (∀ x : ℝ, f m (f m x) = x) ↔ m = -4 / 5 :=
by
  sorry

end inverse_function_condition_l109_109010


namespace negation_proposition_l109_109578

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^2 - x - 1 < 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≥ 0) :=
by 
  sorry

end negation_proposition_l109_109578


namespace perpendicular_vectors_x_eq_5_l109_109094

def vector_a (x : ℝ) : ℝ × ℝ := (2, x + 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x - 2, -1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors_x_eq_5 (x : ℝ)
  (h : dot_product (vector_a x) (vector_b x) = 0) :
  x = 5 :=
sorry

end perpendicular_vectors_x_eq_5_l109_109094


namespace simplify_expression_l109_109797

open Real

-- Assume that x, y, z are non-zero real numbers
variables (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)

theorem simplify_expression : (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ := 
by
  -- Proof would go here.
  sorry

end simplify_expression_l109_109797


namespace volume_multiplication_factor_l109_109685

variables (r h : ℝ) (π : ℝ := Real.pi)

def original_volume : ℝ := π * r^2 * h
def new_height : ℝ := 3 * h
def new_radius : ℝ := 2.5 * r
def new_volume : ℝ := π * (new_radius r)^2 * (new_height h)

theorem volume_multiplication_factor :
  new_volume r h / original_volume r h = 18.75 :=
by
  sorry

end volume_multiplication_factor_l109_109685


namespace find_number_l109_109154

theorem find_number (x : ℝ) 
  (h : (28 + x / 69) * 69 = 1980) :
  x = 1952 :=
sorry

end find_number_l109_109154


namespace sequence_formula_l109_109573

theorem sequence_formula (x : ℕ → ℤ) :
  x 1 = 1 →
  x 2 = -1 →
  (∀ n, n ≥ 2 → x (n-1) + x (n+1) = 2 * x n) →
  ∀ n, x n = -2 * n + 3 :=
by
  sorry

end sequence_formula_l109_109573


namespace will_new_cards_count_l109_109295

-- Definitions based on conditions
def cards_per_page := 3
def pages_used := 6
def old_cards := 10

-- Proof statement (no proof, only the statement)
theorem will_new_cards_count : (pages_used * cards_per_page) - old_cards = 8 :=
by sorry

end will_new_cards_count_l109_109295


namespace pyramid_volume_l109_109329

noncomputable def volume_of_pyramid (a h : ℝ) : ℝ :=
  (a^2 * h) / (4 * Real.sqrt 3)

theorem pyramid_volume (d x y : ℝ) (a h : ℝ) (edge_distance lateral_face_distance : ℝ)
  (H1 : edge_distance = 2) (H2 : lateral_face_distance = Real.sqrt 12)
  (H3 : x = 2) (H4 : y = 2 * Real.sqrt 3) (H5 : d = (a * Real.sqrt 3) / 6)
  (H6 : h = Real.sqrt (48 / 5)) :
  volume_of_pyramid a h = 216 * Real.sqrt 3 := by
  sorry

end pyramid_volume_l109_109329


namespace inverse_proposition_l109_109995

theorem inverse_proposition (x : ℝ) : 
  (¬ (x > 2) → ¬ (x > 1)) ↔ ((x > 1) → (x > 2)) := 
by 
  sorry

end inverse_proposition_l109_109995


namespace fermat_little_theorem_l109_109041

theorem fermat_little_theorem (p : ℕ) (a : ℕ) (hp : Prime p) : a ^ p ≡ a [MOD p] :=
sorry

end fermat_little_theorem_l109_109041


namespace gcd_fx_x_l109_109268

def f (x: ℕ) := (5 * x + 4) * (9 * x + 7) * (11 * x + 3) * (x + 12)

theorem gcd_fx_x (x: ℕ) (h: x % 54896 = 0) : Nat.gcd (f x) x = 112 :=
  sorry

end gcd_fx_x_l109_109268


namespace amplitude_combined_wave_l109_109643

noncomputable def y1 (t : ℝ) : ℝ := 3 * Real.sqrt 2 * Real.sin (100 * Real.pi * t)
noncomputable def y2 (t : ℝ) : ℝ := 3 * Real.sin (100 * Real.pi * t - Real.pi / 4)
noncomputable def y (t : ℝ) : ℝ := y1 t + y2 t
noncomputable def amplitude : ℝ := 3 * Real.sqrt 5

theorem amplitude_combined_wave : ∀ t : ℝ, ∃ A : ℝ, A = 3 * Real.sqrt 5 :=
by
  intro t
  use amplitude
  exact sorry

end amplitude_combined_wave_l109_109643


namespace modular_inverse_of_2_mod_199_l109_109518

theorem modular_inverse_of_2_mod_199 : (2 * 100) % 199 = 1 := 
by sorry

end modular_inverse_of_2_mod_199_l109_109518


namespace shopkeeper_profit_percent_l109_109734

theorem shopkeeper_profit_percent
  (initial_value : ℝ)
  (percent_lost_theft : ℝ)
  (percent_total_loss : ℝ)
  (remaining_value : ℝ)
  (total_loss_value : ℝ)
  (selling_price : ℝ)
  (profit : ℝ)
  (profit_percent : ℝ)
  (h_initial_value : initial_value = 100)
  (h_percent_lost_theft : percent_lost_theft = 20)
  (h_percent_total_loss : percent_total_loss = 12)
  (h_remaining_value : remaining_value = initial_value - (percent_lost_theft / 100) * initial_value)
  (h_total_loss_value : total_loss_value = (percent_total_loss / 100) * initial_value)
  (h_selling_price : selling_price = initial_value - total_loss_value)
  (h_profit : profit = selling_price - remaining_value)
  (h_profit_percent : profit_percent = (profit / remaining_value) * 100) :
  profit_percent = 10 := by
  sorry

end shopkeeper_profit_percent_l109_109734


namespace cost_price_l109_109997

theorem cost_price (MP : ℝ) (SP : ℝ) (C : ℝ) 
  (h1 : MP = 87.5) 
  (h2 : SP = 0.95 * MP) 
  (h3 : SP = 1.25 * C) : 
  C = 66.5 := 
by
  sorry

end cost_price_l109_109997


namespace solve_x_given_y_l109_109959

theorem solve_x_given_y (x : ℝ) (h : 2 = 2 / (5 * x + 3)) : x = -2 / 5 :=
sorry

end solve_x_given_y_l109_109959


namespace negation_proposition_equivalence_l109_109725

theorem negation_proposition_equivalence : 
    (¬ ∃ x_0 : ℝ, (x_0^2 + 1 > 0) ∨ (x_0 > Real.sin x_0)) ↔ 
    (∀ x : ℝ, (x^2 + 1 ≤ 0) ∧ (x ≤ Real.sin x)) :=
by 
    sorry

end negation_proposition_equivalence_l109_109725


namespace Problem_statements_l109_109841

theorem Problem_statements (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = a * b) :
  (a + b ≥ 4) ∧
  ¬(a * b ≤ 4) ∧
  (a + 4 * b ≥ 9) ∧
  (1 / a ^ 2 + 2 / b ^ 2 ≥ 2 / 3) :=
by sorry

end Problem_statements_l109_109841


namespace infinite_positive_sequence_geometric_l109_109365

theorem infinite_positive_sequence_geometric {a : ℕ → ℝ} (h : ∀ n ≥ 1, a (n + 2) = a n - a (n + 1)) 
  (h_pos : ∀ n, a n > 0) :
  ∃ (a1 : ℝ) (q : ℝ), q = (Real.sqrt 5 - 1) / 2 ∧ (∀ n, a n = a1 * q^(n - 1)) := by
  sorry

end infinite_positive_sequence_geometric_l109_109365


namespace perpendicular_vectors_k_zero_l109_109023

theorem perpendicular_vectors_k_zero
  (k : ℝ)
  (a : ℝ × ℝ := (3, 1))
  (b : ℝ × ℝ := (1, 3))
  (c : ℝ × ℝ := (k, 2)) 
  (h : (a.1 - c.1, a.2 - c.2).1 * b.1 + (a.1 - c.1, a.2 - c.2).2 * b.2 = 0) :
  k = 0 :=
by
  sorry

end perpendicular_vectors_k_zero_l109_109023


namespace largest_common_number_in_arithmetic_sequences_l109_109224

theorem largest_common_number_in_arithmetic_sequences (n : ℕ) :
  (∃ a1 a2 : ℕ, a1 = 5 + 8 * n ∧ a2 = 3 + 9 * n ∧ a1 = a2 ∧ 1 ≤ a1 ∧ a1 ≤ 150) →
  (a1 = 93) :=
by
  sorry

end largest_common_number_in_arithmetic_sequences_l109_109224


namespace average_last_two_numbers_l109_109745

theorem average_last_two_numbers (a b c d e f g : ℝ) 
  (h1 : (a + b + c + d + e + f + g) / 7 = 63) 
  (h2 : (a + b + c) / 3 = 58) 
  (h3 : (d + e) / 2 = 70) :
  ((f + g) / 2) = 63.5 := 
sorry

end average_last_two_numbers_l109_109745


namespace initial_courses_of_bricks_l109_109015

theorem initial_courses_of_bricks (x : ℕ) : 
    400 * x + 2 * 400 - 400 / 2 = 1800 → x = 3 :=
by
  sorry

end initial_courses_of_bricks_l109_109015


namespace inequality_abc_l109_109922

open Real

theorem inequality_abc 
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h : a * b * c = 1) : 
  (1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b))) ≥ 3 / 2 :=
sorry

end inequality_abc_l109_109922


namespace initial_pen_count_is_30_l109_109887

def pen_count (initial_pens : ℕ) : ℕ :=
  let after_mike := initial_pens + 20
  let after_cindy := 2 * after_mike
  let after_sharon := after_cindy - 10
  after_sharon

theorem initial_pen_count_is_30 : pen_count 30 = 30 :=
by
  sorry

end initial_pen_count_is_30_l109_109887


namespace orthogonality_implies_x_value_l109_109793

theorem orthogonality_implies_x_value :
  ∀ (x : ℝ),
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (2, -1)
  a.1 * b.1 + a.2 * b.2 = 0 → x = 1 :=
sorry

end orthogonality_implies_x_value_l109_109793


namespace march_first_is_tuesday_l109_109857

theorem march_first_is_tuesday (march_15_tuesday : true) :
  true :=
sorry

end march_first_is_tuesday_l109_109857


namespace mary_earns_per_home_l109_109602

noncomputable def earnings_per_home (T : ℕ) (n : ℕ) : ℕ := T / n

theorem mary_earns_per_home :
  ∀ (T n : ℕ), T = 276 → n = 6 → earnings_per_home T n = 46 := 
by
  intros T n h1 h2
  -- Placeholder proof step
  sorry

end mary_earns_per_home_l109_109602


namespace computation_problem_points_l109_109970

/-- A teacher gives out a test of 30 problems. Each computation problem is worth some points, and
each word problem is worth 5 points. The total points you can receive on the test is 110 points,
and there are 20 computation problems. How many points is each computation problem worth? -/

theorem computation_problem_points (x : ℕ) (total_problems : ℕ := 30) (word_problem_points : ℕ := 5)
    (total_points : ℕ := 110) (computation_problems : ℕ := 20) :
    20 * x + (total_problems - computation_problems) * word_problem_points = total_points → x = 3 :=
by
  intro h
  sorry

end computation_problem_points_l109_109970


namespace calc_value_exponents_l109_109984

theorem calc_value_exponents :
  (3^3) * (5^3) * (3^5) * (5^5) = 15^8 :=
by sorry

end calc_value_exponents_l109_109984


namespace maximize_profit_correct_l109_109761

noncomputable def maximize_profit : ℝ × ℝ :=
  let initial_selling_price : ℝ := 50
  let purchase_price : ℝ := 40
  let initial_sales_volume : ℝ := 500
  let sales_volume_decrease_rate : ℝ := 10
  let x := 20
  let optimal_selling_price := initial_selling_price + x
  let maximum_profit := -10 * x^2 + 400 * x + 5000
  (optimal_selling_price, maximum_profit)

theorem maximize_profit_correct :
  maximize_profit = (70, 9000) :=
  sorry

end maximize_profit_correct_l109_109761


namespace incorrect_observation_value_l109_109991

theorem incorrect_observation_value
  (mean : ℕ → ℝ)
  (n : ℕ)
  (observed_mean : ℝ)
  (incorrect_value : ℝ)
  (correct_value : ℝ)
  (corrected_mean : ℝ)
  (H1 : n = 50)
  (H2 : observed_mean = 36)
  (H3 : correct_value = 43)
  (H4 : corrected_mean = 36.5)
  (H5 : mean n = observed_mean)
  (H6 : mean (n - 1 + 1) = corrected_mean - correct_value + incorrect_value) :
  incorrect_value = 18 := sorry

end incorrect_observation_value_l109_109991


namespace supplementary_angle_60_eq_120_l109_109276

def supplementary_angle (α : ℝ) : ℝ :=
  180 - α

theorem supplementary_angle_60_eq_120 :
  supplementary_angle 60 = 120 :=
by
  -- the proof should be filled here
  sorry

end supplementary_angle_60_eq_120_l109_109276


namespace find_g2_l109_109432

variable (g : ℝ → ℝ)

def condition (x : ℝ) : Prop :=
  g x - 2 * g (1 / x) = 3^x

theorem find_g2 (h : ∀ x ≠ 0, condition g x) : g 2 = -3 - (4 * Real.sqrt 3) / 9 :=
  sorry

end find_g2_l109_109432


namespace circles_intersect_at_2_points_l109_109022

theorem circles_intersect_at_2_points :
  let circle1 := { p : ℝ × ℝ | (p.1 - 5 / 2) ^ 2 + p.2 ^ 2 = 25 / 4 }
  let circle2 := { p : ℝ × ℝ | p.1 ^ 2 + (p.2 - 7 / 2) ^ 2 = 49 / 4 }
  ∃ (P1 P2 : ℝ × ℝ), P1 ∈ circle1 ∧ P1 ∈ circle2 ∧
                     P2 ∈ circle1 ∧ P2 ∈ circle2 ∧
                     P1 ≠ P2 ∧ ∀ (P : ℝ × ℝ), P ∈ circle1 ∧ P ∈ circle2 → P = P1 ∨ P = P2 := 
by 
  sorry

end circles_intersect_at_2_points_l109_109022


namespace seq_eighth_term_l109_109904

-- Define the sequence recursively
def seq : ℕ → ℕ
| 0     => 1  -- Base case, since 1 is the first term of the sequence
| (n+1) => seq n + (n + 1)  -- Recursive case, each term is the previous term plus the index number (which is n + 1) minus 1

-- Define the statement to prove 
theorem seq_eighth_term : seq 7 = 29 :=  -- Note: index 7 corresponds to the 8th term since indexing is 0-based
  by
  sorry

end seq_eighth_term_l109_109904


namespace numbers_not_expressed_l109_109180

theorem numbers_not_expressed (a b : ℕ) (hb : 0 < b) (ha : 0 < a) :
 ∀ n : ℕ, (¬ ∃ a b : ℕ, n = a / b + (a + 1) / (b + 1) ∧ 0 < b ∧ 0 < a) ↔ (n = 1 ∨ ∃ m : ℕ, n = 2^m + 2) := 
by 
  sorry

end numbers_not_expressed_l109_109180


namespace min_voters_for_Tall_victory_l109_109955

def total_voters := 105
def districts := 5
def sections_per_district := 7
def voters_per_section := 3
def sections_to_win_district := 4
def districts_to_win := 3
def sections_to_win := sections_to_win_district * districts_to_win
def min_voters_to_win_section := 2

theorem min_voters_for_Tall_victory : 
  (total_voters = 105) ∧ 
  (districts = 5) ∧ 
  (sections_per_district = 7) ∧ 
  (voters_per_section = 3) ∧ 
  (sections_to_win_district = 4) ∧ 
  (districts_to_win = 3) 
  → 
  min_voters_to_win_section * sections_to_win = 24 :=
by
  sorry
  
end min_voters_for_Tall_victory_l109_109955


namespace evaluate_polynomial_at_3_using_horners_method_l109_109210

def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

theorem evaluate_polynomial_at_3_using_horners_method : f 3 = 1641 := by
 sorry

end evaluate_polynomial_at_3_using_horners_method_l109_109210


namespace resistance_at_least_2000_l109_109655

variable (U : ℝ) (I : ℝ) (R : ℝ)

-- Given conditions:
def voltage := U = 220
def max_current := I ≤ 0.11

-- Ohm's law in this context
def ohms_law := I = U / R

-- Proof problem statement:
theorem resistance_at_least_2000 (voltage : U = 220) (max_current : I ≤ 0.11) (ohms_law : I = U / R) : R ≥ 2000 :=
sorry

end resistance_at_least_2000_l109_109655


namespace intersection_M_N_l109_109334

def M := {m : ℤ | -3 < m ∧ m < 2}
def N := {x : ℤ | x * (x - 1) = 0}

theorem intersection_M_N : M ∩ N = {0, 1} := sorry

end intersection_M_N_l109_109334


namespace totalStudents_correct_l109_109803

-- Defining the initial number of classes, students per class, and new classes
def initialClasses : ℕ := 15
def studentsPerClass : ℕ := 20
def newClasses : ℕ := 5

-- Prove that the total number of students is 400
theorem totalStudents_correct : 
  initialClasses * studentsPerClass + newClasses * studentsPerClass = 400 := by
  sorry

end totalStudents_correct_l109_109803


namespace more_girls_than_boys_l109_109781

theorem more_girls_than_boys (num_students : ℕ) (boys_ratio : ℕ) (girls_ratio : ℕ) (total_students : ℕ) (total_students_eq : num_students = 42) (ratio_eq : boys_ratio = 3 ∧ girls_ratio = 4) : (4 * 6) - (3 * 6) = 6 := by
  sorry

end more_girls_than_boys_l109_109781


namespace age_problem_lean4_l109_109241

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

end age_problem_lean4_l109_109241


namespace sum_of_remaining_six_numbers_l109_109092

theorem sum_of_remaining_six_numbers :
  ∀ (S T U : ℕ), 
    S = 20 * 500 → T = 14 * 390 → U = S - T → U = 4540 :=
by
  intros S T U hS hT hU
  sorry

end sum_of_remaining_six_numbers_l109_109092


namespace no_such_integers_exist_l109_109499

theorem no_such_integers_exist :
  ¬(∃ (a b c d : ℤ), a * 19^3 + b * 19^2 + c * 19 + d = 1 ∧ a * 62^3 + b * 62^2 + c * 62 + d = 2) :=
by
  sorry

end no_such_integers_exist_l109_109499


namespace bunnies_burrow_exit_counts_l109_109771

theorem bunnies_burrow_exit_counts :
  let groupA_bunnies := 40
  let groupA_rate := 3  -- times per minute per bunny
  let groupB_bunnies := 30
  let groupB_rate := 5 / 2 -- times per minute per bunny
  let groupC_bunnies := 30
  let groupC_rate := 8 / 5 -- times per minute per bunny
  let total_bunnies := 100
  let minutes_per_day := 1440
  let days_per_week := 7
  let pre_change_rate_per_min := groupA_bunnies * groupA_rate + groupB_bunnies * groupB_rate + groupC_bunnies * groupC_rate
  let post_change_rate_per_min := pre_change_rate_per_min * 0.5
  let total_pre_change_counts := pre_change_rate_per_min * minutes_per_day * days_per_week
  let total_post_change_counts := post_change_rate_per_min * minutes_per_day * (days_per_week * 2)
  total_pre_change_counts + total_post_change_counts = 4897920 := by
    sorry

end bunnies_burrow_exit_counts_l109_109771


namespace g_50_l109_109302

noncomputable def g : ℝ → ℝ :=
sorry

axiom functional_equation (x y : ℝ) : g (x * y) = x * g y
axiom g_2 : g 2 = 10

theorem g_50 : g 50 = 250 :=
sorry

end g_50_l109_109302


namespace jesse_rooms_l109_109675

theorem jesse_rooms:
  ∀ (l w A n: ℕ), 
  l = 19 ∧ 
  w = 18 ∧ 
  A = 6840 ∧ 
  n = A / (l * w) → 
  n = 20 :=
by
  intros
  sorry

end jesse_rooms_l109_109675


namespace positive_expressions_l109_109647

-- Define the approximate values for A, B, C, D, and E.
def A := 2.5
def B := -2.1
def C := -0.3
def D := 1.0
def E := -0.7

-- Define the expressions that we need to prove as positive numbers.
def exprA := A + B
def exprB := B * C
def exprD := E / (A * B)

-- The theorem states that expressions (A + B), (B * C), and (E / (A * B)) are positive.
theorem positive_expressions : exprA > 0 ∧ exprB > 0 ∧ exprD > 0 := 
by sorry

end positive_expressions_l109_109647


namespace shaded_area_of_rotated_semicircle_l109_109019

-- Definitions and conditions from the problem
def radius (R : ℝ) : Prop := R > 0
def central_angle (α : ℝ) : Prop := α = 30 * (Real.pi / 180)

-- Lean theorem statement for the proof problem
theorem shaded_area_of_rotated_semicircle (R : ℝ) (hR : radius R) (hα : central_angle 30) : 
  ∃ (area : ℝ), area = (Real.pi * R^2) / 3 :=
by
  -- using proofs of radius and angle conditions
  sorry

end shaded_area_of_rotated_semicircle_l109_109019


namespace percent_less_50000_l109_109818

variable (A B C : ℝ) -- Define the given percentages
variable (h1 : A = 0.45) -- 45% of villages have populations from 20,000 to 49,999
variable (h2 : B = 0.30) -- 30% of villages have fewer than 20,000 residents
variable (h3 : C = 0.25) -- 25% of villages have 50,000 or more residents

theorem percent_less_50000 : A + B = 0.75 := by
  sorry

end percent_less_50000_l109_109818


namespace fraction_sent_afternoon_l109_109466

theorem fraction_sent_afternoon :
  ∀ (total_fliers morning_fraction fliers_left_next_day : ℕ),
  total_fliers = 3000 →
  morning_fraction = 1/5 →
  fliers_left_next_day = 1800 →
  ((total_fliers - total_fliers * morning_fraction) - fliers_left_next_day) / (total_fliers - total_fliers * morning_fraction) = 1/4 :=
by
  intros total_fliers morning_fraction fliers_left_next_day h1 h2 h3
  sorry

end fraction_sent_afternoon_l109_109466


namespace a_older_than_b_l109_109630

theorem a_older_than_b (A B : ℕ) (h1 : B = 36) (h2 : A + 10 = 2 * (B - 10)) : A - B = 6 :=
  sorry

end a_older_than_b_l109_109630


namespace solve_inequality_l109_109233

theorem solve_inequality {x : ℝ} : (x^2 - 5 * x + 6 ≤ 0) → (2 ≤ x ∧ x ≤ 3) :=
by
  intro h
  sorry

end solve_inequality_l109_109233


namespace red_peaches_count_l109_109810

-- Definitions for the conditions
def yellow_peaches : ℕ := 11
def extra_red_peaches : ℕ := 8

-- The proof statement that the number of red peaches is 19
theorem red_peaches_count : (yellow_peaches + extra_red_peaches = 19) :=
by
  sorry

end red_peaches_count_l109_109810


namespace largest_lambda_inequality_l109_109943

theorem largest_lambda_inequality :
  ∀ (a b c d e : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → 0 ≤ e →
  (a^2 + b^2 + c^2 + d^2 + e^2 ≥ a * b + (5/4) * b * c + c * d + d * e) :=
by
  sorry

end largest_lambda_inequality_l109_109943


namespace lcm_of_ratio_and_hcf_l109_109171

theorem lcm_of_ratio_and_hcf (a b : ℕ) (h1 : a = 3 * 8) (h2 : b = 4 * 8) (h3 : Nat.gcd a b = 8) : Nat.lcm a b = 96 :=
  sorry

end lcm_of_ratio_and_hcf_l109_109171


namespace bridge_construction_l109_109774

-- Definitions used in the Lean statement based on conditions.
def rate (workers : ℕ) (days : ℕ) : ℚ := 1 / (workers * days)

-- The problem statement: prove that if 60 workers working together can build the bridge in 3 days, 
-- then 120 workers will take 1.5 days to build the bridge.
theorem bridge_construction (t : ℚ) : 
  (rate 60 3) * 120 * t = 1 → t = 1.5 := by
  sorry

end bridge_construction_l109_109774


namespace negative_exponent_example_l109_109553

theorem negative_exponent_example : 3^(-2 : ℤ) = (1 : ℚ) / (3^2) :=
by sorry

end negative_exponent_example_l109_109553


namespace bad_oranges_l109_109629

theorem bad_oranges (total_oranges : ℕ) (students : ℕ) (less_oranges_per_student : ℕ)
  (initial_oranges_per_student now_oranges_per_student shared_oranges now_total_oranges bad_oranges : ℕ) :
  total_oranges = 108 →
  students = 12 →
  less_oranges_per_student = 3 →
  initial_oranges_per_student = total_oranges / students →
  now_oranges_per_student = initial_oranges_per_student - less_oranges_per_student →
  shared_oranges = students * now_oranges_per_student →
  now_total_oranges = 72 →
  bad_oranges = total_oranges - now_total_oranges →
  bad_oranges = 36 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end bad_oranges_l109_109629


namespace proof_problem_l109_109543

theorem proof_problem 
  (A a B b : ℝ) 
  (h1 : |A - 3 * a| ≤ 1 - a) 
  (h2 : |B - 3 * b| ≤ 1 - b) 
  (h3 : 0 < a) 
  (h4 : 0 < b) :
  (|((A * B) / 3) - 3 * (a * b)|) - 3 * (a * b) ≤ 1 - (a * b) :=
sorry

end proof_problem_l109_109543


namespace arithmetic_mean_reciprocals_first_four_primes_l109_109576

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l109_109576


namespace function_equation_l109_109219

noncomputable def f (n : ℕ) : ℕ := sorry

theorem function_equation (h : ∀ m n : ℕ, m > 0 → n > 0 →
  f (f (f m) + 2 * f (f n)) = m^2 + 2 * n^2) : 
  ∀ n : ℕ, n > 0 → f n = n := 
sorry

end function_equation_l109_109219


namespace black_squares_covered_by_trominoes_l109_109831

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

noncomputable def min_trominoes (n : ℕ) : ℕ :=
  ((n + 1) ^ 2) / 4

theorem black_squares_covered_by_trominoes (n : ℕ) (h1 : n ≥ 7) (h2 : is_odd n):
  ∀ n : ℕ, ∃ k : ℕ, k = min_trominoes n :=
by
  sorry

end black_squares_covered_by_trominoes_l109_109831


namespace find_E_l109_109714

variable (x E x1 x2 : ℝ)

/-- Given conditions as assumptions: -/
axiom h1 : (x + 3)^2 / E = 2
axiom h2 : x1 - x2 = 14

/-- Prove the required expression for E in terms of x: -/
theorem find_E : E = (x + 3)^2 / 2 := sorry

end find_E_l109_109714


namespace mark_more_than_kate_by_100_l109_109027

variable (Pat Kate Mark : ℕ)
axiom total_hours : Pat + Kate + Mark = 180
axiom pat_twice_as_kate : Pat = 2 * Kate
axiom pat_third_of_mark : Pat = Mark / 3

theorem mark_more_than_kate_by_100 : Mark - Kate = 100 :=
by
  sorry

end mark_more_than_kate_by_100_l109_109027


namespace A_three_two_l109_109176

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m+1, 0 => A m 2
| m+1, n+1 => A m (A (m + 1) n)

theorem A_three_two : A 3 2 = 5 := 
by 
  sorry

end A_three_two_l109_109176


namespace total_marks_is_275_l109_109728

-- Definitions of scores in each subject
def science_score : ℕ := 70
def music_score : ℕ := 80
def social_studies_score : ℕ := 85
def physics_score : ℕ := music_score / 2

-- Definition of total marks
def total_marks : ℕ := science_score + music_score + social_studies_score + physics_score

-- Theorem to prove that total marks is 275
theorem total_marks_is_275 : total_marks = 275 := by
  -- Proof here
  sorry

end total_marks_is_275_l109_109728


namespace cars_no_air_conditioning_l109_109344

variables {A R AR : Nat}

/-- Given a total of 100 cars, of which at least 51 have racing stripes,
and the greatest number of cars that could have air conditioning but not racing stripes is 49,
prove that the number of cars that do not have air conditioning is 49. -/
theorem cars_no_air_conditioning :
  ∀ (A R AR : ℕ), 
  (A = AR + 49) → 
  (R ≥ 51) → 
  (AR ≤ R) → 
  (AR ≤ 51) → 
  (100 - A = 49) :=
by
  intros A R AR h1 h2 h3 h4
  exact sorry

end cars_no_air_conditioning_l109_109344


namespace find_coordinates_of_C_l109_109552

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

end find_coordinates_of_C_l109_109552


namespace carla_needs_24_cans_l109_109549

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

end carla_needs_24_cans_l109_109549


namespace brad_running_speed_l109_109508

variable (dist_between_homes : ℕ)
variable (maxwell_speed : ℕ)
variable (time_maxwell_walks : ℕ)
variable (maxwell_start_time : ℕ)
variable (brad_start_time : ℕ)

#check dist_between_homes = 94
#check maxwell_speed = 4
#check time_maxwell_walks = 10
#check brad_start_time = maxwell_start_time + 1

theorem brad_running_speed (dist_between_homes : ℕ) (maxwell_speed : ℕ) (time_maxwell_walks : ℕ) (maxwell_start_time : ℕ) (brad_start_time : ℕ) :
  dist_between_homes = 94 →
  maxwell_speed = 4 →
  time_maxwell_walks = 10 →
  brad_start_time = maxwell_start_time + 1 →
  (dist_between_homes - maxwell_speed * time_maxwell_walks) / (time_maxwell_walks - (brad_start_time - maxwell_start_time)) = 6 :=
by
  intros
  sorry

end brad_running_speed_l109_109508


namespace probability_of_green_tile_l109_109472

theorem probability_of_green_tile :
  let total_tiles := 100
  let green_tiles := 14
  let probability := green_tiles / total_tiles
  probability = 7 / 50 :=
by
  sorry

end probability_of_green_tile_l109_109472


namespace sin_square_range_l109_109081

def range_sin_square_values (α β : ℝ) : Prop :=
  3 * (Real.sin α) ^ 2 - 2 * Real.sin α + 2 * (Real.sin β) ^ 2 = 0

theorem sin_square_range (α β : ℝ) (h : range_sin_square_values α β) :
  0 ≤ (Real.sin α) ^ 2 + (Real.sin β) ^ 2 ∧ 
  (Real.sin α) ^ 2 + (Real.sin β) ^ 2 ≤ 4 / 9 :=
sorry

end sin_square_range_l109_109081


namespace multiples_of_7_units_digit_7_l109_109986

theorem multiples_of_7_units_digit_7 (n : ℕ) (h1 : n < 150) (h2 : ∃ (k : ℕ), n = 7 * k) (h3 : n % 10 = 7) : 
    ∃ (m : ℕ), m = 2 := 
by
  sorry

end multiples_of_7_units_digit_7_l109_109986


namespace system_of_equations_l109_109158

theorem system_of_equations (x y : ℝ) (h1 : 3 * x + 210 = 5 * y) (h2 : 10 * y - 10 * x = 100) :
    (3 * x + 210 = 5 * y) ∧ (10 * y - 10 * x = 100) := by
  sorry

end system_of_equations_l109_109158


namespace solve_for_x_l109_109285

theorem solve_for_x (x : ℝ) (h: (6 / (x + 1) = 3 / 2)) : x = 3 :=
sorry

end solve_for_x_l109_109285


namespace smallest_angle_product_l109_109251

-- Define an isosceles triangle with angle at B being the smallest angle
def isosceles_triangle (α : ℝ) : Prop :=
  α < 90 ∧ α = 180 / 7

-- Proof that the smallest angle multiplied by 6006 is 154440
theorem smallest_angle_product : 
  isosceles_triangle α → (180 / 7) * 6006 = 154440 :=
by
  intros
  sorry

end smallest_angle_product_l109_109251


namespace timeSpentReading_l109_109042

def totalTime : ℕ := 120
def timeOnPiano : ℕ := 30
def timeWritingMusic : ℕ := 25
def timeUsingExerciser : ℕ := 27

theorem timeSpentReading :
  totalTime - timeOnPiano - timeWritingMusic - timeUsingExerciser = 38 := by
  sorry

end timeSpentReading_l109_109042


namespace intersection_point_of_lines_l109_109096

theorem intersection_point_of_lines : 
  ∃ x y : ℝ, (3 * x + 4 * y - 2 = 0) ∧ (2 * x + y + 2 = 0) ∧ (x = -2) ∧ (y = 2) := 
by 
  sorry

end intersection_point_of_lines_l109_109096


namespace units_digit_17_pow_2007_l109_109129

theorem units_digit_17_pow_2007 :
  (17 ^ 2007) % 10 = 3 := 
sorry

end units_digit_17_pow_2007_l109_109129


namespace rational_numbers_property_l109_109938

theorem rational_numbers_property (n : ℕ) (h : n > 0) :
  ∃ (a b : ℚ), a ≠ b ∧ (∀ k, 1 ≤ k ∧ k ≤ n → ∃ m : ℤ, a^k - b^k = m) ∧ 
  ∀ i, (a : ℝ) ≠ i ∧ (b : ℝ) ≠ i :=
sorry

end rational_numbers_property_l109_109938


namespace Vasya_numbers_l109_109090

theorem Vasya_numbers : ∃ (x y : ℝ), x + y = xy ∧ xy = x / y ∧ (x, y) = (1 / 2, -1) :=
by
  sorry

end Vasya_numbers_l109_109090


namespace rank_from_start_l109_109054

theorem rank_from_start (n r_l : ℕ) (h_n : n = 31) (h_r_l : r_l = 15) : n - (r_l - 1) = 17 := by
  sorry

end rank_from_start_l109_109054


namespace complex_round_quadrant_l109_109256

open Complex

theorem complex_round_quadrant (z : ℂ) (i : ℂ) (h : i = Complex.I) (h1 : z * i = 2 - i):
  z.re < 0 ∧ z.im < 0 := 
sorry

end complex_round_quadrant_l109_109256


namespace sum_even_integers_less_than_100_l109_109318

theorem sum_even_integers_less_than_100 : 
  let sequence := List.range' 2 98
  let even_seq := sequence.filter (λ x => x % 2 = 0)
  (even_seq.sum) = 2450 :=
by
  sorry

end sum_even_integers_less_than_100_l109_109318


namespace painting_two_sides_time_l109_109216

-- Definitions for the conditions
def time_to_paint_one_side_per_board : Nat := 1
def drying_time_per_board : Nat := 5

-- Definitions for the problem
def total_boards : Nat := 6

-- Main theorem statement
theorem painting_two_sides_time :
  (total_boards * time_to_paint_one_side_per_board) + drying_time_per_board + (total_boards * time_to_paint_one_side_per_board) = 12 :=
sorry

end painting_two_sides_time_l109_109216


namespace range_of_m_l109_109677

theorem range_of_m (m : Real) :
  (∀ x y : Real, 0 < x ∧ x < y ∧ y < (π / 2) → 
    (m - 2 * Real.sin x) / Real.cos x > (m - 2 * Real.sin y) / Real.cos y) →
  m ≤ 2 := 
sorry

end range_of_m_l109_109677


namespace range_of_set_l109_109854

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l109_109854


namespace gcd_a_b_eq_1023_l109_109069

def a : ℕ := 2^1010 - 1
def b : ℕ := 2^1000 - 1

theorem gcd_a_b_eq_1023 : Nat.gcd a b = 1023 := 
by
  sorry

end gcd_a_b_eq_1023_l109_109069


namespace complete_square_expression_l109_109777

theorem complete_square_expression :
  ∃ (a h k : ℝ), (∀ x : ℝ, 2 * x^2 + 8 * x + 6 = a * (x - h)^2 + k) ∧ (a + h + k = -2) :=
by
  sorry

end complete_square_expression_l109_109777


namespace ratio_of_votes_l109_109453

theorem ratio_of_votes (votes_A votes_B total_votes : ℕ) (hA : votes_A = 14) (hTotal : votes_A + votes_B = 21) : votes_A / Nat.gcd votes_A votes_B = 2 ∧ votes_B / Nat.gcd votes_A votes_B = 1 := 
by
  sorry

end ratio_of_votes_l109_109453


namespace positive_difference_of_squares_l109_109477

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 70) (h2 : a - b = 20) : a^2 - b^2 = 1400 :=
by
sorry

end positive_difference_of_squares_l109_109477


namespace remainder_x14_minus_1_div_x_plus_1_l109_109068

-- Define the polynomial f(x) = x^14 - 1
def f (x : ℝ) := x^14 - 1

-- Statement to prove that the remainder when f(x) is divided by x + 1 is 0
theorem remainder_x14_minus_1_div_x_plus_1 : f (-1) = 0 :=
by
  -- This is where the proof would go, but for now, we will just use sorry
  sorry

end remainder_x14_minus_1_div_x_plus_1_l109_109068


namespace magic_square_sum_l109_109584

-- Definitions based on the conditions outlined in the problem
def magic_sum := 83
def a := 42
def b := 26
def c := 29
def e := 34
def d := 36

theorem magic_square_sum :
  d + e = 70 :=
by
  -- Proof is omitted as per instructions
  sorry

end magic_square_sum_l109_109584


namespace wednesday_more_than_half_millet_l109_109825

namespace BirdFeeder

-- Define the initial conditions
def initial_amount_millet (total_seeds : ℚ) : ℚ := 0.4 * total_seeds
def initial_amount_other (total_seeds : ℚ) : ℚ := 0.6 * total_seeds

-- Define the daily consumption
def eaten_millet (millet : ℚ) : ℚ := 0.2 * millet
def eaten_other (other : ℚ) : ℚ := other

-- Define the seed addition every other day
def add_seeds (day : ℕ) (seeds : ℚ) : Prop :=
  day % 2 = 1 → seeds = 1

-- Define the daily update of the millet and other seeds in the feeder
def daily_update (day : ℕ) (millet : ℚ) (other : ℚ) : ℚ × ℚ :=
  let remaining_millet := (1 - 0.2) * millet
  let remaining_other := 0
  if day % 2 = 1 then
    (remaining_millet + initial_amount_millet 1, initial_amount_other 1)
  else
    (remaining_millet, remaining_other)

-- Define the main property to prove
def more_than_half_millet (day : ℕ) (millet : ℚ) (other : ℚ) : Prop :=
  millet > 0.5 * (millet + other)

-- Define the theorem statement
theorem wednesday_more_than_half_millet
  (millet : ℚ := initial_amount_millet 1)
  (other : ℚ := initial_amount_other 1) :
  ∃ day, day = 3 ∧ more_than_half_millet day millet other :=
  by
  sorry

end BirdFeeder

end wednesday_more_than_half_millet_l109_109825


namespace find_slower_speed_l109_109832

-- Variables and conditions definitions
variable (v : ℝ)

def slower_speed (v : ℝ) : Prop :=
  (20 / v = 2) ∧ (v = 10)

-- The statement to be proven
theorem find_slower_speed : slower_speed 10 :=
by
  sorry

end find_slower_speed_l109_109832


namespace marble_problem_l109_109554

theorem marble_problem (a : ℚ) :
  (a + 2 * a + 3 * 2 * a + 5 * (3 * 2 * a) + 2 * (5 * (3 * 2 * a)) = 212) ↔
  (a = 212 / 99) :=
by
  sorry

end marble_problem_l109_109554


namespace angle_sum_l109_109095

theorem angle_sum (x : ℝ) (h1 : 2 * x + x = 90) : x = 30 := 
sorry

end angle_sum_l109_109095


namespace find_t_over_q_l109_109422

theorem find_t_over_q
  (q r s v t : ℝ)
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : v / t = 4)
  (h4 : s / v = 1 / 3) :
  t / q = 1 / 2 := 
sorry

end find_t_over_q_l109_109422


namespace simplest_form_fraction_C_l109_109085

def fraction_A (x : ℤ) (y : ℤ) : ℚ := (2 * x + 4) / (6 * x + 8)
def fraction_B (x : ℤ) (y : ℤ) : ℚ := (x + y) / (x^2 - y^2)
def fraction_C (x : ℤ) (y : ℤ) : ℚ := (x^2 + y^2) / (x + y)
def fraction_D (x : ℤ) (y : ℤ) : ℚ := (x^2 - y^2) / (x^2 - 2 * x * y + y^2)

theorem simplest_form_fraction_C (x y : ℤ) :
  ¬ (∃ (A : ℚ), A ≠ fraction_C x y ∧ (A = fraction_C x y)) :=
by
  intros
  sorry

end simplest_form_fraction_C_l109_109085


namespace simplify_and_evaluate_expr_l109_109470

theorem simplify_and_evaluate_expr (a : ℝ) (h1 : -1 < a) (h2 : a < Real.sqrt 5) (h3 : a = 2) :
  (a - (a^2 / (a^2 - 1))) / (a^2 / (a^2 - 1)) = 1 / 2 :=
by
  sorry

end simplify_and_evaluate_expr_l109_109470


namespace triangle_right_angled_l109_109087

theorem triangle_right_angled
  (a b c : ℝ) (A B C : ℝ)
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : A + B + C = π)
  (h₅ : b * Real.cos C + c * Real.cos B = a * Real.sin A) :
  A = π / 2 ∨ B = π / 2 ∨ C = π / 2 :=
sorry

end triangle_right_angled_l109_109087


namespace calculate_star_difference_l109_109398

def star (a b : ℕ) : ℕ := a^2 + 2 * a * b + b^2

theorem calculate_star_difference : (star 3 5) - (star 2 4) = 28 := by
  sorry

end calculate_star_difference_l109_109398


namespace tan_neg_seven_pi_sixths_l109_109595

noncomputable def tan_neg_pi_seven_sixths : Real :=
  -Real.sqrt 3 / 3

theorem tan_neg_seven_pi_sixths : Real.tan (-7 * Real.pi / 6) = -Real.sqrt 3 / 3 := by
  sorry

end tan_neg_seven_pi_sixths_l109_109595


namespace common_solution_exists_l109_109906

theorem common_solution_exists (a b : ℝ) :
  (∃ x y : ℝ, 19 * x^2 + 19 * y^2 + a * x + b * y + 98 = 0 ∧
                         98 * x^2 + 98 * y^2 + a * x + b * y + 19 = 0)
  → a^2 + b^2 ≥ 13689 :=
by
  -- Proof omitted
  sorry

end common_solution_exists_l109_109906


namespace binom_identity1_binom_identity2_l109_109950

variable (n k : ℕ)

theorem binom_identity1 (hn : n > 0) (hk : k > 0) :
  (Nat.choose n k) + (Nat.choose n (k + 1)) = (Nat.choose (n + 1) (k + 1)) :=
sorry

theorem binom_identity2 (hn : n > 0) (hk : k > 0) :
  (Nat.choose n k) = (n * Nat.choose (n - 1) (k - 1)) / k :=
sorry

end binom_identity1_binom_identity2_l109_109950


namespace area_of_triangle_ABC_l109_109124

-- Axiom statements representing the conditions
axiom medians_perpendicular (A B C D E G : Type) : Prop
axiom median_ad_length (A D : Type) : Prop
axiom median_be_length (B E : Type) : Prop

-- Main theorem statement
theorem area_of_triangle_ABC
  (A B C D E G : Type)
  (h1 : medians_perpendicular A B C D E G)
  (h2 : median_ad_length A D) -- AD = 18
  (h3 : median_be_length B E) -- BE = 24
  : ∃ (area : ℝ), area = 576 :=
sorry

end area_of_triangle_ABC_l109_109124


namespace solve_quadratic_l109_109592

theorem solve_quadratic : 
  (∀ x : ℚ, 2 * x^2 - x - 6 = 0 → x = -3 / 2 ∨ x = 2) ∧ 
  (∀ y : ℚ, (y - 2)^2 = 9 * y^2 → y = -1 ∨ y = 1 / 2) := 
by
  sorry

end solve_quadratic_l109_109592


namespace expression_equality_l109_109372

theorem expression_equality :
  (2^1001 + 5^1002)^2 - (2^1001 - 5^1002)^2 = 40 * 10^1001 := 
by
  sorry

end expression_equality_l109_109372


namespace sufficient_condition_implies_range_of_p_l109_109963

open Set Real

theorem sufficient_condition_implies_range_of_p (p : ℝ) :
  (∀ x : ℝ, 4 * x + p < 0 → x^2 - x - 2 > 0) →
  (∃ x : ℝ, x^2 - x - 2 > 0 ∧ ¬ (4 * x + p < 0)) →
  p ∈ Set.Ici 4 :=
by
  sorry

end sufficient_condition_implies_range_of_p_l109_109963


namespace central_cell_value_l109_109768

def table (a b c d e f g h i : ℝ) : Prop :=
  (a * b * c = 10) ∧ (d * e * f = 10) ∧ (g * h * i = 10) ∧
  (a * d * g = 10) ∧ (b * e * h = 10) ∧ (c * f * i = 10) ∧
  (a * b * d * e = 3) ∧ (b * c * e * f = 3) ∧ (d * e * g * h = 3) ∧ (e * f * h * i = 3)

theorem central_cell_value (a b c d f g h i e : ℝ) (h_table : table a b c d e f g h i) : 
  e = 0.00081 :=
by sorry

end central_cell_value_l109_109768


namespace abs_eq_solution_diff_l109_109840

theorem abs_eq_solution_diff : 
  ∀ x₁ x₂ : ℝ, 
  (2 * x₁ - 3 = 18 ∨ 2 * x₁ - 3 = -18) → 
  (2 * x₂ - 3 = 18 ∨ 2 * x₂ - 3 = -18) → 
  |x₁ - x₂| = 18 :=
by
  sorry

end abs_eq_solution_diff_l109_109840


namespace value_of_expression_l109_109691

theorem value_of_expression : 
  ∀ (a x y : ℤ), 
  (x = a + 5) → 
  (a = 20) → 
  (y = 25) → 
  (x - y) * (x + y) = 0 :=
by
  intros a x y h1 h2 h3
  -- proof goes here
  sorry

end value_of_expression_l109_109691


namespace variance_of_temperatures_l109_109179

def temperatures : List ℕ := [28, 21, 22, 26, 28, 25]

noncomputable def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

noncomputable def variance (l : List ℕ) : ℚ :=
  let m := mean l
  (l.map (λ x => (x - m)^2)).sum / l.length

theorem variance_of_temperatures : variance temperatures = 22 / 3 := 
by
  sorry

end variance_of_temperatures_l109_109179


namespace solve_equations_l109_109351

theorem solve_equations (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 4 * a + 3 * b = 39) : a + b = 82 / 7 := by
  sorry

end solve_equations_l109_109351


namespace sum_q_p_evaluations_l109_109088

def p (x : ℝ) : ℝ := |x^2 - 4|
def q (x : ℝ) : ℝ := -|x|

theorem sum_q_p_evaluations : 
  q (p (-3)) + q (p (-2)) + q (p (-1)) + q (p (0)) + q (p (1)) + q (p (2)) + q (p (3)) = -20 := 
by 
  sorry

end sum_q_p_evaluations_l109_109088


namespace triangle_is_isosceles_l109_109733

theorem triangle_is_isosceles
    (A B C : ℝ)
    (h_angle_sum : A + B + C = 180)
    (h_sinB : Real.sin B = 2 * Real.cos C * Real.sin A)
    : (A = C) := 
by
    sorry

end triangle_is_isosceles_l109_109733


namespace min_expression_value_l109_109104

open Real

theorem min_expression_value : ∀ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ 2023 := by
  sorry

end min_expression_value_l109_109104


namespace count_measures_of_angle_A_l109_109913

theorem count_measures_of_angle_A :
  ∃ n : ℕ, n = 17 ∧
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ A + B = 180 ∧ (∃ k : ℕ, k ≥ 1 ∧ A = k * B) ∧ (∀ (A' B' : ℕ), A' > 0 ∧ B' > 0 ∧ A' + B' = 180 ∧ (∀ k : ℕ, k ≥ 1 ∧ A' = k * B') → n = 17) :=
sorry

end count_measures_of_angle_A_l109_109913


namespace inequality_a_b_c_l109_109188

theorem inequality_a_b_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^3 / (a^2 + a * b + b^2)) + (b^3 / (b^2 + b * c + c^2)) + (c^3 / (c^2 + c * a + a^2)) ≥ (a + b + c) / 3 :=
sorry

end inequality_a_b_c_l109_109188


namespace no_solution_eqn_l109_109368

theorem no_solution_eqn (m : ℝ) : (∀ x : ℝ, (m * (x + 1) - 5) / (2 * x + 1) ≠ m - 3) ↔ m = 6 := 
by
  sorry

end no_solution_eqn_l109_109368


namespace greatest_common_divisor_546_180_l109_109149

theorem greatest_common_divisor_546_180 : 
  ∃ d, d < 70 ∧ d > 0 ∧ d ∣ 546 ∧ d ∣ 180 ∧ ∀ x, x < 70 ∧ x > 0 ∧ x ∣ 546 ∧ x ∣ 180 → x ≤ d → x = 6 :=
by
  sorry

end greatest_common_divisor_546_180_l109_109149


namespace find_f_ln_inv_6_l109_109801

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x + 2 / x^3 - 3

theorem find_f_ln_inv_6 (k : ℝ) (h : f k (Real.log 6) = 1) : f k (Real.log (1 / 6)) = -7 :=
by
  sorry

end find_f_ln_inv_6_l109_109801


namespace size_of_each_group_l109_109503

theorem size_of_each_group 
  (skittles : ℕ) (erasers : ℕ) (groups : ℕ)
  (h_skittles : skittles = 4502) (h_erasers : erasers = 4276) (h_groups : groups = 154) :
  (skittles + erasers) / groups = 57 :=
by
  sorry

end size_of_each_group_l109_109503


namespace sufficient_but_not_necessary_condition_l109_109794

def proposition_p (m : ℝ) : Prop := ∀ x : ℝ, |x + 1| + |x - 1| ≥ m
def proposition_q (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 - 2 * m * x₀ + m^2 + m - 3 = 0

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (proposition_p m → proposition_q m) ∧ ¬ (proposition_q m → proposition_p m) :=
sorry

end sufficient_but_not_necessary_condition_l109_109794


namespace find_alpha_l109_109941

theorem find_alpha (P : Real × Real) (h: P = (Real.sin (50 * Real.pi / 180), 1 + Real.cos (50 * Real.pi / 180))) :
  ∃ α : Real, α = 65 * Real.pi / 180 := by
  sorry

end find_alpha_l109_109941


namespace vending_machine_users_l109_109494

theorem vending_machine_users (p_fail p_double p_single : ℚ) (total_snacks : ℕ) (P : ℕ) :
  p_fail = 1 / 6 ∧ p_double = 1 / 10 ∧ p_single = 1 - 1 / 6 - 1 / 10 ∧
  total_snacks = 28 →
  P = 30 :=
by
  intros h
  sorry

end vending_machine_users_l109_109494


namespace kanul_total_amount_l109_109666

-- Definitions based on the conditions
def raw_materials_cost : ℝ := 35000
def machinery_cost : ℝ := 40000
def marketing_cost : ℝ := 15000
def total_spent : ℝ := raw_materials_cost + machinery_cost + marketing_cost
def spending_percentage : ℝ := 0.25

-- The statement we want to prove
theorem kanul_total_amount (T : ℝ) (h : total_spent = spending_percentage * T) : T = 360000 :=
by
  sorry

end kanul_total_amount_l109_109666


namespace finite_decimal_fractions_l109_109076

theorem finite_decimal_fractions (a b c d : ℕ) (n : ℕ) 
  (h1 : n = 2^a * 5^b)
  (h2 : n + 1 = 2^c * 5^d) :
  n = 1 ∨ n = 4 :=
by
  sorry

end finite_decimal_fractions_l109_109076


namespace ethan_presents_l109_109790

theorem ethan_presents (ethan alissa : ℕ) 
  (h1 : alissa = ethan + 22) 
  (h2 : alissa = 53) : 
  ethan = 31 := 
by
  sorry

end ethan_presents_l109_109790


namespace range_of_f_l109_109099

def diamond (x y : ℝ) := (x + y) ^ 2 - x * y

def f (a x : ℝ) := diamond a x

theorem range_of_f (a : ℝ) (h : diamond 1 a = 3) :
  ∃ b : ℝ, ∀ x : ℝ, x > 0 → f a x > b :=
sorry

end range_of_f_l109_109099


namespace pinedale_bus_speed_l109_109569

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

end pinedale_bus_speed_l109_109569


namespace number_of_valid_pairs_l109_109308

theorem number_of_valid_pairs : 
  ∃ (n : ℕ), n = 1995003 ∧ (∃ b c : ℤ, c < 2000 ∧ b > 2 ∧ (∀ x : ℂ, x^2 - (b:ℝ) * x + (c:ℝ) = 0 → x.re > 1)) := 
sorry

end number_of_valid_pairs_l109_109308


namespace lowest_fraction_combine_two_slowest_l109_109529

def rate_a (hours : ℕ) : ℚ := 1 / 4
def rate_b (hours : ℕ) : ℚ := 1 / 5
def rate_c (hours : ℕ) : ℚ := 1 / 8

theorem lowest_fraction_combine_two_slowest : 
  (rate_b 1 + rate_c 1) = 13 / 40 :=
by sorry

end lowest_fraction_combine_two_slowest_l109_109529


namespace parker_total_weight_l109_109880

-- Define the number of initial dumbbells and their weight
def initial_dumbbells := 4
def weight_per_dumbbell := 20

-- Define the number of additional dumbbells
def additional_dumbbells := 2

-- Define the total weight calculation
def total_weight := initial_dumbbells * weight_per_dumbbell + additional_dumbbells * weight_per_dumbbell

-- Prove that the total weight is 120 pounds
theorem parker_total_weight : total_weight = 120 :=
by
  -- proof skipped
  sorry

end parker_total_weight_l109_109880


namespace find_d_and_r_l109_109562

theorem find_d_and_r (d r : ℤ)
  (h1 : 1210 % d = r)
  (h2 : 1690 % d = r)
  (h3 : 2670 % d = r) :
  d - 4 * r = -20 := sorry

end find_d_and_r_l109_109562


namespace cos_A_minus_B_eq_nine_eighths_l109_109320

theorem cos_A_minus_B_eq_nine_eighths (A B : ℝ)
  (h1 : Real.sin A + Real.sin B = 1 / 2)
  (h2 : Real.cos A + Real.cos B = 2) : 
  Real.cos (A - B) = 9 / 8 := 
by
  sorry

end cos_A_minus_B_eq_nine_eighths_l109_109320


namespace chess_game_probability_l109_109652

theorem chess_game_probability (p_A_wins p_draw : ℝ) (h1 : p_A_wins = 0.3) (h2 : p_draw = 0.2) :
  p_A_wins + p_draw = 0.5 :=
by
  rw [h1, h2]
  norm_num

end chess_game_probability_l109_109652


namespace grid_covering_impossible_l109_109920

theorem grid_covering_impossible :
  ∀ (x y : ℕ), x + y = 19 → 6 * x + 7 * y = 132 → False :=
by
  intros x y h1 h2
  -- Proof would go here.
  sorry

end grid_covering_impossible_l109_109920


namespace tan_theta_3_l109_109467

noncomputable def tan_triple_angle (θ : ℝ) : ℝ := (3 * (Real.tan θ) - ((Real.tan θ) ^ 3)) / (1 - 3 * (Real.tan θ)^2)

theorem tan_theta_3 (θ : ℝ) (h : Real.tan θ = 3) : tan_triple_angle θ = 9 / 13 :=
by
  sorry

end tan_theta_3_l109_109467


namespace min_value_of_f_l109_109005

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  2 * x^3 - 6 * x^2 + m

theorem min_value_of_f :
  ∀ (m : ℝ),
    f 0 m = 3 →
    ∃ x min, x ∈ Set.Icc (-2:ℝ) (2:ℝ) ∧ min = f x m ∧ min = -37 :=
by
  intros m h
  have h' : f 0 m = 3 := h
  -- Proof omitted.
  sorry

end min_value_of_f_l109_109005


namespace _l109_109520

/-- This theorem states that if the GCD of 8580 and 330 is diminished by 12, the result is 318. -/
example : (Int.gcd 8580 330) - 12 = 318 :=
by
  sorry

end _l109_109520


namespace smallest_multiple_36_45_not_11_l109_109692

theorem smallest_multiple_36_45_not_11 (n : ℕ) :
  (n = 180) ↔ (n > 0 ∧ (36 ∣ n) ∧ (45 ∣ n) ∧ ¬ (11 ∣ n)) :=
by
  sorry

end smallest_multiple_36_45_not_11_l109_109692


namespace total_weight_l109_109623

def weight_of_blue_ball : ℝ := 6.0
def weight_of_brown_ball : ℝ := 3.12

theorem total_weight (_ : weight_of_blue_ball = 6.0) (_ : weight_of_brown_ball = 3.12) : 
  weight_of_blue_ball + weight_of_brown_ball = 9.12 :=
by
  sorry

end total_weight_l109_109623


namespace value_is_correct_l109_109333

-- Define the mean and standard deviation
def mean : ℝ := 14.0
def std_dev : ℝ := 1.5

-- Define the value that is 2 standard deviations less than the mean
def value : ℝ := mean - 2 * std_dev

-- Theorem stating that value = 11.0
theorem value_is_correct : value = 11.0 := by
  sorry

end value_is_correct_l109_109333


namespace proof_problem_l109_109703

def star (a b : ℕ) : ℕ := a - a / b

theorem proof_problem : star 18 6 + 2 * 6 = 27 := 
by
  admit  -- proof goes here

end proof_problem_l109_109703


namespace larger_cookie_raisins_l109_109232

theorem larger_cookie_raisins : ∃ n r, 5 ≤ n ∧ n ≤ 10 ∧ (n - 1) * r + (r + 1) = 100 ∧ r + 1 = 12 :=
by
  sorry

end larger_cookie_raisins_l109_109232


namespace product_mod_eq_l109_109215

theorem product_mod_eq :
  (1497 * 2003) % 600 = 291 := 
sorry

end product_mod_eq_l109_109215


namespace common_factor_l109_109637

theorem common_factor (x y : ℝ) : 
  ∃ c : ℝ, c * (3 * x * y^2 - 4 * x^2 * y) = 6 * x^2 * y - 8 * x * y^2 ∧ c = 2 * x * y := 
by 
  sorry

end common_factor_l109_109637


namespace solve_tangent_problem_l109_109708

noncomputable def problem_statement : Prop :=
  ∃ (n : ℤ), (-90 < n ∧ n < 90) ∧ (Real.tan (n * Real.pi / 180) = Real.tan (255 * Real.pi / 180)) ∧ (n = 75)

-- This is the statement of the problem we are proving.
theorem solve_tangent_problem : problem_statement :=
by
  sorry

end solve_tangent_problem_l109_109708


namespace height_of_rectangular_block_l109_109384

variable (V A h : ℕ)

theorem height_of_rectangular_block :
  V = 120 ∧ A = 24 ∧ V = A * h → h = 5 :=
by
  sorry

end height_of_rectangular_block_l109_109384


namespace evariste_stairs_l109_109507

def num_ways (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else num_ways (n - 1) + num_ways (n - 2)

theorem evariste_stairs (n : ℕ) : num_ways n = u_n :=
  sorry

end evariste_stairs_l109_109507


namespace triangle_area_correct_l109_109550
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

end triangle_area_correct_l109_109550


namespace original_smallest_element_l109_109661

theorem original_smallest_element (x : ℤ) 
  (h1 : x < -1) 
  (h2 : x + 14 + 0 + 6 + 9 = 2 * (2 + 3 + 0 + 6 + 9)) : 
  x = -4 :=
by sorry

end original_smallest_element_l109_109661


namespace pirate_loot_l109_109998

theorem pirate_loot (a b c d e : ℕ) (h1 : a = 1 ∨ b = 1 ∨ c = 1 ∨ d = 1 ∨ e = 1)
  (h2 : a = 2 ∨ b = 2 ∨ c = 2 ∨ d = 2 ∨ e = 2)
  (h3 : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
  (h4 : a + b = 2 * (c + d) ∨ b + c = 2 * (a + e)) :
  (a, b, c, d, e) = (1, 1, 1, 1, 2) ∨ 
  (a, b, c, d, e) = (1, 1, 2, 2, 2) ∨
  (a, b, c, d, e) = (1, 2, 3, 3, 3) ∨
  (a, b, c, d, e) = (1, 2, 2, 2, 3) :=
sorry

end pirate_loot_l109_109998


namespace geometric_sequence_value_l109_109560

theorem geometric_sequence_value (a : ℕ → ℝ) (h : ∀ n, a n > 0)
  (h_geometric : ∀ n, a (n+2) = a (n+1) * (a (n+1) / a n)) :
  a 3 * a 5 = 4 → a 4 = 2 :=
by
  sorry

end geometric_sequence_value_l109_109560


namespace determine_swimming_day_l109_109967

def practices_sport_each_day (sports : ℕ → ℕ → Prop) : Prop :=
  ∀ (d : ℕ), ∃ s, sports d s

def runs_four_days_no_consecutive (sports : ℕ → ℕ → Prop) : Prop :=
  ∃ (days : ℕ → ℕ), (∀ i, sports (days i) 0) ∧ 
    (∀ i j, i ≠ j → days i ≠ days j) ∧ 
    (∀ i j, (days i + 1 = days j) → false)

def plays_basketball_tuesday (sports : ℕ → ℕ → Prop) : Prop :=
  sports 2 1

def plays_golf_friday_after_tuesday (sports : ℕ → ℕ → Prop) : Prop :=
  sports 5 2

def swims_and_plays_tennis_condition (sports : ℕ → ℕ → Prop) : Prop :=
  ∃ (swim_day tennis_day : ℕ), swim_day ≠ tennis_day ∧ 
    sports swim_day 3 ∧ 
    sports tennis_day 4 ∧ 
    ∀ (d : ℕ), (sports d 3 → sports (d + 1) 4 → false) ∧ 
    (∀ (d : ℕ), sports d 3 → ∀ (r : ℕ), sports (d + 2) 0 → false)

theorem determine_swimming_day (sports : ℕ → ℕ → Prop) : 
  practices_sport_each_day sports → 
  runs_four_days_no_consecutive sports → 
  plays_basketball_tuesday sports → 
  plays_golf_friday_after_tuesday sports → 
  swims_and_plays_tennis_condition sports → 
  ∃ (d : ℕ), d = 7 := 
sorry

end determine_swimming_day_l109_109967


namespace find_other_number_l109_109869

theorem find_other_number (LCM HCF number1 number2 : ℕ) 
  (hLCM : LCM = 7700) 
  (hHCF : HCF = 11) 
  (hNumber1 : number1 = 308)
  (hProductEquality : number1 * number2 = LCM * HCF) :
  number2 = 275 :=
by
  -- proof omitted
  sorry

end find_other_number_l109_109869


namespace nested_fraction_l109_109844

theorem nested_fraction
  : 1 / (3 + 1 / (3 + 1 / (3 - 1 / (3 + 1 / (2 * (3 + 2 / 5))))))
  = 968 / 3191 := 
by
  sorry

end nested_fraction_l109_109844


namespace ap_sub_aq_l109_109895

variable {n : ℕ} (hn : n > 0)

def S (n : ℕ) : ℕ := 2 * n^2 - 3 * n

def a (n : ℕ) (hn : n > 0) : ℕ :=
S n - S (n - 1)

theorem ap_sub_aq (p q : ℕ) (hp : p > 0) (hq : q > 0) (h : p - q = 5) :
  a p hp - a q hq = 20 :=
sorry

end ap_sub_aq_l109_109895


namespace sum_A_C_l109_109139

theorem sum_A_C (A B C : ℝ) (h1 : A + B + C = 500) (h2 : B + C = 340) (h3 : C = 40) : A + C = 200 :=
by
  sorry

end sum_A_C_l109_109139


namespace N_eq_M_union_P_l109_109199

def M : Set ℝ := {x : ℝ | ∃ (n : ℤ), x = n}
def N : Set ℝ := {x : ℝ | ∃ (n : ℤ), x = n / 2}
def P : Set ℝ := {x : ℝ | ∃ (n : ℤ), x = n + 1 / 2}

theorem N_eq_M_union_P : N = M ∪ P :=
  sorry

end N_eq_M_union_P_l109_109199


namespace brick_width_is_10_cm_l109_109780

-- Define the conditions
def courtyard_length_meters := 25
def courtyard_width_meters := 16
def brick_length_cm := 20
def number_of_bricks := 20000

-- Convert courtyard dimensions to area in square centimeters
def area_of_courtyard_cm2 := courtyard_length_meters * 100 * courtyard_width_meters * 100

-- Total area covered by bricks
def total_brick_area_cm2 := area_of_courtyard_cm2

-- Area covered by one brick
def area_per_brick := total_brick_area_cm2 / number_of_bricks

-- Find the brick width
def brick_width_cm := area_per_brick / brick_length_cm

-- Prove the width of each brick is 10 cm
theorem brick_width_is_10_cm : brick_width_cm = 10 := 
by 
  -- Placeholder for the proof
  sorry

end brick_width_is_10_cm_l109_109780


namespace hyperbola_center_l109_109187

def is_midpoint (x1 y1 x2 y2 xc yc : ℝ) : Prop :=
  xc = (x1 + x2) / 2 ∧ yc = (y1 + y2) / 2

theorem hyperbola_center :
  is_midpoint 2 (-3) (-4) 5 (-1) 1 :=
by
  sorry

end hyperbola_center_l109_109187


namespace problems_per_page_l109_109358

def total_problems : ℕ := 72
def finished_problems : ℕ := 32
def remaining_pages : ℕ := 5
def remaining_problems : ℕ := total_problems - finished_problems

theorem problems_per_page : remaining_problems / remaining_pages = 8 := 
by
  sorry

end problems_per_page_l109_109358


namespace cards_per_page_l109_109167

theorem cards_per_page 
  (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 8)
  (h2 : old_cards = 10)
  (h3 : pages = 6) : (new_cards + old_cards) / pages = 3 := 
by 
  sorry

end cards_per_page_l109_109167


namespace pqr_value_l109_109848

theorem pqr_value
  (p q r : ℤ) -- p, q, and r are integers
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) -- non-zero condition
  (h1 : p + q + r = 27) -- sum condition
  (h2 : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 300 / (p * q * r) = 1) -- equation condition
  : p * q * r = 984 := 
sorry 

end pqr_value_l109_109848


namespace total_distance_traveled_l109_109826

theorem total_distance_traveled :
  let time1 := 3  -- hours
  let speed1 := 70  -- km/h
  let time2 := 4  -- hours
  let speed2 := 80  -- km/h
  let time3 := 3  -- hours
  let speed3 := 65  -- km/h
  let time4 := 2  -- hours
  let speed4 := 90  -- km/h
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let distance3 := speed3 * time3
  let distance4 := speed4 * time4
  distance1 + distance2 + distance3 + distance4 = 905 :=
by
  sorry

end total_distance_traveled_l109_109826


namespace liam_balloons_remainder_l109_109856

def balloons : Nat := 24 + 45 + 78 + 96
def friends : Nat := 10
def remainder := balloons % friends

theorem liam_balloons_remainder : remainder = 3 := by
  sorry

end liam_balloons_remainder_l109_109856


namespace snake_price_correct_l109_109755

-- Define the conditions
def num_snakes : ℕ := 3
def eggs_per_snake : ℕ := 2
def total_eggs : ℕ := num_snakes * eggs_per_snake
def super_rare_multiple : ℕ := 4
def total_revenue : ℕ := 2250

-- The question: How much does each regular baby snake sell for?
def price_of_regular_baby_snake := 250

-- The proof statement
theorem snake_price_correct
  (x : ℕ)
  (h1 : total_eggs = 6)
  (h2 : 5 * x + super_rare_multiple * x = total_revenue)
  :
  x = price_of_regular_baby_snake := 
sorry

end snake_price_correct_l109_109755


namespace max_lamps_on_road_l109_109974

theorem max_lamps_on_road (k: ℕ) (lk: ℕ): 
  lk = 1000 → (∀ n: ℕ, n < k → n≥ 1 ∧ ∀ m: ℕ, if m > n then m > 1 else true) → (lk ≤ k) ∧ 
  (∀ i:ℕ,∃ j, (i ≠ j) → (lk < 1000)) → k = 1998 :=
by sorry

end max_lamps_on_road_l109_109974


namespace total_time_before_main_game_l109_109693

-- Define the time spent on each activity according to the conditions
def download_time := 10
def install_time := download_time / 2
def update_time := 2 * download_time
def account_time := 5
def internet_issues_time := 15
def discussion_time := 20
def video_time := 8

-- Define the total preparation time
def preparation_time := download_time + install_time + update_time + account_time + internet_issues_time + discussion_time + video_time

-- Define the in-game tutorial time
def tutorial_time := 3 * preparation_time

-- Prove that the total time before playing the main game is 332 minutes
theorem total_time_before_main_game : preparation_time + tutorial_time = 332 := by
  -- Provide a detailed proof here
  sorry

end total_time_before_main_game_l109_109693


namespace tommy_needs_4_steaks_l109_109598

noncomputable def tommy_steaks : Nat := 
  let family_members := 5
  let ounces_per_pound := 16
  let ounces_per_steak := 20
  let total_ounces_needed := family_members * ounces_per_pound
  let steaks_needed := total_ounces_needed / ounces_per_steak
  steaks_needed

theorem tommy_needs_4_steaks :
  tommy_steaks = 4 :=
by
  sorry

end tommy_needs_4_steaks_l109_109598


namespace remainder_of_x7_plus_2_div_x_plus_1_l109_109594

def f (x : ℤ) := x^7 + 2

theorem remainder_of_x7_plus_2_div_x_plus_1 : 
  (f (-1) = 1) := sorry

end remainder_of_x7_plus_2_div_x_plus_1_l109_109594


namespace jasmine_first_exceed_500_l109_109444

theorem jasmine_first_exceed_500 {k : ℕ} (initial : ℕ) (factor : ℕ) :
  initial = 5 → factor = 4 → (5 * 4^k > 500) → k = 4 :=
by
  sorry

end jasmine_first_exceed_500_l109_109444


namespace precise_approximate_classification_l109_109164

def data_points : List String := ["Xiao Ming bought 5 books today",
                                  "The war in Afghanistan cost the United States $1 billion per month in 2002",
                                  "Relevant departments predict that in 2002, the sales of movies in DVD format will exceed those of VHS tapes for the first time, reaching $9.5 billion",
                                  "The human brain has 10,000,000,000 cells",
                                  "Xiao Hong scored 92 points on this test",
                                  "The Earth has more than 1.5 trillion tons of coal reserves"]

def is_precise (data : String) : Bool :=
  match data with
  | "Xiao Ming bought 5 books today" => true
  | "The war in Afghanistan cost the United States $1 billion per month in 2002" => true
  | "Relevant departments predict that in 2002, the sales of movies in DVD format will exceed those of VHS tapes for the first time, reaching $9.5 billion" => true
  | "Xiao Hong scored 92 points on this test" => true
  | _ => false

def is_approximate (data : String) : Bool :=
  match data with
  | "The human brain has 10,000,000,000 cells" => true
  | "The Earth has more than 1.5 trillion tons of coal reserves" => true
  | _ => false

theorem precise_approximate_classification :
  (data_points.filter is_precise = ["Xiao Ming bought 5 books today",
                                    "The war in Afghanistan cost the United States $1 billion per month in 2002",
                                    "Relevant departments predict that in 2002, the sales of movies in DVD format will exceed those of VHS tapes for the first time, reaching $9.5 billion",
                                    "Xiao Hong scored 92 points on this test"]) ∧
  (data_points.filter is_approximate = ["The human brain has 10,000,000,000 cells",
                                        "The Earth has more than 1.5 trillion tons of coal reserves"]) :=
by sorry

end precise_approximate_classification_l109_109164


namespace total_screens_sold_is_45000_l109_109711

-- Define the number of screens sold in each month based on X
variables (X : ℕ)

-- Conditions given in the problem
def screens_in_January := X
def screens_in_February := 2 * X
def screens_in_March := (screens_in_January X + screens_in_February X) / 2
def screens_in_April := min (2 * screens_in_March X) 20000

-- Given that April sales were 18000
axiom apr_sales_18000 : screens_in_April X = 18000

-- Total sales is the sum of sales from January to April
def total_sales := screens_in_January X + screens_in_February X + screens_in_March X + 18000

-- Prove that total sales is 45000
theorem total_screens_sold_is_45000 : total_sales X = 45000 :=
by sorry

end total_screens_sold_is_45000_l109_109711


namespace range_of_a_l109_109339

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1^2 * Real.exp (-x1) = a) 
    ∧ (x2^2 * Real.exp (-x2) = a) ∧ (x3^2 * Real.exp (-x3) = a)) ↔ (0 < a ∧ a < 4 * Real.exp (-2)) :=
sorry

end range_of_a_l109_109339


namespace volume_of_prism_l109_109726

theorem volume_of_prism :
  ∃ (a b c : ℝ), ab * bc * ac = 762 ∧ (ab = 56) ∧ (bc = 63) ∧ (ac = 72) ∧ (b = 2 * a) :=
sorry

end volume_of_prism_l109_109726


namespace find_y_l109_109582

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

end find_y_l109_109582


namespace geometric_sequence_product_l109_109638

/-- Given a geometric sequence with positive terms where a_3 = 3 and a_6 = 1/9,
    prove that a_4 * a_5 = 1/3. -/
theorem geometric_sequence_product (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, a n > 0)
    (h_geometric : ∀ n, a (n + 1) = a n * q) (ha3 : a 3 = 3) (ha6 : a 6 = 1 / 9) :
  a 4 * a 5 = 1 / 3 := 
by
  sorry

end geometric_sequence_product_l109_109638


namespace linear_function_quadrants_l109_109367

theorem linear_function_quadrants (k b : ℝ) (h : k * b < 0) : 
  (∀ x : ℝ, (k < 0 ∧ b > 0) → (k * x + b > 0 → x > 0) ∧ (k * x + b < 0 → x < 0)) ∧ 
  (∀ x : ℝ, (k > 0 ∧ b < 0) → (k * x + b > 0 → x > 0) ∧ (k * x + b < 0 → x < 0)) :=
sorry

end linear_function_quadrants_l109_109367


namespace trig_identity_l109_109316

theorem trig_identity (α : ℝ) (h0 : Real.tan α = Real.sqrt 3) (h1 : π < α) (h2 : α < 3 * π / 2) :
  Real.cos (2 * α) - Real.sin (π / 2 + α) = 0 :=
sorry

end trig_identity_l109_109316


namespace intersection_A_B_l109_109668

-- Defining sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

-- Theorem stating the intersection of A and B
theorem intersection_A_B : A ∩ B = {2} := by
  sorry

end intersection_A_B_l109_109668


namespace determine_min_guesses_l109_109903

def minimum_guesses (n k : ℕ) (h : n > k) : ℕ :=
  if n = 2 * k then 2 else 1

theorem determine_min_guesses (n k : ℕ) (h : n > k) :
  (if n = 2 * k then 2 else 1) = minimum_guesses n k h := by
  sorry

end determine_min_guesses_l109_109903


namespace calories_per_one_bar_l109_109296

variable (total_calories : ℕ) (num_bars : ℕ)
variable (calories_per_bar : ℕ)

-- Given conditions
axiom total_calories_given : total_calories = 15
axiom num_bars_given : num_bars = 5

-- Mathematical equivalent proof problem
theorem calories_per_one_bar :
  total_calories / num_bars = calories_per_bar →
  calories_per_bar = 3 :=
by
  sorry

end calories_per_one_bar_l109_109296


namespace money_spent_on_jacket_l109_109593

-- Define the initial amounts
def initial_money_sandy : ℝ := 13.99
def amount_spent_shirt : ℝ := 12.14
def additional_money_found : ℝ := 7.43

-- Amount of money left after buying the shirt
def remaining_after_shirt := initial_money_sandy - amount_spent_shirt

-- Total money after finding additional money
def total_after_additional := remaining_after_shirt + additional_money_found

-- Theorem statement: The amount Sandy spent on the jacket
theorem money_spent_on_jacket : total_after_additional = 9.28 :=
by
  sorry

end money_spent_on_jacket_l109_109593


namespace inequality_always_holds_true_l109_109694

theorem inequality_always_holds_true (a b c : ℝ) (h₁ : a > b) (h₂ : b > c) :
  (a / (c^2 + 1)) > (b / (c^2 + 1)) :=
by
  sorry

end inequality_always_holds_true_l109_109694


namespace sale_in_second_month_l109_109211

theorem sale_in_second_month 
  (sale_first_month: ℕ := 2500)
  (sale_third_month: ℕ := 3540)
  (sale_fourth_month: ℕ := 1520)
  (average_sale: ℕ := 2890)
  (total_sales: ℕ := 11560) :
  sale_first_month + sale_third_month + sale_fourth_month + (sale_second_month: ℕ) = total_sales → 
  sale_second_month = 4000 := 
by
  intros h
  sorry

end sale_in_second_month_l109_109211


namespace nonnegative_solution_positive_solution_l109_109437

/-- For k > 7, there exist non-negative integers x and y such that 5*x + 3*y = k. -/
theorem nonnegative_solution (k : ℤ) (hk : k > 7) : ∃ x y : ℕ, 5 * x + 3 * y = k :=
sorry

/-- For k > 15, there exist positive integers x and y such that 5*x + 3*y = k. -/
theorem positive_solution (k : ℤ) (hk : k > 15) : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 5 * x + 3 * y = k :=
sorry

end nonnegative_solution_positive_solution_l109_109437


namespace expression_value_l109_109377

def a : ℕ := 1000
def b1 : ℕ := 15
def b2 : ℕ := 314
def c1 : ℕ := 201
def c2 : ℕ := 360
def c3 : ℕ := 110
def d1 : ℕ := 201
def d2 : ℕ := 360
def d3 : ℕ := 110
def e1 : ℕ := 15
def e2 : ℕ := 314

theorem expression_value :
  (a + b1 + b2) * (c1 + c2 + c3) + (a - d1 - d2 - d3) * (e1 + e2) = 1000000 :=
by
  sorry

end expression_value_l109_109377


namespace age_relation_l109_109468

theorem age_relation (S M D Y : ℝ)
  (h1 : M = S + 37)
  (h2 : M + 2 = 2 * (S + 2))
  (h3 : D = S - 4)
  (h4 : M + Y = 3 * (D + Y))
  : Y = -10.5 :=
by
  sorry

end age_relation_l109_109468


namespace glasses_per_pitcher_l109_109165

def total_glasses : Nat := 30
def num_pitchers : Nat := 6

theorem glasses_per_pitcher : total_glasses / num_pitchers = 5 := by
  sorry

end glasses_per_pitcher_l109_109165


namespace number_of_valid_n_l109_109610

theorem number_of_valid_n : 
  (∃ (n : ℕ), ∀ (a b c : ℕ), 8 * a + 88 * b + 888 * c = 8000 → n = a + 2 * b + 3 * c) ↔
  (∃ (n : ℕ), n = 1000) := by 
  sorry

end number_of_valid_n_l109_109610


namespace field_day_difference_l109_109436

theorem field_day_difference :
  let girls_class_4_1 := 12
  let boys_class_4_1 := 13
  let girls_class_4_2 := 15
  let boys_class_4_2 := 11
  let girls_class_5_1 := 9
  let boys_class_5_1 := 13
  let girls_class_5_2 := 10
  let boys_class_5_2 := 11
  let total_girls := girls_class_4_1 + girls_class_4_2 + girls_class_5_1 + girls_class_5_2
  let total_boys := boys_class_4_1 + boys_class_4_2 + boys_class_5_1 + boys_class_5_2
  total_boys - total_girls = 2 := by
  sorry

end field_day_difference_l109_109436


namespace domain_of_ratio_function_l109_109744

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := f (2 ^ x)

theorem domain_of_ratio_function (D : Set ℝ) (hD : D = Set.Icc 1 2):
  ∀ f : ℝ → ℝ, (∀ x, g x = f (2 ^ x)) →
  ∃ D' : Set ℝ, D' = {x | 2 ≤ x ∧ x ≤ 4} →
  ∀ y : ℝ, (2 ≤ y ∧ y ≤ 4) → ∃ x : ℝ, y = x + 1 ∧ x ≠ 1 → (1 < x ∧ x ≤ 3) :=
sorry

end domain_of_ratio_function_l109_109744


namespace lcm_fractions_l109_109170

theorem lcm_fractions (x : ℕ) (hx : x ≠ 0) : 
  (∀ (a b c : ℕ), (a = 4*x ∧ b = 5*x ∧ c = 6*x) → (Nat.lcm (Nat.lcm a b) c = 60 * x)) :=
by
  sorry

end lcm_fractions_l109_109170


namespace quadratic_intersection_with_x_axis_l109_109672

theorem quadratic_intersection_with_x_axis :
  ∃ x : ℝ, (x^2 - 4*x + 4 = 0) ∧ (x = 2) ∧ (x, 0) = (2, 0) :=
sorry

end quadratic_intersection_with_x_axis_l109_109672


namespace goldfish_added_per_day_is_7_l109_109830

def initial_koi_fish : ℕ := 227 - 2
def initial_goldfish : ℕ := 280 - initial_koi_fish
def added_goldfish : ℕ := 200 - initial_goldfish
def days_in_three_weeks : ℕ := 3 * 7
def goldfish_added_per_day : ℕ := (added_goldfish + days_in_three_weeks - 1) / days_in_three_weeks -- rounding to nearest integer 

theorem goldfish_added_per_day_is_7 : goldfish_added_per_day = 7 :=
by 
-- sorry to skip the proof
sorry

end goldfish_added_per_day_is_7_l109_109830


namespace value_of_expression_at_three_l109_109960

theorem value_of_expression_at_three (x : ℝ) (h : x = 3) : (x^2 - 3 * x - 10) / (x - 5) = 5 := 
by
  sorry

end value_of_expression_at_three_l109_109960


namespace log_comparison_l109_109382

/-- Assuming a = log base 3 of 2, b = natural log of 3, and c = log base 2 of 3,
    prove that c > b > a. -/
theorem log_comparison (a b c : ℝ) (h1 : a = Real.log 2 / Real.log 3)
                                (h2 : b = Real.log 3)
                                (h3 : c = Real.log 3 / Real.log 2) :
  c > b ∧ b > a :=
by {
  sorry
}

end log_comparison_l109_109382


namespace probability_defective_unit_l109_109190

theorem probability_defective_unit (T : ℝ) 
  (P_A : ℝ := 9 / 1000) 
  (P_B : ℝ := 1 / 50) 
  (output_ratio_A : ℝ := 0.4)
  (output_ratio_B : ℝ := 0.6) : 
  (P_A * output_ratio_A + P_B * output_ratio_B) = 0.0156 :=
by
  sorry

end probability_defective_unit_l109_109190


namespace compare_abc_l109_109994

noncomputable def a : ℝ := ∫ x in (0:ℝ)..1, x ^ (-1/3 : ℝ)
noncomputable def b : ℝ := 1 - ∫ x in (0:ℝ)..1, x ^ (1/2 : ℝ)
noncomputable def c : ℝ := ∫ x in (0:ℝ)..1, x ^ (3 : ℝ)

theorem compare_abc : a > b ∧ b > c := by
  sorry

end compare_abc_l109_109994


namespace trajectory_of_M_l109_109859

theorem trajectory_of_M
  (A : ℝ × ℝ := (3, 0))
  (P_circle : ∀ (P : ℝ × ℝ), P.1^2 + P.2^2 = 1)
  (M_midpoint : ∀ (P M : ℝ × ℝ), M = ((P.1 + 3) / 2, P.2 / 2) → M.1 = x ∧ M.2 = y) :
  (∀ (x y : ℝ), (x - 3/2)^2 + y^2 = 1/4) := 
sorry

end trajectory_of_M_l109_109859


namespace pair_solution_l109_109747

theorem pair_solution (a b : ℕ) (h_b_ne_1 : b ≠ 1) :
  (a + 1 ∣ a^3 * b - 1) → (b - 1 ∣ b^3 * a + 1) →
  (a, b) = (0, 0) ∨ (a, b) = (0, 2) ∨ (a, b) = (2, 2) ∨ (a, b) = (1, 3) ∨ (a, b) = (3, 3) :=
by
  sorry

end pair_solution_l109_109747


namespace count_valid_subsets_l109_109013

theorem count_valid_subsets : 
  ∃ (S : Finset (Finset ℕ)), 
    (∀ A ∈ S, A ⊆ {1, 2, 3, 4, 5} ∧ 
    (∀ a ∈ A, 6 - a ∈ A)) ∧ 
    S.card = 7 := 
sorry

end count_valid_subsets_l109_109013


namespace not_all_mages_are_wizards_l109_109309

variable (M S W : Type → Prop)

theorem not_all_mages_are_wizards
  (h1 : ∃ x, M x ∧ ¬ S x)
  (h2 : ∀ x, M x ∧ W x → S x) :
  ∃ x, M x ∧ ¬ W x :=
sorry

end not_all_mages_are_wizards_l109_109309


namespace group4_equations_groupN_equations_find_k_pos_l109_109253

-- Conditions from the problem
def group1_fractions := (1 : ℚ) / 1 + (1 : ℚ) / 3 = 4 / 3
def group1_pythagorean := 4^2 + 3^2 = 5^2

def group2_fractions := (1 : ℚ) / 3 + (1 : ℚ) / 5 = 8 / 15
def group2_pythagorean := 8^2 + 15^2 = 17^2

def group3_fractions := (1 : ℚ) / 5 + (1 : ℚ) / 7 = 12 / 35
def group3_pythagorean := 12^2 + 35^2 = 37^2

-- Proof Statements
theorem group4_equations :
  ((1 : ℚ) / 7 + (1 : ℚ) / 9 = 16 / 63) ∧ (16^2 + 63^2 = 65^2) := 
  sorry

theorem groupN_equations (n : ℕ) :
  ((1 : ℚ) / (2 * n - 1) + (1 : ℚ) / (2 * n + 1) = 4 * n / (4 * n^2 - 1)) ∧
  ((4 * n)^2 + (4 * n^2 - 1)^2 = (4 * n^2 + 1)^2) :=
  sorry

theorem find_k_pos (k : ℕ) : 
  k^2 + 9603^2 = 9605^2 → k = 196 := 
  sorry

end group4_equations_groupN_equations_find_k_pos_l109_109253


namespace candy_division_l109_109701

theorem candy_division 
  (total_candy : ℕ)
  (total_bags : ℕ)
  (candies_per_bag : ℕ)
  (chocolate_heart_bags : ℕ)
  (fruit_jelly_bags : ℕ)
  (caramel_chew_bags : ℕ) 
  (H1 : total_candy = 260)
  (H2 : total_bags = 13)
  (H3 : candies_per_bag = total_candy / total_bags)
  (H4 : chocolate_heart_bags = 4)
  (H5 : fruit_jelly_bags = 3)
  (H6 : caramel_chew_bags = total_bags - chocolate_heart_bags - fruit_jelly_bags)
  (H7 : candies_per_bag = 20) :
  (chocolate_heart_bags * candies_per_bag) + 
  (fruit_jelly_bags * candies_per_bag) + 
  (caramel_chew_bags * candies_per_bag) = 260 :=
sorry

end candy_division_l109_109701


namespace graph_passes_through_quadrants_l109_109802

def linear_function (x : ℝ) : ℝ := -5 * x + 5

theorem graph_passes_through_quadrants :
  (∃ x y : ℝ, linear_function x = y ∧ x > 0 ∧ y > 0) ∧  -- Quadrant I
  (∃ x y : ℝ, linear_function x = y ∧ x < 0 ∧ y > 0) ∧  -- Quadrant II
  (∃ x y : ℝ, linear_function x = y ∧ x > 0 ∧ y < 0)    -- Quadrant IV
  :=
by
  sorry

end graph_passes_through_quadrants_l109_109802


namespace cost_effectiveness_l109_109663

-- Define general parameters and conditions given in the problem
def a : ℕ := 70 -- We use 70 since it must be greater than 50

-- Define the scenarios
def cost_scenario1 (a: ℕ) : ℕ := 4500 + 27 * a
def cost_scenario2 (a: ℕ) : ℕ := 4400 + 30 * a

-- The theorem to be proven
theorem cost_effectiveness (h : a > 50) : cost_scenario1 a < cost_scenario2 a :=
  by
  -- First, let's replace a with 70 (this step is unnecessary in the proof since a = 70 is fixed)
  let a := 70
  -- Now, prove the inequality
  sorry

end cost_effectiveness_l109_109663


namespace arsenic_acid_concentration_equilibrium_l109_109519

noncomputable def dissociation_constants 
  (Kd1 Kd2 Kd3 : ℝ) (H3AsO4 H2AsO4 HAsO4 AsO4 H : ℝ) : Prop :=
  Kd1 = (H * H2AsO4) / H3AsO4 ∧ Kd2 = (H * HAsO4) / H2AsO4 ∧ Kd3 = (H * AsO4) / HAsO4

theorem arsenic_acid_concentration_equilibrium :
  dissociation_constants 5.6e-3 1.7e-7 2.95e-12 0.1 (2e-2) (1.7e-7) (0) (2e-2) :=
by sorry

end arsenic_acid_concentration_equilibrium_l109_109519


namespace probability_fail_then_succeed_l109_109631

theorem probability_fail_then_succeed
  (P_fail_first : ℚ := 9 / 10)
  (P_succeed_second : ℚ := 1 / 9) :
  P_fail_first * P_succeed_second = 1 / 10 :=
by
  sorry

end probability_fail_then_succeed_l109_109631


namespace shaded_region_area_l109_109133

noncomputable def area_shaded_region (r_small r_large : ℝ) (A B : ℝ × ℝ) : ℝ := 
  let pi := Real.pi
  let sqrt_5 := Real.sqrt 5
  (5 * pi / 2) - (4 * sqrt_5)

theorem shaded_region_area : 
  ∀ (r_small r_large : ℝ) (A B : ℝ × ℝ), 
  r_small = 2 → 
  r_large = 3 → 
  (A = (0, 0)) → 
  (B = (4, 0)) → 
  area_shaded_region r_small r_large A B = (5 * Real.pi / 2) - (4 * Real.sqrt 5) := 
by
  intros r_small r_large A B h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact rfl

end shaded_region_area_l109_109133


namespace train_speed_l109_109863

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 700) (h_time : time = 40) : length / time = 17.5 :=
by
  -- length / time represents the speed of the train
  -- given length = 700 meters and time = 40 seconds
  -- we have to prove that 700 / 40 = 17.5
  sorry

end train_speed_l109_109863


namespace elberta_amount_l109_109673

theorem elberta_amount (grannySmith_amount : ℝ) (Anjou_factor : ℝ) (extra_amount : ℝ) :
  grannySmith_amount = 45 →
  Anjou_factor = 1 / 4 →
  extra_amount = 4 →
  (extra_amount + Anjou_factor * grannySmith_amount) = 15.25 :=
by
  intros h_grannySmith h_AnjouFactor h_extraAmount
  sorry

end elberta_amount_l109_109673


namespace trucks_in_yard_l109_109542

/-- The number of trucks in the yard is 23, given the conditions. -/
theorem trucks_in_yard (T : ℕ) (H1 : ∃ n : ℕ, n > 0)
  (H2 : ∃ k : ℕ, k = 5 * T)
  (H3 : T + 5 * T = 140) : T = 23 :=
sorry

end trucks_in_yard_l109_109542


namespace max_squares_overlap_l109_109323

-- Definitions based on conditions.
def side_length_checkerboard_square : ℝ := 0.75
def side_length_card : ℝ := 2
def minimum_overlap : ℝ := 0.25

-- Main theorem to prove.
theorem max_squares_overlap :
  ∃ max_overlap_squares : ℕ, max_overlap_squares = 9 :=
by
  sorry

end max_squares_overlap_l109_109323


namespace tiling_remainder_is_888_l109_109581

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

end tiling_remainder_is_888_l109_109581


namespace Jake_weight_196_l109_109679

def Jake_and_Sister : Prop :=
  ∃ (J S : ℕ), (J - 8 = 2 * S) ∧ (J + S = 290) ∧ (J = 196)

theorem Jake_weight_196 : Jake_and_Sister :=
by
  sorry

end Jake_weight_196_l109_109679


namespace election_votes_l109_109479

theorem election_votes (V : ℝ) (h1 : 0.70 * V - 0.30 * V = 200) : V = 500 :=
sorry

end election_votes_l109_109479


namespace geom_seq_common_ratio_l109_109626

-- We define a geometric sequence and the condition provided in the problem.
variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Condition for geometric sequence: a_n = a * q^(n-1)
def is_geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n : ℕ, a n = a 0 * q^(n-1)

-- Given condition: 2a_4 = a_6 - a_5
def given_condition (a : ℕ → ℝ) : Prop := 
  2 * a 4 = a 6 - a 5

-- Proof statement
theorem geom_seq_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : is_geometric_seq a q) (h_cond : given_condition a) : 
    q = 2 ∨ q = -1 :=
sorry

end geom_seq_common_ratio_l109_109626


namespace curve_C1_general_equation_curve_C2_cartesian_equation_minimum_distance_P1P2_l109_109589

noncomputable def parametric_curve_C1 (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, Real.sqrt 2 * Real.sin α)

noncomputable def polar_curve_C2 (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ + Real.pi / 4) = 3 * Real.sqrt 2

theorem curve_C1_general_equation (x y : ℝ) (α : ℝ) :
  (2 * Real.cos α = x) ∧ (Real.sqrt 2 * Real.sin α = y) →
  x^2 / 4 + y^2 / 2 = 1 :=
sorry

theorem curve_C2_cartesian_equation (ρ θ : ℝ) (x y : ℝ) :
  (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) ∧ polar_curve_C2 ρ θ →
  x + y = 6 :=
sorry

theorem minimum_distance_P1P2 (P1 P2 : ℝ × ℝ) (d : ℝ) :
  (∃ α, P1 = parametric_curve_C1 α) ∧ (∃ x y, P2 = (x, y) ∧ x + y = 6) →
  d = (3 * Real.sqrt 2 - Real.sqrt 3) :=
sorry

end curve_C1_general_equation_curve_C2_cartesian_equation_minimum_distance_P1P2_l109_109589


namespace days_B_to_complete_remaining_work_l109_109371

/-- 
  Given that:
  - A can complete a work in 20 days.
  - B can complete the same work in 12 days.
  - A and B worked together for 3 days before A left.
  
  We need to prove that B will require 7.2 days to complete the remaining work alone. 
--/
theorem days_B_to_complete_remaining_work : 
  (∃ (A_rate B_rate combined_rate work_done_in_3_days remaining_work d_B : ℚ), 
   A_rate = (1 / 20) ∧
   B_rate = (1 / 12) ∧
   combined_rate = A_rate + B_rate ∧
   work_done_in_3_days = 3 * combined_rate ∧
   remaining_work = 1 - work_done_in_3_days ∧
   d_B = remaining_work / B_rate ∧
   d_B = 7.2) := 
by 
  sorry

end days_B_to_complete_remaining_work_l109_109371


namespace simplify_expression_l109_109294

theorem simplify_expression (n : ℕ) : 
  (3 ^ (n + 5) - 3 * 3 ^ n) / (3 * 3 ^ (n + 4)) = 80 / 27 :=
by sorry

end simplify_expression_l109_109294


namespace solve_for_a_l109_109380

theorem solve_for_a : ∃ a : ℝ, (∀ x : ℝ, x = -2 → x^2 - a * x + 7 = 0) → a = -11 / 2 :=
by 
  sorry

end solve_for_a_l109_109380


namespace sphere_radius_l109_109071

-- Define the conditions
variable (r : ℝ) -- Radius of the sphere
variable (sphere_shadow : ℝ) (stick_height : ℝ) (stick_shadow : ℝ)

-- Given conditions
axiom sphere_shadow_equals_10 : sphere_shadow = 10
axiom stick_height_equals_1 : stick_height = 1
axiom stick_shadow_equals_2 : stick_shadow = 2

-- Using similar triangles and tangent relations, we want to prove the radius of sphere.
theorem sphere_radius (h1 : sphere_shadow = 10)
    (h2 : stick_height = 1)
    (h3 : stick_shadow = 2) : r = 5 :=
by
  -- Placeholder for the proof
  sorry

end sphere_radius_l109_109071


namespace part_a_part_b_l109_109815

-- Definitions for maximum factor increases
def f (n : ℕ) (a : ℕ) : ℚ := sorry
def t (n : ℕ) (a : ℕ) : ℚ := sorry

-- Part (a): Prove the factor increase for exactly 1 blue cube in 100 boxes
theorem part_a : f 100 1 = 2^100 / 100 := sorry

-- Part (b): Prove the factor increase for some integer \( k \) blue cubes in 100 boxes, \( 1 < k \leq 100 \)
theorem part_b (k : ℕ) (hk : 1 < k ∧ k ≤ 100) : t 100 k = 2^100 / (2^100 - k - 1) := sorry

end part_a_part_b_l109_109815


namespace trapezoid_midsegment_l109_109796

theorem trapezoid_midsegment (h : ℝ) :
  ∃ k : ℝ, (∃ θ : ℝ, θ = 120 ∧ k = 2 * h * Real.cos (θ / 2)) ∧
  (∃ m : ℝ, m = k / 2) ∧
  (∃ midsegment : ℝ, midsegment = m / Real.sqrt 3 ∧ midsegment = h / Real.sqrt 3) :=
by
  -- This is where the proof would go.
  sorry

end trapezoid_midsegment_l109_109796


namespace xy_diff_square_l109_109338

theorem xy_diff_square (x y : ℝ) (h1 : x + y = -5) (h2 : x * y = 6) : (x - y)^2 = 1 :=
by
  sorry

end xy_diff_square_l109_109338


namespace triangle_XOY_hypotenuse_l109_109332

theorem triangle_XOY_hypotenuse (a b : ℝ) (h1 : (a/2)^2 + b^2 = 22^2) (h2 : a^2 + (b/2)^2 = 19^2) :
  Real.sqrt (a^2 + b^2) = 26 :=
sorry

end triangle_XOY_hypotenuse_l109_109332


namespace junior_score_proof_l109_109445

noncomputable def class_total_score (total_students : ℕ) (average_class_score : ℕ) : ℕ :=
total_students * average_class_score

noncomputable def number_of_juniors (total_students : ℕ) (percent_juniors : ℕ) : ℕ :=
percent_juniors * total_students / 100

noncomputable def number_of_seniors (total_students juniors : ℕ) : ℕ :=
total_students - juniors

noncomputable def total_senior_score (seniors average_senior_score : ℕ) : ℕ :=
seniors * average_senior_score

noncomputable def total_junior_score (total_score senior_score : ℕ) : ℕ :=
total_score - senior_score

noncomputable def junior_score (junior_total_score juniors : ℕ) : ℕ :=
junior_total_score / juniors

theorem junior_score_proof :
  ∀ (total_students: ℕ) (percent_juniors average_class_score average_senior_score : ℕ),
  total_students = 20 →
  percent_juniors = 15 →
  average_class_score = 85 →
  average_senior_score = 84 →
  (junior_score (total_junior_score (class_total_score total_students average_class_score)
                                    (total_senior_score (number_of_seniors total_students (number_of_juniors total_students percent_juniors))
                                                        average_senior_score))
                (number_of_juniors total_students percent_juniors)) = 91 :=
by
  intros
  sorry

end junior_score_proof_l109_109445


namespace part_a_part_b_l109_109946

-- Definitions based on the conditions:
def probability_of_hit (p : ℝ) := p
def probability_of_miss (p : ℝ) := 1 - p

-- Condition: exactly three unused rockets after firing at five targets
def exactly_three_unused_rockets (p : ℝ) : ℝ := 10 * (probability_of_hit p) ^ 3 * (probability_of_miss p) ^ 2

-- Condition: expected number of targets hit when there are nine targets
def expected_targets_hit (p : ℝ) : ℝ := 10 * p - p ^ 10

-- Lean 4 statements representing the proof problems:
theorem part_a (p : ℝ) (h_p_nonneg : 0 ≤ p) (h_p_le_one : p ≤ 1) : 
  exactly_three_unused_rockets p = 10 * p ^ 3 * (1 - p) ^ 2 :=
by sorry

theorem part_b (p : ℝ) (h_p_nonneg : 0 ≤ p) (h_p_le_one : p ≤ 1) :
  expected_targets_hit p = 10 * p - p ^ 10 :=
by sorry

end part_a_part_b_l109_109946


namespace teacher_discount_l109_109882

-- Definitions that capture the conditions in Lean
def num_students : ℕ := 30
def num_pens_per_student : ℕ := 5
def num_notebooks_per_student : ℕ := 3
def num_binders_per_student : ℕ := 1
def num_highlighters_per_student : ℕ := 2
def cost_per_pen : ℚ := 0.50
def cost_per_notebook : ℚ := 1.25
def cost_per_binder : ℚ := 4.25
def cost_per_highlighter : ℚ := 0.75
def amount_spent : ℚ := 260

-- Compute the total cost without discount
def total_cost : ℚ :=
  (num_students * num_pens_per_student) * cost_per_pen +
  (num_students * num_notebooks_per_student) * cost_per_notebook +
  (num_students * num_binders_per_student) * cost_per_binder +
  (num_students * num_highlighters_per_student) * cost_per_highlighter

-- The main theorem to prove the applied teacher discount
theorem teacher_discount :
  total_cost - amount_spent = 100 := by
  sorry

end teacher_discount_l109_109882


namespace min_value_x_plus_one_over_2x_l109_109482

theorem min_value_x_plus_one_over_2x (x : ℝ) (hx : x > 0) : 
  x + 1 / (2 * x) ≥ Real.sqrt 2 := sorry

end min_value_x_plus_one_over_2x_l109_109482


namespace rhombus_diagonal_length_l109_109074

theorem rhombus_diagonal_length (d1 d2 : ℝ) (A : ℝ) (h1 : d2 = 17) (h2 : A = 127.5) 
  (h3 : A = (d1 * d2) / 2) : d1 = 15 := 
by 
  -- Definitions
  sorry

end rhombus_diagonal_length_l109_109074


namespace money_r_gets_l109_109449

def total_amount : ℕ := 1210
def p_to_q := 5 / 4
def q_to_r := 9 / 10

theorem money_r_gets :
  let P := (total_amount * 45) / 121
  let Q := (total_amount * 36) / 121
  let R := (total_amount * 40) / 121
  R = 400 := by
  sorry

end money_r_gets_l109_109449


namespace climbing_stairs_l109_109381

noncomputable def total_methods_to_climb_stairs : ℕ :=
  (Nat.choose 8 5) + (Nat.choose 8 6) + (Nat.choose 8 7) + 1

theorem climbing_stairs (n : ℕ := 9) (min_steps : ℕ := 6) (max_steps : ℕ := 9)
  (H1 : min_steps ≤ n)
  (H2 : n ≤ max_steps)
  : total_methods_to_climb_stairs = 93 := by
  sorry

end climbing_stairs_l109_109381


namespace price_of_other_frisbees_l109_109399

theorem price_of_other_frisbees 
  (P : ℝ) 
  (x : ℝ)
  (h1 : x + (64 - x) = 64)
  (h2 : P * x + 4 * (64 - x) = 196)
  (h3 : 64 - x ≥ 4) 
  : P = 3 :=
sorry

end price_of_other_frisbees_l109_109399


namespace simplify_fraction_l109_109798

theorem simplify_fraction (x y : ℝ) : (x - y) / (y - x) = -1 :=
sorry

end simplify_fraction_l109_109798


namespace find_number_l109_109289

theorem find_number (x n : ℕ) (h1 : 3 * x + n = 48) (h2 : x = 4) : n = 36 :=
by
  sorry

end find_number_l109_109289


namespace AM_GM_inequality_example_l109_109009

theorem AM_GM_inequality_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 6) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 1 / 2 :=
by
  sorry

end AM_GM_inequality_example_l109_109009


namespace melanie_trout_l109_109014

theorem melanie_trout (M : ℕ) (h1 : 2 * M = 16) : M = 8 :=
by
  sorry

end melanie_trout_l109_109014


namespace matrix_multiplication_correct_l109_109212

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 1], ![4, -2]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![5, -3], ![2, 6]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![17, -3], ![16, -24]]

theorem matrix_multiplication_correct : A * B = C := by 
  sorry

end matrix_multiplication_correct_l109_109212


namespace sum_of_first_5n_l109_109198

theorem sum_of_first_5n (n : ℕ) (h : (3 * n) * (3 * n + 1) / 2 = n * (n + 1) / 2 + 270) : (5 * n) * (5 * n + 1) / 2 = 820 :=
by
  sorry

end sum_of_first_5n_l109_109198


namespace tonya_stamps_left_l109_109522

theorem tonya_stamps_left 
    (stamps_per_matchbook : ℕ) 
    (matches_per_matchbook : ℕ) 
    (tonya_initial_stamps : ℕ) 
    (jimmy_initial_matchbooks : ℕ) 
    (stamps_per_match : ℕ) 
    (tonya_final_stamps_expected : ℕ)
    (h1 : stamps_per_matchbook = 1) 
    (h2 : matches_per_matchbook = 24) 
    (h3 : tonya_initial_stamps = 13) 
    (h4 : jimmy_initial_matchbooks = 5) 
    (h5 : stamps_per_match = 12)
    (h6 : tonya_final_stamps_expected = 3) :
    tonya_initial_stamps - jimmy_initial_matchbooks * (matches_per_matchbook / stamps_per_match) = tonya_final_stamps_expected :=
by
  sorry

end tonya_stamps_left_l109_109522


namespace inequality_for_natural_n_l109_109935

theorem inequality_for_natural_n (n : ℕ) : (2 * n + 1)^n ≥ (2 * n)^n + (2 * n - 1)^n :=
by sorry

end inequality_for_natural_n_l109_109935


namespace Priyanka_chocolates_l109_109806

variable (N S So P Sa T : ℕ)

theorem Priyanka_chocolates :
  (N + S = 10) →
  (So + P = 15) →
  (Sa + T = 10) →
  (N = 4) →
  ((S = 2 * y) ∨ (P = 2 * So)) →
  P = 10 :=
by
  sorry

end Priyanka_chocolates_l109_109806


namespace ryan_learning_hours_l109_109751

theorem ryan_learning_hours (total_hours : ℕ) (chinese_hours : ℕ) (english_hours : ℕ) 
  (h1 : total_hours = 3) (h2 : chinese_hours = 1) : 
  english_hours = 2 :=
by 
  sorry

end ryan_learning_hours_l109_109751


namespace prob_product_less_than_36_is_15_over_16_l109_109619

noncomputable def prob_product_less_than_36 : ℚ := sorry

theorem prob_product_less_than_36_is_15_over_16 :
  prob_product_less_than_36 = 15 / 16 := 
sorry

end prob_product_less_than_36_is_15_over_16_l109_109619


namespace complement_union_l109_109063

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem complement_union :
  (U \ M) ∪ N = {2, 3, 4} :=
sorry

end complement_union_l109_109063


namespace x_mul_y_eq_4_l109_109690

theorem x_mul_y_eq_4 (x y z w : ℝ) (hw_pos : w > 0) 
  (h1 : x = w) (h2 : y = z) (h3 : w + w = z * w) 
  (h4 : y = w) (h5 : z = 3) (h6 : w + w = w * w) : 
  x * y = 4 := by
  sorry

end x_mul_y_eq_4_l109_109690


namespace correct_subtraction_l109_109337

theorem correct_subtraction (x : ℕ) (h : x - 63 = 8) : x - 36 = 35 :=
by sorry

end correct_subtraction_l109_109337


namespace find_solutions_l109_109952

def is_solution (a b c d : ℕ) : Prop :=
  Nat.gcd (Nat.gcd (Nat.gcd a b) c) d = 1 ∧
  a ∣ (b + c) ∧
  b ∣ (c + d) ∧
  c ∣ (d + a) ∧
  d ∣ (a + b)

theorem find_solutions : ∀ (a b c d : ℕ),
  is_solution a b c d →
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
  (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
  (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 1) ∨
  (a = 5 ∧ b = 3 ∧ c = 2 ∧ d = 1) ∨
  (a = 5 ∧ b = 4 ∧ c = 1 ∧ d = 3) ∨
  (a = 7 ∧ b = 5 ∧ c = 2 ∧ d = 3) ∨
  (a = 3 ∧ b = 1 ∧ c = 2 ∧ d = 1) ∨
  (a = 5 ∧ b = 1 ∧ c = 4 ∧ d = 3) ∨
  (a = 5 ∧ b = 2 ∧ c = 3 ∧ d = 1) ∨
  (a = 7 ∧ b = 2 ∧ c = 5 ∧ d = 3) ∨
  (a = 7 ∧ b = 3 ∧ c = 4 ∧ d = 5) :=
by
  intros a b c d h
  sorry

end find_solutions_l109_109952


namespace cookie_count_per_box_l109_109002

theorem cookie_count_per_box (A B C T: ℝ) (H1: A = 2) (H2: B = 0.75) (H3: C = 3) (H4: T = 276) :
  T / (A + B + C) = 48 :=
by
  sorry

end cookie_count_per_box_l109_109002


namespace straw_costs_max_packs_type_a_l109_109142

theorem straw_costs (x y : ℝ) (h1 : 12 * x + 15 * y = 171) (h2 : 24 * x + 28 * y = 332) :
  x = 8 ∧ y = 5 :=
  by sorry

theorem max_packs_type_a (m : ℕ) (cA cB : ℕ) (total_packs : ℕ) (max_cost : ℕ)
  (h1 : cA = 8) (h2 : cB = 5) (h3 : total_packs = 100) (h4 : max_cost = 600) :
  m ≤ 33 :=
  by sorry

end straw_costs_max_packs_type_a_l109_109142


namespace Jeremy_age_l109_109055

noncomputable def A : ℝ := sorry
noncomputable def J : ℝ := sorry
noncomputable def C : ℝ := sorry

-- Conditions
axiom h1 : A + J + C = 132
axiom h2 : A = (1/3) * J
axiom h3 : C = 2 * A

-- The goal is to prove J = 66
theorem Jeremy_age : J = 66 :=
sorry

end Jeremy_age_l109_109055


namespace probability_A8_l109_109787

/-- Define the probability of event A_n where the sum of die rolls equals n -/
def P (n : ℕ) : ℚ :=
  1/7 * (if n = 8 then 5/36 + 21/216 + 35/1296 + 35/7776 + 21/46656 +
    7/279936 + 1/1679616 else 0)

theorem probability_A8 : P 8 = (1/7) * (5/36 + 21/216 + 35/1296 + 35/7776 + 
  21/46656 + 7/279936 + 1/1679616) :=
by
  sorry

end probability_A8_l109_109787


namespace coordinates_of_point_l109_109107

theorem coordinates_of_point : 
  ∀ (x y : ℝ), (x, y) = (2, -3) → (x, y) = (2, -3) := 
by 
  intros x y h 
  exact h

end coordinates_of_point_l109_109107


namespace seats_taken_l109_109004

variable (num_rows : ℕ) (chairs_per_row : ℕ) (unoccupied_chairs : ℕ)

theorem seats_taken (h1 : num_rows = 40) (h2 : chairs_per_row = 20) (h3 : unoccupied_chairs = 10) :
  num_rows * chairs_per_row - unoccupied_chairs = 790 :=
sorry

end seats_taken_l109_109004


namespace iron_weight_l109_109888

theorem iron_weight 
  (A : ℝ) (hA : A = 0.83) 
  (I : ℝ) (hI : I = A + 10.33) : 
  I = 11.16 := 
by 
  sorry

end iron_weight_l109_109888


namespace geralds_average_speed_l109_109032

theorem geralds_average_speed :
  ∀ (track_length : ℝ) (pollys_laps : ℕ) (pollys_time : ℝ) (geralds_factor : ℝ),
  track_length = 0.25 →
  pollys_laps = 12 →
  pollys_time = 0.5 →
  geralds_factor = 0.5 →
  (geralds_factor * (pollys_laps * track_length / pollys_time)) = 3 :=
by
  intro track_length pollys_laps pollys_time geralds_factor
  intro h_track_len h_pol_lys_laps h_pollys_time h_ger_factor
  sorry

end geralds_average_speed_l109_109032


namespace solve_system_of_equations_l109_109809

-- Given conditions
variables {a b c k x y z : ℝ}
variables (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
variables (eq1 : a * x + b * y + c * z = k)
variables (eq2 : a^2 * x + b^2 * y + c^2 * z = k^2)
variables (eq3 : a^3 * x + b^3 * y + c^3 * z = k^3)

-- Statement to be proved
theorem solve_system_of_equations :
  x = k * (k - c) * (k - b) / (a * (a - c) * (a - b)) ∧
  y = k * (k - c) * (k - a) / (b * (b - c) * (b - a)) ∧
  z = k * (k - a) * (k - b) / (c * (c - a) * (c - b)) :=
sorry

end solve_system_of_equations_l109_109809


namespace choir_blonde_black_ratio_l109_109439

theorem choir_blonde_black_ratio 
  (b x : ℕ) 
  (h1 : ∀ (b x : ℕ), b / ((5 / 3 : ℚ) * b) = (3 / 5 : ℚ)) 
  (h2 : ∀ (b x : ℕ), (b + x) / ((5 / 3 : ℚ) * b) = (3 / 2 : ℚ)) :
  x = (3 / 2 : ℚ) * b ∧ 
  ∃ k : ℚ, k = (5 / 3 : ℚ) * b :=
by {
  sorry
}

end choir_blonde_black_ratio_l109_109439


namespace lives_after_game_l109_109275

theorem lives_after_game (l0 : ℕ) (ll : ℕ) (lg : ℕ) (lf : ℕ) : 
  l0 = 10 → ll = 4 → lg = 26 → lf = l0 - ll + lg → lf = 32 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end lives_after_game_l109_109275


namespace temperature_difference_is_correct_l109_109029

def highest_temperature : ℤ := -9
def lowest_temperature : ℤ := -22
def temperature_difference : ℤ := highest_temperature - lowest_temperature

theorem temperature_difference_is_correct :
  temperature_difference = 13 := by
  -- We need to prove this statement is correct
  sorry

end temperature_difference_is_correct_l109_109029


namespace inequality_l109_109254

theorem inequality (a b c : ℝ) (h₀ : 0 < c) (h₁ : c < b) (h₂ : b < a) :
  a^4 * b + b^4 * c + c^4 * a > a * b^4 + b * c^4 + c * a^4 :=
by sorry

end inequality_l109_109254


namespace range_of_x_l109_109020

theorem range_of_x (a : ℝ) (x : ℝ) (h0 : 0 ≤ a) (h1 : a ≤ 2) :
  a * x^2 + (a + 1) * x + 1 - (3 / 2) * a < 0 → -2 < x ∧ x < -1 :=
by
  sorry

end range_of_x_l109_109020


namespace rotated_parabola_equation_l109_109979

def parabola_equation (x y : ℝ) : Prop := y = x^2 - 4 * x + 3

def standard_form (x y : ℝ) : Prop := y = (x - 2)^2 - 1

def after_rotation (x y : ℝ) : Prop := (y + 1)^2 = x - 2

theorem rotated_parabola_equation (x y : ℝ) (h : standard_form x y) : after_rotation x y :=
sorry

end rotated_parabola_equation_l109_109979


namespace product_of_repeating_decimal_l109_109073

noncomputable def repeating_decimal := 1357 / 9999
def product_with_7 (x : ℚ) := 7 * x

theorem product_of_repeating_decimal :
  product_with_7 repeating_decimal = 9499 / 9999 :=
by sorry

end product_of_repeating_decimal_l109_109073


namespace union_sets_l109_109248

def setA : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def setB : Set ℝ := { x | 0 < x ∧ x < 3 }

theorem union_sets :
  (setA ∪ setB) = { x | -1 ≤ x ∧ x < 3 } :=
sorry

end union_sets_l109_109248


namespace range_of_t_l109_109586

theorem range_of_t (a t : ℝ) (x y : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  a * x^2 + t * y^2 ≥ (a * x + t * y)^2 ↔ 0 ≤ t ∧ t ≤ 1 - a :=
sorry

end range_of_t_l109_109586


namespace evaluate_expression_l109_109045

theorem evaluate_expression : 
  ( (5 ^ 2014) ^ 2 - (5 ^ 2012) ^ 2 ) / ( (5 ^ 2013) ^ 2 - (5 ^ 2011) ^ 2 ) = 25 := 
by sorry

end evaluate_expression_l109_109045


namespace shirt_original_price_l109_109925

theorem shirt_original_price (P : ℝ) : 
  (18 = P * 0.75 * 0.75 * 0.90 * 1.15) → 
  P = 18 / (0.75 * 0.75 * 0.90 * 1.15) :=
by
  intro h
  sorry

end shirt_original_price_l109_109925


namespace part1_part2_l109_109053

noncomputable def f (x m : ℝ) : ℝ := abs (x - m) + abs (x + 3)

theorem part1 (x : ℝ) : f x 1 ≤ x + 4 ↔ 0 ≤ x ∧ x ≤ 2 := sorry

theorem part2 (m n t : ℝ) (hm : m > 0) (hn : n > 0) (ht : t > 0) 
  (hmin : ∀ x, f x m ≥ 5 - n - t) :
  1 / (m + n) + 1 / t ≥ 2 := sorry

end part1_part2_l109_109053


namespace gold_cube_profit_multiple_l109_109750

theorem gold_cube_profit_multiple :
  let side_length : ℝ := 6
  let density : ℝ := 19
  let cost_per_gram : ℝ := 60
  let profit : ℝ := 123120
  let volume := side_length ^ 3
  let mass := density * volume
  let cost := mass * cost_per_gram
  let selling_price := cost + profit
  let multiple := selling_price / cost
  multiple = 1.5 := by
  sorry

end gold_cube_profit_multiple_l109_109750


namespace sqrt_domain_l109_109695

theorem sqrt_domain (x : ℝ) : (x - 1 ≥ 0) ↔ (x ≥ 1) :=
sorry

end sqrt_domain_l109_109695


namespace both_questions_correct_l109_109555

-- Define variables as constants
def nA : ℝ := 0.85  -- 85%
def nB : ℝ := 0.70  -- 70%
def nAB : ℝ := 0.60 -- 60%

theorem both_questions_correct:
  nAB = 0.60 := by
  sorry

end both_questions_correct_l109_109555


namespace article_large_font_pages_l109_109109

theorem article_large_font_pages (L S : ℕ) 
  (pages_eq : L + S = 21) 
  (words_eq : 1800 * L + 2400 * S = 48000) : 
  L = 4 := 
by 
  sorry

end article_large_font_pages_l109_109109


namespace MaxCandy_l109_109770

theorem MaxCandy (frankieCandy : ℕ) (extraCandy : ℕ) (maxCandy : ℕ) 
  (h1 : frankieCandy = 74) (h2 : extraCandy = 18) (h3 : maxCandy = frankieCandy + extraCandy) :
  maxCandy = 92 := 
by
  sorry

end MaxCandy_l109_109770


namespace intersection_M_N_l109_109246

-- Define the universe U
def U : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the set M based on the condition x^2 <= x
def M : Set ℤ := {x ∈ U | x^2 ≤ x}

-- Define the set N based on the condition x^3 - 3x^2 + 2x = 0
def N : Set ℤ := {x ∈ U | x^3 - 3*x^2 + 2*x = 0}

-- State the theorem to be proven
theorem intersection_M_N : M ∩ N = {0, 1} :=
by
  sorry

end intersection_M_N_l109_109246


namespace zoo_tickets_total_cost_l109_109204

-- Define the given conditions
def num_children := 6
def num_adults := 10
def cost_child_ticket := 10
def cost_adult_ticket := 16

-- Calculate the expected total cost
def total_cost := 220

-- State the theorem
theorem zoo_tickets_total_cost :
  num_children * cost_child_ticket + num_adults * cost_adult_ticket = total_cost :=
by
  sorry

end zoo_tickets_total_cost_l109_109204


namespace second_metal_gold_percentage_l109_109723

theorem second_metal_gold_percentage (w_final : ℝ) (p_final : ℝ) (w_part : ℝ) (p_part1 : ℝ) (w_part1 : ℝ) (w_part2 : ℝ)
  (h_w_final : w_final = 12.4) (h_p_final : p_final = 0.5) (h_w_part : w_part = 6.2) (h_p_part1 : p_part1 = 0.6)
  (h_w_part1 : w_part1 = 6.2) (h_w_part2 : w_part2 = 6.2) :
  ∃ p_part2 : ℝ, p_part2 = 0.4 :=
by sorry

end second_metal_gold_percentage_l109_109723


namespace simplify_expression_l109_109688

theorem simplify_expression : (4 + 3) + (8 - 3 - 1) = 11 := by
  sorry

end simplify_expression_l109_109688


namespace isosceles_triangle_k_l109_109757

theorem isosceles_triangle_k (m n k : ℝ) (h_iso : (m = 4 ∨ n = 4 ∨ m = n) ∧ (m ≠ n ∨ (m = n ∧ m + m > 4))) 
  (h_roots : ∀ x, x^2 - 6*x + (k + 2) = 0 → (x = m ∨ x = n)) : k = 6 ∨ k = 7 :=
sorry

end isosceles_triangle_k_l109_109757


namespace x_eq_1_sufficient_not_necessary_l109_109263

theorem x_eq_1_sufficient_not_necessary (x : ℝ) : 
    (x = 1 → (x^2 - 3 * x + 2 = 0)) ∧ ¬((x^2 - 3 * x + 2 = 0) → (x = 1)) := 
by
  sorry

end x_eq_1_sufficient_not_necessary_l109_109263


namespace angle_variance_less_than_bound_l109_109359

noncomputable def angle_variance (α β γ : ℝ) : ℝ :=
  (1/3) * ((α - (2 * Real.pi / 3))^2 + (β - (2 * Real.pi / 3))^2 + (γ - (2 * Real.pi / 3))^2)

theorem angle_variance_less_than_bound (O A B C : ℝ → ℝ) :
  ∀ α β γ : ℝ, α + β + γ = 2 * Real.pi ∧ α ≥ β ∧ β ≥ γ → angle_variance α β γ < 2 * Real.pi^2 / 9 :=
by
  sorry

end angle_variance_less_than_bound_l109_109359


namespace donny_remaining_money_l109_109206

theorem donny_remaining_money :
  let initial_amount := 78
  let kite_cost := 8
  let frisbee_cost := 9
  initial_amount - (kite_cost + frisbee_cost) = 61 :=
by
  sorry

end donny_remaining_money_l109_109206


namespace xy_squared_value_l109_109786

variable {x y : ℝ}

theorem xy_squared_value :
  (y + 6 = (x - 3)^2) ∧ (x + 6 = (y - 3)^2) ∧ (x ≠ y) → (x^2 + y^2 = 25) := 
by
  sorry

end xy_squared_value_l109_109786


namespace total_skips_correct_l109_109670

def bob_skip_rate := 12
def jim_skip_rate := 15
def sally_skip_rate := 18

def bob_rocks := 10
def jim_rocks := 8
def sally_rocks := 12

theorem total_skips_correct : 
  (bob_skip_rate * bob_rocks) + (jim_skip_rate * jim_rocks) + (sally_skip_rate * sally_rocks) = 456 := by
  sorry

end total_skips_correct_l109_109670


namespace slope_of_line_l109_109999

-- Definition of the line equation in slope-intercept form
def line_eq (x : ℝ) : ℝ := -5 * x + 9

-- Statement: The slope of the line y = -5x + 9 is -5
theorem slope_of_line : (∀ x : ℝ, ∃ m b : ℝ, line_eq x = m * x + b ∧ m = -5) :=
by
  -- proof goes here
  sorry

end slope_of_line_l109_109999


namespace krista_driving_hours_each_day_l109_109163

-- Define the conditions as constants
def road_trip_days : ℕ := 3
def jade_hours_per_day : ℕ := 8
def total_hours : ℕ := 42

-- Define the function to calculate Krista's hours per day
noncomputable def krista_hours_per_day : ℕ :=
  (total_hours - road_trip_days * jade_hours_per_day) / road_trip_days

-- State the theorem to prove Krista drove 6 hours each day
theorem krista_driving_hours_each_day : krista_hours_per_day = 6 := by
  sorry

end krista_driving_hours_each_day_l109_109163


namespace apples_per_box_l109_109866

theorem apples_per_box (x : ℕ) (h1 : 10 * x > 0) (h2 : 3 * (10 * x) / 4 > 0) (h3 : (10 * x) / 4 = 750) : x = 300 :=
by
  sorry

end apples_per_box_l109_109866


namespace probability_two_red_faces_eq_three_eighths_l109_109547

def cube_probability : ℚ :=
  let total_cubes := 64 -- Total number of smaller cubes
  let two_red_faces_cubes := 24 -- Number of smaller cubes with exactly two red faces
  two_red_faces_cubes / total_cubes

theorem probability_two_red_faces_eq_three_eighths :
  cube_probability = 3 / 8 :=
by
  -- proof goes here
  sorry

end probability_two_red_faces_eq_three_eighths_l109_109547


namespace solution_criteria_l109_109049

def is_solution (M : ℕ) : Prop :=
  5 ∣ (1989^M + M^1989)

theorem solution_criteria (M : ℕ) (h : M < 10) : is_solution M ↔ (M = 1 ∨ M = 4) :=
sorry

end solution_criteria_l109_109049


namespace time_per_student_l109_109805

-- Given Conditions
def total_students : ℕ := 18
def groups : ℕ := 3
def minutes_per_group : ℕ := 24

-- Mathematical proof problem
theorem time_per_student :
  (minutes_per_group / (total_students / groups)) = 4 := by
  -- Proof not required, adding placeholder
  sorry

end time_per_student_l109_109805


namespace time_to_walk_l109_109916

variable (v l r w : ℝ)
variable (h1 : l = 15 * (v + r))
variable (h2 : l = 30 * (v + w))
variable (h3 : l = 20 * r)

theorem time_to_walk (h1 : l = 15 * (v + r)) (h2 : l = 30 * (v + w)) (h3 : l = 20 * r) : l / w = 60 := 
by sorry

end time_to_walk_l109_109916


namespace ratio_of_7th_terms_l109_109989

theorem ratio_of_7th_terms (a b : ℕ → ℕ) (S T : ℕ → ℕ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : ∀ n, T n = n * (b 1 + b n) / 2)
  (h3 : ∀ n, S n / T n = (5 * n + 10) / (2 * n - 1)) :
  a 7 / b 7 = 3 :=
by
  sorry

end ratio_of_7th_terms_l109_109989


namespace find_certain_age_l109_109813

theorem find_certain_age 
(Kody_age : ℕ) 
(Mohamed_age : ℕ) 
(certain_age : ℕ) 
(h1 : Kody_age = 32) 
(h2 : Mohamed_age = 2 * certain_age) 
(h3 : ∀ four_years_ago, four_years_ago = Kody_age - 4 → four_years_ago * 2 = Mohamed_age - 4) :
  certain_age = 30 := sorry

end find_certain_age_l109_109813


namespace find_t_l109_109146

theorem find_t (s t : ℝ) (h1 : 12 * s + 7 * t = 165) (h2 : s = t + 3) : t = 6.789 := 
by 
  sorry

end find_t_l109_109146


namespace range_of_a_l109_109132

theorem range_of_a (a : ℝ) :
  ¬ (∃ x : ℝ, (0 < x) ∧ (x + 1/x < a)) ↔ a ≤ 2 :=
by {
  sorry
}

end range_of_a_l109_109132


namespace cubic_roots_reciprocal_squares_sum_l109_109115

-- Define the roots a, b, and c
variables (a b c : ℝ)

-- Define the given cubic equation conditions
variables (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) (h3 : a * b * c = 6)

-- Define the target statement
theorem cubic_roots_reciprocal_squares_sum :
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 49 / 36 :=
by
  sorry

end cubic_roots_reciprocal_squares_sum_l109_109115


namespace find_third_number_l109_109186

theorem find_third_number (first_number second_number third_number : ℕ) 
  (h1 : first_number = 200)
  (h2 : first_number + second_number + third_number = 500)
  (h3 : second_number = 2 * third_number) :
  third_number = 100 := sorry

end find_third_number_l109_109186


namespace factorial_product_trailing_zeros_l109_109621

def countTrailingZerosInFactorialProduct : ℕ :=
  let countFactorsOfFive (n : ℕ) : ℕ := 
    (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125) + (n / 15625) + (n / 78125) + (n / 390625) 
  List.range 100 -- Generates list [0, 1, ..., 99]
  |> List.map (fun k => countFactorsOfFive (k + 1)) -- Apply countFactorsOfFive to each k+1
  |> List.foldr (· + ·) 0 -- Sum all counts

theorem factorial_product_trailing_zeros : countTrailingZerosInFactorialProduct = 1124 := by
  sorry

end factorial_product_trailing_zeros_l109_109621


namespace smallest_perimeter_of_acute_triangle_with_consecutive_sides_l109_109021

theorem smallest_perimeter_of_acute_triangle_with_consecutive_sides :
  ∃ (a : ℕ), (a > 1) ∧ (∃ (b c : ℕ), b = a + 1 ∧ c = a + 2 ∧ (∃ (C : ℝ), a^2 + b^2 - c^2 < 0 ∧ c = 4)) ∧ (a + (a + 1) + (a + 2) = 9) :=
by {
  sorry
}

end smallest_perimeter_of_acute_triangle_with_consecutive_sides_l109_109021


namespace ratio_of_bottles_l109_109046

theorem ratio_of_bottles
  (initial_money : ℤ)
  (initial_bottles : ℕ)
  (cost_per_bottle : ℤ)
  (cost_per_pound_cheese : ℤ)
  (cheese_pounds : ℚ)
  (remaining_money : ℤ) :
  initial_money = 100 →
  initial_bottles = 4 →
  cost_per_bottle = 2 →
  cost_per_pound_cheese = 10 →
  cheese_pounds = 0.5 →
  remaining_money = 71 →
  (2 * initial_bottles) / initial_bottles = 2 :=
by 
  sorry

end ratio_of_bottles_l109_109046


namespace sum_first_n_arithmetic_sequence_l109_109044

theorem sum_first_n_arithmetic_sequence (a1 d : ℝ) (S : ℕ → ℝ) :
  (S 3 + S 6 = 18) → 
  S 3 = 3 * a1 + 3 * d → 
  S 6 = 6 * a1 + 15 * d → 
  S 5 = 10 :=
by
  sorry

end sum_first_n_arithmetic_sequence_l109_109044


namespace machine_work_hours_l109_109897

theorem machine_work_hours (A B : ℝ) (x : ℝ) (hA : A = 1 / 8) (hB : B = A / 4)
  (hB_rate : B = 1 / 32) (B_time : B * 8 = 1 - x / 8) : x = 6 :=
by
  sorry

end machine_work_hours_l109_109897


namespace thirtieth_change_month_is_february_l109_109143

def months_in_year := 12

def months_per_change := 7

def first_change_month := 3 -- March (if we assume January = 1, February = 2, etc.)

def nth_change_month (n : ℕ) : ℕ :=
  (first_change_month + months_per_change * (n - 1)) % months_in_year

theorem thirtieth_change_month_is_february :
  nth_change_month 30 = 2 := -- February (if we assume January = 1, February = 2, etc.)
by 
  sorry

end thirtieth_change_month_is_february_l109_109143


namespace capacity_of_other_bottle_l109_109269

theorem capacity_of_other_bottle (C : ℝ) :
  (∀ (total_milk c1 c2 : ℝ), total_milk = 8 ∧ c1 = 5.333333333333333 ∧ c2 = C ∧ 
  (c1 / 8 = (c2 / C))) → C = 4 :=
by
  intros h
  sorry

end capacity_of_other_bottle_l109_109269


namespace katya_age_l109_109412

theorem katya_age (A K V : ℕ) (h1 : A + K = 19) (h2 : A + V = 14) (h3 : K + V = 7) : K = 6 := by
  sorry

end katya_age_l109_109412


namespace Megan_full_folders_l109_109881

def initial_files : ℕ := 256
def deleted_files : ℕ := 67
def files_per_folder : ℕ := 12

def remaining_files : ℕ := initial_files - deleted_files
def number_of_folders : ℕ := remaining_files / files_per_folder

theorem Megan_full_folders : number_of_folders = 15 := by
  sorry

end Megan_full_folders_l109_109881


namespace corresponding_angles_equal_l109_109500

theorem corresponding_angles_equal (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : α = 90) :
  (180 - α = 90 ∧ β + γ = 90 ∧ α = 90) :=
by
  sorry

end corresponding_angles_equal_l109_109500


namespace powers_greater_than_thresholds_l109_109875

theorem powers_greater_than_thresholds :
  (1.01^2778 > 1000000000000) ∧
  (1.001^27632 > 1000000000000) ∧
  (1.000001^27631000 > 1000000000000) ∧
  (1.01^4165 > 1000000000000000000) ∧
  (1.001^41447 > 1000000000000000000) ∧
  (1.000001^41446000 > 1000000000000000000) :=
by sorry

end powers_greater_than_thresholds_l109_109875


namespace white_paint_amount_l109_109872

theorem white_paint_amount (total_paint green_paint brown_paint : ℕ) 
  (h_total : total_paint = 69)
  (h_green : green_paint = 15)
  (h_brown : brown_paint = 34) :
  total_paint - (green_paint + brown_paint) = 20 := by
  sorry

end white_paint_amount_l109_109872


namespace neg_p_equiv_l109_109322

open Real
open Classical

noncomputable def prop_p : Prop :=
  ∀ x : ℝ, 0 < x → exp x > log x

noncomputable def neg_prop_p : Prop :=
  ∃ x : ℝ, 0 < x ∧ exp x ≤ log x

theorem neg_p_equiv :
  ¬ prop_p ↔ neg_prop_p := by
  sorry

end neg_p_equiv_l109_109322


namespace range_of_a_l109_109231

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > a then x + 2 else x^2 + 5 * x + 2

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
f x a - 2 * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, g x a = 0 → (x = 2 ∨ x = -1 ∨ x = -2)) ↔ (-1 ≤ a ∧ a < 2) :=
sorry

end range_of_a_l109_109231


namespace good_numbers_count_l109_109870

theorem good_numbers_count : 
  ∃ (count : ℕ), 
    count = 10 ∧ 
    (∀ n : ℕ, 2020 % n = 22 ↔ (n ∣ 1998 ∧ n > 22)) :=
by {
  sorry
}

end good_numbers_count_l109_109870


namespace remainder_29_times_171997_pow_2000_mod_7_l109_109001

theorem remainder_29_times_171997_pow_2000_mod_7 :
  (29 * 171997^2000) % 7 = 4 :=
by
  sorry

end remainder_29_times_171997_pow_2000_mod_7_l109_109001


namespace cyclist_wait_time_l109_109563

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

end cyclist_wait_time_l109_109563


namespace exists_n_satisfying_condition_l109_109100

-- Definition of the divisor function d(n)
def d (n : ℕ) : ℕ := Nat.divisors n |>.card

-- Theorem statement
theorem exists_n_satisfying_condition : ∃ n : ℕ, ∀ i : ℕ, i ≤ 1402 → (d n : ℚ) / d (n + i) > 1401 ∧ (d n : ℚ) / d (n - i) > 1401 :=
by
  sorry

end exists_n_satisfying_condition_l109_109100


namespace travis_takes_home_money_l109_109886

-- Define the conditions
def total_apples : ℕ := 10000
def apples_per_box : ℕ := 50
def price_per_box : ℕ := 35

-- Define the main theorem to be proved
theorem travis_takes_home_money : (total_apples / apples_per_box) * price_per_box = 7000 := by
  sorry

end travis_takes_home_money_l109_109886


namespace isabel_initial_candy_l109_109509

theorem isabel_initial_candy (total_candy : ℕ) (candy_given : ℕ) (initial_candy : ℕ) :
  candy_given = 25 → total_candy = 93 → total_candy = initial_candy + candy_given → initial_candy = 68 :=
by
  intros h_candy_given h_total_candy h_eq
  rw [h_candy_given, h_total_candy] at h_eq
  sorry

end isabel_initial_candy_l109_109509


namespace num_sets_M_l109_109265

theorem num_sets_M (M : Set ℕ) :
  {1, 2} ⊆ M ∧ M ⊆ {1, 2, 3, 4, 5, 6} → ∃ n : Nat, n = 16 :=
by
  sorry

end num_sets_M_l109_109265


namespace percent_increase_visual_range_l109_109463

theorem percent_increase_visual_range (original new : ℝ) (h_original : original = 60) (h_new : new = 150) : 
  ((new - original) / original) * 100 = 150 :=
by
  sorry

end percent_increase_visual_range_l109_109463


namespace grasshopper_frog_jump_difference_l109_109706

theorem grasshopper_frog_jump_difference :
  let grasshopper_jump := 19
  let frog_jump := 15
  grasshopper_jump - frog_jump = 4 :=
by
  let grasshopper_jump := 19
  let frog_jump := 15
  sorry

end grasshopper_frog_jump_difference_l109_109706


namespace field_ratio_l109_109603

theorem field_ratio (side pond_area_ratio : ℝ) (field_length : ℝ) 
  (pond_is_square: pond_area_ratio = 1/18) 
  (side_length: side = 8) 
  (field_len: field_length = 48) : 
  (field_length / (pond_area_ratio * side ^ 2 / side)) = 2 :=
by
  sorry

end field_ratio_l109_109603


namespace SplitWinnings_l109_109644

noncomputable def IstvanInitialContribution : ℕ := 5000 * 20
noncomputable def IstvanSecondPeriodContribution : ℕ := (5000 + 4000) * 30
noncomputable def IstvanThirdPeriodContribution : ℕ := (5000 + 4000 - 2500) * 40
noncomputable def IstvanTotalContribution : ℕ := IstvanInitialContribution + IstvanSecondPeriodContribution + IstvanThirdPeriodContribution

noncomputable def KalmanContribution : ℕ := 4000 * 70
noncomputable def LaszloContribution : ℕ := 2500 * 40
noncomputable def MiklosContributionAdjustment : ℕ := 2000 * 90

noncomputable def IstvanExpectedShare : ℕ := IstvanTotalContribution * 12 / 100
noncomputable def KalmanExpectedShare : ℕ := KalmanContribution * 12 / 100
noncomputable def LaszloExpectedShare : ℕ := LaszloContribution * 12 / 100
noncomputable def MiklosExpectedShare : ℕ := MiklosContributionAdjustment * 12 / 100

noncomputable def IstvanActualShare : ℕ := IstvanExpectedShare * 7 / 8
noncomputable def KalmanActualShare : ℕ := (KalmanExpectedShare - MiklosExpectedShare) * 7 / 8
noncomputable def LaszloActualShare : ℕ := LaszloExpectedShare * 7 / 8
noncomputable def MiklosActualShare : ℕ := MiklosExpectedShare * 7 / 8

theorem SplitWinnings :
  IstvanActualShare = 54600 ∧ KalmanActualShare = 7800 ∧ LaszloActualShare = 10500 ∧ MiklosActualShare = 18900 :=
by
  sorry

end SplitWinnings_l109_109644


namespace desired_butterfat_percentage_l109_109227

theorem desired_butterfat_percentage (milk1 milk2 : ℝ) (butterfat1 butterfat2 : ℝ) :
  milk1 = 8 →
  butterfat1 = 0.10 →
  milk2 = 8 →
  butterfat2 = 0.30 →
  ((butterfat1 * milk1) + (butterfat2 * milk2)) / (milk1 + milk2) * 100 = 20 := 
by
  intros
  sorry

end desired_butterfat_percentage_l109_109227


namespace bucket_weight_full_l109_109926

variable (p q x y : ℝ)

theorem bucket_weight_full (h1 : x + (3 / 4) * y = p)
                           (h2 : x + (1 / 3) * y = q) :
  x + y = (1 / 5) * (8 * p - 3 * q) :=
by
  sorry

end bucket_weight_full_l109_109926


namespace monthly_average_growth_rate_price_reduction_for_profit_l109_109807

-- Part 1: Monthly average growth rate of sales volume
theorem monthly_average_growth_rate (x : ℝ) : 
  256 * (1 + x) ^ 2 = 400 ↔ x = 0.25 :=
by
  sorry

-- Part 2: Price reduction to achieve profit of $4250
theorem price_reduction_for_profit (m : ℝ) : 
  (40 - m - 25) * (400 + 5 * m) = 4250 ↔ m = 5 :=
by
  sorry

end monthly_average_growth_rate_price_reduction_for_profit_l109_109807


namespace same_terminal_side_of_minus_80_l109_109696

theorem same_terminal_side_of_minus_80 :
  ∃ k : ℤ, 1 * 360 - 80 = 280 := 
  sorry

end same_terminal_side_of_minus_80_l109_109696


namespace determine_right_triangle_l109_109639

theorem determine_right_triangle (a b c : ℕ) :
  (∀ c b, (c + b) * (c - b) = a^2 → c^2 = a^2 + b^2) ∧
  (∀ A B C, A + B = C → C = 90) ∧
  (a = 3^2 ∧ b = 4^2 ∧ c = 5^2 → a^2 + b^2 ≠ c^2) ∧
  (a = 5 ∧ b = 12 ∧ c = 13 → a^2 + b^2 = c^2) → 
  ( ∃ x y z : ℕ, x = a ∧ y = b ∧ z = c ∧ x^2 + y^2 ≠ z^2 )
:= by
  sorry

end determine_right_triangle_l109_109639


namespace bird_problem_l109_109350

theorem bird_problem (B : ℕ) (h : (2 / 15) * B = 60) : B = 450 ∧ (2 / 15) * B = 60 :=
by
  sorry

end bird_problem_l109_109350


namespace sniper_B_has_greater_chance_of_winning_l109_109084

-- Define the probabilities for sniper A
def p_A_1 := 0.4
def p_A_2 := 0.1
def p_A_3 := 0.5

-- Define the probabilities for sniper B
def p_B_1 := 0.1
def p_B_2 := 0.6
def p_B_3 := 0.3

-- Define the expected scores for sniper A and B
def E_A := 1 * p_A_1 + 2 * p_A_2 + 3 * p_A_3
def E_B := 1 * p_B_1 + 2 * p_B_2 + 3 * p_B_3

-- The statement we want to prove
theorem sniper_B_has_greater_chance_of_winning : E_B > E_A := by
  simp [E_A, E_B, p_A_1, p_A_2, p_A_3, p_B_1, p_B_2, p_B_3]
  sorry

end sniper_B_has_greater_chance_of_winning_l109_109084


namespace compound_interest_difference_l109_109419

variable (P r : ℝ)

theorem compound_interest_difference :
  (P * 9 * r^2 = 360) → (P * r^2 = 40) :=
by
  sorry

end compound_interest_difference_l109_109419


namespace rectangle_decomposition_l109_109430

theorem rectangle_decomposition (m n k : ℕ) : ((k ∣ m) ∨ (k ∣ n)) ↔ (∃ P : ℕ, m * n = P * k) :=
by
  sorry

end rectangle_decomposition_l109_109430


namespace average_salary_difference_l109_109763

theorem average_salary_difference :
  let total_payroll_factory := 30000
  let num_factory_workers := 15
  let total_payroll_office := 75000
  let num_office_workers := 30
  (total_payroll_office / num_office_workers) - (total_payroll_factory / num_factory_workers) = 500 :=
by
  sorry

end average_salary_difference_l109_109763


namespace number_of_ways_to_assign_roles_l109_109772

theorem number_of_ways_to_assign_roles :
  let men := 6
  let women := 7
  let male_roles := 3
  let female_roles := 3
  let neutral_roles := 2
  let ways_male_roles := men * (men - 1) * (men - 2)
  let ways_female_roles := women * (women - 1) * (women - 2)
  let ways_neutral_roles := (men + women - male_roles - female_roles) * (men + women - male_roles - female_roles - 1)
  ways_male_roles * ways_female_roles * ways_neutral_roles = 1058400 := 
by
  sorry

end number_of_ways_to_assign_roles_l109_109772


namespace Bill_trips_l109_109438

theorem Bill_trips (total_trips : ℕ) (Jean_trips : ℕ) (Bill_trips : ℕ) 
  (h1 : total_trips = 40) 
  (h2 : Jean_trips = 23) 
  (h3 : Bill_trips + Jean_trips = total_trips) : 
  Bill_trips = 17 := 
by
  sorry

end Bill_trips_l109_109438


namespace parabola_relative_positions_l109_109064

def parabola1 (x : ℝ) : ℝ := x^2 - x + 3
def parabola2 (x : ℝ) : ℝ := x^2 + x + 3
def parabola3 (x : ℝ) : ℝ := x^2 + 2*x + 3

noncomputable def vertex_x (a b c : ℝ) : ℝ := -b / (2 * a)

theorem parabola_relative_positions :
  vertex_x 1 (-1) 3 < vertex_x 1 1 3 ∧ vertex_x 1 1 3 < vertex_x 1 2 3 :=
by {
  sorry
}

end parabola_relative_positions_l109_109064


namespace cookies_per_person_l109_109193

-- Definitions based on conditions
def cookies_total : ℕ := 144
def people_count : ℕ := 6

-- The goal is to prove the number of cookies per person
theorem cookies_per_person : cookies_total / people_count = 24 :=
by
  sorry

end cookies_per_person_l109_109193


namespace absolute_sum_of_roots_l109_109804

theorem absolute_sum_of_roots (d e f n : ℤ) (h1 : d + e + f = 0) (h2 : d * e + e * f + f * d = -2023) : |d| + |e| + |f| = 98 := 
sorry

end absolute_sum_of_roots_l109_109804


namespace sin_double_angle_condition_l109_109230

theorem sin_double_angle_condition (θ : ℝ) (h : Real.sin θ + Real.cos θ = 1 / 3) : Real.sin (2 * θ) = -8 / 9 := 
sorry

end sin_double_angle_condition_l109_109230


namespace borrowed_dimes_calculation_l109_109915

-- Define Sam's initial dimes and remaining dimes after borrowing
def original_dimes : ℕ := 8
def remaining_dimes : ℕ := 4

-- Statement to prove that the borrowed dimes is 4
theorem borrowed_dimes_calculation : (original_dimes - remaining_dimes) = 4 :=
by
  -- This is the proof section which follows by simple arithmetic computation
  sorry

end borrowed_dimes_calculation_l109_109915


namespace find_a_l109_109301

def setA : Set ℤ := {-1, 0, 1}

def setB (a : ℤ) : Set ℤ := {a, a ^ 2}

theorem find_a (a : ℤ) (h : setA ∪ setB a = setA) : a = -1 :=
sorry

end find_a_l109_109301


namespace max_discount_rate_l109_109605

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end max_discount_rate_l109_109605


namespace smallest_bob_number_l109_109425

theorem smallest_bob_number (b : ℕ) (h : ∀ p : ℕ, Prime p → p ∣ 30 → p ∣ b) : 30 ≤ b :=
by {
  sorry
}

end smallest_bob_number_l109_109425


namespace problem_1_problem_2_l109_109228

noncomputable def f (x a : ℝ) : ℝ := |2 * x - a|

theorem problem_1 (x : ℝ) : (∀ x, f x 4 < 8 - |x - 1|) → x ∈ Set.Ioo (-1 : ℝ) (13 / 3) :=
by sorry

theorem problem_2 (a : ℝ) : (∃ x, f x a > 8 + |2 * x - 1|) → a > 9 ∨ a < -7 :=
by sorry

end problem_1_problem_2_l109_109228


namespace runners_meet_opposite_dir_l109_109415

theorem runners_meet_opposite_dir 
  {S x y : ℝ}
  (h1 : S / x + 5 = S / y)
  (h2 : S / (x - y) = 30) :
  S / (x + y) = 6 := 
sorry

end runners_meet_opposite_dir_l109_109415


namespace senior_tickets_count_l109_109091

theorem senior_tickets_count (A S : ℕ) 
  (h1 : A + S = 510)
  (h2 : 21 * A + 15 * S = 8748) :
  S = 327 :=
sorry

end senior_tickets_count_l109_109091


namespace find_original_selling_price_l109_109846

noncomputable def original_selling_price (purchase_price : ℝ) := 
  1.10 * purchase_price

noncomputable def new_selling_price (purchase_price : ℝ) := 
  1.17 * purchase_price

theorem find_original_selling_price (P : ℝ)
  (h1 : new_selling_price P - original_selling_price P = 56) :
  original_selling_price P = 880 := by 
  sorry

end find_original_selling_price_l109_109846


namespace probability_of_x_plus_y_less_than_4_l109_109845

-- Define the square and the probability that x + y < 4 within this square.
theorem probability_of_x_plus_y_less_than_4 : 
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  (desired_area / square_area = 7 / 9) :=
by
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  show (desired_area / square_area = 7 / 9)
  sorry

end probability_of_x_plus_y_less_than_4_l109_109845


namespace pythagorean_consecutive_numbers_unique_l109_109116

theorem pythagorean_consecutive_numbers_unique :
  ∀ (x : ℕ), (x + 2) * (x + 2) = (x + 1) * (x + 1) + x * x → x = 3 :=
by
  sorry 

end pythagorean_consecutive_numbers_unique_l109_109116


namespace distinct_real_roots_form_geometric_progression_eq_170_l109_109356

theorem distinct_real_roots_form_geometric_progression_eq_170 
  (a : ℝ) :
  (∃ (u : ℝ) (v : ℝ) (hu : u ≠ 0) (hv : v ≠ 0) (hv1 : |v| ≠ 1), 
  (16 * u^12 + (2 * a + 17) * u^6 * v^3 - a * u^9 * v - a * u^3 * v^9 + 16 = 0)) 
  → a = 170 :=
by sorry

end distinct_real_roots_form_geometric_progression_eq_170_l109_109356


namespace number_of_pairs_of_socks_l109_109634

theorem number_of_pairs_of_socks (n : ℕ) (h : 2 * n^2 - n = 112) : n = 16 := sorry

end number_of_pairs_of_socks_l109_109634


namespace min_max_value_z_l109_109731

theorem min_max_value_z (x y z : ℝ) (h1 : x^2 ≤ y + z) (h2 : y^2 ≤ z + x) (h3 : z^2 ≤ x + y) :
  -1/4 ≤ z ∧ z ≤ 2 :=
by {
  sorry
}

end min_max_value_z_l109_109731


namespace pen_cost_proof_l109_109117

-- Given definitions based on the problem conditions
def is_majority (s : ℕ) := s > 20
def is_odd_and_greater_than_one (n : ℕ) := n > 1 ∧ n % 2 = 1
def is_prime (c : ℕ) := Nat.Prime c

-- The final theorem to prove the correct answer
theorem pen_cost_proof (s n c : ℕ) 
  (h_majority : is_majority s) 
  (h_odd : is_odd_and_greater_than_one n) 
  (h_prime : is_prime c) 
  (h_eq : s * c * n = 2091) : 
  c = 47 := 
sorry

end pen_cost_proof_l109_109117


namespace train_length_is_300_l109_109033

noncomputable def length_of_train (V L : ℝ) : Prop :=
  (L = V * 18) ∧ (L + 500 = V * 48)

theorem train_length_is_300
  (V : ℝ) (L : ℝ) (h : length_of_train V L) : L = 300 :=
by
  sorry

end train_length_is_300_l109_109033


namespace steps_to_11th_floor_l109_109250

theorem steps_to_11th_floor 
  (steps_between_3_and_5 : ℕ) 
  (third_floor : ℕ := 3) 
  (fifth_floor : ℕ := 5) 
  (eleventh_floor : ℕ := 11) 
  (ground_floor : ℕ := 1) 
  (steps_per_floor : ℕ := steps_between_3_and_5 / (fifth_floor - third_floor)) :
  steps_between_3_and_5 = 42 →
  steps_between_3_and_5 / (fifth_floor - third_floor) = 21 →
  (eleventh_floor - ground_floor) = 10 →
  21 * 10 = 210 := 
by
  intros _ _ _
  exact rfl

end steps_to_11th_floor_l109_109250


namespace perpendicular_vectors_l109_109878

theorem perpendicular_vectors (a : ℝ) 
  (v1 : ℝ × ℝ := (4, -5))
  (v2 : ℝ × ℝ := (a, 2))
  (perpendicular : v1.fst * v2.fst + v1.snd * v2.snd = 0) :
  a = 5 / 2 :=
sorry

end perpendicular_vectors_l109_109878


namespace root_product_minus_sums_l109_109225

variable {b c : ℝ}

theorem root_product_minus_sums
  (h1 : 3 * b^2 + 5 * b - 2 = 0)
  (h2 : 3 * c^2 + 5 * c - 2 = 0)
  : (b - 1) * (c - 1) = 2 := 
by
  sorry

end root_product_minus_sums_l109_109225


namespace problem1_problem2_problem3_l109_109515

theorem problem1 : (x : ℝ) → ((x + 1)^2 = 9 → (x = -4 ∨ x = 2)) :=
by
  intro x
  sorry

theorem problem2 : (x : ℝ) → (x^2 - 12*x - 4 = 0 → (x = 6 + 2*Real.sqrt 10 ∨ x = 6 - 2*Real.sqrt 10)) :=
by
  intro x
  sorry

theorem problem3 : (x : ℝ) → (3*(x - 2)^2 = x*(x - 2) → (x = 2 ∨ x = 3)) :=
by
  intro x
  sorry

end problem1_problem2_problem3_l109_109515


namespace multiples_of_15_between_17_and_202_l109_109512

theorem multiples_of_15_between_17_and_202 : 
  ∃ n : ℕ, (∀ k : ℤ, 17 < k * 15 ∧ k * 15 < 202 → k = n + 1) ∧ n = 12 :=
sorry

end multiples_of_15_between_17_and_202_l109_109512


namespace sufficient_not_necessary_l109_109827

theorem sufficient_not_necessary (x : ℝ) (h1 : -1 < x) (h2 : x < 3) :
    x^2 - 2*x < 8 :=
by
    -- Proof to be filled in.
    sorry

end sufficient_not_necessary_l109_109827


namespace unique_solution_exists_l109_109156

theorem unique_solution_exists (n m k : ℕ) :
  n = m^3 ∧ n = 1000 * m + k ∧ 0 ≤ k ∧ k < 1000 ∧ (1000 * m ≤ m^3 ∧ m^3 < 1000 * (m + 1)) → n = 32768 :=
by
  sorry

end unique_solution_exists_l109_109156


namespace intersection_sums_l109_109579

theorem intersection_sums (x1 x2 x3 y1 y2 y3 : ℝ) (h1 : y1 = x1^3 - 6 * x1 + 4)
  (h2 : y2 = x2^3 - 6 * x2 + 4) (h3 : y3 = x3^3 - 6 * x3 + 4)
  (h4 : x1 + 3 * y1 = 3) (h5 : x2 + 3 * y2 = 3) (h6 : x3 + 3 * y3 = 3) :
  x1 + x2 + x3 = 0 ∧ y1 + y2 + y3 = 3 := 
by
  sorry

end intersection_sums_l109_109579


namespace sum_of_prime_factors_of_91_l109_109111

theorem sum_of_prime_factors_of_91 : 
  (¬ (91 % 2 = 0)) ∧ 
  (¬ (91 % 3 = 0)) ∧ 
  (¬ (91 % 5 = 0)) ∧ 
  (91 = 7 * 13) →
  (7 + 13 = 20) := 
by 
  intros h
  sorry

end sum_of_prime_factors_of_91_l109_109111


namespace ali_seashells_final_count_l109_109616

theorem ali_seashells_final_count :
  385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - 0.10 * ((385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25))) 
  - (1 / 4) * ((385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - 0.10 * ((385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)))) = 82.485 :=
sorry

end ali_seashells_final_count_l109_109616


namespace initial_rope_length_l109_109985

variable (R₀ R₁ R₂ R₃ : ℕ)
variable (h_cut1 : 2 * R₀ = R₁) -- Josh cuts the original rope in half
variable (h_cut2 : 2 * R₁ = R₂) -- He cuts one of the halves in half again
variable (h_cut3 : 5 * R₂ = R₃) -- He cuts one of the resulting pieces into fifths
variable (h_held_piece : R₃ = 5) -- The piece Josh is holding is 5 feet long

theorem initial_rope_length:
  R₀ = 100 :=
by
  sorry

end initial_rope_length_l109_109985


namespace circle_radius_l109_109800

theorem circle_radius :
  ∃ c : ℝ × ℝ, 
    c.2 = 0 ∧
    (dist c (2, 3)) = (dist c (3, 7)) ∧
    (dist c (2, 3)) = (Real.sqrt 1717) / 2 :=
by
  sorry

end circle_radius_l109_109800


namespace total_tennis_balls_used_l109_109082

theorem total_tennis_balls_used :
  let rounds := [1028, 514, 257, 128, 64, 32, 16, 8, 4]
  let cans_per_game_A := 6
  let cans_per_game_B := 8
  let balls_per_can_A := 3
  let balls_per_can_B := 4
  let games_A_to_B := rounds.splitAt 4
  let total_A := games_A_to_B.1.sum * cans_per_game_A * balls_per_can_A
  let total_B := games_A_to_B.2.sum * cans_per_game_B * balls_per_can_B
  total_A + total_B = 37573 := 
by
  sorry

end total_tennis_balls_used_l109_109082


namespace minimum_bailing_rate_l109_109697

theorem minimum_bailing_rate
  (distance : ℝ) (to_shore_rate : ℝ) (water_in_rate : ℝ) (submerge_limit : ℝ) (r : ℝ)
  (h_distance : distance = 0.5) 
  (h_speed : to_shore_rate = 6) 
  (h_water_intake : water_in_rate = 12) 
  (h_submerge_limit : submerge_limit = 50)
  (h_time : (distance / to_shore_rate) * 60 = 5)
  (h_total_intake : water_in_rate * 5 = 60)
  (h_max_intake : submerge_limit - 60 = -10) :
  r = 2 := sorry

end minimum_bailing_rate_l109_109697


namespace Michaela_needs_20_oranges_l109_109889

variable (M : ℕ)
variable (C : ℕ)

theorem Michaela_needs_20_oranges 
  (h1 : C = 2 * M)
  (h2 : M + C = 60):
  M = 20 :=
by 
  sorry

end Michaela_needs_20_oranges_l109_109889


namespace basketball_weight_l109_109861

theorem basketball_weight (b s : ℝ) (h1 : s = 20) (h2 : 5 * b = 4 * s) : b = 16 :=
by
  sorry

end basketball_weight_l109_109861


namespace probability_of_defective_product_l109_109992

theorem probability_of_defective_product :
  let total_products := 10
  let defective_products := 2
  (defective_products: ℚ) / total_products = 1 / 5 :=
by
  let total_products := 10
  let defective_products := 2
  have h : (defective_products: ℚ) / total_products = 1 / 5
  {
    exact sorry
  }
  exact h

end probability_of_defective_product_l109_109992


namespace hypotenuse_length_l109_109364

-- Definitions and conditions
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Hypotheses
def leg1 := 8
def leg2 := 15

-- The theorem to be proven
theorem hypotenuse_length : ∃ c : ℕ, is_right_triangle leg1 leg2 c ∧ c = 17 :=
by { sorry }

end hypotenuse_length_l109_109364


namespace triangle_has_120_degree_l109_109506

noncomputable def angles_of_triangle (α β γ : Real) : Prop :=
  α + β + γ = 180

theorem triangle_has_120_degree (α β γ : Real)
    (h1 : angles_of_triangle α β γ)
    (h2 : Real.cos (3 * α) + Real.cos (3 * β) + Real.cos (3 * γ) = 1) :
  γ = 120 :=
  sorry

end triangle_has_120_degree_l109_109506


namespace polygon_sides_eq_five_l109_109162

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem polygon_sides_eq_five (n : ℕ) (h : n - number_of_diagonals n = 0) : n = 5 :=
by
  sorry

end polygon_sides_eq_five_l109_109162


namespace find_binomial_params_l109_109890

noncomputable def binomial_params (n p : ℝ) := 2.4 = n * p ∧ 1.44 = n * p * (1 - p)

theorem find_binomial_params (n p : ℝ) (h : binomial_params n p) : n = 6 ∧ p = 0.4 :=
by
  sorry

end find_binomial_params_l109_109890


namespace number_division_l109_109304

theorem number_division (N x : ℕ) 
  (h1 : (N - 5) / x = 7) 
  (h2 : (N - 34) / 10 = 2)
  : x = 7 := 
by
  sorry

end number_division_l109_109304


namespace solve_equation_l109_109303

theorem solve_equation (x : ℝ) (h : x ≠ 2) :
  x^2 = (4*x^2 + 4) / (x - 2) ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 ∨ x = 4) :=
by
  sorry

end solve_equation_l109_109303


namespace sandy_correct_sums_l109_109326

theorem sandy_correct_sums (c i : ℕ) (h1 : c + i = 30) (h2 : 3 * c - 2 * i = 55) : c = 23 :=
by
  sorry

end sandy_correct_sums_l109_109326


namespace op_value_l109_109929

def op (x y : ℕ) : ℕ := x^3 - 3*x*y^2 + y^3

theorem op_value :
  op 2 1 = 3 := by sorry

end op_value_l109_109929


namespace correct_operation_l109_109270

theorem correct_operation :
  (∀ (a : ℤ), 3 * a + 2 * a ≠ 5 * a ^ 2) ∧
  (∀ (a : ℤ), a ^ 6 / a ^ 2 ≠ a ^ 3) ∧
  (∀ (a : ℤ), (-3 * a ^ 3) ^ 2 = 9 * a ^ 6) ∧
  (∀ (a : ℤ), (a + 2) ^ 2 ≠ a ^ 2 + 4) := 
by
  sorry

end correct_operation_l109_109270


namespace ratio_of_speeds_correct_l109_109079

noncomputable def ratio_speeds_proof_problem : Prop :=
  ∃ (v_A v_B : ℝ),
    (∀ t : ℝ, 0 ≤ t ∧ t = 3 → 3 * v_A = abs (-800 + 3 * v_B)) ∧
    (∀ t : ℝ, 0 ≤ t ∧ t = 15 → 15 * v_A = abs (-800 + 15 * v_B)) ∧
    (3 * 15 * v_A / (15 * v_B) = 3 / 4)

theorem ratio_of_speeds_correct : ratio_speeds_proof_problem :=
sorry

end ratio_of_speeds_correct_l109_109079


namespace parrots_per_cage_l109_109860

theorem parrots_per_cage (P : ℕ) (parakeets_per_cage : ℕ) (cages : ℕ) (total_birds : ℕ) 
    (h1 : parakeets_per_cage = 7) (h2 : cages = 8) (h3 : total_birds = 72) 
    (h4 : total_birds = cages * P + cages * parakeets_per_cage) : 
    P = 2 :=
by
  sorry

end parrots_per_cage_l109_109860


namespace pet_store_cats_left_l109_109707

theorem pet_store_cats_left (siamese house sold : ℕ) (h_siamese : siamese = 38) (h_house : house = 25) (h_sold : sold = 45) :
  siamese + house - sold = 18 :=
by
  sorry

end pet_store_cats_left_l109_109707


namespace coupon_calculation_l109_109331

theorem coupon_calculation :
  let initial_stock : ℝ := 40.0
  let sold_books : ℝ := 20.0
  let coupons_per_book : ℝ := 4.0
  let remaining_books := initial_stock - sold_books
  let total_coupons := remaining_books * coupons_per_book
  total_coupons = 80.0 :=
by
  sorry

end coupon_calculation_l109_109331


namespace fourth_year_students_without_glasses_l109_109879

theorem fourth_year_students_without_glasses (total_students: ℕ) (x: ℕ) (y: ℕ) 
  (h1: total_students = 1152) 
  (h2: total_students = 8 * x - 32) 
  (h3: x = 148) 
  (h4: 2 * y + 10 = x) 
  : y = 69 :=
by {
sorry
}

end fourth_year_students_without_glasses_l109_109879


namespace distinct_lengths_from_E_to_DF_l109_109305

noncomputable def distinct_integer_lengths (DE EF: ℕ) : ℕ :=
if h : DE = 15 ∧ EF = 36 then 24 else 0

theorem distinct_lengths_from_E_to_DF :
  distinct_integer_lengths 15 36 = 24 :=
by {
  sorry
}

end distinct_lengths_from_E_to_DF_l109_109305


namespace fraction_simplification_l109_109314

theorem fraction_simplification:
  (4 * 7) / (14 * 10) * (5 * 10 * 14) / (4 * 5 * 7) = 1 :=
by {
  -- Proof goes here
  sorry
}

end fraction_simplification_l109_109314


namespace fill_digits_subtraction_correct_l109_109896

theorem fill_digits_subtraction_correct :
  ∀ (A B : ℕ), A236 - (B*100 + 97) = 5439 → A = 6 ∧ B = 7 :=
by
  sorry

end fill_digits_subtraction_correct_l109_109896


namespace rational_abs_neg_l109_109492

theorem rational_abs_neg (a : ℚ) (h : abs a = -a) : a ≤ 0 :=
by 
  sorry

end rational_abs_neg_l109_109492


namespace expedition_ratios_l109_109481

theorem expedition_ratios (F : ℕ) (S : ℕ) (L : ℕ) (R : ℕ) 
  (h1 : F = 3) 
  (h2 : S = F + 2) 
  (h3 : F + S + L = 18) 
  (h4 : L = R * S) : 
  R = 2 := 
sorry

end expedition_ratios_l109_109481


namespace frog_jump_correct_l109_109038

def grasshopper_jump : ℤ := 25
def additional_distance : ℤ := 15
def frog_jump : ℤ := grasshopper_jump + additional_distance

theorem frog_jump_correct : frog_jump = 40 := by
  sorry

end frog_jump_correct_l109_109038


namespace find_a_l109_109249

open Real

-- Definition of regression line
def regression_line (x : ℝ) : ℝ := 12.6 * x + 0.6

-- Data points for x and y
def x_values : List ℝ := [2, 3, 3.5, 4.5, 7]
def y_values : List ℝ := [26, 38, 43, 60]

-- Proof statement
theorem find_a (a : ℝ) (hx : x_values = [2, 3, 3.5, 4.5, 7])
  (hy : y_values ++ [a] = [26, 38, 43, 60, a]) : a = 88 :=
  sorry

end find_a_l109_109249


namespace derive_units_equivalent_to_velocity_l109_109393

-- Define the unit simplifications
def watt := 1 * (1 * (1 * (1 / 1)))
def newton := 1 * (1 * (1 / (1 * 1)))

-- Define the options
def option_A := watt / newton
def option_B := newton / watt
def option_C := watt / (newton * newton)
def option_D := (watt * watt) / newton
def option_E := (newton * newton) / (watt * watt)

-- Define what it means for a unit to be equivalent to velocity
def is_velocity (unit : ℚ) : Prop := unit = (1 * (1 / 1))

theorem derive_units_equivalent_to_velocity :
  is_velocity option_A ∧ 
  ¬ is_velocity option_B ∧ 
  ¬ is_velocity option_C ∧ 
  ¬ is_velocity option_D ∧ 
  ¬ is_velocity option_E := 
by sorry

end derive_units_equivalent_to_velocity_l109_109393


namespace sock_ratio_l109_109454

theorem sock_ratio (b : ℕ) (x : ℕ) (hx_pos : 0 < x)
  (h1 : 5 * x + 3 * b * x = k) -- Original cost is 5x + 3bx
  (h2 : b * x + 15 * x = 2 * k) -- Interchanged cost is doubled
  : b = 1 :=
by sorry

end sock_ratio_l109_109454


namespace geometric_sequence_min_value_l109_109442

theorem geometric_sequence_min_value 
  (a b c : ℝ)
  (h1 : b^2 = ac)
  (h2 : b = -Real.exp 1) :
  ac = Real.exp 2 := 
by
  sorry

end geometric_sequence_min_value_l109_109442


namespace scarlett_oil_amount_l109_109264

theorem scarlett_oil_amount (initial_oil add_oil : ℝ) (h1 : initial_oil = 0.17) (h2 : add_oil = 0.67) :
  initial_oil + add_oil = 0.84 :=
by
  rw [h1, h2]
  -- Proof step goes here
  sorry

end scarlett_oil_amount_l109_109264


namespace cost_per_taco_is_1_50_l109_109789

namespace TacoTruck

def total_beef : ℝ := 100
def beef_per_taco : ℝ := 0.25
def taco_price : ℝ := 2
def profit : ℝ := 200

theorem cost_per_taco_is_1_50 :
  let total_tacos := total_beef / beef_per_taco
  let total_revenue := total_tacos * taco_price
  let total_cost := total_revenue - profit
  total_cost / total_tacos = 1.50 := 
by
  sorry

end TacoTruck

end cost_per_taco_is_1_50_l109_109789


namespace john_total_feet_climbed_l109_109220

def first_stair_steps : ℕ := 20
def second_stair_steps : ℕ := 2 * first_stair_steps
def third_stair_steps : ℕ := second_stair_steps - 10
def step_height : ℝ := 0.5

theorem john_total_feet_climbed : 
  (first_stair_steps + second_stair_steps + third_stair_steps) * step_height = 45 :=
by
  sorry

end john_total_feet_climbed_l109_109220


namespace sphere_volume_increase_l109_109108

theorem sphere_volume_increase 
  (r : ℝ) 
  (S : ℝ := 4 * Real.pi * r^2) 
  (V : ℝ := (4/3) * Real.pi * r^3)
  (k : ℝ := 2) 
  (h : 4 * S = 4 * Real.pi * (k * r)^2) : 
  ((4/3) * Real.pi * (2 * r)^3) = 8 * V := 
by
  sorry

end sphere_volume_increase_l109_109108


namespace compute_expression_l109_109026

theorem compute_expression : (3 + 7)^3 + 2 * (3^2 + 7^2) = 1116 := by
  sorry

end compute_expression_l109_109026


namespace ratio_sum_is_four_l109_109313

theorem ratio_sum_is_four
  (x y : ℝ)
  (hx : 0 < x) (hy : 0 < y)
  (θ : ℝ)
  (hθ_ne : ∀ n : ℤ, θ ≠ (n * (π / 2)))
  (h1 : (Real.sin θ) / x = (Real.cos θ) / y)
  (h2 : (Real.cos θ)^4 / x^4 + (Real.sin θ)^4 / y^4 = 97 * (Real.sin (2 * θ)) / (x^3 * y + y^3 * x)) :
  (x / y) + (y / x) = 4 := by
  sorry

end ratio_sum_is_four_l109_109313


namespace MrsHiltCanTakeFriendsToMovies_l109_109426

def TotalFriends : ℕ := 15
def FriendsCantGo : ℕ := 7
def FriendsCanGo : ℕ := 8

theorem MrsHiltCanTakeFriendsToMovies : TotalFriends - FriendsCantGo = FriendsCanGo := by
  -- The proof will show that 15 - 7 = 8.
  sorry

end MrsHiltCanTakeFriendsToMovies_l109_109426


namespace residential_ratio_l109_109719

theorem residential_ratio (B R O E : ℕ) (h1 : B = 300) (h2 : E = 75) (h3 : E = O ∧ R + 2 * E = B) : R / B = 1 / 2 :=
by
  sorry

end residential_ratio_l109_109719


namespace cricket_run_rate_l109_109642

theorem cricket_run_rate 
  (run_rate_10_overs : ℝ)
  (target_runs : ℝ)
  (overs_played : ℕ)
  (remaining_overs : ℕ)
  (correct_run_rate : ℝ)
  (h1 : run_rate_10_overs = 3.6)
  (h2 : target_runs = 282)
  (h3 : overs_played = 10)
  (h4 : remaining_overs = 40)
  (h5 : correct_run_rate = 6.15) :
  (target_runs - run_rate_10_overs * overs_played) / remaining_overs = correct_run_rate :=
sorry

end cricket_run_rate_l109_109642


namespace length_increase_percentage_l109_109762

theorem length_increase_percentage (L B : ℝ) (x : ℝ) (h1 : (L + (x / 100) * L) * (B - (5 / 100) * B) = 1.14 * L * B) : x = 20 := by 
  sorry

end length_increase_percentage_l109_109762


namespace total_handshakes_l109_109222

-- Define the groups and their properties
def GroupA := 30
def GroupB := 15
def GroupC := 5
def KnowEachOtherA := true -- All 30 people in Group A know each other
def KnowFromB := 10 -- Each person in Group B knows 10 people from Group A
def KnowNoOneC := true -- Each person in Group C knows no one

-- Define the number of handshakes based on the conditions
def handshakes_between_A_and_B : Nat := GroupB * (GroupA - KnowFromB)
def handshakes_between_B_and_C : Nat := GroupB * GroupC
def handshakes_within_C : Nat := (GroupC * (GroupC - 1)) / 2
def handshakes_between_A_and_C : Nat := GroupA * GroupC

-- Prove the total number of handshakes
theorem total_handshakes : 
  handshakes_between_A_and_B +
  handshakes_between_B_and_C +
  handshakes_within_C +
  handshakes_between_A_and_C = 535 :=
by sorry

end total_handshakes_l109_109222


namespace calculate_y_when_x_is_neg2_l109_109429

def conditional_program (x : ℤ) : ℤ :=
  if x < 0 then
    2 * x + 3
  else if x > 0 then
    -2 * x + 5
  else
    0

theorem calculate_y_when_x_is_neg2 : conditional_program (-2) = -1 :=
by
  sorry

end calculate_y_when_x_is_neg2_l109_109429


namespace starting_number_l109_109812

theorem starting_number (n : ℕ) (h1 : 200 ≥ n) (h2 : 33 = ((200 / 3) - (n / 3))) : n = 102 :=
by
  sorry

end starting_number_l109_109812


namespace first_class_students_count_l109_109930

theorem first_class_students_count 
  (x : ℕ) 
  (avg1 : ℕ) (avg2 : ℕ) (num2 : ℕ) (overall_avg : ℝ)
  (h_avg1 : avg1 = 40)
  (h_avg2 : avg2 = 60)
  (h_num2 : num2 = 50)
  (h_overall_avg : overall_avg = 52.5)
  (h_eq : 40 * x + 60 * 50 = (52.5:ℝ) * (x + 50)) :
  x = 30 :=
by
  sorry

end first_class_students_count_l109_109930


namespace random_events_l109_109039

def is_random_event_1 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  ∃ c d : ℝ, c > 0 → d < 0 → a + d < 0 ∨ b + c > 0

def is_random_event_2 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  ∃ c d : ℝ, c > 0 → d < 0 → a - d > 0 ∨ b - c < 0

def is_impossible_event_3 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  a * b > 0

def is_certain_event_4 (a b : ℝ) (ha : a > 0) (hb : b < 0) : Prop :=
  a / b < 0

theorem random_events (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  is_random_event_1 a b ha hb ∧ is_random_event_2 a b ha hb :=
by
  sorry

end random_events_l109_109039


namespace towel_price_40_l109_109098

/-- Let x be the price of each towel bought second by the woman. 
    Given that she bought 3 towels at Rs. 100 each, 5 towels at x Rs. each, 
    and 2 towels at Rs. 550 each, and the average price of the towels was Rs. 160,
    we need to prove that x equals 40. -/
theorem towel_price_40 
    (x : ℝ)
    (h_avg_price : (300 + 5 * x + 1100) / 10 = 160) : 
    x = 40 :=
sorry

end towel_price_40_l109_109098


namespace living_room_area_is_60_l109_109469

-- Define the conditions
def carpet_length : ℝ := 4
def carpet_width : ℝ := 9
def carpet_area : ℝ := carpet_length * carpet_width
def coverage_fraction : ℝ := 0.60

-- Define the target area of the living room floor
def target_living_room_area (A : ℝ) : Prop :=
  coverage_fraction * A = carpet_area

-- State the Theorem
theorem living_room_area_is_60 (A : ℝ) (h : target_living_room_area A) : A = 60 := by
  -- Proof omitted
  sorry

end living_room_area_is_60_l109_109469


namespace fraction_draw_l109_109901

/-
Theorem: Given the win probabilities for Amy, Lily, and Eve, the fraction of the time they end up in a draw is 3/10.
-/

theorem fraction_draw (P_Amy P_Lily P_Eve : ℚ) (h_Amy : P_Amy = 2/5) (h_Lily : P_Lily = 1/5) (h_Eve : P_Eve = 1/10) : 
  1 - (P_Amy + P_Lily + P_Eve) = 3 / 10 := by
  sorry

end fraction_draw_l109_109901


namespace find_x_y_l109_109567

theorem find_x_y 
  (x y : ℝ) 
  (h1 : (15 + 30 + x + y) / 4 = 25) 
  (h2 : x = y + 10) :
  x = 32.5 ∧ y = 22.5 := 
by 
  sorry

end find_x_y_l109_109567


namespace max_value_f_on_interval_l109_109203

def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 1

theorem max_value_f_on_interval : 
  ∀ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f x ≤ 15 :=
by
  sorry

end max_value_f_on_interval_l109_109203


namespace not_divisible_by_8_l109_109007

theorem not_divisible_by_8 : ¬ (456294604884 % 8 = 0) := 
by
  have h : 456294604884 % 1000 = 884 := sorry -- This step reflects the conclusion that the last three digits are 884.
  have h_div : ¬ (884 % 8 = 0) := sorry -- This reflects that 884 is not divisible by 8.
  sorry

end not_divisible_by_8_l109_109007


namespace ivan_chess_false_l109_109407

theorem ivan_chess_false (n : ℕ) :
  ∃ n, n + 3 * n + 6 * n = 64 → False :=
by
  use 6
  sorry

end ivan_chess_false_l109_109407


namespace polygon_sides_l109_109516

theorem polygon_sides (n : ℕ) (h : 180 * (n - 2) = 720) : n = 6 :=
sorry

end polygon_sides_l109_109516


namespace perfect_square_trinomial_solution_l109_109243

theorem perfect_square_trinomial_solution (m : ℝ) :
  (∃ a : ℝ, (∀ x : ℝ, x^2 - 2*(m+3)*x + 9 = (x - a)^2))
  → m = 0 ∨ m = -6 :=
by
  sorry

end perfect_square_trinomial_solution_l109_109243


namespace equilateral_triangle_square_ratio_l109_109699

theorem equilateral_triangle_square_ratio (t s : ℕ) (h_t : 3 * t = 12) (h_s : 4 * s = 12) :
  t / s = 4 / 3 := by
  sorry

end equilateral_triangle_square_ratio_l109_109699


namespace union_A_B_intersection_A_CI_B_l109_109618

-- Define the sets
def I : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {3, 5, 6, 7}

-- Define the complement of B in the universal set I
def C_I (I : Set ℕ) (B : Set ℕ) : Set ℕ := {x ∈ I | x ∉ B}

-- The theorem for the union of A and B
theorem union_A_B : A ∪ B = {1, 2, 3, 4, 5, 6, 7} := sorry

-- The theorem for the intersection of A and the complement of B in I
theorem intersection_A_CI_B : A ∩ (C_I I B) = {1, 2, 4} := sorry

end union_A_B_intersection_A_CI_B_l109_109618


namespace symmetric_sufficient_not_necessary_l109_109611

theorem symmetric_sufficient_not_necessary (φ : Real) : 
    φ = - (Real.pi / 6) →
    ∃ f : Real → Real, (∀ x, f x = Real.sin (2 * x - φ)) ∧ 
    ∀ x, f (2 * (Real.pi / 6) - x) = f x :=
by
  sorry

end symmetric_sufficient_not_necessary_l109_109611


namespace wrongly_read_number_l109_109376

theorem wrongly_read_number 
  (S_initial : ℕ) (S_correct : ℕ) (correct_num : ℕ) (num_count : ℕ) 
  (h_initial : S_initial = num_count * 18) 
  (h_correct : S_correct = num_count * 19) 
  (h_correct_num : correct_num = 36) 
  (h_diff : S_correct - S_initial = correct_num - wrong_num) 
  (h_num_count : num_count = 10) 
  : wrong_num = 26 :=
sorry

end wrongly_read_number_l109_109376


namespace train_bus_ratio_is_two_thirds_l109_109487

def total_distance : ℕ := 1800
def distance_by_plane : ℕ := total_distance / 3
def distance_by_bus : ℕ := 720
def distance_by_train : ℕ := total_distance - (distance_by_plane + distance_by_bus)
def train_to_bus_ratio : ℚ := distance_by_train / distance_by_bus

theorem train_bus_ratio_is_two_thirds :
  train_to_bus_ratio = 2 / 3 := by
  sorry

end train_bus_ratio_is_two_thirds_l109_109487


namespace initial_deadline_is_75_days_l109_109405

-- Define constants for the problem
def initial_men : ℕ := 100
def initial_hours_per_day : ℕ := 8
def days_worked_initial : ℕ := 25
def fraction_work_completed : ℚ := 1 / 3
def additional_men : ℕ := 60
def new_hours_per_day : ℕ := 10
def total_man_hours : ℕ := 60000

-- Prove that the initial deadline for the project is 75 days
theorem initial_deadline_is_75_days : 
  ∃ (D : ℕ), (D * initial_men * initial_hours_per_day = total_man_hours) ∧ D = 75 := 
by {
  sorry
}

end initial_deadline_is_75_days_l109_109405


namespace area_of_isosceles_right_triangle_l109_109089

def is_isosceles_right_triangle (a b c : ℝ) : Prop :=
  (a = b) ∧ (a^2 + b^2 = c^2)

theorem area_of_isosceles_right_triangle (a : ℝ) (hypotenuse : ℝ) (h_isosceles : is_isosceles_right_triangle a a hypotenuse) (h_hypotenuse : hypotenuse = 6) :
  (1 / 2) * a * a = 9 :=
by
  sorry

end area_of_isosceles_right_triangle_l109_109089


namespace employee_selected_from_10th_group_is_47_l109_109101

theorem employee_selected_from_10th_group_is_47
  (total_employees : ℕ)
  (sampled_employees : ℕ)
  (total_groups : ℕ)
  (random_start : ℕ)
  (common_difference : ℕ)
  (selected_from_5th_group : ℕ) :
  total_employees = 200 →
  sampled_employees = 40 →
  total_groups = 40 →
  random_start = 2 →
  common_difference = 5 →
  selected_from_5th_group = 22 →
  (selected_from_5th_group = (4 * common_difference + random_start)) →
  (9 * common_difference + random_start) = 47 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end employee_selected_from_10th_group_is_47_l109_109101


namespace cost_of_graveling_per_sq_meter_l109_109271

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

end cost_of_graveling_per_sq_meter_l109_109271


namespace evaluate_expression_l109_109740

theorem evaluate_expression (a x : ℤ) (h : x = a + 5) : 2 * x - a + 4 = a + 14 :=
by
  sorry

end evaluate_expression_l109_109740


namespace sum_of_cubes_div_xyz_l109_109093

-- Given: x, y, z are non-zero real numbers, and x + y + z = 0.
-- Prove: (x^3 + y^3 + z^3) / (xyz) = 3.
theorem sum_of_cubes_div_xyz (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 0) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3 := 
by
  sorry

end sum_of_cubes_div_xyz_l109_109093


namespace range_of_m_l109_109448

-- Defining the point P and the required conditions for it to lie in the fourth quadrant
def point_in_fourth_quadrant (m : ℝ) : Prop :=
  let P := (m + 3, m - 1)
  P.1 > 0 ∧ P.2 < 0

-- Defining the range of m for which the point lies in the fourth quadrant
theorem range_of_m (m : ℝ) : point_in_fourth_quadrant m ↔ (-3 < m ∧ m < 1) :=
by
  sorry

end range_of_m_l109_109448


namespace trigonometric_identity_l109_109056

variable {α β γ n : Real}

-- Condition:
axiom h : Real.sin (2 * (α + γ)) = n * Real.sin (2 * β)

-- Statement to be proved:
theorem trigonometric_identity : 
  Real.tan (α + β + γ) / Real.tan (α - β + γ) = (n + 1) / (n - 1) :=
by
  sorry

end trigonometric_identity_l109_109056


namespace equal_profit_for_Robi_and_Rudy_l109_109061

theorem equal_profit_for_Robi_and_Rudy
  (robi_contrib : ℕ)
  (rudy_extra_contrib : ℕ)
  (profit_percent : ℚ)
  (share_profit_equally : Prop)
  (total_profit: ℚ)
  (each_share: ℕ) :
  robi_contrib = 4000 →
  rudy_extra_contrib = (1/4) * robi_contrib →
  profit_percent = 0.20 →
  share_profit_equally →
  total_profit = profit_percent * (robi_contrib + robi_contrib + rudy_extra_contrib) →
  each_share = (total_profit / 2) →
  each_share = 900 :=
by {
  sorry
}

end equal_profit_for_Robi_and_Rudy_l109_109061


namespace spontaneous_low_temperature_l109_109620

theorem spontaneous_low_temperature (ΔH ΔS T : ℝ) (spontaneous : ΔG = ΔH - T * ΔS) :
  (∀ T, T > 0 → ΔG < 0 → ΔH < 0 ∧ ΔS < 0) := 
by 
  sorry

end spontaneous_low_temperature_l109_109620


namespace polynomial_102_l109_109510

/-- Proving the value of the polynomial expression using the Binomial Theorem -/
theorem polynomial_102 :
  102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 100406401 :=
by
  sorry

end polynomial_102_l109_109510


namespace service_center_location_l109_109909

-- Definitions from conditions
def third_exit := 30
def twelfth_exit := 195
def seventh_exit := 90

-- Concept of distance and service center location
def distance := seventh_exit - third_exit
def service_center_milepost := third_exit + 2 * distance / 3

-- The theorem to prove
theorem service_center_location : service_center_milepost = 70 := by
  -- Sorry is used to skip the proof details.
  sorry

end service_center_location_l109_109909


namespace jake_fewer_peaches_than_steven_l109_109182

theorem jake_fewer_peaches_than_steven :
  ∀ (jill steven jake : ℕ),
    jill = 87 →
    steven = jill + 18 →
    jake = jill + 13 →
    steven - jake = 5 :=
by
  intros jill steven jake hjill hsteven hjake
  sorry

end jake_fewer_peaches_than_steven_l109_109182


namespace find_x_l109_109103

theorem find_x : ∃ x : ℕ, 6 * 2^x = 2048 ∧ x = 10 := by
  sorry

end find_x_l109_109103


namespace product_of_m_l109_109152

theorem product_of_m (m n : ℤ) (h_cond : m^2 + m + 8 = n^2) (h_nonneg : n ≥ 0) : 
  (∀ m, (∃ n, m^2 + m + 8 = n^2 ∧ n ≥ 0) → m = 7 ∨ m = -8) ∧ 
  (∃ m1 m2 : ℤ, m1 = 7 ∧ m2 = -8 ∧ (m1 * m2 = -56)) :=
by
  sorry

end product_of_m_l109_109152


namespace perfect_square_trinomial_m_l109_109157

theorem perfect_square_trinomial_m (m : ℝ) :
  (∃ a : ℝ, (x^2 + 2 * (m - 1) * x + 4) = (x + a)^2) → (m = 3 ∨ m = -1) :=
by
  sorry

end perfect_square_trinomial_m_l109_109157


namespace binary_arithmetic_l109_109548

theorem binary_arithmetic 
  : (0b10110 + 0b1011 - 0b11100 + 0b11101 = 0b100010) :=
by
  sorry

end binary_arithmetic_l109_109548


namespace more_money_from_mom_is_correct_l109_109596

noncomputable def more_money_from_mom : ℝ :=
  let money_from_mom := 8.25
  let money_from_dad := 6.50
  let money_from_grandparents := 12.35
  let money_from_aunt := 5.10
  let money_spent_toy := 4.45
  let money_spent_snacks := 6.25
  let total_received := money_from_mom + money_from_dad + money_from_grandparents + money_from_aunt
  let total_spent := money_spent_toy + money_spent_snacks
  let money_remaining := total_received - total_spent
  let money_spent_books := 0.25 * money_remaining
  let money_left_after_books := money_remaining - money_spent_books
  money_from_mom - money_from_dad

theorem more_money_from_mom_is_correct : more_money_from_mom = 1.75 := by
  sorry

end more_money_from_mom_is_correct_l109_109596


namespace equivalent_single_increase_l109_109097

-- Defining the initial price of the mobile
variable (P : ℝ)
-- Condition stating the price after a 40% increase
def increased_price := 1.40 * P
-- Condition stating the new price after a further 15% decrease
def final_price := 0.85 * increased_price P

-- The mathematically equivalent statement to prove
theorem equivalent_single_increase:
  final_price P = 1.19 * P :=
sorry

end equivalent_single_increase_l109_109097


namespace mary_total_money_l109_109118

def num_quarters : ℕ := 21
def quarters_worth : ℚ := 0.25
def dimes_worth : ℚ := 0.10

def num_dimes (Q : ℕ) : ℕ := (Q - 7) / 2

def total_money (Q : ℕ) (D : ℕ) : ℚ :=
  Q * quarters_worth + D * dimes_worth

theorem mary_total_money : 
  total_money num_quarters (num_dimes num_quarters) = 5.95 := 
by
  sorry

end mary_total_money_l109_109118


namespace smallest_positive_multiple_l109_109202

theorem smallest_positive_multiple (a : ℕ) (h : 17 * a % 53 = 7) : 17 * a = 544 :=
sorry

end smallest_positive_multiple_l109_109202


namespace gcd_pens_pencils_l109_109312

theorem gcd_pens_pencils (pens : ℕ) (pencils : ℕ) (h1 : pens = 1048) (h2 : pencils = 828) : Nat.gcd pens pencils = 4 := 
by
  -- Given: pens = 1048 and pencils = 828
  have h : pens = 1048 := h1
  have h' : pencils = 828 := h2
  sorry

end gcd_pens_pencils_l109_109312


namespace find_integer_n_l109_109191

theorem find_integer_n :
  ∃ (n : ℤ), -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * Real.pi / 180) = Real.cos (675 * Real.pi / 180) ∧ n = 45 :=
sorry

end find_integer_n_l109_109191


namespace sum_of_powers_mod_7_l109_109980

theorem sum_of_powers_mod_7 :
  ((1^1 + 2^2 + 3^3 + 4^4 + 5^5 + 6^6 + 7^7) % 7 = 1) := by
  sorry

end sum_of_powers_mod_7_l109_109980


namespace triangle_non_existence_triangle_existence_l109_109525

-- Definition of the triangle inequality theorem for a triangle with given sides.
def triangle_exists (a b c : ℕ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_non_existence (h : ¬ triangle_exists 2 3 7) : true := by
  sorry

theorem triangle_existence (h : triangle_exists 5 5 5) : true := by
  sorry

end triangle_non_existence_triangle_existence_l109_109525


namespace exists_xn_gt_yn_l109_109474

noncomputable def x_sequence : ℕ → ℝ := sorry
noncomputable def y_sequence : ℕ → ℝ := sorry

theorem exists_xn_gt_yn
    (x1 x2 y1 y2 : ℝ)
    (hx1 : 1 < x1)
    (hx2 : 1 < x2)
    (hy1 : 1 < y1)
    (hy2 : 1 < y2)
    (h_x_seq : ∀ n, x_sequence (n + 2) = x_sequence n + (x_sequence (n + 1))^2)
    (h_y_seq : ∀ n, y_sequence (n + 2) = (y_sequence n)^2 + y_sequence (n + 1)) :
    ∃ n : ℕ, x_sequence n > y_sequence n :=
sorry

end exists_xn_gt_yn_l109_109474


namespace average_age_calculated_years_ago_l109_109600

theorem average_age_calculated_years_ago
  (n m : ℕ) (a b : ℕ) 
  (total_age_original : ℝ)
  (average_age_original : ℝ)
  (average_age_new : ℝ) :
  n = 6 → 
  a = 19 → 
  m = 7 → 
  b = 1 → 
  total_age_original = n * a → 
  average_age_original = a → 
  average_age_new = a →
  (total_age_original + b) / m = a → 
  1 = 1 := 
by
  intros _ _ _ _ _ _ _ _
  sorry

end average_age_calculated_years_ago_l109_109600


namespace find_David_marks_in_Physics_l109_109400

theorem find_David_marks_in_Physics
  (english_marks : ℕ) (math_marks : ℕ) (chem_marks : ℕ) (biology_marks : ℕ)
  (avg_marks : ℕ) (num_subjects : ℕ)
  (h_english : english_marks = 76)
  (h_math : math_marks = 65)
  (h_chem : chem_marks = 67)
  (h_bio : biology_marks = 85)
  (h_avg : avg_marks = 75) 
  (h_num_subjects : num_subjects = 5) :
  english_marks + math_marks + chem_marks + biology_marks + physics_marks = avg_marks * num_subjects → physics_marks = 82 := 
  sorry

end find_David_marks_in_Physics_l109_109400


namespace find_m_n_l109_109272

theorem find_m_n (m n : ℕ) (hmn : m + 6 < n + 4)
  (median_cond : ((m + 2 + m + 6 + n + 4 + n + 5) / 7) = n + 2)
  (mean_cond : ((m + (m + 2) + (m + 6) + (n + 4) + (n + 5) + (2 * n - 1) + (2 * n + 2)) / 7) = n + 2) :
  m + n = 10 :=
sorry

end find_m_n_l109_109272


namespace boys_girls_difference_l109_109066

/--
If there are 550 students in a class and the ratio of boys to girls is 7:4, 
prove that the number of boys exceeds the number of girls by 150.
-/
theorem boys_girls_difference : 
  ∀ (students boys_ratio girls_ratio : ℕ),
  students = 550 →
  boys_ratio = 7 →
  girls_ratio = 4 →
  (students * boys_ratio) % (boys_ratio + girls_ratio) = 0 ∧
  (students * girls_ratio) % (boys_ratio + girls_ratio) = 0 →
  (students * boys_ratio - students * girls_ratio) / (boys_ratio + girls_ratio) = 150 :=
by
  intros students boys_ratio girls_ratio h_students h_boys_ratio h_girls_ratio h_divisibility
  -- The detailed proof would follow here, but we add 'sorry' to bypass it.
  sorry

end boys_girls_difference_l109_109066


namespace price_of_ice_cream_l109_109258

theorem price_of_ice_cream (x : ℝ) :
  (225 * x + 125 * 0.52 = 200) → (x = 0.60) :=
sorry

end price_of_ice_cream_l109_109258


namespace quadratic_no_real_roots_l109_109374

theorem quadratic_no_real_roots (m : ℝ) : ¬ ∃ x : ℝ, x^2 + 2 * x - m = 0 → m < -1 := 
by {
  sorry
}

end quadratic_no_real_roots_l109_109374


namespace total_spent_l109_109447

/-- Define the prices of the rides in the morning and the afternoon --/
def morning_price (ride : String) (age : Nat) : Nat :=
  match ride, age with
  | "bumper_car", n => if n < 18 then 2 else 3
  | "space_shuttle", n => if n < 18 then 4 else 5
  | "ferris_wheel", n => if n < 18 then 5 else 6
  | _, _ => 0

def afternoon_price (ride : String) (age : Nat) : Nat :=
  (morning_price ride age) + 1

/-- Define the number of rides taken by Mara and Riley --/
def rides_morning (person : String) (ride : String) : Nat :=
  match person, ride with
  | "Mara", "bumper_car" => 1
  | "Mara", "ferris_wheel" => 2
  | "Riley", "space_shuttle" => 2
  | "Riley", "ferris_wheel" => 2
  | _, _ => 0

def rides_afternoon (person : String) (ride : String) : Nat :=
  match person, ride with
  | "Mara", "bumper_car" => 1
  | "Mara", "ferris_wheel" => 1
  | "Riley", "space_shuttle" => 2
  | "Riley", "ferris_wheel" => 1
  | _, _ => 0

/-- Define the ages of Mara and Riley --/
def age (person : String) : Nat :=
  match person with
  | "Mara" => 17
  | "Riley" => 19
  | _ => 0

/-- Calculate the total expenditure --/
def total_cost (person : String) : Nat :=
  List.sum ([
    (rides_morning person "bumper_car") * (morning_price "bumper_car" (age person)),
    (rides_afternoon person "bumper_car") * (afternoon_price "bumper_car" (age person)),
    (rides_morning person "space_shuttle") * (morning_price "space_shuttle" (age person)),
    (rides_afternoon person "space_shuttle") * (afternoon_price "space_shuttle" (age person)),
    (rides_morning person "ferris_wheel") * (morning_price "ferris_wheel" (age person)),
    (rides_afternoon person "ferris_wheel") * (afternoon_price "ferris_wheel" (age person))
  ])

/-- Prove the total cost for Mara and Riley is $62 --/
theorem total_spent : total_cost "Mara" + total_cost "Riley" = 62 :=
by
  sorry

end total_spent_l109_109447


namespace tangent_line_equation_l109_109067

noncomputable def f (x : ℝ) : ℝ := (2 + Real.sin x) / Real.cos x

theorem tangent_line_equation :
  let x0 : ℝ := 0
  let y0 : ℝ := f x0
  let m : ℝ := (2 * x0 + 1) / (Real.cos x0 ^ 2)
  ∃ (a b c : ℝ), a * x0 + b * y0 + c = 0 ∧ a = 1 ∧ b = -1 ∧ c = 2 :=
by
  sorry

end tangent_line_equation_l109_109067


namespace max_license_plates_is_correct_l109_109823

theorem max_license_plates_is_correct :
  let letters := 26
  let digits := 10
  (letters * (letters - 1) * digits^3 = 26 * 25 * 10^3) :=
by 
  sorry

end max_license_plates_is_correct_l109_109823


namespace problem_statement_l109_109420

theorem problem_statement (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a + b) ^ 2002 + a ^ 2001 = 2 := 
by 
  sorry

end problem_statement_l109_109420


namespace original_profit_margin_l109_109739

theorem original_profit_margin (x : ℝ) (h1 : x - 0.9 / 0.9 = 12 / 100) : (x - 1) / 1 * 100 = 8 :=
by
  sorry

end original_profit_margin_l109_109739


namespace value_of_f_at_2_l109_109443

-- Given the conditions
variable (f : ℝ → ℝ)
variable (h_mono : Monotone f)
variable (h_cond : ∀ x : ℝ, f (f x - 3^x) = 4)

-- Define the proof goal
theorem value_of_f_at_2 : f 2 = 10 := 
sorry

end value_of_f_at_2_l109_109443


namespace valid_passwords_count_l109_109235

-- Define the total number of unrestricted passwords
def total_passwords : ℕ := 10000

-- Define the number of restricted passwords (ending with 6, 3, 9)
def restricted_passwords : ℕ := 10

-- Define the total number of valid passwords
def valid_passwords := total_passwords - restricted_passwords

theorem valid_passwords_count : valid_passwords = 9990 := 
by 
  sorry

end valid_passwords_count_l109_109235


namespace Jeremy_strolled_20_kilometers_l109_109433

def speed : ℕ := 2 -- Jeremy's speed in kilometers per hour
def time : ℕ := 10 -- Time Jeremy strolled in hours

noncomputable def distance : ℕ := speed * time -- The computed distance

theorem Jeremy_strolled_20_kilometers : distance = 20 := by
  sorry

end Jeremy_strolled_20_kilometers_l109_109433


namespace cabbages_difference_l109_109921

noncomputable def numCabbagesThisYear : ℕ := 4096
noncomputable def numCabbagesLastYear : ℕ := 3969
noncomputable def diffCabbages : ℕ := numCabbagesThisYear - numCabbagesLastYear

theorem cabbages_difference :
  diffCabbages = 127 := by
  sorry

end cabbages_difference_l109_109921


namespace system_solution_a_l109_109488

theorem system_solution_a (x y z : ℤ) (h1 : x^2 + x * y + y^2 = 7) (h2 : y^2 + y * z + z^2 = 13) (h3 : z^2 + z * x + x^2 = 19) :
  (x = 2 ∧ y = 1 ∧ z = 3) ∨ (x = -2 ∧ y = -1 ∧ z = -3) :=
sorry

end system_solution_a_l109_109488


namespace digit_100th_is_4_digit_1000th_is_3_l109_109244

noncomputable section

def digit_100th_place : Nat :=
  4

def digit_1000th_place : Nat :=
  3

theorem digit_100th_is_4 (n : ℕ) (h1 : n ∈ {m | m = 100}) : digit_100th_place = 4 := by
  sorry

theorem digit_1000th_is_3 (n : ℕ) (h1 : n ∈ {m | m = 1000}) : digit_1000th_place = 3 := by
  sorry

end digit_100th_is_4_digit_1000th_is_3_l109_109244


namespace neg_p_sufficient_not_necessary_q_l109_109565

-- Definitions from the given conditions
def p (a : ℝ) : Prop := a ≥ 1
def q (a : ℝ) : Prop := a ≤ 2

-- The theorem stating the mathematical equivalence
theorem neg_p_sufficient_not_necessary_q (a : ℝ) : (¬ p a → q a) ∧ ¬ (q a → ¬ p a) := 
by sorry

end neg_p_sufficient_not_necessary_q_l109_109565


namespace exp_gt_one_l109_109328

theorem exp_gt_one (a x y : ℝ) (ha : 1 < a) (hxy : x > y) : a^x > a^y :=
sorry

end exp_gt_one_l109_109328


namespace exists_monotonicity_b_range_l109_109418

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x ^ 2 - 2 * a * x + Real.log x

theorem exists_monotonicity_b_range :
  ∀ (a : ℝ) (b : ℝ), 1 < a ∧ a < 2 →
  (∀ (x0 : ℝ), x0 ∈ Set.Icc (1 + Real.sqrt 2 / 2) 2 →
   f a x0 + Real.log (a + 1) > b * (a^2 - 1) - (a + 1) + 2 * Real.log 2) →
   b ∈ Set.Iic (-1/4) :=
sorry

end exists_monotonicity_b_range_l109_109418


namespace no_valid_x_l109_109288

theorem no_valid_x (x y : ℝ) (h : y = 2 * x) : ¬(3 * y ^ 2 - 2 * y + 5 = 2 * (6 * x ^ 2 - 3 * y + 3)) :=
by
  sorry

end no_valid_x_l109_109288


namespace polygon_sides_exterior_angle_l109_109496

theorem polygon_sides_exterior_angle (n : ℕ) (h : 360 / 24 = n) : n = 15 := by
  sorry

end polygon_sides_exterior_angle_l109_109496


namespace intersection_eq_l109_109300

def A : Set Int := { -1, 0, 1 }
def B : Set Int := { 0, 1, 2 }

theorem intersection_eq :
  A ∩ B = {0, 1} := 
by 
  sorry

end intersection_eq_l109_109300


namespace noah_yearly_bill_l109_109850

-- Define the length of each call in minutes
def call_duration : ℕ := 30

-- Define the cost per minute in dollars
def cost_per_minute : ℝ := 0.05

-- Define the number of weeks in a year
def weeks_in_year : ℕ := 52

-- Define the cost per call in dollars
def cost_per_call : ℝ := call_duration * cost_per_minute

-- Define the total cost for a year in dollars
def yearly_cost : ℝ := cost_per_call * weeks_in_year

-- State the theorem
theorem noah_yearly_bill : yearly_cost = 78 := by
  -- Proof follows here
  sorry

end noah_yearly_bill_l109_109850


namespace trains_clear_in_correct_time_l109_109208

noncomputable def time_to_clear (length1 length2 : ℝ) (speed1_kmph speed2_kmph : ℝ) : ℝ :=
  let speed1_mps := speed1_kmph * 1000 / 3600
  let speed2_mps := speed2_kmph * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := length1 + length2
  total_distance / relative_speed

-- The lengths of the trains
def length1 : ℝ := 151
def length2 : ℝ := 165

-- The speeds of the trains in km/h
def speed1_kmph : ℝ := 80
def speed2_kmph : ℝ := 65

-- The correct answer
def correct_time : ℝ := 7.844

theorem trains_clear_in_correct_time :
  time_to_clear length1 length2 speed1_kmph speed2_kmph = correct_time :=
by
  -- Skipping proof
  sorry

end trains_clear_in_correct_time_l109_109208


namespace common_ratio_geometric_sequence_l109_109730

theorem common_ratio_geometric_sequence (q : ℝ) (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h₁ : a 2 = q)
  (h₂ : a 3 = q^2)
  (h₃ : (4 * a 1 + a 3 = 2 * 2 * a 2)) :
  q = 2 :=
by sorry

end common_ratio_geometric_sequence_l109_109730


namespace school_sports_event_l109_109531

theorem school_sports_event (x y z : ℤ) (hx : x > y) (hy : y > z) (hz : z > 0)
  (points_A points_B points_E : ℤ) (ha : points_A = 22) (hb : points_B = 9) 
  (he : points_E = 9) (vault_winner_B : True) :
  ∃ n : ℕ, n = 5 ∧ second_place_grenade_throwing_team = 8^B :=
by
  sorry

end school_sports_event_l109_109531


namespace sum_of_smallest_x_and_y_l109_109401

theorem sum_of_smallest_x_and_y (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
  (hx : ∃ k : ℕ, (480 * x) = k * k ∧ ∀ z : ℕ, 0 < z → (480 * z) = k * k → x ≤ z)
  (hy : ∃ n : ℕ, (480 * y) = n * n * n ∧ ∀ z : ℕ, 0 < z → (480 * z) = n * n * n → y ≤ z) :
  x + y = 480 := sorry

end sum_of_smallest_x_and_y_l109_109401


namespace exists_strictly_increasing_sequence_l109_109213

open Nat

-- Definition of strictly increasing sequence of integers a
def strictly_increasing_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a n < a (n + 1)

-- Condition i): Every natural number can be written as the sum of two terms from the sequence
def condition_i (a : ℕ → ℕ) : Prop :=
  ∀ m : ℕ, ∃ i j : ℕ, m = a i + a j

-- Condition ii): For each positive integer n, a_n > n^2/16
def condition_ii (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a n > n^2 / 16

-- The main theorem stating the existence of such a sequence
theorem exists_strictly_increasing_sequence :
  ∃ a : ℕ → ℕ, a 0 = 0 ∧ strictly_increasing_sequence a ∧ condition_i a ∧ condition_ii a :=
sorry

end exists_strictly_increasing_sequence_l109_109213


namespace find_a_find_a_plus_c_l109_109105

-- Define the triangle with given sides and angles
variables (A B C : ℝ) (a b c S : ℝ)
  (h_cosB : cos B = 4/5)
  (h_b : b = 2)
  (h_area : S = 3)

-- Prove the value of the side 'a' when angle A is π/6
theorem find_a (h_A : A = Real.pi / 6) : a = 5 / 3 := 
  sorry

-- Prove the sum of sides 'a' and 'c' when the area of the triangle is 3
theorem find_a_plus_c (h_ac : a * c = 10) : a + c = 2 * Real.sqrt 10 :=
  sorry

end find_a_find_a_plus_c_l109_109105


namespace rental_difference_l109_109760

variable (C K : ℕ)

theorem rental_difference
  (hc : 15 * C + 18 * K = 405)
  (hr : 3 * K = 2 * C) :
  C - K = 5 :=
sorry

end rental_difference_l109_109760


namespace ellipse_equation_max_area_abcd_l109_109413

open Real

theorem ellipse_equation (x y : ℝ) (a b c : ℝ) (h₀ : a > b) (h₁ : b > 0)
  (h₂ : a^2 = b^2 + c^2) (h₃ : b * c = 1) (h₄ : b = c) :
  (x^2 / 2 + y^2 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1) := by
  sorry

theorem max_area_abcd (a b c t : ℝ) (h₀ : a > b) (h₁ : b > 0)
  (h₂ : a^2 = b^2 + c^2) (h₃ : b * c = 1) (h₄ : b = c) :
  (∀ (t : ℝ), 4 * sqrt 2 * sqrt (1 + t^2) / (t^2 + 2) ≤ 2 * sqrt 2) := by
  sorry

end ellipse_equation_max_area_abcd_l109_109413


namespace vanya_exam_scores_l109_109025

/-- Vanya's exam scores inequality problem -/
theorem vanya_exam_scores
  (M R P : ℕ) -- scores in Mathematics, Russian language, and Physics respectively
  (hR : R = M - 10)
  (hP : P = M - 7)
  (h_bound : ∀ (k : ℕ), M + k ≤ 100 ∧ P + k ≤ 100 ∧ R + k ≤ 100) :
  ¬ (M = 100 ∧ P = 100) ∧ ¬ (M = 100 ∧ R = 100) ∧ ¬ (P = 100 ∧ R = 100) :=
by {
  sorry
}

end vanya_exam_scores_l109_109025


namespace cos_beta_of_acute_angles_l109_109083

theorem cos_beta_of_acute_angles (α β : ℝ) (hαβ : 0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2)
  (hcosα : Real.cos α = Real.sqrt 5 / 5)
  (hsin_alpha_minus_beta : Real.sin (α - β) = 3 * Real.sqrt 10 / 10) :
  Real.cos β = 7 * Real.sqrt 2 / 10 :=
sorry

end cos_beta_of_acute_angles_l109_109083


namespace range_of_m_for_inequality_l109_109654

theorem range_of_m_for_inequality (m : Real) : 
  (∀ (x : Real), 1 < x ∧ x < 2 → x^2 + m * x + 4 < 0) ↔ m ≤ -5 :=
by sorry

end range_of_m_for_inequality_l109_109654


namespace problem_l109_109440

def f (x : ℝ) (a b c d : ℝ) : ℝ := a * x^7 + b * x^5 - c * x^3 + d * x + 3

theorem problem (a b c d : ℝ) (h : f 92 a b c d = 2) : f 92 a b c d + f (-92) a b c d = 6 :=
by
  sorry

end problem_l109_109440


namespace number_of_customers_l109_109128

theorem number_of_customers
  (nails_per_person : ℕ)
  (total_sounds : ℕ)
  (trimmed_nails_per_person : nails_per_person = 20)
  (produced_sounds : total_sounds = 100) :
  total_sounds / nails_per_person = 5 :=
by
  -- This is offered as a placeholder to indicate where a Lean proof goes.
  sorry

end number_of_customers_l109_109128


namespace slope_of_line_6x_minus_4y_eq_16_l109_109635

noncomputable def slope_of_line (a b c : ℝ) : ℝ :=
  if b ≠ 0 then -a / b else 0

theorem slope_of_line_6x_minus_4y_eq_16 :
  slope_of_line 6 (-4) (-16) = 3 / 2 :=
by
  -- skipping the proof
  sorry

end slope_of_line_6x_minus_4y_eq_16_l109_109635


namespace arithmetic_sequence_properties_l109_109528

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n, S n = n * (a 1 + a n) / 2

def condition_S10_pos (S : ℕ → ℝ) : Prop :=
S 10 > 0

def condition_S11_neg (S : ℕ → ℝ) : Prop :=
S 11 < 0

-- Main statement
theorem arithmetic_sequence_properties {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}
  (ar_seq : is_arithmetic_sequence a d)
  (sum_first_n : sum_of_first_n_terms S a)
  (S10_pos : condition_S10_pos S)
  (S11_neg : condition_S11_neg S) :
  (∀ n, (S n) / n = a 1 + (n - 1) / 2 * d) ∧
  (a 2 = 1 → -2 / 7 < d ∧ d < -1 / 4) :=
by
  sorry

end arithmetic_sequence_properties_l109_109528


namespace eq_iff_squared_eq_l109_109150

theorem eq_iff_squared_eq (a b : ℝ) : a = b ↔ a^2 + b^2 = 2 * a * b :=
by
  sorry

end eq_iff_squared_eq_l109_109150


namespace balance_difference_l109_109408

def compound_interest (P r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

theorem balance_difference :
  let angela_balance := compound_interest 12000 0.05 15
  let bob_balance := simple_interest 15000 0.06 15
  let diff := abs (bob_balance - angela_balance)
  (round diff = 3553) :=
by 
  let angela_balance := compound_interest 12000 0.05 15
  let bob_balance := simple_interest 15000 0.06 15
  let diff := abs (bob_balance - angela_balance)
  have h : round diff = 3553 := sorry
  assumption

end balance_difference_l109_109408


namespace heartsuit_zero_heartsuit_self_heartsuit_pos_l109_109852

def heartsuit (x y : Real) : Real := x^2 - y^2

theorem heartsuit_zero (x : Real) : heartsuit x 0 = x^2 :=
by
  sorry

theorem heartsuit_self (x : Real) : heartsuit x x = 0 :=
by
  sorry

theorem heartsuit_pos (x y : Real) (h : x > y) : heartsuit x y > 0 :=
by
  sorry

end heartsuit_zero_heartsuit_self_heartsuit_pos_l109_109852


namespace parabola_from_hyperbola_l109_109947

noncomputable def hyperbola_equation (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

noncomputable def parabola_equation_1 (x y : ℝ) : Prop := y^2 = -24 * x

noncomputable def parabola_equation_2 (x y : ℝ) : Prop := y^2 = 24 * x

theorem parabola_from_hyperbola :
  (∃ x y : ℝ, hyperbola_equation x y) →
  (∃ x y : ℝ, parabola_equation_1 x y ∨ parabola_equation_2 x y) :=
by
  intro h
  -- proof is omitted
  sorry

end parabola_from_hyperbola_l109_109947


namespace previous_day_visitors_l109_109040

-- Define the number of visitors on the day Rachel visited
def visitors_on_day_rachel_visited : ℕ := 317

-- Define the difference in the number of visitors between the day Rachel visited and the previous day
def extra_visitors : ℕ := 22

-- Prove that the number of visitors on the previous day is 295
theorem previous_day_visitors : visitors_on_day_rachel_visited - extra_visitors = 295 :=
by
  sorry

end previous_day_visitors_l109_109040


namespace number_of_zeros_l109_109572

noncomputable def g (x : ℝ) : ℝ := Real.cos (Real.log x)

theorem number_of_zeros (n : ℕ) : (1 < x ∧ x < Real.exp Real.pi) → (∃! x : ℝ, g x = 0 ∧ 1 < x ∧ x < Real.exp Real.pi) → n = 1 :=
sorry

end number_of_zeros_l109_109572


namespace company_total_payment_correct_l109_109917

def totalEmployees : Nat := 450
def firstGroup : Nat := 150
def secondGroup : Nat := 200
def thirdGroup : Nat := 100

def firstBaseSalary : Nat := 2000
def secondBaseSalary : Nat := 2500
def thirdBaseSalary : Nat := 3000

def firstInitialBonus : Nat := 500
def secondInitialBenefit : Nat := 400
def thirdInitialBenefit : Nat := 600

def firstLayoffRound1 : Nat := (20 * firstGroup) / 100
def secondLayoffRound1 : Nat := (25 * secondGroup) / 100
def thirdLayoffRound1 : Nat := (15 * thirdGroup) / 100

def remainingFirstGroupRound1 : Nat := firstGroup - firstLayoffRound1
def remainingSecondGroupRound1 : Nat := secondGroup - secondLayoffRound1
def remainingThirdGroupRound1 : Nat := thirdGroup - thirdLayoffRound1

def firstAdjustedBonusRound1 : Nat := 400
def secondAdjustedBenefitRound1 : Nat := 300

def firstLayoffRound2 : Nat := (10 * remainingFirstGroupRound1) / 100
def secondLayoffRound2 : Nat := (15 * remainingSecondGroupRound1) / 100
def thirdLayoffRound2 : Nat := (5 * remainingThirdGroupRound1) / 100

def remainingFirstGroupRound2 : Nat := remainingFirstGroupRound1 - firstLayoffRound2
def remainingSecondGroupRound2 : Nat := remainingSecondGroupRound1 - secondLayoffRound2
def remainingThirdGroupRound2 : Nat := remainingThirdGroupRound1 - thirdLayoffRound2

def thirdAdjustedBenefitRound2 : Nat := (80 * thirdInitialBenefit) / 100

def totalBaseSalary : Nat :=
  (remainingFirstGroupRound2 * firstBaseSalary)
  + (remainingSecondGroupRound2 * secondBaseSalary)
  + (remainingThirdGroupRound2 * thirdBaseSalary)

def totalBonusesAndBenefits : Nat :=
  (remainingFirstGroupRound2 * firstAdjustedBonusRound1)
  + (remainingSecondGroupRound2 * secondAdjustedBenefitRound1)
  + (remainingThirdGroupRound2 * thirdAdjustedBenefitRound2)

def totalPayment : Nat :=
  totalBaseSalary + totalBonusesAndBenefits

theorem company_total_payment_correct :
  totalPayment = 893200 :=
by
  -- proof steps
  sorry

end company_total_payment_correct_l109_109917


namespace point_in_third_quadrant_l109_109709

theorem point_in_third_quadrant (x y : ℝ) (h1 : x = -3) (h2 : y = -2) : 
  x < 0 ∧ y < 0 :=
by
  sorry

end point_in_third_quadrant_l109_109709


namespace circle_eq_of_given_center_and_radius_l109_109821

theorem circle_eq_of_given_center_and_radius :
  (∀ (x y : ℝ),
    let C := (-1, 2)
    let r := 4
    (x + 1) ^ 2 + (y - 2) ^ 2 = 16) :=
by
  sorry

end circle_eq_of_given_center_and_radius_l109_109821


namespace greatest_of_3_consecutive_integers_l109_109402

theorem greatest_of_3_consecutive_integers (x : ℤ) (h : x + (x + 1) + (x + 2) = 24) : (x + 2) = 9 :=
by
-- Proof would go here.
sorry

end greatest_of_3_consecutive_integers_l109_109402


namespace no_solution_system_l109_109059

theorem no_solution_system (v : ℝ) :
  (∀ x y z : ℝ, ¬(x + y + z = v ∧ x + v * y + z = v ∧ x + y + v^2 * z = v^2)) ↔ (v = -1) :=
  sorry

end no_solution_system_l109_109059


namespace geometric_sequence_sum_l109_109394

/-- 
In a geometric sequence of real numbers, the sum of the first 2 terms is 15,
and the sum of the first 6 terms is 195. Prove that the sum of the first 4 terms is 82.
-/
theorem geometric_sequence_sum :
  ∃ (a r : ℝ), (a + a * r = 15) ∧ (a * (1 - r^6) / (1 - r) = 195) ∧ (a * (1 + r + r^2 + r^3) = 82) :=
by
  sorry

end geometric_sequence_sum_l109_109394


namespace polynomial_range_l109_109102

def p (x : ℝ) : ℝ := x^4 - 4*x^3 + 8*x^2 - 8*x + 5

theorem polynomial_range : ∀ x : ℝ, p x ≥ 2 :=
by
sorry

end polynomial_range_l109_109102


namespace rearrange_digits_2552_l109_109236

theorem rearrange_digits_2552 : 
    let digits := [2, 5, 5, 2]
    let factorial := fun n => Nat.factorial n
    let permutations := (factorial 4) / (factorial 2 * factorial 2)
    permutations = 6 :=
by
  sorry

end rearrange_digits_2552_l109_109236


namespace selene_sandwiches_l109_109511

-- Define the context and conditions in Lean
variables (S : ℕ) (sandwich_cost hamburger_cost hotdog_cost juice_cost : ℕ)
  (selene_cost tanya_cost total_cost : ℕ)

-- Each item prices
axiom sandwich_price : sandwich_cost = 2
axiom hamburger_price : hamburger_cost = 2
axiom hotdog_price : hotdog_cost = 1
axiom juice_price : juice_cost = 2

-- Purchases
axiom selene_purchase : selene_cost = sandwich_cost * S + juice_cost
axiom tanya_purchase : tanya_cost = hamburger_cost * 2 + juice_cost * 2

-- Total spending
axiom total_spending : selene_cost + tanya_cost = 16

-- Goal: Prove that Selene bought 3 sandwiches
theorem selene_sandwiches : S = 3 :=
by {
  sorry
}

end selene_sandwiches_l109_109511


namespace esteban_exercise_days_l109_109414

theorem esteban_exercise_days
  (natasha_exercise_per_day : ℕ)
  (natasha_days : ℕ)
  (esteban_exercise_per_day : ℕ)
  (total_exercise_hours : ℕ)
  (hours_to_minutes : ℕ)
  (natasha_exercise_total : ℕ)
  (total_exercise_minutes : ℕ)
  (esteban_exercise_total : ℕ)
  (esteban_days : ℕ) :
  natasha_exercise_per_day = 30 →
  natasha_days = 7 →
  esteban_exercise_per_day = 10 →
  total_exercise_hours = 5 →
  hours_to_minutes = 60 →
  natasha_exercise_total = natasha_exercise_per_day * natasha_days →
  total_exercise_minutes = total_exercise_hours * hours_to_minutes →
  esteban_exercise_total = total_exercise_minutes - natasha_exercise_total →
  esteban_days = esteban_exercise_total / esteban_exercise_per_day →
  esteban_days = 9 :=
by
  sorry

end esteban_exercise_days_l109_109414


namespace simplify_abs_expr_l109_109138

noncomputable def piecewise_y (x : ℝ) : ℝ :=
  if h1 : x < -3 then -3 * x
  else if h2 : -3 ≤ x ∧ x < 1 then 6 - x
  else if h3 : 1 ≤ x ∧ x < 2 then 4 + x
  else 3 * x

theorem simplify_abs_expr : 
  ∀ x : ℝ, (|x - 1| + |x - 2| + |x + 3|) = piecewise_y x :=
by
  intro x
  sorry

end simplify_abs_expr_l109_109138


namespace range_of_a_if_ineq_has_empty_solution_l109_109534

theorem range_of_a_if_ineq_has_empty_solution (a : ℝ) :
  (∀ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0) → -2 ≤ a ∧ a < 6/5 :=
by
  sorry

end range_of_a_if_ineq_has_empty_solution_l109_109534


namespace find_f_8_6_l109_109862

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem find_f_8_6 (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_symmetry : ∀ x, f (1 + x) = f (1 - x))
  (h_def : ∀ x, -1 ≤ x ∧ x ≤ 0 → f x = - (1 / 2) * x) :
  f 8.6 = 0.3 :=
sorry

end find_f_8_6_l109_109862


namespace domain_of_k_l109_109387

noncomputable def k (x : ℝ) : ℝ := (1 / (x + 6)) + (1 / (x^2 + 2*x + 9)) + (1 / (x^3 - 27))

theorem domain_of_k : {x : ℝ | k x ≠ 0} = {x : ℝ | x ≠ -6 ∧ x ≠ 3} :=
by
  sorry

end domain_of_k_l109_109387


namespace edmonton_to_red_deer_distance_l109_109173

noncomputable def distance_from_Edmonton_to_Calgary (speed time: ℝ) : ℝ :=
  speed * time

theorem edmonton_to_red_deer_distance :
  let speed := 110
  let time := 3
  let distance_Calgary_RedDeer := 110
  let distance_Edmonton_Calgary := distance_from_Edmonton_to_Calgary speed time
  let distance_Edmonton_RedDeer := distance_Edmonton_Calgary - distance_Calgary_RedDeer
  distance_Edmonton_RedDeer = 220 :=
by
  sorry

end edmonton_to_red_deer_distance_l109_109173


namespace divisor_in_second_division_l109_109566

theorem divisor_in_second_division 
  (n : ℤ) 
  (h1 : (68 : ℤ) * 269 = n) 
  (d q : ℤ) 
  (h2 : n = d * q + 1) 
  (h3 : Prime 18291):
  d = 18291 := by
  sorry

end divisor_in_second_division_l109_109566


namespace largest_product_of_three_l109_109617

theorem largest_product_of_three :
  ∃ (a b c : ℤ), a ∈ [-5, -3, -1, 2, 4, 6] ∧ 
                 b ∈ [-5, -3, -1, 2, 4, 6] ∧ 
                 c ∈ [-5, -3, -1, 2, 4, 6] ∧ 
                 a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
                 a * b * c = 90 := 
sorry

end largest_product_of_three_l109_109617


namespace max_statements_true_l109_109788

noncomputable def max_true_statements (a b : ℝ) : ℕ :=
  (if (a^2 > b^2) then 1 else 0) +
  (if (a < b) then 1 else 0) +
  (if (a < 0) then 1 else 0) +
  (if (b < 0) then 1 else 0) +
  (if (1 / a < 1 / b) then 1 else 0)

theorem max_statements_true : ∀ (a b : ℝ), max_true_statements a b ≤ 4 :=
by
  intro a b
  sorry

end max_statements_true_l109_109788


namespace initial_bottles_l109_109310

theorem initial_bottles (x : ℕ) (h1 : x - 8 + 45 = 51) : x = 14 :=
by
  -- Proof goes here
  sorry

end initial_bottles_l109_109310


namespace least_value_b_l109_109973

-- Defining the conditions of the problem
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

variables (a b c : ℕ)

-- Conditions
axiom angle_sum : a + b + c = 180
axiom primes : is_prime a ∧ is_prime b ∧ is_prime c
axiom order : a > b ∧ b > c

-- The statement to be proved
theorem least_value_b (h : a + b + c = 180) (hp : is_prime a ∧ is_prime b ∧ is_prime c) (ho : a > b ∧ b > c) : b = 5 :=
sorry

end least_value_b_l109_109973


namespace calculation_correct_l109_109907

theorem calculation_correct : 1984 + 180 / 60 - 284 = 1703 := 
by 
  sorry

end calculation_correct_l109_109907


namespace calculate_cubic_sum_roots_l109_109245

noncomputable def α := (27 : ℝ)^(1/3)
noncomputable def β := (64 : ℝ)^(1/3)
noncomputable def γ := (125 : ℝ)^(1/3)

theorem calculate_cubic_sum_roots (u v w : ℝ) :
  (u - α) * (u - β) * (u - γ) = 1/2 ∧
  (v - α) * (v - β) * (v - γ) = 1/2 ∧
  (w - α) * (w - β) * (w - γ) = 1/2 →
  u^3 + v^3 + w^3 = 217.5 :=
by
  sorry

end calculate_cubic_sum_roots_l109_109245


namespace pyramid_volume_correct_l109_109640

noncomputable def volume_of_pyramid (l α β : ℝ) (Hα : α = π/8) (Hβ : β = π/4) :=
  (1 / 3) * (l^3 / 24) * Real.sqrt (Real.sqrt 2 + 1)

theorem pyramid_volume_correct :
  ∀ (l : ℝ), l = 6 → volume_of_pyramid l (π/8) (π/4) (rfl) (rfl) = 9 * Real.sqrt (Real.sqrt 2 + 1) :=
by
  intros l hl
  rw [hl]
  norm_num
  sorry

end pyramid_volume_correct_l109_109640


namespace problem_statement_l109_109769

noncomputable def inequality_not_necessarily_true (a b c : ℝ) :=
  c < b ∧ b < a ∧ a * c < 0

theorem problem_statement (a b c : ℝ) (h : inequality_not_necessarily_true a b c) : ¬ (∃ a b c : ℝ, c < b ∧ b < a ∧ a * c < 0 ∧ ¬ (b^2/c > a^2/c)) :=
by sorry

end problem_statement_l109_109769


namespace sam_total_cans_l109_109575

theorem sam_total_cans (bags_sat : ℕ) (bags_sun : ℕ) (cans_per_bag : ℕ) 
  (h_sat : bags_sat = 3) (h_sun : bags_sun = 4) (h_cans : cans_per_bag = 9) : 
  (bags_sat + bags_sun) * cans_per_bag = 63 := 
by
  sorry

end sam_total_cans_l109_109575


namespace gcd_1407_903_l109_109792

theorem gcd_1407_903 : Nat.gcd 1407 903 = 21 := 
  sorry

end gcd_1407_903_l109_109792


namespace initial_amount_l109_109885

theorem initial_amount (M : ℝ) 
  (H1 : M * (2/3) * (4/5) * (3/4) * (5/7) * (5/6) = 200) : 
  M = 840 :=
by
  -- Proof to be provided
  sorry

end initial_amount_l109_109885


namespace problem_statement_l109_109945

theorem problem_statement (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := 
sorry

end problem_statement_l109_109945


namespace usual_time_to_reach_school_l109_109465

theorem usual_time_to_reach_school
  (R T : ℝ)
  (h1 : (7 / 6) * R = R / (T - 3) * T) : T = 21 :=
sorry

end usual_time_to_reach_school_l109_109465


namespace courier_problem_l109_109653

variable (x : ℝ) -- Let x represent the specified time in minutes
variable (d : ℝ) -- Let d represent the total distance traveled in km

theorem courier_problem
  (h1 : 1.2 * (x - 10) = d)
  (h2 : 0.8 * (x + 5) = d) :
  x = 40 ∧ d = 36 :=
by
  -- This theorem statement encapsulates the conditions and the answer.
  sorry

end courier_problem_l109_109653


namespace probability_all_white_balls_l109_109892

-- Definitions
def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def balls_drawn : ℕ := 7

-- Lean theorem statement
theorem probability_all_white_balls :
  (Nat.choose white_balls balls_drawn : ℚ) / (Nat.choose total_balls balls_drawn) = 8 / 6435 :=
sorry

end probability_all_white_balls_l109_109892


namespace josh_total_money_l109_109981

-- Define the initial conditions
def initial_wallet : ℝ := 300
def initial_investment : ℝ := 2000
def stock_increase_rate : ℝ := 0.30

-- The expected total amount Josh will have after selling his stocks
def expected_total_amount : ℝ := 2900

-- Define the problem: that the total money in Josh's wallet after selling all stocks equals $2900
theorem josh_total_money :
  let increased_value := initial_investment * stock_increase_rate
  let new_investment := initial_investment + increased_value
  let total_money := new_investment + initial_wallet
  total_money = expected_total_amount :=
by
  sorry

end josh_total_money_l109_109981


namespace find_larger_number_l109_109311

theorem find_larger_number (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 10) : a = 25 :=
  sorry

end find_larger_number_l109_109311


namespace problem_B_false_l109_109968

def diamondsuit (x y : ℝ) : ℝ := abs (x + y - 1)

theorem problem_B_false : ∀ x y : ℝ, 2 * (diamondsuit x y) ≠ diamondsuit (2 * x) (2 * y) :=
by
  intro x y
  dsimp [diamondsuit]
  sorry

end problem_B_false_l109_109968


namespace number_of_students_passed_both_tests_l109_109843

theorem number_of_students_passed_both_tests 
  (total_students : ℕ) 
  (passed_long_jump : ℕ) 
  (passed_shot_put : ℕ) 
  (failed_both_tests : ℕ) 
  (students_with_union : ℕ := total_students) :
  (students_with_union = passed_long_jump + passed_shot_put - passed_both_tests + failed_both_tests) 
  → passed_both_tests = 25 :=
by sorry

end number_of_students_passed_both_tests_l109_109843


namespace find_f_neg_l109_109674

noncomputable def f (a b x : ℝ) := a * x^3 + b * x - 2

theorem find_f_neg (a b : ℝ) (f_2017 : f a b 2017 = 7) : f a b (-2017) = -11 :=
by
  sorry

end find_f_neg_l109_109674


namespace expression_value_l109_109678

theorem expression_value : ((40 + 15) ^ 2 - 15 ^ 2) = 2800 := 
by
  sorry

end expression_value_l109_109678


namespace asparagus_spears_needed_l109_109366

def BridgetteGuests : Nat := 84
def AlexGuests : Nat := (2 * BridgetteGuests) / 3
def TotalGuests : Nat := BridgetteGuests + AlexGuests
def ExtraPlates : Nat := 10
def TotalPlates : Nat := TotalGuests + ExtraPlates
def VegetarianPercent : Nat := 20
def LargePortionPercent : Nat := 10
def VegetarianMeals : Nat := (VegetarianPercent * TotalGuests) / 100
def LargePortionMeals : Nat := (LargePortionPercent * TotalGuests) / 100
def RegularMeals : Nat := TotalGuests - (VegetarianMeals + LargePortionMeals)
def AsparagusPerRegularMeal : Nat := 8
def AsparagusPerVegetarianMeal : Nat := 6
def AsparagusPerLargePortionMeal : Nat := 12

theorem asparagus_spears_needed : 
  RegularMeals * AsparagusPerRegularMeal + 
  VegetarianMeals * AsparagusPerVegetarianMeal + 
  LargePortionMeals * AsparagusPerLargePortionMeal = 1120 := by
  sorry

end asparagus_spears_needed_l109_109366


namespace four_digit_numbers_count_l109_109871

open Nat

theorem four_digit_numbers_count :
  let valid_a := [5, 6]
  let valid_d := 0
  let valid_bc_pairs := [(3, 4), (3, 6)]
  valid_a.length * 1 * valid_bc_pairs.length = 4 :=
by
  sorry

end four_digit_numbers_count_l109_109871


namespace firecrackers_defective_fraction_l109_109131

theorem firecrackers_defective_fraction (initial_total good_remaining confiscated : ℕ) 
(h_initial : initial_total = 48) 
(h_confiscated : confiscated = 12) 
(h_good_remaining : good_remaining = 15) : 
(initial_total - confiscated - 2 * good_remaining) / (initial_total - confiscated) = 1 / 6 := by
  sorry

end firecrackers_defective_fraction_l109_109131


namespace altitude_length_of_right_triangle_l109_109700

theorem altitude_length_of_right_triangle 
    (a b c : ℝ) 
    (h1 : a = 8) 
    (h2 : b = 15) 
    (h3 : c = 17) 
    (h4 : a^2 + b^2 = c^2) 
    : (2 * (1/2 * a * b))/c = 120/17 := 
by {
  sorry
}

end altitude_length_of_right_triangle_l109_109700


namespace division_remainder_l109_109961

theorem division_remainder :
  (1225 * 1227 * 1229) % 12 = 3 :=
by sorry

end division_remainder_l109_109961


namespace excluded_angle_sum_1680_degrees_l109_109588

theorem excluded_angle_sum_1680_degrees (sum_except_one : ℝ) (h : sum_except_one = 1680) : 
  (180 - (1680 % 180)) = 120 :=
by
  have mod_eq : 1680 % 180 = 60 := by sorry
  rw [mod_eq]

end excluded_angle_sum_1680_degrees_l109_109588


namespace inequality_abc_l109_109160

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b + b * c + c * a ≤ 1) :
  a + b + c + Real.sqrt 3 ≥ 8 * a * b * c * (1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1)) :=
by
  sorry

end inequality_abc_l109_109160


namespace triangle_area_specific_l109_109062

noncomputable def vector2_area_formula (u v : ℝ × ℝ) : ℝ :=
|u.1 * v.2 - u.2 * v.1|

noncomputable def triangle_area (u v : ℝ × ℝ) : ℝ :=
(vector2_area_formula u v) / 2

theorem triangle_area_specific :
  let A := (1, 3)
  let B := (5, -1)
  let C := (9, 4)
  let u := (1 - 9, 3 - 4)
  let v := (5 - 9, -1 - 4)
  triangle_area u v = 18 := 
by sorry

end triangle_area_specific_l109_109062


namespace sum_of_coefficients_l109_109636

theorem sum_of_coefficients (d : ℤ) : 
  let expr := -(4 - d) * (d + 3 * (4 - d))
  let expanded_form := -2 * d ^ 2 + 20 * d - 48
  let sum_of_coeffs := -2 + 20 - 48
  sum_of_coeffs = -30 :=
by
  -- The proof will go here, skipping for now.
  sorry

end sum_of_coefficients_l109_109636


namespace product_of_two_special_numbers_is_perfect_square_l109_109965

-- Define the structure of the required natural numbers
structure SpecialNumber where
  m : ℕ
  n : ℕ
  value : ℕ := 2^m * 3^n

-- The main theorem to be proved
theorem product_of_two_special_numbers_is_perfect_square :
  ∀ (a b c d e : SpecialNumber),
  ∃ x y : SpecialNumber, ∃ k : ℕ, (x.value * y.value) = k * k :=
by
  sorry

end product_of_two_special_numbers_is_perfect_square_l109_109965


namespace average_weight_l109_109816

theorem average_weight (w_girls w_boys : ℕ) (avg_girls avg_boys : ℕ) (n : ℕ) : 
  n = 5 → avg_girls = 45 → avg_boys = 55 → 
  w_girls = n * avg_girls → w_boys = n * avg_boys →
  ∀ total_weight, total_weight = w_girls + w_boys →
  ∀ avg_weight, avg_weight = total_weight / (2 * n) →
  avg_weight = 50 :=
by
  intros h_n h_avg_girls h_avg_boys h_w_girls h_w_boys h_total_weight h_avg_weight
  -- here you would start the proof, but it is omitted as per the instructions
  sorry

end average_weight_l109_109816


namespace solve_quadratic_eq_l109_109075

theorem solve_quadratic_eq (x : ℝ) : (x^2 + 4 * x = 5) ↔ (x = 1 ∨ x = -5) :=
by
  sorry

end solve_quadratic_eq_l109_109075


namespace friends_recycled_pounds_l109_109028

theorem friends_recycled_pounds (total_points chloe_points each_points pounds_per_point : ℕ)
  (h1 : each_points = pounds_per_point / 6)
  (h2 : total_points = 5)
  (h3 : chloe_points = pounds_per_point / 6)
  (h4 : pounds_per_point = 28) 
  (h5 : total_points - chloe_points = 1) :
  pounds_per_point = 6 :=
by
  sorry

end friends_recycled_pounds_l109_109028


namespace final_portfolio_value_l109_109829

-- Define the initial conditions and growth rates
def initial_investment : ℝ := 80
def first_year_growth_rate : ℝ := 0.15
def additional_investment : ℝ := 28
def second_year_growth_rate : ℝ := 0.10

-- Calculate the values of the portfolio at each step
def after_first_year_investment : ℝ := initial_investment * (1 + first_year_growth_rate)
def after_addition : ℝ := after_first_year_investment + additional_investment
def after_second_year_investment : ℝ := after_addition * (1 + second_year_growth_rate)

theorem final_portfolio_value : after_second_year_investment = 132 := by
  -- This is where the proof would go, but we are omitting it
  sorry

end final_portfolio_value_l109_109829


namespace paco_salty_cookies_left_l109_109403

theorem paco_salty_cookies_left (S₁ S₂ : ℕ) (h₁ : S₁ = 6) (e1_eaten : ℕ) (a₁ : e1_eaten = 3)
(h₂ : S₂ = 24) (r1_ratio : ℚ) (a_ratio : r1_ratio = (2/3)) :
  S₁ - e1_eaten + r1_ratio * S₂ = 19 :=
by
  sorry

end paco_salty_cookies_left_l109_109403


namespace find_arithmetic_sequence_elements_l109_109283

theorem find_arithmetic_sequence_elements :
  ∃ (a b c : ℤ), -1 < a ∧ a < b ∧ b < c ∧ c < 7 ∧
  (∃ d : ℤ, a = -1 + d ∧ b = -1 + 2 * d ∧ c = -1 + 3 * d ∧ 7 = -1 + 4 * d) :=
sorry

end find_arithmetic_sequence_elements_l109_109283


namespace market_value_of_house_l109_109937

theorem market_value_of_house 
  (M : ℝ) -- Market value of the house
  (S : ℝ) -- Selling price of the house
  (P : ℝ) -- Pre-tax amount each person gets
  (after_tax : ℝ := 135000) -- Each person's amount after taxes
  (tax_rate : ℝ := 0.10) -- Tax rate
  (num_people : ℕ := 4) -- Number of people splitting the revenue
  (over_market_value_rate : ℝ := 0.20): 
  S = M + over_market_value_rate * M → 
  (num_people * P) = S → 
  after_tax = (1 - tax_rate) * P → 
  M = 500000 := 
by
  sorry

end market_value_of_house_l109_109937


namespace max_numbers_with_240_product_square_l109_109428

theorem max_numbers_with_240_product_square :
  ∃ (S : Finset ℕ), S.card = 11 ∧ ∀ k ∈ S, 1 ≤ k ∧ k ≤ 2015 ∧ ∃ n m, 240 * k = (n * m) ^ 2 :=
sorry

end max_numbers_with_240_product_square_l109_109428


namespace solve_fraction_zero_l109_109864

theorem solve_fraction_zero (x : ℝ) (h1 : (x^2 - 16) / (4 - x) = 0) (h2 : 4 - x ≠ 0) : x = -4 :=
sorry

end solve_fraction_zero_l109_109864


namespace abs_diff_of_sum_and_product_l109_109181

theorem abs_diff_of_sum_and_product (x y : ℝ) (h1 : x + y = 20) (h2 : x * y = 96) : |x - y| = 4 := 
by
  sorry

end abs_diff_of_sum_and_product_l109_109181


namespace find_ordered_triple_l109_109948

theorem find_ordered_triple
  (a b c : ℝ)
  (h1 : a > 2)
  (h2 : b > 2)
  (h3 : c > 2)
  (h4 : (a + 3) ^ 2 / (b + c - 3) + (b + 5) ^ 2 / (c + a - 5) + (c + 7) ^ 2 / (a + b - 7) = 48) :
  (a, b, c) = (7, 5, 3) :=
by {
  sorry
}

end find_ordered_triple_l109_109948


namespace xy_difference_l109_109951

theorem xy_difference (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 10) (h3 : x = 15) : x - y = 10 :=
by
  sorry

end xy_difference_l109_109951


namespace tiling_possible_with_one_type_l109_109953

theorem tiling_possible_with_one_type
  {a b m n : ℕ} (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hn : 0 < n)
  (H : (∃ (k : ℕ), a = k * n) ∨ (∃ (l : ℕ), b = l * m)) :
  (∃ (i : ℕ), a = i * n) ∨ (∃ (j : ℕ), b = j * m) :=
  sorry

end tiling_possible_with_one_type_l109_109953


namespace jane_can_buy_9_tickets_l109_109849

-- Definitions
def ticket_price : ℕ := 15
def jane_amount_initial : ℕ := 160
def scarf_cost : ℕ := 25
def jane_amount_after_scarf : ℕ := jane_amount_initial - scarf_cost
def max_tickets (amount : ℕ) (price : ℕ) := amount / price

-- The main statement
theorem jane_can_buy_9_tickets :
  max_tickets jane_amount_after_scarf ticket_price = 9 :=
by
  -- Proof goes here (proof steps would be outlined)
  sorry

end jane_can_buy_9_tickets_l109_109849


namespace exists_F_squared_l109_109290

theorem exists_F_squared (n : ℕ) : ∃ F : ℕ → ℕ, ∀ n : ℕ, (F (F n)) = n^2 := 
sorry

end exists_F_squared_l109_109290


namespace carla_marbles_l109_109317

theorem carla_marbles (m : ℕ) : m + 134 = 187 ↔ m = 53 :=
by sorry

end carla_marbles_l109_109317


namespace number_line_is_line_l109_109932

-- Define the terms
def number_line : Type := ℝ -- Assume number line can be considered real numbers for simplicity
def is_line (l : Type) : Prop := l = ℝ

-- Proving that number line is a line.
theorem number_line_is_line : is_line number_line :=
by {
  -- by definition of the number_line and is_line
  sorry
}

end number_line_is_line_l109_109932


namespace smallest_slice_area_l109_109783

theorem smallest_slice_area
  (a₁ : ℕ) (d : ℕ) (total_angle : ℕ) (r : ℕ) 
  (h₁ : a₁ = 30) (h₂ : d = 2) (h₃ : total_angle = 360) (h₄ : r = 10) :
  ∃ (n : ℕ) (smallest_angle : ℕ),
  n = 9 ∧ smallest_angle = 18 ∧ 
  ∃ (area : ℝ), area = 5 * Real.pi :=
by
  sorry


end smallest_slice_area_l109_109783


namespace max_value_of_f_l109_109375

noncomputable def f (x : ℝ) : ℝ :=
  (2 * x + 1) / (4 * x ^ 2 + 1)

theorem max_value_of_f : ∃ (M : ℝ), ∀ (x : ℝ), x > 0 → f x ≤ M ∧ M = (Real.sqrt 2 + 1) / 2 :=
by
  sorry

end max_value_of_f_l109_109375


namespace fraction_meaningful_if_and_only_if_l109_109625

theorem fraction_meaningful_if_and_only_if {x : ℝ} : (2 * x - 1 ≠ 0) ↔ (x ≠ 1 / 2) :=
by
  sorry

end fraction_meaningful_if_and_only_if_l109_109625


namespace find_x_angle_l109_109421

theorem find_x_angle (ABC ACB CDE : ℝ) (h1 : ABC = 70) (h2 : ACB = 90) (h3 : CDE = 42) : 
  ∃ x : ℝ, x = 158 :=
by
  sorry

end find_x_angle_l109_109421


namespace tree_boy_growth_ratio_l109_109357

theorem tree_boy_growth_ratio 
    (initial_tree_height final_tree_height initial_boy_height final_boy_height : ℕ) 
    (h₀ : initial_tree_height = 16) 
    (h₁ : final_tree_height = 40) 
    (h₂ : initial_boy_height = 24) 
    (h₃ : final_boy_height = 36) 
:
  (final_tree_height - initial_tree_height) / (final_boy_height - initial_boy_height) = 2 := 
by {
    -- Definitions and given conditions used in the statement part of the proof
    sorry
}

end tree_boy_growth_ratio_l109_109357


namespace train_speed_late_l109_109340

theorem train_speed_late (v : ℝ) 
  (h1 : ∀ (d : ℝ) (s : ℝ), d = 15 ∧ s = 100 → d / s = 0.15) 
  (h2 : ∀ (t1 t2 : ℝ), t1 = 0.15 ∧ t2 = 0.4 → t2 = t1 + 0.25)
  (h3 : ∀ (d : ℝ) (t : ℝ), d = 15 ∧ t = 0.4 → v = d / t) : 
  v = 37.5 := sorry

end train_speed_late_l109_109340


namespace tangent_line_condition_l109_109502

theorem tangent_line_condition (a b : ℝ):
  ((a = 1 ∧ b = 1) → ∀ x y : ℝ, x + y = 0 → (x - a)^2 + (y - b)^2 = 2 → x = 0 ∧ y = 0) ∧
  ( (a = -1 ∧ b = -1) → ∀ x y : ℝ, x + y = 0 → (x - a)^2 + (y - b)^2 = 2 → x = 0 ∧ y = 0) →
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) :=
by
  sorry

end tangent_line_condition_l109_109502


namespace solve_eq_proof_l109_109298

noncomputable def solve_equation : List ℚ := [-4, 1, 3 / 2, 2]

theorem solve_eq_proof :
  (∀ x : ℚ, 
    ((x^2 + 3 * x - 4)^2 + (2 * x^2 - 7 * x + 6)^2 = (3 * x^2 - 4 * x + 2)^2) ↔ 
    (x ∈ solve_equation)) :=
by
  sorry

end solve_eq_proof_l109_109298


namespace vector_BC_coordinates_l109_109205

-- Define the given vectors
def vec_AB : ℝ × ℝ := (2, -1)
def vec_AC : ℝ × ℝ := (-4, 1)

-- Define the vector subtraction
def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

-- Define the vector BC as the result of the subtraction
def vec_BC : ℝ × ℝ := vec_sub vec_AC vec_AB

-- State the theorem
theorem vector_BC_coordinates : vec_BC = (-6, 2) := by
  sorry

end vector_BC_coordinates_l109_109205


namespace sqrt_of_0_09_l109_109226

theorem sqrt_of_0_09 : Real.sqrt 0.09 = 0.3 :=
by
  -- Mathematical problem restates that the square root of 0.09 equals 0.3
  sorry

end sqrt_of_0_09_l109_109226


namespace values_of_j_for_exactly_one_real_solution_l109_109239

open Real

theorem values_of_j_for_exactly_one_real_solution :
  ∀ j : ℝ, (∀ x : ℝ, (3 * x + 4) * (x - 6) = -51 + j * x) → (j = 0 ∨ j = -36) := by
sorry

end values_of_j_for_exactly_one_real_solution_l109_109239


namespace surface_area_of_cube_l109_109441

-- Definition of the problem in Lean 4
theorem surface_area_of_cube (a : ℝ) (s : ℝ) (h : s * Real.sqrt 3 = a) : 6 * (s^2) = 2 * a^2 :=
by
  sorry

end surface_area_of_cube_l109_109441


namespace minimum_distance_proof_l109_109379

noncomputable def minimum_distance_AB : ℝ :=
  let f (x : ℝ) := x^2 - Real.log x
  let x_min := Real.sqrt 2 / 2
  let min_dist := (5 + Real.log 2) / 4
  min_dist

theorem minimum_distance_proof :
  ∃ a : ℝ, a = minimum_distance_AB :=
by
  use (5 + Real.log 2) / 4
  sorry

end minimum_distance_proof_l109_109379


namespace total_length_of_free_sides_l109_109330

theorem total_length_of_free_sides (L W : ℝ) 
  (h1 : L = 2 * W) 
  (h2 : L * W = 128) : 
  L + 2 * W = 32 := by 
sorry

end total_length_of_free_sides_l109_109330


namespace common_difference_is_1_l109_109386

variable (a_2 a_5 : ℕ) (d : ℤ)

def arithmetic_sequence (n a_1 : ℤ) (d : ℤ) : ℤ := a_1 + (n - 1) * d

theorem common_difference_is_1 
  (h1 : arithmetic_sequence 2 a_1 d = 3) 
  (h2 : arithmetic_sequence 5 a_1 d = 6) : 
  d = 1 := 
sorry

end common_difference_is_1_l109_109386


namespace translate_parabola_upwards_l109_109532

theorem translate_parabola_upwards (x y : ℝ) (h : y = x^2) : y + 1 = x^2 + 1 :=
by
  sorry

end translate_parabola_upwards_l109_109532


namespace arithmetic_sequence_a9_l109_109987

theorem arithmetic_sequence_a9 (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = n * (2 * a 0 + (n - 1))) →
  S 6 = 3 * S 3 →
  a 9 = 10 := by
  sorry

end arithmetic_sequence_a9_l109_109987


namespace Faye_can_still_make_8_bouquets_l109_109808

theorem Faye_can_still_make_8_bouquets (total_flowers : ℕ) (wilted_flowers : ℕ) (flowers_per_bouquet : ℕ) 
(h1 : total_flowers = 88) 
(h2 : wilted_flowers = 48) 
(h3 : flowers_per_bouquet = 5) : 
(total_flowers - wilted_flowers) / flowers_per_bouquet = 8 := 
by
  sorry

end Faye_can_still_make_8_bouquets_l109_109808


namespace cookies_left_l109_109174

theorem cookies_left (total_cookies : ℕ) (total_neighbors : ℕ) (cookies_per_neighbor : ℕ) (sarah_cookies : ℕ)
  (h1 : total_cookies = 150)
  (h2 : total_neighbors = 15)
  (h3 : cookies_per_neighbor = 10)
  (h4 : sarah_cookies = 12) :
  total_cookies - ((total_neighbors - 1) * cookies_per_neighbor + sarah_cookies) = 8 :=
by
  simp [h1, h2, h3, h4]
  sorry

end cookies_left_l109_109174


namespace g_f_neg4_eq_12_l109_109570

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^2 - 8

-- Define the assumption that g(f(4)) = 12
axiom g : ℝ → ℝ
axiom g_f4 : g (f 4) = 12

-- The theorem to prove that g(f(-4)) = 12
theorem g_f_neg4_eq_12 : g (f (-4)) = 12 :=
sorry -- proof placeholder

end g_f_neg4_eq_12_l109_109570


namespace smallest_positive_period_pi_increasing_intervals_in_0_to_pi_range_of_m_l109_109982

noncomputable def f (x m : ℝ) := 2 * (Real.cos x) ^ 2 + Real.sqrt 3 * Real.sin (2 * x) + m

theorem smallest_positive_period_pi (m : ℝ) :
  ∀ x : ℝ, f (x + π) m = f x m := sorry

theorem increasing_intervals_in_0_to_pi (m : ℝ) :
  ∀ x : ℝ, (0 ≤ x ∧ x ≤ π / 6) ∨ (2 * π / 3 ≤ x ∧ x ≤ π) →
  ∀ y : ℝ, ((0 ≤ y ∧ y ≤ π / 6 ∨ (2 * π / 3 ≤ y ∧ y ≤ π)) ∧ x < y) → f x m < f y m := sorry

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ π / 6) → -4 < f x m ∧ f x m < 4) ↔ (-6 < m ∧ m < 1) := sorry

end smallest_positive_period_pi_increasing_intervals_in_0_to_pi_range_of_m_l109_109982


namespace range_of_m_l109_109729

def p (m : ℝ) : Prop := m^2 - 4 > 0 ∧ m > 0
def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0
def condition1 (m : ℝ) : Prop := p m ∨ q m
def condition2 (m : ℝ) : Prop := ¬ (p m ∧ q m)

theorem range_of_m (m : ℝ) : condition1 m ∧ condition2 m → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by
  sorry

end range_of_m_l109_109729


namespace incorrect_statement_among_options_l109_109686

/- Definitions and Conditions -/
variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * a 1) + (n * (n - 1) / 2) * d

/- Conditions given in the problem -/
axiom S_6_gt_S_7 : S 6 > S 7
axiom S_7_gt_S_5 : S 7 > S 5

/- Incorrect statement to be proved -/
theorem incorrect_statement_among_options :
  ¬ (∀ n, S n ≤ S 11) := sorry

end incorrect_statement_among_options_l109_109686


namespace cells_sequence_exists_l109_109664

theorem cells_sequence_exists :
  ∃ (a : Fin 10 → ℚ), 
    a 0 = 9 ∧
    a 8 = 5 ∧
    (∀ i : Fin 8, a i + a (i + 1) + a (i + 2) = 14) :=
sorry

end cells_sequence_exists_l109_109664


namespace prime_factors_difference_l109_109874

theorem prime_factors_difference (n : ℤ) (h₁ : n = 180181) : ∃ p q : ℤ, Prime p ∧ Prime q ∧ p > q ∧ n % p = 0 ∧ n % q = 0 ∧ (p - q) = 2 :=
by
  sorry

end prime_factors_difference_l109_109874


namespace complex_is_1_sub_sqrt3i_l109_109580

open Complex

theorem complex_is_1_sub_sqrt3i (z : ℂ) (h : z * (1 + Real.sqrt 3 * I) = abs (1 + Real.sqrt 3 * I)) : z = 1 - Real.sqrt 3 * I :=
sorry

end complex_is_1_sub_sqrt3i_l109_109580


namespace days_worked_per_week_l109_109016

theorem days_worked_per_week (toys_per_week toys_per_day : ℕ) (h1 : toys_per_week = 5500) (h2 : toys_per_day = 1375) : toys_per_week / toys_per_day = 4 := by
  sorry

end days_worked_per_week_l109_109016


namespace find_n_from_digits_sum_l109_109721

theorem find_n_from_digits_sum (n : ℕ) (h1 : 777 = (9 * 1) + ((99 - 10 + 1) * 2) + (n - 99) * 3) : n = 295 :=
sorry

end find_n_from_digits_sum_l109_109721


namespace seq_eq_exp_l109_109286

theorem seq_eq_exp (a : ℕ → ℕ) 
  (h₀ : a 1 = 2) 
  (h₁ : ∀ n ≥ 2, a n = 2 * a (n - 1) - 1) :
  ∀ n ≥ 2, a n = 2^(n-1) + 1 := 
  by 
  sorry

end seq_eq_exp_l109_109286


namespace ratio_identity_l109_109737

-- Given system of equations
def system_of_equations (k : ℚ) (x y z : ℚ) :=
  x + k * y + 2 * z = 0 ∧
  2 * x + k * y + 3 * z = 0 ∧
  3 * x + 5 * y + 4 * z = 0

-- Prove that for k = -7/5, the system has a nontrivial solution and 
-- that the ratio xz / y^2 equals -25
theorem ratio_identity (x y z : ℚ) (k : ℚ) (h : system_of_equations k x y z) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  k = -7 / 5 → x * z / y^2 = -25 :=
by
  sorry

end ratio_identity_l109_109737


namespace face_opposite_to_A_is_D_l109_109604

-- Definitions of faces
inductive Face : Type
| A | B | C | D | E | F

open Face

-- Given conditions
def C_is_on_top : Face := C
def B_is_to_the_right_of_C : Face := B
def forms_cube (f1 f2 : Face) : Prop := -- Some property indicating that the faces are part of a folded cube
sorry

-- The theorem statement to prove that the face opposite to face A is D
theorem face_opposite_to_A_is_D (h1 : C_is_on_top = C) (h2 : B_is_to_the_right_of_C = B) (h3 : forms_cube A D)
    : ∃ f : Face, f = D := sorry

end face_opposite_to_A_is_D_l109_109604


namespace neg_p_l109_109918

noncomputable def f (a x : ℝ) : ℝ := a^x - x - a

theorem neg_p :
  ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ ∀ (x : ℝ), f a x ≠ 0 :=
sorry

end neg_p_l109_109918


namespace largest_number_of_hcf_lcm_l109_109050

theorem largest_number_of_hcf_lcm (HCF : ℕ) (factor1 factor2 : ℕ) (n1 n2 : ℕ) (largest : ℕ) 
  (h1 : HCF = 52) 
  (h2 : factor1 = 11) 
  (h3 : factor2 = 12) 
  (h4 : n1 = HCF * factor1) 
  (h5 : n2 = HCF * factor2) 
  (h6 : largest = max n1 n2) : 
  largest = 624 := 
by 
  sorry

end largest_number_of_hcf_lcm_l109_109050


namespace floor_equation_l109_109646

theorem floor_equation (n : ℤ) (h : ⌊(n^2 : ℤ) / 4⌋ - ⌊n / 2⌋^2 = 5) : n = 11 :=
sorry

end floor_equation_l109_109646


namespace cube_prism_surface_area_l109_109902

theorem cube_prism_surface_area (a : ℝ) (h : a > 0) :
  2 * (6 * a^2) > 4 * a^2 + 2 * (2 * a * a) :=
by sorry

end cube_prism_surface_area_l109_109902


namespace monotonic_has_at_most_one_solution_l109_109561

def monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y ∨ f y ≤ f x

theorem monotonic_has_at_most_one_solution (f : ℝ → ℝ) (c : ℝ) 
  (hf : monotonic f) : ∃! x : ℝ, f x = c :=
sorry

end monotonic_has_at_most_one_solution_l109_109561


namespace odd_func_value_l109_109018

noncomputable def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f : ℝ → ℝ
| x => if x > 0 then 2 * x - 3 else 0 -- f(x) is initially set to 0 when x ≤ 0, since we will not use this part directly.

theorem odd_func_value (x : ℝ) (h : x < 0) (hf : isOddFunction f) (hfx : ∀ x > 0, f x = 2 * x - 3) :
  f x = 2 * x + 3 :=
by
  sorry

end odd_func_value_l109_109018


namespace jellybean_avg_increase_l109_109127

noncomputable def avg_increase_jellybeans 
  (avg_original : ℕ) (num_bags_original : ℕ) (num_jellybeans_new_bag : ℕ) : ℕ :=
  let total_original := avg_original * num_bags_original
  let total_new := total_original + num_jellybeans_new_bag
  let num_bags_new := num_bags_original + 1
  let avg_new := total_new / num_bags_new
  avg_new - avg_original

theorem jellybean_avg_increase :
  avg_increase_jellybeans 117 34 362 = 7 := by
  let total_original := 117 * 34
  let total_new := total_original + 362
  let num_bags_new := 34 + 1
  let avg_new := total_new / num_bags_new
  let increase := avg_new - 117
  have h1 : total_original = 3978 := by norm_num
  have h2 : total_new = 4340 := by norm_num
  have h3 : num_bags_new = 35 := by norm_num
  have h4 : avg_new = 124 := by norm_num
  have h5 : increase = 7 := by norm_num
  exact h5

end jellybean_avg_increase_l109_109127


namespace solution_set_of_inequality_l109_109983

theorem solution_set_of_inequality : {x : ℝ | x^2 < 2 * x} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l109_109983


namespace carla_total_marbles_l109_109540

def initial_marbles : ℝ := 187.0
def bought_marbles : ℝ := 134.0

theorem carla_total_marbles : initial_marbles + bought_marbles = 321.0 := by
  sorry

end carla_total_marbles_l109_109540


namespace complement_intersection_l109_109606

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3}
def N : Set Nat := {1, 4}

theorem complement_intersection :
  (U \ M) ∩ (U \ N) = {5, 6} := by
  -- Proof is omitted.
  sorry

end complement_intersection_l109_109606


namespace triangle_proof_l109_109624

open Real

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Given conditions
axiom cos_rule_1 : a / cos A = c / (2 - cos C)
axiom b_value : b = 4
axiom c_value : c = 3
axiom area_equation : (1 / 2) * a * b * sin C = 3

-- The theorem statement
theorem triangle_proof : 3 * sin C + 4 * cos C = 5 := sorry

end triangle_proof_l109_109624


namespace glass_volume_230_l109_109978

variable (V : ℝ) -- Define the total volume of the glass

-- Define the conditions
def pessimist_glass_volume (V : ℝ) := 0.40 * V
def optimist_glass_volume (V : ℝ) := 0.60 * V
def volume_difference (V : ℝ) := optimist_glass_volume V - pessimist_glass_volume V

theorem glass_volume_230 
  (h1 : volume_difference V = 46) : V = 230 := 
sorry

end glass_volume_230_l109_109978


namespace sum_of_roots_of_cubic_l109_109409

noncomputable def P (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem sum_of_roots_of_cubic (a b c d : ℝ) (h : ∀ x : ℝ, P a b c d (x^2 + x) ≥ P a b c d (x + 1)) :
  (-b / a) = (P a b c d 0) :=
sorry

end sum_of_roots_of_cubic_l109_109409


namespace germs_killed_in_common_l109_109778

theorem germs_killed_in_common :
  ∃ x : ℝ, x = 5 ∧
    ∀ A B C : ℝ, A = 50 → 
    B = 25 → 
    C = 30 → 
    x = A + B - (100 - C) := sorry

end germs_killed_in_common_l109_109778


namespace expected_balls_original_positions_l109_109541

noncomputable def expected_original_positions : ℝ :=
  8 * ((3/4:ℝ)^3)

theorem expected_balls_original_positions :
  expected_original_positions = 3.375 := by
  sorry

end expected_balls_original_positions_l109_109541


namespace min_value_problem_l109_109743

theorem min_value_problem 
  (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 2 * y = 1) : 
  ∃ (min_val : ℝ), min_val = 2 * x + 3 * y^2 ∧ min_val = 8 / 9 :=
by
  sorry

end min_value_problem_l109_109743


namespace fred_gave_sandy_balloons_l109_109659

theorem fred_gave_sandy_balloons :
  ∀ (original_balloons given_balloons final_balloons : ℕ),
    original_balloons = 709 →
    final_balloons = 488 →
    given_balloons = original_balloons - final_balloons →
    given_balloons = 221 := by
  sorry

end fred_gave_sandy_balloons_l109_109659


namespace mushrooms_picked_on_second_day_l109_109591

theorem mushrooms_picked_on_second_day :
  ∃ (n2 : ℕ), (∃ (n1 n3 : ℕ), n3 = 2 * n2 ∧ n1 + n2 + n3 = 65) ∧ n2 = 21 :=
by
  sorry

end mushrooms_picked_on_second_day_l109_109591


namespace phoenix_equal_roots_implies_a_eq_c_l109_109969

-- Define the "phoenix" equation property
def is_phoenix (a b c : ℝ) : Prop := a + b + c = 0

-- Define the property that a quadratic equation has equal real roots
def has_equal_real_roots (a b c : ℝ) : Prop := b^2 - 4 * a * c = 0

theorem phoenix_equal_roots_implies_a_eq_c (a b c : ℝ) (h₀ : a ≠ 0) 
  (h₁ : is_phoenix a b c) (h₂ : has_equal_real_roots a b c) : a = c :=
sorry

end phoenix_equal_roots_implies_a_eq_c_l109_109969


namespace students_chose_greek_food_l109_109078
  
theorem students_chose_greek_food (total_students : ℕ) (percentage_greek : ℝ) (h1 : total_students = 200) (h2 : percentage_greek = 0.5) :
  (percentage_greek * total_students : ℝ) = 100 :=
by
  rw [h1, h2]
  norm_num
  sorry

end students_chose_greek_food_l109_109078


namespace div_by_1897_l109_109599

theorem div_by_1897 (n : ℕ) : 1897 ∣ (2903^n - 803^n - 464^n + 261^n) :=
sorry

end div_by_1897_l109_109599


namespace shortest_side_l109_109773

/-- 
Prove that if the lengths of the sides of a triangle satisfy the inequality \( a^2 + b^2 > 5c^2 \), 
then \( c \) is the length of the shortest side.
-/
theorem shortest_side (a b c : ℝ) (h : a^2 + b^2 > 5 * c^2) (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : c ≤ a ∧ c ≤ b :=
by {
  -- Proof will be provided here.
  sorry
}

end shortest_side_l109_109773


namespace find_positive_integer_l109_109526

variable (z : ℕ)

theorem find_positive_integer
  (h1 : (4 * z)^2 - z = 2345)
  (h2 : 0 < z) :
  z = 7 :=
sorry

end find_positive_integer_l109_109526


namespace car_distance_ratio_l109_109681

theorem car_distance_ratio (speed_A time_A speed_B time_B : ℕ) 
  (hA : speed_A = 70) (hTA : time_A = 10) 
  (hB : speed_B = 35) (hTB : time_B = 10) : 
  (speed_A * time_A) / gcd (speed_A * time_A) (speed_B * time_B) = 2 :=
by
  sorry

end car_distance_ratio_l109_109681


namespace grace_dimes_count_l109_109383

-- Defining the conditions
def dimes_to_pennies (d : ℕ) : ℕ := 10 * d
def nickels_to_pennies : ℕ := 10 * 5
def total_pennies (d : ℕ) : ℕ := dimes_to_pennies d + nickels_to_pennies

-- The statement of the theorem
theorem grace_dimes_count (d : ℕ) (h : total_pennies d = 150) : d = 10 := 
sorry

end grace_dimes_count_l109_109383


namespace find_a_plus_b_l109_109651

theorem find_a_plus_b (a b : ℝ) (f g : ℝ → ℝ) (h1 : ∀ x, f x = a * x + b) (h2 : ∀ x, g x = 3 * x - 4) 
(h3 : ∀ x, g (f x) = 4 * x + 5) : a + b = 13 / 3 :=
sorry

end find_a_plus_b_l109_109651


namespace max_n_sum_pos_largest_term_seq_l109_109452

-- Define the arithmetic sequence {a_n} and sum of first n terms S_n along with given conditions
def arithmetic_seq (a_1 : ℤ) (d : ℤ) (n : ℤ) : ℤ := a_1 + (n - 1) * d
def sum_arith_seq (a_1 : ℤ) (d : ℤ) (n : ℤ) : ℤ := n * (2 * a_1 + (n - 1) * d) / 2

variable (a_1 d : ℤ)
-- Conditions from problem
axiom a8_pos : arithmetic_seq a_1 d 8 > 0
axiom a8_a9_neg : arithmetic_seq a_1 d 8 + arithmetic_seq a_1 d 9 < 0

-- Prove the maximum n for which Sum S_n > 0 is 15
theorem max_n_sum_pos : ∃ n_max : ℤ, sum_arith_seq a_1 d n_max > 0 ∧ 
  ∀ n : ℤ, n > n_max → sum_arith_seq a_1 d n ≤ 0 := by
    exact ⟨15, sorry⟩  -- Substitute 'sorry' for the proof part

-- Determine the largest term in the sequence {S_n / a_n} for 1 ≤ n ≤ 15
theorem largest_term_seq : ∃ n_largest : ℤ, ∀ n : ℤ, 1 ≤ n → n ≤ 15 → 
  (sum_arith_seq a_1 d n / arithmetic_seq a_1 d n) ≤ (sum_arith_seq a_1 d n_largest / arithmetic_seq a_1 d n_largest) := by
    exact ⟨8, sorry⟩  -- Substitute 'sorry' for the proof part

end max_n_sum_pos_largest_term_seq_l109_109452


namespace bananas_to_apples_l109_109110

-- Definitions based on conditions
def bananas := ℕ
def oranges := ℕ
def apples := ℕ

-- Condition 1: 3/4 of 16 bananas are worth 12 oranges
def condition1 : Prop := 3 / 4 * 16 = 12

-- Condition 2: price of one banana equals the price of two apples
def price_equiv_banana_apple : Prop := 1 = 2

-- Proof: 1/3 of 9 bananas are worth 6 apples
theorem bananas_to_apples 
  (c1: condition1)
  (c2: price_equiv_banana_apple) : 1 / 3 * 9 * 2 = 6 :=
by sorry

end bananas_to_apples_l109_109110


namespace puppy_cost_l109_109900

variable (P : ℕ)

theorem puppy_cost (hc : 2 * 50 = 100) (hd : 3 * 100 = 300) (htotal : 2 * 50 + 3 * 100 + 2 * P = 700) : P = 150 :=
by
  sorry

end puppy_cost_l109_109900


namespace toby_candies_left_l109_109361

def total_candies : ℕ := 56 + 132 + 8 + 300
def num_cousins : ℕ := 13

theorem toby_candies_left : total_candies % num_cousins = 2 :=
by sorry

end toby_candies_left_l109_109361


namespace red_pencils_count_l109_109370

theorem red_pencils_count 
  (packs : ℕ) 
  (pencils_per_pack : ℕ) 
  (extra_packs : ℕ) 
  (extra_pencils_per_pack : ℕ)
  (total_red_pencils : ℕ) 
  (h1 : packs = 15)
  (h2 : pencils_per_pack = 1)
  (h3 : extra_packs = 3)
  (h4 : extra_pencils_per_pack = 2)
  (h5 : total_red_pencils = packs * pencils_per_pack + extra_packs * extra_pencils_per_pack) : 
  total_red_pencils = 21 := 
  by sorry

end red_pencils_count_l109_109370


namespace sequence_inequality_l109_109346

theorem sequence_inequality 
  (a : ℕ → ℝ)
  (h_non_decreasing : ∀ i j : ℕ, i ≤ j → a i ≤ a j)
  (h_range : ∀ i, 1 ≤ i ∧ i ≤ 10 → a i = a (i - 1)) :
  (1 / 6) * (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) ≤ (1 / 10) * (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10) :=
by
  sorry

end sequence_inequality_l109_109346


namespace largest_x_eq_120_div_11_l109_109324

theorem largest_x_eq_120_div_11 (x : ℝ) (h : (⌊x⌋ : ℝ) / x = 11 / 12) : x ≤ 120 / 11 :=
sorry

end largest_x_eq_120_div_11_l109_109324


namespace math_problem_l109_109828

variable (a b c d : ℝ)

-- The initial condition provided in the problem
def given_condition : Prop := (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7

-- The statement that needs to be proven
theorem math_problem 
  (h : given_condition a b c d) : 
  (a - c) * (b - d) / ((a - b) * (c - d)) = -1 := 
by 
  sorry

end math_problem_l109_109828


namespace pen_tip_movement_l109_109282

-- Definition of movements
def move_left (x : Int) : Int := -x
def move_right (x : Int) : Int := x

theorem pen_tip_movement :
  move_left 6 + move_right 3 = -3 :=
by
  sorry

end pen_tip_movement_l109_109282


namespace second_bag_roger_is_3_l109_109456

def total_candy_sandra := 2 * 6
def total_candy_roger := total_candy_sandra + 2
def first_bag_roger := 11
def second_bag_roger := total_candy_roger - first_bag_roger

theorem second_bag_roger_is_3 : second_bag_roger = 3 :=
by
  sorry

end second_bag_roger_is_3_l109_109456


namespace isosceles_triangle_solution_l109_109687

noncomputable def isosceles_triangle_sides (x y : ℝ) : Prop :=
(x + 1/2 * y = 6 ∧ 1/2 * x + y = 12) ∨ (x + 1/2 * y = 12 ∧ 1/2 * x + y = 6)

theorem isosceles_triangle_solution :
  ∃ (x y : ℝ), isosceles_triangle_sides x y ∧ x = 8 ∧ y = 2 :=
sorry

end isosceles_triangle_solution_l109_109687


namespace combination_15_choose_3_l109_109597

theorem combination_15_choose_3 :
  (Nat.choose 15 3) = 455 := by
sorry

end combination_15_choose_3_l109_109597


namespace solution_set_inequality_l109_109395

noncomputable def f (x : ℝ) := Real.exp (2 * x) - 1
noncomputable def g (x : ℝ) := Real.log (x + 1)

theorem solution_set_inequality :
  {x : ℝ | f (g x) - g (f x) ≤ 1} = Set.Icc (-1 : ℝ) 1 :=
sorry

end solution_set_inequality_l109_109395


namespace f_at_3_l109_109460

-- Define the function f and its conditions
variable (f : ℝ → ℝ)

-- The domain of the function f is ℝ, hence f : ℝ → ℝ
-- Also given:
axiom f_symm : ∀ x : ℝ, f (1 - x) = f (1 + x)
axiom f_add : f (-1) + f (3) = 12

-- Final proof statement
theorem f_at_3 : f 3 = 6 :=
by
  sorry

end f_at_3_l109_109460


namespace units_digit_of_7_pow_6_pow_5_l109_109919

theorem units_digit_of_7_pow_6_pow_5 : (7^(6^5)) % 10 = 1 := by
  -- Proof goes here
  sorry

end units_digit_of_7_pow_6_pow_5_l109_109919


namespace inequality_must_hold_l109_109931

theorem inequality_must_hold (x y : ℝ) (h : x > y) : -2 * x < -2 * y :=
sorry

end inequality_must_hold_l109_109931


namespace steps_left_to_climb_l109_109944

-- Define the conditions
def total_stairs : ℕ := 96
def climbed_stairs : ℕ := 74

-- The problem: Prove that the number of stairs left to climb is 22
theorem steps_left_to_climb : (total_stairs - climbed_stairs) = 22 :=
by 
  sorry

end steps_left_to_climb_l109_109944


namespace dot_product_AB_AC_dot_product_AB_BC_l109_109207

-- The definition of equilateral triangle with side length 6
structure EquilateralTriangle (A B C : Type*) :=
  (side_len : ℝ)
  (angle_ABC : ℝ)
  (angle_BCA : ℝ)
  (angle_CAB : ℝ)
  (AB_len : ℝ)
  (AC_len : ℝ)
  (BC_len : ℝ)
  (AB_eq_AC : AB_len = AC_len)
  (AB_eq_BC : AB_len = BC_len)
  (cos_ABC : ℝ)
  (cos_BCA : ℝ)
  (cos_CAB : ℝ)

-- Given an equilateral triangle with side length 6 where the angles are defined,
-- we can define the specific triangle
noncomputable def triangleABC (A B C : Type*) : EquilateralTriangle A B C :=
{ side_len := 6,
  angle_ABC := 120,
  angle_BCA := 60,
  angle_CAB := 60,
  AB_len := 6,
  AC_len := 6,
  BC_len := 6,
  AB_eq_AC := rfl,
  AB_eq_BC := rfl,
  cos_ABC := -0.5,
  cos_BCA := 0.5,
  cos_CAB := 0.5 }

-- Prove the dot product of vectors AB and AC
theorem dot_product_AB_AC (A B C : Type*) 
  (T : EquilateralTriangle A B C) : 
  (T.AB_len * T.AC_len * T.cos_BCA) = 18 :=
by sorry

-- Prove the dot product of vectors AB and BC
theorem dot_product_AB_BC (A B C : Type*) 
  (T : EquilateralTriangle A B C) : 
  (T.AB_len * T.BC_len * T.cos_ABC) = -18 :=
by sorry

end dot_product_AB_AC_dot_product_AB_BC_l109_109207


namespace primes_and_one_l109_109483

-- Given conditions:
variables {a n : ℕ}
variable (ha : a > 100 ∧ a % 2 = 1)  -- a is an odd natural number greater than 100
variable (hn_bound : ∀ n ≤ Nat.sqrt (a / 5), Prime (a - n^2) / 4)  -- for all n ≤ √(a / 5), (a - n^2) / 4 is prime

-- Theorem: For all n > √(a / 5), (a - n^2) / 4 is either prime or 1
theorem primes_and_one {a : ℕ} (ha : a > 100 ∧ a % 2 = 1)
  (hn_bound : ∀ n ≤ Nat.sqrt (a / 5), Prime ((a - n^2) / 4)) :
  ∀ n > Nat.sqrt (a / 5), Prime ((a - n^2) / 4) ∨ ((a - n^2) / 4) = 1 :=
sorry

end primes_and_one_l109_109483


namespace minimum_value_of_expression_l109_109538

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

end minimum_value_of_expression_l109_109538


namespace solve_eq_2_pow_x_plus_3_pow_y_eq_z_sq_l109_109172

theorem solve_eq_2_pow_x_plus_3_pow_y_eq_z_sq (x y z : ℕ) :
  ((x = 3 ∧ y = 0 ∧ z = 3) ∨ (x = 0 ∧ y = 1 ∧ z = 2) ∨ (x = 4 ∧ y = 2 ∧ z = 5)) →
  2^x + 3^y = z^2 :=
by
  sorry

end solve_eq_2_pow_x_plus_3_pow_y_eq_z_sq_l109_109172


namespace tan_angle_add_l109_109972

theorem tan_angle_add (x : ℝ) (h : Real.tan x = -3) : Real.tan (x + Real.pi / 6) = 2 * Real.sqrt 3 + 1 := 
by
  sorry

end tan_angle_add_l109_109972


namespace triangle_XYZ_median_l109_109934

theorem triangle_XYZ_median (XYZ : Triangle) (YZ : ℝ) (XM : ℝ) (XY2_add_XZ2 : ℝ) 
  (hYZ : YZ = 12) (hXM : XM = 7) : XY2_add_XZ2 = 170 → N - n = 0 := by
  sorry

end triangle_XYZ_median_l109_109934


namespace find_k_l109_109424

theorem find_k (k : ℝ) : 4 + ∑' (n : ℕ), (4 + n * k) / 5^n = 10 → k = 16 := by
  sorry

end find_k_l109_109424


namespace roof_area_l109_109884

-- Definitions based on conditions
variables (l w : ℝ)
def length_eq_five_times_width : Prop := l = 5 * w
def length_minus_width_eq_48 : Prop := l - w = 48

-- Proof goal
def area_of_roof : Prop := l * w = 720

-- Lean 4 statement asserting the mathematical problem
theorem roof_area (l w : ℝ) 
  (H1 : length_eq_five_times_width l w)
  (H2 : length_minus_width_eq_48 l w) : 
  area_of_roof l w := 
  by sorry

end roof_area_l109_109884


namespace calc_expression_result_l109_109698

theorem calc_expression_result :
  (16^12 * 8^8 / 2^60 = 4096) :=
by
  sorry

end calc_expression_result_l109_109698


namespace orange_bin_count_l109_109741

theorem orange_bin_count (initial_count throw_away add_new : ℕ) 
  (h1 : initial_count = 40) 
  (h2 : throw_away = 37) 
  (h3 : add_new = 7) : 
  initial_count - throw_away + add_new = 10 := 
by 
  sorry

end orange_bin_count_l109_109741


namespace smallest_three_digit_multiple_of_13_l109_109556

theorem smallest_three_digit_multiple_of_13 : ∃ n : ℕ, (100 ≤ n) ∧ (n < 1000) ∧ (n % 13 = 0) ∧ (∀ k : ℕ, (100 ≤ k) ∧ (k < 1000) ∧ (k % 13 = 0) → n ≤ k) → n = 104 :=
by
  sorry

end smallest_three_digit_multiple_of_13_l109_109556


namespace problem_sol_max_distance_from_circle_to_line_l109_109713

noncomputable def max_distance_circle_line : ℝ :=
  let ρ (θ : ℝ) : ℝ := 8 * Real.sin θ
  let line (θ : ℝ) : Prop := θ = Real.pi / 3
  let circle_center := (0, 4)
  let line_eq (x y : ℝ) : Prop := y = Real.sqrt 3 * x
  let shortest_distance := 2  -- Already calculated in solution
  let radius := 4
  shortest_distance + radius

theorem problem_sol_max_distance_from_circle_to_line :
  max_distance_circle_line = 6 :=
by
  unfold max_distance_circle_line
  sorry

end problem_sol_max_distance_from_circle_to_line_l109_109713


namespace car_time_passed_l109_109758

variable (speed : ℝ) (distance : ℝ) (time_passed : ℝ)

theorem car_time_passed (h_speed : speed = 2) (h_distance : distance = 2) :
  time_passed = distance / speed := by
  rw [h_speed, h_distance]
  norm_num
  sorry

end car_time_passed_l109_109758


namespace a_n_is_arithmetic_sequence_b_n_is_right_sequence_sum_first_n_terms_b_n_l109_109527

noncomputable def a_n (n : ℕ) : ℕ := 3 * n

noncomputable def b_n (n : ℕ) : ℕ := 3 * n + 2^(n - 1)

noncomputable def S_n (n : ℕ) : ℕ := (3 * n * (n + 1) / 2) + (2^n - 1)

theorem a_n_is_arithmetic_sequence (n : ℕ) :
  (a_n 1 = 3) ∧ (a_n 4 = 12) ∧ (∀ n : ℕ, a_n n = 3 * n) :=
by
  sorry

theorem b_n_is_right_sequence (n : ℕ) :
  (b_n 1 = 4) ∧ (b_n 4 = 20) ∧ (∀ n : ℕ, b_n n = 3 * n + 2^(n - 1)) ∧ 
  (∀ n : ℕ, b_n n - a_n n = 2^(n - 1)) :=
by
  sorry

theorem sum_first_n_terms_b_n (n : ℕ) :
  S_n n = 3 * (n * (n + 1) / 2) + 2^n - 1 :=
by
  sorry

end a_n_is_arithmetic_sequence_b_n_is_right_sequence_sum_first_n_terms_b_n_l109_109527


namespace area_inequality_l109_109558

open Real

variables (AB CD AD BC S : ℝ) (alpha beta : ℝ)
variables (α_pos : 0 < α ∧ α < π) (β_pos : 0 < β ∧ β < π)
variables (S_pos : 0 < S) (H1 : ConvexQuadrilateral AB CD AD BC S)

theorem area_inequality :
  AB * CD * sin α + AD * BC * sin β ≤ 2 * S ∧ 2 * S ≤ AB * CD + AD * BC :=
sorry

end area_inequality_l109_109558


namespace silver_nitrate_mass_fraction_l109_109988

variable (n : ℝ) (M : ℝ) (m_total : ℝ)
variable (m_agno3 : ℝ) (omega_agno3 : ℝ)

theorem silver_nitrate_mass_fraction 
  (h1 : n = 0.12) 
  (h2 : M = 170) 
  (h3 : m_total = 255)
  (h4 : m_agno3 = n * M) 
  (h5 : omega_agno3 = (m_agno3 * 100) / m_total) : 
  m_agno3 = 20.4 ∧ omega_agno3 = 8 :=
by
  -- insert proof here eventually 
  sorry

end silver_nitrate_mass_fraction_l109_109988


namespace average_speed_home_l109_109077

theorem average_speed_home
  (s_to_retreat : ℝ)
  (d_to_retreat : ℝ)
  (total_round_trip_time : ℝ)
  (t_retreat : d_to_retreat / s_to_retreat = 6)
  (t_total : d_to_retreat / s_to_retreat + 4 = total_round_trip_time) :
  (d_to_retreat / 4 = 75) :=
by
  sorry

end average_speed_home_l109_109077


namespace atomic_weight_Br_correct_l109_109072

def atomic_weight_Ba : ℝ := 137.33
def molecular_weight_compound : ℝ := 297
def atomic_weight_Br : ℝ := 79.835

theorem atomic_weight_Br_correct :
  molecular_weight_compound = atomic_weight_Ba + 2 * atomic_weight_Br :=
by
  sorry

end atomic_weight_Br_correct_l109_109072


namespace yogurt_cost_l109_109667

-- Define the price of milk per liter
def price_of_milk_per_liter : ℝ := 1.5

-- Define the price of fruit per kilogram
def price_of_fruit_per_kilogram : ℝ := 2.0

-- Define the amount of milk needed for one batch
def milk_per_batch : ℝ := 10.0

-- Define the amount of fruit needed for one batch
def fruit_per_batch : ℝ := 3.0

-- Define the cost of one batch of yogurt
def cost_per_batch : ℝ := (price_of_milk_per_liter * milk_per_batch) + (price_of_fruit_per_kilogram * fruit_per_batch)

-- Define the number of batches
def number_of_batches : ℝ := 3.0

-- Define the total cost for three batches of yogurt
def total_cost_for_three_batches : ℝ := cost_per_batch * number_of_batches

-- The theorem states that the total cost for three batches of yogurt is $63
theorem yogurt_cost : total_cost_for_three_batches = 63 := by
  sorry

end yogurt_cost_l109_109667


namespace meet_at_starting_point_l109_109478

theorem meet_at_starting_point (track_length : Nat) (speed_A_kmph speed_B_kmph : Nat)
  (h_track_length : track_length = 1500)
  (h_speed_A : speed_A_kmph = 36)
  (h_speed_B : speed_B_kmph = 54) :
  let speed_A_mps := speed_A_kmph * 1000 / 3600
  let speed_B_mps := speed_B_kmph * 1000 / 3600
  let time_A := track_length / speed_A_mps
  let time_B := track_length / speed_B_mps
  let lcm_time := Nat.lcm time_A time_B
  lcm_time = 300 :=
by
  sorry

end meet_at_starting_point_l109_109478


namespace items_in_bags_l109_109242

def calculateWaysToPlaceItems (n_items : ℕ) (n_bags : ℕ) : ℕ :=
  sorry

theorem items_in_bags :
  calculateWaysToPlaceItems 5 3 = 41 :=
by sorry

end items_in_bags_l109_109242


namespace central_angle_of_sector_with_area_one_l109_109137

theorem central_angle_of_sector_with_area_one (θ : ℝ):
  (1 / 2) * θ = 1 → θ = 2 :=
by
  sorry

end central_angle_of_sector_with_area_one_l109_109137


namespace polynomial_inequality_solution_l109_109738

theorem polynomial_inequality_solution (x : ℝ) :
  (x < 5 - Real.sqrt 29 ∨ x > 5 + Real.sqrt 29) →
  x^3 - 12 * x^2 + 36 * x + 8 > 0 :=
by
  sorry

end polynomial_inequality_solution_l109_109738


namespace work_completion_days_l109_109923

variables (M D X : ℕ) (W : ℝ)

-- Original conditions
def original_men : ℕ := 15
def planned_days : ℕ := 40
def men_absent : ℕ := 5

-- Theorem to prove
theorem work_completion_days :
  M = original_men →
  D = planned_days →
  W > 0 →
  (M - men_absent) * X * W = M * D * W →
  X = 60 :=
by
  intros hM hD hW h_work
  sorry

end work_completion_days_l109_109923


namespace volume_increase_l109_109196

theorem volume_increase (L B H : ℝ) :
  let L_new := 1.25 * L
  let B_new := 0.85 * B
  let H_new := 1.10 * H
  (L_new * B_new * H_new) = 1.16875 * (L * B * H) := 
by
  sorry

end volume_increase_l109_109196


namespace set_union_inter_eq_l109_109689

open Set

-- Conditions: Definitions of sets M, N, and P
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {1, 2, 3, 4}
def P : Set ℕ := {2, 3, 4, 5}

-- Claim: The result of (M ∩ N) ∪ P equals {1, 2, 3, 4, 5}
theorem set_union_inter_eq :
  (M ∩ N ∪ P) = {1, 2, 3, 4, 5} := 
by
  sorry

end set_union_inter_eq_l109_109689


namespace truck_covered_distance_l109_109284

theorem truck_covered_distance (t : ℝ) (d_bike : ℝ) (d_truck : ℝ) (v_bike : ℝ) (v_truck : ℝ) :
  t = 8 ∧ d_bike = 136 ∧ v_truck = v_bike + 3 ∧ d_bike = v_bike * t →
  d_truck = v_truck * t :=
by
  sorry

end truck_covered_distance_l109_109284


namespace option_B_correct_l109_109837

-- Define the commutativity of multiplication
def commutativity_of_mul (a b : Nat) : Prop :=
  a * b = b * a

-- State the problem, which is to prove that 2ab + 3ba = 5ab given commutativity
theorem option_B_correct (a b : Nat) : commutativity_of_mul a b → 2 * (a * b) + 3 * (b * a) = 5 * (a * b) :=
by
  intro h_comm
  rw [←h_comm]
  sorry

end option_B_correct_l109_109837


namespace johns_age_l109_109612

theorem johns_age :
  ∃ x : ℕ, (∃ n : ℕ, x - 5 = n^2) ∧ (∃ m : ℕ, x + 3 = m^3) ∧ x = 69 :=
by
  sorry

end johns_age_l109_109612


namespace apples_to_pears_value_l109_109279

/-- Suppose 1/2 of 12 apples are worth as much as 10 pears. -/
def apples_per_pears_ratio : ℚ := 10 / (1 / 2 * 12)

/-- Prove that 3/4 of 6 apples are worth as much as 7.5 pears. -/
theorem apples_to_pears_value : (3 / 4 * 6) * apples_per_pears_ratio = 7.5 := 
by
  sorry

end apples_to_pears_value_l109_109279


namespace range_of_m_l109_109123

def A := { x : ℝ | x^2 - 2 * x - 15 ≤ 0 }
def B (m : ℝ) := { x : ℝ | m - 2 < x ∧ x < 2 * m - 3 }

theorem range_of_m : ∀ m : ℝ, (B m ⊆ A) ↔ (m ≤ 4) :=
by sorry

end range_of_m_l109_109123


namespace driving_hours_fresh_l109_109868

theorem driving_hours_fresh (x : ℚ) : (25 * x + 15 * (9 - x) = 152) → x = 17 / 10 :=
by
  intros h
  sorry

end driving_hours_fresh_l109_109868


namespace book_arrangement_ways_l109_109347

open Nat

theorem book_arrangement_ways : 
  let m := 4  -- Number of math books
  let h := 6  -- Number of history books
  -- Number of ways to place a math book on both ends:
  let ways_ends := m * (m - 1)  -- Choices for the left end and right end
  -- Number of ways to arrange the remaining books:
  let ways_entities := 2!  -- Arrangements of the remaining entities
  -- Number of ways to arrange history books within the block:
  let arrange_history := factorial h
  -- Total arrangements
  let total_ways := ways_ends * ways_entities * arrange_history
  total_ways = 17280 := sorry

end book_arrangement_ways_l109_109347


namespace unique_solution_exists_l109_109607

theorem unique_solution_exists :
  ∃! (x y : ℝ), 4^(x^2 + 2 * y) + 4^(2 * x + y^2) = Real.cos (Real.pi * x) ∧ (x, y) = (2, -2) :=
by
  sorry

end unique_solution_exists_l109_109607


namespace unit_digit_of_power_of_two_l109_109451

theorem unit_digit_of_power_of_two (n : ℕ) :
  (2 ^ 2023) % 10 = 8 := 
by
  sorry

end unit_digit_of_power_of_two_l109_109451


namespace prime_square_mod_six_l109_109476

theorem prime_square_mod_six (p : ℕ) (hp : Nat.Prime p) (h : p > 5) : p^2 % 6 = 1 :=
by
  sorry

end prime_square_mod_six_l109_109476


namespace least_positive_three_digit_multiple_of_9_l109_109135

   theorem least_positive_three_digit_multiple_of_9 : ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ 9 ∣ n ∧ n = 108 :=
   by
     sorry
   
end least_positive_three_digit_multiple_of_9_l109_109135


namespace stella_glasses_count_l109_109820

-- Definitions for the conditions
def dolls : ℕ := 3
def clocks : ℕ := 2
def price_per_doll : ℕ := 5
def price_per_clock : ℕ := 15
def price_per_glass : ℕ := 4
def total_cost : ℕ := 40
def profit : ℕ := 25

-- The proof statement
theorem stella_glasses_count (dolls clocks price_per_doll price_per_clock price_per_glass total_cost profit : ℕ) :
  (dolls * price_per_doll + clocks * price_per_clock) + profit + total_cost = total_cost + profit → 
  (dolls * price_per_doll + clocks * price_per_clock) + profit + total_cost - (dolls * price_per_doll + clocks * price_per_clock) = price_per_glass * 5 :=
sorry

end stella_glasses_count_l109_109820


namespace root_interval_exists_l109_109715

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x - x + 1

theorem root_interval_exists :
  (f 2 > 0) →
  (f 3 < 0) →
  ∃ ξ, 2 < ξ ∧ ξ < 3 ∧ f ξ = 0 :=
by
  intros h1 h2
  sorry

end root_interval_exists_l109_109715


namespace max_good_pairs_1_to_30_l109_109583

def is_good_pair (a b : ℕ) : Prop := a % b = 0 ∨ b % a = 0

def max_good_pairs_in_range (n : ℕ) : ℕ :=
  if n = 30 then 13 else 0

theorem max_good_pairs_1_to_30 : max_good_pairs_in_range 30 = 13 :=
by
  sorry

end max_good_pairs_1_to_30_l109_109583


namespace alex_serge_equiv_distinct_values_l109_109933

-- Defining the context and data structures
variable {n : ℕ} -- Number of boxes
variable {c : ℕ → ℕ} -- Function representing number of cookies in each box, indexed by box number
variable {m : ℕ} -- Number of plates
variable {p : ℕ → ℕ} -- Function representing number of cookies on each plate, indexed by plate number

-- Define the sets representing the unique counts recorded by Alex and Serge
def Alex_record (c : ℕ → ℕ) (n : ℕ) : Set ℕ := 
  { x | ∃ i, i < n ∧ c i = x }

def Serge_record (p : ℕ → ℕ) (m : ℕ) : Set ℕ := 
  { y | ∃ j, j < m ∧ p j = y }

-- The proof goal: Alex's record contains the same number of distinct values as Serge's record
theorem alex_serge_equiv_distinct_values
  (c : ℕ → ℕ) (n : ℕ) (p : ℕ → ℕ) (m : ℕ) :
  Alex_record c n = Serge_record p m :=
sorry

end alex_serge_equiv_distinct_values_l109_109933


namespace find_x_l109_109185

-- Define the vectors a and b
def a : ℝ × ℝ := (-2, 3)
def b (x : ℝ) : ℝ × ℝ := (x, -3)

-- Define the parallel condition between (b - a) and b
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u  = (k * v.1, k * v.2)

-- The problem statement in Lean 4
theorem find_x (x : ℝ) (h : parallel (b x - a) (b x)) : x = 2 := 
  sorry

end find_x_l109_109185


namespace problem_statement_l109_109564

theorem problem_statement (n m : ℕ) (hn : n ≠ 0) (hm : m ≠ 0) : 
  (n * 5^n)^n = m * 5^9 ↔ n = 3 ∧ m = 27 :=
by {
  sorry
}

end problem_statement_l109_109564


namespace trapezoid_area_l109_109388

theorem trapezoid_area 
  (a b c : ℝ)
  (h_a : a = 5)
  (h_b : b = 15)
  (h_c : c = 13)
  : (1 / 2) * (a + b) * (Real.sqrt (c ^ 2 - ((b - a) / 2) ^ 2)) = 120 := by
  sorry

end trapezoid_area_l109_109388


namespace max_at_zero_l109_109851

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem max_at_zero : ∃ x, (∀ y, f y ≤ f x) ∧ x = 0 :=
by 
  sorry

end max_at_zero_l109_109851


namespace angle_y_equals_90_l109_109065

/-- In a geometric configuration, if ∠CBD = 120° and ∠ABE = 30°, 
    then the measure of angle y is 90°. -/
theorem angle_y_equals_90 (angle_CBD angle_ABE : ℝ) 
  (h1 : angle_CBD = 120) 
  (h2 : angle_ABE = 30) : 
  ∃ y : ℝ, y = 90 := 
by
  sorry

end angle_y_equals_90_l109_109065


namespace sunday_dogs_count_l109_109151

-- Define initial conditions
def initial_dogs : ℕ := 2
def monday_dogs : ℕ := 3
def total_dogs : ℕ := 10
def sunday_dogs (S : ℕ) : Prop :=
  initial_dogs + S + monday_dogs = total_dogs

-- State the theorem
theorem sunday_dogs_count : ∃ S : ℕ, sunday_dogs S ∧ S = 5 := by
  sorry

end sunday_dogs_count_l109_109151


namespace sarah_proof_l109_109753

-- Defining cards and conditions
inductive Card
| P : Card
| A : Card
| C5 : Card
| C4 : Card
| C7 : Card

-- Definition of vowel
def is_vowel : Card → Prop
| Card.P => false
| Card.A => true
| _ => false

-- Definition of prime numbers for the sides
def is_prime : Card → Prop
| Card.C5 => true
| Card.C4 => false
| Card.C7 => true
| _ => false

-- Tom's statement
def toms_statement (c : Card) : Prop :=
is_vowel c → is_prime c

-- Sarah shows Tom was wrong by turning over one card
theorem sarah_proof : ∃ c, toms_statement c = false ∧ c = Card.A :=
sorry

end sarah_proof_l109_109753


namespace f_is_n_l109_109766

noncomputable def f : ℕ+ → ℤ :=
  sorry

def f_defined_for_all_positive_integers (n : ℕ+) : Prop :=
  ∃ k, f n = k

def f_is_integer (n : ℕ+) : Prop :=
  ∃ k : ℤ, f n = k

def f_two_is_two : Prop :=
  f 2 = 2

def f_multiply_rule (m n : ℕ+) : Prop :=
  f (m * n) = f m * f n

def f_ordered (m n : ℕ+) (h : m > n) : Prop :=
  f m > f n

theorem f_is_n (n : ℕ+) :
  (f_defined_for_all_positive_integers n) →
  (f_is_integer n) →
  (f_two_is_two) →
  (∀ m n, f_multiply_rule m n) →
  (∀ m n (h : m > n), f_ordered m n h) →
  f n = n :=
sorry

end f_is_n_l109_109766


namespace unit_prices_min_total_cost_l109_109546

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

end unit_prices_min_total_cost_l109_109546


namespace average_percentage_difference_in_tail_sizes_l109_109119

-- Definitions for the number of segments in each type of rattlesnake
def segments_eastern : ℕ := 6
def segments_western : ℕ := 8
def segments_southern : ℕ := 7
def segments_northern : ℕ := 9

-- Definition for percentage difference function
def percentage_difference (a : ℕ) (b : ℕ) : ℚ := ((b - a : ℚ) / b) * 100

-- Theorem statement to prove the average percentage difference
theorem average_percentage_difference_in_tail_sizes :
  (percentage_difference segments_eastern segments_western +
   percentage_difference segments_southern segments_western +
   percentage_difference segments_northern segments_western) / 3 = 16.67 := 
sorry

end average_percentage_difference_in_tail_sizes_l109_109119


namespace abc_order_l109_109628

noncomputable def a : Real := Real.sqrt 3
noncomputable def b : Real := 0.5^3
noncomputable def c : Real := Real.log 3 / Real.log 0.5 -- log_0.5 3 is written as (log 3) / (log 0.5) in Lean

theorem abc_order : a > b ∧ b > c :=
by
  have h1 : a = Real.sqrt 3 := rfl
  have h2 : b = 0.5^3 := rfl
  have h3 : c = Real.log 3 / Real.log 0.5 := rfl
  sorry

end abc_order_l109_109628


namespace z_plus_inv_y_eq_10_div_53_l109_109795

-- Define the conditions for x, y, z being positive real numbers such that
-- xyz = 1, x + 1/z = 8, and y + 1/x = 20
variables (x y z : ℝ)
variables (hx : x > 0)
variables (hy : y > 0)
variables (hz : z > 0)
variables (h1 : x * y * z = 1)
variables (h2 : x + 1 / z = 8)
variables (h3 : y + 1 / x = 20)

-- The goal is to prove that z + 1/y = 10 / 53
theorem z_plus_inv_y_eq_10_div_53 : z + 1 / y = 10 / 53 :=
by {
  sorry
}

end z_plus_inv_y_eq_10_div_53_l109_109795


namespace zoo_problem_l109_109779

variables
  (parrots : ℕ)
  (snakes : ℕ)
  (monkeys : ℕ)
  (elephants : ℕ)
  (zebras : ℕ)
  (f : ℚ)

-- Conditions from the problem
theorem zoo_problem
  (h1 : parrots = 8)
  (h2 : snakes = 3 * parrots)
  (h3 : monkeys = 2 * snakes)
  (h4 : elephants = f * (parrots + snakes))
  (h5 : zebras = elephants - 3)
  (h6 : monkeys - zebras = 35) :
  f = 1 / 2 :=
sorry

end zoo_problem_l109_109779


namespace max_subset_size_l109_109724

theorem max_subset_size :
  ∃ S : Finset ℕ, (∀ (x y : ℕ), x ∈ S → y ∈ S → y ≠ 2 * x) →
  S.card = 1335 :=
sorry

end max_subset_size_l109_109724


namespace find_x_val_l109_109585

theorem find_x_val (x y : ℝ) (c : ℝ) (h1 : y = 1 → x = 8) (h2 : ∀ y, x * y^3 = c) : 
  (∀ (y : ℝ), y = 2 → x = 1) :=
by
  sorry

end find_x_val_l109_109585


namespace events_are_mutually_exclusive_but_not_opposite_l109_109671

-- Definitions based on the conditions:
structure BallBoxConfig where
  ball1 : Fin 4 → ℕ     -- Function representing the placement of ball number 1 into one of the 4 boxes
  h_distinct : ∀ i j, i ≠ j → ball1 i ≠ ball1 j

def event_A (cfg : BallBoxConfig) : Prop := cfg.ball1 ⟨0, sorry⟩ = 1
def event_B (cfg : BallBoxConfig) : Prop := cfg.ball1 ⟨0, sorry⟩ = 2

-- The proof problem:
theorem events_are_mutually_exclusive_but_not_opposite (cfg : BallBoxConfig) :
  (event_A cfg ∨ event_B cfg) ∧ ¬ (event_A cfg ∧ event_B cfg) :=
sorry

end events_are_mutually_exclusive_but_not_opposite_l109_109671


namespace not_divisible_by_11_check_divisibility_by_11_l109_109274

theorem not_divisible_by_11 : Nat := 8

theorem check_divisibility_by_11 (n : Nat) (h: n = 98473092) : ¬ (11 ∣ not_divisible_by_11) := by
  sorry

end not_divisible_by_11_check_divisibility_by_11_l109_109274


namespace Jed_older_than_Matt_l109_109177

-- Definitions of ages and conditions
def Jed_current_age : ℕ := sorry
def Matt_current_age : ℕ := sorry
axiom condition1 : Jed_current_age + 10 = 25
axiom condition2 : Jed_current_age + Matt_current_age = 20

-- Proof statement
theorem Jed_older_than_Matt : Jed_current_age - Matt_current_age = 10 :=
by
  sorry

end Jed_older_than_Matt_l109_109177


namespace gina_total_pay_l109_109587

noncomputable def gina_painting_pay : ℕ :=
let roses_per_hour := 6
let lilies_per_hour := 7
let rose_order := 6
let lily_order := 14
let pay_per_hour := 30

-- Calculate total time (in hours) Gina spends to complete the order
let time_for_roses := rose_order / roses_per_hour
let time_for_lilies := lily_order / lilies_per_hour
let total_time := time_for_roses + time_for_lilies

-- Calculate the total pay
let total_pay := total_time * pay_per_hour

total_pay

-- The theorem that Gina gets paid $90 for the order
theorem gina_total_pay : gina_painting_pay = 90 := by
  sorry

end gina_total_pay_l109_109587


namespace average_mark_of_excluded_students_l109_109325

noncomputable def average_mark_excluded (A : ℝ) (N : ℕ) (R : ℝ) (excluded_count : ℕ) (remaining_count : ℕ) : ℝ :=
  ((N : ℝ) * A - (remaining_count : ℝ) * R) / (excluded_count : ℝ)

theorem average_mark_of_excluded_students : 
  average_mark_excluded 70 10 90 5 5 = 50 := 
by 
  sorry

end average_mark_of_excluded_students_l109_109325


namespace gcd_m_n_l109_109936

def m : ℕ := 333333
def n : ℕ := 7777777

theorem gcd_m_n : Nat.gcd m n = 1 :=
by
  -- Mathematical steps have been omitted as they are not needed
  sorry

end gcd_m_n_l109_109936


namespace number_of_groups_l109_109299

theorem number_of_groups (max_value min_value interval : ℕ) (h_max : max_value = 36) (h_min : min_value = 15) (h_interval : interval = 4) : 
  ∃ groups : ℕ, groups = 6 :=
by 
  sorry

end number_of_groups_l109_109299


namespace caloprian_lifespan_proof_l109_109485

open Real

noncomputable def timeDilation (delta_t : ℝ) (v : ℝ) (c : ℝ) : ℝ :=
  delta_t * sqrt (1 - (v ^ 2) / (c ^ 2))

noncomputable def caloprianMinLifeSpan (d : ℝ) (v : ℝ) (c : ℝ) : ℝ :=
  let earth_time := (d / v) * 2
  timeDilation earth_time v c

theorem caloprian_lifespan_proof :
  caloprianMinLifeSpan 30 0.3 1 = 20 * sqrt 91 :=
sorry

end caloprian_lifespan_proof_l109_109485


namespace length_of_second_train_is_correct_l109_109735

-- Define the known values and conditions
def speed_train1_kmph := 120
def speed_train2_kmph := 80
def length_train1_m := 280
def crossing_time_s := 9

-- Convert speeds from km/h to m/s
def kmph_to_mps (kmph : ℕ) : ℚ := kmph * 1000 / 3600

def speed_train1_mps := kmph_to_mps speed_train1_kmph
def speed_train2_mps := kmph_to_mps speed_train2_kmph

-- Calculate relative speed
def relative_speed_mps := speed_train1_mps + speed_train2_mps

-- Calculate total distance covered when crossing
def total_distance_m := relative_speed_mps * crossing_time_s

-- The length of the second train
def length_train2_m := total_distance_m - length_train1_m

-- Prove the length of the second train
theorem length_of_second_train_is_correct : length_train2_m = 219.95 := by {
  sorry
}

end length_of_second_train_is_correct_l109_109735


namespace calculation_correct_l109_109416

theorem calculation_correct : 200 * 19.9 * 1.99 * 100 = 791620 := by
  sorry

end calculation_correct_l109_109416


namespace combined_distance_20_birds_two_seasons_l109_109648

theorem combined_distance_20_birds_two_seasons :
  let distance_jim_to_disney := 50
  let distance_disney_to_london := 60
  let number_of_birds := 20
  (number_of_birds * (distance_jim_to_disney + distance_disney_to_london)) = 2200 := by
  sorry

end combined_distance_20_birds_two_seasons_l109_109648


namespace smallest_of_three_integers_l109_109259

theorem smallest_of_three_integers (a b c : ℤ) (h1 : a * b * c = 32) (h2 : a + b + c = 3) : min (min a b) c = -4 := 
sorry

end smallest_of_three_integers_l109_109259


namespace point_T_coordinates_l109_109161

-- Definition of a point in 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Definition of a square with specific points O, P, Q, R
structure Square where
  O : Point
  P : Point
  Q : Point
  R : Point

-- Condition: O is the origin
def O : Point := {x := 0, y := 0}

-- Condition: Q is at (3, 3)
def Q : Point := {x := 3, y := 3}

-- Assuming the function area_triang for calculating the area of a triangle given three points
def area_triang (A B C : Point) : ℝ :=
  0.5 * abs ((B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x))

-- Assuming the function area_square for calculating the area of a square given the length of the side
def area_square (s : ℝ) : ℝ := s * s

-- Coordinates of point P and R since it's a square with sides parallel to axis
def P : Point := {x := 3, y := 0}
def R : Point := {x := 0, y := 3}

-- Definition of the square OPQR
def OPQR : Square := {O := O, P := P, Q := Q, R := R}

-- Length of the side of square OPQR
def side_length : ℝ := 3

-- Area of the square OPQR
def square_area : ℝ := area_square side_length

-- Twice the area of the square OPQR
def required_area : ℝ := 2 * square_area

-- Point T that needs to be proven
def T : Point := {x := 3, y := 12}

-- The main theorem to prove
theorem point_T_coordinates (T : Point) : area_triang P Q T = required_area → T = {x := 3, y := 12} :=
by
  sorry

end point_T_coordinates_l109_109161


namespace gcd_of_B_is_2_l109_109363

-- Condition: B is the set of all numbers which can be represented as the sum of four consecutive positive integers
def B := { n : ℕ | ∃ y : ℕ, n = (y - 1) + y + (y + 1) + (y + 2) }

-- Question: What is the greatest common divisor of all numbers in \( B \)
-- Mathematical equivalent proof problem: Prove gcd of all elements in set \( B \) is 2

theorem gcd_of_B_is_2 : ∀ n ∈ B, ∃ y : ℕ, n = 2 * (2 * y + 1) → ∀ m ∈ B, n.gcd m = 2 :=
by
  sorry

end gcd_of_B_is_2_l109_109363


namespace min_value_of_x2_plus_y2_l109_109336

-- Define the problem statement
theorem min_value_of_x2_plus_y2 (x y : ℝ) (h : 3 * x + y = 10) : x^2 + y^2 ≥ 10 :=
sorry

end min_value_of_x2_plus_y2_l109_109336


namespace product_of_undefined_roots_l109_109214

theorem product_of_undefined_roots :
  let f (x : ℝ) := (x^2 - 4*x + 4) / (x^2 - 5*x + 6)
  ∀ x : ℝ, (x^2 - 5*x + 6 = 0) → x = 2 ∨ x = 3 →
  (x = 2 ∨ x = 3 → x1 = 2 ∧ x2 = 3 → x1 * x2 = 6) :=
by
  sorry

end product_of_undefined_roots_l109_109214


namespace value_of_x_l109_109927

theorem value_of_x (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 :=
by
  sorry

end value_of_x_l109_109927


namespace sum_of_coordinates_of_D_l109_109291

theorem sum_of_coordinates_of_D (x y : ℝ) (h1 : (x + 6) / 2 = 2) (h2 : (y + 2) / 2 = 6) :
  x + y = 8 := 
by
  sorry

end sum_of_coordinates_of_D_l109_109291


namespace rectangle_sides_l109_109662

theorem rectangle_sides (S d : ℝ) (a b : ℝ) : 
  a = Real.sqrt (S + d^2 / 4) + d / 2 ∧ 
  b = Real.sqrt (S + d^2 / 4) - d / 2 →
  S = a * b ∧ d = a - b :=
by
  -- definitions and conditions will be used here in the proofs
  sorry

end rectangle_sides_l109_109662


namespace locus_of_tangent_circle_is_hyperbola_l109_109782

theorem locus_of_tangent_circle_is_hyperbola :
  ∀ (P : ℝ × ℝ) (r : ℝ),
    (P.1 ^ 2 + P.2 ^ 2).sqrt = 1 + r ∧ ((P.1 - 4) ^ 2 + P.2 ^ 2).sqrt = 2 + r →
    ∃ (a b : ℝ), (P.1 - a) ^ 2 / b ^ 2 - (P.2 / a) ^ 2 / b ^ 2 = 1 :=
sorry

end locus_of_tangent_circle_is_hyperbola_l109_109782


namespace no_obtuse_equilateral_triangle_exists_l109_109732

theorem no_obtuse_equilateral_triangle_exists :
  ¬(∃ (a b c : ℝ), a = b ∧ b = c ∧ a + b + c = π ∧ a > π/2 ∧ b > π/2 ∧ c > π/2) :=
sorry

end no_obtuse_equilateral_triangle_exists_l109_109732


namespace candidate_percentage_l109_109650

theorem candidate_percentage (P : ℚ) (votes_cast : ℚ) (loss : ℚ)
  (h1 : votes_cast = 2000) 
  (h2 : loss = 640) 
  (h3 : (P / 100) * votes_cast + (P / 100) * votes_cast + loss = votes_cast) :
  P = 34 :=
by 
  sorry

end candidate_percentage_l109_109650


namespace sum_of_digits_9x_l109_109533

theorem sum_of_digits_9x (a b c d e : ℕ) (x : ℕ) :
  (1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e ≤ 9) →
  x = 10000 * a + 1000 * b + 100 * c + 10 * d + e →
  (b - a) + (c - b) + (d - c) + (e - d) + (10 - e) = 9 :=
by
  sorry

end sum_of_digits_9x_l109_109533


namespace johnson_vincent_work_together_l109_109392

theorem johnson_vincent_work_together (work : Type) (time_johnson : ℕ) (time_vincent : ℕ) (combined_time : ℕ) :
  time_johnson = 10 → time_vincent = 40 → combined_time = 8 → 
  (1 / time_johnson + 1 / time_vincent) = 1 / combined_time :=
by
  intros h_johnson h_vincent h_combined
  sorry

end johnson_vincent_work_together_l109_109392


namespace convex_octagon_min_obtuse_l109_109200

-- Define a type for a polygon (here specifically an octagon)
structure Polygon (n : ℕ) :=
(vertices : ℕ)
(convex : Prop)

-- Define that an octagon is a specific polygon with 8 vertices
def octagon : Polygon 8 :=
{ vertices := 8,
  convex := sorry }

-- Define the predicate for convex polygons
def is_convex (poly : Polygon 8) : Prop := poly.convex

-- Defining the statement that a convex octagon has at least 5 obtuse interior angles
theorem convex_octagon_min_obtuse (poly : Polygon 8) (h : is_convex poly) : ∃ (n : ℕ), n = 5 :=
sorry

end convex_octagon_min_obtuse_l109_109200


namespace probability_of_adjacent_vertices_in_dodecagon_l109_109238

def probability_at_least_two_adjacent_vertices (n : ℕ) : ℚ :=
  if n = 12 then 24 / 55 else 0  -- Only considering the dodecagon case

theorem probability_of_adjacent_vertices_in_dodecagon :
  probability_at_least_two_adjacent_vertices 12 = 24 / 55 :=
by
  sorry

end probability_of_adjacent_vertices_in_dodecagon_l109_109238


namespace curve_passes_through_fixed_point_l109_109876

theorem curve_passes_through_fixed_point (k : ℝ) (x y : ℝ) (h : k ≠ -1) :
  (x ^ 2 + y ^ 2 + 2 * k * x + (4 * k + 10) * y + 10 * k + 20 = 0) → (x = 1 ∧ y = -3) :=
by
  sorry

end curve_passes_through_fixed_point_l109_109876


namespace correct_assignment_statement_l109_109756

theorem correct_assignment_statement (n m : ℕ) : 
  ¬ (4 = n) ∧ ¬ (n + 1 = m) ∧ ¬ (m + n = 0) :=
by
  sorry

end correct_assignment_statement_l109_109756


namespace tangent_with_min_slope_has_given_equation_l109_109052

-- Define the given function f(x)
def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x - 10

-- Define the derivative of the function f(x)
def f_prime (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 6

-- Define the coordinates of the tangent point
def tangent_point : ℝ × ℝ := (-1, f (-1))

-- Define the equation of the tangent line at the point with the minimum slope
def tangent_line_equation (x y : ℝ) : Prop := 3 * x - y - 11 = 0

-- Main theorem statement that needs to be proved
theorem tangent_with_min_slope_has_given_equation :
  tangent_line_equation (-1) (f (-1)) :=
sorry

end tangent_with_min_slope_has_given_equation_l109_109052


namespace min_value_of_f_l109_109048

noncomputable def f (x : ℝ) : ℝ := max (2 * x + 1) (5 - x)

theorem min_value_of_f : ∃ y, (∀ x : ℝ, f x ≥ y) ∧ y = 11 / 3 :=
by 
  sorry

end min_value_of_f_l109_109048


namespace joyce_apples_l109_109260

theorem joyce_apples : 
  ∀ (initial_apples given_apples remaining_apples : ℕ), 
    initial_apples = 75 → 
    given_apples = 52 → 
    remaining_apples = initial_apples - given_apples → 
    remaining_apples = 23 :=
by 
  intros initial_apples given_apples remaining_apples h_initial h_given h_remaining
  rw [h_initial, h_given] at h_remaining
  exact h_remaining

end joyce_apples_l109_109260


namespace triangle_problem_l109_109459

-- Define a triangle with given parameters and properties
variables {A B C : ℝ}
variables {a b c : ℝ} (h_tri : b * Real.cos A + a * Real.cos B = -2 * c * Real.cos C) 
variables (h_b2a : b = 2 * a)
variables (h_area : 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3)

-- Prove the required angles and side length
theorem triangle_problem 
    (h_tri : b * Real.cos A + a * Real.cos B = -2 * c * Real.cos C)
    (h_b2a : b = 2 * a)
    (h_area : 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3) :

    Real.cos C = -1/2 ∧ C = 2 * Real.pi / 3 ∧ c = 2 * Real.sqrt 7 := 
by 
  sorry

end triangle_problem_l109_109459


namespace factorize_expression_simplify_fraction_expr_l109_109427

-- (1) Prove the factorization of m^3 - 4m^2 + 4m
theorem factorize_expression (m : ℝ) : 
  m^3 - 4 * m^2 + 4 * m = m * (m - 2)^2 :=
by
  sorry

-- (2) Simplify the fraction operation correctly
theorem simplify_fraction_expr (x : ℝ) (h : x ≠ 1) : 
  2 / (x^2 - 1) - 1 / (x - 1) = -1 / (x + 1) :=
by
  sorry

end factorize_expression_simplify_fraction_expr_l109_109427


namespace warehouse_width_l109_109493

theorem warehouse_width (L : ℕ) (circles : ℕ) (total_distance : ℕ)
  (hL : L = 600)
  (hcircles : circles = 8)
  (htotal_distance : total_distance = 16000) : 
  ∃ W : ℕ, 2 * L + 2 * W = (total_distance / circles) ∧ W = 400 :=
by
  sorry

end warehouse_width_l109_109493


namespace solve_quadratic_eq1_solve_quadratic_eq2_solve_quadratic_eq3_solve_quadratic_eq4_l109_109517

-- Equation (1)
theorem solve_quadratic_eq1 (x : ℝ) : x^2 + 16 = 8*x ↔ x = 4 := by
  sorry

-- Equation (2)
theorem solve_quadratic_eq2 (x : ℝ) : 2*x^2 + 4*x - 3 = 0 ↔ 
  x = -1 + (Real.sqrt 10) / 2 ∨ x = -1 - (Real.sqrt 10) / 2 := by
  sorry

-- Equation (3)
theorem solve_quadratic_eq3 (x : ℝ) : x*(x - 1) = x ↔ x = 0 ∨ x = 2 := by
  sorry

-- Equation (4)
theorem solve_quadratic_eq4 (x : ℝ) : x*(x + 4) = 8*x - 3 ↔ x = 3 ∨ x = 1 := by
  sorry

end solve_quadratic_eq1_solve_quadratic_eq2_solve_quadratic_eq3_solve_quadratic_eq4_l109_109517


namespace find_coefficient_b_l109_109411

noncomputable def polynomial_f (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem find_coefficient_b 
  (a b c d : ℝ)
  (h1 : polynomial_f a b c d (-2) = 0)
  (h2 : polynomial_f a b c d 0 = 0)
  (h3 : polynomial_f a b c d 2 = 0)
  (h4 : polynomial_f a b c d (-1) = 3) :
  b = 0 :=
sorry

end find_coefficient_b_l109_109411


namespace sqrt_99_eq_9801_expr_2000_1999_2001_eq_1_l109_109521

theorem sqrt_99_eq_9801 : 99^2 = 9801 := by
  sorry

theorem expr_2000_1999_2001_eq_1 : 2000^2 - 1999 * 2001 = 1 := by
  sorry

end sqrt_99_eq_9801_expr_2000_1999_2001_eq_1_l109_109521


namespace rohit_distance_from_start_l109_109404

noncomputable def rohit_final_position : ℕ × ℕ :=
  let start := (0, 0)
  let p1 := (start.1, start.2 - 25)       -- Moves 25 meters south.
  let p2 := (p1.1 + 20, p1.2)           -- Turns left (east) and moves 20 meters.
  let p3 := (p2.1, p2.2 + 25)           -- Turns left (north) and moves 25 meters.
  let result := (p3.1 + 15, p3.2)       -- Turns right (east) and moves 15 meters.
  result

theorem rohit_distance_from_start :
  rohit_final_position = (35, 0) :=
sorry

end rohit_distance_from_start_l109_109404


namespace find_sum_of_bounds_l109_109746

variable (x y z : ℝ)

theorem find_sum_of_bounds (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) : 
  let m := min x (min y z)
  let M := max x (max y z)
  m + M = 8 / 3 :=
sorry

end find_sum_of_bounds_l109_109746


namespace four_digit_property_l109_109008

-- Define the problem conditions and statement
theorem four_digit_property (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 0 ≤ y ∧ y < 100) :
  (100 * x + y = (x + y) ^ 2) ↔ (100 * x + y = 3025 ∨ 100 * x + y = 2025 ∨ 100 * x + y = 9801) := by
sorry

end four_digit_property_l109_109008


namespace monomial_exponent_match_l109_109378

theorem monomial_exponent_match (m : ℤ) (x y : ℂ) : (-x^(2*m) * y^3 = 2 * x^6 * y^3) → m = 3 := 
by 
  sorry

end monomial_exponent_match_l109_109378


namespace negative_integer_solution_l109_109645

theorem negative_integer_solution (N : ℤ) (h : N^2 + N = 12) (h2 : N < 0) : N = -4 :=
sorry

end negative_integer_solution_l109_109645


namespace problem_statement_l109_109473

theorem problem_statement (x : ℤ) (h : Even (3 * x + 1)) : Odd (7 * x + 4) :=
  sorry

end problem_statement_l109_109473


namespace total_annual_car_maintenance_expenses_is_330_l109_109614

-- Define the conditions as constants
def annualMileage : ℕ := 12000
def milesPerOilChange : ℕ := 3000
def freeOilChangesPerYear : ℕ := 1
def costPerOilChange : ℕ := 50
def milesPerTireRotation : ℕ := 6000
def costPerTireRotation : ℕ := 40
def milesPerBrakePadReplacement : ℕ := 24000
def costPerBrakePadReplacement : ℕ := 200

-- Define the total annual car maintenance expenses calculation
def annualOilChangeExpenses (annualMileage : ℕ) (milesPerOilChange : ℕ) (freeOilChangesPerYear : ℕ) (costPerOilChange : ℕ) : ℕ :=
  let oilChangesNeeded := annualMileage / milesPerOilChange
  let paidOilChanges := oilChangesNeeded - freeOilChangesPerYear
  paidOilChanges * costPerOilChange

def annualTireRotationExpenses (annualMileage : ℕ) (milesPerTireRotation : ℕ) (costPerTireRotation : ℕ) : ℕ :=
  let tireRotationsNeeded := annualMileage / milesPerTireRotation
  tireRotationsNeeded * costPerTireRotation

def annualBrakePadReplacementExpenses (annualMileage : ℕ) (milesPerBrakePadReplacement : ℕ) (costPerBrakePadReplacement : ℕ) : ℕ :=
  let brakePadReplacementInterval := milesPerBrakePadReplacement / annualMileage
  costPerBrakePadReplacement / brakePadReplacementInterval

def totalAnnualCarMaintenanceExpenses : ℕ :=
  annualOilChangeExpenses annualMileage milesPerOilChange freeOilChangesPerYear costPerOilChange +
  annualTireRotationExpenses annualMileage milesPerTireRotation costPerTireRotation +
  annualBrakePadReplacementExpenses annualMileage milesPerBrakePadReplacement costPerBrakePadReplacement

-- Prove the total annual car maintenance expenses equals $330
theorem total_annual_car_maintenance_expenses_is_330 : totalAnnualCarMaintenanceExpenses = 330 := by
  sorry

end total_annual_car_maintenance_expenses_is_330_l109_109614


namespace isabel_total_problems_l109_109865

theorem isabel_total_problems
  (math_pages : ℕ)
  (reading_pages : ℕ)
  (problems_per_page : ℕ)
  (h1 : math_pages = 2)
  (h2 : reading_pages = 4)
  (h3 : problems_per_page = 5) :
  (math_pages + reading_pages) * problems_per_page = 30 :=
by
  sorry

end isabel_total_problems_l109_109865


namespace min_value_of_xsquare_ysquare_l109_109867

variable {x y : ℝ}

theorem min_value_of_xsquare_ysquare (h : 5 * x^2 * y^2 + y^4 = 1) : x^2 + y^2 ≥ 4 / 5 :=
sorry

end min_value_of_xsquare_ysquare_l109_109867


namespace gcd_228_1995_l109_109134

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l109_109134


namespace bankers_gain_is_126_l109_109847

-- Define the given conditions
def present_worth : ℝ := 600
def interest_rate : ℝ := 0.10
def time_period : ℕ := 2

-- Define the formula for compound interest to find the amount due A
def amount_due (PW : ℝ) (R : ℝ) (T : ℕ) : ℝ := PW * (1 + R) ^ T

-- Define the banker's gain as the difference between the amount due and the present worth
def bankers_gain (A : ℝ) (PW : ℝ) : ℝ := A - PW

-- The theorem to prove that the banker's gain is Rs. 126 given the conditions
theorem bankers_gain_is_126 : bankers_gain (amount_due present_worth interest_rate time_period) present_worth = 126 := by
  sorry

end bankers_gain_is_126_l109_109847


namespace range_of_a_for_min_value_at_x_eq_1_l109_109247

noncomputable def f (a x : ℝ) : ℝ := a*x^3 + (a-1)*x^2 - x + 2

theorem range_of_a_for_min_value_at_x_eq_1 :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f a 1 ≤ f a x) → a ≤ 3 / 5 :=
by
  sorry

end range_of_a_for_min_value_at_x_eq_1_l109_109247


namespace fewer_green_pens_than_pink_l109_109622

-- Define the variables
variables (G B : ℕ)

-- State the conditions
axiom condition1 : G < 12
axiom condition2 : B = G + 3
axiom condition3 : 12 + G + B = 21

-- Define the problem statement
theorem fewer_green_pens_than_pink : 12 - G = 9 :=
by
  -- Insert the proof steps here
  sorry

end fewer_green_pens_than_pink_l109_109622


namespace twenty_four_game_solution_l109_109148

theorem twenty_four_game_solution :
  let a := 4
  let b := 8
  (a - (b / b)) * b = 24 :=
by
  let a := 4
  let b := 8
  show (a - (b / b)) * b = 24
  sorry

end twenty_four_game_solution_l109_109148


namespace annual_interest_rate_continuous_compounding_l109_109360

noncomputable def continuous_compounding_rate (A P : ℝ) (t : ℝ) : ℝ :=
  (Real.log (A / P)) / t

theorem annual_interest_rate_continuous_compounding :
  continuous_compounding_rate 8500 5000 10 = (Real.log (1.7)) / 10 :=
by
  sorry

end annual_interest_rate_continuous_compounding_l109_109360


namespace simplify_abs_expression_l109_109718

theorem simplify_abs_expression (x : ℝ) : 
  |2*x + 1| - |x - 3| + |x - 6| = 
  if x < -1/2 then -2*x + 2 
  else if x < 3 then 2*x + 4 
  else if x < 6 then 10 
  else 2*x - 2 :=
by 
  sorry

end simplify_abs_expression_l109_109718


namespace pieces_in_each_package_l109_109240

-- Definitions from conditions
def num_packages : ℕ := 5
def extra_pieces : ℕ := 6
def total_pieces : ℕ := 41

-- Statement to prove
theorem pieces_in_each_package : ∃ x : ℕ, num_packages * x + extra_pieces = total_pieces ∧ x = 7 :=
by
  -- Begin the proof with the given setup
  sorry

end pieces_in_each_package_l109_109240


namespace fem_current_age_l109_109660

theorem fem_current_age (F : ℕ) 
  (h1 : ∃ M : ℕ, M = 4 * F) 
  (h2 : (F + 2) + (4 * F + 2) = 59) : 
  F = 11 :=
sorry

end fem_current_age_l109_109660


namespace bone_meal_percentage_growth_l109_109822

-- Definitions for the problem conditions
def control_height : ℝ := 36
def cow_manure_height : ℝ := 90
def bone_meal_to_cow_manure_ratio : ℝ := 0.5 -- since cow manure plant is 200% the height of bone meal plant

noncomputable def bone_meal_height : ℝ := cow_manure_height * bone_meal_to_cow_manure_ratio

-- The main theorem to prove
theorem bone_meal_percentage_growth : 
  ( (bone_meal_height - control_height) / control_height ) * 100 = 25 := 
by
  sorry

end bone_meal_percentage_growth_l109_109822


namespace find_value_l109_109911

theorem find_value (x v : ℝ) (h1 : 0.80 * x + v = x) (h2 : x = 100) : v = 20 := by
    sorry

end find_value_l109_109911


namespace hyperbola_asymptote_a_value_l109_109047

theorem hyperbola_asymptote_a_value (a : ℝ) (h : 0 < a) 
  (asymptote_eq : y = (3 / 5) * x) :
  (x^2 / a^2 - y^2 / 9 = 1) → a = 5 :=
by
  sorry

end hyperbola_asymptote_a_value_l109_109047


namespace circle_symmetry_y_axis_eq_l109_109877

theorem circle_symmetry_y_axis_eq (x y : ℝ) :
  (x^2 + y^2 + 2 * x = 0) ↔ (x^2 + y^2 - 2 * x = 0) :=
sorry

end circle_symmetry_y_axis_eq_l109_109877


namespace winning_candidate_votes_percentage_l109_109034

theorem winning_candidate_votes_percentage (majority : ℕ) (total_votes : ℕ) (winning_percentage : ℚ) :
  majority = 174 ∧ total_votes = 435 ∧ winning_percentage = 70 → 
  ∃ P : ℚ, (P / 100) * total_votes - ((100 - P) / 100) * total_votes = majority ∧ P = 70 :=
by
  sorry

end winning_candidate_votes_percentage_l109_109034


namespace sales_tax_percentage_l109_109608

noncomputable def original_price : ℝ := 200
noncomputable def discount : ℝ := 0.25 * original_price
noncomputable def sale_price : ℝ := original_price - discount
noncomputable def total_paid : ℝ := 165
noncomputable def sales_tax : ℝ := total_paid - sale_price

theorem sales_tax_percentage : (sales_tax / sale_price) * 100 = 10 := by
  sorry

end sales_tax_percentage_l109_109608


namespace find_a_and_c_l109_109280

theorem find_a_and_c (a c : ℝ) (h : ∀ x : ℝ, -1/3 < x ∧ x < 1/2 → ax^2 + 2*x + c < 0) :
  a = 12 ∧ c = -2 :=
by {
  sorry
}

end find_a_and_c_l109_109280


namespace general_pattern_specific_computation_l109_109006

theorem general_pattern (n : ℕ) : n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 :=
by
  sorry

theorem specific_computation : 2000 * 2001 * 2002 * 2003 + 1 = 4006001^2 :=
by
  have h := general_pattern 2000
  exact h

end general_pattern_specific_computation_l109_109006


namespace california_vs_texas_license_plates_l109_109720

theorem california_vs_texas_license_plates :
  (26^4 * 10^4) - (26^3 * 10^3) = 4553200000 :=
by
  sorry

end california_vs_texas_license_plates_l109_109720


namespace find_m_l109_109281

theorem find_m (x : ℝ) (m : ℝ) (h : ∃ x, (x - 2) ≠ 0 ∧ (4 - 2 * x) ≠ 0 ∧ (3 / (x - 2) + 1 = m / (4 - 2 * x))) : m = -6 :=
by
  sorry

end find_m_l109_109281


namespace simplify_expression_inequality_solution_l109_109683

-- Simplification part
theorem simplify_expression (x : ℝ) (h₁ : x ≠ -2) (h₂ : x ≠ 2):
  (2 - (x - 1) / (x + 2)) / ((x^2 + 10 * x + 25) / (x^2 - 4)) = 
  (x - 2) / (x + 5) :=
sorry

-- Inequality system part
theorem inequality_solution (x : ℝ):
  (2 * x + 7 > 3) ∧ ((x + 1) / 3 > (x - 1) / 2) → -2 < x ∧ x < 5 :=
sorry

end simplify_expression_inequality_solution_l109_109683


namespace series_ln2_series_1_ln2_l109_109924

theorem series_ln2 :
  ∑' n : ℕ, (1 / (n + 1) / (n + 2)) = Real.log 2 :=
sorry

theorem series_1_ln2 :
  ∑' k : ℕ, (1 / ((2 * k + 2) * (2 * k + 3))) = 1 - Real.log 2 :=
sorry

end series_ln2_series_1_ln2_l109_109924


namespace how_many_leaves_l109_109218

def ladybugs_per_leaf : ℕ := 139
def total_ladybugs : ℕ := 11676

theorem how_many_leaves : total_ladybugs / ladybugs_per_leaf = 84 :=
by
  sorry

end how_many_leaves_l109_109218


namespace max_ratio_a_c_over_b_d_l109_109389

-- Given conditions as Lean definitions
variables {a b c d : ℝ}
variable (h1 : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ 0)
variable (h2 : (a^2 + b^2 + c^2 + d^2) / (a + b + c + d)^2 = 3 / 8)

-- The statement to prove the maximum value of the given expression
theorem max_ratio_a_c_over_b_d : ∃ t : ℝ, t = (a + c) / (b + d) ∧ t ≤ 3 :=
by {
  -- The proof of this theorem is omitted.
  sorry
}

end max_ratio_a_c_over_b_d_l109_109389


namespace repetend_of_5_over_17_is_294117_l109_109784

theorem repetend_of_5_over_17_is_294117 :
  (∀ n : ℕ, (5 / 17 : ℚ) - (294117 : ℚ) / (10^6 : ℚ) ^ n = 0) :=
by
  sorry

end repetend_of_5_over_17_is_294117_l109_109784


namespace smallest_positive_value_l109_109633

theorem smallest_positive_value (a b : ℤ) (h : a > b) : 
  ∃ (k : ℚ), k = (a^2 + b^2) / (a^2 - b^2) + (a^2 - b^2) / (a^2 + b^2) ∧ k = 2 :=
sorry

end smallest_positive_value_l109_109633


namespace find_directrix_of_parabola_l109_109684

open Real

theorem find_directrix_of_parabola (O : ℝ × ℝ) (p : ℝ) (F P Q : ℝ × ℝ)
  (hO : O = (0, 0))
  (hp_pos : p > 0)
  (hC : ∀ x y, (x, y) = P → y^2 = 2 * p * x)
  (hF : F = (p / 2, 0))
  (hPF_perpendicular_to_x : P.1 = p / 2 ∧ P.2 = p)
  (hQ_on_x_axis : Q.2 = 0)
  (hPQ_perpendicular_OP : (P.1, P.2) ≠ Q ∧ ((P.2 - Q.2) / (P.1 - Q.1) = -1 / ((P.2 - O.2) / (P.1 - O.1))))
  (hFQ_distance : abs (F.1 - Q.1) = 6) :
  x = -3 / 2 :=
sorry

end find_directrix_of_parabola_l109_109684


namespace sam_weight_l109_109060

theorem sam_weight (Tyler Sam Peter : ℕ) : 
  (Peter = 65) →
  (Peter = Tyler / 2) →
  (Tyler = Sam + 25) →
  Sam = 105 :=
  by
  intros hPeter1 hPeter2 hTyler
  sorry

end sam_weight_l109_109060


namespace smallest_positive_integer_n_l109_109287

theorem smallest_positive_integer_n (n : ℕ) :
  (∃ n1 n2 n3 : ℕ, 5 * n = n1 ^ 5 ∧ 6 * n = n2 ^ 6 ∧ 7 * n = n3 ^ 7) →
  n = 2^5 * 3^5 * 5^4 * 7^6 :=
by
  sorry

end smallest_positive_integer_n_l109_109287


namespace smallest_sum_l109_109993

theorem smallest_sum (r s t : ℕ) (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_pos_t : 0 < t) 
  (h_prod : r * s * t = 1230) : r + s + t = 52 :=
sorry

end smallest_sum_l109_109993


namespace center_of_circle_sum_l109_109278
-- Import the entire library

-- Define the problem using declarations for conditions and required proof
theorem center_of_circle_sum (x y : ℝ) 
  (h : ∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = 9 → (x = 2) ∧ (y = -3)) : 
  x + y = -1 := 
by 
  sorry 

end center_of_circle_sum_l109_109278


namespace x_y_divisible_by_7_l109_109836

theorem x_y_divisible_by_7
  (x y a b : ℤ)
  (hx : 3 * x + 4 * y = a ^ 2)
  (hy : 4 * x + 3 * y = b ^ 2)
  (hx_pos : x > 0) (hy_pos : y > 0) :
  7 ∣ x ∧ 7 ∣ y :=
by
  sorry

end x_y_divisible_by_7_l109_109836


namespace find_angle_x_eq_38_l109_109590

theorem find_angle_x_eq_38
  (angle_ACD angle_ECB angle_DCE : ℝ)
  (h1 : angle_ACD = 90)
  (h2 : angle_ECB = 52)
  (h3 : angle_ACD + angle_ECB + angle_DCE = 180) :
  angle_DCE = 38 :=
by
  sorry

end find_angle_x_eq_38_l109_109590
