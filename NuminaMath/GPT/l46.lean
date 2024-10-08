import Mathlib

namespace remainder_when_divided_by_7_l46_46082

theorem remainder_when_divided_by_7 
  {k : ℕ} 
  (h1 : k % 5 = 2) 
  (h2 : k % 6 = 5) 
  (h3 : k < 41) : 
  k % 7 = 3 := 
sorry

end remainder_when_divided_by_7_l46_46082


namespace nat_ineq_qr_ps_l46_46808

theorem nat_ineq_qr_ps (a b p q r s : ℕ) (h₀ : q * r - p * s = 1) 
  (h₁ : (p : ℚ) / q < a / b) (h₂ : (a : ℚ) / b < r / s) 
  : b ≥ q + s := sorry

end nat_ineq_qr_ps_l46_46808


namespace cheapest_pie_cost_is_18_l46_46797

noncomputable def crust_cost : ℝ := 2 + 1 + 1.5
noncomputable def blueberry_container_cost : ℝ := 2.25
noncomputable def blueberry_containers_needed : ℕ := 3 * (16 / 8)
noncomputable def blueberry_filling_cost : ℝ := blueberry_containers_needed * blueberry_container_cost
noncomputable def cherry_filling_cost : ℝ := 14
noncomputable def cheapest_filling_cost : ℝ := min blueberry_filling_cost cherry_filling_cost
noncomputable def total_cheapest_pie_cost : ℝ := crust_cost + cheapest_filling_cost

theorem cheapest_pie_cost_is_18 : total_cheapest_pie_cost = 18 := by
  sorry

end cheapest_pie_cost_is_18_l46_46797


namespace cos_5alpha_eq_sin_5alpha_eq_l46_46065

noncomputable def cos_five_alpha (α : ℝ) : ℝ := 16 * (Real.cos α) ^ 5 - 20 * (Real.cos α) ^ 3 + 5 * (Real.cos α)
noncomputable def sin_five_alpha (α : ℝ) : ℝ := 16 * (Real.sin α) ^ 5 - 20 * (Real.sin α) ^ 3 + 5 * (Real.sin α)

theorem cos_5alpha_eq (α : ℝ) : Real.cos (5 * α) = cos_five_alpha α :=
by sorry

theorem sin_5alpha_eq (α : ℝ) : Real.sin (5 * α) = sin_five_alpha α :=
by sorry

end cos_5alpha_eq_sin_5alpha_eq_l46_46065


namespace blueberry_jelly_amount_l46_46868

theorem blueberry_jelly_amount (total_jelly : ℕ) (strawberry_jelly : ℕ) 
  (h_total : total_jelly = 6310) 
  (h_strawberry : strawberry_jelly = 1792) 
  : total_jelly - strawberry_jelly = 4518 := 
by 
  sorry

end blueberry_jelly_amount_l46_46868


namespace find_x_if_delta_phi_x_eq_3_l46_46530

def delta (x : ℚ) : ℚ := 2 * x + 5
def phi (x : ℚ) : ℚ := 9 * x + 6

theorem find_x_if_delta_phi_x_eq_3 :
  ∃ (x : ℚ), delta (phi x) = 3 ∧ x = -7/9 := by
sorry

end find_x_if_delta_phi_x_eq_3_l46_46530


namespace extra_time_A_to_reach_destination_l46_46057

theorem extra_time_A_to_reach_destination (speed_ratio : ℕ -> ℕ -> Prop) (t_A t_B : ℝ)
  (h_ratio : speed_ratio 3 4)
  (time_A : t_A = 2)
  (distance_constant : ∀ a b : ℝ, a / b = (3 / 4)) :
  (t_A - t_B) * 60 = 30 :=
by
  sorry

end extra_time_A_to_reach_destination_l46_46057


namespace initial_value_amount_l46_46388

theorem initial_value_amount (P : ℝ) 
  (h1 : ∀ t, t ≥ 0 → t = P * (1 + (1/8)) ^ t) 
  (h2 : P * (1 + (1/8)) ^ 2 = 105300) : 
  P = 83200 := 
sorry

end initial_value_amount_l46_46388


namespace find_dividend_l46_46851

def quotient : ℝ := -427.86
def divisor : ℝ := 52.7
def remainder : ℝ := -14.5
def dividend : ℝ := (quotient * divisor) + remainder

theorem find_dividend : dividend = -22571.122 := by
  sorry

end find_dividend_l46_46851


namespace parallelogram_and_triangle_area_eq_l46_46580

noncomputable def parallelogram_area (AB AD : ℝ) : ℝ :=
  AB * AD

noncomputable def right_triangle_area (DG FG : ℝ) : ℝ :=
  (DG * FG) / 2

variables (AB AD DG FG : ℝ)
variables (angleDFG : ℝ)

def parallelogram_ABCD (AB : ℝ) (AD : ℝ) (angleDFG : ℝ) (DG : ℝ) : Prop :=
  parallelogram_area AB AD = 24 ∧ angleDFG = 90 ∧ DG = 6

theorem parallelogram_and_triangle_area_eq (h1 : parallelogram_ABCD AB AD angleDFG DG)
    (h2 : parallelogram_area AB AD = right_triangle_area DG FG) : FG = 8 :=
by
  sorry

end parallelogram_and_triangle_area_eq_l46_46580


namespace electricity_cost_per_kWh_is_14_cents_l46_46660

-- Define the conditions
def powerUsagePerHour : ℕ := 125 -- watts
def dailyUsageHours : ℕ := 4 -- hours
def weeklyCostInCents : ℕ := 49 -- cents
def daysInWeek : ℕ := 7 -- days
def wattsToKilowattsFactor : ℕ := 1000 -- conversion factor

-- Define a function to calculate the cost per kWh
def costPerKwh (powerUsagePerHour : ℕ) (dailyUsageHours : ℕ) (weeklyCostInCents : ℕ) (daysInWeek : ℕ) (wattsToKilowattsFactor : ℕ) : ℕ :=
  let dailyConsumption := powerUsagePerHour * dailyUsageHours
  let weeklyConsumption := dailyConsumption * daysInWeek
  let weeklyConsumptionInKwh := weeklyConsumption / wattsToKilowattsFactor
  weeklyCostInCents / weeklyConsumptionInKwh

-- State the theorem
theorem electricity_cost_per_kWh_is_14_cents :
  costPerKwh powerUsagePerHour dailyUsageHours weeklyCostInCents daysInWeek wattsToKilowattsFactor = 14 :=
by
  sorry

end electricity_cost_per_kWh_is_14_cents_l46_46660


namespace initial_books_l46_46047

-- Define the variables and conditions
def B : ℕ := 75
def loaned_books : ℕ := 60
def returned_books : ℕ := (70 * loaned_books) / 100
def not_returned_books : ℕ := loaned_books - returned_books
def end_of_month_books : ℕ := 57

-- State the theorem
theorem initial_books (h1 : returned_books = 42)
                      (h2 : end_of_month_books = 57)
                      (h3 : loaned_books = 60) :
  B = end_of_month_books + not_returned_books :=
by sorry

end initial_books_l46_46047


namespace total_employees_l46_46936

theorem total_employees (x : Nat) (h1 : x < 13) : 13 + 6 * x = 85 :=
by
  sorry

end total_employees_l46_46936


namespace sum_of_coordinates_of_X_l46_46241

theorem sum_of_coordinates_of_X 
  (X Y Z : ℝ × ℝ)
  (h1 : dist X Z / dist X Y = 1 / 2)
  (h2 : dist Z Y / dist X Y = 1 / 2)
  (hY : Y = (1, 7))
  (hZ : Z = (-1, -7)) :
  (X.1 + X.2) = -24 :=
sorry

end sum_of_coordinates_of_X_l46_46241


namespace temperature_at_noon_l46_46323

-- Definitions of the given conditions.
def morning_temperature : ℝ := 4
def temperature_drop : ℝ := 10

-- The theorem statement that needs to be proven.
theorem temperature_at_noon : morning_temperature - temperature_drop = -6 :=
by
  -- The proof can be filled in by solving the stated theorem.
  sorry

end temperature_at_noon_l46_46323


namespace melanie_correct_coins_and_value_l46_46871

def melanie_coins_problem : Prop :=
let dimes_initial := 19
let dimes_dad := 39
let dimes_sister := 15
let dimes_mother := 25
let total_dimes := dimes_initial + dimes_dad + dimes_sister + dimes_mother

let nickels_initial := 12
let nickels_dad := 22
let nickels_sister := 7
let nickels_mother := 10
let nickels_grandmother := 30
let total_nickels := nickels_initial + nickels_dad + nickels_sister + nickels_mother + nickels_grandmother

let quarters_initial := 8
let quarters_dad := 15
let quarters_sister := 12
let quarters_grandmother := 3
let total_quarters := quarters_initial + quarters_dad + quarters_sister + quarters_grandmother

let dimes_value := total_dimes * 0.10
let nickels_value := total_nickels * 0.05
let quarters_value := total_quarters * 0.25
let total_value := dimes_value + nickels_value + quarters_value

total_dimes = 98 ∧ total_nickels = 81 ∧ total_quarters = 38 ∧ total_value = 23.35

theorem melanie_correct_coins_and_value : melanie_coins_problem :=
by sorry

end melanie_correct_coins_and_value_l46_46871


namespace tapB_fill_in_20_l46_46020

-- Conditions definitions
def tapA_rate (A: ℝ) : Prop := A = 3 -- Tap A fills 3 liters per minute
def total_volume (V: ℝ) : Prop := V = 36 -- Total bucket volume is 36 liters
def together_fill_time (t: ℝ) : Prop := t = 10 -- Both taps fill the bucket in 10 minutes

-- Tap B's rate can be derived from these conditions
def tapB_rate (B: ℝ) (A: ℝ) (V: ℝ) (t: ℝ) : Prop := V - (A * t) = B * t

-- The final question we need to prove
theorem tapB_fill_in_20 (B: ℝ) (A: ℝ) (V: ℝ) (t: ℝ) : 
  tapA_rate A → total_volume V → together_fill_time t → tapB_rate B A V t → B * 20 = 12 := by
  sorry

end tapB_fill_in_20_l46_46020


namespace hari_contribution_l46_46275

theorem hari_contribution (c_p: ℕ) (m_p: ℕ) (ratio_p: ℕ) 
                          (m_h: ℕ) (ratio_h: ℕ) (profit_ratio_p: ℕ) (profit_ratio_h: ℕ) 
                          (c_h: ℕ) : 
  (c_p = 3780) → 
  (m_p = 12) → 
  (ratio_p = 2) → 
  (m_h = 7) → 
  (ratio_h = 3) → 
  (profit_ratio_p = 2) →
  (profit_ratio_h = 3) →
  (c_p * m_p * profit_ratio_h) = (c_h * m_h * profit_ratio_p) → 
  c_h = 9720 :=
by
  intros
  sorry

end hari_contribution_l46_46275


namespace max_ab_l46_46750

theorem max_ab (a b : ℝ) (h : 4 * a + b = 1) (ha : a > 0) (hb : b > 0) : ab <= 1 / 16 :=
sorry

end max_ab_l46_46750


namespace count_valid_combinations_l46_46192

-- Define the digits condition
def is_digit (d : ℕ) : Prop := d >= 0 ∧ d <= 9

-- Define the main proof statement
theorem count_valid_combinations (a b c: ℕ) (h1 : is_digit a)(h2 : is_digit b)(h3 : is_digit c) :
    (100 * a + 10 * b + c) + (100 * c + 10 * b + a) = 1069 → 
    ∃ (abc_combinations : ℕ), abc_combinations = 8 :=
by
  sorry

end count_valid_combinations_l46_46192


namespace optimal_fruit_combination_l46_46365

structure FruitPrices :=
  (price_2_apples : ℕ)
  (price_6_apples : ℕ)
  (price_12_apples : ℕ)
  (price_2_oranges : ℕ)
  (price_6_oranges : ℕ)
  (price_12_oranges : ℕ)

def minCostFruits : ℕ :=
  sorry

theorem optimal_fruit_combination (fp : FruitPrices) (total_fruits : ℕ)
  (mult_2_or_3 : total_fruits = 15) :
  fp.price_2_apples = 48 →
  fp.price_6_apples = 126 →
  fp.price_12_apples = 224 →
  fp.price_2_oranges = 60 →
  fp.price_6_oranges = 164 →
  fp.price_12_oranges = 300 →
  minCostFruits = 314 :=
by
  sorry

end optimal_fruit_combination_l46_46365


namespace divides_n3_minus_7n_l46_46178

theorem divides_n3_minus_7n (n : ℕ) : 6 ∣ n^3 - 7 * n := 
sorry

end divides_n3_minus_7n_l46_46178


namespace polynomial_coefficient_sum_l46_46479

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (1 - 2 * x) ^ 4 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 →
  a₀ + a₁ + a₃ = -39 :=
by
  sorry

end polynomial_coefficient_sum_l46_46479


namespace cos_alpha_beta_value_l46_46377

theorem cos_alpha_beta_value
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : -π / 2 < β ∧ β < 0)
  (h3 : Real.cos (π / 4 + α) = 1 / 3)
  (h4 : Real.cos (π / 4 - β) = Real.sqrt 3 / 3) :
  Real.cos (α + β) = (5 * Real.sqrt 3) / 9 := 
by
  sorry

end cos_alpha_beta_value_l46_46377


namespace rectangle_area_perimeter_l46_46787

-- Defining the problem conditions
def positive_int (n : Int) : Prop := n > 0

-- The main statement of the problem
theorem rectangle_area_perimeter (a b : Int) (h1 : positive_int a) (h2 : positive_int b) : 
  ¬ (a + 2) * (b + 2) - 4 = 146 :=
by
  sorry

end rectangle_area_perimeter_l46_46787


namespace magnitude_diff_is_correct_l46_46701

def vector_a : ℝ × ℝ × ℝ := (2, -3, 1)
def vector_b : ℝ × ℝ × ℝ := (-1, 1, -4)

theorem magnitude_diff_is_correct : 
  ‖(2, -3, 1) - (-1, 1, -4)‖ = 5 * Real.sqrt 2 := 
by
  sorry

end magnitude_diff_is_correct_l46_46701


namespace solve_for_x_l46_46150

theorem solve_for_x (x : ℝ) (h : 3 + 5 * x = 28) : x = 5 :=
by {
  sorry
}

end solve_for_x_l46_46150


namespace scale_of_map_l46_46997

theorem scale_of_map 
  (map_distance : ℝ)
  (travel_time : ℝ)
  (average_speed : ℝ)
  (actual_distance : ℝ)
  (scale : ℝ)
  (h1 : map_distance = 5)
  (h2 : travel_time = 6.5)
  (h3 : average_speed = 60)
  (h4 : actual_distance = average_speed * travel_time)
  (h5 : scale = map_distance / actual_distance) :
  scale = 0.01282 :=
by
  sorry

end scale_of_map_l46_46997


namespace cross_shape_rectangle_count_l46_46094

def original_side_length := 30
def smallest_square_side_length := 1
def cut_corner_length := 10
def N : ℕ := sorry  -- total number of rectangles in the resultant graph paper
def result : ℕ := 14413

theorem cross_shape_rectangle_count :
  (1/10 : ℚ) * N = result := 
sorry

end cross_shape_rectangle_count_l46_46094


namespace sufficiency_and_necessity_of_p_and_q_l46_46089

noncomputable def p : Prop := ∀ k, k = Real.sqrt 3
noncomputable def q : Prop := ∀ k, ∃ y x, y = k * x + 2 ∧ x^2 + y^2 = 1

theorem sufficiency_and_necessity_of_p_and_q : (p → q) ∧ (¬ (q → p)) := by
  sorry

end sufficiency_and_necessity_of_p_and_q_l46_46089


namespace correct_transformation_l46_46483

-- Given transformations
def transformation_A (a : ℝ) : Prop := - (1 / a) = -1 / a
def transformation_B (a b : ℝ) : Prop := (1 / a) + (1 / b) = 1 / (a + b)
def transformation_C (a b : ℝ) : Prop := (2 * b^2) / a^2 = (2 * b) / a
def transformation_D (a b : ℝ) : Prop := (a + a * b) / (b + a * b) = a / b

-- Correct transformation is A.
theorem correct_transformation (a b : ℝ) : transformation_A a ∧ ¬transformation_B a b ∧ ¬transformation_C a b ∧ ¬transformation_D a b :=
sorry

end correct_transformation_l46_46483


namespace bose_einstein_distribution_diagrams_fermi_dirac_distribution_diagrams_l46_46066

/-- 
To prove the number of distinct distribution diagrams for particles following the 
Bose-Einstein distribution, satisfying the given conditions. 
-/
theorem bose_einstein_distribution_diagrams (n_particles : ℕ) (total_energy : ℕ) 
  (energy_unit : ℕ) : n_particles = 4 → total_energy = 4 * energy_unit → 
  ∃ (distinct_diagrams : ℕ), distinct_diagrams = 72 := 
  by
  sorry

/-- 
To prove the number of distinct distribution diagrams for particles following the 
Fermi-Dirac distribution, satisfying the given conditions. 
-/
theorem fermi_dirac_distribution_diagrams (n_particles : ℕ) (total_energy : ℕ) 
  (energy_unit : ℕ) : n_particles = 4 → total_energy = 4 * energy_unit → 
  ∃ (distinct_diagrams : ℕ), distinct_diagrams = 246 := 
  by
  sorry

end bose_einstein_distribution_diagrams_fermi_dirac_distribution_diagrams_l46_46066


namespace translate_line_downwards_l46_46565

theorem translate_line_downwards :
  ∀ (x : ℝ), (∀ (y : ℝ), (y = 2 * x + 1) → (y - 2 = 2 * x - 1)) :=
by
  intros x y h
  rw [h]
  sorry

end translate_line_downwards_l46_46565


namespace minimum_vehicles_l46_46823

theorem minimum_vehicles (students adults : ℕ) (van_capacity minibus_capacity : ℕ)
    (severe_allergies_students : ℕ) (vehicle_requires_adult : Prop)
    (h_students : students = 24) (h_adults : adults = 3)
    (h_van_capacity : van_capacity = 8) (h_minibus_capacity : minibus_capacity = 14)
    (h_severe_allergies_students : severe_allergies_students = 2)
    (h_vehicle_requires_adult : vehicle_requires_adult)
    : ∃ (min_vehicles : ℕ), min_vehicles = 5 :=
by
  sorry

end minimum_vehicles_l46_46823


namespace total_toothpicks_in_grid_l46_46142

theorem total_toothpicks_in_grid (l w : ℕ) (h₁ : l = 50) (h₂ : w = 20) : 
  (l + 1) * w + (w + 1) * l + 2 * (l * w) = 4070 :=
by
  sorry

end total_toothpicks_in_grid_l46_46142


namespace union_of_sets_l46_46418

open Set

variable (a : ℤ)

def setA : Set ℤ := {1, 3}
def setB (a : ℤ) : Set ℤ := {a + 2, 5}

theorem union_of_sets (h : {3} = setA ∩ setB a) : setA ∪ setB a = {1, 3, 5} :=
by
  sorry

end union_of_sets_l46_46418


namespace chicken_rabbit_problem_l46_46212

theorem chicken_rabbit_problem (x y : ℕ) (h1 : x + y = 35) (h2 : 2 * x + 4 * y = 94) : x = 23 ∧ y = 12 :=
by
  sorry

end chicken_rabbit_problem_l46_46212


namespace complement_unions_subset_condition_l46_46431

open Set

-- Condition Definitions
def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 3 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | x < a + 1}

-- Questions Translated to Lean Statements
theorem complement_unions (U : Set ℝ)
  (hU : U = univ) : (compl A ∪ compl B) = compl (A ∩ B) := by sorry

theorem subset_condition (a : ℝ)
  (h : B ⊆ C a) : a ≥ 8 := by sorry

end complement_unions_subset_condition_l46_46431


namespace point_A_in_QuadrantIII_l46_46552

-- Define the Cartesian Point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define the condition for point being in Quadrant III
def inQuadrantIII (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

-- Given point A
def A : Point := { x := -1, y := -2 }

-- The theorem stating that point A lies in Quadrant III
theorem point_A_in_QuadrantIII : inQuadrantIII A :=
  by
    sorry

end point_A_in_QuadrantIII_l46_46552


namespace total_weight_of_family_l46_46306

theorem total_weight_of_family (M D C : ℕ) (h1 : D + C = 60) (h2 : C = 1 / 5 * M) (h3 : D = 40) : M + D + C = 160 :=
sorry

end total_weight_of_family_l46_46306


namespace count_zero_vectors_l46_46061

variable {V : Type} [AddCommGroup V]

variables (A B C D M O : V)

def vector_expressions_1 := (A - B) + (B - C) + (C - A) = 0
def vector_expressions_2 := (A - B) + (M - B) + (B - O) + (O - M) ≠ 0
def vector_expressions_3 := (A - B) - (A - C) + (B - D) - (C - D) = 0
def vector_expressions_4 := (O - A) + (O - C) + (B - O) + (C - O) ≠ 0

theorem count_zero_vectors :
  (vector_expressions_1 A B C) ∧
  (vector_expressions_2 A B M O) ∧
  (vector_expressions_3 A B C D) ∧
  (vector_expressions_4 O A C B) →
  (2 = 2) :=
sorry

end count_zero_vectors_l46_46061


namespace unit_digit_of_fourth_number_l46_46269

theorem unit_digit_of_fourth_number
  (n1 n2 n3 n4 : ℕ)
  (h1 : n1 % 10 = 4)
  (h2 : n2 % 10 = 8)
  (h3 : n3 % 10 = 3)
  (h4 : (n1 * n2 * n3 * n4) % 10 = 8) : 
  n4 % 10 = 3 :=
sorry

end unit_digit_of_fourth_number_l46_46269


namespace sum_of_possible_values_d_l46_46919

theorem sum_of_possible_values_d :
  let range_8 := (512, 4095)
  let digits_in_base_16 := 3
  (∀ n, n ∈ Set.Icc range_8.1 range_8.2 → (Nat.digits 16 n).length = digits_in_base_16)
  → digits_in_base_16 = 3 :=
by
  sorry

end sum_of_possible_values_d_l46_46919


namespace quadrilateral_side_difference_l46_46879

variable (a b c d : ℝ)

theorem quadrilateral_side_difference :
  a + b + c + d = 120 →
  a + c = 50 →
  (a^2 + c^2 = 1600) →
  (b + d = 70 ∧ b * d = 450) →
  |b - d| = 2 * Real.sqrt 775 :=
by
  intros ha hb hc hd
  sorry

end quadrilateral_side_difference_l46_46879


namespace tan_alpha_solution_l46_46547

theorem tan_alpha_solution (α : Real) (h : Real.tan α = 1 / 2) : 
  (2 * Real.sin α + Real.cos α) / (4 * Real.sin α - Real.cos α) = 2 := 
by 
  sorry

end tan_alpha_solution_l46_46547


namespace wrench_weight_relation_l46_46627

variables (h w : ℕ)

theorem wrench_weight_relation (h w : ℕ) 
  (cond : 2 * h + 2 * w = (1 / 3) * (8 * h + 5 * w)) : w = 2 * h := 
by sorry

end wrench_weight_relation_l46_46627


namespace number_of_77s_l46_46230

theorem number_of_77s (a b : ℕ) :
  (∃ a : ℕ, 1015 = a + 3 * 77 ∧ a + 21 = 10)
  ∧ (∃ b : ℕ, 2023 = b + 6 * 77 + 2 * 777 ∧ b = 7)
  → 6 = 6 := 
by
    sorry

end number_of_77s_l46_46230


namespace convex_quadrilateral_inequality_l46_46088

variable (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]

theorem convex_quadrilateral_inequality
    (AB CD BC AD AC BD : ℝ)
    (h : AB * CD + BC * AD >= AC * BD)
    (convex_quadrilateral : Prop) :
  AB * CD + BC * AD >= AC * BD :=
by
  sorry

end convex_quadrilateral_inequality_l46_46088


namespace nylon_cord_length_l46_46019

theorem nylon_cord_length {L : ℝ} (hL : L = 30) : ∃ (w : ℝ), w = 5 := 
by sorry

end nylon_cord_length_l46_46019


namespace line_through_parabola_intersects_vertex_l46_46438

theorem line_through_parabola_intersects_vertex (y x k : ℝ) :
  (y = 6 * x ∨ 6 * x - 5 * y - 24 = 0) ∧ 
  (∃ P Q : ℝ × ℝ, (P.1)^2 = 4 * P.2 ∧ (Q.1)^2 = 4 * Q.2 ∧ 
   (P = (0, 0) ∨ Q = (0, 0)) ∧ 
   (y = 6 * x ∨ 6 * x - 5 * y - 24 = 0)) := sorry

end line_through_parabola_intersects_vertex_l46_46438


namespace pyramid_volume_correct_l46_46770

noncomputable def PyramidVolume (base_area : ℝ) (triangle_area_1 : ℝ) (triangle_area_2 : ℝ) : ℝ :=
  let side := Real.sqrt base_area
  let height_1 := (2 * triangle_area_1) / side
  let height_2 := (2 * triangle_area_2) / side
  let h_sq := height_1 ^ 2 - (Real.sqrt (height_1 ^ 2 + height_2 ^ 2 - 512)) ^ 2
  let height := Real.sqrt h_sq
  (1/3) * base_area * height

theorem pyramid_volume_correct :
  PyramidVolume 256 120 112 = 1163 := by
  sorry

end pyramid_volume_correct_l46_46770


namespace complement_intersection_l46_46394

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_intersection : 
  U = {1, 2, 3, 4, 5} → 
  A = {1, 2, 3} → 
  B = {2, 3, 4} → 
  (U \ (A ∩ B) = {1, 4, 5}) := 
by
  sorry

end complement_intersection_l46_46394


namespace nine_chapters_compensation_difference_l46_46021

noncomputable def pig_consumes (x : ℝ) := x
noncomputable def sheep_consumes (x : ℝ) := 2 * x
noncomputable def horse_consumes (x : ℝ) := 4 * x
noncomputable def cow_consumes (x : ℝ) := 8 * x

theorem nine_chapters_compensation_difference :
  ∃ (x : ℝ), 
    cow_consumes x + horse_consumes x + sheep_consumes x + pig_consumes x = 9 ∧
    (horse_consumes x - pig_consumes x) = 9 / 5 :=
by
  sorry

end nine_chapters_compensation_difference_l46_46021


namespace alcohol_by_volume_l46_46990

/-- Solution x is 10% alcohol by volume and is 50 ml.
    Solution y is 30% alcohol by volume and is 150 ml.
    We must prove the final solution is 25% alcohol by volume. -/
theorem alcohol_by_volume (vol_x vol_y : ℕ) (conc_x conc_y : ℕ) (vol_mix : ℕ) (conc_mix : ℕ) :
  vol_x = 50 →
  conc_x = 10 →
  vol_y = 150 →
  conc_y = 30 →
  vol_mix = vol_x + vol_y →
  conc_mix = 100 * (vol_x * conc_x + vol_y * conc_y) / vol_mix →
  conc_mix = 25 :=
by
  intros h1 h2 h3 h4 h5 h_cons
  sorry

end alcohol_by_volume_l46_46990


namespace compound_interest_calculation_l46_46897

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  let A := P * ((1 + r / (n : ℝ)) ^ (n * t))
  A - P

theorem compound_interest_calculation :
  compoundInterest 500 0.05 1 5 = 138.14 := by
  sorry

end compound_interest_calculation_l46_46897


namespace find_sin_E_floor_l46_46393

variable {EF GH EH FG : ℝ}
variable (E G : ℝ)

-- Conditions from the problem
def is_convex_quadrilateral (EF GH EH FG : ℝ) : Prop := true
def angles_congruent (E G : ℝ) : Prop := E = G
def sides_equal (EF GH : ℝ) : Prop := EF = GH ∧ EF = 200
def sides_not_equal (EH FG : ℝ) : Prop := EH ≠ FG
def perimeter (EF GH EH FG : ℝ) : Prop := EF + GH + EH + FG = 800

-- The theorem to be proved
theorem find_sin_E_floor (h_convex : is_convex_quadrilateral EF GH EH FG)
                         (h_angles : angles_congruent E G)
                         (h_sides : sides_equal EF GH)
                         (h_sides_ne : sides_not_equal EH FG)
                         (h_perimeter : perimeter EF GH EH FG) :
  ⌊ 1000 * Real.sin E ⌋ = 0 := by
  sorry

end find_sin_E_floor_l46_46393


namespace maggie_remaining_goldfish_l46_46730

theorem maggie_remaining_goldfish
  (total_goldfish : ℕ)
  (allowed_fraction : ℕ → ℕ)
  (caught_fraction : ℕ → ℕ)
  (halfsies : ℕ)
  (remaining_goldfish : ℕ)
  (h1 : total_goldfish = 100)
  (h2 : allowed_fraction total_goldfish = total_goldfish / 2)
  (h3 : caught_fraction (allowed_fraction total_goldfish) = (3 * allowed_fraction total_goldfish) / 5)
  (h4 : halfsies = allowed_fraction total_goldfish)
  (h5 : remaining_goldfish = halfsies - caught_fraction halfsies) :
  remaining_goldfish = 20 :=
sorry

end maggie_remaining_goldfish_l46_46730


namespace seashells_given_joan_to_mike_l46_46571

-- Declaring the context for the problem: Joan's seashells
def initial_seashells := 79
def remaining_seashells := 16

-- Proving how many seashells Joan gave to Mike
theorem seashells_given_joan_to_mike : (initial_seashells - remaining_seashells) = 63 :=
by
  -- This proof needs to be completed
  sorry

end seashells_given_joan_to_mike_l46_46571


namespace determine_number_l46_46477

noncomputable def is_valid_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧
  (∃ d1 d2 d3, 
    n = d1 * 100 + d2 * 10 + d3 ∧ 
    (
      (d1 = 5 ∨ d1 = 1 ∨ d1 = 5 ∨ d1 = 2) ∧
      (d2 = 4 ∨ d2 = 4 ∨ d2 = 4) ∧
      (d3 = 3 ∨ d3 = 2 ∨ d3 = 6)
    ) ∧
    (
      (d1 ≠ 1 ∧ d1 ≠ 2 ∧ d1 ≠ 6) ∧
      (d2 ≠ 5 ∧ d2 ≠ 4 ∧ d2 ≠ 6 ∧ d2 ≠ 2) ∧
      (d3 ≠ 5 ∧ d3 ≠ 4 ∧ d3 ≠ 1 ∧ d3 ≠ 2)
    )
  )

theorem determine_number : ∃ n : ℕ, is_valid_number n ∧ n = 163 :=
by 
  existsi 163
  unfold is_valid_number
  sorry

end determine_number_l46_46477


namespace team_CB_days_worked_together_l46_46755

def projectA := 1 -- Project A is 1 unit of work
def projectB := 5 / 4 -- Project B is 1.25 units of work
def work_rate_A := 1 / 20 -- Team A's work rate
def work_rate_B := 1 / 24 -- Team B's work rate
def work_rate_C := 1 / 30 -- Team C's work rate

noncomputable def combined_rate_without_C := work_rate_B + work_rate_C

noncomputable def combined_total_work := projectA + projectB

noncomputable def days_for_combined_work := combined_total_work / combined_rate_without_C

-- Statement to prove the number of days team C and team B worked together
theorem team_CB_days_worked_together : 
  days_for_combined_work = 15 := 
  sorry

end team_CB_days_worked_together_l46_46755


namespace binary_to_decimal_10101_l46_46813

theorem binary_to_decimal_10101 : (1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 21 :=
by
  sorry

end binary_to_decimal_10101_l46_46813


namespace locus_of_center_l46_46609

-- Define point A
def PointA : ℝ × ℝ := (-2, 0)

-- Define the tangent line
def TangentLine : ℝ := 2

-- The condition to prove the locus equation
theorem locus_of_center (x₀ y₀ : ℝ) :
  (∃ r : ℝ, abs (x₀ - TangentLine) = r ∧ (x₀ + 2)^2 + y₀^2 = r^2) →
  y₀^2 = -8 * x₀ := by
  sorry

end locus_of_center_l46_46609


namespace geom_sequence_sum_of_first4_l46_46018

noncomputable def geom_sum_first4_terms (a : ℕ → ℝ) (common_ratio : ℝ) (a0 a1 a4 : ℝ) : ℝ :=
  a0 + a0 * common_ratio + a0 * common_ratio^2 + a0 * common_ratio^3

theorem geom_sequence_sum_of_first4 {a : ℕ → ℝ} (a1 a4 : ℝ) (r : ℝ)
  (h1 : a 1 = a1) (h4 : a 4 = a4) 
  (h_geom : ∀ n, a (n + 1) = a n * r) :
  geom_sum_first4_terms a (r) a1 (a 0) (a 4) = 120 :=
by sorry

end geom_sequence_sum_of_first4_l46_46018


namespace age_of_youngest_child_l46_46974

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 50) : x = 6 :=
sorry

end age_of_youngest_child_l46_46974


namespace harrison_grade_levels_l46_46720

theorem harrison_grade_levels
  (total_students : ℕ)
  (percent_moving : ℚ)
  (advanced_class_size : ℕ)
  (num_normal_classes : ℕ)
  (normal_class_size : ℕ)
  (students_moving : ℕ)
  (students_per_grade_level : ℕ)
  (grade_levels : ℕ) :
  total_students = 1590 →
  percent_moving = 40 / 100 →
  advanced_class_size = 20 →
  num_normal_classes = 6 →
  normal_class_size = 32 →
  students_moving = total_students * percent_moving →
  students_per_grade_level = advanced_class_size + num_normal_classes * normal_class_size →
  grade_levels = students_moving / students_per_grade_level →
  grade_levels = 3 :=
by
  intros
  sorry

end harrison_grade_levels_l46_46720


namespace shorter_side_length_l46_46692

theorem shorter_side_length 
  (L W : ℝ) 
  (h1 : L * W = 117) 
  (h2 : 2 * L + 2 * W = 44) :
  L = 9 ∨ W = 9 :=
by
  sorry

end shorter_side_length_l46_46692


namespace sum_of_squares_and_product_l46_46110

theorem sum_of_squares_and_product
  (x y : ℕ) (h1 : x^2 + y^2 = 181) (h2 : x * y = 90) : x + y = 19 := by
  sorry

end sum_of_squares_and_product_l46_46110


namespace number_of_machines_in_first_scenario_l46_46302

noncomputable def machine_work_rate (R : ℝ) (hours_per_job : ℝ) : Prop :=
  (6 * R * 8 = 1)

noncomputable def machines_first_scenario (M : ℝ) (R : ℝ) (hours_per_job_first : ℝ) : Prop :=
  (M * R * hours_per_job_first = 1)

theorem number_of_machines_in_first_scenario (M : ℝ) (R : ℝ) :
  machine_work_rate R 8 ∧ machines_first_scenario M R 6 -> M = 8 :=
sorry

end number_of_machines_in_first_scenario_l46_46302


namespace calc_expression_l46_46331

theorem calc_expression :
  (12^4 + 375) * (24^4 + 375) * (36^4 + 375) * (48^4 + 375) * (60^4 + 375) /
  ((6^4 + 375) * (18^4 + 375) * (30^4 + 375) * (42^4 + 375) * (54^4 + 375)) = 159 :=
by
  sorry

end calc_expression_l46_46331


namespace monkey_climbing_time_l46_46473

-- Define the conditions
def tree_height : ℕ := 20
def hop_distance : ℕ := 3
def slip_distance : ℕ := 2
def net_distance_per_hour : ℕ := hop_distance - slip_distance

-- Define the theorem statement
theorem monkey_climbing_time : ∃ (t : ℕ), t = 18 ∧ (net_distance_per_hour * (t - 1) + hop_distance) >= tree_height :=
by
  sorry

end monkey_climbing_time_l46_46473


namespace dolphins_points_l46_46193

variable (S D : ℕ)

theorem dolphins_points :
  (S + D = 36) ∧ (S = D + 12) → D = 12 :=
by
  sorry

end dolphins_points_l46_46193


namespace remainder_when_sum_divided_by_15_l46_46155

theorem remainder_when_sum_divided_by_15 (a b c : ℕ) 
  (h1 : a % 15 = 11) 
  (h2 : b % 15 = 12) 
  (h3 : c % 15 = 13) : 
  (a + b + c) % 15 = 6 :=
  sorry

end remainder_when_sum_divided_by_15_l46_46155


namespace range_of_2a_plus_3b_l46_46234

theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 ≤ a + b ∧ a + b ≤ 1) (h2 : -1 ≤ a - b ∧ a - b ≤ 1) :
  -3 ≤ 2 * a + 3 * b ∧ 2 * a + 3 * b ≤ 3 :=
by
  sorry

end range_of_2a_plus_3b_l46_46234


namespace sum_of_digits_l46_46457

noncomputable def A : ℕ := 3
noncomputable def B : ℕ := 9
noncomputable def C : ℕ := 2
noncomputable def BC : ℕ := B * 10 + C
noncomputable def ABC : ℕ := A * 100 + B * 10 + C

theorem sum_of_digits (H1: A ≠ 0) (H2: B ≠ 0) (H3: C ≠ 0) (H4: BC + ABC + ABC = 876):
  A + B + C = 14 :=
sorry

end sum_of_digits_l46_46457


namespace wrestling_match_student_count_l46_46100

theorem wrestling_match_student_count (n : ℕ) (h : n * (n - 1) / 2 = 91) : n = 14 := by
  sorry

end wrestling_match_student_count_l46_46100


namespace Tucker_last_number_l46_46819

-- Define the sequence of numbers said by Todd, Tadd, and Tucker
def game_sequence (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 3
  else if n = 4 then 4
  else if n = 5 then 5
  else if n = 6 then 6
  else sorry -- Define recursively for subsequent rounds

-- Condition: The game ends when they reach the number 1000.
def game_end := 1000

-- Define the function to determine the last number said by Tucker
def last_number_said_by_Tucker (end_num : ℕ) : ℕ :=
  -- Assuming this function correctly calculates the last number said by Tucker
  if end_num = game_end then 1000 else sorry

-- Problem statement to prove
theorem Tucker_last_number : last_number_said_by_Tucker game_end = 1000 := by
  sorry

end Tucker_last_number_l46_46819


namespace speed_of_sound_correct_l46_46017

-- Define the given conditions
def heard_second_blast_after : ℕ := 30 * 60 + 24 -- 30 minutes and 24 seconds in seconds
def time_sound_travelled : ℕ := 24 -- The sound traveled for 24 seconds
def distance_travelled : ℕ := 7920 -- Distance in meters

-- Define the expected answer for the speed of sound 
def expected_speed_of_sound : ℕ := 330 -- Speed in meters per second

-- The proposition that states the speed of sound given the conditions
theorem speed_of_sound_correct : (distance_travelled / time_sound_travelled) = expected_speed_of_sound := 
by {
  -- use division to compute the speed of sound
  sorry
}

end speed_of_sound_correct_l46_46017


namespace find_f11_l46_46628

-- Define the odd function properties
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the functional equation property
def functional_eqn (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = -f x

-- Define the specific values of the function on (0,2)
def specific_values (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2

-- The main theorem that needs to be proved
theorem find_f11 (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : functional_eqn f) (h3 : specific_values f) : 
  f 11 = -2 :=
sorry

end find_f11_l46_46628


namespace haley_small_gardens_l46_46771

theorem haley_small_gardens (total_seeds seeds_in_big_garden seeds_per_small_garden : ℕ) (h1 : total_seeds = 56) (h2 : seeds_in_big_garden = 35) (h3 : seeds_per_small_garden = 3) :
  (total_seeds - seeds_in_big_garden) / seeds_per_small_garden = 7 :=
by
  sorry

end haley_small_gardens_l46_46771


namespace cos_double_angle_l46_46010

theorem cos_double_angle (α : ℝ) (h : Real.cos α = 4 / 5) : Real.cos (2 * α) = 7 / 25 := 
by
  sorry

end cos_double_angle_l46_46010


namespace problem_1_problem_2_problem_3_l46_46883

def pair_otimes (a b c d : ℚ) : ℚ := b * c - a * d

-- Problem (1)
theorem problem_1 : pair_otimes 5 3 (-2) 1 = -11 := 
by 
  unfold pair_otimes 
  sorry

-- Problem (2)
theorem problem_2 (x : ℚ) (h : pair_otimes 2 (3 * x - 1) 6 (x + 2) = 22) : x = 2 := 
by 
  unfold pair_otimes at h
  sorry

-- Problem (3)
theorem problem_3 (x k : ℤ) (h : pair_otimes 4 (k - 2) x (2 * x - 1) = 6) : 
  k = 8 ∨ k = 9 ∨ k = 11 ∨ k = 12 := 
by 
  unfold pair_otimes at h
  sorry

end problem_1_problem_2_problem_3_l46_46883


namespace sum_of_primes_between_20_and_30_l46_46496

/-- Define what it means to be a prime number -/
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Define the predicate for numbers being between 20 and 30 -/
def between_20_and_30 (n : ℕ) : Prop :=
  20 < n ∧ n < 30

/-- List of prime numbers between 20 and 30 -/
def prime_list : List ℕ := [23, 29]

/-- The sum of elements in the prime list -/
def prime_sum : ℕ := prime_list.sum

/-- Prove that the sum of prime numbers between 20 and 30 is 52 -/
theorem sum_of_primes_between_20_and_30 :
  prime_sum = 52 :=
by
  sorry

end sum_of_primes_between_20_and_30_l46_46496


namespace circle_intersection_l46_46706

theorem circle_intersection : 
  ∀ (O : ℝ × ℝ), ∃ (m n : ℤ), (dist (O.1, O.2) (m, n) ≤ 100 + 1/14) := 
sorry

end circle_intersection_l46_46706


namespace campers_afternoon_l46_46913

theorem campers_afternoon (total_campers morning_campers afternoon_campers : ℕ)
  (h1 : total_campers = 60)
  (h2 : morning_campers = 53)
  (h3 : afternoon_campers = total_campers - morning_campers) :
  afternoon_campers = 7 := by
  sorry

end campers_afternoon_l46_46913


namespace ellipse_equation_y_intercept_range_l46_46644

noncomputable def a := 2 * Real.sqrt 2
noncomputable def b := Real.sqrt 2
noncomputable def e := Real.sqrt 3 / 2
noncomputable def c := Real.sqrt 6
def M : ℝ × ℝ := (2, 1)

-- Condition: The ellipse equation form
def ellipse (x y : ℝ) : Prop := (x^2) / (a^2) + (y^2) / (b^2) = 1

-- Question 1: Proof that the ellipse equation is as given
theorem ellipse_equation :
  ellipse x y ↔ (x^2) / 8 + (y^2) / 2 = 1 := sorry

-- Condition: Line l is parallel to OM
def slope_OM := 1 / 2
def line_l (m x y : ℝ) : Prop := y = slope_OM * x + m

-- Question 2: Proof of the range for y-intercept m given the conditions
theorem y_intercept_range (m : ℝ) :
  (-Real.sqrt 2 < m ∧ m < 0 ∨ 0 < m ∧ m < Real.sqrt 2) ↔
  ∃ x1 y1 x2 y2,
    line_l m x1 y1 ∧ 
    line_l m x2 y2 ∧ 
    x1 ≠ x2 ∧ 
    y1 ≠ y2 ∧
    x1 * x2 + y1 * y2 < 0 := sorry

end ellipse_equation_y_intercept_range_l46_46644


namespace prove_3a_3b_3c_l46_46815

variable (a b c : ℝ)

def condition1 := b + c = 15 - 2 * a
def condition2 := a + c = -18 - 3 * b
def condition3 := a + b = 8 - 4 * c
def condition4 := a - b + c = 3

theorem prove_3a_3b_3c (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) (h4 : condition4 a b c) :
  3 * a + 3 * b + 3 * c = 24 / 5 :=
sorry

end prove_3a_3b_3c_l46_46815


namespace total_volume_of_four_cubes_is_500_l46_46878

-- Definitions for the problem assumptions
def edge_length := 5
def volume_of_cube (edge_length : ℕ) := edge_length ^ 3
def number_of_boxes := 4

-- Main statement to prove
theorem total_volume_of_four_cubes_is_500 :
  (volume_of_cube edge_length) * number_of_boxes = 500 :=
by
  -- Proof steps will go here
  sorry

end total_volume_of_four_cubes_is_500_l46_46878


namespace inequality1_inequality2_l46_46951

theorem inequality1 (x : ℝ) : 2 * x - 1 > x - 3 → x > -2 := by
  sorry

theorem inequality2 (x : ℝ) : 
  (x - 3 * (x - 2) ≥ 4) ∧ ((x - 1) / 5 < (x + 1) / 2) → -7 / 3 < x ∧ x ≤ 1 := by
  sorry

end inequality1_inequality2_l46_46951


namespace option_A_is_translation_l46_46522

-- Define what constitutes a translation transformation
def is_translation (description : String) : Prop :=
  description = "Pulling open a drawer"

-- Define each option
def option_A : String := "Pulling open a drawer"
def option_B : String := "Viewing text through a magnifying glass"
def option_C : String := "The movement of the minute hand on a clock"
def option_D : String := "You and the image in a plane mirror"

-- The main theorem stating that option A is the translation transformation
theorem option_A_is_translation : is_translation option_A :=
by
  -- skip the proof, adding sorry
  sorry

end option_A_is_translation_l46_46522


namespace total_area_of_forest_and_fields_l46_46348

theorem total_area_of_forest_and_fields (r p k : ℝ) (h1 : k = 12) 
  (h2 : r^2 + 4 * p^2 + 45 = 12 * k) :
  (r^2 + 4 * p^2 + 12 * k = 135) :=
by
  -- Proof goes here
  sorry

end total_area_of_forest_and_fields_l46_46348


namespace difference_twice_cecil_and_catherine_l46_46238

theorem difference_twice_cecil_and_catherine
  (Cecil Catherine Carmela : ℕ)
  (h1 : Cecil = 600)
  (h2 : Carmela = 2 * 600 + 50)
  (h3 : 600 + (2 * 600 - Catherine) + Carmela = 2800) :
  2 * 600 - Catherine = 250 := by
  sorry

end difference_twice_cecil_and_catherine_l46_46238


namespace question1_perpendicular_question2_parallel_l46_46397

structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def dot_product (v1 v2 : Vector2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

noncomputable def vector_k_a_plus_2_b (k : ℝ) (a b : Vector2D) : Vector2D :=
  ⟨k * a.x + 2 * b.x, k * a.y + 2 * b.y⟩

noncomputable def vector_2_a_minus_4_b (a b : Vector2D) : Vector2D :=
  ⟨2 * a.x - 4 * b.x, 2 * a.y - 4 * b.y⟩

def perpendicular (v1 v2 : Vector2D) : Prop :=
  dot_product v1 v2 = 0

def parallel (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y = v1.y * v2.x

def opposite_direction (v1 v2 : Vector2D) : Prop :=
  parallel v1 v2 ∧ v1.x * v2.x + v1.y * v2.y < 0

noncomputable def vector_a : Vector2D := ⟨1, 1⟩
noncomputable def vector_b : Vector2D := ⟨2, 3⟩

theorem question1_perpendicular (k : ℝ) : 
  perpendicular (vector_k_a_plus_2_b k vector_a vector_b) (vector_2_a_minus_4_b vector_a vector_b) ↔ 
  k = -21 / 4 :=
sorry

theorem question2_parallel (k : ℝ) :
  (parallel (vector_k_a_plus_2_b k vector_a vector_b) (vector_2_a_minus_4_b vector_a vector_b) ∧
  opposite_direction (vector_k_a_plus_2_b k vector_a vector_b) (vector_2_a_minus_4_b vector_a vector_b)) ↔ 
  k = -1 / 2 :=
sorry

end question1_perpendicular_question2_parallel_l46_46397


namespace product_of_tangents_is_constant_l46_46415

theorem product_of_tangents_is_constant (a b : ℝ) (h_ab : a > b) (P : ℝ × ℝ)
  (hP_on_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (A1 A2 : ℝ × ℝ)
  (hA1 : A1 = (-a, 0))
  (hA2 : A2 = (a, 0)) :
  ∃ (Q1 Q2 : ℝ × ℝ),
  (A1.1 - Q1.1, A2.1 - Q2.1) = (b^2, b^2) :=
sorry

end product_of_tangents_is_constant_l46_46415


namespace distance_between_homes_l46_46038

-- Define the conditions as Lean functions and values
def walking_speed_maxwell : ℝ := 3
def running_speed_brad : ℝ := 5
def distance_traveled_maxwell : ℝ := 15

-- State the theorem
theorem distance_between_homes : 
  ∃ D : ℝ, 
    (15 = walking_speed_maxwell * (distance_traveled_maxwell / walking_speed_maxwell)) ∧ 
    (D - 15 = running_speed_brad * (distance_traveled_maxwell / walking_speed_maxwell)) ∧ 
    D = 40 :=
by 
  sorry

end distance_between_homes_l46_46038


namespace seed_mixture_ryegrass_l46_46596

theorem seed_mixture_ryegrass (α : ℝ) :
  (0.4667 * 0.4 + 0.5333 * α = 0.32) -> α = 0.25 :=
by
  sorry

end seed_mixture_ryegrass_l46_46596


namespace proof_seq_l46_46512

open Nat

-- Definition of sequence {a_n}
def seq_a : ℕ → ℕ
| 0 => 1
| n + 1 => 3 * seq_a n

-- Definition of sum S_n of sequence {b_n}
def sum_S : ℕ → ℕ
| 0 => 0
| n + 1 => sum_S n + (2^n)

-- Definition of sequence {b_n}
def seq_b : ℕ → ℕ
| 0 => 1
| n + 1 => 2 * seq_b n

-- Definition of sequence {c_n}
def seq_c (n : ℕ) : ℕ := seq_b n * log 3 (seq_a n) -- Note: log base 3

-- Sum of first n terms of {c_n}
def sum_T : ℕ → ℕ
| 0 => 0
| n + 1 => sum_T n + seq_c n

-- Proof statement
theorem proof_seq (n : ℕ) :
  (seq_a n = 3 ^ n) ∧
  (2 * seq_b n - 1 = sum_S 0 * sum_S n) ∧
  (sum_T n = (n - 2) * 2 ^ (n + 2)) :=
sorry

end proof_seq_l46_46512


namespace train_crossing_time_approx_l46_46517

noncomputable def train_length : ℝ := 90 -- in meters
noncomputable def speed_kmh : ℝ := 124 -- in km/hr
noncomputable def conversion_factor : ℝ := 1000 / 3600 -- km/hr to m/s conversion factor
noncomputable def speed_ms : ℝ := speed_kmh * conversion_factor -- speed in m/s
noncomputable def time_to_cross : ℝ := train_length / speed_ms -- time in seconds

theorem train_crossing_time_approx :
  abs (time_to_cross - 2.61) < 0.01 := 
by 
  sorry

end train_crossing_time_approx_l46_46517


namespace derek_books_ratio_l46_46693

theorem derek_books_ratio :
  ∃ (T : ℝ), 960 - T - (1/4) * (960 - T) = 360 ∧ T / 960 = 1 / 2 :=
by
  sorry

end derek_books_ratio_l46_46693


namespace shirt_original_price_l46_46581

theorem shirt_original_price {P : ℝ} :
  (P * 0.80045740423098913 * 0.8745 = 105) → P = 150 :=
by sorry

end shirt_original_price_l46_46581


namespace original_number_of_men_l46_46184

theorem original_number_of_men (x : ℕ) (h1 : x * 50 = (x - 10) * 60) : x = 60 :=
by
  sorry

end original_number_of_men_l46_46184


namespace function_has_two_zeros_for_a_eq_2_l46_46782

noncomputable def f (a x : ℝ) : ℝ := a ^ x - x - 1

theorem function_has_two_zeros_for_a_eq_2 :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f 2 x1 = 0 ∧ f 2 x2 = 0) := sorry

end function_has_two_zeros_for_a_eq_2_l46_46782


namespace number_of_divisors_180_l46_46614

theorem number_of_divisors_180 : (∃ (n : ℕ), n = 180 ∧ (∀ (e1 e2 e3 : ℕ), 180 = 2^e1 * 3^e2 * 5^e3 → (e1 + 1) * (e2 + 1) * (e3 + 1) = 18)) :=
  sorry

end number_of_divisors_180_l46_46614


namespace sector_perimeter_l46_46605

theorem sector_perimeter (R : ℝ) (α : ℝ) (A : ℝ) (P : ℝ) : 
  A = (1 / 2) * R^2 * α → 
  α = 4 → 
  A = 2 → 
  P = 2 * R + R * α → 
  P = 6 := 
by
  intros hArea hAlpha hA hP
  sorry

end sector_perimeter_l46_46605


namespace square_of_neg_3b_l46_46568

theorem square_of_neg_3b (b : ℝ) : (-3 * b)^2 = 9 * b^2 :=
by sorry

end square_of_neg_3b_l46_46568


namespace money_difference_l46_46314

def share_ratio (w x y z : ℝ) (k : ℝ) : Prop :=
  w = k ∧ x = 6 * k ∧ y = 2 * k ∧ z = 4 * k

theorem money_difference (k : ℝ) (h : k = 375) : 
  ∀ w x y z : ℝ, share_ratio w x y z k → (x - y) = 1500 := 
by
  intros w x y z h_ratio
  rw [share_ratio] at h_ratio
  have h_w : w = k := h_ratio.1
  have h_x : x = 6 * k := h_ratio.2.1
  have h_y : y = 2 * k := h_ratio.2.2.1
  rw [h_x, h_y]
  rw [h] at h_x h_y
  sorry

end money_difference_l46_46314


namespace cd_value_l46_46197

theorem cd_value (a b c d : ℝ) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (ab ac bd : ℝ) 
  (h_ab : ab = 2) (h_ac : ac = 5) (h_bd : bd = 6) :
  ∃ (cd : ℝ), cd = 3 :=
by sorry

end cd_value_l46_46197


namespace coefficient_of_term_free_of_x_l46_46702

theorem coefficient_of_term_free_of_x 
  (n : ℕ) 
  (h1 : ∀ k : ℕ, k ≤ n → n = 10) 
  (h2 : (n.choose 4 / n.choose 2) = 14 / 3) : 
  ∃ (c : ℚ), c = 5 :=
by
  sorry

end coefficient_of_term_free_of_x_l46_46702


namespace intersection_complement_l46_46215

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_complement :
  A ∩ (U \ B) = {1, 3} :=
by {
  -- To ensure the validity of the theorem, the proof goes here
  sorry
}

end intersection_complement_l46_46215


namespace avg_first_3_is_6_l46_46465

theorem avg_first_3_is_6 (A B C D : ℝ) (X : ℝ)
  (h1 : (A + B + C) / 3 = X)
  (h2 : (B + C + D) / 3 = 5)
  (h3 : A + D = 11)
  (h4 : D = 4) :
  X = 6 := 
by
  sorry

end avg_first_3_is_6_l46_46465


namespace freshmen_sophomores_without_pets_l46_46768

theorem freshmen_sophomores_without_pets : 
  let total_students := 400
  let percentage_freshmen_sophomores := 0.50
  let percentage_with_pets := 1/5
  let freshmen_sophomores := percentage_freshmen_sophomores * total_students
  160 = (freshmen_sophomores - (percentage_with_pets * freshmen_sophomores)) :=
by
  sorry

end freshmen_sophomores_without_pets_l46_46768


namespace trapezoid_perimeter_l46_46719

-- Define the properties of the isosceles trapezoid
structure IsoscelesTrapezoid (A B C D : Type) :=
  (AB CD BC DA : ℝ)
  (AB_parallel_CD : AB = CD)
  (BC_eq_DA : BC = 13)
  (DA_eq_BC : DA = 13)
  (sum_AB_CD : AB + CD = 24)

-- Define the problem's conditions as Lean definitions
def trapezoidABCD : IsoscelesTrapezoid ℝ ℝ ℝ ℝ :=
{
  AB := 12,
  CD := 12,
  BC := 13,
  DA := 13,
  AB_parallel_CD := by sorry,
  BC_eq_DA := by sorry,
  DA_eq_BC := by sorry,
  sum_AB_CD := by sorry,
}

-- State the theorem we want to prove
theorem trapezoid_perimeter (trapezoid : IsoscelesTrapezoid ℝ ℝ ℝ ℝ) : 
  trapezoid.AB + trapezoid.BC + trapezoid.CD + trapezoid.DA = 50 :=
by sorry

end trapezoid_perimeter_l46_46719


namespace train_pass_time_l46_46280

theorem train_pass_time (train_length : ℕ) (platform_length : ℕ) (speed : ℕ) (h1 : train_length = 50) (h2 : platform_length = 100) (h3 : speed = 15) : 
  (train_length + platform_length) / speed = 10 :=
by
  sorry

end train_pass_time_l46_46280


namespace hearing_aid_cost_l46_46778

theorem hearing_aid_cost
  (cost : ℝ)
  (insurance_coverage : ℝ)
  (personal_payment : ℝ)
  (total_aid_count : ℕ)
  (h : total_aid_count = 2)
  (h_insurance : insurance_coverage = 0.80)
  (h_personal_payment : personal_payment = 1000)
  (h_equation : personal_payment = (1 - insurance_coverage) * (total_aid_count * cost)) :
  cost = 2500 :=
by
  sorry

end hearing_aid_cost_l46_46778


namespace worth_of_stuff_l46_46014

theorem worth_of_stuff (x : ℝ)
  (h1 : 1.05 * x - 8 = 34) :
  x = 40 :=
by
  sorry

end worth_of_stuff_l46_46014


namespace intersection_of_M_and_N_l46_46293

def M : Set ℤ := {0, 1}
def N : Set ℤ := {-1, 0}

theorem intersection_of_M_and_N : M ∩ N = {0} := by
  sorry

end intersection_of_M_and_N_l46_46293


namespace quadruple_application_of_h_l46_46718

-- Define the function as specified in the condition
def h (N : ℝ) : ℝ := 0.6 * N + 2

-- State the theorem
theorem quadruple_application_of_h : h (h (h (h 40))) = 9.536 :=
  by
    sorry

end quadruple_application_of_h_l46_46718


namespace sugar_solution_sweeter_l46_46885

theorem sugar_solution_sweeter (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m > 0) : 
    (b + m) / (a + m) > b / a :=
sorry

end sugar_solution_sweeter_l46_46885


namespace ratio_of_ages_l46_46402

variable (D R : ℕ)

theorem ratio_of_ages : (D = 9) → (R + 6 = 18) → (R / D = 4 / 3) :=
by
  intros hD hR
  -- proof goes here
  sorry

end ratio_of_ages_l46_46402


namespace total_pages_in_book_l46_46337

theorem total_pages_in_book (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) 
  (h1 : pages_per_day = 22) (h2 : days = 569) : total_pages = 12518 :=
by
  sorry

end total_pages_in_book_l46_46337


namespace angle_measure_l46_46009

-- Define the angle in degrees
def angle (x : ℝ) : Prop :=
  180 - x = 3 * (90 - x)

-- Desired proof statement
theorem angle_measure :
  ∀ (x : ℝ), angle x → x = 45 := by
  intros x h
  sorry

end angle_measure_l46_46009


namespace red_pigment_contribution_l46_46143

theorem red_pigment_contribution :
  ∀ (G : ℝ), (2 * G + G + 3 * G = 24) →
  (0.6 * (2 * G) + 0.5 * (3 * G) = 10.8) :=
by
  intro G
  intro h1
  sorry

end red_pigment_contribution_l46_46143


namespace chord_square_l46_46340

/-- 
Circles with radii 3 and 6 are externally tangent and are internally tangent to a circle with radius 9. 
The circle with radius 9 has a chord that is a common external tangent of the other two circles. Prove that 
the square of the length of this chord is 72.
-/
theorem chord_square (O₁ O₂ O₃ : Type) 
  (r₁ r₂ r₃ : ℝ) 
  (O₁_tangent_O₂ : r₁ + r₂ = 9) 
  (O₃_tangent_O₁ : r₃ - r₁ = 6) 
  (O₃_tangent_O₂ : r₃ - r₂ = 3) 
  (tangent_chord : ℝ) : 
  tangent_chord^2 = 72 :=
by sorry

end chord_square_l46_46340


namespace excluded_numbers_range_l46_46890

theorem excluded_numbers_range (S S' E : ℕ) (h1 : S = 31 * 10) (h2 : S' = 28 * 8) (h3 : E = S - S') (h4 : E > 70) :
  ∀ (x y : ℕ), x + y = E → 1 ≤ x ∧ x ≤ 85 ∧ 1 ≤ y ∧ y ≤ 85 := by
  sorry

end excluded_numbers_range_l46_46890


namespace train_crossing_time_l46_46468

def train_length : ℕ := 1000
def train_speed_km_per_h : ℕ := 18
def train_speed_m_per_s := train_speed_km_per_h * 1000 / 3600

theorem train_crossing_time :
  train_length / train_speed_m_per_s = 200 := by
sorry

end train_crossing_time_l46_46468


namespace slices_needed_l46_46185

def slices_per_sandwich : ℕ := 3
def number_of_sandwiches : ℕ := 5

theorem slices_needed : slices_per_sandwich * number_of_sandwiches = 15 :=
by {
  sorry
}

end slices_needed_l46_46185


namespace distance_between_circle_centers_l46_46239

open Real

theorem distance_between_circle_centers :
  let center1 := (1 / 2, 0)
  let center2 := (0, 1 / 2)
  dist center1 center2 = sqrt 2 / 2 :=
by
  sorry

end distance_between_circle_centers_l46_46239


namespace solve_for_x_l46_46037

variable (a b x : ℝ)
variable (a_pos : a > 0) (b_pos : b > 0) (x_pos : x > 0)

theorem solve_for_x : (3 * a) ^ (3 * b) = (a ^ b) * (x ^ b) → x = 27 * a ^ 2 :=
by
  intro h_eq
  sorry

end solve_for_x_l46_46037


namespace translation_preserves_coordinates_l46_46664

-- Given coordinates of point P
def point_P : (Int × Int) := (-2, 3)

-- Translating point P 3 units in the positive direction of the x-axis
def translate_x (p : Int × Int) (dx : Int) : (Int × Int) := 
  (p.1 + dx, p.2)

-- Translating point P 2 units in the negative direction of the y-axis
def translate_y (p : Int × Int) (dy : Int) : (Int × Int) := 
  (p.1, p.2 - dy)

-- Final coordinates after both translations
def final_coordinates (p : Int × Int) (dx dy : Int) : (Int × Int) := 
  translate_y (translate_x p dx) dy

theorem translation_preserves_coordinates :
  final_coordinates point_P 3 2 = (1, 1) :=
by
  sorry

end translation_preserves_coordinates_l46_46664


namespace convert_to_spherical_l46_46954

noncomputable def rectangular_to_spherical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2 + z^2)
  let φ := Real.arccos (z / ρ)
  let θ := if x = 0 ∧ y > 0 then Real.pi / 2
           else if x = 0 ∧ y < 0 then 3 * Real.pi / 2
           else if x > 0 then Real.arctan (y / x)
           else if y >= 0 then Real.arctan (y / x) + Real.pi
           else Real.arctan (y / x) - Real.pi
  (ρ, θ, φ)

theorem convert_to_spherical :
  rectangular_to_spherical (3 * Real.sqrt 2) (-4) 5 =
  (Real.sqrt 59, 2 * Real.pi + Real.arctan ((-4) / (3 * Real.sqrt 2)), Real.arccos (5 / Real.sqrt 59)) :=
by
  sorry

end convert_to_spherical_l46_46954


namespace total_weight_l46_46845

def w1 : ℝ := 9.91
def w2 : ℝ := 4.11

theorem total_weight : w1 + w2 = 14.02 := by 
  sorry

end total_weight_l46_46845


namespace day_of_week_after_10_pow_90_days_l46_46619

theorem day_of_week_after_10_pow_90_days :
  let initial_day := "Friday"
  ∃ day_after_10_pow_90 : String,
  day_after_10_pow_90 = "Saturday" :=
by
  sorry

end day_of_week_after_10_pow_90_days_l46_46619


namespace chapters_page_difference_l46_46219

def chapter1_pages : ℕ := 37
def chapter2_pages : ℕ := 80

theorem chapters_page_difference : chapter2_pages - chapter1_pages = 43 := by
  -- Proof goes here
  sorry

end chapters_page_difference_l46_46219


namespace sequence_sum_problem_l46_46678

theorem sequence_sum_problem :
  let seq := [72, 76, 80, 84, 88, 92, 96, 100, 104, 108]
  3 * (seq.sum) = 2700 :=
by
  sorry

end sequence_sum_problem_l46_46678


namespace at_least_26_equal_differences_l46_46138

theorem at_least_26_equal_differences (x : Fin 102 → ℕ) (h : ∀ i j, i < j → x i < x j) (h' : ∀ i, x i < 255) :
  (∃ d : Fin 101 → ℕ, ∃ s : Finset ℕ, s.card ≥ 26 ∧ (∀ i, d i = x i.succ - x i) ∧ ∃ i j, i ≠ j ∧ (d i = d j)) :=
by {
  sorry
}

end at_least_26_equal_differences_l46_46138


namespace count_valid_b_values_l46_46710

-- Definitions of the inequalities and the condition
def inequality1 (x : ℤ) : Prop := 3 * x > 4 * x - 4
def inequality2 (x b: ℤ) : Prop := 4 * x - b > -8

-- The main statement proving that the count of valid b values is 4
theorem count_valid_b_values (x b : ℤ) (h1 : inequality1 x) (h2 : inequality2 x b) :
  ∃ (b_values : Finset ℤ), 
    ((∀ b' ∈ b_values, ∀ x' : ℤ, inequality2 x' b' → x' ≠ 3) ∧ 
     (∀ b' ∈ b_values, 16 ≤ b' ∧ b' < 20) ∧ 
     b_values.card = 4) := by
  sorry

end count_valid_b_values_l46_46710


namespace no_solutions_xyz_l46_46826

theorem no_solutions_xyz : ∀ (x y z : ℝ), x + y = 3 → xy - z^2 = 2 → false := by
  intros x y z h1 h2
  sorry

end no_solutions_xyz_l46_46826


namespace average_age_l46_46195

def proportion (x y z : ℕ) : Prop :=  y / x = 3 ∧ z / x = 4

theorem average_age (A B C : ℕ) 
    (h1 : proportion 2 6 8)
    (h2 : A = 15)
    (h3 : B = 45)
    (h4 : C = 60) :
    (A + B + C) / 3 = 40 := 
    by
    sorry

end average_age_l46_46195


namespace charlotte_overall_score_l46_46905

theorem charlotte_overall_score :
  (0.60 * 15 + 0.75 * 20 + 0.85 * 25).round / 60 = 0.75 :=
by
  sorry

end charlotte_overall_score_l46_46905


namespace green_marbles_l46_46985

theorem green_marbles :
  ∀ (total: ℕ) (blue: ℕ) (red: ℕ) (yellow: ℕ), 
  total = 164 →
  blue = total / 2 →
  red = total / 4 →
  yellow = 14 →
  (total - (blue + red + yellow)) = 27 :=
by
  intros total blue red yellow h_total h_blue h_red h_yellow
  sorry

end green_marbles_l46_46985


namespace width_of_rectangle_l46_46166

theorem width_of_rectangle (w l : ℝ) (h1 : l = 2 * w) (h2 : l * w = 1) : w = Real.sqrt 2 / 2 :=
sorry

end width_of_rectangle_l46_46166


namespace max_z_value_l46_46948

theorem max_z_value (x y z : ℝ) (h1 : x + y + z = 0) (h2 : x * y + y * z + z * x = -3) : z ≤ 2 := sorry

end max_z_value_l46_46948


namespace sum_possible_values_q_l46_46752

/-- If natural numbers k, l, p, and q satisfy the given conditions,
the sum of all possible values of q is 4 --/
theorem sum_possible_values_q (k l p q : ℕ) 
    (h1 : ∀ a b : ℝ, a ≠ b → a * b = l → a + b = k → (∃ (c d : ℝ), c + d = (k * (l + 1)) / l ∧ c * d = (l + 2 + 1 / l))) 
    (h2 : a + 1 / b ≠ b + 1 / a)
    : q = 4 :=
sorry

end sum_possible_values_q_l46_46752


namespace Carla_servings_l46_46978

-- Define the volumes involved
def volume_watermelon : ℕ := 500
def volume_cream : ℕ := 100
def volume_per_serving : ℕ := 150

-- The total volume is the sum of the watermelon and cream volumes
def total_volume : ℕ := volume_watermelon + volume_cream

-- The number of servings is the total volume divided by the volume per serving
def n_servings : ℕ := total_volume / volume_per_serving

-- The theorem to prove that Carla can make 4 servings of smoothies
theorem Carla_servings : n_servings = 4 := by
  sorry

end Carla_servings_l46_46978


namespace total_votes_l46_46825

theorem total_votes (V : ℝ) (h : 0.60 * V - 0.40 * V = 1200) : V = 6000 :=
sorry

end total_votes_l46_46825


namespace second_quadratic_roots_complex_iff_first_roots_real_distinct_l46_46911

theorem second_quadratic_roots_complex_iff_first_roots_real_distinct (q : ℝ) :
  q < 1 → (∀ x : ℂ, (3 - q) * x^2 + 2 * (1 + q) * x + (q^2 - q + 2) ≠ 0) :=
by
  -- Placeholder for the proof
  sorry

end second_quadratic_roots_complex_iff_first_roots_real_distinct_l46_46911


namespace price_of_each_apple_l46_46818

theorem price_of_each_apple
  (bike_cost: ℝ) (repair_cost_percent: ℝ) (remaining_percentage: ℝ)
  (total_apples_sold: ℕ) (repair_cost: ℝ) (total_money_earned: ℝ)
  (price_per_apple: ℝ) :
  bike_cost = 80 →
  repair_cost_percent = 0.25 →
  remaining_percentage = 0.2 →
  total_apples_sold = 20 →
  repair_cost = repair_cost_percent * bike_cost →
  total_money_earned = repair_cost / (1 - remaining_percentage) →
  price_per_apple = total_money_earned / total_apples_sold →
  price_per_apple = 1.25 := 
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end price_of_each_apple_l46_46818


namespace Jose_age_correct_l46_46357

variable (Jose Zack Inez : ℕ)

-- Define the conditions
axiom Inez_age : Inez = 15
axiom Zack_age : Zack = Inez + 3
axiom Jose_age : Jose = Zack - 4

-- The proof statement
theorem Jose_age_correct : Jose = 14 :=
by
  -- Proof will be filled in later
  sorry

end Jose_age_correct_l46_46357


namespace percent_both_correct_l46_46093

-- Definitions of the given percentages
def A : ℝ := 75
def B : ℝ := 25
def N : ℝ := 20

-- The proof problem statement
theorem percent_both_correct (A B N : ℝ) (hA : A = 75) (hB : B = 25) (hN : N = 20) : A + B - N - 100 = 20 :=
by
  sorry

end percent_both_correct_l46_46093


namespace smallest_y_value_in_set_l46_46573

theorem smallest_y_value_in_set : ∀ y : ℕ, (0 < y) ∧ (y + 4 ≤ 8) → y = 4 :=
by
  intros y h
  have h1 : y + 4 ≤ 8 := h.2
  have h2 : 0 < y := h.1
  sorry

end smallest_y_value_in_set_l46_46573


namespace ab_bc_ca_leq_zero_l46_46044

theorem ab_bc_ca_leq_zero (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end ab_bc_ca_leq_zero_l46_46044


namespace parallel_lines_distance_sum_l46_46835

theorem parallel_lines_distance_sum (b c : ℝ) 
  (h1 : ∃ k : ℝ, 6 = 3 * k ∧ b = 4 * k) 
  (h2 : (abs ((c / 2) - 5) / (Real.sqrt (3^2 + 4^2))) = 3) : 
  b + c = 48 ∨ b + c = -12 := by
  sorry

end parallel_lines_distance_sum_l46_46835


namespace find_abc_sum_l46_46300

theorem find_abc_sum (a b c : ℤ) (h1 : a - 2 * b = 4) (h2 : a * b + c^2 - 1 = 0) :
  a + b + c = 5 ∨ a + b + c = 3 ∨ a + b + c = -1 ∨ a + b + c = -3 :=
  sorry

end find_abc_sum_l46_46300


namespace sea_horses_count_l46_46031

theorem sea_horses_count (S P : ℕ) (h1 : 11 * S = 5 * P) (h2 : P = S + 85) : S = 70 :=
by
  sorry

end sea_horses_count_l46_46031


namespace max_f_alpha_side_a_l46_46992

noncomputable def a_vec (α : ℝ) : ℝ × ℝ := (Real.sin α, Real.cos α)
noncomputable def b_vec (α : ℝ) : ℝ × ℝ := (6 * Real.sin α + Real.cos α, 7 * Real.sin α - 2 * Real.cos α)

noncomputable def f (α : ℝ) : ℝ := (a_vec α).1 * (b_vec α).1 + (a_vec α).2 * (b_vec α).2

theorem max_f_alpha : ∀ α : ℝ, f α ≤ 4 * Real.sqrt 2 + 2 :=
by
sorry

theorem side_a (A : ℝ) (b c : ℝ) (h1 : f A = 6) (h2 : 1/2 * b * c * Real.sin A = 3) (h3 : b + c = 2 + 3 * Real.sqrt 2) : 
  ∃ a : ℝ, a = Real.sqrt 10 :=
by
sorry

end max_f_alpha_side_a_l46_46992


namespace bob_selling_price_per_muffin_l46_46000

variable (dozen_muffins_per_day : ℕ := 12)
variable (cost_per_muffin : ℝ := 0.75)
variable (weekly_profit : ℝ := 63)
variable (days_per_week : ℕ := 7)

theorem bob_selling_price_per_muffin : 
  let daily_cost := dozen_muffins_per_day * cost_per_muffin
  let weekly_cost := daily_cost * days_per_week
  let weekly_revenue := weekly_profit + weekly_cost
  let muffins_per_week := dozen_muffins_per_day * days_per_week
  let selling_price_per_muffin := weekly_revenue / muffins_per_week
  selling_price_per_muffin = 1.50 := 
by
  sorry

end bob_selling_price_per_muffin_l46_46000


namespace smallest_base_for_80_l46_46977

-- Define the problem in terms of inequalities
def smallest_base (n : ℕ) (d : ℕ) :=
  ∃ b : ℕ, b > 1 ∧ b <= (n^(1/d)) ∧ (n^(1/(d+1))) < (b + 1)

-- Assertion that the smallest whole number b such that 80 can be expressed in base b using only three digits
theorem smallest_base_for_80 : ∃ b, smallest_base 80 3 ∧ b = 5 :=
  sorry

end smallest_base_for_80_l46_46977


namespace paul_sandwiches_in_6_days_l46_46696

def sandwiches_eaten_in_n_days (n : ℕ) : ℕ :=
  let day1 := 2
  let day2 := 2 * day1
  let day3 := 2 * day2
  let three_day_total := day1 + day2 + day3
  three_day_total * (n / 3)

theorem paul_sandwiches_in_6_days : sandwiches_eaten_in_n_days 6 = 28 :=
by
  sorry

end paul_sandwiches_in_6_days_l46_46696


namespace salary_recovery_l46_46288

theorem salary_recovery (S : ℝ) : 
  (0.80 * S) + (0.25 * (0.80 * S)) = S :=
by
  sorry

end salary_recovery_l46_46288


namespace shaded_areas_I_and_III_equal_l46_46712

def area_shaded_square_I : ℚ := 1 / 4
def area_shaded_square_II : ℚ := 1 / 2
def area_shaded_square_III : ℚ := 1 / 4

theorem shaded_areas_I_and_III_equal :
  area_shaded_square_I = area_shaded_square_III ∧
   area_shaded_square_I ≠ area_shaded_square_II ∧
   area_shaded_square_III ≠ area_shaded_square_II :=
by {
  sorry
}

end shaded_areas_I_and_III_equal_l46_46712


namespace fraction_sum_l46_46023

variable {a b : ℝ}

theorem fraction_sum (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) (h1 : a^2 + a - 2007 = 0) (h2 : b^2 + b - 2007 = 0) :
  (1/a + 1/b) = 1/2007 :=
by
  sorry

end fraction_sum_l46_46023


namespace manuscript_age_in_decimal_l46_46939

-- Given conditions
def octal_number : ℕ := 12345

-- Translate the problem statement into Lean:
theorem manuscript_age_in_decimal : (1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0) = 5349 :=
by
  sorry

end manuscript_age_in_decimal_l46_46939


namespace items_sold_increase_by_20_percent_l46_46472

-- Assume initial variables P (price per item without discount) and N (number of items sold without discount)
variables (P N : ℝ)

-- Define the conditions and the final proof goal
theorem items_sold_increase_by_20_percent 
  (h1 : ∀ (P N : ℝ), P > 0 → N > 0 → (P * N > 0))
  (h2 : ∀ (P : ℝ), P' = P * 0.90)
  (h3 : ∀ (P' N' : ℝ), P' * N' = P * N * 1.08)
  : (N' - N) / N * 100 = 20 := 
sorry

end items_sold_increase_by_20_percent_l46_46472


namespace no_solution_inequality_C_l46_46001

theorem no_solution_inequality_C : ¬∃ x : ℝ, 2 * x - x^2 > 5 := by
  -- There is no need to include the other options in the Lean theorem, as the proof focuses on the condition C directly.
  sorry

end no_solution_inequality_C_l46_46001


namespace tiles_with_no_gaps_l46_46235

-- Define the condition that the tiling consists of regular octagons
def regular_octagon_internal_angle := 135

-- Define the other regular polygons
def regular_triangle_internal_angle := 60
def regular_square_internal_angle := 90
def regular_pentagon_internal_angle := 108
def regular_hexagon_internal_angle := 120

-- The proposition to be proved: A flat surface without gaps
-- can be achieved using regular squares and regular octagons.
theorem tiles_with_no_gaps :
  ∃ (m n : ℕ), regular_octagon_internal_angle * m + regular_square_internal_angle * n = 360 :=
sorry

end tiles_with_no_gaps_l46_46235


namespace inequality_system_solution_l46_46564

theorem inequality_system_solution (x : ℝ) : 
  (6 * x + 1 ≤ 4 * (x - 1)) ∧ (1 - x / 4 > (x + 5) / 2) → x ≤ -5/2 :=
by
  sorry

end inequality_system_solution_l46_46564


namespace shauna_lowest_score_l46_46441

theorem shauna_lowest_score :
  ∀ (scores : List ℕ) (score1 score2 score3 : ℕ), 
    scores = [score1, score2, score3] → 
    score1 = 82 →
    score2 = 88 →
    score3 = 93 →
    (∃ (s4 s5 : ℕ), s4 + s5 = 162 ∧ s4 ≤ 100 ∧ s5 ≤ 100) ∧
    score1 + score2 + score3 + s4 + s5 = 425 →
    min s4 s5 = 62 := 
by 
  sorry

end shauna_lowest_score_l46_46441


namespace hyperbola_k_range_l46_46840

theorem hyperbola_k_range (k : ℝ) : ((k + 2) * (6 - 2 * k) > 0) ↔ (-2 < k ∧ k < 3) := 
sorry

end hyperbola_k_range_l46_46840


namespace find_sets_A_B_l46_46854

def C : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

def S : Finset ℕ := {4, 5, 9, 14, 23, 37}

theorem find_sets_A_B :
  ∃ (A B : Finset ℕ), 
  (A ∩ B = ∅) ∧ 
  (A ∪ B = C) ∧ 
  (∀ (x y : ℕ), x ≠ y → x ∈ A → y ∈ A → x + y ∉ S) ∧ 
  (∀ (x y : ℕ), x ≠ y → x ∈ B → y ∈ B → x + y ∉ S) ∧ 
  (A = {1, 2, 5, 6, 10, 11, 14, 15, 16, 19, 20}) ∧ 
  (B = {3, 4, 7, 8, 9, 12, 13, 17, 18}) :=
by
  sorry

end find_sets_A_B_l46_46854


namespace angus_total_investment_l46_46311

variable (x T : ℝ)

theorem angus_total_investment (h1 : 0.03 * x + 0.05 * 6000 = 660) (h2 : T = x + 6000) : T = 18000 :=
by
  sorry

end angus_total_investment_l46_46311


namespace determine_a_l46_46157

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

theorem determine_a :
  {a : ℝ | 0 < a ∧ (f (a + 1) ≤ f (2 * a^2))} = {a : ℝ | 1 ≤ a ∧ a ≤ Real.sqrt 6 / 2 } :=
by
  sorry

end determine_a_l46_46157


namespace james_meat_sales_l46_46101

theorem james_meat_sales
  (beef_pounds : ℕ)
  (pork_pounds : ℕ)
  (meat_per_meal : ℝ)
  (meal_price : ℝ)
  (total_meat : ℝ)
  (number_of_meals : ℝ)
  (total_money : ℝ)
  (h1 : beef_pounds = 20)
  (h2 : pork_pounds = beef_pounds / 2)
  (h3 : meat_per_meal = 1.5)
  (h4 : meal_price = 20)
  (h5 : total_meat = beef_pounds + pork_pounds)
  (h6 : number_of_meals = total_meat / meat_per_meal)
  (h7 : total_money = number_of_meals * meal_price) :
  total_money = 400 := by
  sorry

end james_meat_sales_l46_46101


namespace impossible_odd_n_m_even_sum_l46_46053

theorem impossible_odd_n_m_even_sum (n m : ℤ) (h : (n^2 + m^2 + n*m) % 2 = 0) : ¬ (n % 2 = 1 ∧ m % 2 = 1) :=
by sorry

end impossible_odd_n_m_even_sum_l46_46053


namespace complement_intersection_l46_46489

def U : Set ℤ := {1, 2, 3, 4, 5}
def P : Set ℤ := {2, 4}
def Q : Set ℤ := {1, 3, 4, 6}
def C_U_P : Set ℤ := {x ∈ U | x ∉ P}

theorem complement_intersection :
  (C_U_P ∩ Q) = {1, 3} :=
by sorry

end complement_intersection_l46_46489


namespace dice_minimum_rolls_l46_46488

theorem dice_minimum_rolls (d1 d2 d3 d4 : ℕ) (h1 : 1 ≤ d1 ∧ d1 ≤ 6)
                           (h2 : 1 ≤ d2 ∧ d2 ≤ 6) (h3 : 1 ≤ d3 ∧ d3 ≤ 6) 
                           (h4 : 1 ≤ d4 ∧ d4 ≤ 6) :
  ∃ n, n = 43 ∧ ∀ (S : ℕ) (x : ℕ → ℕ), 
  (∀ i, 4 ≤ S ∧ S ≤ 24 ∧ x i = 4 ∧ (x i ≤ 6)) →
  (n ≤ 43) ∧ (∃ (k : ℕ), k ≥ 3) :=
sorry

end dice_minimum_rolls_l46_46488


namespace diminished_gcd_equals_100_l46_46175

theorem diminished_gcd_equals_100 : Nat.gcd 7800 360 - 20 = 100 := by
  sorry

end diminished_gcd_equals_100_l46_46175


namespace complex_expression_calculation_l46_46444

noncomputable def complex_i := Complex.I -- Define the imaginary unit i

theorem complex_expression_calculation : complex_i * (1 - complex_i)^2 = 2 := by
  sorry

end complex_expression_calculation_l46_46444


namespace min_value_of_parabola_in_interval_l46_46351

theorem min_value_of_parabola_in_interval :
  ∀ x : ℝ, -10 ≤ x ∧ x ≤ 0 → (x^2 + 12 * x + 35) ≥ -1 := by
  sorry

end min_value_of_parabola_in_interval_l46_46351


namespace tangent_line_parallel_l46_46550

theorem tangent_line_parallel (x y : ℝ) (h_parab : y = 2 * x^2) (h_parallel : ∃ (m b : ℝ), 4 * x - y + b = 0) : 
    (∃ b, 4 * x - y - b = 0) := 
by
  sorry

end tangent_line_parallel_l46_46550


namespace rectangle_width_l46_46294

theorem rectangle_width (w l A : ℕ) 
  (h1 : l = 3 * w)
  (h2 : A = l * w)
  (h3 : A = 108) : 
  w = 6 := 
sorry

end rectangle_width_l46_46294


namespace second_puppy_weight_l46_46544

variables (p1 p2 c1 c2 : ℝ)

-- Conditions from the problem statement
axiom h1 : p1 + p2 + c1 + c2 = 36
axiom h2 : p1 + c2 = 3 * c1
axiom h3 : p1 + c1 = c2
axiom h4 : p2 = 1.5 * p1

-- The question to prove: how much does the second puppy weigh
theorem second_puppy_weight : p2 = 108 / 11 :=
by sorry

end second_puppy_weight_l46_46544


namespace first_discount_percentage_l46_46908

theorem first_discount_percentage 
  (original_price final_price : ℝ) 
  (successive_discount1 successive_discount2 : ℝ) 
  (h1 : original_price = 10000)
  (h2 : final_price = 6840)
  (h3 : successive_discount1 = 0.10)
  (h4 : successive_discount2 = 0.05)
  : ∃ x, (1 - x / 100) * (1 - successive_discount1) * (1 - successive_discount2) * original_price = final_price ∧ x = 20 :=
by
  sorry

end first_discount_percentage_l46_46908


namespace price_of_necklace_l46_46916

-- Define the necessary conditions.
def num_charms_per_necklace : ℕ := 10
def cost_per_charm : ℕ := 15
def num_necklaces_sold : ℕ := 30
def total_profit : ℕ := 1500

-- Calculation of selling price per necklace
def cost_per_necklace := num_charms_per_necklace * cost_per_charm
def total_cost := cost_per_necklace * num_necklaces_sold
def total_revenue := total_cost + total_profit
def selling_price_per_necklace := total_revenue / num_necklaces_sold

-- Statement of the problem in Lean 4
theorem price_of_necklace : selling_price_per_necklace = 200 := by
  sorry

end price_of_necklace_l46_46916


namespace samson_fuel_calculation_l46_46964

def total_fuel_needed (main_distance : ℕ) (fuel_rate : ℕ) (hilly_distance : ℕ) (hilly_increase : ℚ)
                      (detours : ℕ) (detour_distance : ℕ) : ℚ :=
  let normal_distance := main_distance - hilly_distance
  let normal_fuel := (fuel_rate / 70) * normal_distance
  let hilly_fuel := (fuel_rate / 70) * hilly_distance * hilly_increase
  let detour_fuel := (fuel_rate / 70) * (detours * detour_distance)
  normal_fuel + hilly_fuel + detour_fuel

theorem samson_fuel_calculation :
  total_fuel_needed 140 10 30 1.2 2 5 = 22.28 :=
by sorry

end samson_fuel_calculation_l46_46964


namespace general_term_formula_l46_46670

theorem general_term_formula (f : ℕ → ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ) :
  (∀ x, f x = 1 - 2^x) →
  (∀ n, f n = S n) →
  (∀ n, S n = 1 - 2^n) →
  (∀ n, n = 1 → a n = S 1) →
  (∀ n, n ≥ 2 → a n = S n - S (n-1)) →
  (∀ n, a n = -2^(n-1)) :=
by
  sorry

end general_term_formula_l46_46670


namespace gcd_g_values_l46_46292

def g (x : ℤ) : ℤ := x^2 - 2 * x + 2023

theorem gcd_g_values : gcd (g 102) (g 103) = 1 := by
  sorry

end gcd_g_values_l46_46292


namespace product_of_intersection_points_l46_46600

-- Define the two circles in the plane
def circle1 (x y : ℝ) : Prop := x^2 - 4*x + y^2 - 8*y + 16 = 0
def circle2 (x y : ℝ) : Prop := x^2 - 6*x + y^2 - 8*y + 21 = 0

-- Define the intersection points property
def are_intersection_points (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y

-- The theorem to be proved
theorem product_of_intersection_points : ∃ x y : ℝ, are_intersection_points x y ∧ x * y = 12 := 
by
  sorry

end product_of_intersection_points_l46_46600


namespace find_last_number_of_consecutive_even_numbers_l46_46069

theorem find_last_number_of_consecutive_even_numbers (x : ℕ) (h : 8 * x + 2 + 4 + 6 + 8 + 10 + 12 + 14 = 424) : x + 14 = 60 :=
sorry

end find_last_number_of_consecutive_even_numbers_l46_46069


namespace coefficient_of_x_in_first_equation_is_one_l46_46918

theorem coefficient_of_x_in_first_equation_is_one
  (x y z : ℝ)
  (h1 : x - 5 * y + 3 * z = 22 / 6)
  (h2 : 4 * x + 8 * y - 11 * z = 7)
  (h3 : 5 * x - 6 * y + 2 * z = 12)
  (h4 : x + y + z = 10) :
  (1 : ℝ) = 1 := 
by 
  sorry

end coefficient_of_x_in_first_equation_is_one_l46_46918


namespace rhombus_diagonal_l46_46655

theorem rhombus_diagonal (a b : ℝ) (area_triangle : ℝ) (d1 d2 : ℝ)
  (h1 : 2 * area_triangle = a * b)
  (h2 : area_triangle = 75)
  (h3 : a = 20) :
  b = 15 :=
by
  sorry

end rhombus_diagonal_l46_46655


namespace largest_prime_factor_3136_l46_46033

theorem largest_prime_factor_3136 : ∀ (n : ℕ), n = 3136 → ∃ p : ℕ, Prime p ∧ (p ∣ n) ∧ ∀ q : ℕ, (Prime q ∧ q ∣ n) → p ≥ q :=
by {
  sorry
}

end largest_prime_factor_3136_l46_46033


namespace polar_eq_circle_l46_46857

-- Definition of the problem condition in polar coordinates
def polar_eq (ρ : ℝ) : Prop := ρ = 1

-- Definition of the assertion we want to prove: that it represents a circle
def represents_circle (ρ : ℝ) (θ : ℝ) : Prop := (ρ = 1) → ∃ (x y : ℝ), (ρ = 1) ∧ (x^2 + y^2 = 1)

theorem polar_eq_circle : ∀ (ρ θ : ℝ), polar_eq ρ → represents_circle ρ θ :=
by
  intros ρ θ hρ hs
  sorry

end polar_eq_circle_l46_46857


namespace Mary_current_age_l46_46113

theorem Mary_current_age
  (M J : ℕ) 
  (h1 : J - 5 = (M - 5) + 7) 
  (h2 : J + 5 = 2 * (M + 5)) : 
  M = 2 :=
by
  /- We need to show that the current age of Mary (M) is 2
     given the conditions h1 and h2.-/
  sorry

end Mary_current_age_l46_46113


namespace domain_of_f_l46_46075

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x) + Real.sqrt (x * (x + 1))

theorem domain_of_f :
  {x : ℝ | -x ≥ 0 ∧ x * (x + 1) ≥ 0} = {x : ℝ | x ≤ -1 ∨ x = 0} :=
by
  sorry

end domain_of_f_l46_46075


namespace convert_speed_l46_46582

theorem convert_speed (v_kmph : ℝ) (conversion_factor : ℝ) : 
  v_kmph = 252 → conversion_factor = 0.277778 → v_kmph * conversion_factor = 70 := by
  intros h1 h2
  rw [h1, h2]
  sorry

end convert_speed_l46_46582


namespace normal_intersects_at_l46_46667

def parabola (x : ℝ) : ℝ := x^2

def slope_of_tangent (x : ℝ) : ℝ := 2 * x

-- C = (2, 4) is a point on the parabola
def C : ℝ × ℝ := (2, parabola 2)

-- Normal to the parabola at C intersects again at point D
-- Prove that D = (-9/4, 81/16)
theorem normal_intersects_at (D : ℝ × ℝ) :
  D = (-9/4, 81/16) :=
sorry

end normal_intersects_at_l46_46667


namespace z_in_second_quadrant_l46_46404

def is_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem z_in_second_quadrant (z : ℂ) (i : ℂ) (hi : i^2 = -1) (h : z * (1 + i^3) = i) : 
  is_second_quadrant z := by
  sorry

end z_in_second_quadrant_l46_46404


namespace find_a_value_l46_46927

theorem find_a_value (a x y : ℝ) :
  (|y| + |y - x| ≤ a - |x - 1| ∧ (y - 4) * (y + 3) ≥ (4 - x) * (3 + x)) → a = 7 :=
by
  sorry

end find_a_value_l46_46927


namespace sequence_values_induction_proof_l46_46152

def seq (a : ℕ → ℤ) := a 1 = 3 ∧ ∀ n : ℕ, a (n + 1) = a n ^ 2 - 2 * n * a n + 2

theorem sequence_values (a : ℕ → ℤ) (h : seq a) :
  a 2 = 5 ∧ a 3 = 7 ∧ a 4 = 9 :=
sorry

theorem induction_proof (a : ℕ → ℤ) (h : seq a) :
  ∀ n : ℕ, a n = 2 * n + 1 :=
sorry

end sequence_values_induction_proof_l46_46152


namespace greatest_four_digit_divisible_by_6_l46_46795

-- Define a variable to represent a four-digit number
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define a variable to represent divisibility by 3
def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Define a variable to represent divisibility by 6
def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

-- State the theorem to prove that 9996 is the greatest four-digit number divisible by 6
theorem greatest_four_digit_divisible_by_6 : 
  (∀ n : ℕ, is_four_digit_number n → divisible_by_6 n → n ≤ 9996) ∧ (is_four_digit_number 9996 ∧ divisible_by_6 9996) :=
by
  -- Insert the proof here
  sorry

end greatest_four_digit_divisible_by_6_l46_46795


namespace find_sum_l46_46934

theorem find_sum (a b : ℝ) 
  (h₁ : (a + Real.sqrt b) + (a - Real.sqrt b) = -8) 
  (h₂ : (a + Real.sqrt b) * (a - Real.sqrt b) = 4) : 
  a + b = 8 := 
sorry

end find_sum_l46_46934


namespace DianasInitialSpeed_l46_46917

open Nat

theorem DianasInitialSpeed
  (total_distance : ℕ)
  (initial_time : ℕ)
  (tired_speed : ℕ)
  (total_time : ℕ)
  (distance_when_tired : ℕ)
  (initial_distance : ℕ)
  (initial_speed : ℕ)
  (initial_hours : ℕ) :
  total_distance = 10 →
  initial_time = 2 →
  tired_speed = 1 →
  total_time = 6 →
  distance_when_tired = tired_speed * (total_time - initial_time) →
  initial_distance = total_distance - distance_when_tired →
  initial_distance = initial_speed * initial_time →
  initial_speed = 3 := by
  sorry

end DianasInitialSpeed_l46_46917


namespace inequalities_not_hold_l46_46119

theorem inequalities_not_hold (x y z a b c : ℝ) (h1 : x < a) (h2 : y < b) (h3 : z < c) : 
  ¬ (x * y + y * z + z * x < a * b + b * c + c * a) ∧ 
  ¬ (x^2 + y^2 + z^2 < a^2 + b^2 + c^2) ∧ 
  ¬ (x * y * z < a * b * c) := 
sorry

end inequalities_not_hold_l46_46119


namespace ratio_sum_div_c_l46_46559

theorem ratio_sum_div_c (a b c : ℚ) (h : a / 3 = b / 4 ∧ b / 4 = c / 5) : (a + b + c) / c = 12 / 5 :=
by
  sorry

end ratio_sum_div_c_l46_46559


namespace parallel_perpendicular_trans_l46_46237

variables {Plane Line : Type}

-- Definitions in terms of lines and planes
variables (α β γ : Plane) (a b : Line)

-- Definitions of parallel and perpendicular
def parallel (l1 l2 : Line) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry

-- The mathematical statement to prove
theorem parallel_perpendicular_trans :
  (parallel a b) → (perpendicular b α) → (perpendicular a α) :=
by sorry

end parallel_perpendicular_trans_l46_46237


namespace find_x_l46_46227

-- Define the mean of three numbers
def mean_three (a b c : ℕ) : ℚ := (a + b + c) / 3

-- Define the mean of two numbers
def mean_two (x y : ℕ) : ℚ := (x + y) / 2

-- Main theorem: value of x that satisfies the given condition
theorem find_x : 
  (mean_three 6 9 18) = (mean_two x 15) → x = 7 :=
by
  sorry

end find_x_l46_46227


namespace max_difference_y_coords_l46_46116

noncomputable def maximumDifference : ℝ :=
  (4 * Real.sqrt 6) / 9

theorem max_difference_y_coords :
  let f1 (x : ℝ) := 3 - 2 * x^2 + x^3
  let f2 (x : ℝ) := 1 + x^2 + x^3
  let x1 := Real.sqrt (2/3)
  let x2 := - Real.sqrt (2/3)
  let y1 := f1 x1
  let y2 := f1 x2
  |y1 - y2| = maximumDifference := sorry

end max_difference_y_coords_l46_46116


namespace fraction_to_decimal_l46_46625

theorem fraction_to_decimal : (7 : ℝ) / 250 = 0.028 := 
sorry

end fraction_to_decimal_l46_46625


namespace robbers_can_divide_loot_equally_l46_46699

theorem robbers_can_divide_loot_equally (coins : List ℕ) (h1 : (coins.sum % 2 = 0)) 
    (h2 : ∀ k, (k % 2 = 1 ∧ 1 ≤ k ∧ k ≤ 2017) → k ∈ coins) :
  ∃ (subset1 subset2 : List ℕ), subset1 ∪ subset2 = coins ∧ subset1.sum = subset2.sum :=
by
  sorry

end robbers_can_divide_loot_equally_l46_46699


namespace pick_three_different_cards_in_order_l46_46967

theorem pick_three_different_cards_in_order :
  (52 * 51 * 50) = 132600 :=
by
  sorry

end pick_three_different_cards_in_order_l46_46967


namespace two_common_points_with_x_axis_l46_46960

noncomputable def func (x d : ℝ) : ℝ := x^3 - 3 * x + d

theorem two_common_points_with_x_axis (d : ℝ) :
(∃ x1 x2 : ℝ, x1 ≠ x2 ∧ func x1 d = 0 ∧ func x2 d = 0) ↔ (d = 2 ∨ d = -2) :=
by
  sorry

end two_common_points_with_x_axis_l46_46960


namespace base_conversion_problem_l46_46272

def base_to_dec (base : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (λ x acc => x + base * acc) 0

theorem base_conversion_problem : 
  (base_to_dec 8 [2, 5, 3] : ℝ) / (base_to_dec 4 [1, 3] : ℝ) + 
  (base_to_dec 5 [1, 3, 2] : ℝ) / (base_to_dec 3 [2, 3] : ℝ) = 28.67 := by
  sorry

end base_conversion_problem_l46_46272


namespace area_between_curves_eq_nine_l46_46858

def f (x : ℝ) := 2 * x - x^2 + 3
def g (x : ℝ) := x^2 - 4 * x + 3

theorem area_between_curves_eq_nine :
  ∫ x in (0 : ℝ)..(3 : ℝ), (f x - g x) = 9 := by
  sorry

end area_between_curves_eq_nine_l46_46858


namespace cody_candy_total_l46_46159

theorem cody_candy_total
  (C_c : ℕ) (C_m : ℕ) (P_b : ℕ)
  (h1 : C_c = 7) (h2 : C_m = 3) (h3 : P_b = 8) :
  (C_c + C_m) * P_b = 80 :=
by
  sorry

end cody_candy_total_l46_46159


namespace problem_l46_46704

theorem problem (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 10) : a^2 + b^2 = 29 :=
by
  sorry

end problem_l46_46704


namespace max_matches_l46_46015

theorem max_matches (x y z m : ℕ) (h1 : x + y + z = 19) (h2 : x * y + y * z + x * z = m) : m ≤ 120 :=
sorry

end max_matches_l46_46015


namespace tank_capacity_l46_46124

theorem tank_capacity (fill_rate drain_rate1 drain_rate2 : ℝ)
  (initial_fullness : ℝ) (time_to_fill : ℝ) (capacity_in_liters : ℝ) :
  fill_rate = 1 / 2 ∧
  drain_rate1 = 1 / 4 ∧
  drain_rate2 = 1 / 6 ∧ 
  initial_fullness = 1 / 2 ∧ 
  time_to_fill = 60 →
  capacity_in_liters = 10000 :=
by {
  sorry
}

end tank_capacity_l46_46124


namespace mod_equiv_1_l46_46679

theorem mod_equiv_1 : (179 * 933 / 7) % 50 = 1 := by
  sorry

end mod_equiv_1_l46_46679


namespace inequality_holds_l46_46506

theorem inequality_holds (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (a * b + b * c + c * a)^2 ≥ 3 * a * b * c * (a + b + c) :=
sorry

end inequality_holds_l46_46506


namespace find_x_converges_to_l46_46434

noncomputable def series_sum (x : ℝ) : ℝ := ∑' n : ℕ, (4 * (n + 1) - 2) * x^n

theorem find_x_converges_to (x : ℝ) (h : |x| < 1) :
  series_sum x = 60 → x = 29 / 30 :=
by
  sorry

end find_x_converges_to_l46_46434


namespace coordinates_of_P_respect_to_symmetric_y_axis_l46_46521

-- Definition of points in the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

def symmetric_x_axis (p : Point) : Point :=
  { p with y := -p.y }

def symmetric_y_axis (p : Point) : Point :=
  { p with x := -p.x }

-- The given condition
def P_with_respect_to_symmetric_x_axis := Point.mk (-1) 2

-- The problem statement
theorem coordinates_of_P_respect_to_symmetric_y_axis :
    symmetric_y_axis (symmetric_x_axis P_with_respect_to_symmetric_x_axis) = Point.mk 1 (-2) :=
by
  sorry

end coordinates_of_P_respect_to_symmetric_y_axis_l46_46521


namespace negation_of_universal_proposition_l46_46170

def int_divisible_by_5 (n : ℤ) := ∃ k : ℤ, n = 5 * k
def int_odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

theorem negation_of_universal_proposition :
  (¬ ∀ n : ℤ, int_divisible_by_5 n → int_odd n) ↔ (∃ n : ℤ, int_divisible_by_5 n ∧ ¬ int_odd n) :=
by
  sorry

end negation_of_universal_proposition_l46_46170


namespace second_discount_percentage_l46_46026

-- Definitions for the given conditions
def original_price : ℝ := 33.78
def first_discount_rate : ℝ := 0.25
def final_price : ℝ := 19.0

-- Intermediate calculations based on the conditions
def first_discount : ℝ := first_discount_rate * original_price
def price_after_first_discount : ℝ := original_price - first_discount
def second_discount_amount : ℝ := price_after_first_discount - final_price

-- Lean theorem statement
theorem second_discount_percentage : (second_discount_amount / price_after_first_discount) * 100 = 25 := by
  sorry

end second_discount_percentage_l46_46026


namespace pen_price_ratio_l46_46935

theorem pen_price_ratio (x y : ℕ) (b g : ℝ) (T : ℝ) 
  (h1 : (x + y) * g = 4 * T) 
  (h2 : (x + y) * b = (1 / 2) * T) 
  (hT : T = x * b + y * g) : 
  g = 8 * b := 
sorry

end pen_price_ratio_l46_46935


namespace inequality_proof_l46_46126

variable (x y : ℝ)
variable (hx : 0 < x) (hy : 0 < y)

theorem inequality_proof :
  x / Real.sqrt y + y / Real.sqrt x ≥ Real.sqrt x + Real.sqrt y :=
sorry

end inequality_proof_l46_46126


namespace vasya_numbers_l46_46487

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 := 
by 
  sorry

end vasya_numbers_l46_46487


namespace proper_divisors_condition_l46_46005

theorem proper_divisors_condition (N : ℕ) :
  ∀ x : ℕ, (x ∣ N ∧ x ≠ 1 ∧ x ≠ N) → 
  (∀ L : ℕ, (L ∣ N ∧ L ≠ 1 ∧ L ≠ N) → (L = x^3 + 3 ∨ L = x^3 - 3)) → 
  (N = 10 ∨ N = 22) :=
by
  sorry

end proper_divisors_condition_l46_46005


namespace triangle_area_l46_46190

-- Define the vertices of the triangle
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (6, 1)
def C : ℝ × ℝ := (3, -4)

-- State that the area of the triangle is 12.5 square units
theorem triangle_area :
  let base := 6 - 1
  let height := 1 - -4
  (1 / 2) * base * height = 12.5 := by
  sorry

end triangle_area_l46_46190


namespace value_of_c_l46_46118

-- Define the function f(x)
def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem value_of_c (a b c m : ℝ) (h₀ : ∀ x : ℝ, 0 ≤ f x a b)
  (h₁ : ∀ x : ℝ, f x a b < c ↔ m < x ∧ x < m + 6) :
  c = 9 :=
sorry

end value_of_c_l46_46118


namespace smaller_angle_formed_by_hands_at_3_15_l46_46324

def degrees_per_hour : ℝ := 30
def degrees_per_minute : ℝ := 6
def hour_hand_degrees_per_minute : ℝ := 0.5

def minute_position (minute : ℕ) : ℝ :=
  minute * degrees_per_minute

def hour_position (hour : ℕ) (minute : ℕ) : ℝ :=
  hour * degrees_per_hour + minute * hour_hand_degrees_per_minute

theorem smaller_angle_formed_by_hands_at_3_15 : 
  minute_position 15 = 90 ∧ 
  hour_position 3 15 = 97.5 →
  abs (hour_position 3 15 - minute_position 15) = 7.5 :=
by
  intros h
  sorry

end smaller_angle_formed_by_hands_at_3_15_l46_46324


namespace not_possible_sum_2017_l46_46482

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem not_possible_sum_2017 (A B : ℕ) (h1 : A + B = 2017) (h2 : sum_of_digits A = 2 * sum_of_digits B) : false := 
sorry

end not_possible_sum_2017_l46_46482


namespace total_pamphlets_correct_l46_46759

def mike_initial_speed := 600
def mike_initial_hours := 9
def mike_break_hours := 2
def leo_relative_hours := 1 / 3
def leo_relative_speed := 2

def total_pamphlets (mike_initial_speed mike_initial_hours mike_break_hours leo_relative_hours leo_relative_speed : ℕ) : ℕ :=
  let mike_pamphlets_before_break := mike_initial_speed * mike_initial_hours
  let mike_speed_after_break := mike_initial_speed / 3
  let mike_pamphlets_after_break := mike_speed_after_break * mike_break_hours
  let total_mike_pamphlets := mike_pamphlets_before_break + mike_pamphlets_after_break

  let leo_hours := mike_initial_hours * leo_relative_hours
  let leo_speed := mike_initial_speed * leo_relative_speed
  let leo_pamphlets := leo_hours * leo_speed

  total_mike_pamphlets + leo_pamphlets

theorem total_pamphlets_correct : total_pamphlets 600 9 2 (1 / 3 : ℕ) 2 = 9400 := 
by 
  sorry

end total_pamphlets_correct_l46_46759


namespace large_beds_l46_46926

theorem large_beds {L : ℕ} {M : ℕ} 
    (h1 : M = 2) 
    (h2 : ∀ (x : ℕ), 100 <= x → L = (320 - 60 * M) / 100) : 
  L = 2 :=
by
  sorry

end large_beds_l46_46926


namespace remaining_area_after_cut_l46_46666

theorem remaining_area_after_cut
  (cell_side_length : ℝ)
  (grid_side_length : ℕ)
  (total_area : ℝ)
  (removed_area : ℝ)
  (hyp1 : cell_side_length = 1)
  (hyp2 : grid_side_length = 6)
  (hyp3 : total_area = (grid_side_length * grid_side_length) * cell_side_length * cell_side_length) 
  (hyp4 : removed_area = 9) :
  total_area - removed_area = 27 := by
  sorry

end remaining_area_after_cut_l46_46666


namespace gcd_4830_3289_l46_46732

theorem gcd_4830_3289 : Nat.gcd 4830 3289 = 23 :=
by sorry

end gcd_4830_3289_l46_46732


namespace large_circle_radius_l46_46295

theorem large_circle_radius (s : ℝ) (r : ℝ) (R : ℝ)
  (side_length : s = 6)
  (coverage : ∀ (x y : ℝ), (x - y)^2 + (x - y)^2 = (2 * R)^2) :
  R = 3 * Real.sqrt 2 :=
by
  sorry

end large_circle_radius_l46_46295


namespace range_of_x_l46_46865

theorem range_of_x (x m : ℝ) (h₁ : 1 ≤ m) (h₂ : m ≤ 3) (h₃ : x + 3 * m + 5 > 0) : x > -14 := 
sorry

end range_of_x_l46_46865


namespace square_of_binomial_l46_46970

theorem square_of_binomial (k : ℝ) : (∃ b : ℝ, x^2 - 20 * x + k = (x + b)^2) -> k = 100 :=
sorry

end square_of_binomial_l46_46970


namespace brad_books_this_month_l46_46673

-- Define the number of books William read last month
def william_books_last_month : ℕ := 6

-- Define the number of books Brad read last month
def brad_books_last_month : ℕ := 3 * william_books_last_month

-- Define the number of books Brad read this month as a variable
variable (B : ℕ)

-- Define the total number of books William read over the two months
def total_william_books (B : ℕ) : ℕ := william_books_last_month + 2 * B

-- Define the total number of books Brad read over the two months
def total_brad_books (B : ℕ) : ℕ := brad_books_last_month + B

-- State the condition that William read 4 more books than Brad
def william_read_more_books_condition (B : ℕ) : Prop := total_william_books B = total_brad_books B + 4

-- State the theorem to be proven
theorem brad_books_this_month (B : ℕ) : william_read_more_books_condition B → B = 16 :=
by
  sorry

end brad_books_this_month_l46_46673


namespace remainder_sum_is_74_l46_46957

-- Defining the values from the given conditions
def num1 : ℕ := 1234567
def num2 : ℕ := 890123
def divisor : ℕ := 256

-- We state the theorem to capture the main problem
theorem remainder_sum_is_74 : (num1 + num2) % divisor = 74 := 
sorry

end remainder_sum_is_74_l46_46957


namespace recurring_decimal_to_fraction_l46_46645

theorem recurring_decimal_to_fraction : (56 : ℚ) / 99 = 0.56 :=
by
  -- Problem statement and conditions are set, proof needs to be filled in
  sorry

end recurring_decimal_to_fraction_l46_46645


namespace sum_S10_equals_10_div_21_l46_46690

def a (n : ℕ) : ℚ := 1 / (4 * n^2 - 1)
def S (n : ℕ) : ℚ := (Finset.range n).sum (λ k => a (k + 1))

theorem sum_S10_equals_10_div_21 : S 10 = 10 / 21 :=
by
  sorry

end sum_S10_equals_10_div_21_l46_46690


namespace cos_alpha_plus_beta_l46_46271

theorem cos_alpha_plus_beta (α β : ℝ) (hα : Complex.exp (Complex.I * α) = 4 / 5 + Complex.I * 3 / 5)
  (hβ : Complex.exp (Complex.I * β) = -5 / 13 + Complex.I * 12 / 13) : 
  Real.cos (α + β) = -7 / 13 :=
  sorry

end cos_alpha_plus_beta_l46_46271


namespace honor_students_count_l46_46198

noncomputable def G : ℕ := 13
noncomputable def B : ℕ := 11
def E_G : ℕ := 3
def E_B : ℕ := 4

theorem honor_students_count (h1 : G + B < 30) 
    (h2 : (E_G : ℚ) / G = 3 / 13) 
    (h3 : (E_B : ℚ) / B = 4 / 11) :
    E_G + E_B = 7 := 
sorry

end honor_students_count_l46_46198


namespace max_value_y_interval_l46_46821

noncomputable def y (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem max_value_y_interval : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → y x ≤ 2) ∧ (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ y x = 2) 
:=
by
  sorry

end max_value_y_interval_l46_46821


namespace abs_x_minus_2_plus_abs_x_minus_1_lt_b_iff_b_gt_1_l46_46603

variable (x b : ℝ)

theorem abs_x_minus_2_plus_abs_x_minus_1_lt_b_iff_b_gt_1 :
  (∃ x : ℝ, |x - 2| + |x - 1| < b) ↔ b > 1 := sorry

end abs_x_minus_2_plus_abs_x_minus_1_lt_b_iff_b_gt_1_l46_46603


namespace find_A_range_sinB_sinC_l46_46285

-- Given conditions in a triangle
variable (a b c : ℝ)
variable (A B C : ℝ)
variable (h_cos_eq : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A)

-- Angle A verification
theorem find_A (h_sum_angles : A + B + C = Real.pi) : A = Real.pi / 3 :=
  sorry

-- Range of sin B + sin C
theorem range_sinB_sinC (h_sum_angles : A + B + C = Real.pi) :
  (0 < B ∧ B < 2 * Real.pi / 3) →
  Real.sin B + Real.sin C ∈ Set.Ioo (Real.sqrt 3 / 2) (Real.sqrt 3) :=
  sorry

end find_A_range_sinB_sinC_l46_46285


namespace probability_heads_is_one_eighth_l46_46563

-- Define the probability problem
def probability_heads_penny_nickel_dime (total_coins: ℕ) (successful_events: ℕ) : ℚ :=
  successful_events / total_coins

-- Define the constants
def total_outcomes : ℕ := 2^5  -- Number of possible outcomes with 5 coins
def successful_outcomes : ℕ := 4  -- Number of successful outcomes where penny, nickel, and dime are heads

-- State the theorem to be proven
theorem probability_heads_is_one_eighth : 
  probability_heads_penny_nickel_dime total_outcomes successful_outcomes = 1 / 8 :=
by
  sorry

end probability_heads_is_one_eighth_l46_46563


namespace solution_exists_l46_46986

noncomputable def find_p_q : Prop :=
  ∃ p q : ℕ, (p^q - q^p = 1927) ∧ (p = 2611) ∧ (q = 11)

theorem solution_exists : find_p_q :=
sorry

end solution_exists_l46_46986


namespace truck_travel_distance_l46_46532

def original_distance : ℝ := 300
def original_gas : ℝ := 10
def increased_efficiency_percent : ℝ := 1.10
def new_gas : ℝ := 15

theorem truck_travel_distance :
  let original_efficiency := original_distance / original_gas;
  let new_efficiency := original_efficiency * increased_efficiency_percent;
  let distance := new_gas * new_efficiency;
  distance = 495 :=
by
  sorry

end truck_travel_distance_l46_46532


namespace min_value_of_z_l46_46032

-- Define the conditions and objective function
def constraints (x y : ℝ) : Prop :=
  (y ≥ x + 2) ∧ 
  (x + y ≤ 6) ∧ 
  (x ≥ 1)

def z (x y : ℝ) : ℝ :=
  2 * |x - 2| + |y|

-- The formal theorem stating the minimum value of z under the given constraints
theorem min_value_of_z : ∃ x y : ℝ, constraints x y ∧ z x y = 4 :=
sorry

end min_value_of_z_l46_46032


namespace hyperbola_solution_l46_46321

noncomputable def hyperbola_eq (x y : ℝ) : Prop :=
  y^2 - x^2 / 3 = 1

theorem hyperbola_solution :
  ∃ x y : ℝ,
    (∃ c : ℝ, c = 2) ∧
    (∃ a : ℝ, a = 1) ∧
    (∃ n : ℝ, n = 1) ∧
    (∃ b : ℝ, b^2 = 3) ∧
    (∃ m : ℝ, m = -3) ∧
    hyperbola_eq x y := sorry

end hyperbola_solution_l46_46321


namespace min_value_of_sum_l46_46210

theorem min_value_of_sum (a b : ℝ) (h1 : Real.log a / Real.log 2 + Real.log b / Real.log 2 = 6) :
  a + b ≥ 16 :=
sorry

end min_value_of_sum_l46_46210


namespace triangle_sides_l46_46892
-- Import the entire library mainly used for geometry and algebraic proofs.

-- Define the main problem statement as a theorem.
theorem triangle_sides (a b c : ℕ) (r_incircle : ℕ)
  (r_excircle_a r_excircle_b r_excircle_c : ℕ) (s : ℕ)
  (area : ℕ) : 
  r_incircle = 1 → 
  area = s →
  r_excircle_a * r_excircle_b * r_excircle_c = (s * s * s) →
  s = (a + b + c) / 2 →
  r_excircle_a = s / (s - a) →
  r_excircle_b = s / (s - b) →
  r_excircle_c = s / (s - c) →
  a * b = 12 → 
  a = 3 ∧ b = 4 ∧ c = 5 :=
by {
  -- Placeholder for the proof.
  sorry
}

end triangle_sides_l46_46892


namespace carly_trimmed_nails_correct_l46_46068

-- Definitions based on the conditions
def total_dogs : Nat := 11
def three_legged_dogs : Nat := 3
def paws_per_four_legged_dog : Nat := 4
def paws_per_three_legged_dog : Nat := 3
def nails_per_paw : Nat := 4

-- Mathematically equivalent proof problem in Lean 4 statement
theorem carly_trimmed_nails_correct :
  let four_legged_dogs := total_dogs - three_legged_dogs
  let nails_per_four_legged_dog := paws_per_four_legged_dog * nails_per_paw
  let nails_per_three_legged_dog := paws_per_three_legged_dog * nails_per_paw
  let total_nails_trimmed :=
    (four_legged_dogs * nails_per_four_legged_dog) +
    (three_legged_dogs * nails_per_three_legged_dog)
  total_nails_trimmed = 164 := by
  sorry

end carly_trimmed_nails_correct_l46_46068


namespace number_of_long_sleeved_jerseys_l46_46674

def cost_per_long_sleeved := 15
def cost_per_striped := 10
def num_striped_jerseys := 2
def total_spent := 80

theorem number_of_long_sleeved_jerseys (x : ℕ) :
  total_spent = cost_per_long_sleeved * x + cost_per_striped * num_striped_jerseys →
  x = 4 := by
  sorry

end number_of_long_sleeved_jerseys_l46_46674


namespace average_height_correct_l46_46276

noncomputable def initially_calculated_average_height 
  (num_students : ℕ) (incorrect_height correct_height : ℕ) 
  (actual_average : ℝ) 
  (A : ℝ) : Prop :=
  let incorrect_sum := num_students * A
  let height_difference := incorrect_height - correct_height
  let actual_sum := num_students * actual_average
  incorrect_sum = actual_sum + height_difference

theorem average_height_correct 
  (num_students : ℕ) (incorrect_height correct_height : ℕ) 
  (actual_average : ℝ) :
  initially_calculated_average_height num_students incorrect_height correct_height actual_average 175 :=
by {
  sorry
}

end average_height_correct_l46_46276


namespace students_present_in_class_l46_46776

noncomputable def num_students : ℕ := 100
noncomputable def percent_boys : ℝ := 0.55
noncomputable def percent_girls : ℝ := 0.45
noncomputable def absent_boys_percent : ℝ := 0.16
noncomputable def absent_girls_percent : ℝ := 0.12

theorem students_present_in_class :
  let num_boys := percent_boys * num_students
  let num_girls := percent_girls * num_students
  let absent_boys := absent_boys_percent * num_boys
  let absent_girls := absent_girls_percent * num_girls
  let present_boys := num_boys - absent_boys
  let present_girls := num_girls - absent_girls
  present_boys + present_girls = 86 :=
by
  sorry

end students_present_in_class_l46_46776


namespace lily_jog_time_l46_46982

theorem lily_jog_time :
  (∃ (max_time : ℕ) (lily_miles_max : ℕ) (max_distance : ℕ) (lily_time_ratio : ℕ) (distance_wanted : ℕ)
      (expected_time : ℕ),
    max_time = 36 ∧
    lily_miles_max = 4 ∧
    max_distance = 6 ∧
    lily_time_ratio = 3 ∧
    distance_wanted = 7 ∧
    expected_time = 21 ∧
    lily_miles_max * lily_time_ratio = max_time ∧
    max_distance * lily_time_ratio = distance_wanted * expected_time) := 
sorry

end lily_jog_time_l46_46982


namespace two_thirds_of_5_times_9_l46_46140

theorem two_thirds_of_5_times_9 : (2 / 3) * (5 * 9) = 30 :=
by
  sorry

end two_thirds_of_5_times_9_l46_46140


namespace intersection_unique_l46_46111

noncomputable def f (x : ℝ) := 3 * Real.log x
noncomputable def g (x : ℝ) := Real.log (x + 4)

theorem intersection_unique : ∃! x, f x = g x :=
sorry

end intersection_unique_l46_46111


namespace belts_count_l46_46396

-- Definitions based on conditions
variable (shoes belts hats : ℕ)

-- Conditions from the problem
axiom shoes_eq_14 : shoes = 14
axiom hat_count : hats = 5
axiom shoes_double_of_belts : shoes = 2 * belts

-- Definition of the theorem to prove the number of belts
theorem belts_count : belts = 7 :=
by
  sorry

end belts_count_l46_46396


namespace David_total_swim_time_l46_46780

theorem David_total_swim_time :
  let t_freestyle := 48
  let t_backstroke := t_freestyle + 4
  let t_butterfly := t_backstroke + 3
  let t_breaststroke := t_butterfly + 2
  t_freestyle + t_backstroke + t_butterfly + t_breaststroke = 212 :=
by
  sorry

end David_total_swim_time_l46_46780


namespace estimate_time_pm_l46_46067

-- Definitions from the conditions
def school_start_time : ℕ := 12
def classes : List String := ["Maths", "History", "Geography", "Science", "Music"]
def class_time : ℕ := 45  -- in minutes
def break_time : ℕ := 15  -- in minutes
def classes_up_to_science : List String := ["Maths", "History", "Geography", "Science"]
def total_classes_time : ℕ := classes_up_to_science.length * (class_time + break_time)

-- Lean statement to prove that given the conditions, the time is 4 pm
theorem estimate_time_pm :
  school_start_time + (total_classes_time / 60) = 16 :=
by
  sorry

end estimate_time_pm_l46_46067


namespace rational_zero_quadratic_roots_l46_46606

-- Part 1
theorem rational_zero (a b : ℚ) (h : a + b * Real.sqrt 5 = 0) : a = 0 ∧ b = 0 :=
sorry

-- Part 2
theorem quadratic_roots (k : ℝ) (h : k ≠ 0) (x1 x2 : ℝ)
  (h1 : 4 * k * x1^2 - 4 * k * x1 + k + 1 = 0)
  (h2 : 4 * k * x2^2 - 4 * k * x2 + k + 1 = 0)
  (h3 : x1 ≠ x2) 
  (h4 : x1^2 + x2^2 - 2 * x1 * x2 = 0.5) : k = -2 :=
sorry

end rational_zero_quadratic_roots_l46_46606


namespace lcm_of_12_15_18_is_180_l46_46638

theorem lcm_of_12_15_18_is_180 :
  Nat.lcm 12 (Nat.lcm 15 18) = 180 := by
  sorry

end lcm_of_12_15_18_is_180_l46_46638


namespace sin_double_angle_l46_46373

theorem sin_double_angle
  (α : ℝ) (h1 : Real.sin (3 * Real.pi / 2 - α) = 3 / 5) (h2 : α ∈ Set.Ioo Real.pi (3 * Real.pi / 2)) :
  Real.sin (2 * α) = 24 / 25 :=
sorry

end sin_double_angle_l46_46373


namespace max_x_plus_2y_l46_46358

theorem max_x_plus_2y (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 9) 
  (h2 : 3 * x + 5 * y ≤ 15) : 
  x + 2 * y ≤ 6 :=
sorry

end max_x_plus_2y_l46_46358


namespace ziggy_song_requests_l46_46080

theorem ziggy_song_requests :
  ∃ T : ℕ, 
    (T = (1/2) * T + (1/6) * T + 5 + 2 + 1 + 2) →
    T = 30 :=
by 
  sorry

end ziggy_song_requests_l46_46080


namespace valid_four_digit_numbers_count_l46_46958

noncomputable def num_valid_four_digit_numbers : ℕ := 6 * 65 * 10

theorem valid_four_digit_numbers_count :
  num_valid_four_digit_numbers = 3900 :=
by
  -- We provide the steps in the proof to guide the automation
  let total_valid_numbers := 6 * 65 * 10
  have h1 : total_valid_numbers = 3900 := rfl
  exact h1

end valid_four_digit_numbers_count_l46_46958


namespace find_integer_pairs_l46_46527

theorem find_integer_pairs (x y : ℤ) :
  x^4 + (y+2)^3 = (x+2)^4 ↔ (x, y) = (0, 0) ∨ (x, y) = (-1, -2) := sorry

end find_integer_pairs_l46_46527


namespace count_three_digit_odd_increasing_order_l46_46282

theorem count_three_digit_odd_increasing_order : 
  ∃ n : ℕ, n = 10 ∧
  ∀ a b c : ℕ, (100 * a + 10 * b + c) % 2 = 1 ∧ a < b ∧ b < c ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 → 
    (100 * a + 10 * b + c) % 2 = 1 := 
sorry

end count_three_digit_odd_increasing_order_l46_46282


namespace Grant_made_total_l46_46717

-- Definitions based on the given conditions
def price_cards : ℕ := 25
def price_bat : ℕ := 10
def price_glove_before_discount : ℕ := 30
def glove_discount_rate : ℚ := 0.20
def price_cleats_each : ℕ := 10
def cleats_pairs : ℕ := 2

-- Calculations
def price_glove_after_discount : ℚ := price_glove_before_discount * (1 - glove_discount_rate)
def total_price_cleats : ℕ := price_cleats_each * cleats_pairs
def total_price : ℚ :=
  price_cards + price_bat + total_price_cleats + price_glove_after_discount

-- The statement we need to prove
theorem Grant_made_total :
  total_price = 79 := by sorry

end Grant_made_total_l46_46717


namespace compute_expression_l46_46267

theorem compute_expression : 45 * (28 + 72) + 55 * 45 = 6975 := 
  by
  sorry

end compute_expression_l46_46267


namespace math_city_police_officers_needed_l46_46086

def number_of_streets : Nat := 10
def initial_intersections : Nat := Nat.choose number_of_streets 2
def non_intersections : Nat := 2
def effective_intersections : Nat := initial_intersections - non_intersections

theorem math_city_police_officers_needed :
  effective_intersections = 43 := by
  sorry

end math_city_police_officers_needed_l46_46086


namespace choir_singers_joined_final_verse_l46_46273

theorem choir_singers_joined_final_verse (total_singers : ℕ) (first_verse_fraction : ℚ)
  (second_verse_fraction : ℚ) (initial_remaining : ℕ) (second_verse_joined : ℕ) : 
  total_singers = 30 → 
  first_verse_fraction = 1 / 2 → 
  second_verse_fraction = 1 / 3 → 
  initial_remaining = total_singers / 2 → 
  second_verse_joined = initial_remaining / 3 → 
  (total_singers - (initial_remaining + second_verse_joined)) = 10 := 
by
  intros
  sorry

end choir_singers_joined_final_verse_l46_46273


namespace washing_machine_capacity_l46_46790

def num_shirts : Nat := 19
def num_sweaters : Nat := 8
def num_loads : Nat := 3

theorem washing_machine_capacity :
  (num_shirts + num_sweaters) / num_loads = 9 := by
  sorry

end washing_machine_capacity_l46_46790


namespace problem_statement_l46_46316

theorem problem_statement (x : ℝ) (h : x^2 - 3 * x + 1 = 0) : x^2 + 1 / x^2 = 7 :=
sorry

end problem_statement_l46_46316


namespace range_of_y_when_x_3_l46_46262

variable (a c : ℝ)

theorem range_of_y_when_x_3 (h1 : -4 ≤ a + c ∧ a + c ≤ -1) (h2 : -1 ≤ 4 * a + c ∧ 4 * a + c ≤ 5) :
  -1 ≤ 9 * a + c ∧ 9 * a + c ≤ 20 :=
sorry

end range_of_y_when_x_3_l46_46262


namespace problem1_problem2_l46_46594

-- Proof Problem 1
theorem problem1 (x : ℝ) : -x^2 + 4 * x + 5 < 0 ↔ x < -1 ∨ x > 5 :=
by sorry

-- Proof Problem 2
theorem problem2 (x a : ℝ) :
  if a = -1 then (x^2 + (1 - a) * x - a < 0 ↔ false) else
  if a > -1 then (x^2 + (1 - a) * x - a < 0 ↔ -1 < x ∧ x < a) else
  (x^2 + (1 - a) * x - a < 0 ↔ a < x ∧ x < -1) :=
by sorry

end problem1_problem2_l46_46594


namespace find_linear_function_b_l46_46683

theorem find_linear_function_b (b : ℝ) :
  (∃ b, (∀ x y, y = 2 * x + b - 2 → (x = -1 ∧ y = 0)) → b = 4) :=
sorry

end find_linear_function_b_l46_46683


namespace osborn_friday_time_l46_46901

-- Conditions
def time_monday : ℕ := 2
def time_tuesday : ℕ := 4
def time_wednesday : ℕ := 3
def time_thursday : ℕ := 4
def old_average_time_per_day : ℕ := 3
def school_days_per_week : ℕ := 5

-- Total time needed to match old average
def total_time_needed : ℕ := old_average_time_per_day * school_days_per_week

-- Total time spent from Monday to Thursday
def time_spent_mon_to_thu : ℕ := time_monday + time_tuesday + time_wednesday + time_thursday

-- Goal: Find time on Friday
def time_friday : ℕ := total_time_needed - time_spent_mon_to_thu

theorem osborn_friday_time : time_friday = 2 :=
by
  sorry

end osborn_friday_time_l46_46901


namespace boys_contributions_l46_46216

theorem boys_contributions (x y z : ℝ) (h1 : z = x + 6.4) (h2 : (1 / 2) * x = (1 / 3) * y) (h3 : (1 / 2) * x = (1 / 4) * z) :
  x = 6.4 ∧ y = 9.6 ∧ z = 12.8 :=
by
  -- This is where the proof would go
  sorry

end boys_contributions_l46_46216


namespace no_n_ge_1_such_that_sum_is_perfect_square_l46_46617

theorem no_n_ge_1_such_that_sum_is_perfect_square :
  ¬ ∃ n : ℕ, n ≥ 1 ∧ ∃ k : ℕ, 2^n + 12^n + 2014^n = k^2 :=
by
  sorry

end no_n_ge_1_such_that_sum_is_perfect_square_l46_46617


namespace b_earns_more_than_a_l46_46824

-- Definitions for the conditions
def investments_ratio := (3, 4, 5)
def returns_ratio := (6, 5, 4)
def total_earnings := 10150

-- We need to prove the statement
theorem b_earns_more_than_a (x y : ℕ) (hx : 58 * x * y = 10150) : 2 * x * y = 350 := by
  -- Conditions based on ratios
  let earnings_a := 3 * x * 6 * y
  let earnings_b := 4 * x * 5 * y
  let difference := earnings_b - earnings_a
  
  -- To complete the proof, sorry is used
  sorry

end b_earns_more_than_a_l46_46824


namespace unit_digit_4137_pow_754_l46_46383

theorem unit_digit_4137_pow_754 : (4137 ^ 754) % 10 = 9 := by
  sorry

end unit_digit_4137_pow_754_l46_46383


namespace louie_mistakes_l46_46463

theorem louie_mistakes (total_items : ℕ) (percentage_correct : ℕ) 
  (h1 : total_items = 25) 
  (h2 : percentage_correct = 80) : 
  total_items - ((percentage_correct / 100) * total_items) = 5 := 
by
  sorry

end louie_mistakes_l46_46463


namespace minimum_valid_N_exists_l46_46287

theorem minimum_valid_N_exists (N : ℝ) (a : ℕ → ℕ) :
  (∀ n : ℕ, a n > 0) →
  (∀ n : ℕ, a n < a (n+1)) →
  (∀ n : ℕ, (a (2*n - 1) + a (2*n)) / a n = N) →
  N ≥ 4 :=
by
  sorry

end minimum_valid_N_exists_l46_46287


namespace diagonals_in_polygon_of_150_sides_l46_46723

-- Definition of the number of diagonals formula
def number_of_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- Given condition: the polygon has 150 sides
def n : ℕ := 150

-- Statement to prove
theorem diagonals_in_polygon_of_150_sides : number_of_diagonals n = 11025 :=
by
  sorry

end diagonals_in_polygon_of_150_sides_l46_46723


namespace triangle_side_length_sum_l46_46364

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def distance_squared (p1 p2 : Point3D) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

structure Triangle where
  D : Point3D
  E : Point3D
  F : Point3D

noncomputable def centroid (t : Triangle) : Point3D :=
  let D := t.D
  let E := t.E
  let F := t.F
  { x := (D.x + E.x + F.x) / 3,
    y := (D.y + E.y + F.y) / 3,
    z := (D.z + E.z + F.z) / 3 }

noncomputable def sum_of_squares_centroid_distances (t : Triangle) : ℝ :=
  let G := centroid t
  distance_squared G t.D + distance_squared G t.E + distance_squared G t.F

noncomputable def sum_of_squares_side_lengths (t : Triangle) : ℝ :=
  distance_squared t.D t.E + distance_squared t.D t.F + distance_squared t.E t.F

theorem triangle_side_length_sum (t : Triangle) (h : sum_of_squares_centroid_distances t = 72) :
  sum_of_squares_side_lengths t = 216 :=
sorry

end triangle_side_length_sum_l46_46364


namespace problem1_problem2_l46_46259

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < 0 then -2 * x - 1
  else if 0 < x ∧ x ≤ 1 then -2 * x + 1
  else 0 -- considering the function is not defined outside the given range

-- Statement to prove that f(f(-1)) = -1
theorem problem1 : f (f (-1)) = -1 :=
by
  sorry

-- Statements to prove the solution set for |f(x)| < 1/2
theorem problem2 : { x : ℝ | |f x| < 1 / 2 } = { x : ℝ | -3/4 < x ∧ x < -1/4 } ∪ { x : ℝ | 1/4 < x ∧ x < 3/4 } :=
by
  sorry

end problem1_problem2_l46_46259


namespace coplanar_vectors_set_B_l46_46756

variables {V : Type*} [AddCommGroup V] [Module ℝ V] 
variables (a b c : V)

theorem coplanar_vectors_set_B
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ (k₁ k₂ : ℝ), 
    k₁ • (2 • a + b) + k₂ • (a + b + c) = 7 • a + 5 • b + 3 • c :=
by { sorry }

end coplanar_vectors_set_B_l46_46756


namespace range_of_a_if_exists_x_l46_46392

variable {a x : ℝ}

theorem range_of_a_if_exists_x :
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ (a * x^2 - 1 ≥ 0)) → (a > 1) :=
by
  sorry

end range_of_a_if_exists_x_l46_46392


namespace sector_properties_l46_46172

noncomputable def central_angle (l r : ℝ) : ℝ := l / r

noncomputable def sector_area (alpha r : ℝ) : ℝ := (1/2) * alpha * r^2

theorem sector_properties (l r : ℝ) (h_l : l = Real.pi) (h_r : r = 3) :
  central_angle l r = Real.pi / 3 ∧ sector_area (central_angle l r) r = 3 * Real.pi / 2 := 
  by
  sorry

end sector_properties_l46_46172


namespace total_subjects_is_41_l46_46491

-- Define the number of subjects taken by Monica, Marius, and Millie
def subjects_monica := 10
def subjects_marius := subjects_monica + 4
def subjects_millie := subjects_marius + 3

-- Define the total number of subjects taken by all three
def total_subjects := subjects_monica + subjects_marius + subjects_millie

theorem total_subjects_is_41 : total_subjects = 41 := by
  -- This is where the proof would be, but we only need the statement
  sorry

end total_subjects_is_41_l46_46491


namespace maximize_angle_l46_46208

structure Point where
  x : ℝ
  y : ℝ

def A (a : ℝ) : Point := ⟨0, a⟩
def B (b : ℝ) : Point := ⟨0, b⟩

theorem maximize_angle
  (a b : ℝ)
  (h : a > b)
  (h₁ : b > 0)
  : ∃ (C : Point), C = ⟨Real.sqrt (a * b), 0⟩ :=
sorry

end maximize_angle_l46_46208


namespace sugar_per_chocolate_bar_l46_46722

-- Definitions from conditions
def total_sugar : ℕ := 177
def lollipop_sugar : ℕ := 37
def chocolate_bar_count : ℕ := 14

-- Proof problem statement
theorem sugar_per_chocolate_bar : 
  (total_sugar - lollipop_sugar) / chocolate_bar_count = 10 := 
by 
  sorry

end sugar_per_chocolate_bar_l46_46722


namespace matrix_pow_101_l46_46612

noncomputable def matrixA : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, 0, 1],
    ![1, 0, 0],
    ![0, 1, 0]
  ]

theorem matrix_pow_101 :
  matrixA ^ 101 =
  ![
    ![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]
  ] :=
sorry

end matrix_pow_101_l46_46612


namespace combined_jail_time_in_weeks_l46_46347

-- Definitions based on conditions
def days_of_protest : ℕ := 30
def number_of_cities : ℕ := 21
def daily_arrests_per_city : ℕ := 10
def days_in_jail_pre_trial : ℕ := 4
def sentence_weeks : ℕ := 2
def jail_fraction_of_sentence : ℕ := 1 / 2

-- Calculate the combined weeks of jail time
theorem combined_jail_time_in_weeks : 
  (days_of_protest * daily_arrests_per_city * number_of_cities) * 
  (days_in_jail_pre_trial + (sentence_weeks * 7 * jail_fraction_of_sentence)) / 
  7 = 9900 := 
by sorry

end combined_jail_time_in_weeks_l46_46347


namespace find_f_at_2_l46_46590

variable {R : Type} [Ring R]

def f (a b x : R) : R := a * x ^ 3 + b * x - 3

theorem find_f_at_2 (a b : R) (h : f a b (-2) = 7) : f a b 2 = -13 := 
by 
  have h₁ : f a b (-2) + f a b 2 = -6 := sorry
  have h₂ : f a b 2 = -6 - f a b (-2) := sorry
  rw [h₂, h]
  norm_num

end find_f_at_2_l46_46590


namespace ac_le_bc_if_a_gt_b_and_c_le_zero_a_sq_gt_b_sq_if_ac_sq_gt_bc_sq_and_b_ge_zero_log_a1_gt_log_b1_if_a_gt_b_and_b_gt_neg1_inv_a_lt_inv_b_if_a_gt_b_and_ab_gt_zero_l46_46454

section

variable {a b c : ℝ}

-- Statement 1
theorem ac_le_bc_if_a_gt_b_and_c_le_zero (h1 : a > b) (h2 : c ≤ 0) : a * c ≤ b * c := 
  sorry

-- Statement 2
theorem a_sq_gt_b_sq_if_ac_sq_gt_bc_sq_and_b_ge_zero (h1 : a * c ^ 2 > b * c ^ 2) (h2 : b ≥ 0) : a ^ 2 > b ^ 2 := 
  sorry

-- Statement 3
theorem log_a1_gt_log_b1_if_a_gt_b_and_b_gt_neg1 (h1 : a > b) (h2 : b > -1) : Real.log (a + 1) > Real.log (b + 1) := 
  sorry

-- Statement 4
theorem inv_a_lt_inv_b_if_a_gt_b_and_ab_gt_zero (h1 : a > b) (h2 : a * b > 0) : 1 / a < 1 / b := 
  sorry

end

end ac_le_bc_if_a_gt_b_and_c_le_zero_a_sq_gt_b_sq_if_ac_sq_gt_bc_sq_and_b_ge_zero_log_a1_gt_log_b1_if_a_gt_b_and_b_gt_neg1_inv_a_lt_inv_b_if_a_gt_b_and_ab_gt_zero_l46_46454


namespace sum_of_solutions_l46_46236

theorem sum_of_solutions (a : ℝ) (h : 0 < a ∧ a < 1) :
  let x1 := 3 + a
  let x2 := 3 - a
  let x3 := 1 + a
  let x4 := 1 - a
  x1 + x2 + x3 + x4 = 8 :=
by
  intros
  sorry

end sum_of_solutions_l46_46236


namespace binary101_to_decimal_l46_46558

theorem binary101_to_decimal :
  let binary_101 := 1 * 2^2 + 0 * 2^1 + 1 * 2^0
  binary_101 = 5 := 
by
  let binary_101 := 1 * 2^2 + 0 * 2^1 + 1 * 2^0
  show binary_101 = 5
  sorry

end binary101_to_decimal_l46_46558


namespace new_difference_greater_l46_46446

theorem new_difference_greater (x y a b : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x > y) (ha_pos : 0 < a) (hb_pos : 0 < b) (hab : a ≠ b) :
  (x + a) - (y - b) > x - y :=
by {
  sorry
}

end new_difference_greater_l46_46446


namespace log_sum_zero_l46_46480

theorem log_sum_zero (a b c N : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_N : 0 < N) (h_neq_N : N ≠ 1) (h_geom_mean : b^2 = a * c) : 
  1 / Real.logb a N - 2 / Real.logb b N + 1 / Real.logb c N = 0 :=
  by
  sorry

end log_sum_zero_l46_46480


namespace average_length_of_remaining_strings_l46_46016

theorem average_length_of_remaining_strings :
  ∀ (n_cat : ℕ) 
    (avg_len_total avg_len_one_fourth avg_len_one_third : ℝ)
    (total_length total_length_one_fourth total_length_one_third remaining_length : ℝ),
    n_cat = 12 →
    avg_len_total = 90 →
    avg_len_one_fourth = 75 →
    avg_len_one_third = 65 →
    total_length = n_cat * avg_len_total →
    total_length_one_fourth = (n_cat / 4) * avg_len_one_fourth →
    total_length_one_third = (n_cat / 3) * avg_len_one_third →
    remaining_length = total_length - (total_length_one_fourth + total_length_one_third) →
    remaining_length / (n_cat - (n_cat / 4 + n_cat / 3)) = 119 :=
by sorry

end average_length_of_remaining_strings_l46_46016


namespace arithmetic_sequence_terms_l46_46476

theorem arithmetic_sequence_terms (a : ℕ → ℝ) (n : ℕ) (S : ℝ) 
  (h1 : a 0 + a 1 + a 2 = 4) 
  (h2 : a (n-3) + a (n-2) + a (n-1) = 7) 
  (h3 : (n * (a 0 + a (n-1)) / 2) = 22) : 
  n = 12 :=
sorry

end arithmetic_sequence_terms_l46_46476


namespace product_of_a_and_b_l46_46622

variable (a b : ℕ)

-- Conditions
def LCM(a b : ℕ) : ℕ := Nat.lcm a b
def HCF(a b : ℕ) : ℕ := Nat.gcd a b

-- Assertion: product of a and b
theorem product_of_a_and_b (h_lcm: LCM a b = 72) (h_hcf: HCF a b = 6) : a * b = 432 := by
  sorry

end product_of_a_and_b_l46_46622


namespace range_of_a_l46_46607

theorem range_of_a (a : ℝ) 
  (h : ∀ (f : ℝ → ℝ), 
    (∀ x ≤ a, f x = -x^2 - 2*x) ∧ 
    (∀ x > a, f x = -x) ∧ 
    ¬ ∃ M, ∀ x, f x ≤ M) : 
  a < -1 :=
by
  sorry

end range_of_a_l46_46607


namespace circle_area_l46_46621

theorem circle_area (C : ℝ) (hC : C = 24) : ∃ (A : ℝ), A = 144 / π :=
by
  sorry

end circle_area_l46_46621


namespace solution_set_of_inequality_l46_46785

theorem solution_set_of_inequality (x : ℝ) : (x / (x - 1) < 0) ↔ (0 < x ∧ x < 1) := 
sorry

end solution_set_of_inequality_l46_46785


namespace range_of_a_value_of_a_l46_46841

-- Problem 1
theorem range_of_a (a : ℝ) :
  (∃ x, (2 < x ∧ x < 4) ∧ (a < x ∧ x < 3 * a)) ↔ (4 / 3 ≤ a ∧ a < 4) :=
sorry

-- Problem 2
theorem value_of_a (a : ℝ) :
  (∀ x, (2 < x ∧ x < 4) ∨ (a < x ∧ x < 3 * a) ↔ (2 < x ∧ x < 6)) ↔ (a = 2) :=
sorry

end range_of_a_value_of_a_l46_46841


namespace circle_symmetry_l46_46852

theorem circle_symmetry (a : ℝ) : 
  (∀ x y : ℝ, (x^2 + y^2 - a*x + 2*y + 1 = 0 ↔ x^2 + y^2 = 1) ↔ a = 2) :=
sorry

end circle_symmetry_l46_46852


namespace percent_round_trip_tickets_l46_46972

-- Define the main variables
variables (P R : ℝ)

-- Define the conditions based on the problem statement
def condition1 : Prop := 0.3 * P = 0.3 * R
 
-- State the theorem to prove
theorem percent_round_trip_tickets (h1 : condition1 P R) : R / P * 100 = 30 := by sorry

end percent_round_trip_tickets_l46_46972


namespace employee_price_l46_46380

theorem employee_price (wholesale_cost retail_markup employee_discount : ℝ) 
    (h₁ : wholesale_cost = 200) 
    (h₂ : retail_markup = 0.20) 
    (h₃ : employee_discount = 0.25) : 
    (wholesale_cost * (1 + retail_markup)) * (1 - employee_discount) = 180 := 
by
  sorry

end employee_price_l46_46380


namespace geometric_sequence_product_of_terms_l46_46003

theorem geometric_sequence_product_of_terms 
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_a2 : a 2 = 2)
  (h_a6 : a 6 = 8) : 
  a 3 * a 4 * a 5 = 64 := 
by
  sorry

end geometric_sequence_product_of_terms_l46_46003


namespace problem_part_1_problem_part_2_problem_part_3_l46_46945

open Set

-- Definitions for the given problem conditions
def U : Set ℕ := { x | x > 0 ∧ x < 10 }
def B : Set ℕ := {1, 2, 3, 4}
def C : Set ℕ := {3, 4, 5, 6}
def D : Set ℕ := B ∩ C

-- Prove each part of the problem
theorem problem_part_1 :
  U = {1, 2, 3, 4, 5, 6, 7, 8, 9} := by
  sorry

theorem problem_part_2 :
  D = {3, 4} ∧
  (∀ (s : Set ℕ), s ⊆ D ↔ s = ∅ ∨ s = {3} ∨ s = {4} ∨ s = {3, 4}) := by
  sorry

theorem problem_part_3 :
  (U \ D) = {1, 2, 5, 6, 7, 8, 9} := by
  sorry

end problem_part_1_problem_part_2_problem_part_3_l46_46945


namespace hexagon_colorings_l46_46561

-- Definitions based on conditions
def isValidColoring (A B C D E F : ℕ) (colors : Fin 7 → ℕ) : Prop :=
  -- Adjacent vertices must have different colors
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ F ∧ F ≠ A ∧
  -- Diagonal vertices must have different colors
  A ≠ D ∧ B ≠ E ∧ C ≠ F

-- Function to count all valid colorings
def countValidColorings : ℕ :=
  let colors := List.range 7
  -- Calculate total number of valid colorings
  7 * 6 * 5 * 4 * 3 * 2

theorem hexagon_colorings : countValidColorings = 5040 := by
  sorry

end hexagon_colorings_l46_46561


namespace convert_to_standard_spherical_coordinates_l46_46042

theorem convert_to_standard_spherical_coordinates :
  let ρ := 4
  let θ := (3 * Real.pi) / 4
  let φ := (9 * Real.pi) / 5
  let adjusted_φ := 2 * Real.pi - φ
  let adjusted_θ := θ + Real.pi
  (ρ, adjusted_θ, adjusted_φ) = (4, (7 * Real.pi) / 4, Real.pi / 5) :=
by
  let ρ := 4
  let θ := (3 * Real.pi) / 4
  let φ := (9 * Real.pi) / 5
  let adjusted_φ := 2 * Real.pi - φ
  let adjusted_θ := θ + Real.pi
  sorry

end convert_to_standard_spherical_coordinates_l46_46042


namespace mandy_toys_count_l46_46640

theorem mandy_toys_count (M A Am P : ℕ) 
    (h1 : A = 3 * M) 
    (h2 : A = Am - 2) 
    (h3 : A = P / 2) 
    (h4 : M + A + Am + P = 278) : 
    M = 21 := 
by
  sorry

end mandy_toys_count_l46_46640


namespace minimum_value_l46_46661

theorem minimum_value (x y z : ℝ) (h : x + 2 * y + z = 1) : x^2 + 4 * y^2 + z^2 ≥ 1 / 3 :=
sorry

end minimum_value_l46_46661


namespace problem_inequality_l46_46874

theorem problem_inequality (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
    (y + z) / (2 * x) + (z + x) / (2 * y) + (x + y) / (2 * z) ≥
    2 * x / (y + z) + 2 * y / (z + x) + 2 * z / (x + y) :=
by
  sorry

end problem_inequality_l46_46874


namespace inequality_one_inequality_two_l46_46648

variable (a b c : ℝ)

-- Conditions given in the problem
axiom positive_a : 0 < a
axiom positive_b : 0 < b
axiom positive_c : 0 < c
axiom sum_eq_one : a + b + c = 1

-- Statements to prove
theorem inequality_one : ab + bc + ac ≤ 1 / 3 :=
sorry

theorem inequality_two : a^2 / b + b^2 / c + c^2 / a ≥ 1 :=
sorry

end inequality_one_inequality_two_l46_46648


namespace colten_chickens_l46_46551

/-
Define variables to represent the number of chickens each person has.
-/

variables (C : ℕ)   -- Number of chickens Colten has.
variables (S : ℕ)   -- Number of chickens Skylar has.
variables (Q : ℕ)   -- Number of chickens Quentin has.

/-
Define the given conditions
-/
def condition1 := Q + S + C = 383
def condition2 := Q = 2 * S + 25
def condition3 := S = 3 * C - 4

theorem colten_chickens : C = 37 :=
by
  -- Proof elaboration to be done with sorry for the auto proof
  sorry

end colten_chickens_l46_46551


namespace necessary_but_not_sufficient_l46_46478

theorem necessary_but_not_sufficient (a b : ℝ) :
  (a - b > 0 → a^2 - b^2 > 0) ∧ ¬(a^2 - b^2 > 0 → a - b > 0) := by
sorry

end necessary_but_not_sufficient_l46_46478


namespace seth_sold_candy_bars_l46_46135

theorem seth_sold_candy_bars (max_sold : ℕ) (seth_sold : ℕ) 
  (h1 : max_sold = 24) 
  (h2 : seth_sold = 3 * max_sold + 6) : 
  seth_sold = 78 := 
by sorry

end seth_sold_candy_bars_l46_46135


namespace second_pipe_fill_time_l46_46583

theorem second_pipe_fill_time (x : ℝ) :
  let rate1 := 1 / 8
  let rate2 := 1 / x
  let combined_rate := 1 / 4.8
  rate1 + rate2 = combined_rate → x = 12 :=
by
  intros
  sorry

end second_pipe_fill_time_l46_46583


namespace complex_exp1990_sum_theorem_l46_46002

noncomputable def complex_exp1990_sum (x y : Complex) (h : x ≠ 0 ∧ y ≠ 0 ∧ x^2 + x*y + y^2 = 0) : Prop :=
  (x / (x + y))^1990 + (y / (x + y))^1990 = -1

theorem complex_exp1990_sum_theorem (x y : Complex) (h : x ≠ 0 ∧ y ≠ 0 ∧ x^2 + x*y + y^2 = 0) : complex_exp1990_sum x y h :=
  sorry

end complex_exp1990_sum_theorem_l46_46002


namespace line_through_intersection_of_circles_l46_46515

theorem line_through_intersection_of_circles 
  (x y : ℝ)
  (C1 : x^2 + y^2 = 10)
  (C2 : (x-1)^2 + (y-3)^2 = 20) :
  x + 3 * y = 0 :=
sorry

end line_through_intersection_of_circles_l46_46515


namespace science_book_pages_l46_46169

theorem science_book_pages {history_pages novel_pages science_pages: ℕ} (h1: novel_pages = history_pages / 2) (h2: science_pages = 4 * novel_pages) (h3: history_pages = 300):
  science_pages = 600 :=
by
  sorry

end science_book_pages_l46_46169


namespace part_a_part_b_part_c_part_d_l46_46425

-- (a)
theorem part_a : ∃ x y : ℤ, x > 0 ∧ y > 0 ∧ x ≤ 5 ∧ x^2 - 2 * y^2 = 1 :=
by
  -- proof here
  sorry

-- (b)
theorem part_b : ∃ u v : ℤ, (3 + 2 * Real.sqrt 2)^2 = u + v * Real.sqrt 2 ∧ u^2 - 2 * v^2 = 1 :=
by
  -- proof here
  sorry

-- (c)
theorem part_c : ∀ a b c d : ℤ, a^2 - 2 * b^2 = 1 → (a + b * Real.sqrt 2) * (3 + 2 * Real.sqrt 2) = c + d * Real.sqrt 2
                  → c^2 - 2 * d^2 = 1 :=
by
  -- proof here
  sorry

-- (d)
theorem part_d : ∃ x y : ℤ, y > 100 ∧ x^2 - 2 * y^2 = 1 :=
by
  -- proof here
  sorry

end part_a_part_b_part_c_part_d_l46_46425


namespace sum_of_geometric_sequence_l46_46403

theorem sum_of_geometric_sequence :
  ∀ (a : ℕ → ℝ) (r : ℝ),
  (∃ a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ,
   a 1 = a_1 ∧ a 2 = a_2 ∧ a 3 = a_3 ∧ a 4 = a_4 ∧ a 5 = a_5 ∧ a 6 = a_6 ∧ a 7 = a_7 ∧ a 8 = a_8 ∧ a 9 = a_9 ∧
   a_1 * r^1 = a_2 ∧ a_1 * r^2 = a_3 ∧ a_1 * r^3 = a_4 ∧ a_1 * r^4 = a_5 ∧ a_1 * r^5 = a_6 ∧ a_1 * r^6 = a_7 ∧ a_1 * r^7 = a_8 ∧ a_1 * r^8 = a_9 ∧
   a_1 + a_2 + a_3 = 8 ∧
   a_4 + a_5 + a_6 = -4) →
  a 7 + a 8 + a 9 = 2 :=
sorry

end sum_of_geometric_sequence_l46_46403


namespace largest_multiple_of_8_less_than_100_l46_46862

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, 8 * n < 100 ∧ (∀ m : ℕ, 8 * m < 100 → m ≤ n) ∧ 8 * n = 96 :=
by
  sorry

end largest_multiple_of_8_less_than_100_l46_46862


namespace ladder_slides_out_l46_46850

theorem ladder_slides_out (ladder_length foot_initial_dist ladder_slip_down foot_final_dist : ℝ) 
  (h_ladder_length : ladder_length = 25)
  (h_foot_initial_dist : foot_initial_dist = 7)
  (h_ladder_slip_down : ladder_slip_down = 4)
  (h_foot_final_dist : foot_final_dist = 15) :
  foot_final_dist - foot_initial_dist = 8 :=
  by
  simp [h_ladder_length, h_foot_initial_dist, h_ladder_slip_down, h_foot_final_dist]
  sorry

end ladder_slides_out_l46_46850


namespace Shyam_money_l46_46328

theorem Shyam_money (r g k s : ℕ) 
  (h1 : 7 * g = 17 * r) 
  (h2 : 7 * k = 17 * g)
  (h3 : 11 * s = 13 * k)
  (hr : r = 735) : 
  s = 2119 := 
by
  sorry

end Shyam_money_l46_46328


namespace smallest_geometric_number_l46_46955

noncomputable def is_geometric_sequence (a b c : ℕ) : Prop :=
  b * b = a * c

def is_smallest_geometric_number (n : ℕ) : Prop :=
  n = 261

theorem smallest_geometric_number :
  ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (is_geometric_sequence (n / 100) ((n / 10) % 10) (n % 10)) ∧
  (n / 100 = 2) ∧ (n / 100 ≠ (n / 10) % 10) ∧ (n / 100 ≠ n % 10) ∧ ((n / 10) % 10 ≠ n % 10) ∧
  is_smallest_geometric_number n :=
by
  sorry

end smallest_geometric_number_l46_46955


namespace size_of_can_of_concentrate_l46_46554

theorem size_of_can_of_concentrate
  (can_to_water_ratio : ℕ := 1 + 3)
  (servings_needed : ℕ := 320)
  (serving_size : ℕ := 6)
  (total_volume : ℕ := servings_needed * serving_size) :
  ∃ C : ℕ, C = total_volume / can_to_water_ratio :=
by
  sorry

end size_of_can_of_concentrate_l46_46554


namespace exists_infinite_sets_of_positive_integers_l46_46041

theorem exists_infinite_sets_of_positive_integers (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (S : ℕ → ℕ × ℕ × ℕ), ∀ n : ℕ, S n = (x, y, z) ∧ 
  ((x + y + z)^2 + 2*(x + y + z) = 5*(x*y + y*z + z*x)) :=
sorry

end exists_infinite_sets_of_positive_integers_l46_46041


namespace xiaoming_department_store_profit_l46_46186

theorem xiaoming_department_store_profit:
  let P₁ := 40000   -- average monthly profit in Q1
  let L₂ := -15000  -- average monthly loss in Q2
  let L₃ := -18000  -- average monthly loss in Q3
  let P₄ := 32000   -- average monthly profit in Q4
  let P_total := (P₁ * 3 + L₂ * 3 + L₃ * 3 + P₄ * 3)
  P_total = 117000 := by
  sorry

end xiaoming_department_store_profit_l46_46186


namespace Harriet_sibling_product_l46_46658

-- Definition of the family structure
def Harry : Prop := 
  let sisters := 4
  let brothers := 4
  true

-- Harriet being one of Harry's sisters and calculating her siblings
def Harriet : Prop :=
  let S := 4 - 1 -- Number of Harriet's sisters
  let B := 4 -- Number of Harriet's brothers
  S * B = 12

theorem Harriet_sibling_product : Harry → Harriet := by
  intro h
  let S := 3
  let B := 4
  have : S * B = 12 := by norm_num
  exact this

end Harriet_sibling_product_l46_46658


namespace positive_n_for_one_solution_l46_46820

theorem positive_n_for_one_solution :
  ∀ (n : ℝ), (4 * (0 : ℝ)) ^ 2 + n * (0) + 16 = 0 → (n^2 - 256 = 0) → n = 16 :=
by
  intro n
  intro h
  intro discriminant_eq_zero
  sorry

end positive_n_for_one_solution_l46_46820


namespace consecutive_integer_sum_l46_46335

theorem consecutive_integer_sum (n : ℕ) (h1 : n * (n + 1) = 2720) : n + (n + 1) = 103 :=
sorry

end consecutive_integer_sum_l46_46335


namespace total_votes_l46_46440

theorem total_votes (P R : ℝ) (hP : P = 0.35) (diff : ℝ) (h_diff : diff = 1650) : 
  ∃ V : ℝ, P * V + (P * V + diff) = V ∧ V = 5500 :=
by
  use 5500
  sorry

end total_votes_l46_46440


namespace garden_area_garden_perimeter_l46_46442

noncomputable def length : ℝ := 30
noncomputable def width : ℝ := length / 2
noncomputable def area : ℝ := length * width
noncomputable def perimeter : ℝ := 2 * (length + width)

theorem garden_area :
  area = 450 :=
sorry

theorem garden_perimeter :
  perimeter = 90 :=
sorry

end garden_area_garden_perimeter_l46_46442


namespace reciprocal_of_negative_2023_l46_46560

theorem reciprocal_of_negative_2023 : (1 / (-2023 : ℤ)) = -(1 / (2023 : ℤ)) := by
  sorry

end reciprocal_of_negative_2023_l46_46560


namespace gcf_72_108_l46_46105

def gcf (a b : ℕ) : ℕ := 
  Nat.gcd a b

theorem gcf_72_108 : gcf 72 108 = 36 := by
  sorry

end gcf_72_108_l46_46105


namespace R_and_D_expenditure_l46_46012

theorem R_and_D_expenditure (R_D_t : ℝ) (Delta_APL_t_plus_2 : ℝ) (ratio : ℝ) :
  R_D_t = 3013.94 → Delta_APL_t_plus_2 = 3.29 → ratio = 916 →
  R_D_t / Delta_APL_t_plus_2 = ratio :=
by
  intros hR hD hRto
  rw [hR, hD, hRto]
  sorry

end R_and_D_expenditure_l46_46012


namespace lisa_likes_only_last_digit_zero_l46_46537

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 5

def is_divisible_by_2 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 2 ∨ n % 10 = 4 ∨ n % 10 = 6 ∨ n % 10 = 8

def is_divisible_by_5_and_2 (n : ℕ) : Prop :=
  is_divisible_by_5 n ∧ is_divisible_by_2 n

theorem lisa_likes_only_last_digit_zero : ∀ n, is_divisible_by_5_and_2 n → n % 10 = 0 :=
by
  sorry

end lisa_likes_only_last_digit_zero_l46_46537


namespace interest_difference_l46_46240

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := 
  P * (1 + r)^t - P

theorem interest_difference : 
  simple_interest 500 0.20 2 - (500 * (1 + 0.20)^2 - 500) = 20 := by
  sorry

end interest_difference_l46_46240


namespace solve_for_y_l46_46163

theorem solve_for_y (y : ℤ) : (4 + y) / (6 + y) = (2 + y) / (3 + y) → y = 0 := by 
  sorry

end solve_for_y_l46_46163


namespace part_a_part_b_l46_46281

def square_side_length : ℝ := 10
def square_area (side_length : ℝ) : ℝ := side_length * side_length
def triangle_area (base : ℝ) (height : ℝ) : ℝ := 0.5 * base * height

-- Part (a)
theorem part_a :
  let side_length := square_side_length
  let square := square_area side_length
  let triangle := triangle_area side_length side_length
  square - triangle = 50 := by
  sorry

-- Part (b)
theorem part_b :
  let side_length := square_side_length
  let square := square_area side_length
  let small_triangle_area := square / 8
  2 * small_triangle_area = 25 := by
  sorry

end part_a_part_b_l46_46281


namespace travel_times_either_24_or_72_l46_46382

variable (A B C : String)
variable (travel_time : String → String → Float)
variable (current : Float)

-- Conditions:
-- 1. Travel times are 24 minutes or 72 minutes
-- 2. Traveling from dock B cannot be balanced with current constraints
-- 3. A 3 km travel with the current is 24 minutes
-- 4. A 3 km travel against the current is 72 minutes

theorem travel_times_either_24_or_72 :
  (∀ (P Q : String), P = A ∨ P = B ∨ P = C ∧ Q = A ∨ Q = B ∨ Q = C →
  (travel_time A C = 72 ∨ travel_time C A = 24)) :=
by
  intros
  sorry

end travel_times_either_24_or_72_l46_46382


namespace simplify_and_evaluate_expr_l46_46029

noncomputable def a : ℝ := Real.sqrt 2 - 2

noncomputable def expr (a : ℝ) : ℝ := (a - 2) / (a - 1) / (a + 1 - 3 / (a - 1))

theorem simplify_and_evaluate_expr :
  expr (Real.sqrt 2 - 2) = Real.sqrt 2 / 2 :=
by sorry

end simplify_and_evaluate_expr_l46_46029


namespace max_value_quadratic_l46_46806

theorem max_value_quadratic : ∃ x : ℝ, -9 * x^2 + 27 * x + 15 = 35.25 :=
sorry

end max_value_quadratic_l46_46806


namespace remainder_777_777_mod_13_l46_46520

theorem remainder_777_777_mod_13 : (777^777) % 13 = 1 := by
  sorry

end remainder_777_777_mod_13_l46_46520


namespace place_mat_length_l46_46378

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (x : ℝ)
  (table_is_round : r = 3)
  (number_of_mats : n = 8)
  (mat_width : w = 1)
  (mat_length : ∀ (k: ℕ), 0 ≤ k ∧ k < n → (2 * r * Real.sin (Real.pi / n) = x)) :
  x = (3 * Real.sqrt 35) / 10 + 1 / 2 :=
sorry

end place_mat_length_l46_46378


namespace circle_properties_l46_46257

noncomputable def circle_center (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x = 0

theorem circle_properties (x y : ℝ) :
  circle_center x y ↔ ((x - 2)^2 + y^2 = 2^2) ∧ ((2, 0) = (2, 0)) :=
by
  sorry

end circle_properties_l46_46257


namespace inequality_solution_l46_46395

theorem inequality_solution (a x : ℝ) : 
  (ax^2 + (2 - a) * x - 2 < 0) → 
  ((a = 0) → x < 1) ∧ 
  ((a > 0) → (-2/a < x ∧ x < 1)) ∧ 
  ((a < 0) → 
    ((-2 < a ∧ a < 0) → (x < 1 ∨ x > -2/a)) ∧
    (a = -2 → (x ≠ 1)) ∧
    (a < -2 → (x < -2/a ∨ x > 1)))
:=
sorry

end inequality_solution_l46_46395


namespace mrs_hilt_additional_rocks_l46_46570

-- Definitions from the conditions
def total_rocks : ℕ := 125
def rocks_she_has : ℕ := 64
def additional_rocks_needed : ℕ := total_rocks - rocks_she_has

-- The theorem to prove the question equals the answer given the conditions
theorem mrs_hilt_additional_rocks : additional_rocks_needed = 61 := 
by
  sorry

end mrs_hilt_additional_rocks_l46_46570


namespace find_a20_l46_46076

variables {a : ℕ → ℤ} {S : ℕ → ℤ}
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

theorem find_a20 (h_arith : is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_a1 : a 1 = -1)
  (h_S10 : S 10 = 35) :
  a 20 = 18 :=
sorry

end find_a20_l46_46076


namespace line_has_equal_intercepts_find_a_l46_46873

theorem line_has_equal_intercepts (a : ℝ) :
  (∃ l : ℝ, (l = 0 → ax + y - 2 - a = 0) ∧ (l = 1 → (a = 1 ∨ a = -2))) := sorry

-- formalizing the problem
theorem find_a (a : ℝ) (h_eq_intercepts : ∀ x y : ℝ, (a * x + y - 2 - a = 0 ↔ (x = 2 + a ∧ y = -2 - a))) :
  a = 1 ∨ a = -2 := sorry

end line_has_equal_intercepts_find_a_l46_46873


namespace find_m_l46_46942

theorem find_m (m : ℝ) (A : Set ℝ) (B : Set ℝ) (hA : A = { -1, 2, 2 * m - 1 }) (hB : B = { 2, m^2 }) (hSubset : B ⊆ A) : m = 1 := by
  sorry
 
end find_m_l46_46942


namespace valid_integer_lattice_points_count_l46_46422

def point := (ℤ × ℤ)
def A : point := (-4, 3)
def B : point := (4, -3)

def manhattan_distance (p1 p2 : point) : ℤ :=
  abs (p2.1 - p1.1) + abs (p2.2 - p1.2)

def valid_path_length (p1 p2 : point) : Prop :=
  manhattan_distance p1 p2 ≤ 18

def does_not_cross_y_eq_x (p1 p2 : point) : Prop :=
  ∀ x y, (x, y) ∈ [(p1, p2)] → y ≠ x

def integer_lattice_points_on_path (p1 p2 : point) : ℕ := sorry

theorem valid_integer_lattice_points_count :
  integer_lattice_points_on_path A B = 112 :=
sorry

end valid_integer_lattice_points_count_l46_46422


namespace sum_g_equals_half_l46_46450

noncomputable def g (n : ℕ) : ℝ :=
  ∑' k, if k ≥ 3 then 1 / k ^ n else 0

theorem sum_g_equals_half : ∑' n : ℕ, g n.succ = 1 / 2 := 
sorry

end sum_g_equals_half_l46_46450


namespace Josanna_seventh_test_score_l46_46121

theorem Josanna_seventh_test_score (scores : List ℕ) (h_scores : scores = [95, 85, 75, 65, 90, 70])
                                   (average_increase : ℕ) (h_average_increase : average_increase = 5) :
                                   ∃ x, (List.sum scores + x) / (List.length scores + 1) = (List.sum scores) / (List.length scores) + average_increase := 
by
  sorry

end Josanna_seventh_test_score_l46_46121


namespace cost_per_blue_shirt_l46_46637

theorem cost_per_blue_shirt :
  let pto_spent := 2317
  let num_kindergarten := 101
  let cost_orange := 5.80
  let total_orange := num_kindergarten * cost_orange

  let num_first_grade := 113
  let cost_yellow := 5
  let total_yellow := num_first_grade * cost_yellow

  let num_third_grade := 108
  let cost_green := 5.25
  let total_green := num_third_grade * cost_green

  let total_other_shirts := total_orange + total_yellow + total_green
  let pto_spent_on_blue := pto_spent - total_other_shirts

  let num_second_grade := 107
  let cost_per_blue_shirt := pto_spent_on_blue / num_second_grade

  cost_per_blue_shirt = 5.60 :=
by
  sorry

end cost_per_blue_shirt_l46_46637


namespace x_coordinate_l46_46278

theorem x_coordinate (x : ℝ) (y : ℝ) :
  (∃ m : ℝ, m = (0 + 6) / (4 + 8) ∧
            y + 6 = m * (x + 8) ∧
            y = 3) →
  x = 10 :=
by
  sorry

end x_coordinate_l46_46278


namespace translation_correctness_l46_46458

-- Define the original function
def original_function (x : ℝ) : ℝ := 3 * x + 5

-- Define the translated function
def translated_function (x : ℝ) : ℝ := 3 * x

-- Define the condition for passing through the origin
def passes_through_origin (f : ℝ → ℝ) : Prop := f 0 = 0

-- The theorem to prove the correct translation
theorem translation_correctness : passes_through_origin translated_function := by
  sorry

end translation_correctness_l46_46458


namespace total_time_to_school_and_back_l46_46847

-- Definition of the conditions
def speed_to_school : ℝ := 3 -- in km/hr
def speed_back_home : ℝ := 2 -- in km/hr
def distance : ℝ := 6 -- in km

-- Proof statement
theorem total_time_to_school_and_back : 
  (distance / speed_to_school) + (distance / speed_back_home) = 5 := 
by
  sorry

end total_time_to_school_and_back_l46_46847


namespace length_of_rectangle_l46_46303

-- Define the conditions as given in the problem
variables (width : ℝ) (perimeter : ℝ) (length : ℝ)

-- The conditions provided
def conditions : Prop :=
  width = 15 ∧ perimeter = 70 ∧ perimeter = 2 * (length + width)

-- The statement to prove: the length of the rectangle is 20 feet
theorem length_of_rectangle {width perimeter length : ℝ} (h : conditions width perimeter length) : length = 20 :=
by 
  -- This is where the proof steps would go
  sorry

end length_of_rectangle_l46_46303


namespace wages_of_one_man_l46_46350

variable (R : Type) [DivisionRing R] [DecidableEq R]
variable (money : R)
variable (num_men : ℕ := 5)
variable (num_women : ℕ := 8)
variable (total_wages : R := 180)
variable (wages_men : R := 36)

axiom equal_women : num_men = num_women
axiom total_earnings (wages : ℕ → R) :
  (wages num_men) + (wages num_women) + (wages 8) = total_wages

theorem wages_of_one_man :
  wages_men = total_wages / num_men := by
  sorry

end wages_of_one_man_l46_46350


namespace spherical_coordinates_equivalence_l46_46859

theorem spherical_coordinates_equivalence
  (ρ θ φ : ℝ)
  (h_ρ : ρ > 0)
  (h_θ : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (h_φ : φ = 2 * Real.pi - (7 * Real.pi / 4)) :
  (ρ, θ, φ) = (4, 3 * Real.pi / 4, Real.pi / 4) :=
by 
  sorry

end spherical_coordinates_equivalence_l46_46859


namespace find_second_number_l46_46139

-- Define the given number
def given_number := 220070

-- Define the constants in the problem
def constant_555 := 555
def remainder := 70

-- Define the second number (our unknown)
variable (x : ℕ)

-- Define the condition as an equation
def condition : Prop :=
  given_number = (constant_555 + x) * 2 * (x - constant_555) + remainder

-- The theorem to prove that the second number is 343
theorem find_second_number : ∃ x : ℕ, condition x ∧ x = 343 :=
sorry

end find_second_number_l46_46139


namespace baseball_team_grouping_l46_46072

theorem baseball_team_grouping (new_players returning_players : ℕ) (group_size : ℕ) 
  (h_new : new_players = 4) (h_returning : returning_players = 6) (h_group : group_size = 5) : 
  (new_players + returning_players) / group_size = 2 := 
  by 
  sorry

end baseball_team_grouping_l46_46072


namespace cos_225_eq_neg_sqrt2_div2_l46_46279

theorem cos_225_eq_neg_sqrt2_div2 : Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

end cos_225_eq_neg_sqrt2_div2_l46_46279


namespace line_slope_intercept_l46_46864

theorem line_slope_intercept :
  (∀ (x y : ℝ), 3 * (x + 2) - 4 * (y - 8) = 0 → y = (3/4) * x + 9.5) :=
sorry

end line_slope_intercept_l46_46864


namespace proof_problem_l46_46652

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

theorem proof_problem :
  (∃ A ω φ, (A = 2) ∧ (ω = 2) ∧ (φ = Real.pi / 4) ∧
  f (3 * Real.pi / 8) = 0 ∧
  f (Real.pi / 8) = 2 ∧
  (∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → f x ≤ 2) ∧
  (∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → f x ≥ -Real.sqrt 2) ∧
  f (-Real.pi / 4) = -Real.sqrt 2) :=
sorry

end proof_problem_l46_46652


namespace find_point_C_l46_46146

noncomputable def point_on_z_axis (z : ℝ) : ℝ × ℝ × ℝ := (0, 0, z)
def point_A : ℝ × ℝ × ℝ := (1, 0, 2)
def point_B : ℝ × ℝ × ℝ := (1, 1, 1)

theorem find_point_C :
  ∃ C : ℝ × ℝ × ℝ, (C = point_on_z_axis 1) ∧ (dist C point_A = dist C point_B) :=
by
  sorry

end find_point_C_l46_46146


namespace largest_k_divides_3n_plus_1_l46_46796

theorem largest_k_divides_3n_plus_1 (n : ℕ) (hn : 0 < n) : ∃ k : ℕ, k = 2 ∧ n % 2 = 1 ∧ 2^k ∣ 3^n + 1 ∨ k = 1 ∧ n % 2 = 0 ∧ 2^k ∣ 3^n + 1 :=
sorry

end largest_k_divides_3n_plus_1_l46_46796


namespace at_least_one_bigger_than_44_9_l46_46898

noncomputable def x : ℕ → ℝ := sorry
noncomputable def y : ℕ → ℝ := sorry

axiom x_positive (n : ℕ) : 0 < x n
axiom y_positive (n : ℕ) : 0 < y n
axiom recurrence_x (n : ℕ) : x (n + 1) = x n + 1 / (2 * y n)
axiom recurrence_y (n : ℕ) : y (n + 1) = y n + 1 / (2 * x n)

theorem at_least_one_bigger_than_44_9 : x 2018 > 44.9 ∨ y 2018 > 44.9 :=
sorry

end at_least_one_bigger_than_44_9_l46_46898


namespace if_2_3_4_then_1_if_1_3_4_then_2_l46_46556

variables {Plane Line : Type} 
variables (α β : Plane) (m n : Line)

-- assuming the perpendicular relationships as predicates
variable (perp : Plane → Plane → Prop) -- perpendicularity between planes
variable (perp' : Line → Line → Prop) -- perpendicularity between lines
variable (perp'' : Line → Plane → Prop) -- perpendicularity between line and plane

theorem if_2_3_4_then_1 :
  perp α β → perp'' m β → perp'' n α → perp' m n :=
by
  sorry

theorem if_1_3_4_then_2 :
  perp' m n → perp'' m β → perp'' n α → perp α β :=
by
  sorry

end if_2_3_4_then_1_if_1_3_4_then_2_l46_46556


namespace length_AE_l46_46320

/-- Given points A, B, C, D, and E on a plane with distances:
  - CA = 12,
  - AB = 8,
  - BC = 4,
  - CD = 5,
  - DB = 3,
  - BE = 6,
  - ED = 3.
  Prove that AE = sqrt 113.
--/
theorem length_AE (A B C D E : ℝ × ℝ)
  (h1 : dist C A = 12)
  (h2 : dist A B = 8)
  (h3 : dist B C = 4)
  (h4 : dist C D = 5)
  (h5 : dist D B = 3)
  (h6 : dist B E = 6)
  (h7 : dist E D = 3) : 
  dist A E = Real.sqrt 113 := 
  by 
    sorry

end length_AE_l46_46320


namespace race_lead_distance_l46_46056

theorem race_lead_distance :
  ∀ (d12 d13 : ℝ) (s1 s2 s3 t : ℝ), 
  d12 = 2 →
  d13 = 4 →
  t > 0 →
  s1 = (d12 / t + s2) →
  s1 = (d13 / t + s3) →
  s2 * t - s3 * t = 2.5 :=
by
  sorry

end race_lead_distance_l46_46056


namespace infinite_geometric_series_sum_l46_46510

theorem infinite_geometric_series_sum :
  let a := (4 : ℚ) / 3
  let r := -(9 : ℚ) / 16
  (a / (1 - r)) = (64 : ℚ) / 75 :=
by
  sorry

end infinite_geometric_series_sum_l46_46510


namespace min_value_on_top_layer_l46_46407

-- Definitions reflecting conditions
def bottom_layer : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def block_value (layer : List ℕ) (i : ℕ) : ℕ :=
  layer.getD (i-1) 0 -- assuming 1-based indexing

def second_layer_values : List ℕ :=
  [block_value bottom_layer 1 + block_value bottom_layer 2 + block_value bottom_layer 3,
   block_value bottom_layer 2 + block_value bottom_layer 3 + block_value bottom_layer 4,
   block_value bottom_layer 4 + block_value bottom_layer 5 + block_value bottom_layer 6,
   block_value bottom_layer 5 + block_value bottom_layer 6 + block_value bottom_layer 7,
   block_value bottom_layer 7 + block_value bottom_layer 8 + block_value bottom_layer 9,
   block_value bottom_layer 8 + block_value bottom_layer 9 + block_value bottom_layer 10]

def third_layer_values : List ℕ :=
  [second_layer_values.getD 0 0 + second_layer_values.getD 1 0 + second_layer_values.getD 2 0,
   second_layer_values.getD 1 0 + second_layer_values.getD 2 0 + second_layer_values.getD 3 0,
   second_layer_values.getD 3 0 + second_layer_values.getD 4 0 + second_layer_values.getD 5 0]

def top_layer_value : ℕ :=
  third_layer_values.getD 0 0 + third_layer_values.getD 1 0 + third_layer_values.getD 2 0

theorem min_value_on_top_layer : top_layer_value = 114 :=
by
  have h0 := block_value bottom_layer 1 -- intentionally leaving this incomplete as we're skipping the actual proof
  sorry

end min_value_on_top_layer_l46_46407


namespace digit_in_tens_place_is_nine_l46_46754

/-
Given:
1. Two numbers represented as 6t5 and 5t6 (where t is a digit).
2. The result of subtracting these two numbers is 9?4, where '?' represents a single digit in the tens place.

Prove:
The digit represented by '?' in the tens place is 9.
-/

theorem digit_in_tens_place_is_nine (t : ℕ) (h1 : 0 ≤ t ∧ t ≤ 9) :
  let a := 600 + t * 10 + 5
  let b := 500 + t * 10 + 6
  let result := a - b
  (result % 100) / 10 = 9 :=
by {
  sorry
}

end digit_in_tens_place_is_nine_l46_46754


namespace tangent_line_at_x_is_2_l46_46077

noncomputable def curve (x : ℝ) : ℝ := (1/4) * x^2 - 3 * Real.log x

theorem tangent_line_at_x_is_2 :
  ∃ x₀ : ℝ, (x₀ > 0) ∧ ((1/2) * x₀ - (3 / x₀) = -1/2) ∧ x₀ = 2 :=
by
  sorry

end tangent_line_at_x_is_2_l46_46077


namespace inverse_function_value_l46_46708

theorem inverse_function_value (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x^3 + 4) :
  f⁻¹ 58 = 3 :=
by sorry

end inverse_function_value_l46_46708


namespace intersection_M_N_eq_l46_46966

-- Define the set M
def M : Set ℝ := {0, 1, 2}

-- Define the set N based on the given inequality
def N : Set ℝ := {x | x^2 - 3 * x + 2 ≤ 0}

-- The statement we want to prove
theorem intersection_M_N_eq {M N: Set ℝ} (hm: M = {0, 1, 2}) 
  (hn: N = {x | x^2 - 3 * x + 2 ≤ 0}) : 
  M ∩ N = {1, 2} :=
sorry

end intersection_M_N_eq_l46_46966


namespace expression_value_l46_46528

-- Define the given condition as an assumption
variable (x : ℝ)
variable (h : 2 * x^2 + 3 * x - 1 = 7)

-- Define the target expression and the required result
theorem expression_value :
  4 * x^2 + 6 * x + 9 = 25 :=
by
  sorry

end expression_value_l46_46528


namespace fraction_habitable_surface_l46_46651

def fraction_exposed_land : ℚ := 3 / 8
def fraction_inhabitable_land : ℚ := 2 / 3

theorem fraction_habitable_surface :
  fraction_exposed_land * fraction_inhabitable_land = 1 / 4 := by
    -- proof steps omitted
    sorry

end fraction_habitable_surface_l46_46651


namespace sum_of_squares_eq_l46_46217

theorem sum_of_squares_eq :
  ∀ (M G D : ℝ), 
  (M = G / 3) → 
  (G = 450) → 
  (D = 2 * G) → 
  (M^2 + G^2 + D^2 = 1035000) :=
by
  intros M G D hM hG hD
  sorry

end sum_of_squares_eq_l46_46217


namespace card_arrangement_bound_l46_46322

theorem card_arrangement_bound : 
  ∀ (cards : ℕ) (cells : ℕ), cards = 1000 → cells = 1994 → 
  ∃ arrangements : ℕ, arrangements = cells - cards + 1 ∧ arrangements < 500000 :=
by {
  sorry
}

end card_arrangement_bound_l46_46322


namespace star_sum_interior_angles_l46_46602

theorem star_sum_interior_angles (n : ℕ) (h : n ≥ 6) :
  let S := 180 * n - 360
  S = 180 * (n - 2) :=
by
  let S := 180 * n - 360
  show S = 180 * (n - 2)
  sorry

end star_sum_interior_angles_l46_46602


namespace smallest_square_side_lengths_l46_46641

theorem smallest_square_side_lengths (x : ℕ) 
    (h₁ : ∀ (y : ℕ), y = x + 8) 
    (h₂ : ∀ (z : ℕ), z = 50) 
    (h₃ : ∀ (QS PS RT QT : ℕ), QS = 8 ∧ PS = x ∧ RT = 42 - x ∧ QT = x + 8 ∧ (8 / x) = ((42 - x) / (x + 8))) : 
  x = 2 ∨ x = 32 :=
by 
  sorry

end smallest_square_side_lengths_l46_46641


namespace find_smallest_N_l46_46461

def smallest_possible_N (N : ℕ) : Prop :=
  ∃ (W : Fin N → ℝ), 
  (∀ i j, W i ≤ 1.25 * W j ∧ W j ≤ 1.25 * W i) ∧ 
  (∃ (P : Fin 10 → Finset (Fin N)), ∀ i j, i ≤ j →
    P i ≠ ∅ ∧ 
    Finset.sum (P i) W = Finset.sum (P j) W) ∧
  (∃ (V : Fin 11 → Finset (Fin N)), ∀ i j, i ≤ j →
    V i ≠ ∅ ∧ 
    Finset.sum (V i) W = Finset.sum (V j) W)

theorem find_smallest_N : smallest_possible_N 50 :=
sorry

end find_smallest_N_l46_46461


namespace initial_weight_l46_46595

theorem initial_weight (lost_weight current_weight : ℕ) (h1 : lost_weight = 35) (h2 : current_weight = 34) :
  lost_weight + current_weight = 69 :=
sorry

end initial_weight_l46_46595


namespace find_k_l46_46589

-- Define the equation of line m
def line_m (x : ℝ) : ℝ := 2 * x + 8

-- Define the equation of line n with an unknown slope k
def line_n (k : ℝ) (x : ℝ) : ℝ := k * x - 9

-- Define the point of intersection
def intersection_point := (-4, 0)

-- The proof statement
theorem find_k : ∃ k : ℝ, k = -9 / 4 ∧ line_m (-4) = 0 ∧ line_n k (-4) = 0 :=
by
  exists (-9 / 4)
  simp [line_m, line_n, intersection_point]
  sorry

end find_k_l46_46589


namespace radian_measure_sector_l46_46059

theorem radian_measure_sector (r l : ℝ) (h1 : 2 * r + l = 12) (h2 : (1 / 2) * l * r = 8) :
  l / r = 1 ∨ l / r = 4 := by
  sorry

end radian_measure_sector_l46_46059


namespace min_selling_price_is_400_l46_46765

-- Definitions for the problem conditions
def total_products := 20
def average_price := 1200
def less_than_1000_count := 10
def price_of_most_expensive := 11000
def total_retail_price := total_products * average_price

-- The theorem to state the problem condition and the expected result
theorem min_selling_price_is_400 (x : ℕ) :
  -- Condition 1: Total retail price
  total_retail_price =
  -- 10 products sell for x dollars
  (10 * x) +
  -- 9 products sell for 1000 dollars
  (9 * 1000) +
  -- 1 product sells for the maximum price 11000
  price_of_most_expensive → 
  -- Conclusion: The minimum price x is 400
  x = 400 :=
by
  sorry

end min_selling_price_is_400_l46_46765


namespace total_artworks_created_l46_46162

theorem total_artworks_created
  (students_group1 : ℕ := 24) (students_group2 : ℕ := 12)
  (kits_total : ℕ := 48)
  (kits_per_3_students : ℕ := 3) (kits_per_2_students : ℕ := 2)
  (artwork_types : ℕ := 3)
  (paintings_group1_1 : ℕ := 12 * 2) (drawings_group1_1 : ℕ := 12 * 4) (sculptures_group1_1 : ℕ := 12 * 1)
  (paintings_group1_2 : ℕ := 12 * 1) (drawings_group1_2 : ℕ := 12 * 5) (sculptures_group1_2 : ℕ := 12 * 3)
  (paintings_group2_1 : ℕ := 4 * 3) (drawings_group2_1 : ℕ := 4 * 6) (sculptures_group2_1 : ℕ := 4 * 3)
  (paintings_group2_2 : ℕ := 8 * 4) (drawings_group2_2 : ℕ := 8 * 7) (sculptures_group2_2 : ℕ := 8 * 1)
  : (paintings_group1_1 + paintings_group1_2 + paintings_group2_1 + paintings_group2_2) +
    (drawings_group1_1 + drawings_group1_2 + drawings_group2_1 + drawings_group2_2) +
    (sculptures_group1_1 + sculptures_group1_2 + sculptures_group2_1 + sculptures_group2_2) = 336 :=
by sorry

end total_artworks_created_l46_46162


namespace probability_green_light_is_8_over_15_l46_46743

def total_cycle_duration (red yellow green : ℕ) : ℕ :=
  red + yellow + green

def probability_green_light (red yellow green : ℕ) : ℚ :=
  green / (total_cycle_duration red yellow green : ℚ)

theorem probability_green_light_is_8_over_15 :
  probability_green_light 30 5 40 = 8 / 15 := by
  sorry

end probability_green_light_is_8_over_15_l46_46743


namespace greatest_number_of_donated_oranges_greatest_number_of_donated_cookies_l46_46928

-- Define the given conditions
def totalOranges : ℕ := 81
def totalCookies : ℕ := 65
def numberOfChildren : ℕ := 7

-- Define the floor division for children
def orangesPerChild : ℕ := totalOranges / numberOfChildren
def cookiesPerChild : ℕ := totalCookies / numberOfChildren

-- Calculate leftover (donated) quantities
def orangesLeftover : ℕ := totalOranges % numberOfChildren
def cookiesLeftover : ℕ := totalCookies % numberOfChildren

-- Statements to prove
theorem greatest_number_of_donated_oranges : orangesLeftover = 4 := by {
    sorry
}

theorem greatest_number_of_donated_cookies : cookiesLeftover = 2 := by {
    sorry
}

end greatest_number_of_donated_oranges_greatest_number_of_donated_cookies_l46_46928


namespace greatest_value_x_is_correct_l46_46962

noncomputable def greatest_value_x : ℝ :=
-8 + Real.sqrt 6

theorem greatest_value_x_is_correct :
  ∀ x : ℝ, (x ≠ 9) → ((x^2 - x - 90) / (x - 9) = 2 / (x + 6)) → x ≤ greatest_value_x :=
by
  sorry

end greatest_value_x_is_correct_l46_46962


namespace probability_of_pairing_with_friends_l46_46064

theorem probability_of_pairing_with_friends (n : ℕ) (f : ℕ) (h1 : n = 32) (h2 : f = 2):
  (f / (n - 1) : ℚ) = 2 / 31 :=
by
  rw [h1, h2]
  norm_num

end probability_of_pairing_with_friends_l46_46064


namespace find_a_value_l46_46196

noncomputable def a : ℝ := (384:ℝ)^(1/7)

variables (a b c : ℝ)
variables (h1 : a^2 / b = 2) (h2 : b^2 / c = 4) (h3 : c^2 / a = 6)

theorem find_a_value : a = 384^(1/7) :=
by
  sorry

end find_a_value_l46_46196


namespace eccentricity_squared_l46_46469

-- Define the hyperbola and its properties
variables (a b c e : ℝ) (x₁ y₁ x₂ y₂ : ℝ)

-- Define the hyperbola equation and conditions
def hyperbola_eq (a b x y : ℝ) := (x^2)/(a^2) - (y^2)/(b^2) = 1

def midpoint_eq (x₁ y₁ x₂ y₂ : ℝ) := x₁ + x₂ = -4 ∧ y₁ + y₂ = 2

def slope_eq (a b c : ℝ) := -b / c = (b^2 * (-4)) / (a^2 * 2)

-- Define the proof
theorem eccentricity_squared :
  a > 0 → b > 0 → hyperbola_eq a b x₁ y₁ → hyperbola_eq a b x₂ y₂ → midpoint_eq x₁ y₁ x₂ y₂ →
  slope_eq a b c → c^2 = a^2 + b^2 → (e = c / a) → e^2 = (Real.sqrt 2 + 1) / 2 :=
by
  intro ha hb h1 h2 h3 h4 h5 he
  sorry

end eccentricity_squared_l46_46469


namespace hannahs_trip_cost_l46_46533

noncomputable def calculate_gas_cost (initial_odometer final_odometer : ℕ) (fuel_economy_mpg : ℚ) (cost_per_gallon : ℚ) : ℚ :=
  let distance := final_odometer - initial_odometer
  let fuel_used := distance / fuel_economy_mpg
  fuel_used * cost_per_gallon

theorem hannahs_trip_cost :
  calculate_gas_cost 36102 36131 32 (385 / 100) = 276 / 100 :=
by
  sorry

end hannahs_trip_cost_l46_46533


namespace distance_between_homes_l46_46866

theorem distance_between_homes (Maxwell_distance : ℝ) (Maxwell_speed : ℝ) (Brad_speed : ℝ) (midpoint : ℝ) 
    (h1 : Maxwell_speed = 2) 
    (h2 : Brad_speed = 4) 
    (h3 : Maxwell_distance = 12) 
    (h4 : midpoint = Maxwell_distance * 2 * (Brad_speed / Maxwell_speed) + Maxwell_distance) :
midpoint = 36 :=
by
  sorry

end distance_between_homes_l46_46866


namespace scientific_notation_of_845_billion_l46_46225

/-- Express 845 billion yuan in scientific notation. -/
theorem scientific_notation_of_845_billion :
  (845 * (10^9 : ℝ)) / (10^9 : ℝ) = 8.45 * 10^3 :=
by
  sorry

end scientific_notation_of_845_billion_l46_46225


namespace number_equation_form_l46_46329

variable (a : ℝ)

theorem number_equation_form :
  3 * a + 5 = 4 * a := 
sorry

end number_equation_form_l46_46329


namespace candy_ratio_l46_46359

theorem candy_ratio 
  (tabitha_candy : ℕ)
  (stan_candy : ℕ)
  (julie_candy : ℕ)
  (carlos_candy : ℕ)
  (total_candy : ℕ)
  (h1 : tabitha_candy = 22)
  (h2 : stan_candy = 13)
  (h3 : julie_candy = tabitha_candy / 2)
  (h4 : total_candy = 72)
  (h5 : tabitha_candy + stan_candy + julie_candy + carlos_candy = total_candy) :
  carlos_candy / stan_candy = 2 :=
by
  sorry

end candy_ratio_l46_46359


namespace total_apples_l46_46381

-- Definitions based on the problem conditions
def marin_apples : ℕ := 8
def david_apples : ℕ := (3 * marin_apples) / 4
def amanda_apples : ℕ := (3 * david_apples) / 2 + 2

-- The statement that we need to prove
theorem total_apples : marin_apples + david_apples + amanda_apples = 25 := by
  -- The proof steps will go here
  sorry

end total_apples_l46_46381


namespace largest_common_term_arith_seq_l46_46379

theorem largest_common_term_arith_seq :
  ∃ a, a < 90 ∧ (∃ n : ℤ, a = 3 + 8 * n) ∧ (∃ m : ℤ, a = 5 + 9 * m) ∧ a = 59 :=
by
  sorry

end largest_common_term_arith_seq_l46_46379


namespace negation_of_proposition_l46_46244

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x > 0 → 3 * x^2 - x - 2 > 0) ↔ (∃ x : ℝ, x > 0 ∧ 3 * x^2 - x - 2 ≤ 0) :=
by
  sorry

end negation_of_proposition_l46_46244


namespace find_t_max_value_of_xyz_l46_46317

-- Problem (1)
theorem find_t (t : ℝ) (x : ℝ) (h1 : |2 * x + t| - t ≤ 8) (sol_set : -5 ≤ x ∧ x ≤ 4) : t = 1 :=
sorry

-- Problem (2)
theorem max_value_of_xyz (x y z : ℝ) (h2 : x^2 + (1/4) * y^2 + (1/9) * z^2 = 2) : x + y + z ≤ 2 * Real.sqrt 7 :=
sorry

end find_t_max_value_of_xyz_l46_46317


namespace max_omega_is_2_l46_46672

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6)

theorem max_omega_is_2 {ω : ℝ} (h₀ : ω > 0) (h₁ : MonotoneOn (f ω) (Set.Icc (-Real.pi / 6) (Real.pi / 6))) :
  ω ≤ 2 :=
sorry

end max_omega_is_2_l46_46672


namespace fraction_of_a_mile_additional_charge_l46_46125

-- Define the conditions
def initial_fee : ℚ := 2.25
def charge_per_fraction : ℚ := 0.25
def total_charge : ℚ := 4.50
def total_distance : ℚ := 3.6

-- Define the problem statement to prove
theorem fraction_of_a_mile_additional_charge :
  initial_fee = 2.25 →
  charge_per_fraction = 0.25 →
  total_charge = 4.50 →
  total_distance = 3.6 →
  total_distance - (total_charge - initial_fee) = 1.35 :=
by
  intros
  sorry

end fraction_of_a_mile_additional_charge_l46_46125


namespace solve_for_x_l46_46265

theorem solve_for_x : ∃ x : ℤ, x + 1 = 5 ∧ x = 4 :=
by
  sorry

end solve_for_x_l46_46265


namespace line_through_fixed_point_l46_46592

theorem line_through_fixed_point (a : ℝ) :
  ∃ P : ℝ × ℝ, (P = (1, 2)) ∧ (∀ x y, a * x + y - a - 2 = 0 → P = (x, y)) ∧
  ((∃ a, x + y = a ∧ x = 1 ∧ y = 2) → (a = 3)) :=
by
  sorry

end line_through_fixed_point_l46_46592


namespace seq_a5_eq_one_ninth_l46_46700

theorem seq_a5_eq_one_ninth (a : ℕ → ℚ) (h1 : a 1 = 1) (h_rec : ∀ n, a (n + 1) = a n / (2 * a n + 1)) :
  a 5 = 1 / 9 :=
sorry

end seq_a5_eq_one_ninth_l46_46700


namespace square_plot_area_l46_46669

theorem square_plot_area (cost_per_foot : ℕ) (total_cost : ℕ) (P : ℕ) :
  cost_per_foot = 54 →
  total_cost = 3672 →
  P = 4 * (total_cost / (4 * cost_per_foot)) →
  (total_cost / (4 * cost_per_foot)) ^ 2 = 289 :=
by
  intros h_cost_per_foot h_total_cost h_perimeter
  sorry

end square_plot_area_l46_46669


namespace select_student_B_l46_46956

-- Define the average scores for the students A, B, C, D
def avg_A : ℝ := 85
def avg_B : ℝ := 90
def avg_C : ℝ := 90
def avg_D : ℝ := 85

-- Define the variances for the students A, B, C, D
def var_A : ℝ := 50
def var_B : ℝ := 42
def var_C : ℝ := 50
def var_D : ℝ := 42

-- Theorem stating the selected student should be B
theorem select_student_B (avg_A avg_B avg_C avg_D var_A var_B var_C var_D : ℝ)
  (h_avg_A : avg_A = 85) (h_avg_B : avg_B = 90) (h_avg_C : avg_C = 90) (h_avg_D : avg_D = 85)
  (h_var_A : var_A = 50) (h_var_B : var_B = 42) (h_var_C : var_C = 50) (h_var_D : var_D = 42) :
  (avg_B = 90 ∧ avg_C = 90 ∧ avg_B ≥ avg_A ∧ avg_B ≥ avg_D ∧ var_B < var_C) → 
  (select_student = "B") :=
by
  sorry

end select_student_B_l46_46956


namespace increment_M0_to_M1_increment_M0_to_M2_increment_M0_to_M3_l46_46071

-- Define the function z = x * y
def z (x y : ℝ) : ℝ := x * y

-- Initial point M0
def M0 : ℝ × ℝ := (1, 2)

-- Points to which we move
def M1 : ℝ × ℝ := (1.1, 2)
def M2 : ℝ × ℝ := (1, 1.9)
def M3 : ℝ × ℝ := (1.1, 2.2)

-- Proofs for the increments
theorem increment_M0_to_M1 : z M1.1 M1.2 - z M0.1 M0.2 = 0.2 :=
by sorry

theorem increment_M0_to_M2 : z M2.1 M2.2 - z M0.1 M0.2 = -0.1 :=
by sorry

theorem increment_M0_to_M3 : z M3.1 M3.2 - z M0.1 M0.2 = 0.42 :=
by sorry

end increment_M0_to_M1_increment_M0_to_M2_increment_M0_to_M3_l46_46071


namespace find_x_l46_46662

theorem find_x (x y : ℝ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 :=
by
  sorry

end find_x_l46_46662


namespace necessary_not_sufficient_l46_46083

theorem necessary_not_sufficient (m : ℝ) (x : ℝ) (h₁ : m > 0) (h₂ : 0 < x ∧ x < m) (h₃ : x / (x - 1) < 0) 
: m = 1 / 2 := 
sorry

end necessary_not_sufficient_l46_46083


namespace find_f2_l46_46022

def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := by
  sorry

end find_f2_l46_46022


namespace evaluate_f_l46_46650

def f (n : ℕ) : ℕ :=
  if n < 4 then n^2 - 1 else 3*n - 2

theorem evaluate_f (h : f (f (f 2)) = 22) : f (f (f 2)) = 22 :=
by
  -- we state the final result directly
  sorry

end evaluate_f_l46_46650


namespace two_point_two_five_as_fraction_l46_46508

theorem two_point_two_five_as_fraction : (2.25 : ℚ) = 9 / 4 := 
by 
  -- Proof steps would be added here
  sorry

end two_point_two_five_as_fraction_l46_46508


namespace mike_worked_four_hours_l46_46764

-- Define the time to perform each task in minutes
def time_wash_car : ℕ := 10
def time_change_oil : ℕ := 15
def time_change_tires : ℕ := 30

-- Define the number of tasks Mike performed
def num_wash_cars : ℕ := 9
def num_change_oil : ℕ := 6
def num_change_tires : ℕ := 2

-- Define the total minutes Mike worked
def total_minutes_worked : ℕ :=
  (num_wash_cars * time_wash_car) +
  (num_change_oil * time_change_oil) +
  (num_change_tires * time_change_tires)

-- Define the conversion from minutes to hours
def total_hours_worked : ℕ := total_minutes_worked / 60

-- Formalize the proof statement
theorem mike_worked_four_hours :
  total_hours_worked = 4 :=
by
  sorry

end mike_worked_four_hours_l46_46764


namespace banana_to_pear_equiv_l46_46500

/-
Given conditions:
1. 5 bananas cost as much as 3 apples.
2. 9 apples cost the same as 6 pears.
Prove the equivalence between 30 bananas and 12 pears.

We will define the equivalences as constants and prove the cost equivalence.
-/

variable (cost_banana cost_apple cost_pear : ℤ)

noncomputable def cost_equiv : Prop :=
  (5 * cost_banana = 3 * cost_apple) ∧ 
  (9 * cost_apple = 6 * cost_pear) →
  (30 * cost_banana = 12 * cost_pear)

theorem banana_to_pear_equiv :
  cost_equiv cost_banana cost_apple cost_pear :=
by
  sorry

end banana_to_pear_equiv_l46_46500


namespace sufficient_no_x_axis_intersections_l46_46459

/-- Sufficient condition for no x-axis intersections -/
theorem sufficient_no_x_axis_intersections
    (a b c : ℝ)
    (h : a ≠ 0)
    (h_sufficient : b^2 - 4 * a * c < -1) :
    ∀ x : ℝ, ¬(a * x^2 + b * x + c = 0) :=
by
  sorry

end sufficient_no_x_axis_intersections_l46_46459


namespace counter_example_not_power_of_4_for_25_l46_46233

theorem counter_example_not_power_of_4_for_25 : ∃ n ≥ 2, n = 25 ∧ ¬ ∃ k : ℕ, 2 ^ (2 ^ n) % (2 ^ n - 1) = 4 ^ k :=
by {
  sorry
}

end counter_example_not_power_of_4_for_25_l46_46233


namespace special_number_exists_l46_46466

theorem special_number_exists (a b c d e : ℕ) (h1 : a < b ∧ b < c ∧ c < d ∧ d < e)
    (h2 : a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e) 
    (h_num : a * 10 + b = 13 ∧ c = 4 ∧ d * 10 + e = 52) :
    (10 * a + b) * c = 10 * d + e :=
by
  sorry

end special_number_exists_l46_46466


namespace total_students_count_l46_46405

-- Define the conditions
def num_rows : ℕ := 8
def students_per_row : ℕ := 6
def students_last_row : ℕ := 5
def rows_with_six_students : ℕ := 7

-- Define the total students
def total_students : ℕ :=
  (rows_with_six_students * students_per_row) + students_last_row

-- The theorem to prove
theorem total_students_count : total_students = 47 := by
  sorry

end total_students_count_l46_46405


namespace candies_remaining_l46_46943

theorem candies_remaining (r y b : ℕ) 
  (h_r : r = 40)
  (h_y : y = 3 * r - 20)
  (h_b : b = y / 2) :
  r + b = 90 := by
  sorry

end candies_remaining_l46_46943


namespace area_of_45_45_90_triangle_l46_46870

noncomputable def leg_length (hypotenuse : ℝ) : ℝ :=
  hypotenuse / Real.sqrt 2

theorem area_of_45_45_90_triangle (hypotenuse : ℝ) (h : hypotenuse = 13) : 
  (1 / 2) * (leg_length hypotenuse) * (leg_length hypotenuse) = 84.5 :=
by
  sorry

end area_of_45_45_90_triangle_l46_46870


namespace interest_rate_per_annum_l46_46543

theorem interest_rate_per_annum (P T : ℝ) (r : ℝ) 
  (h1 : P = 15000) 
  (h2 : T = 2)
  (h3 : P * (1 + r)^T - P - (P * r * T) = 150) : 
  r = 0.1 :=
by
  sorry

end interest_rate_per_annum_l46_46543


namespace geometric_progression_fourth_term_l46_46091

theorem geometric_progression_fourth_term :
  ∀ (a₁ a₂ a₃ a₄ : ℝ), a₁ = 2^(1/2) ∧ a₂ = 2^(1/4) ∧ a₃ = 2^(1/6) ∧ (a₂ / a₁ = r) ∧ (a₃ = a₂ * r⁻¹) ∧ (a₄ = a₃ * r) → a₄ = 2^(1/8) := by
intro a₁ a₂ a₃ a₄
intro h
sorry

end geometric_progression_fourth_term_l46_46091


namespace points_earned_l46_46164

-- Definitions from conditions
def points_per_enemy : ℕ := 8
def total_enemies : ℕ := 7
def enemies_not_destroyed : ℕ := 2

-- The proof statement
theorem points_earned :
  points_per_enemy * (total_enemies - enemies_not_destroyed) = 40 := 
by
  sorry

end points_earned_l46_46164


namespace company_members_and_days_l46_46961

theorem company_members_and_days {t n : ℕ} (h : t = 6) :
    n = (t * (t - 1)) / 2 → n = 15 :=
by
  intro hn
  rw [h] at hn
  simp at hn
  exact hn

end company_members_and_days_l46_46961


namespace no_perfect_squares_in_sequence_l46_46519

theorem no_perfect_squares_in_sequence (x : ℕ → ℤ) (h₀ : x 0 = 1) (h₁ : x 1 = 3)
  (h_rec : ∀ n : ℕ, x (n + 1) = 6 * x n - x (n - 1)) 
  : ∀ n : ℕ, ¬ ∃ k : ℤ, x n = k * k := 
sorry

end no_perfect_squares_in_sequence_l46_46519


namespace no_solution_system_l46_46486

theorem no_solution_system :
  ¬ ∃ (x y z : ℝ), (3 * x - 4 * y + z = 10) ∧ (6 * x - 8 * y + 2 * z = 5) ∧ (2 * x - y - z = 4) :=
by {
  sorry
}

end no_solution_system_l46_46486


namespace teal_sales_l46_46070

theorem teal_sales
  (pumpkin_pie_slices : ℕ := 8)
  (custard_pie_slices : ℕ := 6)
  (pumpkin_pie_price : ℕ := 5)
  (custard_pie_price : ℕ := 6)
  (pumpkin_pies_sold : ℕ := 4)
  (custard_pies_sold : ℕ := 5) :
  let total_pumpkin_slices := pumpkin_pie_slices * pumpkin_pies_sold
  let total_custard_slices := custard_pie_slices * custard_pies_sold
  let total_pumpkin_sales := total_pumpkin_slices * pumpkin_pie_price
  let total_custard_sales := total_custard_slices * custard_pie_price
  let total_sales := total_pumpkin_sales + total_custard_sales
  total_sales = 340 :=
by
  sorry

end teal_sales_l46_46070


namespace assistant_professor_pencils_l46_46437

theorem assistant_professor_pencils :
  ∀ (A B P : ℕ), 
    A + B = 7 →
    2 * A + P * B = 10 →
    A + 2 * B = 11 →
    P = 1 :=
by 
  sorry

end assistant_professor_pencils_l46_46437


namespace square_area_l46_46760

theorem square_area :
  ∃ (s : ℝ), (8 * s - 2 = 30) ∧ (s ^ 2 = 16) :=
by
  sorry

end square_area_l46_46760


namespace smallest_palindrome_divisible_by_6_l46_46213

def is_palindrome (x : Nat) : Prop :=
  let d1 := x / 1000
  let d2 := (x / 100) % 10
  let d3 := (x / 10) % 10
  let d4 := x % 10
  d1 = d4 ∧ d2 = d3

def is_divisible_by (x n : Nat) : Prop :=
  x % n = 0

theorem smallest_palindrome_divisible_by_6 : ∃ n : Nat, is_palindrome n ∧ is_divisible_by n 6 ∧ 1000 ≤ n ∧ n < 10000 ∧ ∀ m : Nat, (is_palindrome m ∧ is_divisible_by m 6 ∧ 1000 ≤ m ∧ m < 10000) → n ≤ m := 
  by
    exists 2112
    sorry

end smallest_palindrome_divisible_by_6_l46_46213


namespace stickers_on_fifth_page_l46_46744

theorem stickers_on_fifth_page :
  ∀ (stickers : ℕ → ℕ),
    stickers 1 = 8 →
    stickers 2 = 16 →
    stickers 3 = 24 →
    stickers 4 = 32 →
    (∀ n, stickers (n + 1) = stickers n + 8) →
    stickers 5 = 40 :=
by
  intros stickers h1 h2 h3 h4 pattern
  apply sorry

end stickers_on_fifth_page_l46_46744


namespace sin_480_eq_sqrt3_div_2_l46_46246

theorem sin_480_eq_sqrt3_div_2 : Real.sin (480 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_480_eq_sqrt3_div_2_l46_46246


namespace breadth_of_added_rectangle_l46_46576

theorem breadth_of_added_rectangle 
  (s : ℝ) (b : ℝ) 
  (h_square_side : s = 8) 
  (h_perimeter_new_rectangle : 2 * s + 2 * (s + b) = 40) : 
  b = 4 :=
by
  sorry

end breadth_of_added_rectangle_l46_46576


namespace minimum_value_fraction_l46_46891

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem minimum_value_fraction (a : ℕ → ℝ) (m n : ℕ) (q : ℝ) (h_geometric : geometric_sequence a q)
  (h_positive : ∀ k : ℕ, 0 < a k)
  (h_condition1 : a 7 = a 6 + 2 * a 5)
  (h_condition2 : ∃ r, r ^ 2 = a m * a n ∧ r = 2 * a 1) :
  (1 / m + 9 / n) ≥ 4 :=
  sorry

end minimum_value_fraction_l46_46891


namespace number_of_cows_l46_46909

-- Define the total number of legs and number of legs per cow
def total_legs : ℕ := 460
def legs_per_cow : ℕ := 4

-- Mathematical proof problem as a Lean 4 statement
theorem number_of_cows : total_legs / legs_per_cow = 115 := by
  -- This is the proof statement place. We use 'sorry' as a placeholder for the actual proof.
  sorry

end number_of_cows_l46_46909


namespace compound_interest_is_correct_l46_46176

noncomputable def compound_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * (1 + R / 100)^T - P

noncomputable def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * R * T / 100

theorem compound_interest_is_correct :
  let P := 660 / (0.2 : ℝ)
  (compound_interest P 10 2) = 693 := 
by
  -- Definitions of simple_interest and compound_interest are used
  -- The problem conditions help us conclude
  let P := 660 / (0.2 : ℝ)
  have h1 : simple_interest P 10 2 = 660 := by sorry
  have h2 : compound_interest P 10 2 = 693 := by sorry
  exact h2

end compound_interest_is_correct_l46_46176


namespace LovelyCakeSlices_l46_46384

/-- Lovely cuts her birthday cake into some equal pieces.
    One-fourth of the cake was eaten by her visitors.
    Nine slices of cake were kept, representing three-fourths of the total number of slices.
    Prove: Lovely cut her birthday cake into 12 equal pieces. -/
theorem LovelyCakeSlices (totalSlices : ℕ) 
  (h1 : (3 / 4 : ℚ) * totalSlices = 9) : totalSlices = 12 := by
  sorry

end LovelyCakeSlices_l46_46384


namespace zoo_animal_count_l46_46298

def tiger_enclosures : ℕ := 4
def zebra_enclosures_per_tiger_enclosures : ℕ := 2
def zebra_enclosures : ℕ := tiger_enclosures * zebra_enclosures_per_tiger_enclosures
def giraffe_enclosures_per_zebra_enclosures : ℕ := 3
def giraffe_enclosures : ℕ := zebra_enclosures * giraffe_enclosures_per_zebra_enclosures
def tigers_per_enclosure : ℕ := 4
def zebras_per_enclosure : ℕ := 10
def giraffes_per_enclosure : ℕ := 2

def total_animals_in_zoo : ℕ := 
    (tiger_enclosures * tigers_per_enclosure) + 
    (zebra_enclosures * zebras_per_enclosure) + 
    (giraffe_enclosures * giraffes_per_enclosure)

theorem zoo_animal_count : total_animals_in_zoo = 144 := 
by
  -- proof would go here
  sorry

end zoo_animal_count_l46_46298


namespace min_tiles_to_cover_region_l46_46802

noncomputable def num_tiles_needed (tile_length tile_width region_length region_width : ℕ) : ℕ :=
  let tile_area := tile_length * tile_width
  let region_area := region_length * region_width
  region_area / tile_area

theorem min_tiles_to_cover_region : num_tiles_needed 6 2 36 72 = 216 :=
by 
  -- This is the format needed to include the assumptions and reach the conclusion
  sorry

end min_tiles_to_cover_region_l46_46802


namespace price_of_magic_card_deck_l46_46455

def initial_decks : ℕ := 5
def remaining_decks : ℕ := 3
def total_earnings : ℕ := 4
def decks_sold := initial_decks - remaining_decks
def price_per_deck := total_earnings / decks_sold

theorem price_of_magic_card_deck : price_per_deck = 2 := by
  sorry

end price_of_magic_card_deck_l46_46455


namespace angle_double_of_supplementary_l46_46229

theorem angle_double_of_supplementary (x : ℝ) (h1 : x + (180 - x) = 180) (h2 : x = 2 * (180 - x)) : x = 120 :=
sorry

end angle_double_of_supplementary_l46_46229


namespace range_of_a_l46_46336

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
if h : x < 1 then a * x^2 - 6 * x + a^2 + 1 else x^(5 - 2 * a)

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (5/2 < a ∧ a ≤ 3) :=
sorry

end range_of_a_l46_46336


namespace ab_bc_ca_negative_l46_46671

theorem ab_bc_ca_negative (a b c : ℝ) (h₁ : a + b + c = 0) (h₂ : abc > 0) : ab + bc + ca < 0 :=
sorry

end ab_bc_ca_negative_l46_46671


namespace test_question_count_l46_46741

def total_test_questions 
  (total_points : ℕ) 
  (points_per_2pt : ℕ) 
  (points_per_4pt : ℕ) 
  (num_2pt_questions : ℕ) 
  (num_4pt_questions : ℕ) : Prop :=
  total_points = points_per_2pt * num_2pt_questions + points_per_4pt * num_4pt_questions 

theorem test_question_count 
  (total_points : ℕ) 
  (points_per_2pt : ℕ) 
  (points_per_4pt : ℕ) 
  (num_2pt_questions : ℕ) 
  (correct_total_questions : ℕ) :
  total_test_questions total_points points_per_2pt points_per_4pt num_2pt_questions (correct_total_questions - num_2pt_questions) → correct_total_questions = 40 :=
by
  intros h
  sorry

end test_question_count_l46_46741


namespace marie_erasers_l46_46675

-- Define the initial conditions
def initial_erasers : ℝ := 95.0
def additional_erasers : ℝ := 42.0

-- Define the target final erasers count
def final_erasers : ℝ := 137.0

-- The theorem we need to prove
theorem marie_erasers :
  initial_erasers + additional_erasers = final_erasers := by
  sorry

end marie_erasers_l46_46675


namespace diorama_time_subtraction_l46_46284

theorem diorama_time_subtraction (P B X : ℕ) (h1 : B = 3 * P - X) (h2 : B = 49) (h3 : P + B = 67) : X = 5 :=
by
  sorry

end diorama_time_subtraction_l46_46284


namespace min_value_expression_l46_46855

theorem min_value_expression (a b : ℝ) (h : a > b) (h0 : b > 0) :
  ∃ m : ℝ, m = (a^2 + 1 / (a * b) + 1 / (a * (a - b))) ∧ m = 4 :=
sorry

end min_value_expression_l46_46855


namespace find_n_l46_46179

theorem find_n 
  (n : ℕ) 
  (b : ℕ → ℝ)
  (h₀ : b 0 = 28)
  (h₁ : b 1 = 81)
  (hn : b n = 0)
  (h_rec : ∀ j : ℕ, 1 ≤ j → j < n → b (j+1) = b (j-1) - 5 / b j)
  : n = 455 := 
sorry

end find_n_l46_46179


namespace total_pairs_sold_l46_46900

theorem total_pairs_sold
  (H S : ℕ)
  (price_soft : ℕ := 150)
  (price_hard : ℕ := 85)
  (diff_lenses : S = H + 5)
  (total_sales_eq : price_soft * S + price_hard * H = 1455) :
  H + S = 11 := by
sorry

end total_pairs_sold_l46_46900


namespace domain_of_function_l46_46428

theorem domain_of_function:
  {x : ℝ | x^2 - 5*x + 6 > 0 ∧ x ≠ 3} = {x : ℝ | x < 2 ∨ x > 3} :=
by
  sorry

end domain_of_function_l46_46428


namespace largest_n_for_factoring_l46_46817

theorem largest_n_for_factoring :
  ∃ (n : ℤ), 
    (∀ A B : ℤ, (5 * B + A = n ∧ A * B = 60) → (5 * B + A ≤ n)) ∧
    n = 301 :=
by sorry

end largest_n_for_factoring_l46_46817


namespace parabola_distance_ratio_l46_46946

open Real

theorem parabola_distance_ratio (p : ℝ) (M N : ℝ × ℝ)
  (h1 : p = 4)
  (h2 : M.snd ^ 2 = 2 * p * M.fst)
  (h3 : N.snd ^ 2 = 2 * p * N.fst)
  (h4 : (M.snd - 2 * N.snd) * (M.snd + 2 * N.snd) = 48) :
  |M.fst + 2| = 4 * |N.fst + 2| := sorry

end parabola_distance_ratio_l46_46946


namespace tim_total_trip_time_l46_46763

theorem tim_total_trip_time (drive_time : ℕ) (traffic_multiplier : ℕ) (drive_time_eq : drive_time = 5) (traffic_multiplier_eq : traffic_multiplier = 2) :
  drive_time + drive_time * traffic_multiplier = 15 :=
by
  sorry

end tim_total_trip_time_l46_46763


namespace tim_has_33_books_l46_46635

-- Define the conditions
def b := 24   -- Benny's initial books
def s := 10   -- Books given to Sandy
def total_books : Nat := 47  -- Total books

-- Define the remaining books after Benny gives to Sandy
def remaining_b : Nat := b - s

-- Define Tim's books
def tim_books : Nat := total_books - remaining_b

-- Prove that Tim has 33 books
theorem tim_has_33_books : tim_books = 33 := by
  -- This is a placeholder for the proof
  sorry

end tim_has_33_books_l46_46635


namespace michael_current_chickens_l46_46333

-- Defining variables and constants
variable (initial_chickens final_chickens annual_increase : ℕ)

-- Given conditions
def chicken_increase_condition : Prop :=
  final_chickens = initial_chickens + annual_increase * 9

-- Question to answer
def current_chickens (final_chickens annual_increase : ℕ) : ℕ :=
  final_chickens - annual_increase * 9

-- Proof problem
theorem michael_current_chickens
  (initial_chickens : ℕ)
  (final_chickens : ℕ)
  (annual_increase : ℕ)
  (h1 : chicken_increase_condition final_chickens initial_chickens annual_increase) :
  initial_chickens = 550 :=
by
  -- Formal proof would go here.
  sorry

end michael_current_chickens_l46_46333


namespace number_of_tiles_l46_46145

theorem number_of_tiles (floor_length : ℝ) (floor_width : ℝ) (tile_length : ℝ) (tile_width : ℝ) 
  (h1 : floor_length = 9) 
  (h2 : floor_width = 12) 
  (h3 : tile_length = 1 / 2) 
  (h4 : tile_width = 2 / 3) 
  : (floor_length * floor_width) / (tile_length * tile_width) = 324 := 
by
  sorry

end number_of_tiles_l46_46145


namespace eq1_solutions_eq2_solutions_l46_46950

theorem eq1_solutions (x : ℝ) : x^2 - 6 * x + 3 = 0 ↔ (x = 3 + Real.sqrt 6) ∨ (x = 3 - Real.sqrt 6) :=
by {
  sorry
}

theorem eq2_solutions (x : ℝ) : x * (x - 2) = x - 2 ↔ (x = 2) ∨ (x = 1) :=
by {
  sorry
}

end eq1_solutions_eq2_solutions_l46_46950


namespace john_children_l46_46526

def total_notebooks (john_notebooks : ℕ) (wife_notebooks : ℕ) (children : ℕ) := 
  2 * children + 5 * children

theorem john_children (c : ℕ) (h : total_notebooks 2 5 c = 21) :
  c = 3 :=
sorry

end john_children_l46_46526


namespace length_of_NC_l46_46481

noncomputable def semicircle_radius (AB : ℝ) : ℝ := AB / 2

theorem length_of_NC : 
  ∀ (AB CD AN NB N M C NC : ℝ),
    AB = 10 ∧ AB = CD ∧ AN = NB ∧ AN + NB = AB ∧ M = N ∧ AB / 2 = semicircle_radius AB ∧ (NC^2 + semicircle_radius AB^2 = (2 * semicircle_radius AB)^2) →
    NC = 5 * Real.sqrt 3 := 
by 
  intros AB CD AN NB N M C NC h 
  rcases h with ⟨hAB, hCD, hAN, hSumAN, hMN, hRadius, hPythag⟩
  sorry

end length_of_NC_l46_46481


namespace question1_question2_question3_l46_46944

variables {a x1 x2 : ℝ}

-- Definition of the quadratic equation
def quadratic_eq (a x : ℝ) : ℝ := a * x^2 + x + 1

-- Conditions
axiom a_positive : a > 0
axiom roots_exist : quadratic_eq a x1 = 0 ∧ quadratic_eq a x2 = 0
axiom roots_real : x1 + x2 = -1 / a ∧ x1 * x2 = 1 / a

-- Question 1
theorem question1 : (1 + x1) * (1 + x2) = 1 :=
sorry

-- Question 2
theorem question2 : x1 < -1 ∧ x2 < -1 :=
sorry

-- Additional condition for question 3
axiom ratio_in_range : x1 / x2 ∈ Set.Icc (1 / 10 : ℝ) 10

-- Question 3
theorem question3 : a <= 1 / 4 :=
sorry

end question1_question2_question3_l46_46944


namespace incorrect_statement_d_l46_46707

-- Definitions based on the problem's conditions
def is_acute (θ : ℝ) := 0 < θ ∧ θ < 90

def is_complementary (θ₁ θ₂ : ℝ) := θ₁ + θ₂ = 90

def is_supplementary (θ₁ θ₂ : ℝ) := θ₁ + θ₂ = 180

-- Statement D from the problem
def statement_d (θ : ℝ) := is_acute θ → ∀ θc, is_complementary θ θc → θ > θc

-- The theorem we want to prove
theorem incorrect_statement_d : ¬(∀ θ : ℝ, statement_d θ) := 
by sorry

end incorrect_statement_d_l46_46707


namespace solve_for_a_and_b_l46_46830
-- Import the necessary library

open Classical

variable (a b x : ℝ)

theorem solve_for_a_and_b (h1 : 0 ≤ x) (h2 : x < 1) (h3 : x + 2 * a ≥ 4) (h4 : (2 * x - b) / 3 < 1) : a + b = 1 := 
by
  sorry

end solve_for_a_and_b_l46_46830


namespace fraction_equality_l46_46341

theorem fraction_equality :
  (2 - (1 / 2) * (1 - (1 / 4))) / (2 - (1 - (1 / 3))) = 39 / 32 := 
  sorry

end fraction_equality_l46_46341


namespace first_rectangle_dimensions_second_rectangle_dimensions_l46_46046

theorem first_rectangle_dimensions (x y : ℕ) (h : x * y = 2 * (x + y) + 1) : (x = 7 ∧ y = 3) ∨ (x = 3 ∧ y = 7) :=
sorry

theorem second_rectangle_dimensions (a b : ℕ) (h : a * b = 2 * (a + b) - 1) : (a = 5 ∧ b = 3) ∨ (a = 3 ∧ b = 5) :=
sorry

end first_rectangle_dimensions_second_rectangle_dimensions_l46_46046


namespace ratio_of_james_to_jacob_l46_46646

noncomputable def MarkJumpHeight : ℕ := 6
noncomputable def LisaJumpHeight : ℕ := 2 * MarkJumpHeight
noncomputable def JacobJumpHeight : ℕ := 2 * LisaJumpHeight
noncomputable def JamesJumpHeight : ℕ := 16

theorem ratio_of_james_to_jacob : (JamesJumpHeight : ℚ) / (JacobJumpHeight : ℚ) = 2 / 3 :=
by
  sorry

end ratio_of_james_to_jacob_l46_46646


namespace simplify_expression_l46_46430

theorem simplify_expression (r : ℝ) (h1 : r^2 ≠ 0) (h2 : r^4 > 16) :
  ( ( ( (r^2 + 4) ^ (3 : ℝ) ) ^ (1 / 3 : ℝ) * (1 + 4 / r^2) ^ (1 / 2 : ℝ) ^ (1 / 3 : ℝ)
    - ( (r^2 - 4) ^ (3 : ℝ) ) ^ (1 / 3 : ℝ) * (1 - 4 / r^2) ^ (1 / 2 : ℝ) ^ (1 / 3 : ℝ) ) ^ 2 )
  / ( r^2 - (r^4 - 16) ^ (1 / 2 : ℝ) )
  = 2 * r ^ (-(2 / 3 : ℝ)) := by
  sorry

end simplify_expression_l46_46430


namespace present_value_of_machine_l46_46540

theorem present_value_of_machine (r : ℝ) (t : ℕ) (V : ℝ) (P : ℝ) (h1 : r = 0.10) (h2 : t = 2) (h3 : V = 891) :
  V = P * (1 - r)^t → P = 1100 :=
by
  intro h
  rw [h3, h1, h2] at h
  -- The steps to solve for P are omitted as instructed
  sorry

end present_value_of_machine_l46_46540


namespace number_of_possible_measures_l46_46414

theorem number_of_possible_measures (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : 
  ∃ n : ℕ, n = 17 :=
sorry

end number_of_possible_measures_l46_46414


namespace sum_of_repeating_decimals_l46_46136

-- Definitions of the repeating decimals as fractions
def x : ℚ := 1 / 9
def y : ℚ := 2 / 99
def z : ℚ := 3 / 999

-- Theorem stating the sum of these fractions is equal to the expected result
theorem sum_of_repeating_decimals : x + y + z = 164 / 1221 := 
  sorry

end sum_of_repeating_decimals_l46_46136


namespace fit_seven_rectangles_l46_46781

theorem fit_seven_rectangles (s : ℝ) (a : ℝ) : (s > 0) → (a > 0) → (14 * a ^ 2 ≤ s ^ 2 ∧ 2 * a ≤ s) → 
  (∃ (rectangles : Fin 7 → (ℝ × ℝ)), ∀ i, rectangles i = (a, 2 * a) ∧
   ∀ i j, i ≠ j → rectangles i ≠ rectangles j) :=
sorry

end fit_seven_rectangles_l46_46781


namespace range_of_m_l46_46309

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 1 / y = 1) : 
  ∀ m : ℝ, (x + 2 * y > m) ↔ (m < 8) :=
by 
  sorry

end range_of_m_l46_46309


namespace difference_between_hit_and_unreleased_l46_46953

-- Define the conditions as constants
def hit_songs : Nat := 25
def top_100_songs : Nat := hit_songs + 10
def total_songs : Nat := 80

-- Define the question, conditional on the definitions above
theorem difference_between_hit_and_unreleased : 
  let unreleased_songs := total_songs - (hit_songs + top_100_songs)
  hit_songs - unreleased_songs = 5 :=
by
  sorry

end difference_between_hit_and_unreleased_l46_46953


namespace find_radius_of_semicircular_plot_l46_46104

noncomputable def radius_of_semicircular_plot (π : ℝ) : ℝ :=
  let total_fence_length := 33
  let opening_length := 3
  let effective_fence_length := total_fence_length - opening_length
  let r := effective_fence_length / (π + 2)
  r

theorem find_radius_of_semicircular_plot 
  (π : ℝ) (Hπ : π = Real.pi) :
  radius_of_semicircular_plot π = 30 / (Real.pi + 2) :=
by
  unfold radius_of_semicircular_plot
  rw [Hπ]
  sorry

end find_radius_of_semicircular_plot_l46_46104


namespace area_of_defined_region_eq_14_point_4_l46_46555

def defined_region (x y : ℝ) : Prop :=
  |5 * x - 20| + |3 * y + 9| ≤ 6

def region_area : ℝ :=
  14.4

theorem area_of_defined_region_eq_14_point_4 :
  (∃ (x y : ℝ), defined_region x y) → region_area = 14.4 :=
by
  sorry

end area_of_defined_region_eq_14_point_4_l46_46555


namespace find_mistaken_number_l46_46343

theorem find_mistaken_number : 
  ∃! x : ℕ, (x ∈ {n : ℕ | n ≥ 10 ∧ n < 100 ∧ (n % 10 = 5 ∨ n % 10 = 0)} ∧ 
  (10 + 15 + 20 + 25 + 30 + 35 + 40 + 45 + 50 + 55 + 60 + 65 + 70 + 75 + 80 + 85 + 90 + 95) + 2 * x = 1035) :=
sorry

end find_mistaken_number_l46_46343


namespace total_weight_of_three_packages_l46_46286

theorem total_weight_of_three_packages (a b c d : ℝ)
  (h1 : a + b = 162)
  (h2 : b + c = 164)
  (h3 : c + a = 168) :
  a + b + c = 247 :=
sorry

end total_weight_of_three_packages_l46_46286


namespace percent_round_trip_tickets_l46_46523

variable (P : ℕ) -- total number of passengers

def passengers_with_round_trip_tickets (P : ℕ) : ℕ :=
  2 * (P / 5 / 2)

theorem percent_round_trip_tickets (P : ℕ) : 
  passengers_with_round_trip_tickets P = 2 * (P / 5 / 2) :=
by
  sorry

end percent_round_trip_tickets_l46_46523


namespace difference_largest_smallest_l46_46074

def num1 : ℕ := 10
def num2 : ℕ := 11
def num3 : ℕ := 12

theorem difference_largest_smallest :
  (max num1 (max num2 num3)) - (min num1 (min num2 num3)) = 2 :=
by
  -- Proof can be filled here
  sorry

end difference_largest_smallest_l46_46074


namespace sum_adjacent_angles_pentagon_l46_46224

theorem sum_adjacent_angles_pentagon (n : ℕ) (θ : ℕ) (hn : n = 5) (hθ : θ = 40) :
  let exterior_angle := 360 / n
  let new_adjacent_angle := 180 - (exterior_angle + θ)
  let sum_adjacent_angles := n * new_adjacent_angle
  sum_adjacent_angles = 340 := by
  sorry

end sum_adjacent_angles_pentagon_l46_46224


namespace sum_problem3_equals_50_l46_46305

-- Assume problem3_condition is a placeholder for the actual conditions described in problem 3
-- and sum_problem3 is a placeholder for the sum of elements described in problem 3.

axiom problem3_condition : Prop
axiom sum_problem3 : ℕ

theorem sum_problem3_equals_50 (h : problem3_condition) : sum_problem3 = 50 :=
sorry

end sum_problem3_equals_50_l46_46305


namespace max_marks_400_l46_46516

theorem max_marks_400 {M : ℝ} (h : 0.45 * M = 150 + 30) : M = 400 := 
by
  sorry

end max_marks_400_l46_46516


namespace average_of_last_four_numbers_l46_46370

theorem average_of_last_four_numbers
  (seven_avg : ℝ)
  (first_three_avg : ℝ)
  (seven_avg_is_62 : seven_avg = 62)
  (first_three_avg_is_58 : first_three_avg = 58) :
  (7 * seven_avg - 3 * first_three_avg) / 4 = 65 :=
by
  rw [seven_avg_is_62, first_three_avg_is_58]
  sorry

end average_of_last_four_numbers_l46_46370


namespace quadratic_inequality_solution_l46_46809

theorem quadratic_inequality_solution (z : ℝ) :
  z^2 - 40 * z + 340 ≤ 4 ↔ 12 ≤ z ∧ z ≤ 28 := by 
  sorry

end quadratic_inequality_solution_l46_46809


namespace team_combination_count_l46_46529

theorem team_combination_count (n k : ℕ) (hn : n = 7) (hk : k = 4) :
  ∃ m, m = Nat.choose n k ∧ m = 35 :=
by
  sorry

end team_combination_count_l46_46529


namespace percent_increase_correct_l46_46656

variable (p_initial p_final : ℝ)

theorem percent_increase_correct : p_initial = 25 → p_final = 28 → (p_final - p_initial) / p_initial * 100 = 12 := by
  intros h_initial h_final
  sorry

end percent_increase_correct_l46_46656


namespace count_ordered_triples_l46_46912

def S := Finset.range 20

def succ (a b : ℕ) : Prop := 
  (0 < a - b ∧ a - b ≤ 10) ∨ (b - a > 10)

theorem count_ordered_triples 
  (h : ∃ n : ℕ, (S.card = 20) ∧
                (∀ x y z : ℕ, 
                   x ∈ S → y ∈ S → z ∈ S →
                   (succ x y) → (succ y z) → (succ z x) →
                   n = 1260)) : True := sorry

end count_ordered_triples_l46_46912


namespace rowing_time_ratio_l46_46663

def V_b : ℕ := 57
def V_s : ℕ := 19
def V_up : ℕ := V_b - V_s
def V_down : ℕ := V_b + V_s

theorem rowing_time_ratio :
  ∀ (T_up T_down : ℕ), V_up * T_up = V_down * T_down → T_up = 2 * T_down :=
by
  intros T_up T_down h
  sorry

end rowing_time_ratio_l46_46663


namespace fraction_addition_l46_46040

theorem fraction_addition :
  (3/8 : ℚ) / (4/9 : ℚ) + 1/6 = 97/96 := by
  sorry

end fraction_addition_l46_46040


namespace y1_y2_positive_l46_46304

theorem y1_y2_positive 
  (x1 x2 x3 : ℝ)
  (y1 y2 y3 : ℝ)
  (h_line1 : y1 = -2 * x1 + 3)
  (h_line2 : y2 = -2 * x2 + 3)
  (h_line3 : y3 = -2 * x3 + 3)
  (h_order : x1 < x2 ∧ x2 < x3)
  (h_product_neg : x2 * x3 < 0) :
  y1 * y2 > 0 :=
by
  sorry

end y1_y2_positive_l46_46304


namespace trig_expression_evaluation_l46_46313

theorem trig_expression_evaluation (θ : ℝ) (h : Real.tan θ = 2) :
  Real.sin θ ^ 2 + (Real.sin θ * Real.cos θ) - 2 * (Real.cos θ ^ 2) = 4 / 5 := 
by
  sorry

end trig_expression_evaluation_l46_46313


namespace trigonometric_identity_l46_46733

theorem trigonometric_identity (α : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : Real.tan α = -2) :
  2 * (Real.sin α)^2 - (Real.sin α) * (Real.cos α) + (Real.cos α)^2 = 11 / 5 := by
  sorry

end trigonometric_identity_l46_46733


namespace soccer_field_kids_l46_46738

def a := 14
def b := 22
def c := a + b

theorem soccer_field_kids : c = 36 :=
by
    sorry

end soccer_field_kids_l46_46738


namespace iron_balls_molded_l46_46424

-- Define the dimensions of the iron bar
def length_bar : ℝ := 12
def width_bar : ℝ := 8
def height_bar : ℝ := 6

-- Define the volume calculations
def volume_iron_bar : ℝ := length_bar * width_bar * height_bar
def number_of_bars : ℝ := 10
def total_volume_bars : ℝ := volume_iron_bar * number_of_bars
def volume_iron_ball : ℝ := 8

-- Define the goal statement
theorem iron_balls_molded : total_volume_bars / volume_iron_ball = 720 :=
by
  -- Proof is to be filled in here
  sorry

end iron_balls_molded_l46_46424


namespace at_least_one_fraction_less_than_two_l46_46811

theorem at_least_one_fraction_less_than_two {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := 
by
  sorry

end at_least_one_fraction_less_than_two_l46_46811


namespace exists_f_ff_eq_square_l46_46659

open Nat

theorem exists_f_ff_eq_square : ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n ^ 2 :=
by
  -- proof to be provided
  sorry

end exists_f_ff_eq_square_l46_46659


namespace positive_integer_solutions_l46_46299

theorem positive_integer_solutions : 
  (∀ x : ℤ, ((1 + 2 * (x:ℝ)) / 4 - (1 - 3 * (x:ℝ)) / 10 > -1 / 5) ∧ (3 * (x:ℝ) - 1 < 2 * ((x:ℝ) + 1)) → (x = 1 ∨ x = 2)) :=
by 
  sorry

end positive_integer_solutions_l46_46299


namespace probability_not_red_is_two_thirds_l46_46376

-- Given conditions as definitions
def number_of_orange_marbles : ℕ := 4
def number_of_purple_marbles : ℕ := 7
def number_of_red_marbles : ℕ := 8
def number_of_yellow_marbles : ℕ := 5

-- Define the total number of marbles
def total_marbles : ℕ :=
  number_of_orange_marbles + 
  number_of_purple_marbles + 
  number_of_red_marbles + 
  number_of_yellow_marbles

def number_of_non_red_marbles : ℕ :=
  number_of_orange_marbles + 
  number_of_purple_marbles + 
  number_of_yellow_marbles

-- Define the probability
def probability_not_red : ℚ :=
  number_of_non_red_marbles / total_marbles

-- The theorem that states the probability of not picking a red marble is 2/3
theorem probability_not_red_is_two_thirds :
  probability_not_red = 2 / 3 :=
by
  sorry

end probability_not_red_is_two_thirds_l46_46376


namespace find_x0_l46_46258

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

theorem find_x0 (x0 : ℝ) (h : f' x0 = 2) : x0 = Real.exp 1 :=
by {
  sorry
}

end find_x0_l46_46258


namespace total_cost_l46_46853

-- Define the conditions
def dozen := 12
def cost_of_dozen_cupcakes := 10
def cost_of_dozen_cookies := 8
def cost_of_dozen_brownies := 12

def num_dozen_cupcakes := 4
def num_dozen_cookies := 3
def num_dozen_brownies := 2

-- Define the total cost for each type of treat
def total_cost_cupcakes := num_dozen_cupcakes * cost_of_dozen_cupcakes
def total_cost_cookies := num_dozen_cookies * cost_of_dozen_cookies
def total_cost_brownies := num_dozen_brownies * cost_of_dozen_brownies

-- The theorem to prove the total cost
theorem total_cost : total_cost_cupcakes + total_cost_cookies + total_cost_brownies = 88 := by
  -- Here would go the proof, but it's omitted as per the instructions
  sorry

end total_cost_l46_46853


namespace john_increased_bench_press_factor_l46_46232

theorem john_increased_bench_press_factor (initial current : ℝ) (decrease_percent : ℝ) 
  (h_initial : initial = 500) 
  (h_current : current = 300) 
  (h_decrease : decrease_percent = 0.80) : 
  current / (initial * (1 - decrease_percent)) = 3 := 
by
  -- We'll provide the proof here later
  sorry

end john_increased_bench_press_factor_l46_46232


namespace lara_has_largest_answer_l46_46147

/-- Define the final result for John, given his operations --/
def final_john (n : ℕ) : ℕ :=
  let add_three := n + 3
  let double := add_three * 2
  double - 4

/-- Define the final result for Lara, given her operations --/
def final_lara (n : ℕ) : ℕ :=
  let triple := n * 3
  let add_five := triple + 5
  add_five - 6

/-- Define the final result for Miguel, given his operations --/
def final_miguel (n : ℕ) : ℕ :=
  let double := n * 2
  let subtract_two := double - 2
  subtract_two + 2

/-- Main theorem to be proven --/
theorem lara_has_largest_answer :
  final_lara 12 > final_john 12 ∧ final_lara 12 > final_miguel 12 :=
by {
  sorry
}

end lara_has_largest_answer_l46_46147


namespace num_solution_pairs_l46_46420

theorem num_solution_pairs (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  4 * x + 7 * y = 600 → ∃ n : ℕ, n = 21 :=
by
  sorry

end num_solution_pairs_l46_46420


namespace sum_S5_l46_46353

-- Geometric sequence definitions and conditions
noncomputable def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^n

noncomputable def sum_of_geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

variables (a r : ℝ)

-- Given conditions translated into Lean:
-- a2 * a3 = 2 * a1
def condition1 := (geometric_sequence a r 1) * (geometric_sequence a r 2) = 2 * a

-- Arithmetic mean of a4 and 2 * a7 is 5/4
def condition2 := (geometric_sequence a r 3 + 2 * geometric_sequence a r 6) / 2 = 5 / 4

-- The final goal proving that S5 = 31
theorem sum_S5 (h1 : condition1 a r) (h2 : condition2 a r) : sum_of_geometric_sequence a r 5 = 31 := by
  apply sorry

end sum_S5_l46_46353


namespace minimum_value_of_y_l46_46608

theorem minimum_value_of_y (x : ℝ) (h : x > 2) : 
  ∃ y, y = x + 4 / (x - 2) ∧ y = 6 :=
by
  sorry

end minimum_value_of_y_l46_46608


namespace inequality_proof_l46_46586

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  1 ≤ ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ∧ 
  ((x + y) * (x^3 + y^3)) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end inequality_proof_l46_46586


namespace evaluate_expression_l46_46274

def a : ℚ := 7/3

theorem evaluate_expression :
  (4 * a^2 - 11 * a + 7) * (2 * a - 3) = 140 / 27 :=
by
  sorry

end evaluate_expression_l46_46274


namespace Henry_has_four_Skittles_l46_46315

-- Defining the initial amount of Skittles Bridget has
def Bridget_initial := 4

-- Defining the final amount of Skittles Bridget has after receiving all of Henry's Skittles
def Bridget_final := 8

-- Defining the amount of Skittles Henry has
def Henry_Skittles := Bridget_final - Bridget_initial

-- The proof statement to be proven
theorem Henry_has_four_Skittles : Henry_Skittles = 4 := by
  sorry

end Henry_has_four_Skittles_l46_46315


namespace sum_of_cubes_eq_twice_product_of_roots_l46_46354

theorem sum_of_cubes_eq_twice_product_of_roots (m : ℝ) :
  (∃ a b : ℝ, (3*a^2 + 6*a + m = 0) ∧ (3*b^2 + 6*b + m = 0) ∧ (a ≠ b)) → 
  (a^3 + b^3 = 2 * a * b) → 
  m = 6 :=
by
  intros h_exists sum_eq_twice_product
  sorry

end sum_of_cubes_eq_twice_product_of_roots_l46_46354


namespace circumscribed_sphere_radius_l46_46735

noncomputable def radius_of_circumscribed_sphere (a : ℝ) : ℝ :=
  a * (Real.sqrt (6 + Real.sqrt 20)) / 8

theorem circumscribed_sphere_radius (a : ℝ) :
  radius_of_circumscribed_sphere a = a * (Real.sqrt (6 + Real.sqrt 20)) / 8 :=
by
  sorry

end circumscribed_sphere_radius_l46_46735


namespace midpoint_product_l46_46289

theorem midpoint_product (x y z : ℤ) 
  (h1 : (2 + x) / 2 = 4) 
  (h2 : (10 + y) / 2 = 6) 
  (h3 : (5 + z) / 2 = 3) : 
  x * y * z = 12 := 
by
  sorry

end midpoint_product_l46_46289


namespace other_root_of_quadratic_l46_46433

theorem other_root_of_quadratic (k : ℝ) :
  (∃ x : ℝ, 3 * x^2 + k * x - 5 = 0 ∧ x = 3) →
  ∃ r : ℝ, 3 * r * 3 = -5 / 3 ∧ r = -5 / 9 :=
by
  sorry

end other_root_of_quadratic_l46_46433


namespace last_three_digits_W_555_2_l46_46028

noncomputable def W : ℕ → ℕ → ℕ
| n, 0 => n ^ n
| n, (k + 1) => W (W n k) k

theorem last_three_digits_W_555_2 : (W 555 2) % 1000 = 375 := 
by
  sorry

end last_three_digits_W_555_2_l46_46028


namespace maximum_elephants_l46_46536

theorem maximum_elephants (e_1 e_2 : ℕ) :
  (∃ e_1 e_2 : ℕ, 28 * e_1 + 37 * e_2 = 1036 ∧ (∀ k, 28 * e_1 + 37 * e_2 = k → k ≤ 1036 )) → 
  28 * e_1 + 37 * e_2 = 1036 :=
sorry

end maximum_elephants_l46_46536


namespace sequence_form_l46_46400

theorem sequence_form (c : ℕ) (a : ℕ → ℕ) :
  (∀ n : ℕ, 0 < n →
    (∃! i : ℕ, 0 < i ∧ a i ≤ a (n + 1) + c)) ↔
  (∀ n : ℕ, 0 < n → a n = n + (c + 1)) :=
by
  sorry

end sequence_form_l46_46400


namespace cream_ratio_l46_46941

theorem cream_ratio (john_coffee_initial jane_coffee_initial : ℕ)
  (john_drank john_added_cream jane_added_cream jane_drank : ℕ) :
  john_coffee_initial = 20 →
  jane_coffee_initial = 20 →
  john_drank = 3 →
  john_added_cream = 4 →
  jane_added_cream = 3 →
  jane_drank = 5 →
  john_added_cream / (jane_added_cream * 18 / (23 * 1)) = (46 / 27) := 
by
  intros
  sorry

end cream_ratio_l46_46941


namespace circle_a_center_radius_circle_b_center_radius_circle_c_center_radius_l46_46629

-- Part (a): Prove the center and radius for the given circle equation: (x-3)^2 + (y+2)^2 = 16
theorem circle_a_center_radius :
  (∃ (a b : ℤ) (R : ℕ), (∀ (x y : ℝ), (x - 3) ^ 2 + (y + 2) ^ 2 = 16 ↔ (x - a) ^ 2 + (y - b) ^ 2 = R^2) ∧ a = 3 ∧ b = -2 ∧ R = 4) :=
by {
  sorry
}

-- Part (b): Prove the center and radius for the given circle equation: x^2 + y^2 - 2(x - 3y) - 15 = 0
theorem circle_b_center_radius :
  (∃ (a b : ℤ) (R : ℕ), (∀ (x y : ℝ), x^2 + y^2 - 2 * (x - 3 * y) - 15 = 0 ↔ (x - a) ^ 2 + (y - b) ^ 2 = R^2) ∧ a = 1 ∧ b = -3 ∧ R = 5) :=
by {
  sorry
}

-- Part (c): Prove the center and radius for the given circle equation: x^2 + y^2 = x + y + 1/2
theorem circle_c_center_radius :
  (∃ (a b : ℚ) (R : ℚ), (∀ (x y : ℚ), x^2 + y^2 = x + y + 1/2 ↔ (x - a) ^ 2 + (y - b) ^ 2 = R^2) ∧ a = 1/2 ∧ b = 1/2 ∧ R = 1) :=
by {
  sorry
}

end circle_a_center_radius_circle_b_center_radius_circle_c_center_radius_l46_46629


namespace multiplication_digit_sum_l46_46574

theorem multiplication_digit_sum :
  let a := 879
  let b := 492
  let product := a * b
  let sum_of_digits := (4 + 3 + 2 + 4 + 6 + 8)
  product = 432468 ∧ sum_of_digits = 27 := by
  -- Step 1: Set up the given numbers
  let a := 879
  let b := 492

  -- Step 2: Calculate the product
  let product := a * b
  have product_eq : product = 432468 := by
    sorry

  -- Step 3: Sum the digits of the product
  let sum_of_digits := (4 + 3 + 2 + 4 + 6 + 8)
  have sum_of_digits_eq : sum_of_digits = 27 := by
    sorry

  -- Conclusion
  exact ⟨product_eq, sum_of_digits_eq⟩

end multiplication_digit_sum_l46_46574


namespace find_number_l46_46647

-- Definitions based on conditions
def sum : ℕ := 555 + 445
def difference : ℕ := 555 - 445
def quotient : ℕ := 2 * difference
def remainder : ℕ := 70
def divisor : ℕ := sum

-- Statement to be proved
theorem find_number : (divisor * quotient + remainder) = 220070 := by
  sorry

end find_number_l46_46647


namespace line_intersects_x_axis_at_point_l46_46906

theorem line_intersects_x_axis_at_point :
  (∃ x, 5 * 0 - 2 * x = 10) ↔ (x = -5) ∧ (∃ x, 5 * y - 2 * x = 10 ∧ y = 0) :=
by
  sorry

end line_intersects_x_axis_at_point_l46_46906


namespace range_of_k_l46_46681

theorem range_of_k (k : ℝ) (h : k ≠ 0) : (k^2 - 6 * k + 8 ≥ 0) ↔ (k ≥ 4 ∨ k ≤ 2) := 
by sorry

end range_of_k_l46_46681


namespace contrapositive_a_eq_b_imp_a_sq_eq_b_sq_l46_46887

theorem contrapositive_a_eq_b_imp_a_sq_eq_b_sq (a b : ℝ) :
  (a = b → a^2 = b^2) ↔ (a^2 ≠ b^2 → a ≠ b) :=
by
  sorry

end contrapositive_a_eq_b_imp_a_sq_eq_b_sq_l46_46887


namespace original_price_l46_46933

theorem original_price (P S : ℝ) (h1 : S = 1.25 * P) (h2 : S - P = 625) : P = 2500 := by
  sorry

end original_price_l46_46933


namespace perimeter_difference_l46_46569

-- Define the height of the screen
def height_of_screen : ℕ := 100

-- Define the side length of the square paper
def side_of_square_paper : ℕ := 20

-- Define the perimeter of the square paper
def perimeter_of_paper : ℕ := 4 * side_of_square_paper

-- Prove the difference between the height of the screen and the perimeter of the paper
theorem perimeter_difference : height_of_screen - perimeter_of_paper = 20 := by
  -- Sorry is used here to skip the actual proof
  sorry

end perimeter_difference_l46_46569


namespace isosceles_triangles_l46_46630

theorem isosceles_triangles (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h_triangle : ∀ n : ℕ, (a^n + b^n > c^n ∧ b^n + c^n > a^n ∧ c^n + a^n > b^n)) :
  b = c := 
sorry

end isosceles_triangles_l46_46630


namespace factor_expression_l46_46058

theorem factor_expression (x : ℝ) :
  84 * x ^ 5 - 210 * x ^ 9 = -42 * x ^ 5 * (5 * x ^ 4 - 2) :=
by
  sorry

end factor_expression_l46_46058


namespace quadratic_inequality_solution_l46_46445

theorem quadratic_inequality_solution {a : ℝ} :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ a < -1 ∨ a > 3 :=
by sorry

end quadratic_inequality_solution_l46_46445


namespace find_vertical_shift_l46_46291

theorem find_vertical_shift (A B C D : ℝ) (h1 : ∀ x, -3 ≤ A * Real.cos (B * x + C) + D ∧ A * Real.cos (B * x + C) + D ≤ 5) :
  D = 1 :=
by
  -- Here's where the proof would go
  sorry

end find_vertical_shift_l46_46291


namespace functional_eq_1996_l46_46575

def f (x : ℝ) : ℝ := sorry

theorem functional_eq_1996 (f : ℝ → ℝ)
    (h : ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * ((f x)^2 - (f x) * (f y) + (f y)^2)) :
    ∀ x : ℝ, f (1996 * x) = 1996 * f x := 
sorry

end functional_eq_1996_l46_46575


namespace divisor_probability_of_25_factorial_is_odd_and_multiple_of_5_l46_46588

theorem divisor_probability_of_25_factorial_is_odd_and_multiple_of_5 :
  let prime_factors_25 := 2^22 * 3^10 * 5^6 * 7^3 * 11^2 * 13^1 * 17^1 * 19^1 * 23^1
  let total_divisors := (22+1) * (10+1) * (6+1) * (3+1) * (2+1) * (1+1) * (1+1) * (1+1)
  let odd_and_multiple_of_5_divisors := (6+1) * (3+1) * (2+1) * (1+1) * (1+1)
  (odd_and_multiple_of_5_divisors / total_divisors : ℚ) = 7 / 23 := 
sorry

end divisor_probability_of_25_factorial_is_odd_and_multiple_of_5_l46_46588


namespace smallest_w_factor_l46_46863

theorem smallest_w_factor (w : ℕ) (hw : w > 0) :
  (∃ w, 2^4 ∣ 1452 * w ∧ 3^3 ∣ 1452 * w ∧ 13^3 ∣ 1452 * w) ↔ w = 79092 :=
by sorry

end smallest_w_factor_l46_46863


namespace find_x_l46_46177

namespace IntegerProblem

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 17 := 
by
  sorry

end IntegerProblem

end find_x_l46_46177


namespace find_angle_C_range_of_a_plus_b_l46_46207

variables {A B C a b c : ℝ}

-- Define the conditions
def conditions (a b c : ℝ) (A B C : ℝ) : Prop :=
  (a + c) * (Real.sin A - Real.sin C) = Real.sin B * (a - b)

-- Proof problem 1: show angle C is π/3
theorem find_angle_C (ha : 0 < A) (hb : 0 < B) (hc : 0 < C)
  (hab : A + B + C = π) (h : conditions a b c A B C) : 
  C = π / 3 :=
sorry

-- Proof problem 2: if c = 2, then show the range of a + b
theorem range_of_a_plus_b (ha : 0 < A) (hb : 0 < B) (hc : 0 < C)
  (hab : A + B + C = π) (h : conditions a b 2 A B C) :
  2 < a + b ∧ a + b ≤ 4 :=
sorry

end find_angle_C_range_of_a_plus_b_l46_46207


namespace calc_result_neg2xy2_pow3_l46_46497

theorem calc_result_neg2xy2_pow3 (x y : ℝ) : 
  (-2 * x * y^2)^3 = -8 * x^3 * y^6 := 
by 
  sorry

end calc_result_neg2xy2_pow3_l46_46497


namespace smallest_sum_ab_l46_46758

theorem smallest_sum_ab (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 2^10 * 3^6 = a^b) : a + b = 866 :=
sorry

end smallest_sum_ab_l46_46758


namespace medicine_dosage_l46_46502

theorem medicine_dosage (weight_kg dose_per_kg parts : ℕ) (h_weight : weight_kg = 30) (h_dose_per_kg : dose_per_kg = 5) (h_parts : parts = 3) :
  ((weight_kg * dose_per_kg) / parts) = 50 :=
by sorry

end medicine_dosage_l46_46502


namespace tan_neg405_deg_l46_46390

theorem tan_neg405_deg : Real.tan (-405 * Real.pi / 180) = -1 := by
  -- This is a placeholder for the actual proof
  sorry

end tan_neg405_deg_l46_46390


namespace proof_problem_l46_46220

variables (p q : Prop)

theorem proof_problem (hpq : p ∨ q) (hnp : ¬p) : q :=
by
  sorry

end proof_problem_l46_46220


namespace modulus_of_z_l46_46090

section complex_modulus
open Complex

theorem modulus_of_z (z : ℂ) (h : z * (2 + I) = 10 - 5 * I) : Complex.abs z = 5 :=
by
  sorry
end complex_modulus

end modulus_of_z_l46_46090


namespace add_and_subtract_l46_46048

theorem add_and_subtract (a b c : ℝ) (h1 : a = 0.45) (h2 : b = 52.7) (h3 : c = 0.25) : 
  (a + b) - c = 52.9 :=
by 
  sorry

end add_and_subtract_l46_46048


namespace car_speed_problem_l46_46549

theorem car_speed_problem (x : ℝ) (h1 : ∀ x, x + 30 / 2 = 65) : x = 100 :=
by
  sorry

end car_speed_problem_l46_46549


namespace find_smaller_root_l46_46339

theorem find_smaller_root :
  ∀ x : ℝ, (x - 2 / 3) ^ 2 + (x - 2 / 3) * (x - 1 / 3) = 0 → x = 1 / 2 :=
by
  sorry

end find_smaller_root_l46_46339


namespace xiaobo_probability_not_home_l46_46805

theorem xiaobo_probability_not_home :
  let r1 := 1 / 2
  let r2 := 1 / 4
  let area_circle := Real.pi
  let area_greater_r1 := area_circle * (1 - r1^2)
  let area_less_r2 := area_circle * r2^2
  let area_favorable := area_greater_r1 + area_less_r2
  let probability_not_home := area_favorable / area_circle
  probability_not_home = 13 / 16 := by
  sorry

end xiaobo_probability_not_home_l46_46805


namespace correct_calculation_is_7_88_l46_46375

theorem correct_calculation_is_7_88 (x : ℝ) (h : x * 8 = 56) : (x / 8) + 7 = 7.88 :=
by
  have hx : x = 7 := by
    linarith [h]
  rw [hx]
  norm_num
  sorry

end correct_calculation_is_7_88_l46_46375


namespace problem_conditions_l46_46910

noncomputable def f (x : ℝ) : ℝ := (2 * x - x^2) * Real.exp x

theorem problem_conditions :
  (∀ x, f x > 0 ↔ 0 < x ∧ x < 2) ∧
  (∃ x_max, x_max = Real.sqrt 2 ∧ (∀ y, f y ≤ f x_max)) ∧
  ¬(∃ x_min, ∀ y, f x_min ≤ f y) :=
by sorry

end problem_conditions_l46_46910


namespace shelves_of_picture_books_l46_46979

theorem shelves_of_picture_books
   (total_books : ℕ)
   (books_per_shelf : ℕ)
   (mystery_shelves : ℕ)
   (mystery_books : ℕ)
   (total_mystery_books : mystery_books = mystery_shelves * books_per_shelf)
   (total_books_condition : total_books = 32)
   (mystery_books_condition : mystery_books = 5 * books_per_shelf) :
   (total_books - mystery_books) / books_per_shelf = 3 :=
by
  sorry

end shelves_of_picture_books_l46_46979


namespace least_blue_eyes_and_snack_l46_46620

variable (total_students blue_eyes students_with_snack : ℕ)

theorem least_blue_eyes_and_snack (h1 : total_students = 35) 
                                 (h2 : blue_eyes = 14) 
                                 (h3 : students_with_snack = 22) :
  ∃ n, n = 1 ∧ 
        ∀ k, (k < n → 
                 ∃ no_snack_no_blue : ℕ, no_snack_no_blue = total_students - students_with_snack ∧
                      no_snack_no_blue = blue_eyes - k) := 
by
  sorry

end least_blue_eyes_and_snack_l46_46620


namespace sky_falls_distance_l46_46062

def distance_from_city (x : ℕ) (y : ℕ) : Prop := 50 * x = y

theorem sky_falls_distance :
    ∃ D_s : ℕ, distance_from_city D_s 400 ∧ D_s = 8 :=
by
  sorry

end sky_falls_distance_l46_46062


namespace equation_of_circle_l46_46024

theorem equation_of_circle :
  ∃ (a : ℝ), a < 0 ∧ (∀ (x y : ℝ), (x + 2 * y = 0) → (x + 5)^2 + y^2 = 5) :=
by
  sorry

end equation_of_circle_l46_46024


namespace number_of_nickels_is_three_l46_46591

def coin_problem : Prop :=
  ∃ p n d q : ℕ,
    p + n + d + q = 12 ∧
    p + 5 * n + 10 * d + 25 * q = 128 ∧
    p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1 ∧
    q = 2 * d ∧
    n = 3

theorem number_of_nickels_is_three : coin_problem := 
by 
  sorry

end number_of_nickels_is_three_l46_46591


namespace measure_of_angle_ABC_l46_46492

-- Define the angles involved and their respective measures
def angle_CBD : ℝ := 90 -- Given that angle CBD is a right angle
def angle_sum : ℝ := 160 -- Sum of the angles around point B
def angle_ABD : ℝ := 50 -- Given angle ABD

-- Define angle ABC to be determined
def angle_ABC : ℝ := angle_sum - (angle_ABD + angle_CBD)

-- Define the statement
theorem measure_of_angle_ABC :
  angle_ABC = 20 :=
by 
  -- Calculations omitted
  sorry

end measure_of_angle_ABC_l46_46492


namespace arrangement_count_l46_46786

-- Given conditions
def num_basketballs : ℕ := 5
def num_volleyballs : ℕ := 3
def num_footballs : ℕ := 2
def total_balls : ℕ := num_basketballs + num_volleyballs + num_footballs

-- Way to calculate the permutations of multiset
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Proof statement
theorem arrangement_count : 
  factorial total_balls / (factorial num_basketballs * factorial num_volleyballs * factorial num_footballs) = 2520 :=
by
  sorry

end arrangement_count_l46_46786


namespace evaluate_expression_l46_46920

theorem evaluate_expression (a b c : ℚ) 
  (h1 : c = b - 11) 
  (h2 : b = a + 3) 
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0) 
  (h5 : b - 3 ≠ 0) 
  (h6 : c + 7 ≠ 0) : 
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 9) / (c + 7) = 10 / 7 := 
sorry

end evaluate_expression_l46_46920


namespace magician_act_reappearance_l46_46889

-- Defining the conditions as given in the problem
def total_performances : ℕ := 100

def no_one_reappears (perf : ℕ) : ℕ := perf / 10
def two_reappear (perf : ℕ) : ℕ := perf / 5
def one_reappears (perf : ℕ) : ℕ := perf - no_one_reappears perf - two_reappear perf
def total_reappeared (perf : ℕ) : ℕ := one_reappears perf + 2 * two_reappear perf

-- The statement to be proved
theorem magician_act_reappearance : total_reappeared total_performances = 110 := by
  sorry

end magician_act_reappearance_l46_46889


namespace third_restaurant_meals_per_day_l46_46410

-- Define the daily meals served by the first two restaurants
def meals_first_restaurant_per_day : ℕ := 20
def meals_second_restaurant_per_day : ℕ := 40

-- Define the total meals served by all three restaurants per week
def total_meals_per_week : ℕ := 770

-- Define the weekly meals served by the first two restaurants
def meals_first_restaurant_per_week : ℕ := meals_first_restaurant_per_day * 7
def meals_second_restaurant_per_week : ℕ := meals_second_restaurant_per_day * 7

-- Total weekly meals served by the first two restaurants
def total_meals_first_two_restaurants_per_week : ℕ := meals_first_restaurant_per_week + meals_second_restaurant_per_week

-- Weekly meals served by the third restaurant
def meals_third_restaurant_per_week : ℕ := total_meals_per_week - total_meals_first_two_restaurants_per_week

-- Convert weekly meals served by the third restaurant to daily meals
def meals_third_restaurant_per_day : ℕ := meals_third_restaurant_per_week / 7

-- Goal: Prove the third restaurant serves 50 meals per day
theorem third_restaurant_meals_per_day : meals_third_restaurant_per_day = 50 := by
  -- proof skipped
  sorry

end third_restaurant_meals_per_day_l46_46410


namespace combined_weight_of_candles_l46_46959

theorem combined_weight_of_candles (candles : ℕ) (weight_per_candle : ℕ) (total_weight : ℕ) :
  candles = 10 - 3 →
  weight_per_candle = 8 + 1 →
  total_weight = candles * weight_per_candle →
  total_weight = 63 :=
by
  intros
  subst_vars
  sorry

end combined_weight_of_candles_l46_46959


namespace customer_savings_l46_46915

variables (P : ℝ) (reducedPrice negotiatedPrice savings : ℝ)

-- Conditions:
def initialReduction : reducedPrice = 0.95 * P := by sorry
def finalNegotiation : negotiatedPrice = 0.90 * reducedPrice := by sorry
def savingsCalculation : savings = P - negotiatedPrice := by sorry

-- Proof problem:
theorem customer_savings : savings = 0.145 * P :=
by {
  sorry
}

end customer_savings_l46_46915


namespace farmer_field_area_l46_46842

theorem farmer_field_area (m : ℝ) (h : (3 * m + 5) * (m + 1) = 104) : m = 4.56 :=
sorry

end farmer_field_area_l46_46842


namespace initial_volume_salt_solution_l46_46649

theorem initial_volume_salt_solution (V : ℝ) (V1 : ℝ) (V2 : ℝ) : 
  V1 = 0.20 * V → 
  V2 = 30 →
  V1 = 0.15 * (V + V2) →
  V = 90 := 
by 
  sorry

end initial_volume_salt_solution_l46_46649


namespace find_k_l46_46931

open Real

def vector := ℝ × ℝ

def dot_product (v1 v2 : vector) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

def orthogonal (v1 v2 : vector) : Prop := dot_product v1 v2 = 0

theorem find_k (k : ℝ) :
  let a : vector := (2, 3)
  let b : vector := (1, 4)
  let c : vector := (k, 3)
  orthogonal (a.1 + b.1, a.2 + b.2) c → k = -7 :=
by
  intros
  sorry

end find_k_l46_46931


namespace jenny_jellybeans_original_l46_46804

theorem jenny_jellybeans_original (x : ℝ) 
  (h : 0.75^3 * x = 45) : x = 107 := 
sorry

end jenny_jellybeans_original_l46_46804


namespace Owen_final_turtle_count_l46_46451

variable (Owen_turtles : ℕ) (Johanna_turtles : ℕ)

def final_turtles (Owen_turtles Johanna_turtles : ℕ) : ℕ :=
  let initial_Owen_turtles := Owen_turtles
  let initial_Johanna_turtles := Owen_turtles - 5
  let Owen_after_month := initial_Owen_turtles * 2
  let Johanna_after_losing_half := initial_Johanna_turtles / 2
  let Owen_after_donation := Owen_after_month + Johanna_after_losing_half
  Owen_after_donation

theorem Owen_final_turtle_count : final_turtles 21 (21 - 5) = 50 :=
by
  sorry

end Owen_final_turtle_count_l46_46451


namespace min_product_of_prime_triplet_l46_46546

theorem min_product_of_prime_triplet
  (x y z : ℕ) (hx : Prime x) (hy : Prime y) (hz : Prime z)
  (hx_odd : x % 2 = 1) (hy_odd : y % 2 = 1) (hz_odd : z % 2 = 1)
  (h1 : x ∣ (y^5 + 1)) (h2 : y ∣ (z^5 + 1)) (h3 : z ∣ (x^5 + 1)) :
  (x * y * z) = 2013 := by
  sorry

end min_product_of_prime_triplet_l46_46546


namespace part1_part2_l46_46372

def f (x m : ℝ) : ℝ := |x - 1| - |2 * x + m|

theorem part1 (x : ℝ) (m : ℝ) (h : m = -4) : 
    f x m < 0 ↔ x < 5 / 3 ∨ x > 3 := 
by 
  sorry

theorem part2 (x : ℝ) (h : 1 < x) (h' : ∀ x, 1 < x → f x m < 0) : 
    m ≥ -2 :=
by 
  sorry

end part1_part2_l46_46372


namespace find_x_l46_46413

theorem find_x (x : ℝ) :
  (x^2 - 7 * x + 12) / (x^2 - 9 * x + 20) = (x^2 - 4 * x - 21) / (x^2 - 5 * x - 24) -> x = 11 :=
by
  sorry

end find_x_l46_46413


namespace petya_prevents_vasya_l46_46226

-- Define the nature of fractions and the players' turns
def is_natural_sum (fractions : List ℚ) : Prop :=
  (fractions.sum = ⌊fractions.sum⌋)

def petya_vasya_game_prevent (fractions : List ℚ) : Prop :=
  ∀ k : ℕ, ∀ additional_fractions : List ℚ, 
  (additional_fractions.length = k) →
  ¬ is_natural_sum (fractions ++ additional_fractions)

theorem petya_prevents_vasya : ∀ fractions : List ℚ, petya_vasya_game_prevent fractions :=
by
  sorry

end petya_prevents_vasya_l46_46226


namespace number_of_balls_to_remove_l46_46474

theorem number_of_balls_to_remove:
  ∀ (x : ℕ), 120 - x = (48 : ℕ) / (0.75 : ℝ) → x = 56 :=
by sorry

end number_of_balls_to_remove_l46_46474


namespace donuts_count_is_correct_l46_46255

-- Define the initial number of donuts
def initial_donuts : ℕ := 50

-- Define the number of donuts Bill eats
def eaten_by_bill : ℕ := 2

-- Define the number of donuts taken by the secretary
def taken_by_secretary : ℕ := 4

-- Calculate the remaining donuts after Bill and the secretary take their portions
def remaining_after_bill_and_secretary : ℕ := initial_donuts - eaten_by_bill - taken_by_secretary

-- Define the number of donuts stolen by coworkers (half of the remaining donuts)
def stolen_by_coworkers : ℕ := remaining_after_bill_and_secretary / 2

-- Define the number of donuts left for the meeting
def donuts_left_for_meeting : ℕ := remaining_after_bill_and_secretary - stolen_by_coworkers

-- The theorem to prove
theorem donuts_count_is_correct : donuts_left_for_meeting = 22 :=
by
  sorry

end donuts_count_is_correct_l46_46255


namespace integer_solutions_eq_l46_46779

theorem integer_solutions_eq :
  { (x, y) : ℤ × ℤ | 2 * x ^ 4 - 4 * y ^ 4 - 7 * x ^ 2 * y ^ 2 - 27 * x ^ 2 + 63 * y ^ 2 + 85 = 0 }
  = { (3, 1), (3, -1), (-3, 1), (-3, -1), (2, 3), (2, -3), (-2, 3), (-2, -3) } :=
by sorry

end integer_solutions_eq_l46_46779


namespace find_k_l46_46969

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b (k : ℝ) : ℝ × ℝ := (2 * k, 3)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
noncomputable def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)

theorem find_k : ∃ k : ℝ, dot_product a (vector_add (scalar_mult 2 a) (b k)) = 0 ∧ k = -8 :=
by
  sorry

end find_k_l46_46969


namespace result_of_operation_given_y_l46_46981

def operation (a b : ℤ) : ℤ := (a - 1) * (b - 1)

theorem result_of_operation_given_y :
  ∀ (y : ℤ), y = 11 → operation y 10 = 90 :=
by
  intros y hy
  rw [hy]
  show operation 11 10 = 90
  sorry

end result_of_operation_given_y_l46_46981


namespace max_sum_two_digit_primes_l46_46223

theorem max_sum_two_digit_primes : (89 + 97) = 186 := 
by
  sorry

end max_sum_two_digit_primes_l46_46223


namespace percentage_of_loss_is_25_l46_46334

-- Definitions from conditions
def CP : ℝ := 2800
def SP : ℝ := 2100

-- Proof statement
theorem percentage_of_loss_is_25 : ((CP - SP) / CP) * 100 = 25 := by
  sorry

end percentage_of_loss_is_25_l46_46334


namespace find_a_b_c_eq_32_l46_46856

variables {a b c : ℤ}

theorem find_a_b_c_eq_32
  (h1 : ∃ a b : ℤ, x^2 + 19 * x + 88 = (x + a) * (x + b))
  (h2 : ∃ b c : ℤ, x^2 - 21 * x + 108 = (x - b) * (x - c)) :
  a + b + c = 32 :=
sorry

end find_a_b_c_eq_32_l46_46856


namespace domain_range_g_l46_46319

variable (f : ℝ → ℝ) 

noncomputable def g (x : ℝ) := 2 - f (x + 1)

theorem domain_range_g :
  (∀ x, 0 ≤ x → x ≤ 3 → 0 ≤ f x → f x ≤ 1) →
  (∀ x, -1 ≤ x → x ≤ 2) ∧ (∀ y, 1 ≤ y → y ≤ 2) :=
sorry

end domain_range_g_l46_46319


namespace family_raised_percentage_l46_46117

theorem family_raised_percentage :
  ∀ (total_funds friends_percentage own_savings family_funds remaining_funds : ℝ),
    total_funds = 10000 →
    friends_percentage = 0.40 →
    own_savings = 4200 →
    remaining_funds = total_funds - (friends_percentage * total_funds) →
    family_funds = remaining_funds - own_savings →
    (family_funds / remaining_funds) * 100 = 30 :=
by
  intros total_funds friends_percentage own_savings family_funds remaining_funds
  intros h_total_funds h_friends_percentage h_own_savings h_remaining_funds h_family_funds
  sorry

end family_raised_percentage_l46_46117


namespace option_a_option_d_l46_46975

theorem option_a (n m : ℕ) (h1 : 1 ≤ n) (h2 : 1 ≤ m) (h3 : n > m) : 
  Nat.choose n m = Nat.choose n (n - m) := 
sorry

theorem option_d (n m : ℕ) (h1 : 1 ≤ n) (h2 : 1 ≤ m) (h3 : n > m) : 
  Nat.choose n m + Nat.choose n (m - 1) = Nat.choose (n + 1) m := 
sorry

end option_a_option_d_l46_46975


namespace M_eq_N_l46_46689

def M : Set ℝ := {x | ∃ k : ℤ, x = (7/6) * Real.pi + 2 * k * Real.pi ∨ x = (5/6) * Real.pi + 2 * k * Real.pi}
def N : Set ℝ := {x | ∃ k : ℤ, x = (7/6) * Real.pi + 2 * k * Real.pi ∨ x = -(7/6) * Real.pi + 2 * k * Real.pi}

theorem M_eq_N : M = N := 
by {
  sorry
}

end M_eq_N_l46_46689


namespace correct_operation_l46_46221

variables (a b : ℝ)

theorem correct_operation : (3 * a + b) * (3 * a - b) = 9 * a^2 - b^2 :=
by sorry

end correct_operation_l46_46221


namespace payback_time_l46_46803

theorem payback_time (initial_cost monthly_revenue monthly_expenses : ℕ) 
  (h_initial_cost : initial_cost = 25000) 
  (h_monthly_revenue : monthly_revenue = 4000)
  (h_monthly_expenses : monthly_expenses = 1500) :
  ∃ n : ℕ, n = initial_cost / (monthly_revenue - monthly_expenses) ∧ n = 10 :=
by
  sorry

end payback_time_l46_46803


namespace cost_of_large_poster_is_correct_l46_46894

/-- Problem conditions -/
def posters_per_day : ℕ := 5
def large_posters_per_day : ℕ := 2
def large_poster_sale_price : ℝ := 10
def small_posters_per_day : ℕ := 3
def small_poster_sale_price : ℝ := 6
def small_poster_cost : ℝ := 3
def weekly_profit : ℝ := 95

/-- The cost to make a large poster -/
noncomputable def large_poster_cost : ℝ := 5

/-- Prove that the cost to make a large poster is $5 given the conditions -/
theorem cost_of_large_poster_is_correct :
    large_poster_cost = 5 :=
by
  -- (Condition translation into Lean)
  let daily_profit := weekly_profit / 5
  let daily_revenue := (large_posters_per_day * large_poster_sale_price) + (small_posters_per_day * small_poster_sale_price)
  let daily_cost_small_posters := small_posters_per_day * small_poster_cost
  
  -- Express the daily profit in terms of costs, including unknown large_poster_cost
  have calc_profit : daily_profit = daily_revenue - daily_cost_small_posters - (large_posters_per_day * (large_poster_cost)) :=
    sorry
  
  -- Setting the equation to solve for large_poster_cost
  have eqn : daily_profit = 19 := by
    sorry

  -- Solve for large_poster_cost
  have solve_large_poster_cost : 19 = daily_revenue - daily_cost_small_posters - (large_posters_per_day * 5) :=
    by sorry
  
  sorry

end cost_of_large_poster_is_correct_l46_46894


namespace smallest_mult_to_cube_l46_46301

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_mult_to_cube (n : ℕ) (h : ∃ n, ∃ k, n * y = k^3) : n = 4500 := 
  sorry

end smallest_mult_to_cube_l46_46301


namespace intersection_with_x_axis_l46_46995

noncomputable def f (x : ℝ) : ℝ := 
  (3 * x - 1) * (Real.sqrt (9 * x^2 - 6 * x + 5) + 1) + 
  (2 * x - 3) * (Real.sqrt (4 * x^2 - 12 * x + 13)) + 1

theorem intersection_with_x_axis :
  ∃ x : ℝ, f x = 0 ∧ x = 4 / 5 :=
by
  sorry

end intersection_with_x_axis_l46_46995


namespace exists_travel_route_l46_46464

theorem exists_travel_route (n : ℕ) (cities : Finset ℕ) 
  (ticket_price : ℕ → ℕ → ℕ)
  (h1 : cities.card = n)
  (h2 : ∀ c1 c2, c1 ≠ c2 → ∃ p, (ticket_price c1 c2 = p ∧ ticket_price c1 c2 = ticket_price c2 c1))
  (h3 : ∀ p1 p2 c1 c2 c3 c4,
    p1 ≠ p2 ∧ (ticket_price c1 c2 = p1) ∧ (ticket_price c3 c4 = p2) →
    p1 ≠ p2) :
  ∃ city : ℕ, ∀ m : ℕ, m = n - 1 →
  ∃ route : Finset (ℕ × ℕ),
  route.card = m ∧
  ∀ (t₁ t₂ : ℕ × ℕ), t₁ ∈ route → t₂ ∈ route → (t₁ ≠ t₂ → ticket_price t₁.1 t₁.2 < ticket_price t₂.1 t₂.2) :=
by
  sorry

end exists_travel_route_l46_46464


namespace multiply_add_distribute_l46_46762

theorem multiply_add_distribute :
  42 * 25 + 58 * 42 = 3486 := by
  sorry

end multiply_add_distribute_l46_46762


namespace mix_solutions_l46_46631

theorem mix_solutions {x : ℝ} (h : 0.60 * x + 0.75 * (20 - x) = 0.72 * 20) : x = 4 :=
by
-- skipping the proof with sorry
sorry

end mix_solutions_l46_46631


namespace madeline_part_time_hours_l46_46557

theorem madeline_part_time_hours :
  let hours_in_class := 18
  let days_in_week := 7
  let hours_homework_per_day := 4
  let hours_sleeping_per_day := 8
  let leftover_hours := 46
  let hours_per_day := 24
  let total_hours_per_week := hours_per_day * days_in_week
  let total_homework_hours := hours_homework_per_day * days_in_week
  let total_sleeping_hours := hours_sleeping_per_day * days_in_week
  let total_other_activities := hours_in_class + total_homework_hours + total_sleeping_hours
  let available_hours := total_hours_per_week - total_other_activities
  available_hours - leftover_hours = 20 := by
  sorry

end madeline_part_time_hours_l46_46557


namespace abs_eq_neg_of_nonpos_l46_46419

theorem abs_eq_neg_of_nonpos (a : ℝ) (h : |a| = -a) : a ≤ 0 :=
by
  have ha : |a| ≥ 0 := abs_nonneg a
  rw [h] at ha
  exact neg_nonneg.mp ha

end abs_eq_neg_of_nonpos_l46_46419


namespace therapy_charge_l46_46937

-- Define the charges
def first_hour_charge (S : ℝ) : ℝ := S + 50
def subsequent_hour_charge (S : ℝ) : ℝ := S

-- Define the total charge before service fee for 8 hours
def total_charge_8_hours_before_fee (F S : ℝ) : ℝ := F + 7 * S

-- Define the total charge including the service fee for 8 hours
def total_charge_8_hours (F S : ℝ) : ℝ := 1.10 * (F + 7 * S)

-- Define the total charge before service fee for 3 hours
def total_charge_3_hours_before_fee (F S : ℝ) : ℝ := F + 2 * S

-- Define the total charge including the service fee for 3 hours
def total_charge_3_hours (F S : ℝ) : ℝ := 1.10 * (F + 2 * S)

theorem therapy_charge (S F : ℝ) :
  (F = S + 50) → (1.10 * (F + 7 * S) = 900) → (1.10 * (F + 2 * S) = 371.87) :=
by {
  sorry
}

end therapy_charge_l46_46937


namespace tangerine_initial_count_l46_46462

theorem tangerine_initial_count 
  (X : ℕ) 
  (h1 : X - 9 + 5 = 20) : 
  X = 24 :=
sorry

end tangerine_initial_count_l46_46462


namespace enrollment_difference_l46_46932

theorem enrollment_difference 
  (Varsity_enrollment : ℕ)
  (Northwest_enrollment : ℕ)
  (Central_enrollment : ℕ)
  (Greenbriar_enrollment : ℕ) 
  (h1 : Varsity_enrollment = 1300) 
  (h2 : Northwest_enrollment = 1500)
  (h3 : Central_enrollment = 1800)
  (h4 : Greenbriar_enrollment = 1600) : 
  Varsity_enrollment < Northwest_enrollment ∧ 
  Northwest_enrollment < Greenbriar_enrollment ∧ 
  Greenbriar_enrollment < Central_enrollment → 
    (Greenbriar_enrollment - Varsity_enrollment = 300) :=
by
  sorry

end enrollment_difference_l46_46932


namespace distinct_non_zero_real_numbers_l46_46766

theorem distinct_non_zero_real_numbers (
  a b c : ℝ
) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) (h4 : a ≠ 0) (h5 : b ≠ 0) (h6 : c ≠ 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ax^2 + 2 * b * x1 + c = 0 ∧ ax^2 + 2 * b * x2 + c = 0) 
  ∨ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ bx^2 + 2 * c * x1 + a = 0 ∧ bx^2 + 2 * c * x2 + a = 0)
  ∨ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ cx^2 + 2 * a * x1 + b = 0 ∧ cx^2 + 2 * a * x2 + b = 0) :=
sorry

end distinct_non_zero_real_numbers_l46_46766


namespace base_7_units_digit_l46_46833

theorem base_7_units_digit : ((156 + 97) % 7) = 1 := 
by
  sorry

end base_7_units_digit_l46_46833


namespace intersection_A_B_union_B_C_eq_B_iff_l46_46134

-- Definitions for the sets A, B, and C
def setA : Set ℝ := { x | x^2 - 3 * x < 0 }
def setB : Set ℝ := { x | (x + 2) * (4 - x) ≥ 0 }
def setC (a : ℝ) : Set ℝ := { x | a < x ∧ x ≤ a + 1 }

-- Proving that A ∩ B = { x | 0 < x < 3 }
theorem intersection_A_B : setA ∩ setB = { x : ℝ | 0 < x ∧ x < 3 } :=
sorry

-- Proving that B ∪ C = B implies the range of a is [-2, 3]
theorem union_B_C_eq_B_iff (a : ℝ) : (setB ∪ setC a = setB) ↔ (-2 ≤ a ∧ a ≤ 3) :=
sorry

end intersection_A_B_union_B_C_eq_B_iff_l46_46134


namespace eval_expr_l46_46052

theorem eval_expr (x y : ℕ) (h1 : x = 2) (h2 : y = 1) : 2 * x - 3 * y = 1 := by
  sorry

end eval_expr_l46_46052


namespace total_miles_walked_by_group_in_6_days_l46_46027

-- Conditions translated to Lean definitions
def miles_per_day_group := 3
def additional_miles_per_day := 2
def days_in_week := 6
def total_ladies := 5

-- Question translated to a Lean theorem statement
theorem total_miles_walked_by_group_in_6_days : 
  ∀ (miles_per_day_group additional_miles_per_day days_in_week total_ladies : ℕ),
  (miles_per_day_group * total_ladies * days_in_week) + 
  ((miles_per_day_group * (total_ladies - 1) * days_in_week) + (additional_miles_per_day * days_in_week)) = 120 := 
by
  intros
  sorry

end total_miles_walked_by_group_in_6_days_l46_46027


namespace range_of_m_l46_46326

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) (h_even : ∀ x, f x = f (-x)) 
 (h_decreasing : ∀ {x y}, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x)
 (h_condition : ∀ x, 1 ≤ x → x ≤ 3 → f (2 * m * x - Real.log x - 3) ≥ 2 * f 3 - f (Real.log x + 3 - 2 * m * x)) :
  m ∈ Set.Icc (1 / (2 * Real.exp 1)) ((Real.log 3 + 6) / 6) :=
sorry

end range_of_m_l46_46326


namespace polynomial_p0_l46_46800

theorem polynomial_p0 :
  ∃ p : ℕ → ℚ, (∀ n : ℕ, n ≤ 6 → p (3^n) = 1 / (3^n)) ∧ (p 0 = 1093) :=
by
  sorry

end polynomial_p0_l46_46800


namespace time_until_meeting_l46_46578

theorem time_until_meeting (v1 v2 : ℝ) (t2 t1 : ℝ) 
    (h1 : v1 = 6) 
    (h2 : v2 = 4) 
    (h3 : t2 = 10)
    (h4 : v2 * t1 = v1 * (t1 - t2)) : t1 = 30 := 
sorry

end time_until_meeting_l46_46578


namespace smallest_possible_perimeter_l46_46798

theorem smallest_possible_perimeter (a : ℕ) (h : a > 2) (h_triangle : a < a + (a + 1) ∧ a + (a + 2) > (a + 1) ∧ (a + 1) + (a + 2) > a) :
  3 * a + 3 = 12 :=
by
  sorry

end smallest_possible_perimeter_l46_46798


namespace total_cost_is_eight_times_short_cost_l46_46829

variables (x : ℝ)
def cost_shorts := x
def cost_shirt := x
def cost_boots := 4 * x
def cost_shin_guards := 2 * x

theorem total_cost_is_eight_times_short_cost
    (cond_shirt : cost_shorts x + cost_shirt x = 2*x)
    (cond_boots : cost_shorts x + cost_boots x = 5*x)
    (cond_shin_guards : cost_shorts x + cost_shin_guards x = 3*x) :
    cost_shorts x + cost_shirt x + cost_boots x + cost_shin_guards x = 8*x :=
by
  sorry

end total_cost_is_eight_times_short_cost_l46_46829


namespace horizontal_shift_equivalence_l46_46940

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (x - Real.pi / 6)
noncomputable def resulting_function (x : ℝ) : ℝ := Real.sin (x + Real.pi / 6)

theorem horizontal_shift_equivalence :
  ∀ x : ℝ, resulting_function x = original_function (x + Real.pi / 3) :=
by sorry

end horizontal_shift_equivalence_l46_46940


namespace license_plates_count_l46_46204

/--
Define the conditions and constants.
-/
def num_letters := 26
def num_first_digit := 5  -- Odd digits
def num_second_digit := 5 -- Even digits

theorem license_plates_count : num_letters ^ 3 * num_first_digit * num_second_digit = 439400 := by
  sorry

end license_plates_count_l46_46204


namespace num_rectangles_in_5x5_grid_l46_46173

open Classical

noncomputable def num_rectangles_grid_5x5 : Nat := 
  Nat.choose 5 2 * Nat.choose 5 2

theorem num_rectangles_in_5x5_grid : num_rectangles_grid_5x5 = 100 :=
by
  sorry

end num_rectangles_in_5x5_grid_l46_46173


namespace mean_of_counts_is_7_l46_46593

theorem mean_of_counts_is_7 (counts : List ℕ) (h : counts = [6, 12, 1, 12, 7, 3, 8]) :
  counts.sum / counts.length = 7 :=
by
  sorry

end mean_of_counts_is_7_l46_46593


namespace binom_n_2_l46_46079

theorem binom_n_2 (n : ℕ) (h : 1 ≤ n) : Nat.choose n 2 = (n * (n - 1)) / 2 :=
by sorry

end binom_n_2_l46_46079


namespace n_not_both_perfect_squares_l46_46051

open Int

theorem n_not_both_perfect_squares (n x y : ℤ) (h1 : n > 0) :
  ¬ ((n + 1 = x^2) ∧ (4 * n + 1 = y^2)) :=
by {
  -- Problem restated in Lean, proof not required
  sorry
}

end n_not_both_perfect_squares_l46_46051


namespace find_x_value_l46_46151

theorem find_x_value :
  ∀ (x : ℝ), 0.3 + 0.1 + 0.4 + x = 1 → x = 0.2 :=
by
  intros x h
  sorry

end find_x_value_l46_46151


namespace positive_number_satisfying_condition_l46_46599

theorem positive_number_satisfying_condition :
  ∃ x : ℝ, x > 0 ∧ x^2 = 64 ∧ x = 8 := by sorry

end positive_number_satisfying_condition_l46_46599


namespace speed_of_stream_l46_46924

theorem speed_of_stream (v : ℝ) (d : ℝ) :
  (∀ d : ℝ, d > 0 → (1 / (6 - v) = 2 * (1 / (6 + v)))) → v = 2 := by
  sorry

end speed_of_stream_l46_46924


namespace brother_age_in_5_years_l46_46676

noncomputable def Nick : ℕ := 13
noncomputable def Sister : ℕ := Nick + 6
noncomputable def CombinedAge : ℕ := Nick + Sister
noncomputable def Brother : ℕ := CombinedAge / 2

theorem brother_age_in_5_years : Brother + 5 = 21 := by
  sorry

end brother_age_in_5_years_l46_46676


namespace total_time_spent_in_hours_l46_46144

/-- Miriam's time spent on each task in minutes. -/
def time_laundry := 30
def time_bathroom := 15
def time_room := 35
def time_homework := 40

/-- The function to convert minutes to hours. -/
def minutes_to_hours (minutes : ℕ) := minutes / 60

/-- The total time spent in minutes. -/
def total_time_minutes := time_laundry + time_bathroom + time_room + time_homework

/-- The total time spent in hours. -/
def total_time_hours := minutes_to_hours total_time_minutes

/-- The main statement to be proved: total_time_hours equals 2. -/
theorem total_time_spent_in_hours : total_time_hours = 2 := 
by
  sorry

end total_time_spent_in_hours_l46_46144


namespace no_t_for_xyz_equal_l46_46695

theorem no_t_for_xyz_equal (t : ℝ) (x y z : ℝ) : 
  (x = 1 - 3 * t) → 
  (y = 2 * t - 3) → 
  (z = 4 * t^2 - 5 * t + 1) → 
  ¬ (x = y ∧ y = z) := 
by
  intro h1 h2 h3 h4
  have h5 : t = 4 / 5 := 
    by linarith [h1, h2, h4]
  rw [h5] at h3
  sorry

end no_t_for_xyz_equal_l46_46695


namespace solution_of_inequality_answer_A_incorrect_answer_B_incorrect_answer_C_incorrect_D_is_correct_l46_46436

theorem solution_of_inequality (a b x : ℝ) :
    (b - a * x > 0) ↔
    (a > 0 ∧ x < b / a ∨ 
     a < 0 ∧ x > b / a ∨ 
     a = 0 ∧ false) :=
by sorry

-- Additional theorems to rule out incorrect answers
theorem answer_A_incorrect (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x > |b| / |a|) → false :=
by sorry

theorem answer_B_incorrect (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x < |b| / |a|) → false :=
by sorry

theorem answer_C_incorrect (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x > -|b| / |a|) → false :=
by sorry

theorem D_is_correct (a b : ℝ) :
    (∀ x : ℝ, b - a * x > 0 → x > |b| / |a| ∨ x < |b| / |a| ∨ x > -|b| / |a|) → false :=
by sorry

end solution_of_inequality_answer_A_incorrect_answer_B_incorrect_answer_C_incorrect_D_is_correct_l46_46436


namespace conic_section_focus_l46_46374

theorem conic_section_focus {m : ℝ} (h_non_zero : m ≠ 0) (h_non_five : m ≠ 5)
  (h_focus : ∃ (x_focus y_focus : ℝ), (x_focus, y_focus) = (2, 0) 
  ∧ (x_focus = c ∧ x_focus^2 / 4 = 5 * (1 - c^2 / m))) : m = 9 := 
by
  sorry

end conic_section_focus_l46_46374


namespace arithmetic_to_geometric_seq_l46_46439

theorem arithmetic_to_geometric_seq
  (d a : ℕ) 
  (h1 : d ≠ 0) 
  (a_n : ℕ → ℕ)
  (h2 : ∀ n, a_n n = a + (n - 1) * d)
  (h3 : (a + 2 * d) * (a + 2 * d) = a * (a + 8 * d))
  : (a_n 2 + a_n 4 + a_n 10) / (a_n 1 + a_n 3 + a_n 9) = 16 / 13 :=
by
  sorry

end arithmetic_to_geometric_seq_l46_46439


namespace first_grade_children_count_l46_46742

theorem first_grade_children_count (a : ℕ) (R L : ℕ) :
  200 ≤ a ∧ a ≤ 300 ∧ a = 25 * R + 10 ∧ a = 30 * L - 15 ∧ (R > 0 ∧ L > 0) → a = 285 :=
by
  sorry

end first_grade_children_count_l46_46742


namespace lcm_9_14_l46_46242

/-- Given the definition of the least common multiple (LCM) and the prime factorizations,
    prove that the LCM of 9 and 14 is 126. -/
theorem lcm_9_14 : Int.lcm 9 14 = 126 := by
  sorry

end lcm_9_14_l46_46242


namespace solve_equation_l46_46697

theorem solve_equation (x : ℝ) (h_eq : 1 / (x - 2) = 3 / (x - 5)) : 
  x = 1 / 2 :=
  sorry

end solve_equation_l46_46697


namespace roots_greater_than_one_implies_s_greater_than_zero_l46_46601

theorem roots_greater_than_one_implies_s_greater_than_zero
  (b c : ℝ)
  (h : ∃ α β : ℝ, α > 0 ∧ β > 0 ∧ (1 + α) + (1 + β) = -b ∧ (1 + α) * (1 + β) = c) :
  b + c + 1 > 0 :=
sorry

end roots_greater_than_one_implies_s_greater_than_zero_l46_46601


namespace polynomial_divisibility_l46_46880

theorem polynomial_divisibility (m : ℕ) (odd_m : m % 2 = 1) (x y z : ℤ) :
    ∃ k : ℤ, (x + y + z)^m - x^m - y^m - z^m = k * ((x + y + z)^3 - x^3 - y^3 - z^3) := 
by 
  sorry

end polynomial_divisibility_l46_46880


namespace complex_expression_l46_46949

theorem complex_expression (z : ℂ) (h : z = (i + 1) / (i - 1)) : z^2 + z + 1 = -i := 
by 
  sorry

end complex_expression_l46_46949


namespace maple_trees_cut_down_l46_46685

-- Define the initial number of maple trees.
def initial_maple_trees : ℝ := 9.0

-- Define the final number of maple trees after cutting.
def final_maple_trees : ℝ := 7.0

-- Define the number of maple trees cut down.
def cut_down_maple_trees : ℝ := initial_maple_trees - final_maple_trees

-- Prove that the number of cut down maple trees is 2.
theorem maple_trees_cut_down : cut_down_maple_trees = 2 := by
  sorry

end maple_trees_cut_down_l46_46685


namespace product_of_four_consecutive_naturals_is_square_l46_46330

theorem product_of_four_consecutive_naturals_is_square (n : ℕ) : 
  (n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2) := 
by
  sorry

end product_of_four_consecutive_naturals_is_square_l46_46330


namespace drawing_probability_consecutive_order_l46_46736

theorem drawing_probability_consecutive_order :
  let total_ways := Nat.factorial 12
  let desired_ways := (1 * Nat.factorial 3 * Nat.factorial 5)
  let probability := desired_ways / total_ways
  probability = 1 / 665280 :=
by
  let total_ways := Nat.factorial 12
  let desired_ways := (1 * Nat.factorial 3 * Nat.factorial 5)
  let probability := desired_ways / total_ways
  sorry

end drawing_probability_consecutive_order_l46_46736


namespace percentage_below_cost_l46_46963

variable (CP SP : ℝ)

-- Given conditions
def cost_price : ℝ := 5625
def more_for_profit : ℝ := 1800
def profit_percentage : ℝ := 0.16
def expected_SP : ℝ := cost_price + (cost_price * profit_percentage)
def actual_SP : ℝ := expected_SP - more_for_profit

-- Statement to prove
theorem percentage_below_cost (h1 : CP = cost_price) (h2 : SP = actual_SP) :
  (CP - SP) / CP * 100 = 16 := by
sorry

end percentage_below_cost_l46_46963


namespace zoe_total_songs_l46_46783

-- Define the number of country albums Zoe bought
def country_albums : Nat := 3

-- Define the number of pop albums Zoe bought
def pop_albums : Nat := 5

-- Define the number of songs per album
def songs_per_album : Nat := 3

-- Define the total number of albums
def total_albums : Nat := country_albums + pop_albums

-- Define the total number of songs
def total_songs : Nat := total_albums * songs_per_album

-- Theorem statement asserting the total number of songs
theorem zoe_total_songs : total_songs = 24 := by
  -- Proof will be inserted here (currently skipped)
  sorry

end zoe_total_songs_l46_46783


namespace problem_1_problem_2_l46_46624

theorem problem_1 (P_A P_B P_notA P_notB : ℚ) (hA: P_A = 1/2) (hB: P_B = 2/5) (hNotA: P_notA = 1/2) (hNotB: P_notB = 3/5) : 
  P_A * P_notB + P_B * P_notA = 1/2 := 
by 
  rw [hA, hB, hNotA, hNotB]
  -- exact calculations here
  sorry

theorem problem_2 (P_A P_B : ℚ) (hA: P_A = 1/2) (hB: P_B = 2/5) :
  (1 - (P_A * P_A * (1 - P_B) * (1 - P_B))) = 91/100 := 
by 
  rw [hA, hB]
  -- exact calculations here
  sorry

end problem_1_problem_2_l46_46624


namespace amanda_tickets_l46_46189

theorem amanda_tickets (F : ℕ) (h : 4 * F + 32 + 28 = 80) : F = 5 :=
by
  sorry

end amanda_tickets_l46_46189


namespace smallest_possible_e_l46_46837

-- Definitions based on given conditions
def polynomial (x : ℝ) (a b c d e : ℝ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + e

-- The given polynomial has roots -3, 4, 8, and -1/4, and e is positive integer
theorem smallest_possible_e :
  ∃ (a b c d e : ℤ), polynomial x a b c d e = 4*x^4 - 32*x^3 - 23*x^2 + 104*x + 96 ∧ e > 0 ∧ e = 96 :=
by
  sorry

end smallest_possible_e_l46_46837


namespace intersection_A_B_l46_46218

def setA : Set ℝ := {x | x^2 - 1 > 0}
def setB : Set ℝ := {x | Real.log x / Real.log 2 < 1}

theorem intersection_A_B :
  {x | x ∈ setA ∧ x ∈ setB} = {x | 1 < x ∧ x < 2} :=
sorry

end intersection_A_B_l46_46218


namespace cycle_selling_price_l46_46745

theorem cycle_selling_price
(C : ℝ := 1900)  -- Cost price of the cycle
(Lp : ℝ := 18)  -- Loss percentage
(S : ℝ := 1558) -- Expected selling price
: (S = C - (Lp / 100) * C) :=
by 
  sorry

end cycle_selling_price_l46_46745


namespace multiply_and_simplify_l46_46342
open Classical

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end multiply_and_simplify_l46_46342


namespace smallest_positive_integer_rel_prime_180_l46_46921

theorem smallest_positive_integer_rel_prime_180 : 
  ∃ x : ℕ, x > 1 ∧ Nat.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Nat.gcd y 180 = 1 → y ≥ 7 := 
by 
  sorry

end smallest_positive_integer_rel_prime_180_l46_46921


namespace pizza_problem_l46_46839

theorem pizza_problem (diameter : ℝ) (sectors : ℕ) (h1 : diameter = 18) (h2 : sectors = 4) : 
  let R := diameter / 2 
  let θ := (2 * Real.pi / sectors : ℝ)
  let m := 2 * R * Real.sin (θ / 2) 
  (m^2 = 162) := by
  sorry

end pizza_problem_l46_46839


namespace direction_vector_of_line_l46_46387

noncomputable def direction_vector_of_line_eq : Prop :=
  ∃ u v, ∀ x y, (x / 4) + (y / 2) = 1 → (u, v) = (-2, 1)

theorem direction_vector_of_line :
  direction_vector_of_line_eq := sorry

end direction_vector_of_line_l46_46387


namespace num_impossible_events_l46_46698

def water_boils_at_90C := false
def iron_melts_at_room_temp := false
def coin_flip_results_heads := true
def abs_value_not_less_than_zero := true

theorem num_impossible_events :
  water_boils_at_90C = false ∧ iron_melts_at_room_temp = false ∧ 
  coin_flip_results_heads = true ∧ abs_value_not_less_than_zero = true →
  (if ¬water_boils_at_90C then 1 else 0) + (if ¬iron_melts_at_room_temp then 1 else 0) +
  (if ¬coin_flip_results_heads then 1 else 0) + (if ¬abs_value_not_less_than_zero then 1 else 0) = 2
:= by
  intro h
  have : 
    water_boils_at_90C = false ∧ iron_melts_at_room_temp = false ∧ 
    coin_flip_results_heads = true ∧ abs_value_not_less_than_zero = true := h
  sorry

end num_impossible_events_l46_46698


namespace exists_number_added_to_sum_of_digits_gives_2014_l46_46129

def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem exists_number_added_to_sum_of_digits_gives_2014 : 
  ∃ (n : ℕ), n + sum_of_digits n = 2014 :=
sorry

end exists_number_added_to_sum_of_digits_gives_2014_l46_46129


namespace strawberries_for_mom_l46_46952

-- Define the conditions as Lean definitions
def dozen : ℕ := 12
def strawberries_picked : ℕ := 2 * dozen
def strawberries_eaten : ℕ := 6

-- Define the statement to be proven
theorem strawberries_for_mom : (strawberries_picked - strawberries_eaten) = 18 := by
  sorry

end strawberries_for_mom_l46_46952


namespace multiply_fractions_l46_46812

theorem multiply_fractions :
  (2/3) * (4/7) * (9/11) * (5/8) = 15/77 :=
by
  -- It is just a statement, no need for the proof steps here
  sorry

end multiply_fractions_l46_46812


namespace find_constants_exist_l46_46728

theorem find_constants_exist :
  ∃ A B C, (∀ x, 4 * x / ((x - 5) * (x - 3)^2) = A / (x - 5) + B / (x - 3) + C / (x - 3)^2)
  ∧ (A = 5) ∧ (B = -5) ∧ (C = -6) := 
sorry

end find_constants_exist_l46_46728


namespace ferry_round_trip_time_increases_l46_46903

variable {S V a b : ℝ}

theorem ferry_round_trip_time_increases (h1 : V > 0) (h2 : a < b) (h3 : V > a) (h4 : V > b) :
  (S / (V + b) + S / (V - b)) > (S / (V + a) + S / (V - a)) :=
by sorry

end ferry_round_trip_time_increases_l46_46903


namespace eric_green_marbles_l46_46501

theorem eric_green_marbles (total_marbles white_marbles blue_marbles : ℕ) (h_total : total_marbles = 20)
  (h_white : white_marbles = 12) (h_blue : blue_marbles = 6) :
  total_marbles - (white_marbles + blue_marbles) = 2 := 
by
  sorry

end eric_green_marbles_l46_46501


namespace alice_cookie_fills_l46_46004

theorem alice_cookie_fills :
  (∀ (a b : ℚ), a = 3 + (3/4) ∧ b = 1/3 → (a / b) = 12) :=
sorry

end alice_cookie_fills_l46_46004


namespace scientific_notation_of_3300000_l46_46848

theorem scientific_notation_of_3300000 : 3300000 = 3.3 * 10^6 :=
by
  sorry

end scientific_notation_of_3300000_l46_46848


namespace lcm_4_6_9_l46_46577

/-- The least common multiple (LCM) of 4, 6, and 9 is 36 -/
theorem lcm_4_6_9 : Nat.lcm (Nat.lcm 4 6) 9 = 36 :=
by
  -- sorry replaces the actual proof steps
  sorry

end lcm_4_6_9_l46_46577


namespace least_adjacent_probability_l46_46264

theorem least_adjacent_probability (n : ℕ) 
    (h₀ : 0 < n)
    (h₁ : (∀ m : ℕ, 0 < m ∧ m < n → (4 * m^2 - 4 * m + 8) / (m^2 * (m^2 - 1)) ≥ 1 / 2015)) : 
    (4 * n^2 - 4 * n + 8) / (n^2 * (n^2 - 1)) < 1 / 2015 := by
  sorry

end least_adjacent_probability_l46_46264


namespace total_cement_used_l46_46747

-- Define the amounts of cement used for Lexi's street and Tess's street
def cement_used_lexis_street : ℝ := 10
def cement_used_tess_street : ℝ := 5.1

-- Prove that the total amount of cement used is 15.1 tons
theorem total_cement_used : cement_used_lexis_street + cement_used_tess_street = 15.1 := sorry

end total_cement_used_l46_46747


namespace ratio_x_y_l46_46133

theorem ratio_x_y (x y : ℚ) (h : (14 * x - 5 * y) / (17 * x - 3 * y) = 2 / 7) : x / y = 29 / 64 :=
by
  sorry

end ratio_x_y_l46_46133


namespace range_of_m_for_common_point_l46_46307

-- Define the quadratic function
def quadratic_function (x m : ℝ) : ℝ :=
  -x^2 - 2 * x + m

-- Define the condition for a common point with the x-axis (i.e., it has real roots)
def has_common_point_with_x_axis (m : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_function x m = 0

-- The theorem statement
theorem range_of_m_for_common_point : ∀ m : ℝ, has_common_point_with_x_axis m ↔ m ≥ -1 := 
sorry

end range_of_m_for_common_point_l46_46307


namespace third_derivative_y_l46_46228

noncomputable def y (x : ℝ) : ℝ := x^2 * Real.sin (5 * x - 3)

theorem third_derivative_y (x : ℝ) : 
  (deriv^[3] y x) = -150 * x * Real.sin (5 * x - 3) + (30 - 125 * x^2) * Real.cos (5 * x - 3) :=
by
  sorry

end third_derivative_y_l46_46228


namespace car_speed_in_first_hour_l46_46509

theorem car_speed_in_first_hour (x : ℝ) 
  (second_hour_speed : ℝ := 40)
  (average_speed : ℝ := 60)
  (h : (x + second_hour_speed) / 2 = average_speed) :
  x = 80 := 
by
  -- Additional steps needed to solve this theorem
  sorry

end car_speed_in_first_hour_l46_46509


namespace normal_complaints_calculation_l46_46794

-- Define the normal number of complaints
def normal_complaints (C : ℕ) : ℕ := C

-- Define the complaints when short-staffed
def short_staffed_complaints (C : ℕ) : ℕ := (4 * C) / 3

-- Define the complaints when both conditions are met
def both_conditions_complaints (C : ℕ) : ℕ := (4 * C) / 3 + (4 * C) / 15

-- Main statement to prove
theorem normal_complaints_calculation (C : ℕ) (h : 3 * (both_conditions_complaints C) = 576) : C = 120 :=
by sorry

end normal_complaints_calculation_l46_46794


namespace remainder_a_cubed_l46_46814

theorem remainder_a_cubed {a n : ℤ} (hn : 0 < n) (hinv : a * a ≡ 1 [ZMOD n]) (ha : a ≡ -1 [ZMOD n]) : a^3 ≡ -1 [ZMOD n] := 
sorry

end remainder_a_cubed_l46_46814


namespace proof_problem_l46_46327

-- Definitions of the function and conditions:
def f : ℝ → ℝ := sorry
axiom odd_f : ∀ x, f (-x) = -f x
axiom periodicity_f : ∀ x, f (x + 2) = -f x
axiom f_def_on_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2^x - 1

-- The theorem statement:
theorem proof_problem :
  f 6 < f (11 / 2) ∧ f (11 / 2) < f (-7) :=
by
  sorry

end proof_problem_l46_46327


namespace point_coordinates_l46_46938

theorem point_coordinates (M : ℝ × ℝ) 
  (hx : abs M.2 = 3) 
  (hy : abs M.1 = 2) 
  (h_first_quadrant : 0 < M.1 ∧ 0 < M.2) : 
  M = (2, 3) := 
sorry

end point_coordinates_l46_46938


namespace problem_l46_46485

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 10 / Real.log 5
noncomputable def c : ℝ := Real.log 12 / Real.log 6

theorem problem : a > b ∧ b > c := by
  sorry

end problem_l46_46485


namespace smaller_angle_clock_1245_l46_46494

theorem smaller_angle_clock_1245 
  (minute_rate : ℕ → ℝ) 
  (hour_rate : ℕ → ℝ) 
  (time : ℕ) 
  (minute_angle : ℝ) 
  (hour_angle : ℝ) 
  (larger_angle : ℝ) 
  (smaller_angle : ℝ) :
  (minute_rate 1 = 6) →
  (hour_rate 1 = 0.5) →
  (time = 45) →
  (minute_angle = minute_rate 45 * 45) →
  (hour_angle = hour_rate 45 * 45) →
  (larger_angle = |minute_angle - hour_angle|) →
  (smaller_angle = 360 - larger_angle) →
  smaller_angle = 112.5 :=
by
  intros
  sorry

end smaller_angle_clock_1245_l46_46494


namespace common_rational_root_neg_not_integer_l46_46120

theorem common_rational_root_neg_not_integer : 
  ∃ (p : ℚ), (p < 0) ∧ (¬ ∃ (z : ℤ), p = z) ∧ 
  (50 * p^4 + a * p^3 + b * p^2 + c * p + 20 = 0) ∧ 
  (20 * p^5 + d * p^4 + e * p^3 + f * p^2 + g * p + 50 = 0) := 
sorry

end common_rational_root_neg_not_integer_l46_46120


namespace bottles_produced_by_twenty_machines_l46_46368

-- Definitions corresponding to conditions
def bottles_per_machine_per_minute (total_machines : ℕ) (total_bottles : ℕ) : ℕ :=
  total_bottles / total_machines

def bottles_produced (machines : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  machines * rate * time

-- Given conditions
axiom six_machines_rate : ∀ (machines total_bottles : ℕ), machines = 6 → total_bottles = 270 →
  bottles_per_machine_per_minute machines total_bottles = 45

-- Prove the question == answer given conditions
theorem bottles_produced_by_twenty_machines :
  bottles_produced 20 45 4 = 3600 :=
by sorry

end bottles_produced_by_twenty_machines_l46_46368


namespace floor_div_eq_floor_div_floor_l46_46266

theorem floor_div_eq_floor_div_floor {α : ℝ} {d : ℕ} (h₁ : 0 < α) : 
  (⌊α / d⌋ = ⌊⌊α⌋ / d⌋) := 
sorry

end floor_div_eq_floor_div_floor_l46_46266


namespace large_cube_painted_blue_l46_46788

theorem large_cube_painted_blue (n : ℕ) (hp : 1 ≤ n) 
  (hc : (6 * n^2) = (1 / 3) * 6 * n^3) : n = 3 := by
  have hh := hc
  sorry

end large_cube_painted_blue_l46_46788


namespace probability_all_quitters_from_same_tribe_l46_46183

noncomputable def total_ways_to_choose_quitters : ℕ := Nat.choose 18 3

noncomputable def ways_all_from_tribe (n : ℕ) : ℕ := Nat.choose n 3

noncomputable def combined_ways_same_tribe : ℕ :=
  ways_all_from_tribe 9 + ways_all_from_tribe 9

noncomputable def probability_same_tribe (total : ℕ) (same_tribe : ℕ) : ℚ :=
  same_tribe / total

theorem probability_all_quitters_from_same_tribe :
  probability_same_tribe total_ways_to_choose_quitters combined_ways_same_tribe = 7 / 34 :=
by
  sorry

end probability_all_quitters_from_same_tribe_l46_46183


namespace simplify_pow_prod_eq_l46_46171

noncomputable def simplify_pow_prod : ℝ :=
  (256:ℝ)^(1/4) * (625:ℝ)^(1/2)

theorem simplify_pow_prod_eq :
  simplify_pow_prod = 100 := by
  sorry

end simplify_pow_prod_eq_l46_46171


namespace calculate_value_l46_46254

theorem calculate_value (a b c x : ℕ) (h_a : a = 1) (h_b : b = 2) (h_c : c = 3) (h_x : x = 3) :
  x^(a * (b + c)) - (x^a + x^b + x^c) = 204 := by
  sorry

end calculate_value_l46_46254


namespace maximum_fly_path_length_in_box_l46_46154

theorem maximum_fly_path_length_in_box
  (length width height : ℝ)
  (h_length : length = 1)
  (h_width : width = 1)
  (h_height : height = 2) :
  ∃ l, l = (Real.sqrt 6 + 2 * Real.sqrt 5 + Real.sqrt 2 + 1) :=
by
  sorry

end maximum_fly_path_length_in_box_l46_46154


namespace expression_equals_8_l46_46386

-- Define the expression we are interested in.
def expression : ℚ :=
  (1 + 1 / 2) * (1 + 1 / 3) * (1 + 1 / 4) * (1 + 1 / 5) * (1 + 1 / 6) * (1 + 1 / 7)

-- Statement we need to prove
theorem expression_equals_8 : expression = 8 := by
  sorry

end expression_equals_8_l46_46386


namespace no_positive_n_l46_46248

theorem no_positive_n :
  ¬ ∃ (n : ℕ) (n_pos : n > 0) (a b : ℕ) (a_sd : a < 10) (b_sd : b < 10), 
    (1234 - n) * b = (6789 - n) * a :=
by 
  sorry

end no_positive_n_l46_46248


namespace intersection_A_B_l46_46810

open Set

def A : Set ℤ := {x : ℤ | ∃ y : ℝ, y = Real.sqrt (1 - (x : ℝ)^2)}
def B : Set ℤ := {y : ℤ | ∃ x : ℤ, x ∈ A ∧ y = 2 * x - 1}

theorem intersection_A_B : A ∩ B = {-1, 1} := 
by {
  sorry
}

end intersection_A_B_l46_46810


namespace robin_photo_count_l46_46881

theorem robin_photo_count (photos_per_page : ℕ) (full_pages : ℕ) 
  (h1 : photos_per_page = 6) (h2 : full_pages = 122) :
  photos_per_page * full_pages = 732 :=
by
  sorry

end robin_photo_count_l46_46881


namespace number_of_articles_l46_46426

theorem number_of_articles (C S : ℝ) (h_gain : S = 1.4285714285714286 * C) (h_cost : ∃ X : ℝ, X * C = 35 * S) : ∃ X : ℝ, X = 50 :=
by
  -- Define the specific existence and equality proof here
  sorry

end number_of_articles_l46_46426


namespace parabola_x_intercept_unique_l46_46085

theorem parabola_x_intercept_unique : ∃! (x : ℝ), ∀ (y : ℝ), x = -y^2 + 2*y + 3 → x = 3 :=
by
  sorry

end parabola_x_intercept_unique_l46_46085


namespace solve_equation_l46_46122

theorem solve_equation : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end solve_equation_l46_46122


namespace counties_rained_on_monday_l46_46270

theorem counties_rained_on_monday : 
  ∀ (M T R_no_both R_both : ℝ),
    T = 0.55 → 
    R_no_both = 0.35 →
    R_both = 0.60 →
    (M + T - R_both = 1 - R_no_both) →
    M = 0.70 :=
by
  intros M T R_no_both R_both hT hR_no_both hR_both hInclusionExclusion
  sorry

end counties_rained_on_monday_l46_46270


namespace simplify_expression_l46_46043

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b^3 + 2 * b^2) - 2 * b^2 + 5 = 9 * b^4 + 6 * b^3 - 2 * b^2 + 5 := sorry

end simplify_expression_l46_46043


namespace number_of_sixth_graders_who_bought_more_pens_than_seventh_graders_l46_46432

theorem number_of_sixth_graders_who_bought_more_pens_than_seventh_graders 
  (p : ℕ) (h1 : 178 % p = 0) (h2 : 252 % p = 0) :
  (252 / p) - (178 / p) = 5 :=
sorry

end number_of_sixth_graders_who_bought_more_pens_than_seventh_graders_l46_46432


namespace value_range_of_a_l46_46925

variable (A B : Set ℝ)

noncomputable def A_def : Set ℝ := { x | 2 * x^2 - 3 * x + 1 ≤ 0 }
noncomputable def B_def (a : ℝ) : Set ℝ := { x | x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0 }

theorem value_range_of_a (a : ℝ) (hA : A = A_def) (hB : B = B_def a) :
    (Bᶜ ∩ A = ∅) → (0 ≤ a ∧ a ≤ 0.5) := 
sorry

end value_range_of_a_l46_46925


namespace intersection_complement_eq_l46_46338

-- Definitions as per given conditions
def U : Set ℕ := { x | x > 0 ∧ x < 9 }
def A : Set ℕ := { 1, 2, 3, 4 }
def B : Set ℕ := { 3, 4, 5, 6 }

-- Complement of B with respect to U
def C_U_B : Set ℕ := U \ B

-- Statement of the theorem to be proved
theorem intersection_complement_eq : A ∩ C_U_B = { 1, 2 } :=
by
  sorry

end intersection_complement_eq_l46_46338


namespace manuscript_typing_cost_l46_46167

theorem manuscript_typing_cost 
  (pages_total : ℕ) (pages_first_time : ℕ) (pages_revised_once : ℕ)
  (pages_revised_twice : ℕ) (rate_first_time : ℕ) (rate_revised : ℕ) 
  (cost_total : ℕ) :
  pages_total = 100 →
  pages_first_time = pages_total →
  pages_revised_once = 35 →
  pages_revised_twice = 15 →
  rate_first_time = 6 →
  rate_revised = 4 →
  cost_total = (pages_first_time * rate_first_time) +
              (pages_revised_once * rate_revised) +
              (pages_revised_twice * rate_revised * 2) →
  cost_total = 860 :=
by
  intros htot hfirst hrev1 hrev2 hr1 hr2 hcost
  sorry

end manuscript_typing_cost_l46_46167


namespace wheel_circumferences_satisfy_conditions_l46_46507

def C_f : ℝ := 24
def C_r : ℝ := 18

theorem wheel_circumferences_satisfy_conditions:
  360 / C_f = 360 / C_r + 4 ∧ 360 / (C_f - 3) = 360 / (C_r - 3) + 6 :=
by 
  have h1: 360 / C_f = 360 / C_r + 4 := sorry
  have h2: 360 / (C_f - 3) = 360 / (C_r - 3) + 6 := sorry
  exact ⟨h1, h2⟩

end wheel_circumferences_satisfy_conditions_l46_46507


namespace integer_solutions_l46_46807

theorem integer_solutions (x y : ℤ) : 
  (x^2 + x = y^4 + y^3 + y^2 + y) ↔ 
  (x, y) = (0, -1) ∨ (x, y) = (-1, -1) ∨ (x, y) = (0, 0) ∨ (x, y) = (-1, 0) ∨ (x, y) = (5, 2) :=
by
  sorry

end integer_solutions_l46_46807


namespace team_total_points_l46_46767

theorem team_total_points 
  (n : ℕ)
  (best_score actual : ℕ)
  (desired_avg : ℕ)
  (hypothetical_score : ℕ)
  (current_best_score : ℕ)
  (team_size : ℕ)
  (h1 : team_size = 8)
  (h2 : current_best_score = 85)
  (h3 : hypothetical_score = 92)
  (h4 : desired_avg = 84)
  (h5 : hypothetical_score - current_best_score = 7)
  (h6 : team_size * desired_avg = 672) :
  (actual = 665) :=
sorry

end team_total_points_l46_46767


namespace count_four_digit_multiples_of_5_l46_46449

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def four_digit_multiples_of_5_count (lower upper : ℕ) :=
  (upper - lower + 1)

theorem count_four_digit_multiples_of_5 : 
  four_digit_multiples_of_5_count 200 1999 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l46_46449


namespace area_between_tangent_circles_l46_46682

theorem area_between_tangent_circles (r : ℝ) (h_r : r > 0) :
  let area_trapezoid := 4 * r^2 * Real.sqrt 3
  let area_sector1 := π * r^2 / 3
  let area_sector2 := 3 * π * r^2 / 2
  area_trapezoid - (area_sector1 + area_sector2) = r^2 * (24 * Real.sqrt 3 - 11 * π) / 6 := by
  sorry

end area_between_tangent_circles_l46_46682


namespace units_digit_of_large_powers_l46_46165

theorem units_digit_of_large_powers : 
  (2^1007 * 6^1008 * 14^1009) % 10 = 2 := 
  sorry

end units_digit_of_large_powers_l46_46165


namespace greatest_three_digit_multiple_of_17_l46_46073

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l46_46073


namespace max_sum_of_factors_l46_46130

theorem max_sum_of_factors (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C)
  (h4 : A * B * C = 3003) : A + B + C ≤ 117 :=
sorry

end max_sum_of_factors_l46_46130


namespace expand_array_l46_46713

theorem expand_array (n : ℕ) (h₁ : n ≥ 3) 
  (matrix : Fin (n-2) → Fin n → Fin n)
  (h₂ : ∀ i : Fin (n-2), ∀ j: Fin n, ∀ k: Fin n, j ≠ k → matrix i j ≠ matrix i k)
  (h₃ : ∀ j : Fin n, ∀ k: Fin (n-2), ∀ l: Fin (n-2), k ≠ l → matrix k j ≠ matrix l j) :
  ∃ (expanded_matrix : Fin n → Fin n → Fin n), 
    (∀ i : Fin n, ∀ j: Fin n, ∀ k: Fin n, j ≠ k → expanded_matrix i j ≠ expanded_matrix i k) ∧
    (∀ j : Fin n, ∀ k: Fin n, ∀ l: Fin n, k ≠ l → expanded_matrix k j ≠ expanded_matrix l j) :=
sorry

end expand_array_l46_46713


namespace prob_a_wins_match_l46_46187

-- Define the probability of A winning a single game
def prob_win_a_single_game : ℚ := 1 / 3

-- Define the probability of A winning two consecutive games
def prob_win_a_two_consec_games : ℚ := prob_win_a_single_game * prob_win_a_single_game

-- Define the probability of A winning two games with one loss in between
def prob_win_a_two_wins_one_loss_first : ℚ := prob_win_a_single_game * (1 - prob_win_a_single_game) * prob_win_a_single_game
def prob_win_a_two_wins_one_loss_second : ℚ := (1 - prob_win_a_single_game) * prob_win_a_single_game * prob_win_a_single_game

-- Define the total probability of A winning the match
def prob_a_winning_match : ℚ := prob_win_a_two_consec_games + prob_win_a_two_wins_one_loss_first + prob_win_a_two_wins_one_loss_second

-- The theorem to be proved
theorem prob_a_wins_match : prob_a_winning_match = 7 / 27 :=
by sorry

end prob_a_wins_match_l46_46187


namespace find_other_root_l46_46965

theorem find_other_root (m : ℝ) :
  (∃ m : ℝ, (∀ x : ℝ, (x = -6 → (x^2 + m * x - 6 = 0))) → (x^2 + m * x - 6 = (x + 6) * (x - 1)) → (∀ x : ℝ, (x^2 + 5 * x - 6 = 0) → (x = -6 ∨ x = 1))) :=
sorry

end find_other_root_l46_46965


namespace ladder_slip_l46_46541

theorem ladder_slip (l : ℝ) (d1 d2 : ℝ) (h1 h2 : ℝ) :
  l = 30 → d1 = 8 → h1^2 + d1^2 = l^2 → h2 = h1 - 4 → 
  (h2^2 + (d1 + d2)^2 = l^2) → d2 = 2 :=
by
  intros h_l h_d1 h_h1_eq h_h2 h2_eq_l   
  sorry

end ladder_slip_l46_46541


namespace there_exists_triangle_part_two_l46_46922

noncomputable def exists_triangle (a b c : ℝ) : Prop :=
a > 0 ∧
4 * a - 8 * b + 4 * c ≥ 0 ∧
9 * a - 12 * b + 4 * c ≥ 0 ∧
2 * a ≤ 2 * b ∧
2 * b ≤ 3 * a ∧
b^2 ≥ a*c

theorem there_exists_triangle (a b c : ℝ) (h1 : a > 0)
  (h2 : 4 * a - 8 * b + 4 * c ≥ 0)
  (h3 : 9 * a - 12 * b + 4 * c ≥ 0)
  (h4 : 2 * a ≤ 2 * b)
  (h5 : 2 * b ≤ 3 * a)
  (h6 : b^2 ≥ a * c) : 
 a ≤ b ∧ b ≤ c ∧ a + b > c :=
sorry

theorem part_two (a b c : ℝ) (h1 : a > 0) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c < a + b) :
  ∃ h : a > 0, (a / (a + c) + b / (b + a) > c / (b + c)) :=
sorry

end there_exists_triangle_part_two_l46_46922


namespace molecular_weight_CaCO3_is_100_09_l46_46398

-- Declare the atomic weights
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_C : ℝ := 12.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight constant for calcium carbonate
def molecular_weight_CaCO3 : ℝ :=
  (1 * atomic_weight_Ca) + (1 * atomic_weight_C) + (3 * atomic_weight_O)

-- Prove that the molecular weight of calcium carbonate is 100.09 g/mol
theorem molecular_weight_CaCO3_is_100_09 :
  molecular_weight_CaCO3 = 100.09 :=
by
  -- Proof goes here, placeholder for now
  sorry

end molecular_weight_CaCO3_is_100_09_l46_46398


namespace number_of_games_in_division_l46_46308

theorem number_of_games_in_division (P Q : ℕ) (h1 : P > 2 * Q) (h2 : Q > 6) (schedule_eq : 4 * P + 5 * Q = 82) : 4 * P = 52 :=
by sorry

end number_of_games_in_division_l46_46308


namespace Hillary_sunday_minutes_l46_46989

variable (total_minutes friday_minutes saturday_minutes : ℕ)

theorem Hillary_sunday_minutes 
  (h_total : total_minutes = 60) 
  (h_friday : friday_minutes = 16) 
  (h_saturday : saturday_minutes = 28) : 
  ∃ sunday_minutes : ℕ, total_minutes - (friday_minutes + saturday_minutes) = sunday_minutes ∧ sunday_minutes = 16 := 
by
  sorry

end Hillary_sunday_minutes_l46_46989


namespace problem_solution_l46_46203

theorem problem_solution (a b : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = a * x^2 + (b - 3) * x + 3) →
  (∀ x : ℝ, f x = f (-x)) →
  (a^2 - 2 = -a) →
  a + b = 4 :=
by
  intros h1 h2 h3
  sorry

end problem_solution_l46_46203


namespace captain_age_eq_your_age_l46_46252

-- Represent the conditions as assumptions
variables (your_age : ℕ) -- You, the captain, have an age as a natural number

-- Define the statement
theorem captain_age_eq_your_age (H_cap : ∀ captain, captain = your_age) : ∀ captain, captain = your_age := by
  sorry

end captain_age_eq_your_age_l46_46252


namespace constant_function_of_functional_equation_l46_46475

theorem constant_function_of_functional_equation {f : ℝ → ℝ} (h : ∀ x y : ℝ, 0 < x → 0 < y → f (x + y) = f (x^2 + y^2)) : ∃ c : ℝ, ∀ x : ℝ, 0 < x → f x = c := 
sorry

end constant_function_of_functional_equation_l46_46475


namespace solve_for_a_l46_46249

-- Define the lines
def l1 (x y : ℝ) := x + y - 2 = 0
def l2 (x y a : ℝ) := 2 * x + a * y - 3 = 0

-- Define orthogonality condition
def perpendicular (m₁ m₂ : ℝ) := m₁ * m₂ = -1

-- The theorem to prove
theorem solve_for_a (a : ℝ) :
  (∀ x y : ℝ, l1 x y → ∀ x y : ℝ, l2 x y a → perpendicular (-1) (-2 / a)) → a = 2 := 
sorry

end solve_for_a_l46_46249


namespace gcd_repeated_integer_l46_46107

theorem gcd_repeated_integer (n : ℕ) (h1 : 100 ≤ n ∧ n < 1000) :
  ∃ d, (∀ k : ℕ, k = 1001001001 * n → d = 1001001001 ∧ d ∣ k) :=
sorry

end gcd_repeated_integer_l46_46107


namespace find_value_of_m_l46_46095

variables (x y m : ℝ)

theorem find_value_of_m (h1 : y ≥ x) (h2 : x + 3 * y ≤ 4) (h3 : x ≥ m) (hz_max : ∀ z, (z = x - 3 * y) → z ≤ 8) :
  m = -4 :=
sorry

end find_value_of_m_l46_46095


namespace probability_of_sum_8_9_10_l46_46872

/-- The list of face values for the first die. -/
def first_die : List ℕ := [1, 1, 3, 3, 5, 6]

/-- The list of face values for the second die. -/
def second_die : List ℕ := [1, 2, 4, 5, 7, 9]

/-- The condition to verify if the sum is 8, 9, or 10. -/
def valid_sum (s : ℕ) : Bool := s = 8 ∨ s = 9 ∨ s = 10

/-- Calculate probability of the sum being 8, 9, or 10 for the two dice. -/
def calculate_probability : ℚ :=
  let total_rolls := first_die.length * second_die.length
  let valid_rolls := 
    first_die.foldl (fun acc d1 =>
      acc + second_die.foldl (fun acc' d2 => 
        if valid_sum (d1 + d2) then acc' + 1 else acc') 0) 0
  valid_rolls / total_rolls

/-- The required probability is 7/18. -/
theorem probability_of_sum_8_9_10 : calculate_probability = 7 / 18 := 
  sorry

end probability_of_sum_8_9_10_l46_46872


namespace quadratic_function_properties_l46_46495

noncomputable def quadratic_function (m : ℝ) (x : ℝ) : ℝ :=
  (m + 2) * x^(m^2 + m - 4)

theorem quadratic_function_properties :
  (∀ m, (m^2 + m - 4 = 2) → (m = -3 ∨ m = 2))
  ∧ (m = -3 → quadratic_function m 0 = 0) 
  ∧ (m = -3 → ∀ x, x > 0 → quadratic_function m x ≤ quadratic_function m 0 ∧ quadratic_function m x < 0)
  ∧ (m = -3 → ∀ x, x < 0 → quadratic_function m x ≤ quadratic_function m 0 ∧ quadratic_function m x < 0) :=
by
  -- Proof will be supplied here.
  sorry

end quadratic_function_properties_l46_46495


namespace green_marbles_l46_46849

theorem green_marbles 
  (total_marbles : ℕ)
  (red_marbles : ℕ)
  (at_least_blue_marbles : ℕ)
  (h1 : total_marbles = 63) 
  (h2 : at_least_blue_marbles ≥ total_marbles / 3) 
  (h3 : red_marbles = 38) 
  : ∃ green_marbles : ℕ, total_marbles - red_marbles - at_least_blue_marbles = green_marbles ∧ green_marbles = 4 :=
by
  sorry

end green_marbles_l46_46849


namespace elena_savings_l46_46705

theorem elena_savings :
  let original_cost := 7 * 3
  let discount_rate := 0.25
  let rebate := 5
  let disc_amount := original_cost * discount_rate
  let price_after_discount := original_cost - disc_amount
  let final_price := price_after_discount - rebate
  original_cost - final_price = 10.25 :=
by
  sorry

end elena_savings_l46_46705


namespace digit_difference_l46_46711

variable (X Y : ℕ)

theorem digit_difference (h : 10 * X + Y - (10 * Y + X) = 27) : X - Y = 3 :=
by
  sorry

end digit_difference_l46_46711


namespace concentration_of_acid_in_third_flask_is_correct_l46_46686

noncomputable def concentration_of_acid_in_third_flask
  (acid_flask1 : ℕ) (acid_flask2 : ℕ) (acid_flask3 : ℕ) 
  (water_first_to_first_flask : ℕ) (water_second_to_second_flask : Rat) :
  Rat :=
  let total_water := water_first_to_first_flask + water_second_to_second_flask
  let concentration := (acid_flask3 : Rat) / (acid_flask3 + total_water) * 100
  concentration

theorem concentration_of_acid_in_third_flask_is_correct :
  concentration_of_acid_in_third_flask 10 20 30 190 (460/7) = 10.5 :=
  sorry

end concentration_of_acid_in_third_flask_is_correct_l46_46686


namespace pyramid_volume_formula_l46_46087

noncomputable def pyramid_volume (a α β : ℝ) : ℝ :=
  (1/6) * a^3 * (Real.sin (α/2)) * (Real.tan β)

theorem pyramid_volume_formula (a α β : ℝ) :
  (base_is_isosceles_triangle : Prop) → (lateral_edges_inclined : Prop) → 
  pyramid_volume a α β = (1/6) * a^3 * (Real.sin (α/2)) * (Real.tan β) :=
by
  intros c1 c2
  exact sorry

end pyramid_volume_formula_l46_46087


namespace minimum_value_of_y_l46_46757

theorem minimum_value_of_y (x y : ℝ) (h : x^2 + y^2 = 10 * x + 36 * y) : y ≥ -7 :=
sorry

end minimum_value_of_y_l46_46757


namespace area_ratio_of_circles_l46_46099

theorem area_ratio_of_circles (R_A R_B : ℝ) 
  (h1 : (60 / 360) * (2 * Real.pi * R_A) = (40 / 360) * (2 * Real.pi * R_B)) :
  (Real.pi * R_A ^ 2) / (Real.pi * R_B ^ 2) = 9 / 4 := 
sorry

end area_ratio_of_circles_l46_46099


namespace wealth_ratio_l46_46535

theorem wealth_ratio (W P : ℝ) (hW_pos : 0 < W) (hP_pos : 0 < P) :
  let wX := 0.54 * W / (0.40 * P)
  let wY := 0.30 * W / (0.20 * P)
  wX / wY = 0.9 := 
by
  sorry

end wealth_ratio_l46_46535


namespace question1_l46_46584

def sequence1 (a : ℕ → ℕ) : Prop :=
   a 1 = 1 ∧ ∀ n, n ≥ 2 → a n = 3 * a (n - 1) + 1

noncomputable def a_n1 (n : ℕ) : ℕ := (3^n - 1) / 2

theorem question1 (a : ℕ → ℕ) (n : ℕ) : sequence1 a → a n = a_n1 n :=
by
  sorry

end question1_l46_46584


namespace usual_time_to_school_l46_46050

-- Define the conditions
variables (R T : ℝ) (h1 : 0 < T) (h2 : 0 < R)
noncomputable def boy_reaches_school_early : Prop :=
  (7/6 * R) * (T - 5) = R * T

-- The theorem stating the usual time to reach the school
theorem usual_time_to_school (h : boy_reaches_school_early R T) : T = 35 :=
by
  sorry

end usual_time_to_school_l46_46050


namespace evaluate_expression_l46_46832

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end evaluate_expression_l46_46832


namespace yeast_cells_at_2_20_pm_l46_46598

noncomputable def yeast_population (initial : Nat) (rate : Nat) (intervals : Nat) : Nat :=
  initial * rate ^ intervals

theorem yeast_cells_at_2_20_pm :
  let initial_population := 30
  let triple_rate := 3
  let intervals := 5 -- 20 minutes / 4 minutes per interval
  yeast_population initial_population triple_rate intervals = 7290 :=
by
  let initial_population := 30
  let triple_rate := 3
  let intervals := 5
  show yeast_population initial_population triple_rate intervals = 7290
  sorry

end yeast_cells_at_2_20_pm_l46_46598


namespace max_product_h_k_l46_46084

theorem max_product_h_k {h k : ℝ → ℝ} (h_bound : ∀ x, -3 ≤ h x ∧ h x ≤ 5) (k_bound : ∀ x, -1 ≤ k x ∧ k x ≤ 4) :
  ∃ x y, h x * k y = 20 :=
by
  sorry

end max_product_h_k_l46_46084


namespace nonzero_roots_ratio_l46_46875

theorem nonzero_roots_ratio (m : ℝ) (h : m ≠ 0) :
  (∃ (r s : ℝ), r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 ∧ r + s = 4 ∧ r * s = m) → m = 3 :=
by 
  intro h_exists
  obtain ⟨r, s, hr_ne_zero, hs_ne_zero, h_ratio, h_sum, h_prod⟩ := h_exists
  sorry

end nonzero_roots_ratio_l46_46875


namespace triangle_side_lengths_l46_46566

theorem triangle_side_lengths (a b c : ℝ) 
  (h1 : a + b + c = 18) 
  (h2 : a + b = 2 * c) 
  (h3 : b = 2 * a):
  a = 4 ∧ b = 8 ∧ c = 6 := 
by
  sorry

end triangle_side_lengths_l46_46566


namespace tom_teaching_years_l46_46097

theorem tom_teaching_years (T D : ℝ) (h1 : T + D = 70) (h2 : D = (1/2) * T - 5) : T = 50 :=
by
  -- This is where the proof would normally go if it were required.
  sorry

end tom_teaching_years_l46_46097


namespace art_collection_total_area_l46_46161

-- Define the dimensions and quantities of the paintings
def square_painting_side := 6
def small_painting_width := 2
def small_painting_height := 3
def large_painting_width := 10
def large_painting_height := 15

def num_square_paintings := 3
def num_small_paintings := 4
def num_large_paintings := 1

-- Define areas of individual paintings
def square_painting_area := square_painting_side * square_painting_side
def small_painting_area := small_painting_width * small_painting_height
def large_painting_area := large_painting_width * large_painting_height

-- Define the total area calculation
def total_area :=
  num_square_paintings * square_painting_area +
  num_small_paintings * small_painting_area +
  num_large_paintings * large_painting_area

-- The theorem statement
theorem art_collection_total_area : total_area = 282 := by
  sorry

end art_collection_total_area_l46_46161


namespace function_translation_l46_46391

def translateLeft (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x => f (x + a)
def translateUp (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x => (f x) + b

theorem function_translation :
  (translateUp (translateLeft (λ x => 2 * x^2) 1) 3) = λ x => 2 * (x + 1)^2 + 3 :=
by
  sorry

end function_translation_l46_46391


namespace find_y_given_x_eq_0_l46_46562

theorem find_y_given_x_eq_0 (t : ℚ) (x y : ℚ) (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 6) (h3 : x = 0) : 
  y = 21 / 2 :=
by
  sorry

end find_y_given_x_eq_0_l46_46562


namespace fixed_point_translation_l46_46443

variable {R : Type*} [LinearOrderedField R]

def passes_through (f : R → R) (p : R × R) : Prop := f p.1 = p.2

theorem fixed_point_translation (f : R → R) (h : f 1 = 1) :
  passes_through (fun x => f (x + 2)) (-1, 1) :=
by
  sorry

end fixed_point_translation_l46_46443


namespace andrew_total_payment_l46_46312

-- Given conditions
def quantity_of_grapes := 14
def rate_per_kg_grapes := 54
def quantity_of_mangoes := 10
def rate_per_kg_mangoes := 62

-- Calculations
def cost_of_grapes := quantity_of_grapes * rate_per_kg_grapes
def cost_of_mangoes := quantity_of_mangoes * rate_per_kg_mangoes
def total_amount_paid := cost_of_grapes + cost_of_mangoes

-- Theorem to prove
theorem andrew_total_payment : total_amount_paid = 1376 := by
  sorry

end andrew_total_payment_l46_46312


namespace find_set_M_l46_46156

variable (U M : Set ℕ)
variable [DecidableEq ℕ]

-- Universel set U is {1, 3, 5, 7}
def universal_set : Set ℕ := {1, 3, 5, 7}

-- define the complement C_U M
def complement (U M : Set ℕ) : Set ℕ := U \ M

-- M is the set to find such that complement of M in U is {5, 7}
theorem find_set_M (M : Set ℕ) (h : complement universal_set M = {5, 7}) : M = {1, 3} := by
  sorry

end find_set_M_l46_46156


namespace geom_seq_thm_l46_46406

noncomputable def geom_seq (a : ℕ → ℝ) :=
  a 1 = 2 ∧ (a 2 * a 4 = a 6)

noncomputable def b_seq (a : ℕ → ℝ) (n : ℕ) :=
  1 / (Real.logb 2 (a (2 * n - 1)) * Real.logb 2 (a (2 * n + 1)))

noncomputable def sn_sum (b : ℕ → ℝ) (n : ℕ) :=
  (Finset.range (n + 1)).sum b

theorem geom_seq_thm (a : ℕ → ℝ) (n : ℕ) (b : ℕ → ℝ) :
  geom_seq a →
  ∀ n, a n = 2 ^ n ∧ sn_sum (b_seq a) n = n / (2 * n + 1) :=
by
  sorry

end geom_seq_thm_l46_46406


namespace pqr_problem_l46_46724

noncomputable def pqr_abs (p q r : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : p ≠ q) (h5 : q ≠ r) (h6 : r ≠ p)
  (h7 : p + (2 / q) = q + (2 / r)) 
  (h8 : q + (2 / r) = r + (2 / p)) : ℝ :=
|p * q * r|

theorem pqr_problem (p q r : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : p ≠ q) (h5 : q ≠ r) (h6 : r ≠ p)
  (h7 : p + (2 / q) = q + (2 / r)) 
  (h8 : q + (2 / r) = r + (2 / p)) : pqr_abs p q r h1 h2 h3 h4 h5 h6 h7 h8 = 2 := 
sorry

end pqr_problem_l46_46724


namespace max_banner_area_l46_46677

theorem max_banner_area (x y : ℕ) (cost_constraint : 330 * x + 450 * y ≤ 10000) : x * y ≤ 165 :=
by
  sorry

end max_banner_area_l46_46677


namespace simplest_square_root_l46_46412

theorem simplest_square_root :
  let a := Real.sqrt (1 / 2)
  let b := Real.sqrt 11
  let c := Real.sqrt 27
  let d := Real.sqrt 0.3
  (b < a ∧ b < c ∧ b < d) :=
sorry

end simplest_square_root_l46_46412


namespace set_intersection_l46_46202

def U : Set ℝ := Set.univ
def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | x ≥ 2}
def C_U_B : Set ℝ := {x | x < 2}

theorem set_intersection :
  A ∩ C_U_B = {-1, 0, 1} :=
sorry

end set_intersection_l46_46202


namespace speed_upstream_calculation_l46_46078

def speed_boat_still_water : ℝ := 60
def speed_current : ℝ := 17

theorem speed_upstream_calculation : speed_boat_still_water - speed_current = 43 := by
  sorry

end speed_upstream_calculation_l46_46078


namespace part_I_part_II_l46_46636

noncomputable def f (x : ℝ) (a : ℝ) := x - (2 * a - 1) / x - 2 * a * Real.log x

theorem part_I (a : ℝ) (h : a = 3 / 2) : 
  (∀ x, 0 < x ∧ x < 1 → f x a < 0) ∧ (∀ x, 1 < x ∧ x < 2 → f x a > 0) ∧ (∀ x, 2 < x → f x a < 0) := sorry

theorem part_II (a : ℝ) : (∀ x, 1 ≤ x → f x a ≥ 0) → a ≤ 1 := sorry

end part_I_part_II_l46_46636


namespace find_percentage_l46_46947

noncomputable def percentage_solve (x : ℝ) : Prop :=
  0.15 * 40 = (x / 100) * 16 + 2

theorem find_percentage (x : ℝ) (h : percentage_solve x) : x = 25 :=
by
  sorry

end find_percentage_l46_46947


namespace find_divisors_of_10_pow_10_sum_157_l46_46355

theorem find_divisors_of_10_pow_10_sum_157 
  (x y : ℕ) 
  (hx₁ : 0 < x) 
  (hy₁ : 0 < y) 
  (hx₂ : x ∣ 10^10) 
  (hy₂ : y ∣ 10^10) 
  (hxy₁ : x ≠ y) 
  (hxy₂ : x + y = 157) : 
  (x = 32 ∧ y = 125) ∨ (x = 125 ∧ y = 32) := 
by
  sorry

end find_divisors_of_10_pow_10_sum_157_l46_46355


namespace complex_expression_l46_46421

theorem complex_expression (i : ℂ) (h : i^2 = -1) : ( (1 + i) / (1 - i) )^2006 = -1 :=
by {
  sorry
}

end complex_expression_l46_46421


namespace point_reflection_x_axis_l46_46427

-- Definition of the original point P
def P : ℝ × ℝ := (-2, 5)

-- Function to reflect a point across the x-axis
def reflect_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.1, -point.2)

-- Our theorem
theorem point_reflection_x_axis :
  reflect_x_axis P = (-2, -5) := by
  sorry

end point_reflection_x_axis_l46_46427


namespace match_processes_count_l46_46538

-- Define the sets and the number of interleavings
def team_size : ℕ := 4 -- Each team has 4 players

-- Define the problem statement
theorem match_processes_count :
  (Nat.choose (2 * team_size) team_size) = 70 := by
  -- This is where the proof would go, but we'll use sorry as specified
  sorry

end match_processes_count_l46_46538


namespace min_value_sin_cos_expr_l46_46484

open Real

theorem min_value_sin_cos_expr (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  ∃ min_val : ℝ, min_val = 3 * sqrt 2 ∧ ∀ β, (0 < β ∧ β < π / 2) → 
    sin β + cos β + (2 * sqrt 2) / sin (β + π / 4) ≥ min_val :=
by
  sorry

end min_value_sin_cos_expr_l46_46484


namespace remaining_amount_is_1520_l46_46199

noncomputable def totalAmountToBePaid (deposit : ℝ) (depositRate : ℝ) (taxRate : ℝ) (processingFee : ℝ) : ℝ :=
  let fullPrice := deposit / depositRate
  let salesTax := taxRate * fullPrice
  let totalAdditionalExpenses := salesTax + processingFee
  (fullPrice - deposit) + totalAdditionalExpenses

theorem remaining_amount_is_1520 :
  totalAmountToBePaid 140 0.10 0.15 50 = 1520 := by
  sorry

end remaining_amount_is_1520_l46_46199


namespace determine_a_l46_46860

-- Define the sets A and B
def A : Set ℕ := {1, 3}
def B (a : ℕ) : Set ℕ := {1, 2, a}

-- The proof statement
theorem determine_a (a : ℕ) (h : A ⊆ B a) : a = 3 :=
by 
  sorry

end determine_a_l46_46860


namespace clock_resale_price_l46_46746

theorem clock_resale_price
    (C : ℝ)  -- original cost of the clock to the store
    (H1 : 0.40 * C = 100)  -- condition: difference between original cost and buy-back price is $100
    (H2 : ∀ (C : ℝ), resell_price = 1.80 * (0.60 * C))  -- store sold the clock again with a 80% profit on buy-back
    : resell_price = 270 := 
by
  sorry

end clock_resale_price_l46_46746


namespace regular_polygon_perimeter_l46_46261

theorem regular_polygon_perimeter (n : ℕ) (exterior_angle : ℝ) (side_length : ℝ) 
  (h1 : 360 / exterior_angle = n) (h2 : 20 = exterior_angle)
  (h3 : 10 = side_length) : 180 = n * side_length :=
by
  sorry

end regular_polygon_perimeter_l46_46261


namespace mixed_number_division_l46_46998

theorem mixed_number_division : 
  let a := 9 / 4
  let b := 3 / 5
  a / b = 15 / 4 :=
by
  sorry

end mixed_number_division_l46_46998


namespace problem_I_problem_II_l46_46777

open Real -- To use real number definitions and sin function.
open Set -- To use set constructs like intervals.

noncomputable def f (x : ℝ) : ℝ := sin (4 * x - π / 6) + sqrt 3 * sin (4 * x + π / 3)

-- Proof statement for monotonically decreasing interval of f(x).
theorem problem_I (k : ℤ) : 
  ∃ k : ℤ, ∀ x : ℝ, x ∈ Icc ((π / 12) + (k * π / 2)) ((π / 3) + (k * π / 2)) → 
  (4 * x + π / 6) ∈ Icc ((π / 2) + 2 * k * π) ((3 * π / 2) + 2 * k * π) := 
sorry

noncomputable def g (x : ℝ) : ℝ := 2 * sin (x + π / 4)

-- Proof statement for the range of g(x) on the interval [-π, 0].
theorem problem_II : 
  ∀ x : ℝ, x ∈ Icc (-π) 0 → g x ∈ Icc (-2) (sqrt 2) := 
sorry

end problem_I_problem_II_l46_46777


namespace rational_solution_exists_l46_46180

theorem rational_solution_exists (a b c : ℤ) (x₀ y₀ z₀ : ℤ) (h₀ : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h₁ : a * x₀^2 + b * y₀^2 + c * z₀^2 = 0) (h₂ : x₀ ≠ 0 ∨ y₀ ≠ 0 ∨ z₀ ≠ 0) : 
  ∃ (x y z : ℚ), a * x^2 + b * y^2 + c * z^2 = 1 := 
sorry

end rational_solution_exists_l46_46180


namespace probability_of_third_round_expected_value_of_X_variance_of_X_l46_46721

-- Define the probabilities for passing each round
def P_A : ℚ := 2 / 3
def P_B : ℚ := 3 / 4
def P_C : ℚ := 4 / 5

-- Prove the probability of reaching the third round
theorem probability_of_third_round :
  P_A * P_B = 1 / 2 := sorry

-- Define the probability distribution
def P_X (x : ℕ) : ℚ :=
  if x = 1 then 1 / 3 
  else if x = 2 then 1 / 6
  else if x = 3 then 1 / 2
  else 0

-- Expected value
def EX : ℚ := 1 * (1 / 3) + 2 * (1 / 6) + 3 * (1 / 2)

theorem expected_value_of_X :
  EX = 13 / 6 := sorry

-- E(X^2) computation
def EX2 : ℚ := 1^2 * (1 / 3) + 2^2 * (1 / 6) + 3^2 * (1 / 2)

-- Variance
def variance_X : ℚ := EX2 - EX^2

theorem variance_of_X :
  variance_X = 41 / 36 := sorry

end probability_of_third_round_expected_value_of_X_variance_of_X_l46_46721


namespace general_formula_minimum_n_exists_l46_46181

noncomputable def a_n (n : ℕ) : ℝ := 3 * (-2)^(n-1)
noncomputable def S_n (n : ℕ) : ℝ := 1 - (-2)^n

theorem general_formula (n : ℕ) : a_n n = 3 * (-2)^(n-1) :=
by sorry

theorem minimum_n_exists :
  (∃ n : ℕ, S_n n > 2016) ∧ (∀ m : ℕ, S_n m > 2016 → 11 ≤ m) :=
by sorry

end general_formula_minimum_n_exists_l46_46181


namespace pizzas_ordered_l46_46371

def number_of_people : ℝ := 8.0
def slices_per_person : ℝ := 2.625
def slices_per_pizza : ℝ := 8.0

theorem pizzas_ordered : ⌈number_of_people * slices_per_person / slices_per_pizza⌉ = 3 := 
by
  sorry

end pizzas_ordered_l46_46371


namespace housewife_saving_l46_46511

theorem housewife_saving :
  let total_money := 450
  let groceries_fraction := 3 / 5
  let household_items_fraction := 1 / 6
  let personal_care_items_fraction := 1 / 10
  let groceries_expense := groceries_fraction * total_money
  let household_items_expense := household_items_fraction * total_money
  let personal_care_items_expense := personal_care_items_fraction * total_money
  let total_expense := groceries_expense + household_items_expense + personal_care_items_expense
  total_money - total_expense = 60 :=
by
  sorry

end housewife_saving_l46_46511


namespace age_difference_l46_46822

theorem age_difference
  (A B : ℕ)
  (hB : B = 48)
  (h_condition : A + 10 = 2 * (B - 10)) :
  A - B = 18 :=
by
  sorry

end age_difference_l46_46822


namespace bill_difference_proof_l46_46025

variable (a b c : ℝ)

def alice_condition := (25/100) * a = 5
def bob_condition := (20/100) * b = 6
def carol_condition := (10/100) * c = 7

theorem bill_difference_proof (ha : alice_condition a) (hb : bob_condition b) (hc : carol_condition c) :
  max a (max b c) - min a (min b c) = 50 :=
by sorry

end bill_difference_proof_l46_46025


namespace sin_690_eq_neg_0_5_l46_46930

theorem sin_690_eq_neg_0_5 : Real.sin (690 * Real.pi / 180) = -0.5 := by
  sorry

end sin_690_eq_neg_0_5_l46_46930


namespace length_of_XY_in_triangle_XYZ_l46_46115

theorem length_of_XY_in_triangle_XYZ :
  ∀ (XYZ : Type) (X Y Z : XYZ) (angle : XYZ → XYZ → XYZ → ℝ) (length : XYZ → XYZ → ℝ),
  angle X Z Y = 30 ∧ angle Y X Z = 90 ∧ length X Z = 8 → length X Y = 16 :=
by sorry

end length_of_XY_in_triangle_XYZ_l46_46115


namespace average_brown_mms_l46_46096

def brown_smiley_counts : List Nat := [9, 12, 8, 8, 3]
def brown_star_counts : List Nat := [7, 14, 11, 6, 10]

def average (lst : List Nat) : Float :=
  (lst.foldl (· + ·) 0).toFloat / lst.length.toFloat
  
theorem average_brown_mms :
  average brown_smiley_counts = 8 ∧
  average brown_star_counts = 9.6 :=
by 
  sorry

end average_brown_mms_l46_46096


namespace data_plan_comparison_l46_46727

theorem data_plan_comparison : ∃ (m : ℕ), 500 < m :=
by
  let cost_plan_x (m : ℕ) : ℕ := 15 * m
  let cost_plan_y (m : ℕ) : ℕ := 2500 + 10 * m
  use 501
  have h : 500 < 501 := by norm_num
  exact h

end data_plan_comparison_l46_46727


namespace find_possible_values_a_l46_46253

theorem find_possible_values_a :
  ∃ a : ℤ, ∃ b : ℤ, ∃ c : ℤ, 
  (∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) ∧
  ((b + 5) * (c + 5) = 1 ∨ (b + 5) * (c + 5) = 4) ↔ 
  a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 7 :=
by
  sorry

end find_possible_values_a_l46_46253


namespace investment_ratio_same_period_l46_46567

-- Define the profits of A and B
def profit_A : ℕ := 60000
def profit_B : ℕ := 6000

-- Define their investment ratio given the same time period
theorem investment_ratio_same_period : profit_A / profit_B = 10 :=
by
  -- Proof skipped 
  sorry

end investment_ratio_same_period_l46_46567


namespace minimum_value_quot_l46_46332

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem minimum_value_quot (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : f a = f b) :
  (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2 :=
by
  sorry

end minimum_value_quot_l46_46332


namespace master_zhang_must_sell_100_apples_l46_46055

-- Define the given conditions
def buying_price_per_apple : ℚ := 1 / 4 -- 1 yuan for 4 apples
def selling_price_per_apple : ℚ := 2 / 5 -- 2 yuan for 5 apples
def profit_per_apple : ℚ := selling_price_per_apple - buying_price_per_apple

-- Define the target profit
def target_profit : ℚ := 15

-- Define the number of apples to sell
def apples_to_sell : ℚ := target_profit / profit_per_apple

-- The theorem statement: Master Zhang must sell 100 apples to achieve the target profit of 15 yuan
theorem master_zhang_must_sell_100_apples :
  apples_to_sell = 100 :=
sorry

end master_zhang_must_sell_100_apples_l46_46055


namespace number_of_glass_bottles_l46_46401

theorem number_of_glass_bottles (total_litter : ℕ) (aluminum_cans : ℕ) (glass_bottles : ℕ) : 
  total_litter = 18 → aluminum_cans = 8 → glass_bottles = total_litter - aluminum_cans → glass_bottles = 10 :=
by
  intros h_total h_aluminum h_glass
  rw [h_total, h_aluminum] at h_glass
  exact h_glass.trans rfl


end number_of_glass_bottles_l46_46401


namespace solve_quadratic_eq_l46_46505

theorem solve_quadratic_eq : ∃ (a b : ℕ), a = 145 ∧ b = 7 ∧ a + b = 152 ∧ 
  ∀ x, x = Real.sqrt a - b → x^2 + 14 * x = 96 :=
by 
  use 145, 7
  simp
  sorry

end solve_quadratic_eq_l46_46505


namespace simplify_expression_l46_46369

theorem simplify_expression (a1 a2 a3 a4 : ℝ) (h1 : 1 - a1 ≠ 0) (h2 : 1 - a2 ≠ 0) (h3 : 1 - a3 ≠ 0) (h4 : 1 - a4 ≠ 0) :
  1 + a1 / (1 - a1) + a2 / ((1 - a1) * (1 - a2)) + a3 / ((1 - a1) * (1 - a2) * (1 - a3)) + 
  (a4 - a1) / ((1 - a1) * (1 - a2) * (1 - a3) * (1 - a4)) = 
  1 / ((1 - a2) * (1 - a3) * (1 - a4)) :=
by
  sorry

end simplify_expression_l46_46369


namespace line_equation_l46_46534

-- Define the conditions as given in the problem
def passes_through (P : ℝ × ℝ) (line : ℝ × ℝ) : Prop :=
  line.fst * P.fst + line.snd * P.snd + 1 = 0

def equal_intercepts (line : ℝ × ℝ) : Prop :=
  line.fst = line.snd

theorem line_equation (P : ℝ × ℝ) (hP : P = (-2, -1)) :
  (∃ (k : ℝ), passes_through P (1, -2 * k)) ∨ (∃ (m : ℝ), passes_through P (1, m) ∧ m = - 1) :=
sorry

end line_equation_l46_46534


namespace units_digit_2_pow_2010_5_pow_1004_14_pow_1002_l46_46174

theorem units_digit_2_pow_2010_5_pow_1004_14_pow_1002 :
  (2^2010 * 5^1004 * 14^1002) % 10 = 0 := by
sorry

end units_digit_2_pow_2010_5_pow_1004_14_pow_1002_l46_46174


namespace find_angle_B_l46_46983

def is_triangle (A B C : ℝ) : Prop :=
A + B > C ∧ B + C > A ∧ C + A > B

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Defining the problem conditions
lemma given_condition : 2 * b * Real.cos A = 2 * c - Real.sqrt 3 * a := sorry
-- A triangle with sides a, b, c
lemma triangle_property : is_triangle a b c := sorry

-- The equivalent proof problem
theorem find_angle_B (h_triangle : is_triangle a b c) (h_cond : 2 * b * Real.cos A = 2 * c - Real.sqrt 3 * a) : 
    B = π / 6 := sorry

end find_angle_B_l46_46983


namespace advertisement_revenue_l46_46703

theorem advertisement_revenue
  (cost_per_program : ℝ)
  (num_programs : ℕ)
  (selling_price_per_program : ℝ)
  (desired_profit : ℝ)
  (total_cost_production : ℝ)
  (total_revenue_sales : ℝ)
  (total_revenue_needed : ℝ)
  (revenue_from_advertisements : ℝ) :
  cost_per_program = 0.70 →
  num_programs = 35000 →
  selling_price_per_program = 0.50 →
  desired_profit = 8000 →
  total_cost_production = cost_per_program * num_programs →
  total_revenue_sales = selling_price_per_program * num_programs →
  total_revenue_needed = total_cost_production + desired_profit →
  revenue_from_advertisements = total_revenue_needed - total_revenue_sales →
  revenue_from_advertisements = 15000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end advertisement_revenue_l46_46703


namespace fixed_point_min_value_l46_46423

theorem fixed_point_min_value {a m n : ℝ} (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) (hm_pos : 0 < m) (hn_pos : 0 < n)
  (h : 3 * m + n = 1) : (1 / m + 3 / n) = 12 := sorry

end fixed_point_min_value_l46_46423


namespace cuboid_cutout_l46_46626

theorem cuboid_cutout (x y : ℕ) (h1 : x * y = 36) (h2 : 0 < x) (h3 : x < 4) (h4 : 0 < y) (h5 : y < 15) :
  x + y = 15 :=
sorry

end cuboid_cutout_l46_46626


namespace apples_taken_from_each_basket_l46_46684

theorem apples_taken_from_each_basket (total_apples : ℕ) (baskets : ℕ) (remaining_apples_per_basket : ℕ) 
(h1 : total_apples = 64) (h2 : baskets = 4) (h3 : remaining_apples_per_basket = 13) : 
(total_apples - (remaining_apples_per_basket * baskets)) / baskets = 3 :=
sorry

end apples_taken_from_each_basket_l46_46684


namespace area_of_region_B_l46_46518

noncomputable def region_B_area : ℝ :=
  let square_area := 900
  let excluded_area := 28.125 * Real.pi
  square_area - excluded_area

theorem area_of_region_B : region_B_area = 900 - 28.125 * Real.pi :=
by {
  sorry
}

end area_of_region_B_l46_46518


namespace value_of_b_l46_46352

variable (a b c : ℕ)
variable (h_a_nonzero : a ≠ 0)
variable (h_a : a < 8)
variable (h_b : b < 8)
variable (h_c : c < 8)
variable (h_square : ∃ k, k^2 = a * 8^3 + 3 * 8^2 + b * 8 + c)

theorem value_of_b : b = 1 :=
by sorry

end value_of_b_l46_46352


namespace four_digit_palindromic_squares_with_different_middle_digits_are_zero_l46_46729

theorem four_digit_palindromic_squares_with_different_middle_digits_are_zero :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (∃ k, k * k = n) ∧ (∃ a b, n = 1001 * a + 110 * b) → a ≠ b → false :=
by sorry

end four_digit_palindromic_squares_with_different_middle_digits_are_zero_l46_46729


namespace mean_of_other_two_l46_46211

theorem mean_of_other_two (a b c d e f : ℕ) (h : a = 1867 ∧ b = 1993 ∧ c = 2019 ∧ d = 2025 ∧ e = 2109 ∧ f = 2121):
  ((a + b + c + d + e + f) - (4 * 2008)) / 2 = 2051 := by
  sorry

end mean_of_other_two_l46_46211


namespace subtract_fifteen_result_l46_46680

theorem subtract_fifteen_result (x : ℕ) (h : x / 10 = 6) : x - 15 = 45 :=
by
  sorry

end subtract_fifteen_result_l46_46680


namespace triangle_area_l46_46888

noncomputable def area_triangle (b c angle_C : ℝ) : ℝ :=
  (1 / 2) * b * c * Real.sin angle_C

theorem triangle_area :
  let b := 1
  let c := Real.sqrt 3
  let angle_C := 2 * Real.pi / 3
  area_triangle b c (Real.sin angle_C) = Real.sqrt 3 / 4 :=
by
  sorry

end triangle_area_l46_46888


namespace root_relationship_l46_46435

theorem root_relationship (m n a b : ℝ) 
  (h_eq : ∀ x, 3 - (x - m) * (x - n) = 0 ↔ x = a ∨ x = b) : a < m ∧ m < n ∧ n < b :=
by
  sorry

end root_relationship_l46_46435


namespace medium_size_shoes_initially_stocked_l46_46297

variable {M : ℕ}  -- The number of medium-size shoes initially stocked

noncomputable def initial_pairs_eq (M : ℕ) := 22 + M + 24
noncomputable def shoes_sold (M : ℕ) := initial_pairs_eq M - 13

theorem medium_size_shoes_initially_stocked :
  shoes_sold M = 83 → M = 26 :=
by
  sorry

end medium_size_shoes_initially_stocked_l46_46297


namespace monotonicity_range_of_a_l46_46361

noncomputable def f (x a : ℝ) : ℝ := Real.log x + a * (1 - x)
noncomputable def f' (x a : ℝ) : ℝ := 1 / x - a

-- 1. Monotonicity discussion
theorem monotonicity (a x : ℝ) (h : 0 < x) : 
  (a ≤ 0 → ∀ x, 0 < x → f' x a > 0) ∧
  (a > 0 → (∀ x, 0 < x ∧ x < 1 / a → f' x a > 0) ∧ (∀ x, x > 1 / a → f' x a < 0)) :=
sorry

-- 2. Range of a for maximum value condition
noncomputable def g (a : ℝ) : ℝ := Real.log a + a - 1

theorem range_of_a (a : ℝ) : 
  (0 < a) ∧ (a < 1) ↔ g a < 0 :=
sorry

end monotonicity_range_of_a_l46_46361


namespace range_of_a_l46_46470

noncomputable def f (a x : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 < x → (f a x) > 0) ↔ a ≤ 2 := 
by
  sorry

end range_of_a_l46_46470


namespace train_length_l46_46828

theorem train_length (L : ℕ) :
  (L + 350) / 15 = (L + 500) / 20 → L = 100 := 
by
  intro h
  sorry

end train_length_l46_46828


namespace doctor_lindsay_daily_income_l46_46006

def patients_per_hour_adult : ℕ := 4
def patients_per_hour_child : ℕ := 3
def cost_per_adult : ℕ := 50
def cost_per_child : ℕ := 25
def work_hours_per_day : ℕ := 8

theorem doctor_lindsay_daily_income : 
  (patients_per_hour_adult * cost_per_adult + patients_per_hour_child * cost_per_child) * work_hours_per_day = 2200 := 
by
  sorry

end doctor_lindsay_daily_income_l46_46006


namespace sum_of_squares_of_roots_l46_46791

theorem sum_of_squares_of_roots 
  (r s t : ℝ) 
  (hr : y^3 - 8 * y^2 + 9 * y - 2 = 0) 
  (hs : y ≥ 0) 
  (ht : y ≥ 0):
  r^2 + s^2 + t^2 = 46 :=
sorry

end sum_of_squares_of_roots_l46_46791


namespace jess_height_l46_46886

variable (Jana_height Kelly_height Jess_height : ℕ)

-- Conditions
axiom Jana_height_eq : Jana_height = 74
axiom Jana_taller_than_Kelly : Jana_height = Kelly_height + 5
axiom Kelly_shorter_than_Jess : Kelly_height = Jess_height - 3

-- Prove Jess's height
theorem jess_height : Jess_height = 72 := by
  -- Proof goes here
  sorry

end jess_height_l46_46886


namespace find_Xe_minus_Ye_l46_46345

theorem find_Xe_minus_Ye (e X Y : ℕ) (h1 : 8 < e) (h2 : e^2*X + e*Y + e*X + X + e^2*X + X = 243 * e^2):
  X - Y = (2 * e^2 + 4 * e - 726) / 3 :=
by
  sorry

end find_Xe_minus_Ye_l46_46345


namespace together_work_days_l46_46498

theorem together_work_days (A B C : ℕ) (nine_days : A = 9) (eighteen_days : B = 18) (twelve_days : C = 12) :
  (1 / A + 1 / B + 1 / C) = 1 / 4 :=
by
  sorry

end together_work_days_l46_46498


namespace increasing_function_inv_condition_l46_46844

-- Given a strictly increasing real-valued function f on ℝ with an inverse,
-- satisfying the condition f(x) + f⁻¹(x) = 2x for all x in ℝ,
-- prove that f(x) = x + b, where b is a real constant.

theorem increasing_function_inv_condition (f : ℝ → ℝ) (hf_strict_mono : StrictMono f)
  (hf_inv : ∀ x, f (f⁻¹ x) = x ∧ f⁻¹ (f x) = x)
  (hf_condition : ∀ x, f x + f⁻¹ x = 2 * x) :
  ∃ b : ℝ, ∀ x, f x = x + b :=
sorry

end increasing_function_inv_condition_l46_46844


namespace complex_subtraction_l46_46514

open Complex

def z1 : ℂ := 3 + 4 * I
def z2 : ℂ := 1 + I

theorem complex_subtraction : z1 - z2 = 2 + 3 * I := by
  sorry

end complex_subtraction_l46_46514


namespace machines_working_together_l46_46616

theorem machines_working_together (x : ℝ) :
  let R_time := x + 4
  let Q_time := x + 9
  let P_time := x + 12
  (1 / P_time + 1 / Q_time + 1 / R_time) = 1 / x ↔ x = 1 := 
by
  sorry

end machines_working_together_l46_46616


namespace geometric_sequence_property_l46_46968

noncomputable def geometric_sequence (a : ℕ → ℝ) := 
  ∀ m n : ℕ, a (m + n) = a m * a n / a 0

theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h : geometric_sequence a) 
    (h4 : a 4 = 5) 
    (h8 : a 8 = 6) : 
    a 2 * a 10 = 30 :=
by
  sorry

end geometric_sequence_property_l46_46968


namespace sequence_converges_l46_46346

open Real

theorem sequence_converges (x : ℕ → ℝ) (h₀ : ∀ n, x (n + 1) = 1 + x n - 0.5 * (x n) ^ 2) (h₁ : 1 < x 1 ∧ x 1 < 2) :
  ∀ n ≥ 3, |x n - sqrt 2| < 2 ^ (-n : ℝ) :=
by
  sorry

end sequence_converges_l46_46346


namespace quadratic_sum_l46_46417

theorem quadratic_sum (x : ℝ) (h : x^2 = 16*x - 9) : x = 8 ∨ x = 9 := sorry

end quadratic_sum_l46_46417


namespace mean_transformation_l46_46531

theorem mean_transformation (x1 x2 x3 x4 : ℝ)
                            (h_pos : 0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ 0 < x4)
                            (s2 : ℝ)
                            (h_var : s2 = (1 / 4) * (x1^2 + x2^2 + x3^2 + x4^2 - 16)) :
                            (x1 + 2 + x2 + 2 + x3 + 2 + x4 + 2) / 4 = 4 :=
by
  sorry

end mean_transformation_l46_46531


namespace max_min_values_monotonocity_l46_46209

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 - (1 / 2) * x ^ 2

theorem max_min_values (a : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (ha : a = 1) : 
  f a 0 = 0 ∧ f a 1 = 1 / 2 ∧ f a (1 / 3) = -1 / 54 :=
sorry

theorem monotonocity (a : ℝ) (hx : 0 < x ∧ x < (1 / (6 * a))) (ha : 0 < a) : 
  (3 * a * x ^ 2 - x) < 0 → (f a x) < (f a 0) :=
sorry

end max_min_values_monotonocity_l46_46209


namespace find_x_from_conditions_l46_46899

theorem find_x_from_conditions (a b x y s : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) :
  s = (4 * a)^(4 * b) ∧ s = a^b * y^b ∧ y = 4 * x → x = 64 * a^3 :=
by
  sorry

end find_x_from_conditions_l46_46899


namespace Maria_ate_2_cookies_l46_46980

theorem Maria_ate_2_cookies : 
  ∀ (initial_cookies given_to_friend given_to_family remaining_after_eating : ℕ),
  initial_cookies = 19 →
  given_to_friend = 5 →
  given_to_family = (initial_cookies - given_to_friend) / 2 →
  remaining_after_eating = initial_cookies - given_to_friend - given_to_family - 2 →
  remaining_after_eating = 5 →
  2 = 2 := by
  intros
  sorry

end Maria_ate_2_cookies_l46_46980


namespace value_of_a_l46_46201

theorem value_of_a (a : ℕ) : (∃ (x1 x2 x3 : ℤ),
  abs (abs (x1 - 3) - 1) = a ∧
  abs (abs (x2 - 3) - 1) = a ∧
  abs (abs (x3 - 3) - 1) = a ∧
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3)
  → a = 1 :=
by
  sorry

end value_of_a_l46_46201


namespace exists_indices_l46_46188

-- Define the sequence condition
def is_sequence_of_all_positive_integers (a : ℕ → ℕ) : Prop :=
  (∀ n : ℕ, ∃ m : ℕ, a m = n) ∧ (∀ n m1 m2 : ℕ, a m1 = n ∧ a m2 = n → m1 = m2)

-- Main theorem statement
theorem exists_indices 
  (a : ℕ → ℕ) 
  (h : is_sequence_of_all_positive_integers a) :
  ∃ (ℓ m : ℕ), 1 < ℓ ∧ ℓ < m ∧ (a 0 + a m = 2 * a ℓ) :=
by
  sorry

end exists_indices_l46_46188


namespace black_pieces_more_than_white_l46_46128

theorem black_pieces_more_than_white (B W : ℕ) 
  (h₁ : (B - 1) * 7 = 9 * W)
  (h₂ : B * 5 = 7 * (W - 1)) :
  B - W = 7 :=
sorry

end black_pieces_more_than_white_l46_46128


namespace min_workers_needed_to_make_profit_l46_46688

def wage_per_worker_per_hour := 20
def fixed_cost := 800
def units_per_worker_per_hour := 6
def price_per_unit := 4.5
def hours_per_workday := 9

theorem min_workers_needed_to_make_profit : ∃ (n : ℕ), 243 * n > 800 + 180 * n ∧ n ≥ 13 :=
by
  sorry

end min_workers_needed_to_make_profit_l46_46688


namespace restaurant_cooks_l46_46572

variable (C W : ℕ)

theorem restaurant_cooks : 
  (C / W = 3 / 10) ∧ (C / (W + 12) = 3 / 14) → C = 9 :=
by sorry

end restaurant_cooks_l46_46572


namespace pet_shop_dogs_l46_46325

theorem pet_shop_dogs (D C B : ℕ) (x : ℕ) (h1 : D = 3 * x) (h2 : C = 5 * x) (h3 : B = 9 * x) (h4 : D + B = 204) : D = 51 := by
  -- omitted proof
  sorry

end pet_shop_dogs_l46_46325


namespace trig_identity_l46_46716

theorem trig_identity (x : ℝ) (h : 3 * Real.sin x + Real.cos x = 0) :
  Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + Real.cos x ^ 2 = 2 / 5 :=
sorry

end trig_identity_l46_46716


namespace inequality_proofs_l46_46283

def sinSumInequality (A B C ε : ℝ) : Prop :=
  ε * (Real.sin A + Real.sin B + Real.sin C) ≤ Real.sin A * Real.sin B * Real.sin C + 1 + ε^3

def sinProductInequality (A B C ε : ℝ) : Prop :=
  (1 + ε + Real.sin A) * (1 + ε + Real.sin B) * (1 + ε + Real.sin C) ≥ 9 * ε * (Real.sin A + Real.sin B + Real.sin C)

theorem inequality_proofs (A B C ε : ℝ) (hA : 0 ≤ A ∧ A ≤ Real.pi) (hB : 0 ≤ B ∧ B ≤ Real.pi) 
  (hC : 0 ≤ C ∧ C ≤ Real.pi) (hε : ε ≥ 1) :
  sinSumInequality A B C ε ∧ sinProductInequality A B C ε :=
by
  sorry

end inequality_proofs_l46_46283


namespace cheryl_used_total_material_correct_amount_l46_46123

def material_used (initial leftover : ℚ) : ℚ := initial - leftover

def total_material_used 
  (initial_a initial_b initial_c leftover_a leftover_b leftover_c : ℚ) : ℚ :=
  material_used initial_a leftover_a + material_used initial_b leftover_b + material_used initial_c leftover_c

theorem cheryl_used_total_material_correct_amount :
  total_material_used (2/9) (1/8) (3/10) (4/18) (1/12) (3/15) = 17/120 :=
by
  sorry

end cheryl_used_total_material_correct_amount_l46_46123


namespace teresa_age_when_michiko_born_l46_46344

def conditions (T M Michiko K Yuki : ℕ) : Prop := 
  T = 59 ∧ 
  M = 71 ∧ 
  M - Michiko = 38 ∧ 
  K = Michiko - 4 ∧ 
  Yuki = K - 3 ∧ 
  (Yuki + 3) - (26 - 25) = 25

theorem teresa_age_when_michiko_born :
  ∃ T M Michiko K Yuki, conditions T M Michiko K Yuki → T - Michiko = 26 :=
  by
  sorry

end teresa_age_when_michiko_born_l46_46344


namespace maximum_value_of_expression_l46_46987

theorem maximum_value_of_expression (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  (a^3 + b^3 + c^3) / ((a + b + c)^3 - 26 * a * b * c) ≤ 3 := by
  sorry

end maximum_value_of_expression_l46_46987


namespace negation_of_p_l46_46363

namespace ProofProblem

variable (x : ℝ)

def p : Prop := ∃ x : ℝ, x^2 + x - 1 ≥ 0

def neg_p : Prop := ∀ x : ℝ, x^2 + x - 1 < 0

theorem negation_of_p : ¬p = neg_p := sorry

end ProofProblem

end negation_of_p_l46_46363


namespace value_of_x_l46_46753

-- Define the custom operation * for the problem
def star (a b : ℝ) : ℝ := 4 * a - 2 * b

-- Define the main problem statement
theorem value_of_x (x : ℝ) (h : star 3 (star 7 x) = 5) : x = 49 / 4 :=
by
  have h7x : star 7 x = 28 - 2 * x := by sorry  -- Derived from the definitions
  have h3star7x : star 3 (28 - 2 * x) = -44 + 4 * x := by sorry  -- Derived from substituting star 7 x
  sorry

end value_of_x_l46_46753


namespace find_ab_range_m_l46_46904

-- Part 1
theorem find_ab (a b: ℝ) (h1 : 3 - 6 * a + b = 0) (h2 : -1 + 3 * a - b + a^2 = 0) :
  a = 2 ∧ b = 9 := 
sorry

-- Part 2
theorem range_m (m: ℝ) (h: ∀ x ∈ (Set.Icc (-2) 1), x^3 + 3 * 2 * x^2 + 9 * x + 4 - m ≤ 0) :
  20 ≤ m :=
sorry

end find_ab_range_m_l46_46904


namespace function_intersection_at_most_one_l46_46504

theorem function_intersection_at_most_one (f : ℝ → ℝ) (a : ℝ) :
  ∃! b, f b = a := sorry

end function_intersection_at_most_one_l46_46504


namespace dave_spent_102_dollars_l46_46587

noncomputable def total_cost (books_animals books_space books_trains cost_per_book : ℕ) : ℕ :=
  (books_animals + books_space + books_trains) * cost_per_book

theorem dave_spent_102_dollars :
  total_cost 8 6 3 6 = 102 := by
  sorry

end dave_spent_102_dollars_l46_46587


namespace trajectory_is_plane_l46_46030

/--
Given that the vertical coordinate of a moving point P is always 2, 
prove that the trajectory of the moving point P forms a plane in a 
three-dimensional Cartesian coordinate system.
-/
theorem trajectory_is_plane (P : ℝ × ℝ × ℝ) (hP : ∀ t : ℝ, ∃ x y, P = (x, y, 2)) :
  ∃ a b c d, a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∧ (∀ x y, ∃ z, (a * x + b * y + c * z + d = 0) ∧ z = 2) :=
by
  -- This proof should show that there exist constants a, b, c, and d such that 
  -- the given equation represents a plane and the z-coordinate is always 2.
  sorry

end trajectory_is_plane_l46_46030


namespace relationship_between_p_and_q_l46_46245

variables {x y : ℝ}

def p (x y : ℝ) := (x^2 + y^2) * (x - y)
def q (x y : ℝ) := (x^2 - y^2) * (x + y)

theorem relationship_between_p_and_q (h1 : x < y) (h2 : y < 0) : p x y > q x y := 
  by sorry

end relationship_between_p_and_q_l46_46245


namespace width_of_first_sheet_paper_l46_46137

theorem width_of_first_sheet_paper :
  ∀ (w : ℝ),
  2 * 11 * w = 2 * 4.5 * 11 + 100 → 
  w = 199 / 22 := 
by
  intro w
  intro h
  sorry

end width_of_first_sheet_paper_l46_46137


namespace problem_statement_l46_46063

theorem problem_statement :
  ∀ k : Nat, (∃ r s : Nat, r > 0 ∧ s > 0 ∧ (k^2 - 6 * k + 11)^(r - 1) = (2 * k - 7)^s) ↔ (k = 2 ∨ k = 3 ∨ k = 4 ∨ k = 8) :=
by
  sorry

end problem_statement_l46_46063


namespace sum_of_integers_l46_46618

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 14) (h3 : x * y = 180) :
  x + y = 2 * Int.sqrt 229 :=
sorry

end sum_of_integers_l46_46618


namespace min_value_a_squared_ab_b_squared_l46_46876

theorem min_value_a_squared_ab_b_squared {a b t p : ℝ} (h1 : a + b = t) (h2 : ab = p) :
  a^2 + ab + b^2 ≥ 3 * t^2 / 4 := by
  sorry

end min_value_a_squared_ab_b_squared_l46_46876


namespace diff_of_cubes_is_sum_of_squares_l46_46503

theorem diff_of_cubes_is_sum_of_squares (n : ℤ) : 
  (n+2)^3 - n^3 = n^2 + (n+2)^2 + (2*n+2)^2 := 
by sorry

end diff_of_cubes_is_sum_of_squares_l46_46503


namespace train_length_proof_l46_46988

noncomputable def length_of_train : ℝ := 450.09

theorem train_length_proof
  (speed_kmh : ℝ := 60)
  (time_s : ℝ := 27) :
  (speed_kmh * (5 / 18) * time_s = length_of_train) :=
by
  sorry

end train_length_proof_l46_46988


namespace total_crackers_l46_46250

-- Define the conditions
def boxes_Darren := 4
def crackers_per_box := 24
def boxes_Calvin := 2 * boxes_Darren - 1

-- Define the mathematical proof problem
theorem total_crackers : 
  let total_Darren := boxes_Darren * crackers_per_box
  let total_Calvin := boxes_Calvin * crackers_per_box
  total_Darren + total_Calvin = 264 :=
by
  sorry

end total_crackers_l46_46250


namespace pamela_skittles_correct_l46_46836

def pamela_initial_skittles := 50
def pamela_gives_skittles_to_karen := 7
def pamela_receives_skittles_from_kevin := 3
def pamela_shares_percentage := 20

def pamela_final_skittles : Nat :=
  let after_giving := pamela_initial_skittles - pamela_gives_skittles_to_karen
  let after_receiving := after_giving + pamela_receives_skittles_from_kevin
  let share_amount := (after_receiving * pamela_shares_percentage) / 100
  let rounded_share := Nat.floor share_amount
  let final_count := after_receiving - rounded_share
  final_count

theorem pamela_skittles_correct :
  pamela_final_skittles = 37 := by
  sorry

end pamela_skittles_correct_l46_46836


namespace population_reaches_210_l46_46035

noncomputable def population_function (x : ℕ) : ℝ :=
  200 * (1 + 0.01)^x

theorem population_reaches_210 :
  ∃ x : ℕ, population_function x >= 210 :=
by
  existsi 5
  apply le_of_lt
  sorry

end population_reaches_210_l46_46035


namespace bus_stoppage_time_l46_46999

theorem bus_stoppage_time (speed_excl_stoppages speed_incl_stoppages : ℕ) (h1 : speed_excl_stoppages = 54) (h2 : speed_incl_stoppages = 45) : 
  ∃ (t : ℕ), t = 10 := by
  sorry

end bus_stoppage_time_l46_46999


namespace fraction_comparison_l46_46634

theorem fraction_comparison (a b : ℝ) (h : a > b ∧ b > 0) : 
  (a / b) > (a + 1) / (b + 1) :=
by
  sorry

end fraction_comparison_l46_46634


namespace correct_option_is_A_l46_46714

-- Define the options as terms
def optionA (x : ℝ) := (1/2) * x - 5 * x = 18
def optionB (x : ℝ) := (1/2) * x > 5 * x - 1
def optionC (y : ℝ) := 8 * y - 4
def optionD := 5 - 2 = 3

-- Define a function to check if an option is an equation
def is_equation (option : Prop) : Prop :=
  ∃ (x : ℝ), option = ((1/2) * x - 5 * x = 18)

-- Prove that optionA is the equation
theorem correct_option_is_A : is_equation (optionA x) :=
by
  sorry

end correct_option_is_A_l46_46714


namespace chord_bisected_by_point_l46_46011

theorem chord_bisected_by_point (x1 y1 x2 y2 : ℝ) :
  (x1^2 / 36 + y1^2 / 9 = 1) ∧ (x2^2 / 36 + y2^2 / 9 = 1) ∧ 
  (x1 + x2 = 4) ∧ (y1 + y2 = 4) → (x + 4 * y - 10 = 0) :=
sorry

end chord_bisected_by_point_l46_46011


namespace program1_values_program2_values_l46_46260

theorem program1_values :
  ∃ (a b c : ℤ), a = 3 ∧ b = -5 ∧ c = 8 ∧
  a = b ∧ b = c ∧
  a = -5 ∧ b = 8 ∧ c = 8 :=
by sorry

theorem program2_values :
  ∃ (a b c : ℤ), a = 3 ∧ b = -5 ∧ c = 8 ∧
  a = b ∧ b = c ∧ c = a ∧
  a = -5 ∧ b = 8 ∧ c = -5 :=
by sorry

end program1_values_program2_values_l46_46260


namespace find_m_l46_46613

def line_eq (x y : ℝ) : Prop := x + 2 * y - 3 = 0

def circle_eq (x y m : ℝ) : Prop := x * x + y * y + x - 6 * y + m = 0

def perpendicular_vectors (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

theorem find_m (m : ℝ) :
  (∃ (x y : ℝ), line_eq x y ∧ line_eq (3 - 2 * y) y ∧ circle_eq x y m ∧ circle_eq (3 - 2 * y) y m) ∧
  (∃ (x1 y1 x2 y2 : ℝ), line_eq x1 y1 ∧ line_eq x2 y2 ∧ perpendicular_vectors x1 y1 x2 y2) → m = 3 :=
sorry

end find_m_l46_46613


namespace a_2018_mod_49_l46_46831

def a (n : ℕ) : ℕ := 6^n + 8^n

theorem a_2018_mod_49 : (a 2018) % 49 = 0 := by
  sorry

end a_2018_mod_49_l46_46831


namespace julian_initial_owing_l46_46893

theorem julian_initial_owing (jenny_owing_initial: ℕ) (borrow: ℕ) (total_owing: ℕ):
    borrow = 8 → total_owing = 28 → jenny_owing_initial + borrow = total_owing → jenny_owing_initial = 20 :=
by intros;
   exact sorry

end julian_initial_owing_l46_46893


namespace gnuff_tutor_minutes_l46_46691

/-- Definitions of the given conditions -/
def flat_rate : ℕ := 20
def per_minute_charge : ℕ := 7
def total_paid : ℕ := 146

/-- The proof statement -/
theorem gnuff_tutor_minutes :
  (total_paid - flat_rate) / per_minute_charge = 18 :=
by
  sorry

end gnuff_tutor_minutes_l46_46691


namespace largest_integer_modulo_l46_46114

theorem largest_integer_modulo (a : ℤ) : a < 93 ∧ a % 7 = 4 ∧ (∀ b : ℤ, b < 93 ∧ b % 7 = 4 → b ≤ a) ↔ a = 88 :=
by
    sorry

end largest_integer_modulo_l46_46114


namespace transformed_function_equivalence_l46_46709

-- Define the original function
def original_function (x : ℝ) : ℝ := 2 * x + 1

-- Define the transformation involving shifting 2 units to the right
def transformed_function (x : ℝ) : ℝ := original_function (x - 2)

-- The theorem we want to prove
theorem transformed_function_equivalence : 
  ∀ x : ℝ, transformed_function x = 2 * x - 3 :=
by
  sorry

end transformed_function_equivalence_l46_46709


namespace triangle_area_is_correct_l46_46452

structure Point where
  x : ℝ
  y : ℝ

noncomputable def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * (abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)))

def A : Point := ⟨0, 0⟩
def B : Point := ⟨2, 2⟩
def C : Point := ⟨2, 0⟩

theorem triangle_area_is_correct : area_of_triangle A B C = 2 := by
  sorry

end triangle_area_is_correct_l46_46452


namespace chocolate_bars_in_large_box_l46_46008

def num_small_boxes : ℕ := 17
def chocolate_bars_per_small_box : ℕ := 26
def total_chocolate_bars : ℕ := 17 * 26

theorem chocolate_bars_in_large_box :
  total_chocolate_bars = 442 :=
by
  sorry

end chocolate_bars_in_large_box_l46_46008


namespace average_a_b_l46_46973

-- Defining the variables A, B, C
variables (A B C : ℝ)

-- Given conditions
def condition1 : Prop := (A + B + C) / 3 = 45
def condition2 : Prop := (B + C) / 2 = 43
def condition3 : Prop := B = 31

-- The theorem stating that the average weight of a and b is 40 kg
theorem average_a_b (h1 : condition1 A B C) (h2 : condition2 B C) (h3 : condition3 B) : (A + B) / 2 = 40 :=
sorry

end average_a_b_l46_46973


namespace intersection_x_value_l46_46045

theorem intersection_x_value : ∃ x y : ℝ, y = 3 * x + 7 ∧ 3 * x - 2 * y = -4 ∧ x = -10 / 3 :=
by
  sorry

end intersection_x_value_l46_46045


namespace combined_total_circles_squares_l46_46775

-- Define the problem parameters based on conditions
def US_stars : ℕ := 50
def US_stripes : ℕ := 13
def circles (n : ℕ) : ℕ := (n / 2) - 3
def squares (n : ℕ) : ℕ := (n * 2) + 6

-- Prove that the combined number of circles and squares on Pete's flag is 54
theorem combined_total_circles_squares : 
    circles US_stars + squares US_stripes = 54 := by
  sorry

end combined_total_circles_squares_l46_46775


namespace min_frac_sum_min_frac_sum_achieved_l46_46773

theorem min_frac_sum (a b : ℝ) (h₁ : 2 * a + 3 * b = 6) (h₂ : 0 < a) (h₃ : 0 < b) :
  (2 / a + 3 / b) ≥ 25 / 6 :=
by sorry

theorem min_frac_sum_achieved :
  (2 / (6 / 5) + 3 / (6 / 5)) = 25 / 6 :=
by sorry


end min_frac_sum_min_frac_sum_achieved_l46_46773


namespace factorize_first_poly_factorize_second_poly_l46_46251

variable (x m n : ℝ)

-- Proof statement for the first polynomial
theorem factorize_first_poly : x^2 + 14*x + 49 = (x + 7)^2 := 
by sorry

-- Proof statement for the second polynomial
theorem factorize_second_poly : (m - 1) + n^2 * (1 - m) = (m - 1) * (1 - n) * (1 + n) := 
by sorry

end factorize_first_poly_factorize_second_poly_l46_46251


namespace Dan_reaches_Cate_in_25_seconds_l46_46553

theorem Dan_reaches_Cate_in_25_seconds
  (d : ℝ) (v_d : ℝ) (v_c : ℝ)
  (h1 : d = 50)
  (h2 : v_d = 8)
  (h3 : v_c = 6) :
  (d / (v_d - v_c) = 25) :=
by
  sorry

end Dan_reaches_Cate_in_25_seconds_l46_46553


namespace number_of_zeros_of_f_l46_46726

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem number_of_zeros_of_f :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end number_of_zeros_of_f_l46_46726


namespace proof_x_bounds_l46_46991

noncomputable def x : ℝ :=
  1 / Real.logb (1 / 3) (1 / 2) +
  1 / Real.logb (1 / 3) (1 / 4) +
  1 / Real.logb 7 (1 / 8)

theorem proof_x_bounds : 3 < x ∧ x < 3.5 := 
by
  sorry

end proof_x_bounds_l46_46991


namespace max_non_attacking_mammonths_is_20_l46_46460

def mamonth_attacking_diagonal_count (b: board) (m: mamonth): ℕ := 
    sorry -- define the function to count attacking diagonals of a given mammoth on the board

def max_non_attacking_mamonths_board (b: board) : ℕ :=
    sorry -- function to calculate max non-attacking mammonths given a board setup

theorem max_non_attacking_mammonths_is_20 : 
  ∀ (b : board), (max_non_attacking_mamonths_board b) ≤ 20 :=
by
  sorry

end max_non_attacking_mammonths_is_20_l46_46460


namespace arithmetic_mean_a8_a11_l46_46976

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem arithmetic_mean_a8_a11 {a : ℕ → ℝ} (h1 : geometric_sequence a (-2)) 
    (h2 : a 2 * a 6 = 4 * a 3) :
  ((a 7 + a 10) / 2) = -56 :=
sorry

end arithmetic_mean_a8_a11_l46_46976


namespace condition_is_sufficient_but_not_necessary_l46_46731

variable (P Q : Prop)

theorem condition_is_sufficient_but_not_necessary :
    (P → Q) ∧ ¬(Q → P) :=
sorry

end condition_is_sufficient_but_not_necessary_l46_46731


namespace cost_difference_is_35_88_usd_l46_46761

/-
  Mr. Llesis bought 50 kilograms of rice at different prices per kilogram from various suppliers.
  He bought:
  - 15 kilograms at €1.2 per kilogram from Supplier A
  - 10 kilograms at €1.4 per kilogram from Supplier B
  - 12 kilograms at €1.6 per kilogram from Supplier C
  - 8 kilograms at €1.9 per kilogram from Supplier D
  - 5 kilograms at €2.3 per kilogram from Supplier E

  He kept 7/10 of the total rice in storage and gave the rest to Mr. Everest.
  The current conversion rate is €1 = $1.15.
  
  Prove that the difference in cost in US dollars between the rice kept and the rice given away is $35.88.
-/

def euros_to_usd (euros : ℚ) : ℚ :=
  euros * (115 / 100)

def total_cost : ℚ := 
  (15 * 1.2) + (10 * 1.4) + (12 * 1.6) + (8 * 1.9) + (5 * 2.3)

def cost_kept : ℚ := (7/10) * total_cost
def cost_given : ℚ := (3/10) * total_cost

theorem cost_difference_is_35_88_usd :
  euros_to_usd cost_kept - euros_to_usd cost_given = 35.88 := 
sorry

end cost_difference_is_35_88_usd_l46_46761


namespace no_two_perfect_cubes_between_two_perfect_squares_l46_46737

theorem no_two_perfect_cubes_between_two_perfect_squares :
  ∀ n a b : ℤ, n^2 < a^3 ∧ a^3 < b^3 ∧ b^3 < (n + 1)^2 → False :=
by 
  sorry

end no_two_perfect_cubes_between_two_perfect_squares_l46_46737


namespace probability_difference_l46_46054

noncomputable def Ps (red black : ℕ) : ℚ :=
  let total := red + black
  (red * (red - 1) + black * (black - 1)) / (total * (total - 1))

noncomputable def Pd (red black : ℕ) : ℚ :=
  let total := red + black
  (red * black * 2) / (total * (total - 1))

noncomputable def abs_diff (Ps Pd : ℚ) : ℚ :=
  |Ps - Pd|

theorem probability_difference :
  let red := 1200
  let black := 800
  let total := red + black
  abs_diff (Ps red black) (Pd red black) = 789 / 19990 := by
  sorry

end probability_difference_l46_46054


namespace total_pages_book_l46_46993

-- Define the conditions
def reading_speed1 : ℕ := 10 -- pages per day for first half
def reading_speed2 : ℕ := 5 -- pages per day for second half
def total_days : ℕ := 75 -- total days spent reading

-- This is the main theorem we seek to prove:
theorem total_pages_book (P : ℕ) 
  (h1 : ∃ D1 D2 : ℕ, D1 + D2 = total_days ∧ D1 * reading_speed1 = P / 2 ∧ D2 * reading_speed2 = P / 2) : 
  P = 500 :=
by
  sorry

end total_pages_book_l46_46993


namespace union_of_A_and_B_l46_46610

open Set

variable (A B : Set ℤ)

theorem union_of_A_and_B (hA : A = {0, 1}) (hB : B = {0, -1}) : A ∪ B = {-1, 0, 1} := by
  sorry

end union_of_A_and_B_l46_46610


namespace total_letters_l46_46168

theorem total_letters (brother_letters : ℕ) (greta_more_than_brother : ℕ) (mother_multiple : ℕ)
  (h_brother : brother_letters = 40)
  (h_greta : ∀ (brother_letters greta_letters : ℕ), greta_letters = brother_letters + greta_more_than_brother)
  (h_mother : ∀ (total_letters mother_letters : ℕ), mother_letters = mother_multiple * total_letters) :
  brother_letters + (brother_letters + greta_more_than_brother) + (mother_multiple * (brother_letters + (brother_letters + greta_more_than_brother))) = 270 :=
by
  sorry

end total_letters_l46_46168


namespace boxes_filled_l46_46132

noncomputable def bags_per_box := 6
noncomputable def balls_per_bag := 8
noncomputable def total_balls := 720

theorem boxes_filled (h1 : balls_per_bag = 8) (h2 : bags_per_box = 6) (h3 : total_balls = 720) :
  (total_balls / balls_per_bag) / bags_per_box = 15 :=
by
  sorry

end boxes_filled_l46_46132


namespace pythagorean_triplet_l46_46318

theorem pythagorean_triplet (k : ℕ) :
  let a := k
  let b := 2 * k - 2
  let c := 2 * k - 1
  (a * b) ^ 2 + c ^ 2 = (2 * k ^ 2 - 2 * k + 1) ^ 2 :=
by
  sorry

end pythagorean_triplet_l46_46318


namespace absolute_difference_rectangle_l46_46366

theorem absolute_difference_rectangle 
  (x y r k : ℝ)
  (h1 : 2 * x + 2 * y = 4 * r)
  (h2 : (x^2 + y^2) = (k * x)^2) :
  |x - y| = k * x :=
by
  sorry

end absolute_difference_rectangle_l46_46366


namespace average_effective_increase_correct_l46_46290

noncomputable def effective_increase (initial_price: ℕ) (price_increase_percent: ℕ) (discount_percent: ℕ) : ℕ :=
let increased_price := initial_price + (initial_price * price_increase_percent / 100)
let final_price := increased_price - (increased_price * discount_percent / 100)
(final_price - initial_price) * 100 / initial_price

noncomputable def average_effective_increase : ℕ :=
let increase1 := effective_increase 300 10 5
let increase2 := effective_increase 450 15 7
let increase3 := effective_increase 600 20 10
(increase1 + increase2 + increase3) / 3

theorem average_effective_increase_correct :
  average_effective_increase = 6483 / 100 :=
by
  sorry

end average_effective_increase_correct_l46_46290


namespace john_paid_more_than_jane_l46_46751

theorem john_paid_more_than_jane :
    let original_price : ℝ := 40.00
    let discount_percentage : ℝ := 0.10
    let tip_percentage : ℝ := 0.15
    let discounted_price : ℝ := original_price - (discount_percentage * original_price)
    let john_tip : ℝ := tip_percentage * original_price
    let john_total : ℝ := discounted_price + john_tip
    let jane_tip : ℝ := tip_percentage * discounted_price
    let jane_total : ℝ := discounted_price + jane_tip
    let difference : ℝ := john_total - jane_total
    difference = 0.60 :=
by
  sorry

end john_paid_more_than_jane_l46_46751


namespace range_of_a_l46_46838

-- Define the conditions and the problem
def neg_p (x : ℝ) : Prop := -3 < x ∧ x < 0
def neg_q (x : ℝ) (a : ℝ) : Prop := x > a
def p (x : ℝ) : Prop := x ≤ -3 ∨ x ≥ 0
def q (x : ℝ) (a : ℝ) : Prop := x ≤ a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, neg_p x → ¬ p x) ∧
  (∀ x : ℝ, neg_q x a → ¬ q x a) ∧
  (∀ x : ℝ, q x a → p x) ∧
  (∃ x : ℝ, ¬ (q x a → p x)) →
  a ≤ -3 :=
by
  sorry

end range_of_a_l46_46838


namespace min_sum_of_segments_is_305_l46_46861

noncomputable def min_sum_of_segments : ℕ := 
  let a : ℕ := 3
  let b : ℕ := 5
  100 * a + b

theorem min_sum_of_segments_is_305 : min_sum_of_segments = 305 := by
  sorry

end min_sum_of_segments_is_305_l46_46861


namespace circle_radius_l46_46389

-- Define the main geometric scenario in Lean 4
theorem circle_radius 
  (O P A B : Type) 
  (r OP PA PB : ℝ)
  (h1 : PA * PB = 24)
  (h2 : OP = 5)
  : r = 7 
:= sorry

end circle_radius_l46_46389


namespace b_finishes_remaining_work_correct_time_for_b_l46_46914

theorem b_finishes_remaining_work (a_days : ℝ) (b_days : ℝ) (work_together_days : ℝ) (remaining_work_after : ℝ) : ℝ :=
  let a_work_rate := 1 / a_days
  let b_work_rate := 1 / b_days
  let combined_work_per_day := a_work_rate + b_work_rate
  let work_done_together := combined_work_per_day * work_together_days
  let remaining_work := 1 - work_done_together
  let b_completion_time := remaining_work / b_work_rate
  b_completion_time

theorem correct_time_for_b : b_finishes_remaining_work 2 6 1 (1 - 2/3) = 2 := 
by sorry

end b_finishes_remaining_work_correct_time_for_b_l46_46914


namespace highest_score_is_96_l46_46158

theorem highest_score_is_96 :
  let standard_score := 85
  let deviations := [-9, -4, 11, -7, 0]
  let actual_scores := deviations.map (λ x => standard_score + x)
  actual_scores.maximum = 96 :=
by
  sorry

end highest_score_is_96_l46_46158


namespace find_A_l46_46740

theorem find_A (A M C : Nat) (h1 : (10 * A^2 + 10 * M + C) * (A + M^2 + C^2) = 1050) (h2 : A < 10) (h3 : M < 10) (h4 : C < 10) : A = 2 := by
  sorry

end find_A_l46_46740


namespace john_running_time_l46_46884

theorem john_running_time
  (x : ℚ)
  (h1 : 15 * x + 10 * (9 - x) = 100)
  (h2 : 0 ≤ x)
  (h3 : x ≤ 9) :
  x = 2 := by
  sorry

end john_running_time_l46_46884


namespace parabola_properties_and_intersection_l46_46694

-- Definition of the parabola C: y^2 = -4x
def parabola_C (x y : ℝ) : Prop := y^2 = -4 * x

-- Focus of the parabola
def focus_C : ℝ × ℝ := (-1, 0)

-- Equation of the directrix
def directrix_C (x: ℝ): Prop := x = 1

-- Distance from the focus to the directrix
def distance_focus_to_directrix : ℝ := 2

-- Line l passing through P(1, 2) with slope k
def line_l (k x y : ℝ) : Prop := y = k * x - k + 2

-- Main theorem statement
theorem parabola_properties_and_intersection (k: ℝ) :
  (focus_C = (-1, 0)) ∧
  (∀ x, directrix_C x ↔ x = 1) ∧
  (distance_focus_to_directrix = 2) ∧
  ((k = 1 + Real.sqrt 2 ∨ k = 1 - Real.sqrt 2) →
    ∃ x y, parabola_C x y ∧ line_l k x y ∧
    (∀ x' y', parabola_C x' y' ∧ line_l k x' y' → x = x' ∧ y = y')) ∧
  ((1 - Real.sqrt 2 < k ∧ k < 1 + Real.sqrt 2) →
    ∃ x y x' y', x ≠ x' ∧ y ≠ y' ∧
    parabola_C x y ∧ line_l k x y ∧
    parabola_C x' y' ∧ line_l k x' y') ∧
  ((k > 1 + Real.sqrt 2 ∨ k < 1 - Real.sqrt 2) →
    ∀ x y, ¬(parabola_C x y ∧ line_l k x y)) :=
by sorry

end parabola_properties_and_intersection_l46_46694


namespace temperature_range_l46_46268

-- Conditions: highest temperature and lowest temperature
def highest_temp : ℝ := 5
def lowest_temp : ℝ := -2
variable (t : ℝ) -- given temperature on February 1, 2018

-- Proof problem statement
theorem temperature_range : lowest_temp ≤ t ∧ t ≤ highest_temp :=
sorry

end temperature_range_l46_46268


namespace Ahmad_eight_steps_l46_46349

def reach_top (n : Nat) (holes : List Nat) : Nat := sorry

theorem Ahmad_eight_steps (h : reach_top 8 [6] = 8) : True := by 
  trivial

end Ahmad_eight_steps_l46_46349


namespace triangle_base_length_l46_46639

theorem triangle_base_length (A h b : ℝ) 
  (h1 : A = 30) 
  (h2 : h = 5) 
  (h3 : A = (b * h) / 2) : 
  b = 12 :=
by
  sorry

end triangle_base_length_l46_46639


namespace sandy_total_earnings_l46_46923

-- Define the conditions
def hourly_wage : ℕ := 15
def hours_friday : ℕ := 10
def hours_saturday : ℕ := 6
def hours_sunday : ℕ := 14

-- Define the total hours worked and total earnings
def total_hours := hours_friday + hours_saturday + hours_sunday
def total_earnings := total_hours * hourly_wage

-- State the theorem
theorem sandy_total_earnings : total_earnings = 450 := by
  sorry

end sandy_total_earnings_l46_46923


namespace find_S6_l46_46256

-- sum of the first n terms of an arithmetic sequence
variable (S : ℕ → ℕ)

-- Given conditions
axiom S_2_eq_3 : S 2 = 3
axiom S_4_eq_15 : S 4 = 15

-- Theorem statement
theorem find_S6 : S 6 = 36 := sorry

end find_S6_l46_46256


namespace disproving_proposition_l46_46467

theorem disproving_proposition : ∃ (angle1 angle2 : ℝ), angle1 = angle2 ∧ angle1 + angle2 = 90 :=
by
  sorry

end disproving_proposition_l46_46467


namespace brenda_peaches_remaining_l46_46408

theorem brenda_peaches_remaining (total_peaches : ℕ) (percent_fresh : ℚ) (thrown_away : ℕ) (fresh_peaches : ℕ) (remaining_peaches : ℕ) :
    total_peaches = 250 → 
    percent_fresh = 0.60 → 
    thrown_away = 15 → 
    fresh_peaches = total_peaches * percent_fresh → 
    remaining_peaches = fresh_peaches - thrown_away → 
    remaining_peaches = 135 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end brenda_peaches_remaining_l46_46408


namespace sum_of_m_and_n_l46_46604

theorem sum_of_m_and_n (m n : ℝ) (h : m^2 + n^2 - 6 * m + 10 * n + 34 = 0) : m + n = -2 := 
sorry

end sum_of_m_and_n_l46_46604


namespace jerry_more_votes_l46_46739

-- Definitions based on conditions
def total_votes : ℕ := 196554
def jerry_votes : ℕ := 108375
def john_votes : ℕ := total_votes - jerry_votes

-- Theorem to prove the number of more votes Jerry received than John Pavich
theorem jerry_more_votes : jerry_votes - john_votes = 20196 :=
by
  -- Definitions and proof can be filled out here as required.
  sorry

end jerry_more_votes_l46_46739


namespace abs_sub_self_nonneg_l46_46098

theorem abs_sub_self_nonneg (m : ℚ) : |m| - m ≥ 0 := 
sorry

end abs_sub_self_nonneg_l46_46098


namespace general_term_a_n_l46_46772

theorem general_term_a_n (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hSn : ∀ n, S n = (2/3) * a n + 1/3) :
  ∀ n, a n = (-2)^(n-1) :=
sorry

end general_term_a_n_l46_46772


namespace ratio_average_speed_l46_46792

-- Define the conditions based on given distances and times
def distanceAB : ℕ := 600
def timeAB : ℕ := 3

def distanceBD : ℕ := 540
def timeBD : ℕ := 2

def distanceAC : ℕ := 460
def timeAC : ℕ := 4

def distanceCE : ℕ := 380
def timeCE : ℕ := 3

-- Define the total distances and times for Eddy and Freddy
def distanceEddy : ℕ := distanceAB + distanceBD
def timeEddy : ℕ := timeAB + timeBD

def distanceFreddy : ℕ := distanceAC + distanceCE
def timeFreddy : ℕ := timeAC + timeCE

-- Define the average speeds for Eddy and Freddy
def averageSpeedEddy : ℚ := distanceEddy / timeEddy
def averageSpeedFreddy : ℚ := distanceFreddy / timeFreddy

-- Prove the ratio of their average speeds is 19:10
theorem ratio_average_speed (h1 : distanceAB = 600) (h2 : timeAB = 3) 
                           (h3 : distanceBD = 540) (h4 : timeBD = 2)
                           (h5 : distanceAC = 460) (h6 : timeAC = 4) 
                           (h7 : distanceCE = 380) (h8 : timeCE = 3):
  averageSpeedEddy / averageSpeedFreddy = 19 / 10 := by sorry

end ratio_average_speed_l46_46792


namespace geometric_sequence_sum_ratio_l46_46399

noncomputable def geometric_sum (a q : ℝ) (n : ℕ) : ℝ := a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_ratio (a q : ℝ) (h : a * q^2 = 8 * a * q^5) :
  (geometric_sum a q 4) / (geometric_sum a q 2) = 5 / 4 :=
by
  -- The proof will go here.
  sorry

end geometric_sequence_sum_ratio_l46_46399


namespace simplify_P_eq_l46_46263

noncomputable def P (x y : ℝ) : ℝ := (x^2 - y^2) / (x * y) - (x * y - y^2) / (x * y - x^2)

theorem simplify_P_eq (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy: x ≠ y) : P x y = x / y := 
by
  -- Insert proof here
  sorry

end simplify_P_eq_l46_46263


namespace spending_percentage_A_l46_46409

def combined_salary (S_A S_B : ℝ) : Prop := S_A + S_B = 7000
def A_salary (S_A : ℝ) : Prop := S_A = 5250
def B_salary (S_B : ℝ) : Prop := S_B = 1750
def B_spending (P_B : ℝ) : Prop := P_B = 0.85
def same_savings (S_A S_B P_A P_B : ℝ) : Prop := S_A * (1 - P_A) = S_B * (1 - P_B)
def A_spending (P_A : ℝ) : Prop := P_A = 0.95

theorem spending_percentage_A (S_A S_B P_A P_B : ℝ) 
  (h1: combined_salary S_A S_B) 
  (h2: A_salary S_A) 
  (h3: B_salary S_B) 
  (h4: B_spending P_B) 
  (h5: same_savings S_A S_B P_A P_B) : A_spending P_A :=
sorry

end spending_percentage_A_l46_46409


namespace y_intercept_of_line_eq_l46_46356

theorem y_intercept_of_line_eq (x y : ℝ) (h : x + y - 1 = 0) : y = 1 :=
by
  sorry

end y_intercept_of_line_eq_l46_46356


namespace interval_intersection_l46_46869

/--
  This statement asserts that the intersection of the intervals (2/4, 3/4) and (2/5, 3/5)
  results in the interval (1/2, 0.6), which is the solution to the problem.
-/
theorem interval_intersection :
  { x : ℝ | 2 < 4 * x ∧ 4 * x < 3 ∧ 2 < 5 * x ∧ 5 * x < 3 } = { x : ℝ | 0.5 < x ∧ x < 0.6 } :=
by
  sorry

end interval_intersection_l46_46869


namespace ratio_of_doctors_to_nurses_l46_46896

theorem ratio_of_doctors_to_nurses (total_staff doctors nurses : ℕ) (h1 : total_staff = 456) (h2 : nurses = 264) (h3 : doctors + nurses = total_staff) :
  doctors = 192 ∧ (doctors : ℚ) / nurses = 8 / 11 :=
by
  sorry

end ratio_of_doctors_to_nurses_l46_46896


namespace fraction_identity_l46_46247

theorem fraction_identity (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : y^2 - 1/x ≠ 0) :
  (x^2 - 1/y) / (y^2 - 1/x) = x / y := 
by {
  sorry
}

end fraction_identity_l46_46247


namespace min_value_sin_cos_l46_46633

open Real

theorem min_value_sin_cos : ∃ x : ℝ, sin x * cos x = -1 / 2 := by
  sorry

end min_value_sin_cos_l46_46633


namespace solve_for_x_l46_46799

theorem solve_for_x (x : ℝ) (h : 8 / x + 6 = 8) : x = 4 :=
sorry

end solve_for_x_l46_46799


namespace min_total_cost_of_tank_l46_46734

theorem min_total_cost_of_tank (V D c₁ c₂ : ℝ) (hV : V = 0.18) (hD : D = 0.5)
  (hc₁ : c₁ = 400) (hc₂ : c₂ = 100) : 
  ∃ x : ℝ, x > 0 ∧ (y = c₂*D*(2*x + 0.72/x) + c₁*0.36) ∧ y = 264 := 
sorry

end min_total_cost_of_tank_l46_46734


namespace second_divisor_l46_46034

theorem second_divisor (N : ℤ) (k : ℤ) (D : ℤ) (m : ℤ) 
  (h1 : N = 39 * k + 20) 
  (h2 : N = D * m + 7) : 
  D = 13 := sorry

end second_divisor_l46_46034


namespace find_xy_l46_46103

variable (x y : ℚ)

theorem find_xy (h1 : 1/x + 3/y = 1/2) (h2 : 1/y - 3/x = 1/3) : 
    x = -20 ∧ y = 60/11 := 
by
  sorry

end find_xy_l46_46103


namespace investment_difference_l46_46013

noncomputable def A_Maria : ℝ := 60000 * (1 + 0.045)^3
noncomputable def A_David : ℝ := 60000 * (1 + 0.0175)^6
noncomputable def investment_diff : ℝ := A_Maria - A_David

theorem investment_difference : abs (investment_diff - 1803.30) < 1 :=
by
  have hM : A_Maria = 60000 * (1 + 0.045)^3 := by rfl
  have hD : A_David = 60000 * (1 + 0.0175)^6 := by rfl
  have hDiff : investment_diff = A_Maria - A_David := by rfl
  -- Proof would go here; using the provided approximations
  sorry

end investment_difference_l46_46013


namespace solution_set_of_inequality_l46_46160

theorem solution_set_of_inequality :
  { x : ℝ | (x - 5) / (x + 1) ≤ 0 } = { x : ℝ | -1 < x ∧ x ≤ 5 } :=
sorry

end solution_set_of_inequality_l46_46160


namespace fencing_required_l46_46448

theorem fencing_required (L W : ℕ) (hL : L = 10) (hA : L * W = 600) : L + 2 * W = 130 :=
by
  sorry

end fencing_required_l46_46448


namespace parallelogram_side_length_l46_46310

theorem parallelogram_side_length 
  (s : ℝ) 
  (A : ℝ)
  (angle : ℝ)
  (adj1 adj2 : ℝ) 
  (h : adj1 = s) 
  (h1 : adj2 = 2 * s) 
  (h2 : angle = 30)
  (h3 : A = 8 * Real.sqrt 3): 
  s = 2 * Real.sqrt 2 :=
by
  -- sorry to skip proofs
  sorry

end parallelogram_side_length_l46_46310


namespace mutually_exclusive_event_3_l46_46877

def is_odd (n : ℕ) := n % 2 = 1
def is_even (n : ℕ) := n % 2 = 0

def event_1 (a b : ℕ) := 
(is_odd a ∧ is_even b) ∨ (is_even a ∧ is_odd b)

def event_2 (a b : ℕ) := 
is_odd a ∧ is_odd b

def event_3 (a b : ℕ) := 
is_odd a ∧ is_even a ∧ is_odd b ∧ is_even b

def event_4 (a b : ℕ) :=
(is_odd a ∧ is_even b) ∨ (is_even a ∧ is_odd b)

theorem mutually_exclusive_event_3 :
  ∀ a b : ℕ, event_3 a b → ¬ event_1 a b ∧ ¬ event_2 a b ∧ ¬ event_4 a b := by
sorry

end mutually_exclusive_event_3_l46_46877


namespace part_i_l46_46929

theorem part_i (n : ℕ) (h₁ : n ≥ 1) (h₂ : n ∣ (2^n - 1)) : n = 1 :=
sorry

end part_i_l46_46929


namespace book_price_is_correct_l46_46499

-- Define the cost of the album
def album_cost : ℝ := 20

-- Define the discount percentage
def discount_percentage : ℝ := 0.30

-- Calculate the CD cost
def cd_cost : ℝ := album_cost * (1 - discount_percentage)

-- Define the additional cost of the book over the CD
def book_cd_diff : ℝ := 4

-- Calculate the book cost
def book_cost : ℝ := cd_cost + book_cd_diff

-- State the proposition to be proved
theorem book_price_is_correct : book_cost = 18 := by
  -- Provide the details of the calculations (optionally)
  sorry

end book_price_is_correct_l46_46499


namespace sarah_shirts_l46_46907

theorem sarah_shirts (loads : ℕ) (pieces_per_load : ℕ) (sweaters : ℕ) 
  (total_pieces : ℕ) (shirts : ℕ) : 
  loads = 9 → pieces_per_load = 5 → sweaters = 2 →
  total_pieces = loads * pieces_per_load → shirts = total_pieces - sweaters → 
  shirts = 43 :=
by
  intros h_loads h_pieces_per_load h_sweaters h_total_pieces h_shirts
  sorry

end sarah_shirts_l46_46907


namespace num_floors_each_building_l46_46579

theorem num_floors_each_building
  (floors_each_building num_apartments_per_floor num_doors_per_apartment total_doors : ℕ)
  (h1 : floors_each_building = F)
  (h2 : num_apartments_per_floor = 6)
  (h3 : num_doors_per_apartment = 7)
  (h4 : total_doors = 1008)
  (eq1 : 2 * floors_each_building * num_apartments_per_floor * num_doors_per_apartment = total_doors) :
  F = 12 :=
sorry

end num_floors_each_building_l46_46579


namespace find_value_l46_46846

variable (x y z : ℕ)

-- Condition: x / 4 = y / 3 = z / 2
def ratio_condition := x / 4 = y / 3 ∧ y / 3 = z / 2

-- Theorem: Given the ratio condition, prove that (x - y + 3z) / x = 7 / 4.
theorem find_value (h : ratio_condition x y z) : (x - y + 3 * z) / x = 7 / 4 := 
  by sorry

end find_value_l46_46846


namespace distinct_real_roots_l46_46206

open Real

theorem distinct_real_roots (n : ℕ) (hn : n > 0) (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (2 * n - 1 < x1 ∧ x1 ≤ 2 * n + 1) ∧ 
  (2 * n - 1 < x2 ∧ x2 ≤ 2 * n + 1) ∧ |x1 - 2 * n| = k ∧ |x2 - 2 * n| = k) ↔ (0 < k ∧ k ≤ 1) :=
by
  sorry

end distinct_real_roots_l46_46206


namespace find_A_for_diamond_eq_85_l46_46036

def diamond (A B : ℝ) : ℝ := 4 * A + B^2 + 7

theorem find_A_for_diamond_eq_85 :
  ∃ (A : ℝ), diamond A 3 = 85 ∧ A = 17.25 :=
by
  sorry

end find_A_for_diamond_eq_85_l46_46036


namespace twenty_mul_b_sub_a_not_integer_l46_46668

theorem twenty_mul_b_sub_a_not_integer {a b : ℝ} (hneq : a ≠ b) (hno_roots : ∀ x : ℝ,
  (x^2 + 20 * a * x + 10 * b) * (x^2 + 20 * b * x + 10 * a) ≠ 0) :
  ¬ ∃ n : ℤ, 20 * (b - a) = n :=
sorry

end twenty_mul_b_sub_a_not_integer_l46_46668


namespace run_time_difference_l46_46411

variables (distance duration_injured : ℝ) (initial_speed : ℝ)

theorem run_time_difference (H1 : distance = 20) 
                            (H2 : duration_injured = 22) 
                            (H3 : initial_speed = distance * 2 / duration_injured) :
                            duration_injured - (distance / initial_speed) = 11 :=
by
  sorry

end run_time_difference_l46_46411


namespace calculate_expression_l46_46971

theorem calculate_expression :
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5) + (1 / 6)) = 57 :=
by
  sorry

end calculate_expression_l46_46971


namespace find_m_l46_46524

theorem find_m (m : ℕ) :
  (∀ x : ℝ, -2 * x ^ 2 + 5 * x - 2 <= 9 / m) →
  m = 8 :=
sorry

end find_m_l46_46524


namespace salaries_proof_l46_46816

-- Define salaries as real numbers
variables (a b c d : ℝ)

-- Define assumptions
def conditions := 
  (a + b + c + d = 4000) ∧
  (0.05 * a + 0.15 * b = c) ∧ 
  (0.25 * d = 0.3 * b) ∧
  (b = 3 * c)

-- Define the solution as found
def solution :=
  (a = 2365.55) ∧
  (b = 645.15) ∧
  (c = 215.05) ∧
  (d = 774.18)

-- Prove that given the conditions, the solution holds
theorem salaries_proof : 
  (conditions a b c d) → (solution a b c d) := by
  sorry

end salaries_proof_l46_46816


namespace Carrie_pays_94_l46_46774

-- Formalizing the conditions
def num_shirts : ℕ := 4
def num_pants : ℕ := 2
def num_jackets : ℕ := 2
def cost_shirt : ℕ := 8
def cost_pant : ℕ := 18
def cost_jacket : ℕ := 60

-- The total cost Carrie needs to pay
def Carrie_pay (total_cost : ℕ) : ℕ := total_cost / 2

-- The total cost of all the clothes
def total_cost : ℕ :=
  num_shirts * cost_shirt +
  num_pants * cost_pant +
  num_jackets * cost_jacket

-- The proof statement that Carrie pays $94
theorem Carrie_pays_94 : Carrie_pay total_cost = 94 := 
by
  sorry

end Carrie_pays_94_l46_46774


namespace number_of_nonsimilar_triangles_l46_46642
-- Import the necessary library

-- Define the problem conditions
def angles_in_arithmetic_progression (a d : ℕ) : Prop :=
  0 < d ∧ d < 30 ∧ 
  (a - d > 0) ∧ (a + d < 180) ∧ -- Ensures positive and valid angles
  (a - d) + a + (a + d) = 180  -- Triangle sum property

-- Declare the theorem
theorem number_of_nonsimilar_triangles : 
  ∃ n : ℕ, n = 29 ∧ ∀ (a d : ℕ), angles_in_arithmetic_progression a d → d < 30 → a = 60 :=
sorry

end number_of_nonsimilar_triangles_l46_46642


namespace f_property_f_equals_when_x_lt_1_f_equals_when_x_gt_1_l46_46243

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then x / (1 + x) else 1 / (1 + x)

theorem f_property (x : ℝ) (hx : 0 < x) : 
  f x = f (1 / x) :=
by
  sorry

theorem f_equals_when_x_lt_1 (x : ℝ) (hx0 : 0 < x) (hx1 : x < 1) : 
  f x = 1 / (1 + x) :=
by
  sorry

theorem f_equals_when_x_gt_1 (x : ℝ) (hx : 1 < x) : 
  f x = x / (1 + x) :=
by
  sorry

end f_property_f_equals_when_x_lt_1_f_equals_when_x_gt_1_l46_46243


namespace hexagon_coloring_count_l46_46102

def num_possible_colorings : Nat :=
by
  /- There are 7 choices for first vertex A.
     Once A is chosen, there are 6 choices for the remaining vertices B, C, D, E, F considering the diagonal restrictions. -/
  let total_colorings := 7 * 6 ^ 5
  let restricted_colorings := 7 * 6 ^ 3
  let valid_colorings := total_colorings - restricted_colorings
  exact valid_colorings

theorem hexagon_coloring_count : num_possible_colorings = 52920 :=
  by
    /- Computation steps above show that the number of valid colorings is 52920 -/
    sorry   -- Proof computation already indicated

end hexagon_coloring_count_l46_46102


namespace c_put_15_oxen_l46_46360

theorem c_put_15_oxen (x : ℕ):
  (10 * 7 + 12 * 5 + 3 * x = 130 + 3 * x) →
  (175 * 3 * x / (130 + 3 * x) = 45) →
  x = 15 :=
by
  intros h1 h2
  sorry

end c_put_15_oxen_l46_46360


namespace tim_words_per_day_l46_46715

variable (original_words : ℕ)
variable (years : ℕ)
variable (increase_percent : ℚ)

noncomputable def words_per_day (original_words : ℕ) (years : ℕ) (increase_percent : ℚ) : ℚ :=
  let increase_words := original_words * increase_percent
  let total_days := years * 365
  increase_words / total_days

theorem tim_words_per_day :
    words_per_day 14600 2 (50 / 100) = 10 := by
  sorry

end tim_words_per_day_l46_46715


namespace find_M_l46_46597

theorem find_M (a b c M : ℚ) 
  (h1 : a + b + c = 100)
  (h2 : a - 10 = M)
  (h3 : b + 10 = M)
  (h4 : 10 * c = M) : 
  M = 1000 / 21 :=
sorry

end find_M_l46_46597


namespace arithmetic_geometric_sequence_l46_46548

theorem arithmetic_geometric_sequence (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
  (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : a 3 * 2 = a 1 + 2 * d)
  (h4 : a 4 = a 1 + 3 * d)
  (h5 : a 8 = a 1 + 7 * d)
  (h_geo : (a 1 + 3 * d) ^ 2 = (a 1 + 2 * d) * (a 1 + 7 * d))
  (h_sum : S 4 = (a 1 * 4) + (d * (4 * 3 / 2))) :
  a 1 * d < 0 ∧ d * S 4 < 0 :=
by sorry

end arithmetic_geometric_sequence_l46_46548


namespace daryl_age_l46_46632

theorem daryl_age (d j : ℕ) 
  (h1 : d - 4 = 3 * (j - 4)) 
  (h2 : d + 5 = 2 * (j + 5)) :
  d = 31 :=
by sorry

end daryl_age_l46_46632


namespace power_expression_l46_46222

variable {x : ℂ} -- Define x as a complex number

theorem power_expression (
  h : x - 1/x = 2 * Complex.I * Real.sqrt 2
) : x^(2187:ℕ) - 1/x^(2187:ℕ) = -22 * Complex.I * Real.sqrt 2 :=
by sorry

end power_expression_l46_46222


namespace cost_of_lamps_and_bulbs_l46_46367

theorem cost_of_lamps_and_bulbs : 
    let lamp_cost := 7
    let bulb_cost := lamp_cost - 4
    let total_cost := (lamp_cost * 2) + (bulb_cost * 6)
    total_cost = 32 := by
  let lamp_cost := 7
  let bulb_cost := lamp_cost - 4
  let total_cost := (lamp_cost * 2) + (bulb_cost * 6)
  sorry

end cost_of_lamps_and_bulbs_l46_46367


namespace cara_neighbors_l46_46231

def number_of_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

theorem cara_neighbors : number_of_pairs 7 = 21 :=
by
  sorry

end cara_neighbors_l46_46231


namespace exists_line_through_ellipse_diameter_circle_origin_l46_46141

theorem exists_line_through_ellipse_diameter_circle_origin :
  ∃ m : ℝ, (m = (4 * Real.sqrt 3) / 3 ∨ m = -(4 * Real.sqrt 3) / 3) ∧
  ∀ (x y : ℝ), (x^2 + 2 * y^2 = 8) → (y = x + m) → (x^2 + (x + m)^2 = 8) :=
by
  sorry

end exists_line_through_ellipse_diameter_circle_origin_l46_46141


namespace necessary_and_sufficient_condition_l46_46039

open Real

theorem necessary_and_sufficient_condition 
  {x y : ℝ} (p : x > y) (q : x - y + sin (x - y) > 0) : 
  (x > y) ↔ (x - y + sin (x - y) > 0) :=
sorry

end necessary_and_sufficient_condition_l46_46039


namespace value_of_a_minus_2_b_minus_2_l46_46687

theorem value_of_a_minus_2_b_minus_2 :
  ∀ (a b : ℝ), (a + b = -4/3 ∧ a * b = -7/3) → ((a - 2) * (b - 2) = 0) := by
  sorry

end value_of_a_minus_2_b_minus_2_l46_46687


namespace smallest_satisfying_N_is_2520_l46_46205

open Nat

def smallest_satisfying_N : ℕ :=
  let N := 2520
  if (N + 2) % 2 = 0 ∧
     (N + 3) % 3 = 0 ∧
     (N + 4) % 4 = 0 ∧
     (N + 5) % 5 = 0 ∧
     (N + 6) % 6 = 0 ∧
     (N + 7) % 7 = 0 ∧
     (N + 8) % 8 = 0 ∧
     (N + 9) % 9 = 0 ∧
     (N + 10) % 10 = 0
  then N else 0

-- Statement of the problem in Lean 4
theorem smallest_satisfying_N_is_2520 : smallest_satisfying_N = 2520 :=
  by
    -- Proof would be added here, but is omitted as per instructions
    sorry

end smallest_satisfying_N_is_2520_l46_46205


namespace factor_product_modulo_l46_46131

theorem factor_product_modulo (h1 : 2021 % 23 = 21) (h2 : 2022 % 23 = 22) (h3 : 2023 % 23 = 0) (h4 : 2024 % 23 = 1) (h5 : 2025 % 23 = 2) :
  (2021 * 2022 * 2023 * 2024 * 2025) % 23 = 0 :=
by
  sorry

end factor_product_modulo_l46_46131


namespace mark_deposit_is_88_l46_46191

-- Definitions according to the conditions
def markDeposit := 88
def bryanDeposit (m : ℕ) := 5 * m - 40

-- The theorem we need to prove
theorem mark_deposit_is_88 : markDeposit = 88 := 
by 
  -- Since the condition states Mark deposited $88,
  -- this is trivially true.
  sorry

end mark_deposit_is_88_l46_46191


namespace time_to_travel_downstream_l46_46611

-- Definitions based on the conditions.
def speed_boat_still_water := 40 -- Speed of the boat in still water (km/hr)
def speed_stream := 5 -- Speed of the stream (km/hr)
def distance_downstream := 45 -- Distance to be traveled downstream (km)

-- The proof statement
theorem time_to_travel_downstream : (distance_downstream / (speed_boat_still_water + speed_stream)) = 1 :=
by
  -- This would be the place to include the proven steps, but it's omitted as per instructions.
  sorry

end time_to_travel_downstream_l46_46611


namespace closest_integer_to_a2013_l46_46994

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 100 ∧ ∀ n : ℕ, a (n + 2) = a (n + 1) + (1 / a (n + 1))

theorem closest_integer_to_a2013 (a : ℕ → ℝ) (h : seq a) : abs (a 2013 - 118) < 0.5 :=
sorry

end closest_integer_to_a2013_l46_46994


namespace persimmons_picked_l46_46149

theorem persimmons_picked : 
  ∀ (J H : ℕ), (4 * J = H - 3) → (H = 35) → (J = 8) := 
by
  intros J H hJ hH
  sorry

end persimmons_picked_l46_46149


namespace problem_statement_l46_46007

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 8

theorem problem_statement : 3 * g 2 + 4 * g (-2) = 152 := by
  sorry

end problem_statement_l46_46007


namespace problem_solution_l46_46793

noncomputable def S : ℝ :=
  1 / (4 - Real.sqrt 15) + 1 / (Real.sqrt 15 - Real.sqrt 14) - 1 / (Real.sqrt 14 - Real.sqrt 13) + 1 / (Real.sqrt 13 - 3)

theorem problem_solution : S = 7 := 
by
  sorry

end problem_solution_l46_46793


namespace dave_initial_boxes_l46_46657

def pieces_per_box : ℕ := 3
def boxes_given_away : ℕ := 5
def pieces_left : ℕ := 21
def total_pieces_given_away := boxes_given_away * pieces_per_box
def total_pieces_initially := total_pieces_given_away + pieces_left

theorem dave_initial_boxes : total_pieces_initially / pieces_per_box = 12 := by
  sorry

end dave_initial_boxes_l46_46657


namespace find_coordinates_M_l46_46108

open Real

theorem find_coordinates_M (x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 : ℝ) :
  ∃ (xM yM zM : ℝ), 
  xM = (x1 + x2 + x3 + x4) / 4 ∧
  yM = (y1 + y2 + y3 + y4) / 4 ∧
  zM = (z1 + z2 + z3 + z4) / 4 ∧
  (x1 - xM) + (x2 - xM) + (x3 - xM) + (x4 - xM) = 0 ∧
  (y1 - yM) + (y2 - yM) + (y3 - yM) + (y4 - yM) = 0 ∧
  (z1 - zM) + (z2 - zM) + (z3 - zM) + (z4 - zM) = 0 := by
  sorry

end find_coordinates_M_l46_46108


namespace sum_b_a1_a2_a3_a4_eq_60_l46_46902

def a_n (n : ℕ) : ℕ := n + 2
def b_n (n : ℕ) : ℕ := 2^(n-1)

theorem sum_b_a1_a2_a3_a4_eq_60 :
  b_n (a_n 1) + b_n (a_n 2) + b_n (a_n 3) + b_n (a_n 4) = 60 :=
by
  sorry

end sum_b_a1_a2_a3_a4_eq_60_l46_46902


namespace tan_17pi_over_4_l46_46060

theorem tan_17pi_over_4 : Real.tan (17 * Real.pi / 4) = 1 := by
  sorry

end tan_17pi_over_4_l46_46060


namespace complex_pure_imaginary_l46_46148

theorem complex_pure_imaginary (a : ℝ) 
  (h1 : a^2 + 2*a - 3 = 0) 
  (h2 : a + 3 ≠ 0) : 
  a = 1 := 
by
  sorry

end complex_pure_imaginary_l46_46148


namespace fractions_order_l46_46525

theorem fractions_order:
  (20 / 15) < (25 / 18) ∧ (25 / 18) < (23 / 16) ∧ (23 / 16) < (21 / 14) :=
by
  sorry

end fractions_order_l46_46525


namespace product_f_g_l46_46643

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x * (x + 1))
noncomputable def g (x : ℝ) : ℝ := 1 / Real.sqrt x

theorem product_f_g (x : ℝ) (hx : 0 < x) : f x * g x = Real.sqrt (x + 1) := 
by 
  sorry

end product_f_g_l46_46643


namespace binary_101_eq_5_l46_46112

theorem binary_101_eq_5 : 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 5 := 
by
  sorry

end binary_101_eq_5_l46_46112


namespace describe_T_l46_46653

def T : Set (ℝ × ℝ) := 
  { p | ∃ x y : ℝ, p = (x, y) ∧ (
      (5 = x + 3 ∧ y - 6 ≤ 5) ∨
      (5 = y - 6 ∧ x + 3 ≤ 5) ∨
      (x + 3 = y - 6 ∧ x + 3 ≤ 5 ∧ y - 6 ≤ 5)
  )}

theorem describe_T : T = { p | ∃ x y : ℝ, p = (2, y) ∧ y ≤ 11 ∨
                                      p = (x, 11) ∧ x ≤ 2 ∨
                                      p = (x, x + 9) ∧ x ≤ 2 ∧ x + 9 ≤ 11 } :=
by
  sorry

end describe_T_l46_46653


namespace lemon_bag_mass_l46_46092

variable (m : ℝ)  -- mass of one bag of lemons in kg

-- Conditions
def max_load := 900  -- maximum load in kg
def num_bags := 100  -- number of bags
def extra_load := 100  -- additional load in kg

-- Proof statement (target)
theorem lemon_bag_mass : num_bags * m + extra_load = max_load → m = 8 :=
by
  sorry

end lemon_bag_mass_l46_46092


namespace decimal_to_fraction_l46_46109

theorem decimal_to_fraction : (0.3 + (0.24 - 0.24 / 100)) = (19 / 33) :=
by
  sorry

end decimal_to_fraction_l46_46109


namespace max_distance_of_MN_l46_46296

noncomputable def curve_C_polar (θ : ℝ) : ℝ := 2 * Real.cos θ

noncomputable def curve_C_cartesian (x y : ℝ) := x^2 + y^2 - 2 * x

noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  ( -1 + (Real.sqrt 5 / 5) * t, (2 * Real.sqrt 5 / 5) * t)

def point_M : ℝ × ℝ := (0, 2)

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

noncomputable def center_C : ℝ × ℝ := (1, 0)

theorem max_distance_of_MN :
  ∃ N : ℝ × ℝ, 
  ∀ (θ : ℝ), N = (curve_C_polar θ * Real.cos θ, curve_C_polar θ * Real.sin θ) →
  distance point_M N ≤ Real.sqrt 5 + 1 :=
sorry

end max_distance_of_MN_l46_46296


namespace first_chapter_pages_calculation_l46_46623

-- Define the constants and conditions
def second_chapter_pages : ℕ := 11
def first_chapter_pages_more : ℕ := 37

-- Main proof problem
theorem first_chapter_pages_calculation : first_chapter_pages_more + second_chapter_pages = 48 := by
  sorry

end first_chapter_pages_calculation_l46_46623


namespace total_boxes_is_27_l46_46784

-- Defining the conditions
def stops : ℕ := 3
def boxes_per_stop : ℕ := 9

-- Prove that the total number of boxes is as expected
theorem total_boxes_is_27 : stops * boxes_per_stop = 27 := by
  sorry

end total_boxes_is_27_l46_46784


namespace arithmetic_sequence_properties_l46_46200

noncomputable def common_difference (a d : ℚ) (n : ℕ) : ℚ := a + (n - 1) * d

theorem arithmetic_sequence_properties :
  ∃ d : ℚ, d = 5 / 9 ∧ ∃ S : ℚ, S = -29 / 3 ∧
  ∀ n : ℕ, ∃ a₁ a₅ a₈ : ℚ, a₁ = -3 ∧
    a₅ = common_difference a₁ d 5 ∧
    a₈ = common_difference a₁ d 8 ∧ 
    11 * a₅ = 5 * a₈ - 13 ∧
    S = (n / 2) * (2 * a₁ + (n - 1) * d) ∧
    n = 6 := 
sorry

end arithmetic_sequence_properties_l46_46200


namespace elevator_time_l46_46542

theorem elevator_time :
  ∀ (floors steps_per_floor steps_per_second extra_time : ℕ) (elevator_time_sec elevator_time_min : ℚ),
    floors = 8 →
    steps_per_floor = 30 →
    steps_per_second = 3 →
    extra_time = 30 →
    elevator_time_sec = ((floors * steps_per_floor) / steps_per_second) - extra_time →
    elevator_time_min = elevator_time_sec / 60 →
    elevator_time_min = 0.833 :=
by
  intros floors steps_per_floor steps_per_second extra_time elevator_time_sec elevator_time_min
  intros h_floors h_steps_per_floor h_steps_per_second h_extra_time h_elevator_time_sec h_elevator_time_min
  rw [h_floors, h_steps_per_floor, h_steps_per_second, h_extra_time] at *
  sorry

end elevator_time_l46_46542


namespace first_group_persons_l46_46725

-- Define the conditions as formal variables
variables (P : ℕ) (hours_per_day_1 days_1 hours_per_day_2 days_2 num_persons_2 : ℕ)

-- Define the conditions from the problem
def first_group_work := P * days_1 * hours_per_day_1
def second_group_work := num_persons_2 * days_2 * hours_per_day_2

-- Set the conditions based on the problem statement
axiom conditions : 
  hours_per_day_1 = 5 ∧ 
  days_1 = 12 ∧ 
  hours_per_day_2 = 6 ∧
  days_2 = 26 ∧
  num_persons_2 = 30 ∧
  first_group_work = second_group_work

-- Statement to prove
theorem first_group_persons : P = 78 :=
by
  -- The proof goes here
  sorry

end first_group_persons_l46_46725


namespace john_total_cost_l46_46834

-- Definitions based on given conditions
def yearly_cost_first_8_years : ℕ := 10000
def yearly_cost_next_10_years : ℕ := 20000
def university_tuition : ℕ := 250000
def years_first_phase : ℕ := 8
def years_second_phase : ℕ := 10

-- We need to prove the total cost John pays
theorem john_total_cost : 
  (years_first_phase * yearly_cost_first_8_years + years_second_phase * yearly_cost_next_10_years + university_tuition) / 2 = 265000 :=
by sorry

end john_total_cost_l46_46834


namespace value_of_d_l46_46106

theorem value_of_d (d : ℝ) (h : ∀ y : ℝ, 3 * (5 + d * y) = 15 * y + 15) : d = 5 :=
by
  sorry

end value_of_d_l46_46106


namespace sum_of_x_values_l46_46214

theorem sum_of_x_values (y x : ℝ) (h1 : y = 6) (h2 : x^2 + y^2 = 144) : x + (-x) = 0 :=
by
  sorry

end sum_of_x_values_l46_46214


namespace same_color_probability_correct_l46_46456

noncomputable def prob_same_color (green red blue : ℕ) : ℚ :=
  let total := green + red + blue
  (green / total) * (green / total) +
  (red / total) * (red / total) +
  (blue / total) * (blue / total)

theorem same_color_probability_correct :
  prob_same_color 5 7 3 = 83 / 225 :=
by
  sorry

end same_color_probability_correct_l46_46456


namespace x_of_x35x_div_by_18_l46_46827

theorem x_of_x35x_div_by_18 (x : ℕ) (h₁ : 18 = 2 * 9) (h₂ : (2 * x + 8) % 9 = 0) (h₃ : ∃ k : ℕ, x = 2 * k) : x = 8 :=
sorry

end x_of_x35x_div_by_18_l46_46827


namespace nat_solutions_l46_46749

open Nat

theorem nat_solutions (a b c : ℕ) :
  (a ≤ b ∧ b ≤ c ∧ ab + bc + ca = 2 * (a + b + c)) ↔ (a = 2 ∧ b = 2 ∧ c = 2) ∨ (a = 1 ∧ b = 2 ∧ c = 4) :=
by sorry

end nat_solutions_l46_46749


namespace find_second_discount_l46_46490

theorem find_second_discount 
    (list_price : ℝ)
    (final_price : ℝ)
    (first_discount : ℝ)
    (second_discount : ℝ)
    (h₁ : list_price = 65)
    (h₂ : final_price = 57.33)
    (h₃ : first_discount = 0.10)
    (h₄ : (list_price - (first_discount * list_price)) = 58.5)
    (h₅ : final_price = 58.5 - (second_discount * 58.5)) :
    second_discount = 0.02 := 
by
  sorry

end find_second_discount_l46_46490


namespace simplify_expression_l46_46049

variable (x : ℝ)

theorem simplify_expression :
  (3 * x^6 + 2 * x^5 + x^4 + x - 5) - (x^6 + 3 * x^5 + 2 * x^3 + 6) = 2 * x^6 - x^5 + x^4 - 2 * x^3 + x + 1 := by
  sorry

end simplify_expression_l46_46049


namespace range_of_a_l46_46984

open Real

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 * exp x1 - a = 0) ∧ (x2 * exp x2 - a = 0)) ↔ -1 / exp 1 < a ∧ a < 0 :=
sorry

end range_of_a_l46_46984


namespace geometric_sequence_sixth_term_l46_46493

theorem geometric_sequence_sixth_term (a : ℝ) (r : ℝ) 
  (h1 : a = 512) 
  (h2 : a * r^(7) = 2) :
  a * r^(5) = 16 :=
by
  sorry

end geometric_sequence_sixth_term_l46_46493


namespace factorable_b_even_l46_46081

-- Defining the conditions
def is_factorable (b : ℤ) : Prop :=
  ∃ (m n p q : ℤ), 
    m * p = 15 ∧ n * q = 15 ∧ b = m * q + n * p

-- The theorem to be stated
theorem factorable_b_even (b : ℤ) : is_factorable b ↔ ∃ k : ℤ, b = 2 * k :=
sorry

end factorable_b_even_l46_46081


namespace lolita_milk_per_week_l46_46385

def weekday_milk : ℕ := 3
def saturday_milk : ℕ := 2 * weekday_milk
def sunday_milk : ℕ := 3 * weekday_milk
def total_milk_week : ℕ := 5 * weekday_milk + saturday_milk + sunday_milk

theorem lolita_milk_per_week : total_milk_week = 30 := 
by 
  sorry

end lolita_milk_per_week_l46_46385


namespace post_office_mail_in_six_months_l46_46513

/-- The number of pieces of mail the post office receives per day -/
def mail_per_day : ℕ := 60 + 20

/-- The number of days in six months, assuming each month has 30 days -/
def days_in_six_months : ℕ := 6 * 30

/-- The total number of pieces of mail handled in six months -/
def total_mail_in_six_months : ℕ := mail_per_day * days_in_six_months

/-- The post office handles 14400 pieces of mail in six months -/
theorem post_office_mail_in_six_months : total_mail_in_six_months = 14400 := by
  sorry

end post_office_mail_in_six_months_l46_46513


namespace johns_last_month_savings_l46_46654

theorem johns_last_month_savings (earnings rent dishwasher left_over : ℝ) 
  (h1 : rent = 0.40 * earnings) 
  (h2 : dishwasher = 0.70 * rent) 
  (h3 : left_over = earnings - rent - dishwasher) :
  left_over = 0.32 * earnings :=
by 
  sorry

end johns_last_month_savings_l46_46654


namespace max_leap_years_l46_46843

theorem max_leap_years (years : ℕ) (leap_interval : ℕ) (total_years : ℕ) :
  leap_interval = 5 ∧ total_years = 200 → (years = total_years / leap_interval) :=
by
  sorry

end max_leap_years_l46_46843


namespace meters_of_cloth_l46_46882

variable (total_cost cost_per_meter : ℝ)
variable (h1 : total_cost = 434.75)
variable (h2 : cost_per_meter = 47)

theorem meters_of_cloth : 
  total_cost / cost_per_meter = 9.25 := 
by
  sorry

end meters_of_cloth_l46_46882


namespace union_A_B_l46_46153

noncomputable def U := Set.univ ℝ

def A : Set ℝ := {x | x^2 - x - 2 = 0}

def B : Set ℝ := {y | ∃ x, x ∈ A ∧ y = x + 3}

theorem union_A_B : A ∪ B = { -1, 2, 5 } :=
by
  sorry

end union_A_B_l46_46153


namespace parabola_focus_l46_46277

theorem parabola_focus (x y : ℝ) (h : y = 4 * x^2) : (0, 1) = (0, 1) :=
by 
  -- key steps would go here
  sorry

end parabola_focus_l46_46277


namespace function_increasing_intervals_l46_46194

theorem function_increasing_intervals (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f x < f (x + 1)) :
  (∃ x : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ y : ℝ, abs (y - x) < δ → f y > f x) ∨ 
  (∀ x : ℝ, ∃ ε > 0, ∀ δ > 0, ∃ y : ℝ, abs (y - x) < δ ∧ f y < f x) :=
sorry

end function_increasing_intervals_l46_46194


namespace perp_to_par_perp_l46_46453

variable (m : Line)
variable (α β : Plane)

-- Conditions
axiom parallel_planes (α β : Plane) : Prop
axiom perp (m : Line) (α : Plane) : Prop

-- Statements
axiom parallel_planes_ax : parallel_planes α β
axiom perp_ax : perp m α

-- Goal
theorem perp_to_par_perp {m : Line} {α β : Plane} (h1 : perp m α) (h2 : parallel_planes α β) : perp m β := sorry

end perp_to_par_perp_l46_46453


namespace probability_of_three_heads_in_eight_tosses_l46_46182

noncomputable def coin_toss_probability : ℚ :=
  let total_outcomes := 2^8
  let favorable_outcomes := Nat.choose 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_three_heads_in_eight_tosses : coin_toss_probability = 7 / 32 :=
  by
  sorry

end probability_of_three_heads_in_eight_tosses_l46_46182


namespace fruit_basket_l46_46996

theorem fruit_basket :
  ∀ (oranges apples bananas peaches : ℕ),
  oranges = 6 →
  apples = oranges - 2 →
  bananas = 3 * apples →
  peaches = bananas / 2 →
  oranges + apples + bananas + peaches = 28 :=
by
  intros oranges apples bananas peaches h_oranges h_apples h_bananas h_peaches
  rw [h_oranges, h_apples, h_bananas, h_peaches]
  sorry

end fruit_basket_l46_46996


namespace cube_edge_length_l46_46545

theorem cube_edge_length (n_edges : ℕ) (total_length : ℝ) (length_one_edge : ℝ) 
  (h1: n_edges = 12) (h2: total_length = 96) : length_one_edge = 8 :=
by
  sorry

end cube_edge_length_l46_46545


namespace locus_of_Y_right_angled_triangle_l46_46769

-- Conditions definitions
variables {A B C : Type*} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables (b c m : ℝ) -- Coordinates and slopes related to the problem
variables (x : ℝ) -- Independent variable for the locus line

-- The problem statement
theorem locus_of_Y_right_angled_triangle 
  (A_right_angle : ∀ (α β : ℝ), α * β = 0) 
  (perpendicular_lines : b ≠ m * c) 
  (no_coincide : (b^2 * m - 2 * b * c - c^2 * m) ≠ 0) :
  ∃ (y : ℝ), y = (2 * b * c * (b * m - c) - x * (b^2 + 2 * b * c * m - c^2)) / (b^2 * m - 2 * b * c - c^2 * m) := 
sorry

end locus_of_Y_right_angled_triangle_l46_46769


namespace range_of_k_l46_46447

theorem range_of_k (x y k : ℝ) (h1 : 2 * x - 3 * y = 5) (h2 : 2 * x - y = k) (h3 : x > y) : k > -5 :=
sorry

end range_of_k_l46_46447


namespace tournament_total_games_l46_46429

def total_number_of_games (num_teams : ℕ) (group_size : ℕ) (num_groups : ℕ) (teams_for_knockout : ℕ) : ℕ :=
  let games_per_group := (group_size * (group_size - 1)) / 2
  let group_stage_games := num_groups * games_per_group
  let knockout_teams := num_groups * teams_for_knockout
  let knockout_games := knockout_teams - 1
  group_stage_games + knockout_games

theorem tournament_total_games : total_number_of_games 32 4 8 2 = 63 := by
  sorry

end tournament_total_games_l46_46429


namespace negation_proposition_l46_46615

theorem negation_proposition : 
  (¬ ∃ x_0 : ℝ, 2 * x_0 - 3 > 1) ↔ (∀ x : ℝ, 2 * x - 3 ≤ 1) :=
by
  sorry

end negation_proposition_l46_46615


namespace hurricane_damage_in_GBP_l46_46895

def damage_in_AUD : ℤ := 45000000
def conversion_rate : ℚ := 1 / 2 -- 1 AUD = 1/2 GBP

theorem hurricane_damage_in_GBP : 
  (damage_in_AUD : ℚ) * conversion_rate = 22500000 := 
by
  sorry

end hurricane_damage_in_GBP_l46_46895


namespace largest_of_options_l46_46867

theorem largest_of_options :
  max (2 + 0 + 1 + 3) (max (2 * 0 + 1 + 3) (max (2 + 0 * 1 + 3) (max (2 + 0 + 1 * 3) (2 * 0 * 1 * 3)))) = 2 + 0 + 1 + 3 := by sorry

end largest_of_options_l46_46867


namespace flowers_died_l46_46801

theorem flowers_died : 
  let initial_flowers := 2 * 5
  let grown_flowers := initial_flowers + 20
  let harvested_flowers := 5 * 4
  grown_flowers - harvested_flowers = 10 :=
by
  sorry

end flowers_died_l46_46801


namespace farmer_profit_l46_46539

def piglet_cost_per_month : Int := 10
def pig_revenue : Int := 300
def num_piglets_sold_early : Int := 3
def num_piglets_sold_late : Int := 3
def early_sale_months : Int := 12
def late_sale_months : Int := 16

def total_profit (num_piglets_sold_early num_piglets_sold_late early_sale_months late_sale_months : Int) 
  (piglet_cost_per_month pig_revenue : Int) : Int := 
  let early_cost := num_piglets_sold_early * piglet_cost_per_month * early_sale_months
  let late_cost := num_piglets_sold_late * piglet_cost_per_month * late_sale_months
  let total_cost := early_cost + late_cost
  let total_revenue := (num_piglets_sold_early + num_piglets_sold_late) * pig_revenue
  total_revenue - total_cost

theorem farmer_profit : total_profit num_piglets_sold_early num_piglets_sold_late early_sale_months late_sale_months piglet_cost_per_month pig_revenue = 960 := by
  sorry

end farmer_profit_l46_46539


namespace fraction_correct_l46_46127

-- Define the total number of coins.
def total_coins : ℕ := 30

-- Define the number of states that joined the union in the decade 1800 through 1809.
def states_1800_1809 : ℕ := 4

-- Define the fraction of coins representing states joining in the decade 1800 through 1809.
def fraction_coins_1800_1809 : ℚ := states_1800_1809 / total_coins

-- The theorem statement that needs to be proved.
theorem fraction_correct : fraction_coins_1800_1809 = (2 / 15) := 
by
  sorry

end fraction_correct_l46_46127


namespace freight_train_distance_l46_46585

variable (travel_rate : ℕ) (initial_distance : ℕ) (time_minutes : ℕ) 

def total_distance_traveled (travel_rate : ℕ) (initial_distance : ℕ) (time_minutes : ℕ) : ℕ :=
  let traveled_distance := (time_minutes / travel_rate) 
  traveled_distance + initial_distance

theorem freight_train_distance :
  total_distance_traveled 2 5 90 = 50 :=
by
  sorry

end freight_train_distance_l46_46585


namespace boat_speed_in_still_water_l46_46665

variable (x : ℝ) -- speed of the boat in still water in km/hr
variable (current_rate : ℝ := 4) -- rate of the current in km/hr
variable (downstream_distance : ℝ := 4.8) -- distance traveled downstream in km
variable (downstream_time : ℝ := 18 / 60) -- time traveled downstream in hours

-- The main theorem stating that the speed of the boat in still water is 12 km/hr
theorem boat_speed_in_still_water : x = 12 :=
by
  -- Express the downstream speed and time relation
  have downstream_speed := x + current_rate
  have distance_relation := downstream_distance = downstream_speed * downstream_time
  -- Simplify and solve for x
  simp at distance_relation
  sorry

end boat_speed_in_still_water_l46_46665


namespace turnip_bag_weight_l46_46471

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]
def total_weight : ℕ := 106
def is_divisible_by_three (n : ℕ) : Prop := n % 3 = 0

theorem turnip_bag_weight :
  ∃ T : ℕ, T ∈ bag_weights ∧ (is_divisible_by_three (total_weight - T)) := sorry

end turnip_bag_weight_l46_46471


namespace range_of_k_l46_46416

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 1

noncomputable def g (x : ℝ) : ℝ := x^2 - 1

noncomputable def h (x : ℝ) : ℝ := x

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, x ≠ 0 → g (k * x + k / x) < g (x^2 + 1 / x^2 + 1)) ↔ (-3 / 2 < k ∧ k < 3 / 2) :=
by
  sorry

end range_of_k_l46_46416


namespace lumber_price_increase_l46_46748

noncomputable def percentage_increase_in_lumber_cost : ℝ :=
  let original_cost_lumber := 450
  let cost_nails := 30
  let cost_fabric := 80
  let original_total_cost := original_cost_lumber + cost_nails + cost_fabric
  let increase_in_total_cost := 97
  let new_total_cost := original_total_cost + increase_in_total_cost
  let unchanged_cost := cost_nails + cost_fabric
  let new_cost_lumber := new_total_cost - unchanged_cost
  let increase_lumber_cost := new_cost_lumber - original_cost_lumber
  (increase_lumber_cost / original_cost_lumber) * 100

theorem lumber_price_increase :
  percentage_increase_in_lumber_cost = 21.56 := by
  sorry

end lumber_price_increase_l46_46748


namespace sum_of_terms_l46_46362

noncomputable def u1 := 8
noncomputable def r := 2

def first_geometric (u2 u3 : ℝ) (u1 r : ℝ) : Prop := 
  u2 = r * u1 ∧ u3 = r^2 * u1

def last_arithmetic (u2 u3 u4 : ℝ) : Prop := 
  u3 - u2 = u4 - u3

def terms (u1 u2 u3 u4 : ℝ) (r : ℝ) : Prop :=
  first_geometric u2 u3 u1 r ∧
  last_arithmetic u2 u3 u4 ∧
  u4 = u1 + 40

theorem sum_of_terms (u1 u2 u3 u4 : ℝ)
  (h : terms u1 u2 u3 u4 r) : u1 + u2 + u3 + u4 = 104 :=
by {
  sorry
}

end sum_of_terms_l46_46362


namespace Toms_walking_speed_l46_46789

theorem Toms_walking_speed
  (total_distance : ℝ)
  (total_time : ℝ)
  (run_distance : ℝ)
  (run_speed : ℝ)
  (walk_distance : ℝ)
  (walk_time : ℝ)
  (walk_speed : ℝ)
  (h1 : total_distance = 1800)
  (h2 : total_time ≤ 20)
  (h3 : run_distance = 600)
  (h4 : run_speed = 210)
  (h5 : total_distance = run_distance + walk_distance)
  (h6 : total_time = walk_time + run_distance / run_speed)
  (h7 : walk_speed = walk_distance / walk_time) :
  walk_speed ≤ 70 := sorry

end Toms_walking_speed_l46_46789
