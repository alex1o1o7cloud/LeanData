import Mathlib

namespace train_length_is_correct_l254_25492

variable (speed_km_hr : Float) (time_sec : Float)

def speed_m_s (speed_km_hr : Float) : Float := speed_km_hr * (1000 / 3600)

def length_of_train (speed_km_hr : Float) (time_sec : Float) : Float :=
  speed_m_s speed_km_hr * time_sec

theorem train_length_is_correct :
  length_of_train 60 12 = 200.04 := 
sorry

end train_length_is_correct_l254_25492


namespace smallest_positive_integer_n_mean_squares_l254_25410

theorem smallest_positive_integer_n_mean_squares :
  ∃ n : ℕ, n > 1 ∧ (∃ m : ℕ, (n * m ^ 2 = (n + 1) * (2 * n + 1) / 6) ∧ Nat.gcd (n + 1) (2 * n + 1) = 1 ∧ n = 337) :=
sorry

end smallest_positive_integer_n_mean_squares_l254_25410


namespace express_in_scientific_notation_l254_25428

theorem express_in_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), 388800 = a * 10 ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.888 ∧ n = 5 :=
by
  sorry

end express_in_scientific_notation_l254_25428


namespace sum_divisible_by_11_l254_25434

theorem sum_divisible_by_11 (n : ℕ) : (6^(2*n) + 3^n + 3^(n+2)) % 11 = 0 := by
  sorry

end sum_divisible_by_11_l254_25434


namespace range_of_a_opposite_sides_l254_25425

theorem range_of_a_opposite_sides {a : ℝ} (h : (0 + 0 - a) * (1 + 1 - a) < 0) : 0 < a ∧ a < 2 :=
sorry

end range_of_a_opposite_sides_l254_25425


namespace n_value_l254_25408

theorem n_value (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^18 → n = 9 :=
by
  sorry

end n_value_l254_25408


namespace probability_penny_nickel_dime_heads_l254_25427

noncomputable def probability_heads (n : ℕ) : ℚ := (1 : ℚ) / (2 ^ n)

theorem probability_penny_nickel_dime_heads :
  probability_heads 3 = 1 / 8 := 
by
  sorry

end probability_penny_nickel_dime_heads_l254_25427


namespace remaining_fruit_count_l254_25436

theorem remaining_fruit_count (trees : ℕ) (fruits_per_tree : ℕ) (picked_fraction : ℚ) 
  (trees_eq : trees = 8) (fruits_per_tree_eq : fruits_per_tree = 200) (picked_fraction_eq : picked_fraction = 2/5) :
  let total_fruits := trees * fruits_per_tree
  let picked_fruits := picked_fraction * fruits_per_tree * trees
  let remaining_fruits := total_fruits - picked_fruits
  remaining_fruits = 960 := 
by 
  sorry

end remaining_fruit_count_l254_25436


namespace probability_both_blue_l254_25467

-- Conditions defined as assumptions
def jarC_red := 6
def jarC_blue := 10
def total_buttons_in_C := jarC_red + jarC_blue

def after_transfer_buttons_in_C := (3 / 4) * total_buttons_in_C

-- Carla removes the same number of red and blue buttons
-- and after transfer, 12 buttons remain in Jar C
def removed_buttons := total_buttons_in_C - after_transfer_buttons_in_C
def removed_red_buttons := removed_buttons / 2
def removed_blue_buttons := removed_buttons / 2

def remaining_red_in_C := jarC_red - removed_red_buttons
def remaining_blue_in_C := jarC_blue - removed_blue_buttons
def remaining_buttons_in_C := remaining_red_in_C + remaining_blue_in_C

def total_buttons_in_D := removed_buttons
def transferred_blue_buttons := removed_blue_buttons

-- Probability calculations
def probability_blue_in_C := remaining_blue_in_C / remaining_buttons_in_C
def probability_blue_in_D := transferred_blue_buttons / total_buttons_in_D

-- Proof
theorem probability_both_blue :
  (probability_blue_in_C * probability_blue_in_D) = (1 / 3) := 
by
  -- sorry is used here to skip the actual proof
  sorry

end probability_both_blue_l254_25467


namespace ratio_men_to_women_l254_25496

theorem ratio_men_to_women (M W : ℕ) (h1 : W = M + 4) (h2 : M + W = 18) : M = 7 ∧ W = 11 :=
by
  sorry

end ratio_men_to_women_l254_25496


namespace balls_per_bag_l254_25449

theorem balls_per_bag (total_balls bags_used: Nat) (h1: total_balls = 36) (h2: bags_used = 9) : total_balls / bags_used = 4 := by
  sorry

end balls_per_bag_l254_25449


namespace pizza_order_l254_25486

theorem pizza_order (couple_want: ℕ) (child_want: ℕ) (num_couples: ℕ) (num_children: ℕ) (slices_per_pizza: ℕ)
  (hcouple: couple_want = 3) (hchild: child_want = 1) (hnumc: num_couples = 1) (hnumch: num_children = 6) (hsp: slices_per_pizza = 4) :
  (couple_want * 2 * num_couples + child_want * num_children) / slices_per_pizza = 3 := 
by
  -- Proof here
  sorry

end pizza_order_l254_25486


namespace symmetry_center_range_in_interval_l254_25448

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6) + 1

theorem symmetry_center (k : ℤ) :
  ∃ n : ℤ, ∃ x : ℝ, x = Real.pi / 12 + n * Real.pi / 2 ∧ f x = 1 := 
sorry

theorem range_in_interval :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → ∃ y : ℝ, f y ∈ Set.Icc 0 3 := 
sorry

end symmetry_center_range_in_interval_l254_25448


namespace foxes_hunt_duration_l254_25475

variable (initial_weasels : ℕ) (initial_rabbits : ℕ) (remaining_rodents : ℕ)
variable (foxes : ℕ) (weasels_per_week : ℕ) (rabbits_per_week : ℕ)

def total_rodents_per_week (weasels_per_week rabbits_per_week foxes : ℕ) : ℕ :=
  foxes * (weasels_per_week + rabbits_per_week)

def initial_rodents (initial_weasels initial_rabbits : ℕ) : ℕ :=
  initial_weasels + initial_rabbits

def total_rodents_caught (initial_rodents remaining_rodents : ℕ) : ℕ :=
  initial_rodents - remaining_rodents

def weeks_hunted (total_rodents_caught total_rodents_per_week : ℕ) : ℕ :=
  total_rodents_caught / total_rodents_per_week

theorem foxes_hunt_duration
  (initial_weasels := 100) (initial_rabbits := 50) (remaining_rodents := 96)
  (foxes := 3) (weasels_per_week := 4) (rabbits_per_week := 2) :
  weeks_hunted (total_rodents_caught (initial_rodents initial_weasels initial_rabbits) remaining_rodents) 
                 (total_rodents_per_week weasels_per_week rabbits_per_week foxes) = 3 :=
by
  sorry

end foxes_hunt_duration_l254_25475


namespace cost_of_two_pencils_and_one_pen_l254_25432

variable (a b : ℝ)

-- Given conditions
def condition1 : Prop := (5 * a + b = 2.50)
def condition2 : Prop := (a + 2 * b = 1.85)

-- Statement to prove
theorem cost_of_two_pencils_and_one_pen
  (h1 : condition1 a b) 
  (h2 : condition2 a b) : 
  2 * a + b = 1.45 :=
sorry

end cost_of_two_pencils_and_one_pen_l254_25432


namespace Brandy_can_safely_drink_20_mg_more_l254_25468

variable (maximum_caffeine_per_day : ℕ := 500)
variable (caffeine_per_drink : ℕ := 120)
variable (number_of_drinks : ℕ := 4)
variable (caffeine_consumed : ℕ := caffeine_per_drink * number_of_drinks)

theorem Brandy_can_safely_drink_20_mg_more :
    caffeine_consumed = caffeine_per_drink * number_of_drinks →
    (maximum_caffeine_per_day - caffeine_consumed) = 20 :=
by
  intros h1
  rw [h1]
  sorry

end Brandy_can_safely_drink_20_mg_more_l254_25468


namespace power_div_eq_l254_25440

theorem power_div_eq (a : ℕ) (h : 36 = 6^2) : (6^12 / 36^5) = 36 := by
  sorry

end power_div_eq_l254_25440


namespace tanya_body_lotions_l254_25424

variable {F L : ℕ}  -- Number of face moisturizers (F) and body lotions (L) Tanya bought

theorem tanya_body_lotions
  (price_face_moisturizer : ℕ := 50)
  (price_body_lotion : ℕ := 60)
  (num_face_moisturizers : ℕ := 2)
  (total_spent : ℕ := 1020)
  (christy_spending_factor : ℕ := 2)
  (h_together_spent : total_spent = 3 * (num_face_moisturizers * price_face_moisturizer + L * price_body_lotion)) :
  L = 4 :=
by
  sorry

end tanya_body_lotions_l254_25424


namespace expression_as_polynomial_l254_25493

theorem expression_as_polynomial (x : ℝ) :
  (3 * x^3 + 2 * x^2 + 5 * x + 9) * (x - 2) -
  (x - 2) * (2 * x^3 + 5 * x^2 - 74) +
  (4 * x - 17) * (x - 2) * (x + 4) = 
  x^4 + 2 * x^3 - 5 * x^2 + 9 * x - 30 :=
sorry

end expression_as_polynomial_l254_25493


namespace find_y_l254_25446

theorem find_y (x : ℝ) (h : x^2 + (1 / x)^2 = 7) : x + 1 / x = 3 :=
by
  sorry

end find_y_l254_25446


namespace female_students_proportion_and_count_l254_25482

noncomputable def num_students : ℕ := 30
noncomputable def num_male_students : ℕ := 8
noncomputable def overall_avg_score : ℚ := 90
noncomputable def male_avg_scores : (ℚ × ℚ × ℚ) := (87, 95, 89)
noncomputable def female_avg_scores : (ℚ × ℚ × ℚ) := (92, 94, 91)
noncomputable def avg_attendance_alg_geom : ℚ := 0.85
noncomputable def avg_attendance_calc : ℚ := 0.89

theorem female_students_proportion_and_count :
  ∃ (F : ℕ), F = num_students - num_male_students ∧ (F / num_students : ℚ) = 11 / 15 :=
by
  sorry

end female_students_proportion_and_count_l254_25482


namespace dilation_rotation_l254_25420

noncomputable def center : ℂ := 2 + 3 * Complex.I
noncomputable def scale_factor : ℂ := 3
noncomputable def initial_point : ℂ := -1 + Complex.I
noncomputable def final_image : ℂ := -4 + 12 * Complex.I

theorem dilation_rotation (z : ℂ) :
  z = (-1 + Complex.I) →
  let z' := center + scale_factor * (initial_point - center)
  let rotated_z := center + Complex.I * (z' - center)
  rotated_z = final_image := sorry

end dilation_rotation_l254_25420


namespace arthur_walks_distance_l254_25498

variables (blocks_east blocks_north : ℕ) 
variable (distance_per_block : ℝ)
variable (total_blocks : ℕ)
def total_distance (blocks : ℕ) (distance_per_block : ℝ) : ℝ :=
  blocks * distance_per_block

theorem arthur_walks_distance (h_east : blocks_east = 8) (h_north : blocks_north = 10) 
    (h_total_blocks : total_blocks = blocks_east + blocks_north)
    (h_distance_per_block : distance_per_block = 1 / 4) :
  total_distance total_blocks distance_per_block = 4.5 :=
by {
  -- Here we specify the proof, but as required, we use sorry to skip it.
  sorry
}

end arthur_walks_distance_l254_25498


namespace jill_tax_on_other_items_l254_25447

-- Define the conditions based on the problem statement.
variables (C : ℝ) (x : ℝ)
def tax_on_clothing := 0.04 * 0.60 * C
def tax_on_food := 0
def tax_on_other_items := 0.01 * x * 0.30 * C
def total_tax_paid := 0.048 * C

-- Prove the required percentage tax on other items.
theorem jill_tax_on_other_items :
  tax_on_clothing C + tax_on_food + tax_on_other_items C x = total_tax_paid C →
  x = 8 :=
by
  sorry

end jill_tax_on_other_items_l254_25447


namespace digital_earth_storage_technology_matured_l254_25400

-- Definitions of conditions as technology properties
def NanoStorageTechnology : Prop := 
  -- Assume it has matured (based on solution analysis)
  sorry

def LaserHolographicStorageTechnology : Prop :=
  -- Assume it has matured (based on solution analysis)
  sorry

def ProteinStorageTechnology : Prop :=
  -- Assume it has matured (based on solution analysis)
  sorry

def DistributedStorageTechnology : Prop :=
  -- Assume it has matured (based on solution analysis)
  sorry

def VirtualStorageTechnology : Prop :=
  -- Assume it has not matured or is not relevant
  sorry

def SpatialStorageTechnology : Prop :=
  -- Assume it has not matured or is not relevant
  sorry

def VisualizationStorageTechnology : Prop :=
  -- Assume it has not matured or is not relevant
  sorry

-- Lean statement to prove the combination
theorem digital_earth_storage_technology_matured : 
  NanoStorageTechnology ∧ LaserHolographicStorageTechnology ∧ ProteinStorageTechnology ∧ DistributedStorageTechnology :=
by {
  sorry
}

end digital_earth_storage_technology_matured_l254_25400


namespace loan_proof_l254_25469

-- Definition of the conditions
def interest_rate_year_1 : ℝ := 0.10
def interest_rate_year_2 : ℝ := 0.12
def interest_rate_year_3 : ℝ := 0.14
def total_interest_paid : ℝ := 5400

-- Theorem proving the results
theorem loan_proof (P : ℝ) 
                   (annual_repayment : ℝ)
                   (remaining_principal : ℝ) :
  (interest_rate_year_1 * P) + 
  (interest_rate_year_2 * P) + 
  (interest_rate_year_3 * P) = total_interest_paid →
  3 * annual_repayment = total_interest_paid →
  remaining_principal = P →
  P = 15000 ∧ 
  annual_repayment = 1800 ∧ 
  remaining_principal = 15000 :=
by
  intros h1 h2 h3
  sorry

end loan_proof_l254_25469


namespace find_m_l254_25423

theorem find_m (m : ℝ) (h : 2^2 + 2 * m + 2 = 0) : m = -3 :=
by {
  sorry
}

end find_m_l254_25423


namespace smallest_k_no_real_roots_l254_25450

theorem smallest_k_no_real_roots :
  ∀ (k : ℤ), (∀ x : ℝ, 3 * x * (k * x - 5) - x^2 + 7 ≠ 0) → k ≥ 4 :=
by
  sorry

end smallest_k_no_real_roots_l254_25450


namespace fraction_equals_repeating_decimal_l254_25470

noncomputable def repeating_decimal_fraction : ℚ :=
  let a : ℚ := 46 / 100
  let r : ℚ := 1 / 100
  (a / (1 - r))

theorem fraction_equals_repeating_decimal :
  repeating_decimal_fraction = 46 / 99 :=
by
  sorry

end fraction_equals_repeating_decimal_l254_25470


namespace basic_computer_price_l254_25441

theorem basic_computer_price :
  ∃ C P : ℝ,
    C + P = 2500 ∧
    (C + 800) + (1 / 5) * (C + 800 + P) = 2500 ∧
    (C + 1100) + (1 / 8) * (C + 1100 + P) = 2500 ∧
    (C + 1500) + (1 / 10) * (C + 1500 + P) = 2500 ∧
    C = 1040 :=
by
  sorry

end basic_computer_price_l254_25441


namespace customers_in_other_countries_l254_25480

-- Define the given conditions

def total_customers : ℕ := 7422
def customers_us : ℕ := 723

theorem customers_in_other_countries : total_customers - customers_us = 6699 :=
by
  -- This part will contain the proof, which is not required for this task.
  sorry

end customers_in_other_countries_l254_25480


namespace percentage_decrease_after_raise_l254_25426

theorem percentage_decrease_after_raise
  (original_salary : ℝ) (final_salary : ℝ) (initial_raise_percent : ℝ)
  (initial_salary_raised : original_salary * (1 + initial_raise_percent / 100) = 5500): 
  original_salary = 5000 -> final_salary = 5225 -> initial_raise_percent = 10 ->
  ∃ (percentage_decrease : ℝ),
    final_salary = original_salary * (1 + initial_raise_percent / 100) * (1 - percentage_decrease / 100)
    ∧ percentage_decrease = 5 := by
  intros h1 h2 h3
  use 5
  rw [h1, h2, h3]
  simp
  sorry

end percentage_decrease_after_raise_l254_25426


namespace min_value_of_m_n_squared_l254_25476

theorem min_value_of_m_n_squared 
  (a b c : ℝ)
  (triangle_cond : a^2 + b^2 = c^2)
  (m n : ℝ)
  (line_cond : a * m + b * n + 3 * c = 0) 
  : m^2 + n^2 = 9 := 
by
  sorry

end min_value_of_m_n_squared_l254_25476


namespace g_five_eq_one_l254_25422

noncomputable def g : ℝ → ℝ := sorry

theorem g_five_eq_one 
  (h1 : ∀ x y : ℝ, g (x - y) = g x * g y)
  (h2 : ∀ x : ℝ, g x ≠ 0)
  (h3 : ∀ x : ℝ, g x = g (-x)) : 
  g 5 = 1 :=
sorry

end g_five_eq_one_l254_25422


namespace alpha_values_m_range_l254_25403

noncomputable section

open Real

def f (x : ℝ) (α : ℝ) : ℝ := 2^(x + cos α) - 2^(-x + cos α)

-- Problem 1: Set of values for α
theorem alpha_values (h : f 1 α = 3/4) : ∃ k : ℤ, α = 2 * k * π + π :=
sorry

-- Problem 2: Range of values for real number m
theorem m_range (h0 : 0 ≤ θ ∧ θ ≤ π / 2) 
  (h1 : ∀ (m : ℝ), f (m * cos θ) (-1) + f (1 - m) (-1) > 0) : 
  ∀ (m : ℝ), m < 1 :=
sorry

end alpha_values_m_range_l254_25403


namespace last_two_digits_of_7_pow_2017_l254_25485

theorem last_two_digits_of_7_pow_2017 :
  (7 ^ 2017) % 100 = 7 :=
sorry

end last_two_digits_of_7_pow_2017_l254_25485


namespace triangle_PQR_QR_length_l254_25451

-- Define the given conditions as a Lean statement
theorem triangle_PQR_QR_length 
  (P Q R : ℝ) -- Angles in the triangle PQR in radians
  (PQ QR PR : ℝ) -- Lengths of the sides of the triangle PQR
  (h1 : Real.cos (2 * P - Q) + Real.sin (P + 2 * Q) = 1) 
  (h2 : PQ = 5)
  (h3 : PQ + QR + PR = 12)
  : QR = 3.5 := 
  sorry -- proof omitted

end triangle_PQR_QR_length_l254_25451


namespace petya_board_problem_l254_25463

variable (A B Z : ℕ)

theorem petya_board_problem (h1 : A + B + Z = 10) (h2 : A * B = 15) : Z = 2 := sorry

end petya_board_problem_l254_25463


namespace exists_eleven_consecutive_numbers_sum_cube_l254_25419

theorem exists_eleven_consecutive_numbers_sum_cube :
  ∃ (n k : ℕ), (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) + (n+8) + (n+9) + (n+10)) = k^3 :=
by
  sorry

end exists_eleven_consecutive_numbers_sum_cube_l254_25419


namespace total_cans_collected_l254_25411

-- Definitions based on conditions
def cans_LaDonna : ℕ := 25
def cans_Prikya : ℕ := 2 * cans_LaDonna
def cans_Yoki : ℕ := 10

-- Theorem statement
theorem total_cans_collected : 
  cans_LaDonna + cans_Prikya + cans_Yoki = 85 :=
by
  -- The proof is not required, inserting sorry to complete the statement
  sorry

end total_cans_collected_l254_25411


namespace range_of_m_l254_25488

open Set

variable {α : Type}

noncomputable def A (m : ℝ) : Set ℝ := {x | m+1 ≤ x ∧ x ≤ 2*m-1}
noncomputable def B : Set ℝ := {x | x^2 - 2*x - 15 ≤ 0}

theorem range_of_m (m : ℝ) (hA : A m ⊆ B) (hA_nonempty : A m ≠ ∅) : 1 ≤ m ∧ m ≤ 3 := by
  sorry

end range_of_m_l254_25488


namespace find_e_l254_25429

-- Conditions
def f (x : ℝ) (b : ℝ) := 5 * x + b
def g (x : ℝ) (b : ℝ) := b * x + 4
def f_comp_g (x : ℝ) (b : ℝ) (e : ℝ) := 15 * x + e

-- Statement to prove
theorem find_e (b e : ℝ) (x : ℝ): 
  (f (g x b) b = f_comp_g x b e) → 
  (5 * b = 15) → 
  (20 + b = e) → 
  e = 23 :=
by 
  intros h1 h2 h3
  sorry

end find_e_l254_25429


namespace age_ratio_l254_25413

def Kul : ℕ := 22
def Saras : ℕ := 33

theorem age_ratio : (Saras / Kul : ℚ) = 3 / 2 := by
  sorry

end age_ratio_l254_25413


namespace area_of_sector_one_radian_l254_25474

theorem area_of_sector_one_radian (r θ : ℝ) (hθ : θ = 1) (hr : r = 1) : 
  (1/2 * (r * θ) * r) = 1/2 :=
by
  sorry

end area_of_sector_one_radian_l254_25474


namespace a5_is_3_l254_25421

section
variable {a : ℕ → ℝ} 
variable (h_pos : ∀ n, 0 < a n)
variable (h_a1 : a 1 = 1)
variable (h_a2 : a 2 = Real.sqrt 3)
variable (h_recursive : ∀ n ≥ 2, 2 * (a n)^2 = (a (n + 1))^2 + (a (n - 1))^2)

theorem a5_is_3 : a 5 = 3 :=
  by
  sorry
end

end a5_is_3_l254_25421


namespace solve_equation_solve_proportion_l254_25445

theorem solve_equation (x : ℚ) :
  (3 + x) * (30 / 100) = 4.8 → x = 13 :=
by sorry

theorem solve_proportion (x : ℚ) :
  (5 / x) = (9 / 2) / (8 / 5) → x = (16 / 9) :=
by sorry

end solve_equation_solve_proportion_l254_25445


namespace ratio_of_sums_eq_neg_sqrt_2_l254_25484

open Real

theorem ratio_of_sums_eq_neg_sqrt_2
    (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 6) :
    (x + y) / (x - y) = -Real.sqrt 2 :=
by sorry

end ratio_of_sums_eq_neg_sqrt_2_l254_25484


namespace probability_x_gt_3y_l254_25417

theorem probability_x_gt_3y :
  let width := 3000
  let height := 3001
  let triangle_area := (1 / 2 : ℚ) * width * (width / 3)
  let rectangle_area := (width : ℚ) * height
  triangle_area / rectangle_area = 1500 / 9003 :=
by 
  sorry

end probability_x_gt_3y_l254_25417


namespace range_of_a_for_increasing_l254_25437

noncomputable def f (a : ℝ) : (ℝ → ℝ) := λ x => x^3 + a * x^2 + 3 * x

theorem range_of_a_for_increasing (a : ℝ) : 
  (∀ x : ℝ, (3 * x^2 + 2 * a * x + 3) ≥ 0) ↔ (-3 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_of_a_for_increasing_l254_25437


namespace minimum_bottles_needed_l254_25454

theorem minimum_bottles_needed :
  (∃ n : ℕ, n * 45 ≥ 720 - 20 ∧ (n - 1) * 45 < 720 - 20) ∧ 720 - 20 = 700 :=
by
  sorry

end minimum_bottles_needed_l254_25454


namespace multiple_proof_l254_25458

noncomputable def K := 185  -- Given KJ's stamps
noncomputable def AJ := 370  -- Given AJ's stamps
noncomputable def total_stamps := 930  -- Given total amount

-- Using the conditions to find C
noncomputable def stamps_of_three := AJ + K  -- Total stamps of KJ and AJ
noncomputable def C := total_stamps - stamps_of_three

-- Stating the equivalence we need to prove
theorem multiple_proof : ∃ M: ℕ, M * K + 5 = C := by
  -- The solution proof here if required
  existsi 2
  sorry  -- proof to be completed

end multiple_proof_l254_25458


namespace derivative_at_pi_over_4_l254_25471

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem derivative_at_pi_over_4 : 
  deriv f (Real.pi / 4) = Real.sqrt 2 / 2 + Real.sqrt 2 * Real.pi / 8 :=
by
  -- Proof goes here
  sorry

end derivative_at_pi_over_4_l254_25471


namespace gravel_cost_l254_25412

def cost_per_cubic_foot := 8
def cubic_yards := 3
def cubic_feet_per_cubic_yard := 27

theorem gravel_cost :
  (cubic_yards * cubic_feet_per_cubic_yard) * cost_per_cubic_foot = 648 :=
by sorry

end gravel_cost_l254_25412


namespace parabola_chord_length_eight_l254_25433

noncomputable def parabola_intersection_length (x1 x2: ℝ) (y1 y2: ℝ) : ℝ :=
  if x1 + x2 = 6 ∧ y1^2 = 4 * x1 ∧ y2^2 = 4 * x2 then
    let A := (x1, y1)
    let B := (x2, y2)
    dist A B
  else
    0

theorem parabola_chord_length_eight :
  ∀ (x1 x2 y1 y2 : ℝ), (x1 + x2 = 6) → (y1^2 = 4 * x1) → (y2^2 = 4 * x2) →
  parabola_intersection_length x1 x2 y1 y2 = 8 :=
by
  -- proof goes here
  sorry

end parabola_chord_length_eight_l254_25433


namespace find_a_value_l254_25401

theorem find_a_value (a : ℕ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
sorry

end find_a_value_l254_25401


namespace students_average_comparison_l254_25443

theorem students_average_comparison (t1 t2 t3 : ℝ) (h : t1 < t2) (h' : t2 < t3) :
  (∃ t1 t2 t3 : ℝ, t1 < t2 ∧ t2 < t3 ∧ (t1 + t2 + t3) / 3 = (t1 + t3 + 2 * t2) / 4) ∨
  (∀ t1 t2 t3 : ℝ, t1 < t2 ∧ t2 < t3 → 
     (t1 + t3 + 2 * t2) / 4 > (t1 + t2 + t3) / 3) :=
sorry

end students_average_comparison_l254_25443


namespace solution_set_inequality_range_a_inequality_l254_25435

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) - 2

theorem solution_set_inequality (x : ℝ) (a : ℝ) (h : a = 1) :
  f x a + abs (2*x - 3) > 0 ↔ (x < 2 / 3 ∨ 2 < x) := sorry

theorem range_a_inequality (a : ℝ) :
  (∀ x, f x a < abs (x - 3)) ↔ (1 < a ∧ a < 5) := sorry

end solution_set_inequality_range_a_inequality_l254_25435


namespace saline_drip_duration_l254_25495

theorem saline_drip_duration (rate_drops_per_minute : ℕ) (drop_to_ml_rate : ℕ → ℕ → Prop)
  (ml_received : ℕ) (time_hours : ℕ) :
  rate_drops_per_minute = 20 ->
  drop_to_ml_rate 100 5 ->
  ml_received = 120 ->
  time_hours = 2 :=
by {
  sorry
}

end saline_drip_duration_l254_25495


namespace find_t_l254_25489

noncomputable def f (x t k : ℝ): ℝ := (1/3) * x^3 - (t/2) * x^2 + k * x

theorem find_t (a b t k : ℝ) (h1 : t > 0) (h2 : k > 0) 
  (h3 : a + b = t) (h4 : a * b = k)
  (h5 : 2 * a = b - 2)
  (h6 : (-2) ^ 2 = a * b) : 
  t = 5 :=
by 
  sorry

end find_t_l254_25489


namespace count_congruent_to_3_mod_8_l254_25455

theorem count_congruent_to_3_mod_8 : 
  ∃ (count : ℕ), count = 31 ∧ ∀ (x : ℕ), 1 ≤ x ∧ x ≤ 250 → x % 8 = 3 → x = 8 * ((x - 3) / 8) + 3 := sorry

end count_congruent_to_3_mod_8_l254_25455


namespace yellow_green_block_weight_difference_l254_25414

theorem yellow_green_block_weight_difference :
  let yellow_weight := 0.6
  let green_weight := 0.4
  yellow_weight - green_weight = 0.2 := by
  sorry

end yellow_green_block_weight_difference_l254_25414


namespace problem_inequality_l254_25494

variable {a b c : ℝ}

theorem problem_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_not_all_equal : ¬ (a = b ∧ b = c)) : 
  2 * (a^3 + b^3 + c^3) > a^2 * (b + c) + b^2 * (a + c) + c^2 * (a + b) :=
by sorry

end problem_inequality_l254_25494


namespace second_derivative_at_pi_over_3_l254_25465

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x) * (Real.cos x)

theorem second_derivative_at_pi_over_3 : 
  (deriv (deriv f)) (Real.pi / 3) = -1 :=
  sorry

end second_derivative_at_pi_over_3_l254_25465


namespace fraction_numerator_greater_than_denominator_l254_25409

theorem fraction_numerator_greater_than_denominator (x : ℝ) :
  -1 ≤ x ∧ x ≤ 3 ∧ x ≠ 5 / 3 → (8 / 11 < x ∧ x < 5 / 3) ∨ (5 / 3 < x ∧ x ≤ 3) ↔ (8 * x - 3 > 5 - 3 * x) := by
  sorry

end fraction_numerator_greater_than_denominator_l254_25409


namespace number_of_ways_to_form_team_l254_25491

noncomputable def binomial : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binomial n k + binomial n (k + 1)

theorem number_of_ways_to_form_team :
  let total_selections := binomial 11 5
  let all_boys_selections := binomial 8 5
  total_selections - all_boys_selections = 406 :=
by 
  sorry

end number_of_ways_to_form_team_l254_25491


namespace remove_terms_yield_desired_sum_l254_25487

-- Define the original sum and the terms to be removed
def originalSum : ℚ := 1/3 + 1/6 + 1/9 + 1/12 + 1/15 + 1/18
def termsToRemove : List ℚ := [1/9, 1/12, 1/15, 1/18]

-- Definition of the desired remaining sum
def desiredSum : ℚ := 1/2

noncomputable def sumRemainingTerms : ℚ :=
originalSum - List.sum termsToRemove

-- Lean theorem to prove
theorem remove_terms_yield_desired_sum : sumRemainingTerms = desiredSum :=
by 
  sorry

end remove_terms_yield_desired_sum_l254_25487


namespace calendar_matrix_sum_l254_25404

def initial_matrix : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![5, 6, 7], 
    ![8, 9, 10], 
    ![11, 12, 13]]

def modified_matrix (m : Matrix (Fin 3) (Fin 3) ℕ) : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![m 0 2, m 0 1, m 0 0], 
    ![m 1 0, m 1 1, m 1 2], 
    ![m 2 2, m 2 1, m 2 0]]

def diagonal_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  m 0 0 + m 1 1 + m 2 2

def edge_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  m 0 1 + m 0 2 + m 2 0 + m 2 1

def total_sum (m : Matrix (Fin 3) (Fin 3) ℕ) : ℕ :=
  diagonal_sum m + edge_sum m

theorem calendar_matrix_sum :
  total_sum (modified_matrix initial_matrix) = 63 :=
by
  sorry

end calendar_matrix_sum_l254_25404


namespace num_zeros_g_l254_25442

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  if x > 2 then m * (x - 2) / x
  else if 0 < x ∧ x ≤ 2 then 3 * x - x^2
  else 0

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := f x m - 2

-- Statement to prove
theorem num_zeros_g (m : ℝ) : ∃ n : ℕ, (n = 4 ∨ n = 6) :=
sorry

end num_zeros_g_l254_25442


namespace money_raised_by_full_price_tickets_l254_25453

theorem money_raised_by_full_price_tickets (f h : ℕ) (p revenue total_tickets : ℕ) 
  (full_price : p = 20) (total_cost : f * p + h * (p / 2) = revenue) 
  (ticket_count : f + h = total_tickets) (total_revenue : revenue = 2750)
  (ticket_number : total_tickets = 180) : f * p = 1900 := 
by
  sorry

end money_raised_by_full_price_tickets_l254_25453


namespace no_lighter_sentence_for_liar_l254_25418

theorem no_lighter_sentence_for_liar
  (total_eggs : ℕ)
  (stolen_eggs1 stolen_eggs2 stolen_eggs3 : ℕ)
  (different_stolen_eggs : stolen_eggs1 ≠ stolen_eggs2 ∧ stolen_eggs2 ≠ stolen_eggs3 ∧ stolen_eggs1 ≠ stolen_eggs3)
  (stolen_eggs1_max : stolen_eggs1 > stolen_eggs2 ∧ stolen_eggs1 > stolen_eggs3)
  (stole_7 : stolen_eggs1 = 7)
  (total_eq_20 : stolen_eggs1 + stolen_eggs2 + stolen_eggs3 = 20) :
  false :=
by
  sorry

end no_lighter_sentence_for_liar_l254_25418


namespace value_of_2x_l254_25456

theorem value_of_2x (x y z : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_eq : 2 * x = 6 * z) (h_sum : x + y + z = 26) : 2 * x = 6 := 
by
  sorry

end value_of_2x_l254_25456


namespace pipe_B_leak_time_l254_25431

theorem pipe_B_leak_time (t_B : ℝ) : (1 / 12 - 1 / t_B = 1 / 36) → t_B = 18 :=
by
  intro h
  -- Proof goes here
  sorry

end pipe_B_leak_time_l254_25431


namespace time_to_walk_2_miles_l254_25439

/-- I walked 2 miles in a certain amount of time. -/
def walked_distance : ℝ := 2

/-- If I maintained this pace for 8 hours, I would walk 16 miles. -/
def pace_condition (pace : ℝ) : Prop :=
  pace * 8 = 16

/-- Prove that it took me 1 hour to walk 2 miles. -/
theorem time_to_walk_2_miles (t : ℝ) (pace : ℝ) (h1 : walked_distance = pace * t) (h2 : pace_condition pace) :
  t = 1 :=
sorry

end time_to_walk_2_miles_l254_25439


namespace prob_queen_then_diamond_is_correct_l254_25444

/-- Define the probability of drawing a Queen first and a diamond second -/
def prob_queen_then_diamond : ℚ := (3 / 52) * (13 / 51) + (1 / 52) * (12 / 51)

/-- The probability that the first card is a Queen and the second card is a diamond is 18/221 -/
theorem prob_queen_then_diamond_is_correct : prob_queen_then_diamond = 18 / 221 :=
by
  sorry

end prob_queen_then_diamond_is_correct_l254_25444


namespace cars_transfer_equation_l254_25415

theorem cars_transfer_equation (x : ℕ) : 100 - x = 68 + x :=
sorry

end cars_transfer_equation_l254_25415


namespace power_mod_l254_25452

theorem power_mod (n m : ℕ) (hn : n = 13) (hm : m = 1000) : n ^ 21 % m = 413 :=
by
  rw [hn, hm]
  -- other steps of the proof would go here...
  sorry

end power_mod_l254_25452


namespace line_intersects_ellipse_l254_25473

theorem line_intersects_ellipse (k : ℝ) (m : ℝ) : 
  (∀ x y : ℝ, y = k * x + 1 → (x^2 / 5) + (y^2 / m) = 1 → True) ↔ (1 < m ∧ m < 5) ∨ (5 < m) :=
by
  sorry

end line_intersects_ellipse_l254_25473


namespace nap_duration_is_two_hours_l254_25477

-- Conditions as definitions in Lean
def naps_per_week : ℕ := 3
def days : ℕ := 70
def total_nap_hours : ℕ := 60

-- Calculate the duration of each nap
theorem nap_duration_is_two_hours :
  ∃ (nap_duration : ℕ), nap_duration = 2 ∧
  (days / 7) * naps_per_week * nap_duration = total_nap_hours :=
by
  sorry

end nap_duration_is_two_hours_l254_25477


namespace students_with_uncool_parents_correct_l254_25438

def total_students : ℕ := 30
def cool_dads : ℕ := 12
def cool_moms : ℕ := 15
def cool_both : ℕ := 9

def students_with_uncool_parents : ℕ :=
  total_students - (cool_dads + cool_moms - cool_both)

theorem students_with_uncool_parents_correct :
  students_with_uncool_parents = 12 := by
  sorry

end students_with_uncool_parents_correct_l254_25438


namespace work_completion_l254_25497

variable (A B : Type)

/-- A can do half of the work in 70 days and B can do one third of the work in 35 days.
Together, A and B can complete the work in 60 days. -/
theorem work_completion (hA : (1 : ℚ) / 2 / 70 = (1 : ℚ) / a) 
                      (hB : (1 : ℚ) / 3 / 35 = (1 : ℚ) / b) :
                      (1 / 140 + 1 / 105) = 1 / 60 :=
  sorry

end work_completion_l254_25497


namespace max_area_difference_160_perimeter_rectangles_l254_25478

theorem max_area_difference_160_perimeter_rectangles : 
  ∃ (l1 w1 l2 w2 : ℕ), (2 * l1 + 2 * w1 = 160) ∧ (2 * l2 + 2 * w2 = 160) ∧ 
  (l1 * w1 - l2 * w2 = 1521) := sorry

end max_area_difference_160_perimeter_rectangles_l254_25478


namespace c_d_not_true_l254_25460

variables (Beatles_haircut : Type → Prop) (hooligan : Type → Prop) (rude : Type → Prop)

-- Conditions
axiom a : ∃ x, Beatles_haircut x ∧ hooligan x
axiom b : ∀ y, hooligan y → rude y

-- Prove there is a rude hooligan with a Beatles haircut
theorem c : ∃ z, rude z ∧ Beatles_haircut z ∧ hooligan z :=
sorry

-- Disprove every rude hooligan having a Beatles haircut
theorem d_not_true : ¬(∀ w, rude w ∧ hooligan w → Beatles_haircut w) :=
sorry

end c_d_not_true_l254_25460


namespace length_of_garden_side_l254_25483

theorem length_of_garden_side (perimeter : ℝ) (side_length : ℝ) (h1 : perimeter = 112) (h2 : perimeter = 4 * side_length) : 
  side_length = 28 :=
by
  sorry

end length_of_garden_side_l254_25483


namespace min_value_c_and_d_l254_25466

theorem min_value_c_and_d (c d : ℝ) (h1 : c > 0) (h2 : d > 0)
  (h3 : c^2 - 12 * d ≥ 0)
  (h4 : 9 * d^2 - 4 * c ≥ 0) :
  c + d ≥ 5.74 :=
sorry

end min_value_c_and_d_l254_25466


namespace division_problem_l254_25406

theorem division_problem : 96 / (8 / 4) = 48 := 
by {
  sorry
}

end division_problem_l254_25406


namespace cos_alpha_third_quadrant_l254_25472

theorem cos_alpha_third_quadrant (α : ℝ) (h1 : Real.sin α = -5 / 13) (h2 : Real.tan α > 0) : Real.cos α = -12 / 13 := 
sorry

end cos_alpha_third_quadrant_l254_25472


namespace solve_trig_problem_l254_25499

-- Definition of the given problem for trigonometric identities
def problem_statement : Prop :=
  (1 - Real.tan (Real.pi / 12)) / (1 + Real.tan (Real.pi / 12)) = Real.sqrt 3 / 3

theorem solve_trig_problem : problem_statement :=
  by
  sorry -- No proof is needed here

end solve_trig_problem_l254_25499


namespace andy_paint_total_l254_25416

-- Define the given ratio condition and green paint usage
def paint_ratio (blue green white : ℕ) : Prop :=
  blue / green = 1 / 2 ∧ white / green = 5 / 2

def green_paint_used (green : ℕ) : Prop :=
  green = 6

-- Define the proof goal: total paint used
def total_paint_used (blue green white : ℕ) : ℕ :=
  blue + green + white

-- The statement to be proved
theorem andy_paint_total (blue green white : ℕ)
  (h_ratio : paint_ratio blue green white)
  (h_green : green_paint_used green) :
  total_paint_used blue green white = 24 :=
  sorry

end andy_paint_total_l254_25416


namespace original_number_of_men_l254_25462

theorem original_number_of_men (x : ℕ) (h1 : 40 * x = 60 * (x - 5)) : x = 15 :=
by
  sorry

end original_number_of_men_l254_25462


namespace convex_polygon_sides_l254_25459

theorem convex_polygon_sides (n : ℕ) (h : ∀ angle, angle = 45 → angle * n = 360) : n = 8 :=
  sorry

end convex_polygon_sides_l254_25459


namespace blood_drug_concentration_at_13_hours_l254_25402

theorem blood_drug_concentration_at_13_hours :
  let peak_time := 3
  let test_interval := 2
  let decrease_rate := 0.4
  let target_rate := 0.01024
  let time_to_reach_target := (fun n => (2 * n + 1))
  peak_time + test_interval * 5 = 13 :=
sorry

end blood_drug_concentration_at_13_hours_l254_25402


namespace find_k_l254_25481

def total_balls (k : ℕ) : ℕ := 7 + k

def probability_green (k : ℕ) : ℚ := 7 / (total_balls k)
def probability_purple (k : ℕ) : ℚ := k / (total_balls k)

def expected_value (k : ℕ) : ℚ :=
  (probability_green k) * 3 + (probability_purple k) * (-1)

theorem find_k (k : ℕ) (h_pos : k > 0) (h_exp_value : expected_value k = 1) : k = 7 :=
sorry

end find_k_l254_25481


namespace angle_same_terminal_side_l254_25461

theorem angle_same_terminal_side (k : ℤ) : ∃ k : ℤ, 95 = -265 + k * 360 :=
by
  use 1
  norm_num

end angle_same_terminal_side_l254_25461


namespace james_and_lisa_pizzas_l254_25457

theorem james_and_lisa_pizzas (slices_per_pizza : ℕ) (total_slices : ℕ) :
  slices_per_pizza = 6 →
  2 * total_slices = 3 * 8 →
  total_slices / slices_per_pizza = 2 :=
by
  intros h1 h2
  sorry

end james_and_lisa_pizzas_l254_25457


namespace vertex_of_given_function_l254_25464

noncomputable def vertex_coordinates (f : ℝ → ℝ) : ℝ × ℝ := 
  (-2, 1)  -- Prescribed coordinates for this specific function form.

def function_vertex (x : ℝ) : ℝ :=
  -3 * (x + 2) ^ 2 + 1

theorem vertex_of_given_function : 
  vertex_coordinates function_vertex = (-2, 1) :=
by
  sorry

end vertex_of_given_function_l254_25464


namespace sum_smallest_largest_consecutive_even_integers_l254_25490

theorem sum_smallest_largest_consecutive_even_integers
  (n : ℕ) (a y : ℤ) 
  (hn_even : Even n) 
  (h_mean : y = (a + (a + 2 * (n - 1))) / 2) :
  2 * y = (a + (a + 2 * (n - 1))) :=
by
  sorry

end sum_smallest_largest_consecutive_even_integers_l254_25490


namespace minimum_value_of_quadratic_function_l254_25430

variable (p q : ℝ) (hp : 0 < p) (hq : 0 < q)

theorem minimum_value_of_quadratic_function : 
  (∃ x : ℝ, x = p) ∧ (∀ x : ℝ, (x^2 - 2 * p * x + 4 * q) ≥ (p^2 - 2 * p * p + 4 * q)) :=
sorry

end minimum_value_of_quadratic_function_l254_25430


namespace age_of_17th_student_l254_25479

theorem age_of_17th_student (avg_age_17 : ℕ) (total_students : ℕ) (avg_age_5 : ℕ) (students_5 : ℕ) (avg_age_9 : ℕ) (students_9 : ℕ)
  (h1 : avg_age_17 = 17) (h2 : total_students = 17) (h3 : avg_age_5 = 14) (h4 : students_5 = 5) (h5 : avg_age_9 = 16) (h6 : students_9 = 9) :
  ∃ age_17th_student : ℕ, age_17th_student = 75 :=
by
  sorry

end age_of_17th_student_l254_25479


namespace garden_length_l254_25407

noncomputable def length_of_garden : ℝ := 300

theorem garden_length (P : ℝ) (b : ℝ) (A : ℝ) 
  (h₁ : P = 800) (h₂ : b = 100) (h₃ : A = 10000) : length_of_garden = 300 := 
by 
  sorry

end garden_length_l254_25407


namespace decreasing_sequence_b_l254_25405

def seq_a (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, 2 * a n * a (n + 1) = (a n)^2 + 1

def b_n (a b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n = (a n - 1) / (a n + 1)

theorem decreasing_sequence_b {a b : ℕ → ℝ} (h1 : seq_a a) (h2 : b_n a b) :
  ∀ n : ℕ, b (n + 1) < b n :=
by
  sorry

end decreasing_sequence_b_l254_25405
