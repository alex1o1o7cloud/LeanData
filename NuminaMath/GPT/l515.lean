import Mathlib

namespace pat_mark_ratio_l515_51598

theorem pat_mark_ratio :
  ∃ K P M : ℕ, P + K + M = 189 ∧ P = 2 * K ∧ M = K + 105 ∧ P / gcd P M = 1 ∧ M / gcd P M = 3 :=
by
  sorry

end pat_mark_ratio_l515_51598


namespace seashells_collected_l515_51559

theorem seashells_collected (x y z : ℕ) (hyp : x + y / 2 + z + 5 = 76) : x + y + z = 71 := 
by {
  sorry
}

end seashells_collected_l515_51559


namespace maximum_possible_angle_Z_l515_51503

theorem maximum_possible_angle_Z (X Y Z : ℝ) (h1 : Z ≤ Y) (h2 : Y ≤ X) (h3 : 2 * X = 6 * Z) (h4 : X + Y + Z = 180) : Z = 36 :=
by
  sorry

end maximum_possible_angle_Z_l515_51503


namespace original_number_of_people_l515_51553

theorem original_number_of_people (x : ℕ) (h1 : x - x / 3 + (x / 3) * 3/4 = x * 1/4 + 15) : x = 30 :=
sorry

end original_number_of_people_l515_51553


namespace students_not_reading_l515_51549

theorem students_not_reading (total_girls : ℕ) (total_boys : ℕ)
  (frac_girls_reading : ℚ) (frac_boys_reading : ℚ)
  (h1 : total_girls = 12) (h2 : total_boys = 10)
  (h3 : frac_girls_reading = 5 / 6) (h4 : frac_boys_reading = 4 / 5) :
  let girls_not_reading := total_girls - total_girls * frac_girls_reading
  let boys_not_reading := total_boys - total_boys * frac_boys_reading
  let total_not_reading := girls_not_reading + boys_not_reading
  total_not_reading = 4 := sorry

end students_not_reading_l515_51549


namespace first_discount_percentage_l515_51578

theorem first_discount_percentage
  (list_price : ℝ)
  (second_discount : ℝ)
  (third_discount : ℝ)
  (tax_rate : ℝ)
  (final_price : ℝ)
  (D1 : ℝ)
  (h_list_price : list_price = 150)
  (h_second_discount : second_discount = 12)
  (h_third_discount : third_discount = 5)
  (h_tax_rate : tax_rate = 10)
  (h_final_price : final_price = 105) :
  100 - 100 * (final_price / (list_price * (1 - D1 / 100) * (1 - second_discount / 100) * (1 - third_discount / 100) * (1 + tax_rate / 100))) = 24.24 :=
by
  sorry

end first_discount_percentage_l515_51578


namespace sum_x_coordinates_midpoints_l515_51580

theorem sum_x_coordinates_midpoints (a b c : ℝ) (h : a + b + c = 12) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 12 :=
by
  sorry

end sum_x_coordinates_midpoints_l515_51580


namespace total_views_correct_l515_51573

-- Definitions based on the given conditions
def initial_views : ℕ := 4000
def views_increase := 10 * initial_views
def additional_views := 50000
def total_views_after_6_days := initial_views + views_increase + additional_views

-- The theorem we are going to state
theorem total_views_correct :
  total_views_after_6_days = 94000 :=
sorry

end total_views_correct_l515_51573


namespace correct_X_Y_Z_l515_51560

def nucleotide_types (A_types C_types T_types : ℕ) : ℕ :=
  A_types + C_types + T_types

def lowest_stability_period := "interphase"

def separation_period := "late meiosis I or late meiosis II"

theorem correct_X_Y_Z :
  nucleotide_types 2 2 1 = 3 ∧ 
  lowest_stability_period = "interphase" ∧ 
  separation_period = "late meiosis I or late meiosis II" :=
by
  sorry

end correct_X_Y_Z_l515_51560


namespace min_value_l515_51533

theorem min_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
    (h3 : (a - 1) * 1 + 1 * (2 * b) = 0) :
    (2 / a) + (1 / b) = 8 :=
  sorry

end min_value_l515_51533


namespace solve_for_sum_l515_51552

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := -1
noncomputable def c : ℝ := Real.sqrt 26

theorem solve_for_sum :
  (a * (a - 4) = 5) ∧ (b * (b - 4) = 5) ∧ (c * (c - 4) = 5) ∧ 
  (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧ (a^2 + b^2 = c^2) → (a + b + c = 4 + Real.sqrt 26) :=
by
  sorry

end solve_for_sum_l515_51552


namespace number_of_girls_l515_51529

theorem number_of_girls (B G : ℕ) (ratio_condition : B = G / 2) (total_condition : B + G = 90) : 
  G = 60 := 
by
  -- This is the problem statement, with conditions and required result.
  sorry

end number_of_girls_l515_51529


namespace xyz_inequality_l515_51520

theorem xyz_inequality (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z ≥ 1) :
    (x^4 + y) * (y^4 + z) * (z^4 + x) ≥ (x + y^2) * (y + z^2) * (z + x^2) :=
by
  sorry

end xyz_inequality_l515_51520


namespace total_interest_at_tenth_year_l515_51535

-- Define the conditions for the simple interest problem
variables (P R T : ℝ)

-- Given conditions in the problem
def initial_condition : Prop := (P * R * 10) / 100 = 800
def trebled_principal_condition : Prop := (3 * P * R * 5) / 100 = 1200

-- Statement to prove
theorem total_interest_at_tenth_year (h1 : initial_condition P R) (h2 : trebled_principal_condition P R) :
  (800 + 1200) = 2000 := by
  sorry

end total_interest_at_tenth_year_l515_51535


namespace fraction_sum_to_decimal_l515_51557

theorem fraction_sum_to_decimal :
  (3 / 20 : ℝ) + (5 / 200 : ℝ) + (7 / 2000 : ℝ) = 0.1785 :=
by 
  sorry

end fraction_sum_to_decimal_l515_51557


namespace find_frac_a_b_c_l515_51504

theorem find_frac_a_b_c (a b c : ℝ) (h1 : a = 2 * b) (h2 : a^2 + b^2 = c^2) : (a + b) / c = (3 * Real.sqrt 5) / 5 :=
by
  sorry

end find_frac_a_b_c_l515_51504


namespace leftover_potatoes_l515_51548

theorem leftover_potatoes (fries_per_potato : ℕ) (total_potatoes : ℕ) (required_fries : ℕ)
    (h1 : fries_per_potato = 25) (h2 : total_potatoes = 15) (h3 : required_fries = 200) :
    (total_potatoes - required_fries / fries_per_potato) = 7 :=
sorry

end leftover_potatoes_l515_51548


namespace complement_of_A_in_U_eq_l515_51567

def U : Set ℝ := {x | x > 0}
def A : Set ℝ := {x | x ≥ Real.exp 1}
def complement_U_A : Set ℝ := {x | 0 < x ∧ x ≤ Real.exp 1}

theorem complement_of_A_in_U_eq : 
  (U \ A) = complement_U_A := 
by
  sorry

end complement_of_A_in_U_eq_l515_51567


namespace tunnel_length_proof_l515_51507

variable (train_length : ℝ) (train_speed : ℝ) (time_in_tunnel : ℝ)

noncomputable def tunnel_length (train_length train_speed time_in_tunnel : ℝ) : ℝ :=
  (train_speed / 60) * time_in_tunnel - train_length

theorem tunnel_length_proof 
  (h_train_length : train_length = 2) 
  (h_train_speed : train_speed = 30) 
  (h_time_in_tunnel : time_in_tunnel = 4) : 
  tunnel_length 2 30 4 = 2 := by
    simp [tunnel_length, h_train_length, h_train_speed, h_time_in_tunnel]
    norm_num
    sorry

end tunnel_length_proof_l515_51507


namespace find_y_l515_51558

theorem find_y (x y: ℤ) (h1: x^2 - 3 * x + 2 = y + 6) (h2: x = -4) : y = 24 :=
by
  sorry

end find_y_l515_51558


namespace find_f_2017_l515_51588

theorem find_f_2017 (f : ℤ → ℤ) (h_odd : ∀ x, f (-x) = -f x) (h_period : ∀ x, f (x + 3) = f x) (h_f_neg1 : f (-1) = 1) : 
  f 2017 = -1 :=
sorry

end find_f_2017_l515_51588


namespace find_min_value_l515_51510

theorem find_min_value (a b : ℝ) (h1 : a > 0) (h2 : b > 1) (h3 : a + b = 2) : 
  (1 / (2 * a)) + (2 / (b - 1)) ≥ 9 / 2 :=
by
  sorry

end find_min_value_l515_51510


namespace variance_transformed_is_8_l515_51531

variables {n : ℕ} (x : Fin n → ℝ)

-- Given: the variance of x₁, x₂, ..., xₙ is 2.
def variance_x (x : Fin n → ℝ) : ℝ := sorry

axiom variance_x_is_2 : variance_x x = 2

-- Variance of 2 * x₁ + 3, 2 * x₂ + 3, ..., 2 * xₙ + 3
def variance_transformed (x : Fin n → ℝ) : ℝ :=
  variance_x (fun i => 2 * x i + 3)

-- Prove that the variance is 8.
theorem variance_transformed_is_8 : variance_transformed x = 8 :=
  sorry

end variance_transformed_is_8_l515_51531


namespace satisfies_negative_inverse_l515_51530

noncomputable def f1 (x : ℝ) : ℝ := x - 1/x
noncomputable def f2 (x : ℝ) : ℝ := x + 1/x
noncomputable def f3 (x : ℝ) : ℝ := Real.log x
noncomputable def f4 (x : ℝ) : ℝ :=
  if x < 1 then x
  else if x = 1 then 0
  else -1/x

theorem satisfies_negative_inverse :
  { f | (∀ x : ℝ, f (1 / x) = -f x) } = {f1, f3, f4} :=
sorry

end satisfies_negative_inverse_l515_51530


namespace bushes_needed_l515_51523

theorem bushes_needed
  (num_sides : ℕ) (side_length : ℝ) (bush_fill : ℝ) (total_length : ℝ) (num_bushes : ℕ) :
  num_sides = 3 ∧ side_length = 16 ∧ bush_fill = 4 ∧ total_length = num_sides * side_length ∧ num_bushes = total_length / bush_fill →
  num_bushes = 12 := by
  sorry

end bushes_needed_l515_51523


namespace inequality_solution_l515_51595

theorem inequality_solution (x : ℝ) : x^2 - 2 * x - 5 > 2 * x ↔ x > 5 ∨ x < -1 :=
by
  sorry

end inequality_solution_l515_51595


namespace parker_savings_l515_51547

-- Define the costs of individual items and meals
def burger_cost : ℝ := 5
def fries_cost : ℝ := 3
def drink_cost : ℝ := 3
def special_meal_cost : ℝ := 9.5
def kids_burger_cost : ℝ := 3
def kids_fries_cost : ℝ := 2
def kids_drink_cost : ℝ := 2
def kids_meal_cost : ℝ := 5

-- Define the number of meals Mr. Parker buys
def adult_meals : ℕ := 2
def kids_meals : ℕ := 2

-- Define the total cost of individual items for adults and children
def total_individual_cost_adults : ℝ :=
  adult_meals * (burger_cost + fries_cost + drink_cost)

def total_individual_cost_children : ℝ :=
  kids_meals * (kids_burger_cost + kids_fries_cost + kids_drink_cost)

-- Define the total cost of meal deals
def total_meals_cost : ℝ :=
  adult_meals * special_meal_cost + kids_meals * kids_meal_cost

-- Define the total cost of individual items for both adults and children
def total_individual_cost : ℝ :=
  total_individual_cost_adults + total_individual_cost_children

-- Define the savings
def savings : ℝ := total_individual_cost - total_meals_cost

theorem parker_savings : savings = 7 :=
by
  sorry

end parker_savings_l515_51547


namespace part1_part2_l515_51501

-- Part 1
noncomputable def f (x a : ℝ) : ℝ := (x - 1) * Real.exp x - (1/3) * a * x ^ 3 - (1/2) * x ^ 2

noncomputable def f' (x a : ℝ) : ℝ := x * Real.exp x - a * x ^ 2 - x

noncomputable def g (x a : ℝ) : ℝ := f' x a / x

theorem part1 (a : ℝ) (h : a > 0) : g a a > 0 := by
  sorry

-- Part 2
theorem part2 (a : ℝ) (h : ∃ x, f' x a = 0) : a > 0 := by
  sorry

end part1_part2_l515_51501


namespace distance_24_km_l515_51581

noncomputable def distance_between_house_and_school (D : ℝ) :=
  let speed_to_school := 6
  let speed_to_home := 4
  let total_time := 10
  total_time = (D / speed_to_school) + (D / speed_to_home)

theorem distance_24_km : ∃ D : ℝ, distance_between_house_and_school D ∧ D = 24 :=
by
  use 24
  unfold distance_between_house_and_school
  sorry

end distance_24_km_l515_51581


namespace sequence_increasing_range_of_a_l515_51564

theorem sequence_increasing_range_of_a :
  ∀ {a : ℝ}, (∀ n : ℕ, 
    (n ≤ 7 → (4 - a) * n - 10 ≤ (4 - a) * (n + 1) - 10) ∧ 
    (7 < n → a^(n - 6) ≤ a^(n - 5))
  ) → 2 < a ∧ a < 4 :=
by
  sorry

end sequence_increasing_range_of_a_l515_51564


namespace students_on_bus_after_stops_l515_51599

-- Definitions
def initial_students : ℕ := 10
def first_stop_off : ℕ := 3
def first_stop_on : ℕ := 2
def second_stop_off : ℕ := 1
def second_stop_on : ℕ := 4
def third_stop_off : ℕ := 2
def third_stop_on : ℕ := 3

-- Theorem statement
theorem students_on_bus_after_stops :
  let after_first_stop := initial_students - first_stop_off + first_stop_on
  let after_second_stop := after_first_stop - second_stop_off + second_stop_on
  let after_third_stop := after_second_stop - third_stop_off + third_stop_on
  after_third_stop = 13 := 
by
  sorry

end students_on_bus_after_stops_l515_51599


namespace pyramid_volume_l515_51579

noncomputable def volume_of_pyramid (a b c d: ℝ) (diagonal: ℝ) (angle: ℝ) : ℝ :=
  if (a = 10 ∧ d = 10 ∧ b = 5 ∧ c = 5 ∧ diagonal = 4 * Real.sqrt 5 ∧ angle = 45) then
    let base_area := 1 / 2 * (diagonal) * (Real.sqrt ((c * c) + (b * b)))
    let height := 10 / 3
    let volume := 1 / 3 * base_area * height
    volume
  else 0

theorem pyramid_volume :
  volume_of_pyramid 10 5 5 10 (4 * Real.sqrt 5) 45 = 500 / 9 :=
by
  sorry

end pyramid_volume_l515_51579


namespace sum_of_edges_proof_l515_51590

noncomputable def sum_of_edges (a r : ℝ) : ℝ :=
  let l1 := a / r
  let l2 := a
  let l3 := a * r
  4 * (l1 + l2 + l3)

theorem sum_of_edges_proof : 
  ∀ (a r : ℝ), 
  (a > 0 ∧ r > 0 ∧ (a / r) * a * (a * r) = 512 ∧ 2 * ((a^2 / r) + a^2 + a^2 * r) = 384) → sum_of_edges a r = 96 :=
by
  intros a r h
  -- We skip the proof here with sorry
  sorry

end sum_of_edges_proof_l515_51590


namespace unique_solution_pair_l515_51583

theorem unique_solution_pair (x p : ℕ) (hp : Nat.Prime p) (hx : x ≥ 0) (hp2 : p ≥ 2) :
  x * (x + 1) * (x + 2) * (x + 3) = 1679 ^ (p - 1) + 1680 ^ (p - 1) + 1681 ^ (p - 1) ↔ (x = 4 ∧ p = 2) := 
by
  sorry

end unique_solution_pair_l515_51583


namespace consecutive_ints_product_div_6_l515_51519

theorem consecutive_ints_product_div_6 (n : ℤ) : (n * (n + 1) * (n + 2)) % 6 = 0 := 
sorry

end consecutive_ints_product_div_6_l515_51519


namespace range_of_x_in_function_l515_51586

theorem range_of_x_in_function (x : ℝ) :
  (x - 1 ≥ 0) ∧ (x - 2 ≠ 0) → (x ≥ 1 ∧ x ≠ 2) :=
by
  intro h
  sorry

end range_of_x_in_function_l515_51586


namespace common_difference_of_arithmetic_sequence_l515_51543

/--
Given an arithmetic sequence {a_n}, the sum of the first n terms is S_n,
a_3 and a_7 are the two roots of the equation 2x^2 - 12x + c = 0,
and S_{13} = c.
Prove that the common difference of the sequence {a_n} satisfies d = -3/2 or d = -7/4.
-/
theorem common_difference_of_arithmetic_sequence 
  (S : ℕ → ℚ)
  (a : ℕ → ℚ)
  (c : ℚ)
  (h1 : ∃ a_3 a_7, (2 * a_3^2 - 12 * a_3 + c = 0) ∧ (2 * a_7^2 - 12 * a_7 + c = 0))
  (h2 : S 13 = c) :
  ∃ d : ℚ, d = -3/2 ∨ d = -7/4 :=
sorry

end common_difference_of_arithmetic_sequence_l515_51543


namespace vector_simplification_l515_51551

-- Define vectors AB, CD, AC, and BD
variables {V : Type*} [AddCommGroup V]

-- Given vectors
variables (AB CD AC BD : V)

-- Theorem to be proven
theorem vector_simplification :
  (AB - CD) - (AC - BD) = (0 : V) :=
sorry

end vector_simplification_l515_51551


namespace total_wheels_at_park_l515_51522

-- Define the problem based on the given conditions
def num_bicycles : ℕ := 6
def num_tricycles : ℕ := 15
def wheels_per_bicycle : ℕ := 2
def wheels_per_tricycle : ℕ := 3

-- Statement to prove the total number of wheels is 57
theorem total_wheels_at_park : (num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle) = 57 := by
  -- This will be filled in with the proof.
  sorry

end total_wheels_at_park_l515_51522


namespace quadratic_root_relation_l515_51591

theorem quadratic_root_relation (x₁ x₂ : ℝ) (h₁ : x₁ ^ 2 - 3 * x₁ + 2 = 0) (h₂ : x₂ ^ 2 - 3 * x₂ + 2 = 0) :
  x₁ + x₂ - x₁ * x₂ = 1 := by
sorry

end quadratic_root_relation_l515_51591


namespace task_completion_l515_51570

theorem task_completion (x y z : ℝ) 
  (h1 : 1 / x + 1 / y = 1 / 2)
  (h2 : 1 / y + 1 / z = 1 / 4)
  (h3 : 1 / z + 1 / x = 5 / 12) :
  x = 3 := 
sorry

end task_completion_l515_51570


namespace m_cubed_plus_m_inv_cubed_l515_51506

theorem m_cubed_plus_m_inv_cubed (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 1 = 971 :=
sorry

end m_cubed_plus_m_inv_cubed_l515_51506


namespace solve_for_x_l515_51575

theorem solve_for_x (x : ℝ) : 4 * x - 8 + 3 * x = 12 + 5 * x → x = 10 :=
by
  intro h
  sorry

end solve_for_x_l515_51575


namespace Nicole_has_69_clothes_l515_51554

def clothingDistribution : Prop :=
  let nicole_clothes := 15
  let first_sister_clothes := nicole_clothes / 3
  let second_sister_clothes := nicole_clothes + 5
  let third_sister_clothes := 2 * first_sister_clothes
  let average_clothes := (nicole_clothes + first_sister_clothes + second_sister_clothes + third_sister_clothes) / 4
  let oldest_sister_clothes := 1.5 * average_clothes
  let total_clothes := nicole_clothes + first_sister_clothes + second_sister_clothes + third_sister_clothes + oldest_sister_clothes
  total_clothes = 69

theorem Nicole_has_69_clothes : clothingDistribution :=
by
  -- Proof omitted
  sorry

end Nicole_has_69_clothes_l515_51554


namespace probability_of_odd_sum_rows_columns_l515_51525

open BigOperators

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def probability_odd_sums : ℚ :=
  let even_arrangements := factorial 4
  let odd_positions := factorial 12
  let total_arrangements := factorial 16
  (even_arrangements * odd_positions : ℚ) / total_arrangements

theorem probability_of_odd_sum_rows_columns :
  probability_odd_sums = 1 / 1814400 :=
by
  sorry

end probability_of_odd_sum_rows_columns_l515_51525


namespace four_g_users_scientific_notation_l515_51534

-- Condition for scientific notation
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10

-- Given problem in scientific notation form
theorem four_g_users_scientific_notation :
  ∃ a n, is_scientific_notation a n 1030000000 ∧ a = 1.03 ∧ n = 9 :=
sorry

end four_g_users_scientific_notation_l515_51534


namespace simplify_expression_l515_51544

theorem simplify_expression (a b : ℝ) : 
  (2 * a^2 * b - 5 * a * b) - 2 * (-a * b + a^2 * b) = -3 * a * b :=
by
  sorry

end simplify_expression_l515_51544


namespace smallest_side_is_10_l515_51528

noncomputable def smallest_side_of_triangle (x : ℝ) : ℝ :=
    let side1 := 10
    let side2 := 3 * x + 6
    let side3 := x + 5
    min side1 (min side2 side3)

theorem smallest_side_is_10 (x : ℝ) (h : 10 + (3 * x + 6) + (x + 5) = 60) : 
    smallest_side_of_triangle x = 10 :=
by
    sorry

end smallest_side_is_10_l515_51528


namespace final_price_after_discounts_l515_51505

theorem final_price_after_discounts (m : ℝ) : (0.8 * m - 10) = selling_price :=
by
  sorry

end final_price_after_discounts_l515_51505


namespace roxy_bought_flowering_plants_l515_51556

-- Definitions based on conditions
def initial_flowering_plants : ℕ := 7
def initial_fruiting_plants : ℕ := 2 * initial_flowering_plants
def plants_after_saturday (F : ℕ) : ℕ := initial_flowering_plants + F + initial_fruiting_plants + 2
def plants_after_sunday (F : ℕ) : ℕ := (initial_flowering_plants + F - 1) + (initial_fruiting_plants + 2 - 4)
def final_plants_in_garden : ℕ := 21

-- The proof statement
theorem roxy_bought_flowering_plants (F : ℕ) :
  plants_after_sunday F = final_plants_in_garden → F = 3 := 
sorry

end roxy_bought_flowering_plants_l515_51556


namespace sum_of_ages_is_20_l515_51511

-- Given conditions
variables (age_kiana age_twin : ℕ)
axiom product_of_ages : age_kiana * age_twin * age_twin = 162

-- Required proof
theorem sum_of_ages_is_20 : age_kiana + age_twin + age_twin = 20 :=
sorry

end sum_of_ages_is_20_l515_51511


namespace solve_system_of_equations_l515_51555

theorem solve_system_of_equations :
  ∀ x y z : ℝ,
  (3 * x * y - 5 * y * z - x * z = 3 * y) →
  (x * y + y * z = -y) →
  (-5 * x * y + 4 * y * z + x * z = -4 * y) →
  (x = 2 ∧ y = -1 / 3 ∧ z = -3) ∨ 
  (y = 0 ∧ z = 0) ∨ 
  (x = 0 ∧ y = 0) :=
by
  sorry

end solve_system_of_equations_l515_51555


namespace a_is_constant_l515_51546

variable (a : ℕ → ℝ)
variable (h_pos : ∀ n, 0 < a n)
variable (h_ineq : ∀ n, a n ≥ (a (n+2) + a (n+1) + a (n-1) + a (n-2)) / 4)

theorem a_is_constant : ∀ n m, a n = a m :=
by
  sorry

end a_is_constant_l515_51546


namespace all_palindromes_divisible_by_11_probability_palindrome_divisible_by_11_l515_51532

theorem all_palindromes_divisible_by_11 : 
  (∀ a b : ℕ, 1 <= a ∧ a <= 9 ∧ 0 <= b ∧ b <= 9 →
    (1001 * a + 110 * b) % 11 = 0 ) := sorry

theorem probability_palindrome_divisible_by_11 : 
  (∀ (palindromes : ℕ → Prop), 
  (∀ n, palindromes n ↔ ∃ (a b : ℕ), 
  1 <= a ∧ a <= 9 ∧ 0 <= b ∧ b <= 9 ∧ 
  n = 1001 * a + 110 * b) → 
  (∀ n, palindromes n → n % 11 = 0) →
  ∃ p : ℝ, p = 1) := sorry

end all_palindromes_divisible_by_11_probability_palindrome_divisible_by_11_l515_51532


namespace evaluate_expression_l515_51565

theorem evaluate_expression (x y : ℕ) (hx : x = 3) (hy : y = 2) : 4 * x^y + 5 * y^x = 76 := by
  sorry

end evaluate_expression_l515_51565


namespace temperature_below_75_l515_51593

theorem temperature_below_75
  (T : ℝ)
  (H1 : ∀ T, T ≥ 75 → swimming_area_open)
  (H2 : ¬swimming_area_open) : 
  T < 75 :=
sorry

end temperature_below_75_l515_51593


namespace jill_arrives_30_minutes_before_jack_l515_51542

theorem jill_arrives_30_minutes_before_jack
    (d : ℝ) (s_jill : ℝ) (s_jack : ℝ) (t_diff : ℝ)
    (h_d : d = 2)
    (h_s_jill : s_jill = 12)
    (h_s_jack : s_jack = 3)
    (h_t_diff : t_diff = 30) :
    ((d / s_jack) * 60 - (d / s_jill) * 60) = t_diff :=
by
  sorry

end jill_arrives_30_minutes_before_jack_l515_51542


namespace fixed_errors_correct_l515_51561

-- Conditions
def total_lines_of_code : ℕ := 4300
def lines_per_debug : ℕ := 100
def errors_per_debug : ℕ := 3

-- Question: How many errors has she fixed so far?
theorem fixed_errors_correct :
  (total_lines_of_code / lines_per_debug) * errors_per_debug = 129 := 
by 
  sorry

end fixed_errors_correct_l515_51561


namespace landscape_length_l515_51515

theorem landscape_length (b length : ℕ) (A_playground : ℕ) (h1 : length = 4 * b) (h2 : A_playground = 1200) (h3 : A_playground = (1 / 3 : ℚ) * (length * b)) :
  length = 120 :=
by
  sorry

end landscape_length_l515_51515


namespace range_of_a_l515_51569

open Real

theorem range_of_a (a : ℝ) (H : ∀ b : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → abs (x^2 + a * x + b) ≥ 1)) : a ≥ 1 ∨ a ≤ -3 :=
sorry

end range_of_a_l515_51569


namespace necessary_but_not_sufficient_condition_l515_51566

theorem necessary_but_not_sufficient_condition (a : ℝ)
    (h : -2 ≤ a ∧ a ≤ 2)
    (hq : ∃ x y : ℂ, x ≠ y ∧ (x ^ 2 + (a : ℂ) * x + 1 = 0) ∧ (y ^ 2 + (a : ℂ) * y + 1 = 0)) :
    ∃ z : ℂ, z ^ 2 + (a : ℂ) * z + 1 = 0 ∧ (¬ ∀ b, -2 < b ∧ b < 2 → b = a) :=
sorry

end necessary_but_not_sufficient_condition_l515_51566


namespace solve_for_x_l515_51596

theorem solve_for_x (x y : ℚ) (h1 : 3 * x - y = 7) (h2 : x + 3 * y = 2) : x = 23 / 10 :=
by
  -- Proof is omitted
  sorry

end solve_for_x_l515_51596


namespace robbie_weekly_fat_intake_l515_51597

theorem robbie_weekly_fat_intake
  (morning_cups : ℕ) (afternoon_cups : ℕ) (evening_cups : ℕ)
  (fat_per_cup : ℕ) (days_per_week : ℕ) :
  morning_cups = 3 →
  afternoon_cups = 2 →
  evening_cups = 5 →
  fat_per_cup = 10 →
  days_per_week = 7 →
  (morning_cups * fat_per_cup + afternoon_cups * fat_per_cup + evening_cups * fat_per_cup) * days_per_week = 700 :=
by
  intros
  sorry

end robbie_weekly_fat_intake_l515_51597


namespace compare_groups_l515_51562

noncomputable def mean (scores : List ℝ) : ℝ :=
  (scores.sum / scores.length)

noncomputable def variance (scores : List ℝ) : ℝ :=
  let m := mean scores
  (scores.map (λ x => (x - m) ^ 2)).sum / scores.length

noncomputable def stddev (scores : List ℝ) : ℝ :=
  (variance scores).sqrt

def groupA_scores : List ℝ := [88, 100, 95, 86, 95, 91, 84, 74, 92, 83]
def groupB_scores : List ℝ := [93, 89, 81, 77, 96, 78, 77, 85, 89, 86]

theorem compare_groups :
  mean groupA_scores > mean groupB_scores ∧ stddev groupA_scores > stddev groupB_scores :=
by
  sorry

end compare_groups_l515_51562


namespace complete_square_eq_l515_51509

theorem complete_square_eq (x : ℝ) : x^2 - 2 * x - 5 = 0 → (x - 1)^2 = 6 :=
by
  intro h
  have : x^2 - 2 * x = 5 := by linarith
  have : x^2 - 2 * x + 1 = 6 := by linarith
  exact eq_of_sub_eq_zero (by linarith)

end complete_square_eq_l515_51509


namespace total_cost_is_correct_l515_51517

def gravel_cost_per_cubic_foot : ℝ := 8
def discount_rate : ℝ := 0.10
def volume_in_cubic_yards : ℝ := 8
def conversion_factor : ℝ := 27

-- The initial cost for the given volume of gravel in cubic feet
noncomputable def initial_cost : ℝ := gravel_cost_per_cubic_foot * (volume_in_cubic_yards * conversion_factor)

-- The discount amount
noncomputable def discount_amount : ℝ := initial_cost * discount_rate

-- Total cost after applying discount
noncomputable def total_cost_after_discount : ℝ := initial_cost - discount_amount

theorem total_cost_is_correct : total_cost_after_discount = 1555.20 :=
sorry

end total_cost_is_correct_l515_51517


namespace count_four_digit_multiples_of_5_l515_51516

theorem count_four_digit_multiples_of_5 : 
  let first_4_digit := 1000
  let last_4_digit := 9999
  let first_multiple_of_5 := 1000
  let last_multiple_of_5 := 9995
  let total_multiples_of_5 := (1999 - 200 + 1)
  first_multiple_of_5 % 5 = 0 ∧ last_multiple_of_5 % 5 = 0 ∧ first_4_digit ≤ first_multiple_of_5 ∧ last_multiple_of_5 ≤ last_4_digit
  → total_multiples_of_5 = 1800 :=
by
  sorry

end count_four_digit_multiples_of_5_l515_51516


namespace arithmetic_seq_problem_l515_51585

theorem arithmetic_seq_problem
  (a : ℕ → ℤ)  -- sequence a_n is an arithmetic sequence
  (h0 : ∃ (a1 d : ℤ), ∀ (n : ℕ), a n = a1 + n * d)  -- exists a1 and d such that a_n = a1 + n * d
  (h1 : a 0 + 3 * a 7 + a 14 = 120) :                -- given a1 + 3a8 + a15 = 120
  3 * a 8 - a 10 = 48 :=                             -- prove 3a9 - a11 = 48
sorry

end arithmetic_seq_problem_l515_51585


namespace triangle_perimeter_l515_51527

-- Define the given sides of the triangle
def side_a := 15
def side_b := 6
def side_c := 12

-- Define the function to calculate the perimeter of the triangle
def perimeter (a b c : ℕ) : ℕ :=
  a + b + c

-- The theorem stating that the perimeter of the given triangle is 33
theorem triangle_perimeter : perimeter side_a side_b side_c = 33 := by
  -- We can include the proof later
  sorry

end triangle_perimeter_l515_51527


namespace fraction_of_left_handed_non_throwers_l515_51537

theorem fraction_of_left_handed_non_throwers 
  (total_players : ℕ) (throwers : ℕ) (right_handed_players : ℕ) (all_throwers_right_handed : throwers ≤ right_handed_players) 
  (total_players_eq : total_players = 70) 
  (throwers_eq : throwers = 46) 
  (right_handed_players_eq : right_handed_players = 62) 
  : (total_players - throwers) = 24 → ((right_handed_players - throwers) = 16 → (24 - 16) = 8 → ((8 : ℚ) / 24 = 1/3)) := 
by 
  intros;
  sorry

end fraction_of_left_handed_non_throwers_l515_51537


namespace number_of_violinists_l515_51574

open Nat

/-- There are 3 violinists in the orchestra, based on given conditions. -/
theorem number_of_violinists
  (total : ℕ)
  (percussion : ℕ)
  (brass : ℕ)
  (cellist : ℕ)
  (contrabassist : ℕ)
  (woodwinds : ℕ)
  (maestro : ℕ)
  (total_eq : total = 21)
  (percussion_eq : percussion = 1)
  (brass_eq : brass = 7)
  (strings_excluding_violinists : ℕ)
  (cellist_eq : cellist = 1)
  (contrabassist_eq : contrabassist = 1)
  (woodwinds_eq : woodwinds = 7)
  (maestro_eq : maestro = 1) :
  (total - (percussion + brass + (cellist + contrabassist) + woodwinds + maestro)) = 3 := 
by
  sorry

end number_of_violinists_l515_51574


namespace range_of_m_l515_51587

open Real

theorem range_of_m (m : ℝ) : (m^2 > 2 + m ∧ 2 + m > 0) ↔ (m > 2 ∨ -2 < m ∧ m < -1) :=
by
  sorry

end range_of_m_l515_51587


namespace min_value_of_f_l515_51592

noncomputable def f (x : ℝ) : ℝ := (1 / x) + (9 / (1 - x))

theorem min_value_of_f (x : ℝ) (h1 : 0 < x) (h2 : x < 1) : f x = 16 :=
by
  sorry

end min_value_of_f_l515_51592


namespace sum_of_remaining_two_scores_l515_51550

open Nat

theorem sum_of_remaining_two_scores :
  ∃ x y : ℕ, x + y = 160 ∧ (65 + 75 + 85 + 95 + x + y) / 6 = 80 :=
by
  sorry

end sum_of_remaining_two_scores_l515_51550


namespace angle_between_hour_and_minute_hand_at_5_oclock_l515_51538

theorem angle_between_hour_and_minute_hand_at_5_oclock : 
  let degrees_in_circle := 360
  let hours_in_clock := 12
  let angle_per_hour := degrees_in_circle / hours_in_clock
  let hour_hand_position := 5
  let minute_hand_position := 0
  let angle := (hour_hand_position - minute_hand_position) * angle_per_hour
  angle = 150 :=
by sorry

end angle_between_hour_and_minute_hand_at_5_oclock_l515_51538


namespace gcd_equation_solutions_l515_51540

theorem gcd_equation_solutions:
  ∀ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y^2 + Nat.gcd x y ^ 3 = x * y * Nat.gcd x y 
  → (x = 4 ∧ y = 2) ∨ (x = 4 ∧ y = 6) ∨ (x = 5 ∧ y = 2) ∨ (x = 5 ∧ y = 3) := 
by
  intros x y h
  sorry

end gcd_equation_solutions_l515_51540


namespace geometric_sequence_form_l515_51518

-- Definitions for sequences and common difference/ratio
def isArithmeticSeq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ (m n : ℕ), a n = a m + (n - m) * d

def isGeometricSeq (b : ℕ → ℝ) (q : ℝ) :=
  ∀ (m n : ℕ), b n = b m * q ^ (n - m)

-- Problem statement: given an arithmetic sequence, find the form of the corresponding geometric sequence
theorem geometric_sequence_form
  (b : ℕ → ℝ) (q : ℝ) (m n : ℕ) (b_m : ℝ) (q_pos : q > 0) :
  (∀ (m n : ℕ), b n = b m * q ^ (n - m)) :=
sorry

end geometric_sequence_form_l515_51518


namespace find_x_y_l515_51589

theorem find_x_y (x y : ℝ) : 
  (x - 12) ^ 2 + (y - 13) ^ 2 + (x - y) ^ 2 = 1 / 3 ↔ (x = 37 / 3 ∧ y = 38 / 3) :=
by
  sorry

end find_x_y_l515_51589


namespace ganpat_paint_time_l515_51577

theorem ganpat_paint_time (H_rate G_rate : ℝ) (together_time H_time : ℝ) (h₁ : H_time = 3)
  (h₂ : together_time = 2) (h₃ : H_rate = 1 / H_time) (h₄ : G_rate = 1 / G_time)
  (h₅ : 1/H_time + 1/G_rate = 1/together_time) : G_time = 3 := 
by 
  sorry

end ganpat_paint_time_l515_51577


namespace chemist_sons_ages_l515_51572

theorem chemist_sons_ages 
    (a b c w : ℕ)
    (h1 : a * b * c = 36)
    (h2 : a + b + c = w)
    (h3 : ∃! x, x = max a (max b c)) :
    (a = 2 ∧ b = 2 ∧ c = 9) ∨ 
    (a = 2 ∧ b = 9 ∧ c = 2) ∨ 
    (a = 9 ∧ b = 2 ∧ c = 2) :=
  sorry

end chemist_sons_ages_l515_51572


namespace taxes_taken_out_l515_51521

theorem taxes_taken_out
  (gross_pay : ℕ)
  (retirement_percentage : ℝ)
  (net_pay_after_taxes : ℕ)
  (tax_amount : ℕ) :
  gross_pay = 1120 →
  retirement_percentage = 0.25 →
  net_pay_after_taxes = 740 →
  tax_amount = gross_pay - (gross_pay * retirement_percentage) - net_pay_after_taxes :=
by
  sorry

end taxes_taken_out_l515_51521


namespace solve_trigonometric_equation_l515_51576

theorem solve_trigonometric_equation (x : ℝ) : 
  (2 * (Real.sin x)^6 + 2 * (Real.cos x)^6 - 3 * (Real.sin x)^4 - 3 * (Real.cos x)^4) = Real.cos (2 * x) ↔ 
  ∃ (k : ℤ), x = (π / 2) * (2 * k + 1) :=
sorry

end solve_trigonometric_equation_l515_51576


namespace figure_100_squares_l515_51512

def f (n : ℕ) : ℕ := n^3 + 2 * n^2 + 2 * n + 1

theorem figure_100_squares : f 100 = 1020201 :=
by
  -- The proof will go here
  sorry

end figure_100_squares_l515_51512


namespace men_in_hotel_l515_51513

theorem men_in_hotel (n : ℕ) (A : ℝ) (h1 : 8 * 3 = 24)
  (h2 : A = 32.625 / n)
  (h3 : 24 + (A + 5) = 32.625) :
  n = 9 := 
  by
  sorry

end men_in_hotel_l515_51513


namespace prime_has_two_square_numbers_l515_51541

noncomputable def isSquareNumber (p q : ℕ) : Prop :=
  p > q ∧ Nat.Prime p ∧ Nat.Prime q ∧ ¬ p^2 ∣ (q^(p-1) - 1)

theorem prime_has_two_square_numbers (p : ℕ) (hp : Nat.Prime p) (h5 : p ≥ 5) :
  ∃ q1 q2 : ℕ, isSquareNumber p q1 ∧ isSquareNumber p q2 ∧ q1 ≠ q2 :=
by 
  sorry

end prime_has_two_square_numbers_l515_51541


namespace ratio_of_sums_l515_51508

theorem ratio_of_sums (a b c : ℚ) (h1 : b / a = 2) (h2 : c / b = 3) : (a + b) / (b + c) = 3 / 8 := 
  sorry

end ratio_of_sums_l515_51508


namespace players_odd_sum_probability_l515_51539

theorem players_odd_sum_probability :
  let tiles := (1:ℕ) :: (2:ℕ) :: (3:ℕ) :: (4:ℕ) :: (5:ℕ) :: (6:ℕ) :: (7:ℕ) :: (8:ℕ) :: (9:ℕ) :: (10:ℕ) :: (11:ℕ) :: []
  let m := 1
  let n := 26
  m + n = 27 :=
by
  sorry

end players_odd_sum_probability_l515_51539


namespace night_shift_hours_l515_51594

theorem night_shift_hours
  (hours_first_guard : ℕ := 3)
  (hours_last_guard : ℕ := 2)
  (hours_each_middle_guard : ℕ := 2) :
  hours_first_guard + 2 * hours_each_middle_guard + hours_last_guard = 9 :=
by 
  sorry

end night_shift_hours_l515_51594


namespace smallest_n_for_cube_root_form_l515_51568

theorem smallest_n_for_cube_root_form
  (m n : ℕ) (r : ℝ)
  (h_pos_n : n > 0)
  (h_pos_r : r > 0)
  (h_r_bound : r < 1/500)
  (h_m : m = (n + r)^3)
  (h_min_m : ∀ k : ℕ, k = (n + r)^3 → k ≥ m) :
  n = 13 :=
by
  -- proof goes here
  sorry

end smallest_n_for_cube_root_form_l515_51568


namespace map_length_to_reality_l515_51584

def scale : ℝ := 500
def length_map : ℝ := 7.2
def length_actual : ℝ := 3600

theorem map_length_to_reality : length_actual = length_map * scale :=
by
  sorry

end map_length_to_reality_l515_51584


namespace count_natural_numbers_perfect_square_l515_51526

theorem count_natural_numbers_perfect_square :
  ∃ n1 n2 : ℕ, n1 ≠ n2 ∧ (n1^2 - 19 * n1 + 91) = m^2 ∧ (n2^2 - 19 * n2 + 91) = k^2 ∧
  ∀ n : ℕ, (n^2 - 19 * n + 91) = p^2 → n = n1 ∨ n = n2 := sorry

end count_natural_numbers_perfect_square_l515_51526


namespace weighted_average_plants_per_hour_l515_51545

theorem weighted_average_plants_per_hour :
  let heath_carrot_plants_100 := 100 * 275
  let heath_carrot_plants_150 := 150 * 325
  let heath_total_plants := heath_carrot_plants_100 + heath_carrot_plants_150
  let heath_total_time := 10 + 20
  
  let jake_potato_plants_50 := 50 * 300
  let jake_potato_plants_100 := 100 * 400
  let jake_total_plants := jake_potato_plants_50 + jake_potato_plants_100
  let jake_total_time := 12 + 18

  let total_plants := heath_total_plants + jake_total_plants
  let total_time := heath_total_time + jake_total_time
  let weighted_average := total_plants / total_time
  weighted_average = 2187.5 :=
by
  sorry

end weighted_average_plants_per_hour_l515_51545


namespace quadratic_term_free_polynomial_l515_51563

theorem quadratic_term_free_polynomial (m : ℤ) (h : 36 + 12 * m = 0) : m^3 = -27 := by
  -- Proof goes here
  sorry

end quadratic_term_free_polynomial_l515_51563


namespace stable_performance_l515_51571

/-- The variance of student A's scores is 0.4 --/
def variance_A : ℝ := 0.4

/-- The variance of student B's scores is 0.3 --/
def variance_B : ℝ := 0.3

/-- Prove that student B has more stable performance given the variances --/
theorem stable_performance (h1 : variance_A = 0.4) (h2 : variance_B = 0.3) : variance_B < variance_A :=
by
  rw [h1, h2]
  exact sorry

end stable_performance_l515_51571


namespace milk_production_l515_51500

theorem milk_production (a b c d e f : ℕ) (h₁ : a > 0) (h₂ : c > 0) (h₃ : f > 0) : 
  ((d * e * b * f) / (100 * a * c)) = (d * e * b * f / (100 * a * c)) :=
by
  sorry

end milk_production_l515_51500


namespace ratio_of_fractions_l515_51536

theorem ratio_of_fractions (x y : ℝ) (h1 : 5 * x = 3 * y) (h2 : x * y ≠ 0) : 
  (1 / 5 * x) / (1 / 6 * y) = 0.72 :=
sorry

end ratio_of_fractions_l515_51536


namespace total_students_l515_51524

theorem total_students (rank_right rank_left : ℕ) (h1 : rank_right = 16) (h2 : rank_left = 6) : rank_right + rank_left - 1 = 21 := by
  sorry

end total_students_l515_51524


namespace train_speed_kmh_l515_51514

variable (length_of_train_meters : ℕ) (time_to_cross_seconds : ℕ)

theorem train_speed_kmh (h1 : length_of_train_meters = 50) (h2 : time_to_cross_seconds = 6) :
  (length_of_train_meters * 3600) / (time_to_cross_seconds * 1000) = 30 :=
by
  sorry

end train_speed_kmh_l515_51514


namespace at_least_one_angle_not_greater_than_60_l515_51502

theorem at_least_one_angle_not_greater_than_60 (A B C : ℝ) (hA : A > 60) (hB : B > 60) (hC : C > 60) (hSum : A + B + C = 180) : false :=
by
  sorry

end at_least_one_angle_not_greater_than_60_l515_51502


namespace time_to_wash_car_l515_51582

theorem time_to_wash_car (W : ℕ) 
    (t_oil : ℕ := 15) 
    (t_tires : ℕ := 30) 
    (n_wash : ℕ := 9) 
    (n_oil : ℕ := 6) 
    (n_tires : ℕ := 2) 
    (total_time : ℕ := 240) 
    (h : n_wash * W + n_oil * t_oil + n_tires * t_tires = total_time) 
    : W = 10 := by
  sorry

end time_to_wash_car_l515_51582
