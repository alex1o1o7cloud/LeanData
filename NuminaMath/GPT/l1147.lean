import Mathlib

namespace find_certain_number_l1147_114743

theorem find_certain_number 
  (num : ℝ)
  (h1 : num / 14.5 = 177)
  (h2 : 29.94 / 1.45 = 17.7) : 
  num = 2566.5 := 
by 
  sorry

end find_certain_number_l1147_114743


namespace factorize_expression_l1147_114795

theorem factorize_expression (a : ℝ) : a^3 - 4 * a^2 + 4 * a = a * (a - 2)^2 := 
by
  sorry

end factorize_expression_l1147_114795


namespace point_slope_form_l1147_114707

theorem point_slope_form (k : ℝ) (p : ℝ × ℝ) (h_slope : k = 2) (h_point : p = (2, -3)) :
  (∃ l : ℝ → ℝ, ∀ x y : ℝ, y = l x ↔ y = 2 * (x - 2) + (-3)) := 
sorry

end point_slope_form_l1147_114707


namespace haleigh_cats_l1147_114778

open Nat

def total_pairs := 14
def dog_leggings := 4
def legging_per_animal := 1

theorem haleigh_cats : ∀ (dogs cats : ℕ), 
  dogs = 4 → 
  total_pairs = dogs * legging_per_animal + cats * legging_per_animal → 
  cats = 10 :=
by
  intros dogs cats h1 h2
  sorry

end haleigh_cats_l1147_114778


namespace max_value_of_sum_l1147_114781

theorem max_value_of_sum 
  (a b c : ℝ) 
  (h : a^2 + 2 * b^2 + 3 * c^2 = 6) : 
  a + b + c ≤ Real.sqrt 11 := 
by 
  sorry

end max_value_of_sum_l1147_114781


namespace x_expression_l1147_114754

noncomputable def f (t : ℝ) : ℝ := t / (1 - t)

theorem x_expression {x y : ℝ} (hx : x ≠ 1) (hy : y = f x) : x = y / (1 + y) :=
by {
  sorry
}

end x_expression_l1147_114754


namespace polygon_side_possibilities_l1147_114752

theorem polygon_side_possibilities (n : ℕ) (h : (n-2) * 180 = 1620) :
  n = 10 ∨ n = 11 ∨ n = 12 :=
by
  sorry

end polygon_side_possibilities_l1147_114752


namespace sets_relationship_l1147_114769

variables {U : Type*} (A B C : Set U)

theorem sets_relationship (h1 : A ∩ B = C) (h2 : B ∩ C = A) : A = C ∧ ∃ B, A ⊆ B := by
  sorry

end sets_relationship_l1147_114769


namespace period_and_monotonic_interval_range_of_f_l1147_114704

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 3) * Real.cos (2 * x) + Real.sin (x + Real.pi / 4) ^ 2

theorem period_and_monotonic_interval :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ 
  (∃ k : ℤ, ∀ x, x ∈ Set.Icc (k * Real.pi - 5 * Real.pi / 12) (k * Real.pi + Real.pi / 12) →
    MonotoneOn f (Set.Icc (2 * k * Real.pi - Real.pi / 2) (2 * k * Real.pi + Real.pi / 2))) :=
sorry

theorem range_of_f (x : ℝ) (hx : x ∈ Set.Icc (-Real.pi / 12) (5 * Real.pi / 12)) :
  f x ∈ Set.Icc 0 (3 / 2) :=
sorry

end period_and_monotonic_interval_range_of_f_l1147_114704


namespace solution_to_inequality_l1147_114794

theorem solution_to_inequality : 
  ∀ x : ℝ, (x + 3) * (x - 1) < 0 ↔ -3 < x ∧ x < 1 :=
by
  intro x
  sorry

end solution_to_inequality_l1147_114794


namespace number_of_zeros_in_factorial_30_l1147_114729

theorem number_of_zeros_in_factorial_30 :
  let count_factors (n k : Nat) : Nat := n / k
  count_factors 30 5 + count_factors 30 25 = 7 :=
by
  let count_factors (n k : Nat) : Nat := n / k
  sorry

end number_of_zeros_in_factorial_30_l1147_114729


namespace ratio_Sydney_to_Sherry_l1147_114784

variable (Randolph_age Sydney_age Sherry_age : ℕ)

-- Conditions
axiom Randolph_older_than_Sydney : Randolph_age = Sydney_age + 5
axiom Sherry_age_is_25 : Sherry_age = 25
axiom Randolph_age_is_55 : Randolph_age = 55

-- Theorem to prove
theorem ratio_Sydney_to_Sherry : (Sydney_age : ℝ) / (Sherry_age : ℝ) = 2 := by
  sorry

end ratio_Sydney_to_Sherry_l1147_114784


namespace find_age_of_b_l1147_114720

variables (A B C : ℕ)

def average_abc (A B C : ℕ) : Prop := (A + B + C) / 3 = 28
def average_ac (A C : ℕ) : Prop := (A + C) / 2 = 29

theorem find_age_of_b (h1 : average_abc A B C) (h2 : average_ac A C) : B = 26 :=
by
  sorry

end find_age_of_b_l1147_114720


namespace find_solutions_of_equation_l1147_114728

theorem find_solutions_of_equation (m n : ℝ) 
  (h1 : ∀ x, (x - m)^2 + n = 0 ↔ (x = -1 ∨ x = 3)) :
  (∀ x, (x - 1)^2 + m^2 = 2 * m * (x - 1) - n ↔ (x = 0 ∨ x = 4)) :=
by
  sorry

end find_solutions_of_equation_l1147_114728


namespace loan_difference_l1147_114762

noncomputable def future_value (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def monthly_compounding : ℝ :=
  future_value 8000 0.10 12 5

noncomputable def semi_annual_compounding : ℝ :=
  future_value 8000 0.10 2 5

noncomputable def interest_difference : ℝ :=
  monthly_compounding - semi_annual_compounding

theorem loan_difference (P : ℝ) (r : ℝ) (n_m n_s t : ℝ) :
    interest_difference = 745.02 := by sorry

end loan_difference_l1147_114762


namespace find_c_l1147_114734

theorem find_c (b c : ℝ) (h : (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c * x + 12)) : 
  c = 7 := 
by {
  sorry
}

end find_c_l1147_114734


namespace total_amount_spent_l1147_114790

variable (you friend : ℝ)

theorem total_amount_spent (h1 : friend = you + 3) (h2 : friend = 7) : 
  you + friend = 11 :=
by
  sorry

end total_amount_spent_l1147_114790


namespace trip_early_movie_savings_l1147_114764

theorem trip_early_movie_savings : 
  let evening_ticket_cost : ℝ := 10
  let food_combo_cost : ℝ := 10
  let ticket_discount : ℝ := 0.20
  let food_discount : ℝ := 0.50
  let evening_total_cost := evening_ticket_cost + food_combo_cost
  let savings_on_ticket := evening_ticket_cost * ticket_discount
  let savings_on_food := food_combo_cost * food_discount
  let total_savings := savings_on_ticket + savings_on_food
  total_savings = 7 :=
by
  sorry

end trip_early_movie_savings_l1147_114764


namespace surface_area_of_reassembled_solid_l1147_114710

noncomputable def total_surface_area : ℕ :=
let height_E := 1/4
let height_F := 1/6
let height_G := 1/9 
let height_H := 1 - (height_E + height_F + height_G)
let face_area := 2 * 1
(face_area * 2)     -- Top and bottom surfaces
+ 2                -- Side surfaces (1 foot each side * 2 sides)
+ (face_area * 2)   -- Front and back surfaces 

theorem surface_area_of_reassembled_solid :
  total_surface_area = 10 :=
by
  sorry

end surface_area_of_reassembled_solid_l1147_114710


namespace sandwich_cost_is_5_l1147_114789

-- We define the variables and conditions first
def total_people := 4
def sandwiches := 4
def fruit_salads := 4
def sodas := 8
def snack_bags := 3

def fruit_salad_cost_per_unit := 3
def soda_cost_per_unit := 2
def snack_bag_cost_per_unit := 4
def total_cost := 60

-- We now define the calculations based on the given conditions
def total_fruit_salad_cost := fruit_salads * fruit_salad_cost_per_unit
def total_soda_cost := sodas * soda_cost_per_unit
def total_snack_bag_cost := snack_bags * snack_bag_cost_per_unit
def other_items_cost := total_fruit_salad_cost + total_soda_cost + total_snack_bag_cost
def remaining_budget := total_cost - other_items_cost
def sandwich_cost := remaining_budget / sandwiches

-- The final proof problem statement in Lean 4
theorem sandwich_cost_is_5 : sandwich_cost = 5 := by
  sorry

end sandwich_cost_is_5_l1147_114789


namespace P_at_3_l1147_114751

noncomputable def P (x : ℝ) : ℝ := 1 * x^5 + 0 * x^4 + 0 * x^3 + 2 * x^2 + 1 * x + 4

theorem P_at_3 : P 3 = 268 := by
  sorry

end P_at_3_l1147_114751


namespace arithmetic_sequence_example_l1147_114772

theorem arithmetic_sequence_example (a : ℕ → ℝ) (h : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) (h₁ : a 1 + a 19 = 10) : a 10 = 5 :=
by
  sorry

end arithmetic_sequence_example_l1147_114772


namespace value_of_a3_l1147_114792

theorem value_of_a3 (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) (a : ℝ) (h₀ : (1 + x) * (a - x)^6 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7) 
(h₁ : a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0) : 
a₃ = -5 :=
sorry

end value_of_a3_l1147_114792


namespace expand_expression_l1147_114747

variable (x : ℝ)

theorem expand_expression : (9 * x + 4) * (2 * x ^ 2) = 18 * x ^ 3 + 8 * x ^ 2 :=
by sorry

end expand_expression_l1147_114747


namespace union_is_equivalent_l1147_114718

def A (x : ℝ) : Prop := x ^ 2 - x - 6 ≤ 0
def B (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem union_is_equivalent (x : ℝ) :
  (A x ∨ B x) ↔ (-2 ≤ x ∧ x < 4) :=
sorry

end union_is_equivalent_l1147_114718


namespace original_average_rent_l1147_114799

theorem original_average_rent
    (A : ℝ) -- original average rent per person
    (h1 : 4 * A + 200 = 3400) -- condition derived from the rent problem
    : A = 800 := 
sorry

end original_average_rent_l1147_114799


namespace find_m_l1147_114732

theorem find_m (x m : ℝ) (h1 : 4 * x + 2 * m = 5 * x + 1) (h2 : 3 * x = 6 * x - 1) : m = 2 / 3 :=
by
  sorry

end find_m_l1147_114732


namespace infinitely_many_triples_l1147_114775

theorem infinitely_many_triples (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : ∀ k : ℕ, 
  ∃ (x y z : ℕ), 
    x = 2^(k * m * n + 1) ∧ 
    y = 2^(n + n * k * (m * n + 1)) ∧ 
    z = 2^(m + m * k * (m * n + 1)) ∧ 
    x^(m * n + 1) = y^m + z^n := 
by 
  intros k
  use 2^(k * m * n + 1), 2^(n + n * k * (m * n + 1)), 2^(m + m * k * (m * n + 1))
  simp
  sorry

end infinitely_many_triples_l1147_114775


namespace quadratic_function_properties_l1147_114750

def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 2

theorem quadratic_function_properties :
  (∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, f x ≤ f y) ∧
  (∃ x : ℝ, x = 1.5 ∧ ∀ y : ℝ, f x ≤ f y) :=
by
  sorry

end quadratic_function_properties_l1147_114750


namespace students_at_table_l1147_114770

def numStudents (candies : ℕ) (first_last : ℕ) (st_len : ℕ) : Prop :=
  candies - 1 = st_len * first_last

theorem students_at_table 
  (candies : ℕ)
  (first_last : ℕ)
  (st_len : ℕ)
  (h1 : candies = 120) 
  (h2 : first_last = 1) :
  (st_len = 7 ∨ st_len = 17) :=
by
  sorry

end students_at_table_l1147_114770


namespace inequality_implies_l1147_114733

theorem inequality_implies:
  ∀ (x y : ℝ), (x > y) → (2 * x - 1 > 2 * y - 1) :=
by
  intro x y hxy
  sorry

end inequality_implies_l1147_114733


namespace complex_number_corresponding_to_OB_l1147_114759

theorem complex_number_corresponding_to_OB :
  let OA : ℂ := 6 + 5 * Complex.I
  let AB : ℂ := 4 + 5 * Complex.I
  OB = OA + AB -> OB = 10 + 10 * Complex.I := by
  sorry

end complex_number_corresponding_to_OB_l1147_114759


namespace math_crackers_initial_l1147_114703

def crackers_initial (gave_each : ℕ) (left : ℕ) (num_friends : ℕ) : ℕ :=
  (gave_each * num_friends) + left

theorem math_crackers_initial :
  crackers_initial 7 17 3 = 38 :=
by
  -- The definition of crackers_initial and the theorem statement should be enough.
  -- The exact proof is left as a sorry placeholder.
  sorry

end math_crackers_initial_l1147_114703


namespace simplify_expression_l1147_114724

variable {a b c : ℤ}

theorem simplify_expression (a b c : ℤ) : 3 * a - (4 * a - 6 * b - 3 * c) - 5 * (c - b) = -a + 11 * b - 2 * c :=
by
  sorry

end simplify_expression_l1147_114724


namespace caffeine_in_cup_l1147_114765

-- Definitions based on the conditions
def caffeine_goal : ℕ := 200
def excess_caffeine : ℕ := 40
def total_cups : ℕ := 3

-- The statement proving that the amount of caffeine in a cup is 80 mg given the conditions.
theorem caffeine_in_cup : (3 * (80 : ℕ)) = (caffeine_goal + excess_caffeine) := by
  -- Plug in the value and simplify
  simp [caffeine_goal, excess_caffeine]

end caffeine_in_cup_l1147_114765


namespace miles_to_burger_restaurant_l1147_114785

-- Definitions and conditions
def miles_per_gallon : ℕ := 19
def gallons_of_gas : ℕ := 2
def miles_to_school : ℕ := 15
def miles_to_softball_park : ℕ := 6
def miles_to_friend_house : ℕ := 4
def miles_to_home : ℕ := 11
def total_gas_distance := miles_per_gallon * gallons_of_gas
def total_known_distances := miles_to_school + miles_to_softball_park + miles_to_friend_house + miles_to_home

-- Problem statement to prove
theorem miles_to_burger_restaurant :
  ∃ (miles_to_burger_restaurant : ℕ), 
  total_gas_distance = total_known_distances + miles_to_burger_restaurant ∧ miles_to_burger_restaurant = 2 := 
by
  sorry

end miles_to_burger_restaurant_l1147_114785


namespace sufficient_condition_l1147_114757

variable (a : ℝ)

theorem sufficient_condition (h : ∀ x : ℝ, -1 ≤ x → x ≤ 2 → x^2 - a ≥ 0) : a ≤ -1 := 
sorry

end sufficient_condition_l1147_114757


namespace second_sum_is_1704_l1147_114711

theorem second_sum_is_1704
    (total_sum : ℝ)
    (x : ℝ)
    (interest_rate_first_part : ℝ)
    (time_first_part : ℝ)
    (interest_rate_second_part : ℝ)
    (time_second_part : ℝ)
    (h1 : total_sum = 2769)
    (h2 : interest_rate_first_part = 3)
    (h3 : time_first_part = 8)
    (h4 : interest_rate_second_part = 5)
    (h5 : time_second_part = 3)
    (h6 : 24 * x / 100 = (total_sum - x) * 15 / 100) :
    total_sum - x = 1704 :=
  by
    sorry

end second_sum_is_1704_l1147_114711


namespace number_of_integer_solutions_l1147_114741

theorem number_of_integer_solutions :
  ∃ (n : ℕ), 
  (∀ (x y : ℤ), 2 * x + 3 * y = 7 ∧ 5 * x + n * y = n ^ 2) ∧
  (n = 8) := 
sorry

end number_of_integer_solutions_l1147_114741


namespace vector_collinearity_l1147_114748

variables (a b : ℝ × ℝ)

def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem vector_collinearity : collinear (-1, 2) (1, -2) :=
by
  sorry

end vector_collinearity_l1147_114748


namespace sector_central_angle_l1147_114776

theorem sector_central_angle (r l : ℝ) (α : ℝ) 
  (h1 : l + 2 * r = 12) 
  (h2 : 1 / 2 * l * r = 8) : 
  α = 1 ∨ α = 4 :=
by
  sorry

end sector_central_angle_l1147_114776


namespace abs_a_plus_2_always_positive_l1147_114722

theorem abs_a_plus_2_always_positive (a : ℝ) : |a| + 2 > 0 := 
sorry

end abs_a_plus_2_always_positive_l1147_114722


namespace students_enrolled_both_english_and_german_l1147_114756

def total_students : ℕ := 32
def enrolled_german : ℕ := 22
def only_english : ℕ := 10
def students_enrolled_at_least_one_subject := total_students

theorem students_enrolled_both_english_and_german :
  ∃ (e_g : ℕ), e_g = enrolled_german - only_english :=
by
  sorry

end students_enrolled_both_english_and_german_l1147_114756


namespace initial_avg_weight_proof_l1147_114763

open Classical

variable (A B C D E : ℝ) (W : ℝ)

-- Given conditions
def initial_avg_weight_A_B_C : Prop := W = (A + B + C) / 3
def avg_with_D : Prop := (A + B + C + D) / 4 = 80
def E_weighs_D_plus_8 : Prop := E = D + 8
def avg_with_E_replacing_A : Prop := (B + C + D + E) / 4 = 79
def weight_of_A : Prop := A = 80

-- Question to prove
theorem initial_avg_weight_proof (h1 : initial_avg_weight_A_B_C W A B C)
                                 (h2 : avg_with_D A B C D)
                                 (h3 : E_weighs_D_plus_8 D E)
                                 (h4 : avg_with_E_replacing_A B C D E)
                                 (h5 : weight_of_A A) :
  W = 84 := by
  sorry

end initial_avg_weight_proof_l1147_114763


namespace polar_equation_of_circle_c_range_of_op_oq_l1147_114726

noncomputable def circle_param_eq (φ : ℝ) : ℝ × ℝ :=
  (1 + Real.cos φ, Real.sin φ)

noncomputable def line_kl_eq (θ : ℝ) : ℝ :=
  3 * Real.sqrt 3 / (Real.sin θ + Real.sqrt 3 * Real.cos θ)

theorem polar_equation_of_circle_c :
  ∀ θ : ℝ, ∃ ρ : ℝ, ρ = 2 * Real.cos θ :=
by sorry

theorem range_of_op_oq (θ₁ : ℝ) (hθ : 0 < θ₁ ∧ θ₁ < Real.pi / 2) :
  0 < (2 * Real.cos θ₁) * (3 * Real.sqrt 3 / (Real.sin θ₁ + Real.sqrt 3 * Real.cos θ₁)) ∧
  (2 * Real.cos θ₁) * (3 * Real.sqrt 3 / (Real.sin θ₁ + Real.sqrt 3 * Real.cos θ₁)) < 6 :=
by sorry

end polar_equation_of_circle_c_range_of_op_oq_l1147_114726


namespace part1_f_inequality_part2_a_range_l1147_114786

open Real

-- Proof Problem 1
theorem part1_f_inequality (x : ℝ) : 
    (|x - 1| + |x + 1| ≥ 3 ↔ x ≤ -1.5 ∨ x ≥ 1.5) :=
sorry

-- Proof Problem 2
theorem part2_a_range (a : ℝ) : 
    (∀ x : ℝ, |x - 1| + |x - a| ≥ 2) ↔ (a = 3 ∨ a = -1) :=
sorry

end part1_f_inequality_part2_a_range_l1147_114786


namespace power_division_l1147_114713

theorem power_division (a b : ℕ) (h : 64 = 8^2) : (8 ^ 15) / (64 ^ 7) = 8 := by
  sorry

end power_division_l1147_114713


namespace price_decrease_percentage_l1147_114766

theorem price_decrease_percentage (original_price : ℝ) :
  let first_sale_price := (4/5) * original_price
  let second_sale_price := (1/2) * original_price
  let decrease := first_sale_price - second_sale_price
  let percentage_decrease := (decrease / first_sale_price) * 100
  percentage_decrease = 37.5 := by
  sorry

end price_decrease_percentage_l1147_114766


namespace tan_pi_add_alpha_eq_two_l1147_114705

theorem tan_pi_add_alpha_eq_two
  (α : ℝ)
  (h : Real.tan (Real.pi + α) = 2) :
  (2 * Real.sin α - Real.cos α) / (3 * Real.sin α + 2 * Real.cos α) = 3 / 8 :=
sorry

end tan_pi_add_alpha_eq_two_l1147_114705


namespace probability_at_least_one_defective_item_l1147_114774

def total_products : ℕ := 10
def defective_items : ℕ := 3
def selected_items : ℕ := 3
noncomputable def comb (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_least_one_defective_item :
    let total_combinations := comb total_products selected_items
    let non_defective_combinations := comb (total_products - defective_items) selected_items
    let opposite_probability := (non_defective_combinations : ℚ) / (total_combinations : ℚ)
    let probability := 1 - opposite_probability
    probability = 17 / 24 :=
by
  sorry

end probability_at_least_one_defective_item_l1147_114774


namespace time_2517_hours_from_now_l1147_114780

-- Define the initial time and the function to calculate time after certain hours on a 12-hour clock
def current_time := 3
def hours := 2517

noncomputable def final_time_mod_12 (current_time : ℕ) (hours : ℕ) : ℕ :=
  (current_time + (hours % 12)) % 12

theorem time_2517_hours_from_now :
  final_time_mod_12 current_time hours = 12 :=
by
  sorry

end time_2517_hours_from_now_l1147_114780


namespace number_of_people_needed_to_lift_car_l1147_114796

-- Define the conditions as Lean definitions
def twice_as_many_people_to_lift_truck (C T : ℕ) : Prop :=
  T = 2 * C

def people_needed_for_cars_and_trucks (C T total_people : ℕ) : Prop :=
  60 = 6 * C + 3 * T

-- Define the theorem statement using the conditions
theorem number_of_people_needed_to_lift_car :
  ∃ C, (∃ T, twice_as_many_people_to_lift_truck C T) ∧ people_needed_for_cars_and_trucks C T 60 ∧ C = 5 :=
sorry

end number_of_people_needed_to_lift_car_l1147_114796


namespace mower_next_tangent_point_l1147_114742

theorem mower_next_tangent_point (r_garden r_mower : ℝ) (h_garden : r_garden = 15) (h_mower : r_mower = 5) :
    ∃ θ : ℝ, θ = (2 * π * r_mower / (2 * π * r_garden)) * 360 ∧ θ = 120 :=
sorry

end mower_next_tangent_point_l1147_114742


namespace total_amount_received_l1147_114712

theorem total_amount_received
  (total_books : ℕ := 500)
  (novels_price : ℕ := 8)
  (biographies_price : ℕ := 12)
  (science_books_price : ℕ := 10)
  (novels_discount : ℚ := 0.25)
  (biographies_discount : ℚ := 0.30)
  (science_books_discount : ℚ := 0.20)
  (sales_tax : ℚ := 0.05)
  (remaining_novels : ℕ := 60)
  (remaining_biographies : ℕ := 65)
  (remaining_science_books : ℕ := 50)
  (novel_ratio_sold : ℚ := 3/5)
  (biography_ratio_sold : ℚ := 2/3)
  (science_book_ratio_sold : ℚ := 7/10)
  (original_novels : ℕ := 150)
  (original_biographies : ℕ := 195)
  (original_science_books : ℕ := 167) -- Rounded from 166.67
  (sold_novels : ℕ := 90)
  (sold_biographies : ℕ := 130)
  (sold_science_books : ℕ := 117)
  (total_revenue_before_discount : ℚ := (90 * 8 + 130 * 12 + 117 * 10))
  (total_revenue_after_discount : ℚ := (720 * (1 - 0.25) + 1560 * (1 - 0.30) + 1170 * (1 - 0.20)))
  (total_revenue_after_tax : ℚ := (2568 * 1.05)) :
  total_revenue_after_tax = 2696.4 :=
by
  sorry

end total_amount_received_l1147_114712


namespace digit_150_of_1_over_13_is_3_l1147_114777

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l1147_114777


namespace sequence_equality_l1147_114725

noncomputable def a (x : ℝ) (n : ℕ) : ℝ := 1 + x^(n+1) + x^(n+2)

theorem sequence_equality (x : ℝ) (hx : x = 0 ∨ x = 1 ∨ x = -1) (n : ℕ) (hn : n ≥ 3) :
  (a x n)^2 = (a x (n-1)) * (a x (n+1)) :=
by sorry

end sequence_equality_l1147_114725


namespace total_pies_sold_l1147_114788

def shepherds_pie_slices_per_pie : Nat := 4
def chicken_pot_pie_slices_per_pie : Nat := 5
def shepherds_pie_slices_ordered : Nat := 52
def chicken_pot_pie_slices_ordered : Nat := 80

theorem total_pies_sold :
  shepherds_pie_slices_ordered / shepherds_pie_slices_per_pie +
  chicken_pot_pie_slices_ordered / chicken_pot_pie_slices_per_pie = 29 := by
sorry

end total_pies_sold_l1147_114788


namespace area_relation_l1147_114700

open Real

noncomputable def S_OMN (a b c d θ : ℝ) : ℝ := 1 / 2 * abs (b * c - a * d) * sin θ
noncomputable def S_ABCD (a b c d θ : ℝ) : ℝ := 2 * abs (b * c - a * d) * sin θ

theorem area_relation (a b c d θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : 
    4 * (S_OMN a b c d θ) = S_ABCD a b c d θ :=
by
  sorry

end area_relation_l1147_114700


namespace problem1_problem2_l1147_114721

theorem problem1 : 3 / Real.sqrt 3 + (Real.pi + Real.sqrt 3)^0 + abs (Real.sqrt 3 - 2) = 3 := 
by
  sorry

theorem problem2 : (3 * Real.sqrt 12 - 2 * Real.sqrt (1 / 3) + Real.sqrt 48) / Real.sqrt 3 = 28 / 3 :=
by
  sorry

end problem1_problem2_l1147_114721


namespace binomial_coefficient_times_two_l1147_114779

theorem binomial_coefficient_times_two : 2 * Nat.choose 8 5 = 112 := 
by 
  -- The proof is omitted here
  sorry

end binomial_coefficient_times_two_l1147_114779


namespace ten_times_six_x_plus_fourteen_pi_l1147_114740

theorem ten_times_six_x_plus_fourteen_pi (x : ℝ) (Q : ℝ) (h : 5 * (3 * x + 7 * Real.pi) = Q) : 
  10 * (6 * x + 14 * Real.pi) = 4 * Q :=
by
  sorry

end ten_times_six_x_plus_fourteen_pi_l1147_114740


namespace solve_for_x_l1147_114782

theorem solve_for_x (x : ℝ) (h : (x^2 + 4*x - 5)^0 = 1) : x^2 - 5*x + 5 = 1 → x = 4 := 
by
  intro h2
  have : ∀ x, (x^2 + 4*x - 5 = 0) ↔ false := sorry
  exact sorry

end solve_for_x_l1147_114782


namespace numberOfColoringWays_l1147_114737

-- Define the problem parameters
def totalBalls : Nat := 5
def redBalls : Nat := 1
def blueBalls : Nat := 1
def yellowBalls : Nat := 2
def whiteBalls : Nat := 1

-- Show that the number of permutations of the multiset is 60
theorem numberOfColoringWays : (Nat.factorial totalBalls) / ((Nat.factorial redBalls) * (Nat.factorial blueBalls) * (Nat.factorial yellowBalls) * (Nat.factorial whiteBalls)) = 60 :=
  by
  simp [totalBalls, redBalls, blueBalls, yellowBalls, whiteBalls]
  sorry

end numberOfColoringWays_l1147_114737


namespace pears_value_equivalence_l1147_114715

-- Condition: $\frac{3}{4}$ of $16$ apples are worth $12$ pears
def apples_to_pears (a p : ℕ) : Prop :=
  (3 * 16 / 4 * a = 12 * p)

-- Question: How many pears (p) are equivalent in value to $\frac{2}{3}$ of $9$ apples?
def pears_equivalent_to_apples (p : ℕ) : Prop :=
  (2 * 9 / 3 * p = 6)

theorem pears_value_equivalence (p : ℕ) (a : ℕ) (h1 : apples_to_pears a p) (h2 : pears_equivalent_to_apples p) : 
  p = 6 :=
sorry

end pears_value_equivalence_l1147_114715


namespace interval_width_and_count_l1147_114738

def average_income_intervals := [3000, 4000, 5000, 6000, 7000]
def frequencies := [5, 9, 4, 2]

theorem interval_width_and_count:
  (average_income_intervals[1] - average_income_intervals[0] = 1000) ∧
  (frequencies.length = 4) :=
by
  sorry

end interval_width_and_count_l1147_114738


namespace find_table_height_l1147_114745

theorem find_table_height (b r g h : ℝ) (h1 : h + b - g = 111) (h2 : h + r - b = 80) (h3 : h + g - r = 82) : h = 91 := 
by
  sorry

end find_table_height_l1147_114745


namespace prove_sin_c_minus_b_eq_one_prove_cd_div_bc_eq_l1147_114793

-- Problem 1: Proof of sin(C - B) = 1 given the trigonometric identity
theorem prove_sin_c_minus_b_eq_one
  (A B C : ℝ)
  (h_trig_eq : (1 + Real.sin A) / Real.cos A = Real.sin (2 * B) / (1 - Real.cos (2 * B)))
  : Real.sin (C - B) = 1 := 
sorry

-- Problem 2: Proof of CD/BC given the ratios AB:AD:AC and the trigonometric identity
theorem prove_cd_div_bc_eq
  (A B C : ℝ)
  (AB AD AC BC CD : ℝ)
  (h_ratio : AB / AD = Real.sqrt 3 / Real.sqrt 2)
  (h_ratio_2 : AB / AC = Real.sqrt 3 / 1)
  (h_trig_eq : (1 + Real.sin A) / Real.cos A = Real.sin (2 * B) / (1 - Real.cos (2 * B)))
  (h_D_on_BC : True) -- Placeholder for D lies on BC condition
  : CD / BC = (Real.sqrt 5 - 1) / 2 := 
sorry

end prove_sin_c_minus_b_eq_one_prove_cd_div_bc_eq_l1147_114793


namespace arithmetic_sequence_sum_l1147_114767

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ (n : ℕ), a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a) 
  (h_a3 : a 3 = 4) (h_a101 : a 101 = 36) : 
  a 9 + a 52 + a 95 = 60 :=
sorry

end arithmetic_sequence_sum_l1147_114767


namespace x_range_condition_l1147_114761

-- Define the inequality and conditions
def inequality (x : ℝ) : Prop := x^2 + 2 * x < 8

-- The range of x must be (-4, 2)
theorem x_range_condition (x : ℝ) : inequality x → x > -4 ∧ x < 2 :=
by
  intro h
  sorry

end x_range_condition_l1147_114761


namespace incorrect_statement_maximum_value_l1147_114709

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem incorrect_statement_maximum_value :
  ∃ (a b c : ℝ), 
    (quadratic_function a b c 1 = -40) ∧
    (quadratic_function a b c (-1) = -8) ∧
    (quadratic_function a b c (-3) = 8) ∧
    (∀ (x_max : ℝ), (x_max = -b / (2 * a)) →
      (quadratic_function a b c x_max = 10) ∧
      (quadratic_function a b c x_max ≠ 8)) :=
by
  sorry

end incorrect_statement_maximum_value_l1147_114709


namespace a_n_formula_S_n_formula_T_n_formula_l1147_114797

noncomputable def a_sequence (n : ℕ) : ℕ := 2 * n
noncomputable def S (n : ℕ) : ℕ := n * (n + 1)
noncomputable def b_sequence (n : ℕ) : ℕ := a_sequence (3 ^ n)
noncomputable def T (n : ℕ) : ℕ := 3^(n + 1) - 3

theorem a_n_formula :
  ∀ {n : ℕ}, a_sequence 5 = 10 ∧ S 15 = 240 → a_sequence n = 2 * n :=
sorry

theorem S_n_formula :
  ∀ {n : ℕ}, a_sequence 5 = 10 ∧ S 15 = 240 → S n = n * (n + 1) :=
sorry

theorem T_n_formula :
  ∀ {n : ℕ}, a_sequence 5 = 10 ∧ S 15 = 240 → T n = 3^(n + 1) - 3 :=
sorry

end a_n_formula_S_n_formula_T_n_formula_l1147_114797


namespace sum_remainders_l1147_114730

theorem sum_remainders (n : ℤ) (h : n % 20 = 14) : (n % 4) + (n % 5) = 6 :=
  by
  sorry

end sum_remainders_l1147_114730


namespace hannah_total_spending_l1147_114739

def sweatshirt_price : ℕ := 15
def sweatshirt_quantity : ℕ := 3
def t_shirt_price : ℕ := 10
def t_shirt_quantity : ℕ := 2
def socks_price : ℕ := 5
def socks_quantity : ℕ := 4
def jacket_price : ℕ := 50
def discount_rate : ℚ := 0.10

noncomputable def total_cost_before_discount : ℕ :=
  (sweatshirt_quantity * sweatshirt_price) +
  (t_shirt_quantity * t_shirt_price) +
  (socks_quantity * socks_price) +
  jacket_price

noncomputable def total_cost_after_discount : ℚ :=
  total_cost_before_discount - (discount_rate * total_cost_before_discount)

theorem hannah_total_spending : total_cost_after_discount = 121.50 := by
  sorry

end hannah_total_spending_l1147_114739


namespace range_of_a_l1147_114701

noncomputable def f (x : ℝ) : ℝ := x + 1 / Real.exp x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x > a * x) ↔ (1 - Real.exp 1) < a ∧ a ≤ 1 := 
by
  sorry

end range_of_a_l1147_114701


namespace total_dogs_l1147_114755

axiom brown_dogs : ℕ
axiom white_dogs : ℕ
axiom black_dogs : ℕ

theorem total_dogs (b w bl : ℕ) (h1 : b = 20) (h2 : w = 10) (h3 : bl = 15) : (b + w + bl) = 45 :=
by {
  sorry
}

end total_dogs_l1147_114755


namespace tree_planting_problem_l1147_114753

variables (n t : ℕ)

theorem tree_planting_problem (h1 : 4 * n = t + 11) (h2 : 2 * n = t - 13) : n = 12 ∧ t = 37 :=
by
  sorry

end tree_planting_problem_l1147_114753


namespace equivalent_proof_problem_l1147_114719

def option_A : ℚ := 14 / 10
def option_B : ℚ := 1 + 2 / 5
def option_C : ℚ := 1 + 6 / 15
def option_D : ℚ := 1 + 3 / 8
def option_E : ℚ := 1 + 28 / 20
def target : ℚ := 7 / 5

theorem equivalent_proof_problem : option_D ≠ target :=
by {
  sorry
}

end equivalent_proof_problem_l1147_114719


namespace john_pays_percentage_of_srp_l1147_114749

theorem john_pays_percentage_of_srp (P MP : ℝ) (h1 : P = 1.20 * MP) (h2 : MP > 0): 
  (0.60 * MP / P) * 100 = 50 :=
by
  sorry

end john_pays_percentage_of_srp_l1147_114749


namespace problem_statement_l1147_114760

variable (m : ℝ) -- We declare m as a real number

theorem problem_statement (h : m + 1/m = 10) : m^2 + 1/m^2 + 4 = 102 := 
by 
  sorry -- The proof is omitted

end problem_statement_l1147_114760


namespace possible_integer_radii_l1147_114744

theorem possible_integer_radii (r : ℕ) (h : r < 140) : 
  (3 * 2 * r * π = 2 * 140 * π) → ∃ rs : Finset ℕ, rs.card = 10 := by
  sorry

end possible_integer_radii_l1147_114744


namespace chairs_in_fifth_row_l1147_114783

theorem chairs_in_fifth_row : 
  ∀ (a : ℕ → ℕ), 
    a 1 = 14 ∧ 
    a 2 = 23 ∧ 
    a 3 = 32 ∧ 
    a 4 = 41 ∧ 
    a 6 = 59 ∧ 
    (∀ n, a (n + 1) = a n + 9) → 
  a 5 = 50 :=
by
  sorry

end chairs_in_fifth_row_l1147_114783


namespace kitchen_chairs_count_l1147_114727

-- Define the conditions
def total_chairs : ℕ := 9
def living_room_chairs : ℕ := 3

-- Prove the number of kitchen chairs
theorem kitchen_chairs_count : total_chairs - living_room_chairs = 6 := by
  -- Proof goes here
  sorry

end kitchen_chairs_count_l1147_114727


namespace sum_coordinates_B_l1147_114787

noncomputable def A : (ℝ × ℝ) := (0, 0)
noncomputable def B (x : ℝ) : (ℝ × ℝ) := (x, 4)

theorem sum_coordinates_B 
  (x : ℝ) 
  (h_slope : (4 - 0)/(x - 0) = 3/4) : x + 4 = 28 / 3 := by
sorry

end sum_coordinates_B_l1147_114787


namespace select_student_for_performance_and_stability_l1147_114706

def average_score_A : ℝ := 6.2
def average_score_B : ℝ := 6.0
def average_score_C : ℝ := 5.8
def average_score_D : ℝ := 6.2

def variance_A : ℝ := 0.32
def variance_B : ℝ := 0.58
def variance_C : ℝ := 0.12
def variance_D : ℝ := 0.25

theorem select_student_for_performance_and_stability :
  (average_score_A ≤ average_score_D ∧ variance_D < variance_A) →
  (average_score_B < average_score_A ∧ average_score_B < average_score_D) →
  (average_score_C < average_score_A ∧ average_score_C < average_score_D) →
  "D" = "D" :=
by
  intros h₁ h₂ h₃
  exact rfl

end select_student_for_performance_and_stability_l1147_114706


namespace median_is_70_74_l1147_114708

-- Define the histogram data as given
def histogram : List (ℕ × ℕ) :=
  [(85, 5), (80, 15), (75, 18), (70, 22), (65, 20), (60, 10), (55, 10)]

-- Function to calculate the cumulative sum at each interval
def cumulativeSum (hist : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
  hist.scanl (λ acc pair => (pair.1, acc.2 + pair.2)) (0, 0)

-- Function to find the interval where the median lies
def medianInterval (hist : List (ℕ × ℕ)) : ℕ :=
  let cumSum := cumulativeSum hist
  -- The median is the 50th and 51st scores
  let medianPos := 50
  -- Find the interval that contains the median position
  List.find? (λ pair => medianPos ≤ pair.2) cumSum |>.getD (0, 0) |>.1

-- The theorem stating that the median interval is 70-74
theorem median_is_70_74 : medianInterval histogram = 70 :=
  by sorry

end median_is_70_74_l1147_114708


namespace cos_135_eq_neg_inv_sqrt_2_l1147_114798

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l1147_114798


namespace height_in_meters_l1147_114716

theorem height_in_meters (h: 1 * 100 + 36 = 136) : 1.36 = 1 + 36 / 100 :=
by 
  -- proof steps will go here
  sorry

end height_in_meters_l1147_114716


namespace no_primes_in_range_l1147_114746

theorem no_primes_in_range (n : ℕ) (hn : n > 2) : 
  ∀ k, n! + 2 < k ∧ k < n! + n + 1 → ¬Prime k := 
sorry

end no_primes_in_range_l1147_114746


namespace total_rent_paid_l1147_114736

theorem total_rent_paid
  (weekly_rent : ℕ) (num_weeks : ℕ) 
  (hrent : weekly_rent = 388)
  (hweeks : num_weeks = 1359) :
  weekly_rent * num_weeks = 527292 := 
by
  sorry

end total_rent_paid_l1147_114736


namespace new_cylinder_height_percentage_l1147_114723

variables (r h h_new : ℝ)

theorem new_cylinder_height_percentage :
  (7 / 8) * π * r^2 * h = (3 / 5) * π * (1.25 * r)^2 * h_new →
  (h_new / h) = 14 / 15 :=
by
  intro h_volume_eq
  sorry

end new_cylinder_height_percentage_l1147_114723


namespace correct_avg_weight_of_class_l1147_114714

theorem correct_avg_weight_of_class :
  ∀ (n : ℕ) (avg_wt : ℝ) (mis_A mis_B mis_C actual_A actual_B actual_C : ℝ),
  n = 30 →
  avg_wt = 60.2 →
  mis_A = 54 → actual_A = 64 →
  mis_B = 58 → actual_B = 68 →
  mis_C = 50 → actual_C = 60 →
  (n * avg_wt + (actual_A - mis_A) + (actual_B - mis_B) + (actual_C - mis_C)) / n = 61.2 :=
by
  intros n avg_wt mis_A mis_B mis_C actual_A actual_B actual_C h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end correct_avg_weight_of_class_l1147_114714


namespace joint_probability_l1147_114758

noncomputable def P (A B : Prop) : ℝ := sorry
def A : Prop := sorry
def B : Prop := sorry

axiom prob_A : P A true = 0.005
axiom prob_B_given_A : P B true = 0.99

theorem joint_probability :
  P A B = 0.00495 :=
by sorry

end joint_probability_l1147_114758


namespace fraction_painted_red_l1147_114791

theorem fraction_painted_red :
  let matilda_section := (1:ℚ) / 2 -- Matilda's half section
  let ellie_section := (1:ℚ) / 2    -- Ellie's half section
  let matilda_painted := matilda_section / 2 -- Matilda's painted fraction
  let ellie_painted := ellie_section / 3    -- Ellie's painted fraction
  (matilda_painted + ellie_painted) = 5 / 12 := 
by
  sorry

end fraction_painted_red_l1147_114791


namespace train_cross_duration_l1147_114731

noncomputable def train_length : ℝ := 250
noncomputable def train_speed_kmph : ℝ := 162
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def time_to_cross_pole : ℝ := train_length / train_speed_mps

theorem train_cross_duration :
  time_to_cross_pole = 250 / (162 * (1000 / 3600)) :=
by
  -- The detailed proof is omitted as per instructions
  sorry

end train_cross_duration_l1147_114731


namespace jerry_current_average_l1147_114771

-- Definitions for Jerry's first 3 tests average and conditions
variable (A : ℝ)

-- Condition details
def total_score_of_first_3_tests := 3 * A
def new_desired_average := A + 2
def total_score_needed := (A + 2) * 4
def score_on_fourth_test := 93

theorem jerry_current_average :
  (total_score_needed A = total_score_of_first_3_tests A + score_on_fourth_test) → A = 85 :=
by
  sorry

end jerry_current_average_l1147_114771


namespace cost_of_white_washing_l1147_114735

-- Definitions for room dimensions, doors, windows, and cost per square foot
def length : ℕ := 25
def width : ℕ := 15
def height1 : ℕ := 12
def height2 : ℕ := 8
def door_height : ℕ := 6
def door_width : ℕ := 3
def window_height : ℕ := 4
def window_width : ℕ := 3
def cost_per_sq_ft : ℕ := 10
def ceiling_decoration_area : ℕ := 10

-- Definitions for the areas calculation
def area_walls_height1 : ℕ := 2 * (length * height1)
def area_walls_height2 : ℕ := 2 * (width * height2)
def total_wall_area : ℕ := area_walls_height1 + area_walls_height2

def area_one_door : ℕ := door_height * door_width
def total_doors_area : ℕ := 2 * area_one_door

def area_one_window : ℕ := window_height * window_width
def total_windows_area : ℕ := 3 * area_one_window

def adjusted_wall_area : ℕ := total_wall_area - total_doors_area - total_windows_area - ceiling_decoration_area

def total_cost : ℕ := adjusted_wall_area * cost_per_sq_ft

-- The theorem we want to prove
theorem cost_of_white_washing : total_cost = 7580 := by
  sorry

end cost_of_white_washing_l1147_114735


namespace sequence_term_sequence_sum_l1147_114702

def a_seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else 3^(n-1)

def S_n (n : ℕ) : ℕ :=
  (3^n - 1) / 2

theorem sequence_term (n : ℕ) (h : n ≥ 1) :
  a_seq n = 3^(n-1) :=
sorry

theorem sequence_sum (n : ℕ) :
  S_n n = (3^n - 1) / 2 :=
sorry

end sequence_term_sequence_sum_l1147_114702


namespace makeup_palette_cost_l1147_114773

variable (lipstick_cost : ℝ := 2.5)
variable (num_lipsticks : ℕ := 4)
variable (hair_color_cost : ℝ := 4)
variable (num_boxes_hair_color : ℕ := 3)
variable (total_cost : ℝ := 67)
variable (num_palettes : ℕ := 3)

theorem makeup_palette_cost :
  (total_cost - (num_lipsticks * lipstick_cost + num_boxes_hair_color * hair_color_cost)) / num_palettes = 15 := 
by
  sorry

end makeup_palette_cost_l1147_114773


namespace math_problem_l1147_114768

theorem math_problem :
  |(-3 : ℝ)| - Real.sqrt 8 - (1/2 : ℝ)⁻¹ + 2 * Real.cos (Real.pi / 4) = 1 - Real.sqrt 2 :=
by
  sorry

end math_problem_l1147_114768


namespace sin_x_eq_2ab_div_a2_plus_b2_l1147_114717

theorem sin_x_eq_2ab_div_a2_plus_b2
  (a b : ℝ) (x : ℝ)
  (h_tan : Real.tan x = 2 * a * b / (a^2 - b^2))
  (h_pos : 0 < b) (h_lt : b < a) (h_x : 0 < x ∧ x < Real.pi / 2) :
  Real.sin x = 2 * a * b / (a^2 + b^2) :=
by sorry

end sin_x_eq_2ab_div_a2_plus_b2_l1147_114717
