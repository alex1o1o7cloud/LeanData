import Mathlib

namespace angle_C_obtuse_l1761_176136

theorem angle_C_obtuse (a b c C : ℝ) (h1 : a^2 + b^2 < c^2) (h2 : Real.sin C = Real.sqrt 3 / 2) : C = 2 * Real.pi / 3 :=
by
  sorry

end angle_C_obtuse_l1761_176136


namespace best_scrap_year_limit_l1761_176168

theorem best_scrap_year_limit
    (purchase_cost : ℝ)
    (annual_expenses : ℝ)
    (base_maintenance_cost : ℝ)
    (annual_maintenance_increase : ℝ)
    (n : ℕ)
    (n_min_avg : ℝ) :
    purchase_cost = 150000 ∧
    annual_expenses = 15000 ∧
    base_maintenance_cost = 3000 ∧
    annual_maintenance_increase = 3000 ∧
    n = 10 →
    n_min_avg = 10 := by
  sorry

end best_scrap_year_limit_l1761_176168


namespace solve_sqrt_eq_l1761_176129

theorem solve_sqrt_eq (x : ℝ) :
  (Real.sqrt ((1 + Real.sqrt 2)^x) + Real.sqrt ((1 - Real.sqrt 2)^x) = 3) ↔ (x = 2 ∨ x = -2) := 
by sorry

end solve_sqrt_eq_l1761_176129


namespace sum_of_first_fifteen_terms_l1761_176163

noncomputable def a₃ : ℝ := -5
noncomputable def a₅ : ℝ := 2.4
noncomputable def a₁ : ℝ := -12.4
noncomputable def d : ℝ := 3.7

noncomputable def S₁₅ : ℝ := 15 / 2 * (2 * a₁ + 14 * d)

theorem sum_of_first_fifteen_terms :
  S₁₅ = 202.5 := 
by
  sorry

end sum_of_first_fifteen_terms_l1761_176163


namespace polynomial_value_at_minus_1_l1761_176143

-- Definitions for the problem conditions
def polynomial_1 (a b : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x + 1
def polynomial_2 (a b : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x - 2

theorem polynomial_value_at_minus_1 :
  ∀ (a b : ℤ), (a + b = 2022) → polynomial_2 a b (-1) = -2024 :=
by
  intro a b h
  sorry

end polynomial_value_at_minus_1_l1761_176143


namespace combined_weight_after_removal_l1761_176124

theorem combined_weight_after_removal (weight_sugar weight_salt weight_removed : ℕ) 
                                       (h_sugar : weight_sugar = 16)
                                       (h_salt : weight_salt = 30)
                                       (h_removed : weight_removed = 4) : 
                                       (weight_sugar + weight_salt) - weight_removed = 42 :=
by {
  sorry
}

end combined_weight_after_removal_l1761_176124


namespace expression_equals_two_l1761_176159

noncomputable def expression (a b c : ℝ) : ℝ :=
  (1 + a) / (1 + a + a * b) + (1 + b) / (1 + b + b * c) + (1 + c) / (1 + c + c * a)

theorem expression_equals_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  expression a b c = 2 := by
  sorry

end expression_equals_two_l1761_176159


namespace proof_problem_l1761_176192

theorem proof_problem (f g g_inv : ℝ → ℝ) (hinv : ∀ x, f (x ^ 4 - 1) = g x)
  (hginv : ∀ y, g (g_inv y) = y) (h : ∀ y, f (g_inv y) = g (g_inv y)) :
  g_inv (f 15) = 2 :=
by
  sorry

end proof_problem_l1761_176192


namespace find_number_satisfying_9y_eq_number12_l1761_176137

noncomputable def power_9_y (y : ℝ) := (9 : ℝ) ^ y
noncomputable def root_12 (x : ℝ) := x ^ (1 / 12 : ℝ)

theorem find_number_satisfying_9y_eq_number12 :
  ∃ number : ℝ, power_9_y 6 = number ^ 12 ∧ abs (number - 3) < 0.0001 :=
by
  sorry

end find_number_satisfying_9y_eq_number12_l1761_176137


namespace inequality_holds_equality_condition_l1761_176125

theorem inequality_holds (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_not_all_equal : ¬ (a = b ∧ b = c)) : 
  ( (a - b * c)^2 + (b - c * a)^2 + (c - a * b)^2 ) / ( (a - b)^2 + (b - c)^2 + (c - a)^2 ) ≥ 1 / 2 :=
sorry

theorem equality_condition (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_not_all_equal : ¬ (a = b ∧ b = c)) : 
  ( (a - b * c)^2 + (b - c * a)^2 + (c - a * b)^2 ) / ( (a - b)^2 + (b - c)^2 + (c - a)^2 ) = 1 / 2 ↔ 
  ((a = 0 ∧ b = 0 ∧ 0 < c) ∨ (a = 0 ∧ c = 0 ∧ 0 < b) ∨ (b = 0 ∧ c = 0 ∧ 0 < a)) :=
sorry

end inequality_holds_equality_condition_l1761_176125


namespace expand_product_l1761_176112

-- Definitions of the polynomial functions
def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := x^2 + x + 1

-- Statement of the theorem
theorem expand_product : ∀ x : ℝ, (f x) * (g x) = x^3 + 4*x^2 + 4*x + 3 :=
by
  -- Proof goes here, but is omitted for the statement only
  sorry

end expand_product_l1761_176112


namespace monotone_f_range_a_l1761_176116

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x < 1 then 2 * x^2 - 8 * a * x + 3 else Real.log x / Real.log a

theorem monotone_f_range_a (a : ℝ) :
  (∀ (x y : ℝ), x <= y → f a x >= f a y) →
  1 / 2 <= a ∧ a <= 5 / 8 :=
sorry

end monotone_f_range_a_l1761_176116


namespace smallest_possible_value_of_a_largest_possible_value_of_a_l1761_176107

-- Define that a is a positive integer and there are exactly 10 perfect squares greater than a and less than 2a

variable (a : ℕ) (h1 : a > 0)
variable (h2 : ∃ (s : ℕ) (t : ℕ), s + 10 = t ∧ (s^2 > a) ∧ (s + 9)^2 < 2 * a ∧ (t^2 - 10) + 9 < 2 * a)

-- Prove the smallest value of a
theorem smallest_possible_value_of_a : a = 481 :=
by sorry

-- Prove the largest value of a
theorem largest_possible_value_of_a : a = 684 :=
by sorry

end smallest_possible_value_of_a_largest_possible_value_of_a_l1761_176107


namespace find_k_l1761_176104

variable (k : ℕ) (hk : k > 0)

theorem find_k (h : (24 - k) / (8 + k) = 1) : k = 8 :=
by sorry

end find_k_l1761_176104


namespace find_x_l1761_176181

theorem find_x (a y x : ℤ) (h1 : y = 3) (h2 : a * y + x = 10) (h3 : a = 3) : x = 1 :=
by 
  sorry

end find_x_l1761_176181


namespace final_passenger_count_l1761_176190

def total_passengers (initial : ℕ) (first_stop : ℕ) (off_bus : ℕ) (on_bus : ℕ) : ℕ :=
  (initial + first_stop) - off_bus + on_bus

theorem final_passenger_count :
  total_passengers 50 16 22 5 = 49 := by
  sorry

end final_passenger_count_l1761_176190


namespace total_birds_in_pet_store_l1761_176100

theorem total_birds_in_pet_store
  (number_of_cages : ℕ)
  (parrots_per_cage : ℕ)
  (parakeets_per_cage : ℕ)
  (total_birds_in_cage : ℕ)
  (total_birds : ℕ) :
  number_of_cages = 8 →
  parrots_per_cage = 2 →
  parakeets_per_cage = 7 →
  total_birds_in_cage = parrots_per_cage + parakeets_per_cage →
  total_birds = number_of_cages * total_birds_in_cage →
  total_birds = 72 := by
  intros h1 h2 h3 h4 h5
  sorry

end total_birds_in_pet_store_l1761_176100


namespace rope_length_l1761_176174

-- Definitions and assumptions directly derived from conditions
variable (total_length : ℕ)
variable (part_length : ℕ)
variable (sub_part_length : ℕ)

-- Conditions
def condition1 : Prop := total_length / 4 = part_length
def condition2 : Prop := (part_length / 2) * 2 = part_length
def condition3 : Prop := part_length / 2 = sub_part_length
def condition4 : Prop := sub_part_length = 25

-- Proof problem statement
theorem rope_length (h1 : condition1 total_length part_length)
                    (h2 : condition2 part_length)
                    (h3 : condition3 part_length sub_part_length)
                    (h4 : condition4 sub_part_length) :
                    total_length = 100 := 
sorry

end rope_length_l1761_176174


namespace second_train_length_l1761_176180

noncomputable def length_of_second_train (speed1_kmph speed2_kmph : ℝ) (time_seconds : ℝ) (length1_meters : ℝ) : ℝ :=
  let speed1_mps := (speed1_kmph * 1000) / 3600
  let speed2_mps := (speed2_kmph * 1000) / 3600
  let relative_speed_mps := speed1_mps + speed2_mps
  let distance := relative_speed_mps * time_seconds
  distance - length1_meters

theorem second_train_length :
  length_of_second_train 72 18 17.998560115190784 200 = 250 :=
by
  sorry

end second_train_length_l1761_176180


namespace cos_difference_simplification_l1761_176165

theorem cos_difference_simplification :
  let x := Real.cos (20 * Real.pi / 180)
  let y := Real.cos (40 * Real.pi / 180)
  (y = 2 * x^2 - 1) →
  (x = 1 - 2 * y^2) →
  x - y = 1 / 2 :=
by
  intros x y h1 h2
  sorry

end cos_difference_simplification_l1761_176165


namespace find_y_value_l1761_176117

theorem find_y_value (x y : ℝ) (h1 : x^2 + y^2 - 4 = 0) (h2 : x^2 - y + 2 = 0) : y = 2 :=
by sorry

end find_y_value_l1761_176117


namespace polynomial_divisible_by_x_minus_4_l1761_176158

theorem polynomial_divisible_by_x_minus_4 (m : ℤ) :
  (∀ x, 6 * x ^ 3 - 12 * x ^ 2 + m * x - 24 = 0 → x = 4) ↔ m = -42 :=
by
  sorry

end polynomial_divisible_by_x_minus_4_l1761_176158


namespace tan_square_of_cos_double_angle_l1761_176171

theorem tan_square_of_cos_double_angle (α : ℝ) (h : Real.cos (2 * α) = -1/9) : Real.tan (α)^2 = 5/4 :=
by
  sorry

end tan_square_of_cos_double_angle_l1761_176171


namespace roots_diff_l1761_176111

theorem roots_diff (m : ℝ) : 
  (∃ α β : ℝ, 2 * α * α - m * α - 8 = 0 ∧ 
              2 * β * β - m * β - 8 = 0 ∧ 
              α ≠ β ∧ 
              α - β = m - 1) ↔ (m = 6 ∨ m = -10 / 3) :=
by
  sorry

end roots_diff_l1761_176111


namespace necessary_but_not_sufficient_condition_l1761_176115

def p (x : ℝ) : Prop := x < 3
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (q x → p x) ∧ ¬(p x → q x) :=
by
  sorry

end necessary_but_not_sufficient_condition_l1761_176115


namespace number_of_female_officers_l1761_176128

theorem number_of_female_officers (h1 : 0.19 * T = 76) (h2 : T = 152 / 2) : T = 400 :=
by
  sorry

end number_of_female_officers_l1761_176128


namespace Gerald_initial_notebooks_l1761_176105

variable (J G : ℕ)

theorem Gerald_initial_notebooks (h1 : J = G + 13)
    (h2 : J - 5 - 6 = 10) :
    G = 8 :=
sorry

end Gerald_initial_notebooks_l1761_176105


namespace susan_remaining_spaces_to_win_l1761_176132

/-- Susan's board game has 48 spaces. She makes three moves:
 1. She moves forward 8 spaces
 2. She moves forward 2 spaces and then back 5 spaces
 3. She moves forward 6 spaces
 Prove that the remaining spaces she has to move to reach the end is 37.
-/
theorem susan_remaining_spaces_to_win :
  let total_spaces := 48
  let first_turn := 8
  let second_turn := 2 - 5
  let third_turn := 6
  let total_moved := first_turn + second_turn + third_turn
  total_spaces - total_moved = 37 :=
by
  sorry

end susan_remaining_spaces_to_win_l1761_176132


namespace AmpersandDoubleCalculation_l1761_176110

def ampersand (x : Int) : Int := 7 - x
def doubleAmpersand (x : Int) : Int := (x - 7)

theorem AmpersandDoubleCalculation : doubleAmpersand (ampersand 12) = -12 :=
by
  -- This is where the proof would go, which shows the steps described in the solution.
  sorry

end AmpersandDoubleCalculation_l1761_176110


namespace sphere_surface_area_ratio_l1761_176144

theorem sphere_surface_area_ratio (V1 V2 r1 r2 A1 A2 : ℝ)
    (h_volume_ratio : V1 / V2 = 8 / 27)
    (h_volume_formula1 : V1 = (4/3) * Real.pi * r1^3)
    (h_volume_formula2 : V2 = (4/3) * Real.pi * r2^3)
    (h_surface_area_formula1 : A1 = 4 * Real.pi * r1^2)
    (h_surface_area_formula2 : A2 = 4 * Real.pi * r2^2)
    (h_radius_ratio : r1 / r2 = 2 / 3) :
  A1 / A2 = 4 / 9 :=
sorry

end sphere_surface_area_ratio_l1761_176144


namespace Jim_remaining_miles_l1761_176150

-- Define the total journey miles and miles already driven
def total_miles : ℕ := 1200
def miles_driven : ℕ := 215

-- Define the remaining miles Jim needs to drive
def remaining_miles (total driven : ℕ) : ℕ := total - driven

-- Statement to prove
theorem Jim_remaining_miles : remaining_miles total_miles miles_driven = 985 := by
  -- The proof is omitted
  sorry

end Jim_remaining_miles_l1761_176150


namespace syntheticMethod_correct_l1761_176123

-- Definition: The synthetic method leads from cause to effect.
def syntheticMethod (s : String) : Prop :=
  s = "The synthetic method leads from cause to effect, gradually searching for the necessary conditions that are known."

-- Question: Is the statement correct?
def question : String :=
  "The thought process of the synthetic method is to lead from cause to effect, gradually searching for the necessary conditions that are known."

-- Options given
def options : List String := ["Correct", "Incorrect", "", ""]

-- Correct answer is Option A - "Correct"
def correctAnswer : String := "Correct"

theorem syntheticMethod_correct :
  syntheticMethod question → options.head? = some correctAnswer :=
sorry

end syntheticMethod_correct_l1761_176123


namespace total_distance_joseph_ran_l1761_176193

-- Defining the conditions
def distance_per_day : ℕ := 900
def days_run : ℕ := 3

-- The proof problem statement
theorem total_distance_joseph_ran :
  (distance_per_day * days_run) = 2700 :=
by
  sorry

end total_distance_joseph_ran_l1761_176193


namespace cost_price_of_watch_l1761_176102

-- Let C be the cost price of the watch
variable (C : ℝ)

-- Conditions: The selling price at a loss of 8% and the selling price with a gain of 4% if sold for Rs. 140 more
axiom loss_condition : 0.92 * C + 140 = 1.04 * C

-- Objective: Prove that C = 1166.67
theorem cost_price_of_watch : C = 1166.67 :=
by
  have h := loss_condition
  sorry

end cost_price_of_watch_l1761_176102


namespace work_duration_l1761_176121

/-- Definition of the work problem, showing that the work lasts for 5 days. -/
theorem work_duration (work_rate_p work_rate_q : ℝ) (total_work time_p time_q : ℝ) 
  (p_work_days q_work_days : ℝ) 
  (H1 : p_work_days = 10)
  (H2 : q_work_days = 6)
  (H3 : work_rate_p = total_work / 10)
  (H4 : work_rate_q = total_work / 6)
  (H5 : time_p = 2)
  (H6 : time_q = 4 * total_work / 5 / (total_work / 2 / 3) )
  : (time_p + time_q = 5) := 
by 
  sorry

end work_duration_l1761_176121


namespace min_bailing_rate_l1761_176157

noncomputable def slowest_bailing_rate (distance : ℝ) (rowing_speed : ℝ) (leak_rate : ℝ) (max_capacity : ℝ) : ℝ :=
  let time_to_shore := distance / rowing_speed
  let time_to_shore_in_minutes := time_to_shore * 60
  let total_water_intake := leak_rate * time_to_shore_in_minutes
  let excess_water := total_water_intake - max_capacity
  excess_water / time_to_shore_in_minutes

theorem min_bailing_rate : slowest_bailing_rate 3 3 14 40 = 13.3 :=
by
  sorry

end min_bailing_rate_l1761_176157


namespace total_spending_is_450_l1761_176151

-- Define the costs of items bought by Leonard
def leonard_wallet_cost : ℕ := 50
def pair_of_sneakers_cost : ℕ := 100
def pairs_of_sneakers : ℕ := 2

-- Define the costs of items bought by Michael
def michael_backpack_cost : ℕ := 100
def pair_of_jeans_cost : ℕ := 50
def pairs_of_jeans : ℕ := 2

-- Define the total spending of Leonard and Michael 
def total_spent : ℕ :=
  leonard_wallet_cost + (pair_of_sneakers_cost * pairs_of_sneakers) + 
  michael_backpack_cost + (pair_of_jeans_cost * pairs_of_jeans)

-- The proof statement
theorem total_spending_is_450 : total_spent = 450 := 
by
  sorry

end total_spending_is_450_l1761_176151


namespace at_least_one_not_less_than_2_l1761_176141

theorem at_least_one_not_less_than_2 (x y z : ℝ) (hp : 0 < x ∧ 0 < y ∧ 0 < z) :
  let a := x + 1/y
  let b := y + 1/z
  let c := z + 1/x
  (a ≥ 2 ∨ b ≥ 2 ∨ c ≥ 2) := by
    sorry

end at_least_one_not_less_than_2_l1761_176141


namespace fg_minus_gf_eq_zero_l1761_176142

noncomputable def f (x : ℝ) : ℝ := 4 * x + 6

noncomputable def g (x : ℝ) : ℝ := x / 2 - 1

theorem fg_minus_gf_eq_zero (x : ℝ) : (f (g x)) - (g (f x)) = 0 :=
by
  sorry

end fg_minus_gf_eq_zero_l1761_176142


namespace problem_a_l1761_176188

theorem problem_a : (1038^2 % 1000) ≠ 4 := by
  sorry

end problem_a_l1761_176188


namespace recipe_flour_cups_l1761_176196

theorem recipe_flour_cups (F : ℕ) : 
  (exists (sugar : ℕ) (flourAdded : ℕ) (sugarExtra : ℕ), sugar = 11 ∧ flourAdded = 4 ∧ sugarExtra = 6 ∧ ((F - flourAdded) + sugarExtra = sugar)) →
  F = 9 :=
sorry

end recipe_flour_cups_l1761_176196


namespace exists_valid_arrangement_n_1_exists_valid_arrangement_n_gt_1_l1761_176118

-- Define the conditions
def num_mathematicians (n : ℕ) : ℕ := 6 * n + 4
def num_meetings (n : ℕ) : ℕ := 2 * n + 1
def num_4_person_tables (n : ℕ) : ℕ := 1
def num_6_person_tables (n : ℕ) : ℕ := n

-- Define the constraint on arrangements
def valid_arrangement (n : ℕ) : Prop :=
  -- A placeholder for the actual arrangement checking logic.
  -- This should ensure no two people sit next to or opposite each other more than once.
  sorry

-- Proof of existence of a valid arrangement when n = 1
theorem exists_valid_arrangement_n_1 : valid_arrangement 1 :=
sorry

-- Proof of existence of a valid arrangement when n > 1
theorem exists_valid_arrangement_n_gt_1 (n : ℕ) (h : n > 1) : valid_arrangement n :=
sorry

end exists_valid_arrangement_n_1_exists_valid_arrangement_n_gt_1_l1761_176118


namespace arithmetic_sequence_ratio_l1761_176149

variable {α : Type}
variable [LinearOrderedField α]

def a1 (a_1 : α) : Prop := a_1 ≠ 0 
def a2_eq_3a1 (a_1 a_2 : α) : Prop := a_2 = 3 * a_1 

noncomputable def common_difference (a_1 a_2 : α) : α :=
  a_2 - a_1

noncomputable def S (n : ℕ) (a_1 d : α) : α :=
  n * (2 * a_1 + (n - 1) * d) / 2

theorem arithmetic_sequence_ratio
  (a_1 a_2 : α)
  (h₀ : a1 a_1)
  (h₁ : a2_eq_3a1 a_1 a_2) :
  (S 10 a_1 (common_difference a_1 a_2)) / (S 5 a_1 (common_difference a_1 a_2)) = 4 := 
by
  sorry

end arithmetic_sequence_ratio_l1761_176149


namespace smallest_number_diminished_by_8_divisible_by_9_6_18_equals_62_l1761_176139

theorem smallest_number_diminished_by_8_divisible_by_9_6_18_equals_62
  (n: ℕ) (h1: n - 8 = 44) 
  (h2: (n - 8) % 9 = 0)
  (h3: (n - 8) % 6 = 0)
  (h4: (n - 8) % 18 = 0) : 
  n = 62 :=
sorry

end smallest_number_diminished_by_8_divisible_by_9_6_18_equals_62_l1761_176139


namespace matrix_power_four_l1761_176197

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, -1], ![1, 1]]

theorem matrix_power_four :
  (A^4) = ![![0, -9], ![9, -9]] :=
by
  sorry

end matrix_power_four_l1761_176197


namespace solve_diamond_eq_l1761_176103

noncomputable def diamond_op (a b : ℝ) := a / b

theorem solve_diamond_eq (x : ℝ) (h : x ≠ 0) : diamond_op 2023 (diamond_op 7 x) = 150 ↔ x = 1050 / 2023 := by
  sorry

end solve_diamond_eq_l1761_176103


namespace julia_bill_ratio_l1761_176126

-- Definitions
def saturday_miles_b (s_b : ℕ) (s_su : ℕ) := s_su = s_b + 4
def sunday_miles_j (s_su : ℕ) (t : ℕ) (s_j : ℕ) := s_j = t * s_su
def total_weekend_miles (s_b : ℕ) (s_su : ℕ) (s_j : ℕ) := s_b + s_su + s_j = 36

-- Proof statement
theorem julia_bill_ratio (s_b s_su s_j : ℕ) (h1 : saturday_miles_b s_b s_su) (h3 : total_weekend_miles s_b s_su s_j) (h_su : s_su = 10) : (2 * s_su = s_j) :=
by
  sorry  -- proof

end julia_bill_ratio_l1761_176126


namespace cistern_wet_surface_area_l1761_176160

noncomputable def total_wet_surface_area (length : ℝ) (width : ℝ) (depth : ℝ) : ℝ :=
  let bottom_surface_area := length * width
  let longer_side_area := 2 * (depth * length)
  let shorter_side_area := 2 * (depth * width)
  bottom_surface_area + longer_side_area + shorter_side_area

theorem cistern_wet_surface_area :
  total_wet_surface_area 9 4 1.25 = 68.5 :=
by
  sorry

end cistern_wet_surface_area_l1761_176160


namespace solve_equation_l1761_176134

theorem solve_equation (x : ℝ) (h : x = 5) :
  (3 * x - 5) / (x^2 - 7 * x + 12) + (5 * x - 1) / (x^2 - 5 * x + 6) = (8 * x - 13) / (x^2 - 6 * x + 8) := 
  by 
  rw [h]
  sorry

end solve_equation_l1761_176134


namespace polynomial_divisible_by_x_sub_a_squared_l1761_176153

theorem polynomial_divisible_by_x_sub_a_squared (a x : ℕ) (n : ℕ) 
    (h : a ≠ 0) : ∃ q : ℕ → ℕ, x ^ n - n * a ^ (n - 1) * x + (n - 1) * a ^ n = (x - a) ^ 2 * q x := 
by 
  sorry

end polynomial_divisible_by_x_sub_a_squared_l1761_176153


namespace divisible_2n_minus_3_l1761_176178

theorem divisible_2n_minus_3 (n : ℕ) : (2^n - 1)^n - 3 ≡ 0 [MOD 2^n - 3] :=
by
  sorry

end divisible_2n_minus_3_l1761_176178


namespace total_distance_covered_l1761_176152

theorem total_distance_covered :
  let speed_upstream := 12 -- km/h
  let time_upstream := 2 -- hours
  let speed_downstream := 38 -- km/h
  let time_downstream := 1 -- hour
  let distance_upstream := speed_upstream * time_upstream
  let distance_downstream := speed_downstream * time_downstream
  distance_upstream + distance_downstream = 62 := by
  sorry

end total_distance_covered_l1761_176152


namespace jamie_catches_bus_probability_l1761_176189

noncomputable def probability_jamie_catches_bus : ℝ :=
  let total_area := 120 * 120
  let overlap_area := 20 * 100
  overlap_area / total_area

theorem jamie_catches_bus_probability :
  probability_jamie_catches_bus = (5 / 36) :=
by
  sorry

end jamie_catches_bus_probability_l1761_176189


namespace lucy_first_round_cookies_l1761_176185

theorem lucy_first_round_cookies (x : ℕ) : 
  (x + 27 = 61) → x = 34 :=
by
  intros h
  sorry

end lucy_first_round_cookies_l1761_176185


namespace fraction_inequality_solution_l1761_176199

theorem fraction_inequality_solution (x : ℝ) :
  -1 ≤ x ∧ x ≤ 3 ∧ (4 * x + 3 > 2 * (8 - 3 * x)) → (13 / 10) < x ∧ x ≤ 3 :=
by
  sorry

end fraction_inequality_solution_l1761_176199


namespace solve_symbols_values_l1761_176145

def square_value : Nat := 423 / 47

def boxminus_and_boxtimes_relation (boxminus boxtimes : Nat) : Prop :=
  1448 = 282 * boxminus + 9 * boxtimes

def boxtimes_value : Nat := 38 / 9

def boxplus_value : Nat := 846 / 423

theorem solve_symbols_values :
  ∃ (square boxplus boxtimes boxminus : Nat),
    square = 9 ∧
    boxplus = 2 ∧
    boxtimes = 8 ∧
    boxminus = 5 ∧
    square = 423 / 47 ∧
    1448 = 282 * boxminus + 9 * boxtimes ∧
    9 * boxtimes = 38 ∧
    423 * boxplus / 3 = 282 := by
  sorry

end solve_symbols_values_l1761_176145


namespace bcdeq65_l1761_176131

theorem bcdeq65 (a b c d e f : ℝ)
  (h₁ : a * b * c = 130)
  (h₂ : c * d * e = 500)
  (h₃ : d * e * f = 250)
  (h₄ : (a * f) / (c * d) = 1) :
  b * c * d = 65 :=
sorry

end bcdeq65_l1761_176131


namespace fixed_point_of_line_l1761_176119

theorem fixed_point_of_line :
  ∀ m : ℝ, ∀ x y : ℝ, (y - 2 = m * (x + 1)) → (x = -1 ∧ y = 2) :=
by sorry

end fixed_point_of_line_l1761_176119


namespace measure_of_y_l1761_176154

variables (A B C D : Point) (y : ℝ)
-- Given conditions
def angle_ABC := 120
def angle_BAD := 30
def angle_BDA := 21
def angle_ABD := 180 - angle_ABC

-- Theorem to prove
theorem measure_of_y :
  angle_BAD + angle_ABD + angle_BDA + y = 180 → y = 69 :=
by
  sorry

end measure_of_y_l1761_176154


namespace bus_stop_time_l1761_176187

/-- 
  We are given:
  speed_ns: speed of bus without stoppages (32 km/hr)
  speed_ws: speed of bus including stoppages (16 km/hr)
  
  We need to prove the bus stops for t = 30 minutes each hour.
-/
theorem bus_stop_time
  (speed_ns speed_ws: ℕ)
  (h_ns: speed_ns = 32)
  (h_ws: speed_ws = 16):
  ∃ t: ℕ, t = 30 := 
sorry

end bus_stop_time_l1761_176187


namespace g_five_eq_one_l1761_176186

variable (g : ℝ → ℝ)
variable (h : ∀ x y : ℝ, g (x - y) = g x * g y)
variable (h_ne_zero : ∀ x : ℝ, g x ≠ 0)

theorem g_five_eq_one : g 5 = 1 :=
by
  sorry

end g_five_eq_one_l1761_176186


namespace factorization_correct_l1761_176127

theorem factorization_correct : ∃ a b : ℤ, (5*y + a)*(y + b) = 5*y^2 + 17*y + 6 ∧ a - b = -1 := by
  sorry

end factorization_correct_l1761_176127


namespace transformed_sequence_has_large_element_l1761_176114

noncomputable def transformed_value (a : Fin 25 → ℤ) (i : Fin 25) : ℤ :=
  a i + a ((i + 1) % 25)

noncomputable def perform_transformation (a : Fin 25 → ℤ) (n : ℕ) : Fin 25 → ℤ :=
  if n = 0 then a
  else perform_transformation (fun i => transformed_value a i) (n - 1)

theorem transformed_sequence_has_large_element :
  ∀ a : Fin 25 → ℤ,
    (∀ i : Fin 13, a i = 1) →
    (∀ i : Fin 12, a (i + 13) = -1) →
    ∃ i : Fin 25, perform_transformation a 100 i > 10^20 :=
by
  sorry

end transformed_sequence_has_large_element_l1761_176114


namespace contractor_realized_after_20_days_l1761_176170

-- Defining the conditions as assumptions
variables {W : ℝ} {r : ℝ} {x : ℝ} -- Total work, rate per person per day, and unknown number of days

-- Condition 1: 10 people to complete W work in x days results in one fourth completed
axiom one_fourth_work_done (W : ℝ) (r : ℝ) (x : ℝ) :
  10 * r * x = W / 4

-- Condition 2: After firing 2 people, 8 people complete three fourths of work in 75 days
axiom remaining_three_fourths_work_done (W : ℝ) (r : ℝ) :
  8 * r * 75 = 3 * (W / 4)

-- Theorem: The contractor realized that one fourth of the work was done after 20 days
theorem contractor_realized_after_20_days (W : ℝ) (r : ℝ) (x : ℝ) :
  10 * r * x = W / 4 → (8 * r * 75 = 3 * (W / 4)) → x = 20 := 
sorry

end contractor_realized_after_20_days_l1761_176170


namespace smallest_number_among_four_l1761_176140

theorem smallest_number_among_four (a b c d : ℤ) (h1 : a = 2023) (h2 : b = 2022) (h3 : c = -2023) (h4 : d = -2022) : 
  min (min a (min b c)) d = -2023 :=
by
  rw [h1, h2, h3, h4]
  sorry

end smallest_number_among_four_l1761_176140


namespace pq_plus_sum_eq_20_l1761_176146

theorem pq_plus_sum_eq_20 
  (p q : ℕ) 
  (hp : p > 0) 
  (hq : q > 0) 
  (hpl : p < 30) 
  (hql : q < 30) 
  (heq : p + q + p * q = 119) : 
  p + q = 20 :=
sorry

end pq_plus_sum_eq_20_l1761_176146


namespace combinatorial_problem_correct_l1761_176172

def combinatorial_problem : Prop :=
  let boys := 4
  let girls := 3
  let chosen_boys := 3
  let chosen_girls := 2
  let num_ways_select := Nat.choose boys chosen_boys * Nat.choose girls chosen_girls
  let arrangements_no_consecutive_girls := 6 * Nat.factorial 4 / Nat.factorial 2
  num_ways_select * arrangements_no_consecutive_girls = 864

theorem combinatorial_problem_correct : combinatorial_problem := 
  by 
  -- proof to be provided
  sorry

end combinatorial_problem_correct_l1761_176172


namespace triangle_inequality_l1761_176120

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  a^2 * c * (a - b) + b^2 * a * (b - c) + c^2 * b * (c - a) ≥ 0 :=
sorry

end triangle_inequality_l1761_176120


namespace problem_statement_l1761_176101

noncomputable def f1 (x : ℝ) : ℝ := x ^ 2

noncomputable def f2 (x : ℝ) : ℝ := 8 / x

noncomputable def f (x : ℝ) : ℝ := f1 x + f2 x

theorem problem_statement (a : ℝ) (h : a > 3) : 
  ∃ x1 x2 x3 : ℝ, 
  (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) ∧ 
  (f x1 = f a ∧ f x2 = f a ∧ f x3 = f a) ∧ 
  (x1 ≠ 0 ∧ x2 ≠ 0 ∧ x3 ≠ 0) := 
sorry

end problem_statement_l1761_176101


namespace root_value_l1761_176106

theorem root_value (m : ℝ) (h : 2 * m^2 - 7 * m + 1 = 0) : m * (2 * m - 7) + 5 = 4 := by
  sorry

end root_value_l1761_176106


namespace solve_for_x_l1761_176179

theorem solve_for_x (x : ℤ) (h : 5 * x + 3 = 10 * x - 22) : x = 5 :=
sorry

end solve_for_x_l1761_176179


namespace surface_area_ratio_l1761_176155

noncomputable def surface_area (r : ℝ) : ℝ :=
  4 * Real.pi * r ^ 2

theorem surface_area_ratio (k : ℝ) :
  let r1 := k
  let r2 := 2 * k
  let r3 := 3 * k
  let A1 := surface_area r1
  let A2 := surface_area r2
  let A3 := surface_area r3
  A3 / (A1 + A2) = 9 / 5 :=
by
  sorry

end surface_area_ratio_l1761_176155


namespace total_fuel_l1761_176156

theorem total_fuel (fuel_this_week : ℝ) (reduction_percent : ℝ) :
  fuel_this_week = 15 → reduction_percent = 0.20 → 
  (fuel_this_week + (fuel_this_week * (1 - reduction_percent))) = 27 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end total_fuel_l1761_176156


namespace cos_product_identity_l1761_176130

theorem cos_product_identity :
  3.422 * (Real.cos (π / 15)) * (Real.cos (2 * π / 15)) * (Real.cos (3 * π / 15)) *
  (Real.cos (4 * π / 15)) * (Real.cos (5 * π / 15)) * (Real.cos (6 * π / 15)) * (Real.cos (7 * π / 15)) =
  (1 / 2^7) :=
sorry

end cos_product_identity_l1761_176130


namespace evaluate_expression_l1761_176138

theorem evaluate_expression (x : ℝ) (h : x = -3) : (5 + x * (5 + x) - 5^2) / (x - 5 + x^2) = -26 :=
by
  rw [h]
  sorry

end evaluate_expression_l1761_176138


namespace Patricia_read_21_books_l1761_176162

theorem Patricia_read_21_books
  (Candice_books Amanda_books Kara_books Patricia_books : ℕ)
  (h1 : Candice_books = 18)
  (h2 : Candice_books = 3 * Amanda_books)
  (h3 : Kara_books = Amanda_books / 2)
  (h4 : Patricia_books = 7 * Kara_books) :
  Patricia_books = 21 :=
by
  sorry

end Patricia_read_21_books_l1761_176162


namespace thomas_task_completion_l1761_176113

theorem thomas_task_completion :
  (∃ T E : ℝ, (1 / T + 1 / E = 1 / 8) ∧ (13 / T + 6 / E = 1)) →
  ∃ T : ℝ, T = 14 :=
by
  sorry

end thomas_task_completion_l1761_176113


namespace simplify_expression_l1761_176175

variable (x : ℝ)

theorem simplify_expression : (20 * x^2) * (5 * x) * (1 / (2 * x)^2) * (2 * x)^2 = 100 * x^3 := 
by 
  sorry

end simplify_expression_l1761_176175


namespace density_function_Y_l1761_176166

noncomputable def f (x : ℝ) : ℝ := (1 / Real.sqrt (2 * Real.pi)) * Real.exp (-x^2 / 2)

theorem density_function_Y (y : ℝ) (hy : 0 < y) : 
  (∃ (g : ℝ → ℝ), (∀ y, g y = (1 / Real.sqrt (2 * Real.pi * y)) * Real.exp (- y / 2))) :=
sorry

end density_function_Y_l1761_176166


namespace remainder_of_f_div_r_minus_2_l1761_176167

def f (r : ℝ) : ℝ := r^15 - 3

theorem remainder_of_f_div_r_minus_2 : f 2 = 32765 := by
  sorry

end remainder_of_f_div_r_minus_2_l1761_176167


namespace max_diff_six_digit_even_numbers_l1761_176184

-- Definitions for six-digit numbers with all digits even
def is_6_digit_even (n : ℕ) : Prop :=
  n >= 100000 ∧ n < 1000000 ∧ (∀ (d : ℕ), d < 6 → (n / 10^d) % 10 % 2 = 0)

def contains_odd_digit (n : ℕ) : Prop :=
  ∃ (d : ℕ), d < 6 ∧ (n / 10^d) % 10 % 2 = 1

-- The main theorem
theorem max_diff_six_digit_even_numbers (a b : ℕ) 
  (ha : is_6_digit_even a) 
  (hb : is_6_digit_even b)
  (h_cond : ∀ n : ℕ, a < n ∧ n < b → contains_odd_digit n) 
  : b - a = 111112 :=
sorry

end max_diff_six_digit_even_numbers_l1761_176184


namespace consecutive_odd_numbers_l1761_176183

theorem consecutive_odd_numbers (a b c d e : ℤ) (h1 : b = a + 2) (h2 : c = a + 4) (h3 : d = a + 6) (h4 : e = a + 8) (h5 : a + c = 146) : e = 79 := 
by
  sorry

end consecutive_odd_numbers_l1761_176183


namespace chord_length_squared_l1761_176161

theorem chord_length_squared
  (r5 r10 r15 : ℝ) 
  (externally_tangent : r5 = 5 ∧ r10 = 10)
  (internally_tangent : r15 = 15)
  (common_external_tangent : r15 - r10 - r5 = 0) :
  ∃ PQ_squared : ℝ, PQ_squared = 622.44 :=
by
  sorry

end chord_length_squared_l1761_176161


namespace find_rate_percent_l1761_176191

-- Define the conditions based on the problem statement
def principal : ℝ := 800
def simpleInterest : ℝ := 160
def time : ℝ := 5

-- Create the statement to prove the rate percent
theorem find_rate_percent : ∃ (rate : ℝ), simpleInterest = (principal * rate * time) / 100 := sorry

end find_rate_percent_l1761_176191


namespace miriam_flowers_total_l1761_176182

theorem miriam_flowers_total :
  let monday_flowers := 45
  let tuesday_flowers := 75
  let wednesday_flowers := 35
  let thursday_flowers := 105
  let friday_flowers := 0
  let saturday_flowers := 60
  (monday_flowers + tuesday_flowers + wednesday_flowers + thursday_flowers + friday_flowers + saturday_flowers) = 320 :=
by
  -- Calculations go here but we're using sorry to skip them
  sorry

end miriam_flowers_total_l1761_176182


namespace valid_range_for_b_l1761_176198

noncomputable def f (x b : ℝ) : ℝ := -x^2 + 2 * x + b^2 - b + 1

theorem valid_range_for_b (b : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x b > 0) → (b < -1 ∨ b > 2) :=
by
  sorry

end valid_range_for_b_l1761_176198


namespace x_y_divisible_by_3_l1761_176122

theorem x_y_divisible_by_3
    (x y z t : ℤ)
    (h : x^3 + y^3 = 3 * (z^3 + t^3)) :
    (3 ∣ x) ∧ (3 ∣ y) :=
by sorry

end x_y_divisible_by_3_l1761_176122


namespace math_problems_l1761_176177

theorem math_problems (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  (a * (6 - a) ≤ 9) ∧
  (ab = a + b + 3 → ab ≥ 9) ∧
  ¬(∀ x : ℝ, 0 < x → x^2 + 4 / (x^2 + 3) ≥ 1) ∧
  (a + b = 2 → 1 / a + 2 / b ≥ 3 / 2 + Real.sqrt 2) :=
by
  sorry

end math_problems_l1761_176177


namespace distance_between_foci_of_hyperbola_l1761_176133

-- Define the asymptotes as lines
def asymptote1 (x : ℝ) : ℝ := 2 * x + 3
def asymptote2 (x : ℝ) : ℝ := -2 * x + 7

-- Define the condition that the hyperbola passes through the point (4, 5)
def passes_through (x y : ℝ) : Prop := (x, y) = (4, 5)

-- Statement to prove
theorem distance_between_foci_of_hyperbola : 
  (asymptote1 4 = 5) ∧ (asymptote2 4 = 5) ∧ passes_through 4 5 → 
  (∀ a b c : ℝ, a^2 = 9 ∧ b^2 = 9/4 ∧ c^2 = a^2 + b^2 → 2 * c = 3 * Real.sqrt 5) := 
by
  intro h
  sorry

end distance_between_foci_of_hyperbola_l1761_176133


namespace arithmetic_sequence_sum_ratio_l1761_176194

theorem arithmetic_sequence_sum_ratio
  (S : ℕ → ℝ) (T : ℕ → ℝ) (a b : ℕ → ℝ)
  (k : ℝ)
  (h1 : ∀ n, S n = 3 * k * n^2)
  (h2 : ∀ n, T n = k * n * (2 * n + 1))
  (h3 : ∀ n, a n = S n - S (n - 1))
  (h4 : ∀ n, b n = T n - T (n - 1))
  (h5 : ∀ n, S n / T n = (3 * n) / (2 * n + 1)) :
  (a 1 + a 2 + a 14 + a 19) / (b 1 + b 3 + b 17 + b 19) = 17 / 13 :=
sorry

end arithmetic_sequence_sum_ratio_l1761_176194


namespace find_number_l1761_176108

theorem find_number (x : ℚ) (h : x / 11 + 156 = 178) : x = 242 :=
sorry

end find_number_l1761_176108


namespace _l1761_176164

noncomputable def gear_speeds_relationship (x y z : ℕ) (ω₁ ω₂ ω₃ : ℝ) 
  (h1 : 2 * x * ω₁ = 3 * y * ω₂)
  (h2 : 3 * y * ω₂ = 4 * z * ω₃) : Prop :=
  ω₁ = (2 * z / x) * ω₃ ∧ ω₂ = (4 * z / (3 * y)) * ω₃

-- Example theorem statement
example (x y z : ℕ) (ω₁ ω₂ ω₃ : ℝ)
  (h1 : 2 * x * ω₁ = 3 * y * ω₂)
  (h2 : 3 * y * ω₂ = 4 * z * ω₃) : gear_speeds_relationship x y z ω₁ ω₂ ω₃ h1 h2 :=
by sorry

end _l1761_176164


namespace test_questions_l1761_176148

theorem test_questions (x : ℕ) (h1 : x % 5 = 0) (h2 : 70 < 32 * 100 / x) (h3 : 32 * 100 / x < 77) : x = 45 := 
by sorry

end test_questions_l1761_176148


namespace trip_duration_exactly_six_hours_l1761_176135

theorem trip_duration_exactly_six_hours : 
  ∀ start_time end_time : ℕ,
  (start_time = (8 * 60 + 43 * 60 / 11)) ∧ 
  (end_time = (14 * 60 + 43 * 60 / 11)) → 
  (end_time - start_time) = 6 * 60 :=
by
  sorry

end trip_duration_exactly_six_hours_l1761_176135


namespace jogging_track_circumference_l1761_176176

noncomputable def Deepak_speed : ℝ := 4.5 -- km/hr
noncomputable def Wife_speed : ℝ := 3.75 -- km/hr
noncomputable def time_meet : ℝ := 4.8 / 60 -- hours

noncomputable def Distance_Deepak : ℝ := Deepak_speed * time_meet
noncomputable def Distance_Wife : ℝ := Wife_speed * time_meet

theorem jogging_track_circumference : 2 * (Distance_Deepak + Distance_Wife) = 1.32 := by
  sorry

end jogging_track_circumference_l1761_176176


namespace douglas_weight_proof_l1761_176173

theorem douglas_weight_proof : 
  ∀ (anne_weight douglas_weight : ℕ), 
  anne_weight = 67 →
  anne_weight = douglas_weight + 15 →
  douglas_weight = 52 :=
by 
  intros anne_weight douglas_weight h1 h2 
  sorry

end douglas_weight_proof_l1761_176173


namespace exist_sequences_l1761_176147

def sequence_a (a : ℕ → ℤ) : Prop :=
  a 0 = 4 ∧ a 1 = 22 ∧ ∀ n ≥ 2, a n = 6 * a (n - 1) - a (n - 2)

theorem exist_sequences (a : ℕ → ℤ) (x y : ℕ → ℤ) :
  sequence_a a → (∀ n, 0 < x n ∧ 0 < y n) →
  (∀ n, a n = (y n ^ 2 + 7) / (x n - y n)) :=
by
  intro h_seq_a h_pos
  sorry

end exist_sequences_l1761_176147


namespace value_of_f_at_2_l1761_176109

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem value_of_f_at_2 : f 2 = 3 := sorry

end value_of_f_at_2_l1761_176109


namespace required_workers_l1761_176195

variable (x : ℕ) (y : ℕ)

-- Each worker can produce x units of a craft per day.
-- A craft factory needs to produce 60 units of this craft per day.

theorem required_workers (h : x > 0) : y = 60 / x ↔ x * y = 60 :=
by sorry

end required_workers_l1761_176195


namespace max_value_of_function_l1761_176169

theorem max_value_of_function : 
  ∃ x : ℝ, 
  (∀ y : ℝ, (y == (2*x^2 - 2*x + 3) / (x^2 - x + 1)) → y ≤ 10/3) ∧
  (∃ x : ℝ, (2*x^2 - 2*x + 3) / (x^2 - x + 1) = 10/3) := 
sorry

end max_value_of_function_l1761_176169
