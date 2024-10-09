import Mathlib

namespace explicit_expression_for_f_l708_70813

variable (f : ℕ → ℕ)

-- Define the condition
axiom h : ∀ x : ℕ, f (x + 1) = 3 * x + 2

-- State the theorem
theorem explicit_expression_for_f (x : ℕ) : f x = 3 * x - 1 :=
by {
  sorry
}

end explicit_expression_for_f_l708_70813


namespace cost_price_correct_l708_70806

open Real

-- Define the cost price of the table
def cost_price (C : ℝ) : ℝ := C

-- Define the marked price
def marked_price (C : ℝ) : ℝ := 1.30 * C

-- Define the discounted price
def discounted_price (C : ℝ) : ℝ := 0.85 * (marked_price C)

-- Define the final price after sales tax
def final_price (C : ℝ) : ℝ := 1.12 * (discounted_price C)

-- Given that the final price is 9522.84
axiom final_price_value : final_price 9522.84 = 1.2376 * 7695

-- Main theorem stating the problem to prove
theorem cost_price_correct (C : ℝ) : final_price C = 9522.84 -> C = 7695 := by
  sorry

end cost_price_correct_l708_70806


namespace max_angle_MPN_is_pi_over_2_l708_70827

open Real

noncomputable def max_angle_MPN (θ : ℝ) (P : ℝ × ℝ) (hP : (P.1 - cos θ)^2 + (P.2 - sin θ)^2 = 1/25) : ℝ :=
  sorry

theorem max_angle_MPN_is_pi_over_2 (θ : ℝ) (P : ℝ × ℝ) (hP : (P.1 - cos θ)^2 + (P.2 - sin θ)^2 = 1/25) : 
  max_angle_MPN θ P hP = π / 2 :=
sorry

end max_angle_MPN_is_pi_over_2_l708_70827


namespace tan_45_eq_1_l708_70839

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end tan_45_eq_1_l708_70839


namespace pints_in_two_liters_nearest_tenth_l708_70871

def liters_to_pints (liters : ℝ) : ℝ :=
  2.1 * liters

theorem pints_in_two_liters_nearest_tenth :
  liters_to_pints 2 = 4.2 :=
by
  sorry

end pints_in_two_liters_nearest_tenth_l708_70871


namespace votes_ratio_l708_70848

theorem votes_ratio (V : ℝ) 
  (counted_fraction : ℝ := 2/9) 
  (favor_fraction : ℝ := 3/4) 
  (against_fraction_remaining : ℝ := 0.7857142857142856) :
  let counted := counted_fraction * V
  let favor_counted := favor_fraction * counted
  let remaining := V - counted
  let against_remaining := against_fraction_remaining * remaining
  let against_counted := (1 - favor_fraction) * counted
  let total_against := against_counted + against_remaining
  let total_favor := favor_counted
  (total_against / total_favor) = 4 :=
by
  sorry

end votes_ratio_l708_70848


namespace mary_pizza_order_l708_70885

theorem mary_pizza_order (p e r n : ℕ) (h1 : p = 8) (h2 : e = 7) (h3 : r = 9) :
  n = (r + e) / p → n = 2 :=
by
  sorry

end mary_pizza_order_l708_70885


namespace arithmetic_sequence_15th_term_l708_70856

theorem arithmetic_sequence_15th_term : 
  let a₁ := 3
  let d := 4
  let n := 15
  a₁ + (n - 1) * d = 59 :=
by
  let a₁ := 3
  let d := 4
  let n := 15
  sorry

end arithmetic_sequence_15th_term_l708_70856


namespace find_value_b_in_geometric_sequence_l708_70879

theorem find_value_b_in_geometric_sequence
  (b : ℝ)
  (h1 : 15 ≠ 0) -- to ensure division by zero does not occur
  (h2 : b ≠ 0)  -- to ensure division by zero does not occur
  (h3 : 15 * (b / 15) = b) -- 15 * r = b
  (h4 : b * (b / 15) = 45 / 4) -- b * r = 45 / 4
  : b = 15 * Real.sqrt 3 / 2 :=
sorry

end find_value_b_in_geometric_sequence_l708_70879


namespace find_angle_A_find_side_a_l708_70807

variable {A B C a b c : Real}
variable {area : Real}
variable (h1 : a * Real.sin C = Real.sqrt 3 * c * Real.cos A)
variable (h2 : b = 2)
variable (h3 : area = Real.sqrt 3)
variable (h4 : area = 1 / 2 * b * c * Real.sin A)

theorem find_angle_A (h1 : a * Real.sin C = Real.sqrt 3 * c * Real.cos A) : A = Real.pi / 3 :=
  sorry

theorem find_side_a (h4 : area = 1 / 2 * b * c * Real.sin A) (h2 : b = 2) (h3 : area = Real.sqrt 3) : a = 2 :=
  sorry

end find_angle_A_find_side_a_l708_70807


namespace olivia_total_earnings_l708_70849

variable (rate : ℕ) (hours_monday : ℕ) (hours_wednesday : ℕ) (hours_friday : ℕ)

def olivia_earnings : ℕ := hours_monday * rate + hours_wednesday * rate + hours_friday * rate

theorem olivia_total_earnings :
  rate = 9 → hours_monday = 4 → hours_wednesday = 3 → hours_friday = 6 → olivia_earnings rate hours_monday hours_wednesday hours_friday = 117 :=
by
  sorry

end olivia_total_earnings_l708_70849


namespace multiplication_problem_solution_l708_70887

theorem multiplication_problem_solution (a b c : ℕ) 
  (h1 : a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1) 
  (h2 : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h3 : (a * 100 + b * 10 + b) * c = b * 1000 + c * 100 + b * 10 + 1) : 
  a = 5 ∧ b = 3 ∧ c = 7 := 
sorry

end multiplication_problem_solution_l708_70887


namespace john_can_fix_l708_70886

variable (total_computers : ℕ) (percent_unfixable percent_wait_for_parts : ℕ)

-- Conditions as requirements
def john_condition : Prop :=
  total_computers = 20 ∧
  percent_unfixable = 20 ∧
  percent_wait_for_parts = 40

-- The proof goal based on the conditions
theorem john_can_fix (h : john_condition total_computers percent_unfixable percent_wait_for_parts) :
  total_computers * (100 - percent_unfixable - percent_wait_for_parts) / 100 = 8 :=
by {
  -- Here you can place the corresponding proof details
  sorry
}

end john_can_fix_l708_70886


namespace find_n_eq_l708_70898

theorem find_n_eq : 
  let a := 2^4
  let b := 3^3
  ∃ (n : ℤ), a - 7 = b + n :=
by
  let a := 2^4
  let b := 3^3
  use -18
  sorry

end find_n_eq_l708_70898


namespace least_number_of_cans_l708_70880

theorem least_number_of_cans (maaza pepsi sprite : ℕ) (h_maaza : maaza = 80) (h_pepsi : pepsi = 144) (h_sprite : sprite = 368) :
  ∃ n, n = 37 := sorry

end least_number_of_cans_l708_70880


namespace total_tiles_l708_70829

-- Define the dimensions
def length : ℕ := 16
def width : ℕ := 12

-- Define the number of 1-foot by 1-foot tiles for the border
def tiles_border : ℕ := (2 * length + 2 * width - 4)

-- Define the inner dimensions
def inner_length : ℕ := length - 2
def inner_width : ℕ := width - 2

-- Define the number of 2-foot by 2-foot tiles for the interior
def tiles_interior : ℕ := (inner_length * inner_width) / 4

-- Prove that the total number of tiles is 87
theorem total_tiles : tiles_border + tiles_interior = 87 := by
  sorry

end total_tiles_l708_70829


namespace simplify_and_evaluate_l708_70843

variable (x y : ℚ)
variable (expr : ℚ := 3 * x * y^2 - (x * y - 2 * (2 * x * y - 3 / 2 * x^2 * y) + 3 * x * y^2) + 3 * x^2 * y)

theorem simplify_and_evaluate (h1 : x = 3) (h2 : y = -1 / 3) : expr = -3 :=
by
  sorry

end simplify_and_evaluate_l708_70843


namespace x_pow_y_equals_nine_l708_70850

theorem x_pow_y_equals_nine (x y : ℝ) (h : (|x + 3| * (y - 2)^2 < 0)) : x^y = 9 :=
sorry

end x_pow_y_equals_nine_l708_70850


namespace factor_x4_minus_81_l708_70866

theorem factor_x4_minus_81 (x : ℝ) : 
  x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
sorry

end factor_x4_minus_81_l708_70866


namespace proof_x_exists_l708_70862

noncomputable def find_x : ℝ := 33.33

theorem proof_x_exists (A B C : ℝ) (h1 : A = (1 + find_x / 100) * B) (h2 : C = 0.75 * A) (h3 : A > C) (h4 : C > B) :
  find_x = 33.33 := 
by
  -- Proof steps
  sorry

end proof_x_exists_l708_70862


namespace solve_for_x_l708_70844

theorem solve_for_x (x : ℝ) : 5 + 3.4 * x = 2.1 * x - 30 → x = -26.923 := 
by 
  sorry

end solve_for_x_l708_70844


namespace percentage_exceeds_l708_70800

theorem percentage_exceeds (x y : ℝ) (h₁ : x < y) (h₂ : y = x + 0.35 * x) : ((y - x) / x) * 100 = 35 :=
by sorry

end percentage_exceeds_l708_70800


namespace combined_age_l708_70899

-- Define the conditions as Lean assumptions
def avg_age_three_years_ago := 19
def number_of_original_members := 6
def number_of_years_passed := 3
def current_avg_age := 19

-- Calculate the total age three years ago
def total_age_three_years_ago := number_of_original_members * avg_age_three_years_ago 

-- Calculate the increase in total age over three years
def total_increase_in_age := number_of_original_members * number_of_years_passed 

-- Calculate the current total age of the original members
def current_total_age_of_original_members := total_age_three_years_ago + total_increase_in_age

-- Define the number of current total members and the current total age
def number_of_current_members := 8
def current_total_age := number_of_current_members * current_avg_age

-- Formally state the problem and proof
theorem combined_age : 
  (current_total_age - current_total_age_of_original_members = 20) := 
by
  sorry

end combined_age_l708_70899


namespace root_value_algebraic_expression_l708_70867

theorem root_value_algebraic_expression {a : ℝ} (h : a^2 + 3 * a + 2 = 0) : a^2 + 3 * a = -2 :=
by
  sorry

end root_value_algebraic_expression_l708_70867


namespace find_b_l708_70890

theorem find_b (b : ℝ) (h1 : 0 < b) (h2 : b < 6)
  (h_ratio : ∃ (QRS QOP : ℝ), QRS / QOP = 4 / 25) : b = 6 :=
sorry

end find_b_l708_70890


namespace pills_per_day_l708_70816

theorem pills_per_day (total_days : ℕ) (prescription_days_frac : ℚ) (remaining_pills : ℕ) (days_taken : ℕ) (remaining_days : ℕ) (pills_per_day : ℕ)
  (h1 : total_days = 30)
  (h2 : prescription_days_frac = 4/5)
  (h3 : remaining_pills = 12)
  (h4 : days_taken = prescription_days_frac * total_days)
  (h5 : remaining_days = total_days - days_taken)
  (h6 : pills_per_day = remaining_pills / remaining_days) :
  pills_per_day = 2 := by
  sorry

end pills_per_day_l708_70816


namespace expression_evaluation_l708_70883

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 2) / x) * ((y^2 + 2) / y) + ((x^2 - 2) / y) * ((y^2 - 2) / x) = 2 * x * y + 8 / (x * y) :=
by
  sorry

end expression_evaluation_l708_70883


namespace parabola_passes_through_fixed_point_l708_70889

theorem parabola_passes_through_fixed_point:
  ∀ t : ℝ, ∃ x y : ℝ, (y = 4 * x^2 + 2 * t * x - 3 * t ∧ (x = 3 ∧ y = 36)) :=
by
  intro t
  use 3
  use 36
  sorry

end parabola_passes_through_fixed_point_l708_70889


namespace product_of_consecutive_integers_plus_one_l708_70825

theorem product_of_consecutive_integers_plus_one (n : ℤ) : n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1) ^ 2 := 
sorry

end product_of_consecutive_integers_plus_one_l708_70825


namespace sunflower_seeds_contest_l708_70869

theorem sunflower_seeds_contest 
  (first_player_seeds : ℕ) (second_player_seeds : ℕ) (total_seeds : ℕ) 
  (third_player_seeds : ℕ) (third_more : ℕ) 
  (h1 : first_player_seeds = 78) 
  (h2 : second_player_seeds = 53) 
  (h3 : total_seeds = 214) 
  (h4 : first_player_seeds + second_player_seeds + third_player_seeds = total_seeds) 
  (h5 : third_more = third_player_seeds - second_player_seeds) : 
  third_more = 30 :=
by
  sorry

end sunflower_seeds_contest_l708_70869


namespace probability_at_least_four_same_face_l708_70863

-- Define the total number of outcomes for flipping five coins
def total_outcomes : ℕ := 2^5

-- Define the number of favorable outcomes where at least four coins show the same face
def favorable_outcomes : ℕ := 2 + 5 + 5

-- Define the probability of getting at least four heads or four tails out of five flips
def probability : ℚ := favorable_outcomes / total_outcomes

-- Theorem statement to prove the probability calculation
theorem probability_at_least_four_same_face : 
  probability = 3 / 8 :=
by
  -- Placeholder for the proof
  sorry

end probability_at_least_four_same_face_l708_70863


namespace squares_in_50th_ring_l708_70895

noncomputable def number_of_squares_in_nth_ring (n : ℕ) : ℕ :=
  8 * n + 6

theorem squares_in_50th_ring : number_of_squares_in_nth_ring 50 = 406 := 
  by
  sorry

end squares_in_50th_ring_l708_70895


namespace fraction_equality_l708_70878

theorem fraction_equality (p q x y : ℚ) (hpq : p / q = 4 / 5) (hx : x / y + (2 * q - p) / (2 * q + p) = 1) :
  x / y = 4 / 7 :=
by {
  sorry
}

end fraction_equality_l708_70878


namespace min_value_3x_plus_4y_l708_70881

theorem min_value_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 :=
sorry

end min_value_3x_plus_4y_l708_70881


namespace ants_in_park_l708_70876

theorem ants_in_park:
  let width_meters := 100
  let length_meters := 130
  let cm_per_meter := 100
  let ants_per_sq_cm := 1.2
  let width_cm := width_meters * cm_per_meter
  let length_cm := length_meters * cm_per_meter
  let area_sq_cm := width_cm * length_cm
  let total_ants := ants_per_sq_cm * area_sq_cm
  total_ants = 156000000 := by
  sorry

end ants_in_park_l708_70876


namespace total_selling_price_is_correct_l708_70868

-- Define the given constants
def meters_of_cloth : ℕ := 85
def profit_per_meter : ℕ := 10
def cost_price_per_meter : ℕ := 95

-- Compute the selling price per meter
def selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter

-- Calculate the total selling price
def total_selling_price : ℕ := selling_price_per_meter * meters_of_cloth

-- The theorem statement
theorem total_selling_price_is_correct : total_selling_price = 8925 := by
  sorry

end total_selling_price_is_correct_l708_70868


namespace david_marks_physics_l708_70804

def marks_english := 96
def marks_math := 95
def marks_chemistry := 97
def marks_biology := 95
def average_marks := 93
def number_of_subjects := 5

theorem david_marks_physics : 
  let total_marks := average_marks * number_of_subjects 
  let total_known_marks := marks_english + marks_math + marks_chemistry + marks_biology
  let marks_physics := total_marks - total_known_marks
  marks_physics = 82 :=
by
  sorry

end david_marks_physics_l708_70804


namespace real_set_x_eq_l708_70812

theorem real_set_x_eq :
  {x : ℝ | ⌊x * ⌊x⌋⌋ = 45} = {x : ℝ | 7.5 ≤ x ∧ x < 7.6667} :=
by
  -- The proof would be provided here, but we're skipping it with sorry
  sorry

end real_set_x_eq_l708_70812


namespace find_white_towels_l708_70845

variable (W : ℕ) -- Let W be the number of white towels Maria bought

def green_towels : ℕ := 40
def towels_given : ℕ := 65
def towels_left : ℕ := 19

theorem find_white_towels :
  green_towels + W - towels_given = towels_left →
  W = 44 :=
by
  intro h
  sorry

end find_white_towels_l708_70845


namespace evaluate_f_at_3_l708_70833

def f (x : ℤ) : ℤ := 5 * x^3 + 3 * x^2 + 7 * x - 2

theorem evaluate_f_at_3 : f 3 = 181 := by
  sorry

end evaluate_f_at_3_l708_70833


namespace Alan_eggs_count_l708_70854

theorem Alan_eggs_count (Price_per_egg Chickens_bought Price_per_chicken Total_spent : ℕ)
  (h1 : Price_per_egg = 2) (h2 : Chickens_bought = 6) (h3 : Price_per_chicken = 8) (h4 : Total_spent = 88) :
  ∃ E : ℕ, 2 * E + Chickens_bought * Price_per_chicken = Total_spent ∧ E = 20 :=
by
  sorry

end Alan_eggs_count_l708_70854


namespace integer_add_results_in_perfect_square_l708_70860

theorem integer_add_results_in_perfect_square (x a b : ℤ) :
  (x + 100 = a^2 ∧ x + 164 = b^2) → (x = 125 ∨ x = -64 ∨ x = -100) :=
by
  intros h
  sorry

end integer_add_results_in_perfect_square_l708_70860


namespace divisibility_by_100_l708_70891

theorem divisibility_by_100 (n : ℕ) (k : ℕ) (h : n = 5 * k + 2) :
    100 ∣ (5^n + 12*n^2 + 12*n + 3) :=
sorry

end divisibility_by_100_l708_70891


namespace range_of_a_l708_70861

noncomputable def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

theorem range_of_a (a : ℝ) : (A a ∪ B a = Set.univ) → a ∈ Set.Iic 2 := by
  intro h
  sorry

end range_of_a_l708_70861


namespace goal_amount_is_correct_l708_70865

def earnings_three_families : ℕ := 3 * 10
def earnings_fifteen_families : ℕ := 15 * 5
def total_earned : ℕ := earnings_three_families + earnings_fifteen_families
def goal_amount : ℕ := total_earned + 45

theorem goal_amount_is_correct : goal_amount = 150 :=
by
  -- We are aware of the proof steps but they are not required here
  sorry

end goal_amount_is_correct_l708_70865


namespace houses_per_block_correct_l708_70822

-- Define the conditions
def total_mail_per_block : ℕ := 32
def mail_per_house : ℕ := 8

-- Define the correct answer
def houses_per_block : ℕ := 4

-- Theorem statement
theorem houses_per_block_correct (total_mail_per_block mail_per_house : ℕ) : 
  total_mail_per_block = 32 →
  mail_per_house = 8 →
  total_mail_per_block / mail_per_house = houses_per_block :=
by
  intros h1 h2
  sorry

end houses_per_block_correct_l708_70822


namespace juliette_and_marco_money_comparison_l708_70855

noncomputable def euro_to_dollar (eur : ℝ) : ℝ := eur * 1.5

theorem juliette_and_marco_money_comparison :
  (600 - euro_to_dollar 350) / 600 * 100 = 12.5 := by
sorry

end juliette_and_marco_money_comparison_l708_70855


namespace steps_probability_to_point_3_3_l708_70814

theorem steps_probability_to_point_3_3 : 
  let a := 35
  let b := 4096
  a + b = 4131 :=
by {
  sorry
}

end steps_probability_to_point_3_3_l708_70814


namespace min_rho_squared_l708_70818

noncomputable def rho_squared (x t : ℝ) : ℝ :=
  (x - t)^2 + (x^2 - 4 * x + 7 + t)^2

theorem min_rho_squared : 
  ∃ (x t : ℝ), x = 3/2 ∧ t = -7/8 ∧ 
  ∀ (x' t' : ℝ), rho_squared x' t' ≥ rho_squared (3/2) (-7/8) :=
by
  sorry

end min_rho_squared_l708_70818


namespace f_of_2_l708_70846

-- Definition of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Given conditions
variable (f : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_value : f (-2) = 11)

-- The theorem we want to prove
theorem f_of_2 : f 2 = -11 :=
by 
  sorry

end f_of_2_l708_70846


namespace Krishan_has_4046_l708_70877

variable (Ram Gopal Krishan : ℕ) -- Define the variables

-- Conditions given in the problem
axiom ratio_Ram_Gopal : Ram * 17 = Gopal * 7
axiom ratio_Gopal_Krishan : Gopal * 17 = Krishan * 7
axiom Ram_value : Ram = 686

-- This is the goal to prove
theorem Krishan_has_4046 : Krishan = 4046 :=
by
  -- Here is where the proof would go
  sorry

end Krishan_has_4046_l708_70877


namespace chef_earns_less_than_manager_l708_70808

noncomputable def manager_wage : ℝ := 8.50
noncomputable def dishwasher_wage : ℝ := manager_wage / 2
noncomputable def chef_wage : ℝ := dishwasher_wage * 1.22

theorem chef_earns_less_than_manager :
  manager_wage - chef_wage = 3.315 :=
by
  sorry

end chef_earns_less_than_manager_l708_70808


namespace ben_chairs_in_10_days_l708_70835

def number_of_chairs (days hours_per_shift hours_rocking_chair hours_dining_chair hours_armchair : ℕ) : ℕ × ℕ × ℕ :=
  let rocking_chairs_per_day := hours_per_shift / hours_rocking_chair
  let remaining_hours_after_rocking_chairs := hours_per_shift % hours_rocking_chair
  let dining_chairs_per_day := remaining_hours_after_rocking_chairs / hours_dining_chair
  let remaining_hours_after_dining_chairs := remaining_hours_after_rocking_chairs % hours_dining_chair
  if remaining_hours_after_dining_chairs >= hours_armchair then
    (days * rocking_chairs_per_day, days * dining_chairs_per_day, days * (remaining_hours_after_dining_chairs / hours_armchair))
  else
    (days * rocking_chairs_per_day, days * dining_chairs_per_day, 0)

theorem ben_chairs_in_10_days :
  number_of_chairs 10 8 5 3 6 = (10, 10, 0) :=
by 
  sorry

end ben_chairs_in_10_days_l708_70835


namespace number_of_pickers_is_221_l708_70824
-- Import necessary Lean and math libraries

/--
Given the conditions:
1. The number of pickers fills 100 drums of raspberries per day.
2. The number of pickers fills 221 drums of grapes per day.
3. In 77 days, the pickers would fill 17017 drums of grapes.
Prove that the number of pickers is 221.
-/
theorem number_of_pickers_is_221
  (P : ℕ)
  (d1 : P * 100 = 100 * P)
  (d2 : P * 221 = 221 * P)
  (d17 : P * 221 * 77 = 17017) : 
  P = 221 := 
sorry

end number_of_pickers_is_221_l708_70824


namespace complement_of_A_in_U_l708_70875

open Set

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}
def complementA : Set ℕ := {2, 4}

theorem complement_of_A_in_U :
  (U \ A) = complementA :=
by
  sorry

end complement_of_A_in_U_l708_70875


namespace triangle_cosine_sine_inequality_l708_70828

theorem triangle_cosine_sine_inequality (A B C : ℝ) (h : A + B + C = Real.pi) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hA_lt_pi : A < Real.pi)
  (hB_lt_pi : B < Real.pi)
  (hC_lt_pi : C < Real.pi) :
  Real.cos A * (Real.sin B + Real.sin C) ≥ -2 * Real.sqrt 6 / 9 := 
by
  sorry

end triangle_cosine_sine_inequality_l708_70828


namespace linda_original_savings_l708_70802

theorem linda_original_savings :
  ∃ S : ℝ, 
    (5 / 8) * S + (1 / 4) * S = 400 ∧
    (1 / 8) * S = 600 ∧
    S = 4800 :=
by
  sorry

end linda_original_savings_l708_70802


namespace probability_snow_once_first_week_l708_70853

theorem probability_snow_once_first_week :
  let p_first_two_days := (3 / 4) * (3 / 4)
  let p_next_three_days := (1 / 2) * (1 / 2) * (1 / 2)
  let p_last_two_days := (2 / 3) * (2 / 3)
  let p_no_snow := p_first_two_days * p_next_three_days * p_last_two_days
  let p_at_least_once := 1 - p_no_snow
  p_at_least_once = 31 / 32 :=
by
  sorry

end probability_snow_once_first_week_l708_70853


namespace annual_increase_of_chickens_l708_70847

theorem annual_increase_of_chickens 
  (chickens_now : ℕ)
  (chickens_after_9_years : ℕ)
  (years : ℕ)
  (chickens_now_eq : chickens_now = 550)
  (chickens_after_9_years_eq : chickens_after_9_years = 1900)
  (years_eq : years = 9)
  : ((chickens_after_9_years - chickens_now) / years) = 150 :=
by
  sorry

end annual_increase_of_chickens_l708_70847


namespace max_value_f_on_interval_l708_70837

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  (x^2 - 4) * (x - a)

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ :=
  3 * x^2 - 2 * a * x - 4

theorem max_value_f_on_interval :
  f' (-1) (1 / 2) = 0 →
  ∃ max_f, max_f = 42 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 4, f x (1 / 2) ≤ max_f :=
by
  sorry

end max_value_f_on_interval_l708_70837


namespace roots_polynomial_sum_products_l708_70815

theorem roots_polynomial_sum_products (p q r : ℂ)
  (h : 6 * p^3 - 5 * p^2 + 13 * p - 10 = 0)
  (h' : 6 * q^3 - 5 * q^2 + 13 * q - 10 = 0)
  (h'' : 6 * r^3 - 5 * r^2 + 13 * r - 10 = 0)
  (hpq : p ≠ q) (hqr : q ≠ r) (hrp : r ≠ p) :
  p * q + q * r + r * p = 13 / 6 := 
sorry

end roots_polynomial_sum_products_l708_70815


namespace brick_width_l708_70809

-- Define the dimensions of the wall
def L_wall : Real := 750 -- length in cm
def W_wall : Real := 600 -- width in cm
def H_wall : Real := 22.5 -- height in cm

-- Define the dimensions of the bricks
def L_brick : Real := 25 -- length in cm
def H_brick : Real := 6 -- height in cm

-- Define the number of bricks needed
def n_bricks : Nat := 6000

-- Define the total volume of the wall
def V_wall : Real := L_wall * W_wall * H_wall

-- Define the volume of one brick
def V_brick (W : Real) : Real := L_brick * W * H_brick

-- Statement to prove
theorem brick_width : 
  ∃ W : Real, V_wall = V_brick W * (n_bricks : Real) ∧ W = 11.25 := by 
  sorry

end brick_width_l708_70809


namespace solve_fraction_equation_l708_70841

theorem solve_fraction_equation (x : ℝ) (h : (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2) : x = -2 / 3 :=
by
  sorry

end solve_fraction_equation_l708_70841


namespace eq_a_sub_b_l708_70830

theorem eq_a_sub_b (a b : ℝ) (i : ℂ) (hi : i * i = -1) (h1 : (a + 4 * i) * i = b + i) : a - b = 5 :=
by
  have := hi
  have := h1
  sorry

end eq_a_sub_b_l708_70830


namespace smallest_k_l708_70859

theorem smallest_k (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ k : ℕ, k = 2 * max m n + min m n - 1 ∧ 
  (∀ (persons : Finset ℕ),
    persons.card ≥ k →
    (∃ (acquainted : Finset (ℕ × ℕ)), acquainted.card = m ∧ 
      (∀ (x y : ℕ), (x, y) ∈ acquainted → (x ∈ persons ∧ y ∈ persons))) ∨
    (∃ (unacquainted : Finset (ℕ × ℕ)), unacquainted.card = n ∧ 
      (∀ (x y : ℕ), (x, y) ∈ unacquainted → (x ∈ persons ∧ y ∈ persons ∧ x ≠ y)))) :=
sorry

end smallest_k_l708_70859


namespace sqrt_fraction_difference_l708_70873

theorem sqrt_fraction_difference : 
  (Real.sqrt (16 / 9) - Real.sqrt (9 / 16)) = 7 / 12 :=
by
  sorry

end sqrt_fraction_difference_l708_70873


namespace compute_100a_b_l708_70819

theorem compute_100a_b (a b : ℝ) 
  (h1 : ∀ x : ℝ, (x + a) * (x + b) * (x + 10) = 0 ↔ x = -a ∨ x = -b ∨ x = -10)
  (h2 : a ≠ -4 ∧ b ≠ -4 ∧ 10 ≠ -4)
  (h3 : ∀ x : ℝ, (x + 2 * a) * (x + 5) * (x + 8) = 0 ↔ x = -5)
  (hb : b = 8)
  (ha : 2 * a = 5) :
  100 * a + b = 258 := 
sorry

end compute_100a_b_l708_70819


namespace tree_planting_growth_rate_l708_70831

theorem tree_planting_growth_rate {x : ℝ} :
  400 * (1 + x) ^ 2 = 625 :=
sorry

end tree_planting_growth_rate_l708_70831


namespace incorrect_solution_among_four_l708_70805

theorem incorrect_solution_among_four 
  (x y : ℤ) 
  (h1 : 2 * x - 3 * y = 5) 
  (h2 : 3 * x - 2 * y = 7) : 
  ¬ ((2 * (2 * x - 3 * y) - ((-3) * (3 * x - 2 * y))) = (2 * 5 - (-3) * 7)) :=
sorry

end incorrect_solution_among_four_l708_70805


namespace tan_of_trig_eq_l708_70882

theorem tan_of_trig_eq (x : Real) (h : (1 - Real.cos x + Real.sin x) / (1 + Real.cos x + Real.sin x) = -2) : Real.tan x = 4 / 3 :=
by sorry

end tan_of_trig_eq_l708_70882


namespace calculate_expression_l708_70858

theorem calculate_expression : 
  - 3 ^ 2 + (-12) * abs (-1/2) - 6 / (-1) = -9 := 
by 
  sorry

end calculate_expression_l708_70858


namespace man_age_twice_son_age_l708_70811

-- Definitions based on conditions
def son_age : ℕ := 20
def man_age : ℕ := son_age + 22

-- Definition of the main statement to be proven
theorem man_age_twice_son_age (Y : ℕ) : man_age + Y = 2 * (son_age + Y) → Y = 2 :=
by sorry

end man_age_twice_son_age_l708_70811


namespace min_digits_fraction_l708_70820

theorem min_digits_fraction : 
  let num := 987654321
  let denom := 2^27 * 5^3
  ∃ (digits : ℕ), (10^digits * num = 987654321 * 2^27 * 5^3) ∧ digits = 27 := 
by
  sorry

end min_digits_fraction_l708_70820


namespace count_valid_ways_l708_70851

theorem count_valid_ways (n : ℕ) (h1 : n = 6) : 
  ∀ (library : ℕ), (1 ≤ library) → (library ≤ 5) → ∃ (checked_out : ℕ), 
  (checked_out = n - library) := 
sorry

end count_valid_ways_l708_70851


namespace problem_l708_70801

def P (x : ℝ) : Prop := x^2 - 2*x + 1 > 0

theorem problem (h : ¬ ∀ x : ℝ, P x) : ∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0 :=
by {
  sorry
}

end problem_l708_70801


namespace distance_p_ran_l708_70894

variable (d t v : ℝ)
-- d: head start distance in meters
-- t: time in minutes
-- v: speed of q in meters per minute

theorem distance_p_ran (h1 : d = 0.3 * v * t) : 1.3 * v * t = 1.3 * v * t :=
by
  sorry

end distance_p_ran_l708_70894


namespace peanut_mixture_l708_70842

-- Definitions of given conditions
def virginia_peanuts_weight : ℝ := 10
def virginia_peanuts_cost_per_pound : ℝ := 3.50
def spanish_peanuts_cost_per_pound : ℝ := 3.00
def texan_peanuts_cost_per_pound : ℝ := 4.00
def desired_cost_per_pound : ℝ := 3.60

-- Definitions of unknowns S (Spanish peanuts) and T (Texan peanuts)
variable (S T : ℝ)

-- Equation derived from given conditions
theorem peanut_mixture :
  (0.40 * T) - (0.60 * S) = 1 := sorry

end peanut_mixture_l708_70842


namespace B_subset_A_l708_70803

def A (x : ℝ) : Prop := abs (2 * x - 3) > 1
def B (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem B_subset_A : ∀ x, B x → A x := sorry

end B_subset_A_l708_70803


namespace simplify_expression_l708_70857

theorem simplify_expression : |(-4 : Int)^2 - (3 : Int)^2 + 2| = 9 := by
  sorry

end simplify_expression_l708_70857


namespace sqrt_sum_abs_eq_l708_70832

theorem sqrt_sum_abs_eq (x : ℝ) :
    (Real.sqrt (x^2 + 6 * x + 9) + Real.sqrt (x^2 - 6 * x + 9)) = (|x - 3| + |x + 3|) := 
by 
  sorry

end sqrt_sum_abs_eq_l708_70832


namespace M_subset_N_l708_70836

open Set

noncomputable def M : Set ℚ := {x | ∃ k : ℤ, x = k / 2 + 1 / 4}
noncomputable def N : Set ℚ := {x | ∃ k : ℤ, x = k / 4 + 1 / 2}

theorem M_subset_N : M ⊆ N := 
sorry

end M_subset_N_l708_70836


namespace jake_steps_per_second_l708_70821

/-
Conditions:
1. Austin and Jake start descending from the 9th floor at the same time.
2. The stairs have 30 steps across each floor.
3. The elevator takes 1 minute (60 seconds) to reach the ground floor.
4. Jake reaches the ground floor 30 seconds after Austin.
5. Jake descends 8 floors to reach the ground floor.
-/

def floors : ℕ := 8
def steps_per_floor : ℕ := 30
def time_elevator : ℕ := 60 -- in seconds
def additional_time_jake : ℕ := 30 -- in seconds

def total_time_jake := time_elevator + additional_time_jake -- in seconds
def total_steps := floors * steps_per_floor

def steps_per_second_jake := (total_steps : ℚ) / (total_time_jake : ℚ)

theorem jake_steps_per_second :
  steps_per_second_jake = 2.67 := by
  sorry

end jake_steps_per_second_l708_70821


namespace range_of_a_l708_70893

noncomputable def domain_f (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*x + a ≥ 0
noncomputable def range_g (a : ℝ) : Prop := ∀ x : ℝ, x ≤ 2 → 2^x - a ∈ Set.Ioi (0 : ℝ)

theorem range_of_a (a : ℝ) : (domain_f a ∨ range_g a) ∧ ¬(domain_f a ∧ range_g a) → (a ≥ 1 ∨ a ≤ 0) := by
  sorry

end range_of_a_l708_70893


namespace minimum_distance_from_parabola_to_circle_l708_70872

noncomputable def minimum_distance_sum : ℝ :=
  let focus : ℝ × ℝ := (1, 0)
  let center : ℝ × ℝ := (0, 4)
  let radius : ℝ := 1
  let distance_from_focus_to_center : ℝ := Real.sqrt ((focus.1 - center.1)^2 + (focus.2 - center.2)^2)
  distance_from_focus_to_center - radius

theorem minimum_distance_from_parabola_to_circle : minimum_distance_sum = Real.sqrt 17 - 1 := by
  sorry

end minimum_distance_from_parabola_to_circle_l708_70872


namespace eagles_score_l708_70892

variables (F E : ℕ)

theorem eagles_score (h1 : F + E = 56) (h2 : F = E + 8) : E = 24 := 
sorry

end eagles_score_l708_70892


namespace infinite_geometric_subsequence_exists_l708_70840

theorem infinite_geometric_subsequence_exists
  (a : ℕ) (d : ℕ) (h_d_pos : d > 0)
  (a_n : ℕ → ℕ)
  (h_arith_prog : ∀ n, a_n n = a + n * d) :
  ∃ (g : ℕ → ℕ), (∀ m n, m < n → g m < g n) ∧ (∃ r : ℕ, ∀ n, g (n+1) = g n * r) ∧ (∀ n, ∃ m, a_n m = g n) :=
sorry

end infinite_geometric_subsequence_exists_l708_70840


namespace compound_interest_years_is_four_l708_70870
noncomputable def compoundInterestYears (P : ℝ) (r : ℝ) (n : ℕ) (CI : ℝ) : ℕ :=
  let A := P + CI
  let factor := (1 + r / n)
  let log_A_P := Real.log (A / P)
  let log_factor := Real.log factor
  Nat.floor (log_A_P / log_factor)

theorem compound_interest_years_is_four :
  compoundInterestYears 1200 0.20 1 1288.32 = 4 :=
by
  sorry

end compound_interest_years_is_four_l708_70870


namespace rook_placement_l708_70884

theorem rook_placement : 
  let n := 8
  let k := 6
  let binom := Nat.choose
  binom 8 6 * binom 8 6 * Nat.factorial 6 = 564480 := by
    sorry

end rook_placement_l708_70884


namespace compute_ab_val_l708_70888

variables (a b : ℝ)

theorem compute_ab_val
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 64) :
  |a * b| = Real.sqrt 868.5 :=
sorry

end compute_ab_val_l708_70888


namespace available_milk_for_me_l708_70852

def initial_milk_litres : ℝ := 1
def myeongseok_milk_litres : ℝ := 0.1
def mingu_milk_litres : ℝ := myeongseok_milk_litres + 0.2
def minjae_milk_litres : ℝ := 0.3

theorem available_milk_for_me :
  initial_milk_litres - (myeongseok_milk_litres + mingu_milk_litres + minjae_milk_litres) = 0.3 :=
by sorry

end available_milk_for_me_l708_70852


namespace second_smallest_packs_of_hot_dogs_l708_70826

theorem second_smallest_packs_of_hot_dogs 
  (n : ℕ) 
  (h1 : ∃ (k : ℕ), n = 2 * k + 2)
  (h2 : 12 * n ≡ 6 [MOD 8]) : 
  n = 4 :=
by
  sorry

end second_smallest_packs_of_hot_dogs_l708_70826


namespace max_markers_with_20_dollars_l708_70897

theorem max_markers_with_20_dollars (single_marker_cost : ℕ) (four_pack_cost : ℕ) (eight_pack_cost : ℕ) :
  single_marker_cost = 2 → four_pack_cost = 6 → eight_pack_cost = 10 → (∃ n, n = 16) := by
    intros h1 h2 h3
    existsi 16
    sorry

end max_markers_with_20_dollars_l708_70897


namespace cost_difference_l708_70864

-- Given conditions
def first_present_cost : ℕ := 18
def third_present_cost : ℕ := first_present_cost - 11
def total_cost : ℕ := 50

-- denoting costs of the second present via variable
def second_present_cost (x : ℕ) : Prop :=
  first_present_cost + x + third_present_cost = total_cost

-- Goal statement
theorem cost_difference (x : ℕ) (h : second_present_cost x) : x - first_present_cost = 7 :=
  sorry

end cost_difference_l708_70864


namespace calc_molecular_weight_l708_70838

/-- Atomic weights in g/mol -/
def atomic_weight (e : String) : Float :=
  match e with
  | "Ca"   => 40.08
  | "O"    => 16.00
  | "H"    => 1.01
  | "Al"   => 26.98
  | "S"    => 32.07
  | "K"    => 39.10
  | "N"    => 14.01
  | _      => 0.0

/-- Molecular weight calculation for specific compounds -/
def molecular_weight (compound : String) : Float :=
  match compound with
  | "Ca(OH)2"     => atomic_weight "Ca" + 2 * atomic_weight "O" + 2 * atomic_weight "H"
  | "Al2(SO4)3"   => 2 * atomic_weight "Al" + 3 * (atomic_weight "S" + 4 * atomic_weight "O")
  | "KNO3"        => atomic_weight "K" + atomic_weight "N" + 3 * atomic_weight "O"
  | _             => 0.0

/-- Given moles of different compounds, calculate the total molecular weight -/
def total_molecular_weight (moles : List (String × Float)) : Float :=
  moles.foldl (fun acc (compound, n) => acc + n * molecular_weight compound) 0.0

/-- The given problem -/
theorem calc_molecular_weight :
  total_molecular_weight [("Ca(OH)2", 4), ("Al2(SO4)3", 2), ("KNO3", 3)] = 1284.07 :=
by
  sorry

end calc_molecular_weight_l708_70838


namespace no_distinct_triple_exists_for_any_quadratic_trinomial_l708_70874

theorem no_distinct_triple_exists_for_any_quadratic_trinomial (f : ℝ → ℝ) 
    (hf : ∃ a b c : ℝ, ∀ x, f x = a*x^2 + b*x + c) :
    ¬ ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ f a = b ∧ f b = c ∧ f c = a := 
by 
  sorry

end no_distinct_triple_exists_for_any_quadratic_trinomial_l708_70874


namespace range_of_a_l708_70810

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) → (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l708_70810


namespace players_without_cautions_l708_70834

theorem players_without_cautions (Y N : ℕ) (h1 : Y + N = 11) (h2 : Y = 6) : N = 5 :=
by
  sorry

end players_without_cautions_l708_70834


namespace math_proof_problem_l708_70817

noncomputable def problemStatement : Prop :=
  ∃ (α : ℝ), 5 * Real.sin (2 * α) = Real.sin (2 * Real.pi / 180) ∧ 
  (Real.tan (α + Real.pi / 180) / Real.tan (α - Real.pi / 180) = -3/2)

theorem math_proof_problem : problemStatement := 
by 
  sorry

end math_proof_problem_l708_70817


namespace correct_proposition_l708_70823

-- Definitions based on conditions
def p : Prop := ∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0
def not_p : Prop := ∀ x : ℝ, x^2 + 2 * x + 2016 > 0

-- Proof statement
theorem correct_proposition : p → not_p :=
by sorry

end correct_proposition_l708_70823


namespace triangle_BC_range_l708_70896

open Real

variable {a C : ℝ} (A : ℝ) (ABC : Triangle A C)

/-- Proof problem statement -/
theorem triangle_BC_range (A C : ℝ) (h0 : 0 < A) (h1 : A < π) (c : ℝ) (h2 : c = sqrt 2) (h3 : a * cos C = c * sin A): 
  ∃ (BC : ℝ), sqrt 2 < BC ∧ BC < 2 :=
sorry

end triangle_BC_range_l708_70896
