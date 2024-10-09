import Mathlib

namespace fred_found_28_more_seashells_l71_7108

theorem fred_found_28_more_seashells (tom_seashells : ℕ) (fred_seashells : ℕ) (h_tom : tom_seashells = 15) (h_fred : fred_seashells = 43) : 
  fred_seashells - tom_seashells = 28 := 
by 
  sorry

end fred_found_28_more_seashells_l71_7108


namespace triangle_even_number_in_each_row_from_third_l71_7194

/-- Each number in the (n+1)-th row of the triangle is the sum of three numbers 
  from the n-th row directly above this number and its immediate left and right neighbors.
  If such neighbors do not exist, they are considered as zeros.
  Prove that in each row of the triangle, starting from the third row,
  there is at least one even number. -/

theorem triangle_even_number_in_each_row_from_third (triangle : ℕ → ℕ → ℕ) :
  (∀ n i : ℕ, i > n → triangle n i = 0) →
  (∀ n i : ℕ, triangle (n+1) i = triangle n (i-1) + triangle n i + triangle n (i+1)) →
  ∀ n : ℕ, n ≥ 2 → ∃ i : ℕ, i ≤ n ∧ 2 ∣ triangle n i :=
by
  intros
  sorry

end triangle_even_number_in_each_row_from_third_l71_7194


namespace remaining_cube_height_l71_7184

/-- Given a cube with side length 2 units, where a corner is chopped off such that the cut runs
    through points on the three edges adjacent to a selected vertex, each at 1 unit distance
    from that vertex, the height of the remaining portion of the cube when the freshly cut face 
    is placed on a table is equal to (5 * sqrt 3) / 3. -/
theorem remaining_cube_height (s : ℝ) (h : ℝ) : 
    s = 2 → h = 1 → 
    ∃ height : ℝ, height = (5 * Real.sqrt 3) / 3 := 
by
    sorry

end remaining_cube_height_l71_7184


namespace arcsin_neg_one_half_l71_7155

theorem arcsin_neg_one_half : Real.arcsin (-1 / 2) = -Real.pi / 6 :=
by
  sorry

end arcsin_neg_one_half_l71_7155


namespace matchstick_ratio_is_one_half_l71_7188

def matchsticks_used (houses : ℕ) (matchsticks_per_house : ℕ) : ℕ :=
  houses * matchsticks_per_house

def ratio (a b : ℕ) : ℚ := a / b

def michael_original_matchsticks : ℕ := 600
def michael_houses : ℕ := 30
def matchsticks_per_house : ℕ := 10
def michael_used_matchsticks : ℕ := matchsticks_used michael_houses matchsticks_per_house

theorem matchstick_ratio_is_one_half :
  ratio michael_used_matchsticks michael_original_matchsticks = 1 / 2 :=
by
  sorry

end matchstick_ratio_is_one_half_l71_7188


namespace find_real_num_l71_7136

noncomputable def com_num (a : ℝ) : ℂ := (a + 3 * Complex.I) / (1 + 2 * Complex.I)

theorem find_real_num (a : ℝ) : (∃ b : ℝ, com_num a = b * Complex.I) → a = -6 :=
by
  sorry

end find_real_num_l71_7136


namespace power_computation_l71_7182

theorem power_computation :
  16^10 * 8^6 / 4^22 = 16384 :=
by
  sorry

end power_computation_l71_7182


namespace simplify_expression_l71_7171

variable (c d : ℝ)
variable (hc : 0 < c)
variable (hd : 0 < d)
variable (h : c^3 + d^3 = 3 * (c + d))

theorem simplify_expression : (c / d) + (d / c) - (3 / (c * d)) = 1 := by
  sorry

end simplify_expression_l71_7171


namespace new_interest_rate_l71_7176

theorem new_interest_rate 
  (initial_interest : ℝ) 
  (additional_interest : ℝ) 
  (initial_rate : ℝ) 
  (time : ℝ) 
  (new_total_interest : ℝ)
  (principal : ℝ)
  (new_rate : ℝ) 
  (h1 : initial_interest = principal * initial_rate * time)
  (h2 : new_total_interest = initial_interest + additional_interest)
  (h3 : new_total_interest = principal * new_rate * time)
  (principal_val : principal = initial_interest / initial_rate) :
  new_rate = 0.05 :=
by
  sorry

end new_interest_rate_l71_7176


namespace fraction_of_y_l71_7154

theorem fraction_of_y (w x y : ℝ) (h1 : wx = y) 
  (h2 : (w + x) / 2 = 0.5) : 
  (2 / w + 2 / x = 2 / y) := 
by
  sorry

end fraction_of_y_l71_7154


namespace overall_average_speed_is_six_l71_7119

-- Definitions of the conditions
def cycling_time := 45 / 60 -- hours
def cycling_speed := 12 -- mph
def stopping_time := 15 / 60 -- hours
def walking_time := 75 / 60 -- hours
def walking_speed := 3 -- mph

-- Problem statement: Proving that the overall average speed is 6 mph
theorem overall_average_speed_is_six : 
  (cycling_speed * cycling_time + walking_speed * walking_time) /
  (cycling_time + walking_time + stopping_time) = 6 :=
by
  sorry

end overall_average_speed_is_six_l71_7119


namespace Q_gets_less_than_P_l71_7166

theorem Q_gets_less_than_P (x : Real) (hx : x > 0) (hP : P = 1.25 * x): 
  Q = P * 0.8 := 
sorry

end Q_gets_less_than_P_l71_7166


namespace gina_can_paint_6_rose_cups_an_hour_l71_7169

def number_of_rose_cups_painted_in_an_hour 
  (R : ℕ) (lily_rate : ℕ) (rose_order : ℕ) (lily_order : ℕ) (total_payment : ℕ) (hourly_rate : ℕ)
  (lily_hours : ℕ) (total_hours : ℕ) (rose_hours : ℕ) : Prop :=
  (lily_rate = 7) ∧
  (rose_order = 6) ∧
  (lily_order = 14) ∧
  (total_payment = 90) ∧
  (hourly_rate = 30) ∧
  (lily_hours = lily_order / lily_rate) ∧
  (total_hours = total_payment / hourly_rate) ∧
  (rose_hours = total_hours - lily_hours) ∧
  (rose_order = R * rose_hours)

theorem gina_can_paint_6_rose_cups_an_hour :
  ∃ R, number_of_rose_cups_painted_in_an_hour 
    R 7 6 14 90 30 (14 / 7) (90 / 30)  (90 / 30 - 14 / 7) ∧ R = 6 :=
by
  -- proof is left out intentionally
  sorry

end gina_can_paint_6_rose_cups_an_hour_l71_7169


namespace usual_time_catch_bus_l71_7120

-- Define the problem context
variable (S T : ℝ)

-- Hypotheses for the conditions given
def condition1 : Prop := S * T = (4 / 5) * S * (T + 4)
def condition2 : Prop := S ≠ 0

-- Theorem that states the fact we need to prove
theorem usual_time_catch_bus (h1 : condition1 S T) (h2 : condition2 S) : T = 16 :=
by
  -- proof omitted
  sorry

end usual_time_catch_bus_l71_7120


namespace smallest_four_digit_divisible_by_primes_l71_7146

theorem smallest_four_digit_divisible_by_primes : 
  let p1 := 2 
  let p2 := 3 
  let p3 := 5 
  let p4 := 7 
  let p5 := 11 
  let lcm_val := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm p1 p2) p3) p4) p5 
  1000 ≤ lcm_val ∧ lcm_val < 10000 :=
  sorry

end smallest_four_digit_divisible_by_primes_l71_7146


namespace number_of_distinct_digit_odd_numbers_l71_7177

theorem number_of_distinct_digit_odd_numbers (a b c d : ℕ) :
  1000 ≤ a * 1000 + b * 100 + c * 10 + d ∧
  a * 1000 + b * 100 + c * 10 + d ≤ 9999 ∧
  (a * 1000 + b * 100 + c * 10 + d) % 2 = 1 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a ≠ 0 ∧ b ≠ 0
  → ∃ (n : ℕ), n = 2240 :=
by 
  sorry

end number_of_distinct_digit_odd_numbers_l71_7177


namespace volume_of_water_overflow_l71_7150

-- Definitions based on given conditions
def mass_of_ice : ℝ := 50
def density_of_fresh_ice : ℝ := 0.9
def density_of_salt_ice : ℝ := 0.95
def density_of_fresh_water : ℝ := 1
def density_of_salt_water : ℝ := 1.03

-- Theorem statement corresponding to the problem
theorem volume_of_water_overflow
  (m : ℝ := mass_of_ice) 
  (rho_n : ℝ := density_of_fresh_ice) 
  (rho_c : ℝ := density_of_salt_ice) 
  (rho_fw : ℝ := density_of_fresh_water) 
  (rho_sw : ℝ := density_of_salt_water) :
  ∃ (ΔV : ℝ), ΔV = 2.63 :=
by
  sorry

end volume_of_water_overflow_l71_7150


namespace profit_function_maximize_profit_l71_7127

def cost_per_item : ℝ := 80
def purchase_quantity : ℝ := 1000
def selling_price_initial : ℝ := 100
def price_increase_per_item : ℝ := 1
def sales_decrease_per_yuan : ℝ := 10
def selling_price (x : ℕ) : ℝ := selling_price_initial + x
def profit (x : ℕ) : ℝ := (selling_price x - cost_per_item) * (purchase_quantity - sales_decrease_per_yuan * x)

theorem profit_function (x : ℕ) (h : 0 ≤ x ∧ x ≤ 100) : 
  profit x = -10 * (x : ℝ)^2 + 800 * (x : ℝ) + 20000 :=
by sorry

theorem maximize_profit :
  ∃ max_x, (0 ≤ max_x ∧ max_x ≤ 100) ∧ 
  (∀ x : ℕ, (0 ≤ x ∧ x ≤ 100) → profit x ≤ profit max_x) ∧ 
  max_x = 40 ∧ 
  profit max_x = 36000 :=
by sorry

end profit_function_maximize_profit_l71_7127


namespace minimize_AC_plus_BC_l71_7148

noncomputable def minimize_distance (k : ℝ) : Prop :=
  let A := (5, 5)
  let B := (2, 1)
  let C := (0, k)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let AC := dist A C
  let BC := dist B C
  ∀ k', dist (0, k') A + dist (0, k') B ≥ AC + BC

theorem minimize_AC_plus_BC : minimize_distance (15 / 7) :=
sorry

end minimize_AC_plus_BC_l71_7148


namespace pet_food_weight_in_ounces_l71_7170

-- Define the given conditions
def cat_food_bags := 2
def cat_food_weight_per_bag := 3 -- in pounds
def dog_food_bags := 2
def additional_dog_food_weight := 2 -- additional weight per bag compared to cat food
def pounds_to_ounces := 16

-- Calculate the total weight of cat food in pounds
def total_cat_food_weight := cat_food_bags * cat_food_weight_per_bag

-- Calculate the weight of each bag of dog food in pounds
def dog_food_weight_per_bag := cat_food_weight_per_bag + additional_dog_food_weight

-- Calculate the total weight of dog food in pounds
def total_dog_food_weight := dog_food_bags * dog_food_weight_per_bag

-- Calculate the total weight of pet food in pounds
def total_pet_food_weight_pounds := total_cat_food_weight + total_dog_food_weight

-- Convert the total weight to ounces
def total_pet_food_weight_ounces := total_pet_food_weight_pounds * pounds_to_ounces

-- Statement of the problem in Lean 4
theorem pet_food_weight_in_ounces : total_pet_food_weight_ounces = 256 := by
  sorry

end pet_food_weight_in_ounces_l71_7170


namespace probability_adjacent_vertices_decagon_l71_7122

noncomputable def probability_adjacent_vertices : ℚ :=
  let num_vertices := 10
  let adjacent := 2
  let remaining := num_vertices - 1 -- 9
  adjacent / remaining

theorem probability_adjacent_vertices_decagon :
  probability_adjacent_vertices = 2 / 9 :=
by
  unfold probability_adjacent_vertices
  simp
  sorry

end probability_adjacent_vertices_decagon_l71_7122


namespace solve_polynomial_l71_7162

theorem solve_polynomial (z : ℂ) :
    z^5 - 5 * z^3 + 6 * z = 0 ↔ 
    z = 0 ∨ z = -Real.sqrt 2 ∨ z = Real.sqrt 2 ∨ z = -Real.sqrt 3 ∨ z = Real.sqrt 3 := 
by 
  sorry

end solve_polynomial_l71_7162


namespace paving_stone_width_l71_7164

theorem paving_stone_width :
  ∀ (L₁ L₂ : ℝ) (n : ℕ) (length width : ℝ), 
    L₁ = 30 → L₂ = 16 → length = 2 → n = 240 →
    (L₁ * L₂ = n * (length * width)) → width = 1 :=
by
  sorry

end paving_stone_width_l71_7164


namespace intercept_form_l71_7113

theorem intercept_form (x y : ℝ) : 2 * x - 3 * y - 4 = 0 ↔ x / 2 + y / (-4/3) = 1 := sorry

end intercept_form_l71_7113


namespace number_count_two_digit_property_l71_7115

open Nat

theorem number_count_two_digit_property : 
  (∃ (n : Finset ℕ), (∀ (x : ℕ), x ∈ n ↔ ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 11 * a + 2 * b ≡ 7 [MOD 10] ∧ x = 10 * a + b) ∧ n.card = 5) :=
by
  sorry

end number_count_two_digit_property_l71_7115


namespace infinite_solutions_implies_a_eq_2_l71_7175

theorem infinite_solutions_implies_a_eq_2 (a b : ℝ) (h : b = 1) :
  (∀ x : ℝ, a * (3 * x - 2) + b * (2 * x - 3) = 8 * x - 7) → a = 2 :=
by
  intro H
  sorry

end infinite_solutions_implies_a_eq_2_l71_7175


namespace increasing_function_range_l71_7173

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1 / 2) * m * x^2 - 2 * x + Real.log x

theorem increasing_function_range (m : ℝ) : (∀ x > 0, m * x + (1 / x) - 2 ≥ 0) ↔ m ≥ 1 := 
by 
  sorry

end increasing_function_range_l71_7173


namespace inequality_proof_l71_7193

theorem inequality_proof
  (a b c : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h_eq : a + b + c = 4 * (abc)^(1/3)) :
  2 * (ab + bc + ca) + 4 * min (a^2) (min (b^2) (c^2)) ≥ a^2 + b^2 + c^2 :=
by
  sorry

end inequality_proof_l71_7193


namespace pima_investment_value_l71_7189

noncomputable def pima_investment_worth (initial_investment : ℕ) (first_week_gain_percentage : ℕ) (second_week_gain_percentage : ℕ) : ℕ :=
  let first_week_value := initial_investment + (initial_investment * first_week_gain_percentage / 100)
  let second_week_value := first_week_value + (first_week_value * second_week_gain_percentage / 100)
  second_week_value

-- Conditions
def initial_investment := 400
def first_week_gain_percentage := 25
def second_week_gain_percentage := 50

theorem pima_investment_value :
  pima_investment_worth initial_investment first_week_gain_percentage second_week_gain_percentage = 750 := by
  sorry

end pima_investment_value_l71_7189


namespace last_recess_break_duration_l71_7174

-- Definitions based on the conditions
def first_recess_break : ℕ := 15
def second_recess_break : ℕ := 15
def lunch_break : ℕ := 30
def total_outside_class_time : ℕ := 80

-- The theorem we need to prove
theorem last_recess_break_duration :
  total_outside_class_time = first_recess_break + second_recess_break + lunch_break + 20 :=
sorry

end last_recess_break_duration_l71_7174


namespace calculation_correct_l71_7139

def grid_coloring_probability : ℚ := 591 / 1024

theorem calculation_correct : (m + n = 1615) ↔ (∃ m n : ℕ, m + n = 1615 ∧ gcd m n = 1 ∧ grid_coloring_probability = m / n) := sorry

end calculation_correct_l71_7139


namespace halfway_fraction_l71_7160

theorem halfway_fraction (a b : ℚ) (h1 : a = 2 / 9) (h2 : b = 1 / 3) :
  (a + b) / 2 = 5 / 18 :=
by
  sorry

end halfway_fraction_l71_7160


namespace binom_30_3_l71_7190

theorem binom_30_3 : Nat.choose 30 3 = 4060 := 
by 
  sorry

end binom_30_3_l71_7190


namespace sum_of_exponents_of_1985_eq_40_l71_7149

theorem sum_of_exponents_of_1985_eq_40 :
  ∃ (e₀ e₁ e₂ e₃ e₄ e₅ : ℕ), 1985 = 2^e₀ + 2^e₁ + 2^e₂ + 2^e₃ + 2^e₄ + 2^e₅ 
  ∧ e₀ ≠ e₁ ∧ e₀ ≠ e₂ ∧ e₀ ≠ e₃ ∧ e₀ ≠ e₄ ∧ e₀ ≠ e₅
  ∧ e₁ ≠ e₂ ∧ e₁ ≠ e₃ ∧ e₁ ≠ e₄ ∧ e₁ ≠ e₅
  ∧ e₂ ≠ e₃ ∧ e₂ ≠ e₄ ∧ e₂ ≠ e₅
  ∧ e₃ ≠ e₄ ∧ e₃ ≠ e₅
  ∧ e₄ ≠ e₅
  ∧ e₀ + e₁ + e₂ + e₃ + e₄ + e₅ = 40 := 
by
  sorry

end sum_of_exponents_of_1985_eq_40_l71_7149


namespace no_real_solution_for_x_l71_7116

theorem no_real_solution_for_x
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1/y = 5) (h2 : y + 1/x = 1/3) :
  false :=
by
  sorry

end no_real_solution_for_x_l71_7116


namespace temperature_problem_product_of_possible_N_l71_7156

theorem temperature_problem (M L : ℤ) (N : ℤ) :
  (M = L + N) →
  (M - 8 = L + N - 8) →
  (L + 4 = L + 4) →
  (|((L + N - 8) - (L + 4))| = 3) →
  N = 15 ∨ N = 9 :=
by sorry

theorem product_of_possible_N :
  (∀ M L : ℤ, ∀ N : ℤ,
    (M = L + N) →
    (M - 8 = L + N - 8) →
    (L + 4 = L + 4) →
    (|((L + N - 8) - (L + 4))| = 3) →
    N = 15 ∨ N = 9) →
    15 * 9 = 135 :=
by sorry

end temperature_problem_product_of_possible_N_l71_7156


namespace fred_total_earnings_l71_7142

def fred_earnings (earnings_per_hour hours_worked : ℝ) : ℝ := earnings_per_hour * hours_worked

theorem fred_total_earnings :
  fred_earnings 12.5 8 = 100 := by
sorry

end fred_total_earnings_l71_7142


namespace restaurant_meal_cost_l71_7143

/--
Each adult meal costs $8 and kids eat free. 
If there is a group of 11 people, out of which 2 are kids, 
prove that the total cost for the group to eat is $72.
-/
theorem restaurant_meal_cost (cost_per_adult : ℕ) (group_size : ℕ) (kids : ℕ) 
  (all_free_kids : ℕ → Prop) (total_cost : ℕ)  
  (h1 : cost_per_adult = 8) 
  (h2 : group_size = 11) 
  (h3 : kids = 2) 
  (h4 : all_free_kids kids) 
  (h5 : total_cost = (group_size - kids) * cost_per_adult) : 
  total_cost = 72 := 
by 
  sorry

end restaurant_meal_cost_l71_7143


namespace intersection_of_A_and_B_l71_7109

def A : Set ℝ := {x | x^2 - x = 0}
def B : Set ℝ := {y | y^2 + y = 0}

theorem intersection_of_A_and_B : A ∩ B = {0} :=
by
  sorry

end intersection_of_A_and_B_l71_7109


namespace focus_coordinates_of_parabola_l71_7147

def parabola_focus_coordinates (x y : ℝ) : Prop :=
  x^2 + y = 0 ∧ (0, -1/4) = (0, y)

theorem focus_coordinates_of_parabola (x y : ℝ) :
  parabola_focus_coordinates x y →
  (0, y) = (0, -1/4) := by
  sorry

end focus_coordinates_of_parabola_l71_7147


namespace michael_initial_fish_l71_7114

-- Define the conditions
def benGave : ℝ := 18.0
def totalFish : ℝ := 67

-- Define the statement to be proved
theorem michael_initial_fish :
  (totalFish - benGave) = 49 := by
  sorry

end michael_initial_fish_l71_7114


namespace inconsistency_proof_l71_7145

-- Let TotalBoys be the number of boys, which is 120
def TotalBoys := 120

-- Let AverageMarks be the average marks obtained by 120 boys, which is 40
def AverageMarks := 40

-- Let PassedBoys be the number of boys who passed, which is 125
def PassedBoys := 125

-- Let AverageMarksFailed be the average marks of failed boys, which is 15
def AverageMarksFailed := 15

-- We need to prove the inconsistency
theorem inconsistency_proof :
  ∀ (P : ℝ), 
    (TotalBoys * AverageMarks = PassedBoys * P + (TotalBoys - PassedBoys) * AverageMarksFailed) →
    False :=
by
  intro P h
  sorry

end inconsistency_proof_l71_7145


namespace necessary_but_not_sufficient_l71_7111

def p (x : ℝ) : Prop := x < 1
def q (x : ℝ) : Prop := x^2 + x - 2 < 0

theorem necessary_but_not_sufficient (x : ℝ):
  (p x → q x) ∧ (q x → p x) → False ∧ (q x → p x) :=
sorry

end necessary_but_not_sufficient_l71_7111


namespace quadratic_expression_evaluation_l71_7158

theorem quadratic_expression_evaluation (x y : ℝ) (h1 : 3 * x + y = 10) (h2 : x + 3 * y = 14) :
  10 * x^2 + 12 * x * y + 10 * y^2 = 296 :=
by
  -- Proof goes here
  sorry

end quadratic_expression_evaluation_l71_7158


namespace induction_step_l71_7133

theorem induction_step
  (x y : ℝ)
  (k : ℕ)
  (base : ∀ n, ∃ m, (n = 2 * m - 1) → (x^n + y^n) = (x + y) * m) :
  (x^(2 * k + 1) + y^(2 * k + 1)) = (x + y) * (k + 1) :=
by
  sorry

end induction_step_l71_7133


namespace min_value_inverse_sum_l71_7196

theorem min_value_inverse_sum (a b : ℝ) (h : a > 0) (k : b > 0) (hab : a + 2 * b = 1) : 
  ∃ (y : ℝ), y = 3 + 2 * Real.sqrt 2 ∧ (∀ x, x = (1 / a) + (1 / b) → y ≤ x) :=
sorry

end min_value_inverse_sum_l71_7196


namespace rose_bushes_in_park_l71_7157

theorem rose_bushes_in_park (current_bushes : ℕ) (newly_planted : ℕ) (h1 : current_bushes = 2) (h2 : newly_planted = 4) : current_bushes + newly_planted = 6 :=
by
  sorry

end rose_bushes_in_park_l71_7157


namespace exp_add_l71_7151

theorem exp_add (z w : Complex) : Complex.exp z * Complex.exp w = Complex.exp (z + w) := 
by 
  sorry

end exp_add_l71_7151


namespace inequality_sum_l71_7168

open Real
open BigOperators

theorem inequality_sum 
  (n : ℕ) 
  (h : n > 1) 
  (x : Fin n → ℝ)
  (hx1 : ∀ i, 0 < x i) 
  (hx2 : ∑ i, x i = 1) :
  ∑ i, x i / sqrt (1 - x i) ≥ (∑ i, sqrt (x i)) / sqrt (n - 1) :=
sorry

end inequality_sum_l71_7168


namespace arrangement_condition_l71_7191

theorem arrangement_condition (x y z : ℕ) 
  (hx : x ≥ 1) 
  (hy : y ≥ 1) 
  (hz : z ≥ 1) 
  (H1 : x ≤ y + z) 
  (H2 : y ≤ x + z) 
  (H3 : z ≤ x + y) : 
  ∃ (A : ℕ) (B : ℕ) (C : ℕ), 
    A = x ∧ B = y ∧ C = z ∧
    A ≥ 1 ∧ B ≥ 1 ∧ C ≥ 1 ∧
    (A ≤ B + C) ∧ (B ≤ A + C) ∧ (C ≤ A + B) :=
by
  sorry

end arrangement_condition_l71_7191


namespace evaluate_expression_l71_7172

theorem evaluate_expression : (-3)^4 / 3^2 - 2^5 + 7^2 = 26 := by
  sorry

end evaluate_expression_l71_7172


namespace range_of_m_l71_7137

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (2 * x - m) / (x - 3) - 1 = x / (3 - x)) →
  m > 3 ∧ m ≠ 9 :=
by
  sorry

end range_of_m_l71_7137


namespace gerald_pfennigs_left_l71_7141

theorem gerald_pfennigs_left (cost_of_pie : ℕ) (farthings_initial : ℕ) (farthings_per_pfennig : ℕ) :
  cost_of_pie = 2 → farthings_initial = 54 → farthings_per_pfennig = 6 → 
  (farthings_initial / farthings_per_pfennig) - cost_of_pie = 7 :=
by
  intros h1 h2 h3
  sorry

end gerald_pfennigs_left_l71_7141


namespace river_width_l71_7106

theorem river_width 
  (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) 
  (h_depth : depth = 2) 
  (h_flow_rate: flow_rate_kmph = 3) 
  (h_volume : volume_per_minute = 4500) : 
  the_width_of_the_river = 45 :=
by
  sorry 

end river_width_l71_7106


namespace find_m_l71_7161

-- Define the given vectors and the parallel condition
def vectors_parallel (m : ℝ) : Prop :=
  let a := (1, m)
  let b := (3, 1)
  a.1 * b.2 = a.2 * b.1

-- Statement to be proved
theorem find_m (m : ℝ) : vectors_parallel m → m = 1 / 3 :=
by
  sorry

end find_m_l71_7161


namespace bike_price_l71_7197

theorem bike_price (x : ℝ) (h1 : 0.1 * x = 150) : x = 1500 := 
by sorry

end bike_price_l71_7197


namespace distance_traveled_downstream_l71_7129

noncomputable def speed_boat : ℝ := 20  -- Speed of the boat in still water in km/hr
noncomputable def rate_current : ℝ := 5  -- Rate of current in km/hr
noncomputable def time_minutes : ℝ := 24  -- Time traveled downstream in minutes
noncomputable def time_hours : ℝ := time_minutes / 60  -- Convert time to hours
noncomputable def effective_speed_downstream : ℝ := speed_boat + rate_current  -- Effective speed downstream

theorem distance_traveled_downstream :
  effective_speed_downstream * time_hours = 10 := by {
  sorry
}

end distance_traveled_downstream_l71_7129


namespace max_area_of_rectangular_pen_l71_7153

theorem max_area_of_rectangular_pen (P : ℕ) (hP : P = 60) : 
  ∃ A : ℕ, A = 225 ∧ (∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A) := 
by
  sorry

end max_area_of_rectangular_pen_l71_7153


namespace median_length_angle_bisector_length_l71_7140

variable (a b c : ℝ) (ma n : ℝ)

theorem median_length (h1 : ma = Real.sqrt ((b^2 + c^2) / 2 - a^2 / 4)) : 
  ma = Real.sqrt ((b^2 + c^2) / 2 - a^2 / 4) :=
by
  sorry

theorem angle_bisector_length (h2 : n = b * c * Real.sqrt (((b + c)^2 - a^2) / (b + c)^2)) :
  n = b * c * Real.sqrt (((b + c)^2 - a^2) / (b + c)^2) :=
by
  sorry

end median_length_angle_bisector_length_l71_7140


namespace exists_member_T_divisible_by_3_l71_7199

-- Define the set T of all numbers which are the sum of the squares of four consecutive integers
def T := { x : ℤ | ∃ n : ℤ, x = (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 }

-- Theorem to prove that there exists a member in T which is divisible by 3
theorem exists_member_T_divisible_by_3 : ∃ x ∈ T, x % 3 = 0 :=
by
  sorry

end exists_member_T_divisible_by_3_l71_7199


namespace part_I_part_II_l71_7192

variable (α : ℝ)

-- The given conditions.
variable (h1 : π < α)
variable (h2 : α < (3 * π) / 2)
variable (h3 : Real.sin α = -4/5)

-- Part (I): Prove cos α = -3/5
theorem part_I : Real.cos α = -3/5 :=
sorry

-- Part (II): Prove sin 2α + 3 tan α = 24/25 + 4
theorem part_II : Real.sin (2 * α) + 3 * Real.tan α = 24/25 + 4 :=
sorry

end part_I_part_II_l71_7192


namespace average_annual_reduction_10_percent_l71_7131

theorem average_annual_reduction_10_percent :
  ∀ x : ℝ, (1 - x) ^ 2 = 1 - 0.19 → x = 0.1 :=
by
  intros x h
  -- Proof to be filled in
  sorry

end average_annual_reduction_10_percent_l71_7131


namespace min_value_expr_l71_7124

theorem min_value_expr (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ k : ℝ, k = 6 ∧ (∃ a b c : ℝ,
                  0 < a ∧
                  0 < b ∧
                  0 < c ∧
                  (k = (a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a)) :=
sorry

end min_value_expr_l71_7124


namespace factor_theorem_l71_7125

theorem factor_theorem (t : ℝ) : (5 * t^2 + 15 * t - 20 = 0) ↔ (t = 1 ∨ t = -4) :=
by
  sorry

end factor_theorem_l71_7125


namespace volume_of_water_flowing_per_minute_l71_7183

variable (d w r : ℝ) (V : ℝ)

theorem volume_of_water_flowing_per_minute (h1 : d = 3) 
                                           (h2 : w = 32) 
                                           (h3 : r = 33.33) : 
  V = 3199.68 :=
by
  sorry

end volume_of_water_flowing_per_minute_l71_7183


namespace max_grapes_discarded_l71_7126

theorem max_grapes_discarded (n : ℕ) : 
  ∃ k : ℕ, k ∣ n → 7 * k + 6 = n → ∃ m, m = 6 := by
  sorry

end max_grapes_discarded_l71_7126


namespace prob_of_different_colors_l71_7178

def total_balls_A : ℕ := 4 + 5 + 6
def total_balls_B : ℕ := 7 + 6 + 2

noncomputable def prob_same_color : ℚ :=
  (4 / ↑total_balls_A * 7 / ↑total_balls_B) +
  (5 / ↑total_balls_A * 6 / ↑total_balls_B) +
  (6 / ↑total_balls_A * 2 / ↑total_balls_B)

noncomputable def prob_different_color : ℚ :=
  1 - prob_same_color

theorem prob_of_different_colors :
  prob_different_color = 31 / 45 :=
by
  sorry

end prob_of_different_colors_l71_7178


namespace trajectory_equation_l71_7163

theorem trajectory_equation : ∀ (x y : ℝ),
  (x + 3)^2 + y^2 + (x - 3)^2 + y^2 = 38 → x^2 + y^2 = 10 :=
by
  intros x y h
  sorry

end trajectory_equation_l71_7163


namespace all_d_zero_l71_7102

def d (n m : ℕ) : ℤ := sorry -- or some explicit initial definition

theorem all_d_zero (n m : ℕ) (h₁ : n ≥ 0) (h₂ : 0 ≤ m) (h₃ : m ≤ n) :
  (m = 0 ∨ m = n → d n m = 0) ∧
  (0 < m ∧ m < n → m * d n m = m * d (n - 1) m + (2 * n - m) * d (n - 1) (m - 1))
:=
  sorry

end all_d_zero_l71_7102


namespace simplify_polynomial_l71_7198

theorem simplify_polynomial : 
  (3 * x^3 + 4 * x^2 + 9 * x - 5) - (2 * x^3 + 3 * x^2 + 6 * x - 8) = x^3 + x^2 + 3 * x + 3 :=
by
  sorry

end simplify_polynomial_l71_7198


namespace sum_of_squares_of_four_consecutive_even_numbers_l71_7181

open Int

theorem sum_of_squares_of_four_consecutive_even_numbers (x y z w : ℤ) 
    (hx : x % 2 = 0) (hy : y = x + 2) (hz : z = x + 4) (hw : w = x + 6)
    : x + y + z + w = 36 → x^2 + y^2 + z^2 + w^2 = 344 := by
  sorry

end sum_of_squares_of_four_consecutive_even_numbers_l71_7181


namespace odd_function_max_to_min_l71_7144

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem odd_function_max_to_min (a b : ℝ) (f : ℝ → ℝ)
  (hodd : is_odd_function f)
  (hmax : ∃ x : ℝ, x > 0 ∧ (a * f x + b * x + 1) = 2) :
  ∃ y : ℝ, y < 0 ∧ (a * f y + b * y + 1) = 0 :=
sorry

end odd_function_max_to_min_l71_7144


namespace compare_triangle_operations_l71_7179

def tri_op (a b : ℤ) : ℤ := a * b - a - b + 1

theorem compare_triangle_operations : tri_op (-3) 4 = tri_op 4 (-3) :=
by
  unfold tri_op
  sorry

end compare_triangle_operations_l71_7179


namespace find_sheets_used_l71_7103

variable (x y : ℕ) -- define variables for x and y
variable (h₁ : 82 - x = y) -- 82 - x = number of sheets left
variable (h₂ : y = x - 6) -- number of sheets left = number of sheets used - 6

theorem find_sheets_used (h₁ : 82 - x = x - 6) : x = 44 := 
by
  sorry

end find_sheets_used_l71_7103


namespace birthday_friends_count_l71_7134

theorem birthday_friends_count (n : ℕ) 
    (h1 : ∃ T, T = 12 * (n + 2)) 
    (h2 : ∃ T', T' = 16 * n) 
    (h3 : (∃ T, T = 12 * (n + 2)) → ∃ T', T' = 16 * n) : 
    n = 6 := 
by
    sorry

end birthday_friends_count_l71_7134


namespace equal_sum_sequence_S_9_l71_7104

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Conditions taken from the problem statement
def equal_sum_sequence (a : ℕ → ℕ) (c : ℕ) :=
  ∀ n : ℕ, a n + a (n + 1) = c

def sum_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum a

-- Lean statement of the problem
theorem equal_sum_sequence_S_9
  (h1 : equal_sum_sequence a 5)
  (h2 : a 1 = 2)
  : sum_first_n_terms a 9 = 22 :=
sorry

end equal_sum_sequence_S_9_l71_7104


namespace average_of_first_45_results_l71_7130

theorem average_of_first_45_results
  (A : ℝ)
  (h1 : (45 + 25 : ℝ) = 70)
  (h2 : (25 : ℝ) * 45 = 1125)
  (h3 : (70 : ℝ) * 32.142857142857146 = 2250)
  (h4 : ∀ x y z : ℝ, 45 * x + y = z → x = 25) :
  A = 25 :=
by
  sorry

end average_of_first_45_results_l71_7130


namespace bowling_ball_surface_area_l71_7135

theorem bowling_ball_surface_area (diameter : ℝ) (h : diameter = 9) :
    let r := diameter / 2
    let surface_area := 4 * Real.pi * r^2
    surface_area = 81 * Real.pi := by
  sorry

end bowling_ball_surface_area_l71_7135


namespace second_integer_value_l71_7128

-- Definitions of conditions directly from a)
def consecutive_integers (a b c : ℤ) : Prop :=
  b = a + 1 ∧ c = b + 1

def sum_of_first_and_third (a c : ℤ) (sum : ℤ) : Prop :=
  a + c = sum

-- Translated proof problem
theorem second_integer_value (n: ℤ) (h1: consecutive_integers (n - 1) n (n + 1))
  (h2: sum_of_first_and_third (n - 1) (n + 1) 118) : 
  n = 59 :=
by
  sorry

end second_integer_value_l71_7128


namespace bread_carriers_l71_7101

-- Definitions for the number of men, women, and children
variables (m w c : ℕ)

-- Conditions from the problem
def total_people := m + w + c = 12
def total_bread := 8 * m + 2 * w + c = 48

-- Theorem to prove the correct number of men, women, and children
theorem bread_carriers (h1 : total_people m w c) (h2 : total_bread m w c) : 
  m = 5 ∧ w = 1 ∧ c = 6 :=
sorry

end bread_carriers_l71_7101


namespace sufficient_and_necessary_condition_l71_7117

variable {a_n : ℕ → ℝ}

-- Defining the geometric sequence and the given conditions
def is_geometric_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a_n (n + 1) = a_n n * r

def is_increasing_sequence (a_n : ℕ → ℝ) : Prop :=
  ∀ n, a_n n < a_n (n + 1)

def condition (a_n : ℕ → ℝ) : Prop := a_n 0 < a_n 1 ∧ a_n 1 < a_n 2

-- The proof statement
theorem sufficient_and_necessary_condition (a_n : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a_n) :
  condition a_n ↔ is_increasing_sequence a_n :=
sorry

end sufficient_and_necessary_condition_l71_7117


namespace complement_union_l71_7105

def M := { x : ℝ | (x + 3) * (x - 1) < 0 }
def N := { x : ℝ | x ≤ -3 }
def union_set := M ∪ N

theorem complement_union :
  ∀ x : ℝ, x ∈ (⊤ \ union_set) ↔ x ≥ 1 :=
by
  sorry

end complement_union_l71_7105


namespace light_intensity_after_glass_pieces_minimum_glass_pieces_l71_7167

theorem light_intensity_after_glass_pieces (a : ℝ) (x : ℕ) : 
  (y : ℝ) = a * (0.9 ^ x) :=
sorry

theorem minimum_glass_pieces (a : ℝ) (x : ℕ) : 
  a * (0.9 ^ x) < a / 3 ↔ x ≥ 11 :=
sorry

end light_intensity_after_glass_pieces_minimum_glass_pieces_l71_7167


namespace value_of_q_l71_7138

open Real

theorem value_of_q (p q : ℝ) (hpq_cond1 : 1 < p ∧ p < q) 
  (hpq_cond2 : 1 / p + 1 / q = 1) (hpq_cond3 : p * q = 8) : q = 4 + 2 * sqrt 2 :=
by
  sorry

end value_of_q_l71_7138


namespace paint_needed_l71_7180

theorem paint_needed (wall_area : ℕ) (coverage_per_gallon : ℕ) (number_of_coats : ℕ) (h_wall_area : wall_area = 600) (h_coverage_per_gallon : coverage_per_gallon = 400) (h_number_of_coats : number_of_coats = 2) : 
    ((number_of_coats * wall_area) / coverage_per_gallon) = 3 :=
by
  sorry

end paint_needed_l71_7180


namespace tom_finishes_in_four_hours_l71_7132

noncomputable def maryMowingRate := 1 / 3
noncomputable def tomMowingRate := 1 / 6
noncomputable def timeMaryMows := 1
noncomputable def remainingLawn := 1 - (timeMaryMows * maryMowingRate)

theorem tom_finishes_in_four_hours :
  remainingLawn / tomMowingRate = 4 :=
by sorry

end tom_finishes_in_four_hours_l71_7132


namespace time_jack_first_half_l71_7185

-- Define the conditions
def t_Jill : ℕ := 32
def t_2 : ℕ := 6
def t_Jack : ℕ := t_Jill - 7

-- Define the time Jack took for the first half
def t_1 : ℕ := t_Jack - t_2

-- State the theorem to prove
theorem time_jack_first_half : t_1 = 19 := by
  sorry

end time_jack_first_half_l71_7185


namespace conference_problem_l71_7159

noncomputable def exists_round_table (n : ℕ) (scientists : Finset ℕ) (acquaintance : ℕ → Finset ℕ) : Prop :=
  ∃ (A B C D : ℕ), A ∈ scientists ∧ B ∈ scientists ∧ C ∈ scientists ∧ D ∈ scientists ∧
  ((A ≠ B ∧ A ≠ C ∧ A ≠ D) ∧ (B ≠ C ∧ B ≠ D) ∧ (C ≠ D)) ∧
  (B ∈ acquaintance A ∧ C ∈ acquaintance B ∧ D ∈ acquaintance C ∧ A ∈ acquaintance D)

theorem conference_problem :
  ∀ (scientists : Finset ℕ),
  ∀ (acquaintance : ℕ → Finset ℕ),
    (scientists.card = 50) →
    (∀ s ∈ scientists, (acquaintance s).card ≥ 25) →
    exists_round_table 50 scientists acquaintance :=
sorry

end conference_problem_l71_7159


namespace no_such_function_l71_7107

noncomputable def no_such_function_exists : Prop :=
  ¬∃ f : ℝ → ℝ, 
    (∀ y : ℝ, ∃ x : ℝ, f x = y) ∧ 
    (∀ x : ℝ, f (f x) = (x - 1) * f x + 2)

-- Here's the theorem statement to be proved
theorem no_such_function : no_such_function_exists :=
sorry

end no_such_function_l71_7107


namespace scientist_prob_rain_l71_7195

theorem scientist_prob_rain (x : ℝ) (p0 p1 : ℝ)
  (h0 : p0 + p1 = 1)
  (h1 : ∀ x : ℝ, x = (p0 * x^2 + p0 * (1 - x) * x + p1 * (1 - x) * x) / x + (1 - x) - x^2 / (x + 1))
  (h2 : (x + p0 / (x + 1) - x^2 / (x + 1)) = 0.2) :
  x = 1/9 := 
sorry

end scientist_prob_rain_l71_7195


namespace solve_log_sin_eq_l71_7186

noncomputable def log_base (b : ℝ) (a : ℝ) : ℝ :=
  Real.log a / Real.log b

theorem solve_log_sin_eq :
  ∀ x : ℝ, 
  (0 < Real.sin x ∧ Real.sin x < 1) →
  log_base (Real.sin x) 4 * log_base (Real.sin x ^ 2) 2 = 4 →
  ∃ k : ℤ, x = (-1)^k * (Real.pi / 4) + Real.pi * k := 
by
  sorry

end solve_log_sin_eq_l71_7186


namespace count_of_squares_difference_l71_7152

theorem count_of_squares_difference (h_range : ∀ n, 1 ≤ n ∧ n ≤ 2000) :
  ∃ count, (∀ n, 1 ≤ n ∧ n ≤ 2000 → 
             (∃ a b, n = a^2 - b^2)) ↔ count = 1500 :=
by sorry

end count_of_squares_difference_l71_7152


namespace sufficient_but_not_necessary_l71_7187

theorem sufficient_but_not_necessary (a : ℝ) : (a = 2 → (a - 1) * (a - 2) = 0) ∧ 
  (∀ x : ℝ, (x - 1) * (x - 2) = 0 → x ≠ 2 → x = 1) ∧
  (a = 2 → (1 ≠ 2)) :=
by {
  sorry
}

end sufficient_but_not_necessary_l71_7187


namespace arithmetic_sequence_third_eighth_term_sum_l71_7121

variable {α : Type*} [AddCommGroup α] [Module ℚ α]

def arith_sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem arithmetic_sequence_third_eighth_term_sum {a : ℕ → ℚ} {S : ℕ → ℚ} 
  (h_seq: ∀ n, a n = a 1 + (n - 1) * d)
  (h_sum: arith_sequence_sum a S) 
  (h_S10 : S 10 = 4) : 
  a 3 + a 8 = 4 / 5 :=
by
  sorry

end arithmetic_sequence_third_eighth_term_sum_l71_7121


namespace max_sin_product_proof_l71_7110

noncomputable def max_sin_product : ℝ :=
  let A := (-8, 0)
  let B := (8, 0)
  let C (t : ℝ) := (t, 6)
  let AB : ℝ := 16
  let AC (t : ℝ) := Real.sqrt ((t + 8)^2 + 36)
  let BC (t : ℝ) := Real.sqrt ((t - 8)^2 + 36)
  let area : ℝ := 48
  let sin_ACB (t : ℝ) := 96 / Real.sqrt (((t + 8)^2 + 36) * ((t - 8)^2 + 36))
  let sin_CAB_CBA : ℝ := 3 / 8
  sin_CAB_CBA

theorem max_sin_product_proof : ∀ t : ℝ, max_sin_product = 3 / 8 :=
by
  sorry

end max_sin_product_proof_l71_7110


namespace vegetable_options_l71_7123

open Nat

theorem vegetable_options (V : ℕ) : 
  3 * V + 6 = 57 → V = 5 :=
by
  intro h
  sorry

end vegetable_options_l71_7123


namespace sum_of_diagonals_l71_7118

-- Definitions of the given lengths
def AB := 5
def CD := 5
def BC := 12
def DE := 12
def AE := 18

-- Variables for the diagonal lengths
variables (AC BD CE : ℚ)

-- The Lean 4 theorem statement
theorem sum_of_diagonals (hAC : AC = 723 / 44) (hBD : BD = 44 / 3) (hCE : CE = 351 / 22) :
  AC + BD + CE = 6211 / 132 :=
by
  sorry

end sum_of_diagonals_l71_7118


namespace highway_length_l71_7112

theorem highway_length 
  (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) 
  (h_speed1 : speed1 = 14)
  (h_speed2 : speed2 = 16)
  (h_time : time = 1.5) : 
  speed1 * time + speed2 * time = 45 := 
sorry

end highway_length_l71_7112


namespace proposition_B_proposition_D_l71_7165

open Real

variable (a b : ℝ)

theorem proposition_B (h : a^2 ≠ b^2) : a ≠ b := 
sorry

theorem proposition_D (h : a > abs b) : a^2 > b^2 :=
sorry

end proposition_B_proposition_D_l71_7165


namespace european_stamps_cost_l71_7100

def prices : String → ℕ 
| "Italy"   => 7
| "Japan"   => 7
| "Germany" => 5
| "China"   => 5
| _ => 0

def stamps_1950s : String → ℕ 
| "Italy"   => 5
| "Germany" => 8
| "China"   => 10
| "Japan"   => 6
| _ => 0

def stamps_1960s : String → ℕ 
| "Italy"   => 9
| "Germany" => 12
| "China"   => 5
| "Japan"   => 10
| _ => 0

def total_cost (stamps : String → ℕ) (price : String → ℕ) : ℕ :=
  (stamps "Italy" * price "Italy" +
   stamps "Germany" * price "Germany") 

theorem european_stamps_cost : total_cost stamps_1950s prices + total_cost stamps_1960s prices = 198 :=
by
  sorry

end european_stamps_cost_l71_7100
