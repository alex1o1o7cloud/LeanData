import Mathlib

namespace four_digit_sum_of_digits_divisible_by_101_l1668_166821

theorem four_digit_sum_of_digits_divisible_by_101 (a b c d : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9)
  (h2 : 1 ≤ b ∧ b ≤ 9)
  (h3 : 1 ≤ c ∧ c ≤ 9)
  (h4 : 1 ≤ d ∧ d ≤ 9)
  (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_div : (1001 * a + 110 * b + 110 * c + 1001 * d) % 101 = 0) :
  (a + d) % 101 = (b + c) % 101 :=
by
  sorry

end four_digit_sum_of_digits_divisible_by_101_l1668_166821


namespace sum_p_q_r_l1668_166856

def b (n : ℕ) : ℕ :=
if n < 1 then 0 else
if n < 2 then 2 else
if n < 4 then 4 else
if n < 7 then 6
else 6 -- Continue this pattern for illustration; an infinite structure would need proper handling for all n.

noncomputable def p := 2
noncomputable def q := 0
noncomputable def r := 0

theorem sum_p_q_r : p + q + r = 2 :=
by sorry

end sum_p_q_r_l1668_166856


namespace non_neg_int_solutions_eq_10_l1668_166878

theorem non_neg_int_solutions_eq_10 :
  ∃ n : ℕ, n = 55 ∧
  (∃ (x y z : ℕ), x + y + z = 10) :=
by
  sorry

end non_neg_int_solutions_eq_10_l1668_166878


namespace limit_of_sequence_l1668_166872

theorem limit_of_sequence {ε : ℝ} (hε : ε > 0) : 
  ∃ (N : ℝ), ∀ (n : ℝ), n > N → |(2 * n^3) / (n^3 - 2) - 2| < ε :=
by
  sorry

end limit_of_sequence_l1668_166872


namespace polynomial_product_expansion_l1668_166882

theorem polynomial_product_expansion (x : ℝ) : (x^2 + 3 * x + 3) * (x^2 - 3 * x + 3) = x^4 - 3 * x^2 + 9 := 
by sorry

end polynomial_product_expansion_l1668_166882


namespace triangle_area_is_correct_l1668_166829

noncomputable def triangle_area_inscribed_circle (r : ℝ) (θ1 θ2 θ3 : ℝ) : ℝ := 
  (1 / 2) * r^2 * (Real.sin θ1 + Real.sin θ2 + Real.sin θ3)

theorem triangle_area_is_correct :
  triangle_area_inscribed_circle (18 / Real.pi) (Real.pi / 3) (2 * Real.pi / 3) Real.pi =
  162 * Real.sqrt 3 / (Real.pi^2) :=
by sorry

end triangle_area_is_correct_l1668_166829


namespace inequaliy_pos_real_abc_l1668_166826

theorem inequaliy_pos_real_abc (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_abc : a * b * c = 1) : 
  (a / (a * b + 1)) + (b / (b * c + 1)) + (c / (c * a + 1)) ≥ (3 / 2) := 
by
  sorry

end inequaliy_pos_real_abc_l1668_166826


namespace time_to_reach_6400ft_is_200min_l1668_166875

noncomputable def time_to_reach_ship (depth : ℕ) (rate : ℕ) : ℕ :=
  depth / rate

theorem time_to_reach_6400ft_is_200min :
  time_to_reach_ship 6400 32 = 200 := by
  sorry

end time_to_reach_6400ft_is_200min_l1668_166875


namespace world_book_day_l1668_166831

theorem world_book_day
  (x y : ℕ)
  (h1 : x + y = 22)
  (h2 : x = 2 * y + 1) :
  x = 15 ∧ y = 7 :=
by {
  -- The proof is omitted as per the instructions
  sorry
}

end world_book_day_l1668_166831


namespace sample_size_is_five_l1668_166888

def population := 100
def sample (n : ℕ) := n ≤ population
def sample_size (n : ℕ) := n

theorem sample_size_is_five (n : ℕ) (h : sample 5) : sample_size 5 = 5 :=
by
  sorry

end sample_size_is_five_l1668_166888


namespace four_pow_2024_mod_11_l1668_166879

theorem four_pow_2024_mod_11 : (4 ^ 2024) % 11 = 3 :=
by
  sorry

end four_pow_2024_mod_11_l1668_166879


namespace total_number_of_vehicles_l1668_166800

theorem total_number_of_vehicles 
  (lanes : ℕ) 
  (trucks_per_lane : ℕ) 
  (buses_per_lane : ℕ) 
  (cars_per_lane : ℕ := 2 * lanes * trucks_per_lane) 
  (motorcycles_per_lane : ℕ := 3 * buses_per_lane)
  (total_trucks : ℕ := lanes * trucks_per_lane)
  (total_cars : ℕ := lanes * cars_per_lane)
  (total_buses : ℕ := lanes * buses_per_lane)
  (total_motorcycles : ℕ := lanes * motorcycles_per_lane)
  (total_vehicles : ℕ := total_trucks + total_cars + total_buses + total_motorcycles)
  (hlanes : lanes = 4) 
  (htrucks : trucks_per_lane = 60) 
  (hbuses : buses_per_lane = 40) :
  total_vehicles = 2800 := sorry

end total_number_of_vehicles_l1668_166800


namespace angle_CAB_EQ_angle_EAD_l1668_166807

variable {A B C D E : Type}

-- Define the angles as variables for the pentagon ABCDE
variable (ABC ADE CEA BDA CAB EAD : ℝ)

-- Given conditions
axiom angle_ABC_EQ_angle_ADE : ABC = ADE
axiom angle_CEA_EQ_angle_BDA : CEA = BDA

-- Prove that angle CAB equals angle EAD
theorem angle_CAB_EQ_angle_EAD : CAB = EAD :=
by
  sorry

end angle_CAB_EQ_angle_EAD_l1668_166807


namespace sum_of_first_ten_terms_l1668_166870

theorem sum_of_first_ten_terms (a : ℕ → ℝ)
  (h1 : a 3 ^ 2 + a 8 ^ 2 + 2 * a 3 * a 8 = 9)
  (h2 : ∀ n, a n < 0) :
  (5 * (a 3 + a 8) = -15) :=
sorry

end sum_of_first_ten_terms_l1668_166870


namespace total_volume_cylinder_cone_sphere_l1668_166816

theorem total_volume_cylinder_cone_sphere (r h : ℝ) (π : ℝ)
  (hc : π * r^2 * h = 150 * π)
  (hv : ∀ (r h : ℝ) (π : ℝ), V_cone = 1/3 * π * r^2 * h)
  (hs : ∀ (r : ℝ) (π : ℝ), V_sphere = 4/3 * π * r^3) :
  V_total = 50 * π + (4/3 * π * (150^(2/3))) :=
by
  sorry

end total_volume_cylinder_cone_sphere_l1668_166816


namespace shanna_tomato_ratio_l1668_166865

-- Define the initial conditions
def initial_tomato_plants : ℕ := 6
def initial_eggplant_plants : ℕ := 2
def initial_pepper_plants : ℕ := 4
def pepper_plants_died : ℕ := 1
def vegetables_per_plant : ℕ := 7
def total_vegetables_harvested : ℕ := 56

-- Define the number of tomato plants that died
def tomato_plants_died (total_vegetables : ℕ) (veg_per_plant : ℕ) (initial_tomato : ℕ) 
  (initial_eggplant : ℕ) (initial_pepper : ℕ) (pepper_died : ℕ) : ℕ :=
  let surviving_plants := total_vegetables / veg_per_plant
  let surviving_pepper := initial_pepper - pepper_died
  let surviving_tomato := surviving_plants - (initial_eggplant + surviving_pepper)
  initial_tomato - surviving_tomato

-- Define the ratio
def ratio_tomato_plants_died_to_initial (tomato_died : ℕ) (initial_tomato : ℕ) : ℚ :=
  (tomato_died : ℚ) / (initial_tomato : ℚ)

theorem shanna_tomato_ratio :
  ratio_tomato_plants_died_to_initial (tomato_plants_died total_vegetables_harvested vegetables_per_plant 
    initial_tomato_plants initial_eggplant_plants initial_pepper_plants pepper_plants_died) initial_tomato_plants 
  = 1 / 2 := by
  sorry

end shanna_tomato_ratio_l1668_166865


namespace factor_72x3_minus_252x7_l1668_166881

theorem factor_72x3_minus_252x7 (x : ℝ) : (72 * x^3 - 252 * x^7) = (36 * x^3 * (2 - 7 * x^4)) :=
by
  sorry

end factor_72x3_minus_252x7_l1668_166881


namespace solve_for_a_l1668_166845

theorem solve_for_a (x a : ℤ) (h : x = 3) (heq : 2 * x - 10 = 4 * a) : a = -1 := by
  sorry

end solve_for_a_l1668_166845


namespace afternoon_more_than_evening_l1668_166877

def campers_in_morning : Nat := 33
def campers_in_afternoon : Nat := 34
def campers_in_evening : Nat := 10

theorem afternoon_more_than_evening : campers_in_afternoon - campers_in_evening = 24 := by
  sorry

end afternoon_more_than_evening_l1668_166877


namespace school_population_proof_l1668_166866

variables (x y z: ℕ)
variable (B: ℕ := (50 * y) / 100)

theorem school_population_proof (h1 : 162 = (x * B) / 100)
                               (h2 : B = (50 * y) / 100)
                               (h3 : z = 100 - 50) :
  z = 50 :=
  sorry

end school_population_proof_l1668_166866


namespace sqrt_sum_ineq_l1668_166884

open Real

theorem sqrt_sum_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a^2 + b^2 + c^2 = 1) :
  sqrt (1 - a^2) + sqrt (1 - b^2) + sqrt (1 - c^2) + a + b + c > 3 := by
  sorry

end sqrt_sum_ineq_l1668_166884


namespace water_channel_area_l1668_166880

-- Define the given conditions
def top_width := 14
def bottom_width := 8
def depth := 70

-- The area formula for a trapezium given the top width, bottom width, and height
def trapezium_area (a b h : ℕ) : ℕ :=
  (a + b) * h / 2

-- The main theorem stating the area of the trapezium
theorem water_channel_area : 
  trapezium_area top_width bottom_width depth = 770 := by
  -- Proof can be completed here
  sorry

end water_channel_area_l1668_166880


namespace tail_length_l1668_166851

variable (Length_body Length_tail Length_head : ℝ)

-- Conditions
def tail_half_body (Length_tail Length_body : ℝ) := Length_tail = 1/2 * Length_body
def head_sixth_body (Length_head Length_body : ℝ) := Length_head = 1/6 * Length_body
def overall_length (Length_head Length_body Length_tail : ℝ) := Length_head + Length_body + Length_tail = 30

-- Theorem statement
theorem tail_length (h1 : tail_half_body Length_tail Length_body) 
                  (h2 : head_sixth_body Length_head Length_body) 
                  (h3 : overall_length Length_head Length_body Length_tail) : 
                  Length_tail = 6 := by
  sorry

end tail_length_l1668_166851


namespace quadratic_has_real_roots_l1668_166833

theorem quadratic_has_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m-2) * x^2 - 2 * x + 1 = 0) ↔ m ≤ 3 :=
by sorry

end quadratic_has_real_roots_l1668_166833


namespace meaningful_expression_range_l1668_166812

theorem meaningful_expression_range (x : ℝ) : (∃ (y : ℝ), y = 5 / (x - 2)) ↔ x ≠ 2 := 
by
  sorry

end meaningful_expression_range_l1668_166812


namespace total_yearly_interest_l1668_166813

/-- Mathematical statement:
Given Nina's total inheritance of $12,000, with $5,000 invested at 6% interest and the remainder invested at 8% interest, the total yearly interest from both investments is $860.
-/
theorem total_yearly_interest (principal : ℕ) (principal_part : ℕ) (rate1 rate2 : ℚ) (interest_part1 interest_part2 : ℚ) (total_interest : ℚ) :
  principal = 12000 ∧ principal_part = 5000 ∧ rate1 = 0.06 ∧ rate2 = 0.08 ∧
  interest_part1 = (principal_part : ℚ) * rate1 ∧ interest_part2 = ((principal - principal_part) : ℚ) * rate2 →
  total_interest = interest_part1 + interest_part2 → 
  total_interest = 860 := by
  sorry

end total_yearly_interest_l1668_166813


namespace find_a8_l1668_166847

variable (a : ℕ → ℤ)

def arithmetic_sequence : Prop :=
  ∀ n : ℕ, 2 * a (n + 1) = a n + a (n + 2)

theorem find_a8 (h1 : a 7 + a 9 = 16) (h2 : arithmetic_sequence a) : a 8 = 8 := by
  -- proof would go here
  sorry

end find_a8_l1668_166847


namespace james_present_age_l1668_166836

-- Definitions and conditions
variables (D J : ℕ) -- Dan's and James's ages are natural numbers

-- Condition 1: The ratio between Dan's and James's ages
def ratio_condition : Prop := (D * 5 = J * 6)

-- Condition 2: In 4 years, Dan will be 28
def future_age_condition : Prop := (D + 4 = 28)

-- The proof goal: James's present age is 20
theorem james_present_age : ratio_condition D J ∧ future_age_condition D → J = 20 :=
by
  sorry

end james_present_age_l1668_166836


namespace no_closed_loop_after_replacement_l1668_166890

theorem no_closed_loop_after_replacement (N M : ℕ) 
  (h1 : N = M) 
  (h2 : (N + M) % 4 = 0) :
  ¬((N - 1) - (M + 1)) % 4 = 0 :=
by
  sorry

end no_closed_loop_after_replacement_l1668_166890


namespace sum_of_solutions_l1668_166805

theorem sum_of_solutions (x : ℝ) : 
  (∃ y z, x^2 + 2017 * x - 24 = 2017 ∧ y^2 + 2017 * y - 2041 = 0 ∧ z^2 + 2017 * z - 2041 = 0 ∧ y ≠ z) →
  y + z = -2017 := 
by 
  sorry

end sum_of_solutions_l1668_166805


namespace expected_worth_flip_l1668_166815

/-- A biased coin lands on heads with probability 2/3 and on tails with probability 1/3.
Each heads flip gains $5, and each tails flip loses $9.
If three consecutive flips all result in tails, then an additional loss of $10 is applied.
Prove that the expected worth of a single coin flip is -1/27. -/
theorem expected_worth_flip :
  let P_heads := 2 / 3
  let P_tails := 1 / 3
  (P_heads * 5 + P_tails * -9) - (P_tails ^ 3 * 10) = -1 / 27 :=
by
  sorry

end expected_worth_flip_l1668_166815


namespace find_range_t_l1668_166863

def sequence_increasing (n : ℕ) (t : ℝ) : Prop :=
  (2 * (n + 1) + t^2 - 8) / (n + 1 + t) > (2 * n + t^2 - 8) / (n + t)

theorem find_range_t (t : ℝ) (h : ∀ n : ℕ, sequence_increasing n t) : 
  -1 < t ∧ t < 4 :=
sorry

end find_range_t_l1668_166863


namespace age_ratio_in_two_years_l1668_166840

-- Definitions based on conditions
def lennon_age_current : ℕ := 8
def ophelia_age_current : ℕ := 38
def lennon_age_in_two_years := lennon_age_current + 2
def ophelia_age_in_two_years := ophelia_age_current + 2

-- Statement to prove
theorem age_ratio_in_two_years : 
  (ophelia_age_in_two_years / gcd ophelia_age_in_two_years lennon_age_in_two_years) = 4 ∧
  (lennon_age_in_two_years / gcd ophelia_age_in_two_years lennon_age_in_two_years) = 1 := 
by 
  sorry

end age_ratio_in_two_years_l1668_166840


namespace probability_prime_sum_30_l1668_166835

def prime_numbers_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def prime_pairs_summing_to_30 : List (ℕ × ℕ) := [(7, 23), (11, 19), (13, 17)]

def num_prime_pairs := (prime_numbers_up_to_30.length.choose 2)

theorem probability_prime_sum_30 :
  (prime_pairs_summing_to_30.length / num_prime_pairs : ℚ) = 1 / 15 :=
sorry

end probability_prime_sum_30_l1668_166835


namespace gcd_g_150_151_l1668_166894

def g (x : ℤ) : ℤ := x^2 - 2*x + 3020

theorem gcd_g_150_151 : Int.gcd (g 150) (g 151) = 1 :=
  by
  sorry

end gcd_g_150_151_l1668_166894


namespace segment_length_abs_eq_cubrt_27_five_l1668_166808

theorem segment_length_abs_eq_cubrt_27_five : 
  (∀ x : ℝ, |x - (3 : ℝ)| = 5) → (8 - (-2) = 10) :=
by 
  intros;
  sorry

end segment_length_abs_eq_cubrt_27_five_l1668_166808


namespace functional_equation_solution_l1668_166869

noncomputable def f (x : ℝ) : ℝ := sorry

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x * f y) + f (f x + f y) = y * f x + f (x + f y)) :
  (∀ x, f x = 0) ∨ (∀ x, f x = x) :=
sorry

end functional_equation_solution_l1668_166869


namespace matrix_A_pow_50_l1668_166806

def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![1, 1],
  ![0, 1]
]

theorem matrix_A_pow_50 : A^50 = ![
  ![1, 50],
  ![0, 1]
] :=
sorry

end matrix_A_pow_50_l1668_166806


namespace find_value_of_x_l1668_166830
-- Import the broader Mathlib to bring in the entirety of the necessary library

-- Definitions for the conditions
variables {x y z : ℝ}

-- Assume the given conditions
axiom h1 : x = y
axiom h2 : y = 2 * z
axiom h3 : x * y * z = 256

-- Statement to prove
theorem find_value_of_x : x = 8 :=
by {
  -- Proof goes here
  sorry
}

end find_value_of_x_l1668_166830


namespace special_burger_cost_l1668_166810

/-
  Prices of individual items and meals:
  - Burger: $5
  - French Fries: $3
  - Soft Drink: $3
  - Kid’s Burger: $3
  - Kid’s French Fries: $2
  - Kid’s Juice Box: $2
  - Kids Meal: $5

  Mr. Parker purchases:
  - 2 special burger meals for adults
  - 2 special burger meals and 2 kids' meals for 4 children
  - Saving $10 by buying 6 meals instead of the individual items

  Goal: 
  - Prove that the cost of one special burger meal is $8.
-/

def price_burger : Nat := 5
def price_fries : Nat := 3
def price_drink : Nat := 3
def price_kid_burger : Nat := 3
def price_kid_fries : Nat := 2
def price_kid_juice : Nat := 2
def price_kids_meal : Nat := 5

def total_adults_cost : Nat :=
  2 * price_burger + 2 * price_fries + 2 * price_drink

def total_kids_cost : Nat :=
  2 * price_kid_burger + 2 * price_kid_fries + 2 * price_kid_juice

def total_individual_cost : Nat :=
  total_adults_cost + total_kids_cost

def total_meals_cost : Nat :=
  total_individual_cost - 10

def cost_kids_meals : Nat :=
  2 * price_kids_meal

def total_cost_4_meals : Nat :=
  total_meals_cost

def cost_special_burger_meal : Nat :=
  (total_cost_4_meals - cost_kids_meals) / 2

theorem special_burger_cost : cost_special_burger_meal = 8 := by
  sorry

end special_burger_cost_l1668_166810


namespace defective_pens_l1668_166876

theorem defective_pens :
  ∃ D N : ℕ, (N + D = 9) ∧ (N / 9 * (N - 1) / 8 = 5 / 12) ∧ (D = 3) :=
by
  sorry

end defective_pens_l1668_166876


namespace transform_expression_to_product_l1668_166854

variables (a b c d s: ℝ)

theorem transform_expression_to_product
  (h1 : d = a + b + c)
  (h2 : s = (a + b + c + d) / 2) :
    2 * (a^2 * b^2 + a^2 * c^2 + a^2 * d^2 + b^2 * c^2 + b^2 * d^2 + c^2 * d^2) -
    (a^4 + b^4 + c^4 + d^4) + 8 * a * b * c * d = 16 * (s - a) * (s - b) * (s - c) * (s - d) :=
by
  sorry

end transform_expression_to_product_l1668_166854


namespace evaluate_expression_l1668_166811

theorem evaluate_expression : 7899665 - 12 * 3 * 2 = 7899593 :=
by
  -- This proof is skipped.
  sorry

end evaluate_expression_l1668_166811


namespace arithmetic_sequence_property_l1668_166862

def arith_seq (a : ℕ → ℤ) (a1 a3 : ℤ) (d : ℤ) : Prop :=
  a 1 = a1 ∧ a 3 = a3 ∧ (a 3 - a 1) = 2 * d

theorem arithmetic_sequence_property :
  ∀ (a : ℕ → ℤ), ∃ d : ℤ, arith_seq a 1 (-3) d →
  (1 - (a 2) - a 3 - (a 4) - (a 5) = 17) :=
by
  intros a
  use -2
  simp [arith_seq, *]
  sorry

end arithmetic_sequence_property_l1668_166862


namespace population_density_reduction_l1668_166859

theorem population_density_reduction (scale : ℕ) (real_world_population : ℕ) : 
  scale = 1000000 → real_world_population = 1000000000 → 
  real_world_population / (scale ^ 2) < 1 := 
by 
  intros scale_value rw_population_value
  have h1 : scale ^ 2 = 1000000000000 := by sorry
  have h2 : real_world_population / 1000000000000 = 1 / 1000 := by sorry
  sorry

end population_density_reduction_l1668_166859


namespace window_treatments_cost_l1668_166839

-- Define the costs and the number of windows
def cost_sheers : ℝ := 40.00
def cost_drapes : ℝ := 60.00
def number_of_windows : ℕ := 3

-- Define the total cost calculation
def total_cost := (cost_sheers + cost_drapes) * number_of_windows

-- State the theorem that needs to be proved
theorem window_treatments_cost : total_cost = 300.00 :=
by
  sorry

end window_treatments_cost_l1668_166839


namespace buffaloes_number_l1668_166873

theorem buffaloes_number (B D : ℕ) 
  (h : 4 * B + 2 * D = 2 * (B + D) + 24) : 
  B = 12 :=
sorry

end buffaloes_number_l1668_166873


namespace fault_line_movement_l1668_166887

theorem fault_line_movement (total_movement: ℝ) (past_year: ℝ) (prev_year: ℝ) (total_eq: total_movement = 6.5) (past_eq: past_year = 1.25) :
  prev_year = 5.25 := by
  sorry

end fault_line_movement_l1668_166887


namespace cindy_gave_25_pens_l1668_166817

theorem cindy_gave_25_pens (initial_pens mike_gave pens_given_sharon final_pens : ℕ) (h1 : initial_pens = 5) (h2 : mike_gave = 20) (h3 : pens_given_sharon = 19) (h4 : final_pens = 31) :
  final_pens = initial_pens + mike_gave - pens_given_sharon + 25 :=
by 
  -- Insert the proof here later
  sorry

end cindy_gave_25_pens_l1668_166817


namespace car_speed_ratio_to_pedestrian_speed_l1668_166864

noncomputable def ratio_of_speeds (length_pedestrian_car: ℝ) (length_bridge: ℝ) (speed_pedestrian: ℝ) (speed_car: ℝ) : ℝ :=
  speed_car / speed_pedestrian

theorem car_speed_ratio_to_pedestrian_speed
  (L : ℝ)  -- length of the bridge
  (vc vp : ℝ)  -- speeds of car and pedestrian respectively
  (h1 : (2 / 5) * L + (3 / 5) * vp = L)
  (h2 : (3 / 5) * L = vc * (L / (5 * vp))) :
  ratio_of_speeds L L vp vc = 5 :=
by
  sorry

end car_speed_ratio_to_pedestrian_speed_l1668_166864


namespace evaluate_polynomial_at_neg2_l1668_166849

theorem evaluate_polynomial_at_neg2 : 2 * (-2)^4 + 3 * (-2)^3 + 5 * (-2)^2 + (-2) + 4 = 30 :=
by 
  sorry

end evaluate_polynomial_at_neg2_l1668_166849


namespace megan_picture_shelves_l1668_166896

def books_per_shelf : ℕ := 7
def mystery_shelves : ℕ := 8
def total_books : ℕ := 70
def total_mystery_books : ℕ := mystery_shelves * books_per_shelf
def total_picture_books : ℕ := total_books - total_mystery_books
def picture_shelves : ℕ := total_picture_books / books_per_shelf

theorem megan_picture_shelves : picture_shelves = 2 := 
by sorry

end megan_picture_shelves_l1668_166896


namespace product_of_geometric_terms_l1668_166867

noncomputable def arithmeticSeq (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

noncomputable def geometricSeq (b1 r : ℕ) (n : ℕ) : ℕ :=
  b1 * r^(n - 1)

theorem product_of_geometric_terms :
  ∃ (a1 d b1 r : ℕ),
    (3 * a1 - (arithmeticSeq a1 d 8)^2 + 3 * (arithmeticSeq a1 d 15) = 0) ∧ 
    (arithmeticSeq a1 d 8 = geometricSeq b1 r 10) ∧ 
    (geometricSeq b1 r 3 * geometricSeq b1 r 17 = 36) :=
sorry

end product_of_geometric_terms_l1668_166867


namespace unique_two_digit_solution_l1668_166823

theorem unique_two_digit_solution : ∃! (t : ℕ), 10 ≤ t ∧ t < 100 ∧ 13 * t % 100 = 52 := sorry

end unique_two_digit_solution_l1668_166823


namespace james_age_when_john_turned_35_l1668_166860

theorem james_age_when_john_turned_35 :
  ∀ (J : ℕ) (Tim : ℕ) (John : ℕ),
  (Tim = 5) →
  (Tim + 5 = 2 * John) →
  (Tim = 79) →
  (John = 35) →
  (J = John) →
  J = 35 :=
by
  intros J Tim John h1 h2 h3 h4 h5
  rw [h4] at h5
  exact h5

end james_age_when_john_turned_35_l1668_166860


namespace minimum_people_to_save_cost_l1668_166838

-- Define the costs for the two event planners.
def cost_first_planner (x : ℕ) : ℕ := 120 + 18 * x
def cost_second_planner (x : ℕ) : ℕ := 250 + 15 * x

-- State the theorem to prove the minimum number of people required for the second event planner to be less expensive.
theorem minimum_people_to_save_cost : ∃ x : ℕ, cost_second_planner x < cost_first_planner x ∧ ∀ y : ℕ, y < x → cost_second_planner y ≥ cost_first_planner y :=
sorry

end minimum_people_to_save_cost_l1668_166838


namespace coterminal_angle_in_radians_l1668_166801

theorem coterminal_angle_in_radians (d : ℝ) (h : d = 2010) : 
  ∃ r : ℝ, r = -5 * Real.pi / 6 ∧ (∃ k : ℤ, d = r * 180 / Real.pi + k * 360) :=
by sorry

end coterminal_angle_in_radians_l1668_166801


namespace determine_digits_from_expression_l1668_166828

theorem determine_digits_from_expression (a b c x y z S : ℕ) 
  (hx : x = 100) (hy : y = 10) (hz : z = 1)
  (hS : S = a * x + b * y + c * z) :
  S = 100 * a + 10 * b + c :=
by
  -- Variables
  -- a, b, c : ℕ -- digits to find
  -- x, y, z : ℕ -- chosen numbers
  -- S : ℕ -- the given sum

  -- Assumptions
  -- hx : x = 100
  -- hy : y = 10
  -- hz : z = 1
  -- hS : S = a * x + b * y + c * z
  sorry

end determine_digits_from_expression_l1668_166828


namespace power_mod_equiv_l1668_166891

theorem power_mod_equiv :
  7 ^ 145 % 12 = 7 % 12 :=
by
  -- Here the solution would go
  sorry

end power_mod_equiv_l1668_166891


namespace probability_divisibility_9_correct_l1668_166868

-- Define the set S
def S : Set ℕ := { n | ∃ a b: ℕ, 0 ≤ a ∧ a < 40 ∧ 0 ≤ b ∧ b < 40 ∧ a ≠ b ∧ n = 2^a + 2^b }

-- Define the criteria for divisibility by 9
def divisible_by_9 (n : ℕ) : Prop := 9 ∣ n

-- Define the total size of set S
def size_S : ℕ := 780  -- as calculated from combination

-- Count valid pairs (a, b) such that 2^a + 2^b is divisible by 9
def valid_pairs : ℕ := 133  -- as calculated from summation

-- Define the probability
def probability_divisible_by_9 : ℕ := valid_pairs / size_S

-- The proof statement
theorem probability_divisibility_9_correct:
  (valid_pairs : ℚ) / (size_S : ℚ) = 133 / 780 := sorry

end probability_divisibility_9_correct_l1668_166868


namespace right_triangle_hypotenuse_length_l1668_166819

theorem right_triangle_hypotenuse_length 
    (AB AC x y : ℝ) 
    (P : AB = x) (Q : AC = y) 
    (ratio_AP_PB : AP / PB = 1 / 3) 
    (ratio_AQ_QC : AQ / QC = 2 / 1) 
    (BQ_length : BQ = 18) 
    (CP_length : CP = 24) : 
    BC = 24 := 
by 
  sorry

end right_triangle_hypotenuse_length_l1668_166819


namespace cannot_cut_square_into_7_rectangles_l1668_166889

theorem cannot_cut_square_into_7_rectangles (a : ℝ) :
  ¬ ∃ (x : ℝ), 7 * 2 * x ^ 2 = a ^ 2 ∧ 
    ∀ (i : ℕ), 0 ≤ i → i < 7 → (∃ (rect : ℝ × ℝ), rect.1 = x ∧ rect.2 = 2 * x ) :=
by
  sorry

end cannot_cut_square_into_7_rectangles_l1668_166889


namespace perfect_square_of_sides_of_triangle_l1668_166846

theorem perfect_square_of_sides_of_triangle 
  (a b c : ℤ) 
  (h1: a > 0 ∧ b > 0 ∧ c > 0)
  (h2: a + b > c ∧ b + c > a ∧ c + a > b)
  (gcd_abc: Int.gcd (Int.gcd a b) c = 1)
  (h3: (a^2 + b^2 - c^2) % (a + b - c) = 0)
  (h4: (b^2 + c^2 - a^2) % (b + c - a) = 0)
  (h5: (c^2 + a^2 - b^2) % (c + a - b) = 0) : 
  ∃ n : ℤ, n^2 = (a + b - c) * (b + c - a) * (c + a - b) ∨ 
  ∃ m : ℤ, m^2 = 2 * (a + b - c) * (b + c - a) * (c + a - b) := 
sorry

end perfect_square_of_sides_of_triangle_l1668_166846


namespace problem_a_problem_b_problem_c_l1668_166898

variable (a b : ℝ)

theorem problem_a {a b : ℝ} (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem problem_b {a b : ℝ} (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem problem_c {a b : ℝ} (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

end problem_a_problem_b_problem_c_l1668_166898


namespace probability_businessmen_wait_two_minutes_l1668_166850

theorem probability_businessmen_wait_two_minutes :
  let total_suitcases := 200
  let business_suitcases := 10
  let time_to_wait_seconds := 120
  let suitcases_in_120_seconds := time_to_wait_seconds / 2
  let prob := (Nat.choose 59 9) / (Nat.choose total_suitcases business_suitcases)
  suitcases_in_120_seconds = 60 ->
  prob = (Nat.choose 59 9) / (Nat.choose 200 10) :=
by 
  sorry

end probability_businessmen_wait_two_minutes_l1668_166850


namespace binomial_product_l1668_166824

open Nat

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binomial_product_l1668_166824


namespace part1_part2_l1668_166832

section
variable (x a : ℝ)

def p (a x : ℝ) : Prop :=
  x^2 - 4*a*x + 3*a^2 < 0 ∧ a > 0

def q (x : ℝ) : Prop :=
  (x - 3) / (x - 2) ≤ 0

theorem part1 (h1 : p 1 x ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

theorem part2 (h2 : ∀ x, ¬p a x → ¬q x) : 1 < a ∧ a ≤ 2 := by
  sorry

end

end part1_part2_l1668_166832


namespace red_cookies_count_l1668_166822

-- Definitions of the conditions
def total_cookies : ℕ := 86
def pink_cookies : ℕ := 50

-- The proof problem statement
theorem red_cookies_count : ∃ y : ℕ, y = total_cookies - pink_cookies := by
  use 36
  show 36 = total_cookies - pink_cookies
  sorry

end red_cookies_count_l1668_166822


namespace prime_divides_binom_l1668_166818

-- We define that n is a prime number.
def is_prime (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Lean statement for the problem
theorem prime_divides_binom {n k : ℕ} (h₁ : is_prime n) (h₂ : 0 < k) (h₃ : k < n) :
  n ∣ Nat.choose n k :=
sorry

end prime_divides_binom_l1668_166818


namespace arithmetic_sequence_50th_term_l1668_166827

-- Definitions as per the conditions
def a_1 : ℤ := 48
def d : ℤ := -2
def n : ℕ := 50

-- Statement to prove the 50th term in the series
theorem arithmetic_sequence_50th_term : a_1 + (n - 1) * d = -50 :=
by
  sorry

end arithmetic_sequence_50th_term_l1668_166827


namespace range_of_reciprocals_l1668_166874

theorem range_of_reciprocals (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h_neq : a ≠ b) (h_sum : a + b = 1) :
  4 < (1 / a + 1 / b) :=
sorry

end range_of_reciprocals_l1668_166874


namespace different_purchasing_methods_l1668_166842

noncomputable def number_of_purchasing_methods (n_two_priced : ℕ) (n_one_priced : ℕ) (total_price : ℕ) : ℕ :=
  let combinations_two_price (k : ℕ) := Nat.choose n_two_priced k
  let combinations_one_price (k : ℕ) := Nat.choose n_one_priced k
  combinations_two_price 5 + (combinations_two_price 4 * combinations_one_price 2)

theorem different_purchasing_methods :
  number_of_purchasing_methods 8 3 10 = 266 :=
by
  sorry

end different_purchasing_methods_l1668_166842


namespace glucose_in_mixed_solution_l1668_166814

def concentration1 := 20 / 100  -- concentration of first solution in grams per cubic centimeter
def concentration2 := 30 / 100  -- concentration of second solution in grams per cubic centimeter
def volume1 := 80               -- volume of first solution in cubic centimeters
def volume2 := 50               -- volume of second solution in cubic centimeters

theorem glucose_in_mixed_solution :
  (concentration1 * volume1) + (concentration2 * volume2) = 31 := by
  sorry

end glucose_in_mixed_solution_l1668_166814


namespace similarity_transformation_result_l1668_166837

-- Define the original coordinates of point A and the similarity ratio
def A : ℝ × ℝ := (2, 2)
def ratio : ℝ := 2

-- Define the similarity transformation that scales coordinates, optionally considering reflection
def similarity_transform (p : ℝ × ℝ) (r : ℝ) : ℝ × ℝ :=
  (r * p.1, r * p.2)

-- Use Lean to state the theorem based on the given conditions and expected answer
theorem similarity_transformation_result :
  similarity_transform A ratio = (4, 4) ∨ similarity_transform A (-ratio) = (-4, -4) :=
by
  sorry

end similarity_transformation_result_l1668_166837


namespace product_of_solutions_l1668_166893

theorem product_of_solutions : 
  ∀ y : ℝ, (|y| = 3 * (|y| - 2)) → ∃ a b : ℝ, (a = 3 ∧ b = -3) ∧ (a * b = -9) := 
by 
  sorry

end product_of_solutions_l1668_166893


namespace find_b_l1668_166892

-- Define the variables involved
variables (a b : ℝ)

-- Conditions provided in the problem
def condition_1 : Prop := 2 * a + 1 = 1
def condition_2 : Prop := b + a = 3

-- Theorem statement to prove b = 3 given the conditions
theorem find_b (h1 : condition_1 a) (h2 : condition_2 a b) : b = 3 := by
  sorry

end find_b_l1668_166892


namespace train_pass_time_approx_l1668_166844

noncomputable def time_to_pass_platform
  (L_t L_p : ℝ)
  (V_t : ℝ) : ℝ :=
  (L_t + L_p) / (V_t * (1000 / 3600))

theorem train_pass_time_approx
  (L_t L_p V_t : ℝ)
  (hL_t : L_t = 720)
  (hL_p : L_p = 360)
  (hV_t : V_t = 75) :
  abs (time_to_pass_platform L_t L_p V_t - 51.85) < 0.01 := 
by
  rw [hL_t, hL_p, hV_t]
  sorry

end train_pass_time_approx_l1668_166844


namespace length_of_first_train_l1668_166861

theorem length_of_first_train
    (speed_first_train_kmph : ℝ) 
    (speed_second_train_kmph : ℝ) 
    (time_to_cross_seconds : ℝ) 
    (length_second_train_meters : ℝ)
    (H1 : speed_first_train_kmph = 120)
    (H2 : speed_second_train_kmph = 80)
    (H3 : time_to_cross_seconds = 9)
    (H4 : length_second_train_meters = 300.04) : 
    ∃ (length_first_train : ℝ), length_first_train = 200 :=
by 
    let relative_speed_m_per_s := (speed_first_train_kmph + speed_second_train_kmph) * 1000 / 3600
    let combined_length := relative_speed_m_per_s * time_to_cross_seconds
    let length_first_train := combined_length - length_second_train_meters
    use length_first_train
    sorry

end length_of_first_train_l1668_166861


namespace abs_sub_nonneg_l1668_166899

theorem abs_sub_nonneg (a : ℝ) : |a| - a ≥ 0 :=
sorry

end abs_sub_nonneg_l1668_166899


namespace cost_of_superman_game_l1668_166825

-- Define the costs as constants
def cost_batman_game : ℝ := 13.60
def total_amount_spent : ℝ := 18.66

-- Define the theorem to prove the cost of the Superman game
theorem cost_of_superman_game : total_amount_spent - cost_batman_game = 5.06 :=
by
  sorry

end cost_of_superman_game_l1668_166825


namespace quadratic_root_difference_l1668_166871

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

noncomputable def root_difference (a b c : ℝ) : ℝ :=
  (Real.sqrt (discriminant a b c)) / a

theorem quadratic_root_difference :
  root_difference (3 + 2 * Real.sqrt 2) (5 + Real.sqrt 2) (-4) = Real.sqrt (177 - 122 * Real.sqrt 2) :=
by
  sorry

end quadratic_root_difference_l1668_166871


namespace triangle_isosceles_l1668_166895

-- Definitions involved: Triangle, Circumcircle, Angle Bisector, Isosceles Triangle
universe u

structure Triangle (α : Type u) :=
  (A B C : α)

structure Circumcircle (α : Type u) :=
  (triangle : Triangle α)

structure AngleBisector (α : Type u) :=
  (A : α)
  (triangle : Triangle α)

def IsoscelesTriangle {α : Type u} (P Q R : α) : Prop :=
  ∃ (p₁ p₂ p₃ : α), (p₁ = P ∧ p₂ = Q ∧ p₃ = R) ∧
                  ((∃ θ₁ θ₂, θ₁ + θ₂ = 90) → (∃ θ₃ θ₂, θ₃ + θ₂ = 90))

theorem triangle_isosceles {α : Type u} (T : Triangle α) (S : α)
  (h1 : Circumcircle α) (h2 : AngleBisector α) :
  IsoscelesTriangle T.B T.C S :=
by
  sorry

end triangle_isosceles_l1668_166895


namespace jill_total_phone_time_l1668_166883

def phone_time : ℕ → ℕ
| 0 => 5
| (n + 1) => 2 * phone_time n

theorem jill_total_phone_time (n : ℕ) (h : n = 4) : 
  phone_time 0 + phone_time 1 + phone_time 2 + phone_time 3 + phone_time 4 = 155 :=
by
  cases h
  sorry

end jill_total_phone_time_l1668_166883


namespace orange_balls_count_l1668_166855

theorem orange_balls_count (total_balls red_balls blue_balls yellow_balls green_balls pink_balls orange_balls : ℕ) 
(h_total : total_balls = 100)
(h_red : red_balls = 30)
(h_blue : blue_balls = 20)
(h_yellow : yellow_balls = 10)
(h_green : green_balls = 5)
(h_pink : pink_balls = 2 * green_balls)
(h_orange : orange_balls = 3 * pink_balls)
(h_sum : red_balls + blue_balls + yellow_balls + green_balls + pink_balls + orange_balls = total_balls) :
orange_balls = 30 :=
sorry

end orange_balls_count_l1668_166855


namespace trapezoid_area_l1668_166803

theorem trapezoid_area
  (A B C D : ℝ)
  (BC AD AC : ℝ)
  (radius circle_center : ℝ)
  (h : ℝ)
  (angleBAD angleADC : ℝ)
  (tangency : Bool) :
  BC = 13 → 
  angleBAD = 2 * angleADC →
  radius = 5 →
  tangency = true →
  1/2 * (BC + AD) * h = 157.5 :=
by
  sorry

end trapezoid_area_l1668_166803


namespace age_of_fourth_child_l1668_166820

theorem age_of_fourth_child 
  (avg_age : ℕ) 
  (age1 age2 age3 : ℕ) 
  (age4 : ℕ)
  (h_avg : (age1 + age2 + age3 + age4) / 4 = avg_age) 
  (h1 : age1 = 6) 
  (h2 : age2 = 8) 
  (h3 : age3 = 11) 
  (h_avg_val : avg_age = 9) : 
  age4 = 11 := 
by 
  sorry

end age_of_fourth_child_l1668_166820


namespace sqrt_27_eq_3_sqrt_3_l1668_166843

theorem sqrt_27_eq_3_sqrt_3 : Real.sqrt 27 = 3 * Real.sqrt 3 :=
by
  sorry

end sqrt_27_eq_3_sqrt_3_l1668_166843


namespace no_integers_satisfy_equation_l1668_166853

theorem no_integers_satisfy_equation :
  ∀ (a b c : ℤ), a^2 + b^2 - 8 * c ≠ 6 := by
  sorry

end no_integers_satisfy_equation_l1668_166853


namespace intersection_of_N_and_not_R_M_l1668_166802

def M : Set ℝ := {x | x > 2}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def Not_R_M : Set ℝ := {x | x ≤ 2}

theorem intersection_of_N_and_not_R_M : 
  N ∩ Not_R_M = {x | 1 < x ∧ x ≤ 2} := by
  sorry

end intersection_of_N_and_not_R_M_l1668_166802


namespace emails_difference_l1668_166857

def morning_emails : ℕ := 6
def afternoon_emails : ℕ := 8

theorem emails_difference :
  afternoon_emails - morning_emails = 2 := 
by
  sorry

end emails_difference_l1668_166857


namespace suff_not_necessary_condition_l1668_166804

noncomputable def p : ℝ := 1
noncomputable def q (x : ℝ) : Prop := x^3 - 2 * x + 1 = 0

theorem suff_not_necessary_condition :
  (∀ x, x = p → q x) ∧ (∃ x, q x ∧ x ≠ p) :=
by
  sorry

end suff_not_necessary_condition_l1668_166804


namespace distance_CD_l1668_166809

theorem distance_CD (d_north: ℝ) (d_east: ℝ) (d_south: ℝ) (d_west: ℝ) (distance_CD: ℝ) :
  d_north = 30 ∧ d_east = 80 ∧ d_south = 20 ∧ d_west = 30 → distance_CD = 50 :=
by
  intros h
  sorry

end distance_CD_l1668_166809


namespace rectangle_quadrilateral_inequality_l1668_166858

theorem rectangle_quadrilateral_inequality 
  (a b c d : ℝ)
  (h_a : 0 < a) (h_a_bound : a < 3)
  (h_b : 0 < b) (h_b_bound : b < 4)
  (h_c : 0 < c) (h_c_bound : c < 3)
  (h_d : 0 < d) (h_d_bound : d < 4) :
  25 ≤ ((3 - a)^2 + a^2 + (4 - b)^2 + b^2 + (3 - c)^2 + c^2 + (4 - d)^2 + d^2) ∧
  ((3 - a)^2 + a^2 + (4 - b)^2 + b^2 + (3 - c)^2 + c^2 + (4 - d)^2 + d^2) < 50 :=
by 
  sorry

end rectangle_quadrilateral_inequality_l1668_166858


namespace TotalToysIsNinetyNine_l1668_166848

def BillHasToys : ℕ := 60
def HalfOfBillToys : ℕ := BillHasToys / 2
def AdditionalToys : ℕ := 9
def HashHasToys : ℕ := HalfOfBillToys + AdditionalToys
def TotalToys : ℕ := BillHasToys + HashHasToys

theorem TotalToysIsNinetyNine : TotalToys = 99 := by
  sorry

end TotalToysIsNinetyNine_l1668_166848


namespace problem_statement_l1668_166897

variables {A B x y a : ℝ}

theorem problem_statement (h1 : 1/A = 1 - (1 - x) / y)
                          (h2 : 1/B = 1 - y / (1 - x))
                          (h3 : x = (1 - a) / (1 - 1/a))
                          (h4 : y = 1 - 1/x)
                          (h5 : a ≠ 1) (h6 : a ≠ -1) : 
                          A + B = 1 :=
sorry

end problem_statement_l1668_166897


namespace students_and_ticket_price_l1668_166886

theorem students_and_ticket_price (students teachers ticket_price : ℕ) 
  (h1 : students % 5 = 0)
  (h2 : (students + teachers) * (ticket_price / 2) = 1599)
  (h3 : ∃ n, ticket_price = 2 * n) 
  (h4 : teachers = 1) :
  students = 40 ∧ ticket_price = 78 := 
by
  sorry

end students_and_ticket_price_l1668_166886


namespace three_digit_number_divisible_by_7_l1668_166885

theorem three_digit_number_divisible_by_7
  (a b : ℕ)
  (h1 : (a + b) % 7 = 0) :
  (100 * a + 10 * b + a) % 7 = 0 :=
sorry

end three_digit_number_divisible_by_7_l1668_166885


namespace tobias_charges_for_mowing_l1668_166852

/-- Tobias is buying a new pair of shoes that costs $95.
He has been saving up his money each month for the past three months.
He gets a $5 allowance a month.
He mowed 4 lawns and shoveled 5 driveways.
He charges $7 to shovel a driveway.
After buying the shoes, he has $15 in change.
Prove that Tobias charges $15 to mow a lawn.
--/
theorem tobias_charges_for_mowing 
  (shoes_cost : ℕ)
  (monthly_allowance : ℕ)
  (months_saving : ℕ)
  (lawns_mowed : ℕ)
  (driveways_shoveled : ℕ)
  (charge_per_shovel : ℕ)
  (money_left : ℕ)
  (total_money_before_purchase : ℕ)
  (x : ℕ)
  (h1 : shoes_cost = 95)
  (h2 : monthly_allowance = 5)
  (h3 : months_saving = 3)
  (h4 : lawns_mowed = 4)
  (h5 : driveways_shoveled = 5)
  (h6 : charge_per_shovel = 7)
  (h7 : money_left = 15)
  (h8 : total_money_before_purchase = shoes_cost + money_left)
  (h9 : total_money_before_purchase = (months_saving * monthly_allowance) + (lawns_mowed * x) + (driveways_shoveled * charge_per_shovel)) :
  x = 15 := 
sorry

end tobias_charges_for_mowing_l1668_166852


namespace abc_sum_zero_l1668_166841

theorem abc_sum_zero
  (a b c : ℝ)
  (h1 : ∀ x: ℝ, (a * (c * x^2 + b * x + a)^2 + b * (c * x^2 + b * x + a) + c = x)) :
  (a + b + c = 0) :=
by
  sorry

end abc_sum_zero_l1668_166841


namespace calculate_meals_l1668_166834

-- Given conditions
def meal_cost : ℕ := 7
def total_spent : ℕ := 21

-- The expected number of meals Olivia's dad paid for
def expected_meals : ℕ := 3

-- Proof statement
theorem calculate_meals : total_spent / meal_cost = expected_meals :=
by
  sorry
  -- Proof can be completed using arithmetic simplification.

end calculate_meals_l1668_166834
