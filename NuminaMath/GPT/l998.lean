import Mathlib

namespace fixed_point_is_5_225_l998_99811

theorem fixed_point_is_5_225 : ∃ a b : ℝ, (∀ k : ℝ, 9 * a^2 + k * a - 5 * k = b) → (a = 5 ∧ b = 225) :=
by
  sorry

end fixed_point_is_5_225_l998_99811


namespace complex_solution_l998_99870

theorem complex_solution (x : ℂ) (h : x^2 + 1 = 0) : x = Complex.I ∨ x = -Complex.I :=
by sorry

end complex_solution_l998_99870


namespace remainder_of_division_987543_12_l998_99829

theorem remainder_of_division_987543_12 : 987543 % 12 = 7 := by
  sorry

end remainder_of_division_987543_12_l998_99829


namespace n_is_prime_or_power_of_2_l998_99827

noncomputable def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

noncomputable def is_power_of_2 (n : ℕ) : Prop := ∃ k : ℕ, n = 2 ^ k

noncomputable def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem n_is_prime_or_power_of_2 {n : ℕ} (h1 : n > 6)
  (h2 : ∃ (a : ℕ → ℕ) (k : ℕ), 
    (∀ i : ℕ, i < k → a i < n ∧ coprime (a i) n) ∧ 
    (∀ i : ℕ, 1 ≤ i → i < k → a (i + 1) - a i = a 2 - a 1)) 
  : is_prime n ∨ is_power_of_2 n := 
sorry

end n_is_prime_or_power_of_2_l998_99827


namespace sin_240_deg_l998_99886

theorem sin_240_deg : Real.sin (240 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_deg_l998_99886


namespace total_apple_weight_proof_l998_99821

-- Define the weights of each fruit in terms of ounces
def weight_apple : ℕ := 4
def weight_orange : ℕ := 3
def weight_plum : ℕ := 2

-- Define the bag's capacity and the number of bags
def bag_capacity : ℕ := 49
def number_of_bags : ℕ := 5

-- Define the least common multiple (LCM) of the weights
def lcm_weight : ℕ := Nat.lcm weight_apple (Nat.lcm weight_orange weight_plum)

-- Define the largest multiple of LCM that is less than or equal to the bag's capacity
def max_lcm_multiple : ℕ := (bag_capacity / lcm_weight) * lcm_weight

-- Determine the number of each fruit per bag
def sets_per_bag : ℕ := max_lcm_multiple / lcm_weight
def apples_per_bag : ℕ := sets_per_bag * 1  -- 1 apple per set

-- Calculate the weight of apples per bag and total needed in all bags
def apple_weight_per_bag : ℕ := apples_per_bag * weight_apple
def total_apple_weight : ℕ := apple_weight_per_bag * number_of_bags

-- The statement to be proved in Lean
theorem total_apple_weight_proof : total_apple_weight = 80 := by
  sorry

end total_apple_weight_proof_l998_99821


namespace total_sacks_needed_l998_99860

def first_bakery_needs : ℕ := 2
def second_bakery_needs : ℕ := 4
def third_bakery_needs : ℕ := 12
def weeks : ℕ := 4

theorem total_sacks_needed :
  first_bakery_needs * weeks + second_bakery_needs * weeks + third_bakery_needs * weeks = 72 :=
by
  sorry

end total_sacks_needed_l998_99860


namespace rate_per_kg_for_apples_l998_99852

theorem rate_per_kg_for_apples (A : ℝ) :
  (8 * A + 9 * 45 = 965) → (A = 70) :=
by
  sorry

end rate_per_kg_for_apples_l998_99852


namespace tangent_curves_line_exists_l998_99844

theorem tangent_curves_line_exists (a : ℝ) :
  (∃ l : ℝ → ℝ, ∃ x₀ : ℝ, l 1 = 0 ∧ ∀ x, (l x = x₀^3 ∧ l x = a * x^2 + (15 / 4) * x - 9)) →
  a = -25/64 ∨ a = -1 :=
by
  sorry

end tangent_curves_line_exists_l998_99844


namespace tip_is_24_l998_99881

-- Definitions based on conditions
def women's_haircut_cost : ℕ := 48
def children's_haircut_cost : ℕ := 36
def number_of_children : ℕ := 2
def tip_percentage : ℚ := 0.20

-- Calculating total cost and tip amount
def total_cost : ℕ := women's_haircut_cost + (number_of_children * children's_haircut_cost)
def tip_amount : ℚ := tip_percentage * total_cost

-- Lean theorem statement based on the problem
theorem tip_is_24 : tip_amount = 24 := by
  sorry

end tip_is_24_l998_99881


namespace point_P_lies_on_x_axis_l998_99851

noncomputable def point_on_x_axis (x : ℝ) : Prop :=
  (0 = (0 : ℝ)) -- This is a placeholder definition stating explicitly that point lies on the x-axis

theorem point_P_lies_on_x_axis (x : ℝ) : point_on_x_axis x :=
by
  sorry

end point_P_lies_on_x_axis_l998_99851


namespace minimal_range_of_sample_l998_99871

theorem minimal_range_of_sample (x1 x2 x3 x4 x5 : ℝ) 
  (mean_condition : (x1 + x2 + x3 + x4 + x5) / 5 = 6) 
  (median_condition : x3 = 10) 
  (sample_order : x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧ x4 ≤ x5) : 
  (x5 - x1) = 10 :=
sorry

end minimal_range_of_sample_l998_99871


namespace prove_union_sets_l998_99885

universe u

variable {α : Type u}
variable {M N : Set ℕ}
variable (a b : ℕ)

theorem prove_union_sets (h1 : M = {3, 4^a}) (h2 : N = {a, b}) (h3 : M ∩ N = {1}) : M ∪ N = {0, 1, 3} := sorry

end prove_union_sets_l998_99885


namespace other_acute_angle_in_right_triangle_l998_99878

theorem other_acute_angle_in_right_triangle (a : ℝ) (h : a = 25) :
    ∃ b : ℝ, b = 65 :=
by
  sorry

end other_acute_angle_in_right_triangle_l998_99878


namespace find_m_l998_99896

variable (m : ℝ)

def vector_a : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (3, -2)

theorem find_m (h : (1 + 3, m - 2) = (4, m - 2) ∧ (4 * 3 + (m - 2) * (-2) = 0)) : m = 8 := by
  sorry

end find_m_l998_99896


namespace simplify_and_evaluate_expression_l998_99850

noncomputable def given_expression (x : ℝ) : ℝ :=
  (3 / (x + 2) + x - 2) / ((x^2 - 2*x + 1) / (x + 2))

theorem simplify_and_evaluate_expression (x : ℝ) (hx : |x| = 2) (h_ne : x ≠ -2) :
  given_expression x = 3 :=
by
  sorry

end simplify_and_evaluate_expression_l998_99850


namespace num_double_yolk_eggs_l998_99853

noncomputable def double_yolk_eggs (total_eggs total_yolks : ℕ) (double_yolk_contrib : ℕ) : ℕ :=
(total_yolks - total_eggs + double_yolk_contrib) / double_yolk_contrib

theorem num_double_yolk_eggs (total_eggs total_yolks double_yolk_contrib expected : ℕ)
    (h1 : total_eggs = 12)
    (h2 : total_yolks = 17)
    (h3 : double_yolk_contrib = 2)
    (h4 : expected = 5) :
  double_yolk_eggs total_eggs total_yolks double_yolk_contrib = expected :=
by
  rw [h1, h2, h3, h4]
  dsimp [double_yolk_eggs]
  norm_num
  sorry

end num_double_yolk_eggs_l998_99853


namespace thomas_saves_40_per_month_l998_99819

variables (T J : ℝ) (months : ℝ := 72) 

theorem thomas_saves_40_per_month 
  (h1 : J = (3/5) * T)
  (h2 : 72 * T + 72 * J = 4608) : 
  T = 40 :=
by sorry

end thomas_saves_40_per_month_l998_99819


namespace correct_barometric_pressure_l998_99863

noncomputable def true_barometric_pressure (p1 p2 v1 v2 T1 T2 observed_pressure_final observed_pressure_initial : ℝ) : ℝ :=
  let combined_gas_law : ℝ := (p1 * v1 * T2) / (v2 * T1)
  observed_pressure_final + combined_gas_law

theorem correct_barometric_pressure :
  true_barometric_pressure 58 56 143 155 288 303 692 704 = 748 :=
by
  sorry

end correct_barometric_pressure_l998_99863


namespace max_value_ahn_operation_l998_99893

theorem max_value_ahn_operation :
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (300 - n)^2 - 10 = 39990 :=
by
  sorry

end max_value_ahn_operation_l998_99893


namespace jina_total_mascots_l998_99800

-- Definitions and Conditions
def num_teddies := 5
def num_bunnies := 3 * num_teddies
def num_koala_bears := 1
def additional_teddies := 2 * num_bunnies

-- Total mascots calculation
def total_mascots := num_teddies + num_bunnies + num_koala_bears + additional_teddies

theorem jina_total_mascots : total_mascots = 51 := by
  sorry

end jina_total_mascots_l998_99800


namespace probability_one_excellence_A_probability_one_excellence_B_range_n_for_A_l998_99865

def probability_of_excellence_A : ℚ := 2/5
def probability_of_excellence_B1 : ℚ := 1/4
def probability_of_excellence_B2 : ℚ := 2/5
def probability_of_excellence_B3 (n : ℚ) : ℚ := n

def one_excellence_A : ℚ := 3 * (2/5) * (3/5)^2
def one_excellence_B (n : ℚ) : ℚ := 
    (probability_of_excellence_B1 * (3/5) * (1 - n)) + 
    ((1 - probability_of_excellence_B1) * (2/5) * (1 - n)) + 
    ((1 - probability_of_excellence_B1) * (3/5) * n)

theorem probability_one_excellence_A : one_excellence_A = 54/125 := sorry

theorem probability_one_excellence_B (n : ℚ) (hn : n = 1/3) : one_excellence_B n = 9/20 := sorry

def expected_excellence_A : ℚ := 3 * (2/5)
def expected_excellence_B (n : ℚ) : ℚ := (13/20) + n

theorem range_n_for_A (n : ℚ) (hn1 : 0 < n) (hn2 : n < 11/20): 
    expected_excellence_A > expected_excellence_B n := sorry

end probability_one_excellence_A_probability_one_excellence_B_range_n_for_A_l998_99865


namespace periodic_even_function_value_l998_99830

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := (x + 1) * (x - a)

-- Conditions: 
-- 1. f(x) is even 
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- 2. f(x) is periodic with period 6
def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f x = f (x + p)

-- Main theorem
theorem periodic_even_function_value 
  (a : ℝ) 
  (f_def : ∀ x, -3 ≤ x ∧ x ≤ 3 → f x a = (x + 1) * (x - a))
  (h_even : is_even_function (f · a))
  (h_periodic : is_periodic_function (f · a) 6) : 
  f (-6) a = -1 := 
sorry

end periodic_even_function_value_l998_99830


namespace number_times_half_squared_eq_eight_l998_99825

theorem number_times_half_squared_eq_eight : 
  ∃ n : ℝ, n * (1/2)^2 = 2^3 := 
sorry

end number_times_half_squared_eq_eight_l998_99825


namespace marble_counts_l998_99801

theorem marble_counts (A B C : ℕ) : 
  (∃ x : ℕ, 
    A = 165 ∧ 
    B = 57 ∧ 
    C = 21 ∧ 
    (A = 55 * x / 27) ∧ 
    (B = 19 * x / 27) ∧ 
    (C = 7 * x / 27) ∧ 
    (7 * x / 9 = x / 9 + 54) ∧ 
    (A + B + C) = 3 * x
  ) :=
sorry

end marble_counts_l998_99801


namespace sum_of_three_positives_eq_2002_l998_99894

theorem sum_of_three_positives_eq_2002 : 
  ∃ (n : ℕ), n = 334000 ∧ (∃ (f : ℕ → ℕ → ℕ → Prop), 
    (∀ (A B C : ℕ), f A B C ↔ (0 < A ∧ A ≤ B ∧ B ≤ C ∧ A + B + C = 2002))) := by
  sorry

end sum_of_three_positives_eq_2002_l998_99894


namespace minimum_value_of_sum_2_l998_99837

noncomputable def minimum_value_of_sum 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (inequality : (2 * x + Real.sqrt (4 * x ^ 2 + 1)) * (Real.sqrt (y ^ 2 + 4) - 2) ≥ y) : 
  Prop := 
  x + y = 2

theorem minimum_value_of_sum_2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (inequality : (2 * x + Real.sqrt (4 * x ^ 2 + 1)) * (Real.sqrt (y ^ 2 + 4) - 2) ≥ y) :
  minimum_value_of_sum x y hx hy inequality := 
sorry

end minimum_value_of_sum_2_l998_99837


namespace range_of_a_l998_99834

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, (2 ≤ x ∧ x ≤ 4) ∧ (2 ≤ y ∧ y ≤ 3) → x * y ≤ a * x^2 + 2 * y^2) → a ≥ 0 := 
sorry

end range_of_a_l998_99834


namespace mean_temperature_correct_l998_99831

-- Define the condition (temperatures)
def temperatures : List Int :=
  [-6, -3, -3, -4, 2, 4, 1]

-- Define the total number of days
def num_days : ℕ := 7

-- Define the expected mean temperature
def expected_mean : Rat := (-6 : Int) / (7 : Int)

-- State the theorem that we need to prove
theorem mean_temperature_correct :
  (temperatures.sum : Rat) / (num_days : Rat) = expected_mean := 
by
  sorry

end mean_temperature_correct_l998_99831


namespace directrix_parabola_l998_99802

theorem directrix_parabola (p : ℝ) (h : 4 * p = 2) : 
  ∃ d : ℝ, d = -p / 2 ∧ d = -1/2 :=
by
  sorry

end directrix_parabola_l998_99802


namespace number_of_players_taking_mathematics_l998_99808

def total_players : ℕ := 25
def players_taking_physics : ℕ := 12
def players_taking_both : ℕ := 5

theorem number_of_players_taking_mathematics :
  total_players - players_taking_physics + players_taking_both = 18 :=
by
  sorry

end number_of_players_taking_mathematics_l998_99808


namespace interest_rate_per_annum_is_four_l998_99868

-- Definitions
def P : ℕ := 300
def t : ℕ := 8
def I : ℤ := P - 204

-- Interest formula
def simple_interest (P : ℕ) (r : ℕ) (t : ℕ) : ℤ := P * r * t / 100

-- Statement to prove
theorem interest_rate_per_annum_is_four :
  ∃ r : ℕ, I = simple_interest P r t ∧ r = 4 :=
by sorry

end interest_rate_per_annum_is_four_l998_99868


namespace largest_possible_sum_l998_99810

-- Define whole numbers
def whole_numbers : Set ℕ := Set.univ

-- Define the given conditions
variables (a b : ℕ)
axiom h1 : a ∈ whole_numbers
axiom h2 : b ∈ whole_numbers
axiom h3 : a * b = 48

-- Prove the largest sum condition
theorem largest_possible_sum : a + b ≤ 49 :=
sorry

end largest_possible_sum_l998_99810


namespace find_m_l998_99873

noncomputable def f (x : ℝ) : ℝ := 2^x - 5

theorem find_m (m : ℝ) (h : f m = 3) : m = 3 := 
by
  sorry

end find_m_l998_99873


namespace arithmetic_sequence_terms_count_l998_99899

theorem arithmetic_sequence_terms_count (a d l : Int) (h1 : a = 20) (h2 : d = -3) (h3 : l = -5) :
  ∃ n : Int, l = a + (n - 1) * d ∧ n = 8 :=
by
  sorry

end arithmetic_sequence_terms_count_l998_99899


namespace number_of_ostriches_l998_99842

theorem number_of_ostriches
    (x y : ℕ)
    (h1 : x + y = 150)
    (h2 : 2 * x + 6 * y = 624) :
    x = 69 :=
by
  -- Proof omitted
  sorry

end number_of_ostriches_l998_99842


namespace find_a2_an_le_2an_next_sum_bounds_l998_99849

variable {a : ℕ → ℝ}
variable (S : ℕ → ℝ)

-- Given conditions
axiom seq_condition (n : ℕ) (h_pos : a n > 0) : 
  a n ^ 2 + a n = 3 * (a (n + 1)) ^ 2 + 2 * a (n + 1)
axiom a1_condition : a 1 = 1

-- Question 1: Prove the value of a2
theorem find_a2 : a 2 = (Real.sqrt 7 - 1) / 3 :=
  sorry

-- Question 2: Prove a_n ≤ 2 * a_{n+1} for any n ∈ N*
theorem an_le_2an_next (n : ℕ) (h_n : n > 0) : a n ≤ 2 * a (n + 1) :=
  sorry

-- Question 3: Prove 2 - 1 / 2^(n - 1) ≤ S_n < 3 for any n ∈ N*
theorem sum_bounds (n : ℕ) (h_n : n > 0) : 
  2 - 1 / 2 ^ (n - 1) ≤ S n ∧ S n < 3 :=
  sorry

end find_a2_an_le_2an_next_sum_bounds_l998_99849


namespace operation_addition_l998_99824

theorem operation_addition (a b c : ℝ) (op : ℝ → ℝ → ℝ)
  (H : ∀ a b c : ℝ, op (op a b) c = a + b + c) :
  ∀ a b : ℝ, op a b = a + b :=
sorry

end operation_addition_l998_99824


namespace graduation_ceremony_l998_99880

theorem graduation_ceremony (teachers administrators graduates chairs : ℕ) 
  (h1 : teachers = 20) 
  (h2 : administrators = teachers / 2) 
  (h3 : graduates = 50) 
  (h4 : chairs = 180) :
  (chairs - (teachers + administrators + graduates)) / graduates = 2 :=
by 
  sorry

end graduation_ceremony_l998_99880


namespace cost_price_of_table_l998_99876

theorem cost_price_of_table (C S : ℝ) (h1 : S = 1.25 * C) (h2 : S = 4800) : C = 3840 := 
by 
  sorry

end cost_price_of_table_l998_99876


namespace intersection_of_A_and_B_l998_99833

noncomputable def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 }
noncomputable def B : Set ℝ := { x | 0 ≤ x }

theorem intersection_of_A_and_B :
  { x | x ∈ A ∧ x ∈ B } = { x | 0 ≤ x ∧ x ≤ 3 } :=
  by sorry

end intersection_of_A_and_B_l998_99833


namespace simplify_and_evaluate_expression_l998_99898

def a : ℚ := 1 / 3
def b : ℚ := -1
def expr : ℚ := 4 * (3 * a^2 * b - a * b^2) - (2 * a * b^2 + 3 * a^2 * b)

theorem simplify_and_evaluate_expression : expr = -3 := 
by
  sorry

end simplify_and_evaluate_expression_l998_99898


namespace JodiMilesFourthWeek_l998_99841

def JodiMilesFirstWeek := 1 * 6
def JodiMilesSecondWeek := 2 * 6
def JodiMilesThirdWeek := 3 * 6
def TotalMilesFirstThreeWeeks := JodiMilesFirstWeek + JodiMilesSecondWeek + JodiMilesThirdWeek
def TotalMilesFourWeeks := 60

def MilesInFourthWeek := TotalMilesFourWeeks - TotalMilesFirstThreeWeeks
def DaysInWeek := 6

theorem JodiMilesFourthWeek : (MilesInFourthWeek / DaysInWeek) = 4 := by
  sorry

end JodiMilesFourthWeek_l998_99841


namespace joggers_difference_l998_99869

-- Define the conditions as per the problem statement
variables (Tyson Alexander Christopher : ℕ)
variable (H1 : Alexander = Tyson + 22)
variable (H2 : Christopher = 20 * Tyson)
variable (H3 : Christopher = 80)

-- The theorem statement to prove Christopher bought 54 more joggers than Alexander
theorem joggers_difference : (Christopher - Alexander) = 54 :=
  sorry

end joggers_difference_l998_99869


namespace smallest_possible_n_l998_99839

theorem smallest_possible_n (n : ℕ) (h1 : n % 6 = 4) (h2 : n % 7 = 3) (h3 : n > 20) : n = 52 := 
sorry

end smallest_possible_n_l998_99839


namespace denomination_of_four_bills_l998_99897

theorem denomination_of_four_bills (X : ℕ) (h1 : 10 * 20 + 8 * 10 + 4 * X = 300) : X = 5 :=
by
  -- proof goes here
  sorry

end denomination_of_four_bills_l998_99897


namespace probability_of_drawing_K_is_2_over_27_l998_99872

-- Define the total number of cards in a standard deck of 54 cards
def total_cards : ℕ := 54

-- Define the number of "K" cards in the standard deck
def num_K_cards : ℕ := 4

-- Define the probability function for drawing a "K"
def probability_drawing_K (total : ℕ) (favorable : ℕ) : ℚ :=
  favorable / total

-- Prove that the probability of drawing a "K" is 2/27
theorem probability_of_drawing_K_is_2_over_27 :
  probability_drawing_K total_cards num_K_cards = 2 / 27 :=
by
  sorry

end probability_of_drawing_K_is_2_over_27_l998_99872


namespace number_of_vans_needed_l998_99859

theorem number_of_vans_needed (capacity_per_van : ℕ) (students : ℕ) (adults : ℕ)
  (h_capacity : capacity_per_van = 9)
  (h_students : students = 40)
  (h_adults : adults = 14) :
  (students + adults + capacity_per_van - 1) / capacity_per_van = 6 := by
  sorry

end number_of_vans_needed_l998_99859


namespace quadratic_expression_result_l998_99806

theorem quadratic_expression_result (x y : ℚ) 
  (h1 : 4 * x + y = 11) 
  (h2 : x + 4 * y = 15) : 
  13 * x^2 + 14 * x * y + 13 * y^2 = 275.2 := 
by 
  sorry

end quadratic_expression_result_l998_99806


namespace walk_to_bus_stop_time_l998_99805

theorem walk_to_bus_stop_time 
  (S T : ℝ)   -- Usual speed and time
  (D : ℝ)        -- Distance to bus stop
  (T'_delay : ℝ := 9)   -- Additional delay in minutes
  (T_coffee : ℝ := 6)   -- Coffee shop time in minutes
  (reduced_speed_factor : ℝ := 4/5)  -- Reduced speed factor
  (h1 : D = S * T)
  (h2 : D = reduced_speed_factor * S * (T + T'_delay - T_coffee)) :
  T = 12 :=
by
  sorry

end walk_to_bus_stop_time_l998_99805


namespace probe_distance_before_refuel_l998_99879

def total_distance : ℕ := 5555555555555
def distance_from_refuel : ℕ := 3333333333333
def distance_before_refuel : ℕ := 2222222222222

theorem probe_distance_before_refuel :
  total_distance - distance_from_refuel = distance_before_refuel := by
  sorry

end probe_distance_before_refuel_l998_99879


namespace alice_spent_19_percent_l998_99840

variable (A : ℝ) (x : ℝ)
variable (h1 : ∃ (B : ℝ), B = 0.9 * A) -- Bob's initial amount in terms of Alice's initial amount
variable (h2 : A - x = 0.81 * A) -- Alice's remaining amount after spending x

theorem alice_spent_19_percent (h1 : ∃ (B : ℝ), B = 0.9 * A) (h2 : A - x = 0.81 * A) : (x / A) * 100 = 19 := by
  sorry

end alice_spent_19_percent_l998_99840


namespace find_A_and_height_l998_99826

noncomputable def triangle_properties (a b : ℝ) (B : ℝ) (cos_B : ℝ) (h : ℝ) :=
  a = 7 ∧ b = 8 ∧ cos_B = -1 / 7 ∧ 
  h = (a : ℝ) * (Real.sqrt (1 - (cos_B)^2)) * (1 : ℝ) / b / 2

theorem find_A_and_height : 
  ∀ (a b : ℝ) (B : ℝ) (cos_B : ℝ) (h : ℝ), 
  triangle_properties a b B cos_B h → 
  ∃ A h1, A = Real.pi / 3 ∧ h1 = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end find_A_and_height_l998_99826


namespace tan_identity_given_condition_l998_99874

variable (α : Real)

theorem tan_identity_given_condition :
  (Real.tan α + 1 / Real.tan α = 9 / 4) →
  (Real.tan α ^ 2 + 1 / (Real.sin α * Real.cos α) + 1 / Real.tan α ^ 2 = 85 / 16) := 
by
  sorry

end tan_identity_given_condition_l998_99874


namespace sum_two_consecutive_sum_three_consecutive_sum_five_consecutive_sum_six_consecutive_l998_99895

theorem sum_two_consecutive : ∃ x : ℕ, 75 = x + (x + 1) := by
  sorry

theorem sum_three_consecutive : ∃ x : ℕ, 75 = x + (x + 1) + (x + 2) := by
  sorry

theorem sum_five_consecutive : ∃ x : ℕ, 75 = x + (x + 1) + (x + 2) + (x + 3) + (x + 4) := by
  sorry

theorem sum_six_consecutive : ∃ x : ℕ, 75 = x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) := by
  sorry

end sum_two_consecutive_sum_three_consecutive_sum_five_consecutive_sum_six_consecutive_l998_99895


namespace how_many_green_towels_l998_99858

-- Define the conditions
def initial_white_towels : ℕ := 21
def towels_given_to_mother : ℕ := 34
def towels_left_after_giving : ℕ := 22

-- Define the statement to prove
theorem how_many_green_towels (G : ℕ) (initial_white : ℕ) (given : ℕ) (left_after : ℕ) :
  initial_white = initial_white_towels →
  given = towels_given_to_mother →
  left_after = towels_left_after_giving →
  (G + initial_white) - given = left_after →
  G = 35 :=
by
  intros
  sorry

end how_many_green_towels_l998_99858


namespace train_additional_time_l998_99809

theorem train_additional_time
  (t : ℝ)  -- time the car takes to reach station B
  (x : ℝ)  -- additional time the train takes compared to the car
  (h₁ : t = 4.5)  -- car takes 4.5 hours to reach station B
  (h₂ : t + (t + x) = 11)  -- combined time for both the car and the train to reach station B
  : x = 2 :=
sorry

end train_additional_time_l998_99809


namespace second_solution_lemonade_is_45_l998_99835

-- Define percentages as real numbers for simplicity
def firstCarbonatedWater : ℝ := 0.80
def firstLemonade : ℝ := 0.20
def secondCarbonatedWater : ℝ := 0.55
def mixturePercentageFirst : ℝ := 0.50
def mixtureCarbonatedWater : ℝ := 0.675

-- The ones that already follow from conditions or trivial definitions:
def secondLemonade : ℝ := 1 - secondCarbonatedWater

-- Define the percentage of carbonated water in mixture, based on given conditions
def mixtureIsCorrect : Prop :=
  mixturePercentageFirst * firstCarbonatedWater + (1 - mixturePercentageFirst) * secondCarbonatedWater = mixtureCarbonatedWater

-- The theorem to prove: second solution's lemonade percentage is 45%
theorem second_solution_lemonade_is_45 :
  mixtureIsCorrect → secondLemonade = 0.45 :=
by
  sorry

end second_solution_lemonade_is_45_l998_99835


namespace emily_furniture_assembly_time_l998_99861

-- Definitions based on conditions
def chairs := 4
def tables := 2
def time_per_piece := 8

-- Proof statement
theorem emily_furniture_assembly_time : (chairs + tables) * time_per_piece = 48 :=
by
  sorry

end emily_furniture_assembly_time_l998_99861


namespace units_digit_base6_product_l998_99843

theorem units_digit_base6_product (a b : ℕ) (h1 : a = 168) (h2 : b = 59) : ((a * b) % 6) = 0 := by
  sorry

end units_digit_base6_product_l998_99843


namespace james_worked_41_hours_l998_99803

theorem james_worked_41_hours (x : ℝ) :
  ∃ (J : ℕ), 
    (24 * x + 12 * 1.5 * x = 40 * x + (J - 40) * 2 * x) ∧ 
    J = 41 := 
by 
  sorry

end james_worked_41_hours_l998_99803


namespace staircase_steps_eq_twelve_l998_99815

theorem staircase_steps_eq_twelve (n : ℕ) :
  (3 * n * (n + 1) / 2 = 270) → (n = 12) :=
by
  intro h
  sorry

end staircase_steps_eq_twelve_l998_99815


namespace not_perfect_square_7p_3p_4_l998_99818

theorem not_perfect_square_7p_3p_4 (p : ℕ) (hp : Nat.Prime p) : ¬∃ a : ℕ, a^2 = 7 * p + 3^p - 4 := 
by
  sorry

end not_perfect_square_7p_3p_4_l998_99818


namespace other_religion_students_l998_99848

theorem other_religion_students (total_students : ℕ) 
  (muslims_percent hindus_percent sikhs_percent christians_percent buddhists_percent : ℝ) 
  (h1 : total_students = 1200) 
  (h2 : muslims_percent = 0.35) 
  (h3 : hindus_percent = 0.25) 
  (h4 : sikhs_percent = 0.15) 
  (h5 : christians_percent = 0.10) 
  (h6 : buddhists_percent = 0.05) : 
  ∃ other_religion_students : ℕ, other_religion_students = 120 :=
by
  sorry

end other_religion_students_l998_99848


namespace area_of_right_angled_isosceles_triangle_l998_99862

-- Definitions
variables {x y : ℝ}
def is_right_angled_isosceles (x y : ℝ) : Prop := y^2 = 2 * x^2
def sum_of_square_areas (x y : ℝ) : Prop := x^2 + x^2 + y^2 = 72

-- Theorem
theorem area_of_right_angled_isosceles_triangle (x y : ℝ) 
  (h1 : is_right_angled_isosceles x y) 
  (h2 : sum_of_square_areas x y) : 
  1/2 * x^2 = 9 :=
sorry

end area_of_right_angled_isosceles_triangle_l998_99862


namespace area_of_triangle_formed_by_intercepts_l998_99875

theorem area_of_triangle_formed_by_intercepts :
  let f (x : ℝ) := (x - 4)^2 * (x + 3)
  let x_intercepts := [-3, 4]
  let y_intercept := 48
  let base := 7
  let height := 48
  let area := (1 / 2 : ℝ) * base * height
  area = 168 :=
by
  sorry

end area_of_triangle_formed_by_intercepts_l998_99875


namespace train_speed_correct_l998_99812

def length_of_train : ℕ := 700
def time_to_cross_pole : ℕ := 20
def expected_speed : ℕ := 35

theorem train_speed_correct : (length_of_train / time_to_cross_pole) = expected_speed := by
  sorry

end train_speed_correct_l998_99812


namespace locus_of_point_R_l998_99864

theorem locus_of_point_R :
  ∀ (P Q O F R : ℝ × ℝ)
    (hP_on_parabola : ∃ x1 y1, P = (x1, y1) ∧ y1^2 = 2 * x1)
    (h_directrix : Q.1 = -1 / 2)
    (hQ : ∃ x1 y1, Q = (x1, y1) ∧ P = (x1, y1))
    (hO : O = (0, 0))
    (hF : F = (1 / 2, 0))
    (h_intersection : ∃ x y, 
      R = (x, y) ∧
      ∃ x1 y1,
      P = (x1, y1) ∧ 
      y1^2 = 2 * x1 ∧
      ∃ (m_OP : ℝ), 
        m_OP = y1 / x1 ∧ 
        y = m_OP * x ∧
      ∃ (m_FQ : ℝ), 
        m_FQ = -y1 ∧
        y = m_FQ * x + y1 * (1 + 3 / 2)),
  R.2^2 = -2 * R.1^2 + R.1 :=
by sorry

end locus_of_point_R_l998_99864


namespace marshmallow_per_smore_l998_99855

theorem marshmallow_per_smore (graham_crackers : ℕ) (initial_marshmallows : ℕ) (additional_marshmallows : ℕ) 
                               (graham_crackers_per_smore : ℕ) :
  graham_crackers = 48 ∧ initial_marshmallows = 6 ∧ additional_marshmallows = 18 ∧ graham_crackers_per_smore = 2 →
  (initial_marshmallows + additional_marshmallows) / (graham_crackers / graham_crackers_per_smore) = 1 :=
by
  intro h
  sorry

end marshmallow_per_smore_l998_99855


namespace polynomial_simplification_l998_99891

theorem polynomial_simplification (x : ℤ) :
  (5 * x ^ 12 + 8 * x ^ 11 + 10 * x ^ 9) + (3 * x ^ 13 + 2 * x ^ 12 + x ^ 11 + 6 * x ^ 9 + 7 * x ^ 5 + 8 * x ^ 2 + 9) =
  3 * x ^ 13 + 7 * x ^ 12 + 9 * x ^ 11 + 16 * x ^ 9 + 7 * x ^ 5 + 8 * x ^ 2 + 9 :=
by
  sorry

end polynomial_simplification_l998_99891


namespace integer_powers_of_reciprocal_sum_l998_99832

variable (x: ℝ)

theorem integer_powers_of_reciprocal_sum (hx : x ≠ 0) (hx_int : ∃ k : ℤ, x + 1/x = k) : ∀ n : ℕ, ∃ k : ℤ, x^n + 1/x^n = k :=
by
  sorry

end integer_powers_of_reciprocal_sum_l998_99832


namespace abs_diff_eq_1point5_l998_99823

theorem abs_diff_eq_1point5 (x y : ℝ)
    (hx : (⌊x⌋ : ℝ) + (y - ⌊y⌋) = 3.7)
    (hy : (x - ⌊x⌋) + (⌊y⌋ : ℝ) = 4.2) :
        |x - y| = 1.5 :=
by
  sorry

end abs_diff_eq_1point5_l998_99823


namespace number_of_scenarios_l998_99882

theorem number_of_scenarios :
  ∃ (count : ℕ), count = 42244 ∧
  (∃ (x1 x2 x3 x4 x5 x6 x7 : ℕ),
    x1 % 7 = 0 ∧ x2 % 7 = 0 ∧ x3 % 7 = 0 ∧ x4 % 7 = 0 ∧
    x5 % 13 = 0 ∧ x6 % 13 = 0 ∧ x7 % 13 = 0 ∧
    x1 + x2 + x3 + x4 + x5 + x6 + x7 = 270) :=
sorry

end number_of_scenarios_l998_99882


namespace sum_of_arithmetic_sequence_l998_99883

theorem sum_of_arithmetic_sequence (S : ℕ → ℕ) (S5 : S 5 = 30) (S10 : S 10 = 110) : S 15 = 240 :=
by
  sorry

end sum_of_arithmetic_sequence_l998_99883


namespace ball_hits_ground_at_t_l998_99884

noncomputable def ball_height (t : ℝ) : ℝ := -6 * t^2 - 10 * t + 56

theorem ball_hits_ground_at_t :
  ∃ t : ℝ, ball_height t = 0 ∧ t = 7 / 3 := by
  sorry

end ball_hits_ground_at_t_l998_99884


namespace find_range_of_a_l998_99887

noncomputable def set_A : Set ℝ := {x | x^2 + 4 * x = 0}

noncomputable def set_B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

theorem find_range_of_a : {a : ℝ | set_B a ⊆ set_A} = {a : ℝ | a < -1} ∪ {1} :=
by
  sorry

end find_range_of_a_l998_99887


namespace proof_prob_boy_pass_all_rounds_proof_prob_girl_pass_all_rounds_proof_xi_distribution_proof_exp_xi_l998_99813

noncomputable def prob_boy_pass_all_rounds : ℚ :=
  (5/6) * (4/5) * (3/4) * (2/3)

noncomputable def prob_girl_pass_all_rounds : ℚ :=
  (4/5) * (3/4) * (2/3) * (1/2)

def prob_xi_distribution : (ℚ × ℚ × ℚ × ℚ × ℚ) :=
  (64/225, 96/225, 52/225, 12/225, 1/225)

def exp_xi : ℚ :=
  (0 * (64/225) + 1 * (96/225) + 2 * (52/225) + 3 * (12/225) + 4 * (1/225))

theorem proof_prob_boy_pass_all_rounds :
  prob_boy_pass_all_rounds = 1/3 :=
by
  sorry

theorem proof_prob_girl_pass_all_rounds :
  prob_girl_pass_all_rounds = 1/5 :=
by
  sorry

theorem proof_xi_distribution :
  prob_xi_distribution = (64/225, 96/225, 52/225, 12/225, 1/225) :=
by
  sorry

theorem proof_exp_xi :
  exp_xi = 16/15 :=
by
  sorry

end proof_prob_boy_pass_all_rounds_proof_prob_girl_pass_all_rounds_proof_xi_distribution_proof_exp_xi_l998_99813


namespace binary_difference_l998_99846

theorem binary_difference (n : ℕ) (b_2 : List ℕ) (x y : ℕ) (h1 : n = 157)
  (h2 : b_2 = [1, 0, 0, 1, 1, 1, 0, 1])
  (hx : x = b_2.count 0)
  (hy : y = b_2.count 1) : y - x = 2 := by
  sorry

end binary_difference_l998_99846


namespace floor_tiling_l998_99856

theorem floor_tiling (n : ℕ) (x : ℕ) (h1 : 6 * x = n^2) : 6 ∣ n := sorry

end floor_tiling_l998_99856


namespace average_of_distinct_numbers_l998_99820

theorem average_of_distinct_numbers (A B C D : ℕ) (hA : A = 1 ∨ A = 3 ∨ A = 5 ∨ A = 7)
                                   (hB : B = 1 ∨ B = 3 ∨ B = 5 ∨ B = 7)
                                   (hC : C = 1 ∨ C = 3 ∨ C = 5 ∨ C = 7)
                                   (hD : D = 1 ∨ D = 3 ∨ D = 5 ∨ D = 7)
                                   (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
    (A + B + C + D) / 4 = 4 := by
  sorry

end average_of_distinct_numbers_l998_99820


namespace mork_tax_rate_l998_99814

theorem mork_tax_rate (M R : ℝ) (h1 : 0.15 = 0.15) (h2 : 4 * M = Mindy_income) (h3 : (R / 100 * M + 0.15 * 4 * M) = 0.21 * 5 * M):
  R = 45 :=
sorry

end mork_tax_rate_l998_99814


namespace smallest_k_correct_l998_99889

noncomputable def smallest_k (n m : ℕ) (hn : 0 < n) (hm : 0 < m ∧ m ≤ 5) : ℕ :=
    6

theorem smallest_k_correct (n : ℕ) (m : ℕ) (hn : 0 < n) (hm : 0 < m ∧ m ≤ 5) :
  64 ^ smallest_k n m hn hm + 32 ^ m > 4 ^ (16 + n) :=
sorry

end smallest_k_correct_l998_99889


namespace norm_squared_sum_l998_99892

variables (p q : ℝ × ℝ)
def n : ℝ × ℝ := (4, -2)
variables (h_midpoint : n = ((p.1 + q.1) / 2, (p.2 + q.2) / 2))
variables (h_dot_product : p.1 * q.1 + p.2 * q.2 = 12)

theorem norm_squared_sum : (p.1 ^ 2 + p.2 ^ 2) + (q.1 ^ 2 + q.2 ^ 2) = 56 :=
by
  sorry

end norm_squared_sum_l998_99892


namespace fraction_subtraction_l998_99857

theorem fraction_subtraction : (3 + 5 + 7) / (2 + 4 + 6) - (2 - 4 + 6) / (3 - 5 + 7) = 9 / 20 :=
by
  sorry

end fraction_subtraction_l998_99857


namespace cross_product_u_v_l998_99877

-- Define the vectors u and v
def u : ℝ × ℝ × ℝ := (3, -4, 7)
def v : ℝ × ℝ × ℝ := (2, 5, -3)

-- Define the cross product function
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1, a.2.2 * b.1 - a.1 * b.2.2, a.1 * b.2.1 - a.2.1 * b.1)

-- State the theorem to be proved
theorem cross_product_u_v : cross_product u v = (-23, 23, 23) :=
  sorry

end cross_product_u_v_l998_99877


namespace total_cost_of_car_rental_l998_99828

theorem total_cost_of_car_rental :
  ∀ (rental_cost_per_day mileage_cost_per_mile : ℝ) (days rented : ℕ) (miles_driven : ℕ),
  rental_cost_per_day = 30 →
  mileage_cost_per_mile = 0.25 →
  rented = 5 →
  miles_driven = 500 →
  rental_cost_per_day * rented + mileage_cost_per_mile * miles_driven = 275 := by
  sorry

end total_cost_of_car_rental_l998_99828


namespace flower_garden_width_l998_99817

-- Define the conditions
def gardenArea : ℝ := 143.2
def gardenLength : ℝ := 4
def gardenWidth : ℝ := 35.8

-- The proof statement (question to answer)
theorem flower_garden_width :
    gardenWidth = gardenArea / gardenLength :=
by 
  sorry

end flower_garden_width_l998_99817


namespace radius_circumcircle_l998_99866

variables (R1 R2 R3 : ℝ)
variables (d : ℝ)
variables (R : ℝ)

noncomputable def sum_radii := R1 + R2 = 11
noncomputable def distance_centers := d = 5 * Real.sqrt 17
noncomputable def radius_third_sphere := R3 = 8
noncomputable def touching := R1 + R2 + 2 * R3 = d

theorem radius_circumcircle :
  R = 5 * Real.sqrt 17 / 2 :=
  by
  -- Use conditions here if necessary
  sorry

end radius_circumcircle_l998_99866


namespace simplify_expression_l998_99822

theorem simplify_expression (x : ℝ) (h1 : x^2 - 4*x + 3 ≠ 0) (h2 : x^2 - 6*x + 9 ≠ 0) (h3 : x^2 - 3*x + 2 ≠ 0) (h4 : x^2 - 4*x + 4 ≠ 0) :
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / (x^2 - 3*x + 2) / (x^2 - 4*x + 4) = (x-2) / (x-3) :=
by {
  sorry
}

end simplify_expression_l998_99822


namespace gcd_polynomial_example_l998_99804

theorem gcd_polynomial_example (b : ℤ) (h : ∃ k : ℤ, b = 2 * 1177 * k) :
  Int.gcd (3 * b^2 + 34 * b + 76) (b + 14) = 2 :=
by
  sorry

end gcd_polynomial_example_l998_99804


namespace problem_l998_99845

theorem problem (x y : ℕ) (hxpos : 0 < x ∧ x < 20) (hypos : 0 < y ∧ y < 20) (h : x + y + x * y = 119) : 
  x + y = 24 ∨ x + y = 21 ∨ x + y = 20 :=
by sorry

end problem_l998_99845


namespace part_1_part_2_l998_99890

-- Part (Ⅰ)
def f (x : ℝ) (m : ℝ) : ℝ := 4 * x^2 + (m - 2) * x + 1

theorem part_1 (m : ℝ) : (∀ x : ℝ, ¬ f x m < 0) ↔ (-2 ≤ m ∧ m ≤ 6) :=
by sorry

-- Part (Ⅱ)
theorem part_2 (m : ℝ) (h_even : ∀ ⦃x : ℝ⦄, f x m = f (-x) m) :
  (m = 2) → 
  ((∀ x : ℝ, x ≤ 0 → f x 2 ≥ f 0 2) ∧ (∀ x : ℝ, x ≥ 0 → f x 2 ≥ f 0 2)) :=
by sorry

end part_1_part_2_l998_99890


namespace percentage_of_girls_taking_lunch_l998_99867

theorem percentage_of_girls_taking_lunch 
  (total_students : ℕ)
  (boys_ratio girls_ratio : ℕ)
  (boys_to_girls_ratio : boys_ratio + girls_ratio = 10)
  (boys : ℕ)
  (girls : ℕ)
  (boys_calc : boys = (boys_ratio * total_students) / 10)
  (girls_calc : girls = (girls_ratio * total_students) / 10)
  (boys_lunch_percentage : ℕ)
  (boys_lunch : ℕ)
  (boys_lunch_calc : boys_lunch = (boys_lunch_percentage * boys) / 100)
  (total_lunch_percentage : ℕ)
  (total_lunch : ℕ)
  (total_lunch_calc : total_lunch = (total_lunch_percentage * total_students) / 100)
  (girls_lunch : ℕ)
  (girls_lunch_calc : girls_lunch = total_lunch - boys_lunch) :
  ((girls_lunch * 100) / girls) = 40 :=
by 
  -- The proof can be filled in here
  sorry

end percentage_of_girls_taking_lunch_l998_99867


namespace sequence_of_numbers_exists_l998_99838

theorem sequence_of_numbers_exists :
  ∃ (a b : ℤ), (a + 2 * b > 0) ∧ (7 * a + 13 * b < 0) :=
sorry

end sequence_of_numbers_exists_l998_99838


namespace p1a_p1b_l998_99847

theorem p1a (m : ℕ) (hm : m > 1) : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 - y^2 = m^3 := by
  sorry  -- Proof is omitted

theorem p1b : ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^6 = y^2 + 127 ∧ x = 4 ∧ y = 63 := by
  sorry  -- Proof is omitted

end p1a_p1b_l998_99847


namespace smaller_number_l998_99816

theorem smaller_number (x y : ℤ) (h1 : x + y = 79) (h2 : x - y = 15) : y = 32 := by
  sorry

end smaller_number_l998_99816


namespace right_triangle_roots_l998_99836

theorem right_triangle_roots (α β : ℝ) (k : ℕ) (h_triangle : (α^2 + β^2 = 100) ∧ (α + β = 14) ∧ (α * β = 4 * k - 4)) : k = 13 :=
sorry

end right_triangle_roots_l998_99836


namespace Nina_can_buy_8_widgets_at_reduced_cost_l998_99888

def money_Nina_has : ℕ := 48
def widgets_she_can_buy_initially : ℕ := 6
def reduction_per_widget : ℕ := 2

theorem Nina_can_buy_8_widgets_at_reduced_cost :
  let initial_cost_per_widget := money_Nina_has / widgets_she_can_buy_initially
  let reduced_cost_per_widget := initial_cost_per_widget - reduction_per_widget
  money_Nina_has / reduced_cost_per_widget = 8 :=
by
  sorry

end Nina_can_buy_8_widgets_at_reduced_cost_l998_99888


namespace find_b_if_continuous_l998_99854

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x ≤ 2 then 5 * x^2 + 4 else b * x + 1

theorem find_b_if_continuous (b : ℝ) : 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 2) < δ → abs (f x b - f 2 b) < ε) ↔ b = 23 / 2 :=
by
  sorry

end find_b_if_continuous_l998_99854


namespace cost_of_child_ticket_l998_99807

-- Define the conditions
def adult_ticket_cost : ℕ := 60
def total_people : ℕ := 280
def total_collected_dollars : ℕ := 140
def total_collected_cents : ℕ := total_collected_dollars * 100
def children_attended : ℕ := 80
def adults_attended : ℕ := total_people - children_attended
def total_collected_from_adults : ℕ := adults_attended * adult_ticket_cost

-- State the theorem to prove the cost of a child ticket
theorem cost_of_child_ticket (x : ℕ) :
  total_collected_from_adults + children_attended * x = total_collected_cents →
  x = 25 :=
by
  sorry

end cost_of_child_ticket_l998_99807
