import Mathlib

namespace expansion_contains_no_x2_l193_193050

theorem expansion_contains_no_x2 (n : ℕ) (h1 : 5 ≤ n ∧ n ≤ 8) :
  ¬ (∃ k, (x + 1)^2 * (x + 1 / x^3)^n = k * x^2) → n = 7 :=
sorry

end expansion_contains_no_x2_l193_193050


namespace article_cost_price_l193_193695

theorem article_cost_price (SP : ℝ) (CP : ℝ) (h1 : SP = 455) (h2 : SP = CP + 0.3 * CP) : CP = 350 :=
by sorry

end article_cost_price_l193_193695


namespace factorization_difference_of_squares_l193_193217

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l193_193217


namespace average_mark_of_excluded_students_l193_193807

theorem average_mark_of_excluded_students (N A A_remaining N_excluded N_remaining T T_remaining T_excluded A_excluded : ℝ)
  (hN : N = 33) 
  (hA : A = 90) 
  (hA_remaining : A_remaining = 95)
  (hN_excluded : N_excluded = 3) 
  (hN_remaining : N_remaining = N - N_excluded) 
  (hT : T = N * A) 
  (hT_remaining : T_remaining = N_remaining * A_remaining) 
  (hT_eq : T = T_excluded + T_remaining) : 
  A_excluded = T_excluded / N_excluded :=
by
  have hTN : N = 33 := hN
  have hTA : A = 90 := hA
  have hTAR : A_remaining = 95 := hA_remaining
  have hTN_excluded : N_excluded = 3 := hN_excluded
  have hNrem : N_remaining = N - N_excluded := hN_remaining
  have hT_sum : T = N * A := hT
  have hTRem : T_remaining = N_remaining * A_remaining := hT_remaining
  have h_sum_eq : T = T_excluded + T_remaining := hT_eq
  sorry -- proof yet to be constructed

end average_mark_of_excluded_students_l193_193807


namespace smallest_n_terminating_decimal_l193_193834

-- Define the given condition: n + 150 must be expressible as 2^a * 5^b.
def has_terminating_decimal_property (n : ℕ) := ∃ a b : ℕ, n + 150 = 2^a * 5^b

-- We want to prove that the smallest n satisfying the property is 50.
theorem smallest_n_terminating_decimal :
  (∀ n : ℕ, n > 0 ∧ has_terminating_decimal_property n → n ≥ 50) ∧ (has_terminating_decimal_property 50) :=
by
  sorry

end smallest_n_terminating_decimal_l193_193834


namespace layla_more_points_than_nahima_l193_193788

theorem layla_more_points_than_nahima (layla_points : ℕ) (total_points : ℕ) (h1 : layla_points = 70) (h2 : total_points = 112) :
  layla_points - (total_points - layla_points) = 28 :=
by
  sorry

end layla_more_points_than_nahima_l193_193788


namespace factorization_of_x_squared_minus_one_l193_193245

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l193_193245


namespace population_growth_rate_l193_193814

-- Define initial and final population
def initial_population : ℕ := 240
def final_population : ℕ := 264

-- Define the formula for calculating population increase rate
def population_increase_rate (P_i P_f : ℕ) : ℕ :=
  ((P_f - P_i) * 100) / P_i

-- State the theorem
theorem population_growth_rate :
  population_increase_rate initial_population final_population = 10 := by
  sorry

end population_growth_rate_l193_193814


namespace find_tangent_line_equation_l193_193885

-- Define the curve as a function
def curve (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 4 * x

-- Define the point of tangency
def P : ℝ × ℝ := (-1, 3)

-- Define the slope of the tangent line at point P
def slope_at_P : ℝ := curve_derivative P.1

-- Define the expected equation of the tangent line
def tangent_line (x y : ℝ) : Prop := 4 * x + y + 1 = 0

-- The theorem to prove that the tangent line at point P has the expected equation
theorem find_tangent_line_equation : 
  tangent_line P.1 (curve P.1) :=
  sorry

end find_tangent_line_equation_l193_193885


namespace factorization_difference_of_squares_l193_193215

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l193_193215


namespace sum_of_values_l193_193108

def f (x : Int) : Int := Int.natAbs x - 3
def g (x : Int) : Int := -x

def fogof (x : Int) : Int := f (g (f x))

theorem sum_of_values :
  (fogof (-5)) + (fogof (-4)) + (fogof (-3)) + (fogof (-2)) + (fogof (-1)) + (fogof 0) + (fogof 1) + (fogof 2) + (fogof 3) + (fogof 4) + (fogof 5) = -17 :=
by
  sorry

end sum_of_values_l193_193108


namespace union_example_l193_193081

theorem union_example (P Q : Set ℕ) (hP : P = {1, 2, 3, 4}) (hQ : Q = {2, 4}) :
  P ∪ Q = {1, 2, 3, 4} :=
by
  sorry

end union_example_l193_193081


namespace bucket_volume_l193_193805

theorem bucket_volume :
  ∃ (V : ℝ), -- The total volume of the bucket
    (∀ (rate_A rate_B rate_combined : ℝ),
      rate_A = 3 ∧ 
      rate_B = V / 60 ∧ 
      rate_combined = V / 10 ∧ 
      rate_A + rate_B = rate_combined) →
    V = 36 :=
by
  sorry

end bucket_volume_l193_193805


namespace factorize_difference_of_squares_l193_193307

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l193_193307


namespace max_area_proof_l193_193042

-- Define the original curve
def original_curve (x : ℝ) : ℝ := x^2 + x - 2

-- Reflective symmetry curve about point (p, 2p)
def transformed_curve (p x : ℝ) : ℝ := -x^2 + (4 * p + 1) * x - 4 * p^2 + 2 * p + 2

-- Intersection conditions
def intersecting_curves (p x : ℝ) : Prop :=
original_curve x = transformed_curve p x

-- Range for valid p values
def valid_p (p : ℝ) : Prop := -1 ≤ p ∧ p ≤ 2

-- Prove the problem statement which involves ensuring the curves intersect in the range
theorem max_area_proof :
  ∀ (p : ℝ), valid_p p → ∀ (x : ℝ), intersecting_curves p x →
  ∃ (A : ℝ), A = abs (original_curve x - transformed_curve p x) :=
by
  intros p hp x hx
  sorry

end max_area_proof_l193_193042


namespace Gwen_remaining_homework_l193_193361

def initial_problems_math := 18
def completed_problems_math := 12
def remaining_problems_math := initial_problems_math - completed_problems_math

def initial_problems_science := 11
def completed_problems_science := 6
def remaining_problems_science := initial_problems_science - completed_problems_science

def initial_questions_history := 15
def completed_questions_history := 10
def remaining_questions_history := initial_questions_history - completed_questions_history

def initial_questions_english := 7
def completed_questions_english := 4
def remaining_questions_english := initial_questions_english - completed_questions_english

def total_remaining_problems := remaining_problems_math 
                               + remaining_problems_science 
                               + remaining_questions_history 
                               + remaining_questions_english

theorem Gwen_remaining_homework : total_remaining_problems = 19 :=
by
  sorry

end Gwen_remaining_homework_l193_193361


namespace problem1_problem2_l193_193359

-- Given conditions
def A : Set ℝ := { x | x^2 - 2 * x - 15 > 0 }
def B : Set ℝ := { x | x < 6 }
def p (m : ℝ) : Prop := m ∈ A
def q (m : ℝ) : Prop := m ∈ B

-- Statements to prove
theorem problem1 (m : ℝ) : p m → m ∈ { x | x < -3 } ∪ { x | x > 5 } :=
sorry

theorem problem2 (m : ℝ) : (p m ∨ q m) ∧ (p m ∧ q m) → m ∈ { x | x < -3 } :=
sorry

end problem1_problem2_l193_193359


namespace factorization_difference_of_squares_l193_193219

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l193_193219


namespace choose_numbers_ways_l193_193928

theorem choose_numbers_ways : 
  ∃ (a_1 a_2 a_3 : ℕ) (S : Finset ℕ), 
  S = (Finset.range 15).erase 0 ∧ 
  a_1 ∈ S ∧ a_2 ∈ S ∧ a_3 ∈ S ∧
  a_1 < a_2 ∧ a_2 < a_3 ∧
  a_2 - a_1 ≥ 3 ∧ a_3 - a_2 ≥ 3 ∧ 
  (S.filter (λ (n : ℕ), ∃ b_1 b_2 b_3 : ℕ, (b_1 = n) ∨ (b_2 = n) ∨ (b_3 = n)
                                        ∧ b_1 < b_2 ∧ b_2 < b_3 
                                        ∧ b_2 ≥ b_1 + 3 ∧ b_3 ≥ b_2 + 3)).card = 120 :=
by sorry

end choose_numbers_ways_l193_193928


namespace factorize_x_squared_minus_1_l193_193276

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l193_193276


namespace common_ratio_eq_l193_193913

theorem common_ratio_eq (a : ℕ → ℚ) (q : ℚ) (h_geom : ∀ n : ℕ, a (n + 1) = q * a n)
  (h_arith : 2 * a 2 = 2 * a 1 + 4 * a 0) (a1_ne_zero : a 0 ≠ 0) :
  q = 2 ∨ q = -1 :=
by
  sorry

end common_ratio_eq_l193_193913


namespace maximum_sum_of_composites_l193_193057

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ n = a * b

def pairwise_coprime (A B C : ℕ) : Prop :=
  Nat.gcd A B = 1 ∧ Nat.gcd A C = 1 ∧ Nat.gcd B C = 1

theorem maximum_sum_of_composites (A B C : ℕ)
  (hA : is_composite A) (hB : is_composite B) (hC : is_composite C)
  (h_pairwise : pairwise_coprime A B C)
  (h_prod_eq : A * B * C = 11011 * 28) :
  A + B + C = 1626 := 
sorry

end maximum_sum_of_composites_l193_193057


namespace range_of_x_if_p_and_q_range_of_a_if_not_p_sufficient_for_not_q_l193_193796

variable (x a : ℝ)

-- Condition p
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a > 0

-- Condition q
def q (x : ℝ) : Prop :=
  (x^2 - x - 6 ≤ 0) ∧ (x^2 + 3*x - 10 > 0)

-- Proof problem for question (1)
theorem range_of_x_if_p_and_q (h1 : p x 1) (h2 : q x) : 2 < x ∧ x < 3 :=
  sorry

-- Proof problem for question (2)
theorem range_of_a_if_not_p_sufficient_for_not_q (h : (¬p x a) → (¬q x)) : 1 < a ∧ a ≤ 2 :=
  sorry

end range_of_x_if_p_and_q_range_of_a_if_not_p_sufficient_for_not_q_l193_193796


namespace no_real_roots_l193_193545

-- Define the polynomial Q(x)
def Q (x : ℝ) : ℝ := x^6 - 3 * x^5 + 6 * x^4 - 6 * x^3 - x + 8

-- The problem can be stated as proving that Q(x) has no real roots
theorem no_real_roots : ∀ x : ℝ, Q x ≠ 0 := by
  sorry

end no_real_roots_l193_193545


namespace avg_GPA_is_93_l193_193631

def avg_GPA_school (GPA_6th GPA_8th : ℕ) (GPA_diff : ℕ) : ℕ :=
  (GPA_6th + (GPA_6th + GPA_diff) + GPA_8th) / 3

theorem avg_GPA_is_93 :
  avg_GPA_school 93 91 2 = 93 :=
by
  -- The proof can be handled here 
  sorry

end avg_GPA_is_93_l193_193631


namespace new_bucket_capacity_l193_193510

theorem new_bucket_capacity (init_buckets : ℕ) (init_capacity : ℕ) (new_buckets : ℕ) (total_volume : ℕ) :
  init_buckets * init_capacity = total_volume →
  new_buckets * 9 = total_volume →
  9 = total_volume / new_buckets :=
by
  intros h₁ h₂
  sorry

end new_bucket_capacity_l193_193510


namespace steak_knife_cost_l193_193180

theorem steak_knife_cost :
  ∀ (sets : ℕ) (knives_per_set : ℕ) (cost_per_set : ℕ),
    sets = 2 →
    knives_per_set = 4 →
    cost_per_set = 80 →
    (cost_per_set * sets) / (knives_per_set * sets) = 20 :=
by
  intros sets knives_per_set cost_per_set h_sets h_knives_per_set h_cost_per_set
  rw [h_sets, h_knives_per_set, h_cost_per_set]
  norm_num
  sorry

end steak_knife_cost_l193_193180


namespace original_number_increased_by_40_percent_l193_193145

theorem original_number_increased_by_40_percent (x : ℝ) (h : 1.40 * x = 700) : x = 500 :=
by
  sorry

end original_number_increased_by_40_percent_l193_193145


namespace factorize_difference_of_squares_l193_193222

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l193_193222


namespace min_fence_length_l193_193021

theorem min_fence_length (x : ℝ) (h : x > 0) (A : x * (64 / x) = 64) : 2 * (x + 64 / x) ≥ 32 :=
by
  have t := (2 * (x + 64 / x)) 
  sorry -- Proof omitted, only statement provided as per instructions

end min_fence_length_l193_193021


namespace roger_final_money_l193_193846

variable (initial_money : ℕ)
variable (spent_money : ℕ)
variable (received_money : ℕ)

theorem roger_final_money (h1 : initial_money = 45) (h2 : spent_money = 20) (h3 : received_money = 46) :
  (initial_money - spent_money + received_money) = 71 :=
by
  sorry

end roger_final_money_l193_193846


namespace goldfish_count_equal_in_6_months_l193_193533

def initial_goldfish_brent : ℕ := 3
def initial_goldfish_gretel : ℕ := 243

def goldfish_brent (n : ℕ) : ℕ := initial_goldfish_brent * 4^n
def goldfish_gretel (n : ℕ) : ℕ := initial_goldfish_gretel * 3^n

theorem goldfish_count_equal_in_6_months : 
  (∃ n : ℕ, goldfish_brent n = goldfish_gretel n) ↔ n = 6 :=
by
  sorry

end goldfish_count_equal_in_6_months_l193_193533


namespace percent_not_covering_politics_l193_193025

-- Definitions based on the conditions
def total_reporters : ℕ := 100
def local_politics_reporters : ℕ := 28
def percent_cover_local_politics : ℚ := 0.7

-- To be proved
theorem percent_not_covering_politics :
  let politics_reporters := local_politics_reporters / percent_cover_local_politics 
  (total_reporters - politics_reporters) / total_reporters = 0.6 := 
by
  sorry

end percent_not_covering_politics_l193_193025


namespace rectangle_area_l193_193768

theorem rectangle_area (length width : ℝ) 
  (h1 : width = 0.9 * length) 
  (h2 : length = 15) : 
  length * width = 202.5 := 
by
  sorry

end rectangle_area_l193_193768


namespace trisha_initial_money_l193_193830

-- Definitions based on conditions
def spent_on_meat : ℕ := 17
def spent_on_chicken : ℕ := 22
def spent_on_veggies : ℕ := 43
def spent_on_eggs : ℕ := 5
def spent_on_dog_food : ℕ := 45
def spent_on_cat_food : ℕ := 18
def money_left : ℕ := 35

-- Total amount spent
def total_spent : ℕ :=
  spent_on_meat + spent_on_chicken + spent_on_veggies + spent_on_eggs + spent_on_dog_food + spent_on_cat_food

-- The target amount she brought with her at the beginning
def total_money_brought : ℕ :=
  total_spent + money_left

-- The theorem to be proved
theorem trisha_initial_money :
  total_money_brought = 185 :=
by
  sorry

end trisha_initial_money_l193_193830


namespace num_arithmetic_sequences_l193_193903

-- Definitions of the arithmetic sequence conditions
def is_arithmetic_sequence (a d n : ℕ) : Prop :=
  0 ≤ a ∧ 0 ≤ d ∧ n ≥ 3 ∧ 
  (∃ k : ℕ, k = 97 ∧ 
  (n * (2 * a + (n - 1) * d) = 2 * k ^ 2)) 

-- Prove that there are exactly 4 such sequences
theorem num_arithmetic_sequences : 
  ∃ (n : ℕ) (a d : ℕ), 
  is_arithmetic_sequence a d n ∧ 
  (n * (2 * a + (n - 1) * d) = 2 * 97^2) ∧ (
    (n = 97 ∧ ((a = 97 ∧ d = 0) ∨ (a = 49 ∧ d = 1) ∨ (a = 1 ∧ d = 2))) ∨
    (n = 97^2 ∧ a = 1 ∧ d = 0)
  ) :=
sorry

end num_arithmetic_sequences_l193_193903


namespace polar_circle_equation_l193_193509

theorem polar_circle_equation (ρ θ : ℝ) (O pole : ℝ) (eq_line : ρ * Real.cos θ + ρ * Real.sin θ = 2) :
  (∃ ρ, ρ = 2 * Real.cos θ) :=
sorry

end polar_circle_equation_l193_193509


namespace fraction_second_box_filled_l193_193832

theorem fraction_second_box_filled :
  let capacity_first_box := 80
  let fraction_filled_first_box := (3 / 4 : ℚ)
  let capacity_second_box := 50
  let total_oranges := 90
  let oranges_first_box := capacity_first_box * fraction_filled_first_box
  let fraction_filled_second_box := (3 / 5 : ℚ)
  oranges_first_box + capacity_second_box * fraction_filled_second_box = total_oranges :=
by
  let capacity_first_box := 80
  let fraction_filled_first_box := (3 / 4 : ℚ)
  let capacity_second_box := 50
  let total_oranges := 90
  let oranges_first_box := capacity_first_box * fraction_filled_first_box
  let fraction_filled_second_box := (3 / 5 : ℚ)
  have oranges_first_box_correct : oranges_first_box = 60 := by
    -- Proof to be filled in here
    sorry
  oranges_first_box_correct ▸ rfl
  sorry

end fraction_second_box_filled_l193_193832


namespace find_xyz_values_l193_193056

theorem find_xyz_values (x y z : ℝ) (h₁ : x + y + z = Real.pi) (h₂ : x ≥ 0) (h₃ : y ≥ 0) (h₄ : z ≥ 0) :
    (x = Real.pi ∧ y = 0 ∧ z = 0) ∨
    (x = 0 ∧ y = Real.pi ∧ z = 0) ∨
    (x = 0 ∧ y = 0 ∧ z = Real.pi) ∨
    (x = Real.pi / 6 ∧ y = Real.pi / 3 ∧ z = Real.pi / 2) :=
sorry

end find_xyz_values_l193_193056


namespace retailer_initial_thought_profit_percentage_l193_193710

/-
  An uneducated retailer marks all his goods at 60% above the cost price and thinking that he will still make some profit, 
  offers a discount of 25% on the marked price. 
  His actual profit on the sales is 20.000000000000018%. 
  Prove that the profit percentage the retailer initially thought he would make is 60%.
-/

theorem retailer_initial_thought_profit_percentage
  (cost_price marked_price selling_price : ℝ)
  (h1 : marked_price = cost_price + 0.6 * cost_price)
  (h2 : selling_price = marked_price - 0.25 * marked_price)
  (h3 : selling_price - cost_price = 0.20000000000000018 * cost_price) :
  0.6 * 100 = 60 := by
  sorry

end retailer_initial_thought_profit_percentage_l193_193710


namespace factorization_difference_of_squares_l193_193214

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l193_193214


namespace size_of_angle_B_length_of_side_b_and_area_l193_193069

-- Given problem conditions
variables (A B C : ℝ) (a b c : ℝ)
variables (h1 : a < b) (h2 : b < c) (h3 : a / Real.sin A = 2 * b / Real.sqrt 3)

-- Prove that B = π / 3
theorem size_of_angle_B : B = Real.pi / 3 := 
sorry

-- Additional conditions for part (2)
variables (h4 : a = 2) (h5 : c = 3) (h6 : Real.cos B = 1 / 2)

-- Prove b = √7 and the area of triangle ABC
theorem length_of_side_b_and_area :
  b = Real.sqrt 7 ∧ 1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 2 :=
sorry

end size_of_angle_B_length_of_side_b_and_area_l193_193069


namespace sandwiches_bought_is_2_l193_193835

-- The given costs and totals
def sandwich_cost : ℝ := 3.49
def soda_cost : ℝ := 0.87
def total_cost : ℝ := 10.46
def sodas_bought : ℕ := 4

-- We need to prove that the number of sandwiches bought, S, is 2
theorem sandwiches_bought_is_2 (S : ℕ) :
  sandwich_cost * S + soda_cost * sodas_bought = total_cost → S = 2 :=
by
  intros h
  sorry

end sandwiches_bought_is_2_l193_193835


namespace factorize_x_squared_minus_one_l193_193312

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l193_193312


namespace even_function_a_value_l193_193771

theorem even_function_a_value (a : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = (x + 1) * (x - a))
  (h_even : ∀ x, f x = f (-x)) : a = -1 :=
by
  sorry

end even_function_a_value_l193_193771


namespace exterior_angle_BAC_l193_193149

theorem exterior_angle_BAC 
    (interior_angle_nonagon : ℕ → ℚ) 
    (angle_CAD_angle_BAD : ℚ → ℚ → ℚ)
    (exterior_angle_formula : ℚ → ℚ) :
  (interior_angle_nonagon 9 = 140) ∧ 
  (angle_CAD_angle_BAD 90 140 = 230) ∧ 
  (exterior_angle_formula 230 = 130) := 
sorry

end exterior_angle_BAC_l193_193149


namespace total_photos_l193_193092

def initial_photos : ℕ := 100
def photos_first_week : ℕ := 50
def photos_second_week : ℕ := 2 * photos_first_week
def photos_third_and_fourth_weeks : ℕ := 80

theorem total_photos (initial_photos photos_first_week photos_second_week photos_third_and_fourth_weeks : ℕ) :
  initial_photos = 100 ∧
  photos_first_week = 50 ∧
  photos_second_week = 2 * photos_first_week ∧
  photos_third_and_fourth_weeks = 80 →
  initial_photos + photos_first_week + photos_second_week + photos_third_and_fourth_weeks = 330 :=
by
  sorry

end total_photos_l193_193092


namespace total_worth_of_travelers_checks_l193_193152

variable (x y : ℕ)

theorem total_worth_of_travelers_checks
  (h1 : x + y = 30)
  (h2 : 50 * (x - 15) + 100 * y = 1050) :
  50 * x + 100 * y = 1800 :=
sorry

end total_worth_of_travelers_checks_l193_193152


namespace find_real_numbers_l193_193032

noncomputable def satisfies_equation (x y z : ℝ) : Prop := 
    x + y + z + 3 / (x - 1) + 3 / (y - 1) + 3 / (z - 1) = 
    2 * (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))

theorem find_real_numbers (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
    satisfies_equation x y z ↔ x = (3 + Real.sqrt 13) / 2 ∧ y = (3 + Real.sqrt 13) / 2 ∧ z = (3 + Real.sqrt 13) / 2 :=
sorry

end find_real_numbers_l193_193032


namespace complex_product_magnitude_l193_193564

-- definitions of complex magnitudes and complex product magnitudes
def magnitude (z : Complex) : Real :=
  match z with
  | ⟨a, b⟩ => Real.sqrt (a * a + b * b)

noncomputable def product_magnitude (z1 z2 : Complex) : Real :=
  magnitude z1 * magnitude z2

-- Given complex numbers
def z1 : Complex := ⟨7, -4⟩
def z2 : Complex := ⟨3, 10⟩

-- The proof statement
theorem complex_product_magnitude : magnitude (z1 * z2) = Real.sqrt 7085 :=
  by 
  sorry

end complex_product_magnitude_l193_193564


namespace factorization_of_x_squared_minus_one_l193_193244

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l193_193244


namespace B_share_is_2400_l193_193155

noncomputable def calculate_B_share (total_profit : ℝ) (x : ℝ) : ℝ :=
  let A_investment_months := 3 * x * 12
  let B_investment_months := x * 6
  let C_investment_months := (3/2) * x * 9
  let D_investment_months := (3/2) * x * 8
  let total_investment_months := A_investment_months + B_investment_months + C_investment_months + D_investment_months
  (B_investment_months / total_investment_months) * total_profit

theorem B_share_is_2400 :
  calculate_B_share 27000 1 = 2400 :=
sorry

end B_share_is_2400_l193_193155


namespace edge_c_eq_3_or_5_l193_193066

noncomputable def a := 7
noncomputable def b := 8
noncomputable def A := Real.pi / 3

theorem edge_c_eq_3_or_5 (c : ℝ) (h : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) : c = 3 ∨ c = 5 :=
by
  sorry

end edge_c_eq_3_or_5_l193_193066


namespace relationship_y1_y2_y3_l193_193352

variable (k x y1 y2 y3 : ℝ)
variable (h1 : k < 0)
variable (h2 : y1 = k / -4)
variable (h3 : y2 = k / 2)
variable (h4 : y3 = k / 3)

theorem relationship_y1_y2_y3 (k x y1 y2 y3 : ℝ) 
  (h1 : k < 0)
  (h2 : y1 = k / -4)
  (h3 : y2 = k / 2)
  (h4 : y3 = k / 3) : 
  y1 > y3 ∧ y3 > y2 := 
by sorry

end relationship_y1_y2_y3_l193_193352


namespace jack_second_half_time_l193_193383

variable (time_half1 time_half2 time_jack_total time_jill_total : ℕ)

theorem jack_second_half_time (h1 : time_half1 = 19) 
                              (h2 : time_jill_total = 32) 
                              (h3 : time_jack_total + 7 = time_jill_total) :
  time_jack_total = time_half1 + time_half2 → time_half2 = 6 := by
  sorry

end jack_second_half_time_l193_193383


namespace part1_part2_l193_193758

-- Definitions of propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m * x + 1 = 0 ∧ x < 0 ∧ (∃ y : ℝ, y ≠ x ∧ y^2 + m * y + 1 = 0 ∧ y < 0)
def q (m : ℝ) : Prop := ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 ≠ 0

-- Lean statement for part 1
theorem part1 (m : ℝ) :
  ¬ ¬ p m → m > 2 :=
sorry

-- Lean statement for part 2
theorem part2 (m : ℝ) :
  (p m ∨ q m) ∧ (¬(p m ∧ q m)) → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
sorry

end part1_part2_l193_193758


namespace largest_n_condition_l193_193888

theorem largest_n_condition :
  ∃ n : ℤ, (∃ m : ℤ, n^2 = (m + 1)^3 - m^3) ∧ ∃ k : ℤ, 2 * n + 99 = k^2 ∧ ∀ x : ℤ, 
  (∃ m' : ℤ, x^2 = (m' + 1)^3 - m'^3) ∧ ∃ k' : ℤ, 2 * x + 99 = k'^2 → x ≤ 289 :=
sorry

end largest_n_condition_l193_193888


namespace rice_on_8th_day_l193_193132

variable (a1 : ℕ) (d : ℕ) (n : ℕ)
variable (rice_per_laborer : ℕ)

def is_arithmetic_sequence (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem rice_on_8th_day (ha1 : a1 = 64) (hd : d = 7) (hr : rice_per_laborer = 3) :
  let a8 := is_arithmetic_sequence a1 d 8
  (a8 * rice_per_laborer = 339) :=
by
  sorry

end rice_on_8th_day_l193_193132


namespace olivia_correct_answers_l193_193931

theorem olivia_correct_answers (c w : ℕ) (h1 : c + w = 15) (h2 : 4 * c - 3 * w = 25) : c = 10 :=
by
  sorry

end olivia_correct_answers_l193_193931


namespace factorize_difference_of_squares_l193_193289

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l193_193289


namespace total_pens_bought_l193_193459

theorem total_pens_bought (r : ℕ) (r_gt_10 : r > 10) (r_divides_357 : 357 % r = 0) (r_divides_441 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
by sorry

end total_pens_bought_l193_193459


namespace find_k_l193_193558

def otimes (a b : ℝ) := a * b + a + b^2

theorem find_k (k : ℝ) (h1 : otimes 1 k = 2) (h2 : 0 < k) :
  k = 1 :=
sorry

end find_k_l193_193558


namespace store_owner_oil_l193_193703

noncomputable def liters_of_oil (volume_per_bottle : ℕ) (number_of_bottles : ℕ) : ℕ :=
  (volume_per_bottle * number_of_bottles) / 1000

theorem store_owner_oil : liters_of_oil 200 20 = 4 := by
  sorry

end store_owner_oil_l193_193703


namespace three_numbers_sum_div_by_three_l193_193379

theorem three_numbers_sum_div_by_three (s : Fin 7 → ℕ) : 
  ∃ (a b c : Fin 7), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (s a + s b + s c) % 3 = 0 := 
sorry

end three_numbers_sum_div_by_three_l193_193379


namespace factorize_x_squared_minus_one_l193_193203

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l193_193203


namespace total_pens_bought_l193_193433

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_bought_l193_193433


namespace isosceles_with_60_eq_angle_is_equilateral_l193_193373

open Real

noncomputable def is_equilateral_triangle (a b c : ℝ) (A B C : ℝ) :=
  A = 60 ∧ B = 60 ∧ C = 60

noncomputable def is_isosceles_triangle (a b c : ℝ) (A B C : ℝ) :=
  (a = b ∨ b = c ∨ c = a) ∧ (A + B + C = 180)

theorem isosceles_with_60_eq_angle_is_equilateral
  (a b c A B C : ℝ)
  (h_iso : is_isosceles_triangle a b c A B C)
  (h_angle : A = 60 ∨ B = 60 ∨ C = 60) :
  is_equilateral_triangle a b c A B C :=
sorry

end isosceles_with_60_eq_angle_is_equilateral_l193_193373


namespace sin_pi_over_4_plus_alpha_l193_193909

open Real

theorem sin_pi_over_4_plus_alpha
  (α : ℝ)
  (hα : 0 < α ∧ α < π)
  (h_tan : tan (α - π / 4) = 1 / 3) :
  sin (π / 4 + α) = 3 * sqrt 10 / 10 :=
sorry

end sin_pi_over_4_plus_alpha_l193_193909


namespace fewer_parking_spaces_on_fourth_level_l193_193694

theorem fewer_parking_spaces_on_fourth_level 
  (spaces_first_level : ℕ) (spaces_second_level : ℕ) (spaces_third_level : ℕ) (spaces_fourth_level : ℕ) 
  (total_spaces_garage : ℕ) (cars_parked : ℕ) 
  (h1 : spaces_first_level = 90)
  (h2 : spaces_second_level = spaces_first_level + 8)
  (h3 : spaces_third_level = spaces_second_level + 12)
  (h4 : total_spaces_garage = 299)
  (h5 : cars_parked = 100)
  (h6 : spaces_first_level + spaces_second_level + spaces_third_level + spaces_fourth_level = total_spaces_garage) :
  spaces_third_level - spaces_fourth_level = 109 := 
by
  sorry

end fewer_parking_spaces_on_fourth_level_l193_193694


namespace problem_statement_l193_193176

def operation (a b : ℝ) := (a + b) ^ 2

theorem problem_statement (x y : ℝ) : operation ((x + y) ^ 2) ((y + x) ^ 2) = 4 * (x + y) ^ 4 :=
by
  sorry

end problem_statement_l193_193176


namespace problem_statement_eq_l193_193041

noncomputable def given_sequence (a : ℝ) (n : ℕ) : ℝ :=
  a^n

noncomputable def Sn (a : ℝ) (n : ℕ) (an : ℝ) : ℝ :=
  (a / (a - 1)) * (an - 1)

noncomputable def bn (a : ℝ) (n : ℕ) : ℝ :=
  2 * (Sn a n (given_sequence a n)) / (given_sequence a n) + 1

noncomputable def cn (a : ℝ) (n : ℕ) : ℝ :=
  (n - 1) * (bn a n)

noncomputable def Tn (a : ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (λ acc k => acc + cn a (k + 1)) 0

theorem problem_statement_eq :
  ∀ (a : ℝ) (n : ℕ), a ≠ 0 → a ≠ 1 →
  (bn a n = (3:ℝ)^n) →
  Tn (1 / 3) n = 3^(n+1) * (2 * n - 3) / 4 + 9 / 4 :=
by
  intros
  sorry

end problem_statement_eq_l193_193041


namespace cards_net_cost_equivalence_l193_193065

-- Define the purchase amount
def purchase_amount : ℝ := 10000

-- Define cashback percentages
def debit_card_cashback : ℝ := 0.01
def credit_card_cashback : ℝ := 0.005

-- Define interest rate for keeping money in the debit account
def interest_rate : ℝ := 0.005

-- A function to calculate the net cost after 1 month using the debit card
def net_cost_debit_card (purchase_amount : ℝ) (cashback_percentage : ℝ) : ℝ :=
  purchase_amount - purchase_amount * cashback_percentage

-- A function to calculate the net cost after 1 month using the credit card
def net_cost_credit_card (purchase_amount : ℝ) (cashback_percentage : ℝ) (interest_rate : ℝ) : ℝ :=
  purchase_amount - purchase_amount * cashback_percentage - purchase_amount * interest_rate

-- Final theorem stating that the net cost using both cards is the same
theorem cards_net_cost_equivalence : 
  net_cost_debit_card purchase_amount debit_card_cashback = 
  net_cost_credit_card purchase_amount credit_card_cashback interest_rate :=
by
  sorry

end cards_net_cost_equivalence_l193_193065


namespace solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l193_193963

theorem solve_quadratic_1 (x : Real) : x^2 - 2 * x - 4 = 0 ↔ (x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5) :=
by
  sorry

theorem solve_quadratic_2 (x : Real) : (x - 1)^2 = 2 * (x - 1) ↔ (x = 1 ∨ x = 3) :=
by
  sorry

theorem solve_quadratic_3 (x : Real) : (x + 1)^2 = 4 * x^2 ↔ (x = 1 ∨ x = -1 / 3) :=
by
  sorry

end solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l193_193963


namespace total_pens_bought_l193_193458

theorem total_pens_bought (r : ℕ) (r_gt_10 : r > 10) (r_divides_357 : 357 % r = 0) (r_divides_441 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
by sorry

end total_pens_bought_l193_193458


namespace factorization_difference_of_squares_l193_193209

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l193_193209


namespace max_value_of_f_l193_193329

def f (x : Real) : Real := 8 * Real.sin x + 15 * Real.cos x

theorem max_value_of_f : ∃ x : Real, f(x) ≤ 17 := by
  sorry

end max_value_of_f_l193_193329


namespace factorize_difference_of_squares_l193_193223

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l193_193223


namespace simplify_expression_l193_193467

theorem simplify_expression (y : ℝ) : y - 3 * (2 + y) + 4 * (2 - y) - 5 * (2 + 3 * y) = -21 * y - 8 :=
by
  sorry

end simplify_expression_l193_193467


namespace total_pens_bought_l193_193428

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_bought_l193_193428


namespace D_72_eq_22_l193_193792

def D(n : ℕ) : ℕ :=
  if n = 72 then 22 else 0 -- the actual function logic should define D properly

theorem D_72_eq_22 : D 72 = 22 :=
  by sorry

end D_72_eq_22_l193_193792


namespace max_servings_l193_193991

/-- To prepare one serving of salad we need:
  - 2 cucumbers
  - 2 tomatoes
  - 75 grams of brynza
  - 1 pepper
  The warehouse has the following quantities:
  - 60 peppers
  - 4200 grams of brynza (4.2 kg)
  - 116 tomatoes
  - 117 cucumbers
  We want to prove the maximum number of salad servings we can make is 56.
-/
theorem max_servings (peppers : ℕ) (brynza : ℕ) (tomatoes : ℕ) (cucumbers : ℕ) 
  (h_peppers : peppers = 60)
  (h_brynza : brynza = 4200)
  (h_tomatoes : tomatoes = 116)
  (h_cucumbers : cucumbers = 117) :
  let servings := min (min (peppers / 1) (brynza / 75)) (min (tomatoes / 2) (cucumbers / 2)) in
  servings = 56 := 
by
  sorry

end max_servings_l193_193991


namespace factorize_x_squared_minus_one_l193_193187

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l193_193187


namespace find_uncertain_mushrooms_l193_193393

variable (total_mushrooms safe_mushrooms poisonous_mushrooms uncertain_mushrooms : ℕ)

-- Conditions
def condition1 := total_mushrooms = 32
def condition2 := safe_mushrooms = 9
def condition3 := poisonous_mushrooms = 2 * safe_mushrooms
def condition4 := total_mushrooms = safe_mushrooms + poisonous_mushrooms + uncertain_mushrooms

-- The theorem to prove
theorem find_uncertain_mushrooms (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : uncertain_mushrooms = 5 :=
sorry

end find_uncertain_mushrooms_l193_193393


namespace problem_1_l193_193676

theorem problem_1 (f : ℝ → ℝ) (hf_mul : ∀ x y : ℝ, f (x * y) = f x + f y) (hf_4 : f 4 = 2) : f (Real.sqrt 2) = 1 / 2 :=
sorry

end problem_1_l193_193676


namespace factorize_x_squared_minus_one_l193_193204

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l193_193204


namespace max_servings_possible_l193_193994

def number_of_servings
  (peppers cucumbers tomatoes : Nat) (brynza : Nat) : Nat :=
  min (peppers) (min (brynza / 75) (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings_possible :
  number_of_servings 60 117 116 4200 = 56 := 
by 
  -- sorry statement allows skipping the proof
  sorry

end max_servings_possible_l193_193994


namespace simplify_and_evaluate_expr_evaluate_at_zero_l193_193100

theorem simplify_and_evaluate_expr (x : ℝ) (hx1 : x ≠ 1) (hx2 : x ≠ 2) :
  (3 / (x - 1) - x - 1) / ((x^2 - 4 * x + 4) / (x - 1)) = (2 + x) / (2 - x) :=
by
  sorry

theorem evaluate_at_zero :
  (2 + 0 : ℝ) / (2 - 0) = 1 :=
by
  norm_num

end simplify_and_evaluate_expr_evaluate_at_zero_l193_193100


namespace smallest_top_block_number_l193_193006

-- Define the pyramid structure and number assignment problem
def block_pyramid : Type := sorry

-- Given conditions:
-- 4 layers, specific numberings, and block support structure.
structure Pyramid :=
  (Layer1 : Fin 16 → ℕ)
  (Layer2 : Fin 9 → ℕ)
  (Layer3 : Fin 4 → ℕ)
  (TopBlock : ℕ)

-- Constraints on block numbers
def is_valid (P : Pyramid) : Prop :=
  -- base layer numbers are from 1 to 16
  (∀ i, 1 ≤ P.Layer1 i ∧ P.Layer1 i ≤ 16) ∧
  -- each above block is the sum of directly underlying neighboring blocks
  (∀ i, P.Layer2 i = P.Layer1 (i * 3) + P.Layer1 (i * 3 + 1) + P.Layer1 (i * 3 + 2)) ∧
  (∀ i, P.Layer3 i = P.Layer2 (i * 3) + P.Layer2 (i * 3 + 1) + P.Layer2 (i * 3 + 2)) ∧
  P.TopBlock = P.Layer3 0 + P.Layer3 1 + P.Layer3 2 + P.Layer3 3

-- Statement of the theorem
theorem smallest_top_block_number : ∃ P : Pyramid, is_valid P ∧ P.TopBlock = ComputedValue := sorry

end smallest_top_block_number_l193_193006


namespace determine_swimming_day_l193_193616

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

end determine_swimming_day_l193_193616


namespace question_solution_l193_193578

theorem question_solution
  (f : ℝ → ℝ)
  (h_decreasing : ∀ ⦃x y : ℝ⦄, -3 < x ∧ x < 0 → -3 < y ∧ y < 0 → x < y → f y < f x)
  (h_symmetry : ∀ x : ℝ, f (x) = f (-x + 6)) :
  f (-5) < f (-3/2) ∧ f (-3/2) < f (-7/2) :=
sorry

end question_solution_l193_193578


namespace least_possible_faces_combined_l193_193489

noncomputable def hasValidDiceConfiguration : Prop :=
  ∃ a b : ℕ,
  (∃ s8 s12 s13 : ℕ,
    (s8 = 3) ∧
    (s12 = 4) ∧
    (a ≥ 5 ∧ b = 6 ∧ (a + b = 11) ∧
      (2 * s12 = s8) ∧
      (2 * s8 = s13))
  )

theorem least_possible_faces_combined : hasValidDiceConfiguration :=
  sorry

end least_possible_faces_combined_l193_193489


namespace minimum_value_l193_193891

noncomputable def f (x y : ℝ) : ℝ :=
  (x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2)

theorem minimum_value (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  ∃ a b, x = a + 2 ∧ y = b + 2 ∧ a = sqrt 5 ∧ b = sqrt 5 ∧ f x y = 4 * sqrt 5 + 8 :=
sorry

end minimum_value_l193_193891


namespace total_pens_l193_193425

theorem total_pens (r : ℕ) (h1 : r > 10)
  (h2 : 357 % r = 0)
  (h3 : 441 % r = 0) :
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_l193_193425


namespace confidence_and_inference_l193_193375

-- Suppose K2 represents the observed value of the test statistic
variable {K2 : ℝ}

-- Hypothesis for the confidence level to incorrect inference relationship
theorem confidence_and_inference (K2 : ℝ) := 
  -- Statement (③) describes the relationship correctly and (①) and (②) are incorrect.
  (∀ (c : ℝ), (c = 6.635) → (99 = 100 → 99 = K2)) ∧ 
  -- This statement should be incorrect
  (∀ (c : ℝ), (c = 6.635) → (99% → 99% = P(someone has lung disease))) →
  -- Correct statement
  (∀ (c : ℝ), (c = 3.841) → ((95% = true) ↔ 0.05 = P(incorrect inference)))
sorry

end confidence_and_inference_l193_193375


namespace incorrect_equation_is_wrong_l193_193595

-- Specifications and conditions
def speed_person_a : ℝ := 7
def speed_person_b : ℝ := 6.5
def head_start : ℝ := 5

-- Define the time variable
variable (x : ℝ)

-- The correct equation based on the problem statement
def correct_equation : Prop := speed_person_a * x - head_start = speed_person_b * x

-- The incorrect equation to prove incorrect
def incorrect_equation : Prop := speed_person_b * x = speed_person_a * x - head_start

-- The Lean statement to prove that the incorrect equation is indeed incorrect
theorem incorrect_equation_is_wrong (h : correct_equation x) : ¬ incorrect_equation x := by
  sorry

end incorrect_equation_is_wrong_l193_193595


namespace factorize_expr1_factorize_expr2_l193_193744

-- Problem (1) Statement
theorem factorize_expr1 (x y : ℝ) : 
  -x^5 * y^3 + x^3 * y^5 = -x^3 * y^3 * (x + y) * (x - y) :=
sorry

-- Problem (2) Statement
theorem factorize_expr2 (a : ℝ) : 
  (a^2 + 1)^2 - 4 * a^2 = (a + 1)^2 * (a - 1)^2 :=
sorry

end factorize_expr1_factorize_expr2_l193_193744


namespace total_pens_bought_l193_193455

theorem total_pens_bought (r : ℕ) (r_gt_10 : r > 10) (r_divides_357 : 357 % r = 0) (r_divides_441 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
by sorry

end total_pens_bought_l193_193455


namespace value_of_f_1985_l193_193752

def f : ℝ → ℝ := sorry -- Assuming the existence of f, let ℝ be the type of real numbers

-- Given condition as a hypothesis
axiom functional_eq (x y : ℝ) : f (x + y) = f (x^2) + f (2 * y)

-- The main theorem we want to prove
theorem value_of_f_1985 : f 1985 = 0 :=
by
  sorry

end value_of_f_1985_l193_193752


namespace ellipse_equation_l193_193075

-- Define the conditions for the ellipse
variable (a b : ℝ)
variable (h1 : a > b ∧ b ≥ 1)
variable (h2 : (a^2 - b^2) / a^2 = 3/4)
variable (N : ℝ × ℝ)
variable (h3 : ∃ N : ℝ × ℝ, sqrt ((N.1)^2 + (N.2 - 3)^2) = 4)

-- Define the problem and its statement
theorem ellipse_equation : 
  (∃ a b : ℝ, a > b ∧ b ≥ 1 ∧ (a^2 - b^2) / a^2 = 3/4 ∧ ∀ N : ℝ × ℝ, sqrt((N.1)^2 + (N.2 - 3)^2) = 4 → (4b^2 = 4)) →
  ∃ a b : ℝ, a = 2 ∧ b = 1 ∧ (∀ x y, (x^2) / a^2 + (y^2) / b^2 = 1 ↔ (x^2) / 4 + (y^2) = 1) :=
sorry

end ellipse_equation_l193_193075


namespace find_the_number_l193_193138

-- Define the number we are trying to find
variable (x : ℝ)

-- Define the main condition from the problem
def main_condition : Prop := 0.7 * x - 40 = 30

-- Formalize the goal to prove
theorem find_the_number (h : main_condition x) : x = 100 :=
by
  -- Placeholder for the proof
  sorry

end find_the_number_l193_193138


namespace mitch_earns_correctly_l193_193088

noncomputable def mitch_weekly_earnings : ℝ :=
  let earnings_mw := 3 * (3 * 5 : ℝ) -- Monday to Wednesday
  let earnings_tf := 2 * (6 * 4 : ℝ) -- Thursday and Friday
  let earnings_sat := 4 * 6         -- Saturday
  let earnings_sun := 5 * 8         -- Sunday
  let total_earnings := earnings_mw + earnings_tf + earnings_sat + earnings_sun
  let after_expenses := total_earnings - 25
  let after_tax := after_expenses - 0.10 * after_expenses
  after_tax

theorem mitch_earns_correctly : mitch_weekly_earnings = 118.80 := by
  sorry

end mitch_earns_correctly_l193_193088


namespace smallest_b_for_factors_l193_193336

theorem smallest_b_for_factors (b : ℕ) (h : ∃ r s : ℤ, (x : ℤ) → (x + r) * (x + s) = x^2 + ↑b * x + 2016 ∧ r * s = 2016) :
  b = 90 :=
by
  sorry

end smallest_b_for_factors_l193_193336


namespace evaluate_x_squared_plus_y_squared_l193_193048

theorem evaluate_x_squared_plus_y_squared (x y : ℝ) (h₁ : 3 * x + y = 20) (h₂ : 4 * x + y = 25) :
  x^2 + y^2 = 50 :=
sorry

end evaluate_x_squared_plus_y_squared_l193_193048


namespace max_overlap_l193_193512

variable (A : Type) [Fintype A] [DecidableEq A]
variable (P1 P2 : A → Prop)

theorem max_overlap (hP1 : ∃ X : Finset A, (X.card : ℝ) / Fintype.card A = 0.20 ∧ ∀ a ∈ X, P1 a)
                    (hP2 : ∃ Y : Finset A, (Y.card : ℝ) / Fintype.card A = 0.70 ∧ ∀ a ∈ Y, P2 a) :
  ∃ Z : Finset A, (Z.card : ℝ) / Fintype.card A = 0.20 ∧ ∀ a ∈ Z, P1 a ∧ P2 a :=
sorry

end max_overlap_l193_193512


namespace lightbulbs_working_prob_at_least_5_with_99_percent_confidence_l193_193816

noncomputable def sufficient_lightbulbs (p : ℝ) (k : ℕ) (target_prob : ℝ) : ℕ :=
  let n := 7 -- this is based on our final answer
  have working_prob : (1 - p)^n := 0.95^n
  have non_working_prob : p := 0.05
  nat.run_until_p_ge_k (λ (x : ℝ), x + (nat.choose n k) * (working_prob)^k * (non_working_prob)^(n - k)) target_prob 7

theorem lightbulbs_working_prob_at_least_5_with_99_percent_confidence :
  sufficient_lightbulbs 0.05 5 0.99 = 7 :=
by
  sorry

end lightbulbs_working_prob_at_least_5_with_99_percent_confidence_l193_193816


namespace value_of_ab_l193_193920

theorem value_of_ab (a b : ℤ) (h1 : |a| = 5) (h2 : b = -3) (h3 : a < b) : a * b = 15 :=
by
  sorry

end value_of_ab_l193_193920


namespace binary_arithmetic_l193_193723

def a : ℕ := 0b10110  -- 10110_2
def b : ℕ := 0b1101   -- 1101_2
def c : ℕ := 0b11100  -- 11100_2
def d : ℕ := 0b11101  -- 11101_2
def e : ℕ := 0b101    -- 101_2

theorem binary_arithmetic :
  (a + b - c + d + e) = 0b101101 := by
  sorry

end binary_arithmetic_l193_193723


namespace service_center_location_l193_193519

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

end service_center_location_l193_193519


namespace factorize_difference_of_squares_l193_193290

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l193_193290


namespace solve_eq_nonzero_solve_eq_zero_zero_solve_eq_zero_nonzero_l193_193131

-- Case 1: a ≠ 0
theorem solve_eq_nonzero (a b : ℝ) (h : a ≠ 0) : ∃ x : ℝ, x = -b / a ∧ a * x + b = 0 :=
by
  sorry

-- Case 2: a = 0 and b = 0
theorem solve_eq_zero_zero (a b : ℝ) (h1 : a = 0) (h2 : b = 0) : ∀ x : ℝ, a * x + b = 0 :=
by
  sorry

-- Case 3: a = 0 and b ≠ 0
theorem solve_eq_zero_nonzero (a b : ℝ) (h1 : a = 0) (h2 : b ≠ 0) : ¬ ∃ x : ℝ, a * x + b = 0 :=
by
  sorry

end solve_eq_nonzero_solve_eq_zero_zero_solve_eq_zero_nonzero_l193_193131


namespace total_pens_bought_l193_193453

theorem total_pens_bought (r : ℕ) (r_gt_10 : r > 10) (r_divides_357 : 357 % r = 0) (r_divides_441 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
by sorry

end total_pens_bought_l193_193453


namespace distance_by_which_A_beats_B_l193_193777

noncomputable def speed_of_A : ℝ := 1000 / 192
noncomputable def time_difference : ℝ := 8
noncomputable def distance_beaten : ℝ := speed_of_A * time_difference

theorem distance_by_which_A_beats_B :
  distance_beaten = 41.67 := by
  sorry

end distance_by_which_A_beats_B_l193_193777


namespace trains_cross_time_l193_193999

noncomputable def time_to_cross (len1 len2 speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let speed1_ms := speed1_kmh * (5 / 18)
  let speed2_ms := speed2_kmh * (5 / 18)
  let relative_speed_ms := speed1_ms + speed2_ms
  let total_distance := len1 + len2
  total_distance / relative_speed_ms

theorem trains_cross_time :
  time_to_cross 1500 1000 90 75 = 54.55 := by
  sorry

end trains_cross_time_l193_193999


namespace largest_number_is_C_l193_193130

theorem largest_number_is_C (A B C D E : ℝ) 
  (hA : A = 0.989) 
  (hB : B = 0.9098) 
  (hC : C = 0.9899) 
  (hD : D = 0.9009) 
  (hE : E = 0.9809) : 
  C > A ∧ C > B ∧ C > D ∧ C > E := 
by 
  sorry

end largest_number_is_C_l193_193130


namespace number_of_preferred_groups_l193_193513

def preferred_group_sum_multiple_5 (n : Nat) : Nat := 
  (2^n) * ((2^(4*n) - 1) / 5 + 1) - 1

theorem number_of_preferred_groups :
  preferred_group_sum_multiple_5 400 = 2^400 * (2^1600 - 1) / 5 + 1 - 1 :=
sorry

end number_of_preferred_groups_l193_193513


namespace solve_for_b_l193_193067

theorem solve_for_b (a b c : ℝ) (cosC : ℝ) (h_a : a = 3) (h_c : c = 4) (h_cosC : cosC = -1/4) :
    c^2 = a^2 + b^2 - 2 * a * b * cosC → b = 7 / 2 :=
by 
  intro h_cosine_theorem
  sorry

end solve_for_b_l193_193067


namespace factorize_x_squared_minus_one_l193_193184

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l193_193184


namespace circle_area_solution_l193_193871

def circle_area_problem : Prop :=
  ∀ (x y : ℝ), x^2 + y^2 + 6 * x - 8 * y - 12 = 0 -> ∃ (A : ℝ), A = 37 * Real.pi

theorem circle_area_solution : circle_area_problem :=
by
  sorry

end circle_area_solution_l193_193871


namespace edward_spent_on_books_l193_193562

def money_spent_on_books (initial_amount spent_on_pens amount_left : ℕ) : ℕ :=
  initial_amount - amount_left - spent_on_pens

theorem edward_spent_on_books :
  ∃ (x : ℕ), x = 6 → 
  ∀ {initial_amount spent_on_pens amount_left : ℕ},
    initial_amount = 41 →
    spent_on_pens = 16 →
    amount_left = 19 →
    x = money_spent_on_books initial_amount spent_on_pens amount_left :=
by
  sorry

end edward_spent_on_books_l193_193562


namespace room_length_l193_193642

theorem room_length (L : ℝ) (width : ℝ := 4) (total_cost : ℝ := 20900) (rate : ℝ := 950) :
  L * width = total_cost / rate → L = 5.5 :=
by
  sorry

end room_length_l193_193642


namespace democrats_ratio_l193_193653

noncomputable def F : ℕ := 240
noncomputable def M : ℕ := 480
noncomputable def D_F : ℕ := 120
noncomputable def D_M : ℕ := 120

theorem democrats_ratio (total_participants : ℕ := 720)
  (h1 : F + M = total_participants)
  (h2 : D_F = 120)
  (h3 : D_F = 1/2 * F)
  (h4 : D_M = 1/4 * M)
  (h5 : D_F + D_M = 240)
  (h6 : F + M = 720) : (D_F + D_M) / total_participants = 1 / 3 :=
by
  sorry

end democrats_ratio_l193_193653


namespace factorize_x_squared_minus_one_l193_193197

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l193_193197


namespace perpendicular_bisectors_intersect_at_one_point_l193_193574

-- Define the key geometric concepts
variables {Point : Type*} [MetricSpace Point]

-- Define the given conditions 
variables (A B C M : Point)
variables (h1 : dist M A = dist M B)
variables (h2 : dist M B = dist M C)

-- Define the theorem to be proven
theorem perpendicular_bisectors_intersect_at_one_point :
  dist M A = dist M C :=
by 
  -- Proof to be filled in later
  sorry

end perpendicular_bisectors_intersect_at_one_point_l193_193574


namespace length_of_short_pieces_l193_193528

def total_length : ℕ := 27
def long_piece_length : ℕ := 4
def number_of_long_pieces : ℕ := total_length / long_piece_length
def remainder_length : ℕ := total_length % long_piece_length
def number_of_short_pieces : ℕ := 3

theorem length_of_short_pieces (h1 : remainder_length = 3) : (remainder_length / number_of_short_pieces) = 1 :=
by
  sorry

end length_of_short_pieces_l193_193528


namespace log3_infinite_nested_l193_193735

theorem log3_infinite_nested (x : ℝ) (h : x = Real.logb 3 (64 + x)) : x = 4 :=
by
  sorry

end log3_infinite_nested_l193_193735


namespace minimum_y_l193_193901

theorem minimum_y (x : ℝ) (h : x > 1) : (∃ y : ℝ, y = x + 1 / (x - 1) ∧ y = 3) :=
by
  sorry

end minimum_y_l193_193901


namespace base6_divisible_19_l193_193178

theorem base6_divisible_19 (y : ℤ) : (19 ∣ (615 + 6 * y)) ↔ y = 2 := sorry

end base6_divisible_19_l193_193178


namespace value_of_x_l193_193658

theorem value_of_x (x : ℤ) (h : 3 * x = (26 - x) + 26) : x = 13 :=
by
  sorry

end value_of_x_l193_193658


namespace jack_second_half_time_l193_193385

def time_jack_first_half := 19
def time_between_jill_and_jack := 7
def time_jill := 32

def time_jack (time_jill time_between_jill_and_jack : ℕ) : ℕ :=
  time_jill - time_between_jill_and_jack

def time_jack_second_half (time_jack time_jack_first_half : ℕ) : ℕ :=
  time_jack - time_jack_first_half

theorem jack_second_half_time :
  time_jack_second_half (time_jack time_jill time_between_jill_and_jack) time_jack_first_half = 6 :=
by
  sorry

end jack_second_half_time_l193_193385


namespace factorization_of_difference_of_squares_l193_193250

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l193_193250


namespace ratio_equivalence_l193_193943

theorem ratio_equivalence (x : ℝ) :
  ((20 / 10) * 100 = (25 / x) * 100) → x = 12.5 :=
by
  intro h
  sorry

end ratio_equivalence_l193_193943


namespace share_of_B_is_2400_l193_193157

noncomputable def share_of_B (total_profit : ℝ) (B_investment : ℝ) (A_months B_months C_months D_months : ℝ) : ℝ :=
  let A_investment := 3 * B_investment
  let C_investment := (3/2) * B_investment
  let D_investment := (1/2) * A_investment
  let A_inv_months := A_investment * A_months
  let B_inv_months := B_investment * B_months
  let C_inv_months := C_investment * C_months
  let D_inv_months := D_investment * D_months
  let total_inv_months := A_inv_months + B_inv_months + C_inv_months + D_inv_months
  (B_inv_months / total_inv_months) * total_profit

theorem share_of_B_is_2400 :
  share_of_B 27000 (1000 : ℝ) 12 6 9 8 = 2400 := 
sorry

end share_of_B_is_2400_l193_193157


namespace factorize_difference_of_squares_l193_193302

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l193_193302


namespace intersection_A_B_l193_193575

def A := {x : ℝ | (x - 1) * (x - 4) < 0}
def B := {x : ℝ | x <= 2}

theorem intersection_A_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x <= 2} :=
sorry

end intersection_A_B_l193_193575


namespace pencils_left_proof_l193_193877

noncomputable def total_pencils_left (a d : ℕ) : ℕ :=
  let total_initial_pencils : ℕ := 30
  let total_pencils_given_away : ℕ := 15 * a + 105 * d
  total_initial_pencils - total_pencils_given_away

theorem pencils_left_proof (a d : ℕ) :
  total_pencils_left a d = 30 - (15 * a + 105 * d) :=
by
  sorry

end pencils_left_proof_l193_193877


namespace circle_equation_l193_193028

-- Define the circle's equation as a predicate
def is_circle (x y a b r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Given conditions, defining the known center and passing point
def center_x : ℝ := 2
def center_y : ℝ := -3
def point_M_x : ℝ := -1
def point_M_y : ℝ := 1

-- Prove that the circle with the given conditions has the correct equation
theorem circle_equation :
  is_circle x y center_x center_y 5 ↔ 
  ∀ x y : ℝ, (x - center_x)^2 + (y + center_y)^2 = 25 := sorry

end circle_equation_l193_193028


namespace meet_time_l193_193114

theorem meet_time 
  (circumference : ℝ) 
  (deepak_speed_kmph : ℝ) 
  (wife_speed_kmph : ℝ) 
  (deepak_speed_mpm : ℝ := deepak_speed_kmph * 1000 / 60) 
  (wife_speed_mpm : ℝ := wife_speed_kmph * 1000 / 60) 
  (relative_speed : ℝ := deepak_speed_mpm + wife_speed_mpm)
  (time_to_meet : ℝ := circumference / relative_speed) :
  circumference = 660 → 
  deepak_speed_kmph = 4.5 → 
  wife_speed_kmph = 3.75 → 
  time_to_meet = 4.8 :=
by 
  intros h1 h2 h3 
  sorry

end meet_time_l193_193114


namespace fixed_point_C_D_intersection_l193_193577

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 2) = 1

noncomputable def point_on_line (P : ℝ × ℝ) : Prop :=
  P.1 = 4 ∧ P.2 ≠ 0

noncomputable def line_CD_fixed_point (t : ℝ) (C D : ℝ × ℝ) : Prop :=
  let x1 := (36 - 2 * t^2) / (18 + t^2)
  let y1 := (12 * t) / (18 + t^2)
  let x2 := (2 * t^2 - 4) / (2 + t^2)
  let y2 := -(4 * t) / (t^2 + 2)
  C = (x1, y1) ∧ D = (x2, y2) →
  let k_CD := (4 * t) / (6 - t^2)
  ∀ (x y : ℝ), y + (4 * t) / (t^2 + 2) = k_CD * (x - (2 * t^2 - 4) / (t^2 + 2)) →
  y = 0 → x = 1

theorem fixed_point_C_D_intersection :
  ∀ (t : ℝ) (C D : ℝ × ℝ), point_on_line (4, t) →
  ellipse_equation C.1 C.2 →
  ellipse_equation D.1 D.2 →
  line_CD_fixed_point t C D :=
by
  intros t C D point_on_line_P ellipse_C ellipse_D
  sorry

end fixed_point_C_D_intersection_l193_193577


namespace range_of_x_l193_193907

theorem range_of_x (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) (h_increasing : ∀ {a b : ℝ}, a ≤ b → b ≤ 0 → f a ≤ f b) :
  (∀ x : ℝ, f (2^(2*x^2 - x - 1)) ≥ f (-4)) → ∀ x, x ∈ Set.Icc (-1 : ℝ) (3/2 : ℝ) :=
by 
  sorry

end range_of_x_l193_193907


namespace blue_red_marble_ratio_l193_193521

-- Define the initial counts and conditions
def initial_red_marbles := 20
def initial_blue_marbles := 30
def red_marbles_taken := 3
def total_marbles_left := 35

-- Formulate the equivalent proof problem
theorem blue_red_marble_ratio :
  let red_marbles_left := initial_red_marbles - red_marbles_taken in
  let blue_marbles_left := total_marbles_left - red_marbles_left in
  let blue_marbles_taken := initial_blue_marbles - blue_marbles_left in
  (blue_marbles_taken : ℚ) / red_marbles_taken = 4 :=
by {
  sorry
}

end blue_red_marble_ratio_l193_193521


namespace total_goals_l193_193605

-- Define constants for goals scored in respective seasons
def goalsLastSeason : ℕ := 156
def goalsThisSeason : ℕ := 187

-- Define the theorem for the total number of goals
theorem total_goals : goalsLastSeason + goalsThisSeason = 343 :=
by
  -- Proof is omitted
  sorry

end total_goals_l193_193605


namespace part1_part2_l193_193910

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp (x - 1) + a
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * x + Real.log x

theorem part1 (x : ℝ) (hx : 0 < x) :
  f x 0 ≥ g x 0 + 1 := sorry

theorem part2 {x0 : ℝ} (hx0 : ∃ y0 : ℝ, f x0 0 = g x0 0 ∧ ∀ x ≠ x0, f x 0 ≠ g x 0) :
  x0 < 2 := sorry

end part1_part2_l193_193910


namespace triangle_rectangle_ratio_l193_193708

/--
An equilateral triangle and a rectangle both have perimeters of 60 inches.
The rectangle has a length to width ratio of 2:1.
We need to prove that the ratio of the length of the side of the triangle to
the length of the rectangle is 1.
-/
theorem triangle_rectangle_ratio
  (triangle_perimeter rectangle_perimeter : ℕ)
  (triangle_side rectangle_length rectangle_width : ℕ)
  (h1 : triangle_perimeter = 60)
  (h2 : rectangle_perimeter = 60)
  (h3 : rectangle_length = 2 * rectangle_width)
  (h4 : triangle_side = triangle_perimeter / 3)
  (h5 : rectangle_perimeter = 2 * rectangle_length + 2 * rectangle_width)
  (h6 : rectangle_width = 10)
  (h7 : rectangle_length = 20)
  : triangle_side / rectangle_length = 1 := 
sorry

end triangle_rectangle_ratio_l193_193708


namespace correct_microorganism_dilution_statement_l193_193661

def microorganism_dilution_conditions (A B C D : Prop) : Prop :=
  (A ↔ ∀ (dilutions : ℕ) (n : ℕ), 1000 ≤ dilutions ∧ dilutions ≤ 10000000) ∧
  (B ↔ ∀ (dilutions : ℕ) (actinomycetes : ℕ), dilutions = 1000 ∨ dilutions = 10000 ∨ dilutions = 100000) ∧
  (C ↔ ∀ (dilutions : ℕ) (fungi : ℕ), dilutions = 1000 ∨ dilutions = 10000 ∨ dilutions = 100000) ∧
  (D ↔ ∀ (dilutions : ℕ) (bacteria_first_time : ℕ), 10 ≤ dilutions ∧ dilutions ≤ 10000000)

theorem correct_microorganism_dilution_statement (A B C D : Prop)
  (h : microorganism_dilution_conditions A B C D) : D :=
sorry

end correct_microorganism_dilution_statement_l193_193661


namespace factorization_of_x_squared_minus_one_l193_193241

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l193_193241


namespace find_integer_m_l193_193973

theorem find_integer_m 
  (m : ℤ)
  (h1 : 30 ≤ m ∧ m ≤ 80)
  (h2 : ∃ k : ℤ, m = 6 * k)
  (h3 : m % 8 = 2)
  (h4 : m % 5 = 2) : 
  m = 42 := 
sorry

end find_integer_m_l193_193973


namespace allocation_of_fabric_l193_193682

theorem allocation_of_fabric (x : ℝ) (y : ℝ) 
  (fabric_for_top : 3 * x = 2 * x)
  (fabric_for_pants : 3 * y = 3 * (600 - x))
  (total_fabric : x + y = 600)
  (sets_match : (x / 3) * 2 = (y / 3) * 3) : 
  x = 360 ∧ y = 240 := 
by
  sorry

end allocation_of_fabric_l193_193682


namespace max_servings_l193_193996

theorem max_servings :
  let cucumbers := 117,
      tomatoes := 116,
      bryndza := 4200,  -- converted to grams
      peppers := 60,
      cucumbers_per_serving := 2,
      tomatoes_per_serving := 2,
      bryndza_per_serving := 75,
      peppers_per_serving := 1 in
  min (min (cucumbers / cucumbers_per_serving) (tomatoes / tomatoes_per_serving))
      (min (bryndza / bryndza_per_serving) (peppers / peppers_per_serving)) = 56 := by
  sorry

end max_servings_l193_193996


namespace share_of_B_is_2400_l193_193156

noncomputable def share_of_B (total_profit : ℝ) (B_investment : ℝ) (A_months B_months C_months D_months : ℝ) : ℝ :=
  let A_investment := 3 * B_investment
  let C_investment := (3/2) * B_investment
  let D_investment := (1/2) * A_investment
  let A_inv_months := A_investment * A_months
  let B_inv_months := B_investment * B_months
  let C_inv_months := C_investment * C_months
  let D_inv_months := D_investment * D_months
  let total_inv_months := A_inv_months + B_inv_months + C_inv_months + D_inv_months
  (B_inv_months / total_inv_months) * total_profit

theorem share_of_B_is_2400 :
  share_of_B 27000 (1000 : ℝ) 12 6 9 8 = 2400 := 
sorry

end share_of_B_is_2400_l193_193156


namespace smallest_of_product_and_sum_l193_193742

theorem smallest_of_product_and_sum (a b c : ℤ) 
  (h1 : a * b * c = 32) 
  (h2 : a + b + c = 3) : 
  a = -4 ∨ b = -4 ∨ c = -4 :=
sorry

end smallest_of_product_and_sum_l193_193742


namespace remaining_kibble_l193_193083

def starting_kibble : ℕ := 12
def mary_kibble_morning : ℕ := 1
def mary_kibble_evening : ℕ := 1
def frank_kibble_afternoon : ℕ := 1
def frank_kibble_late_evening : ℕ := 2 * frank_kibble_afternoon

theorem remaining_kibble : starting_kibble - (mary_kibble_morning + mary_kibble_evening + frank_kibble_afternoon + frank_kibble_late_evening) = 7 := by
  sorry

end remaining_kibble_l193_193083


namespace solve_quadratic_eq_l193_193965

theorem solve_quadratic_eq (x : ℝ) : (x^2 + 4 * x = 5) ↔ (x = 1 ∨ x = -5) :=
by
  sorry

end solve_quadratic_eq_l193_193965


namespace factorize_difference_of_squares_l193_193227

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l193_193227


namespace total_pens_l193_193444

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193444


namespace tiffany_total_lives_l193_193985

-- Define the conditions
def initial_lives : Float := 43.0
def hard_part_won : Float := 14.0
def next_level_won : Float := 27.0

-- State the theorem
theorem tiffany_total_lives : 
  initial_lives + hard_part_won + next_level_won = 84.0 :=
by 
  sorry

end tiffany_total_lives_l193_193985


namespace flour_needed_l193_193147

theorem flour_needed (sugar flour : ℕ) (h1 : sugar = 50) (h2 : sugar / 10 = flour) : flour = 5 :=
by
  sorry

end flour_needed_l193_193147


namespace isosceles_triangle_perimeter_l193_193774

theorem isosceles_triangle_perimeter {a b c : ℝ} (h1 : a = 4) (h2 : b = 8) 
  (isosceles : a = c ∨ b = c) (triangle_inequality : a + a > b) :
  a + b + c = 20 :=
by
  sorry

end isosceles_triangle_perimeter_l193_193774


namespace factorize_x_squared_minus_one_l193_193199

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l193_193199


namespace number_of_people_l193_193983

-- Definitions based on the conditions
def average_age (T : ℕ) (n : ℕ) := T / n = 30
def youngest_age := 3
def average_age_when_youngest_born (T : ℕ) (n : ℕ) := (T - youngest_age) / (n - 1) = 27

theorem number_of_people (T n : ℕ) (h1 : average_age T n) (h2 : average_age_when_youngest_born T n) : n = 7 :=
by
  sorry

end number_of_people_l193_193983


namespace factorize_x_squared_minus_one_l193_193202

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l193_193202


namespace factorization_difference_of_squares_l193_193218

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l193_193218


namespace factorize_x_squared_minus_one_l193_193314

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l193_193314


namespace negation_exists_ltx2_plus_x_plus_1_lt_0_l193_193475

theorem negation_exists_ltx2_plus_x_plus_1_lt_0 :
  ¬ (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≥ 0 :=
by
  sorry

end negation_exists_ltx2_plus_x_plus_1_lt_0_l193_193475


namespace probability_donation_to_A_l193_193158

-- Define population proportions
def prob_O : ℝ := 0.50
def prob_A : ℝ := 0.15
def prob_B : ℝ := 0.30
def prob_AB : ℝ := 0.05

-- Define blood type compatibility predicate
def can_donate_to_A (blood_type : ℝ) : Prop := 
  blood_type = prob_O ∨ blood_type = prob_A

-- Theorem statement
theorem probability_donation_to_A : 
  prob_O + prob_A = 0.65 :=
by
  -- proof skipped
  sorry

end probability_donation_to_A_l193_193158


namespace factorize_x_squared_minus_one_l193_193201

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l193_193201


namespace ratio_lcm_gcf_256_162_l193_193492

theorem ratio_lcm_gcf_256_162 : (Nat.lcm 256 162) / (Nat.gcd 256 162) = 10368 := 
by 
  sorry

end ratio_lcm_gcf_256_162_l193_193492


namespace factorize_x_squared_minus_1_l193_193274

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l193_193274


namespace layla_more_points_l193_193791

-- Definitions from the conditions
def layla_points : ℕ := 70
def total_points : ℕ := 112
def nahima_points : ℕ := total_points - layla_points

-- Theorem that states the proof problem
theorem layla_more_points : layla_points - nahima_points = 28 :=
by
  simp [layla_points, nahima_points]
  rw [nat.sub_sub]
  sorry

end layla_more_points_l193_193791


namespace circle_radius_l193_193124

theorem circle_radius (r₂ : ℝ) : 
  (∃ r₁ : ℝ, r₁ = 5 ∧ (∀ d : ℝ, d = 7 → (d = r₁ + r₂ ∨ d = abs (r₁ - r₂)))) → (r₂ = 2 ∨ r₂ = 12) :=
by
  sorry

end circle_radius_l193_193124


namespace pawns_left_l193_193785

-- Definitions of the initial conditions
def initial_pawns : ℕ := 8
def kennedy_lost_pawns : ℕ := 4
def riley_lost_pawns : ℕ := 1

-- Definition of the total pawns left function
def total_pawns_left (initial_pawns kennedy_lost_pawns riley_lost_pawns : ℕ) : ℕ :=
  (initial_pawns - kennedy_lost_pawns) + (initial_pawns - riley_lost_pawns)

-- The statement to prove
theorem pawns_left : total_pawns_left initial_pawns kennedy_lost_pawns riley_lost_pawns = 11 := by
  -- Proof omitted
  sorry

end pawns_left_l193_193785


namespace sandwiches_prepared_l193_193958

-- Define the conditions as given in the problem.
def ruth_ate_sandwiches : ℕ := 1
def brother_ate_sandwiches : ℕ := 2
def first_cousin_ate_sandwiches : ℕ := 2
def each_other_cousin_ate_sandwiches : ℕ := 1
def number_of_other_cousins : ℕ := 2
def sandwiches_left : ℕ := 3

-- Define the total number of sandwiches eaten.
def total_sandwiches_eaten : ℕ := ruth_ate_sandwiches 
                                  + brother_ate_sandwiches
                                  + first_cousin_ate_sandwiches 
                                  + (each_other_cousin_ate_sandwiches * number_of_other_cousins)

-- Define the number of sandwiches prepared by Ruth.
def sandwiches_prepared_by_ruth : ℕ := total_sandwiches_eaten + sandwiches_left

-- Formulate the theorem to prove.
theorem sandwiches_prepared : sandwiches_prepared_by_ruth = 10 :=
by
  -- Use the solution steps to prove the theorem (proof omitted here).
  sorry

end sandwiches_prepared_l193_193958


namespace factorize_x_squared_minus_one_l193_193322

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l193_193322


namespace initial_balls_count_l193_193678

variables (y w : ℕ)

theorem initial_balls_count (h1 : y = 2 * (w - 10)) (h2 : w - 10 = 5 * (y - 9)) :
  y = 10 ∧ w = 15 :=
sorry

end initial_balls_count_l193_193678


namespace houses_built_during_boom_l193_193120

theorem houses_built_during_boom :
  let original_houses := 20817
  let current_houses := 118558
  let houses_built := current_houses - original_houses
  houses_built = 97741 := by
  sorry

end houses_built_during_boom_l193_193120


namespace binary_101011_is_43_l193_193731

def binary_to_decimal_conversion (b : Nat) : Nat := 
  match b with
  | 101011 => 43
  | _ => 0

theorem binary_101011_is_43 : binary_to_decimal_conversion 101011 = 43 := by
  sorry

end binary_101011_is_43_l193_193731


namespace probability_of_7_successes_l193_193674

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_of_successes (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coefficient n k * p^k * (1 - p)^(n - k)

theorem probability_of_7_successes :
  probability_of_successes 7 7 (2/7) = 128 / 823543 :=
by
  sorry

end probability_of_7_successes_l193_193674


namespace probability_of_7_successes_in_7_trials_l193_193668

open Probability

/-- Define the given conditions for the problem -/
def n : ℕ := 7
def k : ℕ := 7
def p : ℚ := 2 / 7

/-- The binomial coefficient and the probability of success in n trials -/
theorem probability_of_7_successes_in_7_trials :
  P(X = k) = (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) :=
by
  have bep_0 : nat.choose 7 7 = 1, from sorry,
  have p_power_k : p ^ k = (2 / 7) ^ 7, from sorry,
  have q_power_rem : (1 - p) ^ (n - k) = 1, from sorry,
  have p_eq_frac : (2 / 7) ^ 7 * 1 = 128 / 823543, from sorry,
  show 1 * (2 / 7) ^ 7 * 1 = 128 / 823543, by sorry

end probability_of_7_successes_in_7_trials_l193_193668


namespace total_pens_l193_193440

theorem total_pens (r : ℕ) (r_gt_10 : r > 10) (r_div_357 : r ∣ 357) (r_div_441 : r ∣ 441) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_l193_193440


namespace minimum_value_inequality_l193_193889

theorem minimum_value_inequality (x y : ℝ) (hx : x > 2) (hy : y > 2) :
    (x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2) ≥ 18 := by
  sorry

end minimum_value_inequality_l193_193889


namespace simplify_expression_l193_193468

theorem simplify_expression (y : ℝ) : y - 3 * (2 + y) + 4 * (2 - y) - 5 * (2 + 3 * y) = -21 * y - 8 :=
sorry

end simplify_expression_l193_193468


namespace ratio_of_7th_terms_l193_193488

theorem ratio_of_7th_terms (a b : ℕ → ℕ) (S T : ℕ → ℕ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : ∀ n, T n = n * (b 1 + b n) / 2)
  (h3 : ∀ n, S n / T n = (5 * n + 10) / (2 * n - 1)) :
  a 7 / b 7 = 3 :=
by
  sorry

end ratio_of_7th_terms_l193_193488


namespace factorization_difference_of_squares_l193_193211

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l193_193211


namespace mutually_exclusive_white_ball_events_l193_193484

-- Definitions of persons and balls
inductive Person | A | B | C
inductive Ball | red | black | white

-- Definitions of events
def eventA (dist : Person → Ball) : Prop := dist Person.A = Ball.white
def eventB (dist : Person → Ball) : Prop := dist Person.B = Ball.white

theorem mutually_exclusive_white_ball_events (dist : Person → Ball) :
  (eventA dist → ¬eventB dist) :=
by
  sorry

end mutually_exclusive_white_ball_events_l193_193484


namespace factorization_of_difference_of_squares_l193_193248

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l193_193248


namespace eccentricity_of_ellipse_l193_193906

open Real

noncomputable def ellipse_eccentricity : ℝ :=
  let a : ℝ := 4
  let b : ℝ := 2 * sqrt 3
  let c : ℝ := sqrt (a^2 - b^2)
  c / a

theorem eccentricity_of_ellipse (a b : ℝ) (ha : a = 4) (hb : b = 2 * sqrt 3) (h_eq : ∀ A B : ℝ, |A - B| = b^2 / 2 → |A - 2 * sqrt 3| + |B - 2 * sqrt 3| ≤ 10) :
  ellipse_eccentricity = 1 / 2 :=
by
  sorry

end eccentricity_of_ellipse_l193_193906


namespace factorize_x_squared_minus_one_l193_193188

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l193_193188


namespace avg_GPA_is_93_l193_193633

def avg_GPA_school (GPA_6th GPA_8th : ℕ) (GPA_diff : ℕ) : ℕ :=
  (GPA_6th + (GPA_6th + GPA_diff) + GPA_8th) / 3

theorem avg_GPA_is_93 :
  avg_GPA_school 93 91 2 = 93 :=
by
  -- The proof can be handled here 
  sorry

end avg_GPA_is_93_l193_193633


namespace h_h_three_l193_193921

def h (x : ℤ) : ℤ := 3 * x^2 + 3 * x - 2

theorem h_h_three : h (h 3) = 3568 := by
  sorry

end h_h_three_l193_193921


namespace combined_work_time_l193_193692

theorem combined_work_time (man_rate : ℚ := 1/5) (wife_rate : ℚ := 1/7) (son_rate : ℚ := 1/15) :
  (man_rate + wife_rate + son_rate)⁻¹ = 105 / 43 :=
by
  sorry

end combined_work_time_l193_193692


namespace exactly_one_gt_one_of_abc_eq_one_l193_193798

theorem exactly_one_gt_one_of_abc_eq_one 
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_abc : a * b * c = 1) 
  (h_sum : a + b + c > 1 / a + 1 / b + 1 / c) : 
  (1 < a ∧ b < 1 ∧ c < 1) ∨ (a < 1 ∧ 1 < b ∧ c < 1) ∨ (a < 1 ∧ b < 1 ∧ 1 < c) :=
sorry

end exactly_one_gt_one_of_abc_eq_one_l193_193798


namespace Heracles_age_l193_193715

variable (A H : ℕ)

theorem Heracles_age :
  (A = H + 7) →
  (A + 3 = 2 * H) →
  H = 10 :=
by
  sorry

end Heracles_age_l193_193715


namespace quadratic_trinomial_with_integral_roots_l193_193016

theorem quadratic_trinomial_with_integral_roots (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  (∃ x : ℤ, a * x^2 + b * x + c = 0) ∧ 
  (∃ x : ℤ, (a + 1) * x^2 + (b + 1) * x + (c + 1) = 0) ∧ 
  (∃ x : ℤ, (a + 2) * x^2 + (b + 2) * x + (c + 2) = 0) :=
sorry

end quadratic_trinomial_with_integral_roots_l193_193016


namespace new_person_age_l193_193969

theorem new_person_age (T : ℕ) (A : ℕ) (n : ℕ) 
  (avg_age : ℕ) (new_avg_age : ℕ) 
  (h1 : avg_age = T / n) 
  (h2 : T = 14 * n)
  (h3 : n = 17) 
  (h4 : new_avg_age = 15) 
  (h5 : new_avg_age = (T + A) / (n + 1)) 
  : A = 32 := 
by 
  sorry

end new_person_age_l193_193969


namespace total_pens_l193_193442

theorem total_pens (r : ℕ) (r_gt_10 : r > 10) (r_div_357 : r ∣ 357) (r_div_441 : r ∣ 441) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_l193_193442


namespace factorize_x_squared_minus_1_l193_193272

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l193_193272


namespace unique_fraction_property_l193_193555

def fractions_property (x y : ℕ) : Prop :=
  Nat.coprime x y ∧ (x + 1) * 5 * y = (y + 1) * 6 * x

theorem unique_fraction_property :
  {x : ℕ // {y : ℕ // fractions_property x y}} = {⟨5, ⟨6, fractions_property 5 6⟩⟩} :=
by
  sorry

end unique_fraction_property_l193_193555


namespace part1_part2_l193_193580

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

-- Part 1: Prove the range of k such that f(x) < k * x for all x
theorem part1 (k : ℝ) : (∀ x : ℝ, x > 0 → f x < k * x) ↔ k > 1 / (2 * Real.exp 1) :=
by sorry

-- Part 2: Define the function g(x) = f(x) - k * x and prove the range of k for which g(x) has two zeros in the interval [1/e, e^2]
noncomputable def g (x k : ℝ) : ℝ := f x - k * x

theorem part2 (k : ℝ) : (∃ x1 x2 : ℝ, 1 / Real.exp 1 ≤ x1 ∧ x1 ≤ Real.exp 2 ∧
                                 1 / Real.exp 1 ≤ x2 ∧ x2 ≤ Real.exp 2 ∧
                                 g x1 k = 0 ∧ g x2 k = 0 ∧ x1 ≠ x2)
                               ↔ 2 / (Real.exp 4) ≤ k ∧ k < 1 / (2 * Real.exp 1) :=
by sorry

end part1_part2_l193_193580


namespace problem_1_problem_2_problem_3_l193_193961

-- Definitions based on problem conditions
def total_people := 12
def choices := 5
def special_people_count := 3

noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Proof problem 1: A, B, and C must be chosen, so select 2 more from the remaining 9 people
theorem problem_1 : choose 9 2 = 36 :=
by sorry

-- Proof problem 2: Only one among A, B, and C is chosen, so select 4 more from the remaining 9 people
theorem problem_2 : choose 3 1 * choose 9 4 = 378 :=
by sorry

-- Proof problem 3: At most two among A, B, and C are chosen
theorem problem_3 : choose 12 5 - choose 9 2 = 756 :=
by sorry

end problem_1_problem_2_problem_3_l193_193961


namespace factorize_difference_of_squares_l193_193232

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l193_193232


namespace range_of_a_for_quad_ineq_false_l193_193761

variable (a : ℝ)

def quad_ineq_holds : Prop := ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0

theorem range_of_a_for_quad_ineq_false :
  ¬ quad_ineq_holds a → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_for_quad_ineq_false_l193_193761


namespace heracles_age_is_10_l193_193714

variable (H : ℕ)

-- Conditions
def audrey_age_now : ℕ := H + 7
def audrey_age_in_3_years : ℕ := audrey_age_now + 3
def heracles_twice_age : ℕ := 2 * H

-- Proof Statement
theorem heracles_age_is_10 (h1 : audrey_age_in_3_years = heracles_twice_age) : H = 10 :=
by 
  sorry

end heracles_age_is_10_l193_193714


namespace proof_of_problem_l193_193503

noncomputable def proof_problem (x y : ℚ) : Prop :=
  (sqrt (x - y) = 2 / 5) ∧ (sqrt (x + y) = 2) ∧ 
  x = 52 / 25 ∧ y = 48 / 25 ∧ 
  let vertices := [(0, 0), (2, 2), (2 / 25, -2 / 25), (52 / 25, 48 / 25)] in
  let area := Rational.from_ints 8 25 in
  ∃ (a b c d : ℚ × ℚ), 
    a ∈ vertices ∧ b ∈ vertices ∧ c ∈ vertices ∧ d ∈ vertices ∧ 
    ((b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = area)

theorem proof_of_problem : proof_problem (52 / 25) (48 / 25) :=
by { sorry } 

end proof_of_problem_l193_193503


namespace eval_x_plus_one_eq_4_l193_193340

theorem eval_x_plus_one_eq_4 (x : ℕ) (h : x = 3) : x + 1 = 4 :=
by
  sorry

end eval_x_plus_one_eq_4_l193_193340


namespace project_profit_starts_from_4th_year_l193_193850

def initial_investment : ℝ := 144
def maintenance_cost (n : ℕ) : ℝ := 4 * n^2 + 40 * n
def annual_income : ℝ := 100

def net_profit (n : ℕ) : ℝ := 
  annual_income * n - maintenance_cost n - initial_investment

theorem project_profit_starts_from_4th_year :
  ∀ n : ℕ, 3 < n ∧ n < 12 → net_profit n > 0 :=
by
  intros n hn
  sorry

end project_profit_starts_from_4th_year_l193_193850


namespace obtuse_triangle_has_two_acute_angles_l193_193058

-- Definitions based on conditions
def is_triangle (angles : list ℝ) : Prop :=
  angles.length = 3 ∧ angles.sum = 180

def is_obtuse_triangle (angles : list ℝ) : Prop :=
  is_triangle angles ∧ angles.any (> 90)

def acute_angles_count (angles : list ℝ) : ℝ :=
  angles.count (< 90)

-- Theorem based on conditions and the correct answer
theorem obtuse_triangle_has_two_acute_angles (angles : list ℝ) :
  is_obtuse_triangle angles → acute_angles_count angles = 2 := by
  sorry

end obtuse_triangle_has_two_acute_angles_l193_193058


namespace amazing_rectangle_area_unique_l193_193750

def isAmazingRectangle (a b : ℕ) : Prop :=
  a = 2 * b ∧ a * b = 3 * (2 * (a + b))

theorem amazing_rectangle_area_unique :
  ∃ (a b : ℕ), isAmazingRectangle a b ∧ a * b = 162 :=
by
  sorry

end amazing_rectangle_area_unique_l193_193750


namespace factorize_x_squared_minus_1_l193_193280

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l193_193280


namespace total_pens_l193_193422

theorem total_pens (r : ℕ) (h1 : r > 10)
  (h2 : 357 % r = 0)
  (h3 : 441 % r = 0) :
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_l193_193422


namespace hexagon_perimeter_l193_193932

-- Define the side length 's' based on the given area condition
def side_length (s : ℝ) : Prop :=
  (3 * Real.sqrt 2 + Real.sqrt 3) / 4 * s^2 = 12

-- The theorem to prove
theorem hexagon_perimeter (s : ℝ) (h : side_length s) : 
  6 * s = 6 * Real.sqrt (48 / (3 * Real.sqrt 2 + Real.sqrt 3)) :=
by
  sorry

end hexagon_perimeter_l193_193932


namespace solve_for_z_l193_193624

variable (z : ℂ) (i : ℂ)

theorem solve_for_z
  (h1 : 3 - 2*i*z = 7 + 4*i*z)
  (h2 : i^2 = -1) :
  z = 2*i / 3 :=
by
  sorry

end solve_for_z_l193_193624


namespace coloring_ways_l193_193076

-- Define a factorial function
def factorial : Nat → Nat
| 0       => 1
| (n + 1) => (n + 1) * factorial n

-- Define a derangement function
def derangement : Nat → Nat
| 0       => 1
| 1       => 0
| (n + 1) => n * (derangement n + derangement (n - 1))

-- Prove the main theorem
theorem coloring_ways : 
  let six_factorial := factorial 6
  let derangement_6 := derangement 6
  let derangement_5 := derangement 5
  720 * (derangement_6 + derangement_5) = 222480 := by
    let six_factorial := 720
    let derangement_6 := derangement 6
    let derangement_5 := derangement 5
    show six_factorial * (derangement_6 + derangement_5) = 222480
    sorry

end coloring_ways_l193_193076


namespace bananas_to_mush_l193_193763

theorem bananas_to_mush (x : ℕ) (h1 : 3 * (20 / x) = 15) : x = 4 :=
by
  sorry

end bananas_to_mush_l193_193763


namespace tom_distance_before_karen_wins_l193_193844

theorem tom_distance_before_karen_wins 
    (karen_speed : ℕ)
    (tom_speed : ℕ) 
    (karen_late_start : ℚ) 
    (karen_additional_distance : ℕ) 
    (T : ℚ) 
    (condition1 : karen_speed = 60) 
    (condition2 : tom_speed = 45)
    (condition3 : karen_late_start = 4 / 60)
    (condition4 : karen_additional_distance = 4)
    (condition5 : 60 * T = 45 * T + 8) :
    (45 * (8 / 15) = 24) :=
by
    sorry 

end tom_distance_before_karen_wins_l193_193844


namespace solution_set_f_l193_193040

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x + x^(1/2) - 1

theorem solution_set_f (x : ℝ) (hx_pos : x > 0) : 
  f x > f (2 * x - 4) ↔ 2 < x ∧ x < 4 :=
sorry

end solution_set_f_l193_193040


namespace Mayor_decision_to_adopt_model_A_l193_193782

-- Define the conditions
def num_people := 17

def radicals_support_model_A := (0 : ℕ)

def socialists_support_model_B (y : ℕ) := y

def republicans_support_model_B (x y : ℕ) := x - y

def independents_support_model_B (x y : ℕ) := (y + (x - y)) / 2

-- The number of individuals supporting model A and model B
def support_model_B (x y : ℕ) := radicals_support_model_A + socialists_support_model_B y + republicans_support_model_B x y + independents_support_model_B x y

def support_model_A (x : ℕ) := 4 * x - support_model_B x x / 2

-- Statement to prove
theorem Mayor_decision_to_adopt_model_A (x : ℕ) (h : x = num_people) : 
  support_model_A x > support_model_B x x := 
by {
  -- Proof goes here
  sorry
}

end Mayor_decision_to_adopt_model_A_l193_193782


namespace division_result_l193_193610

-- Define n in terms of the given condition
def n : ℕ := 9^2023

theorem division_result : n / 3 = 3^4045 :=
by
  sorry

end division_result_l193_193610


namespace wins_per_girl_l193_193900

theorem wins_per_girl (a b c d : ℕ) (h1 : a + b = 8) (h2 : a + c = 10) (h3 : b + c = 12) (h4 : a + d = 12) (h5 : b + d = 14) (h6 : c + d = 16) : 
  a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 9 :=
sorry

end wins_per_girl_l193_193900


namespace age_ratio_l193_193776

theorem age_ratio (B_current A_current B_10_years_ago A_in_10_years : ℕ) 
  (h1 : B_current = 37) 
  (h2 : A_current = B_current + 7) 
  (h3 : B_10_years_ago = B_current - 10) 
  (h4 : A_in_10_years = A_current + 10) : 
  A_in_10_years / B_10_years_ago = 2 :=
by
  sorry

end age_ratio_l193_193776


namespace land_area_of_each_section_l193_193981

theorem land_area_of_each_section (n : ℕ) (total_area : ℕ) (h1 : n = 3) (h2 : total_area = 7305) :
  total_area / n = 2435 :=
by {
  sorry
}

end land_area_of_each_section_l193_193981


namespace trailingZeros_310_fact_l193_193019

-- Define the function to compute trailing zeros in factorials
def trailingZeros (n : ℕ) : ℕ := 
  if n = 0 then 0 else n / 5 + trailingZeros (n / 5)

-- Define the specific case for 310!
theorem trailingZeros_310_fact : trailingZeros 310 = 76 := 
by 
  sorry

end trailingZeros_310_fact_l193_193019


namespace solve_mod_equation_l193_193874

def is_two_digit_positive_integer (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem solve_mod_equation (u : ℕ) (h1 : is_two_digit_positive_integer u) (h2 : 13 * u % 100 = 52) : u = 4 :=
sorry

end solve_mod_equation_l193_193874


namespace factor_difference_of_squares_l193_193266

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l193_193266


namespace simplify_expression_l193_193466

theorem simplify_expression (y : ℝ) : y - 3 * (2 + y) + 4 * (2 - y) - 5 * (2 + 3 * y) = -21 * y - 8 :=
by
  sorry

end simplify_expression_l193_193466


namespace value_of_m_l193_193773

theorem value_of_m (m : ℝ) (h : ∀ x : ℝ, 0 < x → x < 2 → - (1 / 2) * x^2 + 2 * x ≤ m * x) :
  m = 1 :=
sorry

end value_of_m_l193_193773


namespace quadratic_to_vertex_form_l193_193557

theorem quadratic_to_vertex_form :
  ∀ (x : ℝ), (x^2 - 2*x + 3 = (x-1)^2 + 2) :=
by intro x; sorry

end quadratic_to_vertex_form_l193_193557


namespace triangle_formation_l193_193348

theorem triangle_formation (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : x₁ ≠ x₂) (h₂ : x₁ ≠ x₃) (h₃ : x₁ ≠ x₄) (h₄ : x₂ ≠ x₃) (h₅ : x₂ ≠ x₄) (h₆ : x₃ ≠ x₄)
  (h₇ : 0 < x₁) (h₈ : 0 < x₂) (h₉ : 0 < x₃) (h₁₀ : 0 < x₄)
  (h₁₁ : (x₁ + x₂ + x₃ + x₄) * (1/x₁ + 1/x₂ + 1/x₃ + 1/x₄) < 17) :
  (x₁ + x₂ > x₃) ∧ (x₂ + x₃ > x₄) ∧ (x₁ + x₃ > x₂) ∧ 
  (x₁ + x₄ > x₃) ∧ (x₁ + x₂ > x₄) ∧ (x₃ + x₄ > x₁) ∧ 
  (x₂ + x₄ > x₁) ∧ (x₂ + x₃ > x₁) :=
sorry

end triangle_formation_l193_193348


namespace factorization_of_x_squared_minus_one_l193_193240

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l193_193240


namespace factorize_difference_of_squares_l193_193294

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l193_193294


namespace probability_three_one_l193_193677

-- Definitions based on the conditions
def total_balls : ℕ := 18
def black_balls : ℕ := 10
def white_balls : ℕ := 8
def drawn_balls : ℕ := 4

-- Defining the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Definition of the total number of ways to draw 4 balls from 18
def total_ways_to_draw : ℕ := binom total_balls drawn_balls

-- Definition of the number of favorable ways to draw 3 black and 1 white ball
def favorable_black_white : ℕ := binom black_balls 3 * binom white_balls 1

-- Definition of the number of favorable ways to draw 1 black and 3 white balls
def favorable_white_black : ℕ := binom black_balls 1 * binom white_balls 3

-- Total favorable outcomes
def total_favorable_ways : ℕ := favorable_black_white + favorable_white_black

-- The probability of drawing 3 one color and 1 other color
def probability : ℚ := total_favorable_ways / total_ways_to_draw

-- Prove that the probability is 19/38
theorem probability_three_one :
  probability = 19 / 38 :=
sorry

end probability_three_one_l193_193677


namespace relationship_of_ys_l193_193351

variables {k y1 y2 y3 : ℝ}

theorem relationship_of_ys (h : k < 0) 
  (h1 : y1 = k / -4) 
  (h2 : y2 = k / 2) 
  (h3 : y3 = k / 3) : 
  y1 > y3 ∧ y3 > y2 :=
by 
  sorry

end relationship_of_ys_l193_193351


namespace solve_for_n_l193_193625

theorem solve_for_n (n x y : ℤ) (h : n * (x + y) + 17 = n * (-x + y) - 21) (hx : x = 1) : n = -19 :=
by
  sorry

end solve_for_n_l193_193625


namespace Ruth_sandwiches_l193_193957

theorem Ruth_sandwiches (sandwiches_left sandwiches_ruth sandwiches_brother sandwiches_first_cousin sandwiches_two_cousins total_sandwiches : ℕ)
  (h_ruth : sandwiches_ruth = 1)
  (h_brother : sandwiches_brother = 2)
  (h_first_cousin : sandwiches_first_cousin = 2)
  (h_two_cousins : sandwiches_two_cousins = 2)
  (h_left : sandwiches_left = 3) :
  total_sandwiches = sandwiches_left + sandwiches_two_cousins + sandwiches_first_cousin + sandwiches_ruth + sandwiches_brother :=
by
  sorry

end Ruth_sandwiches_l193_193957


namespace g6_eq_16_l193_193474

-- Definition of the function g that satisfies the given conditions
variable (g : ℝ → ℝ)

-- Given conditions
axiom functional_eq : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g3_eq_4 : g 3 = 4

-- The goal is to prove g(6) = 16
theorem g6_eq_16 : g 6 = 16 := by
  sorry

end g6_eq_16_l193_193474


namespace school_avg_GPA_l193_193628

theorem school_avg_GPA (gpa_6th : ℕ) (gpa_7th : ℕ) (gpa_8th : ℕ) 
  (h6 : gpa_6th = 93) 
  (h7 : gpa_7th = 95) 
  (h8 : gpa_8th = 91) : 
  (gpa_6th + gpa_7th + gpa_8th) / 3 = 93 :=
by 
  sorry

end school_avg_GPA_l193_193628


namespace midpoint_quadrilateral_area_l193_193643

theorem midpoint_quadrilateral_area (R : ℝ) (hR : 0 < R) :
  ∃ (Q : ℝ), Q = R / 4 :=
by
  sorry

end midpoint_quadrilateral_area_l193_193643


namespace gcd_102_238_l193_193110

-- Define the two numbers involved
def a : ℕ := 102
def b : ℕ := 238

-- State the theorem
theorem gcd_102_238 : Int.gcd a b = 34 :=
by
  sorry

end gcd_102_238_l193_193110


namespace max_servings_possible_l193_193993

def number_of_servings
  (peppers cucumbers tomatoes : Nat) (brynza : Nat) : Nat :=
  min (peppers) (min (brynza / 75) (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings_possible :
  number_of_servings 60 117 116 4200 = 56 := 
by 
  -- sorry statement allows skipping the proof
  sorry

end max_servings_possible_l193_193993


namespace small_canteens_needed_l193_193659

theorem small_canteens_needed 
  (f t a v : ℕ)
  (hf : f = 9)
  (ht : t = 8)
  (ha : a = 7)
  (hv : v = 6) :
  let total_water := f * t + a,
  small_canteens := (total_water + v - 1) / v
  in small_canteens = 14 :=
by
  sorry

end small_canteens_needed_l193_193659


namespace n_squared_divisible_by_36_l193_193926

theorem n_squared_divisible_by_36 (n : ℕ) (h1 : 0 < n) (h2 : 6 ∣ n) : 36 ∣ n^2 := 
sorry

end n_squared_divisible_by_36_l193_193926


namespace factorize_x_squared_minus_1_l193_193275

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l193_193275


namespace average_of_solutions_l193_193730

theorem average_of_solutions (a b : ℝ) (h : ∃ x1 x2 : ℝ, a * x1 ^ 2 + 3 * a * x1 + b = 0 ∧ a * x2 ^ 2 + 3 * a * x2 + b = 0) :
  ((-3 : ℝ) / 2) = - 3 / 2 :=
by sorry

end average_of_solutions_l193_193730


namespace factorize_x_squared_minus_one_l193_193198

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l193_193198


namespace max_value_of_xy_l193_193576

theorem max_value_of_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2*y = 2) :
  xy ≤ 1 / 2 :=
sorry

end max_value_of_xy_l193_193576


namespace distinct_roots_l193_193609

noncomputable def roots (a b c : ℝ) := ((b^2 - 4 * a * c) ≥ 0) ∧ ((-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a) * Real.sqrt (b^2 - 4 * a * c)) ≠ (0 : ℝ)

theorem distinct_roots{ p q r s : ℝ } (h1 : p ≠ q) (h2 : p ≠ r) (h3 : p ≠ s) (h4 : q ≠ r) 
(h5 : q ≠ s) (h6 : r ≠ s)
(h_roots_1 : roots 1 (-12*p) (-13*q))
(h_roots_2 : roots 1 (-12*r) (-13*s)) : 
(p + q + r + s = 2028) := sorry

end distinct_roots_l193_193609


namespace max_servings_l193_193990

open Nat

def servings (cucumbers tomatoes brynza_peppers brynza_grams: Nat) : Nat :=
  min (floor (cucumbers / 2))
    (min (floor (tomatoes / 2))
      (min (floor (brynza_peppers / 75)) brynza_grams))

theorem max_servings (cucumbers tomatoes peppers: Nat) (brynza_grams: Rat) 
  (cuc_reqs toma_reqs brynza_per pepper_reqs: Nat) (br_in_grams: Nat) : 
  servings cucumbers tomatoes brynza_grams peppers = 56 :=
by
  have cuc_portions : cucumbers / cuc_reqs = 58 := by sorry
  have toma_portions : tomatoes / toma_reqs = 58 := by sorry
  have brynza_portions : (br_in_grams / brynza_per) = 56 := by sorry
  have pepper_portions : peppers / pepper_reqs = 60 := by sorry
  exact min (min (min cuc_portions toma_portions) brynza_portions) pepper_portions
  

end max_servings_l193_193990


namespace max_salad_servings_l193_193987

theorem max_salad_servings :
  let cucumbers_per_serving := 2
  let tomatoes_per_serving := 2
  let bryndza_per_serving := 75 -- in grams
  let pepper_per_serving := 1
  let total_peppers := 60
  let total_bryndza := 4200 -- in grams
  let total_tomatoes := 116
  let total_cucumbers := 117
  let servings_peppers := total_peppers / pepper_per_serving
  let servings_bryndza := total_bryndza / bryndza_per_serving
  let servings_tomatoes := total_tomatoes / tomatoes_per_serving
  let servings_cucumbers := total_cucumbers / cucumbers_per_serving
  let max_servings := Int.min servings_peppers servings_bryndza
    (Int.min servings_tomatoes servings_cucumbers)
  max_servings = 56 :=
by
  sorry

end max_salad_servings_l193_193987


namespace factorize_difference_of_squares_l193_193308

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l193_193308


namespace factorize_x_squared_minus_one_l193_193313

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l193_193313


namespace inequalities_hold_l193_193364

theorem inequalities_hold (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  a^2 + b^2 ≥ 2 ∧ (1 / a + 1 / b) ≥ 2 := by
  sorry

end inequalities_hold_l193_193364


namespace infinite_geometric_series_sum_l193_193867

theorem infinite_geometric_series_sum :
  ∑' (n : ℕ), (1 : ℚ) * (-1 / 4 : ℚ) ^ n = 4 / 5 :=
by
  sorry

end infinite_geometric_series_sum_l193_193867


namespace sum_of_common_ratios_eq_three_l193_193794

theorem sum_of_common_ratios_eq_three
  (k a2 a3 b2 b3 : ℕ)
  (p r : ℕ)
  (h_nonconst1 : k ≠ 0)
  (h_nonconst2 : p ≠ r)
  (h_seq1 : a3 = k * p ^ 2)
  (h_seq2 : b3 = k * r ^ 2)
  (h_seq3 : a2 = k * p)
  (h_seq4 : b2 = k * r)
  (h_eq : a3 - b3 = 3 * (a2 - b2)) :
  p + r = 3 := 
sorry

end sum_of_common_ratios_eq_three_l193_193794


namespace num_starting_lineups_l193_193536

def total_players := 15
def chosen_players := 3 -- Ace, Zeppo, Buddy already chosen
def remaining_players := total_players - chosen_players
def players_to_choose := 2 -- remaining players to choose

noncomputable def combinations (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem num_starting_lineups : combinations remaining_players players_to_choose = 66 := by
  sorry

end num_starting_lineups_l193_193536


namespace cookies_left_l193_193800

theorem cookies_left (initial_cookies : ℕ) (cookies_eaten : ℕ) (cookies_left : ℕ) :
  initial_cookies = 28 → cookies_eaten = 21 → cookies_left = initial_cookies - cookies_eaten → cookies_left = 7 :=
by
  intros h_initial h_eaten h_left
  rw [h_initial, h_eaten] at h_left
  exact h_left

end cookies_left_l193_193800


namespace count_subgroups_multiple_of_11_l193_193863

noncomputable def is_multiple_of (n k : ℕ) : Prop := k > 0 ∧ n % k = 0

theorem count_subgroups_multiple_of_11 :
  let numbers : List ℕ := [1, 4, 8, 10, 16, 19, 21, 25, 30, 43]
  List.length (numbers)
  == 10 →
  (∃ n, count (λ sublist, is_multiple_of sublist.sum 11) sublists (filter_sublists numbers) = 7) :=
sorry

end count_subgroups_multiple_of_11_l193_193863


namespace count_divisible_by_3_in_range_l193_193918

theorem count_divisible_by_3_in_range (a b : ℤ) :
  a = 252 → b = 549 → (∃ n : ℕ, (a ≤ 3 * n ∧ 3 * n ≤ b) ∧ (b - a) / 3 = (100 : ℝ)) :=
by
  intros ha hb
  have h1 : ∃ k : ℕ, a = 3 * k := by sorry
  have h2 : ∃ m : ℕ, b = 3 * m := by sorry
  sorry

end count_divisible_by_3_in_range_l193_193918


namespace MrSlinkums_total_count_l193_193859

variable (T : ℕ)

-- Defining the conditions as given in the problem
def placed_on_shelves (T : ℕ) : ℕ := (20 * T) / 100
def storage (T : ℕ) : ℕ := (80 * T) / 100

-- Stating the main theorem to prove
theorem MrSlinkums_total_count 
    (h : storage T = 120) : 
    T = 150 :=
sorry

end MrSlinkums_total_count_l193_193859


namespace NinaCalculationCorrectAnswer_l193_193090

variable (y : ℝ)

noncomputable def NinaMistakenCalculation (y : ℝ) : ℝ :=
(y + 25) * 5

noncomputable def NinaCorrectCalculation (y : ℝ) : ℝ :=
(y - 25) / 5

theorem NinaCalculationCorrectAnswer (hy : (NinaMistakenCalculation y) = 200) :
  (NinaCorrectCalculation y) = -2 := by
  sorry

end NinaCalculationCorrectAnswer_l193_193090


namespace intersection_of_sets_l193_193055

def setP : Set ℝ := { x | x ≤ 3 }
def setQ : Set ℝ := { x | x > 1 }

theorem intersection_of_sets : setP ∩ setQ = { x | 1 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_of_sets_l193_193055


namespace min_bulbs_l193_193819

theorem min_bulbs (p : ℝ) (k : ℕ) (target_prob : ℝ) (h_p : p = 0.95) (h_k : k = 5) (h_target_prob : target_prob = 0.99) :
  ∃ n : ℕ, (finset.Ico k.succ n.succ).sum (λ i, (finset.range i).sum (λ j, (nat.choose j i) * p^i * (1 - p)^(j-i))) ≥ target_prob ∧ ∀ m : ℕ, m < n → 
  (finset.Ico k.succ m.succ).sum (λ i, (finset.range i).sum (λ j, (nat.choose j i) * p^i * (1 - p)^(j-i))) < target_prob :=
begin
  sorry
end

end min_bulbs_l193_193819


namespace sin_of_alpha_l193_193571

theorem sin_of_alpha 
  (α : ℝ) 
  (h : Real.cos (α - Real.pi / 2) = 1 / 3) : 
  Real.sin α = 1 / 3 := 
by 
  sorry

end sin_of_alpha_l193_193571


namespace cone_sphere_volume_ratio_l193_193854

theorem cone_sphere_volume_ratio (r h : ℝ) 
  (radius_eq : r > 0)
  (volume_rel : (1 / 3 : ℝ) * π * r^2 * h = (1 / 3 : ℝ) * (4 / 3) * π * r^3) : 
  h / r = 4 / 3 :=
by
  sorry

end cone_sphere_volume_ratio_l193_193854


namespace treasure_chest_coins_l193_193602

theorem treasure_chest_coins (hours : ℕ) (coins_per_hour : ℕ) (total_coins : ℕ) :
  hours = 8 → coins_per_hour = 25 → total_coins = hours * coins_per_hour → total_coins = 200 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end treasure_chest_coins_l193_193602


namespace max_pieces_four_cuts_l193_193702

theorem max_pieces_four_cuts (n : ℕ) (h : n = 4) : (by sorry : ℕ) = 14 := 
by sorry

end max_pieces_four_cuts_l193_193702


namespace solve_cubic_root_eq_l193_193881

theorem solve_cubic_root_eq (x : ℝ) (h : (5 - x)^(1/3) = 4) : x = -59 := 
by
  sorry

end solve_cubic_root_eq_l193_193881


namespace find_uncertain_mushrooms_l193_193394

-- Definitions for the conditions based on the problem statement.
variable (totalMushrooms : ℕ)
variable (safeMushrooms : ℕ)
variable (poisonousMushrooms : ℕ)
variable (uncertainMushrooms : ℕ)

-- The conditions given in the problem
-- 1. Lillian found 32 mushrooms.
-- 2. She identified 9 mushrooms as safe to eat.
-- 3. The number of poisonous mushrooms is twice the number of safe mushrooms.
-- 4. The total number of mushrooms is the sum of safe, poisonous, and uncertain mushrooms.

axiom given_conditions : 
  totalMushrooms = 32 ∧
  safeMushrooms = 9 ∧
  poisonousMushrooms = 2 * safeMushrooms ∧
  totalMushrooms = safeMushrooms + poisonousMushrooms + uncertainMushrooms

-- The proof problem: Given the conditions, prove the number of uncertain mushrooms equals 5
theorem find_uncertain_mushrooms : 
  uncertainMushrooms = 5 :=
by sorry

end find_uncertain_mushrooms_l193_193394


namespace dart_probability_l193_193140

noncomputable def area_hexagon (s : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * s^2

noncomputable def area_circle (s : ℝ) : ℝ := Real.pi * s^2

noncomputable def probability (s : ℝ) : ℝ := 
  (area_circle s) / (area_hexagon s)

theorem dart_probability (s : ℝ) (hs : s > 0) :
  probability s = (2 * Real.pi) / (3 * Real.sqrt 3) :=
by
  sorry

end dart_probability_l193_193140


namespace factorize_difference_of_squares_l193_193231

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l193_193231


namespace factorization_of_difference_of_squares_l193_193251

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l193_193251


namespace factorize_difference_of_squares_l193_193286

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l193_193286


namespace remainder_five_n_minus_eleven_l193_193495

theorem remainder_five_n_minus_eleven (n : ℤ) (h : n % 7 = 3) : (5 * n - 11) % 7 = 4 := 
    sorry

end remainder_five_n_minus_eleven_l193_193495


namespace factorization_correct_l193_193840

theorem factorization_correct : ∀ (x : ℕ), x^2 - x = x * (x - 1) :=
by
  intro x
  -- We know the problem reduces to algebraic identity proof
  sorry

end factorization_correct_l193_193840


namespace inequality_wxyz_l193_193464

theorem inequality_wxyz 
  (w x y z : ℝ) 
  (h₁ : w^2 + y^2 ≤ 1) : 
  (w * x + y * z - 1)^2 ≥ (w^2 + y^2 - 1) * (x^2 + z^2 - 1) :=
by
  sorry

end inequality_wxyz_l193_193464


namespace find_a5_l193_193902

-- Define the sequence and its properties
def geom_sequence (a : ℕ → ℕ) : Prop :=
∀ n m : ℕ, a (n + m) = (2^m) * a n

-- Define the problem statement
def sum_of_first_five_terms_is_31 (a : ℕ → ℕ) : Prop :=
a 1 + a 2 + a 3 + a 4 + a 5 = 31

-- State the theorem to prove
theorem find_a5 (a : ℕ → ℕ) (h_geom : geom_sequence a) (h_sum : sum_of_first_five_terms_is_31 a) : a 5 = 16 :=
by
  sorry

end find_a5_l193_193902


namespace max_gold_coins_l193_193841

theorem max_gold_coins (n : ℕ) (k : ℕ) (H1 : n = 13 * k + 3) (H2 : n < 150) : n ≤ 146 := 
by
  sorry

end max_gold_coins_l193_193841


namespace peaches_in_each_basket_l193_193652

variable (R : ℕ)

theorem peaches_in_each_basket (h : 6 * R = 96) : R = 16 :=
by
  sorry

end peaches_in_each_basket_l193_193652


namespace smaller_package_contains_correct_number_of_cupcakes_l193_193732

-- Define the conditions
def number_of_packs_large : ℕ := 4
def cupcakes_per_large_pack : ℕ := 15
def total_children : ℕ := 100
def needed_packs_small : ℕ := 4

-- Define the total cupcakes bought initially
def total_cupcakes_bought : ℕ := number_of_packs_large * cupcakes_per_large_pack

-- Define the total additional cupcakes needed
def additional_cupcakes_needed : ℕ := total_children - total_cupcakes_bought

-- Define the number of cupcakes per smaller package
def cupcakes_per_small_pack : ℕ := additional_cupcakes_needed / needed_packs_small

-- The theorem statement to prove
theorem smaller_package_contains_correct_number_of_cupcakes :
  cupcakes_per_small_pack = 10 :=
by
  -- This is where the proof would go
  sorry

end smaller_package_contains_correct_number_of_cupcakes_l193_193732


namespace identify_false_statement_l193_193499

-- Definitions for the conditions
def isMultipleOf (n k : Nat) : Prop := ∃ m, n = k * m

def conditions : Prop :=
  isMultipleOf 12 2 ∧
  isMultipleOf 123 3 ∧
  isMultipleOf 1234 4 ∧
  isMultipleOf 12345 5 ∧
  isMultipleOf 123456 6

-- The statement which proves which condition is false
theorem identify_false_statement : conditions → ¬ (isMultipleOf 1234 4) :=
by
  intros h
  sorry

end identify_false_statement_l193_193499


namespace number_of_smaller_pipes_l193_193853

theorem number_of_smaller_pipes (D_L D_s : ℝ) (h1 : D_L = 8) (h2 : D_s = 2) (v: ℝ) :
  let A_L := (π * (D_L / 2)^2)
  let A_s := (π * (D_s / 2)^2)
  (A_L / A_s) = 16 :=
by {
  sorry
}

end number_of_smaller_pipes_l193_193853


namespace complete_the_square_l193_193471

theorem complete_the_square (d e f : ℤ) (h1 : d > 0)
  (h2 : 25 * d * d = 25)
  (h3 : 10 * d * e = 30)
  (h4 : 25 * d * d * (d * x + e) * (d * x + e) = 25 * x * x * 25 + 30 * x * 25 * d + 25 * e * e - 9)
  : d + e + f = 41 := 
  sorry

end complete_the_square_l193_193471


namespace unique_fraction_condition_l193_193546

theorem unique_fraction_condition :
  ∃! (x y : ℕ), x.gcd y = 1 ∧ y = x * 6 / 5 ∧ (1.2 * (x : ℚ) / y = (x + 1 : ℚ) / (y + 1)) := by
  sorry

end unique_fraction_condition_l193_193546


namespace layla_more_than_nahima_l193_193787

-- Definitions for the conditions
def layla_points : ℕ := 70
def total_points : ℕ := 112
def nahima_points : ℕ := total_points - layla_points

-- Statement of the theorem
theorem layla_more_than_nahima : layla_points - nahima_points = 28 :=
by
  sorry

end layla_more_than_nahima_l193_193787


namespace machineA_finishing_time_l193_193615

theorem machineA_finishing_time
  (A : ℝ)
  (hA : 0 < A)
  (hB : 0 < 12)
  (hC : 0 < 6)
  (h_total_time : 0 < 2)
  (h_work_done_per_hour : (1 / A) + (1 / 12) + (1 / 6) = 1 / 2) :
  A = 4 := sorry

end machineA_finishing_time_l193_193615


namespace survey_total_people_l193_193748

theorem survey_total_people (number_represented : ℕ) (percentage : ℝ) (h : number_represented = percentage * 200) : 
  (number_represented : ℝ) = 200 := 
by 
 sorry

end survey_total_people_l193_193748


namespace union_inter_eq_union_compl_inter_eq_l193_193915

open Set

variable (U : Set ℕ) (A B C : Set ℕ)
variable [DecidableEq ℕ]

def U_def : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A_def : Set ℕ := {x | x^2 - 3 * x + 2 = 0}
def B_def : Set ℕ := {x | 1 ≤ x ∧ x ≤ 5 ∧ x ∈ ℤ}
def C_def : Set ℕ := {x | 2 < x ∧ x < 9 ∧ x ∈ ℤ}

theorem union_inter_eq :
  A ∪ (B ∩ C) = {1, 2, 3, 4, 5} :=
by
  -- proof would go here
  sorry

theorem union_compl_inter_eq :
  (compl U ∩ B) ∪ (compl U ∩ C) = {1, 2, 6, 7, 8} :=
by
  -- proof would go here
  sorry

end union_inter_eq_union_compl_inter_eq_l193_193915


namespace little_sister_stole_roses_l193_193955

/-- Ricky has 40 roses. His little sister steals some roses. He wants to give away the rest of the roses in equal portions to 9 different people, and each person gets 4 roses. Prove how many roses his little sister stole. -/
theorem little_sister_stole_roses (total_roses stolen_roses remaining_roses people roses_per_person : ℕ)
  (h1 : total_roses = 40)
  (h2 : people = 9)
  (h3 : roses_per_person = 4)
  (h4 : remaining_roses = people * roses_per_person)
  (h5 : remaining_roses = total_roses - stolen_roses) :
  stolen_roses = 4 :=
by
  sorry

end little_sister_stole_roses_l193_193955


namespace cos_two_thirds_pi_l193_193875

theorem cos_two_thirds_pi : Real.cos (2 / 3 * Real.pi) = -1 / 2 :=
by sorry

end cos_two_thirds_pi_l193_193875


namespace area_of_shaded_region_l193_193884

open Real

theorem area_of_shaded_region : 
  let Line1 : ℝ → ℝ := fun x => (3 / 4 * x + 5 / 4)
  let Line2 : ℝ → ℝ := fun x => (3 / 2 * x - 2)
  ∫ x in 1..(13/3), |Line1 x - Line2 x| = 1.7 := 
by
  sorry

end area_of_shaded_region_l193_193884


namespace equilateral_triangle_in_ellipse_l193_193161

def ellipse_equation (x y a b : ℝ) : Prop := 
  ((x - y)^2 / a^2) + ((x + y)^2 / b^2) = 1

theorem equilateral_triangle_in_ellipse 
  {a b x y : ℝ}
  (A B C : ℝ × ℝ)
  (hA : A.1 = 0 ∧ A.2 = b)
  (hBC_parallel : ∃ k : ℝ, B.2 = k * B.1 ∧ C.2 = k * C.1 ∧ k = 1)
  (hF : ∃ F : ℝ × ℝ, F = C)
  (hEllipseA : ellipse_equation A.1 A.2 a b) 
  (hEllipseB : ellipse_equation B.1 B.2 a b)
  (hEllipseC : ellipse_equation C.1 C.2 a b) 
  (equilateral : dist A B = dist B C ∧ dist B C = dist C A) :
  AB / b = 8 / 5 :=
sorry

end equilateral_triangle_in_ellipse_l193_193161


namespace triangle_BC_length_l193_193068

noncomputable def length_BC (α β γ : ℝ) (A B C : ℝ) : ℝ :=
  let AB := 1
  let AC := Real.sqrt 2
  let B := Real.pi / 4 -- 45 degrees in radians
  if AB = 1 ∧ AC = Real.sqrt 2 ∧ B = Real.pi / 4 then
    let C := Real.asin (AB * Real.sin B / AC)
    let A := Real.pi - B - C
    let BC := AB * Real.sin A / Real.sin C
    BC
  else 0

theorem triangle_BC_length :
  length_BC (1 : ℝ) (Real.sqrt 2) (Real.pi / 4) = (Real.sqrt 2 + Real.sqrt 6) / 2 :=
by
  sorry

end triangle_BC_length_l193_193068


namespace product_of_consecutive_integers_even_l193_193623

theorem product_of_consecutive_integers_even (n : ℤ) : Even (n * (n + 1)) :=
sorry

end product_of_consecutive_integers_even_l193_193623


namespace h_h3_eq_3568_l193_193924

def h (x : ℤ) := 3 * x ^ 2 + 3 * x - 2

theorem h_h3_eq_3568 : h (h 3) = 3568 := by
  sorry

end h_h3_eq_3568_l193_193924


namespace min_bulbs_needed_l193_193817

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k) * (p^k) * ((1-p)^(n-k))

theorem min_bulbs_needed (p : ℝ) (k : ℕ) (target_prob : ℝ) :
  p = 0.95 → k = 5 → target_prob = 0.99 →
  ∃ n, (∑ i in finset.range (n+1), if i ≥ 5 then binomial_probability n i p else 0) ≥ target_prob ∧ 
  (¬ ∃ m, m < n ∧ (∑ i in finset.range (m+1), if i ≥ 5 then binomial_probability m i p else 0) ≥ target_prob) :=
by
  sorry

end min_bulbs_needed_l193_193817


namespace density_zero_l193_193387

-- Define the set of integers n such that n divides a^(f(n)) - 1
noncomputable def S (a : ℕ) (f : ℤ[X]) : set ℕ :=
  { n | n ∣ a^((f.eval n).natAbs) - 1 }

-- Define the density function
def density (S : set ℕ) (N : ℕ) : ℚ :=
  (|{ n ∈ S | n ≤ N }| : ℚ) / (N : ℚ)

-- Formalize the main theorem statement
theorem density_zero (a : ℕ) (f : ℤ[X]) (ha : 1 < a) (hf : f.leadingCoeff > 0) :
  filter.Tendsto (λ N, density (S a f) N) filter.atTop (nhds 0) :=
sorry

end density_zero_l193_193387


namespace total_pens_bought_l193_193403

theorem total_pens_bought (r : ℕ) (hr : r > 10) (hm : 357 % r = 0) (ho : 441 % r = 0) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_bought_l193_193403


namespace inscribed_square_area_l193_193648

theorem inscribed_square_area (R : ℝ) (h : (R^2 * (π - 2) / 4) = (2 * π - 4)) : 
  ∃ (a : ℝ), a^2 = 16 := by
  sorry

end inscribed_square_area_l193_193648


namespace speed_of_journey_l193_193693

-- Define the conditions
def journey_time : ℕ := 10
def journey_distance : ℕ := 200
def half_journey_distance : ℕ := journey_distance / 2

-- Define the hypothesis that the journey is split into two equal parts, each traveled at the same speed
def equal_speed (v : ℕ) : Prop :=
  (half_journey_distance / v) + (half_journey_distance / v) = journey_time

-- Prove the speed v is 20 km/hr given the conditions
theorem speed_of_journey : ∃ v : ℕ, equal_speed v ∧ v = 20 :=
by
  have h : equal_speed 20 := sorry
  exact ⟨20, h, rfl⟩

end speed_of_journey_l193_193693


namespace domain_of_f_2x_plus_1_l193_193049

theorem domain_of_f_2x_plus_1 {f : ℝ → ℝ} :
  (∀ x, (-2 : ℝ) ≤ x ∧ x ≤ 3 → (-3 : ℝ) ≤ x - 1 ∧ x - 1 ≤ 2) →
  (∀ x, (-3 : ℝ) ≤ x ∧ x ≤ 2 → (-2 : ℝ) ≤ (x : ℝ) ∧ x ≤ 1/2) →
  ∀ x, (-2 : ℝ) ≤ x ∧ x ≤ 1 / 2 → ∀ y, y = 2 * x + 1 → (-3 : ℝ) ≤ y ∧ y ≤ 2 :=
by
  sorry

end domain_of_f_2x_plus_1_l193_193049


namespace max_x_squared_plus_y_squared_l193_193587

theorem max_x_squared_plus_y_squared (x y : ℝ) 
  (h : 3 * x^2 + 2 * y^2 = 2 * x) : x^2 + y^2 ≤ 4 / 9 :=
sorry

end max_x_squared_plus_y_squared_l193_193587


namespace factor_difference_of_squares_l193_193269

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l193_193269


namespace negation_of_proposition_l193_193644

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x > 0 → (x + 1/x) ≥ 2

-- Define the negation of the original proposition
def negation_prop : Prop := ∃ x > 0, x + 1/x < 2

-- State that the negation of the original proposition is the stated negation
theorem negation_of_proposition : (¬ ∀ x, original_prop x) ↔ negation_prop := 
by sorry

end negation_of_proposition_l193_193644


namespace max_min_difference_l193_193079

variable (x y z : ℝ)

theorem max_min_difference :
  x + y + z = 3 →
  x^2 + y^2 + z^2 = 18 →
  (max z (-z)) - ((min z (-z))) = 6 :=
  by
    intros h1 h2
    sorry

end max_min_difference_l193_193079


namespace layla_more_points_l193_193790

-- Definitions from the conditions
def layla_points : ℕ := 70
def total_points : ℕ := 112
def nahima_points : ℕ := total_points - layla_points

-- Theorem that states the proof problem
theorem layla_more_points : layla_points - nahima_points = 28 :=
by
  simp [layla_points, nahima_points]
  rw [nat.sub_sub]
  sorry

end layla_more_points_l193_193790


namespace geometric_series_product_l193_193724

theorem geometric_series_product (y : ℝ) :
  (∑'n : ℕ, (1 / 3 : ℝ) ^ n) * (∑'n : ℕ, (- 1 / 3 : ℝ) ^ n)
  = ∑'n : ℕ, (y⁻¹ : ℝ) ^ n ↔ y = 9 :=
by
  sorry

end geometric_series_product_l193_193724


namespace factorization_of_difference_of_squares_l193_193249

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l193_193249


namespace second_batch_students_l193_193118

theorem second_batch_students :
  ∃ x : ℕ,
    (40 * 45 + x * 55 + 60 * 65 : ℝ) / (40 + x + 60) = 56.333333333333336 ∧
    x = 50 :=
by
  use 50
  sorry

end second_batch_students_l193_193118


namespace rhombus_longer_diagonal_length_l193_193699

theorem rhombus_longer_diagonal_length
  (side_length : ℕ) (shorter_diagonal : ℕ) 
  (side_length_eq : side_length = 53) 
  (shorter_diagonal_eq : shorter_diagonal = 50) : 
  ∃ longer_diagonal : ℕ, longer_diagonal = 94 := by
  sorry

end rhombus_longer_diagonal_length_l193_193699


namespace solve_inequality_l193_193914

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 1 / x else (1 / 3) ^ x

theorem solve_inequality : { x : ℝ | |f x| ≥ 1 / 3 } = { x : ℝ | -3 ≤ x ∧ x ≤ 1 } :=
by
  sorry

end solve_inequality_l193_193914


namespace two_digit_number_as_expression_l193_193010

-- Define the conditions of the problem
variables (a : ℕ)

-- Statement to be proved
theorem two_digit_number_as_expression (h : 0 ≤ a ∧ a ≤ 9) : 10 * a + 1 = 10 * a + 1 := by
  sorry

end two_digit_number_as_expression_l193_193010


namespace base_b_for_256_l193_193507

theorem base_b_for_256 (b : ℕ) : b^3 ≤ 256 ∧ 256 < b^4 ↔ b = 5 :=
by
  split
  { intro h
    cases h with h1 h2
    have : 5 ≤ b :=
      Nat.le_of_lt_succ ((Nat.lt_of_le_of_lt h1) (Nat.lt_of_lt_of_le (by norm_num) h2)) sorry
  sorry
  
  sorry

end base_b_for_256_l193_193507


namespace lightbulbs_working_prob_at_least_5_with_99_percent_confidence_l193_193815

noncomputable def sufficient_lightbulbs (p : ℝ) (k : ℕ) (target_prob : ℝ) : ℕ :=
  let n := 7 -- this is based on our final answer
  have working_prob : (1 - p)^n := 0.95^n
  have non_working_prob : p := 0.05
  nat.run_until_p_ge_k (λ (x : ℝ), x + (nat.choose n k) * (working_prob)^k * (non_working_prob)^(n - k)) target_prob 7

theorem lightbulbs_working_prob_at_least_5_with_99_percent_confidence :
  sufficient_lightbulbs 0.05 5 0.99 = 7 :=
by
  sorry

end lightbulbs_working_prob_at_least_5_with_99_percent_confidence_l193_193815


namespace max_difference_in_flour_masses_l193_193515

/--
Given three brands of flour with the following mass ranges:
1. Brand A: (48 ± 0.1) kg
2. Brand B: (48 ± 0.2) kg
3. Brand C: (48 ± 0.3) kg

Prove that the maximum difference in mass between any two bags of these different brands is 0.5 kg.
-/
theorem max_difference_in_flour_masses :
  (∀ (a b : ℝ), ((47.9 ≤ a ∧ a ≤ 48.1) ∧ (47.8 ≤ b ∧ b ≤ 48.2)) →
    |a - b| ≤ 0.5) ∧
  (∀ (a c : ℝ), ((47.9 ≤ a ∧ a ≤ 48.1) ∧ (47.7 ≤ c ∧ c ≤ 48.3)) →
    |a - c| ≤ 0.5) ∧
  (∀ (b c : ℝ), ((47.8 ≤ b ∧ b ≤ 48.2) ∧ (47.7 ≤ c ∧ c ≤ 48.3)) →
    |b - c| ≤ 0.5) := 
sorry

end max_difference_in_flour_masses_l193_193515


namespace remaining_soup_feeds_adults_l193_193849

theorem remaining_soup_feeds_adults (C A k c : ℕ) 
    (hC : C= 10) 
    (hA : A = 5) 
    (hk : k = 8) 
    (hc : c = 20) : k - c / C * 10 * A = 30 := sorry

end remaining_soup_feeds_adults_l193_193849


namespace total_pens_bought_l193_193399

theorem total_pens_bought (r : ℕ) (hr : r > 10) (hm : 357 % r = 0) (ho : 441 % r = 0) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_bought_l193_193399


namespace customers_who_left_tip_l193_193011

-- Define the initial number of customers
def initial_customers : ℕ := 39

-- Define the additional number of customers during lunch rush
def additional_customers : ℕ := 12

-- Define the number of customers who didn't leave a tip
def no_tip_customers : ℕ := 49

-- Prove the number of customers who did leave a tip
theorem customers_who_left_tip : (initial_customers + additional_customers) - no_tip_customers = 2 := by
  sorry

end customers_who_left_tip_l193_193011


namespace dave_ice_cubes_total_l193_193733

theorem dave_ice_cubes_total : 
  let trayA_initial := 2
  let trayA_final := trayA_initial + 7
  let trayB := (1 / 3) * trayA_final
  let trayC := 2 * trayA_final
  trayA_final + trayB + trayC = 30 := by
  sorry

end dave_ice_cubes_total_l193_193733


namespace coeff_sum_l193_193599

open BigOperators

namespace Problem

def f (m n : ℕ) (f : ℕ → ℕ → ℕ) := f m n

theorem coeff_sum :
  let f : ℕ → ℕ → ℕ := λ m n, (Nat.choose 6 m) * (Nat.choose 4 n)
  f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 :=
by
  let f := λ m n, (Nat.choose 6 m) * (Nat.choose 4 n)
  show f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120
  sorry

end Problem

end coeff_sum_l193_193599


namespace coefficient_of_x3_in_expansion_l193_193600

noncomputable def binomial_expansion_coefficient (n r : ℕ) : ℕ :=
  Nat.choose n r

theorem coefficient_of_x3_in_expansion : 
  (∀ k : ℕ, binomial_expansion_coefficient 6 k ≤ binomial_expansion_coefficient 6 3) →
  binomial_expansion_coefficient 6 3 = 20 :=
by
  intro h
  -- skipping the proof
  sorry

end coefficient_of_x3_in_expansion_l193_193600


namespace verify_red_light_distribution_verify_intersection_distribution_verify_probability_at_least_one_red_light_l193_193151

namespace TrafficLightProblem

noncomputable def probability_red_light : ℝ := 1 / 3
noncomputable def probability_green_light : ℝ := 2 / 3

/-- The number of red lights encountered by the student on the way, denoted as ξ, follows a binomial distribution with n = 5, p = 1/3. -/
def red_light_distribution (k : ℕ) : ℝ :=
  if k ≤ 5 then 
    binom 5 k * (probability_red_light ^ k) * (probability_green_light ^ (5 - k))
  else 0

/-- The number of intersections passed before encountering a red light for the first time, denoted as η, 
  follows a distribution with:
  P(η = k) = (2/3)^k * (1/3) for 0 ≤ k < 5, and P(η = 5) = (2/3)^5 -/
def intersection_distribution (k : ℕ) : ℝ :=
  if k < 5 then 
    (probability_green_light ^ k) * probability_red_light
  else if k = 5 then
    probability_green_light ^ k
  else 0

/-- The probability that the student encounters at least one red light on the way is 211/243. -/
def probability_at_least_one_red_light : ℝ :=
  1 - (probability_green_light ^ 5)


theorem verify_red_light_distribution :
  ∀ k : ℕ, red_light_distribution k = if k ≤ 5 then 
    binom 5 k * (probability_red_light ^ k) * (probability_green_light ^ (5 - k))
  else 0 := sorry

theorem verify_intersection_distribution :
  ∀ k : ℕ, intersection_distribution k = if k < 5 then 
    (probability_green_light ^ k) * probability_red_light
  else if k = 5 then
    probability_green_light ^ k
  else 0 := sorry

theorem verify_probability_at_least_one_red_light :
  probability_at_least_one_red_light = 211 / 243 := sorry 

end TrafficLightProblem

end verify_red_light_distribution_verify_intersection_distribution_verify_probability_at_least_one_red_light_l193_193151


namespace neither_sufficient_nor_necessary_l193_193061

variable (a b : ℝ)

theorem neither_sufficient_nor_necessary (h1 : 0 < a * b ∧ a * b < 1) : ¬ (b < 1 / a) ∨ ¬ (1 / a < b) := by
  sorry

end neither_sufficient_nor_necessary_l193_193061


namespace max_volume_of_prism_l193_193341

theorem max_volume_of_prism (a b c s : ℝ) (h : a + b + c = 3 * s) : a * b * c ≤ s^3 :=
by {
    -- placeholder for the proof
    sorry
}

end max_volume_of_prism_l193_193341


namespace first_die_sides_l193_193783

theorem first_die_sides (n : ℕ) 
  (h_prob : (1 : ℝ) / n * (1 : ℝ) / 7 = 0.023809523809523808) : 
  n = 6 := by
  sorry

end first_die_sides_l193_193783


namespace maximum_value_l193_193077

theorem maximum_value (R P K : ℝ) (h₁ : 3 * Real.sqrt 3 * R ≥ P) (h₂ : K = P * R / 4) : 
  (K * P) / (R^3) ≤ 27 / 4 :=
by
  sorry

end maximum_value_l193_193077


namespace sqrt_identity_l193_193585

theorem sqrt_identity (x : ℝ) (hx : x = Real.sqrt 5 - 3) : Real.sqrt (x^2 + 6*x + 9) = Real.sqrt 5 :=
by
  sorry

end sqrt_identity_l193_193585


namespace polygon_a_largest_area_l193_193751

open Real

/-- Lean 4 statement to prove that Polygon A has the largest area among the given polygons -/
theorem polygon_a_largest_area :
  let area_polygon_a := 4 + 2 * (1 / 2 * 2 * 2)
  let area_polygon_b := 3 + 3 * (1 / 2 * 1 * 1)
  let area_polygon_c := 6
  let area_polygon_d := 5 + (1 / 2) * π * 1^2
  let area_polygon_e := 7
  area_polygon_a > area_polygon_b ∧
  area_polygon_a > area_polygon_c ∧
  area_polygon_a > area_polygon_d ∧
  area_polygon_a > area_polygon_e :=
by
  let area_polygon_a := 4 + 2 * (1 / 2 * 2 * 2)
  let area_polygon_b := 3 + 3 * (1 / 2 * 1 * 1)
  let area_polygon_c := 6
  let area_polygon_d := 5 + (1 / 2) * π * 1^2
  let area_polygon_e := 7
  sorry

end polygon_a_largest_area_l193_193751


namespace layla_more_points_than_nahima_l193_193789

theorem layla_more_points_than_nahima (layla_points : ℕ) (total_points : ℕ) (h1 : layla_points = 70) (h2 : total_points = 112) :
  layla_points - (total_points - layla_points) = 28 :=
by
  sorry

end layla_more_points_than_nahima_l193_193789


namespace sum_of_six_least_n_l193_193388

def tau (n : ℕ) : ℕ := Nat.totient n -- Assuming as an example for tau definition

theorem sum_of_six_least_n (h1 : tau 8 + tau 9 = 7)
                           (h2 : tau 9 + tau 10 = 7)
                           (h3 : tau 16 + tau 17 = 7)
                           (h4 : tau 25 + tau 26 = 7)
                           (h5 : tau 121 + tau 122 = 7)
                           (h6 : tau 361 + tau 362 = 7) :
  8 + 9 + 16 + 25 + 121 + 361 = 540 :=
by sorry

end sum_of_six_least_n_l193_193388


namespace total_balloons_l193_193343

theorem total_balloons (fred_balloons : ℕ) (sam_balloons : ℕ) (mary_balloons : ℕ) :
  fred_balloons = 5 → sam_balloons = 6 → mary_balloons = 7 → fred_balloons + sam_balloons + mary_balloons = 18 :=
by
  intros
  sorry

end total_balloons_l193_193343


namespace gcd_102_238_l193_193109

-- Define the two numbers involved
def a : ℕ := 102
def b : ℕ := 238

-- State the theorem
theorem gcd_102_238 : Int.gcd a b = 34 :=
by
  sorry

end gcd_102_238_l193_193109


namespace no_fractions_meet_condition_l193_193548

theorem no_fractions_meet_condition :
  ∀ x y : ℕ, Nat.coprime x y → 
  (x > 0) → (y > 0) → 
  ((x + 1) / (y + 1) = (6 * x) / (5 * y)) → 
  false := by
  sorry

end no_fractions_meet_condition_l193_193548


namespace factorization_difference_of_squares_l193_193212

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l193_193212


namespace min_value_sin_cos_squared_six_l193_193894

theorem min_value_sin_cos_squared_six (x : ℝ) :
  ∃ x : ℝ, (sin^6 x + 2 * cos^6 x) = 2/3 :=
sorry

end min_value_sin_cos_squared_six_l193_193894


namespace factorize_x_squared_minus_one_l193_193191

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l193_193191


namespace factorize_difference_of_squares_l193_193296

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l193_193296


namespace arithmetic_geometric_sequence_l193_193045

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

end arithmetic_geometric_sequence_l193_193045


namespace tangent_product_equals_2_pow_23_l193_193136

noncomputable def tangent_product : ℝ :=
  (1 + Real.tan (1 * Real.pi / 180)) *
  (1 + Real.tan (2 * Real.pi / 180)) *
  (1 + Real.tan (3 * Real.pi / 180)) *
  (1 + Real.tan (4 * Real.pi / 180)) *
  (1 + Real.tan (5 * Real.pi / 180)) *
  (1 + Real.tan (6 * Real.pi / 180)) *
  (1 + Real.tan (7 * Real.pi / 180)) *
  (1 + Real.tan (8 * Real.pi / 180)) *
  (1 + Real.tan (9 * Real.pi / 180)) *
  (1 + Real.tan (10 * Real.pi / 180)) *
  (1 + Real.tan (11 * Real.pi / 180)) *
  (1 + Real.tan (12 * Real.pi / 180)) *
  (1 + Real.tan (13 * Real.pi / 180)) *
  (1 + Real.tan (14 * Real.pi / 180)) *
  (1 + Real.tan (15 * Real.pi / 180)) *
  (1 + Real.tan (16 * Real.pi / 180)) *
  (1 + Real.tan (17 * Real.pi / 180)) *
  (1 + Real.tan (18 * Real.pi / 180)) *
  (1 + Real.tan (19 * Real.pi / 180)) *
  (1 + Real.tan (20 * Real.pi / 180)) *
  (1 + Real.tan (21 * Real.pi / 180)) *
  (1 + Real.tan (22 * Real.pi / 180)) *
  (1 + Real.tan (23 * Real.pi / 180)) *
  (1 + Real.tan (24 * Real.pi / 180)) *
  (1 + Real.tan (25 * Real.pi / 180)) *
  (1 + Real.tan (26 * Real.pi / 180)) *
  (1 + Real.tan (27 * Real.pi / 180)) *
  (1 + Real.tan (28 * Real.pi / 180)) *
  (1 + Real.tan (29 * Real.pi / 180)) *
  (1 + Real.tan (30 * Real.pi / 180)) *
  (1 + Real.tan (31 * Real.pi / 180)) *
  (1 + Real.tan (32 * Real.pi / 180)) *
  (1 + Real.tan (33 * Real.pi / 180)) *
  (1 + Real.tan (34 * Real.pi / 180)) *
  (1 + Real.tan (35 * Real.pi / 180)) *
  (1 + Real.tan (36 * Real.pi / 180)) *
  (1 + Real.tan (37 * Real.pi / 180)) *
  (1 + Real.tan (38 * Real.pi / 180)) *
  (1 + Real.tan (39 * Real.pi / 180)) *
  (1 + Real.tan (40 * Real.pi / 180)) *
  (1 + Real.tan (41 * Real.pi / 180)) *
  (1 + Real.tan (42 * Real.pi / 180)) *
  (1 + Real.tan (43 * Real.pi / 180)) *
  (1 + Real.tan (44 * Real.pi / 180)) *
  (1 + Real.tan (45 * Real.pi / 180))

theorem tangent_product_equals_2_pow_23 : tangent_product = 2 ^ 23 :=
  sorry

end tangent_product_equals_2_pow_23_l193_193136


namespace rate_of_interest_is_12_percent_l193_193843

variables (P r : ℝ)
variables (A5 A8 : ℝ)

-- Given conditions: 
axiom A5_condition : A5 = 9800
axiom A8_condition : A8 = 12005
axiom simple_interest_5_year : A5 = P + 5 * P * r / 100
axiom simple_interest_8_year : A8 = P + 8 * P * r / 100

-- The statement we aim to prove
theorem rate_of_interest_is_12_percent : r = 12 := 
sorry

end rate_of_interest_is_12_percent_l193_193843


namespace first_duck_fraction_l193_193119

-- Definitions based on the conditions
variable (total_bread : ℕ) (left_bread : ℕ) (second_duck_bread : ℕ) (third_duck_bread : ℕ)

-- Given values
def given_values : Prop :=
  total_bread = 100 ∧ left_bread = 30 ∧ second_duck_bread = 13 ∧ third_duck_bread = 7

-- Proof statement
theorem first_duck_fraction (h : given_values total_bread left_bread second_duck_bread third_duck_bread) :
  (total_bread - left_bread) - (second_duck_bread + third_duck_bread) = 1/2 * total_bread := by 
  sorry

end first_duck_fraction_l193_193119


namespace log_self_solve_l193_193734

theorem log_self_solve (x : ℝ) (h : x = Real.log 3 (64 + x)) : x = 4 :=
by 
  sorry

end log_self_solve_l193_193734


namespace total_pens_l193_193439

theorem total_pens (r : ℕ) (r_gt_10 : r > 10) (r_div_357 : r ∣ 357) (r_div_441 : r ∣ 441) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_l193_193439


namespace h_h3_eq_3568_l193_193923

def h (x : ℤ) := 3 * x ^ 2 + 3 * x - 2

theorem h_h3_eq_3568 : h (h 3) = 3568 := by
  sorry

end h_h3_eq_3568_l193_193923


namespace no_fractions_meet_condition_l193_193549

theorem no_fractions_meet_condition :
  ∀ x y : ℕ, Nat.coprime x y → 
  (x > 0) → (y > 0) → 
  ((x + 1) / (y + 1) = (6 * x) / (5 * y)) → 
  false := by
  sorry

end no_fractions_meet_condition_l193_193549


namespace factorize_x_squared_minus_one_l193_193200

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l193_193200


namespace inequality_solution_l193_193626

noncomputable def inequality (x : ℝ) : Prop :=
  ((x - 1) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0

theorem inequality_solution :
  {x : ℝ | inequality x} = {x : ℝ | x < 1} ∪ {x : ℝ | 2 < x ∧ x < 4} ∪ {x : ℝ | 5 < x ∧ x < 6} ∪ {x : ℝ | 7 < x} :=
by
  sorry

end inequality_solution_l193_193626


namespace pages_with_same_units_digit_count_l193_193514

def same_units_digit (x : ℕ) (y : ℕ) : Prop :=
  x % 10 = y % 10

theorem pages_with_same_units_digit_count :
  ∃! (n : ℕ), n = 12 ∧ 
  ∀ x, (1 ≤ x ∧ x ≤ 61) → same_units_digit x (62 - x) → 
  (x % 10 = 2 ∨ x % 10 = 7) :=
by
  sorry

end pages_with_same_units_digit_count_l193_193514


namespace factorization_difference_of_squares_l193_193208

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l193_193208


namespace maximize_z_l193_193904

open Real

theorem maximize_z (x y : ℝ) (h1 : x + y ≤ 10) (h2 : 3 * x + y ≤ 18) (h3 : 0 ≤ x) (h4 : 0 ≤ y) :
  (∀ x y, x + y ≤ 10 ∧ 3 * x + y ≤ 18 ∧ 0 ≤ x ∧ 0 ≤ y → x + y / 2 ≤ 7) :=
by
  sorry

end maximize_z_l193_193904


namespace factorization_of_x_squared_minus_one_l193_193239

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l193_193239


namespace number_of_such_fractions_is_one_l193_193553

theorem number_of_such_fractions_is_one :
  ∃! (x y : ℕ), (Nat.gcd x y = 1) ∧ (1 < x) ∧ (1 < y) ∧ (x+1)/(y+1) = (6*x)/(5*y) := 
begin
  sorry
end

end number_of_such_fractions_is_one_l193_193553


namespace factorize_difference_of_squares_l193_193224

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l193_193224


namespace jack_second_half_time_l193_193381

variable (jacksFirstHalf : ℕ) (jillTotalTime : ℕ) (timeDifference : ℕ)

def jacksTotalTime : ℕ := jillTotalTime - timeDifference

def jacksSecondHalf (jacksFirstHalf jacksTotalTime : ℕ) : ℕ :=
  jacksTotalTime - jacksFirstHalf

theorem jack_second_half_time : 
  jacksFirstHalf = 19 ∧ jillTotalTime = 32 ∧ timeDifference = 7 → jacksSecondHalf jacksFirstHalf (jacksTotalTime jillTotalTime timeDifference) = 6 :=
by
  intros h
  cases h with h1 h'
  cases h' with h2 h3
  rw [h1, h2, h3]
  unfold jacksTotalTime
  unfold jacksSecondHalf
  norm_num


end jack_second_half_time_l193_193381


namespace sausage_more_than_pepperoni_l193_193614

noncomputable def pieces_of_meat_per_slice : ℕ := 22
noncomputable def slices : ℕ := 6
noncomputable def total_pieces_of_meat : ℕ := pieces_of_meat_per_slice * slices

noncomputable def pieces_of_pepperoni : ℕ := 30
noncomputable def pieces_of_ham : ℕ := 2 * pieces_of_pepperoni

noncomputable def total_pieces_of_meat_without_sausage : ℕ := pieces_of_pepperoni + pieces_of_ham
noncomputable def pieces_of_sausage : ℕ := total_pieces_of_meat - total_pieces_of_meat_without_sausage

theorem sausage_more_than_pepperoni : (pieces_of_sausage - pieces_of_pepperoni) = 12 := by
  sorry

end sausage_more_than_pepperoni_l193_193614


namespace factorize_x_squared_minus_one_l193_193192

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l193_193192


namespace relationship_of_ys_l193_193350

variables {k y1 y2 y3 : ℝ}

theorem relationship_of_ys (h : k < 0) 
  (h1 : y1 = k / -4) 
  (h2 : y2 = k / 2) 
  (h3 : y3 = k / 3) : 
  y1 > y3 ∧ y3 > y2 :=
by 
  sorry

end relationship_of_ys_l193_193350


namespace factorization_of_difference_of_squares_l193_193255

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l193_193255


namespace factor_difference_of_squares_l193_193265

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l193_193265


namespace factorize_x_squared_minus_1_l193_193277

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l193_193277


namespace no_three_digit_numbers_with_sum_27_are_even_l193_193882

-- We define a 3-digit number and its conditions based on digit-sum and even properties
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + (n % 100 / 10) + (n % 10)

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

theorem no_three_digit_numbers_with_sum_27_are_even :
  ¬ ∃ n : ℕ, is_three_digit n ∧ digit_sum n = 27 ∧ is_even n :=
by sorry

end no_three_digit_numbers_with_sum_27_are_even_l193_193882


namespace find_X_plus_Y_l193_193470

-- Statement of the problem translated from the given problem-solution pair.
theorem find_X_plus_Y (X Y : ℚ) :
  (∀ x : ℚ, x ≠ 5 → x ≠ 6 →
    (Y * x + 8) / (x^2 - 11 * x + 30) = X / (x - 5) + 7 / (x - 6)) →
  X + Y = -22 / 3 :=
by
  sorry

end find_X_plus_Y_l193_193470


namespace inequality_proof_l193_193569

theorem inequality_proof (x : ℝ) : 3 * x - 6 > 5 * (x - 2) → x < 2 :=
by
  sorry

end inequality_proof_l193_193569


namespace base7_number_l193_193967

theorem base7_number (A B C : ℕ) (h1 : 1 ≤ A ∧ A ≤ 6) (h2 : 1 ≤ B ∧ B ≤ 6) (h3 : 1 ≤ C ∧ C ≤ 6)
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_condition1 : B + C = 7)
  (h_condition2 : A + 1 = C)
  (h_condition3 : A + B = C) :
  A = 5 ∧ B = 1 ∧ C = 6 :=
sorry

end base7_number_l193_193967


namespace total_pens_bought_l193_193400

theorem total_pens_bought (r : ℕ) (hr : r > 10) (hm : 357 % r = 0) (ho : 441 % r = 0) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_bought_l193_193400


namespace fifth_power_ends_with_same_digit_l193_193170

theorem fifth_power_ends_with_same_digit (a : ℕ) : a^5 % 10 = a % 10 :=
by sorry

end fifth_power_ends_with_same_digit_l193_193170


namespace solution_satisfies_conditions_l193_193502

noncomputable def sqrt_eq (a b x y : ℝ) : Prop :=
  sqrt (x - y) = a ∧ sqrt (x + y) = b

theorem solution_satisfies_conditions 
  (x y : ℝ)
  (h1 : sqrt_eq (2/5) 2 x y)
  (hexact: x = 52/25 ∧ y = 48/25) :
  sqrt_eq (2/5) 2 x y ∧ 
  (x * y = 8/25) :=
by
  sorry

end solution_satisfies_conditions_l193_193502


namespace complete_the_square_l193_193801

-- Define the quadratic expression as a function.
def quad_expr (k : ℚ) : ℚ := 8 * k^2 + 12 * k + 18

-- Define the completed square form.
def completed_square_expr (k : ℚ) : ℚ := 8 * (k + 3 / 4)^2 + 27 / 2

-- Theorem stating the equality of the original expression in completed square form and the value of r + s.
theorem complete_the_square : ∀ k : ℚ, quad_expr k = completed_square_expr k ∧ (3 / 4 + 27 / 2 = 57 / 4) :=
by
  intro k
  sorry

end complete_the_square_l193_193801


namespace find_value_of_z_l193_193047

theorem find_value_of_z (z : ℂ) (h1 : ∀ a : ℝ, z = a * I) (h2 : ((z + 2) / (1 - I)).im = 0) : z = -2 * I :=
sorry

end find_value_of_z_l193_193047


namespace probability_of_7_successes_in_7_trials_l193_193667

open Probability

/-- Define the given conditions for the problem -/
def n : ℕ := 7
def k : ℕ := 7
def p : ℚ := 2 / 7

/-- The binomial coefficient and the probability of success in n trials -/
theorem probability_of_7_successes_in_7_trials :
  P(X = k) = (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) :=
by
  have bep_0 : nat.choose 7 7 = 1, from sorry,
  have p_power_k : p ^ k = (2 / 7) ^ 7, from sorry,
  have q_power_rem : (1 - p) ^ (n - k) = 1, from sorry,
  have p_eq_frac : (2 / 7) ^ 7 * 1 = 128 / 823543, from sorry,
  show 1 * (2 / 7) ^ 7 * 1 = 128 / 823543, by sorry

end probability_of_7_successes_in_7_trials_l193_193667


namespace ratio_new_values_l193_193663

theorem ratio_new_values (x y x2 y2 : ℝ) (h1 : x / y = 7 / 5) (h2 : x2 = x * y) (h3 : y2 = y * x) : x2 / y2 = 1 := by
  sorry

end ratio_new_values_l193_193663


namespace hyperbola_equation_l193_193358

theorem hyperbola_equation (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b) 
  (h₃ : (2^2 / a^2) - (1^2 / b^2) = 1) (h₄ : a^2 + b^2 = 3) :
  (∀ x y : ℝ,  (x^2 / 2) - y^2 = 1) :=
by 
  sorry

end hyperbola_equation_l193_193358


namespace fraction_of_B_l193_193679

theorem fraction_of_B (A B C : ℝ) 
  (h1 : A = (1/3) * (B + C)) 
  (h2 : A = B + 20) 
  (h3 : A + B + C = 720) : 
  B / (A + C) = 2 / 7 :=
  by 
  sorry

end fraction_of_B_l193_193679


namespace complex_quadrant_l193_193808

-- Define the imaginary unit
def i := Complex.I

-- Define the complex number z satisfying the given condition
variables (z : Complex)
axiom h : (3 - 2 * i) * z = 4 + 3 * i

-- Statement for the proof problem
theorem complex_quadrant (h : (3 - 2 * i) * z = 4 + 3 * i) : 
  (0 < z.re ∧ 0 < z.im) :=
sorry

end complex_quadrant_l193_193808


namespace product_of_two_consecutive_integers_sum_lt_150_l193_193175

theorem product_of_two_consecutive_integers_sum_lt_150 :
  ∃ (n : Nat), n * (n + 1) = 5500 ∧ 2 * n + 1 < 150 :=
by
  sorry

end product_of_two_consecutive_integers_sum_lt_150_l193_193175


namespace inequality_always_true_l193_193706

theorem inequality_always_true (x : ℝ) : (4 * x) / (x ^ 2 + 4) ≤ 1 := by
  sorry

end inequality_always_true_l193_193706


namespace rectangular_prism_inequalities_l193_193347

variable {a b c : ℝ}

noncomputable def p (a b c : ℝ) := 4 * (a + b + c)
noncomputable def S (a b c : ℝ) := 2 * (a * b + b * c + c * a)
noncomputable def d (a b c : ℝ) := Real.sqrt (a^2 + b^2 + c^2)

theorem rectangular_prism_inequalities (h : a > b) (h1 : b > c) :
  a > (1 / 3) * (p a b c / 4 + Real.sqrt (d a b c ^ 2 - (1 / 2) * S a b c)) ∧
  c < (1 / 3) * (p a b c / 4 - Real.sqrt (d a b c ^ 2 - (1 / 2) * S a b c)) :=
by
  sorry

end rectangular_prism_inequalities_l193_193347


namespace jack_second_half_time_l193_193386

def time_jack_first_half := 19
def time_between_jill_and_jack := 7
def time_jill := 32

def time_jack (time_jill time_between_jill_and_jack : ℕ) : ℕ :=
  time_jill - time_between_jill_and_jack

def time_jack_second_half (time_jack time_jack_first_half : ℕ) : ℕ :=
  time_jack - time_jack_first_half

theorem jack_second_half_time :
  time_jack_second_half (time_jack time_jill time_between_jill_and_jack) time_jack_first_half = 6 :=
by
  sorry

end jack_second_half_time_l193_193386


namespace sum_of_b_for_one_solution_l193_193060

theorem sum_of_b_for_one_solution :
  let A := 3
  let C := 12
  ∀ b : ℝ, ((b + 5)^2 - 4 * A * C = 0) → (b = 7 ∨ b = -17) → (7 + (-17)) = -10 :=
by
  intro A C b
  sorry

end sum_of_b_for_one_solution_l193_193060


namespace factorize_x_squared_minus_1_l193_193273

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l193_193273


namespace smallest_k_l193_193749

def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k (k : ℕ) : (∀ z : ℂ, z ≠ 0 → f z ∣ z^k - 1) ↔ k = 40 :=
by sorry

end smallest_k_l193_193749


namespace goshawk_nature_reserve_l193_193070

-- Define the problem statement and conditions
def percent_hawks (H W K : ℝ) : Prop :=
  ∃ H W K : ℝ,
    -- Condition 1: 35% of the birds are neither hawks, paddyfield-warblers, nor kingfishers
    1 - (H + W + K) = 0.35 ∧
    -- Condition 2: 40% of the non-hawks are paddyfield-warblers
    W = 0.40 * (1 - H) ∧
    -- Condition 3: There are 25% as many kingfishers as paddyfield-warblers
    K = 0.25 * W ∧
    -- Given all conditions, calculate the percentage of hawks
    H = 0.65

theorem goshawk_nature_reserve :
  ∃ H W K : ℝ,
    1 - (H + W + K) = 0.35 ∧
    W = 0.40 * (1 - H) ∧
    K = 0.25 * W ∧
    H = 0.65 := by
    -- Proof is omitted
    sorry

end goshawk_nature_reserve_l193_193070


namespace rectangle_solution_l193_193501

-- Define the given conditions
variables (x y : ℚ)

-- Given equations
def condition1 := (Real.sqrt (x - y) = 2 / 5)
def condition2 := (Real.sqrt (x + y) = 2)

-- Solution
theorem rectangle_solution (x y : ℚ) (h1 : condition1 x y) (h2 : condition2 x y) : 
  x = 52 / 25 ∧ y = 48 / 25 ∧ (Real.sqrt ((52 / 25) * (48 / 25)) = 8 / 25) :=
by
  sorry

end rectangle_solution_l193_193501


namespace total_pens_l193_193418

/-- Proof that Masha and Olya bought a total of 38 pens given the cost conditions. -/
theorem total_pens (r : ℕ) (h_r : r > 10) (h1 : 357 % r = 0) (h2 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193418


namespace minimum_lightbulbs_needed_l193_193821

theorem minimum_lightbulbs_needed 
  (p : ℝ) (h : p = 0.95) : ∃ (n : ℕ), n = 7 ∧ (1 - ∑ k in finset.range 5, (nat.choose n k : ℝ) * p^k * (1 - p)^(n - k) ) ≥ 0.99 := 
by 
  sorry

end minimum_lightbulbs_needed_l193_193821


namespace find_a_l193_193905

-- Define sets A and B based on the given real number a
def A (a : ℝ) : Set ℝ := {a^2, a + 1, -3}
def B (a : ℝ) : Set ℝ := {a - 3, 3 * a - 1, a^2 + 1}

-- Given condition
def condition (a : ℝ) : Prop := A a ∩ B a = {-3}

-- Prove that a = -2/3 is the solution satisfying the condition
theorem find_a : ∃ a : ℝ, condition a ∧ a = -2/3 :=
by
  sorry  -- Proof goes here

end find_a_l193_193905


namespace factorization_of_difference_of_squares_l193_193252

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l193_193252


namespace total_pens_l193_193409

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193409


namespace avg_GPA_is_93_l193_193632

def avg_GPA_school (GPA_6th GPA_8th : ℕ) (GPA_diff : ℕ) : ℕ :=
  (GPA_6th + (GPA_6th + GPA_diff) + GPA_8th) / 3

theorem avg_GPA_is_93 :
  avg_GPA_school 93 91 2 = 93 :=
by
  -- The proof can be handled here 
  sorry

end avg_GPA_is_93_l193_193632


namespace max_servings_l193_193992

/-- To prepare one serving of salad we need:
  - 2 cucumbers
  - 2 tomatoes
  - 75 grams of brynza
  - 1 pepper
  The warehouse has the following quantities:
  - 60 peppers
  - 4200 grams of brynza (4.2 kg)
  - 116 tomatoes
  - 117 cucumbers
  We want to prove the maximum number of salad servings we can make is 56.
-/
theorem max_servings (peppers : ℕ) (brynza : ℕ) (tomatoes : ℕ) (cucumbers : ℕ) 
  (h_peppers : peppers = 60)
  (h_brynza : brynza = 4200)
  (h_tomatoes : tomatoes = 116)
  (h_cucumbers : cucumbers = 117) :
  let servings := min (min (peppers / 1) (brynza / 75)) (min (tomatoes / 2) (cucumbers / 2)) in
  servings = 56 := 
by
  sorry

end max_servings_l193_193992


namespace seventh_term_geometric_sequence_l193_193873

theorem seventh_term_geometric_sequence (a : ℝ) (a3 : ℝ) (r : ℝ) (n : ℕ) (term : ℕ → ℝ)
    (h_a : a = 3)
    (h_a3 : a3 = 3 / 64)
    (h_term : ∀ n, term n = a * r ^ (n - 1))
    (h_r : r = 1 / 8) :
    term 7 = 3 / 262144 :=
by
  sorry

end seventh_term_geometric_sequence_l193_193873


namespace simplify_fraction_l193_193169

theorem simplify_fraction (x : ℝ) (h : x ≠ 2) : (x^2 / (x - 2) - 4 / (x - 2)) = x + 2 := by
  sorry

end simplify_fraction_l193_193169


namespace integer_classes_mod4_l193_193377

theorem integer_classes_mod4:
  (2021 % 4) = 1 ∧ (∀ a b : ℤ, (a % 4 = 2) ∧ (b % 4 = 3) → (a + b) % 4 = 1) := by
  sorry

end integer_classes_mod4_l193_193377


namespace painting_faces_not_sum_to_nine_l193_193162

def eight_sided_die_numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

def pairs_that_sum_to_nine : List (ℕ × ℕ) := [(1, 8), (2, 7), (3, 6), (4, 5)]

theorem painting_faces_not_sum_to_nine :
  let total_pairs := (eight_sided_die_numbers.length * (eight_sided_die_numbers.length - 1)) / 2
  let invalid_pairs := pairs_that_sum_to_nine.length
  total_pairs - invalid_pairs = 24 :=
by
  sorry

end painting_faces_not_sum_to_nine_l193_193162


namespace total_pens_l193_193441

theorem total_pens (r : ℕ) (r_gt_10 : r > 10) (r_div_357 : r ∣ 357) (r_div_441 : r ∣ 441) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_l193_193441


namespace puzzles_sold_correct_l193_193472

def science_kits_sold : ℕ := 45
def puzzles_sold : ℕ := science_kits_sold - 9

theorem puzzles_sold_correct : puzzles_sold = 36 := by
  -- Proof will be provided here
  sorry

end puzzles_sold_correct_l193_193472


namespace factorize_x_squared_minus_one_l193_193195

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l193_193195


namespace total_pens_l193_193447

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193447


namespace actual_cost_of_article_l193_193861

theorem actual_cost_of_article (x : ℝ) (h : 0.60 * x = 1050) : x = 1750 := by
  sorry

end actual_cost_of_article_l193_193861


namespace total_pens_bought_l193_193398

theorem total_pens_bought (r : ℕ) (hr : r > 10) (hm : 357 % r = 0) (ho : 441 % r = 0) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_bought_l193_193398


namespace range_of_k_l193_193573

def f : ℝ → ℝ := sorry

axiom cond1 (a b : ℝ) : f (a + b) = f a + f b + 2 * a * b
axiom cond2 (k : ℝ) : ∀ x : ℝ, f (x + k) = f (k - x)
axiom cond3 : ∀ x y : ℝ, 1 ≤ x → x ≤ y → y ≤ 2 → f x ≤ f y

theorem range_of_k (k : ℝ) : k ≤ 1 :=
sorry

end range_of_k_l193_193573


namespace fish_in_pond_estimate_l193_193584

noncomputable def L (N : ℕ) : ℚ :=
  (nat.choose 20 7) * (((N - 20).choose 43) * (50!).to_rat / 
  ((43!).to_rat * (N.choose 50).to_rat))

theorem fish_in_pond_estimate :
  let n_a := 20
  let m := 50
  let k_1 := 7
  let N := 142 in
  L N = nat.choose 20 7 * (((N - 20).choose 43) * 
  (50!).to_rat / ((43!).to_rat * (N.choose 50).to_rat)) := sorry

end fish_in_pond_estimate_l193_193584


namespace maple_trees_cut_down_l193_193654

-- Define the initial number of maple trees.
def initial_maple_trees : ℝ := 9.0

-- Define the final number of maple trees after cutting.
def final_maple_trees : ℝ := 7.0

-- Define the number of maple trees cut down.
def cut_down_maple_trees : ℝ := initial_maple_trees - final_maple_trees

-- Prove that the number of cut down maple trees is 2.
theorem maple_trees_cut_down : cut_down_maple_trees = 2 := by
  sorry

end maple_trees_cut_down_l193_193654


namespace sum_of_ages_is_12_l193_193482

-- Let Y be the age of the youngest child
def Y : ℝ := 1.5

-- Let the ages of the other children
def age2 : ℝ := Y + 1
def age3 : ℝ := Y + 2
def age4 : ℝ := Y + 3

-- Define the sum of the ages
def sum_of_ages : ℝ := Y + age2 + age3 + age4

-- The theorem to prove the sum of the ages is 12 years
theorem sum_of_ages_is_12 : sum_of_ages = 12 :=
by
  -- The detailed proof is to be filled in later, currently skipped.
  sorry

end sum_of_ages_is_12_l193_193482


namespace gcd_102_238_l193_193112

theorem gcd_102_238 : Int.gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l193_193112


namespace camping_trip_percentage_l193_193062

theorem camping_trip_percentage (T : ℝ)
  (h1 : 16 / 100 ≤ 1)
  (h2 : T - 16 / 100 ≤ 1)
  (h3 : T = 64 / 100) :
  T = 64 / 100 := by
  sorry

end camping_trip_percentage_l193_193062


namespace compute_y_geometric_series_l193_193726

theorem compute_y_geometric_series :
  let s1 := ∑' n : ℕ, (1 / 3) ^ n,
      s2 := ∑' n : ℕ, (-1) ^ n * (1 / 3) ^ n in
  s1 = 3 / 2 →
  s2 = 3 / 4 →
  (1 + s1) * (1 + s2) = 1 +
  ∑' n : ℕ, (1 / 9) ^ n :=
by
  sorry

end compute_y_geometric_series_l193_193726


namespace quadratic_inequality_solution_set_l193_193103

theorem quadratic_inequality_solution_set {x : ℝ} : 
  x^2 < x + 6 ↔ (-2 < x ∧ x < 3) :=
by
  sorry

end quadratic_inequality_solution_set_l193_193103


namespace card_draw_count_l193_193651

theorem card_draw_count : 
  let total_cards := 12
  let red_cards := 3
  let yellow_cards := 3
  let blue_cards := 3
  let green_cards := 3
  let total_ways := Nat.choose total_cards 3
  let invalid_same_color := 4 * Nat.choose 3 3
  let invalid_two_red := Nat.choose red_cards 2 * Nat.choose (total_cards - red_cards) 1
  total_ways - invalid_same_color - invalid_two_red = 189 :=
by
  sorry

end card_draw_count_l193_193651


namespace total_pens_l193_193414

/-- Proof that Masha and Olya bought a total of 38 pens given the cost conditions. -/
theorem total_pens (r : ℕ) (h_r : r > 10) (h1 : 357 % r = 0) (h2 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193414


namespace compute_value_l193_193940

theorem compute_value {a b : ℝ} 
  (h1 : ∀ x, (x + a) * (x + b) * (x + 12) = 0 → x ≠ -3 → x = -a ∨ x = -b ∨ x = -12)
  (h2 : ∀ x, (x + 2 * a) * (x + 3) * (x + 6) = 0 → x ≠ -b ∧ x ≠ -12 → x = -3) :
  100 * (3 / 2) + 6 = 156 :=
by
  sorry

end compute_value_l193_193940


namespace base_length_of_isosceles_triangle_l193_193778

-- Definitions for the problem
def isosceles_triangle (A B C : Type) [Nonempty A] [Nonempty B] [Nonempty C] :=
  ∃ (AB BC : ℝ), AB = BC

-- The problem to prove
theorem base_length_of_isosceles_triangle
  {A B C : Type} [Nonempty A] [Nonempty B] [Nonempty C]
  (AB BC : ℝ) (AC x : ℝ)
  (height_base : ℝ) (height_side : ℝ) 
  (h1 : AB = BC)
  (h2 : height_base = 10)
  (h3 : height_side = 12)
  (h4 : AC = x)
  (h5 : ∀ AE BD : ℝ, AE = height_side → BD = height_base) :
  x = 15 := by sorry

end base_length_of_isosceles_triangle_l193_193778


namespace sweets_remainder_l193_193566

theorem sweets_remainder (m : ℕ) (h : m % 7 = 6) : (4 * m) % 7 = 3 :=
by
  sorry

end sweets_remainder_l193_193566


namespace factorize_x_squared_minus_1_l193_193283

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l193_193283


namespace average_interest_rate_l193_193153

theorem average_interest_rate (x : ℝ) (h1 : 0 < x ∧ x < 6000)
  (h2 : 0.03 * (6000 - x) = 0.055 * x) :
  ((0.03 * (6000 - x) + 0.055 * x) / 6000) = 0.0388 :=
by
  sorry

end average_interest_rate_l193_193153


namespace find_a_given_even_l193_193911

def f (x a : ℝ) : ℝ := (x + a) * (x - 4)

theorem find_a_given_even (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 4 :=
by
  unfold f
  sorry

end find_a_given_even_l193_193911


namespace child_current_height_l193_193698

variable (h_last_visit : ℝ) (h_grown : ℝ)

-- Conditions
def last_height (h_last_visit : ℝ) := h_last_visit = 38.5
def height_grown (h_grown : ℝ) := h_grown = 3

-- Theorem statement
theorem child_current_height (h_last_visit h_grown : ℝ) 
    (h_last : last_height h_last_visit) 
    (h_grow : height_grown h_grown) : 
    h_last_visit + h_grown = 41.5 :=
by
  sorry

end child_current_height_l193_193698


namespace class_funds_l193_193074

theorem class_funds (total_contribution : ℕ) (students : ℕ) (contribution_per_student : ℕ) (remaining_amount : ℕ) 
    (h1 : total_contribution = 90) 
    (h2 : students = 19) 
    (h3 : contribution_per_student = 4) 
    (h4 : remaining_amount = total_contribution - (students * contribution_per_student)) : 
    remaining_amount = 14 :=
sorry

end class_funds_l193_193074


namespace find_x_unique_l193_193325

def productOfDigits (x : ℕ) : ℕ :=
  -- Assuming the implementation of product of digits function
  sorry

def sumOfDigits (x : ℕ) : ℕ :=
  -- Assuming the implementation of sum of digits function
  sorry

theorem find_x_unique : ∀ x : ℕ, (productOfDigits x = 44 * x - 86868 ∧ ∃ n : ℕ, sumOfDigits x = n^3) -> x = 1989 :=
by
  intros x h
  sorry

end find_x_unique_l193_193325


namespace fraction_of_sum_l193_193851

theorem fraction_of_sum (S n : ℝ) (h1 : n = S / 6) : n / (S + n) = 1 / 7 :=
by sorry

end fraction_of_sum_l193_193851


namespace length_of_CD_l193_193649

theorem length_of_CD {L : ℝ} (h₁ : 16 * Real.pi * L + (256 / 3) * Real.pi = 432 * Real.pi) :
  L = (50 / 3) :=
by
  sorry

end length_of_CD_l193_193649


namespace factorization_of_x_squared_minus_one_l193_193238

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l193_193238


namespace min_value_reciprocal_sum_l193_193389

theorem min_value_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x + y + z = 3) :
  (1 / x) + (1 / y) + (1 / z) ≥ 3 :=
sorry

end min_value_reciprocal_sum_l193_193389


namespace factor_difference_of_squares_l193_193270

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l193_193270


namespace problem_I_solution_problem_II_solution_l193_193942

noncomputable def f (x : ℝ) : ℝ := |3 * x - 2| + |x - 2|

-- Problem (I): Solve the inequality f(x) <= 8
theorem problem_I_solution (x : ℝ) : 
  f x ≤ 8 ↔ -1 ≤ x ∧ x ≤ 3 :=
sorry

-- Problem (II): Find the range of the real number m
theorem problem_II_solution (x m : ℝ) : 
  f x ≥ (m^2 - m + 2) * |x| ↔ (0 ≤ m ∧ m ≤ 1) :=
sorry

end problem_I_solution_problem_II_solution_l193_193942


namespace not_line_D_l193_193772

-- Defining the points A and B
def A : ℝ × ℝ := (2, -3)
def B : ℝ × ℝ := (3, 1)

-- Defining the candidate equations as functions to check their slopes
def eq_A (x y : ℝ) : Prop := y + 3 = 4 * (x - 2)
def eq_B (x y : ℝ) : Prop := y - 1 = 4 * (x - 3)
def eq_C (x y : ℝ) : Prop := 4 * x - y - 11 = 0
def eq_D (x y : ℝ) : Prop := y + 3 = (x - 2) / 4

-- Defining the main theorem
theorem not_line_D : ¬ (eq_D A.1 A.2) ∧ ¬ (eq_D B.1 B.2) :=
by
  -- Proof steps go here, but we skip them for now
  sorry

end not_line_D_l193_193772


namespace sequence_constant_l193_193105

theorem sequence_constant
  (a : ℕ → ℤ)
  (d : ℤ)
  (h1 : ∀ n, Nat.Prime (Int.natAbs (a n)))
  (h2 : ∀ n, a (n + 2) = a (n + 1) + a n + d) :
  ∃ c : ℤ, ∀ n, a n = c :=
by
  sorry

end sequence_constant_l193_193105


namespace acute_angle_at_3_16_l193_193126

def angle_between_clock_hands (hour minute : ℕ) : ℝ :=
  let minute_angle := (minute / 60) * 360
  let hour_angle := (hour % 12) * 30 + (minute / 60) * 30
  |hour_angle - minute_angle|

theorem acute_angle_at_3_16 : angle_between_clock_hands 3 16 = 2 := 
sorry

end acute_angle_at_3_16_l193_193126


namespace max_distance_proof_l193_193865

-- Definitions for fuel consumption rates per 100 km
def fuel_consumption_U : Nat := 20 -- liters per 100 km
def fuel_consumption_V : Nat := 25 -- liters per 100 km
def fuel_consumption_W : Nat := 5  -- liters per 100 km
def fuel_consumption_X : Nat := 10 -- liters per 100 km

-- Definitions for total available fuel
def total_fuel : Nat := 50 -- liters

-- Distance calculation
def distance (fuel_consumption : Nat) (fuel : Nat) : Nat :=
  (fuel * 100) / fuel_consumption

-- Distances
def distance_U := distance fuel_consumption_U total_fuel
def distance_V := distance fuel_consumption_V total_fuel
def distance_W := distance fuel_consumption_W total_fuel
def distance_X := distance fuel_consumption_X total_fuel

-- Maximum total distance calculation
def maximum_total_distance : Nat :=
  distance_U + distance_V + distance_W + distance_X

-- The statement to be proved
theorem max_distance_proof :
  maximum_total_distance = 1950 := by
  sorry

end max_distance_proof_l193_193865


namespace p_computation_l193_193390

def p (x y : Int) : Int :=
  if x >= 0 ∧ y >= 0 then x + y
  else if x < 0 ∧ y < 0 then x - 3 * y
  else if x + y > 0 then 2 * x + 2 * y
  else x + 4 * y

theorem p_computation : p (p 2 (-3)) (p (-3) (-4)) = 26 := by
  sorry

end p_computation_l193_193390


namespace triangle_height_l193_193106

theorem triangle_height (x y : ℝ) :
  let area := (x^3 * y)^2
  let base := (2 * x * y)^2
  base ≠ 0 →
  (2 * area) / base = x^4 / 2 :=
by
  sorry

end triangle_height_l193_193106


namespace evaluate_expression_l193_193899

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem evaluate_expression : spadesuit 3 (spadesuit 6 5) = -112 := by
  sorry

end evaluate_expression_l193_193899


namespace rahul_matches_l193_193953

theorem rahul_matches
  (initial_avg : ℕ)
  (runs_today : ℕ)
  (final_avg : ℕ)
  (n : ℕ)
  (H1 : initial_avg = 50)
  (H2 : runs_today = 78)
  (H3 : final_avg = 54)
  (H4 : (initial_avg * n + runs_today) = final_avg * (n + 1)) :
  n = 6 :=
by
  sorry

end rahul_matches_l193_193953


namespace factorize_difference_of_squares_l193_193297

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l193_193297


namespace total_pens_bought_l193_193430

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_bought_l193_193430


namespace total_pens_bought_l193_193396

theorem total_pens_bought (r : ℕ) (hr : r > 10) (hm : 357 % r = 0) (ho : 441 % r = 0) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_bought_l193_193396


namespace average_GPA_school_l193_193636

theorem average_GPA_school (GPA6 GPA7 GPA8 : ℕ) (h1 : GPA6 = 93) (h2 : GPA7 = GPA6 + 2) (h3 : GPA8 = 91) : ((GPA6 + GPA7 + GPA8) / 3) = 93 :=
by
  sorry

end average_GPA_school_l193_193636


namespace limiting_reactant_and_products_l193_193537

def balanced_reaction 
  (al_moles : ℕ) (h2so4_moles : ℕ) 
  (al2_so4_3_moles : ℕ) (h2_moles : ℕ) : Prop :=
  2 * al_moles >= 0 ∧ 3 * h2so4_moles >= 0 ∧ 
  al_moles = 2 ∧ h2so4_moles = 3 ∧ 
  al2_so4_3_moles = 1 ∧ h2_moles = 3 ∧ 
  (2 : ℕ) * al_moles + (3 : ℕ) * h2so4_moles = 2 * 2 + 3 * 3

theorem limiting_reactant_and_products :
  balanced_reaction 2 3 1 3 :=
by {
  -- Here we would provide the proof based on the conditions and balances provided in the problem statement.
  sorry
}

end limiting_reactant_and_products_l193_193537


namespace bren_age_indeterminate_l193_193826

/-- The problem statement: The ratio of ages of Aman, Bren, and Charlie are in 
the ratio 5:8:7 respectively. A certain number of years ago, the sum of their ages was 76. 
We need to prove that without additional information, it is impossible to uniquely 
determine Bren's age 10 years from now. -/
theorem bren_age_indeterminate
  (x y : ℕ) 
  (h_ratio : true)
  (h_sum : 20 * x - 3 * y = 76) : 
  ∃ x y : ℕ, (20 * x - 3 * y = 76) ∧ ∀ bren_age_future : ℕ, ∃ x' y' : ℕ, (20 * x' - 3 * y' = 76) ∧ (8 * x' + 10) ≠ bren_age_future :=
sorry

end bren_age_indeterminate_l193_193826


namespace regina_has_20_cows_l193_193462

theorem regina_has_20_cows (C P : ℕ)
  (h1 : P = 4 * C)
  (h2 : 400 * P + 800 * C = 48000) :
  C = 20 :=
by
  sorry

end regina_has_20_cows_l193_193462


namespace sequence_first_term_eq_three_l193_193756

theorem sequence_first_term_eq_three
  (a : ℕ → ℕ)
  (h_rec : ∀ n : ℕ, a (n + 2) = a (n + 1) + a n)
  (h_nz : ∀ n : ℕ, 0 < a n)
  (h_a11 : a 11 = 157) :
  a 1 = 3 :=
sorry

end sequence_first_term_eq_three_l193_193756


namespace lollipop_problem_l193_193603

def arithmetic_sequence_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem lollipop_problem
  (a : ℕ) (h1 : arithmetic_sequence_sum a 5 7 = 175) :
  (a + 15) = 25 :=
by
  sorry

end lollipop_problem_l193_193603


namespace total_money_spent_l193_193691

def total_cost (blades_cost : Nat) (string_cost : Nat) : Nat :=
  blades_cost + string_cost

theorem total_money_spent 
  (num_blades : Nat)
  (cost_per_blade : Nat)
  (string_cost : Nat)
  (h1 : num_blades = 4)
  (h2 : cost_per_blade = 8)
  (h3 : string_cost = 7) :
  total_cost (num_blades * cost_per_blade) string_cost = 39 :=
by
  sorry

end total_money_spent_l193_193691


namespace max_salad_servings_l193_193988

theorem max_salad_servings :
  let cucumbers_per_serving := 2
  let tomatoes_per_serving := 2
  let bryndza_per_serving := 75 -- in grams
  let pepper_per_serving := 1
  let total_peppers := 60
  let total_bryndza := 4200 -- in grams
  let total_tomatoes := 116
  let total_cucumbers := 117
  let servings_peppers := total_peppers / pepper_per_serving
  let servings_bryndza := total_bryndza / bryndza_per_serving
  let servings_tomatoes := total_tomatoes / tomatoes_per_serving
  let servings_cucumbers := total_cucumbers / cucumbers_per_serving
  let max_servings := Int.min servings_peppers servings_bryndza
    (Int.min servings_tomatoes servings_cucumbers)
  max_servings = 56 :=
by
  sorry

end max_salad_servings_l193_193988


namespace negation_of_proposition_l193_193645

theorem negation_of_proposition (a b : ℝ) :
  ¬(a > b → 2 * a > 2 * b) ↔ (a ≤ b → 2 * a ≤ 2 * b) :=
by
  sorry

end negation_of_proposition_l193_193645


namespace total_votes_l193_193072

theorem total_votes (emma_votes : ℕ) (vote_fraction : ℚ) (h_emma : emma_votes = 45) (h_fraction : vote_fraction = 3/7) :
  emma_votes = vote_fraction * 105 :=
by {
  sorry
}

end total_votes_l193_193072


namespace factorize_difference_of_squares_l193_193295

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l193_193295


namespace stone_width_l193_193002

theorem stone_width (length_hall breadth_hall : ℝ) (num_stones length_stone : ℝ) (total_area_hall total_area_stones area_stone : ℝ)
  (h1 : length_hall = 36) (h2 : breadth_hall = 15) (h3 : num_stones = 5400) (h4 : length_stone = 2) 
  (h5 : total_area_hall = length_hall * breadth_hall * (10 * 10))
  (h6 : total_area_stones = num_stones * area_stone) 
  (h7 : area_stone = length_stone * (5 : ℝ)) 
  (h8 : total_area_stones = total_area_hall) : 
  (5 : ℝ) = 5 :=  
by sorry

end stone_width_l193_193002


namespace probability_of_7_successes_l193_193675

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_of_successes (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coefficient n k * p^k * (1 - p)^(n - k)

theorem probability_of_7_successes :
  probability_of_successes 7 7 (2/7) = 128 / 823543 :=
by
  sorry

end probability_of_7_successes_l193_193675


namespace factor_difference_of_squares_l193_193259

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l193_193259


namespace g_h_2_eq_583_l193_193919

def g (x : ℝ) : ℝ := 3*x^2 - 5

def h (x : ℝ) : ℝ := -2*x^3 + 2

theorem g_h_2_eq_583 : g (h 2) = 583 :=
by
  sorry

end g_h_2_eq_583_l193_193919


namespace solve_inequality_l193_193964

theorem solve_inequality (a x : ℝ) : 
  (ax^2 + (a - 1) * x - 1 < 0) ↔ (
  (a = 0 ∧ x > -1) ∨ 
  (a > 0 ∧ -1 < x ∧ x < 1/a) ∨
  (-1 < a ∧ a < 0 ∧ (x < 1/a ∨ x > -1)) ∨ 
  (a = -1 ∧ x ≠ -1) ∨ 
  (a < -1 ∧ (x < -1 ∨ x > 1/a))
) := sorry

end solve_inequality_l193_193964


namespace factorize_x_squared_minus_one_l193_193189

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l193_193189


namespace sum_coeff_expansion_l193_193979

theorem sum_coeff_expansion (x y : ℝ) : 
  (x + 2 * y)^4 = 81 := sorry

end sum_coeff_expansion_l193_193979


namespace average_speed_l193_193480

-- Defining conditions
def speed_first_hour : ℕ := 100  -- The car travels 100 km in the first hour
def speed_second_hour : ℕ := 60  -- The car travels 60 km in the second hour
def total_distance : ℕ := speed_first_hour + speed_second_hour  -- Total distance traveled

def total_time : ℕ := 2  -- Total time taken in hours

-- Stating the theorem
theorem average_speed : total_distance / total_time = 80 := 
by
  sorry

end average_speed_l193_193480


namespace avg_annual_growth_rate_is_20_percent_optimal_selling_price_for_max_discount_l193_193560

/-

Problem 1:
Given:
- Visitors in 2022: 200000
- Visitors in 2024: 288000

Prove:
- The average annual growth rate of visitors from 2022 to 2024 is 20% 

Problem 2:
Given:
- Cost price per cup: 6 yuan
- Selling price per cup at 25 yuan leads to 300 cups sold per day.
- Each 1 yuan reduction leads to 30 more cups sold per day.
- Desired daily profit in 2024: 6300 yuan

Prove:
- The selling price per cup for maximum discount and desired profit is 20 yuan.

-/

-- Definitions for Problem 1
def visitors_2022 : ℕ := 200000
def visitors_2024 : ℕ := 288000

-- Definition for annual growth rate
def annual_growth_rate (P Q : ℕ) (y : ℕ) : ℝ :=
  ((Q.to_real / P.to_real) ^ (1 / y.to_real)) - 1

def expected_growth_rate := annual_growth_rate visitors_2022 visitors_2024 2

-- Statement for the first proof
theorem avg_annual_growth_rate_is_20_percent : expected_growth_rate = 0.2 := sorry

-- Definitions for Problem 2
def cost_price_per_cup : ℕ := 6
def initial_price_per_cup : ℕ := 25
def initial_sales_per_day : ℕ := 300
def additional_sales_per_price_reduction : ℕ := 30
def desired_daily_profit : ℕ := 6300

-- Profit function
def daily_profit (price : ℕ) : ℕ := (price - cost_price_per_cup) * (initial_sales_per_day + additional_sales_per_price_reduction * (initial_price_per_cup - price))

-- Statement for the second proof
theorem optimal_selling_price_for_max_discount : (∃ (price : ℕ), daily_profit price = desired_daily_profit ∧ price = 20) := sorry

end avg_annual_growth_rate_is_20_percent_optimal_selling_price_for_max_discount_l193_193560


namespace find_n_l193_193506

theorem find_n (n : ℕ) (h_lcm : Nat.lcm n 16 = 48) (h_gcf : Nat.gcd n 16 = 4) : n = 12 :=
by
  sorry

end find_n_l193_193506


namespace averagePricePerBook_l193_193952

-- Define the prices and quantities from the first store
def firstStoreFictionBooks : ℕ := 25
def firstStoreFictionPrice : ℝ := 20
def firstStoreNonFictionBooks : ℕ := 15
def firstStoreNonFictionPrice : ℝ := 30
def firstStoreChildrenBooks : ℕ := 20
def firstStoreChildrenPrice : ℝ := 8

-- Define the prices and quantities from the second store
def secondStoreFictionBooks : ℕ := 10
def secondStoreFictionPrice : ℝ := 18
def secondStoreNonFictionBooks : ℕ := 20
def secondStoreNonFictionPrice : ℝ := 25
def secondStoreChildrenBooks : ℕ := 30
def secondStoreChildrenPrice : ℝ := 5

-- Definition of total books from first and second store
def totalBooks : ℕ :=
  firstStoreFictionBooks + firstStoreNonFictionBooks + firstStoreChildrenBooks +
  secondStoreFictionBooks + secondStoreNonFictionBooks + secondStoreChildrenBooks

-- Definition of the total cost from first and second store
def totalCost : ℝ :=
  (firstStoreFictionBooks * firstStoreFictionPrice) +
  (firstStoreNonFictionBooks * firstStoreNonFictionPrice) +
  (firstStoreChildrenBooks * firstStoreChildrenPrice) +
  (secondStoreFictionBooks * secondStoreFictionPrice) +
  (secondStoreNonFictionBooks * secondStoreNonFictionPrice) +
  (secondStoreChildrenBooks * secondStoreChildrenPrice)

-- Theorem: average price per book
theorem averagePricePerBook : (totalCost / totalBooks : ℝ) = 16.17 := by
  sorry

end averagePricePerBook_l193_193952


namespace melanie_bought_books_l193_193460

-- Defining the initial number of books and final number of books
def initial_books : ℕ := 41
def final_books : ℕ := 87

-- Theorem stating that Melanie bought 46 books at the yard sale
theorem melanie_bought_books : (final_books - initial_books) = 46 := by
  sorry

end melanie_bought_books_l193_193460


namespace total_pens_l193_193451

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193451


namespace find_pairs_gcd_lcm_l193_193326

theorem find_pairs_gcd_lcm : 
  { (a, b) : ℕ × ℕ | Nat.gcd a b = 24 ∧ Nat.lcm a b = 360 } = {(24, 360), (72, 120)} := 
by
  sorry

end find_pairs_gcd_lcm_l193_193326


namespace determinant_of_A_l193_193540

def A : Matrix (Fin 3) (Fin 3) ℤ := ![![3, 1, -2], ![8, 5, -4], ![3, 3, 7]]  -- Defining matrix A

def A' : Matrix (Fin 3) (Fin 3) ℤ := ![![3, 1, -2], ![5, 4, -2], ![0, 2, 9]]  -- Defining matrix A' after row operations

theorem determinant_of_A' : Matrix.det A' = 55 := by -- Proving that the determinant of A' is 55
  sorry

end determinant_of_A_l193_193540


namespace factorize_x_squared_minus_1_l193_193278

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l193_193278


namespace factorize_difference_of_squares_l193_193293

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l193_193293


namespace moles_H2O_formed_l193_193895

-- Define the balanced equation as a struct
structure Reaction :=
(reactants : List (String × ℕ)) -- List of reactants with their stoichiometric coefficients
(products : List (String × ℕ)) -- List of products with their stoichiometric coefficients

-- Example reaction: NaHCO3 + HC2H3O2 -> NaC2H3O2 + H2O + CO2
def example_reaction : Reaction :=
{ reactants := [("NaHCO3", 1), ("HC2H3O2", 1)],
  products := [("NaC2H3O2", 1), ("H2O", 1), ("CO2", 1)] }

-- We need a predicate to determine the number of moles of a product based on the reaction
def moles_of_product (reaction : Reaction) (product : String) (moles_reactant₁ moles_reactant₂ : ℕ) : ℕ :=
if product = "H2O" then moles_reactant₁ else 0  -- Only considering H2O for simplicity

-- Now we define our main theorem
theorem moles_H2O_formed : 
  moles_of_product example_reaction "H2O" 3 3 = 3 :=
by
  -- The proof will go here; for now, we use sorry to skip it
  sorry

end moles_H2O_formed_l193_193895


namespace B_share_is_2400_l193_193154

noncomputable def calculate_B_share (total_profit : ℝ) (x : ℝ) : ℝ :=
  let A_investment_months := 3 * x * 12
  let B_investment_months := x * 6
  let C_investment_months := (3/2) * x * 9
  let D_investment_months := (3/2) * x * 8
  let total_investment_months := A_investment_months + B_investment_months + C_investment_months + D_investment_months
  (B_investment_months / total_investment_months) * total_profit

theorem B_share_is_2400 :
  calculate_B_share 27000 1 = 2400 :=
sorry

end B_share_is_2400_l193_193154


namespace shaded_area_is_correct_l193_193544

noncomputable def total_shaded_area : ℝ :=
  let s := 10
  let R := s / (2 * Real.sin (Real.pi / 8))
  let A := (1 / 2) * R^2 * Real.sin (2 * Real.pi / 8)
  4 * A

theorem shaded_area_is_correct :
  total_shaded_area = 200 * Real.sqrt 2 / Real.sin (Real.pi / 8)^2 := 
sorry

end shaded_area_is_correct_l193_193544


namespace matchstick_equality_l193_193037

theorem matchstick_equality :
  abs ((22 : ℝ) / 7 - Real.pi) < 0.1 := 
sorry

end matchstick_equality_l193_193037


namespace average_GPA_school_l193_193634

theorem average_GPA_school (GPA6 GPA7 GPA8 : ℕ) (h1 : GPA6 = 93) (h2 : GPA7 = GPA6 + 2) (h3 : GPA8 = 91) : ((GPA6 + GPA7 + GPA8) / 3) = 93 :=
by
  sorry

end average_GPA_school_l193_193634


namespace total_pens_bought_l193_193434

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_bought_l193_193434


namespace total_pens_l193_193411

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193411


namespace exists_rectangle_with_properties_l193_193556

variables {e a φ : ℝ}

-- Define the given conditions
def diagonal_diff (e a : ℝ) := e - a
def angle_between_diagonals (φ : ℝ) := φ

-- The problem to prove
theorem exists_rectangle_with_properties (e a φ : ℝ) 
  (h_diff : diagonal_diff e a = e - a) 
  (h_angle : angle_between_diagonals φ = φ) : 
  ∃ (rectangle : Type) (A B C D : rectangle), 
    (e - a = e - a) ∧ 
    (φ = φ) := 
sorry

end exists_rectangle_with_properties_l193_193556


namespace LCM_180_504_l193_193491

theorem LCM_180_504 : Nat.lcm 180 504 = 2520 := 
by 
  -- We skip the proof.
  sorry

end LCM_180_504_l193_193491


namespace distinct_sequences_l193_193363

theorem distinct_sequences (N : ℕ) (α : ℝ) 
  (cond1 : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ N → 1 ≤ j ∧ j ≤ N → i ≠ j → 
    Int.floor (i * α) ≠ Int.floor (j * α)) 
  (cond2 : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ N → 1 ≤ j ∧ j ≤ N → i ≠ j → 
    Int.floor (i / α) ≠ Int.floor (j / α)) : 
  (↑(N - 1) / ↑N : ℝ) ≤ α ∧ α ≤ (↑N / ↑(N - 1) : ℝ) := 
sorry

end distinct_sequences_l193_193363


namespace ratio_of_larger_to_smaller_l193_193117

variable {x y : ℝ}

-- Condition for x and y being positive and x > y
axiom x_pos : 0 < x
axiom y_pos : 0 < y
axiom x_gt_y : x > y

-- Condition for sum and difference relationship
axiom sum_diff_relation : x + y = 7 * (x - y)

-- Theorem: Ratio of the larger number to the smaller number is 2
theorem ratio_of_larger_to_smaller : x / y = 2 :=
by
  sorry

end ratio_of_larger_to_smaller_l193_193117


namespace no_fixed_points_implies_no_double_fixed_points_l193_193080

theorem no_fixed_points_implies_no_double_fixed_points (f : ℝ → ℝ) (hf : ∀ x, f x ≠ x) :
  ∀ x, f (f x) ≠ x :=
sorry

end no_fixed_points_implies_no_double_fixed_points_l193_193080


namespace factorization_of_x_squared_minus_one_l193_193234

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l193_193234


namespace factorization_of_x_squared_minus_one_l193_193243

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l193_193243


namespace henley_initial_candies_l193_193362

variables (C : ℝ)
variables (h1 : 0.60 * C = 180)

theorem henley_initial_candies : C = 300 :=
by sorry

end henley_initial_candies_l193_193362


namespace smallest_of_three_integers_l193_193741

theorem smallest_of_three_integers (a b c : ℤ) (h1 : a * b * c = 32) (h2 : a + b + c = 3) : min (min a b) c = -4 := 
sorry

end smallest_of_three_integers_l193_193741


namespace factorize_difference_of_squares_l193_193287

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l193_193287


namespace total_pens_l193_193417

/-- Proof that Masha and Olya bought a total of 38 pens given the cost conditions. -/
theorem total_pens (r : ℕ) (h_r : r > 10) (h1 : 357 % r = 0) (h2 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193417


namespace black_white_tile_ratio_l193_193150

/-- Assume the original pattern has 12 black tiles and 25 white tiles.
    The pattern is extended by attaching a border of black tiles two tiles wide around the square.
    Prove that the ratio of black tiles to white tiles in the new extended pattern is 76/25.-/
theorem black_white_tile_ratio 
  (original_black_tiles : ℕ)
  (original_white_tiles : ℕ)
  (black_border_width : ℕ)
  (new_black_tiles : ℕ)
  (total_new_tiles : ℕ) 
  (total_old_tiles : ℕ) 
  (new_white_tiles : ℕ)
  : original_black_tiles = 12 → 
    original_white_tiles = 25 → 
    black_border_width = 2 → 
    total_old_tiles = 36 →
    total_new_tiles = 100 →
    new_black_tiles = 76 → 
    new_white_tiles = 25 → 
    (new_black_tiles : ℚ) / (new_white_tiles : ℚ) = 76 / 25 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end black_white_tile_ratio_l193_193150


namespace total_pens_bought_l193_193429

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_bought_l193_193429


namespace swiss_probability_is_30_percent_l193_193784

def total_cheese_sticks : Nat := 22 + 34 + 29 + 45 + 20

def swiss_cheese_sticks : Nat := 45

def probability_swiss : Nat :=
  (swiss_cheese_sticks * 100) / total_cheese_sticks

theorem swiss_probability_is_30_percent :
  probability_swiss = 30 := by
  sorry

end swiss_probability_is_30_percent_l193_193784


namespace factorize_difference_of_squares_l193_193292

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l193_193292


namespace number_of_appointments_l193_193711

-- Define the conditions
variables {hours_in_workday : ℕ} {appointments_duration : ℕ} {permit_rate : ℕ} {total_permits : ℕ}
variables (H1 : hours_in_workday = 8) (H2 : appointments_duration = 3) (H3 : permit_rate = 50) (H4: total_permits = 100)

-- Define the question as a theorem with the correct answer
theorem number_of_appointments : 
  (hours_in_workday - (total_permits / permit_rate)) / appointments_duration = 2 :=
by
  -- Proof is not required
  sorry

end number_of_appointments_l193_193711


namespace problem_statement_l193_193613

-- Defining the sets U, M, and N
def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

-- Complement of N in U
def complement_U_N : Set ℕ := U \ N

-- Problem statement
theorem problem_statement : M ∩ complement_U_N = {0, 3} :=
by
  sorry

end problem_statement_l193_193613


namespace area_of_given_triangle_l193_193378

noncomputable def area_of_triangle (a A B : ℝ) : ℝ :=
  let C := Real.pi - A - B
  let b := a * (Real.sin B / Real.sin A)
  let S := (1 / 2) * a * b * Real.sin C
  S

theorem area_of_given_triangle : area_of_triangle 4 (Real.pi / 4) (Real.pi / 3) = 6 + 2 * Real.sqrt 3 := 
by 
  sorry

end area_of_given_triangle_l193_193378


namespace same_type_l193_193657

variable (X Y : Prop) 

-- Definition of witnesses A and B based on their statements
def witness_A (A : Prop) := A ↔ (X → Y)
def witness_B (B : Prop) := B ↔ (¬X ∨ Y)

-- Proposition stating that A and B must be of the same type
theorem same_type (A B : Prop) (HA : witness_A X Y A) (HB : witness_B X Y B) : 
  (A = B) := 
sorry

end same_type_l193_193657


namespace math_proof_problem_l193_193500

-- Define the problem conditions
def problem_conditions (x y : ℚ) := 
  (real.sqrt (x - y) = 2 / 5) ∧ (real.sqrt (x + y) = 2)

-- Define the correct solution
def correct_solution (x y : ℚ) := 
  (x = 52 / 25) ∧ (y = 48 / 25)

-- Define the area of the rectangle
def rectangle_area (a b : ℚ) : ℚ :=
  abs (a * b)

-- Define the proof problem
theorem math_proof_problem : 
  problem_conditions (52 / 25) (48 / 25) ∧ 
  rectangle_area (52 / 25) (48 / 25) = 8 / 25 :=
by 
  sorry

end math_proof_problem_l193_193500


namespace reciprocal_key_problem_l193_193085

theorem reciprocal_key_problem :
  let f : ℝ → ℝ := λ x, 1 / x
  in (f^[2]) 50 = 50 ∧ (∀ n : ℕ, 0 < n → (n < 2 → (f^[n]) 50 ≠ 50)) :=
begin
  sorry,
end

end reciprocal_key_problem_l193_193085


namespace compare_M_N_l193_193606

variable (a : ℝ)

def M : ℝ := 2 * a^2 - 4 * a
def N : ℝ := a^2 - 2 * a - 3

theorem compare_M_N : M a > N a := by
  sorry

end compare_M_N_l193_193606


namespace yellow_mugs_count_l193_193917

variables (R B Y O : ℕ)
variables (B_eq_3R : B = 3 * R)
variables (R_eq_Y_div_2 : R = Y / 2)
variables (O_eq_4 : O = 4)
variables (mugs_eq_40 : R + B + Y + O = 40)

theorem yellow_mugs_count : Y = 12 :=
by 
  sorry

end yellow_mugs_count_l193_193917


namespace factorize_x_squared_minus_one_l193_193311

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l193_193311


namespace range_of_a_l193_193368

theorem range_of_a (a : ℝ) : (∀ x : ℕ, 4 * x + a ≤ 5 → x ≥ 1 → x ≤ 3) ↔ (-11 < a ∧ a ≤ -7) :=
by sorry

end range_of_a_l193_193368


namespace opposite_of_neg_2023_l193_193976

theorem opposite_of_neg_2023 :
  ∃ y : ℝ, (-2023 + y = 0) ∧ y = 2023 :=
by
  sorry

end opposite_of_neg_2023_l193_193976


namespace resulting_polygon_has_30_sides_l193_193870

def polygon_sides : ℕ := 3 + 4 + 5 + 6 + 7 + 8 + 9 - 6 * 2

theorem resulting_polygon_has_30_sides : polygon_sides = 30 := by
  sorry

end resulting_polygon_has_30_sides_l193_193870


namespace contradiction_method_example_l193_193951

variables {a b c : ℝ}
variables (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
variables (h4 : a + b + c > 0) (h5 : ab + bc + ca > 0)
variables (h6 : (a < 0 ∧ b < 0) ∨ (a < 0 ∧ c < 0) ∨ (b < 0 ∧ c < 0))

theorem contradiction_method_example : false :=
by {
  sorry
}

end contradiction_method_example_l193_193951


namespace prob_blue_section_damaged_all_days_l193_193671

noncomputable def prob_of_7_successes (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem prob_blue_section_damaged_all_days :
  prob_of_7_successes 7 7 (2 / 7) = 128 / 823543 :=
by sorry

end prob_blue_section_damaged_all_days_l193_193671


namespace eval_product_eq_1093_l193_193023

noncomputable def z : ℂ := Complex.exp (2 * Real.pi * Complex.I / 7)

theorem eval_product_eq_1093 : (3 - z) * (3 - z^2) * (3 - z^3) * (3 - z^4) * (3 - z^5) * (3 - z^6) = 1093 := by
  sorry

end eval_product_eq_1093_l193_193023


namespace find_interest_rate_l193_193014

noncomputable def annual_interest_rate (P A : ℝ) (n : ℕ) (t r : ℝ) : Prop :=
  A = P * (1 + r / n)^(n * t)

theorem find_interest_rate :
  annual_interest_rate 5000 5100.50 4 0.5 0.04 :=
by
  sorry

end find_interest_rate_l193_193014


namespace abs_lt_one_sufficient_not_necessary_l193_193570

theorem abs_lt_one_sufficient_not_necessary (x : ℝ) : (|x| < 1) -> (x < 1) ∧ ¬(x < 1 -> |x| < 1) :=
by
  sorry

end abs_lt_one_sufficient_not_necessary_l193_193570


namespace max_principals_ten_years_l193_193878

theorem max_principals_ten_years : 
  (∀ (P : ℕ → Prop), (∀ n, n ≥ 10 → ∀ i, ¬P (n - i)) → ∀ p, p ≤ 4 → 
  (∃ n ≤ 10, ∀ k, k ≥ n → P k)) :=
sorry

end max_principals_ten_years_l193_193878


namespace cos_A_value_l193_193591

-- Conditions
variables (a b c : ℝ) (A B C : ℝ)
-- a, b, c are the sides opposite to angles A, B, and C respectively.
-- Assumption 1: b - c = (1/4) * a
def condition1 := b - c = (1/4) * a
-- Assumption 2: 2 * sin B = 3 * sin C
def condition2 := 2 * Real.sin B = 3 * Real.sin C

-- The theorem statement: Under these conditions, prove that cos A = -1/4.
theorem cos_A_value (h1 : condition1 a b c) (h2 : condition2 B C) : 
    Real.cos A = -1/4 :=
sorry -- placeholder for the proof

end cos_A_value_l193_193591


namespace bernoulli_trial_probability_7_successes_l193_193666

theorem bernoulli_trial_probability_7_successes :
  let n := 7
  let k := 7
  let p := (2 : ℝ) / 7
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k)) = (128 / 823543) :=
by
  sorry

end bernoulli_trial_probability_7_successes_l193_193666


namespace book_distribution_l193_193128

theorem book_distribution (x : ℤ) (h : 9 * x + 7 < 11 * x) : 
  ∀ (b : ℤ), (b = 9 * x + 7) → b mod 9 = 7 :=
by
  intro b
  intro hb
  have : b = 9 * x + 7, from hb
  rw [←this]
  sorry

end book_distribution_l193_193128


namespace number_of_streams_is_three_l193_193374

-- Define the lakes and conditions
structure LakeValley where
  lakes : Finset String
  connected_by_streams : String → String → Bool
  prob_stay_in_S_after_4_moves : ℚ
  prob_stay_in_B_after_4_moves : ℚ

-- Given conditions
def valley_conditions : LakeValley :=
  {
    lakes := {"S", "A", "B", "C", "D"},
    connected_by_streams := λ a b, (a, b) ∈ {("S", "A"), ("A", "B"), ("S", "C"), ("C", "B")},
    prob_stay_in_S_after_4_moves := 375 / 1000,
    prob_stay_in_B_after_4_moves := 625 / 1000
  }

-- Statement of the proof problem
theorem number_of_streams_is_three (v : LakeValley)
  (h1 : v = valley_conditions) :
  (Finset.card v.lakes) - 1 = 3 :=
by
  sorry

end number_of_streams_is_three_l193_193374


namespace determinant_of_matrix_l193_193541

def mat : Matrix (Fin 3) (Fin 3) ℤ := 
  ![![3, 0, 2],![8, 5, -2],![3, 3, 6]]

theorem determinant_of_matrix : Matrix.det mat = 90 := 
by 
  sorry

end determinant_of_matrix_l193_193541


namespace max_value_OP_OQ_l193_193596

def circle_1_polar_eq (rho theta : ℝ) : Prop :=
  rho = 4 * Real.cos theta

def circle_2_polar_eq (rho theta : ℝ) : Prop :=
  rho = 2 * Real.sin theta

theorem max_value_OP_OQ (alpha : ℝ) :
  (∃ rho1 rho2 : ℝ, circle_1_polar_eq rho1 alpha ∧ circle_2_polar_eq rho2 alpha) ∧
  (∃ max_OP_OQ : ℝ, max_OP_OQ = 4) :=
sorry

end max_value_OP_OQ_l193_193596


namespace seven_digit_palindromes_count_l193_193168

theorem seven_digit_palindromes_count : 
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  (a_choices * b_choices * c_choices * d_choices) = 9000 := by
  sorry

end seven_digit_palindromes_count_l193_193168


namespace factor_difference_of_squares_l193_193263

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l193_193263


namespace train_speed_km_hr_l193_193860

def train_length : ℝ := 130  -- Length of the train in meters
def bridge_and_train_length : ℝ := 245  -- Total length of the bridge and the train in meters
def crossing_time : ℝ := 30  -- Time to cross the bridge in seconds

theorem train_speed_km_hr : (train_length + bridge_and_train_length) / crossing_time * 3.6 = 45 := by
  sorry

end train_speed_km_hr_l193_193860


namespace factor_difference_of_squares_l193_193261

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l193_193261


namespace factorize_difference_of_squares_l193_193230

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l193_193230


namespace ticket_price_difference_l193_193618

noncomputable def price_difference (adult_price total_cost : ℕ) (num_adults num_children : ℕ) (child_price : ℕ) : ℕ :=
  adult_price - child_price

theorem ticket_price_difference :
  ∀ (adult_price total_cost num_adults num_children child_price : ℕ),
  adult_price = 19 →
  total_cost = 77 →
  num_adults = 2 →
  num_children = 3 →
  num_adults * adult_price + num_children * child_price = total_cost →
  price_difference adult_price total_cost num_adults num_children child_price = 6 :=
by
  intros
  simp [price_difference]
  sorry

end ticket_price_difference_l193_193618


namespace solution_set_f_over_x_lt_0_l193_193641

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_over_x_lt_0 :
  (∀ x, f (2 - x) = f (2 + x)) →
  (∀ x1 x2, x1 < 2 ∧ x2 < 2 ∧ x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) < 0) →
  (f 4 = 0) →
  { x | f x / x < 0 } = { x | x < 0 } ∪ { x | 0 < x ∧ x < 4 } :=
by
  intros _ _ _
  sorry

end solution_set_f_over_x_lt_0_l193_193641


namespace circle_equation_is_correct_l193_193977

def center : Int × Int := (-3, 4)
def radius : Int := 3
def circle_standard_equation (x y : Int) : Int :=
  (x + 3)^2 + (y - 4)^2

theorem circle_equation_is_correct :
  circle_standard_equation x y = 9 :=
sorry

end circle_equation_is_correct_l193_193977


namespace exists_ints_for_inequalities_l193_193461

theorem exists_ints_for_inequalities (a b : ℝ) (ε : ℝ) (hε : ε > 0) :
  ∃ (n : ℕ) (k m : ℤ), |(n * a) - k| < ε ∧ |(n * b) - m| < ε :=
by
  sorry

end exists_ints_for_inequalities_l193_193461


namespace wrapping_paper_area_l193_193852

theorem wrapping_paper_area 
  (l w h : ℝ) :
  (l + 4 + 2 * h) ^ 2 = l^2 + 8 * l + 16 + 4 * l * h + 16 * h + 4 * h^2 := 
by 
  sorry

end wrapping_paper_area_l193_193852


namespace intersection_M_N_l193_193582

-- Definitions:
def M := {x : ℝ | 0 ≤ x}
def N := {y : ℝ | -2 ≤ y}

-- The theorem statement:
theorem intersection_M_N : M ∩ N = {z : ℝ | 0 ≤ z} := sorry

end intersection_M_N_l193_193582


namespace alice_stops_in_quarter_D_l193_193620

-- Definitions and conditions
def indoor_track_circumference : ℕ := 40
def starting_point_S : ℕ := 0
def run_distance : ℕ := 1600

-- Desired theorem statement
theorem alice_stops_in_quarter_D :
  (run_distance % indoor_track_circumference = 0) → 
  (0 ≤ (run_distance % indoor_track_circumference) ∧ 
   (run_distance % indoor_track_circumference) < indoor_track_circumference) → 
  true := by
  sorry

end alice_stops_in_quarter_D_l193_193620


namespace students_watching_l193_193137

theorem students_watching (b g : ℕ) (h : b + g = 33) : (2 / 3 : ℚ) * b + (2 / 3 : ℚ) * g = 22 := by
  sorry

end students_watching_l193_193137


namespace school_avg_GPA_l193_193629

theorem school_avg_GPA (gpa_6th : ℕ) (gpa_7th : ℕ) (gpa_8th : ℕ) 
  (h6 : gpa_6th = 93) 
  (h7 : gpa_7th = 95) 
  (h8 : gpa_8th = 91) : 
  (gpa_6th + gpa_7th + gpa_8th) / 3 = 93 :=
by 
  sorry

end school_avg_GPA_l193_193629


namespace price_difference_VA_NC_l193_193592

/-- Define the initial conditions -/
def NC_price : ℝ := 2
def NC_gallons : ℕ := 10
def VA_gallons : ℕ := 10
def total_spent : ℝ := 50

/-- Define the problem to prove the difference in price per gallon between Virginia and North Carolina -/
theorem price_difference_VA_NC (NC_price VA_price total_spent : ℝ) (NC_gallons VA_gallons : ℕ) :
  total_spent = NC_price * NC_gallons + VA_price * VA_gallons →
  VA_price - NC_price = 1 := 
by
  sorry -- Proof to be filled in

end price_difference_VA_NC_l193_193592


namespace water_bill_august_32m_cubed_water_usage_october_59_8_yuan_l193_193485

noncomputable def tiered_water_bill (usage : ℕ) : ℝ :=
  if usage <= 20 then
    2.3 * usage
  else if usage <= 30 then
    2.3 * 20 + 3.45 * (usage - 20)
  else
    2.3 * 20 + 3.45 * 10 + 4.6 * (usage - 30)

-- (1) Prove that if Xiao Ming's family used 32 cubic meters of water in August, 
-- their water bill is 89.7 yuan.
theorem water_bill_august_32m_cubed : tiered_water_bill 32 = 89.7 := by
  sorry

-- (2) Prove that if Xiao Ming's family paid 59.8 yuan for their water bill in October, 
-- they used 24 cubic meters of water.
theorem water_usage_october_59_8_yuan : ∃ x : ℕ, tiered_water_bill x = 59.8 ∧ x = 24 := by
  use 24
  sorry

end water_bill_august_32m_cubed_water_usage_october_59_8_yuan_l193_193485


namespace factorize_difference_of_squares_l193_193221

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l193_193221


namespace rate_calculation_l193_193004

noncomputable def rate_per_sq_meter
  (lawn_length : ℝ) (lawn_breadth : ℝ)
  (road_width : ℝ) (total_cost : ℝ) : ℝ :=
  let area_road_1 := road_width * lawn_breadth
  let area_road_2 := road_width * lawn_length
  let area_intersection := road_width * road_width
  let total_area_roads := (area_road_1 + area_road_2) - area_intersection
  total_cost / total_area_roads

theorem rate_calculation :
  rate_per_sq_meter 100 60 10 4500 = 3 := by
  sorry

end rate_calculation_l193_193004


namespace compute_y_geometric_series_l193_193727

theorem compute_y_geometric_series :
  let s1 := ∑' n : ℕ, (1 / 3) ^ n,
      s2 := ∑' n : ℕ, (-1) ^ n * (1 / 3) ^ n in
  s1 = 3 / 2 →
  s2 = 3 / 4 →
  (1 + s1) * (1 + s2) = 1 +
  ∑' n : ℕ, (1 / 9) ^ n :=
by
  sorry

end compute_y_geometric_series_l193_193727


namespace total_pens_l193_193438

theorem total_pens (r : ℕ) (r_gt_10 : r > 10) (r_div_357 : r ∣ 357) (r_div_441 : r ∣ 441) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_l193_193438


namespace range_of_a_l193_193799

open Set

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem range_of_a (a : ℝ) (h : A a ∩ B = ∅) :
  a ∈ Iic (-1 / 2) ∪ Ici 2 :=
by
  sorry

end range_of_a_l193_193799


namespace factorization_of_x_squared_minus_one_l193_193235

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l193_193235


namespace factorization_of_difference_of_squares_l193_193257

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l193_193257


namespace polynomial_divisibility_l193_193026

theorem polynomial_divisibility (a b : ℤ) :
  (∀ x : ℤ, x^2 - 1 ∣ x^5 - 3 * x^4 + a * x^3 + b * x^2 - 5 * x - 5) ↔ (a = 4 ∧ b = 8) :=
sorry

end polynomial_divisibility_l193_193026


namespace no_such_fraction_exists_l193_193550

theorem no_such_fraction_exists :
  ∀ (x y : ℕ), (Nat.coprime x y) → (0 < x) → (0 < y) → ¬ (1.2 * (x:ℚ) / (y:ℚ) = (x+1:ℚ) / (y+1:ℚ)) := by
  sorry

end no_such_fraction_exists_l193_193550


namespace num_lineups_l193_193091

-- Define the given conditions
def num_players : ℕ := 12
def num_lineman : ℕ := 4
def num_qb_among_lineman : ℕ := 2
def num_running_backs : ℕ := 3

-- State the problem and the result as a theorem
theorem num_lineups : 
  (num_lineman * (num_qb_among_lineman) * (num_running_backs) * (num_players - num_lineman - num_qb_among_lineman - num_running_backs + 3) = 216) := 
by
  -- The proof will go here
  sorry

end num_lineups_l193_193091


namespace factorize_difference_of_squares_l193_193299

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l193_193299


namespace gcd_102_238_l193_193111

theorem gcd_102_238 : Int.gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l193_193111


namespace solve_system_eqns_l193_193802

theorem solve_system_eqns (x y : ℚ) 
    (h1 : (x - 30) / 3 = (2 * y + 7) / 4)
    (h2 : x - y = 10) :
  x = -81 / 2 ∧ y = -101 / 2 := 
sorry

end solve_system_eqns_l193_193802


namespace div_mult_result_l193_193833

theorem div_mult_result : 150 / (30 / 3) * 2 = 30 :=
by sorry

end div_mult_result_l193_193833


namespace mountain_height_correct_l193_193753

noncomputable def height_of_mountain : ℝ :=
  15 / (1 / Real.tan (Real.pi * 10 / 180) + 1 / Real.tan (Real.pi * 12 / 180))

theorem mountain_height_correct :
  abs (height_of_mountain - 1.445) < 0.001 :=
sorry

end mountain_height_correct_l193_193753


namespace factorize_difference_of_squares_l193_193225

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l193_193225


namespace speed_conversion_l193_193858

noncomputable def mps_to_kmph (speed_mps : ℝ) : ℝ :=
  speed_mps * 3.6

theorem speed_conversion (h : 1 = 3.6) : mps_to_kmph 12.7788 = 45.96 :=
  by
    sorry

end speed_conversion_l193_193858


namespace total_pens_l193_193423

theorem total_pens (r : ℕ) (h1 : r > 10)
  (h2 : 357 % r = 0)
  (h3 : 441 % r = 0) :
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_l193_193423


namespace smallest_b_factors_x2_bx_2016_l193_193337

theorem smallest_b_factors_x2_bx_2016 :
  ∃ (b : ℕ), (∀ (r s : ℤ), r * s = 2016 → r + s = b → b = 92) :=
begin
  sorry
end

end smallest_b_factors_x2_bx_2016_l193_193337


namespace total_pens_l193_193450

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193450


namespace range_of_a_l193_193608

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x < a + 2 → x ≤ 2) ↔ a ≤ 0 := by
  sorry

end range_of_a_l193_193608


namespace seventh_term_of_geometric_sequence_l193_193542

theorem seventh_term_of_geometric_sequence :
  ∀ (a r : ℝ), (a * r ^ 3 = 16) → (a * r ^ 8 = 2) → (a * r ^ 6 = 2) :=
by
  intros a r h1 h2
  sorry

end seventh_term_of_geometric_sequence_l193_193542


namespace trapezium_second_side_length_l193_193883

theorem trapezium_second_side_length
  (side1 : ℝ)
  (height : ℝ)
  (area : ℝ) 
  (h1 : side1 = 20) 
  (h2 : height = 13) 
  (h3 : area = 247) : 
  ∃ side2 : ℝ, 0 ≤ side2 ∧ ∀ side2, area = 1 / 2 * (side1 + side2) * height → side2 = 18 :=
by
  use 18
  sorry

end trapezium_second_side_length_l193_193883


namespace find_b_l193_193757

-- Define complex numbers z1 and z2
def z1 (b : ℝ) : Complex := Complex.mk 3 (-b)

def z2 : Complex := Complex.mk 1 (-2)

-- Statement that needs to be proved
theorem find_b (b : ℝ) (h : (z1 b / z2).re = 0) : b = -3 / 2 :=
by
  -- proof goes here
  sorry

end find_b_l193_193757


namespace telephone_charge_l193_193680

theorem telephone_charge (x : ℝ) (h1 : ∀ t : ℝ, t = 18.70 → x + 39 * 0.40 = t) : x = 3.10 :=
by
  sorry

end telephone_charge_l193_193680


namespace evaluate_expression_eq_neg_one_evaluate_expression_only_value_l193_193880

variable (a y : ℝ)
variable (h1 : a ≠ 0)
variable (h2 : a ≠ 2 * y)
variable (h3 : a ≠ -2 * y)

theorem evaluate_expression_eq_neg_one
  (h : y = -a / 3) :
  ( (a / (a + 2 * y) + y / (a - 2 * y)) / (y / (a + 2 * y) - a / (a - 2 * y)) ) = -1 := 
sorry

theorem evaluate_expression_only_value :
  ( (a / (a + 2 * y) + y / (a - 2 * y)) / (y / (a + 2 * y) - a / (a - 2 * y)) = -1 ) ↔ 
  y = -a / 3 := 
sorry

end evaluate_expression_eq_neg_one_evaluate_expression_only_value_l193_193880


namespace total_pens_bought_l193_193435

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_bought_l193_193435


namespace compute_fraction_square_l193_193538

theorem compute_fraction_square : 6 * (3 / 7) ^ 2 = 54 / 49 :=
by 
  sorry

end compute_fraction_square_l193_193538


namespace log_equation_solution_l193_193736

theorem log_equation_solution :
  ∃ x : ℝ, 0 < x ∧ x = log 3 (64 + x) ∧ abs(x - 4) < 1 :=
sorry

end log_equation_solution_l193_193736


namespace problem_statement_l193_193872

noncomputable def f_B (x : ℝ) : ℝ := -x^2
noncomputable def f_D (x : ℝ) : ℝ := Real.cos x

theorem problem_statement :
  (∀ x : ℝ, f_B (-x) = f_B x) ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < 1 → f_B x1 > f_B x2) ∧
  (∀ x : ℝ, f_D (-x) = f_D x) ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < 1 → f_D x1 > f_D x2) :=
  sorry

end problem_statement_l193_193872


namespace boys_left_hand_to_girl_l193_193142

-- Definitions based on the given conditions
def num_boys : ℕ := 40
def num_girls : ℕ := 28
def boys_right_hand_to_girl : ℕ := 18

-- Statement to prove
theorem boys_left_hand_to_girl : (num_boys - (num_boys - boys_right_hand_to_girl)) = boys_right_hand_to_girl := by
  sorry

end boys_left_hand_to_girl_l193_193142


namespace overall_cost_for_all_projects_l193_193601

-- Define the daily salaries including 10% taxes and insurance.
def daily_salary_entry_level_worker : ℕ := 100 + 10
def daily_salary_experienced_worker : ℕ := 130 + 13
def daily_salary_electrician : ℕ := 2 * 100 + 20
def daily_salary_plumber : ℕ := 250 + 25
def daily_salary_architect : ℕ := (35/10) * 100 + 35

-- Define the total cost for each project.
def project1_cost : ℕ :=
  daily_salary_entry_level_worker +
  daily_salary_experienced_worker +
  daily_salary_electrician +
  daily_salary_plumber +
  daily_salary_architect

def project2_cost : ℕ :=
  2 * daily_salary_experienced_worker +
  daily_salary_electrician +
  daily_salary_plumber +
  daily_salary_architect

def project3_cost : ℕ :=
  2 * daily_salary_entry_level_worker +
  daily_salary_electrician +
  daily_salary_plumber +
  daily_salary_architect

-- Define the overall cost for all three projects.
def total_cost : ℕ :=
  project1_cost + project2_cost + project3_cost

theorem overall_cost_for_all_projects :
  total_cost = 3399 :=
by
  sorry

end overall_cost_for_all_projects_l193_193601


namespace Heracles_age_l193_193716

variable (A H : ℕ)

theorem Heracles_age :
  (A = H + 7) →
  (A + 3 = 2 * H) →
  H = 10 :=
by
  sorry

end Heracles_age_l193_193716


namespace avg_annual_growth_rate_optimal_selling_price_l193_193559

theorem avg_annual_growth_rate (v2022 v2024 : ℕ) (x : ℝ) 
  (h1 : v2022 = 200000) 
  (h2 : v2024 = 288000)
  (h3: v2024 = v2022 * (1 + x)^2) :
  x = 0.2 :=
by
  sorry

theorem optimal_selling_price (cost : ℝ) (initial_price : ℝ) (initial_cups : ℕ) 
  (price_drop_effect : ℝ) (initial_profit : ℝ) (daily_profit : ℕ) (y : ℝ)
  (h1 : cost = 6)
  (h2 : initial_price = 25) 
  (h3 : initial_cups = 300)
  (h4 : price_drop_effect = 1)
  (h5 : initial_profit = 6300)
  (h6 : (y - cost) * (initial_cups + 30 * (initial_price - y)) = daily_profit) :
  y = 20 :=
by
  sorry

end avg_annual_growth_rate_optimal_selling_price_l193_193559


namespace factor_difference_of_squares_l193_193271

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l193_193271


namespace binomial_coefficient_30_3_l193_193539

theorem binomial_coefficient_30_3 :
  Nat.choose 30 3 = 4060 := 
by 
  sorry

end binomial_coefficient_30_3_l193_193539


namespace number_of_numbers_tadd_said_after_20_rounds_l193_193487

-- Define the arithmetic sequence representing the count of numbers Tadd says each round
def tadd_sequence (n : ℕ) : ℕ :=
  1 + 2 * (n - 1)

-- Define the sum of the first n terms of Tadd's sequence
def sum_tadd_sequence (n : ℕ) : ℕ :=
  n * (1 + tadd_sequence n) / 2

-- The main theorem to state the problem
theorem number_of_numbers_tadd_said_after_20_rounds :
  sum_tadd_sequence 20 = 400 :=
by
  -- The actual proof should be filled in here
  sorry

end number_of_numbers_tadd_said_after_20_rounds_l193_193487


namespace factorization_difference_of_squares_l193_193216

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l193_193216


namespace factorize_x_squared_minus_one_l193_193323

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l193_193323


namespace fraction_value_l193_193836

theorem fraction_value :
  (0.02 ^ 2 + 0.52 ^ 2 + 0.035 ^ 2) / (0.002 ^ 2 + 0.052 ^ 2 + 0.0035 ^ 2) = 100 := by
    sorry

end fraction_value_l193_193836


namespace total_pens_l193_193410

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193410


namespace smallest_next_divisor_l193_193617

theorem smallest_next_divisor (n : ℕ) (h_even : n % 2 = 0) (h_4_digit : 1000 ≤ n ∧ n < 10000) (h_div_493 : 493 ∣ n) :
  ∃ d : ℕ, (d > 493 ∧ d ∣ n) ∧ ∀ e, (e > 493 ∧ e ∣ n) → d ≤ e ∧ d = 510 := by
  sorry

end smallest_next_divisor_l193_193617


namespace parallelepiped_inequality_l193_193941

theorem parallelepiped_inequality (a b c d : ℝ) (h : d^2 = a^2 + b^2 + c^2 + 2 * (a * b + a * c + b * c)) :
  a^2 + b^2 + c^2 ≥ (1 / 3) * d^2 :=
by
  sorry

end parallelepiped_inequality_l193_193941


namespace no_positive_integers_exist_l193_193165

theorem no_positive_integers_exist 
  (a b c d : ℕ) 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) 
  (d_pos : 0 < d)
  (h₁ : a * b = c * d)
  (p : ℕ) 
  (hp : Nat.Prime p)
  (h₂ : a + b + c + d = p) : 
  False := 
by
  sorry

end no_positive_integers_exist_l193_193165


namespace simplify_expression1_simplify_expression2_l193_193721

section
variables (a b : ℝ)

theorem simplify_expression1 : -b*(2*a - b) + (a + b)^2 = a^2 + 2*b^2 :=
sorry
end

section
variables (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2)

theorem simplify_expression2 : (1 - (x/(2 + x))) / ((x^2 - 4)/(x^2 + 4*x + 4)) = 2/(x - 2) :=
sorry
end

end simplify_expression1_simplify_expression2_l193_193721


namespace proof_of_calculation_l193_193524

theorem proof_of_calculation : (7^2 - 5^2)^4 = 331776 := by
  sorry

end proof_of_calculation_l193_193524


namespace log_expression_value_l193_193868

theorem log_expression_value : (Real.log 8 / Real.log 10) + 3 * (Real.log 5 / Real.log 10) = 3 :=
by
  -- Assuming necessary properties and steps are already known and prove the theorem accordingly:
  sorry

end log_expression_value_l193_193868


namespace iron_column_lifted_by_9_6_cm_l193_193864

namespace VolumeLift

def base_area_container : ℝ := 200
def base_area_column : ℝ := 40
def height_water : ℝ := 16
def distance_water_surface : ℝ := 4

theorem iron_column_lifted_by_9_6_cm :
  ∃ (h_lift : ℝ),
    h_lift = 9.6 ∧ height_water - distance_water_surface = 16 - h_lift :=
by
sorry

end VolumeLift

end iron_column_lifted_by_9_6_cm_l193_193864


namespace total_pens_l193_193421

theorem total_pens (r : ℕ) (h1 : r > 10)
  (h2 : 357 % r = 0)
  (h3 : 441 % r = 0) :
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_l193_193421


namespace total_pens_l193_193412

/-- Proof that Masha and Olya bought a total of 38 pens given the cost conditions. -/
theorem total_pens (r : ℕ) (h_r : r > 10) (h1 : 357 % r = 0) (h2 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193412


namespace factorize_difference_of_squares_l193_193291

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l193_193291


namespace radius_of_tangent_circle_l193_193681

theorem radius_of_tangent_circle 
    (side_length : ℝ) 
    (tangent_angle : ℝ) 
    (sin_15 : ℝ)
    (circle_radius : ℝ) :
    side_length = 2 * Real.sqrt 3 →
    tangent_angle = 30 →
    sin_15 = (Real.sqrt 3 - 1) / (2 * Real.sqrt 2) →
    circle_radius = 2 :=
by sorry

end radius_of_tangent_circle_l193_193681


namespace obtuse_triangle_two_acute_angles_l193_193059

-- Define the angle type (could be Real between 0 and 180 in degrees).
def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < 90

def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- Define an obtuse triangle using three angles α, β, γ
structure obtuse_triangle :=
(angle1 angle2 angle3 : ℝ)
(sum_angles_eq : angle1 + angle2 + angle3 = 180)
(obtuse_condition : is_obtuse angle1 ∨ is_obtuse angle2 ∨ is_obtuse angle3)

-- The theorem to prove the number of acute angles in an obtuse triangle is 2.
theorem obtuse_triangle_two_acute_angles (T : obtuse_triangle) : 
  (is_acute T.angle1 ∧ is_acute T.angle2 ∧ ¬ is_acute T.angle3) ∨ 
  (is_acute T.angle1 ∧ ¬ is_acute T.angle2 ∧ is_acute T.angle3) ∨ 
  (¬ is_acute T.angle1 ∧ is_acute T.angle2 ∧ is_acute T.angle3) :=
by sorry

end obtuse_triangle_two_acute_angles_l193_193059


namespace right_angled_triangles_count_l193_193647

theorem right_angled_triangles_count :
    ∃ n : ℕ, n = 31 ∧ ∀ (a b : ℕ), (b < 2011) ∧ (a * a = (b + 1) * (b + 1) - b * b) → n = 31 :=
by
  sorry

end right_angled_triangles_count_l193_193647


namespace trebled_resultant_l193_193003

theorem trebled_resultant (n : ℕ) (h : n = 20) : 3 * ((2 * n) + 5) = 135 := 
by
  sorry

end trebled_resultant_l193_193003


namespace carla_smoothies_serving_l193_193722

theorem carla_smoothies_serving :
  ∀ (watermelon_puree : ℕ) (cream : ℕ) (serving_size : ℕ),
  watermelon_puree = 500 → cream = 100 → serving_size = 150 →
  (watermelon_puree + cream) / serving_size = 4 :=
by
  intros watermelon_puree cream serving_size
  intro h1 -- watermelon_puree = 500
  intro h2 -- cream = 100
  intro h3 -- serving_size = 150
  sorry

end carla_smoothies_serving_l193_193722


namespace palmer_total_photos_l193_193094

theorem palmer_total_photos (initial_photos : ℕ) (first_week_photos : ℕ) (third_fourth_weeks_photos : ℕ) :
  (initial_photos = 100) →
  (first_week_photos = 50) →
  (third_fourth_weeks_photos = 80) →
  let second_week_photos := 2 * first_week_photos in
  let total_bali_photos := first_week_photos + second_week_photos + third_fourth_weeks_photos in
  let total_photos := initial_photos + total_bali_photos in
  total_photos = 330 :=
by
  intros h_initial h_first_week h_third_fourth_weeks
  let second_week_photos := 2 * first_week_photos
  let total_bali_photos := first_week_photos + second_week_photos + third_fourth_weeks_photos
  let total_photos := initial_photos + total_bali_photos
  sorry

end palmer_total_photos_l193_193094


namespace factorize_x_squared_minus_one_l193_193315

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l193_193315


namespace total_apples_eaten_l193_193465

theorem total_apples_eaten : (1 / 2) * 16 + (1 / 3) * 15 + (1 / 4) * 20 = 18 := by
  sorry

end total_apples_eaten_l193_193465


namespace a_plus_d_eq_zero_l193_193078

noncomputable def f (a b c d x : ℝ) : ℝ := (2 * a * x + b) / (c * x + 2 * d)

theorem a_plus_d_eq_zero (a b c d : ℝ) (h : a * b * c * d ≠ 0) (hff : ∀ x, f a b c d (f a b c d x) = 3 * x - 4) : a + d = 0 :=
by
  sorry

end a_plus_d_eq_zero_l193_193078


namespace negative_subtraction_result_l193_193869

theorem negative_subtraction_result : -2 - 1 = -3 := 
by
  -- The proof is not required by the prompt, so we use "sorry" to indicate the unfinished proof.
  sorry

end negative_subtraction_result_l193_193869


namespace count_valid_words_l193_193543

def total_words (n : ℕ) : ℕ := 25 ^ n

def words_with_no_A (n : ℕ) : ℕ := 24 ^ n

def words_with_one_A (n : ℕ) : ℕ := n * 24 ^ (n - 1)

def words_with_less_than_two_As : ℕ :=
  (words_with_no_A 2) + (2 * 24) +
  (words_with_no_A 3) + (3 * 24 ^ 2) +
  (words_with_no_A 4) + (4 * 24 ^ 3) +
  (words_with_no_A 5) + (5 * 24 ^ 4)

def valid_words : ℕ :=
  (total_words 1 + total_words 2 + total_words 3 + total_words 4 + total_words 5) -
  words_with_less_than_two_As

theorem count_valid_words : valid_words = sorry :=
by sorry

end count_valid_words_l193_193543


namespace total_area_for_building_l193_193982

theorem total_area_for_building (num_sections : ℕ) (area_per_section : ℝ) (open_space_percentage : ℝ) :
  num_sections = 7 →
  area_per_section = 9473 →
  open_space_percentage = 0.15 →
  (num_sections * (area_per_section * (1 - open_space_percentage))) = 56364.35 :=
by
  intros h1 h2 h3
  sorry

end total_area_for_building_l193_193982


namespace calculate_f_one_l193_193052

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem calculate_f_one : f 1 = 2 := by
  sorry

end calculate_f_one_l193_193052


namespace find_third_discount_percentage_l193_193686

noncomputable def third_discount_percentage (x : ℝ) : Prop :=
  let item_price := 68
  let num_items := 3
  let first_discount := 0.15
  let second_discount := 0.10
  let total_initial_price := num_items * item_price
  let price_after_first_discount := total_initial_price * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  price_after_second_discount * (1 - x / 100) = 105.32

theorem find_third_discount_percentage : ∃ x : ℝ, third_discount_percentage x ∧ x = 32.5 :=
by
  sorry

end find_third_discount_percentage_l193_193686


namespace shipping_cost_correct_l193_193685

noncomputable def shipping_cost (W : ℝ) : ℕ := 7 + 5 * (⌈W⌉₊ - 1)

theorem shipping_cost_correct (W : ℝ) : shipping_cost W = 5 * ⌈W⌉₊ + 2 :=
by
  sorry

end shipping_cost_correct_l193_193685


namespace average_track_width_l193_193516

theorem average_track_width (r1 r2 s1 s2 : ℝ) 
  (h1 : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 20 * Real.pi)
  (h2 : 2 * Real.pi * s1 - 2 * Real.pi * s2 = 30 * Real.pi) :
  (r1 - r2 + (s1 - s2)) / 2 = 12.5 := 
sorry

end average_track_width_l193_193516


namespace estimate_students_l193_193930

noncomputable def numberOfStudentsAbove120 (X : ℝ → ℝ) (μ : ℝ) (σ : ℝ) (P₁ : ℝ → ℝ → ℝ)
  (students : ℝ) : ℝ :=
  let prob_interval := P₁ 100 110
  let prob_above_120 := (1 - (2 * prob_interval)) / 2
  prob_above_120 * students

theorem estimate_students (μ : ℝ := 110) (σ : ℝ := 10) (P₁ : ℝ → ℝ → ℝ)
  (students : ℝ := 50) (hyp : P₁ 100 110 = 0.34) :
  numberOfStudentsAbove120 (λ x => Normal.pdf μ σ x) μ σ P₁ students = 8 :=
by sorry

end estimate_students_l193_193930


namespace factorize_x_squared_minus_1_l193_193279

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l193_193279


namespace special_hash_value_l193_193392

def special_hash (a b c d : ℝ) : ℝ :=
  d * b ^ 2 - 4 * a * c

theorem special_hash_value :
  special_hash 2 3 1 (1 / 2) = -3.5 :=
by
  -- Note: Insert proof here
  sorry

end special_hash_value_l193_193392


namespace present_age_of_B_l193_193505

theorem present_age_of_B 
  (a b : ℕ)
  (h1 : a + 10 = 2 * (b - 10))
  (h2 : a = b + 9) :
  b = 39 :=
by
  sorry

end present_age_of_B_l193_193505


namespace factorize_x_squared_minus_one_l193_193317

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l193_193317


namespace total_pens_bought_l193_193452

theorem total_pens_bought (r : ℕ) (r_gt_10 : r > 10) (r_divides_357 : 357 % r = 0) (r_divides_441 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
by sorry

end total_pens_bought_l193_193452


namespace harmonic_mean_of_4_and_5040_is_8_closest_l193_193972

noncomputable def harmonicMean (a b : ℕ) : ℝ :=
  (2 * a * b) / (a + b)

theorem harmonic_mean_of_4_and_5040_is_8_closest :
  abs (harmonicMean 4 5040 - 8) < 1 :=
by
  -- The proof process would go here
  sorry

end harmonic_mean_of_4_and_5040_is_8_closest_l193_193972


namespace ExpandedOHaraTripleValue_l193_193121

/-- Define an Expanded O'Hara triple -/
def isExpandedOHaraTriple (a b x : ℕ) : Prop :=
  2 * (Nat.sqrt a + Nat.sqrt b) = x

/-- Prove that for given a=64 and b=49, x is equal to 30 if (a, b, x) is an Expanded O'Hara triple -/
theorem ExpandedOHaraTripleValue (a b x : ℕ) (ha : a = 64) (hb : b = 49) (h : isExpandedOHaraTriple a b x) : x = 30 :=
by
  sorry

end ExpandedOHaraTripleValue_l193_193121


namespace number_of_integers_satisfying_condition_l193_193332

def satisfies_condition (n : ℤ) : Prop :=
  1 + Int.floor (101 * n / 102) = Int.ceil (98 * n / 99)

noncomputable def number_of_solutions : ℤ :=
  10198

theorem number_of_integers_satisfying_condition :
  (∃ n : ℤ, satisfies_condition n) ↔ number_of_solutions = 10198 :=
sorry

end number_of_integers_satisfying_condition_l193_193332


namespace find_length_of_second_train_l193_193504

def length_of_second_train (L : ℝ) : Prop :=
  let speed_first_train := 33.33 -- Speed in m/s
  let speed_second_train := 22.22 -- Speed in m/s
  let relative_speed := speed_first_train + speed_second_train -- Relative speed in m/s
  let time_to_cross := 9 -- time in seconds
  let length_first_train := 260 -- Length in meters
  length_first_train + L = relative_speed * time_to_cross

theorem find_length_of_second_train : length_of_second_train 239.95 :=
by
  admit -- To be completed (proof)

end find_length_of_second_train_l193_193504


namespace probability_statements_l193_193856

-- Assigning probabilities
def p_hit := 0.9
def p_miss := 1 - p_hit

-- Definitions based on the problem conditions
def shoot_4_times (shots : List Bool) : Bool :=
  shots.length = 4 ∧ ∀ (s : Bool), s ∈ shots → (s = true → s ≠ false) ∧ (s = false → s ≠ true ∧ s ≠ 0)

-- Statements derived from the conditions
def prob_shot_3 := p_hit

def prob_exact_3_out_of_4 := 
  let binom_4_3 := 4
  binom_4_3 * (p_hit^3) * (p_miss^1)

def prob_at_least_1_out_of_4 := 1 - (p_miss^4)

-- The equivalence proof
theorem probability_statements : 
  (prob_shot_3 = 0.9) ∧ 
  (prob_exact_3_out_of_4 = 0.2916) ∧ 
  (prob_at_least_1_out_of_4 = 0.9999) := 
by 
  sorry

end probability_statements_l193_193856


namespace min_sin_cos_sixth_power_l193_193893

noncomputable def min_value_sin_cos_expr : ℝ :=
  (3 + 2 * Real.sqrt 2) / 12

theorem min_sin_cos_sixth_power :
  ∃ x : ℝ, (∀ y, (Real.sin y) ^ 6 + 2 * (Real.cos y) ^ 6 ≥ min_value_sin_cos_expr) ∧ 
            ((Real.sin x) ^ 6 + 2 * (Real.cos x) ^ 6 = min_value_sin_cos_expr) :=
sorry

end min_sin_cos_sixth_power_l193_193893


namespace equal_papers_per_cousin_l193_193129

-- Given conditions
def haley_origami_papers : Float := 48.0
def cousins_count : Float := 6.0

-- Question and expected answer
def papers_per_cousin (total_papers : Float) (cousins : Float) : Float :=
  total_papers / cousins

-- Proof statement asserting the correct answer
theorem equal_papers_per_cousin :
  papers_per_cousin haley_origami_papers cousins_count = 8.0 :=
sorry

end equal_papers_per_cousin_l193_193129


namespace find_b_if_lines_parallel_l193_193737

-- Definitions of the line equations and parallel condition
def first_line (x y : ℝ) (b : ℝ) : Prop := 3 * y - b = -9 * x + 1
def second_line (x y : ℝ) (b : ℝ) : Prop := 2 * y + 8 = (b - 3) * x - 2

-- Definition of parallel lines (their slopes are equal)
def parallel_lines (m1 m2 : ℝ) : Prop := m1 = m2

-- Given conditions and the conclusion to prove
theorem find_b_if_lines_parallel :
  ∃ b : ℝ, (∀ x y : ℝ, first_line x y b → ∃ m1 : ℝ, ∀ x y : ℝ, ∃ c : ℝ, y = m1 * x + c) ∧ 
           (∀ x y : ℝ, second_line x y b → ∃ m2 : ℝ, ∀ x y : ℝ, ∃ c : ℝ, y = m2 * x + c) ∧ 
           parallel_lines (-3) ((b - 3) / 2) →
           b = -3 :=
by {
  sorry
}

end find_b_if_lines_parallel_l193_193737


namespace card_probability_l193_193984

/-- Three cards are drawn at random from a standard deck of 52 cards.
What is the probability that the first card is an Ace, the second card is a Diamond, 
and the third card is a King?
-/
theorem card_probability :
  let p := (3 / 52) * (12 / 51) * (4 / 50) +
            (3 / 52) * (1 / 51) * (3 / 50) +
            (1 / 52) * (11 / 51) * (4 / 50) +
            (1 / 52) * (1 / 51) * (3 / 50)
  in p = 1 / 663 :=
begin
  sorry
end

end card_probability_l193_193984


namespace min_bulbs_l193_193820

theorem min_bulbs (p : ℝ) (k : ℕ) (target_prob : ℝ) (h_p : p = 0.95) (h_k : k = 5) (h_target_prob : target_prob = 0.99) :
  ∃ n : ℕ, (finset.Ico k.succ n.succ).sum (λ i, (finset.range i).sum (λ j, (nat.choose j i) * p^i * (1 - p)^(j-i))) ≥ target_prob ∧ ∀ m : ℕ, m < n → 
  (finset.Ico k.succ m.succ).sum (λ i, (finset.range i).sum (λ j, (nat.choose j i) * p^i * (1 - p)^(j-i))) < target_prob :=
begin
  sorry
end

end min_bulbs_l193_193820


namespace intersection_points_in_decagon_l193_193862

-- Define the number of sides for a regular decagon
def n : ℕ := 10

-- The formula to calculate the number of ways to choose 4 vertices from n vertices
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The statement that needs to be proven
theorem intersection_points_in_decagon : choose 10 4 = 210 := by
  sorry

end intersection_points_in_decagon_l193_193862


namespace exponent_calculation_l193_193018

theorem exponent_calculation : (-1 : ℤ) ^ 53 + (2 : ℤ) ^ (5 ^ 3 - 2 ^ 3 + 3 ^ 2) = 2 ^ 126 - 1 :=
by 
  sorry

end exponent_calculation_l193_193018


namespace total_pens_bought_l193_193454

theorem total_pens_bought (r : ℕ) (r_gt_10 : r > 10) (r_divides_357 : 357 % r = 0) (r_divides_441 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
by sorry

end total_pens_bought_l193_193454


namespace present_population_l193_193769

theorem present_population (P : ℝ) (h1 : (P : ℝ) * (1 + 0.1) ^ 2 = 14520) : P = 12000 :=
sorry

end present_population_l193_193769


namespace find_m_l193_193020

def f (x m : ℝ) : ℝ := x ^ 2 - 3 * x + m
def g (x m : ℝ) : ℝ := 2 * x ^ 2 - 6 * x + 5 * m

theorem find_m (m : ℝ) (h : 3 * f 3 m = 2 * g 3 m) : m = 0 :=
by sorry

end find_m_l193_193020


namespace factorization_of_x_squared_minus_one_l193_193236

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l193_193236


namespace factorization_of_difference_of_squares_l193_193247

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l193_193247


namespace value_of_x_l193_193345

variable (x y : ℕ)

-- Conditions
axiom cond1 : x / y = 15 / 3
axiom cond2 : y = 27

-- Lean statement for the problem
theorem value_of_x : x = 135 :=
by
  have h1 := cond1
  have h2 := cond2
  sorry

end value_of_x_l193_193345


namespace factorization_example_l193_193531

theorem factorization_example (a b : ℕ) : (a - 2*b)^2 = a^2 - 4*a*b + 4*b^2 := 
by sorry

end factorization_example_l193_193531


namespace max_x_lcm_max_x_lcm_value_l193_193811

theorem max_x_lcm (x : ℕ) (h1 : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 :=
  sorry

theorem max_x_lcm_value (x : ℕ) (h1 : Nat.lcm (Nat.lcm x 15) 21 = 105) : x = 105 :=
  sorry

end max_x_lcm_max_x_lcm_value_l193_193811


namespace floor_diff_bounds_l193_193099

theorem floor_diff_bounds (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  0 ≤ Int.floor (a + b) - (Int.floor a + Int.floor b) ∧ 
  Int.floor (a + b) - (Int.floor a + Int.floor b) ≤ 1 :=
by
  sorry

end floor_diff_bounds_l193_193099


namespace ratio_of_S_to_R_l193_193707

noncomputable def find_ratio (total_amount : ℕ) (diff_SP : ℕ) (n : ℕ) (k : ℕ) (P : ℕ) (Q : ℕ) (R : ℕ) (S : ℕ) (ratio_SR : ℕ) :=
  Q = n ∧ R = n ∧ P = k * n ∧ S = ratio_SR * n ∧ P + Q + R + S = total_amount ∧ S - P = diff_SP

theorem ratio_of_S_to_R :
  ∃ n k ratio_SR, k = 2 ∧ ratio_SR = 4 ∧ 
  find_ratio 1000 250 n k 250 125 125 500 ratio_SR :=
by
  sorry

end ratio_of_S_to_R_l193_193707


namespace boat_trip_duration_l193_193656

noncomputable def boat_trip_time (B P : ℝ) : Prop :=
  (P = 4 * B) ∧ (B + P = 10)

theorem boat_trip_duration (B P : ℝ) (h : boat_trip_time B P) : B = 2 :=
by
  cases h with
  | intro hP hTotal =>
    sorry

end boat_trip_duration_l193_193656


namespace brownies_per_person_l193_193035

-- Define the conditions as constants
def columns : ℕ := 6
def rows : ℕ := 3
def people : ℕ := 6

-- Define the total number of brownies
def total_brownies : ℕ := columns * rows

-- Define the theorem to be proved
theorem brownies_per_person : total_brownies / people = 3 :=
by sorry

end brownies_per_person_l193_193035


namespace average_cups_of_tea_sold_l193_193143

theorem average_cups_of_tea_sold (x_avg : ℝ) (y_regression : ℝ → ℝ) 
  (h1 : x_avg = 12) (h2 : ∀ x, y_regression x = -2*x + 58) : 
  y_regression x_avg = 34 := by
  sorry

end average_cups_of_tea_sold_l193_193143


namespace initial_amount_in_cookie_jar_l193_193935

theorem initial_amount_in_cookie_jar (M : ℝ) (h : 15 / 100 * (85 / 100 * (100 - 10) / 100 * (100 - 15) / 100 * M) = 15) : M = 24.51 :=
sorry

end initial_amount_in_cookie_jar_l193_193935


namespace simplify_expression_l193_193469

theorem simplify_expression (y : ℝ) : y - 3 * (2 + y) + 4 * (2 - y) - 5 * (2 + 3 * y) = -21 * y - 8 :=
sorry

end simplify_expression_l193_193469


namespace total_photos_l193_193093

def initial_photos : ℕ := 100
def photos_first_week : ℕ := 50
def photos_second_week : ℕ := 2 * photos_first_week
def photos_third_fourth_week : ℕ := 80
def photos_from_bali : ℕ := photos_first_week + photos_second_week + photos_third_fourth_week

theorem total_photos (initial_photos photos_from_bali : ℕ) : initial_photos + photos_from_bali = 330 :=
by
  have h1 : initial_photos = 100 := rfl
  have h2 : photos_from_bali = 50 + (2 * 50) + 80 := rfl
  show 100 + (50 + 100 + 80) = 330
  sorry

end total_photos_l193_193093


namespace find_x_value_l193_193829

theorem find_x_value (x : ℝ) (h : (7 / (x - 2) + x / (2 - x) = 4)) : x = 3 :=
sorry

end find_x_value_l193_193829


namespace max_cos_x_l193_193797

theorem max_cos_x (x y : ℝ) (h : Real.cos (x - y) = Real.cos x - Real.cos y) : 
  ∃ M, (∀ x, Real.cos x <= M) ∧ M = 1 := 
sorry

end max_cos_x_l193_193797


namespace find_a_l193_193579

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.log x + (a - 1) * x

theorem find_a {a : ℝ} : 
  (∀ x : ℝ, 0 < x → f x a ≤ x^2 * Real.exp x - Real.log x - 4 * x - 1) → 
  a ≤ -2 :=
sorry

end find_a_l193_193579


namespace mrs_hilt_has_more_money_l193_193089

/-- Mrs. Hilt has two pennies, two dimes, and two nickels. 
    Jacob has four pennies, one nickel, and one dime. 
    Prove that Mrs. Hilt has $0.13 more than Jacob. -/
theorem mrs_hilt_has_more_money 
  (hilt_pennies hilt_dimes hilt_nickels : ℕ)
  (jacob_pennies jacob_dimes jacob_nickels : ℕ)
  (value_penny value_nickel value_dime : ℝ)
  (H1 : hilt_pennies = 2) (H2 : hilt_dimes = 2) (H3 : hilt_nickels = 2)
  (H4 : jacob_pennies = 4) (H5 : jacob_dimes = 1) (H6 : jacob_nickels = 1)
  (H7 : value_penny = 0.01) (H8 : value_nickel = 0.05) (H9 : value_dime = 0.10) :
  ((hilt_pennies * value_penny + hilt_dimes * value_dime + hilt_nickels * value_nickel) 
   - (jacob_pennies * value_penny + jacob_dimes * value_dime + jacob_nickels * value_nickel) 
   = 0.13) :=
by sorry

end mrs_hilt_has_more_money_l193_193089


namespace fraction_of_rectangle_shaded_l193_193144

theorem fraction_of_rectangle_shaded
  (length : ℕ) (width : ℕ)
  (one_third_part : ℕ) (half_of_third : ℕ)
  (H1 : length = 10) (H2 : width = 15)
  (H3 : one_third_part = (1/3 : ℝ) * (length * width)) 
  (H4 : half_of_third = (1/2 : ℝ) * one_third_part) :
  (half_of_third / (length * width) = 1/6) :=
sorry

end fraction_of_rectangle_shaded_l193_193144


namespace smallest_positive_b_l193_193476

theorem smallest_positive_b (b N : ℕ) (h1 : N = 7 * b^2 + 7 * b + 7) (h2 : ∃ x : ℕ, N = x^4) : b = 18 :=
  sorry

end smallest_positive_b_l193_193476


namespace prob_blue_section_damaged_all_days_l193_193670

noncomputable def prob_of_7_successes (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem prob_blue_section_damaged_all_days :
  prob_of_7_successes 7 7 (2 / 7) = 128 / 823543 :=
by sorry

end prob_blue_section_damaged_all_days_l193_193670


namespace translation_of_point_l193_193478

variable (P : ℝ × ℝ) (xT yT : ℝ)

def translate_x (P : ℝ × ℝ) (xT : ℝ) : ℝ × ℝ :=
    (P.1 + xT, P.2)

def translate_y (P : ℝ × ℝ) (yT : ℝ) : ℝ × ℝ :=
    (P.1, P.2 + yT)

theorem translation_of_point : translate_y (translate_x (-5, 1) 2) (-4) = (-3, -3) :=
by
  sorry

end translation_of_point_l193_193478


namespace percentage_return_on_investment_l193_193684

theorem percentage_return_on_investment
  (dividend_rate : ℝ)
  (face_value : ℝ)
  (purchase_price : ℝ)
  (dividend_per_share : ℝ := (dividend_rate / 100) * face_value)
  (percentage_return : ℝ := (dividend_per_share / purchase_price) * 100)
  (h1 : dividend_rate = 15.5)
  (h2 : face_value = 50)
  (h3 : purchase_price = 31) :
  percentage_return = 25 := by
    sorry

end percentage_return_on_investment_l193_193684


namespace luncheon_cost_l193_193970

theorem luncheon_cost (s c p : ℝ) (h1 : 5 * s + 9 * c + 2 * p = 5.95)
  (h2 : 7 * s + 12 * c + 2 * p = 7.90) (h3 : 3 * s + 5 * c + p = 3.50) :
  s + c + p = 1.05 :=
sorry

end luncheon_cost_l193_193970


namespace find_y_when_x_is_minus_2_l193_193804

theorem find_y_when_x_is_minus_2 :
  ∀ (x y t : ℝ), (x = 3 - 2 * t) → (y = 5 * t + 6) → (x = -2) → y = 37 / 2 :=
by
  intros x y t h1 h2 h3
  sorry

end find_y_when_x_is_minus_2_l193_193804


namespace robot_cost_l193_193719

theorem robot_cost (num_friends : ℕ) (total_tax change start_money : ℝ) (h_friends : num_friends = 7) (h_tax : total_tax = 7.22) (h_change : change = 11.53) (h_start : start_money = 80) :
  let spent_money := start_money - change
  let cost_robots := spent_money - total_tax
  let cost_per_robot := cost_robots / num_friends
  cost_per_robot = 8.75 :=
by
  sorry

end robot_cost_l193_193719


namespace relationship_y1_y2_y3_l193_193353

variable (k x y1 y2 y3 : ℝ)
variable (h1 : k < 0)
variable (h2 : y1 = k / -4)
variable (h3 : y2 = k / 2)
variable (h4 : y3 = k / 3)

theorem relationship_y1_y2_y3 (k x y1 y2 y3 : ℝ) 
  (h1 : k < 0)
  (h2 : y1 = k / -4)
  (h3 : y2 = k / 2)
  (h4 : y3 = k / 3) : 
  y1 > y3 ∧ y3 > y2 := 
by sorry

end relationship_y1_y2_y3_l193_193353


namespace smallest_lattice_triangle_area_l193_193367

open Real

def is_lattice_point (p : ℝ × ℝ) : Prop :=
  ∃ (m n : ℤ), p = (m, n)

def is_lattice_triangle (A B C : ℝ × ℝ) : Prop :=
  is_lattice_point A ∧ is_lattice_point B ∧ is_lattice_point C

def no_interior_points (A B C : ℝ × ℝ) : Prop :=
  ∀ (x y : ℝ) (h : 0 < x < 1 ∧ 0 < y < 1), 
    ¬(is_lattice_point (A.1 + x * (B.1 - A.1) + y * (C.1 - A.1), A.2 + x * (B.2 - A.2) + y * (C.2 - A.2)))

def three_boundary_points (A B C : ℝ × ℝ) : Prop :=
  (A ≠ B ∧ B ≠ C ∧ A ≠ C)

def area_of_lattice_triangle_min (A B C : ℝ × ℝ) (unit_area : ℝ) : ℝ :=
  1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1))

theorem smallest_lattice_triangle_area (A B C : ℝ × ℝ) (unit_area : ℝ):
  is_lattice_triangle A B C →
  no_interior_points A B C →
  three_boundary_points A B C →
  unit_area = 1 →
  area_of_lattice_triangle_min A B C unit_area = 1 / 2 := 
by 
  sorry

end smallest_lattice_triangle_area_l193_193367


namespace students_6_to_8_hours_study_l193_193855

-- Condition: 100 students were surveyed
def total_students : ℕ := 100

-- Hypothetical function representing the number of students studying for a specific range of hours based on the histogram
def histogram_students (lower_bound upper_bound : ℕ) : ℕ :=
  sorry  -- this would be defined based on actual histogram data

-- Question: Prove the number of students who studied for 6 to 8 hours
theorem students_6_to_8_hours_study : histogram_students 6 8 = 30 :=
  sorry -- the expected answer based on the histogram data

end students_6_to_8_hours_study_l193_193855


namespace min_xy_value_min_x_plus_y_value_l193_193908

variable {x y : ℝ}

theorem min_xy_value (hx : 0 < x) (hy : 0 < y) (h : (2 / y) + (8 / x) = 1) : xy ≥ 64 := 
sorry

theorem min_x_plus_y_value (hx : 0 < x) (hy : 0 < y) (h : (2 / y) + (8 / x) = 1) : x + y ≥ 18 :=
sorry

end min_xy_value_min_x_plus_y_value_l193_193908


namespace total_pens_l193_193405

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193405


namespace special_four_digit_numbers_l193_193001

noncomputable def count_special_four_digit_numbers : Nat :=
  -- The task is to define the number of four-digit numbers formed using the digits {0, 1, 2, 3, 4}
  -- that contain the digit 0 and have exactly two digits repeating
  144

theorem special_four_digit_numbers : count_special_four_digit_numbers = 144 := by
  sorry

end special_four_digit_numbers_l193_193001


namespace average_first_50_even_numbers_l193_193534

-- Condition: The sequence starts from 2.
-- Condition: The sequence consists of the first 50 even numbers.
def first50EvenNumbers : List ℤ := List.range' 2 100

theorem average_first_50_even_numbers : (first50EvenNumbers.sum / 50 = 51) :=
by
  sorry

end average_first_50_even_numbers_l193_193534


namespace expr1_eval_expr2_eval_l193_193024

theorem expr1_eval : (3 * Real.sqrt 27 - 2 * Real.sqrt 12) * (2 * Real.sqrt (16 / 3) + 3 * Real.sqrt (25 / 3)) = 115 := 
by
  -- Sorry serves as a placeholder for the proof.
  sorry

theorem expr2_eval : (5 * Real.sqrt 21 - 3 * Real.sqrt 15) / (5 * Real.sqrt (8 / 3) - 3 * Real.sqrt (5 / 3)) = 3 := 
by
  -- Sorry serves as a placeholder for the proof.
  sorry

end expr1_eval_expr2_eval_l193_193024


namespace cost_per_steak_knife_l193_193179

theorem cost_per_steak_knife
  (sets : ℕ) (knives_per_set : ℕ) (cost_per_set : ℕ)
  (h1 : sets = 2) (h2 : knives_per_set = 4) (h3 : cost_per_set = 80) :
  (cost_per_set * sets) / (sets * knives_per_set) = 20 := by
  sorry

end cost_per_steak_knife_l193_193179


namespace negation_proposition_l193_193812

variables {a b c : ℝ}

theorem negation_proposition (h : a ≤ b) : a + c ≤ b + c :=
sorry

end negation_proposition_l193_193812


namespace C_completion_time_l193_193371

noncomputable def racer_time (v_C : ℝ) : ℝ := 100 / v_C

theorem C_completion_time
  (v_A v_B v_C : ℝ)
  (h1 : 100 / v_A = 10)
  (h2 : 85 / v_B = 10)
  (h3 : 90 / v_C = 100 / v_B) :
  racer_time v_C = 13.07 :=
by
  sorry

end C_completion_time_l193_193371


namespace determine_constants_l193_193866

theorem determine_constants (a b c d : ℝ) 
  (periodic : (2 * (2 * Real.pi / b) = 4 * Real.pi))
  (vert_shift : d = 3)
  (max_val : (d + a = 8))
  (min_val : (d - a = -2)) :
  a = 5 ∧ b = 1 :=
by
  sorry

end determine_constants_l193_193866


namespace max_a4a7_value_l193_193046

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := ∀ n m : ℕ, a (n + 1) = a n + d

-- Given conditions
def given_conditions (a : ℕ → ℝ) (d : ℝ) : Prop := 
  arithmetic_sequence a d ∧ a 5 = 4 -- a6 = 4 so we use index 5 since Lean is 0-indexed

-- Define the product a4 * a7
def a4a7_product (a : ℕ → ℝ) (d : ℝ) : ℝ := (a 5 - 2 * d) * (a 5 + d)

-- The maximum value of a4 * a7
def max_a4a7 (a : ℕ → ℝ) (d : ℝ) : ℝ := 18

-- The proof problem statement
theorem max_a4a7_value (a : ℕ → ℝ) (d : ℝ) :
  given_conditions a d → a4a7_product a d = max_a4a7 a d :=
by
  sorry

end max_a4a7_value_l193_193046


namespace zookeeper_fish_total_l193_193650

def fish_given : ℕ := 19
def fish_needed : ℕ := 17

theorem zookeeper_fish_total : fish_given + fish_needed = 36 :=
by
  sorry

end zookeeper_fish_total_l193_193650


namespace factorization_difference_of_squares_l193_193213

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l193_193213


namespace proof_of_inequality_l193_193764

theorem proof_of_inequality (a b : ℝ) (h : a > b) : a - 3 > b - 3 :=
sorry

end proof_of_inequality_l193_193764


namespace jack_second_half_time_l193_193382

variable (jacksFirstHalf : ℕ) (jillTotalTime : ℕ) (timeDifference : ℕ)

def jacksTotalTime : ℕ := jillTotalTime - timeDifference

def jacksSecondHalf (jacksFirstHalf jacksTotalTime : ℕ) : ℕ :=
  jacksTotalTime - jacksFirstHalf

theorem jack_second_half_time : 
  jacksFirstHalf = 19 ∧ jillTotalTime = 32 ∧ timeDifference = 7 → jacksSecondHalf jacksFirstHalf (jacksTotalTime jillTotalTime timeDifference) = 6 :=
by
  intros h
  cases h with h1 h'
  cases h' with h2 h3
  rw [h1, h2, h3]
  unfold jacksTotalTime
  unfold jacksSecondHalf
  norm_num


end jack_second_half_time_l193_193382


namespace beggars_society_votes_l193_193845

def total_voting_members (votes_for votes_against additional_against : ℕ) :=
  let majority := additional_against / 4
  let initial_difference := votes_for - votes_against
  let updated_against := votes_against + additional_against
  let updated_for := votes_for - additional_against
  updated_for + updated_against

theorem beggars_society_votes :
  total_voting_members 115 92 12 = 207 :=
by
  -- Proof goes here
  sorry

end beggars_society_votes_l193_193845


namespace jack_second_half_time_l193_193384

variable (time_half1 time_half2 time_jack_total time_jill_total : ℕ)

theorem jack_second_half_time (h1 : time_half1 = 19) 
                              (h2 : time_jill_total = 32) 
                              (h3 : time_jack_total + 7 = time_jill_total) :
  time_jack_total = time_half1 + time_half2 → time_half2 = 6 := by
  sorry

end jack_second_half_time_l193_193384


namespace factorize_difference_of_squares_l193_193285

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l193_193285


namespace smallest_of_product_and_sum_l193_193743

theorem smallest_of_product_and_sum (a b c : ℤ) 
  (h1 : a * b * c = 32) 
  (h2 : a + b + c = 3) : 
  a = -4 ∨ b = -4 ∨ c = -4 :=
sorry

end smallest_of_product_and_sum_l193_193743


namespace ratio_of_lateral_edges_l193_193639

theorem ratio_of_lateral_edges (A B : ℝ) (hA : A > 0) (hB : B > 0) (h : A / B = 4 / 9) : 
  let upper_length_ratio := 2
  let lower_length_ratio := 3
  upper_length_ratio / lower_length_ratio = 2 / 3 :=
by 
  sorry

end ratio_of_lateral_edges_l193_193639


namespace midpoint_3d_l193_193027

/-- Midpoint calculation in 3D space -/
theorem midpoint_3d (x1 y1 z1 x2 y2 z2 : ℝ) : 
  (x1, y1, z1) = (2, -3, 6) → 
  (x2, y2, z2) = (8, 5, -4) → 
  ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2) = (5, 1, 1) := 
by
  intros
  sorry

end midpoint_3d_l193_193027


namespace polygon_at_least_9_sides_l193_193588

theorem polygon_at_least_9_sides (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → (∃ θ, θ < 45 ∧ (∀ j, 1 ≤ j ∧ j ≤ n → θ = 360 / n))):
  9 ≤ n :=
sorry

end polygon_at_least_9_sides_l193_193588


namespace fraction_of_blueberry_tart_l193_193527

/-- Let total leftover tarts be 0.91.
    Let the tart filled with cherries be 0.08.
    Let the tart filled with peaches be 0.08.
    Prove that the fraction of the tart filled with blueberries is 0.75. --/
theorem fraction_of_blueberry_tart (H_total : Real) (H_cherry : Real) (H_peach : Real)
  (H1 : H_total = 0.91) (H2 : H_cherry = 0.08) (H3 : H_peach = 0.08) :
  (H_total - (H_cherry + H_peach)) = 0.75 :=
sorry

end fraction_of_blueberry_tart_l193_193527


namespace total_pens_l193_193413

/-- Proof that Masha and Olya bought a total of 38 pens given the cost conditions. -/
theorem total_pens (r : ℕ) (h_r : r > 10) (h1 : 357 % r = 0) (h2 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193413


namespace exists_even_in_sequence_l193_193073

theorem exists_even_in_sequence 
  (a : ℕ → ℕ)
  (h₀ : ∀ n : ℕ, a (n+1) = a n + (a n % 10)) :
  ∃ n : ℕ, a n % 2 = 0 :=
sorry

end exists_even_in_sequence_l193_193073


namespace brownies_per_person_l193_193036

-- Define the conditions as constants
def columns : ℕ := 6
def rows : ℕ := 3
def people : ℕ := 6

-- Define the total number of brownies
def total_brownies : ℕ := columns * rows

-- Define the theorem to be proved
theorem brownies_per_person : total_brownies / people = 3 :=
by sorry

end brownies_per_person_l193_193036


namespace sphere_volume_increase_factor_l193_193927

theorem sphere_volume_increase_factor (r : Real) : 
  let V_original := (4 / 3) * Real.pi * r^3
  let V_increased := (4 / 3) * Real.pi * (2 * r)^3
  V_increased / V_original = 8 :=
by
  -- Definitions of volumes
  let V_original := (4 / 3) * Real.pi * r^3
  let V_increased := (4 / 3) * Real.pi * (2 * r)^3
  -- Volume ratio
  have h : V_increased / V_original = 8 := sorry
  exact h

end sphere_volume_increase_factor_l193_193927


namespace factorize_x_squared_minus_one_l193_193321

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l193_193321


namespace factorize_difference_of_squares_l193_193220

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l193_193220


namespace total_pens_l193_193415

/-- Proof that Masha and Olya bought a total of 38 pens given the cost conditions. -/
theorem total_pens (r : ℕ) (h_r : r > 10) (h1 : 357 % r = 0) (h2 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193415


namespace max_value_of_f_l193_193331

noncomputable def f (x : ℝ) : ℝ := 8 * Real.sin x + 15 * Real.cos x

theorem max_value_of_f : ∃ x : ℝ, f x = 17 :=
sorry

end max_value_of_f_l193_193331


namespace intersection_primes_evens_l193_193038

open Set

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def evens : Set ℕ := {n | n % 2 = 0}
def primes : Set ℕ := {n | is_prime n}

theorem intersection_primes_evens :
  primes ∩ evens = {2} :=
by sorry

end intersection_primes_evens_l193_193038


namespace total_pens_l193_193437

theorem total_pens (r : ℕ) (r_gt_10 : r > 10) (r_div_357 : r ∣ 357) (r_div_441 : r ∣ 441) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_l193_193437


namespace marble_count_l193_193135

variable (initial_mar: Int) (lost_mar: Int)

def final_mar (initial_mar: Int) (lost_mar: Int) : Int :=
  initial_mar - lost_mar

theorem marble_count : final_mar 16 7 = 9 := by
  trivial

end marble_count_l193_193135


namespace gcd_ab_a2b2_l193_193356

theorem gcd_ab_a2b2 (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_coprime : Nat.gcd a b = 1) :
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
by
  sorry

end gcd_ab_a2b2_l193_193356


namespace minimum_lightbulbs_needed_l193_193822

theorem minimum_lightbulbs_needed 
  (p : ℝ) (h : p = 0.95) : ∃ (n : ℕ), n = 7 ∧ (1 - ∑ k in finset.range 5, (nat.choose n k : ℝ) * p^k * (1 - p)^(n - k) ) ≥ 0.99 := 
by 
  sorry

end minimum_lightbulbs_needed_l193_193822


namespace oil_amount_in_liters_l193_193704

theorem oil_amount_in_liters (b v : ℕ) (hb : b = 20) (hv : v = 200) :
  (b * v) / 1000 = 4 :=
by
  have h1 : b * v = 4000 := by
    rw [hb, hv]
    exact rfl
  rw [h1]
  norm_num

end oil_amount_in_liters_l193_193704


namespace roots_of_polynomial_l193_193172

theorem roots_of_polynomial : 
  ∀ (x : ℝ), (x^2 + 4) * (x^2 - 4) = 0 ↔ (x = -2 ∨ x = 2) :=
by 
  sorry

end roots_of_polynomial_l193_193172


namespace factorize_x_squared_minus_one_l193_193320

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l193_193320


namespace isosceles_right_triangle_area_l193_193113

theorem isosceles_right_triangle_area (h : ℝ) (area : ℝ) (hypotenuse_condition : h = 6 * Real.sqrt 2) : 
  area = 18 :=
  sorry

end isosceles_right_triangle_area_l193_193113


namespace factorize_a_cubed_minus_four_a_l193_193567

theorem factorize_a_cubed_minus_four_a (a : ℝ) : a^3 - 4 * a = a * (a + 2) * (a - 2) :=
sorry

end factorize_a_cubed_minus_four_a_l193_193567


namespace solution_set_of_inequality_l193_193115

theorem solution_set_of_inequality (x : ℝ) : (x - 1 ≤ (1 + x) / 3) → (x ≤ 2) :=
by
  sorry

end solution_set_of_inequality_l193_193115


namespace sequence_general_term_l193_193581

theorem sequence_general_term (a : ℕ → ℕ) (n : ℕ) (h₁ : a 1 = 1) 
  (h₂ : ∀ n > 1, a n = 2 * a (n-1) + 1) : a n = 2^n - 1 :=
by
  sorry

end sequence_general_term_l193_193581


namespace factorize_x_squared_minus_1_l193_193284

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l193_193284


namespace mean_home_runs_l193_193173

theorem mean_home_runs :
  let players6 := 5
  let players8 := 6
  let players10 := 4
  let home_runs6 := players6 * 6
  let home_runs8 := players8 * 8
  let home_runs10 := players10 * 10
  let total_home_runs := home_runs6 + home_runs8 + home_runs10
  let total_players := players6 + players8 + players10
  total_home_runs / total_players = 118 / 15 :=
by
  sorry

end mean_home_runs_l193_193173


namespace right_angle_triangle_exists_l193_193739

theorem right_angle_triangle_exists (color : ℤ × ℤ → ℕ) (H1 : ∀ c : ℕ, ∃ p : ℤ × ℤ, color p = c) : 
  ∃ (A B C : ℤ × ℤ), A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ (color A ≠ color B ∧ color B ≠ color C ∧ color C ≠ color A) ∧
  ((A.1 = B.1 ∧ B.2 = C.2 ∧ A.1 - C.1 = A.2 - B.2) ∨ (A.2 = B.2 ∧ B.1 = C.1 ∧ A.1 - B.1 = A.2 - C.2)) :=
sorry

end right_angle_triangle_exists_l193_193739


namespace factorize_difference_of_squares_l193_193306

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l193_193306


namespace find_integer_pairs_l193_193568

theorem find_integer_pairs :
  ∃ (x y : ℤ), (x = 30 ∧ y = 21) ∨ (x = -21 ∧ y = -30) ∧ (x^2 + y^2 + 27 = 456 * Int.sqrt (x - y)) :=
by
  sorry

end find_integer_pairs_l193_193568


namespace father_age_difference_l193_193813

variables (F S X : ℕ)
variable (h1 : F = 33)
variable (h2 : F = 3 * S + X)
variable (h3 : F + 3 = 2 * (S + 3) + 10)

theorem father_age_difference : X = 3 :=
by
  sorry

end father_age_difference_l193_193813


namespace dad_use_per_brush_correct_l193_193980

def toothpaste_total : ℕ := 105
def mom_use_per_brush : ℕ := 2
def anne_brother_use_per_brush : ℕ := 1
def brushing_per_day : ℕ := 3
def days_to_finish : ℕ := 5

-- Defining the daily use function for Anne's Dad
def dad_use_per_brush (D : ℕ) : ℕ := D

theorem dad_use_per_brush_correct (D : ℕ) 
  (h : brushing_per_day * (mom_use_per_brush + anne_brother_use_per_brush * 2 + dad_use_per_brush D) * days_to_finish = toothpaste_total) 
  : dad_use_per_brush D = 3 :=
by sorry

end dad_use_per_brush_correct_l193_193980


namespace basis_service_B_l193_193013

def vector := ℤ × ℤ

def not_collinear (v1 v2 : vector) : Prop :=
  v1.1 * v2.2 ≠ v1.2 * v2.1

def A : vector × vector := ((0, 0), (2, 3))
def B : vector × vector := ((-1, 3), (5, -2))
def C : vector × vector := ((3, 4), (6, 8))
def D : vector × vector := ((2, -3), (-2, 3))

theorem basis_service_B : not_collinear B.1 B.2 := by
  sorry

end basis_service_B_l193_193013


namespace avg_abc_l193_193107

variable (A B C : ℕ)

-- Conditions
def avg_ac : Prop := (A + C) / 2 = 29
def age_b : Prop := B = 26

-- Theorem stating the average age of a, b, and c
theorem avg_abc (h1 : avg_ac A C) (h2 : age_b B) : (A + B + C) / 3 = 28 := by
  sorry

end avg_abc_l193_193107


namespace factorize_difference_of_squares_l193_193226

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l193_193226


namespace factorize_x_squared_minus_1_l193_193282

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l193_193282


namespace factorize_x_squared_minus_one_l193_193194

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l193_193194


namespace calc_expression_solve_system_inequalities_l193_193847

-- Proof Problem 1: Calculation
theorem calc_expression : 
  |1 - Real.sqrt 3| - Real.sqrt 2 * Real.sqrt 6 + 1 / (2 - Real.sqrt 3) - (2 / 3) ^ (-2 : ℤ) = -5 / 4 := 
by 
  sorry

-- Proof Problem 2: System of Inequalities Solution
variable (m : ℝ)
variable (x : ℝ)
  
theorem solve_system_inequalities (h : m < 0) : 
  (4 * x - 1 > x - 7) ∧ (-1 / 4 * x < 3 / 2 * m - 1) → x > 4 - 6 * m := 
by 
  sorry

end calc_expression_solve_system_inequalities_l193_193847


namespace function_increasing_and_extrema_l193_193876

open Set

theorem function_increasing_and_extrema (f : ℝ → ℝ) (a b : ℝ)
  (h1 : ∀ x ∈ Icc (2 : ℝ) 6, f x = (x - 2) / (x - 1))
  (h2 : a = 2) (h3 : b = 6) :
  (∀ x1 x2 : ℝ, x1 ∈ Icc a b ∧ x2 ∈ Icc a b ∧ x1 < x2 → f x1 < f x2) ∧ f a = 0 ∧ f b = 4 / 5 :=
sorry

end function_increasing_and_extrema_l193_193876


namespace rogers_spending_l193_193098

theorem rogers_spending (B m p : ℝ) (H1 : m = 0.25 * (B - p)) (H2 : p = 0.10 * (B - m)) : 
  m + p = (4 / 13) * B :=
sorry

end rogers_spending_l193_193098


namespace base_b_of_256_has_4_digits_l193_193508

theorem base_b_of_256_has_4_digits : ∃ (b : ℕ), b^3 ≤ 256 ∧ 256 < b^4 ∧ b = 5 :=
by
  sorry

end base_b_of_256_has_4_digits_l193_193508


namespace factorize_x_squared_minus_one_l193_193183

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l193_193183


namespace new_profit_is_122_03_l193_193526

noncomputable def new_profit_percentage (P : ℝ) (tax_rate : ℝ) (profit_rate : ℝ) (market_increase_rate : ℝ) (months : ℕ) : ℝ :=
  let total_cost := P * (1 + tax_rate)
  let initial_selling_price := total_cost * (1 + profit_rate)
  let market_price_after_months := initial_selling_price * (1 + market_increase_rate) ^ months
  let final_selling_price := 2 * initial_selling_price
  let profit := final_selling_price - total_cost
  (profit / total_cost) * 100

theorem new_profit_is_122_03 :
  new_profit_percentage (P : ℝ) 0.18 0.40 0.05 3 = 122.03 := 
by
  sorry

end new_profit_is_122_03_l193_193526


namespace coin_probability_l193_193520

theorem coin_probability :
  let value_quarters : ℚ := 15.00
  let value_nickels : ℚ := 15.00
  let value_dimes : ℚ := 10.00
  let value_pennies : ℚ := 5.00
  let number_quarters := value_quarters / 0.25
  let number_nickels := value_nickels / 0.05
  let number_dimes := value_dimes / 0.10
  let number_pennies := value_pennies / 0.01
  let total_coins := number_quarters + number_nickels + number_dimes + number_pennies
  let probability := (number_quarters + number_dimes) / total_coins
  probability = (1 / 6) := by 
sorry

end coin_probability_l193_193520


namespace factorize_x_squared_minus_1_l193_193281

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_1_l193_193281


namespace ellipse_triangle_is_isosceles_right_l193_193971

theorem ellipse_triangle_is_isosceles_right (e : ℝ) (a b c k : ℝ)
  (H1 : e = (c / a))
  (H2 : e = (Real.sqrt 2) / 2)
  (H3 : b^2 = a^2 * (1 - e^2))
  (H4 : a = 2 * k)
  (H5 : b = k * Real.sqrt 2)
  (H6 : c = k * Real.sqrt 2) :
  (4 * k)^2 = (2 * (k * Real.sqrt 2))^2 + (2 * (k * Real.sqrt 2))^2 :=
by
  sorry

end ellipse_triangle_is_isosceles_right_l193_193971


namespace min_value_expression_l193_193890

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) : 
  ∃ m, m = 20 ∧ (∀ a b : ℝ, a = x - 2 ∧ b = y - 2 →
    let exp := (a + 2) ^ 2 + 1 / (b + (b + 2) ^ 2 + 1 / a) in exp ≥ m) :=
sorry

end min_value_expression_l193_193890


namespace total_turnover_in_first_quarter_l193_193523

theorem total_turnover_in_first_quarter (x : ℝ) : 
  200 + 200 * (1 + x) + 200 * (1 + x) ^ 2 = 1000 :=
sorry

end total_turnover_in_first_quarter_l193_193523


namespace center_of_symmetry_l193_193687

-- Define the symmetry conditions
def is_symmetric_about_x_axis (figure : set (ℝ × ℝ)) : Prop :=
  ∀ {x y : ℝ}, (x, y) ∈ figure → (x, -y) ∈ figure

def is_symmetric_about_y_axis (figure : set (ℝ × ℝ)) : Prop :=
  ∀ {x y : ℝ}, (x, y) ∈ figure → (-x, y) ∈ figure

-- The main theorem stating that a figure with two perpendicular axes of symmetry has a center of symmetry
theorem center_of_symmetry
  (figure : set (ℝ × ℝ))
  (h_x : is_symmetric_about_x_axis figure)
  (h_y : is_symmetric_about_y_axis figure) :
  ∀ {x y : ℝ}, (x, y) ∈ figure → (-x, -y) ∈ figure :=
by
  sorry

end center_of_symmetry_l193_193687


namespace preferred_pets_combination_l193_193697

-- Define the number of puppies, kittens, and hamsters
def num_puppies : ℕ := 20
def num_kittens : ℕ := 10
def num_hamsters : ℕ := 12

-- State the main theorem to prove, that the number of ways Alice, Bob, and Charlie 
-- can buy their preferred pets is 2400
theorem preferred_pets_combination : num_puppies * num_kittens * num_hamsters = 2400 :=
by
  sorry

end preferred_pets_combination_l193_193697


namespace cost_of_dozen_pens_l193_193809

theorem cost_of_dozen_pens 
  (x : ℝ)
  (hx_pos : 0 < x)
  (h1 : 3 * (5 * x) + 5 * x = 150)
  (h2 : 5 * x / x = 5): 
  12 * (5 * x) = 450 :=
by
  sorry

end cost_of_dozen_pens_l193_193809


namespace partial_fraction_decomposition_l193_193324

theorem partial_fraction_decomposition (C D : ℝ): 
  (∀ x : ℝ, (x ≠ 12 ∧ x ≠ -4) → 
    (6 * x + 15) / ((x - 12) * (x + 4)) = C / (x - 12) + D / (x + 4))
  → (C = 87 / 16 ∧ D = 9 / 16) :=
by
  -- This would be the place to provide the proof, but we skip it as per instructions
  sorry

end partial_fraction_decomposition_l193_193324


namespace trig_identity_l193_193842

theorem trig_identity (α : ℝ) :
  1 - Real.cos (2 * α - Real.pi) + Real.cos (4 * α - 2 * Real.pi) =
  4 * Real.cos (2 * α) * Real.cos (Real.pi / 6 + α) * Real.cos (Real.pi / 6 - α) :=
by
  sorry

end trig_identity_l193_193842


namespace factorize_difference_of_squares_l193_193301

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l193_193301


namespace book_distribution_l193_193127

theorem book_distribution (x : ℕ) (h1 : 9 * x + 7 < 11 * x) : 
  9 * x + 7 = totalBooks - 9 * x ∧ totalBooks - 9 * x = 7 :=
by
  sorry

end book_distribution_l193_193127


namespace total_pairs_of_shoes_l193_193655

-- Conditions as Definitions
def blue_shoes := 540
def purple_shoes := 355
def green_shoes := purple_shoes  -- The number of green shoes is equal to the number of purple shoes

-- The theorem we need to prove
theorem total_pairs_of_shoes : blue_shoes + green_shoes + purple_shoes = 1250 := by
  sorry

end total_pairs_of_shoes_l193_193655


namespace algebraic_expression_value_l193_193775

theorem algebraic_expression_value (x : ℝ) (h : 3 * x^2 - 4 * x = 6): 6 * x^2 - 8 * x - 9 = 3 :=
by sorry

end algebraic_expression_value_l193_193775


namespace Ruth_sandwiches_l193_193956

theorem Ruth_sandwiches (sandwiches_left sandwiches_ruth sandwiches_brother sandwiches_first_cousin sandwiches_two_cousins total_sandwiches : ℕ)
  (h_ruth : sandwiches_ruth = 1)
  (h_brother : sandwiches_brother = 2)
  (h_first_cousin : sandwiches_first_cousin = 2)
  (h_two_cousins : sandwiches_two_cousins = 2)
  (h_left : sandwiches_left = 3) :
  total_sandwiches = sandwiches_left + sandwiches_two_cousins + sandwiches_first_cousin + sandwiches_ruth + sandwiches_brother :=
by
  sorry

end Ruth_sandwiches_l193_193956


namespace necessary_but_not_sufficient_l193_193978

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 > 4) → (x > 2 ∨ x < -2) ∧ ¬((x^2 > 4) ↔ (x > 2)) :=
by
  intros h
  have h1 : x > 2 ∨ x < -2 := by sorry
  have h2 : ¬((x^2 > 4) ↔ (x > 2)) := by sorry
  exact And.intro h1 h2

end necessary_but_not_sufficient_l193_193978


namespace Marty_paint_combinations_l193_193087

theorem Marty_paint_combinations :
  let colors := 5 -- blue, green, yellow, black, white
  let styles := 3 -- brush, roller, sponge
  let invalid_combinations := 1 * 1 -- white paint with roller
  let total_combinations := (4 * styles) + (1 * (styles - 1))
  total_combinations = 14 :=
by
  -- Define the total number of combinations excluding the invalid one
  let colors := 5
  let styles := 3
  let invalid_combinations := 1 -- number of invalid combinations (white with roller)
  let valid_combinations := (4 * styles) + (1 * (styles - 1))
  show valid_combinations = 14
  {
    exact rfl -- This will assert that the valid_combinations indeed equals 14
  }

end Marty_paint_combinations_l193_193087


namespace total_population_of_cities_l193_193933

theorem total_population_of_cities 
    (number_of_cities : ℕ) 
    (average_population : ℕ) 
    (h1 : number_of_cities = 25) 
    (h2 : average_population = (5200 + 5700) / 2) : 
    number_of_cities * average_population = 136250 := by 
    sorry

end total_population_of_cities_l193_193933


namespace set_listing_l193_193762

open Set

def A : Set (ℤ × ℤ) := {p | ∃ x y : ℤ, p = (x, y) ∧ x^2 = y + 1 ∧ |x| < 2}

theorem set_listing :
  A = {(-1, 0), (0, -1), (1, 0)} :=
by {
  sorry
}

end set_listing_l193_193762


namespace find_a_b_of_solution_set_l193_193116

theorem find_a_b_of_solution_set :
  ∃ a b : ℝ, (∀ x : ℝ, x^2 + (a + 1) * x + a * b = 0 ↔ x = -1 ∨ x = 4) → a + b = -3 :=
by
  sorry

end find_a_b_of_solution_set_l193_193116


namespace smallest_b_for_factors_l193_193335

theorem smallest_b_for_factors (b : ℕ) (h : ∃ r s : ℤ, (x : ℤ) → (x + r) * (x + s) = x^2 + ↑b * x + 2016 ∧ r * s = 2016) :
  b = 90 :=
by
  sorry

end smallest_b_for_factors_l193_193335


namespace apple_distribution_l193_193662

theorem apple_distribution (x : ℕ) (h₁ : 1430 % x = 0) (h₂ : 1430 % (x + 45) = 0) (h₃ : 1430 / x - 1430 / (x + 45) = 9) : 
  1430 / x = 22 :=
by
  sorry

end apple_distribution_l193_193662


namespace pure_imaginary_real_zero_l193_193767

theorem pure_imaginary_real_zero (a : ℝ) (i : ℂ) (hi : i^2 = -1) (h : a * i = 0 + a * i) : a = 0 := by
  sorry

end pure_imaginary_real_zero_l193_193767


namespace geometric_series_proof_l193_193729

theorem geometric_series_proof (y : ℝ) :
  ((1 + (1/3) + (1/9) + (1/27) + ∑' n : ℕ, (1 / 3^(n+1))) * 
   (1 - (1/3) + (1/9) - (1/27) + ∑' n : ℕ, ((-1)^n * (1 / 3^(n+1)))) = 
   1 + (1/y) + (1/y^2) + (∑' n : ℕ, (1 / y^(n+1)))) → y = 9 := by
  sorry

end geometric_series_proof_l193_193729


namespace billy_video_count_l193_193017

theorem billy_video_count 
  (generate_suggestions : ℕ) 
  (rounds : ℕ) 
  (videos_in_total : ℕ)
  (H1 : generate_suggestions = 15)
  (H2 : rounds = 5)
  (H3 : videos_in_total = generate_suggestions * rounds + 1) : 
  videos_in_total = 76 := 
by
  sorry

end billy_video_count_l193_193017


namespace factorization_of_difference_of_squares_l193_193254

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l193_193254


namespace sum_of_remainders_mod_13_l193_193496

theorem sum_of_remainders_mod_13 
  (a b c d : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := 
by
  sorry

end sum_of_remainders_mod_13_l193_193496


namespace no_such_fraction_exists_l193_193551

theorem no_such_fraction_exists :
  ∀ (x y : ℕ), (Nat.coprime x y) → (0 < x) → (0 < y) → ¬ (1.2 * (x:ℚ) / (y:ℚ) = (x+1:ℚ) / (y+1:ℚ)) := by
  sorry

end no_such_fraction_exists_l193_193551


namespace optionB_unfactorable_l193_193159

-- Definitions for the conditions
def optionA (a b : ℝ) : ℝ := -a^2 + b^2
def optionB (x y : ℝ) : ℝ := x^2 + y^2
def optionC (z : ℝ) : ℝ := 49 - z^2
def optionD (m : ℝ) : ℝ := 16 - 25 * m^2

-- The proof statement that option B cannot be factored over the real numbers
theorem optionB_unfactorable (x y : ℝ) : ¬ ∃ (p q : ℝ → ℝ), p x * q y = x^2 + y^2 :=
sorry -- Proof to be filled in

end optionB_unfactorable_l193_193159


namespace chocolates_bought_l193_193770

theorem chocolates_bought (C S : ℝ) (h1 : N * C = 45 * S) (h2 : 80 = ((S - C) / C) * 100) : 
  N = 81 :=
by
  sorry

end chocolates_bought_l193_193770


namespace fraction_of_peaches_l193_193946

-- Define the number of peaches each person has
def Benjy_peaches : ℕ := 5
def Martine_peaches : ℕ := 16
def Gabrielle_peaches : ℕ := 15

-- Condition that Martine has 6 more than twice Benjy's peaches
def Martine_cond : Prop := Martine_peaches = 2 * Benjy_peaches + 6

-- The goal is to prove the fraction of Gabrielle's peaches that Benjy has
theorem fraction_of_peaches :
  Martine_cond → (Benjy_peaches : ℚ) / (Gabrielle_peaches : ℚ) = 1 / 3 :=
by
  -- Assuming the condition holds
  intro h
  rw [Martine_cond] at h
  -- Use the condition directly, since Martine_cond implies Benjy_peaches = 5
  exact sorry

end fraction_of_peaches_l193_193946


namespace growth_pattern_equation_l193_193005

theorem growth_pattern_equation (x : ℕ) :
  1 + x + x^2 = 73 :=
sorry

end growth_pattern_equation_l193_193005


namespace factorize_x_squared_minus_one_l193_193190

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l193_193190


namespace total_pens_bought_l193_193402

theorem total_pens_bought (r : ℕ) (hr : r > 10) (hm : 357 % r = 0) (ho : 441 % r = 0) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_bought_l193_193402


namespace factorization_of_difference_of_squares_l193_193253

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l193_193253


namespace fifteenth_term_l193_193561

noncomputable def seq : ℕ → ℝ
| 0       => 3
| 1       => 4
| (n + 2) => 12 / seq (n + 1)

theorem fifteenth_term :
  seq 14 = 3 :=
sorry

end fifteenth_term_l193_193561


namespace geometric_series_proof_l193_193728

theorem geometric_series_proof (y : ℝ) :
  ((1 + (1/3) + (1/9) + (1/27) + ∑' n : ℕ, (1 / 3^(n+1))) * 
   (1 - (1/3) + (1/9) - (1/27) + ∑' n : ℕ, ((-1)^n * (1 / 3^(n+1)))) = 
   1 + (1/y) + (1/y^2) + (∑' n : ℕ, (1 / y^(n+1)))) → y = 9 := by
  sorry

end geometric_series_proof_l193_193728


namespace total_points_scored_l193_193395

theorem total_points_scored (m2 m3 m1 o2 o3 o1 : ℕ) 
  (H1 : m2 = 25) 
  (H2 : m3 = 8) 
  (H3 : m1 = 10) 
  (H4 : o2 = 2 * m2) 
  (H5 : o3 = m3 / 2) 
  (H6 : o1 = m1 / 2) : 
  (2 * m2 + 3 * m3 + m1) + (2 * o2 + 3 * o3 + o1) = 201 := 
by
  sorry

end total_points_scored_l193_193395


namespace polynomial_coefficients_l193_193096

noncomputable def a : ℝ := 15
noncomputable def b : ℝ := -198
noncomputable def c : ℝ := 1

theorem polynomial_coefficients :
  (∀ x₁ x₂ x₃ : ℝ, 
    (x₁ + x₂ + x₃ = 0) ∧ 
    (x₁ * x₂ + x₂ * x₃ + x₃ * x₁ = -3) ∧ 
    (x₁ * x₂ * x₃ = -1) → 
    (a = 15) ∧ 
    (b = -198) ∧ 
    (c = 1)) := 
by sorry

end polynomial_coefficients_l193_193096


namespace find_k_l193_193029

theorem find_k (x y z k : ℝ) 
  (h1 : 9 / (x + y) = k / (x + 2 * z)) 
  (h2 : 9 / (x + y) = 14 / (z - y)) 
  (h3 : y = 2 * x) 
  (h4 : x + z = 10) :
  k = 46 :=
by
  sorry

end find_k_l193_193029


namespace min_lightbulbs_for_5_working_l193_193824

noncomputable def minimal_lightbulbs_needed (p : ℝ) (k : ℕ) (threshold : ℝ) : ℕ :=
  Inf {n : ℕ | (∑ i in finset.range (n + 1), if i < k then 0 else nat.choose n i * p ^ i * (1 - p) ^ (n - i)) ≥ threshold}

theorem min_lightbulbs_for_5_working (p : ℝ) (k : ℕ) (threshold : ℝ) :
  p = 0.95 →
  k = 5 →
  threshold = 0.99 →
  minimal_lightbulbs_needed p k threshold = 7 := by
  intros
  sorry

end min_lightbulbs_for_5_working_l193_193824


namespace max_length_CD_l193_193171

open Real

/-- Given a circle with center O and diameter AB = 20 units,
    with points C and D positioned such that C is 6 units away from A
    and D is 7 units away from B on the diameter AB,
    prove that the maximum length of the direct path from C to D is 7 units.
-/
theorem max_length_CD {A B C D : ℝ} 
    (diameter : dist A B = 20) 
    (C_pos : dist A C = 6) 
    (D_pos : dist B D = 7) : 
    dist C D = 7 :=
by
  -- Details of the proof would go here
  sorry

end max_length_CD_l193_193171


namespace root_polynomial_sum_l193_193607

theorem root_polynomial_sum {b c : ℝ} (hb : b^2 - b - 1 = 0) (hc : c^2 - c - 1 = 0) : 
  (1 / (1 - b)) + (1 / (1 - c)) = -1 := 
sorry

end root_polynomial_sum_l193_193607


namespace factorize_x_squared_minus_one_l193_193181

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l193_193181


namespace gcd_150_450_l193_193747

theorem gcd_150_450 : Nat.gcd 150 450 = 150 := by
  sorry

end gcd_150_450_l193_193747


namespace yellow_balls_count_l193_193929

theorem yellow_balls_count (x y z : ℕ) 
  (h1 : x + y + z = 68)
  (h2 : y = 2 * x)
  (h3 : 3 * z = 4 * y) : y = 24 :=
by {
  sorry
}

end yellow_balls_count_l193_193929


namespace contrapositive_even_statement_l193_193638

-- Translate the conditions to Lean 4 definitions
def is_even (n : Int) : Prop := ∃ k : Int, n = 2 * k

theorem contrapositive_even_statement (a b : Int) :
  (¬ is_even (a + b) → ¬ (is_even a ∧ is_even b)) ↔ 
  (is_even a ∧ is_even b → is_even (a + b)) :=
by sorry

end contrapositive_even_statement_l193_193638


namespace factorize_difference_of_squares_l193_193228

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l193_193228


namespace factorize_difference_of_squares_l193_193288

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_difference_of_squares_l193_193288


namespace jimin_yuna_difference_l193_193463

-- Definitions based on the conditions.
def seokjin_marbles : ℕ := 3
def yuna_marbles : ℕ := seokjin_marbles - 1
def jimin_marbles : ℕ := seokjin_marbles * 2

-- Theorem stating the problem we need to prove: the difference in marbles between Jimin and Yuna is 4.
theorem jimin_yuna_difference : jimin_marbles - yuna_marbles = 4 :=
by sorry

end jimin_yuna_difference_l193_193463


namespace numer_greater_than_denom_iff_l193_193897

theorem numer_greater_than_denom_iff (x : ℝ) (h : -1 ≤ x ∧ x ≤ 3) : 
  (4 * x - 3 > 9 - 2 * x) ↔ (2 < x ∧ x ≤ 3) :=
sorry

end numer_greater_than_denom_iff_l193_193897


namespace women_in_first_group_l193_193966

-- Define the number of women in the first group as W
variable (W : ℕ)

-- Define the work parameters
def work_per_day := 75 / 8
def work_per_hour_first_group := work_per_day / 5

def work_per_day_second_group := 30 / 3
def work_per_hour_second_group := work_per_day_second_group / 8

-- The equation comes from work/hour equivalence
theorem women_in_first_group :
  (W : ℝ) * work_per_hour_first_group = 4 * work_per_hour_second_group → W = 5 :=
by 
  sorry

end women_in_first_group_l193_193966


namespace min_bulbs_needed_l193_193818

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k) * (p^k) * ((1-p)^(n-k))

theorem min_bulbs_needed (p : ℝ) (k : ℕ) (target_prob : ℝ) :
  p = 0.95 → k = 5 → target_prob = 0.99 →
  ∃ n, (∑ i in finset.range (n+1), if i ≥ 5 then binomial_probability n i p else 0) ≥ target_prob ∧ 
  (¬ ∃ m, m < n ∧ (∑ i in finset.range (m+1), if i ≥ 5 then binomial_probability m i p else 0) ≥ target_prob) :=
by
  sorry

end min_bulbs_needed_l193_193818


namespace max_servings_l193_193989

open Nat

def servings (cucumbers tomatoes brynza_peppers brynza_grams: Nat) : Nat :=
  min (floor (cucumbers / 2))
    (min (floor (tomatoes / 2))
      (min (floor (brynza_peppers / 75)) brynza_grams))

theorem max_servings (cucumbers tomatoes peppers: Nat) (brynza_grams: Rat) 
  (cuc_reqs toma_reqs brynza_per pepper_reqs: Nat) (br_in_grams: Nat) : 
  servings cucumbers tomatoes brynza_grams peppers = 56 :=
by
  have cuc_portions : cucumbers / cuc_reqs = 58 := by sorry
  have toma_portions : tomatoes / toma_reqs = 58 := by sorry
  have brynza_portions : (br_in_grams / brynza_per) = 56 := by sorry
  have pepper_portions : peppers / pepper_reqs = 60 := by sorry
  exact min (min (min cuc_portions toma_portions) brynza_portions) pepper_portions
  

end max_servings_l193_193989


namespace pipe_B_leak_time_l193_193139

theorem pipe_B_leak_time (t_B : ℝ) : (1 / 12 - 1 / t_B = 1 / 36) → t_B = 18 :=
by
  intro h
  -- Proof goes here
  sorry

end pipe_B_leak_time_l193_193139


namespace solution_set_quadratic_ineq_all_real_l193_193975

theorem solution_set_quadratic_ineq_all_real (a b c : ℝ) :
  (∀ x : ℝ, (a / 3) * x^2 + 2 * b * x - c < 0) ↔ (a > 0 ∧ 4 * b^2 - (4 / 3) * a * c < 0) :=
by
  sorry

end solution_set_quadratic_ineq_all_real_l193_193975


namespace selling_price_l193_193160

theorem selling_price (CP P : ℝ) (hCP : CP = 320) (hP : P = 0.25) : CP + (P * CP) = 400 :=
by
  sorry

end selling_price_l193_193160


namespace union_A_B_l193_193916

def A (x : ℝ) : Set ℝ := {x ^ 2, 2 * x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem union_A_B (x : ℝ) (h : {9} = A x ∩ B x) :
  (A x ∪ B x) = {(-8 : ℝ), -7, -4, 4, 9} := by
  sorry

end union_A_B_l193_193916


namespace arithmetic_seq_question_l193_193597

theorem arithmetic_seq_question (a : ℕ → ℤ) (d : ℤ) (h_arith : ∀ n, a n = a 1 + (n - 1) * d)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  2 * a 10 - a 12 = 24 := 
sorry

end arithmetic_seq_question_l193_193597


namespace total_pens_l193_193427

theorem total_pens (r : ℕ) (h1 : r > 10)
  (h2 : 357 % r = 0)
  (h3 : 441 % r = 0) :
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_l193_193427


namespace max_value_of_f_l193_193330

-- Define the function
def f (x : ℝ) : ℝ := 8 * Real.sin x + 15 * Real.cos x

-- State the theorem
theorem max_value_of_f : ∃ x : ℝ, f x = 17 ∧ ∀ y : ℝ, f y ≤ 17 :=
by
  -- No proof is provided, only the statement
  sorry

end max_value_of_f_l193_193330


namespace factorize_difference_of_squares_l193_193304

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l193_193304


namespace solve_trigonometric_inequality_l193_193033

noncomputable def trigonometric_inequality (x : ℝ) : Prop :=
  x ∈ Set.Ioo 0 (2 * Real.pi) ∧ 2^x * (2 * Real.sin x - Real.sqrt 3) ≥ 0

theorem solve_trigonometric_inequality :
  ∀ x, x ∈ Set.Ioo 0 (2 * Real.pi) → (2^x * (2 * Real.sin x - Real.sqrt 3) ≥ 0 ↔ x ∈ Set.Icc (Real.pi / 3) (2 * Real.pi / 3)) :=
by
  intros x hx
  sorry

end solve_trigonometric_inequality_l193_193033


namespace even_function_l193_193793

noncomputable def f : ℝ → ℝ :=
sorry

theorem even_function (f : ℝ → ℝ) (h1 : ∀ x, f x = f (-x)) 
  (h2 : ∀ x, -1 ≤ x ∧ x ≤ 0 → f x = x - 1) : f (1/2) = -3/2 :=
sorry

end even_function_l193_193793


namespace maria_made_144_cookies_l193_193945

def cookies (C : ℕ) : Prop :=
  (2 * 1 / 4 * C = 72)

theorem maria_made_144_cookies: ∃ (C : ℕ), cookies C ∧ C = 144 :=
by
  existsi 144
  unfold cookies
  sorry

end maria_made_144_cookies_l193_193945


namespace abs_diff_condition_l193_193586

theorem abs_diff_condition {a b : ℝ} (h1 : |a| = 1) (h2 : |b - 1| = 2) (h3 : a > b) : a - b = 2 := 
sorry

end abs_diff_condition_l193_193586


namespace total_pens_bought_l193_193457

theorem total_pens_bought (r : ℕ) (r_gt_10 : r > 10) (r_divides_357 : 357 % r = 0) (r_divides_441 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
by sorry

end total_pens_bought_l193_193457


namespace center_of_symmetry_l193_193688

-- Define the given conditions
def has_axis_symmetry_x (F : Set (ℝ × ℝ)) : Prop := 
  ∀ (x y : ℝ), (x, y) ∈ F → (-x, y) ∈ F

def has_axis_symmetry_y (F : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ F → (x, -y) ∈ F
  
-- Define the central proof goal
theorem center_of_symmetry (F : Set (ℝ × ℝ)) (H1: has_axis_symmetry_x F) (H2: has_axis_symmetry_y F) :
  ∀ (x y : ℝ), (x, y) ∈ F → (-x, -y) ∈ F :=
sorry

end center_of_symmetry_l193_193688


namespace kibble_remaining_l193_193084

theorem kibble_remaining 
  (initial_amount : ℕ) (morning_mary : ℕ) (evening_mary : ℕ) 
  (afternoon_frank : ℕ) (evening_frank : ℕ) :
  initial_amount = 12 →
  morning_mary = 1 →
  evening_mary = 1 →
  afternoon_frank = 1 →
  evening_frank = 2 * afternoon_frank →
  initial_amount - (morning_mary + evening_mary + afternoon_frank + evening_frank) = 7 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  done

end kibble_remaining_l193_193084


namespace factorization_of_x_squared_minus_one_l193_193233

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l193_193233


namespace puppies_adopted_each_day_l193_193696

variable (initial_puppies additional_puppies days total_puppies puppies_per_day : ℕ)

axiom initial_puppies_ax : initial_puppies = 9
axiom additional_puppies_ax : additional_puppies = 12
axiom days_ax : days = 7
axiom total_puppies_ax : total_puppies = initial_puppies + additional_puppies
axiom adoption_rate_ax : total_puppies / days = puppies_per_day

theorem puppies_adopted_each_day : 
  initial_puppies = 9 → additional_puppies = 12 → days = 7 → total_puppies = initial_puppies + additional_puppies → total_puppies / days = puppies_per_day → puppies_per_day = 3 :=
by
  intro initial_puppies_ax additional_puppies_ax days_ax total_puppies_ax adoption_rate_ax
  sorry

end puppies_adopted_each_day_l193_193696


namespace circle_intersects_y_axis_with_constraints_l193_193064

theorem circle_intersects_y_axis_with_constraints {m n : ℝ} 
    (H1 : n = m ^ 2 + 2 * m + 2) 
    (H2 : abs m <= 2) : 
    1 ≤ n ∧ n < 10 :=
sorry

end circle_intersects_y_axis_with_constraints_l193_193064


namespace solve_inequality_l193_193102

theorem solve_inequality (x : ℝ) :
  x * Real.log (x^2 + x + 1) / Real.log 10 < 0 ↔ x < -1 :=
sorry

end solve_inequality_l193_193102


namespace factorize_difference_of_squares_l193_193298

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l193_193298


namespace initial_outlay_l193_193097

-- Definition of given conditions
def manufacturing_cost (I : ℝ) (sets : ℕ) (cost_per_set : ℝ) : ℝ := I + sets * cost_per_set
def revenue (sets : ℕ) (price_per_set : ℝ) : ℝ := sets * price_per_set
def profit (revenue manufacturing_cost : ℝ) : ℝ := revenue - manufacturing_cost

-- Given data
def sets : ℕ := 500
def cost_per_set : ℝ := 20
def price_per_set : ℝ := 50
def given_profit : ℝ := 5000

-- The statement to prove
theorem initial_outlay (I : ℝ) : 
  profit (revenue sets price_per_set) (manufacturing_cost I sets cost_per_set) = given_profit → 
  I = 10000 := by
  sorry

end initial_outlay_l193_193097


namespace total_number_of_cows_l193_193141

theorem total_number_of_cows (n : ℕ) 
  (h1 : n > 0) 
  (h2 : (1/3) * n + (1/6) * n + (1/8) * n + 9 = n) : n = 216 :=
sorry

end total_number_of_cows_l193_193141


namespace factor_difference_of_squares_l193_193267

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l193_193267


namespace cube_surface_area_l193_193759

theorem cube_surface_area (a : ℕ) (h : a = 2) : 6 * a^2 = 24 := 
by
  sorry

end cube_surface_area_l193_193759


namespace N_perfect_square_l193_193355

theorem N_perfect_square (N : ℕ) (hN_pos : N > 0) 
  (h_pairs : ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 2005 ∧ 
  ∀ p ∈ pairs, (1 : ℚ) / (p.1 : ℚ) + (1 : ℚ) / (p.2 : ℚ) = (1 : ℚ) / N ∧ p.1 > 0 ∧ p.2 > 0) : 
  ∃ k : ℕ, N = k^2 := 
sorry

end N_perfect_square_l193_193355


namespace Alan_finish_time_third_task_l193_193530

theorem Alan_finish_time_third_task :
  let start_time := 480 -- 8:00 AM in minutes from midnight
  let finish_time_second_task := 675 -- 11:15 AM in minutes from midnight
  let total_tasks_time := 195 -- Total time spent on first two tasks
  let first_task_time := 65 -- Time taken for the first task calculated as per the solution
  let second_task_time := 130 -- Time taken for the second task calculated as per the solution
  let third_task_time := 65 -- Time taken for the third task
  let finish_time_third_task := 740 -- 12:20 PM in minutes from midnight
  start_time + total_tasks_time + third_task_time = finish_time_third_task :=
by
  -- proof here
  sorry

end Alan_finish_time_third_task_l193_193530


namespace exact_time_now_l193_193380

noncomputable def minute_hand_position (t : ℝ) : ℝ := 6 * (t + 4)
noncomputable def hour_hand_position (t : ℝ) : ℝ := 0.5 * (t - 2) + 270
noncomputable def is_opposite (x y : ℝ) : Prop := |x - y| = 180

theorem exact_time_now (t : ℝ) (h1 : 0 ≤ t) (h2 : t < 60)
  (h3 : is_opposite (minute_hand_position t) (hour_hand_position t)) :
  t = 591/50 :=
by
  sorry

end exact_time_now_l193_193380


namespace technicans_permanent_50pct_l193_193593

noncomputable def percentage_technicians_permanent (p : ℝ) : Prop :=
  let technicians := 0.5
  let non_technicians := 0.5
  let temporary := 0.5
  (0.5 * (1 - 0.5)) + (technicians * p) = 0.5 ->
  p = 0.5

theorem technicans_permanent_50pct (p : ℝ) :
  percentage_technicians_permanent p :=
sorry

end technicans_permanent_50pct_l193_193593


namespace f_one_value_l193_193572

def f (x a: ℝ) : ℝ := x^2 + a*x - 3*a - 9

theorem f_one_value (a : ℝ) (h : ∀ x, f x a ≥ 0) : f 1 a = 4 :=
by
  sorry

end f_one_value_l193_193572


namespace triangle_area_is_120_l193_193009

-- Define the triangle sides
def a : ℕ := 10
def b : ℕ := 24
def c : ℕ := 26

-- Define a function to calculate the area of a right-angled triangle
noncomputable def right_triangle_area (a b : ℕ) : ℕ := (a * b) / 2

-- Statement to prove the area of the triangle
theorem triangle_area_is_120 : right_triangle_area 10 24 = 120 :=
by
  sorry

end triangle_area_is_120_l193_193009


namespace games_in_tournament_l193_193008

def single_elimination_games (n : Nat) : Nat :=
  n - 1

theorem games_in_tournament : single_elimination_games 24 = 23 := by
  sorry

end games_in_tournament_l193_193008


namespace range_of_a_l193_193391

noncomputable def f (a x : ℝ) :=
  if x < 0 then
    9 * x + a^2 / x + 7
  else
    9 * x + a^2 / x - 7

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → f a x ≥ a + 1) → a ≤ -8 / 7 :=
  sorry

end range_of_a_l193_193391


namespace sin_double_angle_l193_193925

theorem sin_double_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin α = 3 / 5) : 
  Real.sin (2 * α) = 24 / 25 :=
by sorry

end sin_double_angle_l193_193925


namespace factorize_x_squared_minus_one_l193_193206

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l193_193206


namespace base_conversion_subtraction_l193_193563

theorem base_conversion_subtraction :
  let n1 := 5 * 7^4 + 2 * 7^3 + 1 * 7^2 + 4 * 7^1 + 3 * 7^0
  let n2 := 1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0
  n1 - n2 = 7422 :=
by
  let n1 := 5 * 7^4 + 2 * 7^3 + 1 * 7^2 + 4 * 7^1 + 3 * 7^0
  let n2 := 1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0
  show n1 - n2 = 7422
  sorry

end base_conversion_subtraction_l193_193563


namespace max_sum_of_four_integers_with_product_360_l193_193825

theorem max_sum_of_four_integers_with_product_360 :
  ∃ a b c d : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧ a * b * c * d = 360 ∧ a + b + c + d = 66 :=
sorry

end max_sum_of_four_integers_with_product_360_l193_193825


namespace total_pens_l193_193445

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193445


namespace PQDE_concyclic_l193_193712

open Classical

variables {A B C D E P Q : Point} (circle : Circle)

-- Conditions
def is_inscribed_tri (triangle : Triangle) := 
  triangle.inscribed_in circle

def is_isosceles (triangle : Triangle) := 
  triangle.AB = triangle.AC

def AE_is_chord := ∃ (E : Point), E ∈ circle ∧ A ≠ E
def AQ_is_chord := ∃ (Q : Point), Q ∈ circle ∧ A ≠ Q

def AE_intersects_BC_at_D :=
  AE_is_chord ∧ ∃ (D : Point), D ∈ BC ∨ D ∈ extension BC

def AQ_intersects_CB_at_P :=
  AQ_is_chord ∧ ∃ (P : Point), P ∈ extension CB

-- Question (Prove that P, Q, D, E are concyclic)
theorem PQDE_concyclic 
  (triangle : Triangle)
  [is_inscribed_tri triangle]
  [is_isosceles triangle]
  [AE_is_chord]
  [AQ_is_chord]
  [AE_intersects_BC_at_D]
  [AQ_intersects_CB_at_P] :
  CyclicOrder.circle4 A B C D E P Q :=
sorry

end PQDE_concyclic_l193_193712


namespace g_neg_one_add_g_one_l193_193349

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f (x - y) = f x * g y - f y * g x
axiom f_one_ne_zero : f 1 ≠ 0
axiom f_one_eq_f_two : f 1 = f 2

theorem g_neg_one_add_g_one : g (-1) + g 1 = 1 := by
  sorry

end g_neg_one_add_g_one_l193_193349


namespace garden_dimensions_l193_193944

theorem garden_dimensions
  (w l : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : l * w = 600) : 
  w = 10 * Real.sqrt 3 ∧ l = 20 * Real.sqrt 3 :=
by
  sorry

end garden_dimensions_l193_193944


namespace probability_of_7_successes_in_7_trials_l193_193669

open Probability

/-- Define the given conditions for the problem -/
def n : ℕ := 7
def k : ℕ := 7
def p : ℚ := 2 / 7

/-- The binomial coefficient and the probability of success in n trials -/
theorem probability_of_7_successes_in_7_trials :
  P(X = k) = (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) :=
by
  have bep_0 : nat.choose 7 7 = 1, from sorry,
  have p_power_k : p ^ k = (2 / 7) ^ 7, from sorry,
  have q_power_rem : (1 - p) ^ (n - k) = 1, from sorry,
  have p_eq_frac : (2 / 7) ^ 7 * 1 = 128 / 823543, from sorry,
  show 1 * (2 / 7) ^ 7 * 1 = 128 / 823543, by sorry

end probability_of_7_successes_in_7_trials_l193_193669


namespace heracles_age_l193_193718

theorem heracles_age
  (H : ℕ)
  (audrey_current_age : ℕ)
  (audrey_in_3_years : ℕ)
  (h1 : audrey_current_age = H + 7)
  (h2 : audrey_in_3_years = audrey_current_age + 3)
  (h3 : audrey_in_3_years = 2 * H)
  : H = 10 :=
by
  sorry

end heracles_age_l193_193718


namespace bernoulli_trial_probability_7_successes_l193_193664

theorem bernoulli_trial_probability_7_successes :
  let n := 7
  let k := 7
  let p := (2 : ℝ) / 7
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k)) = (128 / 823543) :=
by
  sorry

end bernoulli_trial_probability_7_successes_l193_193664


namespace pound_of_rice_cost_l193_193594

theorem pound_of_rice_cost 
(E R K : ℕ) (h1: E = R) (h2: K = 4 * (E / 12)) (h3: K = 11) : R = 33 := by
  sorry

end pound_of_rice_cost_l193_193594


namespace total_pens_l193_193420

theorem total_pens (r : ℕ) (h1 : r > 10)
  (h2 : 357 % r = 0)
  (h3 : 441 % r = 0) :
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_l193_193420


namespace central_angle_of_sector_l193_193522

theorem central_angle_of_sector (r l θ : ℝ) 
  (h1 : 2 * r + l = 8) 
  (h2 : (1 / 2) * l * r = 4) 
  (h3 : θ = l / r) : θ = 2 := 
sorry

end central_angle_of_sector_l193_193522


namespace divisor_of_condition_l193_193365

theorem divisor_of_condition {d z : ℤ} (h1 : ∃ k : ℤ, z = k * d + 6)
  (h2 : ∃ m : ℤ, (z + 3) = d * m) : d = 9 := 
sorry

end divisor_of_condition_l193_193365


namespace base_five_equals_base_b_l193_193031

theorem base_five_equals_base_b : ∃ (b : ℕ), b > 0 ∧ (2 * 5^1 + 4 * 5^0) = (1 * b^2 + 0 * b^1 + 1 * b^0) := by
  sorry

end base_five_equals_base_b_l193_193031


namespace xy_square_difference_l193_193765

variable (x y : ℚ)

theorem xy_square_difference (h1 : x + y = 8/15) (h2 : x - y = 1/45) : 
  x^2 - y^2 = 8/675 := by
  sorry

end xy_square_difference_l193_193765


namespace reflection_over_vector_l193_193328

noncomputable def reflection_matrix (u : ℝ → ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let u : Vector 2 := ![4, 1]
  (Matrix.dot_product u u⁺¹ / (Matrix.dot_product u u ^ 2)) 
  ∙ u⁺¹ 

theorem reflection_over_vector : reflection_matrix ![4, 1] = 
  ![![15 / 17, 8 / 17], ![8 / 17, -15 / 17]] :=
  sorry

end reflection_over_vector_l193_193328


namespace factorize_x_squared_minus_one_l193_193193

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l193_193193


namespace factorize_difference_of_squares_l193_193309

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l193_193309


namespace julian_owes_jenny_l193_193134

-- Define the initial debt and the additional borrowed amount
def initial_debt : ℕ := 20
def additional_borrowed : ℕ := 8

-- Define the total debt
def total_debt : ℕ := initial_debt + additional_borrowed

-- Statement of the problem: Prove that total_debt equals 28
theorem julian_owes_jenny : total_debt = 28 :=
by
  sorry

end julian_owes_jenny_l193_193134


namespace factorize_difference_of_squares_l193_193229

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := 
by 
  sorry

end factorize_difference_of_squares_l193_193229


namespace find_y_when_x_is_minus_2_l193_193803

theorem find_y_when_x_is_minus_2 :
  ∀ (x y t : ℝ), (x = 3 - 2 * t) → (y = 5 * t + 6) → (x = -2) → y = 37 / 2 :=
by
  intros x y t h1 h2 h3
  sorry

end find_y_when_x_is_minus_2_l193_193803


namespace unique_fraction_condition_l193_193547

theorem unique_fraction_condition :
  ∃! (x y : ℕ), x.gcd y = 1 ∧ y = x * 6 / 5 ∧ (1.2 * (x : ℚ) / y = (x + 1 : ℚ) / (y + 1)) := by
  sorry

end unique_fraction_condition_l193_193547


namespace edward_made_in_summer_l193_193022

theorem edward_made_in_summer
  (spring_earnings : ℤ)
  (spent_on_supplies : ℤ)
  (final_amount : ℤ)
  (S : ℤ)
  (h1 : spring_earnings = 2)
  (h2 : spent_on_supplies = 5)
  (h3 : final_amount = 24)
  (h4 : spring_earnings + S - spent_on_supplies = final_amount) :
  S = 27 := 
by
  sorry

end edward_made_in_summer_l193_193022


namespace smallest_b_factors_x2_bx_2016_l193_193338

theorem smallest_b_factors_x2_bx_2016 :
  ∃ (b : ℕ), (∀ (r s : ℤ), r * s = 2016 → r + s = b → b = 92) :=
begin
  sorry
end

end smallest_b_factors_x2_bx_2016_l193_193338


namespace magnitude_product_l193_193565

-- Definitions based on conditions
def z1 : Complex := ⟨7, -4⟩
def z2 : Complex := ⟨3, 10⟩

-- Statement of the theorem to be proved
theorem magnitude_product :
  Complex.abs (z1 * z2) = Real.sqrt 7085 := by
  sorry

end magnitude_product_l193_193565


namespace machine_worked_minutes_l193_193709

theorem machine_worked_minutes
  (shirts_today : ℕ)
  (rate : ℕ)
  (h1 : shirts_today = 8)
  (h2 : rate = 2) :
  (shirts_today / rate) = 4 :=
by
  sorry

end machine_worked_minutes_l193_193709


namespace total_pens_bought_l193_193397

theorem total_pens_bought (r : ℕ) (hr : r > 10) (hm : 357 % r = 0) (ho : 441 % r = 0) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_bought_l193_193397


namespace min_lightbulbs_for_5_working_l193_193823

noncomputable def minimal_lightbulbs_needed (p : ℝ) (k : ℕ) (threshold : ℝ) : ℕ :=
  Inf {n : ℕ | (∑ i in finset.range (n + 1), if i < k then 0 else nat.choose n i * p ^ i * (1 - p) ^ (n - i)) ≥ threshold}

theorem min_lightbulbs_for_5_working (p : ℝ) (k : ℕ) (threshold : ℝ) :
  p = 0.95 →
  k = 5 →
  threshold = 0.99 →
  minimal_lightbulbs_needed p k threshold = 7 := by
  intros
  sorry

end min_lightbulbs_for_5_working_l193_193823


namespace total_pens_l193_193443

theorem total_pens (r : ℕ) (r_gt_10 : r > 10) (r_div_357 : r ∣ 357) (r_div_441 : r ∣ 441) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_l193_193443


namespace intersection_of_S_and_T_l193_193583

open Set

def setS : Set ℝ := { x | (x-2)*(x+3) > 0 }
def setT : Set ℝ := { x | 3 - x ≥ 0 }

theorem intersection_of_S_and_T : setS ∩ setT = { x | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_of_S_and_T_l193_193583


namespace jugglers_count_l193_193646

-- Define the conditions
def num_balls_each_juggler := 6
def total_balls := 2268

-- Define the theorem to prove the number of jugglers
theorem jugglers_count : (total_balls / num_balls_each_juggler) = 378 :=
by
  sorry

end jugglers_count_l193_193646


namespace movie_of_the_year_condition_l193_193986

noncomputable def smallest_needed_lists : Nat :=
  let total_lists := 765
  let required_fraction := 1 / 4
  Nat.ceil (total_lists * required_fraction)

theorem movie_of_the_year_condition :
  smallest_needed_lists = 192 := by
  sorry

end movie_of_the_year_condition_l193_193986


namespace lcm_180_504_is_2520_l193_193490

-- Define what it means for a number to be the least common multiple of two numbers
def is_lcm (a b lcm : ℕ) : Prop :=
  a ∣ lcm ∧ b ∣ lcm ∧ ∀ m, (a ∣ m ∧ b ∣ m) → lcm ∣ m

-- Lean 4 statement to prove that the least common multiple of 180 and 504 is 2520
theorem lcm_180_504_is_2520 : ∀ (a b : ℕ), a = 180 → b = 504 → is_lcm a b 2520 := by
  intro a b
  assume h1 : a = 180
  assume h2 : b = 504
  sorry

end lcm_180_504_is_2520_l193_193490


namespace probability_at_least_one_admitted_l193_193164

-- Define the events and probabilities
variables (A B : Prop)
variables (P_A : ℝ) (P_B : ℝ)
variables (independent : Prop)

-- Assume the given conditions
def P_A_def : Prop := P_A = 0.6
def P_B_def : Prop := P_B = 0.7
def independent_def : Prop := independent = true  -- simplistic representation for independence

-- Statement: Prove the probability that at least one of them is admitted is 0.88
theorem probability_at_least_one_admitted : 
  P_A = 0.6 → P_B = 0.7 → independent = true →
  (1 - (1 - P_A) * (1 - P_B)) = 0.88 :=
by
  intros
  sorry

end probability_at_least_one_admitted_l193_193164


namespace factorization_of_x_squared_minus_one_l193_193242

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l193_193242


namespace translate_graph_upwards_l193_193486

theorem translate_graph_upwards (x : ℝ) :
  (∀ x, (3*x - 1) + 3 = 3*x + 2) :=
by
  intro x
  sorry

end translate_graph_upwards_l193_193486


namespace layla_more_than_nahima_l193_193786

-- Definitions for the conditions
def layla_points : ℕ := 70
def total_points : ℕ := 112
def nahima_points : ℕ := total_points - layla_points

-- Statement of the theorem
theorem layla_more_than_nahima : layla_points - nahima_points = 28 :=
by
  sorry

end layla_more_than_nahima_l193_193786


namespace percentage_decrease_in_selling_price_l193_193690

theorem percentage_decrease_in_selling_price (S M : ℝ) 
  (purchase_price : S = 240 + M)
  (markup_percentage : M = 0.25 * S)
  (gross_profit : S - 16 = 304) : 
  (320 - 304) / 320 * 100 = 5 := 
by
  sorry

end percentage_decrease_in_selling_price_l193_193690


namespace max_sum_abc_l193_193043

theorem max_sum_abc (a b c : ℝ) (h1 : 1 ≤ a) (h2 : 1 ≤ b) (h3 : 1 ≤ c) 
  (h4 : a * b * c + 2 * a^2 + 2 * b^2 + 2 * c^2 + c * a - c * b - 4 * a + 4 * b - c = 28) :
  a + b + c ≤ 6 :=
sorry

end max_sum_abc_l193_193043


namespace mrs_hilt_remaining_cents_l193_193948

-- Define the initial amount of money Mrs. Hilt had
def initial_cents : ℕ := 43

-- Define the cost of the pencil
def pencil_cost : ℕ := 20

-- Define the cost of the candy
def candy_cost : ℕ := 5

-- Define the remaining money Mrs. Hilt has after the purchases
def remaining_cents : ℕ := initial_cents - (pencil_cost + candy_cost)

-- Theorem statement to prove that the remaining amount is 18 cents
theorem mrs_hilt_remaining_cents : remaining_cents = 18 := by
  -- Proof omitted
  sorry

end mrs_hilt_remaining_cents_l193_193948


namespace james_total_pay_l193_193937

def original_prices : List ℝ := [15, 20, 25, 18, 22, 30]
def discounts : List ℝ := [0.30, 0.50, 0.40, 0.20, 0.45, 0.25]

def discounted_price (price discount : ℝ) : ℝ :=
  price * (1 - discount)

def total_price_after_discount (prices discounts : List ℝ) : ℝ :=
  (List.zipWith discounted_price prices discounts).sum

theorem james_total_pay :
  total_price_after_discount original_prices discounts = 84.50 :=
  by sorry

end james_total_pay_l193_193937


namespace apple_pies_l193_193936

theorem apple_pies (total_apples not_ripe_apples apples_per_pie : ℕ) 
    (h1 : total_apples = 34) 
    (h2 : not_ripe_apples = 6) 
    (h3 : apples_per_pie = 4) : 
    (total_apples - not_ripe_apples) / apples_per_pie = 7 :=
by 
    sorry

end apple_pies_l193_193936


namespace sum_x_coordinates_l193_193637

-- Define the equations of the line segments
def segment1 (x : ℝ) := 2 * x + 6
def segment2 (x : ℝ) := -0.5 * x - 1.5
def segment3 (x : ℝ) := 2 * x + 1
def segment4 (x : ℝ) := -0.5 * x + 3.5
def segment5 (x : ℝ) := 2 * x - 4

-- Definition of the problem
theorem sum_x_coordinates (h1 : segment1 (-5) = -4 ∧ segment1 (-3) = 0)
    (h2 : segment2 (-3) = 0 ∧ segment2 (-1) = -1)
    (h3 : segment3 (-1) = -1 ∧ segment3 (1) = 3)
    (h4 : segment4 (1) = 3 ∧ segment4 (3) = 2)
    (h5 : segment5 (3) = 2 ∧ segment5 (5) = 6)
    (hx1 : ∃ x1, segment3 x1 = 2.4 ∧ -1 ≤ x1 ∧ x1 ≤ 1)
    (hx2 : ∃ x2, segment4 x2 = 2.4 ∧ 1 ≤ x2 ∧ x2 ≤ 3)
    (hx3 : ∃ x3, segment5 x3 = 2.4 ∧ 3 ≤ x3 ∧ x3 ≤ 5) :
    (∃ (x1 x2 x3 : ℝ), segment3 x1 = 2.4 ∧ segment4 x2 = 2.4 ∧ segment5 x3 = 2.4 ∧ x1 = 0.7 ∧ x2 = 2.2 ∧ x3 = 3.2 ∧ x1 + x2 + x3 = 6.1) :=
sorry

end sum_x_coordinates_l193_193637


namespace union_of_sets_l193_193054

def setA := { x : ℝ | -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 3 }
def setB := { x : ℝ | (x - 2) / x ≤ 0 }

theorem union_of_sets :
  { x : ℝ | -1 ≤ x ∧ x ≤ 2 } = setA ∪ setB :=
by
  sorry

end union_of_sets_l193_193054


namespace total_pens_l193_193446

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193446


namespace advertising_department_employees_l193_193683

theorem advertising_department_employees (N S A_s x : ℕ) (hN : N = 1000) (hS : S = 80) (hA_s : A_s = 4) 
(h_stratified : x / N = A_s / S) : x = 50 :=
sorry

end advertising_department_employees_l193_193683


namespace johns_profit_l193_193604

/-- Define the number of ducks -/
def numberOfDucks : ℕ := 30

/-- Define the cost per duck -/
def costPerDuck : ℤ := 10

/-- Define the weight per duck -/
def weightPerDuck : ℤ := 4

/-- Define the selling price per pound -/
def pricePerPound : ℤ := 5

/-- Define the total cost to buy the ducks -/
def totalCost : ℤ := numberOfDucks * costPerDuck

/-- Define the selling price per duck -/
def sellingPricePerDuck : ℤ := weightPerDuck * pricePerPound

/-- Define the total revenue from selling all the ducks -/
def totalRevenue : ℤ := numberOfDucks * sellingPricePerDuck

/-- Define the profit John made -/
def profit : ℤ := totalRevenue - totalCost

/-- The theorem stating the profit John made given the conditions is $300 -/
theorem johns_profit : profit = 300 := by
  sorry

end johns_profit_l193_193604


namespace sample_size_stratified_sampling_l193_193700

theorem sample_size_stratified_sampling 
  (teachers : ℕ) (male_students : ℕ) (female_students : ℕ) 
  (n : ℕ) (females_drawn : ℕ) 
  (total_people : ℕ := teachers + male_students + female_students) 
  (females_total : ℕ := female_students) 
  (proportion_drawn : ℚ := (females_drawn : ℚ) / females_total) :
  teachers = 200 → 
  male_students = 1200 → 
  female_students = 1000 → 
  females_drawn = 80 → 
  proportion_drawn = ((n : ℚ) / total_people) → 
  n = 192 :=
by
  sorry

end sample_size_stratified_sampling_l193_193700


namespace range_of_f3_l193_193760

noncomputable def f (a b x : ℝ) : ℝ := a * x ^ 2 + b * x + 1

theorem range_of_f3 {a b : ℝ}
  (h1 : -2 ≤ a - b ∧ a - b ≤ 0) 
  (h2 : -3 ≤ 4 * a + 2 * b ∧ 4 * a + 2 * b ≤ 1) :
  -7 ≤ f a b 3 ∧ f a b 3 ≤ 3 :=
sorry

end range_of_f3_l193_193760


namespace remainder_when_ab_div_by_40_l193_193950

theorem remainder_when_ab_div_by_40 (a b : ℤ) (k j : ℤ)
  (ha : a = 80 * k + 75)
  (hb : b = 90 * j + 85):
  (a + b) % 40 = 0 :=
by sorry

end remainder_when_ab_div_by_40_l193_193950


namespace max_servings_l193_193998

def servings_prepared (peppers brynza tomatoes cucumbers : ℕ) : ℕ :=
  min (peppers)
      (min (brynza / 75)
           (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings :
  servings_prepared 60 4200 116 117 = 56 :=
by sorry

end max_servings_l193_193998


namespace triangle_properties_l193_193372

noncomputable def a : ℝ := (-1 + real.sqrt (1 - 4)) / 2 
noncomputable def b : ℝ := (-1 - real.sqrt (1 - 4)) / 2 

theorem triangle_properties :
  ∃ (C : ℝ) (c : ℝ) (area : ℝ),
  (C = 60) ∧
  (c = real.sqrt 6) ∧
  (area = (1 / 2) * a * b * (real.sin (π / 3))) :=
by
  -- Define angle C as 60 degrees in radians
  let C : ℝ := π / 3
  -- Define the length of side c
  let c : ℝ := real.sqrt 6
  -- Define the area of the triangle
  let area : ℝ := (1 / 2) * a * b * (real.sin C)
  -- Provide the required proof
  use [C, c, area]
  split; sorry

end triangle_properties_l193_193372


namespace inconsistent_linear_system_l193_193104

theorem inconsistent_linear_system :
  ¬ ∃ (x1 x2 x3 : ℝ), 
    (2 * x1 + 5 * x2 - 4 * x3 = 8) ∧
    (3 * x1 + 15 * x2 - 9 * x3 = 5) ∧
    (5 * x1 + 5 * x2 - 7 * x3 = 1) :=
by
  -- Proof of inconsistency
  sorry

end inconsistent_linear_system_l193_193104


namespace smallest_b_for_factorization_l193_193333

theorem smallest_b_for_factorization :
  ∃ b : ℕ, (∀ r s : ℤ, r * s = 2016 → r + s = b) ∧ b = 90 :=
sorry

end smallest_b_for_factorization_l193_193333


namespace range_of_m_l193_193755

def f (x : ℝ) := |x - 3|
def g (x : ℝ) (m : ℝ) := -|x - 7| + m

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x ≥ g x m) → m < 4 :=
by
  sorry

end range_of_m_l193_193755


namespace binary_addition_and_subtraction_correct_l193_193012

def add_binary_and_subtract : ℕ :=
  let n1 := 0b1101  -- binary for 1101_2
  let n2 := 0b0010  -- binary for 10_2
  let n3 := 0b0101  -- binary for 101_2
  let n4 := 0b1011  -- expected result 1011_2
  n1 + n2 + n3 - 0b0011  -- subtract binary for 11_2

theorem binary_addition_and_subtraction_correct : add_binary_and_subtract = 0b1011 := 
by 
  sorry

end binary_addition_and_subtraction_correct_l193_193012


namespace factorize_x_squared_minus_one_l193_193319

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l193_193319


namespace total_pens_l193_193436

theorem total_pens (r : ℕ) (r_gt_10 : r > 10) (r_div_357 : r ∣ 357) (r_div_441 : r ∣ 441) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_l193_193436


namespace find_x_values_l193_193044

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem find_x_values :
  { x : ℕ | combination 10 x = combination 10 (3 * x - 2) } = {1, 3} :=
by
  sorry

end find_x_values_l193_193044


namespace factor_difference_of_squares_l193_193262

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l193_193262


namespace unsold_percentage_l193_193947

def total_harvested : ℝ := 340.2
def sold_mm : ℝ := 125.5  -- Weight sold to Mrs. Maxwell
def sold_mw : ℝ := 78.25  -- Weight sold to Mr. Wilson
def sold_mb : ℝ := 43.8   -- Weight sold to Ms. Brown
def sold_mj : ℝ := 56.65  -- Weight sold to Mr. Johnson

noncomputable def percentage_unsold (total_harvested : ℝ) 
                                   (sold_mm : ℝ) 
                                   (sold_mw : ℝ)
                                   (sold_mb : ℝ) 
                                   (sold_mj : ℝ) : ℝ :=
  let total_sold := sold_mm + sold_mw + sold_mb + sold_mj
  let unsold := total_harvested - total_sold
  (unsold / total_harvested) * 100

theorem unsold_percentage : percentage_unsold total_harvested sold_mm sold_mw sold_mb sold_mj = 10.58 :=
by
  sorry

end unsold_percentage_l193_193947


namespace solve_for_x_l193_193101

-- Define the given condition
def condition (x : ℝ) : Prop := (x - 5) ^ 3 = -((1 / 27)⁻¹)

-- State the problem as a Lean theorem
theorem solve_for_x : ∃ x : ℝ, condition x ∧ x = 2 := by
  sorry

end solve_for_x_l193_193101


namespace minimum_cost_l193_193517

noncomputable def f (x : ℝ) : ℝ := (1000 / (x + 5)) + 5 * x + (1 / 2) * (x^2 + 25)

theorem minimum_cost :
  (2 ≤ x ∧ x ≤ 8) →
  (f 5 = 150 ∧ (∀ y, 2 ≤ y ∧ y ≤ 8 → f y ≥ f 5)) :=
by
  intro h
  have f_exp : f x = (1000 / (x+5)) + 5*x + (1/2)*(x^2 + 25) := rfl
  sorry

end minimum_cost_l193_193517


namespace range_of_x_l193_193590

-- Define the condition: the expression under the square root must be non-negative
def condition (x : ℝ) : Prop := 3 + x ≥ 0

-- Define what we want to prove: the range of x such that the condition holds
theorem range_of_x (x : ℝ) : condition x ↔ x ≥ -3 :=
by
  -- Proof goes here
  sorry

end range_of_x_l193_193590


namespace find_subtracted_number_l193_193848

variable (initial_number : Real)
variable (sum : Real := initial_number + 5)
variable (product : Real := sum * 7)
variable (quotient : Real := product / 5)
variable (remainder : Real := 33)

theorem find_subtracted_number 
  (initial_number_eq : initial_number = 22.142857142857142)
  : quotient - remainder = 5 := by
  sorry

end find_subtracted_number_l193_193848


namespace solve_inequality_l193_193627

theorem solve_inequality (x : ℝ) :
  ((x - 1) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0 ↔ 
  x ∈ set.Ioo (x) (-∞, 1) ∪ (1, 2) ∪ (2, 4) ∪ (4, 5) ∪ (7, ∞) :=
by sorry

end solve_inequality_l193_193627


namespace factorization_difference_of_squares_l193_193207

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l193_193207


namespace net_price_change_is_twelve_percent_l193_193369

variable (P : ℝ)

def net_price_change (P : ℝ) : ℝ := 
  let decreased_price := 0.8 * P
  let increased_price := 1.4 * decreased_price
  increased_price - P

theorem net_price_change_is_twelve_percent (P : ℝ) : net_price_change P = 0.12 * P := by
  sorry

end net_price_change_is_twelve_percent_l193_193369


namespace heracles_age_is_10_l193_193713

variable (H : ℕ)

-- Conditions
def audrey_age_now : ℕ := H + 7
def audrey_age_in_3_years : ℕ := audrey_age_now + 3
def heracles_twice_age : ℕ := 2 * H

-- Proof Statement
theorem heracles_age_is_10 (h1 : audrey_age_in_3_years = heracles_twice_age) : H = 10 :=
by 
  sorry

end heracles_age_is_10_l193_193713


namespace min_value_sin_cos_l193_193892

theorem min_value_sin_cos (x : ℝ) : sin x ^ 6 + 2 * cos x ^ 6 ≥ 2 / 3 :=
sorry

end min_value_sin_cos_l193_193892


namespace tom_read_chapters_l193_193122

theorem tom_read_chapters (chapters pages: ℕ) (h1: pages = 8 * chapters) (h2: pages = 24):
  chapters = 3 :=
by
  sorry

end tom_read_chapters_l193_193122


namespace find_x_l193_193494

theorem find_x (x m n : ℤ) 
  (h₁ : 15 + x = m^2) 
  (h₂ : x - 74 = n^2) :
  x = 2010 :=
by
  sorry

end find_x_l193_193494


namespace total_pens_l193_193407

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193407


namespace tate_total_years_proof_l193_193806

def highSchoolYears: ℕ := 4 - 1
def gapYear: ℕ := 2
def bachelorYears (highSchoolYears: ℕ): ℕ := 2 * highSchoolYears
def workExperience: ℕ := 1
def phdYears (highSchoolYears: ℕ) (bachelorYears: ℕ): ℕ := 3 * (highSchoolYears + bachelorYears)
def totalYears (highSchoolYears: ℕ) (gapYear: ℕ) (bachelorYears: ℕ) (workExperience: ℕ) (phdYears: ℕ): ℕ :=
  highSchoolYears + gapYear + bachelorYears + workExperience + phdYears

theorem tate_total_years_proof : totalYears highSchoolYears gapYear (bachelorYears highSchoolYears) workExperience (phdYears highSchoolYears (bachelorYears highSchoolYears)) = 39 := by
  sorry

end tate_total_years_proof_l193_193806


namespace Giovanni_burgers_l193_193754

theorem Giovanni_burgers : 
  let toppings := 10
  let patty_choices := 4
  let topping_combinations := 2 ^ toppings
  let total_combinations := patty_choices * topping_combinations
  total_combinations = 4096 :=
by
  sorry

end Giovanni_burgers_l193_193754


namespace light_bulbs_circle_l193_193949

theorem light_bulbs_circle : ∀ (f : ℕ → ℕ),
  (f 0 = 1) ∧
  (f 1 = 2) ∧
  (f 2 = 4) ∧
  (f 3 = 8) ∧
  (∀ n, f n = f (n - 1) + f (n - 2) + f (n - 3) + f (n - 4)) →
  (f 9 - 3 * f 3 - 2 * f 2 - f 1 = 367) :=
by
  sorry

end light_bulbs_circle_l193_193949


namespace roots_cubed_l193_193612

noncomputable def q (b c : ℝ) (x : ℝ) : ℝ := x^2 - 2 * b * x + b^2 - c^2
noncomputable def p (b c : ℝ) (x : ℝ) : ℝ := x^2 - 2 * b * (b^2 + 3 * c^2) * x + (b^2 - c^2)^3 
def x1 (b c : ℝ) := b + c
def x2 (b c : ℝ) := b - c

theorem roots_cubed (b c : ℝ) :
  (q b c (x1 b c) = 0 ∧ q b c (x2 b c) = 0) →
  (p b c ((x1 b c)^3) = 0 ∧ p b c ((x2 b c)^3) = 0) :=
by
  sorry

end roots_cubed_l193_193612


namespace find_lambda_l193_193346

open Real

theorem find_lambda
  (λ : ℝ)
  (a : ℝ × ℝ × ℝ := (2, -1, 3))
  (b : ℝ × ℝ × ℝ := (-1, 4, -2))
  (c : ℝ × ℝ × ℝ := (7, 7, λ))
  (coplanar : ∃ m n : ℝ, c = (m * 2 - n * 1, m * -1 + n * 4, m * 3 - n * 2)) :
  λ = 9 := 
by sorry

end find_lambda_l193_193346


namespace amount_brought_by_sisters_l193_193125

-- Definitions based on conditions
def cost_per_ticket : ℕ := 8
def number_of_tickets : ℕ := 2
def change_received : ℕ := 9

-- Statement to prove
theorem amount_brought_by_sisters :
  (cost_per_ticket * number_of_tickets + change_received) = 25 :=
by
  -- Using assumptions directly
  let total_cost := cost_per_ticket * number_of_tickets
  have total_cost_eq : total_cost = 16 := by sorry
  let amount_brought := total_cost + change_received
  have amount_brought_eq : amount_brought = 25 := by sorry
  exact amount_brought_eq

end amount_brought_by_sisters_l193_193125


namespace initial_average_marks_l193_193473

theorem initial_average_marks (A : ℝ) (h1 : 25 * A - 50 = 2450) : A = 100 :=
by
  sorry

end initial_average_marks_l193_193473


namespace treaty_signed_on_tuesday_l193_193968

-- Define a constant for the start date and the number of days
def start_day_of_week : ℕ := 1 -- Monday is represented by 1
def days_until_treaty : ℕ := 1301

-- Function to calculate the resulting day of the week
def day_of_week_after_days (start_day : ℕ) (days : ℕ) : ℕ :=
  (start_day + days) % 7

-- Theorem statement: Prove that 1301 days after Monday is Tuesday
theorem treaty_signed_on_tuesday :
  day_of_week_after_days start_day_of_week days_until_treaty = 2 :=
by
  -- placeholder for the proof
  sorry

end treaty_signed_on_tuesday_l193_193968


namespace possible_days_l193_193532

namespace AnyaVanyaProblem

-- Conditions
def AnyaLiesOn (d : String) : Prop := d = "Tuesday" ∨ d = "Wednesday" ∨ d = "Thursday"
def AnyaTellsTruthOn (d : String) : Prop := ¬AnyaLiesOn d

def VanyaLiesOn (d : String) : Prop := d = "Thursday" ∨ d = "Friday" ∨ d = "Saturday"
def VanyaTellsTruthOn (d : String) : Prop := ¬VanyaLiesOn d

-- Statements
def AnyaStatement (d : String) : Prop := d = "Friday"
def VanyaStatement (d : String) : Prop := d = "Tuesday"

-- Proof problem
theorem possible_days (d : String) : 
  (AnyaTellsTruthOn d ↔ AnyaStatement d) ∧ (VanyaTellsTruthOn d ↔ VanyaStatement d)
  → d = "Tuesday" ∨ d = "Thursday" ∨ d = "Friday" := 
sorry

end AnyaVanyaProblem

end possible_days_l193_193532


namespace sum_of_solutions_eq_eight_l193_193339

theorem sum_of_solutions_eq_eight : 
  ∀ x : ℝ, (x^2 - 6 * x + 5 = 2 * x - 7) → (∃ a b : ℝ, (a = 6) ∧ (b = 2) ∧ (a + b = 8)) :=
by
  sorry

end sum_of_solutions_eq_eight_l193_193339


namespace geometric_sequence_four_seven_prod_l193_193376

def is_geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_four_seven_prod
    (a : ℕ → ℝ)
    (h_geom : is_geometric_sequence a)
    (h_roots : ∀ x, 3 * x^2 - 2 * x - 6 = 0 → (x = a 1 ∨ x = a 10)) :
  a 4 * a 7 = -2 := 
sorry

end geometric_sequence_four_seven_prod_l193_193376


namespace correct_calculation_l193_193497

theorem correct_calculation (x a b : ℝ) : 
  (x^4 * x^4 = x^8) ∧ ((a^3)^2 = a^6) ∧ ((a * (b^2))^3 = a^3 * b^6) → (a + 2*a = 3*a) := 
by 
  sorry

end correct_calculation_l193_193497


namespace find_tangent_point_l193_193030

theorem find_tangent_point (x : ℝ) (y : ℝ) (h_curve : y = x^2) (h_slope : 2 * x = 1) : 
    (x, y) = (1/2, 1/4) :=
sorry

end find_tangent_point_l193_193030


namespace smallest_k_l193_193795

-- Define p as the largest prime number with 2023 digits
def p : ℕ := sorry -- This represents the largest prime number with 2023 digits

-- Define the target k
def k : ℕ := 1

-- The theorem stating that k is the smallest positive integer such that p^2 - k is divisible by 30
theorem smallest_k (p_largest_prime : ∀ m : ℕ, m ≤ p → Nat.Prime m → m = p) 
  (p_digits : 10^2022 ≤ p ∧ p < 10^2023) : 
  ∀ n : ℕ, n > 0 → (p^2 - n) % 30 = 0 → n = k :=
by 
  sorry

end smallest_k_l193_193795


namespace cube_side_length_l193_193857

theorem cube_side_length (n : ℕ) (h1 : 6 * n^2 = (6 * n^3) / 3) : n = 3 :=
sorry

end cube_side_length_l193_193857


namespace factorize_difference_of_squares_l193_193300

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l193_193300


namespace sqrt_two_irrational_l193_193837

theorem sqrt_two_irrational : ¬ ∃ (a b : ℕ), b ≠ 0 ∧ (a / b) ^ 2 = 2 :=
by
  sorry

end sqrt_two_irrational_l193_193837


namespace bernoulli_trial_probability_7_successes_l193_193665

theorem bernoulli_trial_probability_7_successes :
  let n := 7
  let k := 7
  let p := (2 : ℝ) / 7
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k)) = (128 / 823543) :=
by
  sorry

end bernoulli_trial_probability_7_successes_l193_193665


namespace factorization_of_difference_of_squares_l193_193258

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l193_193258


namespace elvin_fixed_monthly_charge_l193_193879

theorem elvin_fixed_monthly_charge
  (F C : ℝ) 
  (h1 : F + C = 50) 
  (h2 : F + 2 * C = 76) : 
  F = 24 := 
sorry

end elvin_fixed_monthly_charge_l193_193879


namespace gcd_20586_58768_l193_193886

theorem gcd_20586_58768 : Int.gcd 20586 58768 = 2 := by
  sorry

end gcd_20586_58768_l193_193886


namespace factorize_x_squared_minus_one_l193_193318

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l193_193318


namespace factorize_difference_of_squares_l193_193305

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l193_193305


namespace value_of_a_l193_193370

theorem value_of_a {a : ℝ} : 
  (∃ x : ℝ, (a - 1) * x^2 + 4 * x - 2 = 0 ∧ ∀ y : ℝ, (a - 1) * y^2 + 4 * y - 2 ≠ 0 → y = x) → 
  (a = 1 ∨ a = -1) :=
by 
  sorry

end value_of_a_l193_193370


namespace unique_fraction_property_l193_193554

def fractions_property (x y : ℕ) : Prop :=
  Nat.coprime x y ∧ (x + 1) * 5 * y = (y + 1) * 6 * x

theorem unique_fraction_property :
  {x : ℕ // {y : ℕ // fractions_property x y}} = {⟨5, ⟨6, fractions_property 5 6⟩⟩} :=
by
  sorry

end unique_fraction_property_l193_193554


namespace average_a_b_l193_193477

theorem average_a_b (a b : ℝ) (h : (4 + 6 + 8 + a + b) / 5 = 20) : (a + b) / 2 = 41 :=
by
  sorry

end average_a_b_l193_193477


namespace factorize_x_squared_minus_one_l193_193186

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l193_193186


namespace factor_difference_of_squares_l193_193260

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l193_193260


namespace factorize_difference_of_squares_l193_193303

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l193_193303


namespace set_intersection_set_union_set_complement_l193_193082

open Set

variable (U : Set ℝ) (A B : Set ℝ)
noncomputable def setA : Set ℝ := {x | x^2 - 3*x - 4 ≥ 0}
noncomputable def setB : Set ℝ := {x | x < 5}

theorem set_intersection : (U = univ) -> (A = setA) -> (B = setB) -> A ∩ B = Ico 4 5 := by
  intros
  sorry

theorem set_union : (U = univ) -> (A = setA) -> (B = setB) -> A ∪ B = univ := by
  intros
  sorry

theorem set_complement : (U = univ) -> (A = setA) -> U \ A = Ioo (-1 : ℝ) 4 := by
  intros
  sorry

end set_intersection_set_union_set_complement_l193_193082


namespace apples_in_basket_l193_193133

noncomputable def total_apples (good_cond: ℕ) (good_ratio: ℝ) := (good_cond : ℝ) / good_ratio

theorem apples_in_basket : total_apples 66 0.88 = 75 :=
by
  sorry

end apples_in_basket_l193_193133


namespace area_of_square_ABCD_l193_193481

theorem area_of_square_ABCD :
  (∃ (x y : ℝ), 2 * x + 2 * y = 40) →
  ∃ (s : ℝ), s = 20 ∧ s * s = 400 :=
by
  sorry

end area_of_square_ABCD_l193_193481


namespace factorize_x_squared_minus_one_l193_193182

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l193_193182


namespace factorize_difference_of_squares_l193_193310

theorem factorize_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) := sorry

end factorize_difference_of_squares_l193_193310


namespace find_fraction_value_l193_193039

noncomputable section

open Real

theorem find_fraction_value (α : ℝ) (h : sin (α / 2) - 2 * cos (α / 2) = 1) :
  (1 + sin α + cos α) / (1 + sin α - cos α) = 1 :=
sorry

end find_fraction_value_l193_193039


namespace factorization_difference_of_squares_l193_193210

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l193_193210


namespace incorrect_statement_l193_193342

noncomputable def function_y (x : ℝ) : ℝ := 4 / x

theorem incorrect_statement (x : ℝ) (hx : x ≠ 0) : ¬(∀ x1 x2 : ℝ, (hx1 : x1 ≠ 0) → (hx2 : x2 ≠ 0) → x1 < x2 → function_y x1 > function_y x2) := 
sorry

end incorrect_statement_l193_193342


namespace total_pens_bought_l193_193456

theorem total_pens_bought (r : ℕ) (r_gt_10 : r > 10) (r_divides_357 : 357 % r = 0) (r_divides_441 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
by sorry

end total_pens_bought_l193_193456


namespace deer_meat_distribution_l193_193779

theorem deer_meat_distribution :
  ∃ (a1 a2 a3 a4 a5 : ℕ), a1 > a2 ∧ a2 > a3 ∧ a3 > a4 ∧ a4 > a5 ∧
  (a1 + a2 + a3 + a4 + a5 = 500) ∧
  (a2 + a3 + a4 = 300) :=
sorry

end deer_meat_distribution_l193_193779


namespace number_line_move_l193_193063

theorem number_line_move (A B: ℤ):  A = -3 → B = A + 4 → B = 1 := by
  intros hA hB
  rw [hA] at hB
  rw [hB]
  sorry

end number_line_move_l193_193063


namespace factorize_x_squared_minus_one_l193_193185

theorem factorize_x_squared_minus_one : ∀ (x : ℝ), x^2 - 1 = (x + 1) * (x - 1) :=
by
  intro x
  calc
    x^2 - 1 = (x + 1) * (x - 1) : sorry

end factorize_x_squared_minus_one_l193_193185


namespace total_pens_l193_193426

theorem total_pens (r : ℕ) (h1 : r > 10)
  (h2 : 357 % r = 0)
  (h3 : 441 % r = 0) :
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_l193_193426


namespace sqrt_9_eq_3_or_neg3_l193_193828

theorem sqrt_9_eq_3_or_neg3 :
  { x : ℝ | x^2 = 9 } = {3, -3} :=
sorry

end sqrt_9_eq_3_or_neg3_l193_193828


namespace trig_expression_correct_l193_193720

noncomputable def trig_expression_value : ℝ := 
  Real.cos (42 * Real.pi / 180) * Real.cos (78 * Real.pi / 180) + 
  Real.sin (42 * Real.pi / 180) * Real.cos (168 * Real.pi / 180)

theorem trig_expression_correct : trig_expression_value = -1 / 2 :=
by 
  sorry

end trig_expression_correct_l193_193720


namespace find_k_l193_193934

variable (m n k : ℝ)

def line (x y : ℝ) : Prop := x = 2 * y + 3
def point1_on_line : Prop := line m n
def point2_on_line : Prop := line (m + 2) (n + k)

theorem find_k (h1 : point1_on_line m n) (h2 : point2_on_line m n k) : k = 0 :=
by
  sorry

end find_k_l193_193934


namespace sum_slope_y_intercept_eq_l193_193598

noncomputable def J : ℝ × ℝ := (0, 8)
noncomputable def K : ℝ × ℝ := (0, 0)
noncomputable def L : ℝ × ℝ := (10, 0)
noncomputable def G : ℝ × ℝ := ((J.1 + K.1) / 2, (J.2 + K.2) / 2)

theorem sum_slope_y_intercept_eq :
  let L := (10, 0)
  let G := (0, 4)
  let slope := (G.2 - L.2) / (G.1 - L.1)
  let y_intercept := G.2
  slope + y_intercept = 18 / 5 :=
by
  -- Place the conditions and setup here
  let L := (10, 0)
  let G := (0, 4)
  let slope := (G.2 - L.2) / (G.1 - L.1)
  let y_intercept := G.2
  -- Proof will be provided here eventually
  sorry

end sum_slope_y_intercept_eq_l193_193598


namespace total_pens_l193_193404

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193404


namespace beetle_number_of_routes_128_l193_193148

noncomputable def beetle_routes (A B : Type) : Nat :=
  let choices_at_first_step := 4
  let choices_at_second_step := 4
  let choices_at_third_step := 4
  let choices_at_final_step := 2
  choices_at_first_step * choices_at_second_step * choices_at_third_step * choices_at_final_step

theorem beetle_number_of_routes_128 (A B : Type) :
  beetle_routes A B = 128 :=
  by sorry

end beetle_number_of_routes_128_l193_193148


namespace division_value_l193_193146

theorem division_value (x : ℝ) (h : 800 / x - 154 = 6) : x = 5 := by
  sorry

end division_value_l193_193146


namespace sum_series_eq_l193_193896

theorem sum_series_eq : 
  ∑' n : ℕ, (n + 1) * (1 / 3 : ℝ)^n = 9 / 4 :=
by sorry

end sum_series_eq_l193_193896


namespace triangle_obtuse_of_inequality_l193_193781

theorem triangle_obtuse_of_inequality
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)
  (ineq : a^2 < (b + c) * (c - b)) :
  ∃ (A B C : ℝ), (A + B + C = π) ∧ (C > π / 2) :=
by
  sorry

end triangle_obtuse_of_inequality_l193_193781


namespace fresh_grapes_weight_l193_193344

theorem fresh_grapes_weight (F D : ℝ) (h1 : D = 0.625) (h2 : 0.10 * F = 0.80 * D) : F = 5 := by
  -- Using premises h1 and h2, we aim to prove that F = 5
  sorry

end fresh_grapes_weight_l193_193344


namespace num_valid_k_l193_193327

open Nat

-- Pre-assumptions for the problem
def real_numbers_on_circle := 2022  -- Number of real numbers arranged on a circle

-- Lean functional definition to calculate the needed count of k
def countValidK (n : ℕ) : ℕ := (Nat.totient 2022)

-- Lean 4 Statement
theorem num_valid_k : ∀ (k : ℕ), (1 ≤ k ∧ k ≤ 2022) → (gcd k 2022 = 1) :=
begin
  sorry,
end

example : countValidK 2022 = 672 :=
begin
  rw countValidK,
  simp,
  sorry,
end

end num_valid_k_l193_193327


namespace quadrilateral_possible_rods_l193_193938

theorem quadrilateral_possible_rods (rods : Finset ℕ) (a b c : ℕ) (ha : a = 3) (hb : b = 7) (hc : c = 15)
  (hrods : rods = (Finset.range 31 \ {3, 7, 15})) :
  ∃ d, d ∈ rods ∧ 5 < d ∧ d < 25 ∧ rods.card - 2 = 17 := 
by
  sorry

end quadrilateral_possible_rods_l193_193938


namespace geometric_sequence_log_sum_l193_193034

theorem geometric_sequence_log_sum (a : ℕ → ℝ) (r : ℝ)
  (h_geo : ∀ n, a (n + 1) = a n * r)
  (h_pos : ∀ n, a n > 0)
  (h_condition : a 5 * a 6 + a 4 * a 7 = 18) :
  log 3 (a 1) + log 3 (a 2) + log 3 (a 3) + log 3 (a 4) +
  log 3 (a 5) + log 3 (a 6) + log 3 (a 7) + log 3 (a 8) +
  log 3 (a 9) + log 3 (a 10) = 10 :=
by 
  sorry

end geometric_sequence_log_sum_l193_193034


namespace cylinder_volume_triple_radius_quadruple_height_l193_193479

open Real

theorem cylinder_volume_triple_radius_quadruple_height (r h : ℝ) (V : ℝ) (hV : V = π * r^2 * h) :
  (3 * r) ^ 2 * 4 * h * π = 360 :=
by
  sorry

end cylinder_volume_triple_radius_quadruple_height_l193_193479


namespace total_pens_l193_193406

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193406


namespace origami_papers_total_l193_193838

-- Define the conditions as Lean definitions
def num_cousins : ℕ := 6
def papers_per_cousin : ℕ := 8

-- Define the total number of origami papers that Haley has to give away
def total_papers : ℕ := num_cousins * papers_per_cousin

-- Statement of the proof
theorem origami_papers_total : total_papers = 48 :=
by
  -- Skipping the proof for now
  sorry

end origami_papers_total_l193_193838


namespace factor_difference_of_squares_l193_193268

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l193_193268


namespace cakes_difference_l193_193015

-- Definitions of the given conditions
def cakes_sold : ℕ := 78
def cakes_bought : ℕ := 31

-- The theorem to prove
theorem cakes_difference : cakes_sold - cakes_bought = 47 :=
by sorry

end cakes_difference_l193_193015


namespace fraction_of_buttons_l193_193086

variable (K S M : ℕ)  -- Kendra's buttons, Sue's buttons, Mari's buttons

theorem fraction_of_buttons (H1 : M = 5 * K + 4) 
                            (H2 : S = 6)
                            (H3 : M = 64) :
  S / K = 1 / 2 := by
  sorry

end fraction_of_buttons_l193_193086


namespace matrix_satisfies_conditions_l193_193166

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

def matrix : List (List ℕ) :=
  [[6, 8, 9], [1, 7, 3], [4, 2, 5]]

noncomputable def sum_list (lst : List ℕ) : ℕ :=
  lst.foldl (· + ·) 0

def valid_matrix (matrix : List (List ℕ)) : Prop :=
  ∀ row_sum col_sum : ℕ, 
    (row_sum ∈ (matrix.map sum_list) ∧ is_prime row_sum) ∧
    (col_sum ∈ (List.transpose matrix).map sum_list ∧ is_prime col_sum)

theorem matrix_satisfies_conditions : valid_matrix matrix :=
by
  sorry

end matrix_satisfies_conditions_l193_193166


namespace probability_A_and_B_l193_193839

def is_fair_die := ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 6 → 1/6

def outcomes : Finset (ℕ × ℕ) := (Finset.range 6).product (Finset.range 6)

def event_A : Finset (ℕ × ℕ) := outcomes.filter (λ (pair : ℕ × ℕ), pair.1 ≠ pair.2)

def event_B : Finset (ℕ × ℕ) := outcomes.filter (λ (pair : ℕ × ℕ), pair.1 = 3 ∨ pair.2 = 3)

def event_A_and_B : Finset (ℕ × ℕ) := event_A ∩ event_B

theorem probability_A_and_B:
  (event_A_and_B.card * 1 : ℚ) / outcomes.card = 5 / 18 :=
by
  -- placeholder for the actual proof
  sorry

end probability_A_and_B_l193_193839


namespace factorize_x_squared_minus_one_l193_193196

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l193_193196


namespace tan_add_pi_over_six_l193_193366

theorem tan_add_pi_over_six (x : ℝ) (h : Real.tan x = 3) :
  Real.tan (x + Real.pi / 6) = 5 + 2 * Real.sqrt 3 :=
sorry

end tan_add_pi_over_six_l193_193366


namespace complete_square_transform_l193_193529

theorem complete_square_transform (x : ℝ) :
  x^2 + 6*x + 5 = 0 ↔ (x + 3)^2 = 4 := 
sorry

end complete_square_transform_l193_193529


namespace total_pens_l193_193419

/-- Proof that Masha and Olya bought a total of 38 pens given the cost conditions. -/
theorem total_pens (r : ℕ) (h_r : r > 10) (h1 : 357 % r = 0) (h2 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193419


namespace a4_equals_8_l193_193780

variable {a : ℕ → ℝ}
variable {r : ℝ}
variable {n : ℕ}

-- Defining the geometric sequence
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) := ∀ n, a (n + 1) = a n * r

-- Given conditions as hypotheses
variable (h_geometric : geometric_sequence a r)
variable (h_root_2 : a 2 * a 6 = 64)
variable (h_roots_eq : ∀ x, x^2 - 34 * x + 64 = 0 → (x = a 2 ∨ x = a 6))

-- The statement to prove
theorem a4_equals_8 : a 4 = 8 :=
by
  sorry

end a4_equals_8_l193_193780


namespace single_elimination_games_l193_193007

theorem single_elimination_games (n : ℕ) (h : n = 24) :
  let games_played := n - 1 in
  games_played = 23 :=
by
  have h2 := h ▸ rfl
  rw [h2]
  sorry

end single_elimination_games_l193_193007


namespace probability_of_7_successes_l193_193673

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_of_successes (n k : ℕ) (p : ℚ) : ℚ :=
  binomial_coefficient n k * p^k * (1 - p)^(n - k)

theorem probability_of_7_successes :
  probability_of_successes 7 7 (2/7) = 128 / 823543 :=
by
  sorry

end probability_of_7_successes_l193_193673


namespace h_h_three_l193_193922

def h (x : ℤ) : ℤ := 3 * x^2 + 3 * x - 2

theorem h_h_three : h (h 3) = 3568 := by
  sorry

end h_h_three_l193_193922


namespace bowls_total_marbles_l193_193123

theorem bowls_total_marbles :
  let C2 := 600
  let C1 := (3 / 4 : ℝ) * C2
  let C3 := (1 / 2 : ℝ) * C1
  C1 = 450 ∧ C3 = 225 ∧ (C1 + C2 + C3 = 1275) := 
by
  let C2 := 600
  let C1 := (3 / 4 : ℝ) * C2
  let C3 := (1 / 2 : ℝ) * C1
  have hC1 : C1 = 450 := by norm_num
  have hC3 : C3 = 225 := by norm_num
  have hTotal : C1 + C2 + C3 = 1275 := by norm_num
  exact ⟨hC1, hC3, hTotal⟩

end bowls_total_marbles_l193_193123


namespace max_servings_l193_193997

def servings_prepared (peppers brynza tomatoes cucumbers : ℕ) : ℕ :=
  min (peppers)
      (min (brynza / 75)
           (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings :
  servings_prepared 60 4200 116 117 = 56 :=
by sorry

end max_servings_l193_193997


namespace restaurant_sales_decrease_l193_193738

-- Conditions
variable (Sales_August : ℝ := 42000)
variable (Sales_October : ℝ := 27000)
variable (a : ℝ) -- monthly average decrease rate as a decimal

-- Theorem statement
theorem restaurant_sales_decrease :
  42 * (1 - a)^2 = 27 := sorry

end restaurant_sales_decrease_l193_193738


namespace smallest_n_divisibility_l193_193493

theorem smallest_n_divisibility:
  ∃ (n : ℕ), n > 0 ∧ n^2 % 24 = 0 ∧ n^3 % 540 = 0 ∧ n = 60 :=
by
  sorry

end smallest_n_divisibility_l193_193493


namespace algebraic_inequality_l193_193827

noncomputable def problem_statement (a b c d : ℝ) : Prop :=
  |a| > 1 ∧ |b| > 1 ∧ |c| > 1 ∧ |d| > 1 ∧
  a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0 →
  (1 / (a - 1)) + (1 / (b - 1)) + (1 / (c - 1)) + (1 / (d - 1)) > 0

theorem algebraic_inequality (a b c d : ℝ) :
  problem_statement a b c d :=
by
  sorry

end algebraic_inequality_l193_193827


namespace determine_a_l193_193912

theorem determine_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (M m : ℝ)
  (hM : M = max (a^1) (a^2))
  (hm : m = min (a^1) (a^2))
  (hM_m : M = 2 * m) :
  a = 1/2 ∨ a = 2 := 
by sorry

end determine_a_l193_193912


namespace max_servings_l193_193995

theorem max_servings :
  let cucumbers := 117,
      tomatoes := 116,
      bryndza := 4200,  -- converted to grams
      peppers := 60,
      cucumbers_per_serving := 2,
      tomatoes_per_serving := 2,
      bryndza_per_serving := 75,
      peppers_per_serving := 1 in
  min (min (cucumbers / cucumbers_per_serving) (tomatoes / tomatoes_per_serving))
      (min (bryndza / bryndza_per_serving) (peppers / peppers_per_serving)) = 56 := by
  sorry

end max_servings_l193_193995


namespace total_pens_bought_l193_193432

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_bought_l193_193432


namespace num_seven_digit_palindromes_l193_193167

-- Define the condition that a seven-digit palindrome takes the form abcdcba
def is_seven_digit_palindrome (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9

-- Lean statement to prove the number of seven-digit palindromes
theorem num_seven_digit_palindromes : 
  (∑ a in finset.range 9+1, ∑ b in finset.range 10, ∑ c in finset.range 10, ∑ d in finset.range 10, 1) = 9000 := 
by sorry

end num_seven_digit_palindromes_l193_193167


namespace graph_passes_through_2_2_l193_193810

theorem graph_passes_through_2_2 (a : ℝ) (h : a > 0) (h_ne : a ≠ 1) : (2, 2) ∈ { p : ℝ × ℝ | ∃ x, p = (x, a^(x-2) + 1) } :=
sorry

end graph_passes_through_2_2_l193_193810


namespace range_of_m_l193_193357

open Set Real

def A : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def B (m : ℝ) : Set ℝ := { x | x^2 - (m + 3) * x + m^2 = 0 }

theorem range_of_m (m : ℝ) :
  (A ∪ (univ \ B m)) = univ ↔ m ∈ Iio (-1) ∪ Ici 3 :=
sorry

end range_of_m_l193_193357


namespace focus_parabola_y_eq_neg4x2_plus_4x_minus_1_l193_193746

noncomputable def focus_of_parabola (a b c : ℝ) : ℝ × ℝ :=
  let p := b^2 / (4 * a) - c / (4 * a)
  (p, 1 / (4 * a))

theorem focus_parabola_y_eq_neg4x2_plus_4x_minus_1 :
  focus_of_parabola (-4) 4 (-1) = (1 / 2, -1 / 8) :=
sorry

end focus_parabola_y_eq_neg4x2_plus_4x_minus_1_l193_193746


namespace geometric_series_product_l193_193725

theorem geometric_series_product (y : ℝ) :
  (∑'n : ℕ, (1 / 3 : ℝ) ^ n) * (∑'n : ℕ, (- 1 / 3 : ℝ) ^ n)
  = ∑'n : ℕ, (y⁻¹ : ℝ) ^ n ↔ y = 9 :=
by
  sorry

end geometric_series_product_l193_193725


namespace sample_size_l193_193689

theorem sample_size (f_c f_o N: ℕ) (h1: f_c = 8) (h2: f_c = 1 / 4 * f_o) (h3: f_c + f_o = N) : N = 40 :=
  sorry

end sample_size_l193_193689


namespace find_largest_int_with_conditions_l193_193887

-- Definition of the problem conditions
def is_diff_of_consecutive_cubes (n : ℤ) : Prop :=
  ∃ m : ℤ, n^2 = (m + 1)^3 - m^3

def is_perfect_square_shifted (n : ℤ) : Prop :=
  ∃ k : ℤ, 2n + 99 = k^2

-- The main statement asserting the proof problem
theorem find_largest_int_with_conditions :
  ∃ n : ℤ, is_diff_of_consecutive_cubes n ∧ is_perfect_square_shifted n ∧
    ∀ m : ℤ, is_diff_of_consecutive_cubes m ∧ is_perfect_square_shifted m → m ≤ 50 :=
sorry

end find_largest_int_with_conditions_l193_193887


namespace gcd_of_abcd_dcba_l193_193640

theorem gcd_of_abcd_dcba : 
  ∀ (a : ℕ), 0 ≤ a ∧ a ≤ 3 → 
  gcd (2332 * a + 7112) (2332 * (a + 1) + 7112) = 2 ∧ 
  gcd (2332 * (a + 1) + 7112) (2332 * (a + 2) + 7112) = 2 ∧ 
  gcd (2332 * (a + 2) + 7112) (2332 * (a + 3) + 7112) = 2 := 
by 
  sorry

end gcd_of_abcd_dcba_l193_193640


namespace factorize_x_squared_minus_one_l193_193205

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end factorize_x_squared_minus_one_l193_193205


namespace total_pens_bought_l193_193431

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_bought_l193_193431


namespace my_and_mothers_ages_l193_193619

-- Definitions based on conditions
noncomputable def my_age (x : ℕ) := x
noncomputable def mothers_age (x : ℕ) := 3 * x
noncomputable def sum_of_ages (x : ℕ) := my_age x + mothers_age x

-- Proposition that needs to be proved
theorem my_and_mothers_ages (x : ℕ) (h : sum_of_ages x = 40) :
  my_age x = 10 ∧ mothers_age x = 30 :=
by
  sorry

end my_and_mothers_ages_l193_193619


namespace intersection_M_N_is_valid_l193_193360

-- Define the conditions given in the problem
def M := {x : ℝ |  3 / 4 < x ∧ x ≤ 1}
def N := {y : ℝ | 0 ≤ y}

-- State the theorem that needs to be proved
theorem intersection_M_N_is_valid : M ∩ N = {x : ℝ | 3 / 4 < x ∧ x ≤ 1} :=
by 
  sorry

end intersection_M_N_is_valid_l193_193360


namespace total_weight_of_watermelons_l193_193939

theorem total_weight_of_watermelons (w1 w2 : ℝ) (h1 : w1 = 9.91) (h2 : w2 = 4.11) :
  w1 + w2 = 14.02 :=
by
  sorry

end total_weight_of_watermelons_l193_193939


namespace product_of_two_real_numbers_sum_three_times_product_l193_193483

variable (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)

theorem product_of_two_real_numbers_sum_three_times_product
    (h : x + y = 3 * x * y) :
  x * y = (x + y) / 3 :=
sorry

end product_of_two_real_numbers_sum_three_times_product_l193_193483


namespace sum_odd_digits_from_1_to_200_l193_193611

/-- Function to compute the sum of odd digits of a number -/
def odd_digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.filter (fun d => d % 2 = 1) |>.sum

/-- Statement of the problem to prove the sum of the odd digits of numbers from 1 to 200 is 1000 -/
theorem sum_odd_digits_from_1_to_200 : (Finset.range 200).sum odd_digit_sum = 1000 := 
  sorry

end sum_odd_digits_from_1_to_200_l193_193611


namespace shadow_length_when_eight_meters_away_l193_193621

noncomputable def lamp_post_height : ℝ := 8
noncomputable def sam_initial_distance : ℝ := 12
noncomputable def shadow_initial_length : ℝ := 4
noncomputable def sam_initial_height : ℝ := 2 -- derived from the problem's steps

theorem shadow_length_when_eight_meters_away :
  ∀ (L : ℝ), (L * lamp_post_height) / (lamp_post_height + sam_initial_distance - shadow_initial_length) = 2 → L = 8 / 3 :=
by
  intro L
  sorry

end shadow_length_when_eight_meters_away_l193_193621


namespace sophomores_sampled_correct_l193_193518

def stratified_sampling_sophomores (total_students num_sophomores sample_size : ℕ) : ℕ :=
  (num_sophomores * sample_size) / total_students

theorem sophomores_sampled_correct :
  stratified_sampling_sophomores 4500 1500 600 = 200 :=
by
  sorry

end sophomores_sampled_correct_l193_193518


namespace correct_operation_l193_193660

theorem correct_operation (a b : ℝ) : 
  (a^2 + a^3 ≠ 2 * a^5) ∧
  ((a - b)^2 ≠ a^2 - b^2) ∧
  (a^3 * a^5 ≠ a^15) ∧
  ((ab^2)^2 = a^2 * b^4) :=
by
  sorry

end correct_operation_l193_193660


namespace color_points_l193_193095

def is_white (p : ℤ × ℤ) : Prop := (p.1 % 2 = 1) ∧ (p.2 % 2 = 1)
def is_black (p : ℤ × ℤ) : Prop := (p.1 % 2 = 0) ∧ (p.2 % 2 = 0)
def is_red (p : ℤ × ℤ) : Prop := (p.1 % 2 = 1 ∧ p.2 % 2 = 0) ∨ (p.1 % 2 = 0 ∧ p.2 % 2 = 1)

theorem color_points :
  (∀ n : ℤ, ∃ (p : ℤ × ℤ), (p.2 = n) ∧ is_white p ∧
                             is_black ⟨p.1, n * 2⟩ ∧
                             is_red ⟨p.1, n * 2 + 1⟩) ∧ 
  (∀ (A B C : ℤ × ℤ), 
    is_white A → is_red B → is_black C → 
    ∃ D : ℤ × ℤ, is_red D ∧ 
    (A.1 + C.1 - B.1 = D.1 ∧
     A.2 + C.2 - B.2 = D.2)) := sorry

end color_points_l193_193095


namespace tangent_line_through_P_line_through_P_chord_length_8_l193_193051

open Set

def circle (x y : ℝ) : Prop := x^2 + y^2 = 25

def point_P : ℝ × ℝ := (3, 4)

def tangent_line (x y : ℝ) : Prop := 3 * x + 4 * y - 25 = 0

def line_m_case1 (x : ℝ) : Prop := x = 3

def line_m_case2 (x y : ℝ) : Prop := 7 * x - 24 * y + 75 = 0

theorem tangent_line_through_P :
  tangent_line point_P.1 point_P.2 :=
sorry

theorem line_through_P_chord_length_8 :
  (∀ x y, circle x y → line_m_case1 x ∨ line_m_case2 x y) :=
sorry

end tangent_line_through_P_line_through_P_chord_length_8_l193_193051


namespace prob_blue_section_damaged_all_days_l193_193672

noncomputable def prob_of_7_successes (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem prob_blue_section_damaged_all_days :
  prob_of_7_successes 7 7 (2 / 7) = 128 / 823543 :=
by sorry

end prob_blue_section_damaged_all_days_l193_193672


namespace reggies_brother_long_shots_l193_193071

-- Define the number of points per type of shot
def layup_points : ℕ := 1
def free_throw_points : ℕ := 2
def long_shot_points : ℕ := 3

-- Define the number of shots made by Reggie
def reggie_layups : ℕ := 3
def reggie_free_throws : ℕ := 2
def reggie_long_shots : ℕ := 1

-- Define the total number of points made by Reggie
def reggie_points : ℕ :=
  reggie_layups * layup_points + reggie_free_throws * free_throw_points + reggie_long_shots * long_shot_points

-- Define the total points by which Reggie loses
def points_lost_by : ℕ := 2

-- Prove the number of long shots made by Reggie's brother
theorem reggies_brother_long_shots : 
  (reggie_points + points_lost_by) / long_shot_points = 4 := by
  sorry

end reggies_brother_long_shots_l193_193071


namespace factorization_of_difference_of_squares_l193_193246

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l193_193246


namespace beavers_still_working_is_one_l193_193511

def initial_beavers : Nat := 2
def beavers_swimming : Nat := 1
def still_working_beavers : Nat := initial_beavers - beavers_swimming

theorem beavers_still_working_is_one : still_working_beavers = 1 :=
by
  sorry

end beavers_still_working_is_one_l193_193511


namespace find_sum_m_n_l193_193831

-- Conditions from the problem definition
def tiles : Finset ℕ := Finset.range 21 -- 1 to 20 tiles

def players : ℕ := 3

-- Total number of ways to distribute tiles to each player
noncomputable def total_ways := (Nat.choose 20 3) * (Nat.choose 17 3) * (Nat.choose 14 3)

-- Number of ways for all players to select tiles that sum to even
noncomputable def even_sum_ways := (450^3 + 120^3) -- Sum of all successful configurations

-- Probability as a rational number
noncomputable def probability : ℚ := ⟨even_sum_ways, total_ways⟩.normalize -- Reduced probability

-- Numerator and Denominator
noncomputable def m : ℕ := probability.num
noncomputable def n : ℕ := probability.den

-- Final sum is m + n
theorem find_sum_m_n : m + n = 587 := by
  sorry

end find_sum_m_n_l193_193831


namespace numbers_written_in_red_l193_193954

theorem numbers_written_in_red :
  ∃ (x : ℕ), x > 0 ∧ x <= 101 ∧ 
  ∀ (largest_blue_num : ℕ) (smallest_red_num : ℕ), 
  (largest_blue_num = x) ∧ 
  (smallest_red_num = x + 1) ∧ 
  (smallest_red_num = (101 - x) / 2) → 
  (101 - x = 68) := by
  sorry

end numbers_written_in_red_l193_193954


namespace total_pens_l193_193416

/-- Proof that Masha and Olya bought a total of 38 pens given the cost conditions. -/
theorem total_pens (r : ℕ) (h_r : r > 10) (h1 : 357 % r = 0) (h2 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193416


namespace heracles_age_l193_193717

theorem heracles_age
  (H : ℕ)
  (audrey_current_age : ℕ)
  (audrey_in_3_years : ℕ)
  (h1 : audrey_current_age = H + 7)
  (h2 : audrey_in_3_years = audrey_current_age + 3)
  (h3 : audrey_in_3_years = 2 * H)
  : H = 10 :=
by
  sorry

end heracles_age_l193_193717


namespace find_second_sum_l193_193705

theorem find_second_sum (x : ℝ) (total_sum : ℝ) (h : total_sum = 2691) 
  (h1 : (24 * x) / 100 = 15 * (total_sum - x) / 100) : total_sum - x = 1656 :=
by
  sorry

end find_second_sum_l193_193705


namespace distinct_numbers_in_S_l193_193174

def sequence_A (k : ℕ) : ℕ := 4 * k - 2
def sequence_B (l : ℕ) : ℕ := 9 * l - 4
def set_A := Finset.image sequence_A (Finset.range 1500)
def set_B := Finset.image sequence_B (Finset.range 1500)
def set_S := set_A ∪ set_B

theorem distinct_numbers_in_S : set_S.card = 2833 :=
by 
  sorry

end distinct_numbers_in_S_l193_193174


namespace c_share_l193_193000

theorem c_share (S : ℝ) (b_share_per_rs c_share_per_rs : ℝ)
  (h1 : S = 246)
  (h2 : b_share_per_rs = 0.65)
  (h3 : c_share_per_rs = 0.40) :
  (c_share_per_rs * S) = 98.40 :=
by sorry

end c_share_l193_193000


namespace quadratic_inequality_has_real_solution_l193_193745

-- Define the quadratic function and the inequality
def quadratic (a x : ℝ) : ℝ := x^2 - 8 * x + a
def quadratic_inequality (a : ℝ) : Prop := ∃ x : ℝ, quadratic a x < 0

-- Define the condition for 'a' within the interval (0, 16)
def condition_on_a (a : ℝ) : Prop := 0 < a ∧ a < 16

-- The main statement to prove
theorem quadratic_inequality_has_real_solution (a : ℝ) (h : condition_on_a a) : quadratic_inequality a :=
sorry

end quadratic_inequality_has_real_solution_l193_193745


namespace range_of_a_l193_193053

noncomputable def f (a x : ℝ) : ℝ := x^2 - a * x + a + 3
noncomputable def g (a x : ℝ) : ℝ := a * x - 2 * a

theorem range_of_a (a : ℝ) : (∃ x₀ : ℝ, f a x₀ < 0 ∧ g a x₀ < 0) → 7 < a :=
by
  intro h
  sorry

end range_of_a_l193_193053


namespace shadow_length_when_eight_meters_away_l193_193622

noncomputable def lamp_post_height : ℝ := 8
noncomputable def sam_initial_distance : ℝ := 12
noncomputable def shadow_initial_length : ℝ := 4
noncomputable def sam_initial_height : ℝ := 2 -- derived from the problem's steps

theorem shadow_length_when_eight_meters_away :
  ∀ (L : ℝ), (L * lamp_post_height) / (lamp_post_height + sam_initial_distance - shadow_initial_length) = 2 → L = 8 / 3 :=
by
  intro L
  sorry

end shadow_length_when_eight_meters_away_l193_193622


namespace school_avg_GPA_l193_193630

theorem school_avg_GPA (gpa_6th : ℕ) (gpa_7th : ℕ) (gpa_8th : ℕ) 
  (h6 : gpa_6th = 93) 
  (h7 : gpa_7th = 95) 
  (h8 : gpa_8th = 91) : 
  (gpa_6th + gpa_7th + gpa_8th) / 3 = 93 :=
by 
  sorry

end school_avg_GPA_l193_193630


namespace smallest_x_solution_l193_193962

theorem smallest_x_solution (x : ℚ) :
  (7 * (8 * x^2 + 8 * x + 11) = x * (8 * x - 45)) →
  (x = -7/3 ∨ x = -11/16) →
  x = -7/3 :=
by
  sorry

end smallest_x_solution_l193_193962


namespace average_GPA_school_l193_193635

theorem average_GPA_school (GPA6 GPA7 GPA8 : ℕ) (h1 : GPA6 = 93) (h2 : GPA7 = GPA6 + 2) (h3 : GPA8 = 91) : ((GPA6 + GPA7 + GPA8) / 3) = 93 :=
by
  sorry

end average_GPA_school_l193_193635


namespace factorize_x_squared_minus_one_l193_193316

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end factorize_x_squared_minus_one_l193_193316


namespace average_hidden_primes_l193_193163

theorem average_hidden_primes
  (visible_card1 visible_card2 visible_card3 : ℕ)
  (hidden_card1 hidden_card2 hidden_card3 : ℕ)
  (h1 : visible_card1 = 68)
  (h2 : visible_card2 = 39)
  (h3 : visible_card3 = 57)
  (prime1 : Nat.Prime hidden_card1)
  (prime2 : Nat.Prime hidden_card2)
  (prime3 : Nat.Prime hidden_card3)
  (common_sum : ℕ)
  (h4 : visible_card1 + hidden_card1 = common_sum)
  (h5 : visible_card2 + hidden_card2 = common_sum)
  (h6 : visible_card3 + hidden_card3 = common_sum) :
  (hidden_card1 + hidden_card2 + hidden_card3) / 3 = 15 + 1/3 :=
sorry

end average_hidden_primes_l193_193163


namespace sandwiches_prepared_l193_193959

-- Define the conditions as given in the problem.
def ruth_ate_sandwiches : ℕ := 1
def brother_ate_sandwiches : ℕ := 2
def first_cousin_ate_sandwiches : ℕ := 2
def each_other_cousin_ate_sandwiches : ℕ := 1
def number_of_other_cousins : ℕ := 2
def sandwiches_left : ℕ := 3

-- Define the total number of sandwiches eaten.
def total_sandwiches_eaten : ℕ := ruth_ate_sandwiches 
                                  + brother_ate_sandwiches
                                  + first_cousin_ate_sandwiches 
                                  + (each_other_cousin_ate_sandwiches * number_of_other_cousins)

-- Define the number of sandwiches prepared by Ruth.
def sandwiches_prepared_by_ruth : ℕ := total_sandwiches_eaten + sandwiches_left

-- Formulate the theorem to prove.
theorem sandwiches_prepared : sandwiches_prepared_by_ruth = 10 :=
by
  -- Use the solution steps to prove the theorem (proof omitted here).
  sorry

end sandwiches_prepared_l193_193959


namespace max_value_expression_l193_193898

open Real

theorem max_value_expression (x : ℝ) : 
  ∃ (y : ℝ), y ≤ (x^6 / (x^10 + 3 * x^8 - 5 * x^6 + 10 * x^4 + 25)) ∧
  y = 1 / (5 + 2 * sqrt 30) :=
sorry

end max_value_expression_l193_193898


namespace total_pens_l193_193448

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193448


namespace original_rectangle_length_l193_193974

-- Define the problem conditions
def length_three_times_width (l w : ℕ) : Prop :=
  l = 3 * w

def length_decreased_width_increased (l w : ℕ) : Prop :=
  l - 5 = w + 5

-- Define the proof problem
theorem original_rectangle_length (l w : ℕ) (H1 : length_three_times_width l w) (H2 : length_decreased_width_increased l w) : l = 15 :=
sorry

end original_rectangle_length_l193_193974


namespace smallest_b_for_factorization_l193_193334

theorem smallest_b_for_factorization :
  ∃ b : ℕ, (∀ r s : ℤ, r * s = 2016 → r + s = b) ∧ b = 90 :=
sorry

end smallest_b_for_factorization_l193_193334


namespace smallest_of_three_integers_l193_193740

theorem smallest_of_three_integers (a b c : ℤ) (h1 : a * b * c = 32) (h2 : a + b + c = 3) : min (min a b) c = -4 := 
sorry

end smallest_of_three_integers_l193_193740


namespace chad_bbq_people_l193_193535

theorem chad_bbq_people (ice_cost_per_pack : ℝ) (packs_included : ℕ) (total_money_spent : ℝ) (pounds_needed_per_person : ℝ) :
  total_money_spent = 9 → 
  ice_cost_per_pack = 3 → 
  packs_included = 10 → 
  pounds_needed_per_person = 2 → 
  ∃ (people : ℕ), people = 15 :=
by intros; sorry

end chad_bbq_people_l193_193535


namespace find_percentage_l193_193766

theorem find_percentage (P : ℝ) (h1 : (3 / 5) * 150 = 90) (h2 : (P / 100) * 90 = 36) : P = 40 :=
by
  sorry

end find_percentage_l193_193766


namespace factorization_of_difference_of_squares_l193_193256

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l193_193256


namespace total_pens_bought_l193_193401

theorem total_pens_bought (r : ℕ) (hr : r > 10) (hm : 357 % r = 0) (ho : 441 % r = 0) :
  357 / r + 441 / r = 38 := by
  sorry

end total_pens_bought_l193_193401


namespace total_pens_l193_193449

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193449


namespace range_of_a_l193_193589

theorem range_of_a (x y a : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∀ (x y : ℝ), 0 < x → 0 < y → (y / 4 - (Real.cos x)^2) ≥ a * (Real.sin x) - 9 / y) ↔ (-3 ≤ a ∧ a ≤ 3) :=
sorry

end range_of_a_l193_193589


namespace gumball_probability_l193_193960

theorem gumball_probability :
  let total_gumballs : ℕ := 25
  let orange_gumballs : ℕ := 10
  let green_gumballs : ℕ := 6
  let yellow_gumballs : ℕ := 9
  let total_gumballs_after_first : ℕ := total_gumballs - 1
  let total_gumballs_after_second : ℕ := total_gumballs - 2
  let orange_probability_first : ℚ := orange_gumballs / total_gumballs
  let green_or_yellow_probability_second : ℚ := (green_gumballs + yellow_gumballs) / total_gumballs_after_first
  let orange_probability_third : ℚ := (orange_gumballs - 1) / total_gumballs_after_second
  orange_probability_first * green_or_yellow_probability_second * orange_probability_third = 9 / 92 :=
by
  sorry

end gumball_probability_l193_193960


namespace total_pens_l193_193424

theorem total_pens (r : ℕ) (h1 : r > 10)
  (h2 : 357 % r = 0)
  (h3 : 441 % r = 0) :
  357 / r + 441 / r = 38 :=
by
  sorry

end total_pens_l193_193424


namespace number_of_adult_males_l193_193701

def population := 480
def ratio_children := 1
def ratio_adult_males := 2
def ratio_adult_females := 2
def total_ratio_parts := ratio_children + ratio_adult_males + ratio_adult_females

theorem number_of_adult_males : 
  (population / total_ratio_parts) * ratio_adult_males = 192 :=
by
  sorry

end number_of_adult_males_l193_193701


namespace tailor_trim_length_l193_193525

theorem tailor_trim_length (x : ℕ) : 
  (18 - x) * 15 = 120 → x = 10 := 
by
  sorry

end tailor_trim_length_l193_193525


namespace factor_difference_of_squares_l193_193264

theorem factor_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factor_difference_of_squares_l193_193264


namespace number_of_such_fractions_is_one_l193_193552

theorem number_of_such_fractions_is_one :
  ∃! (x y : ℕ), (Nat.gcd x y = 1) ∧ (1 < x) ∧ (1 < y) ∧ (x+1)/(y+1) = (6*x)/(5*y) := 
begin
  sorry
end

end number_of_such_fractions_is_one_l193_193552


namespace total_pens_l193_193408

theorem total_pens (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) :
  (357 / r) + (441 / r) = 38 :=
sorry

end total_pens_l193_193408


namespace correct_operation_l193_193498

theorem correct_operation : ¬ (-2 * x + 5 * x = -7 * x) 
                          ∧ (y * x - 3 * x * y = -2 * x * y) 
                          ∧ ¬ (-x^2 - x^2 = 0) 
                          ∧ ¬ (x^2 - x = x) := 
by {
    sorry
}

end correct_operation_l193_193498


namespace smallest_repeating_block_length_l193_193177

-- Define the decimal expansion of 3/11
noncomputable def decimalExpansion : Rational → List Nat :=
  sorry

-- Define the repeating block determination of a given decimal expansion
noncomputable def repeatingBlockLength : List Nat → Nat :=
  sorry

-- Define the fraction 3/11
def frac := (3 : Rat) / 11

-- State the theorem
theorem smallest_repeating_block_length :
  repeatingBlockLength (decimalExpansion frac) = 2 :=
  sorry

end smallest_repeating_block_length_l193_193177


namespace range_of_a_l193_193354

def p (a : ℝ) : Prop :=
(∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0)

def q (a : ℝ) : Prop :=
0 < a ∧ a < 1

theorem range_of_a (a : ℝ) : ((p a ∨ q a) ∧ ¬(p a ∧ q a)) ↔ (1 ≤ a ∧ a ≤ 2) ∨ (-2 < a ∧ a ≤ 0) :=
  sorry

end range_of_a_l193_193354


namespace factorization_of_x_squared_minus_one_l193_193237

-- Let x be an arbitrary real number
variable (x : ℝ)

-- Theorem stating that x^2 - 1 can be factored as (x + 1)(x - 1)
theorem factorization_of_x_squared_minus_one : x^2 - 1 = (x + 1) * (x - 1) := 
sorry

end factorization_of_x_squared_minus_one_l193_193237
