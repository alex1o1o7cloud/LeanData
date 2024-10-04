import Mathlib

namespace gcd_factorials_l139_139211

theorem gcd_factorials : Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  -- Proof steps would go here
  sorry

end gcd_factorials_l139_139211


namespace value_of_expression_l139_139014

theorem value_of_expression (x y : ℤ) (h1 : x = 3) (h2 : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end value_of_expression_l139_139014


namespace sequence_a6_value_l139_139920

theorem sequence_a6_value 
  (a : ℕ → ℝ)
  (h1 : a 1 = 2)
  (h2 : a 2 = 1)
  (h3 : ∀ n : ℕ, n ≥ 1 → (1 / a n) + (1 / a (n + 2)) = 2 / a (n + 1)) :
  a 6 = 1 / 3 :=
by
  sorry

end sequence_a6_value_l139_139920


namespace business_proof_l139_139561

section Business_Problem

variables (investment cost_initial rubles production_capacity : ℕ)
variables (produced_July incomplete_July bottles_August bottles_September days_September : ℕ)
variables (total_depreciation residual_value sales_amount profit_target : ℕ)

def depreciation_per_bottle (cost_initial production_capacity : ℕ) : ℕ := 
    cost_initial / production_capacity

def calculate_total_depreciation (depreciation_per_bottle produced_July bottles_August bottles_September : ℕ) : ℕ :=
    (produced_July * depreciation_per_bottle) + (bottles_August * depreciation_per_bottle) + (bottles_September * depreciation_per_bottle)

def calculate_residual_value (cost_initial total_depreciation : ℕ) : ℕ :=
    cost_initial - total_depreciation

def calculate_sales_amount (residual_value profit_target : ℕ) : ℕ :=
    residual_value + profit_target

theorem business_proof
    (H1: investment = 1500000) 
    (H2: cost_initial = 500000)
    (H3: production_capacity = 100000)
    (H4: produced_July = 200)
    (H5: incomplete_July = 5)
    (H6: bottles_August = 15000)
    (H7: bottles_September = 12300)
    (H8: days_September = 20)
    (H9: total_depreciation = 137500)
    (H10: residual_value = 362500)
    (H11: profit_target = 10000)
    (H12: sales_amount = 372500): 

    total_depreciation = calculate_total_depreciation (depreciation_per_bottle cost_initial production_capacity) produced_July bottles_August bottles_September ∧
    residual_value = calculate_residual_value cost_initial total_depreciation ∧
    sales_amount = calculate_sales_amount residual_value profit_target := 
by 
  sorry

end Business_Problem

end business_proof_l139_139561


namespace find_x_l139_139441

theorem find_x (a b x : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < x)
    (h4 : (4 * a)^(4 * b) = a^b * x^(2 * b)) : 
    x = 16 * a^(3 / 2) :=
by 
  sorry

end find_x_l139_139441


namespace defective_pencil_count_l139_139292

theorem defective_pencil_count (total_pencils : ℕ) (selected_pencils : ℕ) 
  (prob_non_defective : ℚ) (D N : ℕ) (h_total : total_pencils = 6) 
  (h_selected : selected_pencils = 3)
  (h_prob : prob_non_defective = 0.2)
  (h_sum : D + N = 6) 
  (h_comb : (N.choose 3 : ℚ) / (total_pencils.choose 3) = prob_non_defective) : 
  D = 2 := 
sorry

end defective_pencil_count_l139_139292


namespace fraction_value_l139_139701

theorem fraction_value (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 8) : (x + y) / 3 = 5 / 3 :=
by sorry

end fraction_value_l139_139701


namespace logarithm_identity_l139_139905

theorem logarithm_identity (k x : ℝ) (hk : 0 < k ∧ k ≠ 1) (hx : 0 < x) :
  (Real.log x / Real.log k) * (Real.log k / Real.log 7) = 3 → x = 343 :=
by
  intro h
  sorry

end logarithm_identity_l139_139905


namespace sum_of_a_b_l139_139417

variable {a b : ℝ}

theorem sum_of_a_b (h1 : a^2 = 4) (h2 : b^2 = 9) (h3 : a * b < 0) : a + b = 1 ∨ a + b = -1 := 
by 
  sorry

end sum_of_a_b_l139_139417


namespace max_gold_coins_l139_139699

-- Define the conditions as predicates
def divides_with_remainder (n : ℕ) (d r : ℕ) : Prop := n % d = r
def less_than (n k : ℕ) : Prop := n < k

-- Main statement incorporating the conditions and the conclusion
theorem max_gold_coins (n : ℕ) :
  divides_with_remainder n 15 3 ∧ less_than n 120 → n ≤ 105 :=
by
  sorry

end max_gold_coins_l139_139699


namespace total_eggs_collected_l139_139112

-- Define the variables given in the conditions
def Benjamin_eggs := 6
def Carla_eggs := 3 * Benjamin_eggs
def Trisha_eggs := Benjamin_eggs - 4

-- State the theorem using the conditions and correct answer in the equivalent proof problem
theorem total_eggs_collected :
  Benjamin_eggs + Carla_eggs + Trisha_eggs = 26 := by
  -- Proof goes here.
  sorry

end total_eggs_collected_l139_139112


namespace number_of_true_propositions_l139_139753

open Classical

axiom real_numbers (a b : ℝ): Prop

noncomputable def original_proposition (a b : ℝ) : Prop := a > b → a * abs a > b * abs b
noncomputable def converse_proposition (a b : ℝ) : Prop := a * abs a > b * abs b → a > b
noncomputable def negation_proposition (a b : ℝ) : Prop := a ≤ b → a * abs a ≤ b * abs b
noncomputable def contrapositive_proposition (a b : ℝ) : Prop := a * abs a ≤ b * abs b → a ≤ b

theorem number_of_true_propositions (a b : ℝ) (h₁: original_proposition a b) 
  (h₂: converse_proposition a b) (h₃: negation_proposition a b)
  (h₄: contrapositive_proposition a b) : ∃ n, n = 4 := 
by
  -- The proof would go here, proving that ∃ n, n = 4 is true.
  sorry

end number_of_true_propositions_l139_139753


namespace john_saves_money_l139_139924

theorem john_saves_money :
  let original_spending := 4 * 2
  let new_price_per_coffee := 2 + (2 * 0.5)
  let new_coffees := 4 / 2
  let new_spending := new_coffees * new_price_per_coffee
  original_spending - new_spending = 2 :=
by
  -- calculations omitted
  sorry

end john_saves_money_l139_139924


namespace constant_term_in_expansion_l139_139248

theorem constant_term_in_expansion (n k : ℕ) (x : ℝ) (choose : ℕ → ℕ → ℕ):
  (choose 12 3) * (6 ^ 3) = 47520 :=
by
  sorry

end constant_term_in_expansion_l139_139248


namespace nonagon_diagonals_not_parallel_l139_139281

theorem nonagon_diagonals_not_parallel (n : ℕ) (h : n = 9) : 
  ∃ k : ℕ, k = 18 ∧ 
    ∀ v₁ v₂, v₁ ≠ v₂ → (n : ℕ).choose 2 = 27 → 
    (v₂ - v₁) % n ≠ 4 ∧ (v₂ - v₁) % n ≠ n-4 :=
by
  sorry

end nonagon_diagonals_not_parallel_l139_139281


namespace trigonometric_inequality_for_tan_l139_139951

open Real

theorem trigonometric_inequality_for_tan (x : ℝ) (hx1 : 0 < x) (hx2 : x < π / 2) : 
  1 + tan x < 1 / (1 - sin x) :=
sorry

end trigonometric_inequality_for_tan_l139_139951


namespace original_number_is_7_l139_139363

theorem original_number_is_7 (x : ℤ) (h : (((3 * (x + 3) + 3) - 3) / 3) = 10) : x = 7 :=
sorry

end original_number_is_7_l139_139363


namespace a_sq_greater_than_b_sq_neither_sufficient_nor_necessary_l139_139271

theorem a_sq_greater_than_b_sq_neither_sufficient_nor_necessary 
  (a b : ℝ) : ¬ ((a^2 > b^2) → (a > b)) ∧  ¬ ((a > b) → (a^2 > b^2)) := sorry

end a_sq_greater_than_b_sq_neither_sufficient_nor_necessary_l139_139271


namespace ratio_doubled_to_original_l139_139994

theorem ratio_doubled_to_original (x y : ℕ) (h1 : y = 2 * x + 9) (h2 : 3 * y = 57) : 2 * x = 2 * (x / 1) := 
by sorry

end ratio_doubled_to_original_l139_139994


namespace expression_for_M_value_of_M_when_x_eq_negative_2_and_y_eq_1_l139_139147

noncomputable def A (x y : ℝ) := x^2 - 3 * x * y - y^2
noncomputable def B (x y : ℝ) := x^2 - 3 * x * y - 3 * y^2
noncomputable def M (x y : ℝ) := 2 * A x y - B x y

theorem expression_for_M (x y : ℝ) : M x y = x^2 - 3 * x * y + y^2 := by
  sorry

theorem value_of_M_when_x_eq_negative_2_and_y_eq_1 :
  M (-2) 1 = 11 := by
  sorry

end expression_for_M_value_of_M_when_x_eq_negative_2_and_y_eq_1_l139_139147


namespace fraction_of_managers_l139_139966

theorem fraction_of_managers (female_managers : ℕ) (total_female_employees : ℕ)
  (total_employees: ℕ) (male_employees: ℕ) (f: ℝ) :
  female_managers = 200 →
  total_female_employees = 500 →
  total_employees = total_female_employees + male_employees →
  (f * total_employees) = female_managers + (f * male_employees) →
  f = 0.4 :=
by
  intros h1 h2 h3 h4
  sorry

end fraction_of_managers_l139_139966


namespace apples_total_l139_139529

theorem apples_total (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apples_total_l139_139529


namespace circular_film_diameter_l139_139655

-- Definition of the problem conditions
def liquidVolume : ℝ := 576  -- volume of liquid Y in cm^3
def filmThickness : ℝ := 0.2  -- thickness of the film in cm

-- Statement of the theorem to prove the diameter of the film
theorem circular_film_diameter :
  2 * Real.sqrt (2880 / Real.pi) = 2 * Real.sqrt (liquidVolume / (filmThickness * Real.pi)) := by
  sorry

end circular_film_diameter_l139_139655


namespace small_barrel_5_tons_l139_139852

def total_oil : ℕ := 95
def large_barrel_capacity : ℕ := 6
def small_barrel_capacity : ℕ := 5

theorem small_barrel_5_tons :
  ∃ (num_large_barrels num_small_barrels : ℕ),
  num_small_barrels = 1 ∧
  total_oil = (num_large_barrels * large_barrel_capacity) + (num_small_barrels * small_barrel_capacity) :=
by
  sorry

end small_barrel_5_tons_l139_139852


namespace count_squares_within_region_l139_139006

noncomputable def countSquares : Nat := sorry

theorem count_squares_within_region :
  countSquares = 45 :=
sorry

end count_squares_within_region_l139_139006


namespace apple_bags_l139_139527

theorem apple_bags (n : ℕ) (h₁ : n ≥ 70) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 := 
sorry

end apple_bags_l139_139527


namespace inequality_x_y_l139_139394

theorem inequality_x_y (x y : ℝ) (h : x^12 + y^12 ≤ 2) : x^2 + y^2 + x^2 * y^2 ≤ 3 := 
sorry

end inequality_x_y_l139_139394


namespace find_y_value_l139_139149

theorem find_y_value (y : ℕ) : (1/8 * 2^36 = 2^33) ∧ (8^y = 2^(3 * y)) → y = 11 :=
by
  intros h
  -- additional elaboration to verify each step using Lean, skipped for simplicity
  sorry

end find_y_value_l139_139149


namespace maximum_pyramid_volume_l139_139970

-- Variables and constants given in the problem
variables (A B C S : Type) [EuclideanSpace ℝ A] [EuclideanSpace ℝ B] [EuclideanSpace ℝ C] [EuclideanSpace ℝ S]
variables (AB AC : ℝ) (angleABC : ℝ) 
constant (sin_angleBAC : ℝ) (max_angle : ℝ )

-- Given conditions
def AB_len : Prop := AB = 5
def AC_len : Prop := AC = 8
def sin_val : Prop := sin_angleBAC = 3/5
def max_angle_val : Prop := max_angle = 60

-- Statement to prove
theorem maximum_pyramid_volume : 
  AB_len AB → AC_len AC → sin_val sin_angleBAC → max_angle_val max_angle →
  ∃ V : ℝ, V = 10 * Real.sqrt 51 :=
by
  sorry

end maximum_pyramid_volume_l139_139970


namespace find_x_solutions_l139_139930

theorem find_x_solutions (x : ℝ) :
  let f (x : ℝ) := x^2 - 4*x + 1
  let f2 (x : ℝ) := (f x)^2
  f (f x) = f2 x ↔ x = 2 + (Real.sqrt 13) / 2 ∨ x = 2 - (Real.sqrt 13) / 2 := by
  sorry

end find_x_solutions_l139_139930


namespace circle_center_radius_l139_139258

theorem circle_center_radius (x y : ℝ) :
  x^2 - 6*x + y^2 + 2*y - 9 = 0 ↔ (x-3)^2 + (y+1)^2 = 19 :=
sorry

end circle_center_radius_l139_139258


namespace expected_value_correct_l139_139457

-- Define the probability distribution of the user's score in the first round
noncomputable def first_round_prob (X : ℕ) : ℚ :=
  if X = 3 then 1 / 4
  else if X = 2 then 1 / 2
  else if X = 1 then 1 / 4
  else 0

-- Define the conditional probability of the user's score in the second round given the first round score
noncomputable def second_round_prob (X Y : ℕ) : ℚ :=
  if X = 3 then
    if Y = 2 then 1 / 5
    else if Y = 1 then 4 / 5
    else 0
  else
    if Y = 2 then 1 / 3
    else if Y = 1 then 2 / 3
    else 0

-- Define the total score probability
noncomputable def total_score_prob (X Y : ℕ) : ℚ :=
  first_round_prob X * second_round_prob X Y

-- Compute the expected value of the user's total score
noncomputable def expected_value : ℚ :=
  (5 * (total_score_prob 3 2) +
   4 * (total_score_prob 3 1 + total_score_prob 2 2) +
   3 * (total_score_prob 2 1 + total_score_prob 1 2) +
   2 * (total_score_prob 1 1))

-- The theorem to be proven
theorem expected_value_correct : expected_value = 3.3 := 
by sorry

end expected_value_correct_l139_139457


namespace Dorothy_found_57_pieces_l139_139857

def total_pieces_Dorothy_found 
  (B_green B_red R_red R_blue : ℕ)
  (D_red_factor D_blue_factor : ℕ)
  (H1 : B_green = 12)
  (H2 : B_red = 3)
  (H3 : R_red = 9)
  (H4 : R_blue = 11)
  (H5 : D_red_factor = 2)
  (H6 : D_blue_factor = 3) : ℕ := 
  let D_red := D_red_factor * (B_red + R_red)
  let D_blue := D_blue_factor * R_blue
  D_red + D_blue

theorem Dorothy_found_57_pieces 
  (B_green B_red R_red R_blue : ℕ)
  (D_red_factor D_blue_factor : ℕ)
  (H1 : B_green = 12)
  (H2 : B_red = 3)
  (H3 : R_red = 9)
  (H4 : R_blue = 11)
  (H5 : D_red_factor = 2)
  (H6 : D_blue_factor = 3) :
  total_pieces_Dorothy_found B_green B_red R_red R_blue D_red_factor D_blue_factor H1 H2 H3 H4 H5 H6 = 57 := by
    sorry

end Dorothy_found_57_pieces_l139_139857


namespace landmark_postcards_probability_l139_139450

theorem landmark_postcards_probability :
  let total_postcards := 12
  let landmark_postcards := 4
  let total_arrangements := Nat.factorial total_postcards
  let favorable_arrangements := Nat.factorial (total_postcards - landmark_postcards + 1) * Nat.factorial landmark_postcards
  favorable_arrangements / total_arrangements = (1:ℝ) / 55 :=
by
  sorry

end landmark_postcards_probability_l139_139450


namespace sages_success_l139_139345

-- Assume we have a finite type representing our 1000 colors
inductive Color
| mk : Fin 1000 → Color

open Color

-- Define the sages
def Sage : Type := Fin 11

-- Define the problem conditions into a Lean structure
structure Problem :=
  (sages : Fin 11)
  (colors : Fin 1000)
  (assignments : Sage → Color)
  (strategies : Sage → (Fin 1024 → Fin 2))

-- Define the success condition
def success (p : Problem) : Prop :=
  ∃ (strategies : Sage → (Fin 1024 → Fin 2)),
    ∀ (assignment : Sage → Color),
      ∃ (color_guesses : Sage → Color),
        (∀ s, color_guesses s = assignment s)

-- The sages will succeed in determining the colors of their hats.
theorem sages_success : ∀ (p : Problem), success p := by
  sorry

end sages_success_l139_139345


namespace original_number_l139_139759

theorem original_number (x : ℝ) (h1 : 268 * x = 19832) (h2 : 2.68 * x = 1.9832) : x = 74 :=
sorry

end original_number_l139_139759


namespace oplus_calculation_l139_139041

def my_oplus (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem oplus_calculation : my_oplus 2 3 = 23 := 
by
    sorry

end oplus_calculation_l139_139041


namespace function_solution_l139_139605

theorem function_solution (f : ℝ → ℝ) (H : ∀ x y : ℝ, 1 < x → 1 < y → f x - f y = (y - x) * f (x * y)) :
  ∃ k : ℝ, ∀ x : ℝ, 1 < x → f x = k / x :=
by
  sorry

end function_solution_l139_139605


namespace sum_of_cubes_l139_139932

theorem sum_of_cubes {x y : ℝ} (h₁ : x + y = 0) (h₂ : x * y = -1) : x^3 + y^3 = 0 :=
by
  sorry

end sum_of_cubes_l139_139932


namespace quadratic_inequality_solution_set_l139_139909

variables {a b c : ℝ}

theorem quadratic_inequality_solution_set (h : ∀ x, x > -1 ∧ x < 2 → ax^2 - bx + c > 0) :
  a + b + c = 0 :=
sorry

end quadratic_inequality_solution_set_l139_139909


namespace range_of_set_l139_139233

theorem range_of_set (a b c : ℕ) (h1 : a = 2) (h2 : b = 6) (h3 : 2 ≤ c ∧ c ≤ 10) (h4 : (a + b + c) / 3 = 6) : (c - a) = 8 :=
by
  sorry

end range_of_set_l139_139233


namespace possible_apple_counts_l139_139501

theorem possible_apple_counts (n : ℕ) (h₁ : 70 ≤ n) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end possible_apple_counts_l139_139501


namespace bacteria_population_at_15_l139_139435

noncomputable def bacteria_population (t : ℕ) : ℕ := 
  20 * 2 ^ (t / 3)

theorem bacteria_population_at_15 : bacteria_population 15 = 640 := by
  sorry

end bacteria_population_at_15_l139_139435


namespace max_revenue_l139_139094

variable (x y : ℝ)

-- Conditions
def ads_time_constraint := x + y ≤ 300
def ads_cost_constraint := 500 * x + 200 * y ≤ 90000
def revenue := 0.3 * x + 0.2 * y

-- Question: Prove that the maximum revenue is 70 million yuan
theorem max_revenue (h_time : ads_time_constraint (x := 100) (y := 200))
                    (h_cost : ads_cost_constraint (x := 100) (y := 200)) :
  revenue (x := 100) (y := 200) = 70 := 
sorry

end max_revenue_l139_139094


namespace length_BE_l139_139808

-- Define points and distances
variables (A B C D E : Type)
variable {AB : ℝ}
variable {BC : ℝ}
variable {CD : ℝ}
variable {DA : ℝ}

-- Given conditions
axiom AB_length : AB = 5
axiom BC_length : BC = 7
axiom CD_length : CD = 8
axiom DA_length : DA = 6

-- Bugs travelling in opposite directions from point A meet at E
axiom bugs_meet_at_E : True

-- Proving the length BE
theorem length_BE : BE = 6 :=
by
  -- Currently, this is a statement. The proof is not included.
  sorry

end length_BE_l139_139808


namespace apples_total_l139_139530

theorem apples_total (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apples_total_l139_139530


namespace apple_bags_l139_139523

theorem apple_bags (n : ℕ) (h₁ : n ≥ 70) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 := 
sorry

end apple_bags_l139_139523


namespace opposite_of_113_is_114_l139_139944

theorem opposite_of_113_is_114
  (circle : Finset ℕ)
  (h1 : ∀ n ∈ circle, n ∈ (Finset.range 1 200))
  (h2 : ∀ n, ∃ m, m = (n + 100) % 200)
  (h3 : ∀ n, 1 ≤ n ∧ n ≤ 200)
  : ((113 + 100) % 200 = 114) := by
  sorry

end opposite_of_113_is_114_l139_139944


namespace find_smallest_integer_l139_139836

/-- There exists an integer n such that:
   n ≡ 1 [MOD 3],
   n ≡ 2 [MOD 4],
   n ≡ 3 [MOD 5],
   and the smallest such n is 58. -/
theorem find_smallest_integer :
  ∃ n : ℕ, n % 3 = 1 ∧ n % 4 = 2 ∧ n % 5 = 3 ∧ n = 58 :=
by
  -- Proof goes here (not provided as per the instructions)
  sorry

end find_smallest_integer_l139_139836


namespace required_percentage_to_pass_l139_139580

theorem required_percentage_to_pass
  (marks_obtained : ℝ)
  (marks_failed_by : ℝ)
  (max_marks : ℝ)
  (passing_marks := marks_obtained + marks_failed_by)
  (required_percentage : ℝ := (passing_marks / max_marks) * 100)
  (h : marks_obtained = 80)
  (h' : marks_failed_by = 40)
  (h'' : max_marks = 200) :
  required_percentage = 60 := 
by
  sorry

end required_percentage_to_pass_l139_139580


namespace range_of_k_l139_139131

theorem range_of_k (x y k : ℝ) (h1 : x - y = k - 1) (h2 : 3 * x + 2 * y = 4 * k + 5) (hk : 2 * x + 3 * y > 7) : k > 1 / 3 := 
sorry

end range_of_k_l139_139131


namespace distance_to_first_sign_l139_139449

-- Definitions based on conditions
def total_distance : ℕ := 1000
def after_second_sign : ℕ := 275
def between_signs : ℕ := 375

-- Problem statement
theorem distance_to_first_sign 
  (D : ℕ := total_distance) 
  (a : ℕ := after_second_sign) 
  (d : ℕ := between_signs) : 
  (D - a - d = 350) :=
by
  sorry

end distance_to_first_sign_l139_139449


namespace ball_bouncing_height_l139_139986

theorem ball_bouncing_height : ∃ (b : ℕ), 400 * (3/4 : ℝ)^b < 50 ∧ ∀ n < b, 400 * (3/4 : ℝ)^n ≥ 50 :=
by
  use 8
  sorry

end ball_bouncing_height_l139_139986


namespace apple_count_l139_139491

-- Define constants for the problem
constant bags: ℕ → Prop
constant total_apples: ℕ
constant apples_in_bag1: ℕ := 12
constant apples_in_bag2: ℕ := 6

-- Define the conditions
axiom condition1 (n: ℕ) : bags n → (n = apples_in_bag1 ∨ n = apples_in_bag2)
axiom condition2 : 70 ≤ total_apples ∧ total_apples ≤ 80
axiom condition3 : total_apples % 6 = 0

-- State the proof problem
theorem apple_count :
  total_apples = 72 ∨ total_apples = 78 :=
sorry

end apple_count_l139_139491


namespace original_stone_counted_as_99_l139_139339

theorem original_stone_counted_as_99 :
  (99 % 22) = 11 :=
by sorry

end original_stone_counted_as_99_l139_139339


namespace original_perimeter_not_necessarily_multiple_of_four_l139_139105

/-
Define the conditions given in the problem:
1. A rectangle is divided into several smaller rectangles.
2. The perimeter of each of these smaller rectangles is a multiple of 4.
-/
structure Rectangle where
  length : ℕ
  width : ℕ

def perimeter (r : Rectangle) : ℕ :=
  2 * (r.length + r.width)

def is_multiple_of_four (n : ℕ) : Prop :=
  n % 4 = 0

def smaller_rectangles (rs : List Rectangle) : Prop :=
  ∀ r ∈ rs, is_multiple_of_four (perimeter r)

-- Define the main statement to be proved
theorem original_perimeter_not_necessarily_multiple_of_four (original : Rectangle) (rs : List Rectangle)
  (h1 : smaller_rectangles rs) (h2 : ∀ r ∈ rs, r.length * r.width = original.length * original.width) :
  ¬ is_multiple_of_four (perimeter original) :=
by
  sorry

end original_perimeter_not_necessarily_multiple_of_four_l139_139105


namespace apples_total_l139_139528

theorem apples_total (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apples_total_l139_139528


namespace train_journey_duration_l139_139092

def battery_lifespan (talk_time standby_time : ℝ) :=
  talk_time <= 6 ∧ standby_time <= 210

def full_battery_usage (total_time : ℝ) :=
  (total_time / 2) / 6 + (total_time / 2) / 210 = 1

theorem train_journey_duration (t : ℝ) (h1 : battery_lifespan (t / 2) (t / 2)) (h2 : full_battery_usage t) :
  t = 35 / 3 :=
sorry

end train_journey_duration_l139_139092


namespace quadratic_inequality_solution_l139_139325

open Real

theorem quadratic_inequality_solution :
    ∀ x : ℝ, -8 * x^2 + 6 * x - 1 < 0 ↔ 0.25 < x ∧ x < 0.5 :=
by sorry

end quadratic_inequality_solution_l139_139325


namespace cost_price_per_meter_l139_139998

namespace ClothCost

theorem cost_price_per_meter (selling_price_total : ℝ) (meters_sold : ℕ) (loss_per_meter : ℝ) : 
  selling_price_total = 18000 → 
  meters_sold = 300 → 
  loss_per_meter = 5 →
  (selling_price_total / meters_sold) + loss_per_meter = 65 := 
by
  intros hsp hms hloss
  sorry

end ClothCost

end cost_price_per_meter_l139_139998


namespace scientific_notation_of_9280000000_l139_139723

theorem scientific_notation_of_9280000000 :
  9280000000 = 9.28 * 10^9 :=
by
  sorry

end scientific_notation_of_9280000000_l139_139723


namespace range_of_b_l139_139600

theorem range_of_b (b : ℝ) : 
  (¬ (4 ≤ 3 * 3 + b) ∧ (4 ≤ 3 * 4 + b)) ↔ (-8 ≤ b ∧ b < -5) := 
by
  sorry

end range_of_b_l139_139600


namespace paintable_fence_l139_139406

theorem paintable_fence :
  ∃ h t u : ℕ,  h > 1 ∧ t > 1 ∧ u > 1 ∧ 
  (∀ n, 4 + (n * h) ≠ 5 + (m * (2 * t))) ∧
  (∀ n, 4 + (n * h) ≠ 6 + (l * (3 * u))) ∧ 
  (∀ m l, 5 + (m * (2 * t)) ≠ 6 + (l * (3 * u))) ∧
  (100 * h + 20 * t + 2 * u = 390) :=
by 
  sorry

end paintable_fence_l139_139406


namespace find_sum_of_money_l139_139553

theorem find_sum_of_money (P : ℝ) (H1 : P * 0.18 * 2 - P * 0.12 * 2 = 840) : P = 7000 :=
by
  sorry

end find_sum_of_money_l139_139553


namespace smaller_octagon_area_half_l139_139677

theorem smaller_octagon_area_half
  (ABCDEFGH : Type) [is_regular_octagon ABCDEFGH]
  (P Q R S T U V W : Point)
  (H1 : midpoint P A B)
  (H2 : midpoint Q B C)
  (H3 : midpoint R C D)
  (H4 : midpoint S D E)
  (H5 : midpoint T E F)
  (H6 : midpoint U F G)
  (H7 : midpoint V G H)
  (H8 : midpoint W H A):
  area (octagon P Q R S T U V W) = (1 / 2) * area (octagon ABCDEFGH) :=
sorry

end smaller_octagon_area_half_l139_139677


namespace coeff_x7_in_expansion_l139_139543

-- Each definition in Lean 4 statement reflects the conditions of the problem.
def binomial_coefficient (n k : ℕ) : ℕ := nat.choose n k

-- The condition for expansion using Binomial Theorem
def binomial_expansion_term (n k : ℕ) (a x : ℤ) : ℤ :=
  binomial_coefficient n k * a ^ (n - k) * x ^ k

-- Prove that the coefficient of x^7 in the expansion of (x - 2)^{10} is -960
theorem coeff_x7_in_expansion : 
  binomial_coefficient 10 3 * (-2) ^ 3 = -960 := 
sorry

end coeff_x7_in_expansion_l139_139543


namespace donuts_selection_l139_139054

theorem donuts_selection :
  (∃ g c p : ℕ, g + c + p = 6 ∧ g ≥ 1 ∧ c ≥ 1 ∧ p ≥ 1) →
  ∃ k : ℕ, k = 10 :=
by {
  -- The mathematical proof steps are omitted according to the instructions
  sorry
}

end donuts_selection_l139_139054


namespace remainder_div_8_l139_139091

theorem remainder_div_8 (x : ℤ) (h : ∃ k : ℤ, x = 63 * k + 27) : x % 8 = 3 :=
by
  sorry

end remainder_div_8_l139_139091


namespace value_of_expression_l139_139076

def expr : ℕ :=
  8 + 2 * (3^2)

theorem value_of_expression : expr = 26 :=
  by
  sorry

end value_of_expression_l139_139076


namespace sufficient_but_not_necessary_condition_l139_139137

variable (a : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : a = 1) (h2 : |a| = 1) : 
  (a = 1 → |a| = 1) ∧ ¬(|a| = 1 → a = 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l139_139137


namespace simplify_complex_fraction_l139_139662

theorem simplify_complex_fraction :
  let numerator := (5 : ℂ) + 7 * I
  let denominator := (2 : ℂ) + 3 * I
  numerator / denominator = (31 / 13 : ℂ) - (1 / 13) * I :=
by
  let numerator := (5 : ℂ) + 7 * I
  let denominator := (2 : ℂ) + 3 * I
  sorry

end simplify_complex_fraction_l139_139662


namespace abs_eq_abs_implies_l139_139216

theorem abs_eq_abs_implies (x : ℝ) (h : |x - 3| = |x + 1|) : x = 1 := 
sorry

end abs_eq_abs_implies_l139_139216


namespace Ava_watch_minutes_l139_139589

theorem Ava_watch_minutes (hours_watched : ℕ) (minutes_per_hour : ℕ) (h : hours_watched = 4) (m : minutes_per_hour = 60) : 
  hours_watched * minutes_per_hour = 240 :=
by
  sorry

end Ava_watch_minutes_l139_139589


namespace max_det_bound_l139_139910

noncomputable def max_det_estimate : ℕ := 327680 * 2^16

theorem max_det_bound (M : Matrix (Fin 17) (Fin 17) ℤ)
  (h : ∀ i j, M i j = 1 ∨ M i j = -1) :
  abs (Matrix.det M) ≤ max_det_estimate :=
sorry

end max_det_bound_l139_139910


namespace quadrant_of_alpha_l139_139284

theorem quadrant_of_alpha (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) : 
  (π / 2 < α ∧ α < π) := 
sorry

end quadrant_of_alpha_l139_139284


namespace apple_count_l139_139490

-- Define constants for the problem
constant bags: ℕ → Prop
constant total_apples: ℕ
constant apples_in_bag1: ℕ := 12
constant apples_in_bag2: ℕ := 6

-- Define the conditions
axiom condition1 (n: ℕ) : bags n → (n = apples_in_bag1 ∨ n = apples_in_bag2)
axiom condition2 : 70 ≤ total_apples ∧ total_apples ≤ 80
axiom condition3 : total_apples % 6 = 0

-- State the proof problem
theorem apple_count :
  total_apples = 72 ∨ total_apples = 78 :=
sorry

end apple_count_l139_139490


namespace halfway_between_one_fourth_and_one_seventh_l139_139679

theorem halfway_between_one_fourth_and_one_seventh : (1 / 4 + 1 / 7) / 2 = 11 / 56 := by
  sorry

end halfway_between_one_fourth_and_one_seventh_l139_139679


namespace total_books_of_gwen_l139_139703

theorem total_books_of_gwen 
  (mystery_shelves : ℕ) (picture_shelves : ℕ) (books_per_shelf : ℕ)
  (h1 : mystery_shelves = 3) (h2 : picture_shelves = 5) (h3 : books_per_shelf = 9) : 
  mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf = 72 :=
by
  -- Given:
  -- 1. Gwen had 3 shelves of mystery books.
  -- 2. Each shelf had 9 books.
  -- 3. Gwen had 5 shelves of picture books.
  -- 4. Each shelf had 9 books.
  -- Prove:
  -- The total number of books Gwen had is 72.
  sorry

end total_books_of_gwen_l139_139703


namespace Nicky_pace_5_mps_l139_139052

/-- Given the conditions:
  - Cristina runs at a pace of 5 meters per second.
  - Nicky runs for 30 seconds before Cristina catches up to him.
  Prove that Nicky’s pace is 5 meters per second. -/
theorem Nicky_pace_5_mps
  (Cristina_pace : ℝ)
  (time_Nicky : ℝ)
  (catchup : Cristina_pace * time_Nicky = 150)
  (def_Cristina_pace : Cristina_pace = 5)
  (def_time_Nicky : time_Nicky = 30) :
  (150 / 30) = 5 :=
by
  sorry

end Nicky_pace_5_mps_l139_139052


namespace perimeter_of_shaded_area_l139_139587

theorem perimeter_of_shaded_area (AB AD : ℝ) (h1 : AB = 14) (h2 : AD = 12) : 
  2 * AB + 2 * AD = 52 := 
by
  sorry

end perimeter_of_shaded_area_l139_139587


namespace factorial_product_trailing_zeros_l139_139072

def countTrailingZerosInFactorialProduct : ℕ :=
  let countFactorsOfFive (n : ℕ) : ℕ := 
    (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125) + (n / 15625) + (n / 78125) + (n / 390625) 
  List.range 100 -- Generates list [0, 1, ..., 99]
  |> List.map (fun k => countFactorsOfFive (k + 1)) -- Apply countFactorsOfFive to each k+1
  |> List.foldr (· + ·) 0 -- Sum all counts

theorem factorial_product_trailing_zeros : countTrailingZerosInFactorialProduct = 1124 := by
  sorry

end factorial_product_trailing_zeros_l139_139072


namespace sets_are_equal_l139_139963

def X : Set ℝ := {x | ∃ n : ℤ, x = (2 * n + 1) * Real.pi}
def Y : Set ℝ := {y | ∃ k : ℤ, y = (4 * k + 1) * Real.pi ∨ y = (4 * k - 1) * Real.pi}

theorem sets_are_equal : X = Y :=
by sorry

end sets_are_equal_l139_139963


namespace minimum_value_of_f_l139_139396

noncomputable def f (x : ℝ) : ℝ := sorry  -- define f such that f(x + 199) = 4x^2 + 4x + 3 for x ∈ ℝ

theorem minimum_value_of_f : ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ m = 2 := by
  sorry  -- Prove that the minimum value of f(x) is 2

end minimum_value_of_f_l139_139396


namespace binary_division_remainder_correct_l139_139971

-- Define the last two digits of the binary number
def b_1 : ℕ := 1
def b_0 : ℕ := 1

-- Define the function to calculate the remainder when dividing by 4
def binary_remainder (b1 b0 : ℕ) : ℕ := 2 * b1 + b0

-- Expected remainder in binary form
def remainder_in_binary : ℕ := 0b11  -- '11' in binary is 3 in decimal

-- The theorem to prove
theorem binary_division_remainder_correct :
  binary_remainder b_1 b_0 = remainder_in_binary :=
by
  -- Proof goes here
  sorry

end binary_division_remainder_correct_l139_139971


namespace equal_sums_of_squares_l139_139659

-- Define the coordinates of a rectangle in a 3D space.
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define vertices of the rectangle.
def A : Point3D := ⟨0, 0, 0⟩
def B (a : ℝ) : Point3D := ⟨a, 0, 0⟩
def C (a b : ℝ) : Point3D := ⟨a, b, 0⟩
def D (b : ℝ) : Point3D := ⟨0, b, 0⟩

-- Distance squared between two points in 3D space.
def distance_squared (M N : Point3D) : ℝ :=
  (M.x - N.x)^2 + (M.y - N.y)^2 + (M.z - N.z)^2

-- Prove that the sums of the squares of the distances between an arbitrary point M and opposite vertices of the rectangle are equal.
theorem equal_sums_of_squares (a b : ℝ) (M : Point3D) :
  distance_squared M A + distance_squared M (C a b) = distance_squared M (B a) + distance_squared M (D b) :=
by
  sorry

end equal_sums_of_squares_l139_139659


namespace eval_36_pow_five_over_two_l139_139375

theorem eval_36_pow_five_over_two : (36 : ℝ)^(5/2) = 7776 := by
  sorry

end eval_36_pow_five_over_two_l139_139375


namespace system_solution_unique_l139_139622

theorem system_solution_unique
  (a b m n : ℝ)
  (h1 : a * 1 + b * 2 = 10)
  (h2 : m * 1 - n * 2 = 8) :
  (a / 2 * (4 + -2) + b / 3 * (4 - -2) = 10) ∧
  (m / 2 * (4 + -2) - n / 3 * (4 - -2) = 8) := 
  by
    sorry

end system_solution_unique_l139_139622


namespace ninth_term_arithmetic_sequence_l139_139670

def first_term : ℚ := 3 / 4
def seventeenth_term : ℚ := 6 / 7

theorem ninth_term_arithmetic_sequence :
  let a1 := first_term
  let a17 := seventeenth_term
  (a1 + a17) / 2 = 45 / 56 := 
sorry

end ninth_term_arithmetic_sequence_l139_139670


namespace tangent_parabola_line_l139_139179

theorem tangent_parabola_line (a : ℝ) :
  (∃ x : ℝ, ax^2 + 1 = x ∧ ∀ y : ℝ, (y = ax^2 + 1 → y = x)) ↔ a = 1/4 :=
by
  sorry

end tangent_parabola_line_l139_139179


namespace num_distinct_ordered_pairs_l139_139140

theorem num_distinct_ordered_pairs (a b c : ℕ) (h₀ : a + b + c = 50) (h₁ : c = 10) (h₂ : 0 < a ∧ 0 < b) :
  ∃ n : ℕ, n = 39 := 
sorry

end num_distinct_ordered_pairs_l139_139140


namespace phil_quarters_l139_139805

variable (total_money pizza_cost soda_cost jeans_cost remaining_money_in_dollars : ℝ)
variable (num_quarters : ℕ)

def problem_conditions (total_money pizza_cost soda_cost jeans_cost : ℝ) : Prop :=
  total_money = 40 ∧ pizza_cost = 2.75 ∧ soda_cost = 1.50 ∧ jeans_cost = 11.50

theorem phil_quarters (total_money pizza_cost soda_cost jeans_cost remaining_money_in_dollars : ℝ)
                      (num_quarters : ℕ)
                      (h_cond : problem_conditions total_money pizza_cost soda_cost jeans_cost)
                      (h_remaining : remaining_money_in_dollars = total_money - (pizza_cost + soda_cost + jeans_cost))
                      (h_conversion : num_quarters = (remaining_money_in_dollars.to_nat * 4) + ((remaining_money_in_dollars - remaining_money_in_dollars.to_nat) * 4).to_nat) :
  num_quarters = 97 :=
by
  sorry

end phil_quarters_l139_139805


namespace find_other_number_l139_139464

theorem find_other_number (HCF LCM one_number other_number : ℤ)
  (hHCF : HCF = 12)
  (hLCM : LCM = 396)
  (hone_number : one_number = 48)
  (hrelation : HCF * LCM = one_number * other_number) :
  other_number = 99 :=
by
  sorry

end find_other_number_l139_139464


namespace bug_converges_to_final_position_l139_139568

noncomputable def bug_final_position : ℝ × ℝ := 
  let horizontal_sum := ∑' n, if n % 4 = 0 then (1 / 4) ^ (n / 4) else 0
  let vertical_sum := ∑' n, if n % 4 = 1 then (1 / 4) ^ (n / 4) else 0
  (horizontal_sum, vertical_sum)

theorem bug_converges_to_final_position : bug_final_position = (4 / 5, 2 / 5) := 
  sorry

end bug_converges_to_final_position_l139_139568


namespace last_digit_p_adic_l139_139653

theorem last_digit_p_adic (a : ℤ) (p : ℕ) (hp : Nat.Prime p) (h_last_digit_nonzero : a % p ≠ 0) : (a ^ (p - 1) - 1) % p = 0 :=
by
  sorry

end last_digit_p_adic_l139_139653


namespace circle_represents_circle_iff_a_nonzero_l139_139725

-- Define the equation given in the problem
def circleEquation (a x y : ℝ) : Prop :=
  a*x^2 + a*y^2 - 4*(a-1)*x + 4*y = 0

-- State the required theorem
theorem circle_represents_circle_iff_a_nonzero (a : ℝ) :
  (∃ c : ℝ, ∃ h k : ℝ, ∀ x y : ℝ, circleEquation a x y ↔ (x - h)^2 + (y - k)^2 = c)
  ↔ a ≠ 0 :=
by
  sorry

end circle_represents_circle_iff_a_nonzero_l139_139725


namespace mia_bought_more_pencils_l139_139048

theorem mia_bought_more_pencils (p : ℝ) (n1 n2 : ℕ) 
  (price_pos : p > 0.01)
  (liam_spent : 2.10 = p * n1)
  (mia_spent : 2.82 = p * n2) :
  (n2 - n1) = 12 := 
by
  sorry

end mia_bought_more_pencils_l139_139048


namespace least_possible_integer_l139_139226

theorem least_possible_integer (N : ℕ) :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 30 ∧ n ≠ 28 ∧ n ≠ 29 → n ∣ N) ∧
  (∀ m : ℕ, (∀ n : ℕ, 1 ≤ n ∧ n ≤ 30 ∧ n ≠ 28 ∧ n ≠ 29 → n ∣ m) → N ≤ m) →
  N = 2329089562800 :=
sorry

end least_possible_integer_l139_139226


namespace value_of_3y_l139_139400

theorem value_of_3y (x y z : ℤ) (h1 : x = z + 2) (h2 : y = z + 1) (h3 : 2 * x + 3 * y + 3 * z = 5 * y + 11) (h4 : z = 3) :
  3 * y = 12 :=
by
  sorry

end value_of_3y_l139_139400


namespace gcd_factorials_l139_139209

theorem gcd_factorials : Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  -- Proof steps would go here
  sorry

end gcd_factorials_l139_139209


namespace mixed_tea_sale_price_l139_139108

noncomputable def sale_price_of_mixed_tea (weight1 weight2 weight3 price1 price2 price3 profit1 profit2 profit3 : ℝ) : ℝ :=
  let total_cost1 := weight1 * price1
  let total_cost2 := weight2 * price2
  let total_cost3 := weight3 * price3
  let total_profit1 := profit1 * total_cost1
  let total_profit2 := profit2 * total_cost2
  let total_profit3 := profit3 * total_cost3
  let selling_price1 := total_cost1 + total_profit1
  let selling_price2 := total_cost2 + total_profit2
  let selling_price3 := total_cost3 + total_profit3
  let total_selling_price := selling_price1 + selling_price2 + selling_price3
  let total_weight := weight1 + weight2 + weight3
  total_selling_price / total_weight

theorem mixed_tea_sale_price :
  sale_price_of_mixed_tea 120 45 35 30 40 60 0.50 0.30 0.25 = 51.825 :=
by
  sorry

end mixed_tea_sale_price_l139_139108


namespace solution_set_for_log_inequality_l139_139440

noncomputable def f : ℝ → ℝ := sorry

def isEven (f : ℝ → ℝ) := ∀ x, f (-x) = f x

def isIncreasingOnNonNeg (f : ℝ → ℝ) := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

def f_positive_at_third : Prop := f (1 / 3) > 0

theorem solution_set_for_log_inequality
  (hf_even : isEven f)
  (hf_increasing : isIncreasingOnNonNeg f)
  (hf_positive : f_positive_at_third) :
  {x : ℝ | f (Real.log x / Real.log (1/8)) > 0} = {x : ℝ | 0 < x ∧ x < 1/2} ∪ {x : ℝ | 2 < x} := sorry

end solution_set_for_log_inequality_l139_139440


namespace prob_triangle_includes_G_l139_139916

-- Definitions based on conditions in the problem
def total_triangles : ℕ := 6
def triangles_including_G : ℕ := 4

-- The theorem statement proving the probability
theorem prob_triangle_includes_G : (triangles_including_G : ℚ) / total_triangles = 2 / 3 :=
by
  sorry

end prob_triangle_includes_G_l139_139916


namespace siblings_are_Emma_and_Olivia_l139_139911

structure Child where
  name : String
  eyeColor : String
  hairColor : String
  ageGroup : String

def Bella := Child.mk "Bella" "Green" "Red" "Older"
def Derek := Child.mk "Derek" "Gray" "Red" "Younger"
def Olivia := Child.mk "Olivia" "Green" "Brown" "Older"
def Lucas := Child.mk "Lucas" "Gray" "Brown" "Younger"
def Emma := Child.mk "Emma" "Green" "Red" "Older"
def Ryan := Child.mk "Ryan" "Gray" "Red" "Older"
def Sophia := Child.mk "Sophia" "Green" "Brown" "Younger"
def Ethan := Child.mk "Ethan" "Gray" "Brown" "Older"

def sharesCharacteristics (c1 c2 : Child) : Nat :=
  (if c1.eyeColor = c2.eyeColor then 1 else 0) +
  (if c1.hairColor = c2.hairColor then 1 else 0) +
  (if c1.ageGroup = c2.ageGroup then 1 else 0)

theorem siblings_are_Emma_and_Olivia :
  sharesCharacteristics Bella Emma ≥ 2 ∧
  sharesCharacteristics Bella Olivia ≥ 2 ∧
  (sharesCharacteristics Bella Derek < 2) ∧
  (sharesCharacteristics Bella Lucas < 2) ∧
  (sharesCharacteristics Bella Ryan < 2) ∧
  (sharesCharacteristics Bella Sophia < 2) ∧
  (sharesCharacteristics Bella Ethan < 2) :=
by
  sorry

end siblings_are_Emma_and_Olivia_l139_139911


namespace find_integer_roots_l139_139734

open Int Polynomial

def P (x : ℤ) : ℤ := x^3 - 3 * x^2 - 13 * x + 15

theorem find_integer_roots : {x : ℤ | P x = 0} = {-3, 1, 5} := by
  sorry

end find_integer_roots_l139_139734


namespace smaller_octagon_half_area_l139_139674

-- Define what it means to be a regular octagon
def is_regular_octagon (O : Point) (ABCDEFGH : List Point) : Prop :=
  -- Definition capturing the properties of a regular octagon around center O
  sorry

-- Define the function that computes the area of an octagon
def area_of_octagon (ABCDEFGH : List Point) : Real :=
  sorry

-- Define the function to create the smaller octagon by joining midpoints
def smaller_octagon (ABCDEFGH : List Point) : List Point :=
  sorry

theorem smaller_octagon_half_area (O : Point) (ABCDEFGH : List Point) :
  is_regular_octagon O ABCDEFGH →
  area_of_octagon (smaller_octagon ABCDEFGH) = (1 / 2) * area_of_octagon ABCDEFGH :=
by
  sorry

end smaller_octagon_half_area_l139_139674


namespace area_ratio_of_smaller_octagon_l139_139673

theorem area_ratio_of_smaller_octagon
    (A B C D E F G H : ℝ × ℝ) -- Coordinates of vertices of the larger octagon
    (P Q R S T U V W : ℝ × ℝ) -- Coordinates of vertices of the smaller octagon
    (regular_octagon : ∀ (X Y Z W U V T S : ℝ × ℝ), regular_octo X Y Z W U V T S)  -- Predicate for regular octagon
    (midpoints_joined : ∀ (X Y : ℝ × ℝ), midpoint X Y) : -- Condition that midpoints form the smaller octagon
  area (smaller_octo P Q R S T U V W) = (3 : ℝ) / 4 * area (larger_octo A B C D E F G H) :=
sorry

end area_ratio_of_smaller_octagon_l139_139673


namespace maryann_rescue_time_l139_139050

def time_to_free_cheaph (minutes : ℕ) : ℕ := 6
def time_to_free_expenh (minutes : ℕ) : ℕ := 8
def num_friends : ℕ := 3

theorem maryann_rescue_time : (time_to_free_cheaph 6 + time_to_free_expenh 8) * num_friends = 42 := 
by
  sorry

end maryann_rescue_time_l139_139050


namespace width_of_room_l139_139780

theorem width_of_room
  (carpet_has : ℕ)
  (room_length : ℕ)
  (carpet_needs : ℕ)
  (h1 : carpet_has = 18)
  (h2 : room_length = 4)
  (h3 : carpet_needs = 62) :
  (carpet_has + carpet_needs) = room_length * 20 :=
by
  sorry

end width_of_room_l139_139780


namespace find_k_l139_139607

theorem find_k (k b : ℤ) (h1 : -x^2 - (k + 10) * x - b = -(x - 2) * (x - 4))
  (h2 : b = 8) : k = -16 :=
sorry

end find_k_l139_139607


namespace range_of_a_l139_139896

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ 2 - a * x
noncomputable def g (x : ℝ) : ℝ := Real.exp x
noncomputable def h (x : ℝ) : ℝ := Real.log x

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 ∧ f x a = h x) →
  1 ≤ a ∧ a ≤ Real.exp 1 + 1 / Real.exp 1 :=
sorry

end range_of_a_l139_139896


namespace bear_small_animal_weight_l139_139097

theorem bear_small_animal_weight :
  let total_weight_needed := 1200
  let berries_weight := 1/5 * total_weight_needed
  let insects_weight := 1/10 * total_weight_needed
  let acorns_weight := 2 * berries_weight
  let honey_weight := 3 * insects_weight
  let total_weight_gained := berries_weight + insects_weight + acorns_weight + honey_weight
  let remaining_weight := total_weight_needed - total_weight_gained
  remaining_weight = 0 -> 0 = 0 := by
  intros total_weight_needed berries_weight insects_weight acorns_weight honey_weight
         total_weight_gained remaining_weight h
  exact Eq.refl 0

end bear_small_animal_weight_l139_139097


namespace expected_number_of_rounds_l139_139985

-- Define the game and its conditions
structure game :=
  (wins_A_odd : ℚ := 3 / 4)  -- Winning probability of Player A in odd rounds
  (wins_B_even : ℚ := 3 / 4) -- Winning probability of Player B in even rounds
  (no_ties : ∀ (n : ℕ), ¬(wins_A_odd = 1/2 ∧ wins_B_even = 1/2)) -- No ties in any round
  (end_condition : ∀ (a_wins b_wins : ℕ), abs (a_wins - b_wins) = 2 → game_terminated)

-- Define the expected number of rounds 
noncomputable def expected_rounds (g : game) : ℚ :=
sorry

-- Expected number of rounds statement
theorem expected_number_of_rounds (g : game) : expected_rounds g = 16 / 3 :=
sorry

end expected_number_of_rounds_l139_139985


namespace gcd_factorial_l139_139207

theorem gcd_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  sorry

end gcd_factorial_l139_139207


namespace movie_length_l139_139862

theorem movie_length (paused_midway : ∃ t : ℝ, t = t ∧ t / 2 = 30) : 
  ∃ total_length : ℝ, total_length = 60 :=
by {
  sorry
}

end movie_length_l139_139862


namespace taxi_ride_cost_l139_139236

-- Definitions based on the conditions
def fixed_cost : ℝ := 2.00
def variable_cost_per_mile : ℝ := 0.30
def distance_traveled : ℝ := 7

-- Theorem statement
theorem taxi_ride_cost : fixed_cost + (variable_cost_per_mile * distance_traveled) = 4.10 :=
by
  sorry

end taxi_ride_cost_l139_139236


namespace motherGaveMoney_l139_139088

-- Define the given constants and fact
def initialMoney : Real := 0.85
def foundMoney : Real := 0.50
def toyCost : Real := 1.60
def remainingMoney : Real := 0.15

-- Define the unknown amount given by his mother
def motherMoney (M : Real) := initialMoney + M + foundMoney - toyCost = remainingMoney

-- Statement to prove
theorem motherGaveMoney : ∃ M : Real, motherMoney M ∧ M = 0.40 :=
by
  sorry

end motherGaveMoney_l139_139088


namespace top_leftmost_rectangle_is_B_l139_139374

-- Definitions for the side lengths of each rectangle
def A_w : ℕ := 6
def A_x : ℕ := 2
def A_y : ℕ := 7
def A_z : ℕ := 10

def B_w : ℕ := 2
def B_x : ℕ := 1
def B_y : ℕ := 4
def B_z : ℕ := 8

def C_w : ℕ := 5
def C_x : ℕ := 11
def C_y : ℕ := 6
def C_z : ℕ := 3

def D_w : ℕ := 9
def D_x : ℕ := 7
def D_y : ℕ := 5
def D_z : ℕ := 9

def E_w : ℕ := 11
def E_x : ℕ := 4
def E_y : ℕ := 9
def E_z : ℕ := 1

-- The problem statement to prove
theorem top_leftmost_rectangle_is_B : 
  (B_w = 2 ∧ B_y = 4) ∧ 
  (A_w = 6 ∨ D_w = 9 ∨ C_w = 5 ∨ E_w = 11) ∧
  (A_y = 7 ∨ D_y = 5 ∨ C_y = 6 ∨ E_y = 9) → 
  (B_w = 2 ∧ ∀ w : ℕ, w = 6 ∨ w = 5 ∨ w = 9 ∨ w = 11 → B_w < w) :=
by {
  -- skipping the proof
  sorry
}

end top_leftmost_rectangle_is_B_l139_139374


namespace inv_seq_not_arith_seq_l139_139891

theorem inv_seq_not_arith_seq (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_arith : ∃ d : ℝ, d ≠ 0 ∧ b = a + d ∧ c = a + 2 * d) :
  ¬ ∃ d' : ℝ, ∀ i j k : ℝ, i = 1 / a → j = 1 / b → k = 1 / c → j - i = d' ∧ k - j = d' :=
sorry

end inv_seq_not_arith_seq_l139_139891


namespace circles_intersect_in_two_points_l139_139410

def circle1 (x y : ℝ) : Prop := x^2 + (y - 3/2)^2 = (3/2)^2
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

theorem circles_intersect_in_two_points :
  ∃! (p : ℝ × ℝ), (circle1 p.1 p.2) ∧ (circle2 p.1 p.2) := 
sorry

end circles_intersect_in_two_points_l139_139410


namespace jason_seashells_remaining_l139_139922

-- Define the initial number of seashells Jason found
def initial_seashells : ℕ := 49

-- Define the number of seashells Jason gave to Tim
def seashells_given_to_tim : ℕ := 13

-- Define the number of seashells Jason now has
def seashells_now : ℕ := initial_seashells - seashells_given_to_tim

-- The theorem to prove: 
theorem jason_seashells_remaining : seashells_now = 36 := 
by
  -- Proof steps will go here
  sorry

end jason_seashells_remaining_l139_139922


namespace range_of_m_l139_139046

def M := {y : ℝ | ∃ (x : ℝ), y = (1/2)^x}
def N (m : ℝ) := {y : ℝ | ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 2 ∧ y = ((1/(m-1) + 1) * (x - 1) + (|m| - 1) * (x - 2))}

theorem range_of_m (m : ℝ) : (∀ y ∈ N m, y ∈ M) ↔ -1 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l139_139046


namespace value_of_expression_l139_139138

noncomputable def proof_problem (a b c d : ℝ) : Prop :=
  (a + b + c + d).sqrt + (a^2 - 2*a + 3 - b).sqrt - (b - c^2 + 4*c - 8).sqrt = 3

theorem value_of_expression (a b c d : ℝ) (h : proof_problem a b c d) : a - b + c - d = -7 :=
sorry

end value_of_expression_l139_139138


namespace gcd_of_factorials_l139_139204

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Definitions for the two factorial terms involved
def term1 : ℕ := factorial 7
def term2 : ℕ := factorial 10 / factorial 4

-- Definition for computing the GCD
def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

-- The statement to prove
theorem gcd_of_factorials : gcd term1 term2 = 2520 :=
by {
  -- Here, the proof would go.
  sorry
}

end gcd_of_factorials_l139_139204


namespace solution_l139_139810

noncomputable theory

def problem_statement : Prop :=
  ∀ (x : ℝ), (real.arctan (1 / x) + real.arctan (1 / x^2) = π / 4) → x = 2

theorem solution : problem_statement :=
  by
    sorry

end solution_l139_139810


namespace max_b_in_box_l139_139475

theorem max_b_in_box (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b ≤ 12 := 
by
  sorry

end max_b_in_box_l139_139475


namespace ball_bounce_height_l139_139989

theorem ball_bounce_height (b : ℕ) : 
  ∃ b : ℕ, 400 * (3 / 4 : ℝ)^b < 50 ∧ ∀ b' < b, 400 * (3 / 4 : ℝ)^b' ≥ 50 :=
sorry

end ball_bounce_height_l139_139989


namespace stock_price_end_second_year_l139_139373

theorem stock_price_end_second_year
  (P₀ : ℝ) (r₁ r₂ : ℝ) 
  (h₀ : P₀ = 150)
  (h₁ : r₁ = 0.80)
  (h₂ : r₂ = 0.30) :
  let P₁ := P₀ + r₁ * P₀
  let P₂ := P₁ - r₂ * P₁
  P₂ = 189 :=
by
  sorry

end stock_price_end_second_year_l139_139373


namespace apple_bags_l139_139498

theorem apple_bags (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apple_bags_l139_139498


namespace opposite_terminal_sides_l139_139635

theorem opposite_terminal_sides (α β : ℝ) (k : ℤ) (h : ∃ k : ℤ, α = β + 180 + k * 360) :
  α = β + 180 + k * 360 :=
by sorry

end opposite_terminal_sides_l139_139635


namespace coupon_savings_difference_l139_139711

theorem coupon_savings_difference {P : ℝ} (hP : P > 200)
  (couponA_savings : ℝ := 0.20 * P) 
  (couponB_savings : ℝ := 50)
  (couponC_savings : ℝ := 0.30 * (P - 200)) :
  (200 ≤ P - 200 + 50 → 200 ≤ P ∧ P ≤ 200 + 400 → 600 - 250 = 350) :=
by
  sorry

end coupon_savings_difference_l139_139711


namespace monotonically_increasing_condition_l139_139745

theorem monotonically_increasing_condition 
  (a b c d : ℝ) (h : 0 < a) :
  (∀ x : ℝ, 0 ≤ 3 * a * x ^ 2 + 2 * b * x + c) ↔ (b^2 - 3 * a * c ≤ 0) :=
by {
  sorry
}

end monotonically_increasing_condition_l139_139745


namespace ab5_a2_c5_a2_inequality_l139_139045

theorem ab5_a2_c5_a2_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a ^ 5 - a ^ 2 + 3) * (b ^ 5 - b ^ 2 + 3) * (c ^ 5 - c ^ 2 + 3) ≥ (a + b + c) ^ 3 := 
by
  sorry

end ab5_a2_c5_a2_inequality_l139_139045


namespace negation_of_exists_proposition_l139_139931

theorem negation_of_exists_proposition :
  (¬ ∃ n : ℕ, n^2 > 2^n) → (∀ n : ℕ, n^2 ≤ 2^n) := 
by 
  sorry

end negation_of_exists_proposition_l139_139931


namespace ratio_of_jars_to_pots_l139_139991

theorem ratio_of_jars_to_pots 
  (jars : ℕ)
  (pots : ℕ)
  (k : ℕ)
  (marbles_total : ℕ)
  (h1 : jars = 16)
  (h2 : jars = k * pots)
  (h3 : ∀ j, j = 5)
  (h4 : ∀ p, p = 15)
  (h5 : marbles_total = 200) :
  (jars / pots = 2) :=
by
  sorry

end ratio_of_jars_to_pots_l139_139991


namespace find_f_neg_5_l139_139397

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_domain : ∀ x : ℝ, true)
variable (h_positive : ∀ x : ℝ, x > 0 → f x = log 5 x + 1)

theorem find_f_neg_5 : f (-5) = -2 :=
by
  sorry

end find_f_neg_5_l139_139397


namespace total_time_spent_l139_139122

-- Define the conditions
def number_of_chairs : ℕ := 4
def number_of_tables : ℕ := 2
def time_per_piece : ℕ := 8

-- Prove that the total time spent is 48 minutes
theorem total_time_spent : (number_of_chairs + number_of_tables) * time_per_piece = 48 :=
by
  sorry

end total_time_spent_l139_139122


namespace total_cost_is_9220_l139_139355

-- Define the conditions
def hourly_rate := 60
def hours_per_day := 8
def total_days := 14
def cost_of_parts := 2500

-- Define the total cost the car's owner had to pay based on conditions
def total_hours := hours_per_day * total_days
def labor_cost := total_hours * hourly_rate
def total_cost := labor_cost + cost_of_parts

-- Theorem stating that the total cost is $9220
theorem total_cost_is_9220 : total_cost = 9220 := by
  sorry

end total_cost_is_9220_l139_139355


namespace shortest_distance_between_circles_l139_139342

noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₂.1 - p₁.1) ^ 2 + (p₂.2 - p₁.2) ^ 2)

theorem shortest_distance_between_circles :
  let c₁ := (4, -3)
  let r₁ := 4
  let c₂ := (-5, 1)
  let r₂ := 1
  distance c₁ c₂ - (r₁ + r₂) = Real.sqrt 97 - 5 :=
by
  sorry

end shortest_distance_between_circles_l139_139342


namespace caterpillar_reaches_top_in_16_days_l139_139225

-- Define the constants for the problem
def pole_height : ℕ := 20
def daytime_climb : ℕ := 5
def nighttime_slide : ℕ := 4

-- Define the final result we want to prove
theorem caterpillar_reaches_top_in_16_days :
  ∃ days : ℕ, days = 16 ∧ 
  ((20 - 5) / (daytime_climb - nighttime_slide) + 1) = 16 := by
  sorry

end caterpillar_reaches_top_in_16_days_l139_139225


namespace apple_bags_l139_139525

theorem apple_bags (n : ℕ) (h₁ : n ≥ 70) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 := 
sorry

end apple_bags_l139_139525


namespace geometricSeqMinimumValue_l139_139777

noncomputable def isMinimumValue (a : ℕ → ℝ) (n m : ℕ) (value : ℝ) : Prop :=
  ∀ b : ℝ, (1 / a n + b / a m) ≥ value

theorem geometricSeqMinimumValue {a : ℕ → ℝ}
  (h1 : ∀ n, a n > 0)
  (h2 : a 7 = (Real.sqrt 2) / 2)
  (h3 : ∀ n, ∀ m, a n * a m = a (n + m)) :
  isMinimumValue a 3 11 4 :=
sorry

end geometricSeqMinimumValue_l139_139777


namespace find_age_l139_139976

theorem find_age (x : ℕ) (h : 5 * (x + 5) - 5 * (x - 5) = x) : x = 50 :=
by
  sorry

end find_age_l139_139976


namespace symmetrical_circle_equation_l139_139471

theorem symmetrical_circle_equation :
  ∀ (x y : ℝ), (x^2 + y^2 - 2 * x - 1 = 0) ∧ (2 * x - y + 1 = 0) →
  ((x + 7/5)^2 + (y - 6/5)^2 = 2) :=
sorry

end symmetrical_circle_equation_l139_139471


namespace trackball_mice_count_l139_139310

theorem trackball_mice_count 
  (total_mice wireless_mice optical_mice trackball_mice : ℕ)
  (h1 : total_mice = 80)
  (h2 : wireless_mice = total_mice / 2)
  (h3 : optical_mice = total_mice / 4)
  (h4 : trackball_mice = total_mice - (wireless_mice + optical_mice)) :
  trackball_mice = 20 := by 
  sorry

end trackball_mice_count_l139_139310


namespace needed_adjustment_l139_139809

def price_adjustment (P : ℝ) : ℝ :=
  let P_reduced := P - 0.20 * P
  let P_raised := P_reduced + 0.10 * P_reduced
  let P_target := P - 0.10 * P
  P_target - P_raised

theorem needed_adjustment (P : ℝ) : price_adjustment P = 2 * (P / 100) := sorry

end needed_adjustment_l139_139809


namespace average_speed_correct_l139_139978

noncomputable def average_speed (d v_up v_down : ℝ) : ℝ :=
  let t_up := d / v_up
  let t_down := d / v_down
  let total_distance := 2 * d
  let total_time := t_up + t_down
  total_distance / total_time

theorem average_speed_correct :
  average_speed 0.2 24 36 = 28.8 := by {
  sorry
}

end average_speed_correct_l139_139978


namespace dice_sum_is_4_l139_139688

-- Defining the sum of points obtained from two dice rolls
def sum_of_dice (a b : ℕ) : ℕ := a + b

-- The main theorem stating the condition we need to prove
theorem dice_sum_is_4 (a b : ℕ) (h : sum_of_dice a b = 4) :
  (a = 3 ∧ b = 1) ∨ (a = 1 ∧ b = 3) ∨ (a = 2 ∧ b = 2) :=
sorry

end dice_sum_is_4_l139_139688


namespace solve_equation_l139_139257

-- Definitions based on the conditions
def equation (a b c d : ℕ) : Prop :=
  2^a * 3^b - 5^c * 7^d = 1

def nonnegative_integers (a b c d : ℕ) : Prop := 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0

-- Proof to show the exact solutions
theorem solve_equation :
  (∃ (a b c d : ℕ), nonnegative_integers a b c d ∧ equation a b c d) ↔ 
  ( (1, 0, 0, 0) = (1, 0, 0, 0) ∨ (3, 0, 0, 1) = (3, 0, 0, 1) ∨ 
    (1, 1, 1, 0) = (1, 1, 1, 0) ∨ (2, 2, 1, 1) = (2, 2, 1, 1) ) := by
  sorry

end solve_equation_l139_139257


namespace num_pens_l139_139352

theorem num_pens (pencils : ℕ) (students : ℕ) (pens : ℕ)
  (h_pencils : pencils = 520)
  (h_students : students = 40)
  (h_div : pencils % students = 0)
  (h_pens_per_student : pens = (pencils / students) * students) :
  pens = 520 := by
  sorry

end num_pens_l139_139352


namespace probability_at_least_three_heads_l139_139463

def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem probability_at_least_three_heads :
  (∑ k in {3, 4, 5}, binomial 5 k) = 16 → (16 / 32 = 1 / 2) :=
by
  sorry

end probability_at_least_three_heads_l139_139463


namespace ken_gave_manny_10_pencils_l139_139438

theorem ken_gave_manny_10_pencils (M : ℕ) 
  (ken_pencils : ℕ := 50)
  (ken_kept : ℕ := 20)
  (ken_distributed : ℕ := ken_pencils - ken_kept)
  (nilo_pencils : ℕ := M + 10)
  (distribution_eq : M + nilo_pencils = ken_distributed) : 
  M = 10 :=
by
  sorry

end ken_gave_manny_10_pencils_l139_139438


namespace Jason_spent_correct_amount_l139_139779

def flute_cost : ℝ := 142.46
def music_stand_cost : ℝ := 8.89
def song_book_cost : ℝ := 7.00
def total_cost : ℝ := 158.35

theorem Jason_spent_correct_amount :
  flute_cost + music_stand_cost + song_book_cost = total_cost :=
by
  sorry

end Jason_spent_correct_amount_l139_139779


namespace free_endpoints_eq_1001_l139_139748

theorem free_endpoints_eq_1001 : 
  ∃ k : ℕ, 1 + 4 * k = 1001 :=
by {
  sorry
}

end free_endpoints_eq_1001_l139_139748


namespace oil_to_water_ratio_in_bottle_D_l139_139077

noncomputable def bottle_oil_water_ratio (CA : ℝ) (CB : ℝ) (CC : ℝ) (CD : ℝ) : ℝ :=
  let oil_A := (1 / 2) * CA
  let water_A := (1 / 2) * CA
  let oil_B := (1 / 4) * CB
  let water_B := (1 / 4) * CB
  let total_water_B := CB - oil_B - water_B
  let oil_C := (1 / 3) * CC
  let water_C := 0.4 * CC
  let total_water_C := CC - oil_C - water_C
  let total_capacity_D := CD
  let total_oil_D := oil_A + oil_B + oil_C
  let total_water_D := water_A + total_water_B + water_C + total_water_C
  total_oil_D / total_water_D

theorem oil_to_water_ratio_in_bottle_D (CA : ℝ) :
  let CB := 2 * CA
  let CC := 3 * CA
  let CD := CA + CC
  bottle_oil_water_ratio CA CB CC CD = (2 / 3.7) :=
by 
  sorry

end oil_to_water_ratio_in_bottle_D_l139_139077


namespace simplify_complex_expr_l139_139954

theorem simplify_complex_expr : ∀ (i : ℂ), (4 - 2 * i) - (7 - 2 * i) + (6 - 3 * i) = 3 - 3 * i := by
  intro i
  sorry

end simplify_complex_expr_l139_139954


namespace height_of_fourth_person_l139_139221

theorem height_of_fourth_person 
  (H : ℕ) 
  (h_avg : ((H) + (H + 2) + (H + 4) + (H + 10)) / 4 = 79) :
  (H + 10 = 85) :=
by
  sorry

end height_of_fourth_person_l139_139221


namespace xyz_sum_fraction_l139_139124

theorem xyz_sum_fraction (a1 a2 a3 b1 b2 b3 c1 c2 c3 a b c : ℤ) 
  (h1 : a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1) = 9)
  (h2 : a * (b2 * c3 - b3 * c2) - a2 * (b * c3 - b3 * c) + a3 * (b * c2 - b2 * c) = 17)
  (h3 : a1 * (b * c3 - b3 * c) - a * (b1 * c3 - b3 * c1) + a3 * (b1 * c - b * c1) = -8)
  (h4 : a1 * (b2 * c - b * c2) - a2 * (b1 * c - b * c1) + a * (b1 * c2 - b2 * c1) = 7)
  (eq1 : a1 * x + a2 * y + a3 * z = a)
  (eq2 : b1 * x + b2 * y + b3 * z = b)
  (eq3 : c1 * x + c2 * y + c3 * z = c)
  : x + y + z = 16 / 9 := 
sorry

end xyz_sum_fraction_l139_139124


namespace eq1_solutions_eq2_solutions_l139_139458

theorem eq1_solutions (x : ℝ) : x^2 - 6 * x + 3 = 0 ↔ (x = 3 + Real.sqrt 6) ∨ (x = 3 - Real.sqrt 6) :=
by {
  sorry
}

theorem eq2_solutions (x : ℝ) : x * (x - 2) = x - 2 ↔ (x = 2) ∨ (x = 1) :=
by {
  sorry
}

end eq1_solutions_eq2_solutions_l139_139458


namespace possible_apple_counts_l139_139504

theorem possible_apple_counts (n : ℕ) (h₁ : 70 ≤ n) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end possible_apple_counts_l139_139504


namespace termites_ate_black_squares_l139_139663

def chessboard_black_squares_eaten : Nat :=
  12

theorem termites_ate_black_squares :
  let rows := 8;
  let cols := 8;
  let total_squares := rows * cols / 2; -- This simplistically assumes half the squares are black.
  (total_squares = 32) → 
  chessboard_black_squares_eaten = 12 :=
by
  intros h
  sorry

end termites_ate_black_squares_l139_139663


namespace largest_common_term_l139_139586

-- Definitions for the first arithmetic sequence
def arithmetic_seq1 (n : ℕ) : ℕ := 2 + 5 * n

-- Definitions for the second arithmetic sequence
def arithmetic_seq2 (m : ℕ) : ℕ := 5 + 8 * m

-- Main statement of the problem
theorem largest_common_term (n m k : ℕ) (a : ℕ) :
  (a = arithmetic_seq1 n) ∧ (a = arithmetic_seq2 m) ∧ (1 ≤ a) ∧ (a ≤ 150) →
  a = 117 :=
by {
  sorry
}

end largest_common_term_l139_139586


namespace sqrt_of_0_01_l139_139823

theorem sqrt_of_0_01 : Real.sqrt 0.01 = 0.1 :=
by
  sorry

end sqrt_of_0_01_l139_139823


namespace apple_bags_l139_139493

theorem apple_bags (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apple_bags_l139_139493


namespace rectangle_area_l139_139161

theorem rectangle_area (w L : ℝ) (h1 : L = w^2) (h2 : L + w = 25) : 
  L * w = (Real.sqrt 101 - 1)^3 / 8 := 
sorry

end rectangle_area_l139_139161


namespace initial_games_count_l139_139647

-- Definitions used in conditions
def games_given_away : ℕ := 99
def games_left : ℝ := 22.0

-- Theorem statement for the initial number of games
theorem initial_games_count : games_given_away + games_left = 121.0 := by
  sorry

end initial_games_count_l139_139647


namespace wheels_on_floor_l139_139802

def number_of_wheels (n_people : Nat) (w_per_person : Nat) : Nat :=
  n_people * w_per_person

theorem wheels_on_floor (n_people : Nat) (w_per_person : Nat) (h_people : n_people = 40) (h_wheels : w_per_person = 4) :
  number_of_wheels n_people w_per_person = 160 := by
  sorry

end wheels_on_floor_l139_139802


namespace lcm_of_ratio_hcf_l139_139681

theorem lcm_of_ratio_hcf {a b : ℕ} (ratioCond : a = 14 * 28) (ratioCond2 : b = 21 * 28) (hcfCond : Nat.gcd a b = 28) : Nat.lcm a b = 1176 := by
  sorry

end lcm_of_ratio_hcf_l139_139681


namespace probability_odd_product_greater_than_15_l139_139059

-- Definitions of conditions
def balls := {1, 2, 3, 4, 5, 6}

-- Lean proof problem statement
theorem probability_odd_product_greater_than_15 :
  let outcomes : Finset (ℕ × ℕ) := Finset.product (Finset.of_set balls) (Finset.of_set balls),
  odds := {n ∈ balls | n % 2 = 1},
  success := ((5, 5) ∈ outcomes)
  in ((Finset.card success).toNat : ℚ) / (Finset.card outcomes).toNat = 1 / 36 := by 
sorry

end probability_odd_product_greater_than_15_l139_139059


namespace tiffany_max_points_l139_139538

section
  variables
  (initial_money : ℕ := 3)
  (cost_per_game : ℕ := 1)
  (rings_per_game : ℕ := 5)
  (points_red_bucket : ℕ := 2)
  (points_green_bucket : ℕ := 3)
  (points_miss : ℕ := 0)
  (games_played : ℕ := 2)
  (red_buckets : ℕ := 4)
  (green_buckets : ℕ := 5)
  (additional_games : ℕ := initial_money - games_played)
  (points_per_game_from_green_buckets : ℕ := rings_per_game * points_green_bucket)
  (total_points : ℕ := (red_buckets * points_red_bucket) + (green_buckets * points_green_bucket) + (additional_games * points_per_game_from_green_buckets))

  theorem tiffany_max_points : total_points = 38 := 
  sorry
end

end tiffany_max_points_l139_139538


namespace gcd_factorials_l139_139210

theorem gcd_factorials : Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  -- Proof steps would go here
  sorry

end gcd_factorials_l139_139210


namespace handshake_problem_l139_139111

theorem handshake_problem (x : ℕ) (hx : (x * (x - 1)) / 2 = 55) : x = 11 := 
sorry

end handshake_problem_l139_139111


namespace compute_expression_l139_139368

theorem compute_expression : 12 * (1 / 15) * 30 = 24 := 
by 
  sorry

end compute_expression_l139_139368


namespace pedestrians_speed_ratio_l139_139200

-- Definitions based on conditions
variable (v v1 v2 : ℝ)

-- Conditions
def first_meeting (v1 v : ℝ) := (1 / 3) * v1 = (1 / 4) * v
def second_meeting (v2 v : ℝ) := (5 / 12) * v2 = (1 / 6) * v

-- Theorem Statement
theorem pedestrians_speed_ratio (h1 : first_meeting v1 v) (h2 : second_meeting v2 v) : v1 / v2 = 15 / 8 :=
by
  -- Proof will go here
  sorry

end pedestrians_speed_ratio_l139_139200


namespace find_f_2021_l139_139816

noncomputable def f (x : ℝ) : ℝ := sorry

lemma functional_equation (a b : ℝ) : f ((a + 2 * b) / 3) = (f a + 2 * f b) / 3 :=
sorry

lemma f_one : f 1 = 1 :=
sorry

lemma f_four : f 4 = 7 :=
sorry

theorem find_f_2021 : f 2021 = 4041 :=
sorry

end find_f_2021_l139_139816


namespace uncle_welly_roses_l139_139199

theorem uncle_welly_roses :
  let roses_two_days_ago := 50
  let roses_yesterday := roses_two_days_ago + 20
  let roses_today := 2 * roses_two_days_ago
  roses_two_days_ago + roses_yesterday + roses_today = 220 :=
by
  let roses_two_days_ago := 50
  let roses_yesterday := roses_two_days_ago + 20
  let roses_today := 2 * roses_two_days_ago
  show roses_two_days_ago + roses_yesterday + roses_today = 220
  sorry

end uncle_welly_roses_l139_139199


namespace parabola_range_l139_139189

theorem parabola_range (x : ℝ) (h : 0 < x ∧ x < 3) : 
  1 ≤ (x^2 - 4*x + 5) ∧ (x^2 - 4*x + 5) < 5 :=
sorry

end parabola_range_l139_139189


namespace problem1_problem2_problem3_l139_139772

-- Problem 1
theorem problem1 (m : ℝ) (h : -m^2 = m) : m^2 + m + 1 = 1 :=
by sorry

-- Problem 2
theorem problem2 (m n : ℝ) (h : m - n = 2) : 2 * (n - m) - 4 * m + 4 * n - 3 = -15 :=
by sorry

-- Problem 3
theorem problem3 (m n : ℝ) (h1 : m^2 + 2 * m * n = -2) (h2 : m * n - n^2 = -4) : 
  3 * m^2 + (9 / 2) * m * n + (3 / 2) * n^2 = 0 :=
by sorry

end problem1_problem2_problem3_l139_139772


namespace sum_of_four_digit_numbers_l139_139740

theorem sum_of_four_digit_numbers :
  let digits := [2, 4, 5, 3]
  let factorial := Nat.factorial (List.length digits)
  let each_appearance := factorial / (List.length digits)
  (each_appearance * (2 + 4 + 5 + 3) * (1000 + 100 + 10 + 1)) = 93324 :=
by
  let digits := [2, 4, 5, 3]
  let factorial := Nat.factorial (List.length digits)
  let each_appearance := factorial / (List.length digits)
  show (each_appearance * (2 + 4 + 5 + 3) * (1000 + 100 + 10 + 1)) = 93324
  sorry

end sum_of_four_digit_numbers_l139_139740


namespace Willy_Lucy_more_crayons_l139_139698

def Willy_initial : ℕ := 1400
def Lucy_initial : ℕ := 290
def Max_crayons : ℕ := 650
def Willy_giveaway_percent : ℚ := 25 / 100
def Lucy_giveaway_percent : ℚ := 10 / 100

theorem Willy_Lucy_more_crayons :
  let Willy_remaining := Willy_initial - Willy_initial * Willy_giveaway_percent
  let Lucy_remaining := Lucy_initial - Lucy_initial * Lucy_giveaway_percent
  Willy_remaining + Lucy_remaining - Max_crayons = 661 := by
  sorry

end Willy_Lucy_more_crayons_l139_139698


namespace minimum_value_frac_sum_l139_139883

theorem minimum_value_frac_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 2 / y = 3) :
  (2 / x + y) ≥ 8 / 3 :=
sorry

end minimum_value_frac_sum_l139_139883


namespace quadratic_roots_expression_l139_139390

theorem quadratic_roots_expression :
  ∀ (x₁ x₂ : ℝ), 
  (x₁ + x₂ = 3) →
  (x₁ * x₂ = -1) →
  (x₁^2 * x₂ + x₁ * x₂^2 = -3) :=
by
  intros x₁ x₂ h1 h2
  sorry

end quadratic_roots_expression_l139_139390


namespace real_roots_quadratic_l139_139420

theorem real_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, (k - 2) * x^2 - 2 * k * x + k - 6 = 0) ↔ (k ≥ 1.5 ∧ k ≠ 2) :=
by {
  sorry
}

end real_roots_quadratic_l139_139420


namespace intersect_of_given_circles_l139_139145

noncomputable def circle_center (a b c : ℝ) : ℝ × ℝ :=
  let x := -a / 2
  let y := -b / 2
  (x, y)

noncomputable def radius_squared (a b c : ℝ) : ℝ :=
  (a / 2) ^ 2 + (b / 2) ^ 2 - c

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def circles_intersect (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  let center1 := circle_center a1 b1 c1
  let center2 := circle_center a2 b2 c2
  let r1 := Real.sqrt (radius_squared a1 b1 c1)
  let r2 := Real.sqrt (radius_squared a2 b2 c2)
  let d := distance center1 center2
  r1 - r2 < d ∧ d < r1 + r2

theorem intersect_of_given_circles :
  circles_intersect 4 3 2 2 3 1 :=
sorry

end intersect_of_given_circles_l139_139145


namespace percent_of_200_is_400_when_whole_is_50_l139_139984

theorem percent_of_200_is_400_when_whole_is_50 (Part Whole : ℕ) (hPart : Part = 200) (hWhole : Whole = 50) :
  (Part / Whole) * 100 = 400 :=
by {
  -- Proof steps go here.
  sorry
}

end percent_of_200_is_400_when_whole_is_50_l139_139984


namespace min_value_ineq_solve_ineq_l139_139270

theorem min_value_ineq (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / a^3 + 1 / b^3 + 1 / c^3 + 3 * a * b * c) ≥ 6 :=
sorry

theorem solve_ineq (x : ℝ) (h : |x + 1| - 2 * x < 6) : x > -7/3 :=
sorry

end min_value_ineq_solve_ineq_l139_139270


namespace find_abcdef_l139_139956

def repeating_decimal_to_fraction_abcd (a b c d : ℕ) : ℚ :=
  (1000 * a + 100 * b + 10 * c + d) / 9999

def repeating_decimal_to_fraction_abcdef (a b c d e f : ℕ) : ℚ :=
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) / 999999

theorem find_abcdef :
  ∀ a b c d e f : ℕ,
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  0 ≤ e ∧ e ≤ 9 ∧
  0 ≤ f ∧ f ≤ 9 ∧
  (repeating_decimal_to_fraction_abcd a b c d + repeating_decimal_to_fraction_abcdef a b c d e f = 49 / 999) →
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f = 490) :=
by
  repeat {sorry}

end find_abcdef_l139_139956


namespace factory_workers_count_l139_139913

theorem factory_workers_count :
  ∃ (F S_f : ℝ), 
    (F * S_f = 30000) ∧ 
    (30 * (S_f + 500) = 75000) → 
    (F = 15) :=
by
  sorry

end factory_workers_count_l139_139913


namespace bean_inside_inscribed_circle_l139_139328

noncomputable def equilateral_triangle_area (a : ℝ) : ℝ :=
  (Real.sqrt 3 / 4) * a * a

noncomputable def inscribed_circle_radius (a : ℝ) : ℝ :=
  (Real.sqrt 3 / 3) * a

noncomputable def circle_area (r : ℝ) : ℝ :=
  Real.pi * r * r

noncomputable def probability_inside_circle (s_triangle s_circle : ℝ) : ℝ :=
  s_circle / s_triangle

theorem bean_inside_inscribed_circle :
  let a := 2
  let s_triangle := equilateral_triangle_area a
  let r := inscribed_circle_radius a
  let s_circle := circle_area r
  probability_inside_circle s_triangle s_circle = (Real.sqrt 3 * Real.pi / 9) :=
by
  sorry

end bean_inside_inscribed_circle_l139_139328


namespace kenya_peanut_count_l139_139783

-- Define the number of peanuts Jose has
def jose_peanuts : ℕ := 85

-- Define the number of additional peanuts Kenya has more than Jose
def additional_peanuts : ℕ := 48

-- Define the number of peanuts Kenya has
def kenya_peanuts : ℕ := jose_peanuts + additional_peanuts

-- Theorem to prove the number of peanuts Kenya has
theorem kenya_peanut_count : kenya_peanuts = 133 := by
  sorry

end kenya_peanut_count_l139_139783


namespace final_number_proof_l139_139323

/- Define the symbols and their corresponding values -/
def cat := 1
def chicken := 5
def crab := 2
def bear := 4
def goat := 3

/- Define the equations from the conditions -/
axiom row4_eq : 5 * crab = 10
axiom col5_eq : 4 * crab + goat = 11
axiom row2_eq : 2 * goat + crab + 2 * bear = 16
axiom col2_eq : cat + bear + 2 * goat + crab = 13
axiom col3_eq : 2 * crab + 2 * chicken + goat = 17

/- Final number is derived by concatenating digits -/
def final_number := cat * 10000 + chicken * 1000 + crab * 100 + bear * 10 + goat

/- Theorem to prove the final number is 15243 -/
theorem final_number_proof : final_number = 15243 := by
  -- Proof steps to be provided here.
  sorry

end final_number_proof_l139_139323


namespace apple_count_l139_139484

/--
Given the total number of apples in several bags, where each bag contains either 12 or 6 apples,
and knowing that the total number of apples is between 70 and 80 inclusive,
prove that the total number of apples can only be 72 or 78.
-/
theorem apple_count (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : ∃ k : ℕ, n = 6 * k) : n = 72 ∨ n = 78 :=
by {
  sorry
}

end apple_count_l139_139484


namespace sum_of_squares_ne_sum_of_fourth_powers_l139_139660

theorem sum_of_squares_ne_sum_of_fourth_powers :
  ∀ (a b : ℤ), a^2 + (a + 1)^2 ≠ b^4 + (b + 1)^4 :=
by 
  sorry

end sum_of_squares_ne_sum_of_fourth_powers_l139_139660


namespace break_even_performances_l139_139460

def totalCost (x : ℕ) : ℕ := 81000 + 7000 * x
def totalRevenue (x : ℕ) : ℕ := 16000 * x

theorem break_even_performances : ∃ x : ℕ, totalCost x = totalRevenue x ∧ x = 9 := 
by
  sorry

end break_even_performances_l139_139460


namespace inequality_inequality_hold_l139_139034

theorem inequality_inequality_hold (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hab_sum : a^2 + b^2 = 1/2) :
  (1 / (1 - a)) + (1 / (1 - b)) ≥ 4 :=
by
  sorry

end inequality_inequality_hold_l139_139034


namespace matchsticks_20th_stage_l139_139684

theorem matchsticks_20th_stage :
  let a1 := 3
  let d := 3
  let a20 := a1 + 19 * d
  a20 = 60 := by
  sorry

end matchsticks_20th_stage_l139_139684


namespace river_width_l139_139082

theorem river_width (w : ℕ) (speed_const : ℕ) 
(meeting1_from_nearest_shore : ℕ) (meeting2_from_other_shore : ℕ)
(h1 : speed_const = 1) 
(h2 : meeting1_from_nearest_shore = 720) 
(h3 : meeting2_from_other_shore = 400)
(h4 : 3 * w = 3 * meeting1_from_nearest_shore)
(h5 : 2160 = 2 * w - meeting2_from_other_shore) :
w = 1280 :=
by
  {
      sorry
  }

end river_width_l139_139082


namespace find_s_l139_139650

variable {a b n r s : ℝ}

theorem find_s (h1 : Polynomial.aeval a (Polynomial.X ^ 2 - Polynomial.C n * Polynomial.X + Polynomial.C 6) = 0)
              (h2 : Polynomial.aeval b (Polynomial.X ^ 2 - Polynomial.C n * Polynomial.X + Polynomial.C 6) = 0)
              (h_ab : a * b = 6)
              (h_roots : Polynomial.aeval (a + 2/b) (Polynomial.X ^ 2 - Polynomial.C r * Polynomial.X + Polynomial.C s) = 0)
              (h_roots2 : Polynomial.aeval (b + 2/a) (Polynomial.X ^ 2 - Polynomial.C r * Polynomial.X + Polynomial.C s) = 0) :
  s = 32/3 := 
sorry

end find_s_l139_139650


namespace number_of_two_element_subsets_l139_139710

def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem number_of_two_element_subsets (S : Type*) [Fintype S] 
  (h : binomial_coeff (Fintype.card S) 7 = 36) :
  binomial_coeff (Fintype.card S) 2 = 36 :=
by
  sorry

end number_of_two_element_subsets_l139_139710


namespace problem1_question_problem1_contrapositive_problem1_negation_problem2_question_problem2_contrapositive_problem2_negation_l139_139087

-- Proof statement for problem 1

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem problem1_question (x y : ℕ) (h : ¬(is_odd x ∧ is_odd y)) : is_odd (x + y) := sorry

theorem problem1_contrapositive (x y : ℕ) (h : is_odd x ∧ is_odd y) : ¬ is_odd (x + y) := sorry

theorem problem1_negation : ∃ (x y : ℕ), ¬(is_odd x ∧ is_odd y) ∧ ¬ is_odd (x + y) := sorry

-- Proof statement for problem 2

structure Square : Type := (is_rhombus : Prop)

def all_squares_are_rhombuses : Prop := ∀ (sq : Square), sq.is_rhombus

theorem problem2_question : all_squares_are_rhombuses = true := sorry

theorem problem2_contrapositive : ¬ all_squares_are_rhombuses = false := sorry

theorem problem2_negation : ¬(∃ (sq : Square), ¬ sq.is_rhombus) = false := sorry

end problem1_question_problem1_contrapositive_problem1_negation_problem2_question_problem2_contrapositive_problem2_negation_l139_139087


namespace problem1_problem2_l139_139626

variables {p x1 x2 y1 y2 : ℝ} (h₁ : p > 0) (h₂ : x1 * x2 ≠ 0) (h₃ : y1^2 = 2 * p * x1) (h₄ : y2^2 = 2 * p * x2)

theorem problem1 (h₅ : x1 * x2 + y1 * y2 = 0) :
    ∀ (x y : ℝ), (x - x1) * (x - x2) + (y - y1) * (y - y2) = 0 → 
        x^2 + y^2 - (x1 + x2) * x - (y1 + y2) * y = 0 := sorry

theorem problem2 (h₀ : ∀ x y, x = (x1 + x2) / 2 → y = (y1 + y2) / 2 → 
    |((x1 + x2) / 2) - 2 * ((y1 + y2) / 2)| / (Real.sqrt 5) = 2 * (Real.sqrt 5) / 5) :
    p = 2 := sorry

end problem1_problem2_l139_139626


namespace min_val_of_f_l139_139872

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  x + (2 * x) / (x^2 + 1) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

-- Theorem stating the minimum value of f(x) for x > 0 is 5.5
theorem min_val_of_f : ∀ x : ℝ, x > 0 → f x ≥ 5.5 :=
by sorry

end min_val_of_f_l139_139872


namespace inequality_sum_leq_three_l139_139747

theorem inequality_sum_leq_three
  (x y z : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (hxyz : x^2 + y^2 + z^2 ≥ 3) :
  (x^2 + y^2 + z^2) / (x^5 + y^2 + z^2) + 
  (x^2 + y^2 + z^2) / (y^5 + x^2 + z^2) + 
  (x^2 + y^2 + z^2) / (z^5 + x^2 + y^2 + z^2) ≤ 3 := 
sorry

end inequality_sum_leq_three_l139_139747


namespace varphi_solution_l139_139617

noncomputable def varphi (x : ℝ) (m n : ℝ) : ℝ :=
  m * x + n / x

theorem varphi_solution :
  ∃ (m n : ℝ), (varphi 1 m n = 8) ∧ (varphi 16 m n = 16) ∧ (∀ x, varphi x m n = 3 * x + 5 / x) :=
sorry

end varphi_solution_l139_139617


namespace necessary_but_not_sufficient_l139_139789

theorem necessary_but_not_sufficient (x : ℝ) : (x > -1) ↔ (∀ y : ℝ, (2 * y > 2) → (-1 < y)) :=
sorry

end necessary_but_not_sufficient_l139_139789


namespace second_student_catches_up_l139_139833

open Nat

-- Definitions for the problems
def distance_first_student (n : ℕ) : ℕ := 7 * n
def distance_second_student (n : ℕ) : ℕ := n * (n + 1) / 2

-- Theorem statement indicating the second student catches up with the first at n = 13
theorem second_student_catches_up : ∃ n, (distance_first_student n = distance_second_student n) ∧ n = 13 := 
by 
  sorry

end second_student_catches_up_l139_139833


namespace sum_of_x_l139_139044

open Real

def f (x : ℝ) := 3 * x - 2
def f_inv (x : ℝ) := (x + 2) / 3
def f_x_inv (x : ℝ) := f (x⁻¹)

theorem sum_of_x (h : ∀ x, f_inv x = f_x_inv x → x = 1 ∨ x = -9 / 7) :
  (1 : ℝ) + -9 / 7 = -2 / 7 :=
by
  sorry

end sum_of_x_l139_139044


namespace apple_count_l139_139480

/--
Given the total number of apples in several bags, where each bag contains either 12 or 6 apples,
and knowing that the total number of apples is between 70 and 80 inclusive,
prove that the total number of apples can only be 72 or 78.
-/
theorem apple_count (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : ∃ k : ℕ, n = 6 * k) : n = 72 ∨ n = 78 :=
by {
  sorry
}

end apple_count_l139_139480


namespace least_positive_integer_not_representable_as_fraction_l139_139737

theorem least_positive_integer_not_representable_as_fraction : 
  ¬ ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ (2^a - 2^b) / (2^c - 2^d) = 11 :=
sorry

end least_positive_integer_not_representable_as_fraction_l139_139737


namespace problem_statement_l139_139629

theorem problem_statement (f : ℕ → ℤ) (a b : ℤ) 
  (h1 : f 1 = 7) 
  (h2 : f 2 = 11)
  (h3 : ∀ x, f x = a * x^2 + b * x + 3) :
  f 3 = 15 := 
sorry

end problem_statement_l139_139629


namespace sum_of_first_45_natural_numbers_l139_139090

theorem sum_of_first_45_natural_numbers : (45 * (45 + 1)) / 2 = 1035 := by
  sorry

end sum_of_first_45_natural_numbers_l139_139090


namespace expression_value_l139_139218

theorem expression_value : ((40 + 15) ^ 2 - 15 ^ 2) = 2800 := 
by
  sorry

end expression_value_l139_139218


namespace apple_count_l139_139487

-- Define constants for the problem
constant bags: ℕ → Prop
constant total_apples: ℕ
constant apples_in_bag1: ℕ := 12
constant apples_in_bag2: ℕ := 6

-- Define the conditions
axiom condition1 (n: ℕ) : bags n → (n = apples_in_bag1 ∨ n = apples_in_bag2)
axiom condition2 : 70 ≤ total_apples ∧ total_apples ≤ 80
axiom condition3 : total_apples % 6 = 0

-- State the proof problem
theorem apple_count :
  total_apples = 72 ∨ total_apples = 78 :=
sorry

end apple_count_l139_139487


namespace apple_count_l139_139492

-- Define constants for the problem
constant bags: ℕ → Prop
constant total_apples: ℕ
constant apples_in_bag1: ℕ := 12
constant apples_in_bag2: ℕ := 6

-- Define the conditions
axiom condition1 (n: ℕ) : bags n → (n = apples_in_bag1 ∨ n = apples_in_bag2)
axiom condition2 : 70 ≤ total_apples ∧ total_apples ≤ 80
axiom condition3 : total_apples % 6 = 0

-- State the proof problem
theorem apple_count :
  total_apples = 72 ∨ total_apples = 78 :=
sorry

end apple_count_l139_139492


namespace doughnuts_per_box_l139_139567

theorem doughnuts_per_box
  (total_doughnuts : ℕ)
  (boxes_sold : ℕ)
  (doughnuts_given_away : ℕ)
  (doughnuts_per_box : ℕ)
  (h1 : total_doughnuts = 300)
  (h2 : boxes_sold = 27)
  (h3 : doughnuts_given_away = 30) :
  doughnuts_per_box = (total_doughnuts - doughnuts_given_away) / boxes_sold := by
  -- proof goes here
  sorry

end doughnuts_per_box_l139_139567


namespace apple_bags_l139_139499

theorem apple_bags (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apple_bags_l139_139499


namespace non_monotonic_m_range_l139_139895

theorem non_monotonic_m_range (m : ℝ) :
  (∃ x ∈ Set.Ioo (-1 : ℝ) 2, (3 * x^2 + 2 * x + m = 0)) →
  m ∈ Set.Ioo (-16 : ℝ) (1/3 : ℝ) :=
sorry

end non_monotonic_m_range_l139_139895


namespace highland_park_science_fair_l139_139854

noncomputable def juniors_and_seniors_participants (j s : ℕ) : ℕ :=
  (3 * j) / 4 + s / 2

theorem highland_park_science_fair 
  (j s : ℕ)
  (h1 : (3 * j) / 4 = s / 2)
  (h2 : j + s = 240) :
  juniors_and_seniors_participants j s = 144 := by
  sorry

end highland_park_science_fair_l139_139854


namespace no_x_squared_term_l139_139018

theorem no_x_squared_term {m : ℚ} (h : (x+1) * (x^2 + 5*m*x + 3) = x^3 + (5*m + 1)*x^2 + (3 + 5*m)*x + 3) : 
  5*m + 1 = 0 → m = -1/5 := by sorry

end no_x_squared_term_l139_139018


namespace find_davids_marks_in_physics_l139_139371

theorem find_davids_marks_in_physics (marks_english : ℕ) (marks_math : ℕ) (marks_chemistry : ℕ) (marks_biology : ℕ)
  (average_marks : ℕ) (num_subjects : ℕ) (H1 : marks_english = 61) 
  (H2 : marks_math = 65) (H3 : marks_chemistry = 67) 
  (H4 : marks_biology = 85) (H5 : average_marks = 72) (H6 : num_subjects = 5) :
  ∃ (marks_physics : ℕ), marks_physics = 82 :=
by
  sorry

end find_davids_marks_in_physics_l139_139371


namespace problem1_problem2_problem3_problem4_l139_139720

open Rat

-- Problem 1
theorem problem1 : abs (-6) - 7 + (-3) = -4 := by
  sorry

-- Problem 2
theorem problem2 : (1/2 - 5/9 + 2/3) * (-18) = -11 := by
  sorry

-- Problem 3
theorem problem3 : 4 ÷ (-2) * -(3/2) - -4 = 7 := by
  sorry

-- Problem 4
theorem problem4 : - (5/7) * ((-3)^2 * -(4/3) - 2) = 10 := by
  sorry

end problem1_problem2_problem3_problem4_l139_139720


namespace ratio_of_areas_l139_139690

theorem ratio_of_areas (C1 C2 : ℝ) (h1 : (60 : ℝ) / 360 * C1 = (48 : ℝ) / 360 * C2) : 
  (C1 / C2) ^ 2 = 16 / 25 := 
by
  sorry

end ratio_of_areas_l139_139690


namespace marked_price_percentage_l139_139579

variables (L M: ℝ)

-- The store owner purchases items at a 25% discount of the list price.
def cost_price (L : ℝ) := 0.75 * L

-- The store owner plans to mark them up such that after a 10% discount on the marked price,
-- he achieves a 25% profit on the selling price.
def selling_price (M : ℝ) := 0.9 * M

-- Given condition: cost price is 75% of selling price
theorem marked_price_percentage (h : cost_price L = 0.75 * selling_price M) : 
  M = 1.111 * L :=
by 
  sorry

end marked_price_percentage_l139_139579


namespace interest_rate_increase_60_percent_l139_139581

noncomputable def percentage_increase (A P A' t : ℝ) : ℝ :=
  let r₁ := (A - P) / (P * t)
  let r₂ := (A' - P) / (P * t)
  ((r₂ - r₁) / r₁) * 100

theorem interest_rate_increase_60_percent :
  percentage_increase 920 800 992 3 = 60 := by
  sorry

end interest_rate_increase_60_percent_l139_139581


namespace average_marks_passed_l139_139175

noncomputable def total_candidates := 120
noncomputable def total_average_marks := 35
noncomputable def passed_candidates := 100
noncomputable def failed_candidates := total_candidates - passed_candidates
noncomputable def average_marks_failed := 15
noncomputable def total_marks := total_average_marks * total_candidates
noncomputable def total_marks_failed := average_marks_failed * failed_candidates

theorem average_marks_passed :
  ∃ P, P * passed_candidates + total_marks_failed = total_marks ∧ P = 39 := by
  sorry

end average_marks_passed_l139_139175


namespace inequality_proof_l139_139265

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a^2 / (a + b)) + (b^2 / (b + c)) + (c^2 / (c + a)) ≥ (a + b + c) / 2 := 
by
  sorry

end inequality_proof_l139_139265


namespace abs_eq_sum_condition_l139_139015

theorem abs_eq_sum_condition (x y : ℝ) (h : |x - y^2| = x + y^2) : x = 0 ∧ y = 0 :=
  sorry

end abs_eq_sum_condition_l139_139015


namespace find_pairs_l139_139252

theorem find_pairs (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  (∃ k m : ℕ, k ≠ 0 ∧ m ≠ 0 ∧ x + 1 = k * y ∧ y + 1 = m * x) ↔
  (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 2) ∨ (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 3) :=
by
  sorry

end find_pairs_l139_139252


namespace min_value_y1_minus_4y2_l139_139820

/-- 
Suppose a parabola C : y^2 = 4x intersects at points A(x1, y1) and B(x2, y2) with a line 
passing through its focus. Given that A is in the first quadrant, 
the minimum value of |y1 - 4y2| is 8.
--/
theorem min_value_y1_minus_4y2 (x1 y1 x2 y2 : ℝ) 
  (h1 : y1^2 = 4 * x1) 
  (h2 : y2^2 = 4 * x2)
  (h3 : x1 > 0) (h4 : y1 > 0) 
  (focus : (1, 0) ∈ {(x, y) | y^2 = 4 * x}) : 
  (|y1 - 4 * y2|) ≥ 8 :=
sorry

end min_value_y1_minus_4y2_l139_139820


namespace range_of_m_for_ellipse_l139_139299

-- Define the equation of the ellipse
def ellipse_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2

-- The theorem to prove
theorem range_of_m_for_ellipse (m : ℝ) :
  (∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) →
  5 < m :=
sorry

end range_of_m_for_ellipse_l139_139299


namespace total_apples_l139_139519

theorem total_apples (n : ℕ) (h1 : ∀ k : ℕ, 6 * k = n ∨ 12 * k = n) (h2 : 70 ≤ n) (h3 : n ≤ 80) : 
  n = 72 ∨ n = 78 := 
sorry

end total_apples_l139_139519


namespace min_length_ab_l139_139807

noncomputable def min_distance_segment_ab : ℝ :=
  let f (a b : ℝ) := (a - b) ^ 2 + (12 / 5 * a - 9 - b ^ 2) ^ 2 in
  let ∂f_∂a (a b : ℝ) : ℝ := 2 * (a - b) + 24 / 5 * (12 / 5 * a - 9 - b ^ 2) in
  let ∂f_∂b (a b : ℝ) : ℝ := 2 * (a - b) - 4 * b * (12 / 5 * a - 9 - b ^ 2) in
  have h_a : a = 15 / 13, from sorry,
  have h_b : b = 6 / 5, from sorry,
  let d := (a, b) in
  sqrt ((15 / 13 - 6 / 5) ^ 2 + (12 / 5 * 15 / 13 - 9 - (6 / 5) ^ 2) ^ 2)

theorem min_length_ab : min_distance_segment_ab = 189 / 65 :=
by
  unfold min_distance_segment_ab
  admit  -- Proof to be filled


end min_length_ab_l139_139807


namespace geometric_sequence_sum_63_l139_139918

theorem geometric_sequence_sum_63
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_init : a 1 = 1)
  (h_recurrence : ∀ n, a (n + 2) + 2 * a (n + 1) = 8 * a n) :
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) = 63 :=
by
  sorry

end geometric_sequence_sum_63_l139_139918


namespace sum_of_coeffs_is_minus_one_l139_139903

theorem sum_of_coeffs_is_minus_one 
  (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℤ) :
  (∀ x : ℤ, (1 - x^3)^3 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 + a₈ * x^8 + a₉ * x^9)
  → a = 1 
  → a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -1 :=
by
  sorry

end sum_of_coeffs_is_minus_one_l139_139903


namespace find_k_l139_139069

theorem find_k : 
  ∃ x y k : ℝ, y = 7 * x - 2 ∧ y = -3 * x + 14 ∧ y = 4 * x + k ∧ k = 2.8 :=
by
  sorry

end find_k_l139_139069


namespace infinitely_many_H_points_l139_139950

-- Define the curve C as (x^2 / 4) + y^2 = 1
def is_on_curve (x y : ℝ) : Prop :=
  (x^2 / 4) + y^2 = 1

-- Define point P on curve C
def is_H_point (P : ℝ × ℝ) : Prop :=
  is_on_curve P.1 P.2 ∧
  ∃ (A B : ℝ × ℝ), is_on_curve A.1 A.2 ∧ B.1 = 4 ∧
  (dist (P.1, P.2) (A.1, A.2) = dist (P.1, P.2) (B.1, B.2) ∨
   dist (P.1, P.2) (A.1, A.2) = dist (A.1, A.2) (B.1, B.2))

-- Theorem to prove the existence of infinitely many H points
theorem infinitely_many_H_points : ∃ (P : ℝ × ℝ), is_H_point P ∧ ∀ (Q : ℝ × ℝ), Q ≠ P → is_H_point Q :=
sorry


end infinitely_many_H_points_l139_139950


namespace sum_of_coefficients_l139_139775

noncomputable def coeff_sum (x y z : ℝ) : ℝ :=
  let p := (x + 2*y - z)^8  
  -- extract and sum coefficients where exponent of x is 2 and exponent of y is not 1
  sorry

theorem sum_of_coefficients (x y z : ℝ) :
  coeff_sum x y z = 364 := by
  sorry

end sum_of_coefficients_l139_139775


namespace initial_milk_water_ratio_l139_139156

theorem initial_milk_water_ratio
  (M W : ℕ)
  (h1 : M + W = 40000)
  (h2 : (M : ℚ) / (W + 1600) = 3 / 1) :
  (M : ℚ) / W = 3.55 :=
by
  sorry

end initial_milk_water_ratio_l139_139156


namespace fraction_product_equals_1_67_l139_139590

noncomputable def product_of_fractions : ℝ :=
  (2 / 1) * (2 / 3) * (4 / 3) * (4 / 5) * (6 / 5) * (6 / 7) * (8 / 7)

theorem fraction_product_equals_1_67 :
  round (product_of_fractions * 100) / 100 = 1.67 := by
  sorry

end fraction_product_equals_1_67_l139_139590


namespace geometric_series_sum_l139_139343

theorem geometric_series_sum :
  let a := 1
  let r := 3
  let n := 9
  (1 * (3^n - 1) / (3 - 1)) = 9841 :=
by
  sorry

end geometric_series_sum_l139_139343


namespace idiom_describes_random_event_l139_139550

-- Define the idioms as propositions.
def FishingForMoonInWater : Prop := ∀ (x : Type), x -> False
def CastlesInTheAir : Prop := ∀ (y : Type), y -> False
def WaitingByStumpForHare : Prop := ∃ (z : Type), True
def CatchingTurtleInJar : Prop := ∀ (w : Type), w -> False

-- Define the main theorem to state that WaitingByStumpForHare describes a random event.
theorem idiom_describes_random_event : WaitingByStumpForHare :=
  sorry

end idiom_describes_random_event_l139_139550


namespace wine_age_proof_l139_139957

-- Definitions based on conditions
def Age_Carlo_Rosi : ℕ := 40
def Age_Twin_Valley : ℕ := Age_Carlo_Rosi / 4
def Age_Franzia : ℕ := 3 * Age_Carlo_Rosi

-- We'll use a definition to represent the total age of the three brands of wine.
def Total_Age : ℕ := Age_Franzia + Age_Carlo_Rosi + Age_Twin_Valley

-- Statement to be proven
theorem wine_age_proof : Total_Age = 170 :=
by {
  sorry -- Proof goes here
}

end wine_age_proof_l139_139957


namespace sum_of_solutions_l139_139469

theorem sum_of_solutions (x : ℝ) :
  (∀ x : ℝ, x^3 + x^2 - 6*x - 20 = 4*x + 24) →
  let polynomial := (x^3 + x^2 - 10*x - 44);
  (polynomial = 0) →
  let a := 1;
  let b := 1;
  -b/a = -1 :=
sorry

end sum_of_solutions_l139_139469


namespace fewest_printers_l139_139975

/-!
# Fewest Printers Purchase Problem
Given two types of computer printers costing $350 and $200 per unit, respectively,
given that the company wants to spend equal amounts on both types of printers.
Prove that the fewest number of printers the company can purchase is 11.
-/

theorem fewest_printers (p1 p2 : ℕ) (h1 : p1 = 350) (h2 : p2 = 200) :
  ∃ n1 n2 : ℕ, p1 * n1 = p2 * n2 ∧ n1 + n2 = 11 := 
sorry

end fewest_printers_l139_139975


namespace area_of_trapezium_l139_139870

-- Definitions for the given conditions
def parallel_side_a : ℝ := 18  -- in cm
def parallel_side_b : ℝ := 20  -- in cm
def distance_between_sides : ℝ := 5  -- in cm

-- Statement to prove the area is 95 cm²
theorem area_of_trapezium : 
  let a := parallel_side_a
  let b := parallel_side_b
  let h := distance_between_sides
  (1 / 2 * (a + b) * h = 95) :=
by
  sorry  -- Proof is not required here

end area_of_trapezium_l139_139870


namespace initial_machines_count_l139_139294

theorem initial_machines_count (M : ℕ) (h1 : M * 8 = 8 * 1) (h2 : 72 * 6 = 12 * 2) : M = 64 :=
by
  sorry

end initial_machines_count_l139_139294


namespace geometric_sequence_a6_l139_139301

theorem geometric_sequence_a6 (a : ℕ → ℝ) (a1 r : ℝ) (h1 : ∀ n, a n = a1 * r ^ (n - 1)) (h2 : (a 2) * (a 4) * (a 12) = 64) : a 6 = 4 :=
sorry

end geometric_sequence_a6_l139_139301


namespace roller_skate_wheels_l139_139800

theorem roller_skate_wheels (number_of_people : ℕ)
  (feet_per_person : ℕ)
  (skates_per_foot : ℕ)
  (wheels_per_skate : ℕ)
  (h_people : number_of_people = 40)
  (h_feet : feet_per_person = 2)
  (h_skates : skates_per_foot = 1)
  (h_wheels : wheels_per_skate = 4)
  : (number_of_people * feet_per_person * skates_per_foot * wheels_per_skate) = 320 := 
by
  sorry

end roller_skate_wheels_l139_139800


namespace tan_half_angle_product_l139_139008

theorem tan_half_angle_product (a b : ℝ) (h : 3 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0) :
  ∃ (x : ℝ), x = Real.tan (a / 2) * Real.tan (b / 2) ∧ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2) := 
sorry

end tan_half_angle_product_l139_139008


namespace trackball_mice_count_l139_139312

theorem trackball_mice_count 
  (total_mice wireless_mice optical_mice trackball_mice : ℕ)
  (h1 : total_mice = 80)
  (h2 : wireless_mice = total_mice / 2)
  (h3 : optical_mice = total_mice / 4)
  (h4 : trackball_mice = total_mice - (wireless_mice + optical_mice)) :
  trackball_mice = 20 := by 
  sorry

end trackball_mice_count_l139_139312


namespace a_values_in_terms_of_x_l139_139744

open Real

-- Definitions for conditions
variables (a b x y : ℝ)
variables (h1 : a^3 - b^3 = 27 * x^3)
variables (h2 : a - b = y)
variables (h3 : y = 2 * x)

-- Theorem to prove
theorem a_values_in_terms_of_x : 
  (a = x + 5 * x / sqrt 6) ∨ (a = x - 5 * x / sqrt 6) :=
sorry

end a_values_in_terms_of_x_l139_139744


namespace angle_between_vectors_45_degrees_l139_139889

open Real

noncomputable def vec_dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vec_mag (v : ℝ × ℝ) : ℝ := sqrt (vec_dot v v)

noncomputable def vec_angle (v w : ℝ × ℝ) : ℝ := arccos (vec_dot v w / (vec_mag v * vec_mag w))

theorem angle_between_vectors_45_degrees 
  (e1 e2 : ℝ × ℝ)
  (h1 : vec_mag e1 = 1)
  (h2 : vec_mag e2 = 1)
  (h3 : vec_dot e1 e2 = 0)
  (a : ℝ × ℝ := (3, 0) - (0, 1))  -- (3 * e1 - e2) is represented in a direct vector form (3, -1)
  (b : ℝ × ℝ := (2, 0) + (0, 1)): -- (2 * e1 + e2) is represented in a direct vector form (2, 1)
  vec_angle a b = π / 4 :=  -- π / 4 radians is equivalent to 45 degrees
sorry

end angle_between_vectors_45_degrees_l139_139889


namespace cubic_has_real_root_l139_139658

open Real

-- Define the conditions
variables (a0 a1 a2 a3 : ℝ) (h : a0 ≠ 0)

-- Define the cubic polynomial function
def cubic (x : ℝ) : ℝ :=
  a0 * x^3 + a1 * x^2 + a2 * x + a3

-- State the theorem
theorem cubic_has_real_root : ∃ x : ℝ, cubic a0 a1 a2 a3 x = 0 :=
by
  sorry

end cubic_has_real_root_l139_139658


namespace radius_increase_area_triple_l139_139666

theorem radius_increase_area_triple (r m : ℝ) (h : π * (r + m)^2 = 3 * π * r^2) : 
  r = (m * (Real.sqrt 3 - 1)) / 2 := 
sorry

end radius_increase_area_triple_l139_139666


namespace increasing_on_interval_min_value_on_interval_l139_139143

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2

theorem increasing_on_interval (a : ℝ) : 
  (∀ x, 1 < x → (deriv (f a) x) ≥ 0) ↔ (a ≥ -2) := sorry

theorem min_value_on_interval (a : ℝ) :
  let f_min := 
    if a ≥ -2 then 1
    else if -2 * Real.exp 2 < a ∧ a < -2 then (a / 2) * Real.log (-a / 2) - a / 2
    else a + Real.exp 2 in
  let x_min := 
    if a ≥ -2 then 1
    else if -2 * Real.exp 2 < a ∧ a < -2 then Real.sqrt (-a / 2)
    else Real.exp 1 in
  ∃ (x : ℝ), x ∈ Set.Icc 1 (Real.exp 1) ∧ ∀ y ∈ Set.Icc 1 (Real.exp 1), f a y ≥ f a x ∧ f a x = f_min ∧ x = x_min := sorry

end increasing_on_interval_min_value_on_interval_l139_139143


namespace possible_apple_counts_l139_139503

theorem possible_apple_counts (n : ℕ) (h₁ : 70 ≤ n) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end possible_apple_counts_l139_139503


namespace height_of_triangle_l139_139104

-- Define the dimensions of the rectangle
variable (l w : ℝ)

-- Assume the base of the triangle is equal to the length of the rectangle
-- We need to prove that the height of the triangle h = 2w

theorem height_of_triangle (h : ℝ) (hl_eq_length : l > 0) (hw_eq_width : w > 0) :
  (l * w) = (1 / 2) * l * h → h = 2 * w :=
by
  sorry

end height_of_triangle_l139_139104


namespace cost_price_percentage_l139_139065

/-- The cost price (CP) as a percentage of the marked price (MP) given 
that the discount is 18% and the gain percent is 28.125%. -/
theorem cost_price_percentage (MP CP : ℝ) (h1 : CP / MP = 0.64) : 
  (CP / MP) * 100 = 64 :=
by
  sorry

end cost_price_percentage_l139_139065


namespace fraction_of_selected_films_in_color_l139_139554

variables (x y : ℕ)

theorem fraction_of_selected_films_in_color (B C : ℕ) (e : ℚ)
  (h1 : B = 20 * x)
  (h2 : C = 6 * y)
  (h3 : e = (6 * y : ℚ) / (((y / 5 : ℚ) + 6 * y))) :
  e = 30 / 31 :=
by {
  sorry
}

end fraction_of_selected_films_in_color_l139_139554


namespace flour_needed_for_bread_l139_139656

-- Definitions based on conditions
def flour_per_loaf : ℝ := 2.5
def number_of_loaves : ℕ := 2

-- Theorem statement
theorem flour_needed_for_bread : flour_per_loaf * number_of_loaves = 5 :=
by sorry

end flour_needed_for_bread_l139_139656


namespace percentage_land_mr_william_l139_139302

noncomputable def tax_rate_arable := 0.01
noncomputable def tax_rate_orchard := 0.02
noncomputable def tax_rate_pasture := 0.005

noncomputable def subsidy_arable := 100
noncomputable def subsidy_orchard := 50
noncomputable def subsidy_pasture := 20

noncomputable def total_tax_village := 3840
noncomputable def tax_mr_william := 480

theorem percentage_land_mr_william : 
  (tax_mr_william / total_tax_village : ℝ) * 100 = 12.5 :=
by
  sorry

end percentage_land_mr_william_l139_139302


namespace scientific_notation_of_300670_l139_139170

theorem scientific_notation_of_300670 : ∃ a : ℝ, ∃ n : ℤ, (1 ≤ |a| ∧ |a| < 10) ∧ 300670 = a * 10^n ∧ a = 3.0067 ∧ n = 5 :=
  by
    sorry

end scientific_notation_of_300670_l139_139170


namespace solve_eq_64_16_pow_x_minus_1_l139_139171

theorem solve_eq_64_16_pow_x_minus_1 (x : ℝ) (h : 64 = 4 * (16 : ℝ) ^ (x - 1)) : x = 2 :=
sorry

end solve_eq_64_16_pow_x_minus_1_l139_139171


namespace fraction_of_square_above_line_l139_139188

theorem fraction_of_square_above_line :
  let A := (2, 1)
  let B := (5, 1)
  let C := (5, 4)
  let D := (2, 4)
  let P := (2, 3)
  let Q := (5, 1)
  ∃ f : ℚ, f = 2 / 3 := 
by
  -- Placeholder for the proof
  sorry

end fraction_of_square_above_line_l139_139188


namespace molecular_weight_of_9_moles_l139_139693

theorem molecular_weight_of_9_moles (molecular_weight : ℕ) (moles : ℕ) (h₁ : molecular_weight = 1098) (h₂ : moles = 9) :
  molecular_weight * moles = 9882 :=
by {
  sorry
}

end molecular_weight_of_9_moles_l139_139693


namespace area_of_triangles_equal_l139_139307

theorem area_of_triangles_equal {a b c d : ℝ} (h_hyperbola_a : a ≠ 0) (h_hyperbola_b : b ≠ 0) 
    (h_hyperbola_c : c ≠ 0) (h_hyperbola_d : d ≠ 0) (h_parallel : a * b = c * d) :
  (1 / 2) * ((a + c) * (a + c) / (a * c)) = (1 / 2) * ((b + d) * (b + d) / (b * d)) :=
by
  sorry

end area_of_triangles_equal_l139_139307


namespace ball_bounce_height_l139_139988

theorem ball_bounce_height (b : ℕ) : 
  ∃ b : ℕ, 400 * (3 / 4 : ℝ)^b < 50 ∧ ∀ b' < b, 400 * (3 / 4 : ℝ)^b' ≥ 50 :=
sorry

end ball_bounce_height_l139_139988


namespace ordered_triples_count_l139_139576

theorem ordered_triples_count : 
  let b := 3003
  let side_length_squared := b * b
  let num_divisors := (2 + 1) * (2 + 1) * (2 + 1) * (2 + 1)
  let half_divisors := num_divisors / 2
  half_divisors = 40 := by
  sorry

end ordered_triples_count_l139_139576


namespace det_proof_l139_139036

variable {R : Type*} [Field R] [Inhabited R]

def matrix_2x2 (a b c d : R) : Matrix (Fin 2) (Fin 2) R :=
  !![a, b; c, d]

theorem det_proof (x : R) (A : Matrix (Fin 2) (Fin 2) R) (h₁ : 0 < x)
    (h₂ : A.is_square) (h₃ : A.det ≠ 0) (h₄ : (A^2 + (scalarMatrix 2 x)).det = 0) :
    (A^2 + A + (scalarMatrix 2 x)).det = x := by
  sorry

end det_proof_l139_139036


namespace distinct_real_roots_absolute_sum_l139_139973

theorem distinct_real_roots_absolute_sum {r1 r2 p : ℝ} (h_root1 : r1 ^ 2 + p * r1 + 7 = 0) 
(h_root2 : r2 ^ 2 + p * r2 + 7 = 0) (h_distinct : r1 ≠ r2) : 
|r1 + r2| > 2 * Real.sqrt 7 := 
sorry

end distinct_real_roots_absolute_sum_l139_139973


namespace cube_diagonal_length_l139_139827

theorem cube_diagonal_length (s : ℝ) 
    (h₁ : 6 * s^2 = 54) 
    (h₂ : 12 * s = 36) :
    ∃ d : ℝ, d = 3 * Real.sqrt 3 ∧ d = Real.sqrt (3 * s^2) :=
by
  sorry

end cube_diagonal_length_l139_139827


namespace smallest_w_l139_139418

theorem smallest_w (w : ℕ) (w_pos : w > 0) (h1 : ∀ n : ℕ, 2^4 ∣ 1452 * w)
                              (h2 : ∀ n : ℕ, 3^3 ∣ 1452 * w)
                              (h3 : ∀ n : ℕ, 13^3 ∣ 1452 * w) :
  w = 676 := sorry

end smallest_w_l139_139418


namespace B_subscription_difference_l139_139713

noncomputable def subscription_difference (A B C P : ℕ) (delta : ℕ) (comb_sub: A + B + C = 50000) (c_profit: 8400 = 35000 * C / 50000) :=
  B - C

theorem B_subscription_difference (A B C : ℕ) (z: ℕ) 
  (h1 : A + B + C = 50000) 
  (h2 : A = B + 4000) 
  (h3 : (B - C) = z)
  (h4 :  8400 = 35000 * C / 50000):
  B - C = 10000 :=
by {
  sorry
}

end B_subscription_difference_l139_139713


namespace negation_P1_is_false_negation_P2_is_false_l139_139564

-- Define the propositions
def isMultiDigitNumber (n : ℕ) : Prop := n >= 10
def lastDigitIsZero (n : ℕ) : Prop := n % 10 = 0
def isMultipleOfFive (n : ℕ) : Prop := n % 5 = 0
def isEven (n : ℕ) : Prop := n % 2 = 0

-- The propositions
def P1 (n : ℕ) : Prop := isMultiDigitNumber n → (lastDigitIsZero n → isMultipleOfFive n)
def P2 : Prop := ∀ n, isEven n → n % 2 = 0

-- The negations
def notP1 (n : ℕ) : Prop := isMultiDigitNumber n ∧ lastDigitIsZero n → ¬isMultipleOfFive n
def notP2 : Prop := ∃ n, isEven n ∧ ¬(n % 2 = 0)

-- The proof problems
theorem negation_P1_is_false (n : ℕ) : notP1 n → False := by
  sorry

theorem negation_P2_is_false : notP2 → False := by
  sorry

end negation_P1_is_false_negation_P2_is_false_l139_139564


namespace roger_allowance_fraction_l139_139868

noncomputable def allowance_fraction (A m s p : ℝ) : ℝ :=
  m + s + p

theorem roger_allowance_fraction (A : ℝ) (m s p : ℝ) 
  (h_movie : m = 0.25 * (A - s - p))
  (h_soda : s = 0.10 * (A - m - p))
  (h_popcorn : p = 0.05 * (A - m - s)) :
  allowance_fraction A m s p = 0.32 * A :=
by
  sorry

end roger_allowance_fraction_l139_139868


namespace geom_seq_a12_value_l139_139642

-- Define the geometric sequence as a function from natural numbers to real numbers
def geom_seq (a : ℕ → ℝ) : Prop :=
  ∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

theorem geom_seq_a12_value (a : ℕ → ℝ) 
  (H_geom : geom_seq a) 
  (H_7_9 : a 7 * a 9 = 4) 
  (H_4 : a 4 = 1) : 
  a 12 = 4 := 
by 
  sorry

end geom_seq_a12_value_l139_139642


namespace original_angle_measure_l139_139177

theorem original_angle_measure : 
  ∃ x : ℝ, (90 - x) = 3 * x - 2 ∧ x = 23 :=
by
  sorry

end original_angle_measure_l139_139177


namespace john_moves_3594_pounds_l139_139923

def bench_press_weight := 15
def bench_press_reps := 10
def bench_press_sets := 3

def bicep_curls_weight := 12
def bicep_curls_reps := 8
def bicep_curls_sets := 4

def squats_weight := 50
def squats_reps := 12
def squats_sets := 3

def deadlift_weight := 80
def deadlift_reps := 6
def deadlift_sets := 2

def total_weight_moved : Nat :=
  (bench_press_weight * bench_press_reps * bench_press_sets) +
  (bicep_curls_weight * bicep_curls_reps * bicep_curls_sets) +
  (squats_weight * squats_reps * squats_sets) +
  (deadlift_weight * deadlift_reps * deadlift_sets)

theorem john_moves_3594_pounds :
  total_weight_moved = 3594 := by {
    sorry
}

end john_moves_3594_pounds_l139_139923


namespace star_value_l139_139881

-- Define the operation a star b
def star (a b : ℕ) : ℕ := a^2 - 2*a*b + b^2

-- We want to prove that 5 star 3 = 4
theorem star_value : star 5 3 = 4 := by
  sorry

end star_value_l139_139881


namespace abs_neg_two_l139_139465

theorem abs_neg_two : abs (-2) = 2 :=
by
  sorry

end abs_neg_two_l139_139465


namespace marnie_chips_l139_139794

theorem marnie_chips :
  ∀ (initial_chips : ℕ) (first_day_eaten : ℕ) (daily_eaten : ℕ),
    initial_chips = 100 →
    first_day_eaten = 10 →
    daily_eaten = 10 →
    (∃ d : ℕ, (d - 1) * daily_eaten + first_day_eaten = initial_chips ∧ d = 10) :=
by {
  intros initial_chips first_day_eaten daily_eaten h_initial h_first_day h_daily,
  use 10,
  split,
  {
    rw h_initial,
    rw h_first_day,
    rw h_daily,
    norm_num,
  },
  {
    norm_num,
  },
}

end marnie_chips_l139_139794


namespace mutually_exclusive_event_3_l139_139169

-- Definitions based on the conditions.
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Events based on problem conditions
def event_1 (a b : ℕ) : Prop := is_even a ∧ is_odd b ∨ is_odd a ∧ is_even b
def event_2 (a b : ℕ) : Prop := (is_odd a ∨ is_odd b) ∧ is_odd a ∧ is_odd b
def event_3 (a b : ℕ) : Prop := (is_odd a ∨ is_odd b) ∧ is_even a ∧ is_even b
def event_4 (a b : ℕ) : Prop := (is_odd a ∨ is_odd b) ∧ (is_even a ∨ is_even b)

-- Problem: Proving that event_3 is mutually exclusive with other events.
theorem mutually_exclusive_event_3 :
  ∀ (a b : ℕ), (event_3 a b) → ¬ (event_1 a b ∨ event_2 a b ∨ event_4 a b) :=
by
  sorry

end mutually_exclusive_event_3_l139_139169


namespace sally_picked_11_pears_l139_139058

theorem sally_picked_11_pears (total_pears : ℕ) (pears_picked_by_Sara : ℕ) (pears_picked_by_Sally : ℕ) 
    (h1 : total_pears = 56) (h2 : pears_picked_by_Sara = 45) :
    pears_picked_by_Sally = total_pears - pears_picked_by_Sara := by
  sorry

end sally_picked_11_pears_l139_139058


namespace NaOH_HCl_reaction_l139_139609

theorem NaOH_HCl_reaction (m : ℝ) (HCl : ℝ) (NaCl : ℝ) 
  (reaction_eq : NaOH + HCl = NaCl + H2O)
  (HCl_combined : HCl = 1)
  (NaCl_produced : NaCl = 1) :
  m = 1 := by
  sorry

end NaOH_HCl_reaction_l139_139609


namespace total_amount_l139_139850

-- Conditions as given definitions
def ratio_a : Nat := 2
def ratio_b : Nat := 3
def ratio_c : Nat := 4
def share_b : Nat := 1500

-- The final statement
theorem total_amount (parts_b := 3) (one_part := share_b / parts_b) :
  (2 * one_part) + (3 * one_part) + (4 * one_part) = 4500 :=
by
  sorry

end total_amount_l139_139850


namespace number_of_unique_four_digit_numbers_from_2004_l139_139901

-- Definitions representing the conditions
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def uses_digits_from_2004 (n : ℕ) : Prop := 
  ∀ d ∈ [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10], d ∈ [0, 2, 4]

-- The proposition we need to prove
theorem number_of_unique_four_digit_numbers_from_2004 :
  ∃ n : ℕ, is_four_digit_number n ∧ uses_digits_from_2004 n ∧ n = 6 := 
sorry

end number_of_unique_four_digit_numbers_from_2004_l139_139901


namespace convert_spherical_to_rectangular_l139_139865

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : (ℝ × ℝ × ℝ) :=
  (ρ * Real.sin φ * Real.cos θ,
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

theorem convert_spherical_to_rectangular : spherical_to_rectangular 5 (Real.pi / 2) (Real.pi / 3) = 
  (0, 5 * Real.sqrt 3 / 2, 5 / 2) :=
by
  sorry

end convert_spherical_to_rectangular_l139_139865


namespace find_general_formula_prove_inequality_l139_139025

-- Define the sequence condition
def sequence_condition (a : ℕ → ℕ) (n : ℕ) : Prop :=
  (finset.range n).sum (λ k, a k * (1 / (n + 1))) = n^2 + n

-- Define the general formula
def general_formula (a : ℕ → ℕ) : Prop :=
  ∀ n, a n = 2 * n * (n + 1)

-- The inequality to prove
def inequality (a : ℕ → ℕ) : Prop :=
  (finset.range n).sum (λ k, k / ((k + 2) * a k)) < 1 / 4

-- Theorems to prove
theorem find_general_formula : ∃ a : ℕ → ℕ, general_formula a :=
begin
  sorry
end

theorem prove_inequality (a : ℕ → ℕ) (h : general_formula a) : inequality a :=
begin
  sorry
end

end find_general_formula_prove_inequality_l139_139025


namespace difference_between_two_greatest_values_l139_139977

-- Definition of the variables and conditions
def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n < 10

variables (a b c x : ℕ)

def conditions (a b c : ℕ) := is_digit a ∧ is_digit b ∧ is_digit c ∧ 2 * a = b ∧ b = 4 * c ∧ a > 0

-- Definition of x as a 3-digit number given a, b, and c
def three_digit_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

def smallest_x : ℕ := three_digit_number 2 4 1
def largest_x : ℕ := three_digit_number 4 8 2

def difference_two_greatest_values (a b c : ℕ) : ℕ := largest_x - smallest_x

-- The proof statement
theorem difference_between_two_greatest_values (a b c : ℕ) (h : conditions a b c) : 
  ∀ x1 x2 : ℕ, 
    three_digit_number 2 4 1 = x1 →
    three_digit_number 4 8 2 = x2 →
    difference_two_greatest_values a b c = 241 :=
by
  sorry

end difference_between_two_greatest_values_l139_139977


namespace smaller_octagon_area_fraction_l139_139675

theorem smaller_octagon_area_fraction (A B C D E F G H : Point)
  (midpoints_joined : Boolean)
  (regular_octagon : RegularOctagon A B C D E F G H)
  (smaller_octagon : Octagon (midpoint (A, B)) (midpoint (B, C)) (midpoint (C, D)) 
                              (midpoint (D, E)) (midpoint (E, F)) (midpoint (F, G))
                              (midpoint (G, H)) (midpoint (H, A))) :
  midpoints_joined → regular_octagon → 
  (area smaller_octagon) = (3 / 4) * (area regular_octagon) :=
by
  sorry

end smaller_octagon_area_fraction_l139_139675


namespace rectangle_height_l139_139419

-- Defining the conditions
def base : ℝ := 9
def area : ℝ := 33.3

-- Stating the proof problem
theorem rectangle_height : (area / base) = 3.7 :=
by
  sorry

end rectangle_height_l139_139419


namespace apple_count_l139_139483

/--
Given the total number of apples in several bags, where each bag contains either 12 or 6 apples,
and knowing that the total number of apples is between 70 and 80 inclusive,
prove that the total number of apples can only be 72 or 78.
-/
theorem apple_count (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : ∃ k : ℕ, n = 6 * k) : n = 72 ∨ n = 78 :=
by {
  sorry
}

end apple_count_l139_139483


namespace range_of_a_l139_139289

theorem range_of_a (a : ℝ) 
  (h : ∀ (f : ℝ → ℝ), 
    (∀ x ≤ a, f x = -x^2 - 2*x) ∧ 
    (∀ x > a, f x = -x) ∧ 
    ¬ ∃ M, ∀ x, f x ≤ M) : 
  a < -1 :=
by
  sorry

end range_of_a_l139_139289


namespace fraction_product_eq_l139_139365

theorem fraction_product_eq :
  (1 / 3) * (3 / 5) * (5 / 7) * (7 / 9) = 1 / 9 := by
  sorry

end fraction_product_eq_l139_139365


namespace total_players_correct_l139_139424

-- Define the number of players for each type of sport
def cricket_players : Nat := 12
def hockey_players : Nat := 17
def football_players : Nat := 11
def softball_players : Nat := 10

-- The theorem we aim to prove
theorem total_players_correct : 
  cricket_players + hockey_players + football_players + softball_players = 50 := by
  sorry

end total_players_correct_l139_139424


namespace slope_of_line_l139_139965

theorem slope_of_line (x y : ℝ) (h : x + 2 * y + 1 = 0) : y = - (1 / 2) * x - (1 / 2) :=
by
  sorry -- The solution would be filled in here

#check slope_of_line -- additional check to ensure theorem implementation is correct

end slope_of_line_l139_139965


namespace least_number_of_stamps_is_11_l139_139858

theorem least_number_of_stamps_is_11 (s t : ℕ) (h : 5 * s + 6 * t = 60) : s + t = 11 := 
  sorry

end least_number_of_stamps_is_11_l139_139858


namespace cos_20_cos_10_minus_sin_160_sin_10_l139_139093

theorem cos_20_cos_10_minus_sin_160_sin_10 : 
  (Real.cos (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) - 
   Real.sin (160 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)) = 
   Real.cos (30 * Real.pi / 180) :=
by
  sorry

end cos_20_cos_10_minus_sin_160_sin_10_l139_139093


namespace range_of_m_l139_139404

def proposition_p (m : ℝ) : Prop := (m^2 - 4 ≥ 0)
def proposition_q (m : ℝ) : Prop := (4 - 4 * m < 0)
def p_or_q (m : ℝ) : Prop := proposition_p m ∨ proposition_q m
def not_p (m : ℝ) : Prop := ¬ proposition_p m

theorem range_of_m (m : ℝ) (h1 : p_or_q m) (h2 : not_p m) : 1 < m ∧ m < 2 :=
sorry

end range_of_m_l139_139404


namespace janice_remaining_hours_l139_139433

def homework_time : ℕ := 30
def clean_room_time : ℕ := homework_time / 2
def walk_dog_time : ℕ := homework_time + 5
def trash_time : ℕ := homework_time / 6
def total_task_time : ℕ := homework_time + clean_room_time + walk_dog_time + trash_time
def remaining_minutes : ℕ := 35

theorem janice_remaining_hours : (remaining_minutes : ℚ) / 60 = (7 / 12 : ℚ) :=
by
  sorry

end janice_remaining_hours_l139_139433


namespace each_vaccine_costs_45_l139_139080

theorem each_vaccine_costs_45
    (num_vaccines : ℕ)
    (doctor_visit_cost : ℝ)
    (insurance_coverage : ℝ)
    (trip_cost : ℝ)
    (total_payment : ℝ) :
    num_vaccines = 10 ->
    doctor_visit_cost = 250 ->
    insurance_coverage = 0.80 ->
    trip_cost = 1200 ->
    total_payment = 1340 ->
    (∃ (vaccine_cost : ℝ), vaccine_cost = 45) :=
by {
    sorry
}

end each_vaccine_costs_45_l139_139080


namespace spent_on_burgers_l139_139628

noncomputable def money_spent_on_burgers (total_allowance : ℝ) (movie_fraction music_fraction ice_cream_fraction : ℝ) : ℝ :=
  let movie_expense := (movie_fraction * total_allowance)
  let music_expense := (music_fraction * total_allowance)
  let ice_cream_expense := (ice_cream_fraction * total_allowance)
  total_allowance - (movie_expense + music_expense + ice_cream_expense)

theorem spent_on_burgers : 
  money_spent_on_burgers 50 (1/4) (3/10) (2/5) = 2.5 :=
by sorry

end spent_on_burgers_l139_139628


namespace compute_expression_l139_139369

theorem compute_expression : 12 * (1 / 15) * 30 = 24 := 
by 
  sorry

end compute_expression_l139_139369


namespace value_of_expression_l139_139013

theorem value_of_expression (x y : ℤ) (h1 : x = 3) (h2 : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end value_of_expression_l139_139013


namespace initial_distance_between_jack_and_christina_l139_139028

theorem initial_distance_between_jack_and_christina
  (jack_speed : ℝ)
  (christina_speed : ℝ)
  (lindy_speed : ℝ)
  (lindy_total_distance : ℝ)
  (meeting_time : ℝ)
  (combined_speed : ℝ) :
  jack_speed = 5 ∧
  christina_speed = 3 ∧
  lindy_speed = 9 ∧
  lindy_total_distance = 270 ∧
  meeting_time = lindy_total_distance / lindy_speed ∧
  combined_speed = jack_speed + christina_speed →
  meeting_time = 30 ∧
  combined_speed = 8 →
  (combined_speed * meeting_time) = 240 :=
by
  sorry

end initial_distance_between_jack_and_christina_l139_139028


namespace one_neither_prime_nor_composite_l139_139919

/-- Definition of a prime number in the natural numbers -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Definition of a composite number in the natural numbers -/
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n 

/-- Theorem stating that the number 1 is neither prime nor composite -/
theorem one_neither_prime_nor_composite : ¬is_prime 1 ∧ ¬is_composite 1 :=
sorry

end one_neither_prime_nor_composite_l139_139919


namespace find_teacher_age_l139_139064

/-- Given conditions: 
1. The class initially has 30 students with an average age of 10.
2. One student aged 11 leaves the class.
3. The average age of the remaining 29 students plus the teacher is 11.
Prove that the age of the teacher is 30 years.
-/
theorem find_teacher_age (total_students : ℕ) (avg_age : ℕ) (left_student_age : ℕ) 
  (remaining_avg_age : ℕ) (teacher_age : ℕ) :
  total_students = 30 →
  avg_age = 10 →
  left_student_age = 11 →
  remaining_avg_age = 11 →
  289 + teacher_age = 29 * remaining_avg_age + teacher_age →
  teacher_age = 30 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end find_teacher_age_l139_139064


namespace arithmetic_sequence_26th_term_l139_139066

theorem arithmetic_sequence_26th_term (a d : ℤ) (h1 : a = 3) (h2 : a + d = 13) (h3 : a + 2 * d = 23) : 
  a + 25 * d = 253 :=
by
  -- specifications for variables a, d, and hypotheses h1, h2, h3
  sorry

end arithmetic_sequence_26th_term_l139_139066


namespace fixed_point_line_passes_through_range_of_t_l139_139266

-- Definition for first condition: Line with slope k (k ≠ 0)
variables {k : ℝ} (hk : k ≠ 0)

-- Definition for second condition: Ellipse C
def ellipse_C (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1

-- Third condition: Intersections M and N
variables (M N : ℝ × ℝ)
variables (intersection_M : ellipse_C M.1 M.2)
variables (intersection_N : ellipse_C N.1 N.2)

-- Fourth condition: Slopes are k1 and k2
variables {k1 k2 : ℝ}
variables (hk1 : k1 = M.2 / M.1)
variables (hk2 : k2 = N.2 / N.1)

-- Fifth condition: Given equation 3(k1 + k2) = 8k
variables (h_eq : 3 * (k1 + k2) = 8 * k)

-- Proof for question 1: Line passes through a fixed point
theorem fixed_point_line_passes_through 
    (h_eq : 3 * (k1 + k2) = 8 * k) : 
    ∃ n : ℝ, n = 1/2 ∨ n = -1/2 := sorry

-- Additional conditions for question 2
variables {D : ℝ × ℝ} (hD : D = (1, 0))
variables (t : ℝ)
variables (area_ratio : (M.2 / N.2) = t)
variables (h_ineq : k^2 < 5 / 12)

-- Proof for question 2: Range for t
theorem range_of_t
    (hD : D = (1, 0))
    (area_ratio : (M.2 / N.2) = t)
    (h_ineq : k^2 < 5 / 12) : 
    2 < t ∧ t < 3 ∨ 1 / 3 < t ∧ t < 1 / 2 := sorry

end fixed_point_line_passes_through_range_of_t_l139_139266


namespace initial_amount_simple_interest_l139_139855

theorem initial_amount_simple_interest 
  (A : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hA : A = 1125)
  (hR : R = 0.10)
  (hT : T = 5) :
  A = P * (1 + R * T) → P = 750 := 
by
  sorry

end initial_amount_simple_interest_l139_139855


namespace circle_center_l139_139599

theorem circle_center (x y : ℝ) : 
    (∃ x y : ℝ, x^2 - 8*x + y^2 - 4*y = 16) → (x, y) = (4, 2) := by
  sorry

end circle_center_l139_139599


namespace find_dividend_l139_139379

theorem find_dividend (k : ℕ) (quotient : ℕ) (dividend : ℕ) (h1 : k = 8) (h2 : quotient = 8) (h3 : dividend = k * quotient) : dividend = 64 := 
by 
  sorry

end find_dividend_l139_139379


namespace value_of_expression_l139_139011

theorem value_of_expression (x y : ℤ) (h1 : x = 3) (h2 : y = 2) : 3 * x - 4 * y = 1 := by
  rw [h1, h2]
  norm_num
  sorry

end value_of_expression_l139_139011


namespace solve_for_x_l139_139075

-- Define the operation *
def op (a b : ℝ) : ℝ := 2 * a - b

-- The theorem statement
theorem solve_for_x :
  (∃ x : ℝ, op x (op 1 3) = 2) ∧ (∀ x, op x -1 = 2)
  → x = 1/2 := by
  sorry

end solve_for_x_l139_139075


namespace g_5_l139_139178

variable (g : ℝ → ℝ)

axiom additivity_condition : ∀ (x y : ℝ), g (x + y) = g x + g y
axiom g_1_nonzero : g 1 ≠ 0

theorem g_5 : g 5 = 5 * g 1 :=
by
  sorry

end g_5_l139_139178


namespace trackball_mice_count_l139_139313

theorem trackball_mice_count (total_sales wireless_share optical_share : ℕ) 
    (h_total : total_sales = 80)
    (h_wireless : wireless_share = total_sales / 2)
    (h_optical : optical_share = total_sales / 4):
    total_sales - (wireless_share + optical_share) = 20 :=
by
  sorry

end trackball_mice_count_l139_139313


namespace find_original_number_l139_139356

def original_number (x : ℝ) : Prop :=
  let step1 := 1.20 * x
  let step2 := step1 * 0.85
  let final_value := step2 * 1.30
  final_value = 1080

theorem find_original_number : ∃ x : ℝ, original_number x :=
by
  use 1080 / (1.20 * 0.85 * 1.30)
  sorry

end find_original_number_l139_139356


namespace problem_inequality_sol1_problem_inequality_sol2_l139_139618

def f (x a : ℝ) : ℝ := x^2 - 2 * a * x - (2 * a + 2)

theorem problem_inequality_sol1 (a x : ℝ) :
  (a > -3 / 2 ∧ (x > 2 * a + 2 ∨ x < -1)) ∨
  (a = -3 / 2 ∧ x ≠ -1) ∨
  (a < -3 / 2 ∧ (x > -1 ∨ x < 2 * a + 2)) ↔
  f x a > x :=
sorry

theorem problem_inequality_sol2 (a : ℝ) :
  (∀ x : ℝ, x > -1 → f x a + 3 ≥ 0) ↔
  a ≤ Real.sqrt 2 - 1 :=
sorry

end problem_inequality_sol1_problem_inequality_sol2_l139_139618


namespace cars_meet_and_crush_fly_l139_139961

noncomputable def time_to_meet (L v_A v_B : ℝ) : ℝ := L / (v_A + v_B)

theorem cars_meet_and_crush_fly :
  ∀ (L v_A v_B v_fly : ℝ), L = 300 → v_A = 50 → v_B = 100 → v_fly = 150 → time_to_meet L v_A v_B = 2 :=
by
  intros L v_A v_B v_fly L_eq v_A_eq v_B_eq v_fly_eq
  rw [L_eq, v_A_eq, v_B_eq]
  simp [time_to_meet]
  norm_num

end cars_meet_and_crush_fly_l139_139961


namespace scale_division_remainder_l139_139643

theorem scale_division_remainder (a b c r : ℕ) (h1 : a = b * c + r) (h2 : 0 ≤ r) (h3 : r < b) :
  (3 * a) % (3 * b) = 3 * r :=
sorry

end scale_division_remainder_l139_139643


namespace part1_part2_l139_139388

-- Defining the function f(x) and the given conditions
def f (x a : ℝ) := x^2 - a * x + 2 * a - 2

-- Given conditions
variables (a : ℝ)
axiom f_condition : ∀ (x : ℝ), f (2 + x) a * f (2 - x) a = 4
axiom a_gt_0 : a > 0
axiom fx_bounds : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 4 → 1 ≤ f x a ∧ f x a ≤ 3

-- To prove (part 1)
theorem part1 (h : f 2 a + f 3 a = 6) : a = 2 := sorry

-- To prove (part 2)
theorem part2 : (4 - (2 * Real.sqrt 6) / 3) ≤ a ∧ a ≤ 5 / 2 := sorry

end part1_part2_l139_139388


namespace M_inter_N_eq_M_l139_139309

-- Definitions of the sets M and N
def M : Set ℝ := {x | abs (x - 1) < 1}
def N : Set ℝ := {x | x * (x - 3) < 0}

-- The desired equality
theorem M_inter_N_eq_M : M ∩ N = M := 
by
  sorry

end M_inter_N_eq_M_l139_139309


namespace probability_distance_greater_than_2_l139_139623

noncomputable def region (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3

noncomputable def regionArea : ℝ :=
  9

noncomputable def circle (x y : ℝ) : Prop :=
  x^2 + y^2 > 4

noncomputable def excludedCircleArea : ℝ :=
  9 - Real.pi

theorem probability_distance_greater_than_2 :
  ∫ (x y : ℝ) in {p | region p.1 p.2 ∧ circle p.1 p.2}, 1 ∂(MeasureTheory.Measure.prod MeasureTheory.measureSpace.volume MeasureTheory.measureSpace.volume) /
  regionArea = (9 - Real.pi) / 9 := by
sorry

end probability_distance_greater_than_2_l139_139623


namespace arrange_polynomial_ascending_order_l139_139853

variable {R : Type} [Ring R] (x : R)

def p : R := 3 * x ^ 2 - x + x ^ 3 - 1

theorem arrange_polynomial_ascending_order : 
  p x = -1 - x + 3 * x ^ 2 + x ^ 3 :=
by
  sorry

end arrange_polynomial_ascending_order_l139_139853


namespace union_set_eq_l139_139405

open Set

def P := {x : ℝ | 2 ≤ x ∧ x ≤ 3}
def Q := {x : ℝ | x^2 ≤ 4}

theorem union_set_eq : P ∪ Q = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by
  sorry

end union_set_eq_l139_139405


namespace total_apples_l139_139514

theorem total_apples (n : ℕ) (h1 : ∀ k : ℕ, 6 * k = n ∨ 12 * k = n) (h2 : 70 ≤ n) (h3 : n ≤ 80) : 
  n = 72 ∨ n = 78 := 
sorry

end total_apples_l139_139514


namespace kenya_peanuts_count_l139_139781

def peanuts_jose : ℕ := 85
def diff_kenya_jose : ℕ := 48
def peanuts_kenya : ℕ := peanuts_jose + diff_kenya_jose

theorem kenya_peanuts_count : peanuts_kenya = 133 := 
by
  -- proof goes here
  sorry

end kenya_peanuts_count_l139_139781


namespace Tim_marbles_l139_139262

theorem Tim_marbles (Fred_marbles : ℕ) (Tim_marbles : ℕ) (h1 : Fred_marbles = 110) (h2 : Fred_marbles = 22 * Tim_marbles) : 
  Tim_marbles = 5 :=
by
  sorry

end Tim_marbles_l139_139262


namespace nonnegative_for_interval_l139_139381

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 * (x - 2)^2) / ((1 - x) * (1 + x + x^2))

theorem nonnegative_for_interval (x : ℝ) : (f x >= 0) ↔ (0 <= x) :=
by
  sorry

end nonnegative_for_interval_l139_139381


namespace sector_angle_degree_measure_l139_139577

-- Define the variables and conditions
variables (θ r : ℝ)
axiom h1 : (1 / 2) * θ * r^2 = 1
axiom h2 : 2 * r + θ * r = 4

-- Define the theorem to be proved
theorem sector_angle_degree_measure (θ r : ℝ) (h1 : (1 / 2) * θ * r^2 = 1) (h2 : 2 * r + θ * r = 4) : θ = 2 :=
sorry

end sector_angle_degree_measure_l139_139577


namespace probability_at_least_three_heads_l139_139462

theorem probability_at_least_three_heads :
  let outcomes := Finset.powerset (Finset.range 5)
  let favorable := outcomes.filter (λ s, s.card ≥ 3)
  (favorable.card : ℚ) / outcomes.card = 1 / 2 :=
by
  sorry

end probability_at_least_three_heads_l139_139462


namespace avg_visitors_per_day_l139_139254

theorem avg_visitors_per_day :
  let visitors := [583, 246, 735, 492, 639]
  (visitors.sum / visitors.length) = 539 := by
  sorry

end avg_visitors_per_day_l139_139254


namespace geometric_sum_ratio_l139_139634

theorem geometric_sum_ratio (a₁ q : ℝ) (h₁ : q ≠ 1) (h₂ : (1 - q^4) / (1 - q^2) = 5) :
  (1 - q^8) / (1 - q^4) = 17 := 
by
  sorry

end geometric_sum_ratio_l139_139634


namespace chord_line_equation_l139_139898

/-- 
  Given the parabola y^2 = 4x and a chord AB 
  that exactly bisects at point P(1,1), prove 
  that the equation of the line on which chord AB lies is 2x - y - 1 = 0.
-/
theorem chord_line_equation (x y : ℝ) 
  (hx : y^2 = 4 * x)
  (bisect : ∃ A B : ℝ × ℝ, 
             (A.1^2 = 4 * A.2) ∧ (B.1^2 = 4 * B.2) ∧
             (A.1 + B.1 = 2 * 1) ∧ (A.2 + B.2 = 2 * 1)) :
  2 * x - y - 1 = 0 := sorry

end chord_line_equation_l139_139898


namespace prob_at_least_two_diamonds_or_aces_in_three_draws_l139_139224

noncomputable def prob_at_least_two_diamonds_or_aces: ℚ :=
  580 / 2197

def cards_drawn (draws: ℕ) : Prop :=
  draws = 3

def cards_either_diamonds_or_aces: ℕ :=
  16

theorem prob_at_least_two_diamonds_or_aces_in_three_draws:
  cards_drawn 3 →
  cards_either_diamonds_or_aces = 16 →
  prob_at_least_two_diamonds_or_aces = 580 / 2197 :=
  by
  intros
  sorry

end prob_at_least_two_diamonds_or_aces_in_three_draws_l139_139224


namespace ordinate_of_point_A_l139_139321

noncomputable def p : ℝ := 1 / 4
noncomputable def distance_to_focus (y₀ : ℝ) : ℝ := y₀ + p / 2

theorem ordinate_of_point_A :
  ∃ y₀ : ℝ, (distance_to_focus y₀ = 9 / 8) → y₀ = 1 :=
by
  -- Assume solution steps here
  sorry

end ordinate_of_point_A_l139_139321


namespace apple_bags_l139_139522

theorem apple_bags (n : ℕ) (h₁ : n ≥ 70) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 := 
sorry

end apple_bags_l139_139522


namespace determine_m_l139_139017

theorem determine_m (a b : ℝ) (m : ℝ) :
  (a^2 + 2 * a * b - b^2) - (a^2 + m * a * b + 2 * b^2) = (2 - m) * a * b - 3 * b^2 →
  (∀ a b : ℝ, (2 - m) * a * b = 0) →
  m = 2 :=
by
  sorry

end determine_m_l139_139017


namespace find_misread_solution_l139_139839

theorem find_misread_solution:
  ∃ a b : ℝ, 
  a = 5 ∧ b = 2 ∧ 
    (a^2 - 2 * a * b + b^2 = 9) ∧ 
    (∀ x y : ℝ, (5 * x + 4 * y = 23) ∧ (3 * x - 2 * y = 5) → (x = 3) ∧ (y = 2)) := by
    sorry

end find_misread_solution_l139_139839


namespace james_puzzle_completion_time_l139_139029

theorem james_puzzle_completion_time :
  let num_puzzles := 2
  let pieces_per_puzzle := 2000
  let pieces_per_10_minutes := 100
  let total_pieces := num_puzzles * pieces_per_puzzle
  let sets_of_100_pieces := total_pieces / pieces_per_10_minutes
  let total_minutes := sets_of_100_pieces * 10
  total_minutes = 400 :=
by
  -- Definitions based on conditions
  let num_puzzles := 2
  let pieces_per_puzzle := 2000
  let pieces_per_10_minutes := 100
  let total_pieces := num_puzzles * pieces_per_puzzle
  let sets_of_100_pieces := total_pieces / pieces_per_10_minutes
  let total_minutes := sets_of_100_pieces * 10

  -- Using sorry to skip proof
  sorry

end james_puzzle_completion_time_l139_139029


namespace total_depreciation_correct_residual_value_correct_sales_price_correct_l139_139559

-- Definitions and conditions
def initial_cost := 500000
def max_capacity := 100000
def jul_bottles := 200
def aug_bottles := 15000
def sep_bottles := 12300

def depreciation_per_bottle := initial_cost / max_capacity

-- Part (a)
def total_depreciation_jul := jul_bottles * depreciation_per_bottle
def total_depreciation_aug := aug_bottles * depreciation_per_bottle
def total_depreciation_sep := sep_bottles * depreciation_per_bottle
def total_depreciation := total_depreciation_jul + total_depreciation_aug + total_depreciation_sep

theorem total_depreciation_correct :
  total_depreciation = 137500 := 
by sorry

-- Part (b)
def residual_value := initial_cost - total_depreciation

theorem residual_value_correct :
  residual_value = 362500 := 
by sorry

-- Part (c)
def desired_profit := 10000
def sales_price := residual_value + desired_profit

theorem sales_price_correct :
  sales_price = 372500 := 
by sorry

end total_depreciation_correct_residual_value_correct_sales_price_correct_l139_139559


namespace opposite_number_in_circle_l139_139935

theorem opposite_number_in_circle (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 200) (h3 : ∀ k, 1 ≤ k ∧ k ≤ 200 → ∃ m, (m = (k + 100) % 200) ∧ (m ≠ k) ∧ (n + 100) % 200 < k):
  ∃ m : ℕ, m = 114 ∧ (113 + 100) % 200 = m :=
by
  sorry

end opposite_number_in_circle_l139_139935


namespace percentage_increase_in_llama_cost_l139_139338

def cost_of_goat : ℕ := 400
def number_of_goats : ℕ := 3
def total_cost : ℕ := 4800

def llamas_cost (x : ℕ) : Prop :=
  let total_cost_goats := number_of_goats * cost_of_goat
  let total_cost_llamas := total_cost - total_cost_goats
  let number_of_llamas := 2 * number_of_goats
  let cost_per_llama := total_cost_llamas / number_of_llamas
  let increase := cost_per_llama - cost_of_goat
  ((increase / cost_of_goat) * 100) = x

theorem percentage_increase_in_llama_cost :
  llamas_cost 50 :=
sorry

end percentage_increase_in_llama_cost_l139_139338


namespace stockholm_to_uppsala_distance_l139_139102

-- Definition of conditions
def map_distance : ℝ := 45 -- in cm
def scale1 : ℝ := 10 -- first scale 1 cm : 10 km
def scale2 : ℝ := 5 -- second scale 1 cm : 5 km
def boundary : ℝ := 15 -- first 15 cm at scale 2

-- Calculation of the two parts
def part1_distance (boundary : ℝ) (scale2 : ℝ) := boundary * scale2
def remaining_distance (map_distance boundary : ℝ) := map_distance - boundary
def part2_distance (remaining_distance : ℝ) (scale1 : ℝ) := remaining_distance * scale1

-- Total distance
def total_distance (part1 part2: ℝ) := part1 + part2

theorem stockholm_to_uppsala_distance : 
  total_distance (part1_distance boundary scale2) 
                 (part2_distance (remaining_distance map_distance boundary) scale1) 
  = 375 := 
by
  -- Proof to be provided
  sorry

end stockholm_to_uppsala_distance_l139_139102


namespace candidate_final_score_l139_139847

/- Given conditions -/
def interview_score : ℤ := 80
def written_test_score : ℤ := 90
def interview_weight : ℤ := 3
def written_test_weight : ℤ := 2

/- Final score computation -/
noncomputable def final_score : ℤ :=
  (interview_score * interview_weight + written_test_score * written_test_weight) / (interview_weight + written_test_weight)

theorem candidate_final_score : final_score = 84 := 
by
  sorry

end candidate_final_score_l139_139847


namespace sum_of_solutions_eq_l139_139695

theorem sum_of_solutions_eq :
  let A := 100
  let B := 3
  (∃ x₁ x₂ x₃ : ℝ, 
    (x₁ = abs (B*x₁ - abs (A - B*x₁)) ∧ 
    x₂ = abs (B*x₂ - abs (A - B*x₂)) ∧ 
    x₃ = abs (B*x₃ - abs (A - B*x₃))) ∧ 
    (x₁ + x₂ + x₃ = (1900 : ℝ) / 7)) :=
by
  sorry

end sum_of_solutions_eq_l139_139695


namespace inequality_solution_set_l139_139767

theorem inequality_solution_set {m n : ℝ} (h : ∀ x : ℝ, -3 < x ∧ x < 6 ↔ x^2 - m * x - 6 * n < 0) : m + n = 6 :=
by
  sorry

end inequality_solution_set_l139_139767


namespace smoking_lung_cancer_problem_l139_139429

-- Defining the confidence relationship
def smoking_related_to_lung_cancer (confidence: ℝ) := confidence > 0.99

-- Statement 4: Among 100 smokers, it is possible that not a single person has lung cancer.
def statement_4 (N: ℕ) (p: ℝ) := N = 100 ∧ 0 ≤ p ∧ p ≤ 1 ∧ p ^ 100 > 0

-- The main theorem statement in Lean 4
theorem smoking_lung_cancer_problem (confidence: ℝ) (N: ℕ) (p: ℝ) 
  (h1: smoking_related_to_lung_cancer confidence): 
  statement_4 N p :=
by
  sorry -- Proof goes here

end smoking_lung_cancer_problem_l139_139429


namespace minimum_value_of_f_ge_7_l139_139875

noncomputable def f (x : ℝ) : ℝ :=
  x + (2 * x) / (x^2 + 1) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

theorem minimum_value_of_f_ge_7 {x : ℝ} (hx : x > 0) : f x ≥ 7 := 
by
  sorry

end minimum_value_of_f_ge_7_l139_139875


namespace opposite_113_eq_114_l139_139937

-- Define the problem setup and conditions
variable (numbers : List ℕ) (n : ℕ)
variable (h1 : numbers.length = 200)
variable (h2 : numbers.nodup)
variable (h3 : ∀ k : ℕ, k ∈ numbers ↔ k ∈ (List.range 1 201))
variable (h4 : 1 ≤ n ∧ n ≤ 200)

-- Define the function to find the opposite number in the circle
def opposite (n : ℕ) : ℕ :=
  (n + 100) % 200

-- The proof statement that needs to be proven
theorem opposite_113_eq_114 : opposite 113 = 114 :=
  sorry

end opposite_113_eq_114_l139_139937


namespace matrix_A_pow_50_l139_139032

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![5, 1], ![-16, -3]]

theorem matrix_A_pow_50 :
  A ^ 50 = ![![201, 50], ![-800, -199]] :=
sorry

end matrix_A_pow_50_l139_139032


namespace bob_pennies_l139_139288

variable (a b : ℕ)

theorem bob_pennies : 
  (b + 2 = 4 * (a - 2)) →
  (b - 3 = 3 * (a + 3)) →
  b = 78 :=
by
  intros h1 h2
  sorry

end bob_pennies_l139_139288


namespace find_m_when_power_function_decreasing_l139_139290

theorem find_m_when_power_function_decreasing :
  ∃ m : ℝ, (m^2 - 2 * m - 2 = 1) ∧ (-4 * m - 2 < 0) ∧ (m = 3) :=
by
  sorry

end find_m_when_power_function_decreasing_l139_139290


namespace num_supermarkets_in_US_l139_139477

theorem num_supermarkets_in_US (U C : ℕ) (h1 : U + C = 420) (h2 : U = C + 56) : U = 238 :=
by
  sorry

end num_supermarkets_in_US_l139_139477


namespace weng_hourly_rate_l139_139202

theorem weng_hourly_rate (minutes_worked : ℝ) (earnings : ℝ) (fraction_of_hour : ℝ) 
  (conversion_rate : ℝ) (hourly_rate : ℝ) : 
  minutes_worked = 50 → earnings = 10 → 
  fraction_of_hour = minutes_worked / conversion_rate → 
  conversion_rate = 60 → 
  hourly_rate = earnings / fraction_of_hour → 
  hourly_rate = 12 := by
    sorry

end weng_hourly_rate_l139_139202


namespace students_solved_both_l139_139298

theorem students_solved_both (total_students solved_set_problem solved_function_problem both_problems_wrong: ℕ) 
  (h1: total_students = 50)
  (h2 : solved_set_problem = 40)
  (h3 : solved_function_problem = 31)
  (h4 : both_problems_wrong = 4) :
  (solved_set_problem + solved_function_problem - x + both_problems_wrong = total_students) → x = 25 := by
  sorry

end students_solved_both_l139_139298


namespace circle_bisection_relation_l139_139631

theorem circle_bisection_relation (a b : ℝ) :
  (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = b^2 + 1 → (x + 1)^2 + (y + 1)^2 = 4) ↔ 
  a^2 + 2 * a + 2 * b + 5 = 0 :=
by sorry

end circle_bisection_relation_l139_139631


namespace maximum_n_l139_139130

noncomputable def a1 : ℝ := sorry -- define a1 solving a_5 equations
noncomputable def q : ℝ := sorry -- define q solving a_5 and a_6 + a_7 equations
noncomputable def sn (n : ℕ) : ℝ := (a1 * (1 - q^n)) / (1 - q)  -- S_n of geometric series with a1 and q
noncomputable def pin (n : ℕ) : ℝ := (a1 * (q^((1 + n) * n / 2 - (11 * n) / 2 + 19 / 2)))  -- Pi solely in terms of n, a1, and q

theorem maximum_n (n : ℕ) (h1 : (a1 : ℝ) > 0) (h2 : q > 0) (h3 : q ≠ 1)
(h4 : a1 * q^4 = 1 / 4) (h5 : a1 * q^5 + a1 * q^6 = 3 / 2) :
  ∃ n : ℕ, sn n > pin n ∧ ∀ m : ℕ, m > 13 → sn m ≤ pin m := sorry

end maximum_n_l139_139130


namespace find_number_l139_139103

theorem find_number (x : ℝ) (h : x / 100 = 31.76 + 0.28) : x = 3204 := 
  sorry

end find_number_l139_139103


namespace line_symmetric_to_itself_l139_139127

theorem line_symmetric_to_itself :
  ∀ x y : ℝ, y = 3 * x + 3 ↔ ∃ (m b : ℝ), y = m * x + b ∧ m = 3 ∧ b = 3 :=
by
  sorry

end line_symmetric_to_itself_l139_139127


namespace number_of_dots_on_A_number_of_dots_on_B_number_of_dots_on_C_number_of_dots_on_D_l139_139193

variable (A B C D : ℕ)
variable (dice : ℕ → ℕ)

-- Conditions
axiom dice_faces : ∀ {i : ℕ}, 1 ≤ i ∧ i ≤ 6 → ∃ j, dice i = j
axiom opposite_faces_sum : ∀ {i j : ℕ}, dice i + dice j = 7
axiom configuration : True -- Placeholder for the specific arrangement configuration

-- Questions and Proof Statements
theorem number_of_dots_on_A :
  A = 3 := sorry

theorem number_of_dots_on_B :
  B = 5 := sorry

theorem number_of_dots_on_C :
  C = 6 := sorry

theorem number_of_dots_on_D :
  D = 5 := sorry

end number_of_dots_on_A_number_of_dots_on_B_number_of_dots_on_C_number_of_dots_on_D_l139_139193


namespace expression_evaluates_to_one_l139_139552

noncomputable def a := Real.sqrt 2 + 0.8
noncomputable def b := Real.sqrt 2 - 0.2

theorem expression_evaluates_to_one : 
  ( (2 - b) / (b - 1) + 2 * (a - 1) / (a - 2) ) / ( b * (a - 1) / (b - 1) + a * (2 - b) / (a - 2) ) = 1 :=
by
  sorry

end expression_evaluates_to_one_l139_139552


namespace black_balls_probability_both_black_l139_139095

theorem black_balls_probability_both_black (balls_total balls_black balls_gold : ℕ) (prob : ℚ) 
  (h1 : balls_total = 11)
  (h2 : balls_black = 7)
  (h3 : balls_gold = 4)
  (h4 : balls_total = balls_black + balls_gold)
  (h5 : prob = (21 : ℚ) / 55) :
  balls_total.choose 2 * prob = balls_black.choose 2 :=
sorry

end black_balls_probability_both_black_l139_139095


namespace complete_square_transform_l139_139473

theorem complete_square_transform :
  ∀ x : ℝ, x^2 - 4 * x - 6 = 0 → (x - 2)^2 = 10 :=
by
  intros x h
  sorry

end complete_square_transform_l139_139473


namespace det_abs_eq_one_l139_139648

variable {n : ℕ}
variable {A : Matrix (Fin n) (Fin n) ℤ}
variable {p q r : ℕ}
variable (hpq : p^2 = q^2 + r^2)
variable (hodd : Odd r)
variable (hA : p^2 • A ^ p^2 = q^2 • A ^ q^2 + r^2 • 1)

theorem det_abs_eq_one : |A.det| = 1 := by
  sorry

end det_abs_eq_one_l139_139648


namespace ball_bouncing_height_l139_139987

theorem ball_bouncing_height : ∃ (b : ℕ), 400 * (3/4 : ℝ)^b < 50 ∧ ∀ n < b, 400 * (3/4 : ℝ)^n ≥ 50 :=
by
  use 8
  sorry

end ball_bouncing_height_l139_139987


namespace train_speed_in_kmh_l139_139238

theorem train_speed_in_kmh (length_of_train : ℕ) (time_to_cross : ℕ) (speed_in_m_per_s : ℕ) (speed_in_km_per_h : ℕ) :
  length_of_train = 300 →
  time_to_cross = 12 →
  speed_in_m_per_s = length_of_train / time_to_cross →
  speed_in_km_per_h = speed_in_m_per_s * 3600 / 1000 →
  speed_in_km_per_h = 90 :=
by
  sorry

end train_speed_in_kmh_l139_139238


namespace condition1_condition2_condition3_l139_139132

noncomputable def Z (m : ℝ) : ℂ := (m^2 - 4 * m) + (m^2 - m - 6) * Complex.I

-- Condition 1: Point Z is in the third quadrant
theorem condition1 (m : ℝ) (h_quad3 : (m^2 - 4 * m) < 0 ∧ (m^2 - m - 6) < 0) : 0 < m ∧ m < 3 :=
sorry

-- Condition 2: Point Z is on the imaginary axis
theorem condition2 (m : ℝ) (h_imaginary : (m^2 - 4 * m) = 0 ∧ (m^2 - m - 6) ≠ 0) : m = 0 ∨ m = 4 :=
sorry

-- Condition 3: Point Z is on the line x - y + 3 = 0
theorem condition3 (m : ℝ) (h_line : (m^2 - 4 * m) - (m^2 - m - 6) + 3 = 0) : m = 3 :=
sorry

end condition1_condition2_condition3_l139_139132


namespace quadratic_rewrite_as_square_of_binomial_plus_integer_l139_139003

theorem quadratic_rewrite_as_square_of_binomial_plus_integer :
    ∃ a b, ∀ x, x^2 + 16 * x + 72 = (x + a)^2 + b ∧ b = 8 :=
by
  sorry

end quadratic_rewrite_as_square_of_binomial_plus_integer_l139_139003


namespace polynomial_equality_l139_139764

theorem polynomial_equality (x y : ℝ) (h₁ : 3 * x + 2 * y = 6) (h₂ : 2 * x + 3 * y = 7) : 
  14 * x^2 + 25 * x * y + 14 * y^2 = 85 := 
by
  sorry

end polynomial_equality_l139_139764


namespace even_product_probability_l139_139263

theorem even_product_probability
    (die_faces : Finset ℕ)
    (h_die : ∀ n ∈ die_faces, 1 ≤ n ∧ n ≤ 8)
    (h_size : die_faces.card = 8):
    let outcomes := (die_faces.product die_faces).filter (λ p, (p.1 * p.2) % 2 = 0)
    in (outcomes.card / (die_faces.card * die_faces.card) : ℚ) = 3 / 4 :=
by
  sorry

end even_product_probability_l139_139263


namespace trackball_mice_count_l139_139318

theorem trackball_mice_count
  (total_mice : ℕ)
  (wireless_fraction : ℕ)
  (optical_fraction : ℕ)
  (h_total : total_mice = 80)
  (h_wireless : wireless_fraction = total_mice / 2)
  (h_optical : optical_fraction = total_mice / 4) :
  total_mice - (wireless_fraction + optical_fraction) = 20 :=
sorry

end trackball_mice_count_l139_139318


namespace find_bettys_balance_l139_139878

-- Define the conditions as hypotheses
def balance_in_bettys_account (B : ℕ) : Prop :=
  -- Gina has two accounts with a combined balance equal to $1,728
  (2 * (B / 4)) = 1728

-- State the theorem to be proven
theorem find_bettys_balance (B : ℕ) (h : balance_in_bettys_account B) : B = 3456 :=
by
  -- The proof is provided here as a "sorry"
  sorry

end find_bettys_balance_l139_139878


namespace desired_average_score_is_correct_l139_139953

-- Conditions
def average_score_9_tests : ℕ := 82
def score_10th_test : ℕ := 92

-- Desired average score
def desired_average_score : ℕ := 83

-- Total score for 10 tests
def total_score_10_tests (avg9 : ℕ) (score10 : ℕ) : ℕ :=
  9 * avg9 + score10

-- Main theorem statement to prove
theorem desired_average_score_is_correct :
  total_score_10_tests average_score_9_tests score_10th_test / 10 = desired_average_score :=
by
  sorry

end desired_average_score_is_correct_l139_139953


namespace valid_number_of_apples_l139_139507

theorem valid_number_of_apples (n : ℕ) : 
  (∃ k m : ℕ, n = 12 * k + 6 * m) ∧ (70 ≤ n ∧ n ≤ 80) ↔ (n = 72 ∨ n = 78) :=
by
  sorry

end valid_number_of_apples_l139_139507


namespace boat_speed_in_still_water_l139_139350

-- Definitions and conditions
def Vs : ℕ := 5  -- Speed of the stream in km/hr
def distance : ℕ := 135  -- Distance traveled in km
def time : ℕ := 5  -- Time in hours

-- Statement to prove
theorem boat_speed_in_still_water : 
  ((distance = (Vb + Vs) * time) -> Vb = 22) :=
by
  sorry

end boat_speed_in_still_water_l139_139350


namespace minimum_students_for_200_candies_l139_139293

theorem minimum_students_for_200_candies (candies : ℕ) (students : ℕ) (h_candies : candies = 200) : students = 21 :=
by
  sorry

end minimum_students_for_200_candies_l139_139293


namespace trackball_mice_count_l139_139317

theorem trackball_mice_count
  (total_mice : ℕ)
  (wireless_fraction : ℕ)
  (optical_fraction : ℕ)
  (h_total : total_mice = 80)
  (h_wireless : wireless_fraction = total_mice / 2)
  (h_optical : optical_fraction = total_mice / 4) :
  total_mice - (wireless_fraction + optical_fraction) = 20 :=
sorry

end trackball_mice_count_l139_139317


namespace valid_number_of_apples_l139_139513

theorem valid_number_of_apples (n : ℕ) : 
  (∃ k m : ℕ, n = 12 * k + 6 * m) ∧ (70 ≤ n ∧ n ≤ 80) ↔ (n = 72 ∨ n = 78) :=
by
  sorry

end valid_number_of_apples_l139_139513


namespace todd_ratio_boss_l139_139798

theorem todd_ratio_boss
  (total_cost : ℕ)
  (boss_contribution : ℕ)
  (employees_contribution : ℕ)
  (num_employees : ℕ)
  (each_employee_pay : ℕ) 
  (total_contributed : ℕ)
  (todd_contribution : ℕ) :
  total_cost = 100 →
  boss_contribution = 15 →
  num_employees = 5 →
  each_employee_pay = 11 →
  total_contributed = num_employees * each_employee_pay + boss_contribution →
  todd_contribution = total_cost - total_contributed →
  (todd_contribution : ℚ) / (boss_contribution : ℚ) = 2 := by
  sorry

end todd_ratio_boss_l139_139798


namespace probability_exactly_2_hits_probability_at_least_2_hits_probability_exactly_2_hits_third_hit_l139_139358

noncomputable def binomial_prob (n k : ℕ) (p : ℝ) : ℝ := 
  (Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))) * p^k * (1 - p)^(n - k)

theorem probability_exactly_2_hits : 
  binomial_prob 5 2 0.8 ≈ 0.05 := sorry

theorem probability_at_least_2_hits :
  1 - binomial_prob 5 0 0.8 - binomial_prob 5 1 0.8 ≈ 0.99 := sorry

theorem probability_exactly_2_hits_third_hit :
  0.8 * binomial_prob 4 1 0.8 ≈ 0.02 := sorry

end probability_exactly_2_hits_probability_at_least_2_hits_probability_exactly_2_hits_third_hit_l139_139358


namespace driver_spending_increase_l139_139632

theorem driver_spending_increase (P Q : ℝ) (X : ℝ) (h1 : 1.20 * P = (1 + 20 / 100) * P) (h2 : 0.90 * Q = (1 - 10 / 100) * Q) :
  (1 + X / 100) * (P * Q) = 1.20 * P * 0.90 * Q → X = 8 := 
by
  sorry

end driver_spending_increase_l139_139632


namespace fraction_meaningfulness_l139_139549

def fraction_is_meaningful (x : ℝ) : Prop :=
  x ≠ 3 / 2

theorem fraction_meaningfulness (x : ℝ) : 
  (2 * x - 3) ≠ 0 ↔ fraction_is_meaningful x :=
by
  sorry

end fraction_meaningfulness_l139_139549


namespace total_hats_l139_139968

theorem total_hats (B G : ℕ) (cost_blue cost_green total_cost green_quantity : ℕ)
  (h1 : cost_blue = 6)
  (h2 : cost_green = 7)
  (h3 : total_cost = 530)
  (h4 : green_quantity = 20)
  (h5 : G = green_quantity)
  (h6 : total_cost = B * cost_blue + G * cost_green) :
  B + G = 85 :=
by
  sorry

end total_hats_l139_139968


namespace opposite_number_113_is_114_l139_139946

theorem opposite_number_113_is_114 :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 200 → 
  (∀ k, (k + 100) % 200 ≠ 113 → 113 + 100 ≤ 200 → n = 113 →
  (k = 114)) :=
by
  intro n hn h_opposite
  sorry

end opposite_number_113_is_114_l139_139946


namespace greatest_value_of_b_l139_139736

noncomputable def solution : ℝ :=
  (3 + Real.sqrt 21) / 2

theorem greatest_value_of_b :
  ∀ b : ℝ, b^2 - 4 * b + 3 < -b + 6 → b ≤ solution :=
by
  intro b
  intro h
  sorry

end greatest_value_of_b_l139_139736


namespace total_loaves_served_l139_139983

-- Definitions based on the conditions provided
def wheat_bread_loaf : ℝ := 0.2
def white_bread_loaf : ℝ := 0.4

-- Statement that needs to be proven
theorem total_loaves_served : wheat_bread_loaf + white_bread_loaf = 0.6 := 
by
  sorry

end total_loaves_served_l139_139983


namespace inequality_x_y_l139_139392

theorem inequality_x_y (x y : ℝ) (h : x^12 + y^12 ≤ 2) : x^2 + y^2 + x^2 * y^2 ≤ 3 := 
sorry

end inequality_x_y_l139_139392


namespace valid_number_of_apples_l139_139510

theorem valid_number_of_apples (n : ℕ) : 
  (∃ k m : ℕ, n = 12 * k + 6 * m) ∧ (70 ≤ n ∧ n ≤ 80) ↔ (n = 72 ∨ n = 78) :=
by
  sorry

end valid_number_of_apples_l139_139510


namespace luke_points_per_round_l139_139933

-- Define the total number of points scored 
def totalPoints : ℕ := 8142

-- Define the number of rounds played
def rounds : ℕ := 177

-- Define the points gained per round which we need to prove
def pointsPerRound : ℕ := 46

-- Now, we can state: if Luke played 177 rounds and scored a total of 8142 points, then he gained 46 points per round
theorem luke_points_per_round :
  (totalPoints = 8142) → (rounds = 177) → (totalPoints / rounds = pointsPerRound) := by
  sorry

end luke_points_per_round_l139_139933


namespace opposite_number_113_is_114_l139_139948

-- Definitions based on conditions in the problem
def numbers_on_circle (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 200

def are_equal_distances (numbers : Finset ℕ) : Prop :=
  ∀ x, x ∈ numbers → ∃ y, y ∈ numbers ∧ dist x y = k  -- Placeholder, replace dist and k with appropriate logic

def equality_of_smaller_numbers (numbers : Finset ℕ) (x : ℕ) : Prop :=
  (number of numbers smaller than x in 99 clockwise numbers = 
      number of numbers smaller than x in 99 counterclockwise numbers)

-- Proof statement showing that number opposite to 113 is 114
theorem opposite_number_113_is_114 : ∀ numbers : Finset ℕ,
  numbers_on_circle ∧ are_equal_distances numbers ∧ equality_of_smaller_numbers numbers 113 →
  ∃ n, n = 114 :=
begin
  sorry
end

end opposite_number_113_is_114_l139_139948


namespace arithmetic_sequence_length_l139_139408

theorem arithmetic_sequence_length (a d : ℕ) (l : ℕ) (h_a : a = 6) (h_d : d = 4) (h_l : l = 154) :
  ∃ n : ℕ, l = a + (n-1) * d ∧ n = 38 :=
by
  use 38
  split
  { rw [h_a, h_d]
    calc 154 = 6 + (38 - 1) * 4 : by norm_num
          ... = 6 + 37 * 4       : by rfl
          ... = 6 + 148          : by norm_num
          ... = 154              : by norm_num }
  { rfl }

end arithmetic_sequence_length_l139_139408


namespace family_of_sets_properties_l139_139791

variable {X : Type}
variable {t n k : ℕ}
variable (A : Fin t → Set X)
variable (card : Set X → ℕ)
variable (h_card : ∀ (i j : Fin t), i ≠ j → card (A i ∩ A j) = k)

theorem family_of_sets_properties :
  (k = 0 → t ≤ n+1) ∧ (k ≠ 0 → t ≤ n) :=
by
  sorry

end family_of_sets_properties_l139_139791


namespace Sravan_travel_time_l139_139327

theorem Sravan_travel_time :
  let total_distance := 540
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let speed_first_half := 45
  let speed_second_half := 30
  let time_first_half := first_half_distance / speed_first_half
  let time_second_half := second_half_distance / speed_second_half
  let total_time := time_first_half + time_second_half
  total_time = 15 :=
by
  let total_distance := 540
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let speed_first_half := 45
  let speed_second_half := 30
  let time_first_half := first_half_distance / speed_first_half
  let time_second_half := second_half_distance / speed_second_half
  let total_time := time_first_half + time_second_half
  sorry

end Sravan_travel_time_l139_139327


namespace matchsticks_20th_stage_l139_139685

theorem matchsticks_20th_stage :
  let a1 := 3
  let d := 3
  let a20 := a1 + 19 * d
  a20 = 60 := by
  sorry

end matchsticks_20th_stage_l139_139685


namespace inequality_x_y_l139_139395

theorem inequality_x_y (x y : ℝ) (h : x^12 + y^12 ≤ 2) : x^2 + y^2 + x^2 * y^2 ≤ 3 := 
sorry

end inequality_x_y_l139_139395


namespace determine_female_athletes_count_l139_139582

theorem determine_female_athletes_count (m : ℕ) (n : ℕ) (x y : ℕ) (probability : ℚ)
  (h_team : 56 + m = 56 + m) -- redundant, but setting up context
  (h_sample_size : n = 28)
  (h_probability : probability = 1 / 28)
  (h_sample_diff : x - y = 4)
  (h_sample_sum : x + y = n)
  (h_ratio : 56 * y = m * x) : m = 42 :=
by
  sorry

end determine_female_athletes_count_l139_139582


namespace highest_vs_lowest_temp_difference_l139_139186

theorem highest_vs_lowest_temp_difference 
  (highest_temp lowest_temp : ℤ) 
  (h_highest : highest_temp = 26) 
  (h_lowest : lowest_temp = 14) : 
  highest_temp - lowest_temp = 12 := 
by 
  sorry

end highest_vs_lowest_temp_difference_l139_139186


namespace number_of_blobs_of_glue_is_96_l139_139595

def pyramid_blobs_of_glue : Nat :=
  let layer1 := 4 * (4 - 1) * 2
  let layer2 := 3 * (3 - 1) * 2
  let layer3 := 2 * (2 - 1) * 2
  let between1_and_2 := 3 * 3 * 4
  let between2_and_3 := 2 * 2 * 4
  let between3_and_4 := 4
  layer1 + layer2 + layer3 + between1_and_2 + between2_and_3 + between3_and_4

theorem number_of_blobs_of_glue_is_96 :
  pyramid_blobs_of_glue = 96 :=
by
  sorry

end number_of_blobs_of_glue_is_96_l139_139595


namespace variable_value_l139_139565

theorem variable_value 
  (x : ℝ)
  (a k some_variable : ℝ)
  (eqn1 : (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + some_variable)
  (eqn2 : a - some_variable + k = 3)
  (a_val : a = 6)
  (k_val : k = -17) :
  some_variable = -14 :=
by
  sorry

end variable_value_l139_139565


namespace rope_in_two_months_period_l139_139372

theorem rope_in_two_months_period :
  let week1 := 6
  let week2 := 3 * week1
  let week3 := week2 - 4
  let week4 := - (week2 / 2)
  let week5 := week1 + 2
  let week6 := - (2 / 2)
  let week7 := 3 * (2 / 2)
  let week8 := - 10
  let total_length := (week1 + week2 + week3 + week4 + week5 + week6 + week7 + week8)
  total_length * 12 = 348
:= sorry

end rope_in_two_months_period_l139_139372


namespace Douglas_won_72_percent_of_votes_in_county_X_l139_139638

/-- Definition of the problem conditions and the goal -/
theorem Douglas_won_72_percent_of_votes_in_county_X
  (V : ℝ)
  (total_votes_ratio : ∀ county_X county_Y, county_X = 2 * county_Y)
  (total_votes_percentage_both_counties : 0.60 = (1.8 * V) / (2 * V + V))
  (votes_percentage_county_Y : 0.36 = (0.36 * V) / V) : 
  ∃ P : ℝ, P = 72 ∧ P = (1.44 * V) / (2 * V) * 100 :=
sorry

end Douglas_won_72_percent_of_votes_in_county_X_l139_139638


namespace Emma_investment_l139_139727

-- Define the necessary context and variables
variable (E : ℝ) -- Emma's investment
variable (B : ℝ := 500) -- Briana's investment which is a known constant
variable (ROI_Emma : ℝ := 0.30 * E) -- Emma's return on investment after 2 years
variable (ROI_Briana : ℝ := 0.20 * B) -- Briana's return on investment after 2 years
variable (ROI_difference : ℝ := ROI_Emma - ROI_Briana) -- The difference in their ROI

theorem Emma_investment :
  ROI_difference = 10 → E = 366.67 :=
by
  intros h
  sorry

end Emma_investment_l139_139727


namespace solve_for_x_l139_139154

theorem solve_for_x (x : ℝ) (h : 2 * x - 5 = 15) : x = 10 :=
sorry

end solve_for_x_l139_139154


namespace geometric_sequence_seventh_term_l139_139739

theorem geometric_sequence_seventh_term (a₁ : ℤ) (a₂ : ℚ) (r : ℚ) (k : ℕ) (a₇ : ℚ)
  (h₁ : a₁ = 3) 
  (h₂ : a₂ = -1 / 2)
  (h₃ : r = a₂ / a₁)
  (h₄ : k = 7)
  (h₅ : a₇ = a₁ * r^(k-1)) : 
  a₇ = 1 / 15552 := 
by
  sorry

end geometric_sequence_seventh_term_l139_139739


namespace largest_angle_in_isosceles_triangle_l139_139024

-- Definitions of the conditions from the problem
def isosceles_triangle (A B C : ℕ) : Prop :=
  A = B ∨ B = C ∨ A = C

def angle_opposite_equal_side (θ : ℕ) : Prop :=
  θ = 50

-- The proof problem statement
theorem largest_angle_in_isosceles_triangle (A B C : ℕ) (θ : ℕ)
  : isosceles_triangle A B C → angle_opposite_equal_side θ → ∃ γ, γ = 80 :=
by
  sorry

end largest_angle_in_isosceles_triangle_l139_139024


namespace arithmetic_sequence_identity_l139_139749

noncomputable def a (n: ℕ) : ℝ := sorry -- Assume this definition represents the arithmetic sequence

theorem arithmetic_sequence_identity :
  let a := a in
  let integral_value := ∫ x in 0..2, sqrt(4 - x^2) in
  a 4 + a 8 = integral_value →
  a 6 * (a 2 + 2 * a 6 + a 10) = π^2 := 
by
  sorry -- The proof step is not required

end arithmetic_sequence_identity_l139_139749


namespace bs_sequence_bounded_iff_f_null_l139_139997

def is_bs_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = abs (a (n + 1) - a (n + 2))

def f_null (a : ℕ → ℝ) : Prop :=
  ∀ n k, a n * a k * (a n - a k) = 0

def bs_bounded (a : ℕ → ℝ) : Prop :=
  ∃ M, ∀ n, abs (a n) ≤ M

theorem bs_sequence_bounded_iff_f_null (a : ℕ → ℝ) :
  is_bs_sequence a →
  (bs_bounded a ↔ f_null a) := by
  sorry

end bs_sequence_bounded_iff_f_null_l139_139997


namespace calculate_flat_tax_l139_139472

open Real

def price_per_sq_ft (property: String) : Real :=
  if property = "Condo" then 98
  else if property = "BarnHouse" then 84
  else if property = "DetachedHouse" then 102
  else if property = "Townhouse" then 96
  else if property = "Garage" then 60
  else if property = "PoolArea" then 50
  else 0

def area_in_sq_ft (property: String) : Real :=
  if property = "Condo" then 2400
  else if property = "BarnHouse" then 1200
  else if property = "DetachedHouse" then 3500
  else if property = "Townhouse" then 2750
  else if property = "Garage" then 480
  else if property = "PoolArea" then 600
  else 0

def total_value : Real :=
  (price_per_sq_ft "Condo" * area_in_sq_ft "Condo") +
  (price_per_sq_ft "BarnHouse" * area_in_sq_ft "BarnHouse") +
  (price_per_sq_ft "DetachedHouse" * area_in_sq_ft "DetachedHouse") +
  (price_per_sq_ft "Townhouse" * area_in_sq_ft "Townhouse") +
  (price_per_sq_ft "Garage" * area_in_sq_ft "Garage") +
  (price_per_sq_ft "PoolArea" * area_in_sq_ft "PoolArea")

def tax_rate : Real := 0.0125

theorem calculate_flat_tax : total_value * tax_rate = 12697.50 := by
  sorry

end calculate_flat_tax_l139_139472


namespace sufficient_condition_l139_139391

variable (a b c d : ℝ)

-- Condition p: a and b are the roots of the equation.
def condition_p : Prop := a * a + b * b + c * (a + b) + d = 0

-- Condition q: a + b + c = 0
def condition_q : Prop := a + b + c = 0

theorem sufficient_condition : condition_p a b c d → condition_q a b c := by
  sorry

end sufficient_condition_l139_139391


namespace num_quarters_left_l139_139804

-- Define initial amounts and costs
def initial_amount : ℝ := 40
def pizza_cost : ℝ := 2.75
def soda_cost : ℝ := 1.50
def jeans_cost : ℝ := 11.50
def quarter_value : ℝ := 0.25

-- Define the total amount spent
def total_spent : ℝ := pizza_cost + soda_cost + jeans_cost

-- Define the remaining amount
def remaining_amount : ℝ := initial_amount - total_spent

-- Prove the number of quarters left
theorem num_quarters_left : remaining_amount / quarter_value = 97 :=
by
  sorry

end num_quarters_left_l139_139804


namespace solve_system_l139_139378

def F (t : ℝ) : ℝ := 32 * t ^ 5 + 48 * t ^ 3 + 17 * t - 15

def system_of_equations (x y z : ℝ) : Prop :=
  (1 / x = (32 / y ^ 5) + (48 / y ^ 3) + (17 / y) - 15) ∧
  (1 / y = (32 / z ^ 5) + (48 / z ^ 3) + (17 / z) - 15) ∧
  (1 / z = (32 / x ^ 5) + (48 / x ^ 3) + (17 / x) - 15)

theorem solve_system : ∃ (x y z : ℝ), system_of_equations x y z ∧ x = 2 ∧ y = 2 ∧ z = 2 :=
by
  sorry -- Proof not included

end solve_system_l139_139378


namespace range_of_a_l139_139619

noncomputable def f (x a : ℝ) := (x^2 + a * x + 11) / (x + 1)

theorem range_of_a (a : ℝ) :
  (∀ x : ℕ, x > 0 → f x a ≥ 3) ↔ (a ≥ -8 / 3) :=
by sorry

end range_of_a_l139_139619


namespace sum_of_box_weights_l139_139291

theorem sum_of_box_weights (heavy_box_weight : ℚ) (difference : ℚ) 
  (h1 : heavy_box_weight = 14 / 15) (h2 : difference = 1 / 10) :
  heavy_box_weight + (heavy_box_weight - difference) = 53 / 30 := 
  by
  sorry

end sum_of_box_weights_l139_139291


namespace product_of_prs_l139_139412

theorem product_of_prs 
  (p r s : Nat) 
  (h1 : 3^p + 3^5 = 270) 
  (h2 : 2^r + 58 = 122) 
  (h3 : 7^2 + 5^s = 2504) : 
  p * r * s = 54 := 
sorry

end product_of_prs_l139_139412


namespace sum_of_perimeters_l139_139074

theorem sum_of_perimeters (x y : Real) 
  (h1 : x^2 + y^2 = 85)
  (h2 : x^2 - y^2 = 45) :
  4 * (Real.sqrt 65 + 2 * Real.sqrt 5) = 4 * x + 4 * y := by
  sorry

end sum_of_perimeters_l139_139074


namespace appears_every_number_smallest_triplicate_number_l139_139213

open Nat

/-- Pascal's triangle is constructed such that each number 
    is the sum of the two numbers directly above it in the 
    previous row -/
def pascal (r k : ℕ) : ℕ :=
  if k > r then 0 else Nat.choose r k

/-- Every positive integer does appear at least once, but not 
    necessarily more than once for smaller numbers -/
theorem appears_every_number (n : ℕ) : ∃ r k : ℕ, pascal r k = n := sorry

/-- The smallest three-digit number in Pascal's triangle 
    that appears more than once is 102 -/
theorem smallest_triplicate_number : ∃ r1 k1 r2 k2 : ℕ, 
  100 ≤ pascal r1 k1 ∧ pascal r1 k1 < 1000 ∧ 
  pascal r1 k1 = 102 ∧ 
  r1 ≠ r2 ∧ k1 ≠ k2 ∧ 
  pascal r1 k1 = pascal r2 k2 := sorry

end appears_every_number_smallest_triplicate_number_l139_139213


namespace sum_of_numbers_Carolyn_removes_l139_139115

noncomputable def game_carolyn_paul_sum : ℕ :=
  let initial_list := [1, 2, 3, 4, 5]
  let removed_by_paul := [3, 4]
  let removed_by_carolyn := [1, 2, 5]
  removed_by_carolyn.sum

theorem sum_of_numbers_Carolyn_removes :
  game_carolyn_paul_sum = 8 :=
by
  sorry

end sum_of_numbers_Carolyn_removes_l139_139115


namespace coefficient_x2_in_expansion_l139_139866

-- Definition of binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Definition of the problem: Given (2x + 1)^5, find the coefficient of x^2 term
theorem coefficient_x2_in_expansion : 
  binomial 5 3 * (2 ^ 2) = 40 := by 
  sorry

end coefficient_x2_in_expansion_l139_139866


namespace p_distinct_roots_iff_l139_139814

variables {p : ℝ}

def quadratic_has_distinct_roots (a b c : ℝ) : Prop :=
  (b^2 - 4 * a * c) > 0

theorem p_distinct_roots_iff (hp: p > 0 ∨ p = -1) :
  (∀ x : ℝ, x^2 - 2 * |x| - p = 0 → 
    (quadratic_has_distinct_roots 1 (-2) (-p) ∨
      quadratic_has_distinct_roots 1 2 (-p))) :=
by sorry

end p_distinct_roots_iff_l139_139814


namespace sphere_surface_area_l139_139890

theorem sphere_surface_area (R r : ℝ) (h1 : 2 * OM = R) (h2 : ∀ r, π * r^2 = 3 * π) : 4 * π * R^2 = 16 * π :=
by
  sorry

end sphere_surface_area_l139_139890


namespace verify_option_a_l139_139086

-- Define Option A's condition
def option_a_condition (a : ℝ) : Prop :=
  2 * a^2 - 4 * a + 2 = 2 * (a - 1)^2

-- State the theorem that Option A's factorization is correct
theorem verify_option_a (a : ℝ) : option_a_condition a := by sorry

end verify_option_a_l139_139086


namespace matrix_power_B_l139_139439

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 1, 0],
  ![0, 0, 1],
  ![1, 0, 0]
]

theorem matrix_power_B :
  B ^ 150 = 1 :=
by sorry

end matrix_power_B_l139_139439


namespace probability_point_inside_circle_l139_139636

theorem probability_point_inside_circle :
  (∃ (m n : ℕ), 1 ≤ m ∧ m ≤ 6 ∧ 1 ≤ n ∧ n ≤ 6) →
  (∃ (P : ℚ), P = 2/9) :=
by
  sorry

end probability_point_inside_circle_l139_139636


namespace probability_of_hitting_blue_zone_l139_139354

theorem probability_of_hitting_blue_zone :
  let P_red := (2 : ℚ) / 5
  let P_green := (1 : ℚ) / 4
  ∃ P_blue : ℚ, P_blue = 1 - (P_red + P_green) ∧ P_blue = 7 / 20 :=
by
  let P_red := (2 : ℚ) / 5
  let P_green := (1 : ℚ) / 4
  let P_blue := 1 - (P_red + P_green)
  use P_blue
  split
  · exact rfl
  · sorry

end probability_of_hitting_blue_zone_l139_139354


namespace overlapping_area_of_thirty_sixty_ninety_triangles_l139_139195

-- Definitions for 30-60-90 triangle and the overlapping region
def thirty_sixty_ninety_triangle (hypotenuse : ℝ) := 
  (hypotenuse > 0) ∧ 
  (exists (short_leg long_leg : ℝ), short_leg = hypotenuse / 2 ∧ long_leg = short_leg * (Real.sqrt 3))

-- Area of a parallelogram given base and height
def parallelogram_area (base height : ℝ) : ℝ :=
  base * height

theorem overlapping_area_of_thirty_sixty_ninety_triangles :
  ∀ (hypotenuse : ℝ), thirty_sixty_ninety_triangle hypotenuse →
  hypotenuse = 10 →
  (∃ (base height : ℝ), base = height ∧ base * height = parallelogram_area (5 * Real.sqrt 3) (5 * Real.sqrt 3)) →
  parallelogram_area (5 * Real.sqrt 3) (5 * Real.sqrt 3) = 75 :=
by
  sorry

end overlapping_area_of_thirty_sixty_ninety_triangles_l139_139195


namespace intercepts_sum_eq_seven_l139_139185

theorem intercepts_sum_eq_seven :
    (∃ a b c, (∀ y, (3 * y^2 - 9 * y + 4 = a) → y = 0) ∧ 
              (∀ y, (3 * y^2 - 9 * y + 4 = 0) → (y = b ∨ y = c)) ∧ 
              (a + b + c = 7)) := 
sorry

end intercepts_sum_eq_seven_l139_139185


namespace sum_of_k_values_l139_139084

theorem sum_of_k_values :
  (∃ k, ∃ x, x^2 - 4*x + 3 = 0 ∧ x^2 - 6*x + k = 0) →
  (perfect_values = {5, 9}) →
  (∑ i in perfect_values, i = 14) := by
  sorry

end sum_of_k_values_l139_139084


namespace domain_of_f_l139_139341

open Set Real

noncomputable def f (x : ℝ) : ℝ := (x + 6) / sqrt (x^2 - 5 * x + 6)

theorem domain_of_f : {x : ℝ | ∃ y, y = f x} = {x : ℝ | x < 2 ∨ x > 3} :=
by
  sorry

end domain_of_f_l139_139341


namespace money_spent_l139_139005

def initial_money (Henry : Type) : ℤ := 11
def birthday_money (Henry : Type) : ℤ := 18
def final_money (Henry : Type) : ℤ := 19

theorem money_spent (Henry : Type) : (initial_money Henry + birthday_money Henry - final_money Henry = 10) := 
by sorry

end money_spent_l139_139005


namespace total_cost_is_correct_l139_139361

def bus_ride_cost : ℝ := 1.75
def train_ride_cost : ℝ := bus_ride_cost + 6.35
def total_cost : ℝ := bus_ride_cost + train_ride_cost

theorem total_cost_is_correct : total_cost = 9.85 :=
by
  -- proof here
  sorry

end total_cost_is_correct_l139_139361


namespace three_digit_numbers_with_distinct_digits_avg_condition_l139_139148

theorem three_digit_numbers_with_distinct_digits_avg_condition : 
  ∃ (S : Finset (Fin 1000)), 
  (∀ n ∈ S, (n / 100 ≠ (n / 10 % 10) ∧ (n / 100 ≠ n % 10) ∧ (n / 10 % 10 ≠ n % 10))) ∧
  (∀ n ∈ S, ((n / 100 + n % 10) / 2 = n / 10 % 10)) ∧
  (∀ n ∈ S, abs ((n / 100) - (n / 10 % 10)) ≤ 5 ∧ abs ((n / 10 % 10) - (n % 10)) ≤ 5) ∧
  S.card = 120 :=
sorry

end three_digit_numbers_with_distinct_digits_avg_condition_l139_139148


namespace intercept_sum_l139_139183

noncomputable def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 4

def x_intercept := 4

def y_intercepts : ℝ × ℝ :=
  let delta := (9 : ℝ)^2 - 4 * 3 * 4
  ((9 - Real.sqrt delta)/6, (9 + Real.sqrt delta)/6)

def a : ℝ := x_intercept
def b : ℝ := (y_intercepts.1)
def c : ℝ := (y_intercepts.2)

theorem intercept_sum : a + b + c = 7 := by
  have h_delta : (9 : ℝ)^2 - 4 * 3 * 4 = 33 := by
    sorry
  have h_b : b = (9 - Real.sqrt 33) / 6 := by
    simp only [b, y_intercepts]
    sorry
  have h_c : c = (9 + Real.sqrt 33) / 6 := by
    simp only [c, y_intercepts]
    sorry
  simp [a, b, c, h_b, h_c]
  field_simp
  ring
  sorry

end intercept_sum_l139_139183


namespace middle_number_is_nine_l139_139096

theorem middle_number_is_nine (x : ℝ) (h : (2 * x)^2 + (4 * x)^2 = 180) : 3 * x = 9 :=
by
  sorry

end middle_number_is_nine_l139_139096


namespace taxi_ride_cost_l139_139234

-- Define the fixed cost
def fixed_cost : ℝ := 2.00

-- Define the cost per mile
def cost_per_mile : ℝ := 0.30

-- Define the number of miles traveled
def miles_traveled : ℝ := 7.0

-- Define the total cost calculation
def total_cost : ℝ := fixed_cost + (cost_per_mile * miles_traveled)

-- Theorem: Prove the total cost of a 7-mile taxi ride is $4.10
theorem taxi_ride_cost : total_cost = 4.10 := by
  sorry

end taxi_ride_cost_l139_139234


namespace length_real_axis_l139_139897

theorem length_real_axis (x y : ℝ) : 
  (x^2 / 4 - y^2 / 12 = 1) → 4 = 4 :=
by
  intro h
  sorry

end length_real_axis_l139_139897


namespace min_val_of_f_l139_139873

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  x + (2 * x) / (x^2 + 1) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

-- Theorem stating the minimum value of f(x) for x > 0 is 5.5
theorem min_val_of_f : ∀ x : ℝ, x > 0 → f x ≥ 5.5 :=
by sorry

end min_val_of_f_l139_139873


namespace zebra_difference_is_zebra_l139_139047

/-- 
A zebra number is a non-negative integer in which the digits strictly alternate between even and odd.
Given two 100-digit zebra numbers, prove that their difference is still a 100-digit zebra number.
-/
theorem zebra_difference_is_zebra 
  (A B : ℕ) 
  (hA : (∀ i, (A / 10^i % 10) % 2 = i % 2) ∧ (A / 10^100 = 0) ∧ (A > 10^99))
  (hB : (∀ i, (B / 10^i % 10) % 2 = i % 2) ∧ (B / 10^100 = 0) ∧ (B > 10^99)) 
  : (∀ j, (((A - B) / 10^j) % 10) % 2 = j % 2) ∧ ((A - B) / 10^100 = 0) ∧ ((A - B) > 10^99) :=
sorry

end zebra_difference_is_zebra_l139_139047


namespace number_of_cars_l139_139242

theorem number_of_cars (x : ℕ) (h : 3 * (x - 2) = 2 * x + 9) : x = 15 :=
by {
  sorry
}

end number_of_cars_l139_139242


namespace nine_b_equals_eighteen_l139_139413

theorem nine_b_equals_eighteen (a b : ℝ) (h1 : 6 * a + 3 * b = 0) (h2 : a = b - 3) : 9 * b = 18 :=
  sorry

end nine_b_equals_eighteen_l139_139413


namespace softball_players_count_l139_139423

theorem softball_players_count :
  ∀ (cricket hockey football total_players softball : ℕ),
  cricket = 15 →
  hockey = 12 →
  football = 13 →
  total_players = 55 →
  total_players = cricket + hockey + football + softball →
  softball = 15 :=
by
  intros cricket hockey football total_players softball h_cricket h_hockey h_football h_total_players h_total
  sorry

end softball_players_count_l139_139423


namespace find_x_l139_139572

theorem find_x
  (x : ℝ)
  (h1 : (x - 2)^2 + (15 - 5)^2 = 13^2)
  (h2 : x > 0) : 
  x = 2 + Real.sqrt 69 :=
sorry

end find_x_l139_139572


namespace sum_n_max_value_l139_139382

noncomputable def arithmetic_sequence (a_1 : Int) (d : Int) (n : Nat) : Int :=
  a_1 + (n - 1) * d

noncomputable def sum_arithmetic_sequence (a_1 : Int) (d : Int) (n : Nat) : Int :=
  n * a_1 + (n * (n - 1) / 2) * d

theorem sum_n_max_value :
  (∃ n : Nat, n = 9 ∧ sum_arithmetic_sequence 25 (-3) n = 117) :=
by
  let a1 := 25
  let d := -3
  use 9
  -- To complete the proof, we would calculate the sum of the first 9 terms
  -- of the arithmetic sequence with a1 = 25 and difference d = -3.
  sorry

end sum_n_max_value_l139_139382


namespace fill_40x41_table_l139_139027

-- Define the condition on integers in the table
def valid_integer_filling (m n : ℕ) (table : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, i < m → j < n →
    table i j =
    ((if i > 0 then if table i j = table (i - 1) j then 1 else 0 else 0) +
    (if j > 0 then if table i j = table i (j - 1) then 1 else 0 else 0) +
    (if i < m - 1 then if table i j = table (i + 1) j then 1 else 0 else 0) +
    (if j < n - 1 then if table i j = table i (j + 1) then 1 else 0 else 0))

-- Define the specific problem for a 40 × 41 table.
theorem fill_40x41_table :
  ∃ (table : ℕ → ℕ → ℕ), valid_integer_filling 40 41 table :=
by
  sorry

end fill_40x41_table_l139_139027


namespace compare_magnitude_l139_139880

theorem compare_magnitude (a b : ℝ) (h : a ≠ 1) : a^2 + b^2 > 2 * (a - b - 1) :=
by
  sorry

end compare_magnitude_l139_139880


namespace perimeter_of_regular_pentagon_is_75_l139_139078

-- Define the side length and the property of the figure
def side_length : ℝ := 15
def is_regular_pentagon : Prop := true  -- assuming this captures the regular pentagon property

-- Define the perimeter calculation based on the conditions
def perimeter (n : ℕ) (side_length : ℝ) := n * side_length

-- The theorem to prove
theorem perimeter_of_regular_pentagon_is_75 :
  is_regular_pentagon → perimeter 5 side_length = 75 :=
by
  intro _ -- We don't need to use is_regular_pentagon directly
  rw [side_length]
  norm_num
  sorry

end perimeter_of_regular_pentagon_is_75_l139_139078


namespace area_enclosed_by_curve_l139_139176

theorem area_enclosed_by_curve :
  let arc_length := (3 * Real.pi) / 4
  let side_length := 3
  let radius := arc_length / ((3 * Real.pi) / 4)
  let sector_area := (radius ^ 2 * Real.pi * (3 * Real.pi) / (4 * 2 * Real.pi))
  let total_sector_area := 8 * sector_area
  let octagon_area := 2 * (1 + Real.sqrt 2) * (side_length ^ 2)
  total_sector_area + octagon_area = 54 + 54 * Real.sqrt 2 + 3 * Real.pi
:= sorry

end area_enclosed_by_curve_l139_139176


namespace Aren_listening_time_l139_139110

/--
Aren’s flight from New York to Hawaii will take 11 hours 20 minutes. He spends 2 hours reading, 
4 hours watching two movies, 30 minutes eating his dinner, some time listening to the radio, 
and 1 hour 10 minutes playing games. He has 3 hours left to take a nap. 
Prove that he spends 40 minutes listening to the radio.
-/
theorem Aren_listening_time 
  (total_flight_time : ℝ := 11 * 60 + 20)
  (reading_time : ℝ := 2 * 60)
  (watching_movies_time : ℝ := 4 * 60)
  (eating_dinner_time : ℝ := 30)
  (playing_games_time : ℝ := 1 * 60 + 10)
  (nap_time : ℝ := 3 * 60) :
  total_flight_time - (reading_time + watching_movies_time + eating_dinner_time + playing_games_time + nap_time) = 40 :=
by sorry

end Aren_listening_time_l139_139110


namespace find_n_l139_139043

theorem find_n (n : ℕ) (h_pos : n > 0) (h_ineq : n < Real.sqrt 65 ∧ Real.sqrt 65 < n + 1) : n = 8 := by sorry

end find_n_l139_139043


namespace cheryl_walking_speed_l139_139593

theorem cheryl_walking_speed (H : 12 = 6 * v) : v = 2 := 
by
  -- proof here
  sorry

end cheryl_walking_speed_l139_139593


namespace probability_two_red_balls_l139_139219

open Nat

theorem probability_two_red_balls (total_balls red_balls blue_balls green_balls balls_picked : Nat) 
  (total_eq : total_balls = red_balls + blue_balls + green_balls) 
  (red_eq : red_balls = 7) 
  (blue_eq : blue_balls = 5) 
  (green_eq : green_balls = 4) 
  (picked_eq : balls_picked = 2) :
  (choose red_balls balls_picked) / (choose total_balls balls_picked) = 7 / 40 :=
by
  sorry

end probability_two_red_balls_l139_139219


namespace harry_terry_difference_l139_139627

-- Define Harry's answer
def H : ℤ := 8 - (2 + 5)

-- Define Terry's answer
def T : ℤ := 8 - 2 + 5

-- State the theorem to prove H - T = -10
theorem harry_terry_difference : H - T = -10 := by
  sorry

end harry_terry_difference_l139_139627


namespace polynomial_roots_condition_l139_139306

open Real

def f (x a b : ℝ) : ℝ := x^2 + a * x + b

theorem polynomial_roots_condition (a b : ℤ) (h1 : ∀ x ≠ 0, f (x + x⁻¹) a b = f x a b + f x⁻¹ a b) (h2 : ∃ p q : ℤ, f p a b = 0 ∧ f q a b = 0) : a^2 + b^2 = 13 := by
  sorry

end polynomial_roots_condition_l139_139306


namespace packs_of_snacks_l139_139786

theorem packs_of_snacks (kyle_bike_hours : ℝ) (pack_cost : ℝ) (ryan_budget : ℝ) :
  kyle_bike_hours = 2 →
  10 * (2 * kyle_bike_hours) = pack_cost →
  ryan_budget = 2000 →
  ryan_budget / pack_cost = 50 :=
by 
  sorry

end packs_of_snacks_l139_139786


namespace additional_girls_needed_l139_139334

theorem additional_girls_needed (initial_girls initial_boys additional_girls : ℕ)
  (h_initial_girls : initial_girls = 2)
  (h_initial_boys : initial_boys = 6)
  (h_fraction_goal : (initial_girls + additional_girls) = (5 * (initial_girls + initial_boys + additional_girls)) / 8) :
  additional_girls = 8 :=
by
  -- A placeholder for the proof
  sorry

end additional_girls_needed_l139_139334


namespace reunion_handshakes_l139_139083

-- Condition: Number of boys in total
def total_boys : ℕ := 12

-- Condition: Number of left-handed boys
def left_handed_boys : ℕ := 4

-- Condition: Number of right-handed (not exclusively left-handed) boys
def right_handed_boys : ℕ := total_boys - left_handed_boys

-- Function to calculate combinations n choose 2 (number of handshakes in a group)
def combinations (n : ℕ) : ℕ := n * (n - 1) / 2

-- Condition: Number of handshakes among left-handed boys
def handshakes_left (n : ℕ) : ℕ := combinations left_handed_boys

-- Condition: Number of handshakes among right-handed boys
def handshakes_right (n : ℕ) : ℕ := combinations right_handed_boys

-- Problem statement: total number of handshakes
def total_handshakes (total_boys left_handed_boys right_handed_boys : ℕ) : ℕ :=
  handshakes_left left_handed_boys + handshakes_right right_handed_boys

theorem reunion_handshakes : total_handshakes total_boys left_handed_boys right_handed_boys = 34 :=
by sorry

end reunion_handshakes_l139_139083


namespace women_in_the_minority_l139_139787

theorem women_in_the_minority (total_employees : ℕ) (female_employees : ℕ) (h : female_employees < total_employees * 20 / 100) : 
  (female_employees < total_employees / 2) :=
by
  sorry

end women_in_the_minority_l139_139787


namespace sum_of_squares_500_l139_139705

theorem sum_of_squares_500 : (Finset.range 500).sum (λ x => (x + 1) ^ 2) = 41841791750 := by
  sorry

end sum_of_squares_500_l139_139705


namespace geom_seq_general_term_arith_seq_sum_l139_139427

theorem geom_seq_general_term (q : ℕ → ℕ) (a_1 a_2 a_3 : ℕ) (h1 : a_1 = 2)
  (h2 : (a_1 + a_3) / 2 = a_2 + 1) (h3 : a_2 = q 2) (h4 : a_3 = q 3)
  (g : ℕ → ℕ) (Sn : ℕ → ℕ) (gen_term : ∀ n, q n = 2^n) (sum_term : ∀ n, Sn n = 2^(n+1) - 2) :
  q n = g n :=
sorry

theorem arith_seq_sum (a_1 a_2 a_4 : ℕ) (b : ℕ → ℕ) (Tn : ℕ → ℕ) (h1 : a_1 = 2)
  (h2 : a_2 = 4) (h3 : a_4 = 16) (h4 : b 2 = a_1) (h5 : b 8 = a_2 + a_4)
  (gen_term : ∀ n, b n = 1 + 3 * (n - 1)) (sum_term : ∀ n, Tn n = (3 * n^2 - n) / 2) :
  Tn n = (3 * n^2 - 1) / 2 :=
sorry

end geom_seq_general_term_arith_seq_sum_l139_139427


namespace required_pumps_l139_139229

-- Define the conditions in Lean
variables (x a b n : ℝ)

-- Condition 1: x + 40a = 80b
def condition1 : Prop := x + 40 * a = 2 * 40 * b

-- Condition 2: x + 16a = 64b
def condition2 : Prop := x + 16 * a = 4 * 16 * b

-- Main theorem: Given the conditions, prove that n >= 6 satisfies the remaining requirement
theorem required_pumps (h1 : condition1 x a b) (h2 : condition2 x a b) : n >= 6 :=
by
  sorry

end required_pumps_l139_139229


namespace min_value_x_plus_4y_l139_139612

theorem min_value_x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
(h_cond : (1 / x) + (1 / (2 * y)) = 1) : x + 4 * y = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_x_plus_4y_l139_139612


namespace factorize_cubic_l139_139728

theorem factorize_cubic (a : ℝ) : a^3 - 16 * a = a * (a + 4) * (a - 4) :=
sorry

end factorize_cubic_l139_139728


namespace unique_valid_configuration_l139_139995

-- Define the conditions: a rectangular array of chairs organized in rows and columns such that
-- each row contains the same number of chairs as every other row, each column contains the
-- same number of chairs as every other column, with at least two chairs in every row and column.
def valid_array_configuration (rows cols : ℕ) : Prop :=
  2 ≤ rows ∧ 2 ≤ cols ∧ rows * cols = 49

-- The theorem statement: determine how many valid arrays are possible given the conditions.
theorem unique_valid_configuration : ∃! (rows cols : ℕ), valid_array_configuration rows cols :=
sorry

end unique_valid_configuration_l139_139995


namespace opposite_number_in_circle_l139_139936

theorem opposite_number_in_circle (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 200) (h3 : ∀ k, 1 ≤ k ∧ k ≤ 200 → ∃ m, (m = (k + 100) % 200) ∧ (m ≠ k) ∧ (n + 100) % 200 < k):
  ∃ m : ℕ, m = 114 ∧ (113 + 100) % 200 = m :=
by
  sorry

end opposite_number_in_circle_l139_139936


namespace sum_of_primes_eq_24_l139_139273

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

variable (a b c : ℕ)

theorem sum_of_primes_eq_24 (h1 : is_prime a) (h2 : is_prime b) (h3 : is_prime c)
    (h4 : a * b + b * c = 119) : a + b + c = 24 :=
sorry

end sum_of_primes_eq_24_l139_139273


namespace gcd_factorial_l139_139208

theorem gcd_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  sorry

end gcd_factorial_l139_139208


namespace production_equation_l139_139098

-- Definitions based on the problem conditions
def original_production_rate (x : ℕ) := x
def additional_parts_per_day := 4
def original_days := 20
def actual_days := 15
def extra_parts := 10

-- Prove the equation
theorem production_equation (x : ℕ) :
  original_days * original_production_rate x = actual_days * (original_production_rate x + additional_parts_per_day) - extra_parts :=
by
  simp [original_production_rate, additional_parts_per_day, original_days, actual_days, extra_parts]
  sorry

end production_equation_l139_139098


namespace rhombus_perimeter_and_radius_l139_139960

-- Define the rhombus with given diagonals
structure Rhombus where
  d1 : ℝ -- diagonal 1
  d2 : ℝ -- diagonal 2
  h : d1 = 20 ∧ d2 = 16

-- Define the proof problem
theorem rhombus_perimeter_and_radius (r : Rhombus) : 
  let side_length := Real.sqrt ((r.d1 / 2) ^ 2 + (r.d2 / 2) ^ 2)
  let perimeter := 4 * side_length
  let radius := r.d1 / 2
  perimeter = 16 * Real.sqrt 41 ∧ radius = 10 :=
by
  sorry

end rhombus_perimeter_and_radius_l139_139960


namespace parabola_through_point_l139_139191

theorem parabola_through_point (x y : ℝ) (hx : x = 2) (hy : y = 4) : 
  (∃ a : ℝ, y^2 = a * x ∧ a = 8) ∨ (∃ b : ℝ, x^2 = b * y ∧ b = 1) :=
sorry

end parabola_through_point_l139_139191


namespace frequency_of_8th_group_l139_139601

theorem frequency_of_8th_group :
  let sample_size := 100
  let freq1 := 15
  let freq2 := 17
  let freq3 := 11
  let freq4 := 13
  let freq_5_to_7 := 0.32 * sample_size
  let total_freq_1_to_4 := freq1 + freq2 + freq3 + freq4
  let remaining_freq := sample_size - total_freq_1_to_4
  let freq8 := remaining_freq - freq_5_to_7
  (freq8 / sample_size = 0.12) :=
by
  sorry

end frequency_of_8th_group_l139_139601


namespace dunkers_lineup_count_l139_139173

theorem dunkers_lineup_count (players : Finset ℕ) (h_players : players.card = 15) (alice zen : ℕ) 
  (h_alice : alice ∈ players) (h_zen : zen ∈ players) (h_distinct : alice ≠ zen) :
  (∃ (S : Finset (Finset ℕ)), S.card = 2717 ∧ ∀ s ∈ S, s.card = 5 ∧ ¬ (alice ∈ s ∧ zen ∈ s)) :=
by
  sorry

end dunkers_lineup_count_l139_139173


namespace combined_ratio_is_1_l139_139542

-- Conditions
variables (V1 V2 M1 W1 M2 W2 : ℝ)
variables (x : ℝ)
variables (ratio_volumes ratio_milk_water_v1 ratio_milk_water_v2 : ℝ)

-- Given conditions as hypotheses
-- Condition: V1 / V2 = 3 / 5
-- Hypothesis 1: The volume ratio of the first and second vessels
def volume_ratio : Prop :=
  V1 / V2 = 3 / 5

-- Condition: M1 / W1 = 1 / 2 in first vessel
-- Hypothesis 2: The milk to water ratio in the first vessel
def milk_water_ratio_v1 : Prop :=
  M1 / W1 = 1 / 2

-- Condition: M2 / W2 = 3 / 2 in the second vessel
-- Hypothesis 3: The milk to water ratio in the second vessel
def milk_water_ratio_v2 : Prop :=
  M2 / W2 = 3 / 2

-- Definition: Total volumes of milk and water in the larger vessel
def total_milk_water_ratio : Prop :=
  (M1 + M2) / (W1 + W2) = 1 / 1

-- Main theorem: Given the ratios, the ratio of milk to water in the larger vessel is 1:1
theorem combined_ratio_is_1 :
  (volume_ratio V1 V2) →
  (milk_water_ratio_v1 M1 W1) →
  (milk_water_ratio_v2 M2 W2) →
  total_milk_water_ratio M1 W1 M2 W2 :=
by
  -- Proof omitted
  sorry

end combined_ratio_is_1_l139_139542


namespace angle_C_eq_pi_div_3_side_c_eq_7_l139_139921

theorem angle_C_eq_pi_div_3 
  (A B C : ℝ) (a b c : ℝ)
  (h1 : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C) :
  C = Real.pi / 3 :=
sorry

theorem side_c_eq_7 
  (a b c : ℝ) 
  (h1 : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C)
  (h1a : a = 5) 
  (h1b : b = 8) 
  (h2 : C = Real.pi / 3) :
  c = 7 :=
sorry

end angle_C_eq_pi_div_3_side_c_eq_7_l139_139921


namespace strictly_decreasing_interval_l139_139259

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

theorem strictly_decreasing_interval :
  ∀ x y : ℝ, (0 < x ∧ x < 1) ∧ (0 < y ∧ y < 1) ∧ y < x → f y < f x :=
by
  sorry

end strictly_decreasing_interval_l139_139259


namespace pyramid_height_is_correct_l139_139232

noncomputable def pyramid_height (perimeter : ℝ) (apex_distance : ℝ) : ℝ :=
  let side_length := perimeter / 4
  let half_diagonal := side_length * Real.sqrt 2 / 2
  Real.sqrt (apex_distance ^ 2 - half_diagonal ^ 2)

theorem pyramid_height_is_correct :
  pyramid_height 40 15 = 5 * Real.sqrt 7 :=
by
  sorry

end pyramid_height_is_correct_l139_139232


namespace solution_set_of_inequality_l139_139741

theorem solution_set_of_inequality (a : ℝ) :
  ¬ (∀ x : ℝ, ¬ (a * (x - a) * (a * x + a) ≥ 0)) ∧
  ¬ (∀ x : ℝ, (a - x ≤ 0 ∧ x - (-1) ≤ 0 → a * (x - a) * (a * x + a) ≥ 0)) :=
by
  sorry

end solution_set_of_inequality_l139_139741


namespace custom_op_evaluation_l139_139630

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_evaluation : custom_op 6 5 - custom_op 5 6 = -4 := by
  sorry

end custom_op_evaluation_l139_139630


namespace apples_total_l139_139532

theorem apples_total (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apples_total_l139_139532


namespace probability_of_both_chinese_books_l139_139335

def total_books := 5
def chinese_books := 3
def math_books := 2

theorem probability_of_both_chinese_books (select_books : ℕ) 
  (total_choices : ℕ) (favorable_choices : ℕ) :
  select_books = 2 →
  total_choices = (Nat.choose total_books select_books) →
  favorable_choices = (Nat.choose chinese_books select_books) →
  (favorable_choices : ℚ) / (total_choices : ℚ) = 3 / 10 := by
  intros h1 h2 h3
  sorry

end probability_of_both_chinese_books_l139_139335


namespace base_length_of_triangle_l139_139669

theorem base_length_of_triangle (height area : ℕ) (h1 : height = 8) (h2 : area = 24) : 
  ∃ base : ℕ, (1/2 : ℚ) * base * height = area ∧ base = 6 := by
  sorry

end base_length_of_triangle_l139_139669


namespace opposite_number_on_circle_l139_139940

/-- Given numbers 1 through 200 are arranged in a circle with equal distances, this theorem 
    proves that the number opposite to 113 in such an arrangement is 114. -/
theorem opposite_number_on_circle : 
  ∀ (n : ℕ), n ∈ finset.range 1 201 → ((113 + 100) % 200 = 114) :=
by
  intro n hn,
  have h : 113 + 100 = 213 := rfl,
  have h_mod : 213 % 200 = 13 := rfl,
  sorry

end opposite_number_on_circle_l139_139940


namespace check_error_difference_l139_139990

-- Let us define x and y as two-digit natural numbers
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem check_error_difference
    (x y : ℕ)
    (hx : isTwoDigit x)
    (hy : isTwoDigit y)
    (hxy : x > y)
    (h_difference : (100 * y + x) - (100 * x + y) = 2187)
    : x - y = 22 :=
by
  sorry

end check_error_difference_l139_139990


namespace apple_count_l139_139489

-- Define constants for the problem
constant bags: ℕ → Prop
constant total_apples: ℕ
constant apples_in_bag1: ℕ := 12
constant apples_in_bag2: ℕ := 6

-- Define the conditions
axiom condition1 (n: ℕ) : bags n → (n = apples_in_bag1 ∨ n = apples_in_bag2)
axiom condition2 : 70 ≤ total_apples ∧ total_apples ≤ 80
axiom condition3 : total_apples % 6 = 0

-- State the proof problem
theorem apple_count :
  total_apples = 72 ∨ total_apples = 78 :=
sorry

end apple_count_l139_139489


namespace amelia_wins_l139_139240

noncomputable def amelia_wins_probability : ℚ := 21609 / 64328

theorem amelia_wins (h_am_heads : ℚ) (h_bl_heads : ℚ) (game_starts : Prop) (game_alternates : Prop) (win_condition : Prop) :
  h_am_heads = 3/7 ∧ h_bl_heads = 1/3 ∧ game_starts ∧ game_alternates ∧ win_condition →
  amelia_wins_probability = 21609 / 64328 :=
sorry

end amelia_wins_l139_139240


namespace radius_of_circle_eq_l139_139129

-- Define the given quadratic equation representing the circle
noncomputable def circle_eq (x y : ℝ) : ℝ :=
  16 * x^2 - 32 * x + 16 * y^2 - 48 * y + 68

-- State that the radius of the circle given by the equation is 1
theorem radius_of_circle_eq : ∃ r, (∀ x y, circle_eq x y = 0 ↔ (x - 1)^2 + (y - 1.5)^2 = r^2) ∧ r = 1 :=
by 
  use 1
  sorry

end radius_of_circle_eq_l139_139129


namespace largest_reservoir_is_D_l139_139478

variables (a : ℝ) 
def final_amount_A : ℝ := a * (1 + 0.1) * (1 - 0.05)
def final_amount_B : ℝ := a * (1 + 0.09) * (1 - 0.04)
def final_amount_C : ℝ := a * (1 + 0.08) * (1 - 0.03)
def final_amount_D : ℝ := a * (1 + 0.07) * (1 - 0.02)

theorem largest_reservoir_is_D
  (hA : final_amount_A a = a * 1.045)
  (hB : final_amount_B a = a * 1.0464)
  (hC : final_amount_C a = a * 1.0476)
  (hD : final_amount_D a = a * 1.0486) :
  final_amount_D a > final_amount_A a ∧ 
  final_amount_D a > final_amount_B a ∧ 
  final_amount_D a > final_amount_C a :=
by sorry

end largest_reservoir_is_D_l139_139478


namespace coeff_x7_expansion_l139_139545

theorem coeff_x7_expansion : 
  let expansion := (∑ k in Finset.range 11, (Nat.choose 10 k) * x^(10 - k) * (-2)^k)
  ∃ coeff : ℤ, 
  (coeff * x^7 ∈ expansion) ∧ coeff = -960 :=
begin
  let expansion := (∑ k in Finset.range 11, (Nat.choose 10 k) * x^(10 - k) * (-2)^k),
  use -960,
  split,
  { sorry, },
  { reflexivity, }
end

end coeff_x7_expansion_l139_139545


namespace moles_of_HCl_combined_l139_139260

/-- Prove the number of moles of Hydrochloric acid combined is 1, given that 
1 mole of Sodium hydroxide and some moles of Hydrochloric acid react to produce 
1 mole of Water, based on the balanced chemical equation: NaOH + HCl → NaCl + H2O -/
theorem moles_of_HCl_combined (moles_NaOH : ℕ) (moles_HCl : ℕ) (moles_H2O : ℕ)
  (h1 : moles_NaOH = 1) (h2 : moles_H2O = 1) 
  (balanced_eq : moles_NaOH = moles_HCl ∧ moles_HCl = moles_H2O) : 
  moles_HCl = 1 :=
by
  sorry

end moles_of_HCl_combined_l139_139260


namespace problem_statement_l139_139766

variable (a b : ℝ) (f : ℝ → ℝ)
variable (h1 : ∀ x > 0, f x = Real.log x / Real.log 3)
variable (h2 : b = 9 * a)

theorem problem_statement : f a - f b = -2 := by
  sorry

end problem_statement_l139_139766


namespace jose_land_division_l139_139305

/-- Let the total land Jose bought be 20000 square meters. Let Jose divide this land equally among himself and his four siblings. Prove that the land Jose will have after dividing it is 4000 square meters. -/
theorem jose_land_division : 
  let total_land := 20000
  let numberOfPeople := 5
  total_land / numberOfPeople = 4000 := by
sorry

end jose_land_division_l139_139305


namespace termite_ridden_fraction_l139_139164

theorem termite_ridden_fraction:
  ∀ T: ℝ, (3 / 4) * T = 1 / 4 → T = 1 / 3 :=
by
  intro T
  intro h
  sorry

end termite_ridden_fraction_l139_139164


namespace find_y_coordinate_of_Q_l139_139657

noncomputable def y_coordinate_of_Q 
  (P R T S : ℝ × ℝ) (Q : ℝ × ℝ) (areaPentagon areaSquare : ℝ) : Prop :=
  P = (0, 0) ∧ 
  R = (0, 5) ∧ 
  T = (6, 0) ∧ 
  S = (6, 5) ∧ 
  Q.fst = 3 ∧ 
  areaSquare = 25 ∧ 
  areaPentagon = 50 ∧ 
  (1 / 2) * 6 * (Q.snd - 5) + areaSquare = areaPentagon

theorem find_y_coordinate_of_Q : 
  ∃ y_Q : ℝ, y_coordinate_of_Q (0, 0) (0, 5) (6, 0) (6, 5) (3, y_Q) 50 25 ∧ y_Q = 40 / 3 :=
sorry

end find_y_coordinate_of_Q_l139_139657


namespace average_sweater_less_by_21_after_discount_l139_139296

theorem average_sweater_less_by_21_after_discount
  (shirt_count sweater_count jeans_count : ℕ)
  (total_shirt_price total_sweater_price total_jeans_price : ℕ)
  (shirt_discount sweater_discount jeans_discount : ℕ)
  (shirt_avg_before_discount sweater_avg_before_discount jeans_avg_before_discount 
   shirt_avg_after_discount sweater_avg_after_discount jeans_avg_after_discount : ℕ) :
  shirt_count = 20 →
  sweater_count = 45 →
  jeans_count = 30 →
  total_shirt_price = 360 →
  total_sweater_price = 900 →
  total_jeans_price = 1200 →
  shirt_discount = 2 →
  sweater_discount = 4 →
  jeans_discount = 3 →
  shirt_avg_before_discount = total_shirt_price / shirt_count →
  sweater_avg_before_discount = total_sweater_price / sweater_count →
  jeans_avg_before_discount = total_jeans_price / jeans_count →
  shirt_avg_after_discount = shirt_avg_before_discount - shirt_discount →
  sweater_avg_after_discount = sweater_avg_before_discount - sweater_discount →
  jeans_avg_after_discount = jeans_avg_before_discount - jeans_discount →
  sweater_avg_after_discount = shirt_avg_after_discount →
  jeans_avg_after_discount - sweater_avg_after_discount = 21 :=
by
  intros
  sorry

end average_sweater_less_by_21_after_discount_l139_139296


namespace disjoint_subset_remainder_l139_139040

open Finset

noncomputable def S : Finset ℕ := { n | n ∈ range 1 13 }.toFinset

theorem disjoint_subset_remainder :
  let n := (3:ℕ)^12 - 2 * (2:ℕ)^12 + 1 in
  n / 2 % 500 = 125 :=
by
  let n := (3:ℕ)^12 - 2 * (2:ℕ)^12 + 1
  have h : n / 2 % 500 = 125 := by sorry
  exact h

end disjoint_subset_remainder_l139_139040


namespace valid_number_of_apples_l139_139512

theorem valid_number_of_apples (n : ℕ) : 
  (∃ k m : ℕ, n = 12 * k + 6 * m) ∧ (70 ≤ n ∧ n ≤ 80) ↔ (n = 72 ∨ n = 78) :=
by
  sorry

end valid_number_of_apples_l139_139512


namespace line_intersects_circle_two_points_find_line_l139_139387

-- Definitions for circle C and line l
def circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5
def line (m x y : ℝ) : Prop := mx - y + 1 - m = 0

-- Part (1): Prove that for any m ∈ ℝ, line l intersects circle C at two points A, B
theorem line_intersects_circle_two_points (m : ℝ) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), circle x₁ y₁ ∧ circle x₂ y₂ ∧ line m x₁ y₁ ∧ line m x₂ y₂ ∧ (x₁, y₁) ≠ (x₂, y₂) :=
by 
  sorry

-- Part (2): With fixed point P(1,1) dividing chord AB such that |AP| = 1/2|PB|, find the equation of l
theorem find_line (P : ℝ × ℝ) (hP : P = (1, 1)) :
  ∀ (A B : ℝ × ℝ), (fst A) ≠ (fst B) → 
  circle (fst A) (snd A) → 
  circle (fst B) (snd B) → 
  line (fst P) (fst A) (snd A) →
  line (fst P) (fst B) (snd B) →
  |fst A - fst P| = abs (1/2 * (fst B - fst P)) → 
  line (fst P) (fst P) 1 ∨
  line (fst P) (fst P + snd P - 2) 1 :=
by 
  sorry

end line_intersects_circle_two_points_find_line_l139_139387


namespace base_subtraction_l139_139246

-- Define the base 8 number 765432_8 and its conversion to base 10
def base8Number : ℕ := 7 * (8^5) + 6 * (8^4) + 5 * (8^3) + 4 * (8^2) + 3 * (8^1) + 2 * (8^0)

-- Define the base 9 number 543210_9 and its conversion to base 10
def base9Number : ℕ := 5 * (9^5) + 4 * (9^4) + 3 * (9^3) + 2 * (9^2) + 1 * (9^1) + 0 * (9^0)

-- Lean 4 statement for the proof problem
theorem base_subtraction : (base8Number : ℤ) - (base9Number : ℤ) = -67053 := by
    sorry

end base_subtraction_l139_139246


namespace spacesMovedBeforeSetback_l139_139061

-- Let's define the conditions as local constants
def totalSpaces : ℕ := 48
def firstTurnMove : ℕ := 8
def thirdTurnMove : ℕ := 6
def remainingSpacesToWin : ℕ := 37
def setback : ℕ := 5

theorem spacesMovedBeforeSetback (x : ℕ) : 
  (firstTurnMove + thirdTurnMove) + x - setback + remainingSpacesToWin = totalSpaces →
  x = 28 := by
  sorry

end spacesMovedBeforeSetback_l139_139061


namespace remainder_when_divided_by_9_l139_139344

noncomputable def base12_to_dec (x : ℕ) : ℕ :=
  (1 * 12^3) + (5 * 12^2) + (3 * 12) + 4
  
theorem remainder_when_divided_by_9 : base12_to_dec (1534) % 9 = 2 := by
  sorry

end remainder_when_divided_by_9_l139_139344


namespace ravi_overall_profit_l139_139056

-- Define the purchase prices
def refrigerator_purchase_price := 15000
def mobile_phone_purchase_price := 8000

-- Define the percentages
def refrigerator_loss_percent := 2
def mobile_phone_profit_percent := 10

-- Define the calculations for selling prices
def refrigerator_loss_amount := (refrigerator_loss_percent / 100) * refrigerator_purchase_price
def refrigerator_selling_price := refrigerator_purchase_price - refrigerator_loss_amount

def mobile_phone_profit_amount := (mobile_phone_profit_percent / 100) * mobile_phone_purchase_price
def mobile_phone_selling_price := mobile_phone_purchase_price + mobile_phone_profit_amount

-- Define the total purchase and selling prices
def total_purchase_price := refrigerator_purchase_price + mobile_phone_purchase_price
def total_selling_price := refrigerator_selling_price + mobile_phone_selling_price

-- Define the overall profit calculation
def overall_profit := total_selling_price - total_purchase_price

-- Statement of the theorem
theorem ravi_overall_profit :
  overall_profit = 500 := by
  sorry

end ravi_overall_profit_l139_139056


namespace opposite_number_113_is_114_l139_139947

-- Definitions based on conditions in the problem
def numbers_on_circle (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 200

def are_equal_distances (numbers : Finset ℕ) : Prop :=
  ∀ x, x ∈ numbers → ∃ y, y ∈ numbers ∧ dist x y = k  -- Placeholder, replace dist and k with appropriate logic

def equality_of_smaller_numbers (numbers : Finset ℕ) (x : ℕ) : Prop :=
  (number of numbers smaller than x in 99 clockwise numbers = 
      number of numbers smaller than x in 99 counterclockwise numbers)

-- Proof statement showing that number opposite to 113 is 114
theorem opposite_number_113_is_114 : ∀ numbers : Finset ℕ,
  numbers_on_circle ∧ are_equal_distances numbers ∧ equality_of_smaller_numbers numbers 113 →
  ∃ n, n = 114 :=
begin
  sorry
end

end opposite_number_113_is_114_l139_139947


namespace add_to_any_integer_l139_139838

theorem add_to_any_integer (y : ℤ) : (∀ x : ℤ, y + x = x) → y = 0 :=
  by
  sorry

end add_to_any_integer_l139_139838


namespace shanghai_world_expo_l139_139239

theorem shanghai_world_expo (n : ℕ) (total_cost : ℕ) 
  (H1 : total_cost = 4000)
  (H2 : n ≤ 30 → total_cost = n * 120)
  (H3 : n > 30 → total_cost = n * (120 - 2 * (n - 30)) ∧ (120 - 2 * (n - 30)) ≥ 90) :
  n = 40 := 
sorry

end shanghai_world_expo_l139_139239


namespace parallelogram_height_l139_139468

/-- The cost of leveling a field in the form of a parallelogram is Rs. 50 per 10 sq. meter, 
    with the base being 54 m and a certain perpendicular distance from the other side. 
    The total cost is Rs. 6480. What is the perpendicular distance from the other side 
    of the parallelogram? -/
theorem parallelogram_height
  (cost_per_10_sq_meter : ℝ)
  (base_length : ℝ)
  (total_cost : ℝ)
  (height : ℝ)
  (h1 : cost_per_10_sq_meter = 50)
  (h2 : base_length = 54)
  (h3 : total_cost = 6480)
  (area : ℝ)
  (h4 : area = (total_cost / cost_per_10_sq_meter) * 10)
  (h5 : area = base_length * height) :
  height = 24 :=
by { sorry }

end parallelogram_height_l139_139468


namespace trackball_mice_count_l139_139316

theorem trackball_mice_count
  (total_mice : ℕ)
  (wireless_fraction : ℕ)
  (optical_fraction : ℕ)
  (h_total : total_mice = 80)
  (h_wireless : wireless_fraction = total_mice / 2)
  (h_optical : optical_fraction = total_mice / 4) :
  total_mice - (wireless_fraction + optical_fraction) = 20 :=
sorry

end trackball_mice_count_l139_139316


namespace ratio_of_area_of_CDGE_to_ABC_l139_139430

/-- Given a triangle ABC with medians AD and BE meeting at the centroid G,
    and D and E being midpoints of sides BC and AC respectively, 
    prove the ratio of the area of quadrilateral CDGE to the area of triangle ABC is 1/3. -/
theorem ratio_of_area_of_CDGE_to_ABC (A B C D E G : Point) (h1 : IsMedian A D)
  (h2 : IsMedian B E) (h3 : IsCentroid G A D B E) (h4 : Midpoint D B C) (h5 : Midpoint E A C) :
  area (Quadrilateral C D G E) / area (Triangle A B C) = 1 / 3 :=
  sorry

end ratio_of_area_of_CDGE_to_ABC_l139_139430


namespace increased_speed_l139_139359

theorem increased_speed
  (d : ℝ) (s1 s2 : ℝ) (t1 t2 : ℝ) 
  (h1 : d = 2) 
  (h2 : s1 = 2) 
  (h3 : t1 = 1)
  (h4 : t2 = 2 / 3)
  (h5 : s1 * t1 = d)
  (h6 : s2 * t2 = d) :
  s2 - s1 = 1 := 
sorry

end increased_speed_l139_139359


namespace graph_represents_two_intersecting_lines_l139_139724

theorem graph_represents_two_intersecting_lines (x y : ℝ) :
  (x - 1) * (x + y + 2) = (y - 1) * (x + y + 2) → 
  (x + y + 2 = 0 ∨ x = y) ∧ 
  (∃ (x y : ℝ), (x = -1 ∧ y = -1 ∧ x = y ∨ x = -y - 2) ∧ (y = x ∨ y = -x - 2)) :=
by
  sorry

end graph_represents_two_intersecting_lines_l139_139724


namespace find_f_0_abs_l139_139042

noncomputable def f : ℝ → ℝ := sorry -- f is a second-degree polynomial with real coefficients

axiom h1 : ∀ (x : ℝ), x = 1 → |f x| = 9
axiom h2 : ∀ (x : ℝ), x = 2 → |f x| = 9
axiom h3 : ∀ (x : ℝ), x = 3 → |f x| = 9

theorem find_f_0_abs : |f 0| = 9 := sorry

end find_f_0_abs_l139_139042


namespace diameter_increase_l139_139665

theorem diameter_increase (π : ℝ) (D : ℝ) (A A' D' : ℝ)
  (hA : A = (π / 4) * D^2)
  (hA' : A' = 4 * A)
  (hA'_def : A' = (π / 4) * D'^2) :
  D' = 2 * D :=
by
  sorry

end diameter_increase_l139_139665


namespace complex_root_seventh_power_l139_139928

theorem complex_root_seventh_power (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^5 - 1) * (r^6 - 1) = 2 := by
  sorry

end complex_root_seventh_power_l139_139928


namespace find_a_and_solve_inequalities_l139_139633

-- Definitions as per conditions
def inequality1 (a : ℝ) (x : ℝ) : Prop := a*x^2 + 5*x - 2 > 0
def inequality2 (a : ℝ) (x : ℝ) : Prop := a*x^2 - 5*x + a^2 - 1 > 0

-- Statement of the theorem
theorem find_a_and_solve_inequalities :
  ∀ (a : ℝ),
    (∀ x, (1/2 < x ∧ x < 2) ↔ inequality1 a x) →
    a = -2 ∧
    (∀ x, (-1/2 < x ∧ x < 3) ↔ inequality2 (-2) x) :=
by
  intros a h
  sorry

end find_a_and_solve_inequalities_l139_139633


namespace arithmetic_progression_number_of_terms_l139_139023

variable (a d : ℕ)
variable (n : ℕ) (h_n_even : n % 2 = 0)
variable (h_sum_odd : (n / 2) * (2 * a + (n - 2) * d) = 60)
variable (h_sum_even : (n / 2) * (2 * (a + d) + (n - 2) * d) = 80)
variable (h_diff : (n - 1) * d = 16)

theorem arithmetic_progression_number_of_terms : n = 8 :=
by
  sorry

end arithmetic_progression_number_of_terms_l139_139023


namespace opposite_number_113_is_13_l139_139942

theorem opposite_number_113_is_13 :
  let n := 113
  let count := 200
  (n + 100) % count = 13 :=
by
  let n := 113
  let count := 200
  have h : (n + 100) % count = (113 + 100) % 200 := rfl
  rw h
  have result : (113 + 100) % 200 = 13 := by norm_num
  exact result

end opposite_number_113_is_13_l139_139942


namespace div_pow_eq_l139_139264

theorem div_pow_eq {a : ℝ} (h : a ≠ 0) : a^3 / a^2 = a :=
sorry

end div_pow_eq_l139_139264


namespace problem_a_even_triangles_problem_b_even_triangles_l139_139222

-- Definition for problem (a)
def square_divided_by_triangles_3_4_even (a : ℕ) : Prop :=
  let area_triangle := 3 * 4 / 2
  let area_square := a * a
  let k := area_square / area_triangle
  (k % 2 = 0)

-- Definition for problem (b)
def rectangle_divided_by_triangles_1_2_even (l w : ℕ) : Prop :=
  let area_triangle := 1 * 2 / 2
  let area_rectangle := l * w
  let k := area_rectangle / area_triangle
  (k % 2 = 0)

-- Theorem for problem (a)
theorem problem_a_even_triangles {a : ℕ} (h : a > 0) :
  square_divided_by_triangles_3_4_even a :=
sorry

-- Theorem for problem (b)
theorem problem_b_even_triangles {l w : ℕ} (hl : l > 0) (hw : w > 0) :
  rectangle_divided_by_triangles_1_2_even l w :=
sorry

end problem_a_even_triangles_problem_b_even_triangles_l139_139222


namespace fourth_metal_mass_approx_l139_139708

noncomputable def mass_of_fourth_metal 
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 = 1.5 * x2)
  (h2 : x3 = 4 / 3 * x2)
  (h3 : x4 = 6 / 5 * x3)
  (h4 : x1 + x2 + x3 + x4 = 25) : ℝ :=
  x4

theorem fourth_metal_mass_approx 
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 = 1.5 * x2)
  (h2 : x3 = 4 / 3 * x2)
  (h3 : x4 = 6 / 5 * x3)
  (h4 : x1 + x2 + x3 + x4 = 25) : 
  abs (mass_of_fourth_metal x1 x2 x3 x4 h1 h2 h3 h4 - 7.36) < 0.01 :=
by
  sorry

end fourth_metal_mass_approx_l139_139708


namespace triangle_base_length_l139_139668

-- Given conditions
def area_triangle (base height : ℕ) : ℕ := (1 / 2 : ℚ) * base * height

-- Problem statement
theorem triangle_base_length (A h : ℕ) (A_eq : A = 24) (h_eq : h = 8) :
  ∃ b : ℕ, area_triangle b h = A ∧ b = 6 := 
by
  sorry

end triangle_base_length_l139_139668


namespace op_4_6_l139_139009

-- Define the operation @ in Lean
def op (a b : ℕ) : ℤ := 2 * (a : ℤ)^2 - 2 * (b : ℤ)^2

-- State the theorem to prove
theorem op_4_6 : op 4 6 = -40 :=
by sorry

end op_4_6_l139_139009


namespace polynomial_coeff_sum_l139_139904

noncomputable def polynomial_expansion (x : ℝ) :=
  (2 * x + 3) * (4 * x^3 - 2 * x^2 + x - 7)

theorem polynomial_coeff_sum :
  let A := 8
  let B := 8
  let C := -4
  let D := -11
  let E := -21
  A + B + C + D + E = -20 :=
by
  -- The following proof steps are skipped
  sorry

end polynomial_coeff_sum_l139_139904


namespace find_x_l139_139004

theorem find_x (x : ℝ) (a b c : ℝ × ℝ)
  (ha : a = (x, 1))
  (hb : b = (2, x))
  (hc : c = (1, -2))
  (h_perpendicular : (a.1 * (b.1 + c.1) + a.2 * (b.2 + c.2)) = 0) :
  x = 1 / 2 :=
sorry

end find_x_l139_139004


namespace playground_area_l139_139187

noncomputable def length (w : ℝ) := 2 * w + 30
noncomputable def perimeter (l w : ℝ) := 2 * (l + w)
noncomputable def area (l w : ℝ) := l * w

theorem playground_area :
  ∃ (w l : ℝ), length w = l ∧ perimeter l w = 700 ∧ area l w = 25955.56 :=
by {
  sorry
}

end playground_area_l139_139187


namespace negate_existential_l139_139070

theorem negate_existential (p : Prop) : (¬(∃ x : ℝ, x^2 - 2 * x + 2 ≤ 0)) ↔ ∀ x : ℝ, x^2 - 2 * x + 2 > 0 :=
by sorry

end negate_existential_l139_139070


namespace valid_number_of_apples_l139_139509

theorem valid_number_of_apples (n : ℕ) : 
  (∃ k m : ℕ, n = 12 * k + 6 * m) ∧ (70 ≤ n ∧ n ≤ 80) ↔ (n = 72 ∨ n = 78) :=
by
  sorry

end valid_number_of_apples_l139_139509


namespace avg_age_diff_l139_139330

noncomputable def avg_age_team : ℕ := 28
noncomputable def num_players : ℕ := 11
noncomputable def wicket_keeper_age : ℕ := avg_age_team + 3
noncomputable def total_age_team : ℕ := avg_age_team * num_players
noncomputable def age_captain : ℕ := avg_age_team

noncomputable def total_age_remaining_players : ℕ := total_age_team - age_captain - wicket_keeper_age
noncomputable def num_remaining_players : ℕ := num_players - 2
noncomputable def avg_age_remaining_players : ℕ := total_age_remaining_players / num_remaining_players

theorem avg_age_diff :
  avg_age_team - avg_age_remaining_players = 3 :=
by
  sorry

end avg_age_diff_l139_139330


namespace range_of_a_l139_139055

noncomputable def prop_p (a x : ℝ) : Prop := 3 * a < x ∧ x < a

noncomputable def prop_q (x : ℝ) : Prop := x^2 - x - 6 < 0

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, ¬ prop_p a x) ∧ ¬ (∃ x : ℝ, ¬ prop_p a x) → ¬ (∃ x : ℝ, ¬ prop_q x) → -2/3 ≤ a ∧ a < 0 := 
by
  sorry

end range_of_a_l139_139055


namespace max_value_x3y2z_l139_139652

theorem max_value_x3y2z
  (x y z : ℝ)
  (hx_pos : 0 < x)
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (h_total : x + 2 * y + 3 * z = 1)
  : x^3 * y^2 * z ≤ 2048 / 11^6 := 
by
  sorry

end max_value_x3y2z_l139_139652


namespace smallest_positive_integer_satisfying_conditions_l139_139972

theorem smallest_positive_integer_satisfying_conditions :
  ∃ (x : ℕ),
    x % 4 = 1 ∧
    x % 5 = 2 ∧
    x % 7 = 3 ∧
    ∀ y : ℕ, (y % 4 = 1 ∧ y % 5 = 2 ∧ y % 7 = 3) → y ≥ x ∧ x = 93 :=
by
  sorry

end smallest_positive_integer_satisfying_conditions_l139_139972


namespace museum_revenue_l139_139639

theorem museum_revenue (V : ℕ) (H : V = 500)
  (R : ℕ) (H_R : R = 60 * V / 100)
  (C_p : ℕ) (H_C_p : C_p = 40 * R / 100)
  (S_p : ℕ) (H_S_p : S_p = 30 * R / 100)
  (A_p : ℕ) (H_A_p : A_p = 30 * R / 100)
  (C_t S_t A_t : ℕ) (H_C_t : C_t = 4) (H_S_t : S_t = 6) (H_A_t : A_t = 12) :
  C_p * C_t + S_p * S_t + A_p * A_t = 2100 :=
by 
  sorry

end museum_revenue_l139_139639


namespace tangent_line_at_1_intervals_of_monotonicity_and_extrema_l139_139142

open Real

noncomputable def f (x : ℝ) := 6 * log x + (1 / 2) * x^2 - 5 * x

theorem tangent_line_at_1 :
  let f' (x : ℝ) := (6 / x) + x - 5
  (f 1 = -9 / 2) →
  (f' 1 = 2) →
  (∀ x y : ℝ, y + 9 / 2 = 2 * (x - 1) → 4 * x - 2 * y - 13 = 0) := 
by
  sorry

theorem intervals_of_monotonicity_and_extrema :
  let f' (x : ℝ) := (x^2 - 5 * x + 6) / x
  (∀ x, 0 < x ∧ x < 2 → f' x > 0) → 
  (∀ x, 3 < x → f' x > 0) →
  (∀ x, 2 < x ∧ x < 3 → f' x < 0) →
  (f 2 = -8 + 6 * log 2) →
  (f 3 = -21 / 2 + 6 * log 3) :=
by
  sorry

end tangent_line_at_1_intervals_of_monotonicity_and_extrema_l139_139142


namespace ratio_of_ages_l139_139168

theorem ratio_of_ages (Sandy_age : ℕ) (Molly_age : ℕ)
  (h1 : Sandy_age = 56)
  (h2 : Molly_age = Sandy_age + 16) :
  (Sandy_age : ℚ) / Molly_age = 7 / 9 :=
by
  -- Proof goes here
  sorry

end ratio_of_ages_l139_139168


namespace factor_expression_l139_139869

theorem factor_expression (y : ℝ) : 
  5 * y * (y - 2) + 11 * (y - 2) = (y - 2) * (5 * y + 11) :=
by
  sorry

end factor_expression_l139_139869


namespace possible_apple_counts_l139_139505

theorem possible_apple_counts (n : ℕ) (h₁ : 70 ≤ n) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end possible_apple_counts_l139_139505


namespace inequality_proof_l139_139444

theorem inequality_proof (n : ℕ) (a : ℝ) (h₀ : n > 1) (h₁ : 0 < a) (h₂ : a < 1) : 
  1 + a < (1 + a / n) ^ n ∧ (1 + a / n) ^ n < (1 + a / (n + 1)) ^ (n + 1) := 
sorry

end inequality_proof_l139_139444


namespace find_number_l139_139566

theorem find_number (x : ℝ) (h₁ : 0.40 * x = 130 + 190) : x = 800 :=
sorry

end find_number_l139_139566


namespace wire_length_around_square_field_l139_139555

theorem wire_length_around_square_field (area : ℝ) (times : ℕ) (wire_length : ℝ) 
    (h1 : area = 69696) (h2 : times = 15) : wire_length = 15840 :=
by
  sorry

end wire_length_around_square_field_l139_139555


namespace income_remaining_percentage_l139_139849

theorem income_remaining_percentage :
  let initial_income := 100
  let food_percentage := 42
  let education_percentage := 18
  let transportation_percentage := 12
  let house_rent_percentage := 55
  let total_spent := food_percentage + education_percentage + transportation_percentage
  let remaining_after_expenses := initial_income - total_spent
  let house_rent_amount := (house_rent_percentage * remaining_after_expenses) / 100
  let final_remaining_income := remaining_after_expenses - house_rent_amount
  final_remaining_income = 12.6 :=
by
  sorry

end income_remaining_percentage_l139_139849


namespace total_apples_l139_139515

theorem total_apples (n : ℕ) (h1 : ∀ k : ℕ, 6 * k = n ∨ 12 * k = n) (h2 : 70 ≤ n) (h3 : n ≤ 80) : 
  n = 72 ∨ n = 78 := 
sorry

end total_apples_l139_139515


namespace difference_in_speed_l139_139691

theorem difference_in_speed (d : ℕ) (tA tE : ℕ) (vA vE : ℕ) (h1 : d = 300) (h2 : tA = tE - 3) 
    (h3 : vE = 20) (h4 : vE = d / tE) (h5 : vA = d / tA) : vA - vE = 5 := 
    sorry

end difference_in_speed_l139_139691


namespace total_initial_collection_l139_139049

variable (marco strawberries father strawberries_lost : ℕ)
variable (marco : ℕ := 12)
variable (father : ℕ := 16)
variable (strawberries_lost : ℕ := 8)
variable (total_initial_weight : ℕ := marco + father + strawberries_lost)

theorem total_initial_collection : total_initial_weight = 36 :=
by
  sorry

end total_initial_collection_l139_139049


namespace train_speed_correct_l139_139841

def train_length : ℝ := 100
def crossing_time : ℝ := 12
def expected_speed : ℝ := 8.33

theorem train_speed_correct : (train_length / crossing_time) = expected_speed :=
by
  -- Proof goes here
  sorry

end train_speed_correct_l139_139841


namespace arithmetic_sequence_general_term_find_n_given_sum_l139_139885

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h1 : a 10 = 30)
  (h2 : a 15 = 40)
  : ∃ a1 d, (∀ n, a n = a1 + (n - 1) * d) ∧ a 10 = 30 ∧ a 15 = 40 :=
by {
  sorry
}

theorem find_n_given_sum
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (a1 d : ℕ)
  (h_gen : ∀ n, a n = a1 + (n - 1) * d)
  (h_sum : ∀ n, S n = n * a1 + (n * (n - 1) * d) / 2)
  (h_a1 : a1 = 12)
  (h_d : d = 2)
  (h_Sn : S 14 = 210)
  : ∃ n, S n = 210 ∧ n = 14 :=
by {
  sorry
}

end arithmetic_sequence_general_term_find_n_given_sum_l139_139885


namespace ratio_sharks_to_pelicans_l139_139380

-- Define the conditions given in the problem
def original_pelican_count {P : ℕ} (h : (2/3 : ℚ) * P = 20) : Prop :=
  P = 30

-- Define the final ratio we want to prove
def shark_to_pelican_ratio (sharks pelicans : ℕ) : ℚ :=
  sharks / pelicans

theorem ratio_sharks_to_pelicans
  (P : ℕ) (h : (2/3 : ℚ) * P = 20) (number_sharks : ℕ) (number_pelicans : ℕ)
  (H_sharks : number_sharks = 60) (H_pelicans : number_pelicans = P)
  (H_original_pelicans : original_pelican_count h) :
  shark_to_pelican_ratio number_sharks number_pelicans = 2 :=
by
  -- proof skipped
  sorry

end ratio_sharks_to_pelicans_l139_139380


namespace total_problems_completed_l139_139797

variables (p t : ℕ)
variables (hp_pos : 15 < p) (ht_pos : 0 < t)
variables (eq1 : (3 * p - 6) * (t - 3) = p * t)

theorem total_problems_completed : p * t = 120 :=
by sorry

end total_problems_completed_l139_139797


namespace tiffany_max_points_l139_139536

theorem tiffany_max_points : 
  let initial_money := 3
  let cost_per_game := 1
  let points_red_bucket := 2
  let points_green_bucket := 3
  let rings_per_game := 5
  let games_played := 2
  let red_buckets_first_two_games := 4
  let green_buckets_first_two_games := 5
  let remaining_money := initial_money - games_played * cost_per_game
  let remaining_games := remaining_money / cost_per_game
  let points_first_two_games := red_buckets_first_two_games * points_red_bucket + green_buckets_first_two_games * points_green_bucket
  let max_points_third_game := rings_per_game * points_green_bucket
  points_first_two_games + max_points_third_game = 38 := 
by
  sorry

end tiffany_max_points_l139_139536


namespace abs_eq_abs_implies_l139_139217

theorem abs_eq_abs_implies (x : ℝ) (h : |x - 3| = |x + 1|) : x = 1 := 
sorry

end abs_eq_abs_implies_l139_139217


namespace apple_bags_l139_139496

theorem apple_bags (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apple_bags_l139_139496


namespace reverse_addition_unique_l139_139336

theorem reverse_addition_unique (k : ℤ) (h t u : ℕ) (n : ℤ)
  (hk : 100 * h + 10 * t + u = k) 
  (h_k_range : 100 < k ∧ k < 1000)
  (h_reverse_addition : 100 * u + 10 * t + h = k + n)
  (digits_range : 1 ≤ h ∧ h ≤ 9 ∧ 0 ≤ t ∧ t ≤ 9 ∧ 0 ≤ u ∧ u ≤ 9) :
  n = 99 :=
sorry

end reverse_addition_unique_l139_139336


namespace find_integer_roots_l139_139733

open Int Polynomial

def P (x : ℤ) : ℤ := x^3 - 3 * x^2 - 13 * x + 15

theorem find_integer_roots : {x : ℤ | P x = 0} = {-3, 1, 5} := by
  sorry

end find_integer_roots_l139_139733


namespace sum_of_solutions_eq_zero_l139_139837

theorem sum_of_solutions_eq_zero :
  let p := 6
  let q := 150
  (∃ x1 x2 : ℝ, p * x1 = q / x1 ∧ p * x2 = q / x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 0) :=
sorry

end sum_of_solutions_eq_zero_l139_139837


namespace measure_of_angle_C_l139_139026

theorem measure_of_angle_C
  (A B C : ℝ)
  (h1 : 3 * Real.sin A + 4 * Real.cos B = 6)
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1)
  (h3 : A + B + C = Real.pi) :
  C = Real.pi / 6 := 
sorry

end measure_of_angle_C_l139_139026


namespace lily_calculation_l139_139826

theorem lily_calculation (a b c : ℝ) (h1 : a - 2 * b - 3 * c = 2) (h2 : a - 2 * (b - 3 * c) = 14) :
  a - 2 * b = 6 :=
by
  sorry

end lily_calculation_l139_139826


namespace value_of_g_13_l139_139907

def g (n : ℕ) : ℕ := n^2 + 2 * n + 23

theorem value_of_g_13 : g 13 = 218 :=
by 
  sorry

end value_of_g_13_l139_139907


namespace geometric_mean_45_80_l139_139671

theorem geometric_mean_45_80 : ∃ x : ℝ, x^2 = 45 * 80 ∧ (x = 60 ∨ x = -60) := 
by 
  sorry

end geometric_mean_45_80_l139_139671


namespace taxi_ride_cost_l139_139235

-- Define the fixed cost
def fixed_cost : ℝ := 2.00

-- Define the cost per mile
def cost_per_mile : ℝ := 0.30

-- Define the number of miles traveled
def miles_traveled : ℝ := 7.0

-- Define the total cost calculation
def total_cost : ℝ := fixed_cost + (cost_per_mile * miles_traveled)

-- Theorem: Prove the total cost of a 7-mile taxi ride is $4.10
theorem taxi_ride_cost : total_cost = 4.10 := by
  sorry

end taxi_ride_cost_l139_139235


namespace solve_for_a_l139_139287

theorem solve_for_a (a x : ℝ) (h : x^2 + a * x + 4 = (x + 2)^2) : a = 4 :=
by sorry

end solve_for_a_l139_139287


namespace find_h_l139_139019

theorem find_h: 
  ∃ h k, (∀ x, 2 * x ^ 2 + 6 * x + 11 = 2 * (x - h) ^ 2 + k) ∧ h = -3 / 2 :=
by
  sorry

end find_h_l139_139019


namespace phil_has_97_quarters_l139_139803

-- Declare all the conditions as definitions
def initial_amount : ℝ := 40.0
def cost_pizza : ℝ := 2.75
def cost_soda : ℝ := 1.50
def cost_jeans : ℝ := 11.50
def quarter_value : ℝ := 0.25

-- The total cost of the items bought
def total_cost : ℝ := cost_pizza + cost_soda + cost_jeans

-- The remaining amount after purchases
def remaining_amount : ℝ := initial_amount - total_cost

-- The number of quarters in the remaining amount
def quarters_left : ℝ := remaining_amount / quarter_value

theorem phil_has_97_quarters : quarters_left = 97 := 
by 
  have h1 : total_cost = 15.75 := sorry
  have h2 : remaining_amount = 24.25 := sorry
  have h3 : quarters_left = 24.25 / 0.25 := sorry
  have h4 : quarters_left = 97 := sorry
  exact h4

end phil_has_97_quarters_l139_139803


namespace simplify_expr_l139_139661

theorem simplify_expr : 
  (576:ℝ)^(1/4) * (216:ℝ)^(1/2) = 72 := 
by 
  have h1 : 576 = (2^4 * 36 : ℝ) := by norm_num
  have h2 : 36 = (6^2 : ℝ) := by norm_num
  have h3 : 216 = (6^3 : ℝ) := by norm_num
  sorry

end simplify_expr_l139_139661


namespace problem_scores_ordering_l139_139261

variable {J K L R : ℕ}

theorem problem_scores_ordering (h1 : J > K) (h2 : J > L) (h3 : J > R)
                                (h4 : L > min K R) (h5 : R > min K L)
                                (h6 : (J ≠ K) ∧ (J ≠ L) ∧ (J ≠ R) ∧ (K ≠ L) ∧ (K ≠ R) ∧ (L ≠ R)) :
                                K < L ∧ L < R :=
sorry

end problem_scores_ordering_l139_139261


namespace roller_skate_wheels_l139_139799

theorem roller_skate_wheels (number_of_people : ℕ)
  (feet_per_person : ℕ)
  (skates_per_foot : ℕ)
  (wheels_per_skate : ℕ)
  (h_people : number_of_people = 40)
  (h_feet : feet_per_person = 2)
  (h_skates : skates_per_foot = 1)
  (h_wheels : wheels_per_skate = 4)
  : (number_of_people * feet_per_person * skates_per_foot * wheels_per_skate) = 320 := 
by
  sorry

end roller_skate_wheels_l139_139799


namespace apples_total_l139_139534

theorem apples_total (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apples_total_l139_139534


namespace find_angle_A_find_side_a_l139_139136

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

end find_angle_A_find_side_a_l139_139136


namespace volume_of_pyramid_l139_139596

noncomputable def pyramid_volume : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (30, 0)
  let C : ℝ × ℝ := (12, 20)
  let D : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2) -- Midpoint of BC
  let E : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2) -- Midpoint of AC
  let F : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) -- Midpoint of AB
  let height : ℝ := 8.42 -- Vertically above the orthocenter
  let base_area : ℝ := 110 -- Area of the midpoint triangle
  (1 / 3) * base_area * height

theorem volume_of_pyramid : pyramid_volume = 309.07 :=
  by
    sorry

end volume_of_pyramid_l139_139596


namespace students_in_band_l139_139856

theorem students_in_band (total_students : ℕ) (band_percentage : ℚ) (h_total_students : total_students = 840) (h_band_percentage : band_percentage = 0.2) : ∃ band_students : ℕ, band_students = 168 ∧ band_students = band_percentage * total_students := 
sorry

end students_in_band_l139_139856


namespace expression_that_gives_value_8_l139_139016

theorem expression_that_gives_value_8 (a b : ℝ) 
  (h_eq1 : a = 2) 
  (h_eq2 : b = 2) 
  (h_roots : ∀ x, (x - a) * (x - b) = x^2 - 4 * x + 4) : 
  2 * (a + b) = 8 :=
by
  sorry

end expression_that_gives_value_8_l139_139016


namespace no_such_function_l139_139253

noncomputable def no_such_function_exists : Prop :=
  ¬∃ f : ℝ → ℝ, 
    (∀ y : ℝ, ∃ x : ℝ, f x = y) ∧ 
    (∀ x : ℝ, f (f x) = (x - 1) * f x + 2)

-- Here's the theorem statement to be proved
theorem no_such_function : no_such_function_exists :=
sorry

end no_such_function_l139_139253


namespace minimum_n_l139_139268

-- Assume the sequence a_n is defined as part of an arithmetic sequence
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + n * d

-- Define S_n as the sum of the first n terms in the sequence
def sum_arithmetic_sequence (a d n : ℕ) : ℕ := n * a + (n * (n - 1)) / 2 * d

-- Given conditions
def a1 := 2
def d := 1  -- Derived from the condition a1 + a4 = a5

-- Problem Statement
theorem minimum_n (n : ℕ) :
  (sum_arithmetic_sequence a1 d n > 32) ↔ n = 6 :=
sorry

end minimum_n_l139_139268


namespace equal_distances_triangle_incircle_l139_139308

theorem equal_distances_triangle_incircle 
  (A B C E F G R S : Point)
  (h_triangle : Triangle A B C)
  (h_tangency_E : IncircleTangency A C E)
  (h_tangency_F : IncircleTangency B F)
  (h_intersection_G : Intersect G (Line (C, F)) (Line (B, E)))
  (h_parallelogram_R : Parallelogram B C E R)
  (h_parallelogram_S : Parallelogram B C S F) :
  Distance G R = Distance G S :=
sorry

end equal_distances_triangle_incircle_l139_139308


namespace track_meet_total_people_l139_139828

theorem track_meet_total_people (B G : ℕ) (H1 : B = 30)
  (H2 : ∃ G, (3 * G) / 5 + (2 * G) / 5 = G)
  (H3 : ∀ G, 2 * G / 5 = 10) :
  B + G = 55 :=
by
  sorry

end track_meet_total_people_l139_139828


namespace solution1_solution2_solution3_l139_139134

noncomputable def problem1 : Nat :=
  (1) * (2 - 1) * (2 + 1)

theorem solution1 : problem1 = 3 := by
  sorry

noncomputable def problem2 : Nat :=
  (2) * (2 + 1) * (2^2 + 1)

theorem solution2 : problem2 = 15 := by
  sorry

noncomputable def problem3 : Nat :=
  (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) * (2^32 + 1)

theorem solution3 : problem3 = 2^64 - 1 := by
  sorry

end solution1_solution2_solution3_l139_139134


namespace neg_p_sufficient_for_neg_q_l139_139386

def p (x : ℝ) : Prop := |2 * x - 3| > 1
def q (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem neg_p_sufficient_for_neg_q :
  (∀ x, ¬ p x → ¬ q x) ∧ ¬ (∀ x, ¬ q x → ¬ p x) :=
by
  -- Placeholder to indicate skipping the proof
  sorry

end neg_p_sufficient_for_neg_q_l139_139386


namespace kamala_overestimation_l139_139955

theorem kamala_overestimation : 
  let p := 150
  let q := 50
  let k := 2
  let d := 3
  let p_approx := 160
  let q_approx := 45
  let k_approx := 1
  let d_approx := 4
  let true_value := (p / q) - k + d
  let approx_value := (p_approx / q_approx) - k_approx + d_approx
  approx_value > true_value := 
  by 
  -- Skipping the detailed proof steps.
  sorry

end kamala_overestimation_l139_139955


namespace solve_abs_eq_l139_139215

theorem solve_abs_eq (x : ℝ) (h : |x - 3| = |x + 1|) : x = 1 :=
  sorry

end solve_abs_eq_l139_139215


namespace chosen_number_is_reconstructed_l139_139981

theorem chosen_number_is_reconstructed (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 26) :
  ∃ (a0 a1 a2 : ℤ), (a0 = 0 ∨ a0 = 1 ∨ a0 = 2) ∧ 
                     (a1 = 0 ∨ a1 = 1 ∨ a1 = 2) ∧ 
                     (a2 = 0 ∨ a2 = 1 ∨ a2 = 2) ∧ 
                     n = a0 * 3^0 + a1 * 3^1 + a2 * 3^2 ∧ 
                     n = (if a0 = 1 then 1 else 0) + (if a0 = 2 then 2 else 0) +
                         (if a1 = 1 then 3 else 0) + (if a1 = 2 then 6 else 0) +
                         (if a2 = 1 then 9 else 0) + (if a2 = 2 then 18 else 0) := 
sorry

end chosen_number_is_reconstructed_l139_139981


namespace solve_system_l139_139812

theorem solve_system :
  ∃ x y : ℝ, (x^2 + 3 * x * y = 18 ∧ x * y + 3 * y^2 = 6) ∧ ((x = 3 ∧ y = 1) ∨ (x = -3 ∧ y = -1)) :=
by
  sorry

end solve_system_l139_139812


namespace rate_of_current_l139_139573

theorem rate_of_current
  (D U R : ℝ)
  (hD : D = 45)
  (hU : U = 23)
  (hR : R = 34)
  : (D - R = 11) ∧ (R - U = 11) :=
by
  sorry

end rate_of_current_l139_139573


namespace seventh_term_in_geometric_sequence_l139_139738

-- Define the geometric sequence conditions
def first_term : ℝ := 3
def second_term : ℝ := -1/2
def common_ratio : ℝ := second_term / first_term

-- Define the formula for the nth term of the geometric sequence
def nth_term (a r : ℝ) (n : ℕ) : ℝ := a * r^(n-1)

-- The Lean statement for proving the seventh term in the geometric sequence
theorem seventh_term_in_geometric_sequence :
  nth_term first_term common_ratio 7 = 1 / 15552 :=
by
  -- The proof is to be filled in.
  sorry

end seventh_term_in_geometric_sequence_l139_139738


namespace total_hours_charged_l139_139452

theorem total_hours_charged (K P M : ℕ) 
  (h₁ : P = 2 * K)
  (h₂ : P = (1 / 3 : ℚ) * (K + 80))
  (h₃ : M = K + 80) : K + P + M = 144 :=
by {
    sorry
}

end total_hours_charged_l139_139452


namespace range_of_x_l139_139331

theorem range_of_x (x : ℝ) : (4 : ℝ)^(2 * x - 1) > (1 / 2) ^ (-x - 4) → x > 2 := by
  sorry

end range_of_x_l139_139331


namespace pizza_slices_l139_139250

-- Definitions of conditions
def slices (H C : ℝ) : Prop :=
  (H / 2 - 3 + 2 * C / 3 = 11) ∧ (H = C)

-- Stating the theorem to prove
theorem pizza_slices (H C : ℝ) (h : slices H C) : H = 12 :=
sorry

end pizza_slices_l139_139250


namespace calculate_expression_l139_139615

variable (f g : ℝ → ℝ)

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)
def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g x = - g (-x)

theorem calculate_expression 
  (hf : is_even_function f)
  (hg : is_odd_function g)
  (hfg : ∀ x : ℝ, f x - g x = x ^ 3 + x ^ 2 + 1) :
  f 1 + g 1 = 1 :=
  sorry

end calculate_expression_l139_139615


namespace percentage_increase_l139_139436

theorem percentage_increase
  (black_and_white_cost color_cost : ℕ)
  (h_bw : black_and_white_cost = 160)
  (h_color : color_cost = 240) :
  ((color_cost - black_and_white_cost) * 100) / black_and_white_cost = 50 :=
by
  sorry

end percentage_increase_l139_139436


namespace larger_angle_of_nonagon_l139_139476

theorem larger_angle_of_nonagon : 
  ∀ (n : ℕ) (x : ℝ), 
  n = 9 → 
  (∃ a b : ℕ, a + b = n ∧ a * x + b * (3 * x) = 180 * (n - 2)) → 
  3 * (180 * (n - 2) / 15) = 252 :=
by
  sorry

end larger_angle_of_nonagon_l139_139476


namespace S_10_eq_110_l139_139300

-- Conditions
def a (n : ℕ) : ℕ := sorry  -- Assuming general term definition of arithmetic sequence
def S (n : ℕ) : ℕ := sorry  -- Assuming sum definition of arithmetic sequence

axiom a_3_eq_16 : a 3 = 16
axiom S_20_eq_20 : S 20 = 20

-- Prove
theorem S_10_eq_110 : S 10 = 110 :=
  by
  sorry

end S_10_eq_110_l139_139300


namespace opposite_of_113_is_114_l139_139943

theorem opposite_of_113_is_114
  (circle : Finset ℕ)
  (h1 : ∀ n ∈ circle, n ∈ (Finset.range 1 200))
  (h2 : ∀ n, ∃ m, m = (n + 100) % 200)
  (h3 : ∀ n, 1 ≤ n ∧ n ≤ 200)
  : ((113 + 100) % 200 = 114) := by
  sorry

end opposite_of_113_is_114_l139_139943


namespace part1_part2_l139_139251

namespace RationalOp
  -- Define the otimes operation
  def otimes (a b : ℚ) : ℚ := a * b^2 + 2 * a * b + a

  -- Part 1: Prove (-2) ⊗ 4 = -50
  theorem part1 : otimes (-2) 4 = -50 := sorry

  -- Part 2: Given x ⊗ 3 = y ⊗ (-3), prove 8x - 2y + 5 = 5
  theorem part2 (x y : ℚ) (h : otimes x 3 = otimes y (-3)) : 8*x - 2*y + 5 = 5 := sorry
end RationalOp

end part1_part2_l139_139251


namespace remainder_of_2n_div_11_l139_139908

theorem remainder_of_2n_div_11 (n k : ℤ) (h : n = 22 * k + 12) : (2 * n) % 11 = 2 :=
by
  sorry

end remainder_of_2n_div_11_l139_139908


namespace potions_needed_l139_139900

-- Definitions
def galleons_to_knuts (galleons : Int) : Int := galleons * 17 * 23
def sickles_to_knuts (sickles : Int) : Int := sickles * 23

-- Conditions from the problem
def cost_of_owl_in_knuts : Int := galleons_to_knuts 2 + sickles_to_knuts 1 + 5
def knuts_per_potion : Int := 9

-- Prove the number of potions needed is 90
theorem potions_needed : cost_of_owl_in_knuts / knuts_per_potion = 90 := by
  sorry

end potions_needed_l139_139900


namespace machine_subtract_l139_139411

theorem machine_subtract (x : ℤ) (h1 : 26 + 15 - x = 35) : x = 6 :=
by
  sorry

end machine_subtract_l139_139411


namespace cannot_form_62_cents_with_six_coins_l139_139060

-- Define the coin denominations and their values
structure Coin :=
  (value : ℕ)
  (count : ℕ)

def penny : Coin := ⟨1, 6⟩
def nickel : Coin := ⟨5, 6⟩
def dime : Coin := ⟨10, 6⟩
def quarter : Coin := ⟨25, 6⟩
def halfDollar : Coin := ⟨50, 6⟩

-- Define the main theorem statement
theorem cannot_form_62_cents_with_six_coins :
  ¬ (∃ (p n d q h : ℕ),
      p + n + d + q + h = 6 ∧
      1 * p + 5 * n + 10 * d + 25 * q + 50 * h = 62) :=
sorry

end cannot_form_62_cents_with_six_coins_l139_139060


namespace calculate_total_customers_l139_139245

theorem calculate_total_customers 
    (num_no_tip : ℕ) 
    (total_tip_amount : ℕ) 
    (tip_per_customer : ℕ) 
    (number_tipped_customers : ℕ) 
    (number_total_customers : ℕ)
    (h1 : num_no_tip = 5) 
    (h2 : total_tip_amount = 15) 
    (h3 : tip_per_customer = 3) 
    (h4 : number_tipped_customers = total_tip_amount / tip_per_customer) :
    number_total_customers = number_tipped_customers + num_no_tip := 
by {
    sorry
}

end calculate_total_customers_l139_139245


namespace equilateral_triangle_l139_139888

noncomputable def angles_arithmetic_seq (A B C : ℝ) : Prop := B - A = C - B

noncomputable def sides_geometric_seq (a b c : ℝ) : Prop := b / a = c / b

theorem equilateral_triangle 
  (A B C a b c : ℝ) 
  (h_angles : angles_arithmetic_seq A B C) 
  (h_sides : sides_geometric_seq a b c) 
  (h_triangle : A + B + C = π) 
  (h_pos_sides : a > 0 ∧ b > 0 ∧ c > 0) :
  (A = B ∧ B = C) ∧ (a = b ∧ b = c) :=
sorry

end equilateral_triangle_l139_139888


namespace y_not_multiple_of_7_y_not_multiple_of_14_y_not_multiple_of_21_y_not_multiple_of_28_l139_139037

def y : ℕ := 42 + 98 + 210 + 333 + 175 + 28

theorem y_not_multiple_of_7 : ¬ (7 ∣ y) := sorry
theorem y_not_multiple_of_14 : ¬ (14 ∣ y) := sorry
theorem y_not_multiple_of_21 : ¬ (21 ∣ y) := sorry
theorem y_not_multiple_of_28 : ¬ (28 ∣ y) := sorry

end y_not_multiple_of_7_y_not_multiple_of_14_y_not_multiple_of_21_y_not_multiple_of_28_l139_139037


namespace arithmetic_sequence_identity_l139_139269

theorem arithmetic_sequence_identity (a : ℕ → ℝ) (d : ℝ)
    (h_arith : ∀ n, a (n + 1) = a 1 + n * d)
    (h_sum : a 4 + a 7 + a 10 = 30) :
    a 1 - a 3 - a 6 - a 8 - a 11 + a 13 = -20 :=
sorry

end arithmetic_sequence_identity_l139_139269


namespace henry_twice_jill_years_ago_l139_139825

def henry_age : ℕ := 23
def jill_age : ℕ := 17
def sum_of_ages (H J : ℕ) : Prop := H + J = 40

theorem henry_twice_jill_years_ago (H J : ℕ) (H1 : sum_of_ages H J) (H2 : H = 23) (H3 : J = 17) : ∃ x : ℕ, H - x = 2 * (J - x) ∧ x = 11 := 
by
  sorry

end henry_twice_jill_years_ago_l139_139825


namespace cakes_served_yesterday_l139_139575

theorem cakes_served_yesterday (lunch_cakes dinner_cakes total_cakes served_yesterday : ℕ)
  (h1 : lunch_cakes = 5)
  (h2 : dinner_cakes = 6)
  (h3 : total_cakes = 14)
  (h4 : total_cakes = lunch_cakes + dinner_cakes + served_yesterday) :
  served_yesterday = 3 := 
by 
  sorry

end cakes_served_yesterday_l139_139575


namespace katy_books_ratio_l139_139785

theorem katy_books_ratio (J : ℕ) (H1 : 8 + J + (J - 3) = 37) : J / 8 = 2 := 
by
  sorry

end katy_books_ratio_l139_139785


namespace next_ring_together_l139_139707

def nextRingTime (libraryInterval : ℕ) (fireStationInterval : ℕ) (hospitalInterval : ℕ) (start : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm libraryInterval fireStationInterval) hospitalInterval + start

theorem next_ring_together : nextRingTime 18 24 30 (8 * 60) = 14 * 60 :=
by
  sorry

end next_ring_together_l139_139707


namespace ratio_expression_value_l139_139414

theorem ratio_expression_value (p q s u : ℚ) (h1 : p / q = 5 / 2) (h2 : s / u = 11 / 7) : 
  (5 * p * s - 3 * q * u) / (7 * q * u - 2 * p * s) = -233 / 12 :=
by {
  -- Proof will be provided here.
  sorry
}

end ratio_expression_value_l139_139414


namespace forest_coverage_2009_min_annual_growth_rate_l139_139539

variables (a : ℝ)

-- Conditions
def initially_forest_coverage (a : ℝ) := a
def annual_natural_growth_rate := 0.02

-- Questions reformulated:
-- Part 1: Prove the forest coverage at the end of 2009
theorem forest_coverage_2009 : (∃ a : ℝ, (y : ℝ) = a * (1 + 0.02)^5 ∧ y = 1.104 * a) :=
by sorry

-- Part 2: Prove the minimum annual average growth rate by 2014
theorem min_annual_growth_rate : (∀ p : ℝ, (a : ℝ) * (1 + p)^10 ≥ 2 * a → p ≥ 0.072) :=
by sorry

end forest_coverage_2009_min_annual_growth_rate_l139_139539


namespace factorize_difference_of_squares_l139_139123

theorem factorize_difference_of_squares (x y : ℝ) : x^2 - y^2 = (x + y) * (x - y) :=
sorry

end factorize_difference_of_squares_l139_139123


namespace area_of_rhombus_l139_139813

theorem area_of_rhombus (d1 d2 : ℕ) (h1 : d1 = 16) (h2 : d2 = 20) : (d1 * d2) / 2 = 160 := by
  sorry

end area_of_rhombus_l139_139813


namespace tank_holds_gallons_l139_139571

noncomputable def tank_initial_fraction := (7 : ℚ) / 8
noncomputable def tank_partial_fraction := (2 : ℚ) / 3
def gallons_used := 15

theorem tank_holds_gallons
  (x : ℚ) -- number of gallons the tank holds when full
  (h_initial : tank_initial_fraction * x - gallons_used = tank_partial_fraction * x) :
  x = 72 := 
sorry

end tank_holds_gallons_l139_139571


namespace mandy_more_than_three_friends_l139_139863

noncomputable def stickers_given_to_three_friends : ℕ := 4 * 3
noncomputable def total_initial_stickers : ℕ := 72
noncomputable def stickers_left : ℕ := 42
noncomputable def total_given_away : ℕ := total_initial_stickers - stickers_left
noncomputable def mandy_justin_total : ℕ := total_given_away - stickers_given_to_three_friends
noncomputable def mandy_stickers : ℕ := 14
noncomputable def three_friends_stickers : ℕ := stickers_given_to_three_friends

theorem mandy_more_than_three_friends : 
  mandy_stickers - three_friends_stickers = 2 :=
by
  sorry

end mandy_more_than_three_friends_l139_139863


namespace boat_speed_in_still_water_l139_139843

theorem boat_speed_in_still_water (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 5) : b = 8 := 
by
  /- The proof steps would go here -/
  sorry

end boat_speed_in_still_water_l139_139843


namespace triangle_ratio_condition_l139_139422

theorem triangle_ratio_condition (a b c : ℝ) (A B C : ℝ) (h1 : b * Real.cos C + c * Real.cos B = 2 * b)
  (h2 : a = b * Real.sin A / Real.sin B)
  (h3 : b = a * Real.sin B / Real.sin A)
  (h4 : c = a * Real.sin C / Real.sin A)
  (h5 : ∀ x, Real.sin (B + C) = Real.sin x): 
  b / a = 1 / 2 :=
by
  sorry

end triangle_ratio_condition_l139_139422


namespace graphs_intersect_at_one_point_l139_139249

noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x / Real.log 3
noncomputable def g (x : ℝ) : ℝ := Real.log (4 * x) / Real.log 2

theorem graphs_intersect_at_one_point : ∃! x, f x = g x :=
by {
  sorry
}

end graphs_intersect_at_one_point_l139_139249


namespace equation_solution_l139_139980

theorem equation_solution (x y : ℕ) :
  (x^2 + 1)^y - (x^2 - 1)^y = 2 * x^y ↔ 
  (x = 1 ∧ y = 1) ∨ (x = 0 ∧ ∃ k : ℕ, y = 2 * k ∧ k > 0) :=
by sorry

end equation_solution_l139_139980


namespace income_ratio_l139_139822

theorem income_ratio (I1 I2 E1 E2 : ℕ) (h1 : I1 = 5000) (h2 : E1 / E2 = 3 / 2) (h3 : I1 - E1 = 2000) (h4 : I2 - E2 = 2000) : I1 / I2 = 5 / 4 :=
by
  /- Proof omitted -/
  sorry

end income_ratio_l139_139822


namespace inequality_1_inequality_2_inequality_3_inequality_4_l139_139459

-- Definition for the first problem
theorem inequality_1 (x : ℝ) : |2 * x - 1| < 15 ↔ (-7 < x ∧ x < 8) := by
  sorry
  
-- Definition for the second problem
theorem inequality_2 (x : ℝ) : x^2 + 6 * x - 16 < 0 ↔ (-8 < x ∧ x < 2) := by
  sorry

-- Definition for the third problem
theorem inequality_3 (x : ℝ) : |2 * x + 1| > 13 ↔ (x < -7 ∨ x > 6) := by
  sorry

-- Definition for the fourth problem
theorem inequality_4 (x : ℝ) : x^2 - 2 * x > 0 ↔ (x < 0 ∨ x > 2) := by
  sorry

end inequality_1_inequality_2_inequality_3_inequality_4_l139_139459


namespace isosceles_triangle_length_l139_139750

variable (a b : ℝ)

theorem isosceles_triangle_length (h1 : 2 * a + 3 = 16) (h2 : a != 3) : a = 6.5 :=
sorry

end isosceles_triangle_length_l139_139750


namespace sample_size_is_200_l139_139353
-- Define the total number of students and the number of students surveyed
def total_students : ℕ := 3600
def students_surveyed : ℕ := 200

-- Define the sample size
def sample_size := students_surveyed

-- Prove the sample size is 200
theorem sample_size_is_200 : sample_size = 200 :=
by
  -- Placeholder for the actual proof
  sorry

end sample_size_is_200_l139_139353


namespace bounded_variation_iff_diff_non_decreasing_l139_139167

noncomputable theory

open Set

-- Definitions of bounded variation and non-decreasing functions
def bounded_variation (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ M : ℝ, 0 ≤ M ∧ ∀ (P : List (ℝ × ℝ)), (∀ {x y : ℝ}, (x, y) ∈ P → x < y) → 
  (∑ (x_i, x_i1) in P.zip (P.tail ++ [(a, b)]), |f x_i1 - f x_i|) ≤ M

def non_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

-- The theorem we need to prove
theorem bounded_variation_iff_diff_non_decreasing (f : ℝ → ℝ) (a b : ℝ) :
  bounded_variation f a b ↔ 
  ∃ (g h : ℝ → ℝ), non_decreasing g a b ∧ non_decreasing h a b ∧ ∀ x : ℝ, a ≤ x → x ≤ b → f x = g x - h x :=
by sorry

end bounded_variation_iff_diff_non_decreasing_l139_139167


namespace smallest_positive_n_l139_139121

theorem smallest_positive_n (x y z : ℕ) (n : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x ∣ y^3) → (y ∣ z^3) → (z ∣ x^3) → (xyz ∣ (x + y + z)^13) :=
by
  sorry

end smallest_positive_n_l139_139121


namespace round_trip_and_car_percent_single_trip_and_motorcycle_percent_l139_139165

noncomputable def totalPassengers := 100
noncomputable def roundTripPercent := 35
noncomputable def singleTripPercent := 100 - roundTripPercent

noncomputable def roundTripCarPercent := 40
noncomputable def roundTripMotorcyclePercent := 15
noncomputable def roundTripNoVehiclePercent := 60

noncomputable def singleTripCarPercent := 25
noncomputable def singleTripMotorcyclePercent := 10
noncomputable def singleTripNoVehiclePercent := 45

theorem round_trip_and_car_percent : 
  ((roundTripCarPercent / 100) * (roundTripPercent / 100) * totalPassengers) = 14 :=
by
  sorry

theorem single_trip_and_motorcycle_percent :
  ((singleTripMotorcyclePercent / 100) * (singleTripPercent / 100) * totalPassengers) = 6 :=
by
  sorry

end round_trip_and_car_percent_single_trip_and_motorcycle_percent_l139_139165


namespace cos_double_angle_l139_139348
open Real

theorem cos_double_angle (α : ℝ) (h : tan (α - π / 4) = 2) : cos (2 * α) = -4 / 5 := 
sorry

end cos_double_angle_l139_139348


namespace h_h_neg1_l139_139010

def h (x: ℝ) : ℝ := 3 * x^2 - x + 1

theorem h_h_neg1 : h (h (-1)) = 71 := by
  sorry

end h_h_neg1_l139_139010


namespace fewest_four_dollar_frisbees_l139_139107

theorem fewest_four_dollar_frisbees (x y : ℕ) (h1 : x + y = 60) (h2 : 3 * x + 4 * y = 200) : y = 20 :=
by 
  sorry  

end fewest_four_dollar_frisbees_l139_139107


namespace angle_in_first_quadrant_l139_139063

def angle := -999 - 30 / 60 -- defining the angle as -999°30'
def coterminal (θ : Real) : Real := θ + 3 * 360 -- function to compute a coterminal angle

theorem angle_in_first_quadrant : 
  let θ := coterminal angle
  0 <= θ ∧ θ < 90 :=
by
  -- Exact proof steps would go here, but they are omitted as per instructions.
  sorry

end angle_in_first_quadrant_l139_139063


namespace tan_alpha_l139_139286

theorem tan_alpha (α : ℝ) (h : Real.tan (α + Real.pi / 4) = 1 / 5) : Real.tan α = -2 / 3 :=
by
  sorry

end tan_alpha_l139_139286


namespace genuine_product_probability_l139_139231

-- Define the probabilities as constants
def P_second_grade := 0.03
def P_third_grade := 0.01

-- Define the total probability (outcome must be either genuine or substandard)
def P_substandard := P_second_grade + P_third_grade
def P_genuine := 1 - P_substandard

-- The statement to be proved
theorem genuine_product_probability :
  P_genuine = 0.96 :=
sorry

end genuine_product_probability_l139_139231


namespace no_two_perfect_cubes_l139_139160

theorem no_two_perfect_cubes (n : ℕ) : ¬ (∃ a b : ℕ, a^3 = n + 2 ∧ b^3 = n^2 + n + 1) := by
  sorry

end no_two_perfect_cubes_l139_139160


namespace sum_of_c_n_l139_139886

variable {a_n : ℕ → ℕ}    -- Sequence {a_n}
variable {b_n : ℕ → ℕ}    -- Sequence {b_n}
variable {c_n : ℕ → ℕ}    -- Sequence {c_n}
variable {S_n : ℕ → ℕ}    -- Sum of the first n terms of sequence {a_n}
variable {T_n : ℕ → ℕ}    -- Sum of the first n terms of sequence {c_n}

axiom a3 : a_n 3 = 7
axiom S6 : S_n 6 = 48
axiom b_recur : ∀ n : ℕ, 2 * b_n (n + 1) = b_n n + 2
axiom b1 : b_n 1 = 3
axiom c_def : ∀ n : ℕ, c_n n = a_n n * (b_n n - 2)

theorem sum_of_c_n : ∀ n : ℕ, T_n n = 10 - (2*n + 5) * (1 / (2^(n-1))) :=
by
  -- Proof omitted
  sorry

end sum_of_c_n_l139_139886


namespace marly_needs_3_bags_l139_139792

-- Definitions based on the problem conditions
def milk : ℕ := 2
def chicken_stock : ℕ := 3 * milk
def vegetables : ℕ := 1
def total_soup : ℕ := milk + chicken_stock + vegetables
def bag_capacity : ℕ := 3

-- The theorem to prove the number of bags required
theorem marly_needs_3_bags : total_soup / bag_capacity = 3 := 
sorry

end marly_needs_3_bags_l139_139792


namespace zachary_pushups_l139_139700

theorem zachary_pushups (C P : ℕ) (h1 : C = 14) (h2 : P + C = 67) : P = 53 :=
by
  rw [h1] at h2
  linarith

end zachary_pushups_l139_139700


namespace focal_distance_of_ellipse_l139_139894

theorem focal_distance_of_ellipse :
  ∀ (x y : ℝ), (x^2 / 16) + (y^2 / 9) = 1 → (2 * Real.sqrt 7) = 2 * Real.sqrt 7 :=
by
  intros x y hxy
  sorry

end focal_distance_of_ellipse_l139_139894


namespace hexagon_planting_schemes_l139_139771

theorem hexagon_planting_schemes (n m : ℕ) (h : n = 4 ∧ m = 6) : 
  ∃ k, k = 732 := 
by sorry

end hexagon_planting_schemes_l139_139771


namespace value_of_expression_l139_139757

theorem value_of_expression (m : ℝ) (h : m^2 - m - 1 = 0) : m^2 - m + 5 = 6 :=
by
  sorry

end value_of_expression_l139_139757


namespace range_f_x_le_neg_five_l139_139651

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 < x then 2^x - 3 else
if h : x < 0 then 3 - 2^(-x) else 0

theorem range_f_x_le_neg_five :
  ∀ x : ℝ, f x ≤ -5 ↔ x ≤ -3 :=
by sorry

end range_f_x_le_neg_five_l139_139651


namespace find_n_cubes_l139_139126

theorem find_n_cubes (n : ℕ) (x y : ℕ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h1 : 837 + n = y^3) (h2 : 837 - n = x^3) : n = 494 :=
by {
  sorry
}

end find_n_cubes_l139_139126


namespace max_perimeter_convex_quadrilateral_l139_139022

theorem max_perimeter_convex_quadrilateral :
  ∃ (AB BC AD CD AC BD : ℝ), 
    AB = 1 ∧ BC = 1 ∧
    AD ≤ 1 ∧ CD ≤ 1 ∧ AC ≤ 1 ∧ BD ≤ 1 ∧
    2 + 4 * Real.sin (Real.pi / 12) = 
      AB + BC + AD + CD :=
sorry

end max_perimeter_convex_quadrilateral_l139_139022


namespace prime_pairs_divisibility_l139_139125

theorem prime_pairs_divisibility (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (p * q) ∣ (p ^ p + q ^ q + 1) ↔ (p = 2 ∧ q = 5) ∨ (p = 5 ∧ q = 2) :=
by
  sorry

end prime_pairs_divisibility_l139_139125


namespace tom_beach_days_l139_139689

theorem tom_beach_days (total_seashells days_seashells : ℕ) (found_each_day total_found : ℕ) 
    (h1 : found_each_day = 7) (h2 : total_found = 35) : total_found / found_each_day = 5 := 
by 
  sorry

end tom_beach_days_l139_139689


namespace terminal_side_quadrant_l139_139754

-- Given conditions
variables {α : ℝ}
variable (h1 : Real.sin α > 0)
variable (h2 : Real.tan α < 0)

-- Conclusion to be proved
theorem terminal_side_quadrant (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) : 
  (∃ k : ℤ, (k % 2 = 0 ∧ Real.pi * k / 2 < α / 2 ∧ α / 2 < Real.pi / 2 + Real.pi * k) ∨ 
            (k % 2 = 1 ∧ Real.pi * (k - 1) < α / 2 ∧ α / 2 < Real.pi / 4 + Real.pi * (k - 0.5))) :=
by
  sorry

end terminal_side_quadrant_l139_139754


namespace total_volume_correct_l139_139846

-- Define the conditions
def volume_of_hemisphere : ℕ := 4
def number_of_hemispheres : ℕ := 2812

-- Define the target volume
def total_volume_of_water : ℕ := 11248

-- The theorem to be proved
theorem total_volume_correct : volume_of_hemisphere * number_of_hemispheres = total_volume_of_water :=
by
  sorry

end total_volume_correct_l139_139846


namespace island_width_l139_139716

theorem island_width (area length width : ℕ) (h₁ : area = 50) (h₂ : length = 10) : width = area / length := by 
  sorry

end island_width_l139_139716


namespace max_triangles_l139_139830

theorem max_triangles (n : ℕ) (h : n = 10) : 
  ∃ T : ℕ, T = 150 :=
by
  sorry

end max_triangles_l139_139830


namespace max_lambda_leq_64_div_27_l139_139445

theorem max_lambda_leq_64_div_27 (a b c : ℝ) (ha : 0 < a ∧ a ≤ 1) (hb : 0 < b ∧ b ≤ 1) (hc : 0 < c ∧ c ≤ 1) :
  (1:ℝ) + (64:ℝ) / (27:ℝ) * (1 - a) * (1 - b) * (1 - c) ≤ Real.sqrt 3 / Real.sqrt (a + b + c) := 
sorry

end max_lambda_leq_64_div_27_l139_139445


namespace tickets_spent_on_beanie_l139_139588

-- Define the initial number of tickets Jerry had.
def initial_tickets : ℕ := 4

-- Define the number of tickets Jerry won later.
def won_tickets : ℕ := 47

-- Define the current number of tickets Jerry has.
def current_tickets : ℕ := 49

-- The statement of the problem to prove the tickets spent on the beanie.
theorem tickets_spent_on_beanie :
  initial_tickets + won_tickets - 2 = current_tickets := by
  sorry

end tickets_spent_on_beanie_l139_139588


namespace add_mul_of_3_l139_139461

theorem add_mul_of_3 (a b : ℤ) (ha : ∃ m : ℤ, a = 6 * m) (hb : ∃ n : ℤ, b = 9 * n) : ∃ k : ℤ, a + b = 3 * k :=
by
  sorry

end add_mul_of_3_l139_139461


namespace max_min_values_l139_139141

def f (x a : ℝ) : ℝ := -x^2 + 2*x + a

theorem max_min_values (a : ℝ) (h : a ≠ 0) :
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f x a ≤ 1 + a) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 3 ∧ f x a = 1 + a) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 3 → -3 + a ≤ f x a) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 3 ∧ f x a = -3 + a) := 
sorry

end max_min_values_l139_139141


namespace apple_count_l139_139479

/--
Given the total number of apples in several bags, where each bag contains either 12 or 6 apples,
and knowing that the total number of apples is between 70 and 80 inclusive,
prove that the total number of apples can only be 72 or 78.
-/
theorem apple_count (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : ∃ k : ℕ, n = 6 * k) : n = 72 ∨ n = 78 :=
by {
  sorry
}

end apple_count_l139_139479


namespace total_roses_planted_l139_139197

def roses_planted_two_days_ago := 50
def roses_planted_yesterday := roses_planted_two_days_ago + 20
def roses_planted_today := 2 * roses_planted_two_days_ago

theorem total_roses_planted :
  roses_planted_two_days_ago + roses_planted_yesterday + roses_planted_today = 220 := by
  sorry

end total_roses_planted_l139_139197


namespace apples_total_l139_139531

theorem apples_total (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apples_total_l139_139531


namespace function_correct_max_min_values_l139_139765

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

@[simp]
theorem function_correct : (∀ x, f x = 2 * Real.sin (2 * x + Real.pi / 4)) ∧ 
                           (f (3 * Real.pi / 8) = 0) ∧ 
                           (f (Real.pi / 8) = 2) :=
by
  sorry

theorem max_min_values : (∃ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), 
                          f x = -2) ∧ 
                         (∃ x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4), 
                          f x = 2) :=
by
  sorry

end function_correct_max_min_values_l139_139765


namespace apple_bags_l139_139521

theorem apple_bags (n : ℕ) (h₁ : n ≥ 70) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 := 
sorry

end apple_bags_l139_139521


namespace cars_count_l139_139243

-- Define the number of cars as x
variable (x : ℕ)

-- The conditions for the problem
def condition1 := 3 * (x - 2)
def condition2 := 2 * x + 9

-- The main theorem stating that under the given conditions, x = 15
theorem cars_count : condition1 x = condition2 x → x = 15 := by
  sorry

end cars_count_l139_139243


namespace parabola_ordinate_l139_139319

theorem parabola_ordinate (x y : ℝ) (h : y = 2 * x^2) (d : dist (x, y) (0, 1 / 8) = 9 / 8) : y = 1 := 
sorry

end parabola_ordinate_l139_139319


namespace arithmetic_sequence_problem_l139_139861

theorem arithmetic_sequence_problem :
  let sum_first_sequence := (100 / 2) * (2501 + 2600)
  let sum_second_sequence := (100 / 2) * (401 + 500)
  let sum_third_sequence := (50 / 2) * (401 + 450)
  sum_first_sequence - sum_second_sequence - sum_third_sequence = 188725 :=
by
  let sum_first_sequence := (100 / 2) * (2501 + 2600)
  let sum_second_sequence := (100 / 2) * (401 + 500)
  let sum_third_sequence := (50 / 2) * (401 + 450)
  sorry

end arithmetic_sequence_problem_l139_139861


namespace factors_of_P_factorization_of_P_factorize_expression_l139_139349

noncomputable def P (a b c : ℝ) : ℝ :=
  a^2 * (b - c) + b^2 * (c - a) + c^2 * (a - b)

theorem factors_of_P (a b c : ℝ) :
  (a - b ∣ P a b c) ∧ (b - c ∣ P a b c) ∧ (c - a ∣ P a b c) :=
sorry

theorem factorization_of_P (a b c : ℝ) :
  P a b c = -(a - b) * (b - c) * (c - a) :=
sorry

theorem factorize_expression (x y z : ℝ) :
  (x + y + z)^3 - x^3 - y^3 - z^3 = 3 * (x + y) * (y + z) * (z + x) :=
sorry

end factors_of_P_factorization_of_P_factorize_expression_l139_139349


namespace find_a_8_l139_139773

variable {α : Type*} [LinearOrderedField α]
variables (a : ℕ → α) (n : ℕ)

-- Definition of an arithmetic sequence
def is_arithmetic_seq (a : ℕ → α) := ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition
def given_condition (a : ℕ → α) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

-- Main theorem to prove
theorem find_a_8 (h_arith : is_arithmetic_seq a) (h_cond : given_condition a) : a 8 = 24 :=
  sorry

end find_a_8_l139_139773


namespace business_proof_l139_139560

section Business_Problem

variables (investment cost_initial rubles production_capacity : ℕ)
variables (produced_July incomplete_July bottles_August bottles_September days_September : ℕ)
variables (total_depreciation residual_value sales_amount profit_target : ℕ)

def depreciation_per_bottle (cost_initial production_capacity : ℕ) : ℕ := 
    cost_initial / production_capacity

def calculate_total_depreciation (depreciation_per_bottle produced_July bottles_August bottles_September : ℕ) : ℕ :=
    (produced_July * depreciation_per_bottle) + (bottles_August * depreciation_per_bottle) + (bottles_September * depreciation_per_bottle)

def calculate_residual_value (cost_initial total_depreciation : ℕ) : ℕ :=
    cost_initial - total_depreciation

def calculate_sales_amount (residual_value profit_target : ℕ) : ℕ :=
    residual_value + profit_target

theorem business_proof
    (H1: investment = 1500000) 
    (H2: cost_initial = 500000)
    (H3: production_capacity = 100000)
    (H4: produced_July = 200)
    (H5: incomplete_July = 5)
    (H6: bottles_August = 15000)
    (H7: bottles_September = 12300)
    (H8: days_September = 20)
    (H9: total_depreciation = 137500)
    (H10: residual_value = 362500)
    (H11: profit_target = 10000)
    (H12: sales_amount = 372500): 

    total_depreciation = calculate_total_depreciation (depreciation_per_bottle cost_initial production_capacity) produced_July bottles_August bottles_September ∧
    residual_value = calculate_residual_value cost_initial total_depreciation ∧
    sales_amount = calculate_sales_amount residual_value profit_target := 
by 
  sorry

end Business_Problem

end business_proof_l139_139560


namespace solve_system_l139_139172

theorem solve_system :
  ∃ (x1 y1 x2 y2 x3 y3 : ℚ), 
    (x1 = 0 ∧ y1 = 0) ∧ 
    (x2 = -14 ∧ y2 = 6) ∧ 
    (x3 = -85/6 ∧ y3 = 35/6) ∧ 
    ((x1 + 2*y1)*(x1 + 3*y1) = x1 + y1 ∧ (2*x1 + y1)*(3*x1 + y1) = -99*(x1 + y1)) ∧ 
    ((x2 + 2*y2)*(x2 + 3*y2) = x2 + y2 ∧ (2*x2 + y2)*(3*x2 + y2) = -99*(x2 + y2)) ∧ 
    ((x3 + 2*y3)*(x3 + 3*y3) = x3 + y3 ∧ (2*x3 + y3)*(3*x3 + y3) = -99*(x3 + y3)) :=
by
  -- skips the actual proof
  sorry

end solve_system_l139_139172


namespace repeating_decimals_difference_l139_139859

theorem repeating_decimals_difference :
  let x := 234 / 999
  let y := 567 / 999
  let z := 891 / 999
  x - y - z = -408 / 333 :=
by
  sorry

end repeating_decimals_difference_l139_139859


namespace find_f_2021_l139_139815

noncomputable def f (x : ℝ) : ℝ := sorry

lemma functional_equation (a b : ℝ) : f ((a + 2 * b) / 3) = (f a + 2 * f b) / 3 :=
sorry

lemma f_one : f 1 = 1 :=
sorry

lemma f_four : f 4 = 7 :=
sorry

theorem find_f_2021 : f 2021 = 4041 :=
sorry

end find_f_2021_l139_139815


namespace apple_bags_l139_139494

theorem apple_bags (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apple_bags_l139_139494


namespace calculate_expression_l139_139591

theorem calculate_expression :
  3 * (1 / 2) ^ (-2 : ℤ) + |2 - Real.pi| + (-3) ^ (0 : ℤ) = 11 + Real.pi := by
  have h1: (1 / 2) ^ (-2 : ℤ) = 4 := by sorry,
  have h2: |2 - Real.pi| = Real.pi - 2 := by sorry,
  have h3: (-3) ^ (0 : ℤ) = 1 := by sorry,
  rw [h1, h2, h3],
  ring,
  sorry

end calculate_expression_l139_139591


namespace opposite_number_on_circle_l139_139939

/-- Given numbers 1 through 200 are arranged in a circle with equal distances, this theorem 
    proves that the number opposite to 113 in such an arrangement is 114. -/
theorem opposite_number_on_circle : 
  ∀ (n : ℕ), n ∈ finset.range 1 201 → ((113 + 100) % 200 = 114) :=
by
  intro n hn,
  have h : 113 + 100 = 213 := rfl,
  have h_mod : 213 % 200 = 13 := rfl,
  sorry

end opposite_number_on_circle_l139_139939


namespace find_ordered_triplets_l139_139845

theorem find_ordered_triplets (x y z : ℝ) :
  x^3 = z / y - 2 * y / z ∧
  y^3 = x / z - 2 * z / x ∧
  z^3 = y / x - 2 * x / y →
  (x = 1 ∧ y = 1 ∧ z = -1) ∨
  (x = 1 ∧ y = -1 ∧ z = 1) ∨
  (x = -1 ∧ y = 1 ∧ z = 1) ∨
  (x = -1 ∧ y = -1 ∧ z = -1) :=
sorry

end find_ordered_triplets_l139_139845


namespace value_of_double_operation_l139_139611

def op1 (x : ℝ) : ℝ := 9 - x
def op2 (x : ℝ) : ℝ := x - 9

theorem value_of_double_operation :
  op2 (op1 10) = -10 := 
by 
  sorry

end value_of_double_operation_l139_139611


namespace number_of_members_l139_139227

theorem number_of_members (n : ℕ) (h : n * n = 8649) : n = 93 :=
by
  sorry

end number_of_members_l139_139227


namespace ones_digit_of_34_34_times_17_17_is_6_l139_139876

def cyclical_pattern_4 (n : ℕ) : ℕ :=
if n % 2 = 0 then 6 else 4

theorem ones_digit_of_34_34_times_17_17_is_6
  (h1 : 34 % 10 = 4)
  (h2 : ∀ n : ℕ, cyclical_pattern_4 n = if n % 2 = 0 then 6 else 4)
  (h3 : 17 % 2 = 1)
  (h4 : (34 * 17^17) % 2 = 0)
  (h5 : ∀ n : ℕ, cyclical_pattern_4 n = if n % 2 = 0 then 6 else 4) :
  (34^(34 * 17^17)) % 10 = 6 := 
by  
  sorry

end ones_digit_of_34_34_times_17_17_is_6_l139_139876


namespace expression_of_fn_l139_139746

noncomputable def f (n : ℕ) (x : ℝ) : ℝ :=
if n = 0 then x else f (n - 1) x / (1 + n * x)

theorem expression_of_fn (n : ℕ) (x : ℝ) (hn : 1 ≤ n) : f n x = x / (1 + n * x) :=
sorry

end expression_of_fn_l139_139746


namespace intercept_sum_l139_139182

noncomputable def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 4

def x_intercept := 4

def y_intercepts : ℝ × ℝ :=
  let delta := (9 : ℝ)^2 - 4 * 3 * 4
  ((9 - Real.sqrt delta)/6, (9 + Real.sqrt delta)/6)

def a : ℝ := x_intercept
def b : ℝ := (y_intercepts.1)
def c : ℝ := (y_intercepts.2)

theorem intercept_sum : a + b + c = 7 := by
  have h_delta : (9 : ℝ)^2 - 4 * 3 * 4 = 33 := by
    sorry
  have h_b : b = (9 - Real.sqrt 33) / 6 := by
    simp only [b, y_intercepts]
    sorry
  have h_c : c = (9 + Real.sqrt 33) / 6 := by
    simp only [c, y_intercepts]
    sorry
  simp [a, b, c, h_b, h_c]
  field_simp
  ring
  sorry

end intercept_sum_l139_139182


namespace frigate_catches_smuggler_at_five_l139_139099

noncomputable def time_to_catch : ℝ :=
  2 + (12 / 4) -- Initial leading distance / Relative speed before storm
  
theorem frigate_catches_smuggler_at_five 
  (initial_distance : ℝ)
  (frigate_speed_before_storm : ℝ)
  (smuggler_speed_before_storm : ℝ)
  (time_before_storm : ℝ)
  (frigate_speed_after_storm : ℝ)
  (smuggler_speed_after_storm : ℝ) :
  initial_distance = 12 →
  frigate_speed_before_storm = 14 →
  smuggler_speed_before_storm = 10 →
  time_before_storm = 3 →
  frigate_speed_after_storm = 12 →
  smuggler_speed_after_storm = 9 →
  time_to_catch = 5 :=
by
{
  sorry
}

end frigate_catches_smuggler_at_five_l139_139099


namespace correct_proposition_l139_139278

-- Defining the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 2

-- Defining proposition p
def p : Prop := ∃ x : ℝ, 0 < x ∧ x < 2 ∧ f x < 0

-- Defining proposition q
def q : Prop := ∀ x y : ℝ, x + y > 4 → x > 2 ∧ y > 2

-- Theorem statement to prove the correct answer
theorem correct_proposition : (¬ p) ∧ (¬ q) :=
by
  sorry

end correct_proposition_l139_139278


namespace kids_on_soccer_field_l139_139829

theorem kids_on_soccer_field (n f : ℕ) (h1 : n = 14) (h2 : f = 3) :
  n + n * f = 56 :=
by
  sorry

end kids_on_soccer_field_l139_139829


namespace rectangle_area_diagonal_ratio_l139_139474

theorem rectangle_area_diagonal_ratio (d : ℝ) (x : ℝ) (h_ratio : 5 * x ≥ 0 ∧ 2 * x ≥ 0)
  (h_diagonal : d^2 = (5 * x)^2 + (2 * x)^2) :
  ∃ k : ℝ, (5 * x) * (2 * x) = k * d^2 ∧ k = 10 / 29 :=
by
  sorry

end rectangle_area_diagonal_ratio_l139_139474


namespace train_length_calculation_l139_139340

theorem train_length_calculation (L : ℝ) (t : ℝ) (v_faster : ℝ) (v_slower : ℝ) (relative_speed : ℝ) (total_distance : ℝ) :
  (v_faster = 60) →
  (v_slower = 40) →
  (relative_speed = (v_faster - v_slower) * 1000 / 3600) →
  (t = 48) →
  (total_distance = relative_speed * t) →
  (2 * L = total_distance) →
  L = 133.44 :=
by
  intros
  sorry

end train_length_calculation_l139_139340


namespace matchsticks_in_20th_stage_l139_139687

-- Define the first term and common difference
def first_term : ℕ := 4
def common_difference : ℕ := 3

-- Define the mathematical function for the n-th term of the arithmetic sequence
def num_matchsticks (n : ℕ) : ℕ :=
  first_term + (n - 1) * common_difference

-- State the theorem to prove the number of matchsticks in the 20th stage
theorem matchsticks_in_20th_stage : num_matchsticks 20 = 61 :=
by
  -- Proof skipped
  sorry

end matchsticks_in_20th_stage_l139_139687


namespace number_of_terms_in_sequence_l139_139407

theorem number_of_terms_in_sequence : ∃ n : ℕ, 6 + (n-1) * 4 = 154 ∧ n = 38 :=
by
  sorry

end number_of_terms_in_sequence_l139_139407


namespace min_value_expression_eq_2sqrt3_l139_139790

noncomputable def min_value_expression (c d : ℝ) : ℝ :=
  c^2 + d^2 + 4 / c^2 + 2 * d / c

theorem min_value_expression_eq_2sqrt3 (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ y : ℝ, (∀ d : ℝ, min_value_expression c d ≥ y) ∧ y = 2 * Real.sqrt 3 :=
sorry

end min_value_expression_eq_2sqrt3_l139_139790


namespace function_has_zero_in_interval_l139_139714

   theorem function_has_zero_in_interval (fA fB fC fD : ℝ → ℝ) (hA : ∀ x, fA x = x - 3)
       (hB : ∀ x, fB x = 2^x) (hC : ∀ x, fC x = x^2) (hD : ∀ x, fD x = Real.log x) :
       ∃ x, 0 < x ∧ x < 2 ∧ fD x = 0 :=
   by
       sorry
   
end function_has_zero_in_interval_l139_139714


namespace simplify_expression_l139_139610

theorem simplify_expression : 
  let a := (3 + 2 : ℚ)
  let b := a⁻¹ + 2
  let c := b⁻¹ + 2
  let d := c⁻¹ + 2
  d = 65 / 27 := by
  sorry

end simplify_expression_l139_139610


namespace total_skateboarding_distance_l139_139437

def skateboarded_to_park : ℕ := 16
def skateboarded_back_home : ℕ := 9

theorem total_skateboarding_distance : 
  skateboarded_to_park + skateboarded_back_home = 25 := by 
  sorry

end total_skateboarding_distance_l139_139437


namespace ceil_minus_val_eq_one_minus_frac_l139_139150

variable (x : ℝ)

theorem ceil_minus_val_eq_one_minus_frac (h : ⌈x⌉ - ⌊x⌋ = 1) :
  ∃ f : ℝ, 0 ≤ f ∧ f < 1 ∧ ⌈x⌉ - x = 1 - f := 
sorry

end ceil_minus_val_eq_one_minus_frac_l139_139150


namespace inequality_x_y_l139_139393

theorem inequality_x_y (x y : ℝ) (h : x^12 + y^12 ≤ 2) : x^2 + y^2 + x^2 * y^2 ≤ 3 := 
sorry

end inequality_x_y_l139_139393


namespace tall_cupboard_glasses_l139_139118

-- Define the number of glasses held by the tall cupboard (T)
variable (T : ℕ)

-- Condition: Wide cupboard holds twice as many glasses as the tall cupboard
def wide_cupboard_holds_twice_as_many (T : ℕ) : Prop :=
  ∃ W : ℕ, W = 2 * T

-- Condition: Narrow cupboard holds 15 glasses initially, 5 glasses per shelf, one shelf broken
def narrow_cupboard_holds_after_break : Prop :=
  ∃ N : ℕ, N = 10

-- Final statement to prove: Number of glasses in the tall cupboard is 5
theorem tall_cupboard_glasses (T : ℕ) (h1 : wide_cupboard_holds_twice_as_many T) (h2 : narrow_cupboard_holds_after_break) : T = 5 :=
sorry

end tall_cupboard_glasses_l139_139118


namespace total_annual_car_maintenance_expenses_is_330_l139_139645

-- Define the conditions as constants
def annualMileage : ℕ := 12000
def milesPerOilChange : ℕ := 3000
def freeOilChangesPerYear : ℕ := 1
def costPerOilChange : ℕ := 50
def milesPerTireRotation : ℕ := 6000
def costPerTireRotation : ℕ := 40
def milesPerBrakePadReplacement : ℕ := 24000
def costPerBrakePadReplacement : ℕ := 200

-- Define the total annual car maintenance expenses calculation
def annualOilChangeExpenses (annualMileage : ℕ) (milesPerOilChange : ℕ) (freeOilChangesPerYear : ℕ) (costPerOilChange : ℕ) : ℕ :=
  let oilChangesNeeded := annualMileage / milesPerOilChange
  let paidOilChanges := oilChangesNeeded - freeOilChangesPerYear
  paidOilChanges * costPerOilChange

def annualTireRotationExpenses (annualMileage : ℕ) (milesPerTireRotation : ℕ) (costPerTireRotation : ℕ) : ℕ :=
  let tireRotationsNeeded := annualMileage / milesPerTireRotation
  tireRotationsNeeded * costPerTireRotation

def annualBrakePadReplacementExpenses (annualMileage : ℕ) (milesPerBrakePadReplacement : ℕ) (costPerBrakePadReplacement : ℕ) : ℕ :=
  let brakePadReplacementInterval := milesPerBrakePadReplacement / annualMileage
  costPerBrakePadReplacement / brakePadReplacementInterval

def totalAnnualCarMaintenanceExpenses : ℕ :=
  annualOilChangeExpenses annualMileage milesPerOilChange freeOilChangesPerYear costPerOilChange +
  annualTireRotationExpenses annualMileage milesPerTireRotation costPerTireRotation +
  annualBrakePadReplacementExpenses annualMileage milesPerBrakePadReplacement costPerBrakePadReplacement

-- Prove the total annual car maintenance expenses equals $330
theorem total_annual_car_maintenance_expenses_is_330 : totalAnnualCarMaintenanceExpenses = 330 := by
  sorry

end total_annual_car_maintenance_expenses_is_330_l139_139645


namespace grocery_cost_l139_139194

def rent : ℕ := 1100
def utilities : ℕ := 114
def roommate_payment : ℕ := 757

theorem grocery_cost (total_payment : ℕ) (half_rent_utilities : ℕ) (half_groceries : ℕ) (total_groceries : ℕ) :
  total_payment = 757 →
  half_rent_utilities = (rent + utilities) / 2 →
  half_groceries = total_payment - half_rent_utilities →
  total_groceries = half_groceries * 2 →
  total_groceries = 300 :=
by
  intros
  sorry

end grocery_cost_l139_139194


namespace additional_weekly_rate_l139_139466

theorem additional_weekly_rate (rate_first_week : ℝ) (total_days_cost : ℝ) (days_first_week : ℕ) (total_days : ℕ) (cost_total : ℝ) (cost_first_week : ℝ) (days_after_first_week : ℕ) : 
  (rate_first_week * days_first_week = cost_first_week) → 
  (total_days = days_first_week + days_after_first_week) → 
  (cost_total = cost_first_week + (days_after_first_week * (rate_first_week * 7 / days_first_week))) →
  (rate_first_week = 18) →
  (cost_total = 350) →
  total_days = 23 → 
  (days_first_week = 7) → 
  cost_first_week = 126 →
  (days_after_first_week = 16) →
  rate_first_week * 7 / days_first_week * days_after_first_week = 14 := 
by 
  sorry

end additional_weekly_rate_l139_139466


namespace marnie_eats_chips_l139_139795

theorem marnie_eats_chips (total_chips : ℕ) (chips_first_batch : ℕ) (chips_second_batch : ℕ) (daily_chips : ℕ) (remaining_chips : ℕ) (total_days : ℕ) :
  total_chips = 100 →
  chips_first_batch = 5 →
  chips_second_batch = 5 →
  daily_chips = 10 →
  remaining_chips = total_chips - (chips_first_batch + chips_second_batch) →
  total_days = remaining_chips / daily_chips + 1 →
  total_days = 10 :=
by
  sorry

end marnie_eats_chips_l139_139795


namespace geometric_sequence_sum_l139_139776

theorem geometric_sequence_sum (a : ℕ → ℤ) (r : ℤ) (h_geom : ∀ n, a (n + 1) = a n * r)
  (h1 : a 0 + a 1 + a 2 = 8)
  (h2 : a 3 + a 4 + a 5 = -4) :
  a 6 + a 7 + a 8 = 2 := 
sorry

end geometric_sequence_sum_l139_139776


namespace converse_true_inverse_true_contrapositive_false_sufficiency_necessity_l139_139453

-- Define the original proposition with conditions
def prop : Prop := ∀ (m n : ℝ), m ≤ 0 ∨ n ≤ 0 → m + n ≤ 0

-- Identify converse, inverse, and contrapositive
def converse : Prop := ∀ (m n : ℝ), m + n ≤ 0 → m ≤ 0 ∨ n ≤ 0
def inverse : Prop := ∀ (m n : ℝ), m > 0 ∧ n > 0 → m + n > 0
def contrapositive : Prop := ∀ (m n : ℝ), m + n > 0 → m > 0 ∧ n > 0

-- Identifying the conditions of sufficiency and necessity
def necessary_but_not_sufficient (p q : Prop) : Prop := 
  (¬p → ¬q) ∧ (q → p) ∧ ¬(p → q)

-- Prove or provide the statements
theorem converse_true : converse := sorry
theorem inverse_true : inverse := sorry
theorem contrapositive_false : ¬contrapositive := sorry
theorem sufficiency_necessity : necessary_but_not_sufficient 
  (∀ (m n : ℝ), m ≤ 0 ∨ n ≤ 0) 
  (∀ (m n : ℝ), m + n ≤ 0) := sorry

end converse_true_inverse_true_contrapositive_false_sufficiency_necessity_l139_139453


namespace inequality_inverse_l139_139906

theorem inequality_inverse (a b : ℝ) (h : a > b ∧ b > 0) : (1 / a) < (1 / b) :=
by
  sorry

end inequality_inverse_l139_139906


namespace system_of_equations_correct_l139_139964

theorem system_of_equations_correct (x y : ℤ) :
  (8 * x - 3 = y) ∧ (7 * x + 4 = y) :=
sorry

end system_of_equations_correct_l139_139964


namespace matchsticks_in_20th_stage_l139_139686

-- Define the first term and common difference
def first_term : ℕ := 4
def common_difference : ℕ := 3

-- Define the mathematical function for the n-th term of the arithmetic sequence
def num_matchsticks (n : ℕ) : ℕ :=
  first_term + (n - 1) * common_difference

-- State the theorem to prove the number of matchsticks in the 20th stage
theorem matchsticks_in_20th_stage : num_matchsticks 20 = 61 :=
by
  -- Proof skipped
  sorry

end matchsticks_in_20th_stage_l139_139686


namespace usable_parking_lot_percentage_l139_139432

theorem usable_parking_lot_percentage
  (length width : ℝ) (area_per_car : ℝ) (number_of_cars : ℝ)
  (h_len : length = 400)
  (h_wid : width = 500)
  (h_area_car : area_per_car = 10)
  (h_cars : number_of_cars = 16000) :
  ((number_of_cars * area_per_car) / (length * width) * 100) = 80 := 
by
  -- Proof omitted
  sorry

end usable_parking_lot_percentage_l139_139432


namespace solve_parabola_l139_139620

theorem solve_parabola (a b c : ℝ) 
  (h1 : 1 = a * 1^2 + b * 1 + c)
  (h2 : 4 * a + b = 1)
  (h3 : -1 = a * 2^2 + b * 2 + c) :
  a = 3 ∧ b = -11 ∧ c = 9 :=
by {
  sorry
}

end solve_parabola_l139_139620


namespace max_value_of_b_l139_139761

theorem max_value_of_b (a b c : ℝ) (q : ℝ) (hq : q ≠ 0) 
  (h_geom : a = b / q ∧ c = b * q) 
  (h_arith : 2 * b + 4 = a + 6 + (b + 2) + (c + 1) - (b + 2)) :
  b ≤ 3 / 4 :=
sorry

end max_value_of_b_l139_139761


namespace part1_part2_l139_139000

noncomputable def f (a x : ℝ) : ℝ := (a * Real.exp x - a - x) * Real.exp x

theorem part1 (a : ℝ) (h0 : a ≥ 0) (h1 : ∀ x : ℝ, f a x ≥ 0) : a = 1 := 
sorry

theorem part2 (h1 : ∀ x : ℝ, f 1 x ≥ 0) :
  ∃! x0 : ℝ, (∀ x : ℝ, x0 = x → 
  (f 1 x0) = (f 1 x)) ∧ (0 < f 1 x0 ∧ f 1 x0 < 1/4) :=
sorry

end part1_part2_l139_139000


namespace average_of_rest_of_class_l139_139979

theorem average_of_rest_of_class
  (n : ℕ)
  (h1 : n > 0)
  (avg_class : ℝ := 84)
  (avg_one_fourth : ℝ := 96)
  (total_sum : ℝ := avg_class * n)
  (sum_one_fourth : ℝ := avg_one_fourth * (n / 4))
  (sum_rest : ℝ := total_sum - sum_one_fourth)
  (num_rest : ℝ := (3 * n) / 4) :
  sum_rest / num_rest = 80 :=
sorry

end average_of_rest_of_class_l139_139979


namespace possible_apple_counts_l139_139502

theorem possible_apple_counts (n : ℕ) (h₁ : 70 ≤ n) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end possible_apple_counts_l139_139502


namespace problem_1_problem_2_l139_139384

theorem problem_1 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 + c^2 = 9) : abc ≤ 3 * Real.sqrt 3 := 
sorry

theorem problem_2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 + c^2 = 9) : 
  (a^2 / (b + c)) + (b^2 / (c + a)) + (c^2 / (a + b)) > (a + b + c) / 3 := 
sorry

end problem_1_problem_2_l139_139384


namespace population_increase_l139_139959

theorem population_increase (P : ℝ) (h₁ : 11000 * (1 + P / 100) * (1 + P / 100) = 13310) : 
  P = 10 :=
sorry

end population_increase_l139_139959


namespace sum_of_remainders_l139_139038

open Finset

def R : Finset (Zmod 500) :=
  (range 100).image (λ n, (3 ^ n : Zmod 500))

def S : Zmod 500 :=
  R.sum id

theorem sum_of_remainders :
  S = 0 := sorry

end sum_of_remainders_l139_139038


namespace total_students_l139_139770

theorem total_students (S : ℕ) (R : ℕ) :
  (2 * 0 + 12 * 1 + 13 * 2 + R * 3) / S = 2 →
  2 + 12 + 13 + R = S →
  S = 43 :=
by
  sorry

end total_students_l139_139770


namespace find_value_of_expression_l139_139415

variable (α : ℝ)

theorem find_value_of_expression 
  (h : Real.tan α = 3) : 
  (Real.sin (2 * α) / (Real.cos α)^2) = 6 := 
by 
  sorry

end find_value_of_expression_l139_139415


namespace problem1_problem2_problem3_l139_139811

-- First problem: Prove x = 4.2 given x + 2x = 12.6
theorem problem1 (x : ℝ) (h1 : x + 2 * x = 12.6) : x = 4.2 :=
  sorry

-- Second problem: Prove x = 2/5 given 1/4 * x + 1/2 = 3/5
theorem problem2 (x : ℚ) (h2 : (1 / 4) * x + 1 / 2 = 3 / 5) : x = 2 / 5 :=
  sorry

-- Third problem: Prove x = 20 given x + 130% * x = 46 (where 130% is 130/100)
theorem problem3 (x : ℝ) (h3 : x + (130 / 100) * x = 46) : x = 20 :=
  sorry

end problem1_problem2_problem3_l139_139811


namespace apple_count_l139_139482

/--
Given the total number of apples in several bags, where each bag contains either 12 or 6 apples,
and knowing that the total number of apples is between 70 and 80 inclusive,
prove that the total number of apples can only be 72 or 78.
-/
theorem apple_count (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : ∃ k : ℕ, n = 6 * k) : n = 72 ∨ n = 78 :=
by {
  sorry
}

end apple_count_l139_139482


namespace find_f_neg_5_l139_139398

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 5 + 1 else - (log (-x) / log 5 + 1)

theorem find_f_neg_5
  (h_odd : ∀ x : ℝ, f (-x) = -f (x))
  (h_domain : ∀ x : ℝ, x ∈ set.univ)
  (h_positive_def : ∀ x : ℝ, x > 0 → f x = log x / log 5 + 1)
  : f (-5) = -2 :=
by
  sorry

end find_f_neg_5_l139_139398


namespace length_AC_l139_139267

theorem length_AC {AB BC : ℝ} (h1: AB = 6) (h2: BC = 4) : (AC = 2 ∨ AC = 10) :=
sorry

end length_AC_l139_139267


namespace quotient_of_501_div_0_point_5_l139_139680

theorem quotient_of_501_div_0_point_5 : 501 / 0.5 = 1002 := by
  sorry

end quotient_of_501_div_0_point_5_l139_139680


namespace part_I_part_II_l139_139276

noncomputable def f (x : ℝ) := (Real.sin x) * (Real.cos x) + (Real.sin x)^2

-- Part I: Prove that f(π / 4) = 1
theorem part_I : f (Real.pi / 4) = 1 := sorry

-- Part II: Prove that the maximum value of f(x) for x ∈ [0, π / 2] is (√2 + 1) / 2
theorem part_II : ∃ x ∈ Set.Icc 0 (Real.pi / 2), (∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≤ f x) ∧ f x = (Real.sqrt 2 + 1) / 2 := sorry

end part_I_part_II_l139_139276


namespace initial_discount_l139_139367

theorem initial_discount (P D : ℝ) 
  (h1 : P - 71.4 = 5.25)
  (h2 : P * (1 - D) * 1.25 = 71.4) : 
  D = 0.255 :=
by {
  sorry
}

end initial_discount_l139_139367


namespace valid_number_of_apples_l139_139508

theorem valid_number_of_apples (n : ℕ) : 
  (∃ k m : ℕ, n = 12 * k + 6 * m) ∧ (70 ≤ n ∧ n ≤ 80) ↔ (n = 72 ∨ n = 78) :=
by
  sorry

end valid_number_of_apples_l139_139508


namespace half_of_expression_correct_l139_139969

theorem half_of_expression_correct :
  (2^12 + 3 * 2^10) / 2 = 2^9 * 7 :=
by
  sorry

end half_of_expression_correct_l139_139969


namespace shaded_area_l139_139774

theorem shaded_area (area_large : ℝ) (area_small : ℝ) (n_small_squares : ℕ) 
  (n_triangles: ℕ) (area_total : ℝ) : 
  area_large = 16 → 
  area_small = 1 → 
  n_small_squares = 4 → 
  n_triangles = 4 → 
  area_total = 4 → 
  4 * area_small = 4 →
  area_large - (area_total + (n_small_squares * area_small)) = 4 :=
by
  intros
  sorry

end shaded_area_l139_139774


namespace find_c_l139_139159

theorem find_c (c : ℝ) (h : ∀ x y : ℝ, 5 * x + 8 * y + c = 0 ∧ x + y = 26) : c = -80 :=
sorry

end find_c_l139_139159


namespace product_of_fractions_is_3_div_80_l139_139547

def product_fractions (a b c d e f : ℚ) : ℚ := (a / b) * (c / d) * (e / f)

theorem product_of_fractions_is_3_div_80 
  (h₁ : product_fractions 3 8 2 5 1 4 = 3 / 80) : True :=
by
  sorry

end product_of_fractions_is_3_div_80_l139_139547


namespace smallest_sum_l139_139654

-- First, we define the conditions as assumptions:
def is_arithmetic_sequence (x y z : ℕ) : Prop :=
  2 * y = x + z

def is_geometric_sequence (x y z : ℕ) : Prop :=
  y ^ 2 = x * z

-- Given conditions
variables (A B C D : ℕ)
variables (hABC : is_arithmetic_sequence A B C) (hBCD : is_geometric_sequence B C D)
variables (h_ratio : 4 * C = 7 * B)

-- The main theorem to prove
theorem smallest_sum : A + B + C + D = 97 :=
sorry

end smallest_sum_l139_139654


namespace find_p_fulfill_condition_l139_139616

noncomputable def P_fulfills_angle_condition (p : ℝ) : Prop :=
  ∀ (A B : ℝ × ℝ), A ≠ B → (A.1^2 / 4 + A.2^2 = 1) → (B.1^2 / 4 + B.2^2 = 1) → 
  (∃ F : ℝ × ℝ, F = (Real.sqrt 3, 0) ∧ A.x = Real.sqrt 3 ∧ B.x = Real.sqrt 3) →
  let P := (p, 0) in
  (angle A P F = angle B P F)

theorem find_p_fulfill_condition : P_fulfills_angle_condition (Real.sqrt 3) :=
sorry

end find_p_fulfill_condition_l139_139616


namespace pencils_to_make_profit_l139_139578

theorem pencils_to_make_profit
  (total_pencils : ℕ)
  (cost_per_pencil : ℝ)
  (selling_price_per_pencil : ℝ)
  (desired_profit : ℝ)
  (pencils_to_be_sold : ℕ) :
  total_pencils = 2000 →
  cost_per_pencil = 0.08 →
  selling_price_per_pencil = 0.20 →
  desired_profit = 160 →
  pencils_to_be_sold = 1600 :=
sorry

end pencils_to_make_profit_l139_139578


namespace vans_needed_l139_139326

theorem vans_needed (boys girls students_per_van total_vans : ℕ) 
  (hb : boys = 60) 
  (hg : girls = 80) 
  (hv : students_per_van = 28) 
  (t : total_vans = (boys + girls) / students_per_van) : 
  total_vans = 5 := 
by {
  sorry
}

end vans_needed_l139_139326


namespace probability_beautiful_equation_l139_139166

def tetrahedron_faces : Set ℕ := {1, 2, 3, 4}

def is_beautiful_equation (a b : ℕ) : Prop :=
    ∃ m ∈ tetrahedron_faces, a = m + 1 ∨ a = m + 2 ∨ a = m + 3 ∨ a = m + 4 ∧ b = m * (a - m)

theorem probability_beautiful_equation : 
  (∃ a b1 b2, is_beautiful_equation a b1 ∧ is_beautiful_equation a b2) ∧
  (∃ a b1 b2, tetrahedron_faces ⊆ {a} ∧ tetrahedron_faces ⊆ {b1} ∧ tetrahedron_faces ⊆ {b2}) :=
  sorry

end probability_beautiful_equation_l139_139166


namespace apple_bags_l139_139526

theorem apple_bags (n : ℕ) (h₁ : n ≥ 70) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 := 
sorry

end apple_bags_l139_139526


namespace integer_roots_of_polynomial_l139_139731

theorem integer_roots_of_polynomial :
  ∀ x : ℤ, (x^3 - 3 * x^2 - 13 * x + 15 = 0) → (x = -3 ∨ x = 1 ∨ x = 5) :=
by
  sorry

end integer_roots_of_polynomial_l139_139731


namespace possible_apple_counts_l139_139506

theorem possible_apple_counts (n : ℕ) (h₁ : 70 ≤ n) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end possible_apple_counts_l139_139506


namespace total_bike_count_l139_139030

def total_bikes (bikes_jungkook bikes_yoongi : Nat) : Nat :=
  bikes_jungkook + bikes_yoongi

theorem total_bike_count : total_bikes 3 4 = 7 := 
  by 
  sorry

end total_bike_count_l139_139030


namespace number_of_silverware_per_setting_l139_139796

-- Conditions
def silverware_weight_per_piece := 4   -- in ounces
def plates_per_setting := 2
def plate_weight := 12  -- in ounces
def tables := 15
def settings_per_table := 8
def backup_settings := 20
def total_weight := 5040  -- in ounces

-- Let's define variables in our conditions
def settings := tables * settings_per_table + backup_settings
def plates_weight_per_setting := plates_per_setting * plate_weight
def total_silverware_weight (S : Nat) := S * silverware_weight_per_piece * settings
def total_plate_weight := plates_weight_per_setting * settings

-- Define the required proof statement
theorem number_of_silverware_per_setting : 
  ∃ S : Nat, (total_silverware_weight S + total_plate_weight = total_weight) ∧ S = 3 :=
by {
  sorry -- proof will be provided here
}

end number_of_silverware_per_setting_l139_139796


namespace parabola_intercepts_l139_139181

noncomputable def question (y : ℝ) := 3 * y ^ 2 - 9 * y + 4

theorem parabola_intercepts (a b c : ℝ) (h_a : a = question 0) (h_b : 3 * b ^ 2 - 9 * b + 4 = 0) (h_c : 3 * c ^ 2 - 9 * c + 4 = 0) :
  a + b + c = 7 :=
by
  sorry

end parabola_intercepts_l139_139181


namespace cost_of_whistle_l139_139051

theorem cost_of_whistle (cost_yoyo : ℕ) (total_spent : ℕ) (cost_yoyo_equals : cost_yoyo = 24) (total_spent_equals : total_spent = 38) : (total_spent - cost_yoyo) = 14 :=
by
  sorry

end cost_of_whistle_l139_139051


namespace distinct_pairs_l139_139598

theorem distinct_pairs (x y : ℝ) (h : x ≠ y) :
  x^100 - y^100 = 2^99 * (x - y) ∧ x^200 - y^200 = 2^199 * (x - y) ↔ (x = 2 ∧ y = 0) ∨ (x = 0 ∧ y = 2) :=
by
  sorry

end distinct_pairs_l139_139598


namespace greatest_prime_factor_of_factorial_sum_l139_139212

theorem greatest_prime_factor_of_factorial_sum : 
  ∃ p, Prime p ∧ p > 11 ∧ (∀ q, Prime q ∧ q > 11 → q ≤ 61) ∧ p = 61 :=
by
  sorry

end greatest_prime_factor_of_factorial_sum_l139_139212


namespace solve_system_l139_139144

theorem solve_system :
  ∀ (a1 a2 c1 c2 x y : ℝ),
  (a1 * 5 + 10 = c1) →
  (a2 * 5 + 10 = c2) →
  (a1 * x + 2 * y = a1 - c1) →
  (a2 * x + 2 * y = a2 - c2) →
  (x = -4) ∧ (y = -5) := by
  intros a1 a2 c1 c2 x y h1 h2 h3 h4
  sorry

end solve_system_l139_139144


namespace sum_of_two_numbers_is_147_l139_139871

theorem sum_of_two_numbers_is_147 (A B : ℝ) (h1 : A + B = 147) (h2 : A = 0.375 * B + 4) :
  A + B = 147 :=
by
  sorry

end sum_of_two_numbers_is_147_l139_139871


namespace trackball_mice_count_l139_139315

theorem trackball_mice_count (total_sales wireless_share optical_share : ℕ) 
    (h_total : total_sales = 80)
    (h_wireless : wireless_share = total_sales / 2)
    (h_optical : optical_share = total_sales / 4):
    total_sales - (wireless_share + optical_share) = 20 :=
by
  sorry

end trackball_mice_count_l139_139315


namespace domain_sqrt_function_l139_139274

noncomputable def quadratic_nonneg_for_all_x (a : ℝ) : Prop :=
∀ x : ℝ, a * x^2 - a * x + 1 ≥ 0

theorem domain_sqrt_function (a : ℝ) :
  quadratic_nonneg_for_all_x a ↔ (0 ≤ a ∧ a ≤ 4) :=
by sorry

end domain_sqrt_function_l139_139274


namespace jessicas_score_l139_139303

theorem jessicas_score (average_20 : ℕ) (average_21 : ℕ) (n : ℕ) (jessica_score : ℕ) 
  (h1 : average_20 = 75)
  (h2 : average_21 = 76)
  (h3 : n = 20)
  (h4 : jessica_score = (average_21 * (n + 1)) - (average_20 * n)) :
  jessica_score = 96 :=
by 
  sorry

end jessicas_score_l139_139303


namespace sum_of_u_and_v_l139_139929

theorem sum_of_u_and_v (u v : ℤ) (h1 : 1 ≤ v) (h2 : v < u) (h3 : u^2 + v^2 = 500) : u + v = 20 := by
  sorry

end sum_of_u_and_v_l139_139929


namespace christine_amount_l139_139366

theorem christine_amount (S C : ℕ) 
  (h1 : S + C = 50)
  (h2 : C = S + 30) :
  C = 40 :=
by
  -- Proof goes here.
  -- This part should be filled in to complete the proof.
  sorry

end christine_amount_l139_139366


namespace correct_answer_l139_139277

-- Define the function 
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 2

-- Proposition p
def p : Prop := ∃ x : ℝ, (0 < x) ∧ (x < 2) ∧ f(x) < 0

-- Proposition q
def q : Prop := ∀ x y : ℝ, (x + y > 4) → (x > 2) ∧ (y > 2)

-- Correct answer based on the solution
theorem correct_answer : ¬ p ∧ ¬ q = true :=
by
  sorry

end correct_answer_l139_139277


namespace find_eccentricity_l139_139892

noncomputable def ellipse_gamma (a b : ℝ) (ha_gt : a > 0) (hb_gt : b > 0) (h : a > b) : Prop :=
∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

def ellipse_focus (a b : ℝ) : Prop :=
∀ (x y : ℝ), x = 3 → y = 0

def vertex_A (b : ℝ) : Prop :=
∀ (x y : ℝ), x = 0 → y = b

def vertex_B (b : ℝ) : Prop :=
∀ (x y : ℝ), x = 0 → y = -b

def point_N : Prop :=
∀ (x y : ℝ), x = 12 → y = 0

theorem find_eccentricity : 
∀ (a b : ℝ) (ha_gt : a > 0) (hb_gt : b > 0) (h : a > b), 
  ellipse_gamma a b ha_gt hb_gt h → 
  ellipse_focus a b → 
  vertex_A b → 
  vertex_B b → 
  point_N → 
  ∃ e : ℝ, e = 1 / 2 := 
by 
  sorry

end find_eccentricity_l139_139892


namespace pipe_r_fill_time_l139_139806

theorem pipe_r_fill_time (x : ℝ) : 
  (1 / 3 + 1 / 9 + 1 / x = 1 / 2) → 
  x = 18 :=
by 
  sorry

end pipe_r_fill_time_l139_139806


namespace number_of_cars_l139_139241

theorem number_of_cars (x : ℕ) (h : 3 * (x - 2) = 2 * x + 9) : x = 15 :=
by {
  sorry
}

end number_of_cars_l139_139241


namespace seokjin_fewer_books_l139_139925

theorem seokjin_fewer_books (init_books : ℕ) (jungkook_initial : ℕ) (seokjin_initial : ℕ) (jungkook_bought : ℕ) (seokjin_bought : ℕ) :
  jungkook_initial = init_books → seokjin_initial = init_books → jungkook_bought = 18 → seokjin_bought = 11 →
  jungkook_initial + jungkook_bought - (seokjin_initial + seokjin_bought) = 7 :=
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  sorry

end seokjin_fewer_books_l139_139925


namespace relationship_among_a_b_c_l139_139743

noncomputable def a : ℝ := 2^(1/3)
noncomputable def b : ℝ := Real.log 2 / Real.log 3
noncomputable def c : ℝ := Real.cos (100 * Real.pi / 180)

theorem relationship_among_a_b_c : a > b ∧ b > c := by
  sorry

end relationship_among_a_b_c_l139_139743


namespace segment_parallel_to_x_axis_l139_139451

theorem segment_parallel_to_x_axis 
  (f : ℤ → ℤ) 
  (hf : ∀ n, ∃ m, f n = m) 
  (a b : ℤ) 
  (h_dist : ∃ d : ℤ, d * d = (b - a) * (b - a) + (f b - f a) * (f b - f a)) : 
  f a = f b :=
sorry

end segment_parallel_to_x_axis_l139_139451


namespace smaller_octagon_area_fraction_l139_139678

theorem smaller_octagon_area_fraction (A B C D E F G H : Point) (O : Point) :
  is_regular_octagon A B C D E F G H →
  is_center O A B C D E F G H →
  let A' := midpoint A B,
      B' := midpoint B C,
      C' := midpoint C D,
      D' := midpoint D E,
      E' := midpoint E F,
      F' := midpoint F G,
      G' := midpoint G H,
      H' := midpoint H A in
  is_octa_center O A' B' C' D' E' F' G' H' →
  (area_of_octagon A B C D E F G H) * (1 / 4) = area_of_octagon A' B' C' D' E' F' G' H' :=
by
  -- Sorry, proof is omitted.
  sorry

end smaller_octagon_area_fraction_l139_139678


namespace volume_of_rectangular_solid_l139_139158

theorem volume_of_rectangular_solid : 
  let l := 100 -- length in cm
  let w := 20  -- width in cm
  let h := 50  -- height in cm
  let V := l * w * h
  V = 100000 :=
by
  rfl

end volume_of_rectangular_solid_l139_139158


namespace polynomial_root_transformation_l139_139120

noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

theorem polynomial_root_transformation :
  let P (a b c d e : ℝ) (x : ℂ) := (x^6 : ℂ) + (a : ℂ) * x^5 + (b : ℂ) * x^4 + (c : ℂ) * x^3 + (d : ℂ) * x^2 + (e : ℂ) * x + 4096
  (∀ r : ℂ, P 0 0 0 0 0 r = 0 → P 0 0 0 0 0 (ω * r) = 0) →
  ∃ a b c d e : ℝ, ∃ p : ℕ, p = 3 := sorry

end polynomial_root_transformation_l139_139120


namespace child_growth_l139_139709

-- Define variables for heights
def current_height : ℝ := 41.5
def previous_height : ℝ := 38.5

-- Define the problem statement in Lean 4
theorem child_growth :
  current_height - previous_height = 3 :=
by 
  sorry

end child_growth_l139_139709


namespace price_of_brand_Y_pen_l139_139602

theorem price_of_brand_Y_pen (cost_X : ℝ) (num_X : ℕ) (total_pens : ℕ) (total_cost : ℝ) :
  cost_X = 4 ∧ num_X = 6 ∧ total_pens = 12 ∧ total_cost = 42 →
  (∃ (price_Y : ℝ), price_Y = 3) :=
by
  sorry

end price_of_brand_Y_pen_l139_139602


namespace tiffany_max_points_l139_139535

theorem tiffany_max_points : 
  let initial_money := 3
  let cost_per_game := 1
  let points_red_bucket := 2
  let points_green_bucket := 3
  let rings_per_game := 5
  let games_played := 2
  let red_buckets_first_two_games := 4
  let green_buckets_first_two_games := 5
  let remaining_money := initial_money - games_played * cost_per_game
  let remaining_games := remaining_money / cost_per_game
  let points_first_two_games := red_buckets_first_two_games * points_red_bucket + green_buckets_first_two_games * points_green_bucket
  let max_points_third_game := rings_per_game * points_green_bucket
  points_first_two_games + max_points_third_game = 38 := 
by
  sorry

end tiffany_max_points_l139_139535


namespace intersection_of_A_B_l139_139927

variable (A : Set ℝ) (B : Set ℝ)

theorem intersection_of_A_B (hA : A = {-1, 0, 1, 2, 3}) (hB : B = {x : ℝ | 0 < x ∧ x < 3}) :
  A ∩ B = {1, 2} :=
  sorry

end intersection_of_A_B_l139_139927


namespace problem1_problem2_l139_139751

variable (k : ℝ)

-- Definitions of proposition p and q
def p (k : ℝ) : Prop := ∀ x : ℝ, x^2 - k*x + 2*k + 5 ≥ 0

def q (k : ℝ) : Prop := (4 - k > 0) ∧ (1 - k < 0)

-- Theorem statements based on the proof problem
theorem problem1 (hq : q k) : 1 < k ∧ k < 4 :=
by sorry

theorem problem2 (hp_q : p k ∨ q k) (hp_and_q_false : ¬(p k ∧ q k)) : 
  (-2 ≤ k ∧ k ≤ 1) ∨ (4 ≤ k ∧ k ≤ 10) :=
by sorry

end problem1_problem2_l139_139751


namespace husband_weekly_saving_l139_139993

variable (H : ℕ)

-- conditions
def weekly_wife : ℕ := 225
def months : ℕ := 6
def weeks_per_month : ℕ := 4
def weeks := months * weeks_per_month
def amount_per_child : ℕ := 1680
def num_children : ℕ := 4

-- total savings calculation
def total_saving : ℕ := weeks * H + weeks * weekly_wife

-- half of total savings divided among children
def half_savings_div_by_children : ℕ := num_children * amount_per_child

-- proof statement
theorem husband_weekly_saving : H = 335 :=
by
  let total_children_saving := half_savings_div_by_children
  have half_saving : ℕ := total_children_saving 
  have total_saving_eq : total_saving = 2 * total_children_saving := sorry
  have total_saving_eq_simplified : weeks * H + weeks * weekly_wife = 13440 := sorry
  have H_eq : H = 335 := sorry
  exact H_eq

end husband_weekly_saving_l139_139993


namespace total_apples_l139_139518

theorem total_apples (n : ℕ) (h1 : ∀ k : ℕ, 6 * k = n ∨ 12 * k = n) (h2 : 70 ≤ n) (h3 : n ≤ 80) : 
  n = 72 ∨ n = 78 := 
sorry

end total_apples_l139_139518


namespace washing_machines_total_pounds_l139_139712

theorem washing_machines_total_pounds (pounds_per_machine_per_day : ℕ) (number_of_machines : ℕ)
  (h1 : pounds_per_machine_per_day = 28) (h2 : number_of_machines = 8) :
  number_of_machines * pounds_per_machine_per_day = 224 :=
by
  sorry

end washing_machines_total_pounds_l139_139712


namespace total_cost_is_130_l139_139541

-- Defining the number of each type of pet
def n_puppies : ℕ := 2
def n_kittens : ℕ := 2
def n_parakeets : ℕ := 3

-- Defining the cost of one parakeet
def c_parakeet : ℕ := 10

-- Defining the cost of one puppy and one kitten based on the conditions
def c_puppy : ℕ := 3 * c_parakeet
def c_kitten : ℕ := 2 * c_parakeet

-- Defining the total cost of all pets
def total_cost : ℕ :=
  (n_puppies * c_puppy) + (n_kittens * c_kitten) + (n_parakeets * c_parakeet)

-- Lean theorem stating that the total cost is 130 dollars
theorem total_cost_is_130 : total_cost = 130 := by
  -- The proof will be filled in here.
  sorry

end total_cost_is_130_l139_139541


namespace log_base_property_l139_139882

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x / log a

theorem log_base_property
  (a : ℝ)
  (ha_pos : a > 0)
  (ha_ne_one : a ≠ 1)
  (hf9 : f a 9 = 2) :
  f a (3^a) = 3 :=
by
  sorry

end log_base_property_l139_139882


namespace opposite_113_eq_114_l139_139938

-- Define the problem setup and conditions
variable (numbers : List ℕ) (n : ℕ)
variable (h1 : numbers.length = 200)
variable (h2 : numbers.nodup)
variable (h3 : ∀ k : ℕ, k ∈ numbers ↔ k ∈ (List.range 1 201))
variable (h4 : 1 ≤ n ∧ n ≤ 200)

-- Define the function to find the opposite number in the circle
def opposite (n : ℕ) : ℕ :=
  (n + 100) % 200

-- The proof statement that needs to be proven
theorem opposite_113_eq_114 : opposite 113 = 114 :=
  sorry

end opposite_113_eq_114_l139_139938


namespace find_a3_minus_b3_l139_139760

theorem find_a3_minus_b3 (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 47) : a^3 - b^3 = 322 :=
by
  sorry

end find_a3_minus_b3_l139_139760


namespace part_I_part_II_l139_139915

noncomputable def parametric_eq_line (t : ℝ) : ℝ × ℝ :=
  (-3 / 5 * t + 2, 4 / 5 * t)

constant a : ℝ

def polar_eq_circle (θ : ℝ) : ℝ :=
  a * Real.sin θ

theorem part_I (a : ℝ) (h_a : a = 2) :
  (∀ x y : ℝ, (x^2 + (y - 1)^2 = 1)) ∧
  (∀ t : ℝ, parametric_eq_line t = (x, y) → 4 * x + 3 * y - 8 = 0) :=
  sorry

theorem part_II (h : ∀ t : ℝ, | 2 - (3 * -3 / 5 * t + 3 * 2 / 5) | / √(1 + (4 / 5)^2) = √3 * a / 2 → 5 * |a| = 2 * |3 * a - 16|) :
  a = 32 ∨ a = (32 / 11) :=
  sorry

end part_I_part_II_l139_139915


namespace geometric_sequence_sum_l139_139425

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) ^ 2 = a n * a (n + 2))
  (h_pos : ∀ n, 0 < a n) (h_given : a 1 * a 5 + 2 * a 3 * a 6 + a 1 * a 11 = 16) :
  a 3 + a 6 = 4 := 
sorry

end geometric_sequence_sum_l139_139425


namespace part_a_part_b_l139_139446

-- Definition of the function f and the condition it satisfies
variable (f : ℕ → ℕ)
variable (k n : ℕ)

theorem part_a (h1 : ∀ k n : ℕ, (k * f n) ≤ f (k * n) ∧ f (k * n) ≤ (k * f n) + k - 1)
  (a b : ℕ) :
  f a + f b ≤ f (a + b) ∧ f (a + b) ≤ f a + f b + 1 :=
by
  exact sorry  -- Proof to be supplied

theorem part_b (h1 : ∀ k n : ℕ, (k * f n) ≤ f (k * n) ∧ f (k * n) ≤ (k * f n) + k - 1)
  (h2 : ∀ n : ℕ, f (2007 * n) ≤ 2007 * f n + 200) :
  ∃ c : ℕ, f (2007 * c) = 2007 * f c :=
by
  exact sorry  -- Proof to be supplied

end part_a_part_b_l139_139446


namespace distance_between_two_girls_after_12_hours_l139_139556

theorem distance_between_two_girls_after_12_hours :
  let speed1 := 7 -- speed of the first girl (km/hr)
  let speed2 := 3 -- speed of the second girl (km/hr)
  let time := 12 -- time (hours)
  let distance1 := speed1 * time -- distance traveled by the first girl
  let distance2 := speed2 * time -- distance traveled by the second girl
  distance1 + distance2 = 120 := -- total distance
by
  -- Here, we would provide the proof, but we put sorry to skip it
  sorry

end distance_between_two_girls_after_12_hours_l139_139556


namespace cost_of_carpeting_l139_139467

noncomputable def cost_per_meter_in_paise (cost : ℝ) (length_in_meters : ℝ) : ℝ :=
  cost * 100 / length_in_meters

theorem cost_of_carpeting (room_length room_breadth carpet_width_m cost_total : ℝ) (h1 : room_length = 15) 
  (h2 : room_breadth = 6) (h3 : carpet_width_m = 0.75) (h4 : cost_total = 36) :
  cost_per_meter_in_paise cost_total (room_length * room_breadth / carpet_width_m) = 30 :=
by
  sorry

end cost_of_carpeting_l139_139467


namespace max_trains_final_count_l139_139163

-- Define the conditions
def trains_per_birthdays : Nat := 1
def trains_per_christmas : Nat := 2
def trains_per_easter : Nat := 3
def years : Nat := 7

-- Function to calculate total trains after 7 years
def total_trains_after_years (trains_per_years : Nat) (num_years : Nat) : Nat :=
  trains_per_years * num_years

-- Calculate inputs
def trains_per_year : Nat := trains_per_birthdays + trains_per_christmas + trains_per_easter
def total_initial_trains : Nat := total_trains_after_years trains_per_year years

-- Bonus and final steps
def bonus_trains_from_cousins (initial_trains : Nat) : Nat := initial_trains / 2
def final_total_trains (initial_trains : Nat) (bonus_trains : Nat) : Nat :=
  let after_bonus := initial_trains + bonus_trains
  let additional_from_parents := after_bonus * 3
  after_bonus + additional_from_parents

-- Main theorem
theorem max_trains_final_count : final_total_trains total_initial_trains (bonus_trains_from_cousins total_initial_trains) = 252 := by
  sorry

end max_trains_final_count_l139_139163


namespace larger_number_is_38_l139_139683

theorem larger_number_is_38 (x y : ℕ) (h1 : x + y = 64) (h2 : y = x + 12) : y = 38 :=
by
  sorry

end larger_number_is_38_l139_139683


namespace ratio_of_tax_revenue_to_cost_of_stimulus_l139_139067

-- Definitions based on the identified conditions
def bottom_20_percent_people (total_people : ℕ) : ℕ := (total_people * 20) / 100
def stimulus_per_person : ℕ := 2000
def total_people : ℕ := 1000
def government_profit : ℕ := 1600000

-- Cost of the stimulus
def cost_of_stimulus : ℕ := bottom_20_percent_people total_people * stimulus_per_person

-- Tax revenue returned to the government
def tax_revenue : ℕ := government_profit + cost_of_stimulus

-- The Proposition we need to prove
theorem ratio_of_tax_revenue_to_cost_of_stimulus :
  tax_revenue / cost_of_stimulus = 5 :=
by
  sorry

end ratio_of_tax_revenue_to_cost_of_stimulus_l139_139067


namespace remainder_of_product_div_10_l139_139694

theorem remainder_of_product_div_10 : 
  (3251 * 7462 * 93419) % 10 = 8 := 
sorry

end remainder_of_product_div_10_l139_139694


namespace min_value_x_plus_2y_l139_139758

theorem min_value_x_plus_2y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = x * y) :
  x + 2 * y ≥ 8 :=
sorry

end min_value_x_plus_2y_l139_139758


namespace last_digit_of_2_pow_2010_l139_139053

theorem last_digit_of_2_pow_2010 : (2 ^ 2010) % 10 = 4 :=
by
  sorry

end last_digit_of_2_pow_2010_l139_139053


namespace total_cost_of_pets_is_130_l139_139540
noncomputable theory

def cost_of_pets :=
  let parakeet_cost := 10
  let puppy_cost := 3 * parakeet_cost
  let kitten_cost := 2 * parakeet_cost
  let total_cost := 2 * puppy_cost + 2 * kitten_cost + 3 * parakeet_cost
  total_cost

theorem total_cost_of_pets_is_130 : cost_of_pets = 130 :=
by
  -- Showing that cost_of_pets indeed evaluates to 130
  let parakeet_cost := 10
  let puppy_cost := 3 * parakeet_cost
  let kitten_cost := 2 * parakeet_cost
  let total_cost := 2 * puppy_cost + 2 * kitten_cost + 3 * parakeet_cost
  show total_cost = 130 from sorry

end total_cost_of_pets_is_130_l139_139540


namespace min_PM_PN_l139_139279

noncomputable def C1 (x y : ℝ) : Prop := (x + 6)^2 + (y - 5)^2 = 4
noncomputable def C2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

theorem min_PM_PN : ∀ (P M N : ℝ × ℝ),
  P.2 = 0 ∧ C1 M.1 M.2 ∧ C2 N.1 N.2 → (|P.1 - M.1| + (P.1 - N.1)^2 + (P.2 - N.2)^2).sqrt = 7 := by
  sorry

end min_PM_PN_l139_139279


namespace find_nat_numbers_l139_139377

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (· + ·) 0

theorem find_nat_numbers (n : ℕ) :
  (n + sum_of_digits n = 2021) ↔ (n = 2014 ∨ n = 1996) :=
by
  sorry

end find_nat_numbers_l139_139377


namespace quadratic_has_real_roots_l139_139002

-- Define the condition that a quadratic equation has real roots given ac < 0

variable {a b c : ℝ}

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_has_real_roots (h : a * c < 0) : ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by
  sorry

end quadratic_has_real_roots_l139_139002


namespace eggs_left_on_shelf_l139_139447

-- Define the conditions as variables in the Lean statement
variables (x y z : ℝ)

-- Define the final theorem statement
theorem eggs_left_on_shelf (hx : 0 ≤ x) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) :
  x * (1 - y) - z = (x - y * x) - z :=
by
  sorry

end eggs_left_on_shelf_l139_139447


namespace total_cost_correct_l139_139831

noncomputable def cost_4_canvases : ℕ := 40
noncomputable def cost_paints : ℕ := cost_4_canvases / 2
noncomputable def cost_easel : ℕ := 15
noncomputable def cost_paintbrushes : ℕ := 15
noncomputable def total_cost : ℕ := cost_4_canvases + cost_paints + cost_easel + cost_paintbrushes

theorem total_cost_correct : total_cost = 90 :=
by
  unfold total_cost
  unfold cost_4_canvases
  unfold cost_paints
  unfold cost_easel
  unfold cost_paintbrushes
  simp
  sorry

end total_cost_correct_l139_139831


namespace find_t_l139_139763

theorem find_t (t : ℝ) : 
  (∃ a b : ℝ, a^2 = t^2 ∧ b^2 = 5 * t ∧ (a - b = 2 * Real.sqrt 6 ∨ b - a = 2 * Real.sqrt 6)) → 
  (t = 2 ∨ t = 3 ∨ t = 6) := 
by
  sorry

end find_t_l139_139763


namespace ordinate_of_point_A_l139_139322

noncomputable def p : ℝ := 1 / 4
noncomputable def distance_to_focus (y₀ : ℝ) : ℝ := y₀ + p / 2

theorem ordinate_of_point_A :
  ∃ y₀ : ℝ, (distance_to_focus y₀ = 9 / 8) → y₀ = 1 :=
by
  -- Assume solution steps here
  sorry

end ordinate_of_point_A_l139_139322


namespace solve_abs_eq_l139_139214

theorem solve_abs_eq (x : ℝ) (h : |x - 3| = |x + 1|) : x = 1 :=
  sorry

end solve_abs_eq_l139_139214


namespace numbers_of_form_xy9z_div_by_132_l139_139729

theorem numbers_of_form_xy9z_div_by_132 (x y z : ℕ) :
  let N := 1000 * x + 100 * y + 90 + z
  (N % 4 = 0) ∧ ((x + y + 9 + z) % 3 = 0) ∧ ((x + 9 - y - z) % 11 = 0) ↔ 
  (N = 3696) ∨ (N = 4092) ∨ (N = 6996) ∨ (N = 7392) :=
by
  intros
  let N := 1000 * x + 100 * y + 90 + z
  sorry

end numbers_of_form_xy9z_div_by_132_l139_139729


namespace range_of_p_add_q_l139_139275

theorem range_of_p_add_q (p q : ℝ) :
  (∀ x : ℝ, ¬(x^2 + 2 * p * x - (q^2 - 2) = 0)) → 
  (p + q) ∈ Set.Ioo (-2 : ℝ) (2 : ℝ) :=
by
  intro h
  sorry

end range_of_p_add_q_l139_139275


namespace parabola_intercepts_l139_139180

noncomputable def question (y : ℝ) := 3 * y ^ 2 - 9 * y + 4

theorem parabola_intercepts (a b c : ℝ) (h_a : a = question 0) (h_b : 3 * b ^ 2 - 9 * b + 4 = 0) (h_c : 3 * c ^ 2 - 9 * c + 4 = 0) :
  a + b + c = 7 :=
by
  sorry

end parabola_intercepts_l139_139180


namespace birch_tree_taller_than_pine_tree_l139_139717

theorem birch_tree_taller_than_pine_tree :
  let pine_tree_height := (49 : ℚ) / 4
  let birch_tree_height := (37 : ℚ) / 2
  birch_tree_height - pine_tree_height = 25 / 4 :=
by
  sorry

end birch_tree_taller_than_pine_tree_l139_139717


namespace share_difference_l139_139715

theorem share_difference (p q r : ℕ) (x : ℕ) (h_ratio : p = 3 * x ∧ q = 7 * x ∧ r = 12 * x)
  (h_diff_qr : q - r = 5500) : q - p = 4400 :=
by
  sorry

end share_difference_l139_139715


namespace inequality_solution_l139_139190

theorem inequality_solution (x : ℝ) : 
  (x + 1) * (2 - x) < 0 ↔ x < -1 ∨ x > 2 := 
sorry

end inequality_solution_l139_139190


namespace find_h_of_root_l139_139283

theorem find_h_of_root :
  ∀ h : ℝ, (-3)^3 + h * (-3) - 10 = 0 → h = -37/3 := by
  sorry

end find_h_of_root_l139_139283


namespace cosine_greater_sine_cosine_cos_greater_sine_sin_l139_139223

variable {f g : ℝ → ℝ}

-- Problem 1
theorem cosine_greater_sine (h : ∀ x, - (Real.pi / 2) < f x + g x ∧ f x + g x < Real.pi / 2
                            ∧ - (Real.pi / 2) < f x - g x ∧ f x - g x < Real.pi / 2) :
  ∀ x, Real.cos (f x) > Real.sin (g x) :=
sorry

-- Problem 2
theorem cosine_cos_greater_sine_sin (x : ℝ) :  Real.cos (Real.cos x) > Real.sin (Real.sin x) :=
sorry

end cosine_greater_sine_cosine_cos_greater_sine_sin_l139_139223


namespace symmetry_center_of_tangent_l139_139952

noncomputable def tangentFunction (x : ℝ) : ℝ := Real.tan (2 * x - (Real.pi / 3))

theorem symmetry_center_of_tangent :
  (∃ k : ℤ, (Real.pi / 6) + (k * Real.pi / 4) = 5 * Real.pi / 12 ∧ tangentFunction ((5 * Real.pi) / 12) = 0 ) :=
sorry

end symmetry_center_of_tangent_l139_139952


namespace income_final_amount_l139_139357

noncomputable def final_amount (income : ℕ) : ℕ :=
  let children_distribution := (income * 45) / 100
  let wife_deposit := (income * 30) / 100
  let remaining_after_distribution := income - children_distribution - wife_deposit
  let donation := (remaining_after_distribution * 5) / 100
  remaining_after_distribution - donation

theorem income_final_amount : final_amount 200000 = 47500 := by
  -- Proof omitted
  sorry

end income_final_amount_l139_139357


namespace salesman_bonus_l139_139220

theorem salesman_bonus (S B : ℝ) 
  (h1 : S > 10000) 
  (h2 : 0.09 * S + 0.03 * (S - 10000) = 1380) 
  : B = 0.03 * (S - 10000) :=
sorry

end salesman_bonus_l139_139220


namespace savings_correct_l139_139068

noncomputable def savings (income expenditure : ℕ) : ℕ :=
income - expenditure

theorem savings_correct (I E : ℕ) (h_ratio :  I / E = 10 / 4) (h_income : I = 19000) :
  savings I E = 11400 :=
sorry

end savings_correct_l139_139068


namespace perpendicular_line_l139_139608

theorem perpendicular_line (x y : ℝ) (h : 2 * x + y - 10 = 0) : 
    (∃ k : ℝ, (x = 1 ∧ y = 2) → (k * (-2) = -1)) → 
    (∃ m b : ℝ, b = 3 ∧ m = 1/2) → 
    (x - 2 * y + 3 = 0) := 
sorry

end perpendicular_line_l139_139608


namespace apple_count_l139_139485

/--
Given the total number of apples in several bags, where each bag contains either 12 or 6 apples,
and knowing that the total number of apples is between 70 and 80 inclusive,
prove that the total number of apples can only be 72 or 78.
-/
theorem apple_count (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : ∃ k : ℕ, n = 6 * k) : n = 72 ∨ n = 78 :=
by {
  sorry
}

end apple_count_l139_139485


namespace number_of_squares_is_five_l139_139592

-- A function that computes the number of squares obtained after the described operations on a piece of paper.
def folded_and_cut_number_of_squares (initial_shape : Type) (folds : ℕ) (cuts : ℕ) : ℕ :=
  -- sorry is used here as a placeholder for the actual implementation
  sorry

-- The main theorem stating that after two folds and two cuts, we obtain five square pieces.
theorem number_of_squares_is_five (initial_shape : Type) (h_initial_square : initial_shape = square)
  (h_folds : folds = 2) (h_cuts : cuts = 2) : folded_and_cut_number_of_squares initial_shape folds cuts = 5 :=
  sorry

end number_of_squares_is_five_l139_139592


namespace complement_A_in_U_l139_139624

def U := {x : ℝ | -4 < x ∧ x < 4}
def A := {x : ℝ | -3 ≤ x ∧ x < 2}

theorem complement_A_in_U :
  {x : ℝ | x ∈ U ∧ x ∉ A} = {x : ℝ | (-4 < x ∧ x < -3) ∨ (2 ≤ x ∧ x < 4)} :=
by {
  sorry
}

end complement_A_in_U_l139_139624


namespace correct_propositions_l139_139117

structure Proposition :=
  (statement : String)
  (is_correct : Prop)

def prop1 : Proposition := {
  statement := "All sufficiently small positive numbers form a set.",
  is_correct := False -- From step b
}

def prop2 : Proposition := {
  statement := "The set containing 1, 2, 3, 1, 9 is represented by enumeration as {1, 2, 3, 1, 9}.",
  is_correct := False -- From step b
}

def prop3 : Proposition := {
  statement := "{1, 3, 5, 7} and {7, 5, 3, 1} denote the same set.",
  is_correct := True -- From step b
}

def prop4 : Proposition := {
  statement := "{y = -x} represents the collection of all points on the graph of the function y = -x.",
  is_correct := False -- From step b
}

theorem correct_propositions :
  prop3.is_correct ∧ ¬prop1.is_correct ∧ ¬prop2.is_correct ∧ ¬prop4.is_correct :=
by
  -- Here we put the proof steps, but for the exercise's purpose, we use sorry.
  sorry

end correct_propositions_l139_139117


namespace exponent_calculation_l139_139860

theorem exponent_calculation : 10^6 * (10^2)^3 / 10^4 = 10^8 := by
  sorry

end exponent_calculation_l139_139860


namespace find_f_2021_l139_139818

variable (f : ℝ → ℝ)

axiom functional_equation : ∀ a b : ℝ, f ( (a + 2 * b) / 3) = (f a + 2 * f b) / 3
axiom f_one : f 1 = 1
axiom f_four : f 4 = 7

theorem find_f_2021 : f 2021 = 4041 := by
  sorry

end find_f_2021_l139_139818


namespace final_concentration_after_procedure_l139_139570

open Real

def initial_salt_concentration : ℝ := 0.16
def final_salt_concentration : ℝ := 0.107

def volume_ratio_large : ℝ := 10
def volume_ratio_medium : ℝ := 4
def volume_ratio_small : ℝ := 3

def overflow_due_to_small_ball : ℝ := 0.1

theorem final_concentration_after_procedure :
  (initial_salt_concentration * (overflow_due_to_small_ball)) * volume_ratio_small / (volume_ratio_large + volume_ratio_medium + volume_ratio_small) =
  final_salt_concentration :=
sorry

end final_concentration_after_procedure_l139_139570


namespace positive_partial_sum_existence_l139_139613

variable {n : ℕ}
variable {a : Fin n → ℝ}

theorem positive_partial_sum_existence (h : (Finset.univ.sum a) > 0) :
  ∃ i : Fin n, ∀ j : Fin n, i ≤ j → (Finset.Icc i j).sum a > 0 := by
  sorry

end positive_partial_sum_existence_l139_139613


namespace trevor_spending_proof_l139_139832

def trevor_spends (T R Q : ℕ) : Prop :=
  T = R + 20 ∧ R = 2 * Q ∧ 4 * T + 4 * R + 2 * Q = 680

theorem trevor_spending_proof (T R Q : ℕ) (h : trevor_spends T R Q) : T = 80 :=
by sorry

end trevor_spending_proof_l139_139832


namespace trackball_mice_count_l139_139311

theorem trackball_mice_count 
  (total_mice wireless_mice optical_mice trackball_mice : ℕ)
  (h1 : total_mice = 80)
  (h2 : wireless_mice = total_mice / 2)
  (h3 : optical_mice = total_mice / 4)
  (h4 : trackball_mice = total_mice - (wireless_mice + optical_mice)) :
  trackball_mice = 20 := by 
  sorry

end trackball_mice_count_l139_139311


namespace bigger_wheel_roll_distance_l139_139917

/-- The circumference of the bigger wheel is 12 meters -/
def bigger_wheel_circumference : ℕ := 12

/-- The circumference of the smaller wheel is 8 meters -/
def smaller_wheel_circumference : ℕ := 8

/-- The distance the bigger wheel must roll for the points P1 and P2 to coincide again -/
theorem bigger_wheel_roll_distance : Nat.lcm bigger_wheel_circumference smaller_wheel_circumference = 24 :=
by
  -- Proof is omitted
  sorry

end bigger_wheel_roll_distance_l139_139917


namespace option_d_not_equal_four_thirds_l139_139551

theorem option_d_not_equal_four_thirds :
  1 + (2 / 7) ≠ 4 / 3 :=
by
  sorry

end option_d_not_equal_four_thirds_l139_139551


namespace tiffany_max_points_l139_139537

section
  variables
  (initial_money : ℕ := 3)
  (cost_per_game : ℕ := 1)
  (rings_per_game : ℕ := 5)
  (points_red_bucket : ℕ := 2)
  (points_green_bucket : ℕ := 3)
  (points_miss : ℕ := 0)
  (games_played : ℕ := 2)
  (red_buckets : ℕ := 4)
  (green_buckets : ℕ := 5)
  (additional_games : ℕ := initial_money - games_played)
  (points_per_game_from_green_buckets : ℕ := rings_per_game * points_green_bucket)
  (total_points : ℕ := (red_buckets * points_red_bucket) + (green_buckets * points_green_bucket) + (additional_games * points_per_game_from_green_buckets))

  theorem tiffany_max_points : total_points = 38 := 
  sorry
end

end tiffany_max_points_l139_139537


namespace approx_num_chars_in_ten_thousand_units_l139_139347

-- Define the number of characters in the book
def num_chars : ℕ := 731017

-- Define the conversion factor from characters to units of 'ten thousand'
def ten_thousand : ℕ := 10000

-- Define the number of characters in units of 'ten thousand'
def chars_in_ten_thousand_units : ℚ := num_chars / ten_thousand

-- Define the rounded number of units to the nearest whole number
def rounded_chars_in_ten_thousand_units : ℤ := round chars_in_ten_thousand_units

-- Theorem to state the approximate number of characters in units of 'ten thousand' is 73
theorem approx_num_chars_in_ten_thousand_units : rounded_chars_in_ten_thousand_units = 73 := 
by sorry

end approx_num_chars_in_ten_thousand_units_l139_139347


namespace parallel_vectors_l139_139899

variable (y : ℝ)

def vector_a : ℝ × ℝ := (-1, 3)
def vector_b (y : ℝ) : ℝ × ℝ := (2, y)

theorem parallel_vectors (h : (-1 * y - 3 * 2) = 0) : y = -6 :=
by
  sorry

end parallel_vectors_l139_139899


namespace sum_of_reciprocal_squares_leq_reciprocal_product_square_l139_139192

theorem sum_of_reciprocal_squares_leq_reciprocal_product_square (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_sum : a + b + c + d = 3) : 
  1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 ≤ 1 / (a^2 * b^2 * c^2 * d^2) :=
sorry

end sum_of_reciprocal_squares_leq_reciprocal_product_square_l139_139192


namespace total_depreciation_correct_residual_value_correct_sales_price_correct_l139_139558

-- Definitions and conditions
def initial_cost := 500000
def max_capacity := 100000
def jul_bottles := 200
def aug_bottles := 15000
def sep_bottles := 12300

def depreciation_per_bottle := initial_cost / max_capacity

-- Part (a)
def total_depreciation_jul := jul_bottles * depreciation_per_bottle
def total_depreciation_aug := aug_bottles * depreciation_per_bottle
def total_depreciation_sep := sep_bottles * depreciation_per_bottle
def total_depreciation := total_depreciation_jul + total_depreciation_aug + total_depreciation_sep

theorem total_depreciation_correct :
  total_depreciation = 137500 := 
by sorry

-- Part (b)
def residual_value := initial_cost - total_depreciation

theorem residual_value_correct :
  residual_value = 362500 := 
by sorry

-- Part (c)
def desired_profit := 10000
def sales_price := residual_value + desired_profit

theorem sales_price_correct :
  sales_price = 372500 := 
by sorry

end total_depreciation_correct_residual_value_correct_sales_price_correct_l139_139558


namespace kenya_peanuts_count_l139_139782

def peanuts_jose : ℕ := 85
def diff_kenya_jose : ℕ := 48
def peanuts_kenya : ℕ := peanuts_jose + diff_kenya_jose

theorem kenya_peanuts_count : peanuts_kenya = 133 := 
by
  -- proof goes here
  sorry

end kenya_peanuts_count_l139_139782


namespace determine_set_A_l139_139370

-- Define the function f as described
def f (n : ℕ) (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else (x - 1) / 2 + 2^(n - 1)

-- Define the set A
def A (n : ℕ) : Set ℕ :=
  { x | (Nat.iterate (f n) n x) = x }

-- State the theorem
theorem determine_set_A (n : ℕ) (hn : n > 0) :
    A n = { x | 1 ≤ x ∧ x ≤ 2^n } :=
sorry

end determine_set_A_l139_139370


namespace cars_count_l139_139244

-- Define the number of cars as x
variable (x : ℕ)

-- The conditions for the problem
def condition1 := 3 * (x - 2)
def condition2 := 2 * x + 9

-- The main theorem stating that under the given conditions, x = 15
theorem cars_count : condition1 x = condition2 x → x = 15 := by
  sorry

end cars_count_l139_139244


namespace total_legos_156_l139_139113

def pyramid_bottom_legos (side_length : Nat) : Nat := side_length * side_length
def pyramid_second_level_legos (length : Nat) (width : Nat) : Nat := length * width
def pyramid_third_level_legos (side_length : Nat) : Nat :=
  let total_legos := (side_length * (side_length + 1)) / 2
  total_legos - 3  -- Subtracting 3 Legos for the corners

def pyramid_fourth_level_legos : Nat := 1

def total_pyramid_legos : Nat :=
  pyramid_bottom_legos 10 +
  pyramid_second_level_legos 8 6 +
  pyramid_third_level_legos 4 +
  pyramid_fourth_level_legos

theorem total_legos_156 : total_pyramid_legos = 156 := by
  sorry

end total_legos_156_l139_139113


namespace perimeter_of_triangle_l139_139416

namespace TrianglePerimeter

variables {a b c : ℝ}

-- Conditions translated into definitions
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def absolute_sum_condition (a b c : ℝ) : Prop :=
  |a + b - c| + |b + c - a| + |c + a - b| = 12

-- The theorem stating the perimeter under given conditions
theorem perimeter_of_triangle (h : is_valid_triangle a b c) (h_abs_sum : absolute_sum_condition a b c) : 
  a + b + c = 12 := 
sorry

end TrianglePerimeter

end perimeter_of_triangle_l139_139416


namespace citric_acid_molecular_weight_l139_139835

def molecular_weight_citric_acid := 192.12 -- in g/mol

theorem citric_acid_molecular_weight :
  molecular_weight_citric_acid = 192.12 :=
by sorry

end citric_acid_molecular_weight_l139_139835


namespace apple_bags_l139_139495

theorem apple_bags (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apple_bags_l139_139495


namespace max_value_of_expression_l139_139155

theorem max_value_of_expression (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_product : a * b * c = 16) : 
  a^b - b^c + c^a ≤ 263 :=
sorry

end max_value_of_expression_l139_139155


namespace least_x_l139_139842

theorem least_x (x p : ℕ) (h1 : 0 < x) (h2: Nat.Prime p) (h3: ∃ q : ℕ, Nat.Prime q ∧ q % 2 = 1 ∧ x = 11 * p * q) : x ≥ 66 := 
sorry

end least_x_l139_139842


namespace difference_of_squares_65_35_l139_139116

theorem difference_of_squares_65_35 :
  let a := 65
  let b := 35
  a^2 - b^2 = (a + b) * (a - b) :=
by
  sorry

end difference_of_squares_65_35_l139_139116


namespace unit_place_3_pow_34_l139_139557

theorem unit_place_3_pow_34 : Nat.mod (3^34) 10 = 9 :=
by
  sorry

end unit_place_3_pow_34_l139_139557


namespace coefficient_x_pow_7_l139_139544

theorem coefficient_x_pow_7 :
  ∃ c : Int, (∀ k : Nat, (x : Real) → coeff (expand_binom (x-2) 10) k = c → k = 7 ∧ c = -960) :=
by
  sorry

end coefficient_x_pow_7_l139_139544


namespace ordered_pair_proportional_l139_139682

theorem ordered_pair_proportional (p q : ℝ) (h : (3 : ℝ) • (-4 : ℝ) = (5 : ℝ) • p ∧ (3 : ℝ) • q = (5 : ℝ) • (-4 : ℝ)) :
  (p, q) = (5 / 2, -8) :=
by
  sorry

end ordered_pair_proportional_l139_139682


namespace base9_addition_l139_139109

-- Define the numbers in base 9
def num1 : ℕ := 1 * 9^2 + 7 * 9^1 + 5 * 9^0
def num2 : ℕ := 7 * 9^2 + 1 * 9^1 + 4 * 9^0
def num3 : ℕ := 6 * 9^1 + 1 * 9^0
def result : ℕ := 1 * 9^3 + 0 * 9^2 + 6 * 9^1 + 1 * 9^0

-- State the theorem
theorem base9_addition : num1 + num2 + num3 = result := by
  sorry

end base9_addition_l139_139109


namespace Liz_needs_more_money_l139_139448

theorem Liz_needs_more_money (P : ℝ) (h1 : P = 30000 + 2500) (h2 : 0.80 * P = 26000) : 30000 - (0.80 * P) = 4000 :=
by
  sorry

end Liz_needs_more_money_l139_139448


namespace find_radius_of_stationary_tank_l139_139100

theorem find_radius_of_stationary_tank
  (h_stationary : Real) (r_truck : Real) (h_truck : Real) (drop : Real) (V_truck : Real)
  (ht1 : h_stationary = 25)
  (ht2 : r_truck = 4)
  (ht3 : h_truck = 10)
  (ht4 : drop = 0.016)
  (ht5 : V_truck = π * r_truck ^ 2 * h_truck) :
  ∃ R : Real, π * R ^ 2 * drop = V_truck ∧ R = 100 :=
by
  sorry

end find_radius_of_stationary_tank_l139_139100


namespace Bhupathi_amount_l139_139583

variable (A B : ℝ)

theorem Bhupathi_amount
  (h1 : A + B = 1210)
  (h2 : (4 / 15) * A = (2 / 5) * B) :
  B = 484 := by
  sorry

end Bhupathi_amount_l139_139583


namespace common_root_sum_k_l139_139085

theorem common_root_sum_k :
  (∃ x : ℝ, (x^2 - 4 * x + 3 = 0) ∧ (x^2 - 6 * x + k = 0)) → 
  (∃ (k₁ k₂ : ℝ), (k₁ = 5) ∧ (k₂ = 9) ∧ (k₁ + k₂ = 14)) :=
by
  sorry

end common_root_sum_k_l139_139085


namespace jump_rope_difference_l139_139594

noncomputable def cindy_jump_time : ℕ := 12
noncomputable def betsy_jump_time : ℕ := cindy_jump_time / 2
noncomputable def tina_jump_time : ℕ := 3 * betsy_jump_time

theorem jump_rope_difference : tina_jump_time - cindy_jump_time = 6 :=
by
  -- proof steps would go here
  sorry

end jump_rope_difference_l139_139594


namespace buzz_waiter_ratio_l139_139364

def total_slices : Nat := 78
def waiter_condition (W : Nat) : Prop := W - 20 = 28

theorem buzz_waiter_ratio (W : Nat) (h : waiter_condition W) : 
  let buzz_slices := total_slices - W
  let ratio_buzz_waiter := buzz_slices / W
  ratio_buzz_waiter = 5 / 8 :=
by
  sorry

end buzz_waiter_ratio_l139_139364


namespace total_cloth_sold_l139_139362

variable (commissionA commissionB salesA salesB totalWorth : ℝ)

def agentA_commission := 0.025 * salesA
def agentB_commission := 0.03 * salesB
def total_worth_of_cloth_sold := salesA + salesB

theorem total_cloth_sold 
  (hA : agentA_commission = 21) 
  (hB : agentB_commission = 27)
  : total_worth_of_cloth_sold = 1740 :=
by
  sorry

end total_cloth_sold_l139_139362


namespace problem_inequality_l139_139033

theorem problem_inequality (a : ℝ) (h_pos : 0 < a) : 
  ∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → a^(Real.sin x) * (a + 1)^(Real.cos x) ≥ a :=
by 
  sorry

end problem_inequality_l139_139033


namespace bananas_left_l139_139603

theorem bananas_left (original_bananas : ℕ) (bananas_eaten : ℕ) 
  (h1 : original_bananas = 12) (h2 : bananas_eaten = 4) : 
  original_bananas - bananas_eaten = 8 := 
by
  sorry

end bananas_left_l139_139603


namespace jeremy_total_earnings_l139_139434

theorem jeremy_total_earnings :
  let steven_rate : ℚ := 12 / 3
  let mark_rate : ℚ := 10 / 4
  let steven_rooms : ℚ := 8 / 3
  let mark_rooms : ℚ := 9 / 4
  let steven_payment : ℚ := steven_rate * steven_rooms
  let mark_payment : ℚ := mark_rate * mark_rooms
  steven_payment + mark_payment = 391 / 24 :=
by
  let steven_rate : ℚ := 12 / 3
  let mark_rate : ℚ := 10 / 4
  let steven_rooms : ℚ := 8 / 3
  let mark_rooms : ℚ := 9 / 4
  let steven_payment : ℚ := steven_rate * steven_rooms
  let mark_payment : ℚ := mark_rate * mark_rooms
  sorry

end jeremy_total_earnings_l139_139434


namespace num_ways_athletes_seated_together_l139_139637

-- Conditions from part a)
def team_A := 2
def team_B := 2
def team_C := 2

-- Objective: total number of ways to arrange six athletes given conditions
theorem num_ways_athletes_seated_together : 
    (fact 3) * (fact team_A) * (fact team_B) * (fact team_C) = 48 := by
  sorry

end num_ways_athletes_seated_together_l139_139637


namespace det_matrix_A_l139_139867

noncomputable def matrix_A (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x, y, z], ![z, x, y], ![y, z, x]]

theorem det_matrix_A (x y z : ℝ) : 
  Matrix.det (matrix_A x y z) = x^3 + y^3 + z^3 - 3*x*y*z := by
  sorry

end det_matrix_A_l139_139867


namespace marly_needs_3_bags_l139_139793

-- Define the conditions and variables
variables (milk chicken_stock vegetables total_volume bag_capacity bags_needed : ℕ)

-- Given conditions from the problem
def condition1 : milk = 2 := rfl
def condition2 : chicken_stock = 3 * milk := by rw [condition1]; norm_num
def condition3 : vegetables = 1 := rfl
def condition4 : total_volume = milk + chicken_stock + vegetables := 
  by rw [condition1, condition2, condition3]; norm_num
def condition5 : bag_capacity = 3 := rfl

-- The statement to be proved
theorem marly_needs_3_bags (h_conditions : total_volume = 9 ∧ bag_capacity = 3) : bags_needed = 3 :=
  by sorry

end marly_needs_3_bags_l139_139793


namespace sqrt_sum_ineq_l139_139035

open Real

theorem sqrt_sum_ineq (a b c d : ℝ) (h : a ≥ 0) (h1 : b ≥ 0) (h2 : c ≥ 0) (h3 : d ≥ 0)
  (h4 : a + b + c + d = 4) : 
  sqrt (a + b + c) + sqrt (b + c + d) + sqrt (c + d + a) + sqrt (d + a + b) ≥ 6 :=
sorry

end sqrt_sum_ineq_l139_139035


namespace original_number_l139_139295

-- Define the three-digit number and its permutations under certain conditions.
-- Prove the original number given the specific conditions stated.
theorem original_number (a b c : ℕ)
  (ha : a % 2 = 1) -- a being odd
  (m : ℕ := 100 * a + 10 * b + c)
  (sum_permutations : 100*a + 10*b + c + 100*a + 10*c + b + 100*b + 10*c + a + 
                      100*c + 10*a + b + 100*b + 10*a + c + 100*c + 10*b + a = 3300) :
  m = 192 := 
sorry

end original_number_l139_139295


namespace total_apples_l139_139517

theorem total_apples (n : ℕ) (h1 : ∀ k : ℕ, 6 * k = n ∨ 12 * k = n) (h2 : 70 ≤ n) (h3 : n ≤ 80) : 
  n = 72 ∨ n = 78 := 
sorry

end total_apples_l139_139517


namespace product_of_geometric_progressions_is_geometric_general_function_form_geometric_l139_139696

variables {α β γ : Type*} [CommSemiring α] [CommSemiring β] [CommSemiring γ]

-- Define the terms of geometric progressions
def term (a r : α) (k : ℕ) : α := a * r ^ (k - 1)

-- Define a general function with respective powers
def general_term (a r : α) (k p : ℕ) : α := a ^ p * (r ^ p) ^ (k - 1)

theorem product_of_geometric_progressions_is_geometric
  {a b c : α} {r1 r2 r3 : α} (k : ℕ) :
  term a r1 k * term b r2 k * term c r3 k = 
  (a * b * c) * (r1 * r2 * r3) ^ (k - 1) := 
sorry

theorem general_function_form_geometric
  {a b c : α} {r1 r2 r3 : α} {p q r : ℕ} (k : ℕ) :
  general_term a r1 k p * general_term b r2 k q * general_term c r3 k r = 
  (a^p * b^q * c^r) * (r1^p * r2^q * r3^r) ^ (k - 1) := 
sorry

end product_of_geometric_progressions_is_geometric_general_function_form_geometric_l139_139696


namespace rectangle_breadth_l139_139672

/-- The breadth of the rectangle is 10 units given that
1. The length of the rectangle is two-fifths of the radius of a circle.
2. The radius of the circle is equal to the side of the square.
3. The area of the square is 1225 sq. units.
4. The area of the rectangle is 140 sq. units. -/
theorem rectangle_breadth (r l b : ℝ) (h_radius : r = 35) (h_length : l = (2 / 5) * r) (h_square : 35 * 35 = 1225) (h_area_rect : l * b = 140) : b = 10 :=
by
  sorry

end rectangle_breadth_l139_139672


namespace prob_both_black_prob_both_white_prob_at_most_one_black_l139_139297

-- Define probabilities for balls in Bag A and Bag B
def bagA_black_prob : ℝ := 1 / 2
def bagA_white_prob : ℝ := 1 / 2
def bagB_black_prob : ℝ := 2 / 3
def bagB_white_prob : ℝ := 1 / 3

-- Proposition for the probability that both balls are black
theorem prob_both_black :
  bagA_black_prob * bagB_black_prob = 1 / 3 := sorry

-- Proposition for the probability that both balls are white
theorem prob_both_white :
  bagA_white_prob * bagB_white_prob = 1 / 6 := sorry

-- Proposition for the probability that at most one ball is black
theorem prob_at_most_one_black :
  (bagA_white_prob * bagB_white_prob) +
  (bagA_black_prob * bagB_white_prob) +
  (bagA_white_prob * bagB_black_prob) = 2 / 3 := sorry

end prob_both_black_prob_both_white_prob_at_most_one_black_l139_139297


namespace find_stream_speed_l139_139351

-- Define the conditions
def boat_speed_in_still_water : ℝ := 15
def downstream_time : ℝ := 1
def upstream_time : ℝ := 1.5
def speed_of_stream (v : ℝ) : Prop :=
  let downstream_speed := boat_speed_in_still_water + v
  let upstream_speed := boat_speed_in_still_water - v
  (downstream_speed * downstream_time) = (upstream_speed * upstream_time)

-- Define the theorem to prove
theorem find_stream_speed : ∃ v, speed_of_stream v ∧ v = 3 :=
by {
  sorry
}

end find_stream_speed_l139_139351


namespace problem_statement_l139_139403

theorem problem_statement (m : ℝ) : 
  (¬ (∃ x : ℝ, x^2 + 2 * x + m ≤ 0)) → m > 1 :=
by
  sorry

end problem_statement_l139_139403


namespace power_sum_l139_139604

theorem power_sum : 1^234 + 4^6 / 4^4 = 17 :=
by
  sorry

end power_sum_l139_139604


namespace final_value_A_is_5_l139_139778

/-
Problem: Given a 3x3 grid of numbers and a series of operations that add or subtract 1 to two adjacent cells simultaneously, prove that the number in position A in the table on the right is 5.
Conditions:
1. The initial grid is:
   \[
   \begin{array}{ccc}
   a & b & c \\
   d & e & f \\
   g & h & i \\
   \end{array}
   \]
2. Each operation involves adding or subtracting 1 from two adjacent cells.
3. The sum of all numbers in the grid remains unchanged.
-/

def table_operations (a b c d e f g h i : ℤ) : ℤ :=
-- A is determined based on the given problem and conditions
  5

theorem final_value_A_is_5 (a b c d e f g h i : ℤ) : 
  table_operations a b c d e f g h i = 5 :=
sorry

end final_value_A_is_5_l139_139778


namespace find_middle_number_l139_139333

theorem find_middle_number (a b c : ℕ) (h1 : a + b = 16) (h2 : a + c = 21) (h3 : b + c = 27) : b = 11 := by
  sorry

end find_middle_number_l139_139333


namespace quadratic_completion_l139_139597

theorem quadratic_completion (a b : ℤ) (h_eq : (x : ℝ) → x^2 - 10 * x + 25 = 0) :
  (∃ a b : ℤ, ∀ x : ℝ, (x + a) ^ 2 = b) → a + b = -5 := by
  sorry

end quadratic_completion_l139_139597


namespace apple_count_l139_139486

-- Define constants for the problem
constant bags: ℕ → Prop
constant total_apples: ℕ
constant apples_in_bag1: ℕ := 12
constant apples_in_bag2: ℕ := 6

-- Define the conditions
axiom condition1 (n: ℕ) : bags n → (n = apples_in_bag1 ∨ n = apples_in_bag2)
axiom condition2 : 70 ≤ total_apples ∧ total_apples ≤ 80
axiom condition3 : total_apples % 6 = 0

-- State the proof problem
theorem apple_count :
  total_apples = 72 ∨ total_apples = 78 :=
sorry

end apple_count_l139_139486


namespace Jenny_wants_to_read_three_books_l139_139644

noncomputable def books : Nat := 3

-- Definitions based on provided conditions
def reading_speed : Nat := 100 -- words per hour
def book1_words : Nat := 200 
def book2_words : Nat := 400
def book3_words : Nat := 300
def daily_reading_minutes : Nat := 54 
def days : Nat := 10

-- Derived definitions for the proof
def total_words : Nat := book1_words + book2_words + book3_words
def total_hours_needed : ℚ := total_words / reading_speed
def daily_reading_hours : ℚ := daily_reading_minutes / 60
def total_reading_hours : ℚ := daily_reading_hours * days

theorem Jenny_wants_to_read_three_books :
  total_reading_hours = total_hours_needed → books = 3 :=
by
  -- Proof goes here
  sorry

end Jenny_wants_to_read_three_books_l139_139644


namespace min_sum_of_m_n_l139_139385

theorem min_sum_of_m_n (m n : ℕ) (h1 : m ≥ 1) (h2 : n ≥ 3) (h3 : 8 ∣ (180 * m * n - 360 * m)) : m + n = 5 :=
sorry

end min_sum_of_m_n_l139_139385


namespace total_marks_l139_139062

variable (E S M : Nat)

-- Given conditions
def thrice_as_many_marks_in_English_as_in_Science := E = 3 * S
def ratio_of_marks_in_English_and_Maths            := M = 4 * E
def marks_in_Science                               := S = 17

-- Proof problem statement
theorem total_marks (h1 : E = 3 * S) (h2 : M = 4 * E) (h3 : S = 17) :
  E + S + M = 272 :=
by
  sorry

end total_marks_l139_139062


namespace ball_hits_ground_l139_139470

theorem ball_hits_ground (t : ℝ) (y : ℝ) : 
  (y = -8 * t^2 - 12 * t + 72) → 
  (y = 0) → 
  t = 3 := 
by
  sorry

end ball_hits_ground_l139_139470


namespace num_adults_on_field_trip_l139_139119

-- Definitions of the conditions
def num_vans : Nat := 6
def people_per_van : Nat := 9
def num_students : Nat := 40

-- The theorem to prove
theorem num_adults_on_field_trip : (num_vans * people_per_van) - num_students = 14 := by
  sorry

end num_adults_on_field_trip_l139_139119


namespace remainder_S_div_500_l139_139039

def R : Set ℕ := { r | ∃ n : ℕ, r = (3^n) % 500 }

def S : ℕ := ∑ r in R.to_finset, r

theorem remainder_S_div_500 : S % 500 = 453 :=
by sorry

end remainder_S_div_500_l139_139039


namespace simplify_expression_l139_139563

variable (x y : ℝ)

theorem simplify_expression : 2 * x^2 * y - 4 * x * y^2 - (-3 * x * y^2 + x^2 * y) = x^2 * y - x * y^2 :=
by
  sorry

end simplify_expression_l139_139563


namespace pallet_weight_l139_139574

theorem pallet_weight (box_weight : ℕ) (num_boxes : ℕ) (total_weight : ℕ) 
  (h1 : box_weight = 89) (h2 : num_boxes = 3) : total_weight = 267 := by
  sorry

end pallet_weight_l139_139574


namespace angle_B_in_triangle_l139_139421

/-- In triangle ABC, if BC = √3, AC = √2, and ∠A = π/3,
then ∠B = π/4. -/
theorem angle_B_in_triangle
  (BC AC : ℝ) (A B : ℝ)
  (hBC : BC = Real.sqrt 3)
  (hAC : AC = Real.sqrt 2)
  (hA : A = Real.pi / 3) :
  B = Real.pi / 4 :=
sorry

end angle_B_in_triangle_l139_139421


namespace right_angled_triangles_with_cathetus_2021_l139_139409

theorem right_angled_triangles_with_cathetus_2021 :
  ∃ n : Nat, n = 4 ∧ ∀ (a b c : ℕ), ((a = 2021 ∧ a * a + b * b = c * c) ↔ (a = 2021 ∧ 
    ∃ m n, (m > n ∧ m > 0 ∧ n > 0 ∧ 2021 = m^2 - n^2 ∧ b = 2 * m * n ∧ c = m^2 + n^2))) :=
sorry

end right_angled_triangles_with_cathetus_2021_l139_139409


namespace triathlon_minimum_speeds_l139_139162

theorem triathlon_minimum_speeds (x : ℝ) (T : ℝ := 80) (total_time : ℝ := (800 / x + 20000 / (7.5 * x) + 4000 / (3 * x))) :
  total_time ≤ T → x ≥ 60 ∧ 3 * x = 180 ∧ 7.5 * x = 450 :=
by
  sorry

end triathlon_minimum_speeds_l139_139162


namespace page_number_added_twice_l139_139819

theorem page_number_added_twice (n p : ℕ) (Hn : 1 ≤ n) (Hsum : (n * (n + 1)) / 2 + p = 2630) : 
  p = 2 :=
sorry

end page_number_added_twice_l139_139819


namespace robin_gum_pieces_l139_139982

-- Defining the conditions
def packages : ℕ := 9
def pieces_per_package : ℕ := 15
def total_pieces : ℕ := 135

-- Theorem statement
theorem robin_gum_pieces (h1 : packages = 9) (h2 : pieces_per_package = 15) : packages * pieces_per_package = total_pieces := by
  -- According to the problem, the correct answer is 135 pieces
  have h: 9 * 15 = 135 := by norm_num
  rw [h1, h2]
  exact h

end robin_gum_pieces_l139_139982


namespace third_number_is_42_l139_139079

variable (x : ℕ)

def number1 : ℕ := 5 * x
def number2 : ℕ := 6 * x
def number3 : ℕ := 8 * x

theorem third_number_is_42 (h : number1 x + number3 x = number2 x + 49) : number2 x = 42 :=
by
  sorry

end third_number_is_42_l139_139079


namespace find_y_l139_139020

theorem find_y (x y : ℤ) (h1 : x + y = 300) (h2 : x - y = 200) : y = 50 :=
sorry

end find_y_l139_139020


namespace scale_drawing_represents_line_segment_l139_139996

-- Define the given conditions
def scale_factor : ℝ := 800
def line_segment_length_inch : ℝ := 4.75

-- Prove the length in feet
theorem scale_drawing_represents_line_segment :
  line_segment_length_inch * scale_factor = 3800 :=
by
  sorry

end scale_drawing_represents_line_segment_l139_139996


namespace apple_bags_l139_139524

theorem apple_bags (n : ℕ) (h₁ : n ≥ 70) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 := 
sorry

end apple_bags_l139_139524


namespace complement_intersection_l139_139879

-- Define the universal set U and sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {3, 4, 5}
def B : Set ℕ := {4, 7, 8}

-- Compute the complements
def complement_U (s : Set ℕ) : Set ℕ := U \ s
def comp_A : Set ℕ := complement_U A
def comp_B : Set ℕ := complement_U B

-- Define the intersection of the complements
def intersection_complements : Set ℕ := comp_A ∩ comp_B

-- The theorem to prove
theorem complement_intersection :
  intersection_complements = {1, 2, 6} :=
by
  sorry

end complement_intersection_l139_139879


namespace find_inverse_of_512_l139_139153

-- Define the function f with the given properties
def f : ℕ → ℕ := sorry

axiom f_initial : f 5 = 2
axiom f_property : ∀ x, f (2 * x) = 2 * f x

-- State the problem as a theorem
theorem find_inverse_of_512 : ∃ x, f x = 512 ∧ x = 1280 :=
by 
  -- Sorry to skip the proof
  sorry

end find_inverse_of_512_l139_139153


namespace equation_solution_l139_139324

theorem equation_solution (t : ℤ) : 
  ∃ y : ℤ, (21 * t + 2)^3 + 2 * (21 * t + 2)^2 + 5 = 21 * y :=
sorry

end equation_solution_l139_139324


namespace g_at_10_l139_139272

-- Definitions and conditions
def f : ℝ → ℝ := sorry
axiom f_at_1 : f 1 = 10
axiom f_inequality_1 : ∀ x : ℝ, f (x + 20) ≥ f x + 20
axiom f_inequality_2 : ∀ x : ℝ, f (x + 1) ≤ f x + 1
def g (x : ℝ) : ℝ := f x - x + 1

-- Proof statement (no proof required)
theorem g_at_10 : g 10 = 10 := sorry

end g_at_10_l139_139272


namespace solve_quadratic_completing_square_l139_139834

theorem solve_quadratic_completing_square :
  ∃ (a b c : ℤ), a > 0 ∧ 25 * a * a + 30 * b - 45 = (a * x + b)^2 - c ∧
                 a + b + c = 62 :=
by
  sorry

end solve_quadratic_completing_square_l139_139834


namespace b_divisible_by_8_l139_139926

theorem b_divisible_by_8 (b : ℕ) (h_even: ∃ k : ℕ, b = 2 * k) (h_square: ∃ n : ℕ, n > 1 ∧ ∃ m : ℕ, (b ^ n - 1) / (b - 1) = m ^ 2) : b % 8 = 0 := 
by
  sorry

end b_divisible_by_8_l139_139926


namespace possible_apple_counts_l139_139500

theorem possible_apple_counts (n : ℕ) (h₁ : 70 ≤ n) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end possible_apple_counts_l139_139500


namespace expenditure_ratio_l139_139073

variable (P1 P2 : Type)
variable (I1 I2 E1 E2 : ℝ)
variable (R_incomes : I1 / I2 = 5 / 4)
variable (S1 S2 : ℝ)
variable (S_equal : S1 = S2)
variable (I1_fixed : I1 = 4000)
variable (Savings : S1 = 1600)

theorem expenditure_ratio :
  (I1 - E1 = 1600) → 
  (I2 * 4 / 5 - E2 = 1600) →
  I2 = 3200 →
  E1 / E2 = 3 / 2 :=
by
  intro P1_savings P2_savings I2_calc
  -- proof steps go here
  sorry

end expenditure_ratio_l139_139073


namespace find_a4_a5_l139_139428

variable {α : Type*} [LinearOrderedField α]

-- Variables representing the terms of the geometric sequence
variables (a₁ a₂ a₃ a₄ a₅ q : α)

-- Conditions given in the problem
-- Geometric sequence condition
def is_geometric_sequence (a₁ a₂ a₃ a₄ a₅ q : α) : Prop :=
  a₂ = a₁ * q ∧ a₃ = a₂ * q ∧ a₄ = a₃ * q ∧ a₅ = a₄ * q

-- First condition
def condition1 : Prop := a₁ + a₂ = 3

-- Second condition
def condition2 : Prop := a₂ + a₃ = 6

-- Theorem stating that a₄ + a₅ = 24 given the conditions
theorem find_a4_a5
  (h1 : condition1 a₁ a₂)
  (h2 : condition2 a₂ a₃)
  (hg : is_geometric_sequence a₁ a₂ a₃ a₄ a₅ q) :
  a₄ + a₅ = 24 := 
sorry

end find_a4_a5_l139_139428


namespace expression_equals_neg_eight_l139_139902

variable {a b : ℝ}

theorem expression_equals_neg_eight (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : |a| ≠ |b|) :
  ( (b^2 / a^2 + a^2 / b^2 - 2) * 
    ((a + b) / (b - a) + (b - a) / (a + b)) * 
    (((1 / a^2 + 1 / b^2) / (1 / b^2 - 1 / a^2)) - ((1 / b^2 - 1 / a^2) / (1 / a^2 + 1 / b^2)))
  ) = -8 :=
by
  sorry

end expression_equals_neg_eight_l139_139902


namespace minimum_value_of_f_ge_7_l139_139874

noncomputable def f (x : ℝ) : ℝ :=
  x + (2 * x) / (x^2 + 1) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

theorem minimum_value_of_f_ge_7 {x : ℝ} (hx : x > 0) : f x ≥ 7 := 
by
  sorry

end minimum_value_of_f_ge_7_l139_139874


namespace arithmetic_sequence_nine_l139_139614

variable (a : ℕ → ℝ)
variable (d : ℝ)
-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_nine (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : arithmetic_sequence a d)
  (h_cond : a 4 + a 14 = 2) : 
  a 9 = 1 := 
sorry

end arithmetic_sequence_nine_l139_139614


namespace find_z2_l139_139762

theorem find_z2 (z1 z2 : ℂ) (h1 : z1 = 1 - I) (h2 : z1 * z2 = 1 + I) : z2 = I :=
sorry

end find_z2_l139_139762


namespace gcd_of_factorials_l139_139203

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Definitions for the two factorial terms involved
def term1 : ℕ := factorial 7
def term2 : ℕ := factorial 10 / factorial 4

-- Definition for computing the GCD
def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

-- The statement to prove
theorem gcd_of_factorials : gcd term1 term2 = 2520 :=
by {
  -- Here, the proof would go.
  sorry
}

end gcd_of_factorials_l139_139203


namespace find_angle_OPQ_l139_139443

variables {A B C O P Q : Type}
variables [EuclideanGeometry A B C O P Q]

def midpoint (M X Y : Type) [EuclideanGeometry X Y M] : Prop :=
  dist X M = dist M Y

variables (O_center : circle_center A B C O)
variables (P_midpoint_AO : midpoint P A O)
variables (Q_midpoint_BC : midpoint Q B C)
variables (x : Real.Angle.Degree)

def angle_CBA_eq_4x : angle_deg C B A = 4 * x := by sorry
def angle_ACB_eq_6x : angle_deg A C B = 6 * x := by sorry

theorem find_angle_OPQ :
  ∃ x : Real.Angle.Degree, (angle_deg O P Q = x) ∧ x = 12 := by
  use 12
  sorry

end find_angle_OPQ_l139_139443


namespace rectangle_area_l139_139769

theorem rectangle_area (x y : ℝ) (hx : x ≠ 0) (h : x * y = 10) : y = 10 / x :=
sorry

end rectangle_area_l139_139769


namespace limit_sequence_l139_139844

theorem limit_sequence :
  (Real.log (λ n : ℕ, let num := (n + 2)^4 - (n - 2)^4;
                    let denom := (n + 5)^2 + (n - 5)^2;
                    (num / denom))
  ) = +⟹
  sorry

end limit_sequence_l139_139844


namespace opposite_number_113_is_114_l139_139945

theorem opposite_number_113_is_114 :
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 200 → 
  (∀ k, (k + 100) % 200 ≠ 113 → 113 + 100 ≤ 200 → n = 113 →
  (k = 114)) :=
by
  intro n hn h_opposite
  sorry

end opposite_number_113_is_114_l139_139945


namespace find_original_number_l139_139228

variable (x : ℝ)

def tripled := 3 * x
def doubled := 2 * tripled
def subtracted := doubled - 9
def trebled := 3 * subtracted

theorem find_original_number (h : trebled = 90) : x = 6.5 := by
  sorry

end find_original_number_l139_139228


namespace tangent_normal_lines_l139_139133

theorem tangent_normal_lines :
  ∃ m_t b_t m_n b_n,
    (∀ x y, y = 1 / (1 + x^2) → y = m_t * x + b_t → 4 * x + 25 * y - 13 = 0) ∧
    (∀ x y, y = 1 / (1 + x^2) → y = m_n * x + b_n → 125 * x - 20 * y - 246 = 0) :=
by
  sorry

end tangent_normal_lines_l139_139133


namespace gcd_of_75_and_360_l139_139735

theorem gcd_of_75_and_360 : Nat.gcd 75 360 = 15 := by
  sorry

end gcd_of_75_and_360_l139_139735


namespace negation_exists_lt_zero_l139_139071

variable {f : ℝ → ℝ}

theorem negation_exists_lt_zero :
  ¬ (∃ x : ℝ, f x < 0) → ∀ x : ℝ, 0 ≤ f x := by
  sorry

end negation_exists_lt_zero_l139_139071


namespace initial_sodium_chloride_percentage_l139_139360

theorem initial_sodium_chloride_percentage :
  ∀ (P : ℝ),
  (∃ (C : ℝ), C = 24) → -- Tank capacity
  (∃ (E_rate : ℝ), E_rate = 0.4) → -- Evaporation rate per hour
  (∃ (time : ℝ), time = 6) → -- Time in hours
  (1 / 4 * C = 6) → -- Volume of mixture
  (6 * P / 100 + (6 - 6 * P / 100 - E_rate * time) = 3.6) → -- Concentration condition
  P = 30 :=
by
  intros P hC hE_rate htime hvolume hconcentration
  rcases hC with ⟨C, hC⟩
  rcases hE_rate with ⟨E_rate, hE_rate⟩
  rcases htime with ⟨time, htime⟩
  rw [hC, hE_rate, htime] at *
  sorry

end initial_sodium_chloride_percentage_l139_139360


namespace largest_among_given_numbers_l139_139840

theorem largest_among_given_numbers : 
    let a := 24680 + (1 / 1357)
    let b := 24680 - (1 / 1357)
    let c := 24680 * (1 / 1357)
    let d := 24680 / (1 / 1357)
    let e := 24680.1357
    d > a ∧ d > b ∧ d > c ∧ d > e :=
by
  sorry

end largest_among_given_numbers_l139_139840


namespace intercepts_sum_eq_seven_l139_139184

theorem intercepts_sum_eq_seven :
    (∃ a b c, (∀ y, (3 * y^2 - 9 * y + 4 = a) → y = 0) ∧ 
              (∀ y, (3 * y^2 - 9 * y + 4 = 0) → (y = b ∨ y = c)) ∧ 
              (a + b + c = 7)) := 
sorry

end intercepts_sum_eq_seven_l139_139184


namespace part_a_part_b_part_c_l139_139442

open Finset

-- Define N, S1, and S2
variables {N : ℕ} (S1 S2 : Finset ℕ)

-- Define the condition of a "good division"
def is_good_division (S1 S2 : Finset ℕ) : Prop :=
  (S1 ∩ S2 = ∅) ∧ (S1 ∪ S2 = range (N + 1)) ∧ (S1.card ≠ 0) ∧ (S2.card ≠ 0) ∧ 
  (∑ x in S1, x = ∏ x in S2, x)

-- Part (a)
theorem part_a : ∃ (S1 S2 : Finset ℕ), is_good_division 7 {2, 4, 5, 7} {1, 3, 6} :=
by { sorry }

-- Part (b)
theorem part_b : ∃ N ≥ 1, (∃ S1 S2 : Finset ℕ, is_good_division N S1 S2) ∧ (∃ S1' S2' : Finset ℕ, is_good_division N S1' S2' ∧ (S1 ≠ S1' ∨ S2 ≠ S2')) :=
by { sorry }

-- Part (c)
theorem part_c (N : ℕ) (hN : N ≥ 5) : ∃ S1 S2 : Finset ℕ, is_good_division N S1 S2 :=
by { sorry }

end part_a_part_b_part_c_l139_139442


namespace non_rent_extra_expenses_is_3000_l139_139304

-- Define the constants
def cost_parts : ℕ := 800
def markup : ℝ := 1.4
def num_computers : ℕ := 60
def rent : ℕ := 5000
def profit : ℕ := 11200

-- Calculate the selling price per computer
def selling_price : ℝ := cost_parts * markup

-- Calculate the total revenue from selling 60 computers
def total_revenue : ℝ := selling_price * num_computers

-- Calculate the total cost of components for 60 computers
def total_cost_components : ℕ := cost_parts * num_computers

-- Calculate the total expenses
def total_expenses : ℝ := total_revenue - profit

-- Define the non-rent extra expenses
def non_rent_extra_expenses : ℝ := total_expenses - rent - total_cost_components

-- Prove that the non-rent extra expenses equal to $3000
theorem non_rent_extra_expenses_is_3000 : non_rent_extra_expenses = 3000 := sorry

end non_rent_extra_expenses_is_3000_l139_139304


namespace price_of_pants_l139_139934

theorem price_of_pants (S P H : ℝ) (h1 : 0.8 * S + P + H = 340) (h2 : S = (3 / 4) * P) (h3 : H = P + 10) : P = 91.67 :=
by sorry

end price_of_pants_l139_139934


namespace john_votes_l139_139706

theorem john_votes (J : ℝ) (total_votes : ℝ) (third_candidate_votes : ℝ) (james_votes : ℝ) 
  (h1 : total_votes = 1150) 
  (h2 : third_candidate_votes = J + 150) 
  (h3 : james_votes = 0.70 * (total_votes - J - third_candidate_votes)) 
  (h4 : total_votes = J + james_votes + third_candidate_votes) : 
  J = 500 := 
by 
  rw [h1, h2, h3] at h4 
  sorry

end john_votes_l139_139706


namespace distance_between_points_on_line_l139_139007

theorem distance_between_points_on_line 
  (p q r s : ℝ)
  (line_eq : q = 2 * p + 3) 
  (s_eq : s = 2 * r + 6) :
  Real.sqrt ((r - p)^2 + (s - q)^2) = Real.sqrt (5 * (r - p)^2 + 12 * (r - p) + 9) :=
sorry

end distance_between_points_on_line_l139_139007


namespace distinct_three_digit_numbers_distinct_three_digit_odd_numbers_l139_139201

-- Definitions based on conditions
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}
def odd_digits : Finset ℕ := {1, 3, 5}

-- Problem 1: Number of distinct three-digit numbers
theorem distinct_three_digit_numbers : (digits.erase 0).card * (digits.erase 0).card.pred * (digits.erase 0).card.pred.pred = 100 := by
  sorry

-- Problem 2: Number of distinct three-digit odd numbers
theorem distinct_three_digit_odd_numbers : (odd_digits.card) * (digits.erase 0).card.pred * (digits.erase 0).card.pred.pred = 48 := by
  sorry

end distinct_three_digit_numbers_distinct_three_digit_odd_numbers_l139_139201


namespace tidy_up_time_l139_139282

theorem tidy_up_time (A B C : ℕ) (tidyA : A = 5 * 3600) (tidyB : B = 5 * 60) (tidyC : C = 5) :
  B < A ∧ B > C :=
by
  sorry

end tidy_up_time_l139_139282


namespace fraction_division_l139_139151

theorem fraction_division (a b : ℚ) (ha : a = 3) (hb : b = 4) :
  (1 / b) / (1 / a) = 3 / 4 :=
by 
  -- Solve the proof
  sorry

end fraction_division_l139_139151


namespace number_of_arrangements_l139_139851

theorem number_of_arrangements :
  ∃ (n k : ℕ), n = 10 ∧ k = 5 ∧ Nat.choose n k = 252 := by
  sorry

end number_of_arrangements_l139_139851


namespace area_ratio_of_smaller_octagon_l139_139676

theorem area_ratio_of_smaller_octagon (A B C D E F G H P Q R S T U V W : Point) 
  (h1 : is_regular_octagon A B C D E F G H)
  (h2 : midpoint A B = P) (h3 : midpoint B C = Q) (h4 : midpoint C D = R)
  (h5 : midpoint D E = S) (h6 : midpoint E F = T) (h7 : midpoint F G = U)
  (h8 : midpoint G H = V) (h9 : midpoint H A = W):
  area (octagon A B C D E F G H) / area (octagon P Q R S T U V W) = 4 := sorry

end area_ratio_of_smaller_octagon_l139_139676


namespace trackball_mice_count_l139_139314

theorem trackball_mice_count (total_sales wireless_share optical_share : ℕ) 
    (h_total : total_sales = 80)
    (h_wireless : wireless_share = total_sales / 2)
    (h_optical : optical_share = total_sales / 4):
    total_sales - (wireless_share + optical_share) = 20 :=
by
  sorry

end trackball_mice_count_l139_139314


namespace triangle_DEF_area_10_l139_139081

-- Definitions of vertices and line
def D : ℝ × ℝ := (4, 0)
def E : ℝ × ℝ := (0, 4)
def line (x y : ℝ) : Prop := x + y = 9

-- Definition of point F lying on the given line
axiom F_on_line (F : ℝ × ℝ) : line (F.1) (F.2)

-- The proof statement of the area of triangle DEF being 10
theorem triangle_DEF_area_10 : ∃ F : ℝ × ℝ, line F.1 F.2 ∧ 
  (1 / 2) * abs (D.1 - F.1) * abs E.2 = 10 :=
by
  sorry

end triangle_DEF_area_10_l139_139081


namespace coopers_age_l139_139824

theorem coopers_age (C D M E : ℝ) 
  (h1 : D = 2 * C) 
  (h2 : M = 2 * C + 1) 
  (h3 : E = 3 * C)
  (h4 : C + D + M + E = 62) : 
  C = 61 / 8 := 
by 
  sorry

end coopers_age_l139_139824


namespace cube_surface_area_correct_l139_139569

noncomputable def total_surface_area_of_reassembled_cube : ℝ :=
  let height_X := 1 / 4
  let height_Y := 1 / 6
  let height_Z := 1 - (height_X + height_Y)
  let top_bottom_area := 3 * 1 -- Each slab contributes 1 square foot for the top and bottom
  let side_area := 2 * 1 -- Each side slab contributes 1 square foot
  let front_back_area := 2 * 1 -- Each front and back contributes 1 square foot
  top_bottom_area + side_area + front_back_area

theorem cube_surface_area_correct :
  let height_X := 1 / 4
  let height_Y := 1 / 6
  let height_Z := 1 - (height_X + height_Y)
  let total_surface_area := total_surface_area_of_reassembled_cube
  total_surface_area = 10 :=
by
  sorry

end cube_surface_area_correct_l139_139569


namespace expression_evaluation_l139_139152

theorem expression_evaluation (x : ℤ) (hx : x = 4) : 5 * x + 3 - x^2 = 7 :=
by
  sorry

end expression_evaluation_l139_139152


namespace intersection_A_B_l139_139621

def A := {x : ℝ | x < -1 ∨ x > 1}
def B := {x : ℝ | Real.log x / Real.log 2 > 0}

theorem intersection_A_B:
  A ∩ B = {x : ℝ | x > 1} :=
by
  sorry

end intersection_A_B_l139_139621


namespace total_apples_l139_139520

theorem total_apples (n : ℕ) (h1 : ∀ k : ℕ, 6 * k = n ∨ 12 * k = n) (h2 : 70 ≤ n) (h3 : n ≤ 80) : 
  n = 72 ∨ n = 78 := 
sorry

end total_apples_l139_139520


namespace quadratic_function_positive_l139_139756

theorem quadratic_function_positive {a b c : ℝ} (h : a^2 = b^2 + c^2 - 2 * b * c * real.cos A) :
  ∀ x : ℝ, b^2 * x^2 + (b^2 + c^2 - a^2) * x + c^2 > 0 :=
begin
  intro x,
  rw [h],
  let f := λ x, b^2 * x^2 + 2 * b * c * real.cos A * x + c^2,
  have discriminant_neg : 4 * b^2 * c^2 * (real.cos A ^ 2 - 1) < 0,
  {
    calc 4 * b^2 * c^2 * (real.cos A ^ 2 - 1) < 0,
    { -- Use the condition that \(b^2 > 0\) and \(\cos^2 A - 1 < 0\)
      sorry }
  },
  by_cases hbc : b = 0,
  {
    -- Handle the case where \(b = 0\)
    sorry
  },
  {
    -- Handle the case where \(b ≠ 0\)
    sorry
  }
end

end quadratic_function_positive_l139_139756


namespace juice_water_ratio_l139_139702

theorem juice_water_ratio (V : ℝ) :
  let glass_juice_ratio := (2, 1)
  let mug_volume := 2 * V
  let mug_juice_ratio := (4, 1)
  let glass_juice_vol := (2 / 3) * V
  let glass_water_vol := (1 / 3) * V
  let mug_juice_vol := (8 / 5) * V
  let mug_water_vol := (2 / 5) * V
  let total_juice := glass_juice_vol + mug_juice_vol
  let total_water := glass_water_vol + mug_water_vol
  let ratio := total_juice / total_water
  ratio = 34 / 11 :=
by
  sorry

end juice_water_ratio_l139_139702


namespace polynomial_characterization_l139_139730
open Polynomial

noncomputable def satisfies_functional_eq (P : Polynomial ℝ) :=
  ∀ (a b c : ℝ), 
  P.eval (a + b - 2*c) + P.eval (b + c - 2*a) + P.eval (c + a - 2*b) = 
  3 * P.eval (a - b) + 3 * P.eval (b - c) + 3 * P.eval (c - a)

theorem polynomial_characterization (P : Polynomial ℝ) :
  satisfies_functional_eq P ↔ 
  (∃ a b : ℝ, P = Polynomial.C a * Polynomial.X + Polynomial.C b) ∨
  (∃ a b : ℝ, P = Polynomial.C a * Polynomial.X^2 + Polynomial.C b * Polynomial.X) :=
sorry

end polynomial_characterization_l139_139730


namespace sqrt_S_n_arithmetic_seq_seq_sqrt_S_n_condition_l139_139884

-- (1)
theorem sqrt_S_n_arithmetic_seq (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 1) (h2 : ∃ (d : ℝ), ∀ n, a (n + 1) = a n + d) (h3 : S n = (n * (2 * a 1 + (n - 1) * (2 : ℝ))) / 2) :
  ∃ d, ∀ n, Real.sqrt (S (n + 1)) = Real.sqrt (S n) + d :=
by sorry

-- (2)
theorem seq_sqrt_S_n_condition (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 : ℝ) :
  (∃ d, ∀ n, S n / 2 = n * (a1 + (n - 1) * d)) ↔ (∀ n, S n = a1 * n^2) :=
by sorry

end sqrt_S_n_arithmetic_seq_seq_sqrt_S_n_condition_l139_139884


namespace average_difference_l139_139329

def daily_differences : List ℤ := [2, -1, 3, 1, -2, 2, 1]

theorem average_difference :
  (daily_differences.sum : ℚ) / daily_differences.length = 0.857 :=
by
  sorry

end average_difference_l139_139329


namespace intersection_empty_implies_m_leq_neg1_l139_139649

theorem intersection_empty_implies_m_leq_neg1 (m : ℝ) :
  (∀ (x y: ℝ), (x < m) → (y = x^2 + 2*x) → y < -1) →
  m ≤ -1 :=
by
  intro h
  sorry

end intersection_empty_implies_m_leq_neg1_l139_139649


namespace range_of_a_l139_139752

open Set

-- Define proposition p
def p (x : ℝ) : Prop := x^2 + 2 * x - 3 > 0

-- Define proposition q
def q (x a : ℝ) : Prop := (x - a) / (x - a - 1) > 0

-- Define negation of p
def not_p (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 1

-- Define negation of q
def not_q (x a : ℝ) : Prop := a ≤ x ∧ x ≤ a + 1

-- Main theorem to prove the range of a
theorem range_of_a (a : ℝ) : (∀ x : ℝ, a ≤ x ∧ x ≤ a + 1 → -3 ≤ x ∧ x ≤ 1) → a ∈ Icc (-3 : ℝ) (0 : ℝ) :=
by
  intro h
  -- skipped detailed proof
  sorry

end range_of_a_l139_139752


namespace probability_of_both_white_l139_139640

namespace UrnProblem

-- Define the conditions
def firstUrnWhiteBalls : ℕ := 4
def firstUrnTotalBalls : ℕ := 10
def secondUrnWhiteBalls : ℕ := 7
def secondUrnTotalBalls : ℕ := 12

-- Define the probabilities of drawing a white ball from each urn
def P_A1 : ℚ := firstUrnWhiteBalls / firstUrnTotalBalls
def P_A2 : ℚ := secondUrnWhiteBalls / secondUrnTotalBalls

-- Define the combined probability of both events occurring
def P_A1_and_A2 : ℚ := P_A1 * P_A2

-- Theorem statement that checks the combined probability
theorem probability_of_both_white : P_A1_and_A2 = 7 / 30 := by
  sorry

end UrnProblem

end probability_of_both_white_l139_139640


namespace geometric_shape_circle_l139_139877

variables (c φ_0 : ℝ)

-- Assuming positive constants for c and φ_0
axiom h_c_pos : c > 0
axiom h_φ0_pos : φ_0 > 0

-- The main statement: Given that ρ = c and φ = φ_0, the geometric shape is a circle.
theorem geometric_shape_circle (ρ θ φ : ℝ) (h_ρ : ρ = c) (h_φ : φ = φ_0) :
  ∃ (r : ℝ), r > 0 ∧ ∀ θ, (ρ = c ∧ φ = φ_0) → (∃ (x y : ℝ), x^2 + y^2 = r^2) :=
by 
  sorry

end geometric_shape_circle_l139_139877


namespace fx_greater_than_2_l139_139001

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log x

theorem fx_greater_than_2 :
  ∀ x : ℝ, x > 0 → f x > 2 :=
by {
  sorry
}

end fx_greater_than_2_l139_139001


namespace no_55_rooms_l139_139914

theorem no_55_rooms 
  (count_roses count_carnations count_chrysanthemums : ℕ)
  (rooms_with_CC rooms_with_CR rooms_with_HR : ℕ)
  (at_least_one_bouquet_in_each_room: ∀ (room: ℕ), room > 0)
  (total_rooms : ℕ)
  (h_bouquets : count_roses = 30 ∧ count_carnations = 20 ∧ count_chrysanthemums = 10)
  (h_overlap_conditions: rooms_with_CC = 2 ∧ rooms_with_CR = 3 ∧ rooms_with_HR = 4):
  (total_rooms != 55) :=
sorry

end no_55_rooms_l139_139914


namespace circle_diameter_mn_origin_l139_139893

-- Definitions based on conditions in (a)
def circle_equation (m : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 4 * y + m = 0
def line_equation (x y : ℝ) : Prop := x + 2 * y - 4 = 0
def orthogonal (x1 x2 y1 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Main theorem to prove (based on conditions and correct answer in (b))
theorem circle_diameter_mn_origin 
  (m : ℝ) 
  (x1 y1 x2 y2 : ℝ)
  (h1: circle_equation m x1 y1) 
  (h2: circle_equation m x2 y2)
  (h3: line_equation x1 y1)
  (h4: line_equation x2 y2)
  (h5: orthogonal x1 x2 y1 y2) :
  m = 8 / 5 := 
sorry

end circle_diameter_mn_origin_l139_139893


namespace conditional_probability_l139_139958

variable (P : ℕ → ℚ)
variable (A B : ℕ)

def EventRain : Prop := P A = 4/15
def EventWind : Prop := P B = 2/15
def EventBoth : Prop := P (A * B) = 1/10

theorem conditional_probability 
  (h1 : EventRain P A) 
  (h2 : EventWind P B) 
  (h3 : EventBoth P A B) 
  : (P (A * B) / P A) = 3 / 8 := 
by
  sorry

end conditional_probability_l139_139958


namespace proof_problem_l139_139788

def f (x : ℝ) : ℝ := x - 2
def g (x : ℝ) : ℝ := 2 * x + 4

theorem proof_problem : (f (g 3))^2 - (g (f 3))^2 = 28 := by
  sorry

end proof_problem_l139_139788


namespace range_of_x_l139_139399

variable (f : ℝ → ℝ)

def even_function :=
  ∀ x : ℝ, f (-x) = f x

def monotonically_decreasing :=
  ∀ x y : ℝ, 0 ≤ x → x ≤ y → f y ≤ f x

def f_value_at_2 := f 2 = 0

theorem range_of_x (h1 : even_function f) (h2 : monotonically_decreasing f) (h3 : f_value_at_2 f) :
  { x : ℝ | f (x - 1) > 0 } = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end range_of_x_l139_139399


namespace wheels_on_floor_l139_139801

def number_of_wheels (n_people : Nat) (w_per_person : Nat) : Nat :=
  n_people * w_per_person

theorem wheels_on_floor (n_people : Nat) (w_per_person : Nat) (h_people : n_people = 40) (h_wheels : w_per_person = 4) :
  number_of_wheels n_people w_per_person = 160 := by
  sorry

end wheels_on_floor_l139_139801


namespace arithmetic_sequence_a101_eq_52_l139_139864

theorem arithmetic_sequence_a101_eq_52 (a : ℕ → ℝ)
  (h₁ : a 1 = 2)
  (h₂ : ∀ n : ℕ, a (n + 1) - a n = 1 / 2) :
  a 101 = 52 :=
by
  sorry

end arithmetic_sequence_a101_eq_52_l139_139864


namespace find_common_ratio_l139_139389

noncomputable def geometric_seq_sum (a₁ q : ℂ) (n : ℕ) :=
if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem find_common_ratio (a₁ q : ℂ) :
(geometric_seq_sum a₁ q 8) / (geometric_seq_sum a₁ q 4) = 2 → q = 1 :=
by
  intro h
  sorry

end find_common_ratio_l139_139389


namespace pieces_info_at_most_two_identical_digits_l139_139021

def num_pieces_of_information_with_at_most_two_positions_as_0110 : Nat :=
  (Nat.choose 4 2 + Nat.choose 4 1 + Nat.choose 4 0)

theorem pieces_info_at_most_two_identical_digits :
  num_pieces_of_information_with_at_most_two_positions_as_0110 = 11 :=
by
  sorry

end pieces_info_at_most_two_identical_digits_l139_139021


namespace yanna_gave_100_l139_139974

/--
Yanna buys 10 shirts at $5 each and 3 pairs of sandals at $3 each, 
and she receives $41 in change. Prove that she gave $100.
-/
theorem yanna_gave_100 :
  let cost_shirts := 10 * 5
  let cost_sandals := 3 * 3
  let total_cost := cost_shirts + cost_sandals
  let change := 41
  total_cost + change = 100 :=
by
  let cost_shirts := 10 * 5
  let cost_sandals := 3 * 3
  let total_cost := cost_shirts + cost_sandals
  let change := 41
  show total_cost + change = 100
  sorry

end yanna_gave_100_l139_139974


namespace gcd_factorial_l139_139206

theorem gcd_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  sorry

end gcd_factorial_l139_139206


namespace sufficient_but_not_necessary_sufficient_but_not_necessary_rel_l139_139755

theorem sufficient_but_not_necessary (a b : ℝ) (h : 0 < a ∧ a < b) : (1 / a) > (1 / b) :=
by
  sorry

theorem sufficient_but_not_necessary_rel (a b : ℝ) : 0 < a ∧ a < b ↔ (1 / a) > (1 / b) :=
by
  sorry

end sufficient_but_not_necessary_sufficient_but_not_necessary_rel_l139_139755


namespace pool_fill_time_l139_139230

theorem pool_fill_time:
  ∀ (A B C D : ℚ),
  (A + B - D = 1 / 6) →
  (A + C - D = 1 / 5) →
  (B + C - D = 1 / 4) →
  (A + B + C - D = 1 / 3) →
  (1 / (A + B + C) = 60 / 23) :=
by intros A B C D h1 h2 h3 h4; sorry

end pool_fill_time_l139_139230


namespace number_of_children_correct_l139_139337

def total_spectators : ℕ := 25000
def men_spectators : ℕ := 15320
def ratio_children_women : ℕ × ℕ := (7, 3)
def remaining_spectators : ℕ := total_spectators - men_spectators
def total_ratio_parts : ℕ := ratio_children_women.1 + ratio_children_women.2
def spectators_per_part : ℕ := remaining_spectators / total_ratio_parts

def children_spectators : ℕ := spectators_per_part * ratio_children_women.1

theorem number_of_children_correct : children_spectators = 6776 := by
  sorry

end number_of_children_correct_l139_139337


namespace find_f_2021_l139_139817

variable (f : ℝ → ℝ)

axiom functional_equation : ∀ a b : ℝ, f ( (a + 2 * b) / 3) = (f a + 2 * f b) / 3
axiom f_one : f 1 = 1
axiom f_four : f 4 = 7

theorem find_f_2021 : f 2021 = 4041 := by
  sorry

end find_f_2021_l139_139817


namespace parabola_ordinate_l139_139320

theorem parabola_ordinate (x y : ℝ) (h : y = 2 * x^2) (d : dist (x, y) (0, 1 / 8) = 9 / 8) : y = 1 := 
sorry

end parabola_ordinate_l139_139320


namespace carnival_ticket_count_l139_139562

theorem carnival_ticket_count (ferris_wheel_rides bumper_car_rides ride_cost : ℕ) 
  (h1 : ferris_wheel_rides = 7) 
  (h2 : bumper_car_rides = 3) 
  (h3 : ride_cost = 5) : 
  ferris_wheel_rides + bumper_car_rides * ride_cost = 50 := 
by {
  -- proof omitted
  sorry
}

end carnival_ticket_count_l139_139562


namespace cost_price_USD_l139_139106

-- Assume the conditions in Lean as given:
variable {C_USD : ℝ}

def condition1 (C_USD : ℝ) : Prop := 0.9 * C_USD + 200 = 1.04 * C_USD

theorem cost_price_USD (h : condition1 C_USD) : C_USD = 200 / 0.14 :=
by
  sorry

end cost_price_USD_l139_139106


namespace find_vertex_D_l139_139625

structure Point where
  x : ℤ
  y : ℤ

def vector_sub (a b : Point) : Point :=
  Point.mk (a.x - b.x) (a.y - b.y)

def vector_add (a b : Point) : Point :=
  Point.mk (a.x + b.x) (a.y + b.y)

def is_parallelogram (A B C D : Point) : Prop :=
  vector_sub B A = vector_sub D C

theorem find_vertex_D (A B C D : Point)
  (hA : A = Point.mk (-1) (-2))
  (hB : B = Point.mk 3 (-1))
  (hC : C = Point.mk 5 6)
  (hParallelogram: is_parallelogram A B C D) :
  D = Point.mk 1 5 :=
sorry

end find_vertex_D_l139_139625


namespace average_matches_rounded_l139_139157

def total_matches : ℕ := 6 * 1 + 3 * 2 + 3 * 3 + 2 * 4 + 6 * 5

def total_players : ℕ := 6 + 3 + 3 + 2 + 6

noncomputable def average_matches : ℚ := total_matches / total_players

theorem average_matches_rounded : Int.floor (average_matches + 0.5) = 3 :=
by
  unfold average_matches total_matches total_players
  norm_num
  sorry

end average_matches_rounded_l139_139157


namespace factorization_correct_l139_139584

theorem factorization_correct :
  (¬ (x^2 - 2 * x - 1 = x * (x - 2) - 1)) ∧
  (¬ (2 * x + 1 = x * (2 + 1 / x))) ∧
  (¬ ((x + 2) * (x - 2) = x^2 - 4)) ∧
  (x^2 - 1 = (x + 1) * (x - 1)) :=
by
  sorry

end factorization_correct_l139_139584


namespace calculate_total_driving_time_l139_139992

/--
A rancher needs to transport 400 head of cattle to higher ground 60 miles away.
His truck holds 20 head of cattle and travels at 60 miles per hour.
Prove that the total driving time to transport all cattle is 40 hours.
-/
theorem calculate_total_driving_time
  (total_cattle : Nat)
  (cattle_per_trip : Nat)
  (distance_one_way : Nat)
  (speed : Nat)
  (round_trip_miles : Nat)
  (total_miles : Nat)
  (total_time_hours : Nat)
  (h1 : total_cattle = 400)
  (h2 : cattle_per_trip = 20)
  (h3 : distance_one_way = 60)
  (h4 : speed = 60)
  (h5 : round_trip_miles = 2 * distance_one_way)
  (h6 : total_miles = (total_cattle / cattle_per_trip) * round_trip_miles)
  (h7 : total_time_hours = total_miles / speed) :
  total_time_hours = 40 :=
by
  sorry

end calculate_total_driving_time_l139_139992


namespace kevin_hopped_distance_after_four_hops_l139_139031

noncomputable def kevin_total_hopped_distance : ℚ :=
  let hop1 := 1
  let hop2 := 1 / 2
  let hop3 := 1 / 4
  let hop4 := 1 / 8
  hop1 + hop2 + hop3 + hop4

theorem kevin_hopped_distance_after_four_hops :
  kevin_total_hopped_distance = 15 / 8 :=
by
  sorry

end kevin_hopped_distance_after_four_hops_l139_139031


namespace total_reams_of_paper_l139_139280

def reams_for_haley : ℕ := 2
def reams_for_sister : ℕ := 3

theorem total_reams_of_paper : reams_for_haley + reams_for_sister = 5 := by
  sorry

end total_reams_of_paper_l139_139280


namespace apple_count_l139_139481

/--
Given the total number of apples in several bags, where each bag contains either 12 or 6 apples,
and knowing that the total number of apples is between 70 and 80 inclusive,
prove that the total number of apples can only be 72 or 78.
-/
theorem apple_count (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : ∃ k : ℕ, n = 6 * k) : n = 72 ∨ n = 78 :=
by {
  sorry
}

end apple_count_l139_139481


namespace harry_lost_sea_creatures_l139_139768

def initial_sea_stars := 34
def initial_seashells := 21
def initial_snails := 29
def initial_crabs := 17

def sea_stars_reproduced := 5
def seashells_reproduced := 3
def snails_reproduced := 4

def final_items := 105

def sea_stars_after_reproduction := initial_sea_stars + (sea_stars_reproduced * 2 - sea_stars_reproduced)
def seashells_after_reproduction := initial_seashells + (seashells_reproduced * 2 - seashells_reproduced)
def snails_after_reproduction := initial_snails + (snails_reproduced * 2 - snails_reproduced)
def crabs_after_reproduction := initial_crabs

def total_after_reproduction := sea_stars_after_reproduction + seashells_after_reproduction + snails_after_reproduction + crabs_after_reproduction

theorem harry_lost_sea_creatures : total_after_reproduction - final_items = 8 :=
by
  sorry

end harry_lost_sea_creatures_l139_139768


namespace multiple_choice_options_l139_139585

-- Define the problem conditions
def num_true_false_combinations : ℕ := 14
def num_possible_keys (n : ℕ) : ℕ := num_true_false_combinations * n^2
def total_keys : ℕ := 224

-- The theorem problem
theorem multiple_choice_options : ∃ n : ℕ, num_possible_keys n = total_keys ∧ n = 4 := by
  -- We don't need to provide the proof, so we use sorry. 
  sorry

end multiple_choice_options_l139_139585


namespace third_angle_of_triangle_l139_139912

theorem third_angle_of_triangle (a b : ℝ) (ha : a = 50) (hb : b = 60) : 
  ∃ (c : ℝ), a + b + c = 180 ∧ c = 70 :=
by
  sorry

end third_angle_of_triangle_l139_139912


namespace largest_num_blocks_l139_139546

-- Define the volume of the box
def volume_box (l₁ w₁ h₁ : ℕ) : ℕ :=
  l₁ * w₁ * h₁

-- Define the volume of the block
def volume_block (l₂ w₂ h₂ : ℕ) : ℕ :=
  l₂ * w₂ * h₂

-- Define the function to calculate maximum blocks
def max_blocks (V_box V_block : ℕ) : ℕ :=
  V_box / V_block

theorem largest_num_blocks :
  max_blocks (volume_box 5 4 6) (volume_block 3 3 2) = 6 :=
by
  sorry

end largest_num_blocks_l139_139546


namespace net_change_correct_l139_139999
-- Import the necessary library

-- Price calculation function
def price_after_changes (initial_price: ℝ) (changes: List (ℝ -> ℝ)): ℝ :=
  changes.foldl (fun price change => change price) initial_price

-- Define each model's price changes
def modelA_changes: List (ℝ -> ℝ) := [
  fun price => price * 0.9, 
  fun price => price * 1.3, 
  fun price => price * 0.85
]

def modelB_changes: List (ℝ -> ℝ) := [
  fun price => price * 0.85, 
  fun price => price * 1.25, 
  fun price => price * 0.80
]

def modelC_changes: List (ℝ -> ℝ) := [
  fun price => price * 0.80, 
  fun price => price * 1.20, 
  fun price => price * 0.95
]

-- Calculate final prices
def final_price_modelA := price_after_changes 1000 modelA_changes
def final_price_modelB := price_after_changes 1500 modelB_changes
def final_price_modelC := price_after_changes 2000 modelC_changes

-- Calculate net changes
def net_change_modelA := final_price_modelA - 1000
def net_change_modelB := final_price_modelB - 1500
def net_change_modelC := final_price_modelC - 2000

-- Set up theorem
theorem net_change_correct:
  net_change_modelA = -5.5 ∧ net_change_modelB = -225 ∧ net_change_modelC = -176 := by
  -- Proof is skipped
  sorry

end net_change_correct_l139_139999


namespace infinite_primes_4k1_l139_139454

theorem infinite_primes_4k1 : ∀ (P : List ℕ), (∀ (p : ℕ), p ∈ P → Nat.Prime p ∧ ∃ k, p = 4 * k + 1) → 
  ∃ q, Nat.Prime q ∧ ∃ k, q = 4 * k + 1 ∧ q ∉ P :=
sorry

end infinite_primes_4k1_l139_139454


namespace trees_per_square_meter_l139_139667

-- Definitions of the given conditions
def side_length : ℕ := 100
def total_trees : ℕ := 120000

def area_of_street : ℤ := side_length * side_length
def area_of_forest : ℤ := 3 * area_of_street

-- The question translated to Lean theorem statement
theorem trees_per_square_meter (h1: area_of_street = side_length * side_length)
    (h2: area_of_forest = 3 * area_of_street) 
    (h3: total_trees = 120000) : 
    (total_trees / area_of_forest) = 4 :=
sorry

end trees_per_square_meter_l139_139667


namespace united_telephone_additional_charge_l139_139692

theorem united_telephone_additional_charge :
  ∃ x : ℝ, 
    (11 + 20 * x = 16) ↔ (x = 0.25) := by
  sorry

end united_telephone_additional_charge_l139_139692


namespace waste_scientific_notation_correct_l139_139718

def total_waste_in_scientific : ℕ := 500000000000

theorem waste_scientific_notation_correct :
  total_waste_in_scientific = 5 * 10^10 :=
by
  sorry

end waste_scientific_notation_correct_l139_139718


namespace valid_number_of_apples_l139_139511

theorem valid_number_of_apples (n : ℕ) : 
  (∃ k m : ℕ, n = 12 * k + 6 * m) ∧ (70 ≤ n ∧ n ≤ 80) ↔ (n = 72 ∨ n = 78) :=
by
  sorry

end valid_number_of_apples_l139_139511


namespace next_leap_year_visible_after_2017_l139_139174

def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0) ∧ ((y % 100 ≠ 0) ∨ (y % 400 = 0))

def stromquist_visible (start_year interval next_leap : ℕ) : Prop :=
  ∃ k : ℕ, next_leap = start_year + k * interval ∧ is_leap_year next_leap

theorem next_leap_year_visible_after_2017 :
  stromquist_visible 2017 61 2444 :=
  sorry

end next_leap_year_visible_after_2017_l139_139174


namespace expression_evaluates_to_3_l139_139376

theorem expression_evaluates_to_3 :
  (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3)) = 3 :=
sorry

end expression_evaluates_to_3_l139_139376


namespace scott_invests_l139_139456

theorem scott_invests (x r : ℝ) (h1 : 2520 = x + 1260) (h2 : 2520 * 0.08 = x * r) : r = 0.16 :=
by
  -- Proof goes here
  sorry

end scott_invests_l139_139456


namespace taxi_ride_cost_l139_139237

-- Definitions based on the conditions
def fixed_cost : ℝ := 2.00
def variable_cost_per_mile : ℝ := 0.30
def distance_traveled : ℝ := 7

-- Theorem statement
theorem taxi_ride_cost : fixed_cost + (variable_cost_per_mile * distance_traveled) = 4.10 :=
by
  sorry

end taxi_ride_cost_l139_139237


namespace tangent_line_eq_at_1_max_value_on_interval_unique_solution_exists_l139_139135

noncomputable def f (x : ℝ) : ℝ := x^3 - x
noncomputable def g (x : ℝ) : ℝ := 2 * x - 3

theorem tangent_line_eq_at_1 : 
  ∃ c : ℝ, ∀ x y : ℝ, y = f x → (x = 1 → y = 0) → y = 2 * (x - 1) → 2 * x - y - 2 = 0 := 
by sorry

theorem max_value_on_interval :
  ∃ xₘ : ℝ, (0 ≤ xₘ ∧ xₘ ≤ 2) ∧ ∀ x : ℝ, (0 ≤ x ∧ x ≤ 2) → f x ≤ 6 :=
by sorry

theorem unique_solution_exists :
  ∃! x₀ : ℝ, f x₀ = g x₀ :=
by sorry

end tangent_line_eq_at_1_max_value_on_interval_unique_solution_exists_l139_139135


namespace apple_bags_l139_139497

theorem apple_bags (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apple_bags_l139_139497


namespace wellington_population_l139_139962

theorem wellington_population 
  (W P L : ℕ)
  (h1 : P = 7 * W)
  (h2 : P = L + 800)
  (h3 : P + L = 11800) : 
  W = 900 :=
by
  sorry

end wellington_population_l139_139962


namespace total_apples_l139_139516

theorem total_apples (n : ℕ) (h1 : ∀ k : ℕ, 6 * k = n ∨ 12 * k = n) (h2 : 70 ≤ n) (h3 : n ≤ 80) : 
  n = 72 ∨ n = 78 := 
sorry

end total_apples_l139_139516


namespace right_triangle_side_length_l139_139426

theorem right_triangle_side_length (a c b : ℕ) (h₁ : a = 6) (h₂ : c = 10) (h₃ : c * c = a * a + b * b) : b = 8 :=
by {
  sorry
}

end right_triangle_side_length_l139_139426


namespace range_of_t_for_obtuse_triangle_l139_139887

def is_obtuse_triangle (a b c : ℝ) : Prop := ∃t : ℝ, a = t - 1 ∧ b = t + 1 ∧ c = t + 3

theorem range_of_t_for_obtuse_triangle :
  ∀ t : ℝ, is_obtuse_triangle (t-1) (t+1) (t+3) → (3 < t ∧ t < 7) :=
by
  intros t ht
  sorry

end range_of_t_for_obtuse_triangle_l139_139887


namespace find_y_l139_139697

theorem find_y (x : ℤ) (y : ℤ) (h : x = 5) (h1 : 3 * x = (y - x) + 4) : y = 16 :=
by
  sorry

end find_y_l139_139697


namespace sqrt_25_eq_pm_five_l139_139332

theorem sqrt_25_eq_pm_five (x : ℝ) : x^2 = 25 ↔ x = 5 ∨ x = -5 := 
sorry

end sqrt_25_eq_pm_five_l139_139332


namespace evaluate_expression_l139_139255

theorem evaluate_expression : (1 - 1/4) / (1 - 2/3) + 1/6 = 29/12 :=
by
  sorry

end evaluate_expression_l139_139255


namespace apple_count_l139_139488

-- Define constants for the problem
constant bags: ℕ → Prop
constant total_apples: ℕ
constant apples_in_bag1: ℕ := 12
constant apples_in_bag2: ℕ := 6

-- Define the conditions
axiom condition1 (n: ℕ) : bags n → (n = apples_in_bag1 ∨ n = apples_in_bag2)
axiom condition2 : 70 ≤ total_apples ∧ total_apples ≤ 80
axiom condition3 : total_apples % 6 = 0

-- State the proof problem
theorem apple_count :
  total_apples = 72 ∨ total_apples = 78 :=
sorry

end apple_count_l139_139488


namespace hexagon_angle_in_arithmetic_progression_l139_139664

theorem hexagon_angle_in_arithmetic_progression :
  ∃ (a d : ℝ), (a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) + (a + 5 * d) = 720) ∧ 
  (a = 120 ∨ a + d = 120 ∨ a + 2 * d = 120 ∨ a + 3 * d = 120 ∨ a + 4 * d = 120 ∨ a + 5 * d = 120) := by
  sorry

end hexagon_angle_in_arithmetic_progression_l139_139664


namespace minimum_flower_cost_l139_139455

def vertical_strip_width : ℝ := 3
def horizontal_strip_height : ℝ := 2
def bed_width : ℝ := 11
def bed_height : ℝ := 6

def easter_lily_cost : ℝ := 3
def dahlia_cost : ℝ := 2.5
def canna_cost : ℝ := 2

def vertical_strip_area : ℝ := vertical_strip_width * bed_height
def horizontal_strip_area : ℝ := horizontal_strip_height * bed_width
def overlap_area : ℝ := vertical_strip_width * horizontal_strip_height
def remaining_area : ℝ := (bed_width * bed_height) - vertical_strip_area - (horizontal_strip_area - overlap_area)

def easter_lily_area : ℝ := horizontal_strip_area - overlap_area
def dahlia_area : ℝ := vertical_strip_area
def canna_area : ℝ := remaining_area

def easter_lily_total_cost : ℝ := easter_lily_area * easter_lily_cost
def dahlia_total_cost : ℝ := dahlia_area * dahlia_cost
def canna_total_cost : ℝ := canna_area * canna_cost

def total_cost : ℝ := easter_lily_total_cost + dahlia_total_cost + canna_total_cost

theorem minimum_flower_cost : total_cost = 157 := by
  sorry

end minimum_flower_cost_l139_139455


namespace find_x_floor_l139_139256

theorem find_x_floor : ∃ (x : ℚ), (⌊x⌋ : ℚ) + x = 29 / 4 ∧ x = 29 / 4 := 
by
  sorry

end find_x_floor_l139_139256


namespace gcd_of_factorials_l139_139205

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Definitions for the two factorial terms involved
def term1 : ℕ := factorial 7
def term2 : ℕ := factorial 10 / factorial 4

-- Definition for computing the GCD
def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

-- The statement to prove
theorem gcd_of_factorials : gcd term1 term2 = 2520 :=
by {
  -- Here, the proof would go.
  sorry
}

end gcd_of_factorials_l139_139205


namespace line_tangent_to_ellipse_l139_139726

theorem line_tangent_to_ellipse (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + 3 * (m * x + 2)^2 = 3) ∧ 
  (∀ x1 x2 : ℝ, (2 + 3 * m^2) * x1^2 + 12 * m * x1 + 9 = 0 ∧ 
                (2 + 3 * m^2) * x2^2 + 12 * m * x2 + 9 = 0 → x1 = x2) ↔ m^2 = 2 := 
sorry

end line_tangent_to_ellipse_l139_139726


namespace cone_volume_l139_139139

theorem cone_volume (S : ℝ) (h_S : S = 12 * Real.pi) (h_lateral : ∃ r : ℝ, S = 3 * Real.pi * r^2) :
    ∃ V : ℝ, V = (8 * Real.sqrt 3 * Real.pi / 3) :=
by
  sorry

end cone_volume_l139_139139


namespace fourth_root_difference_l139_139247

theorem fourth_root_difference : (81 : ℝ) ^ (1 / 4 : ℝ) - (1296 : ℝ) ^ (1 / 4 : ℝ) = -3 :=
by
  sorry

end fourth_root_difference_l139_139247


namespace functional_equation_solution_l139_139606

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x + f (x + y)) + f (x * y) = x + f (x + y) + y * f x) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = 2 - x) :=
sorry

end functional_equation_solution_l139_139606


namespace uncle_welly_roses_l139_139198

theorem uncle_welly_roses :
  let roses_two_days_ago := 50
  let roses_yesterday := roses_two_days_ago + 20
  let roses_today := 2 * roses_two_days_ago
  roses_two_days_ago + roses_yesterday + roses_today = 220 :=
by
  let roses_two_days_ago := 50
  let roses_yesterday := roses_two_days_ago + 20
  let roses_today := 2 * roses_two_days_ago
  show roses_two_days_ago + roses_yesterday + roses_today = 220
  sorry

end uncle_welly_roses_l139_139198


namespace product_of_primes_l139_139548

theorem product_of_primes : 5 * 7 * 997 = 34895 :=
by
  sorry

end product_of_primes_l139_139548


namespace total_roses_planted_l139_139196

def roses_planted_two_days_ago := 50
def roses_planted_yesterday := roses_planted_two_days_ago + 20
def roses_planted_today := 2 * roses_planted_two_days_ago

theorem total_roses_planted :
  roses_planted_two_days_ago + roses_planted_yesterday + roses_planted_today = 220 := by
  sorry

end total_roses_planted_l139_139196


namespace number_of_male_students_l139_139101

variables (total_students sample_size female_sampled female_students male_students : ℕ)
variables (h_total : total_students = 1600)
variables (h_sample : sample_size = 200)
variables (h_female_sampled : female_sampled = 95)
variables (h_prob : (sample_size : ℚ) / total_students = (female_sampled : ℚ) / female_students)
variables (h_female_students : female_students = 760)

theorem number_of_male_students : male_students = total_students - female_students := by
  sorry

end number_of_male_students_l139_139101


namespace geometric_sequence_properties_l139_139641

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
∀ n, a (n + 1) = q * a n

theorem geometric_sequence_properties :
  ∀ (a : ℕ → ℝ),
    geometric_sequence a q →
    a 2 = 6 →
    a 5 - 2 * a 4 - a 3 + 12 = 0 →
    ∀ n, a n = 6 ∨ a n = 6 * (-1)^(n-2) ∨ a n = 6 * 2^(n-2) :=
by
  sorry

end geometric_sequence_properties_l139_139641


namespace steps_A_l139_139967

theorem steps_A (t_A t_B : ℝ) (a e t : ℝ) :
  t_A = 3 * t_B →
  t_B = t / 75 →
  a + e * t = 100 →
  75 + e * t = 100 →
  a = 75 :=
by sorry

end steps_A_l139_139967


namespace integer_roots_of_polynomial_l139_139732

theorem integer_roots_of_polynomial :
  ∀ x : ℤ, (x^3 - 3 * x^2 - 13 * x + 15 = 0) → (x = -3 ∨ x = 1 ∨ x = 5) :=
by
  sorry

end integer_roots_of_polynomial_l139_139732


namespace juliet_older_than_maggie_l139_139646

-- Definitions from the given conditions
def Juliet_age : ℕ := 10
def Ralph_age (J : ℕ) : ℕ := J + 2
def Maggie_age (R : ℕ) : ℕ := 19 - R

-- Theorem statement
theorem juliet_older_than_maggie :
  Juliet_age - Maggie_age (Ralph_age Juliet_age) = 3 :=
by
  sorry

end juliet_older_than_maggie_l139_139646


namespace quotient_is_8_l139_139128

def dividend : ℕ := 64
def divisor : ℕ := 8
def quotient := dividend / divisor

theorem quotient_is_8 : quotient = 8 := 
by 
  show quotient = 8 
  sorry

end quotient_is_8_l139_139128


namespace opposite_number_113_is_13_l139_139941

theorem opposite_number_113_is_13 :
  let n := 113
  let count := 200
  (n + 100) % count = 13 :=
by
  let n := 113
  let count := 200
  have h : (n + 100) % count = (113 + 100) % 200 := rfl
  rw h
  have result : (113 + 100) % 200 = 13 := by norm_num
  exact result

end opposite_number_113_is_13_l139_139941


namespace book_discount_l139_139821

theorem book_discount (a b : ℕ) (x y : ℕ) (h1 : x = 10 * a + b) (h2 : y = 10 * b + a) (h3 : (3 / 8) * x = y) :
  x - y = 45 := 
sorry

end book_discount_l139_139821


namespace tetrahedron_volume_eq_three_l139_139719

noncomputable def volume_of_tetrahedron : ℝ :=
  let PQ := 3
  let PR := 4
  let PS := 5
  let QR := 5
  let QS := Real.sqrt 34
  let RS := Real.sqrt 41
  have := (PQ = 3) ∧ (PR = 4) ∧ (PS = 5) ∧ (QR = 5) ∧ (QS = Real.sqrt 34) ∧ (RS = Real.sqrt 41)
  3

theorem tetrahedron_volume_eq_three : volume_of_tetrahedron = 3 := 
by { sorry }

end tetrahedron_volume_eq_three_l139_139719


namespace instantaneous_acceleration_at_3_l139_139431

def v (t : ℝ) : ℝ := t^2 + 3

theorem instantaneous_acceleration_at_3 :
  deriv v 3 = 6 :=
by
  sorry

end instantaneous_acceleration_at_3_l139_139431


namespace symmetry_axes_condition_l139_139721

/-- Define the property of having axes of symmetry for a geometric figure -/
def has_symmetry_axes (bounded : Bool) (two_parallel_axes : Bool) : Prop :=
  if bounded then 
    ¬ two_parallel_axes 
  else 
    true

/-- Main theorem stating the condition on symmetry axes for bounded and unbounded geometric figures -/
theorem symmetry_axes_condition (bounded : Bool) : 
  ∃ two_parallel_axes : Bool, has_symmetry_axes bounded two_parallel_axes :=
by
  -- The proof itself is not necessary as per the problem statement
  sorry

end symmetry_axes_condition_l139_139721


namespace marbles_before_purchase_l139_139722

-- Lean 4 statement for the problem
theorem marbles_before_purchase (bought : ℝ) (total_now : ℝ) (initial : ℝ) 
    (h1 : bought = 134.0) 
    (h2 : total_now = 321) 
    (h3 : total_now = initial + bought) : 
    initial = 187 :=
by 
    sorry

end marbles_before_purchase_l139_139722


namespace find_f_pi_over_4_l139_139402

variable (f : ℝ → ℝ)
variable (h : ∀ x, f x = f (Real.pi / 4) * Real.cos x + Real.sin x)

theorem find_f_pi_over_4 : f (Real.pi / 4) = 1 := by
  sorry

end find_f_pi_over_4_l139_139402


namespace approx_log_base_5_10_l139_139346

noncomputable def log_base (b a : ℝ) : ℝ := (Real.log a) / (Real.log b)

theorem approx_log_base_5_10 :
  let lg2 := 0.301
  let lg3 := 0.477
  let lg10 := 1
  let lg5 := lg10 - lg2
  log_base 5 10 = 10 / 7 :=
  sorry

end approx_log_base_5_10_l139_139346


namespace apples_total_l139_139533

theorem apples_total (n : ℕ) (h1 : 70 ≤ n) (h2 : n ≤ 80) (h3 : n % 6 = 0) : n = 72 ∨ n = 78 :=
sorry

end apples_total_l139_139533


namespace value_of_expression_l139_139012

theorem value_of_expression (x y : ℤ) (h1 : x = 3) (h2 : y = 2) : 3 * x - 4 * y = 1 := by
  rw [h1, h2]
  norm_num
  sorry

end value_of_expression_l139_139012


namespace problem1_problem2_problem3_problem4_l139_139114

-- Defining each problem as a theorem statement
theorem problem1 : 20 + 3 - (-27) + (-5) = 45 :=
by sorry

theorem problem2 : (-7) - (-6 + 5 / 6) + abs (-3) + 1 + 1 / 6 = 4 :=
by sorry

theorem problem3 : (1 / 4 + 3 / 8 - 7 / 12) / (1 / 24) = 1 :=
by sorry

theorem problem4 : -1 ^ 4 - (1 - 0.4) + 1 / 3 * ((-2) ^ 2 - 6) = -2 - 4 / 15 :=
by sorry

end problem1_problem2_problem3_problem4_l139_139114


namespace find_varphi_l139_139401

theorem find_varphi (φ : ℝ) (h1 : 0 < φ ∧ φ < 2 * Real.pi) 
    (h2 : ∀ x, x = 2 → Real.sin (Real.pi * x + φ) = 1) : 
    φ = Real.pi / 2 :=
-- The following is left as a proof placeholder
sorry

end find_varphi_l139_139401


namespace inequality_solution_set_impossible_l139_139057

theorem inequality_solution_set_impossible (a b : ℝ) (h_b : b ≠ 0) : ¬ (a = 0 ∧ ∀ x, ax + b > 0 ∧ x > (b / a)) :=
by {
  sorry
}

end inequality_solution_set_impossible_l139_139057


namespace inequality_proof_l139_139285

variable {m n : ℝ}

theorem inequality_proof (h1 : m < n) (h2 : n < 0) : (n / m + m / n > 2) := 
by
  sorry

end inequality_proof_l139_139285


namespace f_2017_eq_2018_l139_139089

def f (n : ℕ) : ℕ := sorry

theorem f_2017_eq_2018 (f : ℕ → ℕ) (h1 : ∀ n, f (f n) + f n = 2 * n + 3) (h2 : f 0 = 1) : f 2017 = 2018 :=
sorry

end f_2017_eq_2018_l139_139089


namespace find_a_solve_inequality_intervals_of_monotonicity_l139_139383

-- Problem 1: Prove a = 2 given conditions
theorem find_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : Real.log 3 / Real.log a > Real.log 2 / Real.log a) 
    (h₃ : Real.log (2 * a) / Real.log a - Real.log a / Real.log a = 1) : a = 2 := 
  by
  sorry

-- Problem 2: Prove the solution interval for inequality
theorem solve_inequality (x a : ℝ) (h₀ : 1 < x) (h₁ : x < 3 / 2) : 
    Real.log (x - 1) / Real.log (1 / 3) > Real.log (a - x) / Real.log (1 / 3) :=
  by
  have ha : a = 2 := sorry
  sorry

-- Problem 3: Prove intervals of monotonicity for g(x)
theorem intervals_of_monotonicity (x : ℝ) : 
  (∀ x : ℝ, 0 < x → x ≤ 2 → (|Real.log x / Real.log 2 - 1| : ℝ) = 1 - Real.log x / Real.log 2) ∧ 
  (∀ x : ℝ, x > 2 → (|Real.log x / Real.log 2 - 1| : ℝ) = Real.log x / Real.log 2 - 1) :=
  by
  sorry

end find_a_solve_inequality_intervals_of_monotonicity_l139_139383


namespace kenya_peanut_count_l139_139784

-- Define the number of peanuts Jose has
def jose_peanuts : ℕ := 85

-- Define the number of additional peanuts Kenya has more than Jose
def additional_peanuts : ℕ := 48

-- Define the number of peanuts Kenya has
def kenya_peanuts : ℕ := jose_peanuts + additional_peanuts

-- Theorem to prove the number of peanuts Kenya has
theorem kenya_peanut_count : kenya_peanuts = 133 := by
  sorry

end kenya_peanut_count_l139_139784


namespace tangent_condition_l139_139704

theorem tangent_condition (a b : ℝ) : 
    a = b → 
    (∀ x y : ℝ, (y = x + 2 → (x - a)^2 + (y - b)^2 = 2 → y = x + 2)) :=
by
  sorry

end tangent_condition_l139_139704


namespace b_10_eq_64_l139_139146

noncomputable def a (n : ℕ) : ℕ := -- Definition of the sequence a_n
  sorry

noncomputable def b (n : ℕ) : ℕ := -- Definition of the sequence b_n
  a n + a (n + 1)

theorem b_10_eq_64 (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a n * a (n + 1) = 2^n) :
  b 10 = 64 :=
sorry

end b_10_eq_64_l139_139146


namespace cement_percentage_of_second_concrete_l139_139848

theorem cement_percentage_of_second_concrete 
  (total_weight : ℝ) (final_percentage : ℝ) (partial_weight : ℝ) 
  (percentage_first_concrete : ℝ) :
  total_weight = 4500 →
  final_percentage = 0.108 →
  partial_weight = 1125 →
  percentage_first_concrete = 0.108 →
  ∃ percentage_second_concrete : ℝ, 
    percentage_second_concrete = 0.324 :=
by
  intros h1 h2 h3 h4
  let total_cement := total_weight * final_percentage
  let cement_first_concrete := partial_weight * percentage_first_concrete
  let cement_second_concrete := total_cement - cement_first_concrete
  let percentage_second_concrete := cement_second_concrete / partial_weight
  use percentage_second_concrete
  sorry

end cement_percentage_of_second_concrete_l139_139848


namespace calculate_value_l139_139742

def f (x : ℝ) : ℝ := 9 - x
def g (x : ℝ) : ℝ := x - 9

theorem calculate_value : g (f 15) = -15 := by
  sorry

end calculate_value_l139_139742


namespace oliver_final_money_l139_139949

-- Define the initial conditions as variables and constants
def initial_amount : Nat := 9
def savings : Nat := 5
def earnings : Nat := 6
def spent_frisbee : Nat := 4
def spent_puzzle : Nat := 3
def spent_stickers : Nat := 2
def movie_ticket_price : Nat := 10
def movie_ticket_discount : Nat := 20 -- 20%
def snack_price : Nat := 3
def snack_discount : Nat := 1
def birthday_gift : Nat := 8

-- Define the final amount of money Oliver has left based on the problem statement
def final_amount : Nat :=
  let total_money := initial_amount + savings + earnings
  let total_spent := spent_frisbee + spent_puzzle + spent_stickers
  let remaining_after_spending := total_money - total_spent
  let discounted_movie_ticket := movie_ticket_price * (100 - movie_ticket_discount) / 100
  let discounted_snack := snack_price - snack_discount
  let total_spent_after_discounts := discounted_movie_ticket + discounted_snack
  let remaining_after_discounts := remaining_after_spending - total_spent_after_discounts
  remaining_after_discounts + birthday_gift

-- Lean theorem statement to prove that Oliver ends up with $9
theorem oliver_final_money : final_amount = 9 := by
  sorry

end oliver_final_money_l139_139949
