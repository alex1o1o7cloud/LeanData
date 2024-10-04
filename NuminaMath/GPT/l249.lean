import Mathlib

namespace sin_300_eq_neg_sqrt3_div_2_l249_249496

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249496


namespace factor_expression_l249_249895

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249895


namespace factor_expression_l249_249901

theorem factor_expression (x : ℝ) : (x * (x + 3) + 2 * (x + 3)) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249901


namespace betty_wallet_l249_249182

theorem betty_wallet :
  let wallet_cost := 125.75
  let initial_amount := wallet_cost / 2
  let parents_contribution := 45.25
  let grandparents_contribution := 2 * parents_contribution
  let brothers_contribution := 3/4 * grandparents_contribution
  let aunts_contribution := 1/2 * brothers_contribution
  let total_amount := initial_amount + parents_contribution + grandparents_contribution + brothers_contribution + aunts_contribution
  total_amount - wallet_cost = 174.6875 :=
by
  sorry

end betty_wallet_l249_249182


namespace sin_300_eq_neg_one_half_l249_249536

-- Definitions from the conditions
def point_on_unit_circle (θ : ℝ) : ℝ × ℝ :=
  (Real.cos (θ), Real.sin (θ))

def foot_of_perpendicular (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, 0)

def is_thirty_sixty_ninety_triangle (P : ℝ × ℝ) : Prop :=
  sqrt(P.1^2 + P.2^2) = 1 ∧ P.1 = sqrt(3) / 2 ∧ P.2 = -1/2

-- Problem statement
theorem sin_300_eq_neg_one_half : Real.sin (5 * Real.pi / 3) = -1/2 := by
  sorry

end sin_300_eq_neg_one_half_l249_249536


namespace product_of_four_consecutive_integers_is_perfect_square_l249_249282

theorem product_of_four_consecutive_integers_is_perfect_square :
  ∃ k : ℤ, ∃ n : ℤ, k = (n-1) * n * (n+1) * (n+2) ∧
    k = 0 ∧
    ((n = 0) ∨ (n = -1) ∨ (n = 1) ∨ (n = -2)) :=
by
  sorry

end product_of_four_consecutive_integers_is_perfect_square_l249_249282


namespace simplify_and_evaluate_expression_l249_249704

variable (x : ℝ) (h : x = Real.sqrt 2 - 1)

theorem simplify_and_evaluate_expression : 
  (1 - 1 / (x + 1)) / (x / (x^2 + 2 * x + 1)) = Real.sqrt 2 :=
by
  -- Using the given definition of x
  have hx : x = Real.sqrt 2 - 1 := h
  
  -- Required proof should go here 
  sorry

end simplify_and_evaluate_expression_l249_249704


namespace rachel_math_homework_l249_249102

theorem rachel_math_homework (reading_hw math_hw : ℕ) 
  (h1 : reading_hw = 4) 
  (h2 : math_hw = reading_hw + 3) : 
  math_hw = 7 := by
  sorry

end rachel_math_homework_l249_249102


namespace sin_300_eq_neg_sin_60_l249_249612

theorem sin_300_eq_neg_sin_60 :
  Real.sin (300 * Real.pi / 180) = - Real.sin (60 * Real.pi / 180) :=
by
suffices : Real.sin (300 * Real.pi / 180) = -sqrt 3 / 2
  sorry

end sin_300_eq_neg_sin_60_l249_249612


namespace parallel_if_perp_to_plane_l249_249879

variable {α m n : Type}

variables (plane : α) (line_m line_n : m)

-- Define what it means for lines to be perpendicular to a plane
def perpendicular_to_plane (line : m) (pl : α) : Prop := sorry

-- Define what it means for lines to be parallel
def parallel (line1 line2 : m) : Prop := sorry

-- The conditions
axiom perp_1 : perpendicular_to_plane line_m plane
axiom perp_2 : perpendicular_to_plane line_n plane

-- The theorem to prove
theorem parallel_if_perp_to_plane : parallel line_m line_n := sorry

end parallel_if_perp_to_plane_l249_249879


namespace ratio_of_men_to_women_l249_249718

/-- Define the number of men and women on a co-ed softball team. -/
def number_of_men : ℕ := 8
def number_of_women : ℕ := 12

/--
  Given:
  1. There are 4 more women than men.
  2. The total number of players is 20.
  Prove that the ratio of men to women is 2 : 3.
-/
theorem ratio_of_men_to_women 
  (h1 : number_of_women = number_of_men + 4)
  (h2 : number_of_men + number_of_women = 20) :
  (number_of_men * 3) = (number_of_women * 2) :=
by
  have h3 : number_of_men = 8 := by sorry
  have h4 : number_of_women = 12 := by sorry
  sorry

end ratio_of_men_to_women_l249_249718


namespace largest_n_l249_249442

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def valid_n (x y : ℕ) : Prop :=
  x < 10 ∧ y < 10 ∧ x ≠ y ∧ is_prime x ∧ is_prime y ∧ is_prime (10 * y + x) ∧
  100 ≤ x * y * (10 * y + x) ∧ x * y * (10 * y + x) < 1000

theorem largest_n : ∃ x y : ℕ, valid_n x y ∧ x * y * (10 * y + x) = 777 := by
  sorry

end largest_n_l249_249442


namespace expand_product_l249_249032

theorem expand_product (x : ℝ) : (x^2 - 2*x + 2) * (x^2 + 2*x + 2) = x^4 + 4 :=
by
  sorry

end expand_product_l249_249032


namespace range_of_a_l249_249280

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem range_of_a (a : ℝ) : (∀ x ∈ Set.Icc (0 : ℝ) a, f x ≤ 3) ∧ (∃ x ∈ Set.Icc (0 : ℝ) a, f x = 3) ∧ (∀ x ∈ Set.Icc (0 : ℝ) a, f x ≥ 2) ∧ (∃ x ∈ Set.Icc (0 : ℝ) a, f x = 2) ↔ 1 ≤ a ∧ a ≤ 2 := 
by 
  sorry

end range_of_a_l249_249280


namespace rounding_no_order_l249_249069

theorem rounding_no_order (x : ℝ) (hx : x > 0) :
  let a := round (x * 100) / 100
  let b := round (x * 1000) / 1000
  let c := round (x * 10000) / 10000
  (¬((a ≥ b ∧ b ≥ c) ∨ (a ≤ b ∧ b ≤ c))) :=
sorry

end rounding_no_order_l249_249069


namespace sum_of_num_denom_repeating_decimal_l249_249729

theorem sum_of_num_denom_repeating_decimal (x : ℚ) (h1 : x = 0.24242424) : 
  (x.num + x.denom) = 41 :=
sorry

end sum_of_num_denom_repeating_decimal_l249_249729


namespace ellipse_foci_coordinates_l249_249131

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ), (x^2 / 64 + y^2 / 100 = 1) → (x = 0 ∧ (y = 6 ∨ y = -6)) :=
by
  sorry

end ellipse_foci_coordinates_l249_249131


namespace price_of_first_variety_of_oil_l249_249676

theorem price_of_first_variety_of_oil 
  (P : ℕ) 
  (x : ℕ) 
  (cost_second_variety : ℕ) 
  (volume_second_variety : ℕ)
  (cost_mixture_per_liter : ℕ) 
  : x = 160 ∧ cost_second_variety = 60 ∧ volume_second_variety = 240 ∧ cost_mixture_per_liter = 52 → P = 40 :=
by
  sorry

end price_of_first_variety_of_oil_l249_249676


namespace tournament_committee_count_l249_249969

-- Given conditions
def num_teams : ℕ := 5
def members_per_team : ℕ := 8
def committee_size : ℕ := 11
def nonhost_member_selection (n : ℕ) : ℕ := (n.choose 2) -- Selection of 2 members from non-host teams
def host_member_selection (n : ℕ) : ℕ := (n.choose 2)   -- Selection of 2 members from the remaining members of the host team; captain not considered in this choose as it's already selected

-- The total number of ways to form the required tournament committee
def total_committee_selections : ℕ :=
  num_teams * host_member_selection 7 * (nonhost_member_selection 8)^4

-- Proof stating the solution to the problem
theorem tournament_committee_count :
  total_committee_selections = 64534080 := by
  sorry

end tournament_committee_count_l249_249969


namespace percentage_sales_tax_on_taxable_purchases_l249_249803

-- Definitions
def total_cost : ℝ := 30
def tax_free_cost : ℝ := 24.7
def tax_rate : ℝ := 0.06

-- Statement to prove
theorem percentage_sales_tax_on_taxable_purchases :
  (tax_rate * (total_cost - tax_free_cost)) / total_cost * 100 = 1 := by
  sorry

end percentage_sales_tax_on_taxable_purchases_l249_249803


namespace sin_300_deg_l249_249495

noncomputable def sin_300_equals_neg_sqrt3_div_2 : Prop :=
  sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2

theorem sin_300_deg : sin_300_equals_neg_sqrt3_div_2 :=
by
  sorry

end sin_300_deg_l249_249495


namespace joe_total_spending_at_fair_l249_249858

-- Definitions based on conditions
def entrance_fee (age : ℕ) : ℝ := if age < 18 then 5 else 6
def ride_cost (rides : ℕ) : ℝ := rides * 0.5

-- Given conditions
def joe_age := 19
def twin_age := 6

def total_cost (joe_age : ℕ) (twin_age : ℕ) (rides_per_person : ℕ) :=
  entrance_fee joe_age + 2 * entrance_fee twin_age + 3 * ride_cost rides_per_person

-- The main statement to be proven
theorem joe_total_spending_at_fair : total_cost joe_age twin_age 3 = 20.5 :=
by
  sorry

end joe_total_spending_at_fair_l249_249858


namespace dot_product_eq_eight_l249_249787

def vec_a : ℝ × ℝ := (0, 4)
def vec_b : ℝ × ℝ := (2, 2)

theorem dot_product_eq_eight : (vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2) = 8 := by
  sorry

end dot_product_eq_eight_l249_249787


namespace sin_300_l249_249603

-- Definitions and conditions in Lean 4
def angle_deg := ℝ
def sin (θ : angle_deg) : ℝ := Real.sin (θ * Real.pi / 180)

noncomputable def Q_angle : angle_deg := 300
noncomputable def first_quadrant_equivalent_angle : angle_deg := 360 - 300
noncomputable def sin_60 : ℝ := Real.sin (60 * Real.pi / 180)

-- Main statement to prove
theorem sin_300 : sin 300 = -sin_60 := by
  sorry

end sin_300_l249_249603


namespace sin_300_eq_neg_sin_60_l249_249610

theorem sin_300_eq_neg_sin_60 :
  Real.sin (300 * Real.pi / 180) = - Real.sin (60 * Real.pi / 180) :=
by
suffices : Real.sin (300 * Real.pi / 180) = -sqrt 3 / 2
  sorry

end sin_300_eq_neg_sin_60_l249_249610


namespace train_length_l249_249174

theorem train_length (time_crossing : ℝ) (speed_train : ℝ) (speed_man : ℝ) (rel_speed : ℝ) (length_train : ℝ) 
    (h1 : time_crossing = 39.99680025597952)
    (h2 : speed_train = 56)
    (h3 : speed_man = 2)
    (h4 : rel_speed = (speed_train - speed_man) * (1000 / 3600))
    (h5 : length_train = rel_speed * time_crossing):
 length_train = 599.9520038396928 :=
by 
  sorry

end train_length_l249_249174


namespace negation_equiv_l249_249463

def is_even (n : ℕ) : Prop := n % 2 = 0

def exactly_one_even (a b c : ℕ) : Prop := 
  (is_even a ∧ ¬is_even b ∧ ¬is_even c) ∨ 
  (¬is_even a ∧ is_even b ∧ ¬is_even c) ∨ 
  (¬is_even a ∧ ¬is_even b ∧ is_even c)

def at_least_two_even_or_all_odd (a b c : ℕ) : Prop := 
  (is_even a ∧ is_even b) ∨ 
  (is_even a ∧ is_even c) ∨ 
  (is_even b ∧ is_even c) ∨ 
  (¬is_even a ∧ ¬is_even b ∧ ¬is_even c)
  
theorem negation_equiv (a b c : ℕ) : 
  ¬(exactly_one_even a b c) ↔ at_least_two_even_or_all_odd a b c := 
sorry

end negation_equiv_l249_249463


namespace total_travel_options_l249_249717

theorem total_travel_options (trains_A_to_B : ℕ) (ferries_B_to_C : ℕ) (flights_A_to_C : ℕ) 
  (h1 : trains_A_to_B = 3) (h2 : ferries_B_to_C = 2) (h3 : flights_A_to_C = 2) :
  (trains_A_to_B * ferries_B_to_C + flights_A_to_C = 8) :=
by
  sorry

end total_travel_options_l249_249717


namespace how_many_candies_eaten_l249_249889

variable (candies_tuesday candies_thursday candies_friday candies_left : ℕ)

def total_candies (candies_tuesday candies_thursday candies_friday : ℕ) : ℕ :=
  candies_tuesday + candies_thursday + candies_friday

theorem how_many_candies_eaten (h_tuesday : candies_tuesday = 3)
                               (h_thursday : candies_thursday = 5)
                               (h_friday : candies_friday = 2)
                               (h_left : candies_left = 4) :
  (total_candies candies_tuesday candies_thursday candies_friday) - candies_left = 6 :=
by
  sorry

end how_many_candies_eaten_l249_249889


namespace solve_problem_l249_249404

noncomputable def find_z_values (x : ℝ) : ℝ :=
  (x - 3)^2 * (x + 4) / (2 * x - 4)

theorem solve_problem (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 72) :
  find_z_values x = 64.8 ∨ find_z_values x = -10.125 :=
by
  sorry

end solve_problem_l249_249404


namespace sin_300_eq_neg_sin_60_l249_249611

theorem sin_300_eq_neg_sin_60 :
  Real.sin (300 * Real.pi / 180) = - Real.sin (60 * Real.pi / 180) :=
by
suffices : Real.sin (300 * Real.pi / 180) = -sqrt 3 / 2
  sorry

end sin_300_eq_neg_sin_60_l249_249611


namespace total_students_l249_249834

theorem total_students (S K : ℕ) (h1 : S = 4000) (h2 : K = 2 * S) :
  S + K = 12000 := by
  sorry

end total_students_l249_249834


namespace car_travel_distance_l249_249742

-- Define the conditions
def speed : ℝ := 23
def time : ℝ := 3

-- Define the formula for distance
def distance_traveled (s : ℝ) (t : ℝ) : ℝ := s * t

-- State the theorem to prove the distance the car traveled
theorem car_travel_distance : distance_traveled speed time = 69 :=
by
  -- The proof would normally go here, but we're skipping it as per the instructions
  sorry

end car_travel_distance_l249_249742


namespace expression_evaluation_l249_249634

noncomputable def evaluate_expression : ℝ :=
  (sqrt 3 * real.tan (12 * real.pi / 180) - 3) / (real.sin (12 * real.pi / 180) * (4 * real.cos (12 * real.pi / 180)^2 - 2))

theorem expression_evaluation : evaluate_expression = -2 * sqrt 3 := by
  sorry

end expression_evaluation_l249_249634


namespace professional_pay_per_hour_l249_249401

def professionals : ℕ := 2
def hours_per_day : ℕ := 6
def days : ℕ := 7
def total_cost : ℕ := 1260

theorem professional_pay_per_hour :
  (total_cost / (professionals * hours_per_day * days) = 15) :=
by
  sorry

end professional_pay_per_hour_l249_249401


namespace factors_of_2550_have_more_than_3_factors_l249_249377

theorem factors_of_2550_have_more_than_3_factors :
  ∃ n: ℕ, n = 5 ∧
    ∃ d: ℕ, d = 2550 ∧
    (∀ x < n, ∃ y: ℕ, y ∣ d ∧ (∃ z, z ∣ y ∧ z > 3)) :=
sorry

end factors_of_2550_have_more_than_3_factors_l249_249377


namespace ceil_neg_seven_fourths_cubed_eq_neg_five_l249_249198

noncomputable def ceil_of_neg_seven_fourths_cubed : ℤ :=
  Int.ceil ((-7 / 4 : ℚ)^3)

theorem ceil_neg_seven_fourths_cubed_eq_neg_five :
  ceil_of_neg_seven_fourths_cubed = -5 := by
  sorry

end ceil_neg_seven_fourths_cubed_eq_neg_five_l249_249198


namespace marcella_pairs_l249_249260

theorem marcella_pairs (pairs_initial : ℕ) (shoes_lost : ℕ) (h1 : pairs_initial = 50) (h2 : shoes_lost = 15) :
  ∃ pairs_left : ℕ, pairs_left = 35 := 
by
  existsi 35
  sorry

end marcella_pairs_l249_249260


namespace probability_penny_nickel_dime_all_heads_l249_249116

-- Define flipping five coins
def flip_five_coins : Type := (bool × bool × bool × bool × bool)

-- Define the function to check if the penny, nickel, and dime are heads
def all_heads_penny_nickel_dime (o : flip_five_coins) : bool :=
  o.1 = tt ∧ o.2 = tt ∧ o.3 = tt

-- Define the total count of possible outcomes
def total_outcomes : ℕ := 32

-- Define the count of favorable outcomes
def favorable_outcomes : ℕ := 4

-- Define the probability calculation
def probability_favorable : ℚ := favorable_outcomes / total_outcomes

-- The statement proving the probability is 1/8
theorem probability_penny_nickel_dime_all_heads :
  probability_favorable = 1 / 8 :=
by
  sorry

end probability_penny_nickel_dime_all_heads_l249_249116


namespace max_right_angle_triangles_l249_249368

open Real

theorem max_right_angle_triangles (a : ℝ) (h1 : a > 1) 
  (h2 : ∀ x y : ℝ, x^2 + a^2 * y^2 = a^2) :
  ∃n : ℕ, n = 3 := 
by
  sorry

end max_right_angle_triangles_l249_249368


namespace exponent_problem_l249_249225

variable (x m n : ℝ)
variable (h1 : x^m = 3)
variable (h2 : x^n = 5)

theorem exponent_problem : x^(2 * m - 3 * n) = 9 / 125 :=
by 
  sorry

end exponent_problem_l249_249225


namespace find_x3_y3_l249_249051

noncomputable def x_y_conditions (x y : ℝ) :=
  x - y = 3 ∧
  x^2 + y^2 = 27

theorem find_x3_y3 (x y : ℝ) (h : x_y_conditions x y) : x^3 - y^3 = 108 :=
  sorry

end find_x3_y3_l249_249051


namespace F_of_3153_max_value_of_N_l249_249036

-- Define friendly number predicate
def is_friendly (M : ℕ) : Prop :=
  let a := M / 1000
  let b := (M / 100) % 10
  let c := (M / 10) % 10
  let d := M % 10
  a - b = c - d

-- Define F(M)
def F (M : ℕ) : ℕ :=
  let a := M / 1000
  let b := (M / 100) % 10
  let s := M / 10
  let t := M % 1000
  s - t - 10 * b

-- Prove F(3153) = 152
theorem F_of_3153 : F 3153 = 152 :=
by sorry

-- Define the given predicate for N
def is_k_special (N : ℕ) : Prop :=
  let x := N / 1000
  let y := (N / 100) % 10
  let m := (N / 30) % 10
  let n := N % 10
  (N % 5 = 1) ∧ (1000 * x + 100 * y + 30 * m + n + 1001 = N) ∧
  (0 ≤ y ∧ y < x ∧ x ≤ 8) ∧ (0 ≤ m ∧ m ≤ 3) ∧ (0 ≤ n ∧ n ≤ 8) ∧ 
  is_friendly N

-- Prove the maximum value satisfying the given constraints
theorem max_value_of_N : ∀ N, is_k_special N → N ≤ 9696 :=
by sorry

end F_of_3153_max_value_of_N_l249_249036


namespace children_tickets_sold_l249_249719

theorem children_tickets_sold (A C : ℝ) (h1 : A + C = 400) (h2 : 6 * A + 4.5 * C = 2100) : C = 200 :=
sorry

end children_tickets_sold_l249_249719


namespace initial_number_of_apples_l249_249103

-- Definitions based on the conditions
def number_of_trees : ℕ := 3
def apples_picked_per_tree : ℕ := 8
def apples_left_on_trees : ℕ := 9

-- The theorem to prove
theorem initial_number_of_apples (t: ℕ := number_of_trees) (a: ℕ := apples_picked_per_tree) (l: ℕ := apples_left_on_trees) : t * a + l = 33 :=
by
  sorry

end initial_number_of_apples_l249_249103


namespace correct_sampling_method_l249_249327

-- Definitions based on conditions
def number_of_classes : ℕ := 16
def sampled_classes : ℕ := 2
def sampling_method := "Lottery then Stratified"

-- The theorem statement based on the proof problem
theorem correct_sampling_method :
  (number_of_classes = 16) ∧ (sampled_classes = 2) → (sampling_method = "Lottery then Stratified") :=
sorry

end correct_sampling_method_l249_249327


namespace find_x3_minus_y3_l249_249048

theorem find_x3_minus_y3 {x y : ℤ} (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : x^3 - y^3 = 108 :=
by 
  sorry

end find_x3_minus_y3_l249_249048


namespace find_missing_ratio_l249_249277

theorem find_missing_ratio
  (x y : ℕ)
  (h : ((2 / 3 : ℚ) * (x / y : ℚ) * (11 / 2 : ℚ) = 2)) :
  x = 6 ∧ y = 11 :=
sorry

end find_missing_ratio_l249_249277


namespace result_after_subtraction_l249_249329

theorem result_after_subtraction (x : ℕ) (h : x = 125) : 2 * x - 138 = 112 :=
by
  sorry

end result_after_subtraction_l249_249329


namespace quad_func_minimum_l249_249849

def quad_func (x : ℝ) : ℝ := x^2 - 8 * x + 5

theorem quad_func_minimum : ∀ x : ℝ, quad_func x ≥ -11 ∧ quad_func 4 = -11 :=
by
  sorry

end quad_func_minimum_l249_249849


namespace sin_300_eq_neg_sqrt_3_div_2_l249_249513

theorem sin_300_eq_neg_sqrt_3_div_2 : Real.sin (300 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt_3_div_2_l249_249513


namespace damaged_books_l249_249633

theorem damaged_books (O D : ℕ) (h1 : O = 6 * D - 8) (h2 : D + O = 69) : D = 11 :=
by
  sorry

end damaged_books_l249_249633


namespace intersection_eq_l249_249373

def setA : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
def setB : Set ℝ := {x : ℝ | x > 2 ∨ x < -1}

theorem intersection_eq : (setA ∩ setB) = {x : ℝ | 2 < x ∧ x ≤ 3} := by
  sorry

end intersection_eq_l249_249373


namespace candy_proof_l249_249707

variable (x s t : ℤ)

theorem candy_proof (H1 : 4 * x - 15 * s = 23)
                    (H2 : 5 * x - 23 * t = 15) :
  x = 302 := by
  sorry

end candy_proof_l249_249707


namespace percentage_increase_l249_249480

theorem percentage_increase (original_interval : ℕ) (new_interval : ℕ) 
  (h1 : original_interval = 30) (h2 : new_interval = 45) :
  ((new_interval - original_interval) / original_interval) * 100 = 50 := 
by 
  -- Provide the proof here
  sorry

end percentage_increase_l249_249480


namespace find_x3_y3_l249_249050

noncomputable def x_y_conditions (x y : ℝ) :=
  x - y = 3 ∧
  x^2 + y^2 = 27

theorem find_x3_y3 (x y : ℝ) (h : x_y_conditions x y) : x^3 - y^3 = 108 :=
  sorry

end find_x3_y3_l249_249050


namespace sin_300_eq_neg_sqrt_3_div_2_l249_249510

theorem sin_300_eq_neg_sqrt_3_div_2 : Real.sin (300 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt_3_div_2_l249_249510


namespace intersection_empty_l249_249410

def setA : Set ℝ := { x | x^2 - 2 * x > 0 }
def setB : Set ℝ := { x | |x + 1| < 0 }

theorem intersection_empty : setA ∩ setB = ∅ :=
by
  sorry

end intersection_empty_l249_249410


namespace system_solution_l249_249706

theorem system_solution (x y z a : ℝ) (h1 : x + y + z = 1) (h2 : 1/x + 1/y + 1/z = 1) (h3 : x * y * z = a) :
    (x = 1 ∧ y = Real.sqrt (-a) ∧ z = -Real.sqrt (-a)) ∨
    (x = 1 ∧ y = -Real.sqrt (-a) ∧ z = Real.sqrt (-a)) ∨
    (x = Real.sqrt (-a) ∧ y = -Real.sqrt (-a) ∧ z = 1) ∨
    (x = -Real.sqrt (-a) ∧ y = Real.sqrt (-a) ∧ z = 1) ∨
    (x = Real.sqrt (-a) ∧ y = 1 ∧ z = -Real.sqrt (-a)) ∨
    (x = -Real.sqrt (-a) ∧ y = 1 ∧ z = Real.sqrt (-a)) :=
sorry

end system_solution_l249_249706


namespace train_length_l249_249476

/-
  Given the speed of a train in km/hr and the time it takes to cross a pole in seconds,
  prove that the length of the train is approximately the specific value in meters.
-/
theorem train_length 
  (speed_kmph : ℝ) 
  (time_sec : ℝ) 
  (h_speed : speed_kmph = 48) 
  (h_time : time_sec = 9) : 
  (speed_kmph * 1000 / 3600) * time_sec ≈ 119.97 :=
  by {
    -- Definitions of speed in m/s and calculation of length will be here in the actual proof.
    sorry
  }

end train_length_l249_249476


namespace solution_l249_249060

theorem solution (x : ℝ) (h : 6 ∈ ({2, 4, x * x - x} : Set ℝ)) : x = 3 ∨ x = -2 := 
by 
  sorry

end solution_l249_249060


namespace limit_ln_a2n_over_an_l249_249700

noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ :=
  (x ^ n) * Real.exp (-x + n * Real.pi) / (n.factorial)

def a_n (n : ℕ) : ℝ := f_n n n

theorem limit_ln_a2n_over_an (h_pos : ∀ n : ℕ, 0 < n) :
  filter.tendsto (λ n, Real.log ((a_n (2 * n)) / (a_n n)) ^ (1 / n : ℝ)) filter.at_top (nhds (Real.pi - 1)) :=
sorry

end limit_ln_a2n_over_an_l249_249700


namespace total_pots_needed_l249_249183

theorem total_pots_needed
    (p : ℕ) (s : ℕ) (h : ℕ)
    (hp : p = 5)
    (hs : s = 3)
    (hh : h = 4) :
    p * s * h = 60 := by
  sorry

end total_pots_needed_l249_249183


namespace sin_300_eq_neg_sqrt3_div_2_l249_249573

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249573


namespace bob_correct_answer_l249_249184

theorem bob_correct_answer (y : ℕ) (h : (y - 7) / 5 = 47) : (y - 5) / 7 = 33 :=
by 
  -- assumption h and the statement to prove
  sorry

end bob_correct_answer_l249_249184


namespace total_students_l249_249831

def numStudents (skiing scavenger : ℕ) : ℕ :=
  skiing + scavenger

theorem total_students (skiing scavenger : ℕ) (h1 : skiing = 2 * scavenger) (h2 : scavenger = 4000) :
  numStudents skiing scavenger = 12000 :=
by
  sorry

end total_students_l249_249831


namespace penguins_seals_ratio_l249_249721

theorem penguins_seals_ratio (t_total t_seals t_elephants t_penguins : ℕ) 
    (h1 : t_total = 130) 
    (h2 : t_seals = 13) 
    (h3 : t_elephants = 13) 
    (h4 : t_penguins = t_total - t_seals - t_elephants) : 
    (t_penguins / t_seals = 8) := by
  sorry

end penguins_seals_ratio_l249_249721


namespace wooden_parallelepiped_length_l249_249003

theorem wooden_parallelepiped_length (n : ℕ) (h1 : n ≥ 7)
    (h2 : ∀ total_cubes unpainted_cubes : ℕ,
      total_cubes = n * (n - 2) * (n - 4) ∧
      unpainted_cubes = (n - 2) * (n - 4) * (n - 6) ∧
      unpainted_cubes = 2 / 3 * total_cubes) :
  n = 18 := 
sorry

end wooden_parallelepiped_length_l249_249003


namespace longest_line_segment_l249_249420

theorem longest_line_segment (total_length_cm : ℕ) (h : total_length_cm = 3000) :
  ∃ n : ℕ, 2 * (n * (n + 1) / 2) ≤ total_length_cm ∧ n = 54 :=
by
  use 54
  sorry

end longest_line_segment_l249_249420


namespace inequality_proof_l249_249809

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b ≥ 1) :
  (a + 2 * b + 2 / (a + 1)) * (b + 2 * a + 2 / (b + 1)) ≥ 16 :=
by
  sorry

end inequality_proof_l249_249809


namespace cost_per_ticket_l249_249339

/-- Adam bought 13 tickets and after riding the ferris wheel, he had 4 tickets left.
    He spent 81 dollars riding the ferris wheel, and we want to determine how much each ticket cost. -/
theorem cost_per_ticket (initial_tickets : ℕ) (tickets_left : ℕ) (total_cost : ℕ) (used_tickets : ℕ) 
    (ticket_cost : ℕ) (h1 : initial_tickets = 13) 
    (h2 : tickets_left = 4) 
    (h3 : total_cost = 81) 
    (h4 : used_tickets = initial_tickets - tickets_left) 
    (h5 : ticket_cost = total_cost / used_tickets) : ticket_cost = 9 :=
by {
    sorry
}

end cost_per_ticket_l249_249339


namespace compute_sum_of_squares_l249_249096

noncomputable def polynomial_roots (p q r : ℂ) : Prop := 
  (p^3 - 15 * p^2 + 22 * p - 8 = 0) ∧ 
  (q^3 - 15 * q^2 + 22 * q - 8 = 0) ∧ 
  (r^3 - 15 * r^2 + 22 * r - 8 = 0) 

theorem compute_sum_of_squares (p q r : ℂ) (h : polynomial_roots p q r) :
  (p + q) ^ 2 + (q + r) ^ 2 + (r + p) ^ 2 = 406 := 
sorry

end compute_sum_of_squares_l249_249096


namespace total_pieces_of_bread_correct_l249_249457

-- Define the constants for the number of bread pieces needed per type of sandwich
def pieces_per_regular_sandwich : ℕ := 2
def pieces_per_double_meat_sandwich : ℕ := 3

-- Define the quantities of each type of sandwich
def regular_sandwiches : ℕ := 14
def double_meat_sandwiches : ℕ := 12

-- Define the total pieces of bread calculation
def total_pieces_of_bread : ℕ := pieces_per_regular_sandwich * regular_sandwiches + pieces_per_double_meat_sandwich * double_meat_sandwiches

-- State the theorem
theorem total_pieces_of_bread_correct : total_pieces_of_bread = 64 :=
by
  -- Proof goes here (using sorry for now)
  sorry

end total_pieces_of_bread_correct_l249_249457


namespace x_intercept_of_line_l249_249310

def point1 := (10, 3)
def point2 := (-12, -8)

theorem x_intercept_of_line :
  let m := (point2.snd - point1.snd) / (point2.fst - point1.fst)
  let line_eq (x : ℝ) := m * (x - point1.fst) + point1.snd
  ∃ x : ℝ, line_eq x = 0 ∧ x = 4 :=
by
  sorry

end x_intercept_of_line_l249_249310


namespace evaluate_expression_right_to_left_l249_249246

variable (a b c d : ℝ)

theorem evaluate_expression_right_to_left:
  (a * b + c - d) = (a * (b + c - d)) :=
by {
  -- Group operations from right to left according to the given condition
  sorry
}

end evaluate_expression_right_to_left_l249_249246


namespace equivalent_proposition_l249_249448

variable (M : Set α) (m n : α)

theorem equivalent_proposition :
  (m ∈ M → n ∉ M) ↔ (n ∈ M → m ∉ M) := by
  sorry

end equivalent_proposition_l249_249448


namespace train_crossing_time_l249_249474

/-- A train 400 m long traveling at a speed of 36 km/h crosses an electric pole in 40 seconds. -/
theorem train_crossing_time (length : ℝ) (speed_kmph : ℝ) (speed_mps : ℝ) (time : ℝ) 
  (h1 : length = 400)
  (h2 : speed_kmph = 36)
  (h3 : speed_mps = speed_kmph * 1000 / 3600)
  (h4 : time = length / speed_mps) :
  time = 40 :=
by {
  sorry
}

end train_crossing_time_l249_249474


namespace gray_area_correct_l249_249695

-- Define the side lengths of the squares
variable (a b : ℝ)

-- Define the areas of the larger and smaller squares
def area_large_square : ℝ := (a + b) * (a + b)
def area_small_square : ℝ := a * a

-- Define the gray area
def gray_area : ℝ := area_large_square a b - area_small_square a

-- The proof statement
theorem gray_area_correct (a b : ℝ) : gray_area a b = 2 * a * b + b ^ 2 := by
  sorry

end gray_area_correct_l249_249695


namespace abc_over_sum_leq_four_thirds_l249_249806

theorem abc_over_sum_leq_four_thirds (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) 
  (h_a_leq_2 : a ≤ 2) (h_b_leq_2 : b ≤ 2) (h_c_leq_2 : c ≤ 2) :
  (abc / (a + b + c) ≤ 4/3) :=
by
  sorry

end abc_over_sum_leq_four_thirds_l249_249806


namespace max_n_for_coloring_l249_249848

noncomputable def maximum_n : ℕ :=
  11

theorem max_n_for_coloring :
  ∃ n : ℕ, (n = maximum_n) ∧ ∀ k ∈ Finset.range n, 
  (∃ x y : ℕ, 1 ≤ x ∧ x ≤ 14 ∧ 1 ≤ y ∧ y ≤ 14 ∧ (x - y = k ∨ y - x = k) ∧ x ≠ y) ∧
  (∃ a b : ℕ, 1 ≤ a ∧ a ≤ 14 ∧ 1 ≤ b ∧ b ≤ 14 ∧ (a - b = k ∨ b - a = k) ∧ a ≠ b) :=
sorry

end max_n_for_coloring_l249_249848


namespace combined_total_circles_squares_l249_249126

-- Define the problem parameters based on conditions
def US_stars : ℕ := 50
def US_stripes : ℕ := 13
def circles (n : ℕ) : ℕ := (n / 2) - 3
def squares (n : ℕ) : ℕ := (n * 2) + 6

-- Prove that the combined number of circles and squares on Pete's flag is 54
theorem combined_total_circles_squares : 
    circles US_stars + squares US_stripes = 54 := by
  sorry

end combined_total_circles_squares_l249_249126


namespace percentage_equivalence_l249_249235

theorem percentage_equivalence (x : ℝ) :
  (70 / 100) * 600 = (x / 100) * 1050 → x = 40 :=
by
  sorry

end percentage_equivalence_l249_249235


namespace total_trip_time_l249_249178

noncomputable def speed_coastal := 10 / 20  -- miles per minute
noncomputable def speed_highway := 4 * speed_coastal  -- miles per minute
noncomputable def time_highway := 50 / speed_highway  -- minutes
noncomputable def total_time := 20 + time_highway  -- minutes

theorem total_trip_time : total_time = 45 := 
by
  -- Proof omitted
  sorry

end total_trip_time_l249_249178


namespace parabola_tangency_point_l249_249773

-- Definitions of the parabola equations
def parabola1 (x : ℝ) : ℝ := x^2 + 10 * x + 20
def parabola2 (y : ℝ) : ℝ := y^2 + 36 * y + 380

-- The proof statement
theorem parabola_tangency_point : 
  ∃ (x y : ℝ), 
    parabola1 x = y ∧ parabola2 y = x ∧ x = -9 / 2 ∧ y = -35 / 2 :=
by
  sorry

end parabola_tangency_point_l249_249773


namespace triangle_side_lengths_l249_249208

open Real

theorem triangle_side_lengths (a b c : ℕ) (R : ℝ)
    (h1 : a * a + 4 * d * d = 2500)
    (h2 : b * b + 4 * e * e = 2500)
    (h3 : R = 12.5)
    (h4 : (2:ℝ) * d ≤ a)
    (h5 : (2:ℝ) * e ≤ b)
    (h6 : a > b)
    (h7 : a ≠ b)
    (h8 : 2 * R = 25) :
    (a, b, c) = (15, 7, 20) := by
  sorry

end triangle_side_lengths_l249_249208


namespace population_net_increase_in_one_day_l249_249854

-- Definitions based on the conditions
def birth_rate_per_two_seconds : ℝ := 4
def death_rate_per_two_seconds : ℝ := 3
def seconds_in_a_day : ℝ := 86400

-- The main theorem to prove
theorem population_net_increase_in_one_day : 
  (birth_rate_per_two_seconds / 2 - death_rate_per_two_seconds / 2) * seconds_in_a_day = 43200 :=
by
  sorry

end population_net_increase_in_one_day_l249_249854


namespace adam_first_half_correct_l249_249308

-- Define the conditions
def second_half_correct := 2
def points_per_question := 8
def final_score := 80

-- Define the number of questions Adam answered correctly in the first half
def first_half_correct :=
  (final_score - (second_half_correct * points_per_question)) / points_per_question

-- Statement to prove
theorem adam_first_half_correct : first_half_correct = 8 :=
by
  -- skipping the proof
  sorry

end adam_first_half_correct_l249_249308


namespace parallelepiped_length_l249_249006

theorem parallelepiped_length (n : ℕ)
  (h1 : ∃ n : ℕ, n = 18) 
  (h2 : one_third_of_cubes_have_red_faces : (∃ k : ℕ, k = ((n * (n - 2) * (n - 4)) / 3)) 
        ∧ (remaining_unpainted_cubes : (∃ m : ℕ , m = (2 * (n * (n - 2) * (n - 4)) / 3))))
  (h3 : painted_and_cut_into_cubes : (∃ a b c : ℕ, a = n ∧ b = (n - 2) ∧ c = (n - 4)))
  (h4 : all_sides_whole_cm : (∃ d : ℕ , d = n ∧ d = (n - 2) ∧ d = (n - 4))) :
  n = 18 :=
begin
  sorry
end

end parallelepiped_length_l249_249006


namespace number_of_months_to_fully_pay_off_car_l249_249416

def total_price : ℕ := 13380
def initial_payment : ℕ := 5400
def monthly_payment : ℕ := 420

theorem number_of_months_to_fully_pay_off_car :
  (total_price - initial_payment) / monthly_payment = 19 :=
by
  sorry

end number_of_months_to_fully_pay_off_car_l249_249416


namespace range_of_a_l249_249931

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ Real.pi / 2 →
    (x + 3 + 2 * Real.sin θ * Real.cos θ) ^ 2 +
    (x + a * Real.sin θ + a * Real.cos θ) ^ 2 ≥ 1 / 8) ↔
  (a ≥ 7 / 2 ∨ a ≤ Real.sqrt 6) :=
by
  sorry

end range_of_a_l249_249931


namespace tangent_lines_through_point_l249_249351

theorem tangent_lines_through_point (x y : ℝ) :
  (x^2 + y^2 + 2*x - 2*y + 1 = 0) ∧ (x = -2 ∨ (15*x + 8*y - 10 = 0)) ↔ 
  (x = -2 ∨ (15*x + 8*y - 10 = 0)) :=
by
  sorry

end tangent_lines_through_point_l249_249351


namespace sin_300_eq_neg_sqrt3_div_2_l249_249581

noncomputable def sin_300 : ℝ := sin (300 * real.pi / 180)

theorem sin_300_eq_neg_sqrt3_div_2 : 
  (∠300° = 360° - 60°) →
  (300° ∈ fourth_quadrant) →
  (∀ θ ∈ fourth_quadrant, sin θ < 0) →
  (sin (60 * real.pi / 180) = sqrt 3 / 2) →
  sin_300 = -sqrt 3 / 2 := 
by
  intros h1 h2 h3 h4
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249581


namespace additional_kgs_l249_249325

variables (P R A : ℝ)
variables (h1 : R = 0.80 * P) (h2 : R = 34.2) (h3 : 684 = A * R)

theorem additional_kgs :
  A = 20 :=
by
  sorry

end additional_kgs_l249_249325


namespace arithmetic_sequence_mod_12_l249_249063

theorem arithmetic_sequence_mod_12 (n : ℕ) (h1 : 2 + 8 + 14 + 20 + 26 + ... + 128 + 134 ≡ n \pmod{12}) (h2 : 0 ≤ n ∧ n < 12) : n = 0 := 
by
  sorry

end arithmetic_sequence_mod_12_l249_249063


namespace sin_300_eq_neg_one_half_l249_249538

-- Definitions from the conditions
def point_on_unit_circle (θ : ℝ) : ℝ × ℝ :=
  (Real.cos (θ), Real.sin (θ))

def foot_of_perpendicular (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, 0)

def is_thirty_sixty_ninety_triangle (P : ℝ × ℝ) : Prop :=
  sqrt(P.1^2 + P.2^2) = 1 ∧ P.1 = sqrt(3) / 2 ∧ P.2 = -1/2

-- Problem statement
theorem sin_300_eq_neg_one_half : Real.sin (5 * Real.pi / 3) = -1/2 := by
  sorry

end sin_300_eq_neg_one_half_l249_249538


namespace remainder_equality_l249_249111

theorem remainder_equality (a b s t d : ℕ) (h1 : a > b) (h2 : a % d = s % d) (h3 : b % d = t % d) :
  ((a + 1) * (b + 1)) % d = ((s + 1) * (t + 1)) % d :=
by
  sorry

end remainder_equality_l249_249111


namespace coin_flip_heads_probability_l249_249118

theorem coin_flip_heads_probability :
  let coins : List String := ["penny", "nickel", "dime", "quarter", "half-dollar"]
  let independent_event (coin : String) : Prop := True
  let outcomes := (2 : ℕ) ^ (List.length coins)
  let successful_outcomes := 4
  let probability := successful_outcomes / outcomes
  probability = 1 / 8 := 
by
  sorry

end coin_flip_heads_probability_l249_249118


namespace repeating_decimal_sum_l249_249728

theorem repeating_decimal_sum (x : ℚ) (h : x = 24/99) :
  let num_denom_sum := (8 + 33) in num_denom_sum = 41 :=
by
  sorry

end repeating_decimal_sum_l249_249728


namespace gcd_square_of_difference_l249_249095

theorem gcd_square_of_difference (x y z : ℕ) (h : 1/x - 1/y = 1/z) :
  ∃ k : ℕ, (Nat.gcd (Nat.gcd x y) z) * (y - x) = k^2 :=
by
  sorry

end gcd_square_of_difference_l249_249095


namespace inequality_solution_l249_249650

theorem inequality_solution (a : ℝ) (x : ℝ) 
  (h₁ : 0 < a) 
  (h₂ : 1 < a) 
  (y₁ : ℝ := a^(2 * x + 1)) 
  (y₂ : ℝ := a^(-3 * x)) :
  y₁ > y₂ → x > - (1 / 5) :=
by
  sorry

end inequality_solution_l249_249650


namespace dryer_cost_l249_249877

theorem dryer_cost (washer_dryer_total_cost washer_cost dryer_cost : ℝ) (h1 : washer_dryer_total_cost = 1200) (h2 : washer_cost = dryer_cost + 220) :
  dryer_cost = 490 :=
by
  sorry

end dryer_cost_l249_249877


namespace base_conversion_proof_l249_249143

-- Definitions of the base-converted numbers
def b1463_7 := 3 * 7^0 + 6 * 7^1 + 4 * 7^2 + 1 * 7^3  -- 1463 in base 7
def b121_5 := 1 * 5^0 + 2 * 5^1 + 1 * 5^2  -- 121 in base 5
def b1754_6 := 4 * 6^0 + 5 * 6^1 + 7 * 6^2 + 1 * 6^3  -- 1754 in base 6
def b3456_7 := 6 * 7^0 + 5 * 7^1 + 4 * 7^2 + 3 * 7^3  -- 3456 in base 7

-- Formalizing the proof goal
theorem base_conversion_proof : (b1463_7 / b121_5 : ℤ) - b1754_6 * 2 + b3456_7 = 278 := by
  sorry  -- Proof is omitted

end base_conversion_proof_l249_249143


namespace bank1_more_advantageous_l249_249265

-- Define the quarterly interest rate for Bank 1
def bank1_quarterly_rate : ℝ := 0.8

-- Define the annual interest rate for Bank 2
def bank2_annual_rate : ℝ := 9.0

-- Define the annual compounded interest rate for Bank 1
def bank1_annual_yield : ℝ :=
  (1 + bank1_quarterly_rate) ^ 4

-- Define the annual rate directly for Bank 2
def bank2_annual_yield : ℝ :=
  1 + bank2_annual_rate

-- The theorem stating that Bank 1 is more advantageous than Bank 2
theorem bank1_more_advantageous : bank1_annual_yield > bank2_annual_yield :=
  sorry

end bank1_more_advantageous_l249_249265


namespace sin_300_eq_neg_sqrt3_div_2_l249_249591

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249591


namespace factor_expression_l249_249915

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := 
by
  sorry

end factor_expression_l249_249915


namespace total_miles_traveled_l249_249085

noncomputable def initial_fee : ℝ := 2.0
noncomputable def charge_per_2_5_mile : ℝ := 0.35
noncomputable def total_charge : ℝ := 5.15

theorem total_miles_traveled :
  ∃ (miles : ℝ), total_charge = initial_fee + (charge_per_2_5_mile * miles * (5 / 2)) ∧ miles = 3.6 :=
by
  sorry

end total_miles_traveled_l249_249085


namespace isabel_ds_games_left_l249_249248

-- Define the initial number of DS games Isabel had
def initial_ds_games : ℕ := 90

-- Define the number of DS games Isabel gave to her friend
def ds_games_given : ℕ := 87

-- Define a function to calculate the remaining DS games
def remaining_ds_games (initial : ℕ) (given : ℕ) : ℕ := initial - given

-- Statement of the theorem we need to prove
theorem isabel_ds_games_left : remaining_ds_games initial_ds_games ds_games_given = 3 := by
  sorry

end isabel_ds_games_left_l249_249248


namespace intersection_eq_l249_249290

open Set

noncomputable def A : Set ℝ := {x : ℝ | x^2 > 4}
noncomputable def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}

theorem intersection_eq : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by
  sorry

end intersection_eq_l249_249290


namespace find_a_l249_249671

noncomputable def binomialExpansion (a : ℚ) (x : ℚ) := (x - a / x) ^ 6

theorem find_a (a : ℚ) (A : ℚ) (B : ℚ) (hA : A = 15 * a ^ 2) (hB : B = -20 * a ^ 3) (hB_value : B = 44) :
  a = -22 / 5 :=
by
  sorry -- skipping the proof

end find_a_l249_249671


namespace descent_time_l249_249100

-- Definitions based on conditions
def time_to_top : ℝ := 4
def avg_speed_up : ℝ := 2.625
def avg_speed_total : ℝ := 3.5
def distance_to_top : ℝ := avg_speed_up * time_to_top -- 10.5 km
def total_distance : ℝ := 2 * distance_to_top       -- 21 km

-- Theorem statement: the time to descend (t_down) should be 2 hours
theorem descent_time (t_down : ℝ) : 
  avg_speed_total * (time_to_top + t_down) = total_distance →
  t_down = 2 := 
by 
  -- skip the proof
  sorry

end descent_time_l249_249100


namespace least_num_subtracted_l249_249154

theorem least_num_subtracted 
  {x : ℤ} 
  (h5 : (642 - x) % 5 = 4) 
  (h7 : (642 - x) % 7 = 4) 
  (h9 : (642 - x) % 9 = 4) : 
  x = 4 := 
sorry

end least_num_subtracted_l249_249154


namespace factor_expression_l249_249923

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249923


namespace dimension_tolerance_l249_249999

theorem dimension_tolerance (base_dim : ℝ) (pos_tolerance : ℝ) (neg_tolerance : ℝ) 
  (max_dim : ℝ) (min_dim : ℝ) 
  (h_base : base_dim = 7) 
  (h_pos_tolerance : pos_tolerance = 0.05) 
  (h_neg_tolerance : neg_tolerance = 0.02) 
  (h_max_dim : max_dim = base_dim + pos_tolerance) 
  (h_min_dim : min_dim = base_dim - neg_tolerance) :
  max_dim = 7.05 ∧ min_dim = 6.98 :=
by
  sorry

end dimension_tolerance_l249_249999


namespace sin_300_eq_neg_sqrt3_div_2_l249_249528

def point_on_unit_circle (θ : ℝ) : ℝ × ℝ :=
⟨Real.cos θ, Real.sin θ⟩

theorem sin_300_eq_neg_sqrt3_div_2 :
  let Q := point_on_unit_circle (300 * Real.pi / 180) in
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
by sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249528


namespace total_registration_methods_l249_249762

theorem total_registration_methods (n : ℕ) (h : n = 5) : (2 ^ n) = 32 :=
by
  sorry

end total_registration_methods_l249_249762


namespace speed_ratio_l249_249994

variable (d_A d_B : ℝ) (t_A t_B : ℝ)

-- Define the conditions
def condition1 : Prop := d_A = (1 + 1/5) * d_B
def condition2 : Prop := t_B = (1 - 1/11) * t_A

-- State the theorem that the speed ratio is 12:11
theorem speed_ratio (h1 : condition1 d_A d_B) (h2 : condition2 t_A t_B) :
  (d_A / t_A) / (d_B / t_B) = 12 / 11 :=
sorry

end speed_ratio_l249_249994


namespace magic_square_sum_l249_249970

theorem magic_square_sum (x y z w v: ℕ) (h1: 27 + w + 22 = 49 + w)
  (h2: 27 + 18 + x = 45 + x) (h3: 22 + 24 + y = 46 + y)
  (h4: 49 + w = 46 + y) (hw: w = y - 3) (hx: x = y + 1)
  (hz: z = x + 3) : x + z = 45 :=
by {
  sorry
}

end magic_square_sum_l249_249970


namespace characteristic_function_expansion_l249_249701

noncomputable def norm {α : Type*} [NormedField α] (x : α) := ∥x∥

theorem characteristic_function_expansion 
  {X : Type*} [measurable_space X] [normed_group X] [borel_space X]
  (X : X → ℝ) (n : ℕ) (t : fin n → ℝ) 
  (h1 : E (λ X, (∥X∥)^n) < ∞) :
  ∃ o : (ℝ^n → ℝ) → (ℝ^n → ℝ),
  ∃ C : ℝ,
    (∀ t : fin n → ℝ, abs (φ(t) - ∑ k in range n, (i^k • (E (λ X, (t, X)^k)) / k!)) ≤ C * ∥t∥ ^ n) ∧
    (∥t∥ → 0) → (o(1) * ∥t∥^n = 0) := sorry

end characteristic_function_expansion_l249_249701


namespace permutations_count_l249_249807

theorem permutations_count :
  let b : Fin 15 → Fin 15 := λ i => i + 1
  let condition1 := b 1 > b 2 ∧ b 2 > b 3 ∧ b 3 > b 4 ∧ b 4 > b 5 ∧ b 5 > b 6 ∧ b 6 > b 7 ∧ b 7 > b 8
  let condition2 := b 8 < b 9 ∧ b 9 < b 10 ∧ b 10 < b 11 ∧ b 11 < b 12 ∧ b 12 < b 13 ∧ b 13 < b 14
  let conditions := condition1 ∧ condition2
  ∃ b' : (Fin 15 → Fin 15), conditions b' ∧ (Fin 15) (1, 2, 3, ..., 13) = 1716 :=
sorry

end permutations_count_l249_249807


namespace large_rectangle_perimeter_l249_249164

-- Definitions from the conditions
def side_length_of_square (perimeter_square : ℕ) : ℕ := perimeter_square / 4
def width_of_small_rectangle (perimeter_rect : ℕ) (side_length : ℕ) : ℕ := (perimeter_rect / 2) - side_length

-- Given conditions
def perimeter_square := 24
def perimeter_rect := 16
def side_length := side_length_of_square perimeter_square
def rect_width := width_of_small_rectangle perimeter_rect side_length
def large_rectangle_height := side_length + rect_width
def large_rectangle_width := 3 * side_length

-- Perimeter calculation
def perimeter_large_rectangle (width height : ℕ) : ℕ := 2 * (width + height)

-- Proof problem statement
theorem large_rectangle_perimeter : 
  perimeter_large_rectangle large_rectangle_width large_rectangle_height = 52 :=
sorry

end large_rectangle_perimeter_l249_249164


namespace salt_percentage_in_first_solution_l249_249755

theorem salt_percentage_in_first_solution
    (S : ℝ)
    (h1 : ∀ w : ℝ, w ≥ 0 → ∃ q : ℝ, q = w)  -- One fourth of the first solution was replaced by the second solution
    (h2 : ∀ w1 w2 w3 : ℝ,
            w1 + w2 = w3 →
            (w1 / w3 * S + w2 / w3 * 25 = 16)) :  -- Resulting solution was 16 percent salt by weight
  S = 13 :=   -- Correct answer
sorry

end salt_percentage_in_first_solution_l249_249755


namespace weight_of_empty_box_l249_249129

theorem weight_of_empty_box (w12 w8 w : ℝ) (h1 : w12 = 11.48) (h2 : w8 = 8.12) (h3 : ∀ b : ℕ, b > 0 → w = 0.84) :
  w8 - 8 * w = 1.40 :=
by
  sorry

end weight_of_empty_box_l249_249129


namespace susan_ate_6_candies_l249_249883

-- Definitions for the conditions
def candies_tuesday : ℕ := 3
def candies_thursday : ℕ := 5
def candies_friday : ℕ := 2
def candies_left : ℕ := 4

-- The total number of candies bought during the week
def total_candies_bought : ℕ := candies_tuesday + candies_thursday + candies_friday

-- The number of candies Susan ate during the week
def candies_eaten : ℕ := total_candies_bought - candies_left

-- Theorem statement
theorem susan_ate_6_candies : candies_eaten = 6 :=
by
  unfold candies_eaten total_candies_bought candies_tuesday candies_thursday candies_friday candies_left
  sorry

end susan_ate_6_candies_l249_249883


namespace solve_prime_equation_l249_249254

theorem solve_prime_equation (x y : ℕ) (p : ℕ) (hp : Nat.Prime p) :
  x^3 + y^3 - 3 * x * y = p - 1 ↔
  (x = 1 ∧ y = 0 ∧ p = 2) ∨
  (x = 0 ∧ y = 1 ∧ p = 2) ∨
  (x = 2 ∧ y = 2 ∧ p = 5) := 
sorry

end solve_prime_equation_l249_249254


namespace total_cost_of_purchase_l249_249428

theorem total_cost_of_purchase :
  let sandwich_cost := 3
  let soda_cost := 2
  let num_sandwiches := 5
  let num_sodas := 8
  let total_cost := (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost)
  total_cost = 31 :=
by
  sorry

end total_cost_of_purchase_l249_249428


namespace custom_operation_example_l249_249065

def custom_operation (x y : Int) : Int :=
  x * y - 3 * x

theorem custom_operation_example : (custom_operation 7 4) - (custom_operation 4 7) = -9 := by
  sorry

end custom_operation_example_l249_249065


namespace race_last_part_length_l249_249433

theorem race_last_part_length (total_len first_part second_part third_part last_part : ℝ) 
  (h1 : total_len = 74.5) 
  (h2 : first_part = 15.5) 
  (h3 : second_part = 21.5) 
  (h4 : third_part = 21.5) :
  last_part = total_len - (first_part + second_part + third_part) → last_part = 16 :=
by {
  intros,
  sorry
}

end race_last_part_length_l249_249433


namespace total_balloons_correct_l249_249791

-- Definitions based on the conditions
def brookes_initial_balloons : Nat := 12
def brooke_additional_balloons : Nat := 8

def tracys_initial_balloons : Nat := 6
def tracy_additional_balloons : Nat := 24

-- Calculate the number of balloons each person has after the additions and Tracy popping half
def brookes_final_balloons : Nat := brookes_initial_balloons + brooke_additional_balloons
def tracys_balloons_after_addition : Nat := tracys_initial_balloons + tracy_additional_balloons
def tracys_final_balloons : Nat := tracys_balloons_after_addition / 2

-- Total number of balloons
def total_balloons : Nat := brookes_final_balloons + tracys_final_balloons

-- The proof statement
theorem total_balloons_correct : total_balloons = 35 := by
  -- Proof would go here (but we'll skip with sorry)
  sorry

end total_balloons_correct_l249_249791


namespace scientific_notation_of_153000_l249_249751

theorem scientific_notation_of_153000 :
  ∃ (a : ℝ) (n : ℤ), 153000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 1.53 ∧ n = 5 := 
by
  sorry

end scientific_notation_of_153000_l249_249751


namespace car_payment_months_l249_249414

theorem car_payment_months 
    (total_price : ℕ) 
    (initial_payment : ℕ)
    (monthly_payment : ℕ) 
    (h_total_price : total_price = 13380) 
    (h_initial_payment : initial_payment = 5400) 
    (h_monthly_payment : monthly_payment = 420) 
    : total_price - initial_payment = 7980 
    ∧ (total_price - initial_payment) / monthly_payment = 19 := 
by 
  sorry

end car_payment_months_l249_249414


namespace min_trips_required_l249_249314

def masses : List ℕ := [150, 62, 63, 66, 70, 75, 79, 84, 95, 96, 99]
def load_capacity : ℕ := 190

theorem min_trips_required :
  ∃ (trips : ℕ), 
  (∀ partition : List (List ℕ), (∀ group : List ℕ, group ∈ partition → 
  group.sum ≤ load_capacity) ∧ partition.join = masses → 
  partition.length ≥ 6) :=
sorry

end min_trips_required_l249_249314


namespace min_trips_correct_l249_249316

-- Define the masses of the individuals and the elevator capacity as constants
def masses : List ℕ := [150, 62, 63, 66, 70, 75, 79, 84, 95, 96, 99]
def elevator_capacity : ℕ := 190

-- Define a function that computes the minimum number of trips required to transport all individuals
noncomputable def min_trips (masses : List ℕ) (capacity : ℕ) : ℕ := sorry

-- State the theorem to be proven
theorem min_trips_correct :
  min_trips masses elevator_capacity = 6 := sorry

end min_trips_correct_l249_249316


namespace intersection_complement_eq_l249_249061

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 4}

theorem intersection_complement_eq : A ∩ (U \ B) = {0, 2} := by
  sorry

end intersection_complement_eq_l249_249061


namespace sin_300_deg_l249_249494

noncomputable def sin_300_equals_neg_sqrt3_div_2 : Prop :=
  sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2

theorem sin_300_deg : sin_300_equals_neg_sqrt3_div_2 :=
by
  sorry

end sin_300_deg_l249_249494


namespace units_produced_today_l249_249776

theorem units_produced_today (n : ℕ) (X : ℕ) 
  (h1 : n = 9) 
  (h2 : (360 + X) / (n + 1) = 45) 
  (h3 : 40 * n = 360) : 
  X = 90 := 
sorry

end units_produced_today_l249_249776


namespace sum_of_first_six_primes_l249_249151

theorem sum_of_first_six_primes : (2 + 3 + 5 + 7 + 11 + 13) = 41 :=
by
  sorry

end sum_of_first_six_primes_l249_249151


namespace binom_19_10_l249_249342

theorem binom_19_10 (h₁ : nat.choose 17 7 = 19448) (h₂ : nat.choose 17 9 = 24310) : nat.choose 19 10 = 92378 := by
  sorry

end binom_19_10_l249_249342


namespace monkeys_bananas_l249_249845

theorem monkeys_bananas (c₁ c₂ c₃ : ℕ) (h1 : ∀ (k₁ k₂ k₃ : ℕ), k₁ = c₁ → k₂ = c₂ → k₃ = c₃ → 4 * (k₁ / 3 + k₂ / 6 + k₃ / 18) = 2 * (k₁ / 6 + k₂ / 3 + k₃ / 18) ∧ 2 * (k₁ / 6 + k₂ / 3 + k₃ / 18) = k₁ / 6 + k₂ / 6 + k₃ / 6)
  (h2 : c₃ % 6 = 0) (h3 : 4 * (c₁ / 3 + c₂ / 6 + c₃ / 18) < 2 * (c₁ / 6 + c₂ / 3 + c₃ / 18 + 1)) :
  c₁ + c₂ + c₃ = 2352 :=
sorry

end monkeys_bananas_l249_249845


namespace Kaleb_candies_l249_249732

theorem Kaleb_candies 
  (tickets_whack_a_mole : ℕ) 
  (tickets_skee_ball : ℕ) 
  (candy_cost : ℕ)
  (h1 : tickets_whack_a_mole = 8)
  (h2 : tickets_skee_ball = 7)
  (h3 : candy_cost = 5) : 
  (tickets_whack_a_mole + tickets_skee_ball) / candy_cost = 3 := 
by
  sorry

end Kaleb_candies_l249_249732


namespace linear_function_behavior_l249_249245

theorem linear_function_behavior (x y : ℝ) (h : y = -3 * x + 6) :
  ∀ x1 x2 : ℝ, x1 < x2 → (y = -3 * x1 + 6) → (y = -3 * x2 + 6) → -3 * (x1 - x2) > 0 :=
by
  sorry

end linear_function_behavior_l249_249245


namespace walkway_area_correct_l249_249386

-- Define the dimensions and conditions
def bed_width : ℝ := 4
def bed_height : ℝ := 3
def walkway_width : ℝ := 2
def num_rows : ℕ := 4
def num_columns : ℕ := 3
def num_beds : ℕ := num_rows * num_columns

-- Total dimensions of garden including walkways
def total_width : ℝ := (num_columns * bed_width) + ((num_columns + 1) * walkway_width)
def total_height : ℝ := (num_rows * bed_height) + ((num_rows + 1) * walkway_width)

-- Areas
def total_garden_area : ℝ := total_width * total_height
def total_bed_area : ℝ := (bed_width * bed_height) * num_beds

-- Correct answer we want to prove
def walkway_area : ℝ := total_garden_area - total_bed_area

theorem walkway_area_correct : walkway_area = 296 := by
  sorry

end walkway_area_correct_l249_249386


namespace range_of_m_l249_249411

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (x-1)^2
  else if x > 0 then -(x+1)^2
  else 0

theorem range_of_m (m : ℝ) (h : f (m^2 + 2*m) + f m > 0) : -3 < m ∧ m < 0 := 
by {
  sorry
}

end range_of_m_l249_249411


namespace tangent_lines_through_origin_l249_249828

-- Definition of the curve y = ln|x|
def curve (x : ℝ) : ℝ :=
  real.log (abs x)

-- Proposition stating that the tangent lines to the curve y = ln|x| passing through
-- the origin are given by x - e y = 0 and x + e y = 0.
theorem tangent_lines_through_origin :
  (∀ (x y : ℝ), curve x = y → (x - real.exp 1 * y = 0 ∨ x + real.exp 1 * y = 0)) ↔
  (∀ (x : ℝ), curve x = real.log (abs x)) :=
sorry

end tangent_lines_through_origin_l249_249828


namespace value_of_a6_l249_249058

theorem value_of_a6 (a : ℕ → ℝ) (h_positive : ∀ n, 0 < a n)
  (h_a1 : a 1 = 1) (h_a2 : a 2 = 2)
  (h_recurrence : ∀ n, 2 * (a n)^2 = (a (n + 1))^2 + (a (n - 1))^2) :
  a 6 = 4 := 
sorry

end value_of_a6_l249_249058


namespace find_cube_difference_l249_249054

theorem find_cube_difference (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : 
  x^3 - y^3 = 108 := 
by
  sorry

end find_cube_difference_l249_249054


namespace probability_rain_weekend_l249_249842

theorem probability_rain_weekend :
  let p_rain_saturday := 0.30
  let p_rain_sunday := 0.60
  let p_rain_sunday_given_rain_saturday := 0.40
  let p_no_rain_saturday := 1 - p_rain_saturday
  let p_no_rain_sunday_given_no_rain_saturday := 1 - p_rain_sunday
  let p_no_rain_both_days := p_no_rain_saturday * p_no_rain_sunday_given_no_rain_saturday
  let p_rain_sunday_given_rain_saturday := 1 - p_rain_sunday_given_rain_saturday
  let p_no_rain_sunday_given_rain_saturday := p_rain_saturday * p_rain_sunday_given_rain_saturday
  let p_no_rain_all_scenarios := p_no_rain_both_days + p_no_rain_sunday_given_rain_saturday
  let p_rain_weekend := 1 - p_no_rain_all_scenarios
  p_rain_weekend = 0.54 :=
sorry

end probability_rain_weekend_l249_249842


namespace factor_expression_l249_249920

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249920


namespace find_b_l249_249188

def h (x : ℝ) : ℝ := 4 * x - 5

theorem find_b (b : ℝ) (h_b : h b = 1) : b = 3 / 2 :=
by
  sorry

end find_b_l249_249188


namespace find_cube_difference_l249_249053

theorem find_cube_difference (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : 
  x^3 - y^3 = 108 := 
by
  sorry

end find_cube_difference_l249_249053


namespace committee_count_l249_249746

-- Definitions based on conditions
def num_males := 15
def num_females := 10

-- Define the binomial coefficient
def binomial (n k : ℕ) := Nat.choose n k

-- Define the total number of committees
def num_committees_with_at_least_two_females : ℕ :=
  binomial num_females 2 * binomial num_males 3 +
  binomial num_females 3 * binomial num_males 2 +
  binomial num_females 4 * binomial num_males 1 +
  binomial num_females 5 * binomial num_males 0

theorem committee_count : num_committees_with_at_least_two_females = 36477 :=
by {
  sorry
}

end committee_count_l249_249746


namespace kai_ice_plate_division_l249_249400

-- Define the "L"-shaped ice plate with given dimensions
structure LShapedIcePlate (a : ℕ) :=
(horiz_length : ℕ)
(vert_length : ℕ)
(horiz_eq_vert : horiz_length = a ∧ vert_length = a)

-- Define the correctness of dividing the L-shaped plate into four equal parts
def can_be_divided_into_four_equal_parts (a : ℕ) (piece : LShapedIcePlate a) : Prop :=
∃ cut_points_v1 cut_points_v2 cut_points_h1 cut_points_h2,
  -- The cut points for vertical and horizontal cuts to turn the large "L" shape into four smaller "L" shapes
  piece.horiz_length = cut_points_v1 + cut_points_v2 ∧
  piece.vert_length = cut_points_h1 + cut_points_h2 ∧
  cut_points_v1 = a / 2 ∧ cut_points_v2 = a - a / 2 ∧
  cut_points_h1 = a / 2 ∧ cut_points_h2 = a - a / 2

-- Prove the main theorem
theorem kai_ice_plate_division (a : ℕ) (h : a > 0) (plate : LShapedIcePlate a) : 
  can_be_divided_into_four_equal_parts a plate :=
sorry

end kai_ice_plate_division_l249_249400


namespace sin_of_300_degrees_l249_249560

theorem sin_of_300_degrees : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 := by
  sorry

end sin_of_300_degrees_l249_249560


namespace sin_300_eq_neg_sqrt3_div_2_l249_249582

noncomputable def sin_300 : ℝ := sin (300 * real.pi / 180)

theorem sin_300_eq_neg_sqrt3_div_2 : 
  (∠300° = 360° - 60°) →
  (300° ∈ fourth_quadrant) →
  (∀ θ ∈ fourth_quadrant, sin θ < 0) →
  (sin (60 * real.pi / 180) = sqrt 3 / 2) →
  sin_300 = -sqrt 3 / 2 := 
by
  intros h1 h2 h3 h4
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249582


namespace kaashish_problem_l249_249399

theorem kaashish_problem (x y : ℤ) (h : 2 * x + 3 * y = 100) (k : x = 28 ∨ y = 28) : x = 8 ∨ y = 8 :=
by
  sorry

end kaashish_problem_l249_249399


namespace largest_angle_is_90_degrees_l249_249673

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem largest_angle_is_90_degrees (u : ℝ) (a b c : ℝ) (v : ℝ) (h_v : v = 1)
  (h_a : a = Real.sqrt (2 * u - 1))
  (h_b : b = Real.sqrt (2 * u + 3))
  (h_c : c = 2 * Real.sqrt (u + v)) :
  is_right_triangle a b c :=
by
  sorry

end largest_angle_is_90_degrees_l249_249673


namespace sin_300_eq_neg_sqrt3_div_2_l249_249552

/-- Prove that the sine of 300 degrees is equal to -√3/2. -/
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249552


namespace cathy_can_win_l249_249405

theorem cathy_can_win (n k : ℕ) (hn : 0 < n) (hk : 0 < k) : 
  (∃ (f : ℕ → ℕ) (hf : ∀ i, f i < n + 1), (∀ i j, (i < j) → (f i < f j) → (f j = f i + 1)) → n ≤ 2^(k-1)) :=
sorry

end cathy_can_win_l249_249405


namespace find_a_plus_b_l249_249710

noncomputable def lines_intersect (a b : ℝ) : Prop := 
  (∃ x y : ℝ, (x = 1/3 * y + a) ∧ (y = 1/3 * x + b) ∧ (x = 3) ∧ (y = 6))

theorem find_a_plus_b (a b : ℝ) (h : lines_intersect a b) : a + b = 6 :=
sorry

end find_a_plus_b_l249_249710


namespace range_of_a_l249_249286

theorem range_of_a (x a : ℝ) :
  (∀ x : ℝ, x - 1 < 0 ∧ x < a + 3 → x < 1) → a ≥ -2 :=
by
  sorry

end range_of_a_l249_249286


namespace min_games_needed_l249_249822

theorem min_games_needed (N : ℕ) : 
  (2 + N) * 10 ≥ 9 * (5 + N) ↔ N ≥ 25 := 
by {
  sorry
}

end min_games_needed_l249_249822


namespace least_addition_for_divisibility_least_subtraction_for_divisibility_least_addition_for_common_divisibility_l249_249300

theorem least_addition_for_divisibility (n : ℕ) : (1100 + n) % 53 = 0 ↔ n = 9 := by
  sorry

theorem least_subtraction_for_divisibility (n : ℕ) : (1100 - n) % 71 = 0 ↔ n = 0 := by
  sorry

theorem least_addition_for_common_divisibility (X : ℕ) : (1100 + X) % (Nat.lcm 19 43) = 0 ∧ X = 534 := by
  sorry

end least_addition_for_divisibility_least_subtraction_for_divisibility_least_addition_for_common_divisibility_l249_249300


namespace rectangle_length_width_l249_249274

-- Given conditions
variables (L W : ℕ)

-- Condition 1: The area of the rectangular field is 300 square meters
def area_condition : Prop := L * W = 300

-- Condition 2: The perimeter of the rectangular field is 70 meters
def perimeter_condition : Prop := 2 * (L + W) = 70

-- Condition 3: One side of the rectangle is 20 meters
def side_condition : Prop := L = 20

-- Conclusion
def length_width_proof : Prop :=
  L = 20 ∧ W = 15

-- The final mathematical proof problem statement
theorem rectangle_length_width (L W : ℕ) 
  (h1 : area_condition L W) 
  (h2 : perimeter_condition L W) 
  (h3 : side_condition L) : 
  length_width_proof L W :=
sorry

end rectangle_length_width_l249_249274


namespace age_of_b_l249_249852

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 42) : b = 16 :=
by
  sorry

end age_of_b_l249_249852


namespace ducks_in_garden_l249_249819

theorem ducks_in_garden (num_rabbits : ℕ) (num_ducks : ℕ) 
  (total_legs : ℕ)
  (rabbit_legs : ℕ) (duck_legs : ℕ) 
  (H1 : num_rabbits = 9)
  (H2 : rabbit_legs = 4)
  (H3 : duck_legs = 2)
  (H4 : total_legs = 48)
  (H5 : num_rabbits * rabbit_legs + num_ducks * duck_legs = total_legs) :
  num_ducks = 6 := 
by {
  sorry
}

end ducks_in_garden_l249_249819


namespace ratio_of_part_to_whole_l249_249266

theorem ratio_of_part_to_whole (N : ℝ) (P : ℝ) (h1 : (1/4) * (2/5) * N = 17) (h2 : 0.40 * N = 204) :
  P = (2/5) * N → P / N = 2 / 5 :=
by
  intro h3
  sorry

end ratio_of_part_to_whole_l249_249266


namespace trapezoid_area_l249_249333

theorem trapezoid_area {a b c d e : ℝ} (h1 : a = 40) (h2 : b = 40) (h3 : c = 50) (h4 : d = 50) (h5 : e = 60) : 
  (a + b = 80) → (c * c = 2500) → 
  (50^2 - 30^2 = 1600) → ((50^2 - 30^2).sqrt = 40) → 
  (((e - 2 * ((a ^ 2 - (30) ^ 2).sqrt)) * 40) / 2 = 1336) :=
sorry

end trapezoid_area_l249_249333


namespace sin_300_eq_neg_sqrt_three_div_two_l249_249547

theorem sin_300_eq_neg_sqrt_three_div_two :
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt_three_div_two_l249_249547


namespace factor_expression_l249_249908

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249908


namespace teacher_age_l249_249736

theorem teacher_age {student_count : ℕ} (avg_age_students : ℕ) (avg_age_with_teacher : ℕ)
    (h1 : student_count = 25) (h2 : avg_age_students = 26) (h3 : avg_age_with_teacher = 27) :
    ∃ (teacher_age : ℕ), teacher_age = 52 :=
by
  sorry

end teacher_age_l249_249736


namespace probability_heads_penny_nickel_dime_l249_249113

variable {Ω : Type} [Fintype Ω] [DecidableEq Ω]
variable (coins : Fin 5 → Ω)
variable (heads tails : Ω)

-- Each coin has two outcomes: heads or tails
axiom coin_outcome (i : Fin 5) : coins i = heads ∨ coins i = tails

-- There are 32 total outcomes
axiom total_outcomes : Fintype.card (Fin 5 → Ω) = 32

-- There are 4 successful outcomes for penny, nickel, and dime being heads
axiom successful_outcomes : let successful := {x // (coins 0 = heads) ∧ (coins 1 = heads) ∧ (coins 2 = heads)} in
                          Fintype.card successful = 4

theorem probability_heads_penny_nickel_dime :
  let probability := (Fintype.card {x // (coins 0 = heads) ∧ (coins 1 = heads) ∧ (coins 2 = heads)} : ℤ) /
                     (Fintype.card (Fin 5 → Ω) : ℤ) in
  probability = 1 / 8 :=
by
  sorry

end probability_heads_penny_nickel_dime_l249_249113


namespace cubes_with_one_face_painted_cubes_with_two_faces_painted_size_of_new_cube_l249_249332

def cube (n : ℕ) : Type := ℕ × ℕ × ℕ

-- Define a 4x4x4 cube and the painting conditions
def four_by_four_cube := cube 4

-- Determine the number of small cubes with exactly one face painted
theorem cubes_with_one_face_painted : 
  ∃ (count : ℕ), count = 24 :=
by
  -- proof goes here
  sorry

-- Determine the number of small cubes with exactly two faces painted
theorem cubes_with_two_faces_painted : 
  ∃ (count : ℕ), count = 24 :=
by
  -- proof goes here
  sorry

-- Given condition and find the size of the new cube
theorem size_of_new_cube (n : ℕ) : 
  (n - 2) ^ 3 = 3 * 12 * (n - 2) → n = 8 :=
by
  -- proof goes here
  sorry

end cubes_with_one_face_painted_cubes_with_two_faces_painted_size_of_new_cube_l249_249332


namespace net_rate_of_pay_equals_39_dollars_per_hour_l249_249162

-- Definitions of the conditions
def hours_travelled : ℕ := 3
def speed_per_hour : ℕ := 60
def car_consumption_rate : ℕ := 30
def earnings_per_mile : ℕ := 75  -- expressing $0.75 as 75 cents to avoid floating-point
def gasoline_cost_per_gallon : ℕ := 300  -- expressing $3.00 as 300 cents to avoid floating-point

-- Proof statement
theorem net_rate_of_pay_equals_39_dollars_per_hour : 
  (earnings_per_mile * (speed_per_hour * hours_travelled) - gasoline_cost_per_gallon * ((speed_per_hour * hours_travelled) / car_consumption_rate)) / hours_travelled = 3900 := 
by 
  -- The statement below essentially expresses 39 dollars per hour in cents (i.e., 3900 cents per hour).
  sorry

end net_rate_of_pay_equals_39_dollars_per_hour_l249_249162


namespace sin_300_eq_neg_sqrt3_div_2_l249_249588

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249588


namespace last_part_length_l249_249432

-- Definitions of the conditions
def total_length : ℝ := 74.5
def part1_length : ℝ := 15.5
def part2_length : ℝ := 21.5
def part3_length : ℝ := 21.5

-- Theorem statement to prove the length of the last part of the race
theorem last_part_length :
  (total_length - (part1_length + part2_length + part3_length)) = 16 := 
  by 
    sorry

end last_part_length_l249_249432


namespace sin_300_eq_neg_sqrt3_div_2_l249_249586

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249586


namespace book_store_sold_total_copies_by_saturday_l249_249324

def copies_sold_on_monday : ℕ := 15
def copies_sold_on_tuesday : ℕ := copies_sold_on_monday * 2
def copies_sold_on_wednesday : ℕ := copies_sold_on_tuesday + (copies_sold_on_tuesday / 2)
def copies_sold_on_thursday : ℕ := copies_sold_on_wednesday + (copies_sold_on_wednesday / 2)
def copies_sold_on_friday_pre_promotion : ℕ := copies_sold_on_thursday + (copies_sold_on_thursday / 2)
def copies_sold_on_friday_post_promotion : ℕ := copies_sold_on_friday_pre_promotion + (copies_sold_on_friday_pre_promotion / 4)
def copies_sold_on_saturday : ℕ := copies_sold_on_friday_pre_promotion * 7 / 10

def total_copies_sold_by_saturday : ℕ :=
  copies_sold_on_monday + copies_sold_on_tuesday + copies_sold_on_wednesday +
  copies_sold_on_thursday + copies_sold_on_friday_post_promotion + copies_sold_on_saturday

theorem book_store_sold_total_copies_by_saturday : total_copies_sold_by_saturday = 357 :=
by
  -- Proof here
  sorry

end book_store_sold_total_copies_by_saturday_l249_249324


namespace positive_real_as_sum_l249_249990

theorem positive_real_as_sum (k : ℝ) (hk : k > 0) : 
  ∃ (a : ℕ → ℕ), (∀ n, a n > 0) ∧ (∀ n, a n < a (n + 1)) ∧ (∑' n, 1 / 10 ^ a n = k) :=
sorry

end positive_real_as_sum_l249_249990


namespace find_b_value_l249_249680

theorem find_b_value (b : ℝ) (h1 : (0 : ℝ) * 0 + (sqrt (b - 1)) * 0 + b^2 - 4 = 0) : b = 2 :=
sorry

end find_b_value_l249_249680


namespace award_distribution_l249_249820

theorem award_distribution (students awards : ℕ) (h_s : students = 4) (h_a : awards = 7) :
  ∃(d:ℕ), 
  (d = 5880) ∧ ∀(award_dist : Fin students → ℕ), 
  (∑ i, award_dist i = awards) ∧ 
  (∀ i, award_dist i ≥ 1) ∧ 
  (∃ i, award_dist i = 3) →
  d = ∑ i : Fin students, award_dist i :=
by
  sorry

end award_distribution_l249_249820


namespace total_students_l249_249836

theorem total_students (S K : ℕ) (h1 : S = 4000) (h2 : K = 2 * S) :
  S + K = 12000 := by
  sorry

end total_students_l249_249836


namespace sin_300_l249_249597

-- Definitions and conditions in Lean 4
def angle_deg := ℝ
def sin (θ : angle_deg) : ℝ := Real.sin (θ * Real.pi / 180)

noncomputable def Q_angle : angle_deg := 300
noncomputable def first_quadrant_equivalent_angle : angle_deg := 360 - 300
noncomputable def sin_60 : ℝ := Real.sin (60 * Real.pi / 180)

-- Main statement to prove
theorem sin_300 : sin 300 = -sin_60 := by
  sorry

end sin_300_l249_249597


namespace probability_heads_is_one_eighth_l249_249120

-- Define the probability problem
def probability_heads_penny_nickel_dime (total_coins: ℕ) (successful_events: ℕ) : ℚ :=
  successful_events / total_coins

-- Define the constants
def total_outcomes : ℕ := 2^5  -- Number of possible outcomes with 5 coins
def successful_outcomes : ℕ := 4  -- Number of successful outcomes where penny, nickel, and dime are heads

-- State the theorem to be proven
theorem probability_heads_is_one_eighth : 
  probability_heads_penny_nickel_dime total_outcomes successful_outcomes = 1 / 8 :=
by
  sorry

end probability_heads_is_one_eighth_l249_249120


namespace proof_of_ratio_l249_249402

def f (x : ℤ) : ℤ := 3 * x + 4

def g (x : ℤ) : ℤ := 4 * x - 3

theorem proof_of_ratio :
  (f (g (f 3))) / (g (f (g 3))) = 151 / 121 :=
by
  sorry

end proof_of_ratio_l249_249402


namespace polly_age_is_33_l249_249816

theorem polly_age_is_33 
  (x : ℕ) 
  (h1 : ∀ y, y = 20 → x - y = x - 20)
  (h2 : ∀ y, y = 22 → x - y = x - 22)
  (h3 : ∀ y, y = 24 → x - y = x - 24) : 
  x = 33 :=
by 
  sorry

end polly_age_is_33_l249_249816


namespace tile_equations_correct_l249_249148

theorem tile_equations_correct (x y : ℕ) (h1 : 24 * x + 12 * y = 2220) (h2 : y = 2 * x - 15) : 
    (24 * x + 12 * y = 2220) ∧ (y = 2 * x - 15) :=
by
  exact ⟨h1, h2⟩

end tile_equations_correct_l249_249148


namespace christine_savings_l249_249341

/-- Christine's commission rate as a percentage. -/
def commissionRate : ℝ := 0.12

/-- Total sales made by Christine this month in dollars. -/
def totalSales : ℝ := 24000

/-- Percentage of commission allocated to personal needs. -/
def personalNeedsRate : ℝ := 0.60

/-- The amount Christine saved this month. -/
def amountSaved : ℝ := 1152

/--
Given the commission rate, total sales, and personal needs rate,
prove the amount saved is correctly calculated.
-/
theorem christine_savings :
  (1 - personalNeedsRate) * (commissionRate * totalSales) = amountSaved :=
by
  sorry

end christine_savings_l249_249341


namespace sin_300_deg_l249_249492

noncomputable def sin_300_equals_neg_sqrt3_div_2 : Prop :=
  sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2

theorem sin_300_deg : sin_300_equals_neg_sqrt3_div_2 :=
by
  sorry

end sin_300_deg_l249_249492


namespace difference_of_numbers_l249_249140

theorem difference_of_numbers (a : ℕ) (h : a + (10 * a + 5) = 30000) : (10 * a + 5) - a = 24548 :=
by
  sorry

end difference_of_numbers_l249_249140


namespace sin_300_eq_neg_sqrt_3_div_2_l249_249517

noncomputable def sin_300_degrees : ℝ := Real.sin (300 * Real.pi / 180)

theorem sin_300_eq_neg_sqrt_3_div_2 : sin_300_degrees = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_300_eq_neg_sqrt_3_div_2_l249_249517


namespace find_higher_selling_price_l249_249013

-- Define the constants and initial conditions
def cost_price : ℕ := 200
def selling_price_1 : ℕ := 340
def gain_1 : ℕ := selling_price_1 - cost_price
def new_gain : ℕ := gain_1 + gain_1 * 5 / 100

-- Define the problem statement
theorem find_higher_selling_price : 
  ∀ P : ℕ, P = cost_price + new_gain → P = 347 :=
by
  intro P
  intro h
  sorry

end find_higher_selling_price_l249_249013


namespace product_lcm_gcd_l249_249725

theorem product_lcm_gcd (a b : ℕ) (h_a : a = 24) (h_b : b = 36):
  Nat.lcm a b * Nat.gcd a b = 864 :=
by
  rw [h_a, h_b]
  sorry

end product_lcm_gcd_l249_249725


namespace radar_coverage_proof_l249_249155

theorem radar_coverage_proof (n : ℕ) (r : ℝ) (w : ℝ) (d : ℝ) (A : ℝ) : 
  n = 9 ∧ r = 37 ∧ w = 24 ∧ d = 35 / Real.sin (Real.pi / 9) ∧
  A = 1680 * Real.pi / Real.tan (Real.pi / 9) → 
  ∃ OB S_ring, OB = d ∧ S_ring = A 
:= by sorry

end radar_coverage_proof_l249_249155


namespace circle_tangent_sum_radii_l249_249865

theorem circle_tangent_sum_radii :
  let r1 := 6 + 2 * Real.sqrt 6
  let r2 := 6 - 2 * Real.sqrt 6
  r1 + r2 = 12 :=
by
  sorry

end circle_tangent_sum_radii_l249_249865


namespace wooden_parallelepiped_length_l249_249004

theorem wooden_parallelepiped_length (n : ℕ) (h1 : n ≥ 7)
    (h2 : ∀ total_cubes unpainted_cubes : ℕ,
      total_cubes = n * (n - 2) * (n - 4) ∧
      unpainted_cubes = (n - 2) * (n - 4) * (n - 6) ∧
      unpainted_cubes = 2 / 3 * total_cubes) :
  n = 18 := 
sorry

end wooden_parallelepiped_length_l249_249004


namespace sum_of_fraction_numerator_and_denominator_l249_249730

theorem sum_of_fraction_numerator_and_denominator : 
  ∀ x : ℚ, (∀ n : ℕ, x = 2 / 3 + (4/9)^n) → 
  let frac := (24 : ℚ) / 99 in 
  let simplified_frac := frac.num.gcd 24 / frac.denom.gcd 99 in 
  simplified_frac.num + simplified_frac.denom = 41 :=
sorry

end sum_of_fraction_numerator_and_denominator_l249_249730


namespace factor_expression_l249_249921

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249921


namespace sqrt_x_div_sqrt_y_as_fraction_l249_249770

theorem sqrt_x_div_sqrt_y_as_fraction 
  (x y : ℝ)
  (h : (1/3)^2 + (1/4)^2 + (1/6)^2 = 54 * x / 115 * y * ((1/5)^2 + (1/7)^2 + (1/8)^2)) : 
  (Real.sqrt x) / (Real.sqrt y) = 49 / 29 :=
by
  sorry

end sqrt_x_div_sqrt_y_as_fraction_l249_249770


namespace number_of_buyers_l249_249452

theorem number_of_buyers 
  (today yesterday day_before : ℕ) 
  (h1 : today = yesterday + 40) 
  (h2 : yesterday = day_before / 2) 
  (h3 : day_before + yesterday + today = 140) : 
  day_before = 67 :=
by
  -- skip the proof
  sorry

end number_of_buyers_l249_249452


namespace no_four_consecutive_powers_l249_249766

/-- 
  There do not exist four consecutive natural numbers 
  such that each of them is a power (greater than 1) of another natural number.
-/
theorem no_four_consecutive_powers : 
  ¬ ∃ (n : ℕ), (∀ (i : ℕ), i < 4 → ∃ (a k : ℕ), k > 1 ∧ n + i = a^k) := sorry

end no_four_consecutive_powers_l249_249766


namespace final_amount_in_account_l249_249296

noncomputable def initial_deposit : ℝ := 1000
noncomputable def first_year_interest_rate : ℝ := 0.2
noncomputable def first_year_balance : ℝ := initial_deposit * (1 + first_year_interest_rate)
noncomputable def withdrawal_amount : ℝ := first_year_balance / 2
noncomputable def after_withdrawal_balance : ℝ := first_year_balance - withdrawal_amount
noncomputable def second_year_interest_rate : ℝ := 0.15
noncomputable def final_balance : ℝ := after_withdrawal_balance * (1 + second_year_interest_rate)

theorem final_amount_in_account : final_balance = 690 := by
  sorry

end final_amount_in_account_l249_249296


namespace vehicle_A_no_speed_increase_needed_l249_249720

noncomputable def V_A := 60 -- Speed of Vehicle A in mph
noncomputable def V_B := 70 -- Speed of Vehicle B in mph
noncomputable def V_C := 50 -- Speed of Vehicle C in mph
noncomputable def dist_AB := 100 -- Initial distance between A and B in ft
noncomputable def dist_AC := 300 -- Initial distance between A and C in ft

theorem vehicle_A_no_speed_increase_needed 
  (V_A V_B V_C : ℝ)
  (dist_AB dist_AC : ℝ)
  (h1 : V_A > V_C)
  (h2 : V_A = 60)
  (h3 : V_B = 70)
  (h4 : V_C = 50)
  (h5 : dist_AB = 100)
  (h6 : dist_AC = 300) : 
  ∀ ΔV : ℝ, ΔV = 0 :=
by
  sorry -- Proof to be filled out

end vehicle_A_no_speed_increase_needed_l249_249720


namespace tracy_dog_food_l249_249455

theorem tracy_dog_food
(f : ℕ) (c : ℝ) (m : ℕ) (d : ℕ)
(hf : f = 4) (hc : c = 2.25) (hm : m = 3) (hd : d = 2) :
  (f * c / m) / d = 1.5 :=
by
  sorry

end tracy_dog_food_l249_249455


namespace how_many_candies_eaten_l249_249888

variable (candies_tuesday candies_thursday candies_friday candies_left : ℕ)

def total_candies (candies_tuesday candies_thursday candies_friday : ℕ) : ℕ :=
  candies_tuesday + candies_thursday + candies_friday

theorem how_many_candies_eaten (h_tuesday : candies_tuesday = 3)
                               (h_thursday : candies_thursday = 5)
                               (h_friday : candies_friday = 2)
                               (h_left : candies_left = 4) :
  (total_candies candies_tuesday candies_thursday candies_friday) - candies_left = 6 :=
by
  sorry

end how_many_candies_eaten_l249_249888


namespace range_a_l249_249214

theorem range_a (a : ℝ) :
  (∀ x : ℝ, (0 < x ∧ x ≤ 2) → x^2 - 2 * a * x + 1 ≥ 0) → a ≤ 1 :=
by
  sorry

end range_a_l249_249214


namespace not_product_24_pair_not_24_l249_249147

theorem not_product_24 (a b : ℤ) : 
  (a, b) = (-4, -6) ∨ (a, b) = (-2, -12) ∨ (a, b) = (2, 12) ∨ (a, b) = (3/4, 32) → a * b = 24 :=
sorry

theorem pair_not_24 :
  ¬(1/3 * -72 = 24) :=
sorry

end not_product_24_pair_not_24_l249_249147


namespace terminating_decimals_count_l249_249933

theorem terminating_decimals_count :
  let n_values := {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ (∃ k, n = 21 * k)} in
  n_values.finite.count = 23 :=
by {
  sorry
}

end terminating_decimals_count_l249_249933


namespace stable_points_of_g_fixed_points_subset_stable_points_range_of_a_l249_249205

-- Definitions of fixed points and stable points
def is_fixed_point(f : ℝ → ℝ) (x : ℝ) : Prop := f x = x
def is_stable_point(f : ℝ → ℝ) (x : ℝ) : Prop := f (f x) = x 

-- Problem 1: Stable points of g(x) = 2x - 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem stable_points_of_g : {x : ℝ | is_stable_point g x} = {1} :=
sorry

-- Problem 2: Prove A ⊂ B for any function f
theorem fixed_points_subset_stable_points (f : ℝ → ℝ) : 
  {x : ℝ | is_fixed_point f x} ⊆ {x : ℝ | is_stable_point f x} :=
sorry

-- Problem 3: Range of a for f(x) = ax^2 - 1 when A = B ≠ ∅
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1

theorem range_of_a (a : ℝ) (h : ∃ x, is_fixed_point (f a) x ∧ is_stable_point (f a) x):
  - (1/4 : ℝ) ≤ a ∧ a ≤ (3/4 : ℝ) :=
sorry

end stable_points_of_g_fixed_points_subset_stable_points_range_of_a_l249_249205


namespace geometric_sequence_tenth_term_l249_249343

theorem geometric_sequence_tenth_term :
  let a := 5
  let r := 3 / 2
  let a_n (n : ℕ) := a * r ^ (n - 1)
  a_n 10 = 98415 / 512 :=
by
  sorry

end geometric_sequence_tenth_term_l249_249343


namespace percentage_transform_l249_249226

theorem percentage_transform (n : ℝ) (h : 0.3 * 0.4 * n = 36) : 0.4 * 0.3 * n = 36 :=
by
  sorry

end percentage_transform_l249_249226


namespace total_trout_caught_l249_249098

theorem total_trout_caught (n_share j_share total_caught : ℕ) (h1 : n_share = 9) (h2 : j_share = 9) (h3 : total_caught = n_share + j_share) :
  total_caught = 18 :=
by
  sorry

end total_trout_caught_l249_249098


namespace factor_expression_l249_249907

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249907


namespace find_cube_difference_l249_249052

theorem find_cube_difference (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : 
  x^3 - y^3 = 108 := 
by
  sorry

end find_cube_difference_l249_249052


namespace probability_of_triangle_or_circle_l249_249984

-- Definitions (conditions)
def total_figures : ℕ := 12
def triangles : ℕ := 4
def circles : ℕ := 3
def squares : ℕ := 5
def figures : ℕ := triangles + circles + squares

-- Probability calculation
def probability_triangle_circle := (triangles + circles) / total_figures

-- Theorem statement (problem)
theorem probability_of_triangle_or_circle : probability_triangle_circle = 7 / 12 :=
by
  -- The proof is omitted, insert the proof here when necessary.
  sorry

end probability_of_triangle_or_circle_l249_249984


namespace min_dot_product_l249_249366

-- Define the conditions of the ellipse and focal points
variables (P : ℝ × ℝ)
def ellipse (x y : ℝ) := (x^2 / 4) + (y^2 / 3) = 1
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define vectors
def OP (P : ℝ × ℝ) : ℝ × ℝ := P
def FP (P : ℝ × ℝ) : ℝ × ℝ := (P.1 + 1, P.2)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Prove that the minimum value of the dot product is 2
theorem min_dot_product (hP : ellipse P.1 P.2) : 
  ∃ (P : ℝ × ℝ), dot_product (OP P) (FP P) = 2 := sorry

end min_dot_product_l249_249366


namespace sin_300_eq_neg_sqrt3_div_2_l249_249554

/-- Prove that the sine of 300 degrees is equal to -√3/2. -/
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249554


namespace sin_300_eq_neg_sqrt_3_div_2_l249_249514

noncomputable def sin_300_degrees : ℝ := Real.sin (300 * Real.pi / 180)

theorem sin_300_eq_neg_sqrt_3_div_2 : sin_300_degrees = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_300_eq_neg_sqrt_3_div_2_l249_249514


namespace total_fencing_cost_l249_249068

-- Definitions based on the conditions
def cost_per_side : ℕ := 69
def number_of_sides : ℕ := 4

-- The proof problem statement
theorem total_fencing_cost : number_of_sides * cost_per_side = 276 := by
  sorry

end total_fencing_cost_l249_249068


namespace terminal_side_in_fourth_quadrant_l249_249944

theorem terminal_side_in_fourth_quadrant 
  (h_sin_half : Real.sin (α / 2) = 3 / 5)
  (h_cos_half : Real.cos (α / 2) = -4 / 5) : 
  (Real.sin α < 0) ∧ (Real.cos α > 0) :=
by
  sorry

end terminal_side_in_fourth_quadrant_l249_249944


namespace robert_salary_loss_l249_249427

theorem robert_salary_loss (S : ℝ) : 
  let decreased_salary := S - 0.3 * S
  let increased_salary := decreased_salary + 0.3 * decreased_salary
  100 * (1 - increased_salary / S) = 9 :=
by
  let decreased_salary := S - 0.3 * S
  let increased_salary := decreased_salary + 0.3 * decreased_salary
  sorry

end robert_salary_loss_l249_249427


namespace sin_300_eq_neg_sqrt3_div_2_l249_249504

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249504


namespace sin_300_eq_neg_sqrt_three_div_two_l249_249542

theorem sin_300_eq_neg_sqrt_three_div_two :
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt_three_div_two_l249_249542


namespace sin_300_eq_neg_sqrt3_div_2_l249_249590

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249590


namespace passengers_at_station_in_an_hour_l249_249023

-- Define the conditions
def train_interval_minutes := 5
def passengers_off_per_train := 200
def passengers_on_per_train := 320

-- Define the time period we're considering
def time_period_minutes := 60

-- Calculate the expected values based on conditions
def expected_trains_per_hour := time_period_minutes / train_interval_minutes
def expected_passengers_off_per_hour := passengers_off_per_train * expected_trains_per_hour
def expected_passengers_on_per_hour := passengers_on_per_train * expected_trains_per_hour
def expected_total_passengers := expected_passengers_off_per_hour + expected_passengers_on_per_hour

theorem passengers_at_station_in_an_hour :
  expected_total_passengers = 6240 :=
by
  -- Structure of the proof omitted. Just ensuring conditions and expected value defined.
  sorry

end passengers_at_station_in_an_hour_l249_249023


namespace find_a_l249_249774

def f (x : ℝ) : ℝ := -x^2 - 2 * x + 3

theorem find_a : ∃ a : ℝ, (a > -1) ∧ (a < 2) ∧ (∀ x : ℝ, a ≤ x ∧ x ≤ 2 → f x ≤ f a) ∧ f a = 15 / 4 :=
by
  exists -1 / 2
  sorry

end find_a_l249_249774


namespace sin_300_eq_neg_sqrt3_div_2_l249_249558

/-- Prove that the sine of 300 degrees is equal to -√3/2. -/
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249558


namespace vertical_asymptote_l249_249189

theorem vertical_asymptote (x : ℚ) : (7 * x + 4 = 0) → (x = -4 / 7) :=
by
  intro h
  sorry

end vertical_asymptote_l249_249189


namespace sin_300_eq_neg_sqrt3_div_2_l249_249618

-- Let Q be the point on the unit circle at 300 degrees counterclockwise from (1,0)
def Q : ℝ × ℝ := (1 / 2, -Real.sqrt 3 / 2)

-- Prove that the sine of 300 degrees is -√3/2
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  -- Here we assume the coordinates given in the conditions are correct.
  have hQ : Q = (1 / 2, -Real.sqrt 3 / 2) := rfl
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249618


namespace total_students_l249_249835

theorem total_students (S K : ℕ) (h1 : S = 4000) (h2 : K = 2 * S) :
  S + K = 12000 := by
  sorry

end total_students_l249_249835


namespace sin_pi_div_three_l249_249451

theorem sin_pi_div_three : Real.sin (π / 3) = Real.sqrt 3 / 2 := 
sorry

end sin_pi_div_three_l249_249451


namespace lcm_gcd_product_24_36_l249_249723

theorem lcm_gcd_product_24_36 : 
  let a := 24
  let b := 36
  let g := Int.gcd a b
  let l := Int.lcm a b
  g * l = 864 := by
  let a := 24
  let b := 36
  let g := Int.gcd a b
  have gcd_eq : g = 12 := by sorry
  let l := Int.lcm a b
  have lcm_eq : l = 72 := by sorry
  show g * l = 864 from by
    rw [gcd_eq, lcm_eq]
    exact calc
      12 * 72 = 864 : by norm_num

end lcm_gcd_product_24_36_l249_249723


namespace find_a1_an_l249_249209

noncomputable def arith_geo_seq (a : ℕ → ℝ) : Prop :=
  (∃ d ≠ 0, (a 2 + a 4 = 10) ∧ (a 2 ^ 2 = a 1 * a 5))

theorem find_a1_an (a : ℕ → ℝ)
  (h_arith_geo_seq : arith_geo_seq a) :
  a 1 = 1 ∧ (∀ n, a n = 2 * n - 1) :=
sorry

end find_a1_an_l249_249209


namespace edward_original_lawns_l249_249197

-- Definitions based on conditions
def dollars_per_lawn : ℕ := 4
def lawns_forgotten : ℕ := 9
def dollars_earned : ℕ := 32

-- The original number of lawns to mow
def original_lawns_to_mow (L : ℕ) : Prop :=
  dollars_per_lawn * (L - lawns_forgotten) = dollars_earned

-- The proof problem statement
theorem edward_original_lawns : ∃ L : ℕ, original_lawns_to_mow L ∧ L = 17 :=
by
  sorry

end edward_original_lawns_l249_249197


namespace divisor_in_second_division_l249_249421

theorem divisor_in_second_division 
  (n : ℤ) 
  (h1 : (68 : ℤ) * 269 = n) 
  (d q : ℤ) 
  (h2 : n = d * q + 1) 
  (h3 : Prime 18291):
  d = 18291 := by
  sorry

end divisor_in_second_division_l249_249421


namespace sin_300_eq_neg_sqrt_3_div_2_l249_249519

noncomputable def sin_300_degrees : ℝ := Real.sin (300 * Real.pi / 180)

theorem sin_300_eq_neg_sqrt_3_div_2 : sin_300_degrees = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_300_eq_neg_sqrt_3_div_2_l249_249519


namespace performance_attendance_l249_249478

theorem performance_attendance (A C : ℕ) (hC : C = 18) (hTickets : 16 * A + 9 * C = 258) : A + C = 24 :=
by
  sorry

end performance_attendance_l249_249478


namespace correct_equation_l249_249124

-- Define the daily paving distances for Team A and Team B
variables (x : ℝ) (h₀ : x > 10)

-- Assuming Team A takes the same number of days to pave 150m as Team B takes to pave 120m
def same_days_to_pave (h₁ : x - 10 > 0) : Prop :=
  (150 / x = 120 / (x - 10))

-- The theorem to be proven
theorem correct_equation (h₁ : x - 10 > 0) : 150 / x = 120 / (x - 10) :=
by
  sorry

end correct_equation_l249_249124


namespace probability_heads_is_one_eighth_l249_249121

-- Define the probability problem
def probability_heads_penny_nickel_dime (total_coins: ℕ) (successful_events: ℕ) : ℚ :=
  successful_events / total_coins

-- Define the constants
def total_outcomes : ℕ := 2^5  -- Number of possible outcomes with 5 coins
def successful_outcomes : ℕ := 4  -- Number of successful outcomes where penny, nickel, and dime are heads

-- State the theorem to be proven
theorem probability_heads_is_one_eighth : 
  probability_heads_penny_nickel_dime total_outcomes successful_outcomes = 1 / 8 :=
by
  sorry

end probability_heads_is_one_eighth_l249_249121


namespace sum_of_integers_is_96_l249_249423

theorem sum_of_integers_is_96 (x y : ℤ) (h1 : x = 32) (h2 : y = 2 * x) : x + y = 96 := 
by
  sorry

end sum_of_integers_is_96_l249_249423


namespace find_interval_l249_249641

theorem find_interval (x : ℝ) : (x > 3/4 ∧ x < 4/5) ↔ (5 * x + 1 > 3 ∧ 5 * x + 1 < 5 ∧ 4 * x > 3 ∧ 4 * x < 5) :=
by
  sorry

end find_interval_l249_249641


namespace sin_300_eq_neg_sqrt3_div_2_l249_249501

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249501


namespace number_conversion_l249_249739

theorem number_conversion (a b c d : ℕ) : 
  4090000 = 409 * 10000 ∧ (a = 800000) ∧ (b = 5000) ∧ (c = 20) ∧ (d = 4) → 
  (a + b + c + d = 805024) :=
by
  sorry

end number_conversion_l249_249739


namespace factor_expression_l249_249903

theorem factor_expression (x : ℝ) : (x * (x + 3) + 2 * (x + 3)) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249903


namespace parallelogram_is_central_not_axis_symmetric_l249_249107

-- Definitions for the shapes discussed in the problem
def is_central_symmetric (shape : Type) : Prop := sorry
def is_axis_symmetric (shape : Type) : Prop := sorry

-- Specific shapes being used in the problem
def rhombus : Type := sorry
def parallelogram : Type := sorry
def equilateral_triangle : Type := sorry
def rectangle : Type := sorry

-- Example additional assumptions about shapes can be added here if needed

-- The problem assertion
theorem parallelogram_is_central_not_axis_symmetric :
  is_central_symmetric parallelogram ∧ ¬ is_axis_symmetric parallelogram :=
sorry

end parallelogram_is_central_not_axis_symmetric_l249_249107


namespace gcd_b_n_b_n_plus_1_l249_249191

-- Definitions based on the conditions in the problem
def b_n (n : ℕ) : ℕ := 150 + n^3

theorem gcd_b_n_b_n_plus_1 (n : ℕ) : gcd (b_n n) (b_n (n + 1)) = 1 := by
  -- We acknowledge that we need to skip the proof steps
  sorry

end gcd_b_n_b_n_plus_1_l249_249191


namespace find_varphi_l249_249683

noncomputable def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g(-x) = -g(x)

theorem find_varphi (f : ℝ → ℝ) (φ : ℝ) (h1 : ∀ x, f(x) = Real.sin(2 * x + φ)) (h2 : 0 < φ ∧ φ < Real.pi) (h3 : is_odd_function (λ x, f(x - Real.pi / 3))) :
  φ = 2 * Real.pi / 3 :=
by
  sorry

end find_varphi_l249_249683


namespace pinocchio_optimal_success_probability_l249_249101

def success_prob (s : List ℚ) : ℚ :=
  s.foldr (λ x acc => (x * acc) / (1 - (1 - x) * acc)) 1

theorem pinocchio_optimal_success_probability :
  let success_probs := [9/10, 8/10, 7/10, 6/10, 5/10, 4/10, 3/10, 2/10, 1/10]
  success_prob success_probs = 0.4315 :=
by 
  sorry

end pinocchio_optimal_success_probability_l249_249101


namespace find_A_l249_249135

def is_valid_A (A : ℕ) : Prop :=
  A = 1 ∨ A = 2 ∨ A = 4 ∨ A = 7 ∨ A = 9

def number (A : ℕ) : ℕ :=
  3 * 100000 + 0 * 10000 + 5 * 1000 + 2 * 100 + 0 * 10 + A

theorem find_A (A : ℕ) (h_valid_A : is_valid_A A) : A = 1 ↔ Nat.Prime (number A) :=
by
  sorry

end find_A_l249_249135


namespace continuity_necessity_not_sufficiency_l249_249318

theorem continuity_necessity_not_sufficiency (f : ℝ → ℝ) (x₀ : ℝ) :
  ((∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - f x₀) < ε) → f x₀ = f x₀) ∧ ¬ ((f x₀ = f x₀) → (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - f x₀) < ε)) := 
sorry

end continuity_necessity_not_sufficiency_l249_249318


namespace find_n_l249_249156

theorem find_n:
  ∃ n : ℕ, (n : ℚ) / 2^n = 5 / 32 :=
sorry

end find_n_l249_249156


namespace number_of_months_to_fully_pay_off_car_l249_249417

def total_price : ℕ := 13380
def initial_payment : ℕ := 5400
def monthly_payment : ℕ := 420

theorem number_of_months_to_fully_pay_off_car :
  (total_price - initial_payment) / monthly_payment = 19 :=
by
  sorry

end number_of_months_to_fully_pay_off_car_l249_249417


namespace line_through_midpoint_l249_249958

theorem line_through_midpoint (x y : ℝ) (P : x = 2 ∧ y = -1) :
  (∃ l : ℝ, ∀ t : ℝ, 
  (1 + 5 * Real.cos t = x) ∧ (5 * Real.sin t = y) →
  (x - y = 3)) :=
by
  sorry

end line_through_midpoint_l249_249958


namespace cost_of_milkshake_l249_249973

theorem cost_of_milkshake
  (initial_money : ℝ)
  (remaining_after_cupcakes : ℝ)
  (remaining_after_sandwich : ℝ)
  (remaining_after_toy : ℝ)
  (final_remaining : ℝ)
  (money_spent_on_milkshake : ℝ) :
  initial_money = 20 →
  remaining_after_cupcakes = initial_money - (1 / 4) * initial_money →
  remaining_after_sandwich = remaining_after_cupcakes - 0.30 * remaining_after_cupcakes →
  remaining_after_toy = remaining_after_sandwich - (1 / 5) * remaining_after_sandwich →
  final_remaining = 3 →
  money_spent_on_milkshake = remaining_after_toy - final_remaining →
  money_spent_on_milkshake = 5.40 :=
by
  intros 
  sorry

end cost_of_milkshake_l249_249973


namespace number_of_real_roots_of_cubic_l249_249408

-- Define the real number coefficients
variables (a b c d : ℝ)

-- Non-zero condition on coefficients
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

-- Statement of the problem: The cubic polynomial typically has 3 real roots
theorem number_of_real_roots_of_cubic :
  ∃ (x : ℝ), (x ^ 3 + x * (c ^ 2 - d ^ 2 - b * d) - (b ^ 2) * c = 0) := by
  sorry

end number_of_real_roots_of_cubic_l249_249408


namespace rachel_speed_painting_video_time_l249_249425

theorem rachel_speed_painting_video_time :
  let num_videos := 4
  let setup_time := 1
  let cleanup_time := 1
  let painting_time_per_video := 1
  let editing_time_per_video := 1.5
  (setup_time + cleanup_time + painting_time_per_video * num_videos + editing_time_per_video * num_videos) / num_videos = 3 :=
by
  sorry

end rachel_speed_painting_video_time_l249_249425


namespace common_difference_l249_249658

noncomputable def a_n (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

theorem common_difference (d : ℕ) (a1 : ℕ) (h1 : a1 = 18) (h2 : d ≠ 0) 
  (h3 : (a1 + 3 * d)^2 = a1 * (a1 + 7 * d)) : d = 2 :=
by
  sorry

end common_difference_l249_249658


namespace x1_x2_lt_one_l249_249649

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * x - log x
noncomputable def g (x : ℝ) : ℝ := x / exp x

theorem x1_x2_lt_one (k : ℝ) (x1 x2 : ℝ) (h : f x1 1 + g x1 - k = 0) (h2 : f x2 1 + g x2 - k = 0) (hx1 : 0 < x1) (hx2 : x1 < x2) : x1 * x2 < 1 :=
by
  sorry

end x1_x2_lt_one_l249_249649


namespace shoe_pair_probability_l249_249272

theorem shoe_pair_probability :
  let m := 7
  let n := 50
  (∀ (k : ℕ), k < 5 → 
    ¬ ∃ (pairs : Finset (Finset (Fin 10))), 
      pairs.card = k ∧ 
      ∀ (pair : Finset (Fin 10)), 
        pair ∈ pairs → 
        ∃ (adult_shoes : Finset (Fin 10)), 
          adult_shoes.card = k ∧ 
          adult_shoes ⊆ pair) → 
  m + n = 57 :=
by
  sorry

end shoe_pair_probability_l249_249272


namespace susan_ate_candies_l249_249885

theorem susan_ate_candies (candies_tuesday candies_thursday candies_friday candies_left : ℕ) 
  (h_tuesday : candies_tuesday = 3) 
  (h_thursday : candies_thursday = 5) 
  (h_friday : candies_friday = 2) 
  (h_left : candies_left = 4) : candies_tuesday + candies_thursday + candies_friday - candies_left = 6 := by
  sorry

end susan_ate_candies_l249_249885


namespace directrix_equation_l249_249660

-- Define the conditions
variable {O : Point} (hO : O = ⟨0, 0⟩)
variable {p : ℝ} (hp : p > 0)
variable {C : ℝ → ℝ} (hC : ∀ x y, y^2 = 2p * x)
variable {F : Point} (hF : F = ⟨p / 2, 0⟩)
variable {P : Point} (hP : P.1 = p / 2 ∧ P.2 ∈ set.range (C (p / 2))) (hPF_perpendicular_x_axis : P.x = F.x)
variable {Q : Point} (hQ : Q.y = 0)
variable {PQ_orthogonal_OP : (P.2 - Q.2) * (Q.1 - O.1) + (P.1 - Q.1) * (Q.2 - O.2) = 0)
variable {FQ_distance : |F.1 - Q.1| = 6}

-- The statement to be proven
theorem directrix_equation : ∃ p : ℝ, p = 3 → ∀ x, x = -p / 2 ↔ x = -(3 / 2) := by
  sorry

end directrix_equation_l249_249660


namespace evaluate_expression_at_x_zero_l249_249270

theorem evaluate_expression_at_x_zero (x : ℕ) (h1 : x < 3) (h2 : x ≠ 1) (h3 : x ≠ 2) : ((3 / (x - 1) - x - 1) / (x - 2) / (x^2 - 2 * x + 1)) = 2 :=
by
  -- Here we need to provide our proof, though for now it’s indicated by sorry
  sorry

end evaluate_expression_at_x_zero_l249_249270


namespace least_subtract_to_divisible_by_14_l249_249153

theorem least_subtract_to_divisible_by_14 (n : ℕ) (h : n = 7538): 
  (n % 14 = 6) -> ∃ m, (m = 6) ∧ ((n - m) % 14 = 0) :=
by
  sorry

end least_subtract_to_divisible_by_14_l249_249153


namespace candy_bag_division_l249_249206

theorem candy_bag_division (total_candy bags_candy : ℕ) (h1 : total_candy = 42) (h2 : bags_candy = 21) : 
  total_candy / bags_candy = 2 := 
by
  sorry

end candy_bag_division_l249_249206


namespace a8_equals_two_or_minus_two_l249_249244

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ r : ℝ, a (n + m) = a n * a m / a 0

theorem a8_equals_two_or_minus_two (a : ℕ → ℝ) 
    (h_geom : geometric_sequence a)
    (h_roots : ∃ x y : ℝ, x^2 - 8 * x + 4 = 0 ∧ y^2 - 8 * y + 4 = 0 ∧ a 6 = x ∧ a 10 = y) :
  a 8 = 2 ∨ a 8 = -2 :=
by
  sorry

end a8_equals_two_or_minus_two_l249_249244


namespace seashells_total_l249_249295

theorem seashells_total (tim_seashells sally_seashells : ℕ) (ht : tim_seashells = 37) (hs : sally_seashells = 13) :
  tim_seashells + sally_seashells = 50 := 
by 
  sorry

end seashells_total_l249_249295


namespace gcd_polynomial_eq_one_l249_249361

theorem gcd_polynomial_eq_one (b : ℤ) (hb : Even b) (hmb : 431 ∣ b) : 
  Int.gcd (8 * b^2 + 63 * b + 143) (4 * b + 17) = 1 := by
  sorry

end gcd_polynomial_eq_one_l249_249361


namespace leftover_value_correct_l249_249170

noncomputable def leftover_value (nickels_per_roll pennies_per_roll : ℕ) (sarah_nickels sarah_pennies tom_nickels tom_pennies : ℕ) : ℚ :=
  let total_nickels := sarah_nickels + tom_nickels
  let total_pennies := sarah_pennies + tom_pennies
  let leftover_nickels := total_nickels % nickels_per_roll
  let leftover_pennies := total_pennies % pennies_per_roll
  (leftover_nickels * 5 + leftover_pennies) / 100

theorem leftover_value_correct :
  leftover_value 40 50 132 245 98 203 = 1.98 := 
by
  sorry

end leftover_value_correct_l249_249170


namespace trapezoid_ratio_l249_249356

theorem trapezoid_ratio (A B C D M N K : Type) 
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup M] [AddCommGroup N] [AddCommGroup K]
  (CM MD CN NA AD BC : ℝ)
  (h1 : CM / MD = 4 / 3)
  (h2 : CN / NA = 4 / 3) 
  : AD / BC = 7 / 12 :=
by
  sorry

end trapezoid_ratio_l249_249356


namespace tangent_lines_ln_l249_249827

theorem tangent_lines_ln (x y: ℝ) : 
    (y = Real.log (abs x)) → 
    (x = 0 ∧ y = 0) ∨ ((x = yup ∨ x = ydown) ∧ (∀ (ey : ℝ), x = ey ∨ x = -ey)) :=
by 
    intro h
    sorry

end tangent_lines_ln_l249_249827


namespace sandy_correct_sums_l249_249311

theorem sandy_correct_sums (x y : ℕ) (h1 : x + y = 30) (h2 : 3 * x - 2 * y = 50) : x = 22 :=
  by
  sorry

end sandy_correct_sums_l249_249311


namespace inclination_angle_of_line_l249_249637

noncomputable def angle_of_inclination (m : ℝ) : ℝ :=
  Real.arctan m

theorem inclination_angle_of_line (α : ℝ) :
  angle_of_inclination (-1) = 3 * Real.pi / 4 :=
by
  sorry

end inclination_angle_of_line_l249_249637


namespace damaged_books_l249_249632

theorem damaged_books (O D : ℕ) (h1 : O = 6 * D - 8) (h2 : D + O = 69) : D = 11 :=
by
  sorry

end damaged_books_l249_249632


namespace factor_expression_l249_249909

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249909


namespace total_people_on_hike_l249_249447

theorem total_people_on_hike
  (cars : ℕ) (cars_people : ℕ)
  (taxis : ℕ) (taxis_people : ℕ)
  (vans : ℕ) (vans_people : ℕ)
  (buses : ℕ) (buses_people : ℕ)
  (minibuses : ℕ) (minibuses_people : ℕ)
  (h_cars : cars = 7) (h_cars_people : cars_people = 4)
  (h_taxis : taxis = 10) (h_taxis_people : taxis_people = 6)
  (h_vans : vans = 4) (h_vans_people : vans_people = 5)
  (h_buses : buses = 3) (h_buses_people : buses_people = 20)
  (h_minibuses : minibuses = 2) (h_minibuses_people : minibuses_people = 8) :
  cars * cars_people + taxis * taxis_people + vans * vans_people + buses * buses_people + minibuses * minibuses_people = 184 :=
by
  sorry

end total_people_on_hike_l249_249447


namespace students_total_l249_249839

theorem students_total (scavenger_hunt_students : ℕ) (ski_trip_students : ℕ) 
  (h1 : ski_trip_students = 2 * scavenger_hunt_students) 
  (h2 : scavenger_hunt_students = 4000) : 
  scavenger_hunt_students + ski_trip_students = 12000 := 
by
  sorry

end students_total_l249_249839


namespace product_of_conversions_l249_249894

-- Define the binary number 1101
def binary_number := 1101

-- Convert binary 1101 to decimal
def binary_to_decimal : ℕ := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0

-- Define the ternary number 212
def ternary_number := 212

-- Convert ternary 212 to decimal
def ternary_to_decimal : ℕ := 2 * 3^2 + 1 * 3^1 + 2 * 3^0

-- Statement to prove
theorem product_of_conversions : (binary_to_decimal) * (ternary_to_decimal) = 299 := by
  sorry

end product_of_conversions_l249_249894


namespace estimate_white_balls_l249_249390

theorem estimate_white_balls :
  (∃ x : ℕ, (6 / (x + 6) : ℝ) = 0.2 ∧ x = 24) :=
by
  use 24
  sorry

end estimate_white_balls_l249_249390


namespace convert_20202_3_l249_249625

def ternary_to_decimal (a4 a3 a2 a1 a0 : ℕ) : ℕ :=
  a4 * 3^4 + a3 * 3^3 + a2 * 3^2 + a1 * 3^1 + a0 * 3^0

theorem convert_20202_3 : ternary_to_decimal 2 0 2 0 2 = 182 :=
  sorry

end convert_20202_3_l249_249625


namespace division_example_l249_249622

theorem division_example : 0.45 / 0.005 = 90 := by
  sorry

end division_example_l249_249622


namespace find_x3_minus_y3_l249_249046

theorem find_x3_minus_y3 {x y : ℤ} (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : x^3 - y^3 = 108 :=
by 
  sorry

end find_x3_minus_y3_l249_249046


namespace pet_store_problem_l249_249871

theorem pet_store_problem 
  (initial_puppies : ℕ) 
  (sold_day1 : ℕ) 
  (sold_day2 : ℕ) 
  (sold_day3 : ℕ) 
  (sold_day4 : ℕ)
  (sold_day5 : ℕ) 
  (puppies_per_cage : ℕ)
  (initial_puppies_eq : initial_puppies = 120) 
  (sold_day1_eq : sold_day1 = 25) 
  (sold_day2_eq : sold_day2 = 10) 
  (sold_day3_eq : sold_day3 = 30) 
  (sold_day4_eq : sold_day4 = 15) 
  (sold_day5_eq : sold_day5 = 28) 
  (puppies_per_cage_eq : puppies_per_cage = 6) : 
  (initial_puppies - (sold_day1 + sold_day2 + sold_day3 + sold_day4 + sold_day5)) / puppies_per_cage = 2 := 
by 
  sorry

end pet_store_problem_l249_249871


namespace dot_product_AB_BC_in_triangle_l249_249972

theorem dot_product_AB_BC_in_triangle :
  ∀ (A B C : EuclideanSpace ℝ (Fin 2)),
  dist A B = 7 →
  dist B C = 5 →
  dist C A = 6 →
  ∥ (B - A) ∥ * ∥ (C - B) ∥ * (⟪B - A, C - B⟫) = -19 :=
by
  -- Naming the points in some finite-dimensional Euclidean space
  intros A B C hAB hBC hCA
  -- Some steps to prove the theorem will go here, but for now we use sorry
  sorry

end dot_product_AB_BC_in_triangle_l249_249972


namespace sin_300_eq_neg_sqrt_3_div_2_l249_249505

theorem sin_300_eq_neg_sqrt_3_div_2 : Real.sin (300 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt_3_div_2_l249_249505


namespace general_formula_a_n_sum_T_n_l249_249940

-- Definitions of the sequences
def a (n : ℕ) : ℕ := 4 + (n - 1) * 1
def S (n : ℕ) : ℕ := n / 2 * (2 * 4 + (n - 1) * 1)
def b (n : ℕ) : ℕ := 2 ^ (a n - 3)
def T (n : ℕ) : ℕ := 2 * (2 ^ n - 1)

-- Given conditions
axiom a4_eq_7 : a 4 = 7
axiom S2_eq_9 : S 2 = 9

-- Theorems to prove
theorem general_formula_a_n : ∀ n, a n = n + 3 := 
by sorry

theorem sum_T_n : ∀ n, T n = 2 ^ (n + 1) - 2 := 
by sorry

end general_formula_a_n_sum_T_n_l249_249940


namespace train_crosses_pole_in_1_5_seconds_l249_249735

noncomputable def time_to_cross_pole (length : ℝ) (speed_km_hr : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * (1000 / 3600)
  length / speed_m_s

theorem train_crosses_pole_in_1_5_seconds :
  time_to_cross_pole 60 144 = 1.5 :=
by
  unfold time_to_cross_pole
  -- simplified proof would be here
  sorry

end train_crosses_pole_in_1_5_seconds_l249_249735


namespace engineers_meeting_probability_l249_249844

theorem engineers_meeting_probability :
  ∀ (x y z : ℝ), 
    (0 ≤ x ∧ x ≤ 2) → 
    (0 ≤ y ∧ y ≤ 2) → 
    (0 ≤ z ∧ z ≤ 2) → 
    (abs (x - y) ≤ 0.5) → 
    (abs (y - z) ≤ 0.5) → 
    (abs (z - x) ≤ 0.5) → 
    Π (volume_region : ℝ) (total_volume : ℝ),
    (volume_region = 1.5 * 1.5 * 1.5) → 
    (total_volume = 2 * 2 * 2) → 
    (volume_region / total_volume = 0.421875) :=
by
  intros x y z hx hy hz hxy hyz hzx volume_region total_volume hr ht
  sorry

end engineers_meeting_probability_l249_249844


namespace total_boat_licenses_l249_249756

/-- A state modifies its boat license requirements to include any one of the letters A, M, or S
followed by any six digits. How many different boat licenses can now be issued? -/
theorem total_boat_licenses : 
  let letters := 3
  let digits := 10
  letters * digits^6 = 3000000 := by
  sorry

end total_boat_licenses_l249_249756


namespace percentage_equivalence_l249_249234

theorem percentage_equivalence (x : ℝ) (h : 0.3 * 0.4 * x = 36) : 0.4 * 0.3 * x = 36 :=
by
  sorry

end percentage_equivalence_l249_249234


namespace groupings_of_guides_and_tourists_l249_249294

theorem groupings_of_guides_and_tourists :
  let guides := 3
  let tourists := 8
  -- The number of different groupings where each guide has at least one tourist
  ∑ (partitions : Fin.tourists -> Fin.guides), (⧸ ∀ g : Fin.guides, ∃ t : Fin.tourists, partitions t = g) = 5796 :=
sorry

end groupings_of_guides_and_tourists_l249_249294


namespace factor_expression_l249_249914

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := 
by
  sorry

end factor_expression_l249_249914


namespace price_per_can_of_spam_l249_249957

-- Definitions of conditions
variable (S : ℝ) -- The price per can of Spam
def cost_peanut_butter := 3 * 5 -- 3 jars of peanut butter at $5 each
def cost_bread := 4 * 2 -- 4 loaves of bread at $2 each
def total_cost := 59 -- Total amount paid

-- Proof problem to verify the price per can of Spam
theorem price_per_can_of_spam :
  12 * S + cost_peanut_butter + cost_bread = total_cost → S = 3 :=
by
  sorry

end price_per_can_of_spam_l249_249957


namespace cyclic_quadrilateral_tangent_sum_l249_249744

theorem cyclic_quadrilateral_tangent_sum (A B C D : Point) (O : Point) (r : ℝ)
    (h_circ : IsCyclic {A, B, C, D})
    (h_center : O ∈ Segment A B)
    (hr : ∀ (P ∈ {A, D, B, C}), Dist P O = r)
    (h_tangent_AD : ∀ (P ∈ Segment A D), Dist P O = r)
    (h_tangent_CD : ∀ (P ∈ Segment C D), Dist P O = r)
    (h_tangent_BC : ∀ (P ∈ Segment B C), Dist P O = r) :
  (length (Segment A D) + length (Segment B C)) = length (Segment A B) :=
begin
  sorry
end

end cyclic_quadrilateral_tangent_sum_l249_249744


namespace sin_double_angle_l249_249039

theorem sin_double_angle (alpha : ℝ) (h1 : Real.cos (alpha + π / 4) = 3 / 5)
  (h2 : π / 2 ≤ alpha ∧ alpha ≤ 3 * π / 2) : Real.sin (2 * alpha) = 7 / 25 := 
sorry

end sin_double_angle_l249_249039


namespace remainder_zero_l249_249727

theorem remainder_zero :
  ∀ (a b c d : ℕ),
  a % 53 = 47 →
  b % 53 = 4 →
  c % 53 = 10 →
  d % 53 = 14 →
  (((a * b * c) % 53) * d) % 47 = 0 := 
by 
  intros a b c d h1 h2 h3 h4
  sorry

end remainder_zero_l249_249727


namespace find_original_cost_price_l249_249843

variables (P : ℝ) (A B C D E : ℝ)

-- Define the conditions as per the problem statement
def with_tax (P : ℝ) : ℝ := P * 1.10
def profit_60 (price : ℝ) : ℝ := price * 1.60
def profit_25 (price : ℝ) : ℝ := price * 1.25
def loss_15 (price : ℝ) : ℝ := price * 0.85
def profit_30 (price : ℝ) : ℝ := price * 1.30

-- The final price E is given.
def final_price (P : ℝ) : ℝ :=
  profit_30 
  (loss_15 
  (profit_25 
  (profit_60 
  (with_tax P))))

-- To find original cost price P given final price of Rs. 500.
theorem find_original_cost_price (h : final_price P = 500) : 
  P = 500 / 2.431 :=
by 
  sorry

end find_original_cost_price_l249_249843


namespace yogurt_combinations_l249_249176

theorem yogurt_combinations : (4 * Nat.choose 8 3) = 224 := by
  sorry

end yogurt_combinations_l249_249176


namespace order_of_xyz_l249_249653

variable (a b c d : ℝ)

noncomputable def x : ℝ := Real.sqrt (a * b) + Real.sqrt (c * d)
noncomputable def y : ℝ := Real.sqrt (a * c) + Real.sqrt (b * d)
noncomputable def z : ℝ := Real.sqrt (a * d) + Real.sqrt (b * c)

theorem order_of_xyz (h₁ : a > b) (h₂ : b > c) (h₃ : c > d) (h₄ : d > 0) : x a b c d > y a b c d ∧ y a b c d > z a b c d :=
by
  sorry

end order_of_xyz_l249_249653


namespace students_total_l249_249838

theorem students_total (scavenger_hunt_students : ℕ) (ski_trip_students : ℕ) 
  (h1 : ski_trip_students = 2 * scavenger_hunt_students) 
  (h2 : scavenger_hunt_students = 4000) : 
  scavenger_hunt_students + ski_trip_students = 12000 := 
by
  sorry

end students_total_l249_249838


namespace cost_of_purchase_l249_249430

theorem cost_of_purchase :
  (5 * 3) + (8 * 2) = 31 :=
by
  sorry

end cost_of_purchase_l249_249430


namespace product_lcm_gcd_l249_249726

theorem product_lcm_gcd (a b : ℕ) (h_a : a = 24) (h_b : b = 36):
  Nat.lcm a b * Nat.gcd a b = 864 :=
by
  rw [h_a, h_b]
  sorry

end product_lcm_gcd_l249_249726


namespace square_side_length_l249_249851

theorem square_side_length (p : ℝ) (h : p = 17.8) : (p / 4) = 4.45 := by
  sorry

end square_side_length_l249_249851


namespace factor_expression_l249_249916

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := 
by
  sorry

end factor_expression_l249_249916


namespace sin_of_300_degrees_l249_249561

theorem sin_of_300_degrees : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 := by
  sorry

end sin_of_300_degrees_l249_249561


namespace absolute_value_inequality_solution_l249_249991

theorem absolute_value_inequality_solution (x : ℝ) :
  |x - 2| + |x - 4| ≤ 3 ↔ (3 / 2 ≤ x ∧ x < 4) :=
by
  sorry

end absolute_value_inequality_solution_l249_249991


namespace at_least_2_boys_and_1_girl_l249_249439

noncomputable def probability_at_least_2_boys_and_1_girl (total_members : ℕ) (boys : ℕ) (girls : ℕ) (committee_size : ℕ) : ℚ :=
  let total_ways := Nat.choose total_members committee_size
  let ways_with_0_boys := Nat.choose girls committee_size
  let ways_with_1_boy := Nat.choose boys 1 * Nat.choose girls (committee_size - 1)
  let ways_with_fewer_than_2_boys := ways_with_0_boys + ways_with_1_boy
  1 - (ways_with_fewer_than_2_boys / total_ways)

theorem at_least_2_boys_and_1_girl :
  probability_at_least_2_boys_and_1_girl 32 14 18 6 = 767676 / 906192 :=
by
  sorry

end at_least_2_boys_and_1_girl_l249_249439


namespace domain_g_l249_249190

noncomputable def g (x : ℝ) : ℝ := Real.sqrt (-8 * x^2 + 14 * x - 3)

theorem domain_g :
  {x : ℝ | -8 * x^2 + 14 * x - 3 ≥ 0} = { x : ℝ | x ≤ 1 / 4 ∨ x ≥ 3 / 2 } :=
by
  sorry

end domain_g_l249_249190


namespace plan_b_rate_l249_249471

noncomputable def cost_plan_a (duration : ℕ) : ℝ :=
  if duration ≤ 4 then 0.60
  else 0.60 + 0.06 * (duration - 4)

def cost_plan_b (duration : ℕ) (rate : ℝ) : ℝ :=
  rate * duration

theorem plan_b_rate (rate : ℝ) : 
  cost_plan_a 18 = cost_plan_b 18 rate → rate = 0.08 := 
by
  -- proof goes here
  sorry

end plan_b_rate_l249_249471


namespace rectangle_area_l249_249926

noncomputable def width := 14
noncomputable def length := width + 6
noncomputable def perimeter := 2 * width + 2 * length
noncomputable def area := width * length

theorem rectangle_area (h1 : length = width + 6) (h2 : perimeter = 68) : area = 280 := 
by 
  have hw : width = 14 := by sorry 
  have hl : length = 20 := by sorry 
  have harea : area = 280 := by sorry
  exact harea

end rectangle_area_l249_249926


namespace factor_expression_l249_249904

theorem factor_expression (x : ℝ) : (x * (x + 3) + 2 * (x + 3)) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249904


namespace driving_time_ratio_l249_249250

theorem driving_time_ratio 
  (t : ℝ)
  (h : 30 * t + 60 * (2 * t) = 75) : 
  t / (2 * t) = 1 / 2 := 
by
  sorry

end driving_time_ratio_l249_249250


namespace age_of_father_l249_249273

theorem age_of_father (F C : ℕ) 
  (h1 : F = C)
  (h2 : C + 5 * 15 = 2 * (F + 15)) : 
  F = 45 := 
by 
sorry

end age_of_father_l249_249273


namespace prince_spending_l249_249020

theorem prince_spending (CDs_total : ℕ) (CDs_10_percent : ℕ) (CDs_10_cost : ℕ) (CDs_5_cost : ℕ) 
  (Prince_10_fraction : ℚ) (Prince_5_fraction : ℚ) 
  (total_10_CDs : ℕ) (total_5_CDs : ℕ) (Prince_10_CDs : ℕ) (Prince_5_CDs : ℕ) (total_cost : ℕ) :
  CDs_total = 200 →
  CDs_10_percent = 40 →
  CDs_10_cost = 10 →
  CDs_5_cost = 5 →
  Prince_10_fraction = 1/2 →
  Prince_5_fraction = 1 →
  total_10_CDs = CDs_total * CDs_10_percent / 100 →
  total_5_CDs = CDs_total - total_10_CDs →
  Prince_10_CDs = total_10_CDs * Prince_10_fraction →
  Prince_5_CDs = total_5_CDs * Prince_5_fraction →
  total_cost = (Prince_10_CDs * CDs_10_cost) + (Prince_5_CDs * CDs_5_cost) →
  total_cost = 1000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end prince_spending_l249_249020


namespace repeating_decimal_as_fraction_l249_249761

def repeating_decimal := 567 / 999

theorem repeating_decimal_as_fraction : repeating_decimal = 21 / 37 := by
  sorry

end repeating_decimal_as_fraction_l249_249761


namespace remainder_of_number_divided_by_39_l249_249743

theorem remainder_of_number_divided_by_39 
  (N : ℤ) 
  (k m : ℤ) 
  (h₁ : N % 195 = 79) 
  (h₂ : N % 273 = 109) : 
  N % 39 = 1 :=
by 
  sorry

end remainder_of_number_divided_by_39_l249_249743


namespace sin_300_eq_neg_sin_60_l249_249605

theorem sin_300_eq_neg_sin_60 :
  Real.sin (300 * Real.pi / 180) = - Real.sin (60 * Real.pi / 180) :=
by
suffices : Real.sin (300 * Real.pi / 180) = -sqrt 3 / 2
  sorry

end sin_300_eq_neg_sin_60_l249_249605


namespace evaluate_difference_of_squares_l249_249892

theorem evaluate_difference_of_squares :
  (50^2 - 30^2 = 1600) :=
by sorry

end evaluate_difference_of_squares_l249_249892


namespace sin_of_300_degrees_l249_249565

theorem sin_of_300_degrees : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 := by
  sorry

end sin_of_300_degrees_l249_249565


namespace tom_and_mary_age_l249_249685

-- Define Tom's and Mary's ages
variables (T M : ℕ)

-- Define the two given conditions
def condition1 : Prop := T^2 + M = 62
def condition2 : Prop := M^2 + T = 176

-- State the theorem
theorem tom_and_mary_age (h1 : condition1 T M) (h2 : condition2 T M) : T = 7 ∧ M = 13 :=
by {
  -- sorry acts as a placeholder for the proof
  sorry
}

end tom_and_mary_age_l249_249685


namespace solve_equation1_solve_equation2_l249_249645

-- Statement for the first equation: x^2 - 16 = 0
theorem solve_equation1 (x : ℝ) : x^2 - 16 = 0 ↔ x = 4 ∨ x = -4 :=
by sorry

-- Statement for the second equation: (x + 10)^3 + 27 = 0
theorem solve_equation2 (x : ℝ) : (x + 10)^3 + 27 = 0 ↔ x = -13 :=
by sorry

end solve_equation1_solve_equation2_l249_249645


namespace sin_300_eq_neg_sqrt_three_div_two_l249_249541

theorem sin_300_eq_neg_sqrt_three_div_two :
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt_three_div_two_l249_249541


namespace find_number_l249_249159

theorem find_number (num : ℝ) (x : ℝ) (h1 : x = 0.08999999999999998) (h2 : num / x = 0.1) : num = 0.008999999999999999 :=
by 
  sorry

end find_number_l249_249159


namespace max_min_of_f_on_interval_l249_249928

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 4 + 4 * x ^ 3 + 34

theorem max_min_of_f_on_interval :
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, f x ≤ 50) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = 50) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, 33 ≤ f x) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = 33) :=
by
  sorry

end max_min_of_f_on_interval_l249_249928


namespace sin_300_eq_neg_sqrt3_div_2_l249_249577

noncomputable def sin_300 : ℝ := sin (300 * real.pi / 180)

theorem sin_300_eq_neg_sqrt3_div_2 : 
  (∠300° = 360° - 60°) →
  (300° ∈ fourth_quadrant) →
  (∀ θ ∈ fourth_quadrant, sin θ < 0) →
  (sin (60 * real.pi / 180) = sqrt 3 / 2) →
  sin_300 = -sqrt 3 / 2 := 
by
  intros h1 h2 h3 h4
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249577


namespace fibonacci_rabbits_l249_249436

theorem fibonacci_rabbits : 
  ∀ (F : ℕ → ℕ), 
    (F 0 = 1) ∧ 
    (F 1 = 1) ∧ 
    (∀ n, F (n + 2) = F n + F (n + 1)) → 
    F 12 = 233 := 
by 
  intro F h; sorry

end fibonacci_rabbits_l249_249436


namespace find_roots_of_poly_l249_249034

-- Define the polynomial
def poly := (Polynomial.X^2 - 5 * Polynomial.X + 6) * (Polynomial.X - 3) * (Polynomial.X + 2)

-- State the theorem
theorem find_roots_of_poly : (Polynomial.roots poly).to_finset = {2, 3, -2} :=
by
  sorry

end find_roots_of_poly_l249_249034


namespace toys_left_after_two_weeks_l249_249336

theorem toys_left_after_two_weeks
  (initial_stock : ℕ)
  (sold_first_week : ℕ)
  (sold_second_week : ℕ)
  (total_stock : initial_stock = 83)
  (first_week_sales : sold_first_week = 38)
  (second_week_sales : sold_second_week = 26) :
  initial_stock - (sold_first_week + sold_second_week) = 19 :=
by
  sorry

end toys_left_after_two_weeks_l249_249336


namespace find_f_l249_249668

-- Define the conditions as hypotheses
def cond1 (f : ℕ) (p : ℕ) : Prop := f + p = 75
def cond2 (f : ℕ) (p : ℕ) : Prop := (f + p) + p = 143

-- The theorem stating that given the conditions, f must be 7
theorem find_f (f p : ℕ) (h1 : cond1 f p) (h2 : cond2 f p) : f = 7 := 
  by
  sorry

end find_f_l249_249668


namespace total_distance_fourth_fifth_days_l249_249798

theorem total_distance_fourth_fifth_days (d : ℕ) (total_distance : ℕ) (n : ℕ) (q : ℚ) 
  (S_6 : d * (1 - q^6) / (1 - q) = 378) (ratio : q = 1/2) (n_six : n = 6) : 
  (d * q^3) + (d * q^4) = 36 :=
by 
  sorry

end total_distance_fourth_fifth_days_l249_249798


namespace alpha_in_fourth_quadrant_l249_249781

def point_in_third_quadrant (α : ℝ) : Prop :=
  (Real.tan α < 0) ∧ (Real.sin α < 0)

theorem alpha_in_fourth_quadrant (α : ℝ) (h : point_in_third_quadrant α) : 
  α ∈ Set.Ioc (3 * Real.pi / 2) (2 * Real.pi) :=
by sorry

end alpha_in_fourth_quadrant_l249_249781


namespace least_integer_nk_l249_249778

noncomputable def min_nk (k : ℕ) : ℕ :=
  (5 * k + 1) / 2

theorem least_integer_nk (k : ℕ) (S : Fin 5 → Finset ℕ) :
  (∀ j : Fin 5, (S j).card = k) →
  (∀ i : Fin 4, (S i ∩ S (i + 1)).card = 0) →
  (S 4 ∩ S 0).card = 0 →
  (∃ nk, (∃ (U : Finset ℕ), (∀ j : Fin 5, S j ⊆ U) ∧ U.card = nk) ∧ nk = min_nk k) :=
by
  sorry

end least_integer_nk_l249_249778


namespace quadratic_form_completion_l249_249283

theorem quadratic_form_completion (b c : ℤ)
  (h : ∀ x:ℂ, x^2 + 520*x + 600 = (x+b)^2 + c) :
  c / b = -258 :=
by sorry

end quadratic_form_completion_l249_249283


namespace percentage_equivalence_l249_249233

theorem percentage_equivalence (x : ℝ) (h : 0.3 * 0.4 * x = 36) : 0.4 * 0.3 * x = 36 :=
by
  sorry

end percentage_equivalence_l249_249233


namespace minimum_teachers_needed_l249_249465

theorem minimum_teachers_needed
  (math_teachers : ℕ) (physics_teachers : ℕ) (chemistry_teachers : ℕ)
  (max_subjects_per_teacher : ℕ) :
  math_teachers = 7 →
  physics_teachers = 6 →
  chemistry_teachers = 5 →
  max_subjects_per_teacher = 3 →
  ∃ t : ℕ, t = 5 ∧ (t * max_subjects_per_teacher ≥ math_teachers + physics_teachers + chemistry_teachers) :=
by
  repeat { sorry }

end minimum_teachers_needed_l249_249465


namespace valid_x_for_sqrt_l249_249382

theorem valid_x_for_sqrt (x : ℝ) (hx : x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 3) : x ≥ 2 ↔ x = 3 := 
sorry

end valid_x_for_sqrt_l249_249382


namespace sin_300_eq_neg_sqrt3_div_2_l249_249619

-- Let Q be the point on the unit circle at 300 degrees counterclockwise from (1,0)
def Q : ℝ × ℝ := (1 / 2, -Real.sqrt 3 / 2)

-- Prove that the sine of 300 degrees is -√3/2
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  -- Here we assume the coordinates given in the conditions are correct.
  have hQ : Q = (1 / 2, -Real.sqrt 3 / 2) := rfl
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249619


namespace min_value_frac_add_x_l249_249859

theorem min_value_frac_add_x (x : ℝ) (h : x > 3) : (∃ m, (∀ (y : ℝ), y > 3 → (4 / y - 3 + y) ≥ m) ∧ m = 7) :=
sorry

end min_value_frac_add_x_l249_249859


namespace sin_300_eq_neg_sqrt3_div_2_l249_249500

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249500


namespace frequency_of_fourth_group_l249_249345

theorem frequency_of_fourth_group (f₁ f₂ f₃ f₄ f₅ f₆ : ℝ) (h1 : f₁ + f₂ + f₃ = 0.65) (h2 : f₅ + f₆ = 0.32) (h3 : f₁ + f₂ + f₃ + f₄ + f₅ + f₆ = 1) :
  f₄ = 0.03 :=
by 
  sorry

end frequency_of_fourth_group_l249_249345


namespace sin_300_eq_neg_sqrt_three_div_two_l249_249543

theorem sin_300_eq_neg_sqrt_three_div_two :
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt_three_div_two_l249_249543


namespace Jana_taller_than_Kelly_l249_249083

-- Definitions and given conditions
def Jess_height := 72
def Jana_height := 74
def Kelly_height := Jess_height - 3

-- Proof statement
theorem Jana_taller_than_Kelly : Jana_height - Kelly_height = 5 := by
  sorry

end Jana_taller_than_Kelly_l249_249083


namespace find_C_equation_l249_249654

def M : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 0], ![0, 2]]
def N : Matrix (Fin 2) (Fin 2) ℝ := ![![0, -1], ![1, 0]]

def C2_equation (x y : ℝ) : Prop := y = (1/8) * x^2

theorem find_C_equation (x y : ℝ) :
  (C2_equation (x) y) → (y^2 = 2 * x) := 
sorry

end find_C_equation_l249_249654


namespace arith_geo_seq_prop_l249_249044

theorem arith_geo_seq_prop (a1 a2 b1 b2 b3 : ℝ)
  (arith_seq_condition : 1 + 2 * (a1 - 1) = a2)
  (geo_seq_condition1 : b1 * b3 = 4)
  (geo_seq_condition2 : b1 > 0)
  (geo_seq_condition3 : 1 * b1 * b2 * b3 * 4 = (b1 * b3 * -4)) :
  (a2 - a1) / b2 = 1/2 :=
by
  sorry

end arith_geo_seq_prop_l249_249044


namespace smallest_number_of_fruits_l249_249031

theorem smallest_number_of_fruits 
  (n_apple_slices : ℕ) (n_grapes : ℕ) (n_orange_wedges : ℕ) (n_cherries : ℕ)
  (h_apple : n_apple_slices = 18)
  (h_grape : n_grapes = 9)
  (h_orange : n_orange_wedges = 12)
  (h_cherry : n_cherries = 6)
  : ∃ (n : ℕ), n = 36 ∧ (n % n_apple_slices = 0) ∧ (n % n_grapes = 0) ∧ (n % n_orange_wedges = 0) ∧ (n % n_cherries = 0) :=
sorry

end smallest_number_of_fruits_l249_249031


namespace factor_expression_l249_249917

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := 
by
  sorry

end factor_expression_l249_249917


namespace distinct_real_roots_unique_l249_249948

theorem distinct_real_roots_unique :
  (∃ k : ℕ, (|x| - (4 / x) = (3 * |x|) / x) → k = 1) := sorry

end distinct_real_roots_unique_l249_249948


namespace find_vector_n_l249_249217

variable (a b : ℝ)

def is_orthogonal (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = 0

def is_same_magnitude (m n : ℝ × ℝ) : Prop :=
  m.1 ^ 2 + m.2 ^ 2 = n.1 ^ 2 + n.2 ^ 2

theorem find_vector_n (m n : ℝ × ℝ) (h1 : is_orthogonal m n) (h2 : is_same_magnitude m n) :
  n = (b, -a) :=
  sorry

end find_vector_n_l249_249217


namespace orange_orchard_land_l249_249863

theorem orange_orchard_land (F H : ℕ) 
  (h1 : F + H = 120) 
  (h2 : ∃ x : ℕ, x + (2 * x + 1) = 10) 
  (h3 : ∃ x : ℕ, 2 * x + 1 = H)
  (h4 : ∃ x : ℕ, F = x) 
  (h5 : ∃ y : ℕ, H = 2 * y + 1) :
  F = 36 ∧ H = 84 :=
by
  sorry

end orange_orchard_land_l249_249863


namespace ratio_area_II_to_III_l249_249426

-- Define the properties of the squares as given in the conditions
def perimeter_region_I : ℕ := 16
def perimeter_region_II : ℕ := 32
def side_length_region_I := perimeter_region_I / 4
def side_length_region_II := perimeter_region_II / 4
def side_length_region_III := 2 * side_length_region_II
def area_region_II := side_length_region_II ^ 2
def area_region_III := side_length_region_III ^ 2

-- Prove that the ratio of the area of region II to the area of region III is 1/4
theorem ratio_area_II_to_III : (area_region_II : ℚ) / (area_region_III : ℚ) = 1 / 4 := 
by sorry

end ratio_area_II_to_III_l249_249426


namespace max_min_values_of_function_l249_249929

theorem max_min_values_of_function :
  let f := (fun x : ℝ => 3 * x^4 + 4 * x^3 + 34)
  ∃ (max min : ℝ), (∀ x ∈ Icc (-2 : ℝ) 1, f x ≤ max) ∧ (∀ x ∈ Icc (-2 : ℝ) 1, min ≤ f x) ∧
                   max = f (-2) ∧ max = 50 ∧
                   min = f (-1) ∧ min = 33 :=
by
  let f := (fun x : ℝ => 3 * x^4 + 4 * x^3 + 34)
  use 50, 33
  have h₁ : ∀ x, f' x = 12 * x^2 * (x + 1), from sorry,
  have critical_points : {x | f' x = 0} = {0, -1}, from sorry,
  -- Check values at endpoints and critical points
  have h_f_neg2 : f (-2) = 50 := by simp [f],
  have h_f_1 : f 1 = 41 := by simp [f],
  have h_f_neg1 : f (-1) = 33 := by simp [f],
  have h_f_0 : f 0 = 34 := by simp [f],
  split,
  -- Proving max value
  { intro x,
    intro hx,
    by_cases hx0 : x = -2 ∨ x = 1 ∨ x = -1 ∨ x = 0,
    any_goals { finish },
    -- f(-2) = 50, rest points have lower values
    show f x ≤ 50, from sorry,},
  -- Proving min value
  split,
  { intro x,
    intro hx,
    by_cases hx0 : x = -2 ∨ x = 1 ∨ x = -1 ∨ x = 0,
    any_goals { finish },
    -- f(-1) = 33, rest points have higher values
    show 33 ≤ f x, from sorry,},
  -- Verifying calculated max and min points are as expected
  repeat {split}; assumption
  sorry

end max_min_values_of_function_l249_249929


namespace peregrines_eat_30_percent_l249_249137

theorem peregrines_eat_30_percent (initial_pigeons : ℕ) (chicks_per_pigeon : ℕ) (pigeons_left : ℕ) :
  initial_pigeons = 40 →
  chicks_per_pigeon = 6 →
  pigeons_left = 196 →
  (100 * (initial_pigeons * chicks_per_pigeon + initial_pigeons - pigeons_left)) / 
  (initial_pigeons * chicks_per_pigeon + initial_pigeons) = 30 :=
by
  intros
  sorry

end peregrines_eat_30_percent_l249_249137


namespace find_g_plus_h_l249_249193

theorem find_g_plus_h (g h : ℚ) (d : ℚ) 
  (h_prod : (7 * d^2 - 4 * d + g) * (3 * d^2 + h * d - 9) = 21 * d^4 - 49 * d^3 - 44 * d^2 + 17 * d - 24) :
  g + h = -107 / 24 :=
sorry

end find_g_plus_h_l249_249193


namespace sin_300_eq_neg_sin_60_l249_249609

theorem sin_300_eq_neg_sin_60 :
  Real.sin (300 * Real.pi / 180) = - Real.sin (60 * Real.pi / 180) :=
by
suffices : Real.sin (300 * Real.pi / 180) = -sqrt 3 / 2
  sorry

end sin_300_eq_neg_sin_60_l249_249609


namespace spinsters_count_l249_249284

variable (S C : ℕ)

-- defining the conditions
def ratio_condition (S C : ℕ) : Prop := 9 * S = 2 * C
def difference_condition (S C : ℕ) : Prop := C = S + 63

-- theorem to prove
theorem spinsters_count 
  (h1 : ratio_condition S C) 
  (h2 : difference_condition S C) : 
  S = 18 :=
sorry

end spinsters_count_l249_249284


namespace probability_three_dice_less_than_seven_l249_249340

open Nat

def probability_of_exactly_three_less_than_seven (dice_count : ℕ) (sides : ℕ) (target_faces : ℕ) : ℚ :=
  let p : ℚ := target_faces / sides
  let q : ℚ := 1 - p
  (Nat.choose dice_count (dice_count / 2)) * (p^(dice_count / 2)) * (q^(dice_count / 2))

theorem probability_three_dice_less_than_seven :
  probability_of_exactly_three_less_than_seven 6 12 6 = 5 / 16 := by
  sorry

end probability_three_dice_less_than_seven_l249_249340


namespace sin_300_eq_neg_one_half_l249_249539

-- Definitions from the conditions
def point_on_unit_circle (θ : ℝ) : ℝ × ℝ :=
  (Real.cos (θ), Real.sin (θ))

def foot_of_perpendicular (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, 0)

def is_thirty_sixty_ninety_triangle (P : ℝ × ℝ) : Prop :=
  sqrt(P.1^2 + P.2^2) = 1 ∧ P.1 = sqrt(3) / 2 ∧ P.2 = -1/2

-- Problem statement
theorem sin_300_eq_neg_one_half : Real.sin (5 * Real.pi / 3) = -1/2 := by
  sorry

end sin_300_eq_neg_one_half_l249_249539


namespace example_problem_l249_249323

-- Define the numbers of students in each grade
def freshmen : ℕ := 240
def sophomores : ℕ := 260
def juniors : ℕ := 300

-- Define the total number of spots for the trip
def total_spots : ℕ := 40

-- Define the total number of students
def total_students : ℕ := freshmen + sophomores + juniors

-- Define the fraction of sophomores relative to the total number of students
def fraction_sophomores : ℚ := sophomores / total_students

-- Define the number of spots allocated to sophomores
def spots_sophomores : ℚ := fraction_sophomores * total_spots

-- The theorem we need to prove
theorem example_problem : spots_sophomores = 13 :=
by 
  sorry

end example_problem_l249_249323


namespace find_x_l249_249932

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : x * floor x = 50) : x = 7.142857 :=
by
  sorry

end find_x_l249_249932


namespace sin_300_eq_neg_sqrt3_div_2_l249_249497

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249497


namespace chosen_number_l249_249752

theorem chosen_number (x : ℕ) (h : (x / 12) - 240 = 8) : x = 2976 :=
sorry

end chosen_number_l249_249752


namespace surface_area_increase_factor_l249_249161

theorem surface_area_increase_factor (n : ℕ) (h : n > 0) : 
  (6 * n^3) / (6 * n^2) = n :=
by {
  sorry -- Proof not required
}

end surface_area_increase_factor_l249_249161


namespace jack_finishes_book_in_13_days_l249_249082

def total_pages : ℕ := 285
def pages_per_day : ℕ := 23

theorem jack_finishes_book_in_13_days : (total_pages + pages_per_day - 1) / pages_per_day = 13 := by
  sorry

end jack_finishes_book_in_13_days_l249_249082


namespace sin_300_eq_neg_sqrt3_div_2_l249_249578

noncomputable def sin_300 : ℝ := sin (300 * real.pi / 180)

theorem sin_300_eq_neg_sqrt3_div_2 : 
  (∠300° = 360° - 60°) →
  (300° ∈ fourth_quadrant) →
  (∀ θ ∈ fourth_quadrant, sin θ < 0) →
  (sin (60 * real.pi / 180) = sqrt 3 / 2) →
  sin_300 = -sqrt 3 / 2 := 
by
  intros h1 h2 h3 h4
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249578


namespace shifted_sine_odd_function_l249_249684

theorem shifted_sine_odd_function (ϕ : ℝ) (hϕ : 0 < ϕ ∧ ϕ < π) :
  ∃ k : ℤ, ϕ = (2 * π / 3) + k * π ∧ 0 < (2 * π / 3) + k * π ∧ (2 * π / 3) + k * π < π :=
sorry

end shifted_sine_odd_function_l249_249684


namespace marble_probability_l249_249985

theorem marble_probability :
  let p_other := 0.4 in
  let draws := 5 in
  (\Sigma i in range draws, p_other) = 2 :=
by
  sorry

end marble_probability_l249_249985


namespace sin_300_eq_neg_sqrt_three_div_two_l249_249548

theorem sin_300_eq_neg_sqrt_three_div_two :
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt_three_div_two_l249_249548


namespace infinite_solutions_or_no_solutions_l249_249817

theorem infinite_solutions_or_no_solutions (a b : ℚ) :
  (∃ (x y : ℚ), a * x^2 + b * y^2 = 1) →
  (∀ (k : ℚ), a * k^2 + b ≠ 0 → ∃ (x_k y_k : ℚ), a * x_k^2 + b * y_k^2 = 1) :=
by
  intro h_sol h_k
  sorry

end infinite_solutions_or_no_solutions_l249_249817


namespace distinct_real_roots_eq_one_l249_249949

theorem distinct_real_roots_eq_one : 
  (∃ x : ℝ, |x| - 4/x = (3 * |x|) / x) ∧ 
  ¬∃ x1 x2 : ℝ, 
    x1 ≠ x2 ∧ 
    (|x1| - 4/x1 = (3 * |x1|) / x1) ∧ 
    (|x2| - 4/x2 = (3 * |x2|) / x2) :=
sorry

end distinct_real_roots_eq_one_l249_249949


namespace weather_on_july_15_l249_249022

theorem weather_on_july_15 
  (T: ℝ) (sunny: Prop) (W: ℝ) (crowded: Prop) 
  (h1: (T ≥ 85 ∧ sunny ∧ W < 15) → crowded) 
  (h2: ¬ crowded) : (T < 85 ∨ ¬ sunny ∨ W ≥ 15) :=
sorry

end weather_on_july_15_l249_249022


namespace ratio_pow_eq_l249_249055

theorem ratio_pow_eq {x y : ℝ} (h : x / y = 7 / 5) : (x^3 / y^2) = 343 / 25 :=
by sorry

end ratio_pow_eq_l249_249055


namespace sin_300_l249_249596

-- Definitions and conditions in Lean 4
def angle_deg := ℝ
def sin (θ : angle_deg) : ℝ := Real.sin (θ * Real.pi / 180)

noncomputable def Q_angle : angle_deg := 300
noncomputable def first_quadrant_equivalent_angle : angle_deg := 360 - 300
noncomputable def sin_60 : ℝ := Real.sin (60 * Real.pi / 180)

-- Main statement to prove
theorem sin_300 : sin 300 = -sin_60 := by
  sorry

end sin_300_l249_249596


namespace John_next_birthday_age_l249_249977

variable (John Mike Lucas : ℝ)

def John_is_25_percent_older_than_Mike := John = 1.25 * Mike
def Mike_is_30_percent_younger_than_Lucas := Mike = 0.7 * Lucas
def sum_of_ages_is_27_point_3_years := John + Mike + Lucas = 27.3

theorem John_next_birthday_age 
  (h1 : John_is_25_percent_older_than_Mike John Mike) 
  (h2 : Mike_is_30_percent_younger_than_Lucas Mike Lucas) 
  (h3 : sum_of_ages_is_27_point_3_years John Mike Lucas) : 
  John + 1 = 10 := 
sorry

end John_next_birthday_age_l249_249977


namespace sin_300_eq_neg_sqrt3_div_2_l249_249617

-- Let Q be the point on the unit circle at 300 degrees counterclockwise from (1,0)
def Q : ℝ × ℝ := (1 / 2, -Real.sqrt 3 / 2)

-- Prove that the sine of 300 degrees is -√3/2
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  -- Here we assume the coordinates given in the conditions are correct.
  have hQ : Q = (1 / 2, -Real.sqrt 3 / 2) := rfl
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249617


namespace difference_between_numbers_l249_249715

theorem difference_between_numbers (a b : ℕ) (h1 : a + b = 27630) (h2 : a = 5 * b + 5) : a - b = 18421 :=
  sorry

end difference_between_numbers_l249_249715


namespace radius_ratio_eq_inv_sqrt_5_l249_249392

noncomputable def ratio_of_radii (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) : ℝ :=
  a / b

theorem radius_ratio_eq_inv_sqrt_5 (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) : 
  ratio_of_radii a b h = 1 / Real.sqrt 5 :=
sorry

end radius_ratio_eq_inv_sqrt_5_l249_249392


namespace one_eq_one_of_ab_l249_249374

variable {a b : ℝ}

theorem one_eq_one_of_ab (h : a * b = a^2 - a * b + b^2) : 1 = 1 := by
  sorry

end one_eq_one_of_ab_l249_249374


namespace find_line_eq_l249_249132

theorem find_line_eq
  (l : ℝ → ℝ → Prop)
  (bisects_circle : ∀ x y : ℝ, x^2 + y^2 - 2*x - 4*y = 0 → l x y)
  (perpendicular_to_line : ∀ x y : ℝ, l x y ↔ y = -1/2 * x)
  : ∀ x y : ℝ, l x y ↔ 2*x - y = 0 := by
  sorry

end find_line_eq_l249_249132


namespace opposite_of_neg_five_l249_249445

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l249_249445


namespace sin_300_eq_neg_sqrt_3_div_2_l249_249520

noncomputable def sin_300_degrees : ℝ := Real.sin (300 * Real.pi / 180)

theorem sin_300_eq_neg_sqrt_3_div_2 : sin_300_degrees = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_300_eq_neg_sqrt_3_div_2_l249_249520


namespace total_trout_caught_l249_249099

theorem total_trout_caught (n_share j_share total_caught : ℕ) (h1 : n_share = 9) (h2 : j_share = 9) (h3 : total_caught = n_share + j_share) :
  total_caught = 18 :=
by
  sorry

end total_trout_caught_l249_249099


namespace polynomial_remainder_theorem_l249_249092

open Polynomial

theorem polynomial_remainder_theorem (Q : Polynomial ℝ)
  (h1 : Q.eval 20 = 120)
  (h2 : Q.eval 100 = 40) :
  ∃ R : Polynomial ℝ, R.degree < 2 ∧ Q = (X - 20) * (X - 100) * R + (-X + 140) :=
by
  sorry

end polynomial_remainder_theorem_l249_249092


namespace penny_makes_from_cheesecakes_l249_249481

-- Definitions based on the conditions
def slices_per_pie : ℕ := 6
def cost_per_slice : ℕ := 7
def pies_sold : ℕ := 7

-- The mathematical equivalent proof problem
theorem penny_makes_from_cheesecakes : slices_per_pie * cost_per_slice * pies_sold = 294 := by
  sorry

end penny_makes_from_cheesecakes_l249_249481


namespace factor_expression_l249_249905

theorem factor_expression (x : ℝ) : (x * (x + 3) + 2 * (x + 3)) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249905


namespace ratio_of_radii_l249_249395

variable (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2)

theorem ratio_of_radii (ha : a > 0) (hb : b > 0) : (a / b = 1 / Real.sqrt 5) :=
by
  sorry

end ratio_of_radii_l249_249395


namespace factor_expression_l249_249912

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249912


namespace jog_to_coffee_shop_l249_249788

def constant_pace_jogging (time_to_park : ℕ) (dist_to_park : ℝ) (dist_to_coffee_shop : ℝ) : Prop :=
  time_to_park / dist_to_park * dist_to_coffee_shop = 6

theorem jog_to_coffee_shop
  (time_to_park : ℕ)
  (dist_to_park : ℝ)
  (dist_to_coffee_shop : ℝ)
  (h1 : time_to_park = 12)
  (h2 : dist_to_park = 1.5)
  (h3 : dist_to_coffee_shop = 0.75)
: constant_pace_jogging time_to_park dist_to_park dist_to_coffee_shop :=
by sorry

end jog_to_coffee_shop_l249_249788


namespace algebraic_expression_value_l249_249379

theorem algebraic_expression_value (x y : ℝ) (h1 : x + 2 * y = 4) (h2 : x - 2 * y = -1) :
  x^2 - 4 * y^2 + 1 = -3 := by
  sorry

end algebraic_expression_value_l249_249379


namespace initial_velocity_calculation_l249_249468

-- Define conditions
def acceleration_due_to_gravity := 10 -- m/s^2
def time_to_highest_point := 2 -- s
def velocity_at_highest_point := 0 -- m/s
def initial_observed_acceleration := 15 -- m/s^2

-- Theorem to prove the initial velocity
theorem initial_velocity_calculation
  (a_gravity : ℝ := acceleration_due_to_gravity)
  (t_highest : ℝ := time_to_highest_point)
  (v_highest : ℝ := velocity_at_highest_point)
  (a_initial : ℝ := initial_observed_acceleration) :
  ∃ (v_initial : ℝ), v_initial = 30 := 
sorry

end initial_velocity_calculation_l249_249468


namespace right_triangle_other_side_l249_249692

theorem right_triangle_other_side (a b c : ℕ) (h : a^2 + b^2 = c^2) (h_c : c = 17) (h_a : a = 15) : b = 8 := 
by
  sorry

end right_triangle_other_side_l249_249692


namespace ellipse_equation_parabola_equation_l249_249644

noncomputable def ellipse_standard_equation (a b c : ℝ) : Prop :=
  a = 6 → b = 2 * Real.sqrt 5 → c = 4 → 
  ((∀ x y : ℝ, (y^2 / 36) + (x^2 / 20) = 1))

noncomputable def parabola_standard_equation (focus_x focus_y : ℝ) : Prop :=
  focus_x = 3 → focus_y = 0 → 
  (∀ x y : ℝ, y^2 = 12 * x)

theorem ellipse_equation : ellipse_standard_equation 6 (2 * Real.sqrt 5) 4 := by
  sorry

theorem parabola_equation : parabola_standard_equation 3 0 := by
  sorry

end ellipse_equation_parabola_equation_l249_249644


namespace avg_eggs_per_nest_l249_249074

/-- In the Caribbean, loggerhead turtles lay three million eggs in twenty thousand nests. 
On average, show that there are 150 eggs in each nest. -/

theorem avg_eggs_per_nest 
  (total_eggs : ℕ) 
  (total_nests : ℕ) 
  (h1 : total_eggs = 3000000) 
  (h2 : total_nests = 20000) :
  total_eggs / total_nests = 150 := 
by {
  sorry
}

end avg_eggs_per_nest_l249_249074


namespace tan_alpha_value_l249_249646

theorem tan_alpha_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin α * Real.cos α = 1 / 4) :
  Real.tan α = 2 - Real.sqrt 3 ∨ Real.tan α = 2 + Real.sqrt 3 :=
sorry

end tan_alpha_value_l249_249646


namespace percent_problem_l249_249230

theorem percent_problem (x : ℝ) (h : 0.30 * 0.40 * x = 36) : 0.40 * 0.30 * x = 36 :=
by
  sorry

end percent_problem_l249_249230


namespace inequality_problem_l249_249652

theorem inequality_problem
  (a b c d e : ℝ)
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : c ≤ d)
  (h4 : d ≤ e)
  (h5 : a + b + c + d + e = 1) :
  a * d + d * c + c * b + b * e + e * a ≤ 1 / 5 :=
by
  sorry

end inequality_problem_l249_249652


namespace percent_problem_l249_249229

theorem percent_problem (x : ℝ) (h : 0.30 * 0.40 * x = 36) : 0.40 * 0.30 * x = 36 :=
by
  sorry

end percent_problem_l249_249229


namespace percentage_transform_l249_249227

theorem percentage_transform (n : ℝ) (h : 0.3 * 0.4 * n = 36) : 0.4 * 0.3 * n = 36 :=
by
  sorry

end percentage_transform_l249_249227


namespace machine_value_correct_l249_249165

-- The present value of the machine
def present_value : ℝ := 1200

-- The depreciation rate function based on the year
def depreciation_rate (year : ℕ) : ℝ :=
  match year with
  | 1 => 0.10
  | 2 => 0.12
  | n => if n > 2 then 0.10 + 0.02 * (n - 1) else 0

-- The repair rate
def repair_rate : ℝ := 0.03

-- Value of the machine after n years
noncomputable def machine_value_after_n_years (initial_value : ℝ) (n : ℕ) : ℝ :=
  let value_first_year := (initial_value - (depreciation_rate 1 * initial_value)) + (repair_rate * initial_value)
  let value_second_year := (value_first_year - (depreciation_rate 2 * value_first_year)) + (repair_rate * value_first_year)
  match n with
  | 1 => value_first_year
  | 2 => value_second_year
  | _ => sorry -- Further generalization would be required for n > 2

-- Theorem statement
theorem machine_value_correct (initial_value : ℝ) :
  machine_value_after_n_years initial_value 2 = 1015.56 := by
  sorry

end machine_value_correct_l249_249165


namespace find_x3_y3_l249_249049

noncomputable def x_y_conditions (x y : ℝ) :=
  x - y = 3 ∧
  x^2 + y^2 = 27

theorem find_x3_y3 (x y : ℝ) (h : x_y_conditions x y) : x^3 - y^3 = 108 :=
  sorry

end find_x3_y3_l249_249049


namespace alex_silver_tokens_l249_249878

-- Definitions and conditions
def initialRedTokens : ℕ := 100
def initialBlueTokens : ℕ := 50
def firstBoothRedChange (x : ℕ) : ℕ := 3 * x
def firstBoothSilverGain (x : ℕ) : ℕ := 2 * x
def firstBoothBlueGain (x : ℕ) : ℕ := x
def secondBoothBlueChange (y : ℕ) : ℕ := 2 * y
def secondBoothSilverGain (y : ℕ) : ℕ := y
def secondBoothRedGain (y : ℕ) : ℕ := y

-- Final conditions when no more exchanges are possible
def finalRedTokens (x y : ℕ) : ℕ := initialRedTokens - firstBoothRedChange x + secondBoothRedGain y
def finalBlueTokens (x y : ℕ) : ℕ := initialBlueTokens + firstBoothBlueGain x - secondBoothBlueChange y

-- Total silver tokens calculation
def totalSilverTokens (x y : ℕ) : ℕ := firstBoothSilverGain x + secondBoothSilverGain y

-- Proof that in the end, Alex has 147 silver tokens
theorem alex_silver_tokens : 
  ∃ (x y : ℕ), finalRedTokens x y = 2 ∧ finalBlueTokens x y = 1 ∧ totalSilverTokens x y = 147 :=
by
  -- the proof logic will be filled here
  sorry

end alex_silver_tokens_l249_249878


namespace total_amount_Rs20_l249_249109

theorem total_amount_Rs20 (x y z : ℕ) 
(h1 : x + y + z = 130) 
(h2 : 95 * x + 45 * y + 20 * z = 7000) : 
∃ z : ℕ, (20 * z) = (7000 - 95 * x - 45 * y) / 20 := sorry

end total_amount_Rs20_l249_249109


namespace sin_of_300_degrees_l249_249563

theorem sin_of_300_degrees : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 := by
  sorry

end sin_of_300_degrees_l249_249563


namespace scientific_notation_28400_is_correct_l249_249018

theorem scientific_notation_28400_is_correct : (28400 : ℝ) = 2.84 * 10^4 := 
by 
  sorry

end scientific_notation_28400_is_correct_l249_249018


namespace sin_300_eq_neg_sqrt3_div_2_l249_249614

-- Let Q be the point on the unit circle at 300 degrees counterclockwise from (1,0)
def Q : ℝ × ℝ := (1 / 2, -Real.sqrt 3 / 2)

-- Prove that the sine of 300 degrees is -√3/2
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  -- Here we assume the coordinates given in the conditions are correct.
  have hQ : Q = (1 / 2, -Real.sqrt 3 / 2) := rfl
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249614


namespace sin_of_300_degrees_l249_249559

theorem sin_of_300_degrees : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 := by
  sorry

end sin_of_300_degrees_l249_249559


namespace base_seven_sum_of_digits_of_product_l249_249840

theorem base_seven_sum_of_digits_of_product :
  let a := 24
  let b := 30
  let product := a * b
  let base_seven_product := 105 -- The product in base seven notation
  let sum_of_digits (n : ℕ) : ℕ := n.digits 7 |> List.sum
  sum_of_digits base_seven_product = 6 :=
by
  sorry

end base_seven_sum_of_digits_of_product_l249_249840


namespace real_roots_of_quadratic_l249_249961

theorem real_roots_of_quadratic (m : ℝ) :
  (∃ x : ℝ, m * x^2 - 4 * x + 3 = 0) ↔ m ≤ 4 / 3 :=
by
  sorry

end real_roots_of_quadratic_l249_249961


namespace square_of_radius_l249_249864

theorem square_of_radius 
  (AP PB CQ QD : ℝ) 
  (hAP : AP = 25)
  (hPB : PB = 35)
  (hCQ : CQ = 30)
  (hQD : QD = 40) 
  : ∃ r : ℝ, r^2 = 13325 := 
sorry

end square_of_radius_l249_249864


namespace second_tap_empties_cistern_l249_249866

theorem second_tap_empties_cistern (t_fill: ℝ) (x: ℝ) (t_net: ℝ) : 
  (1 / 6) - (1 / x) = (1 / 12) → x = 12 := 
by
  sorry

end second_tap_empties_cistern_l249_249866


namespace jar_and_beans_weight_is_60_percent_l249_249716

theorem jar_and_beans_weight_is_60_percent
  (J B : ℝ)
  (h1 : J = 0.10 * (J + B))
  (h2 : ∃ x : ℝ, x = 0.5555555555555556 ∧ (J + x * B = 0.60 * (J + B))) :
  J + 0.5555555555555556 * B = 0.60 * (J + B) :=
by
  sorry

end jar_and_beans_weight_is_60_percent_l249_249716


namespace sin_300_eq_neg_sqrt3_div_2_l249_249579

noncomputable def sin_300 : ℝ := sin (300 * real.pi / 180)

theorem sin_300_eq_neg_sqrt3_div_2 : 
  (∠300° = 360° - 60°) →
  (300° ∈ fourth_quadrant) →
  (∀ θ ∈ fourth_quadrant, sin θ < 0) →
  (sin (60 * real.pi / 180) = sqrt 3 / 2) →
  sin_300 = -sqrt 3 / 2 := 
by
  intros h1 h2 h3 h4
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249579


namespace sin_300_deg_l249_249493

noncomputable def sin_300_equals_neg_sqrt3_div_2 : Prop :=
  sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2

theorem sin_300_deg : sin_300_equals_neg_sqrt3_div_2 :=
by
  sorry

end sin_300_deg_l249_249493


namespace fraction_of_students_between_11_and_13_is_two_fifths_l249_249815

def totalStudents : ℕ := 45
def under11 : ℕ :=  totalStudents / 3
def over13 : ℕ := 12
def between11and13 : ℕ := totalStudents - (under11 + over13)
def fractionBetween11and13 : ℚ := between11and13 / totalStudents

theorem fraction_of_students_between_11_and_13_is_two_fifths :
  fractionBetween11and13 = 2 / 5 := 
by 
  sorry

end fraction_of_students_between_11_and_13_is_two_fifths_l249_249815


namespace totalTrianglesInFigure_l249_249222

-- Definition of the problem involving a rectangle with subdivisions creating triangles
def numberOfTrianglesInRectangle : Nat :=
  let smallestTriangles := 24   -- Number of smallest triangles
  let nextSizeTriangles1 := 8   -- Triangles formed by combining smallest triangles
  let nextSizeTriangles2 := 12
  let nextSizeTriangles3 := 16
  let largestTriangles := 4
  smallestTriangles + nextSizeTriangles1 + nextSizeTriangles2 + nextSizeTriangles3 + largestTriangles

-- The Lean 4 theorem statement, stating that the total number of triangles equals 64
theorem totalTrianglesInFigure : numberOfTrianglesInRectangle = 64 := 
by
  sorry

end totalTrianglesInFigure_l249_249222


namespace cubic_inequality_l249_249703

theorem cubic_inequality (x p q : ℝ) (h : x^3 + p * x + q = 0) : 4 * q * x ≤ p^2 := 
  sorry

end cubic_inequality_l249_249703


namespace jayden_planes_l249_249084

theorem jayden_planes (W : ℕ) (wings_per_plane : ℕ) (total_wings : W = 108) (wpp_pos : wings_per_plane = 2) :
  ∃ n : ℕ, n = W / wings_per_plane ∧ n = 54 :=
by
  sorry

end jayden_planes_l249_249084


namespace triangle_area_l249_249247

theorem triangle_area {a c : ℝ} (h_a : a = 3 * Real.sqrt 3) (h_c : c = 2) (angle_B : ℝ) (h_B : angle_B = Real.pi / 3) : 
  (1 / 2) * a * c * Real.sin angle_B = 9 / 2 :=
by
  rw [h_a, h_c, h_B]
  sorry

end triangle_area_l249_249247


namespace least_possible_k_l249_249960

-- Define the conditions
def prime_factor_form (k : ℕ) : Prop :=
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ k = 2^a * 3^b * 5^c

def divisible_by_1680 (k : ℕ) : Prop :=
  (k ^ 4) % 1680 = 0

-- Define the proof problem
theorem least_possible_k (k : ℕ) (h_div : divisible_by_1680 k) (h_prime : prime_factor_form k) : k = 210 :=
by
  -- Statement of the problem, proof to be filled
  sorry

end least_possible_k_l249_249960


namespace positive_integer_with_four_smallest_divisors_is_130_l249_249636

theorem positive_integer_with_four_smallest_divisors_is_130:
  ∃ n : ℕ, ∀ p1 p2 p3 p4 : ℕ, 
    n = p1^2 + p2^2 + p3^2 + p4^2 ∧
    p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧
    ∀ p : ℕ, p ∣ n → (p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4) → 
    n = 130 :=
  by
  sorry

end positive_integer_with_four_smallest_divisors_is_130_l249_249636


namespace sin_300_eq_neg_sqrt3_div_2_l249_249502

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249502


namespace evaluate_f_at_5_l249_249256

def f (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 38*x^2 - 35*x - 40

theorem evaluate_f_at_5 : f 5 = 110 :=
by
  sorry

end evaluate_f_at_5_l249_249256


namespace shopkeeper_percentage_gain_l249_249856

theorem shopkeeper_percentage_gain 
    (original_price : ℝ) 
    (price_increase : ℝ) 
    (first_discount : ℝ) 
    (second_discount : ℝ)
    (new_price : ℝ) 
    (discounted_price1 : ℝ) 
    (final_price : ℝ) 
    (percentage_gain : ℝ) 
    (h1 : original_price = 100)
    (h2 : price_increase = original_price * 0.34)
    (h3 : new_price = original_price + price_increase)
    (h4 : first_discount = new_price * 0.10)
    (h5 : discounted_price1 = new_price - first_discount)
    (h6 : second_discount = discounted_price1 * 0.15)
    (h7 : final_price = discounted_price1 - second_discount)
    (h8 : percentage_gain = ((final_price - original_price) / original_price) * 100) :
    percentage_gain = 2.51 :=
by sorry

end shopkeeper_percentage_gain_l249_249856


namespace sum_first_4_terms_l249_249693

theorem sum_first_4_terms 
  (a_1 : ℚ) 
  (q : ℚ) 
  (h1 : a_1 * q - a_1 * q^2 = -2) 
  (h2 : a_1 + a_1 * q^2 = 10 / 3) 
  : a_1 * (1 + q + q^2 + q^3) = 40 / 3 := sorry

end sum_first_4_terms_l249_249693


namespace find_second_offset_l249_249638

variable (d : ℕ) (o₁ : ℕ) (A : ℕ)

theorem find_second_offset (hd : d = 20) (ho₁ : o₁ = 5) (hA : A = 90) : ∃ (o₂ : ℕ), o₂ = 4 :=
by
  sorry

end find_second_offset_l249_249638


namespace three_more_than_seven_in_pages_l249_249242

theorem three_more_than_seven_in_pages : 
  ∀ (pages : List Nat), (∀ n, n ∈ pages → 1 ≤ n ∧ n ≤ 530) ∧ (List.length pages = 530) →
  ((List.count 3 (pages.bind (λ n => Nat.digits 10 n))) - (List.count 7 (pages.bind (λ n => Nat.digits 10 n)))) = 100 :=
by
  intros pages h
  sorry

end three_more_than_seven_in_pages_l249_249242


namespace minimum_numbers_to_form_triangle_l249_249460

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem minimum_numbers_to_form_triangle :
  ∀ (S : Finset ℕ), (S ⊆ Finset.range 1001) →
    16 ≤ S.card →
    ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ {a, b, c} ⊆ S ∧ is_triangle a b c :=
by
  sorry

end minimum_numbers_to_form_triangle_l249_249460


namespace find_ending_number_l249_249350

theorem find_ending_number (n : ℕ) 
  (h1 : n ≥ 7) 
  (h2 : ∀ m, 7 ≤ m ∧ m ≤ n → m % 7 = 0)
  (h3 : (7 + n) / 2 = 15) : n = 21 := 
sorry

end find_ending_number_l249_249350


namespace initial_birds_on_fence_l249_249108

theorem initial_birds_on_fence (B S : ℕ) (S_val : S = 2) (total : B + 5 + S = 10) : B = 3 :=
by
  sorry

end initial_birds_on_fence_l249_249108


namespace paint_time_l249_249705

theorem paint_time (n₁ n₂ h: ℕ) (t₁ t₂: ℕ) (constant: ℕ):
  n₁ = 6 → t₁ = 8 → h = 2 → constant = 96 →
  constant = n₁ * t₁ * h → n₂ = 4 → constant = n₂ * t₂ * h →
  t₂ = 12 :=
by
  intros
  sorry

end paint_time_l249_249705


namespace boys_made_mistake_l249_249106

theorem boys_made_mistake (n m : ℕ) (hn : n > 1) (hm : m > 1) (h_eq : Nat.factorial n = 2^m * Nat.factorial m) : False :=
by
  sorry

end boys_made_mistake_l249_249106


namespace avg_of_arithmetic_series_is_25_l249_249185

noncomputable def arithmetic_series_avg : ℝ :=
  let a₁ := 15
  let d := 1 / 4
  let aₙ := 35
  let n := (aₙ - a₁) / d + 1
  let S := n * (a₁ + aₙ) / 2
  S / n

theorem avg_of_arithmetic_series_is_25 : arithmetic_series_avg = 25 := 
by
  -- Sorry, proof omitted due to instruction.
  sorry

end avg_of_arithmetic_series_is_25_l249_249185


namespace sin_300_eq_neg_sqrt3_div_2_l249_249526

def point_on_unit_circle (θ : ℝ) : ℝ × ℝ :=
⟨Real.cos θ, Real.sin θ⟩

theorem sin_300_eq_neg_sqrt3_div_2 :
  let Q := point_on_unit_circle (300 * Real.pi / 180) in
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
by sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249526


namespace find_remainder_l249_249304

-- Definitions
variable (x y : ℕ)
variable (h1 : x > 0)
variable (h2 : y > 0)
variable (h3 : (x : ℝ) / y = 96.15)
variable (h4 : approximately_equal (y : ℝ) 60)

-- Target statement
theorem find_remainder : x % y = 9 :=
sorry

end find_remainder_l249_249304


namespace remainder_of_polynomial_l249_249643

theorem remainder_of_polynomial :
  ∀ (x : ℂ), (x^4 + x^3 + x^2 + x + 1 = 0) → (x^55 + x^44 + x^33 + x^22 + x^11 + 1) % (x^4 + x^3 + x^2 + x + 1) = 2 :=
by
  intro x hx
  sorry

end remainder_of_polynomial_l249_249643


namespace guides_and_tourists_groupings_l249_249293

open Nat

/-- Three tour guides are leading eight tourists. Each tourist must choose one of the guides, 
    but with the stipulation that each guide must take at least one tourist. Prove 
    that the number of different groupings of guides and tourists is 5796. -/
theorem guides_and_tourists_groupings : 
  let total_groupings := 3 ^ 8,
      at_least_one_no_tourists := binom 3 1 * 2 ^ 8,
      at_least_two_no_tourists := binom 3 2 * 1 ^ 8,
      total_valid_groupings := total_groupings - at_least_one_no_tourists + at_least_two_no_tourists
  in total_valid_groupings = 5796 :=
by
  sorry

end guides_and_tourists_groupings_l249_249293


namespace amoeba_growth_one_week_l249_249434

theorem amoeba_growth_one_week :
  (3 ^ 7 = 2187) :=
by
  sorry

end amoeba_growth_one_week_l249_249434


namespace sequence_a19_l249_249657

theorem sequence_a19 :
  ∃ (a : ℕ → ℝ), a 3 = 2 ∧ a 7 = 1 ∧
    (∃ d : ℝ, ∀ n m : ℕ, (1 / (a n + 1) - 1 / (a m + 1)) / (n - m) = d) →
    a 19 = 0 :=
by sorry

end sequence_a19_l249_249657


namespace factor_expression_l249_249899

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249899


namespace find_a2023_l249_249042

variable {a : ℕ → ℕ}
variable {x : ℕ}

def sequence_property (a: ℕ → ℕ) : Prop :=
  ∀ n, a n + a (n + 1) + a (n + 2) = 20

theorem find_a2023 (h1 : sequence_property a) 
                   (h2 : a 2 = 2 * x) 
                   (h3 : a 18 = 9 + x) 
                   (h4 : a 65 = 6 - x) : 
  a 2023 = 5 := 
by
  sorry

end find_a2023_l249_249042


namespace wooden_parallelepiped_length_l249_249002

theorem wooden_parallelepiped_length (n : ℕ) (h1 : n ≥ 7)
    (h2 : ∀ total_cubes unpainted_cubes : ℕ,
      total_cubes = n * (n - 2) * (n - 4) ∧
      unpainted_cubes = (n - 2) * (n - 4) * (n - 6) ∧
      unpainted_cubes = 2 / 3 * total_cubes) :
  n = 18 := 
sorry

end wooden_parallelepiped_length_l249_249002


namespace find_higher_selling_price_l249_249014

-- Define the constants and initial conditions
def cost_price : ℕ := 200
def selling_price_1 : ℕ := 340
def gain_1 : ℕ := selling_price_1 - cost_price
def new_gain : ℕ := gain_1 + gain_1 * 5 / 100

-- Define the problem statement
theorem find_higher_selling_price : 
  ∀ P : ℕ, P = cost_price + new_gain → P = 347 :=
by
  intro P
  intro h
  sorry

end find_higher_selling_price_l249_249014


namespace lions_after_one_year_l249_249292

def initial_lions : ℕ := 100
def birth_rate : ℕ := 5
def death_rate : ℕ := 1
def months_in_year : ℕ := 12

theorem lions_after_one_year : 
  initial_lions + (birth_rate * months_in_year) - (death_rate * months_in_year) = 148 :=
by
  sorry

end lions_after_one_year_l249_249292


namespace probability_penny_nickel_dime_all_heads_l249_249117

-- Define flipping five coins
def flip_five_coins : Type := (bool × bool × bool × bool × bool)

-- Define the function to check if the penny, nickel, and dime are heads
def all_heads_penny_nickel_dime (o : flip_five_coins) : bool :=
  o.1 = tt ∧ o.2 = tt ∧ o.3 = tt

-- Define the total count of possible outcomes
def total_outcomes : ℕ := 32

-- Define the count of favorable outcomes
def favorable_outcomes : ℕ := 4

-- Define the probability calculation
def probability_favorable : ℚ := favorable_outcomes / total_outcomes

-- The statement proving the probability is 1/8
theorem probability_penny_nickel_dime_all_heads :
  probability_favorable = 1 / 8 :=
by
  sorry

end probability_penny_nickel_dime_all_heads_l249_249117


namespace remainder_of_3_pow_2023_mod_7_l249_249145

theorem remainder_of_3_pow_2023_mod_7 :
  (3^2023) % 7 = 3 := 
by
  sorry

end remainder_of_3_pow_2023_mod_7_l249_249145


namespace sum_of_differences_l249_249997

theorem sum_of_differences (x : ℝ) (h : (45 + x) / 2 = 38) : abs (x - 45) + abs (x - 30) = 15 := by
  sorry

end sum_of_differences_l249_249997


namespace find_grape_juice_l249_249801

variables (milk water: ℝ) (limit total_before_test grapejuice: ℝ)

-- Conditions
def milk_amt: ℝ := 8
def water_amt: ℝ := 8
def limit_amt: ℝ := 32

-- The total liquid consumed before the test can be computed
def total_before_test_amt (milk water: ℝ) : ℝ := limit_amt - water_amt

-- The given total liquid consumed must be (milk + grape juice)
def total_consumed (milk grapejuice: ℝ) : ℝ := milk + grapejuice

theorem find_grape_juice :
    total_before_test_amt milk_amt water_amt = total_consumed milk_amt grapejuice →
    grapejuice = 16 :=
by
    unfold total_before_test_amt total_consumed
    sorry

end find_grape_juice_l249_249801


namespace degrees_to_radians_18_l249_249467

theorem degrees_to_radians_18 (degrees : ℝ) (h : degrees = 18) : 
  (degrees * (Real.pi / 180) = Real.pi / 10) :=
by
  sorry

end degrees_to_radians_18_l249_249467


namespace correct_parentheses_l249_249953

theorem correct_parentheses : (1 * 2 * 3 + 4) * 5 = 50 := by
  sorry

end correct_parentheses_l249_249953


namespace algebraic_expression_defined_iff_l249_249239

theorem algebraic_expression_defined_iff (x : ℝ) : (∃ y, y = 3 / (x - 2)) ↔ x ≠ 2 := by
  sorry

end algebraic_expression_defined_iff_l249_249239


namespace sin_300_eq_neg_sqrt3_div_2_l249_249527

def point_on_unit_circle (θ : ℝ) : ℝ × ℝ :=
⟨Real.cos θ, Real.sin θ⟩

theorem sin_300_eq_neg_sqrt3_div_2 :
  let Q := point_on_unit_circle (300 * Real.pi / 180) in
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
by sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249527


namespace factor_expression_l249_249906

theorem factor_expression (x : ℝ) : (x * (x + 3) + 2 * (x + 3)) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249906


namespace dice_game_probability_l249_249689

def is_valid_roll (d1 d2 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6

def score (d1 d2 : ℕ) : ℕ :=
  max d1 d2

def favorable_outcomes : List (ℕ × ℕ) :=
  [ (1, 1), (1, 2), (2, 1), (2, 2), 
    (1, 3), (2, 3), (3, 1), (3, 2), (3, 3) ]

def total_outcomes : ℕ := 36

def favorable_count : ℕ := favorable_outcomes.length

theorem dice_game_probability : 
  (favorable_count : ℚ) / (total_outcomes : ℚ) = 1 / 4 :=
by
  sorry

end dice_game_probability_l249_249689


namespace P_necessary_but_not_sufficient_for_q_l249_249194

def M : Set ℝ := {x : ℝ | (x - 1) * (x - 2) > 0}
def N : Set ℝ := {x : ℝ | x^2 + x < 0}

theorem P_necessary_but_not_sufficient_for_q :
  (∀ x, x ∈ N → x ∈ M) ∧ (∃ x, x ∈ M ∧ x ∉ N) :=
by
  sorry

end P_necessary_but_not_sufficient_for_q_l249_249194


namespace evans_family_children_count_l249_249996

-- Let the family consist of the mother, the father, two grandparents, and children.
-- This proof aims to show x, the number of children, is 1.

theorem evans_family_children_count
  (m g y : ℕ) -- m = mother's age, g = average age of two grandparents, y = average age of children
  (x : ℕ) -- x = number of children
  (avg_family_age : (m + 50 + 2 * g + x * y) / (4 + x) = 30)
  (father_age : 50 = 50)
  (avg_non_father_age : (m + 2 * g + x * y) / (3 + x) = 25) :
  x = 1 :=
sorry

end evans_family_children_count_l249_249996


namespace shortest_altitude_l249_249285

theorem shortest_altitude (a b c : ℕ) (h1 : a = 12) (h2 : b = 16) (h3 : c = 20) (h4 : a^2 + b^2 = c^2) : ∃ x, x = 9.6 :=
by
  sorry

end shortest_altitude_l249_249285


namespace students_exceed_pets_by_70_l249_249179

theorem students_exceed_pets_by_70 :
  let n_classrooms := 5
  let students_per_classroom := 22
  let rabbits_per_classroom := 3
  let hamsters_per_classroom := 5
  let total_students := students_per_classroom * n_classrooms
  let total_rabbits := rabbits_per_classroom * n_classrooms
  let total_hamsters := hamsters_per_classroom * n_classrooms
  let total_pets := total_rabbits + total_hamsters
  total_students - total_pets = 70 :=
  by
    sorry

end students_exceed_pets_by_70_l249_249179


namespace expansion_term_count_l249_249062

theorem expansion_term_count 
  (A : Finset ℕ) (B : Finset ℕ) 
  (hA : A.card = 3) (hB : B.card = 4) : 
  (Finset.card (A.product B)) = 12 :=
by {
  sorry
}

end expansion_term_count_l249_249062


namespace ball_travel_approximately_80_l249_249860

noncomputable def ball_travel_distance : ℝ :=
  let h₀ := 20
  let ratio := 2 / 3
  h₀ + -- first descent
  h₀ * ratio + -- first ascent
  h₀ * ratio + -- second descent
  h₀ * ratio^2 + -- second ascent
  h₀ * ratio^2 + -- third descent
  h₀ * ratio^3 + -- third ascent
  h₀ * ratio^3 + -- fourth descent
  h₀ * ratio^4 -- fourth ascent

theorem ball_travel_approximately_80 :
  abs (ball_travel_distance - 80) < 1 :=
sorry

end ball_travel_approximately_80_l249_249860


namespace coin_stack_height_l249_249966

def alpha_thickness : ℝ := 1.25
def beta_thickness : ℝ := 2.00
def gamma_thickness : ℝ := 0.90
def delta_thickness : ℝ := 1.60
def stack_height : ℝ := 18.00

theorem coin_stack_height :
  (∃ n : ℕ, stack_height = n * beta_thickness) ∨ (∃ n : ℕ, stack_height = n * gamma_thickness) :=
sorry

end coin_stack_height_l249_249966


namespace no_pairs_of_a_and_d_l249_249979

theorem no_pairs_of_a_and_d :
  ∀ (a d : ℝ), (∀ (x y: ℝ), 4 * x + a * y + d = 0 ↔ d * x - 3 * y + 15 = 0) -> False :=
by 
  sorry

end no_pairs_of_a_and_d_l249_249979


namespace factor_expression_l249_249910

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249910


namespace smallest_a_undefined_inverse_l249_249301

theorem smallest_a_undefined_inverse (a : ℕ) (ha : a = 2) :
  (∀ (a : ℕ), 0 < a → ((Nat.gcd a 40 > 1) ∧ (Nat.gcd a 90 > 1)) ↔ a = 2) :=
by
  sorry

end smallest_a_undefined_inverse_l249_249301


namespace sin_300_eq_neg_sqrt3_div_2_l249_249525

def point_on_unit_circle (θ : ℝ) : ℝ × ℝ :=
⟨Real.cos θ, Real.sin θ⟩

theorem sin_300_eq_neg_sqrt3_div_2 :
  let Q := point_on_unit_circle (300 * Real.pi / 180) in
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
by sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249525


namespace range_of_a_l249_249784

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + 1/2 * x^2

theorem range_of_a (a : ℝ)
  (H : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f a (x1 + a) - f a (x2 + a)) / (x1 - x2) ≥ 3) :
  a ≥ 9 / 4 :=
sorry

end range_of_a_l249_249784


namespace determine_n_for_square_l249_249881

theorem determine_n_for_square (n : ℕ) : (∃ a : ℕ, 5^n + 4 = a^2) ↔ n = 1 :=
by
-- The proof will be included here, but for now, we just provide the structure
sorry

end determine_n_for_square_l249_249881


namespace inequality_solution_l249_249880

theorem inequality_solution (x : ℝ) : x^2 + x - 20 < 0 ↔ -5 < x ∧ x < 4 := 
by
  sorry

end inequality_solution_l249_249880


namespace ranking_emily_olivia_nicole_l249_249418

noncomputable def Emily_score : ℝ := sorry
noncomputable def Olivia_score : ℝ := sorry
noncomputable def Nicole_score : ℝ := sorry

theorem ranking_emily_olivia_nicole :
  (Emily_score > Olivia_score) ∧ (Emily_score > Nicole_score) → 
  (Emily_score > Olivia_score) ∧ (Olivia_score > Nicole_score) := 
by sorry

end ranking_emily_olivia_nicole_l249_249418


namespace exists_group_of_three_friends_l249_249965

-- Defining the context of the problem
def people := Fin 10 -- a finite set of 10 people
def quarrel (x y : people) : Prop := -- a predicate indicating a quarrel between two people
sorry

-- Given conditions
axiom quarreled_pairs : ∃ S : Finset (people × people), S.card = 14 ∧ 
  ∀ {x y : people}, (x, y) ∈ S → x ≠ y ∧ quarrel x y

-- Question: Prove there exists a set of 3 friends among these 10 people
theorem exists_group_of_three_friends (p : Finset people):
  ∃ (group : Finset people), group.card = 3 ∧ ∀ {x y : people}, 
  x ∈ group → y ∈ group → x ≠ y → ¬ quarrel x y :=
sorry

end exists_group_of_three_friends_l249_249965


namespace concentric_circle_ratio_l249_249297

theorem concentric_circle_ratio (r R : ℝ) (hRr : R > r)
  (new_circles_tangent : ∀ (C1 C2 C3 : ℝ), C1 = C2 ∧ C2 = C3 ∧ C1 < R ∧ r < C1): 
  R = 3 * r := by sorry

end concentric_circle_ratio_l249_249297


namespace negation_of_prop_l249_249268

theorem negation_of_prop :
  (¬ ∀ (x y : ℝ), x^2 + y^2 ≥ 0) ↔ (∃ (x y : ℝ), x^2 + y^2 < 0) :=
by
  sorry

end negation_of_prop_l249_249268


namespace find_a4_l249_249669

variables {a : ℕ → ℝ} (q : ℝ) (h_positive : ∀ n, 0 < a n)
variables (h_seq : ∀ n, a (n+1) = q * a n)
variables (h1 : a 1 + (2/3) * a 2 = 3)
variables (h2 : (a 4)^2 = (1/9) * a 3 * a 7)

-- Proof problem statement
theorem find_a4 : a 4 = 27 :=
sorry

end find_a4_l249_249669


namespace min_value_of_a_l249_249093

noncomputable def P (x : ℕ) : ℤ := sorry

def smallest_value_of_a (a : ℕ) : Prop :=
  a > 0 ∧
  (P 1 = a ∧ P 3 = a ∧ P 5 = a ∧ P 7 = a ∧ P 9 = a ∧
   P 2 = -a ∧ P 4 = -a ∧ P 6 = -a ∧ P 8 = -a ∧ P 10 = -a)

theorem min_value_of_a : ∃ a : ℕ, smallest_value_of_a a ∧ a = 6930 :=
sorry

end min_value_of_a_l249_249093


namespace largest_shaded_area_l249_249694

noncomputable def figureA_shaded_area : ℝ := 16 - 4 * Real.pi
noncomputable def figureB_shaded_area : ℝ := 16 - 4 * Real.pi
noncomputable def figureC_shaded_area : ℝ := 16 - 4 * Real.sqrt 3

theorem largest_shaded_area : 
  figureC_shaded_area > figureA_shaded_area ∧ figureC_shaded_area > figureB_shaded_area :=
by
  sorry

end largest_shaded_area_l249_249694


namespace exists_fg_pairs_l249_249647

theorem exists_fg_pairs (a b : ℤ) :
  (∃ (f g : ℤ → ℤ), (∀ x : ℤ, f (g x) = x + a) ∧ (∀ x : ℤ, g (f x) = x + b)) ↔ (a = b ∨ a = -b) := 
sorry

end exists_fg_pairs_l249_249647


namespace sin_300_eq_neg_sqrt_3_div_2_l249_249521

noncomputable def sin_300_degrees : ℝ := Real.sin (300 * Real.pi / 180)

theorem sin_300_eq_neg_sqrt_3_div_2 : sin_300_degrees = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_300_eq_neg_sqrt_3_div_2_l249_249521


namespace ratio_of_largest_to_smallest_root_in_geometric_progression_l249_249989

theorem ratio_of_largest_to_smallest_root_in_geometric_progression 
    (a b c d : ℝ) (r s t : ℝ) 
    (h_poly : 81 * r^3 - 243 * r^2 + 216 * r - 64 = 0)
    (h_geo_prog : r > 0 ∧ s > 0 ∧ t > 0 ∧ ∃ (k : ℝ),  k > 0 ∧ s = r * k ∧ t = s * k) :
    ∃ (k : ℝ), k = r^2 ∧ s = r * k ∧ t = s * k := 
sorry

end ratio_of_largest_to_smallest_root_in_geometric_progression_l249_249989


namespace find_z_plus_inverse_y_l249_249435

theorem find_z_plus_inverse_y
  (x y z : ℝ)
  (h1 : x * y * z = 1)
  (h2 : x + 1/z = 10)
  (h3 : y + 1/x = 5) :
  z + 1/y = 17 / 49 :=
by
  sorry

end find_z_plus_inverse_y_l249_249435


namespace HCl_yield_l249_249192

noncomputable def total_moles_HCl (moles_C2H6 moles_Cl2 yield1 yield2 : ℝ) : ℝ :=
  let theoretical_yield1 := if moles_C2H6 ≤ moles_Cl2 then moles_C2H6 else moles_Cl2
  let actual_yield1 := theoretical_yield1 * yield1
  let theoretical_yield2 := actual_yield1
  let actual_yield2 := theoretical_yield2 * yield2
  actual_yield1 + actual_yield2

theorem HCl_yield (moles_C2H6 moles_Cl2 : ℝ) (yield1 yield2 : ℝ) :
  moles_C2H6 = 3 → moles_Cl2 = 3 → yield1 = 0.85 → yield2 = 0.70 →
  total_moles_HCl moles_C2H6 moles_Cl2 yield1 yield2 = 4.335 :=
by
  intros h1 h2 h3 h4
  simp [total_moles_HCl, h1, h2, h3, h4]
  sorry

end HCl_yield_l249_249192


namespace sin_300_eq_neg_sin_60_l249_249606

theorem sin_300_eq_neg_sin_60 :
  Real.sin (300 * Real.pi / 180) = - Real.sin (60 * Real.pi / 180) :=
by
suffices : Real.sin (300 * Real.pi / 180) = -sqrt 3 / 2
  sorry

end sin_300_eq_neg_sin_60_l249_249606


namespace broccoli_area_l249_249747

/--
A farmer grows broccoli in a square-shaped farm. This year, he produced 2601 broccoli,
which is 101 more than last year. The shape of the area used for growing the broccoli 
has remained square in both years. Assuming each broccoli takes up an equal amount of 
area, prove that each broccoli takes up 1 square unit of area.
-/
theorem broccoli_area (x y : ℕ) 
  (h1 : y^2 = x^2 + 101) 
  (h2 : y^2 = 2601) : 
  1 = 1 := 
sorry

end broccoli_area_l249_249747


namespace total_students_l249_249832

def numStudents (skiing scavenger : ℕ) : ℕ :=
  skiing + scavenger

theorem total_students (skiing scavenger : ℕ) (h1 : skiing = 2 * scavenger) (h2 : scavenger = 4000) :
  numStudents skiing scavenger = 12000 :=
by
  sorry

end total_students_l249_249832


namespace inverse_proportion_inequality_l249_249797

variable (x1 x2 k : ℝ)

theorem inverse_proportion_inequality (hA : 2 = k / x1) (hB : 4 = k / x2) (hk : 0 < k) : 
  x1 > x2 ∧ x1 > 0 ∧ x2 > 0 :=
sorry

end inverse_proportion_inequality_l249_249797


namespace sin_300_eq_neg_sqrt3_div_2_l249_249594

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249594


namespace parrots_fraction_l249_249072

variable (P T : ℚ) -- P: fraction of parrots, T: fraction of toucans

def fraction_parrots (P T : ℚ) : Prop :=
  P + T = 1 ∧
  (2 / 3) * P + (1 / 4) * T = 0.5

theorem parrots_fraction (P T : ℚ) (h : fraction_parrots P T) : P = 3 / 5 :=
by
  sorry

end parrots_fraction_l249_249072


namespace remainder_of_x_div_9_is_8_l249_249303

variable (x y r : ℕ)
variable (r_lt_9 : r < 9)
variable (h1 : x = 9 * y + r)
variable (h2 : 2 * x = 14 * y + 1)
variable (h3 : 5 * y - x = 3)

theorem remainder_of_x_div_9_is_8 : r = 8 := by
  sorry

end remainder_of_x_div_9_is_8_l249_249303


namespace initially_calculated_average_height_l249_249823

theorem initially_calculated_average_height 
    (students : ℕ) (incorrect_height : ℕ) (correct_height : ℕ) (actual_avg_height : ℝ) 
    (A : ℝ) 
    (h_students : students = 30) 
    (h_incorrect_height : incorrect_height = 151) 
    (h_correct_height : correct_height = 136) 
    (h_actual_avg_height : actual_avg_height = 174.5)
    (h_A_definition : (students : ℝ) * A + (incorrect_height - correct_height) = (students : ℝ) * actual_avg_height) : 
    A = 174 := 
by sorry

end initially_calculated_average_height_l249_249823


namespace sin_300_eq_neg_sqrt3_div_2_l249_249530

def point_on_unit_circle (θ : ℝ) : ℝ × ℝ :=
⟨Real.cos θ, Real.sin θ⟩

theorem sin_300_eq_neg_sqrt3_div_2 :
  let Q := point_on_unit_circle (300 * Real.pi / 180) in
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
by sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249530


namespace total_bird_families_l249_249453

-- Declare the number of bird families that flew to Africa
def a : Nat := 47

-- Declare the number of bird families that flew to Asia
def b : Nat := 94

-- Condition that Asia's number of bird families matches Africa + 47 more
axiom h : b = a + 47

-- Prove the total number of bird families is 141
theorem total_bird_families : a + b = 141 :=
by
  -- Insert proof here
  sorry

end total_bird_families_l249_249453


namespace sin_300_eq_neg_sqrt3_div_2_l249_249615

-- Let Q be the point on the unit circle at 300 degrees counterclockwise from (1,0)
def Q : ℝ × ℝ := (1 / 2, -Real.sqrt 3 / 2)

-- Prove that the sine of 300 degrees is -√3/2
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  -- Here we assume the coordinates given in the conditions are correct.
  have hQ : Q = (1 / 2, -Real.sqrt 3 / 2) := rfl
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249615


namespace sin_300_eq_neg_sin_60_l249_249608

theorem sin_300_eq_neg_sin_60 :
  Real.sin (300 * Real.pi / 180) = - Real.sin (60 * Real.pi / 180) :=
by
suffices : Real.sin (300 * Real.pi / 180) = -sqrt 3 / 2
  sorry

end sin_300_eq_neg_sin_60_l249_249608


namespace john_total_climb_height_l249_249088

-- Define the heights and conditions
def num_flights : ℕ := 3
def height_per_flight : ℕ := 10
def total_stairs_height : ℕ := num_flights * height_per_flight
def rope_height : ℕ := total_stairs_height / 2
def ladder_height : ℕ := rope_height + 10

-- Prove that the total height John climbed is 70 feet
theorem john_total_climb_height : 
  total_stairs_height + rope_height + ladder_height = 70 := by
  sorry

end john_total_climb_height_l249_249088


namespace exercise_l249_249813

theorem exercise (x y z : ℕ) (h1 : x * y * z = 1) : (7 ^ ((x + y + z) ^ 3) / 7 ^ ((x - y + z) ^ 3)) = 7 ^ 6 := 
by
  sorry

end exercise_l249_249813


namespace sqrt_of_four_is_pm_two_l249_249287

theorem sqrt_of_four_is_pm_two (y : ℤ) : y * y = 4 → y = 2 ∨ y = -2 := by
  sorry

end sqrt_of_four_is_pm_two_l249_249287


namespace households_with_both_car_and_bike_l249_249690

theorem households_with_both_car_and_bike 
  (total_households : ℕ) 
  (households_without_either : ℕ) 
  (households_with_car : ℕ) 
  (households_with_bike_only : ℕ)
  (H1 : total_households = 90)
  (H2 : households_without_either = 11)
  (H3 : households_with_car = 44)
  (H4 : households_with_bike_only = 35)
  : ∃ B : ℕ, households_with_car - households_with_bike_only = B ∧ B = 9 := 
by
  sorry

end households_with_both_car_and_bike_l249_249690


namespace right_angled_triangle_side_length_l249_249962

theorem right_angled_triangle_side_length :
  ∃ c : ℕ, (c = 5) ∧ (3^2 + 4^2 = c^2) ∧ (c = 4 + 1) := by
  sorry

end right_angled_triangle_side_length_l249_249962


namespace find_d_l249_249403

theorem find_d (a b c d : ℤ) (h_poly : ∃ s1 s2 s3 s4 : ℤ, s1 > 0 ∧ s2 > 0 ∧ s3 > 0 ∧ s4 > 0 ∧ 
  ( ∀ x, (Polynomial.eval x (Polynomial.C d + Polynomial.X * Polynomial.C c + Polynomial.X^2 * Polynomial.C b + Polynomial.X^3 * Polynomial.C a + Polynomial.X^4)) =
    (x + s1) * (x + s2) * (x + s3) * (x + s4) ) ) 
  (h_sum : a + b + c + d = 2013) : d = 0 :=
by
  sorry

end find_d_l249_249403


namespace intersection_of_sets_l249_249956

def set_A (x : ℝ) := x + 1 ≤ 3
def set_B (x : ℝ) := 4 - x^2 ≤ 0

theorem intersection_of_sets : {x : ℝ | set_A x} ∩ {x : ℝ | set_B x} = {x : ℝ | x ≤ -2} ∪ {2} :=
by
  sorry

end intersection_of_sets_l249_249956


namespace inverse_proportion_relationship_l249_249365

variable {x1 x2 y1 y2 : ℝ}

theorem inverse_proportion_relationship (h1 : x1 < 0) (h2 : 0 < x2) 
  (hy1 : y1 = 3 / x1) (hy2 : y2 = 3 / x2) : y1 < 0 ∧ 0 < y2 :=
by
  sorry

end inverse_proportion_relationship_l249_249365


namespace base3_to_base10_l249_249626

theorem base3_to_base10 : 
  let n := 20202
  let base := 3
  base_expansion n base = 182 := by
    sorry

end base3_to_base10_l249_249626


namespace martha_savings_l249_249413

-- Definitions based on conditions
def weekly_latte_spending : ℝ := 4.00 * 5
def weekly_iced_coffee_spending : ℝ := 2.00 * 3
def total_weekly_coffee_spending : ℝ := weekly_latte_spending + weekly_iced_coffee_spending
def annual_coffee_spending : ℝ := total_weekly_coffee_spending * 52
def savings_percentage : ℝ := 0.25

-- The theorem to be proven
theorem martha_savings : annual_coffee_spending * savings_percentage = 338.00 := by
  sorry

end martha_savings_l249_249413


namespace value_of_six_prime_prime_l249_249370

-- Define the function q' 
def prime (q : ℝ) : ℝ := 3 * q - 3

-- Stating the main theorem we want to prove
theorem value_of_six_prime_prime : prime (prime 6) = 42 :=
by
  sorry

end value_of_six_prime_prime_l249_249370


namespace time_difference_180_div_vc_l249_249387

open Real

theorem time_difference_180_div_vc
  (V_A V_B V_C : ℝ)
  (h_ratio : V_A / V_C = 5 ∧ V_B / V_C = 4)
  (start_A start_B start_C : ℝ)
  (h_start_A : start_A = 100)
  (h_start_B : start_B = 80)
  (h_start_C : start_C = 0)
  (race_distance : ℝ)
  (h_race_distance : race_distance = 1200) :
  (race_distance - start_A) / V_A - race_distance / V_C = 180 / V_C := 
sorry

end time_difference_180_div_vc_l249_249387


namespace muffins_divide_equally_l249_249976

theorem muffins_divide_equally (friends : ℕ) (total_muffins : ℕ) (Jessie_and_friends : ℕ) (muffins_per_person : ℕ) :
  friends = 6 →
  total_muffins = 35 →
  Jessie_and_friends = friends + 1 →
  muffins_per_person = total_muffins / Jessie_and_friends →
  muffins_per_person = 5 :=
by
  intros h_friends h_muffins h_people h_division
  sorry

end muffins_divide_equally_l249_249976


namespace slices_per_banana_l249_249891

-- Define conditions
def yogurts : ℕ := 5
def slices_per_yogurt : ℕ := 8
def bananas : ℕ := 4
def total_slices_needed : ℕ := yogurts * slices_per_yogurt

-- Statement to prove
theorem slices_per_banana : total_slices_needed / bananas = 10 := by sorry

end slices_per_banana_l249_249891


namespace some_value_correct_l249_249369

theorem some_value_correct (w x y : ℝ) (some_value : ℝ)
  (h1 : 3 / w + some_value = 3 / y)
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 0.5) :
  some_value = 6 := by
  sorry

end some_value_correct_l249_249369


namespace proposition_incorrect_l249_249306

theorem proposition_incorrect :
  ¬(∀ x : ℝ, x^2 + 3 * x + 1 > 0) :=
by
  sorry

end proposition_incorrect_l249_249306


namespace cost_of_purchase_l249_249431

theorem cost_of_purchase :
  (5 * 3) + (8 * 2) = 31 :=
by
  sorry

end cost_of_purchase_l249_249431


namespace number_of_DVDs_sold_l249_249385

theorem number_of_DVDs_sold (C D: ℤ) (h₁ : D = 16 * C / 10) (h₂ : D + C = 273) : D = 168 := 
sorry

end number_of_DVDs_sold_l249_249385


namespace geo_seq_fifth_term_l249_249289

theorem geo_seq_fifth_term (a r : ℝ) (a_pos : 0 < a) (r_pos : 0 < r)
  (h3 : a * r^2 = 8) (h7 : a * r^6 = 18) : a * r^4 = 12 :=
sorry

end geo_seq_fifth_term_l249_249289


namespace triangle_shape_l249_249799

theorem triangle_shape (a b : ℝ) (A B : ℝ)
  (h1 : a ≠ 0) (h2 : b ≠ 0) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π)
  (hTriangle : A + B + (π - A - B) = π)
  (h : a * Real.cos A = b * Real.cos B) : 
  (A = B ∨ A + B = π / 2) := sorry

end triangle_shape_l249_249799


namespace sum_non_solutions_eq_neg21_l249_249257

theorem sum_non_solutions_eq_neg21
  (A B C : ℝ)
  (h1 : ∀ x, ∃ k : ℝ, (x ≠ -C) ∧ (x ≠ -9) → (x + B) * (A * x + 36) = 3 * (x + C) * (x + 9))
  (h2 : ∃ A B C, ∀ x, (x ≠ -C) ∧ (x ≠ -9) → (x + B) * (A * x + 36) = 3 * (x + C) * (x + 9))
  (h3 : ∃! x, (x + C) * (x + 9) = 0)
   :
  -9 + -12 = -21 := by sorry

end sum_non_solutions_eq_neg21_l249_249257


namespace negation_of_exists_l249_249441

theorem negation_of_exists (h : ∃ x : ℝ, x > 0 ∧ x^2 + 3*x + 1 < 0) : ∀ x : ℝ, x > 0 → x^2 + 3*x + 1 ≥ 0 :=
sorry

end negation_of_exists_l249_249441


namespace parallelepiped_length_l249_249009

theorem parallelepiped_length :
  ∃ n : ℕ, (n ≥ 7) ∧ (n * (n - 2) * (n - 4) = 3 * ((n - 2) * (n - 4) * (n - 6))) ∧ n = 18 :=
by
  sorry

end parallelepiped_length_l249_249009


namespace sqrt_meaningful_range_l249_249071

theorem sqrt_meaningful_range (x : ℝ) : x + 1 ≥ 0 ↔ x ≥ -1 :=
by sorry

end sqrt_meaningful_range_l249_249071


namespace factor_expression_l249_249900

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249900


namespace circumference_of_wheels_l249_249708

-- Define the variables and conditions
variables (x y : ℝ)

def condition1 (x y : ℝ) : Prop := (120 / x) - (120 / y) = 6
def condition2 (x y : ℝ) : Prop := (4 / 5) * (120 / x) - (5 / 6) * (120 / y) = 4

-- The main theorem to prove
theorem circumference_of_wheels (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) : x = 4 ∧ y = 5 :=
  sorry  -- Proof is omitted

end circumference_of_wheels_l249_249708


namespace sin_300_eq_neg_sqrt3_div_2_l249_249499

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249499


namespace unique_solution_l249_249677

theorem unique_solution (x : ℝ) : 
  ∃! x, 2003^x + 2004^x = 2005^x := 
sorry

end unique_solution_l249_249677


namespace parallelepiped_length_l249_249008

theorem parallelepiped_length (n : ℕ)
  (h1 : ∃ n : ℕ, n = 18) 
  (h2 : one_third_of_cubes_have_red_faces : (∃ k : ℕ, k = ((n * (n - 2) * (n - 4)) / 3)) 
        ∧ (remaining_unpainted_cubes : (∃ m : ℕ , m = (2 * (n * (n - 2) * (n - 4)) / 3))))
  (h3 : painted_and_cut_into_cubes : (∃ a b c : ℕ, a = n ∧ b = (n - 2) ∧ c = (n - 4)))
  (h4 : all_sides_whole_cm : (∃ d : ℕ , d = n ∧ d = (n - 2) ∧ d = (n - 4))) :
  n = 18 :=
begin
  sorry
end

end parallelepiped_length_l249_249008


namespace original_price_per_lesson_l249_249251

theorem original_price_per_lesson (piano_cost lessons_cost : ℤ) (number_of_lessons discount_percent : ℚ) (total_cost : ℤ) (original_price : ℚ) :
  piano_cost = 500 ∧
  number_of_lessons = 20 ∧
  discount_percent = 0.25 ∧
  total_cost = 1100 →
  lessons_cost = total_cost - piano_cost →
  0.75 * (number_of_lessons * original_price) = lessons_cost →
  original_price = 40 :=
by
  intros h h1 h2
  sorry

end original_price_per_lesson_l249_249251


namespace min_trips_correct_l249_249315

-- Define the masses of the individuals and the elevator capacity as constants
def masses : List ℕ := [150, 62, 63, 66, 70, 75, 79, 84, 95, 96, 99]
def elevator_capacity : ℕ := 190

-- Define a function that computes the minimum number of trips required to transport all individuals
noncomputable def min_trips (masses : List ℕ) (capacity : ℕ) : ℕ := sorry

-- State the theorem to be proven
theorem min_trips_correct :
  min_trips masses elevator_capacity = 6 := sorry

end min_trips_correct_l249_249315


namespace percentage_equivalence_l249_249232

theorem percentage_equivalence (x : ℝ) (h : 0.3 * 0.4 * x = 36) : 0.4 * 0.3 * x = 36 :=
by
  sorry

end percentage_equivalence_l249_249232


namespace combined_total_circles_squares_l249_249125

-- Define the problem parameters based on conditions
def US_stars : ℕ := 50
def US_stripes : ℕ := 13
def circles (n : ℕ) : ℕ := (n / 2) - 3
def squares (n : ℕ) : ℕ := (n * 2) + 6

-- Prove that the combined number of circles and squares on Pete's flag is 54
theorem combined_total_circles_squares : 
    circles US_stars + squares US_stripes = 54 := by
  sorry

end combined_total_circles_squares_l249_249125


namespace gcd_f_l249_249094

def f (x: ℤ) : ℤ := x^2 - x + 2023

theorem gcd_f (x y : ℤ) (hx : x = 105) (hy : y = 106) : Int.gcd (f x) (f y) = 7 := by
  sorry

end gcd_f_l249_249094


namespace card_draw_prob_l249_249470

/-- Define the total number of cards in the deck -/
def total_cards : ℕ := 52

/-- Define the total number of diamonds or aces -/
def diamonds_and_aces : ℕ := 16

/-- Define the probability of drawing a card that is a diamond or an ace in one draw -/
def prob_diamond_or_ace : ℚ := diamonds_and_aces / total_cards

/-- Define the complementary probability of not drawing a diamond nor ace in one draw -/
def prob_not_diamond_or_ace : ℚ := (total_cards - diamonds_and_aces) / total_cards

/-- Define the probability of not drawing a diamond nor ace in three draws with replacement -/
def prob_not_diamond_or_ace_three_draws : ℚ := prob_not_diamond_or_ace ^ 3

/-- Define the probability of drawing at least one diamond or ace in three draws with replacement -/
def prob_at_least_one_diamond_or_ace_in_three_draws : ℚ := 1 - prob_not_diamond_or_ace_three_draws

/-- The final probability calculated -/
def final_prob : ℚ := 1468 / 2197

theorem card_draw_prob :
  prob_at_least_one_diamond_or_ace_in_three_draws = final_prob := by
  sorry

end card_draw_prob_l249_249470


namespace probability_reroll_two_dice_is_given_correct_probability_l249_249802

noncomputable def probability_of_rerolling_two_dice_optimally : ℝ :=
  -- Total favorable outcomes for rerolling two dice to achieve sum 9 (assuming precomputed correct answer, e.g. 7/54)
  let total_favorable_outcomes := 7 in
  let total_possible_outcomes := 54 in 
  (total_favorable_outcomes : ℝ) / (total_possible_outcomes : ℝ)

theorem probability_reroll_two_dice_is_given_correct_probability :
  probability_of_rerolling_two_dice_optimally = 7 / 54 :=
  sorry

end probability_reroll_two_dice_is_given_correct_probability_l249_249802


namespace fraction_of_liars_l249_249477

theorem fraction_of_liars (n : ℕ) (villagers : Fin n → Prop) (right_neighbor : ∀ i, villagers i ↔ ∀ j : Fin n, j = (i + 1) % n → villagers j) :
  ∃ (x : ℚ), x = 1 / 2 :=
by 
  sorry

end fraction_of_liars_l249_249477


namespace john_total_climb_height_l249_249089

-- Define the heights and conditions
def num_flights : ℕ := 3
def height_per_flight : ℕ := 10
def total_stairs_height : ℕ := num_flights * height_per_flight
def rope_height : ℕ := total_stairs_height / 2
def ladder_height : ℕ := rope_height + 10

-- Prove that the total height John climbed is 70 feet
theorem john_total_climb_height : 
  total_stairs_height + rope_height + ladder_height = 70 := by
  sorry

end john_total_climb_height_l249_249089


namespace exponent_sum_l249_249769

theorem exponent_sum (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) :
  i^123 + i^223 + i^323 = -3 * i :=
by
  sorry

end exponent_sum_l249_249769


namespace factor_expression_l249_249918

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := 
by
  sorry

end factor_expression_l249_249918


namespace min_days_required_l249_249326

theorem min_days_required (n : ℕ) (h1 : n ≥ 1) (h2 : 2 * (2^n - 1) ≥ 100) : n = 6 :=
sorry

end min_days_required_l249_249326


namespace number_of_students_in_class_l249_249388

theorem number_of_students_in_class
  (G : ℕ) (E_and_G : ℕ) (E_only: ℕ)
  (h1 : G = 22)
  (h2 : E_and_G = 12)
  (h3 : E_only = 23) :
  ∃ S : ℕ, S = 45 :=
by
  sorry

end number_of_students_in_class_l249_249388


namespace equal_area_bisecting_line_slope_l249_249139

theorem equal_area_bisecting_line_slope 
  (circle1_center circle2_center : ℝ × ℝ) 
  (radius : ℝ) 
  (line_point : ℝ × ℝ) 
  (h1 : circle1_center = (20, 100))
  (h2 : circle2_center = (25, 90))
  (h3 : radius = 4)
  (h4 : line_point = (20, 90))
  : ∃ (m : ℝ), |m| = 2 :=
by
  sorry

end equal_area_bisecting_line_slope_l249_249139


namespace secret_sharing_problem_l249_249263

theorem secret_sharing_problem : 
  ∃ n : ℕ, (3280 = (3^(n + 1) - 1) / 2) ∧ (n = 7) :=
by
  use 7
  sorry

end secret_sharing_problem_l249_249263


namespace sin_300_eq_neg_one_half_l249_249537

-- Definitions from the conditions
def point_on_unit_circle (θ : ℝ) : ℝ × ℝ :=
  (Real.cos (θ), Real.sin (θ))

def foot_of_perpendicular (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, 0)

def is_thirty_sixty_ninety_triangle (P : ℝ × ℝ) : Prop :=
  sqrt(P.1^2 + P.2^2) = 1 ∧ P.1 = sqrt(3) / 2 ∧ P.2 = -1/2

-- Problem statement
theorem sin_300_eq_neg_one_half : Real.sin (5 * Real.pi / 3) = -1/2 := by
  sorry

end sin_300_eq_neg_one_half_l249_249537


namespace sin_300_eq_neg_sqrt3_div_2_l249_249498

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249498


namespace exists_midpoint_with_integer_coordinates_l249_249651

theorem exists_midpoint_with_integer_coordinates (points : Fin 5 → ℤ × ℤ) :
  ∃ (i j : Fin 5), i ≠ j ∧ ((points i).1 + (points j).1) % 2 = 0 ∧ ((points i).2 + (points j).2) % 2 = 0 :=
by
  sorry

end exists_midpoint_with_integer_coordinates_l249_249651


namespace triangle_expression_l249_249384

open Real

variable (D E F : ℝ)
variable (DE DF EF : ℝ)

-- conditions
def triangleDEF : Prop := DE = 7 ∧ DF = 9 ∧ EF = 8

theorem triangle_expression (h : triangleDEF DE DF EF) :
  (cos ((D - E)/2) / sin (F/2) - sin ((D - E)/2) / cos (F/2)) = 81/28 :=
by
  have h1 : DE = 7 := h.1
  have h2 : DF = 9 := h.2.1
  have h3 : EF = 8 := h.2.2
  sorry

end triangle_expression_l249_249384


namespace sin_300_eq_neg_sin_60_l249_249607

theorem sin_300_eq_neg_sin_60 :
  Real.sin (300 * Real.pi / 180) = - Real.sin (60 * Real.pi / 180) :=
by
suffices : Real.sin (300 * Real.pi / 180) = -sqrt 3 / 2
  sorry

end sin_300_eq_neg_sin_60_l249_249607


namespace directrix_of_parabola_l249_249772

theorem directrix_of_parabola (y x : ℝ) : 
  (∃ a h k : ℝ, y = a * (x - h)^2 + k ∧ a = 1/8 ∧ h = 4 ∧ k = 0) → 
  y = -1/2 :=
by
  intro h
  sorry

end directrix_of_parabola_l249_249772


namespace prince_spent_1000_l249_249021

noncomputable def total_CDs := 200
noncomputable def percent_CDs_10 := 0.40
noncomputable def price_per_CD_10 := 10
noncomputable def price_per_CD_5 := 5

-- Number of CDs sold at $10 each
noncomputable def num_CDs_10 := percent_CDs_10 * total_CDs

-- Number of CDs sold at $5 each
noncomputable def num_CDs_5 := total_CDs - num_CDs_10

-- Number of $10 CDs bought by Prince
noncomputable def prince_CDs_10 := num_CDs_10 / 2

-- Total cost of $10 CDs bought by Prince
noncomputable def cost_CDs_10 := prince_CDs_10 * price_per_CD_10

-- Total cost of $5 CDs bought by Prince
noncomputable def cost_CDs_5 := num_CDs_5 * price_per_CD_5

-- Total amount of money Prince spent
noncomputable def total_spent := cost_CDs_10 + cost_CDs_5

theorem prince_spent_1000 : total_spent = 1000 := by
  -- Definitions from conditions
  have h1 : total_CDs = 200 := rfl
  have h2 : percent_CDs_10 = 0.40 := rfl
  have h3 : price_per_CD_10 = 10 := rfl
  have h4 : price_per_CD_5 = 5 := rfl

  -- Calculations from solution steps (insert sorry to skip actual proofs)
  have h5 : num_CDs_10 = 80 := sorry
  have h6 : num_CDs_5 = 120 := sorry
  have h7 : prince_CDs_10 = 40 := sorry
  have h8 : cost_CDs_10 = 400 := sorry
  have h9 : cost_CDs_5 = 600 := sorry

  show total_spent = 1000
  sorry

end prince_spent_1000_l249_249021


namespace factor_expression_l249_249924

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249924


namespace largest_c_constant_l249_249352

noncomputable def max_c : ℝ :=
  sqrt(6) / 6

theorem largest_c_constant
  (n : ℕ) (h : n ≥ 3)
  (A : Fin n → Set ℝ)
  (H : ∃ T : Finset (Fin n) × Finset (Fin n) × Finset (Fin n), T.1 ≠ ∅ ∧ T.1.card = (n.choose 3) / 2 ∧
       ∀ (i j k : Fin n), i ∈ T.1 ∧ j ∈ T.2 ∧ k ∈ T.3 → 1 ≤ i.val ∧ i.val < j.val ∧ j.val < k.val ∧ (A i ∩ A j ∩ A k).nonempty) :
  ∃ I : Finset (Fin n), I.card > (sqrt(6) / 6) * n ∧ (⋂ i in I, A i).nonempty :=
sorry

end largest_c_constant_l249_249352


namespace point_on_right_branch_l249_249378

noncomputable def on_hyperbola_right_branch (a b m : ℝ) :=
  (∀ a b m : ℝ, (a - 2 * b > 0) → (a + 2 * b > 0) → (a ^ 2 - 4 * b ^ 2 = m) → (m ≠ 0) → a > 0)

theorem point_on_right_branch (a b m : ℝ) (h₁ : a - 2 * b > 0) (h₂ : a + 2 * b > 0) (h₃ : a ^ 2 - 4 * b ^ 2 = m) (h₄ : m ≠ 0) :
  a > 0 := 
by 
  sorry

end point_on_right_branch_l249_249378


namespace tetrahedron_labeling_count_l249_249196

def is_valid_tetrahedron_labeling (labeling : Fin 4 → ℕ) : Prop :=
  let f1 := labeling 0 + labeling 1 + labeling 2
  let f2 := labeling 0 + labeling 1 + labeling 3
  let f3 := labeling 0 + labeling 2 + labeling 3
  let f4 := labeling 1 + labeling 2 + labeling 3
  labeling 0 + labeling 1 + labeling 2 + labeling 3 = 10 ∧ 
  f1 = f2 ∧ f2 = f3 ∧ f3 = f4

theorem tetrahedron_labeling_count : 
  ∃ (n : ℕ), n = 3 ∧ (∃ (labelings: Finset (Fin 4 → ℕ)), 
  ∀ labeling ∈ labelings, is_valid_tetrahedron_labeling labeling) :=
sorry

end tetrahedron_labeling_count_l249_249196


namespace sin_300_deg_l249_249489

noncomputable def sin_300_equals_neg_sqrt3_div_2 : Prop :=
  sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2

theorem sin_300_deg : sin_300_equals_neg_sqrt3_div_2 :=
by
  sorry

end sin_300_deg_l249_249489


namespace calculate_expression_evaluate_expression_l249_249186

theorem calculate_expression (a : ℕ) (h : a = 2020) :
  (a^4 - 3*a^3*(a+1) + 4*a*(a+1)^3 - (a+1)^4 + 1) / (a*(a+1)) = a^2 + 4*a + 6 :=
by sorry

theorem evaluate_expression :
  (2020^2 + 4 * 2020 + 6) = 4096046 :=
by sorry

end calculate_expression_evaluate_expression_l249_249186


namespace pete_flag_total_circle_square_l249_249128

theorem pete_flag_total_circle_square : 
  let stars := 50
  let stripes := 13
  let circles := (stars / 2) - 3
  let squares := (stripes * 2) + 6
  circles + squares = 54 := 
by
  let stars := 50
  let stripes := 13
  let circles := (stars / 2) - 3
  let squares := (stripes * 2) + 6
  show circles + squares = 54
  sorry

end pete_flag_total_circle_square_l249_249128


namespace farmer_land_l249_249419

variable (A C G P T : ℝ)
variable (h1 : C = 0.90 * A)
variable (h2 : G = 0.10 * C)
variable (h3 : P = 0.80 * C)
variable (h4 : T = 450)
variable (h5 : C = G + P + T)

theorem farmer_land (A : ℝ) (h1 : C = 0.90 * A) (h2 : G = 0.10 * C) (h3 : P = 0.80 * C) (h4 : T = 450) (h5 : C = G + P + T) : A = 5000 := by
  sorry

end farmer_land_l249_249419


namespace sin_300_eq_neg_sqrt3_div_2_l249_249557

/-- Prove that the sine of 300 degrees is equal to -√3/2. -/
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249557


namespace midpoint_coords_l249_249950

noncomputable def F1 : (ℝ × ℝ) := (-2 * Real.sqrt 2, 0)
noncomputable def F2 : (ℝ × ℝ) := (2 * Real.sqrt 2, 0)
def major_axis_length : ℝ := 6
def line_eq (x y : ℝ) : Prop := x - y + 2 = 0

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  let a := 3
  let b := 1
  (x^2) / (a^2) + y^2 / (b^2) = 1

theorem midpoint_coords :
  ∃ (A B : ℝ × ℝ), ellipse_eq A.1 A.2 ∧ ellipse_eq B.1 B.2 ∧ line_eq A.1 A.2 ∧ line_eq B.1 B.2 →
  (A.1 + B.1) / 2 = -9 / 5 ∧ (A.2 + B.2) / 2 = 1 / 5 :=
by
  sorry

end midpoint_coords_l249_249950


namespace students_neither_math_physics_l249_249983

theorem students_neither_math_physics (total_students math_students physics_students both_students : ℕ) 
  (h1 : total_students = 120)
  (h2 : math_students = 80)
  (h3 : physics_students = 50)
  (h4 : both_students = 15) : 
  total_students - (math_students - both_students + physics_students - both_students + both_students) = 5 :=
by
  -- Each of the hypotheses are used exactly as given in the conditions.
  -- We omit the proof as requested.
  sorry

end students_neither_math_physics_l249_249983


namespace opposite_of_neg_five_l249_249444

theorem opposite_of_neg_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  sorry

end opposite_of_neg_five_l249_249444


namespace sin_300_eq_neg_sqrt3_div_2_l249_249531

def point_on_unit_circle (θ : ℝ) : ℝ × ℝ :=
⟨Real.cos θ, Real.sin θ⟩

theorem sin_300_eq_neg_sqrt3_div_2 :
  let Q := point_on_unit_circle (300 * Real.pi / 180) in
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
by sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249531


namespace youngest_child_age_l249_249288

theorem youngest_child_age (x : ℕ) 
  (h : x + (x + 4) + (x + 8) + (x + 12) + (x + 16) + (x + 20) + (x + 24) = 112) : 
  x = 4 := by
  sorry

end youngest_child_age_l249_249288


namespace a_gt_b_iff_a_ln_a_gt_b_ln_b_l249_249359

theorem a_gt_b_iff_a_ln_a_gt_b_ln_b {a b : ℝ} (ha : a > 0) (hb : b > 0) : 
  (a > b) ↔ (a + Real.log a > b + Real.log b) :=
by sorry

end a_gt_b_iff_a_ln_a_gt_b_ln_b_l249_249359


namespace abs_a_lt_abs_b_sub_abs_c_l249_249937

theorem abs_a_lt_abs_b_sub_abs_c (a b c : ℝ) (h : |a + c| < b) : |a| < |b| - |c| :=
sorry

end abs_a_lt_abs_b_sub_abs_c_l249_249937


namespace maple_taller_than_birch_l249_249253

def birch_tree_height : ℚ := 49 / 4
def maple_tree_height : ℚ := 102 / 5

theorem maple_taller_than_birch : maple_tree_height - birch_tree_height = 163 / 20 :=
by
  sorry

end maple_taller_than_birch_l249_249253


namespace min_sum_x8y4z_l249_249406

theorem min_sum_x8y4z (x y z : ℝ) (h : 4 / x + 2 / y + 1 / z = 1) : x + 8 * y + 4 * z ≥ 64 := 
sorry

end min_sum_x8y4z_l249_249406


namespace sin_300_eq_neg_sqrt3_div_2_l249_249550

/-- Prove that the sine of 300 degrees is equal to -√3/2. -/
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249550


namespace line_circle_intersections_l249_249627

-- Define the line equation as a predicate
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y = 12

-- Define the circle equation as a predicate
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 16

-- The goal is to prove the number of intersections of the line and the circle
theorem line_circle_intersections : (∃ x y : ℝ, line_eq x y ∧ circle_eq x y) ∧ 
                                   (∃ x y : ℝ, line_eq x y ∧ circle_eq x y ∧ x ≠ y) :=
sorry

end line_circle_intersections_l249_249627


namespace probability_heads_penny_nickel_dime_l249_249114

theorem probability_heads_penny_nickel_dime :
  let total_outcomes := 2^5 in
  let successful_outcomes := 2 * 2 in
  (successful_outcomes : ℝ) / total_outcomes = (1 / 8 : ℝ) :=
by
  sorry

end probability_heads_penny_nickel_dime_l249_249114


namespace line_passes_fixed_point_l249_249789

theorem line_passes_fixed_point (k b : ℝ) (h : -1 = (k + b) / 2) :
  ∃ (x y : ℝ), x = 1 ∧ y = -2 ∧ y = k * x + b :=
by
  sorry

end line_passes_fixed_point_l249_249789


namespace cannot_finish_third_l249_249271

-- Definitions for the orders of runners
def order (a b : String) : Prop := a < b

-- The problem statement and conditions
def conditions (P Q R S T U : String) : Prop :=
  order P Q ∧ order P R ∧ order Q S ∧ order P U ∧ order U T ∧ order T Q

theorem cannot_finish_third (P Q R S T U : String) (h : conditions P Q R S T U) :
  (P = "third" → False) ∧ (S = "third" → False) :=
by
  sorry

end cannot_finish_third_l249_249271


namespace sin_300_eq_neg_sqrt3_div_2_l249_249569

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249569


namespace expression_value_l249_249462

theorem expression_value (a : ℝ) (h : a = 1/3) : 
  (4 * a⁻¹ - 2 * a⁻¹ / 3) / a^2 = 90 := by
  sorry

end expression_value_l249_249462


namespace circle_line_intersection_zero_l249_249711

theorem circle_line_intersection_zero (x_0 y_0 r : ℝ) (hP : x_0^2 + y_0^2 < r^2) :
  ∀ (x y : ℝ), (x^2 + y^2 = r^2) → (x_0 * x + y_0 * y = r^2) → false :=
by
  sorry

end circle_line_intersection_zero_l249_249711


namespace magical_stack_example_l249_249438

-- Definitions based on the conditions
def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def belongs_to_pile_A (card : ℕ) (n : ℕ) : Prop :=
  card <= n

def belongs_to_pile_B (card : ℕ) (n : ℕ) : Prop :=
  n < card

def magical_stack (cards : ℕ) (n : ℕ) : Prop :=
  ∀ (card : ℕ), (belongs_to_pile_A card n ∨ belongs_to_pile_B card n) → 
  (card + n) % (2 * n) = 1

-- The theorem to prove
theorem magical_stack_example :
  ∃ (n : ℕ), magical_stack 482 n ∧ (2 * n = 482) :=
by
  sorry

end magical_stack_example_l249_249438


namespace ratio_of_time_l249_249319

theorem ratio_of_time (T_A T_B : ℝ) (h1 : T_A = 8) (h2 : 1 / T_A + 1 / T_B = 0.375) :
  T_B / T_A = 1 / 2 :=
by 
  sorry

end ratio_of_time_l249_249319


namespace S7_is_28_l249_249780

variables {a_n : ℕ → ℤ} -- Sequence definition
variables {S_n : ℕ → ℤ} -- Sum of the first n terms

-- Define an arithmetic sequence condition
def is_arithmetic_sequence (a_n : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

-- Given conditions
axiom sum_condition : a_n 2 + a_n 4 + a_n 6 = 12
axiom sum_formula (n : ℕ) : S_n n = n * (a_n 1 + a_n n) / 2
axiom arith_seq : is_arithmetic_sequence a_n

-- The statement to be proven
theorem S7_is_28 : S_n 7 = 28 :=
sorry

end S7_is_28_l249_249780


namespace find_price_of_100_apples_l249_249870

noncomputable def price_of_100_apples (P : ℕ) : Prop :=
  (12000 / P) - (12000 / (P + 4)) = 5

theorem find_price_of_100_apples : price_of_100_apples 96 :=
by
  sorry

end find_price_of_100_apples_l249_249870


namespace arithmetic_seq_sum_equidistant_l249_249073

variable (a : ℕ → ℤ)

theorem arithmetic_seq_sum_equidistant :
  (∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) → a 4 = 12 → a 1 + a 7 = 24 :=
by
  intros h_seq h_a4
  sorry

end arithmetic_seq_sum_equidistant_l249_249073


namespace upstream_speed_l249_249166

-- Speed of the man in still water
def V_m : ℕ := 32

-- Speed of the man rowing downstream
def V_down : ℕ := 42

-- Speed of the stream
def V_s : ℕ := V_down - V_m

-- Speed of the man rowing upstream
def V_up : ℕ := V_m - V_s

theorem upstream_speed (V_m : ℕ) (V_down : ℕ) (V_s : ℕ) (V_up : ℕ) : 
  V_m = 32 → 
  V_down = 42 → 
  V_s = V_down - V_m → 
  V_up = V_m - V_s → 
  V_up = 22 := 
by intros; 
   repeat {sorry}

end upstream_speed_l249_249166


namespace length_of_parallelepiped_l249_249000

def number_of_cubes_with_painted_faces (n : ℕ) := (n - 2) * (n - 4) * (n - 6) 
def total_number_of_cubes (n : ℕ) := n * (n - 2) * (n - 4)

theorem length_of_parallelepiped (n : ℕ) (h1 : total_number_of_cubes n = 3 * number_of_cubes_with_painted_faces n) : 
  n = 18 :=
by 
  sorry

end length_of_parallelepiped_l249_249000


namespace sin_of_300_degrees_l249_249567

theorem sin_of_300_degrees : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 := by
  sorry

end sin_of_300_degrees_l249_249567


namespace factor_expression_l249_249902

theorem factor_expression (x : ℝ) : (x * (x + 3) + 2 * (x + 3)) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249902


namespace sin_300_eq_neg_sqrt_3_div_2_l249_249511

theorem sin_300_eq_neg_sqrt_3_div_2 : Real.sin (300 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt_3_div_2_l249_249511


namespace group_selection_l249_249691

theorem group_selection (m k n : ℕ) (h_m : m = 6) (h_k : k = 7) 
  (groups : ℕ → ℕ) (h_groups : groups k = n) : 
  n % 10 = (m + k) % 10 :=
by
  sorry

end group_selection_l249_249691


namespace sin_300_eq_neg_sqrt3_div_2_l249_249576

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249576


namespace find_crayons_in_pack_l249_249337

variables (crayons_in_locker : ℕ) (crayons_given_by_bobby : ℕ) (crayons_given_to_mary : ℕ) (crayons_final_count : ℕ) (crayons_in_pack : ℕ)

-- Definitions from the conditions
def initial_crayons := 36
def bobby_gave := initial_crayons / 2
def mary_crayons := 25
def final_crayons := initial_crayons + bobby_gave - mary_crayons

-- The theorem to prove
theorem find_crayons_in_pack : initial_crayons = 36 ∧ bobby_gave = 18 ∧ mary_crayons = 25 ∧ final_crayons = 29 → crayons_in_pack = 29 :=
by
  sorry

end find_crayons_in_pack_l249_249337


namespace simplify_and_evaluate_expression_l249_249105

noncomputable def expression (a : ℝ) : ℝ :=
  ((a^2 - 1) / (a - 3) - a - 1) / ((a + 1) / (a^2 - 6 * a + 9))

theorem simplify_and_evaluate_expression : expression (3 - Real.sqrt 2) = -2 * Real.sqrt 2 :=
by
  sorry

end simplify_and_evaluate_expression_l249_249105


namespace min_positive_announcements_l249_249180

theorem min_positive_announcements (x y : ℕ) 
  (h1 : x * (x - 1) = 90)
  (h2 : y * (y - 1) + (x - y) * (x - y - 1) = 48) 
  : y = 3 :=
sorry

end min_positive_announcements_l249_249180


namespace sin_300_eq_neg_sqrt_three_div_two_l249_249544

theorem sin_300_eq_neg_sqrt_three_div_two :
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt_three_div_two_l249_249544


namespace cakes_sold_l249_249482

theorem cakes_sold (total_made : ℕ) (cakes_left : ℕ) (cakes_sold : ℕ) :
  total_made = 217 ∧ cakes_left = 72 → cakes_sold = 145 :=
by
  -- Assuming total_made is 217 and cakes_left is 72, we need to show cakes_sold = 145
  sorry

end cakes_sold_l249_249482


namespace melanie_turnips_l249_249097

theorem melanie_turnips (b : ℕ) (d : ℕ) (h_b : b = 113) (h_d : d = 26) : b + d = 139 :=
by
  sorry

end melanie_turnips_l249_249097


namespace device_prices_within_budget_l249_249252

-- Given conditions
def x : ℝ := 12 -- Price of each type A device in thousands of dollars
def y : ℝ := 10 -- Price of each type B device in thousands of dollars
def budget : ℝ := 110 -- The budget in thousands of dollars

-- Conditions as given equations and inequalities
def condition1 : Prop := 3 * x - 2 * y = 16
def condition2 : Prop := 3 * y - 2 * x = 6
def budget_condition (a : ℕ) : Prop := 12 * a + 10 * (10 - a) ≤ budget

-- Theorem to prove
theorem device_prices_within_budget :
  condition1 ∧ condition2 ∧
  (∀ a : ℕ, a ≤ 5 → budget_condition a) :=
by sorry

end device_prices_within_budget_l249_249252


namespace apples_left_l249_249981

def Mike_apples : ℝ := 7.0
def Nancy_apples : ℝ := 3.0
def Keith_ate_apples : ℝ := 6.0

theorem apples_left : Mike_apples + Nancy_apples - Keith_ate_apples = 4.0 := by
  sorry

end apples_left_l249_249981


namespace initial_red_marbles_l249_249687

theorem initial_red_marbles (r g : ℕ) 
  (h1 : r = 5 * g / 3) 
  (h2 : (r - 20) * 5 = g + 40) : 
  r = 317 :=
by
  sorry

end initial_red_marbles_l249_249687


namespace sin_300_eq_neg_sqrt3_div_2_l249_249553

/-- Prove that the sine of 300 degrees is equal to -√3/2. -/
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249553


namespace opposite_of_neg_five_l249_249443

theorem opposite_of_neg_five : ∃ x : ℤ, -5 + x = 0 ∧ x = 5 :=
by
  use 5
  sorry

end opposite_of_neg_five_l249_249443


namespace math_problem_l249_249437

theorem math_problem (a b c : ℝ) (h1 : (a + b) / 2 = 30) (h2 : (b + c) / 2 = 60) (h3 : c - a = 60) : c - a = 60 :=
by
  -- Insert proof steps here
  sorry

end math_problem_l249_249437


namespace hexagon_inscribed_in_square_area_l249_249360

noncomputable def hexagon_area (side_length : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * side_length^2

theorem hexagon_inscribed_in_square_area (AB BC : ℝ) (BDEF_square : BDEF_is_square) (hAB : AB = 2) (hBC : BC = 2) :
  hexagon_area (2 * Real.sqrt 2) = 12 * Real.sqrt 3 :=
by
  sorry

-- Definitions to assume the necessary conditions in the theorem (placeholders)
-- Assuming a structure of BDEF_is_square to represent the property that BDEF is a square
structure BDEF_is_square :=
(square : Prop)

end hexagon_inscribed_in_square_area_l249_249360


namespace sin_300_eq_neg_sqrt3_div_2_l249_249551

/-- Prove that the sine of 300 degrees is equal to -√3/2. -/
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249551


namespace greatest_k_for_inquality_l249_249640

theorem greatest_k_for_inquality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a^2 > b*c) :
    (a^2 - b*c)^2 > 4 * ((b^2 - c*a) * (c^2 - a*b)) :=
  sorry

end greatest_k_for_inquality_l249_249640


namespace factor_expression_l249_249896

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249896


namespace arithmetic_sequence_sum_l249_249995

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the arithmetic property of the sequence
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (h1 : is_arithmetic_sequence a d)
  (h2 : a 2 + a 4 + a 7 + a 11 = 44) :
  a 3 + a 5 + a 10 = 33 := 
sorry

end arithmetic_sequence_sum_l249_249995


namespace primes_solution_l249_249771

theorem primes_solution (p q : ℕ) (m n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hm : m ≥ 2) (hn : n ≥ 2) :
    p^n = q^m + 1 ∨ p^n = q^m - 1 → (p = 2 ∧ n = 3 ∧ q = 3 ∧ m = 2) :=
by
  sorry

end primes_solution_l249_249771


namespace find_prime_pairs_l249_249635

def is_solution_pair (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ (p ∣ 5^q + 1) ∧ (q ∣ 5^p + 1)

theorem find_prime_pairs :
  {pq : ℕ × ℕ | is_solution_pair pq.1 pq.2} =
  { (2, 13), (13, 2), (3, 7), (7, 3) } :=
by
  sorry

end find_prime_pairs_l249_249635


namespace conditional_probability_l249_249236

open Probability

def study_group : Finset (Fin 6) := {0, 1, 2, 3, 4, 5}
def halls : Finset (Finite 3) := {0, 1, 2} -- Halls A, B, C

noncomputable def event_A (visiting_order : Fin 6 → Fin 3) : Prop :=
  ∀ h : Fin 3, (study_group.filter (λ i => visiting_order i = h)).card = 2

noncomputable def event_B (visiting_order_2 : Fin 6 → Fin 3) : Prop :=
  (study_group.filter (λ i => visiting_order_2 i = 0)).card = 2

theorem conditional_probability
  (visiting_order_1 visiting_order_2 : Fin 6 → Fin 3)
  (h_A : event_A visiting_order_1) :
  P (event_B visiting_order_2) ∣ event_A visiting_order_1 = 3 / 8 :=
sorry

end conditional_probability_l249_249236


namespace students_total_l249_249837

theorem students_total (scavenger_hunt_students : ℕ) (ski_trip_students : ℕ) 
  (h1 : ski_trip_students = 2 * scavenger_hunt_students) 
  (h2 : scavenger_hunt_students = 4000) : 
  scavenger_hunt_students + ski_trip_students = 12000 := 
by
  sorry

end students_total_l249_249837


namespace solution_set_of_quadratic_inequality_l249_249383

theorem solution_set_of_quadratic_inequality 
  (a b c x₁ x₂ : ℝ)
  (h1 : a > 0) 
  (h2 : a * x₁^2 + b * x₁ + c = 0)
  (h3 : a * x₂^2 + b * x₂ + c = 0)
  : {x : ℝ | a * x^2 + b * x + c > 0} = ({x : ℝ | x > x₁} ∩ {x : ℝ | x > x₂}) ∪ ({x : ℝ | x < x₁} ∩ {x : ℝ | x < x₂}) :=
sorry

end solution_set_of_quadratic_inequality_l249_249383


namespace percent_problem_l249_249231

theorem percent_problem (x : ℝ) (h : 0.30 * 0.40 * x = 36) : 0.40 * 0.30 * x = 36 :=
by
  sorry

end percent_problem_l249_249231


namespace yolks_in_carton_l249_249750

/-- A local farm is famous for having lots of double yolks in their eggs. One carton of 12 eggs had five eggs with double yolks. Prove that the total number of yolks in the whole carton is equal to 17. -/
theorem yolks_in_carton (total_eggs : ℕ) (double_yolk_eggs : ℕ) (single_yolk_per_egg : ℕ) (double_yolk_per_egg : ℕ) 
    (total_eggs = 12) (double_yolk_eggs = 5) (single_yolk_per_egg = 1) (double_yolk_per_egg = 2) : 
    (double_yolk_eggs * double_yolk_per_egg + (total_eggs - double_yolk_eggs) * single_yolk_per_egg) = 17 := 
by
    sorry

end yolks_in_carton_l249_249750


namespace gain_percentage_of_watch_l249_249175

theorem gain_percentage_of_watch :
  let CP := 1076.923076923077
  let S1 := CP * 0.90
  let S2 := S1 + 140
  let gain_percentage := ((S2 - CP) / CP) * 100
  gain_percentage = 3 := by
  sorry

end gain_percentage_of_watch_l249_249175


namespace composite_sum_l249_249702

theorem composite_sum (m n : ℕ) (h : 88 * m = 81 * n) : ∃ p q : ℕ, p > 1 ∧ q > 1 ∧ (m + n) = p * q :=
by sorry

end composite_sum_l249_249702


namespace common_difference_arithmetic_sequence_l249_249357

noncomputable def a_n (n : ℕ) : ℤ := 5 - 4 * n

theorem common_difference_arithmetic_sequence :
  ∀ n ≥ 1, a_n n - a_n (n - 1) = -4 :=
by
  intros n hn
  unfold a_n
  sorry

end common_difference_arithmetic_sequence_l249_249357


namespace sin_300_eq_neg_sqrt3_div_2_l249_249556

/-- Prove that the sine of 300 degrees is equal to -√3/2. -/
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249556


namespace arithmetic_seq_third_sum_l249_249968

-- Define the arithmetic sequence using its first term and common difference
def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + d * n

theorem arithmetic_seq_third_sum
  (a₁ d : ℤ)
  (h1 : (a₁ + (a₁ + 3 * d) + (a₁ + 6 * d) = 39))
  (h2 : ((a₁ + d) + (a₁ + 4 * d) + (a₁ + 7 * d) = 33)) :
  ((a₁ + 2 * d) + (a₁ + 5 * d) + (a₁ + 8 * d) = 27) :=
by
  sorry

end arithmetic_seq_third_sum_l249_249968


namespace num_sol_and_sum_sol_l249_249220

-- Definition of the main problem condition
def equation (x : ℝ) := (4 * x^2 - 9)^2 = 49

-- Proof problem statement
theorem num_sol_and_sum_sol :
  (∃ s : Finset ℝ, (∀ x, equation x ↔ x ∈ s) ∧ s.card = 4 ∧ s.sum id = 0) :=
sorry

end num_sol_and_sum_sol_l249_249220


namespace parabola_directrix_l249_249661

theorem parabola_directrix (p : ℝ) (hp : p > 0) 
  (hC : ∀ (x y : ℝ), y^2 = 2 * p * x → x = (y^2 / (2 * p))) :
  (let F := (p / 2, 0 : ℝ) in
  let P := (p / 2, p : ℝ) in
  let Q := (5 * p / 2, 0 : ℝ) in
  dist F Q = 6 → x = - (3 / 2)) :=
begin
  sorry
end

end parabola_directrix_l249_249661


namespace sin_300_eq_neg_sqrt3_div_2_l249_249592

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249592


namespace line_ellipse_tangent_l249_249057

theorem line_ellipse_tangent (m : ℝ) (h : ∃ x y : ℝ, y = 2 * m * x + 2 ∧ 2 * x^2 + 8 * y^2 = 8) :
  m^2 = 3 / 16 :=
sorry

end line_ellipse_tangent_l249_249057


namespace box_volume_increase_l249_249168

-- Conditions
def volume (l w h : ℝ) : ℝ := l * w * h
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + w * h + h * l)
def sum_of_edges (l w h : ℝ) : ℝ := 4 * (l + w + h)

-- The main theorem we want to state
theorem box_volume_increase
  (l w h : ℝ)
  (h_volume : volume l w h = 5000)
  (h_surface_area : surface_area l w h = 1800)
  (h_sum_of_edges : sum_of_edges l w h = 210) :
  volume (l + 2) (w + 2) (h + 2) = 7018 := 
by sorry

end box_volume_increase_l249_249168


namespace arith_seq_largest_portion_l249_249821

theorem arith_seq_largest_portion (a1 d : ℝ) (h_d_pos : d > 0) 
  (h_sum : 5 * a1 + 10 * d = 100)
  (h_ratio : (3 * a1 + 9 * d) / 7 = 2 * a1 + d) : 
  a1 + 4 * d = 115 / 3 := by
  sorry

end arith_seq_largest_portion_l249_249821


namespace percentage_of_diameter_l249_249440

variable (d_R d_S r_R r_S : ℝ)
variable (A_R A_S : ℝ)
variable (pi : ℝ) (h1 : pi > 0)

theorem percentage_of_diameter 
(h_area : A_R = 0.64 * A_S) 
(h_radius_R : r_R = d_R / 2) 
(h_radius_S : r_S = d_S / 2)
(h_area_R : A_R = pi * r_R^2) 
(h_area_S : A_S = pi * r_S^2) 
: (d_R / d_S) * 100 = 80 := by
  sorry

end percentage_of_diameter_l249_249440


namespace lottery_consecutive_probability_l249_249722

noncomputable def lottery_probability : ℚ :=
1 - (choose 86 5 : ℚ) / (choose 90 5)

theorem lottery_consecutive_probability :
  lottery_probability = 0.2 := sorry

end lottery_consecutive_probability_l249_249722


namespace min_value_of_sum_l249_249218

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 3 * a + 2 * b = 1) : 
  (∃ x, x = (3 / a + 2 / b) ∧ x = 25) :=
sorry

end min_value_of_sum_l249_249218


namespace Dan_reaches_Cate_in_25_seconds_l249_249344

theorem Dan_reaches_Cate_in_25_seconds
  (d : ℝ) (v_d : ℝ) (v_c : ℝ)
  (h1 : d = 50)
  (h2 : v_d = 8)
  (h3 : v_c = 6) :
  (d / (v_d - v_c) = 25) :=
by
  sorry

end Dan_reaches_Cate_in_25_seconds_l249_249344


namespace sin_300_eq_neg_sqrt3_div_2_l249_249575

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249575


namespace exist_ordering_rectangles_l249_249079

open Function

structure Rectangle :=
  (left_bot : ℝ × ℝ)  -- Bottom-left corner
  (right_top : ℝ × ℝ)  -- Top-right corner

def below (R1 R2 : Rectangle) : Prop :=
  ∃ g : ℝ, (∀ (x y : ℝ), R1.left_bot.1 ≤ x ∧ x ≤ R1.right_top.1 ∧ R1.left_bot.2 ≤ y ∧ y ≤ R1.right_top.2 → y < g) ∧
           (∀ (x y : ℝ), R2.left_bot.1 <= x ∧ x <= R2.right_top.1 ∧ R2.left_bot.2 <= y ∧ y <= R2.right_top.2 → y > g)

def to_right_of (R1 R2 : Rectangle) : Prop :=
  ∃ h : ℝ, (∀ (x y : ℝ), R1.left_bot.1 ≤ x ∧ x ≤ R1.right_top.1 ∧ R1.left_bot.2 ≤ y ∧ y ≤ R1.right_top.2 → x > h) ∧
           (∀ (x y : ℝ), R2.left_bot.1 <= x ∧ x <= R2.right_top.1 ∧ R2.left_bot.2 <= y ∧ y <= R2.right_top.2 → x < h)

def disjoint (R1 R2 : Rectangle) : Prop :=
  ¬ ((R1.left_bot.1 < R2.right_top.1) ∧ (R1.right_top.1 > R2.left_bot.1) ∧
     (R1.left_bot.2 < R2.right_top.2) ∧ (R1.right_top.2 > R2.left_bot.2))

theorem exist_ordering_rectangles (n : ℕ) (rectangles : Fin n → Rectangle)
  (h_disjoint : ∀ i j, i ≠ j → disjoint (rectangles i) (rectangles j)) :
  ∃ f : Fin n → Fin n, ∀ i j : Fin n, i < j → 
    (to_right_of (rectangles (f i)) (rectangles (f j)) ∨ 
    below (rectangles (f i)) (rectangles (f j))) := 
sorry

end exist_ordering_rectangles_l249_249079


namespace factor_expression_l249_249911

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249911


namespace min_distance_y_axis_l249_249942

open Real

noncomputable def distance (A B : ℝ × ℝ) : ℝ := 
  (sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2))

theorem min_distance_y_axis (P : ℝ × ℝ) (h_on_y_axis : P.1 = 0) (A := (2, 5) : ℝ × ℝ) (B := (4, -1) : ℝ × ℝ) : 
  let B' := (-B.1, B.2) in
  let P' := (0, 3) in
  ∀ P, P = P' ∧ distance A P + distance B P = distance A P' + distance B P' :=
by
  sorry

end min_distance_y_axis_l249_249942


namespace point_movement_l249_249800

theorem point_movement (P : ℤ) (hP : P = -5) (k : ℤ) (hk : (k = 3 ∨ k = -3)) :
  P + k = -8 ∨ P + k = -2 :=
by {
  sorry
}

end point_movement_l249_249800


namespace prob_both_shoot_in_one_round_prob_specified_shots_in_two_rounds_l249_249759

noncomputable def P_A := 4 / 5
noncomputable def P_B := 3 / 4

def independent (P_X P_Y : ℚ) := P_X * P_Y

theorem prob_both_shoot_in_one_round : independent P_A P_B = 3 / 5 := by
  sorry

noncomputable def P_A_1 := 2 * (4 / 5) * (1 / 5)
noncomputable def P_A_2 := (4 / 5) * (4 / 5)
noncomputable def P_B_1 := 2 * (3 / 4) * (1 / 4)
noncomputable def P_B_2 := (3 / 4) * (3 / 4)

def event_A (P_A_1 P_A_2 P_B_1 P_B_2 : ℚ) := (P_A_1 * P_B_2) + (P_A_2 * P_B_1)

theorem prob_specified_shots_in_two_rounds : event_A P_A_1 P_A_2 P_B_1 P_B_2 = 3 / 10 := by
  sorry

end prob_both_shoot_in_one_round_prob_specified_shots_in_two_rounds_l249_249759


namespace mnpq_product_l249_249825

noncomputable def prove_mnpq_product (a b x y : ℝ) : Prop :=
  ∃ (m n p q : ℤ), (a^m * x - a^n) * (a^p * y - a^q) = a^3 * b^4 ∧
                    m * n * p * q = 4

theorem mnpq_product (a b x y : ℝ) (h : a^7 * x * y - a^6 * y - a^5 * x = a^3 * (b^4 - 1)) :
  prove_mnpq_product a b x y :=
sorry

end mnpq_product_l249_249825


namespace ratio_of_time_charged_l249_249267

theorem ratio_of_time_charged (P K M : ℕ) (r : ℚ) 
  (h1 : P + K + M = 144) 
  (h2 : P = r * K)
  (h3 : P = 1/3 * M)
  (h4 : M = K + 80) : 
  r = 2 := 
  sorry

end ratio_of_time_charged_l249_249267


namespace red_flowers_needed_l249_249281

-- Define the number of white and red flowers
def white_flowers : ℕ := 555
def red_flowers : ℕ := 347

-- Define the problem statement.
theorem red_flowers_needed : red_flowers + 208 = white_flowers := by
  -- The proof goes here.
  sorry

end red_flowers_needed_l249_249281


namespace factor_expression_l249_249919

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249919


namespace number_of_pipes_used_l249_249173

-- Definitions
def T1 : ℝ := 15
def T2 : ℝ := T1 - 5
def T3 : ℝ := T2 - 4
def condition : Prop := 1 / T1 + 1 / T2 = 1 / T3

-- Proof Statement
theorem number_of_pipes_used : condition → 3 = 3 :=
by intros h; sorry

end number_of_pipes_used_l249_249173


namespace acrobats_count_l249_249963

theorem acrobats_count
  (a e c : ℕ)
  (h1 : 2 * a + 4 * e + 2 * c = 58)
  (h2 : a + e + c = 25) :
  a = 11 :=
by
  -- Proof skipped
  sorry

end acrobats_count_l249_249963


namespace profit_shares_difference_l249_249150

theorem profit_shares_difference (total_profit : ℝ) (share_ratio_x share_ratio_y : ℝ) 
  (hx : share_ratio_x = 1/2) (hy : share_ratio_y = 1/3) (profit : ℝ):
  total_profit = 500 → profit = (total_profit * share_ratio_x) / ((share_ratio_x + share_ratio_y)) - (total_profit * share_ratio_y) / ((share_ratio_x + share_ratio_y)) → profit = 100 :=
by
  intros
  sorry

end profit_shares_difference_l249_249150


namespace sin_300_eq_neg_one_half_l249_249532

-- Definitions from the conditions
def point_on_unit_circle (θ : ℝ) : ℝ × ℝ :=
  (Real.cos (θ), Real.sin (θ))

def foot_of_perpendicular (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, 0)

def is_thirty_sixty_ninety_triangle (P : ℝ × ℝ) : Prop :=
  sqrt(P.1^2 + P.2^2) = 1 ∧ P.1 = sqrt(3) / 2 ∧ P.2 = -1/2

-- Problem statement
theorem sin_300_eq_neg_one_half : Real.sin (5 * Real.pi / 3) = -1/2 := by
  sorry

end sin_300_eq_neg_one_half_l249_249532


namespace divides_difference_l249_249204

theorem divides_difference (n : ℕ) (h_composite : ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k) : 
  6 ∣ ((n^2)^3 - n^2) := 
sorry

end divides_difference_l249_249204


namespace adjusted_area_difference_l249_249760

noncomputable def largest_circle_area (d : ℝ) : ℝ :=
  let r := d / 2
  r^2 * Real.pi

noncomputable def middle_circle_area (r : ℝ) : ℝ :=
  r^2 * Real.pi

noncomputable def smaller_circle_area (r : ℝ) : ℝ :=
  r^2 * Real.pi

theorem adjusted_area_difference (d_large r_middle r_small : ℝ) 
  (h_large : d_large = 30) (h_middle : r_middle = 10) (h_small : r_small = 5) :
  largest_circle_area d_large - middle_circle_area r_middle - smaller_circle_area r_small = 100 * Real.pi :=
by
  sorry

end adjusted_area_difference_l249_249760


namespace sum_of_reciprocals_l249_249737

noncomputable def reciprocal_sum (x y : ℝ) : ℝ :=
  (1 / x) + (1 / y)

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 375) :
  reciprocal_sum x y = 8 / 75 :=
by
  unfold reciprocal_sum
  -- Intermediate steps would go here, but we'll use sorry to denote the proof is omitted.
  sorry

end sum_of_reciprocals_l249_249737


namespace find_radius_of_circle_l249_249857

theorem find_radius_of_circle (C : ℝ) (h : C = 72 * Real.pi) : ∃ r : ℝ, 2 * Real.pi * r = C ∧ r = 36 :=
by
  sorry

end find_radius_of_circle_l249_249857


namespace arithmetic_sequence_general_term_l249_249941

theorem arithmetic_sequence_general_term:
  ∃ (a : ℕ → ℕ), 
    (∀ n, a n + 1 > a n) ∧
    (a 1 = 2) ∧ 
    ((a 2) ^ 2 = a 5 + 6) ∧ 
    (∀ n, a n = 2 * n) :=
by
  sorry

end arithmetic_sequence_general_term_l249_249941


namespace probability_heads_penny_nickel_dime_l249_249112

variable {Ω : Type} [Fintype Ω] [DecidableEq Ω]
variable (coins : Fin 5 → Ω)
variable (heads tails : Ω)

-- Each coin has two outcomes: heads or tails
axiom coin_outcome (i : Fin 5) : coins i = heads ∨ coins i = tails

-- There are 32 total outcomes
axiom total_outcomes : Fintype.card (Fin 5 → Ω) = 32

-- There are 4 successful outcomes for penny, nickel, and dime being heads
axiom successful_outcomes : let successful := {x // (coins 0 = heads) ∧ (coins 1 = heads) ∧ (coins 2 = heads)} in
                          Fintype.card successful = 4

theorem probability_heads_penny_nickel_dime :
  let probability := (Fintype.card {x // (coins 0 = heads) ∧ (coins 1 = heads) ∧ (coins 2 = heads)} : ℤ) /
                     (Fintype.card (Fin 5 → Ω) : ℤ) in
  probability = 1 / 8 :=
by
  sorry

end probability_heads_penny_nickel_dime_l249_249112


namespace parallelepiped_length_l249_249005

theorem parallelepiped_length (n : ℕ)
  (h1 : ∃ n : ℕ, n = 18) 
  (h2 : one_third_of_cubes_have_red_faces : (∃ k : ℕ, k = ((n * (n - 2) * (n - 4)) / 3)) 
        ∧ (remaining_unpainted_cubes : (∃ m : ℕ , m = (2 * (n * (n - 2) * (n - 4)) / 3))))
  (h3 : painted_and_cut_into_cubes : (∃ a b c : ℕ, a = n ∧ b = (n - 2) ∧ c = (n - 4)))
  (h4 : all_sides_whole_cm : (∃ d : ℕ , d = n ∧ d = (n - 2) ∧ d = (n - 4))) :
  n = 18 :=
begin
  sorry
end

end parallelepiped_length_l249_249005


namespace subjects_difference_marius_monica_l249_249814

-- Definitions of given conditions.
def Monica_subjects : ℕ := 10
def Total_subjects : ℕ := 41
def Millie_offset : ℕ := 3

-- Theorem to prove the question == answer given conditions
theorem subjects_difference_marius_monica : 
  ∃ (M : ℕ), (M + (M + Millie_offset) + Monica_subjects = Total_subjects) ∧ (M - Monica_subjects = 4) := 
by
  sorry

end subjects_difference_marius_monica_l249_249814


namespace sin_300_eq_neg_one_half_l249_249534

-- Definitions from the conditions
def point_on_unit_circle (θ : ℝ) : ℝ × ℝ :=
  (Real.cos (θ), Real.sin (θ))

def foot_of_perpendicular (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, 0)

def is_thirty_sixty_ninety_triangle (P : ℝ × ℝ) : Prop :=
  sqrt(P.1^2 + P.2^2) = 1 ∧ P.1 = sqrt(3) / 2 ∧ P.2 = -1/2

-- Problem statement
theorem sin_300_eq_neg_one_half : Real.sin (5 * Real.pi / 3) = -1/2 := by
  sorry

end sin_300_eq_neg_one_half_l249_249534


namespace sin_300_eq_neg_sqrt3_div_2_l249_249613

-- Let Q be the point on the unit circle at 300 degrees counterclockwise from (1,0)
def Q : ℝ × ℝ := (1 / 2, -Real.sqrt 3 / 2)

-- Prove that the sine of 300 degrees is -√3/2
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  -- Here we assume the coordinates given in the conditions are correct.
  have hQ : Q = (1 / 2, -Real.sqrt 3 / 2) := rfl
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249613


namespace radius_ratio_eq_inv_sqrt_5_l249_249393

noncomputable def ratio_of_radii (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) : ℝ :=
  a / b

theorem radius_ratio_eq_inv_sqrt_5 (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) : 
  ratio_of_radii a b h = 1 / Real.sqrt 5 :=
sorry

end radius_ratio_eq_inv_sqrt_5_l249_249393


namespace Batman_game_cost_l249_249138

theorem Batman_game_cost (football_cost strategy_cost total_spent batman_cost : ℝ)
  (h₁ : football_cost = 14.02)
  (h₂ : strategy_cost = 9.46)
  (h₃ : total_spent = 35.52)
  (h₄ : total_spent = football_cost + strategy_cost + batman_cost) :
  batman_cost = 12.04 := by
  sorry

end Batman_game_cost_l249_249138


namespace intersection_a_four_range_of_a_l249_249785

variable {x a : ℝ}

-- Problem 1: Intersection of A and B for a = 4
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - 2*a - 5) < 0}
def B (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a^2 + 2}

theorem intersection_a_four : A 4 ∩ B 4 = {x | 8 < x ∧ x < 13} := 
by  sorry

-- Problem 2: Range of a given condition
theorem range_of_a (a : ℝ) (h1 : a > -3/2) (h2 : ∀ x ∈ A a, x ∈ B a) : 1 ≤ a ∧ a ≤ 3 := 
by  sorry

end intersection_a_four_range_of_a_l249_249785


namespace ratio_y_to_x_l249_249367

theorem ratio_y_to_x (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : y / x = 13 / 2 :=
by
  sorry

end ratio_y_to_x_l249_249367


namespace intersect_A_B_l249_249059

def A : Set ℝ := {x | 1/x < 1}
def B : Set ℝ := {-1, 0, 1, 2}
def intersection_result : Set ℝ := {-1, 2}

theorem intersect_A_B : A ∩ B = intersection_result :=
by
  sorry

end intersect_A_B_l249_249059


namespace arithmetic_sequence_terms_l249_249029

theorem arithmetic_sequence_terms (a d n : ℤ) (last_term : ℤ)
  (h_a : a = 5)
  (h_d : d = 3)
  (h_last_term : last_term = 149)
  (h_n_eq : last_term = a + (n - 1) * d) :
  n = 49 :=
by sorry

end arithmetic_sequence_terms_l249_249029


namespace trapezoid_area_l249_249335

-- Define the conditions for the trapezoid
variables (legs : ℝ) (diagonals : ℝ) (longer_base : ℝ)
variables (h : ℝ) (b : ℝ)
hypothesis (leg_length : legs = 40)
hypothesis (diagonal_length : diagonals = 50)
hypothesis (base_length : longer_base = 60)
hypothesis (altitude : h = 100 / 3)
hypothesis (shorter_base : b = 60 - (40 * (Real.sqrt 11)) / 3)

-- The statement to prove the area of the trapezoid
theorem trapezoid_area : 
  ∃ (A : ℝ), 
  (A = ((b + longer_base) * h) / 2) →
  A = (10000 - 2000 * (Real.sqrt 11)) / 9 :=
by
  -- placeholder for the proof
  sorry

end trapezoid_area_l249_249335


namespace symmetrical_circle_proof_l249_249279

open Real

-- Definition of the original circle equation
def original_circle (x y : ℝ) : Prop :=
  (x + 2)^2 + y^2 = 5

-- Defining the symmetrical circle equation to be proven
def symmetrical_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 5

theorem symmetrical_circle_proof :
  ∀ x y : ℝ, original_circle x y ↔ symmetrical_circle x y :=
by sorry

end symmetrical_circle_proof_l249_249279


namespace det_B_squared_minus_3B_l249_249363

theorem det_B_squared_minus_3B (B : Matrix (Fin 2) (Fin 2) ℝ) (hB : B = ![![2, 4], ![3, 2]]) : 
  Matrix.det (B * B - 3 • B) = 88 := by
  sorry

end det_B_squared_minus_3B_l249_249363


namespace half_angle_in_second_quadrant_l249_249380

theorem half_angle_in_second_quadrant 
  {θ : ℝ} (k : ℤ)
  (hθ_quadrant4 : 2 * k * Real.pi + (3 / 2) * Real.pi ≤ θ ∧ θ ≤ 2 * k * Real.pi + 2 * Real.pi)
  (hcos : abs (Real.cos (θ / 2)) = - Real.cos (θ / 2)) : 
  ∃ m : ℤ, (m * Real.pi + (Real.pi / 2) ≤ θ / 2 ∧ θ / 2 ≤ m * Real.pi + Real.pi) :=
sorry

end half_angle_in_second_quadrant_l249_249380


namespace sqrt_square_eq_self_l249_249731

variable (a : ℝ)

theorem sqrt_square_eq_self (h : a > 0) : Real.sqrt (a ^ 2) = a :=
  sorry

end sqrt_square_eq_self_l249_249731


namespace sin_300_eq_neg_sqrt_3_div_2_l249_249508

theorem sin_300_eq_neg_sqrt_3_div_2 : Real.sin (300 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt_3_div_2_l249_249508


namespace water_volume_per_minute_l249_249873

theorem water_volume_per_minute 
  (depth : ℝ) (width : ℝ) (flow_kmph : ℝ)
  (h_depth : depth = 8) (h_width : width = 25) (h_flow_rate : flow_kmph = 8) :
  (width * depth * (flow_kmph * 1000 / 60)) = 26666.67 :=
by 
  have flow_m_per_min := flow_kmph * 1000 / 60
  have area := width * depth
  have volume_per_minute := area * flow_m_per_min
  sorry

end water_volume_per_minute_l249_249873


namespace pete_flag_total_circle_square_l249_249127

theorem pete_flag_total_circle_square : 
  let stars := 50
  let stripes := 13
  let circles := (stars / 2) - 3
  let squares := (stripes * 2) + 6
  circles + squares = 54 := 
by
  let stars := 50
  let stripes := 13
  let circles := (stars / 2) - 3
  let squares := (stripes * 2) + 6
  show circles + squares = 54
  sorry

end pete_flag_total_circle_square_l249_249127


namespace circle_center_sum_l249_249927

/-- Given the equation of a circle, prove that the sum of the x and y coordinates of the center is -1. -/
theorem circle_center_sum (x y : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = 4 * x - 6 * y + 9) → x + y = -1 :=
by 
  sorry

end circle_center_sum_l249_249927


namespace total_height_correct_l249_249086

def height_of_stairs : ℕ := 10

def num_flights : ℕ := 3

def height_of_all_stairs : ℕ := height_of_stairs * num_flights

def height_of_rope : ℕ := height_of_all_stairs / 2

def extra_height_of_ladder : ℕ := 10

def height_of_ladder : ℕ := height_of_rope + extra_height_of_ladder

def total_height_climbed : ℕ := height_of_all_stairs + height_of_rope + height_of_ladder

theorem total_height_correct : total_height_climbed = 70 := by
  sorry

end total_height_correct_l249_249086


namespace oranges_per_tree_correct_l249_249648

-- Definitions for the conditions
def betty_oranges : ℕ := 15
def bill_oranges : ℕ := 12
def total_oranges := betty_oranges + bill_oranges
def frank_oranges := 3 * total_oranges
def seeds_planted := 2 * frank_oranges
def total_trees := seeds_planted
def total_oranges_picked := 810
def oranges_per_tree := total_oranges_picked / total_trees

-- Theorem statement
theorem oranges_per_tree_correct : oranges_per_tree = 5 :=
by
  -- Proof steps would go here
  sorry

end oranges_per_tree_correct_l249_249648


namespace odd_coefficients_in_polynomial_l249_249811

noncomputable def number_of_odd_coefficients (n : ℕ) : ℕ :=
  (2^n - 1) / 3 * 4 + 1

theorem odd_coefficients_in_polynomial (n : ℕ) (hn : 0 < n) :
  (x^2 + x + 1)^n = number_of_odd_coefficients n :=
sorry

end odd_coefficients_in_polynomial_l249_249811


namespace sin_of_300_degrees_l249_249564

theorem sin_of_300_degrees : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 := by
  sorry

end sin_of_300_degrees_l249_249564


namespace perp_line_eq_l249_249639

theorem perp_line_eq (x y : ℝ) (h1 : (x, y) = (1, 1)) (h2 : y = 2 * x) :
  ∃ a b c : ℝ, a * x + b * y + c = 0 ∧ a = 1 ∧ b = 2 ∧ c = -3 :=
by 
  sorry

end perp_line_eq_l249_249639


namespace binomial_term_is_constant_range_of_a_over_b_l249_249078

noncomputable def binomial_term (a b : ℝ) (m n : ℤ) (r : ℕ) : ℝ :=
  Nat.choose 12 r * a^(12 - r) * b^r

theorem binomial_term_is_constant
  (a b : ℝ)
  (m n : ℤ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: m ≠ 0)
  (h4: n ≠ 0)
  (h5: 2 * m + n = 0) :
  ∃ r, r = 4 ∧
  (binomial_term a b m n r) = 1 :=
sorry

theorem range_of_a_over_b 
  (a b : ℝ)
  (m n : ℤ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: m ≠ 0)
  (h4: n ≠ 0)
  (h5: 2 * m + n = 0) :
  8 / 5 ≤ a / b ∧ a / b ≤ 9 / 4 :=
sorry

end binomial_term_is_constant_range_of_a_over_b_l249_249078


namespace possible_values_of_m_l249_249038

def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def S (x : ℝ) (m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem possible_values_of_m (m : ℝ) : (∀ x, S x m → P x) ↔ (m = -1 ∨ m = 1 ∨ m = 3) :=
by
  sorry

end possible_values_of_m_l249_249038


namespace probability_heads_penny_nickel_dime_is_one_eighth_l249_249123

-- Define the setup: flipping 5 coins
def five_coins : Type := (bool × bool × bool × bool × bool)

-- Define the condition: penny, nickel, and dime are heads
def heads_penny_nickel_dime (c: five_coins) : bool := c.1 && c.2 && c.3

-- Define the successful outcomes where penny, nickel, and dime are heads
noncomputable def successful_outcomes : Set five_coins :=
  { c | heads_penny_nickel_dime c = tt }

-- Define the total outcomes for 5 coins
noncomputable def total_outcomes : Set five_coins := {c | true}

-- Probability calculation: |success| / |total|
noncomputable def probability_heads_penny_nickel_dime : real :=
  (Set.card successful_outcomes : real) / (Set.card total_outcomes : real)

-- Given the setup, prove that the probability is 1/8
theorem probability_heads_penny_nickel_dime_is_one_eighth :
  probability_heads_penny_nickel_dime = 1 / 8 := 
sorry

end probability_heads_penny_nickel_dime_is_one_eighth_l249_249123


namespace cost_of_pumpkin_seeds_l249_249868

theorem cost_of_pumpkin_seeds (P : ℝ)
    (h1 : ∃(P_tomato P_chili : ℝ), P_tomato = 1.5 ∧ P_chili = 0.9) 
    (h2 : 3 * P + 4 * 1.5 + 5 * 0.9 = 18) 
    : P = 2.5 :=
by sorry

end cost_of_pumpkin_seeds_l249_249868


namespace rice_mixing_ratio_l249_249697

theorem rice_mixing_ratio (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (4.5 * x + 8.75 * y) / (x + y) = 7.5 → y / x = 2.4 :=
by
  sorry

end rice_mixing_ratio_l249_249697


namespace average_production_per_day_for_entire_month_l249_249795

-- Definitions based on the conditions
def average_first_25_days := 65
def average_last_5_days := 35
def number_of_days_in_first_period := 25
def number_of_days_in_last_period := 5
def total_days_in_month := 30

-- The goal is to prove that the average production per day for the entire month is 60 TVs/day.
theorem average_production_per_day_for_entire_month :
  (average_first_25_days * number_of_days_in_first_period + 
   average_last_5_days * number_of_days_in_last_period) / total_days_in_month = 60 := 
by
  sorry

end average_production_per_day_for_entire_month_l249_249795


namespace sin_300_eq_neg_sqrt3_div_2_l249_249529

def point_on_unit_circle (θ : ℝ) : ℝ × ℝ :=
⟨Real.cos θ, Real.sin θ⟩

theorem sin_300_eq_neg_sqrt3_div_2 :
  let Q := point_on_unit_circle (300 * Real.pi / 180) in
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
by sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249529


namespace total_area_of_plots_l249_249291

theorem total_area_of_plots (n : ℕ) (side_length : ℕ) (area_one_plot : ℕ) (total_plots : ℕ) (total_area : ℕ)
  (h1 : n = 9)
  (h2 : side_length = 6)
  (h3 : area_one_plot = side_length * side_length)
  (h4 : total_plots = n)
  (h5 : total_area = area_one_plot * total_plots) :
  total_area = 324 := 
by
  sorry

end total_area_of_plots_l249_249291


namespace eggs_per_hen_l249_249768

theorem eggs_per_hen (total_eggs : Float) (num_hens : Float) (h1 : total_eggs = 303.0) (h2 : num_hens = 28.0) : 
  total_eggs / num_hens = 10.821428571428571 :=
by 
  sorry

end eggs_per_hen_l249_249768


namespace total_candies_in_store_l249_249741

-- Define the quantities of chocolates in each box
def box_chocolates_1 := 200
def box_chocolates_2 := 320
def box_chocolates_3 := 500
def box_chocolates_4 := 500
def box_chocolates_5 := 768
def box_chocolates_6 := 768

-- Define the quantities of candies in each tub
def tub_candies_1 := 1380
def tub_candies_2 := 1150
def tub_candies_3 := 1150
def tub_candies_4 := 1720

-- Sum of all chocolates and candies
def total_chocolates := box_chocolates_1 + box_chocolates_2 + box_chocolates_3 + box_chocolates_4 + box_chocolates_5 + box_chocolates_6
def total_candies := tub_candies_1 + tub_candies_2 + tub_candies_3 + tub_candies_4
def total_store_candies := total_chocolates + total_candies

theorem total_candies_in_store : total_store_candies = 8456 := by
  sorry

end total_candies_in_store_l249_249741


namespace divisor_is_five_l249_249422

theorem divisor_is_five (n d : ℕ) (h1 : ∃ k, n = k * d + 3) (h2 : ∃ l, n^2 = l * d + 4) : d = 5 :=
sorry

end divisor_is_five_l249_249422


namespace parallelepiped_length_l249_249010

theorem parallelepiped_length :
  ∃ n : ℕ, (n ≥ 7) ∧ (n * (n - 2) * (n - 4) = 3 * ((n - 2) * (n - 4) * (n - 6))) ∧ n = 18 :=
by
  sorry

end parallelepiped_length_l249_249010


namespace expression_value_l249_249733

/-- The value of the expression 1.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) is 1200. -/
theorem expression_value : 
  1.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 1200 :=
by
  sorry

end expression_value_l249_249733


namespace terminating_decimals_count_l249_249934

noncomputable def int_counts_terminating_decimals : ℕ :=
  let n_limit := 500
  let denominator := 2100
  Nat.floor (n_limit / 21)

theorem terminating_decimals_count :
  int_counts_terminating_decimals = 23 :=
by
  /- Proof will be here eventually -/
  sorry

end terminating_decimals_count_l249_249934


namespace find_higher_selling_price_l249_249015

def cost_price := 200
def selling_price_low := 340
def gain_low := selling_price_low - cost_price
def gain_high := gain_low + (5 / 100) * gain_low
def higher_selling_price := cost_price + gain_high

theorem find_higher_selling_price : higher_selling_price = 347 := 
by 
  sorry

end find_higher_selling_price_l249_249015


namespace how_many_candies_eaten_l249_249890

variable (candies_tuesday candies_thursday candies_friday candies_left : ℕ)

def total_candies (candies_tuesday candies_thursday candies_friday : ℕ) : ℕ :=
  candies_tuesday + candies_thursday + candies_friday

theorem how_many_candies_eaten (h_tuesday : candies_tuesday = 3)
                               (h_thursday : candies_thursday = 5)
                               (h_friday : candies_friday = 2)
                               (h_left : candies_left = 4) :
  (total_candies candies_tuesday candies_thursday candies_friday) - candies_left = 6 :=
by
  sorry

end how_many_candies_eaten_l249_249890


namespace kids_played_on_monday_l249_249398

theorem kids_played_on_monday (total : ℕ) (tuesday : ℕ) (monday : ℕ) (h_total : total = 16) (h_tuesday : tuesday = 14) :
  monday = 2 :=
by
  -- Placeholder for the actual proof
  sorry

end kids_played_on_monday_l249_249398


namespace find_m_of_cos_alpha_l249_249686

theorem find_m_of_cos_alpha (m : ℝ) (h₁ : (2 * Real.sqrt 5) / 5 = m / Real.sqrt (m ^ 2 + 1)) (h₂ : m > 0) : m = 2 :=
sorry

end find_m_of_cos_alpha_l249_249686


namespace sin_300_deg_l249_249487

noncomputable def sin_300_equals_neg_sqrt3_div_2 : Prop :=
  sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2

theorem sin_300_deg : sin_300_equals_neg_sqrt3_div_2 :=
by
  sorry

end sin_300_deg_l249_249487


namespace circle_area_l249_249681

theorem circle_area (r : ℝ) (h : 8 * (1 / (2 * π * r)) = (2 * r) ^ 2) : π * r ^ 2 = π ^ (1 / 3) :=
by
  sorry

end circle_area_l249_249681


namespace identify_smart_person_l249_249017

theorem identify_smart_person (F S : ℕ) (h_total : F + S = 30) (h_max_fools : F ≤ 8) : S ≥ 1 :=
by {
  sorry
}

end identify_smart_person_l249_249017


namespace divisors_large_than_8_fact_count_l249_249376

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem divisors_large_than_8_fact_count :
  let n := 9
  let factorial_n := factorial n
  let factorial_n_minus_1 := factorial (n - 1)
  ∃ (num_divisors : ℕ), num_divisors = 8 ∧
    (∀ d, d ∣ factorial_n → d > factorial_n_minus_1 ↔ ∃ k, k ∣ factorial_n ∧ k < 9) :=
by
  sorry

end divisors_large_than_8_fact_count_l249_249376


namespace min_value_polynomial_l249_249812

open Real

theorem min_value_polynomial (x y z : ℝ) (h_posx : 0 < x) (h_posy : 0 < y) (h_posz : 0 < z) (h : x * y * z = 3) :
  x^2 + 4 * x * y + 12 * y^2 + 8 * y * z + 3 * z^2 ≥ 162 := 
sorry

end min_value_polynomial_l249_249812


namespace range_of_b_l249_249211

theorem range_of_b (a b x : ℝ) (ha : 0 < a ∧ a ≤ 5 / 4) (hb : 0 < b) :
  (∀ x, |x - a| < b → |x - a^2| < 1 / 2) ↔ 0 < b ∧ b ≤ 3 / 16 :=
by
  sorry

end range_of_b_l249_249211


namespace find_m_l249_249317

theorem find_m (x1 x2 m : ℝ) (h1 : 2 * x1^2 - 3 * x1 + m = 0) (h2 : 2 * x2^2 - 3 * x2 + m = 0) (h3 : 8 * x1 - 2 * x2 = 7) :
  m = 1 :=
sorry

end find_m_l249_249317


namespace correct_operation_l249_249305

theorem correct_operation (a : ℝ) : (-a^3)^4 = a^12 :=
by sorry

end correct_operation_l249_249305


namespace sin_300_eq_neg_sqrt_3_div_2_l249_249506

theorem sin_300_eq_neg_sqrt_3_div_2 : Real.sin (300 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt_3_div_2_l249_249506


namespace sum_pow_congruent_zero_mod_m_l249_249810

theorem sum_pow_congruent_zero_mod_m
  {a : ℕ → ℤ} {x : ℕ → ℤ} (n r : ℕ) 
  (hn : n ≥ 2) (hr : r ≥ 2)
  (h0 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ r → ∑ j in Finset.range (n+1), a j * x j ^ k = 0) :
  ∀ m : ℕ, r+1 ≤ m ∧ m ≤ 2*r+1 → ∑ j in Finset.range (n+1), a j * x j ^ m ≡ 0 [MOD m] :=
by
  sorry

end sum_pow_congruent_zero_mod_m_l249_249810


namespace min_birthdays_on_wednesday_l249_249486

theorem min_birthdays_on_wednesday 
  (W X : ℕ) 
  (h1 : W + 6 * X = 50) 
  (h2 : W > X) : 
  W = 8 := 
sorry

end min_birthdays_on_wednesday_l249_249486


namespace complement_of_A_in_U_l249_249216

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the set A
def A : Set ℤ := {x | x ∈ Set.univ ∧ x^2 + x - 2 < 0}

-- State the theorem about the complement of A in U
theorem complement_of_A_in_U :
  (U \ A) = {-2, 1, 2} :=
sorry

end complement_of_A_in_U_l249_249216


namespace sin_300_eq_neg_sqrt3_div_2_l249_249584

noncomputable def sin_300 : ℝ := sin (300 * real.pi / 180)

theorem sin_300_eq_neg_sqrt3_div_2 : 
  (∠300° = 360° - 60°) →
  (300° ∈ fourth_quadrant) →
  (∀ θ ∈ fourth_quadrant, sin θ < 0) →
  (sin (60 * real.pi / 180) = sqrt 3 / 2) →
  sin_300 = -sqrt 3 / 2 := 
by
  intros h1 h2 h3 h4
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249584


namespace ring_binder_price_l249_249090

theorem ring_binder_price (x : ℝ) (h1 : 50 + 5 = 55) (h2 : ∀ x, 55 + 3 * (x - 2) = 109) :
  x = 20 :=
by
  sorry

end ring_binder_price_l249_249090


namespace valid_numbers_l249_249091

-- Define the conditions for three-digit numbers
def isThreeDigitNumber (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000

-- Define the splitting cases and the required property
def satisfiesFirstCase (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a ≠ 0 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧
  n = 100 * a + 10 * b + c ∧
  3 * ((10 * a + b) * c) = n

def satisfiesSecondCase (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a ≠ 0 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧
  n = 100 * a + 10 * b + c ∧
  3 * (a * (10 * b + c)) = n

-- Define the main proposition
def validThreeDigitNumber (n : ℕ) : Prop :=
  isThreeDigitNumber n ∧ (satisfiesFirstCase n ∨ satisfiesSecondCase n)

-- The theorem statement which we need to prove
theorem valid_numbers : ∀ n : ℕ, validThreeDigitNumber n ↔ n = 150 ∨ n = 240 ∨ n = 735 :=
by
  sorry

end valid_numbers_l249_249091


namespace clever_value_points_l249_249655

def clever_value_point (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f x₀ = (deriv f) x₀

theorem clever_value_points :
  (clever_value_point (fun x : ℝ => x^2)) ∧
  (clever_value_point (fun x : ℝ => Real.log x)) ∧
  (clever_value_point (fun x : ℝ => x + (1 / x))) :=
by
  -- Proof omitted
  sorry

end clever_value_points_l249_249655


namespace final_probability_l249_249177

-- Define the structure of the problem
structure GameRound :=
  (green_ball : ℕ)
  (red_ball : ℕ)
  (blue_ball : ℕ)
  (white_ball : ℕ)

structure GameState :=
  (coins : ℕ)
  (players : ℕ)

-- Define the game rules and initial conditions
noncomputable def initial_coins := 5
noncomputable def rounds := 5

-- Probability-related functions and game logic
noncomputable def favorable_outcome_count : ℕ := 6
noncomputable def total_outcomes_per_round : ℕ := 120
noncomputable def probability_per_round : ℚ := favorable_outcome_count / total_outcomes_per_round

theorem final_probability :
  probability_per_round ^ rounds = 1 / 3200000 :=
by
  sorry

end final_probability_l249_249177


namespace option_C_sets_same_l249_249479

-- Define the sets for each option
def option_A_set_M : Set (ℕ × ℕ) := {(3, 2)}
def option_A_set_N : Set (ℕ × ℕ) := {(2, 3)}

def option_B_set_M : Set (ℕ × ℕ) := {p | p.1 + p.2 = 1}
def option_B_set_N : Set ℕ := { y | ∃ x, x + y = 1 }

def option_C_set_M : Set ℕ := {4, 5}
def option_C_set_N : Set ℕ := {5, 4}

def option_D_set_M : Set ℕ := {1, 2}
def option_D_set_N : Set (ℕ × ℕ) := {(1, 2)}

-- Prove that option C sets represent the same set
theorem option_C_sets_same : option_C_set_M = option_C_set_N := by
  sorry

end option_C_sets_same_l249_249479


namespace solve_x_l249_249628

theorem solve_x 
  (x : ℝ) 
  (h : (2 / x) + (3 / x) / (6 / x) = 1.25) : 
  x = 8 / 3 := 
sorry

end solve_x_l249_249628


namespace furniture_cost_final_price_l249_249472

theorem furniture_cost_final_price 
  (table_cost : ℤ := 140)
  (chair_ratio : ℚ := 1/7)
  (sofa_ratio : ℕ := 2)
  (discount : ℚ := 0.10)
  (tax : ℚ := 0.07)
  (exchange_rate : ℚ := 1.2) :
  let chair_cost := table_cost * chair_ratio
  let sofa_cost := table_cost * sofa_ratio
  let total_cost_before_discount := table_cost + 4 * chair_cost + sofa_cost
  let table_discount := discount * table_cost
  let discounted_table_cost := table_cost - table_discount
  let total_cost_after_discount := discounted_table_cost + 4 * chair_cost + sofa_cost
  let sales_tax := tax * total_cost_after_discount
  let final_cost := total_cost_after_discount + sales_tax
  final_cost = 520.02 
:= sorry

end furniture_cost_final_price_l249_249472


namespace kids_stay_home_correct_l249_249030

def total_number_of_kids : ℕ := 1363293
def kids_who_go_to_camp : ℕ := 455682
def kids_staying_home : ℕ := total_number_of_kids - kids_who_go_to_camp

theorem kids_stay_home_correct :
  kids_staying_home = 907611 := by 
  sorry

end kids_stay_home_correct_l249_249030


namespace ribbon_cost_l249_249354

variable (c_g c_m s : ℝ)

theorem ribbon_cost (h1 : 5 * c_g + s = 295) (h2 : 7 * c_m + s = 295) (h3 : 2 * c_m + c_g = 102) : s = 85 :=
sorry

end ribbon_cost_l249_249354


namespace num_customers_after_family_l249_249876

-- Definitions
def soft_taco_price : ℕ := 2
def hard_taco_price : ℕ := 5
def family_hard_tacos : ℕ := 4
def family_soft_tacos : ℕ := 3
def total_income : ℕ := 66

-- Intermediate values which can be derived
def family_cost : ℕ := (family_hard_tacos * hard_taco_price) + (family_soft_tacos * soft_taco_price)
def remaining_income : ℕ := total_income - family_cost

-- Proposition: Number of customers after the family
def customers_after_family : ℕ := remaining_income / (2 * soft_taco_price)

-- Theorem to prove the number of customers is 10
theorem num_customers_after_family : customers_after_family = 10 := by
  sorry

end num_customers_after_family_l249_249876


namespace range_of_values_l249_249793

variable (a : ℝ)

-- State the conditions
def prop.false (a : ℝ) : Prop := ¬ ∃ x : ℝ, a * x^2 + 4 * x + a ≤ 0

-- Prove that the range of values for a where the proposition is false is (2, +∞)
theorem range_of_values (ha : prop.false a) : 2 < a :=
sorry

end range_of_values_l249_249793


namespace x_divisible_by_5_l249_249358

theorem x_divisible_by_5 (x y : ℕ) (hx : x > 1) (h : 2 * x^2 - 1 = y^15) : 5 ∣ x := 
sorry

end x_divisible_by_5_l249_249358


namespace poly_coefficients_sum_l249_249066

theorem poly_coefficients_sum :
  ∀ (x A B C D : ℝ),
  (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D →
  A + B + C + D = 2 :=
by sorry

end poly_coefficients_sum_l249_249066


namespace right_angled_triangle_ratio_3_4_5_l249_249307

theorem right_angled_triangle_ratio_3_4_5 : 
  ∀ (a b c : ℕ), 
  (a = 3 * d) → (b = 4 * d) → (c = 5 * d) → (a^2 + b^2 = c^2) :=
by
  intros a b c h1 h2 h3
  sorry

end right_angled_triangle_ratio_3_4_5_l249_249307


namespace problem1_problem2_l249_249672

-- Define the quadratic equation and condition for real roots
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- Problem 1
theorem problem1 (m : ℝ) : ((m - 2) * (m - 2) * (m - 2) + 2 * 2 * (2 - m) * 2 * (-1) ≥ 0) → (m ≤ 3 ∧ m ≠ 2) := sorry

-- Problem 2
theorem problem2 (m : ℝ) : 
  (∀ x, (x = 1 ∨ x = 2) → (m - 2) * x^2 + 2 * x + 1 = 0) → (-1 ≤ m ∧ m < (3 / 4)) := 
sorry

end problem1_problem2_l249_249672


namespace parallelepiped_length_l249_249011

theorem parallelepiped_length :
  ∃ n : ℕ, (n ≥ 7) ∧ (n * (n - 2) * (n - 4) = 3 * ((n - 2) * (n - 4) * (n - 6))) ∧ n = 18 :=
by
  sorry

end parallelepiped_length_l249_249011


namespace total_cost_of_purchase_l249_249429

theorem total_cost_of_purchase :
  let sandwich_cost := 3
  let soda_cost := 2
  let num_sandwiches := 5
  let num_sodas := 8
  let total_cost := (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost)
  total_cost = 31 :=
by
  sorry

end total_cost_of_purchase_l249_249429


namespace saturated_function_2014_l249_249805

def saturated (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, f^[f^[f n] n] n = n

theorem saturated_function_2014 (f : ℕ → ℕ) (m : ℕ) (h : saturated f) :
  (m ∣ 2014) ↔ (f^[2014] m = m) :=
sorry

end saturated_function_2014_l249_249805


namespace avg_height_country_l249_249846

-- Define the parameters for the number of boys and their average heights
def num_boys_north : ℕ := 300
def num_boys_south : ℕ := 200
def avg_height_north : ℝ := 1.60
def avg_height_south : ℝ := 1.50

-- Define the total number of boys
def total_boys : ℕ := num_boys_north + num_boys_south

-- Define the total combined height
def total_height : ℝ := (num_boys_north * avg_height_north) + (num_boys_south * avg_height_south)

-- Prove that the average height of all boys combined is 1.56 meters
theorem avg_height_country : total_height / total_boys = 1.56 := by
  sorry

end avg_height_country_l249_249846


namespace number_of_students_in_club_l249_249971

variable (y : ℕ) -- Number of girls

def total_stickers_given (y : ℕ) : ℕ := y * y + (y + 3) * (y + 3)

theorem number_of_students_in_club :
  (total_stickers_given y = 640) → (2 * y + 3 = 35) := 
by
  intro h1
  sorry

end number_of_students_in_club_l249_249971


namespace number_of_possible_tower_heights_l249_249925

-- Axiom for the possible increment values when switching brick orientations
def possible_increments : Set ℕ := {4, 7}

-- Base height when all bricks contribute the smallest dimension
def base_height (num_bricks : ℕ) (smallest_side : ℕ) : ℕ :=
  num_bricks * smallest_side

-- Check if a given height can be achieved by changing orientations of the bricks
def can_achieve_height (h : ℕ) (n : ℕ) (increments : Set ℕ) : Prop :=
  ∃ m k : ℕ, h = base_height n 2 + m * 4 + k * 7

-- Final proof statement
theorem number_of_possible_tower_heights :
  (50 : ℕ) = 50 →
  (∀ k : ℕ, (100 + k * 4 <= 450) → can_achieve_height (100 + k * 4) 50 possible_increments) →
  ∃ (num_possible_heights : ℕ), num_possible_heights = 90 :=
by
  sorry

end number_of_possible_tower_heights_l249_249925


namespace more_valley_than_humpy_l249_249298

def is_humpy (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  n >= 10000 ∧ n < 100000 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 > d4 ∧ d4 > d5

def is_valley (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  n >= 10000 ∧ n < 100000 ∧ d1 > d2 ∧ d2 > d3 ∧ d3 < d4 ∧ d4 < d5

def starts_with_5 (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  d1 = 5

theorem more_valley_than_humpy :
  (∃ m, starts_with_5 m ∧ is_humpy m) → (∃ n, starts_with_5 n ∧ is_valley n) ∧ 
  (∀ x, starts_with_5 x → is_humpy x → ∃ y, starts_with_5 y ∧ is_valley y ∧ y ≠ x) :=
by sorry

end more_valley_than_humpy_l249_249298


namespace polynomial_roots_l249_249696

theorem polynomial_roots (p q BD DC : ℝ) (h_sum : BD + DC = p) (h_prod : BD * DC = q^2) :
    Polynomial.roots (Polynomial.C 1 * Polynomial.X^2 - Polynomial.C p * Polynomial.X + Polynomial.C (q^2)) = {BD, DC} :=
sorry

end polynomial_roots_l249_249696


namespace susan_ate_6_candies_l249_249884

-- Definitions for the conditions
def candies_tuesday : ℕ := 3
def candies_thursday : ℕ := 5
def candies_friday : ℕ := 2
def candies_left : ℕ := 4

-- The total number of candies bought during the week
def total_candies_bought : ℕ := candies_tuesday + candies_thursday + candies_friday

-- The number of candies Susan ate during the week
def candies_eaten : ℕ := total_candies_bought - candies_left

-- Theorem statement
theorem susan_ate_6_candies : candies_eaten = 6 :=
by
  unfold candies_eaten total_candies_bought candies_tuesday candies_thursday candies_friday candies_left
  sorry

end susan_ate_6_candies_l249_249884


namespace wooden_parallelepiped_length_l249_249001

theorem wooden_parallelepiped_length (n : ℕ) (h1 : n ≥ 7)
    (h2 : ∀ total_cubes unpainted_cubes : ℕ,
      total_cubes = n * (n - 2) * (n - 4) ∧
      unpainted_cubes = (n - 2) * (n - 4) * (n - 6) ∧
      unpainted_cubes = 2 / 3 * total_cubes) :
  n = 18 := 
sorry

end wooden_parallelepiped_length_l249_249001


namespace value_of_M_l249_249714

theorem value_of_M (x y z M : ℝ) (h1 : x + y + z = 90)
    (h2 : x - 5 = M)
    (h3 : y + 5 = M)
    (h4 : 5 * z = M) :
    M = 450 / 11 :=
by
    sorry

end value_of_M_l249_249714


namespace find_room_width_l249_249824

def room_height : ℕ := 12
def room_length : ℕ := 25
def door_height : ℕ := 6
def door_width : ℕ := 3
def window_height : ℕ := 4
def window_width : ℕ := 3
def number_of_windows : ℕ := 3
def cost_per_sqft : ℕ := 8
def total_cost : ℕ := 7248

theorem find_room_width (x : ℕ) (h : 8 * (room_height * (2 * room_length + 2 * x) - (door_height * door_width + window_height * window_width * number_of_windows)) = total_cost) : 
  x = 15 :=
sorry

end find_room_width_l249_249824


namespace repeating_decimal_fraction_l249_249348

noncomputable def x : ℚ := 75 / 99  -- 0.\overline{75}
noncomputable def y : ℚ := 223 / 99  -- 2.\overline{25}

theorem repeating_decimal_fraction : (x / y) = 2475 / 7329 :=
by
  -- Further proof details can be added here
  sorry

end repeating_decimal_fraction_l249_249348


namespace find_higher_selling_price_l249_249016

def cost_price := 200
def selling_price_low := 340
def gain_low := selling_price_low - cost_price
def gain_high := gain_low + (5 / 100) * gain_low
def higher_selling_price := cost_price + gain_high

theorem find_higher_selling_price : higher_selling_price = 347 := 
by 
  sorry

end find_higher_selling_price_l249_249016


namespace sum_of_reciprocals_l249_249136

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) : 
  1 / x + 1 / y = 3 :=
by
  sorry

end sum_of_reciprocals_l249_249136


namespace find_sticker_price_l249_249699

-- Define the conditions
def storeX_discount (x : ℝ) : ℝ := 0.80 * x - 70
def storeY_discount (x : ℝ) : ℝ := 0.70 * x

-- Define the main statement
theorem find_sticker_price (x : ℝ) (h : storeX_discount x = storeY_discount x - 20) : x = 500 :=
sorry

end find_sticker_price_l249_249699


namespace bruce_paid_amount_l249_249484

noncomputable def total_amount_paid :=
  let grapes_cost := 8 * 70
  let mangoes_cost := 9 * 55
  let oranges_cost := 5 * 40
  let strawberries_cost := 4 * 90
  let total_cost := grapes_cost + mangoes_cost + oranges_cost + strawberries_cost
  let discount := 0.10 * total_cost
  let discounted_total := total_cost - discount
  let tax := 0.05 * discounted_total
  let final_amount := discounted_total + tax
  final_amount

theorem bruce_paid_amount :
  total_amount_paid = 1526.18 :=
by
  sorry

end bruce_paid_amount_l249_249484


namespace base_5_to_base_10_l249_249322

theorem base_5_to_base_10 : 
  let n : ℕ := 1 * 5^3 + 2 * 5^2 + 3 * 5^1 + 4 * 5^0
  n = 194 :=
by 
  sorry

end base_5_to_base_10_l249_249322


namespace find_deducted_salary_l249_249262

noncomputable def dailyWage (weeklySalary : ℝ) (workingDays : ℕ) : ℝ := weeklySalary / workingDays

noncomputable def totalDeduction (dailyWage : ℝ) (absentDays : ℕ) : ℝ := dailyWage * absentDays

noncomputable def deductedSalary (weeklySalary : ℝ) (totalDeduction : ℝ) : ℝ := weeklySalary - totalDeduction

theorem find_deducted_salary
  (weeklySalary : ℝ := 791)
  (workingDays : ℕ := 5)
  (absentDays : ℕ := 4)
  (dW := dailyWage weeklySalary workingDays)
  (tD := totalDeduction dW absentDays)
  (dS := deductedSalary weeklySalary tD) :
  dS = 158.20 := 
  by
    sorry

end find_deducted_salary_l249_249262


namespace matt_without_calculator_5_minutes_l249_249249

-- Define the conditions
def time_with_calculator (problems : Nat) : Nat := 2 * problems
def time_without_calculator (problems : Nat) (x : Nat) : Nat := x * problems
def time_saved (problems : Nat) (x : Nat) : Nat := time_without_calculator problems x - time_with_calculator problems

-- State the problem
theorem matt_without_calculator_5_minutes (x : Nat) :
  (time_saved 20 x = 60) → x = 5 := by
  sorry

end matt_without_calculator_5_minutes_l249_249249


namespace number_of_different_pairs_l249_249678

theorem number_of_different_pairs :
  let mystery := 4
  let fantasy := 4
  let science_fiction := 4
  (mystery * fantasy) + (mystery * science_fiction) + (fantasy * science_fiction) = 48 :=
by
  let mystery := 4
  let fantasy := 4
  let science_fiction := 4
  show (mystery * fantasy) + (mystery * science_fiction) + (fantasy * science_fiction) = 48
  sorry

end number_of_different_pairs_l249_249678


namespace initial_number_of_people_l249_249024

theorem initial_number_of_people (X : ℕ) (h : ((X - 10) + 15 = 17)) : X = 12 :=
by
  sorry

end initial_number_of_people_l249_249024


namespace factor_expression_l249_249898

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249898


namespace sin_300_eq_neg_sqrt3_div_2_l249_249571

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249571


namespace relationship_among_abc_l249_249040

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := Real.exp (0.3 * Real.log 2)
noncomputable def c : ℝ := Real.exp (0.2 * Real.log 0.3)

theorem relationship_among_abc :
  b > c ∧ c > a :=
by
  sorry

end relationship_among_abc_l249_249040


namespace ceil_neg_seven_fourths_cubed_eq_neg_five_l249_249199

noncomputable def ceil_of_neg_seven_fourths_cubed : ℤ :=
  Int.ceil ((-7 / 4 : ℚ)^3)

theorem ceil_neg_seven_fourths_cubed_eq_neg_five :
  ceil_of_neg_seven_fourths_cubed = -5 := by
  sorry

end ceil_neg_seven_fourths_cubed_eq_neg_five_l249_249199


namespace area_of_cos_closed_figure_l249_249275

theorem area_of_cos_closed_figure :
  ∫ x in (Real.pi / 2)..(3 * Real.pi / 2), Real.cos x = 2 :=
by
  sorry

end area_of_cos_closed_figure_l249_249275


namespace cars_with_both_features_l249_249264

theorem cars_with_both_features (T P_s P_w N B : ℕ)
  (hT : T = 65) 
  (hPs : P_s = 45) 
  (hPw : P_w = 25) 
  (hN : N = 12) 
  (h_equation : P_s + P_w - B + N = T) :
  B = 17 :=
by
  sorry

end cars_with_both_features_l249_249264


namespace find_daily_rate_of_first_company_l249_249469

-- Define the daily rate of the first car rental company
def daily_rate_first_company (x : ℝ) : ℝ :=
  x + 0.18 * 48.0

-- Define the total cost for City Rentals
def total_cost_city_rentals : ℝ :=
  18.95 + 0.16 * 48.0

-- Prove the daily rate of the first car rental company
theorem find_daily_rate_of_first_company (x : ℝ) (h : daily_rate_first_company x = total_cost_city_rentals) : 
  x = 17.99 := 
by
  sorry

end find_daily_rate_of_first_company_l249_249469


namespace optimal_production_distribution_l249_249458

noncomputable def min_production_time (unitsI_A unitsI_B unitsII_B : ℕ) : ℕ :=
let rateI_A := 30
let rateII_B := 40
let rateI_B := 50
let initial_days_B := 20
let remaining_units_I := 1500 - (rateI_A * initial_days_B)
let combined_rateI_AB := rateI_A + rateI_B
let days_remaining_I := remaining_units_I / combined_rateI_AB
initial_days_B + days_remaining_I

theorem optimal_production_distribution :
  ∃ (unitsI_A unitsI_B unitsII_B : ℕ),
    unitsI_A + unitsI_B = 1500 ∧ unitsII_B = 800 ∧
    min_production_time unitsI_A unitsI_B unitsII_B = 31 := sorry

end optimal_production_distribution_l249_249458


namespace cos_double_angle_nonpositive_l249_249777

theorem cos_double_angle_nonpositive (α β : ℝ) (φ : ℝ) 
  (h : Real.tan φ = 1 / (Real.cos α * Real.cos β + Real.tan α * Real.tan β)) : 
  Real.cos (2 * φ) ≤ 0 := 
sorry

end cos_double_angle_nonpositive_l249_249777


namespace projection_of_a_onto_b_eq_neg_sqrt_2_l249_249935

noncomputable def projection (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dot_product / magnitude_b

theorem projection_of_a_onto_b_eq_neg_sqrt_2 :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (-1, 1)
  projection a b = -Real.sqrt 2 :=
by
  sorry

end projection_of_a_onto_b_eq_neg_sqrt_2_l249_249935


namespace students_juice_count_l249_249338

theorem students_juice_count (students chose_water chose_juice : ℕ) 
  (h1 : chose_water = 140) 
  (h2 : (25 : ℚ) / 100 * (students : ℚ) = chose_juice)
  (h3 : (70 : ℚ) / 100 * (students : ℚ) = chose_water) : 
  chose_juice = 50 :=
by 
  sorry

end students_juice_count_l249_249338


namespace yolk_count_proof_l249_249749

-- Define the conditions of the problem
def eggs_in_carton : ℕ := 12
def double_yolk_eggs : ℕ := 5
def single_yolk_eggs : ℕ := eggs_in_carton - double_yolk_eggs
def yolks_in_double_yolk_eggs : ℕ := double_yolk_eggs * 2
def yolks_in_single_yolk_eggs : ℕ := single_yolk_eggs
def total_yolks : ℕ := yolks_in_single_yolk_eggs + yolks_in_double_yolk_eggs

-- Stating the theorem to prove the total number of yolks is 17
theorem yolk_count_proof : total_yolks = 17 := 
by
  sorry

end yolk_count_proof_l249_249749


namespace problem_min_value_problem_inequality_range_l249_249946

theorem problem_min_value (a b : ℝ) (h : a + b = 1) (ha : a > 0) (hb : b > 0) :
  (1 / a + 4 / b) ≥ 9 :=
sorry

theorem problem_inequality_range (a b : ℝ) (h : a + b = 1) (ha : a > 0) (hb : b > 0) (x : ℝ) :
  (1 / a + 4 / b) ≥ |2 * x - 1| - |x + 1| ↔ -7 ≤ x ∧ x ≤ 11 :=
sorry

end problem_min_value_problem_inequality_range_l249_249946


namespace problem1_correct_problem2_correct_l249_249187

-- Definition for Problem 1
def problem1 (a b c d : ℚ) : ℚ :=
  (a - b + c) * d

-- Statement for Problem 1
theorem problem1_correct : problem1 (1/6) (5/7) (2/3) (-42) = -5 :=
by
  sorry

-- Definitions for Problem 2
def problem2 (a b c d : ℚ) : ℚ :=
  (-a^2 + b^2 * c - d^2 / |d|)

-- Statement for Problem 2
theorem problem2_correct : problem2 (-2) (-3) (-2/3) 4 = -14 :=
by
  sorry

end problem1_correct_problem2_correct_l249_249187


namespace total_height_correct_l249_249087

def height_of_stairs : ℕ := 10

def num_flights : ℕ := 3

def height_of_all_stairs : ℕ := height_of_stairs * num_flights

def height_of_rope : ℕ := height_of_all_stairs / 2

def extra_height_of_ladder : ℕ := 10

def height_of_ladder : ℕ := height_of_rope + extra_height_of_ladder

def total_height_climbed : ℕ := height_of_all_stairs + height_of_rope + height_of_ladder

theorem total_height_correct : total_height_climbed = 70 := by
  sorry

end total_height_correct_l249_249087


namespace trajectory_of_A_fixed_point_MN_l249_249796

-- Definitions from conditions
def vertexB : ℝ × ℝ := (0, 1)
def vertexC : ℝ × ℝ := (0, 1)
def PA (A P : ℝ × ℝ) := let (a_x, a_y) := A in let (p_x, p_y) := P in ((a_x - p_x), (a_y - p_y))
def PB (P : ℝ × ℝ) := let (p_x, p_y) := P in ((-p_x), (1 - p_y))
def PC (P : ℝ × ℝ) := let (p_x, p_y) := P in ((-p_x), (1 - p_y))

def condition1 (A P : ℝ × ℝ) := PA(A,P) + PB(P) + PC(P) = (0,0)
def condition2 (Q : ℝ × ℝ) (A B C : ℝ × ℝ) := let (q_x, q_y) := Q in (q_x - B.1)^2 + q_y^2 = (q_x - C.1)^2 + q_y^2 /\
                                                (q_x - A.1)^2 + q_y^2 = (q_x - B.1)^2 + q_y^2 /\
                                                norm (Q - A) = norm (Q - B)
def condition3 (P Q B C : ℝ × ℝ) := let (p_x, p_y) := P in let (q_x, q_y) := Q in let (b_x, b_y) := B in let (c_x, c_y) := C in 
                           (q_x - p_x) / (q_y - p_y) = (c_x - b_x) / (c_y - b_y)


theorem trajectory_of_A (A B C P Q: ℝ × ℝ) 
  (hB : B = vertexB) (hC : C = vertexC)
  (h1 : condition1 A P) (h2 : condition2 Q A B C)
  (h3 : condition3 P Q B C) :
  (let (a_x, a_y) := A in a_x^2 / 3 + a_y^2 = 1) :=
begin
  sorry
end

theorem fixed_point_MN (F : ℝ × ℝ) 
  (hF: F = (sqrt 2, 0))
  : (3sqrt(2)/4, 0) :=
begin
  sorry
end

end trajectory_of_A_fixed_point_MN_l249_249796


namespace paths_from_C_to_D_l249_249346

theorem paths_from_C_to_D :
  let total_moves := 10 in
  let right_moves := 6 in
  let up_moves := 4 in
  total_moves = right_moves + up_moves →
  @Finset.card _ _ (Finset.range total_moves).powerset (fun s => s.card = up_moves) = 210 := 
by sorry

end paths_from_C_to_D_l249_249346


namespace ellipse_eccentricity_l249_249240

theorem ellipse_eccentricity (m : ℝ) (e : ℝ) : 
  (∀ x y : ℝ, (x^2 / m) + (y^2 / 4) = 1) ∧ foci_y_axis ∧ e = 1 / 2 → m = 3 :=
by
  sorry

end ellipse_eccentricity_l249_249240


namespace exists_nat_n_gt_one_sqrt_expr_nat_l249_249195

theorem exists_nat_n_gt_one_sqrt_expr_nat (n : ℕ) : ∃ (n : ℕ), n > 1 ∧ ∃ (m : ℕ), n^(7 / 8) = m :=
by
  sorry

end exists_nat_n_gt_one_sqrt_expr_nat_l249_249195


namespace social_media_phone_ratio_l249_249893

/-- 
Given that Jonathan spends 8 hours on his phone daily and 28 hours on social media in a week, 
prove that the ratio of the time spent on social media to the total time spent on his phone daily is \( 1 : 2 \).
-/
theorem social_media_phone_ratio (daily_phone_hours : ℕ) (weekly_social_media_hours : ℕ) 
  (h1 : daily_phone_hours = 8) (h2 : weekly_social_media_hours = 28) :
  (weekly_social_media_hours / 7) / daily_phone_hours = 1 / 2 := 
by
  sorry

end social_media_phone_ratio_l249_249893


namespace m_div_x_l249_249134

variable (a b k : ℝ)
variable (ha : a = 4 * k)
variable (hb : b = 5 * k)
variable (k_pos : k > 0)

def x := a * 1.25
def m := b * 0.20

theorem m_div_x : m / x = 1 / 5 := by
  sorry

end m_div_x_l249_249134


namespace parabola_vertex_properties_l249_249861

theorem parabola_vertex_properties :
  ∃ (d e f : ℝ), 
    (∀ (x y : ℝ), x = d * y^2 + e * y + f) ∧ 
    (∀ (y : ℝ), x = d * (y + 6)^2 + 7) ∧
    (x = 2 ∧ y = -3) → 
    d + e + f = -182 / 9 :=
by
  sorry

end parabola_vertex_properties_l249_249861


namespace trees_left_after_typhoon_l249_249219

variable (initial_trees : ℕ)
variable (died_trees : ℕ)
variable (remaining_trees : ℕ)

theorem trees_left_after_typhoon :
  initial_trees = 20 →
  died_trees = 16 →
  remaining_trees = initial_trees - died_trees →
  remaining_trees = 4 :=
by
  intros h_initial h_died h_remaining
  rw [h_initial, h_died] at h_remaining
  exact h_remaining

end trees_left_after_typhoon_l249_249219


namespace solve_equation_l249_249713

theorem solve_equation (x : ℝ) : 
  (x ^ (Real.log x / Real.log 2) = x^5 / 32) ↔ (x = 2^((5 + Real.sqrt 5) / 2) ∨ x = 2^((5 - Real.sqrt 5) / 2)) := 
by 
  sorry

end solve_equation_l249_249713


namespace problem_statement_l249_249355

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f'' (x : ℝ) : ℝ := -Real.sin x - Real.cos x

theorem problem_statement (a : ℝ) (h : f'' a = 3 * f a) : 
  (Real.sin a)^2 - 3 / (Real.cos a)^2 + 1 = -14 / 9 := 
sorry

end problem_statement_l249_249355


namespace B_investment_amount_l249_249473

-- Define given conditions in Lean 4

def A_investment := 400
def total_months := 12
def B_investment_months := 6
def total_profit := 100
def A_share := 80
def B_share := total_profit - A_share

-- The problem statement in Lean 4 that needs to be proven:
theorem B_investment_amount (A_investment B_investment_months total_profit A_share B_share: ℕ)
  (hA_investment : A_investment = 400)
  (htotal_months : total_months = 12)
  (hB_investment_months : B_investment_months = 6)
  (htotal_profit : total_profit = 100)
  (hA_share : A_share = 80)
  (hB_share : B_share = total_profit - A_share) 
  : (∃ (B: ℕ), 
       (5 * (A_investment * total_months) = 4 * (400 * total_months + B * B_investment_months)) 
       ∧ B = 200) :=
sorry

end B_investment_amount_l249_249473


namespace sin_300_eq_neg_sqrt_3_div_2_l249_249515

noncomputable def sin_300_degrees : ℝ := Real.sin (300 * Real.pi / 180)

theorem sin_300_eq_neg_sqrt_3_div_2 : sin_300_degrees = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_300_eq_neg_sqrt_3_div_2_l249_249515


namespace ma_m_gt_mb_l249_249936

theorem ma_m_gt_mb (a b : ℝ) (m : ℝ) (h : a < b) : ¬ (m * a > m * b) → m ≥ 0 := 
  sorry

end ma_m_gt_mb_l249_249936


namespace Brian_traveled_60_miles_l249_249738

theorem Brian_traveled_60_miles (mpg gallons : ℕ) (hmpg : mpg = 20) (hgallons : gallons = 3) :
    mpg * gallons = 60 := by
  sorry

end Brian_traveled_60_miles_l249_249738


namespace no_solutions_to_equation_l249_249818

theorem no_solutions_to_equation (a b c : ℤ) : a^2 + b^2 - 8 * c ≠ 6 := 
by 
-- sorry to skip the proof part
sorry

end no_solutions_to_equation_l249_249818


namespace intersecting_chords_theorem_l249_249456

theorem intersecting_chords_theorem
  (a b : ℝ) (h1 : a = 12) (h2 : b = 18)
  (c d k : ℝ) (h3 : c = 3 * k) (h4 : d = 8 * k) :
  (a * b = c * d) → (k = 3) → (c + d = 33) :=
by 
  sorry

end intersecting_chords_theorem_l249_249456


namespace diff_sum_even_odd_l249_249459

theorem diff_sum_even_odd (n : ℕ) (hn : n = 1500) :
  let sum_odd := n * (2 * n - 1)
  let sum_even := n * (2 * n + 1)
  sum_even - sum_odd = 1500 :=
by
  sorry

end diff_sum_even_odd_l249_249459


namespace total_pairs_of_shoes_equivalence_l249_249269

variable (Scott Anthony Jim Melissa Tim: ℕ)

theorem total_pairs_of_shoes_equivalence
    (h1 : Scott = 7)
    (h2 : Anthony = 3 * Scott)
    (h3 : Jim = Anthony - 2)
    (h4 : Jim = 2 * Melissa)
    (h5 : Tim = (Anthony + Melissa) / 2):

  Scott + Anthony + Jim + Melissa + Tim = 71 :=
  by
  sorry

end total_pairs_of_shoes_equivalence_l249_249269


namespace sin_300_eq_neg_sqrt_3_div_2_l249_249507

theorem sin_300_eq_neg_sqrt_3_div_2 : Real.sin (300 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt_3_div_2_l249_249507


namespace sin_300_eq_neg_sqrt_3_div_2_l249_249518

noncomputable def sin_300_degrees : ℝ := Real.sin (300 * Real.pi / 180)

theorem sin_300_eq_neg_sqrt_3_div_2 : sin_300_degrees = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_300_eq_neg_sqrt_3_div_2_l249_249518


namespace find_a_l249_249670

/-
We define the required properties and theorems to set up our proof context.
-/

variables (ξ : ℝ → ℝ) (a : ℝ)

-- Normal distribution with mean 3 and variance 4
def normal_distribution (ξ : ℝ → ℝ) : Prop :=
  ∃ μ σ, μ = 3 ∧ σ^2 = 4 ∧ ξ ~ Normal μ σ

-- Given condition: P(ξ < 2a - 3) = P(ξ > a + 2)
def probability_condition (ξ : ℝ → ℝ) (a : ℝ) : Prop :=
  P(ξ < 2 * a - 3) = P(ξ > a + 2)

theorem find_a (h1 : normal_distribution ξ) (h2 : probability_condition ξ a) : 
  a = 7 / 3 :=
sorry  -- Proof will be filled in later

end find_a_l249_249670


namespace sin_300_eq_neg_sqrt3_div_2_l249_249587

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249587


namespace max_objective_function_value_l249_249202

def objective_function (x1 x2 : ℝ) := 4 * x1 + 6 * x2

theorem max_objective_function_value :
  ∃ x1 x2 : ℝ, 
    (x1 >= 0) ∧ 
    (x2 >= 0) ∧ 
    (x1 + x2 <= 18) ∧ 
    (0.5 * x1 + x2 <= 12) ∧ 
    (2 * x1 <= 24) ∧ 
    (2 * x2 <= 18) ∧ 
    (∀ y1 y2 : ℝ, 
      (y1 >= 0) ∧ 
      (y2 >= 0) ∧ 
      (y1 + y2 <= 18) ∧ 
      (0.5 * y1 + y2 <= 12) ∧ 
      (2 * y1 <= 24) ∧ 
      (2 * y2 <= 18) -> 
      objective_function y1 y2 <= objective_function x1 x2) ∧
    (objective_function x1 x2 = 84) :=
by
  use 12, 6
  sorry

end max_objective_function_value_l249_249202


namespace damaged_books_count_l249_249630

variables (o d : ℕ)

theorem damaged_books_count (h1 : o + d = 69) (h2 : o = 6 * d - 8) : d = 11 := 
by 
  sorry

end damaged_books_count_l249_249630


namespace parallel_vectors_x_value_l249_249786

def vectors_are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, vectors_are_parallel (-1, 4) (x, 2) → x = -1 / 2 := 
by 
  sorry

end parallel_vectors_x_value_l249_249786


namespace problem_conditions_l249_249988

noncomputable def f (x : ℝ) : ℝ := (2 * x - x^2) * Real.exp x

theorem problem_conditions :
  (∀ x, f x > 0 ↔ 0 < x ∧ x < 2) ∧
  (∃ x_max, x_max = Real.sqrt 2 ∧ (∀ y, f y ≤ f x_max)) ∧
  ¬(∃ x_min, ∀ y, f x_min ≤ f y) :=
by sorry

end problem_conditions_l249_249988


namespace total_simple_interest_l249_249875

theorem total_simple_interest (P R T : ℝ) (hP : P = 6178.846153846154) (hR : R = 0.13) (hT : T = 5) :
    P * R * T = 4011.245192307691 := by
  rw [hP, hR, hT]
  norm_num
  sorry

end total_simple_interest_l249_249875


namespace ratio_of_a_b_l249_249624

-- Define the system of equations as given in the problem
variables (x y a b : ℝ)

-- Conditions: the system of equations and b ≠ 0
def system_of_equations (a b : ℝ) (x y : ℝ) := 
  4 * x - 3 * y = a ∧ 6 * y - 8 * x = b

-- The theorem we aim to prove
theorem ratio_of_a_b (h : system_of_equations a b x y) (h₀ : b ≠ 0) : a / b = -1 / 2 :=
sorry

end ratio_of_a_b_l249_249624


namespace sin_300_eq_neg_sqrt_3_div_2_l249_249512

theorem sin_300_eq_neg_sqrt_3_div_2 : Real.sin (300 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt_3_div_2_l249_249512


namespace angle_remains_acute_l249_249872

noncomputable theory

structure Quadrilateral (α : Type*) [InnerProductSpace ℝ α] :=
(A B C D : α)
(midpoints_acute_angle : ∀ (P Q R : α), 
  (P = (A + B) / 2 ∧ Q = (B + C) / 2 ∧ R = (C + D) / 2) → 
  inner (Q - P) (R - Q) > 0)

theorem angle_remains_acute {α : Type*} [InnerProductSpace ℝ α] (quad : Quadrilateral α) :
  ∀ (P Q R : α), (P = (quad.A + quad.B) / 2 ∧ Q = (quad.B + quad.C) / 2 ∧ R = (quad.C + quad.D) / 2) →
  inner (Q - P) (R - Q) > 0 :=
sorry

end angle_remains_acute_l249_249872


namespace sin_300_l249_249595

-- Definitions and conditions in Lean 4
def angle_deg := ℝ
def sin (θ : angle_deg) : ℝ := Real.sin (θ * Real.pi / 180)

noncomputable def Q_angle : angle_deg := 300
noncomputable def first_quadrant_equivalent_angle : angle_deg := 360 - 300
noncomputable def sin_60 : ℝ := Real.sin (60 * Real.pi / 180)

-- Main statement to prove
theorem sin_300 : sin 300 = -sin_60 := by
  sorry

end sin_300_l249_249595


namespace integer_a_satisfies_equation_l249_249763

theorem integer_a_satisfies_equation (a b c : ℤ) :
  (∃ b c : ℤ, (x - a) * (x - 5) + 2 = (x + b) * (x + c)) → 
    a = 2 :=
by
  intro h_eq
  -- Proof goes here
  sorry

end integer_a_satisfies_equation_l249_249763


namespace opposite_of_neg_five_l249_249446

theorem opposite_of_neg_five : -(-5) = 5 := by
  sorry

end opposite_of_neg_five_l249_249446


namespace cubic_expression_solution_l249_249980

theorem cubic_expression_solution (r s : ℝ) (h₁ : 3 * r^2 - 4 * r - 7 = 0) (h₂ : 3 * s^2 - 4 * s - 7 = 0) :
  (3 * r^3 - 3 * s^3) / (r - s) = 37 / 3 :=
sorry

end cubic_expression_solution_l249_249980


namespace system_of_equations_solution_l249_249992

theorem system_of_equations_solution (x y : ℝ) (h1 : 2 * x ^ 2 - 5 * x + 3 = 0) (h2 : y = 3 * x + 1) : 
  (x = 1.5 ∧ y = 5.5) ∨ (x = 1 ∧ y = 4) :=
sorry

end system_of_equations_solution_l249_249992


namespace percentage_decrease_l249_249745

theorem percentage_decrease (purchase_price selling_price decrease gross_profit : ℝ)
  (h_purchase : purchase_price = 81)
  (h_markup : selling_price = purchase_price + 0.25 * selling_price)
  (h_gross_profit : gross_profit = 5.40)
  (h_decrease : decrease = 108 - 102.60) :
  (decrease / 108) * 100 = 5 :=
by sorry

end percentage_decrease_l249_249745


namespace range_of_f_2x_le_1_l249_249210

-- Given conditions
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonically_decreasing (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

def cond_f_neg_2_eq_1 (f : ℝ → ℝ) : Prop :=
  f (-2) = 1

-- Main theorem
theorem range_of_f_2x_le_1 (f : ℝ → ℝ) 
  (h1 : is_odd f)
  (h2 : is_monotonically_decreasing f (Set.Iic 0))
  (h3 : cond_f_neg_2_eq_1 f) :
  Set.Icc (-1 : ℝ) 1 = { x | |f (2 * x)| ≤ 1 } :=
sorry

end range_of_f_2x_le_1_l249_249210


namespace inequality_solution_range_l249_249764

theorem inequality_solution_range (x : ℝ) : (x^2 + 3*x - 10 < 0) ↔ (-5 < x ∧ x < 2) :=
by
  sorry

end inequality_solution_range_l249_249764


namespace sin_300_eq_neg_sqrt3_div_2_l249_249585

noncomputable def sin_300 : ℝ := sin (300 * real.pi / 180)

theorem sin_300_eq_neg_sqrt3_div_2 : 
  (∠300° = 360° - 60°) →
  (300° ∈ fourth_quadrant) →
  (∀ θ ∈ fourth_quadrant, sin θ < 0) →
  (sin (60 * real.pi / 180) = sqrt 3 / 2) →
  sin_300 = -sqrt 3 / 2 := 
by
  intros h1 h2 h3 h4
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249585


namespace parallelepiped_length_l249_249007

theorem parallelepiped_length (n : ℕ)
  (h1 : ∃ n : ℕ, n = 18) 
  (h2 : one_third_of_cubes_have_red_faces : (∃ k : ℕ, k = ((n * (n - 2) * (n - 4)) / 3)) 
        ∧ (remaining_unpainted_cubes : (∃ m : ℕ , m = (2 * (n * (n - 2) * (n - 4)) / 3))))
  (h3 : painted_and_cut_into_cubes : (∃ a b c : ℕ, a = n ∧ b = (n - 2) ∧ c = (n - 4)))
  (h4 : all_sides_whole_cm : (∃ d : ℕ , d = n ∧ d = (n - 2) ∧ d = (n - 4))) :
  n = 18 :=
begin
  sorry
end

end parallelepiped_length_l249_249007


namespace square_area_eq_36_l249_249172

theorem square_area_eq_36 :
  let triangle_side1 := 5.5
  let triangle_side2 := 7.5
  let triangle_side3 := 11
  let triangle_perimeter := triangle_side1 + triangle_side2 + triangle_side3
  let square_perimeter := triangle_perimeter
  let square_side_length := square_perimeter / 4
  let square_area := square_side_length * square_side_length
  square_area = 36 := by
  sorry

end square_area_eq_36_l249_249172


namespace car_payment_months_l249_249415

theorem car_payment_months 
    (total_price : ℕ) 
    (initial_payment : ℕ)
    (monthly_payment : ℕ) 
    (h_total_price : total_price = 13380) 
    (h_initial_payment : initial_payment = 5400) 
    (h_monthly_payment : monthly_payment = 420) 
    : total_price - initial_payment = 7980 
    ∧ (total_price - initial_payment) / monthly_payment = 19 := 
by 
  sorry

end car_payment_months_l249_249415


namespace find_N_l249_249679

theorem find_N : 
  (1993 + 1994 + 1995 + 1996 + 1997) / N = (3 + 4 + 5 + 6 + 7) / 5 → 
  N = 1995 :=
by
  sorry

end find_N_l249_249679


namespace calculate_expression_l249_249464

theorem calculate_expression :
  427 / 2.68 * 16 * 26.8 / 42.7 * 16 = 25600 :=
sorry

end calculate_expression_l249_249464


namespace simplify_polynomial_l249_249104

variable (x : ℝ)

theorem simplify_polynomial : 
  (2 * x^4 + 3 * x^3 - 5 * x + 6) + (-6 * x^4 - 2 * x^3 + 3 * x^2 + 5 * x - 4) = 
  -4 * x^4 + x^3 + 3 * x^2 + 2 :=
by
  sorry

end simplify_polynomial_l249_249104


namespace abs_x_minus_one_sufficient_not_necessary_l249_249224

variable (x : ℝ) -- x is a real number

theorem abs_x_minus_one_sufficient_not_necessary (h : |x - 1| > 2) :
  (x^2 > 1) ∧ (∃ (y : ℝ), x^2 > 1 ∧ |y - 1| ≤ 2) := by
  sorry

end abs_x_minus_one_sufficient_not_necessary_l249_249224


namespace books_shelved_in_fiction_section_l249_249974

def calculate_books_shelved_in_fiction_section (total_books : ℕ) (remaining_books : ℕ) (books_shelved_in_history : ℕ) (books_shelved_in_children : ℕ) (books_added_back : ℕ) : ℕ :=
  let total_shelved := total_books - remaining_books
  let adjusted_books_shelved_in_children := books_shelved_in_children - books_added_back
  let total_shelved_in_history_and_children := books_shelved_in_history + adjusted_books_shelved_in_children
  total_shelved - total_shelved_in_history_and_children

theorem books_shelved_in_fiction_section:
  calculate_books_shelved_in_fiction_section 51 16 12 8 4 = 19 :=
by 
  -- Definition of the function gives the output directly so proof is trivial.
  rfl

end books_shelved_in_fiction_section_l249_249974


namespace factor_expression_l249_249897

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249897


namespace total_games_l249_249982

-- Defining the conditions.
def games_this_month : ℕ := 9
def games_last_month : ℕ := 8
def games_next_month : ℕ := 7

-- Theorem statement to prove the total number of games.
theorem total_games : games_this_month + games_last_month + games_next_month = 24 := by
  sorry

end total_games_l249_249982


namespace total_people_going_to_museum_l249_249712

def number_of_people_on_first_bus := 12
def number_of_people_on_second_bus := 2 * number_of_people_on_first_bus
def number_of_people_on_third_bus := number_of_people_on_second_bus - 6
def number_of_people_on_fourth_bus := number_of_people_on_first_bus + 9

theorem total_people_going_to_museum :
  number_of_people_on_first_bus + number_of_people_on_second_bus + number_of_people_on_third_bus + number_of_people_on_fourth_bus = 75 :=
by
  sorry

end total_people_going_to_museum_l249_249712


namespace lcm_gcd_product_24_36_l249_249724

theorem lcm_gcd_product_24_36 : 
  let a := 24
  let b := 36
  let g := Int.gcd a b
  let l := Int.lcm a b
  g * l = 864 := by
  let a := 24
  let b := 36
  let g := Int.gcd a b
  have gcd_eq : g = 12 := by sorry
  let l := Int.lcm a b
  have lcm_eq : l = 72 := by sorry
  show g * l = 864 from by
    rw [gcd_eq, lcm_eq]
    exact calc
      12 * 72 = 864 : by norm_num

end lcm_gcd_product_24_36_l249_249724


namespace students_both_l249_249330

noncomputable def students_total : ℕ := 32
noncomputable def students_go : ℕ := 18
noncomputable def students_chess : ℕ := 23

theorem students_both : students_go + students_chess - students_total = 9 := by
  sorry

end students_both_l249_249330


namespace sqrt_expression_real_l249_249070

theorem sqrt_expression_real (x : ℝ) : (∃ y : ℝ, y = sqrt (x + 1)) ↔ x ≥ -1 := 
by
  sorry

end sqrt_expression_real_l249_249070


namespace unique_solution_pairs_l249_249775

theorem unique_solution_pairs :
  ∃! (b c : ℕ), b > 0 ∧ c > 0 ∧ (b^2 = 4 * c) ∧ (c^2 = 4 * b) :=
sorry

end unique_solution_pairs_l249_249775


namespace sin_300_eq_neg_one_half_l249_249540

-- Definitions from the conditions
def point_on_unit_circle (θ : ℝ) : ℝ × ℝ :=
  (Real.cos (θ), Real.sin (θ))

def foot_of_perpendicular (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, 0)

def is_thirty_sixty_ninety_triangle (P : ℝ × ℝ) : Prop :=
  sqrt(P.1^2 + P.2^2) = 1 ∧ P.1 = sqrt(3) / 2 ∧ P.2 = -1/2

-- Problem statement
theorem sin_300_eq_neg_one_half : Real.sin (5 * Real.pi / 3) = -1/2 := by
  sorry

end sin_300_eq_neg_one_half_l249_249540


namespace color_change_probability_is_correct_l249_249757

-- Given definitions
def cycle_time : ℕ := 45 + 5 + 10 + 40

def favorable_time : ℕ := 5 + 5 + 5

def probability_color_change : ℚ := favorable_time / cycle_time

-- Theorem statement to prove the probability
theorem color_change_probability_is_correct :
  probability_color_change = 0.15 := 
sorry

end color_change_probability_is_correct_l249_249757


namespace rationalize_denominator_sqrt_l249_249987

theorem rationalize_denominator_sqrt (x y : ℝ) (hx : x = 5) (hy : y = 12) :
  Real.sqrt (x / y) = Real.sqrt 15 / 6 :=
by
  rw [hx, hy]
  sorry

end rationalize_denominator_sqrt_l249_249987


namespace replaced_person_weight_l249_249998

theorem replaced_person_weight (W : ℝ) (increase : ℝ) (new_weight : ℝ) (average_increase : ℝ) (number_of_persons : ℕ) :
  average_increase = 2.5 →
  new_weight = 70 →
  number_of_persons = 8 →
  increase = number_of_persons * average_increase →
  W + increase = W - replaced_weight + new_weight →
  replaced_weight = 50 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end replaced_person_weight_l249_249998


namespace percentage_transform_l249_249228

theorem percentage_transform (n : ℝ) (h : 0.3 * 0.4 * n = 36) : 0.4 * 0.3 * n = 36 :=
by
  sorry

end percentage_transform_l249_249228


namespace sin_of_300_degrees_l249_249566

theorem sin_of_300_degrees : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 := by
  sorry

end sin_of_300_degrees_l249_249566


namespace sin_of_300_degrees_l249_249562

theorem sin_of_300_degrees : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 := by
  sorry

end sin_of_300_degrees_l249_249562


namespace speed_ratio_l249_249309

theorem speed_ratio (va vb : ℝ) (L : ℝ) (h : va = vb * k) (head_start : vb * (L - 0.05 * L) = vb * L) : 
    (va / vb) = (1 / 0.95) :=
by
  sorry

end speed_ratio_l249_249309


namespace henri_total_miles_l249_249037

noncomputable def g_total : ℕ := 315 * 3
noncomputable def h_total : ℕ := g_total + 305

theorem henri_total_miles : h_total = 1250 :=
by
  -- proof goes here
  sorry

end henri_total_miles_l249_249037


namespace min_director_games_l249_249688

theorem min_director_games (n k : ℕ) (h1 : (n * (n - 1)) / 2 + k = 325) (h2 : (26 * 25) / 2 = 325) : k = 0 :=
by {
  -- The conditions are provided in the hypothesis, and the goal is proving the minimum games by director equals 0.
  sorry
}

end min_director_games_l249_249688


namespace find_sale_month4_l249_249748

-- Define sales for each month
def sale_month1 : ℕ := 5400
def sale_month2 : ℕ := 9000
def sale_month3 : ℕ := 6300
def sale_month5 : ℕ := 4500
def sale_month6 : ℕ := 1200
def avg_sale_per_month : ℕ := 5600

-- Define the total number of months
def num_months : ℕ := 6

-- Define the expression for total sales required
def total_sales_required : ℕ := avg_sale_per_month * num_months

-- Define the expression for total known sales
def total_known_sales : ℕ := sale_month1 + sale_month2 + sale_month3 + sale_month5 + sale_month6

-- State and prove the theorem:
theorem find_sale_month4 : sale_month1 = 5400 → sale_month2 = 9000 → sale_month3 = 6300 → 
                            sale_month5 = 4500 → sale_month6 = 1200 → avg_sale_per_month = 5600 →
                            num_months = 6 → (total_sales_required - total_known_sales = 8200) := 
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end find_sale_month4_l249_249748


namespace simplify_expression_l249_249025

theorem simplify_expression (a : ℝ) : 2 * (a + 2) - 2 * a = 4 :=
by
  sorry

end simplify_expression_l249_249025


namespace triangle_BX_eq_CY_l249_249255

theorem triangle_BX_eq_CY 
  (ABC : Triangle)
  (Γ : Circumcircle ABC)
  (N : Point)
  (HN : IsMidpointOfArcContaining N B A C Γ)
  (𝒞 : Circle)
  (H𝒞1 : PassesThrough 𝒞 A)
  (H𝒞2 : PassesThrough 𝒞 N)
  (X : Point)
  (HX : Intersects 𝒞 (LineSegment A B) X)
  (Y : Point)
  (HY : Intersects 𝒞 (LineSegment A C) Y) 
  : SegmentLength B X = SegmentLength C Y := 
sorry

end triangle_BX_eq_CY_l249_249255


namespace circle_equation_l249_249321

theorem circle_equation :
  ∃ (r : ℝ), ∀ (x y : ℝ), (x - 3) ^ 2 + (y - 1) ^ 2 = r ↔ (x = 0 ∧ y = 0) → ((x - 3) ^ 2 + (y - 1) ^ 2 = 10) :=
by
  sorry

end circle_equation_l249_249321


namespace incorrect_statement_B_l249_249146

axiom statement_A : ¬ (0 > 0 ∨ 0 < 0)
axiom statement_C : ∀ (q : ℚ), (∃ (m : ℤ), q = m) ∨ (∃ (a b : ℤ), b ≠ 0 ∧ q = a / b)
axiom statement_D : abs (0 : ℚ) = 0

theorem incorrect_statement_B : ¬ (∀ (q : ℚ), abs q ≥ 1 → abs 1 = abs q) := sorry

end incorrect_statement_B_l249_249146


namespace tangent_lines_to_ln_abs_through_origin_l249_249826

noncomputable def tangent_line_through_origin (x y: ℝ) : Prop :=
  (y = log (abs x)) ∧ ((x - exp(1) * y = 0) ∨ (x + exp(1) * y = 0))

theorem tangent_lines_to_ln_abs_through_origin :
  ∃ (f : ℝ → ℝ), 
  (∀ x, f x = log (abs x)) ∧ 
  ∀ x y, (tangent_line_through_origin x y) := sorry

end tangent_lines_to_ln_abs_through_origin_l249_249826


namespace min_f_x_eq_one_implies_a_eq_zero_or_two_l249_249792

theorem min_f_x_eq_one_implies_a_eq_zero_or_two (a : ℝ) :
  (∃ x : ℝ, |x + 1| + |x + a| = 1) → (a = 0 ∨ a = 2) := by
  sorry

end min_f_x_eq_one_implies_a_eq_zero_or_two_l249_249792


namespace coin_flip_heads_probability_l249_249119

theorem coin_flip_heads_probability :
  let coins : List String := ["penny", "nickel", "dime", "quarter", "half-dollar"]
  let independent_event (coin : String) : Prop := True
  let outcomes := (2 : ℕ) ^ (List.length coins)
  let successful_outcomes := 4
  let probability := successful_outcomes / outcomes
  probability = 1 / 8 := 
by
  sorry

end coin_flip_heads_probability_l249_249119


namespace find_angle_between_vectors_l249_249782

noncomputable def angle_between_vectors 
  (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) 
  (perp1 : (a + 3*b) * (7*a - 5*b) = 0) 
  (perp2 : (a - 4*b) * (7*a - 2*b) = 0) : ℝ :=
  60

theorem find_angle_between_vectors 
  (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) 
  (perp1 : (a + 3*b) * (7*a - 5*b) = 0) 
  (perp2 : (a - 4*b) * (7*a - 2*b) = 0) : angle_between_vectors a b a_nonzero b_nonzero perp1 perp2 = 60 :=
  by 
  sorry

end find_angle_between_vectors_l249_249782


namespace sin_300_eq_neg_sqrt3_div_2_l249_249572

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249572


namespace Niklaus_walked_distance_l249_249412

noncomputable def MilesToFeet (miles : ℕ) : ℕ := miles * 5280
noncomputable def YardsToFeet (yards : ℕ) : ℕ := yards * 3

theorem Niklaus_walked_distance (n_feet : ℕ) :
  MilesToFeet 4 + YardsToFeet 975 + n_feet = 25332 → n_feet = 1287 := by
  sorry

end Niklaus_walked_distance_l249_249412


namespace total_employees_l249_249276

-- Definitions based on the conditions:
variables (N S : ℕ)
axiom condition1 : 75 % 100 * S = 75 / 100 * S
axiom condition2 : 65 % 100 * S = 65 / 100 * S
axiom condition3 : N - S = 40
axiom condition4 : 5 % 6 * N = 5 / 6 * N

-- The statement to be proven:
theorem total_employees (N S : ℕ)
    (h1 : 75 % 100 * S = 75 / 100 * S)
    (h2 : 65 % 100 * S = 65 / 100 * S)
    (h3 : N - S = 40)
    (h4 : 5 % 6 * N = 5 / 6 * N)
    : N = 240 :=
sorry

end total_employees_l249_249276


namespace min_area_triangle_l249_249075

-- Conditions
def point_on_curve (x y : ℝ) : Prop :=
  y^2 = 2 * x

def incircle (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

-- Theorem statement
theorem min_area_triangle (x₀ y₀ b c : ℝ) (h_curve : point_on_curve x₀ y₀) 
  (h_bc_yaxis : b ≠ c) (h_incircle : incircle x₀ y₀) :
  ∃ P : ℝ × ℝ, 
    ∃ B C : ℝ × ℝ, 
    ∃ S : ℝ,
    point_on_curve P.1 P.2 ∧
    B = (0, b) ∧
    C = (0, c) ∧
    incircle P.1 P.2 ∧
    S = (x₀ - 2) + (4 / (x₀ - 2)) + 4 ∧
    S = 8 :=
sorry

end min_area_triangle_l249_249075


namespace stream_speed_l249_249862

theorem stream_speed (v : ℝ) : 
  (∀ (speed_boat_in_still_water distance time : ℝ), 
    speed_boat_in_still_water = 25 ∧ distance = 90 ∧ time = 3 →
    distance = (speed_boat_in_still_water + v) * time) →
  v = 5 :=
by
  intro h
  have h1 := h 25 90 3 ⟨rfl, rfl, rfl⟩
  sorry

end stream_speed_l249_249862


namespace sin_300_l249_249602

-- Definitions and conditions in Lean 4
def angle_deg := ℝ
def sin (θ : angle_deg) : ℝ := Real.sin (θ * Real.pi / 180)

noncomputable def Q_angle : angle_deg := 300
noncomputable def first_quadrant_equivalent_angle : angle_deg := 360 - 300
noncomputable def sin_60 : ℝ := Real.sin (60 * Real.pi / 180)

-- Main statement to prove
theorem sin_300 : sin 300 = -sin_60 := by
  sorry

end sin_300_l249_249602


namespace max_fruit_to_teacher_l249_249158

theorem max_fruit_to_teacher (A G : ℕ) : (A % 7 ≤ 6) ∧ (G % 7 ≤ 6) :=
by
  sorry

end max_fruit_to_teacher_l249_249158


namespace find_uv_non_integer_l249_249978

def p (b : Fin 14 → ℚ) (x y : ℚ) : ℚ :=
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 + 
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3 + 
  b 10 * x^4 + b 11 * y^4 + b 12 * x^3 * y^2 + b 13 * y^3 * x^2

variables (b : Fin 14 → ℚ)
variables (u v : ℚ)

def zeros_at_specific_points :=
  p b 0 0 = 0 ∧ p b 1 0 = 0 ∧ p b (-1) 0 = 0 ∧
  p b 0 1 = 0 ∧ p b 0 (-1) = 0 ∧ p b 1 1 = 0 ∧
  p b (-1) (-1) = 0 ∧ p b 2 2 = 0 ∧ 
  p b 2 (-2) = 0 ∧ p b (-2) 2 = 0

theorem find_uv_non_integer
  (h : zeros_at_specific_points b) :
  p b (5/19) (16/19) = 0 :=
sorry

end find_uv_non_integer_l249_249978


namespace cone_volume_l249_249939

theorem cone_volume (central_angle : ℝ) (sector_area : ℝ) (h1 : central_angle = 120) (h2 : sector_area = 3 * Real.pi) :
  ∃ V : ℝ, V = (2 * Real.sqrt 2 * Real.pi) / 3 :=
by
  -- We acknowledge the input condition where the angle is 120° and sector area is 3π
  -- The problem requires proving the volume of the cone
  sorry

end cone_volume_l249_249939


namespace intersection_S_T_l249_249943

def S : Set ℝ := {x | x > -2}

def T : Set ℝ := {x | -4 ≤ x ∧ x ≤ 1}

theorem intersection_S_T : S ∩ T = {x | -2 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_S_T_l249_249943


namespace isosceles_triangle_angles_l249_249967

noncomputable def angle_opposite (a b c : ℝ) := real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))

theorem isosceles_triangle_angles :
  let a := 5 in
  let b := 5 in
  let c := real.sqrt 17 - real.sqrt 5 in
  let θ := angle_opposite a b c in
  let φ := (180 - θ) / 2 in
  θ = real.arccos ((14 + real.sqrt 85) / 25) ∧ φ = (180 - θ) / 2 :=
by
  sorry

end isosceles_triangle_angles_l249_249967


namespace compare_b_d_l249_249223

noncomputable def percentage_increase (x : ℝ) (p : ℝ) := x * (1 + p)
noncomputable def percentage_decrease (x : ℝ) (p : ℝ) := x * (1 - p)

theorem compare_b_d (a b c d : ℝ)
  (h1 : 0 < b)
  (h2 : a = percentage_increase b 0.02)
  (h3 : c = percentage_decrease a 0.01)
  (h4 : d = percentage_decrease c 0.01) :
  b > d :=
sorry

end compare_b_d_l249_249223


namespace find_number_l249_249302

theorem find_number 
  (x : ℝ)
  (h : (1 / 10) * x - (1 / 1000) * x = 700) :
  x = 700000 / 99 :=
by 
  sorry

end find_number_l249_249302


namespace cone_cube_volume_ratio_l249_249169

noncomputable def volumeRatio (s : ℝ) : ℝ :=
  let r := s / 2
  let h := s
  let volume_cone := (1 / 3) * Real.pi * r^2 * h
  let volume_cube := s^3
  volume_cone / volume_cube

theorem cone_cube_volume_ratio (s : ℝ) (h_cube_eq_s : s > 0) :
  volumeRatio s = Real.pi / 12 :=
by
  sorry

end cone_cube_volume_ratio_l249_249169


namespace sin_300_l249_249599

-- Definitions and conditions in Lean 4
def angle_deg := ℝ
def sin (θ : angle_deg) : ℝ := Real.sin (θ * Real.pi / 180)

noncomputable def Q_angle : angle_deg := 300
noncomputable def first_quadrant_equivalent_angle : angle_deg := 360 - 300
noncomputable def sin_60 : ℝ := Real.sin (60 * Real.pi / 180)

-- Main statement to prove
theorem sin_300 : sin 300 = -sin_60 := by
  sorry

end sin_300_l249_249599


namespace sin_300_deg_l249_249488

noncomputable def sin_300_equals_neg_sqrt3_div_2 : Prop :=
  sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2

theorem sin_300_deg : sin_300_equals_neg_sqrt3_div_2 :=
by
  sorry

end sin_300_deg_l249_249488


namespace min_trips_required_l249_249313

def masses : List ℕ := [150, 62, 63, 66, 70, 75, 79, 84, 95, 96, 99]
def load_capacity : ℕ := 190

theorem min_trips_required :
  ∃ (trips : ℕ), 
  (∀ partition : List (List ℕ), (∀ group : List ℕ, group ∈ partition → 
  group.sum ≤ load_capacity) ∧ partition.join = masses → 
  partition.length ≥ 6) :=
sorry

end min_trips_required_l249_249313


namespace smallest_pos_int_gcd_gt_one_l249_249461

theorem smallest_pos_int_gcd_gt_one : ∃ n: ℕ, n > 0 ∧ (Nat.gcd (8 * n - 3) (5 * n + 4) > 1) ∧ n = 121 :=
by
  sorry

end smallest_pos_int_gcd_gt_one_l249_249461


namespace repeating_decimal_fraction_l249_249347

theorem repeating_decimal_fraction :
  let x : ℚ := 75 / 99,
      z : ℚ := 25 / 99,
      y : ℚ := 2 + z 
  in (x / y) = 2475 / 7339 :=
by
  sorry

end repeating_decimal_fraction_l249_249347


namespace find_x3_minus_y3_l249_249047

theorem find_x3_minus_y3 {x y : ℤ} (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : x^3 - y^3 = 108 :=
by 
  sorry

end find_x3_minus_y3_l249_249047


namespace complex_number_equation_l249_249381

theorem complex_number_equation
  (f : ℂ → ℂ)
  (z : ℂ)
  (h : f (i - z) = 2 * z - i) :
  (1 - i) * f (2 - i) = -1 + 7 * i := by
  sorry

end complex_number_equation_l249_249381


namespace train_length_at_constant_acceleration_l249_249331

variables (u : ℝ) (t : ℝ) (a : ℝ) (s : ℝ)

theorem train_length_at_constant_acceleration (h₁ : u = 16.67) (h₂ : t = 30) : 
  s = u * t + 0.5 * a * t^2 :=
sorry

end train_length_at_constant_acceleration_l249_249331


namespace polynomial_sum_is_2_l249_249067

theorem polynomial_sum_is_2 :
  ∀ (x : ℝ),
  ∃ (A B C D : ℝ), 
  (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D ∧ A + B + C + D = 2 :=
by
  intros x
  use [4, -10, -13, 21]
  split
  · -- Prove the polynomial expansion
    calc
      (x - 3) * (4 * x^2 + 2 * x - 7) 
          = x * (4 * x^2 + 2 * x - 7) - 3 * (4 * x^2 + 2 * x - 7) : by rw mul_sub
      ... = (x * 4 * x^2 + x * 2 * x - x * 7) - (3 * (4 * x^2) + 3 * (2 * x) - 3 * (-7)) : by distribute
      ... = 4 * x^3 + 2 * x^2 - 7 * x - 12 * x^2 - 6 * x + 21 : by algebra
      ... = 4 * x^3 - 10 * x^2 - 13 * x + 21 : by linarith
  · -- Prove A + B + C + D = 2
    calc
      4 + (-10) + (-13) + 21 = 2 : by linarith

end polynomial_sum_is_2_l249_249067


namespace damaged_books_count_l249_249631

variables (o d : ℕ)

theorem damaged_books_count (h1 : o + d = 69) (h2 : o = 6 * d - 8) : d = 11 := 
by 
  sorry

end damaged_books_count_l249_249631


namespace total_students_l249_249833

def numStudents (skiing scavenger : ℕ) : ℕ :=
  skiing + scavenger

theorem total_students (skiing scavenger : ℕ) (h1 : skiing = 2 * scavenger) (h2 : scavenger = 4000) :
  numStudents skiing scavenger = 12000 :=
by
  sorry

end total_students_l249_249833


namespace Yella_last_week_usage_l249_249149

/-- 
Yella's computer usage last week was some hours. If she plans to use the computer 8 hours a day for this week, 
her computer usage for this week is 35 hours less. Given these conditions, prove that Yella's computer usage 
last week was 91 hours.
-/
theorem Yella_last_week_usage (daily_usage : ℕ) (days_in_week : ℕ) (difference : ℕ)
  (h1: daily_usage = 8)
  (h2: days_in_week = 7)
  (h3: difference = 35) :
  daily_usage * days_in_week + difference = 91 := 
by
  sorry

end Yella_last_week_usage_l249_249149


namespace sin_300_eq_neg_sqrt3_div_2_l249_249568

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249568


namespace resultant_after_trebled_l249_249167

variable (x : ℕ)

theorem resultant_after_trebled (h : x = 7) : 3 * (2 * x + 9) = 69 := by
  sorry

end resultant_after_trebled_l249_249167


namespace constant_term_expansion_l249_249278

theorem constant_term_expansion :
  (∃ c : ℤ, ∀ x : ℝ, (2 * x - 1 / x) ^ 4 = c * x^0) ∧ c = 24 :=
by
  sorry

end constant_term_expansion_l249_249278


namespace ab_value_l249_249328

theorem ab_value (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 18) : a * b = -1 :=
by {
  sorry
}

end ab_value_l249_249328


namespace find_product_xy_l249_249779

theorem find_product_xy (x y : ℝ) 
  (h1 : (9 + 10 + 11 + x + y) / 5 = 10)
  (h2 : ((9 - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (x - 10)^2 + (y - 10)^2) / 5 = 4) :
  x * y = 191 :=
sorry

end find_product_xy_l249_249779


namespace length_of_AB_l249_249133

-- Conditions:
-- The radius of the inscribed circle is 6 cm.
-- The triangle is a right triangle with a 60 degree angle at one vertex.
-- Question: Prove that the length of AB is 12 + 12√3 cm.

theorem length_of_AB (r : ℝ) (angle : ℝ) (h_radius : r = 6) (h_angle : angle = 60) :
  ∃ (AB : ℝ), AB = 12 + 12 * Real.sqrt 3 :=
by
  sorry

end length_of_AB_l249_249133


namespace ordered_pair_represents_5_1_l249_249237

structure OrderedPair (α : Type) :=
  (fst : α)
  (snd : α)

def represents_rows_cols (pair : OrderedPair ℝ) (rows cols : ℕ) : Prop :=
  pair.fst = rows ∧ pair.snd = cols

theorem ordered_pair_represents_5_1 :
  represents_rows_cols (OrderedPair.mk 2 3) 2 3 →
  represents_rows_cols (OrderedPair.mk 5 1) 5 1 :=
by
  intros h
  sorry

end ordered_pair_represents_5_1_l249_249237


namespace sin_300_eq_neg_sqrt_three_div_two_l249_249546

theorem sin_300_eq_neg_sqrt_three_div_two :
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt_three_div_two_l249_249546


namespace question1_question2_l249_249947

theorem question1 (m : ℝ) (x : ℝ) :
  (∀ x, x^2 - m * x + (m - 1) ≥ 0) → m = 2 :=
by
  sorry

theorem question2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (n = (a + 1 / b) * (2 * b + 1 / (2 * a))) → n ≥ (9 / 2) :=
by
  sorry

end question1_question2_l249_249947


namespace statements_imply_conditions_l249_249623

-- Definitions for each condition
def statement1 (p q : Prop) : Prop := ¬p ∧ ¬q
def statement2 (p q : Prop) : Prop := ¬p ∧ q
def statement3 (p q : Prop) : Prop := p ∧ ¬q
def statement4 (p q : Prop) : Prop := p ∧ q

-- Definition for the exclusive condition
def exclusive_condition (p q : Prop) : Prop := ¬(p ∧ q)

theorem statements_imply_conditions (p q : Prop) :
  (statement1 p q → exclusive_condition p q) ∧
  (statement2 p q → exclusive_condition p q) ∧
  (statement3 p q → exclusive_condition p q) ∧
  ¬(statement4 p q → exclusive_condition p q) →
  3 = 3 :=
by
  sorry

end statements_imply_conditions_l249_249623


namespace tan_of_13pi_over_6_l249_249201

theorem tan_of_13pi_over_6 : Real.tan (13 * Real.pi / 6) = 1 / Real.sqrt 3 := by
  sorry

end tan_of_13pi_over_6_l249_249201


namespace makeup_palette_cost_l249_249259

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

end makeup_palette_cost_l249_249259


namespace find_directrix_of_parabola_l249_249664

open Real

theorem find_directrix_of_parabola (O : ℝ × ℝ) (p : ℝ) (F P Q : ℝ × ℝ)
  (hO : O = (0, 0))
  (hp_pos : p > 0)
  (hC : ∀ x y, (x, y) = P → y^2 = 2 * p * x)
  (hF : F = (p / 2, 0))
  (hPF_perpendicular_to_x : P.1 = p / 2 ∧ P.2 = p)
  (hQ_on_x_axis : Q.2 = 0)
  (hPQ_perpendicular_OP : (P.1, P.2) ≠ Q ∧ ((P.2 - Q.2) / (P.1 - Q.1) = -1 / ((P.2 - O.2) / (P.1 - O.1))))
  (hFQ_distance : abs (F.1 - Q.1) = 6) :
  x = -3 / 2 :=
sorry

end find_directrix_of_parabola_l249_249664


namespace sin_300_l249_249600

-- Definitions and conditions in Lean 4
def angle_deg := ℝ
def sin (θ : angle_deg) : ℝ := Real.sin (θ * Real.pi / 180)

noncomputable def Q_angle : angle_deg := 300
noncomputable def first_quadrant_equivalent_angle : angle_deg := 360 - 300
noncomputable def sin_60 : ℝ := Real.sin (60 * Real.pi / 180)

-- Main statement to prove
theorem sin_300 : sin 300 = -sin_60 := by
  sorry

end sin_300_l249_249600


namespace length_of_train_l249_249475

theorem length_of_train (speed : ℝ) (time : ℝ) (h1: speed = 48 * (1000 / 3600) * (1 / 1)) (h2: time = 9) : 
  (speed * time) = 119.97 :=
by
  sorry

end length_of_train_l249_249475


namespace toy_ratio_l249_249975

variable (Jaxon : ℕ) (Gabriel : ℕ) (Jerry : ℕ)

theorem toy_ratio (h1 : Jerry = Gabriel + 8) 
                  (h2 : Jaxon = 15)
                  (h3 : Gabriel + Jerry + Jaxon = 83) :
                  Gabriel / Jaxon = 2 := 
by
  sorry

end toy_ratio_l249_249975


namespace parabola_directrix_l249_249662

theorem parabola_directrix 
  (O : ℝ × ℝ) (hO : O = (0,0))
  (p : ℝ) (hp : p > 0)
  (P : ℝ × ℝ) (hP : P.2^2 = 2 * p * P.1)
  (F : ℝ × ℝ) (hF : F = (p / 2, 0))
  (hPF_perpendicular : P.1 = p / 2)
  (Q : ℝ × ℝ) (hQ : Q.2 = 0)
  (hPQ_perpendicular : 2 * (P.1 - 0)/(P.2 - 0) * (Q.2 - P.2)/(Q.1 - P.1) = -1)
  (hFQ_distance : |F.1 - Q.1| = 6) :
  ∃ p : ℝ, p = 3 → ∃ d : ℝ, d = -3 / 2 ∧ Q.1 = d :=
begin
  sorry
end

end parabola_directrix_l249_249662


namespace min_sum_x8y4z_l249_249407

theorem min_sum_x8y4z (x y z : ℝ) (h : 4 / x + 2 / y + 1 / z = 1) : x + 8 * y + 4 * z ≥ 64 := 
sorry

end min_sum_x8y4z_l249_249407


namespace find_natural_number_with_common_divisor_l249_249349

def commonDivisor (a b : ℕ) (d : ℕ) : Prop :=
  d > 1 ∧ d ∣ a ∧ d ∣ b

theorem find_natural_number_with_common_divisor :
  ∃ n : ℕ, (∀ k : ℕ, 0 ≤ k ∧ k ≤ 20 →
    ∃ d : ℕ, commonDivisor (n + k) 30030 d) ∧ n = 9440 :=
by
  sorry

end find_natural_number_with_common_divisor_l249_249349


namespace molecular_weight_CaO_is_56_08_l249_249144

-- Define the atomic weights of Calcium and Oxygen
def atomic_weight_Ca := 40.08 -- in g/mol
def atomic_weight_O := 16.00 -- in g/mol

-- Define the molecular weight of the compound
def molecular_weight_CaO := atomic_weight_Ca + atomic_weight_O

-- State the theorem
theorem molecular_weight_CaO_is_56_08 : molecular_weight_CaO = 56.08 :=
by
  -- The proof will be filled in here
  sorry

end molecular_weight_CaO_is_56_08_l249_249144


namespace sin_300_eq_neg_one_half_l249_249533

-- Definitions from the conditions
def point_on_unit_circle (θ : ℝ) : ℝ × ℝ :=
  (Real.cos (θ), Real.sin (θ))

def foot_of_perpendicular (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, 0)

def is_thirty_sixty_ninety_triangle (P : ℝ × ℝ) : Prop :=
  sqrt(P.1^2 + P.2^2) = 1 ∧ P.1 = sqrt(3) / 2 ∧ P.2 = -1/2

-- Problem statement
theorem sin_300_eq_neg_one_half : Real.sin (5 * Real.pi / 3) = -1/2 := by
  sorry

end sin_300_eq_neg_one_half_l249_249533


namespace trajectory_of_B_l249_249938

-- Define the points and the line for the given conditions
def A : ℝ × ℝ := (3, -1)
def C : ℝ × ℝ := (2, -3)
def D_line (x : ℝ) (y : ℝ) : Prop := 3 * x - y + 1 = 0

-- Define the statement to be proved
theorem trajectory_of_B (x y : ℝ) :
  D_line x y → ∃ Bx By, (3 * Bx - By - 20 = 0) :=
sorry

end trajectory_of_B_l249_249938


namespace sin_300_eq_neg_sqrt3_div_2_l249_249593

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249593


namespace no_three_parabolas_l249_249027

theorem no_three_parabolas (a b c : ℝ) : ¬ (b^2 > 4*a*c ∧ a^2 > 4*b*c ∧ c^2 > 4*a*b) := by
  sorry

end no_three_parabolas_l249_249027


namespace factor_expression_l249_249922

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l249_249922


namespace aziz_age_l249_249181

-- Definitions of the conditions
def year_moved : ℕ := 1982
def years_before_birth : ℕ := 3
def current_year : ℕ := 2021

-- Prove the main statement
theorem aziz_age : current_year - (year_moved + years_before_birth) = 36 :=
by
  sorry

end aziz_age_l249_249181


namespace cloth_cost_price_l249_249853

theorem cloth_cost_price
  (meters_of_cloth : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ)
  (total_profit : ℕ) (total_cost_price : ℕ) (cost_price_per_meter : ℕ) :
  meters_of_cloth = 45 →
  selling_price = 4500 →
  profit_per_meter = 14 →
  total_profit = profit_per_meter * meters_of_cloth →
  total_cost_price = selling_price - total_profit →
  cost_price_per_meter = total_cost_price / meters_of_cloth →
  cost_price_per_meter = 86 :=
by
  intros
  sorry

end cloth_cost_price_l249_249853


namespace rectangle_length_l249_249152

theorem rectangle_length (P L B : ℕ) (hP : P = 500) (hB : B = 100) (hP_eq : P = 2 * (L + B)) : L = 150 :=
by
  sorry

end rectangle_length_l249_249152


namespace tan_half_alpha_third_quadrant_sine_cos_expression_l249_249740

-- Problem (1): Proof for tan(α/2) = -5 given the conditions
theorem tan_half_alpha_third_quadrant (α : ℝ) (h1 : α ∈ Set.Ioo π (3 * π / 2))
  (h2 : Real.sin α = -5/13) :
  Real.tan (α / 2) = -5 := by
  sorry

-- Problem (2): Proof for sin²(π - α) + 2sin(3π/2 + α)cos(π/2 + α) = 8/5 given the condition
theorem sine_cos_expression (α : ℝ) (h : Real.tan α = 2) :
  Real.sin (π - α) ^ 2 + 2 * Real.sin (3 * π / 2 + α) * Real.cos (π / 2 + α) = 8 / 5 := by
  sorry

end tan_half_alpha_third_quadrant_sine_cos_expression_l249_249740


namespace sin_300_eq_neg_sqrt3_div_2_l249_249523

def point_on_unit_circle (θ : ℝ) : ℝ × ℝ :=
⟨Real.cos θ, Real.sin θ⟩

theorem sin_300_eq_neg_sqrt3_div_2 :
  let Q := point_on_unit_circle (300 * Real.pi / 180) in
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
by sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249523


namespace sin_300_eq_neg_sqrt_three_div_two_l249_249545

theorem sin_300_eq_neg_sqrt_three_div_two :
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt_three_div_two_l249_249545


namespace probability_heads_penny_nickel_dime_l249_249115

theorem probability_heads_penny_nickel_dime :
  let total_outcomes := 2^5 in
  let successful_outcomes := 2 * 2 in
  (successful_outcomes : ℝ) / total_outcomes = (1 / 8 : ℝ) :=
by
  sorry

end probability_heads_penny_nickel_dime_l249_249115


namespace min_number_knights_l249_249389

theorem min_number_knights (h1 : ∃ n : ℕ, n = 7) (h2 : ∃ s : ℕ, s = 42) (h3 : ∃ l : ℕ, l = 24) :
  ∃ k : ℕ, k ≥ 0 ∧ k ≤ 7 ∧ k * (7 - k) = 12 ∧ k = 3 :=
by
  sorry

end min_number_knights_l249_249389


namespace susan_ate_candies_l249_249887

theorem susan_ate_candies (candies_tuesday candies_thursday candies_friday candies_left : ℕ) 
  (h_tuesday : candies_tuesday = 3) 
  (h_thursday : candies_thursday = 5) 
  (h_friday : candies_friday = 2) 
  (h_left : candies_left = 4) : candies_tuesday + candies_thursday + candies_friday - candies_left = 6 := by
  sorry

end susan_ate_candies_l249_249887


namespace remainder_sum_division_by_9_l249_249642

theorem remainder_sum_division_by_9 :
  (9151 + 9152 + 9153 + 9154 + 9155 + 9156 + 9157) % 9 = 6 :=
by
  sorry

end remainder_sum_division_by_9_l249_249642


namespace union_complement_inter_l249_249808

noncomputable def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x ≥ 2 }
def N : Set ℝ := { x | -1 ≤ x ∧ x < 5 }

def C_U_M : Set ℝ := U \ M
def M_inter_N : Set ℝ := { x | x ≥ 2 ∧ x < 5 }

theorem union_complement_inter (C_U_M M_inter_N : Set ℝ) :
  C_U_M ∪ M_inter_N = { x | x < 5 } :=
by
  sorry

end union_complement_inter_l249_249808


namespace number_of_ways_to_form_committee_with_president_l249_249160

open Nat

def number_of_ways_to_choose_members (total_members : ℕ) (committee_size : ℕ) (president_required : Bool) : ℕ :=
  if president_required then choose (total_members - 1) (committee_size - 1) else choose total_members committee_size

theorem number_of_ways_to_form_committee_with_president :
  number_of_ways_to_choose_members 30 5 true = 23741 :=
by
  -- Given that total_members = 30, committee_size = 5, and president_required = true,
  -- we need to show that the number of ways to choose the remaining members is 23741.
  sorry

end number_of_ways_to_form_committee_with_president_l249_249160


namespace players_per_group_l249_249449

theorem players_per_group (new_players : ℕ) (returning_players : ℕ) (groups : ℕ) 
  (h1 : new_players = 48) 
  (h2 : returning_players = 6) 
  (h3 : groups = 9) : 
  (new_players + returning_players) / groups = 6 :=
by
  sorry

end players_per_group_l249_249449


namespace dwayneA_students_l249_249794

-- Define the number of students who received an 'A' in Mrs. Carter's class
def mrsCarterA := 8
-- Define the total number of students in Mrs. Carter's class
def mrsCarterTotal := 20
-- Define the total number of students in Mr. Dwayne's class
def mrDwayneTotal := 30
-- Calculate the ratio of students who received an 'A' in Mrs. Carter's class
def carterRatio := mrsCarterA / mrsCarterTotal
-- Calculate the number of students who received an 'A' in Mr. Dwayne's class based on the same ratio
def mrDwayneA := (carterRatio * mrDwayneTotal)

-- Prove that the number of students who received an 'A' in Mr. Dwayne's class is 12
theorem dwayneA_students :
  mrDwayneA = 12 := 
by
  -- Since def calculation does not automatically prove equality, we will need to use sorry to skip the proof for now.
  sorry

end dwayneA_students_l249_249794


namespace intersection_points_of_quadratic_minimum_value_of_quadratic_in_range_range_of_m_for_intersection_with_segment_PQ_l249_249041

-- Define the quadratic function
def quadratic (m x : ℝ) : ℝ := m * x^2 - 4 * m * x + 3 * m

-- Define the conditions
variables (m : ℝ)
theorem intersection_points_of_quadratic :
    (quadratic m 1 = 0) ∧ (quadratic m 3 = 0) ↔ m ≠ 0 :=
sorry

theorem minimum_value_of_quadratic_in_range :
    ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 4 → quadratic (-2) x ≥ -6 :=
sorry

theorem range_of_m_for_intersection_with_segment_PQ :
    ∀ (m : ℝ), (∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ quadratic m x = (m + 4) / 2) ↔ 
    m ≤ -4 / 3 ∨ m ≥ 4 / 5 :=
sorry

end intersection_points_of_quadratic_minimum_value_of_quadratic_in_range_range_of_m_for_intersection_with_segment_PQ_l249_249041


namespace factor_expression_l249_249913

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := 
by
  sorry

end factor_expression_l249_249913


namespace sin_300_eq_neg_sqrt3_div_2_l249_249583

noncomputable def sin_300 : ℝ := sin (300 * real.pi / 180)

theorem sin_300_eq_neg_sqrt3_div_2 : 
  (∠300° = 360° - 60°) →
  (300° ∈ fourth_quadrant) →
  (∀ θ ∈ fourth_quadrant, sin θ < 0) →
  (sin (60 * real.pi / 180) = sqrt 3 / 2) →
  sin_300 = -sqrt 3 / 2 := 
by
  intros h1 h2 h3 h4
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249583


namespace cat_collars_needed_l249_249698

-- Define the given constants
def nylon_per_dog_collar : ℕ := 18
def nylon_per_cat_collar : ℕ := 10
def total_nylon : ℕ := 192
def dog_collars : ℕ := 9

-- Compute the number of cat collars needed
theorem cat_collars_needed : (total_nylon - (dog_collars * nylon_per_dog_collar)) / nylon_per_cat_collar = 3 :=
by
  sorry

end cat_collars_needed_l249_249698


namespace sin_300_eq_neg_sqrt3_div_2_l249_249620

-- Let Q be the point on the unit circle at 300 degrees counterclockwise from (1,0)
def Q : ℝ × ℝ := (1 / 2, -Real.sqrt 3 / 2)

-- Prove that the sine of 300 degrees is -√3/2
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  -- Here we assume the coordinates given in the conditions are correct.
  have hQ : Q = (1 / 2, -Real.sqrt 3 / 2) := rfl
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249620


namespace trigonometric_identity_l249_249765

theorem trigonometric_identity : 
  (Real.cos (15 * Real.pi / 180) * Real.cos (105 * Real.pi / 180) - Real.cos (75 * Real.pi / 180) * Real.sin (105 * Real.pi / 180))
  = -1 / 2 :=
by
  sorry

end trigonometric_identity_l249_249765


namespace problem_solution_l249_249320

noncomputable def P_conditional (A B : Set Ω) := P (A ∩ B) / P B

variables (Ω : Type) [Fintype Ω] (questions : Set Ω)
variables (mcqs fillintheblanks : Set Ω)
variables (A B : Set Ω)
variables [DecidableEq Ω]

def is_mcq (q : Ω) : Prop := q ∈ mcqs
def is_fill_in_the_blank (q : Ω) : Prop := q ∈ fillintheblanks

-- Conditions
axiom condition1 : mcqs.card = 3
axiom condition2 : fillintheblanks.card = 2
axiom condition3 : A = {q | is_mcq q}
axiom condition4 : B = {q | is_fill_in_the_blank q}

theorem problem_solution : P_conditional Ω A B = 3 / 4 := sorry

end problem_solution_l249_249320


namespace range_of_a_l249_249675

variable (a : ℝ)

def set_A (a : ℝ) : Set ℝ := { x | x^2 - 2 * x + a ≥ 0 }

theorem range_of_a (h : 1 ∉ set_A a) : a < 1 := 
by {
  sorry
}

end range_of_a_l249_249675


namespace impossible_to_arrange_circle_l249_249026

theorem impossible_to_arrange_circle : 
  ¬∃ (f : Fin 10 → Fin 10), 
    (∀ i : Fin 10, (abs ((f i).val - (f (i + 1)).val : Int) = 3 
                ∨ abs ((f i).val - (f (i + 1)).val : Int) = 4 
                ∨ abs ((f i).val - (f (i + 1)).val : Int) = 5)) :=
sorry

end impossible_to_arrange_circle_l249_249026


namespace sum_of_three_numbers_l249_249830

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) 
 (h_median : b = 10) 
 (h_mean_least : (a + b + c) / 3 = a + 8)
 (h_mean_greatest : (a + b + c) / 3 = c - 20) : 
 a + b + c = 66 :=
by 
  sorry

end sum_of_three_numbers_l249_249830


namespace octagon_side_length_eq_l249_249391

theorem octagon_side_length_eq (AB BC : ℝ) (AE FB s : ℝ) :
  AE = FB → AE < 5 → AB = 10 → BC = 12 →
  s = -11 + Real.sqrt 242 →
  EF = (10.5 - (Real.sqrt 242) / 2) :=
by
  -- Identified parameters and included all conditions from step a)
  intros h1 h2 h3 h4 h5
  -- statement of the theorem to be proven
  let EF := (10.5 - (Real.sqrt 242) / 2)
  sorry  -- placeholder for proof

end octagon_side_length_eq_l249_249391


namespace sin_300_eq_neg_sqrt3_div_2_l249_249589

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249589


namespace prince_spent_1000_l249_249019

def total_cds : ℕ := 200
def percentage_ten_dollars : ℚ := 0.40
def percentage_five_dollars : ℚ := 0.60
def price_ten_dollars : ℚ := 10
def price_five_dollars : ℚ := 5
def prince_share_ten_dollars : ℚ := 0.50

def count_ten_dollar_cds : ℕ := (percentage_ten_dollars * total_cds).to_nat
def count_five_dollar_cds : ℕ := (percentage_five_dollars * total_cds).to_nat
def count_prince_ten_dollar_cds : ℕ := (prince_share_ten_dollars * count_ten_dollar_cds).to_nat
def count_prince_five_dollar_cds : ℕ := count_five_dollar_cds

def total_money_spent (count_ten count_five : ℕ) : ℚ := (count_ten * price_ten_dollars) + (count_five * price_five_dollars)

theorem prince_spent_1000 :
  total_money_spent count_prince_ten_dollar_cds count_prince_five_dollar_cds = 1000 := by
  sorry

end prince_spent_1000_l249_249019


namespace frac_pattern_2_11_frac_pattern_general_l249_249045

theorem frac_pattern_2_11 :
  (2 / 11) = (1 / 6) + (1 / 66) :=
sorry

theorem frac_pattern_general (n : ℕ) (hn : n ≥ 3) :
  (2 / (2 * n - 1)) = (1 / n) + (1 / (n * (2 * n - 1))) :=
sorry

end frac_pattern_2_11_frac_pattern_general_l249_249045


namespace alex_walking_distance_l249_249959

theorem alex_walking_distance
  (distance : ℝ)
  (time_45 : ℝ)
  (walking_rate : distance = 1.5 ∧ time_45 = 45):
  ∃ distance_90, distance_90 = 3 :=
by 
  sorry

end alex_walking_distance_l249_249959


namespace sin_300_eq_neg_sqrt3_div_2_l249_249621

-- Let Q be the point on the unit circle at 300 degrees counterclockwise from (1,0)
def Q : ℝ × ℝ := (1 / 2, -Real.sqrt 3 / 2)

-- Prove that the sine of 300 degrees is -√3/2
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  -- Here we assume the coordinates given in the conditions are correct.
  have hQ : Q = (1 / 2, -Real.sqrt 3 / 2) := rfl
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249621


namespace simplify_expression_l249_249450

open Real

-- Assuming lg refers to the common logarithm log base 10
noncomputable def problem_expression : ℝ :=
  log 4 + 2 * log 5 + 4^(-1/2:ℝ)

theorem simplify_expression : problem_expression = 5 / 2 :=
by
  -- Placeholder proof, actual steps not required
  sorry

end simplify_expression_l249_249450


namespace solve_system_l249_249993

variables (a b c d : ℝ)

theorem solve_system :
  (a + c = -4) ∧
  (a * c + b + d = 6) ∧
  (a * d + b * c = -5) ∧
  (b * d = 2) →
  ((a = -3 ∧ b = 2 ∧ c = -1 ∧ d = 1) ∨
   (a = -1 ∧ b = 1 ∧ c = -3 ∧ d = 2)) :=
by
  intro h
  -- Insert proof here
  sorry

end solve_system_l249_249993


namespace geo_seq_a3_equals_one_l249_249364

theorem geo_seq_a3_equals_one (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) (h_T5 : a 1 * a 2 * a 3 * a 4 * a 5 = 1) : a 3 = 1 :=
sorry

end geo_seq_a3_equals_one_l249_249364


namespace notebooks_have_50_pages_l249_249396

theorem notebooks_have_50_pages (notebooks : ℕ) (total_dollars : ℕ) (page_cost_cents : ℕ) 
  (total_cents : ℕ) (total_pages : ℕ) (pages_per_notebook : ℕ)
  (h1 : notebooks = 2) 
  (h2 : total_dollars = 5) 
  (h3 : page_cost_cents = 5) 
  (h4 : total_cents = total_dollars * 100) 
  (h5 : total_pages = total_cents / page_cost_cents) 
  (h6 : pages_per_notebook = total_pages / notebooks) 
  : pages_per_notebook = 50 :=
by
  sorry

end notebooks_have_50_pages_l249_249396


namespace remaining_nails_after_repairs_l249_249867

def fraction_used (perc : ℤ) (total : ℤ) : ℤ :=
  (total * perc) / 100

def after_kitchen (nails : ℤ) : ℤ :=
  nails - fraction_used 35 nails

def after_fence (nails : ℤ) : ℤ :=
  let remaining := after_kitchen nails
  remaining - fraction_used 75 remaining

def after_table (nails : ℤ) : ℤ :=
  let remaining := after_fence nails
  remaining - fraction_used 55 remaining

def after_floorboard (nails : ℤ) : ℤ :=
  let remaining := after_table nails
  remaining - fraction_used 30 remaining

theorem remaining_nails_after_repairs :
  after_floorboard 400 = 21 :=
by
  sorry

end remaining_nails_after_repairs_l249_249867


namespace area_of_R3_l249_249171

theorem area_of_R3 (r1 r2 r3 : ℝ) (h1: r1^2 = 25) 
                   (h2: r2 = (2/3) * r1) (h3: r3 = (2/3) * r2) :
                   r3^2 = 400 / 81 := 
by
  sorry

end area_of_R3_l249_249171


namespace heat_production_example_l249_249028

noncomputable def heat_produced_by_current (R : ℝ) (I : ℝ → ℝ) (t1 t2 : ℝ) : ℝ :=
∫ (t : ℝ) in t1..t2, (I t)^2 * R

theorem heat_production_example :
  heat_produced_by_current 40 (λ t => 5 + 4 * t) 0 10 = 303750 :=
by
  sorry

end heat_production_example_l249_249028


namespace susan_ate_6_candies_l249_249882

-- Definitions for the conditions
def candies_tuesday : ℕ := 3
def candies_thursday : ℕ := 5
def candies_friday : ℕ := 2
def candies_left : ℕ := 4

-- The total number of candies bought during the week
def total_candies_bought : ℕ := candies_tuesday + candies_thursday + candies_friday

-- The number of candies Susan ate during the week
def candies_eaten : ℕ := total_candies_bought - candies_left

-- Theorem statement
theorem susan_ate_6_candies : candies_eaten = 6 :=
by
  unfold candies_eaten total_candies_bought candies_tuesday candies_thursday candies_friday candies_left
  sorry

end susan_ate_6_candies_l249_249882


namespace unique_integer_for_P5_l249_249753

-- Define the polynomial P with integer coefficients
variable (P : ℤ → ℤ)

-- The conditions given in the problem
variable (x1 x2 x3 : ℤ)
variable (Hx1 : P x1 = 1)
variable (Hx2 : P x2 = 2)
variable (Hx3 : P x3 = 3)

-- The main theorem to prove
theorem unique_integer_for_P5 {P : ℤ → ℤ} {x1 x2 x3 : ℤ}
(Hx1 : P x1 = 1) (Hx2 : P x2 = 2) (Hx3 : P x3 = 3) :
  ∃!(x : ℤ), P x = 5 := sorry

end unique_integer_for_P5_l249_249753


namespace inequality_proof_l249_249043

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (h : a + b + c = 1)

theorem inequality_proof :
  (1 + a) / (1 - a) + (1 + b) / (1 - a) + (1 + c) / (1 - c) ≤ 2 * ((b / a) + (c / b) + (a / c)) :=
by sorry

end inequality_proof_l249_249043


namespace sin_300_eq_neg_sqrt_3_div_2_l249_249509

theorem sin_300_eq_neg_sqrt_3_div_2 : Real.sin (300 * Real.pi / 180) = - Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt_3_div_2_l249_249509


namespace ab_value_l249_249666

   variable (log2_3 : Real) (b : Real) (a : Real)

   -- Hypotheses
   def log_condition : Prop := log2_3 = 1
   def exp_condition (b : Real) : Prop := (4:Real) ^ b = 3
   
   -- Final statement to prove
   theorem ab_value (h_log2_3 : log_condition log2_3) (h_exp : exp_condition b) 
   (ha : a = 1) : a * b = 1 / 2 := sorry
   
end ab_value_l249_249666


namespace common_difference_range_l249_249077

variable (d : ℝ)

def a (n : ℕ) : ℝ := -5 + (n - 1) * d

theorem common_difference_range (H1 : a 10 > 0) (H2 : a 9 ≤ 0) :
  (5 / 9 < d) ∧ (d ≤ 5 / 8) :=
by
  sorry

end common_difference_range_l249_249077


namespace no_five_consecutive_divisible_by_2005_l249_249674

def seq (n : ℕ) : ℕ := 1 + 2^n + 3^n + 4^n + 5^n

theorem no_five_consecutive_divisible_by_2005 :
  ¬ (∃ m : ℕ, ∀ k : ℕ, k < 5 → (seq (m + k)) % 2005 = 0) :=
sorry

end no_five_consecutive_divisible_by_2005_l249_249674


namespace min_value_sqrt_expression_l249_249203

open Real

theorem min_value_sqrt_expression : ∃ x : ℝ, ∀ y : ℝ, 
  sqrt (y^2 + (2 - y)^2) + sqrt ((y - 1)^2 + (y + 2)^2) ≥ sqrt 17 :=
by
  sorry

end min_value_sqrt_expression_l249_249203


namespace max_area_triangle_bqc_l249_249081

noncomputable def triangle_problem : ℝ :=
  let a := 112.5
  let b := 56.25
  let c := 3
  a + b + c

theorem max_area_triangle_bqc : triangle_problem = 171.75 :=
by
  -- The proof would involve validating the steps to ensure the computations
  -- for the maximum area of triangle BQC match the expression 112.5 - 56.25 √3,
  -- and thus confirm that a = 112.5, b = 56.25, c = 3
  -- and verifying that a + b + c = 171.75.
  sorry

end max_area_triangle_bqc_l249_249081


namespace find_f_neg2016_l249_249207

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f_neg2016 (a b k : ℝ) (h : f a b 2016 = k) (h_ab : a * b ≠ 0) : f a b (-2016) = 2 - k :=
by
  sorry

end find_f_neg2016_l249_249207


namespace find_m_l249_249215

-- Define the function and conditions
def power_function (x : ℝ) (m : ℕ) : ℝ := x^(m - 2)

theorem find_m (m : ℕ) (x : ℝ) (h1 : 0 < m) (h2 : power_function 0 m = 0 → false) : m = 1 ∨ m = 2 :=
by
  sorry -- Skip the proof

end find_m_l249_249215


namespace greatest_possible_integer_radius_l249_249682

theorem greatest_possible_integer_radius :
  ∃ r : ℤ, (50 < (r : ℝ)^2) ∧ ((r : ℝ)^2 < 75) ∧ 
  (∀ s : ℤ, (50 < (s : ℝ)^2) ∧ ((s : ℝ)^2 < 75) → s ≤ r) :=
sorry

end greatest_possible_integer_radius_l249_249682


namespace trapezoid_area_l249_249334

-- Define the properties of the isosceles trapezoid
structure IsoscelesTrapezoid where
  leg : ℝ
  diagonal : ℝ
  longer_base : ℝ
  is_isosceles : True
  legs_equal : True

-- Provide the specific conditions of the problem
def trapezoid : IsoscelesTrapezoid := {
  leg := 40,
  diagonal := 50,
  longer_base := 60,
  is_isosceles := True.intro,
  legs_equal := True.intro
}

-- State the main theorem to translate the proof problem into Lean
theorem trapezoid_area (T : IsoscelesTrapezoid) : T = trapezoid →
  (∃ A : ℝ, A = (15000 - 2000 * Real.sqrt 11) / 9) :=
by
  intros h
  sorry

end trapezoid_area_l249_249334


namespace sin_300_eq_neg_sin_60_l249_249604

theorem sin_300_eq_neg_sin_60 :
  Real.sin (300 * Real.pi / 180) = - Real.sin (60 * Real.pi / 180) :=
by
suffices : Real.sin (300 * Real.pi / 180) = -sqrt 3 / 2
  sorry

end sin_300_eq_neg_sin_60_l249_249604


namespace susan_ate_candies_l249_249886

theorem susan_ate_candies (candies_tuesday candies_thursday candies_friday candies_left : ℕ) 
  (h_tuesday : candies_tuesday = 3) 
  (h_thursday : candies_thursday = 5) 
  (h_friday : candies_friday = 2) 
  (h_left : candies_left = 4) : candies_tuesday + candies_thursday + candies_friday - candies_left = 6 := by
  sorry

end susan_ate_candies_l249_249886


namespace task1_on_time_task2_not_on_time_l249_249466

/-- Define the probabilities for task 1 and task 2 -/
def P_A : ℚ := 3 / 8
def P_B : ℚ := 3 / 5

/-- The probability that task 1 will be completed on time but task 2 will not is 3 / 20. -/
theorem task1_on_time_task2_not_on_time (P_A : ℚ) (P_B : ℚ) : P_A = 3 / 8 → P_B = 3 / 5 → P_A * (1 - P_B) = 3 / 20 :=
by
  intros hPA hPB
  rw [hPA, hPB]
  norm_num

end task1_on_time_task2_not_on_time_l249_249466


namespace sin_300_deg_l249_249490

noncomputable def sin_300_equals_neg_sqrt3_div_2 : Prop :=
  sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2

theorem sin_300_deg : sin_300_equals_neg_sqrt3_div_2 :=
by
  sorry

end sin_300_deg_l249_249490


namespace largest_five_digit_divisible_by_97_l249_249299

theorem largest_five_digit_divisible_by_97 :
  ∃ n, (99999 - n % 97) = 99930 ∧ n % 97 = 0 ∧ 10000 ≤ n ∧ n ≤ 99999 :=
by
  sorry

end largest_five_digit_divisible_by_97_l249_249299


namespace fraction_e_over_d_l249_249841

theorem fraction_e_over_d :
  ∃ (d e : ℝ), (∀ (x : ℝ), x^2 + 2600 * x + 2600 = (x + d)^2 + e) ∧ e / d = -1298 :=
by 
  sorry

end fraction_e_over_d_l249_249841


namespace fractions_correct_l249_249076
-- Broader import to ensure all necessary libraries are included.

-- Definitions of the conditions
def batman_homes_termite_ridden : ℚ := 1/3
def batman_homes_collapsing : ℚ := 7/10 * batman_homes_termite_ridden
def robin_homes_termite_ridden : ℚ := 3/7
def robin_homes_collapsing : ℚ := 4/5 * robin_homes_termite_ridden
def joker_homes_termite_ridden : ℚ := 1/2
def joker_homes_collapsing : ℚ := 3/8 * joker_homes_termite_ridden

-- Definitions of the fractions of homes that are termite-ridden but not collapsing
def batman_non_collapsing_fraction : ℚ := batman_homes_termite_ridden - batman_homes_collapsing
def robin_non_collapsing_fraction : ℚ := robin_homes_termite_ridden - robin_homes_collapsing
def joker_non_collapsing_fraction : ℚ := joker_homes_termite_ridden - joker_homes_collapsing

-- Proof statement
theorem fractions_correct :
  batman_non_collapsing_fraction = 1/10 ∧
  robin_non_collapsing_fraction = 3/35 ∧
  joker_non_collapsing_fraction = 5/16 :=
sorry

end fractions_correct_l249_249076


namespace part_1_conditions_part_2_min_value_l249_249954

theorem part_1_conditions
  (a b x : ℝ)
  (h1: 2 * a * x^2 - 8 * x - 3 * a^2 < 0)
  (h2: ∀ x, -1 < x -> x < b)
  : a = 2 ∧ b = 3 := sorry

theorem part_2_min_value
  (a b x y : ℝ)
  (h1: x > 0)
  (h2: y > 0)
  (h3: a = 2)
  (h4: b = 3)
  (h5: (a / x) + (b / y) = 1)
  : ∃ min_val : ℝ, min_val = 3 * x + 2 * y ∧ min_val = 24 := sorry

end part_1_conditions_part_2_min_value_l249_249954


namespace find_a_plus_d_l249_249951

variables (a b c d e : ℝ)

theorem find_a_plus_d :
  a + b = 12 ∧ b + c = 9 ∧ c + d = 3 ∧ d + e = 7 ∧ e + a = 10 → a + d = 6 :=
by
  intros h
  have h1 : a + b = 12 := h.1
  have h2 : b + c = 9 := h.2.1
  have h3 : c + d = 3 := h.2.2.1
  have h4 : d + e = 7 := h.2.2.2.1
  have h5 : e + a = 10 := h.2.2.2.2
  sorry

end find_a_plus_d_l249_249951


namespace sin_300_eq_neg_sqrt3_div_2_l249_249580

noncomputable def sin_300 : ℝ := sin (300 * real.pi / 180)

theorem sin_300_eq_neg_sqrt3_div_2 : 
  (∠300° = 360° - 60°) →
  (300° ∈ fourth_quadrant) →
  (∀ θ ∈ fourth_quadrant, sin θ < 0) →
  (sin (60 * real.pi / 180) = sqrt 3 / 2) →
  sin_300 = -sqrt 3 / 2 := 
by
  intros h1 h2 h3 h4
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249580


namespace sin_300_eq_neg_sqrt3_div_2_l249_249574

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249574


namespace total_books_in_classroom_l249_249964

-- Define the given conditions using Lean definitions
def num_children : ℕ := 15
def books_per_child : ℕ := 12
def additional_books : ℕ := 22

-- Define the hypothesis and the corresponding proof statement
theorem total_books_in_classroom : num_children * books_per_child + additional_books = 202 := 
by sorry

end total_books_in_classroom_l249_249964


namespace equivalent_single_discount_l249_249110

theorem equivalent_single_discount :
  ∀ (x : ℝ), ((1 - 0.15) * (1 - 0.10) * (1 - 0.05) * x) = (1 - 0.273) * x :=
by
  intros x
  --- This proof is left blank intentionally.
  sorry

end equivalent_single_discount_l249_249110


namespace max_sum_real_part_l249_249409

-- This would likely be useful for complex number root calculations
noncomputable def max_real_part (w : ℕ → ℂ) : ℝ :=
∑ j in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, (w j).re

theorem max_sum_real_part :
  ∃ w_j : ℕ → ℂ, (∀ j, w_j j = z_j j ∨ w_j j = -complex.I * z_j j) ∧ max_real_part w_j = 64 * real.sqrt 5 :=
sorry

end max_sum_real_part_l249_249409


namespace height_of_box_l249_249163

-- Definitions of given conditions
def length_box : ℕ := 9
def width_box : ℕ := 12
def num_cubes : ℕ := 108
def volume_cube : ℕ := 3
def volume_box : ℕ := num_cubes * volume_cube  -- Volume calculated from number of cubes and volume of each cube

-- The statement to prove
theorem height_of_box : 
  ∃ h : ℕ, volume_box = length_box * width_box * h ∧ h = 3 := by
  sorry

end height_of_box_l249_249163


namespace max_min_values_l249_249353

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_values :
  (∀ x ∈ (Set.Icc 0 2), f x ≤ 5) ∧ (∃ x ∈ (Set.Icc 0 2), f x = 5) ∧
  (∀ x ∈ (Set.Icc 0 2), f x ≥ -15) ∧ (∃ x ∈ (Set.Icc 0 2), f x = -15) :=
by
  sorry

end max_min_values_l249_249353


namespace sin_300_eq_neg_sqrt3_div_2_l249_249555

/-- Prove that the sine of 300 degrees is equal to -√3/2. -/
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249555


namespace probability_heads_penny_nickel_dime_is_one_eighth_l249_249122

-- Define the setup: flipping 5 coins
def five_coins : Type := (bool × bool × bool × bool × bool)

-- Define the condition: penny, nickel, and dime are heads
def heads_penny_nickel_dime (c: five_coins) : bool := c.1 && c.2 && c.3

-- Define the successful outcomes where penny, nickel, and dime are heads
noncomputable def successful_outcomes : Set five_coins :=
  { c | heads_penny_nickel_dime c = tt }

-- Define the total outcomes for 5 coins
noncomputable def total_outcomes : Set five_coins := {c | true}

-- Probability calculation: |success| / |total|
noncomputable def probability_heads_penny_nickel_dime : real :=
  (Set.card successful_outcomes : real) / (Set.card total_outcomes : real)

-- Given the setup, prove that the probability is 1/8
theorem probability_heads_penny_nickel_dime_is_one_eighth :
  probability_heads_penny_nickel_dime = 1 / 8 := 
sorry

end probability_heads_penny_nickel_dime_is_one_eighth_l249_249122


namespace total_cost_shoes_and_jerseys_l249_249397

theorem total_cost_shoes_and_jerseys 
  (shoes : ℕ) (jerseys : ℕ) (cost_shoes : ℕ) (cost_jersey : ℕ) 
  (cost_total_shoes : ℕ) (cost_per_shoe : ℕ) (cost_per_jersey : ℕ) 
  (h1 : shoes = 6)
  (h2 : jerseys = 4) 
  (h3 : cost_per_jersey = cost_per_shoe / 4)
  (h4 : cost_total_shoes = 480)
  (h5 : cost_per_shoe = cost_total_shoes / shoes)
  (h6 : cost_per_jersey = cost_per_shoe / 4)
  (total_cost : ℕ) 
  (h7 : total_cost = cost_total_shoes + cost_per_jersey * jerseys) :
  total_cost = 560 :=
sorry

end total_cost_shoes_and_jerseys_l249_249397


namespace multiple_of_p_l249_249238

variable {p q : ℚ}
variable (m : ℚ)

theorem multiple_of_p (h1 : p / q = 3 / 11) (h2 : m * p + q = 17) : m = 2 :=
by sorry

end multiple_of_p_l249_249238


namespace correct_statements_l249_249424

-- Definitions
def p_A : ℚ := 1 / 2
def p_B : ℚ := 1 / 3

-- Statements to be verified
def statement1 := (p_A * (1 - p_B) + (1 - p_A) * p_B) = (1 / 2 + 1 / 3)
def statement2 := (p_A * p_B) = (1 / 2 * 1 / 3)
def statement3 := (p_A * (1 - p_B) + p_A * p_B) = (1 / 2 * 2 / 3 + 1 / 2 * 1 / 3)
def statement4 := (1 - (1 - p_A) * (1 - p_B)) = (1 - 1 / 2 * 2 / 3)

-- Theorem stating the correct sequence of statements
theorem correct_statements : (statement2 ∧ statement4) ∧ ¬(statement1 ∨ statement3) :=
by
  sorry

end correct_statements_l249_249424


namespace planted_fraction_correct_l249_249200

-- Define the vertices of the triangle
def A : (ℝ × ℝ) := (0, 0)
def B : (ℝ × ℝ) := (5, 0)
def C : (ℝ × ℝ) := (0, 12)

-- Define the length of the legs
def leg1 := 5
def leg2 := 12

-- Define the shortest distance from the square to the hypotenuse
def distance_to_hypotenuse := 3

-- Define the area of the triangle
def triangle_area := (1 / 2) * (leg1 * leg2)

-- Assume the side length of the square
def s := 6 / 13

-- Define the area of the square
def square_area := s^2

-- Define the fraction of the field that is unplanted
def unplanted_fraction := square_area / triangle_area

-- Define the fraction of the field that is planted
def planted_fraction := 1 - unplanted_fraction

theorem planted_fraction_correct :
  planted_fraction = 5034 / 5070 :=
sorry

end planted_fraction_correct_l249_249200


namespace value_of_knife_l249_249141

/-- Two siblings sold their flock of sheep. Each sheep was sold for as many florins as 
the number of sheep originally in the flock. They divided the revenue by giving out 
10 florins at a time. First, the elder brother took 10 florins, then the younger brother, 
then the elder again, and so on. In the end, the younger brother received less than 10 florins, 
so the elder brother gave him his knife, making their earnings equal. 
Prove that the value of the knife in florins is 2. -/
theorem value_of_knife (n : ℕ) (k m : ℕ) (h1 : n^2 = 20 * k + 10 + m) (h2 : 1 ≤ m ∧ m ≤ 9) : 
  (∃ b : ℕ, 10 - b = m + b ∧ b = 2) :=
by
  sorry

end value_of_knife_l249_249141


namespace subset_implication_l249_249955

noncomputable def M (x : ℝ) : Prop := -2 * x + 1 ≥ 0
noncomputable def N (a x : ℝ) : Prop := x < a

theorem subset_implication (a : ℝ) :
  (∀ x, M x → N a x) → a > 1 / 2 :=
by
  sorry

end subset_implication_l249_249955


namespace problem1_problem2_problem3_l249_249667

variables (x y a b c : ℚ)

-- Definition of the operation *
def op_star (x y : ℚ) : ℚ := x * y + 1

-- Prove that 2 * 3 = 7 using the operation *
theorem problem1 : op_star 2 3 = 7 :=
by
  sorry

-- Prove that (1 * 4) * (-1/2) = -3/2 using the operation *
theorem problem2 : op_star (op_star 1 4) (-1/2) = -3/2 :=
by
  sorry

-- Prove the relationship a * (b + c) + 1 = a * b + a * c using the operation *
theorem problem3 : op_star a (b + c) + 1 = op_star a b + op_star a c :=
by
  sorry

end problem1_problem2_problem3_l249_249667


namespace find_natural_triples_l249_249033

open Nat

noncomputable def satisfies_conditions (a b c : ℕ) : Prop :=
  (a + b) % c = 0 ∧ (b + c) % a = 0 ∧ (c + a) % b = 0

theorem find_natural_triples :
  ∀ (a b c : ℕ), satisfies_conditions a b c ↔
    (∃ a, (a = b ∧ b = c) ∨ 
          (a = b ∧ c = 2 * a) ∨ 
          (b = 2 * a ∧ c = 3 * a) ∨ 
          (b = 3 * a ∧ c = 2 * a) ∨ 
          (a = 2 * b ∧ c = 3 * b) ∨ 
          (a = 3 * b ∧ c = 2 * b)) :=
sorry

end find_natural_triples_l249_249033


namespace directrix_equation_of_parabola_l249_249665

theorem directrix_equation_of_parabola (O : Point) (C : Parabola) (p : ℝ) (hp : p > 0) (F P Q : Point) 
  (hC : C = parabola 2 p) 
  (hF : F = (p / 2, 0)) 
  (hP : on_parabola P C) 
  (hPF_perp_xaxis : PF ⊥ x_axis) 
  (hQ_on_xaxis : on_x_axis Q) 
  (hPQ_perp_OP : PQ ⊥ OP) 
  (hFQ : distance F Q = 6) :
  directrix (parabola 2 p) = x = -p/2 :=
sorry

end directrix_equation_of_parabola_l249_249665


namespace triangle_perimeter_l249_249241

theorem triangle_perimeter (A B C : Type) 
  (x : ℝ) 
  (a b c : ℝ) 
  (h₁ : a = x + 1) 
  (h₂ : b = x) 
  (h₃ : c = x - 1) 
  (α β γ : ℝ) 
  (angle_condition : α = 2 * γ) 
  (law_of_sines : a / Real.sin α = c / Real.sin γ)
  (law_of_cosines : Real.cos γ = ((a^2 + b^2 - c^2) / (2 * b * a))) :
  a + b + c = 15 :=
  by
  sorry

end triangle_perimeter_l249_249241


namespace incorrect_statement_B_l249_249952

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + φ)

theorem incorrect_statement_B
  (ω φ : ℝ)
  (h_ω_pos : ω > 0)
  (h_φ_range : 0 < φ ∧ φ < π)
  (h_f_neg_pi_by_10 : f ω φ (-π / 10) = 0)
  (h_f_le_max_at_2pi_by_5 : ∀ x, f ω φ x ≤ |f ω φ (2 * π / 5)|)
  (h_monotonic : ∀ x1 x2, -π / 5 < x1 ∧ x1 < x2 ∧ x2 < π / 10 → f ω φ x1 < f ω φ x2)
  : ¬ (φ = 3 * π / 5) :=
sorry

end incorrect_statement_B_l249_249952


namespace polynomial_remainder_l249_249656

theorem polynomial_remainder (p q r : Polynomial ℝ) (h1 : p.eval 2 = 6) (h2 : p.eval 4 = 14)
  (r_deg : r.degree < 2) :
  p = q * (X - 2) * (X - 4) + r → r = 4 * X - 2 :=
by
  sorry

end polynomial_remainder_l249_249656


namespace sin_300_eq_neg_sqrt3_div_2_l249_249570

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249570


namespace hcl_reaction_l249_249375

theorem hcl_reaction
  (stoichiometry : ∀ (HCl NaHCO3 H2O CO2 NaCl : ℕ), HCl = NaHCO3 ∧ H2O = NaHCO3 ∧ CO2 = NaHCO3 ∧ NaCl = NaHCO3)
  (naHCO3_moles : ℕ)
  (reaction_moles : naHCO3_moles = 3) :
  ∃ (HCl_moles : ℕ), HCl_moles = naHCO3_moles :=
by
  sorry

end hcl_reaction_l249_249375


namespace sin_300_eq_neg_sqrt3_div_2_l249_249524

def point_on_unit_circle (θ : ℝ) : ℝ × ℝ :=
⟨Real.cos θ, Real.sin θ⟩

theorem sin_300_eq_neg_sqrt3_div_2 :
  let Q := point_on_unit_circle (300 * Real.pi / 180) in
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
by sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249524


namespace sum_first_75_odd_numbers_l249_249767

theorem sum_first_75_odd_numbers : (75^2) = 5625 :=
by
  sorry

end sum_first_75_odd_numbers_l249_249767


namespace condition_A_is_necessary_but_not_sufficient_for_condition_B_l249_249659

-- Define conditions
variables (a b : ℝ)

-- Condition A: ab > 0
def condition_A : Prop := a * b > 0

-- Condition B: a > 0 and b > 0
def condition_B : Prop := a > 0 ∧ b > 0

-- Prove that condition_A is a necessary but not sufficient condition for condition_B
theorem condition_A_is_necessary_but_not_sufficient_for_condition_B :
  (condition_A a b → condition_B a b) ∧ ¬(condition_B a b → condition_A a b) :=
by
  sorry

end condition_A_is_necessary_but_not_sufficient_for_condition_B_l249_249659


namespace circle_center_and_sum_l249_249142

/-- Given the equation of a circle x^2 + y^2 - 6x + 14y = -28,
    prove that the coordinates (h, k) of the center of the circle are (3, -7)
    and compute h + k. -/
theorem circle_center_and_sum (x y : ℝ) :
  (∃ h k, (x^2 + y^2 - 6*x + 14*y = -28) ∧ (h = 3) ∧ (k = -7) ∧ (h + k = -4)) :=
by {
  sorry
}

end circle_center_and_sum_l249_249142


namespace sin_300_eq_neg_sqrt3_div_2_l249_249503

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249503


namespace square_side_to_diagonal_ratio_l249_249874

theorem square_side_to_diagonal_ratio (s : ℝ) : 
  s / (s * Real.sqrt 2) = Real.sqrt 2 / 2 :=
by
  sorry

end square_side_to_diagonal_ratio_l249_249874


namespace sin_300_eq_neg_sqrt_3_div_2_l249_249522

noncomputable def sin_300_degrees : ℝ := Real.sin (300 * Real.pi / 180)

theorem sin_300_eq_neg_sqrt_3_div_2 : sin_300_degrees = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_300_eq_neg_sqrt_3_div_2_l249_249522


namespace find_wrong_observation_value_l249_249829

theorem find_wrong_observation_value :
  ∃ (wrong_value : ℝ),
    let n := 50
    let mean_initial := 36
    let mean_corrected := 36.54
    let observation_incorrect := 48
    let sum_initial := n * mean_initial
    let sum_corrected := n * mean_corrected
    let difference := sum_corrected - sum_initial
    wrong_value = observation_incorrect - difference := sorry

end find_wrong_observation_value_l249_249829


namespace xander_pages_left_to_read_l249_249850

theorem xander_pages_left_to_read :
  let total_pages := 500
  let read_first_night := 0.2 * 500
  let read_second_night := 0.2 * 500
  let read_third_night := 0.3 * 500
  total_pages - (read_first_night + read_second_night + read_third_night) = 150 :=
by 
  sorry

end xander_pages_left_to_read_l249_249850


namespace reciprocal_problem_l249_249790

theorem reciprocal_problem (x : ℚ) (h : 8 * x = 5) : 150 * (x⁻¹) = 240 := 
by 
  sorry

end reciprocal_problem_l249_249790


namespace max_chocolates_eaten_by_Ben_l249_249847

-- Define the situation with Ben and Carol sharing chocolates
variable (b c k : ℕ) -- b for Ben, c for Carol, k is the multiplier

-- Define the conditions
def chocolates_shared (b c : ℕ) : Prop := b + c = 30
def carol_eats_multiple (b c k : ℕ) : Prop := c = k * b ∧ k > 0

-- The theorem statement that we want to prove
theorem max_chocolates_eaten_by_Ben 
  (h1 : chocolates_shared b c) 
  (h2 : carol_eats_multiple b c k) : 
  b ≤ 15 := by
  sorry

end max_chocolates_eaten_by_Ben_l249_249847


namespace number_of_blue_marbles_l249_249157

-- Definitions based on the conditions
def total_marbles : ℕ := 20
def red_marbles : ℕ := 9
def probability_red_or_white : ℚ := 0.7

-- The question to prove: the number of blue marbles (B)
theorem number_of_blue_marbles (B W : ℕ) (h1 : B + W + red_marbles = total_marbles)
  (h2: (red_marbles + W : ℚ) / total_marbles = probability_red_or_white) : 
  B = 6 := 
by
  sorry

end number_of_blue_marbles_l249_249157


namespace line_does_not_pass_through_third_quadrant_l249_249709

-- Define the Cartesian equation of the line
def line_eq (x y : ℝ) : Prop :=
  x + 2 * y = 1

-- Define the property that a point (x, y) belongs to the third quadrant
def in_third_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

-- State the theorem
theorem line_does_not_pass_through_third_quadrant :
  ¬ ∃ (x y : ℝ), line_eq x y ∧ in_third_quadrant x y :=
by
  sorry

end line_does_not_pass_through_third_quadrant_l249_249709


namespace sin_300_deg_l249_249491

noncomputable def sin_300_equals_neg_sqrt3_div_2 : Prop :=
  sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2

theorem sin_300_deg : sin_300_equals_neg_sqrt3_div_2 :=
by
  sorry

end sin_300_deg_l249_249491


namespace chessboard_game_winner_l249_249869

theorem chessboard_game_winner (m n : ℕ) (initial_position : ℕ × ℕ) :
  (m * n) % 2 = 0 → (∃ A_wins : Prop, A_wins) ∧ 
  (m * n) % 2 = 1 → (∃ B_wins : Prop, B_wins) :=
by
  sorry

end chessboard_game_winner_l249_249869


namespace directrix_of_parabola_l249_249663

open Real

-- Define main parameters and assumptions
variables (p : ℝ) (h₁ : p > 0)
variables (focus : ℝ × ℝ := (p / 2, 0))
variables (H_focus : focus = (p / 2, 0))
variables (P : ℝ × ℝ) (H_P : P.1 = p / 2 ∧ P.2 = p)
variables (Q : ℝ × ℝ) (H_Q : Q.2 = 0)
variables (h_perpendicular_PF_x_axis : P.1 = focus.1)
variables (h_perpendicular_PQ_OP : slope(Q, P) * slope(P, (0, 0)) = -1)
variables (distance_FQ : dist(focus, Q) = 6)

-- Definition of the slope between two points
def slope (A B : ℝ × ℝ) : ℝ := (B.2 - A.2) / (B.1 - A.1)

-- Definition of the distance between two points
def dist (A B : ℝ × ℝ) : ℝ := sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

-- The problem statement
theorem directrix_of_parabola (hyp : slope (Q, P) * slope (P, (0, 0)) = -1)
(exists_p : p = 3)
: ∀ p > 0, ∀ focus = (p / 2, 0), ∃ x_ : ℝ, C : parabola := x_ = -3 / 2 := sorry

end directrix_of_parabola_l249_249663


namespace condition_sufficient_but_not_necessary_l249_249855

variable (a b : ℝ)

theorem condition_sufficient_but_not_necessary :
  (|a| < 1 ∧ |b| < 1) → (|1 - a * b| > |a - b|) ∧
  ((|1 - a * b| > |a - b|) → (|a| < 1 ∧ |b| < 1) ∨ (|a| ≥ 1 ∧ |b| ≥ 1)) :=
by
  sorry

end condition_sufficient_but_not_necessary_l249_249855


namespace polynomial_must_be_196_l249_249454

-- Define the polynomial P(x) with the additional constant m
def P (x m : ℝ) : ℝ := (x - 1)*(x + 3)*(x - 4)*(x - 8) + m

-- Statement to prove that for P(x) to be a perfect square polynomial, m must be 196
theorem polynomial_must_be_196 (m : ℝ) : 
  (∀ x : ℝ, ∃ g : ℝ → ℝ, (P x m = (g x) ^ 2)) ↔ (m = 196) :=
begin
  sorry
end

end polynomial_must_be_196_l249_249454


namespace sin_300_eq_neg_sqrt_3_div_2_l249_249516

noncomputable def sin_300_degrees : ℝ := Real.sin (300 * Real.pi / 180)

theorem sin_300_eq_neg_sqrt_3_div_2 : sin_300_degrees = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_300_eq_neg_sqrt_3_div_2_l249_249516


namespace amoeba_count_after_ten_days_l249_249754

theorem amoeba_count_after_ten_days : 
  let initial_amoebas := 1
  let splits_per_day := 3
  let days := 10
  (initial_amoebas * splits_per_day ^ days) = 59049 := 
by 
  let initial_amoebas := 1
  let splits_per_day := 3
  let days := 10
  show (initial_amoebas * splits_per_day ^ days) = 59049
  sorry

end amoeba_count_after_ten_days_l249_249754


namespace max_a_value_l249_249372

def f (a x : ℝ) : ℝ := x^3 - a*x^2 + (a^2 - 2)*x + 1

theorem max_a_value (a : ℝ) :
  (∃ m : ℝ, m > 0 ∧ f a m ≤ 0) → a ≤ 1 :=
by
  intro h
  sorry

end max_a_value_l249_249372


namespace tan_of_11pi_over_4_l249_249485

theorem tan_of_11pi_over_4 :
  Real.tan (11 * Real.pi / 4) = -1 := by
  sorry

end tan_of_11pi_over_4_l249_249485


namespace weight_of_bowling_ball_l249_249035

variable (b c : ℝ)

axiom h1 : 5 * b = 2 * c
axiom h2 : 3 * c = 84

theorem weight_of_bowling_ball : b = 11.2 :=
by
  sorry

end weight_of_bowling_ball_l249_249035


namespace spatial_relationship_l249_249945

variables {a b c : Type}          -- Lines a, b, c
variables {α β γ : Type}          -- Planes α, β, γ

-- Parallel relationship between planes
def plane_parallel (α β : Type) : Prop := sorry
-- Perpendicular relationship between planes
def plane_perpendicular (α β : Type) : Prop := sorry
-- Parallel relationship between lines and planes
def line_parallel_plane (a α : Type) : Prop := sorry
-- Perpendicular relationship between lines and planes
def line_perpendicular_plane (a α : Type) : Prop := sorry
-- Parallel relationship between lines
def line_parallel (a b : Type) : Prop := sorry
-- The angle formed by a line and a plane
def angle (a : Type) (α : Type) : Type := sorry

theorem spatial_relationship :
  (plane_parallel α γ ∧ plane_parallel β γ → plane_parallel α β) ∧
  ¬ (line_parallel_plane a α ∧ line_parallel_plane b α → line_parallel a b) ∧
  ¬ (plane_perpendicular α γ ∧ plane_perpendicular β γ → plane_parallel α β) ∧
  ¬ (line_perpendicular_plane a c ∧ line_perpendicular_plane b c → line_parallel a b) ∧
  (line_parallel a b ∧ plane_parallel α β → angle a α = angle b β) :=
sorry

end spatial_relationship_l249_249945


namespace find_number_l249_249130

noncomputable def some_number : ℝ :=
  0.27712 / 9.237333333333334

theorem find_number :
  (69.28 * 0.004) / some_number = 9.237333333333334 :=
by 
  sorry

end find_number_l249_249130


namespace john_average_speed_l249_249804

theorem john_average_speed:
  (∃ J : ℝ, Carla_speed = 35 ∧ Carla_time = 3 ∧ John_time = 3.5 ∧ J * John_time = Carla_speed * Carla_time) →
  (∃ J : ℝ, J = 30) :=
by
  -- Given Variables
  let Carla_speed : ℝ := 35
  let Carla_time : ℝ := 3
  let John_time : ℝ := 3.5
  -- Proof goal
  sorry

end john_average_speed_l249_249804


namespace min_value_of_f_l249_249371

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + Real.sin (2 * x)

theorem min_value_of_f : 
  ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y :=
sorry

end min_value_of_f_l249_249371


namespace max_possible_x_l249_249930

noncomputable section

def tan_deg (x : ℕ) : ℝ := Real.tan (x * Real.pi / 180)

theorem max_possible_x (x y : ℕ) (h₁ : tan_deg x - tan_deg y = 1 + tan_deg x * tan_deg y)
  (h₂ : tan_deg x * tan_deg y = 1) (h₃ : x = 98721) : x = 98721 := sorry

end max_possible_x_l249_249930


namespace x_14_and_inverse_x_14_l249_249783

theorem x_14_and_inverse_x_14 (x : ℂ) (h : x^2 + x + 1 = 0) : x^14 + x⁻¹^14 = -1 :=
by
  sorry

end x_14_and_inverse_x_14_l249_249783


namespace sin_300_eq_neg_sqrt3_div_2_l249_249616

-- Let Q be the point on the unit circle at 300 degrees counterclockwise from (1,0)
def Q : ℝ × ℝ := (1 / 2, -Real.sqrt 3 / 2)

-- Prove that the sine of 300 degrees is -√3/2
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  -- Here we assume the coordinates given in the conditions are correct.
  have hQ : Q = (1 / 2, -Real.sqrt 3 / 2) := rfl
  sorry

end sin_300_eq_neg_sqrt3_div_2_l249_249616


namespace find_n_coefficient_x2_rational_terms_in_expansion_l249_249212

open BigOperators

noncomputable def T (n r : ℕ) (x : ℕ) : ℚ :=
  (-1 / 2) ^ r * Nat.choose n r * x ^ ((n - 2 * r) / 3)

theorem find_n (const_term : ℚ) (h_const : T 10 5 const_term = 1) : n = 10 := 
by
  sorry

theorem coefficient_x2 (h_n : n = 10) : 
  T 10 2 x = 45 / 4 := 
by
  sorry

theorem rational_terms_in_expansion (h_n : n = 10) : 
  ∀ r, T 10 r x ∈ { 45 / 4 * x ^ 2, -63 / 8, 45 / 256 * x ^ (-2) } := 
by
  sorry

end find_n_coefficient_x2_rational_terms_in_expansion_l249_249212


namespace Eric_eggs_collected_l249_249629

theorem Eric_eggs_collected : 
  (∀ (chickens : ℕ) (eggs_per_chicken_per_day : ℕ) (days : ℕ),
    chickens = 4 ∧ eggs_per_chicken_per_day = 3 ∧ days = 3 → 
    chickens * eggs_per_chicken_per_day * days = 36) :=
by
  sorry

end Eric_eggs_collected_l249_249629


namespace ratio_of_radii_l249_249394

variable (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2)

theorem ratio_of_radii (ha : a > 0) (hb : b > 0) : (a / b = 1 / Real.sqrt 5) :=
by
  sorry

end ratio_of_radii_l249_249394


namespace exists_positive_n_with_m_zeros_l249_249986

theorem exists_positive_n_with_m_zeros (m : ℕ) (hm : 0 < m) :
  ∃ n : ℕ, 0 < n ∧ ∃ k : ℕ, 7^n = k * 10^m :=
sorry

end exists_positive_n_with_m_zeros_l249_249986


namespace total_good_vegetables_l249_249483

theorem total_good_vegetables :
  let carrots_day1 := 23
  let carrots_day2 := 47
  let tomatoes_day1 := 34
  let cucumbers_day1 := 42
  let tomatoes_day2 := 50
  let cucumbers_day2 := 38
  let rotten_carrots_day1 := 10
  let rotten_carrots_day2 := 15
  let rotten_tomatoes_day1 := 5
  let rotten_cucumbers_day1 := 7
  let rotten_tomatoes_day2 := 7
  let rotten_cucumbers_day2 := 12
  let good_carrots := (carrots_day1 - rotten_carrots_day1) + (carrots_day2 - rotten_carrots_day2)
  let good_tomatoes := (tomatoes_day1 - rotten_tomatoes_day1) + (tomatoes_day2 - rotten_tomatoes_day2)
  let good_cucumbers := (cucumbers_day1 - rotten_cucumbers_day1) + (cucumbers_day2 - rotten_cucumbers_day2)
  good_carrots + good_tomatoes + good_cucumbers = 178 := 
  sorry

end total_good_vegetables_l249_249483


namespace sin_300_eq_neg_sqrt_three_div_two_l249_249549

theorem sin_300_eq_neg_sqrt_three_div_two :
  Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt_three_div_two_l249_249549


namespace probability_quadrant_l249_249312

theorem probability_quadrant
    (r : ℝ) (x y : ℝ)
    (h : x^2 + y^2 ≤ r^2) :
    (∃ p : ℝ, p = (1 : ℚ)/4) :=
by
  sorry

end probability_quadrant_l249_249312


namespace count_positive_integers_l249_249221

theorem count_positive_integers (n : ℕ) (m : ℕ) :
  (∀ (k : ℕ), 1 ≤ k ∧ k < 100 ∧ (∃ (n : ℕ), n = 2 * k + 1 ∧ n < 200) 
  ∧ (∃ (m : ℤ), m = k * (k + 1) ∧ m % 5 = 0)) → 
  ∃ (cnt : ℕ), cnt = 20 :=
by
  sorry

end count_positive_integers_l249_249221


namespace strictly_increasing_range_l249_249213

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 * a - 3) * x + 1 else a ^ x

theorem strictly_increasing_range (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3 / 2 < a ∧ a ≤ 2) :=
sorry

end strictly_increasing_range_l249_249213


namespace relationship_among_vars_l249_249064

theorem relationship_among_vars {a b c d : ℝ} (h : (a + 2 * b) / (b + 2 * c) = (c + 2 * d) / (d + 2 * a)) :
  b = 2 * a ∨ a + b + c + d = 0 :=
sorry

end relationship_among_vars_l249_249064


namespace sin_300_l249_249598

-- Definitions and conditions in Lean 4
def angle_deg := ℝ
def sin (θ : angle_deg) : ℝ := Real.sin (θ * Real.pi / 180)

noncomputable def Q_angle : angle_deg := 300
noncomputable def first_quadrant_equivalent_angle : angle_deg := 360 - 300
noncomputable def sin_60 : ℝ := Real.sin (60 * Real.pi / 180)

-- Main statement to prove
theorem sin_300 : sin 300 = -sin_60 := by
  sorry

end sin_300_l249_249598


namespace sin_300_eq_neg_one_half_l249_249535

-- Definitions from the conditions
def point_on_unit_circle (θ : ℝ) : ℝ × ℝ :=
  (Real.cos (θ), Real.sin (θ))

def foot_of_perpendicular (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, 0)

def is_thirty_sixty_ninety_triangle (P : ℝ × ℝ) : Prop :=
  sqrt(P.1^2 + P.2^2) = 1 ∧ P.1 = sqrt(3) / 2 ∧ P.2 = -1/2

-- Problem statement
theorem sin_300_eq_neg_one_half : Real.sin (5 * Real.pi / 3) = -1/2 := by
  sorry

end sin_300_eq_neg_one_half_l249_249535


namespace smallest_a_for_f_iter_3_l249_249258

def f (x : Int) : Int :=
  if x % 4 = 0 ∧ x % 9 = 0 then x / 36
  else if x % 9 = 0 then 4 * x
  else if x % 4 = 0 then 9 * x
  else x + 4

def f_iter (f : Int → Int) (a : Nat) (x : Int) : Int :=
  if a = 0 then x else f_iter f (a - 1) (f x)

theorem smallest_a_for_f_iter_3 (a : Nat) (h : a > 1) : 
  (∀b, b > 1 → b < a → f_iter f b 3 ≠ f 3) ∧ f_iter f a 3 = f 3 ↔ a = 9 := 
  by
  sorry

end smallest_a_for_f_iter_3_l249_249258


namespace chestnut_picking_l249_249261

theorem chestnut_picking 
  (P : ℕ)
  (h1 : 12 + P + (P + 2) = 26) :
  12 / P = 2 :=
sorry

end chestnut_picking_l249_249261


namespace sin_300_l249_249601

-- Definitions and conditions in Lean 4
def angle_deg := ℝ
def sin (θ : angle_deg) : ℝ := Real.sin (θ * Real.pi / 180)

noncomputable def Q_angle : angle_deg := 300
noncomputable def first_quadrant_equivalent_angle : angle_deg := 360 - 300
noncomputable def sin_60 : ℝ := Real.sin (60 * Real.pi / 180)

-- Main statement to prove
theorem sin_300 : sin 300 = -sin_60 := by
  sorry

end sin_300_l249_249601


namespace min_solutions_f_eq_zero_l249_249362

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)
variable (h_period : ∀ x : ℝ, f (x + 3) = f x)
variable (h_zero_at_2 : f 2 = 0)

theorem min_solutions_f_eq_zero : ∃ S : Finset ℝ, (∀ x ∈ S, f x = 0) ∧ 7 ≤ S.card ∧ (∀ x ∈ S, x > 0 ∧ x < 6) := 
sorry

end min_solutions_f_eq_zero_l249_249362


namespace unrelated_statement_l249_249758

-- Definitions
def timely_snow_promises_harvest : Prop := true -- assumes it has a related factor
def upper_beam_not_straight_lower_beam_crooked : Prop := true -- assumes it has a related factor
def smoking_harmful_to_health : Prop := true -- assumes it has a related factor
def magpies_signify_joy_crows_signify_mourning : Prop := false -- does not have an inevitable relationship

-- Theorem
theorem unrelated_statement :
  ¬magpies_signify_joy_crows_signify_mourning :=
by 
  -- proof to be provided
  sorry

end unrelated_statement_l249_249758


namespace range_of_m_l249_249056

noncomputable def f (x : ℝ) := |x - 3| - 2
noncomputable def g (x : ℝ) := -|x + 1| + 4

theorem range_of_m (m : ℝ) : (∀ x, f x - g x ≥ m + 1) ↔ m ≤ -3 :=
by
  sorry

end range_of_m_l249_249056


namespace fraction_data_less_than_mode_is_one_third_l249_249734

-- Given list of data
def data_list : List ℕ := [1, 2, 3, 4, 5, 5, 5, 5, 7, 11, 21]

-- Definition of mode
def mode (l : List ℕ) : ℕ :=
  let grouped := l.groupBy id
  grouped.maxBy (λ g => g.length) |>.headD 0

-- Count of numbers less than the mode
def count_less_than_mode (mode : ℕ) (l : List ℕ) : Nat :=
  l.filter (λ x => x < mode).length

-- Total number of data points
def total_data_points (l : List ℕ) : Nat := l.length

-- The fraction of data that is less than the mode
def fraction_less_than_mode (l : List ℕ) : ℚ :=
  let m := mode l
  let count := count_less_than_mode m l
  count /. total_data_points l

-- Prove that the fraction of data less than the mode is 1/3
theorem fraction_data_less_than_mode_is_one_third : fraction_less_than_mode data_list = 1/3 := by
  sorry

end fraction_data_less_than_mode_is_one_third_l249_249734


namespace parallelepiped_length_l249_249012

theorem parallelepiped_length :
  ∃ n : ℕ, (n ≥ 7) ∧ (n * (n - 2) * (n - 4) = 3 * ((n - 2) * (n - 4) * (n - 6))) ∧ n = 18 :=
by
  sorry

end parallelepiped_length_l249_249012


namespace students_passed_in_both_subjects_l249_249243

theorem students_passed_in_both_subjects:
  ∀ (F_H F_E F_HE : ℝ), F_H = 0.30 → F_E = 0.42 → F_HE = 0.28 → (1 - (F_H + F_E - F_HE)) = 0.56 :=
by
  intros F_H F_E F_HE h1 h2 h3
  sorry

end students_passed_in_both_subjects_l249_249243


namespace triangle_area_l249_249080

theorem triangle_area (BC AC : ℝ) (angle_BAC : ℝ) (h1 : BC = 12) (h2 : AC = 5) (h3 : angle_BAC = π / 6) :
  1/2 * BC * (AC * Real.sin angle_BAC) = 15 :=
by
  sorry

end triangle_area_l249_249080
