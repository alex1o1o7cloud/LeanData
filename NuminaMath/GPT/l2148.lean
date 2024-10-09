import Mathlib

namespace tip_customers_count_l2148_214807

-- Definitions and given conditions
def initial_customers : ℕ := 29
def added_customers : ℕ := 20
def no_tip_customers : ℕ := 34

-- Total customers computation
def total_customers : ℕ := initial_customers + added_customers

-- Lean 4 statement for proof problem
theorem tip_customers_count : (total_customers - no_tip_customers) = 15 := by
  sorry

end tip_customers_count_l2148_214807


namespace find_total_cows_l2148_214827

-- Definitions as per the conditions
variables (D C L H : ℕ)

-- Condition 1: Total number of legs
def total_legs : ℕ := 2 * D + 4 * C

-- Condition 2: Total number of heads
def total_heads : ℕ := D + C

-- Condition 3: Legs are 28 more than twice the number of heads
def legs_heads_relation : Prop := total_legs D C = 2 * total_heads D C + 28

-- The theorem to prove
theorem find_total_cows (h : legs_heads_relation D C) : C = 14 :=
sorry

end find_total_cows_l2148_214827


namespace value_of_a_l2148_214857

def hyperbolaFociSharedEllipse : Prop :=
  ∃ a > 0, 
    (∃ c h k : ℝ, c = 3 ∧ (h, k) = (3, 0) ∨ (h, k) = (-3, 0)) ∧ 
    ∃ x y : ℝ, ((x^2) / 4) - ((y^2) / 5) = 1 ∧ ((x^2) / (a^2)) + ((y^2) / 16) = 1

theorem value_of_a : ∃ a > 0, hyperbolaFociSharedEllipse ∧ a = 5 :=
by
  sorry

end value_of_a_l2148_214857


namespace area_ratio_parallelogram_to_triangle_l2148_214825

variables {A B C D R E : Type*}
variables (s_AB s_AD : ℝ)

-- Given AR = 2/3 AB and AE = 1/3 AD
axiom AR_proportion : s_AB > 0 → s_AB * (2/3) = s_AB
axiom AE_proportion : s_AD > 0 → s_AD * (1/3) = s_AD

-- Given the relationship, we need to prove
theorem area_ratio_parallelogram_to_triangle (hAB : s_AB > 0) (hAD : s_AD > 0) :
  ∃ (S_ABCD S_ARE : ℝ), S_ABCD / S_ARE = 9 :=
by {
  sorry
}

end area_ratio_parallelogram_to_triangle_l2148_214825


namespace math_proof_problem_l2148_214838

theorem math_proof_problem : (10^8 / (2 * 10^5) - 50) = 450 := 
  by
  sorry

end math_proof_problem_l2148_214838


namespace find_multiple_of_t_l2148_214870

theorem find_multiple_of_t (k t x y : ℝ) (h1 : x = 1 - k * t) (h2 : y = 2 * t - 2) :
  t = 0.5 → x = y → k = 4 :=
by
  intros ht hxy
  sorry

end find_multiple_of_t_l2148_214870


namespace not_possible_to_create_3_piles_l2148_214830

def similar_sizes (a b : ℝ) : Prop := a / b ≤ Real.sqrt 2

theorem not_possible_to_create_3_piles (x : ℝ) (hx : 0 < x) : ¬ ∃ (y z w : ℝ), 
  y + z + w = x ∧ 
  similar_sizes y z ∧ similar_sizes z w ∧ similar_sizes y w := 
by 
  sorry

end not_possible_to_create_3_piles_l2148_214830


namespace find_new_ratio_l2148_214862

def initial_ratio (H C : ℕ) : Prop := H = 6 * C

def transaction (H C : ℕ) : Prop :=
  H - 15 = (C + 15) + 70

def new_ratio (H C : ℕ) : Prop := H - 15 = 3 * (C + 15)

theorem find_new_ratio (H C : ℕ) (h1 : initial_ratio H C) (h2 : transaction H C) : 
  new_ratio H C :=
sorry

end find_new_ratio_l2148_214862


namespace mrs_lovely_class_l2148_214871

-- Define the number of students in Mrs. Lovely's class
def number_of_students (g b : ℕ) : ℕ := g + b

theorem mrs_lovely_class (g b : ℕ): 
  (b = g + 3) →
  (500 - 10 = g * g + b * b) →
  number_of_students g b = 23 :=
by
  sorry

end mrs_lovely_class_l2148_214871


namespace find_r_in_arithmetic_sequence_l2148_214866

-- Define an arithmetic sequence
def is_arithmetic_sequence (a b c d e f : ℤ) : Prop :=
  (b - a = c - b) ∧ (c - b = d - c) ∧ (d - c = e - d) ∧ (e - d = f - e)

-- Define the given problem
theorem find_r_in_arithmetic_sequence :
  ∃ r : ℤ, ∀ p q s : ℤ, is_arithmetic_sequence 23 p q r s 59 → r = 41 :=
by
  sorry

end find_r_in_arithmetic_sequence_l2148_214866


namespace remaining_pens_l2148_214864

theorem remaining_pens (blue_initial black_initial red_initial green_initial purple_initial : ℕ)
                        (blue_removed black_removed red_removed green_removed purple_removed : ℕ) :
  blue_initial = 15 → black_initial = 27 → red_initial = 12 → green_initial = 10 → purple_initial = 8 →
  blue_removed = 8 → black_removed = 9 → red_removed = 3 → green_removed = 5 → purple_removed = 6 →
  blue_initial - blue_removed + black_initial - black_removed + red_initial - red_removed +
  green_initial - green_removed + purple_initial - purple_removed = 41 :=
by
  intros
  sorry

end remaining_pens_l2148_214864


namespace rocco_piles_of_quarters_proof_l2148_214850

-- Define the value of a pile of different types of coins
def pile_value (coin_value : ℕ) (num_coins_in_pile : ℕ) : ℕ :=
  coin_value * num_coins_in_pile

-- Define the number of piles for different coins
def num_piles_of_dimes : ℕ := 6
def num_piles_of_nickels : ℕ := 9
def num_piles_of_pennies : ℕ := 5
def num_coins_in_pile : ℕ := 10

-- Define the total value of each type of coin
def value_of_a_dime : ℕ := 10  -- in cents
def value_of_a_nickel : ℕ := 5  -- in cents
def value_of_a_penny : ℕ := 1  -- in cents
def value_of_a_quarter : ℕ := 25  -- in cents

-- Define the total money Rocco has in cents
def total_money : ℕ := 2100  -- since $21 = 2100 cents

-- Calculate the value of all piles of each type of coin
def total_dimes_value : ℕ := num_piles_of_dimes * (pile_value value_of_a_dime num_coins_in_pile)
def total_nickels_value : ℕ := num_piles_of_nickels * (pile_value value_of_a_nickel num_coins_in_pile)
def total_pennies_value : ℕ := num_piles_of_pennies * (pile_value value_of_a_penny num_coins_in_pile)

-- Calculate the value of the quarters
def value_of_quarters : ℕ := total_money - (total_dimes_value + total_nickels_value + total_pennies_value)
def num_piles_of_quarters : ℕ := value_of_quarters / 250 -- since each pile of quarters is worth 250 cents

-- Theorem to prove
theorem rocco_piles_of_quarters_proof : num_piles_of_quarters = 4 := by
  sorry

end rocco_piles_of_quarters_proof_l2148_214850


namespace solve_inequality_l2148_214868

theorem solve_inequality (x : ℝ) : 
  (x - 2) / (x + 1) ≤ 2 ↔ x ∈ Set.Iic (-4) ∪ Set.Ioi (-1) := 
sorry

end solve_inequality_l2148_214868


namespace intersection_of_sets_l2148_214820

-- Definitions of sets A and B based on given conditions
def setA : Set ℤ := {x | x + 2 = 0}
def setB : Set ℤ := {x | x^2 - 4 = 0}

-- The theorem to prove the intersection of A and B
theorem intersection_of_sets : setA ∩ setB = {-2} := by
  sorry

end intersection_of_sets_l2148_214820


namespace functional_equation_solution_exists_l2148_214824

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution_exists (f : ℝ → ℝ) (h : ∀ x y, 0 < x → 0 < y → f x * f y = 2 * f (x + y * f x)) :
  ∃ c : ℝ, ∀ x, 0 < x → f x = x + c := 
sorry

end functional_equation_solution_exists_l2148_214824


namespace find_n_in_geom_series_l2148_214808

noncomputable def geom_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem find_n_in_geom_series :
  ∃ n : ℕ, geom_sum 1 (1/2) n = 31 / 16 :=
sorry

end find_n_in_geom_series_l2148_214808


namespace ordered_triple_l2148_214836

theorem ordered_triple (a b c : ℝ) (h1 : 4 < a) (h2 : 4 < b) (h3 : 4 < c) 
  (h_eq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) 
  : (a, b, c) = (12, 10, 8) :=
  sorry

end ordered_triple_l2148_214836


namespace kelly_needs_more_apples_l2148_214879

theorem kelly_needs_more_apples (initial_apples : ℕ) (total_apples : ℕ) (needed_apples : ℕ) :
  initial_apples = 128 → total_apples = 250 → needed_apples = total_apples - initial_apples → needed_apples = 122 :=
by
  intros h_initial h_total h_needed
  rw [h_initial, h_total] at h_needed
  exact h_needed

end kelly_needs_more_apples_l2148_214879


namespace digits_problem_solution_l2148_214809

def digits_proof_problem (E F G H : ℕ) : Prop :=
  (E, F, G) = (5, 0, 5) → H = 0

theorem digits_problem_solution 
  (E F G H : ℕ)
  (h1 : F + E = E ∨ F + E = E + 10)
  (h2 : E ≠ 0)
  (h3 : E = 5)
  (h4 : 5 + G = H)
  (h5 : 5 - G = 0) :
  H = 0 := 
by {
  sorry -- proof goes here
}

end digits_problem_solution_l2148_214809


namespace circle_center_sum_l2148_214878

theorem circle_center_sum (x y : ℝ) :
  (x^2 + y^2 = 10*x - 12*y + 40) →
  x + y = -1 :=
by {
  sorry
}

end circle_center_sum_l2148_214878


namespace ratio_eq_two_l2148_214821

theorem ratio_eq_two (a b c d : ℤ) (h1 : b * c + a * d = 1) (h2 : a * c + 2 * b * d = 1) : 
  (a^2 + c^2 : ℚ) / (b^2 + d^2) = 2 :=
sorry

end ratio_eq_two_l2148_214821


namespace solve_for_x_l2148_214897

theorem solve_for_x (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 18) : x = 14 := by
  sorry

end solve_for_x_l2148_214897


namespace new_volume_eq_7352_l2148_214817

variable (l w h : ℝ)

-- Given conditions
def volume_eq : Prop := l * w * h = 5184
def surface_area_eq : Prop := l * w + w * h + h * l = 972
def edge_sum_eq : Prop := l + w + h = 54

-- Question: New volume when dimensions are increased by two inches
def new_volume : ℝ := (l + 2) * (w + 2) * (h + 2)

-- Correct Answer: Prove that the new volume equals 7352
theorem new_volume_eq_7352 (h_vol : volume_eq l w h) (h_surf : surface_area_eq l w h) (h_edge : edge_sum_eq l w h) 
    : new_volume l w h = 7352 :=
by
  -- Proof omitted
  sorry

#check new_volume_eq_7352

end new_volume_eq_7352_l2148_214817


namespace right_triangle_leg_length_l2148_214829

theorem right_triangle_leg_length (a b c : ℕ) (h_c : c = 13) (h_a : a = 12) (h_pythagorean : a^2 + b^2 = c^2) :
  b = 5 := 
by {
  -- Provide a placeholder for the proof
  sorry
}

end right_triangle_leg_length_l2148_214829


namespace sum_of_numbers_of_large_cube_l2148_214818

def sum_faces_of_die := 1 + 2 + 3 + 4 + 5 + 6

def number_of_dice := 125

def number_of_faces_per_die := 6

def total_exposed_faces (side_length: ℕ) : ℕ := 6 * (side_length * side_length)

theorem sum_of_numbers_of_large_cube (side_length : ℕ) (dice_count : ℕ) 
    (sum_per_die : ℕ) (opposite_face_sum : ℕ) :
    dice_count = 125 →
    total_exposed_faces side_length = 150 →
    sum_per_die = 21 →
    (∀ f1 f2, (f1 + f2 = opposite_face_sum)) →
    dice_count * sum_per_die = 2625 →
    (210 ≤ dice_count * sum_per_die ∧ dice_count * sum_per_die ≤ 840) :=
by 
  intro h_dice_count
  intro h_exposed_faces
  intro h_sum_per_die
  intro h_opposite_faces
  intro h_total_sum
  sorry

end sum_of_numbers_of_large_cube_l2148_214818


namespace lollipop_cases_l2148_214856

theorem lollipop_cases (total_cases : ℕ) (chocolate_cases : ℕ) (lollipop_cases : ℕ) 
  (h1 : total_cases = 80) (h2 : chocolate_cases = 25) : lollipop_cases = 55 :=
by
  sorry

end lollipop_cases_l2148_214856


namespace desired_on_time_departure_rate_l2148_214867

theorem desired_on_time_departure_rate :
  let first_late := 1
  let on_time_flights_next := 3
  let additional_on_time_flights := 2
  let total_on_time_flights := on_time_flights_next + additional_on_time_flights
  let total_flights := first_late + on_time_flights_next + additional_on_time_flights
  let on_time_departure_rate := (total_on_time_flights : ℚ) / (total_flights : ℚ) * 100
  on_time_departure_rate > 83.33 :=
by
  sorry

end desired_on_time_departure_rate_l2148_214867


namespace domain_of_f_l2148_214822

noncomputable def f (x : ℝ) : ℝ := (Real.tan (2 * x)) / Real.sqrt (x - x^2)

theorem domain_of_f :
  { x : ℝ | ∃ k : ℤ, 2*x ≠ k*π + π/2 ∧ x ∈ (Set.Ioo 0 (π/4) ∪ Set.Ioo (π/4) 1) } = 
  { x : ℝ | x ∈ Set.Ioo 0 (π/4) ∪ Set.Ioo (π/4) 1 } :=
sorry

end domain_of_f_l2148_214822


namespace carl_sold_each_watermelon_for_3_l2148_214860

def profit : ℕ := 105
def final_watermelons : ℕ := 18
def starting_watermelons : ℕ := 53
def sold_watermelons : ℕ := starting_watermelons - final_watermelons
def price_per_watermelon : ℕ := profit / sold_watermelons

theorem carl_sold_each_watermelon_for_3 :
  price_per_watermelon = 3 :=
by
  sorry

end carl_sold_each_watermelon_for_3_l2148_214860


namespace geometric_number_difference_l2148_214880

theorem geometric_number_difference : 
  ∀ (a b c : ℕ), 8 = a → b ≠ c → (∃ k : ℕ, 8 ≠ k ∧ b = k ∧ c = k * k / 8) → (10^2 * a + 10 * b + c = 842) ∧ (10^2 * a + 10 * b + c = 842) → (10^2 * a + 10 * b + c) - (10^2 * a + 10 * b + c) = 0 :=
by
  intro a b c
  intro ha hb
  intro hk
  intro hseq
  sorry

end geometric_number_difference_l2148_214880


namespace vector_subtraction_l2148_214831

def vector_a : ℝ × ℝ := (3, 5)
def vector_b : ℝ × ℝ := (-2, 1)

theorem vector_subtraction :
  vector_a - 2 • vector_b = (7, 3) :=
sorry

end vector_subtraction_l2148_214831


namespace f_2012_eq_3_l2148_214875

noncomputable def f (a b α β : ℝ) (x : ℝ) : ℝ := a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem f_2012_eq_3 
  (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) 
  (h : f a b α β 2011 = 5) : 
  f a b α β 2012 = 3 :=
by
  sorry

end f_2012_eq_3_l2148_214875


namespace solve_square_l2148_214801

theorem solve_square:
  ∃ (square: ℚ), 
    ((13/5) - ((17/2) - square) / (7/2)) / (1 / ((61/20) + (89/20))) = 2 → 
    square = 1/3 :=
  sorry

end solve_square_l2148_214801


namespace simplify_fraction_eq_one_over_thirty_nine_l2148_214889

theorem simplify_fraction_eq_one_over_thirty_nine :
  let a1 := (1 / 3)^1
  let a2 := (1 / 3)^2
  let a3 := (1 / 3)^3
  (1 / (1 / a1 + 1 / a2 + 1 / a3)) = 1 / 39 :=
by
  sorry

end simplify_fraction_eq_one_over_thirty_nine_l2148_214889


namespace calculate_expression_l2148_214892

theorem calculate_expression : (3^5 * 4^5) / 6^5 = 32 := 
by
  sorry

end calculate_expression_l2148_214892


namespace unique_intersection_point_l2148_214806

theorem unique_intersection_point {c : ℝ} :
  (∀ x y : ℝ, y = |x - 20| + |x + 18| → y = x + c → (x = 20 ∧ y = 38)) ↔ c = 18 :=
by
  sorry

end unique_intersection_point_l2148_214806


namespace two_digit_number_exists_l2148_214873

theorem two_digit_number_exists (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) :
  (9 * x + 8) * (80 - 9 * x) = 1855 → (9 * x + 8 = 35 ∨ 9 * x + 8 = 53) := by
  sorry

end two_digit_number_exists_l2148_214873


namespace tree_height_l2148_214865

theorem tree_height (future_height : ℕ) (growth_per_year : ℕ) (years : ℕ) (inches_per_foot : ℕ) :
  future_height = 1104 →
  growth_per_year = 5 →
  years = 8 →
  inches_per_foot = 12 →
  (future_height / inches_per_foot - growth_per_year * years) = 52 := 
by
  intros h1 h2 h3 h4
  sorry

end tree_height_l2148_214865


namespace money_last_weeks_l2148_214819

-- Define the amounts of money earned and spent per week
def money_mowing : ℕ := 5
def money_weed_eating : ℕ := 58
def weekly_spending : ℕ := 7

-- Define the total money earned
def total_money : ℕ := money_mowing + money_weed_eating

-- Define the number of weeks the money will last
def weeks_last (total : ℕ) (weekly : ℕ) : ℕ := total / weekly

-- Theorem stating the number of weeks the money will last
theorem money_last_weeks : weeks_last total_money weekly_spending = 9 := by
  sorry

end money_last_weeks_l2148_214819


namespace cat_toy_cost_l2148_214861

-- Define the conditions
def cost_of_cage : ℝ := 11.73
def total_cost_of_purchases : ℝ := 21.95

-- Define the proof statement
theorem cat_toy_cost : (total_cost_of_purchases - cost_of_cage) = 10.22 := by
  sorry

end cat_toy_cost_l2148_214861


namespace males_listen_l2148_214811

theorem males_listen (males_dont_listen females_listen total_listen total_dont_listen : ℕ) 
  (h1 : males_dont_listen = 70)
  (h2 : females_listen = 75)
  (h3 : total_listen = 180)
  (h4 : total_dont_listen = 120) :
  ∃ m, m = 105 :=
by {
  sorry
}

end males_listen_l2148_214811


namespace chase_cardinals_count_l2148_214816

variable (gabrielle_robins : Nat)
variable (gabrielle_cardinals : Nat)
variable (gabrielle_blue_jays : Nat)
variable (chase_robins : Nat)
variable (chase_blue_jays : Nat)
variable (chase_cardinals : Nat)

variable (gabrielle_total : Nat)
variable (chase_total : Nat)

variable (percent_more : Nat)

axiom gabrielle_robins_def : gabrielle_robins = 5
axiom gabrielle_cardinals_def : gabrielle_cardinals = 4
axiom gabrielle_blue_jays_def : gabrielle_blue_jays = 3

axiom chase_robins_def : chase_robins = 2
axiom chase_blue_jays_def : chase_blue_jays = 3

axiom gabrielle_total_def : gabrielle_total = gabrielle_robins + gabrielle_cardinals + gabrielle_blue_jays
axiom chase_total_def : chase_total = chase_robins + chase_blue_jays + chase_cardinals
axiom percent_more_def : percent_more = 20

axiom gabrielle_more_birds : gabrielle_total = Nat.ceil ((chase_total * (100 + percent_more)) / 100)

theorem chase_cardinals_count : chase_cardinals = 5 := by sorry

end chase_cardinals_count_l2148_214816


namespace probability_of_exactly_one_hitting_l2148_214802

variable (P_A_hitting B_A_hitting : ℝ)

theorem probability_of_exactly_one_hitting (hP_A : P_A_hitting = 0.6) (hP_B : B_A_hitting = 0.6) :
  ((P_A_hitting * (1 - B_A_hitting)) + ((1 - P_A_hitting) * B_A_hitting)) = 0.48 := 
by 
  sorry

end probability_of_exactly_one_hitting_l2148_214802


namespace original_numbers_correct_l2148_214855

noncomputable def restore_original_numbers : List ℕ :=
  let T : ℕ := 5
  let EL : ℕ := 12
  let EK : ℕ := 19
  let LA : ℕ := 26
  let SS : ℕ := 33
  [T, EL, EK, LA, SS]

theorem original_numbers_correct :
  restore_original_numbers = [5, 12, 19, 26, 33] :=
by
  sorry

end original_numbers_correct_l2148_214855


namespace centimeters_per_inch_l2148_214815

theorem centimeters_per_inch (miles_per_map_inch : ℝ) (cm_measured : ℝ) (approx_miles : ℝ) (miles_per_inch : ℝ) (inches_from_cm : ℝ) : 
  miles_per_map_inch = 16 →
  inches_from_cm = 18.503937007874015 →
  miles_per_map_inch = 24 / 1.5 →
  approx_miles = 296.06299212598424 →
  cm_measured = 47 →
  (cm_measured / inches_from_cm) = 2.54 :=
by
  sorry

end centimeters_per_inch_l2148_214815


namespace price_reduction_is_not_10_yuan_l2148_214858

theorem price_reduction_is_not_10_yuan (current_price original_price : ℝ)
  (CurrentPrice : current_price = 45)
  (Reduction : current_price = 0.9 * original_price)
  (TenPercentReduction : 0.1 * original_price = 10) :
  (original_price - current_price) ≠ 10 := by
  sorry

end price_reduction_is_not_10_yuan_l2148_214858


namespace domain_of_f_l2148_214846

noncomputable def f (x : ℝ) := (Real.sqrt (x + 3)) / x

theorem domain_of_f :
  { x : ℝ | x ≥ -3 ∧ x ≠ 0 } = { x : ℝ | ∃ y, f y ≠ 0 } :=
by
  sorry

end domain_of_f_l2148_214846


namespace value_of_y_l2148_214832

theorem value_of_y (x y z : ℕ) (h_positive_x : 0 < x) (h_positive_y : 0 < y) (h_positive_z : 0 < z)
    (h_sum : x + y + z = 37) (h_eq : 4 * x = 6 * z) : y = 32 :=
sorry

end value_of_y_l2148_214832


namespace sampling_interval_is_100_l2148_214834

-- Define the total number of numbers (N), the number of samples to be taken (k), and the condition for systematic sampling.
def N : ℕ := 2005
def k : ℕ := 20

-- Define the concept of systematic sampling interval
def sampling_interval (N k : ℕ) : ℕ := N / k

-- The proof that the sampling interval is 100 when 2005 numbers are sampled as per the systematic sampling method.
theorem sampling_interval_is_100 (N k : ℕ) 
  (hN : N = 2005) 
  (hk : k = 20) 
  (h1 : N % k ≠ 0) : 
  sampling_interval (N - (N % k)) k = 100 :=
by
  -- Initialization
  sorry

end sampling_interval_is_100_l2148_214834


namespace smallest_tree_height_l2148_214893

theorem smallest_tree_height (tallest middle smallest : ℝ)
  (h1 : tallest = 108)
  (h2 : middle = (tallest / 2) - 6)
  (h3 : smallest = middle / 4) : smallest = 12 :=
by
  sorry

end smallest_tree_height_l2148_214893


namespace smallest_k_satisfying_condition_l2148_214887

def is_smallest_prime_greater_than (n : ℕ) (p : ℕ) : Prop :=
  Nat.Prime p ∧ n < p ∧ ∀ q, Nat.Prime q ∧ q > n → q >= p

def is_divisible_by (m k : ℕ) : Prop := k % m = 0

theorem smallest_k_satisfying_condition :
  ∃ k, is_smallest_prime_greater_than 19 23 ∧ is_divisible_by 3 k ∧ 64 ^ k > 4 ^ (19 * 23) ∧ (∀ k' < k, is_divisible_by 3 k' → 64 ^ k' ≤ 4 ^ (19 * 23)) :=
by
  sorry

end smallest_k_satisfying_condition_l2148_214887


namespace pyr_sphere_ineq_l2148_214876

open Real

theorem pyr_sphere_ineq (h a : ℝ) (R r : ℝ) 
  (h_pos : h > 0) (a_pos : a > 0) 
  (pyr_in_sphere : ∀ h a : ℝ, R = (2*a^2 + h^2) / (2*h))
  (pyr_circ_sphere : ∀ h a : ℝ, r = (a * h) / (sqrt (h^2 + a^2) + a)) :
  R ≥ (sqrt 2 + 1) * r := 
sorry

end pyr_sphere_ineq_l2148_214876


namespace cistern_fill_time_l2148_214833

theorem cistern_fill_time (F E : ℝ) (hF : F = 1/3) (hE : E = 1/6) : (1 / (F - E)) = 6 :=
by sorry

end cistern_fill_time_l2148_214833


namespace max_f_theta_l2148_214841

noncomputable def determinant (a b c d : ℝ) : ℝ := a*d - b*c

noncomputable def f (θ : ℝ) : ℝ :=
  determinant (Real.sin θ) (Real.cos θ) (-1) 1

theorem max_f_theta :
  ∀ θ : ℝ, 0 < θ ∧ θ < (Real.pi / 3) →
  f θ ≤ (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  sorry

end max_f_theta_l2148_214841


namespace solve_r_l2148_214842

def E (a : ℝ) (b : ℝ) (c : ℕ) : ℝ := a * b^c

theorem solve_r : ∃ (r : ℝ), E r r 5 = 1024 ∧ r = 2^(5/3) :=
by
  sorry

end solve_r_l2148_214842


namespace max_marks_tests_l2148_214847

theorem max_marks_tests :
  ∃ (T1 T2 T3 T4 : ℝ),
    0.30 * T1 = 80 + 40 ∧
    0.40 * T2 = 105 + 35 ∧
    0.50 * T3 = 150 + 50 ∧
    0.60 * T4 = 180 + 60 ∧
    T1 = 400 ∧
    T2 = 350 ∧
    T3 = 400 ∧
    T4 = 400 :=
by
    sorry

end max_marks_tests_l2148_214847


namespace parabola_x_intercepts_count_l2148_214843

theorem parabola_x_intercepts_count :
  ∃! x, ∃ y, x = -3 * y^2 + 2 * y + 2 ∧ y = 0 :=
by
  sorry

end parabola_x_intercepts_count_l2148_214843


namespace simplify_eval_expression_l2148_214884

theorem simplify_eval_expression (a : ℝ) (h : a^2 + 2 * a - 1 = 0) :
  ((a^2 - 1) / (a^2 - 2 * a + 1) - 1 / (1 - a)) / (1 / (a^2 - a)) = 1 :=
  sorry

end simplify_eval_expression_l2148_214884


namespace sum_of_squares_not_divisible_by_5_or_13_l2148_214803

-- Definition of the set T
def T (n : ℤ) : ℤ :=
  (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2

-- The theorem to prove
theorem sum_of_squares_not_divisible_by_5_or_13 (n : ℤ) :
  ¬ (T n % 5 = 0) ∧ ¬ (T n % 13 = 0) :=
by
  sorry

end sum_of_squares_not_divisible_by_5_or_13_l2148_214803


namespace isabella_houses_l2148_214826

theorem isabella_houses (green yellow red : ℕ)
  (h1 : green = 3 * yellow)
  (h2 : yellow = red - 40)
  (h3 : green = 90) :
  green + red = 160 := 
by sorry

end isabella_houses_l2148_214826


namespace least_possible_b_l2148_214891

noncomputable def a : ℕ := 8

theorem least_possible_b (b : ℕ) (h1 : ∀ n : ℕ, n > 0 → a.factors.count n = 1 → a = n^3)
  (h2 : b.factors.count a = 1)
  (h3 : b % a = 0) :
  b = 24 :=
sorry

end least_possible_b_l2148_214891


namespace prime_square_condition_no_prime_cube_condition_l2148_214898

-- Part (a): Prove p = 3 given 8*p + 1 = n^2 and p is a prime
theorem prime_square_condition (p : ℕ) (n : ℕ) (h_prime : Prime p) 
  (h_eq : 8 * p + 1 = n ^ 2) : 
  p = 3 :=
sorry

-- Part (b): Prove no p exists given 8*p + 1 = n^3 and p is a prime
theorem no_prime_cube_condition (p : ℕ) (n : ℕ) (h_prime : Prime p) 
  (h_eq : 8 * p + 1 = n ^ 3) : 
  False :=
sorry

end prime_square_condition_no_prime_cube_condition_l2148_214898


namespace trapezoid_area_l2148_214869

theorem trapezoid_area (AD BC AC : ℝ) (BD : ℝ) 
  (hAD : AD = 24) 
  (hBC : BC = 8) 
  (hAC : AC = 13) 
  (hBD : BD = 5 * Real.sqrt 17) : 
  (1 / 2 * (AD + BC) * Real.sqrt (AC^2 - (BC + (AD - BC) / 2)^2)) = 80 :=
by
  sorry

end trapezoid_area_l2148_214869


namespace f_properties_l2148_214812

noncomputable def f : ℝ → ℝ := sorry -- we define f as a noncomputable function for generality 

-- Given conditions as Lean hypotheses
axiom functional_eq : ∀ x y : ℝ, f x + f y = 2 * f ((x + y) / 2) * f ((x - y) / 2)
axiom not_always_zero : ¬(∀ x : ℝ, f x = 0)

-- The statement we need to prove
theorem f_properties : f 0 = 1 ∧ (∀ x : ℝ, f (-x) = f x) := 
  by 
    sorry

end f_properties_l2148_214812


namespace sum_int_values_l2148_214823

theorem sum_int_values (sum : ℤ) : 
  (∀ n : ℤ, (20 % (2 * n - 1) = 0) → sum = 2) :=
by
  sorry

end sum_int_values_l2148_214823


namespace number_of_slices_per_package_l2148_214859

-- Define the problem's conditions
def packages_of_bread := 2
def slices_per_package_of_ham := 8
def packages_of_ham := 2
def leftover_slices_of_bread := 8
def total_ham_slices := packages_of_ham * slices_per_package_of_ham
def total_ham_required_bread := total_ham_slices * 2
def total_initial_bread_slices (B : ℕ) := packages_of_bread * B
def total_bread_used (B : ℕ) := total_ham_required_bread
def slices_leftover (B : ℕ) := total_initial_bread_slices B - total_bread_used B

-- Specify the goal
theorem number_of_slices_per_package (B : ℕ) (h : total_initial_bread_slices B = total_bread_used B + leftover_slices_of_bread) : B = 20 :=
by
  -- Use the provided conditions along with the hypothesis
  -- of the initial bread slices equation equating to used and leftover slices
  sorry

end number_of_slices_per_package_l2148_214859


namespace train_speed_l2148_214844

theorem train_speed (length_of_train time_to_cross : ℝ) (h_length : length_of_train = 800) (h_time : time_to_cross = 12) : (length_of_train / time_to_cross) = 66.67 :=
by
  sorry

end train_speed_l2148_214844


namespace arithmetic_sequence_sum_l2148_214839

variable (a : ℕ → ℝ)
variable (d : ℝ)

noncomputable def arithmetic_sequence := ∀ n : ℕ, a n = a 0 + n * d

theorem arithmetic_sequence_sum (h₁ : a 1 + a 2 = 3) (h₂ : a 3 + a 4 = 5) :
  a 7 + a 8 = 9 :=
by
  sorry

end arithmetic_sequence_sum_l2148_214839


namespace magnitude_2a_sub_b_l2148_214874

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-1, 2)

theorem magnitude_2a_sub_b : (‖(2 * a.1 - b.1, 2 * a.2 - b.2)‖ = 5) :=
by {
  sorry
}

end magnitude_2a_sub_b_l2148_214874


namespace misread_number_is_correct_l2148_214896

-- Definitions for the given conditions
def avg_incorrect : ℕ := 19
def incorrect_number : ℕ := 26
def avg_correct : ℕ := 24

-- Statement to prove the actual number that was misread
theorem misread_number_is_correct (x : ℕ) (h : 10 * avg_correct - 10 * avg_incorrect = x - incorrect_number) : x = 76 :=
by {
  sorry
}

end misread_number_is_correct_l2148_214896


namespace arithmetic_mean_solution_l2148_214849

/-- Given the arithmetic mean of six expressions is 30, prove the values of x and y are as follows. -/
theorem arithmetic_mean_solution (x y : ℝ) (h : ((2 * x - y) + 20 + (3 * x + y) + 16 + (x + 5) + (y + 8)) / 6 = 30) (hy : y = 10) : 
  x = 18.5 :=
by {
  sorry
}

end arithmetic_mean_solution_l2148_214849


namespace find_equation_of_BC_l2148_214814

theorem find_equation_of_BC :
  ∃ (BC : ℝ → ℝ → Prop), 
  (∀ x y, (BC x y ↔ 2 * x - y + 5 = 0)) :=
sorry

end find_equation_of_BC_l2148_214814


namespace correct_calculation_l2148_214899

theorem correct_calculation (x : ℝ) : x * x^2 = x^3 :=
by sorry

end correct_calculation_l2148_214899


namespace total_courses_attended_l2148_214877

-- Define the number of courses attended by Max
def maxCourses : ℕ := 40

-- Define the number of courses attended by Sid (four times as many as Max)
def sidCourses : ℕ := 4 * maxCourses

-- Define the total number of courses attended by both Max and Sid
def totalCourses : ℕ := maxCourses + sidCourses

-- The proof statement
theorem total_courses_attended : totalCourses = 200 := by
  sorry

end total_courses_attended_l2148_214877


namespace scientific_notation_of_308000000_l2148_214854

theorem scientific_notation_of_308000000 :
  ∃ (a : ℝ) (n : ℤ), (a = 3.08) ∧ (n = 8) ∧ (308000000 = a * 10 ^ n) :=
by
  sorry

end scientific_notation_of_308000000_l2148_214854


namespace simplify_complex_fraction_l2148_214828

theorem simplify_complex_fraction :
  (⟨3, 5⟩ : ℂ) / (⟨-2, 7⟩ : ℂ) = (29 / 53) - (31 / 53) * I :=
by sorry

end simplify_complex_fraction_l2148_214828


namespace cindy_correct_answer_l2148_214895

theorem cindy_correct_answer (x : ℝ) (h : (x - 5) / 7 = 15) :
  (x - 7) / 5 = 20.6 :=
by
  sorry

end cindy_correct_answer_l2148_214895


namespace find_a_value_l2148_214848

theorem find_a_value (a : ℤ) (h1 : 0 < a) (h2 : a < 13) (h3 : (53^2017 + a) % 13 = 0) : a = 12 :=
by
  -- proof steps
  sorry

end find_a_value_l2148_214848


namespace div_by_37_l2148_214886

theorem div_by_37 : (333^555 + 555^333) % 37 = 0 :=
by sorry

end div_by_37_l2148_214886


namespace second_term_of_geometric_series_l2148_214851

noncomputable def geometric_series_second_term (a r : ℝ) (S : ℝ) : ℝ :=
a * r

theorem second_term_of_geometric_series 
  (a r S : ℝ) 
  (h1 : r = 1 / 4) 
  (h2 : S = 10) 
  (h3 : S = a / (1 - r)) 
  : geometric_series_second_term a r S = 1.875 :=
by
  sorry

end second_term_of_geometric_series_l2148_214851


namespace popsicle_sum_l2148_214840

-- Gino has 63 popsicle sticks
def gino_popsicle_sticks : Nat := 63

-- I have 50 popsicle sticks
def my_popsicle_sticks : Nat := 50

-- The sum of our popsicle sticks
def total_popsicle_sticks : Nat := gino_popsicle_sticks + my_popsicle_sticks

-- Prove that the total is 113
theorem popsicle_sum : total_popsicle_sticks = 113 :=
by
  -- Proof goes here
  sorry

end popsicle_sum_l2148_214840


namespace sheep_count_l2148_214890

theorem sheep_count (cows sheep shepherds : ℕ) 
  (h_cows : cows = 12) 
  (h_ears : 2 * cows < sheep) 
  (h_legs : sheep < 4 * cows) 
  (h_shepherds : sheep = 12 * shepherds) :
  sheep = 36 :=
by {
  sorry
}

end sheep_count_l2148_214890


namespace cistern_length_is_correct_l2148_214810

-- Definitions for the conditions mentioned in the problem
def cistern_width : ℝ := 6
def water_depth : ℝ := 1.25
def wet_surface_area : ℝ := 83

-- The length of the cistern to be proven
def cistern_length : ℝ := 8

-- Theorem statement that length of the cistern must be 8 meters given the conditions
theorem cistern_length_is_correct :
  ∃ (L : ℝ), (wet_surface_area = (L * cistern_width) + (2 * L * water_depth) + (2 * cistern_width * water_depth)) ∧ L = cistern_length :=
  sorry

end cistern_length_is_correct_l2148_214810


namespace trigonometric_identity_l2148_214872

theorem trigonometric_identity
  (θ : ℝ)
  (h1 : θ > -π/2)
  (h2 : θ < 0)
  (h3 : Real.tan θ = -2) :
  (Real.sin θ)^2 / (Real.cos (2 * θ) + 2) = 4 / 7 :=
sorry

end trigonometric_identity_l2148_214872


namespace number_of_chocolates_bought_l2148_214888

theorem number_of_chocolates_bought (C S : ℝ) 
  (h1 : ∃ n : ℕ, n * C = 21 * S) 
  (h2 : (S - C) / C * 100 = 66.67) : 
  ∃ n : ℕ, n = 35 := 
by
  sorry

end number_of_chocolates_bought_l2148_214888


namespace no_real_solution_for_inequality_l2148_214804

theorem no_real_solution_for_inequality :
  ¬ ∃ a : ℝ, ∃ x : ℝ, ∀ b : ℝ, |x^2 + 4*a*x + 5*a| ≤ 3 :=
by
  sorry

end no_real_solution_for_inequality_l2148_214804


namespace smaller_base_length_trapezoid_l2148_214881

variable (p q a b : ℝ)
variable (h : p < q)
variable (angle_ratio : ∃ α, ((2 * α) : ℝ) = α + (α : ℝ))

theorem smaller_base_length_trapezoid :
  b = (p^2 + a * p - q^2) / p :=
sorry

end smaller_base_length_trapezoid_l2148_214881


namespace dessert_distribution_l2148_214863

theorem dessert_distribution 
  (mini_cupcakes : ℕ) 
  (donut_holes : ℕ) 
  (total_desserts : ℕ) 
  (students : ℕ) 
  (h1 : mini_cupcakes = 14)
  (h2 : donut_holes = 12) 
  (h3 : students = 13)
  (h4 : total_desserts = mini_cupcakes + donut_holes)
  : total_desserts / students = 2 :=
by sorry

end dessert_distribution_l2148_214863


namespace Dawn_sold_glasses_l2148_214885

variable (x : ℕ)

def Bea_price_per_glass : ℝ := 0.25
def Dawn_price_per_glass : ℝ := 0.28
def Bea_glasses_sold : ℕ := 10
def Bea_extra_earnings : ℝ := 0.26
def Bea_total_earnings : ℝ := Bea_glasses_sold * Bea_price_per_glass
def Dawn_total_earnings (x : ℕ) : ℝ := x * Dawn_price_per_glass

theorem Dawn_sold_glasses :
  Bea_total_earnings - Bea_extra_earnings = Dawn_total_earnings x → x = 8 :=
by
  sorry

end Dawn_sold_glasses_l2148_214885


namespace find_a_l2148_214805

theorem find_a (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) 
  (h1 : a^2 / b = 1) (h2 : b^2 / c = 2) (h3 : c^2 / a = 3) : 
  a = 12^(1/7 : ℝ) :=
by
  sorry

end find_a_l2148_214805


namespace farmer_pomelos_dozen_l2148_214835

theorem farmer_pomelos_dozen (pomelos_last_week : ℕ) (boxes_last_week : ℕ) (boxes_this_week : ℕ) :
  pomelos_last_week = 240 → boxes_last_week = 10 → boxes_this_week = 20 →
  (pomelos_last_week / boxes_last_week) * boxes_this_week / 12 = 40 := 
by
  intro h1 h2 h3
  sorry

end farmer_pomelos_dozen_l2148_214835


namespace exists_disjoint_nonempty_subsets_with_equal_sum_l2148_214852

theorem exists_disjoint_nonempty_subsets_with_equal_sum :
  ∀ (A : Finset ℕ), (A.card = 11) → (∀ a ∈ A, 1 ≤ a ∧ a ≤ 100) →
  ∃ (B C : Finset ℕ), B ≠ ∅ ∧ C ≠ ∅ ∧ B ∩ C = ∅ ∧ (B ∪ C ⊆ A) ∧ (B.sum id = C.sum id) :=
by
  sorry

end exists_disjoint_nonempty_subsets_with_equal_sum_l2148_214852


namespace sum_of_first_3_geometric_terms_eq_7_l2148_214837

theorem sum_of_first_3_geometric_terms_eq_7 
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * r)
  (h_ratio_gt_1 : r > 1)
  (h_eq : (a 0 + a 2 = 5) ∧ (a 0 * a 2 = 4)) 
  : (a 0 + a 1 + a 2) = 7 := 
by
  sorry

end sum_of_first_3_geometric_terms_eq_7_l2148_214837


namespace sum_divisible_by_ten_l2148_214883

    -- Given conditions
    def is_natural_number (n : ℕ) : Prop := true

    -- Sum S as defined in the conditions
    def S (n : ℕ) : ℕ := n ^ 2 + (n + 1) ^ 2 + (n + 2) ^ 2 + (n + 3) ^ 2

    -- The equivalent math proof problem in Lean 4 statement
    theorem sum_divisible_by_ten (n : ℕ) : S n % 10 = 0 ↔ n % 5 = 1 := by
      sorry
    
end sum_divisible_by_ten_l2148_214883


namespace option_costs_more_cost_effective_x30_more_cost_effective_plan_x30_l2148_214800

def racket_price : ℕ := 80
def ball_price : ℕ := 20
def discount_rate : ℕ := 90

def option_1_cost (n_rackets : ℕ) : ℕ :=
  n_rackets * racket_price

def option_2_cost (n_rackets : ℕ) (n_balls : ℕ) : ℕ :=
  (discount_rate * (n_rackets * racket_price + n_balls * ball_price)) / 100

-- Part 1: Express in Algebraic Terms
theorem option_costs (n_rackets : ℕ) (n_balls : ℕ) :
  option_1_cost n_rackets = 1600 ∧ option_2_cost n_rackets n_balls = 1440 + 18 * n_balls := 
by
  sorry

-- Part 2: For x = 30, determine more cost-effective option
theorem more_cost_effective_x30 (x : ℕ) (h : x = 30) :
  option_1_cost 20 < option_2_cost 20 x := 
by
  sorry

-- Part 3: More cost-effective Plan for x = 30
theorem more_cost_effective_plan_x30 :
  1600 + (discount_rate * (10 * ball_price)) / 100 < option_2_cost 20 30 :=
by
  sorry

end option_costs_more_cost_effective_x30_more_cost_effective_plan_x30_l2148_214800


namespace sin_and_tan_alpha_l2148_214894

variable (x : ℝ) (α : ℝ)

-- Conditions
def vertex_is_origin : Prop := true
def initial_side_is_non_negative_half_axis : Prop := true
def terminal_side_passes_through_P : Prop := ∃ (P : ℝ × ℝ), P = (x, -Real.sqrt 2)
def cos_alpha_eq : Prop := x ≠ 0 ∧ Real.cos α = (Real.sqrt 3 / 6) * x

-- Proof Problem Statement
theorem sin_and_tan_alpha (h1 : vertex_is_origin) 
                         (h2 : initial_side_is_non_negative_half_axis) 
                         (h3 : terminal_side_passes_through_P x) 
                         (h4 : cos_alpha_eq x α) 
                         : Real.sin α = -Real.sqrt 6 / 6 ∧ (Real.tan α = Real.sqrt 5 / 5 ∨ Real.tan α = -Real.sqrt 5 / 5) := 
sorry

end sin_and_tan_alpha_l2148_214894


namespace original_game_start_player_wins_modified_game_start_player_wins_l2148_214882

def divisor_game_condition (num : ℕ) := ∀ d : ℕ, d ∣ num → ∀ x : ℕ, x ∣ d → x = d ∨ x = 1
def modified_divisor_game_condition (num d_prev : ℕ) := ∀ d : ℕ, d ∣ num → d ≠ d_prev → ∃ k l : ℕ, d = k * l ∧ k ≠ 1 ∧ l ≠ 1 ∧ k ≤ l

/-- Prove that if the starting player plays wisely, they will always win the original game. -/
theorem original_game_start_player_wins : ∀ d : ℕ, divisor_game_condition 1000 → d = 100 → (∃ p : ℕ, p != 1000) := 
sorry

/-- What happens if the game is modified such that a divisor cannot be mentioned if it has fewer divisors than any previously mentioned number? -/
theorem modified_game_start_player_wins : ∀ d_prev : ℕ, modified_divisor_game_condition 1000 d_prev → d_prev = 100 → (∃ p : ℕ, p != 1000) := 
sorry

end original_game_start_player_wins_modified_game_start_player_wins_l2148_214882


namespace plane_equation_l2148_214813

-- We will create a structure for 3D points to use in our problem
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the problem conditions and the equation we want to prove
def containsPoint (p: Point3D) : Prop := p.x = 1 ∧ p.y = 4 ∧ p.z = -8

def onLine (p: Point3D) : Prop := 
  ∃ t : ℝ, 
    (p.x = 4 * t + 2) ∧ 
    (p.y = - t - 1) ∧ 
    (p.z = 5 * t + 3)

def planeEq (p: Point3D) : Prop := 
  -4 * p.x + 2 * p.y - 5 * p.z + 3 = 0

-- Now state the theorem
theorem plane_equation (p: Point3D) : 
  containsPoint p ∨ onLine p → planeEq p := 
  sorry

end plane_equation_l2148_214813


namespace Vasya_fraction_impossible_l2148_214853

theorem Vasya_fraction_impossible
  (a b n : ℕ) (h_ab : a < b) (h_na : n < a) (h_nb : n < b)
  (h1 : (a + n) / (b + n) > 3 * a / (2 * b))
  (h2 : (a - n) / (b - n) > a / (2 * b)) : false :=
by
  sorry

end Vasya_fraction_impossible_l2148_214853


namespace f_1982_l2148_214845

-- Define the function f and the essential properties and conditions
def f : ℕ → ℕ := sorry

axiom f_nonneg (n : ℕ) : f n ≥ 0
axiom f_add_property (m n : ℕ) : f (m + n) - f m - f n = 0 ∨ f (m + n) - f m - f n = 1
axiom f_2 : f 2 = 0
axiom f_3_pos : f 3 > 0
axiom f_9999 : f 9999 = 3333

-- Statement of the theorem we want to prove
theorem f_1982 : f 1982 = 660 := 
  by sorry

end f_1982_l2148_214845
