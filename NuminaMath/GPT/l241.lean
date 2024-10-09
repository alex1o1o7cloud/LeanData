import Mathlib

namespace three_five_seven_sum_fraction_l241_24197

theorem three_five_seven_sum_fraction :
  (3 * 5 * 7) * ((1 / 3) + (1 / 5) + (1 / 7)) = 71 :=
by
  sorry

end three_five_seven_sum_fraction_l241_24197


namespace Cameron_list_count_l241_24165

theorem Cameron_list_count : 
  let lower_bound := 900
  let upper_bound := 27000
  let step := 30
  let n_min := lower_bound / step
  let n_max := upper_bound / step
  n_max - n_min + 1 = 871 :=
by
  sorry

end Cameron_list_count_l241_24165


namespace sin_105_value_cos_75_value_trigonometric_identity_l241_24194

noncomputable def sin_105_eq : Real := Real.sin (105 * Real.pi / 180)
noncomputable def cos_75_eq : Real := Real.cos (75 * Real.pi / 180)
noncomputable def cos_pi_div_5_cos_3pi_div_10_minus_sin_pi_div_5_sin_3pi_div_10_eq : Real := 
  Real.cos (Real.pi / 5) * Real.cos (3 * Real.pi / 10) - Real.sin (Real.pi / 5) * Real.sin (3 * Real.pi / 10)

theorem sin_105_value : sin_105_eq = (Real.sqrt 6 + Real.sqrt 2) / 4 := 
  by sorry

theorem cos_75_value : cos_75_eq = (Real.sqrt 6 - Real.sqrt 2) / 4 := 
  by sorry

theorem trigonometric_identity : cos_pi_div_5_cos_3pi_div_10_minus_sin_pi_div_5_sin_3pi_div_10_eq = 0 := 
  by sorry

end sin_105_value_cos_75_value_trigonometric_identity_l241_24194


namespace triangle_area_ratio_l241_24107

open Set 

variables {X Y Z W : Type} 
variable [LinearOrder X]

noncomputable def ratio_areas (XW WZ : ℕ) (h : ℕ) : ℚ :=
  (8 * h : ℚ) / (12 * h)

theorem triangle_area_ratio (XW WZ : ℕ) (h : ℕ)
  (hXW : XW = 8)
  (hWZ : WZ = 12) :
  ratio_areas XW WZ h = 2 / 3 :=
by
  rw [hXW, hWZ]
  unfold ratio_areas
  norm_num
  sorry

end triangle_area_ratio_l241_24107


namespace number_of_senior_citizen_tickets_l241_24192

theorem number_of_senior_citizen_tickets 
    (A S : ℕ)
    (h1 : A + S = 529)
    (h2 : 25 * A + 15 * S = 9745) 
    : S = 348 := 
by
  sorry

end number_of_senior_citizen_tickets_l241_24192


namespace equal_papers_per_cousin_l241_24120

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

end equal_papers_per_cousin_l241_24120


namespace expression_values_l241_24119

variable {a b c : ℚ}

theorem expression_values (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 0) :
  (a / abs a + b / abs b + c / abs c - (a * b * c) / abs (a * b * c) = 2) ∨ 
  (a / abs a + b / abs b + c / abs c - (a * b * c) / abs (a * b * c) = -2) := 
sorry

end expression_values_l241_24119


namespace profit_correct_l241_24158

-- Conditions
def initial_outlay : ℕ := 10000
def cost_per_set : ℕ := 20
def selling_price_per_set : ℕ := 50
def sets : ℕ := 500

-- Definitions used in the problem
def manufacturing_cost : ℕ := initial_outlay + (sets * cost_per_set)
def revenue : ℕ := sets * selling_price_per_set
def profit : ℕ := revenue - manufacturing_cost

-- The theorem statement
theorem profit_correct : profit = 5000 := by
  sorry

end profit_correct_l241_24158


namespace arithmetic_sequence_15th_term_l241_24144

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

theorem arithmetic_sequence_15th_term :
  arithmetic_sequence (-3) 4 15 = 53 :=
by
  sorry

end arithmetic_sequence_15th_term_l241_24144


namespace birds_per_cup_l241_24125

theorem birds_per_cup :
  ∀ (C B S T : ℕ) (H1 : C = 2) (H2 : S = 1 / 2 * C) (H3 : T = 21) (H4 : B = 14),
    ((C - S) * B = T) :=
by
  sorry

end birds_per_cup_l241_24125


namespace outer_circle_increase_l241_24164

theorem outer_circle_increase : 
  let R_o := 6
  let R_i := 4
  let R_i_new := (3 : ℝ)  -- 4 * (3/4)
  let A_original := 20 * Real.pi  -- π * (6^2 - 4^2)
  let A_new := 72 * Real.pi  -- 3.6 * A_original
  ∃ (x : ℝ), 
    let R_o_new := R_o * (1 + x / 100)
    π * R_o_new^2 - π * R_i_new^2 = A_new →
    x = 50 := 
sorry

end outer_circle_increase_l241_24164


namespace jill_earnings_l241_24180

theorem jill_earnings :
  ∀ (hourly_wage : ℝ) (tip_rate : ℝ) (num_shifts : ℕ) (hours_per_shift : ℕ) (avg_orders_per_hour : ℝ),
  hourly_wage = 4.00 →
  tip_rate = 0.15 →
  num_shifts = 3 →
  hours_per_shift = 8 →
  avg_orders_per_hour = 40 →
  (num_shifts * hours_per_shift * hourly_wage + num_shifts * hours_per_shift * avg_orders_per_hour * tip_rate = 240) :=
by
  intros hourly_wage tip_rate num_shifts hours_per_shift avg_orders_per_hour
  intros hwage_eq trip_rate_eq nshifts_eq hshift_eq avgorder_eq
  sorry

end jill_earnings_l241_24180


namespace total_interest_correct_l241_24109

-- Definitions
def total_amount : ℝ := 3500
def P1 : ℝ := 1550
def P2 : ℝ := total_amount - P1
def rate1 : ℝ := 0.03
def rate2 : ℝ := 0.05

-- Total interest calculation
noncomputable def interest1 : ℝ := P1 * rate1
noncomputable def interest2 : ℝ := P2 * rate2
noncomputable def total_interest : ℝ := interest1 + interest2

-- Theorem statement
theorem total_interest_correct : total_interest = 144 := 
by
  -- Proof steps would go here
  sorry

end total_interest_correct_l241_24109


namespace find_other_endpoint_diameter_l241_24191

-- Define the given conditions
def center : ℝ × ℝ := (1, 2)
def endpoint_A : ℝ × ℝ := (4, 6)

-- Define a function to find the other endpoint
def other_endpoint (center endpoint_A : ℝ × ℝ) : ℝ × ℝ := 
  let vector_CA := (center.1 - endpoint_A.1, center.2 - endpoint_A.2)
  let vector_CB := (-vector_CA.1, -vector_CA.2)
  (center.1 + vector_CB.1, center.2 + vector_CB.2)

-- State the theorem
theorem find_other_endpoint_diameter : 
  ∀ center endpoint_A, other_endpoint center endpoint_A = (4, 6) :=
by
  intro center endpoint_A
  -- Proof would go here
  sorry

end find_other_endpoint_diameter_l241_24191


namespace light_year_scientific_notation_l241_24146

def sci_not_eq : Prop := 
  let x := 9500000000000
  let y := 9.5 * 10^12
  x = y

theorem light_year_scientific_notation : sci_not_eq :=
  by sorry

end light_year_scientific_notation_l241_24146


namespace gcd_60_90_l241_24183

theorem gcd_60_90 : Nat.gcd 60 90 = 30 := by
  let f1 : 60 = 2 ^ 2 * 3 * 5 := by sorry
  let f2 : 90 = 2 * 3 ^ 2 * 5 := by sorry
  sorry

end gcd_60_90_l241_24183


namespace seven_digit_number_insertion_l241_24117

theorem seven_digit_number_insertion (num : ℕ) (h : num = 52115) : (∃ (count : ℕ), count = 21) :=
by 
  sorry

end seven_digit_number_insertion_l241_24117


namespace parking_cost_savings_l241_24196

theorem parking_cost_savings
  (weekly_rate : ℕ := 10)
  (monthly_rate : ℕ := 24)
  (weeks_in_year : ℕ := 52)
  (months_in_year : ℕ := 12) :
  (weekly_rate * weeks_in_year) - (monthly_rate * months_in_year) = 232 :=
by
  sorry

end parking_cost_savings_l241_24196


namespace circle_symmetry_l241_24150

theorem circle_symmetry (a b : ℝ) 
  (h1 : ∀ x y : ℝ, (x - a)^2 + (y - b)^2 = 1 ↔ (x - 1)^2 + (y - 3)^2 = 1) 
  (symm_line : ∀ x y : ℝ, y = x + 1) : a + b = 2 :=
sorry

end circle_symmetry_l241_24150


namespace cubic_root_relationship_l241_24173

theorem cubic_root_relationship 
  (r : ℝ) (h : r^3 - r + 3 = 0) : 
  (r^2)^3 - 2 * (r^2)^2 + (r^2) - 9 = 0 := 
by 
  sorry

end cubic_root_relationship_l241_24173


namespace parabola_vertex_l241_24113

theorem parabola_vertex :
  (∃ x y : ℝ, y^2 + 6 * y + 4 * x - 7 = 0 ∧ (x, y) = (4, -3)) :=
sorry

end parabola_vertex_l241_24113


namespace number_of_friends_with_pears_l241_24131

-- Each friend either carries pears or oranges
def total_friends : Nat := 15
def friends_with_oranges : Nat := 6
def friends_with_pears : Nat := total_friends - friends_with_oranges

theorem number_of_friends_with_pears :
  friends_with_pears = 9 := by
  -- Proof steps would go here
  sorry

end number_of_friends_with_pears_l241_24131


namespace purchase_price_of_furniture_l241_24141

theorem purchase_price_of_furniture (marked_price discount_rate profit_rate : ℝ) 
(h_marked_price : marked_price = 132) 
(h_discount_rate : discount_rate = 0.1)
(h_profit_rate : profit_rate = 0.1)
: ∃ a : ℝ, (marked_price * (1 - discount_rate) - a = profit_rate * a) ∧ a = 108 := by
  sorry

end purchase_price_of_furniture_l241_24141


namespace problem1_problem2_l241_24126

theorem problem1 (m : ℝ) (H : m > 0) (p : ∀ x : ℝ, (x+1)*(x-5) ≤ 0 → 1 - m ≤ x ∧ x ≤ 1 + m) : m ≥ 4 :=
sorry

theorem problem2 (x : ℝ) (m : ℝ) (H : m = 5) (disj : ∀ x : ℝ, ((x+1)*(x-5) ≤ 0 ∨ (1 - m ≤ x ∧ x ≤ 1 + m))
) (conj : ¬ ∃ x : ℝ, (x+1)*(x-5) ≤ 0 ∧ (1 - m ≤ x ∧ x ≤ 1 + m)) : (-4 ≤ x ∧ x < -1) ∨ (5 < x ∧ x < 6) :=
sorry

end problem1_problem2_l241_24126


namespace habitat_limits_are_correct_l241_24101

-- Definitions of the conditions
def colonyA_doubling_days : ℕ := 22
def colonyB_tripling_days : ℕ := 30
def tripling_interval : ℕ := 2

-- Definitions to confirm they grow as described
def is_colonyA_habitat_limit_reached (days : ℕ) : Prop := days = colonyA_doubling_days
def is_colonyB_habitat_limit_reached (days : ℕ) : Prop := days = colonyB_tripling_days

-- Proof statement
theorem habitat_limits_are_correct :
  (is_colonyA_habitat_limit_reached colonyA_doubling_days) ∧ (is_colonyB_habitat_limit_reached colonyB_tripling_days) :=
by
  sorry

end habitat_limits_are_correct_l241_24101


namespace distinct_products_count_is_26_l241_24137

open Finset

def set_numbers : Finset ℕ := {2, 3, 5, 7, 11}

def distinct_products_count (s : Finset ℕ) : ℕ :=
  let pairs := s.powerset.filter (λ t => 2 ≤ t.card)
  pairs.card

theorem distinct_products_count_is_26 : distinct_products_count set_numbers = 26 := by
  sorry

end distinct_products_count_is_26_l241_24137


namespace set_M_real_l241_24160

noncomputable def set_M : Set ℂ := {z : ℂ | (z - 1) ^ 2 = Complex.abs (z - 1) ^ 2}

theorem set_M_real :
  set_M = {z : ℂ | ∃ x : ℝ, z = x} :=
by
  sorry

end set_M_real_l241_24160


namespace evaluate_expression_l241_24152

theorem evaluate_expression (a b : ℝ) (h1 : a = 4) (h2 : b = -1) : -2 * a ^ 2 - 3 * b ^ 2 + 2 * a * b = -43 :=
by
  sorry

end evaluate_expression_l241_24152


namespace maximum_value_of_d_l241_24189

theorem maximum_value_of_d 
  (d e : ℕ) 
  (h1 : 0 ≤ d ∧ d < 10) 
  (h2: 0 ≤ e ∧ e < 10) 
  (h3 : (18 + d + e) % 3 = 0) 
  (h4 : (15 - (d + e)) % 11 = 0) 
  : d ≤ 0 := 
sorry

end maximum_value_of_d_l241_24189


namespace relationship_between_a_and_b_l241_24104

variable (a b : ℝ)

-- Conditions: Points lie on the line y = 2x + 1
def point_M (a : ℝ) : Prop := a = 2 * 2 + 1
def point_N (b : ℝ) : Prop := b = 2 * 3 + 1

-- Prove that a < b given the conditions
theorem relationship_between_a_and_b (hM : point_M a) (hN : point_N b) : a < b := 
sorry

end relationship_between_a_and_b_l241_24104


namespace Pat_worked_days_eq_57_l241_24108

def Pat_earnings (x : ℕ) : ℤ := 100 * x
def Pat_food_costs (x : ℕ) : ℤ := 20 * (70 - x)
def total_balance (x : ℕ) : ℤ := Pat_earnings x - Pat_food_costs x

theorem Pat_worked_days_eq_57 (x : ℕ) (h : total_balance x = 5440) : x = 57 :=
by
  sorry

end Pat_worked_days_eq_57_l241_24108


namespace compare_y1_y2_l241_24132

-- Define the function
def f (x : ℝ) : ℝ := -3 * x + 1

-- Define the points
def y1 := f 1
def y2 := f 3

-- The theorem to be proved
theorem compare_y1_y2 : y1 > y2 :=
by
  -- Proof placeholder
  sorry

end compare_y1_y2_l241_24132


namespace greatest_power_of_two_factor_l241_24162

theorem greatest_power_of_two_factor (n m : ℕ) (h1 : n = 12) (h2 : m = 8) :
  ∃ k, k = 1209 ∧ 2^k ∣ n^603 - m^402 :=
by
  sorry

end greatest_power_of_two_factor_l241_24162


namespace harvest_weeks_l241_24182

/-- Lewis earns $403 every week during a certain number of weeks of harvest. 
If he has to pay $49 rent every week, and he earns $93,899 during the harvest season, 
we need to prove that the number of weeks in the harvest season is 265. --/
theorem harvest_weeks 
  (E : ℕ) (R : ℕ) (T : ℕ) (W : ℕ) 
  (hE : E = 403) (hR : R = 49) (hT : T = 93899) 
  (hW : W = 265) : 
  W = (T / (E - R)) := 
by sorry

end harvest_weeks_l241_24182


namespace neg_proposition_P_l241_24170

theorem neg_proposition_P : 
  (¬ (∀ x : ℝ, 2^x + x^2 > 0)) ↔ (∃ x0 : ℝ, 2^x0 + x0^2 ≤ 0) :=
by
  sorry

end neg_proposition_P_l241_24170


namespace max_product_of_xy_on_circle_l241_24178

theorem max_product_of_xy_on_circle (x y : ℤ) (h : x^2 + y^2 = 100) : 
  ∃ (x y : ℤ), (x^2 + y^2 = 100) ∧ (∀ x y : ℤ, x^2 + y^2 = 100 → x * y ≤ 48) ∧ x * y = 48 := by
  sorry

end max_product_of_xy_on_circle_l241_24178


namespace mark_purchased_cans_l241_24116

theorem mark_purchased_cans : ∀ (J M : ℕ), 
    (J = 40) → 
    (100 - J = 6 * M / 5) → 
    M = 27 := by
  sorry

end mark_purchased_cans_l241_24116


namespace oranges_in_shop_l241_24140

-- Define the problem conditions
def ratio (M O A : ℕ) : Prop := (10 * O = 2 * M) ∧ (10 * A = 3 * M)

noncomputable def numMangoes : ℕ := 120
noncomputable def numApples : ℕ := 36

-- Statement of the problem
theorem oranges_in_shop (ratio_factor : ℕ) (h_ratio : ratio numMangoes (2 * ratio_factor) numApples) :
  (2 * ratio_factor) = 24 := by
  sorry

end oranges_in_shop_l241_24140


namespace combination_exists_l241_24100

theorem combination_exists 
  (S T Ti : ℝ) (x y z : ℝ)
  (h : 3 * S + 4 * T + 2 * Ti = 40) :
  ∃ x y z : ℝ, x * S + y * T + z * Ti = 60 :=
sorry

end combination_exists_l241_24100


namespace domain_of_function_l241_24185

def domain_sqrt_log : Set ℝ :=
  {x | (2 - x ≥ 0) ∧ ((2 * x - 1) / (3 - x) > 0)}

theorem domain_of_function :
  domain_sqrt_log = {x | (1/2 < x) ∧ (x ≤ 2)} :=
by
  sorry

end domain_of_function_l241_24185


namespace exponent_division_simplification_l241_24102

theorem exponent_division_simplification :
  ((18^18 / 18^17)^2 * 9^2) / 3^4 = 324 :=
by
  sorry

end exponent_division_simplification_l241_24102


namespace solve_quadratic_substitution_l241_24115

theorem solve_quadratic_substitution : 
  (∀ x : ℝ, (2 * x - 5) ^ 2 - 2 * (2 * x - 5) - 3 = 0 ↔ x = 2 ∨ x = 4) :=
by
  sorry

end solve_quadratic_substitution_l241_24115


namespace find_y_given_conditions_l241_24122

theorem find_y_given_conditions (x y : ℝ) (h1 : x^(3 * y) = 27) (h2 : x = 3) : y = 1 := 
by
  sorry

end find_y_given_conditions_l241_24122


namespace jelly_beans_total_l241_24169

-- Definitions from the conditions
def vanilla : Nat := 120
def grape : Nat := 5 * vanilla + 50
def total : Nat := vanilla + grape

-- Statement to prove
theorem jelly_beans_total :
  total = 770 := 
by 
  sorry

end jelly_beans_total_l241_24169


namespace sum_first_60_terms_l241_24176

theorem sum_first_60_terms {a : ℕ → ℤ}
  (h : ∀ n, a (n + 1) + (-1)^n * a n = 2 * n - 1) :
  (Finset.range 60).sum a = 1830 :=
sorry

end sum_first_60_terms_l241_24176


namespace cos_F_l241_24184

theorem cos_F (D E F : ℝ) (hDEF : D + E + F = 180)
  (sin_D : Real.sin D = 4 / 5)
  (cos_E : Real.cos E = 12 / 13) :
  Real.cos F = -16 / 65 :=
  sorry

end cos_F_l241_24184


namespace gcd_g_x_l241_24112

def g (x : ℕ) : ℕ := (5 * x + 7) * (11 * x + 3) * (17 * x + 8) * (4 * x + 5)

theorem gcd_g_x (x : ℕ) (hx : 17280 ∣ x) : Nat.gcd (g x) x = 120 :=
by sorry

end gcd_g_x_l241_24112


namespace total_points_scored_l241_24157

theorem total_points_scored 
  (darius_score : ℕ) 
  (marius_score : ℕ) 
  (matt_score : ℕ) 
  (h1 : marius_score = darius_score + 3) 
  (h2 : darius_score = matt_score - 5)
  (h3 : darius_score = 10) : darius_score + marius_score + matt_score = 38 :=
by
  sorry

end total_points_scored_l241_24157


namespace greatest_x_plus_z_l241_24153

theorem greatest_x_plus_z (x y z c d : ℕ) (h1 : 1 ≤ x ∧ x ≤ 9) (h2 : 1 ≤ y ∧ y ≤ 9) (h3 : 1 ≤ z ∧ z ≤ 9)
  (h4 : 700 - c = 700)
  (h5 : 100 * x + 10 * y + z - (100 * z + 10 * y + x) = 693)
  (h6 : x > z) :
  d = 11 :=
by
  sorry

end greatest_x_plus_z_l241_24153


namespace correct_transformation_l241_24130

theorem correct_transformation (x : ℝ) :
  (6 * ((2 * x + 1) / 3) - 6 * ((10 * x + 1) / 6) = 6) ↔ (4 * x + 2 - 10 * x - 1 = 6) :=
by
  sorry

end correct_transformation_l241_24130


namespace possible_vertex_angles_of_isosceles_triangle_l241_24143

def isosceles_triangle (α β γ : ℝ) : Prop := (α = β) ∨ (β = γ) ∨ (γ = α)

def altitude_half_side (α β γ a b c : ℝ) : Prop :=
  (a = α / 2) ∨ (b = β / 2) ∨ (c = γ / 2)

theorem possible_vertex_angles_of_isosceles_triangle (α β γ a b c : ℝ) :
  isosceles_triangle α β γ →
  altitude_half_side α β γ a b c →
  α = 30 ∨ α = 120 ∨ α = 150 :=
by
  sorry

end possible_vertex_angles_of_isosceles_triangle_l241_24143


namespace balls_into_boxes_l241_24139

theorem balls_into_boxes :
  let balls := 7
  let boxes := 3
  let min_balls_per_box := 1
  let remaining_balls := balls - boxes * min_balls_per_box
  let combination := Nat.choose (remaining_balls + boxes - 1) (boxes - 1) 
  combination = 15 :=
by
  let balls := 7
  let boxes := 3
  let min_balls_per_box := 1
  let remaining_balls := balls - boxes * min_balls_per_box
  let combination := Nat.choose (remaining_balls + boxes - 1) (boxes - 1)
  show combination = 15
  sorry

end balls_into_boxes_l241_24139


namespace minimum_and_maximum_attendees_more_than_one_reunion_l241_24133

noncomputable def minimum_attendees_more_than_one_reunion (total_guests oates_attendees hall_attendees brown_attendees : ℕ) : ℕ :=
  let total_unique_attendees := oates_attendees + hall_attendees + brown_attendees
  total_unique_attendees - total_guests

noncomputable def maximum_attendees_more_than_one_reunion (total_guests oates_attendees hall_attendees brown_attendees : ℕ) : ℕ :=
  oates_attendees

theorem minimum_and_maximum_attendees_more_than_one_reunion
  (total_guests oates_attendees hall_attendees brown_attendees : ℕ)
  (H1 : total_guests = 200)
  (H2 : oates_attendees = 60)
  (H3 : hall_attendees = 90)
  (H4 : brown_attendees = 80) :
  minimum_attendees_more_than_one_reunion total_guests oates_attendees hall_attendees brown_attendees = 30 ∧
  maximum_attendees_more_than_one_reunion total_guests oates_attendees hall_attendees brown_attendees = 60 :=
by
  sorry

end minimum_and_maximum_attendees_more_than_one_reunion_l241_24133


namespace smallest_unwritable_number_l241_24193

theorem smallest_unwritable_number :
  ∀ a b c d : ℕ, 11 ≠ (2^a - 2^b) / (2^c - 2^d) := sorry

end smallest_unwritable_number_l241_24193


namespace five_to_one_ratio_to_eleven_is_fifty_five_l241_24151

theorem five_to_one_ratio_to_eleven_is_fifty_five (y : ℚ) (h : 5 / 1 = y / 11) : y = 55 :=
by
  sorry

end five_to_one_ratio_to_eleven_is_fifty_five_l241_24151


namespace min_value_frac_inv_sum_l241_24114

theorem min_value_frac_inv_sum (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + 3 * y = 1) : 
  ∃ (minimum_value : ℝ), minimum_value = 4 + 2 * Real.sqrt 3 ∧ (∀ (a b : ℝ), 0 < a → 0 < b → a + 3 * b = 1 → (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3) := 
sorry

end min_value_frac_inv_sum_l241_24114


namespace parabola_y_coordinate_l241_24134

theorem parabola_y_coordinate (x y : ℝ) :
  x^2 = 4 * y ∧ (x - 0)^2 + (y - 1)^2 = 16 → y = 3 :=
by
  sorry

end parabola_y_coordinate_l241_24134


namespace oldest_child_age_l241_24199

theorem oldest_child_age (x : ℕ) (h : (6 + 8 + x) / 3 = 9) : x = 13 := 
by {
  sorry
}

end oldest_child_age_l241_24199


namespace find_speed_l241_24161

theorem find_speed (v d : ℝ) (h1 : d > 0) (h2 : 1.10 * v > 0) (h3 : 84 = 2 * d / (d / v + d / (1.10 * v))) : v = 80.18 := 
sorry

end find_speed_l241_24161


namespace tel_aviv_rain_days_l241_24186

-- Define the conditions
def chance_of_rain : ℝ := 0.5
def days_considered : ℕ := 6
def given_probability : ℝ := 0.234375

-- Helper function to compute binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability function P(X = k)
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k) * (p ^ k) * ((1 - p) ^ (n - k))

-- The main theorem to prove
theorem tel_aviv_rain_days :
  ∃ k, binomial_probability days_considered k chance_of_rain = given_probability ∧ k = 2 := by
  sorry

end tel_aviv_rain_days_l241_24186


namespace point_of_tangency_of_circles_l241_24148

/--
Given two circles defined by the following equations:
1. \( x^2 - 2x + y^2 - 10y + 17 = 0 \)
2. \( x^2 - 8x + y^2 - 10y + 49 = 0 \)
Prove that the coordinates of the point of tangency of these circles are \( (2.5, 5) \).
-/
theorem point_of_tangency_of_circles :
  (∃ x y : ℝ, (x^2 - 2*x + y^2 - 10*y + 17 = 0) ∧ (x = 2.5) ∧ (y = 5)) ∧ 
  (∃ x' y' : ℝ, (x'^2 - 8*x' + y'^2 - 10*y' + 49 = 0) ∧ (x' = 2.5) ∧ (y' = 5)) :=
sorry

end point_of_tangency_of_circles_l241_24148


namespace probability_triangle_l241_24175

noncomputable def points : List (ℕ × ℕ) := [(0, 0), (2, 0), (1, 1), (0, 2), (2, 2), (3, 3)]

def collinear (p1 p2 p3 : (ℕ × ℕ)) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

def is_triangle (p1 p2 p3 : (ℕ × ℕ)) : Prop := ¬ collinear p1 p2 p3

axiom collinear_ACEF : collinear (0, 0) (1, 1) (2, 2) ∧ collinear (0, 0) (1, 1) (3, 3) ∧ collinear (1, 1) (2, 2) (3, 3)
axiom collinear_BCD : collinear (2, 0) (1, 1) (0, 2)

theorem probability_triangle : 
  let total := 20
  let collinear_ACEF := 4
  let collinear_BCD := 1
  (total - collinear_ACEF - collinear_BCD) / total = 3 / 4 :=
by
  sorry

end probability_triangle_l241_24175


namespace solve_for_x_l241_24149

variable {x : ℝ}

def is_positive (x : ℝ) : Prop := x > 0

def area_of_triangle_is_150 (x : ℝ) : Prop :=
  let base := 2 * x
  let height := 3 * x
  (1/2) * base * height = 150

theorem solve_for_x (hx : is_positive x) (ha : area_of_triangle_is_150 x) : x = 5 * Real.sqrt 2 := by
  sorry

end solve_for_x_l241_24149


namespace A_on_curve_slope_at_A_l241_24188

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x ^ 2

-- Define the point A on the curve f
def A : ℝ × ℝ := (2, 8)

-- Define the condition that A is on the curve f
theorem A_on_curve : A.2 = f A.1 := by
  -- * left as a proof placeholder
  sorry

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4 * x

-- State and prove the main theorem
theorem slope_at_A : (deriv f) 2 = 8 := by
  -- * left as a proof placeholder
  sorry

end A_on_curve_slope_at_A_l241_24188


namespace each_partner_percentage_l241_24128

-- Defining the conditions as variables
variables (total_profit majority_share combined_amount : ℝ) (num_partners : ℕ)

-- Given conditions
def majority_owner_received_25_percent_of_total : total_profit * 0.25 = majority_share := sorry
def remaining_profit_distribution : total_profit - majority_share = 60000 := sorry
def combined_share_of_three : majority_share + 30000 = combined_amount := sorry
def total_profit_amount : total_profit = 80000 := sorry
def number_of_partners : num_partners = 4 := sorry

-- The theorem to be proven
theorem each_partner_percentage :
  ∃ (percent : ℝ), percent = 25 :=
sorry

end each_partner_percentage_l241_24128


namespace circumference_of_base_l241_24179

-- Definitions used for the problem
def radius : ℝ := 6
def sector_angle : ℝ := 300
def full_circle_angle : ℝ := 360

-- Ask for the circumference of the base of the cone formed by the sector
theorem circumference_of_base (r : ℝ) (theta_sector : ℝ) (theta_full : ℝ) :
  (theta_sector / theta_full) * (2 * π * r) = 10 * π :=
by
  sorry

end circumference_of_base_l241_24179


namespace new_dressing_contains_12_percent_vinegar_l241_24168

-- Definitions
def new_dressing_vinegar_percentage (p_vinegar q_vinegar p_fraction q_fraction : ℝ) : ℝ :=
  p_vinegar * p_fraction + q_vinegar * q_fraction

-- Conditions
def p_vinegar : ℝ := 0.30
def q_vinegar : ℝ := 0.10
def p_fraction : ℝ := 0.10
def q_fraction : ℝ := 0.90

-- The theorem to be proven
theorem new_dressing_contains_12_percent_vinegar :
  new_dressing_vinegar_percentage p_vinegar q_vinegar p_fraction q_fraction = 0.12 := 
by
  -- The proof is omitted here
  sorry

end new_dressing_contains_12_percent_vinegar_l241_24168


namespace evaluate_expression_l241_24155

def acbd (a b c d : ℝ) : ℝ := a * d - b * c

theorem evaluate_expression (x : ℝ) (h : x^2 - 3 * x + 1 = 0) :
  acbd (x + 1) (x - 2) (3 * x) (x - 1) = 1 := 
by
  sorry

end evaluate_expression_l241_24155


namespace same_side_of_line_l241_24106

open Real

theorem same_side_of_line (a : ℝ) :
  let O := (0, 0)
  let A := (1, 1)
  (O.1 + O.2 < a ↔ A.1 + A.2 < a) →
  a < 0 ∨ a > 2 := by
  sorry

end same_side_of_line_l241_24106


namespace gcd_lcm_identity_l241_24129

theorem gcd_lcm_identity (a b: ℕ) (h_lcm: (Nat.lcm a b) = 4620) (h_gcd: Nat.gcd a b = 33) (h_a: a = 231) : b = 660 := by
  sorry

end gcd_lcm_identity_l241_24129


namespace cuboid_length_l241_24124

variable (L W H V : ℝ)

theorem cuboid_length (W_eq : W = 4) (H_eq : H = 6) (V_eq : V = 96) (Volume_eq : V = L * W * H) : L = 4 :=
by
  sorry

end cuboid_length_l241_24124


namespace sum_lent_l241_24118

theorem sum_lent (P : ℝ) (r : ℝ) (t : ℝ) (I : ℝ) 
  (h1 : r = 0.06) (h2 : t = 8) (h3 : I = P - 520) : P * r * t = I → P = 1000 :=
by
  -- Given conditions
  intros
  -- Sorry placeholder
  sorry

end sum_lent_l241_24118


namespace correct_transformation_l241_24195

-- Given conditions
variables {a b : ℝ}
variable (h : 3 * a = 4 * b)
variable (a_nonzero : a ≠ 0)
variable (b_nonzero : b ≠ 0)

-- Statement of the problem
theorem correct_transformation : (a / 4) = (b / 3) :=
sorry

end correct_transformation_l241_24195


namespace right_triangle_legs_sum_l241_24198

-- Definitions
def sum_of_legs (a b : ℕ) : ℕ := a + b

-- Main theorem statement
theorem right_triangle_legs_sum (x : ℕ) (h : x^2 + (x + 1)^2 = 53^2) :
  sum_of_legs x (x + 1) = 75 :=
sorry

end right_triangle_legs_sum_l241_24198


namespace math_problem_l241_24103

noncomputable def f (x : ℝ) : ℝ := sorry

theorem math_problem (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (1 - x) = f (1 + x))
  (h2 : ∀ x : ℝ, f (-x) = -f x)
  (h3 : ∀ {x y : ℝ}, (0 ≤ x → x < y → y ≤ 1 → f x < f y)) :
  (f 0 = 0) ∧ 
  (∀ x : ℝ, f (x + 2) = f (-x)) ∧ 
  (∀ x : ℝ, x = -1 ∨ ∀ ε > 0, ε ≠ (x + 1))
:= sorry

end math_problem_l241_24103


namespace sin_cos_sixth_power_l241_24156

theorem sin_cos_sixth_power (θ : ℝ) (h : Real.sin (2 * θ) = 1/2) : 
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 13 / 16 :=
by
  sorry

end sin_cos_sixth_power_l241_24156


namespace sum_complex_l241_24181

-- Define the given complex numbers
def z1 : ℂ := ⟨2, 5⟩
def z2 : ℂ := ⟨3, -7⟩

-- State the theorem to prove the sum
theorem sum_complex : z1 + z2 = ⟨5, -2⟩ :=
by
  sorry

end sum_complex_l241_24181


namespace max_chain_triangles_l241_24110

theorem max_chain_triangles (n : ℕ) (h : n > 0) : 
  ∃ k, k = n^2 - n + 1 := 
sorry

end max_chain_triangles_l241_24110


namespace Albert_cabbage_count_l241_24166

-- Define the conditions
def rows := 12
def heads_per_row := 15

-- State the theorem
theorem Albert_cabbage_count : rows * heads_per_row = 180 := 
by sorry

end Albert_cabbage_count_l241_24166


namespace radius_inscribed_sphere_quadrilateral_pyramid_l241_24177

noncomputable def radius_of_inscribed_sphere (a : ℝ) : ℝ :=
  a * (Real.sqrt 5 - 1) / 4

theorem radius_inscribed_sphere_quadrilateral_pyramid (a : ℝ) :
  r = radius_of_inscribed_sphere a :=
by
  -- problem conditions:
  -- side of the base a
  -- height a
  -- result: r = a * (Real.sqrt 5 - 1) / 4
  sorry

end radius_inscribed_sphere_quadrilateral_pyramid_l241_24177


namespace marc_average_speed_l241_24190

theorem marc_average_speed 
  (d : ℝ) -- Define d as a real number representing distance
  (chantal_speed1 : ℝ := 3) -- Chantal's speed for the first half
  (chantal_speed2 : ℝ := 1.5) -- Chantal's speed for the second half
  (chantal_speed3 : ℝ := 2) -- Chantal's speed while descending
  (marc_meeting_point : ℝ := (2 / 3) * d) -- One-third point from the trailhead
  (chantal_time1 : ℝ := d / chantal_speed1) 
  (chantal_time2 : ℝ := (d / chantal_speed2))
  (chantal_time3 : ℝ := (d / 6)) -- Chantal's time for the descent from peak to one-third point
  (total_time : ℝ := chantal_time1 + chantal_time2 + chantal_time3) : 
  marc_meeting_point / total_time = 12 / 13 := 
  by 
  -- Leaving the proof as sorry to indicate where the proof would be
  sorry

end marc_average_speed_l241_24190


namespace total_nails_l241_24127

def num_planks : Nat := 1
def nails_per_plank : Nat := 3
def additional_nails : Nat := 8

theorem total_nails : (num_planks * nails_per_plank + additional_nails) = 11 :=
by
  sorry

end total_nails_l241_24127


namespace hours_between_dates_not_thirteen_l241_24111

def total_hours (start_date: ℕ × ℕ × ℕ × ℕ) (end_date: ℕ × ℕ × ℕ × ℕ) (days_in_dec: ℕ) : ℕ :=
  let (start_year, start_month, start_day, start_hour) := start_date
  let (end_year, end_month, end_day, end_hour) := end_date
  (days_in_dec - start_day) * 24 - start_hour + end_day * 24 + end_hour

theorem hours_between_dates_not_thirteen :
  let start_date := (2015, 12, 30, 23)
  let end_date := (2016, 1, 1, 12)
  let days_in_dec := 31
  total_hours start_date end_date days_in_dec ≠ 13 :=
by
  sorry

end hours_between_dates_not_thirteen_l241_24111


namespace sequence_an_general_formula_sequence_bn_sum_l241_24163

theorem sequence_an_general_formula
  (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h1 : ∀ n, S n = (3 / 2) * a n - (1 / 2) * a 1)
  (h2 : 2 * (a 2 + 6) = a 1 + a 3) :
  ∀ n, a n = 3 ^ n := sorry

theorem sequence_bn_sum
  (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h1 : ∀ n, S n = (3 / 2) * a n - (1 / 2) * a 1)
  (h2 : 2 * (a 2 + 6) = a 1 + a 3)
  (h3 : ∀ n, b n = a (n + 1) / (S n * S (n + 1))) :
  ∀ n, T n = (2 / 3) * (1 / 2 - 1 / (3 ^ (n + 1) - 1)) := sorry

end sequence_an_general_formula_sequence_bn_sum_l241_24163


namespace set_non_neg_even_set_primes_up_to_10_eq_sol_set_l241_24123

noncomputable def non_neg_even (x : ℕ) : Prop := x % 2 = 0 ∧ x ≤ 10
def primes_up_to_10 (x : ℕ) : Prop := Nat.Prime x ∧ x ≤ 10
def eq_sol (x : ℤ) : Prop := x^2 + 2*x - 15 = 0

theorem set_non_neg_even :
  {x : ℕ | non_neg_even x} = {0, 2, 4, 6, 8, 10} := by
  sorry

theorem set_primes_up_to_10 :
  {x : ℕ | primes_up_to_10 x} = {2, 3, 5, 7} := by
  sorry

theorem eq_sol_set :
  {x : ℤ | eq_sol x} = {-5, 3} := by
  sorry

end set_non_neg_even_set_primes_up_to_10_eq_sol_set_l241_24123


namespace terminating_decimal_expansion_l241_24145

theorem terminating_decimal_expansion (a b : ℕ) (h : 1600 = 2^6 * 5^2) :
  (13 : ℚ) / 1600 = 65 / 1000 :=
by
  sorry

end terminating_decimal_expansion_l241_24145


namespace sum_of_interior_edges_l241_24171

theorem sum_of_interior_edges (frame_width : ℕ) (frame_area : ℕ) (outer_edge : ℕ) 
  (H1 : frame_width = 2) (H2 : frame_area = 32) (H3 : outer_edge = 7) : 
  2 * (outer_edge - 2 * frame_width) + 2 * (x : ℕ) = 8 :=
by
  sorry

end sum_of_interior_edges_l241_24171


namespace median_number_of_children_is_three_l241_24174

/-- Define the context of the problem with total number of families. -/
def total_families : Nat := 15

/-- Prove that given the conditions, the median number of children is 3. -/
theorem median_number_of_children_is_three 
  (h : total_families = 15) : 
  ∃ median : Nat, median = 3 :=
by
  sorry

end median_number_of_children_is_three_l241_24174


namespace incorrect_statement_among_given_options_l241_24187

theorem incorrect_statement_among_given_options :
  (∀ (b h : ℝ), 3 * (b * h) = (3 * b) * h) ∧
  (∀ (b h : ℝ), 3 * (1 / 2 * b * h) = 1 / 2 * b * (3 * h)) ∧
  (∀ (π r : ℝ), 9 * (π * r * r) ≠ (π * (3 * r) * (3 * r))) ∧
  (∀ (a b : ℝ), (3 * a) / (2 * b) ≠ a / b) ∧
  (∀ (x : ℝ), x < 0 → 3 * x < x) →
  false :=
by
  sorry

end incorrect_statement_among_given_options_l241_24187


namespace circular_garden_radius_l241_24142

theorem circular_garden_radius
  (r : ℝ) -- radius of the circular garden
  (h : 2 * Real.pi * r = (1 / 6) * Real.pi * r^2) :
  r = 12 := 
by {
  sorry
}

end circular_garden_radius_l241_24142


namespace geometric_sequence_problem_l241_24167

variable {α : Type*} [LinearOrder α] [Field α]

def is_geometric_sequence (a : ℕ → α) :=
  ∀ n : ℕ, a (n + 1) * a (n - 1) = a n ^ 2

theorem geometric_sequence_problem (a : ℕ → α) (r : α) (h1 : a 1 = 1) (h2 : is_geometric_sequence a) (h3 : a 3 * a 5 = 4 * (a 4 - 1)) : 
  a 7 = 4 :=
by
  sorry

end geometric_sequence_problem_l241_24167


namespace determine_x_l241_24136

theorem determine_x (x : ℚ) : 
  x + 5 / 8 = 2 + 3 / 16 - 2 / 3 → 
  x = 43 / 48 := 
by
  intro h
  sorry

end determine_x_l241_24136


namespace geometric_sequence_ratio_l241_24154

theorem geometric_sequence_ratio (a b : ℕ → ℝ) (A B : ℕ → ℝ)
  (hA9 : A 9 = (a 5) ^ 9)
  (hB9 : B 9 = (b 5) ^ 9)
  (h_ratio : a 5 / b 5 = 2) :
  (A 9 / B 9) = 512 := by
  sorry

end geometric_sequence_ratio_l241_24154


namespace representable_as_product_l241_24147

theorem representable_as_product (n : ℤ) (p q : ℚ) (h1 : n > 1995) (h2 : 0 < p) (h3 : p < 1) :
  ∃ (terms : List ℚ), p = terms.prod ∧ ∀ t ∈ terms, ∃ n, t = (n^2 - 1995^2) / (n^2 - 1994^2) ∧ n > 1995 :=
sorry

end representable_as_product_l241_24147


namespace part1_part2_l241_24135

noncomputable def seq (n : ℕ) : ℚ :=
  match n with
  | 0     => 0  -- since there is no a_0 (we use ℕ*), we set it to 0
  | 1     => 1/3
  | n + 1 => seq n + (seq n) ^ 2 / (n : ℚ) ^ 2

theorem part1 (n : ℕ) (h : 0 < n) :
  seq n < seq (n + 1) ∧ seq (n + 1) < 1 :=
sorry

theorem part2 (n : ℕ) (h : 0 < n) :
  seq n > 1/2 - 1/(4 * n) :=
sorry

end part1_part2_l241_24135


namespace hyperbola_eccentricity_l241_24121

-- Definitions based on conditions
def hyperbola (x y : ℝ) (a : ℝ) := x^2 / a^2 - y^2 / 5 = 1

-- Main theorem
theorem hyperbola_eccentricity (a : ℝ) (c : ℝ) (h_focus : c = 3) (h_hyperbola : hyperbola 0 0 a) (focus_condition : c^2 = a^2 + 5) :
  c / a = 3 / 2 :=
by
  sorry

end hyperbola_eccentricity_l241_24121


namespace infinite_series_sum_l241_24159

theorem infinite_series_sum :
  (∑' n : ℕ, (2 * (n + 1) * (n + 1) + (n + 1) + 1) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 2))) = 5 / 6 := by
  sorry

end infinite_series_sum_l241_24159


namespace dave_apps_added_l241_24138

theorem dave_apps_added (initial_apps : ℕ) (total_apps_after_adding : ℕ) (apps_added : ℕ) 
  (h1 : initial_apps = 17) (h2 : total_apps_after_adding = 18) 
  (h3 : total_apps_after_adding = initial_apps + apps_added) : 
  apps_added = 1 := 
by
  -- proof omitted
  sorry

end dave_apps_added_l241_24138


namespace fans_who_received_all_three_l241_24105

theorem fans_who_received_all_three (n : ℕ) :
  (∀ k, k ≤ 5000 → (k % 100 = 0 → k % 40 = 0 → k % 60 = 0 → k / 600 ≤ n)) ∧
  (∀ k, k ≤ 5000 → (k % 100 = 0 → k % 40 = 0 → k % 60 = 0 → k / 600 ≤ 8)) :=
by
  sorry

end fans_who_received_all_three_l241_24105


namespace union_of_A_and_B_l241_24172

open Set

theorem union_of_A_and_B : 
  let A := {x : ℝ | x + 2 > 0}
  let B := {y : ℝ | ∃ (x : ℝ), y = Real.cos x}
  A ∪ B = {z : ℝ | z > -2} := 
by
  intros
  sorry

end union_of_A_and_B_l241_24172
