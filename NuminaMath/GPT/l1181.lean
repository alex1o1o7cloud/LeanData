import Mathlib

namespace cone_slant_height_l1181_118177

noncomputable def slant_height (r : ℝ) (CSA : ℝ) : ℝ := CSA / (Real.pi * r)

theorem cone_slant_height : slant_height 10 628.3185307179587 = 20 :=
by
  sorry

end cone_slant_height_l1181_118177


namespace vampire_conversion_l1181_118174

theorem vampire_conversion (x : ℕ) 
  (h_population : village_population = 300)
  (h_initial_vampires : initial_vampires = 2)
  (h_two_nights_vampires : 2 + 2 * x + x * (2 + 2 * x) = 72) :
  x = 5 :=
by
  -- Proof will be added here
  sorry

end vampire_conversion_l1181_118174


namespace factorial_fraction_eq_zero_l1181_118141

theorem factorial_fraction_eq_zero :
  ((5 * (Nat.factorial 7) - 35 * (Nat.factorial 6)) / Nat.factorial 8 = 0) :=
by
  sorry

end factorial_fraction_eq_zero_l1181_118141


namespace axis_of_symmetry_of_quadratic_l1181_118146

theorem axis_of_symmetry_of_quadratic (m : ℝ) :
  (∀ x : ℝ, -x^2 + 2 * m * x - m^2 + 3 = -x^2 + 2 * m * x - m^2 + 3) ∧ (∃ x : ℝ, x + 2 = 0) → m = -2 :=
by
  sorry

end axis_of_symmetry_of_quadratic_l1181_118146


namespace set_intersection_complement_l1181_118130

open Set

theorem set_intersection_complement (U A B : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2}) (hB : B = {2, 3}) :
  (U \ A) ∩ B = {3} :=
by
  sorry

end set_intersection_complement_l1181_118130


namespace largest_4_digit_congruent_to_17_mod_26_l1181_118136

theorem largest_4_digit_congruent_to_17_mod_26 :
  ∃ x, x < 10000 ∧ x ≥ 1000 ∧ x % 26 = 17 ∧ (∀ y, y < 10000 ∧ y ≥ 1000 ∧ y % 26 = 17 → y ≤ x) ∧ x = 9972 := 
by
  sorry

end largest_4_digit_congruent_to_17_mod_26_l1181_118136


namespace infinite_solutions_iff_c_is_5_over_2_l1181_118124

theorem infinite_solutions_iff_c_is_5_over_2 (c : ℝ) :
  (∀ y : ℝ, 3 * (2 + 2 * c * y) = 15 * y + 6) ↔ c = 5 / 2 :=
by 
  sorry

end infinite_solutions_iff_c_is_5_over_2_l1181_118124


namespace find_b_for_perpendicular_lines_l1181_118182

theorem find_b_for_perpendicular_lines:
  (∃ b : ℝ, ∀ (x y : ℝ), (3 * x + y - 5 = 0) ∧ (b * x + y + 2 = 0) → b = -1/3) :=
by
  sorry

end find_b_for_perpendicular_lines_l1181_118182


namespace project_inflation_cost_increase_l1181_118188

theorem project_inflation_cost_increase :
  let original_lumber_cost := 450
  let original_nails_cost := 30
  let original_fabric_cost := 80
  let lumber_inflation := 0.2
  let nails_inflation := 0.1
  let fabric_inflation := 0.05
  
  let new_lumber_cost := original_lumber_cost * (1 + lumber_inflation)
  let new_nails_cost := original_nails_cost * (1 + nails_inflation)
  let new_fabric_cost := original_fabric_cost * (1 + fabric_inflation)
  
  let total_increased_cost := (new_lumber_cost - original_lumber_cost) 
                            + (new_nails_cost - original_nails_cost) 
                            + (new_fabric_cost - original_fabric_cost)
  total_increased_cost = 97 := sorry

end project_inflation_cost_increase_l1181_118188


namespace distinct_terms_count_l1181_118138

theorem distinct_terms_count
  (x y z w p q r s t : Prop)
  (h1 : ¬(x = y ∨ x = z ∨ x = w ∨ y = z ∨ y = w ∨ z = w))
  (h2 : ¬(p = q ∨ p = r ∨ p = s ∨ p = t ∨ q = r ∨ q = s ∨ q = t ∨ r = s ∨ r = t ∨ s = t)) :
  ∃ (n : ℕ), n = 20 := by
  sorry

end distinct_terms_count_l1181_118138


namespace commute_time_absolute_difference_l1181_118133

theorem commute_time_absolute_difference (x y : ℝ)
  (h1 : (x + y + 10 + 11 + 9) / 5 = 10)
  (h2 : (x - 10)^2 + (y - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (9 - 10)^2 = 10) :
  |x - y| = 4 :=
by
  sorry

end commute_time_absolute_difference_l1181_118133


namespace max_value_of_z_l1181_118180

theorem max_value_of_z : ∀ x : ℝ, (x^2 - 14 * x + 10 ≤ 0 - 39) :=
by
  sorry

end max_value_of_z_l1181_118180


namespace cooper_age_l1181_118144

variable (Cooper Dante Maria : ℕ)

-- Conditions
def sum_of_ages : Prop := Cooper + Dante + Maria = 31
def dante_twice_cooper : Prop := Dante = 2 * Cooper
def maria_one_year_older : Prop := Maria = Dante + 1

theorem cooper_age (h1 : sum_of_ages Cooper Dante Maria) (h2 : dante_twice_cooper Cooper Dante) (h3 : maria_one_year_older Dante Maria) : Cooper = 6 :=
by
  sorry

end cooper_age_l1181_118144


namespace range_of_a_l1181_118117

theorem range_of_a (f : ℝ → ℝ) (h_mono_dec : ∀ x1 x2, -2 ≤ x1 ∧ x1 ≤ 2 ∧ -2 ≤ x2 ∧ x2 ≤ 2 → x1 < x2 → f x1 > f x2) 
  (h_cond : ∀ a, -2 ≤ a + 1 ∧ a + 1 ≤ 2 ∧ -2 ≤ 2 * a ∧ 2 * a ≤ 2 → f (a + 1) < f (2 * a)) :
  { a : ℝ | -1 ≤ a ∧ a < 1 } :=
sorry

end range_of_a_l1181_118117


namespace amount_a_receives_l1181_118131

theorem amount_a_receives (a b c : ℕ) (h1 : a + b + c = 50000) (h2 : a = b + 4000) (h3 : b = c + 5000) :
  (21000 / 50000) * 36000 = 15120 :=
by
  sorry

end amount_a_receives_l1181_118131


namespace apples_total_l1181_118173

theorem apples_total (lexie_apples : ℕ) (tom_apples : ℕ) (h1 : lexie_apples = 12) (h2 : tom_apples = 2 * lexie_apples) : lexie_apples + tom_apples = 36 :=
by
  sorry

end apples_total_l1181_118173


namespace solve_inequality_l1181_118159

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def given_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, x >= 0 → f x = x^3 - 8

theorem solve_inequality (f : ℝ → ℝ) (h_even : even_function f) (h_given : given_function f) :
  {x | f (x - 2) > 0} = {x | x < 0 ∨ x > 4} :=
by
  sorry

end solve_inequality_l1181_118159


namespace probability_three_dice_sum_to_fourth_l1181_118125

-- Define the probability problem conditions
def total_outcomes : ℕ := 8^4
def favorable_outcomes : ℕ := 1120

-- Final probability for the problem
def probability (favorable : ℕ) (total : ℕ) : ℚ := favorable / total

-- Lean statement for the proof problem
theorem probability_three_dice_sum_to_fourth :
  probability favorable_outcomes total_outcomes = 35 / 128 :=
by sorry

end probability_three_dice_sum_to_fourth_l1181_118125


namespace ann_susan_age_sum_l1181_118163

theorem ann_susan_age_sum (ann_age : ℕ) (susan_age : ℕ) (h1 : ann_age = 16) (h2 : ann_age = susan_age + 5) : ann_age + susan_age = 27 :=
by
  sorry

end ann_susan_age_sum_l1181_118163


namespace number_of_multiples_of_4_l1181_118123

theorem number_of_multiples_of_4 (a b : ℤ) (h1 : 100 < a) (h2 : b < 500) (h3 : a % 4 = 0) (h4 : b % 4 = 0) : 
  ∃ n : ℤ, n = 99 :=
by
  sorry

end number_of_multiples_of_4_l1181_118123


namespace triangle_inequality_l1181_118178

variable {a b c : ℝ}

theorem triangle_inequality (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) :
  (a / Real.sqrt (2*b^2 + 2*c^2 - a^2)) + (b / Real.sqrt (2*c^2 + 2*a^2 - b^2)) + 
  (c / Real.sqrt (2*a^2 + 2*b^2 - c^2)) ≥ Real.sqrt 3 := by
  sorry

end triangle_inequality_l1181_118178


namespace chef_additional_wings_l1181_118109

theorem chef_additional_wings
    (n : ℕ) (w_initial : ℕ) (w_per_friend : ℕ) (w_additional : ℕ)
    (h1 : n = 4)
    (h2 : w_initial = 9)
    (h3 : w_per_friend = 4)
    (h4 : w_additional = 7) :
    n * w_per_friend - w_initial = w_additional :=
by
  sorry

end chef_additional_wings_l1181_118109


namespace final_price_of_coat_after_discounts_l1181_118161

def original_price : ℝ := 120
def first_discount : ℝ := 0.25
def second_discount : ℝ := 0.20

theorem final_price_of_coat_after_discounts : 
    (1 - second_discount) * (1 - first_discount) * original_price = 72 := 
by
    sorry

end final_price_of_coat_after_discounts_l1181_118161


namespace problem_1_problem_2_l1181_118111

noncomputable def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  let numerator := |C1 - C2|
  let denominator := Real.sqrt (A^2 + B^2)
  numerator / denominator

noncomputable def distance_point_to_line (A B C x0 y0 : ℝ) : ℝ :=
  let numerator := |A * x0 + B * y0 + C|
  let denominator := Real.sqrt (A^2 + B^2)
  numerator / denominator

theorem problem_1 : distance_between_parallel_lines 2 1 (-1) 1 = 2 * Real.sqrt 5 / 5 :=
  by sorry

theorem problem_2 : distance_point_to_line 2 1 (-1) 0 2 = Real.sqrt 5 / 5 :=
  by sorry

end problem_1_problem_2_l1181_118111


namespace james_calories_per_minute_l1181_118121

variable (classes_per_week : ℕ) (hours_per_class : ℝ) (total_calories_per_week : ℕ)

theorem james_calories_per_minute
  (h1 : classes_per_week = 3)
  (h2 : hours_per_class = 1.5)
  (h3 : total_calories_per_week = 1890) :
  total_calories_per_week / (classes_per_week * (hours_per_class * 60)) = 7 := 
by
  sorry

end james_calories_per_minute_l1181_118121


namespace inequality_proof_equality_conditions_l1181_118198

theorem inequality_proof
  (x y : ℝ)
  (h1 : x ≥ y)
  (h2 : y ≥ 1) :
  (x / Real.sqrt (x + y) + y / Real.sqrt (y + 1) + 1 / Real.sqrt (x + 1) ≥
   y / Real.sqrt (x + y) + x / Real.sqrt (x + 1) + 1 / Real.sqrt (y + 1)) :=
by
  sorry

theorem equality_conditions
  (x y : ℝ) :
  (x = y ∨ x = 1 ∨ y = 1) ↔
  (x / Real.sqrt (x + y) + y / Real.sqrt (y + 1) + 1 / Real.sqrt (x + 1) =
   y / Real.sqrt (x + y) + x / Real.sqrt (x + 1) + 1 / Real.sqrt (y + 1)) :=
by
  sorry

end inequality_proof_equality_conditions_l1181_118198


namespace jeremys_school_distance_l1181_118145

def distance_to_school (rush_hour_time : ℚ) (no_traffic_time : ℚ) (speed_increase : ℚ) (distance : ℚ) : Prop :=
  ∃ v : ℚ, distance = v * rush_hour_time ∧ distance = (v + speed_increase) * no_traffic_time

theorem jeremys_school_distance :
  distance_to_school (3/10 : ℚ) (1/5 : ℚ) 20 12 :=
sorry

end jeremys_school_distance_l1181_118145


namespace equiv_or_neg_equiv_l1181_118196

theorem equiv_or_neg_equiv (x y : ℤ) (h : (x^2) % 239 = (y^2) % 239) :
  (x % 239 = y % 239) ∨ (x % 239 = (-y) % 239) :=
by
  sorry

end equiv_or_neg_equiv_l1181_118196


namespace olaf_total_cars_l1181_118104

noncomputable def olaf_initial_cars : ℕ := 150
noncomputable def uncle_cars : ℕ := 5
noncomputable def grandpa_cars : ℕ := 2 * uncle_cars
noncomputable def dad_cars : ℕ := 10
noncomputable def mum_cars : ℕ := dad_cars + 5
noncomputable def auntie_cars : ℕ := 6
noncomputable def liam_cars : ℕ := dad_cars / 2
noncomputable def emma_cars : ℕ := uncle_cars / 3
noncomputable def grandma_cars : ℕ := 3 * auntie_cars

noncomputable def total_gifts : ℕ := 
  grandpa_cars + dad_cars + mum_cars + auntie_cars + uncle_cars + liam_cars + emma_cars + grandma_cars

noncomputable def total_cars_after_gifts : ℕ := olaf_initial_cars + total_gifts

theorem olaf_total_cars : total_cars_after_gifts = 220 := by
  sorry

end olaf_total_cars_l1181_118104


namespace minimum_value_of_f_l1181_118108

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x + 5) + abs (x + 6)

theorem minimum_value_of_f : ∃ x : ℝ, f x = 1 :=
by sorry

end minimum_value_of_f_l1181_118108


namespace range_of_m_l1181_118147

noncomputable def f (m x : ℝ) : ℝ :=
  Real.log x + m / x

theorem range_of_m (m : ℝ) :
  (∀ (a b : ℝ), a > 0 → b > 0 → a ≠ b → (f m b - f m a) / (b - a) < 1) →
  m ≥ 1 / 4 :=
by
  sorry

end range_of_m_l1181_118147


namespace number_of_real_solutions_l1181_118110

-- Definition of the equation
def equation (x : ℝ) : Prop := x / 50 = Real.cos x

-- The main theorem stating the number of solutions
theorem number_of_real_solutions : 
  ∃ (n : ℕ), n = 32 ∧ ∀ x : ℝ, equation x → -50 ≤ x ∧ x ≤ 50 :=
sorry

end number_of_real_solutions_l1181_118110


namespace find_n_l1181_118101

theorem find_n (x y : ℝ) (h1 : (7 * x + 2 * y) / (x - n * y) = 23) (h2 : x / (2 * y) = 3 / 2) :
  ∃ n : ℝ, n = 2 := by
  sorry

end find_n_l1181_118101


namespace negation_of_proposition_l1181_118114

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x^2 - x + 2 ≥ 0)) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by
  sorry

end negation_of_proposition_l1181_118114


namespace smallest_value_l1181_118160

theorem smallest_value : 54 * Real.sqrt 3 < 144 ∧ 54 * Real.sqrt 3 < 108 * Real.sqrt 6 - 108 * Real.sqrt 2 := by
  sorry

end smallest_value_l1181_118160


namespace part1_part2_l1181_118143

def f (x : ℝ) : ℝ := abs (x - 5) + abs (x + 4)

theorem part1 (x : ℝ) : f x ≥ 12 ↔ x ≥ 13 / 2 ∨ x ≤ -11 / 2 :=
by
    sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x - 2 ^ (1 - 3 * a) - 1 ≥ 0) ↔ -2 / 3 ≤ a :=
by
    sorry

end part1_part2_l1181_118143


namespace base_sum_correct_l1181_118195

theorem base_sum_correct :
  let C := 12
  let a := 3 * 9^2 + 5 * 9^1 + 7 * 9^0
  let b := 4 * 13^2 + C * 13^1 + 2 * 13^0
  a + b = 1129 :=
by
  sorry

end base_sum_correct_l1181_118195


namespace calculate_total_cost_l1181_118132

def initial_price_orange : ℝ := 40
def initial_price_mango : ℝ := 50
def price_increase_percentage : ℝ := 0.15

-- Hypotheses
def new_price (initial_price : ℝ) (percentage_increase : ℝ) : ℝ :=
  initial_price * (1 + percentage_increase)

noncomputable def total_cost (num_oranges num_mangoes : ℕ) : ℝ :=
  (num_oranges * new_price initial_price_orange price_increase_percentage) +
  (num_mangoes * new_price initial_price_mango price_increase_percentage)

theorem calculate_total_cost :
  total_cost 10 10 = 1035 := by
  sorry

end calculate_total_cost_l1181_118132


namespace cone_lateral_area_l1181_118172

theorem cone_lateral_area (C l r A : ℝ) (hC : C = 4 * Real.pi) (hl : l = 3) 
  (hr : 2 * Real.pi * r = 4 * Real.pi) (hA : A = Real.pi * r * l) : A = 6 * Real.pi :=
by
  sorry

end cone_lateral_area_l1181_118172


namespace crayons_left_l1181_118190

theorem crayons_left (initial_crayons erasers_left more_crayons_than_erasers : ℕ)
    (H1 : initial_crayons = 531)
    (H2 : erasers_left = 38)
    (H3 : more_crayons_than_erasers = 353) :
    (initial_crayons - (initial_crayons - (erasers_left + more_crayons_than_erasers)) = 391) :=
by 
  sorry

end crayons_left_l1181_118190


namespace remainder_of_7_pow_205_mod_12_l1181_118150

theorem remainder_of_7_pow_205_mod_12 : (7^205) % 12 = 7 :=
by
  sorry

end remainder_of_7_pow_205_mod_12_l1181_118150


namespace members_not_playing_any_sport_l1181_118152

theorem members_not_playing_any_sport {total_members badminton_players tennis_players both_players : ℕ}
  (h_total : total_members = 28)
  (h_badminton : badminton_players = 17)
  (h_tennis : tennis_players = 19)
  (h_both : both_players = 10) :
  total_members - (badminton_players + tennis_players - both_players) = 2 :=
by
  sorry

end members_not_playing_any_sport_l1181_118152


namespace zinc_to_copper_ratio_l1181_118193

theorem zinc_to_copper_ratio (total_weight zinc_weight copper_weight : ℝ) 
  (h1 : total_weight = 64) 
  (h2 : zinc_weight = 28.8) 
  (h3 : copper_weight = total_weight - zinc_weight) : 
  (zinc_weight / 0.4) / (copper_weight / 0.4) = 9 / 11 :=
by
  sorry

end zinc_to_copper_ratio_l1181_118193


namespace intersection_A_B_complement_l1181_118199

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x ≤ 1}
def B_complement : Set ℝ := U \ B

theorem intersection_A_B_complement : A ∩ B_complement = {x | x > 1} := 
by 
  sorry

end intersection_A_B_complement_l1181_118199


namespace percent_problem_l1181_118171

theorem percent_problem (x : ℝ) (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 :=
by
  sorry

end percent_problem_l1181_118171


namespace car_speed_l1181_118194

theorem car_speed (distance time : ℝ) (h₁ : distance = 50) (h₂ : time = 5) : (distance / time) = 10 :=
by
  rw [h₁, h₂]
  norm_num

end car_speed_l1181_118194


namespace decreasing_function_l1181_118189

-- Define the functions
noncomputable def fA (x : ℝ) : ℝ := 3^x
noncomputable def fB (x : ℝ) : ℝ := Real.logb 0.5 x
noncomputable def fC (x : ℝ) : ℝ := Real.sqrt x
noncomputable def fD (x : ℝ) : ℝ := 1/x

-- Define the domains
def domainA : Set ℝ := Set.univ
def domainB : Set ℝ := {x | x > 0}
def domainC : Set ℝ := {x | x ≥ 0}
def domainD : Set ℝ := {x | x < 0} ∪ {x | x > 0}

-- Prove that fB is the only decreasing function in its domain
theorem decreasing_function:
  (∀ x y, x ∈ domainA → y ∈ domainA → x < y → fA x > fA y) = false ∧
  (∀ x y, x ∈ domainB → y ∈ domainB → x < y → fB x > fB y) ∧
  (∀ x y, x ∈ domainC → y ∈ domainC → x < y → fC x > fC y) = false ∧
  (∀ x y, x ∈ domainD → y ∈ domainD → x < y → fD x > fD y) = false :=
  sorry

end decreasing_function_l1181_118189


namespace larry_channels_l1181_118197

-- Initial conditions
def init_channels : ℕ := 150
def channels_taken_away : ℕ := 20
def channels_replaced : ℕ := 12
def channels_reduce_request : ℕ := 10
def sports_package : ℕ := 8
def supreme_sports_package : ℕ := 7

-- Calculation representing the overall change step-by-step
theorem larry_channels : 
  init_channels - channels_taken_away + channels_replaced - channels_reduce_request + sports_package + supreme_sports_package = 147 :=
by sorry

end larry_channels_l1181_118197


namespace prove_expression_value_l1181_118184

theorem prove_expression_value (m n : ℝ) (h : m^2 + 3 * n - 1 = 2) : 2 * m^2 + 6 * n + 1 = 7 := by
  sorry

end prove_expression_value_l1181_118184


namespace amount_per_person_l1181_118112

theorem amount_per_person (total_amount : ℕ) (num_persons : ℕ) (amount_each : ℕ)
  (h1 : total_amount = 42900) (h2 : num_persons = 22) (h3 : amount_each = 1950) :
  total_amount / num_persons = amount_each :=
by
  -- Proof to be filled
  sorry

end amount_per_person_l1181_118112


namespace divisor_iff_even_l1181_118137

noncomputable def hasDivisor (k : ℕ) : Prop := 
  ∃ n : ℕ, n > 0 ∧ (8 * k * n - 1) ∣ (4 * k ^ 2 - 1) ^ 2

theorem divisor_iff_even (k : ℕ) (h : k > 0) : hasDivisor k ↔ (k % 2 = 0) :=
by
  sorry

end divisor_iff_even_l1181_118137


namespace relationship_between_variables_l1181_118119

theorem relationship_between_variables
  (a b x y : ℚ)
  (h1 : x + y = a + b)
  (h2 : y - x < a - b)
  (h3 : b > a) :
  y < a ∧ a < b ∧ b < x :=
sorry

end relationship_between_variables_l1181_118119


namespace fraction_of_men_married_is_two_thirds_l1181_118170

-- Define the total number of faculty members
def total_faculty_members : ℕ := 100

-- Define the number of women as 70% of the faculty members
def women : ℕ := (70 * total_faculty_members) / 100

-- Define the number of men as 30% of the faculty members
def men : ℕ := (30 * total_faculty_members) / 100

-- Define the number of married faculty members as 40% of the faculty members
def married_faculty : ℕ := (40 * total_faculty_members) / 100

-- Define the number of single men as 1/3 of the men
def single_men : ℕ := men / 3

-- Define the number of married men as 2/3 of the men
def married_men : ℕ := (2 * men) / 3

-- Define the fraction of men who are married
def fraction_married_men : ℚ := married_men / men

-- The proof statement
theorem fraction_of_men_married_is_two_thirds : fraction_married_men = 2 / 3 := 
by sorry

end fraction_of_men_married_is_two_thirds_l1181_118170


namespace relation_correct_l1181_118115

def M := {x : ℝ | x < 2}
def N := {x : ℝ | 0 < x ∧ x < 1}
def CR (S : Set ℝ) := {x : ℝ | x ∈ (Set.univ : Set ℝ) \ S}

theorem relation_correct : M ∪ CR N = (Set.univ : Set ℝ) :=
by sorry

end relation_correct_l1181_118115


namespace desiree_age_l1181_118116

variables (D C : ℕ)
axiom condition1 : D = 2 * C
axiom condition2 : D + 30 = (2 * (C + 30)) / 3 + 14

theorem desiree_age : D = 6 :=
by
  sorry

end desiree_age_l1181_118116


namespace smaller_successive_number_l1181_118158

theorem smaller_successive_number (n : ℕ) (h : n * (n + 1) = 9506) : n = 97 :=
sorry

end smaller_successive_number_l1181_118158


namespace sequence_a8_l1181_118168

theorem sequence_a8 (a : ℕ → ℕ) 
  (h1 : ∀ n ≥ 1, a (n + 2) = a n + a (n + 1)) 
  (h2 : a 7 = 120) : 
  a 8 = 194 :=
sorry

end sequence_a8_l1181_118168


namespace volume_of_inscribed_sphere_l1181_118139

theorem volume_of_inscribed_sphere {cube_edge : ℝ} (h : cube_edge = 6) : 
  ∃ V : ℝ, V = 36 * Real.pi :=
by
  sorry

end volume_of_inscribed_sphere_l1181_118139


namespace proj_w_v_is_v_l1181_118120

noncomputable def proj_w_v (v w : ℝ × ℝ) : ℝ × ℝ :=
  let c := (v.1 * w.1 + v.2 * w.2) / (w.1 * w.1 + w.2 * w.2)
  (c * w.1, c * w.2)

def v : ℝ × ℝ := (-3, 2)
def w : ℝ × ℝ := (4, -2)

theorem proj_w_v_is_v : proj_w_v v w = v := 
  sorry

end proj_w_v_is_v_l1181_118120


namespace arithmetic_sequence_min_sum_l1181_118128

theorem arithmetic_sequence_min_sum (x : ℝ) (d : ℝ) (h₁ : d > 0) :
  (∃ n : ℕ, n > 0 ∧ (n^2 - 4 * n < 0) ∧ (n = 6 ∨ n = 7)) :=
by
  sorry

end arithmetic_sequence_min_sum_l1181_118128


namespace earnings_total_l1181_118162

-- Define the earnings for each day based on given conditions
def Monday_earnings : ℝ := 0.20 * 10 * 3
def Tuesday_earnings : ℝ := 0.25 * 12 * 4
def Wednesday_earnings : ℝ := 0.10 * 15 * 5
def Thursday_earnings : ℝ := 0.15 * 8 * 6
def Friday_earnings : ℝ := 0.30 * 20 * 2

-- Compute total earnings over the five days
def total_earnings : ℝ :=
  Monday_earnings + Tuesday_earnings + Wednesday_earnings + Thursday_earnings + Friday_earnings

-- Lean statement to prove the total earnings
theorem earnings_total :
  total_earnings = 44.70 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end earnings_total_l1181_118162


namespace find_m_n_sum_l1181_118148

theorem find_m_n_sum (m n : ℕ) (hm : m > 1) (hn : n > 1) 
  (h : 2005^2 + m^2 = 2004^2 + n^2) : 
  m + n = 211 :=
sorry

end find_m_n_sum_l1181_118148


namespace abs_eq_4_l1181_118129

theorem abs_eq_4 (a : ℝ) : |a| = 4 ↔ a = 4 ∨ a = -4 :=
by
  sorry

end abs_eq_4_l1181_118129


namespace turnips_total_l1181_118135

def melanie_turnips := 139
def benny_turnips := 113

def total_turnips (melanie_turnips benny_turnips : Nat) : Nat :=
  melanie_turnips + benny_turnips

theorem turnips_total :
  total_turnips melanie_turnips benny_turnips = 252 :=
by
  sorry

end turnips_total_l1181_118135


namespace trigonometric_identity_l1181_118100

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : (Real.sin (2 * α) / Real.cos α ^ 2) = 6 :=
sorry

end trigonometric_identity_l1181_118100


namespace sufficient_but_not_necessary_condition_l1181_118164

-- Define a sequence of positive terms
def is_positive_sequence (seq : Fin 8 → ℝ) : Prop :=
  ∀ i, 0 < seq i

-- Define what it means for a sequence to be geometric
def is_geometric_sequence (seq : Fin 8 → ℝ) : Prop :=
  ∃ q > 0, q ≠ 1 ∧ ∀ i j, i < j → seq j = (q ^ (j - i : ℤ)) * seq i

-- State the theorem
theorem sufficient_but_not_necessary_condition (seq : Fin 8 → ℝ) (h_pos : is_positive_sequence seq) :
  ¬is_geometric_sequence seq → seq 0 + seq 7 < seq 3 + seq 4 ∧ 
  (seq 0 + seq 7 < seq 3 + seq 4 → ¬is_geometric_sequence seq) ∧
  (¬is_geometric_sequence seq → ¬(seq 0 + seq 7 < seq 3 + seq 4) -> ¬ is_geometric_sequence seq) :=
sorry

end sufficient_but_not_necessary_condition_l1181_118164


namespace arithmetic_mean_end_number_l1181_118118

theorem arithmetic_mean_end_number (n : ℤ) :
  (100 + n) / 2 = 150 + 100 → n = 400 := by
  sorry

end arithmetic_mean_end_number_l1181_118118


namespace length_of_lunch_break_is_48_minutes_l1181_118149

noncomputable def paula_and_assistants_lunch_break : ℝ := sorry

theorem length_of_lunch_break_is_48_minutes
  (p h L : ℝ)
  (h_monday : (9 - L) * (p + h) = 0.6)
  (h_tuesday : (7 - L) * h = 0.3)
  (h_wednesday : (10 - L) * p = 0.1) :
  L = 0.8 :=
sorry

end length_of_lunch_break_is_48_minutes_l1181_118149


namespace complex_power_six_l1181_118142

theorem complex_power_six (i : ℂ) (hi : i * i = -1) : (1 + i)^6 = -8 * i :=
by
  sorry

end complex_power_six_l1181_118142


namespace total_students_l1181_118105

theorem total_students (a b c d e f : ℕ)  (h : a + b = 15) (h1 : a = 5) (h2 : b = 10) 
(h3 : c = 15) (h4 : d = 10) (h5 : e = 5) (h6 : f = 0) (h_total : a + b + c + d + e + f = 50) : a + b + c + d + e + f = 50 :=
by {exact h_total}

end total_students_l1181_118105


namespace gcd_of_6Tn2_and_nplus1_eq_2_l1181_118156

theorem gcd_of_6Tn2_and_nplus1_eq_2 (n : ℕ) (h_pos : 0 < n) :
  Nat.gcd (6 * ((n * (n + 1) / 2)^2)) (n + 1) = 2 :=
sorry

end gcd_of_6Tn2_and_nplus1_eq_2_l1181_118156


namespace rhombus_area_l1181_118126

theorem rhombus_area (a b : ℝ) (h : (a - 1) ^ 2 + Real.sqrt (b - 4) = 0) : (1 / 2) * a * b = 2 := by
  sorry

end rhombus_area_l1181_118126


namespace baking_trays_used_l1181_118155

-- Let T be the number of baking trays Anna used.
variable (T : ℕ)

-- Condition: Each tray has 20 cupcakes.
def cupcakes_per_tray : ℕ := 20

-- Condition: Each cupcake was sold for $2.
def cupcake_price : ℕ := 2

-- Condition: Only 3/5 of the cupcakes were sold.
def fraction_sold : ℚ := 3 / 5

-- Condition: Anna earned $96 from sold cupcakes.
def earnings : ℕ := 96

-- Derived expressions:
def total_cupcakes (T : ℕ) : ℕ := cupcakes_per_tray * T

def sold_cupcakes (T : ℕ) : ℚ := fraction_sold * total_cupcakes T

def total_earnings (T : ℕ) : ℚ := cupcake_price * sold_cupcakes T

-- The statement to be proved: Given the conditions, the number of trays T must be 4.
theorem baking_trays_used (h : total_earnings T = earnings) : T = 4 := by
  sorry

end baking_trays_used_l1181_118155


namespace distinct_values_count_l1181_118192

noncomputable def f : ℕ → ℤ := sorry -- The actual function definition is not required

theorem distinct_values_count :
  ∃! n, n = 3 ∧ 
  (∀ x : ℕ, 
    (f x = f (x - 1) + f (x + 1) ∧ 
     (x = 1 → f x = 2009) ∧ 
     (x = 3 → f x = 0))) := 
sorry

end distinct_values_count_l1181_118192


namespace find_f_neg2_l1181_118167

noncomputable def f : ℝ → ℝ := sorry

axiom cond1 : ∀ a b : ℝ, f (a + b) = f a * f b
axiom cond2 : ∀ x : ℝ, f x > 0
axiom cond3 : f 1 = 1 / 3

theorem find_f_neg2 : f (-2) = 9 := sorry

end find_f_neg2_l1181_118167


namespace greatest_number_of_consecutive_integers_sum_to_91_l1181_118176

theorem greatest_number_of_consecutive_integers_sum_to_91 :
  ∃ N, (∀ (a : ℤ), (N : ℕ) > 0 → (N * (2 * a + N - 1) = 182)) ∧ (N = 182) :=
by {
  sorry
}

end greatest_number_of_consecutive_integers_sum_to_91_l1181_118176


namespace carrots_weight_l1181_118134

theorem carrots_weight (carrots_bed1: ℕ) (carrots_bed2: ℕ) (carrots_bed3: ℕ) (carrots_per_pound: ℕ)
  (h_bed1: carrots_bed1 = 55)
  (h_bed2: carrots_bed2 = 101)
  (h_bed3: carrots_bed3 = 78)
  (h_c_per_p: carrots_per_pound = 6) :
  (carrots_bed1 + carrots_bed2 + carrots_bed3) / carrots_per_pound = 39 := by
  sorry

end carrots_weight_l1181_118134


namespace sum_of_odd_integers_l1181_118175

theorem sum_of_odd_integers (n : ℕ) (h : n * (n + 1) = 4970) : (n * n = 4900) :=
by sorry

end sum_of_odd_integers_l1181_118175


namespace inverse_g_neg1_l1181_118181

noncomputable def g (c d x : ℝ) : ℝ := 1 / (c * x + d)

theorem inverse_g_neg1 (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ y : ℝ, g c d y = -1 ∧ y = (-1 - d) / c := 
by
  unfold g
  sorry

end inverse_g_neg1_l1181_118181


namespace compare_rental_fees_l1181_118187

namespace HanfuRental

def store_A_rent_price : ℝ := 120
def store_B_rent_price : ℝ := 160
def store_A_discount : ℝ := 0.20
def store_B_discount_limit : ℕ := 6
def store_B_excess_rate : ℝ := 0.50
def x : ℕ := 40 -- number of Hanfu costumes

def y₁ (x : ℕ) : ℝ := (store_A_rent_price * (1 - store_A_discount)) * x

def y₂ (x : ℕ) : ℝ :=
  if x ≤ store_B_discount_limit then store_B_rent_price * x
  else store_B_rent_price * store_B_discount_limit + store_B_excess_rate * store_B_rent_price * (x - store_B_discount_limit)

theorem compare_rental_fees (x : ℕ) (hx : x = 40) :
  y₂ x ≤ y₁ x :=
sorry

end HanfuRental

end compare_rental_fees_l1181_118187


namespace num_students_yes_R_l1181_118153

noncomputable def num_students_total : ℕ := 800
noncomputable def num_students_yes_only_M : ℕ := 150
noncomputable def num_students_no_to_both : ℕ := 250

theorem num_students_yes_R : (num_students_total - num_students_no_to_both) - num_students_yes_only_M = 400 :=
by
  sorry

end num_students_yes_R_l1181_118153


namespace perimeter_of_triangle_l1181_118191

def point (x y : ℝ) := (x, y)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

noncomputable def perimeter_triangle (a b c : ℝ × ℝ) : ℝ :=
  distance a b + distance b c + distance c a

theorem perimeter_of_triangle :
  let A := point 1 2
  let B := point 6 8
  let C := point 1 5
  perimeter_triangle A B C = Real.sqrt 61 + Real.sqrt 34 + 3 :=
by
  -- proof steps can be provided here
  sorry

end perimeter_of_triangle_l1181_118191


namespace biography_percentage_increase_l1181_118113

variable {T : ℝ}
variable (hT : T > 0 ∧ T ≤ 10000)
variable (B : ℝ := 0.20 * T)
variable (B' : ℝ := 0.32 * T)
variable (percentage_increase : ℝ := ((B' - B) / B) * 100)

theorem biography_percentage_increase :
  percentage_increase = 60 :=
by
  sorry

end biography_percentage_increase_l1181_118113


namespace tangency_condition_for_parabola_and_line_l1181_118103

theorem tangency_condition_for_parabola_and_line (k : ℚ) :
  (∀ x y : ℚ, (6 * x - 4 * y + k = 0) ↔ (y^2 = 16 * x)) ↔ (k = 32 / 3) :=
  sorry

end tangency_condition_for_parabola_and_line_l1181_118103


namespace solve_equation_l1181_118154

theorem solve_equation (x : ℝ) : (x + 3) * (x - 1) = 12 ↔ (x = -5 ∨ x = 3) := sorry

end solve_equation_l1181_118154


namespace total_branches_in_pine_tree_l1181_118102

-- Definitions based on the conditions
def middle_branch : ℕ := 0 -- arbitrary assignment to represent the middle branch

def jumps_up_5 (b : ℕ) : ℕ := b + 5
def jumps_down_7 (b : ℕ) : ℕ := b - 7
def jumps_up_4 (b : ℕ) : ℕ := b + 4
def jumps_up_9 (b : ℕ) : ℕ := b + 9

-- The statement to be proven
theorem total_branches_in_pine_tree : 
  (jumps_up_9 (jumps_up_4 (jumps_down_7 (jumps_up_5 middle_branch))) = 11) →
  ∃ n, n = 23 :=
by
  sorry

end total_branches_in_pine_tree_l1181_118102


namespace H_iterated_l1181_118186

variable (H : ℝ → ℝ)

-- Conditions as hypotheses
axiom H_2 : H 2 = -4
axiom H_neg4 : H (-4) = 6
axiom H_6 : H 6 = 6

-- The theorem we want to prove
theorem H_iterated (H : ℝ → ℝ) (h1 : H 2 = -4) (h2 : H (-4) = 6) (h3 : H 6 = 6) : 
  H (H (H (H (H 2)))) = 6 := by
  sorry

end H_iterated_l1181_118186


namespace gloves_needed_l1181_118140

theorem gloves_needed (participants : ℕ) (gloves_per_participant : ℕ) (total_gloves : ℕ)
  (h1 : participants = 82)
  (h2 : gloves_per_participant = 2)
  (h3 : total_gloves = participants * gloves_per_participant) :
  total_gloves = 164 :=
by
  sorry

end gloves_needed_l1181_118140


namespace ming_wins_inequality_l1181_118165

variables (x : ℕ)

def remaining_distance (x : ℕ) : ℕ := 10000 - 200 * x
def ming_remaining_distance (x : ℕ) : ℕ := remaining_distance x - 200

-- Ensure that Xiao Ming's winning inequality holds:
theorem ming_wins_inequality (h1 : 0 < x) :
  (ming_remaining_distance x) / 250 > (remaining_distance x) / 300 :=
sorry

end ming_wins_inequality_l1181_118165


namespace abs_condition_sufficient_not_necessary_l1181_118122

theorem abs_condition_sufficient_not_necessary:
  (∀ x : ℝ, (-2 < x ∧ x < 3) → (-1 < x ∧ x < 3)) :=
by
  sorry

end abs_condition_sufficient_not_necessary_l1181_118122


namespace trajectory_equation_l1181_118157

-- Definitions and conditions
noncomputable def tangent_to_x_axis (M : ℝ × ℝ) := M.snd = 0
noncomputable def internally_tangent (M : ℝ × ℝ) := ∃ (r : ℝ), 0 < r ∧ M.1^2 + (M.2 - r)^2 = 4

-- The theorem stating the proof problem
theorem trajectory_equation (M : ℝ × ℝ) (h_tangent : tangent_to_x_axis M) (h_internal_tangent : internally_tangent M) :
  (∃ y : ℝ, 0 < y ∧ y ≤ 1 ∧ M.fst^2 = 4 * (y - 1)) :=
sorry

end trajectory_equation_l1181_118157


namespace reciprocal_neg_six_l1181_118169

-- Define the concept of reciprocal
def reciprocal (a : ℤ) (h : a ≠ 0) : ℚ := 1 / a

theorem reciprocal_neg_six : reciprocal (-6) (by norm_num) = -1 / 6 := 
by 
  sorry

end reciprocal_neg_six_l1181_118169


namespace tan_double_angle_l1181_118166

theorem tan_double_angle (α : ℝ) (h : Real.tan (Real.pi - α) = 2) : Real.tan (2 * α) = 4 / 3 := 
by 
  sorry

end tan_double_angle_l1181_118166


namespace find_subtracted_value_l1181_118151

theorem find_subtracted_value (x y : ℕ) (h1 : x = 120) (h2 : 2 * x - y = 102) : y = 138 :=
by
  sorry

end find_subtracted_value_l1181_118151


namespace difference_of_squares_l1181_118107

theorem difference_of_squares {a b : ℝ} (h1 : a + b = 75) (h2 : a - b = 15) : a^2 - b^2 = 1125 :=
by
  sorry

end difference_of_squares_l1181_118107


namespace second_month_interest_l1181_118185

def compounded_interest (initial_loan : ℝ) (rate_per_month : ℝ) : ℝ :=
  initial_loan * rate_per_month

theorem second_month_interest :
  let initial_loan := 200
  let rate_per_month := 0.10
  compounded_interest (initial_loan + compounded_interest initial_loan rate_per_month) rate_per_month = 22 :=
by
  sorry

end second_month_interest_l1181_118185


namespace successive_discounts_final_price_l1181_118127

noncomputable def initial_price : ℝ := 10000
noncomputable def discount1 : ℝ := 0.20
noncomputable def discount2 : ℝ := 0.10
noncomputable def discount3 : ℝ := 0.05

theorem successive_discounts_final_price :
  let price_after_first_discount := initial_price * (1 - discount1)
  let price_after_second_discount := price_after_first_discount * (1 - discount2)
  let final_selling_price := price_after_second_discount * (1 - discount3)
  final_selling_price = 6840 := by
  sorry

end successive_discounts_final_price_l1181_118127


namespace cos_alpha_value_l1181_118183

theorem cos_alpha_value (α : ℝ) (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) : Real.cos α = 1 / 5 :=
sorry

end cos_alpha_value_l1181_118183


namespace shenzhen_vaccination_count_l1181_118106

theorem shenzhen_vaccination_count :
  2410000 = 2.41 * 10^6 :=
  sorry

end shenzhen_vaccination_count_l1181_118106


namespace uranus_appears_7_minutes_after_6AM_l1181_118179

def mars_last_seen := 0 * 60 + 10 -- 12:10 AM in minutes after midnight
def jupiter_after_mars := 2 * 60 + 41 -- 2 hours and 41 minutes in minutes
def uranus_after_jupiter := 3 * 60 + 16 -- 3 hours and 16 minutes in minutes
def uranus_appearance := mars_last_seen + jupiter_after_mars + uranus_after_jupiter

theorem uranus_appears_7_minutes_after_6AM : uranus_appearance - (6 * 60) = 7 := by
  sorry

end uranus_appears_7_minutes_after_6AM_l1181_118179
