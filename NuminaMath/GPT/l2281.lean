import Mathlib

namespace more_whistles_sean_than_charles_l2281_228127

def whistles_sean : ℕ := 223
def whistles_charles : ℕ := 128

theorem more_whistles_sean_than_charles : (whistles_sean - whistles_charles) = 95 :=
by
  sorry

end more_whistles_sean_than_charles_l2281_228127


namespace magnitude_of_root_of_quadratic_eq_l2281_228138

open Complex

theorem magnitude_of_root_of_quadratic_eq (z : ℂ) 
  (h : z^2 - (2 : ℂ) * z + 2 = 0) : abs z = Real.sqrt 2 :=
by 
  sorry

end magnitude_of_root_of_quadratic_eq_l2281_228138


namespace geometric_sum_of_first_five_terms_l2281_228197

theorem geometric_sum_of_first_five_terms (a_1 l : ℝ)
  (h₁ : ∀ r : ℝ, (2 * l = a_1 * (r - 1) ^ 2)) 
  (h₂ : ∀ (r : ℝ), a_1 * r ^ 3 = 8 * a_1):
  (a_1 + a_1 * (2 : ℝ) + a_1 * (2 : ℝ)^2 + a_1 * (2 : ℝ)^3 + a_1 * (2 : ℝ)^4) = 62 :=
by
  sorry

end geometric_sum_of_first_five_terms_l2281_228197


namespace largest_b_for_denom_has_nonreal_roots_l2281_228125

theorem largest_b_for_denom_has_nonreal_roots :
  ∃ b : ℤ, 
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 12 ≠ 0) 
  ∧ (∀ b' : ℤ, (∀ x : ℝ, x^2 + (b' : ℝ) * x + 12 ≠ 0) → b' ≤ b)
  ∧ b = 6 :=
sorry

end largest_b_for_denom_has_nonreal_roots_l2281_228125


namespace books_loaned_l2281_228192

theorem books_loaned (L : ℕ)
  (initial_books : ℕ := 150)
  (end_year_books : ℕ := 100)
  (return_rate : ℝ := 0.60)
  (loan_rate : ℝ := 0.40)
  (returned_books : ℕ := (initial_books - end_year_books)) :
  loan_rate * (L : ℝ) = (returned_books : ℝ) → L = 125 := by
  intro h
  sorry

end books_loaned_l2281_228192


namespace problem1_problem2_l2281_228159

-- Define the function f(x)
def f (x m : ℝ) : ℝ := abs (x - m) - abs (x + 3 * m)

-- Condition that m must be greater than 0
variable {m : ℝ} (hm : m > 0)

-- First problem statement: When m=1, the solution set for f(x) ≥ 1 is x ≤ -3/2.
theorem problem1 (x : ℝ) (h : f x 1 ≥ 1) : x ≤ -3 / 2 :=
sorry

-- Second problem statement: The range of values for m such that f(x) < |2 + t| + |t - 1| holds for all x and t is 0 < m < 3/4.
theorem problem2 (m : ℝ) : (∀ (x t : ℝ), f x m < abs (2 + t) + abs (t - 1)) ↔ (0 < m ∧ m < 3 / 4) :=
sorry

end problem1_problem2_l2281_228159


namespace sufficient_but_not_necessary_condition_l2281_228149

theorem sufficient_but_not_necessary_condition (h1 : 1^2 - 1 = 0) (h2 : ∀ x, x^2 - 1 = 0 → (x = 1 ∨ x = -1)) :
  (∀ x, x = 1 → x^2 - 1 = 0) ∧ ¬ (∀ x, x^2 - 1 = 0 → x = 1) := by
  sorry

end sufficient_but_not_necessary_condition_l2281_228149


namespace peter_situps_eq_24_l2281_228137

noncomputable def situps_peter_did : ℕ :=
  let ratio_peter_greg := 3 / 4
  let situps_greg := 32
  let situps_peter := (3 * situps_greg) / 4
  situps_peter

theorem peter_situps_eq_24 : situps_peter_did = 24 := 
by 
  let h := situps_peter_did
  show h = 24
  sorry

end peter_situps_eq_24_l2281_228137


namespace sqrt3_pow_log_sqrt3_8_eq_8_l2281_228119

theorem sqrt3_pow_log_sqrt3_8_eq_8 : (Real.sqrt 3) ^ (Real.log 8 / Real.log (Real.sqrt 3)) = 8 :=
by
  sorry

end sqrt3_pow_log_sqrt3_8_eq_8_l2281_228119


namespace percent_sugar_in_resulting_solution_l2281_228169

theorem percent_sugar_in_resulting_solution (W : ℝ) (hW : W > 0) :
  let original_sugar_percent := 22 / 100
  let second_solution_sugar_percent := 74 / 100
  let remaining_original_weight := (3 / 4) * W
  let removed_weight := (1 / 4) * W
  let sugar_from_remaining_original := (original_sugar_percent * remaining_original_weight)
  let sugar_from_added_second_solution := (second_solution_sugar_percent * removed_weight)
  let total_sugar := sugar_from_remaining_original + sugar_from_added_second_solution
  let resulting_sugar_percent := total_sugar / W
  resulting_sugar_percent = 35 / 100 :=
by
  sorry

end percent_sugar_in_resulting_solution_l2281_228169


namespace problem_statement_l2281_228146

noncomputable def x : ℕ := 4
noncomputable def y : ℤ := 3  -- alternatively, we could define y as -3 and the equality would still hold

theorem problem_statement : x^2 + y^2 + x + 2023 = 2052 := by
  sorry  -- Proof goes here

end problem_statement_l2281_228146


namespace tan_theta_plus_pi_over_eight_sub_inv_l2281_228161

/-- Given the trigonometric identity, we can prove the tangent calculation -/
theorem tan_theta_plus_pi_over_eight_sub_inv (θ : ℝ)
  (h : 3 * Real.sin θ + Real.cos θ = Real.sqrt 10) :
  Real.tan (θ + Real.pi / 8) - 1 / Real.tan (θ + Real.pi / 8) = -14 := 
sorry

end tan_theta_plus_pi_over_eight_sub_inv_l2281_228161


namespace external_angle_bisector_lengths_l2281_228103

noncomputable def f_a (a b c : ℝ) : ℝ := 4 * Real.sqrt 3
noncomputable def f_b (b : ℝ) : ℝ := 6 / Real.sqrt 7
noncomputable def f_c (a b c : ℝ) : ℝ := 4 * Real.sqrt 3

theorem external_angle_bisector_lengths (a b c : ℝ) 
  (ha : a = 5 - Real.sqrt 7)
  (hb : b = 6)
  (hc : c = 5 + Real.sqrt 7) :
  f_a a b c = 4 * Real.sqrt 3 ∧
  f_b b = 6 / Real.sqrt 7 ∧
  f_c a b c = 4 * Real.sqrt 3 := by
  sorry

end external_angle_bisector_lengths_l2281_228103


namespace first_number_in_proportion_is_60_l2281_228168

theorem first_number_in_proportion_is_60 : 
  ∀ (x : ℝ), (x / 6 = 2 / 0.19999999999999998) → x = 60 :=
by
  intros x hx
  sorry

end first_number_in_proportion_is_60_l2281_228168


namespace last_two_digits_28_l2281_228167

theorem last_two_digits_28 (n : ℕ) (h1 : n > 0) (h2 : n % 2 = 1) : 
  (2^(2*n) * (2^(2*n+1) - 1)) % 100 = 28 :=
by
  sorry

end last_two_digits_28_l2281_228167


namespace evaluate_expression_l2281_228139

theorem evaluate_expression : 
  (Int.ceil ((Int.floor ((15 / 8 : Rat) ^ 2) : Rat) - (19 / 5 : Rat) : Rat) : Int) = 0 :=
sorry

end evaluate_expression_l2281_228139


namespace dividend_percentage_l2281_228123

theorem dividend_percentage (face_value : ℝ) (investment : ℝ) (roi : ℝ) (dividend_percentage : ℝ) 
    (h1 : face_value = 40) 
    (h2 : investment = 20) 
    (h3 : roi = 0.25) : dividend_percentage = 12.5 := 
  sorry

end dividend_percentage_l2281_228123


namespace probability_of_less_than_20_l2281_228108

variable (total_people : ℕ) (people_over_30 : ℕ)
variable (people_under_20 : ℕ) (probability_under_20 : ℝ)

noncomputable def group_size := total_people = 150
noncomputable def over_30 := people_over_30 = 90
noncomputable def under_20 := people_under_20 = total_people - people_over_30

theorem probability_of_less_than_20
  (total_people_eq : total_people = 150)
  (people_over_30_eq : people_over_30 = 90)
  (people_under_20_eq : people_under_20 = 60)
  (under_20_eq : 60 = total_people - people_over_30) :
  probability_under_20 = people_under_20 / total_people := by
  sorry

end probability_of_less_than_20_l2281_228108


namespace find_quotient_l2281_228134

-- Variables for larger number L and smaller number S
variables (L S: ℕ)

-- Conditions as definitions
def condition1 := L - S = 1325
def condition2 (quotient: ℕ) := L = S * quotient + 5
def condition3 := L = 1650

-- Statement to prove the quotient is 5
theorem find_quotient : ∃ (quotient: ℕ), condition1 L S ∧ condition2 L S quotient ∧ condition3 L → quotient = 5 := by
  sorry

end find_quotient_l2281_228134


namespace ben_paints_area_l2281_228148

variable (allen_ratio : ℕ) (ben_ratio : ℕ) (total_area : ℕ)
variable (total_ratio : ℕ := allen_ratio + ben_ratio)
variable (part_size : ℕ := total_area / total_ratio)

theorem ben_paints_area 
  (h1 : allen_ratio = 2)
  (h2 : ben_ratio = 6)
  (h3 : total_area = 360) : 
  ben_ratio * part_size = 270 := sorry

end ben_paints_area_l2281_228148


namespace expression_is_minus_two_l2281_228141

noncomputable def A : ℝ := (Real.sqrt 6 + Real.sqrt 2) * (Real.sqrt 3 - 2) * Real.sqrt (Real.sqrt 3 + 2)

theorem expression_is_minus_two : A = -2 := by
  sorry

end expression_is_minus_two_l2281_228141


namespace find_x_l2281_228154

-- Define the initial point A with coordinates A(x, -2)
def A (x : ℝ) : ℝ × ℝ := (x, -2)

-- Define the transformation of moving 5 units up and 3 units to the right to obtain point B
def transform (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 3, p.2 + 5)

-- Define the final point B with coordinates B(1, y)
def B (y : ℝ) : ℝ × ℝ := (1, y)

-- Define the proof problem
theorem find_x (x y : ℝ) (h : transform (A x) = B y) : x = -2 :=
by sorry

end find_x_l2281_228154


namespace compute_product_l2281_228156

variables (x1 y1 x2 y2 x3 y3 : ℝ)

def condition1 (x y : ℝ) : Prop :=
  x^3 - 3 * x * y^2 = 2017

def condition2 (x y : ℝ) : Prop :=
  y^3 - 3 * x^2 * y = 2016

theorem compute_product :
  condition1 x1 y1 → condition2 x1 y1 →
  condition1 x2 y2 → condition2 x2 y2 →
  condition1 x3 y3 → condition2 x3 y3 →
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 1008 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end compute_product_l2281_228156


namespace determine_y_l2281_228135

variable (x y : ℝ)

theorem determine_y (h1 : 0.25 * x = 0.15 * y - 15) (h2 : x = 840) : y = 1500 := 
by 
  sorry

end determine_y_l2281_228135


namespace system_solutions_range_b_l2281_228118

theorem system_solutions_range_b (b : ℝ) :
  (∀ x y : ℝ, x^2 - y^2 = 0 → x^2 + (y - b)^2 = 2 → x = 0 ∧ y = 0 ∨ y = b) →
  b ≥ 2 ∨ b ≤ -2 :=
sorry

end system_solutions_range_b_l2281_228118


namespace club_population_after_five_years_l2281_228170

noncomputable def a : ℕ → ℕ
| 0     => 18
| (n+1) => 3 * (a n - 5) + 5

theorem club_population_after_five_years : a 5 = 3164 := by
  sorry

end club_population_after_five_years_l2281_228170


namespace julias_total_spending_l2281_228142

def adoption_fee : ℝ := 20.00
def dog_food_cost : ℝ := 20.00
def treat_cost_per_bag : ℝ := 2.50
def num_treat_bags : ℝ := 2
def toy_box_cost : ℝ := 15.00
def crate_cost : ℝ := 20.00
def bed_cost : ℝ := 20.00
def collar_leash_cost : ℝ := 15.00
def discount_rate : ℝ := 0.20

def total_items_cost : ℝ :=
  dog_food_cost + (treat_cost_per_bag * num_treat_bags) + toy_box_cost +
  crate_cost + bed_cost + collar_leash_cost

def discount_amount : ℝ := total_items_cost * discount_rate
def discounted_items_cost : ℝ := total_items_cost - discount_amount
def total_expenditure : ℝ := adoption_fee + discounted_items_cost

theorem julias_total_spending :
  total_expenditure = 96.00 := by
  sorry

end julias_total_spending_l2281_228142


namespace total_capacity_of_bowl_l2281_228102

theorem total_capacity_of_bowl (L C : ℕ) (h1 : L / C = 3 / 5) (h2 : C = L + 18) : L + C = 72 := 
by
  sorry

end total_capacity_of_bowl_l2281_228102


namespace gain_percent_40_l2281_228144

theorem gain_percent_40 (cost_price selling_price : ℝ) (h1 : cost_price = 900) (h2 : selling_price = 1260) :
  ((selling_price - cost_price) / cost_price) * 100 = 40 :=
by
  sorry

end gain_percent_40_l2281_228144


namespace maxim_is_correct_l2281_228104

-- Define the mortgage rate as 12.5%
def mortgage_rate : ℝ := 0.125

-- Define the dividend yield rate as 17%
def dividend_rate : ℝ := 0.17

-- Define the net return as the difference between the dividend rate and the mortgage rate
def net_return (D M : ℝ) : ℝ := D - M

-- The main theorem to prove Maxim Sergeyevich is correct
theorem maxim_is_correct : net_return dividend_rate mortgage_rate > 0 :=
by
  sorry

end maxim_is_correct_l2281_228104


namespace cone_volume_l2281_228147

theorem cone_volume (d h : ℝ) (V : ℝ) (hd : d = 12) (hh : h = 8) :
  V = (1 / 3) * Real.pi * (d / 2) ^ 2 * h → V = 96 * Real.pi :=
by
  rw [hd, hh]
  sorry

end cone_volume_l2281_228147


namespace shaded_region_area_l2281_228173

noncomputable def area_of_shaded_region (r : ℝ) (oa : ℝ) (ab_length : ℝ) : ℝ :=
  18 * (Real.sqrt 2) - 9 - (9 * Real.pi / 4)

theorem shaded_region_area (r : ℝ) (oa : ℝ) (ab_length : ℝ) : 
  r = 3 ∧ oa = 3 * Real.sqrt 2 ∧ ab_length = 6 * Real.sqrt 2 → 
  area_of_shaded_region r oa ab_length = 18 * (Real.sqrt 2) - 9 - (9 * Real.pi / 4) :=
by
  intro h
  obtain ⟨hr, hoa, hab⟩ := h
  rw [hr, hoa, hab]
  exact rfl

end shaded_region_area_l2281_228173


namespace tour_groups_and_savings_minimum_people_for_savings_l2281_228191

theorem tour_groups_and_savings (x y : ℕ) (m : ℕ):
  (x + y = 102) ∧ (45 * x + 50 * y - 40 * 102 = 730) → 
  (x = 58 ∧ y = 44) :=
by
  sorry

theorem minimum_people_for_savings (m : ℕ):
  (∀ m, m < 50 → 50 * m > 45 * 51) → 
  (m ≥ 46) :=
by
  sorry

end tour_groups_and_savings_minimum_people_for_savings_l2281_228191


namespace black_balls_number_l2281_228114

-- Define the given conditions and the problem statement as Lean statements
theorem black_balls_number (n : ℕ) (h : (2 : ℝ) / (n + 2 : ℝ) = 0.4) : n = 3 :=
by
  sorry

end black_balls_number_l2281_228114


namespace trigonometric_identity_l2281_228177

theorem trigonometric_identity :
  let cos60 := (1 / 2)
  let sin30 := (1 / 2)
  let tan45 := (1 : ℝ)
  4 * cos60 + 8 * sin30 - 5 * tan45 = 1 :=
by
  let cos60 := (1 / 2 : ℝ)
  let sin30 := (1 / 2 : ℝ)
  let tan45 := (1 : ℝ)
  show 4 * cos60 + 8 * sin30 - 5 * tan45 = 1
  sorry

end trigonometric_identity_l2281_228177


namespace values_of_a_for_equation_l2281_228116

theorem values_of_a_for_equation :
  ∃ S : Finset ℤ, (∀ a ∈ S, |3 * a + 7| + |3 * a - 5| = 12) ∧ S.card = 4 :=
by
  sorry

end values_of_a_for_equation_l2281_228116


namespace average_TV_sets_in_shops_l2281_228193

def shop_a := 20
def shop_b := 30
def shop_c := 60
def shop_d := 80
def shop_e := 50
def total_shops := 5

theorem average_TV_sets_in_shops : (shop_a + shop_b + shop_c + shop_d + shop_e) / total_shops = 48 :=
by
  have h1 : shop_a + shop_b + shop_c + shop_d + shop_e = 240
  { sorry }
  have h2 : 240 / total_shops = 48
  { sorry }
  exact Eq.trans (congrArg (fun x => x / total_shops) h1) h2

end average_TV_sets_in_shops_l2281_228193


namespace justify_misha_decision_l2281_228109

-- Define the conditions based on the problem description
def reviews_smartphone_A := (7, 4) -- 7 positive and 4 negative reviews for A
def reviews_smartphone_B := (4, 1) -- 4 positive and 1 negative reviews for B

-- Define the ratios for each smartphone based on their reviews
def ratio_A := (reviews_smartphone_A.1 : ℚ) / reviews_smartphone_A.2
def ratio_B := (reviews_smartphone_B.1 : ℚ) / reviews_smartphone_B.2

-- Goal: to show that ratio_B > ratio_A, justifying Misha's decision
theorem justify_misha_decision : ratio_B > ratio_A := by
  -- placeholders to bypass the proof steps
  sorry

end justify_misha_decision_l2281_228109


namespace combined_area_is_correct_l2281_228165

def tract1_length := 300
def tract1_width  := 500
def tract2_length := 250
def tract2_width  := 630
def tract3_length := 350
def tract3_width  := 450
def tract4_length := 275
def tract4_width  := 600
def tract5_length := 325
def tract5_width  := 520

def area (length width : ℕ) : ℕ := length * width

theorem combined_area_is_correct :
  area tract1_length tract1_width +
  area tract2_length tract2_width +
  area tract3_length tract3_width +
  area tract4_length tract4_width +
  area tract5_length tract5_width = 799000 :=
by
  sorry

end combined_area_is_correct_l2281_228165


namespace find_z_solutions_l2281_228181

theorem find_z_solutions (r : ℚ) (z : ℤ) (h : 2^z + 2 = r^2) : 
  (r = 2 ∧ z = 1) ∨ (r = -2 ∧ z = 1) ∨ (r = 3/2 ∧ z = -2) ∨ (r = -3/2 ∧ z = -2) :=
sorry

end find_z_solutions_l2281_228181


namespace triangle_reflection_not_necessarily_perpendicular_l2281_228136

theorem triangle_reflection_not_necessarily_perpendicular
  (P Q R : ℝ × ℝ)
  (hP : 0 ≤ P.1 ∧ 0 ≤ P.2)
  (hQ : 0 ≤ Q.1 ∧ 0 ≤ Q.2)
  (hR : 0 ≤ R.1 ∧ 0 ≤ R.2)
  (not_on_y_eq_x_P : P.1 ≠ P.2)
  (not_on_y_eq_x_Q : Q.1 ≠ Q.2)
  (not_on_y_eq_x_R : R.1 ≠ R.2) :
  ¬ (∃ (mPQ mPQ' : ℝ), 
      mPQ = (Q.2 - P.2) / (Q.1 - P.1) ∧ 
      mPQ' = (Q.1 - P.1) / (Q.2 - P.2) ∧ 
      mPQ * mPQ' = -1) :=
sorry

end triangle_reflection_not_necessarily_perpendicular_l2281_228136


namespace find_set_A_l2281_228164

def M : Set ℤ := {1, 3, 5, 7, 9}

def satisfiesCondition (A : Set ℤ) : Prop :=
  A ≠ ∅ ∧
  (∀ a ∈ A, a + 4 ∈ M) ∧
  (∀ a ∈ A, a - 4 ∈ M)

theorem find_set_A : ∃ A : Set ℤ, satisfiesCondition A ∧ A = {5} :=
  by
    sorry

end find_set_A_l2281_228164


namespace find_unique_digit_sets_l2281_228151

theorem find_unique_digit_sets (a b c : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c)
 (h4 : 22 * (a + b + c) = 462) :
  (a = 4 ∧ b = 8 ∧ c = 9) ∨ 
  (a = 4 ∧ b = 9 ∧ c = 8) ∨ 
  (a = 8 ∧ b = 4 ∧ c = 9) ∨
  (a = 8 ∧ b = 9 ∧ c = 4) ∨ 
  (a = 9 ∧ b = 4 ∧ c = 8) ∨ 
  (a = 9 ∧ b = 8 ∧ c = 4) ∨
  (a = 5 ∧ b = 7 ∧ c = 9) ∨ 
  (a = 5 ∧ b = 9 ∧ c = 7) ∨ 
  (a = 7 ∧ b = 5 ∧ c = 9) ∨
  (a = 7 ∧ b = 9 ∧ c = 5) ∨ 
  (a = 9 ∧ b = 5 ∧ c = 7) ∨ 
  (a = 9 ∧ b = 7 ∧ c = 5) ∨
  (a = 6 ∧ b = 7 ∧ c = 8) ∨ 
  (a = 6 ∧ b = 8 ∧ c = 7) ∨ 
  (a = 7 ∧ b = 6 ∧ c = 8) ∨
  (a = 7 ∧ b = 8 ∧ c = 6) ∨ 
  (a = 8 ∧ b = 6 ∧ c = 7) ∨ 
  (a = 8 ∧ b = 7 ∧ c = 6) :=
sorry

end find_unique_digit_sets_l2281_228151


namespace rhombus_area_l2281_228140

theorem rhombus_area (d1 d2 : ℕ) (h1 : d1 = 18) (h2 : d2 = 14) : 
  (d1 * d2) / 2 = 126 := 
  by sorry

end rhombus_area_l2281_228140


namespace average_speed_of_car_l2281_228188

noncomputable def avgSpeed (Distance_uphill Speed_uphill Distance_downhill Speed_downhill : ℝ) : ℝ :=
  let Time_uphill := Distance_uphill / Speed_uphill
  let Time_downhill := Distance_downhill / Speed_downhill
  let Total_time := Time_uphill + Time_downhill
  let Total_distance := Distance_uphill + Distance_downhill
  Total_distance / Total_time

theorem average_speed_of_car:
  avgSpeed 100 30 50 60 = 36 := by
  sorry

end average_speed_of_car_l2281_228188


namespace birth_date_of_id_number_l2281_228100

def extract_birth_date (id_number : String) := 
  let birth_str := id_number.drop 6 |>.take 8
  let year := birth_str.take 4
  let month := birth_str.drop 4 |>.take 2
  let day := birth_str.drop 6
  (year, month, day)

theorem birth_date_of_id_number :
  extract_birth_date "320106194607299871" = ("1946", "07", "29") := by
  sorry

end birth_date_of_id_number_l2281_228100


namespace watermelon_cost_l2281_228184

-- Define the problem conditions
def container_full_conditions (w m : ℕ) : Prop :=
  w + m = 150 ∧ (w / 160) + (m / 120) = 1

def equal_total_values (w m w_value m_value : ℕ) : Prop :=
  w * w_value = m * m_value ∧ w * w_value + m * m_value = 24000

-- Define the proof problem
theorem watermelon_cost (w m w_value m_value : ℕ) (hw : container_full_conditions w m) (hv : equal_total_values w m w_value m_value) :
  w_value = 100 :=
by
  -- precise proof goes here
  sorry

end watermelon_cost_l2281_228184


namespace abs_neg_one_over_2023_l2281_228174

theorem abs_neg_one_over_2023 : abs (-1 / 2023) = 1 / 2023 :=
by
  sorry

end abs_neg_one_over_2023_l2281_228174


namespace model_lighthouse_height_l2281_228126

theorem model_lighthouse_height (h_actual : ℝ) (V_actual : ℝ) (V_model : ℝ) (h_actual_val : h_actual = 60) (V_actual_val : V_actual = 150000) (V_model_val : V_model = 0.15) :
  (h_actual * (V_model / V_actual)^(1/3)) = 0.6 :=
by
  rw [h_actual_val, V_actual_val, V_model_val]
  sorry

end model_lighthouse_height_l2281_228126


namespace student_weight_loss_l2281_228196

theorem student_weight_loss {S R L : ℕ} (h1 : S = 90) (h2 : S + R = 132) (h3 : S - L = 2 * R) : L = 6 := by
  sorry

end student_weight_loss_l2281_228196


namespace expenses_denoted_as_negative_l2281_228180

theorem expenses_denoted_as_negative (income_yuan expenses_yuan : Int) (h : income_yuan = 6) : 
  expenses_yuan = -4 :=
by
  sorry

end expenses_denoted_as_negative_l2281_228180


namespace find_b_l2281_228186

-- Define the curve and the line equations
def curve (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x + 1
def line (k : ℝ) (b : ℝ) (x : ℝ) : ℝ := k * x + b

-- Define the conditions in the problem
def passes_through_point (a : ℝ) : Prop := curve a 2 = 3
def is_tangent_at_point (a k b : ℝ) : Prop :=
  ∀ x : ℝ, curve a x = 3 → line k b 2 = 3

-- Main theorem statement
theorem find_b (a k b : ℝ) (h1 : passes_through_point a) (h2 : is_tangent_at_point a k b) : b = -15 :=
by sorry

end find_b_l2281_228186


namespace universal_inequality_l2281_228163

theorem universal_inequality (x y : ℝ) : x^2 + y^2 ≥ 2 * x * y := 
by 
  sorry

end universal_inequality_l2281_228163


namespace yadav_spends_50_percent_on_clothes_and_transport_l2281_228128

variable (S : ℝ)
variable (monthly_savings : ℝ := 46800 / 12)
variable (clothes_transport_expense : ℝ := 3900)
variable (remaining_salary : ℝ := 0.40 * S)

theorem yadav_spends_50_percent_on_clothes_and_transport (h1 : remaining_salary = 2 * 3900) :
  (clothes_transport_expense / remaining_salary) * 100 = 50 :=
by
  -- skipping the proof steps
  sorry

end yadav_spends_50_percent_on_clothes_and_transport_l2281_228128


namespace balloon_totals_l2281_228179

-- Definitions
def Joan_blue := 40
def Joan_red := 30
def Joan_green := 0
def Joan_yellow := 0

def Melanie_blue := 41
def Melanie_red := 0
def Melanie_green := 20
def Melanie_yellow := 0

def Eric_blue := 0
def Eric_red := 25
def Eric_green := 0
def Eric_yellow := 15

-- Total counts
def total_blue := Joan_blue + Melanie_blue + Eric_blue
def total_red := Joan_red + Melanie_red + Eric_red
def total_green := Joan_green + Melanie_green + Eric_green
def total_yellow := Joan_yellow + Melanie_yellow + Eric_yellow

-- Statement of the problem
theorem balloon_totals :
  total_blue = 81 ∧
  total_red = 55 ∧
  total_green = 20 ∧
  total_yellow = 15 :=
by
  -- Proof omitted
  sorry

end balloon_totals_l2281_228179


namespace loss_percent_l2281_228195

theorem loss_percent (CP SP : ℝ) (h₁ : CP = 600) (h₂ : SP = 300) : 
  (CP - SP) / CP * 100 = 50 :=
by
  rw [h₁, h₂]
  norm_num

end loss_percent_l2281_228195


namespace positive_integers_between_300_and_1000_squared_l2281_228111

theorem positive_integers_between_300_and_1000_squared :
  ∃ n : ℕ, 300 < n^2 ∧ n^2 < 1000 → ∃ m : ℕ, m = 14 := sorry

end positive_integers_between_300_and_1000_squared_l2281_228111


namespace triangle_side_ineq_l2281_228160

theorem triangle_side_ineq (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : 
  a^2 * c + b^2 * a + c^2 * b < 1 / 8 := 
by 
  sorry

end triangle_side_ineq_l2281_228160


namespace total_monthly_bill_working_from_home_l2281_228143

def original_monthly_bill : ℝ := 60
def percentage_increase : ℝ := 0.30

theorem total_monthly_bill_working_from_home :
  original_monthly_bill + (original_monthly_bill * percentage_increase) = 78 := by
  sorry

end total_monthly_bill_working_from_home_l2281_228143


namespace zack_marbles_number_l2281_228145

-- Define the conditions as Lean definitions
def zack_initial_marbles (x : ℕ) :=
  (∃ k : ℕ, x = 3 * k + 5) ∧ (3 * 20 + 5 = 65)

-- State the theorem using the conditions
theorem zack_marbles_number : ∃ x : ℕ, zack_initial_marbles x ∧ x = 65 :=
by
  sorry

end zack_marbles_number_l2281_228145


namespace chord_length_l2281_228158

theorem chord_length
  (a b c A B C : ℝ)
  (h₁ : c * Real.sin C = 3 * a * Real.sin A + 3 * b * Real.sin B)
  (O : ℝ → ℝ → Prop)
  (hO : ∀ x y, O x y ↔ x^2 + y^2 = 12)
  (l : ℝ → ℝ → Prop)
  (hl : ∀ x y, l x y ↔ a * x - b * y + c = 0) :
  (2 * Real.sqrt ( (2 * Real.sqrt 3)^2 - (Real.sqrt 3)^2 )) = 6 :=
by
  sorry

end chord_length_l2281_228158


namespace boxes_contain_fruits_l2281_228171

-- Define the weights of the boxes
def box_weights : List ℕ := [15, 16, 18, 19, 20, 31]

-- Define the weight requirement for apples and pears
def weight_rel (apple_weight pear_weight : ℕ) : Prop := apple_weight = pear_weight / 2

-- Define the statement with the constraints, given conditions and assignments.
theorem boxes_contain_fruits (h1 : box_weights = [15, 16, 18, 19, 20, 31])
                             (h2 : ∃ apple_weight pear_weight, 
                                   weight_rel apple_weight pear_weight ∧ 
                                   pear_weight ∈ box_weights ∧ apple_weight ∈ box_weights)
                             (h3 : ∃ orange_weight, orange_weight ∈ box_weights ∧ 
                                   ∀ w, w ∈ box_weights → w ≠ orange_weight)
                             : (15 = 2 ∧ 19 = 3 ∧ 20 = 1 ∧ 31 = 3) := 
                             sorry

end boxes_contain_fruits_l2281_228171


namespace average_increase_fraction_l2281_228117

-- First, we define the given conditions:
def incorrect_mark : ℕ := 82
def correct_mark : ℕ := 62
def number_of_students : ℕ := 80

-- We state the theorem to prove that the fraction by which the average marks increased is 1/4. 
theorem average_increase_fraction (incorrect_mark correct_mark : ℕ) (number_of_students : ℕ) :
  (incorrect_mark - correct_mark) / number_of_students = 1 / 4 :=
by
  sorry

end average_increase_fraction_l2281_228117


namespace find_d_l2281_228172

noncomputable def median (x : ℕ) : ℕ := x + 4
noncomputable def mean (x d : ℕ) : ℕ := x + (13 + d) / 5

theorem find_d (x d : ℕ) (h : mean x d = median x + 5) : d = 32 := by
  sorry

end find_d_l2281_228172


namespace sum_gcd_lcm_63_2898_l2281_228107

theorem sum_gcd_lcm_63_2898 : Nat.gcd 63 2898 + Nat.lcm 63 2898 = 182575 :=
by
  sorry

end sum_gcd_lcm_63_2898_l2281_228107


namespace toy_poodle_height_l2281_228113

-- Define the heights of the poodles
variables (S M T : ℝ)

-- Conditions
def std_taller_min : Prop := S = M + 8
def min_taller_toy : Prop := M = T + 6
def std_height : Prop := S = 28

-- Goal: How tall is the toy poodle?
theorem toy_poodle_height (h1 : std_taller_min S M)
                          (h2 : min_taller_toy M T)
                          (h3 : std_height S) : T = 14 :=
by 
  sorry

end toy_poodle_height_l2281_228113


namespace min_ratio_area_of_incircle_circumcircle_rt_triangle_l2281_228194

variables (a b: ℝ)
variables (a' b' c: ℝ)

-- Conditions
def area_of_right_triangle (a b : ℝ) : ℝ := 
    0.5 * a * b

def incircle_radius (a' b' c : ℝ) : ℝ := 
    0.5 * (a' + b' - c)

def circumcircle_radius (c : ℝ) : ℝ := 
    0.5 * c

-- Condition of the problem
def condition (a b a' b' c : ℝ) : Prop :=
    incircle_radius a' b' c = circumcircle_radius c ∧ 
    a' + b' = 2 * c

-- The final proof problem
theorem min_ratio_area_of_incircle_circumcircle_rt_triangle (a b a' b' c : ℝ)
    (h_area_a : a = area_of_right_triangle a' b')
    (h_area_b : b = area_of_right_triangle a b)
    (h_condition : condition a b a' b' c) :
    (a / b ≥ 3 + 2 * Real.sqrt 2) :=
by
  sorry

end min_ratio_area_of_incircle_circumcircle_rt_triangle_l2281_228194


namespace average_monthly_balance_correct_l2281_228129

def january_balance : ℕ := 100
def february_balance : ℕ := 200
def march_balance : ℕ := 250
def april_balance : ℕ := 250
def may_balance : ℕ := 150
def june_balance : ℕ := 100

def total_balance : ℕ :=
  january_balance + february_balance + march_balance + april_balance + may_balance + june_balance

def number_of_months : ℕ := 6

def average_monthly_balance : ℕ :=
  total_balance / number_of_months

theorem average_monthly_balance_correct :
  average_monthly_balance = 175 := by
  sorry

end average_monthly_balance_correct_l2281_228129


namespace positive_difference_l2281_228175

theorem positive_difference:
  let a := (7^3 + 7^3) / 7
  let b := (7^3)^2 / 7
  b - a = 16709 :=
by
  sorry

end positive_difference_l2281_228175


namespace find_angle_C_find_side_c_l2281_228155

noncomputable section

-- Definitions and conditions for Part 1
def vectors_dot_product_sin_2C (A B C : ℝ) (m : ℝ × ℝ) (n : ℝ × ℝ) : Prop :=
  m = (Real.sin A, Real.cos A) ∧ n = (Real.cos B, Real.sin B) ∧ 
  ((m.1 * n.1 + m.2 * n.2) = Real.sin (2 * C))

def angles_of_triangle (A B C : ℝ) : Prop := 
  A + B + C = Real.pi

theorem find_angle_C (A B C : ℝ) (m n : ℝ × ℝ) :
  vectors_dot_product_sin_2C A B C m n → angles_of_triangle A B C → C = Real.pi / 3 :=
sorry

-- Definitions and conditions for Part 2
def sin_in_arithmetic_sequence (x y z : ℝ) : Prop :=
  x + z = 2 * y

def product_of_sides_cos_C (a b c : ℝ) (C : ℝ) : Prop :=
  (a * b * Real.cos C = 18) ∧ (Real.cos C = 1 / 2)

theorem find_side_c (A B C a b c : ℝ) (m n : ℝ × ℝ) :
  sin_in_arithmetic_sequence (Real.sin A) (Real.sin C) (Real.sin B) → 
  angles_of_triangle A B C → 
  product_of_sides_cos_C a b c C → 
  C = Real.pi / 3 → 
  c = 6 :=
sorry

end find_angle_C_find_side_c_l2281_228155


namespace inequality_solution_l2281_228110

theorem inequality_solution (x : ℝ) : (5 * x + 3 > 9 - 3 * x ∧ x ≠ 3) ↔ (x > 3 / 4 ∧ x ≠ 3) :=
by {
  sorry
}

end inequality_solution_l2281_228110


namespace head_start_proofs_l2281_228121

def HeadStartAtoB : ℕ := 150
def HeadStartAtoC : ℕ := 310
def HeadStartAtoD : ℕ := 400

def HeadStartBtoC : ℕ := HeadStartAtoC - HeadStartAtoB
def HeadStartCtoD : ℕ := HeadStartAtoD - HeadStartAtoC
def HeadStartBtoD : ℕ := HeadStartAtoD - HeadStartAtoB

theorem head_start_proofs :
  (HeadStartBtoC = 160) ∧
  (HeadStartCtoD = 90) ∧
  (HeadStartBtoD = 250) :=
by
  sorry

end head_start_proofs_l2281_228121


namespace cake_remaining_after_4_trips_l2281_228183

theorem cake_remaining_after_4_trips :
  ∀ (cake_portion_left_after_trip : ℕ → ℚ), 
    cake_portion_left_after_trip 0 = 1 ∧
    (∀ n, cake_portion_left_after_trip (n + 1) = cake_portion_left_after_trip n / 2) →
    cake_portion_left_after_trip 4 = 1 / 16 :=
by
  intros cake_portion_left_after_trip h
  have h0 : cake_portion_left_after_trip 0 = 1 := h.1
  have h1 : ∀ n, cake_portion_left_after_trip (n + 1) = cake_portion_left_after_trip n / 2 := h.2
  sorry

end cake_remaining_after_4_trips_l2281_228183


namespace octagon_area_in_square_l2281_228122

def main : IO Unit :=
  IO.println s!"Hello, Lean!"

theorem octagon_area_in_square :
  ∀ (s : ℝ), ∀ (area_square : ℝ), ∀ (area_octagon : ℝ),
  (s * 4 = 160) →
  (s = 40) →
  (area_square = s * s) →
  (area_square = 1600) →
  (∃ (area_triangle : ℝ), area_triangle = 50 ∧ 8 * area_triangle = 400) →
  (area_octagon = area_square - 400) →
  (area_octagon = 1200) :=
by
  intros s area_square area_octagon h1 h2 h3 h4 h5 h6
  sorry

end octagon_area_in_square_l2281_228122


namespace anne_cleaning_time_l2281_228187

theorem anne_cleaning_time (B A : ℝ)
  (h1 : 4 * (B + A) = 1)
  (h2 : 3 * (B + 2 * A) = 1) : 1 / A = 12 :=
by {
  sorry
}

end anne_cleaning_time_l2281_228187


namespace increase_in_area_l2281_228124

-- Define the initial side length and the increment.
def initial_side_length : ℕ := 6
def increment : ℕ := 1

-- Define the original area of the land.
def original_area : ℕ := initial_side_length * initial_side_length

-- Define the new side length after the increase.
def new_side_length : ℕ := initial_side_length + increment

-- Define the new area of the land.
def new_area : ℕ := new_side_length * new_side_length

-- Define the theorem that states the increase in area.
theorem increase_in_area : new_area - original_area = 13 := by
  sorry

end increase_in_area_l2281_228124


namespace find_other_solution_l2281_228153

theorem find_other_solution (x : ℚ) (hx : 45 * (2 / 5 : ℚ)^2 + 22 = 56 * (2 / 5 : ℚ) - 9) : x = 7 / 9 :=
by 
  sorry

end find_other_solution_l2281_228153


namespace find_a_if_x_is_1_root_l2281_228115

theorem find_a_if_x_is_1_root {a : ℝ} (h : (1 : ℝ)^2 + a * 1 - 2 = 0) : a = 1 :=
by sorry

end find_a_if_x_is_1_root_l2281_228115


namespace intersection_of_A_and_B_l2281_228133

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 - 1 > 0}

theorem intersection_of_A_and_B :
  A ∩ B = {2} :=
sorry

end intersection_of_A_and_B_l2281_228133


namespace intersection_unique_one_point_l2281_228157

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 7 * x + a
noncomputable def g (x : ℝ) : ℝ := -3 * x^2 + 5 * x - 6

theorem intersection_unique_one_point (a : ℝ) :
  (∃ x y, y = f x a ∧ y = g x) ↔ a = 3 := by
  sorry

end intersection_unique_one_point_l2281_228157


namespace images_per_memory_card_l2281_228178

-- Define the constants based on the conditions given in the problem
def daily_pictures : ℕ := 10
def years : ℕ := 3
def days_per_year : ℕ := 365
def cost_per_card : ℕ := 60
def total_spent : ℕ := 13140

-- Define the properties to be proved
theorem images_per_memory_card :
  (years * days_per_year * daily_pictures) / (total_spent / cost_per_card) = 50 :=
by
  sorry

end images_per_memory_card_l2281_228178


namespace total_boxes_l2281_228176

theorem total_boxes (w1 w2 : ℕ) (h1 : w1 = 400) (h2 : w1 = 2 * w2) : w1 + w2 = 600 := 
by
  sorry

end total_boxes_l2281_228176


namespace factorize_expr_l2281_228189

theorem factorize_expr (a b : ℝ) : a^2 - 2 * a * b = a * (a - 2 * b) := 
by 
  sorry

end factorize_expr_l2281_228189


namespace number_of_friends_l2281_228162

-- Let n be the number of friends
-- Given the conditions:
-- 1. 9 chicken wings initially.
-- 2. 7 more chicken wings cooked.
-- 3. Each friend gets 4 chicken wings.

theorem number_of_friends :
  let initial_wings := 9
  let additional_wings := 7
  let wings_per_friend := 4
  let total_wings := initial_wings + additional_wings
  let n := total_wings / wings_per_friend
  n = 4 :=
by
  sorry

end number_of_friends_l2281_228162


namespace part1_part2_l2281_228182

section Problem

open Real

noncomputable def f (x : ℝ) := exp x
noncomputable def g (x : ℝ) := log x - 2

theorem part1 (x : ℝ) (hx : x > 0) : g x ≥ - (exp 1) / x :=
by sorry

theorem part2 (a : ℝ) (h : ∀ x, x ≥ 0 → f x - 1 / (f x) ≥ a * x) : a ≤ 2 :=
by sorry

end Problem

end part1_part2_l2281_228182


namespace compare_neg_fractions_l2281_228150

theorem compare_neg_fractions : (-2 / 3 : ℚ) < -3 / 5 :=
by
  sorry

end compare_neg_fractions_l2281_228150


namespace rachel_hw_diff_l2281_228190

-- Definitions based on the conditions of the problem
def math_hw_pages := 15
def reading_hw_pages := 6

-- The statement we need to prove, including the conditions
theorem rachel_hw_diff : 
  math_hw_pages - reading_hw_pages = 9 := 
by
  sorry

end rachel_hw_diff_l2281_228190


namespace value_of_otimes_l2281_228112

variable (a b : ℚ)

/-- Define the operation ⊗ -/
def otimes (x y : ℚ) : ℚ := a^2 * x + b * y - 3

/-- Given conditions -/
axiom condition1 : otimes a b 1 (-3) = 2 

/-- Target proof -/
theorem value_of_otimes : otimes a b 2 (-6) = 7 :=
by
  sorry

end value_of_otimes_l2281_228112


namespace sufficient_but_not_necessary_l2281_228166

noncomputable def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

def z (a : ℝ) : ℂ := ⟨a^2 - 4, a + 1⟩

theorem sufficient_but_not_necessary (a : ℝ) (h : a = -2) : 
  is_purely_imaginary (z a) ∧ ¬(∀ a, is_purely_imaginary (z a) → a = -2) :=
by
  sorry

end sufficient_but_not_necessary_l2281_228166


namespace christine_final_throw_difference_l2281_228106

def christine_first_throw : ℕ := 20
def janice_first_throw : ℕ := christine_first_throw - 4
def christine_second_throw : ℕ := christine_first_throw + 10
def janice_second_throw : ℕ := janice_first_throw * 2
def janice_final_throw : ℕ := christine_first_throw + 17
def highest_throw : ℕ := 37

theorem christine_final_throw_difference :
  ∃ x : ℕ, christine_second_throw + x = highest_throw ∧ x = 7 := by 
sorry

end christine_final_throw_difference_l2281_228106


namespace Jovana_final_addition_l2281_228185

theorem Jovana_final_addition 
  (initial_amount added_initial removed final_amount x : ℕ)
  (h1 : initial_amount = 5)
  (h2 : added_initial = 9)
  (h3 : removed = 2)
  (h4 : final_amount = 28) :
  final_amount = initial_amount + added_initial - removed + x → x = 16 :=
by
  intros h
  sorry

end Jovana_final_addition_l2281_228185


namespace solve_xyz_l2281_228199

theorem solve_xyz (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 3 * y + x) : x + y + z = 14 * x :=
by
  sorry

end solve_xyz_l2281_228199


namespace probability_computation_l2281_228130

-- Definitions of individual success probabilities
def probability_Xavier_solving_problem : ℚ := 1 / 4
def probability_Yvonne_solving_problem : ℚ := 2 / 3
def probability_William_solving_problem : ℚ := 7 / 10
def probability_Zelda_solving_problem : ℚ := 5 / 8
def probability_Zelda_notsolving_problem : ℚ := 1 - probability_Zelda_solving_problem

-- The target probability that only Xavier, Yvonne, and William, but not Zelda, will solve the problem
def target_probability : ℚ := (1 / 4) * (2 / 3) * (7 / 10) * (3 / 8)

-- The simplified form of the computed probability
def simplified_target_probability : ℚ := 7 / 160

-- Lean 4 statement to prove the equality of the computed and the target probabilities
theorem probability_computation :
  target_probability = simplified_target_probability := by
  sorry

end probability_computation_l2281_228130


namespace simplify_expression_l2281_228152

theorem simplify_expression (x : ℝ) : 
  (x^3 * x^2 * x + (x^3)^2 + (-2 * x^2)^3) = -6 * x^6 := 
by 
  sorry

end simplify_expression_l2281_228152


namespace sum_binomial_2k_eq_2_2n_l2281_228101

open scoped BigOperators

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem sum_binomial_2k_eq_2_2n (n : ℕ) :
  ∑ k in Finset.range (n + 1), 2^k * binomial_coeff (2*n - k) n = 2^(2*n) := 
by
  sorry

end sum_binomial_2k_eq_2_2n_l2281_228101


namespace general_term_of_c_l2281_228198

theorem general_term_of_c (a b : ℕ → ℕ) (c : ℕ → ℕ) : 
  (∀ n, a n = 2 ^ n) →
  (∀ n, b n = 3 * n + 2) →
  (∀ n, ∃ m k, a n = b m ∧ n = 2 * k + 1 → c k = a n) →
  ∀ n, c n = 2 ^ (2 * n + 1) :=
by
  intros ha hb hc n
  have h' := hc n
  sorry

end general_term_of_c_l2281_228198


namespace num_triangles_with_area_2_l2281_228105

-- Define the grid and points
def is_grid_point (x y : ℕ) : Prop := x ≤ 3 ∧ y ≤ 3

-- Function to calculate the area of a triangle using vertices (x1, y1), (x2, y2), and (x3, y3)
def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℕ) : ℤ := 
  (x1 * y2 + x2 * y3 + x3 * y1) 
  - (y1 * x2 + y2 * x3 + y3 * x1)

-- Check if the area is 2 (since we are dealing with a lattice grid, 
-- we can consider non-fractional form by multiplying by 2 to avoid half-area)
def has_area_2 (x1 y1 x2 y2 x3 y3 : ℕ) : Prop :=
  abs (area_of_triangle x1 y1 x2 y2 x3 y3) = 4

-- Define the main theorem that needs to be proved
theorem num_triangles_with_area_2 : 
  ∃ (n : ℕ), n = 64 ∧
  ∀ (x1 y1 x2 y2 x3 y3 : ℕ), 
  is_grid_point x1 y1 ∧ is_grid_point x2 y2 ∧ is_grid_point x3 y3 ∧ 
  has_area_2 x1 y1 x2 y2 x3 y3 → n = 64 :=
sorry

end num_triangles_with_area_2_l2281_228105


namespace find_a_l2281_228131

-- Defining the problem conditions
def rational_eq (x a : ℝ) :=
  x / (x - 3) - 2 * a / (x - 3) = 2

def extraneous_root (x : ℝ) : Prop :=
  x = 3

-- Theorem: Given the conditions, prove that a = 3 / 2
theorem find_a (a : ℝ) : (∃ x, extraneous_root x ∧ rational_eq x a) → a = 3 / 2 :=
  by
    sorry

end find_a_l2281_228131


namespace problem_result_l2281_228132

def elongation_A : List ℕ := [545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
def elongation_B : List ℕ := [536, 527, 543, 530, 560, 533, 522, 550, 576, 536]

def z_i : List ℤ := List.zipWith (λ x y => x - y) elongation_A elongation_B

def sample_mean (lst : List ℤ) : ℚ :=
  (List.sum lst : ℚ) / List.length lst

def sample_variance (lst : List ℤ) : ℚ :=
  let mean := sample_mean lst
  (List.sum (lst.map (λ z => (z - mean) * (z - mean))) : ℚ) / List.length lst

def improvement_significance (mean : ℚ) (variance : ℚ) : Prop :=
  mean ≥ 2 * Real.sqrt (variance / 10)

theorem problem_result :
  sample_mean z_i = 11 ∧
  sample_variance z_i = 61 ∧
  improvement_significance (sample_mean z_i) (sample_variance z_i) :=
by
  sorry

end problem_result_l2281_228132


namespace greatest_divisor_less_than_30_l2281_228120

theorem greatest_divisor_less_than_30 :
  (∃ d, d ∈ {n | n ∣ 540 ∧ n < 30 ∧ n ∣ 180} ∧ ∀ m, m ∈ {n | n ∣ 540 ∧ n < 30 ∧ n ∣ 180} → m ≤ d) → 
  18 ∈ {n | n ∣ 540 ∧ n < 30 ∧ n ∣ 180} :=
by
  sorry

end greatest_divisor_less_than_30_l2281_228120
