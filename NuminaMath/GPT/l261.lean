import Mathlib

namespace power_minus_self_even_l261_26137

theorem power_minus_self_even (a n : ℕ) (ha : 0 < a) (hn : 0 < n) : Even (a^n - a) := by
  sorry

end power_minus_self_even_l261_26137


namespace auntie_em_can_park_l261_26158

-- Define the conditions as formal statements in Lean
def parking_lot_spaces : ℕ := 20
def cars_arriving : ℕ := 14
def suv_adjacent_spaces : ℕ := 2

-- Define the total number of ways to park 14 cars in 20 spaces
def total_ways_to_park : ℕ := Nat.choose parking_lot_spaces cars_arriving
-- Define the number of unfavorable configurations where the SUV cannot park
def unfavorable_configs : ℕ := Nat.choose (parking_lot_spaces - suv_adjacent_spaces + 1) (parking_lot_spaces - cars_arriving)

-- Final probability calculation
def probability_park_suv : ℚ := 1 - (unfavorable_configs / total_ways_to_park)

-- Mathematically equivalent statement to be proved
theorem auntie_em_can_park : probability_park_suv = 850 / 922 :=
by sorry

end auntie_em_can_park_l261_26158


namespace g_3_2_plus_g_3_5_l261_26134

def g (x y : ℚ) : ℚ :=
  if x + y ≤ 5 then (x * y - x + 3) / (3 * x) else (x * y - y - 3) / (-3 * y)

theorem g_3_2_plus_g_3_5 : g 3 2 + g 3 5 = 1/5 := by
  sorry

end g_3_2_plus_g_3_5_l261_26134


namespace value_of_seventh_observation_l261_26116

-- Given conditions
def sum_of_first_six_observations : ℕ := 90
def new_total_sum : ℕ := 98

-- Problem: prove the value of the seventh observation
theorem value_of_seventh_observation : new_total_sum - sum_of_first_six_observations = 8 :=
by
  sorry

end value_of_seventh_observation_l261_26116


namespace solve_diff_eq_l261_26148

def solution_of_diff_eq (x y : ℝ) (y' : ℝ → ℝ) : Prop :=
  (x + y) * y' x = 1

def initial_condition (y x : ℝ) : Prop :=
  y = 0 ∧ x = -1

theorem solve_diff_eq (x : ℝ) (y : ℝ) (y' : ℝ → ℝ) (h1 : initial_condition y x) (h2 : solution_of_diff_eq x y y') :
  y = -(x + 1) :=
by 
  sorry

end solve_diff_eq_l261_26148


namespace B_alone_finishes_in_19_point_5_days_l261_26193

-- Define the conditions
def is_half_good(A B : ℝ) : Prop := A = 1 / 2 * B
def together_finish_in_13_days(A B : ℝ) : Prop := (A + B) * 13 = 1

-- Define the statement
theorem B_alone_finishes_in_19_point_5_days (A B : ℝ) (h1 : is_half_good A B) (h2 : together_finish_in_13_days A B) :
  B * 19.5 = 1 :=
by
  sorry

end B_alone_finishes_in_19_point_5_days_l261_26193


namespace engineering_department_men_l261_26140

theorem engineering_department_men (total_students men_percentage women_count : ℕ) (h_percentage : men_percentage = 70) (h_women : women_count = 180) (h_total : total_students = (women_count * 100) / (100 - men_percentage)) : 
  (total_students * men_percentage / 100) = 420 :=
by
  sorry

end engineering_department_men_l261_26140


namespace radius_of_circle_l261_26124

-- Define the polar coordinates equation
def polar_circle (ρ θ : ℝ) : Prop := ρ = 6 * Real.cos θ

-- Define the conversion to Cartesian coordinates and the circle equation
def cartesian_circle (x y : ℝ) : Prop := (x - 3) ^ 2 + y ^ 2 = 9

-- Prove that given the polar coordinates equation, the radius of the circle is 3
theorem radius_of_circle : ∀ (ρ θ : ℝ), polar_circle ρ θ → ∃ r, r = 3 := by
  sorry

end radius_of_circle_l261_26124


namespace xyz_not_divisible_by_3_l261_26147

theorem xyz_not_divisible_by_3 (x y z : ℕ) (h1 : x % 2 = 1) (h2 : y % 2 = 1) (h3 : z % 2 = 1) 
  (h4 : Nat.gcd (Nat.gcd x y) z = 1) (h5 : (x^2 + y^2 + z^2) % (x + y + z) = 0) : 
  (x + y + z - 2) % 3 ≠ 0 :=
by
  sorry

end xyz_not_divisible_by_3_l261_26147


namespace general_form_equation_l261_26103

theorem general_form_equation (x : ℝ) : 
  x * (2 * x - 1) = 5 * (x + 3) ↔ 2 * x^2 - 6 * x - 15 = 0 := 
by 
  sorry

end general_form_equation_l261_26103


namespace largest_c_for_range_of_f_l261_26135

def has_real_roots (a b c : ℝ) : Prop :=
  b * b - 4 * a * c ≥ 0

theorem largest_c_for_range_of_f (c : ℝ) :
  (∃ x : ℝ, x^2 + 3 * x + c = 7) ↔ c ≤ 37 / 4 := by
  sorry

end largest_c_for_range_of_f_l261_26135


namespace value_of_a_l261_26113

theorem value_of_a {a : ℝ} (A : Set ℝ) (B : Set ℝ) (hA : A = {-1, 0, 2}) (hB : B = {2^a}) (hSub : B ⊆ A) : a = 1 := 
sorry

end value_of_a_l261_26113


namespace fritz_has_40_dollars_l261_26102

variable (F S R : ℝ)
variable (h1 : S = (1 / 2) * F + 4)
variable (h2 : R = 3 * S)
variable (h3 : R + S = 96)

theorem fritz_has_40_dollars : F = 40 :=
by
  sorry

end fritz_has_40_dollars_l261_26102


namespace min_inverse_ab_l261_26182

theorem min_inverse_ab (a b : ℝ) (h1 : a + a * b + 2 * b = 30) (h2 : a > 0) (h3 : b > 0) :
  ∃ m : ℝ, m = 1 / 18 ∧ (∀ x y : ℝ, (x + x * y + 2 * y = 30) → (x > 0) → (y > 0) → 1 / (x * y) ≥ m) :=
sorry

end min_inverse_ab_l261_26182


namespace total_miles_walked_l261_26183

def weekly_group_walk_miles : ℕ := 3 * 6

def Jamie_additional_walk_miles_per_week : ℕ := 2 * 6
def Sue_additional_walk_miles_per_week : ℕ := 1 * 6 -- half of Jamie's additional walk
def Laura_additional_walk_miles_per_week : ℕ := 1 * 3 -- 1 mile every two days for 6 days
def Melissa_additional_walk_miles_per_week : ℕ := 2 * 2 -- 2 miles every three days for 6 days
def Katie_additional_walk_miles_per_week : ℕ := 1 * 6

def Jamie_weekly_miles : ℕ := weekly_group_walk_miles + Jamie_additional_walk_miles_per_week
def Sue_weekly_miles : ℕ := weekly_group_walk_miles + Sue_additional_walk_miles_per_week
def Laura_weekly_miles : ℕ := weekly_group_walk_miles + Laura_additional_walk_miles_per_week
def Melissa_weekly_miles : ℕ := weekly_group_walk_miles + Melissa_additional_walk_miles_per_week
def Katie_weekly_miles : ℕ := weekly_group_walk_miles + Katie_additional_walk_miles_per_week

def weeks_in_month : ℕ := 4

def Jamie_monthly_miles : ℕ := Jamie_weekly_miles * weeks_in_month
def Sue_monthly_miles : ℕ := Sue_weekly_miles * weeks_in_month
def Laura_monthly_miles : ℕ := Laura_weekly_miles * weeks_in_month
def Melissa_monthly_miles : ℕ := Melissa_weekly_miles * weeks_in_month
def Katie_monthly_miles : ℕ := Katie_weekly_miles * weeks_in_month

def total_monthly_miles : ℕ :=
  Jamie_monthly_miles + Sue_monthly_miles + Laura_monthly_miles + Melissa_monthly_miles + Katie_monthly_miles

theorem total_miles_walked (month_has_30_days : Prop) : total_monthly_miles = 484 :=
by
  unfold total_monthly_miles
  unfold Jamie_monthly_miles Sue_monthly_miles Laura_monthly_miles Melissa_monthly_miles Katie_monthly_miles
  unfold Jamie_weekly_miles Sue_weekly_miles Laura_weekly_miles Melissa_weekly_miles Katie_weekly_miles
  unfold weekly_group_walk_miles Jamie_additional_walk_miles_per_week Sue_additional_walk_miles_per_week Laura_additional_walk_miles_per_week Melissa_additional_walk_miles_per_week Katie_additional_walk_miles_per_week
  unfold weeks_in_month
  sorry

end total_miles_walked_l261_26183


namespace min_value_of_inverse_sum_l261_26130

noncomputable def min_value (a b : ℝ) := ¬(1 ≤ a + 2*b)

theorem min_value_of_inverse_sum (a b : ℝ) (h : a + 2 * b = 1) (h_nonneg : 0 < a ∧ 0 < b) :
  (1 / a + 2 / b) ≥ 9 :=
sorry

end min_value_of_inverse_sum_l261_26130


namespace angle_in_third_quadrant_l261_26191

theorem angle_in_third_quadrant (θ : ℝ) (h : θ = 2010) : ((θ % 360) > 180 ∧ (θ % 360) < 270) :=
by
  sorry

end angle_in_third_quadrant_l261_26191


namespace rent_increase_percentage_l261_26115

theorem rent_increase_percentage :
  ∀ (initial_avg new_avg rent : ℝ) (num_friends : ℝ),
    num_friends = 4 →
    initial_avg = 800 →
    new_avg = 850 →
    rent = 800 →
    ((num_friends * new_avg) - (num_friends * initial_avg)) / rent * 100 = 25 :=
by
  intros initial_avg new_avg rent num_friends h_num h_initial h_new h_rent
  sorry

end rent_increase_percentage_l261_26115


namespace smallest_b_l261_26110

open Real

theorem smallest_b (a b : ℝ) (h1 : 2 < a) (h2 : a < b)
  (h3 : ¬ ∃ A B C : ℝ, A > 0 ∧ B > 0 ∧ C > 0 ∧ (A = 2 ∧ B = a ∧ C = b ∨ A = 2 ∧ B = b ∧ C = a ∨ A = a ∧ B = b ∧ C = 2) ∧ A + B > C ∧ A + C > B ∧ B + C > A)
  (h4 : ¬ ∃ A B C : ℝ, A > 0 ∧ B > 0 ∧ C > 0 ∧ (A = 1 / b ∧ B = 1 / a ∧ C = 2 ∨ A = 1 / a ∧ B = 1 / b ∧ C = 2 ∨ A = 1 / b ∧ B = 2 ∧ C = 1 / a ∨ A = 1 / a ∧ B = 2 ∧ C = 1 / b ∨ A = 2 ∧ B = 1 / a ∧ C = 1 / b ∨ A = 2 ∧ B = 1 / b ∧ C = 1 / a) ∧ A + B > C ∧ A + C > B ∧ B + C > A) :
  b = 2 := 
sorry

end smallest_b_l261_26110


namespace largest_divisor_of_square_difference_l261_26196

theorem largest_divisor_of_square_difference (m n : ℤ) (hm : m % 2 = 0) (hn : n % 2 = 0) (h : n < m) : 
  ∃ d, ∀ m n, (m % 2 = 0) → (n % 2 = 0) → (n < m) → d ∣ (m^2 - n^2) ∧ ∀ k, (∀ m n, (m % 2 = 0) → (n % 2 = 0) → (n < m) → k ∣ (m^2 - n^2)) → k ≤ d :=
sorry

end largest_divisor_of_square_difference_l261_26196


namespace expenditure_increase_36_percent_l261_26169

theorem expenditure_increase_36_percent
  (m : ℝ) -- mass of the bread
  (p_bread : ℝ) -- price of the bread
  (p_crust : ℝ) -- price of the crust
  (h1 : p_crust = 1.2 * p_bread) -- condition: crust is 20% more expensive
  (h2 : p_crust = 1.2 * p_bread) -- condition: crust is 20% more expensive
  (h3 : ∃ (m_crust : ℝ), m_crust = 0.75 * m) -- condition: crust is 25% lighter in weight
  (h4 : ∃ (m_consumed_bread : ℝ), m_consumed_bread = 0.85 * m) -- condition: 15% of bread dries out
  (h5 : ∃ (m_consumed_crust : ℝ), m_consumed_crust = 0.75 * m) -- condition: crust is consumed completely
  : (17 / 15) * (1.2 : ℝ) = 1.36 := 
by sorry

end expenditure_increase_36_percent_l261_26169


namespace sum_even_102_to_600_l261_26184

def sum_first_50_even : ℕ := 2550
def sum_even_602_to_700 : ℕ := 32550

theorem sum_even_102_to_600 : sum_even_602_to_700 - sum_first_50_even = 30000 :=
by
  -- The given sum of the first 50 positive even integers is 2550
  have h1 : sum_first_50_even = 2550 := by rfl
  
  -- The given sum of the even integers from 602 to 700 inclusive is 32550
  have h2 : sum_even_602_to_700 = 32550 := by rfl
  
  -- Therefore, the sum of the even integers from 102 to 600 is:
  have h3 : sum_even_602_to_700 - sum_first_50_even = 32550 - 2550 := by
    rw [h1, h2]
  
  -- Calculate the result
  exact h3

end sum_even_102_to_600_l261_26184


namespace area_of_triangle_l261_26186

noncomputable def triangle_area (AB AC θ : ℝ) : ℝ := 
  0.5 * AB * AC * Real.sin θ

theorem area_of_triangle (AB AC : ℝ) (θ : ℝ) (hAB : AB = 1) (hAC : AC = 2) (hθ : θ = 2 * Real.pi / 3) :
  triangle_area AB AC θ = 3 * Real.sqrt 3 / 14 :=
by
  rw [triangle_area, hAB, hAC, hθ]
  sorry

end area_of_triangle_l261_26186


namespace sum_of_smallest_integers_l261_26195

theorem sum_of_smallest_integers (x y : ℕ) (h1 : ∃ x, x > 0 ∧ (∃ n : ℕ, 720 * x = n^2) ∧ (∀ m : ℕ, m > 0 ∧ (∃ k : ℕ, 720 * m = k^2) → x ≤ m))
  (h2 : ∃ y, y > 0 ∧ (∃ p : ℕ, 720 * y = p^4) ∧ (∀ q : ℕ, q > 0 ∧ (∃ r : ℕ, 720 * q = r^4) → y ≤ q)) :
  x + y = 1130 := 
sorry

end sum_of_smallest_integers_l261_26195


namespace category_B_count_solution_hiring_probability_l261_26107

-- Definitions and conditions
def category_A_count : Nat := 12

def total_selected_housekeepers : Nat := 20
def category_B_selected_housekeepers : Nat := 16
def category_A_selected_housekeepers := total_selected_housekeepers - category_B_selected_housekeepers

-- The value of x
def category_B_count (x : Nat) : Prop :=
  (category_A_selected_housekeepers * x) / category_A_count = category_B_selected_housekeepers

-- Assertion for the value of x
theorem category_B_count_solution : category_B_count 48 :=
by sorry

-- Conditions for the second part of the problem
def remaining_category_A : Nat := 3
def remaining_category_B : Nat := 2
def total_remaining := remaining_category_A + remaining_category_B

def possible_choices := remaining_category_A * (remaining_category_A - 1) / 2 + remaining_category_A * remaining_category_B + remaining_category_B * (remaining_category_B - 1) / 2
def successful_choices := remaining_category_A * remaining_category_B

def probability (a b : Nat) := (successful_choices % total_remaining) / (possible_choices % total_remaining)

-- Assertion for the probability
theorem hiring_probability : probability remaining_category_A remaining_category_B = 3 / 5 :=
by sorry

end category_B_count_solution_hiring_probability_l261_26107


namespace max_value_l261_26188

-- Define the vector types
structure Vector2 where
  x : ℝ
  y : ℝ

-- Define the properties given in the problem
def a_is_unit_vector (a : Vector2) : Prop :=
  a.x^2 + a.y^2 = 1

def a_plus_b (a b : Vector2) : Prop :=
  a.x + b.x = 3 ∧ a.y + b.y = 4

-- Define dot product for the vectors
def dot_product (a b : Vector2) : ℝ :=
  a.x * b.x + a.y * b.y

-- The theorem statement
theorem max_value (a b : Vector2) (h1 : a_is_unit_vector a) (h2 : a_plus_b a b) :
  ∃ m, m = 5 ∧ ∀ c : ℝ, |1 + dot_product a b| ≤ m :=
  sorry

end max_value_l261_26188


namespace gumball_problem_l261_26109
-- Step d: Lean 4 statement conversion

/-- 
  Suppose Joanna initially had 40 gumballs, Jacques had 60 gumballs, 
  and Julia had 80 gumballs.
  Joanna purchased 5 times the number of gumballs she initially had,
  Jacques purchased 3 times the number of gumballs he initially had,
  and Julia purchased 2 times the number of gumballs she initially had.
  Prove that after adding their purchases:
  1. Each person will have 240 gumballs.
  2. If they combine all their gumballs and share them equally, 
     each person will still get 240 gumballs.
-/
theorem gumball_problem :
  let joanna_initial := 40 
  let jacques_initial := 60 
  let julia_initial := 80 
  let joanna_final := joanna_initial + 5 * joanna_initial 
  let jacques_final := jacques_initial + 3 * jacques_initial 
  let julia_final := julia_initial + 2 * julia_initial 
  let total_gumballs := joanna_final + jacques_final + julia_final 
  (joanna_final = 240) ∧ (jacques_final = 240) ∧ (julia_final = 240) ∧ 
  (total_gumballs / 3 = 240) :=
by
  let joanna_initial := 40 
  let jacques_initial := 60 
  let julia_initial := 80 
  let joanna_final := joanna_initial + 5 * joanna_initial 
  let jacques_final := jacques_initial + 3 * jacques_initial 
  let julia_final := julia_initial + 2 * julia_initial 
  let total_gumballs := joanna_final + jacques_final + julia_final 
  
  have h_joanna : joanna_final = 240 := sorry
  have h_jacques : jacques_final = 240 := sorry
  have h_julia : julia_final = 240 := sorry
  have h_total : total_gumballs / 3 = 240 := sorry
  
  exact ⟨h_joanna, h_jacques, h_julia, h_total⟩

end gumball_problem_l261_26109


namespace reece_climbs_15_times_l261_26123

/-
Given:
1. Keaton's ladder height: 30 feet.
2. Keaton climbs: 20 times.
3. Reece's ladder is 4 feet shorter than Keaton's ladder.
4. Total length of ladders climbed by both is 11880 inches.

Prove:
Reece climbed his ladder 15 times.
-/

theorem reece_climbs_15_times :
  let keaton_ladder_feet := 30
  let keaton_climbs := 20
  let reece_ladder_feet := keaton_ladder_feet - 4
  let total_length_inches := 11880
  let feet_to_inches (feet : ℕ) := 12 * feet
  let keaton_ladder_inches := feet_to_inches keaton_ladder_feet
  let reece_ladder_inches := feet_to_inches reece_ladder_feet
  let keaton_total_climbed := keaton_ladder_inches * keaton_climbs
  let reece_total_climbed := total_length_inches - keaton_total_climbed
  let reece_climbs := reece_total_climbed / reece_ladder_inches
  reece_climbs = 15 :=
by
  sorry

end reece_climbs_15_times_l261_26123


namespace acute_triangle_B_area_l261_26100

-- Basic setup for the problem statement
variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to respective angles

-- The theorem to be proven
theorem acute_triangle_B_area (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2) 
                              (h_sides : a = 2 * b * Real.sin A)
                              (h_a : a = 3 * Real.sqrt 3) 
                              (h_c : c = 5) : 
  B = π / 6 ∧ (1/2) * a * c * Real.sin B = (15 * Real.sqrt 3) / 4 :=
by
  sorry

end acute_triangle_B_area_l261_26100


namespace solution_set_of_inequality_l261_26153

theorem solution_set_of_inequality (x : ℝ) : x^2 - |x| - 2 ≤ 0 ↔ -2 ≤ x ∧ x ≤ 2 := by
  sorry

end solution_set_of_inequality_l261_26153


namespace find_m_values_l261_26129

theorem find_m_values (m : ℝ) : 
  (∃ A B : ℝ × ℝ, A = (2, 2) ∧ B = (m, 0) ∧ 
   ∃ r R : ℝ, r = 1 ∧ R = 3 ∧ 
   ∃ d : ℝ, d = abs (dist A B) ∧ d = (R + r)) →
  (m = 2 - 2 * Real.sqrt 3 ∨ m = 2 + 2 * Real.sqrt 3) := 
sorry

end find_m_values_l261_26129


namespace hannah_dog_food_l261_26155

def dog_food_consumption : Prop :=
  let dog1 : ℝ := 1.5 * 2
  let dog2 : ℝ := (1.5 * 2) * 1
  let dog3 : ℝ := (dog2 + 2.5) * 3
  let dog4 : ℝ := 1.2 * (dog2 + 2.5) * 2
  let dog5 : ℝ := 0.8 * 1.5 * 4
  let total_food := dog1 + dog2 + dog3 + dog4 + dog5
  total_food = 40.5

theorem hannah_dog_food : dog_food_consumption :=
  sorry

end hannah_dog_food_l261_26155


namespace better_sequence_is_BAB_l261_26162

def loss_prob_andrei : ℝ := 0.4
def loss_prob_boris : ℝ := 0.3

def win_prob_andrei : ℝ := 1 - loss_prob_andrei
def win_prob_boris : ℝ := 1 - loss_prob_boris

def prob_qualify_ABA : ℝ :=
  win_prob_andrei * loss_prob_boris * win_prob_andrei +
  win_prob_andrei * win_prob_boris +
  loss_prob_andrei * win_prob_boris * win_prob_andrei

def prob_qualify_BAB : ℝ :=
  win_prob_boris * loss_prob_andrei * win_prob_boris +
  win_prob_boris * win_prob_andrei +
  loss_prob_boris * win_prob_andrei * win_prob_boris

theorem better_sequence_is_BAB : prob_qualify_BAB = 0.742 ∧ prob_qualify_BAB > prob_qualify_ABA :=
by 
  sorry

end better_sequence_is_BAB_l261_26162


namespace new_monthly_savings_l261_26177

-- Definitions based on conditions
def monthly_salary := 4166.67
def initial_savings_percent := 0.20
def expense_increase_percent := 0.10

-- Calculations
def initial_savings := initial_savings_percent * monthly_salary
def initial_expenses := (1 - initial_savings_percent) * monthly_salary
def increased_expenses := initial_expenses + expense_increase_percent * initial_expenses
def new_savings := monthly_salary - increased_expenses

-- Lean statement to prove the question equals the answer given conditions
theorem new_monthly_savings :
  new_savings = 499.6704 := 
by
  sorry

end new_monthly_savings_l261_26177


namespace find_x_l261_26111

theorem find_x : 
  ∃ x : ℝ, 4.7 * 13.26 + 4.7 * 9.43 + 4.7 * x = 470 ∧ x = 77.31 :=
by
  sorry

end find_x_l261_26111


namespace farmer_apples_l261_26101

theorem farmer_apples (initial_apples : ℕ) (given_apples : ℕ) (final_apples : ℕ) 
  (h1 : initial_apples = 127) (h2 : given_apples = 88) 
  (h3 : final_apples = initial_apples - given_apples) : final_apples = 39 :=
by {
  -- proof steps would go here, but since only the statement is needed, we use 'sorry' to skip the proof
  sorry
}

end farmer_apples_l261_26101


namespace more_roses_than_orchids_l261_26192

-- Definitions
def roses_now : Nat := 12
def orchids_now : Nat := 2

-- Theorem statement
theorem more_roses_than_orchids : (roses_now - orchids_now) = 10 := by
  sorry

end more_roses_than_orchids_l261_26192


namespace number_of_ways_to_represent_5030_l261_26174

theorem number_of_ways_to_represent_5030 :
  let even := {x : ℕ | x % 2 = 0}
  let in_range := {x : ℕ | x ≤ 98}
  let valid_b := even ∩ in_range
  ∃ (M : ℕ), M = 150 ∧ ∀ (b3 b2 b1 b0 : ℕ), 
    b3 ∈ valid_b ∧ b2 ∈ valid_b ∧ b1 ∈ valid_b ∧ b0 ∈ valid_b →
    5030 = b3 * 10 ^ 3 + b2 * 10 ^ 2 + b1 * 10 + b0 → 
    M = 150 :=
  sorry

end number_of_ways_to_represent_5030_l261_26174


namespace actual_price_of_food_l261_26152

noncomputable def food_price (total_spent: ℝ) (tip_percent: ℝ) (tax_percent: ℝ) (discount_percent: ℝ) : ℝ :=
  let P := total_spent / ((1 + tip_percent) * (1 + tax_percent) * (1 - discount_percent))
  P

theorem actual_price_of_food :
  food_price 198 0.20 0.10 0.15 = 176.47 :=
by
  sorry

end actual_price_of_food_l261_26152


namespace find_x_value_l261_26157

/-- Defining the conditions given in the problem -/
structure HenrikhConditions where
  x : ℕ
  walking_time_per_block : ℕ := 60
  bicycle_time_per_block : ℕ := 20
  skateboard_time_per_block : ℕ := 40
  added_time_walking_over_bicycle : ℕ := 480
  added_time_walking_over_skateboard : ℕ := 240

/-- Defining a hypothesis based on the conditions -/
noncomputable def henrikh (c : HenrikhConditions) : Prop :=
  c.walking_time_per_block * c.x = c.bicycle_time_per_block * c.x + c.added_time_walking_over_bicycle ∧
  c.walking_time_per_block * c.x = c.skateboard_time_per_block * c.x + c.added_time_walking_over_skateboard

/-- The theorem to be proved -/
theorem find_x_value (c : HenrikhConditions) (h : henrikh c) : c.x = 12 := by
  sorry

end find_x_value_l261_26157


namespace product_of_numbers_l261_26126

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x^3 + y^3 = 9450) : x * y = -585 :=
  sorry

end product_of_numbers_l261_26126


namespace g_at_5_l261_26117

def g (x : ℝ) : ℝ := 3 * x^5 - 15 * x^4 + 30 * x^3 - 45 * x^2 + 24 * x + 50

theorem g_at_5 : g 5 = 2795 :=
by
  sorry

end g_at_5_l261_26117


namespace equation_of_line_perpendicular_l261_26150

theorem equation_of_line_perpendicular 
  (P : ℝ × ℝ) (hx : P.1 = -1) (hy : P.2 = 2)
  (a b c : ℝ) (h_line : 2 * a - 3 * b + 4 = 0)
  (l : ℝ → ℝ) (h_perpendicular : ∀ x, l x = -(3/2) * x)
  (h_passing : l (-1) = 2)
  : a * 3 + b * 2 - 1 = 0 :=
sorry

end equation_of_line_perpendicular_l261_26150


namespace avg_visitors_on_sundays_l261_26142

theorem avg_visitors_on_sundays (avg_other_days : ℕ) (avg_month : ℕ) (days_in_month sundays other_days : ℕ) (total_month_visitors : ℕ) (total_other_days_visitors : ℕ) (S : ℕ):
  avg_other_days = 240 →
  avg_month = 285 →
  days_in_month = 30 →
  sundays = 5 →
  other_days = 25 →
  total_month_visitors = avg_month * days_in_month →
  total_other_days_visitors = avg_other_days * other_days →
  5 * S + total_other_days_visitors = total_month_visitors →
  S = 510 :=
by
  intros _
          _
          _
          _
          _
          _
          _
          h
  -- Proof goes here
  sorry

end avg_visitors_on_sundays_l261_26142


namespace sherry_needs_bananas_l261_26131

/-
Conditions:
- Sherry wants to make 99 loaves.
- Her recipe makes enough batter for 3 loaves.
- The recipe calls for 1 banana per batch of 3 loaves.

Question:
- How many bananas does Sherry need?

Equivalent Proof Problem:
- Prove that given the conditions, the number of bananas needed is 33.
-/

def total_loaves : ℕ := 99
def loaves_per_batch : ℕ := 3
def bananas_per_batch : ℕ := 1

theorem sherry_needs_bananas :
  (total_loaves / loaves_per_batch) * bananas_per_batch = 33 :=
sorry

end sherry_needs_bananas_l261_26131


namespace molecular_weight_proof_l261_26104

noncomputable def molecular_weight_C7H6O2 := 
  (7 * 12.01) + (6 * 1.008) + (2 * 16.00) -- molecular weight of one mole of C7H6O2

noncomputable def total_molecular_weight_9_moles := 
  9 * molecular_weight_C7H6O2 -- total molecular weight of 9 moles of C7H6O2

theorem molecular_weight_proof : 
  total_molecular_weight_9_moles = 1099.062 := 
by
  sorry

end molecular_weight_proof_l261_26104


namespace find_x_parallel_l261_26181

theorem find_x_parallel (x : ℝ) 
  (a : ℝ × ℝ := (x, 2)) 
  (b : ℝ × ℝ := (2, 4)) 
  (h : a.1 * b.2 = a.2 * b.1) :
  x = 1 := 
by
  sorry

end find_x_parallel_l261_26181


namespace length_of_MN_eq_5_sqrt_10_div_3_l261_26108

theorem length_of_MN_eq_5_sqrt_10_div_3 
  (A : ℝ × ℝ) 
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)
  (D : ℝ × ℝ)
  (M : ℝ × ℝ)
  (N : ℝ × ℝ)
  (hyp_A : A = (1, 3))
  (hyp_B : B = (25 / 3, 5 / 3))
  (hyp_C : C = (22 / 3, 14 / 3))
  (hyp_eq_edges : (dist (0, 0) M = dist M N) ∧ (dist M N = dist N B))
  (hyp_D : D = (5 / 2, 15 / 2))
  (hyp_M : M = (5 / 3, 5)) :
  dist M N = 5 * Real.sqrt 10 / 3 :=
sorry

end length_of_MN_eq_5_sqrt_10_div_3_l261_26108


namespace lloyd_earnings_l261_26121

theorem lloyd_earnings:
  let regular_hours := 7.5
  let regular_rate := 4.50
  let overtime_multiplier := 2.0
  let hours_worked := 10.5
  let overtime_hours := hours_worked - regular_hours
  let overtime_rate := overtime_multiplier * regular_rate
  let regular_pay := regular_hours * regular_rate
  let overtime_pay := overtime_hours * overtime_rate
  let total_earnings := regular_pay + overtime_pay
  total_earnings = 60.75 :=
by
  sorry

end lloyd_earnings_l261_26121


namespace ratio_of_areas_inequality_l261_26185

theorem ratio_of_areas_inequality (a x m : ℝ) (h1 : a > 0) (h2 : x > 0) (h3 : x < a) :
  m = (3 * x^2 - 3 * a * x + a^2) / a^2 →
  (1 / 4 ≤ m ∧ m < 1) :=
sorry

end ratio_of_areas_inequality_l261_26185


namespace sum_of_digits_palindrome_l261_26138

theorem sum_of_digits_palindrome 
  (r : ℕ) 
  (h1 : r ≤ 36) 
  (x p q : ℕ) 
  (h2 : 2 * q = 5 * p) 
  (h3 : x = p * r^3 + p * r^2 + q * r + q) 
  (h4 : ∃ (a b c : ℕ), (x * x = a * r^6 + b * r^5 + c * r^4 + 0 * r^3 + c * r^2 + b * r + a)) : 
  (2 * (a + b + c) = 36) := 
sorry

end sum_of_digits_palindrome_l261_26138


namespace cubic_equation_unique_real_solution_l261_26198

theorem cubic_equation_unique_real_solution :
  (∃ (m : ℝ), ∀ x : ℝ, x^3 - 4*x - m = 0 → x = 2) ↔ m = -8 :=
by sorry

end cubic_equation_unique_real_solution_l261_26198


namespace diagonal_length_l261_26170

theorem diagonal_length (d : ℝ) 
  (offset1 offset2 : ℝ) 
  (area : ℝ) 
  (h_offsets : offset1 = 11) 
  (h_offsets2 : offset2 = 9) 
  (h_area : area = 400) : d = 40 :=
by 
  sorry

end diagonal_length_l261_26170


namespace cube_root_of_neg_eight_squared_is_neg_four_l261_26187

-- Define the value of -8^2
def neg_eight_squared : ℤ := -8^2

-- Define what it means for a number to be the cube root of another number
def is_cube_root (a b : ℤ) : Prop := a^3 = b

-- The desired proof statement
theorem cube_root_of_neg_eight_squared_is_neg_four :
  neg_eight_squared = -64 → is_cube_root (-4) neg_eight_squared :=
by
  sorry

end cube_root_of_neg_eight_squared_is_neg_four_l261_26187


namespace num_rectangles_in_grid_l261_26171

theorem num_rectangles_in_grid : 
  let width := 35
  let height := 44
  ∃ n, n = 87 ∧ 
  ∀ x y, (1 ≤ x ∧ x ≤ width) ∧ (1 ≤ y ∧ y ≤ height) → 
    n = (x * (x + 1) / 2) * (y * (y + 1) / 2) := 
by
  sorry

end num_rectangles_in_grid_l261_26171


namespace playground_girls_l261_26149

theorem playground_girls (total_children boys girls : ℕ) (h1 : boys = 40) (h2 : total_children = 117) (h3 : total_children = boys + girls) : girls = 77 := 
by 
  sorry

end playground_girls_l261_26149


namespace B_work_time_l261_26139

noncomputable def workRateA (W : ℝ): ℝ := W / 14
noncomputable def combinedWorkRate (W : ℝ): ℝ := W / 10

theorem B_work_time (W : ℝ) :
  ∃ T : ℝ, (W / T) = (combinedWorkRate W) - (workRateA W) ∧ T = 35 :=
by {
  use 35,
  sorry
}

end B_work_time_l261_26139


namespace sum_of_n_terms_l261_26119

noncomputable def S : ℕ → ℕ :=
sorry -- We define S, but its exact form is not used in the statement directly

noncomputable def a : ℕ → ℕ := 
sorry -- We define a, but its exact form is not used in the statement directly

-- Conditions
axiom S3_eq : S 3 = 1
axiom a_rec : ∀ n : ℕ, 0 < n → a (n + 3) = 2 * (a n)

-- Proof problem
theorem sum_of_n_terms : S 2019 = 2^673 - 1 :=
sorry

end sum_of_n_terms_l261_26119


namespace find_k_series_sum_l261_26172

theorem find_k_series_sum :
  (∃ k : ℝ, 5 + ∑' n : ℕ, ((5 + (n + 1) * k) / 5^n.succ) = 10) →
  k = 12 :=
sorry

end find_k_series_sum_l261_26172


namespace discount_percentage_correct_l261_26161

-- Define the problem parameters as variables
variables (sale_price marked_price : ℝ) (discount_percentage : ℝ)

-- Provide the conditions from the problem
def conditions : Prop :=
  sale_price = 147.60 ∧ marked_price = 180

-- State the problem: Prove the discount percentage is 18%
theorem discount_percentage_correct (h : conditions sale_price marked_price) : 
  discount_percentage = 18 :=
by
  sorry

end discount_percentage_correct_l261_26161


namespace rachel_total_time_l261_26105

-- Define the conditions
def num_chairs : ℕ := 20
def num_tables : ℕ := 8
def time_per_piece : ℕ := 6

-- Proof statement
theorem rachel_total_time : (num_chairs + num_tables) * time_per_piece = 168 := by
  sorry

end rachel_total_time_l261_26105


namespace negation_of_existential_l261_26143

theorem negation_of_existential (h : ∃ x_0 : ℝ, x_0^3 - x_0^2 + 1 ≤ 0) : 
  ∀ x : ℝ, x^3 - x^2 + 1 > 0 :=
sorry

end negation_of_existential_l261_26143


namespace remainder_of_sum_of_integers_mod_15_l261_26120

theorem remainder_of_sum_of_integers_mod_15 (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) : (a + b + c) % 15 = 8 :=
by
  sorry

end remainder_of_sum_of_integers_mod_15_l261_26120


namespace smallest_positive_integer_congruence_l261_26160

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ, 0 < x ∧ x < 17 ∧ (3 * x ≡ 14 [MOD 17]) := sorry

end smallest_positive_integer_congruence_l261_26160


namespace weight_removed_l261_26144

-- Definitions for the given conditions
def weight_sugar : ℕ := 16
def weight_salt : ℕ := 30
def new_combined_weight : ℕ := 42

-- The proof problem statement
theorem weight_removed : (weight_sugar + weight_salt) - new_combined_weight = 4 := by
  -- Proof will be provided here
  sorry

end weight_removed_l261_26144


namespace speed_of_current_l261_26167

-- Define the context and variables
variables (m c : ℝ)
-- State the conditions
variables (h1 : m + c = 12) (h2 : m - c = 8)

-- State the goal which is to prove the speed of the current
theorem speed_of_current : c = 2 :=
by
  sorry

end speed_of_current_l261_26167


namespace quotient_of_division_l261_26168

theorem quotient_of_division (a b : ℕ) (r q : ℕ) (h1 : a = 1637) (h2 : b + 1365 = a) (h3 : a = b * q + r) (h4 : r = 5) : q = 6 :=
by
  -- Placeholder for proof
  sorry

end quotient_of_division_l261_26168


namespace ratio_hooper_bay_to_other_harbors_l261_26178

-- Definitions based on conditions
def other_harbors_lobster : ℕ := 80
def total_lobster : ℕ := 480
def combined_other_harbors_lobster := 2 * other_harbors_lobster
def hooper_bay_lobster := total_lobster - combined_other_harbors_lobster

-- The theorem to prove
theorem ratio_hooper_bay_to_other_harbors : hooper_bay_lobster / combined_other_harbors_lobster = 2 :=
by
  sorry

end ratio_hooper_bay_to_other_harbors_l261_26178


namespace tetrahedron_fourth_face_possibilities_l261_26197

theorem tetrahedron_fourth_face_possibilities :
  ∃ (S : Set String), S = {"right-angled triangle", "acute-angled triangle", "isosceles triangle", "isosceles right-angled triangle", "equilateral triangle"} :=
sorry

end tetrahedron_fourth_face_possibilities_l261_26197


namespace ordered_pair_l261_26118

-- Definitions
def P (x : ℝ) := x^4 - 8 * x^3 + 20 * x^2 - 34 * x + 15
def D (k : ℝ) (x : ℝ) := x^2 - 3 * x + k
def R (a : ℝ) (x : ℝ) := x + a

-- Hypothesis
def condition (k a : ℝ) : Prop := ∀ x : ℝ, P x % D k x = R a x

-- Theorem
theorem ordered_pair (k a : ℝ) (h : condition k a) : (k, a) = (5, 15) := 
  sorry

end ordered_pair_l261_26118


namespace three_times_greater_than_two_l261_26154

theorem three_times_greater_than_two (x : ℝ) : 3 * x - 2 > 0 → 3 * x > 2 :=
by
  sorry

end three_times_greater_than_two_l261_26154


namespace arithmetic_example_l261_26125

theorem arithmetic_example : 4 * (9 - 6) - 8 = 4 := by
  sorry

end arithmetic_example_l261_26125


namespace chord_length_range_l261_26166

variable {x y : ℝ}

def center : ℝ × ℝ := (4, 5)
def radius : ℝ := 13
def point : ℝ × ℝ := (1, 1)
def circle_eq (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 169

-- statement: prove the range of |AB| for specific conditions
theorem chord_length_range :
  ∀ line : (ℝ × ℝ) → (ℝ × ℝ) → Prop,
  (line center point → line (x, y) (x, y) ∧ circle_eq x y)
  → 24 ≤ abs (dist (x, y) (x, y)) ∧ abs (dist (x, y) (x, y)) ≤ 26 :=
by
  sorry

end chord_length_range_l261_26166


namespace students_in_both_band_and_chorus_l261_26179

-- Definitions for conditions
def total_students : ℕ := 300
def students_in_band : ℕ := 100
def students_in_chorus : ℕ := 120
def students_in_band_or_chorus : ℕ := 195

-- Theorem: Prove the number of students in both band and chorus
theorem students_in_both_band_and_chorus : ℕ :=
  students_in_band + students_in_chorus - students_in_band_or_chorus

example : students_in_both_band_and_chorus = 25 := by
  sorry

end students_in_both_band_and_chorus_l261_26179


namespace base_angles_isosceles_triangle_l261_26199

-- Define the conditions
def isIsoscelesTriangle (A B C : ℝ) : Prop :=
  (A = B ∨ B = C ∨ C = A)

def exteriorAngle (A B C : ℝ) (ext_angle : ℝ) : Prop :=
  ext_angle = (180 - (A + B)) ∨ ext_angle = (180 - (B + C)) ∨ ext_angle = (180 - (C + A))

-- Define the theorem
theorem base_angles_isosceles_triangle (A B C : ℝ) (ext_angle : ℝ) :
  isIsoscelesTriangle A B C ∧ exteriorAngle A B C ext_angle ∧ ext_angle = 110 →
  A = 55 ∨ A = 70 ∨ B = 55 ∨ B = 70 ∨ C = 55 ∨ C = 70 :=
by sorry

end base_angles_isosceles_triangle_l261_26199


namespace sum_abc_l261_26136

noncomputable def f (a b c : ℕ) (x : ℤ) : ℤ :=
  if x > 0 then a * x + 3
  else if x = 0 then a * b
  else b * x^2 + c

theorem sum_abc (a b c : ℕ) (h1 : f a b c 2 = 7) (h2 : f a b c 0 = 6) (h3 : f a b c (-1) = 8) :
  a + b + c = 10 :=
by {
  sorry
}

end sum_abc_l261_26136


namespace MrsHiltReadTotalChapters_l261_26176

-- Define the number of books and chapters per book
def numberOfBooks : ℕ := 4
def chaptersPerBook : ℕ := 17

-- Define the total number of chapters Mrs. Hilt read
def totalChapters (books : ℕ) (chapters : ℕ) : ℕ := books * chapters

-- The main statement to be proved
theorem MrsHiltReadTotalChapters : totalChapters numberOfBooks chaptersPerBook = 68 := by
  sorry

end MrsHiltReadTotalChapters_l261_26176


namespace find_range_g_l261_26145

noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x) + abs x

theorem find_range_g :
  {x : ℝ | g (2 * x - 1) < g 3} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end find_range_g_l261_26145


namespace smallest_four_digit_palindrome_div7_eq_1661_l261_26141

theorem smallest_four_digit_palindrome_div7_eq_1661 :
  ∃ (A B : ℕ), (A == 1 ∨ A == 3 ∨ A == 5 ∨ A == 7 ∨ A == 9) ∧
  (1000 ≤ 1100 * A + 11 * B ∧ 1100 * A + 11 * B < 10000) ∧
  (1100 * A + 11 * B) % 7 = 0 ∧
  (1100 * A + 11 * B) = 1661 :=
by
  sorry

end smallest_four_digit_palindrome_div7_eq_1661_l261_26141


namespace probability_of_event_A_l261_26122

def total_balls : ℕ := 10
def white_balls : ℕ := 7
def black_balls : ℕ := 3

def event_A : Prop := (black_balls / total_balls) * (white_balls / (total_balls - 1)) = 7 / 30

theorem probability_of_event_A : event_A := by
  sorry

end probability_of_event_A_l261_26122


namespace map_distance_ratio_l261_26175

theorem map_distance_ratio (actual_distance_km : ℕ) (map_distance_cm : ℕ) (h1 : actual_distance_km = 6) (h2 : map_distance_cm = 20) : map_distance_cm / (actual_distance_km * 100000) = 1 / 30000 :=
by
  -- Proof goes here
  sorry

end map_distance_ratio_l261_26175


namespace smallest_x_for_multiple_l261_26159

theorem smallest_x_for_multiple 
  (x : ℕ) (h₁ : ∀ m : ℕ, 450 * x = 800 * m) 
  (h₂ : ∀ y : ℕ, (∀ m : ℕ, 450 * y = 800 * m) → x ≤ y) : 
  x = 16 := 
sorry

end smallest_x_for_multiple_l261_26159


namespace find_value_l261_26146

variable (a : ℝ) (h : a + 1/a = 7)

theorem find_value :
  a^2 + 1/a^2 = 47 :=
sorry

end find_value_l261_26146


namespace sum_of_four_triangles_l261_26194

theorem sum_of_four_triangles :
  ∀ (x y : ℝ), 3 * x + 2 * y = 27 → 2 * x + 3 * y = 23 → 4 * y = 12 :=
by
  intros x y h1 h2
  sorry

end sum_of_four_triangles_l261_26194


namespace number_of_questions_in_exam_l261_26112

theorem number_of_questions_in_exam :
  ∀ (typeA : ℕ) (typeB : ℕ) (timeA : ℝ) (timeB : ℝ) (totalTime : ℝ),
    typeA = 100 →
    timeA = 1.2 →
    timeB = 0.6 →
    totalTime = 180 →
    120 = typeA * timeA →
    totalTime - 120 = typeB * timeB →
    typeA + typeB = 200 :=
by
  intros typeA typeB timeA timeB totalTime h_typeA h_timeA h_timeB h_totalTime h_timeA_calc h_remaining_time
  sorry

end number_of_questions_in_exam_l261_26112


namespace power_function_properties_l261_26128

theorem power_function_properties (α : ℝ) (h : (3 : ℝ) ^ α = 27) :
  (α = 3) →
  (∀ x : ℝ, (x ^ α) = x ^ 3) ∧
  (∀ x : ℝ, x ^ α = -(((-x) ^ α))) ∧
  (∀ x y : ℝ, x < y → x ^ α < y ^ α) ∧
  (∀ y : ℝ, ∃ x : ℝ, x ^ α = y) :=
by
  sorry

end power_function_properties_l261_26128


namespace probability_of_grid_being_black_l261_26133

noncomputable def probability_grid_black_after_rotation : ℚ := sorry

theorem probability_of_grid_being_black:
  probability_grid_black_after_rotation = 429 / 21845 :=
sorry

end probability_of_grid_being_black_l261_26133


namespace intersection_singleton_one_l261_26165

-- Define sets A and B according to the given conditions
def setA : Set ℤ := { x | 0 < x ∧ x < 4 }
def setB : Set ℤ := { x | (x+1)*(x-2) < 0 }

-- Statement to prove A ∩ B = {1}
theorem intersection_singleton_one : setA ∩ setB = {1} :=
by 
  sorry

end intersection_singleton_one_l261_26165


namespace one_fourth_to_fourth_power_is_decimal_l261_26180

def one_fourth : ℚ := 1 / 4

theorem one_fourth_to_fourth_power_is_decimal :
  (one_fourth ^ 4 : ℚ) = 0.00390625 := 
by sorry

end one_fourth_to_fourth_power_is_decimal_l261_26180


namespace portia_high_school_students_l261_26151

theorem portia_high_school_students (P L : ℕ) (h1 : P = 4 * L) (h2 : P + L = 2500) : P = 2000 := by
  sorry

end portia_high_school_students_l261_26151


namespace evaluate_expression_l261_26164

def g (x : ℝ) := 3 * x^2 - 5 * x + 7

theorem evaluate_expression : 3 * g 2 + 2 * g (-4) = 177 := by
  sorry

end evaluate_expression_l261_26164


namespace value_of_x_squared_plus_inverse_squared_l261_26132

theorem value_of_x_squared_plus_inverse_squared (x : ℝ) (hx : x + 1 / x = 8) : x^2 + 1 / x^2 = 62 := 
sorry

end value_of_x_squared_plus_inverse_squared_l261_26132


namespace CarriageSharingEquation_l261_26190

theorem CarriageSharingEquation (x : ℕ) :
  (x / 3 + 2 = (x - 9) / 2) ↔
  (3 * ((x - 9) / 2) + 2 * 3 = x / 3 + 2) ∧ 
  (2 * ((x - 9) / 2) + 9 = x ∨ 2 * ((x - 9) / 2) + 9 < x) ∧ 
  (x / 3 + 2 < 3 * (x / 2) + 2 * 2 ∨ x / 3 + 2 = 3 * (x / 2) + 2 * 2) :=
sorry

end CarriageSharingEquation_l261_26190


namespace susie_initial_amount_l261_26106

-- Definitions for conditions:
def initial_amount (X : ℝ) : Prop :=
  X + 0.20 * X = 240

-- Main theorem to prove:
theorem susie_initial_amount (X : ℝ) (h : initial_amount X) : X = 200 :=
by 
  -- structured proof will go here
  sorry

end susie_initial_amount_l261_26106


namespace largest_remainder_division_by_11_l261_26127

theorem largest_remainder_division_by_11 (A B C : ℕ) (h : A = 11 * B + C) (hC : 0 ≤ C ∧ C < 11) : C ≤ 10 :=
  sorry

end largest_remainder_division_by_11_l261_26127


namespace pump_capacity_l261_26163

-- Define parameters and assumptions
def tank_volume : ℝ := 1000
def fill_percentage : ℝ := 0.85
def fill_time : ℝ := 1
def num_pumps : ℝ := 8
def pump_efficiency : ℝ := 0.75
def required_fill_volume : ℝ := fill_percentage * tank_volume

-- Assumed total effective capacity must meet the required fill volume
theorem pump_capacity (C : ℝ) : 
  (num_pumps * pump_efficiency * C = required_fill_volume) → 
  C = 850.0 / 6.0 :=
by
  sorry

end pump_capacity_l261_26163


namespace train_speed_l261_26156

-- Define the conditions as given in part (a)
def train_length : ℝ := 160
def crossing_time : ℝ := 6

-- Define the statement to prove
theorem train_speed :
  train_length / crossing_time = 26.67 :=
by
  sorry

end train_speed_l261_26156


namespace arithmetic_sequence_problem_l261_26189

theorem arithmetic_sequence_problem (a : ℕ → ℚ) (d : ℚ) (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 4 + 1 / 2 * a 7 + a 10 = 10) : a 3 + a 11 = 8 :=
sorry

end arithmetic_sequence_problem_l261_26189


namespace find_x_l261_26114
-- Lean 4 equivalent problem setup

-- Assuming a and b are the tens and units digits respectively.
def number (a b : ℕ) := 10 * a + b
def interchangedNumber (a b : ℕ) := 10 * b + a
def digitsDifference (a b : ℕ) := a - b

-- Given conditions
variable (a b k : ℕ)

def condition1 := number a b = k * digitsDifference a b
def condition2 (x : ℕ) := interchangedNumber a b = x * digitsDifference a b

-- Theorem to prove
theorem find_x (h1 : condition1 a b k) : ∃ x, condition2 a b x ∧ x = k - 9 := 
by sorry

end find_x_l261_26114


namespace find_desired_expression_l261_26173

variable (y : ℝ)

theorem find_desired_expression
  (h : y + Real.sqrt (y^2 - 4) + (1 / (y - Real.sqrt (y^2 - 4))) = 12) :
  y^2 + Real.sqrt (y^4 - 4) + (1 / (y^2 - Real.sqrt (y^4 - 4))) = 200 / 9 :=
sorry

end find_desired_expression_l261_26173
