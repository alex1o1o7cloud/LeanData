import Mathlib

namespace sqrt_inequality_l1821_182184

theorem sqrt_inequality (x : ℝ) (h₁ : 3 / 2 ≤ x) (h₂ : x ≤ 5) : 
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := 
sorry

end sqrt_inequality_l1821_182184


namespace find_length_of_AB_l1821_182198

open Real

theorem find_length_of_AB (A B C : ℝ) 
    (h1 : tan A = 3 / 4) 
    (h2 : B = 6) 
    (h3 : C = π / 2) : sqrt (B^2 + ((3/4) * B)^2) = 7.5 :=
by
  sorry

end find_length_of_AB_l1821_182198


namespace oil_leak_l1821_182106

theorem oil_leak (a b c : ℕ) (h₁ : a = 6522) (h₂ : b = 11687) (h₃ : c = b - a) : c = 5165 :=
by
  rw [h₁, h₂] at h₃
  exact h₃

end oil_leak_l1821_182106


namespace solve_inequalities_l1821_182115

theorem solve_inequalities (x : ℝ) :
  ( (-x + 3)/2 < x ∧ 2*(x + 6) ≥ 5*x ) ↔ (1 < x ∧ x ≤ 4) :=
by
  sorry

end solve_inequalities_l1821_182115


namespace Q_value_ratio_l1821_182195

noncomputable def g (x : ℂ) : ℂ := x^2009 + 19*x^2008 + 1

noncomputable def roots : Fin 2009 → ℂ := sorry -- Define distinct roots s1, s2, ..., s2009

noncomputable def Q (z : ℂ) : ℂ := sorry -- Define the polynomial Q of degree 2009

theorem Q_value_ratio :
  (∀ j : Fin 2009, Q (roots j + 2 / roots j) = 0) →
  (Q (2) / Q (-2) = 361 / 400) :=
sorry

end Q_value_ratio_l1821_182195


namespace polynomial_constant_l1821_182194

theorem polynomial_constant (P : ℝ → ℝ → ℝ) (h : ∀ x y : ℝ, P (x + y) (y - x) = P x y) : 
  ∃ c : ℝ, ∀ x y : ℝ, P x y = c := 
sorry

end polynomial_constant_l1821_182194


namespace charging_time_is_correct_l1821_182153

-- Lean definitions for the given conditions
def smartphone_charge_time : ℕ := 26
def tablet_charge_time : ℕ := 53
def phone_half_charge_time : ℕ := smartphone_charge_time / 2

-- Definition for the total charging time based on conditions
def total_charging_time : ℕ :=
  tablet_charge_time + phone_half_charge_time

-- Proof problem statement
theorem charging_time_is_correct : total_charging_time = 66 := by
  sorry

end charging_time_is_correct_l1821_182153


namespace condition_sufficient_but_not_necessary_l1821_182163

variable (a b : ℝ)

theorem condition_sufficient_but_not_necessary :
  (|a| < 1 ∧ |b| < 1) → (|1 - a * b| > |a - b|) ∧
  ((|1 - a * b| > |a - b|) → (|a| < 1 ∧ |b| < 1) ∨ (|a| ≥ 1 ∧ |b| ≥ 1)) :=
by
  sorry

end condition_sufficient_but_not_necessary_l1821_182163


namespace solution_set_of_inequality_l1821_182120

theorem solution_set_of_inequality :
  {x : ℝ | 2 * x^2 - 3 * x - 2 > 0} = {x : ℝ | x < -0.5 ∨ x > 2} := 
sorry

end solution_set_of_inequality_l1821_182120


namespace sara_dozen_quarters_l1821_182167

theorem sara_dozen_quarters (dollars : ℕ) (quarters_per_dollar : ℕ) (quarters_per_dozen : ℕ) 
  (h1 : dollars = 9) (h2 : quarters_per_dollar = 4) (h3 : quarters_per_dozen = 12) : 
  dollars * quarters_per_dollar / quarters_per_dozen = 3 := 
by 
  sorry

end sara_dozen_quarters_l1821_182167


namespace y_intercept_l1821_182128

theorem y_intercept (x y : ℝ) (h : 2 * x - 3 * y = 6) : x = 0 → y = -2 :=
by
  intro h₁
  sorry

end y_intercept_l1821_182128


namespace intersection_A_B_union_A_B_range_of_a_l1821_182164

open Set

-- Definitions for the given sets
def Universal : Set ℝ := univ
def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | 2 < x ∧ x ≤ 7}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a + 6}

-- Propositions to prove
theorem intersection_A_B : 
  A ∩ B = {x : ℝ | 3 ≤ x ∧ x ≤ 7} := 
  sorry

theorem union_A_B : 
  A ∪ B = {x : ℝ | 2 < x ∧ x < 10} := 
  sorry

theorem range_of_a (a : ℝ) : 
  (A ∪ C a = C a) → (2 ≤ a ∧ a < 3) := 
  sorry

end intersection_A_B_union_A_B_range_of_a_l1821_182164


namespace largest_even_integer_of_product_2880_l1821_182100

theorem largest_even_integer_of_product_2880 :
  ∃ n : ℤ, (n-2) * n * (n+2) = 2880 ∧ n + 2 = 22 := 
by {
  sorry
}

end largest_even_integer_of_product_2880_l1821_182100


namespace polygon_to_triangle_l1821_182148

theorem polygon_to_triangle {n : ℕ} (h : n > 4) :
  ∃ (a b c : ℕ), (a + b > c ∧ a + c > b ∧ b + c > a) :=
sorry

end polygon_to_triangle_l1821_182148


namespace weight_of_a_l1821_182182

-- Define conditions
def weight_of_b : ℕ := 750 -- weight of one liter of ghee packet of brand 'b' in grams
def ratio_a_to_b : ℕ × ℕ := (3, 2)
def total_volume_liters : ℕ := 4 -- total volume of the mixture in liters
def total_weight_grams : ℕ := 3360 -- total weight of the mixture in grams

-- Target proof statement
theorem weight_of_a (W_a : ℕ) 
  (h_ratio : (ratio_a_to_b.1 + ratio_a_to_b.2) = 5)
  (h_mix_vol_a : (ratio_a_to_b.1 * total_volume_liters) = 12)
  (h_mix_vol_b : (ratio_a_to_b.2 * total_volume_liters) = 8)
  (h_weight_eq : (ratio_a_to_b.1 * W_a * total_volume_liters + ratio_a_to_b.2 * weight_of_b * total_volume_liters) = total_weight_grams * 5) : 
  W_a = 900 :=
by {
  sorry
}

end weight_of_a_l1821_182182


namespace terry_nora_age_relation_l1821_182162

variable {N : ℕ} -- Nora's current age

theorem terry_nora_age_relation (h₁ : Terry_current_age = 30) (h₂ : Terry_future_age = 4 * N) : N = 10 :=
by
  --- additional assumptions
  have Terry_future_age_def : Terry_future_age = 30 + 10 := by sorry
  rw [Terry_future_age_def] at h₂
  linarith

end terry_nora_age_relation_l1821_182162


namespace ratio_of_fractions_proof_l1821_182118

noncomputable def ratio_of_fractions (x y : ℝ) : Prop :=
  (5 * x = 6 * y) → (x ≠ 0 ∧ y ≠ 0) → ((1/3) * x / ((1/5) * y) = 2)

theorem ratio_of_fractions_proof (x y : ℝ) (hx: 5 * x = 6 * y) (hnz: x ≠ 0 ∧ y ≠ 0) : ((1/3) * x / ((1/5) * y) = 2) :=
  by 
  sorry

end ratio_of_fractions_proof_l1821_182118


namespace puppies_per_dog_l1821_182183

def dogs := 15
def puppies := 75

theorem puppies_per_dog : puppies / dogs = 5 :=
by {
  sorry
}

end puppies_per_dog_l1821_182183


namespace regular_polygon_perimeter_l1821_182185

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l1821_182185


namespace fraction_absent_l1821_182116

theorem fraction_absent (p : ℕ) (x : ℚ) (h : (W / p) * 1.2 = W / (p * (1 - x))) : x = 1 / 6 :=
by
  sorry

end fraction_absent_l1821_182116


namespace basketball_player_ft_rate_l1821_182122

theorem basketball_player_ft_rate :
  ∃ P : ℝ, 1 - P^2 = 16 / 25 ∧ P = 3 / 5 := sorry

end basketball_player_ft_rate_l1821_182122


namespace line_tangent_to_parabola_l1821_182191

theorem line_tangent_to_parabola (k : ℝ) : 
  (∀ x y : ℝ, y^2 = 16 * x ∧ 4 * x + 3 * y + k = 0 → ∀ y, y^2 + 12 * y + 4 * k = 0 → (12)^2 - 4 * 1 * 4 * k = 0) → 
  k = 9 :=
by
  sorry

end line_tangent_to_parabola_l1821_182191


namespace points_opposite_sides_line_l1821_182155

theorem points_opposite_sides_line (m : ℝ) :
  (3 * 3 - 2 * 1 + m) * (3 * (-4) - 2 * 6 + m) < 0 ↔ -7 < m ∧ m < 24 :=
by
  sorry

end points_opposite_sides_line_l1821_182155


namespace trapezium_area_l1821_182157

theorem trapezium_area (a b h : ℝ) (ha : a = 24) (hb : b = 18) (hh : h = 15) : 
  1/2 * (a + b) * h = 315 ∧ h = 15 :=
by 
  -- The proof steps would go here
  sorry

end trapezium_area_l1821_182157


namespace find_x_when_water_added_l1821_182196

variable (m x : ℝ)

theorem find_x_when_water_added 
  (h1 : m > 25)
  (h2 : (m * m / 100) = ((m - 15) / 100) * (m + x)) :
  x = 15 * m / (m - 15) :=
sorry

end find_x_when_water_added_l1821_182196


namespace complex_equality_l1821_182171

theorem complex_equality (a b : ℝ) (h : (⟨0, 1⟩ : ℂ) ^ 3 = ⟨a, -b⟩) : a + b = 1 :=
by
  sorry

end complex_equality_l1821_182171


namespace find_value_l1821_182143

theorem find_value (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 2) : a^2 + b^2 + a * b = 7 :=
by
  sorry

end find_value_l1821_182143


namespace set_operation_correct_l1821_182150

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

-- Define the operation A * B
def set_operation (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- State the theorem to be proved
theorem set_operation_correct : set_operation A B = {1, 3} :=
sorry

end set_operation_correct_l1821_182150


namespace total_eggs_today_l1821_182179

def eggs_morning : ℕ := 816
def eggs_afternoon : ℕ := 523

theorem total_eggs_today : eggs_morning + eggs_afternoon = 1339 :=
by {
  sorry
}

end total_eggs_today_l1821_182179


namespace other_point_on_circle_l1821_182111

noncomputable def circle_center_radius (p : ℝ × ℝ) (r : ℝ) : Prop :=
  dist p (0, 0) = r

theorem other_point_on_circle (r : ℝ) (h : r = 16) (point_on_circle : circle_center_radius (16, 0) r) :
  circle_center_radius (-16, 0) r :=
by
  sorry

end other_point_on_circle_l1821_182111


namespace pet_store_earnings_l1821_182140

theorem pet_store_earnings :
  let kitten_price := 6
  let puppy_price := 5
  let kittens_sold := 2
  let puppies_sold := 1 
  let total_earnings := kittens_sold * kitten_price + puppies_sold * puppy_price
  total_earnings = 17 :=
by
  sorry

end pet_store_earnings_l1821_182140


namespace vehicle_combinations_count_l1821_182136

theorem vehicle_combinations_count :
  ∃ (x y : ℕ), (4 * x + y = 79) ∧ (∃ (n : ℕ), n = 19) :=
sorry

end vehicle_combinations_count_l1821_182136


namespace parallelogram_angle_B_eq_130_l1821_182147

theorem parallelogram_angle_B_eq_130 (A C B D : ℝ) (parallelogram_ABCD : true) 
(angles_sum_A_C : A + C = 100) (A_eq_C : A = C): B = 130 := by
  sorry

end parallelogram_angle_B_eq_130_l1821_182147


namespace loaned_books_count_l1821_182159

variable (x : ℕ)

def initial_books : ℕ := 75
def percentage_returned : ℝ := 0.65
def end_books : ℕ := 54
def non_returned_books : ℕ := initial_books - end_books
def percentage_non_returned : ℝ := 1 - percentage_returned

theorem loaned_books_count :
  percentage_non_returned * (x:ℝ) = non_returned_books → x = 60 :=
by
  sorry

end loaned_books_count_l1821_182159


namespace line_intersects_circle_midpoint_trajectory_l1821_182176

-- Definitions based on conditions
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

def line_eq (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Statement of the problem
theorem line_intersects_circle :
  ∀ m : ℝ, ∃ (x y : ℝ), circle_eq x y ∧ line_eq m x y :=
sorry

theorem midpoint_trajectory :
  ∀ (x y : ℝ), 
  (∃ (xa ya xb yb : ℝ), circle_eq xa ya ∧ line_eq m xa ya ∧ 
   circle_eq xb yb ∧ line_eq m xb yb ∧ (x, y) = ((xa + xb) / 2, (ya + yb) / 2)) ↔
   ( x - 1 / 2)^2 + (y - 1)^2 = 1 / 4 :=
sorry

end line_intersects_circle_midpoint_trajectory_l1821_182176


namespace value_of_a_l1821_182139

theorem value_of_a (m : ℝ) (f : ℝ → ℝ) (h : f = fun x => (1/3)^x + m - 1/3) 
  (h_m : ∀ x, f x ≥ 0 ↔ m ≥ -2/3) : m ≥ -2/3 :=
by
  sorry

end value_of_a_l1821_182139


namespace find_a_and_other_root_l1821_182154

theorem find_a_and_other_root (a : ℝ) (h : (2 : ℝ) ^ 2 - 3 * (2 : ℝ) + a = 0) :
  a = 2 ∧ ∃ x : ℝ, x ^ 2 - 3 * x + a = 0 ∧ x ≠ 2 ∧ x = 1 := 
by
  sorry

end find_a_and_other_root_l1821_182154


namespace boris_climbs_needed_l1821_182166

-- Definitions
def elevation_hugo : ℕ := 10000
def shorter_difference : ℕ := 2500
def climbs_hugo : ℕ := 3

-- Derived Definitions
def elevation_boris : ℕ := elevation_hugo - shorter_difference
def total_climbed_hugo : ℕ := climbs_hugo * elevation_hugo

-- Theorem
theorem boris_climbs_needed : (total_climbed_hugo / elevation_boris) = 4 :=
by
  -- conditions and definitions are used here
  sorry

end boris_climbs_needed_l1821_182166


namespace simplify_polynomial_l1821_182180

variable (r : ℝ)

theorem simplify_polynomial : (2 * r^2 + 5 * r - 7) - (r^2 + 9 * r - 3) = r^2 - 4 * r - 4 := by
  sorry

end simplify_polynomial_l1821_182180


namespace two_trains_meet_at_distance_l1821_182188

theorem two_trains_meet_at_distance 
  (D_slow D_fast : ℕ)  -- Distances traveled by the slower and faster trains
  (T : ℕ)  -- Time taken to meet
  (h0 : 16 * T = D_slow)  -- Distance formula for slower train
  (h1 : 21 * T = D_fast)  -- Distance formula for faster train
  (h2 : D_fast = D_slow + 60)  -- Faster train travels 60 km more than slower train
  : (D_slow + D_fast = 444) := sorry

end two_trains_meet_at_distance_l1821_182188


namespace intersecting_lines_l1821_182138

theorem intersecting_lines {c d : ℝ} 
  (h₁ : 12 = 2 * 4 + c) 
  (h₂ : 12 = -4 + d) : 
  c + d = 20 := 
sorry

end intersecting_lines_l1821_182138


namespace intersection_has_one_element_l1821_182177

noncomputable def A (a : ℝ) : Set ℝ := {1, a, 5}
noncomputable def B (a : ℝ) : Set ℝ := {2, a^2 + 1}

theorem intersection_has_one_element (a : ℝ) (h : ∃ x, A a ∩ B a = {x}) : a = 0 ∨ a = -2 :=
by {
  sorry
}

end intersection_has_one_element_l1821_182177


namespace binomial_7_4_l1821_182172

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_7_4 : binomial 7 4 = 35 := by
  sorry

end binomial_7_4_l1821_182172


namespace intersection_subset_l1821_182199

def set_A : Set ℝ := {x | -4 < x ∧ x < 2}
def set_B : Set ℝ := {x | x > 1 ∨ x < -5}
def set_C (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < m}

theorem intersection_subset (m : ℝ) :
  (set_A ∩ set_B) ⊆ set_C m ↔ m = 2 :=
by
  sorry

end intersection_subset_l1821_182199


namespace max_omega_l1821_182175

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x)

theorem max_omega :
  (∃ ω > 0, (∃ k : ℤ, (f ω (2 * π / 3) = 0) ∧ (ω = 3 / 2 * k)) ∧ (0 < ω * π / 14 ∧ ω * π / 14 ≤ π / 2)) →
  ∃ ω, ω = 6 :=
by
  sorry

end max_omega_l1821_182175


namespace sum_of_tens_and_units_digit_of_7_pow_2023_l1821_182103

theorem sum_of_tens_and_units_digit_of_7_pow_2023 :
  let n := 7 ^ 2023
  (n % 100).div 10 + (n % 10) = 16 :=
by
  sorry

end sum_of_tens_and_units_digit_of_7_pow_2023_l1821_182103


namespace student_question_choice_l1821_182156

/-- A student needs to choose 8 questions from part A and 5 questions from part B. Both parts contain 10 questions each.
   This Lean statement proves that the student can choose the questions in 11340 different ways. -/
theorem student_question_choice : (Nat.choose 10 8) * (Nat.choose 10 5) = 11340 := by
  sorry

end student_question_choice_l1821_182156


namespace son_age_l1821_182113

-- Defining the variables
variables (S F : ℕ)

-- The conditions
def condition1 : Prop := F = S + 25
def condition2 : Prop := F + 2 = 2 * (S + 2)

-- The statement to be proved
theorem son_age (h1 : condition1 S F) (h2 : condition2 S F) : S = 23 :=
sorry

end son_age_l1821_182113


namespace number_of_classmates_late_l1821_182170

-- Definitions based on conditions from problem statement
def charlizeLate : ℕ := 20
def classmateLate : ℕ := charlizeLate + 10
def totalLateTime : ℕ := 140

-- The proof statement
theorem number_of_classmates_late (x : ℕ) (h1 : totalLateTime = charlizeLate + x * classmateLate) : x = 4 :=
by
  sorry

end number_of_classmates_late_l1821_182170


namespace cost_of_45_daffodils_equals_75_l1821_182152

-- Conditions
def cost_of_15_daffodils : ℝ := 25
def number_of_daffodils_in_bouquet_15 : ℕ := 15
def number_of_daffodils_in_bouquet_45 : ℕ := 45
def directly_proportional (n m : ℕ) (c_n c_m : ℝ) : Prop := c_n / n = c_m / m

-- Statement to prove
theorem cost_of_45_daffodils_equals_75 :
  ∀ (c : ℝ), directly_proportional number_of_daffodils_in_bouquet_45 number_of_daffodils_in_bouquet_15 c cost_of_15_daffodils → c = 75 :=
by
  intro c hypothesis
  -- Proof would go here.
  sorry

end cost_of_45_daffodils_equals_75_l1821_182152


namespace complex_eq_l1821_182178

theorem complex_eq (a b : ℝ) (i : ℂ) (hi : i^2 = -1) (h : (a + 2 * i) / i = b + i) : a + b = 1 :=
sorry

end complex_eq_l1821_182178


namespace area_of_triangle_DEF_l1821_182109

-- Define point D
def pointD : ℝ × ℝ := (2, 5)

-- Reflect D over the y-axis to get E
def reflectY (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, P.2)
def pointE : ℝ × ℝ := reflectY pointD

-- Reflect E over the line y = -x to get F
def reflectYX (P : ℝ × ℝ) : ℝ × ℝ := (-P.2, -P.1)
def pointF : ℝ × ℝ := reflectYX pointE

-- Define function to calculate the area of the triangle given three points
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

-- Define the Lean 4 statement
theorem area_of_triangle_DEF : triangle_area pointD pointE pointF = 6 := by
  sorry

end area_of_triangle_DEF_l1821_182109


namespace increasing_power_function_l1821_182130

theorem increasing_power_function (m : ℝ) (h_power : m^2 - 1 = 1)
    (h_increasing : ∀ x : ℝ, x > 0 → (m^2 - 1) * m * x^(m-1) > 0) : m = Real.sqrt 2 :=
by
  sorry

end increasing_power_function_l1821_182130


namespace coin_difference_l1821_182145

/-- 
  Given that Paul has 5-cent, 20-cent, and 15-cent coins, 
  prove that the difference between the maximum and minimum number of coins
  needed to make exactly 50 cents is 6.
-/
theorem coin_difference :
  ∃ (coins : Nat → Nat),
    (coins 5 + coins 20 + coins 15) = 6 ∧
    (5 * coins 5 + 20 * coins 20 + 15 * coins 15 = 50) :=
sorry

end coin_difference_l1821_182145


namespace trivia_team_total_members_l1821_182104

theorem trivia_team_total_members (x : ℕ) (h1 : 4 ≤ x) (h2 : (x - 4) * 8 = 64) : x = 12 :=
sorry

end trivia_team_total_members_l1821_182104


namespace seed_selection_valid_l1821_182119

def seeds : List Nat := [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07]

def extractValidSeeds (lst : List Nat) (startIndex : Nat) (maxValue : Nat) (count : Nat) : List Nat :=
  lst.drop startIndex
  |>.filter (fun n => n < maxValue)
  |>.take count

theorem seed_selection_valid :
  extractValidSeeds seeds 10 850 4 = [169, 555, 671, 105] :=
by
  sorry

end seed_selection_valid_l1821_182119


namespace smallest_number_of_students_l1821_182165

theorem smallest_number_of_students
    (g11 g10 g9 : Nat)
    (h_ratio1 : 4 * g9 = 3 * g11)
    (h_ratio2 : 6 * g10 = 5 * g11) :
  g11 + g10 + g9 = 31 :=
sorry

end smallest_number_of_students_l1821_182165


namespace cricket_target_runs_l1821_182105

theorem cricket_target_runs 
  (run_rate1 : ℝ) (run_rate2 : ℝ) (overs : ℕ)
  (h1 : run_rate1 = 5.4) (h2 : run_rate2 = 10.6) (h3 : overs = 25) :
  (run_rate1 * overs + run_rate2 * overs = 400) :=
by sorry

end cricket_target_runs_l1821_182105


namespace safe_zone_inequality_l1821_182181

theorem safe_zone_inequality (x : ℝ) (fuse_burn_rate : ℝ) (run_speed : ℝ) (safe_zone_dist : ℝ) (H1: fuse_burn_rate = 0.5) (H2: run_speed = 4) (H3: safe_zone_dist = 150) :
  run_speed * (x / fuse_burn_rate) ≥ safe_zone_dist :=
sorry

end safe_zone_inequality_l1821_182181


namespace region_in_plane_l1821_182169

def f (x : ℝ) : ℝ := x^2 - 6 * x + 5

theorem region_in_plane (x y : ℝ) :
  (f x + f y ≤ 0) ∧ (f x - f y ≥ 0) ↔
  ((x - 3)^2 + (y - 3)^2 ≤ 8) ∧ ((x ≥ y ∧ x + y ≥ 6) ∨ (x ≤ y ∧ x + y ≤ 6)) :=
by
  sorry

end region_in_plane_l1821_182169


namespace find_f_zero_l1821_182158

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_zero (h : ∀ x : ℝ, x ≠ 0 → f (2 * x - 1) = (1 - x^2) / x^2) : f 0 = 3 :=
sorry

end find_f_zero_l1821_182158


namespace arithmetic_sequence_common_difference_l1821_182110

theorem arithmetic_sequence_common_difference :
  let a := 5
  let a_n := 50
  let S_n := 330
  exists (d n : ℤ), (a + (n - 1) * d = a_n) ∧ (n * (a + a_n) / 2 = S_n) ∧ (d = 45 / 11) :=
by
  let a := 5
  let a_n := 50
  let S_n := 330
  use 45 / 11, 12
  sorry

end arithmetic_sequence_common_difference_l1821_182110


namespace triangle_side_ratio_sqrt2_l1821_182101

variables (A B C A1 B1 C1 X Y : Point)
variable (triangle : IsAcuteAngledTriangle A B C)
variable (altitudes : AreAltitudes A B C A1 B1 C1)
variable (midpoints : X = Midpoint A C1 ∧ Y = Midpoint A1 C)
variable (equality : Distance X Y = Distance B B1)

theorem triangle_side_ratio_sqrt2 :
  ∃ (AC AB : ℝ), (AC / AB = Real.sqrt 2) := sorry

end triangle_side_ratio_sqrt2_l1821_182101


namespace parabola_tangent_line_l1821_182193

noncomputable def verify_a_value (a : ℝ) : Prop :=
  ∃ x₀ y₀ : ℝ, (y₀ = a * x₀^2) ∧ (x₀ - y₀ - 1 = 0) ∧ (2 * a * x₀ = 1)

theorem parabola_tangent_line :
  verify_a_value (1 / 4) :=
by
  sorry

end parabola_tangent_line_l1821_182193


namespace range_is_fixed_points_l1821_182117

variable (f : ℕ → ℕ)

axiom functional_eq : ∀ m n, f (m + f n) = f (f m) + f n

theorem range_is_fixed_points :
  {n : ℕ | ∃ m : ℕ, f m = n} = {n : ℕ | f n = n} :=
sorry

end range_is_fixed_points_l1821_182117


namespace ways_to_place_7_balls_in_3_boxes_l1821_182127

theorem ways_to_place_7_balls_in_3_boxes : ∃ n : ℕ, n = 8 ∧ (∀ x y z : ℕ, x + y + z = 7 → x ≥ y → y ≥ z → z ≥ 0) := 
by
  sorry

end ways_to_place_7_balls_in_3_boxes_l1821_182127


namespace choose_stick_l1821_182133

-- Define the lengths of the sticks Xiaoming has
def xm_stick1 : ℝ := 4
def xm_stick2 : ℝ := 7

-- Define the lengths of the sticks Xiaohong has
def stick2 : ℝ := 2
def stick3 : ℝ := 3
def stick8 : ℝ := 8
def stick12 : ℝ := 12

-- Define the condition for a valid stick choice from Xiaohong's sticks
def valid_stick (x : ℝ) : Prop := 3 < x ∧ x < 11

-- State the problem as a theorem to be proved
theorem choose_stick : valid_stick stick8 := by
  sorry

end choose_stick_l1821_182133


namespace eval_fraction_l1821_182121

theorem eval_fraction : (144 : ℕ) = 12 * 12 → (12 ^ 10 / (144 ^ 4) : ℝ) = 144 := by
  intro h
  have h1 : (144 : ℕ) = 12 ^ 2 := by
    exact h
  sorry

end eval_fraction_l1821_182121


namespace place_value_ratio_56439_2071_l1821_182124

theorem place_value_ratio_56439_2071 :
  let n := 56439.2071
  let digit_6_place_value := 1000
  let digit_2_place_value := 0.1
  digit_6_place_value / digit_2_place_value = 10000 :=
by
  sorry

end place_value_ratio_56439_2071_l1821_182124


namespace least_amount_of_money_l1821_182142

variable (money : String → ℝ)
variable (Bo Coe Flo Jo Moe Zoe : String)

theorem least_amount_of_money :
  (money Bo ≠ money Coe) ∧ (money Bo ≠ money Flo) ∧ (money Bo ≠ money Jo) ∧ (money Bo ≠ money Moe) ∧ (money Bo ≠ money Zoe) ∧ 
  (money Coe ≠ money Flo) ∧ (money Coe ≠ money Jo) ∧ (money Coe ≠ money Moe) ∧ (money Coe ≠ money Zoe) ∧ 
  (money Flo ≠ money Jo) ∧ (money Flo ≠ money Moe) ∧ (money Flo ≠ money Zoe) ∧ 
  (money Jo ≠ money Moe) ∧ (money Jo ≠ money Zoe) ∧ 
  (money Moe ≠ money Zoe) ∧ 
  (money Flo > money Jo) ∧ (money Flo > money Bo) ∧
  (money Bo > money Moe) ∧ (money Coe > money Moe) ∧ 
  (money Jo > money Moe) ∧ (money Jo < money Bo) ∧ 
  (money Zoe > money Jo) ∧ (money Zoe < money Coe) →
  money Moe < money Bo ∧ money Moe < money Coe ∧ money Moe < money Flo ∧ money Moe < money Jo ∧ money Moe < money Zoe := 
sorry

end least_amount_of_money_l1821_182142


namespace inequality_inequality_l1821_182168

theorem inequality_inequality (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) :
  (a - b) ^ 2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧
  (a + b) / 2 - Real.sqrt (a * b) < (a - b) ^ 2 / (8 * b) :=
sorry

end inequality_inequality_l1821_182168


namespace earliest_time_100_degrees_l1821_182161

def temperature (t : ℝ) : ℝ := -t^2 + 15 * t + 40

theorem earliest_time_100_degrees :
  ∃ t : ℝ, temperature t = 100 ∧ (∀ t' : ℝ, temperature t' = 100 → t' ≥ t) :=
by
  sorry

end earliest_time_100_degrees_l1821_182161


namespace extreme_values_of_f_a_zero_on_interval_monotonicity_of_f_l1821_182123

-- Define the function f(x) = 2*x^3 + 3*(a-2)*x^2 - 12*a*x
def f (x : ℝ) (a : ℝ) := 2*x^3 + 3*(a-2)*x^2 - 12*a*x

-- Define the function f(x) when a = 0
def f_a_zero (x : ℝ) := f x 0

-- Define the intervals and extreme values problem
theorem extreme_values_of_f_a_zero_on_interval :
  ∃ (max min : ℝ), (∀ x ∈ Set.Icc (-2 : ℝ) 4, f_a_zero x ≤ max ∧ f_a_zero x ≥ min) ∧ max = 32 ∧ min = -40 :=
sorry

-- Define the function for the derivative of f(x)
def f_derivative (x : ℝ) (a : ℝ) := 6*x^2 + 6*(a-2)*x - 12*a

-- Prove the monotonicity based on the value of a
theorem monotonicity_of_f (a : ℝ) :
  (a > -2 → (∀ x, x < -a → f_derivative x a > 0) ∧ (∀ x, -a < x ∧ x < 2 → f_derivative x a < 0) ∧ (∀ x, x > 2 → f_derivative x a > 0)) ∧
  (a = -2 → ∀ x, f_derivative x a ≥ 0) ∧
  (a < -2 → (∀ x, x < 2 → f_derivative x a > 0) ∧ (∀ x, 2 < x ∧ x < -a → f_derivative x a < 0) ∧ (∀ x, x > -a → f_derivative x a > 0)) :=
sorry

end extreme_values_of_f_a_zero_on_interval_monotonicity_of_f_l1821_182123


namespace carly_dogs_total_l1821_182187

theorem carly_dogs_total (total_nails : ℕ) (three_legged_dogs : ℕ) (nails_per_paw : ℕ) (total_dogs : ℕ) 
  (h1 : total_nails = 164) (h2 : three_legged_dogs = 3) (h3 : nails_per_paw = 4) : total_dogs = 11 :=
by
  sorry

end carly_dogs_total_l1821_182187


namespace boxes_needed_l1821_182114

theorem boxes_needed (balls : ℕ) (balls_per_box : ℕ) (h1 : balls = 10) (h2 : balls_per_box = 5) : balls / balls_per_box = 2 := by
  sorry

end boxes_needed_l1821_182114


namespace correct_statement_is_A_l1821_182151

theorem correct_statement_is_A : 
  (∀ x : ℝ, 0 ≤ x → abs x = x) ∧
  ¬ (∀ x : ℝ, x ≤ 0 → -x = x) ∧
  ¬ (∀ x : ℝ, (x ≠ 0 ∧ x⁻¹ = x) → (x = 1 ∨ x = -1 ∨ x = 0)) ∧
  ¬ (∀ x y : ℝ, x < 0 ∧ y < 0 → abs x < abs y → x < y) :=
by
  sorry

end correct_statement_is_A_l1821_182151


namespace min_unplowed_cells_l1821_182135

theorem min_unplowed_cells (n k : ℕ) (hn : n > 0) (hk : k > 0) (hnk : n > k) :
  ∃ M : ℕ, M = (n - k)^2 := by
  sorry

end min_unplowed_cells_l1821_182135


namespace golden_section_PB_l1821_182125

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem golden_section_PB {A B P : ℝ} (h1 : P = (1 - 1/(golden_ratio)) * A + (1/(golden_ratio)) * B)
  (h2 : AB = 2)
  (h3 : A ≠ B) : PB = 3 - Real.sqrt 5 :=
by
  sorry

end golden_section_PB_l1821_182125


namespace possible_value_is_121_l1821_182134

theorem possible_value_is_121
  (x a y z b : ℕ) 
  (hx : x = 1 / 6 * a) 
  (hz : z = 1 / 6 * b) 
  (hy : y = (a + b) % 5) 
  (h_single_digit : ∀ n, n ∈ [x, a, y, z, b] → n < 10 ∧ 0 < n) : 
  100 * x + 10 * y + z = 121 :=
by
  sorry

end possible_value_is_121_l1821_182134


namespace pickles_per_cucumber_l1821_182149

theorem pickles_per_cucumber (jars cucumbers vinegar_initial vinegar_left pickles_per_jar vinegar_per_jar total_pickles_per_cucumber : ℕ) 
    (h1 : jars = 4) 
    (h2 : cucumbers = 10) 
    (h3 : vinegar_initial = 100) 
    (h4 : vinegar_left = 60) 
    (h5 : pickles_per_jar = 12) 
    (h6 : vinegar_per_jar = 10) 
    (h7 : total_pickles_per_cucumber = 4): 
    total_pickles_per_cucumber = (vinegar_initial - vinegar_left) / vinegar_per_jar * pickles_per_jar / cucumbers := 
by 
  sorry

end pickles_per_cucumber_l1821_182149


namespace part1_part2_part3_l1821_182173

def is_beautiful_point (x y : ℝ) (a b : ℝ) : Prop :=
  a = -x ∧ b = x - y

def beautiful_points (x y : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let a := -x
  let b := x - y
  ((a, b), (b, a))

theorem part1 (x y : ℝ) (h : (x, y) = (4, 1)) :
  beautiful_points x y = ((-4, 3), (3, -4)) := by
  sorry

theorem part2 (x y : ℝ) (h : x = 2) (h' : (-x = 2 - y)) :
  y = 4 := by
  sorry

theorem part3 (x y : ℝ) (h : ((-x, x-y) = (-2, 7)) ∨ ((x-y, -x) = (-2, 7))) :
  (x = 2 ∧ y = -5) ∨ (x = -7 ∧ y = -5) := by
  sorry

end part1_part2_part3_l1821_182173


namespace algebra_geometry_probabilities_l1821_182197

theorem algebra_geometry_probabilities :
  let total := 5
  let algebra := 2
  let geometry := 3
  let prob_first_algebra := algebra / total
  let prob_second_geometry_after_algebra := geometry / (total - 1)
  let prob_both := prob_first_algebra * prob_second_geometry_after_algebra
  let total_after_first_algebra := total - 1
  let remaining_geometry := geometry
  prob_both = 3 / 10 ∧ remaining_geometry / total_after_first_algebra = 3 / 4 :=
by
  sorry

end algebra_geometry_probabilities_l1821_182197


namespace jessies_initial_weight_l1821_182192

-- Definitions based on the conditions
def weight_lost : ℕ := 126
def current_weight : ℕ := 66

-- The statement to prove
theorem jessies_initial_weight :
  (weight_lost + current_weight = 192) :=
by 
  sorry

end jessies_initial_weight_l1821_182192


namespace product_increased_l1821_182132

theorem product_increased (a b c : ℕ) (h1 : a = 1) (h2: b = 1) (h3: c = 676) :
  ((a - 3) * (b - 3) * (c - 3) = a * b * c + 2016) :=
by
  simp [h1, h2, h3]
  sorry

end product_increased_l1821_182132


namespace line_parallel_l1821_182102

theorem line_parallel (a : ℝ) : (∀ x y : ℝ, ax + y = 0) ↔ (x + ay + 1 = 0) → a = 1 ∨ a = -1 := 
sorry

end line_parallel_l1821_182102


namespace intersection_count_l1821_182129

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem intersection_count (ω φ : ℝ) (hω : ω > 0) (hφ : |φ| < Real.pi / 2) 
  (h_max : ∀ x, f x ω φ ≤ f (Real.pi / 6) ω φ)
  (h_period : ∀ x, f x ω φ = f (x + 2 * Real.pi / ω) ω φ) :
  ∃! x : ℝ, f x ω φ = -x + 2 * Real.pi / 3 :=
sorry

end intersection_count_l1821_182129


namespace problem_statement_l1821_182141

variables {c c' d d' : ℝ}

theorem problem_statement (hc : c ≠ 0) (hc' : c' ≠ 0)
  (h : (-d) / (2 * c) = 2 * ((-d') / (3 * c'))) :
  (d / (2 * c)) = 2 * (d' / (3 * c')) :=
by
  sorry

end problem_statement_l1821_182141


namespace sin_cos_identity_l1821_182189

theorem sin_cos_identity (θ : ℝ) (h : Real.tan (θ + (Real.pi / 4)) = 2) : 
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = -7/5 := 
by 
  sorry

end sin_cos_identity_l1821_182189


namespace soap_box_height_l1821_182126

theorem soap_box_height
  (carton_length carton_width carton_height : ℕ)
  (soap_length soap_width h : ℕ)
  (max_soap_boxes : ℕ)
  (h_carton_dim : carton_length = 30)
  (h_carton_width : carton_width = 42)
  (h_carton_height : carton_height = 60)
  (h_soap_length : soap_length = 7)
  (h_soap_width : soap_width = 6)
  (h_max_soap_boxes : max_soap_boxes = 360) :
  h = 1 :=
by
  sorry

end soap_box_height_l1821_182126


namespace sum_of_consecutive_evens_l1821_182186

theorem sum_of_consecutive_evens (E1 E2 E3 E4 : ℕ) (h1 : E4 = 38) (h2 : E3 = E4 - 2) (h3 : E2 = E3 - 2) (h4 : E1 = E2 - 2) : 
  E1 + E2 + E3 + E4 = 140 := 
by 
  sorry

end sum_of_consecutive_evens_l1821_182186


namespace circumcircle_eq_l1821_182146

noncomputable def A : (ℝ × ℝ) := (0, 0)
noncomputable def B : (ℝ × ℝ) := (4, 0)
noncomputable def C : (ℝ × ℝ) := (0, 6)

theorem circumcircle_eq :
  ∃ h k r, h = 2 ∧ k = 3 ∧ r = 13 ∧ (∀ x y, ((x - h)^2 + (y - k)^2 = r) ↔ (x - 2)^2 + (y - 3)^2 = 13) := sorry

end circumcircle_eq_l1821_182146


namespace curve_to_polar_l1821_182144

noncomputable def polar_eq_of_curve (x y : ℝ) (ρ θ : ℝ) : Prop :=
  (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) ∧ (x ^ 2 + y ^ 2 - 2 * x = 0) → (ρ = 2 * Real.cos θ)

theorem curve_to_polar (x y ρ θ : ℝ) :
  polar_eq_of_curve x y ρ θ :=
sorry

end curve_to_polar_l1821_182144


namespace two_f_of_x_l1821_182137

noncomputable def f (x : ℝ) : ℝ := 3 / (3 + x)

theorem two_f_of_x (x : ℝ) (h : x > 0) : 2 * f x = 18 / (9 + x) :=
  sorry

end two_f_of_x_l1821_182137


namespace time_distribution_l1821_182160

noncomputable def total_hours_at_work (hours_task1 day : ℕ) (hours_task2 day : ℕ) (work_days : ℕ) (reduce_per_week : ℕ) : ℕ :=
  (hours_task1 + hours_task2) * work_days

theorem time_distribution (h1 : 5 = 5) (h2 : 3 = 3) (days : 5 = 5) (reduction : 5 = 5) :
  total_hours_at_work 5 3 5 5 = 40 :=
by
  sorry

end time_distribution_l1821_182160


namespace determine_angle_A_l1821_182112

noncomputable section

open Real

-- Definition of an acute triangle and its sides
variables {A B : ℝ} {a b : ℝ}

-- Additional conditions that are given before providing the theorem
variables (h1 : 0 < A) (h2 : A < π / 2) (h3 : 0 < B) (h4 : B < π / 2)
          (h5 : 2 * a * sin B = sqrt 3 * b)

-- Theorem statement
theorem determine_angle_A (h1 : 0 < A) (h2 : A < π / 2) (h3 : 0 < B) (h4 : B < π / 2)
  (h5 : 2 * a * sin B = sqrt 3 * b) : A = π / 3 :=
sorry

end determine_angle_A_l1821_182112


namespace proof_of_inequality_l1821_182108

theorem proof_of_inequality (a b : ℝ) (h : a > b) : a - 3 > b - 3 :=
sorry

end proof_of_inequality_l1821_182108


namespace algebraic_expression_value_l1821_182190

theorem algebraic_expression_value (x y : ℝ) (h : x^2 - 4 * x - 1 = 0) : 
  (2 * x - 3) ^ 2 - (x + y) * (x - y) - y ^ 2 = 12 := 
by {
  sorry
}

end algebraic_expression_value_l1821_182190


namespace cost_of_flute_l1821_182131

def total_spent : ℝ := 158.35
def music_stand_cost : ℝ := 8.89
def song_book_cost : ℝ := 7
def flute_cost : ℝ := 142.46

theorem cost_of_flute :
  total_spent - (music_stand_cost + song_book_cost) = flute_cost :=
by
  sorry

end cost_of_flute_l1821_182131


namespace veg_eaters_l1821_182174

variable (n_veg_only n_both : ℕ)

theorem veg_eaters
  (h1 : n_veg_only = 15)
  (h2 : n_both = 11) :
  n_veg_only + n_both = 26 :=
by sorry

end veg_eaters_l1821_182174


namespace initial_total_fish_l1821_182107

def total_days (weeks : ℕ) : ℕ := weeks * 7
def fish_added (rate : ℕ) (days : ℕ) : ℕ := rate * days
def initial_fish (final_count : ℕ) (added : ℕ) : ℕ := final_count - added

theorem initial_total_fish {final_goldfish final_koi rate_goldfish rate_koi days init_goldfish init_koi : ℕ}
    (h_final_goldfish : final_goldfish = 200)
    (h_final_koi : final_koi = 227)
    (h_rate_goldfish : rate_goldfish = 5)
    (h_rate_koi : rate_koi = 2)
    (h_days : days = total_days 3)
    (h_init_goldfish : init_goldfish = initial_fish final_goldfish (fish_added rate_goldfish days))
    (h_init_koi : init_koi = initial_fish final_koi (fish_added rate_koi days)) :
    init_goldfish + init_koi = 280 :=
by
    sorry -- skipping the proof

end initial_total_fish_l1821_182107
