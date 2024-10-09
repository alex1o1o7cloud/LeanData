import Mathlib

namespace root_equation_identity_l2381_238160

theorem root_equation_identity {a b c p q : ℝ} 
  (h1 : a^2 + p*a + 1 = 0)
  (h2 : b^2 + p*b + 1 = 0)
  (h3 : b^2 + q*b + 2 = 0)
  (h4 : c^2 + q*c + 2 = 0) 
  : (b - a) * (b - c) = p*q - 6 := 
sorry

end root_equation_identity_l2381_238160


namespace top_layer_blocks_l2381_238147

theorem top_layer_blocks (x : Nat) (h : x + 3 * x + 9 * x + 27 * x = 40) : x = 1 :=
by
  sorry

end top_layer_blocks_l2381_238147


namespace extreme_values_f_range_of_a_l2381_238138

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 - x - a
noncomputable def df (x : ℝ) : ℝ := 3 * x^2 - 2 * x - 1

theorem extreme_values_f (a : ℝ) :
  ∃ (x₁ x₂ : ℝ), df x₁ = 0 ∧ df x₂ = 0 ∧ f x₁ a = (5 / 27) - a ∧ f x₂ a = -1 - a :=
sorry

theorem range_of_a (a : ℝ) :
  (∃ (a : ℝ), f (-1/3) a < 0 ∧ f 1 a > 0) ↔ (a < -1 ∨ a > 5 / 27) :=
sorry

end extreme_values_f_range_of_a_l2381_238138


namespace prime_even_intersection_l2381_238114

-- Define P as the set of prime numbers
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def P : Set ℕ := { n | is_prime n }

-- Define Q as the set of even numbers
def Q : Set ℕ := { n | n % 2 = 0 }

-- Statement to prove
theorem prime_even_intersection : P ∩ Q = {2} :=
by
  sorry

end prime_even_intersection_l2381_238114


namespace total_sections_after_admissions_l2381_238155

theorem total_sections_after_admissions (S : ℕ) (h1 : (S * 24 + 24 = (S + 3) * 21)) :
  (S + 3) = 16 :=
  sorry

end total_sections_after_admissions_l2381_238155


namespace monotonicity_and_max_of_f_g_range_of_a_l2381_238183

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x^2

noncomputable def g (x a : ℝ) : ℝ := x * Real.exp x - (a - 1) * x^2 - x - 2 * Real.log x

theorem monotonicity_and_max_of_f : 
  (∀ x, 0 < x → x < 1 → f x > f (x + 1)) ∧ 
  (∀ x, x > 1 → f x < f (x - 1)) ∧ 
  (f 1 = -1) := 
by
  sorry

theorem g_range_of_a (a : ℝ) : 
  (∀ x, x > 0 → f x + g x a ≥ 0) → (a ≤ 1) := 
by
  sorry

end monotonicity_and_max_of_f_g_range_of_a_l2381_238183


namespace number_of_newborn_members_in_group_l2381_238162

noncomputable def N : ℝ :=
  let p_death := 1 / 10
  let p_survive := 1 - p_death
  let prob_survive_3_months := p_survive * p_survive * p_survive
  218.7 / prob_survive_3_months

theorem number_of_newborn_members_in_group : N = 300 := by
  sorry

end number_of_newborn_members_in_group_l2381_238162


namespace price_of_horse_and_cow_l2381_238156

theorem price_of_horse_and_cow (x y : ℝ) (h1 : 4 * x + 6 * y = 48) (h2 : 3 * x + 5 * y = 38) :
  (4 * x + 6 * y = 48) ∧ (3 * x + 5 * y = 38) := 
by
  exact ⟨h1, h2⟩

end price_of_horse_and_cow_l2381_238156


namespace mike_total_hours_worked_l2381_238149

-- Define the conditions
def time_to_wash_car := 10
def time_to_change_oil := 15
def time_to_change_tires := 30

def number_of_cars_washed := 9
def number_of_oil_changes := 6
def number_of_tire_changes := 2

-- Define the conversion factor
def minutes_per_hour := 60

-- Prove that the total time worked equals 4 hours
theorem mike_total_hours_worked : 
  (number_of_cars_washed * time_to_wash_car + 
   number_of_oil_changes * time_to_change_oil + 
   number_of_tire_changes * time_to_change_tires) / minutes_per_hour = 4 := by
  sorry

end mike_total_hours_worked_l2381_238149


namespace jina_mascots_l2381_238122

variables (x y z x_new Total : ℕ)

def mascots_problem :=
  (y = 3 * x) ∧
  (x_new = x + 2 * y) ∧
  (z = 2 * y) ∧
  (Total = x_new + y + z) →
  Total = 16 * x

-- The statement only, no proof is required
theorem jina_mascots : mascots_problem x y z x_new Total := sorry

end jina_mascots_l2381_238122


namespace total_fertilizer_used_l2381_238102

def daily_fertilizer := 3
def num_days := 12
def extra_final_day := 6

theorem total_fertilizer_used : 
    (daily_fertilizer * num_days + (daily_fertilizer + extra_final_day)) = 45 :=
by
  sorry

end total_fertilizer_used_l2381_238102


namespace gecko_third_day_crickets_l2381_238182

def total_crickets : ℕ := 70
def first_day_percentage : ℝ := 0.30
def first_day_crickets : ℝ := first_day_percentage * total_crickets
def second_day_crickets : ℝ := first_day_crickets - 6
def third_day_crickets : ℝ := total_crickets - (first_day_crickets + second_day_crickets)

theorem gecko_third_day_crickets :
  third_day_crickets = 34 :=
by
  sorry

end gecko_third_day_crickets_l2381_238182


namespace shortest_segment_length_l2381_238121

theorem shortest_segment_length :
  let total_length := 1
  let red_dot := 0.618
  let yellow_dot := total_length - red_dot  -- yellow_dot is at the same point after fold
  let first_cut := red_dot  -- Cut the strip at the red dot
  let remaining_strip := red_dot
  let distance_between_red_and_yellow := total_length - 2 * yellow_dot
  let second_cut := distance_between_red_and_yellow
  let shortest_segment := remaining_strip - 2 * distance_between_red_and_yellow
  shortest_segment = 0.146 :=
by
  sorry

end shortest_segment_length_l2381_238121


namespace oxen_b_is_12_l2381_238178

variable (oxen_b : ℕ)

def share (oxen months : ℕ) : ℕ := oxen * months

def total_share (oxen_a oxen_b oxen_c months_a months_b months_c : ℕ) : ℕ :=
  share oxen_a months_a + share oxen_b months_b + share oxen_c months_c

def proportion (rent_c rent total_share_c total_share : ℕ) : Prop :=
  rent_c * total_share = rent * total_share_c

theorem oxen_b_is_12 : oxen_b = 12 := by
  let oxen_a := 10
  let oxen_c := 15
  let months_a := 7
  let months_b := 5
  let months_c := 3
  let rent := 210
  let rent_c := 54
  let share_a := share oxen_a months_a
  let share_c := share oxen_c months_c
  let total_share_val := total_share oxen_a oxen_b oxen_c months_a months_b months_c
  let total_rent := share_a + 5 * oxen_b + share_c
  have h1 : proportion rent_c rent share_c total_rent := by sorry
  rw [proportion] at h1
  sorry

end oxen_b_is_12_l2381_238178


namespace factorize_3a_squared_minus_6a_plus_3_l2381_238140

theorem factorize_3a_squared_minus_6a_plus_3 (a : ℝ) : 
  3 * a^2 - 6 * a + 3 = 3 * (a - 1)^2 :=
by 
  sorry

end factorize_3a_squared_minus_6a_plus_3_l2381_238140


namespace max_value_expression_l2381_238110

open Real

theorem max_value_expression (x : ℝ) : 
  ∃ (y : ℝ), y ≤ (x^6 / (x^10 + 3 * x^8 - 5 * x^6 + 10 * x^4 + 25)) ∧
  y = 1 / (5 + 2 * sqrt 30) :=
sorry

end max_value_expression_l2381_238110


namespace hem_dress_time_l2381_238131

theorem hem_dress_time
  (hem_length_feet : ℕ)
  (stitch_length_inches : ℝ)
  (stitches_per_minute : ℕ)
  (hem_length_inches : ℝ)
  (total_stitches : ℕ)
  (time_minutes : ℝ)
  (h1 : hem_length_feet = 3)
  (h2 : stitch_length_inches = 1 / 4)
  (h3 : stitches_per_minute = 24)
  (h4 : hem_length_inches = 12 * hem_length_feet)
  (h5 : total_stitches = hem_length_inches / stitch_length_inches)
  (h6 : time_minutes = total_stitches / stitches_per_minute) :
  time_minutes = 6 := 
sorry

end hem_dress_time_l2381_238131


namespace expression_not_defined_l2381_238128

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : ℝ := x^2 - 25*x + 125

-- Theorem statement that the expression is not defined for specific values of x
theorem expression_not_defined (x : ℝ) : quadratic_eq x = 0 ↔ (x = 5 ∨ x = 20) :=
by
  sorry

end expression_not_defined_l2381_238128


namespace unoccupied_volume_proof_l2381_238175

-- Definitions based on conditions
def tank_length : ℕ := 12
def tank_width : ℕ := 10
def tank_height : ℕ := 8
def tank_volume : ℕ := tank_length * tank_width * tank_height

def oil_fill_ratio : ℚ := 2 / 3
def ice_cube_volume : ℕ := 1
def number_of_ice_cubes : ℕ := 15

-- Volume calculations
def oil_volume : ℚ := oil_fill_ratio * tank_volume
def total_ice_volume : ℚ := number_of_ice_cubes * ice_cube_volume
def occupied_volume : ℚ := oil_volume + total_ice_volume

-- The final question to be proved
theorem unoccupied_volume_proof : tank_volume - occupied_volume = 305 := by
  sorry

end unoccupied_volume_proof_l2381_238175


namespace problem1_problem2_l2381_238118

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

theorem problem1 :
  f 1 + f 2 + f 3 + f (1 / 2) + f (1 / 3) = 5 / 2 :=
by
  sorry

theorem problem2 : ∀ x : ℝ, 0 < f x ∧ f x ≤ 1 :=
by
  intro x
  sorry

end problem1_problem2_l2381_238118


namespace find_m_l2381_238135

theorem find_m (m x1 x2 : ℝ) (h1 : x1^2 + m * x1 + 5 = 0) (h2 : x2^2 + m * x2 + 5 = 0) (h3 : x1 = 2 * |x2| - 3) : 
  m = -9 / 2 :=
sorry

end find_m_l2381_238135


namespace area_of_ABC_l2381_238126

noncomputable def area_of_triangle (AB AC angleB : ℝ) : ℝ :=
  0.5 * AB * AC * Real.sin angleB

theorem area_of_ABC :
  area_of_triangle 5 3 (120 * Real.pi / 180) = (15 * Real.sqrt 3) / 4 :=
by
  sorry

end area_of_ABC_l2381_238126


namespace pay_for_notebook_with_change_l2381_238197

theorem pay_for_notebook_with_change : ∃ (a b : ℤ), 16 * a - 27 * b = 1 :=
by
  sorry

end pay_for_notebook_with_change_l2381_238197


namespace warehouse_length_l2381_238170

theorem warehouse_length (L W : ℕ) (times supposed_times : ℕ) (total_distance : ℕ)
  (h1 : W = 400)
  (h2 : supposed_times = 10)
  (h3 : times = supposed_times - 2)
  (h4 : total_distance = times * (2 * L + 2 * W))
  (h5 : total_distance = 16000) :
  L = 600 := by
  sorry

end warehouse_length_l2381_238170


namespace no_integer_roots_of_polynomial_l2381_238142

theorem no_integer_roots_of_polynomial :
  ¬ ∃ (x : ℤ), x^3 - 3 * x^2 - 10 * x + 20 = 0 :=
by
  sorry

end no_integer_roots_of_polynomial_l2381_238142


namespace problem1_solution_problem2_solution_l2381_238158

-- Problem 1: Prove the solution set for the given inequality
theorem problem1_solution (x : ℝ) : (2 < x ∧ x ≤ (7 / 2)) ↔ ((x + 1) / (x - 2) ≥ 3) := 
sorry

-- Problem 2: Prove the solution set for the given inequality
theorem problem2_solution (x a : ℝ) : 
  (a = 0 ∧ x = 0) ∨ 
  (a > 0 ∧ -a ≤ x ∧ x ≤ 2 * a) ∨ 
  (a < 0 ∧ 2 * a ≤ x ∧ x ≤ -a) ↔ 
  x^2 - a * x - 2 * a^2 ≤ 0 := 
sorry

end problem1_solution_problem2_solution_l2381_238158


namespace exists_real_root_in_interval_l2381_238151

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 3

theorem exists_real_root_in_interval (f : ℝ → ℝ)
  (h_mono : ∀ x y, x < y → f x < f y)
  (h1 : f 1 < 0)
  (h2 : f 2 > 0) : 
  ∃ x, 1 < x ∧ x < 2 ∧ f x = 0 := 
sorry

end exists_real_root_in_interval_l2381_238151


namespace ratio_of_ducks_l2381_238166

theorem ratio_of_ducks (lily_ducks lily_geese rayden_geese rayden_ducks : ℕ) 
  (h1 : lily_ducks = 20) 
  (h2 : lily_geese = 10) 
  (h3 : rayden_geese = 4 * lily_geese) 
  (h4 : rayden_ducks + rayden_geese = lily_ducks + lily_geese + 70) : 
  rayden_ducks / lily_ducks = 3 :=
by
  sorry

end ratio_of_ducks_l2381_238166


namespace first_hour_rain_l2381_238165

variable (x : ℝ)
variable (rain_1st_hour : ℝ) (rain_2nd_hour : ℝ)
variable (total_rain : ℝ)

-- Define the conditions
def condition_1 (x rain_2nd_hour : ℝ) : Prop :=
  rain_2nd_hour = 2 * x + 7

def condition_2 (x rain_2nd_hour total_rain : ℝ) : Prop :=
  x + rain_2nd_hour = total_rain

-- Prove the amount of rain in the first hour
theorem first_hour_rain (h1 : condition_1 x rain_2nd_hour)
                         (h2 : condition_2 x rain_2nd_hour total_rain)
                         (total_rain_is_22 : total_rain = 22) :
  x = 5 :=
by
  -- Proof steps go here
  sorry

end first_hour_rain_l2381_238165


namespace students_left_l2381_238161

theorem students_left (initial_students new_students final_students students_left : ℕ)
  (h1 : initial_students = 10)
  (h2 : new_students = 42)
  (h3 : final_students = 48)
  : initial_students + new_students - students_left = final_students → students_left = 4 :=
by
  intros
  sorry

end students_left_l2381_238161


namespace cups_of_flour_per_pound_of_pasta_l2381_238173

-- Definitions from conditions
def pounds_of_pasta_per_rack : ℕ := 3
def racks_owned : ℕ := 3
def additional_rack_needed : ℕ := 1
def cups_per_bag : ℕ := 8
def bags_used : ℕ := 3

-- Derived definitions from above conditions
def total_cups_of_flour : ℕ := bags_used * cups_per_bag  -- 24 cups
def total_racks_needed : ℕ := racks_owned + additional_rack_needed  -- 4 racks
def total_pounds_of_pasta : ℕ := total_racks_needed * pounds_of_pasta_per_rack  -- 12 pounds

theorem cups_of_flour_per_pound_of_pasta (x : ℕ) :
  (total_cups_of_flour / total_pounds_of_pasta) = x → x = 2 :=
by
  intro h
  sorry

end cups_of_flour_per_pound_of_pasta_l2381_238173


namespace general_formula_sum_formula_l2381_238186

-- Define the geometric sequence
def geoseq (n : ℕ) : ℕ := 2^n

-- Define the sum of the first n terms of the geometric sequence
def sum_first_n_terms (n : ℕ) : ℕ := 2^(n+1) - 2

-- Given conditions
def a1 : ℕ := 2
def a4 : ℕ := 16

-- Theorem statements
theorem general_formula (n : ℕ) : 
  (geoseq 1 = a1) → (geoseq 4 = a4) → geoseq n = 2^n := sorry

theorem sum_formula (n : ℕ) : 
  (geoseq 1 = a1) → (geoseq 4 = a4) → sum_first_n_terms n = 2^(n+1) - 2 := sorry

end general_formula_sum_formula_l2381_238186


namespace gas_cost_l2381_238164

theorem gas_cost (x : ℝ) (h₁ : 5 * (x / 5 - 9) = 8 * (x / 8)) : x = 120 :=
by
  sorry

end gas_cost_l2381_238164


namespace ratio_of_perimeters_of_squares_l2381_238187

theorem ratio_of_perimeters_of_squares (d1 d11 : ℝ) (s1 s11 : ℝ) (P1 P11 : ℝ) 
  (h1 : d11 = 11 * d1)
  (h2 : d1 = s1 * Real.sqrt 2)
  (h3 : d11 = s11 * Real.sqrt 2) :
  P11 / P1 = 11 :=
by
  sorry

end ratio_of_perimeters_of_squares_l2381_238187


namespace mean_score_l2381_238180

theorem mean_score (M SD : ℝ) (h₁ : 58 = M - 2 * SD) (h₂ : 98 = M + 3 * SD) : M = 74 :=
by
  sorry

end mean_score_l2381_238180


namespace total_flour_correct_l2381_238101

-- Define the quantities specified in the conditions
def cups_of_flour_already_added : ℕ := 2
def cups_of_flour_to_add : ℕ := 7

-- Define the total cups of flour required by the recipe as a sum of the quantities
def cups_of_flour_required : ℕ := cups_of_flour_already_added + cups_of_flour_to_add

-- Prove that the total cups of flour required is 9
theorem total_flour_correct : cups_of_flour_required = 9 := by
  -- use auto proof placeholder
  rfl

end total_flour_correct_l2381_238101


namespace rubles_exchange_l2381_238132

theorem rubles_exchange (x : ℕ) : 
  (3000 * x - 7000 = 2950 * x) → x = 140 := by
  sorry

end rubles_exchange_l2381_238132


namespace find_cost_price_l2381_238185

variable (CP : ℝ) -- cost price
variable (SP_loss SP_gain : ℝ) -- selling prices

-- Conditions
def loss_condition := SP_loss = 0.9 * CP
def gain_condition := SP_gain = 1.04 * CP
def difference_condition := SP_gain - SP_loss = 190

-- Theorem to prove
theorem find_cost_price (h_loss : loss_condition CP SP_loss)
                        (h_gain : gain_condition CP SP_gain)
                        (h_diff : difference_condition SP_loss SP_gain) :
  CP = 1357.14 := 
sorry

end find_cost_price_l2381_238185


namespace hotel_people_per_room_l2381_238100

theorem hotel_people_per_room
  (total_rooms : ℕ := 10)
  (towels_per_person : ℕ := 2)
  (total_towels : ℕ := 60) :
  (total_towels / towels_per_person) / total_rooms = 3 :=
by
  sorry

end hotel_people_per_room_l2381_238100


namespace fn_conjecture_l2381_238109

theorem fn_conjecture (f : ℕ → ℝ → ℝ) (x : ℝ) (h_pos : x > 0) :
  (f 1 x = x / (Real.sqrt (1 + x^2))) →
  (∀ n, f (n + 1) x = f 1 (f n x)) →
  (∀ n, f n x = x / (Real.sqrt (1 + n * x ^ 2))) := by
  sorry

end fn_conjecture_l2381_238109


namespace A_subscribed_fraction_l2381_238181

theorem A_subscribed_fraction 
  (total_profit : ℝ) (A_share : ℝ) 
  (B_fraction : ℝ) (C_fraction : ℝ) 
  (A_fraction : ℝ) :
  total_profit = 2430 →
  A_share = 810 →
  B_fraction = 1/4 →
  C_fraction = 1/5 →
  A_fraction = A_share / total_profit →
  A_fraction = 1/3 :=
by
  intros h_total_profit h_A_share h_B_fraction h_C_fraction h_A_fraction
  sorry

end A_subscribed_fraction_l2381_238181


namespace set_of_x_satisfying_2f_less_than_x_plus_1_l2381_238113

theorem set_of_x_satisfying_2f_less_than_x_plus_1 (f : ℝ → ℝ) 
  (h1 : f 1 = 1) 
  (h2 : ∀ x : ℝ, deriv f x > 1 / 2) :
  { x : ℝ | 2 * f x < x + 1 } = { x : ℝ | x < 1 } :=
by
  sorry

end set_of_x_satisfying_2f_less_than_x_plus_1_l2381_238113


namespace maximum_value_a1_l2381_238191

noncomputable def max_possible_value (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0)
  (h3 : a 1 = a 10) : ℝ :=
  16

theorem maximum_value_a1 (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0)
  (h3 : a 1 = a 10) : a 1 ≤ max_possible_value a h1 h2 h3 :=
  sorry

end maximum_value_a1_l2381_238191


namespace decreasing_interval_f_l2381_238144

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

theorem decreasing_interval_f :
  ∀ x : ℝ, x > 0 → (f (x) = (1 / 2) * x^2 - Real.log x) →
  (∃ a b : ℝ, 0 < a ∧ a ≤ b ∧ b = 1 ∧ ∀ y, a < y ∧ y ≤ b → f (y) ≤ f (y+1)) := sorry

end decreasing_interval_f_l2381_238144


namespace polygon_sides_l2381_238119

theorem polygon_sides (n : ℕ) (h : 144 * n = 180 * (n - 2)) : n = 10 :=
by { sorry }

end polygon_sides_l2381_238119


namespace arithmetic_progression_contains_sixth_power_l2381_238108

theorem arithmetic_progression_contains_sixth_power
  (a h : ℕ) (a_pos : 0 < a) (h_pos : 0 < h)
  (sq : ∃ n : ℕ, a + n * h = k^2)
  (cube : ∃ m : ℕ, a + m * h = l^3) :
  ∃ p : ℕ, ∃ q : ℕ, a + q * h = p^6 := sorry

end arithmetic_progression_contains_sixth_power_l2381_238108


namespace find_k_l2381_238148

noncomputable def vec_a : ℝ × ℝ := (1, 2)
noncomputable def vec_b : ℝ × ℝ := (-3, 2)
noncomputable def vec_k_a_plus_b (k : ℝ) : ℝ × ℝ := (k - 3, 2 * k + 2)
noncomputable def vec_a_minus_3b : ℝ × ℝ := (10, -4)

theorem find_k :
  ∃! k : ℝ, (vec_k_a_plus_b k).1 * vec_a_minus_3b.2 = (vec_k_a_plus_b k).2 * vec_a_minus_3b.1 ∧ k = -1 / 3 :=
by
  sorry

end find_k_l2381_238148


namespace monotonically_increasing_interval_l2381_238146

def f (x : ℝ) (a : ℝ) : ℝ := |2 * x + a| + 3

theorem monotonically_increasing_interval (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ a ≤ f x₂ a) → a ≥ -2 :=
by
  sorry

end monotonically_increasing_interval_l2381_238146


namespace imaginary_part_of_complex_l2381_238199

theorem imaginary_part_of_complex (z : ℂ) (h : (1 - I) * z = I) : z.im = 1 / 2 :=
sorry

end imaginary_part_of_complex_l2381_238199


namespace pow_divisible_by_13_l2381_238157

theorem pow_divisible_by_13 (n : ℕ) (h : 0 < n) : (4^(2*n+1) + 3^(n+2)) % 13 = 0 :=
sorry

end pow_divisible_by_13_l2381_238157


namespace alma_score_l2381_238167

variables (A M S : ℕ)

-- Given conditions
axiom h1 : M = 60
axiom h2 : M = 3 * A
axiom h3 : A + M = 2 * S

theorem alma_score : S = 40 :=
by
  -- proof goes here
  sorry

end alma_score_l2381_238167


namespace a_minus_two_sufficient_but_not_necessary_for_pure_imaginary_l2381_238174

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

def complex_from_a (a : ℝ) : ℂ :=
  (a^2 - 4 : ℝ) + (a + 1 : ℝ) * Complex.I

theorem a_minus_two_sufficient_but_not_necessary_for_pure_imaginary :
  (is_pure_imaginary (complex_from_a (-2))) ∧ ¬ (∀ (a : ℝ), is_pure_imaginary (complex_from_a a) → a = -2) :=
by
  sorry

end a_minus_two_sufficient_but_not_necessary_for_pure_imaginary_l2381_238174


namespace sum_of_midpoints_l2381_238198

variable (a b c : ℝ)

def sum_of_vertices := a + b + c

theorem sum_of_midpoints (h : sum_of_vertices a b c = 15) :
  (a + b)/2 + (a + c)/2 + (b + c)/2 = 15 :=
by
  sorry

end sum_of_midpoints_l2381_238198


namespace pet_preferences_l2381_238154

/-- A store has several types of pets: 20 puppies, 10 kittens, 8 hamsters, and 5 birds.
Alice, Bob, Charlie, and David each want a different kind of pet, with the following preferences:
- Alice does not want a bird.
- Bob does not want a hamster.
- Charlie does not want a kitten.
- David does not want a puppy.
Prove that the number of ways they can choose different types of pets satisfying
their preferences is 791440. -/
theorem pet_preferences :
  let P := 20    -- Number of puppies
  let K := 10    -- Number of kittens
  let H := 8     -- Number of hamsters
  let B := 5     -- Number of birds
  let Alice_options := P + K + H -- Alice does not want a bird
  let Bob_options := P + K + B   -- Bob does not want a hamster
  let Charlie_options := P + H + B -- Charlie does not want a kitten
  let David_options := K + H + B   -- David does not want a puppy
  let Alice_pick := Alice_options
  let Bob_pick := Bob_options - 1
  let Charlie_pick := Charlie_options - 2
  let David_pick := David_options - 3
  Alice_pick * Bob_pick * Charlie_pick * David_pick = 791440 :=
by
  sorry

end pet_preferences_l2381_238154


namespace sqrt_c_is_202_l2381_238190

theorem sqrt_c_is_202 (a b c : ℝ) (h1 : a + b = -2020) (h2 : a * b = c) (h3 : a / b + b / a = 98) : 
  Real.sqrt c = 202 :=
by
  sorry

end sqrt_c_is_202_l2381_238190


namespace vec_a_squared_minus_vec_b_squared_l2381_238124

variable (a b : ℝ × ℝ)
variable (h1 : a + b = (-3, 6))
variable (h2 : a - b = (-3, 2))

theorem vec_a_squared_minus_vec_b_squared : (a.1 * a.1 + a.2 * a.2) - (b.1 * b.1 + b.2 * b.2) = 32 :=
sorry

end vec_a_squared_minus_vec_b_squared_l2381_238124


namespace correct_option_B_l2381_238125

variable {a b x y : ℤ}

def option_A (a : ℤ) : Prop := -a - a = 0
def option_B (x y : ℤ) : Prop := -(x + y) = -x - y
def option_C (b a : ℤ) : Prop := 3 * (b - 2 * a) = 3 * b - 2 * a
def option_D (a : ℤ) : Prop := 8 * a^4 - 6 * a^2 = 2 * a^2

theorem correct_option_B (x y : ℤ) : option_B x y := by
  -- The proof would go here
  sorry

end correct_option_B_l2381_238125


namespace quadratic_solution_1_quadratic_solution_2_l2381_238130

theorem quadratic_solution_1 (x : ℝ) :
  x^2 + 3 * x - 1 = 0 ↔ (x = (-3 + Real.sqrt 13) / 2) ∨ (x = (-3 - Real.sqrt 13) / 2) :=
by
  sorry

theorem quadratic_solution_2 (x : ℝ) :
  (x - 2)^2 = 2 * (x - 2) ↔ (x = 2) ∨ (x = 4) :=
by
  sorry

end quadratic_solution_1_quadratic_solution_2_l2381_238130


namespace circle_equation_through_intersections_l2381_238103

theorem circle_equation_through_intersections 
  (h₁ : ∀ x y : ℝ, x^2 + y^2 + 6 * x - 4 = 0 ↔ x^2 + y^2 + 6 * y - 28 = 0)
  (h₂ : ∀ x y : ℝ, x - y - 4 = 0) : 
  ∃ x y : ℝ, (x - 1/2) ^ 2 + (y + 7 / 2) ^ 2 = 89 / 2 :=
by sorry

end circle_equation_through_intersections_l2381_238103


namespace no_mult_of_5_end_in_2_l2381_238194

theorem no_mult_of_5_end_in_2 (n : ℕ) : n < 500 → ∃ k, n = 5 * k → (n % 10 = 2) = false :=
by
  sorry

end no_mult_of_5_end_in_2_l2381_238194


namespace total_clothes_washed_l2381_238117

def number_of_clothing_items (Cally Danny Emily shared_socks : ℕ) : ℕ :=
  Cally + Danny + Emily + shared_socks

theorem total_clothes_washed :
  let Cally_clothes := (10 + 5 + 7 + 6 + 3)
  let Danny_clothes := (6 + 8 + 10 + 6 + 4)
  let Emily_clothes := (8 + 6 + 9 + 5 + 2)
  let shared_socks := (3 + 2)
  number_of_clothing_items Cally_clothes Danny_clothes Emily_clothes shared_socks = 100 :=
by
  sorry

end total_clothes_washed_l2381_238117


namespace no_integer_solutions_l2381_238184

theorem no_integer_solutions (x y z : ℤ) (h : 2 * x^4 + 2 * x^2 * y^2 + y^4 = z^2) (hx : x ≠ 0) : false :=
sorry

end no_integer_solutions_l2381_238184


namespace geometric_sequence_alpha_5_l2381_238115

theorem geometric_sequence_alpha_5 (α : ℕ → ℝ) (h1 : α 4 * α 5 * α 6 = 27) (h2 : α 4 * α 6 = (α 5) ^ 2) : α 5 = 3 := 
sorry

end geometric_sequence_alpha_5_l2381_238115


namespace value_of_expression_l2381_238192

variable (p q r s : ℝ)

-- Given condition in a)
def polynomial_function (x : ℝ) := p * x^3 + q * x^2 + r * x + s
def passes_through_point := polynomial_function p q r s (-1) = 4

-- Proof statement in c)
theorem value_of_expression (h : passes_through_point p q r s) : 6 * p - 3 * q + r - 2 * s = -24 := by
  sorry

end value_of_expression_l2381_238192


namespace volume_of_cuboid_l2381_238196

theorem volume_of_cuboid (l w h : ℝ) (hl_pos : 0 < l) (hw_pos : 0 < w) (hh_pos : 0 < h) 
  (h1 : l * w = 120) (h2 : w * h = 72) (h3 : h * l = 60) : l * w * h = 4320 :=
by
  sorry

end volume_of_cuboid_l2381_238196


namespace beavers_working_l2381_238129

theorem beavers_working (a b : ℝ) (h₁ : a = 2.0) (h₂ : b = 1.0) : a + b = 3.0 := 
by 
  rw [h₁, h₂]
  norm_num

end beavers_working_l2381_238129


namespace range_of_t_l2381_238188

theorem range_of_t 
  (k t : ℝ)
  (tangent_condition : (t + 1)^2 = 1 + k^2)
  (intersect_condition : ∃ x y, y = k * x + t ∧ y = x^2 / 4) : 
  t > 0 ∨ t < -3 :=
sorry

end range_of_t_l2381_238188


namespace perimeter_of_regular_polygon_l2381_238163

/-- 
Given a regular polygon with a central angle of 45 degrees and a side length of 5,
the perimeter of the polygon is 40.
-/
theorem perimeter_of_regular_polygon 
  (central_angle : ℝ) (side_length : ℝ) (h1 : central_angle = 45)
  (h2 : side_length = 5) :
  ∃ P, P = 40 :=
by
  sorry

end perimeter_of_regular_polygon_l2381_238163


namespace birds_on_fence_total_l2381_238193

variable (initial_birds : ℕ) (additional_birds : ℕ)

theorem birds_on_fence_total {initial_birds additional_birds : ℕ} (h1 : initial_birds = 4) (h2 : additional_birds = 6) :
    initial_birds + additional_birds = 10 :=
  by
  sorry

end birds_on_fence_total_l2381_238193


namespace intersection_of_A_and_B_l2381_238141

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 4}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} :=
by
  sorry

end intersection_of_A_and_B_l2381_238141


namespace percent_women_non_union_employees_is_65_l2381_238137

-- Definitions based on the conditions
variables {E : ℝ} -- Denoting the total number of employees as a real number

def percent_men (E : ℝ) : ℝ := 0.56 * E
def percent_union_employees (E : ℝ) : ℝ := 0.60 * E
def percent_non_union_employees (E : ℝ) : ℝ := 0.40 * E
def percent_women_non_union (percent_non_union_employees : ℝ) : ℝ := 0.65 * percent_non_union_employees

-- Theorem statement
theorem percent_women_non_union_employees_is_65 :
  percent_women_non_union (percent_non_union_employees E) / (percent_non_union_employees E) = 0.65 :=
by
  sorry

end percent_women_non_union_employees_is_65_l2381_238137


namespace cost_per_millisecond_l2381_238106

theorem cost_per_millisecond
  (C : ℝ)
  (h1 : 1.07 + (C * 1500) + 5.35 = 40.92) :
  C = 0.023 :=
sorry

end cost_per_millisecond_l2381_238106


namespace trajectory_of_P_l2381_238150

-- Define points P, A, and B in a 2D plane
variable {P A B : EuclideanSpace ℝ (Fin 2)}

-- Define the condition that the sum of the distances from P to A and P to B equals the distance between A and B
def sum_of_distances_condition (P A B : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist P A + dist P B = dist A B

-- Main theorem statement: If P satisfies the above condition, then P lies on the line segment AB
theorem trajectory_of_P (P A B : EuclideanSpace ℝ (Fin 2)) (h : sum_of_distances_condition P A B) :
    ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = t • A + (1 - t) • B :=
  sorry

end trajectory_of_P_l2381_238150


namespace combined_profit_is_14000_l2381_238133

-- Define constants and conditions
def center1_daily_packages : ℕ := 10000
def daily_profit_per_package : ℝ := 0.05
def center2_multiplier : ℕ := 3
def days_per_week : ℕ := 7

-- Define the profit for the first center
def center1_daily_profit : ℝ := center1_daily_packages * daily_profit_per_package

-- Define the packages processed by the second center
def center2_daily_packages : ℕ := center1_daily_packages * center2_multiplier

-- Define the profit for the second center
def center2_daily_profit : ℝ := center2_daily_packages * daily_profit_per_package

-- Define the combined daily profit
def combined_daily_profit : ℝ := center1_daily_profit + center2_daily_profit

-- Define the combined weekly profit
def combined_weekly_profit : ℝ := combined_daily_profit * days_per_week

-- Prove that the combined weekly profit is $14,000
theorem combined_profit_is_14000 : combined_weekly_profit = 14000 := by
  -- You can replace sorry with the steps to solve the proof.
  sorry

end combined_profit_is_14000_l2381_238133


namespace value_of_y_l2381_238189

theorem value_of_y (x y : ℤ) (h1 : x^2 = y - 2) (h2 : x = -6) : y = 38 :=
by
  sorry

end value_of_y_l2381_238189


namespace students_per_table_l2381_238104

theorem students_per_table (total_students tables students_bathroom students_canteen added_students exchange_students : ℕ) 
  (h1 : total_students = 47)
  (h2 : tables = 6)
  (h3 : students_bathroom = 3)
  (h4 : students_canteen = 3 * students_bathroom)
  (h5 : added_students = 2 * 4)
  (h6 : exchange_students = 3 + 3 + 3) :
  (total_students - (students_bathroom + students_canteen + added_students + exchange_students)) / tables = 3 := 
by 
  sorry

end students_per_table_l2381_238104


namespace simplify_fraction_l2381_238145

theorem simplify_fraction :
  ((3^2008)^2 - (3^2006)^2) / ((3^2007)^2 - (3^2005)^2) = 9 :=
by
  sorry

end simplify_fraction_l2381_238145


namespace ways_to_draw_at_least_two_defective_l2381_238153

-- Definitions based on the conditions of the problem
def total_products : ℕ := 100
def defective_products : ℕ := 3
def selected_products : ℕ := 5

-- Binomial coefficient function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- The theorem to prove
theorem ways_to_draw_at_least_two_defective :
  C defective_products 2 * C (total_products - defective_products) 3 + C defective_products 3 * C (total_products - defective_products) 2 =
  (C total_products selected_products - C defective_products 1 * C (total_products - defective_products) 4) :=
sorry

end ways_to_draw_at_least_two_defective_l2381_238153


namespace total_pairs_of_jeans_purchased_l2381_238111

-- Definitions based on the problem conditions
def price_fox : ℝ := 15
def price_pony : ℝ := 18
def discount_save : ℝ := 8.64
def pairs_fox : ℕ := 3
def pairs_pony : ℕ := 2
def sum_discount_rate : ℝ := 0.22
def discount_rate_pony : ℝ := 0.13999999999999993

-- Lean 4 statement to prove the total number of pairs of jeans purchased
theorem total_pairs_of_jeans_purchased :
  pairs_fox + pairs_pony = 5 :=
by
  sorry

end total_pairs_of_jeans_purchased_l2381_238111


namespace tan_product_l2381_238116

theorem tan_product :
    (Real.tan (Real.pi / 9)) * (Real.tan (2 * Real.pi / 9)) * (Real.tan (4 * Real.pi / 9)) = 1 / 3 :=
by sorry

end tan_product_l2381_238116


namespace answer_key_combinations_l2381_238105

theorem answer_key_combinations : 
  (2^3 - 2) * 4^2 = 96 := 
by 
  -- Explanation about why it equals to this multi-step skipped, directly written as sorry.
  sorry

end answer_key_combinations_l2381_238105


namespace time_saved_calculator_l2381_238120

-- Define the conditions
def time_with_calculator (n : ℕ) : ℕ := 2 * n
def time_without_calculator (n : ℕ) : ℕ := 5 * n
def total_problems : ℕ := 20

-- State the theorem to prove the time saved is 60 minutes
theorem time_saved_calculator : 
  time_without_calculator total_problems - time_with_calculator total_problems = 60 :=
sorry

end time_saved_calculator_l2381_238120


namespace solution_set_of_inequality_l2381_238139

open Set

theorem solution_set_of_inequality :
  {x : ℝ | - x ^ 2 - 4 * x + 5 > 0} = {x : ℝ | -5 < x ∧ x < 1} :=
sorry

end solution_set_of_inequality_l2381_238139


namespace best_fit_model_l2381_238172

-- Definition of the given R^2 values for different models
def R2_A : ℝ := 0.62
def R2_B : ℝ := 0.63
def R2_C : ℝ := 0.68
def R2_D : ℝ := 0.65

-- Theorem statement that model with R2_C has the best fitting effect
theorem best_fit_model : R2_C = max R2_A (max R2_B (max R2_C R2_D)) :=
by
  sorry -- Proof is not required

end best_fit_model_l2381_238172


namespace cake_slices_l2381_238134

open Nat

theorem cake_slices (S : ℕ) (h1 : 2 * S - 12 = 10) : S = 8 := by
  sorry

end cake_slices_l2381_238134


namespace digit_difference_one_l2381_238176

theorem digit_difference_one (p q : ℕ) (h_pq : p < 10 ∧ q < 10) (h_diff : (10 * p + q) - (10 * q + p) = 9) :
  p - q = 1 :=
by
  sorry

end digit_difference_one_l2381_238176


namespace solution_set_of_inequality_l2381_238179

theorem solution_set_of_inequality (x : ℝ) (h : 3 * x + 2 > 5) : x > 1 :=
sorry

end solution_set_of_inequality_l2381_238179


namespace find_f_neg3_l2381_238168

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
  if h : x > 0 then x * (1 - x) else -x * (1 + x)

theorem find_f_neg3 :
  is_odd_function f →
  (∀ x, x > 0 → f x = x * (1 - x)) →
  f (-3) = 6 :=
by
  intros h_odd h_condition
  sorry

end find_f_neg3_l2381_238168


namespace at_least_six_consecutive_heads_l2381_238143

noncomputable def flip_probability : ℚ :=
  let total_outcomes := 2^8
  let successful_outcomes := 7
  successful_outcomes / total_outcomes

theorem at_least_six_consecutive_heads : 
  flip_probability = 7 / 256 :=
by
  sorry

end at_least_six_consecutive_heads_l2381_238143


namespace fence_poles_count_l2381_238107

def length_path : ℕ := 900
def length_bridge : ℕ := 42
def distance_between_poles : ℕ := 6

theorem fence_poles_count :
  2 * (length_path - length_bridge) / distance_between_poles = 286 :=
by
  sorry

end fence_poles_count_l2381_238107


namespace kevin_hopping_distance_l2381_238112

theorem kevin_hopping_distance :
  let hop_distance (n : Nat) : ℚ :=
    let factor : ℚ := (3/4 : ℚ)^n
    1/4 * factor
  let total_distance : ℚ :=
    (hop_distance 0 + hop_distance 1 + hop_distance 2 + hop_distance 3 + hop_distance 4 + hop_distance 5)
  total_distance = 39677 / 40960 :=
by
  sorry

end kevin_hopping_distance_l2381_238112


namespace markup_correct_l2381_238159

theorem markup_correct (purchase_price : ℝ) (overhead_percentage : ℝ) (net_profit : ℝ) :
  purchase_price = 48 → overhead_percentage = 0.15 → net_profit = 12 →
  (purchase_price * (1 + overhead_percentage) + net_profit - purchase_price) = 19.2 :=
by
  intros
  sorry

end markup_correct_l2381_238159


namespace triangle_is_isosceles_if_median_bisects_perimeter_l2381_238195

-- Defining the sides of the triangle
variables {a b c : ℝ}

-- Defining the median condition
def median_bisects_perimeter (a b c : ℝ) : Prop :=
  a + b + c = 2 * (a/2 + b)

-- The main theorem stating that the triangle is isosceles if the median bisects the perimeter
theorem triangle_is_isosceles_if_median_bisects_perimeter (a b c : ℝ) 
  (h : median_bisects_perimeter a b c) : b = c :=
by
  sorry

end triangle_is_isosceles_if_median_bisects_perimeter_l2381_238195


namespace brand_z_percentage_correct_l2381_238127

noncomputable def percentage_of_brand_z (capacity : ℝ := 1) (brand_z1 : ℝ := 1) (brand_x1 : ℝ := 0) 
(brand_z2 : ℝ := 1/4) (brand_x2 : ℝ := 3/4) (brand_z3 : ℝ := 5/8) (brand_x3 : ℝ := 3/8) 
(brand_z4 : ℝ := 5/16) (brand_x4 : ℝ := 11/16) : ℝ :=
    (brand_z4 / (brand_z4 + brand_x4)) * 100

theorem brand_z_percentage_correct : percentage_of_brand_z = 31.25 := by
  sorry

end brand_z_percentage_correct_l2381_238127


namespace man_older_than_son_l2381_238171

theorem man_older_than_son (S M : ℕ) (h1 : S = 23) (h2 : M + 2 = 2 * (S + 2)) : M - S = 25 :=
by
  sorry

end man_older_than_son_l2381_238171


namespace soda_preference_l2381_238152

theorem soda_preference (total_surveyed : ℕ) (angle_soda_sector : ℕ) (h_total_surveyed : total_surveyed = 540) (h_angle_soda_sector : angle_soda_sector = 270) :
  let fraction_soda := angle_soda_sector / 360
  let people_soda := fraction_soda * total_surveyed
  people_soda = 405 :=
by
  sorry

end soda_preference_l2381_238152


namespace integer_solutions_of_quadratic_eq_l2381_238169

theorem integer_solutions_of_quadratic_eq (b : ℤ) :
  ∃ p q : ℤ, (p+9) * (q+9) = 81 ∧ p + q = -b ∧ p * q = 9*b :=
sorry

end integer_solutions_of_quadratic_eq_l2381_238169


namespace ratio_of_distances_l2381_238136

-- Definitions based on conditions in a)
variables (x y w : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_w : 0 ≤ w)
variables (h_w_ne_zero : w ≠ 0) (h_y_ne_zero : y ≠ 0) (h_eq_times : y / w = x / w + (x + y) / (9 * w))

-- The proof statement
theorem ratio_of_distances (x y w : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y)
  (h_nonneg_w : 0 ≤ w) (h_w_ne_zero : w ≠ 0) (h_y_ne_zero : y ≠ 0)
  (h_eq_times : y / w = x / w + (x + y) / (9 * w)) :
  x / y = 4 / 5 :=
sorry

end ratio_of_distances_l2381_238136


namespace Buffy_whiskers_is_40_l2381_238123

def number_of_whiskers (Puffy Scruffy Buffy Juniper : ℕ) : Prop :=
  Puffy = 3 * Juniper ∧
  Puffy = Scruffy / 2 ∧
  Buffy = (Puffy + Scruffy + Juniper) / 3 ∧
  Juniper = 12

theorem Buffy_whiskers_is_40 :
  ∃ (Puffy Scruffy Buffy Juniper : ℕ), 
    number_of_whiskers Puffy Scruffy Buffy Juniper ∧ Buffy = 40 := 
by
  sorry

end Buffy_whiskers_is_40_l2381_238123


namespace shopkeeper_packets_l2381_238177

noncomputable def milk_packets (oz_to_ml: ℝ) (ml_per_packet: ℝ) (total_milk_oz: ℝ) : ℝ :=
  (total_milk_oz * oz_to_ml) / ml_per_packet

theorem shopkeeper_packets (oz_to_ml: ℝ) (ml_per_packet: ℝ) (total_milk_oz: ℝ) :
  oz_to_ml = 30 → ml_per_packet = 250 → total_milk_oz = 1250 → milk_packets oz_to_ml ml_per_packet total_milk_oz = 150 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end shopkeeper_packets_l2381_238177
