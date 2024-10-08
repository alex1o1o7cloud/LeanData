import Mathlib

namespace regular_polygon_perimeter_l230_230371

theorem regular_polygon_perimeter (s : ℝ) (n : ℕ) (h1 : n = 4) (h2 : s = 7) : 
  4 * s = 28 :=
by
  sorry

end regular_polygon_perimeter_l230_230371


namespace number_added_is_10_l230_230760

-- Define the conditions.
def number_thought_of : ℕ := 55
def result : ℕ := 21

-- Define the statement of the problem.
theorem number_added_is_10 : ∃ (y : ℕ), (number_thought_of / 5 + y = result) ∧ (y = 10) := by
  sorry

end number_added_is_10_l230_230760


namespace handshake_even_acquaintance_l230_230940

theorem handshake_even_acquaintance (n : ℕ) (hn : n = 225) : 
  ∃ (k : ℕ), k < n ∧ (∀ m < n, k ≠ m) :=
by sorry

end handshake_even_acquaintance_l230_230940


namespace candies_per_house_l230_230988

theorem candies_per_house (candies_per_block : ℕ) (houses_per_block : ℕ) 
  (h1 : candies_per_block = 35) (h2 : houses_per_block = 5) :
  candies_per_block / houses_per_block = 7 := by
  sorry

end candies_per_house_l230_230988


namespace arithmetic_sequence_1005th_term_l230_230255

theorem arithmetic_sequence_1005th_term (p r : ℤ) 
  (h1 : 11 = p + 2 * r)
  (h2 : 11 + 2 * r = 4 * p - r) :
  (5 + 1004 * 6) = 6029 :=
by
  sorry

end arithmetic_sequence_1005th_term_l230_230255


namespace how_many_peaches_l230_230736

-- Define the variables
variables (Jake Steven : ℕ)

-- Conditions
def has_fewer_peaches : Prop := Jake = Steven - 7
def jake_has_9_peaches : Prop := Jake = 9

-- The theorem that proves Steven's number of peaches
theorem how_many_peaches (Jake Steven : ℕ) (h1 : has_fewer_peaches Jake Steven) (h2 : jake_has_9_peaches Jake) : Steven = 16 :=
by
  -- Proof goes here
  sorry

end how_many_peaches_l230_230736


namespace greatest_of_given_numbers_l230_230109

-- Defining the given conditions
def a := 1000 + 0.01
def b := 1000 * 0.01
def c := 1000 / 0.01
def d := 0.01 / 1000
def e := 1000 - 0.01

-- Prove that c is the greatest
theorem greatest_of_given_numbers : c = max a (max b (max d e)) :=
by
  -- Placeholder for the proof
  sorry

end greatest_of_given_numbers_l230_230109


namespace perpendicular_condition_l230_230312

-- Definitions based on the conditions
def line_l1 (m : ℝ) (x y : ℝ) : Prop := (m + 1) * x + (1 - m) * y - 1 = 0
def line_l2 (m : ℝ) (x y : ℝ) : Prop := (m - 1) * x + (2 * m + 1) * y + 4 = 0

-- Perpendicularity condition based on the definition in conditions
def perpendicular (m : ℝ) : Prop :=
  (m + 1) * (m - 1) + (1 - m) * (2 * m + 1) = 0

-- Sufficient but not necessary condition
def sufficient_but_not_necessary (m : ℝ) : Prop :=
  m = 0

-- Final statement to prove
theorem perpendicular_condition :
  sufficient_but_not_necessary 0 -> perpendicular 0 :=
by
  sorry

end perpendicular_condition_l230_230312


namespace function_satisfy_f1_function_satisfy_f2_l230_230316

noncomputable def f1 (x : ℝ) : ℝ := 2
noncomputable def f2 (x : ℝ) : ℝ := x

theorem function_satisfy_f1 : 
  ∀ x y : ℝ, x > 0 → y > 0 → f1 (x + y) + f1 x * f1 y = f1 (x * y) + f1 x + f1 y :=
by 
  intros x y hx hy
  unfold f1
  sorry

theorem function_satisfy_f2 :
  ∀ x y : ℝ, x > 0 → y > 0 → f2 (x + y) + f2 x * f2 y = f2 (x * y) + f2 x + f2 y :=
by 
  intros x y hx hy
  unfold f2
  sorry

end function_satisfy_f1_function_satisfy_f2_l230_230316


namespace farmer_land_l230_230909

theorem farmer_land (A : ℝ) (h1 : 0.9 * A = A_cleared) (h2 : 0.3 * A_cleared = A_soybeans) 
  (h3 : 0.6 * A_cleared = A_wheat) (h4 : 0.1 * A_cleared = 540) : A = 6000 :=
by
  sorry

end farmer_land_l230_230909


namespace divide_rope_into_parts_l230_230897

theorem divide_rope_into_parts:
  (∀ rope_length : ℝ, rope_length = 5 -> ∀ parts : ℕ, parts = 4 -> (∀ i : ℕ, i < parts -> ((rope_length / parts) = (5 / 4)))) :=
by sorry

end divide_rope_into_parts_l230_230897


namespace problem1_problem2_problem3_l230_230399

theorem problem1 : 2013^2 - 2012 * 2014 = 1 := 
by 
  sorry

variables (m n : ℤ)

theorem problem2 : ((m-n)^6 / (n-m)^4) * (m-n)^3 = (m-n)^5 :=
by 
  sorry

variables (a b c : ℤ)

theorem problem3 : (a - 2*b + 3*c) * (a - 2*b - 3*c) = a^2 - 4*a*b + 4*b^2 - 9*c^2 :=
by 
  sorry

end problem1_problem2_problem3_l230_230399


namespace element_in_set_l230_230856

open Set

noncomputable def A : Set ℝ := { x | x < 2 * Real.sqrt 3 }
def a : ℝ := 2

theorem element_in_set : a ∈ A := by
  sorry

end element_in_set_l230_230856


namespace false_statement_l230_230256

theorem false_statement :
  ¬ (∀ x : ℝ, x^2 + 1 > 3 * x) = (∃ x : ℝ, x^2 + 1 ≤ 3 * x) := sorry

end false_statement_l230_230256


namespace tara_dad_second_year_attendance_l230_230582

theorem tara_dad_second_year_attendance :
  let games_played_per_year := 20
  let attendance_rate := 0.90
  let first_year_games_attended := attendance_rate * games_played_per_year
  let second_year_games_difference := 4
  first_year_games_attended - second_year_games_difference = 14 :=
by
  -- We skip the proof here
  sorry

end tara_dad_second_year_attendance_l230_230582


namespace loaned_books_count_l230_230386

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

end loaned_books_count_l230_230386


namespace carl_speed_l230_230449

theorem carl_speed 
  (time : ℝ) (distance : ℝ) 
  (h_time : time = 5) 
  (h_distance : distance = 10) 
  : (distance / time) = 2 :=
by
  rw [h_time, h_distance]
  sorry

end carl_speed_l230_230449


namespace sampling_method_sequential_is_systematic_l230_230443

def is_sequential_ids (ids : List Nat) : Prop :=
  ids = [5, 10, 15, 20, 25, 30, 35, 40]

def is_systematic_sampling (sampling_method : Prop) : Prop :=
  sampling_method

theorem sampling_method_sequential_is_systematic :
  ∀ ids, is_sequential_ids ids → 
    is_systematic_sampling (ids = [5, 10, 15, 20, 25, 30, 35, 40]) :=
by
  intros
  apply id
  sorry

end sampling_method_sequential_is_systematic_l230_230443


namespace problem_inequality_l230_230305

theorem problem_inequality {a : ℝ} (h : ∀ x : ℝ, (x - a) * (1 - x - a) < 1) : 
  -1/2 < a ∧ a < 3/2 := by
  sorry

end problem_inequality_l230_230305


namespace marias_workday_ends_at_3_30_pm_l230_230622
open Nat

theorem marias_workday_ends_at_3_30_pm :
  let start_time := (7 : Nat)
  let lunch_start_time := (11 + (30 / 60))
  let work_duration := (8 : Nat)
  let lunch_break := (30 / 60 : Nat)
  let end_time := (15 + (30 / 60) : Nat)
  (start_time + work_duration + lunch_break) - (lunch_start_time - start_time) = end_time := by
  sorry

end marias_workday_ends_at_3_30_pm_l230_230622


namespace shrimp_price_l230_230570

theorem shrimp_price (y : ℝ) (h : 0.6 * (y / 4) = 2.25) : y = 15 :=
sorry

end shrimp_price_l230_230570


namespace lexie_crayons_count_l230_230667

variable (number_of_boxes : ℕ) (crayons_per_box : ℕ)

theorem lexie_crayons_count (h1: number_of_boxes = 10) (h2: crayons_per_box = 8) :
  (number_of_boxes * crayons_per_box) = 80 := by
  sorry

end lexie_crayons_count_l230_230667


namespace capacity_of_bucket_in_first_scenario_l230_230607

theorem capacity_of_bucket_in_first_scenario (x : ℝ) 
  (h1 : 28 * x = 378) : x = 13.5 :=
by
  sorry

end capacity_of_bucket_in_first_scenario_l230_230607


namespace analysis_hours_l230_230224

-- Define the conditions: number of bones and minutes per bone
def number_of_bones : Nat := 206
def minutes_per_bone : Nat := 45

-- Define the conversion factor: minutes per hour
def minutes_per_hour : Nat := 60

-- Define the total minutes spent analyzing all bones
def total_minutes (number_of_bones minutes_per_bone : Nat) : Nat :=
  number_of_bones * minutes_per_bone

-- Define the total hours required for analysis
def total_hours (total_minutes minutes_per_hour : Nat) : Float :=
  total_minutes.toFloat / minutes_per_hour.toFloat

-- Prove that total_hours equals 154.5 hours
theorem analysis_hours : total_hours (total_minutes number_of_bones minutes_per_bone) minutes_per_hour = 154.5 := by
  sorry

end analysis_hours_l230_230224


namespace gnome_voting_l230_230998

theorem gnome_voting (n : ℕ) :
  (∀ g : ℕ, g < n →  
   (g % 3 = 0 → (∃ k : ℕ, k * 4 = n))
   ∧ (n ≠ 0 ∧ (∀ i : ℕ, i < n → (i + 1) % n ≠ (i + 2) % n) → (∃ k : ℕ, k * 4 = n))) := 
sorry

end gnome_voting_l230_230998


namespace symmetric_y_axis_l230_230229

theorem symmetric_y_axis (a b : ℝ) (h₁ : a = -4) (h₂ : b = 3) : a - b = -7 :=
by
  rw [h₁, h₂]
  norm_num

end symmetric_y_axis_l230_230229


namespace value_large_cube_l230_230751

-- Definitions based on conditions
def volume_small := 1 -- volume of one-inch cube in cubic inches
def volume_large := 64 -- volume of four-inch cube in cubic inches
def value_small : ℝ := 1000 -- value of one-inch cube of gold in dollars
def proportion (x y : ℝ) : Prop := y = 64 * x -- proportionality condition

-- Prove that the value of the four-inch cube of gold is $64000
theorem value_large_cube : proportion value_small 64000 := by
  -- Proof skipped
  sorry

end value_large_cube_l230_230751


namespace jessies_initial_weight_l230_230366

-- Definitions based on the conditions
def weight_lost : ℕ := 126
def current_weight : ℕ := 66

-- The statement to prove
theorem jessies_initial_weight :
  (weight_lost + current_weight = 192) :=
by 
  sorry

end jessies_initial_weight_l230_230366


namespace train_speed_km_hr_calc_l230_230062

theorem train_speed_km_hr_calc :
  let length := 175 -- length of the train in meters
  let time := 3.499720022398208 -- time to cross the pole in seconds
  let speed_mps := length / time -- speed in meters per second
  let speed_kmph := speed_mps * 3.6 -- converting speed from m/s to km/hr
  speed_kmph = 180.025923226 := 
sorry

end train_speed_km_hr_calc_l230_230062


namespace exists_two_same_remainder_l230_230464

theorem exists_two_same_remainder (n : ℤ) (a : ℕ → ℤ) :
  ∃ i j : ℕ, i ≠ j ∧ 0 ≤ i ∧ i ≤ n ∧ 0 ≤ j ∧ j ≤ n ∧ (a i % n = a j % n) := sorry

end exists_two_same_remainder_l230_230464


namespace remainder_444_power_444_mod_13_l230_230646

theorem remainder_444_power_444_mod_13 : (444 ^ 444) % 13 = 1 := by
  sorry

end remainder_444_power_444_mod_13_l230_230646


namespace constant_term_zero_l230_230477

theorem constant_term_zero (h1 : x^2 + x = 0)
                          (h2 : 2*x^2 - x - 12 = 0)
                          (h3 : 2*(x^2 - 1) = 3*(x - 1))
                          (h4 : 2*(x^2 + 1) = x + 4) :
                          (∃ (c : ℤ), c = 0 ∧ (c = 0 ∨ c = -12 ∨ c = 1 ∨ c = -2) → c = 0) :=
sorry

end constant_term_zero_l230_230477


namespace total_distance_100_l230_230299

-- Definitions for the problem conditions:
def initial_velocity : ℕ := 40
def common_difference : ℕ := 10
def total_time (v₀ : ℕ) (d : ℕ) : ℕ := (v₀ / d) + 1  -- The total time until the car stops
def distance_traveled (v₀ : ℕ) (d : ℕ) : ℕ :=
  (v₀ * total_time v₀ d) - (d * total_time v₀ d * (total_time v₀ d - 1)) / 2

-- Statement to prove:
theorem total_distance_100 : distance_traveled initial_velocity common_difference = 100 := by
  sorry

end total_distance_100_l230_230299


namespace cat_count_l230_230700

def initial_cats : ℕ := 2
def female_kittens : ℕ := 3
def male_kittens : ℕ := 2
def total_kittens : ℕ := female_kittens + male_kittens
def total_cats : ℕ := initial_cats + total_kittens

theorem cat_count : total_cats = 7 := by
  unfold total_cats
  unfold initial_cats total_kittens
  unfold female_kittens male_kittens
  rfl

end cat_count_l230_230700


namespace monroe_legs_total_l230_230290

def num_spiders : ℕ := 8
def num_ants : ℕ := 12
def legs_per_spider : ℕ := 8
def legs_per_ant : ℕ := 6

theorem monroe_legs_total :
  num_spiders * legs_per_spider + num_ants * legs_per_ant = 136 :=
by
  sorry

end monroe_legs_total_l230_230290


namespace max_blue_points_l230_230150

theorem max_blue_points (n : ℕ) (h_n : n = 2016) :
  ∃ r : ℕ, r * (2016 - r) = 1008 * 1008 :=
by {
  sorry
}

end max_blue_points_l230_230150


namespace f_of_f_3_eq_3_l230_230983

noncomputable def f (x : ℝ) : ℝ :=
if x < 2 then 1 - Real.logb 2 (2 - x) else 2^(1 - x) + 3 / 2

theorem f_of_f_3_eq_3 : f (f 3) = 3 := by
  sorry

end f_of_f_3_eq_3_l230_230983


namespace veg_eaters_l230_230343

variable (n_veg_only n_both : ℕ)

theorem veg_eaters
  (h1 : n_veg_only = 15)
  (h2 : n_both = 11) :
  n_veg_only + n_both = 26 :=
by sorry

end veg_eaters_l230_230343


namespace solve_abs_inequality_l230_230633

theorem solve_abs_inequality :
  { x : ℝ | 3 ≤ |x - 2| ∧ |x - 2| ≤ 6 } = { x : ℝ | -4 ≤ x ∧ x ≤ -1 } ∪ { x : ℝ | 5 ≤ x ∧ x ≤ 8 } :=
sorry

end solve_abs_inequality_l230_230633


namespace fraction_of_menu_l230_230323

def total_dishes (total : ℕ) : Prop := 
  6 = (1/4:ℚ) * total

def vegan_dishes (vegan : ℕ) (soy_free : ℕ) : Prop :=
  vegan = 6 ∧ soy_free = vegan - 5

theorem fraction_of_menu (total vegan soy_free : ℕ) (h1 : total_dishes total)
  (h2 : vegan_dishes vegan soy_free) : (soy_free:ℚ) / total = 1 / 24 := 
by sorry

end fraction_of_menu_l230_230323


namespace problem_equivalence_l230_230997

theorem problem_equivalence :
  (1 / Real.sin (Real.pi / 18) - Real.sqrt 3 / Real.sin (4 * Real.pi / 18)) = 4 := 
sorry

end problem_equivalence_l230_230997


namespace find_value_of_x_squared_plus_inverse_squared_l230_230209

theorem find_value_of_x_squared_plus_inverse_squared (x : ℝ) (hx : x + (1/x) = 2) : x^2 + (1/x^2) = 2 :=
sorry

end find_value_of_x_squared_plus_inverse_squared_l230_230209


namespace simplify_condition_l230_230165

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  Real.sqrt (1 + x) - Real.sqrt (-1 - x)

theorem simplify_condition (x : ℝ) (h1 : 1 + x ≥ 0) (h2 : -1 - x ≥ 0) : simplify_expression x = 0 :=
by
  rw [simplify_expression]
  sorry

end simplify_condition_l230_230165


namespace actual_average_speed_l230_230451

theorem actual_average_speed (v t : ℝ) (h1 : v > 0) (h2: t > 0) (h3 : (t / (t - (1 / 4) * t)) = ((v + 12) / v)) : v = 36 :=
by
  sorry

end actual_average_speed_l230_230451


namespace eggs_not_eaten_per_week_l230_230918

theorem eggs_not_eaten_per_week : 
  let trays_bought := 2
  let eggs_per_tray := 24
  let days_per_week := 7
  let eggs_eaten_by_children_per_day := 2 * 2 -- 2 eggs each by 2 children
  let eggs_eaten_by_parents_per_day := 4
  let total_eggs_eaten_per_week := (eggs_eaten_by_children_per_day + eggs_eaten_by_parents_per_day) * days_per_week
  let total_eggs_bought := trays_bought * eggs_per_tray * 2  -- Re-calculated trays
  total_eggs_bought - total_eggs_eaten_per_week = 40 :=
by
  let trays_bought := 2
  let eggs_per_tray := 24
  let days_per_week := 7
  let eggs_eaten_by_children_per_day := 2 * 2
  let eggs_eaten_by_parents_per_day := 4
  let total_eggs_eaten_per_week := (eggs_eaten_by_children_per_day + eggs_eaten_by_parents_per_day) * days_per_week
  let total_eggs_bought := trays_bought * eggs_per_tray * 2
  show total_eggs_bought - total_eggs_eaten_per_week = 40
  sorry

end eggs_not_eaten_per_week_l230_230918


namespace hyperbola_foci_l230_230104

theorem hyperbola_foci :
  (∀ x y : ℝ, x^2 - 2 * y^2 = 1) →
  (∃ c : ℝ, c = (Real.sqrt 6) / 2 ∧ (x = c ∨ x = -c) ∧ y = 0) :=
by
  sorry

end hyperbola_foci_l230_230104


namespace ratio_of_tshirts_l230_230534

def spending_on_tshirts (Lisa_tshirts Carly_tshirts Lisa_jeans Lisa_coats Carly_jeans Carly_coats : ℝ) : Prop :=
  Lisa_tshirts = 40 ∧
  Lisa_jeans = Lisa_tshirts / 2 ∧
  Lisa_coats = 2 * Lisa_tshirts ∧
  Carly_jeans = 3 * Lisa_jeans ∧
  Carly_coats = Lisa_coats / 4 ∧
  Lisa_tshirts + Lisa_jeans + Lisa_coats + Carly_tshirts + Carly_jeans + Carly_coats = 230

theorem ratio_of_tshirts 
  (Lisa_tshirts Carly_tshirts Lisa_jeans Lisa_coats Carly_jeans Carly_coats : ℝ)
  (h : spending_on_tshirts Lisa_tshirts Carly_tshirts Lisa_jeans Lisa_coats Carly_jeans Carly_coats)
  : Carly_tshirts / Lisa_tshirts = 1 / 4 := 
sorry

end ratio_of_tshirts_l230_230534


namespace sufficient_material_for_box_l230_230596

theorem sufficient_material_for_box :
  ∃ (l w h : ℕ), l * w * h ≥ 1995 ∧ 2 * (l * w + w * h + h * l) ≤ 958 :=
  sorry

end sufficient_material_for_box_l230_230596


namespace weight_of_a_l230_230363

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

end weight_of_a_l230_230363


namespace total_insects_eaten_l230_230154

-- Definitions from the conditions
def numGeckos : Nat := 5
def insectsPerGecko : Nat := 6
def numLizards : Nat := 3
def insectsPerLizard : Nat := insectsPerGecko * 2

-- Theorem statement, proving total insects eaten is 66
theorem total_insects_eaten : numGeckos * insectsPerGecko + numLizards * insectsPerLizard = 66 := by
  sorry

end total_insects_eaten_l230_230154


namespace mod_1237_17_l230_230351

theorem mod_1237_17 : 1237 % 17 = 13 := by
  sorry

end mod_1237_17_l230_230351


namespace no_intersecting_axes_l230_230906

theorem no_intersecting_axes (m : ℝ) : (m^2 + 2 * m - 7 = 0) → m = -4 :=
sorry

end no_intersecting_axes_l230_230906


namespace distance_between_points_l230_230922

theorem distance_between_points :
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  (Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 * Real.sqrt 2) :=
by
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  sorry

end distance_between_points_l230_230922


namespace factor_expression_l230_230948

theorem factor_expression (x : ℝ) : 
  ((4 * x^3 + 64 * x^2 - 8) - (-6 * x^3 + 2 * x^2 - 8)) = 2 * x^2 * (5 * x + 31) := 
by sorry

end factor_expression_l230_230948


namespace axis_of_symmetry_parabola_l230_230311

theorem axis_of_symmetry_parabola : 
  (∃ a b c : ℝ, ∀ x : ℝ, (y = x^2 + 4 * x - 5) ∧ (a = 1) ∧ (b = 4) → ( x = -b / (2 * a) ) → ( x = -2 ) ) :=
by
  sorry

end axis_of_symmetry_parabola_l230_230311


namespace gcf_72_108_l230_230819

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end gcf_72_108_l230_230819


namespace quarter_more_than_whole_l230_230636

theorem quarter_more_than_whole (x : ℝ) (h : x / 4 = 9 + x) : x = -12 :=
by
  sorry

end quarter_more_than_whole_l230_230636


namespace find_n_l230_230324

theorem find_n (n : ℤ) : 43^2 = 1849 ∧ 44^2 = 1936 ∧ 45^2 = 2025 ∧ 46^2 = 2116 ∧ n < Real.sqrt 2023 ∧ Real.sqrt 2023 < n + 1 → n = 44 :=
by
  sorry

end find_n_l230_230324


namespace find_d_l230_230131

theorem find_d (d : ℤ) :
  (∀ x : ℤ, (4 * x^3 + 13 * x^2 + d * x + 18 = 0 ↔ x = -3)) →
  d = 9 :=
by
  sorry

end find_d_l230_230131


namespace verify_triangle_operation_l230_230515

def triangle (a b c : ℕ) : ℕ := a^2 + b^2 + c^2

theorem verify_triangle_operation : triangle 2 3 6 + triangle 1 2 2 = 58 := by
  sorry

end verify_triangle_operation_l230_230515


namespace smallest_positive_x_l230_230505

theorem smallest_positive_x 
  (x : ℝ) 
  (H : 0 < x) 
  (H_eq : ⌊x^2⌋ - x * ⌊x⌋ = 10) : 
  x = 131 / 11 :=
sorry

end smallest_positive_x_l230_230505


namespace determine_a_if_derivative_is_even_l230_230026

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2 + x

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 1

theorem determine_a_if_derivative_is_even (a : ℝ) :
  (∀ x : ℝ, f' x a = f' (-x) a) → a = 0 :=
by
  intros h
  sorry

end determine_a_if_derivative_is_even_l230_230026


namespace problem_number_of_true_propositions_l230_230184

open Set

variable {α : Type*} {A B : Set α}

def card (s : Set α) : ℕ := sorry -- The actual definition of cardinality is complex and in LF (not imperative here).

-- Statement of the problem translated into a Lean statement
theorem problem_number_of_true_propositions :
  (∀ {A B : Set ℕ}, A ∩ B = ∅ ↔ card (A ∪ B) = card A + card B) ∧
  (∀ {A B : Set ℕ}, A ⊆ B → card A ≤ card B) ∧
  (∀ {A B : Set ℕ}, A ⊂ B → card A < card B) →
   (3 = 3) :=
by 
  sorry


end problem_number_of_true_propositions_l230_230184


namespace simple_interest_years_l230_230249

variables (T R : ℝ)

def principal : ℝ := 1000
def additional_interest : ℝ := 90

theorem simple_interest_years
  (H: principal * (R + 3) * T / 100 - principal * R * T / 100 = additional_interest) :
  T = 3 :=
by sorry

end simple_interest_years_l230_230249


namespace coefficients_sum_l230_230565

theorem coefficients_sum:
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ),
  (1+x)^5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 →
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 :=
by
  intros a_0 a_1 a_2 a_3 a_4 a_5 h_eq
  have h0 : a_0 = 1
  sorry -- proof when x=0
  have h1 : a_1 + a_2 + a_3 + a_4 + a_5 = 31
  sorry -- proof when x=1
  exact h1

end coefficients_sum_l230_230565


namespace A_and_B_together_complete_work_in_24_days_l230_230295

-- Define the variables
variables {W_A W_B : ℝ} (completeTime : ℝ → ℝ → ℝ)

-- Define conditions
def A_better_than_B (W_A W_B : ℝ) := W_A = 2 * W_B
def A_takes_36_days (W_A : ℝ) := W_A = 1 / 36

-- The proposition to prove
theorem A_and_B_together_complete_work_in_24_days 
  (h1 : A_better_than_B W_A W_B)
  (h2 : A_takes_36_days W_A) :
  completeTime W_A W_B = 24 :=
sorry

end A_and_B_together_complete_work_in_24_days_l230_230295


namespace cycling_time_difference_l230_230587

-- Definitions from the conditions
def youth_miles : ℤ := 20
def youth_hours : ℤ := 2
def adult_miles : ℤ := 12
def adult_hours : ℤ := 3

-- Conversion from hours to minutes
def hours_to_minutes (hours : ℤ) : ℤ := hours * 60

-- Time per mile calculations
def youth_time_per_mile : ℤ := hours_to_minutes youth_hours / youth_miles
def adult_time_per_mile : ℤ := hours_to_minutes adult_hours / adult_miles

-- The difference in time per mile
def time_difference : ℤ := adult_time_per_mile - youth_time_per_mile

-- Theorem to prove the difference is 9 minutes
theorem cycling_time_difference : time_difference = 9 := by
  -- Proof steps would go here
  sorry

end cycling_time_difference_l230_230587


namespace equation_of_circle_l230_230576

variable (x y : ℝ)

def center_line : ℝ → ℝ := fun x => -4 * x
def tangent_line : ℝ → ℝ := fun x => 1 - x

def P : ℝ × ℝ := (3, -2)
def center_O : ℝ × ℝ := (1, -4)

theorem equation_of_circle :
  (x - 1)^2 + (y + 4)^2 = 8 :=
sorry

end equation_of_circle_l230_230576


namespace vector_addition_l230_230652

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-3, 4)

-- State the problem as a theorem
theorem vector_addition : a + b = (-1, 5) := by
  -- the proof should go here
  sorry

end vector_addition_l230_230652


namespace meaningful_iff_gt_3_l230_230507

section meaningful_expression

variable (a : ℝ)

def is_meaningful (a : ℝ) : Prop :=
  (a > 3)

theorem meaningful_iff_gt_3 : (∃ b, b = (a + 3) / Real.sqrt (a - 3)) ↔ is_meaningful a :=
by
  sorry

end meaningful_expression

end meaningful_iff_gt_3_l230_230507


namespace european_stamp_costs_l230_230422

theorem european_stamp_costs :
  let P_Italy := 0.07
  let P_Germany := 0.03
  let N_Italy := 9
  let N_Germany := 15
  N_Italy * P_Italy + N_Germany * P_Germany = 1.08 :=
by
  sorry

end european_stamp_costs_l230_230422


namespace algebraic_expression_value_l230_230824

theorem algebraic_expression_value (m x n : ℝ)
  (h1 : (m + 3) * x ^ (|m| - 2) + 6 * m = 0)
  (h2 : n * x - 5 = x * (3 - n))
  (h3 : |m| = 2)
  (h4 : (m + 3) ≠ 0) :
  (m + x) ^ 2000 * (-m ^ 2 * n + x * n ^ 2) + 1 = 1 := by
  sorry

end algebraic_expression_value_l230_230824


namespace proof_problem_l230_230687

-- Definitions based on the given conditions
def cond1 : Prop := 1 * 9 + 2 = 11
def cond2 : Prop := 12 * 9 + 3 = 111
def cond3 : Prop := 123 * 9 + 4 = 1111
def cond4 : Prop := 1234 * 9 + 5 = 11111
def cond5 : Prop := 12345 * 9 + 6 = 111111

-- Main statement to prove
theorem proof_problem (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) (h5 : cond5) : 
  123456 * 9 + 7 = 1111111 :=
sorry

end proof_problem_l230_230687


namespace expression_as_fraction_l230_230172

theorem expression_as_fraction :
  1 + (4 / (5 + (6 / 7))) = (69 : ℚ) / 41 := 
by
  sorry

end expression_as_fraction_l230_230172


namespace minimum_abs_a_l230_230258

-- Given conditions as definitions
def has_integer_coeffs (a b c : ℤ) : Prop := true
def has_roots_in_range (a b c : ℤ) (x1 x2 : ℚ) : Prop :=
  x1 ≠ x2 ∧ 0 < x1 ∧ x1 < 1 ∧ 0 < x2 ∧ x2 < 1 ∧
  (a : ℚ) * x1^2 + (b : ℚ) * x1 + (c : ℚ) = 0 ∧
  (a : ℚ) * x2^2 + (b : ℚ) * x2 + (c : ℚ) = 0

-- Main statement (abstractly mentioning existence of x1, x2 such that they fulfill the polynomial conditions)
theorem minimum_abs_a (a b c : ℤ) (x1 x2 : ℚ) :
  has_integer_coeffs a b c →
  has_roots_in_range a b c x1 x2 →
  |a| ≥ 5 :=
by
  intros _ _
  sorry

end minimum_abs_a_l230_230258


namespace min_value_x_plus_2_div_x_minus_2_l230_230800

theorem min_value_x_plus_2_div_x_minus_2 (x : ℝ) (h : x > 2) : 
  ∃ m, m = 2 + 2 * Real.sqrt 2 ∧ x + 2/(x-2) ≥ m :=
by sorry

end min_value_x_plus_2_div_x_minus_2_l230_230800


namespace initial_apples_proof_l230_230781

-- Define the variables and conditions
def initial_apples (handed_out: ℕ) (pies: ℕ) (apples_per_pie: ℕ): ℕ := 
  handed_out + pies * apples_per_pie

-- Define the proof statement
theorem initial_apples_proof : initial_apples 30 7 8 = 86 := by 
  sorry

end initial_apples_proof_l230_230781


namespace least_value_QGK_l230_230815

theorem least_value_QGK :
  ∃ (G K Q : ℕ), (10 * G + G) * G = 100 * Q + 10 * G + K ∧ G ≠ K ∧ (10 * G + G) ≥ 10 ∧ (10 * G + G) < 100 ∧  ∃ x, x = 44 ∧ 100 * G + 10 * 4 + 4 = (100 * Q + 10 * G + K) ∧ 100 * 0 + 10 * 4 + 4 = 044  :=
by
  sorry

end least_value_QGK_l230_230815


namespace min_transport_cost_l230_230404

theorem min_transport_cost :
  let large_truck_capacity := 7
  let large_truck_cost := 600
  let small_truck_capacity := 4
  let small_truck_cost := 400
  let total_goods := 20
  ∃ (n_large n_small : ℕ),
    n_large * large_truck_capacity + n_small * small_truck_capacity ≥ total_goods ∧ 
    (n_large * large_truck_cost + n_small * small_truck_cost) = 1800 :=
sorry

end min_transport_cost_l230_230404


namespace evaluate_expression_l230_230420

theorem evaluate_expression (a b x : ℝ) (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) :
    (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by
  sorry

end evaluate_expression_l230_230420


namespace train_cross_time_approx_l230_230228

noncomputable def length_of_train : ℝ := 100
noncomputable def speed_of_train_km_hr : ℝ := 80
noncomputable def length_of_bridge : ℝ := 142
noncomputable def total_distance : ℝ := length_of_train + length_of_bridge
noncomputable def speed_of_train_m_s : ℝ := speed_of_train_km_hr * 1000 / 3600
noncomputable def time_to_cross_bridge : ℝ := total_distance / speed_of_train_m_s

theorem train_cross_time_approx :
  abs (time_to_cross_bridge - 10.89) < 0.01 :=
by
  sorry

end train_cross_time_approx_l230_230228


namespace find_s_l230_230645

theorem find_s (s : ℝ) :
  let P := (s - 3, 2)
  let Q := (1, s + 2)
  let M := ((s - 2) / 2, (s + 4) / 2)
  let dist_sq := (M.1 - P.1) ^ 2 + (M.2 - P.2) ^ 2
  dist_sq = 3 * s^2 / 4 →
  s = -5 + 5 * Real.sqrt 2 ∨ s = -5 - 5 * Real.sqrt 2 :=
by
  intros P Q M dist_sq h
  sorry

end find_s_l230_230645


namespace sam_hourly_rate_l230_230448

theorem sam_hourly_rate
  (first_month_earnings : ℕ)
  (second_month_earnings : ℕ)
  (total_hours : ℕ)
  (h1 : first_month_earnings = 200)
  (h2 : second_month_earnings = first_month_earnings + 150)
  (h3 : total_hours = 55) :
  (first_month_earnings + second_month_earnings) / total_hours = 10 := 
  by
  sorry

end sam_hourly_rate_l230_230448


namespace probability_of_winning_l230_230079

def total_products_in_box : ℕ := 6
def winning_products_in_box : ℕ := 2

theorem probability_of_winning : (winning_products_in_box : ℚ) / (total_products_in_box : ℚ) = 1 / 3 :=
by sorry

end probability_of_winning_l230_230079


namespace find_number_l230_230497

theorem find_number (x : ℝ) (h : (25 / 100) * x = 20 / 100 * 30) : x = 24 :=
by
  sorry

end find_number_l230_230497


namespace M_positive_l230_230713

theorem M_positive (x y : ℝ) : (3 * x^2 - 8 * x * y + 9 * y^2 - 4 * x + 6 * y + 13) > 0 :=
by
  sorry

end M_positive_l230_230713


namespace observable_sea_creatures_l230_230548

theorem observable_sea_creatures (P_shark : ℝ) (P_truth : ℝ) (n : ℕ)
  (h1 : P_shark = 0.027777777777777773)
  (h2 : P_truth = 1/6)
  (h3 : P_shark = P_truth * (1/n : ℝ)) : 
  n = 6 := 
  sorry

end observable_sea_creatures_l230_230548


namespace original_price_of_wand_l230_230975

theorem original_price_of_wand (P : ℝ) (h1 : 8 = P / 8) : P = 64 :=
by sorry

end original_price_of_wand_l230_230975


namespace count_7_digit_nums_180_reversible_count_7_digit_nums_180_reversible_divis_by_4_sum_of_7_digit_nums_180_reversible_l230_230651

open Nat

def num180Unchanged : Nat := 
  let valid_pairs := [(0, 0), (1, 1), (8, 8), (6, 9), (9, 6)];
  let middle_digits := [0, 1, 8];
  (valid_pairs.length) * ((valid_pairs.length + 1) * (valid_pairs.length + 1) * middle_digits.length)

def num180UnchangedDivBy4 : Nat :=
  let valid_div4_pairs := [(0, 0), (1, 6), (6, 0), (6, 8), (8, 0), (8, 8), (9, 6)];
  let middle_digits := [0, 1, 8];
  valid_div4_pairs.length * (valid_div4_pairs.length / 5) * middle_digits.length

def sum180UnchangedNumbers : Nat :=
   1959460200 -- The sum by the given problem

theorem count_7_digit_nums_180_reversible : num180Unchanged = 300 :=
sorry

theorem count_7_digit_nums_180_reversible_divis_by_4 : num180UnchangedDivBy4 = 75 :=
sorry

theorem sum_of_7_digit_nums_180_reversible : sum180UnchangedNumbers = 1959460200 :=
sorry

end count_7_digit_nums_180_reversible_count_7_digit_nums_180_reversible_divis_by_4_sum_of_7_digit_nums_180_reversible_l230_230651


namespace range_of_b_l230_230698

theorem range_of_b (b : ℤ) : 
  (∃ x1 x2 : ℤ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 ∧ x1 - b > 0 ∧ x2 - b > 0 ∧ (∀ x : ℤ, x < 0 ∧ x - b > 0 → (x = x1 ∨ x = x2))) ↔ (-3 ≤ b ∧ b < -2) :=
by sorry

end range_of_b_l230_230698


namespace value_of_m_l230_230157
noncomputable def y (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^(m^2 - 3)

theorem value_of_m (m : ℝ) (x : ℝ) (h1 : x > 0) (h2 : ∀ x1 x2 : ℝ, x1 > x2 → y m x1 < y m x2) :
  m = 2 :=
sorry

end value_of_m_l230_230157


namespace regular_polygon_sides_l230_230976

theorem regular_polygon_sides (n : ℕ) (h : (n - 2) * 180 = 144 * n) : n = 10 := 
by 
  sorry

end regular_polygon_sides_l230_230976


namespace fraction_evaluation_l230_230936

theorem fraction_evaluation :
  (2 + 4 - 8 + 16 + 32 - 64 + 128 : ℚ) / (4 + 8 - 16 + 32 + 64 - 128 + 256) = 1 / 2 :=
by
  sorry

end fraction_evaluation_l230_230936


namespace arithmetic_sequence_S30_l230_230583

variable {α : Type*} [OrderedAddCommGroup α]

-- Definitions from the conditions
def arithmetic_sum (n : ℕ) : α :=
  sorry -- Placeholder for the sequence sum definition

axiom S10 : arithmetic_sum 10 = 20
axiom S20 : arithmetic_sum 20 = 15

-- The theorem to prove
theorem arithmetic_sequence_S30 : arithmetic_sum 30 = -15 :=
  sorry -- Proof will be completed here

end arithmetic_sequence_S30_l230_230583


namespace complex_eq_l230_230336

theorem complex_eq (a b : ℝ) (i : ℂ) (hi : i^2 = -1) (h : (a + 2 * i) / i = b + i) : a + b = 1 :=
sorry

end complex_eq_l230_230336


namespace production_cost_percentage_l230_230629

theorem production_cost_percentage
    (initial_cost final_cost : ℝ)
    (final_cost_eq : final_cost = 48)
    (initial_cost_eq : initial_cost = 50)
    (h : (initial_cost + 0.5 * x) * (1 - x / 100) = final_cost) :
    x = 20 :=
by
  sorry

end production_cost_percentage_l230_230629


namespace find_quadratic_minimum_value_l230_230403

noncomputable def quadraticMinimumPoint (a b c : ℝ) : ℝ :=
  -b / (2 * a)

theorem find_quadratic_minimum_value :
  quadraticMinimumPoint 3 6 9 = -1 :=
by
  sorry

end find_quadratic_minimum_value_l230_230403


namespace bamboo_node_volume_5_l230_230391

theorem bamboo_node_volume_5 {a_1 d : ℚ} :
  (a_1 + (a_1 + d) + (a_1 + 2 * d) + (a_1 + 3 * d) = 3) →
  ((a_1 + 6 * d) + (a_1 + 7 * d) + (a_1 + 8 * d) = 4) →
  (a_1 + 4 * d = 67 / 66) :=
by sorry

end bamboo_node_volume_5_l230_230391


namespace thirty_sixty_ninety_triangle_area_l230_230926

theorem thirty_sixty_ninety_triangle_area (hypotenuse : ℝ) (angle : ℝ) (area : ℝ)
  (h_hypotenuse : hypotenuse = 12)
  (h_angle : angle = 30)
  (h_area : area = 18 * Real.sqrt 3) :
  ∃ (base height : ℝ), 
    base = hypotenuse / 2 ∧ 
    height = (hypotenuse / 2) * Real.sqrt 3 ∧ 
    area = (1 / 2) * base * height :=
by {
  sorry
}

end thirty_sixty_ninety_triangle_area_l230_230926


namespace domain_f_l230_230552

def domain_of_f (x : ℝ) : Prop :=
  (2 ≤ x ∧ x < 3) ∨ (3 < x ∧ x < 4)

theorem domain_f :
  ∀ x, domain_of_f x ↔ (x ≥ 2 ∧ x < 4) ∧ x ≠ 3 :=
by
  sorry

end domain_f_l230_230552


namespace avg_GPA_is_93_l230_230789

def avg_GPA_school (GPA_6th GPA_8th : ℕ) (GPA_diff : ℕ) : ℕ :=
  (GPA_6th + (GPA_6th + GPA_diff) + GPA_8th) / 3

theorem avg_GPA_is_93 :
  avg_GPA_school 93 91 2 = 93 :=
by
  -- The proof can be handled here 
  sorry

end avg_GPA_is_93_l230_230789


namespace problem_1_problem_2_l230_230536

open Real

theorem problem_1
  (a b m n : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hm : m > 0)
  (hn : n > 0) :
  (m ^ 2 / a + n ^ 2 / b) ≥ ((m + n) ^ 2 / (a + b)) :=
sorry

theorem problem_2
  (x : ℝ)
  (hx1 : 0 < x)
  (hx2 : x < 1 / 2) :
  (2 / x + 9 / (1 - 2 * x)) ≥ 25 ∧ (2 / x + 9 / (1 - 2 * x)) = 25 ↔ x = 1 / 5 :=
sorry

end problem_1_problem_2_l230_230536


namespace range_of_a_l230_230562

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x > 2 then 2^x + a else x + a^2

theorem range_of_a (a : ℝ) : (∀ y : ℝ, ∃ x : ℝ, f x a = y) ↔ (a ≤ -1 ∨ a ≥ 2) :=
by
  sorry

end range_of_a_l230_230562


namespace total_bugs_eaten_l230_230058

theorem total_bugs_eaten :
  let gecko_bugs := 12
  let lizard_bugs := gecko_bugs / 2
  let frog_bugs := lizard_bugs * 3
  let toad_bugs := frog_bugs + (frog_bugs / 2)
  gecko_bugs + lizard_bugs + frog_bugs + toad_bugs = 63 :=
by
  sorry

end total_bugs_eaten_l230_230058


namespace necessary_and_sufficient_condition_l230_230572

def f (a x : ℝ) : ℝ := x^2 - a * x + 1

theorem necessary_and_sufficient_condition (a : ℝ) : 
  (∃ x : ℝ, f a x < 0) ↔ |a| > 2 :=
by
  sorry

end necessary_and_sufficient_condition_l230_230572


namespace pass_rate_correct_l230_230216

variable {a b : ℝ}

-- Assumptions: defect rates are between 0 and 1
axiom h_a : 0 ≤ a ∧ a ≤ 1
axiom h_b : 0 ≤ b ∧ b ≤ 1

-- Definition: Pass rate is 1 minus the defect rate
def pass_rate (a b : ℝ) : ℝ := (1 - a) * (1 - b)

-- Theorem: Proving the pass rate is (1 - a) * (1 - b)
theorem pass_rate_correct : pass_rate a b = (1 - a) * (1 - b) := 
by
  sorry

end pass_rate_correct_l230_230216


namespace customer_difference_l230_230025

theorem customer_difference (X Y Z : ℕ) (h1 : X - Y = 10) (h2 : 10 - Z = 4) : X - 4 = 10 :=
by sorry

end customer_difference_l230_230025


namespace unique_triple_l230_230016

theorem unique_triple (x y p : ℕ) (hx : 0 < x) (hy : 0 < y) (hp : Nat.Prime p) (h1 : p = x^2 + 1) (h2 : 2 * p^2 = y^2 + 1) :
  (x, y, p) = (2, 7, 5) :=
sorry

end unique_triple_l230_230016


namespace pure_alcohol_addition_l230_230119

variables (P : ℝ) (V : ℝ := 14.285714285714286 ) (initial_volume : ℝ := 100) (final_percent_alcohol : ℝ := 0.30)

theorem pure_alcohol_addition :
  P / 100 * initial_volume + V = final_percent_alcohol * (initial_volume + V) :=
by
  sorry

end pure_alcohol_addition_l230_230119


namespace floor_area_not_greater_than_10_l230_230676

theorem floor_area_not_greater_than_10 (L W H : ℝ) (h_height : H = 3)
  (h_more_paint_wall1 : L * 3 > L * W)
  (h_more_paint_wall2 : W * 3 > L * W) :
  L * W ≤ 9 :=
by
  sorry

end floor_area_not_greater_than_10_l230_230676


namespace newspaper_cost_over_8_weeks_l230_230499

def cost (day : String) : Real := 
  if day = "Sunday" then 2.00 
  else if day = "Wednesday" ∨ day = "Thursday" ∨ day = "Friday" then 0.50 
  else 0

theorem newspaper_cost_over_8_weeks : 
  (8 * ((cost "Wednesday" + cost "Thursday" + cost "Friday") + cost "Sunday")) = 28.00 :=
  by sorry

end newspaper_cost_over_8_weeks_l230_230499


namespace shaded_area_concentric_circles_l230_230426

theorem shaded_area_concentric_circles (R : ℝ) (r : ℝ) (hR : π * R^2 = 100 * π) (hr : r = R / 2) :
  (1 / 2) * π * R^2 + (1 / 2) * π * r^2 = 62.5 * π :=
by
  -- Given conditions
  have R10 : R = 10 := sorry  -- Derived from hR
  have r5 : r = 5 := sorry    -- Derived from hr and R10
  -- Proof steps likely skipped
  sorry

end shaded_area_concentric_circles_l230_230426


namespace percent_of_value_l230_230245

theorem percent_of_value : (2 / 5) * (1 / 100) * 450 = 1.8 :=
by sorry

end percent_of_value_l230_230245


namespace find_dividend_l230_230393

noncomputable def quotient : ℕ := 2015
noncomputable def remainder : ℕ := 0
noncomputable def divisor : ℕ := 105

theorem find_dividend : quotient * divisor + remainder = 20685 := by
  sorry

end find_dividend_l230_230393


namespace average_infections_per_round_infections_after_three_rounds_l230_230194

-- Define the average number of infections per round such that the total after two rounds is 36 and x > 0
theorem average_infections_per_round :
  ∃ x : ℤ, (1 + x)^2 = 36 ∧ x > 0 :=
by
  sorry

-- Given x = 5, prove that the total number of infections after three rounds exceeds 200
theorem infections_after_three_rounds (x : ℤ) (H : x = 5) :
  (1 + x)^3 > 200 :=
by
  sorry

end average_infections_per_round_infections_after_three_rounds_l230_230194


namespace number_of_carbon_atoms_l230_230765

-- Definitions and Conditions
def hydrogen_atoms : ℕ := 6
def molecular_weight : ℕ := 78
def hydrogen_atomic_weight : ℕ := 1
def carbon_atomic_weight : ℕ := 12

-- Theorem Statement: Number of Carbon Atoms
theorem number_of_carbon_atoms 
  (H_atoms : ℕ := hydrogen_atoms)
  (M_weight : ℕ := molecular_weight)
  (H_weight : ℕ := hydrogen_atomic_weight)
  (C_weight : ℕ := carbon_atomic_weight) : 
  (M_weight - H_atoms * H_weight) / C_weight = 6 :=
sorry

end number_of_carbon_atoms_l230_230765


namespace solve_inequality_l230_230145

def satisfies_inequality (x : ℝ) : Prop :=
  (3 * x - 4) * (x + 1) / x ≥ 0

theorem solve_inequality :
  {x : ℝ | satisfies_inequality x} = {x : ℝ | -1 ≤ x ∧ x < 0 ∨ x ≥ 4 / 3} :=
by
  sorry

end solve_inequality_l230_230145


namespace binomial_expansion_const_term_l230_230088

theorem binomial_expansion_const_term (a : ℝ) (h : a > 0) 
  (A : ℝ) (B : ℝ) :
  (A = (15 * a ^ 4)) ∧ (B = 15 * a ^ 2) ∧ (A = 4 * B) → B = 60 := 
by 
  -- The actual proof is omitted
  sorry

end binomial_expansion_const_term_l230_230088


namespace average_of_21_numbers_l230_230540

theorem average_of_21_numbers (n₁ n₂ : ℕ) (a b c : ℕ)
  (h₁ : n₁ = 11 * 48) -- Sum of the first 11 numbers
  (h₂ : n₂ = 11 * 41) -- Sum of the last 11 numbers
  (h₃ : c = 55) -- The 11th number
  : (n₁ + n₂ - c) / 21 = 44 := -- Average of all 21 numbers
by
  sorry

end average_of_21_numbers_l230_230540


namespace solution_set_of_inequality_l230_230321

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_of_inequality 
  (hf_even : ∀ x : ℝ, f x = f (|x|))
  (hf_increasing : ∀ x y : ℝ, x < y → x < 0 → y < 0 → f x < f y)
  (hf_value : f 3 = 1) :
  {x : ℝ | f (x - 1) < 1} = {x : ℝ | x > 4 ∨ x < -2} := 
sorry

end solution_set_of_inequality_l230_230321


namespace proof_problem_l230_230211

def A : Set ℝ := {x | x < 4}
def B : Set ℝ := {x | x^2 - 4 * x + 3 > 0}

theorem proof_problem : {x | x ∈ A ∧ x ∉ (A ∩ B)} = {x | 1 ≤ x ∧ x ≤ 3} :=
by {
  sorry
}

end proof_problem_l230_230211


namespace length_of_boat_l230_230234

-- Definitions based on the conditions
def breadth : ℝ := 3
def sink_depth : ℝ := 0.01
def man_mass : ℝ := 120
def g : ℝ := 9.8 -- acceleration due to gravity

-- Derived from the conditions
def weight_man : ℝ := man_mass * g
def density_water : ℝ := 1000

-- Statement to be proved
theorem length_of_boat : ∃ L : ℝ, (breadth * sink_depth * L * density_water * g = weight_man) → L = 4 :=
by
  sorry

end length_of_boat_l230_230234


namespace arithmetic_square_root_of_nine_l230_230503

theorem arithmetic_square_root_of_nine : Real.sqrt 9 = 3 :=
sorry

end arithmetic_square_root_of_nine_l230_230503


namespace noon_temperature_l230_230158

variable (a : ℝ)

theorem noon_temperature (h1 : ∀ (x : ℝ), x = a) (h2 : ∀ (y : ℝ), y = a + 10) :
  a + 10 = y :=
by
  sorry

end noon_temperature_l230_230158


namespace intersection_of_lines_l230_230442

theorem intersection_of_lines :
  ∃ x y : ℚ, 12 * x - 5 * y = 8 ∧ 10 * x + 2 * y = 20 ∧ x = 58 / 37 ∧ y = 667 / 370 :=
by
  sorry

end intersection_of_lines_l230_230442


namespace andy_tomatoes_l230_230862

theorem andy_tomatoes (P : ℕ) (h1 : ∀ P, 7 * P / 3 = 42) : P = 18 := by
  sorry

end andy_tomatoes_l230_230862


namespace part1_part2_part3_l230_230382

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

end part1_part2_part3_l230_230382


namespace div_30_prime_ge_7_l230_230928

theorem div_30_prime_ge_7 (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_7 : p ≥ 7) : 30 ∣ (p^2 - 1) := 
sorry

end div_30_prime_ge_7_l230_230928


namespace find_f2_l230_230428

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry
noncomputable def a : ℝ := sorry

axiom odd_f : ∀ x, f (-x) = -f x
axiom even_g : ∀ x, g (-x) = g x
axiom fg_eq : ∀ x, f x + g x = a^x - a^(-x) + 2
axiom g2_a : g 2 = a
axiom a_pos : a > 0
axiom a_ne1 : a ≠ 1

theorem find_f2 : f 2 = 15 / 4 := 
by sorry

end find_f2_l230_230428


namespace car_rental_total_cost_l230_230648

theorem car_rental_total_cost 
  (rental_cost : ℕ)
  (gallons : ℕ)
  (cost_per_gallon : ℕ)
  (cost_per_mile : ℚ)
  (miles_driven : ℕ)
  (H1 : rental_cost = 150)
  (H2 : gallons = 8)
  (H3 : cost_per_gallon = 350 / 100)
  (H4 : cost_per_mile = 50 / 100)
  (H5 : miles_driven = 320) :
  rental_cost + gallons * cost_per_gallon + miles_driven * cost_per_mile = 338 :=
  sorry

end car_rental_total_cost_l230_230648


namespace point_in_fourth_quadrant_coords_l230_230061

theorem point_in_fourth_quadrant_coords 
  (P : ℝ × ℝ)
  (h1 : P.2 < 0)
  (h2 : abs P.2 = 2)
  (h3 : P.1 > 0)
  (h4 : abs P.1 = 5) :
  P = (5, -2) :=
sorry

end point_in_fourth_quadrant_coords_l230_230061


namespace sqrt_sum_l230_230077

theorem sqrt_sum (m n : ℝ) (h1 : m + n = 0) (h2 : m * n = -2023) : m + 2 * m * n + n = -4046 :=
by sorry

end sqrt_sum_l230_230077


namespace range_of_m_l230_230930

def proposition_p (m : ℝ) : Prop :=
  ∀ x > 0, m^2 + 2 * m - 1 ≤ x + 1 / x

def proposition_q (m : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → (5 - m^2) ^ x > (5 - m^2) ^ (x - 1)

theorem range_of_m (m : ℝ) : (proposition_p m ∨ proposition_q m) ∧ ¬ (proposition_p m ∧ proposition_q m) ↔ (-3 ≤ m ∧ m ≤ -2) ∨ (1 < m ∧ m < 2) :=
sorry

end range_of_m_l230_230930


namespace problem_l230_230120

noncomputable def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
noncomputable def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

variable (f g : ℝ → ℝ)
variable (h₁ : is_odd f)
variable (h₂ : is_even g)
variable (h₃ : ∀ x, f x - g x = 2 * x^3 + x^2 + 3)

theorem problem : f 2 + g 2 = 9 :=
by sorry

end problem_l230_230120


namespace problem_l230_230471

def expr : ℤ := 7^2 - 4 * 5 + 2^2

theorem problem : expr = 33 := by
  sorry

end problem_l230_230471


namespace no_valid_m_l230_230514

theorem no_valid_m
  (m : ℕ)
  (hm : m > 0)
  (h1 : ∃ k1 : ℕ, k1 > 0 ∧ 1806 = k1 * (m^2 - 2))
  (h2 : ∃ k2 : ℕ, k2 > 0 ∧ 1806 = k2 * (m^2 + 2)) :
  false :=
sorry

end no_valid_m_l230_230514


namespace max_grandchildren_l230_230292

theorem max_grandchildren (children_count : ℕ) (common_gc : ℕ) (special_gc_count : ℕ) : 
  children_count = 8 ∧ common_gc = 8 ∧ special_gc_count = 5 →
  (6 * common_gc + 2 * special_gc_count) = 58 := by
  sorry

end max_grandchildren_l230_230292


namespace compare_decimal_fraction_l230_230452

theorem compare_decimal_fraction : 0.8 - (1 / 2) = 0.3 := by
  sorry

end compare_decimal_fraction_l230_230452


namespace isosceles_right_triangle_leg_length_l230_230501

theorem isosceles_right_triangle_leg_length (H : Real)
  (median_to_hypotenuse_is_half : ∀ H, (H / 2) = 12) :
  (H / Real.sqrt 2) = 12 * Real.sqrt 2 :=
by
  -- Proof goes here
  sorry

end isosceles_right_triangle_leg_length_l230_230501


namespace angle_B_is_pi_over_3_l230_230542

theorem angle_B_is_pi_over_3
  (A B C a b c : ℝ)
  (h1 : b * Real.cos B = (a * Real.cos C + c * Real.cos A) / 2)
  (h2 : 0 < B)
  (h3 : B < Real.pi)
  (h4 : 0 < A)
  (h5 : A < Real.pi)
  (h6 : 0 < C)
  (h7 : C < Real.pi) :
  B = Real.pi / 3 :=
by
  sorry

end angle_B_is_pi_over_3_l230_230542


namespace ribbon_original_length_l230_230663

theorem ribbon_original_length (x : ℕ) (h1 : 11 * 35 = 7 * x) : x = 55 :=
by
  sorry

end ribbon_original_length_l230_230663


namespace geometric_sequence_mean_l230_230569

theorem geometric_sequence_mean (a : ℕ → ℝ) (q : ℝ) (h_q : q = -2) 
  (h_condition : a 3 * a 7 = 4 * a 4) : 
  ((a 8 + a 11) / 2 = -56) 
:= sorry

end geometric_sequence_mean_l230_230569


namespace find_S_l230_230002

theorem find_S (R S T : ℝ) (c : ℝ)
  (h1 : R = c * (S / T))
  (h2 : R = 2) (h3 : S = 1/2) (h4 : T = 4/3) (h_c : c = 16/3)
  (h_R : R = Real.sqrt 75) (h_T : T = Real.sqrt 32) :
  S = 45/4 := by
  sorry

end find_S_l230_230002


namespace transformed_eq_l230_230758

theorem transformed_eq (a b c : ℤ) (h : a > 0) :
  (∀ x : ℝ, 16 * x^2 + 32 * x - 40 = 0 → (a * x + b)^2 = c) →
  a + b + c = 64 :=
by
  sorry

end transformed_eq_l230_230758


namespace quadratic_common_root_distinct_real_numbers_l230_230591

theorem quadratic_common_root_distinct_real_numbers:
  ∀ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a →
  (∃ x, x^2 + a * x + b = 0 ∧ x^2 + c * x + a = 0) ∧
  (∃ y, y^2 + a * y + b = 0 ∧ y^2 + b * y + c = 0) ∧
  (∃ z, z^2 + b * z + c = 0 ∧ z^2 + c * z + a = 0) →
  a^2 + b^2 + c^2 = 6 :=
by
  intros a b c h_distinct h_common_root
  sorry

end quadratic_common_root_distinct_real_numbers_l230_230591


namespace john_avg_increase_l230_230907

theorem john_avg_increase (a b c d : ℝ) (h₁ : a = 90) (h₂ : b = 85) (h₃ : c = 92) (h₄ : d = 95) :
    let initial_avg := (a + b + c) / 3
    let new_avg := (a + b + c + d) / 4
    new_avg - initial_avg = 1.5 :=
by
  sorry

end john_avg_increase_l230_230907


namespace system_solutions_are_equivalent_l230_230718

theorem system_solutions_are_equivalent :
  ∀ (a b x y : ℝ),
  (2 * a - 3 * b = 13 ∧ 3 * a + 5 * b = 30.9) ∧
  (a = 8.3 ∧ b = 1.2) ∧
  (x + 2 = a ∧ y - 1 = b) →
  x = 6.3 ∧ y = 2.2 :=
by
  -- Sorry is added intentionally to skip the proof
  sorry

end system_solutions_are_equivalent_l230_230718


namespace age_difference_l230_230053

theorem age_difference (b_age : ℕ) (bro_age : ℕ) (h1 : b_age = 5) (h2 : b_age + bro_age = 19) : 
  bro_age - b_age = 9 :=
by
  sorry

end age_difference_l230_230053


namespace number_division_l230_230982

theorem number_division (x : ℤ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by 
  sorry

end number_division_l230_230982


namespace hcf_of_two_numbers_l230_230076

noncomputable def H : ℕ := 322 / 14

theorem hcf_of_two_numbers (H k : ℕ) (lcm_val : ℕ) :
  lcm_val = H * 13 * 14 ∧ 322 = H * k ∧ 322 / 14 = H → H = 23 :=
by
  sorry

end hcf_of_two_numbers_l230_230076


namespace area_of_roof_l230_230298

def roof_area (w l : ℕ) : ℕ := l * w

theorem area_of_roof :
  ∃ (w l : ℕ), l = 4 * w ∧ l - w = 45 ∧ roof_area w l = 900 :=
by
  -- Defining witnesses for width and length
  use 15, 60
  -- Splitting the goals for clarity
  apply And.intro
  -- Proving the first condition: l = 4 * w
  · show 60 = 4 * 15
    rfl
  apply And.intro
  -- Proving the second condition: l - w = 45
  · show 60 - 15 = 45
    rfl
  -- Proving the area calculation: roof_area w l = 900
  · show roof_area 15 60 = 900
    rfl

end area_of_roof_l230_230298


namespace smallest_N_l230_230615

theorem smallest_N (N : ℕ) (h : 7 * N = 999999) : N = 142857 :=
sorry

end smallest_N_l230_230615


namespace scientific_notation_of_20000_l230_230535

def number : ℕ := 20000

theorem scientific_notation_of_20000 : number = 2 * 10 ^ 4 :=
by
  sorry

end scientific_notation_of_20000_l230_230535


namespace james_speed_downhill_l230_230181

theorem james_speed_downhill (T1 T2 v : ℝ) (h1 : T1 = 20 / v) (h2 : T2 = 12 / 3 + 1) (h3 : T1 = T2 - 1) : v = 5 :=
by
  -- Declare variables
  have hT2 : T2 = 5 := by linarith
  have hT1 : T1 = 4 := by linarith
  have hv : v = 20 / 4 := by sorry
  linarith

#exit

end james_speed_downhill_l230_230181


namespace cos_five_pi_over_six_l230_230883

theorem cos_five_pi_over_six :
  Real.cos (5 * Real.pi / 6) = -(Real.sqrt 3 / 2) :=
sorry

end cos_five_pi_over_six_l230_230883


namespace exists_k_with_three_different_real_roots_exists_k_with_two_different_real_roots_l230_230188

noncomputable def equation (x : ℝ) (k : ℝ) := x^2 - 2 * |x| - (2 * k + 1)^2

theorem exists_k_with_three_different_real_roots :
  ∃ k : ℝ, ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ equation x1 k = 0 ∧ equation x2 k = 0 ∧ equation x3 k = 0 :=
sorry

theorem exists_k_with_two_different_real_roots :
  ∃ k : ℝ, ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ equation x1 k = 0 ∧ equation x2 k = 0 :=
sorry

end exists_k_with_three_different_real_roots_exists_k_with_two_different_real_roots_l230_230188


namespace sum_of_terms_l230_230030

-- Given the condition that the sequence a_n is an arithmetic sequence
-- with Sum S_n of first n terms such that S_3 = 9 and S_6 = 36,
-- prove that a_7 + a_8 + a_9 is 45.

variable (a : ℕ → ℝ) -- arithmetic sequence
variable (S : ℕ → ℝ) -- sum of the first n terms of the sequence

axiom sum_3 : S 3 = 9
axiom sum_6 : S 6 = 36
axiom sum_seq_arith : ∀ n : ℕ, S n = n * (a 1) + (n - 1) * n / 2 * (a 2 - a 1)

theorem sum_of_terms : a 7 + a 8 + a 9 = 45 :=
by {
  sorry
}

end sum_of_terms_l230_230030


namespace polynomial_constant_l230_230342

theorem polynomial_constant (P : ℝ → ℝ → ℝ) (h : ∀ x y : ℝ, P (x + y) (y - x) = P x y) : 
  ∃ c : ℝ, ∀ x y : ℝ, P x y = c := 
sorry

end polynomial_constant_l230_230342


namespace remainder_2456789_div_7_l230_230166

theorem remainder_2456789_div_7 :
  2456789 % 7 = 6 := 
by 
  sorry

end remainder_2456789_div_7_l230_230166


namespace least_integer_value_y_l230_230239

theorem least_integer_value_y (y : ℤ) (h : abs (3 * y - 4) ≤ 25) : y = -7 :=
sorry

end least_integer_value_y_l230_230239


namespace boat_speed_in_still_water_l230_230114

theorem boat_speed_in_still_water (D V_s t_down t_up : ℝ) (h_val : V_s = 3) (h_down : D = (15 + V_s) * t_down) (h_up : D = (15 - V_s) * t_up) : 15 = 15 :=
by
  have h1 : 15 = (D / 1 - V_s) := sorry
  have h2 : 15 = (D / 1.5 + V_s) := sorry
  sorry

end boat_speed_in_still_water_l230_230114


namespace geometric_sequence_sum_of_first_four_terms_l230_230785

theorem geometric_sequence_sum_of_first_four_terms 
  (a q : ℝ)
  (h1 : a * (1 + q) = 7)
  (h2 : a * (q^6 - 1) / (q - 1) = 91) :
  a * (1 + q + q^2 + q^3) = 28 := by
  sorry

end geometric_sequence_sum_of_first_four_terms_l230_230785


namespace minimum_possible_value_of_BC_l230_230613

def triangle_ABC_side_lengths_are_integers (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

def angle_A_is_twice_angle_B (A B C : ℝ) : Prop :=
  A = 2 * B

def CA_is_nine (CA : ℕ) : Prop :=
  CA = 9

theorem minimum_possible_value_of_BC
  (a b c : ℕ) (A B C : ℝ) (CA : ℕ)
  (h1 : triangle_ABC_side_lengths_are_integers a b c)
  (h2 : angle_A_is_twice_angle_B A B C)
  (h3 : CA_is_nine CA) :
  ∃ (BC : ℕ), BC = 12 := 
sorry

end minimum_possible_value_of_BC_l230_230613


namespace brownies_each_l230_230917

theorem brownies_each (num_columns : ℕ) (num_rows : ℕ) (total_people : ℕ) (total_brownies : ℕ) 
(h1 : num_columns = 6) (h2 : num_rows = 3) (h3 : total_people = 6) 
(h4 : total_brownies = num_columns * num_rows) : 
total_brownies / total_people = 3 := 
by
  -- Placeholder for the actual proof
  sorry

end brownies_each_l230_230917


namespace find_f_of_4_l230_230771

noncomputable def power_function (x : ℝ) (α : ℝ) : ℝ := x^α

theorem find_f_of_4 :
  (∃ α : ℝ, power_function 3 α = Real.sqrt 3) →
  power_function 4 (1/2) = 2 :=
by
  sorry

end find_f_of_4_l230_230771


namespace smallest_x_for_multiple_l230_230895

theorem smallest_x_for_multiple (x : ℕ) (h : x > 0) :
  (450 * x) % 500 = 0 ↔ x = 10 := by
  sorry

end smallest_x_for_multiple_l230_230895


namespace trackball_mice_count_l230_230250

theorem trackball_mice_count
  (total_mice : ℕ)
  (wireless_fraction : ℕ)
  (optical_fraction : ℕ)
  (h_total : total_mice = 80)
  (h_wireless : wireless_fraction = total_mice / 2)
  (h_optical : optical_fraction = total_mice / 4) :
  total_mice - (wireless_fraction + optical_fraction) = 20 :=
sorry

end trackball_mice_count_l230_230250


namespace quadratic_vertex_coordinates_l230_230932

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ :=
  -2 * (x + 1)^2 - 4

-- State the main theorem to be proved: The vertex of the quadratic function is at (-1, -4)
theorem quadratic_vertex_coordinates : 
  ∃ h k : ℝ, ∀ x : ℝ, quadratic x = -2 * (x + h)^2 + k ∧ h = -1 ∧ k = -4 := 
by
  -- proof required here
  sorry

end quadratic_vertex_coordinates_l230_230932


namespace perimeter_of_rectangle_l230_230095

theorem perimeter_of_rectangle (area : ℝ) (num_squares : ℕ) (square_side : ℝ) (width : ℝ) (height : ℝ) 
  (h1 : area = 216) (h2 : num_squares = 6) (h3 : area / num_squares = square_side^2)
  (h4 : width = 3 * square_side) (h5 : height = 2 * square_side) : 
  2 * (width + height) = 60 :=
by
  sorry

end perimeter_of_rectangle_l230_230095


namespace ratio_of_girls_to_boys_l230_230783

variables (g b : ℕ)

theorem ratio_of_girls_to_boys (h₁ : b = g - 6) (h₂ : g + b = 36) :
  (g / gcd g b) / (b / gcd g b) = 7 / 5 :=
by
  sorry

end ratio_of_girls_to_boys_l230_230783


namespace fencing_required_l230_230307

theorem fencing_required (L W : ℕ) (hL : L = 30) (hArea : L * W = 720) : L + 2 * W = 78 :=
by
  sorry

end fencing_required_l230_230307


namespace area_of_quadrilateral_is_correct_l230_230731

noncomputable def area_of_quadrilateral_BGFAC : ℝ :=
  let a := 3 -- side of the equilateral triangle
  let triangle_area := (a^2 * Real.sqrt 3) / 4 -- area of ABC
  let ratio_AG_GC := 2 -- ratio AG:GC = 2:1
  let area_AGC := triangle_area / 3 -- area of triangle AGC
  let area_BGC := triangle_area / 3 -- area of triangle BGC
  let area_BFC := (2 : ℝ) * triangle_area / 3 -- area of triangle BFC
  let area_BGFC := area_BGC + area_BFC -- area of quadrilateral BGFC
  area_BGFC

theorem area_of_quadrilateral_is_correct :
  area_of_quadrilateral_BGFAC = (3 * Real.sqrt 3) / 2 :=
by
  -- Proof will be provided here
  sorry

end area_of_quadrilateral_is_correct_l230_230731


namespace picking_ball_is_random_event_l230_230950

-- Definitions based on problem conditions
def total_balls := 201
def black_balls := 200
def white_balls := 1

-- The goal to prove
theorem picking_ball_is_random_event : 
  (total_balls = black_balls + white_balls) ∧ 
  (black_balls > 0) ∧ 
  (white_balls > 0) → 
  random_event :=
by sorry

end picking_ball_is_random_event_l230_230950


namespace craig_age_l230_230723

theorem craig_age (C M : ℕ) (h1 : C = M - 24) (h2 : C + M = 56) : C = 16 := 
by
  sorry

end craig_age_l230_230723


namespace total_area_of_figure_l230_230511

noncomputable def radius_of_circle (d : ℝ) : ℝ := d / 2

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r ^ 2

def side_length_of_square (d : ℝ) : ℝ := d

def area_of_square (s : ℝ) : ℝ := s ^ 2

noncomputable def total_area (d : ℝ) : ℝ := area_of_square d + area_of_circle (radius_of_circle d)

theorem total_area_of_figure (d : ℝ) (h : d = 6) : total_area d = 36 + 9 * Real.pi :=
by
  -- skipping proof with sorry
  sorry

end total_area_of_figure_l230_230511


namespace fifth_equation_l230_230951

theorem fifth_equation
: 1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2 :=
by
  sorry

end fifth_equation_l230_230951


namespace greatest_value_of_squares_exists_max_value_of_squares_l230_230623

theorem greatest_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 18)
  (h2 : ab + c + d = 83)
  (h3 : ad + bc = 174)
  (h4 : cd = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 702 :=
sorry

theorem exists_max_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 18)
  (h2 : ab + c + d = 83)
  (h3 : ad + bc = 174)
  (h4 : cd = 105) :
  ∃ (a b c d : ℝ), a^2 + b^2 + c^2 + d^2 = 702 :=
sorry

end greatest_value_of_squares_exists_max_value_of_squares_l230_230623


namespace combined_value_of_a_and_b_l230_230263

theorem combined_value_of_a_and_b :
  (∃ a b : ℝ,
    0.005 * a = 95 / 100 ∧
    b = 3 * a - 50 ∧
    a + b = 710) :=
sorry

end combined_value_of_a_and_b_l230_230263


namespace maddie_total_cost_l230_230086

theorem maddie_total_cost :
  let price_palette := 15
  let price_lipstick := 2.5
  let price_hair_color := 4
  let num_palettes := 3
  let num_lipsticks := 4
  let num_hair_colors := 3
  let total_cost := (num_palettes * price_palette) + (num_lipsticks * price_lipstick) + (num_hair_colors * price_hair_color)
  total_cost = 67 := by
  sorry

end maddie_total_cost_l230_230086


namespace units_digit_of_7_pow_y_plus_6_is_9_l230_230672

theorem units_digit_of_7_pow_y_plus_6_is_9 (y : ℕ) (hy : 0 < y) : 
  (7^y + 6) % 10 = 9 ↔ ∃ k : ℕ, y = 4 * k + 3 := by
  sorry

end units_digit_of_7_pow_y_plus_6_is_9_l230_230672


namespace sqrt_nested_eq_x_pow_eleven_eighths_l230_230750

theorem sqrt_nested_eq_x_pow_eleven_eighths (x : ℝ) (hx : 0 ≤ x) : 
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (11 / 8) :=
  sorry

end sqrt_nested_eq_x_pow_eleven_eighths_l230_230750


namespace find_smaller_number_l230_230556

-- Define the conditions
def condition1 (x y : ℤ) : Prop := x + y = 30
def condition2 (x y : ℤ) : Prop := x - y = 10

-- Define the theorem to prove the smaller number is 10
theorem find_smaller_number (x y : ℤ) (h1 : condition1 x y) (h2 : condition2 x y) : y = 10 := 
sorry

end find_smaller_number_l230_230556


namespace license_plate_combinations_l230_230985

open Nat

theorem license_plate_combinations : 
  (∃ (choose_two_letters: ℕ) (place_first_letter: ℕ) (place_second_letter: ℕ) (choose_non_repeated: ℕ)
     (first_digit: ℕ) (second_digit: ℕ) (third_digit: ℕ),
    choose_two_letters = choose 26 2 ∧
    place_first_letter = choose 5 2 ∧
    place_second_letter = choose 3 2 ∧
    choose_non_repeated = 24 ∧
    first_digit = 10 ∧
    second_digit = 9 ∧
    third_digit = 8 ∧
    choose_two_letters * place_first_letter * place_second_letter * choose_non_repeated * first_digit * second_digit * third_digit = 56016000) :=
sorry

end license_plate_combinations_l230_230985


namespace max_strings_cut_volleyball_net_l230_230461

-- Define the structure of a volleyball net with 10x20 cells where each cell is divided into 4 triangles.
structure VolleyballNet : Type where
  -- The dimensions of the volleyball net
  rows : ℕ
  cols : ℕ
  -- Number of nodes (vertices + centers)
  nodes : ℕ
  -- Maximum number of strings (edges) connecting neighboring nodes that can be cut without disconnecting the net
  max_cut_without_disconnection : ℕ

-- Define the specific volleyball net in question
def volleyball_net : VolleyballNet := 
  { rows := 10, 
    cols := 20, 
    nodes := (11 * 21) + (10 * 20), -- vertices + center nodes
    max_cut_without_disconnection := 800 
  }

-- The main theorem stating that we can cut these strings without the net falling apart
theorem max_strings_cut_volleyball_net (net : VolleyballNet) 
    (h_dim : net.rows = 10) 
    (h_dim2 : net.cols = 20) :
  net.max_cut_without_disconnection = 800 :=
sorry -- The proof is omitted

end max_strings_cut_volleyball_net_l230_230461


namespace saline_solution_concentration_l230_230938

theorem saline_solution_concentration
  (C : ℝ) -- concentration of the first saline solution
  (h1 : 3.6 * C + 1.4 * 9 = 5 * 3.24) : -- condition based on the total salt content
  C = 1 := 
sorry

end saline_solution_concentration_l230_230938


namespace max_intersections_l230_230920

theorem max_intersections (X Y : Type) [Fintype X] [Fintype Y]
  (hX : Fintype.card X = 20) (hY : Fintype.card Y = 10) : 
  ∃ (m : ℕ), m = 8550 := by
  sorry

end max_intersections_l230_230920


namespace part1_part2_l230_230657

-- Definitions for the sets A and B
def A := {x : ℝ | x^2 - 2 * x - 8 = 0}
def B (a : ℝ) := {x : ℝ | x^2 + a * x + a^2 - 12 = 0}

-- Proof statements
theorem part1 (a : ℝ) : (A ∩ B a = A) → a = -2 :=
by
  sorry

theorem part2 (a : ℝ) : (A ∪ B a = A) → (a ≥ 4 ∨ a < -4 ∨ a = -2) :=
by
  sorry

end part1_part2_l230_230657


namespace mike_spent_l230_230816

def trumpet_price : ℝ := 145.16
def song_book_price : ℝ := 5.84
def total_price : ℝ := 151.00

theorem mike_spent :
  trumpet_price + song_book_price = total_price :=
by
  sorry

end mike_spent_l230_230816


namespace john_total_distance_l230_230674

theorem john_total_distance :
  let speed1 := 35
  let time1 := 2
  let distance1 := speed1 * time1

  let speed2 := 55
  let time2 := 3
  let distance2 := speed2 * time2

  let total_distance := distance1 + distance2

  total_distance = 235 := by
    sorry

end john_total_distance_l230_230674


namespace a_lt_one_l230_230028

-- Define the function f(x) = |x-3| + |x+7|
def f (x : ℝ) : ℝ := |x-3| + |x+7|

-- The statement of the problem
theorem a_lt_one (a : ℝ) :
  (∀ x : ℝ, a < Real.log (f x)) → a < 1 :=
by
  intro h
  have H : f (-7) = 10 := by sorry -- piecewise definition
  have H1 : Real.log (f (-7)) = 1 := by sorry -- minimum value of log
  specialize h (-7)
  rw [H1] at h
  exact h

end a_lt_one_l230_230028


namespace nicky_cards_value_l230_230889

theorem nicky_cards_value 
  (x : ℝ)
  (h : 21 = 2 * x + 5) : 
  x = 8 := by
  sorry

end nicky_cards_value_l230_230889


namespace four_positive_reals_inequality_l230_230874

theorem four_positive_reals_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a^3 + b^3 + c^3 + d^3 ≥ a^2 * b + b^2 * c + c^2 * d + d^2 * a :=
sorry

end four_positive_reals_inequality_l230_230874


namespace integer_satisfies_mod_l230_230280

theorem integer_satisfies_mod (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 23) (h3 : 38635 % 23 = n % 23) :
  n = 18 := 
sorry

end integer_satisfies_mod_l230_230280


namespace greatest_triangle_perimeter_l230_230984

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end greatest_triangle_perimeter_l230_230984


namespace mod_computation_l230_230083

theorem mod_computation (a b n : ℕ) (h_modulus : n = 7) (h_a : a = 47) (h_b : b = 28) :
  (a^2023 - b^2023) % n = 5 :=
by
  sorry

end mod_computation_l230_230083


namespace marathon_distance_l230_230220

theorem marathon_distance (d_1 : ℕ) (n : ℕ) (h1 : d_1 = 3) (h2 : n = 5): 
  (2 ^ (n - 1)) * d_1 = 48 :=
by
  sorry

end marathon_distance_l230_230220


namespace zero_in_interval_l230_230981

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + 2 * x^2 - 4 * x

theorem zero_in_interval : ∃ (c : ℝ), 1 < c ∧ c < Real.exp 1 ∧ f c = 0 := sorry

end zero_in_interval_l230_230981


namespace probability_exactly_3_divisible_by_3_of_7_fair_12_sided_dice_l230_230155
-- Import all necessary libraries

-- Define the conditions as variables
variable (n k : ℕ) (p q : ℚ)
variable (dice_divisible_by_3_prob : ℚ)
variable (dice_not_divisible_by_3_prob : ℚ)

-- Assign values based on the problem statement
noncomputable def cond_replicate_n_fair_12_sided_dice := n = 7
noncomputable def cond_exactly_k_divisible_by_3 := k = 3
noncomputable def cond_prob_divisible_by_3 := dice_divisible_by_3_prob = 1 / 3
noncomputable def cond_prob_not_divisible_by_3 := dice_not_divisible_by_3_prob = 2 / 3

-- The theorem statement with the final answer incorporated
theorem probability_exactly_3_divisible_by_3_of_7_fair_12_sided_dice :
  cond_replicate_n_fair_12_sided_dice n →
  cond_exactly_k_divisible_by_3 k →
  cond_prob_divisible_by_3 dice_divisible_by_3_prob →
  cond_prob_not_divisible_by_3 dice_not_divisible_by_3_prob →
  p = (35 : ℚ) * ((1 / 3) ^ 3) * ((2 / 3) ^ 4) →
  q = (560 / 2187 : ℚ) →
  p = q :=
by
  intros
  sorry

end probability_exactly_3_divisible_by_3_of_7_fair_12_sided_dice_l230_230155


namespace tim_stacked_bales_today_l230_230033

theorem tim_stacked_bales_today (initial_bales : ℕ) (current_bales : ℕ) (initial_eq : initial_bales = 54) (current_eq : current_bales = 82) : 
  current_bales - initial_bales = 28 :=
by
  -- conditions
  have h1 : initial_bales = 54 := initial_eq
  have h2 : current_bales = 82 := current_eq
  sorry

end tim_stacked_bales_today_l230_230033


namespace december_sales_fraction_l230_230550

variable (A : ℝ)

-- Define the total sales for January through November
def total_sales_jan_to_nov := 11 * A

-- Define the sales total for December, which is given as 5 times the average monthly sales from January to November
def sales_dec := 5 * A

-- Define the total sales for the year as the sum of January-November sales and December sales
def total_sales_year := total_sales_jan_to_nov + sales_dec

-- We need to prove that the fraction of the December sales to the total annual sales is 5/16
theorem december_sales_fraction : sales_dec / total_sales_year = 5 / 16 := by
  sorry

end december_sales_fraction_l230_230550


namespace school_supply_cost_l230_230873

theorem school_supply_cost (num_students : ℕ) (pens_per_student : ℕ) (pen_cost : ℝ) 
  (notebooks_per_student : ℕ) (notebook_cost : ℝ) 
  (binders_per_student : ℕ) (binder_cost : ℝ) 
  (highlighters_per_student : ℕ) (highlighter_cost : ℝ) 
  (teacher_discount : ℝ) : 
  num_students = 30 →
  pens_per_student = 5 →
  pen_cost = 0.50 →
  notebooks_per_student = 3 →
  notebook_cost = 1.25 →
  binders_per_student = 1 →
  binder_cost = 4.25 →
  highlighters_per_student = 2 →
  highlighter_cost = 0.75 →
  teacher_discount = 100 →
  (num_students * 
    (pens_per_student * pen_cost + notebooks_per_student * notebook_cost + 
    binders_per_student * binder_cost + highlighters_per_student * highlighter_cost) - 
    teacher_discount) = 260 :=
by
  intros _ _ _ _ _ _ _ _ _ _

  -- Sorry added to skip the proof
  sorry

end school_supply_cost_l230_230873


namespace prove_f_x1_minus_f_x2_lt_zero_l230_230738

variable {f : ℝ → ℝ}

-- Define even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Specify that f is decreasing for x < 0
def decreasing_on_negative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < 0 → y < 0 → x < y → f x > f y

theorem prove_f_x1_minus_f_x2_lt_zero (hx1x2 : |x1| < |x2|)
  (h_even : even_function f)
  (h_decreasing : decreasing_on_negative f) :
  f x1 - f x2 < 0 :=
sorry

end prove_f_x1_minus_f_x2_lt_zero_l230_230738


namespace number_division_l230_230647

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l230_230647


namespace total_cost_is_correct_l230_230586

-- Conditions
def cost_per_object : ℕ := 11
def objects_per_person : ℕ := 5  -- 2 shoes, 2 socks, 1 mobile per person
def number_of_people : ℕ := 3

-- Expected total cost
def expected_total_cost : ℕ := 165

-- Proof problem: Prove that the total cost for storing all objects is 165 dollars
theorem total_cost_is_correct :
  (number_of_people * objects_per_person * cost_per_object) = expected_total_cost :=
by
  sorry

end total_cost_is_correct_l230_230586


namespace Alyssa_weekly_allowance_l230_230349

theorem Alyssa_weekly_allowance
  (A : ℝ)
  (h1 : A / 2 + 8 = 12) :
  A = 8 := 
sorry

end Alyssa_weekly_allowance_l230_230349


namespace speed_of_first_bus_l230_230071

theorem speed_of_first_bus (v : ℕ) (h : (v + 60) * 4 = 460) : v = 55 :=
by
  sorry

end speed_of_first_bus_l230_230071


namespace second_expression_l230_230389

theorem second_expression (a x : ℕ) (h₁ : (2 * a + 16 + x) / 2 = 79) (h₂ : a = 30) : x = 82 := by
  sorry

end second_expression_l230_230389


namespace solution_set_of_inequality_l230_230858

theorem solution_set_of_inequality (x : ℝ) : (0 < x ∧ x < 1/3) ↔ (1/x > 3) := 
sorry

end solution_set_of_inequality_l230_230858


namespace sqrt_of_36_is_6_l230_230947

-- Define the naturals
def arithmetic_square_root (x : ℕ) : ℕ := Nat.sqrt x

theorem sqrt_of_36_is_6 : arithmetic_square_root 36 = 6 :=
by
  -- The proof goes here, but we use sorry to skip it as per instructions.
  sorry

end sqrt_of_36_is_6_l230_230947


namespace intersection_subset_l230_230337

def set_A : Set ℝ := {x | -4 < x ∧ x < 2}
def set_B : Set ℝ := {x | x > 1 ∨ x < -5}
def set_C (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < m}

theorem intersection_subset (m : ℝ) :
  (set_A ∩ set_B) ⊆ set_C m ↔ m = 2 :=
by
  sorry

end intersection_subset_l230_230337


namespace rationalize_denominator_eq_l230_230788

noncomputable def rationalize_denominator : ℝ :=
  18 / (Real.sqrt 36 + Real.sqrt 2)

theorem rationalize_denominator_eq : rationalize_denominator = (54 / 17) - (9 * Real.sqrt 2 / 17) := 
by
  sorry

end rationalize_denominator_eq_l230_230788


namespace negation_correct_l230_230221

def original_statement (a : ℝ) : Prop :=
  a > 0 → a^2 > 0

def negated_statement (a : ℝ) : Prop :=
  a ≤ 0 → a^2 ≤ 0

theorem negation_correct (a : ℝ) : ¬ (original_statement a) ↔ negated_statement a :=
by
  sorry

end negation_correct_l230_230221


namespace number_of_positive_integer_solutions_l230_230283

theorem number_of_positive_integer_solutions :
  ∃ n : ℕ, n = 84 ∧ (∀ x y z t : ℕ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < t ∧ x + y + z + t = 10 → true) :=
sorry

end number_of_positive_integer_solutions_l230_230283


namespace safe_zone_inequality_l230_230346

theorem safe_zone_inequality (x : ℝ) (fuse_burn_rate : ℝ) (run_speed : ℝ) (safe_zone_dist : ℝ) (H1: fuse_burn_rate = 0.5) (H2: run_speed = 4) (H3: safe_zone_dist = 150) :
  run_speed * (x / fuse_burn_rate) ≥ safe_zone_dist :=
sorry

end safe_zone_inequality_l230_230346


namespace solve_for_difference_l230_230925

variable (a b : ℝ)

theorem solve_for_difference (h1 : a^3 - b^3 = 4) (h2 : a^2 + ab + b^2 + a - b = 4) : a - b = 2 :=
sorry

end solve_for_difference_l230_230925


namespace arithmetic_sequence_sum_l230_230127

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ) (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 4 + a 7 = 45)
  (h2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 33 := 
sorry

end arithmetic_sequence_sum_l230_230127


namespace carpet_area_l230_230802

def width : ℝ := 8
def length : ℝ := 1.5

theorem carpet_area : width * length = 12 := by
  sorry

end carpet_area_l230_230802


namespace haylee_has_36_guppies_l230_230214

variables (H J C N : ℝ)
variables (total_guppies : ℝ := 84)

def jose_has_half_of_haylee := J = H / 2
def charliz_has_third_of_jose := C = J / 3
def nicolai_has_four_times_charliz := N = 4 * C
def total_guppies_eq_84 := H + J + C + N = total_guppies

theorem haylee_has_36_guppies 
  (hJ : jose_has_half_of_haylee H J)
  (hC : charliz_has_third_of_jose J C)
  (hN : nicolai_has_four_times_charliz C N)
  (htotal : total_guppies_eq_84 H J C N) :
  H = 36 := 
  sorry

end haylee_has_36_guppies_l230_230214


namespace hyunwoo_cookies_l230_230735

theorem hyunwoo_cookies (packs_initial : Nat) (pieces_per_pack : Nat) (packs_given_away : Nat)
  (h1 : packs_initial = 226) (h2 : pieces_per_pack = 3) (h3 : packs_given_away = 3) :
  (packs_initial - packs_given_away) * pieces_per_pack = 669 := 
by
  sorry

end hyunwoo_cookies_l230_230735


namespace white_tiles_in_square_l230_230338

theorem white_tiles_in_square :
  ∀ (n : ℕ), (n * n = 81) → (n ^ 2 - (2 * n - 1)) = 6480 :=
by
  intro n
  intro hn
  sorry

end white_tiles_in_square_l230_230338


namespace triangle_side_height_inequality_l230_230300

theorem triangle_side_height_inequality (a b h_a h_b S : ℝ) (h1 : a > b) 
  (h2: h_a = 2 * S / a) (h3: h_b = 2 * S / b) :
  a + h_a ≥ b + h_b :=
by sorry

end triangle_side_height_inequality_l230_230300


namespace parabola_conditions_l230_230579

-- Definitions based on conditions
def quadratic_function (a : ℝ) (x : ℝ) : ℝ :=
  x^2 - 4*x - 3 + a

def passes_through (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f x = y

def intersects_at_2_points (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

-- Proof Problem Statement
theorem parabola_conditions (a : ℝ) :
  (passes_through (quadratic_function a) 0 1 → a = 4) ∧
  (intersects_at_2_points (quadratic_function a) → (a = 3 ∨ a = 7)) :=
by
  sorry

end parabola_conditions_l230_230579


namespace tan_150_eq_neg_inv_sqrt_3_l230_230871

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end tan_150_eq_neg_inv_sqrt_3_l230_230871


namespace average_expenditure_whole_week_l230_230787

theorem average_expenditure_whole_week (a b : ℕ) (h₁ : a = 3 * 350) (h₂ : b = 4 * 420) : 
  (a + b) / 7 = 390 :=
by 
  sorry

end average_expenditure_whole_week_l230_230787


namespace ratio_final_to_initial_l230_230173

theorem ratio_final_to_initial (P R T : ℝ) (hR : R = 5) (hT : T = 20) :
  let SI := P * R * T / 100
  let A := P + SI
  A / P = 2 := 
by
  sorry

end ratio_final_to_initial_l230_230173


namespace fraction_equality_l230_230678

theorem fraction_equality {x y : ℝ} (h : x + y ≠ 0) (h1 : x - y ≠ 0) : 
  (-x + y) / (-x - y) = (x - y) / (x + y) := 
sorry

end fraction_equality_l230_230678


namespace find_y_l230_230775

theorem find_y (x y : ℝ) (h1 : x - y = 20) (h2 : x + y = 10) : y = -5 := 
sorry

end find_y_l230_230775


namespace total_fruits_is_43_apple_to_pear_ratio_is_24_to_19_l230_230270

def keith_pears : ℕ := 6
def keith_apples : ℕ := 4
def jason_pears : ℕ := 9
def jason_apples : ℕ := 8
def joan_pears : ℕ := 4
def joan_apples : ℕ := 12

def total_pears : ℕ := keith_pears + jason_pears + joan_pears
def total_apples : ℕ := keith_apples + jason_apples + joan_apples
def total_fruits : ℕ := total_pears + total_apples
def apple_to_pear_ratio : ℚ := total_apples / total_pears

theorem total_fruits_is_43 : total_fruits = 43 := by
  sorry

theorem apple_to_pear_ratio_is_24_to_19 : apple_to_pear_ratio = 24/19 := by
  sorry

end total_fruits_is_43_apple_to_pear_ratio_is_24_to_19_l230_230270


namespace mean_proportion_of_3_and_4_l230_230696

theorem mean_proportion_of_3_and_4 : ∃ x : ℝ, 3 / x = x / 4 ∧ (x = 2 * Real.sqrt 3 ∨ x = - (2 * Real.sqrt 3)) :=
by
  sorry

end mean_proportion_of_3_and_4_l230_230696


namespace find_number_l230_230764

theorem find_number 
  (x : ℝ)
  (h : (1 / 10) * x - (1 / 1000) * x = 700) :
  x = 700000 / 99 :=
by 
  sorry

end find_number_l230_230764


namespace symmetric_circle_l230_230864

theorem symmetric_circle
    (x y : ℝ)
    (circle_eq : x^2 + y^2 + 4 * x - 1 = 0) :
    (x - 2)^2 + y^2 = 5 :=
sorry

end symmetric_circle_l230_230864


namespace difference_of_squares_example_product_calculation_factorization_by_completing_square_l230_230949

/-
  Theorem: The transformation in the step \(195 \times 205 = 200^2 - 5^2\) uses the difference of squares formula.
-/

theorem difference_of_squares_example : 
  (195 * 205 = (200 - 5) * (200 + 5)) ∧ ((200 - 5) * (200 + 5) = 200^2 - 5^2) :=
  sorry

/-
  Theorem: Calculate \(9 \times 11 \times 101 \times 10001\) using a simple method.
-/

theorem product_calculation : 
  9 * 11 * 101 * 10001 = 99999999 :=
  sorry

/-
  Theorem: Factorize \(a^2 - 6a + 8\) using the completing the square method.
-/

theorem factorization_by_completing_square (a : ℝ) :
  a^2 - 6 * a + 8 = (a - 2) * (a - 4) :=
  sorry

end difference_of_squares_example_product_calculation_factorization_by_completing_square_l230_230949


namespace curve_three_lines_intersect_at_origin_l230_230126

theorem curve_three_lines_intersect_at_origin (a : ℝ) :
  ((∀ x y : ℝ, (x + 2 * y + a) * (x^2 - y^2) = 0 → 
    ((y = x ∨ y = -x ∨ y = - (1/2) * x - a/2) ∧ 
     (x = 0 ∧ y = 0)))) ↔ a = 0 :=
sorry

end curve_three_lines_intersect_at_origin_l230_230126


namespace quadratic_roots_equation_l230_230293

theorem quadratic_roots_equation (a b c r s : ℝ)
    (h1 : a ≠ 0)
    (h2 : a * r^2 + b * r + c = 0)
    (h3 : a * s^2 + b * s + c = 0) :
    ∃ p q : ℝ, (x^2 - b * x + a * c = 0) ∧ (ar + b, as + b) = (p, q) :=
by
  sorry

end quadratic_roots_equation_l230_230293


namespace complement_union_eq_ge_two_l230_230525

def U : Set ℝ := Set.univ
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

theorem complement_union_eq_ge_two : { x : ℝ | x ≥ 2 } = U \ (M ∪ N) :=
by
  sorry

end complement_union_eq_ge_two_l230_230525


namespace a4_value_a_n_formula_l230_230457

theorem a4_value : a_4 = 30 := 
by
    sorry

noncomputable def a_n (n : ℕ) : ℕ :=
    (n * (n + 1)^2 * (2 * n + 1)) / 6

theorem a_n_formula (n : ℕ) : a_n n = (n * (n + 1)^2 * (2 * n + 1)) / 6 := 
by
    sorry

end a4_value_a_n_formula_l230_230457


namespace toaster_total_cost_l230_230013

theorem toaster_total_cost :
  let MSRP := 30
  let insurance_rate := 0.20
  let premium_upgrade := 7
  let recycling_fee := 5
  let tax_rate := 0.50

  -- Calculate costs
  let insurance_cost := insurance_rate * MSRP
  let total_insurance_cost := insurance_cost + premium_upgrade
  let cost_before_tax := MSRP + total_insurance_cost + recycling_fee
  let state_tax := tax_rate * cost_before_tax
  let total_cost := cost_before_tax + state_tax

  -- Total cost Jon must pay
  total_cost = 72 :=
by
  sorry

end toaster_total_cost_l230_230013


namespace illumination_ways_l230_230203

def ways_to_illuminate_traffic_lights (n : ℕ) : ℕ :=
  3^n

theorem illumination_ways (n : ℕ) : ways_to_illuminate_traffic_lights n = 3 ^ n :=
by
  sorry

end illumination_ways_l230_230203


namespace max_m_value_l230_230161

theorem max_m_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_sum : a + b + c = 12) (h_prod_sum : a * b + b * c + c * a = 35) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m = 3 :=
by
  sorry

end max_m_value_l230_230161


namespace parabola_tangent_line_l230_230370

noncomputable def verify_a_value (a : ℝ) : Prop :=
  ∃ x₀ y₀ : ℝ, (y₀ = a * x₀^2) ∧ (x₀ - y₀ - 1 = 0) ∧ (2 * a * x₀ = 1)

theorem parabola_tangent_line :
  verify_a_value (1 / 4) :=
by
  sorry

end parabola_tangent_line_l230_230370


namespace intersection_is_correct_l230_230619

noncomputable def M : Set ℝ := { x | 1 + x ≥ 0 }
noncomputable def N : Set ℝ := { x | 4 / (1 - x) > 0 }
noncomputable def intersection : Set ℝ := { x | -1 ≤ x ∧ x < 1 }

theorem intersection_is_correct : M ∩ N = intersection := by
  sorry

end intersection_is_correct_l230_230619


namespace halfway_fraction_l230_230415

theorem halfway_fraction (a b : ℚ) (h_a : a = 3 / 4) (h_b : b = 5 / 7) :
  ((a + b) / 2) = 41 / 56 :=
by
  rw [h_a, h_b]
  sorry

end halfway_fraction_l230_230415


namespace riding_time_fraction_l230_230769

-- Definitions for conditions
def M : ℕ := 6
def total_days : ℕ := 6
def max_time_days : ℕ := 2
def part_time_days : ℕ := 2
def fixed_time : ℝ := 1.5
def total_riding_time : ℝ := 21

-- Prove the statement
theorem riding_time_fraction :
  ∃ F : ℝ, 2 * M + 2 * fixed_time + 2 * F * M = total_riding_time ∧ F = 0.5 :=
by
  exists 0.5
  sorry

end riding_time_fraction_l230_230769


namespace batsman_average_increase_l230_230051

theorem batsman_average_increase :
  ∀ (A : ℝ), (10 * A + 110 = 11 * 60) → (60 - A = 5) :=
by
  intros A h
  -- Proof goes here
  sorry

end batsman_average_increase_l230_230051


namespace tiling_possible_if_and_only_if_one_dimension_is_integer_l230_230691

-- Define our conditions: a, b are dimensions of the board and t is the positive dimension of the small rectangles
variable (a b : ℝ) (t : ℝ)

-- Define corresponding properties for these variables
axiom pos_t : t > 0

-- Theorem stating the condition for tiling
theorem tiling_possible_if_and_only_if_one_dimension_is_integer (a_non_int : ¬ ∃ z : ℤ, a = z) (b_non_int : ¬ ∃ z : ℤ, b = z) :
  ∃ n m : ℕ, n * 1 + m * t = a * b :=
sorry

end tiling_possible_if_and_only_if_one_dimension_is_integer_l230_230691


namespace Q_value_ratio_l230_230327

noncomputable def g (x : ℂ) : ℂ := x^2009 + 19*x^2008 + 1

noncomputable def roots : Fin 2009 → ℂ := sorry -- Define distinct roots s1, s2, ..., s2009

noncomputable def Q (z : ℂ) : ℂ := sorry -- Define the polynomial Q of degree 2009

theorem Q_value_ratio :
  (∀ j : Fin 2009, Q (roots j + 2 / roots j) = 0) →
  (Q (2) / Q (-2) = 361 / 400) :=
sorry

end Q_value_ratio_l230_230327


namespace count_good_numbers_l230_230685

theorem count_good_numbers : 
  (∃ (f : Fin 10 → ℕ), ∀ n, 
    (2020 % n = 22 ↔ 
      (n = f 0 ∨ n = f 1 ∨ n = f 2 ∨ n = f 3 ∨ 
       n = f 4 ∨ n = f 5 ∨ n = f 6 ∨ n = f 7 ∨ 
       n = f 8 ∨ n = f 9))) :=
sorry

end count_good_numbers_l230_230685


namespace additional_distance_to_achieve_target_average_speed_l230_230304

-- Given conditions
def initial_distance : ℕ := 20
def initial_speed : ℕ := 40
def target_average_speed : ℕ := 55

-- Prove that the additional distance required to average target speed is 90 miles
theorem additional_distance_to_achieve_target_average_speed 
  (total_distance : ℕ) 
  (total_time : ℚ) 
  (additional_distance : ℕ) 
  (additional_speed : ℕ) :
  total_distance = initial_distance + additional_distance →
  total_time = (initial_distance / initial_speed) + (additional_distance / additional_speed) →
  additional_speed = 60 →
  total_distance / total_time = target_average_speed →
  additional_distance = 90 :=
by 
  sorry

end additional_distance_to_achieve_target_average_speed_l230_230304


namespace total_employees_l230_230766

variable (E : ℕ)
variable (employees_prefer_X employees_prefer_Y number_of_prefers : ℕ)
variable (X_percentage Y_percentage : ℝ)

-- Conditions based on the problem
axiom prefer_X : X_percentage = 0.60
axiom prefer_Y : Y_percentage = 0.40
axiom max_preference_relocation : number_of_prefers = 140

-- Defining the total number of employees who prefer city X or Y and get relocated accordingly:
axiom equation : X_percentage * E + Y_percentage * E = number_of_prefers

-- The theorem we are proving
theorem total_employees : E = 140 :=
by
  -- Proof placeholder
  sorry

end total_employees_l230_230766


namespace coeff_a_zero_l230_230878

theorem coeff_a_zero
  (a b c : ℝ)
  (h : ∀ p : ℝ, 0 < p → ∀ (x : ℝ), (a * x^2 + b * x + c + p = 0) → x > 0) :
  a = 0 :=
sorry

end coeff_a_zero_l230_230878


namespace simplify_expression_l230_230513

theorem simplify_expression :
  (2 * 6 / (12 * 14)) * (3 * 12 * 14 / (2 * 6 * 3)) * 2 = 2 := 
  sorry

end simplify_expression_l230_230513


namespace no_solution_exists_l230_230577

theorem no_solution_exists :
  ¬ ∃ x : ℝ, (x - 2) / (x + 2) - 16 / (x^2 - 4) = (x + 2) / (x - 2) :=
by sorry

end no_solution_exists_l230_230577


namespace vector_magnitude_l230_230149

noncomputable def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
(v1.1 + v2.1, v1.2 + v2.2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem vector_magnitude :
  ∀ (x y : ℝ), let a := (x, 2)
               let b := (1, y)
               let c := (2, -6)
               (a.1 * c.1 + a.2 * c.2 = 0) →
               (b.1 * (-c.2) - b.2 * c.1 = 0) →
               magnitude (vec_add a b) = 5 * Real.sqrt 2 :=
by
  intros x y a b c h₁ h₂
  let a := (x, 2)
  let b := (1, y)
  let c := (2, -6)
  sorry

end vector_magnitude_l230_230149


namespace complex_values_l230_230467

open Complex

theorem complex_values (a b : ℝ) (i : ℂ) (h1 : i = Complex.I) (h2 : a - b * i = (1 + i) * i^3) : a = 1 ∧ b = -1 :=
by
  sorry

end complex_values_l230_230467


namespace sin_cos_identity_l230_230383

theorem sin_cos_identity (θ : ℝ) (h : Real.tan (θ + (Real.pi / 4)) = 2) : 
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = -7/5 := 
by 
  sorry

end sin_cos_identity_l230_230383


namespace find_a_l230_230326

noncomputable def circle1 (x y : ℝ) := x^2 + y^2 + 4 * y = 0

noncomputable def circle2 (x y a : ℝ) := x^2 + y^2 + 2 * (a - 1) * x + 2 * y + a^2 = 0

theorem find_a (a : ℝ) :
  (∀ x y, circle1 x y → circle2 x y a → false) → a = -2 :=
by sorry

end find_a_l230_230326


namespace saved_percent_l230_230763

-- Definitions for conditions:
def last_year_saved (S : ℝ) : ℝ := 0.10 * S
def this_year_salary (S : ℝ) : ℝ := 1.10 * S
def this_year_saved (S : ℝ) : ℝ := 0.06 * (1.10 * S)

-- Given conditions and proof goal:
theorem saved_percent (S : ℝ) (hl_last_year_saved : last_year_saved S = 0.10 * S)
  (hl_this_year_salary : this_year_salary S = 1.10 * S)
  (hl_this_year_saved : this_year_saved S = 0.066 * S) :
  (this_year_saved S / last_year_saved S) * 100 = 66 :=
by
  sorry

end saved_percent_l230_230763


namespace max_cards_possible_l230_230670

-- Define the dimensions for the cardboard and the card.
def cardboard_length : ℕ := 48
def cardboard_width : ℕ := 36
def card_length : ℕ := 16
def card_width : ℕ := 12

-- State the theorem to prove the maximum number of cards.
theorem max_cards_possible : (cardboard_length / card_length) * (cardboard_width / card_width) = 9 :=
by
  sorry -- Skip the proof, as only the statement is required.

end max_cards_possible_l230_230670


namespace sum_of_cubes_identity_l230_230100

theorem sum_of_cubes_identity (a b : ℝ) (h : a / (1 + b) + b / (1 + a) = 1) : a^3 + b^3 = a + b := by
  sorry

end sum_of_cubes_identity_l230_230100


namespace not_right_triangle_D_right_triangle_A_right_triangle_B_right_triangle_C_l230_230180

def right_angle_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem not_right_triangle_D (a b c : ℝ):
  ¬ (a = 3^2 ∧ b = 4^2 ∧ c = 5^2 ∧ right_angle_triangle a b c) :=
sorry

theorem right_triangle_A (a b c x : ℝ):
  a = 5 * x → b = 12 * x → c = 13 * x → x > 0 → right_angle_triangle a b c :=
sorry

theorem right_triangle_B (angleA angleB angleC : ℝ):
  angleA / angleB / angleC = 2 / 3 / 5 → angleC = 90 → angleA + angleB + angleC = 180 → right_angle_triangle angleA angleB angleC :=
sorry

theorem right_triangle_C (a b c k : ℝ):
  a = 9 * k → b = 40 * k → c = 41 * k → k > 0 → right_angle_triangle a b c :=
sorry

end not_right_triangle_D_right_triangle_A_right_triangle_B_right_triangle_C_l230_230180


namespace largest_N_l230_230231

-- Definition of the problem conditions
def problem_conditions (n : ℕ) (N : ℕ) (a : Fin (N + 1) → ℝ) : Prop :=
  (n ≥ 2) ∧
  (a 0 + a 1 = -(1 : ℝ) / n) ∧  
  (∀ k : ℕ, 1 ≤ k → k ≤ N - 1 → (a k + a (k - 1)) * (a k + a (k + 1)) = a (k - 1) - a (k + 1))

-- The theorem stating that the largest integer N is n
theorem largest_N (n : ℕ) (N : ℕ) (a : Fin (N + 1) → ℝ) :
  problem_conditions n N a → N = n :=
sorry

end largest_N_l230_230231


namespace solveSALE_l230_230433

namespace Sherlocked

open Nat

def areDistinctDigits (d₁ d₂ d₃ d₄ d₅ d₆ : Nat) : Prop :=
  d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₁ ≠ d₅ ∧ d₁ ≠ d₆ ∧ 
  d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₂ ≠ d₅ ∧ d₂ ≠ d₆ ∧ 
  d₃ ≠ d₄ ∧ d₃ ≠ d₅ ∧ d₃ ≠ d₆ ∧ 
  d₄ ≠ d₅ ∧ d₄ ≠ d₆ ∧ 
  d₅ ≠ d₆

theorem solveSALE :
  ∃ (S C A L E T : ℕ),
    SCALE - SALE = SLATE ∧
    areDistinctDigits S C A L E T ∧
    S < 10 ∧ C < 10 ∧ A < 10 ∧
    L < 10 ∧ E < 10 ∧ T < 10 ∧
    SALE = 1829 :=
by
  sorry

end Sherlocked

end solveSALE_l230_230433


namespace percentage_taken_l230_230704

theorem percentage_taken (P : ℝ) (h : (P / 100) * 150 - 40 = 50) : P = 60 :=
by
  sorry

end percentage_taken_l230_230704


namespace find_sum_of_digits_in_base_l230_230406

theorem find_sum_of_digits_in_base (d A B : ℕ) (hd : d > 8) (hA : A < d) (hB : B < d) (h : (A * d + B) + (A * d + A) - (B * d + A) = 1 * d^2 + 8 * d + 0) : A + B = 10 :=
sorry

end find_sum_of_digits_in_base_l230_230406


namespace gear_revolutions_difference_l230_230784

noncomputable def gear_revolution_difference (t : ℕ) : ℕ :=
  let p := 10 * t
  let q := 40 * t
  q - p

theorem gear_revolutions_difference (t : ℕ) : gear_revolution_difference t = 30 * t :=
by
  sorry

end gear_revolutions_difference_l230_230784


namespace remainder_6n_mod_4_l230_230463

theorem remainder_6n_mod_4 (n : ℕ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := by
  sorry

end remainder_6n_mod_4_l230_230463


namespace find_number_l230_230041

theorem find_number (x : ℝ) (h : (1/4) * x = (1/5) * (x + 1) + 1) : x = 24 := 
sorry

end find_number_l230_230041


namespace inverse_proportion_l230_230609

theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x = 3) (h3 : y = 15) (h4 : y = -30) : x = -3 / 2 :=
by
  sorry

end inverse_proportion_l230_230609


namespace inequality_solution_l230_230418

theorem inequality_solution (x : ℝ) : (x < -4 ∨ x > -4) → (x + 3) / (x + 4) > (2 * x + 7) / (3 * x + 12) :=
by
  intro h
  sorry

end inequality_solution_l230_230418


namespace Derek_is_42_l230_230724

def Aunt_Anne_age : ℕ := 36

def Brianna_age : ℕ := (2 * Aunt_Anne_age) / 3

def Caitlin_age : ℕ := Brianna_age - 3

def Derek_age : ℕ := 2 * Caitlin_age

theorem Derek_is_42 : Derek_age = 42 := by
  sorry

end Derek_is_42_l230_230724


namespace intersection_A_B_l230_230510

def A : Set ℝ := { x | x * Real.sqrt (x^2 - 4) ≥ 0 }
def B : Set ℝ := { x | |x - 1| + |x + 1| ≥ 2 }

theorem intersection_A_B : (A ∩ B) = ({-2} ∪ Set.Ici 2) :=
by
  sorry

end intersection_A_B_l230_230510


namespace find_t_l230_230774

-- Definitions of the vectors and parallel condition
def a : ℝ × ℝ := (-1, 1)
def b (t : ℝ) : ℝ × ℝ := (3, t)
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = k • v ∨ v = k • u

-- The theorem statement
theorem find_t (t : ℝ) (h : is_parallel (b t) (a + b t)) : t = -3 := by
  sorry

end find_t_l230_230774


namespace product_of_second_and_fourth_term_l230_230660

theorem product_of_second_and_fourth_term (a : ℕ → ℤ) (d : ℤ) (h₁ : a 10 = 25) (h₂ : d = 3)
  (h₃ : ∀ n, a n = a 1 + (n - 1) * d) : a 2 * a 4 = 7 :=
by
  -- Assuming necessary conditions are defined
  sorry

end product_of_second_and_fourth_term_l230_230660


namespace root_k_value_l230_230780

theorem root_k_value
  (k : ℝ)
  (h : Polynomial.eval 4 (Polynomial.C 2 * Polynomial.X^2 + Polynomial.C 3 * Polynomial.X - Polynomial.C k) = 0) :
  k = 44 :=
sorry

end root_k_value_l230_230780


namespace speed_ratio_l230_230286

variable (vA vB : ℝ)
variable (H1 : 3 * vA = abs (-400 + 3 * vB))
variable (H2 : 10 * vA = abs (-400 + 10 * vB))

theorem speed_ratio (vA vB : ℝ) (H1 : 3 * vA = abs (-400 + 3 * vB)) (H2 : 10 * vA = abs (-400 + 10 * vB)) : 
  vA / vB = 5 / 6 :=
  sorry

end speed_ratio_l230_230286


namespace problem_l230_230779

theorem problem (a b c : ℂ) 
  (h1 : a + b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 3)
  (h3 : a^3 + b^3 + c^3 = 6) :
  (a - 1)^(2023) + (b - 1)^(2023) + (c - 1)^(2023) = 0 :=
by
  sorry

end problem_l230_230779


namespace stationary_train_length_l230_230995

-- Definitions
def speed_km_per_h := 72
def speed_m_per_s := speed_km_per_h * (1000 / 3600) -- conversion from km/h to m/s
def time_to_pass_pole := 10 -- in seconds
def time_to_cross_stationary_train := 35 -- in seconds
def speed := 20 -- speed in m/s, 72 km/h = 20 m/s, can be inferred from conversion

-- Length of moving train
def length_of_moving_train := speed * time_to_pass_pole

-- Total distance in crossing stationary train
def total_distance := speed * time_to_cross_stationary_train

-- Length of stationary train
def length_of_stationary_train := total_distance - length_of_moving_train

-- Proof statement
theorem stationary_train_length :
  length_of_stationary_train = 500 := by
  sorry

end stationary_train_length_l230_230995


namespace find_area_triangle_boc_l230_230714

noncomputable def area_ΔBOC := 21

theorem find_area_triangle_boc (A B C K O : Type) 
  [NormedAddCommGroup A] [NormedAddCommGroup B] [NormedAddCommGroup C] [NormedAddCommGroup K] [NormedAddCommGroup O]
  (AC : ℝ) (AB : ℝ) (h1 : AC = 14) (h2 : AB = 6)
  (circle_centered_on_AC : Prop)
  (K_on_BC : Prop)
  (angle_BAK_eq_angle_ACB : Prop)
  (midpoint_O_AC : Prop)
  (angle_AKC_eq_90 : Prop)
  (area_ABC : Prop) : 
  area_ΔBOC = 21 := 
sorry

end find_area_triangle_boc_l230_230714


namespace angle_B_l230_230882

open Set

variables {Point Line : Type}

variable (l m n p : Line)
variable (A B C D : Point)
variable (angle : Point → Point → Point → ℝ)

-- Definitions of the conditions
def parallel (x y : Line) : Prop := sorry
def intersects (x y : Line) (P : Point) : Prop := sorry
def measure_angle (P Q R : Point) : ℝ := sorry

-- Assumptions based on conditions
axiom parallel_lm : parallel l m
axiom intersection_n_l : intersects n l A
axiom angle_A : measure_angle B A D = 140
axiom intersection_p_m : intersects p m C
axiom angle_C : measure_angle A C B = 70
axiom intersection_p_l : intersects p l D
axiom not_parallel_np : ¬ parallel n p

-- Proof goal
theorem angle_B : measure_angle C B D = 140 := sorry

end angle_B_l230_230882


namespace tom_distance_before_karen_wins_l230_230908

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

end tom_distance_before_karen_wins_l230_230908


namespace sum_of_primes_less_than_twenty_is_77_l230_230777

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l230_230777


namespace sum_of_squares_l230_230778

theorem sum_of_squares (a b : ℝ) (h1 : a + b = 16) (h2 : a * b = 20) : a^2 + b^2 = 216 :=
by
  sorry

end sum_of_squares_l230_230778


namespace second_number_l230_230182

theorem second_number (A B : ℝ) (h1 : A = 200) (h2 : 0.30 * A = 0.60 * B + 30) : B = 50 :=
by
  -- proof goes here
  sorry

end second_number_l230_230182


namespace train_speed_l230_230078

/--
Given:
- The speed of the first person \(V_p\) is 4 km/h.
- The train takes 9 seconds to pass the first person completely.
- The length of the train is approximately 50 meters (49.999999999999986 meters).

Prove:
- The speed of the train \(V_t\) is 24 km/h.
-/
theorem train_speed (V_p : ℝ) (t : ℝ) (L : ℝ) (V_t : ℝ) 
  (hV_p : V_p = 4) 
  (ht : t = 9)
  (hL : L = 49.999999999999986)
  (hrel_speed : (L / t) * 3.6 = V_t - V_p) :
  V_t = 24 :=
by
  sorry

end train_speed_l230_230078


namespace horizontal_distance_travel_l230_230456

noncomputable def radius : ℝ := 2
noncomputable def angle_degrees : ℝ := 30
noncomputable def angle_radians : ℝ := angle_degrees * (Real.pi / 180)
noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
noncomputable def cos_theta : ℝ := Real.cos angle_radians
noncomputable def horizontal_distance (r : ℝ) (θ : ℝ) : ℝ := (circumference r) * (Real.cos θ)

theorem horizontal_distance_travel (r : ℝ) (θ : ℝ) (h_radius : r = 2) (h_angle : θ = angle_radians) :
  horizontal_distance r θ = 2 * Real.pi * Real.sqrt 3 := 
by
  sorry

end horizontal_distance_travel_l230_230456


namespace condition_sufficient_but_not_necessary_l230_230348

variable (a b : ℝ)

theorem condition_sufficient_but_not_necessary :
  (|a| < 1 ∧ |b| < 1) → (|1 - a * b| > |a - b|) ∧
  ((|1 - a * b| > |a - b|) → (|a| < 1 ∧ |b| < 1) ∨ (|a| ≥ 1 ∧ |b| ≥ 1)) :=
by
  sorry

end condition_sufficient_but_not_necessary_l230_230348


namespace discount_correct_l230_230436

-- Define the prices of items and the total amount paid
def t_shirt_price : ℕ := 30
def backpack_price : ℕ := 10
def cap_price : ℕ := 5
def total_paid : ℕ := 43

-- Define the total cost before discount
def total_cost := t_shirt_price + backpack_price + cap_price

-- Define the discount
def discount := total_cost - total_paid

-- Prove that the discount is 2 dollars
theorem discount_correct : discount = 2 :=
by
  -- We need to prove that (30 + 10 + 5) - 43 = 2
  sorry

end discount_correct_l230_230436


namespace other_employee_number_l230_230135

-- Define the conditions
variables (total_employees : ℕ) (sample_size : ℕ) (e1 e2 e3 : ℕ)

-- Define the systematic sampling interval
def sampling_interval (total : ℕ) (size : ℕ) : ℕ := total / size

-- The Lean statement for the proof problem
theorem other_employee_number
  (h1 : total_employees = 52)
  (h2 : sample_size = 4)
  (h3 : e1 = 6)
  (h4 : e2 = 32)
  (h5 : e3 = 45) :
  ∃ e4 : ℕ, e4 = 19 := 
sorry

end other_employee_number_l230_230135


namespace sqrt_inequality_l230_230380

theorem sqrt_inequality (x : ℝ) (h₁ : 3 / 2 ≤ x) (h₂ : x ≤ 5) : 
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := 
sorry

end sqrt_inequality_l230_230380


namespace tiles_needed_correct_l230_230853

noncomputable def tiles_needed (floor_length : ℝ) (floor_width : ℝ) (tile_length_inch : ℝ) (tile_width_inch : ℝ) (border_width : ℝ) : ℝ :=
  let tile_length := tile_length_inch / 12
  let tile_width := tile_width_inch / 12
  let main_length := floor_length - 2 * border_width
  let main_width := floor_width - 2 * border_width
  let main_area := main_length * main_width
  let tile_area := tile_length * tile_width
  main_area / tile_area

theorem tiles_needed_correct :
  tiles_needed 15 20 3 9 1 = 1248 := 
by 
  sorry -- Proof skipped.

end tiles_needed_correct_l230_230853


namespace binomial_7_4_l230_230356

noncomputable def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_7_4 : binomial 7 4 = 35 := by
  sorry

end binomial_7_4_l230_230356


namespace area_of_hexagon_l230_230340

theorem area_of_hexagon (c d : ℝ) (a b : ℝ)
  (h₁ : a^2 + b^2 = c^2)
  (h₂ : a + b = d) : 
  (c^2 + d^2 = c^2 + a^2 + b^2 + 2*a*b) :=
by
  sorry

end area_of_hexagon_l230_230340


namespace ratio_of_inscribed_squares_l230_230768

open Real

-- Condition: A square inscribed in a right triangle with sides 3, 4, and 5
def inscribedSquareInRightTriangle1 (x : ℝ) (a b c : ℝ) : Prop :=
  a = 3 ∧ b = 4 ∧ c = 5 ∧ x = 12 / 7

-- Condition: A square inscribed in a different right triangle with sides 5, 12, and 13
def inscribedSquareInRightTriangle2 (y : ℝ) (d e f : ℝ) : Prop :=
  d = 5 ∧ e = 12 ∧ f = 13 ∧ y = 169 / 37

-- The ratio x / y is 444 / 1183
theorem ratio_of_inscribed_squares (x y : ℝ) (a b c d e f : ℝ) :
  inscribedSquareInRightTriangle1 x a b c →
  inscribedSquareInRightTriangle2 y d e f →
  x / y = 444 / 1183 :=
by
  intros h1 h2
  sorry

end ratio_of_inscribed_squares_l230_230768


namespace intersection_A_B_l230_230891

def A := {x : ℤ | ∃ k : ℤ, x = 2 * k + 1}
def B := {x : ℤ | 0 < x ∧ x < 5}

theorem intersection_A_B : A ∩ B = {1, 3} :=
by
  sorry

end intersection_A_B_l230_230891


namespace animal_shelter_dogs_l230_230134

theorem animal_shelter_dogs (D C R : ℕ) 
  (h₁ : 15 * C = 7 * D)
  (h₂ : 15 * R = 4 * D)
  (h₃ : 15 * (C + 20) = 11 * D)
  (h₄ : 15 * (R + 10) = 6 * D) : 
  D = 75 :=
by
  -- Proof part is omitted
  sorry

end animal_shelter_dogs_l230_230134


namespace no_solutions_in_natural_numbers_l230_230967

theorem no_solutions_in_natural_numbers (x y : ℕ) : x^2 + x * y + y^2 ≠ x^2 * y^2 :=
  sorry

end no_solutions_in_natural_numbers_l230_230967


namespace triangle_area_l230_230424

theorem triangle_area (a b c : ℝ) (C : ℝ) (h1 : c^2 = (a - b)^2 + 6) (h2 : C = π / 3) :
    abs ((1 / 2) * a * b * Real.sin C) = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end triangle_area_l230_230424


namespace faster_ship_speed_l230_230746

theorem faster_ship_speed :
  ∀ (x y : ℕ),
    (200 + 100 = 300) → -- Total distance covered for both directions
    (x + y) * 10 = 300 → -- Opposite direction equation
    (x - y) * 25 = 300 → -- Same direction equation
    x = 21 := 
by
  intros x y _ eq1 eq2
  sorry

end faster_ship_speed_l230_230746


namespace final_cost_cooking_gear_sets_l230_230881

-- Definitions based on conditions
def hand_mitts_cost : ℕ := 14
def apron_cost : ℕ := 16
def utensils_cost : ℕ := 10
def knife_cost : ℕ := 2 * utensils_cost
def discount_rate : ℚ := 0.25
def sales_tax_rate : ℚ := 0.08
def number_of_recipients : ℕ := 3 + 5

-- Proof statement: calculate the final cost
theorem final_cost_cooking_gear_sets :
  let total_cost_before_discount := hand_mitts_cost + apron_cost + utensils_cost + knife_cost
  let discounted_cost_per_set := (total_cost_before_discount : ℚ) * (1 - discount_rate)
  let total_cost_for_recipients := (discounted_cost_per_set * number_of_recipients : ℚ)
  let final_cost := total_cost_for_recipients * (1 + sales_tax_rate)
  final_cost = 388.80 :=
by
  sorry

end final_cost_cooking_gear_sets_l230_230881


namespace total_photos_l230_230261

-- Define the number of photos Claire has taken
def photos_by_Claire : ℕ := 8

-- Define the number of photos Lisa has taken
def photos_by_Lisa : ℕ := 3 * photos_by_Claire

-- Define the number of photos Robert has taken
def photos_by_Robert : ℕ := photos_by_Claire + 16

-- State the theorem we want to prove
theorem total_photos : photos_by_Lisa + photos_by_Robert = 48 :=
by
  sorry

end total_photos_l230_230261


namespace sum_of_ages_is_55_l230_230627

def sum_of_ages (Y : ℕ) (interval : ℕ) (number_of_children : ℕ) : ℕ :=
  let ages := List.range number_of_children |>.map (λ i => Y + i * interval)
  ages.sum

theorem sum_of_ages_is_55 :
  sum_of_ages 7 2 5 = 55 :=
by
  sorry

end sum_of_ages_is_55_l230_230627


namespace cubic_identity_l230_230585

theorem cubic_identity (x y z : ℝ) (h1 : x + y + z = 13) (h2 : xy + xz + yz = 32) :
  x^3 + y^3 + z^3 - 3 * x * y * z = 949 :=
by
  sorry

end cubic_identity_l230_230585


namespace percent_not_filler_l230_230159

theorem percent_not_filler (sandwich_weight filler_weight : ℕ) (h_sandwich : sandwich_weight = 180) (h_filler : filler_weight = 45) : 
  (sandwich_weight - filler_weight) * 100 / sandwich_weight = 75 :=
by
  -- proof here
  sorry

end percent_not_filler_l230_230159


namespace arithmetic_progression_15th_term_l230_230855

theorem arithmetic_progression_15th_term :
  let a := 2
  let d := 3
  let n := 15
  a + (n - 1) * d = 44 :=
by
  let a := 2
  let d := 3
  let n := 15
  sorry

end arithmetic_progression_15th_term_l230_230855


namespace initial_erasers_calculation_l230_230813

variable (initial_erasers added_erasers total_erasers : ℕ)

theorem initial_erasers_calculation
  (total_erasers_eq : total_erasers = 270)
  (added_erasers_eq : added_erasers = 131) :
  initial_erasers = total_erasers - added_erasers → initial_erasers = 139 := by
  intro h
  rw [total_erasers_eq, added_erasers_eq] at h
  simp at h
  exact h

end initial_erasers_calculation_l230_230813


namespace probability_is_stable_frequency_l230_230417

/-- Definition of probability: the stable theoretical value reflecting the likelihood of event occurrence. -/
def probability (event : Type) : ℝ := sorry 

/-- Definition of frequency: the empirical count of how often an event occurs in repeated experiments. -/
def frequency (event : Type) (trials : ℕ) : ℝ := sorry 

/-- The statement that "probability is the stable value of frequency" is correct. -/
theorem probability_is_stable_frequency (event : Type) (trials : ℕ) :
  probability event = sorry ↔ true := 
by 
  -- This is where the proof would go, but is replaced with sorry for now. 
  sorry

end probability_is_stable_frequency_l230_230417


namespace find_a_if_even_function_l230_230957

-- Problem statement in Lean 4
theorem find_a_if_even_function (a : ℝ) (f : ℝ → ℝ) 
  (h : ∀ x, f x = x^2 - 2 * (a + 1) * x + 1) 
  (hf_even : ∀ x, f x = f (-x)) : a = -1 :=
sorry

end find_a_if_even_function_l230_230957


namespace fraction_simplification_l230_230171

theorem fraction_simplification 
  (d e f : ℝ) 
  (h : d + e + f ≠ 0) : 
  (d^2 + e^2 - f^2 + 2 * d * e) / (d^2 + f^2 - e^2 + 3 * d * f) = (d + e - f) / (d + f - e) :=
sorry

end fraction_simplification_l230_230171


namespace linear_dependence_k_l230_230090

theorem linear_dependence_k :
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ 
    (a * (2 : ℝ) + b * (5 : ℝ) = 0) ∧ 
    (a * (3 : ℝ) + b * k = 0) →
  k = 15 / 2 := by
  sorry

end linear_dependence_k_l230_230090


namespace rationalize_and_subtract_l230_230754

theorem rationalize_and_subtract :
  (7 / (3 + Real.sqrt 15)) * (3 - Real.sqrt 15) / (3^2 - (Real.sqrt 15)^2) 
  - (1 / 2) = -4 + (7 * Real.sqrt 15) / 6 :=
by
  sorry

end rationalize_and_subtract_l230_230754


namespace weight_of_B_l230_230200

/-- Let A, B, and C be the weights in kg of three individuals. If the average weight of A, B, and C is 45 kg,
and the average weight of A and B is 41 kg, and the average weight of B and C is 43 kg,
then the weight of B is 33 kg. -/
theorem weight_of_B (A B C : ℝ) 
  (h1 : A + B + C = 135) 
  (h2 : A + B = 82) 
  (h3 : B + C = 86) : 
  B = 33 := 
by 
  sorry

end weight_of_B_l230_230200


namespace ellipse_eccentricity_l230_230867

-- Define the geometric sequence condition and the ellipse properties
theorem ellipse_eccentricity :
  ∀ (a b c e : ℝ), 
  (b^2 = a * c) ∧ (a^2 - c^2 = b^2) ∧ (e = c / a) ∧ (0 < e ∧ e < 1) →
  e = (Real.sqrt 5 - 1) / 2 := 
by 
  sorry

end ellipse_eccentricity_l230_230867


namespace simplify_expression_l230_230708

variables {K : Type*} [Field K]

theorem simplify_expression (a b c : K) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b) : 
    (a^3 - b^3) / (a * b) - (a * b - c * b) / (a * b - a^2) = (a^2 + a * b + b^2 + a * c) / (a * b) :=
by
  sorry

end simplify_expression_l230_230708


namespace prove_statement_II_must_be_true_l230_230148

-- Definitions of the statements
def statement_I (d : ℕ) : Prop := d = 5
def statement_II (d : ℕ) : Prop := d ≠ 6
def statement_III (d : ℕ) : Prop := d = 7
def statement_IV (d : ℕ) : Prop := d ≠ 8

-- Condition: Exactly three of these statements are true and one is false
def exactly_three_true (P Q R S : Prop) : Prop :=
  (P ∧ Q ∧ R ∧ ¬S) ∨ (P ∧ Q ∧ ¬R ∧ S) ∨ (P ∧ ¬Q ∧ R ∧ S) ∨ (¬P ∧ Q ∧ R ∧ S)

-- Problem statement
theorem prove_statement_II_must_be_true (d : ℕ) (h : exactly_three_true (statement_I d) (statement_II d) (statement_III d) (statement_IV d)) : 
  statement_II d :=
by
  -- proof goes here
  sorry

end prove_statement_II_must_be_true_l230_230148


namespace largest_pos_integer_binary_op_l230_230240

def binary_op (n : ℤ) : ℤ := n - n * 5

theorem largest_pos_integer_binary_op :
  ∃ n : ℕ, binary_op n < 14 ∧ ∀ m : ℕ, binary_op m < 14 → m ≤ 1 :=
sorry

end largest_pos_integer_binary_op_l230_230240


namespace greatest_integer_c_l230_230876

theorem greatest_integer_c (c : ℤ) :
  (∀ x : ℝ, x^2 + (c : ℝ) * x + 10 ≠ 0) → c = 6 :=
by
  sorry

end greatest_integer_c_l230_230876


namespace num_dislikers_tv_books_games_is_correct_l230_230614

-- Definitions of the conditions as given in step A
def total_people : ℕ := 1500
def pct_dislike_tv : ℝ := 0.4
def pct_dislike_tv_books : ℝ := 0.15
def pct_dislike_tv_books_games : ℝ := 0.5

-- Calculate intermediate values
def num_tv_dislikers := pct_dislike_tv * total_people
def num_tv_books_dislikers := pct_dislike_tv_books * num_tv_dislikers
def num_tv_books_games_dislikers := pct_dislike_tv_books_games * num_tv_books_dislikers

-- Final proof statement ensuring the correctness of the solution
theorem num_dislikers_tv_books_games_is_correct :
  num_tv_books_games_dislikers = 45 := by
  -- Sorry placeholder for the proof. In actual Lean usage, this would require fulfilling the proof obligations.
  sorry

end num_dislikers_tv_books_games_is_correct_l230_230614


namespace certain_event_l230_230693

-- Definitions of the events as propositions
def EventA : Prop := ∃ n : ℕ, n ≥ 1 ∧ (n % 2 = 0)
def EventB : Prop := ∃ t : ℝ, t ≥ 0  -- Simplifying as the event of an advertisement airing
def EventC : Prop := ∃ w : ℕ, w ≥ 1  -- Simplifying as the event of rain in Weinan on a specific future date
def EventD : Prop := true  -- The sun rises from the east in the morning is always true

-- The statement that Event D is the only certain event among the given options
theorem certain_event : EventD ∧ ¬EventA ∧ ¬EventB ∧ ¬EventC :=
by
  sorry

end certain_event_l230_230693


namespace largest_6_digit_div_by_88_l230_230811

theorem largest_6_digit_div_by_88 : ∃ n : ℕ, 100000 ≤ n ∧ n ≤ 999999 ∧ 88 ∣ n ∧ (∀ m : ℕ, 100000 ≤ m ∧ m ≤ 999999 ∧ 88 ∣ m → m ≤ n) ∧ n = 999944 :=
by
  sorry

end largest_6_digit_div_by_88_l230_230811


namespace find_b_value_l230_230807

theorem find_b_value
  (b : ℝ)
  (eq1 : ∀ y x, 3 * y - 3 * b = 9 * x)
  (eq2 : ∀ y x, y - 2 = (b + 9) * x)
  (parallel : ∀ y1 y2 x1 x2, 
    (3 * y1 - 3 * b = 9 * x1) ∧ (y2 - 2 = (b + 9) * x2) → 
    ((3 * x1 = (b + 9) * x2) ↔ (3 = b + 9)))
  : b = -6 := 
  sorry

end find_b_value_l230_230807


namespace banana_price_l230_230396

theorem banana_price (x y : ℕ) (b : ℕ) 
  (hx : x + y = 4) 
  (cost_eq : 50 * x + 60 * y + b = 275) 
  (banana_cheaper_than_pear : b < 60) 
  : b = 35 ∨ b = 45 ∨ b = 55 :=
by
  sorry

end banana_price_l230_230396


namespace part1_part2_l230_230600

-- Define the conditions for part (1)
def nonEmptyBoxes := ∀ i j k: Nat, (i ≠ j ∧ i ≠ k ∧ j ≠ k)
def ball3inBoxB := ∀ (b3: Nat) (B: Nat), b3 = 3 ∧ B > 0

-- Define the conditions for part (2)
def ball1notInBoxA := ∀ (b1: Nat) (A: Nat), b1 ≠ 1 ∧ A > 0
def ball2notInBoxB := ∀ (b2: Nat) (B: Nat), b2 ≠ 2 ∧ B > 0

-- Theorems to be proved
theorem part1 (h1: nonEmptyBoxes) (h2: ball3inBoxB) : ∃ n, n = 12 := by sorry

theorem part2 (h3: ball1notInBoxA) (h4: ball2notInBoxB) : ∃ n, n = 36 := by sorry

end part1_part2_l230_230600


namespace total_eggs_today_l230_230357

def eggs_morning : ℕ := 816
def eggs_afternoon : ℕ := 523

theorem total_eggs_today : eggs_morning + eggs_afternoon = 1339 :=
by {
  sorry
}

end total_eggs_today_l230_230357


namespace gcd_pow_sub_one_l230_230638

theorem gcd_pow_sub_one (m n : ℕ) (h1 : m = 2^2024 - 1) (h2 : n = 2^2000 - 1) : Nat.gcd m n = 2^24 - 1 := 
by
  sorry

end gcd_pow_sub_one_l230_230638


namespace coordinate_plane_condition_l230_230320

theorem coordinate_plane_condition (a : ℝ) :
  a - 1 < 0 ∧ (3 * a + 1) / (a - 1) < 0 ↔ - (1 : ℝ)/3 < a ∧ a < 1 :=
by
  sorry

end coordinate_plane_condition_l230_230320


namespace class_groups_l230_230605

open Nat

def combinations (n k : ℕ) : ℕ :=
  n.choose k

theorem class_groups (boys girls : ℕ) (group_size : ℕ) :
  boys = 9 → girls = 12 → group_size = 3 →
  (combinations boys 1 * combinations girls 2) + (combinations boys 2 * combinations girls 1) = 1026 :=
by
  intros
  sorry

end class_groups_l230_230605


namespace fraction_sum_l230_230828

theorem fraction_sum : (1 / 4 : ℚ) + (3 / 8) = 5 / 8 :=
by
  sorry

end fraction_sum_l230_230828


namespace roots_equal_implies_a_eq_3_l230_230193

theorem roots_equal_implies_a_eq_3 (x a : ℝ) (h1 : 3 * x - 2 * a = 0) (h2 : 2 * x + 3 * a - 13 = 0) : a = 3 :=
sorry

end roots_equal_implies_a_eq_3_l230_230193


namespace factoring_difference_of_squares_l230_230223

theorem factoring_difference_of_squares (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) := 
sorry

end factoring_difference_of_squares_l230_230223


namespace max_omega_l230_230339

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x)

theorem max_omega :
  (∃ ω > 0, (∃ k : ℤ, (f ω (2 * π / 3) = 0) ∧ (ω = 3 / 2 * k)) ∧ (0 < ω * π / 14 ∧ ω * π / 14 ≤ π / 2)) →
  ∃ ω, ω = 6 :=
by
  sorry

end max_omega_l230_230339


namespace percentage_reduction_is_20_percent_l230_230174

-- Defining the initial and final prices
def initial_price : ℝ := 25
def final_price : ℝ := 16

-- Defining the percentage reduction
def percentage_reduction (x : ℝ) := 1 - x

-- The equation representing the two reductions:
def equation (x : ℝ) := initial_price * (percentage_reduction x) * (percentage_reduction x)

theorem percentage_reduction_is_20_percent :
  ∃ x, 0 ≤ x ∧ x ≤ 1 ∧ equation x = final_price ∧ x = 0.20 :=
by 
  sorry

end percentage_reduction_is_20_percent_l230_230174


namespace f_neg_one_f_eq_half_l230_230458

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then 2^(-x) else Real.log x / Real.log 2

theorem f_neg_one : f (-1) = 2 := by
  sorry

theorem f_eq_half (x : ℝ) : f x = 1 / 2 ↔ x = Real.sqrt 2 := by
  sorry

end f_neg_one_f_eq_half_l230_230458


namespace optimal_bicycle_point_l230_230093

noncomputable def distance_A_B : ℝ := 30  -- Distance between A and B is 30 km
noncomputable def midpoint_distance : ℝ := distance_A_B / 2  -- Distance between midpoint C to both A and B is 15 km
noncomputable def walking_speed : ℝ := 5  -- Walking speed is 5 km/h
noncomputable def biking_speed : ℝ := 20  -- Biking speed is 20 km/h

theorem optimal_bicycle_point : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ (30 - x + 4 * x = 60 - 3 * x) → x = 5 :=
by sorry

end optimal_bicycle_point_l230_230093


namespace cup_of_coffee_price_l230_230756

def price_cheesecake : ℝ := 10
def price_set : ℝ := 12
def discount : ℝ := 0.75

theorem cup_of_coffee_price (C : ℝ) (h : price_set = discount * (C + price_cheesecake)) : C = 6 :=
by
  sorry

end cup_of_coffee_price_l230_230756


namespace kan_krao_park_walkways_l230_230207

-- Definitions for the given conditions
structure Park (α : Type*) := 
  (entrances : Finset α)
  (walkways : α → α → Prop)
  (brick_paved : α → α → Prop)
  (asphalt_paved : α → α → Prop)
  (no_three_intersections : ∀ (x y z w : α), x ≠ y → y ≠ z → z ≠ w → w ≠ x → (walkways x y ∧ walkways z w) → ¬ (walkways x z ∧ walkways y w))

-- Conditions based on the given problem
variables {α : Type*} [Finite α] [DecidableRel (@walkways α)]
variable (p : Park α)
variables [Fintype α]

-- Translate conditions to definitions
def has_lotuses (p : α → α → Prop) (q : α → α → Prop) (x y : α) : Prop := p x y ∧ p x y
def has_waterlilies (p : α → α → Prop) (q : α → α → Prop) (x y : α) : Prop := (p x y ∧ q x y) ∨ (q x y ∧ p x y)
def is_lit (p : α → α → Prop) (q : α → α → Prop) : Prop := ∃ (x y : α), x ≠ y ∧ (has_lotuses p q x y ∧ has_lotuses p q x y ∧ ∃ sz, sz ≥ 45)

-- Mathematically equivalent proof problem
theorem kan_krao_park_walkways (p : Park α) :
  (∃ walkways_same_material : α → α → Prop, ∃ (lit_walkways : Finset (α × α)), lit_walkways.card ≥ 11) :=
sorry

end kan_krao_park_walkways_l230_230207


namespace complement_union_l230_230573

open Set

-- Define U to be the set of all real numbers
def U := @univ ℝ

-- Define the domain A for the function y = sqrt(x-2) + sqrt(x+1)
def A := {x : ℝ | x ≥ 2}

-- Define the domain B for the function y = sqrt(2x+4) / (x-3)
def B := {x : ℝ | x ≥ -2 ∧ x ≠ 3}

-- Theorem about the union of the complements
theorem complement_union : (U \ A ∪ U \ B) = {x : ℝ | x < 2 ∨ x = 3} := 
by
  sorry

end complement_union_l230_230573


namespace correct_mean_l230_230618

theorem correct_mean (mean n incorrect_value correct_value : ℝ) 
  (hmean : mean = 150) (hn : n = 20) (hincorrect : incorrect_value = 135) (hcorrect : correct_value = 160):
  (mean * n - incorrect_value + correct_value) / n = 151.25 :=
by
  sorry

end correct_mean_l230_230618


namespace students_left_early_l230_230246

theorem students_left_early :
  let initial_groups := 3
  let students_per_group := 8
  let students_remaining := 22
  let total_students := initial_groups * students_per_group
  total_students - students_remaining = 2 :=
by
  -- Define the initial conditions
  let initial_groups := 3
  let students_per_group := 8
  let students_remaining := 22
  let total_students := initial_groups * students_per_group
  -- Proof (to be completed)
  sorry

end students_left_early_l230_230246


namespace value_of_a_l230_230919

def f (a x : ℝ) : ℝ := a * x ^ 3 + 3 * x ^ 2 + 2

def f_prime (a x : ℝ) : ℝ := 3 * a * x ^ 2 + 6 * x

theorem value_of_a (a : ℝ) (h : f_prime a (-1) = 4) : a = 10 / 3 :=
by
  -- Proof goes here
  sorry

end value_of_a_l230_230919


namespace solution_to_system_l230_230799

theorem solution_to_system : ∃ x y : ℤ, (2 * x + 3 * y = -11 ∧ 6 * x - 5 * y = 9) ↔ (x = -1 ∧ y = -3) :=
by
  sorry

end solution_to_system_l230_230799


namespace range_of_a_l230_230164

theorem range_of_a {A B : Set ℝ} (hA : A = {x | x > 5}) (hB : B = {x | x > a}) 
  (h_sufficient_not_necessary : A ⊆ B ∧ ¬(B ⊆ A)) 
  : a < 5 :=
sorry

end range_of_a_l230_230164


namespace difference_of_fractions_l230_230697

theorem difference_of_fractions (x y : ℝ) (h1 : x = 497) (h2 : y = 325) :
  (2/5) * (3 * x + 7 * y) - (3/5) * (x * y) = -95408.6 := by
  rw [h1, h2]
  sorry

end difference_of_fractions_l230_230697


namespace pauls_weekly_spending_l230_230592

def mowing_lawns : ℕ := 3
def weed_eating : ℕ := 3
def total_weeks : ℕ := 2
def total_money : ℕ := mowing_lawns + weed_eating
def spending_per_week : ℕ := total_money / total_weeks

theorem pauls_weekly_spending : spending_per_week = 3 := by
  sorry

end pauls_weekly_spending_l230_230592


namespace cone_diameter_l230_230529

theorem cone_diameter (S : ℝ) (hS : S = 3 * Real.pi) (unfold_semicircle : ∃ (r l : ℝ), l = 2 * r ∧ S = π * r^2 + (1 / 2) * π * l^2) : 
∃ d : ℝ, d = Real.sqrt 6 := 
by
  sorry

end cone_diameter_l230_230529


namespace fraction_of_area_above_line_l230_230612

open Real

-- Define the points and the line between them
noncomputable def pointA : (ℝ × ℝ) := (2, 3)
noncomputable def pointB : (ℝ × ℝ) := (5, 1)

-- Define the vertices of the square
noncomputable def square_vertices : List (ℝ × ℝ) := [(2, 1), (5, 1), (5, 4), (2, 4)]

-- Define the equation of the line
noncomputable def line_eq (x : ℝ) : ℝ :=
  (-2/3) * x + 13/3

-- Define the vertical and horizontal boundaries
noncomputable def x_min : ℝ := 2
noncomputable def x_max : ℝ := 5
noncomputable def y_min : ℝ := 1
noncomputable def y_max : ℝ := 4

-- Calculate the area of the triangle formed below the line
noncomputable def triangle_area : ℝ := 0.5 * 2 * 3

-- Calculate the area of the square
noncomputable def square_area : ℝ := 3 * 3

-- The fraction of the area above the line
noncomputable def area_fraction_above : ℝ := (square_area - triangle_area) / square_area

-- Prove the fraction of the area of the square above the line is 2/3
theorem fraction_of_area_above_line : area_fraction_above = 2 / 3 :=
  sorry

end fraction_of_area_above_line_l230_230612


namespace pauline_total_spent_l230_230532

variable {items_total : ℝ} (discount_rate : ℝ) (discount_limit : ℝ) (sales_tax_rate : ℝ)

def total_spent (items_total discount_rate discount_limit sales_tax_rate : ℝ) : ℝ :=
  let discount_amount := discount_rate * discount_limit
  let discounted_total := discount_limit - discount_amount
  let non_discounted_total := items_total - discount_limit
  let subtotal := discounted_total + non_discounted_total
  let sales_tax := sales_tax_rate * subtotal
  subtotal + sales_tax

theorem pauline_total_spent :
  total_spent 250 0.15 100 0.08 = 253.80 :=
by
  sorry

end pauline_total_spent_l230_230532


namespace problem_part_one_problem_part_two_l230_230523

theorem problem_part_one : 23 - 17 - (-6) + (-16) = -4 :=
by
  sorry

theorem problem_part_two : 0 - 32 / ((-2)^3 - (-4)) = 8 :=
by
  sorry

end problem_part_one_problem_part_two_l230_230523


namespace fraction_meaningful_l230_230689

-- Define the condition for the fraction being meaningful
def denominator_not_zero (x : ℝ) : Prop := x + 1 ≠ 0

-- Define the statement to be proved
theorem fraction_meaningful (x : ℝ) : denominator_not_zero x ↔ x ≠ -1 :=
by
  sorry

end fraction_meaningful_l230_230689


namespace remainder_sum_of_squares_25_mod_6_l230_230860

def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

theorem remainder_sum_of_squares_25_mod_6 :
  (sum_of_squares 25) % 6 = 5 :=
by
  sorry

end remainder_sum_of_squares_25_mod_6_l230_230860


namespace total_people_in_house_l230_230160

-- Define the number of people in various locations based on the given conditions.
def charlie_and_susan := 2
def sarah_and_friends := 5
def people_in_bedroom := charlie_and_susan + sarah_and_friends
def people_in_living_room := 8

-- Prove the total number of people in the house is 14.
theorem total_people_in_house : people_in_bedroom + people_in_living_room = 14 := by
  -- Here we can use Lean's proof system, but we skip with 'sorry'
  sorry

end total_people_in_house_l230_230160


namespace race_distance_l230_230260

theorem race_distance
  (A B : Type)
  (D : ℕ) -- D is the total distance of the race
  (Va Vb : ℕ) -- A's speed and B's speed
  (H1 : D / 28 = Va) -- A's speed calculated from D and time
  (H2 : (D - 56) / 28 = Vb) -- B's speed calculated from distance and time
  (H3 : 56 / 7 = Vb) -- B's speed can also be calculated directly
  (H4 : Va = D / 28)
  (H5 : Vb = (D - 56) / 28) :
  D = 280 := sorry

end race_distance_l230_230260


namespace initial_percentage_of_milk_l230_230740

theorem initial_percentage_of_milk 
  (initial_solution_volume : ℝ)
  (extra_water_volume : ℝ)
  (desired_percentage : ℝ)
  (new_total_volume : ℝ)
  (initial_percentage : ℝ) :
  initial_solution_volume = 60 →
  extra_water_volume = 33.33333333333333 →
  desired_percentage = 54 →
  new_total_volume = initial_solution_volume + extra_water_volume →
  (initial_percentage / 100 * initial_solution_volume = desired_percentage / 100 * new_total_volume) →
  initial_percentage = 84 := 
by 
  intros initial_volume_eq extra_water_eq desired_perc_eq new_volume_eq equation
  -- proof steps here
  sorry

end initial_percentage_of_milk_l230_230740


namespace remainder_of_sum_mod_eight_l230_230741

theorem remainder_of_sum_mod_eight (m : ℤ) : 
  ((10 - 3 * m) + (5 * m + 6)) % 8 = (2 * m) % 8 :=
by
  sorry

end remainder_of_sum_mod_eight_l230_230741


namespace hua_luogeng_optimal_selection_l230_230959

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end hua_luogeng_optimal_selection_l230_230959


namespace find_x_l230_230278

theorem find_x (x y : ℚ) (h1 : 3 * x - 2 * y = 7) (h2 : x + 3 * y = 8) : x = 37 / 11 := by
  sorry

end find_x_l230_230278


namespace randy_biscuits_left_l230_230094

-- Define the function biscuits_left
def biscuits_left (initial: ℚ) (father_gift: ℚ) (mother_gift: ℚ) (brother_eat_percent: ℚ) : ℚ :=
  let total_before_eat := initial + father_gift + mother_gift
  let brother_ate := brother_eat_percent * total_before_eat
  total_before_eat - brother_ate

-- Given conditions
def initial_biscuits : ℚ := 32
def father_gift : ℚ := 2 / 3
def mother_gift : ℚ := 15
def brother_eat_percent : ℚ := 0.3

-- Correct answer as an approximation since we're dealing with real-world numbers
def approx (x y : ℚ) := abs (x - y) < 0.01

-- The proof problem statement in Lean 4
theorem randy_biscuits_left :
  approx (biscuits_left initial_biscuits father_gift mother_gift brother_eat_percent) 33.37 :=
by
  sorry

end randy_biscuits_left_l230_230094


namespace line_tangent_to_parabola_l230_230365

theorem line_tangent_to_parabola (k : ℝ) : 
  (∀ x y : ℝ, y^2 = 16 * x ∧ 4 * x + 3 * y + k = 0 → ∀ y, y^2 + 12 * y + 4 * k = 0 → (12)^2 - 4 * 1 * 4 * k = 0) → 
  k = 9 :=
by
  sorry

end line_tangent_to_parabola_l230_230365


namespace distance_travelled_l230_230818

variables (S D : ℝ)

-- conditions
def cond1 : Prop := D = S * 7
def cond2 : Prop := D = (S + 12) * 5

-- Define the main theorem
theorem distance_travelled (h1 : cond1 S D) (h2 : cond2 S D) : D = 210 :=
by {
  sorry
}

end distance_travelled_l230_230818


namespace choir_members_unique_l230_230219

theorem choir_members_unique (n : ℕ) :
  (n % 10 = 6) ∧ 
  (n % 11 = 6) ∧ 
  (150 ≤ n) ∧ 
  (n ≤ 300) → 
  n = 226 := 
by
  sorry

end choir_members_unique_l230_230219


namespace water_tower_excess_consumption_l230_230444

def water_tower_problem : Prop :=
  let initial_water := 2700
  let first_neighborhood := 300
  let second_neighborhood := 2 * first_neighborhood
  let third_neighborhood := second_neighborhood + 100
  let fourth_neighborhood := 3 * first_neighborhood
  let fifth_neighborhood := third_neighborhood / 2
  let leakage := 50
  let first_neighborhood_final := first_neighborhood + 0.10 * first_neighborhood
  let second_neighborhood_final := second_neighborhood - 0.05 * second_neighborhood
  let third_neighborhood_final := third_neighborhood + 0.10 * third_neighborhood
  let fifth_neighborhood_final := fifth_neighborhood - 0.05 * fifth_neighborhood
  let total_consumption := 
    first_neighborhood_final + second_neighborhood_final + third_neighborhood_final +
    fourth_neighborhood + fifth_neighborhood_final + leakage
  let excess_consumption := total_consumption - initial_water
  excess_consumption = 252.5

theorem water_tower_excess_consumption : water_tower_problem := by
  sorry

end water_tower_excess_consumption_l230_230444


namespace isosceles_triangle_angle_l230_230022

theorem isosceles_triangle_angle (x : ℕ) (h1 : 2 * x + x + x = 180) :
  x = 45 ∧ 2 * x = 90 :=
by
  have h2 : 4 * x = 180 := by linarith
  have h3 : x = 45 := by linarith
  have h4 : 2 * x = 90 := by linarith
  exact ⟨h3, h4⟩

end isosceles_triangle_angle_l230_230022


namespace problem_solution_l230_230965

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

theorem problem_solution : a^10 + b^10 = 123 := sorry

end problem_solution_l230_230965


namespace solution_set_of_inequality_l230_230726

theorem solution_set_of_inequality :
  ∀ x : ℝ, (x > -1 ∧ x < 1) ↔ (x^2 - 1 < 0) :=
by
  sorry

end solution_set_of_inequality_l230_230726


namespace quadratic_roots_l230_230215

theorem quadratic_roots (k : ℝ) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + k*x1 + (k - 1) = 0) ∧ (x2^2 + k*x2 + (k - 1) = 0) :=
by
  sorry

end quadratic_roots_l230_230215


namespace point_in_first_quadrant_l230_230151

theorem point_in_first_quadrant (x y : ℝ) (h₁ : x = 3) (h₂ : y = 2) (hx : x > 0) (hy : y > 0) :
  ∃ q : ℕ, q = 1 := 
by
  sorry

end point_in_first_quadrant_l230_230151


namespace part_a_part_b_l230_230254

-- Define the conditions for part (a)
def psychic_can_guess_at_least_19_cards : Prop :=
  ∃ (deck : Fin 36 → Fin 4) (psychic_guess : Fin 36 → Fin 4)
    (assistant_arrangement : Fin 36 → Bool),
    -- assistant and psychic agree on a method ensuring at least 19 correct guesses
    (∃ n : ℕ, n ≥ 19 ∧
      ∃ correct_guesses_set : Finset (Fin 36),
        correct_guesses_set.card = n ∧
        ∀ i ∈ correct_guesses_set, psychic_guess i = deck i)

-- Prove that the above condition is satisfied
theorem part_a : psychic_can_guess_at_least_19_cards :=
by
  sorry

-- Define the conditions for part (b)
def psychic_can_guess_at_least_23_cards : Prop :=
  ∃ (deck : Fin 36 → Fin 4) (psychic_guess : Fin 36 → Fin 4)
    (assistant_arrangement : Fin 36 → Bool),
    -- assistant and psychic agree on a method ensuring at least 23 correct guesses
    (∃ n : ℕ, n ≥ 23 ∧
      ∃ correct_guesses_set : Finset (Fin 36),
        correct_guesses_set.card = n ∧
        ∀ i ∈ correct_guesses_set, psychic_guess i = deck i)

-- Prove that the above condition is satisfied
theorem part_b : psychic_can_guess_at_least_23_cards :=
by
  sorry

end part_a_part_b_l230_230254


namespace innokentiy_games_l230_230310

def games_played_egor := 13
def games_played_nikita := 27
def games_played_innokentiy (N : ℕ) := N - games_played_egor

theorem innokentiy_games (N : ℕ) (h : N = games_played_nikita) : games_played_innokentiy N = 14 :=
by {
  sorry
}

end innokentiy_games_l230_230310


namespace smallest_n_divisible_by_one_billion_l230_230021

-- Define the sequence parameters and the common ratio
def first_term : ℚ := 5 / 8
def second_term : ℚ := 50
def common_ratio : ℚ := second_term / first_term -- this is 80

-- Define the n-th term of the geometric sequence
noncomputable def nth_term (n : ℕ) : ℚ :=
  first_term * (common_ratio ^ (n - 1))

-- Define the target divisor (one billion)
def target_divisor : ℤ := 10 ^ 9

-- Prove that the smallest n such that nth_term n is divisible by 10^9 is 9
theorem smallest_n_divisible_by_one_billion :
  ∃ n : ℕ, nth_term n = (first_term * (common_ratio ^ (n - 1))) ∧ 
           (target_divisor : ℚ) ∣ nth_term n ∧
           n = 9 :=
by sorry

end smallest_n_divisible_by_one_billion_l230_230021


namespace calculate_expression_value_l230_230809

theorem calculate_expression_value : 
  3 - ((-3 : ℚ) ^ (-3 : ℤ) * 2) = 83 / 27 := 
by
  sorry

end calculate_expression_value_l230_230809


namespace least_non_lucky_multiple_of_11_l230_230680

/--
A lucky integer is a positive integer which is divisible by the sum of its digits.
Example:
- 18 is a lucky integer because 1 + 8 = 9 and 18 is divisible by 9.
- 20 is not a lucky integer because 2 + 0 = 2 and 20 is not divisible by 2.
-/

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

theorem least_non_lucky_multiple_of_11 : ∃ n, n > 0 ∧ n % 11 = 0 ∧ ¬ is_lucky n ∧ ∀ m, m > 0 → m % 11 = 0 → ¬ is_lucky m → n ≤ m := 
by
  sorry

end least_non_lucky_multiple_of_11_l230_230680


namespace max_perimeter_isosceles_triangle_l230_230291

/-- Out of all triangles with the same base and the same angle at the vertex, 
    the triangle with the largest perimeter is isosceles -/
theorem max_perimeter_isosceles_triangle {α β γ : ℝ} (b : ℝ) (B : ℝ) (A C : ℝ) 
  (hB : 0 < B ∧ B < π) (hβ : α + C = B) (h1 : A = β) (h2 : γ = β) :
  α = γ := sorry

end max_perimeter_isosceles_triangle_l230_230291


namespace cylinder_is_defined_sphere_is_defined_hyperbolic_cylinder_is_defined_parabolic_cylinder_is_defined_l230_230810

-- 1) Cylinder
theorem cylinder_is_defined (R : ℝ) :
  ∀ (x y z : ℝ), x^2 + y^2 = R^2 → ∃ (r : ℝ), r = R ∧ x^2 + y^2 = r^2 :=
sorry

-- 2) Sphere
theorem sphere_is_defined (R : ℝ) :
  ∀ (x y z : ℝ), x^2 + y^2 + z^2 = R^2 → ∃ (r : ℝ), r = R ∧ x^2 + y^2 + z^2 = r^2 :=
sorry

-- 3) Hyperbolic Cylinder
theorem hyperbolic_cylinder_is_defined (m : ℝ) :
  ∀ (x y z : ℝ), xy = m → ∃ (k : ℝ), k = m ∧ xy = k :=
sorry

-- 4) Parabolic Cylinder
theorem parabolic_cylinder_is_defined :
  ∀ (x z : ℝ), z = x^2 → ∃ (k : ℝ), k = 1 ∧ z = k*x^2 :=
sorry

end cylinder_is_defined_sphere_is_defined_hyperbolic_cylinder_is_defined_parabolic_cylinder_is_defined_l230_230810


namespace find_ratio_l230_230933

noncomputable def p (x : ℝ) : ℝ := 3 * x * (x - 5)
noncomputable def q (x : ℝ) : ℝ := (x + 2) * (x - 5)

theorem find_ratio : (p 3) / (q 3) = 9 / 5 := by
  sorry

end find_ratio_l230_230933


namespace gcd_problem_l230_230248

theorem gcd_problem :
  ∃ n : ℕ, (80 ≤ n) ∧ (n ≤ 100) ∧ (n % 9 = 0) ∧ (Nat.gcd n 27 = 9) ∧ (n = 90) :=
by sorry

end gcd_problem_l230_230248


namespace age_of_B_l230_230599

theorem age_of_B (A B C : ℕ) (h1 : A + B + C = 90)
                  (h2 : (A - 10) = (B - 10) / 2)
                  (h3 : (B - 10) / 2 = (C - 10) / 3) : 
                  B = 30 :=
by sorry

end age_of_B_l230_230599


namespace sufficient_but_not_necessary_condition_l230_230147

def sufficient_condition (a : ℝ) : Prop := 
  (a > 1) → (1 / a < 1)

def necessary_condition (a : ℝ) : Prop := 
  (1 / a < 1) → (a > 1)

theorem sufficient_but_not_necessary_condition (a : ℝ) : sufficient_condition a ∧ ¬necessary_condition a := by
  sorry

end sufficient_but_not_necessary_condition_l230_230147


namespace platform_length_eq_train_length_l230_230447

noncomputable def length_of_train : ℝ := 900
noncomputable def speed_of_train_kmh : ℝ := 108
noncomputable def speed_of_train_mpm : ℝ := (speed_of_train_kmh * 1000) / 60
noncomputable def crossing_time_min : ℝ := 1
noncomputable def total_distance_covered : ℝ := speed_of_train_mpm * crossing_time_min

theorem platform_length_eq_train_length :
  total_distance_covered - length_of_train = length_of_train :=
by
  sorry

end platform_length_eq_train_length_l230_230447


namespace power_multiplication_same_base_l230_230110

theorem power_multiplication_same_base :
  (10 ^ 655 * 10 ^ 650 = 10 ^ 1305) :=
by {
  sorry
}

end power_multiplication_same_base_l230_230110


namespace arithmetic_sequence_a8_l230_230014

theorem arithmetic_sequence_a8 (a : ℕ → ℤ) (d : ℤ) :
  a 2 = 4 → a 4 = 2 → a 8 = -2 :=
by intros ha2 ha4
   sorry

end arithmetic_sequence_a8_l230_230014


namespace women_in_village_l230_230530

theorem women_in_village (W : ℕ) (men_present : ℕ := 150) (p : ℝ := 140.78099890167377) 
    (men_reduction_per_year: ℝ := 0.10) (year1_men : ℝ := men_present * (1 - men_reduction_per_year)) 
    (year2_men : ℝ := year1_men * (1 - men_reduction_per_year)) 
    (formula : ℝ := (year2_men^2 + W^2).sqrt) 
    (h : formula = p) : W = 71 := 
by
  sorry

end women_in_village_l230_230530


namespace steven_weight_l230_230060

theorem steven_weight (danny_weight : ℝ) (steven_more : ℝ) (steven_weight : ℝ) 
  (h₁ : danny_weight = 40) 
  (h₂ : steven_more = 0.2 * danny_weight) 
  (h₃ : steven_weight = danny_weight + steven_more) : 
  steven_weight = 48 := 
  by 
  sorry

end steven_weight_l230_230060


namespace james_weight_gain_l230_230112

def cheezits_calories (bags : ℕ) (oz_per_bag : ℕ) (cal_per_oz : ℕ) : ℕ :=
  bags * oz_per_bag * cal_per_oz

def chocolate_calories (bars : ℕ) (cal_per_bar : ℕ) : ℕ :=
  bars * cal_per_bar

def popcorn_calories (bags : ℕ) (cal_per_bag : ℕ) : ℕ :=
  bags * cal_per_bag

def run_calories (mins : ℕ) (cal_per_min : ℕ) : ℕ :=
  mins * cal_per_min

def swim_calories (mins : ℕ) (cal_per_min : ℕ) : ℕ :=
  mins * cal_per_min

def cycle_calories (mins : ℕ) (cal_per_min : ℕ) : ℕ :=
  mins * cal_per_min

def total_calories_consumed : ℕ :=
  cheezits_calories 3 2 150 + chocolate_calories 2 250 + popcorn_calories 1 500

def total_calories_burned : ℕ :=
  run_calories 40 12 + swim_calories 30 15 + cycle_calories 20 10

def excess_calories : ℕ :=
  total_calories_consumed - total_calories_burned

def weight_gain (excess_cal : ℕ) (cal_per_lb : ℕ) : ℚ :=
  excess_cal / cal_per_lb

theorem james_weight_gain :
  weight_gain excess_calories 3500 = 770 / 3500 :=
sorry

end james_weight_gain_l230_230112


namespace total_profit_is_18900_l230_230115

-- Defining the conditions
variable (x : ℕ)  -- A's initial investment
variable (A_share : ℕ := 6300)  -- A's share in rupees

-- Total profit calculation
def total_annual_gain : ℕ :=
  (x * 12) + (2 * x * 6) + (3 * x * 4)

-- The main statement
theorem total_profit_is_18900 (x : ℕ) (A_share : ℕ := 6300) :
  3 * A_share = total_annual_gain x :=
by sorry

end total_profit_is_18900_l230_230115


namespace neg_three_is_square_mod_p_l230_230970

theorem neg_three_is_square_mod_p (q : ℤ) (p : ℕ) (prime_p : Nat.Prime p) (condition : p = 3 * q + 1) :
  ∃ x : ℤ, (x^2 ≡ -3 [ZMOD p]) :=
sorry

end neg_three_is_square_mod_p_l230_230970


namespace fraction_product_equals_12_l230_230175

theorem fraction_product_equals_12 :
  (1 / 3) * (9 / 2) * (1 / 27) * (54 / 1) * (1 / 81) * (162 / 1) * (1 / 243) * (486 / 1) = 12 := 
by
  sorry

end fraction_product_equals_12_l230_230175


namespace beta_max_two_day_ratio_l230_230669

noncomputable def alpha_first_day_score : ℚ := 160 / 300
noncomputable def alpha_second_day_score : ℚ := 140 / 200
noncomputable def alpha_two_day_ratio : ℚ := 300 / 500

theorem beta_max_two_day_ratio :
  ∃ (p q r : ℕ), 
  p < 300 ∧
  q < (8 * p / 15) ∧
  r < ((3500 - 7 * p) / 10) ∧
  q + r = 299 ∧
  gcd 299 500 = 1 ∧
  (299 + 500) = 799 := 
sorry

end beta_max_two_day_ratio_l230_230669


namespace soda_original_price_l230_230080

theorem soda_original_price (P : ℝ) (h1 : 1.5 * P = 6) : P = 4 :=
by
  sorry

end soda_original_price_l230_230080


namespace perpendicular_planes_l230_230167

-- Definitions for lines and planes and their relationships
variable {a b : Line}
variable {α β : Plane}

-- Given conditions for the problem
axiom line_perpendicular (l1 l2 : Line) : Prop -- l1 ⊥ l2
axiom line_parallel (l1 l2 : Line) : Prop -- l1 ∥ l2
axiom line_plane_perpendicular (l : Line) (p : Plane) : Prop -- l ⊥ p
axiom line_plane_parallel (l : Line) (p : Plane) : Prop -- l ∥ p
axiom plane_perpendicular (p1 p2 : Plane) : Prop -- p1 ⊥ p2

-- Problem statement
theorem perpendicular_planes (h1 : line_perpendicular a b)
                            (h2 : line_plane_perpendicular a α)
                            (h3 : line_plane_perpendicular b β) :
                            plane_perpendicular α β :=
sorry

end perpendicular_planes_l230_230167


namespace cos_and_sin_double_angle_l230_230011

variables (θ : ℝ)

-- Conditions
def is_in_fourth_quadrant (θ : ℝ) : Prop :=
  θ > 3 * Real.pi / 2 ∧ θ < 2 * Real.pi

def sin_theta (θ : ℝ) : Prop :=
  Real.sin θ = -1 / 3

-- Problem statement
theorem cos_and_sin_double_angle (h1 : is_in_fourth_quadrant θ) (h2 : sin_theta θ) :
  Real.cos θ = 2 * Real.sqrt 2 / 3 ∧ Real.sin (2 * θ) = -(4 * Real.sqrt 2 / 9) :=
sorry

end cos_and_sin_double_angle_l230_230011


namespace initial_observations_l230_230484

theorem initial_observations (n : ℕ) (S : ℕ) 
  (h1 : S / n = 11)
  (h2 : ∃ (new_obs : ℕ), (S + new_obs) / (n + 1) = 10 ∧ new_obs = 4):
  n = 6 := 
sorry

end initial_observations_l230_230484


namespace smallest_divisor_sum_of_squares_of_1_to_7_l230_230962

def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

theorem smallest_divisor_sum_of_squares_of_1_to_7 (S : ℕ) (h : S = 1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2) :
  ∃ m, is_divisor m S ∧ (∀ d, is_divisor d S → 2 ≤ d) :=
sorry

end smallest_divisor_sum_of_squares_of_1_to_7_l230_230962


namespace area_of_inscribed_octagon_l230_230836

-- Define the given conditions and required proof
theorem area_of_inscribed_octagon (r : ℝ) (h : π * r^2 = 400 * π) :
  let A := r^2 * (1 + Real.sqrt 2)
  A = 20^2 * (1 + Real.sqrt 2) :=
by 
  sorry

end area_of_inscribed_octagon_l230_230836


namespace determine_c_l230_230555

theorem determine_c (a c : ℝ) (h : (2 * a - 1) / -3 < - (c + 1) / -4) : c ≠ -1 ∧ (c > 0 ∨ c < 0) :=
by sorry

end determine_c_l230_230555


namespace necklaces_caught_l230_230914

theorem necklaces_caught
  (LatchNecklaces RhondaNecklaces BoudreauxNecklaces: ℕ)
  (h1 : LatchNecklaces = 3 * RhondaNecklaces - 4)
  (h2 : RhondaNecklaces = BoudreauxNecklaces / 2)
  (h3 : BoudreauxNecklaces = 12) :
  LatchNecklaces = 14 := by
  sorry

end necklaces_caught_l230_230914


namespace joseph_drives_more_l230_230465

-- Definitions for the problem
def v_j : ℝ := 50 -- Joseph's speed in mph
def t_j : ℝ := 2.5 -- Joseph's time in hours
def v_k : ℝ := 62 -- Kyle's speed in mph
def t_k : ℝ := 2 -- Kyle's time in hours

-- Prove that Joseph drives 1 more mile than Kyle
theorem joseph_drives_more : (v_j * t_j) - (v_k * t_k) = 1 := 
by 
  sorry

end joseph_drives_more_l230_230465


namespace find_N_l230_230042

theorem find_N : (2 + 3 + 4) / 3 = (1990 + 1991 + 1992) / (N : ℚ) → N = 1991 := by
sorry

end find_N_l230_230042


namespace rectangle_perimeter_of_right_triangle_l230_230218

noncomputable def right_triangle_area (a b: ℕ) : ℝ := (1/2 : ℝ) * a * b

noncomputable def rectangle_length (area width: ℝ) : ℝ := area / width

noncomputable def rectangle_perimeter (length width: ℝ) : ℝ := 2 * (length + width)

theorem rectangle_perimeter_of_right_triangle :
  rectangle_perimeter (rectangle_length (right_triangle_area 7 24) 5) 5 = 43.6 :=
by
  sorry

end rectangle_perimeter_of_right_triangle_l230_230218


namespace terminating_fraction_count_l230_230626

theorem terminating_fraction_count :
  (∃ n_values : Finset ℕ, (∀ n ∈ n_values, 1 ≤ n ∧ n ≤ 500 ∧ (∃ k : ℕ, n = k * 49)) ∧ n_values.card = 10) :=
by
  -- Placeholder for the proof, does not contribute to the conditions-direct definitions.
  sorry

end terminating_fraction_count_l230_230626


namespace find_k_l230_230952

noncomputable section

variables {a b k : ℝ}

theorem find_k 
  (h1 : 4^a = k) 
  (h2 : 9^b = k)
  (h3 : 1 / a + 1 / b = 2) : 
  k = 6 :=
sorry

end find_k_l230_230952


namespace remainder_of_3_pow_19_mod_10_l230_230833

-- Definition of the problem and conditions
def q := 3^19

-- Statement to prove
theorem remainder_of_3_pow_19_mod_10 : q % 10 = 7 :=
by
  sorry

end remainder_of_3_pow_19_mod_10_l230_230833


namespace min_value_of_expression_l230_230269

theorem min_value_of_expression (a b c : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (habc : a + b + c = 1) (expected_value : 3 * a + 2 * b = 2) :
  ∃ a b, (a + b + (1 - a - b) = 1) ∧ (3 * a + 2 * b = 2) ∧ (∀ a b, ∃ m, m = (2/a + 1/(3*b)) ∧ m = 16/3) :=
sorry

end min_value_of_expression_l230_230269


namespace wall_height_correct_l230_230163

-- Define the dimensions of the brick in meters
def brick_length : ℝ := 0.2
def brick_width  : ℝ := 0.1
def brick_height : ℝ := 0.08

-- Define the volume of one brick
def volume_brick : ℝ := brick_length * brick_width * brick_height

-- Total number of bricks used
def number_of_bricks : ℕ := 12250

-- Define the wall dimensions except height
def wall_length : ℝ := 10
def wall_width  : ℝ := 24.5

-- Total volume of all bricks
def volume_total_bricks : ℝ := number_of_bricks * volume_brick

-- Volume of the wall
def volume_wall (h : ℝ) : ℝ := wall_length * h * wall_width

-- The height of the wall
def wall_height : ℝ := 0.08

-- The theorem to prove
theorem wall_height_correct : volume_total_bricks = volume_wall wall_height :=
by
  sorry

end wall_height_correct_l230_230163


namespace probability_of_snow_at_least_once_first_week_l230_230257

theorem probability_of_snow_at_least_once_first_week :
  let p_first4 := 1 / 4
  let p_next3 := 1 / 3
  let p_no_snow_first4 := (1 - p_first4) ^ 4
  let p_no_snow_next3 := (1 - p_next3) ^ 3
  let p_no_snow_week := p_no_snow_first4 * p_no_snow_next3
  1 - p_no_snow_week = 29 / 32 :=
by
  sorry

end probability_of_snow_at_least_once_first_week_l230_230257


namespace deepak_current_age_l230_230430

theorem deepak_current_age (x : ℕ) (rahul_age deepak_age : ℕ) :
  (rahul_age = 4 * x) →
  (deepak_age = 3 * x) →
  (rahul_age + 10 = 26) →
  deepak_age = 12 :=
by
  intros h1 h2 h3
  -- You would write the proof here
  sorry

end deepak_current_age_l230_230430


namespace equal_probability_of_selection_l230_230468

-- Define a structure representing the scenario of the problem.
structure SamplingProblem :=
  (total_students : ℕ)
  (eliminated_students : ℕ)
  (remaining_students : ℕ)
  (selection_size : ℕ)
  (systematic_step : ℕ)

-- Instantiate the specific problem.
def problem_instance : SamplingProblem :=
  { total_students := 3001
  , eliminated_students := 1
  , remaining_students := 3000
  , selection_size := 50
  , systematic_step := 60 }

-- Define the main theorem to be proven.
theorem equal_probability_of_selection (prob : SamplingProblem) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ prob.remaining_students → 
  (prob.remaining_students - prob.systematic_step * ((i - 1) / prob.systematic_step) = i) :=
sorry

end equal_probability_of_selection_l230_230468


namespace daffodil_stamps_count_l230_230106

theorem daffodil_stamps_count (r d : ℕ) (h1 : r = 2) (h2 : r = d) : d = 2 := by
  sorry

end daffodil_stamps_count_l230_230106


namespace find_g_26_l230_230055

variable {g : ℕ → ℕ}

theorem find_g_26 (hg : ∀ x, g (x + g x) = 5 * g x) (h1 : g 1 = 5) : g 26 = 120 :=
  sorry

end find_g_26_l230_230055


namespace probability_red_or_white_l230_230792

-- Definitions based on the conditions
def total_marbles := 20
def blue_marbles := 5
def red_marbles := 9
def white_marbles := total_marbles - (blue_marbles + red_marbles)

-- Prove that the probability of selecting a red or white marble is 3/4
theorem probability_red_or_white : (red_marbles + white_marbles : ℚ) / total_marbles = 3 / 4 :=
by sorry

end probability_red_or_white_l230_230792


namespace division_of_fractions_l230_230825

theorem division_of_fractions :
  (10 / 21) / (4 / 9) = 15 / 14 :=
by
  -- Proof will be provided here 
  sorry

end division_of_fractions_l230_230825


namespace gcd_20244_46656_l230_230019

theorem gcd_20244_46656 : Nat.gcd 20244 46656 = 54 := by
  sorry

end gcd_20244_46656_l230_230019


namespace hyperbola_inequality_l230_230178

-- Define point P on the hyperbola in terms of a and b
theorem hyperbola_inequality (a b : ℝ) (h : (3*a + 3*b)^2 / 9 - (a - b)^2 = 1) : |a + b| ≥ 1 :=
sorry

end hyperbola_inequality_l230_230178


namespace time_difference_l230_230140

-- Definitions of speeds and distance
def distance : Nat := 12
def alice_speed : Nat := 7
def bob_speed : Nat := 9

-- Calculations of total times based on speeds and distance
def alice_time : Nat := alice_speed * distance
def bob_time : Nat := bob_speed * distance

-- Statement of the problem
theorem time_difference : bob_time - alice_time = 24 := by
  sorry

end time_difference_l230_230140


namespace iced_tea_cost_is_correct_l230_230681

noncomputable def iced_tea_cost (cost_cappuccino cost_latte cost_espresso : ℝ) (num_cappuccino num_iced_tea num_latte num_espresso : ℕ) (bill_amount change_amount : ℝ) : ℝ :=
  let total_cappuccino_cost := cost_cappuccino * num_cappuccino
  let total_latte_cost := cost_latte * num_latte
  let total_espresso_cost := cost_espresso * num_espresso
  let total_spent := bill_amount - change_amount
  let total_other_cost := total_cappuccino_cost + total_latte_cost + total_espresso_cost
  let total_iced_tea_cost := total_spent - total_other_cost
  total_iced_tea_cost / num_iced_tea

theorem iced_tea_cost_is_correct:
  iced_tea_cost 2 1.5 1 3 2 2 2 20 3 = 3 :=
by
  sorry

end iced_tea_cost_is_correct_l230_230681


namespace valid_numbers_count_l230_230743

def count_valid_numbers : ℕ :=
  sorry

theorem valid_numbers_count :
  count_valid_numbers = 7 :=
sorry

end valid_numbers_count_l230_230743


namespace gcd_5280_12155_l230_230005

theorem gcd_5280_12155 : Nat.gcd 5280 12155 = 55 := by
  sorry

end gcd_5280_12155_l230_230005


namespace only_integer_solution_l230_230543

theorem only_integer_solution (n : ℕ) (h1 : n > 1) (h2 : (2 * n + 1) % n ^ 2 = 0) : n = 3 := 
sorry

end only_integer_solution_l230_230543


namespace parallel_lines_l230_230838

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + a + 3 = 0) ∧ (∀ x y : ℝ, x + (a + 1) * y + 4 = 0) 
  → a = -2 :=
sorry

end parallel_lines_l230_230838


namespace prove_k_in_terms_of_x_l230_230098

variables {A B k x : ℝ}

-- given conditions
def positive_numbers (A B : ℝ) := A > 0 ∧ B > 0
def ratio_condition (A B k : ℝ) := A = k * B
def percentage_condition (A B x : ℝ) := A = B + (x / 100) * B

-- proof statement
theorem prove_k_in_terms_of_x (A B k x : ℝ) (h1 : positive_numbers A B) (h2 : ratio_condition A B k) (h3 : percentage_condition A B x) (h4 : k > 1) :
  k = 1 + x / 100 :=
sorry

end prove_k_in_terms_of_x_l230_230098


namespace determine_f_101_l230_230869

theorem determine_f_101 (f : ℕ → ℕ) (h : ∀ m n : ℕ, m * n + 1 ∣ f m * f n + 1) : 
  ∃ k : ℕ, k % 2 = 1 ∧ f 101 = 101 ^ k :=
sorry

end determine_f_101_l230_230869


namespace tile_arrangement_probability_l230_230611

theorem tile_arrangement_probability :
  let X := 5
  let O := 4
  let total_tiles := 9
  (1 : ℚ) / (Nat.choose total_tiles X) = 1 / 126 :=
by
  sorry

end tile_arrangement_probability_l230_230611


namespace darnell_phone_minutes_l230_230846

theorem darnell_phone_minutes
  (unlimited_cost : ℕ)
  (text_cost : ℕ)
  (call_cost : ℕ)
  (texts_per_dollar : ℕ)
  (minutes_per_dollar : ℕ)
  (total_texts : ℕ)
  (cost_difference : ℕ)
  (alternative_total_cost : ℕ)
  (M : ℕ)
  (text_cost_condition : unlimited_cost - cost_difference = alternative_total_cost)
  (text_formula : M / minutes_per_dollar * call_cost + total_texts / texts_per_dollar * text_cost = alternative_total_cost)
  : M = 60 :=
sorry

end darnell_phone_minutes_l230_230846


namespace find_a10_l230_230390

def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

theorem find_a10 
  (a1 d : ℤ)
  (h_condition : a1 + (a1 + 18 * d) = -18) :
  arithmetic_sequence a1 d 10 = -9 := 
by
  sorry

end find_a10_l230_230390


namespace area_of_triangle_with_rational_vertices_on_unit_circle_is_rational_l230_230317

def rational_coords_on_unit_circle (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ) : Prop :=
  x₁^2 + y₁^2 = 1 ∧ x₂^2 + y₂^2 = 1 ∧ x₃^2 + y₃^2 = 1

theorem area_of_triangle_with_rational_vertices_on_unit_circle_is_rational
  (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ)
  (h : rational_coords_on_unit_circle x₁ y₁ x₂ y₂ x₃ y₃) :
  ∃ (A : ℚ), A = 1 / 2 * abs (x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)) :=
sorry

end area_of_triangle_with_rational_vertices_on_unit_circle_is_rational_l230_230317


namespace g_triple_application_l230_230554

def g (x : ℤ) : ℤ := 7 * x - 3

theorem g_triple_application : g (g (g 3)) = 858 := by
  sorry

end g_triple_application_l230_230554


namespace arithmetic_sequence_sum_l230_230966

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (h1 : a 1 + a 2 = -1)
  (h2 : a 3 = 4)
  (h3 : ∀ n, a (n + 1) - a n = a 2 - a 1) :
  a 4 + a 5 = 17 :=
  sorry

end arithmetic_sequence_sum_l230_230966


namespace percentage_selected_in_state_A_l230_230662

-- Definitions
def num_candidates : ℕ := 8000
def percentage_selected_state_B : ℕ := 7
def extra_selected_candidates : ℕ := 80

-- Question
theorem percentage_selected_in_state_A :
  ∃ (P : ℕ), ((P / 100) * 8000 + 80 = 560) ∧ (P = 6) := sorry

end percentage_selected_in_state_A_l230_230662


namespace find_side_length_l230_230341

noncomputable def side_length_of_equilateral_triangle (t : ℝ) (Q : ℝ × ℝ) : Prop :=
  let D := (0, 0)
  let E := (t, 0)
  let F := (t/2, t * (Real.sqrt 3) / 2)
  let DQ := Real.sqrt ((Q.1 - D.1) ^ 2 + (Q.2 - D.2) ^ 2)
  let EQ := Real.sqrt ((Q.1 - E.1) ^ 2 + (Q.2 - E.2) ^ 2)
  let FQ := Real.sqrt ((Q.1 - F.1) ^ 2 + (Q.2 - F.2) ^ 2)
  DQ = 2 ∧ EQ = 2 * Real.sqrt 2 ∧ FQ = 3

theorem find_side_length :
  ∃ t Q, side_length_of_equilateral_triangle t Q → t = 2 * Real.sqrt 5 :=
sorry

end find_side_length_l230_230341


namespace number_of_chickens_l230_230102

variables (C G Ch : ℕ)

theorem number_of_chickens (h1 : C = 9) (h2 : G = 4 * C) (h3 : G = 2 * Ch) : Ch = 18 :=
by
  sorry

end number_of_chickens_l230_230102


namespace sum_coords_A_eq_neg9_l230_230268

variable (A B C : ℝ × ℝ)
variable (h1 : (C.1 - A.1)^2 + (C.2 - A.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 / 3)
variable (h2 : (C.1 - B.1)^2 + (C.2 - B.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 / 3)
variable (hB : B = (2, 5))
variable (hC : C = (4, 11))

theorem sum_coords_A_eq_neg9 
  (A B C : ℝ × ℝ)
  (h1 : (C.1 - A.1)^2 + (C.2 - A.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 / 3)
  (h2 : (C.1 - B.1)^2 + (C.2 - B.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 / 3)
  (hB : B = (2, 5))
  (hC : C = (4, 11)) : 
  A.1 + A.2 = -9 :=
  sorry

end sum_coords_A_eq_neg9_l230_230268


namespace multiplier_of_first_integer_l230_230059

theorem multiplier_of_first_integer :
  ∃ m x : ℤ, x + 4 = 15 ∧ x * m = 3 + 2 * 15 ∧ m = 3 := by
  sorry

end multiplier_of_first_integer_l230_230059


namespace score_order_l230_230842

variables (L N O P : ℕ)

def conditions : Prop := 
  O = L ∧ 
  N < max O P ∧ 
  P > L

theorem score_order (h : conditions L N O P) : N < O ∧ O < P :=
by
  sorry

end score_order_l230_230842


namespace find_original_price_l230_230018

-- Given conditions:
-- 1. 10% cashback
-- 2. $25 mail-in rebate
-- 3. Final cost is $110

def original_price (P : ℝ) (cashback : ℝ) (rebate : ℝ) (final_cost : ℝ) :=
  final_cost = P - (cashback * P + rebate)

theorem find_original_price :
  ∀ (P : ℝ), original_price P 0.10 25 110 → P = 150 :=
by
  sorry

end find_original_price_l230_230018


namespace gilda_stickers_left_l230_230664

variable (S : ℝ) (hS : S > 0)

def remaining_after_olga : ℝ := 0.70 * S
def remaining_after_sam : ℝ := 0.80 * remaining_after_olga S
def remaining_after_max : ℝ := 0.70 * remaining_after_sam S
def remaining_after_charity : ℝ := 0.90 * remaining_after_max S

theorem gilda_stickers_left :
  remaining_after_charity S / S * 100 = 35.28 := by
  sorry

end gilda_stickers_left_l230_230664


namespace find_13th_result_l230_230273

theorem find_13th_result 
  (average_25 : ℕ) (average_12_first : ℕ) (average_12_last : ℕ) 
  (total_25 : average_25 * 25 = 600) 
  (total_12_first : average_12_first * 12 = 168) 
  (total_12_last : average_12_last * 12 = 204) 
: average_25 - average_12_first - average_12_last = 228 :=
by
  sorry

end find_13th_result_l230_230273


namespace trigonometric_eq_solution_count_l230_230084

theorem trigonometric_eq_solution_count :
  ∃ B : Finset ℤ, B.card = 250 ∧ ∀ x ∈ B, 2000 ≤ x ∧ x ≤ 3000 ∧ 
  2 * Real.sqrt 2 * Real.sin (Real.pi * x / 4)^3 = Real.sin (Real.pi / 4 * (1 + x)) :=
sorry

end trigonometric_eq_solution_count_l230_230084


namespace weigh_80_grams_is_false_l230_230024

def XiaoGang_weight_grams : Nat := 80000  -- 80 kilograms in grams
def weight_claim : Nat := 80  -- 80 grams claim

theorem weigh_80_grams_is_false : weight_claim ≠ XiaoGang_weight_grams :=
by
  sorry

end weigh_80_grams_is_false_l230_230024


namespace parallelogram_diagonal_square_l230_230665

theorem parallelogram_diagonal_square (A B C D P Q R S : Type)
    (area_ABCD : ℝ) (proj_A_P_BD proj_C_Q_BD proj_B_R_AC proj_D_S_AC : Prop)
    (PQ RS : ℝ) (d_squared : ℝ) 
    (h_area : area_ABCD = 24)
    (h_proj_A_P : proj_A_P_BD) (h_proj_C_Q : proj_C_Q_BD)
    (h_proj_B_R : proj_B_R_AC) (h_proj_D_S : proj_D_S_AC)
    (h_PQ_length : PQ = 8) (h_RS_length : RS = 10)
    : d_squared = 62 + 20*Real.sqrt 61 := sorry

end parallelogram_diagonal_square_l230_230665


namespace exists_five_distinct_nat_numbers_l230_230656

theorem exists_five_distinct_nat_numbers 
  (a b c d e : ℕ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h_no_div_3 : ¬(3 ∣ a) ∧ ¬(3 ∣ b) ∧ ¬(3 ∣ c) ∧ ¬(3 ∣ d) ∧ ¬(3 ∣ e))
  (h_no_div_4 : ¬(4 ∣ a) ∧ ¬(4 ∣ b) ∧ ¬(4 ∣ c) ∧ ¬(4 ∣ d) ∧ ¬(4 ∣ e))
  (h_no_div_5 : ¬(5 ∣ a) ∧ ¬(5 ∣ b) ∧ ¬(5 ∣ c) ∧ ¬(5 ∣ d) ∧ ¬(5 ∣ e)) :
  (∃ (a b c d e : ℕ),
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
    (¬(3 ∣ a) ∧ ¬(3 ∣ b) ∧ ¬(3 ∣ c) ∧ ¬(3 ∣ d) ∧ ¬(3 ∣ e)) ∧
    (¬(4 ∣ a) ∧ ¬(4 ∣ b) ∧ ¬(4 ∣ c) ∧ ¬(4 ∣ d) ∧ ¬(4 ∣ e)) ∧
    (¬(5 ∣ a) ∧ ¬(5 ∣ b) ∧ ¬(5 ∣ c) ∧ ¬(5 ∣ d) ∧ ¬(5 ∣ e)) ∧
    (∀ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z → x + y + z = a + b + c + d + e → (x + y + z) % 3 = 0) ∧
    (∀ w x y z : ℕ, w ≠ x ∧ x ≠ y ∧ y ≠ z ∧ w ≠ y ∧ w ≠ z ∧ x ≠ z → w + x + y + z = a + b + c + d + e → (w + x + y + z) % 4 = 0) ∧
    (a + b + c + d + e) % 5 = 0) :=
  sorry

end exists_five_distinct_nat_numbers_l230_230656


namespace binary1011_eq_11_l230_230096

-- Define a function to convert a binary number represented as a list of bits to a decimal number.
def binaryToDecimal (bits : List (Fin 2)) : Nat :=
  bits.foldr (λ (bit : Fin 2) (acc : Nat) => acc * 2 + bit.val) 0

-- The binary number 1011 represented as a list of bits.
def binary1011 : List (Fin 2) := [1, 0, 1, 1]

-- The theorem stating that the decimal equivalent of binary 1011 is 11.
theorem binary1011_eq_11 : binaryToDecimal binary1011 = 11 :=
by
  sorry

end binary1011_eq_11_l230_230096


namespace tom_read_books_l230_230653

theorem tom_read_books :
  let books_may := 2
  let books_june := 6
  let books_july := 10
  books_may + books_june + books_july = 18 := by
  sorry

end tom_read_books_l230_230653


namespace pair_divisibility_l230_230012

theorem pair_divisibility (m n : ℕ) : 
  (m * n ∣ m ^ 2019 + n) ↔ ((m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 2 ^ 2019)) := sorry

end pair_divisibility_l230_230012


namespace inequality_solution_l230_230500

noncomputable def f (x : ℝ) : ℝ := 
  Real.log (Real.sqrt (x^2 + 1) + x) - (2 / (Real.exp x + 1))

theorem inequality_solution :
  { x : ℝ | f x + f (2 * x - 1) > -2 } = { x : ℝ | x > 1 / 3 } :=
sorry

end inequality_solution_l230_230500


namespace find_moles_of_NaCl_l230_230880

-- Define the chemical reaction as an equation
def chemical_reaction (NaCl KNO3 NaNO3 KCl : ℕ) : Prop :=
  NaCl + KNO3 = NaNO3 + KCl

-- Define the problem conditions
def problem_conditions (naCl : ℕ) : Prop :=
  ∃ (kno3 naNo3 kcl : ℕ),
    kno3 = 3 ∧
    naNo3 = 3 ∧
    chemical_reaction naCl kno3 naNo3 kcl

-- Define the goal statement
theorem find_moles_of_NaCl (naCl : ℕ) : problem_conditions naCl → naCl = 3 :=
by
  sorry -- proof to be filled in later

end find_moles_of_NaCl_l230_230880


namespace distance_rowed_upstream_l230_230313

noncomputable def speed_of_boat_in_still_water := 18 -- from solution step; b = 18 km/h
def speed_of_stream := 3 -- given
def time := 4 -- given
def distance_downstream := 84 -- given

theorem distance_rowed_upstream 
  (b : ℕ) (s : ℕ) (t : ℕ) (d_down : ℕ) (d_up : ℕ)
  (h_stream : s = 3) 
  (h_time : t = 4)
  (h_distance_downstream : d_down = 84) 
  (h_speed_boat : b = 18) 
  (h_effective_downstream_speed : b + s = d_down / t) :
  d_up = 60 := by
  sorry

end distance_rowed_upstream_l230_230313


namespace smallest_x_l230_230990

theorem smallest_x (x : ℕ) : (x + 3457) % 15 = 1537 % 15 → x = 15 :=
by
  sorry

end smallest_x_l230_230990


namespace yogurt_combinations_l230_230902

-- Define the conditions from a)
def num_flavors : ℕ := 5
def num_toppings : ℕ := 8
def num_sizes : ℕ := 3

-- Define the problem in a theorem statement
theorem yogurt_combinations : num_flavors * ((num_toppings * (num_toppings - 1)) / 2) * num_sizes = 420 :=
by
  -- sorry is used here to skip the proof
  sorry

end yogurt_combinations_l230_230902


namespace interior_box_surface_area_l230_230091

-- Given conditions
def original_length : ℕ := 40
def original_width : ℕ := 60
def corner_side : ℕ := 8

-- Calculate the initial area
def area_original : ℕ := original_length * original_width

-- Calculate the area of one corner
def area_corner : ℕ := corner_side * corner_side

-- Calculate the total area removed by four corners
def total_area_removed : ℕ := 4 * area_corner

-- Theorem to state the final area remaining
theorem interior_box_surface_area : 
  area_original - total_area_removed = 2144 :=
by
  -- Place the proof here
  sorry

end interior_box_surface_area_l230_230091


namespace tan_a6_of_arithmetic_sequence_l230_230896

noncomputable def arithmetic_sequence (a : ℕ → ℝ) := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := 
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem tan_a6_of_arithmetic_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (H1 : arithmetic_sequence a)
  (H2 : sum_of_first_n_terms a S)
  (H3 : S 11 = 22 * Real.pi / 3) : 
  Real.tan (a 6) = -Real.sqrt 3 :=
sorry

end tan_a6_of_arithmetic_sequence_l230_230896


namespace average_height_corrected_l230_230761

theorem average_height_corrected (students : ℕ) (incorrect_avg_height : ℝ) (incorrect_height : ℝ) (actual_height : ℝ)
  (h1 : students = 20)
  (h2 : incorrect_avg_height = 175)
  (h3 : incorrect_height = 151)
  (h4 : actual_height = 111) :
  (incorrect_avg_height * students - incorrect_height + actual_height) / students = 173 :=
by
  sorry

end average_height_corrected_l230_230761


namespace product_mod_7_l230_230492

theorem product_mod_7 (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 5) : (a * b * c) % 7 = 2 := by
  sorry

end product_mod_7_l230_230492


namespace number_of_triangles_l230_230992

-- Definition of given conditions
def original_wire_length : ℝ := 84
def remaining_wire_length : ℝ := 12
def wire_per_triangle : ℝ := 3

-- The goal is to prove that the number of triangles that can be made is 24
theorem number_of_triangles : (original_wire_length - remaining_wire_length) / wire_per_triangle = 24 := by
  sorry

end number_of_triangles_l230_230992


namespace trig_expression_evaluation_l230_230709

-- Define the given conditions
axiom sin_390 : Real.sin (390 * Real.pi / 180) = 1 / 2
axiom tan_neg_45 : Real.tan (-45 * Real.pi / 180) = -1
axiom cos_360 : Real.cos (360 * Real.pi / 180) = 1

-- Formulate the theorem
theorem trig_expression_evaluation : 
  2 * Real.sin (390 * Real.pi / 180) - Real.tan (-45 * Real.pi / 180) + 5 * Real.cos (360 * Real.pi / 180) = 7 :=
by
  rw [sin_390, tan_neg_45, cos_360]
  sorry

end trig_expression_evaluation_l230_230709


namespace sebastian_age_correct_l230_230752

-- Define the ages involved
def sebastian_age_now := 40
def sister_age_now (S : ℕ) := S - 10
def father_age_now := 85

-- Define the conditions
def age_difference_condition (S : ℕ) := (sister_age_now S) = S - 10
def father_age_condition := father_age_now = 85
def past_age_sum_condition (S : ℕ) := (S - 5) + (sister_age_now S - 5) = 3 / 4 * (father_age_now - 5)

theorem sebastian_age_correct (S : ℕ) 
  (h1 : age_difference_condition S) 
  (h2 : father_age_condition) 
  (h3 : past_age_sum_condition S) : 
  S = sebastian_age_now := 
  by sorry

end sebastian_age_correct_l230_230752


namespace haley_fuel_consumption_ratio_l230_230244

theorem haley_fuel_consumption_ratio (gallons: ℕ) (miles: ℕ) (h_gallons: gallons = 44) (h_miles: miles = 77) :
  (gallons / Nat.gcd gallons miles) = 4 ∧ (miles / Nat.gcd gallons miles) = 7 :=
by
  sorry

end haley_fuel_consumption_ratio_l230_230244


namespace find_positive_integer_x_l230_230296

def positive_integer (x : ℕ) : Prop :=
  x > 0

def n (x : ℕ) : ℕ :=
  x^2 + 3 * x + 20

def d (x : ℕ) : ℕ :=
  3 * x + 4

def division_property (x : ℕ) : Prop :=
  ∃ q r : ℕ, q = x ∧ r = 8 ∧ n x = q * d x + r

theorem find_positive_integer_x :
  ∃ x : ℕ, positive_integer x ∧ n x = x * d x + 8 :=
sorry

end find_positive_integer_x_l230_230296


namespace jerry_age_l230_230945

theorem jerry_age (M J : ℕ) (h1 : M = 20) (h2 : M = 2 * J - 8) : J = 14 := 
by
  sorry

end jerry_age_l230_230945


namespace find_equation_of_line_l230_230706

theorem find_equation_of_line
  (midpoint : ℝ × ℝ)
  (ellipse : ℝ → ℝ → Prop)
  (l_eq : ℝ → ℝ → Prop)
  (H_mid : midpoint = (1, 2))
  (H_ellipse : ∀ (x y : ℝ), ellipse x y ↔ x^2 / 64 + y^2 / 16 = 1)
  (H_line : ∀ (x y : ℝ), l_eq x y ↔ y - 2 = - (1/8) * (x - 1))
  : ∃ (a b c : ℝ), (a, b, c) = (1, 8, -17) ∧ (∀ (x y : ℝ), l_eq x y ↔ a * x + b * y + c = 0) :=
by 
  sorry

end find_equation_of_line_l230_230706


namespace inequalities_region_quadrants_l230_230690

theorem inequalities_region_quadrants:
  (∀ x y : ℝ, y > -2 * x + 3 → y > x / 2 + 1 → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0)) :=
sorry

end inequalities_region_quadrants_l230_230690


namespace tangency_point_exists_l230_230408

theorem tangency_point_exists :
  ∃ (x y : ℝ), y = x^2 + 18 * x + 47 ∧ x = y^2 + 36 * y + 323 ∧ x = -17 / 2 ∧ y = -35 / 2 :=
by
  sorry

end tangency_point_exists_l230_230408


namespace choir_members_count_l230_230872

theorem choir_members_count (n : ℕ) (h1 : n % 10 = 4) (h2 : n % 11 = 5) (h3 : 200 ≤ n) (h4 : n ≤ 300) : n = 234 := 
sorry

end choir_members_count_l230_230872


namespace possible_values_of_expression_l230_230624

theorem possible_values_of_expression (x : ℝ) (h : 3 ≤ x ∧ x ≤ 4) : 
  40 ≤ x^2 + 7 * x + 10 ∧ x^2 + 7 * x + 10 ≤ 54 := 
sorry

end possible_values_of_expression_l230_230624


namespace sphere_volume_in_cone_l230_230520

theorem sphere_volume_in_cone (d : ℝ) (r : ℝ) (π : ℝ) (V : ℝ) (h1 : d = 12) (h2 : r = d / 2) (h3 : V = (4 / 3) * π * r^3) :
  V = 288 * π :=
by 
  sorry

end sphere_volume_in_cone_l230_230520


namespace find_fourth_number_l230_230494

theorem find_fourth_number (x : ℝ) (h : (3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5) = 800.0000000000001) : x = 0.3 :=
by
  sorry

end find_fourth_number_l230_230494


namespace imaginary_part_of_complex_num_l230_230425

def imaginary_unit : ℂ := Complex.I

noncomputable def complex_num : ℂ := 10 * imaginary_unit / (1 - 2 * imaginary_unit)

theorem imaginary_part_of_complex_num : complex_num.im = 2 := by
  sorry

end imaginary_part_of_complex_num_l230_230425


namespace train_speed_l230_230199

noncomputable def speed_of_each_train (v : ℕ) : ℕ := 27

theorem train_speed
  (length_of_each_train : ℕ)
  (crossing_time : ℕ)
  (crossing_condition : 2 * (length_of_each_train * crossing_time) / (2 * crossing_time) = 15 / 2)
  (conversion_factor : ∀ n, 1 = 3.6 * n → ℕ) :
  speed_of_each_train 27 = 27 :=
by
  exact rfl

end train_speed_l230_230199


namespace optimal_years_minimize_cost_l230_230395

noncomputable def initial_cost : ℝ := 150000
noncomputable def annual_expenses (n : ℕ) : ℝ := 15000 * n
noncomputable def maintenance_cost (n : ℕ) : ℝ := (n * (3000 + 3000 * n)) / 2
noncomputable def total_cost (n : ℕ) : ℝ := initial_cost + annual_expenses n + maintenance_cost n
noncomputable def average_annual_cost (n : ℕ) : ℝ := total_cost n / n

theorem optimal_years_minimize_cost : ∀ n : ℕ, n = 10 ↔ average_annual_cost 10 ≤ average_annual_cost n :=
by sorry

end optimal_years_minimize_cost_l230_230395


namespace candy_partition_l230_230187

theorem candy_partition :
  let candies := 10
  let boxes := 3
  ∃ ways : ℕ, ways = Nat.choose (candies + boxes - 1) (boxes - 1) ∧ ways = 66 :=
by
  let candies := 10
  let boxes := 3
  let ways := Nat.choose (candies + boxes - 1) (boxes - 1)
  have h : ways = 66 := sorry
  exact ⟨ways, ⟨rfl, h⟩⟩

end candy_partition_l230_230187


namespace find_a_odd_function_l230_230004

noncomputable def f (a x : ℝ) := Real.log (Real.sqrt (x^2 + 1) - a * x)

theorem find_a_odd_function :
  ∀ a : ℝ, (∀ x : ℝ, f a (-x) + f a x = 0) ↔ (a = 1 ∨ a = -1) := by
  sorry

end find_a_odd_function_l230_230004


namespace max_value_a_l230_230034

theorem max_value_a (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a = 3 :=
sorry

end max_value_a_l230_230034


namespace find_floors_l230_230767

theorem find_floors
  (a b : ℕ)
  (alexie_bathrooms_per_floor : ℕ := 3)
  (alexie_bedrooms_per_floor : ℕ := 2)
  (baptiste_bathrooms_per_floor : ℕ := 4)
  (baptiste_bedrooms_per_floor : ℕ := 3)
  (total_bathrooms : ℕ := 25)
  (total_bedrooms : ℕ := 18)
  (h1 : alexie_bathrooms_per_floor * a + baptiste_bathrooms_per_floor * b = total_bathrooms)
  (h2 : alexie_bedrooms_per_floor * a + baptiste_bedrooms_per_floor * b = total_bedrooms) :
  a = 3 ∧ b = 4 :=
by
  sorry

end find_floors_l230_230767


namespace DVDs_per_season_l230_230397

theorem DVDs_per_season (total_DVDs : ℕ) (seasons : ℕ) (h1 : total_DVDs = 40) (h2 : seasons = 5) : total_DVDs / seasons = 8 :=
by
  sorry

end DVDs_per_season_l230_230397


namespace algebraic_expression_value_l230_230384

theorem algebraic_expression_value (x y : ℝ) (h : x^2 - 4 * x - 1 = 0) : 
  (2 * x - 3) ^ 2 - (x + y) * (x - y) - y ^ 2 = 12 := 
by {
  sorry
}

end algebraic_expression_value_l230_230384


namespace bananas_to_oranges_l230_230133

theorem bananas_to_oranges :
  (3 / 4 : ℝ) * 16 = 12 →
  (2 / 3 : ℝ) * 9 = 6 :=
by
  intro h
  sorry

end bananas_to_oranges_l230_230133


namespace algebra_geometry_probabilities_l230_230373

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

end algebra_geometry_probabilities_l230_230373


namespace determine_CD_l230_230226

theorem determine_CD (AB : ℝ) (BD : ℝ) (BC : ℝ) (CD : ℝ) (Angle_ADB : ℝ)
  (sin_A : ℝ) (sin_C : ℝ)
  (h1 : AB = 30)
  (h2 : Angle_ADB = 90)
  (h3 : sin_A = 4/5)
  (h4 : sin_C = 1/5)
  (h5 : BD = sin_A * AB)
  (h6 : BC = BD / sin_C) :
  CD = 24 * Real.sqrt 23 := by
  sorry

end determine_CD_l230_230226


namespace min_y_value_l230_230931

theorem min_y_value (x : ℝ) : 
  (∀ y : ℝ, y = 4 * x^2 + 8 * x + 16 → y ≥ 12 ∧ (y = 12 ↔ x = -1)) :=
sorry

end min_y_value_l230_230931


namespace find_b_l230_230153

-- Definitions from conditions
def f (x : ℚ) := 3 * x - 2
def g (x : ℚ) := 7 - 2 * x

-- Problem statement
theorem find_b (b : ℚ) (h : g (f b) = 1) : b = 5 / 3 := sorry

end find_b_l230_230153


namespace solve_frac_eq_l230_230851

theorem solve_frac_eq (x : ℝ) (h : 3 - 5 / x + 2 / (x^2) = 0) : 
  ∃ y : ℝ, (y = 3 / x ∧ (y = 9 / 2 ∨ y = 3)) :=
sorry

end solve_frac_eq_l230_230851


namespace region_in_plane_l230_230372

def f (x : ℝ) : ℝ := x^2 - 6 * x + 5

theorem region_in_plane (x y : ℝ) :
  (f x + f y ≤ 0) ∧ (f x - f y ≥ 0) ↔
  ((x - 3)^2 + (y - 3)^2 ≤ 8) ∧ ((x ≥ y ∧ x + y ≥ 6) ∨ (x ≤ y ∧ x + y ≤ 6)) :=
by
  sorry

end region_in_plane_l230_230372


namespace problem_inverse_range_m_l230_230192

theorem problem_inverse_range_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 2 / x + 1 / y = 1) : 
  (2 * x + y > m^2 + 8 * m) ↔ (m > -9 ∧ m < 1) := 
by
  sorry

end problem_inverse_range_m_l230_230192


namespace repeating_decimal_subtraction_l230_230978

noncomputable def x := (0.246 : Real)
noncomputable def y := (0.135 : Real)
noncomputable def z := (0.579 : Real)

theorem repeating_decimal_subtraction :
  x - y - z = (-156 : ℚ) / 333 :=
by
  sorry

end repeating_decimal_subtraction_l230_230978


namespace range_of_m_l230_230476

theorem range_of_m (m : ℝ) : (¬ ∃ x : ℝ, 4 ^ x + 2 ^ (x + 1) + m = 0) → m ≥ 0 := 
by
  sorry

end range_of_m_l230_230476


namespace find_f_zero_l230_230385

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_zero (h : ∀ x : ℝ, x ≠ 0 → f (2 * x - 1) = (1 - x^2) / x^2) : f 0 = 3 :=
sorry

end find_f_zero_l230_230385


namespace glasses_needed_l230_230105

theorem glasses_needed (total_juice : ℕ) (juice_per_glass : ℕ) : Prop :=
  total_juice = 153 ∧ juice_per_glass = 30 → (total_juice + juice_per_glass - 1) / juice_per_glass = 6

-- This will state our theorem but we include sorry to omit the proof.

end glasses_needed_l230_230105


namespace cars_difference_proof_l230_230834

theorem cars_difference_proof (U M : ℕ) :
  let initial_cars := 150
  let total_cars := 196
  let cars_from_uncle := U
  let cars_from_grandpa := 2 * U
  let cars_from_dad := 10
  let cars_from_auntie := U + 1
  let cars_from_mum := M
  let total_given_cars := cars_from_dad + cars_from_auntie + cars_from_uncle + cars_from_grandpa + cars_from_mum
  initial_cars + total_given_cars = total_cars ->
  (cars_from_mum - cars_from_dad = 5) := 
by
  sorry

end cars_difference_proof_l230_230834


namespace burger_cost_l230_230776

theorem burger_cost
  (B P : ℝ)
  (h₁ : P = 2 * B)
  (h₂ : P + 3 * B = 45) :
  B = 9 := by
  sorry

end burger_cost_l230_230776


namespace mural_lunch_break_duration_l230_230009

variable (a t L : ℝ)

theorem mural_lunch_break_duration
  (h1 : (8 - L) * (a + t) = 0.6)
  (h2 : (6.5 - L) * t = 0.3)
  (h3 : (11 - L) * a = 0.1) :
  L = 40 :=
by
  sorry

end mural_lunch_break_duration_l230_230009


namespace tan_sum_inequality_l230_230863

noncomputable def pi : ℝ := Real.pi

theorem tan_sum_inequality (x α : ℝ) (hx1 : 0 ≤ x) (hx2 : x ≤ pi / 2) (hα1 : pi / 6 < α) (hα2 : α < pi / 3) :
  Real.tan (pi * (Real.sin x) / (4 * Real.sin α)) + Real.tan (pi * (Real.cos x) / (4 * Real.cos α)) > 1 :=
by
  sorry

end tan_sum_inequality_l230_230863


namespace factor_expression_l230_230654

theorem factor_expression (x y : ℝ) :
  75 * x^10 * y^3 - 150 * x^20 * y^6 = 75 * x^10 * y^3 * (1 - 2 * x^10 * y^3) :=
by
  sorry

end factor_expression_l230_230654


namespace expression_of_24ab_in_P_and_Q_l230_230901

theorem expression_of_24ab_in_P_and_Q (a b : ℕ) (P Q : ℝ)
  (hP : P = 2^a) (hQ : Q = 5^b) : 24^(a*b) = P^(3*b) * 3^(a*b) := 
  by
  sorry

end expression_of_24ab_in_P_and_Q_l230_230901


namespace sum_of_variables_l230_230844

theorem sum_of_variables (x y z : ℝ) (h₁ : x + y = 1) (h₂ : y + z = 1) (h₃ : z + x = 1) : x + y + z = 3 / 2 := 
sorry

end sum_of_variables_l230_230844


namespace hyperbola_s_eq_l230_230963

theorem hyperbola_s_eq (s : ℝ) 
  (hyp1 : ∃ b > 0, ∀ x y : ℝ, (x, y) = (5, -3) → x^2 / 9 - y^2 / b^2 = 1) 
  (hyp2 : ∃ b > 0, ∀ x y : ℝ, (x, y) = (3, 0) → x^2 / 9 - y^2 / b^2 = 1) 
  (hyp3 : ∃ b > 0, ∀ x y : ℝ, (x, y) = (s, -1) → x^2 / 9 - y^2 / b^2 = 1) :
  s^2 = 873 / 81 :=
sorry

end hyperbola_s_eq_l230_230963


namespace number_of_smoothies_l230_230512

-- Definitions of the given conditions
def burger_cost : ℕ := 5
def sandwich_cost : ℕ := 4
def smoothie_cost : ℕ := 4
def total_cost : ℕ := 17

-- Statement of the proof problem
theorem number_of_smoothies (S : ℕ) : burger_cost + sandwich_cost + S * smoothie_cost = total_cost → S = 2 :=
by
  intro h
  sorry

end number_of_smoothies_l230_230512


namespace Yoque_monthly_payment_l230_230675

theorem Yoque_monthly_payment :
  ∃ m : ℝ, m = 15 ∧ ∀ a t : ℝ, a = 150 ∧ t = 11 ∧ (a + 0.10 * a) / t = m :=
by
  sorry

end Yoque_monthly_payment_l230_230675


namespace oblique_projection_intuitive_diagrams_correct_l230_230884

-- Definitions based on conditions
structure ObliqueProjection :=
  (lines_parallel_x_axis_same_length : Prop)
  (lines_parallel_y_axis_halved_length : Prop)
  (perpendicular_relationship_becomes_45_angle : Prop)

-- Definitions based on statements
def intuitive_triangle_projection (P : ObliqueProjection) : Prop :=
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

def intuitive_parallelogram_projection (P : ObliqueProjection) : Prop := 
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

def intuitive_square_projection (P : ObliqueProjection) : Prop := 
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

def intuitive_rhombus_projection (P : ObliqueProjection) : Prop := 
  P.lines_parallel_x_axis_same_length ∧ 
  P.lines_parallel_y_axis_halved_length ∧ 
  P.perpendicular_relationship_becomes_45_angle

-- Theorem stating which intuitive diagrams are correctly represented under the oblique projection method.
theorem oblique_projection_intuitive_diagrams_correct : 
  ∀ (P : ObliqueProjection), 
    intuitive_triangle_projection P ∧ 
    intuitive_parallelogram_projection P ∧
    ¬intuitive_square_projection P ∧
    ¬intuitive_rhombus_projection P :=
by 
  sorry

end oblique_projection_intuitive_diagrams_correct_l230_230884


namespace no_integer_solutions_system_l230_230401

theorem no_integer_solutions_system :
  ¬∃ (x y z : ℤ), x^6 + x^3 + x^3 * y + y = 147^157 ∧ x^3 + x^3 * y + y^2 + y + z^9 = 157^147 := 
sorry

end no_integer_solutions_system_l230_230401


namespace hyperbola_m_range_l230_230369

-- Define the equation of the hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / (m + 2)) - (y^2 / (m - 1)) = 1

-- State the equivalent range problem
theorem hyperbola_m_range (m : ℝ) :
  is_hyperbola m ↔ (m < -2 ∨ m > 1) :=
by
  sorry

end hyperbola_m_range_l230_230369


namespace certain_amount_eq_3_l230_230387

theorem certain_amount_eq_3 (x A : ℕ) (hA : A = 5) (h : A + (11 + x) = 19) : x = 3 :=
by
  sorry

end certain_amount_eq_3_l230_230387


namespace simplify_expression_l230_230859

theorem simplify_expression :
  (3 / 4 : ℚ) * 60 - (8 / 5 : ℚ) * 60 + x = 12 → x = 63 :=
by
  intro h
  sorry

end simplify_expression_l230_230859


namespace liquid_mixture_ratio_l230_230125

theorem liquid_mixture_ratio (m1 m2 m3 : ℝ) (ρ1 ρ2 ρ3 : ℝ) (k : ℝ)
  (hρ1 : ρ1 = 6 * k) (hρ2 : ρ2 = 3 * k) (hρ3 : ρ3 = 2 * k)
  (h_condition : m1 ≥ 3.5 * m2)
  (h_arith_mean : (m1 + m2 + m3) / (m1 / ρ1 + m2 / ρ2 + m3 / ρ3) = (ρ1 + ρ2 + ρ3) / 3) :
    ∃ x y : ℝ, x ≤ 2/7 ∧ (4 * x + 15 * y = 7) := sorry

end liquid_mixture_ratio_l230_230125


namespace part_I_part_II_l230_230658

variable (f : ℝ → ℝ)

-- Condition 1: f is an even function
axiom even_function : ∀ x : ℝ, f (-x) = f x

-- Condition 2: f is symmetric about x = 1
axiom symmetric_about_1 : ∀ x : ℝ, f x = f (2 - x)

-- Condition 3: f(x₁ + x₂) = f(x₁) * f(x₂) for x₁, x₂ ∈ [0, 1/2]
axiom multiplicative_on_interval : ∀ x₁ x₂ : ℝ, (0 ≤ x₁ ∧ x₁ ≤ 1/2) ∧ (0 ≤ x₂ ∧ x₂ ≤ 1/2) → f (x₁ + x₂) = f x₁ * f x₂

-- Given f(1) = 2
axiom f_one : f 1 = 2

-- Part I: Prove f(1/2) = √2 and f(1/4) = 2^(1/4).
theorem part_I : f (1 / 2) = Real.sqrt 2 ∧ f (1 / 4) = Real.sqrt (Real.sqrt 2) := by
  sorry

-- Part II: Prove that f(x) is a periodic function with period 2.
theorem part_II : ∀ x : ℝ, f x = f (x + 2) := by
  sorry

end part_I_part_II_l230_230658


namespace find_parenthesis_value_l230_230432

theorem find_parenthesis_value (x : ℝ) (h : x * (-2/3) = 2) : x = -3 :=
by
  sorry

end find_parenthesis_value_l230_230432


namespace contractor_absent_days_l230_230531

-- Definition of conditions
def total_days : ℕ := 30
def payment_per_work_day : ℝ := 25
def fine_per_absent_day : ℝ := 7.5
def total_payment : ℝ := 490

-- The proof statement
theorem contractor_absent_days : ∃ y : ℕ, (∃ x : ℕ, x + y = total_days ∧ payment_per_work_day * (x : ℝ) - fine_per_absent_day * (y : ℝ) = total_payment) ∧ y = 8 := 
by 
  sorry

end contractor_absent_days_l230_230531


namespace tan_alpha_fraction_l230_230890

theorem tan_alpha_fraction (α : ℝ) (hα1 : 0 < α ∧ α < (Real.pi / 2)) (hα2 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) :
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_fraction_l230_230890


namespace sum_of_roots_l230_230960

theorem sum_of_roots {x1 x2 x3 k m : ℝ} (h1 : x1 ≠ x2) (h2 : x2 ≠ x3) (h3 : x1 ≠ x3)
  (h4 : 2 * x1^3 - k * x1 = m) (h5 : 2 * x2^3 - k * x2 = m) (h6 : 2 * x3^3 - k * x3 = m) :
  x1 + x2 + x3 = 0 :=
sorry

end sum_of_roots_l230_230960


namespace sum_of_integers_product_neg17_l230_230821

theorem sum_of_integers_product_neg17 (a b c : ℤ) (h : a * b * c = -17) : a + b + c = -15 ∨ a + b + c = 17 :=
sorry

end sum_of_integers_product_neg17_l230_230821


namespace find_x_value_l230_230913

theorem find_x_value (x : ℝ) (h₀ : 0 < x ∧ x < 180) (h₁ : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : x = 130 := 
by
  sorry

end find_x_value_l230_230913


namespace highest_place_joker_can_achieve_is_6_l230_230993

-- Define the total number of teams
def total_teams : ℕ := 16

-- Define conditions for points in football
def points_win : ℕ := 3
def points_draw : ℕ := 1
def points_loss : ℕ := 0

-- Condition definitions for Joker's performance in the tournament
def won_against_strong_teams (j k : ℕ) : Prop := j < k
def lost_against_weak_teams (j k : ℕ) : Prop := j > k

-- Define the performance of all teams
def teams (t : ℕ) := {n // n < total_teams}

-- Function to calculate Joker's points based on position k
def joker_points (k : ℕ) : ℕ := (total_teams - k) * points_win

theorem highest_place_joker_can_achieve_is_6 : ∃ k, k = 6 ∧ 
  (∀ j, 
    (j < k → won_against_strong_teams j k) ∧ 
    (j > k → lost_against_weak_teams j k) ∧
    (∃! p, p = joker_points k)) :=
by
  sorry

end highest_place_joker_can_achieve_is_6_l230_230993


namespace monotonic_interval_range_l230_230847

noncomputable def f (a x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

theorem monotonic_interval_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < 2 → 1 < x₂ → x₂ < 2 → x₁ < x₂ → f a x₁ ≤ f a x₂ ∨ f a x₁ ≥ f a x₂) ↔
  (a ∈ Set.Iic (-1) ∪ Set.Ici 0) :=
sorry

end monotonic_interval_range_l230_230847


namespace smallest_value_between_0_and_1_l230_230518

theorem smallest_value_between_0_and_1 (y : ℝ) (h : 0 < y ∧ y < 1) :
  y^3 < y ∧ y^3 < 3 * y ∧ y^3 < y^(1/3 : ℝ) ∧ y^3 < 1 ∧ y^3 < 1 / y :=
by
  sorry

end smallest_value_between_0_and_1_l230_230518


namespace robert_total_balls_l230_230683

-- Define the conditions
def robert_initial_balls : ℕ := 25
def tim_balls : ℕ := 40

-- Mathematically equivalent proof problem
theorem robert_total_balls : 
  robert_initial_balls + (tim_balls / 2) = 45 := by
  sorry

end robert_total_balls_l230_230683


namespace freeRangingChickens_l230_230725

-- Define the number of chickens in the coop
def chickensInCoop : Nat := 14

-- Define the number of chickens in the run
def chickensInRun : Nat := 2 * chickensInCoop

-- Define the number of chickens free ranging
def chickensFreeRanging : Nat := 2 * chickensInRun - 4

-- State the theorem
theorem freeRangingChickens : chickensFreeRanging = 52 := by
  -- We cannot provide the proof, so we use sorry
  sorry

end freeRangingChickens_l230_230725


namespace probability_of_negative_cosine_value_l230_230302

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ a1 d, ∀ n, a n = a1 + (n - 1) * d

def sum_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

theorem probability_of_negative_cosine_value (a : ℕ → ℝ) (S : ℕ → ℝ) 
(h_arith_seq : arithmetic_sequence a)
(h_sum_seq : sum_arithmetic_sequence a S)
(h_S4 : S 4 = Real.pi)
(h_a4_eq_2a2 : a 4 = 2 * a 2) :
∃ p : ℝ, p = 7 / 15 ∧
  ∀ n, 1 ≤ n ∧ n ≤ 30 → 
  ((Real.cos (a n) < 0) → p = 7 / 15) :=
by sorry

end probability_of_negative_cosine_value_l230_230302


namespace correct_equations_l230_230722

variable (x y : ℝ)

theorem correct_equations :
  (18 * x = y + 3) ∧ (17 * x = y - 4) ↔ (18 * x = y + 3) ∧ (17 * x = y - 4) :=
by
  sorry

end correct_equations_l230_230722


namespace shaded_region_area_eq_108_l230_230289

/-- There are two concentric circles, where the outer circle has twice the radius of the inner circle,
and the total boundary length of the shaded region is 36π. Prove that the area of the shaded region
is nπ, where n = 108. -/
theorem shaded_region_area_eq_108 (r : ℝ) (h_outer : ∀ (c₁ c₂ : ℝ), c₁ = 2 * c₂) 
  (h_boundary : 2 * Real.pi * r + 2 * Real.pi * (2 * r) = 36 * Real.pi) : 
  ∃ (n : ℕ), n = 108 ∧ (Real.pi * (2 * r)^2 - Real.pi * r^2) = n * Real.pi := 
sorry

end shaded_region_area_eq_108_l230_230289


namespace total_shapes_proof_l230_230488

def stars := 50
def stripes := 13

def circles : ℕ := (stars / 2) - 3
def squares : ℕ := (2 * stripes) + 6
def triangles : ℕ := (stars - stripes) * 2
def diamonds : ℕ := (stars + stripes) / 4

def total_shapes : ℕ := circles + squares + triangles + diamonds

theorem total_shapes_proof : total_shapes = 143 := by
  sorry

end total_shapes_proof_l230_230488


namespace simplify_expression_l230_230601

variable (a b c d x : ℝ)
variable (hab : a ≠ b)
variable (hac : a ≠ c)
variable (had : a ≠ d)
variable (hbc : b ≠ c)
variable (hbd : b ≠ d)
variable (hcd : c ≠ d)

theorem simplify_expression :
  ( ( (x + a)^4 / ((a - b)*(a - c)*(a - d)) )
  + ( (x + b)^4 / ((b - a)*(b - c)*(b - d)) )
  + ( (x + c)^4 / ((c - a)*(c - b)*(c - d)) )
  + ( (x + d)^4 / ((d - a)*(d - b)*(d - c)) ) = a + b + c + d + 4*x ) :=
  sorry

end simplify_expression_l230_230601


namespace pepperoni_slices_left_l230_230866

theorem pepperoni_slices_left :
  ∀ (total_friends : ℕ) (total_slices : ℕ) (cheese_left : ℕ),
    (total_friends = 4) →
    (total_slices = 16) →
    (cheese_left = 7) →
    (∃ p_slices_left : ℕ, p_slices_left = 4) :=
by
  intros total_friends total_slices cheese_left h_friends h_slices h_cheese
  sorry

end pepperoni_slices_left_l230_230866


namespace track_and_field_unit_incorrect_l230_230744

theorem track_and_field_unit_incorrect :
  ∀ (L : ℝ), L = 200 → "mm" ≠ "m" → false :=
by
  intros L hL hUnit
  sorry

end track_and_field_unit_incorrect_l230_230744


namespace dinesh_loop_l230_230027

noncomputable def number_of_pentagons (n : ℕ) : ℕ :=
  if (20 * n) % 11 = 0 then 10 else 0

theorem dinesh_loop (n : ℕ) : number_of_pentagons n = 10 :=
by sorry

end dinesh_loop_l230_230027


namespace Oliver_has_9_dollars_left_l230_230939

def initial_amount := 9
def saved := 5
def earned := 6
def spent_frisbee := 4
def spent_puzzle := 3
def spent_stickers := 2
def spent_movie_ticket := 7
def spent_snack := 3
def gift := 8

def final_amount (initial_amount : ℕ) (saved : ℕ) (earned : ℕ) (spent_frisbee : ℕ)
                 (spent_puzzle : ℕ) (spent_stickers : ℕ) (spent_movie_ticket : ℕ)
                 (spent_snack : ℕ) (gift : ℕ) : ℕ :=
  initial_amount + saved + earned - spent_frisbee - spent_puzzle - spent_stickers - 
  spent_movie_ticket - spent_snack + gift

theorem Oliver_has_9_dollars_left :
  final_amount initial_amount saved earned spent_frisbee 
               spent_puzzle spent_stickers spent_movie_ticket 
               spent_snack gift = 9 :=
  by
  sorry

end Oliver_has_9_dollars_left_l230_230939


namespace prob_sunny_l230_230225

variables (A B C : Prop) 
variables (P : Prop → ℝ)

-- Conditions
axiom prob_A : P A = 0.45
axiom prob_B : P B = 0.2
axiom mutually_exclusive : P A + P B + P C = 1

-- Proof problem
theorem prob_sunny : P C = 0.35 :=
by sorry

end prob_sunny_l230_230225


namespace min_employees_needed_l230_230330

theorem min_employees_needed (forest_jobs : ℕ) (marine_jobs : ℕ) (both_jobs : ℕ)
    (h1 : forest_jobs = 95) (h2 : marine_jobs = 80) (h3 : both_jobs = 35) :
    (forest_jobs - both_jobs) + (marine_jobs - both_jobs) + both_jobs = 140 :=
by
  sorry

end min_employees_needed_l230_230330


namespace relative_complement_correct_l230_230549

noncomputable def M : Set ℤ := {x : ℤ | -1 ≤ x ∧ x ≤ 3}
def N : Set ℤ := {1, 2}
def complement_M_N : Set ℤ := {x ∈ M | x ∉ N}

theorem relative_complement_correct : complement_M_N = {-1, 0, 3} := 
by
  sorry

end relative_complement_correct_l230_230549


namespace students_per_van_l230_230886

def number_of_boys : ℕ := 60
def number_of_girls : ℕ := 80
def number_of_vans : ℕ := 5

theorem students_per_van : (number_of_boys + number_of_girls) / number_of_vans = 28 := by
  sorry

end students_per_van_l230_230886


namespace determine_k_and_a_n_and_T_n_l230_230049

noncomputable def S_n (n : ℕ) (k : ℝ) : ℝ := -0.5 * n^2 + k * n

/-- Given the sequence S_n with sum of the first n terms S_n := -1/2 n^2 + k*n,
where k is a positive natural number. The maximum value of S_n is 8. -/
theorem determine_k_and_a_n_and_T_n (k : ℝ) (h : k = 4) :
  (∀ n : ℕ, S_n n k ≤ 8) ∧ 
  (∀ n : ℕ, ∃ a : ℝ, a = 9/2 - n) ∧
  (∀ n : ℕ, ∃ T : ℝ, T = 4 - (n + 2)/2^(n-1)) :=
by
  sorry

end determine_k_and_a_n_and_T_n_l230_230049


namespace mass_percentage_oxygen_NaBrO3_l230_230977

-- Definitions
def molar_mass_Na : ℝ := 22.99
def molar_mass_Br : ℝ := 79.90
def molar_mass_O : ℝ := 16.00

def molar_mass_NaBrO3 : ℝ := molar_mass_Na + molar_mass_Br + 3 * molar_mass_O

-- Theorem: proof that the mass percentage of oxygen in NaBrO3 is 31.81%
theorem mass_percentage_oxygen_NaBrO3 :
  ((3 * molar_mass_O) / molar_mass_NaBrO3) * 100 = 31.81 := by
  sorry

end mass_percentage_oxygen_NaBrO3_l230_230977


namespace value_of_xyz_l230_230786

variable (x y z : ℝ)

theorem value_of_xyz (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
                     (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 22) 
                     : x * y * z = 14 / 3 := 
sorry

end value_of_xyz_l230_230786


namespace tommy_writing_time_l230_230394

def numUniqueLettersTommy : Nat := 5
def numRearrangementsPerMinute : Nat := 20
def totalRearrangements : Nat := numUniqueLettersTommy.factorial
def minutesToComplete : Nat := totalRearrangements / numRearrangementsPerMinute
def hoursToComplete : Rat := minutesToComplete / 60

theorem tommy_writing_time :
  hoursToComplete = 0.1 := by
  sorry

end tommy_writing_time_l230_230394


namespace sin_x_lt_a_l230_230621

theorem sin_x_lt_a (a θ : ℝ) (h1 : -1 < a) (h2 : a < 0) (hθ : θ = Real.arcsin a) :
  {x : ℝ | ∃ n : ℤ, (2 * n - 1) * Real.pi - θ < x ∧ x < 2 * n * Real.pi + θ} = {x : ℝ | Real.sin x < a} :=
sorry

end sin_x_lt_a_l230_230621


namespace distinct_values_l230_230318

-- Define the expressions as terms in Lean
def expr1 : ℕ := 3 ^ (3 ^ 3)
def expr2 : ℕ := (3 ^ 3) ^ 3

-- State the theorem that these terms yield exactly two distinct values
theorem distinct_values : (expr1 ≠ expr2) ∧ ((expr1 = 3^27) ∨ (expr1 = 19683)) ∧ ((expr2 = 3^27) ∨ (expr2 = 19683)) := 
  sorry

end distinct_values_l230_230318


namespace sara_spent_on_bought_movie_l230_230509

-- Define the costs involved
def cost_ticket : ℝ := 10.62
def cost_rent : ℝ := 1.59
def total_spent : ℝ := 36.78

-- Define the quantity of tickets
def number_of_tickets : ℝ := 2

-- Define the total cost on tickets
def cost_on_tickets : ℝ := cost_ticket * number_of_tickets

-- Define the total cost on tickets and rented movie
def cost_on_tickets_and_rent : ℝ := cost_on_tickets + cost_rent

-- Define the total amount spent on buying the movie
def cost_bought_movie : ℝ := total_spent - cost_on_tickets_and_rent

-- The statement we need to prove
theorem sara_spent_on_bought_movie : cost_bought_movie = 13.95 :=
by
  sorry

end sara_spent_on_bought_movie_l230_230509


namespace linear_function_quadrants_l230_230480

theorem linear_function_quadrants (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : ¬ ∃ x : ℝ, ∃ y : ℝ, x > 0 ∧ y < 0 ∧ y = b * x - a :=
sorry

end linear_function_quadrants_l230_230480


namespace bike_race_difference_l230_230560

-- Define the conditions
def carlos_miles : ℕ := 70
def dana_miles : ℕ := 50
def time_period : ℕ := 5

-- State the theorem to prove the difference in miles biked
theorem bike_race_difference :
  carlos_miles - dana_miles = 20 := 
sorry

end bike_race_difference_l230_230560


namespace product_of_terms_eq_72_l230_230198

theorem product_of_terms_eq_72
  (a b c : ℝ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 12) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) = 72 :=
by
  sorry

end product_of_terms_eq_72_l230_230198


namespace find_S7_l230_230580

variable {a : ℕ → ℚ} {S : ℕ → ℚ}

axiom a1_def : a 1 = 1 / 2
axiom a_next_def : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * S n + 1
axiom S_def : ∀ n : ℕ, S (n + 1) = S n + a (n + 1)

theorem find_S7 : S 7 = 1457 / 2 := by
  sorry

end find_S7_l230_230580


namespace min_policemen_needed_l230_230265

-- Definitions of the problem parameters
def city_layout (n m : ℕ) := n > 0 ∧ m > 0

-- Function to calculate the minimum number of policemen
def min_policemen (n m : ℕ) : ℕ := (m - 1) * (n - 1)

-- The theorem to prove
theorem min_policemen_needed (n m : ℕ) (h : city_layout n m) : min_policemen n m = (m - 1) * (n - 1) :=
by
  unfold city_layout at h
  unfold min_policemen
  sorry

end min_policemen_needed_l230_230265


namespace minimum_value_of_T_l230_230793

theorem minimum_value_of_T (a b c : ℝ) (h1 : ∀ x : ℝ, (1 / a) * x^2 + b * x + c ≥ 0) (h2 : a * b > 1) :
  ∃ T : ℝ, T = 4 ∧ T = (1 / (2 * (a * b - 1))) + (a * (b + 2 * c) / (a * b - 1)) :=
by
  sorry

end minimum_value_of_T_l230_230793


namespace cubic_polynomial_k_l230_230475

noncomputable def h (x : ℝ) : ℝ := x^3 - x - 2

theorem cubic_polynomial_k (k : ℝ → ℝ)
  (hk : ∃ (B : ℝ), ∀ (x : ℝ), k x = B * (x - (root1 ^ 2)) * (x - (root2 ^ 2)) * (x - (root3 ^ 2)))
  (hroots : h (root1) = 0 ∧ h (root2) = 0 ∧ h (root3) = 0)
  (h_values : k 0 = 2) :
  k (-8) = -20 :=
sorry

end cubic_polynomial_k_l230_230475


namespace first_number_eq_l230_230400

theorem first_number_eq (x y : ℝ) (h1 : x * 120 = 346) (h2 : y * 240 = 346) : x = 346 / 120 :=
by
  -- The final proof will be inserted here
  sorry

end first_number_eq_l230_230400


namespace determine_x_l230_230217

theorem determine_x (x : ℕ) (hx : 27^3 + 27^3 + 27^3 = 3^x) : x = 10 :=
sorry

end determine_x_l230_230217


namespace fractional_sum_identity_l230_230801

noncomputable def distinct_real_roots (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem fractional_sum_identity :
  ∀ (p q r A B C : ℝ),
  (x^3 - 22*x^2 + 80*x - 67 = (x - p) * (x - q) * (x - r)) →
  distinct_real_roots (λ x => x^3 - 22*x^2 + 80*x - 67) p q r →
  (∀ (s : ℝ), s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 22*s^2 + 80*s - 67) = A / (s - p) + B / (s - q) + C / (s - r)) →
  (1 / (A) + 1 / (B) + 1 / (C) = 244) :=
by 
  intros p q r A B C h_poly h_distinct h_fractional
  sorry

end fractional_sum_identity_l230_230801


namespace value_of_a_plus_b_l230_230074

theorem value_of_a_plus_b 
  (a b : ℝ) 
  (f g : ℝ → ℝ)
  (h₁ : ∀ x, f x = a * x + b)
  (h₂ : ∀ x, g x = 3 * x - 6)
  (h₃ : ∀ x, g (f x) = 4 * x + 5) : 
  a + b = 5 :=
sorry

end value_of_a_plus_b_l230_230074


namespace earliest_time_100_degrees_l230_230374

def temperature (t : ℝ) : ℝ := -t^2 + 15 * t + 40

theorem earliest_time_100_degrees :
  ∃ t : ℝ, temperature t = 100 ∧ (∀ t' : ℝ, temperature t' = 100 → t' ≥ t) :=
by
  sorry

end earliest_time_100_degrees_l230_230374


namespace handshake_problem_l230_230068

theorem handshake_problem :
  ∃ (a b : ℕ), a + b = 20 ∧ (a * (a - 1)) / 2 + (b * (b - 1)) / 2 = 106 ∧ a * b = 84 :=
by
  sorry

end handshake_problem_l230_230068


namespace distance_home_gym_l230_230551

theorem distance_home_gym 
  (v_WangLei v_ElderSister : ℕ)  -- speeds in meters per minute
  (d_meeting : ℕ)                -- distance in meters from the gym to the meeting point
  (t_gym : ℕ)                    -- time in minutes for the older sister to the gym
  (speed_diff : v_ElderSister = v_WangLei + 20)  -- speed difference
  (t_gym_reached : d_meeting / 2 = (25 * (v_WangLei + 20)) - d_meeting): 
  v_WangLei * t_gym = 1500 :=
by
  sorry

end distance_home_gym_l230_230551


namespace reading_enhusiasts_not_related_to_gender_l230_230282

noncomputable def contingency_table (boys_scores : List Nat) (girls_scores : List Nat) :
  (Nat × Nat × Nat × Nat × Nat × Nat) × (Nat × Nat × Nat × Nat × Nat × Nat) :=
  let boys_range := (2, 3, 5, 15, 18, 12)
  let girls_range := (0, 5, 10, 10, 7, 13)
  ((2, 3, 5, 15, 18, 12), (0, 5, 10, 10, 7, 13))

theorem reading_enhusiasts_not_related_to_gender (boys_scores : List Nat) (girls_scores : List Nat) :
  let table := contingency_table boys_scores girls_scores
  let (boys_range, girls_range) := table
  let a := 45 -- Boys who are reading enthusiasts
  let b := 10 -- Boys who are non-reading enthusiasts
  let c := 30 -- Girls who are reading enthusiasts
  let d := 15 -- Girls who are non-reading enthusiasts
  let n := a + b + c + d
  let k_squared := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))
  k_squared < 3.841 := 
sorry

end reading_enhusiasts_not_related_to_gender_l230_230282


namespace unanswered_questions_l230_230490

variables (c w u : ℕ)

theorem unanswered_questions :
  (c + w + u = 50) ∧
  (6 * c + u = 120) ∧
  (3 * c - 2 * w = 45) →
  u = 37 :=
by {
  sorry
}

end unanswered_questions_l230_230490


namespace problem_inequality_l230_230419

variable (x y z : ℝ)

theorem problem_inequality (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) :
  2 * (x^3 + y^3 + z^3) ≥ x^2 * y + x^2 * z + y^2 * z + y^2 * x + z^2 * x + z^2 * y := by
  sorry

end problem_inequality_l230_230419


namespace number_of_tshirts_sold_l230_230942

theorem number_of_tshirts_sold 
    (original_price discounted_price revenue : ℕ)
    (discount : ℕ) 
    (no_of_tshirts: ℕ)
    (h1 : original_price = 51)
    (h2 : discount = 8)
    (h3 : discounted_price = original_price - discount)
    (h4 : revenue = 5590)
    (h5 : revenue = no_of_tshirts * discounted_price) : 
    no_of_tshirts = 130 :=
by
  sorry

end number_of_tshirts_sold_l230_230942


namespace five_digit_integers_count_l230_230472
open BigOperators

noncomputable def permutations_with_repetition (n : ℕ) (reps : List ℕ) : ℕ :=
  n.factorial / ((reps.map (λ x => x.factorial)).prod)

theorem five_digit_integers_count :
  permutations_with_repetition 5 [2, 2] = 30 :=
by
  sorry

end five_digit_integers_count_l230_230472


namespace a_n_less_than_inverse_n_minus_1_l230_230575

theorem a_n_less_than_inverse_n_minus_1 
  (n : ℕ) (h1 : 2 ≤ n) 
  (a : ℕ → ℝ) 
  (h2 : ∀ k : ℕ, 1 ≤ k → k ≤ n-1 → (a (k-1) + a k) * (a k + a (k+1)) = a (k-1) - a (k+1)) 
  (h3 : ∀ m : ℕ, m ≤ n → 0 < a m) : 
  a n < 1 / (n - 1) :=
sorry

end a_n_less_than_inverse_n_minus_1_l230_230575


namespace sin_theta_tan_theta_iff_first_third_quadrant_l230_230036

open Real

-- Definitions from conditions
def in_first_or_third_quadrant (θ : ℝ) : Prop :=
  (0 < θ ∧ θ < π / 2) ∨ (π < θ ∧ θ < 3 * π / 2)

def sin_theta_plus_tan_theta_positive (θ : ℝ) : Prop :=
  sin θ + tan θ > 0

-- Proof statement
theorem sin_theta_tan_theta_iff_first_third_quadrant (θ : ℝ) :
  sin_theta_plus_tan_theta_positive θ ↔ in_first_or_third_quadrant θ :=
sorry

end sin_theta_tan_theta_iff_first_third_quadrant_l230_230036


namespace least_integer_value_l230_230242

-- Define the condition and then prove the statement
theorem least_integer_value (x : ℤ) (h : 3 * |x| - 2 > 13) : x = -6 :=
by
  sorry

end least_integer_value_l230_230242


namespace min_S_value_l230_230046

theorem min_S_value (n : ℕ) (h₁ : n ≥ 375) :
    let R := 3000
    let S := 9 * n - R
    let dice_sum (s : ℕ) := ∃ L : List ℕ, (∀ x ∈ L, 1 ≤ x ∧ x ≤ 8) ∧ L.sum = s
    dice_sum R ∧ S = 375 := 
by
  sorry

end min_S_value_l230_230046


namespace largest_distinct_arithmetic_sequence_number_l230_230539

theorem largest_distinct_arithmetic_sequence_number :
  ∃ a b c d : ℕ, 
    (100 * a + 10 * b + c = 789) ∧ 
    (b - a = d) ∧ 
    (c - b = d) ∧ 
    (a ≠ b) ∧ 
    (b ≠ c) ∧ 
    (a ≠ c) ∧ 
    (a < 10) ∧ 
    (b < 10) ∧ 
    (c < 10) :=
sorry

end largest_distinct_arithmetic_sequence_number_l230_230539


namespace symmetry_axis_of_sine_function_l230_230829

theorem symmetry_axis_of_sine_function (x : ℝ) :
  (∃ k : ℤ, 2 * x + π / 4 = k * π + π / 2) ↔ x = π / 8 :=
by sorry

end symmetry_axis_of_sine_function_l230_230829


namespace total_pictures_on_wall_l230_230720

theorem total_pictures_on_wall (oil_paintings watercolor_paintings : ℕ) (h1 : oil_paintings = 9) (h2 : watercolor_paintings = 7) :
  oil_paintings + watercolor_paintings = 16 := 
by
  sorry

end total_pictures_on_wall_l230_230720


namespace tessa_initial_apples_l230_230564

theorem tessa_initial_apples (x : ℝ) (h : x + 5.0 - 4.0 = 11) : x = 10 :=
by
  sorry

end tessa_initial_apples_l230_230564


namespace sum_of_selected_primes_divisible_by_3_probability_l230_230823

def first_fifteen_primes : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def count_combinations_divisible_3 (nums : List ℕ) (k : ℕ) : ℕ :=
sorry -- Combines over the list to count combinations summing divisible by 3

noncomputable def probability_divisible_by_3 : ℚ :=
  let total_combinations := (Nat.choose 15 4)
  let favorable_combinations := count_combinations_divisible_3 first_fifteen_primes 4
  favorable_combinations / total_combinations

theorem sum_of_selected_primes_divisible_by_3_probability :
  probability_divisible_by_3 = 1/3 :=
sorry

end sum_of_selected_primes_divisible_by_3_probability_l230_230823


namespace B_alone_can_do_work_in_9_days_l230_230835

-- Define the conditions
def A_completes_work_in : ℕ := 15
def A_completes_portion_in (days : ℕ) : ℚ := days / 15
def portion_of_work_left (days : ℕ) : ℚ := 1 - A_completes_portion_in days
def B_completes_remaining_work_in_left_days (days_left : ℕ) : ℕ := 6
def B_completes_work_in (days_left : ℕ) : ℚ := B_completes_remaining_work_in_left_days days_left / (portion_of_work_left 5)

-- Define the theorem to be proven
theorem B_alone_can_do_work_in_9_days (days_left : ℕ) : B_completes_work_in days_left = 9 := by
  sorry

end B_alone_can_do_work_in_9_days_l230_230835


namespace well_diameter_l230_230050

noncomputable def calculateDiameter (volume depth : ℝ) : ℝ :=
  2 * Real.sqrt (volume / (Real.pi * depth))

theorem well_diameter :
  calculateDiameter 678.5840131753953 24 = 6 :=
by
  sorry

end well_diameter_l230_230050


namespace sector_radius_l230_230478

theorem sector_radius (P : ℝ) (c : ℝ → ℝ) (θ : ℝ) (r : ℝ) (π : ℝ) 
  (h1 : P = 144) 
  (h2 : θ = π)
  (h3 : P = θ * r + 2 * r) 
  (h4 : π = Real.pi)
  : r = 144 / (Real.pi + 2) := 
by
  sorry

end sector_radius_l230_230478


namespace lesser_fraction_l230_230202

theorem lesser_fraction (x y : ℚ) (hx : x + y = 13 / 14) (hy : x * y = 1 / 8) : 
  x = (13 - Real.sqrt 57) / 28 ∨ y = (13 - Real.sqrt 57) / 28 :=
by
  sorry

end lesser_fraction_l230_230202


namespace prove_fraction_l230_230632

variables {a : ℕ → ℝ} {b : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
∃ q : ℝ, ∀ n : ℕ, b (n + 1) = b n * q

def forms_arithmetic_sequence (x y z : ℝ) : Prop :=
2 * y = x + z

theorem prove_fraction
  (ha : is_arithmetic_sequence a)
  (hb : is_geometric_sequence b)
  (h_ar : forms_arithmetic_sequence (a 1 + 2 * b 1) (a 3 + 4 * b 3) (a 5 + 8 * b 5)) :
  (b 3 * b 7) / (b 4 ^ 2) = 1 / 4 :=
sorry

end prove_fraction_l230_230632


namespace ashok_total_subjects_l230_230015

/-- Ashok secured an average of 78 marks in some subjects. If the average of marks in 5 subjects 
is 74, and he secured 98 marks in the last subject, how many subjects are there in total? -/
theorem ashok_total_subjects (n : ℕ) 
  (avg_all : 78 * n = 74 * (n - 1) + 98) : n = 6 :=
sorry

end ashok_total_subjects_l230_230015


namespace find_k_from_hexadecimal_to_decimal_l230_230686

theorem find_k_from_hexadecimal_to_decimal 
  (k : ℕ) 
  (h : 1 * 6^3 + k * 6 + 5 = 239) : 
  k = 3 := by
  sorry

end find_k_from_hexadecimal_to_decimal_l230_230686


namespace machine_does_not_require_repair_l230_230266

variable (nominal_mass max_deviation standard_deviation : ℝ)
variable (nominal_mass_ge : nominal_mass ≥ 370)
variable (max_deviation_le : max_deviation ≤ 0.1 * nominal_mass)
variable (all_deviations_le_max : ∀ d, d < max_deviation → d < 37)
variable (std_dev_le_max_dev : standard_deviation ≤ max_deviation)

theorem machine_does_not_require_repair :
  ¬ (standard_deviation > 37) :=
by 
  -- sorry annotation indicates the proof goes here
  sorry

end machine_does_not_require_repair_l230_230266


namespace c_S_power_of_2_l230_230142

variables (m : ℕ) (S : String)

-- condition: m > 1
def is_valid_m (m : ℕ) : Prop := m > 1

-- function c(S)
def c (S : String) : ℕ := sorry  -- actual implementation is skipped

-- function to check if a number represented by a string is divisible by m
def is_divisible_by (n m : ℕ) : Prop := n % m = 0

-- Property that c(S) can take only powers of 2
def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

theorem c_S_power_of_2 (m : ℕ) (S : String) (h1 : is_valid_m m) :
  is_power_of_two (c S) :=
sorry

end c_S_power_of_2_l230_230142


namespace net_gain_loss_l230_230247

-- Definitions of the initial conditions
structure InitialState :=
  (cash_x : ℕ) (painting_value : ℕ) (cash_y : ℕ)

-- Definitions of transactions
structure Transaction :=
  (sell_price : ℕ) (commission_rate : ℕ)

def apply_transaction (initial_cash : ℕ) (tr : Transaction) : ℕ :=
  initial_cash + (tr.sell_price - (tr.sell_price * tr.commission_rate / 100))

def revert_transaction (initial_cash : ℕ) (tr : Transaction) : ℕ :=
  initial_cash - tr.sell_price + (tr.sell_price * tr.commission_rate / 100)

def compute_final_cash (initial_states : InitialState) (trans1 : Transaction) (trans2 : Transaction) : ℕ :=
  let cash_x_after_first := apply_transaction initial_states.cash_x trans1
  let cash_y_after_first := initial_states.cash_y - trans1.sell_price
  let cash_x_after_second := revert_transaction cash_x_after_first trans2
  let cash_y_after_second := cash_y_after_first + (trans2.sell_price - (trans2.sell_price * trans2.commission_rate / 100))
  cash_x_after_second - initial_states.cash_x + (cash_y_after_second - initial_states.cash_y)

-- Statement of the theorem
theorem net_gain_loss (initial_states : InitialState) (trans1 : Transaction) (trans2 : Transaction)
  (h1 : initial_states.cash_x = 15000)
  (h2 : initial_states.painting_value = 15000)
  (h3 : initial_states.cash_y = 18000)
  (h4 : trans1.sell_price = 20000)
  (h5 : trans1.commission_rate = 5)
  (h6 : trans2.sell_price = 14000)
  (h7 : trans2.commission_rate = 5) : 
  compute_final_cash initial_states trans1 trans2 = 5000 - 6700 :=
sorry

end net_gain_loss_l230_230247


namespace eccentricity_of_ellipse_l230_230177

open Real

theorem eccentricity_of_ellipse (a b c : ℝ) 
  (h1 : a > b ∧ b > 0)
  (h2 : c^2 = a^2 - b^2)
  (x : ℝ)
  (h3 : 3 * x = 2 * a)
  (h4 : sqrt 3 * x = 2 * c) :
  c / a = sqrt 3 / 3 :=
by
  sorry

end eccentricity_of_ellipse_l230_230177


namespace lowest_dropped_score_l230_230991

theorem lowest_dropped_score (A B C D : ℕ) 
  (h1 : (A + B + C + D) / 4 = 90)
  (h2 : (A + B + C) / 3 = 85) :
  D = 105 :=
by
  sorry

end lowest_dropped_score_l230_230991


namespace batting_average_drop_l230_230107

theorem batting_average_drop 
    (avg : ℕ)
    (innings : ℕ)
    (high : ℕ)
    (high_low_diff : ℕ)
    (low : ℕ)
    (total_runs : ℕ)
    (new_avg : ℕ)

    (h1 : avg = 50)
    (h2 : innings = 40)
    (h3 : high = 174)
    (h4 : high = low + 172)
    (h5 : total_runs = avg * innings)
    (h6 : new_avg = (total_runs - high - low) / (innings - 2)) :

  avg - new_avg = 2 :=
by
  sorry

end batting_average_drop_l230_230107


namespace trigonometric_identity_l230_230485

theorem trigonometric_identity :
  (let cos30 : ℝ := (Real.sqrt 3) / 2
   let sin60 : ℝ := (Real.sqrt 3) / 2
   let sin30 : ℝ := 1 / 2
   let cos60 : ℝ := 1 / 2
   (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1) :=
by
  sorry

end trigonometric_identity_l230_230485


namespace sequence_general_formula_l230_230888

theorem sequence_general_formula (a : ℕ → ℝ) (h₁ : a 1 = 1)
  (h₂ : ∀ n, a (n + 1) = a n / (1 + 2 * a n)) :
  ∀ n, a n = 1 / (2 * n - 1) :=
sorry

end sequence_general_formula_l230_230888


namespace probability_white_or_red_l230_230574

theorem probability_white_or_red (a b c : ℕ) : 
  (a + b) / (a + b + c) = (a + b) / (a + b + c) := by
  -- Conditions
  let total_balls := a + b + c
  let white_red_balls := a + b
  -- Goal
  have prob_white_or_red := white_red_balls / total_balls
  exact rfl

end probability_white_or_red_l230_230574


namespace pencils_on_desk_l230_230047

theorem pencils_on_desk (pencils_in_drawer pencils_on_desk_initial pencils_total pencils_placed : ℕ)
  (h_drawer : pencils_in_drawer = 43)
  (h_desk_initial : pencils_on_desk_initial = 19)
  (h_total : pencils_total = 78) :
  pencils_placed = 16 := by
  sorry

end pencils_on_desk_l230_230047


namespace expansion_term_count_l230_230668

theorem expansion_term_count 
  (A : Finset ℕ) (B : Finset ℕ) 
  (hA : A.card = 3) (hB : B.card = 4) : 
  (Finset.card (A.product B)) = 12 :=
by {
  sorry
}

end expansion_term_count_l230_230668


namespace current_walnut_trees_l230_230594

theorem current_walnut_trees (x : ℕ) (h : x + 55 = 77) : x = 22 :=
by
  sorry

end current_walnut_trees_l230_230594


namespace exists_natural_number_starting_and_ending_with_pattern_l230_230437

theorem exists_natural_number_starting_and_ending_with_pattern (n : ℕ) : 
  ∃ (m : ℕ), 
  (m % 10 = 1) ∧ 
  (∃ t : ℕ, 
    m^2 / 10^t = 10^(n - 1) * (10^n - 1) / 9) ∧ 
  (m^2 % 10^n = 1 ∨ m^2 % 10^n = 2) :=
sorry

end exists_natural_number_starting_and_ending_with_pattern_l230_230437


namespace racers_final_segment_l230_230679

def final_racer_count : Nat := 9

def segment_eliminations (init_count: Nat) : Nat :=
  let seg1 := init_count - Int.toNat (Nat.sqrt init_count)
  let seg2 := seg1 - seg1 / 3
  let seg3 := seg2 - (seg2 / 4 + (2 ^ 2))
  let seg4 := seg3 - seg3 / 3
  let seg5 := seg4 / 2
  let seg6 := seg5 - (seg5 * 3 / 4)
  seg6

theorem racers_final_segment
  (init_count: Nat)
  (h: init_count = 225) :
  segment_eliminations init_count = final_racer_count :=
  by
  rw [h]
  unfold segment_eliminations
  sorry

end racers_final_segment_l230_230679


namespace total_pay_is_correct_l230_230870

-- Define the weekly pay for employee B
def pay_B : ℝ := 228

-- Define the multiplier for employee A's pay relative to employee B's pay
def multiplier_A : ℝ := 1.5

-- Define the weekly pay for employee A
def pay_A : ℝ := multiplier_A * pay_B

-- Define the total weekly pay for both employees
def total_pay : ℝ := pay_A + pay_B

-- Prove the total pay
theorem total_pay_is_correct : total_pay = 570 := by
  -- Use the definitions and compute the total pay
  sorry

end total_pay_is_correct_l230_230870


namespace gcd_n_squared_plus_4_n_plus_3_l230_230798

theorem gcd_n_squared_plus_4_n_plus_3 (n : ℕ) (hn_gt_four : n > 4) : 
  (gcd (n^2 + 4) (n + 3)) = if n % 13 = 10 then 13 else 1 := 
sorry

end gcd_n_squared_plus_4_n_plus_3_l230_230798


namespace evaluate_h_j_l230_230865

def h (x : ℝ) : ℝ := 3 * x - 4
def j (x : ℝ) : ℝ := x - 2

theorem evaluate_h_j : h (2 + j 3) = 5 := by
  sorry

end evaluate_h_j_l230_230865


namespace iris_total_spending_l230_230259

theorem iris_total_spending :
  ∀ (price_jacket price_shorts price_pants : ℕ), 
  price_jacket = 10 → 
  price_shorts = 6 → 
  price_pants = 12 → 
  (3 * price_jacket + 2 * price_shorts + 4 * price_pants) = 90 :=
by
  intros price_jacket price_shorts price_pants
  sorry

end iris_total_spending_l230_230259


namespace sum_of_integers_l230_230274

theorem sum_of_integers (a b c : ℕ) :
  a > 1 → b > 1 → c > 1 →
  a * b * c = 1728 →
  gcd a b = 1 → gcd b c = 1 → gcd a c = 1 →
  a + b + c = 43 :=
by
  intro ha
  intro hb
  intro hc
  intro hproduct
  intro hgcd_ab
  intro hgcd_bc
  intro hgcd_ac
  sorry

end sum_of_integers_l230_230274


namespace circles_symmetric_sin_cos_l230_230035

noncomputable def sin_cos_product (θ : Real) : Real := Real.sin θ * Real.cos θ

theorem circles_symmetric_sin_cos (a θ : Real) 
(h1 : ∃ x1 y1, x1 = -a / 2 ∧ y1 = 0 ∧ 2*x1 - y1 - 1 = 0) 
(h2 : ∃ x2 y2, x2 = -a ∧ y2 = -Real.tan θ / 2 ∧ 2*x2 - y2 - 1 = 0) :
sin_cos_product θ = -2 / 5 := 
sorry

end circles_symmetric_sin_cos_l230_230035


namespace no_solution_l230_230423

theorem no_solution : ∀ x : ℝ, ¬ (3 * x + 2 < (x + 2)^2 ∧ (x + 2)^2 < 5 * x + 1) :=
by
  intro x
  -- Solve each part of the inequality
  have h1 : ¬ (3 * x + 2 < (x + 2)^2) ↔ x^2 + x + 2 ≤ 0 := by sorry
  have h2 : ¬ ((x + 2)^2 < 5 * x + 1) ↔ x^2 - x + 3 ≥ 0 := by sorry
  -- Combine the results
  exact sorry

end no_solution_l230_230423


namespace max_xy_under_constraint_l230_230790

theorem max_xy_under_constraint (x y : ℝ) (h1 : x + 2 * y = 1) (h2 : x > 0) (h3 : y > 0) : 
  xy ≤ 1 / 8 
  := sorry

end max_xy_under_constraint_l230_230790


namespace pairs_of_socks_now_l230_230481

def initial_socks : Nat := 28
def socks_thrown_away : Nat := 4
def socks_bought : Nat := 36

theorem pairs_of_socks_now : (initial_socks - socks_thrown_away + socks_bought) / 2 = 30 := by
  sorry

end pairs_of_socks_now_l230_230481


namespace total_chocolate_pieces_l230_230466

def total_chocolates (boxes : ℕ) (per_box : ℕ) : ℕ :=
  boxes * per_box

theorem total_chocolate_pieces :
  total_chocolates 6 500 = 3000 :=
by
  sorry

end total_chocolate_pieces_l230_230466


namespace haley_spent_32_dollars_l230_230017

noncomputable def total_spending (ticket_price : ℕ) (tickets_bought_self_friends : ℕ) (extra_tickets : ℕ) : ℕ :=
  ticket_price * (tickets_bought_self_friends + extra_tickets)

theorem haley_spent_32_dollars :
  total_spending 4 3 5 = 32 :=
by
  sorry

end haley_spent_32_dollars_l230_230017


namespace cube_and_fourth_power_remainders_l230_230001

theorem cube_and_fourth_power_remainders (
  b : Fin 2018 → ℕ) 
  (h1 : StrictMono b) 
  (h2 : (Finset.univ.sum b) = 2018^3) :
  ((Finset.univ.sum (λ i => b i ^ 3)) % 5 = 3) ∧
  ((Finset.univ.sum (λ i => b i ^ 4)) % 5 = 1) := 
sorry

end cube_and_fourth_power_remainders_l230_230001


namespace parallel_vectors_x_value_l230_230617

theorem parallel_vectors_x_value (x : ℝ) :
  (∀ k : ℝ, k ≠ 0 → (4, 2) = (k * x, k * (-3))) → x = -6 :=
by
  sorry

end parallel_vectors_x_value_l230_230617


namespace max_ratio_of_right_triangle_l230_230301

theorem max_ratio_of_right_triangle (a b c: ℝ) (h1: (1/2) * a * b = 30) (h2: a^2 + b^2 = c^2) : 
  (∀ x y z, (1/2 * x * y = 30) → (x^2 + y^2 = z^2) → 
  (x + y + z) / 30 ≤ (7.75 + 7.75 + 10.95) / 30) :=
by 
  sorry  -- The proof will show the maximum value is approximately 0.8817.

noncomputable def max_value := (7.75 + 7.75 + 10.95) / 30

end max_ratio_of_right_triangle_l230_230301


namespace acquaintances_condition_l230_230563

theorem acquaintances_condition (n : ℕ) (hn : n > 1) (acquainted : ℕ → ℕ → Prop) :
  (∀ X Y, acquainted X Y → acquainted Y X) ∧
  (∀ X, ¬acquainted X X) →
  (∀ n, n ≠ 2 → n ≠ 4 → ∃ (A B : ℕ), (∃ (C : ℕ), acquainted C A ∧ acquainted C B) ∨ (∃ (D : ℕ), ¬acquainted D A ∧ ¬acquainted D B)) :=
by
  intros
  sorry

end acquaintances_condition_l230_230563


namespace wire_cut_circle_square_area_eq_l230_230453

theorem wire_cut_circle_square_area_eq (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : (a^2 / (4 * π)) = ((b^2) / 16)) : 
  a / b = 2 / Real.sqrt π :=
by
  sorry

end wire_cut_circle_square_area_eq_l230_230453


namespace num_values_between_l230_230232

theorem num_values_between (x y : ℕ) (h1 : x + y ≥ 200) (h2 : x + y ≤ 1000) 
  (h3 : (x * (x - 1) + y * (y - 1)) * 2 = (x + y) * (x + y - 1)) : 
  ∃ n : ℕ, n - 1 = 17 := by
  sorry

end num_values_between_l230_230232


namespace ab_value_l230_230971

theorem ab_value (a b : ℝ) (log_two_3 : ℝ := Real.log 3 / Real.log 2) :
  a * log_two_3 = 1 ∧ (4 : ℝ)^b = 3 → a * b = 1 / 2 := by
  sorry

end ab_value_l230_230971


namespace sufficient_but_not_necessary_l230_230610

variables {a b : ℝ}

theorem sufficient_but_not_necessary (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
by sorry

end sufficient_but_not_necessary_l230_230610


namespace saeyoung_yen_value_l230_230943

-- Define the exchange rate
def exchange_rate : ℝ := 17.25

-- Define Saeyoung's total yuan
def total_yuan : ℝ := 1000 + 10

-- Define the total yen based on the exchange rate
def total_yen : ℝ := total_yuan * exchange_rate

-- State the theorem
theorem saeyoung_yen_value : total_yen = 17422.5 :=
by
  sorry

end saeyoung_yen_value_l230_230943


namespace distance_traveled_by_light_in_10_seconds_l230_230040

theorem distance_traveled_by_light_in_10_seconds :
  ∃ (a : ℝ) (n : ℕ), (300000 * 10 : ℝ) = a * 10 ^ n ∧ n = 6 :=
sorry

end distance_traveled_by_light_in_10_seconds_l230_230040


namespace probability_black_given_not_white_l230_230044

theorem probability_black_given_not_white
  (total_balls : ℕ)
  (white_balls : ℕ)
  (yellow_balls : ℕ)
  (black_balls : ℕ)
  (H1 : total_balls = 25)
  (H2 : white_balls = 10)
  (H3 : yellow_balls = 5)
  (H4 : black_balls = 10)
  (H5 : total_balls = white_balls + yellow_balls + black_balls)
  (H6 : ¬white_balls = total_balls) :
  (10 / (25 - 10) : ℚ) = 2 / 3 :=
by
  sorry

end probability_black_given_not_white_l230_230044


namespace quadratic_function_properties_l230_230606

noncomputable def f (x : ℝ) : ℝ := -5 / 2 * x^2 + 15 * x - 25 / 2

theorem quadratic_function_properties :
  (∃ a : ℝ, ∀ x : ℝ, (f x = a * (x - 1) * (x - 5)) ∧ (f 3 = 10)) → 
  (f x = -5 / 2 * x^2 + 15 * x - 25 / 2) :=
by 
  sorry

end quadratic_function_properties_l230_230606


namespace veronica_photo_choices_l230_230694

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

noncomputable def choose (n k : ℕ) : ℕ :=
factorial n / (factorial k * factorial (n - k))

theorem veronica_photo_choices : choose 5 3 + choose 5 4 = 15 := by
  sorry

end veronica_photo_choices_l230_230694


namespace average_age_of_new_students_l230_230103

theorem average_age_of_new_students :
  ∀ (initial_group_avg_age new_group_avg_age : ℝ) (initial_students new_students total_students : ℕ),
  initial_group_avg_age = 14 →
  initial_students = 10 →
  new_group_avg_age = 15 →
  new_students = 5 →
  total_students = initial_students + new_students →
  (new_group_avg_age * total_students - initial_group_avg_age * initial_students) / new_students = 17 :=
by
  intros initial_group_avg_age new_group_avg_age initial_students new_students total_students
  sorry

end average_age_of_new_students_l230_230103


namespace circle_equation_of_tangent_circle_l230_230628

theorem circle_equation_of_tangent_circle
  (h : ∀ x y: ℝ, x^2/4 - y^2 = 1 → (x = 2 ∨ x = -2) → y = 0)
  (asymptote : ∀ x y : ℝ, (y = (1/2)*x ∨ y = -(1/2)*x) → (x - 2)^2 + y^2 = (4/5))
  : ∃ k : ℝ, (∀ x y : ℝ, (x - 2)^2 + y^2 = k) → k = 4/5 := by
  sorry

end circle_equation_of_tangent_circle_l230_230628


namespace sabrina_total_leaves_l230_230659

theorem sabrina_total_leaves (num_basil num_sage num_verbena : ℕ)
  (h1 : num_basil = 2 * num_sage)
  (h2 : num_sage = num_verbena - 5)
  (h3 : num_basil = 12) :
  num_sage + num_verbena + num_basil = 29 :=
by
  sorry

end sabrina_total_leaves_l230_230659


namespace exists_nested_rectangles_l230_230537

theorem exists_nested_rectangles (rectangles : ℕ × ℕ → Prop) :
  (∀ n m : ℕ, rectangles (n, m)) → ∃ (n1 m1 n2 m2 : ℕ), n1 ≤ n2 ∧ m1 ≤ m2 ∧ rectangles (n1, m1) ∧ rectangles (n2, m2) :=
by {
  sorry
}

end exists_nested_rectangles_l230_230537


namespace number_of_integer_values_for_a_l230_230597

theorem number_of_integer_values_for_a :
  (∃ (a : Int), ∃ (p q : Int), p * q = -12 ∧ p + q = a ∧ p ≠ q) →
  (∃ (n : Nat), n = 6) := by
  sorry

end number_of_integer_values_for_a_l230_230597


namespace sin_double_angle_identity_l230_230121

theorem sin_double_angle_identity (x : ℝ) (h : Real.sin (x + π/4) = -3/5) : Real.sin (2 * x) = -7/25 := 
by 
  sorry

end sin_double_angle_identity_l230_230121


namespace quadratic_roots_identity_l230_230707

theorem quadratic_roots_identity
  (a b c : ℝ)
  (x1 x2 : ℝ)
  (hx1 : x1 = Real.sin (42 * Real.pi / 180))
  (hx2 : x2 = Real.sin (48 * Real.pi / 180))
  (hx2_trig_identity : x2 = Real.cos (42 * Real.pi / 180))
  (hroots : ∀ x, a * x^2 + b * x + c = 0 ↔ (x = x1 ∨ x = x2)) :
  b^2 = a^2 + 2 * a * c :=
by
  sorry

end quadratic_roots_identity_l230_230707


namespace smallest_sum_is_minus_half_l230_230168

def smallest_sum (x: ℝ) : ℝ := x^2 + x

theorem smallest_sum_is_minus_half : ∃ x : ℝ, ∀ y : ℝ, smallest_sum y ≥ smallest_sum (-1/2) :=
by
  use -1/2
  intros y
  sorry

end smallest_sum_is_minus_half_l230_230168


namespace pictures_per_album_l230_230826

-- Definitions based on the conditions
def phone_pics := 35
def camera_pics := 5
def total_pics := phone_pics + camera_pics
def albums := 5 

-- Statement that needs to be proven
theorem pictures_per_album : total_pics / albums = 8 := by
  sorry

end pictures_per_album_l230_230826


namespace roger_piles_of_quarters_l230_230924

theorem roger_piles_of_quarters (Q : ℕ) 
  (h₀ : ∃ Q : ℕ, True) 
  (h₁ : ∀ p, (p = Q) → True)
  (h₂ : ∀ c, (c = 7) → True) 
  (h₃ : Q * 14 = 42) : 
  Q = 3 := 
sorry

end roger_piles_of_quarters_l230_230924


namespace solution_to_system_l230_230533

theorem solution_to_system :
  (∀ (x y : ℚ), (y - x - 1 = 0) ∧ (y + x - 2 = 0) ↔ (x = 1/2 ∧ y = 3/2)) :=
by
  sorry

end solution_to_system_l230_230533


namespace area_of_triangle_ABC_l230_230285

variable {α : Type} [LinearOrder α] [Field α]

-- Given: 
variables (A B C D E F : α) (area_ABC area_BDA area_DCA : α)

-- Conditions:
variable (midpoint_D : 2 * D = B + C)
variable (ratio_AE_EC : 3 * E = A + C)
variable (ratio_AF_FD : 2 * F = A + D)
variable (area_DEF : area_ABC / 6 = 12)

-- To Show:
theorem area_of_triangle_ABC :
  area_ABC = 96 :=
by
  sorry

end area_of_triangle_ABC_l230_230285


namespace Jack_has_18_dimes_l230_230522

theorem Jack_has_18_dimes :
  ∃ d q : ℕ, (d = q + 3 ∧ 10 * d + 25 * q = 555) ∧ d = 18 :=
by
  sorry

end Jack_has_18_dimes_l230_230522


namespace sequence_an_form_l230_230045

-- Definitions based on the given conditions
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ := (n : ℝ)^2 * a n
def a_1 : ℝ := 1

-- The conjecture we need to prove
theorem sequence_an_form (a : ℕ → ℝ) (h₁ : ∀ n ≥ 2, sum_first_n_terms a n = (n : ℝ)^2 * a n)
  (h₂ : a 1 = a_1) :
  ∀ n ≥ 2, a n = 2 / (n * (n + 1)) :=
by
  sorry

end sequence_an_form_l230_230045


namespace number_of_men_in_club_l230_230008

variables (M W : ℕ)

theorem number_of_men_in_club 
  (h1 : M + W = 30) 
  (h2 : (1 / 3 : ℝ) * W + M = 18) : 
  M = 12 := 
sorry

end number_of_men_in_club_l230_230008


namespace factorize_l230_230710

theorem factorize (a b : ℝ) : 2 * a ^ 2 - 8 * b ^ 2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by 
  sorry

end factorize_l230_230710


namespace simplify_eval_expression_l230_230898

theorem simplify_eval_expression (x y : ℝ) (hx : x = -2) (hy : y = -1) :
  3 * (2 * x^2 + x * y + 1 / 3) - (3 * x^2 + 4 * x * y - y^2) = 11 :=
by
  rw [hx, hy]
  sorry

end simplify_eval_expression_l230_230898


namespace total_trophies_after_five_years_l230_230388

theorem total_trophies_after_five_years (michael_current_trophies : ℕ) (michael_increase : ℕ) (jack_multiplier : ℕ) (h1 : michael_current_trophies = 50) (h2 : michael_increase = 150) (h3 : jack_multiplier = 15) :
  let michael_five_years : ℕ := michael_current_trophies + michael_increase
  let jack_five_years : ℕ := jack_multiplier * michael_current_trophies
  michael_five_years + jack_five_years = 950 :=
by
  sorry

end total_trophies_after_five_years_l230_230388


namespace percentage_of_boys_l230_230519

def ratio_boys_girls := 2 / 3
def ratio_teacher_students := 1 / 6
def total_people := 36

theorem percentage_of_boys : ∃ (n_student n_teacher n_boys n_girls : ℕ), 
  n_student + n_teacher = 35 ∧
  n_student * (1 + 1/6) = total_people ∧
  n_boys / n_student = ratio_boys_girls ∧
  n_teacher / n_student = ratio_teacher_students ∧
  ((n_boys : ℚ) / total_people) * 100 = 400 / 7 :=
sorry

end percentage_of_boys_l230_230519


namespace log_sum_property_l230_230969

noncomputable def f (a : ℝ) (x : ℝ) := Real.log x / Real.log a
noncomputable def f_inv (a : ℝ) (y : ℝ) := a ^ y

theorem log_sum_property (a : ℝ) (h1 : f_inv a 2 = 9) (h2 : f a 9 = 2) : f a 9 + f a 6 = 1 :=
by
  sorry

end log_sum_property_l230_230969


namespace not_integer_division_l230_230797

def P : ℕ := 1
def Q : ℕ := 2

theorem not_integer_division : ¬ (∃ (n : ℤ), (P : ℤ) / (Q : ℤ) = n) := by
sorry

end not_integer_division_l230_230797


namespace extremal_values_d_l230_230450

theorem extremal_values_d (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ → Prop)
  (hC : ∀ (x y : ℝ), C (x, y) ↔ (x - 3)^2 + (y - 4)^2 = 1)
  (hA : A = (-1, 0)) (hB : B = (1, 0)) (hP : ∃ (x y : ℝ), C (x, y)) :
  ∃ (max_d min_d : ℝ), max_d = 14 ∧ min_d = 10 :=
by
  -- Necessary assumptions
  have h₁ : ∀ (x y : ℝ), C (x, y) ↔ (x - 3)^2 + (y - 4)^2 = 1 := hC
  have h₂ : A = (-1, 0) := hA
  have h₃ : B = (1, 0) := hB
  have h₄ : ∃ (x y : ℝ), C (x, y) := hP
  sorry

end extremal_values_d_l230_230450


namespace terry_nora_age_relation_l230_230329

variable {N : ℕ} -- Nora's current age

theorem terry_nora_age_relation (h₁ : Terry_current_age = 30) (h₂ : Terry_future_age = 4 * N) : N = 10 :=
by
  --- additional assumptions
  have Terry_future_age_def : Terry_future_age = 30 + 10 := by sorry
  rw [Terry_future_age_def] at h₂
  linarith

end terry_nora_age_relation_l230_230329


namespace question1_question2_l230_230595

variable (α : ℝ)

theorem question1 (h1 : (π / 2) < α) (h2 : α < π) (h3 : Real.sin α = 3 / 5) :
    (Real.sin α ^ 2 + Real.sin (2 * α)) / (Real.cos α ^ 2 + Real.cos (2 * α)) = -15 / 23 := by
  sorry

theorem question2 (h1 : (π / 2) < α) (h2 : α < π) (h3 : Real.sin α = 3 / 5) :
    Real.tan (α - 5 * π / 4) = -7 := by
  sorry

end question1_question2_l230_230595


namespace probability_perfect_square_l230_230849

theorem probability_perfect_square (choose_numbers : Finset (Fin 49)) (ticket : Finset (Fin 49))
  (h_choose_size : choose_numbers.card = 6) 
  (h_ticket_size : ticket.card = 6)
  (h_choose_square : ∃ (n : ℕ), (choose_numbers.prod id = n * n))
  (h_ticket_square : ∃ (m : ℕ), (ticket.prod id = m * m)) :
  ∃ T, (1 / T = 1 / T) :=
by
  sorry

end probability_perfect_square_l230_230849


namespace volume_of_prism_l230_230937

theorem volume_of_prism (x y z : ℝ) (h1 : x * y = 100) (h2 : z = 10) (h3 : x * z = 50) (h4 : y * z = 40):
  x * y * z = 200 :=
by
  sorry

end volume_of_prism_l230_230937


namespace tax_amount_self_employed_l230_230288

noncomputable def gross_income : ℝ := 350000.00
noncomputable def tax_rate : ℝ := 0.06

theorem tax_amount_self_employed :
  gross_income * tax_rate = 21000.00 :=
by
  sorry

end tax_amount_self_employed_l230_230288


namespace marble_weight_l230_230968

theorem marble_weight (m d : ℝ) : (9 * m = 4 * d) → (3 * d = 36) → (m = 16 / 3) :=
by
  intro h1 h2
  sorry

end marble_weight_l230_230968


namespace hypotenuse_length_l230_230309

theorem hypotenuse_length (a b c : ℝ) (h1 : a + b + c = 36) (h2 : 0.5 * a * b = 24) (h3 : a^2 + b^2 = c^2) :
  c = 50 / 3 :=
sorry

end hypotenuse_length_l230_230309


namespace greatest_possible_value_of_a_l230_230271

theorem greatest_possible_value_of_a :
  ∃ (a : ℕ), (∀ (x : ℤ), x * (x + a) = -21 → x^2 + a * x + 21 = 0) ∧
  (∀ (a' : ℕ), (∀ (x : ℤ), x * (x + a') = -21 → x^2 + a' * x + 21 = 0) → a' ≤ a) ∧
  a = 22 :=
sorry

end greatest_possible_value_of_a_l230_230271


namespace ab_zero_proof_l230_230561

-- Given conditions
def square_side : ℝ := 3
def rect_short_side : ℝ := 3
def rect_long_side : ℝ := 6
def rect_area : ℝ := rect_short_side * rect_long_side
def split_side_proof (a b : ℝ) : Prop := a + b = rect_short_side

-- Lean theorem proving that ab = 0 given the conditions
theorem ab_zero_proof (a b : ℝ) 
  (h1 : square_side = 3)
  (h2 : rect_short_side = 3)
  (h3 : rect_long_side = 6)
  (h4 : rect_area = 18)
  (h5 : split_side_proof a b) : a * b = 0 := by
  sorry

end ab_zero_proof_l230_230561


namespace david_lewis_meeting_point_l230_230197

theorem david_lewis_meeting_point :
  ∀ (D : ℝ),
  (∀ t : ℝ, t ≥ 0 →
    ∀ distance_to_meeting_point : ℝ, 
    distance_to_meeting_point = D →
    ∀ speed_david speed_lewis distance_cities : ℝ,
    speed_david = 50 →
    speed_lewis = 70 →
    distance_cities = 350 →
    ((distance_cities + distance_to_meeting_point) / speed_lewis = distance_to_meeting_point / speed_david) →
    D = 145.83) :=
by
  intros D t ht distance_to_meeting_point h_distance speed_david speed_lewis distance_cities h_speed_david h_speed_lewis h_distance_cities h_meeting_time
  -- We need to prove D = 145.83 under the given conditions
  sorry

end david_lewis_meeting_point_l230_230197


namespace min_value_x_plus_2y_l230_230303

variable (x y : ℝ) (hx : x > 0) (hy : y > 0)

theorem min_value_x_plus_2y (h : (2 / x) + (1 / y) = 1) : x + 2 * y ≥ 8 := 
  sorry

end min_value_x_plus_2y_l230_230303


namespace trigonometric_identity_l230_230640

theorem trigonometric_identity (x : ℝ) (h : Real.tan x = 2) : 
  (6 * Real.sin (2 * x) + 2 * Real.cos (2 * x)) / (Real.cos (2 * x) - 3 * Real.sin (2 * x)) = -2 / 5 := by
  sorry

end trigonometric_identity_l230_230640


namespace tangent_line_at_point_l230_230414

theorem tangent_line_at_point (x y : ℝ) (h_curve : y = Real.exp x - 2 * x) (h_point : (0, 1) = (x, y)) :
  x + y - 1 = 0 := 
by 
  sorry

end tangent_line_at_point_l230_230414


namespace soldiers_first_side_l230_230243

theorem soldiers_first_side (x : ℤ) (h1 : ∀ s1 : ℤ, s1 = 10)
                           (h2 : ∀ s2 : ℤ, s2 = 8)
                           (h3 : ∀ y : ℤ, y = x - 500)
                           (h4 : (10 * x + 8 * (x - 500)) = 68000) : x = 4000 :=
by
  -- Left blank for Lean to fill in the required proof steps
  sorry

end soldiers_first_side_l230_230243


namespace season_duration_l230_230043

-- Define the given conditions.
def games_per_month : ℕ := 7
def games_per_season : ℕ := 14

-- Define the property we want to prove.
theorem season_duration : games_per_season / games_per_month = 2 :=
by
  sorry

end season_duration_l230_230043


namespace convex_polyhedron_formula_l230_230644

theorem convex_polyhedron_formula
  (V E F t h T H : ℕ)
  (hF : F = 40)
  (hFaces : F = t + h)
  (hVertex : 2 * T + H = 7)
  (hEdges : E = (3 * t + 6 * h) / 2)
  (hEuler : V - E + F = 2)
  : 100 * H + 10 * T + V = 367 := 
sorry

end convex_polyhedron_formula_l230_230644


namespace quadratic_symmetry_l230_230483

noncomputable def f (x : ℝ) (a b : ℝ) := a * x^2 + b * x + 1

theorem quadratic_symmetry 
  (a b x1 x2 : ℝ) 
  (h_quad : f x1 a b = f x2 a b) 
  (h_diff : x1 ≠ x2) 
  (h_nonzero : a ≠ 0) :
  f (x1 + x2) a b = 1 := 
by
  sorry

end quadratic_symmetry_l230_230483


namespace math_problem_l230_230144

noncomputable def a : ℕ := 1265
noncomputable def b : ℕ := 168
noncomputable def c : ℕ := 21
noncomputable def d : ℕ := 6
noncomputable def e : ℕ := 3

theorem math_problem : 
  ( ( b / 100 : ℚ ) * (a ^ 2 / c) / (d - e ^ 2) : ℚ ) = -42646.27 :=
by sorry

end math_problem_l230_230144


namespace last_digit_sum_l230_230598

theorem last_digit_sum (a b : ℕ) (exp : ℕ)
  (h₁ : a = 1993) (h₂ : b = 1995) (h₃ : exp = 2002) :
  ((a ^ exp + b ^ exp) % 10) = 4 := 
by
  sorry

end last_digit_sum_l230_230598


namespace slope_of_line_between_solutions_l230_230994

theorem slope_of_line_between_solutions (x1 y1 x2 y2 : ℝ) (h1 : 3 / x1 + 4 / y1 = 0) (h2 : 3 / x2 + 4 / y2 = 0) (h3 : x1 ≠ x2) :
  (y2 - y1) / (x2 - x1) = -4 / 3 := 
sorry

end slope_of_line_between_solutions_l230_230994


namespace sin_75_l230_230132

theorem sin_75 :
  Real.sin (75 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  sorry

end sin_75_l230_230132


namespace max_sum_n_of_arithmetic_sequence_l230_230702

/-- Let \( S_n \) be the sum of the first \( n \) terms of an arithmetic sequence \( \{a_n\} \) with 
a non-zero common difference, and \( a_1 > 0 \). If \( S_5 = S_9 \), then when \( S_n \) is maximum, \( n = 7 \). -/
theorem max_sum_n_of_arithmetic_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) (d : ℕ) 
  (a1_pos : a 1 > 0) (common_difference : ∀ n, a (n + 1) = a n + d)
  (s5_eq_s9 : S 5 = S 9) :
  ∃ n, (∀ m, m ≤ n → S m ≤ S n) ∧ n = 7 :=
sorry

end max_sum_n_of_arithmetic_sequence_l230_230702


namespace simplify_expression_l230_230817

theorem simplify_expression :
  (Real.sqrt 15 + Real.sqrt 45 - (Real.sqrt (4/3) - Real.sqrt 108)) = 
  (Real.sqrt 15 + 3 * Real.sqrt 5 + 16 * Real.sqrt 3 / 3) :=
by
  sorry

end simplify_expression_l230_230817


namespace find_g_of_polynomial_l230_230179

variable (x : ℝ)

theorem find_g_of_polynomial :
  ∃ g : ℝ → ℝ, (4 * x^4 + 8 * x^3 + g x = 2 * x^4 - 5 * x^3 + 7 * x + 4) → (g x = -2 * x^4 - 13 * x^3 + 7 * x + 4) :=
sorry

end find_g_of_polynomial_l230_230179


namespace max_value_expression_l230_230557

theorem max_value_expression (x y : ℤ) (h : 3 * x^2 + 5 * y^2 = 345) : 
  ∃ (x y : ℤ), 3 * x^2 + 5 * y^2 = 345 ∧ (x + y = 13) := 
sorry

end max_value_expression_l230_230557


namespace max_value_of_function_f_l230_230703

noncomputable def f (t : ℝ) : ℝ := (4^t - 2 * t) * t / 16^t

theorem max_value_of_function_f : ∃ t : ℝ, ∀ x : ℝ, f x ≤ f t ∧ f t = 1 / 8 := sorry

end max_value_of_function_f_l230_230703


namespace correct_statement_l230_230141

-- Conditions as definitions
def deductive_reasoning (p q r : Prop) : Prop :=
  (p → q) → (q → r) → (p → r)

def correctness_of_conclusion := true  -- Indicates statement is defined to be correct

def pattern_of_reasoning (p q r : Prop) : Prop :=
  deductive_reasoning p q r

-- Statement to prove
theorem correct_statement (p q r : Prop) :
  pattern_of_reasoning p q r = deductive_reasoning p q r :=
by sorry

end correct_statement_l230_230141


namespace find_x_value_l230_230227

theorem find_x_value :
  let a := (2021 : ℝ)
  let b := (2022 : ℝ)
  ∀ x : ℝ, (a / b - b / a + x = 0) → (x = b / a - a / b) :=
  by
    intros a b x h
    sorry

end find_x_value_l230_230227


namespace find_value_of_c_l230_230279

variable (a b c : ℚ)
variable (x : ℚ)

-- Conditions converted to Lean statements
def condition1 := a = 2 * x ∧ b = 3 * x ∧ c = 7 * x
def condition2 := a - b + 3 = c - 2 * b

theorem find_value_of_c : condition1 x a b c ∧ condition2 a b c → c = 21 / 2 :=
by 
  sorry

end find_value_of_c_l230_230279


namespace two_trains_meet_at_distance_l230_230377

theorem two_trains_meet_at_distance 
  (D_slow D_fast : ℕ)  -- Distances traveled by the slower and faster trains
  (T : ℕ)  -- Time taken to meet
  (h0 : 16 * T = D_slow)  -- Distance formula for slower train
  (h1 : 21 * T = D_fast)  -- Distance formula for faster train
  (h2 : D_fast = D_slow + 60)  -- Faster train travels 60 km more than slower train
  : (D_slow + D_fast = 444) := sorry

end two_trains_meet_at_distance_l230_230377


namespace remainder_when_3x_7y_5z_div_31517_l230_230123

theorem remainder_when_3x_7y_5z_div_31517
  (x y z : ℕ)
  (hx : x % 23 = 9)
  (hy : y % 29 = 15)
  (hz : z % 37 = 12) :
  (3 * x + 7 * y - 5 * z) % 31517 = ((69 * (x / 23) + 203 * (y / 29) - 185 * (z / 37) + 72) % 31517) := 
sorry

end remainder_when_3x_7y_5z_div_31517_l230_230123


namespace smallest_possible_positive_value_l230_230169

theorem smallest_possible_positive_value (l w : ℕ) (hl : l > 0) (hw : w > 0) : ∃ x : ℕ, x = w - l + 1 ∧ x = 1 := 
by {
  sorry
}

end smallest_possible_positive_value_l230_230169


namespace Lily_balls_is_3_l230_230206

-- Definitions from conditions
variable (L : ℕ)

def Frodo_balls := L + 8
def Brian_balls := 2 * (L + 8)

axiom Brian_has_22 : Brian_balls L = 22

-- The goal is to prove that Lily has 3 tennis balls
theorem Lily_balls_is_3 : L = 3 :=
by
  sorry

end Lily_balls_is_3_l230_230206


namespace number_of_children_l230_230979

theorem number_of_children (total_passengers men women : ℕ) (h1 : total_passengers = 54) (h2 : men = 18) (h3 : women = 26) : 
  total_passengers - men - women = 10 :=
by sorry

end number_of_children_l230_230979


namespace solve_system_l230_230038

theorem solve_system :
  ∀ x y : ℚ, (3 * x + 4 * y = 12) ∧ (9 * x - 12 * y = -24) →
  (x = 2 / 3) ∧ (y = 5 / 2) :=
by
  intro x y
  intro h
  sorry

end solve_system_l230_230038


namespace scientific_notation_correct_l230_230099

/-- Given the weight of the "人" shaped gate of the Three Gorges ship lock -/
def weight_kg : ℝ := 867000

/-- The scientific notation representation of the given weight -/
def scientific_notation_weight_kg : ℝ := 8.67 * 10^5

theorem scientific_notation_correct :
  weight_kg = scientific_notation_weight_kg :=
sorry

end scientific_notation_correct_l230_230099


namespace calculate_expression_l230_230262

theorem calculate_expression :
  ( (5^1010)^2 - (5^1008)^2) / ( (5^1009)^2 - (5^1007)^2) = 25 := 
by
  sorry

end calculate_expression_l230_230262


namespace evaluate_expression_l230_230287

def S (a b c : ℤ) := a + b + c

theorem evaluate_expression (a b c : ℤ) (h1 : a = 12) (h2 : b = 14) (h3 : c = 18) :
  (144 * ((1 : ℚ) / b - (1 : ℚ) / c) + 196 * ((1 : ℚ) / c - (1 : ℚ) / a) + 324 * ((1 : ℚ) / a - (1 : ℚ) / b)) /
  (12 * ((1 : ℚ) / b - (1 : ℚ) / c) + 14 * ((1 : ℚ) / c - (1 : ℚ) / a) + 18 * ((1 : ℚ) / a - (1 : ℚ) / b)) = 44 := 
sorry

end evaluate_expression_l230_230287


namespace opposite_of_neg_sqrt_two_l230_230955

theorem opposite_of_neg_sqrt_two : -(-Real.sqrt 2) = Real.sqrt 2 := 
by {
  sorry
}

end opposite_of_neg_sqrt_two_l230_230955


namespace christine_siri_total_money_l230_230820

-- Define the conditions
def christine_has_more_than_siri : ℝ := 20 -- Christine has 20 rs more than Siri
def christine_amount : ℝ := 20.5 -- Christine has 20.5 rs

-- Define the proof problem
theorem christine_siri_total_money :
  (∃ (siri_amount : ℝ), christine_amount = siri_amount + christine_has_more_than_siri) →
  ∃ total : ℝ, total = christine_amount + (christine_amount - christine_has_more_than_siri) ∧ total = 21 :=
by sorry

end christine_siri_total_money_l230_230820


namespace rachels_milk_consumption_l230_230392

theorem rachels_milk_consumption :
  let bottle1 := (3 / 8 : ℚ)
  let bottle2 := (1 / 4 : ℚ)
  let total_milk := bottle1 + bottle2
  let rachel_ratio := (3 / 4 : ℚ)
  rachel_ratio * total_milk = (15 / 32 : ℚ) :=
by
  let bottle1 := (3 / 8 : ℚ)
  let bottle2 := (1 / 4 : ℚ)
  let total_milk := bottle1 + bottle2
  let rachel_ratio := (3 / 4 : ℚ)
  -- proof placeholder
  sorry

end rachels_milk_consumption_l230_230392


namespace part_a_39x55_5x11_l230_230839

theorem part_a_39x55_5x11 :
  ¬ (∃ (a1 a2 b1 b2 : ℕ), 
    39 = 5 * a1 + 11 * b1 ∧ 
    55 = 5 * a2 + 11 * b2) := 
  by sorry

end part_a_39x55_5x11_l230_230839


namespace no_such_function_exists_l230_230705

theorem no_such_function_exists : ¬ ∃ f : ℕ → ℕ, ∀ n > 2, f (f (n - 1)) = f (n + 1) - f n :=
by {
  sorry
}

end no_such_function_exists_l230_230705


namespace cut_wire_l230_230822

theorem cut_wire (x y : ℕ) : 
  15 * x + 12 * y = 102 ↔ (x = 2 ∧ y = 6) ∨ (x = 6 ∧ y = 1) :=
by
  sorry

end cut_wire_l230_230822


namespace year_when_P_costs_40_paise_more_than_Q_l230_230412

def price_of_P (n : ℕ) : ℝ := 4.20 + 0.40 * n
def price_of_Q (n : ℕ) : ℝ := 6.30 + 0.15 * n

theorem year_when_P_costs_40_paise_more_than_Q :
  ∃ n : ℕ, price_of_P n = price_of_Q n + 0.40 ∧ 2001 + n = 2011 :=
by
  sorry

end year_when_P_costs_40_paise_more_than_Q_l230_230412


namespace num_valid_m_values_for_distributing_marbles_l230_230070

theorem num_valid_m_values_for_distributing_marbles : 
  ∃ (m_values : Finset ℕ), m_values.card = 22 ∧ 
  ∀ m ∈ m_values, ∃ n : ℕ, m * n = 360 ∧ n > 1 ∧ m > 1 :=
by
  sorry

end num_valid_m_values_for_distributing_marbles_l230_230070


namespace greatest_three_digit_number_condition_l230_230479

theorem greatest_three_digit_number_condition :
  ∃ n : ℕ, (100 ≤ n) ∧ (n ≤ 999) ∧ (n % 7 = 2) ∧ (n % 6 = 4) ∧ (n = 982) := 
by
  sorry

end greatest_three_digit_number_condition_l230_230479


namespace cost_of_bananas_and_cantaloupe_l230_230558

theorem cost_of_bananas_and_cantaloupe (a b c d h : ℚ) 
  (h1: a + b + c + d + h = 30)
  (h2: d = 4 * a)
  (h3: c = 2 * a - b) :
  b + c = 50 / 7 := 
sorry

end cost_of_bananas_and_cantaloupe_l230_230558


namespace number_of_girls_in_first_year_l230_230832

theorem number_of_girls_in_first_year
  (total_students : ℕ)
  (sample_size : ℕ)
  (boys_in_sample : ℕ)
  (girls_in_first_year : ℕ) :
  total_students = 2400 →
  sample_size = 80 →
  boys_in_sample = 42 →
  girls_in_first_year = total_students * (sample_size - boys_in_sample) / sample_size →
  girls_in_first_year = 1140 :=
by 
  intros h1 h2 h3 h4
  sorry

end number_of_girls_in_first_year_l230_230832


namespace expand_polynomials_eq_l230_230541

-- Define the polynomials P(z) and Q(z)
def P (z : ℝ) : ℝ := 3 * z^3 + 2 * z^2 - 4 * z + 1
def Q (z : ℝ) : ℝ := 4 * z^4 - 3 * z^2 + 2

-- Define the result polynomial R(z)
def R (z : ℝ) : ℝ := 12 * z^7 + 8 * z^6 - 25 * z^5 - 2 * z^4 + 18 * z^3 + z^2 - 8 * z + 2

-- State the theorem that proves P(z) * Q(z) = R(z)
theorem expand_polynomials_eq :
  ∀ (z : ℝ), (P z) * (Q z) = R z :=
by
  intros z
  sorry

end expand_polynomials_eq_l230_230541


namespace company_food_purchase_1_l230_230306

theorem company_food_purchase_1 (x y : ℕ) (h1: x + y = 170) (h2: 15 * x + 20 * y = 3000) : 
  x = 80 ∧ y = 90 := by
  sorry

end company_food_purchase_1_l230_230306


namespace fred_seashells_now_l230_230284

def seashells_initial := 47
def seashells_given := 25

theorem fred_seashells_now : seashells_initial - seashells_given = 22 := 
by 
  sorry

end fred_seashells_now_l230_230284


namespace expected_value_of_winning_is_2550_l230_230128

-- Definitions based on the conditions
def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]
def probability (n : ℕ) : ℚ := 1 / 8
def winnings (n : ℕ) : ℕ := n^2

-- Expected value calculation based on the conditions
noncomputable def expected_value : ℚ :=
  (outcomes.map (λ n => probability n * winnings n)).sum

-- Proposition stating that the expected value is 25.50
theorem expected_value_of_winning_is_2550 : expected_value = 25.50 :=
by
  sorry

end expected_value_of_winning_is_2550_l230_230128


namespace triangle_area_l230_230117

theorem triangle_area (base height : ℕ) (h_base : base = 35) (h_height : height = 12) :
  (1 / 2 : ℚ) * base * height = 210 := by
  sorry

end triangle_area_l230_230117


namespace Jori_water_left_l230_230411

theorem Jori_water_left (a b : ℚ) (h1 : a = 7/2) (h2 : b = 7/4) : a - b = 7/4 := by
  sorry

end Jori_water_left_l230_230411


namespace range_of_m_l230_230701

def proposition_p (m : ℝ) : Prop :=
  ∀ (x : ℝ), (1 ≤ x) → (x^2 - 2*m*x + 1/2 > 0)

def proposition_q (m : ℝ) : Prop :=
  ∃ (x : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (x^2 - m*x - 2 = 0)

theorem range_of_m (m : ℝ) (h1 : ¬ proposition_q m) (h2 : proposition_p m ∨ proposition_q m) :
  -1 < m ∧ m < 3/4 :=
  sorry

end range_of_m_l230_230701


namespace proof_problem_l230_230189

variable {a : ℕ → ℝ} -- sequence a
variable {S : ℕ → ℝ} -- partial sums sequence S 
variable {n : ℕ} -- index

-- Define the conditions
def is_arith_seq (a : ℕ → ℝ) : Prop := 
  ∃ d, ∀ n, a (n+1) = a n + d

def S_is_partial_sum (a S : ℕ → ℝ) : Prop := 
  ∀ n, S (n+1) = S n + a (n+1)

-- The properties given in the problem
def conditions (a S : ℕ → ℝ) : Prop :=
  is_arith_seq a ∧ 
  S_is_partial_sum a S ∧ 
  S 6 < S 7 ∧ 
  S 7 > S 8

-- The conclusions that need to be proved
theorem proof_problem (a S : ℕ → ℝ) (h : conditions a S) : 
  S 9 < S 6 ∧
  (∀ n, a 1 ≥ a (n+1)) ∧
  (∀ m, S 7 ≥ S m) := by 
  sorry

end proof_problem_l230_230189


namespace gas_fee_calculation_l230_230308

theorem gas_fee_calculation (x : ℚ) (h_usage : x > 60) :
  60 * 0.8 + (x - 60) * 1.2 = 0.88 * x → x * 0.88 = 66 := by
  sorry

end gas_fee_calculation_l230_230308


namespace negation_of_universal_prop_l230_230677

-- Define the conditions
variable (f : ℝ → ℝ)

-- Theorem statement
theorem negation_of_universal_prop : 
  (¬ ∀ x : ℝ, f x > 0) ↔ (∃ x : ℝ, f x ≤ 0) :=
by
  sorry

end negation_of_universal_prop_l230_230677


namespace simplify_expr_l230_230023

theorem simplify_expr (x y : ℝ) (P Q : ℝ) (hP : P = x^2 + y^2) (hQ : Q = x^2 - y^2) : 
  (P * Q / (P + Q)) + ((P + Q) / (P * Q)) = ((x^4 + y^4) ^ 2) / (2 * x^2 * (x^4 - y^4)) :=
by sorry

end simplify_expr_l230_230023


namespace units_digit_of_150_factorial_is_zero_l230_230827

theorem units_digit_of_150_factorial_is_zero : (Nat.factorial 150) % 10 = 0 := by
sorry

end units_digit_of_150_factorial_is_zero_l230_230827


namespace translated_circle_eq_l230_230900

theorem translated_circle_eq (x y : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = 16) →
  (x + 5) ^ 2 + (y + 3) ^ 2 = 16 :=
by
  sorry

end translated_circle_eq_l230_230900


namespace last_digit_base_4_of_77_l230_230006

theorem last_digit_base_4_of_77 : (77 % 4) = 1 :=
by
  sorry

end last_digit_base_4_of_77_l230_230006


namespace pythagorean_set_A_l230_230795

theorem pythagorean_set_A : 
  let x := Real.sqrt 2
  let y := Real.sqrt 3
  let z := Real.sqrt 5
  x^2 + y^2 = z^2 := 
by
  let x := Real.sqrt 2
  let y := Real.sqrt 3
  let z := Real.sqrt 5
  sorry

end pythagorean_set_A_l230_230795


namespace max_license_plates_l230_230137

noncomputable def max_distinct_plates (m n : ℕ) : ℕ :=
  m ^ (n - 1)

theorem max_license_plates :
  max_distinct_plates 10 6 = 100000 := by
  sorry

end max_license_plates_l230_230137


namespace four_roots_sum_eq_neg8_l230_230506

def op (a b : ℝ) : ℝ := a^2 + 2 * a * b - b^2

def f (x : ℝ) : ℝ := op x 2

theorem four_roots_sum_eq_neg8 :
  ∃ (x1 x2 x3 x4 : ℝ), 
  (x1 ≠ -2) ∧ (x2 ≠ -2) ∧ (x3 ≠ -2) ∧ (x4 ≠ -2) ∧
  (f x1 = Real.log (abs (x1 + 2))) ∧ 
  (f x2 = Real.log (abs (x2 + 2))) ∧ 
  (f x3 = Real.log (abs (x3 + 2))) ∧ 
  (f x4 = Real.log (abs (x4 + 2))) ∧ 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ 
  x2 ≠ x3 ∧ x2 ≠ x4 ∧ 
  x3 ≠ x4 ∧ 
  x1 + x2 + x3 + x4 = -8 :=
by 
  sorry

end four_roots_sum_eq_neg8_l230_230506


namespace heart_op_ratio_l230_230212

def heart_op (n m : ℕ) : ℕ := n^3 * m^2

theorem heart_op_ratio : heart_op 3 5 / heart_op 5 3 = 5 / 9 := 
by 
  sorry

end heart_op_ratio_l230_230212


namespace expected_value_is_correct_l230_230111

def probability_of_rolling_one : ℚ := 1 / 4

def probability_of_other_numbers : ℚ := 3 / 4 / 5

def win_amount : ℚ := 8

def loss_amount : ℚ := -3

def expected_value : ℚ := (probability_of_rolling_one * win_amount) + 
                          (probability_of_other_numbers * 5 * loss_amount)

theorem expected_value_is_correct : expected_value = -0.25 :=
by 
  unfold expected_value probability_of_rolling_one probability_of_other_numbers win_amount loss_amount
  sorry

end expected_value_is_correct_l230_230111


namespace smallest_a_such_that_sqrt_50a_is_integer_l230_230845

theorem smallest_a_such_that_sqrt_50a_is_integer : ∃ a : ℕ, (∀ b : ℕ, (b > 0 ∧ (∃ k : ℕ, 50 * b = k^2)) → (a ≤ b)) ∧ (∃ k : ℕ, 50 * a = k^2) ∧ a = 2 := 
by
  sorry

end smallest_a_such_that_sqrt_50a_is_integer_l230_230845


namespace temperature_fraction_l230_230717

def current_temperature : ℤ := 84
def temperature_decrease : ℤ := 21

theorem temperature_fraction :
  (current_temperature - temperature_decrease) = (3 * current_temperature / 4) := 
by
  sorry

end temperature_fraction_l230_230717


namespace sum_of_consecutive_evens_l230_230361

theorem sum_of_consecutive_evens (E1 E2 E3 E4 : ℕ) (h1 : E4 = 38) (h2 : E3 = E4 - 2) (h3 : E2 = E3 - 2) (h4 : E1 = E2 - 2) : 
  E1 + E2 + E3 + E4 = 140 := 
by 
  sorry

end sum_of_consecutive_evens_l230_230361


namespace gcd_of_polynomials_l230_230454

theorem gcd_of_polynomials (n : ℕ) (h : n > 2^5) : gcd (n^3 + 5^2) (n + 6) = 1 :=
by sorry

end gcd_of_polynomials_l230_230454


namespace number_of_adult_tickets_l230_230524

-- Let's define our conditions and the theorem to prove.
theorem number_of_adult_tickets (A C : ℕ) (h₁ : A + C = 522) (h₂ : 15 * A + 8 * C = 5086) : A = 131 :=
by
  sorry

end number_of_adult_tickets_l230_230524


namespace determine_a_l230_230728

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 + (Real.exp x - a)^2

theorem determine_a (a x₀ : ℝ)
  (h₀ : f x₀ a ≤ 1/2) : a = 1/2 :=
sorry

end determine_a_l230_230728


namespace max_terms_in_arithmetic_seq_l230_230953

variable (a n : ℝ)

def arithmetic_seq_max_terms (a n : ℝ) : Prop :=
  let d := 4
  a^2 + (n - 1) * (a + d) + (n - 1) * n / 2 * d ≤ 100

theorem max_terms_in_arithmetic_seq (a n : ℝ) (h : arithmetic_seq_max_terms a n) : n ≤ 8 :=
sorry

end max_terms_in_arithmetic_seq_l230_230953


namespace average_temperature_is_95_l230_230553

noncomputable def tempNY := 80
noncomputable def tempMiami := tempNY + 10
noncomputable def tempSD := tempMiami + 25
noncomputable def avg_temp := (tempNY + tempMiami + tempSD) / 3

theorem average_temperature_is_95 :
  avg_temp = 95 :=
by
  sorry

end average_temperature_is_95_l230_230553


namespace vegetarian_eaters_l230_230712

-- Define the conditions
theorem vegetarian_eaters : 
  ∀ (total family_size : ℕ) 
  (only_veg only_nonveg both_veg_nonveg eat_veg : ℕ), 
  family_size = 45 → 
  only_veg = 22 → 
  only_nonveg = 15 → 
  both_veg_nonveg = 8 → 
  eat_veg = only_veg + both_veg_nonveg → 
  eat_veg = 30 :=
by
  intros total family_size only_veg only_nonveg both_veg_nonveg eat_veg
  sorry

end vegetarian_eaters_l230_230712


namespace min_students_with_both_l230_230546

-- Given conditions
def total_students : ℕ := 35
def students_with_brown_eyes : ℕ := 18
def students_with_lunch_box : ℕ := 25

-- Mathematical statement to prove the least number of students with both attributes
theorem min_students_with_both :
  ∃ x : ℕ, students_with_brown_eyes + students_with_lunch_box - total_students ≤ x ∧ x = 8 :=
sorry

end min_students_with_both_l230_230546


namespace fraction_remains_unchanged_l230_230999

theorem fraction_remains_unchanged (x y : ℝ) : 
  (3 * (3 * x)) / (2 * (3 * x) - 3 * y) = (3 * x) / (2 * x - y) :=
by
  sorry

end fraction_remains_unchanged_l230_230999


namespace ratio_of_numbers_l230_230737

theorem ratio_of_numbers (a b : ℕ) (h1 : a.gcd b = 5) (h2 : a.lcm b = 60) (h3 : a = 3 * 5) (h4 : b = 4 * 5) : (a / a.gcd b) / (b / a.gcd b) = 3 / 4 :=
by
  sorry

end ratio_of_numbers_l230_230737


namespace total_bees_including_queen_at_end_of_14_days_l230_230762

-- Conditions definitions
def bees_hatched_per_day : ℕ := 5000
def bees_lost_per_day : ℕ := 1800
def duration_days : ℕ := 14
def initial_bees : ℕ := 20000
def queen_bees : ℕ := 1

-- Question statement as Lean theorem
theorem total_bees_including_queen_at_end_of_14_days :
  (initial_bees + (bees_hatched_per_day - bees_lost_per_day) * duration_days + queen_bees) = 64801 := 
by
  sorry

end total_bees_including_queen_at_end_of_14_days_l230_230762


namespace find_a_l230_230934

-- Define the function f
def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

-- Define the derivative of f
def f_prime (x a b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

theorem find_a (a b : ℝ) (h1 : f_prime 1 a b = 0) (h2 : f 1 a b = 10) : a = 4 :=
by
  sorry

end find_a_l230_230934


namespace scientific_notation_361000000_l230_230571

theorem scientific_notation_361000000 :
  361000000 = 3.61 * 10^8 :=
sorry

end scientific_notation_361000000_l230_230571


namespace proof_problem_l230_230964

variable (a b c : ℝ)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem proof_problem 
  (h0 : f a b c 0 = f a b c 4)
  (h1 : f a b c 0 > f a b c 1) : 
  a > 0 ∧ 4 * a + b = 0 :=
by
  sorry

end proof_problem_l230_230964


namespace nearest_integer_to_expansion_l230_230113

theorem nearest_integer_to_expansion : 
  let a := (3 + 2 * Real.sqrt 2)
  let b := (3 - 2 * Real.sqrt 2)
  abs (a^4 - 1090) < 1 :=
by
  let a := (3 + 2 * Real.sqrt 2)
  let b := (3 - 2 * Real.sqrt 2)
  sorry

end nearest_integer_to_expansion_l230_230113


namespace ball_bounce_height_l230_230695

open Real

theorem ball_bounce_height :
  ∃ k : ℕ, (20 * (2 / 3 : ℝ)^k < 2) ∧ (∀ n : ℕ, n < k → 20 * (2 / 3 : ℝ)^n ≥ 2) ∧ k = 6 :=
sorry

end ball_bounce_height_l230_230695


namespace ring_stack_vertical_distance_l230_230804

theorem ring_stack_vertical_distance :
  let ring_thickness := 2
  let top_ring_outer_diameter := 36
  let bottom_ring_outer_diameter := 12
  let decrement := 2
  ∃ n, (top_ring_outer_diameter - bottom_ring_outer_diameter) / decrement + 1 = n ∧
       n * ring_thickness = 260 :=
by {
  let ring_thickness := 2
  let top_ring_outer_diameter := 36
  let bottom_ring_outer_diameter := 12
  let decrement := 2
  sorry
}

end ring_stack_vertical_distance_l230_230804


namespace evaluate_expression_l230_230733

theorem evaluate_expression : (2^2010 * 3^2012 * 25) / 6^2011 = 37.5 := by
  sorry

end evaluate_expression_l230_230733


namespace solve_consecutive_integers_solve_consecutive_even_integers_l230_230770

-- Conditions: x, y, z, w are positive integers and x + y + z + w = 46.
def consecutive_integers_solution (x y z w : ℕ) : Prop :=
  x < y ∧ y < z ∧ z < w ∧ (x + 1 = y) ∧ (y + 1 = z) ∧ (z + 1 = w) ∧ (x + y + z + w = 46)

def consecutive_even_integers_solution (x y z w : ℕ) : Prop :=
  x < y ∧ y < z ∧ z < w ∧ (x + 2 = y) ∧ (y + 2 = z) ∧ (z + 2 = w) ∧ (x + y + z + w = 46)

-- Proof that consecutive integers can solve the equation II (x + y + z + w = 46)
theorem solve_consecutive_integers : ∃ x y z w : ℕ, consecutive_integers_solution x y z w :=
sorry

-- Proof that consecutive even integers can solve the equation II (x + y + z + w = 46)
theorem solve_consecutive_even_integers : ∃ x y z w : ℕ, consecutive_even_integers_solution x y z w :=
sorry

end solve_consecutive_integers_solve_consecutive_even_integers_l230_230770


namespace number_of_roses_picked_later_l230_230796

-- Given definitions
def initial_roses : ℕ := 50
def sold_roses : ℕ := 15
def final_roses : ℕ := 56

-- Compute the number of roses left after selling.
def roses_left := initial_roses - sold_roses

-- Define the final goal: number of roses picked later.
def picked_roses_later := final_roses - roses_left

-- State the theorem
theorem number_of_roses_picked_later : picked_roses_later = 21 :=
by
  sorry

end number_of_roses_picked_later_l230_230796


namespace remaining_number_larger_than_4_l230_230521

theorem remaining_number_larger_than_4 (m : ℕ) (h : 2 ≤ m) (a : ℚ) (b : ℚ) (h_sum_inv : (1 : ℚ) - 1 / (2 * m + 1 : ℚ) = 3 / 4 + 1 / b) :
  b > 4 :=
by sorry

end remaining_number_larger_than_4_l230_230521


namespace max_value_of_expression_l230_230029

noncomputable def f (x y : ℝ) := x * y^2 * (x^2 + x + 1) * (y^2 + y + 1)

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  ∃ m, m = 951625 / 256 ∧ ∀ a b : ℝ, a + b = 5 → f a b ≤ m :=
sorry

end max_value_of_expression_l230_230029


namespace solve_for_y_l230_230129

noncomputable def determinant3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h

noncomputable def determinant2x2 (a b c d : ℝ) : ℝ := 
  a*d - b*c

theorem solve_for_y (b y : ℝ) (h : b ≠ 0) :
  determinant3x3 (y + 2 * b) y y y (y + 2 * b) y y y (y + 2 * b) = 0 → 
  y = -b / 2 :=
by
  sorry

end solve_for_y_l230_230129


namespace find_length_of_AB_l230_230368

open Real

theorem find_length_of_AB (A B C : ℝ) 
    (h1 : tan A = 3 / 4) 
    (h2 : B = 6) 
    (h3 : C = π / 2) : sqrt (B^2 + ((3/4) * B)^2) = 7.5 :=
by
  sorry

end find_length_of_AB_l230_230368


namespace credit_card_balance_l230_230446

theorem credit_card_balance :
  ∀ (initial_balance groceries_charge gas_charge return_credit : ℕ),
  initial_balance = 126 →
  groceries_charge = 60 →
  gas_charge = groceries_charge / 2 →
  return_credit = 45 →
  initial_balance + groceries_charge + gas_charge - return_credit = 171 :=
by
  intros initial_balance groceries_charge gas_charge return_credit
  intros h_initial h_groceries h_gas h_return
  rw [h_initial, h_groceries, h_gas, h_return]
  norm_num
  sorry

end credit_card_balance_l230_230446


namespace market_trips_l230_230634

theorem market_trips (d_school_round: ℝ) (d_market_round: ℝ) (num_school_trips_per_day: ℕ) (num_school_days_per_week: ℕ) (total_week_mileage: ℝ) :
  d_school_round = 5 →
  d_market_round = 4 →
  num_school_trips_per_day = 2 →
  num_school_days_per_week = 4 →
  total_week_mileage = 44 →
  (total_week_mileage - (d_school_round * num_school_trips_per_day * num_school_days_per_week)) / d_market_round = 1 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end market_trips_l230_230634


namespace remaining_wire_length_l230_230358

theorem remaining_wire_length (total_length : ℝ) (fraction_cut : ℝ) (remaining_length : ℝ) (h1 : total_length = 3) (h2 : fraction_cut = 1 / 3) (h3 : remaining_length = 2) :
  total_length * (1 - fraction_cut) = remaining_length :=
by
  -- Proof goes here
  sorry

end remaining_wire_length_l230_230358


namespace beths_total_crayons_l230_230473

def packs : ℕ := 4
def crayons_per_pack : ℕ := 10
def extra_crayons : ℕ := 6

theorem beths_total_crayons : packs * crayons_per_pack + extra_crayons = 46 := by
  sorry

end beths_total_crayons_l230_230473


namespace probability_walk_450_feet_or_less_l230_230359

theorem probability_walk_450_feet_or_less 
  (gates : List ℕ) (initial_gate new_gate : ℕ) 
  (n : ℕ) (dist_between_adjacent_gates : ℕ) 
  (valid_gates : gates.length = n)
  (distance : dist_between_adjacent_gates = 90) :
  n = 15 → 
  (initial_gate ∈ gates ∧ new_gate ∈ gates) → 
  ∃ (m1 m2 : ℕ), m1 = 59 ∧ m2 = 105 ∧ gcd m1 m2 = 1 ∧ 
  (∃ probability : ℚ, probability = (59 / 105 : ℚ) ∧ 
  (∃ sum_m1_m2 : ℕ, sum_m1_m2 = m1 + m2 ∧ sum_m1_m2 = 164)) :=
by
  sorry

end probability_walk_450_feet_or_less_l230_230359


namespace players_scores_l230_230753

/-- Lean code to verify the scores of three players in a guessing game -/
theorem players_scores (H F S : ℕ) (h1 : H = 42) (h2 : F - H = 24) (h3 : S - F = 18) (h4 : H < F) (h5 : H < S) : 
  F = 66 ∧ S = 84 :=
by
  sorry

end players_scores_l230_230753


namespace jill_trips_to_fill_tank_l230_230892

def tank_capacity : ℕ := 600
def bucket_volume : ℕ := 5
def jack_buckets_per_trip : ℕ := 2
def jill_buckets_per_trip : ℕ := 1
def jack_to_jill_trip_ratio : ℕ := 3 / 2

theorem jill_trips_to_fill_tank : (tank_capacity / bucket_volume) = 120 → 
                                   ((jack_to_jill_trip_ratio * jack_buckets_per_trip) + 2 * jill_buckets_per_trip) = 8 →
                                   15 * 2 = 30 :=
by
  intros h1 h2
  sorry

end jill_trips_to_fill_tank_l230_230892


namespace cherries_on_June_5_l230_230987

theorem cherries_on_June_5 : 
  ∃ c : ℕ, (c + (c + 8) + (c + 16) + (c + 24) + (c + 32) = 130) ∧ (c + 32 = 42) :=
by
  sorry

end cherries_on_June_5_l230_230987


namespace total_sales_in_december_correct_l230_230435

def ear_muffs_sales_in_december : ℝ :=
  let typeB_sold := 3258
  let typeB_price := 6.9
  let typeC_sold := 3186
  let typeC_price := 7.4
  let total_typeB_sales := typeB_sold * typeB_price
  let total_typeC_sales := typeC_sold * typeC_price
  total_typeB_sales + total_typeC_sales

theorem total_sales_in_december_correct :
  ear_muffs_sales_in_december = 46056.6 :=
by
  sorry

end total_sales_in_december_correct_l230_230435


namespace remainder_product_191_193_197_mod_23_l230_230603

theorem remainder_product_191_193_197_mod_23 :
  (191 * 193 * 197) % 23 = 14 := by
  sorry

end remainder_product_191_193_197_mod_23_l230_230603


namespace where_to_place_minus_sign_l230_230277

theorem where_to_place_minus_sign :
  (6 + 9 + 12 + 15 + 18 + 21 - 2 * 18) = 45 :=
by
  sorry

end where_to_place_minus_sign_l230_230277


namespace overlapping_squares_area_l230_230954

theorem overlapping_squares_area :
  let s : ℝ := 5
  let total_area := 3 * s^2
  let redundant_area := s^2 / 8 * 4
  total_area - redundant_area = 62.5 := by
  sorry

end overlapping_squares_area_l230_230954


namespace solve_eq1_solve_eq2_l230_230916

theorem solve_eq1 (x : ℝ):
  (x - 1) * (x + 3) = x - 1 ↔ x = 1 ∨ x = -2 :=
by 
  sorry

theorem solve_eq2 (x : ℝ):
  2 * x^2 - 6 * x = -3 ↔ x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2 :=
by 
  sorry

end solve_eq1_solve_eq2_l230_230916


namespace percentage_of_200_l230_230493

theorem percentage_of_200 : ((1/4) / 100) * 200 = 0.5 := 
by
  sorry

end percentage_of_200_l230_230493


namespace min_value_at_zero_max_value_a_l230_230904

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + 1) - (a * x / (x + 1))

-- Part (I)
theorem min_value_at_zero {a : ℝ} (h : ∀ x, f x a ≥ f 0 a) : a = 1 :=
sorry

-- Part (II)
theorem max_value_a (h : ∀ x > 0, f x a > 0) : a ≤ 1 :=
sorry

end min_value_at_zero_max_value_a_l230_230904


namespace div_by_13_l230_230495

theorem div_by_13 (n : ℕ) (h : 0 < n) : 13 ∣ (4^(2*n - 1) + 3^(n + 1)) :=
by 
  sorry

end div_by_13_l230_230495


namespace total_animal_legs_is_12_l230_230333

-- Define the number of legs per dog and chicken
def legs_per_dog : Nat := 4
def legs_per_chicken : Nat := 2

-- Define the number of dogs and chickens Mrs. Hilt saw
def number_of_dogs : Nat := 2
def number_of_chickens : Nat := 2

-- Calculate the total number of legs seen
def total_legs_seen : Nat :=
  (number_of_dogs * legs_per_dog) + (number_of_chickens * legs_per_chicken)

-- The theorem to be proven
theorem total_animal_legs_is_12 : total_legs_seen = 12 :=
by
  sorry

end total_animal_legs_is_12_l230_230333


namespace complex_expression_l230_230236

-- The condition: n is a positive integer
variable (n : ℕ) (hn : 0 < n)

-- Definition of the problem to be proved
theorem complex_expression (n : ℕ) (hn : 0 < n) : 
  (Complex.I ^ (4 * n) + Complex.I ^ (4 * n + 1) + Complex.I ^ (4 * n + 2) + Complex.I ^ (4 * n + 3)) = 0 :=
sorry

end complex_expression_l230_230236


namespace Vasya_can_win_l230_230056

theorem Vasya_can_win 
  (a : ℕ → ℕ) -- initial sequence of natural numbers
  (x : ℕ) -- number chosen by Vasya
: ∃ (i : ℕ), ∀ (k : ℕ), ∃ (j : ℕ), (a j + k * x = 1) :=
by
  sorry

end Vasya_can_win_l230_230056


namespace binomial_coeff_arith_seq_expansion_l230_230528

open BigOperators

-- Given the binomial expansion of (sqrt(x) + 2/sqrt(x))^n
-- we need to prove that the condition on binomial coefficients
-- implies that n = 7, and the expansion contains no constant term.
theorem binomial_coeff_arith_seq_expansion (x : ℝ) (n : ℕ) :
  (2 * Nat.choose n 2 = Nat.choose n 1 + Nat.choose n 3) ↔ n = 7 ∧ ∀ r : ℕ, x ^ (7 - 2 * r) / 2 ≠ x ^ 0 := by
  sorry

end binomial_coeff_arith_seq_expansion_l230_230528


namespace sin_identity_l230_230097

theorem sin_identity (α : ℝ) (h : Real.sin (π * α) = 4 / 5) : 
  Real.sin (π / 2 + 2 * α) = -24 / 25 :=
by
  sorry

end sin_identity_l230_230097


namespace megan_seashells_l230_230469

theorem megan_seashells (current_seashells desired_seashells diff_seashells : ℕ)
  (h1 : current_seashells = 307)
  (h2 : desired_seashells = 500)
  (h3 : diff_seashells = desired_seashells - current_seashells) :
  diff_seashells = 193 :=
by
  sorry

end megan_seashells_l230_230469


namespace ratio_of_larger_to_smaller_l230_230911

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : x > y) (h2 : x + y = 7 * (x - y)) (h3 : 0 < x) (h4 : 0 < y) : x / y = 4 / 3 := by
  sorry

end ratio_of_larger_to_smaller_l230_230911


namespace problem1_problem2_l230_230431

variable (α : ℝ)

-- Equivalent problem 1
theorem problem1 (h : Real.tan α = 7) : (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 13 := 
  sorry

-- Equivalent problem 2
theorem problem2 (h : Real.tan α = 7) : Real.sin α * Real.cos α = 7 / 50 := 
  sorry

end problem1_problem2_l230_230431


namespace water_remaining_l230_230620

theorem water_remaining (initial_water : ℕ) (evap_rate : ℕ) (days : ℕ) : 
  initial_water = 500 → evap_rate = 1 → days = 50 → 
  initial_water - evap_rate * days = 450 :=
by
  intros h₁ h₂ h₃
  sorry

end water_remaining_l230_230620


namespace line_through_point_with_opposite_intercepts_l230_230139

theorem line_through_point_with_opposite_intercepts :
  (∃ m : ℝ, (∀ x y : ℝ, y = m * x → (2,3) = (x, y)) ∧ ((∀ a : ℝ, a ≠ 0 → (x / a + y / (-a) = 1) → (2 - 3 = a ∧ a = -1)))) →
  ((∀ x y : ℝ, 3 * x - 2 * y = 0) ∨ (∀ x y : ℝ, x - y + 1 = 0)) :=
by
  sorry

end line_through_point_with_opposite_intercepts_l230_230139


namespace proof_problem_l230_230568

theorem proof_problem 
  {a b c : ℝ} (h_cond : 1/a + 1/b + 1/c = 1/(a + b + c))
  (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (n : ℕ) :
  1/a^(2*n+1) + 1/b^(2*n+1) + 1/c^(2*n+1) = 1/(a^(2*n+1) + b^(2*n+1) + c^(2*n+1)) :=
sorry

end proof_problem_l230_230568


namespace cut_difference_l230_230980

-- define the conditions
def skirt_cut : ℝ := 0.75
def pants_cut : ℝ := 0.5

-- theorem to prove the correctness of the difference
theorem cut_difference : (skirt_cut - pants_cut = 0.25) :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end cut_difference_l230_230980


namespace line_intersects_circle_midpoint_trajectory_l230_230344

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

end line_intersects_circle_midpoint_trajectory_l230_230344


namespace fraction_exponentiation_l230_230267

theorem fraction_exponentiation :
  (1 / 3) ^ 5 = 1 / 243 :=
sorry

end fraction_exponentiation_l230_230267


namespace find_AX_length_l230_230275

theorem find_AX_length (t BC AC BX : ℝ) (AX AB : ℝ)
  (h1 : t = 0.75)
  (h2 : AX = t * AB)
  (h3 : BC = 40)
  (h4 : AC = 35)
  (h5 : BX = 15) :
  AX = 105 / 8 := 
  sorry

end find_AX_length_l230_230275


namespace install_time_for_windows_l230_230054

theorem install_time_for_windows
  (total_windows installed_windows hours_per_window : ℕ)
  (h1 : total_windows = 200)
  (h2 : installed_windows = 65)
  (h3 : hours_per_window = 12) :
  (total_windows - installed_windows) * hours_per_window = 1620 :=
by
  sorry

end install_time_for_windows_l230_230054


namespace prob_first_diamond_second_ace_or_face_l230_230854

theorem prob_first_diamond_second_ace_or_face :
  let deck_size := 52
  let first_card_diamonds := 13 / deck_size
  let prob_ace_after_diamond := 4 / (deck_size - 1)
  let prob_face_after_diamond := 12 / (deck_size - 1)
  first_card_diamonds * (prob_ace_after_diamond + prob_face_after_diamond) = 68 / 867 :=
by
  let deck_size := 52
  let first_card_diamonds := 13 / deck_size
  let prob_ace_after_diamond := 4 / (deck_size - 1)
  let prob_face_after_diamond := 12 / (deck_size - 1)
  sorry

end prob_first_diamond_second_ace_or_face_l230_230854


namespace winner_percentage_of_votes_l230_230409

theorem winner_percentage_of_votes (V W O : ℕ) (W_votes : W = 720) (won_by : W - O = 240) (total_votes : V = W + O) :
  (W * 100) / V = 60 :=
by
  sorry

end winner_percentage_of_votes_l230_230409


namespace circle_symmetric_about_line_l230_230276

-- The main proof statement
theorem circle_symmetric_about_line (x y : ℝ) (k : ℝ) :
  (x - 1)^2 + (y - 1)^2 = 2 ∧ y = k * x + 3 → k = -2 :=
by
  sorry

end circle_symmetric_about_line_l230_230276


namespace annie_purchases_l230_230699

theorem annie_purchases (x y z : ℕ) 
  (h1 : x + y + z = 50) 
  (h2 : 20 * x + 400 * y + 500 * z = 5000) :
  x = 40 :=
by sorry

end annie_purchases_l230_230699


namespace arithmetic_square_root_of_9_is_3_l230_230748

-- Define the arithmetic square root property
def is_arithmetic_square_root (x : ℝ) (n : ℝ) : Prop :=
  x * x = n ∧ x ≥ 0

-- The main theorem: The arithmetic square root of 9 is 3
theorem arithmetic_square_root_of_9_is_3 : 
  is_arithmetic_square_root 3 9 :=
by
  -- This is where the proof would go, but since only the statement is required:
  sorry

end arithmetic_square_root_of_9_is_3_l230_230748


namespace sum_of_nonnegative_reals_l230_230204

theorem sum_of_nonnegative_reals (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 27) :
  x + y + z = Real.sqrt 106 :=
sorry

end sum_of_nonnegative_reals_l230_230204


namespace certain_number_eq_14_l230_230039

theorem certain_number_eq_14 (x y : ℤ) (h1 : 4 * x + y = 34) (h2 : y^2 = 4) : 2 * x - y = 14 :=
by
  sorry

end certain_number_eq_14_l230_230039


namespace simplify_polynomial_l230_230345

variable (r : ℝ)

theorem simplify_polynomial : (2 * r^2 + 5 * r - 7) - (r^2 + 9 * r - 3) = r^2 - 4 * r - 4 := by
  sorry

end simplify_polynomial_l230_230345


namespace problem_a_lt_zero_b_lt_neg_one_l230_230066

theorem problem_a_lt_zero_b_lt_neg_one (a b : ℝ) (ha : a < 0) (hb : b < -1) : 
  ab > a ∧ a > ab^2 := 
by
  sorry

end problem_a_lt_zero_b_lt_neg_one_l230_230066


namespace fruit_vendor_sold_fruits_l230_230407

def total_dozen_fruits_sold (lemons_dozen avocados_dozen : ℝ) (dozen : ℝ) : ℝ :=
  (lemons_dozen * dozen) + (avocados_dozen * dozen)

theorem fruit_vendor_sold_fruits (hl : ∀ (lemons_dozen avocados_dozen : ℝ) (dozen : ℝ), lemons_dozen = 2.5 ∧ avocados_dozen = 5 ∧ dozen = 12) :
  total_dozen_fruits_sold 2.5 5 12 = 90 :=
by
  sorry

end fruit_vendor_sold_fruits_l230_230407


namespace factor_polynomial_l230_230830

theorem factor_polynomial (x y : ℝ) : 
  2*x^2 - x*y - 15*y^2 = (2*x - 5*y) * (x - 3*y) :=
sorry

end factor_polynomial_l230_230830


namespace alpha_range_l230_230241

theorem alpha_range (α : ℝ) (hα : 0 < α ∧ α < 2 * Real.pi) : 
  (Real.sin α < Real.sqrt 3 / 2 ∧ Real.cos α > 1 / 2) ↔ 
  (0 < α ∧ α < Real.pi / 3 ∨ 5 * Real.pi / 3 < α ∧ α < 2 * Real.pi) := 
sorry

end alpha_range_l230_230241


namespace puppies_per_dog_l230_230334

def dogs := 15
def puppies := 75

theorem puppies_per_dog : puppies / dogs = 5 :=
by {
  sorry
}

end puppies_per_dog_l230_230334


namespace eccentricity_of_ellipse_l230_230875

theorem eccentricity_of_ellipse (a c : ℝ) (h : 4 * a = 7 * 2 * (a - c)) : 
    c / a = 5 / 7 :=
by {
  sorry
}

end eccentricity_of_ellipse_l230_230875


namespace cost_of_one_each_l230_230496

theorem cost_of_one_each (x y z : ℝ) (h1 : 3 * x + 7 * y + z = 24) (h2 : 4 * x + 10 * y + z = 33) :
  x + y + z = 6 :=
sorry

end cost_of_one_each_l230_230496


namespace total_students_stratified_sampling_l230_230191

namespace HighSchool

theorem total_students_stratified_sampling 
  (sample_size : ℕ)
  (sample_grade10 : ℕ)
  (sample_grade11 : ℕ)
  (students_grade12 : ℕ) 
  (n : ℕ)
  (H1 : sample_size = 100)
  (H2 : sample_grade10 = 24)
  (H3 : sample_grade11 = 26)
  (H4 : students_grade12 = 600)
  (H5 : ∀ n, (students_grade12 / n * sample_size = sample_size - sample_grade10 - sample_grade11) → n = 1200) :
  n = 1200 :=
sorry

end HighSchool

end total_students_stratified_sampling_l230_230191


namespace remainder_3_pow_1503_mod_7_l230_230413

theorem remainder_3_pow_1503_mod_7 : 
  (3 ^ 1503) % 7 = 6 := 
by sorry

end remainder_3_pow_1503_mod_7_l230_230413


namespace polynomial_real_root_inequality_l230_230057

theorem polynomial_real_root_inequality (a b : ℝ) : 
  (∃ x : ℝ, x^4 - a * x^3 + 2 * x^2 - b * x + 1 = 0) → (a^2 + b^2 ≥ 8) :=
sorry

end polynomial_real_root_inequality_l230_230057


namespace triangle_acute_l230_230734

theorem triangle_acute
  (A B C : ℝ)
  (h_sum : A + B + C = 180)
  (h_ratio : A / B = 2 / 3 ∧ B / C = 3 / 4) :
  A < 90 ∧ B < 90 ∧ C < 90 :=
by
  -- proof goes here
  sorry

end triangle_acute_l230_230734


namespace monotonicity_m_eq_zero_range_of_m_l230_230210

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.exp x - m * x^2 - 2 * x

theorem monotonicity_m_eq_zero :
  ∀ x : ℝ, (x < Real.log 2 → f x 0 < f (x + 1) 0) ∧ (x > Real.log 2 → f x 0 > f (x - 1) 0) := 
sorry

theorem range_of_m :
  ∀ x : ℝ, x ∈ Set.Ici 0 → f x m > (Real.exp 1 / 2 - 1) → m < (Real.exp 1 / 2 - 1) := 
sorry

end monotonicity_m_eq_zero_range_of_m_l230_230210


namespace can_measure_all_weights_l230_230879

theorem can_measure_all_weights (a b c : ℕ) 
  (h_sum : a + b + c = 10) 
  (h_unique : (a = 1 ∧ b = 2 ∧ c = 7) ∨ (a = 1 ∧ b = 3 ∧ c = 6)) : 
  ∀ w : ℕ, 1 ≤ w ∧ w ≤ 10 → 
    ∃ (k l m : ℤ), w = k * a + l * b + m * c ∨ w = k * -a + l * -b + m * -c :=
  sorry

end can_measure_all_weights_l230_230879


namespace money_left_is_correct_l230_230755

-- Define initial amount of money Dan has
def initial_amount : ℕ := 3

-- Define the cost of the candy bar
def candy_cost : ℕ := 1

-- Define the money left after the purchase
def money_left : ℕ := initial_amount - candy_cost

-- The theorem stating that the money left is 2
theorem money_left_is_correct : money_left = 2 := by
  sorry

end money_left_is_correct_l230_230755


namespace min_vans_proof_l230_230230

-- Define the capacity and availability of each type of van
def capacity_A : Nat := 7
def capacity_B : Nat := 9
def capacity_C : Nat := 12

def available_A : Nat := 3
def available_B : Nat := 4
def available_C : Nat := 2

-- Define the number of people going on the trip
def students : Nat := 40
def adults : Nat := 14

-- Define the total number of people
def total_people : Nat := students + adults

-- Define the minimum number of vans needed
def min_vans_needed : Nat := 6

-- Define the number of each type of van used
def vans_A_used : Nat := 0
def vans_B_used : Nat := 4
def vans_C_used : Nat := 2

-- Prove the minimum number of vans needed to accommodate everyone is 6
theorem min_vans_proof : min_vans_needed = 6 ∧ 
  (vans_A_used * capacity_A + vans_B_used * capacity_B + vans_C_used * capacity_C = total_people) ∧
  vans_A_used <= available_A ∧ vans_B_used <= available_B ∧ vans_C_used <= available_C :=
by 
  sorry

end min_vans_proof_l230_230230


namespace gcd_pow_minus_one_l230_230092

theorem gcd_pow_minus_one {m n : ℕ} (hm : 0 < m) (hn : 0 < n) :
  Nat.gcd (2^m - 1) (2^n - 1) = 2^Nat.gcd m n - 1 :=
sorry

end gcd_pow_minus_one_l230_230092


namespace find_integers_10_le_n_le_20_mod_7_l230_230502

theorem find_integers_10_le_n_le_20_mod_7 :
  ∃ n, (10 ≤ n ∧ n ≤ 20 ∧ n % 7 = 4) ∧
  (n = 11 ∨ n = 18) := by
  sorry

end find_integers_10_le_n_le_20_mod_7_l230_230502


namespace arithmetic_sequence_sixth_term_l230_230146

variables (a d : ℤ)

theorem arithmetic_sequence_sixth_term :
  a + (a + d) + (a + 2 * d) = 12 →
  a + 3 * d = 0 →
  a + 5 * d = -4 :=
by
  intros h1 h2
  sorry

end arithmetic_sequence_sixth_term_l230_230146


namespace quadratic_equal_roots_l230_230803

theorem quadratic_equal_roots (a : ℝ) :
  (∃ x : ℝ, x ≠ 0 ∧ (x * (x + 1) + a * x = 0) ∧ ((1 + a)^2 = 0)) →
  a = -1 :=
by
  sorry

end quadratic_equal_roots_l230_230803


namespace total_students_l230_230332

-- Definition of the problem conditions
def buses : ℕ := 18
def seats_per_bus : ℕ := 15
def empty_seats_per_bus : ℕ := 3

-- Formulating the mathematically equivalent proof problem
theorem total_students :
  (buses * (seats_per_bus - empty_seats_per_bus) = 216) :=
by
  sorry

end total_students_l230_230332


namespace hyperbola_focus_l230_230252

theorem hyperbola_focus :
    ∃ (f : ℝ × ℝ), f = (-2 - Real.sqrt 6, -2) ∧
    ∀ (x y : ℝ), 2 * x^2 - y^2 + 8 * x - 4 * y - 8 = 0 → 
    ∃ a b h k : ℝ, 
        (a = Real.sqrt 2) ∧ (b = 2) ∧ (h = -2) ∧ (k = -2) ∧
        ((2 * (x + h)^2 - (y + k)^2 = 4) ∧ 
         (x, y) = f) :=
sorry

end hyperbola_focus_l230_230252


namespace parity_of_solutions_l230_230759

theorem parity_of_solutions
  (n m x y : ℤ)
  (hn : Odd n) 
  (hm : Odd m) 
  (h1 : x + 2 * y = n) 
  (h2 : 3 * x - y = m) :
  Odd x ∧ Even y :=
by
  sorry

end parity_of_solutions_l230_230759


namespace expression_increase_fraction_l230_230721

theorem expression_increase_fraction (x y : ℝ) :
  let x' := 1.4 * x
  let y' := 1.4 * y
  let original := x * y^2
  let increased := x' * y'^2
  increased - original = (1744/1000) * original := by
sorry

end expression_increase_fraction_l230_230721


namespace inequality_proof_l230_230438

theorem inequality_proof (s r : ℝ) (h1 : s > 0) (h2 : r > 0) (h3 : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by
  sorry

end inequality_proof_l230_230438


namespace complement_B_def_union_A_B_def_intersection_A_B_def_intersection_A_complement_B_def_intersection_complements_def_l230_230482

-- Definitions of the sets A and B
def set_A : Set ℝ := {y : ℝ | -1 < y ∧ y < 4}
def set_B : Set ℝ := {y : ℝ | 0 < y ∧ y < 5}

-- Complement of B in the universal set U (ℝ)
def complement_B : Set ℝ := {y : ℝ | y ≤ 0 ∨ y ≥ 5}

theorem complement_B_def : (complement_B = {y : ℝ | y ≤ 0 ∨ y ≥ 5}) :=
by sorry

-- Union of A and B
def union_A_B : Set ℝ := {y : ℝ | -1 < y ∧ y < 5}

theorem union_A_B_def : (set_A ∪ set_B = union_A_B) :=
by sorry

-- Intersection of A and B
def intersection_A_B : Set ℝ := {y : ℝ | 0 < y ∧ y < 4}

theorem intersection_A_B_def : (set_A ∩ set_B = intersection_A_B) :=
by sorry

-- Intersection of A and the complement of B
def intersection_A_complement_B : Set ℝ := {y : ℝ | -1 < y ∧ y ≤ 0}

theorem intersection_A_complement_B_def : (set_A ∩ complement_B = intersection_A_complement_B) :=
by sorry

-- Intersection of the complements of A and B
def complement_A : Set ℝ := {y : ℝ | y ≤ -1 ∨ y ≥ 4} -- Derived from complement of A
def intersection_complements : Set ℝ := {y : ℝ | y ≤ -1 ∨ y ≥ 5}

theorem intersection_complements_def : (complement_A ∩ complement_B = intersection_complements) :=
by sorry

end complement_B_def_union_A_B_def_intersection_A_B_def_intersection_A_complement_B_def_intersection_complements_def_l230_230482


namespace real_solutions_l230_230067

theorem real_solutions :
  ∀ x : ℝ, 
  (1 / ((x - 1) * (x - 2)) + 
   1 / ((x - 2) * (x - 3)) + 
   1 / ((x - 3) * (x - 4)) + 
   1 / ((x - 4) * (x - 5)) = 1 / 10) 
  ↔ (x = 10 ∨ x = -3.5) :=
by
  sorry

end real_solutions_l230_230067


namespace intersection_of_M_and_N_l230_230116

noncomputable def M : Set ℕ := { x | 1 < x ∧ x < 7 }
noncomputable def N : Set ℕ := { x | x % 3 ≠ 0 }

theorem intersection_of_M_and_N :
  M ∩ N = {2, 4, 5} := sorry

end intersection_of_M_and_N_l230_230116


namespace A_plays_D_third_day_l230_230812

section GoTournament

variables (Player : Type) (A B C D : Player) 

-- Define the condition that each player competes with every other player exactly once.
def each_plays_once (P : Player → Player → Prop) : Prop :=
  ∀ x y, x ≠ y → (P x y ∨ P y x)

-- Define the tournament setup and the play conditions.
variables (P : Player → Player → Prop)
variable [∀ x y, Decidable (P x y)] -- Assuming decidability for the play relation

-- The given conditions of the problem
axiom A_plays_C_first_day : P A C
axiom C_plays_D_second_day : P C D
axiom only_one_match_per_day : ∀ x, ∃! y, P x y

-- We aim to prove that A will play against D on the third day.
theorem A_plays_D_third_day : P A D :=
sorry

end GoTournament

end A_plays_D_third_day_l230_230812


namespace problem1_problem2_l230_230350

open Real

noncomputable def f (a x : ℝ) : ℝ :=
  x^2 + 2 * (a - 2) * x + 4

theorem problem1 (a : ℝ) :
  (∀ x, f a x > 0) → 0 < a ∧ a < 4 :=
sorry

theorem problem2 (a : ℝ) :
  (∀ x, -3 <= x ∧ x <= 1 → f a x > 0) → (-1/2 < a ∧ a < 4) :=
sorry

end problem1_problem2_l230_230350


namespace max_min_x_plus_y_l230_230020

theorem max_min_x_plus_y (x y : ℝ) (h : |x + 2| + |1 - x| = 9 - |y - 5| - |1 + y|) :
  -3 ≤ x + y ∧ x + y ≤ 6 := 
sorry

end max_min_x_plus_y_l230_230020


namespace min_distance_point_to_line_l230_230848

theorem min_distance_point_to_line :
    ∀ (x y : ℝ), (x^2 + y^2 - 6 * x - 4 * y + 12 = 0) -> 
    (3 * x + 4 * y - 2 = 0) -> 
    ∃ d: ℝ, d = 2 :=
by sorry

end min_distance_point_to_line_l230_230848


namespace part1_part2_l230_230516
-- Import the entire Mathlib library for broader usage

-- Definition of the given vectors
def a : ℝ × ℝ := (4, 7)
def b (x : ℝ) : ℝ × ℝ := (x, x + 6)

-- Part 1: Prove the dot product when x = -1 is 31
theorem part1 : (a.1 * (-1) + a.2 * (5)) = 31 := by
  sorry

-- Part 2: Prove the value of x when the vectors are parallel
theorem part2 : (4 : ℝ) / x = (7 : ℝ) / (x + 6) → x = 8 := by
  sorry

end part1_part2_l230_230516


namespace universal_friendship_l230_230944

-- Define the inhabitants and their relationships
def inhabitants (n : ℕ) : Type := Fin n

-- Condition for friends and enemies
inductive Relationship (n : ℕ) : inhabitants n → inhabitants n → Prop
| friend (A B : inhabitants n) : Relationship n A B
| enemy (A B : inhabitants n) : Relationship n A B

-- Transitivity condition
axiom transitivity {n : ℕ} {A B C : inhabitants n} :
  Relationship n A B = Relationship n B C → Relationship n A C = Relationship n A B

-- At least two friends among any three inhabitants
axiom at_least_two_friends {n : ℕ} (A B C : inhabitants n) :
  ∃ X Y : inhabitants n, X ≠ Y ∧ Relationship n X Y = Relationship n X Y

-- Inhabitants can start a new life switching relationships
axiom start_new_life {n : ℕ} (A : inhabitants n) :
  ∀ B : inhabitants n, Relationship n A B = Relationship n B A

-- The main theorem we need to prove
theorem universal_friendship (n : ℕ) : 
  ∀ A B : inhabitants n, ∃ C : inhabitants n, Relationship n A C = Relationship n B C :=
sorry

end universal_friendship_l230_230944


namespace tori_current_height_l230_230794

theorem tori_current_height :
  let original_height := 4.4
  let growth := 2.86
  original_height + growth = 7.26 := 
by
  sorry

end tori_current_height_l230_230794


namespace evaluate_g_at_3_l230_230643

def g (x : ℝ) : ℝ := 7 * x^3 - 8 * x^2 - 5 * x + 7

theorem evaluate_g_at_3 : g 3 = 109 := by
  sorry

end evaluate_g_at_3_l230_230643


namespace container_alcohol_amount_l230_230545

theorem container_alcohol_amount
  (A : ℚ) -- Amount of alcohol in quarts
  (initial_water : ℚ) -- Initial amount of water in quarts
  (added_water : ℚ) -- Amount of water added in quarts
  (final_ratio_alcohol_to_water : ℚ) -- Final ratio of alcohol to water
  (h_initial_water : initial_water = 4) -- Container initially contains 4 quarts of water.
  (h_added_water : added_water = 8/3) -- 2.666666666666667 quarts of water added.
  (h_final_ratio : final_ratio_alcohol_to_water = 3/5) -- Final ratio is 3 parts alcohol to 5 parts water.
  (h_final_water : initial_water + added_water = 20/3) -- Total final water quarts after addition.
  : A = 4 := 
sorry

end container_alcohol_amount_l230_230545


namespace range_of_k_l230_230852

-- Define the set M
def M := {x : ℝ | -1 ≤ x ∧ x ≤ 7}

-- Define the set N based on k
def N (k : ℝ) := {x : ℝ | k + 1 ≤ x ∧ x ≤ 2 * k - 1}

-- The main statement to prove
theorem range_of_k (k : ℝ) : M ∩ N k = ∅ → 6 < k :=
by
  -- skipping the proof as instructed
  sorry

end range_of_k_l230_230852


namespace orchestra_club_members_l230_230429

theorem orchestra_club_members : ∃ (n : ℕ), 150 < n ∧ n < 250 ∧ n % 8 = 1 ∧ n % 6 = 2 ∧ n % 9 = 3 ∧ n = 169 := 
by {
  sorry
}

end orchestra_club_members_l230_230429


namespace maria_remaining_towels_l230_230899

def total_towels_initial := 40 + 44
def towels_given_away := 65

theorem maria_remaining_towels : (total_towels_initial - towels_given_away) = 19 := by
  sorry

end maria_remaining_towels_l230_230899


namespace max_value_of_f_l230_230489

noncomputable def f (x : ℝ) := x^3 - 3 * x + 1

theorem max_value_of_f (h: ∃ x, f x = -1) : ∃ y, f y = 3 :=
by
  -- We'll later prove this with appropriate mathematical steps using Lean tactics
  sorry

end max_value_of_f_l230_230489


namespace gear_q_revolutions_per_minute_is_40_l230_230434

-- Definitions corresponding to conditions
def gear_p_revolutions_per_minute : ℕ := 10
def gear_q_revolutions_per_minute (r : ℕ) : Prop :=
  ∃ (r : ℕ), (r * 20 / 60) - (10 * 20 / 60) = 10

-- Statement we need to prove
theorem gear_q_revolutions_per_minute_is_40 :
  gear_q_revolutions_per_minute 40 :=
sorry

end gear_q_revolutions_per_minute_is_40_l230_230434


namespace carly_dogs_total_l230_230381

theorem carly_dogs_total (total_nails : ℕ) (three_legged_dogs : ℕ) (nails_per_paw : ℕ) (total_dogs : ℕ) 
  (h1 : total_nails = 164) (h2 : three_legged_dogs = 3) (h3 : nails_per_paw = 4) : total_dogs = 11 :=
by
  sorry

end carly_dogs_total_l230_230381


namespace parabola_focus_l230_230474

-- Define the parabola
def parabolaEquation (x y : ℝ) : Prop := y^2 = -6 * x

-- Define the focus
def focus (x y : ℝ) : Prop := x = -3 / 2 ∧ y = 0

-- The proof problem: showing the focus of the given parabola
theorem parabola_focus : ∃ x y : ℝ, parabolaEquation x y ∧ focus x y :=
by
    sorry

end parabola_focus_l230_230474


namespace rest_area_location_l230_230631

theorem rest_area_location :
  ∀ (A B : ℝ), A = 50 → B = 230 → (5 / 8 * (B - A) + A = 162.5) :=
by
  intros A B hA hB
  rw [hA, hB]
  -- doing the computation to show the rest area is at 162.5 km
  sorry

end rest_area_location_l230_230631


namespace marbles_leftover_l230_230996

theorem marbles_leftover (r p g : ℕ) (hr : r % 7 = 5) (hp : p % 7 = 4) (hg : g % 7 = 2) : 
  (r + p + g) % 7 = 4 :=
by
  sorry

end marbles_leftover_l230_230996


namespace least_product_of_primes_gt_30_l230_230508

theorem least_product_of_primes_gt_30 :
  ∃ (p q : ℕ), p > 30 ∧ q > 30 ∧ Prime p ∧ Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end least_product_of_primes_gt_30_l230_230508


namespace sum_of_money_l230_230927

-- Conditions
def mass_record_coin_kg : ℝ := 100  -- 100 kg
def mass_one_pound_coin_g : ℝ := 10  -- 10 g

-- Conversion factor
def kg_to_g : ℝ := 1000

-- Question: Prove the sum of money in £1 coins that weighs the same as the record-breaking coin is £10,000.
theorem sum_of_money 
  (mass_record_coin_g := mass_record_coin_kg * kg_to_g)
  (number_of_coins := mass_record_coin_g / mass_one_pound_coin_g) 
  (sum_of_money := number_of_coins) : 
  sum_of_money = 10000 :=
  sorry

end sum_of_money_l230_230927


namespace max_value_of_expression_l230_230941

noncomputable def max_expression_value (a b : ℝ) := a * b * (100 - 5 * a - 2 * b)

theorem max_value_of_expression :
  ∀ (a b : ℝ), 0 < a → 0 < b → 5 * a + 2 * b < 100 →
  max_expression_value a b ≤ 78125 / 36 := by
  intros a b ha hb h
  sorry

end max_value_of_expression_l230_230941


namespace boris_climbs_needed_l230_230347

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

end boris_climbs_needed_l230_230347


namespace emily_required_sixth_score_is_99_l230_230861

/-- Emily's quiz scores and the required mean score -/
def emily_scores : List ℝ := [85, 90, 88, 92, 98]
def required_mean_score : ℝ := 92

/-- The function to calculate the required sixth quiz score for Emily -/
def required_sixth_score (scores : List ℝ) (mean : ℝ) : ℝ :=
  let sum_current := scores.sum
  let total_required := mean * (scores.length + 1)
  total_required - sum_current

/-- Emily needs to score 99 on her sixth quiz for an average of 92 -/
theorem emily_required_sixth_score_is_99 : 
  required_sixth_score emily_scores required_mean_score = 99 :=
by
  sorry

end emily_required_sixth_score_is_99_l230_230861


namespace jane_reading_period_l230_230716

theorem jane_reading_period (total_pages pages_per_day : ℕ) (H1 : pages_per_day = 5 + 10) (H2 : total_pages = 105) : 
  total_pages / pages_per_day = 7 :=
by
  sorry

end jane_reading_period_l230_230716


namespace quadratic_solution_l230_230773

theorem quadratic_solution (x : ℝ) (h : x^2 - 4 * x + 2 = 0) : x + 2 / x = 4 :=
by sorry

end quadratic_solution_l230_230773


namespace larger_number_is_seventy_two_l230_230806

def five_times_larger_is_six_times_smaller (x y : ℕ) : Prop := 5 * y = 6 * x
def difference_is_twelve (x y : ℕ) : Prop := y - x = 12

theorem larger_number_is_seventy_two (x y : ℕ) 
  (h1 : five_times_larger_is_six_times_smaller x y)
  (h2 : difference_is_twelve x y) : y = 72 :=
sorry

end larger_number_is_seventy_two_l230_230806


namespace find_intervals_of_monotonicity_find_value_of_a_l230_230459

noncomputable def f (x a : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6) + a + 1

theorem find_intervals_of_monotonicity (k : ℤ) (a : ℝ) :
  ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3), MonotoneOn (λ x => f x a) (Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3)) :=
sorry

theorem find_value_of_a (a : ℝ) (max_value_condition : ∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ 4) :
  a = 1 :=
sorry

end find_intervals_of_monotonicity_find_value_of_a_l230_230459


namespace johnnyMoneyLeft_l230_230238

noncomputable def johnnySavingsSeptember : ℝ := 30
noncomputable def johnnySavingsOctober : ℝ := 49
noncomputable def johnnySavingsNovember : ℝ := 46
noncomputable def johnnySavingsDecember : ℝ := 55

noncomputable def johnnySavingsJanuary : ℝ := johnnySavingsDecember * 1.15

noncomputable def totalSavings : ℝ := johnnySavingsSeptember + johnnySavingsOctober + johnnySavingsNovember + johnnySavingsDecember + johnnySavingsJanuary

noncomputable def videoGameCost : ℝ := 58
noncomputable def bookCost : ℝ := 25
noncomputable def birthdayPresentCost : ℝ := 40

noncomputable def totalSpent : ℝ := videoGameCost + bookCost + birthdayPresentCost

noncomputable def moneyLeft : ℝ := totalSavings - totalSpent

theorem johnnyMoneyLeft : moneyLeft = 120.25 := by
  sorry

end johnnyMoneyLeft_l230_230238


namespace isosceles_triangle_perimeter_l230_230887

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : a = 6) (h₂ : b = 5) :
  ∃ p : ℝ, (p = a + a + b ∨ p = b + b + a) ∧ (p = 16 ∨ p = 17) :=
by
  sorry

end isosceles_triangle_perimeter_l230_230887


namespace sovereign_states_upper_bound_l230_230642

theorem sovereign_states_upper_bound (n : ℕ) (k : ℕ) : 
  (∃ (lines : ℕ) (border_stop_moving : Prop) (countries_disappear : Prop)
     (create_un : Prop) (total_countries : ℕ),
        (lines = n)
        ∧ (border_stop_moving = true)
        ∧ (countries_disappear = true)
        ∧ (create_un = true)
        ∧ (total_countries = k)) 
  → k ≤ (n^3 + 5*n) / 6 + 1 := 
sorry

end sovereign_states_upper_bound_l230_230642


namespace min_value_x_y_l230_230732

open Real

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + 1/y) * (x + 1/y - 1024) + (y + 1/x) * (y + 1/x - 1024) ≥ -524288 :=
by 
  sorry

end min_value_x_y_l230_230732


namespace find_y_l230_230727

theorem find_y (y : ℝ) (hy_pos : y > 0) (hy_prop : y^2 / 100 = 9) : y = 30 := by
  sorry

end find_y_l230_230727


namespace find_length_of_c_find_measure_of_B_l230_230961

-- Definition of the conditions
def triangle (A B C a b c : ℝ) : Prop :=
  c - b = 2 * b * Real.cos A

noncomputable def value_c (a b : ℝ) : ℝ := sorry

noncomputable def value_B (A B : ℝ) : ℝ := sorry

-- Statement for problem (I)
theorem find_length_of_c (a b : ℝ) (h1 : a = 2 * Real.sqrt 6) (h2 : b = 3) (h3 : ∀ A B C, triangle A B C a b (value_c a b)) : 
  value_c a b = 5 :=
by 
  sorry

-- Statement for problem (II)
theorem find_measure_of_B (B : ℝ) (h1 : ∀ A, A + B = Real.pi / 2) (h2 : B = value_B A B) : 
  value_B A B = Real.pi / 6 :=
by 
  sorry

end find_length_of_c_find_measure_of_B_l230_230961


namespace intersection_A_B_union_A_B_range_of_a_l230_230378

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

end intersection_A_B_union_A_B_range_of_a_l230_230378


namespace factor_expression_l230_230593

theorem factor_expression (b : ℤ) : 53 * b^2 + 159 * b = 53 * b * (b + 3) :=
by
  sorry

end factor_expression_l230_230593


namespace complex_equality_l230_230355

theorem complex_equality (a b : ℝ) (h : (⟨0, 1⟩ : ℂ) ^ 3 = ⟨a, -b⟩) : a + b = 1 :=
by
  sorry

end complex_equality_l230_230355


namespace michael_class_choosing_l230_230319

open Nat

theorem michael_class_choosing :
  (choose 6 3) * (choose 4 2) + (choose 6 4) * (choose 4 1) + (choose 6 5) = 186 := 
by
  sorry

end michael_class_choosing_l230_230319


namespace original_planned_production_l230_230602

theorem original_planned_production (x : ℝ) (hx1 : x ≠ 0) (hx2 : 210 / x - 210 / (1.5 * x) = 5) : x = 14 :=
by sorry

end original_planned_production_l230_230602


namespace black_squares_in_20th_row_l230_230739

noncomputable def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

noncomputable def squares_in_row (n : ℕ) : ℕ := 1 + sum_natural (n - 2)

noncomputable def black_squares_in_row (n : ℕ) : ℕ := 
  if squares_in_row n % 2 = 1 then (squares_in_row n - 1) / 2 else squares_in_row n / 2

theorem black_squares_in_20th_row : black_squares_in_row 20 = 85 := 
by
  sorry

end black_squares_in_20th_row_l230_230739


namespace ceil_sum_sqrt_eval_l230_230130

theorem ceil_sum_sqrt_eval : 
  (⌈Real.sqrt 2⌉ + ⌈Real.sqrt 22⌉ + ⌈Real.sqrt 222⌉) = 22 := 
by
  sorry

end ceil_sum_sqrt_eval_l230_230130


namespace rhombus_diagonal_l230_230877

/-- Given a rhombus with one diagonal being 11 cm and the area of the rhombus being 88 cm²,
prove that the length of the other diagonal is 16 cm. -/
theorem rhombus_diagonal 
  (d1 : ℝ) (d2 : ℝ) (area : ℝ)
  (h_d1 : d1 = 11)
  (h_area : area = 88)
  (h_area_eq : area = (d1 * d2) / 2) : d2 = 16 :=
sorry

end rhombus_diagonal_l230_230877


namespace ab_value_l230_230584

theorem ab_value (a b : ℝ) (h1 : b^2 - a^2 = 4) (h2 : a^2 + b^2 = 25) : abs (a * b) = Real.sqrt (609 / 4) := 
sorry

end ab_value_l230_230584


namespace higher_amount_is_sixty_l230_230272

theorem higher_amount_is_sixty (R : ℕ) (n : ℕ) (H : ℝ) 
  (h1 : 2000 = 40 * n + H * R)
  (h2 : 1800 = 40 * (n + 10) + H * (R - 10)) :
  H = 60 :=
by
  sorry

end higher_amount_is_sixty_l230_230272


namespace students_not_picked_l230_230085

def total_students : ℕ := 58
def number_of_groups : ℕ := 8
def students_per_group : ℕ := 6

theorem students_not_picked :
  total_students - (number_of_groups * students_per_group) = 10 := by 
  sorry

end students_not_picked_l230_230085


namespace carrie_payment_l230_230590

def num_shirts := 8
def cost_per_shirt := 12
def total_shirt_cost := num_shirts * cost_per_shirt

def num_pants := 4
def cost_per_pant := 25
def total_pant_cost := num_pants * cost_per_pant

def num_jackets := 4
def cost_per_jacket := 75
def total_jacket_cost := num_jackets * cost_per_jacket

def num_skirts := 3
def cost_per_skirt := 30
def total_skirt_cost := num_skirts * cost_per_skirt

def num_shoes := 2
def cost_per_shoe := 50
def total_shoe_cost := num_shoes * cost_per_shoe

def total_cost := total_shirt_cost + total_pant_cost + total_jacket_cost + total_skirt_cost + total_shoe_cost

def mom_share := (2 / 3 : ℚ) * total_cost
def carrie_share := total_cost - mom_share

theorem carrie_payment : carrie_share = 228.67 :=
by
  sorry

end carrie_payment_l230_230590


namespace find_x_l230_230322

theorem find_x
  (a b c d k : ℝ)
  (h1 : a ≠ b)
  (h2 : b ≠ 0)
  (h3 : d ≠ 0)
  (h4 : k ≠ 0)
  (h5 : k ≠ 1)
  (h_frac_change : (a + k * x) / (b + x) = c / d) :
  x = (b * c - a * d) / (k * d - c) := by
  sorry

end find_x_l230_230322


namespace JackOfHeartsIsSane_l230_230251

inductive Card
  | Ace
  | Two
  | Three
  | Four
  | Five
  | Six
  | Seven
  | JackOfHearts

open Card

def Sane (c : Card) : Prop := sorry

axiom Condition1 : Sane Three → ¬ Sane Ace
axiom Condition2 : Sane Four → (¬ Sane Three ∨ ¬ Sane Two)
axiom Condition3 : Sane Five → (Sane Ace ↔ Sane Four)
axiom Condition4 : Sane Six → (Sane Ace ∧ Sane Two)
axiom Condition5 : Sane Seven → ¬ Sane Five
axiom Condition6 : Sane JackOfHearts → (¬ Sane Six ∨ ¬ Sane Seven)

theorem JackOfHeartsIsSane : Sane JackOfHearts := by
  sorry

end JackOfHeartsIsSane_l230_230251


namespace weight_of_bag_l230_230064

-- Definitions
def chicken_price : ℝ := 1.50
def bag_cost : ℝ := 2
def feed_per_chicken : ℝ := 2
def profit_from_50_chickens : ℝ := 65
def total_chickens : ℕ := 50

-- Theorem
theorem weight_of_bag : 
  (bag_cost / (profit_from_50_chickens - 
               (total_chickens * chicken_price)) / 
               (feed_per_chicken * total_chickens)) = 20 := 
sorry

end weight_of_bag_l230_230064


namespace complement_of_A_l230_230162

variables (U : Set ℝ) (A : Set ℝ)
def universal_set : Prop := U = Set.univ
def range_of_function : Prop := A = {x : ℝ | 0 ≤ x}

theorem complement_of_A (hU : universal_set U) (hA : range_of_function A) : 
  U \ A = {x : ℝ | x < 0} :=
by 
  sorry

end complement_of_A_l230_230162


namespace greatest_radius_l230_230566

theorem greatest_radius (r : ℕ) (h : π * (r : ℝ)^2 < 50 * π) : r = 7 :=
sorry

end greatest_radius_l230_230566


namespace exists_number_divisible_by_5_pow_1000_with_no_zeros_l230_230460

theorem exists_number_divisible_by_5_pow_1000_with_no_zeros :
  ∃ n : ℕ, (5 ^ 1000 ∣ n) ∧ (∀ d ∈ n.digits 10, d ≠ 0) := 
sorry

end exists_number_divisible_by_5_pow_1000_with_no_zeros_l230_230460


namespace part1_store_a_cost_part1_store_b_cost_part2_cost_comparison_part3_cost_effective_plan_l230_230003

-- Defining the conditions
def racket_price : ℕ := 50
def ball_price : ℕ := 20
def num_rackets : ℕ := 10

-- Store A cost function
def store_A_cost (x : ℕ) : ℕ := 20 * x + 300

-- Store B cost function
def store_B_cost (x : ℕ) : ℕ := 16 * x + 400

-- Part (1): Express the costs in algebraic form
theorem part1_store_a_cost (x : ℕ) (hx : 10 < x) : store_A_cost x = 20 * x + 300 := by
  sorry

theorem part1_store_b_cost (x : ℕ) (hx : 10 < x) : store_B_cost x = 16 * x + 400 := by
  sorry

-- Part (2): Cost for x = 40
theorem part2_cost_comparison : store_A_cost 40 > store_B_cost 40 := by
  sorry

-- Part (3): Most cost-effective purchasing plan
def store_a_cost_rackets : ℕ := racket_price * num_rackets
def store_a_free_balls : ℕ := num_rackets
def remaining_balls (total_balls : ℕ) : ℕ := total_balls - store_a_free_balls
def store_b_cost_remaining_balls (remaining_balls : ℕ) : ℕ := remaining_balls * ball_price * 4 / 5

theorem part3_cost_effective_plan : store_a_cost_rackets + store_b_cost_remaining_balls (remaining_balls 40) = 980 := by
  sorry

end part1_store_a_cost_part1_store_b_cost_part2_cost_comparison_part3_cost_effective_plan_l230_230003


namespace part_I_part_II_l230_230089

def f (x a : ℝ) : ℝ := |2 * x + 1| + |2 * x - a| + a

theorem part_I (x : ℝ) (h₁ : f x 3 > 7) : sorry := sorry

theorem part_II (a : ℝ) (h₂ : ∀ (x : ℝ), f x a ≥ 3) : sorry := sorry

end part_I_part_II_l230_230089


namespace work_completion_l230_230684

theorem work_completion (days_A : ℕ) (days_B : ℕ) (hA : days_A = 14) (hB : days_B = 35) :
  let rate_A := 1 / (days_A : ℚ)
  let rate_B := 1 / (days_B : ℚ)
  let combined_rate := rate_A + rate_B
  let days_AB := 1 / combined_rate
  days_AB = 10 := by
  sorry

end work_completion_l230_230684


namespace cyclist_total_heartbeats_l230_230069

theorem cyclist_total_heartbeats
  (heart_rate : ℕ := 120) -- beats per minute
  (race_distance : ℕ := 50) -- miles
  (pace : ℕ := 4) -- minutes per mile
  : (race_distance * pace) * heart_rate = 24000 := by
  sorry

end cyclist_total_heartbeats_l230_230069


namespace min_x8_x9_x10_eq_618_l230_230588

theorem min_x8_x9_x10_eq_618 (x : ℕ → ℕ) (h1 : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 10 → x i < x j)
  (h2 : x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 + x 9 + x 10 = 2023) :
  x 8 + x 9 + x 10 = 618 :=
sorry

end min_x8_x9_x10_eq_618_l230_230588


namespace intersection_has_one_element_l230_230335

noncomputable def A (a : ℝ) : Set ℝ := {1, a, 5}
noncomputable def B (a : ℝ) : Set ℝ := {2, a^2 + 1}

theorem intersection_has_one_element (a : ℝ) (h : ∃ x, A a ∩ B a = {x}) : a = 0 ∨ a = -2 :=
by {
  sorry
}

end intersection_has_one_element_l230_230335


namespace tangent_parallel_l230_230715

noncomputable def f (x : ℝ) : ℝ := x^4 - x

theorem tangent_parallel (P : ℝ × ℝ) (hP : P.1 = 1) (hP_cond : P.2 = f P.1) 
  (tangent_parallel : ∀ x, deriv f x = 3) : P = (1, 0) := 
by 
  have h_deriv : deriv f 1 = 4 * 1^3 - 1 := by sorry
  have slope_eq : deriv f (P.1) = 3 := by sorry
  have solve_a : P.1 = 1 := by sorry
  have solve_b : f 1 = 0 := by sorry
  exact sorry

end tangent_parallel_l230_230715


namespace tire_circumference_l230_230398

variable (rpm : ℕ) (car_speed_kmh : ℕ) (circumference : ℝ)

-- Define the conditions
def conditions : Prop :=
  rpm = 400 ∧ car_speed_kmh = 24

-- Define the statement to prove
theorem tire_circumference (h : conditions rpm car_speed_kmh) : circumference = 1 :=
sorry

end tire_circumference_l230_230398


namespace factorize_problem1_factorize_problem2_l230_230729

theorem factorize_problem1 (x : ℝ) : 2 * x^2 - 4 * x + 2 = 2 * (x - 1)^2 :=
by sorry

theorem factorize_problem2 (x y : ℝ) : 
  (x - y)^3 - 16 * (x - y) = (x - y) * (x - y + 4) * (x - y - 4) :=
by sorry

end factorize_problem1_factorize_problem2_l230_230729


namespace primes_in_arithmetic_sequence_have_specific_ones_digit_l230_230630

-- Define the properties of the primes and the arithmetic sequence
theorem primes_in_arithmetic_sequence_have_specific_ones_digit
  (p q r s : ℕ) 
  (prime_p : Nat.Prime p)
  (prime_q : Nat.Prime q)
  (prime_r : Nat.Prime r)
  (prime_s : Nat.Prime s)
  (arithmetic_sequence : q = p + 4 ∧ r = q + 4 ∧ s = r + 4)
  (p_gt_3 : p > 3) : 
  p % 10 = 9 := 
sorry

end primes_in_arithmetic_sequence_have_specific_ones_digit_l230_230630


namespace find_m_of_line_with_slope_l230_230921

theorem find_m_of_line_with_slope (m : ℝ) (h_pos : m > 0)
(h_slope : (m - 4) / (2 - m) = m^2) : m = 2 := by
  sorry

end find_m_of_line_with_slope_l230_230921


namespace rod_total_length_l230_230325

theorem rod_total_length
  (n : ℕ) (l : ℝ)
  (h₁ : n = 50)
  (h₂ : l = 0.85) :
  n * l = 42.5 := by
  sorry

end rod_total_length_l230_230325


namespace even_function_a_eq_4_l230_230903

noncomputable def f (x a : ℝ) : ℝ := (x + a) * (x - 4)

theorem even_function_a_eq_4 (a : ℝ) (h : ∀ x : ℝ, f (-x) a = f x a) : a = 4 := by
  sorry

end even_function_a_eq_4_l230_230903


namespace working_mom_work_percentage_l230_230297

theorem working_mom_work_percentage :
  let total_hours_in_day := 24
  let work_hours := 8
  let gym_hours := 2
  let cooking_hours := 1.5
  let bath_hours := 0.5
  let homework_hours := 1
  let packing_hours := 0.5
  let cleaning_hours := 0.5
  let leisure_hours := 2
  let total_activity_hours := work_hours + gym_hours + cooking_hours + bath_hours + homework_hours + packing_hours + cleaning_hours + leisure_hours
  16 = total_activity_hours →
  (work_hours / total_hours_in_day) * 100 = 33.33 :=
by
  sorry

end working_mom_work_percentage_l230_230297


namespace product_of_two_integers_l230_230604

theorem product_of_two_integers (x y : ℕ) (h1 : x + y = 26) (h2 : x^2 - y^2 = 52) (h3 : x > y) : x * y = 168 := by
  sorry

end product_of_two_integers_l230_230604


namespace rectangle_area_l230_230912

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

end rectangle_area_l230_230912


namespace factor_expression_l230_230730

variable (a b : ℤ)

theorem factor_expression : 2 * a^2 * b - 4 * a * b^2 + 2 * b^3 = 2 * b * (a - b)^2 := 
sorry

end factor_expression_l230_230730


namespace problem_statement_l230_230843

theorem problem_statement (x : ℝ) (h : x = -3) :
  (5 + x * (5 + x) - 5^2) / (x - 5 + x^2) = -26 := by
  rw [h]
  sorry

end problem_statement_l230_230843


namespace determine_a_l230_230213

theorem determine_a (a : ℝ) (h : 2 * (-1) + a = 3) : a = 5 := sorry

end determine_a_l230_230213


namespace smallest_part_division_l230_230666

theorem smallest_part_division (y : ℝ) (h1 : y > 0) :
  ∃ (x : ℝ), x = y / 9 ∧ (∃ (a b c : ℝ), a = x ∧ b = 3 * x ∧ c = 5 * x ∧ a + b + c = y) :=
sorry

end smallest_part_division_l230_230666


namespace domain_of_function_l230_230281

theorem domain_of_function :
  {x : ℝ | x^3 + 5*x^2 + 6*x ≠ 0} =
  {x : ℝ | x < -3} ∪ {x : ℝ | -3 < x ∧ x < -2} ∪ {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 0 < x} :=
by
  sorry

end domain_of_function_l230_230281


namespace iterated_kernels_l230_230526

noncomputable def K (x t : ℝ) : ℝ := 
  if 0 ≤ x ∧ x < t then 
    x + t 
  else if t < x ∧ x ≤ 1 then 
    x - t 
  else 
    0

noncomputable def K1 (x t : ℝ) : ℝ := K x t

noncomputable def K2 (x t : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < t then 
    (-2 / 3) * x^3 + t^3 - x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else if t < x ∧ x ≤ 1 then 
    (-2 / 3) * x^3 - t^3 + x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else
    0

theorem iterated_kernels (x t : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1) :
  K1 x t = K x t ∧
  K2 x t = 
  if 0 ≤ x ∧ x < t then 
    (-2 / 3) * x^3 + t^3 - x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else if t < x ∧ x ≤ 1 then 
    (-2 / 3) * x^3 - t^3 + x^2 * t + 2 * x * t^2 - x * t + (x - t) / 2 + 1 / 3
  else
    0 := by
  sorry

end iterated_kernels_l230_230526


namespace cost_of_one_dozen_pens_l230_230364

theorem cost_of_one_dozen_pens (x n : ℕ) (h₁ : 5 * n * x + 5 * x = 200) (h₂ : ∀ p : ℕ, p > 0 → p ≠ x * 5 → x * 5 ≠ x) :
  12 * 5 * x = 120 :=
by
  sorry

end cost_of_one_dozen_pens_l230_230364


namespace max_value_of_g_l230_230315

def g (n : ℕ) : ℕ :=
  if n < 20 then n + 20 else g (n - 7)

theorem max_value_of_g : ∀ n : ℕ, g n ≤ 39 ∧ (∃ m : ℕ, g m = 39) := by
  sorry

end max_value_of_g_l230_230315


namespace side_length_of_square_l230_230048

theorem side_length_of_square (total_length : ℝ) (sides : ℕ) (h1 : total_length = 100) (h2 : sides = 4) :
  (total_length / (sides : ℝ) = 25) :=
by
  sorry

end side_length_of_square_l230_230048


namespace determine_x_l230_230143

theorem determine_x (x y : ℤ) (h1 : x + 2 * y = 20) (h2 : y = 5) : x = 10 := 
by 
  sorry

end determine_x_l230_230143


namespace mixture_weight_l230_230910

theorem mixture_weight (C : ℚ) (W : ℚ)
  (H1: C > 0) -- C represents the cost per pound of milk powder and coffee in June, and is a positive number
  (H2: C * 0.2 = 0.2) -- The price per pound of milk powder in July
  (H3: (W / 2) * 0.2 + (W / 2) * 4 * C = 6.30) -- The cost of the mixture in July

  : W = 3 := 
sorry

end mixture_weight_l230_230910


namespace medium_sized_fir_trees_count_l230_230986

theorem medium_sized_fir_trees_count 
  (total_trees : ℕ) (ancient_oaks : ℕ) (saplings : ℕ)
  (h1 : total_trees = 96)
  (h2 : ancient_oaks = 15)
  (h3 : saplings = 58) :
  total_trees - ancient_oaks - saplings = 23 :=
by 
  sorry

end medium_sized_fir_trees_count_l230_230986


namespace sara_dozen_quarters_l230_230376

theorem sara_dozen_quarters (dollars : ℕ) (quarters_per_dollar : ℕ) (quarters_per_dozen : ℕ) 
  (h1 : dollars = 9) (h2 : quarters_per_dollar = 4) (h3 : quarters_per_dozen = 12) : 
  dollars * quarters_per_dollar / quarters_per_dozen = 3 := 
by 
  sorry

end sara_dozen_quarters_l230_230376


namespace parabola_equation_l230_230087

theorem parabola_equation (p : ℝ) (h1 : 0 < p) (h2 : p / 2 = 2) : ∀ y x : ℝ, y^2 = -8 * x :=
by
  sorry

end parabola_equation_l230_230087


namespace average_age_increase_l230_230673

theorem average_age_increase
  (n : ℕ)
  (A : ℝ)
  (w : ℝ)
  (h1 : (n + 1) * (A + w) = n * A + 39)
  (h2 : (n + 1) * (A - 1) = n * A + 15)
  (hw : w = 7) :
  w = 7 := 
by
  sorry

end average_age_increase_l230_230673


namespace find_a8_l230_230567

theorem find_a8 (a : ℕ → ℤ) (x : ℤ) :
  (1 + x)^10 = a 0 + a 1 * (1 - x) + a 2 * (1 - x)^2 + a 3 * (1 - x)^3 +
               a 4 * (1 - x)^4 + a 5 * (1 - x)^5 + a 6 * (1 - x)^6 +
               a 7 * (1 - x)^7 + a 8 * (1 - x)^8 + a 9 * (1 - x)^9 +
               a 10 * (1 - x)^10 → a 8 = 180 := by
  sorry

end find_a8_l230_230567


namespace ella_spent_on_video_games_last_year_l230_230655

theorem ella_spent_on_video_games_last_year 
  (new_salary : ℝ) 
  (raise : ℝ) 
  (percentage_spent_on_video_games : ℝ) 
  (h_new_salary : new_salary = 275) 
  (h_raise : raise = 0.10) 
  (h_percentage_spent : percentage_spent_on_video_games = 0.40) :
  (new_salary / (1 + raise) * percentage_spent_on_video_games = 100) :=
by
  sorry

end ella_spent_on_video_games_last_year_l230_230655


namespace percentage_both_questions_correct_l230_230625

-- Definitions for the conditions in the problem
def percentage_first_question_correct := 85
def percentage_second_question_correct := 65
def percentage_neither_question_correct := 5
def percentage_one_or_more_questions_correct := 100 - percentage_neither_question_correct

-- Theorem stating that 55 percent answered both questions correctly
theorem percentage_both_questions_correct :
  percentage_first_question_correct + percentage_second_question_correct - percentage_one_or_more_questions_correct = 55 :=
by
  sorry

end percentage_both_questions_correct_l230_230625


namespace range_of_m_to_satisfy_quadratic_l230_230440

def quadratic_positive_forall_m (m : ℝ) : Prop :=
  ∀ x : ℝ, m * x^2 + m * x + 100 > 0

theorem range_of_m_to_satisfy_quadratic :
  {m : ℝ | quadratic_positive_forall_m m} = {m : ℝ | 0 ≤ m ∧ m < 400} :=
by
  sorry

end range_of_m_to_satisfy_quadratic_l230_230440


namespace vector_coordinates_l230_230156

theorem vector_coordinates (b : ℝ × ℝ)
  (a : ℝ × ℝ := (Real.sqrt 3, 1))
  (angle : ℝ := 2 * Real.pi / 3)
  (norm_b : ℝ := 1)
  (dot_product_eq : (a.fst * b.fst + a.snd * b.snd = -1))
  (norm_b_eq : (b.fst ^ 2 + b.snd ^ 2 = 1)) :
  b = (0, -1) ∨ b = (-Real.sqrt 3 / 2, 1 / 2) :=
sorry

end vector_coordinates_l230_230156


namespace r_sq_plus_s_sq_l230_230065

variable {r s : ℝ}

theorem r_sq_plus_s_sq (h1 : r * s = 16) (h2 : r + s = 10) : r^2 + s^2 = 68 := 
by
  sorry

end r_sq_plus_s_sq_l230_230065


namespace book_price_l230_230893

theorem book_price (P : ℝ) : 
  (3 * 12 * P - 500 = 220) → 
  P = 20 :=
by
  intro h
  sorry

end book_price_l230_230893


namespace micah_total_strawberries_l230_230929

theorem micah_total_strawberries (eaten saved total : ℕ) 
  (h1 : eaten = 6) 
  (h2 : saved = 18) 
  (h3 : total = eaten + saved) : 
  total = 24 := 
by
  sorry

end micah_total_strawberries_l230_230929


namespace price_per_foot_of_fencing_l230_230841

theorem price_per_foot_of_fencing
  (area : ℝ) (total_cost : ℝ) (price_per_foot : ℝ)
  (h1 : area = 36) (h2 : total_cost = 1392) :
  price_per_foot = 58 :=
by
  sorry

end price_per_foot_of_fencing_l230_230841


namespace trig_expression_zero_l230_230441

theorem trig_expression_zero (θ : ℝ) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 :=
sorry

end trig_expression_zero_l230_230441


namespace vector_addition_proof_l230_230547

variables {Point : Type} [AddCommGroup Point]

variables (A B C D : Point)

theorem vector_addition_proof :
  (D - A) + (C - D) - (C - B) = B - A :=
by
  sorry

end vector_addition_proof_l230_230547


namespace total_cost_maria_l230_230791

-- Define the cost of the pencil
def cost_pencil : ℕ := 8

-- Define the cost of the pen as half the price of the pencil
def cost_pen : ℕ := cost_pencil / 2

-- Define the total cost for both the pen and the pencil
def total_cost : ℕ := cost_pencil + cost_pen

-- Prove that total cost is equal to 12
theorem total_cost_maria : total_cost = 12 := 
by
  -- skip the proof
  sorry

end total_cost_maria_l230_230791


namespace marathon_finishers_l230_230402

-- Define the conditions
def totalParticipants : ℕ := 1250
def peopleGaveUp (F : ℕ) : ℕ := F + 124

-- Define the final statement to be proved
theorem marathon_finishers (F : ℕ) (h1 : totalParticipants = F + peopleGaveUp F) : F = 563 :=
by sorry

end marathon_finishers_l230_230402


namespace total_price_of_order_l230_230294

theorem total_price_of_order :
  let num_ice_cream_bars := 225
  let price_per_ice_cream_bar := 0.60
  let num_sundaes := 125
  let price_per_sundae := 0.52
  (num_ice_cream_bars * price_per_ice_cream_bar + num_sundaes * price_per_sundae) = 200 := 
by
  -- The proof steps go here
  sorry

end total_price_of_order_l230_230294


namespace volume_formula_correct_l230_230073

def volume_of_box (x : ℝ) : ℝ :=
  x * (16 - 2 * x) * (12 - 2 * x)

theorem volume_formula_correct (x : ℝ) (h : x ≤ 12 / 5) :
  volume_of_box x = 4 * x^3 - 56 * x^2 + 192 * x :=
by sorry

end volume_formula_correct_l230_230073


namespace scalar_mult_l230_230905

variables {α : Type*} [AddCommGroup α] [Module ℝ α]

theorem scalar_mult (a : α) (h : a ≠ 0) : (-4) • (3 • a) = -12 • a :=
  sorry

end scalar_mult_l230_230905


namespace count_heads_at_night_l230_230081

variables (J T D : ℕ)

theorem count_heads_at_night (h1 : 2 * J + 4 * T + 2 * D = 56) : J + T + D = 14 :=
by
  -- Skip the proof
  sorry

end count_heads_at_night_l230_230081


namespace symmetric_circle_equation_l230_230850

theorem symmetric_circle_equation :
  ∀ (x y : ℝ), (x + 2) ^ 2 + y ^ 2 = 5 → (x - 2) ^ 2 + y ^ 2 = 5 :=
by 
  sorry

end symmetric_circle_equation_l230_230850


namespace probability_of_all_co_captains_l230_230989

def team_sizes : List ℕ := [6, 8, 9, 10]

def captains_per_team : ℕ := 3

noncomputable def probability_all_co_captains (s : ℕ) : ℚ :=
  1 / (Nat.choose s 3 : ℚ)

noncomputable def total_probability : ℚ :=
  (1 / 4 : ℚ) * 
  (probability_all_co_captains 6 + 
   probability_all_co_captains 8 +
   probability_all_co_captains 9 +
   probability_all_co_captains 10)

theorem probability_of_all_co_captains : total_probability = 1 / 84 :=
  sorry

end probability_of_all_co_captains_l230_230989


namespace prove_AB_and_circle_symmetry_l230_230264

-- Definition of point A
def pointA : ℝ × ℝ := (4, -3)

-- Lengths relation |AB| = 2|OA|
def lengths_relation(u v : ℝ) : Prop :=
  u^2 + v^2 = 100

-- Orthogonality condition for AB and OA
def orthogonality_condition(u v : ℝ) : Prop :=
  4 * u - 3 * v = 0

-- Condition that ordinate of B is greater than 0
def ordinate_condition(v : ℝ) : Prop :=
  v - 3 > 0

-- Equation of the circle given in the problem
def given_circle_eqn(x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 1)^2 = 10

-- Symmetric circle equation to be proved
def symmetric_circle_eqn(x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 10

theorem prove_AB_and_circle_symmetry :
  (∃ u v : ℝ, lengths_relation u v ∧ orthogonality_condition u v ∧ ordinate_condition v ∧ u = 6 ∧ v = 8) ∧
  (∃ x y : ℝ, given_circle_eqn x y → symmetric_circle_eqn x y) :=
by
  sorry

end prove_AB_and_circle_symmetry_l230_230264


namespace type_B_machine_time_l230_230487

theorem type_B_machine_time :
  (2 * (1 / 5) + 3 * (1 / B) = 5 / 6) → B = 90 / 13 :=
by 
  intro h
  sorry

end type_B_machine_time_l230_230487


namespace find_p_fifth_plus_3_l230_230857

theorem find_p_fifth_plus_3 (p : ℕ) (hp : Nat.Prime p) (h : Nat.Prime (p^4 + 3)) :
  p^5 + 3 = 35 :=
sorry

end find_p_fifth_plus_3_l230_230857


namespace matt_new_average_commission_l230_230138

noncomputable def new_average_commission (x : ℝ) : ℝ :=
  (5 * x + 1000) / 6

theorem matt_new_average_commission
  (x : ℝ)
  (h1 : (5 * x + 1000) / 6 = x + 150)
  (h2 : x = 100) :
  new_average_commission x = 250 :=
by
  sorry

end matt_new_average_commission_l230_230138


namespace Linda_total_distance_is_25_l230_230749

theorem Linda_total_distance_is_25 : 
  ∃ (x : ℤ), x > 0 ∧ 
  (60/x + 60/(x+5) + 60/(x+10) + 60/(x+15) = 25) :=
by 
  sorry

end Linda_total_distance_is_25_l230_230749


namespace alice_min_speed_l230_230772

open Real

theorem alice_min_speed (d : ℝ) (bob_speed : ℝ) (alice_delay : ℝ) (alice_time : ℝ) :
  d = 180 → bob_speed = 40 → alice_delay = 0.5 → alice_time = 4 → d / alice_time > (d / bob_speed) - alice_delay →
  d / alice_time > 45 := by
  sorry


end alice_min_speed_l230_230772


namespace gcd_m_n_l230_230498

   -- Define m and n according to the problem statement
   def m : ℕ := 33333333
   def n : ℕ := 666666666

   -- State the theorem we want to prove
   theorem gcd_m_n : Int.gcd m n = 3 := by
     -- put proof here
     sorry
   
end gcd_m_n_l230_230498


namespace lines_intersect_and_not_perpendicular_l230_230837

theorem lines_intersect_and_not_perpendicular (a : ℝ) :
  (∃ (x y : ℝ), 3 * x + 3 * y + a = 0 ∧ 3 * x - 2 * y + 1 = 0) ∧ 
  ¬ (∃ k1 k2 : ℝ, k1 = -1 ∧ k2 = 3 / 2 ∧ k1 ≠ k2 ∧ k1 * k2 = -1) :=
by
  sorry

end lines_intersect_and_not_perpendicular_l230_230837


namespace arrangement_problem_l230_230185

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem arrangement_problem 
  (p1 p2 p3 p4 p5 : Type)  -- Representing the five people
  (youngest : p1)         -- Specifying the youngest
  (oldest : p5)           -- Specifying the oldest
  (unique_people : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ p3 ≠ p4 ∧ p3 ≠ p5 ∧ p4 ≠ p5) -- Ensuring five unique people
  : (factorial 5) - (factorial 4 * 2) = 72 :=
by sorry

end arrangement_problem_l230_230185


namespace problem_correct_l230_230235

theorem problem_correct (x : ℝ) : 
  14 * ((150 / 3) + (35 / 7) + (16 / 32) + x) = 777 + 14 * x := 
by
  sorry

end problem_correct_l230_230235


namespace customer_B_cost_effectiveness_customer_A_boxes_and_consumption_l230_230974

theorem customer_B_cost_effectiveness (box_orig_cost box_spec_cost : ℕ) (orig_price spec_price eggs_per_box remaining_eggs : ℕ) 
    (h1 : orig_price = 15) (h2 : spec_price = 12) (h3 : eggs_per_box = 30) 
    (h4 : remaining_eggs = 20) : 
    ¬ (spec_price * 2 / (eggs_per_box * 2 - remaining_eggs) < orig_price / eggs_per_box) :=
by
  sorry

theorem customer_A_boxes_and_consumption (orig_price spec_price eggs_per_box total_cost_savings : ℕ) 
    (h1 : orig_price = 15) (h2 : spec_price = 12) (h3 : eggs_per_box = 30) 
    (h4 : total_cost_savings = 90): 
  ∃ (boxes_bought : ℕ) (avg_daily_consumption : ℕ), 
    (spec_price * boxes_bought = orig_price * boxes_bought * 2 - total_cost_savings) ∧ 
    (avg_daily_consumption = eggs_per_box * boxes_bought / 15) :=
by
  sorry

end customer_B_cost_effectiveness_customer_A_boxes_and_consumption_l230_230974


namespace problem_solution_l230_230641

-- Define the conditions
variables {a c b d x y z q : Real}
axiom h1 : a^x = c^q ∧ c^q = b
axiom h2 : c^y = a^z ∧ a^z = d

-- State the theorem
theorem problem_solution : xy = zq :=
by
  sorry

end problem_solution_l230_230641


namespace tallest_is_jie_l230_230972

variable (Igor Jie Faye Goa Han : Type)
variable (Shorter : Type → Type → Prop) -- Shorter relation

axiom igor_jie : Shorter Igor Jie
axiom faye_goa : Shorter Goa Faye
axiom jie_faye : Shorter Faye Jie
axiom han_goa : Shorter Han Goa

theorem tallest_is_jie : ∀ p, p = Jie :=
by
  sorry

end tallest_is_jie_l230_230972


namespace semicircle_radius_l230_230410

theorem semicircle_radius (b h : ℝ) (base_eq_b : b = 16) (height_eq_h : h = 15) :
  let s := (2 * 17) / 2
  let area := 240 
  s * (r : ℝ) = area → r = 120 / 17 :=
  by
  intros s area
  sorry

end semicircle_radius_l230_230410


namespace evaluate_composite_function_l230_230719

def f (x : ℝ) : ℝ := x^2 - 2 * x + 2
def g (x : ℝ) : ℝ := 3 * x + 2

theorem evaluate_composite_function :
  f (g (-2)) = 26 := by
  sorry

end evaluate_composite_function_l230_230719


namespace marathon_fraction_l230_230122

theorem marathon_fraction :
  ∃ (f : ℚ), (2 * 7) = (6 + (6 + 6 * f)) ∧ f = 1 / 3 :=
by 
  sorry

end marathon_fraction_l230_230122


namespace smallest_n_solution_unique_l230_230190

theorem smallest_n_solution_unique (a b c d : ℤ) (h : a^2 + b^2 + c^2 = 4 * d^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
sorry

end smallest_n_solution_unique_l230_230190


namespace mr_brown_financial_outcome_l230_230616

theorem mr_brown_financial_outcome :
  ∃ (C₁ C₂ : ℝ), (2.40 = 1.25 * C₁) ∧ (2.40 = 0.75 * C₂) ∧ ((2.40 + 2.40) - (C₁ + C₂) = -0.32) :=
by
  sorry

end mr_brown_financial_outcome_l230_230616


namespace pq_ratio_at_0_l230_230032

noncomputable def p (x : ℝ) : ℝ := -3 * (x + 4) * x
noncomputable def q (x : ℝ) : ℝ := (x + 4) * (x - 3)

theorem pq_ratio_at_0 : (p 0) / (q 0) = 0 := by
  sorry

end pq_ratio_at_0_l230_230032


namespace number_of_classmates_late_l230_230331

-- Definitions based on conditions from problem statement
def charlizeLate : ℕ := 20
def classmateLate : ℕ := charlizeLate + 10
def totalLateTime : ℕ := 140

-- The proof statement
theorem number_of_classmates_late (x : ℕ) (h1 : totalLateTime = charlizeLate + x * classmateLate) : x = 4 :=
by
  sorry

end number_of_classmates_late_l230_230331


namespace find_age_l230_230101

variable (x : ℤ)

def age_4_years_hence := x + 4
def age_4_years_ago := x - 4
def brothers_age := x - 6

theorem find_age (hx : x = 4 * (x + 4) - 4 * (x - 4) + 1/2 * (x - 6)) : x = 58 :=
sorry

end find_age_l230_230101


namespace items_sold_each_house_l230_230649

-- Define the conditions
def visits_day_one : ℕ := 20
def visits_day_two : ℕ := 2 * visits_day_one
def sale_percentage_day_two : ℝ := 0.8
def total_sales : ℕ := 104

-- Define the number of items sold at each house
variable (x : ℕ)

-- Define the main Lean 4 statement for the proof
theorem items_sold_each_house (h1 : 20 * x + 32 * x = 104) : x = 2 :=
by
  -- Proof would go here
  sorry

end items_sold_each_house_l230_230649


namespace max_difference_y_coords_intersection_l230_230923

def f (x : ℝ) : ℝ := 4 - x^2 + x^3
def g (x : ℝ) : ℝ := x^2 + x^4

theorem max_difference_y_coords_intersection : ∀ x : ℝ, 
  (f x = g x) → 
  (∀ x₁ x₂ : ℝ, f x₁ = g x₁ ∧ f x₂ = g x₂ → |f x₁ - f x₂| = 0) := 
by
  sorry

end max_difference_y_coords_intersection_l230_230923


namespace function_value_range_l230_230233

noncomputable def f (x : ℝ) : ℝ := 9^x - 3^(x+1) + 2

theorem function_value_range :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → -1/4 ≤ f x ∧ f x ≤ 2 :=
by
  sorry

end function_value_range_l230_230233


namespace geometric_sequence_formula_l230_230637

variable {q : ℝ} -- Common ratio
variable {m n : ℕ} -- Positive natural numbers
variable {b : ℕ → ℝ} -- Geometric sequence

-- This is only necessary if importing Mathlib didn't bring it in
noncomputable def geom_sequence (m n : ℕ) (b : ℕ → ℝ) (q : ℝ) : Prop :=
  b n = b m * q^(n - m)

theorem geometric_sequence_formula (q : ℝ) (m n : ℕ) (b : ℕ → ℝ) 
  (hmn : 0 < m ∧ 0 < n) :
  geom_sequence m n b q :=
by sorry

end geometric_sequence_formula_l230_230637


namespace students_play_long_tennis_l230_230973

-- Define the parameters for the problem
def total_students : ℕ := 38
def football_players : ℕ := 26
def both_sports_players : ℕ := 17
def neither_sports_players : ℕ := 9

-- Total students playing at least one sport
def at_least_one := total_students - neither_sports_players

-- Define the Lean theorem statement
theorem students_play_long_tennis : at_least_one = football_players + (20 : ℕ) - both_sports_players := 
by 
  -- Translate the given facts into the Lean proof structure
  have h1 : at_least_one = 29 := by rfl -- total_students - neither_sports_players
  have h2 : football_players = 26 := by rfl
  have h3 : both_sports_players = 17 := by rfl
  show 29 = 26 + 20 - 17
  sorry

end students_play_long_tennis_l230_230973


namespace digits_satisfy_sqrt_l230_230063

theorem digits_satisfy_sqrt (a b : ℕ) (ha : a ≤ 9) (hb : b ≤ 9) :
  (b = 0 ∧ a = 0) ∨ (b = 3 ∧ a = 1) ∨ (b = 6 ∧ a = 4) ∨ (b = 9 ∧ a = 9) ↔ b^2 = 9 * a :=
by
  sorry

end digits_satisfy_sqrt_l230_230063


namespace price_of_bracelets_max_type_a_bracelets_l230_230661

-- Part 1: Proving the prices of the bracelets
theorem price_of_bracelets :
  ∃ (x y : ℝ), (3 * x + y = 128 ∧ x + 2 * y = 76) ∧ (x = 36 ∧ y = 20) :=
sorry

-- Part 2: Proving the maximum number of type A bracelets they can buy within the budget
theorem max_type_a_bracelets :
  ∃ (m : ℕ), 36 * m + 20 * (100 - m) ≤ 2500 ∧ m = 31 :=
sorry

end price_of_bracelets_max_type_a_bracelets_l230_230661


namespace smallest_relatively_prime_210_l230_230201

theorem smallest_relatively_prime_210 : ∃ x : ℕ, x > 1 ∧ Nat.gcd x 210 = 1 ∧ (∀ y : ℕ, y > 1 → y < x → Nat.gcd y 210 ≠ 1) :=
sorry

end smallest_relatively_prime_210_l230_230201


namespace initial_quantity_of_milk_l230_230946

-- Define initial condition for the quantity of milk in container A
noncomputable def container_A : ℝ := 1184

-- Define the quantities of milk in containers B and C
def container_B (A : ℝ) : ℝ := 0.375 * A
def container_C (A : ℝ) : ℝ := 0.625 * A

-- Define the final equal quantities of milk after transfer
def equal_quantity (A : ℝ) : ℝ := container_B A + 148

-- The proof statement that must be true
theorem initial_quantity_of_milk :
  ∀ (A : ℝ), container_B A + 148 = equal_quantity A → A = container_A :=
by
  intros A h
  rw [equal_quantity] at h
  sorry

end initial_quantity_of_milk_l230_230946


namespace toys_produced_each_day_l230_230170

theorem toys_produced_each_day (total_weekly_production : ℕ) (days_per_week : ℕ) (H1 : total_weekly_production = 6500) (H2 : days_per_week = 5) : (total_weekly_production / days_per_week = 1300) :=
by {
  sorry
}

end toys_produced_each_day_l230_230170


namespace root_range_of_quadratic_eq_l230_230486

theorem root_range_of_quadratic_eq (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 < x2 ∧ x1^2 + k * x1 - k = 0 ∧ x2^2 + k * x2 - k = 0 ∧ 1 < x1 ∧ x1 < 2 ∧ 2 < x2 ∧ x2 < 3) ↔  (-9 / 2) < k ∧ k < -4 :=
by
  sorry

end root_range_of_quadratic_eq_l230_230486


namespace original_triangle_area_l230_230353

theorem original_triangle_area (A_new : ℝ) (r : ℝ) (A_original : ℝ) 
  (h1 : r = 3) 
  (h2 : A_new = 54) 
  (h3 : A_new = r^2 * A_original) : 
  A_original = 6 := 
by 
  sorry

end original_triangle_area_l230_230353


namespace harry_morning_routine_time_l230_230650

-- Define the conditions in Lean.
def buy_coffee_and_bagel_time : ℕ := 15 -- minutes
def read_and_eat_time : ℕ := 2 * buy_coffee_and_bagel_time -- twice the time for buying coffee and bagel is 30 minutes

-- Define the total morning routine time in Lean.
def total_morning_routine_time : ℕ := buy_coffee_and_bagel_time + read_and_eat_time

-- The final proof problem statement.
theorem harry_morning_routine_time :
  total_morning_routine_time = 45 :=
by
  unfold total_morning_routine_time
  unfold read_and_eat_time
  unfold buy_coffee_and_bagel_time
  sorry

end harry_morning_routine_time_l230_230650


namespace find_numbers_l230_230000

theorem find_numbers (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (geom_mean_cond : Real.sqrt (a * b) = Real.sqrt 5)
  (harm_mean_cond : 2 / ((1 / a) + (1 / b)) = 2) :
  (a = (5 + Real.sqrt 5) / 2 ∧ b = (5 - Real.sqrt 5) / 2) ∨
  (a = (5 - Real.sqrt 5) / 2 ∧ b = (5 + Real.sqrt 5) / 2) :=
by
  sorry

end find_numbers_l230_230000


namespace loom_weaving_rate_l230_230052

theorem loom_weaving_rate :
  (119.04761904761905 : ℝ) > 0 ∧ (15 : ℝ) > 0 ∧ ∃ rate : ℝ, rate = 15 / 119.04761904761905 → rate = 0.126 :=
by sorry

end loom_weaving_rate_l230_230052


namespace gcd_228_1995_l230_230814

theorem gcd_228_1995 : Int.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l230_230814


namespace find_angle_A_l230_230517

theorem find_angle_A (a b c A : ℝ) (h1 : b = c) (h2 : a^2 = 2 * b^2 * (1 - Real.sin A)) : 
  A = Real.pi / 4 :=
by
  sorry

end find_angle_A_l230_230517


namespace four_digit_number_l230_230470

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 1000 + ((n / 10) % 10) * 100 + ((n / 100) % 10) * 10 + (n / 1000)

theorem four_digit_number (n : ℕ) (hn1 : 1000 ≤ n) (hn2 : n < 10000) (condition : n = 9 * (reverse_digits n)) :
  n = 9801 :=
by
  sorry

end four_digit_number_l230_230470


namespace sum_of_3_consecutive_multiples_of_3_l230_230152

theorem sum_of_3_consecutive_multiples_of_3 (a b c : ℕ) (h₁ : a = b + 3) (h₂ : b = c + 3) (h₃ : a = 42) : a + b + c = 117 :=
by sorry

end sum_of_3_consecutive_multiples_of_3_l230_230152


namespace enrique_commission_l230_230037

-- Define parameters for the problem
def suit_price : ℝ := 700
def suits_sold : ℝ := 2

def shirt_price : ℝ := 50
def shirts_sold : ℝ := 6

def loafer_price : ℝ := 150
def loafers_sold : ℝ := 2

def commission_rate : ℝ := 0.15

-- Calculate total sales for each category
def total_suit_sales : ℝ := suit_price * suits_sold
def total_shirt_sales : ℝ := shirt_price * shirts_sold
def total_loafer_sales : ℝ := loafer_price * loafers_sold

-- Calculate total sales
def total_sales : ℝ := total_suit_sales + total_shirt_sales + total_loafer_sales

-- Calculate commission
def commission : ℝ := commission_rate * total_sales

-- Proof statement that Enrique's commission is $300
theorem enrique_commission : commission = 300 := sorry

end enrique_commission_l230_230037


namespace platform_length_is_correct_l230_230831

noncomputable def length_of_platform (time_to_pass_man : ℝ) (time_to_cross_platform : ℝ) (length_of_train : ℝ) : ℝ := 
  length_of_train * time_to_cross_platform / time_to_pass_man - length_of_train

theorem platform_length_is_correct : length_of_platform 8 20 178 = 267 := 
  sorry

end platform_length_is_correct_l230_230831


namespace compound_interest_is_correct_l230_230082

noncomputable def compoundInterest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * (1 + R)^T - P

theorem compound_interest_is_correct
  (P : ℝ)
  (R : ℝ)
  (T : ℝ)
  (SI : ℝ) : SI = P * R * T / 100 ∧ R = 0.10 ∧ T = 2 ∧ SI = 600 → compoundInterest P R T = 630 :=
by
  sorry

end compound_interest_is_correct_l230_230082


namespace exam_time_ratio_l230_230455

-- Lean statements to define the problem conditions and goal
theorem exam_time_ratio (x M : ℝ) (h1 : x > 0) (h2 : M = x / 18) : 
  (5 * x / 6 + 2 * M) / (x / 6 - 2 * M) = 17 := by
  sorry

end exam_time_ratio_l230_230455


namespace heather_counts_209_l230_230108

def alice_numbers (n : ℕ) : ℕ := 5 * n - 2
def general_skip_numbers (m : ℕ) : ℕ := 3 * m - 1
def heather_number := 209

theorem heather_counts_209 :
  (∀ n, alice_numbers n > 0 ∧ alice_numbers n ≤ 500 → ¬heather_number = alice_numbers n) ∧
  (∀ m, general_skip_numbers m > 0 ∧ general_skip_numbers m ≤ 500 → ¬heather_number = general_skip_numbers m) ∧
  (1 ≤ heather_number ∧ heather_number ≤ 500) :=
by
  sorry

end heather_counts_209_l230_230108


namespace math_problem_l230_230352

variable (a b c : ℤ)

theorem math_problem
  (h₁ : 3 * a + 4 * b + 5 * c = 0)
  (h₂ : |a| = 1)
  (h₃ : |b| = 1)
  (h₄ : |c| = 1) :
  a * (b + c) = - (3 / 5) :=
sorry

end math_problem_l230_230352


namespace marked_cells_in_grid_l230_230176

theorem marked_cells_in_grid :
  ∀ (grid : Matrix (Fin 5) (Fin 5) Bool), 
  (∀ (i j : Fin 3), ∃! (a b : Fin 3), grid (i + a + 1) (j + b + 1) = true) → ∃ (n : ℕ), 1 ≤ n ∧ n ≤ 4 :=
by
  sorry

end marked_cells_in_grid_l230_230176


namespace age_difference_l230_230136

variables (X Y Z : ℕ)

theorem age_difference (h : X + Y = Y + Z + 12) : X - Z = 12 :=
sorry

end age_difference_l230_230136


namespace largest_integer_divides_difference_l230_230421

theorem largest_integer_divides_difference (n : ℕ) 
  (h_composite : ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ m * k = n) :
  6 ∣ (n^4 - n) :=
sorry

end largest_integer_divides_difference_l230_230421


namespace external_angle_bisector_proof_l230_230491

variables {A T C L K : Type} [Nonempty A] [Nonempty T] [Nonempty C] [Nonempty L] [Nonempty K]

noncomputable def angle_bisector_theorem (AL LC AB BC AK KC : ℝ) : Prop :=
(AL / LC) = (AB / BC) ∧ (AK / KC) = (AL / LC)

noncomputable def internal_angle_bisector (AT TC AL LC : ℝ) : Prop :=
(AT / TC) = (AL / LC)

noncomputable def external_angle_bisector (AT TC AK KC : ℝ) : Prop :=
(AT / TC) = (AK / KC)

theorem external_angle_bisector_proof (AL LC AB BC AK KC AT TC : ℝ) 
(h1 : angle_bisector_theorem AL LC AB BC AK KC)
(h2 : internal_angle_bisector AT TC AL LC) :
external_angle_bisector AT TC AK KC :=
sorry

end external_angle_bisector_proof_l230_230491


namespace combined_avg_score_l230_230559

noncomputable def classA_student_count := 45
noncomputable def classB_student_count := 55
noncomputable def classA_avg_score := 110
noncomputable def classB_avg_score := 90

theorem combined_avg_score (nA nB : ℕ) (avgA avgB : ℕ) 
  (h1 : nA = classA_student_count) 
  (h2 : nB = classB_student_count) 
  (h3 : avgA = classA_avg_score) 
  (h4 : avgB = classB_avg_score) : 
  (nA * avgA + nB * avgB) / (nA + nB) = 99 := 
by 
  rw [h1, h2, h3, h4]
  -- Substitute the values to get:
  -- (45 * 110 + 55 * 90) / (45 + 55) 
  -- = (4950 + 4950) / 100 
  -- = 9900 / 100 
  -- = 99
  sorry

end combined_avg_score_l230_230559


namespace largest_multiple_11_lt_neg85_l230_230894

-- Define the conditions: a multiple of 11 and smaller than -85
def largest_multiple_lt (m n : Int) : Int :=
  let k := (m / n) - 1
  n * k

-- Define our specific problem
theorem largest_multiple_11_lt_neg85 : largest_multiple_lt (-85) 11 = -88 := 
  by
  sorry

end largest_multiple_11_lt_neg85_l230_230894


namespace find_x_when_water_added_l230_230354

variable (m x : ℝ)

theorem find_x_when_water_added 
  (h1 : m > 25)
  (h2 : (m * m / 100) = ((m - 15) / 100) * (m + x)) :
  x = 15 * m / (m - 15) :=
sorry

end find_x_when_water_added_l230_230354


namespace sphere_volume_l230_230671

/-- A sphere is perfectly inscribed in a cube. 
If the edge of the cube measures 10 inches, the volume of the sphere in cubic inches is \(\frac{500}{3}\pi\). -/
theorem sphere_volume (a : ℝ) (h : a = 10) : 
  ∃ V : ℝ, V = (4 / 3) * Real.pi * (a / 2)^3 ∧ V = (500 / 3) * Real.pi :=
by
  use (4 / 3) * Real.pi * (a / 2)^3
  sorry

end sphere_volume_l230_230671


namespace combined_mpg_is_30_l230_230222

-- Define the constants
def ray_efficiency : ℕ := 50 -- miles per gallon
def tom_efficiency : ℕ := 25 -- miles per gallon
def ray_distance : ℕ := 100 -- miles
def tom_distance : ℕ := 200 -- miles

-- Define the combined miles per gallon calculation and the proof statement.
theorem combined_mpg_is_30 :
  (ray_distance + tom_distance) /
  ((ray_distance / ray_efficiency) + (tom_distance / tom_efficiency)) = 30 :=
by
  -- All proof steps are skipped using sorry
  sorry

end combined_mpg_is_30_l230_230222


namespace novels_in_shipment_l230_230195

theorem novels_in_shipment (N : ℕ) (H1: 225 = (3/4:ℚ) * N) : N = 300 := 
by
  sorry

end novels_in_shipment_l230_230195


namespace part1_part2_l230_230538

def my_mul (x y : Int) : Int :=
  if x = 0 then abs y
  else if y = 0 then abs x
  else if (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) then abs x + abs y
  else - (abs x + abs y)

theorem part1 : my_mul (-15) (my_mul 3 0) = -18 := 
  by
  sorry

theorem part2 (a : Int) : 
  my_mul 3 a + a = 
  if a < 0 then 2 * a - 3 
  else if a = 0 then 3
  else 2 * a + 3 :=
  by
  sorry

end part1_part2_l230_230538


namespace union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_in_U_l230_230635

open Set

noncomputable def U : Set ℤ := {x | -2 < x ∧ x < 2}
def A : Set ℤ := {x | x^2 - 5 * x - 6 = 0}
def B : Set ℤ := {x | x^2 = 1}

theorem union_of_A_and_B : A ∪ B = {-1, 1, 6} :=
by
  sorry

theorem intersection_of_A_and_B : A ∩ B = {-1} :=
by
  sorry

theorem complement_of_intersection_in_U : U \ (A ∩ B) = {0, 1} :=
by
  sorry

end union_of_A_and_B_intersection_of_A_and_B_complement_of_intersection_in_U_l230_230635


namespace units_digit_17_pow_28_l230_230578

theorem units_digit_17_pow_28 : (17 ^ 28) % 10 = 1 :=
by
  sorry

end units_digit_17_pow_28_l230_230578


namespace lowest_score_for_average_l230_230840

theorem lowest_score_for_average
  (score1 score2 score3 : ℕ)
  (h1 : score1 = 81)
  (h2 : score2 = 72)
  (h3 : score3 = 93)
  (max_score : ℕ := 100)
  (desired_average : ℕ := 86)
  (number_of_exams : ℕ := 5) :
  ∃ x y : ℕ, x ≤ 100 ∧ y ≤ 100 ∧ (score1 + score2 + score3 + x + y) / number_of_exams = desired_average ∧ min x y = 84 :=
by
  sorry

end lowest_score_for_average_l230_230840


namespace smallest_number_of_students_l230_230379

theorem smallest_number_of_students
    (g11 g10 g9 : Nat)
    (h_ratio1 : 4 * g9 = 3 * g11)
    (h_ratio2 : 6 * g10 = 5 * g11) :
  g11 + g10 + g9 = 31 :=
sorry

end smallest_number_of_students_l230_230379


namespace sum_of_sequence_l230_230405

theorem sum_of_sequence :
  3 + 15 + 27 + 53 + 65 + 17 + 29 + 41 + 71 + 83 = 404 :=
by
  sorry

end sum_of_sequence_l230_230405


namespace trapezium_area_l230_230328

theorem trapezium_area (a b h : ℝ) (ha : a = 24) (hb : b = 18) (hh : h = 15) : 
  1/2 * (a + b) * h = 315 ∧ h = 15 :=
by 
  -- The proof steps would go here
  sorry

end trapezium_area_l230_230328


namespace misread_signs_in_front_of_6_terms_l230_230124

/-- Define the polynomial function --/
def poly (x : ℝ) : ℝ :=
  10 * x ^ 9 + 9 * x ^ 8 + 8 * x ^ 7 + 7 * x ^ 6 + 6 * x ^ 5 + 5 * x ^ 4 + 4 * x ^ 3 + 3 * x ^ 2 + 2 * x + 1

/-- Xiao Ming's mistaken result --/
def mistaken_result : ℝ := 7

/-- Correct value of the expression at x = -1 --/
def correct_value : ℝ := poly (-1)

/-- The difference due to misreading signs --/
def difference : ℝ := mistaken_result - correct_value

/-- Prove that Xiao Ming misread the signs in front of 6 terms --/
theorem misread_signs_in_front_of_6_terms :
  difference / 2 = 6 :=
by
  simp [difference, correct_value, poly]
  -- the proof steps would go here
  sorry

#eval poly (-1)  -- to validate the correct value
#eval mistaken_result - poly (-1)  -- to validate the difference

end misread_signs_in_front_of_6_terms_l230_230124


namespace students_at_school_yy_l230_230183

theorem students_at_school_yy (X Y : ℝ) 
    (h1 : X + Y = 4000)
    (h2 : 0.07 * X - 0.03 * Y = 40) : 
    Y = 2400 :=
by
  sorry

end students_at_school_yy_l230_230183


namespace inverse_proportion_range_l230_230527

theorem inverse_proportion_range (m : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (y = (m + 5) / x) → ((x > 0 → y < 0) ∧ (x < 0 → y > 0))) →
  m < -5 :=
by
  intros h
  -- Skipping proof with sorry as specified
  sorry

end inverse_proportion_range_l230_230527


namespace find_B_current_age_l230_230439

variable {A B C : ℕ}

theorem find_B_current_age (h1 : A + 10 = 2 * (B - 10))
                          (h2 : A = B + 7)
                          (h3 : C = (A + B) / 2) :
                          B = 37 := by
  sorry

end find_B_current_age_l230_230439


namespace total_students_l230_230782

theorem total_students (students_per_classroom : ℕ) (num_classrooms : ℕ) (h1 : students_per_classroom = 30) (h2 : num_classrooms = 13) : students_per_classroom * num_classrooms = 390 :=
by
  -- Begin the proof
  sorry

end total_students_l230_230782


namespace min_squares_to_cover_5x5_l230_230608

theorem min_squares_to_cover_5x5 : 
  (∀ (cover : ℕ → ℕ), (cover 1 + cover 2 + cover 3 + cover 4) * (1^2 + 2^2 + 3^2 + 4^2) = 25 → 
  cover 1 + cover 2 + cover 3 + cover 4 = 10) :=
sorry

end min_squares_to_cover_5x5_l230_230608


namespace at_least_one_woman_probability_l230_230367

noncomputable def probability_at_least_one_woman_selected 
  (total_men : ℕ) (total_women : ℕ) (selected_people : ℕ) : ℚ :=
  1 - (8 / 12 * 7 / 11 * 6 / 10 * 5 / 9)

theorem at_least_one_woman_probability :
  probability_at_least_one_woman_selected 8 4 4 = 85 / 99 := 
sorry

end at_least_one_woman_probability_l230_230367


namespace difference_students_guinea_pigs_l230_230885

-- Define the conditions as constants
def students_per_classroom : Nat := 20
def guinea_pigs_per_classroom : Nat := 3
def number_of_classrooms : Nat := 6

-- Calculate the total number of students
def total_students : Nat := students_per_classroom * number_of_classrooms

-- Calculate the total number of guinea pigs
def total_guinea_pigs : Nat := guinea_pigs_per_classroom * number_of_classrooms

-- Define the theorem to prove the equality
theorem difference_students_guinea_pigs :
  total_students - total_guinea_pigs = 102 :=
by
  sorry -- Proof to be filled in

end difference_students_guinea_pigs_l230_230885


namespace total_money_l230_230205

theorem total_money (total_coins nickels dimes : ℕ) (val_nickel val_dime : ℕ)
  (h1 : total_coins = 8)
  (h2 : nickels = 2)
  (h3 : total_coins = nickels + dimes)
  (h4 : val_nickel = 5)
  (h5 : val_dime = 10) :
  (nickels * val_nickel + dimes * val_dime) = 70 :=
by
  sorry

end total_money_l230_230205


namespace mean_value_of_quadrilateral_angles_l230_230314

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end mean_value_of_quadrilateral_angles_l230_230314


namespace find_a5_find_a31_div_a29_l230_230589

noncomputable def geo_diff_seq (a : ℕ → ℕ) (d : ℕ) :=
∀ n : ℕ, n > 0 → (a (n + 2) / a (n + 1)) - (a (n + 1) / a n) = d

theorem find_a5 (a : ℕ → ℕ) (d : ℕ) (h_geo_diff : geo_diff_seq a d)
  (h_init : a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 3) : a 5 = 105 :=
sorry

theorem find_a31_div_a29 (a : ℕ → ℕ) (d : ℕ) (h_geo_diff : geo_diff_seq a d)
  (h_init : a 1 = 1 ∧ a 2 = 1 ∧ a 3 = 3) : a 31 / a 29 = 3363 :=
sorry

end find_a5_find_a31_div_a29_l230_230589


namespace largest_num_consecutive_integers_sum_45_l230_230639

theorem largest_num_consecutive_integers_sum_45 : 
  ∃ n : ℕ, (0 < n) ∧ (n * (n + 1) / 2 = 45) ∧ (∀ m : ℕ, (0 < m) → m * (m + 1) / 2 = 45 → m ≤ n) :=
by {
  sorry
}

end largest_num_consecutive_integers_sum_45_l230_230639


namespace inequality_solution_l230_230196

theorem inequality_solution (x : ℝ) (hx : 0 ≤ x ∧ x < 2) :
  ∀ y : ℝ, y > 0 → 4 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y) / (x + y) > 3 * x^2 * y :=
by
  intro y hy
  sorry

end inequality_solution_l230_230196


namespace ratio_of_boys_to_total_students_l230_230915

theorem ratio_of_boys_to_total_students
  (p : ℝ)
  (h : p = (3/4) * (1 - p)) :
  p = 3 / 7 :=
by
  sorry

end ratio_of_boys_to_total_students_l230_230915


namespace simplify_log_expression_l230_230072

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem simplify_log_expression :
  let term1 := 1 / (log_base 20 3 + 1)
  let term2 := 1 / (log_base 12 5 + 1)
  let term3 := 1 / (log_base 8 7 + 1)
  term1 + term2 + term3 = 2 :=
by
  sorry

end simplify_log_expression_l230_230072


namespace train_length_l230_230544

theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (h_speed : speed_kmh = 60) (h_time : time_s = 21) :
  (speed_kmh * (1000 / 3600) * time_s) = 350.07 := 
by
  sorry

end train_length_l230_230544


namespace area_of_triangle_l230_230427

theorem area_of_triangle (a b c : ℝ) (A B C : ℝ) (h₁ : b = 2) (h₂ : c = 2 * Real.sqrt 2) (h₃ : C = Real.pi / 4) :
  1 / 2 * b * c * Real.sin (Real.pi - B - C) = Real.sqrt 3 + 1 := 
by
  sorry

end area_of_triangle_l230_230427


namespace feifei_sheep_count_l230_230956

noncomputable def sheep_number (x y : ℕ) : Prop :=
  (y = 3 * x + 15) ∧ (x = y - y / 3)

theorem feifei_sheep_count :
  ∃ x y : ℕ, sheep_number x y ∧ x = 5 :=
sorry

end feifei_sheep_count_l230_230956


namespace number_of_suits_sold_l230_230445

theorem number_of_suits_sold
  (commission_rate: ℝ)
  (price_per_suit: ℝ)
  (price_per_shirt: ℝ)
  (price_per_loafer: ℝ)
  (number_of_shirts: ℕ)
  (number_of_loafers: ℕ)
  (total_commission: ℝ)
  (suits_sold: ℕ)
  (total_sales: ℝ)
  (total_sales_from_non_suits: ℝ)
  (sales_needed_from_suits: ℝ)
  : 
  (commission_rate = 0.15) → 
  (price_per_suit = 700.0) → 
  (price_per_shirt = 50.0) → 
  (price_per_loafer = 150.0) → 
  (number_of_shirts = 6) → 
  (number_of_loafers = 2) → 
  (total_commission = 300.0) →
  (total_sales = total_commission / commission_rate) →
  (total_sales_from_non_suits = number_of_shirts * price_per_shirt + number_of_loafers * price_per_loafer) →
  (sales_needed_from_suits = total_sales - total_sales_from_non_suits) →
  (suits_sold = sales_needed_from_suits / price_per_suit) →
  suits_sold = 2 :=
by
  sorry

end number_of_suits_sold_l230_230445


namespace arithmetic_sequence_sum_l230_230010

theorem arithmetic_sequence_sum : 
  ∃ x y, (∃ d, 
  d = 12 - 5 ∧ 
  19 + d = x ∧ 
  x + d = y ∧ 
  y + d = 40 ∧ 
  x + y = 59) :=
by {
  sorry
}

end arithmetic_sequence_sum_l230_230010


namespace part1_part2_l230_230742

def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + a

theorem part1 (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
by
  sorry

theorem part2 (a x : ℝ) (h : a ≠ -3) :
  (f x a > 4 * a - (a + 3) * x) ↔ 
  ((a > -3 ∧ (x < -3 ∨ x > a)) ∨ (a < -3 ∧ (x < a ∨ x > -3))) :=
by
  sorry

end part1_part2_l230_230742


namespace terminating_decimal_l230_230747

-- Define the given fraction
def frac : ℚ := 21 / 160

-- Define the decimal representation
def dec : ℚ := 13125 / 100000

-- State the theorem to be proved
theorem terminating_decimal : frac = dec := by
  sorry

end terminating_decimal_l230_230747


namespace fraction_of_rectangle_shaded_l230_230031

theorem fraction_of_rectangle_shaded
  (length : ℕ) (width : ℕ)
  (one_third_part : ℕ) (half_of_third : ℕ)
  (H1 : length = 10) (H2 : width = 15)
  (H3 : one_third_part = (1/3 : ℝ) * (length * width)) 
  (H4 : half_of_third = (1/2 : ℝ) * one_third_part) :
  (half_of_third / (length * width) = 1/6) :=
sorry

end fraction_of_rectangle_shaded_l230_230031


namespace Anya_walks_to_school_l230_230253

theorem Anya_walks_to_school
  (t_f t_b : ℝ)
  (h1 : t_f + t_b = 1.5)
  (h2 : 2 * t_b = 0.5) :
  2 * t_f = 2.5 :=
by
  -- The proof details will go here eventually.
  sorry

end Anya_walks_to_school_l230_230253


namespace problem_equivalent_final_answer_l230_230958

noncomputable def a := 12
noncomputable def b := 27
noncomputable def c := 6

theorem problem_equivalent :
  2 * Real.sqrt 3 + (2 / Real.sqrt 3) + 3 * Real.sqrt 2 + (3 / Real.sqrt 2) = (a * Real.sqrt 3 + b * Real.sqrt 2) / c :=
  sorry

theorem final_answer :
  a + b + c = 45 :=
  by
    unfold a b c
    simp
    done

end problem_equivalent_final_answer_l230_230958


namespace polygon_sides_arithmetic_progression_l230_230682

theorem polygon_sides_arithmetic_progression
  (n : ℕ)
  (h1 : ∀ i, 1 ≤ i → i ≤ n → 172 - (i - 1) * 8 > 0) -- Each angle in the sequence is positive
  (h2 : (∀ i, 1 ≤ i → i ≤ n → (172 - (i - 1) * 8) < 180)) -- Each angle < 180 degrees
  (h3 : n * (172 - (n-1) * 4) = 180 * (n - 2)) -- Sum of interior angles formula
  : n = 10 :=
sorry

end polygon_sides_arithmetic_progression_l230_230682


namespace eliminate_denominators_l230_230237

theorem eliminate_denominators (x : ℝ) :
  (6 : ℝ) * ((x - 1) / 3) = (6 : ℝ) * (4 - (2 * x + 1) / 2) ↔ 2 * (x - 1) = 24 - 3 * (2 * x + 1) :=
by
  intros
  sorry

end eliminate_denominators_l230_230237


namespace triangle_tangent_half_angle_l230_230208

theorem triangle_tangent_half_angle (a b c : ℝ) (A : ℝ) (C : ℝ)
  (h : a + c = 2 * b) :
  Real.tan (A / 2) * Real.tan (C / 2) = 1 / 3 := 
sorry

end triangle_tangent_half_angle_l230_230208


namespace routes_from_A_to_B_l230_230868

-- Definitions based on conditions given in the problem
variables (A B C D E F : Type)
variables (AB AD AE BC BD CD DE EF : Prop) 

-- Theorem statement
theorem routes_from_A_to_B (route_criteria : AB ∧ AD ∧ AE ∧ BC ∧ BD ∧ CD ∧ DE ∧ EF)
  : ∃ n : ℕ, n = 16 :=
sorry

end routes_from_A_to_B_l230_230868


namespace team_A_more_points_than_team_B_l230_230007

theorem team_A_more_points_than_team_B :
  let number_of_teams := 8
  let number_of_remaining_games := 6
  let win_probability_each_game := (1 : ℚ) / 2
  let team_A_beats_team_B_initial : Prop := True -- Corresponding to the condition team A wins the first game
  let probability_A_wins := 1087 / 2048
  team_A_beats_team_B_initial → win_probability_each_game = 1 / 2 → number_of_teams = 8 → 
    let A_more_points_than_B := team_A_beats_team_B_initial ∧ win_probability_each_game ^ number_of_remaining_games = probability_A_wins
    A_more_points_than_B :=
  sorry

end team_A_more_points_than_team_B_l230_230007


namespace sequence_value_l230_230805

theorem sequence_value : 
  ∃ (x y r : ℝ), 
    (4096 * r = 1024) ∧ 
    (1024 * r = 256) ∧ 
    (256 * r = x) ∧ 
    (x * r = y) ∧ 
    (y * r = 4) ∧  
    (4 * r = 1) ∧ 
    (x + y = 80) :=
by
  sorry

end sequence_value_l230_230805


namespace park_length_l230_230186

theorem park_length (width : ℕ) (trees_per_sqft : ℕ) (num_trees : ℕ) (total_area : ℕ) (length : ℕ)
  (hw : width = 2000)
  (ht : trees_per_sqft = 20)
  (hn : num_trees = 100000)
  (ha : total_area = num_trees * trees_per_sqft)
  (hl : length = total_area / width) :
  length = 1000 :=
by
  sorry

end park_length_l230_230186


namespace range_of_a_l230_230075

noncomputable def S : Set ℝ := {x | |x - 1| + |x + 2| > 5}
noncomputable def T (a : ℝ) : Set ℝ := {x | |x - a| ≤ 4}

theorem range_of_a (a : ℝ) : 
  (S ∪ T a) = Set.univ ↔ -2 ≤ a ∧ a ≤ 1 := 
by 
  sorry

end range_of_a_l230_230075


namespace largest_lcm_value_is_90_l230_230711

def lcm_vals (a b : ℕ) : ℕ := Nat.lcm a b

theorem largest_lcm_value_is_90 :
  max (lcm_vals 18 3)
      (max (lcm_vals 18 9)
           (max (lcm_vals 18 6)
                (max (lcm_vals 18 12)
                     (max (lcm_vals 18 15)
                          (lcm_vals 18 18))))) = 90 :=
by
  -- Use the fact that the calculations of LCMs are as follows:
  -- lcm(18, 3) = 18
  -- lcm(18, 9) = 18
  -- lcm(18, 6) = 18
  -- lcm(18, 12) = 36
  -- lcm(18, 15) = 90
  -- lcm(18, 18) = 18
  -- therefore, the largest value among these is 90
  sorry

end largest_lcm_value_is_90_l230_230711


namespace number_of_cats_l230_230362

theorem number_of_cats (total_animals : ℕ) (dogs : ℕ) (cats : ℕ) 
  (h1 : total_animals = 1212) 
  (h2 : dogs = 567) 
  (h3 : cats = total_animals - dogs) : 
  cats = 645 := 
by 
  sorry

end number_of_cats_l230_230362


namespace parts_purchased_l230_230581

noncomputable def price_per_part : ℕ := 80
noncomputable def total_paid_after_discount : ℕ := 439
noncomputable def total_discount : ℕ := 121

theorem parts_purchased : 
  ∃ n : ℕ, price_per_part * n - total_discount = total_paid_after_discount → n = 7 :=
by
  sorry

end parts_purchased_l230_230581


namespace range_of_function_l230_230808

theorem range_of_function :
  ∀ y : ℝ, (∃ x : ℝ, y = (1 / 2) ^ (x^2 + 2 * x - 1)) ↔ (0 < y ∧ y ≤ 4) :=
by
  sorry

end range_of_function_l230_230808


namespace geometric_seq_a4_l230_230688

variable {a : ℕ → ℝ}

-- Definition: a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Condition
axiom h : a 2 * a 6 = 4

-- Theorem that needs to be proved
theorem geometric_seq_a4 (h_seq: is_geometric_sequence a) (h: a 2 * a 6 = 4) : a 4 = 2 ∨ a 4 = -2 := by
  sorry

end geometric_seq_a4_l230_230688


namespace proof_firstExpr_proof_secondExpr_l230_230692

noncomputable def firstExpr : ℝ :=
  Real.logb 2 (Real.sqrt (7 / 48)) + Real.logb 2 12 - (1 / 2) * Real.logb 2 42 - 1

theorem proof_firstExpr :
  firstExpr = -3 / 2 :=
by
  sorry

noncomputable def secondExpr : ℝ :=
  (Real.logb 10 2) ^ 2 + Real.logb 10 (2 * Real.logb 10 50 + Real.logb 10 25)

theorem proof_secondExpr :
  secondExpr = 0.0906 + Real.logb 10 5.004 :=
by
  sorry

end proof_firstExpr_proof_secondExpr_l230_230692


namespace time_distribution_l230_230360

noncomputable def total_hours_at_work (hours_task1 day : ℕ) (hours_task2 day : ℕ) (work_days : ℕ) (reduce_per_week : ℕ) : ℕ :=
  (hours_task1 + hours_task2) * work_days

theorem time_distribution (h1 : 5 = 5) (h2 : 3 = 3) (days : 5 = 5) (reduction : 5 = 5) :
  total_hours_at_work 5 3 5 5 = 40 :=
by
  sorry

end time_distribution_l230_230360


namespace total_red_cards_l230_230416

def num_standard_decks : ℕ := 3
def num_special_decks : ℕ := 2
def num_custom_decks : ℕ := 2
def red_cards_standard_deck : ℕ := 26
def red_cards_special_deck : ℕ := 30
def red_cards_custom_deck : ℕ := 20

theorem total_red_cards : num_standard_decks * red_cards_standard_deck +
                          num_special_decks * red_cards_special_deck +
                          num_custom_decks * red_cards_custom_deck = 178 :=
by
  -- Calculation omitted
  sorry

end total_red_cards_l230_230416


namespace find_a_l230_230504

theorem find_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x = 0 ∧ x = 1) → a = -1 := by
  intro h
  obtain ⟨x, hx, rfl⟩ := h
  have H : 1^2 + a * 1 = 0 := hx
  linarith

end find_a_l230_230504


namespace inequality_inequality_l230_230375

theorem inequality_inequality (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) :
  (a - b) ^ 2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧
  (a + b) / 2 - Real.sqrt (a * b) < (a - b) ^ 2 / (8 * b) :=
sorry

end inequality_inequality_l230_230375


namespace families_received_boxes_l230_230935

theorem families_received_boxes (F : ℕ) (box_decorations total_decorations : ℕ)
  (h_box_decorations : box_decorations = 10)
  (h_total_decorations : total_decorations = 120)
  (h_eq : box_decorations * (F + 1) = total_decorations) :
  F = 11 :=
by
  sorry

end families_received_boxes_l230_230935


namespace decimal_equivalent_l230_230118

theorem decimal_equivalent (x : ℚ) (h : x = 16 / 50) : x = 32 / 100 :=
by
  sorry

end decimal_equivalent_l230_230118


namespace annual_return_percentage_l230_230757

theorem annual_return_percentage (initial_value final_value gain : ℕ)
    (h1 : initial_value = 8000)
    (h2 : final_value = initial_value + 400)
    (h3 : gain = final_value - initial_value) :
    (gain * 100 / initial_value) = 5 := by
  sorry

end annual_return_percentage_l230_230757


namespace sales_in_fourth_month_l230_230745

theorem sales_in_fourth_month (sale_m1 sale_m2 sale_m3 sale_m5 sale_m6 avg_sales total_months : ℕ)
    (H1 : sale_m1 = 7435) (H2 : sale_m2 = 7927) (H3 : sale_m3 = 7855) 
    (H4 : sale_m5 = 7562) (H5 : sale_m6 = 5991) (H6 : avg_sales = 7500) (H7 : total_months = 6) :
    ∃ sale_m4 : ℕ, sale_m4 = 8230 := by
  sorry

end sales_in_fourth_month_l230_230745


namespace hoseok_basketballs_l230_230462

theorem hoseok_basketballs (v s b : ℕ) (h₁ : v = 40) (h₂ : s = v + 18) (h₃ : b = s - 23) : b = 35 := by
  sorry

end hoseok_basketballs_l230_230462
