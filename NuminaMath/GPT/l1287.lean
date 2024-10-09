import Mathlib

namespace negation_of_proposition_l1287_128777

theorem negation_of_proposition:
  (¬ ∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2 * x + 1 > 0) :=
by
  sorry

end negation_of_proposition_l1287_128777


namespace number_divided_by_189_l1287_128724

noncomputable def target_number : ℝ := 3486

theorem number_divided_by_189 :
  target_number / 189 = 18.444444444444443 :=
by
  sorry

end number_divided_by_189_l1287_128724


namespace new_train_distance_l1287_128769

-- Define the given conditions
def distance_old : ℝ := 300
def percentage_increase : ℝ := 0.3

-- Define the target distance to prove
def distance_new : ℝ := distance_old + (percentage_increase * distance_old)

-- State the theorem
theorem new_train_distance : distance_new = 390 := by
  sorry

end new_train_distance_l1287_128769


namespace problem_statement_l1287_128723

def a := 596
def b := 130
def c := 270

theorem problem_statement : a - b - c = a - (b + c) := by
  sorry

end problem_statement_l1287_128723


namespace width_of_wide_flags_l1287_128718

def total_fabric : ℕ := 1000
def leftover_fabric : ℕ := 294
def num_square_flags : ℕ := 16
def square_flag_area : ℕ := 16
def num_tall_flags : ℕ := 10
def tall_flag_area : ℕ := 15
def num_wide_flags : ℕ := 20
def wide_flag_height : ℕ := 3

theorem width_of_wide_flags :
  (total_fabric - leftover_fabric - (num_square_flags * square_flag_area + num_tall_flags * tall_flag_area)) / num_wide_flags / wide_flag_height = 5 :=
by
  sorry

end width_of_wide_flags_l1287_128718


namespace sara_jim_savings_eq_l1287_128789

theorem sara_jim_savings_eq (w : ℕ) : 
  let sara_init_savings := 4100
  let sara_weekly_savings := 10
  let jim_weekly_savings := 15
  (sara_init_savings + sara_weekly_savings * w = jim_weekly_savings * w) → w = 820 :=
by
  intros
  sorry

end sara_jim_savings_eq_l1287_128789


namespace value_of_x_l1287_128797

theorem value_of_x (x : ℝ) (h1 : (x^2 - 4) / (x + 2) = 0) : x = 2 := by
  sorry

end value_of_x_l1287_128797


namespace part_to_third_fraction_is_six_five_l1287_128762

noncomputable def ratio_of_part_to_third_fraction (P N : ℝ) (h1 : (1/4) * (1/3) * P = 20) (h2 : 0.40 * N = 240) : ℝ :=
  P / (N / 3)

theorem part_to_third_fraction_is_six_five (P N : ℝ) (h1 : (1/4) * (1/3) * P = 20) (h2 : 0.40 * N = 240) : ratio_of_part_to_third_fraction P N h1 h2 = 6 / 5 :=
  sorry

end part_to_third_fraction_is_six_five_l1287_128762


namespace skateboarded_one_way_distance_l1287_128760

-- Define the total skateboarded distance and the walked distance.
def total_skateboarded : ℕ := 24
def walked_distance : ℕ := 4

-- Define the proof theorem.
theorem skateboarded_one_way_distance : 
    (total_skateboarded - walked_distance) / 2 = 10 := 
by sorry

end skateboarded_one_way_distance_l1287_128760


namespace twelve_star_three_eq_four_star_eight_eq_star_assoc_l1287_128748

def star (a b : ℕ) : ℕ := 10^a * 10^b

theorem twelve_star_three_eq : star 12 3 = 10^15 :=
by 
  -- Proof here
  sorry

theorem four_star_eight_eq : star 4 8 = 10^12 :=
by 
  -- Proof here
  sorry

theorem star_assoc (a b c : ℕ) : star (a + b) c = star a (b + c) :=
by 
  -- Proof here
  sorry

end twelve_star_three_eq_four_star_eight_eq_star_assoc_l1287_128748


namespace broken_line_coverable_l1287_128759

noncomputable def cover_broken_line (length_of_line : ℝ) (radius_of_circle : ℝ) : Prop :=
  length_of_line = 5 ∧ radius_of_circle > 1.25

theorem broken_line_coverable :
  ∃ radius_of_circle, cover_broken_line 5 radius_of_circle :=
by sorry

end broken_line_coverable_l1287_128759


namespace derivative_at_zero_l1287_128722

-- Given conditions
def f (x : ℝ) : ℝ := -2 * x^2 + 3

-- Theorem statement to prove
theorem derivative_at_zero : 
  deriv f 0 = 0 := 
by 
  sorry

end derivative_at_zero_l1287_128722


namespace polygon_sides_l1287_128784

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 2 * 360) : n = 6 :=
sorry

end polygon_sides_l1287_128784


namespace zoe_spent_amount_l1287_128774

def flower_price : ℕ := 3
def roses_bought : ℕ := 8
def daisies_bought : ℕ := 2

theorem zoe_spent_amount :
  roses_bought + daisies_bought = 10 ∧
  flower_price = 3 →
  (roses_bought + daisies_bought) * flower_price = 30 :=
by
  sorry

end zoe_spent_amount_l1287_128774


namespace andrea_avg_km_per_day_l1287_128750

theorem andrea_avg_km_per_day
  (total_distance : ℕ := 168)
  (total_days : ℕ := 6)
  (completed_fraction : ℚ := 3/7)
  (completed_days : ℕ := 3) :
  (total_distance * (1 - completed_fraction)) / (total_days - completed_days) = 32 := 
sorry

end andrea_avg_km_per_day_l1287_128750


namespace jackson_total_calories_l1287_128752

def lettuce_calories : ℕ := 50
def carrots_calories : ℕ := 2 * lettuce_calories
def dressing_calories : ℕ := 210
def salad_calories : ℕ := lettuce_calories + carrots_calories + dressing_calories

def crust_calories : ℕ := 600
def pepperoni_calories : ℕ := crust_calories / 3
def cheese_calories : ℕ := 400
def pizza_calories : ℕ := crust_calories + pepperoni_calories + cheese_calories

def jackson_salad_fraction : ℚ := 1 / 4
def jackson_pizza_fraction : ℚ := 1 / 5

noncomputable def total_calories : ℚ := 
  jackson_salad_fraction * salad_calories + jackson_pizza_fraction * pizza_calories

theorem jackson_total_calories : total_calories = 330 := by
  sorry

end jackson_total_calories_l1287_128752


namespace projectile_highest_point_l1287_128764

noncomputable def highest_point (v w_h w_v θ g : ℝ) : ℝ × ℝ :=
  let t := (v * Real.sin θ + w_v) / g
  let x := (v * t + w_h * t) * Real.cos θ
  let y := (v * t + w_v * t) * Real.sin θ - (1/2) * g * t^2
  (x, y)

theorem projectile_highest_point : highest_point 100 10 (-2) (Real.pi / 4) 9.8 = (561.94, 236) :=
  sorry

end projectile_highest_point_l1287_128764


namespace sum_of_numbers_is_60_l1287_128751

-- Define the primary values used in the conditions
variables (a b c : ℝ)

-- Define the conditions in the problem
def mean_condition_1 : Prop := (a + b + c) / 3 = a + 20
def mean_condition_2 : Prop := (a + b + c) / 3 = c - 30
def median_condition : Prop := b = 10

-- Prove that the sum of the numbers is 60 given the conditions
theorem sum_of_numbers_is_60 (hac1 : mean_condition_1 a b c) (hac2 : mean_condition_2 a b c) (hbm : median_condition b) : a + b + c = 60 :=
by 
  sorry

end sum_of_numbers_is_60_l1287_128751


namespace sqrt2_over_2_not_covered_by_rationals_l1287_128726

noncomputable def rational_not_cover_sqrt2_over_2 : Prop :=
  ∀ (a b : ℤ) (h_ab : Int.gcd a b = 1) (h_b_pos : b > 0)
  (h_frac : (a : ℚ) / b ∈ Set.Ioo 0 1),
  abs ((Real.sqrt 2) / 2 - (a : ℚ) / b) > 1 / (4 * b^2)

-- Placeholder for the proof
theorem sqrt2_over_2_not_covered_by_rationals :
  rational_not_cover_sqrt2_over_2 := 
by sorry

end sqrt2_over_2_not_covered_by_rationals_l1287_128726


namespace ratio_surface_area_l1287_128749

open Real

theorem ratio_surface_area (R a : ℝ) 
  (h1 : 4 * R^2 = 6 * a^2) 
  (H : R = (sqrt 6 / 2) * a) : 
  3 * π * R^2 / (6 * a^2) = 3 * π / 4 :=
by {
  sorry
}

end ratio_surface_area_l1287_128749


namespace opposite_of_neg5_is_pos5_l1287_128710

theorem opposite_of_neg5_is_pos5 : -(-5) = 5 := 
by
  sorry

end opposite_of_neg5_is_pos5_l1287_128710


namespace functional_expression_y_x_maximize_profit_price_reduction_and_profit_l1287_128746

-- Define the conditions
variable (C_selling C_cost : ℝ := 80) (C_costComponent : ℝ := 30) (initialSales : ℝ := 600) 
variable (dec_price : ℝ := 2) (inc_sales : ℝ := 30)
variable (decrease x : ℝ)

-- Define and prove part 1: Functional expression between y and x
theorem functional_expression_y_x : (decrease : ℝ) → (15 * decrease + initialSales : ℝ) = (inc_sales / dec_price * decrease + initialSales) :=
by sorry

-- Define the function for weekly profit
def weekly_profit (x : ℝ) : ℝ := 
  let selling_price := C_selling - x
  let cost_price := C_costComponent
  let sales_volume := 15 * x + initialSales
  (selling_price - cost_price) * sales_volume

-- Prove the condition for maximizing weekly sales profit
theorem maximize_profit_price_reduction_and_profit : 
  (∀ x : ℤ, x % 2 = 0 → weekly_profit x ≤ 30360) ∧
  weekly_profit 4 = 30360 ∧ 
  weekly_profit 6 = 30360 :=
by sorry

end functional_expression_y_x_maximize_profit_price_reduction_and_profit_l1287_128746


namespace find_value_l1287_128721

-- Defining the known conditions
def number : ℕ := 20
def half (n : ℕ) : ℕ := n / 2
def value_added (V : ℕ) : Prop := half number + V = 17

-- Proving that the value added to half the number is 7
theorem find_value : value_added 7 :=
by
  -- providing the proof for the theorem
  -- skipping the proof steps with sorry
  sorry

end find_value_l1287_128721


namespace lollipops_initial_count_l1287_128736

theorem lollipops_initial_count (L : ℕ) (k : ℕ) 
  (h1 : L % 42 ≠ 0) 
  (h2 : (L + 22) % 42 = 0) : 
  L = 62 :=
by
  sorry

end lollipops_initial_count_l1287_128736


namespace div_c_a_l1287_128732

theorem div_c_a (a b c : ℚ) (h1 : a = 3 * b) (h2 : b = 2 / 5 * c) :
  c / a = 5 / 6 := 
by
  sorry

end div_c_a_l1287_128732


namespace sam_total_coins_l1287_128734

theorem sam_total_coins (nickel_count : ℕ) (dime_count : ℕ) (total_value_cents : ℤ) (nickel_value : ℤ) (dime_value : ℤ)
  (h₁ : nickel_count = 12)
  (h₂ : total_value_cents = 240)
  (h₃ : nickel_value = 5)
  (h₄ : dime_value = 10)
  (h₅ : nickel_count * nickel_value + dime_count * dime_value = total_value_cents) :
  nickel_count + dime_count = 30 := 
  sorry

end sam_total_coins_l1287_128734


namespace altitudes_sum_of_triangle_formed_by_line_and_axes_l1287_128780

noncomputable def sum_of_altitudes (x y : ℝ) : ℝ :=
  let intercept_x := 6
  let intercept_y := 16
  let altitude_3 := 48 / Real.sqrt (8^2 + 3^2)
  intercept_x + intercept_y + altitude_3

theorem altitudes_sum_of_triangle_formed_by_line_and_axes :
  ∀ (x y : ℝ), (8 * x + 3 * y = 48) →
  sum_of_altitudes x y = 22 + 48 / Real.sqrt 73 :=
by
  sorry

end altitudes_sum_of_triangle_formed_by_line_and_axes_l1287_128780


namespace day_after_2_pow_20_is_friday_l1287_128725

-- Define the given conditions
def today_is_monday : ℕ := 0 -- Assuming Monday is represented by 0

-- Define the number of days after \(2^{20}\) days
def days_after : ℕ := 2^20

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the function to find the day of the week after a given number of days
def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed) % days_in_week

-- The theorem to prove
theorem day_after_2_pow_20_is_friday :
  day_of_week today_is_monday days_after = 5 := -- Friday is represented by 5 here
sorry

end day_after_2_pow_20_is_friday_l1287_128725


namespace female_students_in_sample_l1287_128707

-- Definitions of the given conditions
def male_students : ℕ := 28
def female_students : ℕ := 21
def total_students : ℕ := male_students + female_students
def sample_size : ℕ := 14
def stratified_sampling_fraction : ℚ := (sample_size : ℚ) / (total_students : ℚ)
def female_sample_count : ℚ := stratified_sampling_fraction * (female_students : ℚ)

-- The theorem to prove
theorem female_students_in_sample : female_sample_count = 6 :=
by
  sorry

end female_students_in_sample_l1287_128707


namespace integral_right_angled_triangles_unique_l1287_128747

theorem integral_right_angled_triangles_unique : 
  ∀ a b c : ℤ, (a < b) ∧ (b < c) ∧ (a^2 + b^2 = c^2) ∧ (a * b = 4 * (a + b + c))
  ↔ (a = 10 ∧ b = 24 ∧ c = 26)
  ∨ (a = 12 ∧ b = 16 ∧ c = 20)
  ∨ (a = 9 ∧ b = 40 ∧ c = 41) :=
by {
  sorry
}

end integral_right_angled_triangles_unique_l1287_128747


namespace fg_of_neg2_l1287_128728

def f (x : ℤ) : ℤ := x^2 + 4
def g (x : ℤ) : ℤ := 3 * x + 2

theorem fg_of_neg2 : f (g (-2)) = 20 := by
  sorry

end fg_of_neg2_l1287_128728


namespace find_number_divided_l1287_128715

theorem find_number_divided (n : ℕ) (h : n = 21 * 9 + 1) : n = 190 :=
by
  sorry

end find_number_divided_l1287_128715


namespace product_mod_32_l1287_128719

def product_of_all_odd_primes_less_than_32 : ℕ :=
  3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

theorem product_mod_32 :
  (product_of_all_odd_primes_less_than_32) % 32 = 9 :=
sorry

end product_mod_32_l1287_128719


namespace total_tiles_to_be_replaced_l1287_128782

-- Define the given conditions
def horizontal_paths : List ℕ := [30, 50, 30, 20, 20, 50]
def vertical_paths : List ℕ := [20, 50, 20, 50, 50]
def intersections : ℕ := List.sum [2, 3, 3, 4, 4]

-- Problem statement: Prove that the total number of tiles to be replaced is 374
theorem total_tiles_to_be_replaced : List.sum horizontal_paths + List.sum vertical_paths - intersections = 374 := 
by sorry

end total_tiles_to_be_replaced_l1287_128782


namespace no_such_integers_l1287_128713

theorem no_such_integers (x y : ℤ) : ¬ ∃ x y : ℤ, (x^4 + 6) % 13 = y^3 % 13 :=
sorry

end no_such_integers_l1287_128713


namespace characterize_functions_l1287_128705

open Function

noncomputable def f : ℚ → ℚ := sorry
noncomputable def g : ℚ → ℚ := sorry

axiom f_g_condition_1 : ∀ x y : ℚ, f (g (x) - g (y)) = f (g (x)) - y
axiom f_g_condition_2 : ∀ x y : ℚ, g (f (x) - f (y)) = g (f (x)) - y

theorem characterize_functions : 
  (∃ c : ℚ, ∀ x, f x = c * x) ∧ (∃ c : ℚ, ∀ x, g x = x / c) := 
sorry

end characterize_functions_l1287_128705


namespace computation_l1287_128793

def g (x : ℕ) : ℕ := 7 * x - 3

theorem computation : g (g (g (g 1))) = 1201 := by
  sorry

end computation_l1287_128793


namespace find_original_number_l1287_128727

-- Defining the conditions as given in the problem
def original_number_condition (x : ℤ) : Prop :=
  3 * (3 * x - 6) = 141

-- Stating the main theorem to be proven
theorem find_original_number (x : ℤ) (h : original_number_condition x) : x = 17 :=
sorry

end find_original_number_l1287_128727


namespace general_formula_l1287_128753

def a (n : ℕ) : ℕ :=
match n with
| 0 => 1
| k+1 => 2 * a k + 4

theorem general_formula (n : ℕ) : a (n+1) = 5 * 2^n - 4 :=
by
  sorry

end general_formula_l1287_128753


namespace chess_team_boys_count_l1287_128712

theorem chess_team_boys_count : 
  ∃ (B G : ℕ), B + G = 30 ∧ (2 / 3 : ℚ) * G + B = 18 ∧ B = 6 := by
  sorry

end chess_team_boys_count_l1287_128712


namespace solve_for_x_l1287_128765

theorem solve_for_x (x : ℝ) (h : (3 + 2 / x)^(1 / 3) = 2) : x = 2 / 5 :=
by
  sorry

end solve_for_x_l1287_128765


namespace find_n_eq_130_l1287_128791

theorem find_n_eq_130 
  (n : ℕ)
  (d1 d2 d3 d4 : ℕ)
  (h1 : 0 < n)
  (h2 : d1 < d2)
  (h3 : d2 < d3)
  (h4 : d3 < d4)
  (h5 : ∀ d, d ∣ n → d = d1 ∨ d = d2 ∨ d = d3 ∨ d = d4 ∨ d ∣ n → ¬(1 < d ∧ d < d1))
  (h6 : n = d1^2 + d2^2 + d3^2 + d4^2) : n = 130 := 
  sorry

end find_n_eq_130_l1287_128791


namespace sheets_of_paper_in_each_box_l1287_128733

theorem sheets_of_paper_in_each_box (E S : ℕ) (h1 : 2 * E + 40 = S) (h2 : 4 * (E - 40) = S) : S = 240 :=
by
  sorry

end sheets_of_paper_in_each_box_l1287_128733


namespace Annette_more_than_Sara_l1287_128735

variable (A C S : ℕ)

-- Define the given conditions as hypotheses
def Annette_Caitlin_weight : Prop := A + C = 95
def Caitlin_Sara_weight : Prop := C + S = 87

-- The theorem to prove: Annette weighs 8 pounds more than Sara
theorem Annette_more_than_Sara (h1 : Annette_Caitlin_weight A C)
                               (h2 : Caitlin_Sara_weight C S) :
  A - S = 8 := by
  sorry

end Annette_more_than_Sara_l1287_128735


namespace minimum_groups_l1287_128703

theorem minimum_groups (total_players : ℕ) (max_per_group : ℕ)
  (h_total : total_players = 30)
  (h_max : max_per_group = 12) :
  ∃ x y, y ∣ total_players ∧ y ≤ max_per_group ∧ total_players / y = x ∧ x = 3 :=
by {
  sorry
}

end minimum_groups_l1287_128703


namespace eval_expression_l1287_128755

theorem eval_expression : 6 + 15 / 3 - 4^2 + 1 = -4 := by
  sorry

end eval_expression_l1287_128755


namespace weight_of_seventh_person_l1287_128739

noncomputable def weight_of_six_people : ℕ := 6 * 156
noncomputable def new_average_weight (x : ℕ) : Prop := (weight_of_six_people + x) / 7 = 151

theorem weight_of_seventh_person (x : ℕ) (h : new_average_weight x) : x = 121 :=
by
  sorry

end weight_of_seventh_person_l1287_128739


namespace lorraine_initial_brownies_l1287_128731

theorem lorraine_initial_brownies (B : ℝ) 
(h1: (0.375 * B - 1 = 5)) : B = 16 := 
sorry

end lorraine_initial_brownies_l1287_128731


namespace no_such_natural_numbers_exist_l1287_128701

theorem no_such_natural_numbers_exist :
  ¬ ∃ (x y : ℕ), ∃ (k m : ℕ), x^2 + x + 1 = y^k ∧ y^2 + y + 1 = x^m := 
by sorry

end no_such_natural_numbers_exist_l1287_128701


namespace domain_f_monotonicity_f_inequality_solution_l1287_128720

noncomputable def f (x: ℝ) := Real.log ((1 - x) / (1 + x))

variable {x : ℝ}

theorem domain_f : ∀ x, x ∈ Set.Ioo (-1 : ℝ) 1 -> Set.Ioo (-1 : ℝ) 1 := sorry

theorem monotonicity_f : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, ∀ y ∈ Set.Ioo (-1 : ℝ) 1, x < y → f y < f x := sorry

theorem inequality_solution :
  {x : ℝ | f (2 * x - 1) < 0} = {x | x > 1 / 2 ∧ x < 1} := sorry

end domain_f_monotonicity_f_inequality_solution_l1287_128720


namespace digits_conditions_l1287_128711

noncomputable def original_number : ℕ := 253
noncomputable def reversed_number : ℕ := 352

theorem digits_conditions (a b c : ℕ) : 
  a + b + c = 10 → 
  b = a + c → 
  (original_number = a * 100 + b * 10 + c) → 
  (reversed_number = c * 100 + b * 10 + a) → 
  reversed_number - original_number = 99 :=
by
  intros h1 h2 h3 h4
  sorry

end digits_conditions_l1287_128711


namespace percentage_profit_double_price_l1287_128768

theorem percentage_profit_double_price (C S1 S2 : ℝ) (h1 : S1 = 1.5 * C) (h2 : S2 = 2 * S1) : 
  ((S2 - C) / C) * 100 = 200 := by
  sorry

end percentage_profit_double_price_l1287_128768


namespace solve_for_c_l1287_128786

theorem solve_for_c (c : ℚ) :
  (c - 35) / 14 = (2 * c + 9) / 49 →
  c = 1841 / 21 :=
by
  sorry

end solve_for_c_l1287_128786


namespace polynomial_divisibility_l1287_128794

theorem polynomial_divisibility (a : ℤ) (n : ℕ) (h_pos : 0 < n) : 
  (a ^ (2 * n + 1) + (a - 1) ^ (n + 2)) % (a ^ 2 - a + 1) = 0 :=
sorry

end polynomial_divisibility_l1287_128794


namespace missing_weights_l1287_128744

theorem missing_weights :
  ∃ (n k : ℕ), (n > 10) ∧ (606060 % 8 = 4) ∧ (606060 % 9 = 0) ∧ 
  (5 * k + 24 * k + 43 * k = 606060 + 72 * n) :=
sorry

end missing_weights_l1287_128744


namespace area_bounded_by_curves_eq_l1287_128770

open Real

noncomputable def area_bounded_by_curves : ℝ :=
  1 / 2 * (∫ (φ : ℝ) in (π/4)..(π/2), (sqrt 2 * cos (φ - π / 4))^2) +
  1 / 2 * (∫ (φ : ℝ) in (π/2)..(3 * π / 4), (sqrt 2 * sin (φ - π / 4))^2)

theorem area_bounded_by_curves_eq : area_bounded_by_curves = (π + 2) / 4 :=
  sorry

end area_bounded_by_curves_eq_l1287_128770


namespace sum_of_coordinates_D_l1287_128799

theorem sum_of_coordinates_D (x y : Int) :
  let N := (4, 10)
  let C := (14, 6)
  let D := (x, y)
  N = ((x + 14) / 2, (y + 6) / 2) →
  x + y = 8 :=
by
  intros
  sorry

end sum_of_coordinates_D_l1287_128799


namespace opposite_of_neg_twelve_l1287_128730

def opposite (n : Int) : Int := -n

theorem opposite_of_neg_twelve : opposite (-12) = 12 := by
  sorry

end opposite_of_neg_twelve_l1287_128730


namespace tina_total_time_l1287_128792

-- Define constants for the problem conditions
def assignment_time : Nat := 20
def dinner_time : Nat := 17 * 60 + 30 -- 5:30 PM in minutes
def clean_time_per_key : Nat := 7
def total_keys : Nat := 30
def remaining_keys : Nat := total_keys - 1
def dry_time_per_key : Nat := 10
def break_time : Nat := 3
def keys_per_break : Nat := 5

-- Define a function to compute total cleaning time for remaining keys
def total_cleaning_time (keys : Nat) (clean_time : Nat) : Nat :=
  keys * clean_time

-- Define a function to compute total drying time for all keys
def total_drying_time (keys : Nat) (dry_time : Nat) : Nat :=
  keys * dry_time

-- Define a function to compute total break time
def total_break_time (keys : Nat) (keys_per_break : Nat) (break_time : Nat) : Nat :=
  (keys / keys_per_break) * break_time

-- Define a function to compute the total time including cleaning, drying, breaks, and assignment
def total_time (cleaning_time drying_time break_time assignment_time : Nat) : Nat :=
  cleaning_time + drying_time + break_time + assignment_time

-- The theorem to be proven
theorem tina_total_time : 
  total_time (total_cleaning_time remaining_keys clean_time_per_key) 
              (total_drying_time total_keys dry_time_per_key)
              (total_break_time total_keys keys_per_break break_time)
              assignment_time = 541 :=
by sorry

end tina_total_time_l1287_128792


namespace vector_addition_correct_dot_product_correct_l1287_128778

def vector_add (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def dot_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)

theorem vector_addition_correct :
  let a := (1, 2)
  let b := (3, 1)
  vector_add a b = (4, 3) := by
  sorry

theorem dot_product_correct :
  let a := (1, 2)
  let b := (3, 1)
  dot_product a b = 5 := by
  sorry

end vector_addition_correct_dot_product_correct_l1287_128778


namespace oranges_to_apples_equivalence_l1287_128729

theorem oranges_to_apples_equivalence :
  (forall (o l a : ℝ), 4 * o = 3 * l ∧ 5 * l = 7 * a -> 20 * o = 21 * a) :=
by
  intro o l a
  intro h
  sorry

end oranges_to_apples_equivalence_l1287_128729


namespace a_plus_b_eq_neg2_l1287_128756

noncomputable def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

variable (a b : ℝ)

axiom h1 : f a = 1
axiom h2 : f b = 19

theorem a_plus_b_eq_neg2 : a + b = -2 :=
sorry

end a_plus_b_eq_neg2_l1287_128756


namespace range_of_m_for_inequality_l1287_128779

theorem range_of_m_for_inequality :
  {m : ℝ | ∀ x : ℝ, |x-1| + |x+m| > 3} = {m : ℝ | m < -4 ∨ m > 2} :=
sorry

end range_of_m_for_inequality_l1287_128779


namespace car_stops_at_three_seconds_l1287_128771

theorem car_stops_at_three_seconds (t : ℝ) (h : -3 * t^2 + 18 * t = 0) : t = 3 := 
sorry

end car_stops_at_three_seconds_l1287_128771


namespace circle_eq_l1287_128795

theorem circle_eq (A B : ℝ × ℝ) (hA1 : A = (5, 2)) (hA2 : B = (-1, 4)) (hx : ∃ (c : ℝ), (c, 0) = (c, 0)) :
  ∃ (C : ℝ) (D : ℝ) (x y : ℝ), (x + C) ^ 2 + y ^ 2 = D ∧ D = 20 ∧ (x - 1) ^ 2 + y ^ 2 = 20 :=
by
  sorry

end circle_eq_l1287_128795


namespace divides_mn_minus_one_l1287_128708

theorem divides_mn_minus_one (m n p : ℕ) (hp : p.Prime) (h1 : m < n) (h2 : n < p) 
    (hm2 : p ∣ m^2 + 1) (hn2 : p ∣ n^2 + 1) : p ∣ m * n - 1 :=
by
  sorry

end divides_mn_minus_one_l1287_128708


namespace fill_tank_time_l1287_128758

theorem fill_tank_time (R L E : ℝ) (fill_time : ℝ) (leak_time : ℝ) (effective_rate : ℝ) : 
  (R = 1 / fill_time) → 
  (L = 1 / leak_time) →
  (E = R - L) →
  (fill_time = 10) →
  (leak_time = 110) →
  (E = 1 / effective_rate) →
  effective_rate = 11 :=
by
  sorry

end fill_tank_time_l1287_128758


namespace find_missing_number_l1287_128763

theorem find_missing_number
  (mean : ℝ)
  (n : ℕ)
  (nums : List ℝ)
  (total_sum : ℝ)
  (sum_known_numbers : ℝ)
  (missing_number : ℝ) :
  mean = 20 → 
  n = 8 →
  nums = [1, 22, 23, 24, 25, missing_number, 27, 2] →
  total_sum = mean * n →
  sum_known_numbers = 1 + 22 + 23 + 24 + 25 + 27 + 2 →
  missing_number = total_sum - sum_known_numbers :=
by
  intros
  sorry

end find_missing_number_l1287_128763


namespace avg_minutes_eq_170_div_9_l1287_128745

-- Define the conditions
variables (s : ℕ) -- number of seventh graders
def sixth_graders := 3 * s
def seventh_graders := s
def eighth_graders := s / 2
def sixth_grade_minutes := 18
def seventh_grade_run_minutes := 20
def seventh_grade_stretching_minutes := 5
def eighth_grade_minutes := 12

-- Define the total activity minutes for each grade
def total_activity_minutes_sixth := sixth_grade_minutes * sixth_graders
def total_activity_minutes_seventh := (seventh_grade_run_minutes + seventh_grade_stretching_minutes) * seventh_graders
def total_activity_minutes_eighth := eighth_grade_minutes * eighth_graders

-- Calculate total activity minutes
def total_activity_minutes := total_activity_minutes_sixth + total_activity_minutes_seventh + total_activity_minutes_eighth

-- Calculate total number of students
def total_students := sixth_graders + seventh_graders + eighth_graders

-- Calculate average minutes per student
def average_minutes_per_student := total_activity_minutes / total_students

theorem avg_minutes_eq_170_div_9 : average_minutes_per_student s = 170 / 9 := by
  sorry

end avg_minutes_eq_170_div_9_l1287_128745


namespace notebook_pen_ratio_l1287_128704

theorem notebook_pen_ratio (pen_cost notebook_total_cost : ℝ) (num_notebooks : ℕ)
  (h1 : pen_cost = 1.50) (h2 : notebook_total_cost = 18) (h3 : num_notebooks = 4) :
  (notebook_total_cost / num_notebooks) / pen_cost = 3 :=
by
  -- The steps to prove this would go here
  sorry

end notebook_pen_ratio_l1287_128704


namespace initial_cost_of_milk_l1287_128772

theorem initial_cost_of_milk (total_money : ℝ) (bread_cost : ℝ) (detergent_cost : ℝ) (banana_cost_per_pound : ℝ) (banana_pounds : ℝ) (detergent_coupon : ℝ) (milk_discount_rate : ℝ) (money_left : ℝ)
  (h_total_money : total_money = 20) (h_bread_cost : bread_cost = 3.50) (h_detergent_cost : detergent_cost = 10.25) (h_banana_cost_per_pound : banana_cost_per_pound = 0.75) (h_banana_pounds : banana_pounds = 2)
  (h_detergent_coupon : detergent_coupon = 1.25) (h_milk_discount_rate : milk_discount_rate = 0.5) (h_money_left : money_left = 4) : 
  ∃ (initial_milk_cost : ℝ), initial_milk_cost = 4 := 
sorry

end initial_cost_of_milk_l1287_128772


namespace three_power_not_square_l1287_128775

theorem three_power_not_square (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) : ¬ ∃ k : ℕ, k * k = 3^m + 3^n + 1 := by 
  sorry

end three_power_not_square_l1287_128775


namespace unique_solution_k_values_l1287_128700

theorem unique_solution_k_values (k : ℝ) :
  (∃! x : ℝ, k * x ^ 2 - 3 * x + 2 = 0) ↔ (k = 0 ∨ k = 9 / 8) :=
by
  sorry

end unique_solution_k_values_l1287_128700


namespace one_minus_repeating_eight_l1287_128796

-- Given the condition
def b : ℚ := 8 / 9

-- The proof problem statement
theorem one_minus_repeating_eight : 1 - b = 1 / 9 := 
by
  sorry  -- proof to be provided

end one_minus_repeating_eight_l1287_128796


namespace sum_of_numbers_l1287_128741

-- Define the conditions
variables (a b : ℝ) (r d : ℝ)
def geometric_progression := a = 3 * r ∧ b = 3 * r^2
def arithmetic_progression := b = a + d ∧ 9 = b + d

-- Define the problem as proving the sum of a and b
theorem sum_of_numbers (h1 : geometric_progression a b r)
                       (h2 : arithmetic_progression a b d) : 
  a + b = 45 / 4 :=
sorry

end sum_of_numbers_l1287_128741


namespace simultaneous_solution_exists_l1287_128717

-- Definitions required by the problem
def eqn1 (m x : ℝ) : ℝ := m * x + 2
def eqn2 (m x : ℝ) : ℝ := (3 * m - 2) * x + 5

-- Proof statement
theorem simultaneous_solution_exists (m : ℝ) : 
  (∃ x y : ℝ, y = eqn1 m x ∧ y = eqn2 m x) ↔ (m ≠ 1) := 
sorry

end simultaneous_solution_exists_l1287_128717


namespace dmitry_black_socks_l1287_128740

theorem dmitry_black_socks :
  let blue_socks := 10
  let initial_black_socks := 22
  let white_socks := 12
  let total_initial_socks := blue_socks + initial_black_socks + white_socks
  ∀ x : ℕ,
    let total_socks := total_initial_socks + x
    let black_socks := initial_black_socks + x
    (black_socks : ℚ) / (total_socks : ℚ) = 2 / 3 → x = 22 :=
by
  sorry

end dmitry_black_socks_l1287_128740


namespace additional_plates_added_l1287_128754

def initial_plates : ℕ := 27
def added_plates : ℕ := 37
def total_plates : ℕ := 83

theorem additional_plates_added :
  total_plates - (initial_plates + added_plates) = 19 :=
by
  sorry

end additional_plates_added_l1287_128754


namespace ellipse_equation_no_match_l1287_128785

-- Definitions based on conditions in a)
def a : ℝ := 6
def c : ℝ := 1

-- Calculation for b² based on solution steps
def b_squared := a^2 - c^2

-- Standard forms of ellipse equations
def standard_ellipse_eq1 (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b_squared) = 1
def standard_ellipse_eq2 (x y : ℝ) : Prop := (y^2 / a^2) + (x^2 / b_squared) = 1

-- The proof problem statement
theorem ellipse_equation_no_match : 
  ∀ (x y : ℝ), ¬(standard_ellipse_eq1 x y) ∧ ¬(standard_ellipse_eq2 x y) := 
sorry

end ellipse_equation_no_match_l1287_128785


namespace range_of_m_value_of_m_l1287_128788

variable (α β m : ℝ)

open Real

-- Conditions: α and β are positive roots.
def quadratic_roots (α β m : ℝ) : Prop :=
  (α > 0) ∧ (β > 0) ∧ (α + β = 1 - 2*m) ∧ (α * β = m^2)

-- Part 1: Range of values for m.
theorem range_of_m (h : quadratic_roots α β m) : m ≤ 1/4 ∧ m ≠ 0 :=
sorry

-- Part 2: Given α^2 + β^2 = 49, find the value of m.
theorem value_of_m (h : quadratic_roots α β m) (h' : α^2 + β^2 = 49) : m = -4 :=
sorry

end range_of_m_value_of_m_l1287_128788


namespace count_positive_integers_l1287_128787

theorem count_positive_integers (n : ℕ) : ∃ k : ℕ, k = 9 ∧  ∀ n, 1 ≤ n → n < 10 → 3 * n + 20 < 50 :=
by
  sorry

end count_positive_integers_l1287_128787


namespace seq_proof_l1287_128737

noncomputable def arithmetic_seq (a1 a2 : ℤ) : Prop :=
  ∃ (d : ℤ), a1 = -1 + d ∧ a2 = a1 + d ∧ -4 = a1 + 3 * d

noncomputable def geometric_seq (b : ℤ) : Prop :=
  b = 2 ∨ b = -2

theorem seq_proof (a1 a2 b : ℤ) 
  (h1 : arithmetic_seq a1 a2) 
  (h2 : geometric_seq b) : 
  (a2 + a1 : ℚ) / b = 5 / 2 ∨ (a2 + a1 : ℚ) / b = -5 / 2 := by
  sorry

end seq_proof_l1287_128737


namespace totalGamesPlayed_l1287_128714

def numPlayers : ℕ := 30

def numGames (n : ℕ) : ℕ := (n * (n - 1)) / 2

theorem totalGamesPlayed :
  numGames numPlayers = 435 :=
by
  sorry

end totalGamesPlayed_l1287_128714


namespace A_share_in_profit_l1287_128776

-- Define the investments and profits
def A_investment : ℕ := 6300
def B_investment : ℕ := 4200
def C_investment : ℕ := 10500
def total_profit : ℕ := 12200

-- Define the total investment
def total_investment : ℕ := A_investment + B_investment + C_investment

-- Define A's ratio in the investment
def A_ratio : ℚ := A_investment / total_investment

-- Define A's share in the profit
def A_share : ℚ := total_profit * A_ratio

-- The theorem to prove
theorem A_share_in_profit : A_share = 3660 := by
  sorry

end A_share_in_profit_l1287_128776


namespace g_decreasing_on_neg1_0_l1287_128706

noncomputable def f (x : ℝ) : ℝ := 8 + 2 * x - x^2 
noncomputable def g (x : ℝ) : ℝ := f (2 - x^2)

theorem g_decreasing_on_neg1_0 : 
  ∀ x y : ℝ, -1 < x ∧ x < 0 ∧ -1 < y ∧ y < 0 ∧ x < y → g y < g x :=
sorry

end g_decreasing_on_neg1_0_l1287_128706


namespace validate_option_B_l1287_128781

theorem validate_option_B (a b : ℝ) : 
  (2 * a + 3 * a^2 ≠ 5 * a^3) ∧ 
  ((-a^3)^2 = a^6) ∧ 
  (¬ (-4 * a^3 * b / (2 * a) = -2 * a^2)) ∧ 
  ((5 * a * b)^2 ≠ 10 * a^2 * b^2) := 
by
  sorry

end validate_option_B_l1287_128781


namespace gcd_b_squared_plus_11b_plus_28_and_b_plus_6_l1287_128761

theorem gcd_b_squared_plus_11b_plus_28_and_b_plus_6 (b : ℤ) (h : ∃ k : ℤ, b = 1573 * k) : 
  Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 := 
sorry

end gcd_b_squared_plus_11b_plus_28_and_b_plus_6_l1287_128761


namespace find_k_l1287_128773

theorem find_k (x y k : ℤ) (h1 : 2 * x - y = 5 * k + 6) (h2 : 4 * x + 7 * y = k) (h3 : x + y = 2023) : k = 2022 := 
by {
  sorry
}

end find_k_l1287_128773


namespace unique_solution_exists_l1287_128783

theorem unique_solution_exists (ell : ℚ) (h : ell ≠ -2) : 
  (∃! x : ℚ, (x + 3) / (ell * x + 2) = x) ↔ ell = -1 / 12 := 
by
  sorry

end unique_solution_exists_l1287_128783


namespace parabola_coefficients_sum_l1287_128709

theorem parabola_coefficients_sum :
  ∃ a b c : ℝ, 
  (∀ y : ℝ, (7 = -(6 ^ 2) * a + b * 6 + c)) ∧
  (5 = a * (-4) ^ 2 + b * (-4) + c) ∧
  (a + b + c = -42) := 
sorry

end parabola_coefficients_sum_l1287_128709


namespace Bryan_deposited_312_l1287_128716

-- Definitions based on conditions
def MarkDeposit : ℕ := 88
def TotalDeposit : ℕ := 400
def MaxBryanDeposit (MarkDeposit : ℕ) : ℕ := 5 * MarkDeposit 

def BryanDeposit (B : ℕ) : Prop := B < MaxBryanDeposit MarkDeposit ∧ MarkDeposit + B = TotalDeposit

theorem Bryan_deposited_312 : BryanDeposit 312 :=
by
   -- Proof steps go here
   sorry

end Bryan_deposited_312_l1287_128716


namespace triangle_side_length_b_l1287_128743

/-
In a triangle ABC with angles such that ∠C = 4∠A, and sides such that a = 35 and c = 64, prove that the length of side b is 140 * cos²(A).
-/
theorem triangle_side_length_b (A C : ℝ) (a c : ℝ) (hC : C = 4 * A) (ha : a = 35) (hc : c = 64) :
  ∃ (b : ℝ), b = 140 * (Real.cos A) ^ 2 :=
by
  sorry

end triangle_side_length_b_l1287_128743


namespace expression_value_l1287_128742

variable (m n : ℝ)

theorem expression_value (hm : 3 * m ^ 2 + 5 * m - 3 = 0)
                         (hn : 3 * n ^ 2 - 5 * n - 3 = 0)
                         (hneq : m * n ≠ 1) :
                         (1 / n ^ 2) + (m / n) - (5 / 3) * m = 25 / 9 :=
by {
  sorry
}

end expression_value_l1287_128742


namespace football_team_throwers_l1287_128757

theorem football_team_throwers {T N : ℕ} (h1 : 70 - T = N)
                                (h2 : 62 = T + (2 / 3 * N)) : 
                                T = 46 := 
by
  sorry

end football_team_throwers_l1287_128757


namespace number_of_outliers_l1287_128738

def data_set : List ℕ := [4, 23, 27, 27, 35, 37, 37, 39, 47, 53]

def Q1 : ℕ := 27
def Q3 : ℕ := 39

def IQR : ℕ := Q3 - Q1
def lower_threshold : ℕ := Q1 - (3 * IQR / 2)
def upper_threshold : ℕ := Q3 + (3 * IQR / 2)

def outliers (s : List ℕ) (low high : ℕ) : List ℕ :=
  s.filter (λ x => x < low ∨ x > high)

theorem number_of_outliers :
  outliers data_set lower_threshold upper_threshold = [4] :=
by
  sorry

end number_of_outliers_l1287_128738


namespace four_pow_sub_divisible_iff_l1287_128790

open Nat

theorem four_pow_sub_divisible_iff (m n k : ℕ) (h₁ : m > n) : 
  (3^(k + 1)) ∣ (4^m - 4^n) ↔ (3^k) ∣ (m - n) := 
by sorry

end four_pow_sub_divisible_iff_l1287_128790


namespace homework_problem1_homework_problem2_l1287_128766

-- Definition and conditions for the first equation
theorem homework_problem1 (a b : ℕ) (h1 : a + b = a * b) : a = 2 ∧ b = 2 :=
by sorry

-- Definition and conditions for the second equation
theorem homework_problem2 (a b : ℕ) (h2 : a * b * (a + b) = 182) : 
    (a = 1 ∧ b = 13) ∨ (a = 13 ∧ b = 1) :=
by sorry

end homework_problem1_homework_problem2_l1287_128766


namespace students_taking_all_three_classes_l1287_128798

variable (students : Finset ℕ)
variable (yoga bridge painting : Finset ℕ)

variables (yoga_count bridge_count painting_count at_least_two exactly_two all_three : ℕ)

variable (total_students : students.card = 25)
variable (yoga_students : yoga.card = 12)
variable (bridge_students : bridge.card = 15)
variable (painting_students : painting.card = 11)
variable (at_least_two_classes : at_least_two = 10)
variable (exactly_two_classes : exactly_two = 7)

theorem students_taking_all_three_classes :
  all_three = 3 :=
sorry

end students_taking_all_three_classes_l1287_128798


namespace find_f2_l1287_128767

def f (x a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
sorry

end find_f2_l1287_128767


namespace remainder_when_divided_by_95_l1287_128702

theorem remainder_when_divided_by_95 (x : ℤ) (h1 : x % 19 = 12) :
  x % 95 = 12 := 
sorry

end remainder_when_divided_by_95_l1287_128702
