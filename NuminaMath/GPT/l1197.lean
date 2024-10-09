import Mathlib

namespace shooting_average_l1197_119762

noncomputable def total_points (a b c d : ℕ) : ℕ :=
  (a * 10) + (b * 9) + (c * 8) + (d * 7)

noncomputable def average_points (total : ℕ) (shots : ℕ) : ℚ :=
  total / shots

theorem shooting_average :
  let a := 1
  let b := 4
  let c := 3
  let d := 2
  let shots := 10
  total_points a b c d = 84 ∧
  average_points (total_points a b c d) shots = 8.4 :=
by {
  sorry
}

end shooting_average_l1197_119762


namespace apple_juice_fraction_correct_l1197_119771

def problem_statement : Prop :=
  let pitcher1_capacity := 800
  let pitcher2_capacity := 500
  let pitcher1_apple_fraction := 1 / 4
  let pitcher2_apple_fraction := 1 / 5
  let pitcher1_apple_volume := pitcher1_capacity * pitcher1_apple_fraction
  let pitcher2_apple_volume := pitcher2_capacity * pitcher2_apple_fraction
  let total_apple_volume := pitcher1_apple_volume + pitcher2_apple_volume
  let total_volume := pitcher1_capacity + pitcher2_capacity
  total_apple_volume / total_volume = 3 / 13

theorem apple_juice_fraction_correct : problem_statement := 
  sorry

end apple_juice_fraction_correct_l1197_119771


namespace exactly_one_correct_l1197_119785

theorem exactly_one_correct (P_A P_B : ℚ) (hA : P_A = 1/5) (hB : P_B = 1/4) :
  P_A * (1 - P_B) + (1 - P_A) * P_B = 7/20 :=
by
  sorry

end exactly_one_correct_l1197_119785


namespace max_withdrawal_l1197_119739

def initial_balance : ℕ := 500
def withdraw_amount : ℕ := 300
def add_amount : ℕ := 198
def remaining_balance (x : ℕ) : Prop := 
  x % 6 = 0 ∧ x ≤ initial_balance

theorem max_withdrawal : ∃(max_withdrawal_amount : ℕ), 
  max_withdrawal_amount = initial_balance - 498 :=
sorry

end max_withdrawal_l1197_119739


namespace inequality_solution_l1197_119783

theorem inequality_solution :
  {x : Real | (2 * x - 5) * (x - 3) / x ≥ 0} = {x : Real | (x ∈ Set.Ioc 0 (5 / 2)) ∨ (x ∈ Set.Ici 3)} := 
sorry

end inequality_solution_l1197_119783


namespace quadratic_function_positive_difference_l1197_119714

/-- Given a quadratic function y = ax^2 + bx + c, where the coefficient a
indicates a downward-opening parabola (a < 0) and the y-intercept is positive (c > 0),
prove that the expression (c - a) is always positive. -/
theorem quadratic_function_positive_difference (a b c : ℝ) (h1 : a < 0) (h2 : c > 0) : c - a > 0 := 
by
  sorry

end quadratic_function_positive_difference_l1197_119714


namespace lindy_total_distance_l1197_119741

def meet_distance (d v_j v_c : ℕ) : ℕ :=
  d / (v_j + v_c)

def lindy_distance (v_l t : ℕ) : ℕ :=
  v_l * t

theorem lindy_total_distance
  (d : ℕ)
  (v_j : ℕ)
  (v_c : ℕ)
  (v_l : ℕ)
  (h1 : d = 360)
  (h2 : v_j = 5)
  (h3 : v_c = 7)
  (h4 : v_l = 12)
  :
  lindy_distance v_l (meet_distance d v_j v_c) = 360 :=
by
  sorry

end lindy_total_distance_l1197_119741


namespace change_digit_correct_sum_l1197_119722

theorem change_digit_correct_sum :
  ∃ d e, 
  d = 2 ∧ e = 8 ∧ 
  653479 + 938521 ≠ 1616200 ∧
  (658479 + 938581 = 1616200) ∧ 
  d + e = 10 := 
by {
  -- our proof goes here
  sorry
}

end change_digit_correct_sum_l1197_119722


namespace percentage_B_of_C_l1197_119718

theorem percentage_B_of_C 
  (A C B : ℝ)
  (h1 : A = (7 / 100) * C)
  (h2 : A = (50 / 100) * B) :
  B = (14 / 100) * C := 
sorry

end percentage_B_of_C_l1197_119718


namespace initial_cats_count_l1197_119716

theorem initial_cats_count :
  ∀ (initial_birds initial_puppies initial_spiders final_total initial_cats: ℕ),
    initial_birds = 12 →
    initial_puppies = 9 →
    initial_spiders = 15 →
    final_total = 25 →
    (initial_birds / 2 + initial_puppies - 3 + initial_spiders - 7 + initial_cats = final_total) →
    initial_cats = 5 := by
  intros initial_birds initial_puppies initial_spiders final_total initial_cats h1 h2 h3 h4 h5
  sorry

end initial_cats_count_l1197_119716


namespace average_monthly_increase_is_20_percent_l1197_119773

-- Define the given conditions in Lean
def V_Jan : ℝ := 2 
def V_Mar : ℝ := 2.88 

-- Percentage increase each month over the previous month is the same
def consistent_growth_rate (x : ℝ) : Prop := 
  V_Jan * (1 + x)^2 = V_Mar

-- We need to prove that the monthly growth rate x is 0.2 (or 20%)
theorem average_monthly_increase_is_20_percent : 
  ∃ x : ℝ, consistent_growth_rate x ∧ x = 0.2 :=
by
  sorry

end average_monthly_increase_is_20_percent_l1197_119773


namespace volume_CO2_is_7_l1197_119736

-- Definitions based on conditions
def Avogadro_law (V1 V2 : ℝ) : Prop := V1 = V2
def molar_ratio (V_CO2 V_O2 : ℝ) : Prop := V_CO2 = 1 / 2 * V_O2
def volume_O2 : ℝ := 14

-- Statement to be proved
theorem volume_CO2_is_7 : ∃ V_CO2 : ℝ, molar_ratio V_CO2 volume_O2 ∧ V_CO2 = 7 := by
  sorry

end volume_CO2_is_7_l1197_119736


namespace lamps_on_bridge_l1197_119746

theorem lamps_on_bridge (bridge_length : ℕ) (lamp_spacing : ℕ) (num_intervals : ℕ) (num_lamps : ℕ) 
  (h1 : bridge_length = 30) 
  (h2 : lamp_spacing = 5)
  (h3 : num_intervals = bridge_length / lamp_spacing)
  (h4 : num_lamps = num_intervals + 1) :
  num_lamps = 7 := 
by
  sorry

end lamps_on_bridge_l1197_119746


namespace chocolate_bars_percentage_l1197_119726

noncomputable def total_chocolate_bars (milk dark almond white caramel : ℕ) : ℕ :=
  milk + dark + almond + white + caramel

noncomputable def percentage (count total : ℕ) : ℚ :=
  (count : ℚ) / (total : ℚ) * 100

theorem chocolate_bars_percentage :
  let milk := 36
  let dark := 21
  let almond := 40
  let white := 15
  let caramel := 28
  let total := total_chocolate_bars milk dark almond white caramel
  total = 140 ∧
  percentage milk total = 25.71 ∧
  percentage dark total = 15 ∧
  percentage almond total = 28.57 ∧
  percentage white total = 10.71 ∧
  percentage caramel total = 20 :=
by
  sorry

end chocolate_bars_percentage_l1197_119726


namespace mapping_f_of_neg2_and_3_l1197_119707

-- Define the mapping f
def f (x y : ℝ) : ℝ × ℝ := (x + y, x * y)

-- Define the given point
def p : ℝ × ℝ := (-2, 3)

-- Define the expected corresponding point
def expected_p : ℝ × ℝ := (1, -6)

-- The theorem stating the problem to be proved
theorem mapping_f_of_neg2_and_3 :
  f p.1 p.2 = expected_p := by
  sorry

end mapping_f_of_neg2_and_3_l1197_119707


namespace not_necessarily_divisible_by_66_l1197_119723

theorem not_necessarily_divisible_by_66 (m : ℤ) (h1 : ∃ k : ℤ, m = k * (k + 1) * (k + 2) * (k + 3) * (k + 4)) (h2 : 11 ∣ m) : ¬ (66 ∣ m) :=
sorry

end not_necessarily_divisible_by_66_l1197_119723


namespace calc_m_l1197_119745

theorem calc_m (m : ℤ) (h : (64 : ℝ)^(1 / 3) = 2^m) : m = 2 :=
sorry

end calc_m_l1197_119745


namespace g_h_value_l1197_119709

def g (x : ℕ) : ℕ := 3 * x^2 + 2
def h (x : ℕ) : ℕ := 5 * x^3 - 2

theorem g_h_value : g (h 2) = 4334 := by
  sorry

end g_h_value_l1197_119709


namespace minimum_photos_l1197_119788

-- Given conditions:
-- 1. There are exactly three monuments.
-- 2. A group of 42 tourists visited the city.
-- 3. Each tourist took at most one photo of each of the three monuments.
-- 4. Any two tourists together had photos of all three monuments.

theorem minimum_photos (n m : ℕ) (h₁ : n = 42) (h₂ : m = 3) 
  (photographs : ℕ → ℕ → Bool) -- whether tourist i took a photo of monument j
  (h₃ : ∀ i : ℕ, i < n → ∀ j : ℕ, j < m → photographs i j = tt ∨ photographs i j = ff) -- each tourist took at most one photo of each monument
  (h₄ : ∀ i₁ i₂ : ℕ, i₁ < n → i₂ < n → i₁ ≠ i₂ → ∀ j : ℕ, j < m → photographs i₁ j = tt ∨ photographs i₂ j = tt) -- any two tourists together had photos of all three monuments
  : ∃ min_photos : ℕ, min_photos = 123 :=
by
  sorry

end minimum_photos_l1197_119788


namespace exists_solution_for_lambda_9_l1197_119781

theorem exists_solution_for_lambda_9 :
  ∃ x y : ℝ, (x^2 + y^2 = 8 * x + 6 * y) ∧ (9 * x^2 + y^2 = 6 * y) ∧ (y^2 + 9 = 9 * x + 6 * y + 9) :=
by
  sorry

end exists_solution_for_lambda_9_l1197_119781


namespace sets_difference_M_star_N_l1197_119761

def M (y : ℝ) : Prop := y ≤ 2

def N (y : ℝ) : Prop := 0 ≤ y ∧ y ≤ 3

def M_star_N (y : ℝ) : Prop := y < 0

theorem sets_difference_M_star_N : {y : ℝ | M y ∧ ¬ N y} = {y : ℝ | M_star_N y} :=
by {
  sorry
}

end sets_difference_M_star_N_l1197_119761


namespace tickets_count_l1197_119729

theorem tickets_count (x y: ℕ) (h : 3 * x + 5 * y = 78) : 
  ∃ n : ℕ , n = 6 :=
sorry

end tickets_count_l1197_119729


namespace simplify_expression_l1197_119780

variable {s r : ℝ}

theorem simplify_expression :
  (2 * s^2 + 5 * r - 4) - (3 * s^2 + 9 * r - 7) = -s^2 - 4 * r + 3 := 
by
  sorry

end simplify_expression_l1197_119780


namespace composite_numbers_quotient_l1197_119700

theorem composite_numbers_quotient :
  (14 * 15 * 16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 : ℚ) /
  (27 * 28 * 30 * 32 * 33 * 34 * 35 * 36 * 38 * 39) =
  (14 * 15 * 16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 : ℚ) / 
  (27 * 28 * 30 * 32 * 33 * 34 * 35 * 36 * 38 * 39) :=
by sorry

end composite_numbers_quotient_l1197_119700


namespace greatest_divisor_lemma_l1197_119752

theorem greatest_divisor_lemma : ∃ (d : ℕ), d = Nat.gcd 1636 1852 ∧ d = 4 := by
  sorry

end greatest_divisor_lemma_l1197_119752


namespace inequality_l1197_119749

theorem inequality (a b : ℝ) : a^2 + b^2 + 7/4 ≥ a * b + 2 * a + b / 2 :=
sorry

end inequality_l1197_119749


namespace dogs_bunnies_ratio_l1197_119757

theorem dogs_bunnies_ratio (total : ℕ) (dogs : ℕ) (bunnies : ℕ) (h1 : total = 375) (h2 : dogs = 75) (h3 : bunnies = total - dogs) : (75 / 75 : ℚ) / (300 / 75 : ℚ) = 1 / 4 := by
  sorry

end dogs_bunnies_ratio_l1197_119757


namespace value_of_a_is_negative_one_l1197_119756

-- Conditions
def I (a : ℤ) : Set ℤ := {2, 4, a^2 - a - 3}
def A (a : ℤ) : Set ℤ := {4, 1 - a}
def complement_I_A (a : ℤ) : Set ℤ := {x ∈ I a | x ∉ A a}

-- Theorem statement
theorem value_of_a_is_negative_one (a : ℤ) (h : complement_I_A a = {-1}) : a = -1 :=
by
  sorry

end value_of_a_is_negative_one_l1197_119756


namespace decimal_multiplication_l1197_119744

theorem decimal_multiplication : (3.6 * 0.3 = 1.08) := by
  sorry

end decimal_multiplication_l1197_119744


namespace h_at_3_l1197_119747

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 5
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (f x) + 1
noncomputable def h (x : ℝ) : ℝ := f (g x)

theorem h_at_3 : h 3 = 74 + 28 * Real.sqrt 2 :=
by
  sorry

end h_at_3_l1197_119747


namespace find_digit_x_l1197_119798

def base7_number (x : ℕ) : ℕ := 5 * 7^3 + 2 * 7^2 + x * 7 + 4

def is_divisible_by_19 (n : ℕ) : Prop := ∃ k : ℕ, n = 19 * k

theorem find_digit_x : is_divisible_by_19 (base7_number 4) :=
sorry

end find_digit_x_l1197_119798


namespace isosceles_triangle_perimeter_l1197_119725

theorem isosceles_triangle_perimeter {a b c : ℝ} (h1 : a = 4) (h2 : b = 8) 
  (isosceles : a = c ∨ b = c) (triangle_inequality : a + a > b) :
  a + b + c = 20 :=
by
  sorry

end isosceles_triangle_perimeter_l1197_119725


namespace quadratic_equation_properties_l1197_119767

theorem quadratic_equation_properties (m : ℝ) (h : m < 4) (root_one : ℝ) (root_two : ℝ) 
  (eq1 : root_one + root_two = 4) (eq2 : root_one * root_two = m) (root_one_eq : root_one = -1) :
  m = -5 ∧ root_two = 5 ∧ (root_one ≠ root_two) :=
by
  -- Sorry is added to skip the proof because only the statement is needed.
  sorry

end quadratic_equation_properties_l1197_119767


namespace acute_not_greater_than_right_l1197_119794

-- Definitions for conditions
def is_right_angle (α : ℝ) : Prop := α = 90
def is_acute_angle (α : ℝ) : Prop := α < 90

-- Statement to be proved
theorem acute_not_greater_than_right (α : ℝ) (h1 : is_right_angle 90) (h2 : is_acute_angle α) : ¬ (α > 90) :=
by
    sorry

end acute_not_greater_than_right_l1197_119794


namespace quadratic_distinct_real_roots_range_l1197_119770

open Real

theorem quadratic_distinct_real_roots_range (k : ℝ) :
    (∃ a b c : ℝ, a = k^2 ∧ b = 4 * k - 1 ∧ c = 4 ∧ (b^2 - 4 * a * c > 0) ∧ a ≠ 0) ↔ (k < 1 / 8 ∧ k ≠ 0) :=
by
  sorry

end quadratic_distinct_real_roots_range_l1197_119770


namespace x_intercept_of_line_l1197_119765

theorem x_intercept_of_line (x y : ℝ) : (4 * x + 7 * y = 28) ∧ (y = 0) → x = 7 :=
by
  sorry

end x_intercept_of_line_l1197_119765


namespace num_distinct_units_digits_of_cubes_l1197_119717

theorem num_distinct_units_digits_of_cubes : 
  ∃ S : Finset ℕ, (∀ d : ℕ, (d < 10) → (d^3 % 10) ∈ S) ∧ S.card = 10 := by
  sorry

end num_distinct_units_digits_of_cubes_l1197_119717


namespace consecutive_product_solution_l1197_119791

theorem consecutive_product_solution :
  ∀ (n : ℤ), (∃ a : ℤ, n^4 + 8 * n + 11 = a * (a + 1)) ↔ n = 1 :=
by
  sorry

end consecutive_product_solution_l1197_119791


namespace carrie_worked_days_l1197_119724

theorem carrie_worked_days (d : ℕ) 
  (h1: ∀ n : ℕ, d = n → (2 * 22 * n - 54 = 122)) : d = 4 :=
by
  -- The proof will go here.
  sorry

end carrie_worked_days_l1197_119724


namespace max_campaign_making_animals_prime_max_campaign_making_animals_nine_l1197_119759

theorem max_campaign_making_animals_prime (n : ℕ) (h_prime : Nat.Prime n) (h_ge : n ≥ 3) : 
  ∃ k, k = (n - 1) / 2 :=
by
  sorry

theorem max_campaign_making_animals_nine : ∃ k, k = 4 :=
by
  sorry

end max_campaign_making_animals_prime_max_campaign_making_animals_nine_l1197_119759


namespace range_a_l1197_119786

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 3

theorem range_a (H : ∀ x1 x2 : ℝ, 2 < x1 → 2 < x2 → (f x1 a - f x2 a) / (x1 - x2) > 0) : a ≤ 2 := 
sorry

end range_a_l1197_119786


namespace hairstylist_earnings_per_week_l1197_119702

theorem hairstylist_earnings_per_week :
  let cost_normal := 5
  let cost_special := 6
  let cost_trendy := 8
  let haircuts_normal := 5
  let haircuts_special := 3
  let haircuts_trendy := 2
  let days_per_week := 7
  let daily_earnings := cost_normal * haircuts_normal + cost_special * haircuts_special + cost_trendy * haircuts_trendy
  let weekly_earnings := daily_earnings * days_per_week
  weekly_earnings = 413 := sorry

end hairstylist_earnings_per_week_l1197_119702


namespace verify_formula_n1_l1197_119764

theorem verify_formula_n1 (a : ℝ) (ha : a ≠ 1) : 1 + a = (a^3 - 1) / (a - 1) :=
by 
  sorry

end verify_formula_n1_l1197_119764


namespace largest_square_area_l1197_119753

theorem largest_square_area (a b c : ℝ) 
  (h1 : c^2 = a^2 + b^2) 
  (h2 : a = b - 5) 
  (h3 : a^2 + b^2 + c^2 = 450) : 
  c^2 = 225 :=
by 
  sorry

end largest_square_area_l1197_119753


namespace minimum_value_op_dot_fp_l1197_119711

theorem minimum_value_op_dot_fp (x y : ℝ) (h_ellipse : x^2 / 2 + y^2 = 1) :
  let OP := (x, y)
  let FP := (x - 1, y)
  let dot_product := x * (x - 1) + y^2
  dot_product ≥ 1 / 2 :=
by
  sorry

end minimum_value_op_dot_fp_l1197_119711


namespace maximum_a_pos_integer_greatest_possible_value_of_a_l1197_119777

theorem maximum_a_pos_integer (a : ℕ) (h : ∃ x : ℤ, x^2 + (a * x : ℤ) = -20) : a ≤ 21 :=
by
  sorry

theorem greatest_possible_value_of_a : ∃ (a : ℕ), (∀ b : ℕ, (∃ x : ℤ, x^2 + (b * x : ℤ) = -20) → b ≤ 21) ∧ 21 = a :=
by
  sorry

end maximum_a_pos_integer_greatest_possible_value_of_a_l1197_119777


namespace total_fruits_in_baskets_l1197_119775

def total_fruits (apples1 oranges1 bananas1 apples2 oranges2 bananas2 : ℕ) :=
  apples1 + oranges1 + bananas1 + apples2 + oranges2 + bananas2

theorem total_fruits_in_baskets :
  total_fruits 9 15 14 (9 - 2) (15 - 2) (14 - 2) = 70 :=
by
  sorry

end total_fruits_in_baskets_l1197_119775


namespace nina_running_distance_l1197_119755

theorem nina_running_distance (x : ℝ) (hx : 2 * x + 0.67 = 0.83) : x = 0.08 := by
  sorry

end nina_running_distance_l1197_119755


namespace smallest_integer_l1197_119763

theorem smallest_integer (x : ℤ) (h : 3 * (Int.natAbs x)^3 + 5 < 56) : x = -2 :=
sorry

end smallest_integer_l1197_119763


namespace men_hours_per_day_l1197_119719

theorem men_hours_per_day
  (H : ℕ)
  (men_days := 15 * 21 * H)
  (women_days := 21 * 20 * 9)
  (conversion_ratio := 3 / 2)
  (equivalent_man_hours := women_days * conversion_ratio)
  (same_work : men_days = equivalent_man_hours) :
  H = 8 :=
by
  sorry

end men_hours_per_day_l1197_119719


namespace area_of_quadrilateral_ABDF_l1197_119787

theorem area_of_quadrilateral_ABDF :
  let length := 40
  let width := 30
  let rectangle_area := length * width
  let B := (1/4 : ℝ) * length
  let F := (1/2 : ℝ) * width
  let area_BCD := (1/2 : ℝ) * (3/4 : ℝ) * length * width
  let area_EFD := (1/2 : ℝ) * F * length
  rectangle_area - area_BCD - area_EFD = 450 := sorry

end area_of_quadrilateral_ABDF_l1197_119787


namespace rationalize_denominator_l1197_119772

def cbrt (x : ℝ) : ℝ := x^(1/3)

theorem rationalize_denominator :
  let a := cbrt 2
  let b := cbrt 27
  b = 3 -> ( 1 / (a + b)) = (cbrt 4 / (2 + 3 * cbrt 4))
:= by
  intro a
  intro b
  sorry

end rationalize_denominator_l1197_119772


namespace peanut_raising_ratio_l1197_119768

theorem peanut_raising_ratio
  (initial_peanuts : ℝ)
  (remove_peanuts_1 : ℝ)
  (add_raisins_1 : ℝ)
  (remove_mixture : ℝ)
  (add_raisins_2 : ℝ)
  (final_peanuts : ℝ)
  (final_raisins : ℝ)
  (ratio : ℝ) :
  initial_peanuts = 10 ∧
  remove_peanuts_1 = 2 ∧
  add_raisins_1 = 2 ∧
  remove_mixture = 2 ∧
  add_raisins_2 = 2 ∧
  final_peanuts = initial_peanuts - remove_peanuts_1 - (remove_mixture * (initial_peanuts - remove_peanuts_1) / (initial_peanuts - remove_peanuts_1 + add_raisins_1)) ∧
  final_raisins = add_raisins_1 - (remove_mixture * add_raisins_1 / (initial_peanuts - remove_peanuts_1 + add_raisins_1)) + add_raisins_2 ∧
  ratio = final_peanuts / final_raisins →
  ratio = 16 / 9 := by
  sorry

end peanut_raising_ratio_l1197_119768


namespace money_per_percentage_point_l1197_119784

theorem money_per_percentage_point
  (plates : ℕ) (total_states : ℕ) (total_amount : ℤ)
  (h_plates : plates = 40) (h_total_states : total_states = 50) (h_total_amount : total_amount = 160) :
  total_amount / (plates * 100 / total_states) = 2 :=
by
  -- Omitted steps of the proof
  sorry

end money_per_percentage_point_l1197_119784


namespace james_louise_age_sum_l1197_119731

variables (J L : ℝ)

theorem james_louise_age_sum
  (h₁ : J = L + 9)
  (h₂ : J + 5 = 3 * (L - 3)) :
  J + L = 32 :=
by
  /- Proof goes here -/
  sorry

end james_louise_age_sum_l1197_119731


namespace michael_digging_time_equals_700_l1197_119710

-- Conditions defined
def digging_rate := 4
def father_depth := digging_rate * 400
def michael_depth := 2 * father_depth - 400
def time_for_michael := michael_depth / digging_rate

-- Statement to prove
theorem michael_digging_time_equals_700 : time_for_michael = 700 :=
by
  -- Here we would provide the proof steps, but we use sorry for now
  sorry

end michael_digging_time_equals_700_l1197_119710


namespace minimum_seats_l1197_119740

-- Condition: 150 seats in a row.
def seats : ℕ := 150

-- Assertion: The fewest number of seats that must be occupied so that any additional person seated must sit next to someone.
def minOccupiedSeats : ℕ := 50

theorem minimum_seats (s : ℕ) (m : ℕ) (h_seats : s = 150) (h_min : m = 50) :
  (∀ x, x = 150 → ∀ n, n ≥ 0 ∧ n ≤ m → 
    ∃ y, y ≥ 0 ∧ y ≤ x ∧ ∀ z, z = n + 1 → ∃ w, w ≥ 0 ∧ w ≤ x ∧ w = n ∨ w = n + 1) := 
sorry

end minimum_seats_l1197_119740


namespace radius_squared_l1197_119735

-- Definitions of the conditions
def point_A := (2, -1)
def line_l1 (x y : ℝ) := x + y = 1
def line_l2 (x y : ℝ) := 2 * x + y = 0

-- Circle with center (h, k) and radius r
def circle_equation (h k r x y : ℝ) := (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2

-- Prove statement: r^2 = 2 given the conditions
theorem radius_squared (h k r : ℝ) 
  (H1 : circle_equation h k r 2 (-1))
  (H2 : line_l1 h k)
  (H3 : line_l2 h k):
  r ^ 2 = 2 := sorry

end radius_squared_l1197_119735


namespace car_traveled_miles_per_gallon_city_l1197_119708

noncomputable def miles_per_gallon_city (H C G : ℝ) : Prop :=
  (C = H - 18) ∧ (462 = H * G) ∧ (336 = C * G)

theorem car_traveled_miles_per_gallon_city :
  ∃ H G, miles_per_gallon_city H 48 G :=
by
  sorry

end car_traveled_miles_per_gallon_city_l1197_119708


namespace ellipse_is_correct_l1197_119797

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 12) = -1

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 16) = 1

-- Define the conditions
def ellipse_focus_vertex_of_hyperbola_vertex_and_focus (x y : ℝ) : Prop :=
  hyperbola_eq x y ∧ ellipse_eq x y

-- Theorem stating that the ellipse equation holds given the conditions
theorem ellipse_is_correct :
  ∀ (x y : ℝ), ellipse_focus_vertex_of_hyperbola_vertex_and_focus x y →
  ellipse_eq x y := by
  intros x y h
  sorry

end ellipse_is_correct_l1197_119797


namespace crayons_and_erasers_difference_l1197_119779

theorem crayons_and_erasers_difference 
  (initial_crayons : ℕ) (initial_erasers : ℕ) (remaining_crayons : ℕ) 
  (h1 : initial_crayons = 601) (h2 : initial_erasers = 406) (h3 : remaining_crayons = 336) : 
  initial_erasers - remaining_crayons = 70 :=
by
  sorry

end crayons_and_erasers_difference_l1197_119779


namespace john_income_l1197_119743

theorem john_income 
  (john_tax_rate : ℝ) (ingrid_tax_rate : ℝ) (ingrid_income : ℝ) (combined_tax_rate : ℝ)
  (jt_30 : john_tax_rate = 0.30) (it_40 : ingrid_tax_rate = 0.40) (ii_72000 : ingrid_income = 72000) 
  (ctr_35625 : combined_tax_rate = 0.35625) :
  ∃ J : ℝ, (0.30 * J + ingrid_tax_rate * ingrid_income = combined_tax_rate * (J + ingrid_income)) ∧ (J = 56000) :=
by
  sorry

end john_income_l1197_119743


namespace taxi_fare_distance_condition_l1197_119758

theorem taxi_fare_distance_condition (x : ℝ) (h1 : 7 + (max (x - 3) 0) * 2.4 = 19) : x ≤ 8 := 
by
  sorry

end taxi_fare_distance_condition_l1197_119758


namespace increase_in_volume_eq_l1197_119727

theorem increase_in_volume_eq (x : ℝ) (l w h : ℝ) (h₀ : l = 6) (h₁ : w = 4) (h₂ : h = 5) :
  (6 + x) * 4 * 5 = 6 * 4 * (5 + x) :=
by
  sorry

end increase_in_volume_eq_l1197_119727


namespace paint_cost_contribution_l1197_119701

theorem paint_cost_contribution
  (paint_cost_per_gallon : ℕ) 
  (coverage_per_gallon : ℕ) 
  (total_wall_area : ℕ) 
  (two_coats : ℕ) 
  : paint_cost_per_gallon = 45 → coverage_per_gallon = 400 → total_wall_area = 1600 → two_coats = 2 → 
    ((total_wall_area / coverage_per_gallon) * two_coats * paint_cost_per_gallon) / 2 = 180 :=
by
  intros h1 h2 h3 h4
  sorry

end paint_cost_contribution_l1197_119701


namespace nurses_count_l1197_119789

theorem nurses_count (D N : ℕ) (h1 : D + N = 456) (h2 : D * 11 = 8 * N) : N = 264 :=
by
  sorry

end nurses_count_l1197_119789


namespace Brittany_age_after_vacation_l1197_119776

variable (Rebecca_age Brittany_age vacation_years : ℕ)
variable (h1 : Rebecca_age = 25)
variable (h2 : Brittany_age = Rebecca_age + 3)
variable (h3 : vacation_years = 4)

theorem Brittany_age_after_vacation : 
    Brittany_age + vacation_years = 32 := 
by
  sorry

end Brittany_age_after_vacation_l1197_119776


namespace johns_speed_final_push_l1197_119712

-- Definitions for the given conditions
def john_behind_steve : ℝ := 14
def steve_speed : ℝ := 3.7
def john_ahead_steve : ℝ := 2
def john_final_push_time : ℝ := 32

-- Proving the statement
theorem johns_speed_final_push : 
  (∃ (v : ℝ), v * john_final_push_time = steve_speed * john_final_push_time + john_behind_steve + john_ahead_steve) -> 
  ∃ (v : ℝ), v = 4.2 :=
by
  sorry

end johns_speed_final_push_l1197_119712


namespace total_people_in_club_after_5_years_l1197_119715

noncomputable def club_initial_people := 18
noncomputable def executives_per_year := 6
noncomputable def initial_regular_members := club_initial_people - executives_per_year

-- Define the function for regular members growth
noncomputable def regular_members_after_n_years (n : ℕ) : ℕ := initial_regular_members * 2 ^ n

-- Total people in the club after 5 years
theorem total_people_in_club_after_5_years : 
  club_initial_people + regular_members_after_n_years 5 - initial_regular_members = 390 :=
by
  sorry

end total_people_in_club_after_5_years_l1197_119715


namespace tile_count_l1197_119750

theorem tile_count (room_length room_width tile_length tile_width : ℝ)
  (h1 : room_length = 10)
  (h2 : room_width = 15)
  (h3 : tile_length = 1 / 4)
  (h4 : tile_width = 3 / 4) :
  (room_length * room_width) / (tile_length * tile_width) = 800 :=
by
  sorry

end tile_count_l1197_119750


namespace natural_number_pairs_int_l1197_119734

theorem natural_number_pairs_int {
  a b : ℕ
} : 
  (∃ a b : ℕ, 
    (b^2 - a ≠ 0 ∧ (a^2 + b) % (b^2 - a) = 0) ∧ 
    (a^2 - b ≠ 0 ∧ (b^2 + a) % (a^2 - b) = 0)
  ) ↔ ((a, b) = (1, 2) ∨ (a, b) = (2, 1) ∨ (a, b) = (2, 2) ∨ (a, b) = (2, 3) ∨ (a, b) = (3, 2) ∨ (a, b) = (3, 3)) :=
by sorry

end natural_number_pairs_int_l1197_119734


namespace ellipse_h_k_a_c_sum_l1197_119737

theorem ellipse_h_k_a_c_sum :
  let h := -3
  let k := 1
  let a := 4
  let c := 2
  h + k + a + c = 4 :=
by
  let h := -3
  let k := 1
  let a := 4
  let c := 2
  show h + k + a + c = 4
  sorry

end ellipse_h_k_a_c_sum_l1197_119737


namespace prob_part1_prob_part2_l1197_119720

-- Define the probability that Person A hits the target
def pA : ℚ := 2 / 3

-- Define the probability that Person B hits the target
def pB : ℚ := 3 / 4

-- Define the number of shots
def nShotsA : ℕ := 3
def nShotsB : ℕ := 2

-- The problem posed to Person A
def probA_miss_at_least_once : ℚ := 1 - (pA ^ nShotsA)

-- The problem posed to Person A (exactly twice in 2 shots)
def probA_hits_exactly_twice : ℚ := pA ^ 2

-- The problem posed to Person B (exactly once in 2 shots)
def probB_hits_exactly_once : ℚ :=
  2 * (pB * (1 - pB))

-- The combined probability for Part 2
def combined_prob : ℚ := probA_hits_exactly_twice * probB_hits_exactly_once

theorem prob_part1 :
  probA_miss_at_least_once = 19 / 27 := by
  sorry

theorem prob_part2 :
  combined_prob = 1 / 6 := by
  sorry

end prob_part1_prob_part2_l1197_119720


namespace greatest_odd_integer_l1197_119754

theorem greatest_odd_integer (x : ℕ) (h_odd : x % 2 = 1) (h_pos : x > 0) (h_ineq : x^2 < 50) : x = 7 :=
by sorry

end greatest_odd_integer_l1197_119754


namespace female_students_count_l1197_119706

theorem female_students_count 
  (total_average : ℝ)
  (male_count : ℕ)
  (male_average : ℝ)
  (female_average : ℝ)
  (female_count : ℕ) 
  (correct_female_count : female_count = 12)
  (h1 : total_average = 90)
  (h2 : male_count = 8)
  (h3 : male_average = 87)
  (h4 : female_average = 92) :
  total_average * (male_count + female_count) = male_count * male_average + female_count * female_average :=
by sorry

end female_students_count_l1197_119706


namespace hannah_sweatshirts_l1197_119782

theorem hannah_sweatshirts (S : ℕ) (h1 : 15 * S + 2 * 10 = 65) : S = 3 := 
by
  sorry

end hannah_sweatshirts_l1197_119782


namespace quadratic_eq_has_two_distinct_real_roots_l1197_119703

theorem quadratic_eq_has_two_distinct_real_roots (m : ℝ) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (∀ x : ℝ, x^2 - 2*m*x - m - 1 = 0 ↔ x = x1 ∨ x = x2) :=
by
  sorry

end quadratic_eq_has_two_distinct_real_roots_l1197_119703


namespace total_cost_of_vacation_l1197_119738

variable (C : ℚ)

def cost_per_person_divided_among_3 := C / 3
def cost_per_person_divided_among_4 := C / 4
def per_person_difference := 40

theorem total_cost_of_vacation
  (h : cost_per_person_divided_among_3 C - cost_per_person_divided_among_4 C = per_person_difference) :
  C = 480 := by
  sorry

end total_cost_of_vacation_l1197_119738


namespace find_a_minus_b_l1197_119799

theorem find_a_minus_b
  (f : ℝ → ℝ)
  (a b : ℝ)
  (hf : ∀ x, f x = x^2 + 3 * a * x + 4)
  (h_even : ∀ x, f (-x) = f x)
  (hb_condition : b - 3 = -2 * b) :
  a - b = -1 :=
sorry

end find_a_minus_b_l1197_119799


namespace necessary_but_not_sufficient_l1197_119732

theorem necessary_but_not_sufficient (a : ℝ) : 
  (a > 2 → a^2 > 2 * a) ∧ (a^2 > 2 * a → (a > 2 ∨ a < 0)) :=
by
  sorry

end necessary_but_not_sufficient_l1197_119732


namespace total_laces_needed_l1197_119792

variable (x : ℕ) -- Eva has x pairs of shoes
def long_laces_per_pair : ℕ := 3
def short_laces_per_pair : ℕ := 3
def laces_per_pair : ℕ := long_laces_per_pair + short_laces_per_pair

theorem total_laces_needed : 6 * x = 6 * x :=
by
  have h : laces_per_pair = 6 := rfl
  sorry

end total_laces_needed_l1197_119792


namespace find_common_ratio_l1197_119795

noncomputable def geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 5 - a 1 = 15) ∧ (a 4 - a 2 = 6) → (q = 1/2 ∨ q = 2)

-- We declare this as a theorem statement
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) : geometric_sequence_common_ratio a q :=
sorry

end find_common_ratio_l1197_119795


namespace sale_price_relative_to_original_l1197_119728

variable (x : ℝ)

def increased_price (x : ℝ) := 1.30 * x
def sale_price (increased_price : ℝ) := 0.90 * increased_price

theorem sale_price_relative_to_original (x : ℝ) :
  sale_price (increased_price x) = 1.17 * x :=
by
  sorry

end sale_price_relative_to_original_l1197_119728


namespace abs_diff_squares_104_98_l1197_119751

theorem abs_diff_squares_104_98 : abs ((104 : ℤ)^2 - (98 : ℤ)^2) = 1212 := by
  sorry

end abs_diff_squares_104_98_l1197_119751


namespace twins_ages_sum_equals_20_l1197_119730

def sum_of_ages (A K : ℕ) := 2 * A + K

theorem twins_ages_sum_equals_20 (A K : ℕ) (h1 : A = A) (h2 : A * A * K = 256) : 
  sum_of_ages A K = 20 :=
by
  sorry

end twins_ages_sum_equals_20_l1197_119730


namespace find_triplets_l1197_119704

theorem find_triplets (x y z : ℕ) :
  (x^2 + y^2 = 3 * 2016^z + 77) →
  (x, y, z) = (77, 14, 1) ∨ (x, y, z) = (14, 77, 1) ∨ 
  (x, y, z) = (70, 35, 1) ∨ (x, y, z) = (35, 70, 1) ∨ 
  (x, y, z) = (8, 4, 0) ∨ (x, y, z) = (4, 8, 0) :=
by
  sorry

end find_triplets_l1197_119704


namespace speed_of_man_rowing_upstream_l1197_119713

theorem speed_of_man_rowing_upstream (V_m V_downstream V_upstream : ℝ) 
  (H1 : V_m = 60) 
  (H2 : V_downstream = 65) 
  (H3 : V_upstream = V_m - (V_downstream - V_m)) : 
  V_upstream = 55 := 
by 
  subst H1 
  subst H2 
  rw [H3] 
  norm_num

end speed_of_man_rowing_upstream_l1197_119713


namespace percentage_of_men_speaking_french_l1197_119790

theorem percentage_of_men_speaking_french {total_employees men women french_speaking_employees french_speaking_women french_speaking_men : ℕ}
    (h1 : total_employees = 100)
    (h2 : men = 60)
    (h3 : women = 40)
    (h4 : french_speaking_employees = 50)
    (h5 : french_speaking_women = 14)
    (h6 : french_speaking_men = french_speaking_employees - french_speaking_women)
    (h7 : french_speaking_men * 100 / men = 60) : true :=
by
  sorry

end percentage_of_men_speaking_french_l1197_119790


namespace AM_GM_inequality_l1197_119733

theorem AM_GM_inequality (a : List ℝ) (h : ∀ x ∈ a, 0 < x) :
  (a.sum / a.length) ≥ a.prod ^ (1 / a.length) := 
sorry

end AM_GM_inequality_l1197_119733


namespace polynomial_factorization_l1197_119778

theorem polynomial_factorization (m : ℝ) : 
  (∀ x : ℝ, (x - 7) * (x + 5) = x^2 + mx - 35) → m = -2 :=
by
  sorry

end polynomial_factorization_l1197_119778


namespace base_8_to_base_10_2671_to_1465_l1197_119721

theorem base_8_to_base_10_2671_to_1465 :
  (2 * 8^3 + 6 * 8^2 + 7 * 8^1 + 1 * 8^0) = 1465 := by
  sorry

end base_8_to_base_10_2671_to_1465_l1197_119721


namespace round_robin_10_players_l1197_119748

theorem round_robin_10_players : @Nat.choose 10 2 = 45 := by
  sorry

end round_robin_10_players_l1197_119748


namespace marble_probability_l1197_119774

theorem marble_probability (W G R B : ℕ) (h_total : W + G + R + B = 84) 
  (h_white : W / 84 = 1 / 4) (h_green : G / 84 = 1 / 7) :
  (R + B) / 84 = 17 / 28 :=
by
  sorry

end marble_probability_l1197_119774


namespace correct_factorization_l1197_119742

theorem correct_factorization:
  (∃ a : ℝ, (a + 3) * (a - 3) = a ^ 2 - 9) ∧
  (∃ x : ℝ, x ^ 2 + x - 5 = x * (x + 1) - 5) ∧
  ¬ (∃ x : ℝ, x ^ 2 + 1 = x * (x + 1 / x)) ∧
  (∃ x : ℝ, x ^ 2 + 4 * x + 4 = (x + 2) ^ 2) →
  (∃ x : ℝ, x ^ 2 + 4 * x + 4 = (x + 2) ^ 2)
  := by
  sorry

end correct_factorization_l1197_119742


namespace no_consecutive_days_played_l1197_119705

theorem no_consecutive_days_played (john_interval mary_interval : ℕ) :
  john_interval = 16 ∧ mary_interval = 25 → 
  ¬ ∃ (n : ℕ), (n * john_interval + 1 = m * mary_interval ∨ n * john_interval = m * mary_interval + 1) :=
by
  sorry

end no_consecutive_days_played_l1197_119705


namespace zoe_spent_amount_l1197_119793

theorem zoe_spent_amount :
  (3 * (8 + 2) = 30) :=
by sorry

end zoe_spent_amount_l1197_119793


namespace min_sum_of_M_and_N_l1197_119796

noncomputable def Alice (x : ℕ) : ℕ := 3 * x + 2
noncomputable def Bob (x : ℕ) : ℕ := 2 * x + 27

-- Define the result after 4 moves
noncomputable def Alice_4_moves (M : ℕ) : ℕ := Alice (Alice (Alice (Alice M)))
noncomputable def Bob_4_moves (N : ℕ) : ℕ := Bob (Bob (Bob (Bob N)))

theorem min_sum_of_M_and_N :
  ∃ (M N : ℕ), Alice_4_moves M = Bob_4_moves N ∧ M + N = 10 :=
sorry

end min_sum_of_M_and_N_l1197_119796


namespace minimum_benches_for_equal_occupancy_l1197_119760

theorem minimum_benches_for_equal_occupancy (M : ℕ) :
  (∃ x y, x = y ∧ 8 * M = x ∧ 12 * M = y) ↔ M = 3 := by
  sorry

end minimum_benches_for_equal_occupancy_l1197_119760


namespace martin_total_distance_l1197_119769

-- Define the conditions
def total_trip_time : ℕ := 8
def first_half_speed : ℕ := 70
def second_half_speed : ℕ := 85
def half_trip_time : ℕ := total_trip_time / 2

-- Define the total distance traveled 
def total_distance : ℕ := (first_half_speed * half_trip_time) + (second_half_speed * half_trip_time)

-- Statement to prove
theorem martin_total_distance : total_distance = 620 :=
by
  -- This is a placeholder to represent that a proof is needed
  -- Actual proof steps are omitted as instructed
  sorry

end martin_total_distance_l1197_119769


namespace find_y_l1197_119766

theorem find_y (y : ℝ) (h : |2 * y - 44| + |y - 24| = |3 * y - 66|) : y = 23 := 
by 
  sorry

end find_y_l1197_119766
