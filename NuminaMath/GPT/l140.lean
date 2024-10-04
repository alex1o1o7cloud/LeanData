import Mathlib

namespace total_paintable_wall_area_l140_140483

/-- 
  Conditions:
  - John's house has 4 bedrooms.
  - Each bedroom is 15 feet long, 12 feet wide, and 10 feet high.
  - Doorways, windows, and a fireplace occupy 85 square feet per bedroom.
  Question: Prove that the total paintable wall area is 1820 square feet.
--/
theorem total_paintable_wall_area 
  (num_bedrooms : ℕ)
  (length width height non_paintable_area : ℕ)
  (h_num_bedrooms : num_bedrooms = 4)
  (h_length : length = 15)
  (h_width : width = 12)
  (h_height : height = 10)
  (h_non_paintable_area : non_paintable_area = 85) :
  (num_bedrooms * ((2 * (length * height) + 2 * (width * height)) - non_paintable_area) = 1820) :=
by
  sorry

end total_paintable_wall_area_l140_140483


namespace find_k_l140_140683

noncomputable def digit_sum (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem find_k :
  ∃ k : ℕ, digit_sum (5 * (5 * (10 ^ (k - 1) - 1) / 9)) = 600 ∧ k = 87 :=
by
  sorry

end find_k_l140_140683


namespace sum_seven_consecutive_integers_l140_140109

theorem sum_seven_consecutive_integers (n : ℕ) : 
  ∃ k : ℕ, (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)) = 7 * k := 
by 
  -- Use sum of integers and factor to demonstrate that the sum is multiple of 7
  sorry

end sum_seven_consecutive_integers_l140_140109


namespace smallest_integer_y_l140_140124

theorem smallest_integer_y (y : ℤ) (h: y < 3 * y - 15) : y ≥ 8 :=
by sorry

end smallest_integer_y_l140_140124


namespace trout_ratio_l140_140836

theorem trout_ratio (caleb_trouts dad_trouts : ℕ) (h_c : caleb_trouts = 2) (h_d : dad_trouts = caleb_trouts + 4) :
  dad_trouts / (Nat.gcd dad_trouts caleb_trouts) = 3 ∧ caleb_trouts / (Nat.gcd dad_trouts caleb_trouts) = 1 :=
by
  sorry

end trout_ratio_l140_140836


namespace division_result_l140_140642

theorem division_result : (5 * 6 + 4) / 8 = 4.25 :=
by
  sorry

end division_result_l140_140642


namespace tyrone_gave_15_marbles_l140_140118

variables (x : ℕ)

-- Define initial conditions for Tyrone and Eric
def initial_tyrone := 120
def initial_eric := 20

-- Define the condition after giving marbles
def condition_after_giving (x : ℕ) := 120 - x = 3 * (20 + x)

theorem tyrone_gave_15_marbles (x : ℕ) : condition_after_giving x → x = 15 :=
by
  intro h
  sorry

end tyrone_gave_15_marbles_l140_140118


namespace simplify_fraction_mul_l140_140757

theorem simplify_fraction_mul (a b c d : ℕ) (h1 : a = 210) (h2 : b = 7350) (h3 : c = 1) (h4 : d = 35) (h5 : 210 / gcd 210 7350 = 1) (h6: 7350 / gcd 210 7350 = 35) :
  (a / b) * 14 = 2 / 5 :=
by
  sorry

end simplify_fraction_mul_l140_140757


namespace percent_markdown_l140_140175

theorem percent_markdown (P S : ℝ) (h : S * 1.25 = P) : (P - S) / P * 100 = 20 := by
  sorry

end percent_markdown_l140_140175


namespace customers_not_wanting_change_l140_140670

-- Given Conditions
def cars_initial := 4
def cars_additional := 6
def cars_total := cars_initial + cars_additional
def tires_per_car := 4
def half_change_customers := 2
def tires_for_half_change_customers := 2 * 2 -- 2 cars, 2 tires each
def tires_left := 20

-- Theorem to Prove
theorem customers_not_wanting_change : 
  (cars_total * tires_per_car) - (tires_left + tires_for_half_change_customers) = 
  4 * tires_per_car -> 
  cars_total - ((tires_left + tires_for_half_change_customers) / tires_per_car) - half_change_customers = 4 :=
by
  sorry

end customers_not_wanting_change_l140_140670


namespace number_of_customers_who_did_not_want_tires_change_l140_140672

noncomputable def total_cars_in_shop : Nat := 4 + 6
noncomputable def tires_per_car : Nat := 4
noncomputable def total_tires_bought : Nat := total_cars_in_shop * tires_per_car
noncomputable def half_tires_left : Nat := 2 * (tires_per_car / 2)
noncomputable def total_half_tires_left : Nat := 2 * half_tires_left
noncomputable def tires_left_after_half : Nat := 20
noncomputable def tires_left_after_half_customers : Nat := tires_left_after_half - total_half_tires_left
noncomputable def customers_who_did_not_change_tires : Nat := tires_left_after_half_customers / tires_per_car

theorem number_of_customers_who_did_not_want_tires_change : 
  customers_who_did_not_change_tires = 4 :=
by
  sorry 

end number_of_customers_who_did_not_want_tires_change_l140_140672


namespace coeffs_sum_of_binomial_expansion_l140_140028

theorem coeffs_sum_of_binomial_expansion :
  (3 * x - 2) ^ 6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 →
  a_0 = 64 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = -63 :=
by
  sorry

end coeffs_sum_of_binomial_expansion_l140_140028


namespace peach_pies_l140_140354

theorem peach_pies (total_pies : ℕ) (apple_ratio blueberry_ratio peach_ratio : ℕ)
  (h_ratio : apple_ratio + blueberry_ratio + peach_ratio = 10)
  (h_total : total_pies = 30)
  (h_ratios : apple_ratio = 3 ∧ blueberry_ratio = 2 ∧ peach_ratio = 5) :
  total_pies / (apple_ratio + blueberry_ratio + peach_ratio) * peach_ratio = 15 :=
by
  sorry

end peach_pies_l140_140354


namespace point_on_x_axis_l140_140220

theorem point_on_x_axis (m : ℤ) (hx : 2 + m = 0) : (m - 3, 2 + m) = (-5, 0) :=
by sorry

end point_on_x_axis_l140_140220


namespace mass_percentage_iodine_neq_662_l140_140853

theorem mass_percentage_iodine_neq_662 (atomic_mass_Al : ℝ) (atomic_mass_I : ℝ) (molar_mass_AlI3 : ℝ) :
  atomic_mass_Al = 26.98 ∧ atomic_mass_I = 126.90 ∧ molar_mass_AlI3 = ((1 * atomic_mass_Al) + (3 * atomic_mass_I)) →
  (3 * atomic_mass_I / molar_mass_AlI3 * 100) ≠ 6.62 :=
by
  sorry

end mass_percentage_iodine_neq_662_l140_140853


namespace radius_of_sphere_is_approximately_correct_l140_140147

noncomputable def radius_of_sphere_in_cylinder_cone : ℝ :=
  let radius_cylinder := 12
  let height_cylinder := 30
  let radius_sphere := 21 - 0.5 * Real.sqrt (30^2 + 12^2)
  radius_sphere

theorem radius_of_sphere_is_approximately_correct : abs (radius_of_sphere_in_cylinder_cone - 4.84) < 0.01 :=
by
  sorry

end radius_of_sphere_is_approximately_correct_l140_140147


namespace appropriate_chart_for_temperature_statistics_l140_140787

theorem appropriate_chart_for_temperature_statistics (chart_type : String) (is_line_chart : chart_type = "line chart") : chart_type = "line chart" :=
by
  sorry

end appropriate_chart_for_temperature_statistics_l140_140787


namespace height_of_prism_l140_140947

-- Definitions based on conditions
def Volume : ℝ := 120
def edge1 : ℝ := 3
def edge2 : ℝ := 4
def BaseArea : ℝ := edge1 * edge2

-- Define the problem statement
theorem height_of_prism (h : ℝ) : (BaseArea * h / 2 = Volume) → (h = 20) :=
by
  intro h_value
  have Volume_equiv : h = 2 * Volume / BaseArea := sorry
  sorry

end height_of_prism_l140_140947


namespace partnership_profit_l140_140396

noncomputable def totalProfit (P Q R : ℕ) (unit_value_per_share : ℕ) : ℕ :=
  let profit_p := 36 * 2 + 18 * 10
  let profit_q := 24 * 12
  let profit_r := 36 * 12
  (profit_p + profit_q + profit_r) * unit_value_per_share

theorem partnership_profit (P Q R : ℕ) (unit_value_per_share : ℕ) :
  (P / Q = 3 / 2) → (Q / R = 4 / 3) → 
  (unit_value_per_share = 144 / 288) → 
  totalProfit P Q R (unit_value_per_share * 1) = 486 := 
by
  intros h1 h2 h3
  sorry

end partnership_profit_l140_140396


namespace election_votes_l140_140345

theorem election_votes (V : ℝ) (h1 : ∃ geoff_votes : ℝ, geoff_votes = 0.01 * V)
                       (h2 : ∀ candidate_votes : ℝ, (candidate_votes > 0.51 * V) → candidate_votes > 0.51 * V)
                       (h3 : ∃ needed_votes : ℝ, needed_votes = 3000 ∧ 0.01 * V + needed_votes = 0.51 * V) :
                       V = 6000 :=
by sorry

end election_votes_l140_140345


namespace janessa_gives_dexter_cards_l140_140738

def initial_cards : Nat := 4
def father_cards : Nat := 13
def ordered_cards : Nat := 36
def bad_cards : Nat := 4
def kept_cards : Nat := 20

theorem janessa_gives_dexter_cards :
  initial_cards + father_cards + ordered_cards - bad_cards - kept_cards = 29 := 
by
  sorry

end janessa_gives_dexter_cards_l140_140738


namespace cost_per_mile_l140_140827

theorem cost_per_mile 
    (round_trip_distance : ℝ)
    (num_days : ℕ)
    (total_cost : ℝ) 
    (h1 : round_trip_distance = 200 * 2)
    (h2 : num_days = 7)
    (h3 : total_cost = 7000) 
  : (total_cost / (round_trip_distance * num_days) = 2.5) :=
by
  sorry

end cost_per_mile_l140_140827


namespace smallest_four_digit_divisible_by_33_l140_140520

theorem smallest_four_digit_divisible_by_33 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 33 = 0 ∧ n = 1023 := by 
  sorry

end smallest_four_digit_divisible_by_33_l140_140520


namespace tax_rate_for_remaining_l140_140885

variable (total_earnings deductions first_tax_rate total_tax taxed_amount remaining_taxable_income rem_tax_rate : ℝ)

def taxable_income (total_earnings deductions : ℝ) := total_earnings - deductions

def tax_on_first_portion (portion tax_rate : ℝ) := portion * tax_rate

def remaining_taxable (total_taxable first_portion : ℝ) := total_taxable - first_portion

def total_tax_payable (tax_first tax_remaining : ℝ) := tax_first + tax_remaining

theorem tax_rate_for_remaining :
  total_earnings = 100000 ∧ 
  deductions = 30000 ∧ 
  first_tax_rate = 0.10 ∧
  total_tax = 12000 ∧
  tax_on_first_portion 20000 first_tax_rate = 2000 ∧
  taxed_amount = 2000 ∧
  remaining_taxable_income = taxable_income total_earnings deductions - 20000 ∧
  total_tax_payable taxed_amount (remaining_taxable_income * rem_tax_rate) = total_tax →
  rem_tax_rate = 0.20 := 
sorry

end tax_rate_for_remaining_l140_140885


namespace original_number_abc_l140_140593

theorem original_number_abc (a b c : ℕ)
  (h : 100 * a + 10 * b + c = 528)
  (N : ℕ)
  (h1 : N + (100 * a + 10 * b + c) = 222 * (a + b + c))
  (hN : N = 2670) :
  100 * a + 10 * b + c = 528 := by
  sorry

end original_number_abc_l140_140593


namespace scenario1_ways_scenario2_ways_scenario3_ways_scenario4_ways_l140_140977

-- Scenario 1: All five balls into four distinct boxes
theorem scenario1_ways : 4^5 = 1024 := by sorry

-- Scenario 2: Each of the four distinct boxes receives one ball
theorem scenario2_ways : ∀ (n : ℕ), (nat.perm 5 4) = 120 := by sorry

-- Scenario 3: Four out of the five balls are placed into one of the four boxes (the other ball is not placed)
theorem scenario3_ways : ∀ (n : ℕ), (nat.choose 5 4) * (nat.choose 4 1) = 20 := by sorry 

-- Scenario 4: All five balls into four distinct boxes with no box left empty
theorem scenario4_ways : ∀ (n : ℕ), (nat.choose 5 2) * (nat.perm 4 4) = 240 := by sorry 

end scenario1_ways_scenario2_ways_scenario3_ways_scenario4_ways_l140_140977


namespace part1_part2_l140_140460

-- Part 1
noncomputable def f (x a : ℝ) : ℝ := (x - 1) * Real.exp x - (1/3) * a * x ^ 3 - (1/2) * x ^ 2

noncomputable def f' (x a : ℝ) : ℝ := x * Real.exp x - a * x ^ 2 - x

noncomputable def g (x a : ℝ) : ℝ := f' x a / x

theorem part1 (a : ℝ) (h : a > 0) : g a a > 0 := by
  sorry

-- Part 2
theorem part2 (a : ℝ) (h : ∃ x, f' x a = 0) : a > 0 := by
  sorry

end part1_part2_l140_140460


namespace cube_surface_area_increase_l140_140923

theorem cube_surface_area_increase (s : ℝ) : 
  let original_surface_area := 6 * s^2
  let new_edge := 1.3 * s
  let new_surface_area := 6 * (new_edge)^2
  let percentage_increase := ((new_surface_area - original_surface_area) / original_surface_area) * 100
  percentage_increase = 69 := 
by
  sorry

end cube_surface_area_increase_l140_140923


namespace compare_f_values_l140_140583

noncomputable def f : ℝ → ℝ := sorry

def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def is_monotonically_decreasing_on_nonnegative (f : ℝ → ℝ) :=
  ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 ≠ x2 → x1 < x2 → f x2 < f x1

axiom even_property : is_even_function f
axiom decreasing_property : is_monotonically_decreasing_on_nonnegative f

theorem compare_f_values : f 3 < f (-2) ∧ f (-2) < f 1 :=
by {
  sorry
}

end compare_f_values_l140_140583


namespace four_digit_numbers_divisible_by_11_and_5_with_sum_12_l140_140046

theorem four_digit_numbers_divisible_by_11_and_5_with_sum_12:
  ∀ a b c d : ℕ, (a + b + c + d = 12) ∧ ((a + c) - (b + d)) % 11 = 0 ∧ (d = 0 ∨ d = 5) →
  false :=
by
  intro a b c d
  intro h
  sorry

end four_digit_numbers_divisible_by_11_and_5_with_sum_12_l140_140046


namespace symmetry_about_origin_l140_140728

theorem symmetry_about_origin (m : ℝ) (A B : ℝ × ℝ) (hA : A = (2, -1)) (hB : B = (-2, m)) (h_sym : B = (-A.1, -A.2)) :
  m = 1 :=
by
  sorry

end symmetry_about_origin_l140_140728


namespace solve_linear_system_l140_140417

theorem solve_linear_system :
  ∃ x y : ℚ, (3 * x - y = 4) ∧ (6 * x - 3 * y = 10) ∧ (x = 2 / 3) ∧ (y = -2) :=
by
  sorry

end solve_linear_system_l140_140417


namespace inequality_proof_l140_140190

theorem inequality_proof
  (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a^2 + b^2) * (b^2 + c^2) * (c^2 + d^2) * (d^2 + a^2) ≥ 
  64 * a * b * c * d * abs ((a - b) * (b - c) * (c - d) * (d - a)) := 
by
  sorry

end inequality_proof_l140_140190


namespace land_value_moon_l140_140906

-- Define the conditions
def surface_area_earth : ℕ := 200
def surface_area_ratio : ℕ := 5
def value_ratio : ℕ := 6
def total_value_earth : ℕ := 80

-- Define the question and the expected answer
noncomputable def total_value_moon : ℕ := 96

-- State the proof problem
theorem land_value_moon :
  (surface_area_earth / surface_area_ratio * value_ratio) * (surface_area_earth / surface_area_ratio) = total_value_moon := 
sorry

end land_value_moon_l140_140906


namespace least_positive_multiple_of_primes_l140_140640

theorem least_positive_multiple_of_primes :
  11 * 13 * 17 * 19 = 46189 :=
by
  sorry

end least_positive_multiple_of_primes_l140_140640


namespace sachin_age_l140_140397

variable {S R : ℕ}

theorem sachin_age
  (h1 : R = S + 7)
  (h2 : S * 3 = 2 * R) :
  S = 14 :=
sorry

end sachin_age_l140_140397


namespace correct_survey_option_l140_140948

-- Definitions for survey options
inductive SurveyOption
| A
| B
| C
| D

-- Predicate that checks if an option is suitable for a comprehensive survey method
def suitable_for_comprehensive_survey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.A => false
  | SurveyOption.B => false
  | SurveyOption.C => false
  | SurveyOption.D => true

-- Theorem statement
theorem correct_survey_option : suitable_for_comprehensive_survey SurveyOption.D := 
  by sorry

end correct_survey_option_l140_140948


namespace parabola_coefficients_sum_l140_140380

theorem parabola_coefficients_sum (a b c : ℝ)
  (h_eqn : ∀ y, (-1) = a * y^2 + b * y + c)
  (h_vertex : (-1, -10) = (-a/(2*a), (4*a*c - b^2)/(4*a)))
  (h_pass_point : 0 = a * (-9)^2 + b * (-9) + c) 
  : a + b + c = 120 := 
sorry

end parabola_coefficients_sum_l140_140380


namespace convex_polygons_count_l140_140710

def binomial (n k : ℕ) : ℕ := if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def count_convex_polygons_with_two_acute_angles (m n : ℕ) : ℕ :=
  if 4 < m ∧ m < n then
    (2 * n + 1) * (binomial (n + 1) (m - 1) + binomial n (m - 1))
  else 0

theorem convex_polygons_count (m n : ℕ) (h : 4 < m ∧ m < n) :
  count_convex_polygons_with_two_acute_angles m n = 
  (2 * n + 1) * (binomial (n + 1) (m - 1) + binomial n (m - 1)) :=
by sorry

end convex_polygons_count_l140_140710


namespace vertex_angle_of_isosceles_with_angle_30_l140_140203

def isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ c = a) ∧ a + b + c = 180

theorem vertex_angle_of_isosceles_with_angle_30 (a b c : ℝ) 
  (ha : isosceles_triangle a b c) 
  (h1 : a = 30 ∨ b = 30 ∨ c = 30) :
  (a = 30 ∨ b = 30 ∨ c = 30) ∨ (a = 120 ∨ b = 120 ∨ c = 120) := 
sorry

end vertex_angle_of_isosceles_with_angle_30_l140_140203


namespace fraction_to_decimal_l140_140972

theorem fraction_to_decimal (a b : ℕ) (h₀ : a = 49) (h₁ : b = 160) : a / b = 0.30625 :=
by {
  -- Assume the given conditions
  assume h₀ : a = 49,
  assume h₁ : b = 160,
  -- Prove the theorem
  sorry
}

end fraction_to_decimal_l140_140972


namespace sum_of_35_consecutive_squares_div_by_35_l140_140499

def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_35_consecutive_squares_div_by_35 (n : ℕ) :
  (sum_of_squares (n + 35) - sum_of_squares n) % 35 = 0 :=
by
  sorry

end sum_of_35_consecutive_squares_div_by_35_l140_140499


namespace smallest_n_such_that_floor_eq_1989_l140_140187

theorem smallest_n_such_that_floor_eq_1989 :
  ∃ (n : ℕ), (∀ k, k < n -> ¬(∃ x : ℤ, ⌊(10^k : ℚ) / x⌋ = 1989)) ∧ (∃ x : ℤ, ⌊(10^n : ℚ) / x⌋ = 1989) :=
sorry

end smallest_n_such_that_floor_eq_1989_l140_140187


namespace moon_land_value_l140_140909

theorem moon_land_value (surface_area_earth : ℕ) (surface_area_moon : ℕ) (total_value_earth : ℕ) (worth_factor : ℕ)
  (h_moon_surface_area : surface_area_moon = surface_area_earth / 5)
  (h_surface_area_earth : surface_area_earth = 200) 
  (h_worth_factor : worth_factor = 6) 
  (h_total_value_earth : total_value_earth = 80) : (total_value_earth / 5) * worth_factor = 96 := 
by 
  -- Simplify using the given conditions
  -- total_value_earth / 5 is the value of the moon's land if it had the same value per square acre as Earth's land
  -- multiplying by worth_factor to get the total value on the moon
  sorry

end moon_land_value_l140_140909


namespace quadratic_roots_l140_140186

-- Definitions based on problem conditions
def sum_of_roots (p q : ℝ) : Prop := p + q = 12
def abs_diff_of_roots (p q : ℝ) : Prop := |p - q| = 4

-- The theorem we want to prove
theorem quadratic_roots : ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ p q, sum_of_roots p q ∧ abs_diff_of_roots p q → a * (x - p) * (x - q) = x^2 - 12 * x + 32) := sorry

end quadratic_roots_l140_140186


namespace probability_of_all_selected_l140_140116

theorem probability_of_all_selected :
  let p_x := 1 / 7
  let p_y := 2 / 9
  let p_z := 3 / 11
  p_x * p_y * p_z = 1 / 115.5 :=
by
  let p_x := 1 / 7
  let p_y := 2 / 9
  let p_z := 3 / 11
  sorry

end probability_of_all_selected_l140_140116


namespace number_of_customers_who_did_not_want_tires_change_l140_140673

noncomputable def total_cars_in_shop : Nat := 4 + 6
noncomputable def tires_per_car : Nat := 4
noncomputable def total_tires_bought : Nat := total_cars_in_shop * tires_per_car
noncomputable def half_tires_left : Nat := 2 * (tires_per_car / 2)
noncomputable def total_half_tires_left : Nat := 2 * half_tires_left
noncomputable def tires_left_after_half : Nat := 20
noncomputable def tires_left_after_half_customers : Nat := tires_left_after_half - total_half_tires_left
noncomputable def customers_who_did_not_change_tires : Nat := tires_left_after_half_customers / tires_per_car

theorem number_of_customers_who_did_not_want_tires_change : 
  customers_who_did_not_change_tires = 4 :=
by
  sorry 

end number_of_customers_who_did_not_want_tires_change_l140_140673


namespace number_of_boys_in_class_l140_140624

theorem number_of_boys_in_class
  (g_ratio : ℕ) (b_ratio : ℕ) (total_students : ℕ)
  (h_ratio : g_ratio / b_ratio = 4 / 3)
  (h_total_students : g_ratio + b_ratio = 7 * (total_students / 56)) :
  total_students = 56 → 3 * (total_students / (4 + 3)) = 24 :=
by
  intros total_students_56
  sorry

end number_of_boys_in_class_l140_140624


namespace value_of_a_l140_140065

theorem value_of_a (a : ℝ) (x : ℝ) (h : 2 * x + 3 * a = -1) (hx : x = 1) : a = -1 :=
by
  sorry

end value_of_a_l140_140065


namespace library_hospital_community_center_bells_ring_together_l140_140659

theorem library_hospital_community_center_bells_ring_together :
  ∀ (library hospital community : ℕ), 
    (library = 18) → (hospital = 24) → (community = 30) → 
    (∀ t, (t = 0) ∨ (∃ n₁ n₂ n₃ : ℕ, 
      t = n₁ * library ∧ t = n₂ * hospital ∧ t = n₃ * community)) → 
    true :=
by
  intros
  sorry

end library_hospital_community_center_bells_ring_together_l140_140659


namespace percent_defective_shipped_l140_140079

theorem percent_defective_shipped
  (P_d : ℝ) (P_s : ℝ)
  (hP_d : P_d = 0.1)
  (hP_s : P_s = 0.05) :
  P_d * P_s = 0.005 :=
by
  sorry

end percent_defective_shipped_l140_140079


namespace trees_falling_count_l140_140826

/-- Definition of the conditions of the problem. --/
def initial_mahogany_trees : ℕ := 50
def initial_narra_trees : ℕ := 30
def trees_on_farm_after_typhoon : ℕ := 88

/-- The mathematical proof problem statement in Lean 4:
Prove the total number of trees that fell during the typhoon (N + M) is equal to 5,
given the conditions.
--/
theorem trees_falling_count (M N : ℕ) 
  (h1 : M = N + 1)
  (h2 : (initial_mahogany_trees - M + 3 * M) + (initial_narra_trees - N + 2 * N) = trees_on_farm_after_typhoon) :
  N + M = 5 := sorry

end trees_falling_count_l140_140826


namespace jamal_books_remaining_l140_140480

variable (initial_books : ℕ := 51)
variable (history_books : ℕ := 12)
variable (fiction_books : ℕ := 19)
variable (children_books : ℕ := 8)
variable (misplaced_books : ℕ := 4)

theorem jamal_books_remaining : 
  initial_books - history_books - fiction_books - children_books + misplaced_books = 16 := by
  sorry

end jamal_books_remaining_l140_140480


namespace car_rental_cost_eq_800_l140_140089

-- Define the number of people
def num_people : ℕ := 8

-- Define the cost of the Airbnb rental
def airbnb_cost : ℕ := 3200

-- Define each person's share
def share_per_person : ℕ := 500

-- Define the total contribution of all people
def total_contribution : ℕ := num_people * share_per_person

-- Define the car rental cost
def car_rental_cost : ℕ := total_contribution - airbnb_cost

-- State the theorem to be proved
theorem car_rental_cost_eq_800 : car_rental_cost = 800 :=
  by sorry

end car_rental_cost_eq_800_l140_140089


namespace sum_of_products_l140_140726

variable (a b c : ℝ)

theorem sum_of_products (h1 : a^2 + b^2 + c^2 = 250) (h2 : a + b + c = 16) : 
  ab + bc + ca = 3 :=
sorry

end sum_of_products_l140_140726


namespace initial_ratio_milk_water_l140_140229

theorem initial_ratio_milk_water (M W : ℕ) 
  (h1 : M + W = 60) 
  (h2 : ∀ k, k = M → M * 2 = W + 60) : (M:ℚ) / (W:ℚ) = 4 / 1 :=
by
  sorry

end initial_ratio_milk_water_l140_140229


namespace ordered_pair_solution_l140_140633

-- Definitions for the problem
def v1 := (3 : ℝ, -1 : ℝ)
def v2 := (0 : ℝ, 2 : ℝ)
def u1 := (8 : ℝ, -3 : ℝ)
def u2 := (-1 : ℝ, 4 : ℝ)
def x := -15 / 29
def y := 33 / 29

theorem ordered_pair_solution :
  v1 + x • u1 = v2 + y • u2 :=
by
  -- Here should be the proof which we're not providing as per instructions
  sorry

end ordered_pair_solution_l140_140633


namespace target_more_tools_l140_140121

theorem target_more_tools (walmart_tools : ℕ) (target_tools : ℕ) (walmart_tools_is_6 : walmart_tools = 6) (target_tools_is_11 : target_tools = 11) :
  target_tools - walmart_tools = 5 :=
by
  rw [walmart_tools_is_6, target_tools_is_11]
  exact rfl

end target_more_tools_l140_140121


namespace men_complete_units_per_day_l140_140110

noncomputable def UnitsCompletedByMen (total_units : ℕ) (units_by_women : ℕ) : ℕ :=
  total_units - units_by_women

theorem men_complete_units_per_day :
  UnitsCompletedByMen 12 3 = 9 := by
  -- Proof skipped
  sorry

end men_complete_units_per_day_l140_140110


namespace equation_one_solution_equation_two_solution_l140_140759

theorem equation_one_solution (x : ℝ) : ((x + 3) ^ 2 - 9 = 0) ↔ (x = 0 ∨ x = -6) := by
  sorry

theorem equation_two_solution (x : ℝ) : (x ^ 2 - 4 * x + 1 = 0) ↔ (x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) := by
  sorry

end equation_one_solution_equation_two_solution_l140_140759


namespace profitWednesday_l140_140144

-- Define the total profit
def totalProfit : ℝ := 1200

-- Define the profit made on Monday
def profitMonday : ℝ := totalProfit / 3

-- Define the profit made on Tuesday
def profitTuesday : ℝ := totalProfit / 4

-- Theorem to prove the profit made on Wednesday
theorem profitWednesday : 
  let profitWednesday := totalProfit - (profitMonday + profitTuesday)
  profitWednesday = 500 :=
by
  -- proof goes here
  sorry

end profitWednesday_l140_140144


namespace minimize_sum_of_squares_l140_140556

open Real

-- Assume x, y are positive real numbers and x + y = s
variables {x y s : ℝ}
variables (hx_pos : 0 < x) (hy_pos : 0 < y) (h_sum : x + y = s)

theorem minimize_sum_of_squares :
  (x = y) ∧ (2 * x * x = s * s / 2) → (x = s / 2 ∧ y = s / 2 ∧ x^2 + y^2 = s^2 / 2) :=
by
  sorry

end minimize_sum_of_squares_l140_140556


namespace option_B_correct_l140_140164

theorem option_B_correct (x m : ℕ) : (x^3)^m / (x^m)^2 = x^m := sorry

end option_B_correct_l140_140164


namespace rotations_per_block_l140_140867

/--
If Greg's bike wheels have already rotated 600 times and need to rotate 
1000 more times to reach his goal of riding at least 8 blocks,
then the number of rotations per block is 200.
-/
theorem rotations_per_block (r1 r2 n b : ℕ) (h1 : r1 = 600) (h2 : r2 = 1000) (h3 : n = 8) :
  (r1 + r2) / n = 200 := by
  sorry

end rotations_per_block_l140_140867


namespace desired_percentage_of_alcohol_l140_140142

theorem desired_percentage_of_alcohol 
  (original_volume : ℝ)
  (original_percentage : ℝ)
  (added_volume : ℝ)
  (added_percentage : ℝ)
  (final_percentage : ℝ) :
  original_volume = 6 →
  original_percentage = 0.35 →
  added_volume = 1.8 →
  added_percentage = 1.0 →
  final_percentage = 50 :=
by
  intros h1 h2 h3 h4
  sorry

end desired_percentage_of_alcohol_l140_140142


namespace range_of_a_l140_140570

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : ∀ n : ℕ, a_seq n = a + n - 1)
  (h2 : ∀ n : ℕ, b n = (1 + a_seq n) / a_seq n)
  (h3 : ∀ n : ℕ, n > 0 → b n ≤ b 5) :
  -4 < a ∧ a < -3 :=
by
  sorry

end range_of_a_l140_140570


namespace target_has_more_tools_l140_140120

-- Define the number of tools in the Walmart multitool
def walmart_screwdriver : ℕ := 1
def walmart_knives : ℕ := 3
def walmart_other_tools : ℕ := 2
def walmart_total_tools : ℕ := walmart_screwdriver + walmart_knives + walmart_other_tools

-- Define the number of tools in the Target multitool
def target_screwdriver : ℕ := 1
def target_knives : ℕ := 2 * walmart_knives
def target_files_scissors : ℕ := 3 + 1
def target_total_tools : ℕ := target_screwdriver + target_knives + target_files_scissors

-- The theorem stating the difference in the number of tools
theorem target_has_more_tools : (target_total_tools - walmart_total_tools) = 6 := by
  sorry

end target_has_more_tools_l140_140120


namespace cylinder_intersection_in_sphere_l140_140755

theorem cylinder_intersection_in_sphere
  (a b c d e f : ℝ)
  (x y z : ℝ)
  (h1 : (x - a)^2 + (y - b)^2 < 1)
  (h2 : (y - c)^2 + (z - d)^2 < 1)
  (h3 : (z - e)^2 + (x - f)^2 < 1) :
  (x - (a + f) / 2)^2 + (y - (b + c) / 2)^2 + (z - (d + e) / 2)^2 < 3 / 2 := 
sorry

end cylinder_intersection_in_sphere_l140_140755


namespace find_d_l140_140855

-- Define the six-digit number as a function of d
def six_digit_num (d : ℕ) : ℕ := 3 * 100000 + 2 * 10000 + 5 * 1000 + 4 * 100 + 7 * 10 + d

-- Define the sum of digits of the six-digit number
def sum_of_digits (d : ℕ) : ℕ := 3 + 2 + 5 + 4 + 7 + d

-- The statement we want to prove
theorem find_d (d : ℕ) : sum_of_digits d % 3 = 0 ↔ d = 3 :=
by
  sorry

end find_d_l140_140855


namespace polynomial_binomial_square_l140_140062

theorem polynomial_binomial_square (b : ℚ) :
  (∃ c : ℚ, (3 * polynomial.X + polynomial.C c)^2 = 9 * polynomial.X^2 + 27 * polynomial.X + polynomial.C b) →
  b = 81 / 4 :=
by
  intro h
  rcases h with ⟨c, hc⟩
  have : 6 * c = 27 := by sorry -- This corresponds to solving 6c = 27
  have : c = 9 / 2 := by sorry -- This follows from the above
  have : b = (9 / 2)^2 := by sorry -- This follows from substituting back c and expanding
  simp [this]

end polynomial_binomial_square_l140_140062


namespace smallest_integer_cube_ends_in_576_l140_140976

theorem smallest_integer_cube_ends_in_576 : ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 576 ∧ ∀ m : ℕ, m > 0 → m^3 % 1000 = 576 → m ≥ n := 
by
  sorry

end smallest_integer_cube_ends_in_576_l140_140976


namespace student_good_probability_l140_140347

-- Defining the conditions as given in the problem
def P_A1 := 0.25          -- Probability of selecting a student from School A
def P_A2 := 0.4           -- Probability of selecting a student from School B
def P_A3 := 0.35          -- Probability of selecting a student from School C

def P_B_given_A1 := 0.3   -- Probability that a student's level is good given they are from School A
def P_B_given_A2 := 0.6   -- Probability that a student's level is good given they are from School B
def P_B_given_A3 := 0.5   -- Probability that a student's level is good given they are from School C

-- Main theorem statement
theorem student_good_probability : 
  P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3 = 0.49 := 
by sorry

end student_good_probability_l140_140347


namespace sum_irreducible_fractions_not_integer_l140_140268

theorem sum_irreducible_fractions_not_integer {a b c d : ℕ} (h1 : Int.gcd a b = 1) (h2 : Int.gcd c d = 1) (h3 : b ≠ d) :
  ¬ ∃ k : ℤ, a * d + c * b = k * b * d :=
by
  sorry

end sum_irreducible_fractions_not_integer_l140_140268


namespace ray_nickels_left_l140_140260

theorem ray_nickels_left (h1 : 285 % 5 = 0) (h2 : 55 % 5 = 0) (h3 : 3 * 55 % 5 = 0) (h4 : 45 % 5 = 0) : 
  285 / 5 - ((55 / 5) + (3 * 55 / 5) + (45 / 5)) = 4 := sorry

end ray_nickels_left_l140_140260


namespace simplify_and_evaluate_l140_140369

-- Define the condition as a predicate
def condition (a b : ℝ) : Prop := (a + 1/2)^2 + |b - 2| = 0

-- The simplified expression
def simplified_expression (a b : ℝ) : ℝ := 12 * a^2 * b - 6 * a * b^2

-- Statement: Given the condition, prove that the simplified expression evaluates to 18
theorem simplify_and_evaluate : ∀ (a b : ℝ), condition a b → simplified_expression a b = 18 :=
by
  intros a b hc
  sorry  -- Proof omitted

end simplify_and_evaluate_l140_140369


namespace land_for_cattle_l140_140355

-- Define the conditions as Lean definitions
def total_land : ℕ := 150
def house_and_machinery : ℕ := 25
def future_expansion : ℕ := 15
def crop_production : ℕ := 70

-- Statement to prove
theorem land_for_cattle : total_land - (house_and_machinery + future_expansion + crop_production) = 40 :=
by
  sorry

end land_for_cattle_l140_140355


namespace gunther_cleaning_free_time_l140_140868

theorem gunther_cleaning_free_time :
  let vacuum := 45
  let dusting := 60
  let mopping := 30
  let bathroom := 40
  let windows := 15
  let brushing_per_cat := 5
  let cats := 4

  let free_time_hours := 4
  let free_time_minutes := 25

  let cleaning_time := vacuum + dusting + mopping + bathroom + windows + (brushing_per_cat * cats)
  let free_time_total := (free_time_hours * 60) + free_time_minutes

  free_time_total - cleaning_time = 55 :=
by
  sorry

end gunther_cleaning_free_time_l140_140868


namespace rhombus_diagonal_length_l140_140816

theorem rhombus_diagonal_length (side : ℝ) (shorter_diagonal : ℝ) 
  (h1 : side = 51) (h2 : shorter_diagonal = 48) : 
  ∃ longer_diagonal : ℝ, longer_diagonal = 90 :=
by
  sorry

end rhombus_diagonal_length_l140_140816


namespace polar_to_rect_eq_point_D_min_dist_l140_140233

noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem polar_to_rect_eq {ρ θ : ℝ} (hρ : ρ = 2 * Real.sin θ) : 
  ∃ x y : ℝ, (x, y) = polar_to_rectangular ρ θ ∧ x^2 + y^2 = 2 * y :=
by
  apply Exists.intro (ρ * Real.cos θ)
  apply Exists.intro (ρ * Real.sin θ)
  have h_rect : (ρ * Real.cos θ, ρ * Real.sin θ) = polar_to_rectangular ρ θ := rfl
  split
  . exact h_rect
  . rw [hρ, real_mul_commρ, Real.sin_sq, by sorry -- proof omitted for brevity]

theorem point_D_min_dist : 
  ∃ α : ℝ, let D := (Real.cos α, 1 + Real.sin α) in 
  α = π / 6 ∧ D = ( √3 / 2, 3 / 2) :=
by sorry -- proof omitted for brevity

end polar_to_rect_eq_point_D_min_dist_l140_140233


namespace no_perfect_square_l140_140598

theorem no_perfect_square (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h : ∃ (a : ℕ), p + q^2 = a^2) : ∀ (n : ℕ), n > 0 → ¬ (∃ (b : ℕ), p^2 + q^n = b^2) := 
by
  sorry

end no_perfect_square_l140_140598


namespace not_linear_eq_l140_140645

-- Representing the given equations
def eq1 (x : ℝ) : Prop := 5 * x + 3 = 3 * x - 7
def eq2 (x : ℝ) : Prop := 1 + 2 * x = 3
def eq4 (x : ℝ) : Prop := x - 7 = 0

-- The equation to verify if it's not linear
def eq3 (x : ℝ) : Prop := abs (2 * x) / 3 + 5 / x = 3

-- Stating the Lean statement to be proved
theorem not_linear_eq : ¬ (eq3 x) := by
  sorry

end not_linear_eq_l140_140645


namespace statistical_measure_mode_l140_140916

theorem statistical_measure_mode (fav_dishes : List ℕ) :
  (∀ measure, (measure = "most frequently occurring value" → measure = "Mode")) :=
by
  intro measure
  intro h
  sorry

end statistical_measure_mode_l140_140916


namespace black_balls_count_l140_140654

theorem black_balls_count :
  ∀ (r k : ℕ), r = 10 -> (2 : ℚ) / 7 = r / (r + k : ℚ) -> k = 25 := by
  intros r k hr hprob
  sorry

end black_balls_count_l140_140654


namespace count_selection_4_balls_count_selection_5_balls_score_at_least_7_points_l140_140663

-- Setup the basic context
def Pocket := Finset (Fin 11)

-- The pocket contains 4 red balls and 7 white balls
def red_balls : Finset (Fin 11) := {0, 1, 2, 3}
def white_balls : Finset (Fin 11) := {4, 5, 6, 7, 8, 9, 10}

-- Question 1
theorem count_selection_4_balls :
  (red_balls.card.choose 4) + (red_balls.card.choose 3 * white_balls.card.choose 1) +
  (red_balls.card.choose 2 * white_balls.card.choose 2) = 115 := 
sorry

-- Question 2
theorem count_selection_5_balls_score_at_least_7_points :
  (red_balls.card.choose 2 * white_balls.card.choose 3) +
  (red_balls.card.choose 3 * white_balls.card.choose 2) +
  (red_balls.card.choose 4 * white_balls.card.choose 1) = 301 := 
sorry

end count_selection_4_balls_count_selection_5_balls_score_at_least_7_points_l140_140663


namespace ratio_of_areas_l140_140619

theorem ratio_of_areas (AB CD AH BG CF DG S_ABCD S_KLMN : ℕ)
  (h1 : AB = 15)
  (h2 : CD = 19)
  (h3 : DG = 17)
  (condition1 : S_ABCD = 17 * (AH + BG))
  (midpoints_AH_CF : AH = BG)
  (midpoints_CF_CD : CF = CD/2)
  (condition2 : (∃ h₁ h₂ : ℕ, S_KLMN = h₁ * AH + h₂ * CF / 2))
  (h_case1 : (S_KLMN = (AH + BG + CD)))
  (h_case2 : (S_KLMN = (AB + (CD - DG)))) :
  (S_ABCD / S_KLMN = 2 / 3 ∨ S_ABCD / S_KLMN = 2) :=
  sorry

end ratio_of_areas_l140_140619


namespace least_number_correct_l140_140929

def least_number_to_add_to_make_perfect_square (x : ℝ) : ℝ :=
  let y := 1 - x -- since 1 is the smallest whole number > sqrt(0.0320)
  y

theorem least_number_correct (x : ℝ) (h : x = 0.0320) : least_number_to_add_to_make_perfect_square x = 0.9680 :=
by {
  -- Proof is skipped
  -- The proof would involve verifying that adding this number to x results in a perfect square (1 in this case).
  sorry
}

end least_number_correct_l140_140929


namespace max_value_proof_l140_140612

noncomputable def max_value (x y : ℝ) : ℝ := x^2 + 2 * x * y + 3 * y^2

theorem max_value_proof (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - 2 * x * y + 3 * y^2 = 12) : 
  max_value x y = 24 + 12 * Real.sqrt 3 := 
sorry

end max_value_proof_l140_140612


namespace symmetric_about_pi_over_4_l140_140904

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x + Real.cos x

theorem symmetric_about_pi_over_4 (a : ℝ) :
  (∀ x : ℝ, f a (x + π / 4) = f a (-(x + π / 4))) → a = 1 := by
  unfold f
  sorry

end symmetric_about_pi_over_4_l140_140904


namespace root_of_unity_product_l140_140339

theorem root_of_unity_product (ω : ℂ) (h1 : ω^3 = 1) (h2 : ω ≠ 1) :
  (1 - ω + ω^2) * (1 + ω - ω^2) = 1 :=
  sorry

end root_of_unity_product_l140_140339


namespace passes_through_fixed_point_l140_140643

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a^(x-2) - 3

theorem passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 = -2 :=
by
  sorry

end passes_through_fixed_point_l140_140643


namespace find_sticker_price_l140_140211

-- Define the conditions and the question
def storeA_price (x : ℝ) : ℝ := 0.80 * x - 80
def storeB_price (x : ℝ) : ℝ := 0.70 * x - 40
def heather_saves_30 (x : ℝ) : Prop := storeA_price x = storeB_price x + 30

-- Define the main theorem
theorem find_sticker_price : ∃ x : ℝ, heather_saves_30 x ∧ x = 700 :=
by
  sorry

end find_sticker_price_l140_140211


namespace pentagon_probability_l140_140265

/-- Ten points are equally spaced around the circumference of a regular pentagon,
with each side being divided into two equal segments.

We need to prove that the probability of choosing two points randomly and
having them be exactly one side of the pentagon apart is 2/9.
-/
theorem pentagon_probability : 
  let total_points := 10
  let favorable_pairs := 10
  let total_pairs := total_points * (total_points - 1) / 2
  (favorable_pairs / total_pairs : ℚ) = 2 / 9 :=
by
  sorry

end pentagon_probability_l140_140265


namespace first_payment_amount_l140_140146

-- The number of total payments
def total_payments : Nat := 65

-- The number of the first payments
def first_payments : Nat := 20

-- The number of remaining payments
def remaining_payments : Nat := total_payments - first_payments

-- The extra amount added to the remaining payments
def extra_amount : Int := 65

-- The average payment
def average_payment : Int := 455

-- The total amount paid over the year
def total_amount_paid : Int := average_payment * total_payments

-- The variable we want to solve for: amount of each of the first 20 payments
variable (x : Int)

-- The equation for total amount paid
def total_payments_equation : Prop :=
  20 * x + 45 * (x + 65) = 455 * 65

-- The theorem stating the amount of each of the first 20 payments
theorem first_payment_amount : x = 410 :=
  sorry

end first_payment_amount_l140_140146


namespace percentage_good_oranges_tree_A_l140_140368

theorem percentage_good_oranges_tree_A
  (total_trees : ℕ)
  (trees_A : ℕ)
  (trees_B : ℕ)
  (total_good_oranges : ℕ)
  (oranges_A_per_month : ℕ) 
  (oranges_B_per_month : ℕ)
  (good_oranges_B_ratio : ℚ)
  (good_oranges_total_B : ℕ) 
  (good_oranges_total_A : ℕ)
  (good_oranges_total : ℕ)
  (x : ℚ) 
  (total_trees_eq : total_trees = 10)
  (tree_percentage_eq : trees_A = total_trees / 2 ∧ trees_B = total_trees / 2)
  (oranges_A_per_month_eq : oranges_A_per_month = 10)
  (oranges_B_per_month_eq : oranges_B_per_month = 15)
  (good_oranges_B_ratio_eq : good_oranges_B_ratio = 1/3)
  (good_oranges_total_eq : total_good_oranges = 55)
  (good_oranges_total_B_eq : good_oranges_total_B = trees_B * oranges_B_per_month * good_oranges_B_ratio)
  (good_oranges_total_A_eq : good_oranges_total_A = total_good_oranges - good_oranges_total_B):
  trees_A * oranges_A_per_month * x = good_oranges_total_A → 
  x = 0.6 := by
  sorry

end percentage_good_oranges_tree_A_l140_140368


namespace quadrilateral_area_l140_140435

theorem quadrilateral_area {d o1 o2 : ℝ} (hd : d = 15) (ho1 : o1 = 6) (ho2 : o2 = 4) :
  (d * (o1 + o2)) / 2 = 75 := by
  sorry

end quadrilateral_area_l140_140435


namespace length_of_bridge_is_l140_140158

noncomputable def train_length : ℝ := 100
noncomputable def time_to_cross_bridge : ℝ := 21.998240140788738
noncomputable def speed_kmph : ℝ := 36
noncomputable def speed_mps : ℝ := speed_kmph * (1000 / 3600)
noncomputable def total_distance : ℝ := speed_mps * time_to_cross_bridge
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem length_of_bridge_is : bridge_length = 119.98240140788738 :=
by
  have speed_mps_val : speed_mps = 10 := by
    norm_num [speed_kmph, speed_mps]
  have total_distance_val : total_distance = 219.98240140788738 := by
    norm_num [total_distance, speed_mps_val, time_to_cross_bridge]
  have bridge_length_val : bridge_length = 119.98240140788738 := by
    norm_num [bridge_length, total_distance_val, train_length]
  exact bridge_length_val

end length_of_bridge_is_l140_140158


namespace zero_points_of_function_l140_140426

theorem zero_points_of_function : 
  (∃ x y : ℝ, y = x - 4 / x ∧ y = 0) → (∃! x : ℝ, x = -2 ∨ x = 2) :=
by
  sorry

end zero_points_of_function_l140_140426


namespace longest_side_length_quadrilateral_l140_140171

theorem longest_side_length_quadrilateral :
  (∀ (x y : ℝ),
    (x + y ≤ 4) ∧
    (2 * x + y ≥ 3) ∧
    (x ≥ 0) ∧
    (y ≥ 0)) →
  (∃ d : ℝ, d = 4 * Real.sqrt 2) :=
by sorry

end longest_side_length_quadrilateral_l140_140171


namespace total_distance_traveled_l140_140873

theorem total_distance_traveled
  (bike_time_min : ℕ) (bike_rate_mph : ℕ)
  (jog_time_min : ℕ) (jog_rate_mph : ℕ)
  (total_time_min : ℕ)
  (h_bike_time : bike_time_min = 30)
  (h_bike_rate : bike_rate_mph = 6)
  (h_jog_time : jog_time_min = 45)
  (h_jog_rate : jog_rate_mph = 8)
  (h_total_time : total_time_min = 75) :
  (bike_rate_mph * bike_time_min / 60) + (jog_rate_mph * jog_time_min / 60) = 9 :=
by sorry

end total_distance_traveled_l140_140873


namespace suraj_average_l140_140762

theorem suraj_average : 
  ∀ (A : ℝ), 
    (16 * A + 92 = 17 * (A + 4)) → 
      (A + 4) = 28 :=
by
  sorry

end suraj_average_l140_140762


namespace vector_problem_l140_140994

noncomputable def t_value : ℝ :=
  (-5 - Real.sqrt 13) / 2

theorem vector_problem 
  (t : ℝ)
  (a : ℝ × ℝ := (1, 1))
  (b : ℝ × ℝ := (2, t))
  (h : Real.sqrt ((1 - 2)^2 + (1 - t)^2) = (1 * 2 + 1 * t)) :
  t = t_value := 
sorry

end vector_problem_l140_140994


namespace combined_tax_rate_l140_140955

theorem combined_tax_rate (Mork_income Mindy_income : ℝ) (Mork_tax_rate Mindy_tax_rate : ℝ)
  (h1 : Mork_tax_rate = 0.4) (h2 : Mindy_tax_rate = 0.3) (h3 : Mindy_income = 4 * Mork_income) :
  ((Mork_tax_rate * Mork_income + Mindy_tax_rate * Mindy_income) / (Mork_income + Mindy_income)) * 100 = 32 :=
by
  sorry

end combined_tax_rate_l140_140955


namespace perimeter_of_shaded_region_l140_140591

noncomputable def circle_center : Type := sorry -- Define the object type for circle's center
noncomputable def radius_length : ℝ := 10 -- Define the radius length as 10
noncomputable def central_angle : ℝ := 270 -- Define the central angle corresponding to the arc RS

-- Function to calculate the perimeter of the shaded region
noncomputable def perimeter_shaded_region (radius : ℝ) (angle : ℝ) : ℝ :=
  2 * radius + (angle / 360) * 2 * Real.pi * radius

-- Theorem stating that the perimeter of the shaded region is 20 + 15π given the conditions
theorem perimeter_of_shaded_region : 
  perimeter_shaded_region radius_length central_angle = 20 + 15 * Real.pi :=
by
  -- skipping the actual proof
  sorry

end perimeter_of_shaded_region_l140_140591


namespace arithmetic_seq_and_general_formula_find_Tn_l140_140445

-- Given definitions
def S : ℕ → ℕ := sorry
def a : ℕ → ℕ := sorry

-- Conditions
axiom a1 : a 1 = 1
axiom a2 : ∀ n : ℕ, n > 0 → n * S n.succ = (n+1) * S n + n^2 + n

-- Problem 1: Prove and derive general formula for Sₙ
theorem arithmetic_seq_and_general_formula (n : ℕ) (h : n > 0) :
  ∃ S : ℕ → ℕ, (∀ n : ℕ, n > 0 → (S (n+1)) / (n+1) - (S n) / n = 1) ∧ (S n = n^2) := sorry

-- Problem 2: Given bₙ and Tₙ, find Tₙ
def b (n : ℕ) : ℕ := 1 / (a n * a (n+1))
def T : ℕ → ℕ := sorry

axiom b1 : ∀ n : ℕ, n > 0 → b 1 = 1
axiom b2 : ∀ n : ℕ, n > 0 → T n = 1 / (2 * n + 1)

theorem find_Tn (n : ℕ) (h : n > 0) : T n = n / (2 * n + 1) := sorry

end arithmetic_seq_and_general_formula_find_Tn_l140_140445


namespace simplify_expr1_simplify_expr2_l140_140609

variable (x y : ℝ)

theorem simplify_expr1 : 
  3 * x^2 - 2 * x * y + y^2 - 3 * x^2 + 3 * x * y = x * y + y^2 :=
by
  sorry

theorem simplify_expr2 : 
  (7 * x^2 - 3 * x * y) - 6 * (x^2 - 1/3 * x * y) = x^2 - x * y :=
by
  sorry

end simplify_expr1_simplify_expr2_l140_140609


namespace slope_negative_l140_140592

theorem slope_negative (m : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → mx1 + 5 > mx2 + 5) → m < 0 :=
by
  sorry

end slope_negative_l140_140592


namespace find_x_l140_140077

-- Definitions based on the given conditions
variables {B C D : Type} (A : Type)

-- Angles in degrees
variables (angle_ACD : ℝ := 100)
variables (angle_ADB : ℝ)
variables (angle_ABD : ℝ := 2 * angle_ADB)
variables (angle_DAC : ℝ)
variables (angle_BAC : ℝ := angle_DAC)
variables (angle_ACB : ℝ := 180 - angle_ACD)
variables (y : ℝ := angle_DAC)
variables (x : ℝ := angle_ADB)

-- The proof statement
theorem find_x (h1 : B = C) (h2 : C = D) 
    (h3: angle_ACD = 100) 
    (h4: angle_ADB = x) 
    (h5: angle_ABD = 2 * x) 
    (h6: angle_DAC = angle_BAC) 
    (h7: angle_DAC = y)
    : x = 20 :=
sorry

end find_x_l140_140077


namespace unknown_number_is_10_l140_140872

def operation_e (x y : ℕ) : ℕ := 2 * x * y

theorem unknown_number_is_10 (n : ℕ) (h : operation_e 8 (operation_e n 5) = 640) : n = 10 :=
by
  sorry

end unknown_number_is_10_l140_140872


namespace problem_l140_140085

theorem problem (x y : ℕ) (hy : y > 3) (h : x^2 + y^4 = 2 * ((x-6)^2 + (y+1)^2)) : x^2 + y^4 = 1994 := by
  sorry

end problem_l140_140085


namespace emma_possible_lists_l140_140930

-- Define the number of balls
def number_of_balls : ℕ := 24

-- Define the number of draws Emma repeats independently
def number_of_draws : ℕ := 4

-- Define the calculation for the total number of different lists
def total_number_of_lists : ℕ := number_of_balls ^ number_of_draws

theorem emma_possible_lists : total_number_of_lists = 331776 := by
  sorry

end emma_possible_lists_l140_140930


namespace ratio_of_side_length_to_radius_l140_140400

theorem ratio_of_side_length_to_radius (r s : ℝ) (c d : ℝ) 
  (h1 : s = 2 * r)
  (h2 : s^2 = (c / d) * (s^2 - π * r^2)) : 
  (s / r) = (Real.sqrt (c * π) / Real.sqrt (d - c)) := by
  sorry

end ratio_of_side_length_to_radius_l140_140400


namespace number_division_l140_140525

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l140_140525


namespace unique_perpendicular_line_through_point_l140_140031

-- Definitions of the geometric entities and their relationships
structure Point := (x : ℝ) (y : ℝ)

structure Line := (m : ℝ) (b : ℝ)

-- A function to check if a point lies on a given line
def point_on_line (P : Point) (l : Line) : Prop := P.y = l.m * P.x + l.b

-- A function to represent that a line is perpendicular to another line at a given point
def perpendicular_lines_at_point (P : Point) (l1 l2 : Line) : Prop :=
  l1.m = -(1 / l2.m) ∧ point_on_line P l1 ∧ point_on_line P l2

-- The statement to be proved
theorem unique_perpendicular_line_through_point (P : Point) (l : Line) (h : point_on_line P l) :
  ∃! l' : Line, perpendicular_lines_at_point P l' l :=
by
  sorry

end unique_perpendicular_line_through_point_l140_140031


namespace customers_not_wanting_change_l140_140671

-- Given Conditions
def cars_initial := 4
def cars_additional := 6
def cars_total := cars_initial + cars_additional
def tires_per_car := 4
def half_change_customers := 2
def tires_for_half_change_customers := 2 * 2 -- 2 cars, 2 tires each
def tires_left := 20

-- Theorem to Prove
theorem customers_not_wanting_change : 
  (cars_total * tires_per_car) - (tires_left + tires_for_half_change_customers) = 
  4 * tires_per_car -> 
  cars_total - ((tires_left + tires_for_half_change_customers) / tires_per_car) - half_change_customers = 4 :=
by
  sorry

end customers_not_wanting_change_l140_140671


namespace delta_y_over_delta_x_l140_140415

def curve (x : ℝ) : ℝ := x^2 + x

theorem delta_y_over_delta_x (Δx Δy : ℝ) 
  (hQ : (2 + Δx, 6 + Δy) = (2 + Δx, curve (2 + Δx)))
  (hP : 6 = curve 2) : 
  (Δy / Δx) = Δx + 5 :=
by
  sorry

end delta_y_over_delta_x_l140_140415


namespace insert_arithmetic_sequence_l140_140881

theorem insert_arithmetic_sequence (d a b : ℤ) 
  (h1 : (-1) + 3 * d = 8) 
  (h2 : a = (-1) + d) 
  (h3 : b = a + d) : 
  a = 2 ∧ b = 5 := by
  sorry

end insert_arithmetic_sequence_l140_140881


namespace repeated_digit_in_mod_sequence_l140_140429

theorem repeated_digit_in_mod_sequence : 
  ∃ (x y : ℕ), x ≠ y ∧ (2^1970 % 9 = 4) ∧ 
  (∀ n : ℕ, n < 10 → n = 2^1970 % 9 → n = x ∨ n = y) :=
sorry

end repeated_digit_in_mod_sequence_l140_140429


namespace time_spent_on_type_a_l140_140804

theorem time_spent_on_type_a (num_questions : ℕ) 
                             (exam_duration : ℕ)
                             (type_a_count : ℕ)
                             (time_ratio : ℕ)
                             (type_b_count : ℕ)
                             (x : ℕ)
                             (total_time : ℕ) :
  num_questions = 200 ∧
  exam_duration = 180 ∧
  type_a_count = 20 ∧
  time_ratio = 2 ∧
  type_b_count = 180 ∧
  total_time = 36 →
  time_ratio * x * type_a_count + x * type_b_count = exam_duration →
  total_time = 36 :=
by
  sorry

end time_spent_on_type_a_l140_140804


namespace maximum_value_is_l140_140615

noncomputable def maximum_value (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x^2 - 2 * x * y + 3 * y^2 = 12) : ℝ :=
  x^2 + 2 * x * y + 3 * y^2

theorem maximum_value_is (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x^2 - 2 * x * y + 3 * y^2 = 12) :
  maximum_value x y h₁ h₂ h₃ ≤ 18 + 12 * Real.sqrt 3 :=
sorry

end maximum_value_is_l140_140615


namespace function_value_l140_140036

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem function_value (a b : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : log_base a (2 + b) = 1) (h₃ : log_base a (8 + b) = 2) : a + b = 4 :=
by
  sorry

end function_value_l140_140036


namespace correct_statements_count_l140_140617

/-- The Fibonacci sequence -/
def Fibonacci (n : ℕ) : ℕ :=
  Nat.fib (n + 1)

theorem correct_statements_count :
  (∃ m : ℕ, m > 0 ∧ (Fibonacci m + Fibonacci (m+2) = 2 * Fibonacci (m+1))) ∧
  ¬ (∃ m : ℕ, m > 0 ∧ (Fibonacci (m+1) * Fibonacci (m+1) = Fibonacci m * Fibonacci (m+2))) ∧
  (∃ t : ℚ, t = 3/2 ∧ ∀ n : ℕ, n > 0 → (Fibonacci n + Fibonacci (n+4) = 2 * t * Fibonacci (n+2))) ∧
  (∃ (i₁ i₂ : ℕ), 1≤i₁ ∧ i₁<i₂ ∧ (Fibonacci i₁ + Fibonacci i₂ = 2023)) →
  ∃ n : ℕ, n = 3 :=
sorry

end correct_statements_count_l140_140617


namespace cos_double_angle_l140_140049

theorem cos_double_angle
  {x : ℝ}
  (h : Real.sin x = -2 / 3) :
  Real.cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_l140_140049


namespace round_trip_percentage_l140_140805

-- Definitions based on the conditions
variable (P : ℝ) -- Total number of passengers
variable (R : ℝ) -- Number of round-trip ticket holders

-- First condition: 20% of passengers held round-trip tickets and took their cars aboard
def condition1 := 0.20 * P = 0.60 * R

-- Second condition: 40% of passengers with round-trip tickets did not take their cars aboard (implies 60% did)
theorem round_trip_percentage (h1 : condition1 P R) : (R / P) * 100 = 33.33 := by
  sorry

end round_trip_percentage_l140_140805


namespace savings_calculation_l140_140943

def price_per_window : ℕ := 120
def discount_offer (n : ℕ) : ℕ := if n ≥ 10 then 2 else 0

def george_needs : ℕ := 9
def anne_needs : ℕ := 11

def cost (n : ℕ) : ℕ :=
  let free_windows := discount_offer n
  (n - free_windows) * price_per_window

theorem savings_calculation :
  let total_separate_cost := cost george_needs + cost anne_needs
  let total_windows := george_needs + anne_needs
  let total_cost_together := cost total_windows
  total_separate_cost - total_cost_together = 240 :=
by
  sorry

end savings_calculation_l140_140943


namespace total_jokes_after_eight_days_l140_140753

def jokes_counted (start_jokes : ℕ) (n : ℕ) : ℕ :=
  -- Sum of initial jokes until the nth day by doubling each day
  start_jokes * (2 ^ n - 1)

theorem total_jokes_after_eight_days (jessy_jokes : ℕ) (alan_jokes : ℕ) (tom_jokes : ℕ) (emily_jokes : ℕ)
  (total_days : ℕ) (days_per_week : ℕ) :
  total_days = 5 → days_per_week = 8 →
  jessy_jokes = 11 → alan_jokes = 7 → tom_jokes = 5 → emily_jokes = 3 →
  (jokes_counted jessy_jokes (days_per_week - total_days) +
   jokes_counted alan_jokes (days_per_week - total_days) +
   jokes_counted tom_jokes (days_per_week - total_days) +
   jokes_counted emily_jokes (days_per_week - total_days)) = 806 :=
by
  intros
  sorry

end total_jokes_after_eight_days_l140_140753


namespace max_arithmetic_sum_l140_140920

def a1 : ℤ := 113
def d : ℤ := -4

def S (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem max_arithmetic_sum : S 29 = 1653 :=
by
  sorry

end max_arithmetic_sum_l140_140920


namespace sum_of_coefficients_l140_140842

theorem sum_of_coefficients (f : ℕ → ℕ) :
  (5 * 1 + 2)^7 = 823543 :=
by
  sorry

end sum_of_coefficients_l140_140842


namespace percentage_increase_l140_140931

theorem percentage_increase (initial final : ℝ)
  (h_initial: initial = 60) (h_final: final = 90) :
  (final - initial) / initial * 100 = 50 :=
by
  sorry

end percentage_increase_l140_140931


namespace larger_integer_l140_140280

theorem larger_integer (x y : ℕ) (h_diff : y - x = 8) (h_prod : x * y = 272) : y = 20 :=
by
  sorry

end larger_integer_l140_140280


namespace power_of_128_div_7_eq_16_l140_140958

theorem power_of_128_div_7_eq_16 : (128 : ℝ) ^ (4 / 7) = 16 := by
  sorry

end power_of_128_div_7_eq_16_l140_140958


namespace probability_sum_7_or_11_l140_140290

theorem probability_sum_7_or_11 (total_outcomes favorable_7 favorable_11 : ℕ) 
  (h1 : total_outcomes = 36) (h2 : favorable_7 = 6) (h3 : favorable_11 = 2) :
  (favorable_7 + favorable_11 : ℚ) / total_outcomes = 2 / 9 := by 
  sorry

end probability_sum_7_or_11_l140_140290


namespace solve_for_s_l140_140214

theorem solve_for_s (m : ℝ) (s : ℝ) 
  (h1 : 5 = m * 3^s) 
  (h2 : 45 = m * 9^s) : 
  s = 2 :=
sorry

end solve_for_s_l140_140214


namespace friends_came_over_later_l140_140939

def original_friends : ℕ := 4
def total_people : ℕ := 7

theorem friends_came_over_later : (total_people - original_friends = 3) :=
sorry

end friends_came_over_later_l140_140939


namespace power_comparison_l140_140128

theorem power_comparison (A B : ℝ) (h1 : A = 1997 ^ (1998 ^ 1999)) (h2 : B = 1999 ^ (1998 ^ 1997)) (h3 : 1997 < 1999) :
  A > B :=
by
  sorry

end power_comparison_l140_140128


namespace number_division_l140_140524

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l140_140524


namespace solve_for_Theta_l140_140183

-- Define the two-digit number representation condition
def fourTheta (Θ : ℕ) : ℕ := 40 + Θ

-- Main theorem statement
theorem solve_for_Theta (Θ : ℕ) (h1 : 198 / Θ = fourTheta Θ + Θ) (h2 : 0 < Θ ∧ Θ < 10) : Θ = 4 :=
by
  sorry

end solve_for_Theta_l140_140183


namespace prob_two_red_two_blue_is_3_over_14_l140_140298

def red_marbles : ℕ := 15
def blue_marbles : ℕ := 10
def total_marbles : ℕ := red_marbles + blue_marbles
def chosen_marbles : ℕ := 4

noncomputable def prob_two_red_two_blue : ℚ :=
  let total_ways := (Nat.choose total_marbles chosen_marbles : ℚ)
  let ways_two_red := (Nat.choose red_marbles 2)
  let ways_two_blue := (Nat.choose blue_marbles 2)
  let favorable_outcomes := 6 * ways_two_red * ways_two_blue
  favorable_outcomes / total_ways

theorem prob_two_red_two_blue_is_3_over_14 : prob_two_red_two_blue = 3 / 14 :=
  sorry

end prob_two_red_two_blue_is_3_over_14_l140_140298


namespace abs_nested_expression_l140_140360

theorem abs_nested_expression (x : ℝ) (h : x = 2023) : 
  abs (abs (abs x - x) - abs x) - x = 0 :=
by
  subst h
  sorry

end abs_nested_expression_l140_140360


namespace unit_digit_of_25_17_18_factorials_l140_140314

theorem unit_digit_of_25_17_18_factorials : (25! + 17! - 18!) % 10 = 0 := by
  sorry

end unit_digit_of_25_17_18_factorials_l140_140314


namespace simplify_fraction_l140_140829

-- Define factorial (or use the existing factorial definition if available in Mathlib)
def fact : ℕ → ℕ 
| 0       => 1
| (n + 1) => (n + 1) * fact n

-- Problem statement
theorem simplify_fraction :
  (5 * fact 7 + 35 * fact 6) / fact 8 = 5 / 4 := by
  sorry

end simplify_fraction_l140_140829


namespace distribute_coins_l140_140952

theorem distribute_coins (x y : ℕ) (h₁ : x + y = 16) (h₂ : x^2 - y^2 = 16 * (x - y)) :
  x = 8 ∧ y = 8 :=
by {
  sorry
}

end distribute_coins_l140_140952


namespace sequence_value_x_l140_140594

theorem sequence_value_x (x : ℕ) (h1 : 1 + 3 = 4) (h2 : 4 + 3 = 7) (h3 : 7 + 3 = 10) (h4 : 10 + 3 = x) (h5 : x + 3 = 16) : x = 13 := by
  sorry

end sequence_value_x_l140_140594


namespace ratio_of_volumes_l140_140964

theorem ratio_of_volumes (A B : ℝ) (h : (3 / 4) * A = (2 / 3) * B) : A / B = 8 / 9 :=
by
  sorry

end ratio_of_volumes_l140_140964


namespace cos_13pi_over_4_eq_neg_one_div_sqrt_two_l140_140179

noncomputable def cos_13pi_over_4 : Real :=
  Real.cos (13 * Real.pi / 4)

theorem cos_13pi_over_4_eq_neg_one_div_sqrt_two : 
  cos_13pi_over_4 = -1 / Real.sqrt 2 := by 
  sorry

end cos_13pi_over_4_eq_neg_one_div_sqrt_two_l140_140179


namespace sum_largest_and_smallest_l140_140113

-- Define the three-digit number properties
def hundreds_digit := 4
def tens_digit := 8
def A : ℕ := sorry  -- Placeholder for the digit A

-- Define the number based on the digits
def number (A : ℕ) : ℕ := 100 * hundreds_digit + 10 * tens_digit + A

-- Hypotheses
axiom A_range : 0 ≤ A ∧ A ≤ 9

-- Largest and smallest possible numbers
def largest_number := number 9
def smallest_number := number 0

-- Prove the sum
theorem sum_largest_and_smallest : largest_number + smallest_number = 969 :=
by
  sorry

end sum_largest_and_smallest_l140_140113


namespace arithmetic_sequence_common_difference_l140_140877

theorem arithmetic_sequence_common_difference
  (a : ℤ)
  (a_n : ℤ)
  (S_n : ℤ)
  (n : ℤ)
  (d : ℚ)
  (h1 : a = 3)
  (h2 : a_n = 34)
  (h3 : S_n = 222)
  (h4 : S_n = n * (a + a_n) / 2)
  (h5 : a_n = a + (n - 1) * d) :
  d = 31 / 11 :=
by
  sorry

end arithmetic_sequence_common_difference_l140_140877


namespace find_f_six_l140_140448

noncomputable def f : ℕ → ℤ := sorry

axiom f_one_eq_one : f 1 = 1
axiom f_add (x y : ℕ) : f (x + y) = f x + f y + 8 * x * y - 2
axiom f_seven_eq_163 : f 7 = 163

theorem find_f_six : f 6 = 116 := 
by {
  sorry
}

end find_f_six_l140_140448


namespace football_team_people_count_l140_140778

theorem football_team_people_count (original_count : ℕ) (new_members : ℕ) (total_count : ℕ) 
  (h1 : original_count = 36) (h2 : new_members = 14) : total_count = 50 :=
by
  -- This is where the proof would go. We write 'sorry' because it is not required.
  sorry

end football_team_people_count_l140_140778


namespace chairs_stools_legs_l140_140094

theorem chairs_stools_legs (x : ℕ) (h1 : 4 * x + 3 * (16 - x) = 60) : 4 * x + 3 * (16 - x) = 60 :=
by
  exact h1

end chairs_stools_legs_l140_140094


namespace find_original_number_l140_140934

theorem find_original_number (n a b: ℤ) 
  (h1 : n > 1000) 
  (h2 : n + 79 = a^2) 
  (h3 : n + 204 = b^2) 
  (h4 : b^2 - a^2 = 125) : 
  n = 3765 := 
by 
  sorry

end find_original_number_l140_140934


namespace math_problem_l140_140247

variable (a b c d : ℝ)
variable (h1 : a > b)
variable (h2 : c < d)

theorem math_problem : a - c > b - d :=
by {
  sorry
}

end math_problem_l140_140247


namespace arithmetic_sequence_sum_abs_values_l140_140197

theorem arithmetic_sequence_sum_abs_values (n : ℕ) (a : ℕ → ℤ)
  (h₁ : a 1 = 13)
  (h₂ : ∀ k, a (k + 1) = a k + (-4)) :
  T_n = if n ≤ 4 then 15 * n - 2 * n^2 else 2 * n^2 - 15 * n + 56 :=
by sorry

end arithmetic_sequence_sum_abs_values_l140_140197


namespace transformation_correctness_l140_140231

variable (x x' y y' : ℝ)

-- Conditions
def original_curve : Prop := y^2 = 4
def transformed_curve : Prop := (x'^2)/1 + (y'^2)/4 = 1
def transformation_formula : Prop := (x = 2 * x') ∧ (y = y')

-- Proof Statement
theorem transformation_correctness (h1 : original_curve y) (h2 : transformed_curve x' y') :
  transformation_formula x x' y y' :=
  sorry

end transformation_correctness_l140_140231


namespace total_amount_silver_l140_140477

theorem total_amount_silver (x y : ℝ) (h₁ : y = 7 * x + 4) (h₂ : y = 9 * x - 8) : y = 46 :=
by {
  sorry
}

end total_amount_silver_l140_140477


namespace display_stands_arrangements_l140_140111

theorem display_stands_arrangements :
  ∃ n : ℕ, n = 48 ∧
    let stands := finset.range 9 in
    let valid_stands := stands \ {0, 8} in
    ∃ (s1 s2 s3 : ℕ) (h1 : s1 ∈ valid_stands) (h2 : s2 ∈ valid_stands) (h3 : s3 ∈ valid_stands),
      nat.abs (s1 - s2) > 1 ∧ nat.abs (s1 - s2) <= 2 ∧
      nat.abs (s2 - s3) > 1 ∧ nat.abs (s2 - s3) <= 2 ∧
      nat.abs (s1 - s3) > 1 ∧ nat.abs (s1 - s3) <= 2 ∧
      finset.card {s1, s2, s3} = 3 := sorry

end display_stands_arrangements_l140_140111


namespace ratio_tuesday_monday_l140_140365

-- Define the conditions
variables (M T W : ℕ) (hM : M = 450) (hW : W = 300) (h_rel : W = T + 75)

-- Define the theorem
theorem ratio_tuesday_monday : (T : ℚ) / M = 1 / 2 :=
by
  -- Sorry means the proof has been omitted in Lean.
  sorry

end ratio_tuesday_monday_l140_140365


namespace dragon_jewels_end_l140_140149

-- Given conditions
variables (D : ℕ) (jewels_taken_by_king jewels_taken_from_king new_jewels final_jewels : ℕ)

-- Conditions corresponding to the problem
axiom h1 : jewels_taken_by_king = 3
axiom h2 : jewels_taken_from_king = 2 * jewels_taken_by_king
axiom h3 : new_jewels = jewels_taken_from_king
axiom h4 : new_jewels = D / 3

-- Equation derived from the problem setting
def number_of_jewels_initial := D
def number_of_jewels_after_king_stole := number_of_jewels_initial - jewels_taken_by_king
def number_of_jewels_final := number_of_jewels_after_king_stole + jewels_taken_from_king

-- Final proof obligation
theorem dragon_jewels_end : ∃ (D : ℕ), number_of_jewels_final D 3 6 = 21 :=
by
  sorry

end dragon_jewels_end_l140_140149


namespace sequence_converges_and_limit_l140_140170

theorem sequence_converges_and_limit {a : ℝ} (m : ℕ) (h_a_pos : 0 < a) (h_m_pos : 0 < m) :
  (∃ (x : ℕ → ℝ), 
  (x 1 = 1) ∧ 
  (x 2 = a) ∧ 
  (∀ n : ℕ, x (n + 2) = (x (n + 1) ^ m * x n) ^ (↑(1 : ℕ) / (m + 1))) ∧ 
  ∃ l : ℝ, (∀ ε > 0, ∃ N, ∀ n > N, |x n - l| < ε) ∧ l = a ^ (↑(m + 1) / ↑(m + 2))) :=
sorry

end sequence_converges_and_limit_l140_140170


namespace value_at_x12_l140_140416

def quadratic_function (d e f x : ℝ) : ℝ :=
  d * x^2 + e * x + f

def axis_of_symmetry (d e f : ℝ) : ℝ := 10.5

def point_on_graph (d e f : ℝ) : Prop :=
  quadratic_function d e f 3 = -5

theorem value_at_x12 (d e f : ℝ)
  (Hsymm : axis_of_symmetry d e f = 10.5)
  (Hpoint : point_on_graph d e f) :
  quadratic_function d e f 12 = -5 :=
sorry

end value_at_x12_l140_140416


namespace min_length_BC_l140_140398

theorem min_length_BC (A B C D : Type) (AB AC DC BD BC : ℝ) :
  AB = 8 → AC = 15 → DC = 10 → BD = 25 → (BC > AC - AB) ∧ (BC > BD - DC) → BC ≥ 15 :=
by
  intros hAB hAC hDC hBD hIneq
  sorry

end min_length_BC_l140_140398


namespace derivative_f_intervals_of_monotonicity_extrema_l140_140989

noncomputable def f (x : ℝ) := (x + 1)^2 * (x - 1)

theorem derivative_f (x : ℝ) : deriv f x = 3 * x^2 + 2 * x - 1 := sorry

theorem intervals_of_monotonicity :
  (∀ x, x < -1 → deriv f x > 0) ∧
  (∀ x, -1 < x ∧ x < -1/3 → deriv f x < 0) ∧
  (∀ x, x > -1/3 → deriv f x > 0) := sorry

theorem extrema :
  f (-1) = 0 ∧
  f (-1/3) = -(32 / 27) := sorry

end derivative_f_intervals_of_monotonicity_extrema_l140_140989


namespace find_a_value_l140_140468

theorem find_a_value (a x y : ℝ) (h1 : x = 4) (h2 : y = 5) (h3 : a * x - 2 * y = 2) : a = 3 :=
by
  sorry

end find_a_value_l140_140468


namespace fraction_division_l140_140833

-- Define the fractions and the operation result.
def complex_fraction := 5 / (8 / 15)
def result := 75 / 8

-- State the theorem indicating that these should be equal.
theorem fraction_division :
  complex_fraction = result :=
  by
  sorry

end fraction_division_l140_140833


namespace arithmetic_series_sum_l140_140441

theorem arithmetic_series_sum : 
  let a1 := -41
  let d := 2
  let n := 22 in
  let an := a1 + (n - 1) * d in
  let S := n / 2 * (a1 + an) in
  an = 1 ∧ S = -440 :=
by
  sorry

end arithmetic_series_sum_l140_140441


namespace actual_cost_of_article_l140_140395

theorem actual_cost_of_article (x : ℝ) (hx : 0.76 * x = 988) : x = 1300 :=
sorry

end actual_cost_of_article_l140_140395


namespace find_slope_intercept_l140_140155

def line_eqn (x y : ℝ) : Prop :=
  -3 * (x - 5) + 2 * (y + 1) = 0

theorem find_slope_intercept :
  ∃ (m b : ℝ), (∀ x y : ℝ, line_eqn x y → y = m * x + b) ∧ (m = 3/2) ∧ (b = -17/2) := sorry

end find_slope_intercept_l140_140155


namespace number_whose_square_is_64_l140_140106

theorem number_whose_square_is_64 (x : ℝ) (h : x^2 = 64) : x = 8 ∨ x = -8 :=
sorry

end number_whose_square_is_64_l140_140106


namespace problem_statement_l140_140742

theorem problem_statement
  (a b A B : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ θ : ℝ, f θ ≥ 0)
  (def_f : ∀ θ : ℝ, f θ = 1 - a * Real.cos θ - b * Real.sin θ - A * Real.cos (2 * θ) - B * Real.sin (2 * θ)) :
  a ^ 2 + b ^ 2 ≤ 2 ∧ A ^ 2 + B ^ 2 ≤ 1 := 
by
  sorry

end problem_statement_l140_140742


namespace expression_equals_500_l140_140794

theorem expression_equals_500 :
  let A := 5 * 99 + 1
  let B := 100 + 25 * 4
  let C := 88 * 4 + 37 * 4
  let D := 100 * 0 * 5
  C = 500 :=
by
  let A := 5 * 99 + 1
  let B := 100 + 25 * 4
  let C := 88 * 4 + 37 * 4
  let D := 100 * 0 * 5
  sorry

end expression_equals_500_l140_140794


namespace max_product_two_four_digit_numbers_l140_140919

theorem max_product_two_four_digit_numbers :
  ∃ (a b : ℕ), 
    (a * b = max (8564 * 7321) (8531 * 7642)) 
    ∧ max 8531 8564 = 8531 ∧ 
    (∀ x y : ℕ, x * y ≤ 8531 * 7642 → x * y = max (8564 * 7321) (8531 * 7642)) :=
sorry

end max_product_two_four_digit_numbers_l140_140919


namespace problem1_problem2_problem3_l140_140419

-- Problem 1
theorem problem1 : -2.8 + (-3.6) + 3 - (-3.6) = 0.2 := 
by
  sorry

-- Problem 2
theorem problem2 : (-4) ^ 2010 * (-0.25) ^ 2009 + (-12) * (1 / 3 - 3 / 4 + 5 / 6) = -9 := 
by
  sorry

-- Problem 3
theorem problem3 : 13 * (16/60 : ℝ) * 5 - 19 * (12/60 : ℝ) / 6 = 13 * (8/60 : ℝ) + 50 := 
by
  sorry

end problem1_problem2_problem3_l140_140419


namespace problem_l140_140705

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem problem (a b c : ℝ) (h0 : f a b c 0 = f a b c 4) (h1 : f a b c 0 > f a b c 1) :
  a > 0 ∧ 4 * a + b = 0 :=
by
  sorry

end problem_l140_140705


namespace enchilada_cost_l140_140133

theorem enchilada_cost : ∃ T E : ℝ, 2 * T + 3 * E = 7.80 ∧ 3 * T + 5 * E = 12.70 ∧ E = 2.00 :=
by
  sorry

end enchilada_cost_l140_140133


namespace cos_double_angle_example_l140_140057

def cos_double_angle_identity (x : ℝ) : Prop :=
  cos (2 * x) = 1 - 2 * (sin x) ^ 2

theorem cos_double_angle_example : cos_double_angle_identity (x : ℝ) 
  (h : sin x = - 2 / 3) : cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_example_l140_140057


namespace find_m_l140_140225

theorem find_m (m : ℝ) : (∀ x : ℝ, 0 < x → x < 2 → - (1/2)*x^2 + 2*x > -m*x) ↔ m = -1 := 
sorry

end find_m_l140_140225


namespace find_larger_integer_l140_140279

theorem find_larger_integer (a b : ℕ) (h₁ : a * b = 272) (h₂ : |a - b| = 8) : max a b = 17 :=
sorry

end find_larger_integer_l140_140279


namespace determine_true_proposition_l140_140602

def proposition_p : Prop :=
  ∃ x : ℝ, Real.tan x > 1

def proposition_q : Prop :=
  let focus_distance := 3/4 -- Distance from the focus to the directrix in y = (1/3)x^2
  focus_distance = 1/6

def true_proposition : Prop :=
  proposition_p ∧ ¬proposition_q

theorem determine_true_proposition :
  (proposition_p ∧ ¬proposition_q) = true_proposition :=
by
  sorry -- Proof will go here

end determine_true_proposition_l140_140602


namespace real_part_z_pow_2017_l140_140201

open Complex

noncomputable def z : ℂ := 1 + I

theorem real_part_z_pow_2017 : re (z ^ 2017) = 2 ^ 1008 := sorry

end real_part_z_pow_2017_l140_140201


namespace exists_composite_l140_140453

theorem exists_composite (x y : ℕ) (hx : 2 ≤ x ∧ x ≤ 100) (hy : 2 ≤ y ∧ y ≤ 100) :
  ∃ n : ℕ, ∃ k : ℕ, x^(2^n) + y^(2^n) = k * (k + 1) :=
by {
  sorry -- proof goes here
}

end exists_composite_l140_140453


namespace monotonic_increasing_condition_l140_140999

open Real

noncomputable def f (x : ℝ) (l a : ℝ) : ℝ := x^2 - x + l + a * log x

theorem monotonic_increasing_condition (l a : ℝ) (x : ℝ) (hx : x > 0) 
  (h : ∀ x, x > 0 → deriv (f l a) x ≥ 0) : 
  a > 1 / 8 :=
by
  sorry

end monotonic_increasing_condition_l140_140999


namespace negation_of_p_l140_140454

-- Define the proposition p
def p : Prop := ∃ n : ℕ, 2^n > 100

-- Goal is to show the negation of p
theorem negation_of_p : (¬ p) = (∀ n : ℕ, 2^n ≤ 100) :=
by
  sorry

end negation_of_p_l140_140454


namespace cos_A_of_triangle_l140_140343

theorem cos_A_of_triangle (a b c : ℝ) (A B C : ℝ) (h1 : b = Real.sqrt 2 * c)
  (h2 : Real.sin A + Real.sqrt 2 * Real.sin C = 2 * Real.sin B)
  (h3 : a = Real.sin A / Real.sin A * b) -- Sine rule used implicitly

: Real.cos A = Real.sqrt 2 / 4 := by
  -- proof will be skipped, hence 'sorry' included
  sorry

end cos_A_of_triangle_l140_140343


namespace find_m_l140_140205

theorem find_m (a : ℕ → ℤ) (S : ℕ → ℤ) (m : ℕ) 
  (hS : ∀ n, S n = n^2 - 6 * n) :
  (forall m, (5 < a m ∧ a m < 8) → m = 7)
:= 
by
  sorry

end find_m_l140_140205


namespace infinite_points_inside_circle_l140_140839

theorem infinite_points_inside_circle:
  ∀ c : ℝ, c = 3 → ∀ x y : ℚ, 0 < x ∧ 0 < y  ∧ x^2 + y^2 < 9 → ∃ a b : ℚ, 0 < a ∧ 0 < b ∧ a^2 + b^2 < 9 :=
sorry

end infinite_points_inside_circle_l140_140839


namespace polynomial_binomial_square_l140_140063

theorem polynomial_binomial_square (b : ℚ) :
  (∃ c : ℚ, (3 * polynomial.X + polynomial.C c)^2 = 9 * polynomial.X^2 + 27 * polynomial.X + polynomial.C b) →
  b = 81 / 4 :=
by
  intro h
  rcases h with ⟨c, hc⟩
  have : 6 * c = 27 := by sorry -- This corresponds to solving 6c = 27
  have : c = 9 / 2 := by sorry -- This follows from the above
  have : b = (9 / 2)^2 := by sorry -- This follows from substituting back c and expanding
  simp [this]

end polynomial_binomial_square_l140_140063


namespace tan_of_angle_l140_140982

open Real

-- Given conditions in the problem
variables {α : ℝ}

-- Define the given conditions
def sinα_condition (α : ℝ) : Prop := sin α = 3 / 5
def α_in_quadrant_2 (α : ℝ) : Prop := π / 2 < α ∧ α < π

-- Define the Lean statement
theorem tan_of_angle {α : ℝ} (h1 : sinα_condition α) (h2 : α_in_quadrant_2 α) :
  tan α = -3 / 4 :=
sorry

end tan_of_angle_l140_140982


namespace find_larger_integer_l140_140277

theorem find_larger_integer (a b : ℕ) (h₁ : a * b = 272) (h₂ : |a - b| = 8) : max a b = 17 :=
sorry

end find_larger_integer_l140_140277


namespace find_constants_and_extrema_l140_140029

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

theorem find_constants_and_extrema (a b c : ℝ) (h : a ≠ 0) 
    (ext1 : deriv (f a b c) 1 = 0) (ext2 : deriv (f a b c) (-1) = 0) (value1 : f a b c 1 = -1) :
    a = -1/2 ∧ b = 0 ∧ c = 1/2 ∧ 
    (∃ x : ℝ, x = 1 ∧ deriv (deriv (f a b c)) x < 0) ∧
    (∃ x : ℝ, x = -1 ∧ deriv (deriv (f a b c)) x > 0) :=
sorry

end find_constants_and_extrema_l140_140029


namespace solve_for_a_l140_140010

def g (x : ℝ) : ℝ := 5 * x - 6

theorem solve_for_a (a : ℝ) : g a = 4 → a = 2 := by
  sorry

end solve_for_a_l140_140010


namespace complex_power_identity_l140_140030

theorem complex_power_identity (i : ℂ) (hi : i^2 = -1) :
  ( (1 + i) / (1 - i) ) ^ 2013 = i :=
by sorry

end complex_power_identity_l140_140030


namespace limit_problem_l140_140216

theorem limit_problem
  {f : ℝ → ℝ} {x₀ : ℝ}
  (h_deriv : deriv f x₀ = -3) :
  (tendsto (λ h, (f (x₀ + h) - f (x₀ - 3 * h)) / h) (𝓝 0) (𝓝 (-12))) :=
by {
  sorry
}

end limit_problem_l140_140216


namespace radius_of_circle_l140_140375
open Real

theorem radius_of_circle (a b r : ℝ) (h1 : a + b = 6 * r) (h2 : 1/2 * a * b = 27) : r = 3 := by
  have area_eq : 1/2 * a * b = 3 * r^2 := by
    sorry
  have eq_r_squared : 3 * r^2 = 27 := by
    sorry
  show r = 3
  sorry

end radius_of_circle_l140_140375


namespace anja_equal_integers_l140_140953

theorem anja_equal_integers (S : Finset ℤ) (h_card : S.card = 2014)
  (h_mean : ∀ (x y z : ℤ), x ∈ S → y ∈ S → z ∈ S → (x + y + z) / 3 ∈ S) :
  ∃ k, ∀ x ∈ S, x = k :=
sorry

end anja_equal_integers_l140_140953


namespace symmetrical_point_of_P_is_correct_l140_140350

-- Define the point P
def P : ℝ × ℝ := (-1, 2)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the function to get the symmetric point with respect to the origin
def symmetrical_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Prove that the symmetrical point of P with respect to the origin is (1, -2)
theorem symmetrical_point_of_P_is_correct : symmetrical_point P = (1, -2) :=
  sorry

end symmetrical_point_of_P_is_correct_l140_140350


namespace sum_of_abs_coeffs_in_binomial_expansion_l140_140455

theorem sum_of_abs_coeffs_in_binomial_expansion :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ), 
  (3 * x - 1) ^ 7 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4 + a₅ * x ^ 5 + a₆ * x ^ 6 + a₇ * x ^ 7
  → |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 4 ^ 7 :=
by
  sorry

end sum_of_abs_coeffs_in_binomial_expansion_l140_140455


namespace courtyard_length_l140_140117

/-- Given the following conditions:
  1. The width of the courtyard is 16.5 meters.
  2. 66 paving stones are required.
  3. Each paving stone measures 2.5 meters by 2 meters.
  Prove that the length of the rectangular courtyard is 20 meters. -/
theorem courtyard_length :
  ∃ L : ℝ, L = 20 ∧ 
           (∃ W : ℝ, W = 16.5) ∧ 
           (∃ n : ℕ, n = 66) ∧ 
           (∃ A : ℝ, A = 2.5 * 2) ∧
           n * A = L * W :=
by
  sorry

end courtyard_length_l140_140117


namespace coefficient_of_x_l140_140320

theorem coefficient_of_x : 
  let expr := 2 * (x - 5) + 5 * (8 - 3 * x^2 + 6 * x) - 9 * (3 * x - 2)
  ∃ (a b c : ℝ), expr = a * x^2 + b * x + c ∧ b = 5 := by
    let expr := 2 * (x - 5) + 5 * (8 - 3 * x^2 + 6 * x) - 9 * (3 * x - 2)
    exact sorry

end coefficient_of_x_l140_140320


namespace average_monthly_balance_l140_140547

def january_balance : ℕ := 150
def february_balance : ℕ := 300
def march_balance : ℕ := 450
def april_balance : ℕ := 300
def number_of_months : ℕ := 4

theorem average_monthly_balance :
  (january_balance + february_balance + march_balance + april_balance) / number_of_months = 300 := by
  sorry

end average_monthly_balance_l140_140547


namespace line_intersects_curve_C_l140_140353

noncomputable def polar_to_rect_coor (ρ θ : ℝ) : ℝ × ℝ :=
(ρ * Real.cos θ, ρ * Real.sin θ)

theorem line_intersects_curve_C (α : ℝ) :
  let P := polar_to_rect_coor 2 π,
      rect_eq : ℝ × ℝ → Prop := λ ⟨x, y⟩, (x^2 + 3 * y^2 = 12),
      line_eq : ℝ → ℝ × ℝ := λ t, ⟨-2 + t * Real.cos α, t * Real.sin α⟩ in
  ∃ A B : ℝ × ℝ, rect_eq A ∧ rect_eq B ∧ (∃ t₁ t₂ : ℝ, line_eq t₁ = A ∧ line_eq t₂ = B ∧
  (1 / dist P A) + (1 / dist P B) ∈ Set.Icc (Real.sqrt 3 / 2) (Real.sqrt 6 / 2)) :=
by
  sorry

end line_intersects_curve_C_l140_140353


namespace necessary_but_not_sufficient_condition_proof_l140_140406

noncomputable def necessary_but_not_sufficient_condition (x : ℝ) : Prop :=
  2 * x ^ 2 - 5 * x - 3 ≥ 0

theorem necessary_but_not_sufficient_condition_proof (x : ℝ) :
  (x < 0 ∨ x > 2) → necessary_but_not_sufficient_condition x :=
  sorry

end necessary_but_not_sufficient_condition_proof_l140_140406


namespace series_sum_equals_three_fourths_l140_140961

noncomputable def infinite_series_sum : ℝ :=
  (∑' n : ℕ, (3 * (n + 1) + 2) / ((n + 1) * (n + 1 + 1) * (n + 1 + 3)))

theorem series_sum_equals_three_fourths :
  infinite_series_sum = 3 / 4 :=
sorry

end series_sum_equals_three_fourths_l140_140961


namespace quadratic_roots_sum_product_l140_140427

noncomputable def quadratic_sum (a b c : ℝ) : ℝ := -b / a
noncomputable def quadratic_product (a b c : ℝ) : ℝ := c / a

theorem quadratic_roots_sum_product :
  let a := 9
  let b := -45
  let c := 50
  quadratic_sum a b c = 5 ∧ quadratic_product a b c = 50 / 9 :=
by
  sorry

end quadratic_roots_sum_product_l140_140427


namespace find_value_of_x_y_l140_140870

theorem find_value_of_x_y (x y : ℝ) (h1 : |x| + x + y = 10) (h2 : |y| + x - y = 12) : x + y = 18 / 5 :=
by
  sorry

end find_value_of_x_y_l140_140870


namespace beadshop_wednesday_profit_l140_140145

theorem beadshop_wednesday_profit (total_profit : ℝ) (monday_fraction : ℝ) (tuesday_fraction : ℝ) :
  monday_fraction = 1/3 → tuesday_fraction = 1/4 → total_profit = 1200 →
  let monday_profit := monday_fraction * total_profit;
  let tuesday_profit := tuesday_fraction * total_profit;
  let wednesday_profit := total_profit - monday_profit - tuesday_profit;
  wednesday_profit = 500 :=
sorry

end beadshop_wednesday_profit_l140_140145


namespace solve_system_l140_140264

theorem solve_system :
  ∃ a b c d e : ℤ, 
    (a * b + a + 2 * b = 78) ∧
    (b * c + 3 * b + c = 101) ∧
    (c * d + 5 * c + 3 * d = 232) ∧
    (d * e + 4 * d + 5 * e = 360) ∧
    (e * a + 2 * e + 4 * a = 192) ∧
    ((a = 8 ∧ b = 7 ∧ c = 10 ∧ d = 14 ∧ e = 16) ∨ (a = -12 ∧ b = -9 ∧ c = -16 ∧ d = -24 ∧ e = -24)) :=
by
  sorry

end solve_system_l140_140264


namespace problem1_problem2_l140_140420

-- Problem 1: Proove that the given expression equals 1
theorem problem1 : (2021 * 2023) / (2022^2 - 1) = 1 :=
  by
  sorry

-- Problem 2: Proove that the given expression equals 45000
theorem problem2 : 2 * 101^2 + 2 * 101 * 98 + 2 * 49^2 = 45000 :=
  by
  sorry

end problem1_problem2_l140_140420


namespace WangLi_final_score_l140_140405

def weightedFinalScore (writtenScore : ℕ) (demoScore : ℕ) (interviewScore : ℕ)
    (writtenWeight : ℕ) (demoWeight : ℕ) (interviewWeight : ℕ) : ℕ :=
  (writtenScore * writtenWeight + demoScore * demoWeight + interviewScore * interviewWeight) /
  (writtenWeight + demoWeight + interviewWeight)

theorem WangLi_final_score :
  weightedFinalScore 96 90 95 5 3 2 = 94 :=
  by
  -- proof goes here
  sorry

end WangLi_final_score_l140_140405


namespace expression_bounds_l140_140245

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  2 + Real.sqrt 2 ≤ 
  (Real.sqrt (a^2 + (1 - b)^2 + 1) + 
   Real.sqrt (b^2 + (1 - c)^2 + 1) + 
   Real.sqrt (c^2 + (1 - d)^2 + 1) + 
   Real.sqrt (d^2 + (1 - a)^2 + 1)) ∧ 
  (Real.sqrt (a^2 + (1 - b)^2 + 1) + 
   Real.sqrt (b^2 + (1 - c)^2 + 1) + 
   Real.sqrt (c^2 + (1 - d)^2 + 1) + 
   Real.sqrt (d^2 + (1 - a)^2 + 1)) ≤ 4 * Real.sqrt 2 := 
sorry

end expression_bounds_l140_140245


namespace percentage_managers_decrease_l140_140515

theorem percentage_managers_decrease
  (employees : ℕ)
  (initial_percentage : ℝ)
  (managers_leave : ℝ)
  (new_percentage : ℝ)
  (h1 : employees = 200)
  (h2 : initial_percentage = 99)
  (h3 : managers_leave = 100)
  (h4 : new_percentage = 98) :
  ((initial_percentage / 100 * employees - managers_leave) / (employees - managers_leave) * 100 = new_percentage) :=
by
  -- To be proven
  sorry

end percentage_managers_decrease_l140_140515


namespace correct_calculated_value_l140_140337

theorem correct_calculated_value (x : ℤ) (h : x - 749 = 280) : x + 479 = 1508 :=
by 
  sorry

end correct_calculated_value_l140_140337


namespace work_completion_time_l140_140927

theorem work_completion_time (T : ℚ) :
  (∀ (a b : ℚ), (a + b) = 9 → (1 / a + 1 / b) * (9 / 18) = 1 / 9) ∧
  (∀ (a : ℚ), 1 / a * (9 / 18) = 1 / 18) ∧
  (∀ (c : ℚ), 1 / c * (9 / 24) = 1 / 24) →
  T = 72 / 11 := sorry

end work_completion_time_l140_140927


namespace jessica_quarters_l140_140597

theorem jessica_quarters (initial_quarters borrowed_quarters remaining_quarters : ℕ)
  (h1 : initial_quarters = 8)
  (h2 : borrowed_quarters = 3) :
  remaining_quarters = initial_quarters - borrowed_quarters → remaining_quarters = 5 :=
by
  intro h3
  rw [h1, h2] at h3
  exact h3

end jessica_quarters_l140_140597


namespace probability_straight_flush_l140_140517

theorem probability_straight_flush (num_total_hands : ℕ) (num_straight_flushes : ℕ) : 
  num_total_hands = (nat.choose 52 5) → 
  num_straight_flushes = 40 → 
  (num_straight_flushes : ℚ) / (num_total_hands : ℚ) = 1 / 64974 :=
by
  intros h_total_hands h_straight_flushes
  rw [h_total_hands, h_straight_flushes]
  norm_num
  sorry

end probability_straight_flush_l140_140517


namespace eclipse_time_coincide_eclipse_start_time_eclipse_end_time_l140_140004

noncomputable def relative_speed_moon_sun := (17/16 : ℝ) - (1/12 : ℝ)
noncomputable def initial_distance := (47/10 : ℝ)
noncomputable def time_coincide := initial_distance / relative_speed_moon_sun + (9 + 13/60 : ℝ)

theorem eclipse_time_coincide : 
  (time_coincide - 12 : ℝ) = (2 + 1/60 : ℝ) :=
sorry

noncomputable def start_distance := (37/10 : ℝ)
noncomputable def time_start := start_distance / relative_speed_moon_sun + (9 + 13/60 : ℝ)

theorem eclipse_start_time : 
  (time_start - 12 : ℝ) = (1 + 59/60 : ℝ) :=
sorry

noncomputable def end_distance := (57/10 : ℝ)
noncomputable def time_end := end_distance / relative_speed_moon_sun + (9 + 13/60 : ℝ)

theorem eclipse_end_time : 
  (time_end - 12 : ℝ) = (3 + 2/60 : ℝ) :=
sorry

end eclipse_time_coincide_eclipse_start_time_eclipse_end_time_l140_140004


namespace fly_total_distance_l140_140938

noncomputable def total_distance_traveled (r : ℝ) (d3 : ℝ) : ℝ :=
  let d1 := 2 * r
  let d2 := Real.sqrt (d1^2 - d3^2)
  d1 + d2 + d3

theorem fly_total_distance (r : ℝ) (h_r : r = 60) (d3 : ℝ) (h_d3 : d3 = 90) :
  total_distance_traveled r d3 = 289.37 :=
by
  rw [h_r, h_d3]
  simp [total_distance_traveled]
  sorry

end fly_total_distance_l140_140938


namespace combine_sum_l140_140199

def A (n m : Nat) : Nat := n.factorial / (n - m).factorial
def C (n m : Nat) : Nat := n.factorial / (m.factorial * (n - m).factorial)

theorem combine_sum (n m : Nat) (hA : A n m = 272) (hC : C n m = 136) : m + n = 19 := by
  sorry

end combine_sum_l140_140199


namespace exponential_function_value_l140_140102

noncomputable def f (x : ℝ) : ℝ := 2^x

theorem exponential_function_value :
  f (f 2) = 16 := by
  simp only [f]
  sorry

end exponential_function_value_l140_140102


namespace gambler_largest_amount_proof_l140_140152

noncomputable def largest_amount_received_back (initial_amount : ℝ) (value_25 : ℝ) (value_75 : ℝ) (value_250 : ℝ) 
                                               (total_lost_chips : ℝ) (coef_25_75_lost : ℝ) (coef_75_250_lost : ℝ) : ℝ :=
    initial_amount - (
    coef_25_75_lost * (total_lost_chips / (coef_25_75_lost + 1 + 1)) * value_25 +
    (total_lost_chips / (coef_25_75_lost + 1 + 1)) * value_75 +
    coef_75_250_lost * (total_lost_chips / (coef_25_75_lost + 1 + 1)) * value_250)

theorem gambler_largest_amount_proof :
    let initial_amount := 15000
    let value_25 := 25
    let value_75 := 75
    let value_250 := 250
    let total_lost_chips := 40
    let coef_25_75_lost := 2 -- number of lost $25 chips is twice the number of lost $75 chips
    let coef_75_250_lost := 2 -- number of lost $250 chips is twice the number of lost $75 chips
    largest_amount_received_back initial_amount value_25 value_75 value_250 total_lost_chips coef_25_75_lost coef_75_250_lost = 10000 :=
by {
    sorry
}

end gambler_largest_amount_proof_l140_140152


namespace perimeter_division_l140_140107

-- Define the given conditions
def is_pentagon (n : ℕ) : Prop := n = 5
def side_length (s : ℕ) : Prop := s = 25
def perimeter (P : ℕ) (n s : ℕ) : Prop := P = n * s

-- Define the Lean statement to prove
theorem perimeter_division (n s P x : ℕ) 
  (h1 : is_pentagon n) 
  (h2 : side_length s) 
  (h3 : perimeter P n s) 
  (h4 : P = 125) 
  (h5 : s = 25) : 
  P / x = s → x = 5 := 
by
  sorry

end perimeter_division_l140_140107


namespace integer_conditions_satisfy_eq_l140_140014

theorem integer_conditions_satisfy_eq (
  a b c : ℤ 
) : (a > b ∧ b = c → (a * (a - b) + b * (b - c) + c * (c - a) = 2)) ∧
    (¬(a = b - 1 ∧ b = c - 2) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) ∧
    (¬(a = c + 1 ∧ b = a + 2) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) ∧
    (¬(a = c ∧ b - 2 = c) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) ∧
    (¬(a + b + c = 2) → (a * (a - b) + b * (b - c) + c * (c - a) ≠ 2)) :=
by
sorry

end integer_conditions_satisfy_eq_l140_140014


namespace sum_three_times_m_and_half_n_square_diff_minus_square_sum_l140_140178

-- Problem (1) Statement
theorem sum_three_times_m_and_half_n (m n : ℝ) : 3 * m + 1 / 2 * n = 3 * m + 1 / 2 * n :=
by
  sorry

-- Problem (2) Statement
theorem square_diff_minus_square_sum (a b : ℝ) : (a - b) ^ 2 - (a + b) ^ 2 = (a - b) ^ 2 - (a + b) ^ 2 :=
by
  sorry

end sum_three_times_m_and_half_n_square_diff_minus_square_sum_l140_140178


namespace min_value_sin_cos_l140_140323

open Real

theorem min_value_sin_cos (x : ℝ) : 
  ∃ x : ℝ, sin x ^ 4 + 2 * cos x ^ 4 = 2 / 3 ∧ ∀ x : ℝ, sin x ^ 4 + 2 * cos x ^ 4 ≥ 2 / 3 :=
sorry

end min_value_sin_cos_l140_140323


namespace type_B_ratio_l140_140490

theorem type_B_ratio
    (num_A : ℕ)
    (total_bricks : ℕ)
    (other_bricks : ℕ)
    (h1 : num_A = 40)
    (h2 : total_bricks = 150)
    (h3 : other_bricks = 90) :
    (total_bricks - num_A - other_bricks) / num_A = 1 / 2 :=
by
  sorry

end type_B_ratio_l140_140490


namespace sum_first_6_is_correct_l140_140331

namespace ProofProblem

def sequence (a : ℕ → ℚ) : Prop :=
  (a 1 = 1) ∧ ∀ n : ℕ, n ≥ 2 → a (n - 1) = 2 * a n

def sum_first_6 (a : ℕ → ℚ) : ℚ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

theorem sum_first_6_is_correct (a : ℕ → ℚ) (h : sequence a) :
  sum_first_6 a = 63 / 32 :=
sorry

end ProofProblem

end sum_first_6_is_correct_l140_140331


namespace parabola_c_value_l140_140408

theorem parabola_c_value (b c : ℝ) 
  (h1 : 20 = 2*(-2)^2 + b*(-2) + c) 
  (h2 : 28 = 2*2^2 + b*2 + c) : 
  c = 16 :=
by
  sorry

end parabola_c_value_l140_140408


namespace distinct_sums_l140_140447

theorem distinct_sums (n : ℕ) (a : Fin n → ℕ) (h_distinct : Function.Injective a) :
  ∃ S : Finset ℕ, S.card ≥ n * (n + 1) / 2 :=
by
  sorry

end distinct_sums_l140_140447


namespace max_value_proof_l140_140611

noncomputable def max_value (x y : ℝ) : ℝ := x^2 + 2 * x * y + 3 * y^2

theorem max_value_proof (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - 2 * x * y + 3 * y^2 = 12) : 
  max_value x y = 24 + 12 * Real.sqrt 3 := 
sorry

end max_value_proof_l140_140611


namespace T_10_mod_5_eq_3_l140_140563

def a_n (n : ℕ) : ℕ := -- Number of sequences of length n ending in A
sorry

def b_n (n : ℕ) : ℕ := -- Number of sequences of length n ending in B
sorry

def c_n (n : ℕ) : ℕ := -- Number of sequences of length n ending in C
sorry

def T (n : ℕ) : ℕ := -- Number of valid sequences of length n
  a_n n + b_n n

theorem T_10_mod_5_eq_3 :
  T 10 % 5 = 3 :=
sorry

end T_10_mod_5_eq_3_l140_140563


namespace complement_intersection_l140_140462

open Set

namespace UniversalSetProof

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5}

theorem complement_intersection :
  (U \ A) ∩ B = {4, 5} :=
by
  sorry

end UniversalSetProof

end complement_intersection_l140_140462


namespace equal_commissions_implies_list_price_l140_140821

theorem equal_commissions_implies_list_price (x : ℝ) :
  (0.15 * (x - 15) = 0.25 * (x - 25)) → x = 40 :=
by
  intro h
  sorry

end equal_commissions_implies_list_price_l140_140821


namespace planes_parallel_l140_140713

variables (m n : Line) (α β : Plane)

-- Non-overlapping lines and planes conditions
axiom non_overlapping_lines : m ≠ n
axiom non_overlapping_planes : α ≠ β

-- Parallel and perpendicular definitions
axiom parallel_lines (l k : Line) : Prop
axiom parallel_planes (π ρ : Plane) : Prop
axiom perpendicular (l : Line) (π : Plane) : Prop

-- Given conditions
axiom m_perpendicular_to_alpha : perpendicular m α
axiom m_perpendicular_to_beta : perpendicular m β

-- Proof statement
theorem planes_parallel (m_perpendicular_to_alpha : perpendicular m α)
  (m_perpendicular_to_beta : perpendicular m β) :
  parallel_planes α β := sorry

end planes_parallel_l140_140713


namespace partI_inequality_partII_inequality_l140_140037

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 3|

-- Part (Ⅰ): Prove f(x) ≤ x + 1 for 1 ≤ x ≤ 5
theorem partI_inequality (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 5) : f x ≤ x + 1 := by
  sorry

-- Part (Ⅱ): Prove (a^2)/(a+1) + (b^2)/(b+1) ≥ 1 when a + b = 2 and a > 0, b > 0
theorem partII_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) : 
    (a^2) / (a + 1) + (b^2) / (b + 1) ≥ 1 := by
  sorry

end partI_inequality_partII_inequality_l140_140037


namespace find_number_l140_140532

theorem find_number (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 := 
sorry

end find_number_l140_140532


namespace Kelly_weight_is_M_l140_140819

variable (M : ℝ) -- Megan's weight
variable (K : ℝ) -- Kelly's weight
variable (Mike : ℝ) -- Mike's weight

-- Conditions based on the problem statement
def Kelly_less_than_Megan (M K : ℝ) : Prop := K = 0.85 * M
def Mike_greater_than_Megan (M Mike : ℝ) : Prop := Mike = M + 5
def Total_weight_exceeds_bridge (total_weight : ℝ) : Prop := total_weight = 100 + 19
def Total_weight_of_children (M K Mike total_weight : ℝ) : Prop := total_weight = M + K + Mike

theorem Kelly_weight_is_M : (M = 40) → (Total_weight_exceeds_bridge 119) → (Kelly_less_than_Megan M K) → (Mike_greater_than_Megan M Mike) → K = 34 :=
by
  -- Insert proof here
  sorry

end Kelly_weight_is_M_l140_140819


namespace perpendicular_lines_b_eq_neg_six_l140_140016

theorem perpendicular_lines_b_eq_neg_six
    (b : ℝ) :
    (∀ x y : ℝ, 3 * y + 2 * x - 4 = 0 → y = (-2/3) * x + 4/3) →
    (∀ x y : ℝ, 4 * y + b * x - 6 = 0 → y = (-b/4) * x + 3/2) →
    - (2/3) * (-b/4) = -1 →
    b = -6 := 
sorry

end perpendicular_lines_b_eq_neg_six_l140_140016


namespace Lisa_initial_pencils_l140_140995

-- Variables
variable (G_L_initial : ℕ) (L_L_initial : ℕ) (G_L_total : ℕ)

-- Conditions
def G_L_initial_def := G_L_initial = 2
def G_L_total_def := G_L_total = 101
def Lisa_gives_pencils : Prop := G_L_total = G_L_initial + L_L_initial

-- Proof statement
theorem Lisa_initial_pencils (G_L_initial : ℕ) (G_L_total : ℕ)
  (h1 : G_L_initial = 2) (h2 : G_L_total = 101) (h3 : G_L_total = G_L_initial + L_L_initial) :
  L_L_initial = 99 := 
by 
  sorry

end Lisa_initial_pencils_l140_140995


namespace heather_initial_oranges_l140_140212

theorem heather_initial_oranges (given_oranges: ℝ) (total_oranges: ℝ) (initial_oranges: ℝ) 
    (h1: given_oranges = 35.0) 
    (h2: total_oranges = 95) : 
    initial_oranges = 60 :=
by
  sorry

end heather_initial_oranges_l140_140212


namespace length_of_other_train_l140_140390

def speed1 := 90 -- speed in km/hr
def speed2 := 90 -- speed in km/hr
def length_train1 := 1.10 -- length in km
def crossing_time := 40 -- time in seconds

theorem length_of_other_train : 
  ∀ s1 s2 l1 t l2 : ℝ,
  s1 = 90 → s2 = 90 → l1 = 1.10 → t = 40 → 
  ((s1 + s2) / 3600 * t - l1 = l2) → 
  l2 = 0.90 :=
by
  intros s1 s2 l1 t l2 hs1 hs2 hl1 ht hdist
  sorry

end length_of_other_train_l140_140390


namespace simplify_expression_l140_140974

theorem simplify_expression : 
  2^345 - 3^4 * (3^2)^2 = 2^345 - 6561 := by
sorry

end simplify_expression_l140_140974


namespace tank_capacity_l140_140802

theorem tank_capacity (C : ℝ) 
  (h1 : 10 > 0) 
  (h2 : 16 > (10 : ℝ))
  (h3 : ((C/10) - 480 = (C/16))) : C = 1280 := 
by 
  sorry

end tank_capacity_l140_140802


namespace percentage_calculation_l140_140692

-- Define total and part amounts
def total_amount : ℕ := 800
def part_amount : ℕ := 200

-- Define the percentage calculation
def percentage (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- Theorem to show the percentage is 25%
theorem percentage_calculation :
  percentage part_amount total_amount = 25 :=
sorry

end percentage_calculation_l140_140692


namespace find_a_given_solution_l140_140571

theorem find_a_given_solution (a : ℝ) (x : ℝ) (h : x = 1) (eqn : a * (x + 1) = 2 * (2 * x - a)) : a = 1 := 
by
  sorry

end find_a_given_solution_l140_140571


namespace dragon_jewels_end_l140_140148

-- Given conditions
variables (D : ℕ) (jewels_taken_by_king jewels_taken_from_king new_jewels final_jewels : ℕ)

-- Conditions corresponding to the problem
axiom h1 : jewels_taken_by_king = 3
axiom h2 : jewels_taken_from_king = 2 * jewels_taken_by_king
axiom h3 : new_jewels = jewels_taken_from_king
axiom h4 : new_jewels = D / 3

-- Equation derived from the problem setting
def number_of_jewels_initial := D
def number_of_jewels_after_king_stole := number_of_jewels_initial - jewels_taken_by_king
def number_of_jewels_final := number_of_jewels_after_king_stole + jewels_taken_from_king

-- Final proof obligation
theorem dragon_jewels_end : ∃ (D : ℕ), number_of_jewels_final D 3 6 = 21 :=
by
  sorry

end dragon_jewels_end_l140_140148


namespace check_random_event_l140_140129

def random_event (A B C D : Prop) : Prop := ∃ E, D = E

def event_A : Prop :=
  ∀ (probability : ℝ), probability = 0

def event_B : Prop :=
  ∀ (probability : ℝ), probability = 0

def event_C : Prop :=
  ∀ (probability : ℝ), probability = 1

def event_D : Prop :=
  ∀ (probability : ℝ), 0 < probability ∧ probability < 1

theorem check_random_event :
  random_event event_A event_B event_C event_D :=
sorry

end check_random_event_l140_140129


namespace max_radius_approx_l140_140154

open Real

def angle_constraint (θ : ℝ) : Prop :=
  π / 4 ≤ θ ∧ θ ≤ 3 * π / 4

def wire_constraint (r θ : ℝ) : Prop :=
  16 = r * (2 + θ)

noncomputable def max_radius (θ : ℝ) : ℝ :=
  16 / (2 + θ)

theorem max_radius_approx :
  ∃ r θ, angle_constraint θ ∧ wire_constraint r θ ∧ abs (r - 3.673) < 0.001 :=
by
  sorry

end max_radius_approx_l140_140154


namespace score_87_not_possible_l140_140751

def max_score := 15 * 6
def score (correct unanswered incorrect : ℕ) := 6 * correct + unanswered

theorem score_87_not_possible :
  ¬∃ (correct unanswered incorrect : ℕ), 
    correct + unanswered + incorrect = 15 ∧
    6 * correct + unanswered = 87 := 
sorry

end score_87_not_possible_l140_140751


namespace parallel_planes_if_perpendicular_to_same_line_l140_140714

variables {m n : Type} [line m] [line n]
variables {α β : Type} [plane α] [plane β]

theorem parallel_planes_if_perpendicular_to_same_line (h1 : m ⟂ α) (h2 : m ⟂ β) : α ∥ β :=
sorry

end parallel_planes_if_perpendicular_to_same_line_l140_140714


namespace sign_of_x_and_y_l140_140469

theorem sign_of_x_and_y (x y : ℝ) (h1 : x * y > 1) (h2 : x + y ≥ 0) : x > 0 ∧ y > 0 :=
sorry

end sign_of_x_and_y_l140_140469


namespace geom_seq_not_necessary_sufficient_l140_140078

theorem geom_seq_not_necessary_sufficient (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q) (h2 : q > 1) :
  ¬(∀ n, a n > a (n + 1) → false) ∨ ¬(∀ n, a (n + 1) > a n) :=
sorry

end geom_seq_not_necessary_sufficient_l140_140078


namespace volume_ratio_of_spheres_l140_140383

theorem volume_ratio_of_spheres (r1 r2 r3 : ℝ) 
  (h : r1 / r2 = 1 / 2 ∧ r2 / r3 = 2 / 3) : 
  (4/3 * π * r3^3) = 3 * (4/3 * π * r1^3 + 4/3 * π * r2^3) :=
by
  sorry

end volume_ratio_of_spheres_l140_140383


namespace speed_limit_of_friend_l140_140837

theorem speed_limit_of_friend (total_distance : ℕ) (christina_speed : ℕ) (christina_time_min : ℕ) (friend_time_hr : ℕ) 
(h1 : total_distance = 210)
(h2 : christina_speed = 30)
(h3 : christina_time_min = 180)
(h4 : friend_time_hr = 3)
(h5 : total_distance = (christina_speed * (christina_time_min / 60)) + (christina_speed * friend_time_hr)) :
  (total_distance - christina_speed * (christina_time_min / 60)) / friend_time_hr = 40 := 
by
  sorry

end speed_limit_of_friend_l140_140837


namespace Owen_spending_on_burgers_in_June_l140_140026

theorem Owen_spending_on_burgers_in_June (daily_burgers : ℕ) (cost_per_burger : ℕ) (days_in_June : ℕ) :
  daily_burgers = 2 → 
  cost_per_burger = 12 → 
  days_in_June = 30 → 
  daily_burgers * cost_per_burger * days_in_June = 720 :=
by
  intros
  sorry

end Owen_spending_on_burgers_in_June_l140_140026


namespace negation_of_proposition_l140_140382

theorem negation_of_proposition
  (h : ∀ x : ℝ, x^2 - 2 * x + 2 > 0) :
  ∃ x : ℝ, x^2 - 2 * x + 2 ≤ 0 :=
sorry

end negation_of_proposition_l140_140382


namespace jessica_has_three_dozens_of_red_marbles_l140_140239

-- Define the number of red marbles Sandy has
def sandy_red_marbles : ℕ := 144

-- Define the relationship between Sandy's and Jessica's red marbles
def relationship (jessica_red_marbles : ℕ) : Prop :=
  sandy_red_marbles = 4 * jessica_red_marbles

-- Define the question to find out how many dozens of red marbles Jessica has
def jessica_dozens (jessica_red_marbles : ℕ) := jessica_red_marbles / 12

-- Theorem stating that given the conditions, Jessica has 3 dozens of red marbles
theorem jessica_has_three_dozens_of_red_marbles (jessica_red_marbles : ℕ)
  (h : relationship jessica_red_marbles) : jessica_dozens jessica_red_marbles = 3 :=
by
  -- The proof is omitted
  sorry

end jessica_has_three_dozens_of_red_marbles_l140_140239


namespace find_m_l140_140414

theorem find_m (m : ℝ) (a b : ℝ) (r s : ℝ) (S1 S2 : ℝ)
  (h1 : a = 10)
  (h2 : b = 10)
  (h3 : 10 * r = 5)
  (h4 : S1 = 20)
  (h5 : 10 * s = 5 + m)
  (h6 : S2 = 100 / (5 - m))
  (h7 : S2 = 3 * S1) :
  m = 10 / 3 := by
  sorry

end find_m_l140_140414


namespace carson_gets_clawed_39_times_l140_140959

-- Conditions
def number_of_wombats : ℕ := 9
def claws_per_wombat : ℕ := 4
def number_of_rheas : ℕ := 3
def claws_per_rhea : ℕ := 1

-- Theorem statement
theorem carson_gets_clawed_39_times :
  (number_of_wombats * claws_per_wombat + number_of_rheas * claws_per_rhea) = 39 :=
by
  sorry

end carson_gets_clawed_39_times_l140_140959


namespace no_integer_a_for_integer_roots_l140_140317

theorem no_integer_a_for_integer_roots :
  ∀ a : ℤ, ¬ (∃ x : ℤ, x^2 - 2023 * x + 2022 * a + 1 = 0) := 
by
  intro a
  rintro ⟨x, hx⟩
  sorry

end no_integer_a_for_integer_roots_l140_140317


namespace train_speed_l140_140308

theorem train_speed (length : ℕ) (time : ℕ) (v : ℕ)
  (h1 : length = 750)
  (h2 : time = 1)
  (h3 : v = (length + length) / time)
  (h4 : v = 1500) :
  (v * 60 / 1000 = 90) :=
by
  sorry

end train_speed_l140_140308


namespace pyarelal_loss_l140_140003

theorem pyarelal_loss (P : ℝ) (total_loss : ℝ) (h1 : total_loss = 670) (h2 : 1 / 9 * P + P = 10 / 9 * P):
  (9 / (1 + 9)) * total_loss = 603 :=
by
  sorry

end pyarelal_loss_l140_140003


namespace evaluate_expression_l140_140852

theorem evaluate_expression (x : ℤ) (h : x = 5) : 
  3 * (3 * (3 * (3 * (3 * x + 2) + 2) + 2) + 2) + 2 = 1457 := 
by
  rw [h]
  sorry

end evaluate_expression_l140_140852


namespace quadratic_transformation_l140_140449

theorem quadratic_transformation (y m n : ℝ) 
  (h1 : 2 * y^2 - 2 = 4 * y) 
  (h2 : (y - m)^2 = n) : 
  (m - n)^2023 = -1 := 
  sorry

end quadratic_transformation_l140_140449


namespace evaporation_amount_l140_140884

variable (E : ℝ)

def initial_koolaid_powder : ℝ := 2
def initial_water : ℝ := 16
def final_percentage : ℝ := 0.04

theorem evaporation_amount :
  (initial_koolaid_powder = 2) →
  (initial_water = 16) →
  (0.04 * (initial_koolaid_powder + 4 * (initial_water - E)) = initial_koolaid_powder) →
  E = 4 :=
by
  intros h1 h2 h3
  sorry

end evaporation_amount_l140_140884


namespace brian_spent_on_kiwis_l140_140143

theorem brian_spent_on_kiwis :
  ∀ (cost_per_dozen_apples : ℝ)
    (cost_for_24_apples : ℝ)
    (initial_money : ℝ)
    (subway_fare_one_way : ℝ)
    (total_remaining : ℝ)
    (kiwis_spent : ℝ)
    (bananas_spent : ℝ),
  cost_per_dozen_apples = 14 →
  cost_for_24_apples = 2 * cost_per_dozen_apples →
  initial_money = 50 →
  subway_fare_one_way = 3.5 →
  total_remaining = initial_money - 2 * subway_fare_one_way - cost_for_24_apples →
  total_remaining = 15 →
  bananas_spent = kiwis_spent / 2 →
  kiwis_spent + bananas_spent = total_remaining →
  kiwis_spent = 10 :=
by
  -- Sorry means we are skipping the proof
  sorry

end brian_spent_on_kiwis_l140_140143


namespace exists_additive_function_close_to_f_l140_140503

variable (f : ℝ → ℝ)

theorem exists_additive_function_close_to_f (h : ∀ x y : ℝ, |f (x + y) - f x - f y| ≤ 1) :
  ∃ g : ℝ → ℝ, (∀ x : ℝ, |f x - g x| ≤ 1) ∧ (∀ x y : ℝ, g (x + y) = g x + g y) := by
  sorry

end exists_additive_function_close_to_f_l140_140503


namespace average_of_xyz_l140_140466

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 15) : 
  (x + y + z) / 3 = 4 :=
sorry

end average_of_xyz_l140_140466


namespace probability_miss_at_least_once_l140_140108
-- Importing the entirety of Mathlib

-- Defining the conditions and question
variable (P : ℝ) (hP : 0 ≤ P ∧ P ≤ 1)

-- The main statement for the proof problem
theorem probability_miss_at_least_once (P : ℝ) (hP : 0 ≤ P ∧ P ≤ 1) : P ≤ 1 → 0 ≤ P ∧ 1 - P^3 ≥ 0 := 
by
  sorry

end probability_miss_at_least_once_l140_140108


namespace inequality_div_c_squared_l140_140215

theorem inequality_div_c_squared (a b c : ℝ) (h : a > b) : (a / (c^2 + 1) > b / (c^2 + 1)) :=
by
  sorry

end inequality_div_c_squared_l140_140215


namespace sum_of_real_values_l140_140384

theorem sum_of_real_values (x : ℝ) (h : |3 * x + 1| = 3 * |x - 3|) : x = 4 / 3 := sorry

end sum_of_real_values_l140_140384


namespace complement_U_A_l140_140722

def U : Finset ℤ := {-2, -1, 0, 1, 2}
def A : Finset ℤ := {-2, -1, 1, 2}

theorem complement_U_A : (U \ A) = {0} := by
  sorry

end complement_U_A_l140_140722


namespace inequality_system_correctness_l140_140191

theorem inequality_system_correctness :
  (∀ (x a b : ℝ), 
    (x - a ≥ 1) ∧ (x - b < 2) →
    ((∀ x, -1 ≤ x ∧ x < 3 → (a = -2 ∧ b = 1)) ∧
     (a = b → (a + 1 ≤ x ∧ x < a + 2)) ∧
     (¬(∃ x, a + 1 ≤ x ∧ x < b + 2) → a > b + 1) ∧
     ((∃ n : ℤ, n < 0 ∧ n ≥ -6 - a ∧ n ≥ -5) → -7 < a ∧ a ≤ -6))) :=
sorry

end inequality_system_correctness_l140_140191


namespace inequality_proof_l140_140446

theorem inequality_proof
  (x y z : ℝ)
  (hx : x > y)
  (hy : y > 1)
  (hz : 1 > z)
  (hzpos : z > 0)
  (a : ℝ := (1 + x * z) / z)
  (b : ℝ := (1 + x * y) / x)
  (c : ℝ := (1 + y * z) / y) :
  a > b ∧ a > c :=
by
  sorry

end inequality_proof_l140_140446


namespace distance_from_Beijing_to_Lanzhou_l140_140122

-- Conditions
def distance_Beijing_Lanzhou_Lhasa : ℕ := 3985
def distance_Lanzhou_Lhasa : ℕ := 2054

-- Define the distance from Beijing to Lanzhou
def distance_Beijing_Lanzhou : ℕ := distance_Beijing_Lanzhou_Lhasa - distance_Lanzhou_Lhasa

-- Proof statement that given conditions imply the correct answer
theorem distance_from_Beijing_to_Lanzhou :
  distance_Beijing_Lanzhou = 1931 :=
by
  -- conditions and definitions are already given
  sorry

end distance_from_Beijing_to_Lanzhou_l140_140122


namespace smallest_y_l140_140126

theorem smallest_y (y : ℤ) (h : y < 3 * y - 15) : y = 8 :=
  sorry

end smallest_y_l140_140126


namespace length_of_AB_l140_140544

-- Defining the parabola and the condition on x1 and x2
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def condition (x1 x2 : ℝ) : Prop := x1 + x2 = 9

-- The main statement to prove |AB| = 11
theorem length_of_AB (x1 x2 y1 y2 : ℝ) (h1 : parabola x1 y1) (h2 : parabola x2 y2) (hx : condition x1 x2) :
  abs (x1 - x2) + abs (y1 - y2) = 11 :=
sorry

end length_of_AB_l140_140544


namespace find_A_range_sinB_sinC_l140_140730

-- Given conditions in a triangle
variable (a b c : ℝ)
variable (A B C : ℝ)
variable (h_cos_eq : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A)

-- Angle A verification
theorem find_A (h_sum_angles : A + B + C = Real.pi) : A = Real.pi / 3 :=
  sorry

-- Range of sin B + sin C
theorem range_sinB_sinC (h_sum_angles : A + B + C = Real.pi) :
  (0 < B ∧ B < 2 * Real.pi / 3) →
  Real.sin B + Real.sin C ∈ Set.Ioo (Real.sqrt 3 / 2) (Real.sqrt 3) :=
  sorry

end find_A_range_sinB_sinC_l140_140730


namespace cost_of_socks_l140_140496

theorem cost_of_socks (x : ℝ) : 
  let initial_amount := 20
  let hat_cost := 7 
  let final_amount := 5
  let socks_pairs := 4
  let remaining_amount := initial_amount - hat_cost
  remaining_amount - socks_pairs * x = final_amount 
  -> x = 2 := 
by 
  sorry

end cost_of_socks_l140_140496


namespace find_number_l140_140533

theorem find_number (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 := 
sorry

end find_number_l140_140533


namespace correct_options_l140_140204

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function : Prop := ∀ x : ℝ, f x = f (-x)
def function_definition : Prop := ∀ x : ℝ, (0 < x) → f x = x^2 + x

-- Statements to be proved
def option_A : Prop := f (-1) = 2
def option_B_incorrect : Prop := ¬ (∀ x : ℝ, (f x ≥ f 0) ↔ x ≥ 0) -- Reformulated as not a correct statement
def option_C : Prop := ∀ x : ℝ, x < 0 → f x = x^2 - x
def option_D : Prop := ∀ x : ℝ, (0 < x ∧ x < 2) ↔ f (x - 1) < 2

-- Prove that the correct statements are A, C, and D
theorem correct_options (h_even : is_even_function f) (h_def : function_definition f) :
  option_A f ∧ option_C f ∧ option_D f := by
  sorry

end correct_options_l140_140204


namespace filter_probability_l140_140838

noncomputable def letter_probability : ℚ :=
  let p_river := (Nat.choose 5 3 : ℚ)⁻¹
  let p_stone := 6 / (Nat.choose 5 3 : ℚ)
  let p_flight := 6 / (Nat.choose 6 4 : ℚ)
  p_river * p_stone * p_flight

theorem filter_probability :
  letter_probability = 3 / 125 := by
  sorry

end filter_probability_l140_140838


namespace dragon_jewels_l140_140151

theorem dragon_jewels (x : ℕ) (h1 : (x / 3 = 6)) : x + 6 = 24 :=
sorry

end dragon_jewels_l140_140151


namespace max_type_a_workers_l140_140402

theorem max_type_a_workers (x y : ℕ) (h1 : x + y = 150) (h2 : y ≥ 3 * x) : x ≤ 37 :=
sorry

end max_type_a_workers_l140_140402


namespace max_sum_of_ten_consecutive_in_hundred_l140_140015

theorem max_sum_of_ten_consecutive_in_hundred :
  ∀ (s : Fin 100 → ℕ), (∀ i : Fin 100, 1 ≤ s i ∧ s i ≤ 100) → 
  (∃ i : Fin 91, (s i + s (i + 1) + s (i + 2) + s (i + 3) +
  s (i + 4) + s (i + 5) + s (i + 6) + s (i + 7) + s (i + 8) + s (i + 9)) ≥ 505) :=
by
  intro s hs
  sorry

end max_sum_of_ten_consecutive_in_hundred_l140_140015


namespace min_value_expression_l140_140486

theorem min_value_expression (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : a * b * c = 1/2) :
  a^2 + 4 * a * b + 12 * b^2 + 8 * b * c + 3 * c^2 ≥ 18 :=
sorry

end min_value_expression_l140_140486


namespace value_of_2x_plus_3y_l140_140585

theorem value_of_2x_plus_3y {x y : ℝ} (h1 : 2 * x - 1 = 5) (h2 : 3 * y + 2 = 17) : 2 * x + 3 * y = 21 :=
by
  sorry

end value_of_2x_plus_3y_l140_140585


namespace value_of_expression_l140_140289

theorem value_of_expression : 
  ∀ (a x y : ℤ), 
  (x = a + 5) → 
  (a = 20) → 
  (y = 25) → 
  (x - y) * (x + y) = 0 :=
by
  intros a x y h1 h2 h3
  -- proof goes here
  sorry

end value_of_expression_l140_140289


namespace largest_rectangle_area_l140_140776

theorem largest_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 60) : x * y ≤ 225 :=
by
  sorry

end largest_rectangle_area_l140_140776


namespace powers_of_two_div7_l140_140297

theorem powers_of_two_div7 (n : ℕ) : (2^n - 1) % 7 = 0 ↔ ∃ k : ℕ, n = 3 * k := sorry

end powers_of_two_div7_l140_140297


namespace determine_angle_XZY_l140_140596

variable (X Y Z P Q R : Point)
variable (θ : ℝ)

-- Define the conditions
def XY_eq_3XZ (X Y Z : Point) : Prop := dist X Y = 3 * dist X Z
def angle_XPQ_eq_angle_ZQP (X P Q Z : Point) : Prop := ∠ X P Q = ∠ Z Q P
def triangle_ZQR_equilateral (Z Q R : Point) : Prop := 
  dist Z Q = dist Q R ∧ dist Q R = dist R Z ∧ dist R Z = dist Z Q 

-- Main theorem statement
theorem determine_angle_XZY 
  (h1 : XY_eq_3XZ X Y Z)
  (h2 : angle_XPQ_eq_angle_ZQP X P Q Z)
  (h3 : triangle_ZQR_equilateral Z Q R) : 
  ∠ X Z Y = 30 :=
sorry

end determine_angle_XZY_l140_140596


namespace problem_statement_l140_140223

theorem problem_statement (x : ℤ) (h₁ : (x - 5) / 7 = 7) : (x - 24) / 10 = 3 := 
sorry

end problem_statement_l140_140223


namespace f_pos_for_all_x_g_le_ax_plus_1_for_a_eq_1_l140_140038

noncomputable def f (x : ℝ) : ℝ := Real.exp x - (x + 1)^2 / 2
noncomputable def g (x : ℝ) : ℝ := 2 * Real.log (x + 1) + Real.exp (-x)

theorem f_pos_for_all_x (x : ℝ) (hx : x > -1) : f x > 0 := by
  sorry

theorem g_le_ax_plus_1_for_a_eq_1 (a : ℝ) (ha : a > 0) : (∀ x : ℝ, -1 < x → g x ≤ a * x + 1) ↔ a = 1 := by
  sorry

end f_pos_for_all_x_g_le_ax_plus_1_for_a_eq_1_l140_140038


namespace quadratic_inequality_solution_l140_140096

theorem quadratic_inequality_solution (a b: ℝ) (h1: ∀ x: ℝ, 1 < x ∧ x < 2 → ax^2 + bx - 4 > 0) (h2: ∀ x: ℝ, x ≤ 1 ∨ x ≥ 2 → ax^2 + bx - 4 ≤ 0) : a + b = 4 :=
sorry

end quadratic_inequality_solution_l140_140096


namespace Problem_statement_l140_140086

open EuclideanGeometry

noncomputable def Problem : Prop :=
  ∀ (A B C D E F : Point) (l1 l2 : Line),
  Parallelogram ABCD ∧
  ∠ABC = 100 ∧
  dist A B = 20 ∧
  dist B C = 14 ∧
  D ∈ l1 ∧
  E ∈ l1 ∧
  F ∈ l2 ∧
  D ≠ E ∧
  dist D E = 6 ∧
  Meq (LineSegment B E) l2 ∧ 
  Meq (LineSegment A D) l2 → 
  dist F D = 4.2

theorem Problem_statement : Problem := by
  sorry

end Problem_statement_l140_140086


namespace min_value_expr_l140_140439

noncomputable def min_value (a b c : ℝ) := 4 * a^3 + 8 * b^3 + 18 * c^3 + 1 / (9 * a * b * c)

theorem min_value_expr (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  min_value a b c ≥ 8 / Real.sqrt 3 :=
by
  sorry

end min_value_expr_l140_140439


namespace days_B_can_finish_alone_l140_140399

theorem days_B_can_finish_alone (x : ℚ) : 
  (1 / 3 : ℚ) + (1 / x) = (1 / 2 : ℚ) → x = 6 := 
by
  sorry

end days_B_can_finish_alone_l140_140399


namespace sum_of_real_roots_l140_140325

theorem sum_of_real_roots (P : Polynomial ℝ) (hP : P = Polynomial.C 1 * X^4 - Polynomial.C 8 * X - Polynomial.C 2) :
  P.roots.sum = 2 :=
by {
  sorry
}

end sum_of_real_roots_l140_140325


namespace unique_solution_triple_l140_140696

theorem unique_solution_triple {a b c : ℝ} (h₀ : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h₁ : a^2 + b^2 + c^2 = 3) (h₂ : (a + b + c) * (a^2 * b + b^2 * c + c^2 * a) = 9) :
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ c = 1 ∧ b = 1) ∨ (b = 1 ∧ a = 1 ∧ c = 1) ∨ (b = 1 ∧ c = 1 ∧ a = 1) ∨ (c = 1 ∧ a = 1 ∧ b = 1) ∨ (c = 1 ∧ b = 1 ∧ a = 1) :=
sorry

end unique_solution_triple_l140_140696


namespace four_digit_numbers_count_l140_140579

open Nat

theorem four_digit_numbers_count :
  let valid_a := [5, 6]
  let valid_d := 0
  let valid_bc_pairs := [(3, 4), (3, 6)]
  valid_a.length * 1 * valid_bc_pairs.length = 4 :=
by
  sorry

end four_digit_numbers_count_l140_140579


namespace total_study_hours_during_semester_l140_140767

-- Definitions of the given conditions
def semester_weeks : ℕ := 15
def weekday_study_hours_per_day : ℕ := 3
def saturday_study_hours : ℕ := 4
def sunday_study_hours : ℕ := 5

-- Theorem statement to prove the total study hours during the semester
theorem total_study_hours_during_semester : 
  (semester_weeks * ((5 * weekday_study_hours_per_day) + saturday_study_hours + sunday_study_hours)) = 360 := by
  -- We are skipping the proof step and adding a placeholder
  sorry

end total_study_hours_during_semester_l140_140767


namespace f_of_3_is_log2_3_l140_140330

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_condition : ∀ x : ℝ, f (2 ^ x) = x

theorem f_of_3_is_log2_3 : f 3 = Real.log 3 / Real.log 2 := sorry

end f_of_3_is_log2_3_l140_140330


namespace nonempty_solution_iff_a_gt_one_l140_140072

theorem nonempty_solution_iff_a_gt_one (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < a) ↔ a > 1 :=
sorry

end nonempty_solution_iff_a_gt_one_l140_140072


namespace even_numbers_average_l140_140898

theorem even_numbers_average (n : ℕ) (h : (n / 2 * (2 + 2 * n)) / n = 16) : n = 15 :=
by
  have hn : n ≠ 0 := sorry -- n > 0 because the first some even numbers were mentioned
  have hn_pos : 0 < n / 2 * (2 + 2 * n) := sorry -- n / 2 * (2 + 2n) > 0
  sorry

end even_numbers_average_l140_140898


namespace carson_clawed_39_times_l140_140007

def wombats_count := 9
def wombat_claws_per := 4
def rheas_count := 3
def rhea_claws_per := 1

def wombat_total_claws := wombats_count * wombat_claws_per
def rhea_total_claws := rheas_count * rhea_claws_per
def total_claws := wombat_total_claws + rhea_total_claws

theorem carson_clawed_39_times : total_claws = 39 :=
  by sorry

end carson_clawed_39_times_l140_140007


namespace eight_points_on_circle_l140_140254

theorem eight_points_on_circle
  (R : ℝ) (hR : R > 0)
  (points : Fin 8 → (ℝ × ℝ))
  (hpoints : ∀ i : Fin 8, (points i).1 ^ 2 + (points i).2 ^ 2 ≤ R ^ 2) :
  ∃ (i j : Fin 8), i ≠ j ∧ (dist (points i) (points j) < R) :=
sorry

end eight_points_on_circle_l140_140254


namespace cos_trig_identity_l140_140711

theorem cos_trig_identity (α : Real) 
  (h : Real.cos (Real.pi / 6 - α) = 3 / 5) : 
  Real.cos (5 * Real.pi / 6 + α) = - (3 / 5) :=
by
  sorry

end cos_trig_identity_l140_140711


namespace julian_comic_pages_l140_140083

-- Definitions from conditions
def frames_per_page : ℝ := 143.0
def total_frames : ℝ := 1573.0

-- The theorem stating the proof problem
theorem julian_comic_pages : total_frames / frames_per_page = 11 :=
by
  sorry

end julian_comic_pages_l140_140083


namespace incorrect_option_l140_140985

theorem incorrect_option (a : ℝ) (h : a ≠ 0) : (a + 2) ^ 0 ≠ 1 ↔ a = -2 :=
by {
  sorry
}

end incorrect_option_l140_140985


namespace complement_of_angle_correct_l140_140577

noncomputable def complement_of_angle (α : ℝ) : ℝ := 90 - α

theorem complement_of_angle_correct (α : ℝ) (h : complement_of_angle α = 125 + 12 / 60) :
  complement_of_angle α = 35 + 12 / 60 :=
by
  sorry

end complement_of_angle_correct_l140_140577


namespace union_of_M_N_l140_140865

-- Definitions of sets M and N
def M : Set ℕ := {0, 1}
def N : Set ℕ := {1, 2}

-- The theorem to prove
theorem union_of_M_N : M ∪ N = {0, 1, 2} :=
  by sorry

end union_of_M_N_l140_140865


namespace man_and_son_work_together_l140_140660

theorem man_and_son_work_together (man_days son_days : ℕ) (h_man : man_days = 15) (h_son : son_days = 10) :
  (1 / (1 / man_days + 1 / son_days) = 6) :=
by
  rw [h_man, h_son]
  sorry

end man_and_son_work_together_l140_140660


namespace ratio_of_p_to_q_l140_140067

theorem ratio_of_p_to_q (p q r : ℚ) (h1: p = r * q) (h2: 18 / 7 + (2 * q - p) / (2 * q + p) = 3) : r = 29 / 10 :=
by
  sorry

end ratio_of_p_to_q_l140_140067


namespace find_integers_l140_140894

theorem find_integers (x y : ℕ) (h : 2 * x * y = 21 + 2 * x + y) : (x = 1 ∧ y = 23) ∨ (x = 6 ∧ y = 3) :=
by
  sorry

end find_integers_l140_140894


namespace arithmetic_series_sum_l140_140440

theorem arithmetic_series_sum : 
  let a := -41
  let d := 2
  let n := 22
  let l := 1
  let Sn := n * (a + l) / 2
  a = -41 ∧ d = 2 ∧ l = 1 ∧ n = 22 → Sn = -440 :=
by 
  intros a d n l Sn h
  sorry

end arithmetic_series_sum_l140_140440


namespace bridget_block_collection_l140_140677

-- Defining the number of groups and blocks per group.
def num_groups : ℕ := 82
def blocks_per_group : ℕ := 10

-- Defining the total number of blocks calculation.
def total_blocks : ℕ := num_groups * blocks_per_group

-- Theorem stating the total number of blocks is 820.
theorem bridget_block_collection : total_blocks = 820 :=
  by
  sorry

end bridget_block_collection_l140_140677


namespace evaluate_expression_l140_140689

theorem evaluate_expression:
  (125 = 5^3) ∧ (81 = 3^4) ∧ (32 = 2^5) → 
  125^(1/3) * 81^(-1/4) * 32^(1/5) = 10 / 3 := by
  sorry

end evaluate_expression_l140_140689


namespace alex_loan_comparison_l140_140310

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r * t)

theorem alex_loan_comparison :
  let P : ℝ := 15000
  let r1 : ℝ := 0.08
  let r2 : ℝ := 0.10
  let n : ℕ := 12
  let t1_10 : ℝ := 10
  let t1_5 : ℝ := 5
  let t2 : ℝ := 15
  let owed_after_10 := compound_interest P r1 n t1_10
  let payment_after_10 := owed_after_10 / 2
  let remaining_after_10 := owed_after_10 / 2
  let owed_after_15 := compound_interest remaining_after_10 r1 n t1_5
  let total_payment_option1 := payment_after_10 + owed_after_15
  let total_payment_option2 := simple_interest P r2 t2
  total_payment_option1 - total_payment_option2 = 4163 :=
by
  sorry

end alex_loan_comparison_l140_140310


namespace kelly_baking_powder_difference_l140_140647

theorem kelly_baking_powder_difference :
  let amount_yesterday := 0.4
  let amount_now := 0.3
  amount_yesterday - amount_now = 0.1 :=
by
  -- Definitions for amounts 
  let amount_yesterday := 0.4
  let amount_now := 0.3
  
  -- Applying definitions in the computation
  show amount_yesterday - amount_now = 0.1
  sorry

end kelly_baking_powder_difference_l140_140647


namespace passing_probability_l140_140095

theorem passing_probability :
  let num_students := 6
  let probability :=
    1 - (2/6) * (2/5) * (2/4) * (2/3) * (2/2)
  probability = 44 / 45 :=
by
  let num_students := 6
  let probability :=
    1 - (2/6) * (2/5) * (2/4) * (2/3) * (2/2)
  have p_eq : probability = 44 / 45 := sorry
  exact p_eq

end passing_probability_l140_140095


namespace number_of_full_rows_in_first_field_l140_140302

-- Define the conditions
def total_corn_cobs : ℕ := 116
def rows_in_second_field : ℕ := 16
def cobs_per_row : ℕ := 4
def cobs_in_second_field : ℕ := rows_in_second_field * cobs_per_row
def cobs_in_first_field : ℕ := total_corn_cobs - cobs_in_second_field

-- Define the theorem to be proven
theorem number_of_full_rows_in_first_field : 
  cobs_in_first_field / cobs_per_row = 13 :=
by
  sorry

end number_of_full_rows_in_first_field_l140_140302


namespace distance_between_city_A_and_B_is_180_l140_140620

theorem distance_between_city_A_and_B_is_180
  (D : ℝ)
  (h1 : ∀ T_C : ℝ, T_C = D / 30)
  (h2 : ∀ T_D : ℝ, T_D = T_C - 1)
  (h3 : ∀ V_D : ℝ, V_D > 36 → T_D = D / V_D) :
  D = 180 := 
by
  sorry

end distance_between_city_A_and_B_is_180_l140_140620


namespace num_diamonds_F10_l140_140962

def num_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 4 else 4 * (3 * n - 2)

theorem num_diamonds_F10 : num_diamonds 10 = 112 := by
  sorry

end num_diamonds_F10_l140_140962


namespace probability_red_or_blue_l140_140516

theorem probability_red_or_blue 
  (total_marbles : ℕ)
  (p_white p_green p_orange p_violet : ℚ)
  (h_total : total_marbles = 120)
  (h_white_prob : p_white = 1/5)
  (h_green_prob: p_green = 1/10)
  (h_orange_prob: p_orange = 1/6)
  (h_violet_prob: p_violet = 1/8)
  : (49 / 120 : ℚ) = 1 - (p_white + p_green + p_orange + p_violet) :=
by
  sorry

end probability_red_or_blue_l140_140516


namespace range_sum_of_h_l140_140841

noncomputable def h (x : ℝ) : ℝ := 5 / (5 + 3 * x^2)

theorem range_sum_of_h : 
  (∃ a b : ℝ, (∀ x : ℝ, 0 < h x ∧ h x ≤ 1) ∧ a = 0 ∧ b = 1 ∧ a + b = 1) :=
sorry

end range_sum_of_h_l140_140841


namespace land_value_moon_l140_140907

-- Define the conditions
def surface_area_earth : ℕ := 200
def surface_area_ratio : ℕ := 5
def value_ratio : ℕ := 6
def total_value_earth : ℕ := 80

-- Define the question and the expected answer
noncomputable def total_value_moon : ℕ := 96

-- State the proof problem
theorem land_value_moon :
  (surface_area_earth / surface_area_ratio * value_ratio) * (surface_area_earth / surface_area_ratio) = total_value_moon := 
sorry

end land_value_moon_l140_140907


namespace cos_double_angle_l140_140202

variable (α : ℝ)

theorem cos_double_angle (h1 : 0 < α ∧ α < π / 2) 
                         (h2 : Real.cos ( α + π / 4) = 3 / 5) : 
    Real.cos (2 * α) = 24 / 25 :=
by
  sorry

end cos_double_angle_l140_140202


namespace malcolm_walked_uphill_l140_140250

-- Define the conditions as variables and parameters
variables (x : ℕ)

-- Define the conditions given in the problem
def first_route_time := x + 2 * x + x
def second_route_time := 14 + 28
def time_difference := 18

-- Theorem statement - proving that Malcolm walked uphill for 6 minutes in the first route
theorem malcolm_walked_uphill : first_route_time - second_route_time = time_difference → x = 6 := by
  sorry

end malcolm_walked_uphill_l140_140250


namespace max_value_func1_l140_140296

theorem max_value_func1 (x : ℝ) (h : 0 < x ∧ x < 2) : 
  ∃ y, y = x * (4 - 2 * x) ∧ (∀ z, z = x * (4 - 2 * x) → z ≤ 2) :=
sorry

end max_value_func1_l140_140296


namespace calculate_expr_at_3_l140_140552

-- Definition of the expression
def expr (x : ℕ) : ℕ := (x + x * x^(x^2)) * 3

-- The proof statement
theorem calculate_expr_at_3 : expr 3 = 177156 := 
by
  sorry

end calculate_expr_at_3_l140_140552


namespace average_value_l140_140465

theorem average_value (x y z : ℝ) (h : (5/4) * (x + y + z) = 15) : (x + y + z) / 3 = 4 := 
by
  have h1 : x + y + z = 15 * (4 / 5) := sorry
  have h2 : x + y + z = 12 := sorry
  have h3 : (x + y + z) / 3 = 12 / 3 := by rw [h2]
  have h4 : 12 / 3 = 4 := sorry
  rw [h4] at h3
  exact h3

end average_value_l140_140465


namespace tan_a4_a12_eq_neg_sqrt3_l140_140033

-- Definitions and conditions
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

variables {a : ℕ → ℝ} (h_arith : is_arithmetic_sequence a)
          (h_sum : a 1 + a 8 + a 15 = Real.pi)

-- The main statement to prove
theorem tan_a4_a12_eq_neg_sqrt3 : 
  Real.tan (a 4 + a 12) = -Real.sqrt 3 :=
sorry

end tan_a4_a12_eq_neg_sqrt3_l140_140033


namespace investment_ratio_l140_140820

-- Definitions of all the conditions
variables (A B C profit b_share: ℝ)

-- Conditions based on the provided problem
def condition1 (n : ℝ) : Prop := A = n * B
def condition2 : Prop := B = (2 / 3) * C
def condition3 : Prop := profit = 4400
def condition4 : Prop := b_share = 800

-- The theorem we want to prove
theorem investment_ratio (n : ℝ) :
  (condition1 A B n) ∧ (condition2 B C) ∧ (condition3 profit) ∧ (condition4 b_share) → A / B = 3 :=
by
  sorry

end investment_ratio_l140_140820


namespace probability_no_correct_letter_for_7_envelopes_l140_140631

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1
  else n * factorial (n - 1)

def derangement (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement (n - 1) + derangement (n - 2))

noncomputable def probability_no_correct_letter (n : ℕ) : ℚ :=
  derangement n / factorial n

theorem probability_no_correct_letter_for_7_envelopes :
  probability_no_correct_letter 7 = 427 / 1160 :=
by sorry

end probability_no_correct_letter_for_7_envelopes_l140_140631


namespace two_numbers_and_sum_l140_140892

theorem two_numbers_and_sum (x y : ℕ) (hx : x * y = 18) (hy : x - y = 4) : x + y = 10 :=
sorry

end two_numbers_and_sum_l140_140892


namespace bernardo_vs_silvia_probability_l140_140005

theorem bernardo_vs_silvia_probability :
  let bernardo_set := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let silvia_set := {1, 2, 3, 4, 5, 6, 7, 8}
  let bernardo_choices := (bernardo_set.to_finset.powerset.filter (λ s, s.card = 3)).to_finset
  let silvia_choices := (silvia_set.to_finset.powerset.filter (λ s, s.card = 3)).to_finset
  let total_bernardo := bernardo_choices.card
  let total_silvia := silvia_choices.card
  let favorable_cases := (bernardo_choices.to_list.product silvia_choices.to_list).countp
    (λ (p : finset ℕ × finset ℕ), p.1.to_list.sort (· > ·) > p.2.to_list.sort (· > ·))
  let total_cases := total_bernardo * total_silvia
  (favorable_cases / total_cases : ℚ) = 37 / 56 :=
by sorry

end bernardo_vs_silvia_probability_l140_140005


namespace probability_exactly_one_each_is_correct_l140_140590

def probability_one_each (total forks spoons knives teaspoons : ℕ) : ℚ :=
  (forks * spoons * knives * teaspoons : ℚ) / ((total.choose 4) : ℚ)

theorem probability_exactly_one_each_is_correct :
  probability_one_each 34 8 9 10 7 = 40 / 367 :=
by sorry

end probability_exactly_one_each_is_correct_l140_140590


namespace loss_eq_cost_price_of_x_balls_l140_140492

theorem loss_eq_cost_price_of_x_balls (cp ball_count sp : ℕ) (cp_ball : ℕ) 
  (hc1 : cp_ball = 60) (hc2 : cp = ball_count * cp_ball) (hs : sp = 720) 
  (hb : ball_count = 17) :
  ∃ x : ℕ, (cp - sp = x * cp_ball) ∧ x = 5 :=
by
  sorry

end loss_eq_cost_price_of_x_balls_l140_140492


namespace sum_of_solutions_l140_140850

-- Given the quadratic equation: x^2 + 3x - 20 = 7x + 8
def quadratic_equation (x : ℝ) : Prop := x^2 + 3*x - 20 = 7*x + 8

-- Prove that the sum of the solutions to this quadratic equation is 4
theorem sum_of_solutions : 
  ∀ x1 x2 : ℝ, (quadratic_equation x1) ∧ (quadratic_equation x2) → x1 + x2 = 4 :=
by
  sorry

end sum_of_solutions_l140_140850


namespace number_of_real_roots_l140_140022

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem number_of_real_roots (a : ℝ) :
    ((|a| < (2 * Real.sqrt 3) / 9) → (∃ x₁ x₂ x₃ : ℝ, f x₁ = a ∧ f x₂ = a ∧ f x₃ = a ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)) ∧
    ((|a| > (2 * Real.sqrt 3) / 9) → (∃ x : ℝ, f x = a ∧ ∀ y : ℝ, f y = a → y = x)) ∧
    ((|a| = (2 * Real.sqrt 3) / 9) → (∃ x₁ x₂ : ℝ, f x₁ = a ∧ f x₂ = a ∧ x₁ ≠ x₂ ∧ ∀ y : ℝ, (f y = a → (y = x₁ ∨ y = x₂)) ∧ (x₁ = x₂ ∨ ∀ z : ℝ, (f z = a → z = x₁ ∨ z = x₂)))) := sorry

end number_of_real_roots_l140_140022


namespace kirill_is_62_5_l140_140242

variable (K : ℝ)

def kirill_height := K
def brother_height := K + 14
def sister_height := 2 * K
def total_height := K + (K + 14) + 2 * K

theorem kirill_is_62_5 (h1 : total_height K = 264) : K = 62.5 := by
  sorry

end kirill_is_62_5_l140_140242


namespace alternating_colors_probability_l140_140811

theorem alternating_colors_probability :
  let total_balls : ℕ := 10
  let white_balls : ℕ := 5
  let black_balls : ℕ := 5
  let successful_outcomes : ℕ := 2
  let total_outcomes : ℕ := Nat.choose total_balls white_balls
  (successful_outcomes : ℚ) / (total_outcomes : ℚ) = (1 / 126) := 
by
  let total_balls := 10
  let white_balls := 5
  let black_balls := 5
  let successful_outcomes := 2
  let total_outcomes := Nat.choose total_balls white_balls
  have h_total_outcomes : total_outcomes = 252 := sorry
  have h_probability : (successful_outcomes : ℚ) / (total_outcomes : ℚ) = (1 / 126) := sorry
  exact h_probability

end alternating_colors_probability_l140_140811


namespace two_pow_a_add_three_pow_b_eq_square_l140_140020

theorem two_pow_a_add_three_pow_b_eq_square (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) 
(h : 2 ^ a + 3 ^ b = n ^ 2) : (a = 4 ∧ b = 2) :=
sorry

end two_pow_a_add_three_pow_b_eq_square_l140_140020


namespace rhombus_diagonal_length_l140_140815

theorem rhombus_diagonal_length (side : ℝ) (shorter_diagonal : ℝ) 
  (h1 : side = 51) (h2 : shorter_diagonal = 48) : 
  ∃ longer_diagonal : ℝ, longer_diagonal = 90 :=
by
  sorry

end rhombus_diagonal_length_l140_140815


namespace rectangular_sheet_integers_l140_140545

noncomputable def at_least_one_integer (a b : ℝ) : Prop :=
  ∃ i : ℤ, a = i ∨ b = i

theorem rectangular_sheet_integers (a b : ℝ)
  (h_positive_a : a > 0)
  (h_positive_b : b > 0)
  (h_cut_lines : ∀ x y : ℝ, (∃ k : ℤ, x = k ∧ y = 1 ∨ y = k ∧ x = 1) → (∃ z : ℤ, x = z ∨ y = z)) :
  at_least_one_integer a b :=
sorry

end rectangular_sheet_integers_l140_140545


namespace train_length_is_150_l140_140159

noncomputable def train_length (v_km_hr : ℝ) (t_sec : ℝ) : ℝ :=
  let v_m_s := v_km_hr * (5 / 18)
  v_m_s * t_sec

theorem train_length_is_150 :
  train_length 122 4.425875438161669 = 150 :=
by
  -- It follows directly from the given conditions and known conversion factor
  -- The actual proof steps would involve arithmetic simplifications.
  sorry

end train_length_is_150_l140_140159


namespace second_shift_production_l140_140336

-- Question: Prove that the number of cars produced by the second shift is 1,100 given the conditions
-- Conditions:
-- 1. P_day = 4 * P_second
-- 2. P_day + P_second = 5,500

theorem second_shift_production (P_day P_second : ℕ) (h1 : P_day = 4 * P_second) (h2 : P_day + P_second = 5500) :
  P_second = 1100 := by
  sorry

end second_shift_production_l140_140336


namespace evaluate_expression_l140_140690

theorem evaluate_expression :
  125^(1/3 : ℝ) * 81^(-1/4 : ℝ) * 32^(1/5 : ℝ) = 10/3 := by
  sorry

end evaluate_expression_l140_140690


namespace evaluate_expression_l140_140688

theorem evaluate_expression : (125^(1/3 : ℝ)) * (81^(-1/4 : ℝ)) * (32^(1/5 : ℝ)) = (10 / 3 : ℝ) :=
by
  sorry

end evaluate_expression_l140_140688


namespace value_of_g_at_3_l140_140100

def g (x : ℝ) : ℝ := x^2 - 2 * x

theorem value_of_g_at_3 : g 3 = 3 := by
  sorry

end value_of_g_at_3_l140_140100


namespace percentage_x_y_l140_140467

variable (x y P : ℝ)

theorem percentage_x_y 
  (h1 : 0.5 * (x - y) = (P / 100) * (x + y))
  (h2 : y = (1 / 9) * x) : 
  P = 40 :=
sorry

end percentage_x_y_l140_140467


namespace find_c_l140_140729

theorem find_c (a b c d y1 y2 : ℝ) (h1 : y1 = a * 2^3 + b * 2^2 + c * 2 + d)
  (h2 : y2 = a * (-2)^3 + b * (-2)^2 + c * (-2) + d)
  (h3 : y1 - y2 = 12) : c = 3 - 4 * a := by
  sorry

end find_c_l140_140729


namespace second_horse_revolutions_l140_140304

noncomputable def circumference (radius : ℝ) : ℝ := 2 * Real.pi * radius
noncomputable def distance_traveled (circumference : ℝ) (revolutions : ℕ) : ℝ := circumference * (revolutions : ℝ)
noncomputable def revolutions_needed (distance : ℝ) (circumference : ℝ) : ℕ := ⌊distance / circumference⌋₊

theorem second_horse_revolutions :
  let r1 := 30
  let r2 := 10
  let revolutions1 := 40
  let c1 := circumference r1
  let c2 := circumference r2
  let d1 := distance_traveled c1 revolutions1
  (revolutions_needed d1 c2) = 120 :=
by
  sorry

end second_horse_revolutions_l140_140304


namespace animal_population_l140_140629

def total_population (L P E : ℕ) : ℕ :=
L + P + E

theorem animal_population 
    (L P E : ℕ) 
    (h1 : L = 2 * P) 
    (h2 : E = (L + P) / 2) 
    (h3 : L = 200) : 
  total_population L P E = 450 := 
  by 
    sorry

end animal_population_l140_140629


namespace original_area_area_after_translation_l140_140887

-- Defining vectors v, w, and t
def v : ℝ × ℝ := (6, -4)
def w : ℝ × ℝ := (-8, 3)
def t : ℝ × ℝ := (3, 2)

-- Function to compute the determinant of two vectors in R^2
def det (v w : ℝ × ℝ) : ℝ := v.1 * w.2 - v.2 * w.1

-- The area of a parallelogram is the absolute value of the determinant
def parallelogram_area (v w : ℝ × ℝ) : ℝ := |det v w|

-- Proving the original area is 14
theorem original_area : parallelogram_area v w = 14 := by
  sorry

-- Proving the area remains the same after translation
theorem area_after_translation : parallelogram_area v w = parallelogram_area (v.1 + t.1, v.2 + t.2) (w.1 + t.1, w.2 + t.2) := by
  sorry

end original_area_area_after_translation_l140_140887


namespace square_area_l140_140665

theorem square_area : ∃ (s: ℝ), (∀ x: ℝ, x^2 + 4*x + 1 = 7 → ∃ t: ℝ, t = x ∧ ∃ x2: ℝ, (x2 - x)^2 = s^2 ∧ ∀ y : ℝ, y = 7 ∧ y = x2^2 + 4*x2 + 1) ∧ s^2 = 40 :=
by
  sorry

end square_area_l140_140665


namespace ratio_volumes_equal_ratio_areas_l140_140608

-- Defining necessary variables and functions
variables (R : ℝ) (S_sphere S_cone V_sphere V_cone : ℝ)

-- Conditions
def surface_area_sphere : Prop := S_sphere = 4 * Real.pi * R^2
def volume_sphere : Prop := V_sphere = (4 / 3) * Real.pi * R^3
def volume_polyhedron : Prop := V_cone = (S_cone * R) / 3

-- Theorem statement
theorem ratio_volumes_equal_ratio_areas
  (h1 : surface_area_sphere R S_sphere)
  (h2 : volume_sphere R V_sphere)
  (h3 : volume_polyhedron R S_cone V_cone)
  : (V_sphere / V_cone) = (S_sphere / S_cone) :=
sorry

end ratio_volumes_equal_ratio_areas_l140_140608


namespace dynaco_shares_l140_140801

theorem dynaco_shares (M D : ℕ) 
  (h1 : M + D = 300)
  (h2 : 36 * M + 44 * D = 12000) : 
  D = 150 :=
sorry

end dynaco_shares_l140_140801


namespace find_digit_P_l140_140099

theorem find_digit_P (P Q R S T : ℕ) (digits : Finset ℕ) (h1 : digits = {1, 2, 3, 6, 8}) 
(h2 : P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ R ≠ S ∧ R ≠ T ∧ S ≠ T)
(h3 : P ∈ digits ∧ Q ∈ digits ∧ R ∈ digits ∧ S ∈ digits ∧ T ∈ digits)
(hPQR_div_6 : (100 * P + 10 * Q + R) % 6 = 0)
(hQRS_div_8 : (100 * Q + 10 * R + S) % 8 = 0)
(hRST_div_3 : (100 * R + 10 * S + T) % 3 = 0) : 
P = 2 := 
sorry

end find_digit_P_l140_140099


namespace trace_bag_weight_is_two_l140_140388

-- Define the weights of Gordon's shopping bags
def weight_gordon1 : ℕ := 3
def weight_gordon2 : ℕ := 7

-- Summarize Gordon's total weight
def total_weight_gordon : ℕ := weight_gordon1 + weight_gordon2

-- Provide necessary conditions from problem statement
def trace_bags_count : ℕ := 5
def trace_total_weight : ℕ := total_weight_gordon
def trace_one_bag_weight : ℕ := trace_total_weight / trace_bags_count

theorem trace_bag_weight_is_two : trace_one_bag_weight = 2 :=
by 
  -- Placeholder for proof
  sorry

end trace_bag_weight_is_two_l140_140388


namespace sequence_converges_to_zero_and_N_for_epsilon_l140_140502

theorem sequence_converges_to_zero_and_N_for_epsilon :
  (∀ ε > 0, ∃ N : ℕ, ∀ n > N, |1 / (n : ℝ) - 0| < ε) ∧ 
  (∃ N : ℕ, ∀ n > N, |1 / (n : ℝ)| < 0.001) :=
by
  sorry

end sequence_converges_to_zero_and_N_for_epsilon_l140_140502


namespace polygon_sides_l140_140782

-- Definition of the conditions used in the problem
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- Statement of the theorem
theorem polygon_sides (n : ℕ) (h : sum_of_interior_angles n = 1080) : n = 8 :=
by
  sorry  -- Proof placeholder

end polygon_sides_l140_140782


namespace largest_is_21_l140_140153

theorem largest_is_21(a b c d : ℕ) 
  (h1 : (a + b + c) / 3 + d = 17)
  (h2 : (a + b + d) / 3 + c = 21)
  (h3 : (a + c + d) / 3 + b = 23)
  (h4 : (b + c + d) / 3 + a = 29):
  d = 21 := 
sorry

end largest_is_21_l140_140153


namespace students_agreed_total_l140_140262

theorem students_agreed_total :
  let third_grade_agreed : ℕ := 154
  let fourth_grade_agreed : ℕ := 237
  third_grade_agreed + fourth_grade_agreed = 391 := 
by
  let third_grade_agreed : ℕ := 154
  let fourth_grade_agreed : ℕ := 237
  show third_grade_agreed + fourth_grade_agreed = 391
  sorry

end students_agreed_total_l140_140262


namespace processing_time_l140_140162

theorem processing_time 
  (pictures : ℕ) (minutes_per_picture : ℕ) (minutes_per_hour : ℕ)
  (h1 : pictures = 960) (h2 : minutes_per_picture = 2) (h3 : minutes_per_hour = 60) : 
  (pictures * minutes_per_picture) / minutes_per_hour = 32 :=
by 
  sorry

end processing_time_l140_140162


namespace fraction_power_multiplication_l140_140956

theorem fraction_power_multiplication :
  ( (5 / 8: ℚ) ^ 2 * (3 / 4) ^ 2 * (2 / 3) = 75 / 512) := 
  by
  sorry

end fraction_power_multiplication_l140_140956


namespace vents_per_zone_l140_140356

theorem vents_per_zone (total_cost : ℝ) (number_of_zones : ℝ) (cost_per_vent : ℝ) (h_total_cost : total_cost = 20000) (h_zones : number_of_zones = 2) (h_cost_per_vent : cost_per_vent = 2000) : 
  (total_cost / cost_per_vent) / number_of_zones = 5 :=
by 
  sorry

end vents_per_zone_l140_140356


namespace cube_surface_area_increase_l140_140922

theorem cube_surface_area_increase (s : ℝ) : 
  let original_surface_area := 6 * s^2
  let new_edge := 1.3 * s
  let new_surface_area := 6 * (new_edge)^2
  let percentage_increase := ((new_surface_area - original_surface_area) / original_surface_area) * 100
  percentage_increase = 69 := 
by
  sorry

end cube_surface_area_increase_l140_140922


namespace carrie_spent_l140_140292

-- Definitions derived from the problem conditions
def cost_of_one_tshirt : ℝ := 9.65
def number_of_tshirts : ℕ := 12

-- The statement to prove
theorem carrie_spent :
  cost_of_one_tshirt * number_of_tshirts = 115.80 :=
by
  sorry

end carrie_spent_l140_140292


namespace new_computer_price_l140_140588

theorem new_computer_price (d : ℕ) (h : 2 * d = 560) : d + 3 * d / 10 = 364 :=
by
  sorry

end new_computer_price_l140_140588


namespace find_positive_real_solution_l140_140697

theorem find_positive_real_solution (x : ℝ) (h1 : x > 0) (h2 : (x - 5) / 8 = 5 / (x - 8)) : x = 13 := 
sorry

end find_positive_real_solution_l140_140697


namespace interval_length_l140_140770

theorem interval_length (c : ℝ) (h : ∀ x : ℝ, 3 ≤ 3 * x + 4 ∧ 3 * x + 4 ≤ c → 
                             (3 * (x) + 4 ≤ c ∧ 3 ≤ 3 * x + 4)) :
  (∃ c : ℝ, ((c - 4) / 3) - ((-1) / 3) = 15) → (c - 3 = 45) :=
sorry

end interval_length_l140_140770


namespace calc_expression_l140_140835

-- Define the fractions and whole number in the problem
def frac1 : ℚ := 5/6
def frac2 : ℚ := 1 + 1/6
def whole : ℚ := 2

-- Define the expression to be proved
def expression : ℚ := (frac1) - (-whole) + (frac2)

-- The theorem to be proved
theorem calc_expression : expression = 4 :=
by { sorry }

end calc_expression_l140_140835


namespace tan_condition_then_expression_value_l140_140983

theorem tan_condition_then_expression_value (θ : ℝ) (h : Real.tan θ = 2) :
  (2 * Real.sin θ) / (Real.sin θ + 2 * Real.cos θ) = 1 :=
sorry

end tan_condition_then_expression_value_l140_140983


namespace ivanov_family_net_worth_l140_140137

theorem ivanov_family_net_worth :
  let apartment_value := 3000000
  let car_value := 900000
  let bank_deposit := 300000
  let securities_value := 200000
  let liquid_cash := 100000
  let mortgage_balance := 1500000
  let car_loan_balance := 500000
  let debt_to_relatives := 200000
  let total_assets := apartment_value + car_value + bank_deposit + securities_value + liquid_cash
  let total_liabilities := mortgage_balance + car_loan_balance + debt_to_relatives
  let net_worth := total_assets - total_liabilities
  in net_worth = 2300000 := 
by
  let apartment_value := 3000000
  let car_value := 900000
  let bank_deposit := 300000
  let securities_value := 200000
  let liquid_cash := 100000
  let mortgage_balance := 1500000
  let car_loan_balance := 500000
  let debt_to_relatives := 200000
  let total_assets := apartment_value + car_value + bank_deposit + securities_value + liquid_cash
  let total_liabilities := mortgage_balance + car_loan_balance + debt_to_relatives
  let net_worth := total_assets - total_liabilities
  show net_worth = 2300000 from sorry

end ivanov_family_net_worth_l140_140137


namespace medicine_duration_l140_140481

theorem medicine_duration (days_per_third_pill : ℕ) (pills : ℕ) (days_per_month : ℕ)
  (h1 : days_per_third_pill = 3)
  (h2 : pills = 90)
  (h3 : days_per_month = 30) :
  ((pills * (days_per_third_pill * 3)) / days_per_month) = 27 :=
sorry

end medicine_duration_l140_140481


namespace hcf_of_two_numbers_of_given_conditions_l140_140266

theorem hcf_of_two_numbers_of_given_conditions :
  ∃ B H, (588 = H * 84) ∧ H = Nat.gcd 588 B ∧ H = 7 :=
by
  use 84, 7
  have h₁ : 588 = 7 * 84 := by sorry
  have h₂ : 7 = Nat.gcd 588 84 := by sorry
  exact ⟨h₁, h₂, rfl⟩

end hcf_of_two_numbers_of_given_conditions_l140_140266


namespace polygon_angle_pairs_l140_140267

theorem polygon_angle_pairs
  {r k : ℕ}
  (h_ratio : (180 * r - 360) / r = (4 / 3) * (180 * k - 360) / k)
  (h_k_lt_15 : k < 15)
  (h_r_ge_3 : r ≥ 3) :
  (k = 7 ∧ r = 42) ∨ (k = 6 ∧ r = 18) ∨ (k = 5 ∧ r = 10) ∨ (k = 4 ∧ r = 6) :=
sorry

end polygon_angle_pairs_l140_140267


namespace Jessica_has_3_dozens_l140_140238

variable (j : ℕ)

def Sandy_red_marbles (j : ℕ) : ℕ := 4 * j * 12  

theorem Jessica_has_3_dozens {j : ℕ} : Sandy_red_marbles j = 144 → j = 3 := by
  intros h
  sorry

end Jessica_has_3_dozens_l140_140238


namespace stratified_sampling_by_edu_stage_is_reasonable_l140_140273

variable (visionConditions : String → Type) -- visionConditions for different sampling methods
variable (primaryVision : Type) -- vision condition for primary school
variable (juniorVision : Type) -- vision condition for junior high school
variable (seniorVision : Type) -- vision condition for senior high school
variable (insignificantDiffGender : Prop) -- insignificant differences between boys and girls

-- Given conditions
variable (sigDiffEduStage : Prop) -- significant differences between educational stages

-- Stating the theorem
theorem stratified_sampling_by_edu_stage_is_reasonable (h1 : sigDiffEduStage) (h2 : insignificantDiffGender) : 
  visionConditions "Stratified_sampling_by_educational_stage" = visionConditions C :=
sorry

end stratified_sampling_by_edu_stage_is_reasonable_l140_140273


namespace number_of_pizzas_ordered_l140_140482

-- Define the total number of people
def total_people : ℕ := 6

-- Define the number of slices per pizza
def slices_per_pizza : ℕ := 8

-- Define the number of slices each person ate
def slices_per_person : ℕ := 4

-- Define the total number of slices eaten
def total_slices_eaten : ℕ := total_people * slices_per_person

-- Prove that the number of pizzas needed is 3
theorem number_of_pizzas_ordered : total_slices_eaten / slices_per_pizza = 3 := by
  sorry

end number_of_pizzas_ordered_l140_140482


namespace range_a_l140_140978

theorem range_a (a : ℝ) (x : ℝ) : 
    (∀ x, (x = 1 → x - a ≥ 1) ∧ (x = -1 → ¬(x - a ≥ 1))) ↔ (-2 < a ∧ a ≤ 0) :=
by
  sorry

end range_a_l140_140978


namespace triangle_is_obtuse_l140_140221

def is_obtuse_triangle (a b c : ℕ) : Prop := a^2 + b^2 < c^2

theorem triangle_is_obtuse :
    is_obtuse_triangle 4 6 8 :=
by
    sorry

end triangle_is_obtuse_l140_140221


namespace greatest_third_side_l140_140283

theorem greatest_third_side
  (a b : ℕ)
  (h₁ : a = 7)
  (h₂ : b = 10)
  (c : ℕ)
  (h₃ : a + b + c ≤ 30)
  (h₄ : 3 < c)
  (h₅ : c ≤ 13) :
  c = 13 := 
sorry

end greatest_third_side_l140_140283


namespace rent_of_first_apartment_l140_140661

theorem rent_of_first_apartment (R : ℝ) :
  let cost1 := R + 260 + (31 * 20 * 0.58)
  let cost2 := 900 + 200 + (21 * 20 * 0.58)
  (cost1 - cost2 = 76) → R = 800 :=
by
  intro h
  sorry

end rent_of_first_apartment_l140_140661


namespace card_distribution_l140_140132

-- Definitions of the total cards and distribution rules
def total_cards : ℕ := 363

def ratio_xiaoming_xiaohua (k : ℕ) : Prop := ∃ x y, x = 7 * k ∧ y = 6 * k
def ratio_xiaogang_xiaoming (m : ℕ) : Prop := ∃ x z, z = 8 * m ∧ x = 5 * m

-- Final values to prove
def xiaoming_cards : ℕ := 105
def xiaohua_cards : ℕ := 90
def xiaogang_cards : ℕ := 168

-- The proof statement
theorem card_distribution (x y z k m : ℕ) 
  (hk : total_cards = 7 * k + 6 * k + 8 * m)
  (hx : ratio_xiaoming_xiaohua k)
  (hz : ratio_xiaogang_xiaoming m) :
  x = xiaoming_cards ∧ y = xiaohua_cards ∧ z = xiaogang_cards :=
by
  -- Placeholder for the proof
  sorry

end card_distribution_l140_140132


namespace trigonometric_identity_l140_140327

theorem trigonometric_identity (α : Real) (h : Real.sin (Real.pi + α) = -1/3) : 
  (Real.sin (2 * α) / Real.cos α) = 2/3 :=
by
  sorry

end trigonometric_identity_l140_140327


namespace height_of_first_podium_l140_140997

noncomputable def height_of_podium_2_cm := 53.0
noncomputable def height_of_podium_2_mm := 7.0
noncomputable def height_on_podium_2_cm := 190.0
noncomputable def height_on_podium_1_cm := 232.0
noncomputable def height_on_podium_1_mm := 5.0

def expected_height_of_podium_1_cm := 96.2

theorem height_of_first_podium :
  let height_podium_2 := height_of_podium_2_cm + height_of_podium_2_mm / 10.0
  let height_podium_1 := height_on_podium_1_cm + height_on_podium_1_mm / 10.0
  let hyeonjoo_height := height_on_podium_2_cm - height_podium_2
  height_podium_1 - hyeonjoo_height = expected_height_of_podium_1_cm :=
by sorry

end height_of_first_podium_l140_140997


namespace circleAtBottomAfterRotation_l140_140736

noncomputable def calculateFinalCirclePosition (initialPosition : String) (sides : ℕ) : String :=
  if (sides = 8) then (if initialPosition = "bottom" then "bottom" else "unknown") else "unknown"

theorem circleAtBottomAfterRotation :
  calculateFinalCirclePosition "bottom" 8 = "bottom" :=
by
  sorry

end circleAtBottomAfterRotation_l140_140736


namespace sum_infinite_geometric_l140_140566

theorem sum_infinite_geometric (a r : ℝ) (ha : a = 2) (hr : r = 1/3) : 
  ∑' n : ℕ, a * r^n = 3 := by
  sorry

end sum_infinite_geometric_l140_140566


namespace roots_sum_condition_l140_140087

theorem roots_sum_condition (a b : ℝ) 
  (h1 : ∃ (x y z : ℝ), (x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 9) 
    ∧ (x * y + y * z + x * z = a) ∧ (x * y * z = b)) :
  a + b = 38 := 
sorry

end roots_sum_condition_l140_140087


namespace investment_rate_l140_140000

theorem investment_rate (total_investment : ℝ) (invest1 : ℝ) (rate1 : ℝ) (invest2 : ℝ) (rate2 : ℝ) (desired_income : ℝ) (remaining_investment : ℝ) (remaining_rate : ℝ) : 
( total_investment = 12000 ∧ invest1 = 5000 ∧ rate1 = 0.06 ∧ invest2 = 4000 ∧ rate2 = 0.035 ∧ desired_income = 700 ∧ remaining_investment = 3000 ) → remaining_rate = 0.0867 :=
by
  sorry

end investment_rate_l140_140000


namespace quadratic_equation_same_solutions_l140_140883

theorem quadratic_equation_same_solutions :
  ∃ b c : ℝ, (b, c) = (1, -7) ∧ (∀ x : ℝ, (x - 3 = 4 ∨ 3 - x = 4) ↔ (x^2 + b * x + c = 0)) :=
by
  sorry

end quadratic_equation_same_solutions_l140_140883


namespace max_intersections_quadrilateral_l140_140519

-- Define intersection properties
def max_intersections_side : ℕ := 2
def sides_of_quadrilateral : ℕ := 4

theorem max_intersections_quadrilateral : 
  (max_intersections_side * sides_of_quadrilateral) = 8 :=
by 
  -- The proof goes here
  sorry

end max_intersections_quadrilateral_l140_140519


namespace summation_values_l140_140485

theorem summation_values (x y : ℝ) (h1 : x = y * (3 - y) ^ 2) (h2 : y = x * (3 - x) ^ 2) : 
  x + y = 0 ∨ x + y = 3 ∨ x + y = 4 ∨ x + y = 5 ∨ x + y = 8 :=
sorry

end summation_values_l140_140485


namespace seat_39_l140_140513

-- Defining the main structure of the problem
def circle_seating_arrangement (n k : ℕ) : ℕ :=
  if k = 1 then 1
  else sorry -- The pattern-based implementation goes here

-- The theorem to state the problem
theorem seat_39 (n k : ℕ) (h_n : n = 128) (h_k : k = 39) :
  circle_seating_arrangement n k = 51 :=
sorry

end seat_39_l140_140513


namespace inverse_proportion_range_l140_140461

theorem inverse_proportion_range (k : ℝ) (x : ℝ) :
  (∀ x : ℝ, (x < 0 -> (k - 1) / x > 0) ∧ (x > 0 -> (k - 1) / x < 0)) -> k < 1 :=
by
  sorry

end inverse_proportion_range_l140_140461


namespace quadratic_completion_l140_140335

noncomputable def find_b (n : ℝ) : ℝ := 2 * n

theorem quadratic_completion (n b : ℝ)
  (h1 : (x : ℝ) : (x + n)^2 + 16 = x^2 + b * x + 24)
  (h2 : b > 0) : 
  b = find_b (real.sqrt 2) :=
sorry

end quadratic_completion_l140_140335


namespace problem_one_l140_140538

def S_n (n : Nat) : Nat := 
  List.foldl (fun acc x => acc * 10 + 2) 0 (List.replicate n 2)

theorem problem_one : ∃ n ∈ Finset.range 2011, S_n n % 2011 = 0 := 
  sorry

end problem_one_l140_140538


namespace ivanov_family_net_worth_l140_140136

-- Define the financial values
def value_of_apartment := 3000000
def market_value_of_car := 900000
def bank_savings := 300000
def value_of_securities := 200000
def liquid_cash := 100000
def remaining_mortgage := 1500000
def car_loan := 500000
def debt_to_relatives := 200000

-- Calculate total assets and total liabilities
def total_assets := value_of_apartment + market_value_of_car + bank_savings + value_of_securities + liquid_cash
def total_liabilities := remaining_mortgage + car_loan + debt_to_relatives

-- Define the hypothesis and the final result of the net worth calculation
theorem ivanov_family_net_worth : total_assets - total_liabilities = 2300000 := by
  sorry

end ivanov_family_net_worth_l140_140136


namespace cube_surface_area_increase_l140_140921

theorem cube_surface_area_increase (s : ℝ) : 
  let original_surface_area := 6 * s^2
  let new_edge := 1.3 * s
  let new_surface_area := 6 * (new_edge)^2
  let percentage_increase := ((new_surface_area - original_surface_area) / original_surface_area) * 100
  percentage_increase = 69 := 
by
  sorry

end cube_surface_area_increase_l140_140921


namespace fourth_term_correct_l140_140422

def fourth_term_sequence : Nat :=
  4^0 + 4^1 + 4^2 + 4^3

theorem fourth_term_correct : fourth_term_sequence = 85 :=
by
  sorry

end fourth_term_correct_l140_140422


namespace find_r_l140_140246

theorem find_r (a b m p r : ℝ) (h_roots1 : a * b = 6) 
  (h_eq1 : ∀ x, x^2 - m*x + 6 = 0) 
  (h_eq2 : ∀ x, x^2 - p*x + r = 0) :
  r = 32 / 3 :=
by
  sorry

end find_r_l140_140246


namespace arrangement_books_l140_140941

def combination (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)

theorem arrangement_books : combination 9 4 = 126 := by
  sorry

end arrangement_books_l140_140941


namespace paying_students_pay_7_l140_140675

/-- At a school, 40% of the students receive a free lunch. 
These lunches are paid for by making sure the price paid by the 
paying students is enough to cover everyone's meal. 
It costs $210 to feed 50 students. 
Prove that each paying student pays $7. -/
theorem paying_students_pay_7 (total_students : ℕ) 
  (free_lunch_percentage : ℤ)
  (cost_per_50_students : ℕ) : 
  free_lunch_percentage = 40 ∧ cost_per_50_students = 210 →
  ∃ (paying_students_pay : ℕ), paying_students_pay = 7 :=
by
  -- Let the proof steps and conditions be set up as follows
  -- (this part is not required, hence using sorry)
  sorry

end paying_students_pay_7_l140_140675


namespace fraction_meaningful_l140_140272

theorem fraction_meaningful (x : ℝ) : 2 * x - 1 ≠ 0 ↔ x ≠ 1 / 2 :=
by
  sorry

end fraction_meaningful_l140_140272


namespace sin_cos_sixth_power_l140_140599

theorem sin_cos_sixth_power (θ : ℝ) 
  (h : Real.sin (3 * θ) = 1 / 2) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 11 / 12 :=
  sorry

end sin_cos_sixth_power_l140_140599


namespace integer_solutions_to_equation_l140_140847

-- Define the problem statement in Lean 4
theorem integer_solutions_to_equation :
  ∀ (x y : ℤ), (x ≠ 0) → (y ≠ 0) → (1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 19) →
      (x, y) = (38, 38) ∨ (x, y) = (380, 20) ∨ (x, y) = (-342, 18) ∨ 
      (x, y) = (20, 380) ∨ (x, y) = (18, -342) :=
by
  sorry

end integer_solutions_to_equation_l140_140847


namespace wrongly_copied_value_l140_140622

theorem wrongly_copied_value (mean_initial mean_correct : ℕ) (n : ℕ) 
  (wrong_copied_value : ℕ) (total_sum_initial total_sum_correct : ℕ) : 
  (mean_initial = 150) ∧ (mean_correct = 151) ∧ (n = 30) ∧ 
  (wrong_copied_value = 135) ∧ (total_sum_initial = n * mean_initial) ∧ 
  (total_sum_correct = n * mean_correct) → 
  (total_sum_correct - (total_sum_initial - wrong_copied_value) + wrong_copied_value = 300) :=
by
  intros h
  have h1 : mean_initial = 150 := by sorry
  have h2 : mean_correct = 151 := by sorry
  have h3 : n = 30 := by sorry
  have h4 : wrong_copied_value = 135 := by sorry
  have h5 : total_sum_initial = n * mean_initial := by sorry
  have h6 : total_sum_correct = n * mean_correct := by sorry
  sorry -- This is where the proof would go, but is not required per instructions.

end wrongly_copied_value_l140_140622


namespace single_intersection_l140_140727

theorem single_intersection (k : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, y^2 = x ∧ y + 1 = k * x) ↔ (k = 0 ∨ k = -1 / 4) :=
sorry

end single_intersection_l140_140727


namespace eval_expression_l140_140687

theorem eval_expression : (125 ^ (1/3) * 81 ^ (-1/4) * 32 ^ (1/5) = (10/3)) :=
by
  have h1 : 125 = 5^3 := by norm_num
  have h2 : 81 = 3^4 := by norm_num
  have h3 : 32 = 2^5 := by norm_num
  sorry

end eval_expression_l140_140687


namespace carson_clawed_total_l140_140008

theorem carson_clawed_total :
  let wombats := 9
  let wombat_claws := 4
  let rheas := 3
  let rhea_claws := 1
  wombats * wombat_claws + rheas * rhea_claws = 39 := by
  let wombats := 9
  let wombat_claws := 4
  let rheas := 3
  let rhea_claws := 1
  show wombats * wombat_claws + rheas * rhea_claws = 39
  sorry

end carson_clawed_total_l140_140008


namespace number_division_l140_140527

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l140_140527


namespace smallest_positive_debt_resolves_l140_140639

theorem smallest_positive_debt_resolves :
  ∃ (c t : ℤ), (240 * c + 180 * t = 60) ∧ (60 > 0) :=
by
  sorry

end smallest_positive_debt_resolves_l140_140639


namespace unique_x0_implies_a_in_range_l140_140578

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp x * (3 * x - 1) - a * x + a

theorem unique_x0_implies_a_in_range :
  ∃ x0 : ℤ, f x0 a ≤ 0 ∧ a < 1 -> a ∈ Set.Ico (2 / Real.exp 1) 1 := 
sorry

end unique_x0_implies_a_in_range_l140_140578


namespace frequency_of_zero_in_3021004201_l140_140080

def digit_frequency (n : Nat) (d : Nat) :  Rat :=
  let digits := n.digits 10
  let count_d := digits.count d
  (count_d : Rat) / digits.length

theorem frequency_of_zero_in_3021004201 : 
  digit_frequency 3021004201 0 = 0.4 := 
by 
  sorry

end frequency_of_zero_in_3021004201_l140_140080


namespace drawing_probability_consecutive_order_l140_140653

theorem drawing_probability_consecutive_order :
  let total_ways := Nat.factorial 12
  let desired_ways := (1 * Nat.factorial 3 * Nat.factorial 5)
  let probability := desired_ways / total_ways
  probability = 1 / 665280 :=
by
  let total_ways := Nat.factorial 12
  let desired_ways := (1 * Nat.factorial 3 * Nat.factorial 5)
  let probability := desired_ways / total_ways
  sorry

end drawing_probability_consecutive_order_l140_140653


namespace inequality_holds_l140_140607

theorem inequality_holds (k n : ℕ) (x : ℝ) (hx1 : 0 ≤ x) (hx2 : x ≤ 1) :
  (1 - (1 - x)^n)^k ≥ 1 - (1 - x^k)^n :=
by
  sorry

end inequality_holds_l140_140607


namespace smallest_positive_period_max_min_values_l140_140866

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.sin x)
noncomputable def f (x : ℝ) : ℝ := (vector_a x).1 * (vector_b x).1 + (vector_a x).2 * (vector_b x).2 - 1 / 2

-- Theorem 1: Smallest positive period of the function f(x)
theorem smallest_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi :=
  sorry

-- Theorem 2: Maximum and minimum values of the function f(x) on [0, π/2]
theorem max_min_values : 
  ∀ x ∈ Set.Icc 0 (Real.pi / 2),
    f x ≤ 1 ∧ f x ≥ -1 / 2 ∧ (∃ (x_max : ℝ), x_max ∈ Set.Icc 0 (Real.pi / 2) ∧ f x_max = 1) ∧
    (∃ (x_min : ℝ), x_min ∈ Set.Icc 0 (Real.pi / 2) ∧ f x_min = -1 / 2) :=
  sorry

end smallest_positive_period_max_min_values_l140_140866


namespace polygon_sides_with_diagonals_44_l140_140848

theorem polygon_sides_with_diagonals_44 (n : ℕ) (hD : 44 = n * (n - 3) / 2) : n = 11 :=
by
  sorry

end polygon_sides_with_diagonals_44_l140_140848


namespace find_number_l140_140530

theorem find_number (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 := 
sorry

end find_number_l140_140530


namespace units_digit_7_pow_5_l140_140793

theorem units_digit_7_pow_5 : (7^5) % 10 = 7 := 
by
  sorry

end units_digit_7_pow_5_l140_140793


namespace animal_population_l140_140630

def total_population (L P E : ℕ) : ℕ :=
L + P + E

theorem animal_population 
    (L P E : ℕ) 
    (h1 : L = 2 * P) 
    (h2 : E = (L + P) / 2) 
    (h3 : L = 200) : 
  total_population L P E = 450 := 
  by 
    sorry

end animal_population_l140_140630


namespace even_function_has_specific_m_l140_140070

theorem even_function_has_specific_m (m : ℝ) (f : ℝ → ℝ) (h_def : ∀ x : ℝ, f x = x^2 + (m - 1) * x - 3) (h_even : ∀ x : ℝ, f x = f (-x)) :
  m = 1 :=
by
  sorry

end even_function_has_specific_m_l140_140070


namespace amount_of_bill_is_720_l140_140512

-- Definitions and conditions
def TD : ℝ := 360
def BD : ℝ := 428.21

-- The relationship between TD, BD, and FV
axiom relationship (FV : ℝ) : BD = TD + (TD * BD) / (FV - TD)

-- The main theorem to prove
theorem amount_of_bill_is_720 : ∃ FV : ℝ, BD = TD + (TD * BD) / (FV - TD) ∧ FV = 720 :=
by
  use 720
  sorry

end amount_of_bill_is_720_l140_140512


namespace increasing_or_decreasing_subseq_l140_140359

theorem increasing_or_decreasing_subseq (a : Fin (m * n + 1) → ℝ) :
  ∃ (s : Fin (m + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (s i) ≤ a (s j)) ∨
  ∃ (t : Fin (n + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (t i) ≥ a (t j)) :=
sorry

end increasing_or_decreasing_subseq_l140_140359


namespace ratio_volumes_l140_140785

variables (V1 V2 : ℝ)
axiom h1 : (3 / 5) * V1 = (2 / 3) * V2

theorem ratio_volumes : V1 / V2 = 10 / 9 := by
  sorry

end ratio_volumes_l140_140785


namespace ratio_of_building_heights_l140_140657

theorem ratio_of_building_heights (F_h F_s A_s B_s : ℝ) (hF_h : F_h = 18) (hF_s : F_s = 45)
  (hA_s : A_s = 60) (hB_s : B_s = 72) :
  let h_A := (F_h / F_s) * A_s
  let h_B := (F_h / F_s) * B_s
  (h_A / h_B) = 5 / 6 :=
by
  sorry

end ratio_of_building_heights_l140_140657


namespace unit_digit_2_pow_15_l140_140253

theorem unit_digit_2_pow_15 : (2^15) % 10 = 8 := by
  sorry

end unit_digit_2_pow_15_l140_140253


namespace exists_real_polynomial_l140_140428

noncomputable def has_negative_coeff (p : Polynomial ℝ) : Prop :=
  ∃ i, (p.coeff i) < 0

noncomputable def all_positive_coeff (n : ℕ) (p : Polynomial ℝ) : Prop :=
  ∀ i, (Polynomial.derivative^[n] p).coeff i > 0

theorem exists_real_polynomial :
  ∃ p : Polynomial ℝ, has_negative_coeff p ∧ (∀ n > 1, all_positive_coeff n p) :=
sorry

end exists_real_polynomial_l140_140428


namespace expression_simplification_l140_140808

theorem expression_simplification :
  (- (1 / 2)) ^ 2023 * 2 ^ 2024 = -2 :=
by
  sorry

end expression_simplification_l140_140808


namespace remainder_of_exponentiated_sum_modulo_seven_l140_140316

theorem remainder_of_exponentiated_sum_modulo_seven :
  (9^6 + 8^8 + 7^9) % 7 = 2 := by
  sorry

end remainder_of_exponentiated_sum_modulo_seven_l140_140316


namespace range_of_a_l140_140035

theorem range_of_a {a : ℝ} (h : ∀ x : ℝ, a * x^2 + a * x + 1 > 0) : a ∈ Set.Icc 0 4 :=
sorry

end range_of_a_l140_140035


namespace average_movers_per_hour_l140_140352

-- Define the main problem parameters
def total_people : ℕ := 3200
def days : ℕ := 4
def hours_per_day : ℕ := 24
def total_hours : ℕ := hours_per_day * days
def average_people_per_hour := total_people / total_hours

-- State the theorem to prove
theorem average_movers_per_hour :
  average_people_per_hour = 33 :=
by
  -- Proof is omitted
  sorry

end average_movers_per_hour_l140_140352


namespace find_n_for_geometric_series_l140_140950

theorem find_n_for_geometric_series
  (n : ℝ)
  (a1 : ℝ := 12)
  (a2 : ℝ := 4)
  (r1 : ℝ)
  (S1 : ℝ)
  (b1 : ℝ := 12)
  (b2 : ℝ := 4 + n)
  (r2 : ℝ)
  (S2 : ℝ) :
  (r1 = a2 / a1) →
  (S1 = a1 / (1 - r1)) →
  (S2 = 4 * S1) →
  (r2 = b2 / b1) →
  (S2 = b1 / (1 - r2)) →
  n = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_n_for_geometric_series_l140_140950


namespace negation_exists_to_forall_l140_140103

theorem negation_exists_to_forall :
  (¬ ∃ x : ℝ, x^2 + 2*x - 3 > 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0) :=
by
  sorry

end negation_exists_to_forall_l140_140103


namespace cube_surface_area_increase_l140_140924

theorem cube_surface_area_increase (s : ℝ) : 
    let initial_area := 6 * s^2
    let new_edge := 1.3 * s
    let new_area := 6 * (new_edge)^2
    let incr_area := new_area - initial_area
    let percentage_increase := (incr_area / initial_area) * 100
    percentage_increase = 69 :=
by
  let initial_area := 6 * s^2
  let new_edge := 1.3 * s
  let new_area := 6 * (new_edge)^2
  let incr_area := new_area - initial_area
  let percentage_increase := (incr_area / initial_area) * 100
  sorry

end cube_surface_area_increase_l140_140924


namespace not_age_of_child_digit_l140_140893

variable {n : Nat}

theorem not_age_of_child_digit : 
  ∀ (ages : List Nat), 
    (∀ x ∈ ages, 5 ≤ x ∧ x ≤ 13) ∧ -- condition 1
    ages.Nodup ∧                    -- condition 2: distinct ages
    ages.length = 9 ∧               -- condition 1: 9 children
    (∃ num : Nat, 
       10000 ≤ num ∧ num < 100000 ∧         -- 5-digit number
       (∀ d : Nat, d ∈ num.digits 10 →     -- condition 3 & 4: each digit appears once and follows a consecutive pattern in increasing order
          1 ≤ d ∧ d ≤ 9) ∧
       (∀ age ∈ ages, num % age = 0)       -- condition 4: number divisible by all children's ages
    ) →
    ¬(9 ∈ ages) :=                         -- question: Prove that '9' is not the age of any child
by
  intro ages h
  -- The proof would go here
  sorry

end not_age_of_child_digit_l140_140893


namespace simply_connected_polyhedron_faces_l140_140895

def polyhedron_faces_condition (σ3 σ4 σ5 : Nat) (V E F : Nat) : Prop :=
  V - E + F = 2

theorem simply_connected_polyhedron_faces : 
  ∀ (σ3 σ4 σ5 : Nat) (V E F : Nat),
  polyhedron_faces_condition σ3 σ4 σ5 V E F →
  (σ4 = 0 ∧ σ5 = 0 → σ3 ≥ 4) ∧
  (σ3 = 0 ∧ σ5 = 0 → σ4 ≥ 6) ∧
  (σ3 = 0 ∧ σ4 = 0 → σ5 ≥ 12) := 
by
  intros
  sorry

end simply_connected_polyhedron_faces_l140_140895


namespace smallest_four_digit_number_divisible_by_4_l140_140286

theorem smallest_four_digit_number_divisible_by_4 : 
  ∃ n : ℕ, (1000 ≤ n ∧ n < 10000) ∧ (n % 4 = 0) ∧ n = 1000 := by
  sorry

end smallest_four_digit_number_divisible_by_4_l140_140286


namespace arithmetic_sequence_difference_l140_140313

theorem arithmetic_sequence_difference (a d : ℕ) (n m : ℕ) (hnm : m > n) (h_a : a = 3) (h_d : d = 7) (h_n : n = 1001) (h_m : m = 1004) :
  (a + (m - 1) * d) - (a + (n - 1) * d) = 21 :=
by
  sorry

end arithmetic_sequence_difference_l140_140313


namespace Timmy_ramp_speed_l140_140270

theorem Timmy_ramp_speed
  (h : ℤ)
  (v_required : ℤ)
  (v1 v2 v3 : ℤ)
  (average_speed : ℤ) :
  (h = 50) →
  (v_required = 40) →
  (v1 = 36) →
  (v2 = 34) →
  (v3 = 38) →
  average_speed = (v1 + v2 + v3) / 3 →
  v_required - average_speed = 4 :=
by
  intros h_val v_required_val v1_val v2_val v3_val avg_speed_val
  rw [h_val, v_required_val, v1_val, v2_val, v3_val, avg_speed_val]
  sorry

end Timmy_ramp_speed_l140_140270


namespace percentage_difference_l140_140580

theorem percentage_difference :
  ((75 / 100 : ℝ) * 40 - (4 / 5 : ℝ) * 25) = 10 := 
by
  sorry

end percentage_difference_l140_140580


namespace find_a_l140_140039

-- Define what it means for P(X = k) to be given by a particular function
def P (X : ℕ) (a : ℕ) := X / (2 * a)

-- Define the condition on the probabilities
def sum_of_probabilities_is_one (a : ℕ) :=
  (1 / (2 * a) + 2 / (2 * a) + 3 / (2 * a) + 4 / (2 * a)) = 1

-- The theorem to prove
theorem find_a (a : ℕ) (h : sum_of_probabilities_is_one a) : a = 5 :=
by sorry

end find_a_l140_140039


namespace q_can_be_true_or_false_l140_140071

-- Define the propositions p and q
variables (p q : Prop)

-- The assumptions given in the problem
axiom h1 : ¬ (p ∧ q)
axiom h2 : ¬ p

-- The statement we want to prove
theorem q_can_be_true_or_false : ∀ q, q ∨ ¬ q :=
by
  intro q
  exact em q -- Use the principle of excluded middle

end q_can_be_true_or_false_l140_140071


namespace number_of_divisors_of_3003_l140_140184

theorem number_of_divisors_of_3003 :
  ∃ d, d = 16 ∧ 
  (3003 = 3^1 * 7^1 * 11^1 * 13^1) →
  d = (1 + 1) * (1 + 1) * (1 + 1) * (1 + 1) := 
by 
  sorry

end number_of_divisors_of_3003_l140_140184


namespace find_inequality_solution_l140_140318

theorem find_inequality_solution :
  {x : ℝ | (x + 1) / (x - 2) + (x + 3) / (2 * x + 1) ≤ 2}
  = {x : ℝ | -1 / 2 ≤ x ∧ x ≤ 1 ∨ 2 ≤ x ∧ x ≤ 9} :=
by
  -- The proof steps are omitted.
  sorry

end find_inequality_solution_l140_140318


namespace approximation_of_11_28_relative_to_10000_l140_140167

def place_value_to_approximate (x : Float) (reference : Float) : String :=
  if x < reference / 10 then "tens"
  else if x < reference / 100 then "hundreds"
  else if x < reference / 1000 then "thousands"
  else if x < reference / 10000 then "ten thousands"
  else "greater than ten thousands"

theorem approximation_of_11_28_relative_to_10000:
  place_value_to_approximate 11.28 10000 = "hundreds" :=
by
  -- Insert proof here
  sorry

end approximation_of_11_28_relative_to_10000_l140_140167


namespace problem1_problem2_l140_140553

-- Define the first problem
theorem problem1 : ( (9 / 4) ^ (1 / 2) - (-8.6) ^ 0 - (8 / 27) ^ (-1 / 3)) = -1 := by
  sorry

-- Define the second problem
theorem problem2 : log 10 25 + log 10 4 + 7 ^ (log 7 2) + 2 * log 3 (sqrt 3) = 5 := by
  sorry

end problem1_problem2_l140_140553


namespace total_number_of_chips_l140_140284

theorem total_number_of_chips 
  (viviana_chocolate : ℕ) (susana_chocolate : ℕ) (viviana_vanilla : ℕ) (susana_vanilla : ℕ)
  (manuel_vanilla : ℕ) (manuel_chocolate : ℕ)
  (h1 : viviana_chocolate = susana_chocolate + 5)
  (h2 : susana_vanilla = (3 * viviana_vanilla) / 4)
  (h3 : viviana_vanilla = 20)
  (h4 : susana_chocolate = 25)
  (h5 : manuel_vanilla = 2 * susana_vanilla)
  (h6 : manuel_chocolate = viviana_chocolate / 2) :
  viviana_chocolate + susana_chocolate + manuel_chocolate + viviana_vanilla + susana_vanilla + manuel_vanilla = 135 :=
sorry

end total_number_of_chips_l140_140284


namespace largest_rectangle_area_l140_140775

theorem largest_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 60) : x * y ≤ 225 :=
by
  sorry

end largest_rectangle_area_l140_140775


namespace highest_score_is_151_l140_140766

-- Definitions for the problem conditions
def total_runs : ℕ := 2704
def total_runs_excluding_HL : ℕ := 2552

variables (H L : ℕ) 

-- Problem conditions as hypotheses
axiom h1 : H - L = 150
axiom h2 : H + L = 152
axiom h3 : 2704 = 2552 + H + L

-- Proof statement
theorem highest_score_is_151 (H L : ℕ) (h1 : H - L = 150) (h2 : H + L = 152) (h3 : 2704 = 2552 + H + L) : H = 151 :=
by sorry

end highest_score_is_151_l140_140766


namespace garden_roller_diameter_l140_140658

theorem garden_roller_diameter 
  (length : ℝ) 
  (total_area : ℝ) 
  (num_revolutions : ℕ) 
  (pi : ℝ) 
  (A : length = 2)
  (B : total_area = 37.714285714285715)
  (C : num_revolutions = 5)
  (D : pi = 22 / 7) : 
  ∃ d : ℝ, d = 1.2 :=
by
  sorry

end garden_roller_diameter_l140_140658


namespace lines_skew_l140_140691

def line1 (b : ℝ) (t : ℝ) : ℝ × ℝ × ℝ := 
  (2 + 3 * t, 3 + 2 * t, b + 5 * t)

def line2 (u : ℝ) : ℝ × ℝ × ℝ := 
  (5 + 6 * u, 4 + 3 * u, 1 + 2 * u)

theorem lines_skew (b : ℝ) : 
  ¬ ∃ t u : ℝ, line1 b t = line2 u ↔ b ≠ 4 := 
sorry

end lines_skew_l140_140691


namespace average_side_length_of_squares_l140_140765

theorem average_side_length_of_squares (a b c : ℕ) (h₁ : a = 36) (h₂ : b = 64) (h₃ : c = 144) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 26 / 3 := by
  sorry

end average_side_length_of_squares_l140_140765


namespace sequence_problem_l140_140719

/-- Given sequence a_n with specific values for a_2 and a_4 and the assumption that a_(n+1)
    is a geometric sequence, prove that a_6 equals 63. -/
theorem sequence_problem 
  {a : ℕ → ℝ} 
  (h1 : a 2 = 3) 
  (h2 : a 4 = 15) 
  (h3 : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n ∧ q^2 = 4) : 
  a 6 = 63 := by
  sorry

end sequence_problem_l140_140719


namespace range_a_I_range_a_II_l140_140194

variable (a: ℝ)

-- Define the proposition p and q
def p := (Real.sqrt (a^2 + 13) > Real.sqrt 17)
def q := ∀ x, (0 < x ∧ x < 3) → (x^2 - 2 * a * x - 2 = 0)

-- Prove question (I): If proposition p is true, find the range of the real number $a$
theorem range_a_I (h_p : p a) : a < -2 ∨ a > 2 :=
by sorry

-- Prove question (II): If both the proposition "¬q" and "p ∧ q" are false, find the range of the real number $a$
theorem range_a_II (h_neg_q : ¬ q a) (h_p_and_q : ¬ (p a ∧ q a)) : -2 ≤ a ∧ a ≤ 0 :=
by sorry

end range_a_I_range_a_II_l140_140194


namespace cube_surface_area_increase_l140_140925

theorem cube_surface_area_increase (s : ℝ) : 
    let initial_area := 6 * s^2
    let new_edge := 1.3 * s
    let new_area := 6 * (new_edge)^2
    let incr_area := new_area - initial_area
    let percentage_increase := (incr_area / initial_area) * 100
    percentage_increase = 69 :=
by
  let initial_area := 6 * s^2
  let new_edge := 1.3 * s
  let new_area := 6 * (new_edge)^2
  let incr_area := new_area - initial_area
  let percentage_increase := (incr_area / initial_area) * 100
  sorry

end cube_surface_area_increase_l140_140925


namespace cone_lateral_surface_area_l140_140860

noncomputable def lateralSurfaceArea (r l : ℝ) : ℝ := Real.pi * r * l

theorem cone_lateral_surface_area : 
  ∀ (r l : ℝ), r = 2 → l = 5 → lateralSurfaceArea r l = 10 * Real.pi :=
by 
  intros r l hr hl
  rw [hr, hl]
  unfold lateralSurfaceArea
  norm_num
  sorry

end cone_lateral_surface_area_l140_140860


namespace solve_for_x_l140_140048

theorem solve_for_x (x : ℝ) (h : 3 * x - 5 = -2 * x + 10) : x = 3 := 
sorry

end solve_for_x_l140_140048


namespace combined_population_after_two_years_l140_140423

def population_after_years (initial_population : ℕ) (yearly_changes : List (ℕ → ℕ)) : ℕ :=
  yearly_changes.foldl (fun pop change => change pop) initial_population

def townA_change_year1 (pop : ℕ) : ℕ :=
  pop + (pop * 8 / 100) + 200 - 100

def townA_change_year2 (pop : ℕ) : ℕ :=
  pop + (pop * 10 / 100) + 200 - 100

def townB_change_year1 (pop : ℕ) : ℕ :=
  pop - (pop * 2 / 100) + 50 - 200

def townB_change_year2 (pop : ℕ) : ℕ :=
  pop - (pop * 1 / 100) + 50 - 200

theorem combined_population_after_two_years :
  population_after_years 15000 [townA_change_year1, townA_change_year2] +
  population_after_years 10000 [townB_change_year1, townB_change_year2] = 27433 := 
  sorry

end combined_population_after_two_years_l140_140423


namespace product_of_equal_numbers_l140_140618

theorem product_of_equal_numbers (a b : ℕ) (mean : ℕ) (sum : ℕ)
  (h1 : mean = 20)
  (h2 : a = 22)
  (h3 : b = 34)
  (h4 : sum = 4 * mean)
  (h5 : sum - a - b = 2 * x)
  (h6 : sum = 80)
  (h7 : x = 12) 
  : x * x = 144 :=
by
  sorry

end product_of_equal_numbers_l140_140618


namespace katie_candy_l140_140484

theorem katie_candy (K : ℕ) (H1 : K + 6 - 9 = 7) : K = 10 :=
by
  sorry

end katie_candy_l140_140484


namespace baseball_tickets_l140_140505

theorem baseball_tickets (B : ℕ) 
  (h1 : 25 = 2 * B + 6) : B = 9 :=
sorry

end baseball_tickets_l140_140505


namespace find_coordinates_of_P_l140_140572

-- Definitions based on the conditions:
-- Point P has coordinates (a, 2a-1) and lies on the line y = x.

def lies_on_bisector (a : ℝ) : Prop :=
  (2 * a - 1) = a -- This is derived from the line y = x for the given point coordinates.

-- The final statement to prove:
theorem find_coordinates_of_P (a : ℝ) (P : ℝ × ℝ) (h1 : P = (a, 2 * a - 1)) (h2 : lies_on_bisector a) :
  P = (1, 1) :=
by
  -- Proof steps are omitted and replaced with sorry.
  sorry

end find_coordinates_of_P_l140_140572


namespace thomas_payment_weeks_l140_140634

theorem thomas_payment_weeks 
    (weekly_rate : ℕ) 
    (total_amount_paid : ℕ) 
    (h1 : weekly_rate = 4550) 
    (h2 : total_amount_paid = 19500) :
    (19500 / 4550 : ℕ) = 4 :=
by {
  sorry
}

end thomas_payment_weeks_l140_140634


namespace max_value_l140_140745

theorem max_value (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 2) : 
  2 * x * y + 2 * y * z * Real.sqrt 3 ≤ 4 :=
sorry

end max_value_l140_140745


namespace loss_equals_cost_price_of_some_balls_l140_140495

-- Conditions
def cost_price_per_ball := 60
def selling_price_for_17_balls := 720
def number_of_balls := 17

-- Calculations
def total_cost_price := number_of_balls * cost_price_per_ball
def loss := total_cost_price - selling_price_for_17_balls

-- Proof statement
theorem loss_equals_cost_price_of_some_balls : (loss / cost_price_per_ball) = 5 :=
by
  -- Proof would go here
  sorry

end loss_equals_cost_price_of_some_balls_l140_140495


namespace simplify_polynomial_l140_140756

variable (x : ℝ)

theorem simplify_polynomial : (2 * x^2 + 5 * x - 3) - (2 * x^2 + 9 * x - 6) = -4 * x + 3 :=
by
  sorry

end simplify_polynomial_l140_140756


namespace green_balls_l140_140105

variable (B G : ℕ)

theorem green_balls (h1 : B = 20) (h2 : B / G = 5 / 3) : G = 12 :=
by
  -- Proof goes here
  sorry

end green_balls_l140_140105


namespace max_rectangle_area_l140_140774

theorem max_rectangle_area (a b : ℝ) (h : 2 * a + 2 * b = 60) :
  a * b ≤ 225 :=
by
  sorry

end max_rectangle_area_l140_140774


namespace find_x_plus_3y_l140_140348

variables {α : Type*} {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (x y : ℝ)
variables (OA OB OC OD OE : V)

-- Defining the conditions
def condition1 := OA = (1/2) • OB + x • OC + y • OD
def condition2 := OB = 2 • x • OC + (1/3) • OD + y • OE

-- Writing the theorem statement
theorem find_x_plus_3y (h1 : condition1 x y OA OB OC OD) (h2 : condition2 x y OB OC OD OE) : 
  x + 3 * y = 7 / 6 := 
sorry

end find_x_plus_3y_l140_140348


namespace angles_of_triangle_l140_140479

theorem angles_of_triangle 
  (α β γ : ℝ)
  (triangle_ABC : α + β + γ = 180)
  (median_bisector_height : (γ / 4) * 4 = 90) :
  α = 22.5 ∧ β = 67.5 ∧ γ = 90 :=
by
  sorry

end angles_of_triangle_l140_140479


namespace find_abc_l140_140340

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

axiom condition1 : a * b = 45 * (3 : ℝ)^(1/3)
axiom condition2 : a * c = 75 * (3 : ℝ)^(1/3)
axiom condition3 : b * c = 30 * (3 : ℝ)^(1/3)

theorem find_abc : a * b * c = 75 * (2 : ℝ)^(1/2) := sorry

end find_abc_l140_140340


namespace given_sequence_find_a_and_b_l140_140456

-- Define the general pattern of the sequence
def sequence_pattern (n a b : ℕ) : Prop :=
  n + (b / a : ℚ) = (n^2 : ℚ) * (b / a : ℚ)

-- State the specific case for n = 9
def sequence_case_for_9 (a b : ℕ) : Prop :=
  sequence_pattern 9 a b ∧ a + b = 89

-- Now, structure this as a theorem to be proven in Lean
theorem given_sequence_find_a_and_b :
  ∃ (a b : ℕ), sequence_case_for_9 a b :=
sorry

end given_sequence_find_a_and_b_l140_140456


namespace largest_of_three_consecutive_integers_sum_90_is_31_l140_140625

theorem largest_of_three_consecutive_integers_sum_90_is_31 :
  ∃ (a b c : ℤ), (a + b + c = 90) ∧ (b = a + 1) ∧ (c = b + 1) ∧ (c = 31) :=
by
  sorry

end largest_of_three_consecutive_integers_sum_90_is_31_l140_140625


namespace point_N_in_second_quadrant_l140_140606

theorem point_N_in_second_quadrant (a b : ℝ) (h1 : 1 + a < 0) (h2 : 2 * b - 1 < 0) :
    (a - 1 < 0) ∧ (1 - 2 * b > 0) :=
by
  -- Insert proof here
  sorry

end point_N_in_second_quadrant_l140_140606


namespace inequality_proof_l140_140568

variable {α β γ : ℝ}

theorem inequality_proof (h1 : β * γ ≠ 0) (h2 : (1 - γ^2) / (β * γ) ≥ 0) :
  10 * (α^2 + β^2 + γ^2 - β * γ^2) ≥ 2 * α * β + 5 * α * γ :=
sorry

end inequality_proof_l140_140568


namespace intersection_of_symmetric_set_and_naturals_l140_140586

namespace SymmetricSet

open Set

-- Definitions
def symmetric_set (A : Set ℤ) : Prop := ∀ x : ℤ, x ∈ A → -x ∈ A

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k ∨ x = 0 ∨ x = k^2 + k}

def B : Set ℕ := {n | n ∈ ℕ}

-- Correct answer
def answer : Set ℤ := {0, 6}

-- Proof statement
theorem intersection_of_symmetric_set_and_naturals (hA : symmetric_set A) : 
  (A ∩ B : Set ℤ) = answer := 
by
  sorry
  
end SymmetricSet

end intersection_of_symmetric_set_and_naturals_l140_140586


namespace max_a_correct_answers_l140_140734

theorem max_a_correct_answers : 
  ∃ (a b c x y z w : ℕ), 
  a + b + c + x + y + z + w = 39 ∧
  a = b + c ∧
  (a + x + y + w) = a + 5 + (x + y + w) ∧
  b + z = 2 * (c + z) ∧
  23 ≤ a :=
sorry

end max_a_correct_answers_l140_140734


namespace remainder_27_pow_482_div_13_l140_140790

theorem remainder_27_pow_482_div_13 :
  27^482 % 13 = 1 :=
sorry

end remainder_27_pow_482_div_13_l140_140790


namespace multiples_of_7_between_50_and_200_l140_140213

theorem multiples_of_7_between_50_and_200 : 
  ∃ n, n = 21 ∧ ∀ k, (k ≥ 50 ∧ k ≤ 200) ↔ ∃ m, k = 7 * m := sorry

end multiples_of_7_between_50_and_200_l140_140213


namespace correct_operation_among_given_ones_l140_140534

theorem correct_operation_among_given_ones
  (a : ℝ) :
  (a^2)^3 = a^6 :=
by {
  sorry
}

-- Auxiliary lemmas if needed (based on conditions):
lemma mul_powers_add_exponents (a : ℝ) (m n : ℕ) : a^m * a^n = a^(m + n) := by sorry

lemma power_of_a_power (a : ℝ) (m n : ℕ) : (a^m)^n = a^(m * n) := by sorry

lemma div_powers_subtract_exponents (a : ℝ) (m n : ℕ) : a^m / a^n = a^(m - n) := by sorry

lemma square_of_product (x y : ℝ) : (x * y)^2 = x^2 * y^2 := by sorry

end correct_operation_among_given_ones_l140_140534


namespace probability_all_three_blue_l140_140797

theorem probability_all_three_blue :
  let total_jellybeans := 20
  let initial_blue := 10
  let initial_red := 10
  let prob_first_blue := initial_blue / total_jellybeans
  let prob_second_blue := (initial_blue - 1) / (total_jellybeans - 1)
  let prob_third_blue := (initial_blue - 2) / (total_jellybeans - 2)
  prob_first_blue * prob_second_blue * prob_third_blue = 2 / 19 := 
by
  sorry

end probability_all_three_blue_l140_140797


namespace range_of_m_for_monotonic_f_l140_140206

noncomputable def f (m x : ℝ) : ℝ := Real.exp x - m * x

theorem range_of_m_for_monotonic_f :
  (∀ x ≥ 0, (Real.exp x - m : ℝ) ≥ 0) → m ≤ 1 := by
  sorry

end range_of_m_for_monotonic_f_l140_140206


namespace h_of_neg_one_l140_140243

def f (x : ℝ) : ℝ := 3 * x + 7
def g (x : ℝ) : ℝ := (f x) ^ 2 - 3
def h (x : ℝ) : ℝ := f (g x)

theorem h_of_neg_one :
  h (-1) = 298 :=
by
  sorry

end h_of_neg_one_l140_140243


namespace Laticia_knitted_socks_l140_140886

theorem Laticia_knitted_socks (x : ℕ) (cond1 : x ≥ 0)
  (cond2 : ∃ y, y = x + 4)
  (cond3 : ∃ z, z = (x + (x + 4)) / 2)
  (cond4 : ∃ w, w = z - 3)
  (cond5 : x + (x + 4) + z + w = 57) : x = 13 := by
  sorry

end Laticia_knitted_socks_l140_140886


namespace find_a_l140_140699

theorem find_a (a : ℝ) (x : ℝ) (h₀ : a > 0) (h₁ : x > 0)
  (h₂ : a * Real.sqrt x = Real.log (Real.sqrt x))
  (h₃ : (a / (2 * Real.sqrt x)) = (1 / (2 * x))) : a = Real.exp (-1) :=
by
  sorry

end find_a_l140_140699


namespace driver_net_rate_of_pay_is_25_l140_140656

noncomputable def net_rate_of_pay_per_hour (hours_traveled : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) (pay_per_mile : ℝ) (fuel_cost_per_gallon : ℝ) : ℝ :=
  let total_distance := speed * hours_traveled
  let total_fuel_used := total_distance / fuel_efficiency
  let total_earnings := pay_per_mile * total_distance
  let total_fuel_cost := fuel_cost_per_gallon * total_fuel_used
  let net_earnings := total_earnings - total_fuel_cost
  net_earnings / hours_traveled

theorem driver_net_rate_of_pay_is_25 :
  net_rate_of_pay_per_hour 3 50 25 0.6 2.5 = 25 := sorry

end driver_net_rate_of_pay_is_25_l140_140656


namespace total_ways_to_split_is_12_l140_140514

-- Define the set of people
def people : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the knows relationship based on given conditions
def knows (a b : ℕ) : Prop :=
  (a = b + 1) ∨ (a = b - 1) ∨ (a = b + 2) ∨ (a = b - 2) ∨ (a = b + 6) ∨ (a = b - 6)

-- Define a valid pairing
def valid_pairing (pairs : Finset (ℕ × ℕ)) : Prop :=
  pairs.card = 4 ∧
  (∀ {a b c}, (a, b) ∈ pairs → (c, a) ∈ pairs → c = b ∨ c = a) ∧
  (∀ (a b : ℕ), (a, b) ∈ pairs → knows a b)

-- Define the total valid pair configurations
def total_valid_pairs (people : Finset ℕ) : ℕ :=
  (Finset.powerset people).filter (λ pairs, valid_pairing pairs).card

theorem total_ways_to_split_is_12 :
  total_valid_pairs people = 12 := sorry

end total_ways_to_split_is_12_l140_140514


namespace fourth_game_water_correct_fourth_game_sports_drink_l140_140935

noncomputable def total_bottled_water_cases : ℕ := 10
noncomputable def total_sports_drink_cases : ℕ := 5
noncomputable def bottles_per_case_water : ℕ := 20
noncomputable def bottles_per_case_sports_drink : ℕ := 15
noncomputable def initial_bottled_water : ℕ := total_bottled_water_cases * bottles_per_case_water
noncomputable def initial_sports_drinks : ℕ := total_sports_drink_cases * bottles_per_case_sports_drink

noncomputable def first_game_water : ℕ := 70
noncomputable def first_game_sports_drink : ℕ := 30
noncomputable def second_game_water : ℕ := 40
noncomputable def second_game_sports_drink : ℕ := 20
noncomputable def third_game_water : ℕ := 50
noncomputable def third_game_sports_drink : ℕ := 25

noncomputable def total_consumed_water : ℕ := first_game_water + second_game_water + third_game_water
noncomputable def total_consumed_sports_drink : ℕ := first_game_sports_drink + second_game_sports_drink + third_game_sports_drink

noncomputable def remaining_water_before_fourth_game : ℕ := initial_bottled_water - total_consumed_water
noncomputable def remaining_sports_drink_before_fourth_game : ℕ := initial_sports_drinks - total_consumed_sports_drink

noncomputable def remaining_water_after_fourth_game : ℕ := 20
noncomputable def remaining_sports_drink_after_fourth_game : ℕ := 10

noncomputable def fourth_game_water_consumed : ℕ := remaining_water_before_fourth_game - remaining_water_after_fourth_game

theorem fourth_game_water_correct : fourth_game_water_consumed = 20 :=
by
  unfold fourth_game_water_consumed remaining_water_before_fourth_game
  sorry

theorem fourth_game_sports_drink : false :=
by
  sorry

end fourth_game_water_correct_fourth_game_sports_drink_l140_140935


namespace Leroy_min_bail_rate_l140_140131

noncomputable def min_bailing_rate
    (distance_to_shore : ℝ)
    (leak_rate : ℝ)
    (max_tolerable_water : ℝ)
    (rowing_speed : ℝ)
    : ℝ :=
  let time_to_shore := distance_to_shore / rowing_speed * 60
  let total_water_intake := leak_rate * time_to_shore
  let required_bailing := total_water_intake - max_tolerable_water
  required_bailing / time_to_shore

theorem Leroy_min_bail_rate
    (distance_to_shore : ℝ := 2)
    (leak_rate : ℝ := 15)
    (max_tolerable_water : ℝ := 60)
    (rowing_speed : ℝ := 4)
    : min_bailing_rate 2 15 60 4 = 13 := 
by
  simp [min_bailing_rate]
  sorry

end Leroy_min_bail_rate_l140_140131


namespace ants_of_species_X_on_day_6_l140_140732

/-- Given the initial populations of Species X and Species Y and their growth rates,
    prove the number of Species X ants on Day 6. -/
theorem ants_of_species_X_on_day_6 
  (x y : ℕ)  -- Number of Species X and Y ants on Day 0
  (h1 : x + y = 40)  -- Total number of ants on Day 0
  (h2 : 64 * x + 4096 * y = 21050)  -- Total number of ants on Day 6
  :
  64 * x = 2304 := 
sorry

end ants_of_species_X_on_day_6_l140_140732


namespace graph_passes_through_fixed_point_l140_140621

theorem graph_passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ y : ℝ, y = a^0 + 3 ∧ (0, y) = (0, 4)) :=
by
  use 4
  have h : a^0 = 1 := by simp
  rw [h]
  simp
  sorry

end graph_passes_through_fixed_point_l140_140621


namespace number_of_black_balls_l140_140914

variable (T : ℝ)
variable (red_balls : ℝ := 21)
variable (prop_red : ℝ := 0.42)
variable (prop_white : ℝ := 0.28)
variable (white_balls : ℝ := 0.28 * T)

noncomputable def total_balls : ℝ := red_balls / prop_red

theorem number_of_black_balls :
  T = total_balls → 
  ∃ black_balls : ℝ, black_balls = total_balls - red_balls - white_balls ∧ black_balls = 15 := 
by
  intro hT
  let black_balls := total_balls - red_balls - white_balls
  use black_balls
  simp [total_balls]
  sorry

end number_of_black_balls_l140_140914


namespace number_division_l140_140529

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l140_140529


namespace slope_at_two_l140_140573

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2
noncomputable def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

theorem slope_at_two (a b : ℝ) (h1 : f' 1 a b = 0) (h2 : f 1 a b = 10) :
  f' 2 4 (-11) = 17 :=
sorry

end slope_at_two_l140_140573


namespace work_problem_l140_140812

theorem work_problem (hA : ∀ n : ℝ, n = 15)
  (h_work_together : ∀ n : ℝ, 3 * (1/15 + 1/n) = 0.35) :  
  1/20 = 1/20 :=
by
  sorry

end work_problem_l140_140812


namespace multiplication_result_l140_140169

theorem multiplication_result : 
  (500 * 2468 * 0.2468 * 100) = 30485120 :=
by
  sorry

end multiplication_result_l140_140169


namespace sum_of_roots_is_zero_l140_140068

variables {R : Type*} [Field R] {a b c p q : R}

theorem sum_of_roots_is_zero (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : a ≠ c)
  (h₄ : a^3 + p * a + q = 0) (h₅ : b^3 + p * b + q = 0) (h₆ : c^3 + p * c + q = 0) :
  a + b + c = 0 :=
by
  sorry

end sum_of_roots_is_zero_l140_140068


namespace fractional_exponent_equality_l140_140138

theorem fractional_exponent_equality :
  (3 / 4 : ℚ) ^ 2017 * (- ((1:ℚ) + 1 / 3)) ^ 2018 = 4 / 3 :=
by
  sorry

end fractional_exponent_equality_l140_140138


namespace cos_double_angle_example_l140_140058

def cos_double_angle_identity (x : ℝ) : Prop :=
  cos (2 * x) = 1 - 2 * (sin x) ^ 2

theorem cos_double_angle_example : cos_double_angle_identity (x : ℝ) 
  (h : sin x = - 2 / 3) : cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_example_l140_140058


namespace parallel_lines_l140_140222

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, (ax + 2 * y + a = 0 ∧ 3 * a * x + (a - 1) * y + 7 = 0) →
    - (a / 2) = - (3 * a / (a - 1))) ↔ (a = 0 ∨ a = 7) :=
by
  sorry

end parallel_lines_l140_140222


namespace like_terms_solutions_l140_140047

theorem like_terms_solutions (x y : ℤ) (h1 : 5 = 4 * x + 1) (h2 : 3 * y = 6) :
  x = 1 ∧ y = 2 := 
by 
  -- proof goes here
  sorry

end like_terms_solutions_l140_140047


namespace cycle_selling_price_l140_140543

theorem cycle_selling_price 
  (CP : ℝ) (gain_percent : ℝ) (SP : ℝ) 
  (h1 : CP = 840) 
  (h2 : gain_percent = 45.23809523809524 / 100)
  (h3 : SP = CP * (1 + gain_percent)) :
  SP = 1220 :=
sorry

end cycle_selling_price_l140_140543


namespace quadratic_has_real_roots_find_k_values_l140_140980

-- Define the quadratic equation
def quadratic_eq (k x : ℝ) : Prop := (x - 1)^2 + k * (x - 1) = 0

-- Prove that the equation quadratic_eq has real roots for any k ∈ ℝ
theorem quadratic_has_real_roots (k : ℝ) : ∃ x : ℝ, quadratic_eq k x :=
by {
  sorry
}

-- Prove the values of k based on given conditions with roots condition
theorem find_k_values (k : ℝ) (x1 x2 : ℝ) (h : quadratic_eq k x1 ∧ quadratic_eq k x2)
  (hx : x1^2 + x2^2 = 7 - x1 * x2) : k = 4 ∨ k = -1 :=
by {
  sorry
}

end quadratic_has_real_roots_find_k_values_l140_140980


namespace integral_evaluation_l140_140834

open Set
open Real

noncomputable def integral_cos_div_sin_cos := 
  ∫ x in 0..(π/2), (cos x) / (1 + sin x + cos x) = (π / 4) - (1 / 2) * log 2

theorem integral_evaluation : integral_cos_div_sin_cos := by
  sorry

end integral_evaluation_l140_140834


namespace total_duration_in_seconds_l140_140942

theorem total_duration_in_seconds :
  let hours_in_seconds := 2 * 3600
  let minutes_in_seconds := 45 * 60
  let extra_seconds := 30
  hours_in_seconds + minutes_in_seconds + extra_seconds = 9930 := by
  sorry

end total_duration_in_seconds_l140_140942


namespace cos_neg245_l140_140703

-- Define the given condition and declare the theorem to prove the required equality
variable (a : ℝ)
def cos_25_eq_a : Prop := (Real.cos 25 * Real.pi / 180 = a)

theorem cos_neg245 :
  cos_25_eq_a a → Real.cos (-245 * Real.pi / 180) = -Real.sqrt (1 - a^2) :=
by
  intro h
  sorry

end cos_neg245_l140_140703


namespace question_solution_l140_140551

variable (a b : ℝ)

theorem question_solution : 2 * a - 3 * (a - b) = -a + 3 * b := by
  sorry

end question_solution_l140_140551


namespace solution_set_of_inequality_l140_140779

theorem solution_set_of_inequality : {x : ℝ | -2 < x ∧ x < 1} = {x : ℝ | -x^2 - x + 2 > 0} :=
by
  sorry

end solution_set_of_inequality_l140_140779


namespace largest_integer_divisor_of_p_squared_minus_3q_squared_l140_140249

theorem largest_integer_divisor_of_p_squared_minus_3q_squared (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) (h : q < p) :
  ∃ d : ℤ, (∀ p q : ℤ, p % 2 = 1 → q % 2 = 1 → q < p → d ∣ (p^2 - 3*q^2)) ∧ 
           (∀ k : ℤ, (∀ p q : ℤ, p % 2 = 1 → q % 2 = 1 → q < p → k ∣ (p^2 - 3*q^2)) → k ≤ d) ∧ d = 2 :=
sorry

end largest_integer_divisor_of_p_squared_minus_3q_squared_l140_140249


namespace upper_limit_of_people_l140_140019

theorem upper_limit_of_people (T : ℕ) (h1 : (3/7) * T = 24) (h2 : T > 50) : T ≤ 56 :=
by
  -- The steps to solve this proof would go here.
  sorry

end upper_limit_of_people_l140_140019


namespace grant_received_money_l140_140412

theorem grant_received_money :
  let total_teeth := 20
  let lost_teeth := 2
  let first_tooth_amount := 20
  let other_tooth_amount_per_tooth := 2
  let remaining_teeth := total_teeth - lost_teeth - 1
  let total_amount_received := first_tooth_amount + remaining_teeth * other_tooth_amount_per_tooth
  total_amount_received = 54 :=
by  -- Start the proof mode
  sorry  -- This is where the actual proof would go

end grant_received_money_l140_140412


namespace Antoinette_weight_l140_140165

variable (A R : ℝ)

theorem Antoinette_weight :
  A = 63 → 
  (A = 2 * R - 7) → 
  (A + R = 98) →
  True :=
by
  intros hA hAR hsum
  have h := hAR.symm
  rw [h] at hsum
  have h1 : 2 * R - 7 + R = 98 := hsum
  have h2 : 3 * R - 7 = 98 := h1
  have h3 : 3 * R = 105 := by linarith
  have hR : R = 35 := by linarith
  have hA' : A = 2 * 35 - 7 := by rwa [←h]
  have hA' : A = 63 := by linarith
  sorry

end Antoinette_weight_l140_140165


namespace speed_of_faster_train_l140_140786

-- Definitions based on the conditions.
def length_train_1 : ℝ := 180
def length_train_2 : ℝ := 360
def time_to_cross : ℝ := 21.598272138228943
def speed_slow_train_kmph : ℝ := 30
def speed_fast_train_kmph : ℝ := 60

-- The theorem that needs to be proven.
theorem speed_of_faster_train :
  (length_train_1 + length_train_2) / time_to_cross * 3.6 = speed_slow_train_kmph + speed_fast_train_kmph :=
sorry

end speed_of_faster_train_l140_140786


namespace average_value_of_T_l140_140896

def average_T (boys girls : ℕ) (starts_with_boy : Bool) (ends_with_girl : Bool) : ℕ :=
  if boys = 9 ∧ girls = 15 ∧ starts_with_boy ∧ ends_with_girl then 12 else 0

theorem average_value_of_T :
  average_T 9 15 true true = 12 :=
sorry

end average_value_of_T_l140_140896


namespace number_of_trees_planted_l140_140269

theorem number_of_trees_planted (initial_trees final_trees trees_planted : ℕ) 
  (h_initial : initial_trees = 22)
  (h_final : final_trees = 77)
  (h_planted : trees_planted = final_trees - initial_trees) : 
  trees_planted = 55 := by
  sorry

end number_of_trees_planted_l140_140269


namespace remainder_17_pow_45_div_5_l140_140123

theorem remainder_17_pow_45_div_5 : (17 ^ 45) % 5 = 2 :=
by
  -- proof goes here
  sorry

end remainder_17_pow_45_div_5_l140_140123


namespace division_by_fraction_l140_140831

theorem division_by_fraction :
  (5 / (8 / 15) : ℚ) = 75 / 8 :=
by
  sorry

end division_by_fraction_l140_140831


namespace wife_weekly_savings_correct_l140_140306

-- Define constants
def monthly_savings_husband := 225
def num_months := 4
def weeks_per_month := 4
def num_weeks := num_months * weeks_per_month
def stocks_per_share := 50
def num_shares := 25
def invested_amount := num_shares * stocks_per_share
def total_savings := 2 * invested_amount

-- Weekly savings amount to prove
def weekly_savings_wife := 100

-- Total savings calculation condition
theorem wife_weekly_savings_correct :
  (monthly_savings_husband * num_months + weekly_savings_wife * num_weeks) = total_savings :=
by
  sorry

end wife_weekly_savings_correct_l140_140306


namespace count_divisors_divisible_exactly_2007_l140_140996

-- Definitions and conditions
def prime_factors_2006 : List Nat := [2, 17, 59]

def prime_factors_2006_pow_2006 : List (Nat × Nat) := [(2, 2006), (17, 2006), (59, 2006)]

def number_of_divisors (n : Nat) : Nat :=
  prime_factors_2006_pow_2006.foldl (λ acc ⟨p, exp⟩ => acc * (exp + 1)) 1

theorem count_divisors_divisible_exactly_2007 : 
  (number_of_divisors (2^2006 * 17^2006 * 59^2006) = 3) :=
  sorry

end count_divisors_divisible_exactly_2007_l140_140996


namespace part1_part2_l140_140041

noncomputable def set_A : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def set_B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

theorem part1 (a : ℝ) : (set_A ∪ set_B a = set_A ∩ set_B a) → a = 1 :=
sorry

theorem part2 (a : ℝ) : (set_A ∪ set_B a = set_A) → (a ≤ -1 ∨ a = 1) :=
sorry

end part1_part2_l140_140041


namespace marble_weights_total_l140_140605

theorem marble_weights_total:
  0.33 + 0.33 + 0.08 + 0.25 + 0.02 + 0.12 + 0.15 = 1.28 :=
by {
  sorry
}

end marble_weights_total_l140_140605


namespace min_distance_from_origin_to_line_l140_140349

open Real

theorem min_distance_from_origin_to_line :
    ∀ x y : ℝ, (3 * x + 4 * y - 4 = 0) -> dist (0, 0) (x, y) = 4 / 5 :=
by
  sorry

end min_distance_from_origin_to_line_l140_140349


namespace no_adjacent_alder_trees_probability_l140_140303

noncomputable def trees_probability : ℚ := 2 / 4290

theorem no_adjacent_alder_trees_probability :
  let cedar_trees := 4
  let pine_trees := 3
  let alder_trees := 6
  let total_trees := cedar_trees + pine_trees + alder_trees
  ∀ arrangements : Finset (Finset (Fin total_trees)),
  (∀ a1 a2 ∈ arrangements, a1 ≠ a2 → (a1 ∩ a2 = ∅)) →
  (∀ t ∈ arrangements, t.card = alder_trees) →
  ( ∃ valid_arrangement : Finset (Fin total_trees), 
    valid_arrangement.card = combinations total_trees alder_trees.val ∧ -- Total valid arrangements
    set.card {arr ∈ arrangements | no two alder trees are adjacent (in arr)} = 28) →
  probability = trees_probability :=
sorry

end no_adjacent_alder_trees_probability_l140_140303


namespace squares_area_relation_l140_140674

/-- 
Given:
1. $\alpha$ such that $\angle 1 = \angle 2 = \angle 3 = \alpha$
2. The areas of the squares are given by:
   - $S_A = \cos^4 \alpha$
   - $S_D = \sin^4 \alpha$
   - $S_B = \cos^2 \alpha \sin^2 \alpha$
   - $S_C = \cos^2 \alpha \sin^2 \alpha$

Prove that:
$S_A \cdot S_D = S_B \cdot S_C$
--/

theorem squares_area_relation (α : ℝ) :
  (Real.cos α)^4 * (Real.sin α)^4 = (Real.cos α)^2 * (Real.sin α)^2 * (Real.cos α)^2 * (Real.sin α)^2 :=
by sorry

end squares_area_relation_l140_140674


namespace train_speed_l140_140309

theorem train_speed (length : ℕ) (time : ℕ) (v : ℕ)
  (h1 : length = 750)
  (h2 : time = 1)
  (h3 : v = (length + length) / time)
  (h4 : v = 1500) :
  (v * 60 / 1000 = 90) :=
by
  sorry

end train_speed_l140_140309


namespace min_value_sin4_plus_2_cos4_l140_140322

theorem min_value_sin4_plus_2_cos4 : ∀ x : ℝ, sin x ^ 4 + 2 * cos x ^ 4 ≥ 2 / 3 :=
by
  intro x
  sorry

end min_value_sin4_plus_2_cos4_l140_140322


namespace find_a_l140_140487

def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem find_a (a : ℝ) 
  (h : deriv (f a) (-1) = 4) : 
  a = 10 / 3 :=
sorry

end find_a_l140_140487


namespace bills_equal_at_80_minutes_l140_140391

variable (m : ℝ)

def C_U : ℝ := 8 + 0.25 * m
def C_A : ℝ := 12 + 0.20 * m

theorem bills_equal_at_80_minutes (h : C_U m = C_A m) : m = 80 :=
by {
  sorry
}

end bills_equal_at_80_minutes_l140_140391


namespace find_other_parallel_side_length_l140_140436

variable (a b h A : ℝ)

-- Conditions
def length_one_parallel_side := a = 18
def distance_between_sides := h = 12
def area_trapezium := A = 228
def trapezium_area_formula := A = 1 / 2 * (a + b) * h

-- Target statement to prove
theorem find_other_parallel_side_length
    (h1 : length_one_parallel_side a)
    (h2 : distance_between_sides h)
    (h3 : area_trapezium A)
    (h4 : trapezium_area_formula a b h A) :
    b = 20 :=
sorry

end find_other_parallel_side_length_l140_140436


namespace coin_flip_probability_l140_140504

/--
Suppose we flip five coins simultaneously: a penny, a nickel, a dime, a quarter, and a half-dollar.
What is the probability that the penny and dime both come up heads, and the half-dollar comes up tails?
-/

theorem coin_flip_probability :
  let outcomes := 2^5
  let success := 1 * 1 * 1 * 2 * 2
  success / outcomes = (1 : ℚ) / 8 :=
by
  /- Proof goes here -/
  sorry

end coin_flip_probability_l140_140504


namespace balloon_height_l140_140097

-- Define the problem given in the question
def angle_elevation_A: ℝ := 45
def angle_elevation_B: ℝ := 22.5
def distance_AB: ℝ := 1600
def angle_directions: ℝ := 135

-- Prove that the height of the balloon above the ground is approximately 500 meters given the above conditions
theorem balloon_height : 
  ∀ (m : ℝ) (α β δ d : ℝ),
  α = angle_elevation_A → β = angle_elevation_B → δ = angle_directions → d = distance_AB →
  (∃ m : ℝ, abs(m - 500) < 0.001) :=
by
  sorry

end balloon_height_l140_140097


namespace inequality_solution_l140_140371

theorem inequality_solution (x : ℝ) (h : x ≠ -7) : 
  (x^2 - 49) / (x + 7) < 0 ↔ x ∈ Set.Ioo (-∞) (-7) ∪ Set.Ioo (-7) 7 := by 
  sorry

end inequality_solution_l140_140371


namespace tory_needs_to_sell_more_packs_l140_140021

theorem tory_needs_to_sell_more_packs 
  (total_goal : ℤ) (packs_grandmother : ℤ) (packs_uncle : ℤ) (packs_neighbor : ℤ) 
  (total_goal_eq : total_goal = 50)
  (packs_grandmother_eq : packs_grandmother = 12)
  (packs_uncle_eq : packs_uncle = 7)
  (packs_neighbor_eq : packs_neighbor = 5) :
  total_goal - (packs_grandmother + packs_uncle + packs_neighbor) = 26 :=
by
  rw [total_goal_eq, packs_grandmother_eq, packs_uncle_eq, packs_neighbor_eq]
  norm_num

end tory_needs_to_sell_more_packs_l140_140021


namespace previous_monthly_income_l140_140749

variable (I : ℝ)

-- Conditions from the problem
def condition1 (I : ℝ) : Prop := 0.40 * I = 0.25 * (I + 600)

theorem previous_monthly_income (h : condition1 I) : I = 1000 := by
  sorry

end previous_monthly_income_l140_140749


namespace geometric_sequence_term_302_l140_140351

def geometric_sequence (a r : ℤ) (n : ℕ) : ℤ := a * r ^ (n - 1)

theorem geometric_sequence_term_302 :
  let a := 8
  let r := -2
  geometric_sequence a r 302 = -2^304 := by
  sorry

end geometric_sequence_term_302_l140_140351


namespace A_can_finish_remaining_work_in_6_days_l140_140540

-- Condition: A can finish the work in 18 days
def A_work_rate := 1 / 18

-- Condition: B can finish the work in 15 days
def B_work_rate := 1 / 15

-- Given B worked for 10 days
def B_days_worked := 10

-- Calculation of the remaining work
def remaining_work := 1 - B_days_worked * B_work_rate

-- Calculation of the time for A to finish the remaining work
def A_remaining_days := remaining_work / A_work_rate

-- The theorem to prove
theorem A_can_finish_remaining_work_in_6_days : A_remaining_days = 6 := 
by 
  -- The proof is not required, so we use sorry to skip it.
  sorry

end A_can_finish_remaining_work_in_6_days_l140_140540


namespace second_and_fourth_rows_identical_l140_140735

def count_occurrences (lst : List ℕ) (a : ℕ) (i : ℕ) : ℕ :=
  (lst.take (i + 1)).count a

def fill_next_row (current_row : List ℕ) : List ℕ :=
  current_row.enum.map (λ ⟨i, a⟩ => count_occurrences current_row a i)

theorem second_and_fourth_rows_identical (first_row : List ℕ) :
  let second_row := fill_next_row first_row 
  let third_row := fill_next_row second_row 
  let fourth_row := fill_next_row third_row 
  second_row = fourth_row :=
by
  sorry

end second_and_fourth_rows_identical_l140_140735


namespace find_analytical_expression_and_a_l140_140718

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 3)

theorem find_analytical_expression_and_a :
  (A > 0) → (ω > 0) → (0 < φ ∧ φ < π / 2) →
  (∀ x, ∃ k : ℤ, f (x + k * π / 2) = f (x)) →
  (∃ A, ∀ x, A * sin (ω * x + φ) ≤ 2) →
  ((∀ x, f (x - π / 6) = -f (-x + π / 6)) ∨ f 0 = sqrt 3 ∨ (∃ x, 2 * x + φ = k * π + π / 2)) →
  (∀ x, f x = 2 * sin (2 * x + π / 3)) ∧
  (∀ (A : ℝ), (0 < A ∧ A < π) → (f A = sqrt 3) →
  (c = 3 ∧ S = 3 * sqrt 3) →
  (a ^ 2 = ((4 * sqrt 3) ^ 2 + 3 ^ 2 - 2 * (4 * sqrt 3) * 3 * cos (π / 6))) → a = sqrt 21) :=
  sorry

end find_analytical_expression_and_a_l140_140718


namespace find_number_l140_140967

theorem find_number (x : ℕ) (h : (18 / 100) * x = 90) : x = 500 :=
sorry

end find_number_l140_140967


namespace union_sets_l140_140720

def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem union_sets : A ∪ B = {x | -1 < x ∧ x < 2} := by
  sorry

end union_sets_l140_140720


namespace number_modulo_conditions_l140_140814

theorem number_modulo_conditions : 
  ∃ n : ℕ, 
  (n % 10 = 9) ∧ 
  (n % 9 = 8) ∧ 
  (n % 8 = 7) ∧ 
  (n % 7 = 6) ∧ 
  (n % 6 = 5) ∧ 
  (n % 5 = 4) ∧ 
  (n % 4 = 3) ∧ 
  (n % 3 = 2) ∧ 
  (n % 2 = 1) ∧ 
  (n = 2519) :=
by
  sorry

end number_modulo_conditions_l140_140814


namespace total_legs_l140_140256

-- Define the number of each type of animal
def num_horses : ℕ := 2
def num_dogs : ℕ := 5
def num_cats : ℕ := 7
def num_turtles : ℕ := 3
def num_goat : ℕ := 1

-- Define the number of legs per animal
def legs_per_animal : ℕ := 4

-- Define the total number of legs for each type of animal
def horse_legs : ℕ := num_horses * legs_per_animal
def dog_legs : ℕ := num_dogs * legs_per_animal
def cat_legs : ℕ := num_cats * legs_per_animal
def turtle_legs : ℕ := num_turtles * legs_per_animal
def goat_legs : ℕ := num_goat * legs_per_animal

-- Define the problem statement
theorem total_legs : horse_legs + dog_legs + cat_legs + turtle_legs + goat_legs = 72 := by
  -- Sum up all the leg counts
  sorry

end total_legs_l140_140256


namespace sum_of_proper_divisors_of_81_l140_140288

theorem sum_of_proper_divisors_of_81 :
  ∑ d in ({1, 3, 9, 27}: Finset ℕ), d = 40 :=
by
  sorry

end sum_of_proper_divisors_of_81_l140_140288


namespace line_segment_length_l140_140305

theorem line_segment_length (x : ℝ) (h : x > 0) :
  (Real.sqrt ((x - 2)^2 + (6 - 2)^2) = 5) → (x = 5) :=
by
  intro h1
  sorry

end line_segment_length_l140_140305


namespace cos_thirteen_pi_over_four_l140_140182

theorem cos_thirteen_pi_over_four : Real.cos (13 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_thirteen_pi_over_four_l140_140182


namespace running_time_around_pentagon_l140_140045

theorem running_time_around_pentagon :
  let l₁ := 40
  let l₂ := 50
  let l₃ := 60
  let l₄ := 45
  let l₅ := 55
  let v₁ := 9 * 1000 / 60
  let v₂ := 8 * 1000 / 60
  let v₃ := 7 * 1000 / 60
  let v₄ := 6 * 1000 / 60
  let v₅ := 5 * 1000 / 60
  let t₁ := l₁ / v₁
  let t₂ := l₂ / v₂
  let t₃ := l₃ / v₃
  let t₄ := l₄ / v₄
  let t₅ := l₅ / v₅
  t₁ + t₂ + t₃ + t₄ + t₅ = 2.266 := by
    sorry

end running_time_around_pentagon_l140_140045


namespace episodes_per_monday_l140_140252

theorem episodes_per_monday (M : ℕ) (h : 67 * (M + 2) = 201) : M = 1 :=
sorry

end episodes_per_monday_l140_140252


namespace find_number_l140_140968

theorem find_number (x : ℕ) (h : (18 / 100) * x = 90) : x = 500 :=
sorry

end find_number_l140_140968


namespace power_of_128_div_7_eq_16_l140_140957

theorem power_of_128_div_7_eq_16 : (128 : ℝ) ^ (4 / 7) = 16 := by
  sorry

end power_of_128_div_7_eq_16_l140_140957


namespace negation_of_existence_l140_140104

variable (x : ℝ)

theorem negation_of_existence :
  (¬ ∃ x₀ : ℝ, x₀^2 + 2 * x₀ - 3 > 0) ↔ (∀ x : ℝ, x^2 + 2*x - 3 ≤ 0) :=
by sorry

end negation_of_existence_l140_140104


namespace common_chord_length_l140_140917

theorem common_chord_length (r d : ℝ) (hr : r = 12) (hd : d = 16) : 
  ∃ l : ℝ, l = 8 * Real.sqrt 5 := 
by
  sorry

end common_chord_length_l140_140917


namespace prices_correct_minimum_cost_correct_l140_140807

-- Define the prices of the mustard brands
variables (x y m : ℝ)

def brandACost : ℝ := 9 * x + 6 * y
def brandBCost : ℝ := 5 * x + 8 * y

-- Conditions for prices
axiom cost_condition1 : brandACost x y = 390
axiom cost_condition2 : brandBCost x y = 310

-- Solution for prices
def priceA : ℝ := 30
def priceB : ℝ := 20

theorem prices_correct : x = priceA ∧ y = priceB :=
sorry

-- Conditions for minimizing cost
def totalCost (m : ℝ) : ℝ := 30 * m + 20 * (30 - m)
def totalPacks : ℝ := 30

-- Constraints
def constraint1 (m : ℝ) : Prop := m ≥ 5 + (30 - m)
def constraint2 (m : ℝ) : Prop := m ≤ 2 * (30 - m)

-- Minimum cost condition
def min_cost : ℝ := 780
def optimal_m : ℝ := 18

theorem minimum_cost_correct : constraint1 optimal_m ∧ constraint2 optimal_m ∧ totalCost optimal_m = min_cost :=
sorry

end prices_correct_minimum_cost_correct_l140_140807


namespace y_value_l140_140463

theorem y_value (x y : ℝ) (hx : 1 < x) (hy : 1 < y) (h_eq1 : (1 / x) + (1 / y) = 3 / 2) (h_eq2 : x * y = 9) : y = 6 :=
sorry

end y_value_l140_140463


namespace moon_land_value_l140_140908

theorem moon_land_value (surface_area_earth : ℕ) (surface_area_moon : ℕ) (total_value_earth : ℕ) (worth_factor : ℕ)
  (h_moon_surface_area : surface_area_moon = surface_area_earth / 5)
  (h_surface_area_earth : surface_area_earth = 200) 
  (h_worth_factor : worth_factor = 6) 
  (h_total_value_earth : total_value_earth = 80) : (total_value_earth / 5) * worth_factor = 96 := 
by 
  -- Simplify using the given conditions
  -- total_value_earth / 5 is the value of the moon's land if it had the same value per square acre as Earth's land
  -- multiplying by worth_factor to get the total value on the moon
  sorry

end moon_land_value_l140_140908


namespace cricket_average_l140_140936

theorem cricket_average (x : ℕ) (h : 20 * x + 158 = 21 * (x + 6)) : x = 32 :=
by
  sorry

end cricket_average_l140_140936


namespace decreasing_exponential_iff_l140_140379

theorem decreasing_exponential_iff {a : ℝ} :
  (∀ x y : ℝ, x < y → (a - 1)^y < (a - 1)^x) ↔ (1 < a ∧ a < 2) :=
by 
  sorry

end decreasing_exponential_iff_l140_140379


namespace smallest_k_for_abk_l140_140536

theorem smallest_k_for_abk : ∃ (k : ℝ), (∀ (a b : ℝ), a + b = k ∧ ab = k → k = 4) :=
sorry

end smallest_k_for_abk_l140_140536


namespace range_of_p_l140_140991

noncomputable def a_n (p : ℝ) (n : ℕ) : ℝ := -2 * n + p
noncomputable def b_n (n : ℕ) : ℝ := 2 ^ (n - 7)

noncomputable def c_n (p : ℝ) (n : ℕ) : ℝ :=
if a_n p n <= b_n n then a_n p n else b_n n

theorem range_of_p (p : ℝ) :
  (∀ n : ℕ, n ≠ 10 → c_n p 10 > c_n p n) ↔ 24 < p ∧ p < 30 :=
sorry

end range_of_p_l140_140991


namespace cos_double_angle_example_l140_140055

def cos_double_angle_identity (x : ℝ) : Prop :=
  cos (2 * x) = 1 - 2 * (sin x) ^ 2

theorem cos_double_angle_example : cos_double_angle_identity (x : ℝ) 
  (h : sin x = - 2 / 3) : cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_example_l140_140055


namespace total_distance_is_75_l140_140334

def distance1 : ℕ := 30
def distance2 : ℕ := 20
def distance3 : ℕ := 25

def total_distance : ℕ := distance1 + distance2 + distance3

theorem total_distance_is_75 : total_distance = 75 := by
  sorry

end total_distance_is_75_l140_140334


namespace students_shorter_than_yoongi_l140_140796

variable (total_students taller_than_yoongi : Nat)

theorem students_shorter_than_yoongi (h₁ : total_students = 20) (h₂ : taller_than_yoongi = 11) : 
    total_students - (taller_than_yoongi + 1) = 8 :=
by
  -- Here would be the proof
  sorry

end students_shorter_than_yoongi_l140_140796


namespace MEMOrable_rectangle_count_l140_140357

section MEMOrable_rectangles

variables (K L : ℕ) (hK : K > 0) (hL : L > 0) 

/-- In a 2K x 2L board, if the ant starts at (1,1) and ends at (2K, 2L),
    and some squares may remain unvisited forming a MEMOrable rectangle,
    then the number of such MEMOrable rectangles is (K(K+1)L(L+1))/2. -/
theorem MEMOrable_rectangle_count :
  ∃ (n : ℕ), n = K * (K + 1) * L * (L + 1) / 2 :=
by
  sorry

end MEMOrable_rectangles

end MEMOrable_rectangle_count_l140_140357


namespace min_value_l140_140981

theorem min_value (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_ab : a * b = 1) (h_a_2b : a = 2 * b) :
  a + 2 * b = 2 * Real.sqrt 2 := by
  sorry

end min_value_l140_140981


namespace cos_double_angle_example_l140_140054

def cos_double_angle_identity (x : ℝ) : Prop :=
  cos (2 * x) = 1 - 2 * (sin x) ^ 2

theorem cos_double_angle_example : cos_double_angle_identity (x : ℝ) 
  (h : sin x = - 2 / 3) : cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_example_l140_140054


namespace ratio_total_length_to_perimeter_l140_140664

noncomputable def length_initial : ℝ := 25
noncomputable def width_initial : ℝ := 15
noncomputable def extension : ℝ := 10
noncomputable def length_total : ℝ := length_initial + extension
noncomputable def perimeter_new : ℝ := 2 * (length_total + width_initial)
noncomputable def ratio : ℝ := length_total / perimeter_new

theorem ratio_total_length_to_perimeter : ratio = 35 / 100 := by
  sorry

end ratio_total_length_to_perimeter_l140_140664


namespace length_of_train_is_125_l140_140803

noncomputable def speed_kmph : ℝ := 90
noncomputable def time_sec : ℝ := 5
noncomputable def speed_mps : ℝ := speed_kmph * (1000 / 3600)
noncomputable def length_train : ℝ := speed_mps * time_sec

theorem length_of_train_is_125 :
  length_train = 125 := 
by
  sorry

end length_of_train_is_125_l140_140803


namespace solve_ineq_case1_solve_ineq_case2_l140_140263

theorem solve_ineq_case1 {a x : ℝ} (ha_pos : 0 < a) (ha_lt_one : a < 1) : 
  a^(x + 5) < a^(4 * x - 1) ↔ x < 2 :=
sorry

theorem solve_ineq_case2 {a x : ℝ} (ha_gt_one : a > 1) : 
  a^(x + 5) < a^(4 * x - 1) ↔ x > 2 :=
sorry

end solve_ineq_case1_solve_ineq_case2_l140_140263


namespace circle_equation_tangent_x_axis_l140_140903

theorem circle_equation_tangent_x_axis (x y : ℝ) (center : ℝ × ℝ) (r : ℝ) 
  (h_center : center = (-1, 2)) 
  (h_tangent : r = |2 - 0|) :
  (x + 1)^2 + (y - 2)^2 = 4 := 
sorry

end circle_equation_tangent_x_axis_l140_140903


namespace send_messages_ways_l140_140366

theorem send_messages_ways : (3^4 = 81) :=
by
  sorry

end send_messages_ways_l140_140366


namespace batsman_average_after_17th_inning_l140_140655

theorem batsman_average_after_17th_inning 
    (A : ℕ) 
    (hA : A = 15) 
    (runs_17th_inning : ℕ)
    (increase_in_average : ℕ) 
    (hscores : runs_17th_inning = 100)
    (hincrease : increase_in_average = 5) :
    (A + increase_in_average = 20) :=
by
  sorry

end batsman_average_after_17th_inning_l140_140655


namespace max_correct_questions_prime_score_l140_140876

-- Definitions and conditions
def total_questions := 20
def points_correct := 5
def points_no_answer := 0
def points_wrong := -2

-- Main statement to prove
theorem max_correct_questions_prime_score :
  ∃ (correct : ℕ) (no_answer wrong : ℕ), 
    correct + no_answer + wrong = total_questions ∧ 
    correct * points_correct + no_answer * points_no_answer + wrong * points_wrong = 83 ∧
    correct = 17 :=
sorry

end max_correct_questions_prime_score_l140_140876


namespace range_k_fx_greater_than_ln_l140_140207

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem range_k (k : ℝ) : 0 ≤ k ∧ k ≤ Real.exp 1 ↔ ∀ x : ℝ, f x ≥ k * x := 
by 
  sorry

theorem fx_greater_than_ln (t : ℝ) (x : ℝ) : t ≤ 2 ∧ 0 < x → f x > t + Real.log x :=
by
  sorry

end range_k_fx_greater_than_ln_l140_140207


namespace derivative_at_neg_one_l140_140069

noncomputable def f (a b c x : ℝ) : ℝ := a * x^4 + b * x^2 + c

theorem derivative_at_neg_one (a b c : ℝ) (h : (4 * a * 1^3 + 2 * b * 1) = 2) : 
  (4 * a * (-1)^3 + 2 * b * (-1)) = -2 := 
sorry

end derivative_at_neg_one_l140_140069


namespace probability_inside_circle_is_2_div_9_l140_140394

noncomputable def probability_point_in_circle : ℚ := 
  let total_points := 36
  let points_inside := 8
  points_inside / total_points

theorem probability_inside_circle_is_2_div_9 :
  probability_point_in_circle = 2 / 9 :=
by
  -- we acknowledge the mathematical computation here
  sorry

end probability_inside_circle_is_2_div_9_l140_140394


namespace fraction_to_terminating_decimal_l140_140973

theorem fraction_to_terminating_decimal : (49 : ℚ) / 160 = 0.30625 := 
sorry

end fraction_to_terminating_decimal_l140_140973


namespace Timmy_needs_to_go_faster_l140_140271

-- Define the trial speeds and the required speed
def s1 : ℕ := 36
def s2 : ℕ := 34
def s3 : ℕ := 38
def s_req : ℕ := 40

-- Statement of the theorem
theorem Timmy_needs_to_go_faster :
  s_req - (s1 + s2 + s3) / 3 = 4 :=
by
  sorry

end Timmy_needs_to_go_faster_l140_140271


namespace loan_difference_calculation_l140_140001

noncomputable def compounded_amount (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest_amount (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r * t)

theorem loan_difference_calculation 
(P : ℝ)
(r1 r2 : ℝ)
(n : ℕ)
(t1 t2: ℝ)
(three_year_payment_fraction : ℝ)
: 
let compounded_initial := compounded_amount P r1 n t1 in
let payment := three_year_payment_fraction * compounded_initial in
let remaining := compounded_initial - payment in
let compounded_final := compounded_amount remaining r1 n t2 in
let total_compounded := payment + compounded_final in
let simple_total := simple_interest_amount P r2 (t1 + t2) in
(abs (simple_total - total_compounded) ≈ 2834) := 
  sorry

end loan_difference_calculation_l140_140001


namespace total_legs_correct_l140_140258

-- Number of animals
def horses : ℕ := 2
def dogs : ℕ := 5
def cats : ℕ := 7
def turtles : ℕ := 3
def goat : ℕ := 1

-- Total number of animals
def total_animals : ℕ := horses + dogs + cats + turtles + goat

-- Total number of legs
def total_legs : ℕ := total_animals * 4

theorem total_legs_correct : total_legs = 72 := by
  -- proof skipped
  sorry

end total_legs_correct_l140_140258


namespace work_completion_days_l140_140300

theorem work_completion_days (A B C : ℕ) 
  (hA : A = 4) (hB : B = 8) (hC : C = 8) : 
  2 = 1 / (1 / A + 1 / B + 1 / C) :=
by
  -- skip the proof for now
  sorry

end work_completion_days_l140_140300


namespace drawings_per_neighbor_l140_140501

theorem drawings_per_neighbor (n_neighbors animals : ℕ) (h1 : n_neighbors = 6) (h2 : animals = 54) : animals / n_neighbors = 9 :=
by
  sorry

end drawings_per_neighbor_l140_140501


namespace length_of_first_train_is_270_l140_140541

/-- 
Given:
1. Speed of the first train = 120 kmph
2. Speed of the second train = 80 kmph
3. Time to cross each other = 9 seconds
4. Length of the second train = 230.04 meters
  
Prove that the length of the first train is 270 meters.
-/
theorem length_of_first_train_is_270
  (speed_first_train : ℝ := 120)
  (speed_second_train : ℝ := 80)
  (time_to_cross : ℝ := 9)
  (length_second_train : ℝ := 230.04)
  (conversion_factor : ℝ := 1000/3600) :
  (length_second_train + (speed_first_train + speed_second_train) * conversion_factor * time_to_cross - length_second_train) = 270 :=
by
  sorry

end length_of_first_train_is_270_l140_140541


namespace infinite_series_value_l140_140555

theorem infinite_series_value :
  ∑' n : ℕ, (n^3 + 2*n^2 + 5*n + 2) / (3^n * (n^3 + 3)) = 1 / 2 :=
sorry

end infinite_series_value_l140_140555


namespace find_p_at_8_l140_140489

noncomputable def h (x : ℝ) : ℝ := x^3 - x^2 + x - 1

noncomputable def p (x : ℝ) : ℝ :=
  let a := sorry ; -- root 1 of h
  let b := sorry ; -- root 2 of h
  let c := sorry ; -- root 3 of h
  let B := 2 / ((1 - a^3) * (1 - b^3) * (1 - c^3))
  B * (x - a^3) * (x - b^3) * (x - c^3)

theorem find_p_at_8 : p 8 = 1008 := sorry

end find_p_at_8_l140_140489


namespace percentage_of_temporary_workers_l140_140294

theorem percentage_of_temporary_workers (total_workers technicians non_technicians permanent_technicians permanent_non_technicians : ℕ) 
  (h1 : total_workers = 100)
  (h2 : technicians = total_workers / 2) 
  (h3 : non_technicians = total_workers / 2) 
  (h4 : permanent_technicians = technicians / 2) 
  (h5 : permanent_non_technicians = non_technicians / 2) :
  ((total_workers - (permanent_technicians + permanent_non_technicians)) / total_workers) * 100 = 50 :=
by
  sorry

end percentage_of_temporary_workers_l140_140294


namespace second_pipe_fill_time_l140_140918

theorem second_pipe_fill_time
  (rate1: ℝ) (rate_outlet: ℝ) (combined_time: ℝ)
  (h1: rate1 = 1 / 18)
  (h2: rate_outlet = 1 / 45)
  (h_combined: combined_time = 0.05):
  ∃ (x: ℝ), (1 / x) = 60 :=
by
  sorry

end second_pipe_fill_time_l140_140918


namespace kendra_bought_3_hats_l140_140508

-- Define the price of a wooden toy
def price_of_toy : ℕ := 20

-- Define the price of a hat
def price_of_hat : ℕ := 10

-- Define the amount Kendra went to the shop with
def initial_amount : ℕ := 100

-- Define the number of wooden toys Kendra bought
def number_of_toys : ℕ := 2

-- Define the amount of change Kendra received
def change_received : ℕ := 30

-- Prove that Kendra bought 3 hats
theorem kendra_bought_3_hats : 
  initial_amount - change_received - (number_of_toys * price_of_toy) = 3 * price_of_hat := by
  sorry

end kendra_bought_3_hats_l140_140508


namespace find_admission_score_l140_140075

noncomputable def admission_score : ℝ := 87

theorem find_admission_score :
  ∀ (total_students admitted_students not_admitted_students : ℝ) 
    (admission_score admitted_avg not_admitted_avg overall_avg : ℝ),
    admitted_students = total_students / 4 →
    not_admitted_students = 3 * admitted_students →
    admitted_avg = admission_score + 10 →
    not_admitted_avg = admission_score - 26 →
    overall_avg = 70 →
    total_students * overall_avg = 
    (admitted_students * admitted_avg + not_admitted_students * not_admitted_avg) →
    admission_score = 87 :=
by
  intros total_students admitted_students not_admitted_students 
         admission_score admitted_avg not_admitted_avg overall_avg
         h1 h2 h3 h4 h5 h6
  sorry

end find_admission_score_l140_140075


namespace jessica_has_three_dozens_of_red_marbles_l140_140240

-- Define the number of red marbles Sandy has
def sandy_red_marbles : ℕ := 144

-- Define the relationship between Sandy's and Jessica's red marbles
def relationship (jessica_red_marbles : ℕ) : Prop :=
  sandy_red_marbles = 4 * jessica_red_marbles

-- Define the question to find out how many dozens of red marbles Jessica has
def jessica_dozens (jessica_red_marbles : ℕ) := jessica_red_marbles / 12

-- Theorem stating that given the conditions, Jessica has 3 dozens of red marbles
theorem jessica_has_three_dozens_of_red_marbles (jessica_red_marbles : ℕ)
  (h : relationship jessica_red_marbles) : jessica_dozens jessica_red_marbles = 3 :=
by
  -- The proof is omitted
  sorry

end jessica_has_three_dozens_of_red_marbles_l140_140240


namespace restaurant_bill_l140_140011

theorem restaurant_bill 
  (salisbury_steak : ℝ := 16.00)
  (chicken_fried_steak : ℝ := 18.00)
  (mozzarella_sticks : ℝ := 8.00)
  (caesar_salad : ℝ := 6.00)
  (bowl_chili : ℝ := 7.00)
  (chocolate_lava_cake : ℝ := 7.50)
  (cheesecake : ℝ := 6.50)
  (iced_tea : ℝ := 3.00)
  (soda : ℝ := 3.50)
  (half_off_meal : ℝ := 0.5)
  (dessert_discount : ℝ := 0.1)
  (tip_percent : ℝ := 0.2)
  (sales_tax : ℝ := 0.085) :
  let total : ℝ :=
    (salisbury_steak * half_off_meal) +
    (chicken_fried_steak * half_off_meal) +
    mozzarella_sticks +
    caesar_salad +
    bowl_chili +
    (chocolate_lava_cake * (1 - dessert_discount)) +
    (cheesecake * (1 - dessert_discount)) +
    iced_tea +
    soda
  let total_with_tax : ℝ := total * (1 + sales_tax)
  let final_total : ℝ := total_with_tax * (1 + tip_percent)
  final_total = 73.04 :=
by
  sorry

end restaurant_bill_l140_140011


namespace second_less_than_first_third_less_than_first_l140_140115

variable (X : ℝ)

def first_number : ℝ := 0.70 * X
def second_number : ℝ := 0.63 * X
def third_number : ℝ := 0.59 * X

theorem second_less_than_first : 
  ((first_number X - second_number X) / first_number X * 100) = 10 :=
by
  sorry

theorem third_less_than_first : 
  ((third_number X - first_number X) / first_number X * 100) = -15.71 :=
by
  sorry

end second_less_than_first_third_less_than_first_l140_140115


namespace find_x0_l140_140979

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

theorem find_x0 (x0 : ℝ) (h : f' x0 = 2) : x0 = Real.exp 1 :=
by {
  sorry
}

end find_x0_l140_140979


namespace transformations_map_onto_self_l140_140963

/-- Define the transformations -/
def T1 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for a 90 degree rotation around the center of a square
  sorry

def T2 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for a translation parallel to line ℓ
  sorry

def T3 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for reflection across line ℓ
  sorry

def T4 (pattern : ℝ × ℝ → Type) : (ℝ × ℝ) → Type :=
  -- Transformation for reflection across a line perpendicular to line ℓ
  sorry

/-- Define the pattern -/
def pattern (p : ℝ × ℝ) : Type :=
  -- Representation of alternating right triangles and squares along line ℓ
  sorry

/-- The main theorem:
    Prove that there are exactly 3 transformations (T1, T2, T3) that will map the pattern onto itself. -/
theorem transformations_map_onto_self : (∃ pattern : ℝ × ℝ → Type,
  (T1 pattern = pattern) ∧
  (T2 pattern = pattern) ∧
  (T3 pattern = pattern) ∧
  ¬ (T4 pattern = pattern)) → (3 = 3) :=
by
  sorry

end transformations_map_onto_self_l140_140963


namespace employee_salaries_l140_140635

theorem employee_salaries 
  (x y z : ℝ)
  (h1 : x + y + z = 638)
  (h2 : x = 1.20 * y)
  (h3 : z = 0.80 * y) :
  x = 255.20 ∧ y = 212.67 ∧ z = 170.14 :=
sorry

end employee_salaries_l140_140635


namespace gift_wrapping_combinations_l140_140403

theorem gift_wrapping_combinations :
  (10 * 4 * 5 * 2 = 400) := by
  sorry

end gift_wrapping_combinations_l140_140403


namespace max_pieces_with_three_cuts_l140_140649

def cake := Type

noncomputable def max_identical_pieces (cuts : ℕ) (max_cuts : ℕ) : ℕ :=
  if cuts = 3 ∧ max_cuts = 3 then 8 else sorry

theorem max_pieces_with_three_cuts : ∀ (c : cake), max_identical_pieces 3 3 = 8 :=
by
  intro c
  sorry

end max_pieces_with_three_cuts_l140_140649


namespace arithmetic_series_remainder_l140_140788

-- Define the sequence parameters
def a : ℕ := 2
def l : ℕ := 12
def d : ℕ := 1
def n : ℕ := (l - a) / d + 1

-- Define the sum of the arithmetic series
def S : ℕ := n * (a + l) / 2

-- The final theorem statement
theorem arithmetic_series_remainder : S % 9 = 5 := 
by sorry

end arithmetic_series_remainder_l140_140788


namespace problem_statement_l140_140601

def atOp (a b : ℝ) := a * b ^ (1 / 2)

theorem problem_statement : atOp ((2 * 3) ^ 2) ((3 * 5) ^ 2 / 9) = 180 := by
  sorry

end problem_statement_l140_140601


namespace lcm_prime_factors_l140_140321

-- Conditions
def n1 := 48
def n2 := 180
def n3 := 250

-- The equivalent proof problem
theorem lcm_prime_factors (l : ℕ) (h1: l = Nat.lcm n1 (Nat.lcm n2 n3)) :
  l = 18000 ∧ (∀ a : ℕ, a ∣ l ↔ a ∣ 2^4 * 3^2 * 5^3) :=
by
  sorry

end lcm_prime_factors_l140_140321


namespace winter_expenditure_l140_140905

theorem winter_expenditure (exp_end_nov : Real) (exp_end_feb : Real) 
  (h_nov : exp_end_nov = 3.0) (h_feb : exp_end_feb = 5.5) : 
  (exp_end_feb - exp_end_nov) = 2.5 :=
by 
  sorry

end winter_expenditure_l140_140905


namespace carol_first_six_probability_l140_140002

theorem carol_first_six_probability :
  let p := 1 / 6
  let q := 5 / 6
  let prob_cycle := q^4
  (p * q^3) / (1 - prob_cycle) = 125 / 671 :=
by
  sorry

end carol_first_six_probability_l140_140002


namespace cubes_sum_identity_l140_140600

variable {a b : ℝ}

theorem cubes_sum_identity (h : (a / (1 + b) + b / (1 + a) = 1)) : a^3 + b^3 = a + b :=
sorry

end cubes_sum_identity_l140_140600


namespace sum_fractions_l140_140554

theorem sum_fractions : 
  (1/2 + 1/6 + 1/12 + 1/20 + 1/30 + 1/42 = 6/7) :=
by
  sorry

end sum_fractions_l140_140554


namespace milk_processing_days_required_l140_140275

variable (a m x : ℝ) (n : ℝ)

theorem milk_processing_days_required
  (h1 : (n - a) * (x + m) = nx)
  (h2 : ax + (10 * a / 9) * x + (5 * a / 9) * m = 2 / 3)
  (h3 : nx = 1 / 2) :
  n = 2 * a :=
by sorry

end milk_processing_days_required_l140_140275


namespace desargues_theorem_l140_140771

open Point Line Collinear

variable {A A1 B B1 C C1 : Point}
variable {AA1 BB1 CC1 : Line}
variable {O A2 B2 C2 : Point}
variable {AB A1B1 BC B1C1 AC A1C1 : Line}

-- lines intersections as given conditions
axiom h1 : AA1 = Line_through A A1
axiom h2 : BB1 = Line_through B B1
axiom h3 : CC1 = Line_through C C1
axiom h4 : intersect AA1 BB1 CC1 = O

-- intersection points definitions
def A2 : Point := intersection BC B1C1
def B2 : Point := intersection AC A1C1
def C2 : Point := intersection AB A1B1

-- to prove they are collinear
theorem desargues_theorem (h1 : intersect AA1 BB1 CC1 = O) :
    Collinear A2 B2 C2 := by
  sorry

end desargues_theorem_l140_140771


namespace determinant_in_terms_of_roots_l140_140890

noncomputable def determinant_3x3 (a b c : ℝ) : ℝ :=
  (1 + a) * ((1 + b) * (1 + c) - 1) - 1 * (1 + c) + (1 + b) * 1

theorem determinant_in_terms_of_roots (a b c s p q : ℝ)
  (h1 : a + b + c = -s)
  (h2 : a * b + a * c + b * c = p)
  (h3 : a * b * c = -q) :
  determinant_3x3 a b c = -q + p - s :=
by
  sorry

end determinant_in_terms_of_roots_l140_140890


namespace prob_B_given_A_l140_140637

/-
Define events A and B:
- A: Blue die results in 4 or 6
- B: Sum of results of red die and blue die is greater than 8
-/

def event_A (blue : ℕ) : Prop := blue = 4 ∨ blue = 6
def event_B (red blue : ℕ) : Prop := red + blue > 8

theorem prob_B_given_A :
  let outcomes := (1..6).bind (fun red => (1..6).map (fun blue => (red, blue))) in
  let PA := (outcomes.filter (fun ⟨r, b⟩ => event_A b)).length / outcomes.length.toFloat in
  let PAB := (outcomes.filter (fun ⟨r, b⟩ => event_A b ∧ event_B r b)).length / outcomes.length.toFloat in
  PAB / PA = 1 / 2 := sorry


end prob_B_given_A_l140_140637


namespace difference_counts_l140_140443

noncomputable def τ (n : ℕ) : ℕ := n.factors.length + 1

noncomputable def S (n : ℕ) : ℕ := (list.range n.succ).sum (λ i, τ i)

def count_odd_S_up_to (n : ℕ) : ℕ :=
(list.range n.succ).countp (λ i, odd (S i))

def count_even_S_up_to (n : ℕ) : ℕ :=
(list.range n.succ).countp (λ i, even (S i))

theorem difference_counts {c d : ℕ} :
  c = count_odd_S_up_to 3000 →
  d = count_even_S_up_to 3000 →
  |c - d| = 1733 :=
by
  intros h_c h_d
  sorry

end difference_counts_l140_140443


namespace max_unmarried_women_l140_140750

theorem max_unmarried_women (total_people : ℕ) (frac_women : ℚ) (frac_married : ℚ)
  (h_total : total_people = 80) (h_frac_women : frac_women = 1 / 4) (h_frac_married : frac_married = 3 / 4) :
  ∃ (max_unmarried_women : ℕ), max_unmarried_women = 20 :=
by
  -- The proof will be filled here
  sorry

end max_unmarried_women_l140_140750


namespace find_x_value_l140_140652

theorem find_x_value : ∃ x : ℝ, 35 - (23 - (15 - x)) = 12 * 2 / 1 / 2 → x = -21 :=
by
  sorry

end find_x_value_l140_140652


namespace team_arrangement_count_l140_140413

-- Definitions of the problem
def veteran_players := 2
def new_players := 3
def total_players := veteran_players + new_players
def team_size := 3

-- Conditions
def condition_veteran : Prop := 
  ∀ (team : Finset ℕ), team.card = team_size → Finset.card (team ∩ (Finset.range veteran_players)) ≥ 1

def condition_new_player : Prop := 
  ∀ (team : Finset ℕ), team.card = team_size → 
    ∃ (p1 p2 : ℕ), p1 ∈ team ∧ p2 ∈ team ∧ 
    p1 ≠ p2 ∧ p1 < team_size ∧ p2 < team_size ∧
    (p1 ∈ (Finset.Ico veteran_players total_players) ∨ p2 ∈ (Finset.Ico veteran_players total_players))

-- Goal
def number_of_arrangements := 48

-- The statement to prove
theorem team_arrangement_count : condition_veteran → condition_new_player → 
  (∃ (arrangements : ℕ), arrangements = number_of_arrangements) :=
by
  sorry

end team_arrangement_count_l140_140413


namespace total_medals_1996_l140_140874

variable (g s b : Nat)

theorem total_medals_1996 (h_g : g = 16) (h_s : s = 22) (h_b : b = 12) :
  g + s + b = 50 :=
by
  sorry

end total_medals_1996_l140_140874


namespace sum_of_powers_modulo_seven_l140_140780

theorem sum_of_powers_modulo_seven :
  ((1^1 + 2^2 + 3^3 + 4^4 + 5^5 + 6^6 + 7^7) % 7) = 1 := by
  sorry

end sum_of_powers_modulo_seven_l140_140780


namespace determine_a7_l140_140358

noncomputable def arithmetic_seq (a1 d : ℤ) : ℕ → ℤ
| 0     => a1
| (n+1) => a1 + n * d

noncomputable def sum_arithmetic_seq (a1 d : ℤ) : ℕ → ℤ
| 0     => 0
| (n+1) => a1 * (n + 1) + (n * (n + 1) * d) / 2

theorem determine_a7 (a1 d : ℤ) (a2 : a1 + d = 7) (S7 : sum_arithmetic_seq a1 d 7 = -7) : arithmetic_seq a1 d 7 = -13 :=
by
  sorry

end determine_a7_l140_140358


namespace problem_conditions_l140_140706

theorem problem_conditions (x y : ℝ) (hx : x * (Real.exp x + Real.log x + x) = 1) (hy : y * (2 * Real.log y + Real.log (Real.log y)) = 1) :
  (0 < x ∧ x < 1) ∧ (y - x > 1) ∧ (y - x < 3 / 2) :=
by
  sorry

end problem_conditions_l140_140706


namespace most_accurate_reading_l140_140491

def temperature_reading (temp: ℝ) : Prop := 
  98.6 ≤ temp ∧ temp ≤ 99.1 ∧ temp ≠ 98.85 ∧ temp > 98.85

theorem most_accurate_reading (temp: ℝ) : temperature_reading temp → temp = 99.1 :=
by
  intros h
  sorry 

end most_accurate_reading_l140_140491


namespace geometric_sequence_a3_l140_140232

theorem geometric_sequence_a3 (q : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : a 5 = 4) (h3 : ∀ n, a n = a 1 * q ^ (n - 1)) : a 3 = 2 :=
by
  sorry

end geometric_sequence_a3_l140_140232


namespace calculate_expression_l140_140828

theorem calculate_expression : 6 * (8 + 1/3) = 50 := by
  sorry

end calculate_expression_l140_140828


namespace selene_total_payment_l140_140261

def price_instant_camera : ℝ := 110
def num_instant_cameras : ℕ := 2
def discount_instant_camera : ℝ := 0.07
def price_photo_frame : ℝ := 120
def num_photo_frames : ℕ := 3
def discount_photo_frame : ℝ := 0.05
def sales_tax : ℝ := 0.06

theorem selene_total_payment :
  let total_instant_cameras := num_instant_cameras * price_instant_camera
  let discount_instant := total_instant_cameras * discount_instant_camera
  let discounted_instant := total_instant_cameras - discount_instant
  let total_photo_frames := num_photo_frames * price_photo_frame
  let discount_photo := total_photo_frames * discount_photo_frame
  let discounted_photo := total_photo_frames - discount_photo
  let subtotal := discounted_instant + discounted_photo
  let tax := subtotal * sales_tax
  let total_payment := subtotal + tax
  total_payment = 579.40 :=
by
  sorry

end selene_total_payment_l140_140261


namespace radius_of_smaller_molds_l140_140813

noncomputable def hemisphereVolume (r : ℝ) : ℝ :=
  (2 / 3) * Real.pi * r ^ 3

theorem radius_of_smaller_molds :
  ∀ (R r : ℝ), R = 2 ∧ (64 * hemisphereVolume r) = hemisphereVolume R → r = 1 / 2 :=
by
  intros R r h
  sorry

end radius_of_smaller_molds_l140_140813


namespace cubic_common_roots_l140_140438

noncomputable def roots : List ℝ := [1, -1]  -- Assume u and v can be roots 1 and -1 for simplicity

theorem cubic_common_roots (c d : ℝ) :
  (∀ u v : ℝ, u ≠ v ∧ u ∈ roots ∧ v ∈ roots →
    (u ^ 3 + c * u ^ 2 + 8 * u + 5 = 0) ∧
    (v ^ 3 + d * v ^ 2 + 10 * v + 7 = 0) ) → 
  (c = 5 ∧ d = 6) :=
by
  intros h
  sorry

end cubic_common_roots_l140_140438


namespace max_value_of_expression_l140_140613

variables (x y : ℝ)

theorem max_value_of_expression (hx : 0 < x) (hy : 0 < y) (h : x^2 - 2*x*y + 3*y^2 = 12) : x^2 + 2*x*y + 3*y^2 ≤ 24 + 24*sqrt 3 :=
sorry

end max_value_of_expression_l140_140613


namespace remainder_of_product_mod_7_l140_140387

theorem remainder_of_product_mod_7
  (a b c : ℕ)
  (ha : a ≡ 2 [MOD 7])
  (hb : b ≡ 3 [MOD 7])
  (hc : c ≡ 4 [MOD 7]) :
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_mod_7_l140_140387


namespace unique_identity_element_l140_140119

variable {G : Type*} [Group G]

theorem unique_identity_element (e e' : G) (h1 : ∀ g : G, e * g = g ∧ g * e = g) (h2 : ∀ g : G, e' * g = g ∧ g * e' = g) : e = e' :=
by 
sorry

end unique_identity_element_l140_140119


namespace driver_net_rate_of_pay_l140_140937

theorem driver_net_rate_of_pay
  (hours : ℕ)
  (speed : ℕ)
  (fuel_efficiency : ℕ)
  (pay_per_mile : ℚ)
  (gas_cost_per_gallon : ℚ)
  (net_rate_of_pay : ℚ)
  (h1 : hours = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_per_mile = 0.60)
  (h5 : gas_cost_per_gallon = 2.50)
  (h6 : net_rate_of_pay = 25) :
  net_rate_of_pay = (hours * speed * pay_per_mile - (hours * speed / fuel_efficiency) * gas_cost_per_gallon) / hours := 
by sorry

end driver_net_rate_of_pay_l140_140937


namespace deepaks_age_l140_140910

theorem deepaks_age (R D : ℕ) (h1 : R / D = 5 / 2) (h2 : R + 6 = 26) : D = 8 := 
sorry

end deepaks_age_l140_140910


namespace golden_ratio_problem_l140_140879

noncomputable def m := 2 * Real.sin (Real.pi * 18 / 180)
noncomputable def n := 4 - m^2
noncomputable def target_expression := m * Real.sqrt n / (2 * (Real.cos (Real.pi * 27 / 180))^2 - 1)

theorem golden_ratio_problem :
  target_expression = 2 :=
by
  -- Proof will be placed here
  sorry

end golden_ratio_problem_l140_140879


namespace converse_proposition_l140_140377

theorem converse_proposition (x : ℝ) (h : x = 1 → x^2 = 1) : x^2 = 1 → x = 1 :=
by
  sorry

end converse_proposition_l140_140377


namespace pints_in_two_liters_nearest_tenth_l140_140715

def liters_to_pints (liters : ℝ) : ℝ :=
  2.1 * liters

theorem pints_in_two_liters_nearest_tenth :
  liters_to_pints 2 = 4.2 :=
by
  sorry

end pints_in_two_liters_nearest_tenth_l140_140715


namespace area_of_overlap_l140_140901

def area_of_square_1 : ℝ := 1
def area_of_square_2 : ℝ := 4
def area_of_square_3 : ℝ := 9
def area_of_square_4 : ℝ := 16
def total_area_of_rectangle : ℝ := 27.5
def unshaded_area : ℝ := 1.5

def total_area_of_squares : ℝ := area_of_square_1 + area_of_square_2 + area_of_square_3 + area_of_square_4
def total_area_covered_by_squares : ℝ := total_area_of_rectangle - unshaded_area

theorem area_of_overlap :
  total_area_of_squares - total_area_covered_by_squares = 4 := 
sorry

end area_of_overlap_l140_140901


namespace domain_f_2x_l140_140988

-- Given conditions as definitions
def domain_f_x_minus_1 (x : ℝ) := 3 < x ∧ x ≤ 7

-- The main theorem statement that needs a proof
theorem domain_f_2x : (∀ x : ℝ, domain_f_x_minus_1 (x-1) → (1 < x ∧ x ≤ 3)) :=
by
  -- Proof steps will be here, however, as requested, they are omitted.
  sorry

end domain_f_2x_l140_140988


namespace midpoint_line_l140_140498

theorem midpoint_line (a : ℝ) (P Q M : ℝ × ℝ) (hP : P = (a, 5 * a + 3)) (hQ : Q = (3, -2))
  (hM : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) : M.2 = 5 * M.1 - 7 := 
sorry

end midpoint_line_l140_140498


namespace sqrt_diff_approx_l140_140291

theorem sqrt_diff_approx : abs ((Real.sqrt 122) - (Real.sqrt 120) - 0.15) < 0.01 := 
sorry

end sqrt_diff_approx_l140_140291


namespace factor_expression_l140_140682

-- Define the expressions E1 and E2
def E1 (y : ℝ) : ℝ := 12 * y^6 + 35 * y^4 - 5
def E2 (y : ℝ) : ℝ := 2 * y^6 - 4 * y^4 + 5

-- Define the target expression E
def E (y : ℝ) : ℝ := E1 y - E2 y

-- The main theorem to prove
theorem factor_expression (y : ℝ) : E y = 10 * (y^6 + 3.9 * y^4 - 1) := by
  sorry

end factor_expression_l140_140682


namespace hyperbola_focus_l140_140889

theorem hyperbola_focus (m : ℝ) (h : (0, 5) = (0, 5)) : 
  (∀ x y : ℝ, (y^2 / m - x^2 / 9 = 1) → m = 16) :=
sorry

end hyperbola_focus_l140_140889


namespace slower_pipe_filling_time_l140_140650

-- Definitions based on conditions
def faster_pipe_rate (S : ℝ) : ℝ := 3 * S
def combined_rate (S : ℝ) : ℝ := (faster_pipe_rate S) + S

-- Statement of what needs to be proved 
theorem slower_pipe_filling_time :
  (∀ S : ℝ, combined_rate S * 40 = 1) →
  ∃ t : ℝ, t = 160 :=
by
  intro h
  sorry

end slower_pipe_filling_time_l140_140650


namespace range_of_x_l140_140557

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_x (f_even : ∀ x : ℝ, f x = f (-x))
                   (f_increasing : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)
                   (f_at_one_third : f (1/3) = 0) :
  {x : ℝ | f (Real.log x / Real.log (1/8)) > 0} = {x : ℝ | (0 < x ∧ x < 1/2) ∨ 2 < x} :=
sorry

end range_of_x_l140_140557


namespace brody_battery_fraction_l140_140678

theorem brody_battery_fraction (full_battery : ℕ) (battery_left_after_exam : ℕ) (exam_duration : ℕ) 
  (battery_before_exam : ℕ) (battery_used : ℕ) (fraction_used : ℚ) 
  (h1 : full_battery = 60)
  (h2 : battery_left_after_exam = 13)
  (h3 : exam_duration = 2)
  (h4 : battery_before_exam = battery_left_after_exam + exam_duration)
  (h5 : battery_used = full_battery - battery_before_exam)
  (h6 : fraction_used = battery_used / full_battery) :
  fraction_used = 3 / 4 := 
sorry

end brody_battery_fraction_l140_140678


namespace intervals_of_monotonicity_and_extreme_values_l140_140174

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem intervals_of_monotonicity_and_extreme_values :
  (∀ x : ℝ, x < 1 → deriv f x > 0) ∧
  (∀ x : ℝ, x > 1 → deriv f x < 0) ∧
  (∀ x : ℝ, f 1 = 1 / Real.exp 1) :=
by
  sorry

end intervals_of_monotonicity_and_extreme_values_l140_140174


namespace car_speed_l140_140798

theorem car_speed (v : ℝ) (h₁ : (1/75 * 3600) + 12 = 1/v * 3600) : v = 60 := 
by 
  sorry

end car_speed_l140_140798


namespace Petya_wrong_example_l140_140497

def a := 8
def b := 128

theorem Petya_wrong_example : (a^7 ∣ b^3) ∧ ¬ (a^2 ∣ b) :=
by {
  -- Prove the divisibility conditions and the counterexample
  sorry
}

end Petya_wrong_example_l140_140497


namespace sum_even_coeffs_l140_140581

theorem sum_even_coeffs (a : ℝ)
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ)
  (h_eq : ((a : ℝ) * x + 1)^5 * (x + 2)^4 = 
           a_0 * (x + 2) ^ 9 + a_1 * (x + 2) ^ 8 + 
           a_2 * (x + 2) ^ 7 + a_3 * (x + 2) ^ 6 + 
           a_4 * (x + 2) ^ 5 + a_5 * (x + 2) ^ 4 + 
           a_6 * (x + 2) ^ 3 + a_7 * (x + 2) ^ 2 + 
           a_8 * (x + 2) + a_9)
  (h_sum : (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + 
           a_6 + a_7 + a_8 + a_9 = 1024)) :
  a_0 + a_2 + a_4 + a_6 + a_8 = (2^10 - 14^5) / 2 :=
sorry

end sum_even_coeffs_l140_140581


namespace sum_of_squares_of_roots_l140_140217

/-- If r, s, and t are the roots of the cubic equation x³ - ax² + bx - c = 0, then r² + s² + t² = a² - 2b. -/
theorem sum_of_squares_of_roots (r s t a b c : ℝ) (h1 : r + s + t = a) (h2 : r * s + r * t + s * t = b) (h3 : r * s * t = c) :
    r ^ 2 + s ^ 2 + t ^ 2 = a ^ 2 - 2 * b := 
by 
  sorry

end sum_of_squares_of_roots_l140_140217


namespace exists_indices_non_decreasing_l140_140754

theorem exists_indices_non_decreasing
    (a b c : ℕ → ℕ) :
    ∃ p q : ℕ, p ≠ q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
  sorry

end exists_indices_non_decreasing_l140_140754


namespace find_x_y_l140_140196

theorem find_x_y (x y : ℝ)
  (h1 : (x - 1) ^ 2003 + 2002 * (x - 1) = -1)
  (h2 : (y - 2) ^ 2003 + 2002 * (y - 2) = 1) :
  x + y = 3 :=
sorry

end find_x_y_l140_140196


namespace intersection_with_xz_plane_l140_140849

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def direction_vector (p1 p2 : Point3D) : Point3D :=
  Point3D.mk (p2.x - p1.x) (p2.y - p1.y) (p2.z - p1.z)

def parametric_eqn (p : Point3D) (d : Point3D) (t : ℝ) : Point3D :=
  Point3D.mk (p.x + t * d.x) (p.y + t * d.y) (p.z + t * d.z)

theorem intersection_with_xz_plane (p1 p2 : Point3D) :
  let d := direction_vector p1 p2
  let t := (p1.y / d.y)
  parametric_eqn p1 d t = Point3D.mk 4 0 9 :=
sorry

#check intersection_with_xz_plane

end intersection_with_xz_plane_l140_140849


namespace max_min_fraction_l140_140694

-- Given condition
def circle_condition (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y + 1 = 0

-- Problem statement
theorem max_min_fraction (x y : ℝ) (h : circle_condition x y) :
  -20 / 21 ≤ y / (x - 4) ∧ y / (x - 4) ≤ 0 :=
sorry

end max_min_fraction_l140_140694


namespace base_b_eq_five_l140_140218

theorem base_b_eq_five (b : ℕ) (h1 : 1225 = b^3 + 2 * b^2 + 2 * b + 5) (h2 : 35 = 3 * b + 5) :
    (3 * b + 5)^2 = b^3 + 2 * b^2 + 2 * b + 5 ↔ b = 5 :=
by
  sorry

end base_b_eq_five_l140_140218


namespace molecular_weight_of_7_moles_boric_acid_l140_140393

-- Define the given constants.
def atomic_weight_H : ℝ := 1.008
def atomic_weight_B : ℝ := 10.81
def atomic_weight_O : ℝ := 16.00

-- Define the molecular formula for boric acid.
def molecular_weight_H3BO3 : ℝ :=
  3 * atomic_weight_H + 1 * atomic_weight_B + 3 * atomic_weight_O

-- Define the number of moles.
def moles_boric_acid : ℝ := 7

-- Calculate the total weight for 7 moles of boric acid.
def total_weight_boric_acid : ℝ :=
  moles_boric_acid * molecular_weight_H3BO3

-- The target statement to prove.
theorem molecular_weight_of_7_moles_boric_acid :
  total_weight_boric_acid = 432.838 := by
  sorry

end molecular_weight_of_7_moles_boric_acid_l140_140393


namespace triangle_side_length_l140_140074

theorem triangle_side_length (a : ℝ) (B : ℝ) (C : ℝ) (c : ℝ) 
  (h₀ : a = 10) (h₁ : B = 60) (h₂ : C = 45) : 
  c = 10 * (Real.sqrt 3 - 1) :=
sorry

end triangle_side_length_l140_140074


namespace average_production_is_correct_l140_140875

noncomputable def average_tv_production_last_5_days
  (daily_production : ℕ)
  (ill_workers : List ℕ)
  (decrease_rate : ℕ) : ℚ :=
  let productivity_decrease (n : ℕ) : ℚ := (1 - (decrease_rate * n) / 100 : ℚ) * daily_production
  let total_production := (ill_workers.map productivity_decrease).sum
  total_production / ill_workers.length

theorem average_production_is_correct :
  average_tv_production_last_5_days 50 [3, 5, 2, 4, 3] 2 = 46.6 :=
by
  -- proof needed here
  sorry

end average_production_is_correct_l140_140875


namespace smaller_number_l140_140626

theorem smaller_number (x y : ℤ) (h1 : x + y = 12) (h2 : x - y = 20) : y = -4 := 
by 
  sorry

end smaller_number_l140_140626


namespace telescoping_series_sum_l140_140960

theorem telescoping_series_sum :
  (∑' (n : ℕ) in (Finset.range (0) \ Finset.singleton (0)), (↑(3 * n + 2) / (↑n * (↑n + 1) * (↑n + 3)))) = (5 / 6) := sorry

end telescoping_series_sum_l140_140960


namespace matrix_eq_value_satisfied_for_two_values_l140_140871

variable (a b c d x : ℝ)

def matrix_value (a b c d : ℝ) : ℝ := a * b - c * d

-- Define the specific instance for the given matrix problem
def matrix_eq_value (x : ℝ) : Prop :=
  matrix_value (2 * x) x 1 x = 3

-- Prove that the equation is satisfied for exactly two values of x
theorem matrix_eq_value_satisfied_for_two_values :
  (∃! (x : ℝ), matrix_value (2 * x) x 1 x = 3) :=
sorry

end matrix_eq_value_satisfied_for_two_values_l140_140871


namespace percentage_increase_l140_140651

def old_price : ℝ := 300
def new_price : ℝ := 330

theorem percentage_increase : ((new_price - old_price) / old_price) * 100 = 10 := by
  sorry

end percentage_increase_l140_140651


namespace solve_inequality_l140_140760

theorem solve_inequality (a : ℝ) (x : ℝ) :
  (a = 0 → x > 1 → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (0 < a → a < 2 → 1 < x → x < 2 / a → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (a = 2 → False → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (a > 2 → 2 / a < x → x < 1 → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (a < 0 → ((x < 2 / a ∨ x > 1) → (ax^2 - (a + 2) * x + 2 < 0))) := sorry

end solve_inequality_l140_140760


namespace find_ages_l140_140141

theorem find_ages (J sister cousin : ℝ)
  (h1 : J + 9 = 3 * (J - 11))
  (h2 : sister = 2 * J)
  (h3 : cousin = (J + sister) / 2) :
  J = 21 ∧ sister = 42 ∧ cousin = 31.5 :=
by
  sorry

end find_ages_l140_140141


namespace filter_replacement_in_December_l140_140431

theorem filter_replacement_in_December
  (replacement_interval : ℕ)
  (initial_month : ℕ)
  (nth_replacement : ℕ)
  (months_in_year : ℕ)
  (nth_replacement_month : ℕ) :
  replacement_interval = 7 →
  initial_month = 1 →
  nth_replacement = 18 →
  months_in_year = 12 →
  nth_replacement_month = (replacement_interval * (nth_replacement - 1) % months_in_year) + 1 →
  nth_replacement_month = 12 :=
begin
  intros h1 h2 h3 h4 h5,
  rw [h1, h2, h3, h4] at h5,
  norm_num at h5,
  exact h5,
end

end filter_replacement_in_December_l140_140431


namespace exists_segment_satisfying_condition_l140_140230

theorem exists_segment_satisfying_condition :
  ∃ (x₁ x₂ x₃ : ℚ) (f : ℚ → ℤ), x₃ = (x₁ + x₂) / 2 ∧ f x₁ + f x₂ ≤ 2 * f x₃ :=
sorry

end exists_segment_satisfying_condition_l140_140230


namespace sum_of_repeating_decimals_correct_l140_140679

/-- Convert repeating decimals to fractions -/
def rep_dec_1 : ℚ := 1 / 9
def rep_dec_2 : ℚ := 2 / 9
def rep_dec_3 : ℚ := 1 / 3
def rep_dec_4 : ℚ := 4 / 9
def rep_dec_5 : ℚ := 5 / 9
def rep_dec_6 : ℚ := 2 / 3
def rep_dec_7 : ℚ := 7 / 9
def rep_dec_8 : ℚ := 8 / 9

/-- Define the terms in the sum -/
def term_1 : ℚ := 8 + rep_dec_1
def term_2 : ℚ := 7 + 1 + rep_dec_2
def term_3 : ℚ := 6 + 2 + rep_dec_3
def term_4 : ℚ := 5 + 3 + rep_dec_4
def term_5 : ℚ := 4 + 4 + rep_dec_5
def term_6 : ℚ := 3 + 5 + rep_dec_6
def term_7 : ℚ := 2 + 6 + rep_dec_7
def term_8 : ℚ := 1 + 7 + rep_dec_8

/-- Define the sum of the terms -/
def total_sum : ℚ := term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8

/-- Proof problem statement -/
theorem sum_of_repeating_decimals_correct : total_sum = 39.2 := 
sorry

end sum_of_repeating_decimals_correct_l140_140679


namespace slope_line_OM_l140_140863

theorem slope_line_OM : 
  let t := (Real.pi / 3)
  let x := 2 * Real.cos t
  let y := 4 * Real.sin t
  let M := (x, y)
  let O := (0 : ℝ, 0 : ℝ)
  (M = (1, 2 * Real.sqrt 3)) →
  (O = (0, 0)) →
  (if (O.1 ≠ M.1) then (((M.2 - O.2) / (M.1 - O.1)) = 2 * Real.sqrt 3) else False) :=
by
  intro t x y M O hM hO
  exact sorry

end slope_line_OM_l140_140863


namespace cost_price_of_apple_is_18_l140_140928

noncomputable def cp (sp : ℝ) (loss_fraction : ℝ) : ℝ := sp / (1 - loss_fraction)

theorem cost_price_of_apple_is_18 :
  cp 15 (1/6) = 18 :=
by
  sorry

end cost_price_of_apple_is_18_l140_140928


namespace arithmetic_series_first_term_l140_140700

theorem arithmetic_series_first_term 
  (a d : ℚ)
  (h1 : 15 * (2 * a +  29 * d) = 450)
  (h2 : 15 * (2 * a + 89 * d) = 1650) :
  a = -13 / 3 :=
by
  sorry

end arithmetic_series_first_term_l140_140700


namespace crayons_difference_l140_140090

def initial_crayons : ℕ := 8597
def crayons_given : ℕ := 7255
def crayons_lost : ℕ := 3689

theorem crayons_difference : crayons_given - crayons_lost = 3566 := by
  sorry

end crayons_difference_l140_140090


namespace strawberry_rows_l140_140364

theorem strawberry_rows (yield_per_row total_harvest : ℕ) (h1 : yield_per_row = 268) (h2 : total_harvest = 1876) :
  total_harvest / yield_per_row = 7 := 
by 
  sorry

end strawberry_rows_l140_140364


namespace range_of_a_l140_140984

theorem range_of_a (a : ℝ) (h : a ≤ 1) :
  (∃! n : ℕ, n = (2 - a) - a + 1) → -1 < a ∧ a ≤ 0 :=
by 
  sorry

end range_of_a_l140_140984


namespace fit_small_boxes_l140_140346

def larger_box_volume (length width height : ℕ) : ℕ :=
  length * width * height

def small_box_volume (length width height : ℕ) : ℕ :=
  length * width * height

theorem fit_small_boxes (L W H l w h : ℕ)
  (larger_box_dim : L = 12 ∧ W = 14 ∧ H = 16)
  (small_box_dim : l = 3 ∧ w = 7 ∧ h = 2)
  (min_boxes : larger_box_volume L W H / small_box_volume l w h = 64) :
  ∃ n, n ≥ 64 :=
by
  sorry

end fit_small_boxes_l140_140346


namespace sqrt_meaningful_real_domain_l140_140073

theorem sqrt_meaningful_real_domain (x : ℝ) (h : 6 - 4 * x ≥ 0) : x ≤ 3 / 2 :=
by sorry

end sqrt_meaningful_real_domain_l140_140073


namespace zarnin_staffing_l140_140680

theorem zarnin_staffing (n total unsuitable : ℕ) (unsuitable_factor : ℕ) (job_openings : ℕ)
  (h1 : total = 30) 
  (h2 : unsuitable_factor = 2 / 3) 
  (h3 : unsuitable = unsuitable_factor * total) 
  (h4 : n = total - unsuitable)
  (h5 : job_openings = 5) :
  (n - 0) * (n - 1) * (n - 2) * (n - 3) * (n - 4) = 30240 := by
    sorry

end zarnin_staffing_l140_140680


namespace arithmetic_series_sum_proof_middle_term_proof_l140_140189

def arithmetic_series_sum (a d n : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

def middle_term (a l : ℤ) : ℤ :=
  (a + l) / 2

theorem arithmetic_series_sum_proof :
  let a := -51
  let d := 2
  let n := 27
  let l := 1
  arithmetic_series_sum a d n = -675 :=
by
  sorry

theorem middle_term_proof :
  let a := -51
  let l := 1
  middle_term a l = -25 :=
by
  sorry

end arithmetic_series_sum_proof_middle_term_proof_l140_140189


namespace class_average_l140_140998

theorem class_average (p1 p2 p3 avg1 avg2 avg3 overall_avg : ℕ) 
  (h1 : p1 = 45) 
  (h2 : p2 = 50) 
  (h3 : p3 = 100 - p1 - p2) 
  (havg1 : avg1 = 95) 
  (havg2 : avg2 = 78) 
  (havg3 : avg3 = 60) 
  (hoverall : overall_avg = (p1 * avg1 + p2 * avg2 + p3 * avg3) / 100) : 
  overall_avg = 85 :=
by
  sorry

end class_average_l140_140998


namespace square_must_rotate_at_least_5_turns_l140_140043

-- Define the square and pentagon as having equal side lengths
def square_sides : Nat := 4
def pentagon_sides : Nat := 5

-- The problem requires us to prove that the square needs to rotate at least 5 full turns
theorem square_must_rotate_at_least_5_turns :
  let lcm := Nat.lcm square_sides pentagon_sides
  lcm / square_sides = 5 :=
by
  -- Proof to be provided
  sorry

end square_must_rotate_at_least_5_turns_l140_140043


namespace man_walking_speed_l140_140404

-- This statement introduces the assumptions and goals of the proof problem.
theorem man_walking_speed
  (x : ℝ)
  (h1 : (25 * (1 / 12)) = (x * (1 / 3)))
  : x = 6.25 :=
sorry

end man_walking_speed_l140_140404


namespace Antoinette_weight_l140_140166

-- Define weights for Antoinette and Rupert
variables (A R : ℕ)

-- Define the given conditions
def condition1 := A = 2 * R - 7
def condition2 := A + R = 98

-- The theorem to prove under the given conditions
theorem Antoinette_weight : condition1 A R → condition2 A R → A = 63 := 
by {
  -- The proof is omitted
  sorry
}

end Antoinette_weight_l140_140166


namespace cos_double_angle_l140_140050

theorem cos_double_angle
  {x : ℝ}
  (h : Real.sin x = -2 / 3) :
  Real.cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_l140_140050


namespace probability_at_most_two_heads_l140_140784

open Finset

-- Definitions for the problem
def sample_space : Finset (Finset ℕ) := 
  { {0, 1, 2}, {0, 1}, {0, 2}, {1, 2}, {0}, {1}, {2}, ∅ }

-- Outcome conditions
def at_most_two_heads : Finset (Finset ℕ) :=
  { {0, 1, 2}, {0, 1}, {0, 2}, {1, 2}, {0}, {1}, {2} } -- Excludes ∅

def favorable_outcomes : ℕ := (at_most_two_heads.card : ℕ)
def total_outcomes : ℕ := (sample_space.card)

def probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_at_most_two_heads : probability = 7 / 8 := by
  -- The exact proof to be filled here
  -- For now, we leave a sorry as per the instruction
  sorry

end probability_at_most_two_heads_l140_140784


namespace sum_first_100_even_numbers_divisible_by_6_l140_140561

-- Define the sequence of even numbers divisible by 6 between 100 and 300 inclusive.
def even_numbers_divisible_by_6 (n : ℕ) : ℕ := 102 + n * 6

-- Define the sum of the first 100 even numbers divisible by 6.
def sum_even_numbers_divisible_by_6 (k : ℕ) : ℕ := k / 2 * (102 + (102 + (k - 1) * 6))

-- Define the problem statement as a theorem.
theorem sum_first_100_even_numbers_divisible_by_6 :
  sum_even_numbers_divisible_by_6 100 = 39900 :=
by
  sorry

end sum_first_100_even_numbers_divisible_by_6_l140_140561


namespace find_larger_integer_l140_140278

theorem find_larger_integer (a b : ℕ) (h₁ : a * b = 272) (h₂ : |a - b| = 8) : max a b = 17 :=
sorry

end find_larger_integer_l140_140278


namespace arrangements_three_balls_four_boxes_l140_140259

theorem arrangements_three_balls_four_boxes : 
  ∃ (f : Fin 4 → Fin 4), Function.Injective f :=
sorry

end arrangements_three_balls_four_boxes_l140_140259


namespace problem1_problem2_l140_140140

theorem problem1 : ∃ (m : ℝ) (b : ℝ), ∀ (x y : ℝ),
  3 * x + 4 * y - 2 = 0 ∧ x - y + 4 = 0 →
  y = m * x + b ∧ (1 / m = -2) ∧ (y = - (2 * x + 2)) :=
sorry

theorem problem2 : ∀ (x y a : ℝ), (x = -1) ∧ (y = 3) → 
  (x + y = a) →
  a = 2 ∧ (x + y - 2 = 0) :=
sorry

end problem1_problem2_l140_140140


namespace number_of_real_roots_l140_140023

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem number_of_real_roots (a : ℝ) :
    ((|a| < (2 * Real.sqrt 3) / 9) → (∃ x₁ x₂ x₃ : ℝ, f x₁ = a ∧ f x₂ = a ∧ f x₃ = a ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)) ∧
    ((|a| > (2 * Real.sqrt 3) / 9) → (∃ x : ℝ, f x = a ∧ ∀ y : ℝ, f y = a → y = x)) ∧
    ((|a| = (2 * Real.sqrt 3) / 9) → (∃ x₁ x₂ : ℝ, f x₁ = a ∧ f x₂ = a ∧ x₁ ≠ x₂ ∧ ∀ y : ℝ, (f y = a → (y = x₁ ∨ y = x₂)) ∧ (x₁ = x₂ ∨ ∀ z : ℝ, (f z = a → z = x₁ ∨ z = x₂)))) := sorry

end number_of_real_roots_l140_140023


namespace Wayne_blocks_l140_140392

theorem Wayne_blocks (initial_blocks : ℕ) (additional_blocks : ℕ) (total_blocks : ℕ) 
  (h1 : initial_blocks = 9) (h2 : additional_blocks = 6) 
  (h3 : total_blocks = initial_blocks + additional_blocks) : 
  total_blocks = 15 :=
by {
  -- h1: initial_blocks = 9
  -- h2: additional_blocks = 6
  -- h3: total_blocks = initial_blocks + additional_blocks
  sorry
}

end Wayne_blocks_l140_140392


namespace athlete_heartbeats_l140_140949

def heart_beats_per_minute : ℕ := 120
def running_pace_minutes_per_mile : ℕ := 6
def race_distance_miles : ℕ := 30
def total_heartbeats : ℕ := 21600

theorem athlete_heartbeats :
  (running_pace_minutes_per_mile * race_distance_miles * heart_beats_per_minute) = total_heartbeats :=
by
  sorry

end athlete_heartbeats_l140_140949


namespace fraction_zero_solution_l140_140226

theorem fraction_zero_solution (x : ℝ) (h₁ : x - 1 = 0) (h₂ : x + 3 ≠ 0) : x = 1 :=
by
  sorry

end fraction_zero_solution_l140_140226


namespace difference_between_two_numbers_l140_140912

theorem difference_between_two_numbers (a : ℕ) (b : ℕ)
  (h1 : a + b = 24300)
  (h2 : b = 100 * a) :
  b - a = 23760 :=
by {
  sorry
}

end difference_between_two_numbers_l140_140912


namespace evaluate_expression_l140_140970

theorem evaluate_expression (a : ℕ) (h : a = 4) : (a^a - a*(a-2)^a)^a = 1358954496 :=
by
  rw [h]  -- Substitute a with 4
  sorry

end evaluate_expression_l140_140970


namespace other_root_of_quadratic_l140_140472

theorem other_root_of_quadratic (a b : ℝ) (h : a ≠ 0) (h_eq : a * 2^2 = b) : 
  ∃ m : ℝ, a * m^2 = b ∧ 2 + m = 0 :=
begin
  use -2,
  split,
  { rw [mul_pow, h_eq, pow_two, mul_assoc, mul_comm 2, ←mul_assoc, mul_comm a, pow_two (-2)],
    sorry },
  { linarith }
end

end other_root_of_quadratic_l140_140472


namespace anna_least_days_l140_140825

theorem anna_least_days (borrow : ℝ) (interest_rate : ℝ) (days : ℕ) :
  (borrow = 20) → (interest_rate = 0.10) → borrow + (borrow * interest_rate * days) ≥ 2 * borrow → days ≥ 10 :=
by
  intros h1 h2 h3
  sorry

end anna_least_days_l140_140825


namespace sequence_general_formula_l140_140451

theorem sequence_general_formula (a : ℕ → ℕ) 
    (h₀ : a 1 = 3) 
    (h : ∀ n : ℕ, a (n + 1) = 2 * a n + 1) : 
    ∀ n : ℕ, a n = 2^(n+1) - 1 :=
by 
  sorry

end sequence_general_formula_l140_140451


namespace pocket_money_calculation_l140_140112

theorem pocket_money_calculation
  (a b c d e : ℝ)
  (h1 : (a + b + c + d + e) / 5 = 2300)
  (h2 : (a + b) / 2 = 3000)
  (h3 : (b + c) / 2 = 2100)
  (h4 : (c + d) / 2 = 2750)
  (h5 : a = b + 800) :
  d = 3900 :=
by
  sorry

end pocket_money_calculation_l140_140112


namespace division_remainder_l140_140789

/-- The remainder when 3572 is divided by 49 is 44. -/
theorem division_remainder :
  3572 % 49 = 44 :=
by
  sorry

end division_remainder_l140_140789


namespace find_number_l140_140969

theorem find_number (x : ℕ) (h : (18 / 100) * x = 90) : x = 500 :=
sorry

end find_number_l140_140969


namespace intersection_of_sets_l140_140992

noncomputable def setA : Set ℝ := { x | (x + 2) / (x - 2) ≤ 0 }
noncomputable def setB : Set ℝ := { x | x ≥ 1 }
noncomputable def expectedSet : Set ℝ := { x | 1 ≤ x ∧ x < 2 }

theorem intersection_of_sets : (setA ∩ setB) = expectedSet := by
  sorry

end intersection_of_sets_l140_140992


namespace mirror_area_correct_l140_140362

noncomputable def width_of_mirror (frame_width : ℕ) (side_width : ℕ) : ℕ :=
  frame_width - 2 * side_width

noncomputable def height_of_mirror (frame_height : ℕ) (side_width : ℕ) : ℕ :=
  frame_height - 2 * side_width

noncomputable def area_of_mirror (frame_width : ℕ) (frame_height : ℕ) (side_width : ℕ) : ℕ :=
  width_of_mirror frame_width side_width * height_of_mirror frame_height side_width

theorem mirror_area_correct :
  area_of_mirror 50 70 7 = 2016 :=
by
  sorry

end mirror_area_correct_l140_140362


namespace half_radius_circle_y_l140_140293

-- Conditions
def circle_x_circumference (C : ℝ) : Prop :=
  C = 20 * Real.pi

def circle_x_and_y_same_area (r R : ℝ) : Prop :=
  Real.pi * r^2 = Real.pi * R^2

-- Problem statement: Prove that half the radius of circle y is 5
theorem half_radius_circle_y (r R : ℝ) (hx : circle_x_circumference (2 * Real.pi * r)) (hy : circle_x_and_y_same_area r R) : R / 2 = 5 :=
by sorry

end half_radius_circle_y_l140_140293


namespace pair_d_are_equal_l140_140822

theorem pair_d_are_equal : -(2 ^ 3) = (-2) ^ 3 :=
by
  -- Detailed proof steps go here, but are omitted for this task.
  sorry

end pair_d_are_equal_l140_140822


namespace find_multiple_l140_140433

theorem find_multiple (n m : ℕ) (h_n : n = 5) (h_eq : m * n - 15 = 2 * n + 10) : m = 7 :=
by
  sorry

end find_multiple_l140_140433


namespace find_positive_integer_pair_l140_140434

theorem find_positive_integer_pair (a b : ℕ) (h : ∀ n : ℕ, n > 0 → ∃ c_n : ℕ, a^n + b^n = c_n^(n + 1)) : a = 2 ∧ b = 2 := 
sorry

end find_positive_integer_pair_l140_140434


namespace tan_sum_identity_l140_140329

theorem tan_sum_identity (theta : Real) (h : Real.tan theta = 1 / 3) :
  Real.tan (theta + Real.pi / 4) = 2 :=
by
  sorry

end tan_sum_identity_l140_140329


namespace trig_identity_l140_140882

theorem trig_identity (α a : ℝ) (h1 : 0 < α) (h2 : α < π / 2)
    (h3 : (Real.tan α) + (1 / (Real.tan α)) = a) : 
    (1 / Real.sin α) + (1 / Real.cos α) = Real.sqrt (a^2 + 2 * a) :=
by
  sorry

end trig_identity_l140_140882


namespace smallest_t_for_circle_covered_l140_140101

theorem smallest_t_for_circle_covered:
  ∃ t, (∀ θ, 0 ≤ θ → θ ≤ t → (∃ r, r = Real.sin θ)) ∧
         (∀ t', (∀ θ, 0 ≤ θ → θ ≤ t' → (∃ r, r = Real.sin θ)) → t' ≥ t) :=
sorry

end smallest_t_for_circle_covered_l140_140101


namespace loss_eq_cost_price_of_x_balls_l140_140493

theorem loss_eq_cost_price_of_x_balls (cp ball_count sp : ℕ) (cp_ball : ℕ) 
  (hc1 : cp_ball = 60) (hc2 : cp = ball_count * cp_ball) (hs : sp = 720) 
  (hb : ball_count = 17) :
  ∃ x : ℕ, (cp - sp = x * cp_ball) ∧ x = 5 :=
by
  sorry

end loss_eq_cost_price_of_x_balls_l140_140493


namespace sum_lent_l140_140799

theorem sum_lent (P : ℝ) (r t : ℝ) (I : ℝ) (h1 : r = 6) (h2 : t = 6) (h3 : I = P - 672) (h4 : I = P * r * t / 100) :
  P = 1050 := by
  sorry

end sum_lent_l140_140799


namespace find_a9_l140_140709

variable (a : ℕ → ℤ)
variable (h1 : a 2 = -3)
variable (h2 : a 3 = -5)
variable (d : ℤ := a 3 - a 2)

theorem find_a9 : a 9 = -17 :=
by
  sorry

end find_a9_l140_140709


namespace investor_wait_time_l140_140951

noncomputable def compound_interest_time (P A r : ℝ) (n : ℕ) : ℝ :=
  (Real.log (A / P)) / (n * Real.log (1 + r / n))

theorem investor_wait_time :
  compound_interest_time 600 661.5 0.10 2 = 1 := 
sorry

end investor_wait_time_l140_140951


namespace parabola_tangent_hyperbola_l140_140407

theorem parabola_tangent_hyperbola (m : ℝ) :
  (∀ x : ℝ, (x^2 + 5)^2 - m * x^2 = 4 → y = x^2 + 5)
  ∧ (∀ y : ℝ, y ≥ 5 → y^2 - m * x^2 = 4) →
  (m = 10 + 2 * Real.sqrt 21 ∨ m = 10 - 2 * Real.sqrt 21) :=
  sorry

end parabola_tangent_hyperbola_l140_140407


namespace reading_time_difference_l140_140795

theorem reading_time_difference 
  (xanthia_reading_speed : ℕ) 
  (molly_reading_speed : ℕ) 
  (book_pages : ℕ) 
  (time_conversion_factor : ℕ)
  (hx : xanthia_reading_speed = 150)
  (hm : molly_reading_speed = 75)
  (hp : book_pages = 300)
  (ht : time_conversion_factor = 60) :
  ((book_pages / molly_reading_speed - book_pages / xanthia_reading_speed) * time_conversion_factor = 120) := 
by
  sorry

end reading_time_difference_l140_140795


namespace find_price_of_fourth_variety_theorem_l140_140326

-- Define the variables and conditions
variables (P1 P2 P3 P4 : ℝ) (Q1 Q2 Q3 Q4 : ℝ) (P_avg : ℝ)

-- Given conditions
def price_of_fourth_variety : Prop :=
  P1 = 126 ∧
  P2 = 135 ∧
  P3 = 156 ∧
  P_avg = 165 ∧
  Q1 / Q2 = 2 / 3 ∧
  Q1 / Q3 = 2 / 4 ∧
  Q1 / Q4 = 2 / 5 ∧
  (P1 * Q1 + P2 * Q2 + P3 * Q3 + P4 * Q4) / (Q1 + Q2 + Q3 + Q4) = P_avg

-- Prove that the price of the fourth variety of tea is Rs. 205.8 per kg
theorem find_price_of_fourth_variety_theorem : price_of_fourth_variety P1 P2 P3 P4 Q1 Q2 Q3 Q4 P_avg → P4 = 205.8 :=
by {
  sorry
}

end find_price_of_fourth_variety_theorem_l140_140326


namespace find_f_neg2014_l140_140208

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x^3 + b * x - 2

theorem find_f_neg2014 (a b : ℝ) (h : f 2014 a b = 3) : f (-2014) a b = -7 :=
by sorry

end find_f_neg2014_l140_140208


namespace nancy_potatoes_l140_140604

theorem nancy_potatoes (sandy_potatoes total_potatoes : ℕ) (h1 : sandy_potatoes = 7) (h2 : total_potatoes = 13) :
    total_potatoes - sandy_potatoes = 6 :=
by
  sorry

end nancy_potatoes_l140_140604


namespace total_votes_l140_140301

theorem total_votes (P R : ℝ) (hP : P = 0.35) (diff : ℝ) (h_diff : diff = 1650) : 
  ∃ V : ℝ, P * V + (P * V + diff) = V ∧ V = 5500 :=
by
  use 5500
  sorry

end total_votes_l140_140301


namespace number_of_undeveloped_sections_l140_140915

def undeveloped_sections (total_area section_area : ℕ) : ℕ :=
  total_area / section_area

theorem number_of_undeveloped_sections :
  undeveloped_sections 7305 2435 = 3 :=
by
  unfold undeveloped_sections
  exact rfl

end number_of_undeveloped_sections_l140_140915


namespace geometric_seq_sum_l140_140457

theorem geometric_seq_sum (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_a1_pos : a 1 > 0)
  (h_a4_7 : a 4 + a 7 = 2)
  (h_a5_6 : a 5 * a 6 = -8) :
  a 1 + a 4 + a 7 + a 10 = -5 := 
sorry

end geometric_seq_sum_l140_140457


namespace largest_possible_a_l140_140840

theorem largest_possible_a (a b c e : ℕ) (h1 : a < 2 * b) (h2 : b < 3 * c) (h3 : c < 5 * e) (h4 : e < 100) : a ≤ 2961 :=
by
  sorry

end largest_possible_a_l140_140840


namespace magnitude_vec_sum_l140_140210

open Real

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
a.1 * b.1 + a.2 * b.2

theorem magnitude_vec_sum
    (a b : ℝ × ℝ)
    (h_angle : ∃ θ, θ = 150 * (π / 180) ∧ cos θ = cos (5 * π / 6))
    (h_norm_a : ‖a‖ = sqrt 3)
    (h_norm_b : ‖b‖ = 2) :
  ‖(2 * a.1 + b.1, 2 * a.2 + b.2)‖ = 2 :=
  by
  sorry

end magnitude_vec_sum_l140_140210


namespace sum_of_first_six_terms_l140_140900

theorem sum_of_first_six_terms 
  (a₁ : ℝ) 
  (r : ℝ) 
  (h_ratio : r = 2) 
  (h_sum_three : a₁ + 2*a₁ + 4*a₁ = 3) 
  : a₁ * (r^6 - 1) / (r - 1) = 27 := 
by {
  sorry
}

end sum_of_first_six_terms_l140_140900


namespace mul_mixed_number_eq_l140_140139

theorem mul_mixed_number_eq :
  99 + 24 / 25 * -5 = -499 - 4 / 5 :=
by
  sorry

end mul_mixed_number_eq_l140_140139


namespace percentage_increase_l140_140761

variables (J T P : ℝ)

def income_conditions (J T P : ℝ) : Prop :=
  (T = 0.5 * J) ∧ (P = 0.8 * J)

theorem percentage_increase (J T P : ℝ) (h : income_conditions J T P) :
  ((P / T) - 1) * 100 = 60 :=
by
  sorry

end percentage_increase_l140_140761


namespace probability_divisible_by_256_l140_140341

theorem probability_divisible_by_256 (n : ℕ) (h : 1 ≤ n ∧ n ≤ 1000) :
  ((n * (n + 1) * (n + 2)) % 256 = 0) → (∃ p : ℚ, p = 0.006 ∧ (∃ k : ℕ, k ≤ 1000 ∧ (n = k))) :=
sorry

end probability_divisible_by_256_l140_140341


namespace no_valid_cross_exists_l140_140932

-- Define a function that calculates the value at position (i, j) 
-- based on the given rules
def table_value (i j : ℕ) : ℕ := 70 * i + j + 1

theorem no_valid_cross_exists : 
  ¬ (∃ x, 1 ≤ x ∧ x ≤ 4900 ∧ (x + x - 70 + x + 70 + x - 1 + x + 1 = 2018)) :=
begin
  -- Start with the assumption that such an x exists
  intro h,
  cases h with x hx,
  cases hx with h1 h,
  cases h with h2 h3,

  -- Simplify the sum of the cross:
  have h4 : 5 * x = 2018,
    calc 5 * x = x + (x - 70) + (x + 70) + (x - 1) + (x + 1) : by ring
           ... = 2018 : h3,

  -- Solve for x
  have h5 : x = 403.6, from (nat.div_eq_of_eq_mul_right zero_lt_five.symm h4.symm),

  -- x must be an integer, contradiction
  exfalso,
  linarith,
end

end no_valid_cross_exists_l140_140932


namespace race_placement_l140_140475

def finished_places (nina zoey sam liam vince : ℕ) : Prop :=
  nina = 12 ∧
  sam = nina + 1 ∧
  zoey = nina - 2 ∧
  liam = zoey - 3 ∧
  vince = liam + 2 ∧
  vince = nina - 3

theorem race_placement (nina zoey sam liam vince : ℕ) :
  finished_places nina zoey sam liam vince →
  nina = 12 →
  sam = 13 →
  zoey = 10 →
  liam = 7 →
  vince = 5 →
  (8 ≠ sam ∧ 8 ≠ nina ∧ 8 ≠ zoey ∧ 8 ≠ liam ∧ 8 ≠ jodi ∧ 8 ≠ vince) := by
  sorry

end race_placement_l140_140475


namespace cos_double_angle_l140_140052

theorem cos_double_angle
  {x : ℝ}
  (h : Real.sin x = -2 / 3) :
  Real.cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_l140_140052


namespace no_real_solutions_for_equation_l140_140758

theorem no_real_solutions_for_equation : ¬ (∃ x : ℝ, x + Real.sqrt (2 * x - 6) = 5) :=
sorry

end no_real_solutions_for_equation_l140_140758


namespace even_function_x_lt_0_l140_140716

noncomputable def f (x : ℝ) : ℝ :=
if h : x >= 0 then 2^x + 1 else 2^(-x) + 1

theorem even_function_x_lt_0 (x : ℝ) (hx : x < 0) : f x = 2^(-x) + 1 :=
by {
  sorry
}

end even_function_x_lt_0_l140_140716


namespace circle_O2_tangent_circle_O2_intersect_l140_140902

-- Condition: The equation of circle O_1 is \(x^2 + (y + 1)^2 = 4\)
def circle_O1 (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4

-- Condition: The center of circle O_2 is \(O_2(2, 1)\)
def center_O2 : (ℝ × ℝ) := (2, 1)

-- Prove the equation of circle O_2 if it is tangent to circle O_1
theorem circle_O2_tangent : 
  ∀ (x y : ℝ), circle_O1 x y → (x - 2)^2 + (y - 1)^2 = 12 - 8 * Real.sqrt 2 :=
sorry

-- Prove the equations of circle O_2 if it intersects circle O_1 and \(|AB| = 2\sqrt{2}\)
theorem circle_O2_intersect :
  ∀ (x y : ℝ), 
  circle_O1 x y → 
  (2 * Real.sqrt 2 = |(x - 2)^2 + (y - 1)^2 - 4| ∨ (x - 2)^2 + (y - 1)^2 = 20) :=
sorry

end circle_O2_tangent_circle_O2_intersect_l140_140902


namespace Yan_ratio_distance_l140_140648

theorem Yan_ratio_distance (w x y : ℕ) (h : w > 0) (h_eq : y/w = x/w + (x + y)/(5 * w)) : x/y = 2/3 := by
  sorry

end Yan_ratio_distance_l140_140648


namespace find_b_l140_140061

theorem find_b (b : ℚ) (h : ∃ c : ℚ, (3 * x + c)^2 = 9 * x^2 + 27 * x + b) : b = 81 / 4 := 
sorry

end find_b_l140_140061


namespace partA_l140_140800

theorem partA (n : ℕ) : 
  1 < (n + 1 / 2) * Real.log (1 + 1 / n) ∧ (n + 1 / 2) * Real.log (1 + 1 / n) < 1 + 1 / (12 * n * (n + 1)) := 
sorry

end partA_l140_140800


namespace sum_in_base_6_l140_140725

theorem sum_in_base_6 (S H E : ℕ) (h1 : S ≠ H) (h2 : H ≠ E) (h3 : S ≠ E)
  (h4 : 0 < S) (h5 : S < 6)
  (h6 : 0 < H) (h7 : H < 6)
  (h8 : 0 < E) (h9 : E < 6)
  (h_addition : (S + H) % 6 = S)
  (h_carry : (E + H) % 6 = E): 
  S + H + E = 13_6 := sorry

end sum_in_base_6_l140_140725


namespace percentage_passed_exam_l140_140878

theorem percentage_passed_exam (total_students failed_students : ℕ) (h_total : total_students = 540) (h_failed : failed_students = 351) :
  (total_students - failed_students) * 100 / total_students = 35 :=
by
  sorry

end percentage_passed_exam_l140_140878


namespace seq_2016_2017_l140_140880

-- Define the sequence a_n
def seq (n : ℕ) : ℚ := sorry

-- Given conditions
axiom a1_cond : seq 1 = 1/2
axiom a2_cond : seq 2 = 1/3
axiom seq_rec : ∀ n : ℕ, seq n * seq (n + 2) = 1

-- The main goal
theorem seq_2016_2017 : seq 2016 + seq 2017 = 7/2 := sorry

end seq_2016_2017_l140_140880


namespace min_value_expression_l140_140488

theorem min_value_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 4) :
  ∃ c : ℝ, (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x * y * z = 4 → 
  (2 * (x / y) + 3 * (y / z) + 4 * (z / x)) ≥ c) ∧ c = 6 :=
by
  sorry

end min_value_expression_l140_140488


namespace smallest_integer_quad_ineq_l140_140641

-- Definition of the condition
def quad_ineq (n : ℤ) := n^2 - 14 * n + 45 > 0

-- Lean 4 statement of the math proof problem
theorem smallest_integer_quad_ineq : ∃ n : ℤ, quad_ineq n ∧ ∀ m : ℤ, quad_ineq m → n ≤ m :=
  by
    existsi 10
    sorry

end smallest_integer_quad_ineq_l140_140641


namespace customers_not_tipping_l140_140311

theorem customers_not_tipping (number_of_customers tip_per_customer total_earned_in_tips : ℕ)
  (h_number : number_of_customers = 7)
  (h_tip : tip_per_customer = 3)
  (h_earned : total_earned_in_tips = 6) :
  number_of_customers - (total_earned_in_tips / tip_per_customer) = 5 :=
by
  sorry

end customers_not_tipping_l140_140311


namespace sum_proper_divisors_81_l140_140287

theorem sum_proper_divisors_81 :
  let proper_divisors : List Nat := [1, 3, 9, 27]
  List.sum proper_divisors = 40 :=
by
  sorry

end sum_proper_divisors_81_l140_140287


namespace flight_duration_l140_140897

noncomputable def departure_time_pst := 9 * 60 + 15 -- in minutes
noncomputable def arrival_time_est := 17 * 60 + 40 -- in minutes
noncomputable def time_difference := 3 * 60 -- in minutes

theorem flight_duration (h m : ℕ) 
  (h_cond : 0 < m ∧ m < 60) 
  (total_flight_time : (arrival_time_est - (departure_time_pst + time_difference)) = h * 60 + m) : 
  h + m = 30 :=
sorry

end flight_duration_l140_140897


namespace Juliska_correct_l140_140236

-- Definitions according to the conditions in a)
def has_three_rum_candy (candies : List String) : Prop :=
  ∀ (selected_triplet : List String), selected_triplet.length = 3 → "rum" ∈ selected_triplet

def has_three_coffee_candy (candies : List String) : Prop :=
  ∀ (selected_triplet : List String), selected_triplet.length = 3 → "coffee" ∈ selected_triplet

-- Proof problem statement
theorem Juliska_correct 
  (candies : List String) 
  (h_rum : has_three_rum_candy candies)
  (h_coffee : has_three_coffee_candy candies) : 
  (∀ (selected_triplet : List String), selected_triplet.length = 3 → "walnut" ∈ selected_triplet) :=
sorry

end Juliska_correct_l140_140236


namespace dan_violet_marbles_l140_140012

def InitMarbles : ℕ := 128
def MarblesGivenMary : ℕ := 24
def MarblesGivenPeter : ℕ := 16
def MarblesReceived : ℕ := 10

def FinalMarbles : ℕ := InitMarbles - MarblesGivenMary - MarblesGivenPeter + MarblesReceived

theorem dan_violet_marbles : FinalMarbles = 98 := 
by 
  sorry

end dan_violet_marbles_l140_140012


namespace problem1_problem2_l140_140295

-- Problem 1: Calculation Proof
theorem problem1 : (3 - Real.pi)^0 - Real.sqrt 4 + 4 * Real.sin (Real.pi * 60 / 180) + |Real.sqrt 3 - 3| = 2 + Real.sqrt 3 :=
by
  sorry

-- Problem 2: Inequality Systems Proof
theorem problem2 (x : ℝ) :
  (5 * (x + 3) > 4 * x + 8) ∧ (x / 6 - 1 < (x - 2) / 3) → x > -2 :=
by
  sorry

end problem1_problem2_l140_140295


namespace train_speed_kmph_l140_140945

theorem train_speed_kmph (len_train : ℝ) (len_platform : ℝ) (time_cross : ℝ) (total_distance : ℝ) (speed_mps : ℝ) (speed_kmph : ℝ) 
  (h1 : len_train = 250) 
  (h2 : len_platform = 150.03) 
  (h3 : time_cross = 20) 
  (h4 : total_distance = len_train + len_platform) 
  (h5 : speed_mps = total_distance / time_cross) 
  (h6 : speed_kmph = speed_mps * 3.6) : 
  speed_kmph = 72.0054 := 
by 
  -- This is where the proof would go
  sorry

end train_speed_kmph_l140_140945


namespace fourth_number_pascal_row_l140_140234

theorem fourth_number_pascal_row : (Nat.choose 12 3) = 220 := sorry

end fourth_number_pascal_row_l140_140234


namespace smallest_y_l140_140127

theorem smallest_y (y : ℤ) (h : y < 3 * y - 15) : y = 8 :=
  sorry

end smallest_y_l140_140127


namespace mary_james_not_adjacent_l140_140746

open Finset

-- Define the set of chairs and relevant probabilities
def chairs := range 10

-- Define the event they sit next to each other
def adjacent_pairs : Finset (ℕ × ℕ) := 
  Finset.filter (λ (p : ℕ × ℕ), abs (p.1 - p.2) = 1) (chairs.product chairs)

def total_pairs : Finset (ℕ × ℕ) := 
  chairs.product chairs \ (chairs.product {x | x = 10 - 1})

-- Calculate the probability that they sit next to each other
def prob_adjacent : ℚ :=
  adjacent_pairs.card / total_pairs.card

noncomputable def prob_not_adjacent : ℚ :=
  1 - prob_adjacent

theorem mary_james_not_adjacent :
  prob_not_adjacent = 4 / 5 :=
by
  sorry

end mary_james_not_adjacent_l140_140746


namespace range_of_t_l140_140459

theorem range_of_t (t : ℝ) (h : ∃ x : ℝ, x ∈ Set.Iic t ∧ (x^2 - 4*x + t ≤ 0)) : 0 ≤ t ∧ t ≤ 4 :=
sorry

end range_of_t_l140_140459


namespace percentage_area_covered_by_pentagons_l140_140474

theorem percentage_area_covered_by_pentagons :
  ∀ (a : ℝ), (∃ (large_square_area small_square_area pentagon_area : ℝ),
    large_square_area = 16 * a^2 ∧
    small_square_area = a^2 ∧
    pentagon_area = 10 * small_square_area ∧
    (pentagon_area / large_square_area) * 100 = 62.5) :=
sorry

end percentage_area_covered_by_pentagons_l140_140474


namespace right_triangle_area_l140_140244

theorem right_triangle_area (a b c p S : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : a^2 + b^2 = c^2)
  (h4 : p = (a + b + c) / 2) (h5 : S = a * b / 2) :
  p * (p - c) = S ∧ (p - a) * (p - b) = S :=
sorry

end right_triangle_area_l140_140244


namespace correct_order_l140_140425

noncomputable def f : ℝ → ℝ := sorry

axiom periodic : ∀ x : ℝ, f (x + 4) = f x
axiom increasing : ∀ (x₁ x₂ : ℝ), (0 ≤ x₁ ∧ x₁ < 2) → (0 ≤ x₂ ∧ x₂ ≤ 2) → x₁ < x₂ → f x₁ < f x₂
axiom symmetric : ∀ x : ℝ, f (x + 2) = f (2 - x)

theorem correct_order : f 4.5 < f 7 ∧ f 7 < f 6.5 :=
by
  sorry

end correct_order_l140_140425


namespace expand_polynomial_l140_140177

theorem expand_polynomial (t : ℝ) :
  (3 * t^3 - 2 * t^2 + t - 4) * (2 * t^2 - t + 3) = 6 * t^5 - 7 * t^4 + 5 * t^3 - 15 * t^2 + 7 * t - 12 :=
by sorry

end expand_polynomial_l140_140177


namespace waiter_tables_l140_140546

theorem waiter_tables (initial_customers remaining_customers people_per_table : ℕ) (hc : initial_customers = 44) (hl : remaining_customers = initial_customers - 12) (ht : people_per_table = 8) :
  remaining_customers / people_per_table = 4 :=
by {
  rw [hc, hl, ht],
  exact rfl,
}

end waiter_tables_l140_140546


namespace cost_of_candy_bar_l140_140424

theorem cost_of_candy_bar (t c b : ℕ) (h1 : t = 13) (h2 : c = 6) (h3 : t = b + c) : b = 7 := 
by
  sorry

end cost_of_candy_bar_l140_140424


namespace train_speed_correct_l140_140933

-- Definitions based on the conditions in a)
def train_length_meters : ℝ := 160
def time_seconds : ℝ := 4

-- Correct answer identified in b)
def expected_speed_kmh : ℝ := 144

-- Proof statement verifying that speed computed from the conditions equals the expected speed
theorem train_speed_correct :
  train_length_meters / 1000 / (time_seconds / 3600) = expected_speed_kmh :=
by
  sorry

end train_speed_correct_l140_140933


namespace boat_speed_in_still_water_l140_140809

theorem boat_speed_in_still_water
  (V_s : ℝ) (t : ℝ) (d : ℝ) (V_b : ℝ)
  (h_stream_speed : V_s = 4)
  (h_travel_time : t = 7)
  (h_distance : d = 196)
  (h_downstream_speed : d / t = V_b + V_s) :
  V_b = 24 :=
by
  sorry

end boat_speed_in_still_water_l140_140809


namespace number_division_l140_140523

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l140_140523


namespace total_handshakes_l140_140550

def people := 40
def groupA := 25
def groupB := 15
def knownByGroupB (x : ℕ) : ℕ := 5
def interactionsWithinGroupB : ℕ := 105
def interactionsBetweenGroups : ℕ := 75

theorem total_handshakes : (groupB * knownByGroupB 0) + interactionsWithinGroupB = 180 :=
by
  sorry

end total_handshakes_l140_140550


namespace reciprocal_relationship_l140_140193

theorem reciprocal_relationship (a b : ℝ) (h₁ : a = 2 - Real.sqrt 3) (h₂ : b = Real.sqrt 3 + 2) : 
  a * b = 1 :=
by
  rw [h₁, h₂]
  sorry

end reciprocal_relationship_l140_140193


namespace fastest_slowest_difference_l140_140636

-- Given conditions
def length_A : ℕ := 8
def length_B : ℕ := 10
def length_C : ℕ := 6
def section_length : ℕ := 2

def sections_A : ℕ := 24
def sections_B : ℕ := 25
def sections_C : ℕ := 27

-- Calculate number of cuts required
def cuts_per_segment_A := length_A / section_length - 1
def cuts_per_segment_B := length_B / section_length - 1
def cuts_per_segment_C := length_C / section_length - 1

-- Calculate total number of cuts
def total_cuts_A := cuts_per_segment_A * (sections_A / (length_A / section_length))
def total_cuts_B := cuts_per_segment_B * (sections_B / (length_B / section_length))
def total_cuts_C := cuts_per_segment_C * (sections_C / (length_C / section_length))

-- Finding min and max cuts
def max_cuts := max total_cuts_A (max total_cuts_B total_cuts_C)
def min_cuts := min total_cuts_A (min total_cuts_B total_cuts_C)

-- Prove that the difference between max cuts and min cuts is 2
theorem fastest_slowest_difference :
  max_cuts - min_cuts = 2 := by
  sorry

end fastest_slowest_difference_l140_140636


namespace number_of_real_roots_l140_140024

theorem number_of_real_roots (a : ℝ) :
  (|a| < (2 * Real.sqrt 3 / 9) → ∃ x y z : ℝ, x^3 - x - a = 0 ∧ y^3 - y - a = 0 ∧ z^3 - z - a = 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x) ∧
  (|a| = (2 * Real.sqrt 3 / 9) → ∃ x y : ℝ, x^3 - x - a = 0 ∧ y^3 - y - a = 0 ∧ x = y) ∧
  (|a| > (2 * Real.sqrt 3 / 9) → ∃ x : ℝ, x^3 - x - a = 0 ∧ ∀ y : ℝ, y^3 - y - a ≠ 0 ∨ y = x) :=
sorry

end number_of_real_roots_l140_140024


namespace max_rectangle_area_l140_140773

theorem max_rectangle_area (a b : ℝ) (h : 2 * a + 2 * b = 60) :
  a * b ≤ 225 :=
by
  sorry

end max_rectangle_area_l140_140773


namespace circumscribed_triangle_BCK_tangent_to_AB_l140_140542

-- Define objects in Lean
variables {A B C D K : Point}

-- Definitions of geometric conditions
def is_cyclic_trapezoid (ABCD : Trapezoid) : Prop :=
  ∃ (circle : Circle), circle.passes_through A ∧ circle.passes_through B ∧ circle.passes_through D

def intersecting_point (circ : Circle) (CD : Line) : Prop :=
  circ.intersects CD ∧ K ∈ circ ∧ K ∈ CD

def parallel_sides (AD BC : Line) : Prop :=
  AD.parallel BC

-- Main theorem statement
theorem circumscribed_triangle_BCK_tangent_to_AB
  (h1 : is_cyclic_trapezoid ABCD)
  (h2 : intersecting_point circle CD)
  (h3 : parallel_sides AD BC) :
  circle_circumscribed_around_triangle B C K.tangent_to AB :=
sorry

end circumscribed_triangle_BCK_tangent_to_AB_l140_140542


namespace quadratic_rewrite_correct_a_b_c_l140_140509

noncomputable def quadratic_rewrite (x : ℝ) : ℝ := -6*x^2 + 36*x + 216

theorem quadratic_rewrite_correct_a_b_c :
  ∃ a b c : ℝ, quadratic_rewrite x = a * (x + b)^2 + c ∧ a + b + c = 261 :=
by
  sorry

end quadratic_rewrite_correct_a_b_c_l140_140509


namespace arthur_walk_distance_l140_140548

def blocks_east : ℕ := 8
def blocks_north : ℕ := 15
def block_length : ℚ := 1 / 4

theorem arthur_walk_distance :
  (blocks_east + blocks_north) * block_length = 23 * (1 / 4) := by
  sorry

end arthur_walk_distance_l140_140548


namespace total_legs_l140_140255

-- Define the number of each type of animal
def num_horses : ℕ := 2
def num_dogs : ℕ := 5
def num_cats : ℕ := 7
def num_turtles : ℕ := 3
def num_goat : ℕ := 1

-- Define the number of legs per animal
def legs_per_animal : ℕ := 4

-- Define the total number of legs for each type of animal
def horse_legs : ℕ := num_horses * legs_per_animal
def dog_legs : ℕ := num_dogs * legs_per_animal
def cat_legs : ℕ := num_cats * legs_per_animal
def turtle_legs : ℕ := num_turtles * legs_per_animal
def goat_legs : ℕ := num_goat * legs_per_animal

-- Define the problem statement
theorem total_legs : horse_legs + dog_legs + cat_legs + turtle_legs + goat_legs = 72 := by
  -- Sum up all the leg counts
  sorry

end total_legs_l140_140255


namespace avg_rest_students_l140_140227

/- Definitions based on conditions -/
def total_students : ℕ := 28
def students_scored_95 : ℕ := 4
def students_scored_0 : ℕ := 3
def avg_whole_class : ℚ := 47.32142857142857
def total_marks_95 : ℚ := students_scored_95 * 95
def total_marks_0 : ℚ := students_scored_0 * 0
def marks_whole_class : ℚ := total_students * avg_whole_class
def rest_students : ℕ := total_students - students_scored_95 - students_scored_0

/- Theorem to prove the average of the rest students given the conditions -/
theorem avg_rest_students : (total_marks_95 + total_marks_0 + rest_students * 45) = marks_whole_class :=
by
  sorry

end avg_rest_students_l140_140227


namespace rings_on_fingers_arrangement_l140_140724

-- Definitions based on the conditions
def rings : ℕ := 5
def fingers : ℕ := 5

-- Theorem statement
theorem rings_on_fingers_arrangement : (fingers ^ rings) = 5 ^ 5 := by
  sorry  -- Proof skipped

end rings_on_fingers_arrangement_l140_140724


namespace carson_clawed_39_times_l140_140006

def wombats_count := 9
def wombat_claws_per := 4
def rheas_count := 3
def rhea_claws_per := 1

def wombat_total_claws := wombats_count * wombat_claws_per
def rhea_total_claws := rheas_count * rhea_claws_per
def total_claws := wombat_total_claws + rhea_total_claws

theorem carson_clawed_39_times : total_claws = 39 :=
  by sorry

end carson_clawed_39_times_l140_140006


namespace long_diagonal_length_l140_140818

-- Define the lengths of the rhombus sides and diagonals
variables (a b : ℝ) (s : ℝ)
variable (side_length : ℝ)
variable (short_diagonal : ℝ)
variable (long_diagonal : ℝ)

-- Given conditions
def rhombus (side_length: ℝ) (short_diagonal: ℝ) : Prop :=
  side_length = 51 ∧ short_diagonal = 48

-- To prove: length longer diagonal is 90 units
theorem long_diagonal_length (side_length: ℝ) (short_diagonal: ℝ) (long_diagonal: ℝ) :
  rhombus side_length short_diagonal →
  long_diagonal = 90 :=
by
  sorry 

end long_diagonal_length_l140_140818


namespace evaluate_f_difference_l140_140248

def f (x : ℝ) : ℝ := x^4 + x^2 + 3*x^3 + 5*x

theorem evaluate_f_difference : f 5 - f (-5) = 800 := by
  sorry

end evaluate_f_difference_l140_140248


namespace smallest_sum_zero_l140_140668

theorem smallest_sum_zero : ∃ x ∈ ({-1, -2, 1, 2} : Set ℤ), ∀ y ∈ ({-1, -2, 1, 2} : Set ℤ), x + 0 ≤ y + 0 :=
sorry

end smallest_sum_zero_l140_140668


namespace triangle_inequality_l140_140478

theorem triangle_inequality {A B C : ℝ} {n : ℕ} (h : B = n * C) (hA : A + B + C = π) :
  B ≤ n * C :=
by
  sorry

end triangle_inequality_l140_140478


namespace train_passes_bridge_in_128_seconds_l140_140946

/-- A proof problem regarding a train passing a bridge -/
theorem train_passes_bridge_in_128_seconds 
  (train_length : ℕ) 
  (train_speed_kmh : ℕ) 
  (bridge_length : ℕ) 
  (conversion_factor : ℚ) 
  (time_to_pass : ℚ) :
  train_length = 1200 →
  train_speed_kmh = 90 →
  bridge_length = 2000 →
  conversion_factor = (5 / 18) →
  time_to_pass = (train_length + bridge_length) / (train_speed_kmh * conversion_factor) →
  time_to_pass = 128 := 
by
  -- We are skipping the proof itself
  sorry

end train_passes_bridge_in_128_seconds_l140_140946


namespace polynomial_value_l140_140224

theorem polynomial_value :
  let a := -4
  let b := 23
  let c := -17
  let d := 10
  5 * a + 3 * b + 2 * c + d = 25 :=
by
  let a := -4
  let b := 23
  let c := -17
  let d := 10
  sorry

end polynomial_value_l140_140224


namespace time_to_cross_platform_l140_140539

-- Definitions of the given conditions
def train_length : ℝ := 900
def time_to_cross_pole : ℝ := 18
def platform_length : ℝ := 1050

-- Goal statement in Lean 4 format
theorem time_to_cross_platform : 
  let speed := train_length / time_to_cross_pole;
  let total_distance := train_length + platform_length;
  let time := total_distance / speed;
  time = 39 := 
by
  sorry

end time_to_cross_platform_l140_140539


namespace Isabella_paint_area_l140_140235

def bedroom1_length : ℕ := 14
def bedroom1_width : ℕ := 11
def bedroom1_height : ℕ := 9

def bedroom2_length : ℕ := 13
def bedroom2_width : ℕ := 12
def bedroom2_height : ℕ := 9

def unpaintable_area_per_bedroom : ℕ := 70

theorem Isabella_paint_area :
  let wall_area (length width height : ℕ) := 2 * (length * height) + 2 * (width * height)
  let paintable_area (length width height : ℕ) := wall_area length width height - unpaintable_area_per_bedroom
  paintable_area bedroom1_length bedroom1_width bedroom1_height +
  paintable_area bedroom1_length bedroom1_width bedroom1_height +
  paintable_area bedroom2_length bedroom2_width bedroom2_height +
  paintable_area bedroom2_length bedroom2_width bedroom2_height =
  1520 := 
by
  sorry

end Isabella_paint_area_l140_140235


namespace solve_for_x_l140_140701

theorem solve_for_x (x : ℝ) (h : (2 + x) / (4 + x) = (3 + x) / (7 + x)) : x = -1 :=
by {
  sorry
}

end solve_for_x_l140_140701


namespace sparrow_population_decline_l140_140971

theorem sparrow_population_decline {P : ℕ} (initial_year : ℕ) (initial_population : ℕ) (decrease_by_half : ∀ year, year ≥ initial_year →  init_population * (1 / (2 ^ (year - initial_year))) < init_population / 20) :
  ∃ year, year ≥ initial_year + 5 ∧ init_population * (1 / (2 ^ (year - initial_year))) < init_population / 20 :=
by
  sorry

end sparrow_population_decline_l140_140971


namespace intersection_of_A_and_B_l140_140198

def A : Set ℤ := {1, 2, -3}
def B : Set ℤ := {1, -4, 5}

theorem intersection_of_A_and_B : A ∩ B = {1} :=
by sorry

end intersection_of_A_and_B_l140_140198


namespace dogs_with_pointy_ears_l140_140363

theorem dogs_with_pointy_ears (total_dogs with_spots with_pointy_ears: ℕ) 
  (h1: with_spots = total_dogs / 2)
  (h2: total_dogs = 30) :
  with_pointy_ears = total_dogs / 5 :=
by
  sorry

end dogs_with_pointy_ears_l140_140363


namespace distinct_paper_count_l140_140632

theorem distinct_paper_count (n : ℕ) :
  let sides := 4  -- 4 rotations and 4 reflections
  let identity_fixed := n^25 
  let rotation_90_fixed := n^7
  let rotation_270_fixed := n^7
  let rotation_180_fixed := n^13
  let reflection_fixed := n^15
  (1 / 8) * (identity_fixed + 4 * reflection_fixed + rotation_180_fixed + 2 * rotation_90_fixed) 
  = (1 / 8) * (n^25 + 4 * n^15 + n^13 + 2 * n^7) :=
  by 
    sorry

end distinct_paper_count_l140_140632


namespace smallest_positive_period_l140_140558

def det2x2 (a1 a2 a3 a4 : ℝ) : ℝ :=
  a1 * a4 - a2 * a3

def f (x : ℝ) : ℝ :=
  det2x2 (Real.sin x) (-1) 1 (Real.cos x)

theorem smallest_positive_period :
  ∀ x, f (x + π) = f x := by
  sorry

end smallest_positive_period_l140_140558


namespace domain_of_sqrt_function_l140_140378

noncomputable def domain_of_function : Set ℝ :=
  {x : ℝ | 3 - 2 * x - x^2 ≥ 0}

theorem domain_of_sqrt_function : domain_of_function = {x : ℝ | -3 ≤ x ∧ x ≤ 1} :=
by
  sorry

end domain_of_sqrt_function_l140_140378


namespace sum_of_first_and_third_is_68_l140_140783

theorem sum_of_first_and_third_is_68
  (A B C : ℕ)
  (h1 : A + B + C = 98)
  (h2 : A * 3 = B * 2)  -- implying A / B = 2 / 3
  (h3 : B * 8 = C * 5)  -- implying B / C = 5 / 8
  (h4 : B = 30) :
  A + C = 68 :=
sorry

end sum_of_first_and_third_is_68_l140_140783


namespace cos_double_angle_l140_140053

theorem cos_double_angle
  {x : ℝ}
  (h : Real.sin x = -2 / 3) :
  Real.cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_l140_140053


namespace pizzas_served_today_l140_140157

theorem pizzas_served_today (lunch_pizzas : ℕ) (dinner_pizzas : ℕ) (h1 : lunch_pizzas = 9) (h2 : dinner_pizzas = 6) : lunch_pizzas + dinner_pizzas = 15 :=
by sorry

end pizzas_served_today_l140_140157


namespace range_of_a_l140_140723

-- Definitions of the sets U and A
def U := {x : ℝ | 0 < x ∧ x < 9}
def A (a : ℝ) := {x : ℝ | 1 < x ∧ x < a}

-- Theorem stating the range of a
theorem range_of_a (a : ℝ) (H_non_empty : A a ≠ ∅) (H_not_subset : ¬ ∀ x, x ∈ A a → x ∈ U) : 
  1 < a ∧ a ≤ 9 :=
sorry

end range_of_a_l140_140723


namespace price_of_orange_is_60_l140_140549

-- Definitions from the conditions
def price_of_apple : ℕ := 40 -- The price of each apple is 40 cents
def total_fruits : ℕ := 10 -- Mary selects a total of 10 apples and oranges
def avg_price_initial : ℕ := 48 -- The average price of the 10 pieces of fruit is 48 cents
def put_back_oranges : ℕ := 2 -- Mary puts back 2 oranges
def avg_price_remaining : ℕ := 45 -- The average price of the remaining fruits is 45 cents

-- Variable definition for the price of an orange which will be solved for
variable (price_of_orange : ℕ)

-- Theorem: proving the price of each orange is 60 cents given the conditions
theorem price_of_orange_is_60 : 
  (∀ a o : ℕ, a + o = total_fruits →
  40 * a + price_of_orange * o = total_fruits * avg_price_initial →
  40 * a + price_of_orange * (o - put_back_oranges) = (total_fruits - put_back_oranges) * avg_price_remaining)
  → price_of_orange = 60 :=
by
  -- Proof is omitted
  sorry

end price_of_orange_is_60_l140_140549


namespace find_number_l140_140531

theorem find_number (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 := 
sorry

end find_number_l140_140531


namespace magician_draws_two_cards_l140_140772

-- Define the total number of unique cards
def total_cards : ℕ := 15^2

-- Define the number of duplicate cards
def duplicate_cards : ℕ := 15

-- Define the number of ways to choose 2 cards from the duplicate cards
def choose_two_duplicates : ℕ := Nat.choose 15 2

-- Define the number of ways to choose 1 duplicate card and 1 non-duplicate card
def choose_one_duplicate_one_nonduplicate : ℕ := (15 * (total_cards - 15 - 14 - 14))

-- The main theorem to prove
theorem magician_draws_two_cards : choose_two_duplicates + choose_one_duplicate_one_nonduplicate = 2835 := by
  sorry

end magician_draws_two_cards_l140_140772


namespace angle_measure_l140_140376

theorem angle_measure (x : ℝ) (h1 : x + 3 * x^2 + 10 = 90) : x = 5 :=
by
  sorry

end angle_measure_l140_140376


namespace arctan_sum_l140_140693

theorem arctan_sum : 
  Real.arctan (1/2) + Real.arctan (1/5) + Real.arctan (1/8) = Real.pi / 4 := 
by 
  sorry

end arctan_sum_l140_140693


namespace negation_proposition_l140_140506

theorem negation_proposition:
  ¬(∃ x : ℝ, x^2 - x + 1 > 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≤ 0 :=
by
  sorry -- Proof not required as per instructions

end negation_proposition_l140_140506


namespace binomial_alternating_sum_eq_zero_l140_140430

theorem binomial_alternating_sum_eq_zero :
  (∑ k in Finset.range 51, (-1)^k * Nat.choose 50 k) = 0 :=
by
  sorry

end binomial_alternating_sum_eq_zero_l140_140430


namespace diamond_calculation_l140_140172

def diamond (a b : ℚ) : ℚ := (a - b) / (1 + a * b)

theorem diamond_calculation : diamond 1 (diamond 2 (diamond 3 (diamond 4 5))) = 87 / 59 :=
by
  sorry

end diamond_calculation_l140_140172


namespace ratio_problem_l140_140040

theorem ratio_problem
  (w x y z : ℝ)
  (h1 : w / x = 1 / 3)
  (h2 : w / y = 2 / 3)
  (h3 : w / z = 3 / 5) :
  (x + y) / z = 27 / 10 :=
by
  sorry

end ratio_problem_l140_140040


namespace num_unique_triangle_areas_correct_l140_140559

noncomputable def num_unique_triangle_areas : ℕ :=
  let A := 0
  let B := 1
  let C := 3
  let D := 6
  let E := 0
  let F := 2
  let base_lengths := [1, 2, 3, 5, 6]
  (base_lengths.eraseDups).length

theorem num_unique_triangle_areas_correct : num_unique_triangle_areas = 5 :=
  by sorry

end num_unique_triangle_areas_correct_l140_140559


namespace perpendicular_lines_parallel_lines_l140_140993

-- Define the lines l1 and l2 in terms of a
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + 6 = 0

def line2 (a : ℝ) (x y : ℝ) : Prop :=
  x + (a - 1) * y + a ^ 2 - 1 = 0

-- Define the perpendicular condition
def perp (a : ℝ) : Prop :=
  a * 1 + 2 * (a - 1) = 0

-- Define the parallel condition
def parallel (a : ℝ) : Prop :=
  a / 1 = 2 / (a - 1)

-- Theorem for perpendicular lines
theorem perpendicular_lines (a : ℝ) : perp a → a = 2 / 3 := by
  intro h
  sorry

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) : parallel a → a = -1 := by
  intro h
  sorry

end perpendicular_lines_parallel_lines_l140_140993


namespace max_projection_area_of_tetrahedron_l140_140667

theorem max_projection_area_of_tetrahedron (a : ℝ) (h1 : a > 0) :
  ∃ (A : ℝ), (A = a^2 / 2) :=
by
  sorry

end max_projection_area_of_tetrahedron_l140_140667


namespace polynomial_division_l140_140324

-- Define the polynomials P and D
noncomputable def P : Polynomial ℤ := 5 * Polynomial.X ^ 4 - 3 * Polynomial.X ^ 3 + 7 * Polynomial.X ^ 2 - 9 * Polynomial.X + 12
noncomputable def D : Polynomial ℤ := Polynomial.X - 3
noncomputable def Q : Polynomial ℤ := 5 * Polynomial.X ^ 3 + 12 * Polynomial.X ^ 2 + 43 * Polynomial.X + 120
def R : ℤ := 372

-- State the theorem
theorem polynomial_division :
  P = D * Q + Polynomial.C R := 
sorry

end polynomial_division_l140_140324


namespace rate_of_mangoes_is_60_l140_140044

-- Define the conditions
def kg_grapes : ℕ := 8
def rate_per_kg_grapes : ℕ := 70
def kg_mangoes : ℕ := 9
def total_paid : ℕ := 1100

-- Define the cost of grapes and total cost
def cost_of_grapes : ℕ := kg_grapes * rate_per_kg_grapes
def cost_of_mangoes : ℕ := total_paid - cost_of_grapes
def rate_per_kg_mangoes : ℕ := cost_of_mangoes / kg_mangoes

-- Prove that the rate of mangoes per kg is 60
theorem rate_of_mangoes_is_60 : rate_per_kg_mangoes = 60 := by
  -- Here we would provide the proof
  sorry

end rate_of_mangoes_is_60_l140_140044


namespace at_least_one_basketball_selected_l140_140564

theorem at_least_one_basketball_selected (balls : Finset ℕ) (basketballs : Finset ℕ) (volleyballs : Finset ℕ) :
  basketballs.card = 6 → volleyballs.card = 2 → balls ⊆ (basketballs ∪ volleyballs) →
  balls.card = 3 → ∃ b ∈ balls, b ∈ basketballs :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end at_least_one_basketball_selected_l140_140564


namespace brendan_cuts_yards_l140_140168

theorem brendan_cuts_yards (x : ℝ) (h : 7 * 1.5 * x = 84) : x = 8 :=
sorry

end brendan_cuts_yards_l140_140168


namespace number_of_correct_statements_l140_140669

-- Define the statements
def statement_1 : Prop := ∀ (a : ℚ), |a| < |0| → a = 0
def statement_2 : Prop := ∃ (b : ℚ), ∀ (c : ℚ), b < 0 ∧ b ≥ c → c = b
def statement_3 : Prop := -4^6 = (-4) * (-4) * (-4) * (-4) * (-4) * (-4)
def statement_4 : Prop := ∀ (a b : ℚ), a + b = 0 → a ≠ 0 → b ≠ 0 → (a / b = -1)
def statement_5 : Prop := ∀ (c : ℚ), (0 / c = 0 ↔ c ≠ 0)

-- Define the overall proof problem
theorem number_of_correct_statements : (statement_1 ∧ statement_4) ∧ ¬(statement_2 ∨ statement_3 ∨ statement_5) :=
by
  sorry

end number_of_correct_statements_l140_140669


namespace odd_function_condition_l140_140888

noncomputable def f (A ω φ : ℝ) (x : ℝ) : ℝ :=
  A * Real.sin (ω * x + φ)

theorem odd_function_condition (A ω : ℝ) (hA : 0 < A) (hω : 0 < ω) (φ : ℝ) :
  (f A ω φ 0 = 0) ↔ (f A ω φ) = fun x => -f A ω φ (-x) := 
by
  sorry

end odd_function_condition_l140_140888


namespace variance_of_scores_l140_140409

-- Define the student's scores
def scores : List ℕ := [130, 125, 126, 126, 128]

-- Define a function to calculate the mean
def mean (l : List ℕ) : ℕ :=
  l.sum / l.length

-- Define a function to calculate the variance
def variance (l : List ℕ) : ℕ :=
  let avg := mean l
  (l.map (λ x => (x - avg) * (x - avg))).sum / l.length

-- The proof statement (no proof provided, use sorry)
theorem variance_of_scores : variance scores = 3 := by sorry

end variance_of_scores_l140_140409


namespace edge_length_increase_l140_140521

theorem edge_length_increase (e e' : ℝ) (A : ℝ) (hA : ∀ e, A = 6 * e^2)
  (hA' : 2.25 * A = 6 * e'^2) :
  (e' - e) / e * 100 = 50 :=
by
  sorry

end edge_length_increase_l140_140521


namespace area_of_right_angled_isosceles_triangle_l140_140911

-- Definitions
variables {x y : ℝ}
def is_right_angled_isosceles (x y : ℝ) : Prop := y^2 = 2 * x^2
def sum_of_square_areas (x y : ℝ) : Prop := x^2 + x^2 + y^2 = 72

-- Theorem
theorem area_of_right_angled_isosceles_triangle (x y : ℝ) 
  (h1 : is_right_angled_isosceles x y) 
  (h2 : sum_of_square_areas x y) : 
  1/2 * x^2 = 9 :=
sorry

end area_of_right_angled_isosceles_triangle_l140_140911


namespace alexis_suit_coat_expense_l140_140411

theorem alexis_suit_coat_expense :
  let budget := 200
  let shirt_cost := 30
  let pants_cost := 46
  let socks_cost := 11
  let belt_cost := 18
  let shoes_cost := 41
  let leftover := 16
  let other_expenses := shirt_cost + pants_cost + socks_cost + belt_cost + shoes_cost
  budget - leftover - other_expenses = 38 := 
by
  let budget := 200
  let shirt_cost := 30
  let pants_cost := 46
  let socks_cost := 11
  let belt_cost := 18
  let shoes_cost := 41
  let leftover := 16
  let other_expenses := shirt_cost + pants_cost + socks_cost + belt_cost + shoes_cost
  sorry

end alexis_suit_coat_expense_l140_140411


namespace zero_in_interval_l140_140913

noncomputable def f (x : ℝ) : ℝ := 2 * x - 8 + Real.logb 3 x

theorem zero_in_interval : 
  (0 < 3) ∧ (3 < 4) → (f 3 < 0) ∧ (f 4 > 0) → ∃ x, 3 < x ∧ x < 4 ∧ f x = 0 :=
by
  intro h1 h2
  obtain ⟨h3, h4⟩ := h2
  sorry

end zero_in_interval_l140_140913


namespace determine_range_of_x_l140_140064

theorem determine_range_of_x (x : ℝ) (h₁ : 1/x < 3) (h₂ : 1/x > -2) : x > 1/3 ∨ x < -1/2 :=
sorry

end determine_range_of_x_l140_140064


namespace diamond_19_98_l140_140452

variable {R : Type} [LinearOrderedField R]

noncomputable def diamond (x y : R) : R := sorry

axiom diamond_axiom1 : ∀ (x y : R) (hx : 0 < x) (hy : 0 < y), diamond (x * y) y = x * (diamond y y)

axiom diamond_axiom2 : ∀ (x : R) (hx : 0 < x), diamond (diamond x 1) x = diamond x 1

axiom diamond_axiom3 : diamond 1 1 = 1

theorem diamond_19_98 : diamond (19 : R) (98 : R) = 19 := 
sorry

end diamond_19_98_l140_140452


namespace trigonometric_identity_l140_140418

theorem trigonometric_identity : 
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = Real.csc (20 * Real.pi / 180) := 
by
  sorry

end trigonometric_identity_l140_140418


namespace line_passes_through_fixed_point_l140_140381

theorem line_passes_through_fixed_point (k : ℝ) : ∀ x y : ℝ, (y - 1 = k * (x + 2)) → (x = -2 ∧ y = 1) :=
by
  intro x y h
  sorry

end line_passes_through_fixed_point_l140_140381


namespace max_min_value_function_l140_140695

noncomputable def given_function (x : ℝ) : ℝ :=
  (Real.sin x) ^ 2 + Real.cos x + 1

theorem max_min_value_function :
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) → given_function x ≤ 9 / 4) ∧ 
  (∃ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), given_function x = 9 / 4) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) → given_function x ≥ 2) ∧ 
  (∃ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), given_function x = 2) := by
  sorry

end max_min_value_function_l140_140695


namespace fraction_division_l140_140832

-- Define the fractions and the operation result.
def complex_fraction := 5 / (8 / 15)
def result := 75 / 8

-- State the theorem indicating that these should be equal.
theorem fraction_division :
  complex_fraction = result :=
  by
  sorry

end fraction_division_l140_140832


namespace larger_integer_l140_140282

theorem larger_integer (x y : ℕ) (h_diff : y - x = 8) (h_prod : x * y = 272) : y = 20 :=
by
  sorry

end larger_integer_l140_140282


namespace smallest_integral_k_no_real_roots_l140_140791

theorem smallest_integral_k_no_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, 2 * x * (k * x - 4) - x^2 + 6 ≠ 0) ∧ 
           (∀ j : ℤ, j < k → (∃ x : ℝ, 2 * x * (j * x - 4) - x^2 + 6 = 0)) ∧
           k = 2 :=
by sorry

end smallest_integral_k_no_real_roots_l140_140791


namespace animal_population_l140_140627

theorem animal_population
  (number_of_lions : ℕ)
  (number_of_leopards : ℕ)
  (number_of_elephants : ℕ)
  (h1 : number_of_lions = 200)
  (h2 : number_of_lions = 2 * number_of_leopards)
  (h3 : number_of_elephants = (number_of_lions + number_of_leopards) / 2) :
  number_of_lions + number_of_leopards + number_of_elephants = 450 :=
sorry

end animal_population_l140_140627


namespace increasing_or_decreasing_subseq_l140_140707

theorem increasing_or_decreasing_subseq {m n : ℕ} (a : Fin (m * n + 1) → ℝ) :
  ∃ (idx_incr : Fin (m + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (idx_incr i) < a (idx_incr j)) ∨ 
  ∃ (idx_decr : Fin (n + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (idx_decr i) > a (idx_decr j)) :=
by
  sorry

end increasing_or_decreasing_subseq_l140_140707


namespace find_f_of_f_l140_140342

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 then 0 else (4 * x + 1 - 2 / x) / 3

theorem find_f_of_f (h : ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = 2 * x + 1) : 
  f 2 = -1/3 :=
sorry

end find_f_of_f_l140_140342


namespace backpack_price_equation_l140_140944

-- Define the original price of the backpack
variable (x : ℝ)

-- Define the conditions
def discount1 (x : ℝ) : ℝ := 0.8 * x
def discount2 (d : ℝ) : ℝ := d - 10
def final_price (p : ℝ) : Prop := p = 90

-- Final statement to be proved
theorem backpack_price_equation : final_price (discount2 (discount1 x)) ↔ 0.8 * x - 10 = 90 := sorry

end backpack_price_equation_l140_140944


namespace f_2023_value_l140_140777

noncomputable def f : ℕ → ℝ := sorry

axiom f_condition (a b n : ℕ) (ha : a > 0) (hb : b > 0) (hn : 2^n = a + b) : f a + f b = n^2 + 1

theorem f_2023_value : f 2023 = 107 :=
by 
  sorry

end f_2023_value_l140_140777


namespace solve_for_n_l140_140299

def number_of_balls : ℕ := sorry

axiom A : number_of_balls = 2

theorem solve_for_n (n : ℕ) (h : (1 + 1 + n = number_of_balls) ∧ ((n : ℝ) / (1 + 1 + n) = 1 / 2)) : n = 2 :=
sorry

end solve_for_n_l140_140299


namespace range_of_a_l140_140034

theorem range_of_a 
  (e : ℝ) (h_e_pos : 0 < e) 
  (a : ℝ) 
  (h_equation : ∃ x₁ x₂ : ℝ, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₁ ≠ x₂ ∧ (1 / e ^ x₁ - a / x₁ = 0) ∧ (1 / e ^ x₂ - a / x₂ = 0)) :
  0 < a ∧ a < 1 / e :=
by
  sorry

end range_of_a_l140_140034


namespace cos_13pi_over_4_eq_neg_one_div_sqrt_two_l140_140180

noncomputable def cos_13pi_over_4 : Real :=
  Real.cos (13 * Real.pi / 4)

theorem cos_13pi_over_4_eq_neg_one_div_sqrt_two : 
  cos_13pi_over_4 = -1 / Real.sqrt 2 := by 
  sorry

end cos_13pi_over_4_eq_neg_one_div_sqrt_two_l140_140180


namespace chromatic_number_bound_l140_140744

variables {G : SimpleGraph V} {n : ℕ} (V : Type)

noncomputable def chromatic_number_le_n_plus_one : Prop :=
  ∀ (G : SimpleGraph V), (∀ (e : G.edge_set), e.card ≤ n) → (G.chromatic_number ≤ n + 1)

theorem chromatic_number_bound (h₁ : 2 ≤ n) (h₂ : ∀ (e : G.edge_set), e.card ≤ n) : 
  (G.chromatic_number ≤ n + 1) :=
begin
  sorry
end

end chromatic_number_bound_l140_140744


namespace k_gt_4_l140_140721

theorem k_gt_4 {x y k : ℝ} (h1 : 2 * x + y = 2 * k - 1) (h2 : x + 2 * y = -4) (h3 : x + y > 1) : k > 4 :=
by
  -- This 'sorry' serves as a placeholder for the actual proof steps
  sorry

end k_gt_4_l140_140721


namespace first_day_exceeds_target_l140_140733

-- Definitions based on the conditions
def initial_count : ℕ := 5
def daily_growth_factor : ℕ := 3
def target_count : ℕ := 200

-- The proof problem in Lean
theorem first_day_exceeds_target : ∃ n : ℕ, 5 * 3 ^ n > 200 ∧ ∀ m < n, ¬ (5 * 3 ^ m > 200) :=
by
  sorry

end first_day_exceeds_target_l140_140733


namespace number_division_l140_140522

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l140_140522


namespace shopkeeper_profit_percent_l140_140307

noncomputable def profit_percent : ℚ := 
let cp_each := 1       -- Cost price of each article
let sp_each := 1.2     -- Selling price of each article without discount
let discount := 0.05   -- 5% discount
let tax := 0.10        -- 10% sales tax
let articles := 30     -- Number of articles
let cp_total := articles * cp_each      -- Total cost price
let sp_after_discount := sp_each * (1 - discount)    -- Selling price after discount
let revenue_before_tax := articles * sp_after_discount   -- Total revenue before tax
let tax_amount := revenue_before_tax * tax   -- Sales tax amount
let revenue_after_tax := revenue_before_tax + tax_amount -- Total revenue after tax
let profit := revenue_after_tax - cp_total -- Profit
(profit / cp_total) * 100 -- Profit percent

theorem shopkeeper_profit_percent : profit_percent = 25.4 :=
by
  -- Here follows the proof based on the conditions and steps above
  sorry

end shopkeeper_profit_percent_l140_140307


namespace determine_m_value_l140_140569

theorem determine_m_value 
  (a b m : ℝ)
  (h1 : 2^a = m)
  (h2 : 5^b = m)
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := 
sorry

end determine_m_value_l140_140569


namespace john_memory_card_cost_l140_140741

-- Define conditions
def pictures_per_day : ℕ := 10
def days_per_year : ℕ := 365
def years : ℕ := 3
def pictures_per_card : ℕ := 50
def cost_per_card : ℕ := 60

-- Define total days
def total_days (years : ℕ) (days_per_year : ℕ) : ℕ := years * days_per_year

-- Define total pictures
def total_pictures (pictures_per_day : ℕ) (total_days : ℕ) : ℕ := pictures_per_day * total_days

-- Define required cards
def required_cards (total_pictures : ℕ) (pictures_per_card : ℕ) : ℕ :=
  (total_pictures + pictures_per_card - 1) / pictures_per_card  -- ceiling division

-- Define total cost
def total_cost (required_cards : ℕ) (cost_per_card : ℕ) : ℕ := required_cards * cost_per_card

-- Prove the total cost equals $13,140
theorem john_memory_card_cost : total_cost (required_cards (total_pictures pictures_per_day (total_days years days_per_year)) pictures_per_card) cost_per_card = 13140 :=
by
  sorry

end john_memory_card_cost_l140_140741


namespace total_animals_peppersprayed_l140_140082

-- Define the conditions
def number_of_raccoons : ℕ := 12
def squirrels_vs_raccoons : ℕ := 6
def number_of_squirrels (raccoons : ℕ) (factor : ℕ) : ℕ := raccoons * factor

-- Define the proof statement
theorem total_animals_peppersprayed : 
  number_of_squirrels number_of_raccoons squirrels_vs_raccoons + number_of_raccoons = 84 :=
by
  -- The proof would go here
  sorry

end total_animals_peppersprayed_l140_140082


namespace sqrt_inequality_l140_140743

theorem sqrt_inequality (a b c : ℝ) (θ : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a * (Real.cos θ)^2 + b * (Real.sin θ)^2 < c) :
  Real.sqrt a * (Real.cos θ)^2 + Real.sqrt b * (Real.sin θ)^2 < Real.sqrt c :=
sorry

end sqrt_inequality_l140_140743


namespace length_of_AB_l140_140975

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2) / 25 + (y^2) / 16 = 1

-- Define the line perpendicular to the x-axis passing through the right focus of the ellipse
def line_perpendicular_y_axis_through_focus (y : ℝ) : Prop := true

-- Define the right focus of the ellipse
def right_focus : ℝ × ℝ := (3, 0)

-- Statement to prove the length of the line segment AB
theorem length_of_AB : 
  ∃ A B : ℝ × ℝ, 
  (ellipse A.1 A.2 ∧ ellipse B.1 B.2) ∧ 
  (A.1 = 3 ∧ B.1 = 3) ∧
  (|A.2 - B.2| = 2 * 16 / 5) :=
sorry

end length_of_AB_l140_140975


namespace trig_identity_l140_140856

theorem trig_identity {α : ℝ} (h : Real.tan α = 2) : 
  (Real.sin (π + α) - Real.cos (π - α)) / 
  (Real.sin (π / 2 + α) - Real.cos (3 * π / 2 - α)) 
  = -1 / 3 := 
by 
  sorry

end trig_identity_l140_140856


namespace S2014_value_l140_140510

variable (S : ℕ → ℤ) -- S_n represents sum of the first n terms of the arithmetic sequence
variable (a1 : ℤ) -- First term of the arithmetic sequence
variable (d : ℤ) -- Common difference of the arithmetic sequence

-- Given conditions
variable (h1 : a1 = -2016)
variable (h2 : (S 2016) / 2016 - (S 2010) / 2010 = 6)

-- The proof problem
theorem S2014_value :
  S 2014 = -6042 :=
sorry -- Proof omitted

end S2014_value_l140_140510


namespace perpendicular_lines_to_parallel_planes_l140_140712

-- Define non-overlapping lines and planes in a 3D geometry space
variables {m n : line} {α β : plane}

-- Conditions:
-- m is a line
-- α and β are planes
-- m is perpendicular to α
-- m is perpendicular to β

-- To prove:
-- α is parallel to β

theorem perpendicular_lines_to_parallel_planes 
  (non_overlap_mn : m ≠ n) 
  (non_overlap_ab : α ≠ β) 
  (m_perp_α : m ⊥ α) 
  (m_perp_β : m ⊥ β) : parallel α β :=
sorry

end perpendicular_lines_to_parallel_planes_l140_140712


namespace hexagon_arithmetic_sum_l140_140763

theorem hexagon_arithmetic_sum (a n : ℝ) (h : 6 * a + 15 * n = 720) : 2 * a + 5 * n = 240 :=
by
  sorry

end hexagon_arithmetic_sum_l140_140763


namespace time_for_a_and_b_together_l140_140066

variable (R_a R_b : ℝ)
variable (T_ab : ℝ)

-- Given conditions
def condition_1 : Prop := R_a = 3 * R_b
def condition_2 : Prop := R_a * 28 = 1  -- '1' denotes the entire work

-- Proof goal
theorem time_for_a_and_b_together (h1 : condition_1 R_a R_b) (h2 : condition_2 R_a) : T_ab = 21 := 
by
  sorry

end time_for_a_and_b_together_l140_140066


namespace solution_problem_l140_140032

noncomputable def proof_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : Prop :=
  (-1 < (x - y)) ∧ ((x - y) < 1) ∧ (∀ (x y : ℝ), (0 < x) ∧ (0 < y) ∧ (x + y = 1) → (min ((1/x) + (x/y)) = 3))

theorem solution_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) :
  proof_problem x y hx hy h := 
sorry

end solution_problem_l140_140032


namespace range_a_for_increasing_f_l140_140768

theorem range_a_for_increasing_f :
  (∀ (x : ℝ), 1 ≤ x → (2 * x - 2 * a) ≥ 0) → a ≤ 1 := by
  intro h
  sorry

end range_a_for_increasing_f_l140_140768


namespace parallel_lines_coplanar_l140_140458

axiom Plane : Type
axiom Point : Type
axiom Line : Type

axiom A : Point
axiom B : Point
axiom C : Point
axiom D : Point

axiom α : Plane
axiom β : Plane

axiom in_plane (p : Point) (π : Plane) : Prop
axiom parallel_plane (π1 π2 : Plane) : Prop
axiom parallel_line (l1 l2 : Line) : Prop
axiom line_through (P Q : Point) : Line
axiom coplanar (P Q R S : Point) : Prop

-- Conditions
axiom A_in_α : in_plane A α
axiom C_in_α : in_plane C α
axiom B_in_β : in_plane B β
axiom D_in_β : in_plane D β
axiom α_parallel_β : parallel_plane α β

-- Statement
theorem parallel_lines_coplanar :
  parallel_line (line_through A C) (line_through B D) ↔ coplanar A B C D :=
sorry

end parallel_lines_coplanar_l140_140458


namespace bridge_length_l140_140806

-- Definitions based on conditions
def Lt : ℕ := 148
def Skm : ℕ := 45
def T : ℕ := 30

-- Conversion from km/h to m/s
def conversion_factor : ℕ := 1000 / 3600
def Sm : ℝ := Skm * conversion_factor

-- Calculation of distance traveled in 30 seconds
def distance : ℝ := Sm * T

-- The length of the bridge
def L_bridge : ℝ := distance - Lt

theorem bridge_length : L_bridge = 227 := sorry

end bridge_length_l140_140806


namespace nat_nums_division_by_7_l140_140560

theorem nat_nums_division_by_7 (n : ℕ) : 
  (∃ q r, n = 7 * q + r ∧ q = r ∧ 1 ≤ r ∧ r < 7) ↔ 
  n = 8 ∨ n = 16 ∨ n = 24 ∨ n = 32 ∨ n = 40 ∨ n = 48 := by
  sorry

end nat_nums_division_by_7_l140_140560


namespace hyperbola_center_is_equidistant_l140_140940

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem hyperbola_center_is_equidistant (F1 F2 C : ℝ × ℝ) 
  (hF1 : F1 = (3, -2)) 
  (hF2 : F2 = (11, 6))
  (hC : C = ((F1.1 + F2.1) / 2, (F1.2 + F2.2) / 2)) :
  C = (7, 2) ∧ distance C F1 = distance C F2 :=
by
  -- Fill in with the appropriate proofs
  sorry

end hyperbola_center_is_equidistant_l140_140940


namespace final_building_height_l140_140013

noncomputable def height_of_final_building 
    (Crane1_height : ℝ)
    (Building1_height : ℝ)
    (Crane2_height : ℝ)
    (Building2_height : ℝ)
    (Crane3_height : ℝ)
    (Average_difference : ℝ) : ℝ :=
    Crane3_height / (1 + Average_difference)

theorem final_building_height
    (Crane1_height : ℝ := 228)
    (Building1_height : ℝ := 200)
    (Crane2_height : ℝ := 120)
    (Building2_height : ℝ := 100)
    (Crane3_height : ℝ := 147)
    (Average_difference : ℝ := 0.13)
    (HCrane1 : 1 + (Crane1_height - Building1_height) / Building1_height = 1.14)
    (HCrane2 : 1 + (Crane2_height - Building2_height) / Building2_height = 1.20)
    (HAvg : (1.14 + 1.20) / 2 = 1.13) :
    height_of_final_building Crane1_height Building1_height Crane2_height Building2_height Crane3_height Average_difference = 130 := 
sorry

end final_building_height_l140_140013


namespace dhoni_leftover_percentage_l140_140685

variable (E : ℝ) (spent_on_rent : ℝ) (spent_on_dishwasher : ℝ)

def percent_spent_on_rent : ℝ := 0.40
def percent_spent_on_dishwasher : ℝ := 0.32

theorem dhoni_leftover_percentage (E : ℝ) :
  (1 - (percent_spent_on_rent + percent_spent_on_dishwasher)) * E / E = 0.28 :=
by
  sorry

end dhoni_leftover_percentage_l140_140685


namespace find_solutions_l140_140319

theorem find_solutions (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x^2 + 4^y = 5^z ↔ (x = 3 ∧ y = 2 ∧ z = 2) ∨ (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 11 ∧ y = 1 ∧ z = 3) :=
by sorry

end find_solutions_l140_140319


namespace fuel_relationship_l140_140511

theorem fuel_relationship (y : ℕ → ℕ) (h₀ : y 0 = 80) (h₁ : y 1 = 70) (h₂ : y 2 = 60) (h₃ : y 3 = 50) :
  ∀ x : ℕ, y x = 80 - 10 * x :=
by
  sorry

end fuel_relationship_l140_140511


namespace rectangles_in_5x5_grid_l140_140464

/-- 
Proof that the number of different rectangles with sides parallel to the grid 
that can be formed by connecting four of the dots in a 5x5 square array of dots 
is equal to 100.
-/
theorem rectangles_in_5x5_grid : ∃ (n : ℕ), n = 100 ∧ n = (Nat.choose 5 2 * Nat.choose 5 2) :=
by
  use 100
  split
  . rfl
  . rw [Nat.choose, Nat.choose]
  . sorry

end rectangles_in_5x5_grid_l140_140464


namespace parabola_exists_l140_140859

noncomputable def parabola_conditions (a b : ℝ) : Prop :=
  (a + b = -3) ∧ (4 * a - 2 * b = 12)

noncomputable def translated_min_equals_six (m : ℝ) : Prop :=
  (m > 0) ∧ ((-1 - 2 + m)^2 - 3 = 6) ∨ ((3 - 2 - m)^2 - 3 = 6)

theorem parabola_exists (a b m : ℝ) (x y : ℝ) :
  parabola_conditions a b → y = x^2 + b * x + 1 → translated_min_equals_six m →
  (y = x^2 - 4 * x + 1) ∧ (m = 6 ∨ m = 4) := 
by 
  sorry

end parabola_exists_l140_140859


namespace circus_tent_sections_l140_140114

noncomputable def sections_in_circus_tent (total_capacity : ℕ) (section_capacity : ℕ) : ℕ :=
  total_capacity / section_capacity

theorem circus_tent_sections : sections_in_circus_tent 984 246 = 4 := 
  by 
  sorry

end circus_tent_sections_l140_140114


namespace hyperbola_asymptote_m_value_l140_140444

theorem hyperbola_asymptote_m_value (m : ℝ) :
  (∀ x y : ℝ, (x^2 / m - y^2 / 6 = 1) → (y = x)) → m = 6 :=
by
  intros hx
  sorry

end hyperbola_asymptote_m_value_l140_140444


namespace maximum_value_is_l140_140616

noncomputable def maximum_value (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x^2 - 2 * x * y + 3 * y^2 = 12) : ℝ :=
  x^2 + 2 * x * y + 3 * y^2

theorem maximum_value_is (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x^2 - 2 * x * y + 3 * y^2 = 12) :
  maximum_value x y h₁ h₂ h₃ ≤ 18 + 12 * Real.sqrt 3 :=
sorry

end maximum_value_is_l140_140616


namespace correct_operations_l140_140432

theorem correct_operations : 6 * 3 + 4 + 2 = 24 := by
  -- Proof goes here
  sorry

end correct_operations_l140_140432


namespace range_of_k_real_roots_l140_140990

variable (k : ℝ)
def quadratic_has_real_roots : Prop :=
  let a := k - 1
  let b := 2
  let c := 1
  let Δ := b^2 - 4 * a * c
  Δ ≥ 0 ∧ a ≠ 0

theorem range_of_k_real_roots :
  quadratic_has_real_roots k ↔ (k ≤ 2 ∧ k ≠ 1) := by
  sorry

end range_of_k_real_roots_l140_140990


namespace f_36_l140_140857

variable {R : Type*} [CommRing R]
variable (f : R → R) (p q : R)

-- Conditions
axiom f_mult_add : ∀ x y, f (x * y) = f x + f y
axiom f_2 : f 2 = p
axiom f_3 : f 3 = q

-- Statement to prove
theorem f_36 : f 36 = 2 * (p + q) :=
by
  sorry

end f_36_l140_140857


namespace bert_fraction_spent_l140_140312

theorem bert_fraction_spent (f : ℝ) :
  let initial := 52
  let after_hardware := initial - initial * f
  let after_cleaners := after_hardware - 9
  let after_grocery := after_cleaners / 2
  let final := 15
  after_grocery = final → f = 1/4 :=
by
  intros h
  sorry

end bert_fraction_spent_l140_140312


namespace complex_number_purely_imaginary_l140_140470

theorem complex_number_purely_imaginary (m : ℝ) :
  (m^2 - 2 * m - 3 = 0) ∧ (m^2 - 1 ≠ 0) → m = 3 :=
by
  intros h
  sorry

end complex_number_purely_imaginary_l140_140470


namespace f_eq_zero_of_le_zero_l140_140361

variable {R : Type*} [LinearOrderedField R]
variable {f : R → R}
variable (cond : ∀ x y : R, f (x + y) ≤ y * f x + f (f x))

theorem f_eq_zero_of_le_zero (x : R) (h : x ≤ 0) : f x = 0 :=
sorry

end f_eq_zero_of_le_zero_l140_140361


namespace angle_tuvels_equiv_l140_140891

-- Defining the conditions
def full_circle_tuvels : ℕ := 400
def degree_angle_in_circle : ℕ := 360
def specific_angle_degrees : ℕ := 45

-- Proof statement showing the equivalence
theorem angle_tuvels_equiv :
  (specific_angle_degrees * full_circle_tuvels) / degree_angle_in_circle = 50 :=
by
  sorry

end angle_tuvels_equiv_l140_140891


namespace carol_meets_alice_in_30_minutes_l140_140163

def time_to_meet (alice_speed carol_speed initial_distance : ℕ) : ℕ :=
((initial_distance * 60) / (alice_speed + carol_speed))

theorem carol_meets_alice_in_30_minutes :
  time_to_meet 4 6 5 = 30 := 
by 
  sorry

end carol_meets_alice_in_30_minutes_l140_140163


namespace toothpicks_15th_stage_l140_140344
-- Import the required library

-- Define the arithmetic sequence based on the provided conditions.
def toothpicks (n : ℕ) : ℕ :=
  if n = 1 then 5 else 3 * (n - 1) + 5

-- State the theorem
theorem toothpicks_15th_stage : toothpicks 15 = 47 :=
by {
  -- Provide the proof here, but currently using sorry as instructed
  sorry
}

end toothpicks_15th_stage_l140_140344


namespace cos_double_angle_l140_140582

theorem cos_double_angle (α : ℝ) (h : Real.sin (π + α) = 2 / 3) : Real.cos (2 * α) = 1 / 9 := 
by
  sorry

end cos_double_angle_l140_140582


namespace bricks_in_row_l140_140389

theorem bricks_in_row 
  (total_bricks : ℕ) 
  (rows_per_wall : ℕ) 
  (num_walls : ℕ)
  (total_rows : ℕ)
  (h1 : total_bricks = 3000)
  (h2 : rows_per_wall = 50)
  (h3 : num_walls = 2) 
  (h4 : total_rows = rows_per_wall * num_walls) :
  total_bricks / total_rows = 30 :=
by
  sorry

end bricks_in_row_l140_140389


namespace total_books_gwen_has_l140_140869

-- Definitions based on conditions in part a
def mystery_shelves : ℕ := 5
def picture_shelves : ℕ := 3
def books_per_shelf : ℕ := 4

-- Problem statement in Lean 4
theorem total_books_gwen_has : 
  mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf = 32 := by
  -- This is where the proof would go, but we include sorry to skip for now
  sorry

end total_books_gwen_has_l140_140869


namespace find_original_number_l140_140662

theorem find_original_number (x : ℝ) : 1.5 * x = 525 → x = 350 := by
  sorry

end find_original_number_l140_140662


namespace find_crew_members_l140_140824

noncomputable def passengers_initial := 124
noncomputable def passengers_texas := passengers_initial - 58 + 24
noncomputable def passengers_nc := passengers_texas - 47 + 14
noncomputable def total_people_virginia := 67

theorem find_crew_members (passengers_initial passengers_texas passengers_nc total_people_virginia : ℕ) :
  passengers_initial = 124 →
  passengers_texas = passengers_initial - 58 + 24 →
  passengers_nc = passengers_texas - 47 + 14 →
  total_people_virginia = 67 →
  ∃ crew_members : ℕ, total_people_virginia = passengers_nc + crew_members ∧ crew_members = 10 :=
by
  sorry

end find_crew_members_l140_140824


namespace ratio_of_roots_l140_140698

theorem ratio_of_roots (a b c x₁ x₂ : ℝ) (h₁ : a ≠ 0) (h₂ : c ≠ 0) (h₃ : a * x₁^2 + b * x₁ + c = 0) (h₄ : a * x₂^2 + b * x₂ + c = 0) (h₅ : x₁ = 4 * x₂) : (b^2) / (a * c) = 25 / 4 :=
by
  sorry

end ratio_of_roots_l140_140698


namespace contractor_total_engaged_days_l140_140401

-- Definitions based on conditions
def earnings_per_work_day : ℝ := 25
def fine_per_absent_day : ℝ := 7.5
def total_earnings : ℝ := 425
def days_absent : ℝ := 10

-- The proof problem statement
theorem contractor_total_engaged_days :
  ∃ (x y : ℝ), y = days_absent ∧ total_earnings = earnings_per_work_day * x - fine_per_absent_day * y ∧ x + y = 30 :=
by
  -- let x be the number of working days
  -- let y be the number of absent days
  -- y is given as 10
  -- total_earnings = 25 * x - 7.5 * 10
  -- solve for x and sum x and y to get 30
  sorry

end contractor_total_engaged_days_l140_140401


namespace cos_thirteen_pi_over_four_l140_140181

theorem cos_thirteen_pi_over_four : Real.cos (13 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_thirteen_pi_over_four_l140_140181


namespace penthouse_floors_l140_140156

theorem penthouse_floors (R P : ℕ) (h1 : R + P = 23) (h2 : 12 * R + 2 * P = 256) : P = 2 :=
by
  sorry

end penthouse_floors_l140_140156


namespace ratio_a_c_l140_140135

theorem ratio_a_c (a b c : ℕ) (h1 : a / b = 8 / 3) (h2 : b / c = 1 / 5) : a / c = 8 / 15 := 
by
  sorry

end ratio_a_c_l140_140135


namespace problem_statement_l140_140200

noncomputable def f (x : ℝ) : ℝ := x + 1
noncomputable def g (x : ℝ) : ℝ := -x + 1
noncomputable def h (x : ℝ) : ℝ := f x * g x

theorem problem_statement :
  (h (-x) = h x) :=
by
  sorry

end problem_statement_l140_140200


namespace find_x_l140_140134

theorem find_x (m n k : ℝ) (x z : ℝ) (h1 : x = m * (n / (Real.sqrt z))^3)
  (h2 : x = 3 ∧ z = 12 ∧ 3 * 12 * Real.sqrt 12 = k) :
  (z = 75) → x = 24 / 125 :=
by
  -- Placeholder for proof, these assumptions and conditions would form the basis of the proof.
  sorry

end find_x_l140_140134


namespace arithmetic_seq_a8_l140_140986

theorem arithmetic_seq_a8 : ∀ (a : ℕ → ℤ), 
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) → 
  (a 5 + a 6 = 22) → 
  (a 3 = 7) → 
  a 8 = 15 :=
by
  intros a ha_arithmetic hsum h3
  sorry

end arithmetic_seq_a8_l140_140986


namespace cos_double_angle_l140_140338

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 2/3) : Real.cos (2 * θ) = -1/9 := 
  sorry

end cos_double_angle_l140_140338


namespace max_intersections_circle_quadrilateral_l140_140518

theorem max_intersections_circle_quadrilateral (circle : Type) (quadrilateral : Type) 
  (intersects : circle → quadrilateral → ℕ) (h : ∀ (c : circle) (line_segment : Type), intersects c line_segment ≤ 2) :
  ∃ (q : quadrilateral), intersects circle quadrilateral = 8 :=
by
  sorry

end max_intersections_circle_quadrilateral_l140_140518


namespace number_of_divisors_of_3003_l140_140185

noncomputable def count_divisors (n : ℕ) : ℕ :=
  (finset.filter (λ d, n % d = 0) (finset.range (n + 1))).card

theorem number_of_divisors_of_3003 : count_divisors 3003 = 16 :=
by
  sorry

end number_of_divisors_of_3003_l140_140185


namespace least_value_of_x_l140_140473

theorem least_value_of_x (x p : ℕ) (h1 : (x / (11 * p)) = 3) (h2 : x > 0) (h3 : Nat.Prime p) : x = 66 := by
  sorry

end least_value_of_x_l140_140473


namespace find_J_l140_140219

variables (J S B : ℕ)

-- Conditions
def condition1 : Prop := J - 20 = 2 * S
def condition2 : Prop := B = J / 2
def condition3 : Prop := J + S + B = 330
def condition4 : Prop := (J - 20) + S + B = 318

-- Theorem to prove
theorem find_J (h1 : condition1 J S) (h2 : condition2 J B) (h3 : condition3 J S B) (h4 : condition4 J S B) :
  J = 170 :=
sorry

end find_J_l140_140219


namespace range_of_a_l140_140575

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 4 → a < x ∧ x < 5) → a ≤ 1 := 
sorry

end range_of_a_l140_140575


namespace cover_square_with_rectangles_l140_140792

theorem cover_square_with_rectangles :
  ∃ n : ℕ, n = 24 ∧
  ∀ (rect_area : ℕ) (square_area : ℕ), rect_area = 2 * 3 → square_area = 12 * 12 → square_area / rect_area = n :=
by
  use 24
  sorry

end cover_square_with_rectangles_l140_140792


namespace point_in_second_quadrant_l140_140476

def point (x : ℤ) (y : ℤ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant : point (-1) 3 = true := by
  sorry

end point_in_second_quadrant_l140_140476


namespace smallest_natural_number_multiple_of_36_with_unique_digits_digit_count_1023457896_sum_of_digits_1023457896_l140_140854

open Nat

theorem smallest_natural_number_multiple_of_36_with_unique_digits :
  ∃ (n : ℕ), (∀ k : ℕ, (k % 36 = 0 → (∀ d : ℕ, d ∈ List.range 10 → d ∈ digit_list k) → n ≤ k)) ∧
    n = 1023457896 :=
by
  -- Placeholder for the proof
  sorry

def digit_list (n : ℕ) : list ℕ :=
  let chars := n.digits 10
  chars.to_finset.val.to_list.sort

theorem digit_count_1023457896 :
  ∀ d : ℕ, d < 10 → d ∈ digit_list 1023457896 :=
by
  -- Placeholder for the proof
  sorry

theorem sum_of_digits_1023457896 :
  finset.sum (finset.range 10) (digit_list 1023457896) = 45 :=
by
  -- Placeholder for the proof
  sorry

end smallest_natural_number_multiple_of_36_with_unique_digits_digit_count_1023457896_sum_of_digits_1023457896_l140_140854


namespace find_unknown_number_l140_140081

-- Definitions

-- Declaring that we have an inserted number 'a' between 3 and unknown number 'b'
variable (a b : ℕ)

-- Conditions provided in the problem
def arithmetic_sequence_condition (a b : ℕ) : Prop := 
  a - 3 = b - a

def geometric_sequence_condition (a b : ℕ) : Prop :=
  (a - 6) / 3 = b / (a - 6)

-- The theorem statement equivalent to the problem
theorem find_unknown_number (h1 : arithmetic_sequence_condition a b) (h2 : geometric_sequence_condition a b) : b = 27 :=
sorry

end find_unknown_number_l140_140081


namespace max_value_of_m_l140_140861

variable (m : ℝ)

noncomputable def satisfies_inequality (m : ℝ) : Prop :=
∀ x > 0, m * x * Real.log x - (x + m) * Real.exp ((x - m) / m) ≤ 0

theorem max_value_of_m (h1 : 0 < m) (h2 : satisfies_inequality m) : m ≤ Real.exp 2 := sorry

end max_value_of_m_l140_140861


namespace cos_double_angle_example_l140_140056

def cos_double_angle_identity (x : ℝ) : Prop :=
  cos (2 * x) = 1 - 2 * (sin x) ^ 2

theorem cos_double_angle_example : cos_double_angle_identity (x : ℝ) 
  (h : sin x = - 2 / 3) : cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_example_l140_140056


namespace inequality_solution_l140_140373

variable {x : ℝ}

theorem inequality_solution :
  x ∈ Set.Ioo (-∞ : ℝ) 7 ∪ Set.Ioo (-∞ : ℝ) (-7) ∪ Set.Ioo (-7) 7 ↔ (x^2 - 49) / (x + 7) < 0 :=
by
  sorry

end inequality_solution_l140_140373


namespace total_games_won_l140_140731

-- Define the number of games won by the Chicago Bulls
def bulls_games : ℕ := 70

-- Define the number of games won by the Miami Heat
def heat_games : ℕ := bulls_games + 5

-- Define the total number of games won by both the Bulls and the Heat
def total_games : ℕ := bulls_games + heat_games

-- The theorem stating that the total number of games won by both teams is 145
theorem total_games_won : total_games = 145 := by
  -- Proof is omitted
  sorry

end total_games_won_l140_140731


namespace new_paint_intensity_l140_140610

theorem new_paint_intensity : 
  let I_original : ℝ := 0.5
  let I_added : ℝ := 0.2
  let replacement_fraction : ℝ := 1 / 3
  let remaining_fraction : ℝ := 2 / 3
  let I_new := remaining_fraction * I_original + replacement_fraction * I_added
  I_new = 0.4 :=
by
  -- sorry is used to skip the actual proof
  sorry

end new_paint_intensity_l140_140610


namespace rectangle_dimensions_l140_140764

-- Definitions from conditions
def is_rectangle (length width : ℝ) : Prop :=
  3 * width = length ∧ 3 * width^2 = 8 * width

-- The theorem to prove
theorem rectangle_dimensions :
  ∃ (length width : ℝ), is_rectangle length width ∧ width = 8 / 3 ∧ length = 8 := by
  sorry

end rectangle_dimensions_l140_140764


namespace problem_bound_l140_140328

theorem problem_bound (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by 
  sorry

end problem_bound_l140_140328


namespace fraction_incorrect_like_music_l140_140954

-- Define the conditions as given in the problem
def total_students : ℕ := 100
def like_music_percentage : ℝ := 0.7
def dislike_music_percentage : ℝ := 1 - like_music_percentage

def correct_like_percentage : ℝ := 0.75
def incorrect_like_percentage : ℝ := 1 - correct_like_percentage

def correct_dislike_percentage : ℝ := 0.85
def incorrect_dislike_percentage : ℝ := 1 - correct_dislike_percentage

-- The number of students liking music
def like_music_students : ℝ := total_students * like_music_percentage
-- The number of students disliking music
def dislike_music_students : ℝ := total_students * dislike_music_percentage

-- The number of students who correctly say they like music
def correct_like_music_say : ℝ := like_music_students * correct_like_percentage
-- The number of students who incorrectly say they dislike music
def incorrect_dislike_music_say : ℝ := like_music_students * incorrect_like_percentage

-- The number of students who correctly say they dislike music
def correct_dislike_music_say : ℝ := dislike_music_students * correct_dislike_percentage
-- The number of students who incorrectly say they like music
def incorrect_like_music_say : ℝ := dislike_music_students * incorrect_dislike_percentage

-- The total number of students who say they like music
def total_say_like_music : ℝ := correct_like_music_say + incorrect_like_music_say

-- The final theorem we want to prove
theorem fraction_incorrect_like_music : ((incorrect_like_music_say : ℝ) / total_say_like_music) = (5 / 58) :=
by
  -- here we would provide the proof, but for now, we use sorry
  sorry

end fraction_incorrect_like_music_l140_140954


namespace sum_a_b_range_l140_140684

noncomputable def f (x : ℝ) : ℝ := 3 / (1 + 3 * x^4)

theorem sum_a_b_range : let a := 0
                       let b := 3
                       a + b = 3 := by
  sorry

end sum_a_b_range_l140_140684


namespace rotated_clockwise_120_correct_l140_140535

-- Problem setup definitions
structure ShapePosition :=
  (triangle : Point)
  (smaller_circle : Point)
  (square : Point)

-- Conditions for the initial positions of the shapes
variable (initial : ShapePosition)

def rotated_positions (initial: ShapePosition) : ShapePosition :=
  { 
    triangle := initial.smaller_circle,
    smaller_circle := initial.square,
    square := initial.triangle 
  }

-- Problem statement: show that after a 120° clockwise rotation, 
-- the shapes move to the specified new positions.
theorem rotated_clockwise_120_correct (initial : ShapePosition) 
  (after_rotation : ShapePosition) :
  after_rotation = rotated_positions initial := 
sorry

end rotated_clockwise_120_correct_l140_140535


namespace solution_l140_140686

variable (x y z : ℝ)
variable (hx : x > 0)
variable (hy : y > 0)
variable (hz : z > 0)

-- Condition 1: 20/x + 6/y = 1
axiom eq1 : 20 / x + 6 / y = 1

-- Condition 2: 4/x + 2/y = 2/9
axiom eq2 : 4 / x + 2 / y = 2 / 9

-- What we need to prove: 1/z = 1/x + 1/y
axiom eq3 : 1 / x + 1 / y = 1 / z

theorem solution : z = 14.4 := by
  -- Omitted proof, just the statement
  sorry

end solution_l140_140686


namespace total_votes_cast_l140_140228

theorem total_votes_cast (V: ℕ) (invalid_votes: ℕ) (diff_votes: ℕ) 
  (H1: invalid_votes = 200) 
  (H2: diff_votes = 700) 
  (H3: (0.01 : ℝ) * V = diff_votes) 
  : (V + invalid_votes = 70200) :=
by
  sorry

end total_votes_cast_l140_140228


namespace students_in_class_C_l140_140666

theorem students_in_class_C 
    (total_students : ℕ := 80) 
    (percent_class_A : ℕ := 40) 
    (class_B_difference : ℕ := 21) 
    (h_percent : percent_class_A = 40) 
    (h_class_B_diff : class_B_difference = 21) 
    (h_total_students : total_students = 80) : 
    total_students - ((percent_class_A * total_students) / 100 - class_B_difference + (percent_class_A * total_students) / 100) = 37 := by
    sorry

end students_in_class_C_l140_140666


namespace range_of_x_l140_140845

noncomputable def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
noncomputable def specific_function (f : ℝ → ℝ) := ∀ x : ℝ, x ≥ 0 → f x = 2^x

theorem range_of_x (f : ℝ → ℝ)  
  (hf_even : even_function f) 
  (hf_specific : specific_function f) : {x : ℝ | f (1 - 2 * x) < f 3} = {x : ℝ | -1 < x ∧ x < 2} := 
by
  sorry

end range_of_x_l140_140845


namespace pyramid_certain_height_l140_140374

noncomputable def certain_height (h : ℝ) : Prop :=
  let height := h + 20
  let width := height + 234
  (height + width = 1274) → h = 1000 / 3

theorem pyramid_certain_height (h : ℝ) : certain_height h :=
by
  let height := h + 20
  let width := height + 234
  have h_eq : (height + width = 1274) → h = 1000 / 3 := sorry
  exact h_eq

end pyramid_certain_height_l140_140374


namespace words_per_page_l140_140810

/-- 
  Let p denote the number of words per page.
  Given conditions:
  - A book contains 154 pages.
  - Each page has the same number of words, p, and no page contains more than 120 words.
  - The total number of words in the book (154p) is congruent to 250 modulo 227.
  Prove that the number of words in each page p is congruent to 49 modulo 227.
 -/
theorem words_per_page (p : ℕ) (h1 : p ≤ 120) (h2 : 154 * p ≡ 250 [MOD 227]) : p ≡ 49 [MOD 227] :=
sorry

end words_per_page_l140_140810


namespace carson_clawed_total_l140_140009

theorem carson_clawed_total :
  let wombats := 9
  let wombat_claws := 4
  let rheas := 3
  let rhea_claws := 1
  wombats * wombat_claws + rheas * rhea_claws = 39 := by
  let wombats := 9
  let wombat_claws := 4
  let rheas := 3
  let rhea_claws := 1
  show wombats * wombat_claws + rheas * rhea_claws = 39
  sorry

end carson_clawed_total_l140_140009


namespace grandfather_older_than_xiaoming_dad_age_when_twenty_times_xiaoming_l140_140644

-- Definition of the conditions
def grandfather_age (gm_age dad_age : ℕ) := gm_age = 2 * dad_age
def dad_age_eight_times_xiaoming (dad_age xm_age : ℕ) := dad_age = 8 * xm_age
def grandfather_age_61 (gm_age : ℕ) := gm_age = 61
def twenty_times_xiaoming (gm_age xm_age : ℕ) := gm_age = 20 * xm_age

-- Question 1: Proof that Grandpa is 57 years older than Xiaoming 
theorem grandfather_older_than_xiaoming (gm_age dad_age xm_age : ℕ) 
  (h1 : grandfather_age gm_age dad_age) (h2 : dad_age_eight_times_xiaoming dad_age xm_age)
  (h3 : grandfather_age_61 gm_age)
  : gm_age - xm_age = 57 := 
sorry

-- Question 2: Proof that Dad is 31 years old when Grandpa's age is twenty times Xiaoming's age
theorem dad_age_when_twenty_times_xiaoming (gm_age dad_age xm_age : ℕ) 
  (h1 : twenty_times_xiaoming gm_age xm_age)
  (hm : grandfather_age gm_age dad_age)
  : dad_age = 31 :=
sorry

end grandfather_older_than_xiaoming_dad_age_when_twenty_times_xiaoming_l140_140644


namespace exists_ten_positive_integers_l140_140017

theorem exists_ten_positive_integers :
  ∃ (a : ℕ → ℕ), (∀ i j, i ≠ j → ¬ (a i ∣ a j))
  ∧ (∀ i j, (a i)^2 ∣ a j) :=
sorry

end exists_ten_positive_integers_l140_140017


namespace ravi_nickels_l140_140093

variables (n q d : ℕ)

-- Defining the conditions
def quarters (n : ℕ) : ℕ := n + 2
def dimes (q : ℕ) : ℕ := q + 4

-- Using these definitions to form the Lean theorem
theorem ravi_nickels : 
  ∃ n, q = quarters n ∧ d = dimes q ∧ 
  (0.05 * n + 0.25 * q + 0.10 * d : ℝ) = 3.50 ∧ n = 6 :=
sorry

end ravi_nickels_l140_140093


namespace firecrackers_defective_fraction_l140_140739

theorem firecrackers_defective_fraction (initial_total good_remaining confiscated : ℕ) 
(h_initial : initial_total = 48) 
(h_confiscated : confiscated = 12) 
(h_good_remaining : good_remaining = 15) : 
(initial_total - confiscated - 2 * good_remaining) / (initial_total - confiscated) = 1 / 6 := by
  sorry

end firecrackers_defective_fraction_l140_140739


namespace triangle_angle_contradiction_l140_140500

theorem triangle_angle_contradiction (A B C : ℝ) (hA : A > 60) (hB : B > 60) (hC : C > 60) (h_sum : A + B + C = 180) :
  false :=
by
  -- Here "A > 60, B > 60, C > 60 and A + B + C = 180" leads to a contradiction
  sorry

end triangle_angle_contradiction_l140_140500


namespace find_sum_of_squares_l140_140843

variable (x y : ℝ)

theorem find_sum_of_squares (h₁ : x * y = 8) (h₂ : x^2 * y + x * y^2 + x + y = 94) : 
  x^2 + y^2 = 7540 / 81 :=
by
  sorry

end find_sum_of_squares_l140_140843


namespace total_bending_angle_l140_140315

theorem total_bending_angle (n : ℕ) (h : n > 4) (θ : ℝ) (hθ : θ = 360 / (2 * n)) : 
  ∃ α : ℝ, α = 180 :=
by
  sorry

end total_bending_angle_l140_140315


namespace long_diagonal_length_l140_140817

-- Define the lengths of the rhombus sides and diagonals
variables (a b : ℝ) (s : ℝ)
variable (side_length : ℝ)
variable (short_diagonal : ℝ)
variable (long_diagonal : ℝ)

-- Given conditions
def rhombus (side_length: ℝ) (short_diagonal: ℝ) : Prop :=
  side_length = 51 ∧ short_diagonal = 48

-- To prove: length longer diagonal is 90 units
theorem long_diagonal_length (side_length: ℝ) (short_diagonal: ℝ) (long_diagonal: ℝ) :
  rhombus side_length short_diagonal →
  long_diagonal = 90 :=
by
  sorry 

end long_diagonal_length_l140_140817


namespace larger_integer_l140_140281

theorem larger_integer (x y : ℕ) (h_diff : y - x = 8) (h_prod : x * y = 272) : y = 20 :=
by
  sorry

end larger_integer_l140_140281


namespace mean_of_four_numbers_l140_140781

theorem mean_of_four_numbers (a b c d : ℚ) (h : a + b + c + d = 1/2) : (a + b + c + d) / 4 = 1 / 8 :=
by
  -- proof skipped
  sorry

end mean_of_four_numbers_l140_140781


namespace solve_for_x_y_l140_140595

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

noncomputable def triangle_ABC (A B C E F : V) (x y : ℝ) : Prop :=
  (E - A) = (1 / 2) • (B - A) ∧
  (C - F) = (2 : ℝ) • (A - F) ∧
  (E - F) = x • (B - A) + y • (C - A)

theorem solve_for_x_y (A B C E F : V) (x y : ℝ) :
  triangle_ABC A B C E F x y →
  x + y = - (1 / 6 : ℝ) :=
by
  sorry

end solve_for_x_y_l140_140595


namespace greatest_ln_2_l140_140646

theorem greatest_ln_2 (x1 x2 x3 x4 : ℝ) (h1 : x1 = (Real.log 2) ^ 2) (h2 : x2 = Real.log (Real.log 2)) (h3 : x3 = Real.log (Real.sqrt 2)) (h4 : x4 = Real.log 2) 
  (h5 : Real.log 2 < 1) : 
  x4 = max x1 (max x2 (max x3 x4)) := by 
  sorry

end greatest_ln_2_l140_140646


namespace max_value_l140_140965

theorem max_value (x y : ℝ) : 
  (x + 3 * y + 4) / (Real.sqrt (x ^ 2 + y ^ 2 + 4)) ≤ Real.sqrt 26 :=
by
  -- Proof should be here
  sorry

end max_value_l140_140965


namespace total_legs_correct_l140_140257

-- Number of animals
def horses : ℕ := 2
def dogs : ℕ := 5
def cats : ℕ := 7
def turtles : ℕ := 3
def goat : ℕ := 1

-- Total number of animals
def total_animals : ℕ := horses + dogs + cats + turtles + goat

-- Total number of legs
def total_legs : ℕ := total_animals * 4

theorem total_legs_correct : total_legs = 72 := by
  -- proof skipped
  sorry

end total_legs_correct_l140_140257


namespace amount_subtracted_for_new_ratio_l140_140276

theorem amount_subtracted_for_new_ratio (x a : ℝ) (h1 : 3 * x = 72) (h2 : 8 * x = 192)
(h3 : (3 * x - a) / (8 * x - a) = 4 / 9) : a = 24 := by
  -- Proof will go here
  sorry

end amount_subtracted_for_new_ratio_l140_140276


namespace inequality_solution_l140_140372

theorem inequality_solution (x : ℝ) (hx : x ≠ -7) :
  (x^2 - 49) / (x + 7) < 0 ↔ x ∈ set.Ioo (-∞) (-7) ∪ set.Ioo (-7) 7 := by
  sorry

end inequality_solution_l140_140372


namespace dad_strawberries_final_weight_l140_140603

variable {M D : ℕ}

theorem dad_strawberries_final_weight :
  M + D = 22 →
  36 - M + 30 + D = D' →
  D' = 46 :=
by
  intros h h1
  sorry

end dad_strawberries_final_weight_l140_140603


namespace right_triangle_inequality_l140_140092

-- Definition of a right-angled triangle with given legs a, b, hypotenuse c, and altitude h_c to the hypotenuse
variables {a b c h_c : ℝ}

-- Right-angled triangle condition definition with angle at C is right
def right_angled_triangle (a b c : ℝ) : Prop :=
  ∃ (a b c : ℝ), c^2 = a^2 + b^2

-- Definition of the altitude to the hypotenuse
def altitude_to_hypotenuse (a b c h_c : ℝ) : Prop :=
  h_c = (a * b) / c

-- Theorem statement to prove the inequality for any right-angled triangle
theorem right_triangle_inequality (a b c h_c : ℝ) (h1 : right_angled_triangle a b c) (h2 : altitude_to_hypotenuse a b c h_c) : 
  a + b < c + h_c :=
by
  sorry

end right_triangle_inequality_l140_140092


namespace problem_curves_l140_140098

theorem problem_curves (x y : ℝ) : 
  ((x * (x^2 + y^2 - 4) = 0 → (x = 0 ∨ x^2 + y^2 = 4)) ∧
  (x^2 + (x^2 + y^2 - 4)^2 = 0 → ((x = 0 ∧ y = -2) ∨ (x = 0 ∧ y = 2)))) :=
by
  sorry -- proof to be filled in later

end problem_curves_l140_140098


namespace intersection_M_N_l140_140332

def M : Set ℝ := { x | -2 ≤ x ∧ x < 2 }
def N : Set ℝ := { x | x ≥ -2 }

theorem intersection_M_N : M ∩ N = { x | -2 ≤ x ∧ x < 2 } := by
  sorry

end intersection_M_N_l140_140332


namespace sum_of_acute_angles_l140_140576

theorem sum_of_acute_angles (α β : ℝ) (t : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_tanα : Real.tan α = 2 / t) (h_tanβ : Real.tan β = t / 15)
  (h_min : 10 * Real.tan α + 3 * Real.tan β = 4) :
  α + β = π / 4 :=
sorry

end sum_of_acute_angles_l140_140576


namespace wire_length_ratio_l140_140676

def bonnie_wire_length : ℕ := 12 * 8
def roark_prism_volume : ℕ := 2^3
def bonnie_prism_volume : ℕ := 8^3
def number_of_roark_prisms : ℕ := bonnie_prism_volume / roark_prism_volume
def roark_wire_per_prism : ℕ := 12 * 2
def total_roark_wire_length : ℕ := number_of_roark_prisms * roark_wire_per_prism

theorem wire_length_ratio : (96 : ℚ) / (1536 : ℚ) = 1 / 16 :=
by
  sorry

end wire_length_ratio_l140_140676


namespace smallest_integer_y_l140_140125

theorem smallest_integer_y (y : ℤ) (h: y < 3 * y - 15) : y ≥ 8 :=
by sorry

end smallest_integer_y_l140_140125


namespace sum_of_numbers_l140_140623

theorem sum_of_numbers {a b c : ℝ} (h1 : b = 7) (h2 : (a + b + c) / 3 = a + 8) (h3 : (a + b + c) / 3 = c - 20) : a + b + c = 57 :=
sorry

end sum_of_numbers_l140_140623


namespace probability_four_of_eight_show_three_l140_140966

def probability_exactly_four_show_three : ℚ :=
  let num_ways := Nat.choose 8 4
  let prob_four_threes := (1 / 6) ^ 4
  let prob_four_not_threes := (5 / 6) ^ 4
  (num_ways * prob_four_threes * prob_four_not_threes)

theorem probability_four_of_eight_show_three :
  probability_exactly_four_show_three = 43750 / 1679616 :=
by 
  sorry

end probability_four_of_eight_show_three_l140_140966


namespace loss_percentage_second_venture_l140_140160

theorem loss_percentage_second_venture 
  (investment_total : ℝ)
  (investment_each : ℝ)
  (profit_percentage_first_venture : ℝ)
  (total_return_percentage : ℝ)
  (L : ℝ) 
  (H1 : investment_total = 25000) 
  (H2 : investment_each = 16250)
  (H3 : profit_percentage_first_venture = 0.15)
  (H4 : total_return_percentage = 0.08)
  (H5 : (investment_total * total_return_percentage) = ((investment_each * profit_percentage_first_venture) - (investment_each * L))) :
  L = 0.0269 := 
by
  sorry

end loss_percentage_second_venture_l140_140160


namespace expected_value_lin_transform_l140_140702

noncomputable def E_5xi_plus_1 : ℝ := 3

namespace DefectiveItems

variables (ξ : ℕ) (σ : Type) [Fintype σ] [Uniform_Laws σ] (genuine defective : Finset σ)
variable [decidable_eq σ]

axiom batch_items : ∃ (genuine defective : Finset σ), 
  genuine.card = 13 ∧ defective.card = 2

axiom draw_items : 
  ∀ (s : Finset σ), s.card = 3 → 
  ∃ (ξ : ℕ), ξ = s.filter (λ x, x ∈ defective).card

theorem expected_value_lin_transform (ξ : ℕ) (E : ℕ → ℝ) :
  E (5 * ξ + 1) = 3 :=
sorry

end DefectiveItems

end expected_value_lin_transform_l140_140702


namespace number_of_points_on_parabola_l140_140437

theorem number_of_points_on_parabola :
  let f (x : ℕ) := - (x * x) / 3 + 13 * x + 42
  (finset.filter (λ x, f x ∈ finset.range 43) (finset.range 42)).card = 13 :=
by sorry

end number_of_points_on_parabola_l140_140437


namespace number_division_l140_140526

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l140_140526


namespace other_root_of_quadratic_l140_140471

theorem other_root_of_quadratic (a b : ℝ) (h₀ : a ≠ 0) (h₁ : ∃ x : ℝ, (a * x ^ 2 = b) ∧ (x = 2)) : 
  ∃ m : ℝ, (a * m ^ 2 = b) ∧ (m = -2) := 
sorry

end other_root_of_quadratic_l140_140471


namespace MN_equal_l140_140864

def M : Set ℝ := {x | ∃ (m : ℤ), x = Real.sin ((2 * m - 3) * Real.pi / 6)}
def N : Set ℝ := {y | ∃ (n : ℤ), y = Real.cos (n * Real.pi / 3)}

theorem MN_equal : M = N := by
  sorry

end MN_equal_l140_140864


namespace necessary_but_not_sufficient_condition_not_sufficient_condition_l140_140195

theorem necessary_but_not_sufficient_condition (x y : ℝ) (h : x > 0) : 
  (x > |y|) → (x > y) :=
by
  sorry

theorem not_sufficient_condition (x y : ℝ) (h : x > 0) :
  ¬ ((x > y) → (x > |y|)) :=
by
  sorry

end necessary_but_not_sufficient_condition_not_sufficient_condition_l140_140195


namespace yen_exchange_rate_l140_140410

theorem yen_exchange_rate (yen_per_dollar : ℕ) (dollars : ℕ) (y : ℕ) (h1 : yen_per_dollar = 120) (h2 : dollars = 10) : y = 1200 :=
by
  have h3 : y = yen_per_dollar * dollars := by sorry
  rw [h1, h2] at h3
  exact h3

end yen_exchange_rate_l140_140410


namespace ratio_of_areas_l140_140018

theorem ratio_of_areas
  (s: ℝ) (h₁: s > 0)
  (large_square_area: ℝ)
  (inscribed_square_area: ℝ)
  (harea₁: large_square_area = s * s)
  (harea₂: inscribed_square_area = (s / 2) * (s / 2)) :
  inscribed_square_area / large_square_area = 1 / 4 :=
by
  sorry

end ratio_of_areas_l140_140018


namespace part1_part2_l140_140333

variables {R : Type} [LinearOrderedField R]

def setA := {x : R | -1 < x ∧ x ≤ 5}
def setB (m : R) := {x : R | x^2 - 2*x - m < 0}
def complementB (m : R) := {x : R | x ≤ -1 ∨ x ≥ 3}

theorem part1 : 
  {x : R | 6 / (x + 1) ≥ 1} = setA := 
by 
  sorry

theorem part2 (m : R) (hm : m = 3) : 
  setA ∩ complementB m = {x : R | 3 ≤ x ∧ x ≤ 5} := 
by 
  sorry

end part1_part2_l140_140333


namespace fraction_eval_l140_140176

theorem fraction_eval :
  (8 : ℝ) / (4 * 25) = (0.8 : ℝ) / (0.4 * 25) :=
sorry

end fraction_eval_l140_140176


namespace sum_of_coefficients_eq_3125_l140_140584

theorem sum_of_coefficients_eq_3125 
  {b_5 b_4 b_3 b_2 b_1 b_0 : ℤ}
  (h : (2 * x + 3)^5 = b_5 * x^5 + b_4 * x^4 + b_3 * x^3 + b_2 * x^2 + b_1 * x + b_0) :
  b_5 + b_4 + b_3 + b_2 + b_1 + b_0 = 3125 := 
by 
  sorry

end sum_of_coefficients_eq_3125_l140_140584


namespace percentage_increase_l140_140589

-- defining the given values
def Z := 150
def total := 555
def x_from_y (Y : ℝ) := 1.25 * Y

-- defining the condition that x gets 25% more than y and z out of 555 is Rs. 150
def condition1 (X Y : ℝ) := X = x_from_y Y
def condition2 (X Y : ℝ) := X + Y + Z = total

-- theorem to prove
theorem percentage_increase (Y : ℝ) :
  condition1 (x_from_y Y) Y →
  condition2 (x_from_y Y) Y →
  ((Y - Z) / Z) * 100 = 20 :=
by
  sorry

end percentage_increase_l140_140589


namespace max_remainder_209_lt_120_l140_140537

theorem max_remainder_209_lt_120 : 
  ∃ n : ℕ, n < 120 ∧ (209 % n = 104) := 
sorry

end max_remainder_209_lt_120_l140_140537


namespace Jessica_has_3_dozens_l140_140237

variable (j : ℕ)

def Sandy_red_marbles (j : ℕ) : ℕ := 4 * j * 12  

theorem Jessica_has_3_dozens {j : ℕ} : Sandy_red_marbles j = 144 → j = 3 := by
  intros h
  sorry

end Jessica_has_3_dozens_l140_140237


namespace division_by_fraction_l140_140830

theorem division_by_fraction :
  (5 / (8 / 15) : ℚ) = 75 / 8 :=
by
  sorry

end division_by_fraction_l140_140830


namespace simultaneous_eq_solution_l140_140027

theorem simultaneous_eq_solution (n : ℝ) (hn : n ≠ 1 / 2) : 
  ∃ (x y : ℝ), (y = (3 * n + 1) * x + 2) ∧ (y = (5 * n - 2) * x + 5) := 
sorry

end simultaneous_eq_solution_l140_140027


namespace distance_between_A_and_B_is_40_l140_140274

theorem distance_between_A_and_B_is_40
  (v1 v2 : ℝ)
  (h1 : ∃ t: ℝ, t = (40 / 2) / v1 ∧ t = (40 - 24) / v2)
  (h2 : ∃ t: ℝ, t = (40 - 15) / v1 ∧ t = 40 / (2 * v2)) :
  40 = 40 := by
  sorry

end distance_between_A_and_B_is_40_l140_140274


namespace f_expression_when_x_gt_1_l140_140769

variable (f : ℝ → ℝ)

-- conditions
def f_even : Prop := ∀ x, f (x + 1) = f (-x + 1)
def f_defn_when_x_lt_1 : Prop := ∀ x, x < 1 → f x = x ^ 2 + 1

-- theorem to prove
theorem f_expression_when_x_gt_1 (h_even : f_even f) (h_defn : f_defn_when_x_lt_1 f) : 
  ∀ x, x > 1 → f x = x ^ 2 - 4 * x + 5 := 
by
  sorry

end f_expression_when_x_gt_1_l140_140769


namespace fraction_of_milk_in_cup1_l140_140241

def initial_tea_cup1 : ℚ := 6
def initial_milk_cup2 : ℚ := 6

def tea_transferred_step2 : ℚ := initial_tea_cup1 / 3
def tea_cup1_after_step2 : ℚ := initial_tea_cup1 - tea_transferred_step2
def total_cup2_after_step2 : ℚ := initial_milk_cup2 + tea_transferred_step2

def mixture_transfer_step3 : ℚ := total_cup2_after_step2 / 2
def tea_ratio_cup2 : ℚ := tea_transferred_step2 / total_cup2_after_step2
def milk_ratio_cup2 : ℚ := initial_milk_cup2 / total_cup2_after_step2
def tea_transferred_step3 : ℚ := mixture_transfer_step3 * tea_ratio_cup2
def milk_transferred_step3 : ℚ := mixture_transfer_step3 * milk_ratio_cup2

def tea_cup1_after_step3 : ℚ := tea_cup1_after_step2 + tea_transferred_step3
def milk_cup1_after_step3 : ℚ := milk_transferred_step3

def mixture_transfer_step4 : ℚ := (tea_cup1_after_step3 + milk_cup1_after_step3) / 4
def tea_ratio_cup1_step4 : ℚ := tea_cup1_after_step3 / (tea_cup1_after_step3 + milk_cup1_after_step3)
def milk_ratio_cup1_step4 : ℚ := milk_cup1_after_step3 / (tea_cup1_after_step3 + milk_cup1_after_step3)

def tea_transferred_step4 : ℚ := mixture_transfer_step4 * tea_ratio_cup1_step4
def milk_transferred_step4 : ℚ := mixture_transfer_step4 * milk_ratio_cup1_step4

def final_tea_cup1 : ℚ := tea_cup1_after_step3 - tea_transferred_step4
def final_milk_cup1 : ℚ := milk_cup1_after_step3 - milk_transferred_step4
def final_total_liquid_cup1 : ℚ := final_tea_cup1 + final_milk_cup1

theorem fraction_of_milk_in_cup1 : final_milk_cup1 / final_total_liquid_cup1 = 3/8 := by
  sorry

end fraction_of_milk_in_cup1_l140_140241


namespace geologists_probability_l140_140076

theorem geologists_probability :
  let r := 4 -- speed of each geologist in km/h
  let d := 6 -- distance in km
  let sectors := 8 -- number of sectors (roads)
  let total_outcomes := sectors * sectors
  let favorable_outcomes := sectors * 3 -- when distance > 6 km

  -- Calculating probability
  let P := (favorable_outcomes: ℝ) / (total_outcomes: ℝ)

  P = 0.375 :=
by
  sorry

end geologists_probability_l140_140076


namespace number_of_real_roots_l140_140025

theorem number_of_real_roots (a : ℝ) :
  (|a| < (2 * Real.sqrt 3 / 9) → ∃ x y z : ℝ, x^3 - x - a = 0 ∧ y^3 - y - a = 0 ∧ z^3 - z - a = 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x) ∧
  (|a| = (2 * Real.sqrt 3 / 9) → ∃ x y : ℝ, x^3 - x - a = 0 ∧ y^3 - y - a = 0 ∧ x = y) ∧
  (|a| > (2 * Real.sqrt 3 / 9) → ∃ x : ℝ, x^3 - x - a = 0 ∧ ∀ y : ℝ, y^3 - y - a ≠ 0 ∨ y = x) :=
sorry

end number_of_real_roots_l140_140025


namespace inequality_true_l140_140059

theorem inequality_true (a b : ℝ) (h : a > b) (x : ℝ) : 
  (a > b) → (x ≥ 0) → (a / ((2^x) + 1) > b / ((2^x) + 1)) :=
by 
  sorry

end inequality_true_l140_140059


namespace max_value_of_expression_l140_140614

variables (x y : ℝ)

theorem max_value_of_expression (hx : 0 < x) (hy : 0 < y) (h : x^2 - 2*x*y + 3*y^2 = 12) : x^2 + 2*x*y + 3*y^2 ≤ 24 + 24*sqrt 3 :=
sorry

end max_value_of_expression_l140_140614


namespace cube_surface_area_increase_l140_140926

theorem cube_surface_area_increase (s : ℝ) : 
    let initial_area := 6 * s^2
    let new_edge := 1.3 * s
    let new_area := 6 * (new_edge)^2
    let incr_area := new_area - initial_area
    let percentage_increase := (incr_area / initial_area) * 100
    percentage_increase = 69 :=
by
  let initial_area := 6 * s^2
  let new_edge := 1.3 * s
  let new_area := 6 * (new_edge)^2
  let incr_area := new_area - initial_area
  let percentage_increase := (incr_area / initial_area) * 100
  sorry

end cube_surface_area_increase_l140_140926


namespace animal_population_l140_140628

theorem animal_population
  (number_of_lions : ℕ)
  (number_of_leopards : ℕ)
  (number_of_elephants : ℕ)
  (h1 : number_of_lions = 200)
  (h2 : number_of_lions = 2 * number_of_leopards)
  (h3 : number_of_elephants = (number_of_lions + number_of_leopards) / 2) :
  number_of_lions + number_of_leopards + number_of_elephants = 450 :=
sorry

end animal_population_l140_140628


namespace find_a_l140_140862

noncomputable def f (x : ℝ) (a : ℝ) := (2 / x) - 2 + 2 * a * Real.log x

theorem find_a (a : ℝ) (h : ∃ x ∈ Set.Icc (1/2 : ℝ) 2, f x a = 0) : a = 1 := by
  sorry

end find_a_l140_140862


namespace dragon_jewels_l140_140150

theorem dragon_jewels (x : ℕ) (h1 : (x / 3 = 6)) : x + 6 = 24 :=
sorry

end dragon_jewels_l140_140150


namespace equation_correct_l140_140042

variable (x y : ℝ)

-- Define the conditions
def condition1 : Prop := (x + y) / 3 = 1.888888888888889
def condition2 : Prop := 2 * x + y = 7

-- Prove the required equation under given conditions
theorem equation_correct : condition1 x y → condition2 x y → (x + y) = 5.666666666666667 := by
  intros _ _
  sorry

end equation_correct_l140_140042


namespace loss_equals_cost_price_of_some_balls_l140_140494

-- Conditions
def cost_price_per_ball := 60
def selling_price_for_17_balls := 720
def number_of_balls := 17

-- Calculations
def total_cost_price := number_of_balls * cost_price_per_ball
def loss := total_cost_price - selling_price_for_17_balls

-- Proof statement
theorem loss_equals_cost_price_of_some_balls : (loss / cost_price_per_ball) = 5 :=
by
  -- Proof would go here
  sorry

end loss_equals_cost_price_of_some_balls_l140_140494


namespace arithmetic_sequence_sum_l140_140708

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (h1 : S 4 = 3) (h2 : S 8 = 7) : S 12 = 12 :=
by
  -- placeholder for the proof, details omitted
  sorry

end arithmetic_sequence_sum_l140_140708


namespace algebra_problem_l140_140987

-- Definition of variable y
variable (y : ℝ)

-- Given the condition
axiom h : 2 * y^2 + 3 * y + 7 = 8

-- We need to prove that 4 * y^2 + 6 * y - 9 = -7 given the condition
theorem algebra_problem : 4 * y^2 + 6 * y - 9 = -7 :=
by sorry

end algebra_problem_l140_140987


namespace fifth_boy_pays_l140_140562

def problem_conditions (a b c d e : ℝ) : Prop :=
  d = 20 ∧
  a = (1 / 3) * (b + c + d + e) ∧
  b = (1 / 4) * (a + c + d + e) ∧
  c = (1 / 5) * (a + b + d + e) ∧
  a + b + c + d + e = 120 

theorem fifth_boy_pays (a b c d e : ℝ) (h : problem_conditions a b c d e) : 
  e = 35 :=
sorry

end fifth_boy_pays_l140_140562


namespace shortest_minor_arc_line_equation_l140_140567

noncomputable def pointM : (ℝ × ℝ) := (1, -2)
noncomputable def circleC (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

theorem shortest_minor_arc_line_equation :
  (∀ x y : ℝ, (x + 2 * y + 3 = 0) ↔ 
  ((x = 1 ∧ y = -2) ∨ ∃ (k_l : ℝ), (k_l * (2) = -1) ∧ (y + 2 = -k_l * (x - 1)))) :=
sorry

end shortest_minor_arc_line_equation_l140_140567


namespace systematic_sampling_second_group_l140_140638

theorem systematic_sampling_second_group
    (N : ℕ) (n : ℕ) (k : ℕ := N / n)
    (number_from_16th_group : ℕ)
    (number_from_1st_group : ℕ := number_from_16th_group - 15 * k)
    (number_from_2nd_group : ℕ := number_from_1st_group + k) :
    N = 160 → n = 20 → number_from_16th_group = 123 → number_from_2nd_group = 11 :=
by
  sorry

end systematic_sampling_second_group_l140_140638


namespace graph_of_equation_is_shifted_hyperbola_l140_140717

-- Definitions
def given_equation (x y : ℝ) : Prop := x^2 - 4*y^2 - 2*x = 0

-- Theorem statement
theorem graph_of_equation_is_shifted_hyperbola :
  ∀ x y : ℝ, given_equation x y = ((x - 1)^2 = 1 + 4*y^2) :=
by
  sorry

end graph_of_equation_is_shifted_hyperbola_l140_140717


namespace smallest_x_l140_140188

theorem smallest_x (x : ℕ) : (x + 3457) % 15 = 1537 % 15 → x = 15 :=
by
  sorry

end smallest_x_l140_140188


namespace total_cube_volume_l140_140421

theorem total_cube_volume 
  (carl_cubes : ℕ)
  (carl_cube_side : ℕ)
  (kate_cubes : ℕ)
  (kate_cube_side : ℕ)
  (hcarl : carl_cubes = 4)
  (hcarl_side : carl_cube_side = 3)
  (hkate : kate_cubes = 6)
  (hkate_side : kate_cube_side = 4) :
  (carl_cubes * carl_cube_side ^ 3) + (kate_cubes * kate_cube_side ^ 3) = 492 :=
by
  sorry

end total_cube_volume_l140_140421


namespace number_division_l140_140528

theorem number_division (x : ℕ) (h : x / 5 = 80 + x / 6) : x = 2400 :=
sorry

end number_division_l140_140528


namespace john_pool_cleanings_per_month_l140_140740

noncomputable def tip_percent : ℝ := 0.10
noncomputable def cost_per_cleaning : ℝ := 150
noncomputable def total_cost_per_cleaning : ℝ := cost_per_cleaning + (tip_percent * cost_per_cleaning)
noncomputable def chemical_cost_bi_monthly : ℝ := 200
noncomputable def monthly_chemical_cost : ℝ := 2 * chemical_cost_bi_monthly
noncomputable def total_monthly_pool_cost : ℝ := 2050
noncomputable def total_cleaning_cost : ℝ := total_monthly_pool_cost - monthly_chemical_cost

theorem john_pool_cleanings_per_month : total_cleaning_cost / total_cost_per_cleaning = 10 := by
  sorry

end john_pool_cleanings_per_month_l140_140740


namespace cos_double_angle_l140_140051

theorem cos_double_angle
  {x : ℝ}
  (h : Real.sin x = -2 / 3) :
  Real.cos (2 * x) = 1 / 9 :=
by
  sorry

end cos_double_angle_l140_140051


namespace sum_of_eight_numbers_l140_140587

theorem sum_of_eight_numbers (avg : ℝ) (num_of_items : ℕ) (h_avg : avg = 5.3) (h_items : num_of_items = 8) :
  avg * num_of_items = 42.4 :=
by
  sorry

end sum_of_eight_numbers_l140_140587


namespace prob_correct_l140_140385

def distances : List (Nat × Nat × Nat) := [
  (0, 1, 6500), (0, 2, 6700), (0, 3, 6100), (0, 4, 8600),
  (1, 2, 11800), (1, 3, 6200), (1, 4, 7800),
  (2, 3, 7300), (2, 4, 4900),
  (3, 4, 3500)
]

def city_pairs := Finset.univ.sublistsLen 2

noncomputable def distance_less_than_8000_count : ℕ :=
  distances.countp (λ ⟨_, _, d⟩ => d < 8000)

def total_pairs_count : ℕ :=
  city_pairs.card

def probability : ℚ :=
  distance_less_than_8000_count / total_pairs_count

theorem prob_correct : probability = 3 / 5 := by
  sorry

end prob_correct_l140_140385


namespace find_k_l140_140574

theorem find_k (k : ℝ) : 
  (1 : ℝ)^2 + k * 1 - 3 = 0 → k = 2 :=
by
  intro h
  sorry

end find_k_l140_140574


namespace david_chemistry_marks_l140_140844

theorem david_chemistry_marks (marks_english marks_math marks_physics marks_biology : ℝ)
  (average_marks: ℝ) (marks_english_val: marks_english = 72) (marks_math_val: marks_math = 45)
  (marks_physics_val: marks_physics = 72) (marks_biology_val: marks_biology = 75)
  (average_marks_val: average_marks = 68.2) : 
  ∃ marks_chemistry : ℝ, (marks_english + marks_math + marks_physics + marks_biology + marks_chemistry) / 5 = average_marks ∧ 
    marks_chemistry = 77 := 
by
  sorry

end david_chemistry_marks_l140_140844


namespace jessies_initial_weight_l140_140161

-- Definitions based on the conditions
def weight_lost : ℕ := 126
def current_weight : ℕ := 66

-- The statement to prove
theorem jessies_initial_weight :
  (weight_lost + current_weight = 192) :=
by 
  sorry

end jessies_initial_weight_l140_140161


namespace hyperbola_eccentricity_l140_140899

theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 : ℝ) = 2 →
  a^2 = 2 * b^2 →
  (c : ℝ) = Real.sqrt (a^2 + b^2) →
  Real.sqrt (a^2 + b^2) = Real.sqrt (3 / 2 * a^2) →
  (e : ℝ) = c / a →
  e = Real.sqrt (6) / 2 :=
by
  sorry

end hyperbola_eccentricity_l140_140899


namespace fred_current_dimes_l140_140192

-- Definitions based on the conditions
def original_dimes : ℕ := 7
def borrowed_dimes : ℕ := 3

-- The theorem to prove
theorem fred_current_dimes : original_dimes - borrowed_dimes = 4 := by
  sorry

end fred_current_dimes_l140_140192


namespace find_p_of_abs_sum_roots_eq_five_l140_140846

theorem find_p_of_abs_sum_roots_eq_five (p : ℝ) : 
  (∃ x y : ℝ, x + y = -p ∧ x * y = -6 ∧ |x| + |y| = 5) → (p = 1 ∨ p = -1) := by
  sorry

end find_p_of_abs_sum_roots_eq_five_l140_140846


namespace common_tangents_l140_140507

noncomputable def circle1 := { p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 4 }
noncomputable def circle2 := { p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 2)^2 = 9 }

theorem common_tangents (h : ∀ p : ℝ × ℝ, p ∈ circle1 → p ∈ circle2) : 
  ∃ tangents : ℕ, tangents = 2 :=
sorry

end common_tangents_l140_140507


namespace find_x_plus_y_l140_140088

noncomputable def det3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

noncomputable def det2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem find_x_plus_y (x y : ℝ) (h1 : x ≠ y)
  (h2 : det3x3 2 5 10 4 x y 4 y x = 0)
  (h3 : det2x2 x y y x = 16) : x + y = 30 := by
  sorry

end find_x_plus_y_l140_140088


namespace seq_inv_an_is_arithmetic_seq_fn_over_an_has_minimum_l140_140209

-- Problem 1
theorem seq_inv_an_is_arithmetic (a : ℕ → ℝ) (h1 : a 1 = 1/2) (h2 : ∀ n, n ≥ 2 → a (n - 1) / a n = (a (n - 1) + 2) / (2 - a n)) :
  ∃ d, ∀ n, n ≥ 2 → (1 / a n) = 2 + (n - 1) * d :=
sorry

-- Problem 2
theorem seq_fn_over_an_has_minimum (a f : ℕ → ℝ) (h1 : a 1 = 1/2) (h2 : ∀ n, n ≥ 2 → a (n - 1) / a n = (a (n - 1) + 2) / (2 - a n)) (h3 : ∀ n, f n = (9 / 10) ^ n) :
  ∃ m, ∀ n, n ≠ m → f n / a n ≥ f m / a m :=
sorry

end seq_inv_an_is_arithmetic_seq_fn_over_an_has_minimum_l140_140209


namespace find_b_l140_140060

theorem find_b (b : ℚ) (h : ∃ c : ℚ, (3 * x + c)^2 = 9 * x^2 + 27 * x + b) : b = 81 / 4 := 
sorry

end find_b_l140_140060


namespace find_c_value_l140_140442

-- Given condition: x^2 + 300x + c = (x + a)^2
-- Problem statement: Prove that c = 22500 for the given conditions
theorem find_c_value (x a c : ℝ) : (x^2 + 300 * x + c = (x + 150)^2) → (c = 22500) :=
by
  intro h
  sorry

end find_c_value_l140_140442


namespace midpoints_distance_l140_140752

theorem midpoints_distance
  (A B C D M N : ℝ)
  (h1 : M = (A + C) / 2)
  (h2 : N = (B + D) / 2)
  (h3 : D - A = 68)
  (h4 : C - B = 26)
  : abs (M - N) = 21 := 
sorry

end midpoints_distance_l140_140752


namespace triangle_inequality_l140_140367

theorem triangle_inequality (a b c p S r : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b)
  (hp : p = (a + b + c) / 2)
  (hS : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (hr : r = S / p):
  1 / (p - a) ^ 2 + 1 / (p - b) ^ 2 + 1 / (p - c) ^ 2 ≥ 1 / r ^ 2 :=
sorry

end triangle_inequality_l140_140367


namespace solve_equation_1_solve_equation_2_l140_140370

theorem solve_equation_1 (x : ℝ) : x * (x - 2) + x - 2 = 0 ↔ (x = 2 ∨ x = -1) :=
by sorry

theorem solve_equation_2 (x : ℝ) : 2 * x^2 + 5 * x + 3 = 0 ↔ (x = -1 ∨ x = -3/2) :=
by sorry

end solve_equation_1_solve_equation_2_l140_140370


namespace number_of_squares_l140_140851

-- Define the conditions and the goal
theorem number_of_squares {x : ℤ} (hx0 : 0 ≤ x) (hx6 : x ≤ 6) {y : ℤ} (hy0 : -1 ≤ y) (hy : y ≤ 3 * x) :
  ∃ (n : ℕ), n = 123 :=
by 
  sorry

end number_of_squares_l140_140851


namespace previous_income_l140_140748

-- Define the conditions as Lean definitions
variables (p : ℝ) -- Mrs. Snyder's previous monthly income

-- Condition 1: Mrs. Snyder used to spend 40% of her income on rent and utilities
def rent_and_utilities_initial (p : ℝ) : ℝ := (2 * p) / 5

-- Condition 2: Her salary was increased by $600
def new_income (p : ℝ) : ℝ := p + 600

-- Condition 3: After the increase, rent and utilities account for 25% of her new income
def rent_and_utilities_new (p : ℝ) : ℝ := (new_income p) / 4

-- Theorem: Proving that Mrs. Snyder's previous monthly income was $1000
theorem previous_income : (2 * p) / 5 = (new_income p) / 4 → p = 1000 :=
begin
  -- By mathlib, sorry as placeholder for proof
  sorry
end

end previous_income_l140_140748


namespace greatest_possible_y_l140_140285

theorem greatest_possible_y (y : ℕ) (h1 : (y^4 / y^2) < 18) : y ≤ 4 := 
  sorry -- Proof to be filled in later

end greatest_possible_y_l140_140285


namespace absolute_difference_volumes_l140_140823

/-- The absolute difference in volumes of the cylindrical tubes formed by Amy and Carlos' papers. -/
theorem absolute_difference_volumes :
  let h_A := 12
  let C_A := 10
  let r_A := C_A / (2 * Real.pi)
  let V_A := Real.pi * r_A^2 * h_A
  let h_C := 8
  let C_C := 14
  let r_C := C_C / (2 * Real.pi)
  let V_C := Real.pi * r_C^2 * h_C
  abs (V_C - V_A) = 92 / Real.pi :=
by
  sorry

end absolute_difference_volumes_l140_140823


namespace total_oranges_picked_l140_140251

theorem total_oranges_picked :
  let Mary_oranges := 14
  let Jason_oranges := 41
  let Amanda_oranges := 56
  Mary_oranges + Jason_oranges + Amanda_oranges = 111 := by
    sorry

end total_oranges_picked_l140_140251


namespace max_value_of_g_l140_140173

def g : ℕ → ℕ
| n => if n < 7 then n + 7 else g (n - 3)

theorem max_value_of_g : ∀ (n : ℕ), g n ≤ 13 ∧ (∃ n0, g n0 = 13) := by
  sorry

end max_value_of_g_l140_140173


namespace find_possible_first_term_l140_140386

noncomputable def geometric_sequence_first_term (a r : ℝ) : Prop :=
  (a * r^2 = 3) ∧ (a * r^4 = 27)

theorem find_possible_first_term (a r : ℝ) (h : geometric_sequence_first_term a r) :
    a = 1 / 3 :=
by
  sorry

end find_possible_first_term_l140_140386


namespace area_gray_region_in_terms_of_pi_l140_140737

variable (r : ℝ)

theorem area_gray_region_in_terms_of_pi 
    (h1 : ∀ (r : ℝ), ∃ (outer_r : ℝ), outer_r = r + 3)
    (h2 : width_gray_region = 3)
    : ∃ (area_gray : ℝ), area_gray = π * (6 * r + 9) := 
sorry

end area_gray_region_in_terms_of_pi_l140_140737


namespace range_of_x_l140_140858

noncomputable def f : ℝ → ℝ := sorry -- Define the function f

variable (f_increasing : ∀ x y, x < y → f x < f y) -- f is increasing
variable (f_at_2 : f 2 = 0) -- f(2) = 0

theorem range_of_x (x : ℝ) : f (x - 2) > 0 ↔ x > 4 :=
by
  sorry

end range_of_x_l140_140858


namespace parabola_focus_coordinates_l140_140450

noncomputable def parabola_focus (a b : ℝ) := (0, (1 / (4 * a)) + 2)

theorem parabola_focus_coordinates (a b : ℝ) (h₀ : a ≠ 0) (h₁ : ∀ x : ℝ, abs (a * x^2 + b * x + 2) ≥ 2) :
  parabola_focus a b = (0, 2 + (1 / (4 * a))) := sorry

end parabola_focus_coordinates_l140_140450


namespace general_term_a_general_term_b_sum_c_l140_140565

-- Problem 1: General term formula for the sequence {a_n}
theorem general_term_a (a : ℕ → ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = 2 - a n) :
  ∀ n, a n = (1 / 2) ^ (n - 1) := 
sorry

-- Problem 2: General term formula for the sequence {b_n}
theorem general_term_b (b : ℕ → ℝ) (a : ℕ → ℝ) (h_b1 : b 1 = 1)
  (h_b : ∀ n, b (n + 1) = b n + a n) (h_a : ∀ n, a n = (1 / 2) ^ (n - 1)) :
  ∀ n, b n = 3 - 2 * (1 / 2) ^ (n - 1) := 
sorry

-- Problem 3: Sum of the first n terms for the sequence {c_n}
theorem sum_c (c : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h_b : ∀ n, b n = 3 - 2 * (1 / 2) ^ (n - 1)) (h_c : ∀ n, c n = n * (3 - b n)) :
  ∀ n, T n = 8 - (8 + 4 * n) * (1 / 2) ^ n := 
sorry

end general_term_a_general_term_b_sum_c_l140_140565


namespace area_of_triangle_ABC_l140_140747

noncomputable def area_of_ABC : ℝ :=
  let BE := 10 in
  let AD := 2 * BE in
  let AG := (2 / 3) * AD in
  let BG := (2 / 3) * BE in
  let area_ABG := 1 / 2 * AG * BG in
  6 * area_ABG

theorem area_of_triangle_ABC :
  let BE := 10 in
  let AD := 2 * BE in
  (AD = 20) →
  (BE = 10) →
  (AD / 2 = BE) →
  ∃ (area : ℝ), area = area_of_ABC ∧ area = 2400 / 9 :=
by
  intro BE AD AD_eq_20 BE_eq_10 AD_BE_relation
  suffices area_of_ABC = 2400 / 9 by
  use area_of_ABC
  split
  exact rfl
  exact this
  -- Proceed with complete proof (omitted)
  sorry

end area_of_triangle_ABC_l140_140747


namespace proof1_proof2_proof3_proof4_l140_140091

noncomputable def calc1 : ℝ := 3.21 - 1.05 - 1.95
noncomputable def calc2 : ℝ := 15 - (2.95 + 8.37)
noncomputable def calc3 : ℝ := 14.6 * 2 - 0.6 * 2
noncomputable def calc4 : ℝ := 0.25 * 1.25 * 32

theorem proof1 : calc1 = 0.21 := by
  sorry

theorem proof2 : calc2 = 3.68 := by
  sorry

theorem proof3 : calc3 = 28 := by
  sorry

theorem proof4 : calc4 = 10 := by
  sorry

end proof1_proof2_proof3_proof4_l140_140091


namespace sin_alpha_minus_beta_l140_140704

theorem sin_alpha_minus_beta (α β : Real) 
  (h1 : Real.sin α = 12 / 13) 
  (h2 : Real.cos β = 4 / 5)
  (hα : π / 2 ≤ α ∧ α ≤ π)
  (hβ : -π / 2 ≤ β ∧ β ≤ 0) :
  Real.sin (α - β) = 33 / 65 := 
sorry

end sin_alpha_minus_beta_l140_140704


namespace log_base_eq_l140_140130

theorem log_base_eq (x : ℝ) (h₁ : x > 0) (h₂ : x ≠ 1) : 
  (Real.log x / Real.log 4) * (Real.log 8 / Real.log x) = Real.log 8 / Real.log 4 := 
by 
  sorry

end log_base_eq_l140_140130


namespace yoongi_has_smallest_points_l140_140084

def points_jungkook : ℕ := 6 + 3
def points_yoongi : ℕ := 4
def points_yuna : ℕ := 5

theorem yoongi_has_smallest_points : points_yoongi < points_jungkook ∧ points_yoongi < points_yuna :=
by
  sorry

end yoongi_has_smallest_points_l140_140084


namespace company_percentage_increase_l140_140681

theorem company_percentage_increase (employees_jan employees_dec : ℝ) (P_increase : ℝ) 
  (h_jan : employees_jan = 391.304347826087)
  (h_dec : employees_dec = 450)
  (h_P : P_increase = 15) : 
  (employees_dec - employees_jan) / employees_jan * 100 = P_increase :=
by 
  sorry

end company_percentage_increase_l140_140681
