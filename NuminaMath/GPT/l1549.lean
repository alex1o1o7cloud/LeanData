import Mathlib

namespace NUMINAMATH_GPT_part_1_part_2_l1549_154972

noncomputable def f (a m x : ℝ) := a ^ m / x

theorem part_1 (a : ℝ) (m : ℝ) (H1 : a > 1) (H2 : ∀ x, x ∈ Set.Icc a (2*a) → f a m x ∈ Set.Icc (a^2) (a^3)) :
  a = 2 :=
sorry

theorem part_2 (t : ℝ) (s : ℝ) (H1 : ∀ x, x ∈ Set.Icc 0 s → (x + t) ^ 2 + 2 * (x + t) ≤ 3 * x) :
  s ∈ Set.Ioc 0 5 :=
sorry

end NUMINAMATH_GPT_part_1_part_2_l1549_154972


namespace NUMINAMATH_GPT_find_num_officers_l1549_154925

noncomputable def num_officers (O : ℕ) : Prop :=
  let avg_salary_all := 120
  let avg_salary_officers := 440
  let avg_salary_non_officers := 110
  let num_non_officers := 480
  let total_salary :=
    avg_salary_all * (O + num_non_officers)
  let salary_officers :=
    avg_salary_officers * O
  let salary_non_officers :=
    avg_salary_non_officers * num_non_officers
  total_salary = salary_officers + salary_non_officers

theorem find_num_officers : num_officers 15 :=
sorry

end NUMINAMATH_GPT_find_num_officers_l1549_154925


namespace NUMINAMATH_GPT_unique_functional_equation_l1549_154923

theorem unique_functional_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = x + y :=
sorry

end NUMINAMATH_GPT_unique_functional_equation_l1549_154923


namespace NUMINAMATH_GPT_factorization_sum_l1549_154959

theorem factorization_sum (a b c : ℤ) 
  (h1 : ∀ x : ℤ, x^2 + 9 * x + 20 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x^2 + 7 * x - 60 = (x + b) * (x - c)) :
  a + b + c = 21 :=
by
  sorry

end NUMINAMATH_GPT_factorization_sum_l1549_154959


namespace NUMINAMATH_GPT_find_n_from_ratio_l1549_154940

theorem find_n_from_ratio (a b n : ℕ) (h : (a + 3 * b) ^ n = 4 ^ n)
  (h_ratio : 4 ^ n / 2 ^ n = 64) : 
  n = 6 := 
by
  sorry

end NUMINAMATH_GPT_find_n_from_ratio_l1549_154940


namespace NUMINAMATH_GPT_rectangle_area_given_conditions_l1549_154970

theorem rectangle_area_given_conditions
  (l w : ℝ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by sorry

end NUMINAMATH_GPT_rectangle_area_given_conditions_l1549_154970


namespace NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l1549_154969

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n = 986 ∧ n < 1000 ∧ 17 ∣ n :=
by 
  sorry

end NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l1549_154969


namespace NUMINAMATH_GPT_find_k_l1549_154909

open Real

variables (a b : ℝ × ℝ) (k : ℝ)

def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

theorem find_k (ha : a = (1, 2)) (hb : b = (-2, 4)) (perpendicular : dot_product (k • a + b) b = 0) :
  k = - (10 / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1549_154909


namespace NUMINAMATH_GPT_total_profit_for_the_month_l1549_154950

theorem total_profit_for_the_month (mean_profit_month : ℕ) (num_days_month : ℕ)
(mean_profit_first15 : ℕ) (num_days_first15 : ℕ) 
(mean_profit_last15 : ℕ) (num_days_last15 : ℕ) 
(h1 : mean_profit_month = 350) (h2 : num_days_month = 30) 
(h3 : mean_profit_first15 = 285) (h4 : num_days_first15 = 15) 
(h5 : mean_profit_last15 = 415) (h6 : num_days_last15 = 15) : 
(mean_profit_first15 * num_days_first15 + mean_profit_last15 * num_days_last15) = 10500 := by
  sorry

end NUMINAMATH_GPT_total_profit_for_the_month_l1549_154950


namespace NUMINAMATH_GPT_dawn_hours_l1549_154946

-- Define the conditions
def pedestrian_walked_from_A_to_B (x : ℕ) : Prop :=
  x > 0

def pedestrian_walked_from_B_to_A (x : ℕ) : Prop :=
  x > 0

def met_at_noon (x : ℕ) : Prop :=
  x > 0

def arrived_at_B_at_4pm (x : ℕ) : Prop :=
  x > 0

def arrived_at_A_at_9pm (x : ℕ) : Prop :=
  x > 0

-- Define the theorem to prove
theorem dawn_hours (x : ℕ) :
  pedestrian_walked_from_A_to_B x ∧ 
  pedestrian_walked_from_B_to_A x ∧
  met_at_noon x ∧ 
  arrived_at_B_at_4pm x ∧ 
  arrived_at_A_at_9pm x → 
  x = 6 := 
sorry

end NUMINAMATH_GPT_dawn_hours_l1549_154946


namespace NUMINAMATH_GPT_remainder_13_plus_y_l1549_154971

theorem remainder_13_plus_y :
  (∃ y : ℕ, (0 < y) ∧ (7 * y ≡ 1 [MOD 31])) → (∃ y : ℕ, (13 + y ≡ 22 [MOD 31])) :=
by 
  sorry

end NUMINAMATH_GPT_remainder_13_plus_y_l1549_154971


namespace NUMINAMATH_GPT_max_value_sqrt_expression_l1549_154978

noncomputable def expression_max_value (a b: ℝ) : ℝ :=
  Real.sqrt (a * b) + Real.sqrt ((1 - a) * (1 - b))

theorem max_value_sqrt_expression : 
  ∀ (a b : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 → expression_max_value a b ≤ 1 :=
by
  intros a b h
  sorry

end NUMINAMATH_GPT_max_value_sqrt_expression_l1549_154978


namespace NUMINAMATH_GPT_cube_side_length_of_paint_cost_l1549_154915

theorem cube_side_length_of_paint_cost (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (total_cost : ℝ) (side_length : ℝ) :
  cost_per_kg = 20 ∧ coverage_per_kg = 15 ∧ total_cost = 200 →
  6 * side_length ^ 2 = (total_cost / cost_per_kg) * coverage_per_kg →
  side_length = 5 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_cube_side_length_of_paint_cost_l1549_154915


namespace NUMINAMATH_GPT_book_distribution_methods_l1549_154944

theorem book_distribution_methods :
  let novels := 2
  let picture_books := 2
  let students := 3
  (number_ways : ℕ) = 12 :=
by
  sorry

end NUMINAMATH_GPT_book_distribution_methods_l1549_154944


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1549_154900

variable (a x : ℝ)

theorem solution_set_of_inequality (h : 0 < a ∧ a < 1) :
  (a - x) * (x - (1/a)) > 0 ↔ a < x ∧ x < 1/a :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1549_154900


namespace NUMINAMATH_GPT_solve_system_of_equations_l1549_154942

theorem solve_system_of_equations (m b : ℤ) 
  (h1 : 3 * m + b = 11)
  (h2 : -4 * m - b = 11) : 
  m = -22 ∧ b = 77 :=
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1549_154942


namespace NUMINAMATH_GPT_lily_milk_quantity_l1549_154937

theorem lily_milk_quantity :
  let init_gallons := (5 : ℝ)
  let given_away := (18 / 4 : ℝ)
  let received_back := (7 / 4 : ℝ)
  init_gallons - given_away + received_back = 2 + 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_lily_milk_quantity_l1549_154937


namespace NUMINAMATH_GPT_original_numbers_l1549_154964

theorem original_numbers (a b c d : ℕ) (x : ℕ)
  (h1 : a + b + c + d = 45)
  (h2 : a + 2 = x)
  (h3 : b - 2 = x)
  (h4 : 2 * c = x)
  (h5 : d / 2 = x) : 
  (a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20) :=
sorry

end NUMINAMATH_GPT_original_numbers_l1549_154964


namespace NUMINAMATH_GPT_cannot_lie_on_line_l1549_154992

theorem cannot_lie_on_line (m b : ℝ) (h : m * b < 0) : ¬ (0 = m * (-2022) + b) := 
  by
  sorry

end NUMINAMATH_GPT_cannot_lie_on_line_l1549_154992


namespace NUMINAMATH_GPT_surface_area_of_box_l1549_154996

def cube_edge_length : ℕ := 1
def cubes_required : ℕ := 12

theorem surface_area_of_box (l w h : ℕ) (h1 : l * w * h = cubes_required / cube_edge_length ^ 3) :
  (2 * (l * w + w * h + h * l) = 32 ∨ 2 * (l * w + w * h + h * l) = 38 ∨ 2 * (l * w + w * h + h * l) = 40) :=
  sorry

end NUMINAMATH_GPT_surface_area_of_box_l1549_154996


namespace NUMINAMATH_GPT_determine_values_a_b_l1549_154979

theorem determine_values_a_b (a b x : ℝ) (h₁ : x > 1)
  (h₂ : 3 * (Real.log x / Real.log a)^2 + 5 * (Real.log x / Real.log b)^2 = (10 * (Real.log x)^2) / (Real.log a + Real.log b)) :
  b = a ^ ((5 + Real.sqrt 10) / 3) ∨ b = a ^ ((5 - Real.sqrt 10) / 3) :=
by sorry

end NUMINAMATH_GPT_determine_values_a_b_l1549_154979


namespace NUMINAMATH_GPT_cape_may_shark_sightings_l1549_154994

def total_shark_sightings (D C : ℕ) : Prop :=
  D + C = 40

def cape_may_sightings (D C : ℕ) : Prop :=
  C = 2 * D - 8

theorem cape_may_shark_sightings : 
  ∃ (C D : ℕ), total_shark_sightings D C ∧ cape_may_sightings D C ∧ C = 24 :=
by
  sorry

end NUMINAMATH_GPT_cape_may_shark_sightings_l1549_154994


namespace NUMINAMATH_GPT_domain_of_g_l1549_154928

-- Define the function f and specify the domain of f(x+1)
def f : ℝ → ℝ := sorry
def domain_f_x_plus_1 : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3} -- Domain of f(x+1) is [-1, 3]

-- Define the definition of the function g where g(x) = f(x^2)
def g (x : ℝ) : ℝ := f (x^2)

-- Prove that the domain of g(x) is [-2, 2]
theorem domain_of_g : {x | -2 ≤ x ∧ x ≤ 2} = {x | ∃ (y : ℝ), (0 ≤ y ∧ y ≤ 4) ∧ (x = y ∨ x = -y)} :=
by 
  sorry

end NUMINAMATH_GPT_domain_of_g_l1549_154928


namespace NUMINAMATH_GPT_floor_add_self_eq_14_5_iff_r_eq_7_5_l1549_154904

theorem floor_add_self_eq_14_5_iff_r_eq_7_5 (r : ℝ) : 
  (⌊r⌋ + r = 14.5) ↔ r = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_floor_add_self_eq_14_5_iff_r_eq_7_5_l1549_154904


namespace NUMINAMATH_GPT_remainder_of_173_mod_13_l1549_154958

theorem remainder_of_173_mod_13 : ∀ (m : ℤ), 173 = 8 * m + 5 → 173 < 180 → 173 % 13 = 4 :=
by
  intro m hm h
  sorry

end NUMINAMATH_GPT_remainder_of_173_mod_13_l1549_154958


namespace NUMINAMATH_GPT_expected_value_of_groups_l1549_154948

noncomputable def expectedNumberOfGroups (k m : ℕ) : ℝ :=
  1 + (2 * k * m) / (k + m)

theorem expected_value_of_groups (k m : ℕ) :
  k > 0 → m > 0 → expectedNumberOfGroups k m = 1 + 2 * k * m / (k + m) :=
by
  intros
  unfold expectedNumberOfGroups
  sorry

end NUMINAMATH_GPT_expected_value_of_groups_l1549_154948


namespace NUMINAMATH_GPT_find_the_added_number_l1549_154953

theorem find_the_added_number (n : ℤ) : (1 + n) / (3 + n) = 3 / 4 → n = 5 :=
  sorry

end NUMINAMATH_GPT_find_the_added_number_l1549_154953


namespace NUMINAMATH_GPT_find_t_value_l1549_154951

theorem find_t_value (t : ℝ) (h1 : (t - 6) * (2 * t - 5) = (2 * t - 8) * (t - 5)) : t = 10 :=
sorry

end NUMINAMATH_GPT_find_t_value_l1549_154951


namespace NUMINAMATH_GPT_alley_width_l1549_154911

noncomputable def calculate_width (l k h : ℝ) : ℝ :=
  l / 2

theorem alley_width (k h l w : ℝ) (h1 : k = (l * (Real.sin (Real.pi / 3)))) (h2 : h = (l * (Real.sin (Real.pi / 6)))) :
  w = calculate_width l k h :=
by
  sorry

end NUMINAMATH_GPT_alley_width_l1549_154911


namespace NUMINAMATH_GPT_apples_left_l1549_154931

theorem apples_left (initial_apples : ℕ) (ricki_removes : ℕ) (samson_removes : ℕ) 
  (h1 : initial_apples = 74) 
  (h2 : ricki_removes = 14) 
  (h3 : samson_removes = 2 * ricki_removes) : 
  initial_apples - (ricki_removes + samson_removes) = 32 := 
by
  sorry

end NUMINAMATH_GPT_apples_left_l1549_154931


namespace NUMINAMATH_GPT_parallel_lines_l1549_154973

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, (ax + 2 * y + a = 0 ∧ 3 * a * x + (a - 1) * y + 7 = 0) →
    - (a / 2) = - (3 * a / (a - 1))) ↔ (a = 0 ∨ a = 7) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_l1549_154973


namespace NUMINAMATH_GPT_find_x_l1549_154918

theorem find_x :
  ∀ x : ℝ, (7 / (Real.sqrt (x - 5) - 10) + 2 / (Real.sqrt (x - 5) - 3) +
  8 / (Real.sqrt (x - 5) + 3) + 13 / (Real.sqrt (x - 5) + 10) = 0) →
  x = 1486 / 225 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1549_154918


namespace NUMINAMATH_GPT_trapezoid_base_count_l1549_154917

theorem trapezoid_base_count (A h : ℕ) (multiple : ℕ) (bases_sum pairs_count : ℕ) : 
  A = 1800 ∧ h = 60 ∧ multiple = 10 ∧ pairs_count = 4 ∧ 
  bases_sum = (A / (1/2 * h)) / multiple → pairs_count > 3 := 
by 
  sorry

end NUMINAMATH_GPT_trapezoid_base_count_l1549_154917


namespace NUMINAMATH_GPT_necessarily_positive_l1549_154910

theorem necessarily_positive (x y z : ℝ) (hx : -1 < x ∧ x < 1) 
                      (hy : -1 < y ∧ y < 0) 
                      (hz : 1 < z ∧ z < 2) : 
    y + z > 0 := 
by
  sorry

end NUMINAMATH_GPT_necessarily_positive_l1549_154910


namespace NUMINAMATH_GPT_sum_possible_distances_l1549_154999

theorem sum_possible_distances {A B : ℝ} (hAB : |A - B| = 2) (hA : |A| = 3) : 
  (if A = 3 then |B + 2| + |B - 2| else |B + 4| + |B - 4|) = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_possible_distances_l1549_154999


namespace NUMINAMATH_GPT_fraction_of_students_participated_l1549_154966

theorem fraction_of_students_participated (total_students : ℕ) (did_not_participate : ℕ)
  (h_total : total_students = 39) (h_did_not_participate : did_not_participate = 26) :
  (total_students - did_not_participate) / total_students = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_students_participated_l1549_154966


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_product_336_l1549_154974

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y : ℕ), x * (x + 1) = 336 ∧ (y - 1) * y * (y + 1) = 336 ∧ x + (x + 1) + (y - 1) + y + (y + 1) = 54 :=
by
  -- The formal proof would go here
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_product_336_l1549_154974


namespace NUMINAMATH_GPT_pens_purchased_is_30_l1549_154957

def num_pens_purchased (cost_total: ℕ) 
                       (num_pencils: ℕ) 
                       (price_per_pencil: ℚ) 
                       (price_per_pen: ℚ)
                       (expected_pens: ℕ): Prop :=
   let cost_pencils := num_pencils * price_per_pencil
   let cost_pens := cost_total - cost_pencils
   let num_pens := cost_pens / price_per_pen
   num_pens = expected_pens

theorem pens_purchased_is_30 : num_pens_purchased 630 75 2.00 16 30 :=
by
  -- Unfold the definition manually if needed
  sorry

end NUMINAMATH_GPT_pens_purchased_is_30_l1549_154957


namespace NUMINAMATH_GPT_find_2008_star_2010_l1549_154936

-- Define the operation
def operation_star (x y : ℕ) : ℕ := sorry  -- We insert a sorry here because the precise definition is given by the conditions

-- The properties given in the problem
axiom property1 : operation_star 2 2010 = 1
axiom property2 : ∀ n : ℕ, operation_star (2 * (n + 1)) 2010 = 3 * operation_star (2 * n) 2010

-- The main proof statement
theorem find_2008_star_2010 : operation_star 2008 2010 = 3 ^ 1003 :=
by
  -- Here we would provide the proof, but it's omitted.
  sorry

end NUMINAMATH_GPT_find_2008_star_2010_l1549_154936


namespace NUMINAMATH_GPT_haniMoreSitupsPerMinute_l1549_154924

-- Define the conditions given in the problem
def totalSitups : Nat := 110
def situpsByDiana : Nat := 40
def rateDianaPerMinute : Nat := 4

-- Define the derived conditions from the solution steps
def timeDianaMinutes := situpsByDiana / rateDianaPerMinute -- 10 minutes
def situpsByHani := totalSitups - situpsByDiana -- 70 situps
def rateHaniPerMinute := situpsByHani / timeDianaMinutes -- 7 situps per minute

-- The theorem we need to prove
theorem haniMoreSitupsPerMinute : rateHaniPerMinute - rateDianaPerMinute = 3 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_haniMoreSitupsPerMinute_l1549_154924


namespace NUMINAMATH_GPT_min_value_frac_l1549_154989

theorem min_value_frac (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) : 
  ∃ (min : ℝ), min = 9 / 2 ∧ (∀ (x y : ℝ), 0 < x → 0 < y → x + y = 2 → 4 / x + 1 / y ≥ min) :=
by
  sorry

end NUMINAMATH_GPT_min_value_frac_l1549_154989


namespace NUMINAMATH_GPT_abby_damon_weight_l1549_154916

theorem abby_damon_weight (a' b' c' d' : ℕ) (h1 : a' + b' = 265) (h2 : b' + c' = 250) (h3 : c' + d' = 280) :
  a' + d' = 295 :=
  sorry -- Proof goes here

end NUMINAMATH_GPT_abby_damon_weight_l1549_154916


namespace NUMINAMATH_GPT_sine_ratio_triangle_area_l1549_154983

variables {A B C : ℝ}
variables {a b c : ℝ}
variables {area : ℝ}

-- Main statement for part 1
theorem sine_ratio 
  (h1 : a * c * Real.cos B - b * c * Real.cos A = 3 * b^2) :
  (Real.sin A / Real.sin B) = Real.sqrt 7 := 
sorry

-- Main statement for part 2
theorem triangle_area 
  (h1 : a * c * Real.cos B - b * c * Real.cos A = 3 * b^2)
  (h2 : c = Real.sqrt 11)
  (h3 : Real.sin C = (2 * Real.sqrt 2)/3)
  (h4 : C < π / 2) :
  area = Real.sqrt 14 :=
sorry

end NUMINAMATH_GPT_sine_ratio_triangle_area_l1549_154983


namespace NUMINAMATH_GPT_samantha_coins_value_l1549_154976

theorem samantha_coins_value (n d : ℕ) (h1 : n + d = 25) 
    (original_value : ℕ := 250 - 5 * n) 
    (swapped_value : ℕ := 125 + 5 * n)
    (h2 : swapped_value = original_value + 100) : original_value = 140 := 
by
  sorry

end NUMINAMATH_GPT_samantha_coins_value_l1549_154976


namespace NUMINAMATH_GPT_α_eq_β_plus_two_l1549_154935

-- Definitions based on the given conditions:
-- α(n): number of ways n can be expressed as a sum of the integers 1 and 2, considering different orders as distinct ways.
-- β(n): number of ways n can be expressed as a sum of integers greater than 1, considering different orders as distinct ways.

def α (n : ℕ) : ℕ := sorry
def β (n : ℕ) : ℕ := sorry

-- The proof statement that needs to be proved.
theorem α_eq_β_plus_two (n : ℕ) (h : 0 < n) : α n = β (n + 2) := 
  sorry

end NUMINAMATH_GPT_α_eq_β_plus_two_l1549_154935


namespace NUMINAMATH_GPT_find_principal_l1549_154932

theorem find_principal
  (A : ℝ) (r : ℝ) (t : ℝ) (P : ℝ)
  (hA : A = 896)
  (hr : r = 0.05)
  (ht : t = 12 / 5) :
  P = 800 ↔ A = P * (1 + r * t) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_principal_l1549_154932


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1549_154985

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * x + a ≥ 0) → a ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1549_154985


namespace NUMINAMATH_GPT_solve_fractional_equation_l1549_154901

theorem solve_fractional_equation (x : ℚ) (h: x ≠ 1) : 
  (x / (x - 1) = 3 / (2 * x - 2) - 2) ↔ (x = 7 / 6) := 
by
  sorry

end NUMINAMATH_GPT_solve_fractional_equation_l1549_154901


namespace NUMINAMATH_GPT_two_digit_subtraction_pattern_l1549_154991

theorem two_digit_subtraction_pattern (a b : ℕ) (h_a : 1 ≤ a ∧ a ≤ 9) (h_b : 0 ≤ b ∧ b ≤ 9) :
  (10 * a + b) - (10 * b + a) = 9 * (a - b) := 
by
  sorry

end NUMINAMATH_GPT_two_digit_subtraction_pattern_l1549_154991


namespace NUMINAMATH_GPT_sqrt_identity_l1549_154984

theorem sqrt_identity (x : ℝ) (hx : x = Real.sqrt 5 - 3) : Real.sqrt (x^2 + 6*x + 9) = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_identity_l1549_154984


namespace NUMINAMATH_GPT_find_B_l1549_154949

theorem find_B (A B : ℕ) (h : 5 * 100 + 10 * A + 8 - (B * 100 + 14) = 364) : B = 2 :=
sorry

end NUMINAMATH_GPT_find_B_l1549_154949


namespace NUMINAMATH_GPT_trig_identity_problem_l1549_154967

theorem trig_identity_problem
  (x : ℝ) (a b c : ℕ)
  (h1 : 0 < x ∧ x < (Real.pi / 2))
  (h2 : Real.sin x - Real.cos x = Real.pi / 4)
  (h3 : Real.tan x + 1 / Real.tan x = (a : ℝ) / (b - Real.pi^c)) :
  a + b + c = 50 :=
sorry

end NUMINAMATH_GPT_trig_identity_problem_l1549_154967


namespace NUMINAMATH_GPT_annual_interest_payment_l1549_154914

noncomputable def principal : ℝ := 9000
noncomputable def rate : ℝ := 9 / 100
noncomputable def time : ℝ := 1
noncomputable def interest : ℝ := principal * rate * time

theorem annual_interest_payment : interest = 810 := by
  sorry

end NUMINAMATH_GPT_annual_interest_payment_l1549_154914


namespace NUMINAMATH_GPT_samantha_routes_l1549_154926

-- Define the positions relative to the grid
structure Position where
  x : Int
  y : Int

-- Define the initial conditions and path constraints
def house : Position := ⟨-3, -2⟩
def sw_corner_of_park : Position := ⟨0, 0⟩
def ne_corner_of_park : Position := ⟨8, 5⟩
def school : Position := ⟨11, 8⟩

-- Define the combinatorial function for calculating number of ways
def binom (n k : Nat) : Nat := Nat.choose n k

-- Route segments based on the constraints
def ways_house_to_sw_corner : Nat := binom 5 2
def ways_through_park : Nat := 1
def ways_ne_corner_to_school : Nat := binom 6 3

-- Total number of routes
def total_routes : Nat := ways_house_to_sw_corner * ways_through_park * ways_ne_corner_to_school

-- The statement to be proven
theorem samantha_routes : total_routes = 200 := by
  sorry

end NUMINAMATH_GPT_samantha_routes_l1549_154926


namespace NUMINAMATH_GPT_Zhenya_Venya_are_truth_tellers_l1549_154908

-- Definitions
def is_truth_teller(dwarf : String) (truth_teller : String → Bool) : Prop :=
  truth_teller dwarf = true

def is_liar(dwarf : String) (truth_teller : String → Bool) : Prop :=
  truth_teller dwarf = false

noncomputable def BenyaStatement := "V is a liar"
noncomputable def ZhenyaStatement := "B is a liar"
noncomputable def SenyaStatement1 := "B and V are liars"
noncomputable def SenyaStatement2 := "Zh is a liar"

-- Conditions and proving the statement
theorem Zhenya_Venya_are_truth_tellers (truth_teller : String → Bool) :
  (∀ dwarf, truth_teller dwarf = true ∨ truth_teller dwarf = false) →
  (is_truth_teller "Benya" truth_teller → is_liar "Venya" truth_teller) →
  (is_truth_teller "Zhenya" truth_teller → is_liar "Benya" truth_teller) →
  (is_truth_teller "Senya" truth_teller → 
    is_liar "Benya" truth_teller ∧ is_liar "Venya" truth_teller ∧ is_liar "Zhenya" truth_teller) →
  is_truth_teller "Zhenya" truth_teller ∧ is_truth_teller "Venya" truth_teller :=
by
  sorry

end NUMINAMATH_GPT_Zhenya_Venya_are_truth_tellers_l1549_154908


namespace NUMINAMATH_GPT_unique_k_for_equal_power_l1549_154938

theorem unique_k_for_equal_power (k : ℕ) (hk : 0 < k) (h : ∃ m n : ℕ, n > 1 ∧ (3 ^ k + 5 ^ k = m ^ n)) : k = 1 :=
by
  sorry

end NUMINAMATH_GPT_unique_k_for_equal_power_l1549_154938


namespace NUMINAMATH_GPT_solve_trig_eq_l1549_154902

theorem solve_trig_eq :
  ∀ x k : ℤ, 
    (x = 2 * π / 3 + 2 * k * π ∨
     x = 7 * π / 6 + 2 * k * π ∨
     x = -π / 6 + 2 * k * π)
    → (|Real.cos x| + Real.cos (3 * x)) / (Real.sin x * Real.cos (2 * x)) = -2 * Real.sqrt 3 := 
by
  intros x k h
  sorry

end NUMINAMATH_GPT_solve_trig_eq_l1549_154902


namespace NUMINAMATH_GPT_total_worth_of_stock_l1549_154977

theorem total_worth_of_stock :
  let cost_expensive := 10
  let cost_cheaper := 3.5
  let total_modules := 11
  let cheaper_modules := 10
  let expensive_modules := total_modules - cheaper_modules
  let worth_cheaper_modules := cheaper_modules * cost_cheaper
  let worth_expensive_module := expensive_modules * cost_expensive 
  worth_cheaper_modules + worth_expensive_module = 45 := by
  sorry

end NUMINAMATH_GPT_total_worth_of_stock_l1549_154977


namespace NUMINAMATH_GPT_find_integer_divisible_by_18_and_square_root_in_range_l1549_154960

theorem find_integer_divisible_by_18_and_square_root_in_range :
  ∃ x : ℕ, 28 < Real.sqrt x ∧ Real.sqrt x < 28.2 ∧ 18 ∣ x ∧ x = 792 :=
by
  sorry

end NUMINAMATH_GPT_find_integer_divisible_by_18_and_square_root_in_range_l1549_154960


namespace NUMINAMATH_GPT_find_divisor_l1549_154955

theorem find_divisor (D Q R Div : ℕ) (h1 : Q = 40) (h2 : R = 64) (h3 : Div = 2944) 
  (h4 : Div = (D * Q) + R) : D = 72 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l1549_154955


namespace NUMINAMATH_GPT_inequality_true_l1549_154939

theorem inequality_true (a b : ℝ) (hab : a < b) (hb : b < 0) (ha : a < 0) : (b / a) < 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_true_l1549_154939


namespace NUMINAMATH_GPT_arithmetic_sequence_nth_term_l1549_154986

theorem arithmetic_sequence_nth_term (S : ℕ → ℕ) (h : ∀ n, S n = 5 * n + 4 * n^2) (r : ℕ) : 
  S r - S (r - 1) = 8 * r + 1 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_nth_term_l1549_154986


namespace NUMINAMATH_GPT_reeya_average_l1549_154954

theorem reeya_average (s1 s2 s3 s4 s5 : ℕ) (h1 : s1 = 65) (h2 : s2 = 67) (h3 : s3 = 76) (h4 : s4 = 82) (h5 : s5 = 85) :
  (s1 + s2 + s3 + s4 + s5) / 5 = 75 := by
  sorry

end NUMINAMATH_GPT_reeya_average_l1549_154954


namespace NUMINAMATH_GPT_range_of_a_l1549_154933

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ x^2 + 2*x + a ≥ 0) ↔ a ≥ -8 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1549_154933


namespace NUMINAMATH_GPT_trips_needed_to_fill_pool_l1549_154980

def caleb_gallons_per_trip : ℕ := 7
def cynthia_gallons_per_trip : ℕ := 8
def pool_capacity : ℕ := 105

theorem trips_needed_to_fill_pool : (pool_capacity / (caleb_gallons_per_trip + cynthia_gallons_per_trip) = 7) :=
by
  sorry

end NUMINAMATH_GPT_trips_needed_to_fill_pool_l1549_154980


namespace NUMINAMATH_GPT_magician_red_marbles_taken_l1549_154990

theorem magician_red_marbles_taken:
  ∃ R : ℕ, (20 - R) + (30 - 4 * R) = 35 ∧ R = 3 :=
by
  sorry

end NUMINAMATH_GPT_magician_red_marbles_taken_l1549_154990


namespace NUMINAMATH_GPT_sum_of_all_four_numbers_is_zero_l1549_154921

theorem sum_of_all_four_numbers_is_zero 
  {a b c d : ℝ}
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum : a + b = c + d)
  (h_prod : a * c = b * d) 
  : a + b + c + d = 0 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_all_four_numbers_is_zero_l1549_154921


namespace NUMINAMATH_GPT_Ed_more_marbles_than_Doug_l1549_154981

-- Definitions based on conditions
def Ed_marbles_initial : ℕ := 45
def Doug_loss : ℕ := 11
def Doug_marbles_initial : ℕ := Ed_marbles_initial - 10
def Doug_marbles_after_loss : ℕ := Doug_marbles_initial - Doug_loss

-- Theorem statement
theorem Ed_more_marbles_than_Doug :
  Ed_marbles_initial - Doug_marbles_after_loss = 21 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_Ed_more_marbles_than_Doug_l1549_154981


namespace NUMINAMATH_GPT_combination_simplify_l1549_154919

theorem combination_simplify : (Nat.choose 6 2) + 3 = 18 := by
  sorry

end NUMINAMATH_GPT_combination_simplify_l1549_154919


namespace NUMINAMATH_GPT_complex_division_correct_l1549_154963

theorem complex_division_correct : (3 - 1 * Complex.I) / (1 + Complex.I) = 1 - 2 * Complex.I := 
by
  sorry

end NUMINAMATH_GPT_complex_division_correct_l1549_154963


namespace NUMINAMATH_GPT_polynomial_coefficients_correct_l1549_154952

-- Define the polynomial equation
def polynomial_equation (x a b c d : ℝ) : Prop :=
  x^4 + 4*x^3 + 3*x^2 + 2*x + 1 = (x+1)^4 + a*(x+1)^3 + b*(x+1)^2 + c*(x+1) + d

-- The problem to prove
theorem polynomial_coefficients_correct :
  ∀ x : ℝ, polynomial_equation x 0 (-3) 4 (-1) :=
by
  intro x
  unfold polynomial_equation
  sorry

end NUMINAMATH_GPT_polynomial_coefficients_correct_l1549_154952


namespace NUMINAMATH_GPT_ratio_of_longer_side_to_square_l1549_154941

theorem ratio_of_longer_side_to_square (s a b : ℝ) (h1 : a * b = 2 * s^2) (h2 : a = 2 * b) : a / s = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_longer_side_to_square_l1549_154941


namespace NUMINAMATH_GPT_inequality_solution_l1549_154997

theorem inequality_solution (x : ℝ) :
  (-1 : ℝ) < (x^2 - 14*x + 11) / (x^2 - 2*x + 3) ∧
  (x^2 - 14*x + 11) / (x^2 - 2*x + 3) < (1 : ℝ) ↔
  (2/3 < x ∧ x < 1) ∨ (7 < x) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1549_154997


namespace NUMINAMATH_GPT_other_root_of_quadratic_eq_l1549_154922

theorem other_root_of_quadratic_eq (m : ℝ) (q : ℝ) :
  (∃ x : ℝ, x ≠ q ∧ 3 * x^2 + m * x - 7 = 0) →
  (3 * q^2 + m * q - 7 = 0) →
  q = -7 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_other_root_of_quadratic_eq_l1549_154922


namespace NUMINAMATH_GPT_time_after_12345_seconds_is_13_45_45_l1549_154903

def seconds_in_a_minute := 60
def minutes_in_an_hour := 60
def initial_hour := 10
def initial_minute := 45
def initial_second := 0
def total_seconds := 12345

def time_after_seconds (hour minute second : Nat) (elapsed_seconds : Nat) : (Nat × Nat × Nat) :=
  let total_initial_seconds := hour * 3600 + minute * 60 + second
  let total_final_seconds := total_initial_seconds + elapsed_seconds
  let final_hour := total_final_seconds / 3600
  let remaining_seconds_after_hour := total_final_seconds % 3600
  let final_minute := remaining_seconds_after_hour / 60
  let final_second := remaining_seconds_after_hour % 60
  (final_hour, final_minute, final_second)

theorem time_after_12345_seconds_is_13_45_45 :
  time_after_seconds initial_hour initial_minute initial_second total_seconds = (13, 45, 45) :=
by
  sorry

end NUMINAMATH_GPT_time_after_12345_seconds_is_13_45_45_l1549_154903


namespace NUMINAMATH_GPT_pig_count_correct_l1549_154968

def initial_pigs : ℝ := 64.0
def additional_pigs : ℝ := 86.0
def total_pigs : ℝ := 150.0

theorem pig_count_correct : initial_pigs + additional_pigs = total_pigs := by
  show 64.0 + 86.0 = 150.0
  sorry

end NUMINAMATH_GPT_pig_count_correct_l1549_154968


namespace NUMINAMATH_GPT_range_of_m_l1549_154988

noncomputable def f : ℝ → ℝ := sorry

lemma function_symmetric {x : ℝ} : f (2 + x) = f (-x) := sorry

lemma f_decreasing_on_pos_halfline {x y : ℝ} (hx : x ≥ 1) (hy : y ≥ 1) (hxy : x < y) : f x ≥ f y := sorry

theorem range_of_m {m : ℝ} (h : f (1 - m) < f m) : m > (1 / 2) := sorry

end NUMINAMATH_GPT_range_of_m_l1549_154988


namespace NUMINAMATH_GPT_right_triangle_lengths_l1549_154920

theorem right_triangle_lengths (a b c : ℝ) (h1 : c + b = 2 * a) (h2 : c^2 = a^2 + b^2) : 
  b = 3 / 4 * a ∧ c = 5 / 4 * a := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_lengths_l1549_154920


namespace NUMINAMATH_GPT_relationship_between_roots_l1549_154907

-- Define the number of real roots of the equations
def number_real_roots_lg_eq_sin : ℕ := 3
def number_real_roots_x_eq_sin : ℕ := 1
def number_real_roots_x4_eq_sin : ℕ := 2

-- Define the variables
def a : ℕ := number_real_roots_lg_eq_sin
def b : ℕ := number_real_roots_x_eq_sin
def c : ℕ := number_real_roots_x4_eq_sin

-- State the theorem
theorem relationship_between_roots : a > c ∧ c > b :=
by
  -- the proof is skipped
  sorry

end NUMINAMATH_GPT_relationship_between_roots_l1549_154907


namespace NUMINAMATH_GPT_updated_mean_of_observations_l1549_154913

theorem updated_mean_of_observations
    (number_of_observations : ℕ)
    (initial_mean : ℝ)
    (decrement_per_observation : ℝ)
    (h1 : number_of_observations = 50)
    (h2 : initial_mean = 200)
    (h3 : decrement_per_observation = 15) :
    (initial_mean * number_of_observations - decrement_per_observation * number_of_observations) / number_of_observations = 185 :=
by {
    sorry
}

end NUMINAMATH_GPT_updated_mean_of_observations_l1549_154913


namespace NUMINAMATH_GPT_quarters_and_dimes_l1549_154945

theorem quarters_and_dimes (n : ℕ) (qval : ℕ := 25) (dval : ℕ := 10) 
  (hq : 20 * qval + 10 * dval = 10 * qval + n * dval) : 
  n = 35 :=
by
  sorry

end NUMINAMATH_GPT_quarters_and_dimes_l1549_154945


namespace NUMINAMATH_GPT_arithmetic_sequence_prop_l1549_154975

theorem arithmetic_sequence_prop (a1 d : ℝ) (S : ℕ → ℝ) 
  (h1 : S 6 > S 7) (h2 : S 7 > S 5)
  (hSn : ∀ n, S n = n * a1 + (n * (n - 1) / 2) * d) :
  (d < 0) ∧ (S 11 > 0) ∧ (|a1 + 5 * d| > |a1 + 6 * d|) := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_prop_l1549_154975


namespace NUMINAMATH_GPT_value_of_expression_when_x_eq_4_l1549_154995

theorem value_of_expression_when_x_eq_4 : (3 * 4 + 4)^2 = 256 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_when_x_eq_4_l1549_154995


namespace NUMINAMATH_GPT_gift_distribution_l1549_154965

theorem gift_distribution :
  let bags := [1, 2, 3, 4, 5]
  let num_people := 4
  ∃ d: ℕ, d = 96 := by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_gift_distribution_l1549_154965


namespace NUMINAMATH_GPT_number_of_spiders_l1549_154998

theorem number_of_spiders (total_legs : ℕ) (legs_per_spider : ℕ) (h1 : total_legs = 32) (h2 : legs_per_spider = 8) :
  total_legs / legs_per_spider = 4 := by
  sorry

end NUMINAMATH_GPT_number_of_spiders_l1549_154998


namespace NUMINAMATH_GPT_triangle_side_lengths_exist_l1549_154961

theorem triangle_side_lengths_exist :
  ∃ (a b c : ℕ), a ≥ b ∧ b ≥ c ∧ a + b > c ∧ b + c > a ∧ a + c > b ∧ abc = 2 * (a - 1) * (b - 1) * (c - 1) ∧
  ((a, b, c) = (8, 7, 3) ∨ (a, b, c) = (6, 5, 4)) :=
by sorry

end NUMINAMATH_GPT_triangle_side_lengths_exist_l1549_154961


namespace NUMINAMATH_GPT_nancy_history_books_l1549_154930

/-- Nancy started with 46 books in total on the cart.
    She shelved 8 romance books and 4 poetry books from the top section.
    She shelved 5 Western novels and 6 biographies from the bottom section.
    Half the books on the bottom section were mystery books.
    Prove that Nancy shelved 12 history books.
-/
theorem nancy_history_books 
  (total_books : ℕ)
  (romance_books : ℕ)
  (poetry_books : ℕ)
  (western_novels : ℕ)
  (biographies : ℕ)
  (bottom_books_half_mystery : ℕ)
  (history_books : ℕ) :
  (total_books = 46) →
  (romance_books = 8) →
  (poetry_books = 4) →
  (western_novels = 5) →
  (biographies = 6) →
  (bottom_books_half_mystery = 11) →
  (history_books = total_books - ((romance_books + poetry_books) + (2 * (western_novels + biographies)))) →
  history_books = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_nancy_history_books_l1549_154930


namespace NUMINAMATH_GPT_squareInPentagon_l1549_154912

-- Definitions pertinent to the problem
structure Pentagon (α : Type) [AddCommGroup α] :=
(A B C D E : α) 

def isRegularPentagon {α : Type} [AddCommGroup α] [LinearOrder α] (P : Pentagon α) : Prop :=
  -- Conditions for a regular pentagon (typically involving equal side lengths and equal angles)
  sorry

def inscribedSquareExists {α : Type} [AddCommGroup α] (P : Pentagon α) : Prop :=
  -- There exists a square inscribed in the pentagon P with vertices on four different sides
  sorry

-- The main theorem to state the proof problem
theorem squareInPentagon {α : Type} [AddCommGroup α] [LinearOrder α] (P : Pentagon α)
  (hP : isRegularPentagon P) : inscribedSquareExists P :=
sorry

end NUMINAMATH_GPT_squareInPentagon_l1549_154912


namespace NUMINAMATH_GPT_positive_triple_l1549_154962

theorem positive_triple
  (a b c : ℝ)
  (h1 : a + b + c > 0)
  (h2 : ab + bc + ca > 0)
  (h3 : abc > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_GPT_positive_triple_l1549_154962


namespace NUMINAMATH_GPT_yield_is_eight_percent_l1549_154987

noncomputable def par_value : ℝ := 100
noncomputable def annual_dividend : ℝ := 0.12 * par_value
noncomputable def market_value : ℝ := 150
noncomputable def yield_percentage : ℝ := (annual_dividend / market_value) * 100

theorem yield_is_eight_percent : yield_percentage = 8 := 
by 
  sorry

end NUMINAMATH_GPT_yield_is_eight_percent_l1549_154987


namespace NUMINAMATH_GPT_total_tissues_brought_l1549_154943

def number_students_group1 : Nat := 9
def number_students_group2 : Nat := 10
def number_students_group3 : Nat := 11
def tissues_per_box : Nat := 40

theorem total_tissues_brought : 
  (number_students_group1 + number_students_group2 + number_students_group3) * tissues_per_box = 1200 := 
by 
  sorry

end NUMINAMATH_GPT_total_tissues_brought_l1549_154943


namespace NUMINAMATH_GPT_minimum_meals_needed_l1549_154929

theorem minimum_meals_needed (total_jam : ℝ) (max_per_meal : ℝ) (jars : ℕ) (max_jar_weight : ℝ):
  (total_jam = 50) → (max_per_meal = 5) → (jars ≥ 50) → (max_jar_weight ≤ 1) →
  (jars * max_jar_weight = total_jam) →
  jars ≥ 12 := sorry

end NUMINAMATH_GPT_minimum_meals_needed_l1549_154929


namespace NUMINAMATH_GPT_eight_digit_increasing_numbers_mod_1000_l1549_154934

theorem eight_digit_increasing_numbers_mod_1000 : 
  ((Nat.choose 17 8) % 1000) = 310 := 
by 
  sorry -- Proof not required as per instructions

end NUMINAMATH_GPT_eight_digit_increasing_numbers_mod_1000_l1549_154934


namespace NUMINAMATH_GPT_total_people_surveyed_l1549_154982

theorem total_people_surveyed (x y : ℝ) (h1 : 0.536 * x = 30) (h2 : 0.794 * y = x) : y = 71 :=
by
  sorry

end NUMINAMATH_GPT_total_people_surveyed_l1549_154982


namespace NUMINAMATH_GPT_max_items_for_2019_students_l1549_154956

noncomputable def max_items (students : ℕ) : ℕ :=
  students / 2

theorem max_items_for_2019_students : max_items 2019 = 1009 := by
  sorry

end NUMINAMATH_GPT_max_items_for_2019_students_l1549_154956


namespace NUMINAMATH_GPT_min_score_needed_l1549_154927

/-- 
Given the list of scores and the targeted increase in the average score,
ascertain that the minimum score required on the next test to achieve the
new average is 110.
 -/
theorem min_score_needed 
  (scores : List ℝ) 
  (target_increase : ℝ) 
  (new_score : ℝ) 
  (total_scores : ℝ)
  (current_average : ℝ) 
  (target_average : ℝ) 
  (needed_score : ℝ) :
  (total_scores = 86 + 92 + 75 + 68 + 88 + 84) ∧
  (current_average = total_scores / 6) ∧
  (target_average = current_average + target_increase) ∧
  (new_score = total_scores + needed_score) ∧
  (target_average = new_score / 7) ->
  needed_score = 110 :=
by
  sorry

end NUMINAMATH_GPT_min_score_needed_l1549_154927


namespace NUMINAMATH_GPT_perpendicular_vectors_relation_l1549_154993

theorem perpendicular_vectors_relation (a b : ℝ) (h : 3 * a - 7 * b = 0) : a = 7 * b / 3 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_relation_l1549_154993


namespace NUMINAMATH_GPT_polynomial_bound_implies_l1549_154947

theorem polynomial_bound_implies :
  (∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) →
  (∀ x : ℝ, |x| ≤ 1 → |c * x^2 + b * x + a| ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_bound_implies_l1549_154947


namespace NUMINAMATH_GPT_no_solution_exists_l1549_154906

theorem no_solution_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ (x y : ℝ), f (x ^ 2 + f y) = 2 * x - f y :=
by
  sorry

end NUMINAMATH_GPT_no_solution_exists_l1549_154906


namespace NUMINAMATH_GPT_maximum_elements_in_A_l1549_154905

theorem maximum_elements_in_A (n : ℕ) (h : n > 0)
  (A : Finset (Finset (Fin n))) 
  (hA : ∀ a ∈ A, ∀ b ∈ A, a ≠ b → ¬ a ⊆ b) :  
  A.card ≤ Nat.choose n (n / 2) :=
sorry

end NUMINAMATH_GPT_maximum_elements_in_A_l1549_154905
