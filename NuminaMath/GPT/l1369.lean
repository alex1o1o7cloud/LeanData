import Mathlib

namespace NUMINAMATH_GPT_not_prime_for_any_n_l1369_136914

theorem not_prime_for_any_n (k : ℕ) (hk : 1 < k) (n : ℕ) : 
  ¬ Prime (n^4 + 4 * k^4) :=
sorry

end NUMINAMATH_GPT_not_prime_for_any_n_l1369_136914


namespace NUMINAMATH_GPT_original_number_is_repeating_decimal_l1369_136995

theorem original_number_is_repeating_decimal :
  ∃ N : ℚ, (N * 10 ^ 28) % 10^30 = 15 ∧ N * 5 = 0.7894736842105263 ∧ 
  (N = 3 / 19) :=
sorry

end NUMINAMATH_GPT_original_number_is_repeating_decimal_l1369_136995


namespace NUMINAMATH_GPT_shorter_side_length_l1369_136938

theorem shorter_side_length (a b : ℕ) (h1 : 2 * a + 2 * b = 50) (h2 : a * b = 126) : b = 9 :=
sorry

end NUMINAMATH_GPT_shorter_side_length_l1369_136938


namespace NUMINAMATH_GPT_original_stone_counted_as_99_l1369_136917

theorem original_stone_counted_as_99 :
  (99 % 22) = 11 :=
by sorry

end NUMINAMATH_GPT_original_stone_counted_as_99_l1369_136917


namespace NUMINAMATH_GPT_distance_AB_l1369_136960

theorem distance_AB : 
  let A := -1
  let B := 2020
  |A - B| = 2021 := by
  sorry

end NUMINAMATH_GPT_distance_AB_l1369_136960


namespace NUMINAMATH_GPT_alpha_necessary_not_sufficient_for_beta_l1369_136909

def alpha (x : ℝ) : Prop := x^2 = 4
def beta (x : ℝ) : Prop := x = 2

theorem alpha_necessary_not_sufficient_for_beta :
  (∀ x : ℝ, beta x → alpha x) ∧ ¬(∀ x : ℝ, alpha x → beta x) :=
by
  sorry

end NUMINAMATH_GPT_alpha_necessary_not_sufficient_for_beta_l1369_136909


namespace NUMINAMATH_GPT_problem_statement_l1369_136911

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 4)
def c : ℝ × ℝ := (3, 2)

def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def vec_scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vec_dot (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem problem_statement : vec_dot (vec_add a (vec_scalar_mul 2 b)) c = -3 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1369_136911


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1369_136993

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (h_arith_seq: ∀ n: ℕ, S n = S 0 + n * (S 1 - S 0)) 
  (h5 : S 5 = 10) (h10 : S 10 = 30) : S 15 = 60 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1369_136993


namespace NUMINAMATH_GPT_first_quarter_days_2016_l1369_136942

theorem first_quarter_days_2016 : 
  let leap_year := 2016
  let jan_days := 31
  let feb_days := if leap_year % 4 = 0 ∧ (leap_year % 100 ≠ 0 ∨ leap_year % 400 = 0) then 29 else 28
  let mar_days := 31
  (jan_days + feb_days + mar_days) = 91 := 
by
  let leap_year := 2016
  let jan_days := 31
  let feb_days := if leap_year % 4 = 0 ∧ (leap_year % 100 ≠ 0 ∨ leap_year % 400 = 0) then 29 else 28
  let mar_days := 31
  have h_leap_year : leap_year % 4 = 0 ∧ (leap_year % 100 ≠ 0 ∨ leap_year % 400 = 0) := by sorry
  have h_feb_days : feb_days = 29 := by sorry
  have h_first_quarter : jan_days + feb_days + mar_days = 31 + 29 + 31 := by sorry
  have h_sum : 31 + 29 + 31 = 91 := by norm_num
  exact h_sum

end NUMINAMATH_GPT_first_quarter_days_2016_l1369_136942


namespace NUMINAMATH_GPT_intersection_is_interval_l1369_136940

-- Let M be the set of numbers where the domain of the function y = log x is defined.
def M : Set ℝ := {x | 0 < x}

-- Let N be the set of numbers where x^2 - 4 > 0.
def N : Set ℝ := {x | x^2 - 4 > 0}

-- The complement of N in the real numbers ℝ.
def complement_N : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- We need to prove that the intersection of M and the complement of N is the interval (0, 2].
theorem intersection_is_interval : (M ∩ complement_N) = {x | 0 < x ∧ x ≤ 2} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_is_interval_l1369_136940


namespace NUMINAMATH_GPT_quadratic_unique_root_l1369_136958

theorem quadratic_unique_root (b c : ℝ)
  (h₁ : b = c^2 + 1)
  (h₂ : (x^2 + b * x + c = 0) → ∃! x : ℝ, x^2 + b * x + c = 0) :
  c = 1 ∨ c = -1 := 
sorry

end NUMINAMATH_GPT_quadratic_unique_root_l1369_136958


namespace NUMINAMATH_GPT_jar_weight_percentage_l1369_136999

theorem jar_weight_percentage (J B : ℝ) (h : 0.60 * (J + B) = J + 1 / 3 * B) :
  (J / (J + B)) = 0.403 :=
by
  sorry

end NUMINAMATH_GPT_jar_weight_percentage_l1369_136999


namespace NUMINAMATH_GPT_time_shortened_by_opening_both_pipes_l1369_136926

theorem time_shortened_by_opening_both_pipes 
  (a b p : ℝ) 
  (hp : a * p > 0) -- To ensure p > 0 and reservoir volume is positive
  (h1 : p = (a * p) / a) -- Given that pipe A alone takes p hours
  : p - (a * p) / (a + b) = (b * p) / (a + b) := 
sorry

end NUMINAMATH_GPT_time_shortened_by_opening_both_pipes_l1369_136926


namespace NUMINAMATH_GPT_final_price_is_correct_l1369_136961

-- Define the original price and percentages as constants
def original_price : ℝ := 160
def increase_percentage : ℝ := 0.25
def discount_percentage : ℝ := 0.25

-- Calculate increased price
def increased_price : ℝ := original_price * (1 + increase_percentage)
-- Calculate the discount on the increased price
def discount_amount : ℝ := increased_price * discount_percentage
-- Calculate final price after discount
def final_price : ℝ := increased_price - discount_amount

-- Statement of the theorem: prove final price is $150
theorem final_price_is_correct : final_price = 150 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_final_price_is_correct_l1369_136961


namespace NUMINAMATH_GPT_like_terms_sum_l1369_136972

theorem like_terms_sum (m n : ℕ) (h1 : 2 * m = 4) (h2 : 3 - n = 1) : m + n = 4 :=
by
  sorry

end NUMINAMATH_GPT_like_terms_sum_l1369_136972


namespace NUMINAMATH_GPT_period_of_f_cos_theta_l1369_136991

open Real

noncomputable def alpha (x : ℝ) : ℝ × ℝ :=
  (sqrt 3 * sin (2 * x), cos x + sin x)

noncomputable def beta (x : ℝ) : ℝ × ℝ :=
  (1, cos x - sin x)

noncomputable def f (x : ℝ) : ℝ :=
  let (α1, α2) := alpha x
  let (β1, β2) := beta x
  α1 * β1 + α2 * β2

theorem period_of_f :
  (∀ x : ℝ, f (x + π) = f x) ∧ (∀ T : ℝ, (T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) → T = π) :=
sorry

theorem cos_theta :
  ∀ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ f θ = 1 → cos (θ - π / 6) = sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_period_of_f_cos_theta_l1369_136991


namespace NUMINAMATH_GPT_union_of_P_and_neg_RQ_l1369_136979

noncomputable def R : Set ℝ := Set.univ

noncomputable def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

noncomputable def Q : Set ℝ := {x | -2 < x ∧ x < 2}

noncomputable def neg_RQ : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}

theorem union_of_P_and_neg_RQ : 
  P ∪ neg_RQ = {x | x ≤ -2 ∨ 1 ≤ x ∧ x ≤ 3} :=
by
  sorry

end NUMINAMATH_GPT_union_of_P_and_neg_RQ_l1369_136979


namespace NUMINAMATH_GPT_volume_of_cube_in_pyramid_l1369_136922

theorem volume_of_cube_in_pyramid :
  (∃ (s : ℝ), 
    ( ∀ (b h l : ℝ),
      b = 2 ∧ 
      h = 3 ∧ 
      l = 2 * Real.sqrt 2 →
      s = 4 * Real.sqrt 2 - 3 ∧ 
      ((4 * Real.sqrt 2 - 3) ^ 3 = (4 * Real.sqrt 2 - 3) ^ 3))) :=
sorry

end NUMINAMATH_GPT_volume_of_cube_in_pyramid_l1369_136922


namespace NUMINAMATH_GPT_minimize_perimeter_isosceles_l1369_136988

noncomputable def inradius (A B C : ℝ) (r : ℝ) : Prop := sorry -- Define inradius

theorem minimize_perimeter_isosceles (A B C : ℝ) (r : ℝ) 
  (h1 : A + B + C = 180) -- Angles sum to 180 degrees
  (h2 : inradius A B C r) -- Given inradius
  (h3 : A = fixed_angle) -- Given fixed angle A
  : B = C :=
by sorry

end NUMINAMATH_GPT_minimize_perimeter_isosceles_l1369_136988


namespace NUMINAMATH_GPT_bananas_and_cantaloupe_cost_l1369_136934

noncomputable def prices (a b c d : ℕ) : Prop :=
  a + b + c + d = 40 ∧
  d = 3 * a ∧
  b = c - 2

theorem bananas_and_cantaloupe_cost (a b c d : ℕ) (h : prices a b c d) : b + c = 20 :=
by
  obtain ⟨h1, h2, h3⟩ := h
  -- Using the given conditions:
  --     a + b + c + d = 40
  --     d = 3 * a
  --     b = c - 2
  -- We find that b + c = 20
  sorry

end NUMINAMATH_GPT_bananas_and_cantaloupe_cost_l1369_136934


namespace NUMINAMATH_GPT_factorize_expression_l1369_136963

variable (a b : ℝ)

theorem factorize_expression : a^2 - 4 * b^2 - 2 * a + 4 * b = (a + 2 * b - 2) * (a - 2 * b) := 
  sorry

end NUMINAMATH_GPT_factorize_expression_l1369_136963


namespace NUMINAMATH_GPT_sarah_toads_l1369_136932

theorem sarah_toads (tim_toads : ℕ) (jim_toads : ℕ) (sarah_toads : ℕ)
  (h1 : tim_toads = 30)
  (h2 : jim_toads = tim_toads + 20)
  (h3 : sarah_toads = 2 * jim_toads) :
  sarah_toads = 100 :=
by
  sorry

end NUMINAMATH_GPT_sarah_toads_l1369_136932


namespace NUMINAMATH_GPT_range_values_for_a_l1369_136951

def p (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0
def q (x a : ℝ) (ha : 0 < a) : Prop := x^2 - 2 * x + 1 - a^2 ≥ 0

theorem range_values_for_a (a : ℝ) : (∃ ha : 0 < a, (∀ x : ℝ, (¬ p x → q x a ha))) → (0 < a ∧ a ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_range_values_for_a_l1369_136951


namespace NUMINAMATH_GPT_probability_no_adjacent_birch_trees_l1369_136923

open Nat

theorem probability_no_adjacent_birch_trees : 
    let m := 7
    let n := 990
    m + n = 106 := 
by
  sorry

end NUMINAMATH_GPT_probability_no_adjacent_birch_trees_l1369_136923


namespace NUMINAMATH_GPT_problem_solution_l1369_136925

def complex_expression : ℕ := 3 * (3 * (4 * (3 * (4 * (2 + 1) + 1) + 2) + 1) + 2) + 1

theorem problem_solution : complex_expression = 1492 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1369_136925


namespace NUMINAMATH_GPT_divisibility_of_solutions_l1369_136941

theorem divisibility_of_solutions (p : ℕ) (k : ℕ) (x₀ y₀ z₀ t₀ : ℕ) 
  (hp_prime : Nat.Prime p)
  (hp_form : p = 4 * k + 3)
  (h_eq : x₀^(2*p) + y₀^(2*p) + z₀^(2*p) = t₀^(2*p)) : 
  p ∣ x₀ ∨ p ∣ y₀ ∨ p ∣ z₀ ∨ p ∣ t₀ :=
sorry

end NUMINAMATH_GPT_divisibility_of_solutions_l1369_136941


namespace NUMINAMATH_GPT_motorcycle_price_l1369_136936

variable (x : ℝ) -- selling price of each motorcycle
variable (car_cost material_car material_motorcycle : ℝ)

theorem motorcycle_price
  (h1 : car_cost = 100)
  (h2 : material_car = 4 * 50)
  (h3 : material_motorcycle = 250)
  (h4 : 8 * x - material_motorcycle = material_car - car_cost + 50)
  : x = 50 := 
sorry

end NUMINAMATH_GPT_motorcycle_price_l1369_136936


namespace NUMINAMATH_GPT_triangles_with_positive_area_l1369_136973

theorem triangles_with_positive_area (x y : ℕ) (h₁ : 1 ≤ x ∧ x ≤ 5) (h₂ : 1 ≤ y ∧ y ≤ 3) : 
    ∃ (n : ℕ), n = 420 := 
sorry

end NUMINAMATH_GPT_triangles_with_positive_area_l1369_136973


namespace NUMINAMATH_GPT_total_dogs_on_farm_l1369_136987

-- Definitions based on conditions from part a)
def num_dog_houses : ℕ := 5
def num_dogs_per_house : ℕ := 4

-- Statement to prove
theorem total_dogs_on_farm : num_dog_houses * num_dogs_per_house = 20 :=
by
  sorry

end NUMINAMATH_GPT_total_dogs_on_farm_l1369_136987


namespace NUMINAMATH_GPT_catches_difference_is_sixteen_l1369_136943

noncomputable def joe_catches : ℕ := 23
noncomputable def derek_catches : ℕ := 2 * joe_catches - 4
noncomputable def tammy_catches : ℕ := 30
noncomputable def one_third_derek : ℕ := derek_catches / 3
noncomputable def difference : ℕ := tammy_catches - one_third_derek

theorem catches_difference_is_sixteen :
  difference = 16 := 
by
  sorry

end NUMINAMATH_GPT_catches_difference_is_sixteen_l1369_136943


namespace NUMINAMATH_GPT_solve_for_square_l1369_136902

theorem solve_for_square (x : ℤ) (s : ℤ) 
  (h1 : s + x = 80) 
  (h2 : 3 * (s + x) - 2 * x = 164) : 
  s = 42 :=
by 
  -- Include the implementation with sorry
  sorry

end NUMINAMATH_GPT_solve_for_square_l1369_136902


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l1369_136950

noncomputable def ellipse (a b c : ℝ) :=
  (a > b) ∧ (b > 0) ∧ (a^2 = b^2 + c^2) ∧ (b = 2 * c)

theorem eccentricity_of_ellipse (a b c : ℝ) (h : ellipse a b c) :
  (c / a = Real.sqrt 5 / 5) :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l1369_136950


namespace NUMINAMATH_GPT_number_sum_20_eq_30_l1369_136968

theorem number_sum_20_eq_30 : ∃ x : ℤ, 20 + x = 30 → x = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_sum_20_eq_30_l1369_136968


namespace NUMINAMATH_GPT_smallest_a_mod_remainders_l1369_136998

theorem smallest_a_mod_remainders:
  (∀ d ∈ [2, 3, 4, 5, 6, 7, 8, 9], 2521 % d = 1) ∧
  (∀ n : ℕ, ∃ a : ℕ, a = 2520 * n + 1 ∧ (∀ d ∈ [2, 3, 4, 5, 6, 7, 8, 9], a % d = 1)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_a_mod_remainders_l1369_136998


namespace NUMINAMATH_GPT_rachel_steps_l1369_136935

theorem rachel_steps (x : ℕ) (h1 : x + 325 = 892) : x = 567 :=
sorry

end NUMINAMATH_GPT_rachel_steps_l1369_136935


namespace NUMINAMATH_GPT_equal_values_of_means_l1369_136980

theorem equal_values_of_means (f : ℤ × ℤ → ℤ) 
  (h_pos : ∀ p, 0 < f p)
  (h_mean : ∀ p, f p = (f (p.1 + 1, p.2) + f (p.1 - 1, p.2) + f (p.1, p.2 + 1) + f (p.1, p.2 - 1)) / 4):
  ∃ m : ℤ, ∀ p, f p = m := sorry

end NUMINAMATH_GPT_equal_values_of_means_l1369_136980


namespace NUMINAMATH_GPT_price_diff_is_correct_l1369_136955

-- Define initial conditions
def initial_price : ℝ := 30
def flat_discount : ℝ := 5
def percent_discount : ℝ := 0.25
def sales_tax : ℝ := 0.10

def price_after_flat_discount (price : ℝ) : ℝ :=
  price - flat_discount

def price_after_percent_discount (price : ℝ) : ℝ :=
  price * (1 - percent_discount)

def price_after_tax (price : ℝ) : ℝ :=
  price * (1 + sales_tax)

def final_price_method1 : ℝ :=
  price_after_tax (price_after_percent_discount (price_after_flat_discount initial_price))

def final_price_method2 : ℝ :=
  price_after_tax (price_after_flat_discount (price_after_percent_discount initial_price))

def difference_in_cents : ℝ :=
  (final_price_method1 - final_price_method2) * 100

-- Lean statement to prove the final difference in cents
theorem price_diff_is_correct : difference_in_cents = 137.5 :=
  by sorry

end NUMINAMATH_GPT_price_diff_is_correct_l1369_136955


namespace NUMINAMATH_GPT_find_diameter_of_hemisphere_l1369_136983

theorem find_diameter_of_hemisphere (r a : ℝ) (hr : r = a / 2) (volume : ℝ) (hV : volume = 18 * Real.pi) : 
  2/3 * Real.pi * r ^ 3 = 18 * Real.pi → a = 6 := by
  intro h
  sorry

end NUMINAMATH_GPT_find_diameter_of_hemisphere_l1369_136983


namespace NUMINAMATH_GPT_simplify_sqrt_product_l1369_136964

theorem simplify_sqrt_product : (Real.sqrt (3 * 5) * Real.sqrt (3 ^ 5 * 5 ^ 5) = 3375) :=
  sorry

end NUMINAMATH_GPT_simplify_sqrt_product_l1369_136964


namespace NUMINAMATH_GPT_person_saves_2000_l1369_136906

variable (income expenditure savings : ℕ)
variable (h_ratio : income / expenditure = 7 / 6)
variable (h_income : income = 14000)

theorem person_saves_2000 (h_ratio : income / expenditure = 7 / 6) (h_income : income = 14000) :
  savings = income - (6 * (14000 / 7)) :=
by
  sorry

end NUMINAMATH_GPT_person_saves_2000_l1369_136906


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1369_136939

theorem solution_set_of_inequality (x : ℝ) : |x^2 - 2| < 2 ↔ ((-2 < x ∧ x < 0) ∨ (0 < x ∧ x < 2)) :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1369_136939


namespace NUMINAMATH_GPT_quadratic_roots_l1369_136976

theorem quadratic_roots (x : ℝ) : (x^2 + 4*x + 3 = 0) ↔ (x = -3 ∨ x = -1) := 
sorry

end NUMINAMATH_GPT_quadratic_roots_l1369_136976


namespace NUMINAMATH_GPT_cube_volume_l1369_136982

theorem cube_volume (a : ℕ) (h1 : 9 * 12 * 3 = 324) (h2 : 108 * a^3 = 324) : a^3 = 27 :=
by {
  sorry
}

end NUMINAMATH_GPT_cube_volume_l1369_136982


namespace NUMINAMATH_GPT_smallest_number_with_property_l1369_136904

theorem smallest_number_with_property: 
  ∃ (N : ℕ), N = 25 ∧ (∀ (x : ℕ) (h : N = x + (x / 5)), N ≤ x) := 
  sorry

end NUMINAMATH_GPT_smallest_number_with_property_l1369_136904


namespace NUMINAMATH_GPT_conversion_points_worth_two_l1369_136947

theorem conversion_points_worth_two
  (touchdowns_per_game : ℕ := 4)
  (points_per_touchdown : ℕ := 6)
  (games_in_season : ℕ := 15)
  (total_touchdowns_scored : ℕ := touchdowns_per_game * games_in_season)
  (total_points_from_touchdowns : ℕ := total_touchdowns_scored * points_per_touchdown)
  (old_record_points : ℕ := 300)
  (points_above_record : ℕ := 72)
  (total_points_scored : ℕ := old_record_points + points_above_record)
  (conversions_scored : ℕ := 6)
  (total_points_from_conversions : ℕ := total_points_scored - total_points_from_touchdowns) :
  total_points_from_conversions / conversions_scored = 2 := by
sorry

end NUMINAMATH_GPT_conversion_points_worth_two_l1369_136947


namespace NUMINAMATH_GPT_count_triangles_l1369_136974

-- Define the conditions for the problem
def P (x1 x2 : ℕ) : Prop := 37 * x1 ≤ 2022 ∧ 37 * x2 ≤ 2022

def valid_points (x y : ℕ) : Prop := 37 * x + y = 2022

def area_multiple_of_3 (x1 x2 : ℕ): Prop :=
  (∃ k : ℤ, 3 * k = x1 - x2) ∧ x1 ≠ x2 ∧ P x1 x2

-- The final theorem to prove the number of such distinct triangles
theorem count_triangles : 
  (∃ (n : ℕ), n = 459 ∧ 
    ∃ x1 x2 : ℕ, area_multiple_of_3 x1 x2 ∧ x1 ≠ x2) :=
by
  sorry

end NUMINAMATH_GPT_count_triangles_l1369_136974


namespace NUMINAMATH_GPT_john_toy_store_fraction_l1369_136905

theorem john_toy_store_fraction :
  let allowance := 4.80
  let arcade_spent := 3 / 5 * allowance
  let remaining_after_arcade := allowance - arcade_spent
  let candy_store_spent := 1.28
  let toy_store_spent := remaining_after_arcade - candy_store_spent
  (toy_store_spent / remaining_after_arcade) = 1 / 3 := by
    sorry

end NUMINAMATH_GPT_john_toy_store_fraction_l1369_136905


namespace NUMINAMATH_GPT_student_number_choice_l1369_136920

theorem student_number_choice (x : ℤ) (h : 2 * x - 138 = 104) : x = 121 :=
sorry

end NUMINAMATH_GPT_student_number_choice_l1369_136920


namespace NUMINAMATH_GPT_triangle_no_solution_l1369_136962

def angleSumOfTriangle : ℝ := 180

def hasNoSolution (a b A : ℝ) : Prop :=
  A >= angleSumOfTriangle

theorem triangle_no_solution {a b A : ℝ} (ha : a = 181) (hb : b = 209) (hA : A = 121) :
  hasNoSolution a b A := sorry

end NUMINAMATH_GPT_triangle_no_solution_l1369_136962


namespace NUMINAMATH_GPT_jennas_total_ticket_cost_l1369_136900

theorem jennas_total_ticket_cost :
  let normal_price := 50
  let tickets_from_website := 2 * normal_price
  let scalper_price := 2 * normal_price * 2.4 - 10
  let friend_discounted_ticket := normal_price * 0.6
  tickets_from_website + scalper_price + friend_discounted_ticket = 360 :=
by
  sorry

end NUMINAMATH_GPT_jennas_total_ticket_cost_l1369_136900


namespace NUMINAMATH_GPT_correct_regression_equation_l1369_136928

variable (x y : ℝ)

-- Assume that y is negatively correlated with x
axiom negative_correlation : x * y ≤ 0

-- The candidate regression equations
def regression_A : ℝ := -2 * x - 100
def regression_B : ℝ := 2 * x - 100
def regression_C : ℝ := -2 * x + 100
def regression_D : ℝ := 2 * x + 100

-- Prove that the correct regression equation reflecting the negative correlation is regression_C
theorem correct_regression_equation : regression_C x = -2 * x + 100 := by
  sorry

end NUMINAMATH_GPT_correct_regression_equation_l1369_136928


namespace NUMINAMATH_GPT_parallel_statements_l1369_136989

-- Definitions for lines and planes
variables {Line Plane : Type}

-- Parallelism between a line and another line or a plane
variables (a b : Line) (α : Plane)

-- Parallel relationship assertions
axiom parallel_lines (l1 l2 : Line) : Prop -- l1 is parallel to l2
axiom line_in_plane (l : Line) (p : Plane) : Prop -- line l is in plane p
axiom parallel_line_plane (l : Line) (p : Plane) : Prop -- line l is parallel to plane p

-- Problem statement
theorem parallel_statements :
  (parallel_lines a b ∧ line_in_plane b α → parallel_line_plane a α) ∧
  (parallel_lines a b ∧ parallel_line_plane a α → parallel_line_plane b α) :=
sorry

end NUMINAMATH_GPT_parallel_statements_l1369_136989


namespace NUMINAMATH_GPT_function_domain_l1369_136929

theorem function_domain (x : ℝ) : x ≠ 3 → ∃ y : ℝ, y = (1 / (x - 3)) :=
by
  sorry

end NUMINAMATH_GPT_function_domain_l1369_136929


namespace NUMINAMATH_GPT_magic_square_sum_l1369_136930

theorem magic_square_sum (a b c d e : ℕ) 
    (h1 : a + c + e = 55)
    (h2 : 30 + 10 + a = 55)
    (h3 : 30 + e + 15 = 55)
    (h4 : 10 + 30 + d = 55) :
    d + e = 25 := by
  sorry

end NUMINAMATH_GPT_magic_square_sum_l1369_136930


namespace NUMINAMATH_GPT_fred_initial_money_l1369_136918

def initial_money (book_count : ℕ) (average_cost : ℕ) (money_left : ℕ) : ℕ :=
  book_count * average_cost + money_left

theorem fred_initial_money :
  initial_money 6 37 14 = 236 :=
by
  sorry

end NUMINAMATH_GPT_fred_initial_money_l1369_136918


namespace NUMINAMATH_GPT_original_number_value_l1369_136956

theorem original_number_value (x : ℝ) (h : 0 < x) (h_eq : 10^4 * x = 4 / x) : x = 0.02 :=
sorry

end NUMINAMATH_GPT_original_number_value_l1369_136956


namespace NUMINAMATH_GPT_total_stamps_in_collection_l1369_136907

-- Definitions reflecting the problem conditions
def foreign_stamps : ℕ := 90
def old_stamps : ℕ := 60
def both_foreign_and_old_stamps : ℕ := 20
def neither_foreign_nor_old_stamps : ℕ := 70

-- The expected total number of stamps in the collection
def total_stamps : ℕ :=
  (foreign_stamps + old_stamps - both_foreign_and_old_stamps) + neither_foreign_nor_old_stamps

-- Statement to prove the total number of stamps is 200
theorem total_stamps_in_collection : total_stamps = 200 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_stamps_in_collection_l1369_136907


namespace NUMINAMATH_GPT_parabola_equation_exists_line_m_equation_exists_l1369_136916

noncomputable def problem_1 : Prop :=
  ∃ (p : ℝ), p > 0 ∧ (∀ (x y : ℝ), x^2 = 2 * p * y → y = x^2 / (2 * p)) ∧ 
  (∀ (x1 x2 y1 y2 : ℝ), x1^2 = 2 * p * y1 → x2^2 = 2 * p * y2 → 
    (y1 + y2 = 8 - p) ∧ ((y1 + y2) / 2 = 3) → p = 2)

noncomputable def problem_2 : Prop :=
  ∃ (k : ℝ), (k^2 = 1 / 4) ∧ (∀ (x : ℝ), (x^2 - 4 * k * x - 24 = 0) → 
    (∃ (x1 x2 : ℝ), x1 + x2 = 4 * k ∧ x1 * x2 = -24)) ∧
  (∀ (x1 x2 : ℝ), x1^2 = 4 * (k * x1 + 6) ∧ x2^2 = 4 * (k * x2 + 6) → 
    ∀ (x3 x4 : ℝ), (x1 * x2) ^ 2 - 4 * ((x1 + x2) ^ 2 - 2 * x1 * x2) + 16 + 16 * x1 * x2 = 0 → 
    (k = 1 / 2 ∨ k = -1 / 2))

theorem parabola_equation_exists : problem_1 :=
by {
  sorry
}

theorem line_m_equation_exists : problem_2 :=
by {
  sorry
}

end NUMINAMATH_GPT_parabola_equation_exists_line_m_equation_exists_l1369_136916


namespace NUMINAMATH_GPT_x_can_be_positive_negative_or_zero_l1369_136954

noncomputable
def characteristics_of_x (x y z w : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : w ≠ 0) 
  (h5 : (x / y) > (z / w)) (h6 : (y * w) > 0) : Prop :=
  ∃ r : ℝ, r = x

theorem x_can_be_positive_negative_or_zero (x y z w : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : w ≠ 0) (h5 : (x / y) > (z / w)) (h6 : (y * w) > 0) : 
  (characteristics_of_x x y z w h1 h2 h3 h4 h5 h6) :=
sorry

end NUMINAMATH_GPT_x_can_be_positive_negative_or_zero_l1369_136954


namespace NUMINAMATH_GPT_equal_circumradii_l1369_136970

-- Define the points and triangles involved
variable (A B C M : Type*) 

-- The circumcircle radius of a triangle is at least R
variable (R R1 R2 R3 : ℝ)

-- Hypotheses: the given conditions
variable (hR1 : R1 ≥ R)
variable (hR2 : R2 ≥ R)
variable (hR3 : R3 ≥ R)

-- The goal: to show that all four radii are equal
theorem equal_circumradii {A B C M : Type*} (R R1 R2 R3 : ℝ) 
    (hR1 : R1 ≥ R) 
    (hR2 : R2 ≥ R) 
    (hR3 : R3 ≥ R): 
    R1 = R ∧ R2 = R ∧ R3 = R := 
by 
  sorry

end NUMINAMATH_GPT_equal_circumradii_l1369_136970


namespace NUMINAMATH_GPT_circles_tangent_l1369_136903

/--
Two equal circles each with a radius of 5 are externally tangent to each other and both are internally tangent to a larger circle with a radius of 13. 
Let the points of tangency be A and B. Let AB = m/n where m and n are positive integers and gcd(m, n) = 1. 
We need to prove that m + n = 69.
-/
theorem circles_tangent (r1 r2 r3 : ℝ) (tangent_external : ℝ) (tangent_internal : ℝ) (AB : ℝ) (m n : ℕ) 
  (hmn_coprime : Nat.gcd m n = 1) (hr1 : r1 = 5) (hr2 : r2 = 5) (hr3 : r3 = 13) 
  (ht_external : tangent_external = r1 + r2) (ht_internal : tangent_internal = r3 - r1) 
  (hAB : AB = (130 / 8)): m + n = 69 :=
by
  sorry

end NUMINAMATH_GPT_circles_tangent_l1369_136903


namespace NUMINAMATH_GPT_ben_total_distance_walked_l1369_136949

-- Definitions based on conditions
def walking_speed : ℝ := 4  -- 4 miles per hour.
def total_time : ℝ := 2  -- 2 hours.
def break_time : ℝ := 0.25  -- 0.25 hours (15 minutes).

-- Proof goal: Prove that the total distance walked is 7.0 miles.
theorem ben_total_distance_walked : (walking_speed * (total_time - break_time) = 7.0) :=
by
  sorry

end NUMINAMATH_GPT_ben_total_distance_walked_l1369_136949


namespace NUMINAMATH_GPT_minimum_a_value_l1369_136986

theorem minimum_a_value (a : ℝ) : 
  (∀ (x y : ℝ), 0 < x → 0 < y → x^2 + 2 * x * y ≤ a * (x^2 + y^2)) ↔ a ≥ (Real.sqrt 5 + 1) / 2 := 
sorry

end NUMINAMATH_GPT_minimum_a_value_l1369_136986


namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_condition_l1369_136924

theorem arithmetic_sequence_geometric_condition (a : ℕ → ℤ) (d : ℤ) (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_a1 : a 1 = 1) (h_d_nonzero : d ≠ 0)
  (h_geom : (1 + d) * (1 + d) = 1 * (1 + 4 * d)) : a 2013 = 4025 := by sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_condition_l1369_136924


namespace NUMINAMATH_GPT_ratio_problem_l1369_136910

theorem ratio_problem {q r s t : ℚ} (h1 : q / r = 8) (h2 : s / r = 4) (h3 : s / t = 1 / 3) :
  t / q = 3 / 2 :=
sorry

end NUMINAMATH_GPT_ratio_problem_l1369_136910


namespace NUMINAMATH_GPT_range_of_m_l1369_136948

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, mx^2 + 2 * m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 1) := by
  sorry

end NUMINAMATH_GPT_range_of_m_l1369_136948


namespace NUMINAMATH_GPT_least_number_of_candles_l1369_136953

theorem least_number_of_candles (b : ℕ) :
  (b ≡ 5 [MOD 6]) ∧ (b ≡ 7 [MOD 8]) ∧ (b ≡ 3 [MOD 9]) → b = 119 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_least_number_of_candles_l1369_136953


namespace NUMINAMATH_GPT_fraction_simplification_l1369_136967

theorem fraction_simplification (a b : ℚ) (h : b / a = 3 / 5) : (a - b) / a = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1369_136967


namespace NUMINAMATH_GPT_parallelogram_base_is_36_l1369_136913

def parallelogram_base (area height : ℕ) : ℕ :=
  area / height

theorem parallelogram_base_is_36 (h : parallelogram_base 864 24 = 36) : True :=
by
  trivial

end NUMINAMATH_GPT_parallelogram_base_is_36_l1369_136913


namespace NUMINAMATH_GPT_find_number_l1369_136931

theorem find_number (x : ℝ) : 14 * x + 15 * x + 18 * x + 11 = 152 → x = 3 := by
  sorry

end NUMINAMATH_GPT_find_number_l1369_136931


namespace NUMINAMATH_GPT_main_theorem_l1369_136994

-- Define the interval (3π/4, π)
def theta_range (θ : ℝ) : Prop :=
  (3 * Real.pi / 4) < θ ∧ θ < Real.pi

-- Define the condition
def inequality_condition (θ x : ℝ) : Prop :=
  x^2 * Real.sin θ - x * (1 - x) + (1 - x)^2 * Real.cos θ + 2 * x * (1 - x) * Real.sqrt (Real.cos θ * Real.sin θ) > 0

-- The main theorem
theorem main_theorem (θ x : ℝ) (hθ : theta_range θ) (hx : 0 ≤ x ∧ x ≤ 1) : inequality_condition θ x :=
by
  sorry

end NUMINAMATH_GPT_main_theorem_l1369_136994


namespace NUMINAMATH_GPT_team_a_builds_per_day_l1369_136919

theorem team_a_builds_per_day (x : ℝ) (h1 : (150 / x = 100 / (2 * x - 30))) : x = 22.5 := by
  sorry

end NUMINAMATH_GPT_team_a_builds_per_day_l1369_136919


namespace NUMINAMATH_GPT_remainder_with_conditions_l1369_136966

theorem remainder_with_conditions (a b c d : ℕ) (h1 : a % 53 = 33) (h2 : b % 53 = 15) (h3 : c % 53 = 27) (h4 : d % 53 = 8) :
  ((a + b + c + d + 10) % 53) = 40 :=
by
  sorry

end NUMINAMATH_GPT_remainder_with_conditions_l1369_136966


namespace NUMINAMATH_GPT_sum_d_e_f_equals_23_l1369_136984

theorem sum_d_e_f_equals_23
  (d e f : ℤ)
  (h1 : ∀ x : ℝ, x^2 + 9 * x + 20 = (x + d) * (x + e))
  (h2 : ∀ x : ℝ, x^2 + 11 * x - 60 = (x + e) * (x - f)) :
  d + e + f = 23 :=
by
  sorry

end NUMINAMATH_GPT_sum_d_e_f_equals_23_l1369_136984


namespace NUMINAMATH_GPT_ezekiel_new_shoes_l1369_136977

-- condition Ezekiel bought 3 pairs of shoes
def pairs_of_shoes : ℕ := 3

-- condition Each pair consists of 2 shoes
def shoes_per_pair : ℕ := 2

-- proving the number of new shoes Ezekiel has
theorem ezekiel_new_shoes (pairs_of_shoes shoes_per_pair : ℕ) : pairs_of_shoes * shoes_per_pair = 6 :=
by
  sorry

end NUMINAMATH_GPT_ezekiel_new_shoes_l1369_136977


namespace NUMINAMATH_GPT_tangent_line_at_point_l1369_136969

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x - 4 * (x - 1)

theorem tangent_line_at_point (x y : ℝ) (h : f 1 = 0) (h' : deriv f 1 = -2) :
  2 * x + y - 2 = 0 :=
sorry

end NUMINAMATH_GPT_tangent_line_at_point_l1369_136969


namespace NUMINAMATH_GPT_problem1_problem2_l1369_136912

theorem problem1 (m : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = m - |x - 2|) 
  (h2 : ∀ x, f (x + 2) ≥ 0 → -1 ≤ x ∧ x ≤ 1) : 
  m = 1 := 
sorry

theorem problem2 (a b c : ℝ) 
  (h : 1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) : 
  a + 2 * b + 3 * c ≥ 9 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l1369_136912


namespace NUMINAMATH_GPT_kanul_cash_percentage_l1369_136981

theorem kanul_cash_percentage (raw_materials : ℕ) (machinery : ℕ) (total_amount : ℕ) (cash_percentage : ℕ)
  (H1 : raw_materials = 80000)
  (H2 : machinery = 30000)
  (H3 : total_amount = 137500)
  (H4 : cash_percentage = 20) :
  ((total_amount - (raw_materials + machinery)) * 100 / total_amount) = cash_percentage := by
    sorry

end NUMINAMATH_GPT_kanul_cash_percentage_l1369_136981


namespace NUMINAMATH_GPT_winner_is_Junsu_l1369_136975

def Younghee_water_intake : ℝ := 1.4
def Jimin_water_intake : ℝ := 1.8
def Junsu_water_intake : ℝ := 2.1

theorem winner_is_Junsu : 
  Junsu_water_intake > Younghee_water_intake ∧ Junsu_water_intake > Jimin_water_intake :=
by sorry

end NUMINAMATH_GPT_winner_is_Junsu_l1369_136975


namespace NUMINAMATH_GPT_tunnel_length_l1369_136933

theorem tunnel_length (x : ℕ) (y : ℕ) 
  (h1 : 300 + x = 60 * y) 
  (h2 : x - 300 = 30 * y) : 
  x = 900 := 
by
  sorry

end NUMINAMATH_GPT_tunnel_length_l1369_136933


namespace NUMINAMATH_GPT_minimize_expression_l1369_136927

theorem minimize_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^3 * y^2 * z = 1) : 
  x + 2*y + 3*z ≥ 2 :=
sorry

end NUMINAMATH_GPT_minimize_expression_l1369_136927


namespace NUMINAMATH_GPT_major_axis_length_l1369_136959

-- Define the problem setup
structure Cylinder :=
  (base_radius : ℝ)
  (height : ℝ)

structure Sphere :=
  (radius : ℝ)

-- Define the conditions
def cylinder : Cylinder :=
  { base_radius := 6, height := 0 }  -- height isn't significant for this problem

def sphere1 : Sphere :=
  { radius := 6 }

def sphere2 : Sphere :=
  { radius := 6 }

def distance_between_centers : ℝ :=
  13

-- Statement of the problem in Lean 4
theorem major_axis_length : 
  cylinder.base_radius = 6 →
  sphere1.radius = 6 →
  sphere2.radius = 6 →
  distance_between_centers = 13 →
  ∃ major_axis_length : ℝ, major_axis_length = 13 :=
by
  intros h1 h2 h3 h4
  existsi 13
  sorry

end NUMINAMATH_GPT_major_axis_length_l1369_136959


namespace NUMINAMATH_GPT_hyperbola_eccentricity_sqrt_five_l1369_136946

/-- Given a hyperbola with the equation x^2/a^2 - y^2/b^2 = 1 where a > 0 and b > 0,
and its focus lies symmetrically with respect to the asymptote lines and on the hyperbola,
proves that the eccentricity of the hyperbola is sqrt(5). -/
theorem hyperbola_eccentricity_sqrt_five 
  (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) 
  (c : ℝ) (h_focus : c^2 = 5 * a^2) : 
  (c / a = Real.sqrt 5) := sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_sqrt_five_l1369_136946


namespace NUMINAMATH_GPT_rational_function_value_l1369_136971

theorem rational_function_value (g : ℚ → ℚ) (h : ∀ x : ℚ, x ≠ 0 → 4 * g (x⁻¹) + 3 * g x / x = 2 * x^3) : g (-1) = -2 :=
sorry

end NUMINAMATH_GPT_rational_function_value_l1369_136971


namespace NUMINAMATH_GPT_solve_for_a_l1369_136985

def quadratic_has_roots (a x1 x2 : ℝ) : Prop :=
  x1 + x2 = a ∧ x1 * x2 = -6 * a^2

theorem solve_for_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : quadratic_has_roots a x1 x2) (h3 : x2 - x1 = 10) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l1369_136985


namespace NUMINAMATH_GPT_smallest_k_repr_19_pow_n_sub_5_pow_m_exists_l1369_136915

theorem smallest_k_repr_19_pow_n_sub_5_pow_m_exists :
  ∃ (k n m : ℕ), k > 0 ∧ n > 0 ∧ m > 0 ∧ k = 19 ^ n - 5 ^ m ∧ k = 14 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_repr_19_pow_n_sub_5_pow_m_exists_l1369_136915


namespace NUMINAMATH_GPT_connie_num_markers_l1369_136937

def num_red_markers (T : ℝ) := 0.41 * T
def num_total_markers (num_blue_markers : ℝ) (T : ℝ) := num_red_markers T + num_blue_markers

theorem connie_num_markers (T : ℝ) (h1 : num_total_markers 23 T = T) : T = 39 :=
by
sorry

end NUMINAMATH_GPT_connie_num_markers_l1369_136937


namespace NUMINAMATH_GPT_daisies_sold_on_fourth_day_l1369_136990

-- Number of daisies sold on the first day
def first_day_daisies : ℕ := 45

-- Number of daisies sold on the second day
def second_day_daisies : ℕ := first_day_daisies + 20

-- Number of daisies sold on the third day
def third_day_daisies : ℕ := 2 * second_day_daisies - 10

-- Total number of daisies sold in the first three days
def total_first_three_days_daisies : ℕ := first_day_daisies + second_day_daisies + third_day_daisies

-- Total number of daisies sold in four days
def total_four_days_daisies : ℕ := 350

-- Number of daisies sold on the fourth day
def fourth_day_daisies : ℕ := total_four_days_daisies - total_first_three_days_daisies

-- Theorem that states the number of daisies sold on the fourth day is 120
theorem daisies_sold_on_fourth_day : fourth_day_daisies = 120 :=
by sorry

end NUMINAMATH_GPT_daisies_sold_on_fourth_day_l1369_136990


namespace NUMINAMATH_GPT_principal_amount_is_1200_l1369_136965

-- Define the given conditions
def simple_interest (P : ℝ) : ℝ := 0.10 * P
def compound_interest (P : ℝ) : ℝ := 0.1025 * P

-- Define given difference
def interest_difference (P : ℝ) : ℝ := compound_interest P - simple_interest P

-- The main goal is to prove that the principal amount P that satisfies the difference condition is 1200
theorem principal_amount_is_1200 : ∃ P : ℝ, interest_difference P = 3 ∧ P = 1200 :=
by
  sorry -- Proof to be completed

end NUMINAMATH_GPT_principal_amount_is_1200_l1369_136965


namespace NUMINAMATH_GPT_hancho_height_l1369_136952

theorem hancho_height (Hansol_height : ℝ) (h1 : Hansol_height = 134.5) (ratio : ℝ) (h2 : ratio = 1.06) :
  Hansol_height * ratio = 142.57 := by
  sorry

end NUMINAMATH_GPT_hancho_height_l1369_136952


namespace NUMINAMATH_GPT_correct_calculation_l1369_136978

variable {a : ℝ}

theorem correct_calculation : a^2 * a^3 = a^5 :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l1369_136978


namespace NUMINAMATH_GPT_trader_marked_price_percentage_above_cost_price_l1369_136996

theorem trader_marked_price_percentage_above_cost_price 
  (CP MP SP : ℝ) 
  (discount loss : ℝ)
  (h_discount : discount = 0.07857142857142857)
  (h_loss : loss = 0.01)
  (h_SP_discount : SP = MP * (1 - discount))
  (h_SP_loss : SP = CP * (1 - loss)) :
  (MP / CP - 1) * 100 = 7.4285714285714 := 
sorry

end NUMINAMATH_GPT_trader_marked_price_percentage_above_cost_price_l1369_136996


namespace NUMINAMATH_GPT_negation_of_proposition_l1369_136957

variables (x : ℝ)

def proposition (x : ℝ) : Prop := x > 0 → (x ≠ 2 → (x^3 / (x - 2) > 0))

theorem negation_of_proposition : ∃ x : ℝ, x > 0 ∧ 0 ≤ x ∧ x ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1369_136957


namespace NUMINAMATH_GPT_semicircle_radius_l1369_136921

noncomputable def radius_of_semicircle (P : ℝ) : ℝ :=
  P / (Real.pi + 2)

theorem semicircle_radius (P : ℝ) (hP : P = 180) : radius_of_semicircle P = 180 / (Real.pi + 2) :=
by
  sorry

end NUMINAMATH_GPT_semicircle_radius_l1369_136921


namespace NUMINAMATH_GPT_rectangle_diagonal_length_l1369_136992

theorem rectangle_diagonal_length
    (PQ QR : ℝ) (RT RU ST : ℝ) (Area_RST : ℝ)
    (hPQ : PQ = 8) (hQR : QR = 10)
    (hRT_RU : RT = RU)
    (hArea_RST: Area_RST = (1/5) * (PQ * QR)) :
    ST = 8 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_diagonal_length_l1369_136992


namespace NUMINAMATH_GPT_problem1_problem2_l1369_136944

-- Define the function f
def f (x b : ℝ) := |2 * x + b|

-- First problem: prove if the solution set of |2x + b| <= 3 is {x | -1 ≤ x ≤ 2}, then b = -1.
theorem problem1 (b : ℝ) : (∀ x : ℝ, (-1 ≤ x ∧ x ≤ 2 → |2 * x + b| ≤ 3)) → b = -1 :=
sorry

-- Second problem: given b = -1, prove that for all x ∈ ℝ, |2(x+3)-1| + |2(x+1)-1| ≥ -4.
theorem problem2 : (∀ x : ℝ, f (x + 3) (-1) + f (x + 1) (-1) ≥ -4) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1369_136944


namespace NUMINAMATH_GPT_cost_per_square_inch_l1369_136997

def length : ℕ := 9
def width : ℕ := 12
def total_cost : ℕ := 432

theorem cost_per_square_inch :
  total_cost / ((length * width) / 2) = 8 := 
by 
  sorry

end NUMINAMATH_GPT_cost_per_square_inch_l1369_136997


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1369_136908

theorem sufficient_but_not_necessary_condition (a1 d : ℝ) : 
  (2 * a1 + 11 * d > 0) → (2 * a1 + 11 * d ≥ 0) :=
by
  intro h
  apply le_of_lt
  exact h

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1369_136908


namespace NUMINAMATH_GPT_largest_common_term_in_sequences_l1369_136901

/-- An arithmetic sequence starts with 3 and has a common difference of 10. A second sequence starts
with 5 and has a common difference of 8. In the range of 1 to 150, the largest number common to 
both sequences is 133. -/
theorem largest_common_term_in_sequences : ∃ (b : ℕ), b < 150 ∧ (∃ (n m : ℤ), b = 3 + 10 * n ∧ b = 5 + 8 * m) ∧ (b = 133) := 
by
  sorry

end NUMINAMATH_GPT_largest_common_term_in_sequences_l1369_136901


namespace NUMINAMATH_GPT_odometer_problem_l1369_136945

theorem odometer_problem (a b c : ℕ) (h₀ : a + b + c = 7) (h₁ : 1 ≤ a)
  (h₂ : a < 10) (h₃ : b < 10) (h₄ : c < 10) (h₅ : (c - a) % 20 = 0) : a^2 + b^2 + c^2 = 37 := 
  sorry

end NUMINAMATH_GPT_odometer_problem_l1369_136945
