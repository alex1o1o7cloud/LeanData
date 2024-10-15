import Mathlib

namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l478_47888

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a < -1) → (|a| > 1) ∧ ¬((|a| > 1) → (a < -1)) :=
by
-- This statement represents the required proof.
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l478_47888


namespace NUMINAMATH_GPT_find_y_l478_47857

theorem find_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x = 2 + 1/y) (h2 : y = 2 + 1/x) : y = x :=
sorry

end NUMINAMATH_GPT_find_y_l478_47857


namespace NUMINAMATH_GPT_A_not_divisible_by_B_l478_47800

variable (A B : ℕ)
variable (h1 : A ≠ B)
variable (h2 : (∀ i, (1 ≤ i ∧ i ≤ 7) → (∃! j, (1 ≤ j ∧ j ≤ 7) ∧ (j = i))))
variable (h3 : (∀ i, (1 ≤ i ∧ i ≤ 7) → (∃! j, (1 ≤ j ∧ j ≤ 7) ∧ (j = i))))

theorem A_not_divisible_by_B : ¬ (A % B = 0) :=
sorry

end NUMINAMATH_GPT_A_not_divisible_by_B_l478_47800


namespace NUMINAMATH_GPT_odd_function_expression_l478_47899

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_expression (x : ℝ) (h1 : x < 0 → f x = x^2 - x) (h2 : ∀ x, f (-x) = -f x) (h3 : 0 < x) :
  f x = -x^2 - x :=
sorry

end NUMINAMATH_GPT_odd_function_expression_l478_47899


namespace NUMINAMATH_GPT_remainder_of_power_l478_47859

theorem remainder_of_power :
  (4^215) % 9 = 7 := by
sorry

end NUMINAMATH_GPT_remainder_of_power_l478_47859


namespace NUMINAMATH_GPT_negation_of_universal_prop_l478_47858

theorem negation_of_universal_prop :
  (¬ (∀ x : ℝ, x^2 - 5 * x + 3 ≤ 0)) ↔ (∃ x : ℝ, x^2 - 5 * x + 3 > 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_prop_l478_47858


namespace NUMINAMATH_GPT_heather_initial_oranges_l478_47860

theorem heather_initial_oranges (given_oranges: ℝ) (total_oranges: ℝ) (initial_oranges: ℝ) 
    (h1: given_oranges = 35.0) 
    (h2: total_oranges = 95) : 
    initial_oranges = 60 :=
by
  sorry

end NUMINAMATH_GPT_heather_initial_oranges_l478_47860


namespace NUMINAMATH_GPT_repeating_decimal_as_fraction_l478_47850

-- Define the repeating decimal
def repeating_decimal_2_35 := 2 + (35 / 99 : ℚ)

-- Define the fraction form
def fraction_form := (233 / 99 : ℚ)

-- Theorem statement asserting the equivalence
theorem repeating_decimal_as_fraction : repeating_decimal_2_35 = fraction_form :=
by 
  -- Skipped proof
  sorry

end NUMINAMATH_GPT_repeating_decimal_as_fraction_l478_47850


namespace NUMINAMATH_GPT_acute_triangle_altitude_inequality_l478_47813

theorem acute_triangle_altitude_inequality (a b c d e f : ℝ) 
  (A B C : ℝ) 
  (acute_triangle : (d = b * Real.sin C) ∧ (d = c * Real.sin B) ∧
                    (e = a * Real.sin C) ∧ (f = a * Real.sin B))
  (projections : (de = b * Real.cos B) ∧ (df = c * Real.cos C))
  : (de + df ≤ a) := 
sorry

end NUMINAMATH_GPT_acute_triangle_altitude_inequality_l478_47813


namespace NUMINAMATH_GPT_symmetry_y_axis_l478_47884

theorem symmetry_y_axis (A B C D : ℝ → ℝ → Prop) 
  (A_eq : ∀ x y : ℝ, A x y ↔ (x^2 - x + y^2 = 1))
  (B_eq : ∀ x y : ℝ, B x y ↔ (x^2 * y + x * y^2 = 1))
  (C_eq : ∀ x y : ℝ, C x y ↔ (x^2 - y^2 = 1))
  (D_eq : ∀ x y : ℝ, D x y ↔ (x - y = 1)) : 
  (∀ x y : ℝ, C x y ↔ C (-x) y) ∧ 
  ¬(∀ x y : ℝ, A x y ↔ A (-x) y) ∧ 
  ¬(∀ x y : ℝ, B x y ↔ B (-x) y) ∧ 
  ¬(∀ x y : ℝ, D x y ↔ D (-x) y) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_symmetry_y_axis_l478_47884


namespace NUMINAMATH_GPT_general_formula_an_general_formula_bn_exists_arithmetic_sequence_bn_l478_47844

variable (a_n : ℕ → ℝ)
variable (b_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)
variable (d : ℝ)

-- Define the initial conditions
axiom a2_a3_condition : a_n 2 * a_n 3 = 15
axiom S4_condition : S_n 4 = 16
axiom b_recursion : ∀ (n : ℕ), b_n (n + 1) - b_n n = 1 / (a_n n * a_n (n + 1))

-- Define the proofs
theorem general_formula_an : ∀ (n : ℕ), a_n n = 2 * n - 1 :=
sorry

theorem general_formula_bn : ∀ (n : ℕ), b_n n = (3 * n - 2) / (2 * n - 1) :=
sorry

theorem exists_arithmetic_sequence_bn : ∃ (m n : ℕ), m ≠ n ∧ b_n 2 + b_n n = 2 * b_n m ∧ b_n 2 = 4 / 3 ∧ (n = 8 ∧ m = 3) :=
sorry

end NUMINAMATH_GPT_general_formula_an_general_formula_bn_exists_arithmetic_sequence_bn_l478_47844


namespace NUMINAMATH_GPT_polygon_sides_l478_47871

/-- If the sum of the interior angles of a polygon is three times the sum of its exterior angles,
    then the number of sides of the polygon is 8. -/
theorem polygon_sides (n : ℕ) (h1 : 180 * (n - 2) = 3 * 360) : n = 8 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l478_47871


namespace NUMINAMATH_GPT_rectangle_perimeter_l478_47865
-- Refined definitions and setup
variables (AB BC AE BE CF : ℝ)
-- Conditions provided in the problem
def conditions := AB = 2 * BC ∧ AE = 10 ∧ BE = 26 ∧ CF = 5
-- Perimeter calculation based on the conditions
def perimeter (AB BC : ℝ) : ℝ := 2 * (AB + BC)
-- Main theorem stating the conditions and required result
theorem rectangle_perimeter {m n : ℕ} (h: conditions AB BC AE BE CF) :
  m + n = 105 ∧ Int.gcd m n = 1 ∧ perimeter AB BC = m / n := sorry

end NUMINAMATH_GPT_rectangle_perimeter_l478_47865


namespace NUMINAMATH_GPT_tennis_preference_combined_percentage_l478_47853

theorem tennis_preference_combined_percentage :
  let total_north_students := 1500
  let total_south_students := 1800
  let north_tennis_percentage := 0.30
  let south_tennis_percentage := 0.35
  let north_tennis_students := total_north_students * north_tennis_percentage
  let south_tennis_students := total_south_students * south_tennis_percentage
  let total_tennis_students := north_tennis_students + south_tennis_students
  let total_students := total_north_students + total_south_students
  let combined_percentage := (total_tennis_students / total_students) * 100
  combined_percentage = 33 := 
by
  sorry

end NUMINAMATH_GPT_tennis_preference_combined_percentage_l478_47853


namespace NUMINAMATH_GPT_value_of_x_l478_47866

theorem value_of_x (x : ℝ) (h : x = 88 + 0.25 * 88) : x = 110 :=
sorry

end NUMINAMATH_GPT_value_of_x_l478_47866


namespace NUMINAMATH_GPT_value_of_linear_combination_l478_47898

theorem value_of_linear_combination :
  ∀ (x1 x2 x3 x4 x5 : ℝ),
    2*x1 + x2 + x3 + x4 + x5 = 6 →
    x1 + 2*x2 + x3 + x4 + x5 = 12 →
    x1 + x2 + 2*x3 + x4 + x5 = 24 →
    x1 + x2 + x3 + 2*x4 + x5 = 48 →
    x1 + x2 + x3 + x4 + 2*x5 = 96 →
    3*x4 + 2*x5 = 181 :=
by
  intros x1 x2 x3 x4 x5 h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_value_of_linear_combination_l478_47898


namespace NUMINAMATH_GPT_afternoon_sales_l478_47894

theorem afternoon_sales (x : ℕ) (h : 3 * x = 510) : 2 * x = 340 :=
by sorry

end NUMINAMATH_GPT_afternoon_sales_l478_47894


namespace NUMINAMATH_GPT_min_value_of_expression_l478_47805

noncomputable def f (m : ℝ) : ℝ :=
  let x1 := -m - (m^2 + 3 * m - 2)
  let x2 := -2 * m - x1
  x1 * (x2 + x1) + x2^2

theorem min_value_of_expression :
  ∃ m : ℝ, f m = 3 * (m - 1/2)^2 + 5/4 ∧ f m ≥ f (1/2) := by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l478_47805


namespace NUMINAMATH_GPT_rachel_older_than_leah_l478_47893

theorem rachel_older_than_leah (rachel_age leah_age : ℕ) (h1 : rachel_age = 19) (h2 : rachel_age + leah_age = 34) :
  rachel_age - leah_age = 4 :=
by sorry

end NUMINAMATH_GPT_rachel_older_than_leah_l478_47893


namespace NUMINAMATH_GPT_find_h_l478_47825

theorem find_h (h j k : ℤ) (y_intercept1 : 3 * h ^ 2 + j = 2013) 
  (y_intercept2 : 2 * h ^ 2 + k = 2014)
  (x_intercepts1 : ∃ (y : ℤ), j = -3 * y ^ 2)
  (x_intercepts2 : ∃ (x : ℤ), k = -2 * x ^ 2) :
  h = 36 :=
by sorry

end NUMINAMATH_GPT_find_h_l478_47825


namespace NUMINAMATH_GPT_square_form_l478_47804

theorem square_form (m n : ℤ) : 
  ∃ k l : ℤ, (2 * m^2 + n^2)^2 = 2 * k^2 + l^2 :=
by
  let x := (2 * m^2 + n^2)
  let y := x^2
  let k := 2 * m * n
  let l := 2 * m^2 - n^2
  use k, l
  sorry

end NUMINAMATH_GPT_square_form_l478_47804


namespace NUMINAMATH_GPT_ratio_of_candies_l478_47880

theorem ratio_of_candies (candiesEmily candiesBob : ℕ) (candiesJennifer : ℕ) 
  (hEmily : candiesEmily = 6) 
  (hBob : candiesBob = 4)
  (hJennifer : candiesJennifer = 3 * candiesBob) : 
  (candiesJennifer / Nat.gcd candiesJennifer candiesEmily) = 2 ∧ (candiesEmily / Nat.gcd candiesJennifer candiesEmily) = 1 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_candies_l478_47880


namespace NUMINAMATH_GPT_range_of_b_l478_47897

-- Given a function f(x)
def f (b x : ℝ) : ℝ := x^3 - 3 * b * x + 3 * b

-- Derivative of the function f(x)
def f' (b x : ℝ) : ℝ := 3 * x^2 - 3 * b

-- The theorem to prove the range of b
theorem range_of_b (b : ℝ) : (∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ f' b x = 0) → (0 < b ∧ b < 1) := by
  sorry

end NUMINAMATH_GPT_range_of_b_l478_47897


namespace NUMINAMATH_GPT_fraction_of_cream_in_cup1_after_operations_l478_47848

/-
We consider two cups of liquids with the following contents initially:
Cup 1 has 6 ounces of coffee.
Cup 2 has 2 ounces of coffee and 4 ounces of cream.
After pouring half of Cup 1's content into Cup 2, stirring, and then pouring half of Cup 2's new content back into Cup 1, we need to show that 
the fraction of the liquid in Cup 1 that is now cream is 4/15.
-/

theorem fraction_of_cream_in_cup1_after_operations :
  let cup1_initial_coffee := 6
  let cup2_initial_coffee := 2
  let cup2_initial_cream := 4
  let cup2_initial_liquid := cup2_initial_coffee + cup2_initial_cream
  let cup1_to_cup2_coffee := cup1_initial_coffee / 2
  let cup1_final_coffee := cup1_initial_coffee - cup1_to_cup2_coffee
  let cup2_final_coffee := cup2_initial_coffee + cup1_to_cup2_coffee
  let cup2_final_liquid := cup2_final_coffee + cup2_initial_cream
  let cup2_to_cup1_liquid := cup2_final_liquid / 2
  let cup2_coffee_fraction := cup2_final_coffee / cup2_final_liquid
  let cup2_cream_fraction := cup2_initial_cream / cup2_final_liquid
  let cup2_to_cup1_coffee := cup2_to_cup1_liquid * cup2_coffee_fraction
  let cup2_to_cup1_cream := cup2_to_cup1_liquid * cup2_cream_fraction
  let cup1_final_liquid_coffee := cup1_final_coffee + cup2_to_cup1_coffee
  let cup1_final_liquid_cream := cup2_to_cup1_cream
  let cup1_final_liquid := cup1_final_liquid_coffee + cup1_final_liquid_cream
  (cup1_final_liquid_cream / cup1_final_liquid) = 4 / 15 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_cream_in_cup1_after_operations_l478_47848


namespace NUMINAMATH_GPT_cookie_total_l478_47840

-- Definitions of the conditions
def rows_large := 5
def rows_medium := 4
def rows_small := 6
def cookies_per_row_large := 6
def cookies_per_row_medium := 7
def cookies_per_row_small := 8
def number_of_trays := 4
def extra_row_large_first_tray := 1
def total_large_cookies := rows_large * cookies_per_row_large * number_of_trays + extra_row_large_first_tray * cookies_per_row_large
def total_medium_cookies := rows_medium * cookies_per_row_medium * number_of_trays
def total_small_cookies := rows_small * cookies_per_row_small * number_of_trays

-- Theorem to prove the total number of cookies is 430
theorem cookie_total : 
  total_large_cookies + total_medium_cookies + total_small_cookies = 430 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_cookie_total_l478_47840


namespace NUMINAMATH_GPT_exists_positive_integer_divisible_by_14_with_sqrt_between_25_and_25_3_l478_47831

theorem exists_positive_integer_divisible_by_14_with_sqrt_between_25_and_25_3 :
  ∃ (x : ℕ), x % 14 = 0 ∧ 625 <= x ∧ x <= 640 ∧ x = 630 := 
by 
  sorry

end NUMINAMATH_GPT_exists_positive_integer_divisible_by_14_with_sqrt_between_25_and_25_3_l478_47831


namespace NUMINAMATH_GPT_probability_toner_never_displayed_l478_47891

theorem probability_toner_never_displayed:
  let total_votes := 129
  let toner_votes := 63
  let celery_votes := 66
  (toner_votes + celery_votes = total_votes) →
  let probability := (celery_votes - toner_votes) / (celery_votes + toner_votes)
  probability = 1 / 43 := 
by
  sorry

end NUMINAMATH_GPT_probability_toner_never_displayed_l478_47891


namespace NUMINAMATH_GPT_refrigerator_cost_l478_47854

theorem refrigerator_cost
  (R : ℝ)
  (mobile_phone_cost : ℝ := 8000)
  (loss_percent_refrigerator : ℝ := 0.04)
  (profit_percent_mobile_phone : ℝ := 0.09)
  (overall_profit : ℝ := 120)
  (selling_price_refrigerator : ℝ := 0.96 * R)
  (selling_price_mobile_phone : ℝ := 8720)
  (total_selling_price : ℝ := selling_price_refrigerator + selling_price_mobile_phone)
  (total_cost_price : ℝ := R + mobile_phone_cost)
  (balance_profit_eq : total_selling_price = total_cost_price + overall_profit):
  R = 15000 :=
by
  sorry

end NUMINAMATH_GPT_refrigerator_cost_l478_47854


namespace NUMINAMATH_GPT_find_k_for_infinite_solutions_l478_47824

noncomputable def has_infinitely_many_solutions (k : ℝ) : Prop :=
  ∀ x : ℝ, 5 * (3 * x - k) = 3 * (5 * x + 15)

theorem find_k_for_infinite_solutions :
  has_infinitely_many_solutions (-9) :=
by
  sorry

end NUMINAMATH_GPT_find_k_for_infinite_solutions_l478_47824


namespace NUMINAMATH_GPT_bird_families_flew_away_l478_47821

theorem bird_families_flew_away (original : ℕ) (left : ℕ) (flew_away : ℕ) (h1 : original = 67) (h2 : left = 35) (h3 : flew_away = original - left) : flew_away = 32 :=
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_bird_families_flew_away_l478_47821


namespace NUMINAMATH_GPT_value_of_5_T_3_l478_47829

def operation (a b : ℕ) : ℕ := 4 * a + 6 * b

theorem value_of_5_T_3 : operation 5 3 = 38 :=
by
  -- proof (which is not required)
  sorry

end NUMINAMATH_GPT_value_of_5_T_3_l478_47829


namespace NUMINAMATH_GPT_keats_library_percentage_increase_l478_47896

theorem keats_library_percentage_increase :
  let total_books_A := 8000
  let total_books_B := 10000
  let total_books_C := 12000
  let initial_bio_A := 0.20 * total_books_A
  let initial_bio_B := 0.25 * total_books_B
  let initial_bio_C := 0.28 * total_books_C
  let total_initial_bio := initial_bio_A + initial_bio_B + initial_bio_C
  let final_bio_A := 0.32 * total_books_A
  let final_bio_B := 0.35 * total_books_B
  let final_bio_C := 0.40 * total_books_C
  --
  let total_final_bio := final_bio_A + final_bio_B + final_bio_C
  let increase_in_bio := total_final_bio - total_initial_bio
  let percentage_increase := (increase_in_bio / total_initial_bio) * 100
  --
  percentage_increase = 45.58 := 
by
  sorry

end NUMINAMATH_GPT_keats_library_percentage_increase_l478_47896


namespace NUMINAMATH_GPT_inverse_of_49_mod_89_l478_47823

theorem inverse_of_49_mod_89 (h : (7 * 55 ≡ 1 [MOD 89])) : (49 * 1 ≡ 1 [MOD 89]) := 
by
  sorry

end NUMINAMATH_GPT_inverse_of_49_mod_89_l478_47823


namespace NUMINAMATH_GPT_platform_length_l478_47863

theorem platform_length (train_length : ℕ) (time_post : ℕ) (time_platform : ℕ) (speed : ℕ)
    (h1 : train_length = 150)
    (h2 : time_post = 15)
    (h3 : time_platform = 25)
    (h4 : speed = train_length / time_post)
    : (train_length + 100) / time_platform = speed :=
by
  sorry

end NUMINAMATH_GPT_platform_length_l478_47863


namespace NUMINAMATH_GPT_Bruce_initial_eggs_l478_47830

variable (B : ℕ)

theorem Bruce_initial_eggs (h : B - 70 = 5) : B = 75 := by
  sorry

end NUMINAMATH_GPT_Bruce_initial_eggs_l478_47830


namespace NUMINAMATH_GPT_fib_subsequence_fib_l478_47828

noncomputable def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | 1     => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

theorem fib_subsequence_fib (p : ℕ) (hp : p > 0) :
  ∀ n : ℕ, fibonacci ((n - 1) * p) + fibonacci (n * p) = fibonacci ((n + 1) * p) := 
by
  sorry

end NUMINAMATH_GPT_fib_subsequence_fib_l478_47828


namespace NUMINAMATH_GPT_triangle_side_AC_l478_47812

theorem triangle_side_AC (B : Real) (BC AB : Real) (AC : Real) (hB : B = 30 * Real.pi / 180) (hBC : BC = 2) (hAB : AB = Real.sqrt 3) : AC = 1 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_AC_l478_47812


namespace NUMINAMATH_GPT_travel_time_l478_47810

theorem travel_time (time_Ngapara_Zipra : ℝ) 
  (h1 : time_Ngapara_Zipra = 60) 
  (h2 : ∃ time_Ningi_Zipra, time_Ningi_Zipra = 0.8 * time_Ngapara_Zipra) 
  : ∃ total_travel_time, total_travel_time = time_Ningi_Zipra + time_Ngapara_Zipra ∧ total_travel_time = 108 := 
by
  sorry

end NUMINAMATH_GPT_travel_time_l478_47810


namespace NUMINAMATH_GPT_domain_of_g_l478_47875

noncomputable def g (x : ℝ) : ℝ := Real.logb 3 (Real.logb 4 (Real.logb 5 (Real.logb 6 x)))

theorem domain_of_g : {x : ℝ | x > 6^625} = {x : ℝ | ∃ y : ℝ, y = g x } := sorry

end NUMINAMATH_GPT_domain_of_g_l478_47875


namespace NUMINAMATH_GPT_equality_equiv_l478_47890

-- Problem statement
theorem equality_equiv (a b c : ℝ) :
  (a + b + c ≠ 0 → ( (a * (b - c)) / (b + c) + (b * (c - a)) / (c + a) + (c * (a - b)) / (a + b) = 0 ↔
  (a^2 * (b - c)) / (b + c) + (b^2 * (c - a)) / (c + a) + (c^2 * (a - b)) / (a + b) = 0)) ∧
  (a + b + c = 0 → ∀ w x y z: ℝ, w * x + y * z = 0) :=
by
  sorry

end NUMINAMATH_GPT_equality_equiv_l478_47890


namespace NUMINAMATH_GPT_train_cross_signal_pole_time_l478_47845

theorem train_cross_signal_pole_time :
  ∀ (l_t l_p t_p : ℕ), l_t = 450 → l_p = 525 → t_p = 39 → 
  (l_t * t_p) / (l_t + l_p) = 18 := by
  sorry

end NUMINAMATH_GPT_train_cross_signal_pole_time_l478_47845


namespace NUMINAMATH_GPT_x_squared_y_cubed_plus_y_squared_x_cubed_eq_zero_l478_47818

theorem x_squared_y_cubed_plus_y_squared_x_cubed_eq_zero
  (x y : ℝ)
  (h1 : x + y = 2)
  (h2 : x * y = 4) : x^2 * y^3 + y^2 * x^3 = 0 := 
sorry

end NUMINAMATH_GPT_x_squared_y_cubed_plus_y_squared_x_cubed_eq_zero_l478_47818


namespace NUMINAMATH_GPT_cube_volume_surface_area_value_l478_47841

theorem cube_volume_surface_area_value (x : ℝ) : 
  (∃ s : ℝ, s = (6 * x)^(1 / 3) ∧ 6 * s^2 = 2 * x) → 
  x = 1 / 972 :=
by {
  sorry
}

end NUMINAMATH_GPT_cube_volume_surface_area_value_l478_47841


namespace NUMINAMATH_GPT_payback_period_l478_47827

def system_unit_cost : ℕ := 9499 -- cost in RUB
def graphics_card_cost : ℕ := 20990 -- cost per card in RUB
def num_graphics_cards : ℕ := 2
def system_unit_power : ℕ := 120 -- power in watts
def graphics_card_power : ℕ := 185 -- power per card in watts
def earnings_per_card_per_day_ethereum : ℚ := 0.00630
def ethereum_to_rub : ℚ := 27790.37 -- RUB per ETH
def electricity_cost_per_kwh : ℚ := 5.38 -- RUB per kWh
def total_investment : ℕ := system_unit_cost + num_graphics_cards * graphics_card_cost
def total_power_consumption_watts : ℕ := system_unit_power + num_graphics_cards * graphics_card_power
def total_power_consumption_kwh_per_day : ℚ := total_power_consumption_watts / 1000 * 24
def daily_earnings_rub : ℚ := earnings_per_card_per_day_ethereum * num_graphics_cards * ethereum_to_rub
def daily_energy_cost : ℚ := total_power_consumption_kwh_per_day * electricity_cost_per_kwh
def net_daily_profit : ℚ := daily_earnings_rub - daily_energy_cost

theorem payback_period : total_investment / net_daily_profit = 179 := by
  sorry

end NUMINAMATH_GPT_payback_period_l478_47827


namespace NUMINAMATH_GPT_interval_monotonicity_no_zeros_min_a_l478_47886

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

theorem interval_monotonicity (a : ℝ) :
  a = 1 →
  (∀ x, 0 < x ∧ x ≤ 2 → f a x < f a (x+1)) ∧
  (∀ x, x ≥ 2 → f a x < f a (x-1)) :=
by
  sorry

theorem no_zeros_min_a : 
  (∀ x, x ∈ Set.Ioo 0 (1/2 : ℝ) → f a x ≠ 0) →
  a ≥ 2 - 4 * Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_interval_monotonicity_no_zeros_min_a_l478_47886


namespace NUMINAMATH_GPT_clerks_needed_eq_84_l478_47889

def forms_processed_per_hour : ℕ := 25
def type_a_forms_count : ℕ := 3000
def type_b_forms_count : ℕ := 4000
def type_a_form_time_minutes : ℕ := 3
def type_b_form_time_minutes : ℕ := 4
def working_hours_per_day : ℕ := 5
def total_minutes_in_an_hour : ℕ := 60
def forms_time_needed (count : ℕ) (time_per_form : ℕ) : ℕ := count * time_per_form
def total_forms_time_needed : ℕ := forms_time_needed type_a_forms_count type_a_form_time_minutes +
                                    forms_time_needed type_b_forms_count type_b_form_time_minutes
def total_hours_needed : ℕ := total_forms_time_needed / total_minutes_in_an_hour
def clerk_hours_needed : ℕ := total_hours_needed / working_hours_per_day
def required_clerks : ℕ := Nat.ceil (clerk_hours_needed)

theorem clerks_needed_eq_84 :
  required_clerks = 84 :=
by
  sorry

end NUMINAMATH_GPT_clerks_needed_eq_84_l478_47889


namespace NUMINAMATH_GPT_wall_paint_area_l478_47892

theorem wall_paint_area
  (A₁ : ℕ) (A₂ : ℕ) (A₃ : ℕ) (A₄ : ℕ)
  (H₁ : A₁ = 32)
  (H₂ : A₂ = 48)
  (H₃ : A₃ = 32)
  (H₄ : A₄ = 48) :
  A₁ + A₂ + A₃ + A₄ = 160 :=
by
  sorry

end NUMINAMATH_GPT_wall_paint_area_l478_47892


namespace NUMINAMATH_GPT_total_rowing_proof_l478_47856

def morning_rowing := 13
def afternoon_rowing := 21
def total_rowing := 34

theorem total_rowing_proof :
  morning_rowing + afternoon_rowing = total_rowing :=
by
  sorry

end NUMINAMATH_GPT_total_rowing_proof_l478_47856


namespace NUMINAMATH_GPT_jake_delay_l478_47802

-- Define the conditions as in a)
def floors_jake_descends : ℕ := 8
def steps_per_floor : ℕ := 30
def steps_per_second_jake : ℕ := 3
def elevator_time_seconds : ℕ := 60 -- 1 minute = 60 seconds

-- Define the statement based on c)
theorem jake_delay (floors : ℕ) (steps_floor : ℕ) (steps_second : ℕ) (elevator_time : ℕ) :
  (floors = floors_jake_descends) →
  (steps_floor = steps_per_floor) →
  (steps_second = steps_per_second_jake) →
  (elevator_time = elevator_time_seconds) →
  (floors * steps_floor / steps_second - elevator_time = 20) :=
by
  intros
  sorry

end NUMINAMATH_GPT_jake_delay_l478_47802


namespace NUMINAMATH_GPT_tom_and_mary_age_l478_47882

-- Define Tom's and Mary's ages
variables (T M : ℕ)

-- Define the two given conditions
def condition1 : Prop := T^2 + M = 62
def condition2 : Prop := M^2 + T = 176

-- State the theorem
theorem tom_and_mary_age (h1 : condition1 T M) (h2 : condition2 T M) : T = 7 ∧ M = 13 :=
by {
  -- sorry acts as a placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_tom_and_mary_age_l478_47882


namespace NUMINAMATH_GPT_max_non_overlapping_areas_l478_47868

theorem max_non_overlapping_areas (n : ℕ) : 
  ∃ (max_areas : ℕ), max_areas = 3 * n := by
  sorry

end NUMINAMATH_GPT_max_non_overlapping_areas_l478_47868


namespace NUMINAMATH_GPT_trapezoid_area_l478_47819

def isosceles_triangle (Δ : Type) (A B C : Δ) : Prop :=
  -- Define the property that triangle ABC is isosceles with AB = AC
  sorry

def similar_triangles (Δ₁ Δ₂ : Type) (A₁ B₁ C₁ : Δ₁) (A₂ B₂ C₂ : Δ₂) : Prop :=
  -- Define the property that triangles Δ₁ and Δ₂ are similar
  sorry

def area (Δ : Type) (A B C : Δ) : ℝ :=
  -- Define the area of a triangle Δ with vertices A, B, and C
  sorry

theorem trapezoid_area
  (Δ : Type)
  {A B C D E : Δ}
  (ABC_is_isosceles : isosceles_triangle Δ A B C)
  (all_similar : ∀ (Δ₁ Δ₂ : Type) (A₁ B₁ C₁ : Δ₁) (A₂ B₂ C₂ : Δ₂), 
    similar_triangles Δ₁ Δ₂ A₁ B₁ C₁ A₂ B₂ C₂ → (area Δ₁ A₁ B₁ C₁ = 1 → area Δ₂ A₂ B₂ C₂ = 1))
  (smallest_triangles_area : area Δ A B C = 50)
  (area_ADE : area Δ A D E = 5) :
  area Δ D B C + area Δ C E B = 45 := 
sorry

end NUMINAMATH_GPT_trapezoid_area_l478_47819


namespace NUMINAMATH_GPT_perimeter_of_square_C_l478_47883

theorem perimeter_of_square_C (s_A s_B s_C : ℕ) (hpA : 4 * s_A = 16) (hpB : 4 * s_B = 32) (hC : s_C = s_A + s_B - 2) :
  4 * s_C = 40 := 
by
  sorry

end NUMINAMATH_GPT_perimeter_of_square_C_l478_47883


namespace NUMINAMATH_GPT_find_c_l478_47847

theorem find_c (a b c : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c)
(h_asc : a < b) (h_asc2 : b < c)
(h_sum : a + b + c = 11)
(h_eq : 1 / a + 1 / b + 1 / c = 1) : c = 6 := 
sorry

end NUMINAMATH_GPT_find_c_l478_47847


namespace NUMINAMATH_GPT_measles_cases_in_1990_l478_47843

noncomputable def measles_cases_1970 := 480000
noncomputable def measles_cases_2000 := 600
noncomputable def years_between := 2000 - 1970
noncomputable def total_decrease := measles_cases_1970 - measles_cases_2000
noncomputable def decrease_per_year := total_decrease / years_between
noncomputable def years_from_1970_to_1990 := 1990 - 1970
noncomputable def decrease_to_1990 := years_from_1970_to_1990 * decrease_per_year
noncomputable def measles_cases_1990 := measles_cases_1970 - decrease_to_1990

theorem measles_cases_in_1990 : measles_cases_1990 = 160400 := by
  sorry

end NUMINAMATH_GPT_measles_cases_in_1990_l478_47843


namespace NUMINAMATH_GPT_triangle_area_202_2192_pi_squared_l478_47870

noncomputable def triangle_area (a b c : ℝ) : ℝ := 
  let r := (a + b + c) / (2 * Real.pi)
  let theta := 20.0 * Real.pi / 180.0  -- converting 20 degrees to radians
  let angle1 := 5 * theta
  let angle2 := 6 * theta
  let angle3 := 7 * theta
  (1 / 2) * r * r * (Real.sin angle1 + Real.sin angle2 + Real.sin angle3)

theorem triangle_area_202_2192_pi_squared (a b c : ℝ) (h1 : a = 5) (h2 : b = 6) (h3 : c = 7) : 
  triangle_area a b c = 202.2192 / (Real.pi * Real.pi) := 
by {
  sorry
}

end NUMINAMATH_GPT_triangle_area_202_2192_pi_squared_l478_47870


namespace NUMINAMATH_GPT_area_of_combined_rectangle_l478_47809

theorem area_of_combined_rectangle
  (short_side : ℝ) (num_small_rectangles : ℕ) (total_area : ℝ)
  (h1 : num_small_rectangles = 4)
  (h2 : short_side = 7)
  (h3 : total_area = (3 * short_side + short_side) * (2 * short_side)) :
  total_area = 392 := by
  sorry

end NUMINAMATH_GPT_area_of_combined_rectangle_l478_47809


namespace NUMINAMATH_GPT_not_all_inequalities_true_l478_47808

theorem not_all_inequalities_true (a b c : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : 0 < b ∧ b < 1) (h₂ : 0 < c ∧ c < 1) :
  ¬(a * (1 - b) > 1 / 4 ∧ b * (1 - c) > 1 / 4 ∧ c * (1 - a) > 1 / 4) :=
  sorry

end NUMINAMATH_GPT_not_all_inequalities_true_l478_47808


namespace NUMINAMATH_GPT_cos_alpha_add_beta_over_2_l478_47849

variable (α β : ℝ)

-- Conditions
variables (h1 : 0 < α ∧ α < π / 2)
variables (h2 : -π / 2 < β ∧ β < 0)
variables (h3 : Real.cos (π / 4 + α) = 1 / 3)
variables (h4 : Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3)

-- Result
theorem cos_alpha_add_beta_over_2 :
  Real.cos (α + β / 2) = 5 * Real.sqrt 3 / 9 :=
sorry

end NUMINAMATH_GPT_cos_alpha_add_beta_over_2_l478_47849


namespace NUMINAMATH_GPT_rabbits_and_raccoons_l478_47806

variable (b_r t_r x : ℕ)

theorem rabbits_and_raccoons : 
  2 * b_r = x ∧ 3 * t_r = x ∧ b_r = t_r + 3 → x = 18 := 
by
  sorry

end NUMINAMATH_GPT_rabbits_and_raccoons_l478_47806


namespace NUMINAMATH_GPT_village_population_rate_l478_47842

theorem village_population_rate (r : ℕ) :
  let PX := 72000
  let PY := 42000
  let decrease_rate_X := 1200
  let years := 15
  let population_X_after_years := PX - decrease_rate_X * years
  let population_Y_after_years := PY + r * years
  population_X_after_years = population_Y_after_years → r = 800 :=
by
  sorry

end NUMINAMATH_GPT_village_population_rate_l478_47842


namespace NUMINAMATH_GPT_x_midpoint_of_MN_l478_47852

-- Definition: Given the parabola y^2 = 4x
def parabola (y x : ℝ) : Prop := y^2 = 4 * x

-- Definition: Point F is the focus of the parabola y^2 = 4x
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Definition: Points M and N are on the parabola
def on_parabola (M N : ℝ × ℝ) : Prop :=
  parabola M.2 M.1 ∧ parabola N.2 N.1

-- Definition: The sum of distances |MF| + |NF| = 6
def sum_of_distances (M N : ℝ × ℝ) (F : ℝ × ℝ) : Prop :=
  dist M F + dist N F = 6

-- Theorem: Prove that the x-coordinate of the midpoint of MN is 2
theorem x_midpoint_of_MN (M N : ℝ × ℝ) (F : ℝ × ℝ) 
  (hF : focus F) (hM_N : on_parabola M N) (hDist : sum_of_distances M N F) :
  (M.1 + N.1) / 2 = 2 :=
sorry

end NUMINAMATH_GPT_x_midpoint_of_MN_l478_47852


namespace NUMINAMATH_GPT_fraction_lost_down_sewer_l478_47885

-- Definitions of the conditions derived from the problem
def initial_marbles := 100
def street_loss_percent := 60 / 100
def sewer_loss := 40 - 20
def remaining_marbles_after_street := initial_marbles - (initial_marbles * street_loss_percent)
def marbles_left := 20

-- The theorem statement proving the fraction of remaining marbles lost down the sewer
theorem fraction_lost_down_sewer :
  (sewer_loss / remaining_marbles_after_street) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_lost_down_sewer_l478_47885


namespace NUMINAMATH_GPT_profit_distribution_l478_47878

theorem profit_distribution (investment_LiWei investment_WangGang profit total_investment : ℝ)
  (h1 : investment_LiWei = 16000)
  (h2 : investment_WangGang = 12000)
  (h3 : profit = 14000)
  (h4 : total_investment = investment_LiWei + investment_WangGang) :
  (profit * (investment_LiWei / total_investment) = 8000) ∧ 
  (profit * (investment_WangGang / total_investment) = 6000) :=
by
  sorry

end NUMINAMATH_GPT_profit_distribution_l478_47878


namespace NUMINAMATH_GPT_find_f1_find_fx_find_largest_m_l478_47816

noncomputable def f (a b c : ℝ) (x : ℝ) := a * x ^ 2 + b * x + c

axiom min_value_eq_zero (a b c : ℝ) : ∀ x : ℝ, f a b c x ≥ 0 ∨ f a b c x ≤ 0
axiom symmetry_condition (a b c : ℝ) : ∀ x : ℝ, f a b c (x - 1) = f a b c (-x - 1)
axiom inequality_condition (a b c : ℝ) : ∀ x : ℝ, 0 < x ∧ x < 5 → x ≤ f a b c x ∧ f a b c x ≤ 2 * |x - 1| + 1

theorem find_f1 (a b c : ℝ) : f a b c 1 = 1 := sorry

theorem find_fx (a b c : ℝ) : ∀ x : ℝ, f a b c x = (1 / 4) * (x + 1) ^ 2 := sorry

theorem find_largest_m (a b c : ℝ) : ∃ m : ℝ, m > 1 ∧ ∀ t x : ℝ, 1 ≤ x ∧ x ≤ m → f a b c (x + t) ≤ x := sorry

end NUMINAMATH_GPT_find_f1_find_fx_find_largest_m_l478_47816


namespace NUMINAMATH_GPT_books_bought_l478_47869

theorem books_bought (initial_books bought_books total_books : ℕ) 
    (h_initial : initial_books = 35)
    (h_total : total_books = 56) :
    bought_books = total_books - initial_books → bought_books = 21 := 
by
  sorry

end NUMINAMATH_GPT_books_bought_l478_47869


namespace NUMINAMATH_GPT_calculate_expression_l478_47874

theorem calculate_expression : (632^2 - 568^2 + 100) = 76900 :=
by sorry

end NUMINAMATH_GPT_calculate_expression_l478_47874


namespace NUMINAMATH_GPT_integer_pairs_satisfying_equation_l478_47820

theorem integer_pairs_satisfying_equation :
  {p : ℤ × ℤ | (p.1)^3 + (p.2)^3 - 3*(p.1)^2 + 6*(p.2)^2 + 3*(p.1) + 12*(p.2) + 6 = 0}
  = {(1, -1), (2, -2)} := 
sorry

end NUMINAMATH_GPT_integer_pairs_satisfying_equation_l478_47820


namespace NUMINAMATH_GPT_fixer_used_30_percent_kitchen_l478_47862

def fixer_percentage (x : ℝ) : Prop :=
  let initial_nails := 400
  let remaining_after_kitchen := initial_nails * ((100 - x) / 100)
  let remaining_after_fence := remaining_after_kitchen * 0.3
  remaining_after_fence = 84

theorem fixer_used_30_percent_kitchen : fixer_percentage 30 :=
by
  exact sorry

end NUMINAMATH_GPT_fixer_used_30_percent_kitchen_l478_47862


namespace NUMINAMATH_GPT_cloth_cost_price_l478_47864

theorem cloth_cost_price
  (meters_of_cloth : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ)
  (total_profit : ℕ) (total_cost_price : ℕ) (cost_price_per_meter : ℕ) :
  meters_of_cloth = 45 →
  selling_price = 4500 →
  profit_per_meter = 14 →
  total_profit = profit_per_meter * meters_of_cloth →
  total_cost_price = selling_price - total_profit →
  cost_price_per_meter = total_cost_price / meters_of_cloth →
  cost_price_per_meter = 86 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cloth_cost_price_l478_47864


namespace NUMINAMATH_GPT_jack_sugar_remaining_l478_47834

-- Define the initial amount of sugar and all daily transactions
def jack_initial_sugar : ℝ := 65
def jack_use_day1 : ℝ := 18.5
def alex_borrow_day1 : ℝ := 5.3
def jack_buy_day2 : ℝ := 30.2
def jack_use_day2 : ℝ := 12.7
def emma_give_day2 : ℝ := 4.75
def jack_buy_day3 : ℝ := 20.5
def jack_use_day3 : ℝ := 8.25
def alex_return_day3 : ℝ := 2.8
def alex_borrow_day3 : ℝ := 1.2
def jack_use_day4 : ℝ := 9.5
def olivia_give_day4 : ℝ := 6.35
def jack_use_day5 : ℝ := 10.75
def emma_borrow_day5 : ℝ := 3.1
def alex_return_day5 : ℝ := 3

-- Calculate the remaining sugar each day
def jack_sugar_day1 : ℝ := jack_initial_sugar - jack_use_day1 - alex_borrow_day1
def jack_sugar_day2 : ℝ := jack_sugar_day1 + jack_buy_day2 - jack_use_day2 + emma_give_day2
def jack_sugar_day3 : ℝ := jack_sugar_day2 + jack_buy_day3 - jack_use_day3 + alex_return_day3 - alex_borrow_day3
def jack_sugar_day4 : ℝ := jack_sugar_day3 - jack_use_day4 + olivia_give_day4
def jack_sugar_day5 : ℝ := jack_sugar_day4 - jack_use_day5 - emma_borrow_day5 + alex_return_day5

-- Final proof statement: Jack ends up with 63.3 pounds of sugar
theorem jack_sugar_remaining : jack_sugar_day5 = 63.3 := 
by sorry

end NUMINAMATH_GPT_jack_sugar_remaining_l478_47834


namespace NUMINAMATH_GPT_five_people_six_chairs_l478_47839

/-- Number of ways to sit 5 people in 6 chairs -/
def ways_to_sit_in_chairs : ℕ :=
  6 * 5 * 4 * 3 * 2

theorem five_people_six_chairs : ways_to_sit_in_chairs = 720 := by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_five_people_six_chairs_l478_47839


namespace NUMINAMATH_GPT_workbooks_needed_l478_47807

theorem workbooks_needed (classes : ℕ) (workbooks_per_class : ℕ) (spare_workbooks : ℕ) (total_workbooks : ℕ) :
  classes = 25 → workbooks_per_class = 144 → spare_workbooks = 80 → total_workbooks = 25 * 144 + 80 → 
  total_workbooks = classes * workbooks_per_class + spare_workbooks :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  exact h4

end NUMINAMATH_GPT_workbooks_needed_l478_47807


namespace NUMINAMATH_GPT_toothpicks_needed_for_8_step_staircase_l478_47811

theorem toothpicks_needed_for_8_step_staircase:
  ∀ n toothpicks : ℕ, n = 4 → toothpicks = 30 → 
  (∃ additional_toothpicks : ℕ, additional_toothpicks = 88) :=
by
  sorry

end NUMINAMATH_GPT_toothpicks_needed_for_8_step_staircase_l478_47811


namespace NUMINAMATH_GPT_find_x_l478_47877

open Real

def vector (a b : ℝ) : ℝ × ℝ := (a, b)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

def problem_statement (x : ℝ) : Prop :=
  let m := vector 2 x
  let n := vector 4 (-2)
  let m_minus_n := vector (2 - 4) (x - (-2))
  perpendicular m m_minus_n → x = -1 + sqrt 5 ∨ x = -1 - sqrt 5

-- We assert the theorem based on the problem statement
theorem find_x (x : ℝ) : problem_statement x :=
  sorry

end NUMINAMATH_GPT_find_x_l478_47877


namespace NUMINAMATH_GPT_train_cross_bridge_time_l478_47855

noncomputable def time_to_cross_bridge (L_train : ℕ) (v_kmph : ℕ) (L_bridge : ℕ) : ℝ :=
  let v_mps := (v_kmph * 1000) / 3600
  let total_distance := L_train + L_bridge
  total_distance / v_mps

theorem train_cross_bridge_time :
  time_to_cross_bridge 145 54 660 = 53.67 := by
    sorry

end NUMINAMATH_GPT_train_cross_bridge_time_l478_47855


namespace NUMINAMATH_GPT_ratio_black_bears_to_white_bears_l478_47838

theorem ratio_black_bears_to_white_bears
  (B W Br : ℕ)
  (hB : B = 60)
  (hBr : Br = B + 40)
  (h_total : B + W + Br = 190) :
  B / W = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_black_bears_to_white_bears_l478_47838


namespace NUMINAMATH_GPT_larger_number_is_437_l478_47881

-- Definitions from the conditions
def hcf : ℕ := 23
def factor1 : ℕ := 13
def factor2 : ℕ := 19

-- The larger number should be the product of H.C.F and the larger factor.
theorem larger_number_is_437 : hcf * factor2 = 437 := by
  sorry

end NUMINAMATH_GPT_larger_number_is_437_l478_47881


namespace NUMINAMATH_GPT_inequality_am_gm_l478_47801

theorem inequality_am_gm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 1/a + 1/b + 1/c ≥ a + b + c) : a + b + c ≥ 3 * a * b * c :=
sorry

end NUMINAMATH_GPT_inequality_am_gm_l478_47801


namespace NUMINAMATH_GPT_valid_values_for_D_l478_47822

-- Definitions for the distinct digits and the non-zero condition
def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9
def distinct_nonzero_digits (A B C D : ℕ) : Prop :=
  is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

-- Condition for the carry situation
def carry_in_addition (A B C D : ℕ) : Prop :=
  ∃ carry1 carry2 carry3 carry4 : ℕ,
  (A + B + carry1) % 10 = D ∧ (B + C + carry2) % 10 = A ∧
  (C + C + carry3) % 10 = B ∧ (A + B + carry4) % 10 = C ∧
  (carry1 = 1 ∨ carry2 = 1 ∨ carry3 = 1 ∨ carry4 = 1)

-- Main statement
theorem valid_values_for_D (A B C D : ℕ) :
  distinct_nonzero_digits A B C D →
  carry_in_addition A B C D →
  ∃ n, n = 5 :=
sorry

end NUMINAMATH_GPT_valid_values_for_D_l478_47822


namespace NUMINAMATH_GPT_inequality_range_l478_47873

theorem inequality_range (y : ℝ) (b : ℝ) (hb : 0 < b) : (|y-5| + 2 * |y-2| > b) ↔ (b < 3) := 
sorry

end NUMINAMATH_GPT_inequality_range_l478_47873


namespace NUMINAMATH_GPT_bug_travel_distance_half_l478_47895

-- Define the conditions
def isHexagonalGrid (side_length : ℝ) : Prop :=
  side_length = 1

def shortest_path_length (path_length : ℝ) : Prop :=
  path_length = 100

-- Define a theorem that encapsulates the problem statement
theorem bug_travel_distance_half (side_length path_length : ℝ)
  (H1 : isHexagonalGrid side_length)
  (H2 : shortest_path_length path_length) :
  ∃ one_direction_distance : ℝ, one_direction_distance = path_length / 2 :=
sorry -- Proof to be provided.

end NUMINAMATH_GPT_bug_travel_distance_half_l478_47895


namespace NUMINAMATH_GPT_hall_ratio_l478_47872

open Real

theorem hall_ratio (w l : ℝ) (h_area : w * l = 288) (h_diff : l - w = 12) : w / l = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_hall_ratio_l478_47872


namespace NUMINAMATH_GPT_max_element_sum_l478_47815

-- Definitions based on conditions
def S : Set ℚ :=
  {r | ∃ (p q : ℕ), r = p / q ∧ q ≤ 2009 ∧ p / q < 1257/2009}

-- Maximum element of S in reduced form
def max_element_S (r : ℚ) : Prop := r ∈ S ∧ ∀ s ∈ S, r ≥ s

-- Main statement to be proven
theorem max_element_sum : 
  ∃ p0 q0 : ℕ, max_element_S (p0 / q0) ∧ Nat.gcd p0 q0 = 1 ∧ p0 + q0 = 595 := 
sorry

end NUMINAMATH_GPT_max_element_sum_l478_47815


namespace NUMINAMATH_GPT_simplify_expression_l478_47832

theorem simplify_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) *
  (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l478_47832


namespace NUMINAMATH_GPT_cistern_filling_time_l478_47836

open Real

theorem cistern_filling_time :
  let rate1 := 1 / 10
  let rate2 := 1 / 12
  let rate3 := -1 / 25
  let rate4 := 1 / 15
  let rate5 := -1 / 30
  let combined_rate := rate1 + rate2 + rate4 + rate3 + rate5
  (300 / combined_rate) = (300 / 53) := by
  let rate1 := 1 / 10
  let rate2 := 1 / 12
  let rate3 := -1 / 25
  let rate4 := 1 / 15
  let rate5 := -1 / 30
  let combined_rate := rate1 + rate2 + rate4 + rate3 + rate5
  sorry

end NUMINAMATH_GPT_cistern_filling_time_l478_47836


namespace NUMINAMATH_GPT_determine_sum_l478_47861

theorem determine_sum (P R : ℝ) (h : 3 * P * (R + 1) / 100 - 3 * P * R / 100 = 78) : 
  P = 2600 :=
sorry

end NUMINAMATH_GPT_determine_sum_l478_47861


namespace NUMINAMATH_GPT_customers_in_each_car_l478_47867

def total_customers (sports_store_sales music_store_sales : ℕ) : ℕ :=
  sports_store_sales + music_store_sales

def customers_per_car (total_customers cars : ℕ) : ℕ :=
  total_customers / cars

theorem customers_in_each_car :
  let cars := 10
  let sports_store_sales := 20
  let music_store_sales := 30
  let total_customers := total_customers sports_store_sales music_store_sales
  total_customers / cars = 5 := by
  let cars := 10
  let sports_store_sales := 20
  let music_store_sales := 30
  let total_customers := total_customers sports_store_sales music_store_sales
  show total_customers / cars = 5
  sorry

end NUMINAMATH_GPT_customers_in_each_car_l478_47867


namespace NUMINAMATH_GPT_greatest_possible_median_l478_47835

theorem greatest_possible_median (k m r s t : ℕ) (h_avg : (k + m + r + s + t) / 5 = 10) (h_order : k < m ∧ m < r ∧ r < s ∧ s < t) (h_t : t = 20) : r = 8 :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_median_l478_47835


namespace NUMINAMATH_GPT_complement_set_l478_47814

def U : Set ℝ := Set.univ
def M : Set ℝ := {y | ∃ x : ℝ, 0 < x ∧ x < 1 ∧ y = Real.log x / Real.log 2}

theorem complement_set :
  Set.compl M = {y : ℝ | y ≥ 0} :=
by
  sorry

end NUMINAMATH_GPT_complement_set_l478_47814


namespace NUMINAMATH_GPT_overhead_percentage_l478_47826

def purchase_price : ℝ := 48
def markup : ℝ := 30
def net_profit : ℝ := 12

-- Define the theorem to be proved
theorem overhead_percentage : ((markup - net_profit) / purchase_price) * 100 = 37.5 := by
  sorry

end NUMINAMATH_GPT_overhead_percentage_l478_47826


namespace NUMINAMATH_GPT_solution_set_of_inequality_l478_47876

variable {a b x : ℝ}

theorem solution_set_of_inequality (h : ∃ y, y = 3*(-5) + a ∧ y = -2*(-5) + b) :
  (3*x + a < -2*x + b) ↔ (x < -5) :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l478_47876


namespace NUMINAMATH_GPT_optimal_discount_sequence_saves_more_l478_47803

theorem optimal_discount_sequence_saves_more :
  (let initial_price := 30
   let flat_discount := 5
   let percent_discount := 0.25
   let first_seq_price := ((initial_price - flat_discount) * (1 - percent_discount))
   let second_seq_price := ((initial_price * (1 - percent_discount)) - flat_discount)
   first_seq_price - second_seq_price = 1.25) :=
by
  sorry

end NUMINAMATH_GPT_optimal_discount_sequence_saves_more_l478_47803


namespace NUMINAMATH_GPT_increase_in_p_does_not_imply_increase_in_equal_points_probability_l478_47846

noncomputable def probability_equal_points (p : ℝ) : ℝ :=
  (3 * p^2 - 2 * p + 1) / 4

theorem increase_in_p_does_not_imply_increase_in_equal_points_probability :
  ¬ ∀ p1 p2 : ℝ, p1 < p2 → p1 ≥ 0 → p2 ≤ 1 → probability_equal_points p1 < probability_equal_points p2 := 
sorry

end NUMINAMATH_GPT_increase_in_p_does_not_imply_increase_in_equal_points_probability_l478_47846


namespace NUMINAMATH_GPT_fair_coin_heads_probability_l478_47887

theorem fair_coin_heads_probability
  (fair_coin : ∀ n : ℕ, (∀ (heads tails : ℕ), heads + tails = n → (heads / n = 1 / 2) ∧ (tails / n = 1 / 2)))
  (n : ℕ)
  (heads : ℕ)
  (tails : ℕ)
  (h1 : n = 20)
  (h2 : heads = 8)
  (h3 : tails = 12)
  (h4 : heads + tails = n)
  : heads / n = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fair_coin_heads_probability_l478_47887


namespace NUMINAMATH_GPT_parallelogram_base_length_l478_47817

theorem parallelogram_base_length (b h : ℝ) (area : ℝ) (angle : ℝ) (h_area : area = 200) 
(h_altitude : h = 2 * b) (h_angle : angle = 60) : b = 10 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_parallelogram_base_length_l478_47817


namespace NUMINAMATH_GPT_Randy_trip_distance_l478_47851

theorem Randy_trip_distance (x : ℝ) (h1 : x = 4 * (x / 4 + 30 + x / 6)) : x = 360 / 7 :=
by
  have h2 : x = ((3 * x + 36 * 30 + 2 * x) / 12) := sorry
  have h3 : x = (5 * x / 12 + 30) := sorry
  have h4 : 30 = x - (5 * x / 12) := sorry
  have h5 : 30 = 7 * x / 12 := sorry
  have h6 : x = (12 * 30) / 7 := sorry
  have h7 : x = 360 / 7 := sorry
  exact h7

end NUMINAMATH_GPT_Randy_trip_distance_l478_47851


namespace NUMINAMATH_GPT_shorter_tree_height_l478_47833

theorem shorter_tree_height
  (s : ℝ)
  (h₁ : ∀ s, s > 0 )
  (h₂ : s + (s + 20) = 240)
  (h₃ : s / (s + 20) = 5 / 7) :
  s = 110 :=
by
sorry

end NUMINAMATH_GPT_shorter_tree_height_l478_47833


namespace NUMINAMATH_GPT_total_toothpicks_grid_area_l478_47879

open Nat

-- Definitions
def grid_length : Nat := 30
def grid_width : Nat := 50

-- Prove the total number of toothpicks
theorem total_toothpicks : (31 * grid_width + 51 * grid_length) = 3080 := by
  sorry

-- Prove the area enclosed by the grid
theorem grid_area : (grid_length * grid_width) = 1500 := by
  sorry

end NUMINAMATH_GPT_total_toothpicks_grid_area_l478_47879


namespace NUMINAMATH_GPT_interest_rate_l478_47837

theorem interest_rate (SI P T : ℕ) (h1 : SI = 2000) (h2 : P = 5000) (h3 : T = 10) :
  (SI = (P * R * T) / 100) -> R = 4 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_l478_47837
