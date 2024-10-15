import Mathlib

namespace NUMINAMATH_GPT_solve_quadratic_roots_l313_31300

theorem solve_quadratic_roots (x : ℝ) : (x - 3) ^ 2 = 3 - x ↔ x = 3 ∨ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_roots_l313_31300


namespace NUMINAMATH_GPT_gasoline_needed_l313_31327

variable (distance_trip : ℕ) (fuel_per_trip_distance : ℕ) (trip_distance : ℕ) (fuel_needed : ℕ)

theorem gasoline_needed (h1 : distance_trip = 140)
                       (h2 : fuel_per_trip_distance = 10)
                       (h3 : trip_distance = 70)
                       (h4 : fuel_needed = 20) :
  (fuel_per_trip_distance * (distance_trip / trip_distance)) = fuel_needed :=
by sorry

end NUMINAMATH_GPT_gasoline_needed_l313_31327


namespace NUMINAMATH_GPT_find_number_l313_31387

-- Definitions based on conditions
def condition (x : ℝ) : Prop := (x - 5) / 3 = 4

-- The target theorem to prove
theorem find_number (x : ℝ) (h : condition x) : x = 17 :=
sorry

end NUMINAMATH_GPT_find_number_l313_31387


namespace NUMINAMATH_GPT_problem_statement_l313_31341

-- Define the repeating decimal and the required gcd condition
def repeating_decimal_value := (356 : ℚ) / 999
def gcd_condition (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the main theorem stating the required sum
theorem problem_statement (a b : ℕ) 
                          (h_a : a = 356) 
                          (h_b : b = 999) 
                          (h_gcd : gcd_condition a b) : 
    a + b = 1355 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l313_31341


namespace NUMINAMATH_GPT_solve_custom_eq_l313_31389

-- Define the custom operation a * b = ab + a + b, we will use ∗ instead of * to avoid confusion with multiplication

def custom_op (a b : Nat) : Nat := a * b + a + b

-- State the problem in Lean 4
theorem solve_custom_eq (x : Nat) : custom_op 3 x = 27 → x = 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_custom_eq_l313_31389


namespace NUMINAMATH_GPT_rate_of_interest_is_20_l313_31328

-- Definitions of the given conditions
def principal := 400
def simple_interest := 160
def time := 2

-- Definition of the rate of interest based on the given formula
def rate_of_interest (P SI T : ℕ) : ℕ := (SI * 100) / (P * T)

-- Theorem stating that the rate of interest is 20% given the conditions
theorem rate_of_interest_is_20 :
  rate_of_interest principal simple_interest time = 20 := by
  sorry

end NUMINAMATH_GPT_rate_of_interest_is_20_l313_31328


namespace NUMINAMATH_GPT_fixed_point_of_exponential_function_l313_31322

-- The function definition and conditions are given as hypotheses
theorem fixed_point_of_exponential_function
  (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  ∃ P : ℝ × ℝ, (∀ x : ℝ, (x = 1) → P = (x, a^(x-1) - 2)) → P = (1, -1) :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_of_exponential_function_l313_31322


namespace NUMINAMATH_GPT_four_digit_number_count_l313_31326

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end NUMINAMATH_GPT_four_digit_number_count_l313_31326


namespace NUMINAMATH_GPT_value_of_expression_l313_31337

theorem value_of_expression (m : ℝ) 
  (h : m^2 - 2 * m - 1 = 0) : 3 * m^2 - 6 * m + 2020 = 2023 := 
by 
  /- Proof is omitted -/
  sorry

end NUMINAMATH_GPT_value_of_expression_l313_31337


namespace NUMINAMATH_GPT_hydrated_aluminum_iodide_props_l313_31312

noncomputable def Al_mass : ℝ := 26.98
noncomputable def I_mass : ℝ := 126.90
noncomputable def H2O_mass : ℝ := 18.015
noncomputable def AlI3_mass (mass_AlI3: ℝ) : ℝ := 26.98 + 3 * 126.90

noncomputable def mass_percentage_iodine (mass_AlI3 mass_sample: ℝ) : ℝ :=
  (mass_AlI3 * (3 * I_mass / (Al_mass + 3 * I_mass)) / mass_sample) * 100

noncomputable def value_x (mass_H2O mass_AlI3: ℝ) : ℝ :=
  (mass_H2O / H2O_mass) / (mass_AlI3 / (Al_mass + 3 * I_mass))

theorem hydrated_aluminum_iodide_props (mass_AlI3 mass_H2O mass_sample: ℝ)
    (h_sample: mass_AlI3 + mass_H2O = mass_sample) :
    ∃ (percentage: ℝ) (x: ℝ), percentage = mass_percentage_iodine mass_AlI3 mass_sample ∧
                                      x = value_x mass_H2O mass_AlI3 :=
by
  sorry

end NUMINAMATH_GPT_hydrated_aluminum_iodide_props_l313_31312


namespace NUMINAMATH_GPT_suitableTempForPreservingBoth_l313_31386

-- Definitions for the temperature ranges of types A and B vegetables
def suitableTempRangeA := {t : ℝ | 3 ≤ t ∧ t ≤ 8}
def suitableTempRangeB := {t : ℝ | 5 ≤ t ∧ t ≤ 10}

-- The intersection of the suitable temperature ranges
def suitableTempRangeForBoth := {t : ℝ | 5 ≤ t ∧ t ≤ 8}

-- The theorem statement we need to prove
theorem suitableTempForPreservingBoth :
  suitableTempRangeForBoth = suitableTempRangeA ∩ suitableTempRangeB :=
sorry

end NUMINAMATH_GPT_suitableTempForPreservingBoth_l313_31386


namespace NUMINAMATH_GPT_f_neg1_l313_31344

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom symmetry_about_x2 : ∀ x : ℝ, f (2 + x) = f (2 - x)
axiom f3_value : f 3 = 3

theorem f_neg1 : f (-1) = 3 := by
  sorry

end NUMINAMATH_GPT_f_neg1_l313_31344


namespace NUMINAMATH_GPT_prove_smallest_solution_l313_31384

noncomputable def smallest_solution : ℝ :=
  if h : 0 ≤ (3 - Real.sqrt 17) / 2 then min ((3 - Real.sqrt 17) / 2) 1
  else (3 - Real.sqrt 17) / 2  -- Assumption as sqrt(17) > 3, so (3 - sqrt(17))/2 < 0

theorem prove_smallest_solution :
  ∃ x : ℝ, (x * |x| = 3 * x - 2) ∧ 
           (∀ y : ℝ, (y * |y| = 3 * y - 2) → x ≤ y) ∧
           x = (3 - Real.sqrt 17) / 2 :=
sorry

end NUMINAMATH_GPT_prove_smallest_solution_l313_31384


namespace NUMINAMATH_GPT_emily_pen_selections_is_3150_l313_31376

open Function

noncomputable def emily_pen_selections : ℕ :=
  (Nat.choose 10 4) * (Nat.choose 6 2)

theorem emily_pen_selections_is_3150 : emily_pen_selections = 3150 :=
by
  sorry

end NUMINAMATH_GPT_emily_pen_selections_is_3150_l313_31376


namespace NUMINAMATH_GPT_trees_died_l313_31302

theorem trees_died (initial_trees dead surviving : ℕ) 
  (h_initial : initial_trees = 11) 
  (h_surviving : surviving = dead + 7) 
  (h_total : dead + surviving = initial_trees) : 
  dead = 2 :=
by
  sorry

end NUMINAMATH_GPT_trees_died_l313_31302


namespace NUMINAMATH_GPT_age_difference_l313_31382

variable (a b c : ℕ)

theorem age_difference (h : a + b = b + c + 13) : a - c = 13 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l313_31382


namespace NUMINAMATH_GPT_cannot_be_square_difference_l313_31392

def square_difference_formula (a b : ℝ) : ℝ := a^2 - b^2

def expression_A (x : ℝ) : ℝ := (x + 1) * (x - 1)
def expression_B (x : ℝ) : ℝ := (-x + 1) * (-x - 1)
def expression_C (x : ℝ) : ℝ := (x + 1) * (-x + 1)
def expression_D (x : ℝ) : ℝ := (x + 1) * (1 + x)

theorem cannot_be_square_difference (x : ℝ) : 
  ¬ (∃ a b, (x + 1) * (1 + x) = square_difference_formula a b) := 
sorry

end NUMINAMATH_GPT_cannot_be_square_difference_l313_31392


namespace NUMINAMATH_GPT_time_for_one_mile_l313_31363

theorem time_for_one_mile (d v : ℝ) (mile_in_feet : ℝ) (num_circles : ℕ) 
  (circle_circumference : ℝ) (distance_in_miles : ℝ) (time : ℝ) :
  d = 50 ∧ v = 10 ∧ mile_in_feet = 5280 ∧ num_circles = 106 ∧ 
  circle_circumference = 50 * Real.pi ∧ 
  distance_in_miles = (106 * 50 * Real.pi) / 5280 ∧ 
  time = distance_in_miles / v →
  time = Real.pi / 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_time_for_one_mile_l313_31363


namespace NUMINAMATH_GPT_vector_dot_product_l313_31380

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 2)

-- Define the operation to calculate (a + 2b)
def two_b : ℝ × ℝ := (2 * b.1, 2 * b.2)
def a_plus_2b : ℝ × ℝ := (a.1 + two_b.1, a.2 + two_b.2)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- State the theorem
theorem vector_dot_product : dot_product a_plus_2b b = 14 := by
  sorry

end NUMINAMATH_GPT_vector_dot_product_l313_31380


namespace NUMINAMATH_GPT_regression_decrease_by_three_l313_31356

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 2 - 3 * x

-- Prove that when the explanatory variable increases by 1 unit, the predicted variable decreases by 3 units
theorem regression_decrease_by_three : ∀ x : ℝ, regression_equation (x + 1) = regression_equation x - 3 :=
by
  intro x
  unfold regression_equation
  sorry

end NUMINAMATH_GPT_regression_decrease_by_three_l313_31356


namespace NUMINAMATH_GPT_sum_of_x_and_y_l313_31307

theorem sum_of_x_and_y (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
    (hx : ∃ (a : ℕ), 720 * x = a^2)
    (hy : ∃ (b : ℕ), 720 * y = b^4) :
    x + y = 1130 :=
sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l313_31307


namespace NUMINAMATH_GPT_zachary_needs_more_money_l313_31343

def cost_in_usd_football (euro_to_usd : ℝ) (football_cost_eur : ℝ) : ℝ :=
  football_cost_eur * euro_to_usd

def cost_in_usd_shorts (gbp_to_usd : ℝ) (shorts_cost_gbp : ℝ) (pairs : ℕ) : ℝ :=
  shorts_cost_gbp * pairs * gbp_to_usd

def cost_in_usd_shoes (shoes_cost_usd : ℝ) : ℝ :=
  shoes_cost_usd

def cost_in_usd_socks (jpy_to_usd : ℝ) (socks_cost_jpy : ℝ) (pairs : ℕ) : ℝ :=
  socks_cost_jpy * pairs * jpy_to_usd

def cost_in_usd_water_bottle (krw_to_usd : ℝ) (water_bottle_cost_krw : ℝ) : ℝ :=
  water_bottle_cost_krw * krw_to_usd

def total_cost_before_discount (cost_football_usd cost_shorts_usd cost_shoes_usd
                                cost_socks_usd cost_water_bottle_usd : ℝ) : ℝ :=
  cost_football_usd + cost_shorts_usd + cost_shoes_usd + cost_socks_usd + cost_water_bottle_usd

def discounted_total_cost (total_cost : ℝ) (discount : ℝ) : ℝ :=
  total_cost * (1 - discount)

def additional_money_needed (discounted_total_cost current_money : ℝ) : ℝ :=
  discounted_total_cost - current_money

theorem zachary_needs_more_money (euro_to_usd : ℝ) (gbp_to_usd : ℝ) (jpy_to_usd : ℝ) (krw_to_usd : ℝ)
  (football_cost_eur : ℝ) (shorts_cost_gbp : ℝ) (pairs_shorts : ℕ) (shoes_cost_usd : ℝ)
  (socks_cost_jpy : ℝ) (pairs_socks : ℕ) (water_bottle_cost_krw : ℝ) (current_money_usd : ℝ)
  (discount : ℝ) : additional_money_needed 
      (discounted_total_cost
          (total_cost_before_discount
            (cost_in_usd_football euro_to_usd football_cost_eur)
            (cost_in_usd_shorts gbp_to_usd shorts_cost_gbp pairs_shorts)
            (cost_in_usd_shoes shoes_cost_usd)
            (cost_in_usd_socks jpy_to_usd socks_cost_jpy pairs_socks)
            (cost_in_usd_water_bottle krw_to_usd water_bottle_cost_krw)) 
          discount) 
      current_money_usd = 7.127214 := 
sorry

end NUMINAMATH_GPT_zachary_needs_more_money_l313_31343


namespace NUMINAMATH_GPT_arithmetic_proof_l313_31306

theorem arithmetic_proof : (28 + 48 / 69) * 69 = 1980 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_proof_l313_31306


namespace NUMINAMATH_GPT_second_child_birth_year_l313_31365

theorem second_child_birth_year (first_child_birth : ℕ)
  (second_child_birth : ℕ)
  (third_child_birth : ℕ)
  (fourth_child_birth : ℕ)
  (first_child_years_ago : first_child_birth = 15)
  (third_child_on_second_child_fourth_birthday : third_child_birth = second_child_birth + 4)
  (fourth_child_two_years_after_third : fourth_child_birth = third_child_birth + 2)
  (fourth_child_age : fourth_child_birth = 8) :
  second_child_birth = first_child_birth - 14 := 
by
  sorry

end NUMINAMATH_GPT_second_child_birth_year_l313_31365


namespace NUMINAMATH_GPT_acute_angled_triangle_range_l313_31372

theorem acute_angled_triangle_range (x : ℝ) (h : (x^2 + 6)^2 < (x^2 + 4)^2 + (4 * x)^2) : x > (Real.sqrt 15) / 3 := sorry

end NUMINAMATH_GPT_acute_angled_triangle_range_l313_31372


namespace NUMINAMATH_GPT_cos_75_eq_sqrt6_sub_sqrt2_div_4_l313_31325

theorem cos_75_eq_sqrt6_sub_sqrt2_div_4 :
  Real.cos (75 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := sorry

end NUMINAMATH_GPT_cos_75_eq_sqrt6_sub_sqrt2_div_4_l313_31325


namespace NUMINAMATH_GPT_integer_solutions_count_l313_31364

theorem integer_solutions_count : 
  ∃ (s : Finset ℤ), 
    (∀ x ∈ s, 2 * x + 1 > -3 ∧ -x + 3 ≥ 0) ∧ 
    s.card = 5 := 
by 
  sorry

end NUMINAMATH_GPT_integer_solutions_count_l313_31364


namespace NUMINAMATH_GPT_period_cosine_l313_31314

noncomputable def period_of_cosine_function : ℝ := 2 * Real.pi / 3

theorem period_cosine (x : ℝ) : ∃ T, ∀ x, Real.cos (3 * x - Real.pi) = Real.cos (3 * (x + T) - Real.pi) :=
  ⟨period_of_cosine_function, by sorry⟩

end NUMINAMATH_GPT_period_cosine_l313_31314


namespace NUMINAMATH_GPT_can_lid_boxes_count_l313_31396

theorem can_lid_boxes_count 
  (x y : ℕ) 
  (h1 : 3 * x + y + 14 = 75) : 
  x = 20 ∧ y = 1 :=
by 
  sorry

end NUMINAMATH_GPT_can_lid_boxes_count_l313_31396


namespace NUMINAMATH_GPT_congruent_semicircles_ratio_l313_31304

theorem congruent_semicircles_ratio (N : ℕ) (r : ℝ) (hN : N > 0) 
    (A : ℝ) (B : ℝ) (hA : A = (N * π * r^2) / 2)
    (hB : B = (π * N^2 * r^2) / 2 - (N * π * r^2) / 2)
    (h_ratio : A / B = 1 / 9) : 
    N = 10 :=
by
  -- The proof will be filled in here.
  sorry

end NUMINAMATH_GPT_congruent_semicircles_ratio_l313_31304


namespace NUMINAMATH_GPT_solution_to_system_of_equations_l313_31369

def augmented_matrix_system_solution (x y : ℝ) : Prop :=
  (x + 3 * y = 5) ∧ (2 * x + 4 * y = 6)

theorem solution_to_system_of_equations :
  ∃! (x y : ℝ), augmented_matrix_system_solution x y ∧ x = -1 ∧ y = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_to_system_of_equations_l313_31369


namespace NUMINAMATH_GPT_total_students_l313_31381

def Varsity_students : ℕ := 1300
def Northwest_students : ℕ := 1400
def Central_students : ℕ := 1800
def Greenbriar_students : ℕ := 1650

theorem total_students : Varsity_students + Northwest_students + Central_students + Greenbriar_students = 6150 :=
by
  -- Proof is omitted, so we use sorry.
  sorry

end NUMINAMATH_GPT_total_students_l313_31381


namespace NUMINAMATH_GPT_buckets_needed_l313_31350

variable {C : ℝ} (hC : C > 0)

theorem buckets_needed (h : 42 * C = 42 * C) : 
  (42 * C) / ((2 / 5) * C) = 105 :=
by
  sorry

end NUMINAMATH_GPT_buckets_needed_l313_31350


namespace NUMINAMATH_GPT_dime_quarter_problem_l313_31383

theorem dime_quarter_problem :
  15 * 25 + 10 * 10 = 25 * 25 + 35 * 10 :=
by
  sorry

end NUMINAMATH_GPT_dime_quarter_problem_l313_31383


namespace NUMINAMATH_GPT_quadratic_real_solutions_l313_31346

theorem quadratic_real_solutions (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) ↔ (m ≤ 7 ∧ m ≠ 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_solutions_l313_31346


namespace NUMINAMATH_GPT_james_balloons_l313_31385

-- Definitions
def amy_balloons : ℕ := 513
def extra_balloons_james_has : ℕ := 709

-- Statement of the problem
theorem james_balloons : amy_balloons + extra_balloons_james_has = 1222 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_james_balloons_l313_31385


namespace NUMINAMATH_GPT_solve_real_eq_l313_31301

theorem solve_real_eq (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ -2) :
  (x = (23 + Real.sqrt 145) / 6 ∨ x = (23 - Real.sqrt 145) / 6) ↔
  ((x ^ 3 - 3 * x ^ 2) / (x ^ 2 - 4) + 2 * x = -16) :=
by sorry

end NUMINAMATH_GPT_solve_real_eq_l313_31301


namespace NUMINAMATH_GPT_angle_is_30_degrees_l313_31320

theorem angle_is_30_degrees (A : ℝ) (h_acute : A > 0 ∧ A < π / 2) (h_sin : Real.sin A = 1/2) : A = π / 6 := 
by 
  sorry

end NUMINAMATH_GPT_angle_is_30_degrees_l313_31320


namespace NUMINAMATH_GPT_find_four_digit_number_l313_31371

theorem find_four_digit_number : ∃ N : ℕ, 999 < N ∧ N < 10000 ∧ (∃ a : ℕ, a^2 = N) ∧ 
  (∃ b : ℕ, b^3 = N % 1000) ∧ (∃ c : ℕ, c^4 = N % 100) ∧ N = 9216 := 
by
  sorry

end NUMINAMATH_GPT_find_four_digit_number_l313_31371


namespace NUMINAMATH_GPT_sweet_cookies_more_than_salty_l313_31353

-- Definitions for the given conditions
def sweet_cookies_ate : Nat := 32
def salty_cookies_ate : Nat := 23

-- The statement to prove
theorem sweet_cookies_more_than_salty :
  sweet_cookies_ate - salty_cookies_ate = 9 := by
  sorry

end NUMINAMATH_GPT_sweet_cookies_more_than_salty_l313_31353


namespace NUMINAMATH_GPT_identity_of_polynomials_l313_31335

theorem identity_of_polynomials (a b : ℝ) : 
  (2 * x + a)^3 = 
  5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x 
  → a = -1 ∧ b = 1 := 
by 
  sorry

end NUMINAMATH_GPT_identity_of_polynomials_l313_31335


namespace NUMINAMATH_GPT_find_integer_m_l313_31338

theorem find_integer_m 
  (m : ℤ)
  (h1 : 30 ≤ m ∧ m ≤ 80)
  (h2 : ∃ k : ℤ, m = 6 * k)
  (h3 : m % 8 = 2)
  (h4 : m % 5 = 2) : 
  m = 42 := 
sorry

end NUMINAMATH_GPT_find_integer_m_l313_31338


namespace NUMINAMATH_GPT_triangle_height_relationship_l313_31333

theorem triangle_height_relationship
  (b : ℝ) (h1 h2 h3 : ℝ)
  (area1 area2 area3 : ℝ)
  (h_equal_angle : area1 / area2 = 16 / 25)
  (h_diff_angle : area1 / area3 = 4 / 9) :
  4 * h2 = 5 * h1 ∧ 6 * h2 = 5 * h3 := by
    sorry

end NUMINAMATH_GPT_triangle_height_relationship_l313_31333


namespace NUMINAMATH_GPT_probability_one_die_shows_4_given_sum_7_l313_31330

def outcomes_with_sum_7 : List (ℕ × ℕ) := [(1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1)]

def outcome_has_4 (outcome : ℕ × ℕ) : Bool :=
  outcome.fst = 4 ∨ outcome.snd = 4

def favorable_outcomes : List (ℕ × ℕ) :=
  outcomes_with_sum_7.filter outcome_has_4

theorem probability_one_die_shows_4_given_sum_7 :
  (favorable_outcomes.length : ℚ) / (outcomes_with_sum_7.length : ℚ) = 1 / 3 := sorry

end NUMINAMATH_GPT_probability_one_die_shows_4_given_sum_7_l313_31330


namespace NUMINAMATH_GPT_continuous_stripe_probability_l313_31318

-- Define a structure representing the configuration of each face.
structure FaceConfiguration where
  is_diagonal : Bool
  edge_pair_or_vertex_pair : Bool

-- Define the cube configuration.
structure CubeConfiguration where
  face1 : FaceConfiguration
  face2 : FaceConfiguration
  face3 : FaceConfiguration
  face4 : FaceConfiguration
  face5 : FaceConfiguration
  face6 : FaceConfiguration

noncomputable def total_configurations : ℕ := 4^6

-- Define the function that checks if a configuration results in a continuous stripe.
def results_in_continuous_stripe (c : CubeConfiguration) : Bool := sorry

-- Define the number of configurations resulting in a continuous stripe.
noncomputable def configurations_with_continuous_stripe : ℕ :=
  Nat.card {c : CubeConfiguration // results_in_continuous_stripe c}

-- Define the probability calculation.
noncomputable def probability_continuous_stripe : ℚ :=
  configurations_with_continuous_stripe / total_configurations

-- The statement of the problem: Prove the probability of continuous stripe is 3/256.
theorem continuous_stripe_probability :
  probability_continuous_stripe = 3 / 256 :=
sorry

end NUMINAMATH_GPT_continuous_stripe_probability_l313_31318


namespace NUMINAMATH_GPT_sum_of_fifth_powers_divisibility_l313_31324

theorem sum_of_fifth_powers_divisibility (a b c d e : ℤ) :
  (a^5 + b^5 + c^5 + d^5 + e^5) % 25 = 0 → (a % 5 = 0) ∨ (b % 5 = 0) ∨ (c % 5 = 0) ∨ (d % 5 = 0) ∨ (e % 5 = 0) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fifth_powers_divisibility_l313_31324


namespace NUMINAMATH_GPT_no_real_solutions_iff_k_gt_4_l313_31354

theorem no_real_solutions_iff_k_gt_4 (k : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x + k ≠ 0) ↔ k > 4 :=
sorry

end NUMINAMATH_GPT_no_real_solutions_iff_k_gt_4_l313_31354


namespace NUMINAMATH_GPT_Tanika_total_boxes_sold_l313_31329

theorem Tanika_total_boxes_sold:
  let friday_boxes := 60
  let saturday_boxes := friday_boxes + 0.5 * friday_boxes
  let sunday_boxes := saturday_boxes - 0.3 * saturday_boxes
  friday_boxes + saturday_boxes + sunday_boxes = 213 :=
by
  sorry

end NUMINAMATH_GPT_Tanika_total_boxes_sold_l313_31329


namespace NUMINAMATH_GPT_find_number_l313_31379

theorem find_number (x : ℝ) : 
  (72 = 0.70 * x + 30) -> x = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l313_31379


namespace NUMINAMATH_GPT_marguerites_fraction_l313_31357

variable (x r b s : ℕ)

theorem marguerites_fraction
  (h1 : r = 5 * (x - r))
  (h2 : b = (x - b) / 5)
  (h3 : r + b + s = x) : s = 0 := by sorry

end NUMINAMATH_GPT_marguerites_fraction_l313_31357


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_value_l313_31315

variable (a : ℕ → ℝ)
variable (a_2 a_5 a_8 : ℝ)
variable (h1 : a 2 + a 8 = 15 - a 5)

/-- In an arithmetic sequence {a_n}, given that a_2 + a_8 = 15 - a_5, prove that a_5 equals 5. -/ 
theorem arithmetic_sequence_a5_value (h1 : a 2 + a 8 = 15 - a 5) : a 5 = 5 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_value_l313_31315


namespace NUMINAMATH_GPT_problem_l313_31377

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 3}
def complement_U (S : Set ℕ) : Set ℕ := U \ S

theorem problem : ((complement_U M) ∩ (complement_U N)) = {5} :=
by
  sorry

end NUMINAMATH_GPT_problem_l313_31377


namespace NUMINAMATH_GPT_range_of_a_l313_31367

   noncomputable section

   variable {f : ℝ → ℝ}

   /-- The requried theorem based on the given conditions and the correct answer -/
   theorem range_of_a (even_f : ∀ x, f (-x) = f x)
                      (increasing_f : ∀ x y, x ≤ y → y ≤ 0 → f x ≤ f y)
                      (h : f a ≤ f 2) : a ≤ -2 ∨ a ≥ 2 :=
   sorry
   
end NUMINAMATH_GPT_range_of_a_l313_31367


namespace NUMINAMATH_GPT_route_Y_quicker_than_route_X_l313_31391

theorem route_Y_quicker_than_route_X :
    let dist_X := 9  -- distance of Route X in miles
    let speed_X := 45  -- speed of Route X in miles per hour
    let dist_Y := 8  -- total distance of Route Y in miles
    let normal_dist_Y := 6.5  -- normal speed distance of Route Y in miles
    let construction_dist_Y := 1.5  -- construction zone distance of Route Y in miles
    let normal_speed_Y := 50  -- normal speed of Route Y in miles per hour
    let construction_speed_Y := 25  -- construction zone speed of Route Y in miles per hour
    let time_X := (dist_X / speed_X) * 60  -- time for Route X in minutes
    let time_Y1 := (normal_dist_Y / normal_speed_Y) * 60  -- time for normal speed segment of Route Y in minutes
    let time_Y2 := (construction_dist_Y / construction_speed_Y) * 60  -- time for construction zone segment of Route Y in minutes
    let time_Y := time_Y1 + time_Y2  -- total time for Route Y in minutes
    time_X - time_Y = 0.6 :=  -- the difference in time between Route X and Route Y in minutes
by
  sorry

end NUMINAMATH_GPT_route_Y_quicker_than_route_X_l313_31391


namespace NUMINAMATH_GPT_fertilizer_needed_l313_31360

def p_flats := 4
def p_per_flat := 8
def p_ounces := 8

def r_flats := 3
def r_per_flat := 6
def r_ounces := 3

def s_flats := 5
def s_per_flat := 10
def s_ounces := 6

def o_flats := 2
def o_per_flat := 4
def o_ounces := 4

def vf_quantity := 2
def vf_ounces := 2

def total_fertilizer : ℕ := 
  p_flats * p_per_flat * p_ounces +
  r_flats * r_per_flat * r_ounces +
  s_flats * s_per_flat * s_ounces +
  o_flats * o_per_flat * o_ounces +
  vf_quantity * vf_ounces

theorem fertilizer_needed : total_fertilizer = 646 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_fertilizer_needed_l313_31360


namespace NUMINAMATH_GPT_smaller_of_x_and_y_l313_31361

theorem smaller_of_x_and_y 
  (x y a b c d : ℝ) 
  (h1 : 0 < a) 
  (h2 : a < b + 1) 
  (h3 : x + y = c) 
  (h4 : x - y = d) 
  (h5 : x / y = a / (b + 1)) :
  min x y = (ac/(a + b + 1)) := 
by
  sorry

end NUMINAMATH_GPT_smaller_of_x_and_y_l313_31361


namespace NUMINAMATH_GPT_compute_star_difference_l313_31347

def star (x y : ℤ) : ℤ := x^2 * y - 3 * x

theorem compute_star_difference : (star 6 3) - (star 3 6) = 45 := by
  sorry

end NUMINAMATH_GPT_compute_star_difference_l313_31347


namespace NUMINAMATH_GPT_exists_N_minimal_l313_31349

-- Assuming m and n are positive and coprime
variables (m n : ℕ)
variables (h_pos_m : 0 < m) (h_pos_n : 0 < n)
variables (h_coprime : Nat.gcd m n = 1)

-- Statement of the mathematical problem
theorem exists_N_minimal :
  ∃ N : ℕ, (∀ k : ℕ, k ≥ N → ∃ a b : ℕ, k = a * m + b * n) ∧
           (N = m * n - m - n + 1) := 
  sorry

end NUMINAMATH_GPT_exists_N_minimal_l313_31349


namespace NUMINAMATH_GPT_determinant_eq_sum_of_products_l313_31321

theorem determinant_eq_sum_of_products (x y z : ℝ) :
  Matrix.det (Matrix.of ![![1, x + z, y], ![1, x + y + z, y + z], ![1, x + z, x + y + z]]) = x * y + y * z + z * x :=
by
  sorry

end NUMINAMATH_GPT_determinant_eq_sum_of_products_l313_31321


namespace NUMINAMATH_GPT_exists_two_numbers_l313_31308

theorem exists_two_numbers (x : Fin 7 → ℝ) :
  ∃ i j, 0 ≤ (x i - x j) / (1 + x i * x j) ∧ (x i - x j) / (1 + x i * x j) ≤ 1 / Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_exists_two_numbers_l313_31308


namespace NUMINAMATH_GPT_range_of_a_fixed_point_l313_31317

open Function

def f (x a : ℝ) := x^3 - a * x

theorem range_of_a (a : ℝ) (h1 : 0 < a) : 0 < a ∧ a ≤ 3 ↔ ∀ x ≥ 1, 3 * x^2 - a > 0 :=
sorry

theorem fixed_point (a x0 : ℝ) (h_a : 0 < a) (h_b : a ≤ 3)
  (h1 : x0 ≥ 1) (h2 : f x0 a ≥ 1) (h3 : f (f x0 a) a = x0) (strict_incr : ∀ x y, x ≥ 1 → y ≥ 1 → x < y → f x a < f y a) :
  f x0 a = x0 :=
sorry

end NUMINAMATH_GPT_range_of_a_fixed_point_l313_31317


namespace NUMINAMATH_GPT_parabola_vertex_l313_31348

theorem parabola_vertex:
  ∃ x y: ℝ, y^2 + 8 * y + 2 * x + 1 = 0 ∧ (x, y) = (7.5, -4) := sorry

end NUMINAMATH_GPT_parabola_vertex_l313_31348


namespace NUMINAMATH_GPT_total_height_of_sandcastles_l313_31303

structure Sandcastle :=
  (feet : Nat)
  (fraction_num : Nat)
  (fraction_den : Nat)

def janet : Sandcastle := ⟨3, 5, 6⟩
def sister : Sandcastle := ⟨2, 7, 12⟩
def tom : Sandcastle := ⟨1, 11, 20⟩
def lucy : Sandcastle := ⟨2, 13, 24⟩

-- a function to convert a Sandcastle to a common denominator
def convert_to_common_denominator (s : Sandcastle) : Sandcastle :=
  let common_den := 120 -- LCM of 6, 12, 20, 24
  ⟨s.feet, (s.fraction_num * (common_den / s.fraction_den)), common_den⟩

-- Definition of heights after conversion to common denominator
def janet_converted : Sandcastle := convert_to_common_denominator janet
def sister_converted : Sandcastle := convert_to_common_denominator sister
def tom_converted : Sandcastle := convert_to_common_denominator tom
def lucy_converted : Sandcastle := convert_to_common_denominator lucy

-- Proof problem
def total_height_proof_statement : Sandcastle :=
  let total_feet := janet.feet + sister.feet + tom.feet + lucy.feet
  let total_numerator := janet_converted.fraction_num + sister_converted.fraction_num + tom_converted.fraction_num + lucy_converted.fraction_num
  let total_denominator := 120
  ⟨total_feet + (total_numerator / total_denominator), total_numerator % total_denominator, total_denominator⟩

theorem total_height_of_sandcastles :
  total_height_proof_statement = ⟨10, 61, 120⟩ :=
by
  sorry

end NUMINAMATH_GPT_total_height_of_sandcastles_l313_31303


namespace NUMINAMATH_GPT_sum_of_squares_and_cubes_l313_31362

theorem sum_of_squares_and_cubes (a b : ℤ) (h : ∃ k : ℤ, a^2 - 4*b = k^2) :
  ∃ x1 x2 : ℤ, a^2 - 2*b = x1^2 + x2^2 ∧ 3*a*b - a^3 = x1^3 + x2^3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_and_cubes_l313_31362


namespace NUMINAMATH_GPT_tan_2theta_l313_31388

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + Real.cos x

theorem tan_2theta (θ : ℝ) (h : ∀ x, f x ≤ f θ) : Real.tan (2 * θ) = -4 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_tan_2theta_l313_31388


namespace NUMINAMATH_GPT_find_nabla_l313_31368

theorem find_nabla : ∀ (nabla : ℤ), 5 * (-4) = nabla + 2 → nabla = -22 :=
by
  intros nabla h
  sorry

end NUMINAMATH_GPT_find_nabla_l313_31368


namespace NUMINAMATH_GPT_math_problem_proof_l313_31395

-- Define the fractions involved
def frac1 : ℚ := -49
def frac2 : ℚ := 4 / 7
def frac3 : ℚ := -8 / 7

-- The original expression
def original_expr : ℚ :=
  frac1 * frac2 - frac2 / frac3

-- Declare the theorem to be proved
theorem math_problem_proof : original_expr = -27.5 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_proof_l313_31395


namespace NUMINAMATH_GPT_negation_of_universal_l313_31342

theorem negation_of_universal:
  ¬(∀ x : ℕ, x^2 > 1) ↔ ∃ x : ℕ, x^2 ≤ 1 :=
by sorry

end NUMINAMATH_GPT_negation_of_universal_l313_31342


namespace NUMINAMATH_GPT_factorization_identity_l313_31340

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorization_identity_l313_31340


namespace NUMINAMATH_GPT_math_proof_l313_31352

noncomputable def f (ω x : ℝ) := Real.sin (ω * x) - Real.sqrt 3 * Real.cos (ω * x)

theorem math_proof (h1 : ∀ x, f ω x = f ω (x + π)) (h2 : 0 < ω) :
  (ω = 2) ∧ (f 2 (-5 * Real.pi / 6) = 0) ∧ ¬∀ x : ℝ, x ∈ Set.Ioo (Real.pi / 3) (11 * Real.pi / 12) → 
  (∃ x₁ x₂ : ℝ, f 2 x₁ < f 2 x₂) ∧ (∀ x : ℝ, f 2 (x - Real.pi / 3) ≠ Real.cos (2 * x - Real.pi / 6)) := 
by
  sorry

end NUMINAMATH_GPT_math_proof_l313_31352


namespace NUMINAMATH_GPT_fraction_of_surface_area_is_red_l313_31374

structure Cube :=
  (edge_length : ℕ)
  (small_cubes : ℕ)
  (num_red_cubes : ℕ)
  (num_blue_cubes : ℕ)
  (blue_cube_edge_length : ℕ)
  (red_outer_layer : ℕ)

def surface_area (c : Cube) : ℕ := 6 * (c.edge_length * c.edge_length)

theorem fraction_of_surface_area_is_red (c : Cube) 
  (h_edge_length : c.edge_length = 4)
  (h_small_cubes : c.small_cubes = 64)
  (h_num_red_cubes : c.num_red_cubes = 40)
  (h_num_blue_cubes : c.num_blue_cubes = 24)
  (h_blue_cube_edge_length : c.blue_cube_edge_length = 2)
  (h_red_outer_layer : c.red_outer_layer = 1)
  : (surface_area c) / (surface_area c) = 1 := 
by
  sorry

end NUMINAMATH_GPT_fraction_of_surface_area_is_red_l313_31374


namespace NUMINAMATH_GPT_sam_total_pennies_l313_31334

def a : ℕ := 98
def b : ℕ := 93

theorem sam_total_pennies : a + b = 191 :=
by
  sorry

end NUMINAMATH_GPT_sam_total_pennies_l313_31334


namespace NUMINAMATH_GPT_smaller_circle_radius_l313_31313

theorem smaller_circle_radius (R : ℝ) (r : ℝ) (h1 : R = 10) (h2 : R = (2 * r) / Real.sqrt 3) : r = 5 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_smaller_circle_radius_l313_31313


namespace NUMINAMATH_GPT_probability_at_least_one_tree_survives_l313_31332

noncomputable def prob_at_least_one_survives (survival_rate_A survival_rate_B : ℚ) (n_A n_B : ℕ) : ℚ :=
  1 - ((1 - survival_rate_A)^(n_A) * (1 - survival_rate_B)^(n_B))

theorem probability_at_least_one_tree_survives :
  prob_at_least_one_survives (5/6) (4/5) 2 2 = 899 / 900 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_tree_survives_l313_31332


namespace NUMINAMATH_GPT_gcd_A_B_l313_31399

theorem gcd_A_B (a b : ℕ) (h1 : Nat.gcd a b = 1) (h2 : a > 0) (h3 : b > 0) : 
  Nat.gcd (8 * a + 3 * b) (3 * a + 2 * b) ≠ 1 → Nat.gcd (8 * a + 3 * b) (3 * a + 2 * b) = 7 :=
by
  sorry

end NUMINAMATH_GPT_gcd_A_B_l313_31399


namespace NUMINAMATH_GPT_rectangular_prism_lateral_edge_length_l313_31393

-- Definition of the problem conditions
def is_rectangular_prism (v : ℕ) : Prop := v = 8
def sum_lateral_edges (l : ℕ) : ℕ := 4 * l

-- Theorem stating the problem to prove
theorem rectangular_prism_lateral_edge_length :
  ∀ (v l : ℕ), is_rectangular_prism v → sum_lateral_edges l = 56 → l = 14 :=
by
  intros v l h1 h2
  sorry

end NUMINAMATH_GPT_rectangular_prism_lateral_edge_length_l313_31393


namespace NUMINAMATH_GPT_no_nat_nums_satisfy_gcd_lcm_condition_l313_31373

theorem no_nat_nums_satisfy_gcd_lcm_condition :
  ¬ ∃ (x y : ℕ), Nat.gcd x y + Nat.lcm x y = x + y + 2021 := 
sorry

end NUMINAMATH_GPT_no_nat_nums_satisfy_gcd_lcm_condition_l313_31373


namespace NUMINAMATH_GPT_zoe_bought_8_roses_l313_31390

-- Define the conditions
def each_flower_costs : ℕ := 3
def roses_bought (R : ℕ) : Prop := true
def daisies_bought : ℕ := 2
def total_spent : ℕ := 30

-- The main theorem to prove
theorem zoe_bought_8_roses (R : ℕ) (h1 : total_spent = 30) 
  (h2 : 3 * R + 3 * daisies_bought = total_spent) : R = 8 := by
  sorry

end NUMINAMATH_GPT_zoe_bought_8_roses_l313_31390


namespace NUMINAMATH_GPT_fraction_of_number_l313_31366

theorem fraction_of_number (F : ℚ) (h : 0.5 * F * 120 = 36) : F = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_number_l313_31366


namespace NUMINAMATH_GPT_number_of_real_solutions_of_equation_l313_31305

theorem number_of_real_solutions_of_equation :
  (∀ x : ℝ, ((2 : ℝ)^(4 * x + 2)) * ((4 : ℝ)^(2 * x + 8)) = ((8 : ℝ)^(3 * x + 7))) ↔ x = -3 :=
by sorry

end NUMINAMATH_GPT_number_of_real_solutions_of_equation_l313_31305


namespace NUMINAMATH_GPT_expected_value_of_biased_die_l313_31355

-- Definitions for probabilities
def prob1 : ℚ := 1 / 15
def prob2 : ℚ := 1 / 15
def prob3 : ℚ := 1 / 15
def prob4 : ℚ := 1 / 15
def prob5 : ℚ := 1 / 5
def prob6 : ℚ := 3 / 5

-- Definition for expected value
def expected_value : ℚ := (prob1 * 1) + (prob2 * 2) + (prob3 * 3) + (prob4 * 4) + (prob5 * 5) + (prob6 * 6)

theorem expected_value_of_biased_die : expected_value = 16 / 3 :=
by sorry

end NUMINAMATH_GPT_expected_value_of_biased_die_l313_31355


namespace NUMINAMATH_GPT_value_of_f_neg_4_l313_31394

noncomputable def f : ℝ → ℝ := λ x => if x ≥ 0 then Real.sqrt x else - (Real.sqrt (-x))

-- Definition that f is an odd function
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem value_of_f_neg_4 :
  isOddFunction f ∧ (∀ x, x ≥ 0 → f x = Real.sqrt x) → f (-4) = -2 := 
by
  sorry

end NUMINAMATH_GPT_value_of_f_neg_4_l313_31394


namespace NUMINAMATH_GPT_average_sleep_per_day_l313_31358

-- Define a structure for time duration
structure TimeDuration where
  hours : ℕ
  minutes : ℕ

-- Define instances for each day
def mondayNight : TimeDuration := ⟨8, 15⟩
def mondayNap : TimeDuration := ⟨0, 30⟩
def tuesdayNight : TimeDuration := ⟨7, 45⟩
def tuesdayNap : TimeDuration := ⟨0, 45⟩
def wednesdayNight : TimeDuration := ⟨8, 10⟩
def wednesdayNap : TimeDuration := ⟨0, 50⟩
def thursdayNight : TimeDuration := ⟨10, 25⟩
def thursdayNap : TimeDuration := ⟨0, 20⟩
def fridayNight : TimeDuration := ⟨7, 50⟩
def fridayNap : TimeDuration := ⟨0, 40⟩

-- Function to convert TimeDuration to total minutes
def totalMinutes (td : TimeDuration) : ℕ :=
  td.hours * 60 + td.minutes

-- Define the total sleep time for each day
def mondayTotal := totalMinutes mondayNight + totalMinutes mondayNap
def tuesdayTotal := totalMinutes tuesdayNight + totalMinutes tuesdayNap
def wednesdayTotal := totalMinutes wednesdayNight + totalMinutes wednesdayNap
def thursdayTotal := totalMinutes thursdayNight + totalMinutes thursdayNap
def fridayTotal := totalMinutes fridayNight + totalMinutes fridayNap

-- Sum of all sleep times
def totalSleep := mondayTotal + tuesdayTotal + wednesdayTotal + thursdayTotal + fridayTotal
-- Average sleep in minutes per day
def averageSleep := totalSleep / 5
-- Convert average sleep in total minutes back to hours and minutes
def averageHours := averageSleep / 60
def averageMinutes := averageSleep % 60

theorem average_sleep_per_day :
  averageHours = 9 ∧ averageMinutes = 6 := by
  sorry

end NUMINAMATH_GPT_average_sleep_per_day_l313_31358


namespace NUMINAMATH_GPT_surface_area_of_sphere_l313_31398

theorem surface_area_of_sphere (V : ℝ) (hV : V = 72 * π) : 
  ∃ A : ℝ, A = 36 * π * (2^(2/3)) := by 
  sorry

end NUMINAMATH_GPT_surface_area_of_sphere_l313_31398


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l313_31351

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

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l313_31351


namespace NUMINAMATH_GPT_general_term_l313_31309

def S (n : ℕ) : ℤ := n^2 - 4*n

noncomputable def a (n : ℕ) : ℤ := 
  if n = 1 then S 1
  else S n - S (n - 1)

theorem general_term (n : ℕ) (hn : n ≥ 1) : a n = (2 * n - 5) := by
  sorry

end NUMINAMATH_GPT_general_term_l313_31309


namespace NUMINAMATH_GPT_same_sum_sufficient_days_l313_31370

variable {S Wb Wc : ℝ}
variable (h1 : S = 12 * Wb)
variable (h2 : S = 24 * Wc)

theorem same_sum_sufficient_days : ∃ D : ℝ, D = 8 ∧ S = D * (Wb + Wc) :=
by
  use 8
  sorry

end NUMINAMATH_GPT_same_sum_sufficient_days_l313_31370


namespace NUMINAMATH_GPT_fraction_always_irreducible_l313_31310

-- Define the problem statement
theorem fraction_always_irreducible (n : ℤ) : gcd (39 * n + 4) (26 * n + 3) = 1 :=
sorry

end NUMINAMATH_GPT_fraction_always_irreducible_l313_31310


namespace NUMINAMATH_GPT_number_of_triangles_l313_31375

-- Defining the problem conditions
def ten_points : Finset (ℕ) := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- The main theorem to prove
theorem number_of_triangles : (ten_points.card.choose 3) = 120 :=
by
  sorry

end NUMINAMATH_GPT_number_of_triangles_l313_31375


namespace NUMINAMATH_GPT_common_ratio_of_series_l313_31311

-- Definition of the terms in the series
def term1 : ℚ := 7 / 8
def term2 : ℚ := - (5 / 12)

-- Definition of the common ratio
def common_ratio (a1 a2 : ℚ) : ℚ := a2 / a1

-- The theorem we need to prove that the common ratio is -10/21
theorem common_ratio_of_series : common_ratio term1 term2 = -10 / 21 :=
by
  -- We skip the proof with 'sorry'
  sorry

end NUMINAMATH_GPT_common_ratio_of_series_l313_31311


namespace NUMINAMATH_GPT_intersection_A_B_l313_31331

def setA : Set (ℝ × ℝ) := {p | ∃ (x: ℝ), p = (x, x^2)}
def setB : Set (ℝ × ℝ) := {p | ∃ (x: ℝ), p = (x, Real.sqrt x)}

theorem intersection_A_B :
  (setA ∩ setB) = {(0, 0), (1, 1)} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l313_31331


namespace NUMINAMATH_GPT_inradius_circumradius_inequality_l313_31345

variable {R r a b c : ℝ}

def inradius (ABC : Triangle) := r
def circumradius (ABC : Triangle) := R
def side_a (ABC : Triangle) := a
def side_b (ABC : Triangle) := b
def side_c (ABC : Triangle) := c

theorem inradius_circumradius_inequality (ABC : Triangle) :
  R / (2 * r) ≥ (64 * a^2 * b^2 * c^2 / ((4 * a^2 - (b - c)^2) * (4 * b^2 - (c - a)^2) * (4 * c^2 - (a - b)^2)))^2 :=
sorry

end NUMINAMATH_GPT_inradius_circumradius_inequality_l313_31345


namespace NUMINAMATH_GPT_pascals_triangle_ratio_456_l313_31336

theorem pascals_triangle_ratio_456 (n : ℕ) :
  (∃ r : ℕ,
    (n.choose r * 5 = (n.choose (r + 1)) * 4) ∧
    ((n.choose (r + 1)) * 6 = (n.choose (r + 2)) * 5)) →
  n = 98 :=
sorry

end NUMINAMATH_GPT_pascals_triangle_ratio_456_l313_31336


namespace NUMINAMATH_GPT_imaginary_part_of_l313_31378

theorem imaginary_part_of (i : ℂ) (h : i.im = 1) : (1 + i) ^ 5 = -14 - 4 * i := by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_l313_31378


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l313_31319

-- System 1 Definitions
def eq1 (x y : ℝ) : Prop := 3 * x - 2 * y = 9
def eq2 (x y : ℝ) : Prop := 2 * x + 3 * y = 19

-- System 2 Definitions
def eq3 (x y : ℝ) : Prop := (2 * x + 1) / 5 - 1 = (y - 1) / 3
def eq4 (x y : ℝ) : Prop := 2 * (y - x) - 3 * (1 - y) = 6

-- Theorem Statements
theorem system1_solution (x y : ℝ) : eq1 x y ∧ eq2 x y ↔ x = 5 ∧ y = 3 := by
  sorry

theorem system2_solution (x y : ℝ) : eq3 x y ∧ eq4 x y ↔ x = 4 ∧ y = 17 / 5 := by
  sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l313_31319


namespace NUMINAMATH_GPT_milk_water_equal_l313_31316

theorem milk_water_equal (a : ℕ) :
  let glass_a_initial := a
  let glass_b_initial := a
  let mixture_in_a := glass_a_initial + 1
  let milk_portion_in_a := 1 / mixture_in_a
  let water_portion_in_a := glass_a_initial / mixture_in_a
  let water_in_milk_glass := water_portion_in_a
  let milk_in_water_glass := milk_portion_in_a
  water_in_milk_glass = milk_in_water_glass := by
  sorry

end NUMINAMATH_GPT_milk_water_equal_l313_31316


namespace NUMINAMATH_GPT_value_of_M_l313_31323

theorem value_of_M
  (M : ℝ)
  (h : 25 / 100 * M = 55 / 100 * 4500) :
  M = 9900 :=
sorry

end NUMINAMATH_GPT_value_of_M_l313_31323


namespace NUMINAMATH_GPT_calculate_paint_area_l313_31397

def barn_length : ℕ := 12
def barn_width : ℕ := 15
def barn_height : ℕ := 6
def window_length : ℕ := 2
def window_width : ℕ := 2

def area_to_paint : ℕ := 796

theorem calculate_paint_area 
    (b_len : ℕ := barn_length) 
    (b_wid : ℕ := barn_width) 
    (b_hei : ℕ := barn_height) 
    (win_len : ℕ := window_length) 
    (win_wid : ℕ := window_width) : 
    b_len = 12 → 
    b_wid = 15 → 
    b_hei = 6 → 
    win_len = 2 → 
    win_wid = 2 →
    area_to_paint = 796 :=
by
  -- Here, the proof would be provided.
  -- This line is a placeholder (sorry) indicating that the proof is yet to be constructed.
  sorry

end NUMINAMATH_GPT_calculate_paint_area_l313_31397


namespace NUMINAMATH_GPT_total_legs_l313_31339

-- Define the conditions
def chickens : Nat := 7
def sheep : Nat := 5
def legs_chicken : Nat := 2
def legs_sheep : Nat := 4

-- State the problem as a theorem
theorem total_legs :
  chickens * legs_chicken + sheep * legs_sheep = 34 :=
by
  sorry -- Proof not provided

end NUMINAMATH_GPT_total_legs_l313_31339


namespace NUMINAMATH_GPT_shortest_wire_length_l313_31359

theorem shortest_wire_length
  (d1 d2 : ℝ) (h_d1 : d1 = 10) (h_d2 : d2 = 30) :
  let r1 := d1 / 2
  let r2 := d2 / 2
  let straight_sections := 2 * (r2 - r1)
  let curved_sections := 2 * Real.pi * r1 + 2 * Real.pi * r2
  let total_wire_length := straight_sections + curved_sections
  total_wire_length = 20 + 40 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_shortest_wire_length_l313_31359
