import Mathlib

namespace NUMINAMATH_GPT_solve_variables_l493_49325

theorem solve_variables (x y z : ℝ)
  (h1 : (x / 6) * 12 = 10)
  (h2 : (y / 4) * 8 = x)
  (h3 : (z / 3) * 5 + y = 20) :
  x = 5 ∧ y = 2.5 ∧ z = 10.5 :=
by { sorry }

end NUMINAMATH_GPT_solve_variables_l493_49325


namespace NUMINAMATH_GPT_jellybeans_left_l493_49349

theorem jellybeans_left :
  let initial_jellybeans := 500
  let total_kindergarten := 10
  let total_firstgrade := 10
  let total_secondgrade := 10
  let sick_kindergarten := 2
  let sick_secondgrade := 3
  let jellybeans_sick_kindergarten := 5
  let jellybeans_sick_secondgrade := 10
  let jellybeans_remaining_kindergarten := 3
  let jellybeans_firstgrade := 5
  let jellybeans_secondgrade_per_firstgrade := 5 / 2 * total_firstgrade
  let consumed_by_sick := sick_kindergarten * jellybeans_sick_kindergarten + sick_secondgrade * jellybeans_sick_secondgrade
  let remaining_kindergarten := total_kindergarten - sick_kindergarten
  let consumed_by_remaining := remaining_kindergarten * jellybeans_remaining_kindergarten + total_firstgrade * jellybeans_firstgrade + total_secondgrade * jellybeans_secondgrade_per_firstgrade
  let total_consumed := consumed_by_sick + consumed_by_remaining
  initial_jellybeans - total_consumed = 176 := by 
  sorry

end NUMINAMATH_GPT_jellybeans_left_l493_49349


namespace NUMINAMATH_GPT_circumcircle_radius_of_right_triangle_l493_49374

theorem circumcircle_radius_of_right_triangle (a b c : ℝ) (h1: a = 8) (h2: b = 6) (h3: c = 10) (h4: a^2 + b^2 = c^2) : (c / 2) = 5 := 
by
  sorry

end NUMINAMATH_GPT_circumcircle_radius_of_right_triangle_l493_49374


namespace NUMINAMATH_GPT_scientific_notation_103M_l493_49353

theorem scientific_notation_103M : 103000000 = 1.03 * 10^8 := sorry

end NUMINAMATH_GPT_scientific_notation_103M_l493_49353


namespace NUMINAMATH_GPT_minimum_dimes_needed_l493_49327

theorem minimum_dimes_needed (n : ℕ) 
  (sneaker_cost : ℝ := 58) 
  (ten_bills : ℝ := 50)
  (five_quarters : ℝ := 1.25) :
  ten_bills + five_quarters + (0.10 * n) ≥ sneaker_cost ↔ n ≥ 68 := 
by 
  sorry

end NUMINAMATH_GPT_minimum_dimes_needed_l493_49327


namespace NUMINAMATH_GPT_evaluate_polynomial_at_minus_two_l493_49338

def P (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x + 4

theorem evaluate_polynomial_at_minus_two :
  P (-2) = -18 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_minus_two_l493_49338


namespace NUMINAMATH_GPT_range_of_a_l493_49397

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (|x-2| + |x+3| < a) → false) → a ≤ 5 :=
sorry

end NUMINAMATH_GPT_range_of_a_l493_49397


namespace NUMINAMATH_GPT_pascal_triangle_row51_sum_l493_49382

theorem pascal_triangle_row51_sum : (Nat.choose 51 4) + (Nat.choose 51 6) = 18249360 :=
by
  sorry

end NUMINAMATH_GPT_pascal_triangle_row51_sum_l493_49382


namespace NUMINAMATH_GPT_cos_2beta_correct_l493_49399

open Real

theorem cos_2beta_correct (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
    (h3 : tan α = 1 / 7) (h4 : cos (α + β) = 2 * sqrt 5 / 5) :
    cos (2 * β) = 4 / 5 := 
  sorry

end NUMINAMATH_GPT_cos_2beta_correct_l493_49399


namespace NUMINAMATH_GPT_value_of_expression_l493_49320

theorem value_of_expression (x y : ℝ) (h1 : x + y = 3) (h2 : x^2 + y^2 - x * y = 4) : 
  x^4 + y^4 + x^3 * y + x * y^3 = 36 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l493_49320


namespace NUMINAMATH_GPT_dr_jones_remaining_salary_l493_49357

noncomputable def remaining_salary (salary rent food utilities insurances taxes transport emergency loan retirement : ℝ) : ℝ :=
  salary - (rent + food + utilities + insurances + taxes + transport + emergency + loan + retirement)

theorem dr_jones_remaining_salary :
  remaining_salary 6000 640 385 (1/4 * 6000) (1/5 * 6000) (0.10 * 6000) (0.03 * 6000) (0.02 * 6000) 300 (0.05 * 6000) = 1275 :=
by
  sorry

end NUMINAMATH_GPT_dr_jones_remaining_salary_l493_49357


namespace NUMINAMATH_GPT_total_area_correct_l493_49394

def initial_length : ℕ := 13
def initial_width : ℕ := 18
def increase : ℕ := 2
def num_extra_rooms_same_size : ℕ := 3
def num_double_size_rooms : ℕ := 1

def increased_length : ℕ := initial_length + increase
def increased_width : ℕ := initial_width + increase

def area_of_one_room (length width : ℕ) : ℕ := length * width

def number_of_rooms : ℕ := 1 + num_extra_rooms_same_size

def total_area_same_size_rooms : ℕ := number_of_rooms * area_of_one_room increased_length increased_width

def total_area_extra_size_room : ℕ := num_double_size_rooms * 2 * area_of_one_room increased_length increased_width

def total_area : ℕ := total_area_same_size_rooms + total_area_extra_size_room

theorem total_area_correct : total_area = 1800 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_area_correct_l493_49394


namespace NUMINAMATH_GPT_cone_height_from_sphere_l493_49385

theorem cone_height_from_sphere (d_sphere d_base : ℝ) (h : ℝ) (V_sphere : ℝ) (V_cone : ℝ) 
  (h₁ : d_sphere = 6) 
  (h₂ : d_base = 12)
  (h₃ : V_sphere = 36 * Real.pi)
  (h₄ : V_cone = (1/3) * Real.pi * (d_base / 2)^2 * h) 
  (h₅ : V_sphere = V_cone) :
  h = 3 := by
  sorry

end NUMINAMATH_GPT_cone_height_from_sphere_l493_49385


namespace NUMINAMATH_GPT_glass_bottles_count_l493_49316

-- Declare the variables for the conditions
variable (G : ℕ)

-- Define the conditions
def aluminum_cans : ℕ := 8
def total_litter : ℕ := 18

-- State the theorem
theorem glass_bottles_count : G + aluminum_cans = total_litter → G = 10 :=
by
  intro h
  -- place proof here
  sorry

end NUMINAMATH_GPT_glass_bottles_count_l493_49316


namespace NUMINAMATH_GPT_rectangular_prism_width_l493_49369

variables (w : ℝ)

theorem rectangular_prism_width (h : ℝ) (l : ℝ) (d : ℝ) (hyp_l : l = 5) (hyp_h : h = 7) (hyp_d : d = 15) :
  w = Real.sqrt 151 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_rectangular_prism_width_l493_49369


namespace NUMINAMATH_GPT_car_travel_distance_l493_49366

noncomputable def distance_in_miles (b t : ℝ) : ℝ :=
  (25 * b) / (1320 * t)

theorem car_travel_distance (b t : ℝ) : 
  let distance_in_feet := (b / 3) * (300 / t)
  let distance_in_miles' := distance_in_feet / 5280
  distance_in_miles' = distance_in_miles b t := 
by
  sorry

end NUMINAMATH_GPT_car_travel_distance_l493_49366


namespace NUMINAMATH_GPT_min_slope_of_tangent_l493_49324

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x - 10

theorem min_slope_of_tangent : (∀ x : ℝ, 3 * (x + 1)^2 + 3 ≥ 3) :=
by 
  sorry

end NUMINAMATH_GPT_min_slope_of_tangent_l493_49324


namespace NUMINAMATH_GPT_find_function_value_at_2_l493_49337

variables {f : ℕ → ℕ}

theorem find_function_value_at_2 (H : ∀ x : ℕ, Nat.succ (Nat.succ x * Nat.succ x + f x) = 12) : f 2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_function_value_at_2_l493_49337


namespace NUMINAMATH_GPT_sin_zero_degrees_l493_49387

theorem sin_zero_degrees : Real.sin 0 = 0 := 
by {
  -- The proof is added here (as requested no proof is required, hence using sorry)
  sorry
}

end NUMINAMATH_GPT_sin_zero_degrees_l493_49387


namespace NUMINAMATH_GPT_each_dolphin_training_hours_l493_49329

theorem each_dolphin_training_hours
  (num_dolphins : ℕ)
  (num_trainers : ℕ)
  (hours_per_trainer : ℕ)
  (total_hours : ℕ := num_trainers * hours_per_trainer)
  (hours_per_dolphin_daily : ℕ := total_hours / num_dolphins)
  (h1 : num_dolphins = 4)
  (h2 : num_trainers = 2)
  (h3 : hours_per_trainer = 6) :
  hours_per_dolphin_daily = 3 :=
  by sorry

end NUMINAMATH_GPT_each_dolphin_training_hours_l493_49329


namespace NUMINAMATH_GPT_sally_quarters_l493_49358

theorem sally_quarters (initial_quarters spent_quarters final_quarters : ℕ) 
  (h1 : initial_quarters = 760) 
  (h2 : spent_quarters = 418) 
  (calc_final : final_quarters = initial_quarters - spent_quarters) : 
  final_quarters = 342 := 
by 
  rw [h1, h2] at calc_final 
  exact calc_final

end NUMINAMATH_GPT_sally_quarters_l493_49358


namespace NUMINAMATH_GPT_line_equation_through_M_P_Q_l493_49301

-- Given that M is the midpoint between P and Q, we should have:
-- M = (1, -2)
-- P = (2, 0)
-- Q = (0, -4)
-- We need to prove that the line passing through these points has the equation 2x - y - 4 = 0

theorem line_equation_through_M_P_Q :
  ∀ (x y : ℝ), (1 - 2 = (2 * (x - 1)) ∧ 0 - 2 = (2 * (0 - (-2)))) ->
  (x - y - 4 = 0) := 
by
  sorry

end NUMINAMATH_GPT_line_equation_through_M_P_Q_l493_49301


namespace NUMINAMATH_GPT_smallest_solution_eq_l493_49306

theorem smallest_solution_eq (x : ℝ) (h : 1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) :
  x = 4 - Real.sqrt 2 := 
  sorry

end NUMINAMATH_GPT_smallest_solution_eq_l493_49306


namespace NUMINAMATH_GPT_inverse_of_A_cubed_l493_49328

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![3, 7],
    ![-2, -5]]

theorem inverse_of_A_cubed :
  (A_inv ^ 3)⁻¹ = ![![13, -15],
                     ![-14, -29]] :=
by
  sorry

end NUMINAMATH_GPT_inverse_of_A_cubed_l493_49328


namespace NUMINAMATH_GPT_lowest_point_on_graph_l493_49360

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2 * x + 2) / (x + 1)

theorem lowest_point_on_graph : ∃ (x y : ℝ), x = 0 ∧ y = 2 ∧ ∀ z > -1, f z ≥ y ∧ f x = y := by
  sorry

end NUMINAMATH_GPT_lowest_point_on_graph_l493_49360


namespace NUMINAMATH_GPT_solve_fractional_eq1_l493_49302

theorem solve_fractional_eq1 : ¬ ∃ (x : ℝ), 1 / (x - 2) = (1 - x) / (2 - x) - 3 :=
by sorry

end NUMINAMATH_GPT_solve_fractional_eq1_l493_49302


namespace NUMINAMATH_GPT_shaded_region_correct_l493_49379

def side_length_ABCD : ℝ := 8
def side_length_BEFG : ℝ := 6

def area_square (side_length : ℝ) : ℝ := side_length ^ 2

def area_ABCD : ℝ := area_square side_length_ABCD
def area_BEFG : ℝ := area_square side_length_BEFG

def shaded_region_area : ℝ :=
  area_ABCD + area_BEFG - 32

theorem shaded_region_correct :
  shaded_region_area = 32 :=
by
  -- Proof omitted, but placeholders match problem conditions and answer
  sorry

end NUMINAMATH_GPT_shaded_region_correct_l493_49379


namespace NUMINAMATH_GPT_sale_in_third_month_l493_49307

def average_sale (s1 s2 s3 s4 s5 s6 : ℕ) : ℕ :=
  (s1 + s2 + s3 + s4 + s5 + s6) / 6

theorem sale_in_third_month
  (S1 S2 S3 S4 S5 S6 : ℕ)
  (h1 : S1 = 6535)
  (h2 : S2 = 6927)
  (h4 : S4 = 7230)
  (h5 : S5 = 6562)
  (h6 : S6 = 4891)
  (havg : average_sale S1 S2 S3 S4 S5 S6 = 6500) :
  S3 = 6855 := 
sorry

end NUMINAMATH_GPT_sale_in_third_month_l493_49307


namespace NUMINAMATH_GPT_total_goals_l493_49386

theorem total_goals (B M : ℕ) (hB : B = 4) (hM : M = 3 * B) : B + M = 16 := by
  sorry

end NUMINAMATH_GPT_total_goals_l493_49386


namespace NUMINAMATH_GPT_mark_buttons_l493_49370

theorem mark_buttons (initial_buttons : ℕ) (shane_buttons : ℕ) (sam_buttons : ℕ) :
  initial_buttons = 14 →
  shane_buttons = 3 * initial_buttons →
  sam_buttons = (initial_buttons + shane_buttons) / 2 →
  final_buttons = (initial_buttons + shane_buttons) - sam_buttons →
  final_buttons = 28 :=
by
  sorry

end NUMINAMATH_GPT_mark_buttons_l493_49370


namespace NUMINAMATH_GPT_set_intersection_l493_49321

theorem set_intersection :
  let A := {x : ℝ | 0 < x}
  let B := {x : ℝ | -1 ≤ x ∧ x < 3}
  A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := 
by
  sorry

end NUMINAMATH_GPT_set_intersection_l493_49321


namespace NUMINAMATH_GPT_chess_group_unique_pairings_l493_49365

theorem chess_group_unique_pairings:
  ∀ (players games : ℕ), players = 50 → games = 1225 →
  (∃ (games_per_pair : ℕ), games_per_pair = 1 ∧ (∀ p: ℕ, p < players → (players - 1) * games_per_pair = games)) :=
by
  sorry

end NUMINAMATH_GPT_chess_group_unique_pairings_l493_49365


namespace NUMINAMATH_GPT_perpendicular_lines_l493_49318

theorem perpendicular_lines (a : ℝ) :
  (∃ x y : ℝ, ax + 2 * y + 6 = 0) ∧ (∃ x y : ℝ, x + (a - 1) * y + a^2 - 1 = 0) ∧ (∀ m1 m2 : ℝ, m1 * m2 = -1) →
  a = 2/3 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_l493_49318


namespace NUMINAMATH_GPT_find_minimum_l493_49310

theorem find_minimum (a b c : ℝ) : ∃ (m : ℝ), m = min a (min b c) := 
  sorry

end NUMINAMATH_GPT_find_minimum_l493_49310


namespace NUMINAMATH_GPT_tangent_line_at_a1_one_zero_per_interval_l493_49393

noncomputable def f (a x : ℝ) : ℝ := Real.log (1 + x) + a * x * Real.exp (-x)

theorem tangent_line_at_a1 (a : ℝ) (h : a = 1) : 
  (∃ (m b : ℝ), ∀ x, f a x = m * x + b ∧ m = 2 ∧ b = 0) :=
by
  sorry

theorem one_zero_per_interval (a : ℝ) :
  (∃ x : ℝ, -1 < x ∧ x < 0 ∧ f a x = 0) ∧ (∃ x : ℝ, 0 < x ∧ f a x = 0) ↔ a < -1 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_a1_one_zero_per_interval_l493_49393


namespace NUMINAMATH_GPT_peter_age_fraction_l493_49347

theorem peter_age_fraction 
  (harriet_age : ℕ) 
  (mother_age : ℕ) 
  (peter_age_plus_four : ℕ) 
  (harriet_age_plus_four : ℕ) 
  (harriet_age_current : harriet_age = 13)
  (mother_age_current : mother_age = 60)
  (peter_age_condition : peter_age_plus_four = 2 * harriet_age_plus_four)
  (harriet_four_years : harriet_age_plus_four = harriet_age + 4)
  (peter_four_years : ∀ P : ℕ, peter_age_plus_four = P + 4)
: ∃ P : ℕ, P = 30 ∧ P = mother_age / 2 := 
sorry

end NUMINAMATH_GPT_peter_age_fraction_l493_49347


namespace NUMINAMATH_GPT_ellipse_eccentricity_l493_49309

theorem ellipse_eccentricity (a : ℝ) (h : a > 0) 
  (ell_eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / 5 = 1 ↔ x^2 / a^2 + y^2 / 5 = 1)
  (ecc_eq : (eccentricity : ℝ) = 2 / 3) : 
  a = 3 := 
sorry

end NUMINAMATH_GPT_ellipse_eccentricity_l493_49309


namespace NUMINAMATH_GPT_coin_and_die_probability_l493_49350

-- Probability of a coin showing heads
def P_heads : ℚ := 2 / 3

-- Probability of a die showing 5
def P_die_5 : ℚ := 1 / 6

-- Probability of both events happening together
def P_heads_and_die_5 : ℚ := P_heads * P_die_5

-- Theorem statement: Proving the calculated probability equals the expected value
theorem coin_and_die_probability : P_heads_and_die_5 = 1 / 9 := by
  -- The detailed proof is omitted here.
  sorry

end NUMINAMATH_GPT_coin_and_die_probability_l493_49350


namespace NUMINAMATH_GPT_minimum_additional_marbles_l493_49323

theorem minimum_additional_marbles (friends marbles : ℕ) (h_friends : friends = 12) (h_marbles : marbles = 34) : 
  ∃ additional_marbles : ℕ, additional_marbles = 44 :=
by
  -- The formal proof would go here.
  sorry

end NUMINAMATH_GPT_minimum_additional_marbles_l493_49323


namespace NUMINAMATH_GPT_sum_of_digits_of_largest_n_l493_49339

def is_prime (p : ℕ) : Prop := Nat.Prime p

def is_single_digit_prime (p : ℕ) : Prop := is_prime p ∧ p < 10

noncomputable def required_n (d e : ℕ) : ℕ := d * e * (d^2 + 10 * e)

def sum_of_digits (n : ℕ) : ℕ := 
  if n < 10 then n 
  else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_largest_n : 
  ∃ (d e : ℕ), 
    is_single_digit_prime d ∧ is_single_digit_prime e ∧ 
    is_prime (d^2 + 10 * e) ∧ 
    (∀ d' e' : ℕ, is_single_digit_prime d' ∧ is_single_digit_prime e' ∧ is_prime (d'^2 + 10 * e') → required_n d e ≥ required_n d' e') ∧ 
    sum_of_digits (required_n d e) = 9 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_of_largest_n_l493_49339


namespace NUMINAMATH_GPT_algebraic_expression_equals_one_l493_49322

variable (m n : ℝ)

theorem algebraic_expression_equals_one
  (hm : m ≠ 0)
  (hn : n ≠ 0)
  (h_eq : m - n = 1 / 2) :
  (m^2 - n^2) / (2 * m^2 + 2 * m * n) / (m - (2 * m * n - n^2) / m) = 1 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_equals_one_l493_49322


namespace NUMINAMATH_GPT_probability_of_number_less_than_three_l493_49315

theorem probability_of_number_less_than_three :
  let faces : Finset ℕ := {1, 2, 3, 4, 5, 6}
  let favorable_outcomes : Finset ℕ := {1, 2}
  (favorable_outcomes.card : ℚ) / (faces.card : ℚ) = 1 / 3 :=
by
  -- This is the placeholder for the actual proof.
  sorry

end NUMINAMATH_GPT_probability_of_number_less_than_three_l493_49315


namespace NUMINAMATH_GPT_record_expenditure_20_l493_49326

-- Define the concept of recording financial transactions
def record_income (amount : ℤ) : ℤ := amount

def record_expenditure (amount : ℤ) : ℤ := -amount

-- Given conditions
variable (income : ℤ) (expenditure : ℤ)

-- Condition: the income of 30 yuan is recorded as +30 yuan
axiom income_record : record_income 30 = 30

-- Prove an expenditure of 20 yuan is recorded as -20 yuan
theorem record_expenditure_20 : record_expenditure 20 = -20 := 
  by sorry

end NUMINAMATH_GPT_record_expenditure_20_l493_49326


namespace NUMINAMATH_GPT_hyperbola_eccentricity_a_l493_49368

theorem hyperbola_eccentricity_a (a : ℝ) (ha : a > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / 3 = 1) ∧ (∃ (e : ℝ), e = 2 ∧ e = Real.sqrt (a^2 + 3) / a) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_a_l493_49368


namespace NUMINAMATH_GPT_average_of_sequence_l493_49304

theorem average_of_sequence (z : ℝ) : 
  (0 + 3 * z + 9 * z + 27 * z + 81 * z) / 5 = 24 * z :=
by
  sorry

end NUMINAMATH_GPT_average_of_sequence_l493_49304


namespace NUMINAMATH_GPT_perp_line_eq_l493_49348

theorem perp_line_eq (x y : ℝ) (h1 : (x, y) = (1, 1)) (h2 : y = 2 * x) :
  ∃ a b c : ℝ, a * x + b * y + c = 0 ∧ a = 1 ∧ b = 2 ∧ c = -3 :=
by 
  sorry

end NUMINAMATH_GPT_perp_line_eq_l493_49348


namespace NUMINAMATH_GPT_factorize_difference_of_squares_l493_49305

theorem factorize_difference_of_squares (a : ℝ) : a^2 - 4 = (a + 2) * (a - 2) :=
by
  sorry

end NUMINAMATH_GPT_factorize_difference_of_squares_l493_49305


namespace NUMINAMATH_GPT_reciprocal_of_5_over_7_l493_49372

theorem reciprocal_of_5_over_7 : (5 / 7 : ℚ) * (7 / 5) = 1 := by
  sorry

end NUMINAMATH_GPT_reciprocal_of_5_over_7_l493_49372


namespace NUMINAMATH_GPT_traveler_drank_32_ounces_l493_49378

-- Definition of the given condition
def total_gallons : ℕ := 2
def ounces_per_gallon : ℕ := 128
def total_ounces := total_gallons * ounces_per_gallon
def camel_multiple : ℕ := 7
def traveler_ounces (T : ℕ) := T
def camel_ounces (T : ℕ) := camel_multiple * T
def total_drunk (T : ℕ) := traveler_ounces T + camel_ounces T

-- Theorem to prove
theorem traveler_drank_32_ounces :
  ∃ T : ℕ, total_drunk T = total_ounces ∧ T = 32 :=
by 
  sorry

end NUMINAMATH_GPT_traveler_drank_32_ounces_l493_49378


namespace NUMINAMATH_GPT_identify_counterfeit_13_coins_identify_and_determine_weight_14_coins_impossible_with_14_coins_l493_49352

-- Proving the identification of the counterfeit coin among 13 coins in 3 weighings
theorem identify_counterfeit_13_coins (coins : Fin 13 → Real) (is_counterfeit : ∃! i, coins i ≠ coins 0) :
  ∃ measure_count : ℕ, measure_count <= 3 ∧ 
    ∃ i, (coins i ≠ coins 0) :=
sorry

-- Proving counterfeit coin weight determination with an additional genuine coin using 3 weighings
theorem identify_and_determine_weight_14_coins (coins : Fin 14 → Real) (genuine : Real) (is_counterfeit : ∃! i, coins i ≠ genuine) :
  ∃ method_exists : Prop, 
    (method_exists ∧ ∃ measure_count : ℕ, measure_count <= 3 ∧ 
    ∃ (i : Fin 14), coins i ≠ genuine) :=
sorry

-- Proving the impossibility of identifying counterfeit coin among 14 coins using 3 weighings
theorem impossible_with_14_coins (coins : Fin 14 → Real) (is_counterfeit : ∃! i, coins i ≠ coins 0) :
  ¬ (∃ measure_count : ℕ, measure_count <= 3 ∧ 
    ∃ i, (coins i ≠ coins 0)) :=
sorry

end NUMINAMATH_GPT_identify_counterfeit_13_coins_identify_and_determine_weight_14_coins_impossible_with_14_coins_l493_49352


namespace NUMINAMATH_GPT_stratified_sampling_number_of_products_drawn_l493_49303

theorem stratified_sampling_number_of_products_drawn (T S W X : ℕ) 
  (h1 : T = 1024) (h2 : S = 64) (h3 : W = 128) :
  X = S * (W / T) → X = 8 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_number_of_products_drawn_l493_49303


namespace NUMINAMATH_GPT_percentage_increase_in_expenditure_l493_49388

/-- Given conditions:
- The price of sugar increased by 32%
- The family's original monthly sugar consumption was 30 kg
- The family's new monthly sugar consumption is 25 kg
- The family's expenditure on sugar increased by 10%

Prove that the percentage increase in the family's expenditure on sugar is 10%. -/
theorem percentage_increase_in_expenditure (P : ℝ) :
  let initial_consumption := 30
  let new_consumption := 25
  let price_increase := 0.32
  let original_price := P
  let new_price := (1 + price_increase) * original_price
  let original_expenditure := initial_consumption * original_price
  let new_expenditure := new_consumption * new_price
  let expenditure_increase := new_expenditure - original_expenditure
  let percentage_increase := (expenditure_increase / original_expenditure) * 100
  percentage_increase = 10 := sorry

end NUMINAMATH_GPT_percentage_increase_in_expenditure_l493_49388


namespace NUMINAMATH_GPT_cost_to_fill_pool_l493_49308

noncomputable def pool_cost : ℝ :=
  let base_width := 6
  let top_width := 4
  let length := 20
  let depth := 10
  let conversion_factor := 25
  let price_per_liter := 3
  let tax_rate := 0.08
  let discount_rate := 0.05
  let volume := 0.5 * depth * (base_width + top_width) * length
  let liters := volume * conversion_factor
  let initial_cost := liters * price_per_liter
  let cost_with_tax := initial_cost * (1 + tax_rate)
  let final_cost := cost_with_tax * (1 - discount_rate)
  final_cost

theorem cost_to_fill_pool : pool_cost = 76950 := by
  sorry

end NUMINAMATH_GPT_cost_to_fill_pool_l493_49308


namespace NUMINAMATH_GPT_distinct_ways_to_distribute_l493_49334

theorem distinct_ways_to_distribute :
  ∃ (n : ℕ), n = 7 ∧ ∀ (balls : ℕ) (boxes : ℕ)
    (indistinguishable_balls : Prop := true) 
    (indistinguishable_boxes : Prop := true), 
    balls = 6 → boxes = 3 → 
    indistinguishable_balls → 
    indistinguishable_boxes → 
    n = 7 :=
by
  sorry

end NUMINAMATH_GPT_distinct_ways_to_distribute_l493_49334


namespace NUMINAMATH_GPT_Janice_earnings_l493_49392

theorem Janice_earnings (days_worked_per_week : ℕ) (earnings_per_day : ℕ) (overtime_shifts : ℕ) (overtime_earnings_per_shift : ℕ)
  (h1 : days_worked_per_week = 5)
  (h2 : earnings_per_day = 30)
  (h3 : overtime_shifts = 3)
  (h4 : overtime_earnings_per_shift = 15) :
  (days_worked_per_week * earnings_per_day) + (overtime_shifts * overtime_earnings_per_shift) = 195 :=
by {
  sorry
}

end NUMINAMATH_GPT_Janice_earnings_l493_49392


namespace NUMINAMATH_GPT_price_increase_count_l493_49371

-- Conditions
def original_price (P : ℝ) : ℝ := P
def increase_factor : ℝ := 1.15
def final_factor : ℝ := 1.3225

-- The theorem that states the number of times the price increased
theorem price_increase_count (n : ℕ) :
  increase_factor ^ n = final_factor → n = 2 :=
by
  sorry

end NUMINAMATH_GPT_price_increase_count_l493_49371


namespace NUMINAMATH_GPT_simplify_tan_alpha_l493_49361

noncomputable def f (α : ℝ) : ℝ :=
(Real.sin (Real.pi / 2 + α) + Real.sin (-Real.pi - α)) /
  (3 * Real.cos (2 * Real.pi - α) + Real.cos (3 * Real.pi / 2 - α))

theorem simplify_tan_alpha (α : ℝ) (h : Real.tan α = 1) : f α = 1 := by
  sorry

end NUMINAMATH_GPT_simplify_tan_alpha_l493_49361


namespace NUMINAMATH_GPT_jeffs_mean_l493_49364

-- Define Jeff's scores as a list or array
def jeffsScores : List ℚ := [86, 94, 87, 96, 92, 89]

-- Prove that the arithmetic mean of Jeff's scores is 544 / 6
theorem jeffs_mean : (jeffsScores.sum / jeffsScores.length) = (544 / 6) := by
  sorry

end NUMINAMATH_GPT_jeffs_mean_l493_49364


namespace NUMINAMATH_GPT_min_bottles_needed_l493_49383

theorem min_bottles_needed (num_people : ℕ) (exchange_rate : ℕ) (bottles_needed_per_person : ℕ) (total_bottles_purchased : ℕ):
  num_people = 27 → exchange_rate = 3 → bottles_needed_per_person = 1 → total_bottles_purchased = 18 → 
  ∀ n, n = num_people → (n / bottles_needed_per_person) = 27 ∧ (num_people * 2 / 3) = 18 :=
by
  intros
  sorry

end NUMINAMATH_GPT_min_bottles_needed_l493_49383


namespace NUMINAMATH_GPT_problem_l493_49384

noncomputable def f (x : ℝ) : ℝ := (Real.sin (x / 4)) ^ 6 + (Real.cos (x / 4)) ^ 6

theorem problem : (derivative^[2008] f 0) = 3 / 8 := by sorry

end NUMINAMATH_GPT_problem_l493_49384


namespace NUMINAMATH_GPT_employees_without_any_benefit_l493_49300

def employees_total : ℕ := 480
def employees_salary_increase : ℕ := 48
def employees_travel_increase : ℕ := 96
def employees_both_increases : ℕ := 24
def employees_vacation_days : ℕ := 72

theorem employees_without_any_benefit : (employees_total - ((employees_salary_increase + employees_travel_increase + employees_vacation_days) - employees_both_increases)) = 288 :=
by
  sorry

end NUMINAMATH_GPT_employees_without_any_benefit_l493_49300


namespace NUMINAMATH_GPT_haley_deleted_pictures_l493_49363

variable (zoo_pictures : ℕ) (museum_pictures : ℕ) (remaining_pictures : ℕ) (deleted_pictures : ℕ)

theorem haley_deleted_pictures :
  zoo_pictures = 50 → museum_pictures = 8 → remaining_pictures = 20 →
  deleted_pictures = zoo_pictures + museum_pictures - remaining_pictures →
  deleted_pictures = 38 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end NUMINAMATH_GPT_haley_deleted_pictures_l493_49363


namespace NUMINAMATH_GPT_upper_seat_ticket_price_l493_49391

variable (U : ℝ) 

-- Conditions
def lower_seat_price : ℝ := 30
def total_tickets_sold : ℝ := 80
def total_revenue : ℝ := 2100
def lower_tickets_sold : ℝ := 50

theorem upper_seat_ticket_price :
  (lower_seat_price * lower_tickets_sold + (total_tickets_sold - lower_tickets_sold) * U = total_revenue) →
  U = 20 := by
  sorry

end NUMINAMATH_GPT_upper_seat_ticket_price_l493_49391


namespace NUMINAMATH_GPT_man_age_year_l493_49336

theorem man_age_year (x : ℕ) (h1 : x^2 = 1892) (h2 : 1850 ≤ x ∧ x ≤ 1900) :
  (x = 44) → (1892 = 1936) := by
sorry

end NUMINAMATH_GPT_man_age_year_l493_49336


namespace NUMINAMATH_GPT_quadratic_passing_point_l493_49395

theorem quadratic_passing_point :
  ∃ (m : ℝ), (∀ (x : ℝ), y = 18 * (x + 1) ^ 2 - 10 → y = 8 → x = 0) →
  (∀ (x : ℝ), y = 18 * (x + 1) ^ 2 - 10 → y = -10 → x = -1) →
  (∀ (x : ℝ), y = 18 * (x + 1) ^ 2 - 10 → y = m → x = 5) →
  m = 638 := by
  sorry

end NUMINAMATH_GPT_quadratic_passing_point_l493_49395


namespace NUMINAMATH_GPT_min_number_of_each_coin_l493_49367

def total_cost : ℝ := 1.30 + 0.75 + 0.50 + 0.45

def nickel_value : ℝ := 0.05
def dime_value : ℝ := 0.10
def quarter_value : ℝ := 0.25
def half_dollar_value : ℝ := 0.50

def min_coins :=
  ∃ (n q d h : ℕ), 
  (n ≥ 1) ∧ (q ≥ 1) ∧ (d ≥ 1) ∧ (h ≥ 1) ∧ 
  ((n * nickel_value) + (q * quarter_value) + (d * dime_value) + (h * half_dollar_value) = total_cost)

theorem min_number_of_each_coin :
  min_coins ↔ (5 * half_dollar_value + 1 * quarter_value + 2 * dime_value + 1 * nickel_value = total_cost) :=
by sorry

end NUMINAMATH_GPT_min_number_of_each_coin_l493_49367


namespace NUMINAMATH_GPT_fred_fewer_games_l493_49373

/-- Fred went to 36 basketball games last year -/
def games_last_year : ℕ := 36

/-- Fred went to 25 basketball games this year -/
def games_this_year : ℕ := 25

/-- Prove that Fred went to 11 fewer games this year compared to last year -/
theorem fred_fewer_games : games_last_year - games_this_year = 11 := by
  sorry

end NUMINAMATH_GPT_fred_fewer_games_l493_49373


namespace NUMINAMATH_GPT_sum_of_a_b_l493_49390

theorem sum_of_a_b (a b : ℝ) (h1 : a > b) (h2 : |a| = 9) (h3 : b^2 = 4) : a + b = 11 ∨ a + b = 7 := 
sorry

end NUMINAMATH_GPT_sum_of_a_b_l493_49390


namespace NUMINAMATH_GPT_max_equilateral_triangles_l493_49331

theorem max_equilateral_triangles (length : ℕ) (n : ℕ) (segments : ℕ) : 
  (length = 2) → (segments = 6) → (∀ t, 1 ≤ t ∧ t ≤ 4 → t = 4) :=
by 
  intros length_eq segments_eq h
  sorry

end NUMINAMATH_GPT_max_equilateral_triangles_l493_49331


namespace NUMINAMATH_GPT_product_value_l493_49396

noncomputable def product_of_integers (A B C D : ℕ) : ℕ :=
  A * B * C * D

theorem product_value :
  ∃ (A B C D : ℕ), A + B + C + D = 72 ∧ 
                    A + 2 = B - 2 ∧ 
                    A + 2 = C * 2 ∧ 
                    A + 2 = D / 2 ∧ 
                    product_of_integers A B C D = 64512 :=
by
  sorry

end NUMINAMATH_GPT_product_value_l493_49396


namespace NUMINAMATH_GPT_parallel_vectors_implies_value_of_x_l493_49355

-- Define the vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Define the condition for parallel vectors
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (u.1 = k * v.1) ∧ (u.2 = k * v.2)

-- The proof statement
theorem parallel_vectors_implies_value_of_x : ∀ (x : ℝ), parallel a (b x) → x = 6 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_parallel_vectors_implies_value_of_x_l493_49355


namespace NUMINAMATH_GPT_square_vertex_distance_l493_49362

noncomputable def inner_square_perimeter : ℝ := 24
noncomputable def outer_square_perimeter : ℝ := 32
noncomputable def greatest_distance : ℝ := 7 * Real.sqrt 2

theorem square_vertex_distance :
  let inner_side := inner_square_perimeter / 4
  let outer_side := outer_square_perimeter / 4
  let inner_diagonal := Real.sqrt (inner_side ^ 2 + inner_side ^ 2)
  let outer_diagonal := Real.sqrt (outer_side ^ 2 + outer_side ^ 2)
  let distance := (inner_diagonal / 2) + (outer_diagonal / 2)
  distance = greatest_distance :=
by
  sorry

end NUMINAMATH_GPT_square_vertex_distance_l493_49362


namespace NUMINAMATH_GPT_trigonometric_identity_example_l493_49314

theorem trigonometric_identity_example :
  2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_example_l493_49314


namespace NUMINAMATH_GPT_arrasta_um_proof_l493_49356

variable (n : ℕ)

def arrasta_um_possible_moves (n : ℕ) : ℕ :=
  6 * n - 8

theorem arrasta_um_proof (n : ℕ) (h : n ≥ 2) : arrasta_um_possible_moves n =
6 * n - 8 := by
  sorry

end NUMINAMATH_GPT_arrasta_um_proof_l493_49356


namespace NUMINAMATH_GPT_batsman_average_l493_49330

theorem batsman_average (A : ℕ) (H : (16 * A + 82) / 17 = A + 3) : (A + 3 = 34) :=
sorry

end NUMINAMATH_GPT_batsman_average_l493_49330


namespace NUMINAMATH_GPT_find_fifth_number_l493_49381

-- Define the sets and their conditions
def first_set : List ℕ := [28, 70, 88, 104]
def second_set : List ℕ := [50, 62, 97, 124]

-- Define the means
def mean_first_set (x : ℕ) (y : ℕ) : ℚ := (28 + x + 70 + 88 + y) / 5
def mean_second_set (x : ℕ) : ℚ := (50 + 62 + 97 + 124 + x) / 5

-- Conditions given in the problem
axiom mean_first_set_condition (x y : ℕ) : mean_first_set x y = 67
axiom mean_second_set_condition (x : ℕ) : mean_second_set x = 75.6

-- Lean 4 theorem statement to prove the fifth number in the first set is 104 given above conditions
theorem find_fifth_number : ∃ x y, mean_first_set x y = 67 ∧ mean_second_set x = 75.6 ∧ y = 104 := by
  sorry

end NUMINAMATH_GPT_find_fifth_number_l493_49381


namespace NUMINAMATH_GPT_jenicek_decorated_cookies_total_time_for_work_jenicek_decorating_time_l493_49375

/-- Conditions:
1. The grandmother decorates five gingerbread cookies for every cycle.
2. Little Mary decorates three gingerbread cookies for every cycle.
3. Little John decorates two gingerbread cookies for every cycle.
4. All three together decorated five trays, with each tray holding twelve gingerbread cookies.
5. Little John also sorted the gingerbread cookies onto trays twelve at a time and carried them to the pantry.
6. The grandmother decorates one gingerbread cookie in four minutes.
-/

def decorated_cookies_per_cycle := 10
def total_trays := 5
def cookies_per_tray := 12
def total_cookies := total_trays * cookies_per_tray
def babicka_cookies_per_cycle := 5
def marenka_cookies_per_cycle := 3
def jenicek_cookies_per_cycle := 2
def babicka_time_per_cookie := 4

theorem jenicek_decorated_cookies :
  (total_cookies - (total_cookies / decorated_cookies_per_cycle * marenka_cookies_per_cycle + total_cookies / decorated_cookies_per_cycle * babicka_cookies_per_cycle)) = 4 :=
sorry

theorem total_time_for_work :
  (total_cookies / decorated_cookies_per_cycle * babicka_time_per_cookie * babicka_cookies_per_cycle) = 140 :=
sorry

theorem jenicek_decorating_time :
  (4 / jenicek_cookies_per_cycle * babicka_time_per_cookie * babicka_cookies_per_cycle) = 40 :=
sorry

end NUMINAMATH_GPT_jenicek_decorated_cookies_total_time_for_work_jenicek_decorating_time_l493_49375


namespace NUMINAMATH_GPT_minimum_elapsed_time_l493_49346

theorem minimum_elapsed_time : 
  let initial_time := 45  -- in minutes
  let final_time := 3 * 60 + 30  -- 3 hours 30 minutes in minutes
  let elapsed_time := final_time - initial_time
  elapsed_time = 2 * 60 + 45 :=
by
  sorry

end NUMINAMATH_GPT_minimum_elapsed_time_l493_49346


namespace NUMINAMATH_GPT_roots_of_polynomial_l493_49313

theorem roots_of_polynomial : 
  ∀ (x : ℝ), (x^2 + 4) * (x^2 - 4) = 0 ↔ (x = -2 ∨ x = 2) :=
by 
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_l493_49313


namespace NUMINAMATH_GPT_range_of_m_l493_49342

noncomputable def satisfies_inequality (m : ℝ) : Prop :=
  ∀ x : ℝ, |x + 3| - |x - 1| ≤ m^2 - 3 * m

theorem range_of_m (m : ℝ) : 
  satisfies_inequality m ↔ (m ≥ 4 ∨ m ≤ -1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l493_49342


namespace NUMINAMATH_GPT_find_arith_seq_sum_l493_49377

noncomputable def arith_seq_sum : ℕ → ℕ → ℕ
| 0, d => 2
| (n+1), d => arith_seq_sum n d + d

theorem find_arith_seq_sum :
  ∃ d : ℕ, 
    arith_seq_sum 1 d + arith_seq_sum 2 d = 13 ∧
    arith_seq_sum 3 d + arith_seq_sum 4 d + arith_seq_sum 5 d = 42 :=
by
  sorry

end NUMINAMATH_GPT_find_arith_seq_sum_l493_49377


namespace NUMINAMATH_GPT_no_solution_for_k_eq_six_l493_49317

theorem no_solution_for_k_eq_six :
  ∀ x k : ℝ, k = 6 → (x ≠ 2 ∧ x ≠ 7) → (x - 1) / (x - 2) = (x - k) / (x - 7) → false :=
by 
  intros x k hk hnx_eq h_eq
  sorry

end NUMINAMATH_GPT_no_solution_for_k_eq_six_l493_49317


namespace NUMINAMATH_GPT_incorrect_statement_D_l493_49351

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2

theorem incorrect_statement_D :
  (∃ T : ℝ, ∀ x : ℝ, f (x + T) = f x) ∧
  (∀ x : ℝ, f (π / 2 + x) = f (π / 2 - x)) ∧
  (f (π / 2 + π / 4) = 0) ∧ ¬(∀ x : ℝ, (π / 2 < x ∧ x < π) → f x < f (x - 0.1)) := by
  sorry

end NUMINAMATH_GPT_incorrect_statement_D_l493_49351


namespace NUMINAMATH_GPT_sum_first_40_terms_l493_49398

-- Defining the sequence a_n following the given conditions
noncomputable def a : ℕ → ℝ
| 0 => 1
| 1 => 3
| n + 2 => a (n + 1) * a (n - 1)

-- Defining the sum of the first 40 terms of the sequence
noncomputable def S40 := (Finset.range 40).sum a

-- The theorem stating the desired property
theorem sum_first_40_terms : S40 = 60 :=
sorry

end NUMINAMATH_GPT_sum_first_40_terms_l493_49398


namespace NUMINAMATH_GPT_intersection_A_B_l493_49344

def A : Set ℝ := { x | Real.log x > 0 }

def B : Set ℝ := { x | Real.exp x < 3 }

theorem intersection_A_B :
  A ∩ B = { x | 1 < x ∧ x < Real.log 3 / Real.log 2 } :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l493_49344


namespace NUMINAMATH_GPT_tank_height_l493_49311

theorem tank_height
  (r_A r_B h_A h_B : ℝ)
  (h₁ : 8 = 2 * Real.pi * r_A)
  (h₂ : h_B = 8)
  (h₃ : 10 = 2 * Real.pi * r_B)
  (h₄ : π * r_A ^ 2 * h_A = 0.56 * (π * r_B ^ 2 * h_B)) :
  h_A = 7 :=
sorry

end NUMINAMATH_GPT_tank_height_l493_49311


namespace NUMINAMATH_GPT_limo_cost_is_correct_l493_49335

def prom_tickets_cost : ℕ := 2 * 100
def dinner_cost : ℕ := 120
def dinner_tip : ℕ := (30 * dinner_cost) / 100
def total_cost_before_limo : ℕ := prom_tickets_cost + dinner_cost + dinner_tip
def total_cost : ℕ := 836
def limo_hours : ℕ := 6
def limo_total_cost : ℕ := total_cost - total_cost_before_limo
def limo_cost_per_hour : ℕ := limo_total_cost / limo_hours

theorem limo_cost_is_correct : limo_cost_per_hour = 80 := 
by
  sorry

end NUMINAMATH_GPT_limo_cost_is_correct_l493_49335


namespace NUMINAMATH_GPT_solution_set_of_inequality_l493_49312

theorem solution_set_of_inequality (x : ℝ) : (x * (2 - x) ≤ 0) ↔ (x ≤ 0 ∨ x ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l493_49312


namespace NUMINAMATH_GPT_simplify_complex_subtraction_l493_49343

-- Definition of the nested expression
def complex_subtraction (x : ℝ) : ℝ :=
  1 - (2 - (3 - (4 - (5 - (6 - x)))))

-- Statement of the theorem to be proven
theorem simplify_complex_subtraction (x : ℝ) : complex_subtraction x = x - 3 :=
by {
  -- This proof will need to be filled in to verify the statement
  sorry
}

end NUMINAMATH_GPT_simplify_complex_subtraction_l493_49343


namespace NUMINAMATH_GPT_board_partition_possible_l493_49380

-- Definition of natural numbers m and n greater than 15
variables (m n : ℕ)
-- m > 15
def m_greater_than_15 := m > 15
-- n > 15
def n_greater_than_15 := n > 15

-- Definition of m and n divisibility conditions
def divisible_by_4_or_5 (x : ℕ) : Prop :=
  x % 4 = 0 ∨ x % 5 = 0

def partition_possible (m n : ℕ) : Prop :=
  (m % 4 = 0 ∧ n % 5 = 0) ∨ (m % 5 = 0 ∧ n % 4 = 0)

-- The final statement of Lean
theorem board_partition_possible :
  m_greater_than_15 m → n_greater_than_15 n → partition_possible m n :=
by
  intro h_m h_n
  sorry

end NUMINAMATH_GPT_board_partition_possible_l493_49380


namespace NUMINAMATH_GPT_max_value_of_expression_l493_49354

theorem max_value_of_expression (x y : ℝ) (h : 3 * x^2 + y^2 ≤ 3) : 2 * x + 3 * y ≤ Real.sqrt 31 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l493_49354


namespace NUMINAMATH_GPT_number_of_figures_l493_49345

theorem number_of_figures (num_squares num_rectangles : ℕ) 
  (h1 : 8 * 8 / 4 = num_squares + num_rectangles) 
  (h2 : 2 * 54 + 4 * 8 = 8 * num_squares + 10 * num_rectangles) :
  num_squares = 10 ∧ num_rectangles = 6 :=
sorry

end NUMINAMATH_GPT_number_of_figures_l493_49345


namespace NUMINAMATH_GPT_tom_shirts_total_cost_l493_49389

theorem tom_shirts_total_cost 
  (num_tshirts_per_fandom : ℕ)
  (num_fandoms : ℕ)
  (cost_per_shirt : ℕ)
  (discount_rate : ℚ)
  (tax_rate : ℚ)
  (total_shirts : ℕ := num_tshirts_per_fandom * num_fandoms)
  (discount_per_shirt : ℚ := (cost_per_shirt : ℚ) * discount_rate)
  (cost_per_shirt_after_discount : ℚ := (cost_per_shirt : ℚ) - discount_per_shirt)
  (total_cost_before_tax : ℚ := (total_shirts * cost_per_shirt_after_discount))
  (tax_added : ℚ := total_cost_before_tax * tax_rate)
  (total_amount_paid : ℚ := total_cost_before_tax + tax_added)
  (h1 : num_tshirts_per_fandom = 5)
  (h2 : num_fandoms = 4)
  (h3 : cost_per_shirt = 15) 
  (h4 : discount_rate = 0.2)
  (h5 : tax_rate = 0.1)
  : total_amount_paid = 264 := 
by 
  sorry

end NUMINAMATH_GPT_tom_shirts_total_cost_l493_49389


namespace NUMINAMATH_GPT_smallest_positive_multiple_of_37_l493_49332

theorem smallest_positive_multiple_of_37 (a : ℕ) (h1 : 37 * a ≡ 3 [MOD 101]) (h2 : ∀ b : ℕ, 0 < b ∧ (37 * b ≡ 3 [MOD 101]) → a ≤ b) : 37 * a = 1628 :=
sorry

end NUMINAMATH_GPT_smallest_positive_multiple_of_37_l493_49332


namespace NUMINAMATH_GPT_min_unit_cubes_intersect_all_l493_49376

theorem min_unit_cubes_intersect_all (n : ℕ) : 
  let A_n := if n % 2 = 0 then n^2 / 2 else (n^2 + 1) / 2
  A_n = if n % 2 = 0 then n^2 / 2 else (n^2 + 1) / 2 :=
sorry

end NUMINAMATH_GPT_min_unit_cubes_intersect_all_l493_49376


namespace NUMINAMATH_GPT_correct_choice_d_l493_49333

def is_quadrant_angle (alpha : ℝ) (k : ℤ) : Prop :=
  2 * k * Real.pi - Real.pi / 2 < alpha ∧ alpha < 2 * k * Real.pi

theorem correct_choice_d (alpha : ℝ) (k : ℤ) :
  is_quadrant_angle alpha k ↔ (2 * k * Real.pi - Real.pi / 2 < alpha ∧ alpha < 2 * k * Real.pi) := by
sorry

end NUMINAMATH_GPT_correct_choice_d_l493_49333


namespace NUMINAMATH_GPT_problem_statement_l493_49340

theorem problem_statement (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  (x^3 + 3 * y^2) / 7 = 75 / 7 :=
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_problem_statement_l493_49340


namespace NUMINAMATH_GPT_find_cost_price_l493_49319

-- Given conditions
variables (CP SP1 SP2 : ℝ)
def condition1 : Prop := SP1 = 0.90 * CP
def condition2 : Prop := SP2 = 1.10 * CP
def condition3 : Prop := SP2 - SP1 = 500

-- Prove that CP is 2500 
theorem find_cost_price 
  (CP SP1 SP2 : ℝ)
  (h1 : condition1 CP SP1)
  (h2 : condition2 CP SP2)
  (h3 : condition3 SP1 SP2) : 
  CP = 2500 :=
sorry -- proof not required

end NUMINAMATH_GPT_find_cost_price_l493_49319


namespace NUMINAMATH_GPT_Vanya_Journey_Five_times_Anya_Journey_l493_49359

theorem Vanya_Journey_Five_times_Anya_Journey (a_start a_end v_start v_end : ℕ)
  (h1 : a_start = 1) (h2 : a_end = 2) (h3 : v_start = 1) (h4 : v_end = 6) :
  (v_end - v_start) = 5 * (a_end - a_start) :=
  sorry

end NUMINAMATH_GPT_Vanya_Journey_Five_times_Anya_Journey_l493_49359


namespace NUMINAMATH_GPT_roots_of_quadratic_solve_inequality_l493_49341

theorem roots_of_quadratic (a b : ℝ) (h1 : ∀ x : ℝ, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :
  a = 1 ∧ b = 2 :=
by
  sorry

theorem solve_inequality (c : ℝ) :
  let a := 1
  let b := 2
  ∀ x : ℝ, a * x^2 - (a * c + b) * x + b * x < 0 ↔
    (c > 0 → (0 < x ∧ x < c)) ∧
    (c = 0 → false) ∧
    (c < 0 → (c < x ∧ x < 0)) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_solve_inequality_l493_49341
