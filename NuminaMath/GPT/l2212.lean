import Mathlib

namespace NUMINAMATH_GPT_total_pay_of_two_employees_l2212_221208

theorem total_pay_of_two_employees
  (Y_pay : ℝ)
  (X_pay : ℝ)
  (h1 : Y_pay = 280)
  (h2 : X_pay = 1.2 * Y_pay) :
  X_pay + Y_pay = 616 :=
by
  sorry

end NUMINAMATH_GPT_total_pay_of_two_employees_l2212_221208


namespace NUMINAMATH_GPT_number_of_TVs_in_shop_c_l2212_221222

theorem number_of_TVs_in_shop_c 
  (a b d e : ℕ) 
  (avg : ℕ) 
  (num_shops : ℕ) 
  (total_TVs_in_other_shops : ℕ) 
  (total_TVs : ℕ) 
  (sum_shops : a + b + d + e = total_TVs_in_other_shops) 
  (avg_sets : avg = total_TVs / num_shops) 
  (number_shops : num_shops = 5)
  (avg_value : avg = 48)
  (T_a : a = 20) 
  (T_b : b = 30) 
  (T_d : d = 80) 
  (T_e : e = 50) 
  : (total_TVs - total_TVs_in_other_shops = 60) := 
by 
  sorry

end NUMINAMATH_GPT_number_of_TVs_in_shop_c_l2212_221222


namespace NUMINAMATH_GPT_smaller_acute_angle_l2212_221228

theorem smaller_acute_angle (x : ℝ) (h : 5 * x + 4 * x = 90) : 4 * x = 40 :=
by 
  -- proof steps can be added here, but are omitted as per the instructions
  sorry

end NUMINAMATH_GPT_smaller_acute_angle_l2212_221228


namespace NUMINAMATH_GPT_farmer_total_acres_l2212_221273

theorem farmer_total_acres (x : ℕ) (H1 : 4 * x = 376) : 
  5 * x + 2 * x + 4 * x = 1034 :=
by
  -- This placeholder is indicating unfinished proof
  sorry

end NUMINAMATH_GPT_farmer_total_acres_l2212_221273


namespace NUMINAMATH_GPT_round_trip_time_l2212_221200

def boat_speed := 9 -- speed of the boat in standing water (kmph)
def stream_speed := 6 -- speed of the stream (kmph)
def distance := 210 -- distance to the place (km)

def upstream_speed := boat_speed - stream_speed
def downstream_speed := boat_speed + stream_speed

def time_upstream := distance / upstream_speed
def time_downstream := distance / downstream_speed
def total_time := time_upstream + time_downstream

theorem round_trip_time : total_time = 84 := by
  sorry

end NUMINAMATH_GPT_round_trip_time_l2212_221200


namespace NUMINAMATH_GPT_hash_nesting_example_l2212_221204

def hash (N : ℝ) : ℝ :=
  0.5 * N + 2

theorem hash_nesting_example : hash (hash (hash (hash 20))) = 5 :=
by
  sorry

end NUMINAMATH_GPT_hash_nesting_example_l2212_221204


namespace NUMINAMATH_GPT_num_perfect_square_factors_1800_l2212_221245

theorem num_perfect_square_factors_1800 :
  let factors_1800 := [(2, 3), (3, 2), (5, 2)]
  ∃ n : ℕ, (n = 8) ∧
           (∀ p_k ∈ factors_1800, ∃ (e : ℕ), (e = 0 ∨ e = 2) ∧ n = 2 * 2 * 2 → n = 8) :=
sorry

end NUMINAMATH_GPT_num_perfect_square_factors_1800_l2212_221245


namespace NUMINAMATH_GPT_garden_perimeter_l2212_221251

-- formally defining the conditions of the problem
variables (x y : ℝ)
def diagonal_of_garden : Prop := x^2 + y^2 = 900
def area_of_garden : Prop := x * y = 216

-- final statement to prove the perimeter of the garden
theorem garden_perimeter (h1 : diagonal_of_garden x y) (h2 : area_of_garden x y) : 2 * (x + y) = 73 := sorry

end NUMINAMATH_GPT_garden_perimeter_l2212_221251


namespace NUMINAMATH_GPT_problem_l2212_221250

noncomputable def f (x : ℝ) : ℝ := (1 / x) * Real.cos x

noncomputable def f_deriv (x : ℝ) : ℝ := - (1 / x^2) * Real.cos x - (1 / x) * Real.sin x

theorem problem (h_pi_ne_zero : Real.pi ≠ 0) (h_pi_div_two_ne_zero : Real.pi / 2 ≠ 0) :
  f Real.pi + f_deriv (Real.pi / 2) = -3 / Real.pi  := by
  sorry

end NUMINAMATH_GPT_problem_l2212_221250


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l2212_221239

theorem hyperbola_eccentricity (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (e : ℝ) (h3 : e = (Real.sqrt 3) / 2) 
  (h4 : a ^ 2 = b ^ 2 + (Real.sqrt 3) ^ 2) : (Real.sqrt 5) / 2 = 
    (Real.sqrt (a ^ 2 + b ^ 2)) / a :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l2212_221239


namespace NUMINAMATH_GPT_solve_for_M_l2212_221212

theorem solve_for_M (a b M : ℝ) (h : (a + 2 * b) ^ 2 = (a - 2 * b) ^ 2 + M) : M = 8 * a * b :=
by sorry

end NUMINAMATH_GPT_solve_for_M_l2212_221212


namespace NUMINAMATH_GPT_simplify_expression_l2212_221213

theorem simplify_expression (a: ℤ) (h₁: a ≠ 0) (h₂: a ≠ 1) (h₃: a ≠ -3) :
  (2 * a = 4) → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2212_221213


namespace NUMINAMATH_GPT_probability_of_problem_being_solved_l2212_221274

-- Define the probabilities of solving the problem.
def prob_A_solves : ℚ := 1 / 5
def prob_B_solves : ℚ := 1 / 3

-- Define the proof statement
theorem probability_of_problem_being_solved :
  (1 - ((1 - prob_A_solves) * (1 - prob_B_solves))) = 7 / 15 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_problem_being_solved_l2212_221274


namespace NUMINAMATH_GPT_discount_percentage_l2212_221279

theorem discount_percentage (p : ℝ) : 
  (1 + 0.25) * p * (1 - 0.20) = p :=
by
  sorry

end NUMINAMATH_GPT_discount_percentage_l2212_221279


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_general_term_l2212_221243

theorem arithmetic_geometric_sequence_general_term :
  ∃ q a1 : ℕ, (∀ n : ℕ, a2 = 6 ∧ 6 * a1 + a3 = 30) →
  (∀ n : ℕ, (q = 2 ∧ a1 = 3 → a_n = 3 * 3^(n-1)) ∨ (q = 3 ∧ a1 = 2 → a_n = 2 * 2^(n-1))) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_general_term_l2212_221243


namespace NUMINAMATH_GPT_Carolina_Winning_Probability_Beto_Winning_Probability_Ana_Winning_Probability_l2212_221206

section
  -- Define the types of participants and the colors
  inductive Participant
  | Ana | Beto | Carolina

  inductive Color
  | blue | green

  -- Define the strategies for each participant
  inductive Strategy
  | guessBlue | guessGreen | pass

  -- Probability calculations for each strategy
  def carolinaStrategyProbability : ℚ := 1 / 8
  def betoStrategyProbability : ℚ := 1 / 2
  def anaStrategyProbability : ℚ := 3 / 4

  -- Statements to prove the probabilities
  theorem Carolina_Winning_Probability :
    carolinaStrategyProbability = 1 / 8 :=
  sorry

  theorem Beto_Winning_Probability :
    betoStrategyProbability = 1 / 2 :=
  sorry

  theorem Ana_Winning_Probability :
    anaStrategyProbability = 3 / 4 :=
  sorry
end

end NUMINAMATH_GPT_Carolina_Winning_Probability_Beto_Winning_Probability_Ana_Winning_Probability_l2212_221206


namespace NUMINAMATH_GPT_max_product_ge_993_squared_l2212_221209

theorem max_product_ge_993_squared (a : Fin 1985 → Fin 1985) (hperm : ∀ n : Fin 1985, ∃ k : Fin 1985, a k = n ∧ ∃ m : Fin 1985, a m = n) :
  ∃ k : Fin 1985, a k * k ≥ 993^2 :=
sorry

end NUMINAMATH_GPT_max_product_ge_993_squared_l2212_221209


namespace NUMINAMATH_GPT_yellow_beads_needed_l2212_221255

variable (Total green yellow : ℕ)

theorem yellow_beads_needed (h_green : green = 4) (h_yellow : yellow = 0) (h_fraction : (4 / 5 : ℚ) = 4 / (green + yellow + 16)) :
    4 + 16 + green = Total := by
  sorry

end NUMINAMATH_GPT_yellow_beads_needed_l2212_221255


namespace NUMINAMATH_GPT_girls_to_boys_ratio_l2212_221275

variable (g b : ℕ)
variable (h_total : g + b = 36)
variable (h_diff : g = b + 6)

theorem girls_to_boys_ratio (g b : ℕ) (h_total : g + b = 36) (h_diff : g = b + 6) :
  g / b = 7 / 5 := by
  sorry

end NUMINAMATH_GPT_girls_to_boys_ratio_l2212_221275


namespace NUMINAMATH_GPT_geometric_series_sixth_term_l2212_221244

theorem geometric_series_sixth_term :
  ∃ r : ℝ, r > 0 ∧ (16 * r^7 = 11664) ∧ (16 * r^5 = 3888) :=
by 
  sorry

end NUMINAMATH_GPT_geometric_series_sixth_term_l2212_221244


namespace NUMINAMATH_GPT_oldest_son_cookies_l2212_221218

def youngest_son_cookies : Nat := 2
def total_cookies : Nat := 54
def days : Nat := 9

theorem oldest_son_cookies : ∃ x : Nat, 9 * (x + youngest_son_cookies) = total_cookies ∧ x = 4 := by
  sorry

end NUMINAMATH_GPT_oldest_son_cookies_l2212_221218


namespace NUMINAMATH_GPT_smallest_option_l2212_221267

-- Define the problem with the given condition
def x : ℕ := 10

-- Define all the options in the problem
def option_a := 6 / x
def option_b := 6 / (x + 1)
def option_c := 6 / (x - 1)
def option_d := x / 6
def option_e := (x + 1) / 6
def option_f := (x - 2) / 6

-- The proof problem statement to show that option_b is the smallest
theorem smallest_option :
  option_b < option_a ∧ option_b < option_c ∧ option_b < option_d ∧ option_b < option_e ∧ option_b < option_f :=
by
  sorry

end NUMINAMATH_GPT_smallest_option_l2212_221267


namespace NUMINAMATH_GPT_line_circle_intersection_l2212_221248

theorem line_circle_intersection (a : ℝ) : 
  (∀ x y : ℝ, (4 * x + 3 * y + a = 0) → ((x - 1)^2 + (y - 2)^2 = 9)) ∧
  (∃ A B : ℝ, dist A B = 4 * Real.sqrt 2) →
  (a = -5 ∨ a = -15) :=
by 
  sorry

end NUMINAMATH_GPT_line_circle_intersection_l2212_221248


namespace NUMINAMATH_GPT_smallest_digit_to_correct_l2212_221293

def incorrect_sum : ℕ := 2104
def correct_sum : ℕ := 738 + 625 + 841
def difference : ℕ := correct_sum - incorrect_sum

theorem smallest_digit_to_correct (d : ℕ) (h : difference = 100) :
  d = 6 := 
sorry

end NUMINAMATH_GPT_smallest_digit_to_correct_l2212_221293


namespace NUMINAMATH_GPT_find_natural_numbers_l2212_221214

theorem find_natural_numbers (a b : ℕ) (p : ℕ) (hp : Nat.Prime p)
  (h : a^3 - b^3 = 633 * p) : a = 16 ∧ b = 13 :=
by
  sorry

end NUMINAMATH_GPT_find_natural_numbers_l2212_221214


namespace NUMINAMATH_GPT_min_value_of_fraction_l2212_221220

theorem min_value_of_fraction (n : ℕ) (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  (1 / (1 + a^n) + 1 / (1 + b^n)) = 1 :=
sorry

end NUMINAMATH_GPT_min_value_of_fraction_l2212_221220


namespace NUMINAMATH_GPT_average_temperature_for_july_4th_l2212_221252

def avg_temperature_july_4th : ℤ := 
  let temperatures := [90, 90, 90, 79, 71]
  let sum := List.sum temperatures
  sum / temperatures.length

theorem average_temperature_for_july_4th :
  avg_temperature_july_4th = 84 := 
by
  sorry

end NUMINAMATH_GPT_average_temperature_for_july_4th_l2212_221252


namespace NUMINAMATH_GPT_prime_cube_difference_l2212_221278

theorem prime_cube_difference (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (eqn : p^3 - q^3 = 11 * r) : 
  (p = 13 ∧ q = 2 ∧ r = 199) :=
sorry

end NUMINAMATH_GPT_prime_cube_difference_l2212_221278


namespace NUMINAMATH_GPT_candidates_appeared_l2212_221216

-- Define the number of appeared candidates in state A and state B
variables (X : ℝ)

-- The conditions given in the problem
def condition1 : Prop := (0.07 * X = 0.06 * X + 83)

-- The claim that needs to be proved
def claim : Prop := (X = 8300)

-- The theorem statement in Lean 4
theorem candidates_appeared (X : ℝ) (h1 : condition1 X) : claim X := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_candidates_appeared_l2212_221216


namespace NUMINAMATH_GPT_ratio_of_investments_l2212_221240

-- Define the conditions
def ratio_of_profits (p q : ℝ) : Prop := 7/12 = (p * 5) / (q * 12)

-- Define the problem: given the conditions, prove the ratio of investments is 7/5
theorem ratio_of_investments (P Q : ℝ) (h : ratio_of_profits P Q) : P / Q = 7 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_investments_l2212_221240


namespace NUMINAMATH_GPT_even_numbers_with_specific_square_properties_l2212_221229

theorem even_numbers_with_specific_square_properties (n : ℕ) :
  (10^13 ≤ n^2 ∧ n^2 < 10^14 ∧ (n^2 % 100) / 10 = 5) → 
  (2 ∣ n ∧ 273512 > 10^5) := 
sorry

end NUMINAMATH_GPT_even_numbers_with_specific_square_properties_l2212_221229


namespace NUMINAMATH_GPT_polynomial_non_negative_for_all_real_iff_l2212_221268

theorem polynomial_non_negative_for_all_real_iff (a : ℝ) :
  (∀ x : ℝ, x^4 + (a - 1) * x^2 + 1 ≥ 0) ↔ a ≥ -1 :=
by sorry

end NUMINAMATH_GPT_polynomial_non_negative_for_all_real_iff_l2212_221268


namespace NUMINAMATH_GPT_contrapositive_true_l2212_221227

theorem contrapositive_true (x : ℝ) : (x^2 - 2*x - 8 ≤ 0 → x ≥ -3) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_contrapositive_true_l2212_221227


namespace NUMINAMATH_GPT_find_f_of_2_l2212_221221

-- Definitions based on problem conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def g (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f (x) + 9

-- The main statement to proof that f(2) = 6 under the given conditions
theorem find_f_of_2 (f : ℝ → ℝ)
  (hf : is_odd_function f)
  (hg : ∀ x, g f x = f x + 9)
  (h : g f (-2) = 3) :
  f 2 = 6 := 
sorry

end NUMINAMATH_GPT_find_f_of_2_l2212_221221


namespace NUMINAMATH_GPT_vasya_days_without_purchase_l2212_221237

theorem vasya_days_without_purchase
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) :
  w = 7 :=
by
  sorry

end NUMINAMATH_GPT_vasya_days_without_purchase_l2212_221237


namespace NUMINAMATH_GPT_distinct_cube_arrangements_count_l2212_221258

def is_valid_face_sum (face : Finset ℕ) : Prop :=
  face.sum id = 34

def is_valid_opposite_sum (v1 v2 : ℕ) : Prop :=
  v1 + v2 = 16

def is_unique_up_to_rotation (cubes : List (Finset ℕ)) : Prop := sorry -- Define rotational uniqueness check

noncomputable def count_valid_arrangements : ℕ := sorry -- Define counting logic

theorem distinct_cube_arrangements_count : count_valid_arrangements = 3 :=
  sorry

end NUMINAMATH_GPT_distinct_cube_arrangements_count_l2212_221258


namespace NUMINAMATH_GPT_polar_curve_is_circle_l2212_221201

theorem polar_curve_is_circle (θ ρ : ℝ) (h : 4 * Real.sin θ = 5 * ρ) : 
  ∃ c : ℝ×ℝ, ∀ (x y : ℝ), x^2 + y^2 = c.1^2 + c.2^2 :=
by
  sorry

end NUMINAMATH_GPT_polar_curve_is_circle_l2212_221201


namespace NUMINAMATH_GPT_profit_functions_properties_l2212_221295

noncomputable def R (x : ℝ) : ℝ := 3000 * x - 20 * x^2
noncomputable def C (x : ℝ) : ℝ := 500 * x + 4000
noncomputable def P (x : ℝ) : ℝ := R x - C x
noncomputable def MP (x : ℝ) : ℝ := P (x + 1) - P x

theorem profit_functions_properties :
  (P x = -20 * x^2 + 2500 * x - 4000) ∧ 
  (MP x = -40 * x + 2480) ∧ 
  (∃ x_max₁, ∀ x, P x_max₁ ≥ P x) ∧ 
  (∃ x_max₂, ∀ x, MP x_max₂ ≥ MP x) ∧ 
  P x_max₁ ≠ MP x_max₂ := by
  sorry

end NUMINAMATH_GPT_profit_functions_properties_l2212_221295


namespace NUMINAMATH_GPT_total_number_of_toys_l2212_221210

theorem total_number_of_toys (average_cost_Dhoni_toys : ℕ) (number_Dhoni_toys : ℕ) 
    (price_David_toy : ℕ) (new_avg_cost : ℕ) 
    (h1 : average_cost_Dhoni_toys = 10) (h2 : number_Dhoni_toys = 5) 
    (h3 : price_David_toy = 16) (h4 : new_avg_cost = 11) : 
    (number_Dhoni_toys + 1) = 6 := 
by
  sorry

end NUMINAMATH_GPT_total_number_of_toys_l2212_221210


namespace NUMINAMATH_GPT_total_fruit_cost_is_173_l2212_221234

-- Define the cost of a single orange and a single apple
def orange_cost := 2
def apple_cost := 3
def banana_cost := 1

-- Define the number of fruits each person has
def louis_oranges := 5
def louis_apples := 3

def samantha_oranges := 8
def samantha_apples := 7

def marley_oranges := 2 * louis_oranges
def marley_apples := 3 * samantha_apples

def edward_oranges := 3 * louis_oranges
def edward_bananas := 4

-- Define the cost of fruits for each person
def louis_cost := (louis_oranges * orange_cost) + (louis_apples * apple_cost)
def samantha_cost := (samantha_oranges * orange_cost) + (samantha_apples * apple_cost)
def marley_cost := (marley_oranges * orange_cost) + (marley_apples * apple_cost)
def edward_cost := (edward_oranges * orange_cost) + (edward_bananas * banana_cost)

-- Define the total cost for all four people
def total_cost := louis_cost + samantha_cost + marley_cost + edward_cost

-- Statement to prove that the total cost is $173
theorem total_fruit_cost_is_173 : total_cost = 173 :=
by
  sorry

end NUMINAMATH_GPT_total_fruit_cost_is_173_l2212_221234


namespace NUMINAMATH_GPT_sales_quota_50_l2212_221257

theorem sales_quota_50 :
  let cars_sold_first_three_days := 5 * 3
  let cars_sold_next_four_days := 3 * 4
  let additional_cars_needed := 23
  let total_quota := cars_sold_first_three_days + cars_sold_next_four_days + additional_cars_needed
  total_quota = 50 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_sales_quota_50_l2212_221257


namespace NUMINAMATH_GPT_largest_n_unique_k_l2212_221297

theorem largest_n_unique_k :
  ∃ n : ℕ, n = 1 ∧ ∀ k : ℕ, (3 : ℚ) / 7 < (n : ℚ) / ((n + k : ℕ) : ℚ) ∧ 
  (n : ℚ) / ((n + k : ℕ) : ℚ) < (8 : ℚ) / 19 → k = 1 := by
sorry

end NUMINAMATH_GPT_largest_n_unique_k_l2212_221297


namespace NUMINAMATH_GPT_tan_addition_formula_15_30_l2212_221254

-- Define tangent function for angles in degrees.
noncomputable def tanDeg (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

-- State the theorem for the given problem
theorem tan_addition_formula_15_30 :
  tanDeg 15 + tanDeg 30 + tanDeg 15 * tanDeg 30 = 1 :=
by
  -- Here we use the given conditions and properties in solution
  sorry

end NUMINAMATH_GPT_tan_addition_formula_15_30_l2212_221254


namespace NUMINAMATH_GPT_red_blue_pencil_difference_l2212_221219

theorem red_blue_pencil_difference :
  let total_pencils := 36
  let red_fraction := 5 / 9
  let blue_fraction := 5 / 12
  let red_pencils := red_fraction * total_pencils
  let blue_pencils := blue_fraction * total_pencils
  red_pencils - blue_pencils = 5 :=
by
  -- placeholder proof
  sorry

end NUMINAMATH_GPT_red_blue_pencil_difference_l2212_221219


namespace NUMINAMATH_GPT_only_set_d_forms_triangle_l2212_221253

/-- Definition of forming a triangle given three lengths -/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem only_set_d_forms_triangle :
  ¬ can_form_triangle 3 5 10 ∧ ¬ can_form_triangle 5 4 9 ∧ 
  ¬ can_form_triangle 5 5 10 ∧ can_form_triangle 4 6 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_only_set_d_forms_triangle_l2212_221253


namespace NUMINAMATH_GPT_boat_distance_against_water_flow_l2212_221299

variable (a : ℝ) -- speed of the boat in still water

theorem boat_distance_against_water_flow 
  (speed_boat_still_water : ℝ := a)
  (speed_water_flow : ℝ := 3)
  (time_travel : ℝ := 3) :
  (speed_boat_still_water - speed_water_flow) * time_travel = 3 * (a - 3) := 
by
  sorry

end NUMINAMATH_GPT_boat_distance_against_water_flow_l2212_221299


namespace NUMINAMATH_GPT_lattice_point_exists_l2212_221235

noncomputable def exists_distant_lattice_point : Prop :=
∃ (X Y : ℤ), ∀ (x y : ℤ), gcd x y = 1 → (X - x) ^ 2 + (Y - y) ^ 2 ≥ 1995 ^ 2

theorem lattice_point_exists : exists_distant_lattice_point :=
sorry

end NUMINAMATH_GPT_lattice_point_exists_l2212_221235


namespace NUMINAMATH_GPT_solution_m_plus_n_l2212_221291

variable (m n : ℝ)

theorem solution_m_plus_n 
  (h₁ : m ≠ 0)
  (h₂ : m^2 + m * n - m = 0) :
  m + n = 1 := by
  sorry

end NUMINAMATH_GPT_solution_m_plus_n_l2212_221291


namespace NUMINAMATH_GPT_intersecting_lines_sum_l2212_221276

theorem intersecting_lines_sum (a b : ℝ) 
  (h1 : a * 1 + 1 + 1 = 0)
  (h2 : 2 * 1 - b * 1 - 1 = 0) : 
  a + b = -1 := 
by 
  have ha : a = -2 := by linarith [h1]
  have hb : b = 1 := by linarith [h2]
  rw [ha, hb]
  exact by norm_num

end NUMINAMATH_GPT_intersecting_lines_sum_l2212_221276


namespace NUMINAMATH_GPT_solve_for_three_times_x_plus_ten_l2212_221294

theorem solve_for_three_times_x_plus_ten (x : ℝ) (h_eq : 5 * x - 7 = 15 * x + 21) : 3 * (x + 10) = 21.6 := by
  sorry

end NUMINAMATH_GPT_solve_for_three_times_x_plus_ten_l2212_221294


namespace NUMINAMATH_GPT_evaluate_expression_l2212_221203

variable {a b c : ℝ}

theorem evaluate_expression
  (h : a / (35 - a) + b / (75 - b) + c / (85 - c) = 5) :
  7 / (35 - a) + 15 / (75 - b) + 17 / (85 - c) = 8 / 5 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2212_221203


namespace NUMINAMATH_GPT_solve_for_y_l2212_221247

theorem solve_for_y (y : ℝ) (h : 9 / (y^2) = y / 81) : y = 9 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l2212_221247


namespace NUMINAMATH_GPT_arithmetic_sequence_product_l2212_221285

theorem arithmetic_sequence_product (a : ℕ → ℤ) (d : ℤ) (h_inc : ∀ n m, n < m → a n < a m) 
  (h_arith : ∀ n, a (n + 1) = a n + d) (h_prod : a 4 * a 5 = 12) : a 2 * a 7 = 6 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_product_l2212_221285


namespace NUMINAMATH_GPT_skittles_transfer_l2212_221263

-- Define the initial number of Skittles Bridget and Henry have
def bridget_initial_skittles := 4
def henry_initial_skittles := 4

-- The main statement we want to prove
theorem skittles_transfer :
  bridget_initial_skittles + henry_initial_skittles = 8 :=
by
  sorry

end NUMINAMATH_GPT_skittles_transfer_l2212_221263


namespace NUMINAMATH_GPT_find_slope_of_line_l_l2212_221281

theorem find_slope_of_line_l :
  ∃ k : ℝ, (k = 3 * Real.sqrt 5 / 10 ∨ k = -3 * Real.sqrt 5 / 10) :=
by
  -- Given conditions
  let F1 : ℝ := 6 / 5 * Real.sqrt 5
  let PF : ℝ := 4 / 5 * Real.sqrt 5
  let slope_PQ : ℝ := 1
  let slope_RF1 : ℝ := sorry  -- we need to prove/extract this from the given
  let k := 3 / 2 * slope_RF1
  -- to prove this
  sorry

end NUMINAMATH_GPT_find_slope_of_line_l_l2212_221281


namespace NUMINAMATH_GPT_height_of_spherical_cap_case1_height_of_spherical_cap_case2_l2212_221266

variable (R : ℝ) (c : ℝ)
variable (h_c_gt_1 : c > 1)

-- Case 1: Not including the circular cap in the surface area
theorem height_of_spherical_cap_case1 : ∃ m : ℝ, m = (2 * R * (c - 1)) / c :=
by
  sorry

-- Case 2: Including the circular cap in the surface area
theorem height_of_spherical_cap_case2 : ∃ m : ℝ, m = (2 * R * (c - 2)) / (c - 1) :=
by
  sorry

end NUMINAMATH_GPT_height_of_spherical_cap_case1_height_of_spherical_cap_case2_l2212_221266


namespace NUMINAMATH_GPT_bookstore_price_change_l2212_221231

theorem bookstore_price_change (P : ℝ) (x : ℝ) (h : P > 0) : 
  (P * (1 + x / 100) * (1 - x / 100)) = 0.75 * P → x = 50 :=
by
  sorry

end NUMINAMATH_GPT_bookstore_price_change_l2212_221231


namespace NUMINAMATH_GPT_tax_deduction_is_correct_l2212_221261

-- Define the hourly wage and tax rate
def hourly_wage_dollars : ℝ := 25
def tax_rate : ℝ := 0.021

-- Define the conversion from dollars to cents
def dollars_to_cents (dollars : ℝ) : ℝ := dollars * 100

-- Calculate the hourly wage in cents
def hourly_wage_cents : ℝ := dollars_to_cents hourly_wage_dollars

-- Calculate the tax deducted in cents per hour
def tax_deduction_cents (wage : ℝ) (rate : ℝ) : ℝ := rate * wage

-- State the theorem that needs to be proven
theorem tax_deduction_is_correct :
  tax_deduction_cents hourly_wage_cents tax_rate = 52.5 :=
by
  sorry

end NUMINAMATH_GPT_tax_deduction_is_correct_l2212_221261


namespace NUMINAMATH_GPT_patty_weeks_without_chores_correct_l2212_221215

noncomputable def patty_weeks_without_chores : ℕ := by
  let cookie_per_chore := 3
  let chores_per_week_per_sibling := 4
  let siblings := 2
  let dollars := 15
  let cookie_pack_size := 24
  let cookie_pack_cost := 3

  let packs := dollars / cookie_pack_cost
  let total_cookies := packs * cookie_pack_size
  let weekly_cookies_needed := chores_per_week_per_sibling * cookie_per_chore * siblings

  exact total_cookies / weekly_cookies_needed

theorem patty_weeks_without_chores_correct : patty_weeks_without_chores = 5 := sorry

end NUMINAMATH_GPT_patty_weeks_without_chores_correct_l2212_221215


namespace NUMINAMATH_GPT_find_xyz_l2212_221233

theorem find_xyz (x y z : ℝ) (h₁ : x + 1 / y = 5) (h₂ : y + 1 / z = 2) (h₃ : z + 2 / x = 10 / 3) : x * y * z = (21 + Real.sqrt 433) / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_xyz_l2212_221233


namespace NUMINAMATH_GPT_proof_C_I_M_cap_N_l2212_221238

open Set

variable {𝕜 : Type _} [LinearOrderedField 𝕜]

def I : Set 𝕜 := Set.univ
def M : Set 𝕜 := {x : 𝕜 | -2 ≤ x ∧ x ≤ 2}
def N : Set 𝕜 := {x : 𝕜 | x < 1}
def C_I (A : Set 𝕜) : Set 𝕜 := I \ A

theorem proof_C_I_M_cap_N :
  C_I M ∩ N = {x : 𝕜 | x < -2} := by
  sorry

end NUMINAMATH_GPT_proof_C_I_M_cap_N_l2212_221238


namespace NUMINAMATH_GPT_james_hours_per_year_l2212_221290

def hours_per_day (trainings_per_day : Nat) (hours_per_training : Nat) : Nat :=
  trainings_per_day * hours_per_training

def days_per_week (total_days : Nat) (rest_days : Nat) : Nat :=
  total_days - rest_days

def hours_per_week (hours_day : Nat) (days_week : Nat) : Nat :=
  hours_day * days_week

def hours_per_year (hours_week : Nat) (weeks_year : Nat) : Nat :=
  hours_week * weeks_year

theorem james_hours_per_year :
  let trainings_per_day := 2
  let hours_per_training := 4
  let total_days_per_week := 7
  let rest_days_per_week := 2
  let weeks_per_year := 52
  hours_per_year 
    (hours_per_week 
      (hours_per_day trainings_per_day hours_per_training) 
      (days_per_week total_days_per_week rest_days_per_week)
    ) weeks_per_year
  = 2080 := by
  sorry

end NUMINAMATH_GPT_james_hours_per_year_l2212_221290


namespace NUMINAMATH_GPT_larger_exceeds_smaller_by_5_l2212_221284

-- Define the problem's parameters and conditions.
variables (x n m : ℕ)
variables (subtracted : ℕ := 5)

-- Define the two numbers based on the given ratio.
def larger_number := 6 * x
def smaller_number := 5 * x

-- Condition when a number is subtracted
def new_ratio_condition := (larger_number - subtracted) * 4 = (smaller_number - subtracted) * 5

-- The main goal
theorem larger_exceeds_smaller_by_5 (hx : new_ratio_condition) : larger_number - smaller_number = 5 :=
sorry

end NUMINAMATH_GPT_larger_exceeds_smaller_by_5_l2212_221284


namespace NUMINAMATH_GPT_perpendicular_lines_iff_a_eq_1_l2212_221241

theorem perpendicular_lines_iff_a_eq_1 :
  ∀ a : ℝ, (∀ x y, (y = a * x + 1) → (y = (a - 2) * x - 1) → (a = 1)) ↔ (a = 1) :=
by sorry

end NUMINAMATH_GPT_perpendicular_lines_iff_a_eq_1_l2212_221241


namespace NUMINAMATH_GPT_rainfall_second_week_january_l2212_221224

-- Define the conditions
def total_rainfall_2_weeks (rainfall_first_week rainfall_second_week : ℝ) : Prop :=
  rainfall_first_week + rainfall_second_week = 20

def rainfall_second_week_is_1_5_times_first (rainfall_first_week rainfall_second_week : ℝ) : Prop :=
  rainfall_second_week = 1.5 * rainfall_first_week

-- Define the statement to prove
theorem rainfall_second_week_january (rainfall_first_week rainfall_second_week : ℝ) :
  total_rainfall_2_weeks rainfall_first_week rainfall_second_week →
  rainfall_second_week_is_1_5_times_first rainfall_first_week rainfall_second_week →
  rainfall_second_week = 12 :=
by
  sorry

end NUMINAMATH_GPT_rainfall_second_week_january_l2212_221224


namespace NUMINAMATH_GPT_total_heads_l2212_221298

def number_of_heads := 1
def number_of_feet_hen := 2
def number_of_feet_cow := 4
def total_feet := 144

theorem total_heads (H C : ℕ) (h_hens : H = 24) (h_feet : number_of_feet_hen * H + number_of_feet_cow * C = total_feet) :
  H + C = 48 :=
sorry

end NUMINAMATH_GPT_total_heads_l2212_221298


namespace NUMINAMATH_GPT_translated_graph_pass_through_origin_l2212_221259

theorem translated_graph_pass_through_origin 
    (φ : ℝ) (h : 0 < φ ∧ φ < π / 2) 
    (passes_through_origin : 0 = Real.sin (-2 * φ + π / 3)) : 
    φ = π / 6 := 
sorry

end NUMINAMATH_GPT_translated_graph_pass_through_origin_l2212_221259


namespace NUMINAMATH_GPT_al_original_portion_l2212_221232

variables (a b c d : ℝ)

theorem al_original_portion :
  a + b + c + d = 1200 →
  a - 150 + 2 * b + 2 * c + 3 * d = 1800 →
  a = 450 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_al_original_portion_l2212_221232


namespace NUMINAMATH_GPT_no_prime_divisible_by_91_l2212_221226

theorem no_prime_divisible_by_91 : ¬ ∃ p : ℕ, p > 1 ∧ Prime p ∧ 91 ∣ p :=
by
  sorry

end NUMINAMATH_GPT_no_prime_divisible_by_91_l2212_221226


namespace NUMINAMATH_GPT_find_range_of_a_l2212_221296

-- Define the operation ⊗ on ℝ: x ⊗ y = x(1 - y)
def tensor (x y : ℝ) : ℝ := x * (1 - y)

-- State the inequality condition for all real numbers x
def inequality_condition (a : ℝ) : Prop :=
  ∀ (x : ℝ), tensor (x - a) (x + 1) < 1

-- State the theorem to prove the range of a
theorem find_range_of_a (a : ℝ) (h : inequality_condition a) : -2 < a ∧ a < 2 :=
  sorry

end NUMINAMATH_GPT_find_range_of_a_l2212_221296


namespace NUMINAMATH_GPT_triangle_area_is_64_l2212_221289

noncomputable def area_triangle_lines (y : ℝ → ℝ) (x : ℝ → ℝ) (neg_x : ℝ → ℝ) : ℝ :=
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := 16
  let height := 8
  1 / 2 * base * height

theorem triangle_area_is_64 :
  let y := fun x => 8
  let x := fun y => y
  let neg_x := fun y => -y
  area_triangle_lines y x neg_x = 64 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_area_is_64_l2212_221289


namespace NUMINAMATH_GPT_probability_hit_10_or_7_ring_probability_below_7_ring_l2212_221211

noncomputable def P_hit_10_ring : ℝ := 0.21
noncomputable def P_hit_9_ring : ℝ := 0.23
noncomputable def P_hit_8_ring : ℝ := 0.25
noncomputable def P_hit_7_ring : ℝ := 0.28
noncomputable def P_below_7_ring : ℝ := 0.03

theorem probability_hit_10_or_7_ring :
  P_hit_10_ring + P_hit_7_ring = 0.49 :=
  by sorry

theorem probability_below_7_ring :
  P_below_7_ring = 0.03 :=
  by sorry

end NUMINAMATH_GPT_probability_hit_10_or_7_ring_probability_below_7_ring_l2212_221211


namespace NUMINAMATH_GPT_printing_presses_equivalence_l2212_221230

theorem printing_presses_equivalence :
  ∃ P : ℕ, (500000 / 12) / P = (500000 / 14) / 30 ∧ P = 26 :=
by
  sorry

end NUMINAMATH_GPT_printing_presses_equivalence_l2212_221230


namespace NUMINAMATH_GPT_calc_op_l2212_221205

def op (a b : ℕ) := (a + b) * (a - b)

theorem calc_op : (op 5 2)^2 = 441 := 
by 
  sorry

end NUMINAMATH_GPT_calc_op_l2212_221205


namespace NUMINAMATH_GPT_megan_final_balance_same_as_starting_balance_l2212_221217

theorem megan_final_balance_same_as_starting_balance :
  let starting_balance : ℝ := 125
  let increased_balance := starting_balance * (1 + 0.25)
  let final_balance := increased_balance * (1 - 0.20)
  final_balance = starting_balance :=
by
  sorry

end NUMINAMATH_GPT_megan_final_balance_same_as_starting_balance_l2212_221217


namespace NUMINAMATH_GPT_symmetric_points_x_axis_l2212_221223

theorem symmetric_points_x_axis (a b : ℝ) (P : ℝ × ℝ := (a, 1)) (Q : ℝ × ℝ := (-4, b)) :
  (Q.1 = -P.1 ∧ Q.2 = -P.2) → (a = -4 ∧ b = -1) :=
by {
  sorry
}

end NUMINAMATH_GPT_symmetric_points_x_axis_l2212_221223


namespace NUMINAMATH_GPT_find_asterisk_value_l2212_221264

theorem find_asterisk_value :
  ∃ x : ℤ, (x / 21) * (42 / 84) = 1 ↔ x = 21 :=
by
  sorry

end NUMINAMATH_GPT_find_asterisk_value_l2212_221264


namespace NUMINAMATH_GPT_sum_of_prime_factors_l2212_221280

theorem sum_of_prime_factors (x : ℕ) (h1 : x = 2^10 - 1) 
  (h2 : 2^10 - 1 = (2^5 + 1) * (2^5 - 1)) 
  (h3 : 2^5 - 1 = 31) 
  (h4 : 2^5 + 1 = 33) 
  (h5 : 33 = 3 * 11) : 
  (31 + 3 + 11 = 45) := 
  sorry

end NUMINAMATH_GPT_sum_of_prime_factors_l2212_221280


namespace NUMINAMATH_GPT_mixed_doubles_pairing_l2212_221286

def num_ways_to_pair (men women : ℕ) (select_men select_women : ℕ) : ℕ :=
  (Nat.choose men select_men) * (Nat.choose women select_women) * 2

theorem mixed_doubles_pairing : num_ways_to_pair 5 4 2 2 = 120 := by
  sorry

end NUMINAMATH_GPT_mixed_doubles_pairing_l2212_221286


namespace NUMINAMATH_GPT_average_remaining_two_numbers_l2212_221236

theorem average_remaining_two_numbers 
  (h1 : (40.5 : ℝ) = 10 * 4.05)
  (h2 : (11.1 : ℝ) = 3 * 3.7)
  (h3 : (11.85 : ℝ) = 3 * 3.95)
  (h4 : (8.6 : ℝ) = 2 * 4.3)
  : (4.475 : ℝ) = (40.5 - (11.1 + 11.85 + 8.6)) / 2 := 
sorry

end NUMINAMATH_GPT_average_remaining_two_numbers_l2212_221236


namespace NUMINAMATH_GPT_altitudes_bounded_by_perimeter_l2212_221283

theorem altitudes_bounded_by_perimeter (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a + b + c = 2) :
  ¬ (∀ (ha hb hc : ℝ), ha = 2 / a * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ∧ 
                     hb = 2 / b * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ∧ 
                     hc = 2 / c * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ∧ 
                     ha > 1 / Real.sqrt 3 ∧ 
                     hb > 1 / Real.sqrt 3 ∧ 
                     hc > 1 / Real.sqrt 3 ) :=
sorry

end NUMINAMATH_GPT_altitudes_bounded_by_perimeter_l2212_221283


namespace NUMINAMATH_GPT_length_of_platform_l2212_221272

theorem length_of_platform {train_length platform_crossing_time signal_pole_crossing_time : ℚ}
  (h_train_length : train_length = 300)
  (h_platform_crossing_time : platform_crossing_time = 40)
  (h_signal_pole_crossing_time : signal_pole_crossing_time = 18) :
  ∃ L : ℚ, L = 1100 / 3 :=
by
  sorry

end NUMINAMATH_GPT_length_of_platform_l2212_221272


namespace NUMINAMATH_GPT_raptors_points_l2212_221287

theorem raptors_points (x y z : ℕ) (h1 : x + y + z = 48) (h2 : x - y = 18) :
  (z = 0 → y = 15) ∧
  (z = 12 → y = 9) ∧
  (z = 18 → y = 6) ∧
  (z = 30 → y = 0) :=
by sorry

end NUMINAMATH_GPT_raptors_points_l2212_221287


namespace NUMINAMATH_GPT_LCM_180_504_l2212_221282

theorem LCM_180_504 : Nat.lcm 180 504 = 2520 := 
by 
  -- We skip the proof.
  sorry

end NUMINAMATH_GPT_LCM_180_504_l2212_221282


namespace NUMINAMATH_GPT_min_cubes_required_l2212_221269

/--
A lady builds a box with dimensions 10 cm length, 18 cm width, and 4 cm height using 12 cubic cm cubes. Prove that the minimum number of cubes required to build the box is 60.
-/
def min_cubes_for_box (length width height volume_cube : ℕ) : ℕ :=
  (length * width * height) / volume_cube

theorem min_cubes_required :
  min_cubes_for_box 10 18 4 12 = 60 :=
by
  -- The proof details are omitted.
  sorry

end NUMINAMATH_GPT_min_cubes_required_l2212_221269


namespace NUMINAMATH_GPT_race_winner_and_liar_l2212_221246

def Alyosha_statement (pos : ℕ → Prop) : Prop := ¬ pos 1 ∧ ¬ pos 4
def Borya_statement (pos : ℕ → Prop) : Prop := ¬ pos 4
def Vanya_statement (pos : ℕ → Prop) : Prop := pos 1
def Grisha_statement (pos : ℕ → Prop) : Prop := pos 4

def three_true_one_false (s1 s2 s3 s4 : Prop) : Prop := 
  (s1 ∧ s2 ∧ s3 ∧ ¬ s4) ∨
  (s1 ∧ s2 ∧ ¬ s3 ∧ s4) ∨
  (s1 ∧ ¬ s2 ∧ s3 ∧ s4) ∨
  (¬ s1 ∧ s2 ∧ s3 ∧ s4)

def race_result (pos : ℕ → Prop) : Prop :=
  Vanya_statement pos ∧
  three_true_one_false (Alyosha_statement pos) (Borya_statement pos) (Vanya_statement pos) (Grisha_statement pos) ∧
  Borya_statement pos = false

theorem race_winner_and_liar:
  ∃ (pos : ℕ → Prop), race_result pos :=
sorry

end NUMINAMATH_GPT_race_winner_and_liar_l2212_221246


namespace NUMINAMATH_GPT_num_special_fractions_eq_one_l2212_221277

-- Definitions of relatively prime and positive
def are_rel_prime (a b : ℕ) : Prop := Nat.gcd a b = 1
def is_positive (n : ℕ) : Prop := n > 0

-- Statement to prove the number of such fractions
theorem num_special_fractions_eq_one : 
  (∀ (x y : ℕ), is_positive x → is_positive y → are_rel_prime x y → 
    (x + 1) * 10 * y = (y + 1) * 11 * x →
    ((x = 5 ∧ y = 11) ∨ False)) := sorry

end NUMINAMATH_GPT_num_special_fractions_eq_one_l2212_221277


namespace NUMINAMATH_GPT_find_sum_l2212_221270

variable {f : ℝ → ℝ}

-- Conditions of the problem
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def condition_2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (2 + x) + f (2 - x) = 0
def condition_3 (f : ℝ → ℝ) : Prop := f 1 = 9

theorem find_sum (h_odd : odd_function f) (h_cond2 : condition_2 f) (h_cond3 : condition_3 f) :
  f 2010 + f 2011 + f 2012 = -9 :=
sorry

end NUMINAMATH_GPT_find_sum_l2212_221270


namespace NUMINAMATH_GPT_smallest_K_exists_l2212_221262

theorem smallest_K_exists (S : Finset ℕ) (h_S : S = (Finset.range 51).erase 0) :
  ∃ K, ∀ (T : Finset ℕ), T ⊆ S ∧ T.card = K → 
  ∃ a b, a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ (a + b) ∣ (a * b) ∧ K = 39 :=
by
  use 39
  sorry

end NUMINAMATH_GPT_smallest_K_exists_l2212_221262


namespace NUMINAMATH_GPT_competition_total_races_l2212_221249

theorem competition_total_races (sprinters : ℕ) (sprinters_with_bye : ℕ) (lanes_preliminary : ℕ) (lanes_subsequent : ℕ) 
  (eliminated_per_race : ℕ) (first_round_advance : ℕ) (second_round_advance : ℕ) (third_round_advance : ℕ) 
  : sprinters = 300 → sprinters_with_bye = 16 → lanes_preliminary = 8 → lanes_subsequent = 6 → 
    eliminated_per_race = 7 → first_round_advance = 36 → second_round_advance = 9 → third_round_advance = 2 
    → first_round_races = 36 → second_round_races = 9 → third_round_races = 2 → final_race = 1
    → first_round_races + second_round_races + third_round_races + final_race = 48 :=
by 
  intros sprinters_eq sprinters_with_bye_eq lanes_preliminary_eq lanes_subsequent_eq eliminated_per_race_eq 
         first_round_advance_eq second_round_advance_eq third_round_advance_eq 
         first_round_races_eq second_round_races_eq third_round_races_eq final_race_eq
  sorry

end NUMINAMATH_GPT_competition_total_races_l2212_221249


namespace NUMINAMATH_GPT_domain_of_f_l2212_221271

noncomputable def f (x : ℝ) : ℝ :=
  (x - 4)^0 + Real.sqrt (2 / (x - 1))

theorem domain_of_f :
  ∀ x : ℝ, (1 < x ∧ x < 4) ∨ (4 < x) ↔
    ∃ y : ℝ, f y = f x :=
sorry

end NUMINAMATH_GPT_domain_of_f_l2212_221271


namespace NUMINAMATH_GPT_probability_of_staying_in_dark_l2212_221202

theorem probability_of_staying_in_dark (revolutions_per_minute : ℕ) (time_in_seconds : ℕ) (dark_time : ℕ) :
  revolutions_per_minute = 2 →
  time_in_seconds = 60 →
  dark_time = 5 →
  (5 / 6 : ℝ) = 5 / 6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_probability_of_staying_in_dark_l2212_221202


namespace NUMINAMATH_GPT_walnuts_left_in_burrow_l2212_221225

-- Define the initial quantities
def boy_initial_walnuts : Nat := 6
def boy_dropped_walnuts : Nat := 1
def initial_burrow_walnuts : Nat := 12
def girl_added_walnuts : Nat := 5
def girl_eaten_walnuts : Nat := 2

-- Define the resulting quantity and the proof goal
theorem walnuts_left_in_burrow : boy_initial_walnuts - boy_dropped_walnuts + initial_burrow_walnuts + girl_added_walnuts - girl_eaten_walnuts = 20 :=
by
  sorry

end NUMINAMATH_GPT_walnuts_left_in_burrow_l2212_221225


namespace NUMINAMATH_GPT_simplify_expression_l2212_221265

noncomputable def q (x a b c d : ℝ) :=
  (x + a)^4 / ((a - b) * (a - c) * (a - d))
  + (x + b)^4 / ((b - a) * (b - c) * (b - d))
  + (x + c)^4 / ((c - a) * (c - b) * (c - d))
  + (x + d)^4 / ((d - a) * (d - b) * (d - c))

theorem simplify_expression (a b c d x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) :
  q x a b c d = a + b + c + d + 4 * x :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2212_221265


namespace NUMINAMATH_GPT_find_alpha_l2212_221292

variable (α β k : ℝ)

axiom h1 : α * β = k
axiom h2 : α = -4
axiom h3 : β = -8
axiom k_val : k = 32
axiom β_val : β = 12

theorem find_alpha (h1 : α * β = k) (h2 : α = -4) (h3 : β = -8) (k_val : k = 32) (β_val : β = 12) :
  α = 8 / 3 :=
sorry

end NUMINAMATH_GPT_find_alpha_l2212_221292


namespace NUMINAMATH_GPT_plane_speed_west_l2212_221260

theorem plane_speed_west (v t : ℝ) : 
  (300 * t + 300 * t = 1200) ∧ (t = 7 - t) → 
  (v = 300 * t / (7 - t)) ∧ (t = 2) → 
  v = 120 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_plane_speed_west_l2212_221260


namespace NUMINAMATH_GPT_preimage_exists_l2212_221207

-- Define the mapping function f
def f (x y : ℚ) : ℚ × ℚ :=
  (x + 2 * y, 2 * x - y)

-- Define the statement
theorem preimage_exists (x y : ℚ) :
  f x y = (3, 1) → (x, y) = (-1/3, 5/3) :=
by
  sorry

end NUMINAMATH_GPT_preimage_exists_l2212_221207


namespace NUMINAMATH_GPT_calc_305_squared_minus_295_squared_l2212_221288

theorem calc_305_squared_minus_295_squared :
  305^2 - 295^2 = 6000 := 
  by
    sorry

end NUMINAMATH_GPT_calc_305_squared_minus_295_squared_l2212_221288


namespace NUMINAMATH_GPT_smallest_sum_of_squares_l2212_221242

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 187) : x^2 + y^2 ≥ 205 := sorry

end NUMINAMATH_GPT_smallest_sum_of_squares_l2212_221242


namespace NUMINAMATH_GPT_students_going_to_tournament_l2212_221256

-- Defining the conditions
def total_students : ℕ := 24
def fraction_in_chess_program : ℚ := 1 / 3
def fraction_going_to_tournament : ℚ := 1 / 2

-- The final goal to prove
theorem students_going_to_tournament : 
  (total_students • fraction_in_chess_program) • fraction_going_to_tournament = 4 := 
by
  sorry

end NUMINAMATH_GPT_students_going_to_tournament_l2212_221256
