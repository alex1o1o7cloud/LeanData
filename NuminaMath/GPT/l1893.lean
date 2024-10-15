import Mathlib

namespace NUMINAMATH_GPT_orange_ratio_l1893_189352

theorem orange_ratio (total_oranges : ℕ) (brother_fraction : ℚ) (friend_receives : ℕ)
  (H1 : total_oranges = 12)
  (H2 : friend_receives = 2)
  (H3 : 1 / 4 * ((1 - brother_fraction) * total_oranges) = friend_receives) :
  brother_fraction * total_oranges / total_oranges = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_orange_ratio_l1893_189352


namespace NUMINAMATH_GPT_values_of_quadratic_expression_l1893_189398

variable {x : ℝ}

theorem values_of_quadratic_expression (h : x^2 - 4 * x + 3 < 0) : 
  (8 < x^2 + 4 * x + 3) ∧ (x^2 + 4 * x + 3 < 24) :=
sorry

end NUMINAMATH_GPT_values_of_quadratic_expression_l1893_189398


namespace NUMINAMATH_GPT_domain_correct_l1893_189385

def domain_of_function (x : ℝ) : Prop :=
  (∃ y : ℝ, y = 2 / Real.sqrt (x + 1)) ∧ Real.sqrt (x + 1) ≠ 0

theorem domain_correct (x : ℝ) : domain_of_function x ↔ (x > -1) := by
  sorry

end NUMINAMATH_GPT_domain_correct_l1893_189385


namespace NUMINAMATH_GPT_carlson_max_candies_l1893_189351

theorem carlson_max_candies : 
  (∀ (erase_two_and_sum : ℕ → ℕ → ℕ) 
    (eat_candies : ℕ → ℕ → ℕ), 
  ∃ (maximum_candies : ℕ), 
  (erase_two_and_sum 1 1 = 2) ∧
  (eat_candies 1 1 = 1) ∧ 
  (maximum_candies = 496)) :=
by
  sorry

end NUMINAMATH_GPT_carlson_max_candies_l1893_189351


namespace NUMINAMATH_GPT_max_value_of_m_l1893_189386

-- Define the function f(x)
def f (x : ℝ) := x^2 + 2 * x

-- Define the property of t and m such that the condition holds for all x in [1, m]
def valid_t_m (t m : ℝ) : Prop :=
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → f (x + t) ≤ 3 * x

-- The proof statement ensuring the maximum value of m is 8
theorem max_value_of_m 
  (t : ℝ) (m : ℝ) 
  (ht : ∃ x : ℝ, valid_t_m t x ∧ x = 8) : 
  ∀ m, valid_t_m t m → m ≤ 8 :=
  sorry

end NUMINAMATH_GPT_max_value_of_m_l1893_189386


namespace NUMINAMATH_GPT_parallel_lines_a_eq_neg1_l1893_189347

theorem parallel_lines_a_eq_neg1 (a : ℝ) :
  ∀ (x y : ℝ), 
    (x + a * y + 6 = 0) ∧ ((a - 2) * x + 3 * y + 2 * a = 0) →
    (-1 / a = - (a - 2) / 3) → 
    a = -1 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_a_eq_neg1_l1893_189347


namespace NUMINAMATH_GPT_original_average_l1893_189324

theorem original_average (A : ℝ) (h : (10 * A = 70)) : A = 7 :=
sorry

end NUMINAMATH_GPT_original_average_l1893_189324


namespace NUMINAMATH_GPT_no_integer_root_quadratic_trinomials_l1893_189325

theorem no_integer_root_quadratic_trinomials :
  ¬ ∃ (a b c : ℤ),
    (∃ r1 r2 : ℤ, a * r1^2 + b * r1 + c = 0 ∧ a * r2^2 + b * r2 + c = 0 ∧ r1 ≠ r2) ∧
    (∃ s1 s2 : ℤ, (a + 1) * s1^2 + (b + 1) * s1 + (c + 1) = 0 ∧ (a + 1) * s2^2 + (b + 1) * s2 + (c + 1) = 0 ∧ s1 ≠ s2) :=
by
  sorry

end NUMINAMATH_GPT_no_integer_root_quadratic_trinomials_l1893_189325


namespace NUMINAMATH_GPT_investment_change_l1893_189337

theorem investment_change (x : ℝ) :
  (1 : ℝ) > (0 : ℝ) → 
  1.05 * x / x - 1 * 100 = 5 :=
by
  sorry

end NUMINAMATH_GPT_investment_change_l1893_189337


namespace NUMINAMATH_GPT_fruit_shop_apples_l1893_189343

-- Given conditions
def morning_fraction : ℚ := 3 / 10
def afternoon_fraction : ℚ := 4 / 10
def total_sold : ℕ := 140

-- Define the total number of apples and the resulting condition
def total_fraction_sold : ℚ := morning_fraction + afternoon_fraction

theorem fruit_shop_apples (A : ℕ) (h : total_fraction_sold * A = total_sold) : A = 200 := 
by sorry

end NUMINAMATH_GPT_fruit_shop_apples_l1893_189343


namespace NUMINAMATH_GPT_problem_solution_l1893_189327

noncomputable def otimes (a b : ℝ) : ℝ := (a^3) / b

theorem problem_solution :
  (otimes (otimes 2 3) 4) - (otimes 2 (otimes 3 4)) = (32/9) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1893_189327


namespace NUMINAMATH_GPT_sum_of_numbers_l1893_189312

-- Definitions that come directly from the conditions
def product_condition (A B : ℕ) : Prop := A * B = 9375
def quotient_condition (A B : ℕ) : Prop := A / B = 15

-- Theorem that proves the sum of A and B is 400, based on the given conditions
theorem sum_of_numbers (A B : ℕ) (h1 : product_condition A B) (h2 : quotient_condition A B) : A + B = 400 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_l1893_189312


namespace NUMINAMATH_GPT_max_min_f_m1_possible_ns_l1893_189372

noncomputable def f (a b : ℝ) (x : ℝ) (m : ℝ) : ℝ :=
  let a := (Real.sqrt 2 * Real.sin (Real.pi / 4 + m * x), -Real.sqrt 3)
  let b := (Real.sqrt 2 * Real.sin (Real.pi / 4 + m * x), Real.cos (2 * m * x))
  a.1 * b.1 + a.2 * b.2

theorem max_min_f_m1 (x : ℝ) (h₁ : x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)) :
  2 ≤ f (Real.sqrt 2) 1 x 1 ∧ f (Real.sqrt 2) 1 x 1 ≤ 3 :=
by
  sorry

theorem possible_ns (n : ℤ) (h₂ : ∃ x : ℝ, (0 ≤ x ∧ x ≤ 2017) ∧ f (Real.sqrt 2) ((n * Real.pi) / 2) x ((n * Real.pi) / 2) = 0) :
  n = 1 ∨ n = -1 :=
by
  sorry

end NUMINAMATH_GPT_max_min_f_m1_possible_ns_l1893_189372


namespace NUMINAMATH_GPT_sum_mod_16_l1893_189381

theorem sum_mod_16 :
  (70 + 71 + 72 + 73 + 74 + 75 + 76 + 77) % 16 = 0 := 
by
  sorry

end NUMINAMATH_GPT_sum_mod_16_l1893_189381


namespace NUMINAMATH_GPT_pipe_fill_rate_l1893_189397

theorem pipe_fill_rate 
  (C : ℝ) (t : ℝ) (capacity : C = 4000) (time_to_fill : t = 300) :
  (3/4 * C / t) = 10 := 
by 
  sorry

end NUMINAMATH_GPT_pipe_fill_rate_l1893_189397


namespace NUMINAMATH_GPT_positive_m_for_one_solution_l1893_189373

theorem positive_m_for_one_solution :
  ∀ (m : ℝ), (∃ x : ℝ, 9 * x^2 + m * x + 36 = 0) ∧ 
  (∀ x y : ℝ, 9 * x^2 + m * x + 36 = 0 → 9 * y^2 + m * y + 36 = 0 → x = y) → m = 36 := 
by {
  sorry
}

end NUMINAMATH_GPT_positive_m_for_one_solution_l1893_189373


namespace NUMINAMATH_GPT_johns_weekly_allowance_l1893_189342

theorem johns_weekly_allowance (A : ℝ) 
  (arcade_spent : A * (3/5) = 3 * (A/5)) 
  (remainder_after_arcade : (2/5) * A = A - 3 * (A/5))
  (toy_store_spent : (1/3) * (2/5) * A = 2 * (A/15)) 
  (remainder_after_toy_store : (2/5) * A - (2/15) * A = 4 * (A/15))
  (last_spent : (4/15) * A = 0.4) :
  A = 1.5 :=
sorry

end NUMINAMATH_GPT_johns_weekly_allowance_l1893_189342


namespace NUMINAMATH_GPT_max_value_of_xyz_l1893_189387

noncomputable def max_product (x y z : ℝ) : ℝ :=
  x * y * z

theorem max_value_of_xyz (x y z : ℝ) (h1 : x + y + z = 1) (h2 : x = y) (h3 : 0 < x) (h4 : 0 < y) (h5 : 0 < z) (h6 : x ≤ z) (h7 : z ≤ 2 * x) :
  max_product x y z ≤ (1 / 27) := 
by
  sorry

end NUMINAMATH_GPT_max_value_of_xyz_l1893_189387


namespace NUMINAMATH_GPT_original_square_area_l1893_189379

theorem original_square_area (s : ℕ) (h1 : s + 5 = s + 5) (h2 : (s + 5)^2 = s^2 + 225) : s^2 = 400 :=
by
  sorry

end NUMINAMATH_GPT_original_square_area_l1893_189379


namespace NUMINAMATH_GPT_modulus_problem_l1893_189311

theorem modulus_problem : (13 ^ 13 + 13) % 14 = 12 :=
by
  sorry

end NUMINAMATH_GPT_modulus_problem_l1893_189311


namespace NUMINAMATH_GPT_sum_of_vars_l1893_189365

theorem sum_of_vars (x y z : ℝ) (h1 : y = 2 * x) (h2 : z = 2 * y) : x + y + z = 7 * x := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_vars_l1893_189365


namespace NUMINAMATH_GPT_triangle_ratio_perimeter_l1893_189376

theorem triangle_ratio_perimeter (AC BC : ℝ) (CD : ℝ) (AB : ℝ) (m n : ℕ) :
  AC = 15 → BC = 20 → AB = 25 → CD = 10 * Real.sqrt 3 →
  gcd m n = 1 → (2 * Real.sqrt ((AC * BC) / AB) + AB) / AB = m / n → m + n = 7 :=
by
  intros hAC hBC hAB hCD hmn hratio
  sorry

end NUMINAMATH_GPT_triangle_ratio_perimeter_l1893_189376


namespace NUMINAMATH_GPT_concert_ticket_cost_l1893_189319

-- Definitions based on the conditions
def hourlyWage : ℝ := 18
def hoursPerWeek : ℝ := 30
def drinkTicketCost : ℝ := 7
def numberOfDrinkTickets : ℝ := 5
def outingPercentage : ℝ := 0.10
def weeksPerMonth : ℝ := 4

-- Proof statement
theorem concert_ticket_cost (hourlyWage hoursPerWeek drinkTicketCost numberOfDrinkTickets outingPercentage weeksPerMonth : ℝ)
  (monthlySalary := weeksPerMonth * (hoursPerWeek * hourlyWage))
  (outingAmount := outingPercentage * monthlySalary)
  (costOfDrinkTickets := numberOfDrinkTickets * drinkTicketCost)
  (costOfConcertTicket := outingAmount - costOfDrinkTickets)
  : costOfConcertTicket = 181 := 
sorry

end NUMINAMATH_GPT_concert_ticket_cost_l1893_189319


namespace NUMINAMATH_GPT_stratified_sampling_girls_count_l1893_189304

theorem stratified_sampling_girls_count :
  (boys girls sampleSize totalSample : ℕ) →
  boys = 36 →
  girls = 18 →
  sampleSize = 6 →
  totalSample = boys + girls →
  (sampleSize * girls) / totalSample = 2 :=
by
  intros boys girls sampleSize totalSample h_boys h_girls h_sampleSize h_totalSample
  sorry

end NUMINAMATH_GPT_stratified_sampling_girls_count_l1893_189304


namespace NUMINAMATH_GPT_value_of_a_minus_b_l1893_189348

theorem value_of_a_minus_b (a b : ℝ) :
  (∀ x, - (1 / 2 : ℝ) < x ∧ x < (1 / 3 : ℝ) → ax^2 + bx + 2 > 0) → a - b = -10 := by
sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l1893_189348


namespace NUMINAMATH_GPT_algebraic_expression_l1893_189344

-- Define a variable x
variable (x : ℝ)

-- State the theorem
theorem algebraic_expression : (5 * x - 3) = 5 * x - 3 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_l1893_189344


namespace NUMINAMATH_GPT_square_ratio_short_to_long_side_l1893_189309

theorem square_ratio_short_to_long_side (a b : ℝ) (h : a / b + 1 / 2 = b / (Real.sqrt (a^2 + b^2))) : (a / b)^2 = (3 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_GPT_square_ratio_short_to_long_side_l1893_189309


namespace NUMINAMATH_GPT_series_sum_to_4_l1893_189321

theorem series_sum_to_4 (x : ℝ) (hx : ∑' n : ℕ, (n + 1) * x^n = 4) : x = 1 / 2 := 
sorry

end NUMINAMATH_GPT_series_sum_to_4_l1893_189321


namespace NUMINAMATH_GPT_line_slope_through_origin_intersects_parabola_l1893_189323

theorem line_slope_through_origin_intersects_parabola (k : ℝ) :
  (∃ x1 x2 : ℝ, 5 * (kx1) = 2 * x1 ^ 2 - 9 * x1 + 10 ∧ 5 * (kx2) = 2 * x2 ^ 2 - 9 * x2 + 10 ∧ x1 + x2 = 77) → k = 29 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_line_slope_through_origin_intersects_parabola_l1893_189323


namespace NUMINAMATH_GPT_distance_with_wind_l1893_189363

-- Define constants
def distance_against_wind : ℝ := 320
def speed_wind : ℝ := 20
def speed_plane_still_air : ℝ := 180

-- Calculate effective speeds
def effective_speed_with_wind : ℝ := speed_plane_still_air + speed_wind
def effective_speed_against_wind : ℝ := speed_plane_still_air - speed_wind

-- Define the proof statement
theorem distance_with_wind :
  ∃ (D : ℝ), (D / effective_speed_with_wind) = (distance_against_wind / effective_speed_against_wind) ∧ D = 400 :=
by
  sorry

end NUMINAMATH_GPT_distance_with_wind_l1893_189363


namespace NUMINAMATH_GPT_value_of_Priyanka_l1893_189307

-- Defining the context with the conditions
variables (X : ℕ) (Neha : ℕ) (Sonali Priyanka Sadaf Tanu : ℕ)
-- The conditions given in the problem
axiom h1 : Neha = X
axiom h2 : Sonali = 15
axiom h3 : Priyanka = 15
axiom h4 : Sadaf = Neha
axiom h5 : Tanu = Neha

-- Stating the theorem we need to prove
theorem value_of_Priyanka : Priyanka = 15 :=
by
  sorry

end NUMINAMATH_GPT_value_of_Priyanka_l1893_189307


namespace NUMINAMATH_GPT_money_made_l1893_189310

def initial_amount : ℕ := 26
def final_amount : ℕ := 52

theorem money_made : (final_amount - initial_amount) = 26 :=
by sorry

end NUMINAMATH_GPT_money_made_l1893_189310


namespace NUMINAMATH_GPT_point_D_is_on_y_axis_l1893_189374

def is_on_y_axis (p : ℝ × ℝ) : Prop := p.fst = 0

def point_A : ℝ × ℝ := (3, 0)
def point_B : ℝ × ℝ := (1, 2)
def point_C : ℝ × ℝ := (2, 1)
def point_D : ℝ × ℝ := (0, -3)

theorem point_D_is_on_y_axis : is_on_y_axis point_D :=
by
  sorry

end NUMINAMATH_GPT_point_D_is_on_y_axis_l1893_189374


namespace NUMINAMATH_GPT_find_n_square_divides_exponential_plus_one_l1893_189364

theorem find_n_square_divides_exponential_plus_one :
  ∀ n : ℕ, (n^2 ∣ 2^n + 1) → (n = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_n_square_divides_exponential_plus_one_l1893_189364


namespace NUMINAMATH_GPT_projection_multiplier_l1893_189338

noncomputable def a : ℝ × ℝ := (3, 6)
noncomputable def b : ℝ × ℝ := (-1, 0)

theorem projection_multiplier :
  let dot_product := a.1 * b.1 + a.2 * b.2
  let b_norm_sq := b.1 * b.1 + b.2 * b.2
  let proj := (dot_product / b_norm_sq) * 2
  (proj * b.1, proj * b.2) = (6, 0) :=
by 
  sorry

end NUMINAMATH_GPT_projection_multiplier_l1893_189338


namespace NUMINAMATH_GPT_fraction_of_airing_time_spent_on_commercials_l1893_189332

theorem fraction_of_airing_time_spent_on_commercials 
  (num_programs : ℕ) (minutes_per_program : ℕ) (total_commercial_time : ℕ) 
  (h1 : num_programs = 6) (h2 : minutes_per_program = 30) (h3 : total_commercial_time = 45) : 
  (total_commercial_time : ℚ) / (num_programs * minutes_per_program : ℚ) = 1 / 4 :=
by {
  -- The proof is omitted here as only the statement is required according to the instruction.
  sorry
}

end NUMINAMATH_GPT_fraction_of_airing_time_spent_on_commercials_l1893_189332


namespace NUMINAMATH_GPT_quadratic_equation_with_roots_sum_and_difference_l1893_189388

theorem quadratic_equation_with_roots_sum_and_difference (p q : ℚ)
  (h1 : p + q = 10)
  (h2 : abs (p - q) = 2) :
  (Polynomial.eval₂ (RingHom.id ℚ) p (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-10) * Polynomial.X + Polynomial.C 24) = 0) ∧
  (Polynomial.eval₂ (RingHom.id ℚ) q (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-10) * Polynomial.X + Polynomial.C 24) = 0) :=
by sorry

end NUMINAMATH_GPT_quadratic_equation_with_roots_sum_and_difference_l1893_189388


namespace NUMINAMATH_GPT_base_five_to_base_ten_modulo_seven_l1893_189390

-- Define the base five number 21014_5 as the corresponding base ten conversion
def base_five_number : ℕ := 2 * 5^4 + 1 * 5^3 + 0 * 5^2 + 1 * 5^1 + 4 * 5^0

-- The equivalent base ten result
def base_ten_number : ℕ := 1384

-- Verify the base ten equivalent of 21014_5
theorem base_five_to_base_ten : base_five_number = base_ten_number :=
by
  -- The expected proof should compute the value of base_five_number
  -- and check that it equals 1384
  sorry

-- Find the modulo operation result of 1384 % 7
def modulo_seven_result : ℕ := 6

-- Verify 1384 % 7 gives 6
theorem modulo_seven : base_ten_number % 7 = modulo_seven_result :=
by
  -- The expected proof should compute 1384 % 7
  -- and check that it equals 6
  sorry

end NUMINAMATH_GPT_base_five_to_base_ten_modulo_seven_l1893_189390


namespace NUMINAMATH_GPT_largest_possible_difference_l1893_189326

theorem largest_possible_difference (A_est : ℕ) (B_est : ℕ) (A : ℝ) (B : ℝ)
(hA_est : A_est = 40000) (hB_est : B_est = 70000)
(hA_range : 36000 ≤ A ∧ A ≤ 44000)
(hB_range : 60870 ≤ B ∧ B ≤ 82353) :
  abs (B - A) = 46000 :=
by sorry

end NUMINAMATH_GPT_largest_possible_difference_l1893_189326


namespace NUMINAMATH_GPT_simplify_expression_l1893_189331

theorem simplify_expression : 
  (1 - 1 / (2 + Real.sqrt 5) + 1 / (2 - Real.sqrt 5) = 1) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1893_189331


namespace NUMINAMATH_GPT_pages_left_to_read_l1893_189389

-- Define the given conditions
def total_pages : ℕ := 563
def pages_read : ℕ := 147

-- Define the proof statement
theorem pages_left_to_read : total_pages - pages_read = 416 :=
by
  -- The proof will be given here
  sorry

end NUMINAMATH_GPT_pages_left_to_read_l1893_189389


namespace NUMINAMATH_GPT_addition_of_two_odds_is_even_subtraction_of_two_odds_is_even_l1893_189302

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem addition_of_two_odds_is_even (a b : ℤ) (ha : is_odd a) (hb : is_odd b) : is_even (a + b) :=
sorry

theorem subtraction_of_two_odds_is_even (a b : ℤ) (ha : is_odd a) (hb : is_odd b) : is_even (a - b) :=
sorry

end NUMINAMATH_GPT_addition_of_two_odds_is_even_subtraction_of_two_odds_is_even_l1893_189302


namespace NUMINAMATH_GPT_min_value_of_3x_2y_l1893_189392

noncomputable def min_value (x y: ℝ) : ℝ := 3 * x + 2 * y

theorem min_value_of_3x_2y (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + y - x * y = 0) :
  min_value x y = 5 + 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_min_value_of_3x_2y_l1893_189392


namespace NUMINAMATH_GPT_systematic_sampling_result_l1893_189367

-- Define the set of bags numbered from 1 to 30
def bags : Set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the systematic sampling function
def systematic_sampling (n k interval : ℕ) : List ℕ :=
  List.range k |> List.map (λ i => n + i * interval)

-- Specific parameters for the problem
def number_of_bags := 30
def bags_drawn := 6
def interval := 5
def expected_samples := [2, 7, 12, 17, 22, 27]

-- Statement of the theorem
theorem systematic_sampling_result : 
  systematic_sampling 2 bags_drawn interval = expected_samples :=
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_result_l1893_189367


namespace NUMINAMATH_GPT_theta_in_second_quadrant_l1893_189314

theorem theta_in_second_quadrant (θ : ℝ) (h₁ : Real.sin θ > 0) (h₂ : Real.cos θ < 0) : 
  π / 2 < θ ∧ θ < π := 
sorry

end NUMINAMATH_GPT_theta_in_second_quadrant_l1893_189314


namespace NUMINAMATH_GPT_carries_jellybeans_l1893_189346

/-- Bert's box holds 150 jellybeans. --/
def bert_jellybeans : ℕ := 150

/-- Carrie's box is three times as high, three times as wide, and three times as long as Bert's box. --/
def volume_ratio : ℕ := 27

/-- Given that Carrie's box dimensions are three times those of Bert's and Bert's box holds 150 jellybeans, 
    we need to prove that Carrie's box holds 4050 jellybeans. --/
theorem carries_jellybeans : bert_jellybeans * volume_ratio = 4050 := 
by sorry

end NUMINAMATH_GPT_carries_jellybeans_l1893_189346


namespace NUMINAMATH_GPT_orthogonal_vectors_l1893_189399

open Real

variables (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h : (a + b)^2 = (a - b)^2)

theorem orthogonal_vectors (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) 
  (h : (a + b)^2 = (a - b)^2) : a * b = 0 :=
by 
  sorry

end NUMINAMATH_GPT_orthogonal_vectors_l1893_189399


namespace NUMINAMATH_GPT_percent_problem_l1893_189361

theorem percent_problem (x : ℝ) (h : 0.35 * 400 = 0.20 * x) : x = 700 :=
by sorry

end NUMINAMATH_GPT_percent_problem_l1893_189361


namespace NUMINAMATH_GPT_monotonicity_of_f_solve_inequality_l1893_189380

noncomputable def f (x : ℝ) : ℝ := sorry

def f_defined : ∀ x > 0, ∃ y, f y = f x := sorry

axiom functional_eq : ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x + f y 

axiom f_gt_zero : ∀ x, x > 1 → f x > 0

theorem monotonicity_of_f : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → f x1 < f x2 :=
sorry

theorem solve_inequality (x : ℝ) (h1 : f 2 = 1) (h2 : 0 < x) : 
  f x + f (x - 3) ≤ 2 ↔ 3 < x ∧ x ≤ 4 :=
sorry

end NUMINAMATH_GPT_monotonicity_of_f_solve_inequality_l1893_189380


namespace NUMINAMATH_GPT_minimum_value_8_l1893_189301

noncomputable def minimum_value (x : ℝ) : ℝ :=
  3 * x + 2 / x^5 + 3 / x

theorem minimum_value_8 (x : ℝ) (hx : x > 0) :
  ∃ y : ℝ, (∀ z > 0, minimum_value z ≥ y) ∧ (y = 8) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_8_l1893_189301


namespace NUMINAMATH_GPT_debra_probability_l1893_189322

theorem debra_probability :
  let p_tail := 1 / 2
  let p_head := 1 / 2
  let p_THTHT := p_tail * p_head * p_tail * p_head * p_tail
  let P := (1 / 4) / (3 / 4)
  (p_THTHT * P) = 1 / 96 :=
by
  -- Definitions of p_tail, p_head, p_THTHT, and P
  let p_tail := 1 / 2
  let p_head := 1 / 2
  let p_THTHT := p_tail * p_head * p_tail * p_head * p_tail
  let P := (1 / 4) / (3 / 4)
  -- Placeholder for proof computation
  sorry

end NUMINAMATH_GPT_debra_probability_l1893_189322


namespace NUMINAMATH_GPT_general_formula_a_sum_sn_l1893_189366

-- Define the sequence {a_n}
def a (n : ℕ) : ℕ :=
  if n = 0 then 2 else 2 * n

-- Define the sequence {b_n}
def b (n : ℕ) : ℕ :=
  a n + 2 ^ (a n)

-- Define the sum of the first n terms of the sequence {b_n}
def S (n : ℕ) : ℕ :=
  (Finset.range n).sum b

theorem general_formula_a :
  ∀ n, a n = 2 * n :=
sorry

theorem sum_sn :
  ∀ n, S n = n * (n + 1) + (4^(n + 1) - 4) / 3 :=
sorry

end NUMINAMATH_GPT_general_formula_a_sum_sn_l1893_189366


namespace NUMINAMATH_GPT_reflection_of_point_l1893_189359

def reflect_across_y_neg_x (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P
  (y, x)

theorem reflection_of_point
  (P : ℝ × ℝ)
  (h : P = (8, -3)) :
  reflect_across_y_neg_x P = (3, -8) :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_reflection_of_point_l1893_189359


namespace NUMINAMATH_GPT_system_of_equations_solution_l1893_189315

theorem system_of_equations_solution:
  ∀ (x y : ℝ), 
    x^2 + y^2 + x + y = 42 ∧ x * y = 15 → 
      (x = 3 ∧ y = 5) ∨ (x = 5 ∧ y = 3) ∨ 
      (x = (-9 + Real.sqrt 21) / 2 ∧ y = (-9 - Real.sqrt 21) / 2) ∨ 
      (x = (-9 - Real.sqrt 21) / 2 ∧ y = (-9 + Real.sqrt 21) / 2) := 
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l1893_189315


namespace NUMINAMATH_GPT_k_n_sum_l1893_189382

theorem k_n_sum (k n : ℕ) (x y : ℕ):
  2 * x^k * y^(k+2) + 3 * x^2 * y^n = 5 * x^2 * y^n → k + n = 6 :=
by sorry

end NUMINAMATH_GPT_k_n_sum_l1893_189382


namespace NUMINAMATH_GPT_sum_abcd_l1893_189377

variable {a b c d : ℚ}

theorem sum_abcd 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 4) : 
  a + b + c + d = -2/3 :=
by sorry

end NUMINAMATH_GPT_sum_abcd_l1893_189377


namespace NUMINAMATH_GPT_largest_prime_divisor_for_primality_check_l1893_189355

theorem largest_prime_divisor_for_primality_check (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1050) : 
  ∃ p, Prime p ∧ p ≤ Int.sqrt 1050 ∧ ∀ q, Prime q → q ≤ Int.sqrt n → q ≤ p := sorry

end NUMINAMATH_GPT_largest_prime_divisor_for_primality_check_l1893_189355


namespace NUMINAMATH_GPT_length_of_real_axis_of_hyperbola_l1893_189354

theorem length_of_real_axis_of_hyperbola :
  ∀ (x y : ℝ), 2 * x^2 - y^2 = 8 -> ∃ a : ℝ, 2 * a = 4 :=
by
intro x y h
sorry

end NUMINAMATH_GPT_length_of_real_axis_of_hyperbola_l1893_189354


namespace NUMINAMATH_GPT_n_squared_divisible_by_144_l1893_189368

-- Definitions based on the conditions
variables (n k : ℕ)
def is_positive (n : ℕ) : Prop := n > 0
def largest_divisor_of_n_is_twelve (n : ℕ) : Prop := ∃ k, n = 12 * k
def divisible_by (m n : ℕ) : Prop := ∃ k, m = n * k

theorem n_squared_divisible_by_144
  (h1 : is_positive n)
  (h2 : largest_divisor_of_n_is_twelve n) :
  divisible_by (n * n) 144 :=
sorry

end NUMINAMATH_GPT_n_squared_divisible_by_144_l1893_189368


namespace NUMINAMATH_GPT_calculate_percentage_passed_l1893_189330

theorem calculate_percentage_passed (F_H F_E F_HE : ℝ) (h1 : F_H = 0.32) (h2 : F_E = 0.56) (h3 : F_HE = 0.12) :
  1 - (F_H + F_E - F_HE) = 0.24 := by
  sorry

end NUMINAMATH_GPT_calculate_percentage_passed_l1893_189330


namespace NUMINAMATH_GPT_vertical_angles_congruent_l1893_189369

namespace VerticalAngles

-- Define what it means for two angles to be vertical
def Vertical (α β : Real) : Prop :=
  -- Definition of vertical angles goes here
  sorry

-- Hypothesis: If two angles are vertical angles
theorem vertical_angles_congruent (α β : Real) (h : Vertical α β) : α = β :=
sorry

end VerticalAngles

end NUMINAMATH_GPT_vertical_angles_congruent_l1893_189369


namespace NUMINAMATH_GPT_find_s_l1893_189308

def is_monic_cubic (p : Polynomial ℝ) : Prop :=
  p.degree = 3 ∧ p.leadingCoeff = 1

def has_roots (p : Polynomial ℝ) (roots : Set ℝ) : Prop :=
  ∀ x ∈ roots, p.eval x = 0

def poly_condition (f g : Polynomial ℝ) (s : ℝ) : Prop :=
  ∀ x : ℝ, f.eval x - g.eval x = 2 * s

theorem find_s (s : ℝ)
  (f g : Polynomial ℝ)
  (hf_monic : is_monic_cubic f)
  (hg_monic : is_monic_cubic g)
  (hf_roots : has_roots f {s + 2, s + 6})
  (hg_roots : has_roots g {s + 4, s + 10})
  (h_condition : poly_condition f g s) :
  s = 10.67 :=
sorry

end NUMINAMATH_GPT_find_s_l1893_189308


namespace NUMINAMATH_GPT_sum_SHE_equals_6_l1893_189358

-- Definitions for conditions
variables {S H E : ℕ}

-- Conditions as stated in the problem
def distinct_non_zero_digits (S H E : ℕ) : Prop :=
  S ≠ H ∧ H ≠ E ∧ S ≠ E ∧ 1 ≤ S ∧ S < 8 ∧ 1 ≤ H ∧ H < 8 ∧ 1 ≤ E ∧ E < 8

-- Base 8 addition problem
def addition_holds_in_base8 (S H E : ℕ) : Prop :=
  (E + H + (S + E + H) / 8) % 8 = S ∧    -- First column carry
  (H + S + (E + H + S) / 8) % 8 = E ∧    -- Second column carry
  (S + E + (H + S + E) / 8) % 8 = H      -- Third column carry

-- Final statement
theorem sum_SHE_equals_6 :
  distinct_non_zero_digits S H E → addition_holds_in_base8 S H E → S + H + E = 6 :=
by sorry

end NUMINAMATH_GPT_sum_SHE_equals_6_l1893_189358


namespace NUMINAMATH_GPT_value_of_expression_l1893_189334

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) : 
  a * b / (c * d) = 180 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1893_189334


namespace NUMINAMATH_GPT_triangle_area_l1893_189328

theorem triangle_area {a b m : ℝ} (h1 : a = 27) (h2 : b = 29) (h3 : m = 26) : 
  ∃ (area : ℝ), area = 270 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l1893_189328


namespace NUMINAMATH_GPT_log_addition_property_l1893_189335

noncomputable def logFunction (x : ℝ) : ℝ := Real.log x

theorem log_addition_property (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : logFunction (a * b) = 1) :
  logFunction (a^2) + logFunction (b^2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_log_addition_property_l1893_189335


namespace NUMINAMATH_GPT_distinct_real_numbers_inequality_l1893_189383

theorem distinct_real_numbers_inequality
  (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) :
  ( (2 * a - b) / (a - b) )^2 + ( (2 * b - c) / (b - c) )^2 + ( (2 * c - a) / (c - a) )^2 ≥ 5 :=
by {
    sorry
}

end NUMINAMATH_GPT_distinct_real_numbers_inequality_l1893_189383


namespace NUMINAMATH_GPT_percentage_increase_from_March_to_January_l1893_189391

variable {F J M : ℝ}

def JanuaryCondition (F J : ℝ) : Prop :=
  J = 0.90 * F

def MarchCondition (F M : ℝ) : Prop :=
  M = 0.75 * F

theorem percentage_increase_from_March_to_January (F J M : ℝ) (h1 : JanuaryCondition F J) (h2 : MarchCondition F M) :
  (J / M) = 1.20 := by 
  sorry

end NUMINAMATH_GPT_percentage_increase_from_March_to_January_l1893_189391


namespace NUMINAMATH_GPT_team_A_wins_exactly_4_of_7_l1893_189329

noncomputable def probability_team_A_wins_4_of_7 : ℚ :=
  (Nat.choose 7 4) * ((1/2)^4) * ((1/2)^3)

theorem team_A_wins_exactly_4_of_7 :
  probability_team_A_wins_4_of_7 = 35 / 128 := by
sorry

end NUMINAMATH_GPT_team_A_wins_exactly_4_of_7_l1893_189329


namespace NUMINAMATH_GPT_average_value_of_x_l1893_189339

theorem average_value_of_x
  (x : ℝ)
  (h : (5 + 5 + x + 6 + 8) / 5 = 6) :
  x = 6 :=
sorry

end NUMINAMATH_GPT_average_value_of_x_l1893_189339


namespace NUMINAMATH_GPT_simplify_expression_l1893_189360

noncomputable def simplify_expr : ℝ :=
  (3 + 2 * Real.sqrt 2) ^ Real.sqrt 3

theorem simplify_expression :
  (Real.sqrt 2 - 1) ^ (2 - Real.sqrt 3) / (Real.sqrt 2 + 1) ^ (2 + Real.sqrt 3) = simplify_expr :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1893_189360


namespace NUMINAMATH_GPT_minimum_quadratic_expression_l1893_189353

theorem minimum_quadratic_expression : ∃ (x : ℝ), (∀ y : ℝ, y^2 - 6*y + 5 ≥ -4) ∧ (x^2 - 6*x + 5 = -4) :=
by
  sorry

end NUMINAMATH_GPT_minimum_quadratic_expression_l1893_189353


namespace NUMINAMATH_GPT_total_cost_of_apples_and_bananas_l1893_189345

variable (a b : ℝ)

theorem total_cost_of_apples_and_bananas (a b : ℝ) : 2 * a + 3 * b = 2 * a + 3 * b :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_apples_and_bananas_l1893_189345


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l1893_189317

variable {a b : ℝ} -- Assume a and b are arbitrary real numbers

-- Part 1: Prove that 2a - [-3b - 3(3a - b)] = 11a
theorem simplify_expr1 : (2 * a - (-3 * b - 3 * (3 * a - b))) = 11 * a :=
by
  sorry

-- Part 2: Prove that 12ab^2 - [7a^2b - (ab^2 - 3a^2b)] = 13ab^2 - 10a^2b
theorem simplify_expr2 : (12 * a * b^2 - (7 * a^2 * b - (a * b^2 - 3 * a^2 * b))) = (13 * a * b^2 - 10 * a^2 * b) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l1893_189317


namespace NUMINAMATH_GPT_num_solutions_l1893_189378

-- Let x be a real number
variable (x : ℝ)

-- Define the given equation
def equation := (x^2 - 4) * (x^2 - 1) = (x^2 + 3*x + 2) * (x^2 - 8*x + 7)

-- Theorem: The number of values of x that satisfy the equation is 3
theorem num_solutions : ∃ (S : Finset ℝ), (∀ x, x ∈ S ↔ equation x) ∧ S.card = 3 := 
by
  sorry

end NUMINAMATH_GPT_num_solutions_l1893_189378


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l1893_189395

theorem repeating_decimal_to_fraction 
  (h : ∀ {x : ℝ}, (0.01 : ℝ) = 1 / 99 → x = 1.06 → (0.06 : ℝ) = 6 * 1 / 99): 
  1.06 = 35 / 33 :=
by sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l1893_189395


namespace NUMINAMATH_GPT_acute_triangle_tangent_sum_range_l1893_189313

theorem acute_triangle_tangent_sum_range
  (a b c : ℝ) (A B C : ℝ)
  (triangle_ABC_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (opposite_sides : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (side_relation : b^2 - a^2 = a * c)
  (angle_relation : A + B + C = π)
  (angles_in_radians : 0 < A ∧ A < π)
  (angles_positive : A > 0 ∧ B > 0 ∧ C > 0) :
  1 < (1 / Real.tan A + 1 / Real.tan B) ∧ (1 / Real.tan A + 1 / Real.tan B) < 2 * Real.sqrt 3 / 3 :=
sorry 

end NUMINAMATH_GPT_acute_triangle_tangent_sum_range_l1893_189313


namespace NUMINAMATH_GPT_sum_of_coefficients_l1893_189371

-- Define the polynomial P(x)
def P (x : ℤ) : ℤ := (2 * x^2021 - x^2020 + x^2019)^11 - 29

-- State the theorem we intend to prove
theorem sum_of_coefficients : P 1 = 2019 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1893_189371


namespace NUMINAMATH_GPT_complement_intersection_l1893_189356

open Set

theorem complement_intersection {x : ℝ} :
  (x ∉ {x | -2 ≤ x ∧ x ≤ 2}) ∧ (x < 1) ↔ (x < -2) := 
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1893_189356


namespace NUMINAMATH_GPT_factorization_identity_l1893_189375

noncomputable def factor_expression (a b c : ℝ) : ℝ :=
  ((a ^ 2 + 1 - (b ^ 2 + 1)) ^ 3 + ((b ^ 2 + 1) - (c ^ 2 + 1)) ^ 3 + ((c ^ 2 + 1) - (a ^ 2 + 1)) ^ 3) /
  ((a - b) ^ 3 + (b - c) ^ 3 + (c - a) ^ 3)

theorem factorization_identity (a b c : ℝ) : 
  factor_expression a b c = (a + b) * (b + c) * (c + a) := 
by 
  sorry

end NUMINAMATH_GPT_factorization_identity_l1893_189375


namespace NUMINAMATH_GPT_Jill_age_l1893_189336

theorem Jill_age 
  (G H I J : ℕ)
  (h1 : G = H - 4)
  (h2 : H = I + 5)
  (h3 : I + 2 = J)
  (h4 : G = 18) : 
  J = 19 := 
sorry

end NUMINAMATH_GPT_Jill_age_l1893_189336


namespace NUMINAMATH_GPT_mean_reciprocals_first_three_composites_l1893_189306

theorem mean_reciprocals_first_three_composites :
  (1 / 4 + 1 / 6 + 1 / 8) / 3 = (13 : ℚ) / 72 := 
by
  sorry

end NUMINAMATH_GPT_mean_reciprocals_first_three_composites_l1893_189306


namespace NUMINAMATH_GPT_product_divisible_by_3_l1893_189393

noncomputable def dice_prob_divisible_by_3 (n : ℕ) (faces : List ℕ) : ℚ := 
  let probability_div_3 := (1 / 3 : ℚ)
  let probability_not_div_3 := (2 / 3 : ℚ)
  1 - probability_not_div_3 ^ n

theorem product_divisible_by_3 (faces : List ℕ) (h_faces : faces = [1, 2, 3, 4, 5, 6]) :
  dice_prob_divisible_by_3 6 faces = 665 / 729 := 
  by 
    sorry

end NUMINAMATH_GPT_product_divisible_by_3_l1893_189393


namespace NUMINAMATH_GPT_subset_of_inter_eq_self_l1893_189396

variable {α : Type*}
variables (M N : Set α)

theorem subset_of_inter_eq_self (h : M ∩ N = M) : M ⊆ N :=
sorry

end NUMINAMATH_GPT_subset_of_inter_eq_self_l1893_189396


namespace NUMINAMATH_GPT_average_speed_stan_l1893_189318

theorem average_speed_stan (d1 d2 : ℝ) (h1 h2 rest : ℝ) (total_distance total_time : ℝ) (avg_speed : ℝ) :
  d1 = 350 → 
  d2 = 400 → 
  h1 = 6 → 
  h2 = 7 → 
  rest = 0.5 → 
  total_distance = d1 + d2 → 
  total_time = h1 + h2 + rest → 
  avg_speed = total_distance / total_time → 
  avg_speed = 55.56 :=
by 
  intros h_d1 h_d2 h_h1 h_h2 h_rest h_total_distance h_total_time h_avg_speed
  sorry

end NUMINAMATH_GPT_average_speed_stan_l1893_189318


namespace NUMINAMATH_GPT_glen_pop_l1893_189384

/-- In the village of Glen, the total population can be formulated as 21h + 6c
given the relationships between people, horses, sheep, cows, and ducks.
We need to prove that 96 cannot be expressed in the form 21h + 6c for
non-negative integers h and c. -/
theorem glen_pop (h c : ℕ) : 21 * h + 6 * c ≠ 96 :=
by
sorry

end NUMINAMATH_GPT_glen_pop_l1893_189384


namespace NUMINAMATH_GPT_sin_of_2000_deg_l1893_189333

theorem sin_of_2000_deg (a : ℝ) (h : Real.tan (160 * Real.pi / 180) = a) : 
  Real.sin (2000 * Real.pi / 180) = -a / Real.sqrt (1 + a^2) := 
by
  sorry

end NUMINAMATH_GPT_sin_of_2000_deg_l1893_189333


namespace NUMINAMATH_GPT_radius_of_sphere_l1893_189341

theorem radius_of_sphere (R : ℝ) (shots_count : ℕ) (shot_radius : ℝ) :
  shots_count = 125 →
  shot_radius = 1 →
  (shots_count : ℝ) * (4 / 3 * Real.pi * shot_radius^3) = 4 / 3 * Real.pi * R^3 →
  R = 5 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_radius_of_sphere_l1893_189341


namespace NUMINAMATH_GPT_surface_area_of_cylinder_l1893_189350

noncomputable def cylinder_surface_area
    (r : ℝ) (V : ℝ) (S : ℝ) : Prop :=
    r = 1 ∧ V = 2 * Real.pi ∧ S = 6 * Real.pi

theorem surface_area_of_cylinder
    (r : ℝ) (V : ℝ) : ∃ S : ℝ, cylinder_surface_area r V S :=
by
  use 6 * Real.pi
  sorry

end NUMINAMATH_GPT_surface_area_of_cylinder_l1893_189350


namespace NUMINAMATH_GPT_parabola_equation_line_intersection_proof_l1893_189300

-- Define the parabola and its properties
def parabola (p x y : ℝ) := y^2 = 2 * p * x

-- Define point A
def A_point (x y₀ : ℝ) := (x, y₀)

-- Define the conditions
axiom p_pos (p : ℝ) : p > 0
axiom passes_A (y₀ : ℝ) (p : ℝ) : parabola p 2 y₀
axiom distance_A_axis (p : ℝ) : 2 + p / 2 = 4

-- Prove the equation of the parabola given the conditions
theorem parabola_equation : ∃ p, parabola p x y ∧ p = 4 := sorry

-- Define line l and its intersection properties
def line_l (m x y : ℝ) := y = x + m
def intersection_PQ (m x₁ x₂ y₁ y₂ : ℝ) := 
  line_l m x₁ y₁ ∧ line_l m x₂ y₂ ∧ y₁^2 = 8 * x₁ ∧ y₂^2 = 8 * x₂ ∧ 
  x₁ + x₂ = 8 - 2 * m ∧ x₁ * x₂ = m^2 ∧ y₁ + y₂ = 8 ∧ y₁ * y₂ = 8 * m ∧ 
  x₁ * x₂ + y₁ * y₂ = 0

-- Prove the value of m
theorem line_intersection_proof : ∃ m, ∀ (x₁ x₂ y₁ y₂ : ℝ), 
  intersection_PQ m x₁ x₂ y₁ y₂ -> m = -8 := sorry

end NUMINAMATH_GPT_parabola_equation_line_intersection_proof_l1893_189300


namespace NUMINAMATH_GPT_decimal_expansion_of_13_over_625_l1893_189349

theorem decimal_expansion_of_13_over_625 : (13 : ℚ) / 625 = 0.0208 :=
by sorry

end NUMINAMATH_GPT_decimal_expansion_of_13_over_625_l1893_189349


namespace NUMINAMATH_GPT_ben_points_l1893_189316

theorem ben_points (zach_points : ℝ) (total_points : ℝ) (ben_points : ℝ) 
  (h1 : zach_points = 42.0) 
  (h2 : total_points = 63) 
  (h3 : total_points = zach_points + ben_points) : 
  ben_points = 21 :=
by
  sorry

end NUMINAMATH_GPT_ben_points_l1893_189316


namespace NUMINAMATH_GPT_avg_temp_l1893_189320

theorem avg_temp (M T W Th F : ℝ) (h1 : M = 41) (h2 : F = 33) (h3 : (T + W + Th + F) / 4 = 46) : 
  (M + T + W + Th) / 4 = 48 :=
by
  -- insert proof steps here
  sorry

end NUMINAMATH_GPT_avg_temp_l1893_189320


namespace NUMINAMATH_GPT_total_weight_moved_l1893_189362

-- Define the given conditions as Lean definitions
def weight_per_rep : ℕ := 15
def number_of_reps : ℕ := 10
def number_of_sets : ℕ := 3

-- Define the theorem to prove total weight moved is 450 pounds
theorem total_weight_moved : weight_per_rep * number_of_reps * number_of_sets = 450 := by
  sorry

end NUMINAMATH_GPT_total_weight_moved_l1893_189362


namespace NUMINAMATH_GPT_range_of_a_l1893_189394

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-4 : ℝ) a, (x^2 - 4 * x) ∈ Set.Icc (-4 : ℝ) 32) →
  2 ≤ a ∧ a ≤ 8 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1893_189394


namespace NUMINAMATH_GPT_angle_with_same_terminal_side_l1893_189305

-- Given conditions in the problem: angles to choose from
def angles : List ℕ := [60, 70, 100, 130]

-- Definition of the equivalence relation (angles having the same terminal side)
def same_terminal_side (θ α : ℕ) : Prop :=
  ∃ k : ℤ, θ = α + k * 360

-- Proof goal: 420° has the same terminal side as one of the angles in the list
theorem angle_with_same_terminal_side :
  ∃ α ∈ angles, same_terminal_side 420 α :=
sorry  -- proof not required

end NUMINAMATH_GPT_angle_with_same_terminal_side_l1893_189305


namespace NUMINAMATH_GPT_find_annual_interest_rate_l1893_189340

variable (r : ℝ) -- The annual interest rate we want to prove

-- Define the conditions based on the problem statement
variable (I : ℝ := 300) -- interest earned
variable (P : ℝ := 10000) -- principal amount
variable (t : ℝ := 9 / 12) -- time in years

-- Define the simple interest formula condition
def simple_interest_formula : Prop :=
  I = P * r * t

-- The statement to prove
theorem find_annual_interest_rate : simple_interest_formula r ↔ r = 0.04 :=
  by
    unfold simple_interest_formula
    simp
    sorry

end NUMINAMATH_GPT_find_annual_interest_rate_l1893_189340


namespace NUMINAMATH_GPT_expression_equals_neg_one_l1893_189357

theorem expression_equals_neg_one (b y : ℝ) (hb : b ≠ 0) (h₁ : y ≠ b) (h₂ : y ≠ -b) :
  ( (b / (b + y) + y / (b - y)) / (y / (b + y) - b / (b - y)) ) = -1 :=
sorry

end NUMINAMATH_GPT_expression_equals_neg_one_l1893_189357


namespace NUMINAMATH_GPT_greatest_int_lt_neg_31_div_6_l1893_189303

theorem greatest_int_lt_neg_31_div_6 : ∃ (n : ℤ), n < -31 / 6 ∧ ∀ m : ℤ, m < -31 / 6 → m ≤ n := 
sorry

end NUMINAMATH_GPT_greatest_int_lt_neg_31_div_6_l1893_189303


namespace NUMINAMATH_GPT_intersection_A_B_l1893_189370

def A : Set ℤ := {-2, -1, 0, 1, 2, 3}
def B : Set ℤ := {x | x^2 - 2 * x - 3 < 0}

theorem intersection_A_B : A ∩ B = {0, 1, 2} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1893_189370
