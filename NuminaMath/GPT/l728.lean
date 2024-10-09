import Mathlib

namespace least_x_l728_72867

noncomputable def is_odd_prime (n : ℕ) : Prop :=
  n > 1 ∧ Prime n ∧ n % 2 = 1

theorem least_x (x p : ℕ) (hp : Prime p) (hx : x > 0) (hodd_prime : is_odd_prime (x / (12 * p))) : x = 72 := 
  sorry

end least_x_l728_72867


namespace pipe_fills_tank_in_10_hours_l728_72841

variables (pipe_rate leak_rate : ℝ)

-- Conditions
def combined_rate := pipe_rate - leak_rate
def leak_time := 30
def combined_time := 15

-- Express leak_rate from leak_time
noncomputable def leak_rate_def : ℝ := 1 / leak_time

-- Express pipe_rate from combined_time with leak_rate considered
noncomputable def pipe_rate_def : ℝ := 1 / combined_time + leak_rate_def

-- Theorem to be proved
theorem pipe_fills_tank_in_10_hours :
  (1 / pipe_rate_def) = 10 :=
by
  sorry

end pipe_fills_tank_in_10_hours_l728_72841


namespace reflection_across_y_axis_coordinates_l728_72856

def coordinates_after_reflection (x y : ℤ) : ℤ × ℤ :=
  (-x, y)

theorem reflection_across_y_axis_coordinates :
  coordinates_after_reflection (-3) 4 = (3, 4) :=
by
  sorry

end reflection_across_y_axis_coordinates_l728_72856


namespace trapezoidal_garden_solutions_l728_72866

theorem trapezoidal_garden_solutions :
  ∃ (b1 b2 : ℕ), 
    (1800 = (60 * (b1 + b2)) / 2) ∧
    (b1 % 10 = 0) ∧ (b2 % 10 = 0) ∧
    (∃ (n : ℕ), n = 4) := 
sorry

end trapezoidal_garden_solutions_l728_72866


namespace consecutive_integer_quadratic_l728_72803

theorem consecutive_integer_quadratic :
  ∃ (a b c : ℤ) (x₁ x₂ : ℤ),
  (a * x₁ ^ 2 + b * x₁ + c = 0 ∧ a * x₂ ^ 2 + b * x₂ + c = 0) ∧
  (a = 2 ∧ b = 0 ∧ c = -2) ∨ (a = -2 ∧ b = 0 ∧ c = 2) := sorry

end consecutive_integer_quadratic_l728_72803


namespace invitations_sent_out_l728_72872

-- Define the conditions
def RSVPed (I : ℝ) : ℝ := 0.9 * I
def Showed_up (I : ℝ) : ℝ := 0.8 * RSVPed I
def No_gift : ℝ := 10
def Thank_you_cards : ℝ := 134

-- Prove the number of invitations
theorem invitations_sent_out : ∃ I : ℝ, Showed_up I - No_gift = Thank_you_cards ∧ I = 200 :=
by
  sorry

end invitations_sent_out_l728_72872


namespace point_on_xOz_plane_l728_72833

def point : ℝ × ℝ × ℝ := (1, 0, 4)

theorem point_on_xOz_plane : point.snd = 0 :=
by 
  -- Additional definitions and conditions might be necessary,
  -- but they should come directly from the problem statement:
  -- * Define conditions for being on the xOz plane.
  -- For the purpose of this example, we skip the proof.
  sorry

end point_on_xOz_plane_l728_72833


namespace surface_area_of_large_cube_correct_l728_72865

-- Definition of the surface area problem

def edge_length_of_small_cube := 3 -- centimeters
def number_of_small_cubes := 27
def surface_area_of_large_cube (edge_length_of_small_cube : ℕ) (number_of_small_cubes : ℕ) : ℕ :=
  let edge_length_of_large_cube := edge_length_of_small_cube * (number_of_small_cubes^(1/3))
  6 * edge_length_of_large_cube^2

theorem surface_area_of_large_cube_correct :
  surface_area_of_large_cube edge_length_of_small_cube number_of_small_cubes = 486 := by
  sorry

end surface_area_of_large_cube_correct_l728_72865


namespace second_caterer_cheaper_l728_72834

theorem second_caterer_cheaper (x : ℕ) :
  (150 + 18 * x > 250 + 14 * x) → x ≥ 26 :=
by
  intro h
  sorry

end second_caterer_cheaper_l728_72834


namespace geometric_sequence_n_l728_72879

theorem geometric_sequence_n (a : ℕ → ℝ) (n : ℕ) 
  (h1 : a 1 * a 2 * a 3 = 4) 
  (h2 : a 4 * a 5 * a 6 = 12) 
  (h3 : a (n-1) * a n * a (n+1) = 324) : 
  n = 14 := 
  sorry

end geometric_sequence_n_l728_72879


namespace min_value_x_plus_2y_l728_72844

open Real

theorem min_value_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 8 / x + 1 / y = 1) :
  x + 2 * y ≥ 16 :=
sorry

end min_value_x_plus_2y_l728_72844


namespace units_digit_product_odd_integers_10_to_110_l728_72826

-- Define the set of odd integer numbers between 10 and 110
def oddNumbersInRange : List ℕ := List.filter (fun n => n % 2 = 1) (List.range' 10 101)

-- Define the set of relevant odd multiples of 5 within the range
def oddMultiplesOfFive : List ℕ := List.filter (fun n => n % 5 = 0) oddNumbersInRange

-- Prove that the product of all odd positive integers between 10 and 110 has units digit 5
theorem units_digit_product_odd_integers_10_to_110 :
  let product : ℕ := List.foldl (· * ·) 1 oddNumbersInRange
  product % 10 = 5 :=
by
  sorry

end units_digit_product_odd_integers_10_to_110_l728_72826


namespace adult_ticket_cost_l728_72852

theorem adult_ticket_cost (C : ℝ) (h1 : ∀ (a : ℝ), a = C + 8)
  (h2 : ∀ (s : ℝ), s = C + 4)
  (h3 : 5 * C + 2 * (C + 8) + 2 * (C + 4) = 150) :
  ∃ (a : ℝ), a = 22 :=
by {
  sorry
}

end adult_ticket_cost_l728_72852


namespace express_vector_c_as_linear_combination_l728_72804

noncomputable def a : ℝ × ℝ := (1, 1)
noncomputable def b : ℝ × ℝ := (1, -1)
noncomputable def c : ℝ × ℝ := (2, 3)

theorem express_vector_c_as_linear_combination :
  ∃ x y : ℝ, c = (x * (1, 1).1 + y * (1, -1).1, x * (1, 1).2 + y * (1, -1).2) ∧
             x = 5 / 2 ∧ y = -1 / 2 :=
by
  sorry

end express_vector_c_as_linear_combination_l728_72804


namespace find_x_l728_72847

theorem find_x (p q r x : ℝ) (h1 : (p + q + r) / 3 = 4) (h2 : (p + q + r + x) / 4 = 5) : x = 8 :=
sorry

end find_x_l728_72847


namespace solve_for_x_l728_72876

theorem solve_for_x (x : ℚ) (h : (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2) : x = -7 / 6 :=
sorry

end solve_for_x_l728_72876


namespace find_a_l728_72812

theorem find_a (a : ℝ) : (∃ x y : ℝ, 3 * x + a * y - 5 = 0 ∧ x = 1 ∧ y = 2) → a = 1 :=
by
  intro h
  match h with
  | ⟨x, y, hx, hx1, hy2⟩ => 
    have h1 : x = 1 := hx1
    have h2 : y = 2 := hy2
    rw [h1, h2] at hx
    sorry

end find_a_l728_72812


namespace reena_interest_paid_l728_72815

-- Definitions based on conditions
def principal : ℝ := 1200
def rate : ℝ := 0.03
def time : ℝ := 3

-- Definition of simple interest calculation based on conditions
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- Statement to prove that Reena paid $108 as interest
theorem reena_interest_paid : simple_interest principal rate time = 108 := by
  sorry

end reena_interest_paid_l728_72815


namespace total_is_83_l728_72813

def number_of_pirates := 45
def number_of_noodles := number_of_pirates - 7
def total_number_of_noodles_and_pirates := number_of_noodles + number_of_pirates

theorem total_is_83 : total_number_of_noodles_and_pirates = 83 := by
  sorry

end total_is_83_l728_72813


namespace ab_equals_five_l728_72832

variable (a m b n : ℝ)

def arithmetic_seq (x y z : ℝ) : Prop :=
  2 * y = x + z

def geometric_seq (w x y z u : ℝ) : Prop :=
  x * x = w * y ∧ y * y = x * z ∧ z * z = y * u

theorem ab_equals_five
  (h1 : arithmetic_seq (-9) a (-1))
  (h2 : geometric_seq (-9) m b n (-1)) :
  a * b = 5 := sorry

end ab_equals_five_l728_72832


namespace triangle_inequality_range_isosceles_triangle_perimeter_l728_72820

-- Define the parameters for the triangle
variables (AB BC AC a : ℝ)
variables (h_AB : AB = 8) (h_BC : BC = 2 * a + 2) (h_AC : AC = 22)

-- Define the lean proof problem for the given conditions
theorem triangle_inequality_range (h_triangle : AB = 8 ∧ BC = 2 * a + 2 ∧ AC = 22) :
  6 < a ∧ a < 14 := sorry

-- Define the isosceles condition and perimeter calculation
theorem isosceles_triangle_perimeter (h_isosceles : BC = AC) :
  perimeter = 52 := sorry

end triangle_inequality_range_isosceles_triangle_perimeter_l728_72820


namespace factorial_divisibility_l728_72851

theorem factorial_divisibility {n : ℕ} (h : 2011^(2011) ∣ n!) : 2011^(2012) ∣ n! :=
sorry

end factorial_divisibility_l728_72851


namespace cos_double_plus_cos_l728_72874

theorem cos_double_plus_cos (α : ℝ) (h : Real.sin (Real.pi / 2 + α) = 1 / 3) :
  Real.cos (2 * α) + Real.cos α = -4 / 9 :=
by
  sorry

end cos_double_plus_cos_l728_72874


namespace problem_statement_l728_72835

variable (a b x : ℝ)

theorem problem_statement (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) : 
  a / (a - b) = x / (x - 1) :=
sorry

end problem_statement_l728_72835


namespace power_mod_result_l728_72805

theorem power_mod_result :
  (47 ^ 1235 - 22 ^ 1235) % 8 = 7 := by
  sorry

end power_mod_result_l728_72805


namespace rectangle_area_error_l728_72875

/-
  Problem: 
  Given:
  1. One side of the rectangle is taken 20% in excess.
  2. The other side of the rectangle is taken 10% in deficit.
  Prove:
  The error percentage in the calculated area is 8%.
-/

noncomputable def error_percentage (L W : ℝ) := 
  let actual_area : ℝ := L * W
  let measured_length : ℝ := 1.20 * L
  let measured_width : ℝ := 0.90 * W
  let measured_area : ℝ := measured_length * measured_width
  ((measured_area - actual_area) / actual_area) * 100

theorem rectangle_area_error
  (L W : ℝ) : error_percentage L W = 8 := 
  sorry

end rectangle_area_error_l728_72875


namespace xy_product_approx_25_l728_72860

noncomputable def approx_eq (a b : ℝ) (ε : ℝ := 1e-6) : Prop :=
  |a - b| < ε

theorem xy_product_approx_25 (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) 
  (hxy : x / y = 36) (hy : y = 0.8333333333333334) : approx_eq (x * y) 25 :=
by
  sorry

end xy_product_approx_25_l728_72860


namespace odd_solutions_eq_iff_a_le_neg3_or_a_ge3_l728_72800

theorem odd_solutions_eq_iff_a_le_neg3_or_a_ge3 (a : ℝ) :
  (∃! x : ℝ, -1 ≤ x ∧ x ≤ 5 ∧ (a - 3 * x^2 + Real.cos (9 * Real.pi * x / 2)) * Real.sqrt (3 - a * x) = 0) ↔ (a ≤ -3 ∨ a ≥ 3) := 
by
  sorry

end odd_solutions_eq_iff_a_le_neg3_or_a_ge3_l728_72800


namespace fractions_lcm_l728_72855

noncomputable def lcm_of_fractions_lcm (numerators : List ℕ) (denominators : List ℕ) : ℕ :=
  let lcm_nums := numerators.foldr Nat.lcm 1
  let gcd_denom := denominators.foldr Nat.gcd (denominators.headD 1)
  lcm_nums / gcd_denom

theorem fractions_lcm (hnum : List ℕ := [4, 5, 7, 9, 13, 16, 19])
                      (hdenom : List ℕ := [9, 7, 15, 13, 21, 35, 45]) :
  lcm_of_fractions_lcm hnum hdenom = 1244880 :=
by
  sorry

end fractions_lcm_l728_72855


namespace maisy_earns_more_l728_72889

theorem maisy_earns_more 
    (current_hours : ℕ) (current_wage : ℕ) 
    (new_hours : ℕ) (new_wage : ℕ) (bonus : ℕ)
    (h_current_job : current_hours = 8) 
    (h_current_wage : current_wage = 10)
    (h_new_job : new_hours = 4) 
    (h_new_wage : new_wage = 15)
    (h_bonus : bonus = 35) :
  (new_hours * new_wage + bonus) - (current_hours * current_wage) = 15 := 
by 
  sorry

end maisy_earns_more_l728_72889


namespace alice_unanswered_questions_l728_72898

theorem alice_unanswered_questions :
  ∃ (c w u : ℕ), (5 * c - 2 * w = 54) ∧ (2 * c + u = 36) ∧ (c + w + u = 30) ∧ (u = 8) :=
by
  -- proof omitted
  sorry

end alice_unanswered_questions_l728_72898


namespace correct_calculation_for_A_l728_72899

theorem correct_calculation_for_A (x : ℝ) : (-2 * x) ^ 3 = -8 * x ^ 3 :=
by
  sorry

end correct_calculation_for_A_l728_72899


namespace selling_price_l728_72840

theorem selling_price (CP P : ℝ) (hCP : CP = 320) (hP : P = 0.25) : CP + (P * CP) = 400 :=
by
  sorry

end selling_price_l728_72840


namespace zilla_savings_l728_72854

theorem zilla_savings
  (monthly_earnings : ℝ)
  (h_rent : monthly_earnings * 0.07 = 133)
  (h_expenses : monthly_earnings * 0.5 = monthly_earnings / 2) :
  monthly_earnings - (133 + monthly_earnings / 2) = 817 :=
by
  sorry

end zilla_savings_l728_72854


namespace polynomial_roots_ratio_l728_72808

theorem polynomial_roots_ratio (a b c d : ℝ) (h₀ : a ≠ 0) 
    (h₁ : a * 64 + b * 16 + c * 4 + d = 0)
    (h₂ : -a + b - c + d = 0) : 
    (b + c) / a = -13 :=
by {
    sorry
}

end polynomial_roots_ratio_l728_72808


namespace blue_first_yellow_second_probability_l728_72801

open Classical

-- Definition of initial conditions
def total_marbles : Nat := 3 + 4 + 9
def blue_marbles : Nat := 3
def yellow_marbles : Nat := 4
def pink_marbles : Nat := 9

-- Probability functions
def probability_first_blue : ℚ := blue_marbles / total_marbles
def probability_second_yellow_given_blue : ℚ := yellow_marbles / (total_marbles - 1)

-- Combined probability
def combined_probability_first_blue_second_yellow : ℚ := 
  probability_first_blue * probability_second_yellow_given_blue

-- Theorem statement
theorem blue_first_yellow_second_probability :
  combined_probability_first_blue_second_yellow = 1 / 20 :=
by
  -- Proof will be provided here
  sorry

end blue_first_yellow_second_probability_l728_72801


namespace smallest_fraction_denominator_l728_72809

theorem smallest_fraction_denominator (p q : ℕ) :
  (1:ℚ) / 2014 < p / q ∧ p / q < (1:ℚ) / 2013 → q = 4027 :=
sorry

end smallest_fraction_denominator_l728_72809


namespace average_books_per_student_l728_72819

theorem average_books_per_student
  (total_students : ℕ)
  (students_0_books : ℕ)
  (students_1_book : ℕ)
  (students_2_books : ℕ)
  (students_at_least_3_books : ℕ)
  (total_students_eq : total_students = 38)
  (students_0_books_eq : students_0_books = 2)
  (students_1_book_eq : students_1_book = 12)
  (students_2_books_eq : students_2_books = 10)
  (students_at_least_3_books_eq : students_at_least_3_books = 14)
  (students_count_consistent : total_students = students_0_books + students_1_book + students_2_books + students_at_least_3_books) :
  (students_0_books * 0 + students_1_book * 1 + students_2_books * 2 + students_at_least_3_books * 3 : ℝ) / total_students = 1.947 :=
by
  sorry

end average_books_per_student_l728_72819


namespace painter_remaining_time_l728_72858

-- Define the initial conditions
def total_rooms : ℕ := 11
def hours_per_room : ℕ := 7
def painted_rooms : ℕ := 2

-- Define the remaining rooms to paint
def remaining_rooms : ℕ := total_rooms - painted_rooms

-- Define the proof problem: the remaining time to paint the rest of the rooms
def remaining_hours : ℕ := remaining_rooms * hours_per_room

theorem painter_remaining_time :
  remaining_hours = 63 :=
sorry

end painter_remaining_time_l728_72858


namespace abc_inequality_l728_72838

-- Required conditions and proof statement
theorem abc_inequality 
  {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b * c = 1 / 8) : 
  a^2 + b^2 + c^2 + a^2 * b^2 + a^2 * c^2 + b^2 * c^2 ≥ 15 / 16 := 
sorry

end abc_inequality_l728_72838


namespace geometric_seq_b6_l728_72827

variable {b : ℕ → ℝ}

theorem geometric_seq_b6 (h1 : b 3 * b 9 = 9) (h2 : ∃ r, ∀ n, b (n + 1) = r * b n) : b 6 = 3 ∨ b 6 = -3 :=
by
  sorry

end geometric_seq_b6_l728_72827


namespace minimum_solutions_in_interval_l728_72868

open Function Real

-- Define what it means for a function to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define what it means for a function to be periodic
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x

-- Main theorem statement
theorem minimum_solutions_in_interval :
  ∀ (f : ℝ → ℝ),
  is_even f → is_periodic f 3 → f 2 = 0 →
  (∃ x1 x2 x3 x4 : ℝ, 0 < x1 ∧ x1 < 6 ∧ f x1 = 0 ∧
                     0 < x2 ∧ x2 < 6 ∧ f x2 = 0 ∧
                     0 < x3 ∧ x3 < 6 ∧ f x3 = 0 ∧
                     0 < x4 ∧ x4 < 6 ∧ f x4 = 0 ∧
                     x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧
                     x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) :=
by
  sorry

end minimum_solutions_in_interval_l728_72868


namespace repeating_decimal_fraction_equiv_l728_72870

noncomputable def repeating_decimal_to_fraction (x : ℚ) : Prop :=
  x = 0.4 + 37 / 990

theorem repeating_decimal_fraction_equiv : repeating_decimal_to_fraction (433 / 990) :=
by
  sorry

end repeating_decimal_fraction_equiv_l728_72870


namespace total_students_l728_72891

-- Definition of the conditions given in the problem
def num5 : ℕ := 12
def num6 : ℕ := 6 * num5

-- The theorem representing the mathematically equivalent proof problem
theorem total_students : num5 + num6 = 84 :=
by
  sorry

end total_students_l728_72891


namespace selling_price_per_pound_is_correct_l728_72897

noncomputable def cost_of_40_lbs : ℝ := 40 * 0.38
noncomputable def cost_of_8_lbs : ℝ := 8 * 0.50
noncomputable def total_cost : ℝ := cost_of_40_lbs + cost_of_8_lbs
noncomputable def total_weight : ℝ := 40 + 8
noncomputable def profit : ℝ := total_cost * 0.20
noncomputable def total_selling_price : ℝ := total_cost + profit
noncomputable def selling_price_per_pound : ℝ := total_selling_price / total_weight

theorem selling_price_per_pound_is_correct :
  selling_price_per_pound = 0.48 :=
by
  sorry

end selling_price_per_pound_is_correct_l728_72897


namespace value_of_x_squared_y_plus_xy_squared_l728_72885

variable {R : Type} [CommRing R] (x y : R)

-- Given conditions
def cond1 : Prop := x + y = 3
def cond2 : Prop := x * y = 2

-- The main theorem to prove
theorem value_of_x_squared_y_plus_xy_squared (h1 : cond1 x y) (h2 : cond2 x y) : x^2 * y + x * y^2 = 6 :=
by
  sorry

end value_of_x_squared_y_plus_xy_squared_l728_72885


namespace taco_truck_earnings_l728_72871

/-
Question: How many dollars did the taco truck make during the lunch rush?
Conditions:
1. Soft tacos are $2 each.
2. Hard shell tacos are $5 each.
3. The family buys 4 hard shell tacos and 3 soft tacos.
4. There are ten other customers.
5. Each of the ten other customers buys 2 soft tacos.
Answer: The taco truck made $66 during the lunch rush.
-/

theorem taco_truck_earnings :
  let soft_taco_price := 2
  let hard_taco_price := 5
  let family_hard_tacos := 4
  let family_soft_tacos := 3
  let other_customers := 10
  let other_customers_soft_tacos := 2
  (family_hard_tacos * hard_taco_price + family_soft_tacos * soft_taco_price +
   other_customers * other_customers_soft_tacos * soft_taco_price) = 66 := by
  sorry

end taco_truck_earnings_l728_72871


namespace train_length_l728_72896

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h1 : speed_km_hr = 52) (h2 : time_sec = 9) (h3 : length_m = 129.96) : 
  length_m = (speed_km_hr * 1000 / 3600) * time_sec := 
sorry

end train_length_l728_72896


namespace correct_relation_l728_72842

def A : Set ℝ := { x | x > 1 }

theorem correct_relation : 2 ∈ A := by
  -- Proof would go here
  sorry

end correct_relation_l728_72842


namespace adult_meal_cost_l728_72895

theorem adult_meal_cost (x : ℝ) 
  (total_people : ℕ) (kids : ℕ) (total_cost : ℝ)  
  (h_total_people : total_people = 11) 
  (h_kids : kids = 2) 
  (h_total_cost : total_cost = 72)
  (h_adult_meals : (total_people - kids : ℕ) • x = total_cost) : 
  x = 8 := 
by
  -- Proof will go here
  sorry

end adult_meal_cost_l728_72895


namespace Aaron_final_cards_l728_72810

-- Definitions from conditions
def initial_cards_Aaron : Nat := 5
def found_cards_Aaron : Nat := 62

-- Theorem statement
theorem Aaron_final_cards : initial_cards_Aaron + found_cards_Aaron = 67 :=
by
  sorry

end Aaron_final_cards_l728_72810


namespace find_square_side_l728_72894

theorem find_square_side (a b x : ℕ) (h_triangle : a^2 + x^2 = b^2)
  (h_trapezoid : 2 * a + 2 * b + 2 * x = 60)
  (h_rectangle : 4 * a + 2 * x = 58) :
  a = 12 := by
  sorry

end find_square_side_l728_72894


namespace sin_double_angle_fourth_quadrant_l728_72881

theorem sin_double_angle_fourth_quadrant (α : ℝ) (k : ℤ) (h : -π/2 + 2*k*π < α ∧ α < 2*k*π) : Real.sin (2 * α) < 0 := 
sorry

end sin_double_angle_fourth_quadrant_l728_72881


namespace first_platform_length_l728_72878

noncomputable def length_of_first_platform (t1 t2 l_train l_plat2 time1 time2 : ℕ) : ℕ :=
  let s1 := (l_train + t1) / time1
  let s2 := (l_train + l_plat2) / time2
  if s1 = s2 then t1 else 0

theorem first_platform_length:
  ∀ (time1 time2 : ℕ) (l_train l_plat2 : ℕ), time1 = 15 → time2 = 20 → l_train = 350 → l_plat2 = 250 → length_of_first_platform 100 l_plat2 l_train l_plat2 time1 time2 = 100 :=
by
  intros time1 time2 l_train l_plat2 ht1 ht2 ht3 ht4
  rw [ht1, ht2, ht3, ht4]
  dsimp [length_of_first_platform]
  rfl

end first_platform_length_l728_72878


namespace k_value_l728_72830

theorem k_value (k : ℝ) (x : ℝ) (y : ℝ) (hk : k^2 - 5 = -1) (hx : x > 0) (hy : y = (k - 1) * x^(k^2 - 5)) (h_dec : ∀ (x1 x2 : ℝ), x1 > 0 → x2 > x1 → (k - 1) * x2^(k^2 - 5) < (k - 1) * x1^(k^2 - 5)):
  k = 2 := by
  sorry

end k_value_l728_72830


namespace sitting_break_frequency_l728_72863

theorem sitting_break_frequency (x : ℕ) (h1 : 240 % x = 0) (h2 : 240 / 20 = 12) (h3 : 240 / x + 10 = 12) : x = 120 := 
sorry

end sitting_break_frequency_l728_72863


namespace triangle_area_division_l728_72893

theorem triangle_area_division (T T_1 T_2 T_3 : ℝ) 
  (hT1_pos : 0 < T_1) (hT2_pos : 0 < T_2) (hT3_pos : 0 < T_3) (hT : T = T_1 + T_2 + T_3) :
  T = (Real.sqrt T_1 + Real.sqrt T_2 + Real.sqrt T_3) ^ 2 :=
sorry

end triangle_area_division_l728_72893


namespace range_of_values_l728_72821

noncomputable def f (x : ℝ) : ℝ := 2^(1 + x^2) - 1 / (1 + x^2)

theorem range_of_values (x : ℝ) : f (2 * x) > f (x - 3) ↔ x < -3 ∨ x > 1 := 
by
  sorry

end range_of_values_l728_72821


namespace area_of_triangle_formed_by_lines_l728_72829

def line1 (x : ℝ) : ℝ := 5
def line2 (x : ℝ) : ℝ := 1 + x
def line3 (x : ℝ) : ℝ := 1 - x

theorem area_of_triangle_formed_by_lines :
  let A := (4, 5)
  let B := (-4, 5)
  let C := (0, 1)
  (1 / 2) * abs (4 * 5 + (-4) * 1 + 0 * 5 - (5 * (-4) + 1 * 4 + 5 * 0)) = 16 := by
  sorry

end area_of_triangle_formed_by_lines_l728_72829


namespace emma_troy_wrapping_time_l728_72849

theorem emma_troy_wrapping_time (emma_rate troy_rate total_task_time together_time emma_remaining_time : ℝ) 
  (h1 : emma_rate = 1 / 6) 
  (h2 : troy_rate = 1 / 8) 
  (h3 : total_task_time = 1) 
  (h4 : together_time = 2) 
  (h5 : emma_remaining_time = (total_task_time - (emma_rate + troy_rate) * together_time) / emma_rate) : 
  emma_remaining_time = 2.5 := 
sorry

end emma_troy_wrapping_time_l728_72849


namespace length_of_ship_l728_72822

-- Variables and conditions
variables (E L S : ℝ)
variables (W : ℝ := 0.9) -- Wind reducing factor

-- Conditions as equations
def condition1 : Prop := 150 * E = L + 150 * S
def condition2 : Prop := 70 * E = L - 63 * S

-- Theorem to prove
theorem length_of_ship (hc1 : condition1 E L S) (hc2 : condition2 E L S) : L = (19950 / 213) * E :=
sorry

end length_of_ship_l728_72822


namespace range_of_a_l728_72884

variable (a : ℝ)

def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x ^ 2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x ^ 2 + 2 * a * x + 2 - a = 0

theorem range_of_a (h₁ : p a) (h₂ : q a) : a ≤ -2 ∨ a = 1 := 
sorry

end range_of_a_l728_72884


namespace find_y_for_two_thirds_l728_72873

theorem find_y_for_two_thirds (x y : ℝ) (h₁ : (2 / 3) * x + y = 10) (h₂ : x = 6) : y = 6 :=
by
  sorry

end find_y_for_two_thirds_l728_72873


namespace cos_beta_acos_l728_72816

theorem cos_beta_acos {α β : ℝ} (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_cos_α : Real.cos α = 1 / 7) (h_cos_sum : Real.cos (α + β) = -11 / 14) :
  Real.cos β = 1 / 2 := by
  sorry

end cos_beta_acos_l728_72816


namespace remainder_of_sum_of_integers_l728_72824

theorem remainder_of_sum_of_integers (a b c : ℕ)
  (h₁ : a % 30 = 15) (h₂ : b % 30 = 5) (h₃ : c % 30 = 10) :
  (a + b + c) % 30 = 0 := by
  sorry

end remainder_of_sum_of_integers_l728_72824


namespace time_to_cover_escalator_l728_72892

def escalator_speed := 11 -- ft/sec
def escalator_length := 126 -- feet
def person_speed := 3 -- ft/sec

theorem time_to_cover_escalator :
  (escalator_length / (escalator_speed + person_speed)) = 9 := by
  sorry

end time_to_cover_escalator_l728_72892


namespace porter_monthly_earnings_l728_72836

def daily_rate : ℕ := 8

def regular_days : ℕ := 5

def extra_day_rate : ℕ := daily_rate * 3 / 2  -- 50% increase on the daily rate

def weekly_earnings_with_overtime : ℕ := (daily_rate * regular_days) + extra_day_rate

def weeks_in_month : ℕ := 4

theorem porter_monthly_earnings : weekly_earnings_with_overtime * weeks_in_month = 208 :=
by
  sorry

end porter_monthly_earnings_l728_72836


namespace johns_original_earnings_l728_72831

-- Definitions from conditions
variables (x : ℝ) (raise_percentage : ℝ) (new_salary : ℝ)

-- Conditions
def conditions : Prop :=
  raise_percentage = 0.25 ∧ new_salary = 75 ∧ x + raise_percentage * x = new_salary

-- Theorem statement
theorem johns_original_earnings (h : conditions x 0.25 75) : x = 60 :=
sorry

end johns_original_earnings_l728_72831


namespace part_I_part_II_l728_72848

noncomputable def f (x : ℝ) : ℝ := abs (x - 2) + abs (x + 1)

theorem part_I (x : ℝ) : f x > 4 ↔ x < -1.5 ∨ x > 2.5 := 
sorry

theorem part_II (a : ℝ) : (∀ x, f x ≥ a) ↔ a ≤ 3 := 
sorry

end part_I_part_II_l728_72848


namespace susie_total_savings_is_correct_l728_72811

variable (initial_amount : ℝ) (year1_addition_pct : ℝ) (year2_addition_pct : ℝ) (interest_rate : ℝ)

def susies_savings (initial_amount year1_addition_pct year2_addition_pct interest_rate : ℝ) : ℝ :=
  let end_of_first_year := initial_amount + initial_amount * year1_addition_pct
  let first_year_interest := end_of_first_year * interest_rate
  let total_after_first_year := end_of_first_year + first_year_interest
  let end_of_second_year := total_after_first_year + total_after_first_year * year2_addition_pct
  let second_year_interest := end_of_second_year * interest_rate
  end_of_second_year + second_year_interest

theorem susie_total_savings_is_correct : 
  susies_savings 200 0.20 0.30 0.05 = 343.98 := 
by
  sorry

end susie_total_savings_is_correct_l728_72811


namespace MrMartinSpent_l728_72802

theorem MrMartinSpent : 
  ∀ (C B : ℝ), 
    3 * C + 2 * B = 12.75 → 
    B = 1.5 → 
    2 * C + 5 * B = 14 := 
by
  intros C B h1 h2
  sorry

end MrMartinSpent_l728_72802


namespace exists_person_with_girls_as_neighbors_l728_72877

theorem exists_person_with_girls_as_neighbors (boys girls : Nat) (sitting : Nat) 
  (h_boys : boys = 25) (h_girls : girls = 25) (h_sitting : sitting = boys + girls) :
  ∃ p : Nat, p < sitting ∧ (p % 2 = 1 → p.succ % sitting % 2 = 0) := 
by
  sorry

end exists_person_with_girls_as_neighbors_l728_72877


namespace pension_equality_l728_72880

theorem pension_equality (x c d r s: ℝ) (h₁ : d ≠ c) 
    (h₂ : x > 0) (h₃ : 2 * x * (d - c) + d^2 - c^2 ≠ 0)
    (h₄ : ∀ k:ℝ, k * (x + c)^2 - k * x^2 = r)
    (h₅ : ∀ k:ℝ, k * (x + d)^2 - k * x^2 = s) 
    : ∃ k : ℝ, k = (s - r) / (2 * x * (d - c) + d^2 - c^2) 
    → k * x^2 = (s - r) * x^2 / (2 * x * (d - c) + d^2 - c^2) :=
by {
    sorry
}

end pension_equality_l728_72880


namespace find_x_from_w_condition_l728_72886

theorem find_x_from_w_condition :
  ∀ (x u y z w : ℕ), 
  (x = u + 7) → 
  (u = y + 5) → 
  (y = z + 12) → 
  (z = w + 25) → 
  (w = 100) → 
  x = 149 :=
by intros x u y z w h1 h2 h3 h4 h5
   sorry

end find_x_from_w_condition_l728_72886


namespace intersection_of_A_and_B_l728_72861

theorem intersection_of_A_and_B :
  let A := {x : ℝ | x > 0}
  let B := {x : ℝ | x^2 - 2*x - 3 < 0}
  (A ∩ B) = {x : ℝ | 0 < x ∧ x < 3} := by
  sorry

end intersection_of_A_and_B_l728_72861


namespace angle_bisectors_l728_72807

open Real

noncomputable def r1 : ℝ × ℝ × ℝ := (1, 1, 0)
noncomputable def r2 : ℝ × ℝ × ℝ := (0, 1, 1)

theorem angle_bisectors :
  ∃ (phi : ℝ), 0 ≤ phi ∧ phi ≤ π ∧ cos phi = 1 / 2 :=
sorry

end angle_bisectors_l728_72807


namespace max_value_l728_72828

theorem max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  ∃ m ≤ 4, ∀ (z w : ℝ), z > 0 → w > 0 → (x + y = z + w) → (z^3 + w^3 ≥ x^3 + y^3 → 
  (z + w)^3 / (z^3 + w^3) ≤ m) :=
sorry

end max_value_l728_72828


namespace sqrt_sub_sqrt_frac_eq_l728_72818

theorem sqrt_sub_sqrt_frac_eq : (Real.sqrt 3) - (Real.sqrt (1 / 3)) = (2 * Real.sqrt 3) / 3 := 
by 
  sorry

end sqrt_sub_sqrt_frac_eq_l728_72818


namespace monotonic_on_interval_l728_72859

theorem monotonic_on_interval (k : ℝ) :
  (∀ x y : ℝ, x ≤ y → x ≤ 8 → y ≤ 8 → (4 * x ^ 2 - k * x - 8) ≤ (4 * y ^ 2 - k * y - 8)) ↔ (64 ≤ k) :=
sorry

end monotonic_on_interval_l728_72859


namespace bakery_storage_l728_72890

theorem bakery_storage (S F B : ℕ) (h1 : S * 8 = 3 * F) (h2 : F * 1 = 10 * B) (h3 : F * 1 = 8 * (B + 60)) : S = 900 :=
by
  -- We would normally put the proof steps here, but since it's specified to include only the statement
  sorry

end bakery_storage_l728_72890


namespace converse_proposition_false_l728_72887

theorem converse_proposition_false (a b c : ℝ) : ¬(∀ a b c : ℝ, (a > b) → (a * c^2 > b * c^2)) :=
by {
  -- proof goes here
  sorry
}

end converse_proposition_false_l728_72887


namespace inequality_cube_of_greater_l728_72850

variable {a b : ℝ}

theorem inequality_cube_of_greater (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a > b) : a^3 > b^3 :=
sorry

end inequality_cube_of_greater_l728_72850


namespace treShaun_marker_ink_left_l728_72843

noncomputable def ink_left_percentage (marker_area : ℕ) (total_colored_area : ℕ) : ℕ :=
if total_colored_area >= marker_area then 0 else ((marker_area - total_colored_area) * 100) / marker_area

theorem treShaun_marker_ink_left :
  let marker_area := 3 * (4 * 4)
  let colored_area := (2 * (6 * 2) + 8 * 4)
  ink_left_percentage marker_area colored_area = 0 :=
by
  sorry

end treShaun_marker_ink_left_l728_72843


namespace a_minus_b_is_15_l728_72862

variables (a b c : ℝ)

-- Conditions from the problem statement
axiom cond1 : a = 1/3 * (b + c)
axiom cond2 : b = 2/7 * (a + c)
axiom cond3 : a + b + c = 540

-- The theorem we need to prove
theorem a_minus_b_is_15 : a - b = 15 :=
by
  sorry

end a_minus_b_is_15_l728_72862


namespace rectangular_table_capacity_l728_72823

variable (R : ℕ) -- The number of pupils a rectangular table can seat

-- Conditions
variable (rectangular_tables : ℕ)
variable (square_tables : ℕ)
variable (square_table_capacity : ℕ)
variable (total_pupils : ℕ)

-- Setting the values based on the conditions
axiom h1 : rectangular_tables = 7
axiom h2 : square_tables = 5
axiom h3 : square_table_capacity = 4
axiom h4 : total_pupils = 90

-- The proof statement
theorem rectangular_table_capacity :
  7 * R + 5 * 4 = 90 → R = 10 :=
by
  intro h
  sorry

end rectangular_table_capacity_l728_72823


namespace jill_spending_on_clothing_l728_72869

theorem jill_spending_on_clothing (C : ℝ) (T : ℝ)
  (h1 : 0.2 * T = 0.2 * T)
  (h2 : 0.3 * T = 0.3 * T)
  (h3 : (C / 100) * T * 0.04 + 0.3 * T * 0.08 = 0.044 * T) :
  C = 50 :=
by
  -- This line indicates the point where the proof would typically start
  sorry

end jill_spending_on_clothing_l728_72869


namespace ratio_of_place_values_l728_72837

-- Definitions based on conditions
def place_value_tens_digit : ℝ := 10
def place_value_hundredths_digit : ℝ := 0.01

-- Statement to prove
theorem ratio_of_place_values :
  (place_value_tens_digit / place_value_hundredths_digit) = 1000 :=
by
  sorry

end ratio_of_place_values_l728_72837


namespace minimum_value_of_option_C_l728_72857

noncomputable def optionA (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def optionB (x : ℝ) : ℝ := |Real.sin x| + 4 / |Real.sin x|
noncomputable def optionC (x : ℝ) : ℝ := 2^x + 2^(2 - x)
noncomputable def optionD (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem minimum_value_of_option_C :
  ∃ x : ℝ, optionC x = 4 :=
by
  sorry

end minimum_value_of_option_C_l728_72857


namespace cara_sitting_pairs_l728_72888

theorem cara_sitting_pairs : ∀ (n : ℕ), n = 7 → ∃ (pairs : ℕ), pairs = 6 :=
by
  intros n hn
  have h : n - 1 = 6 := sorry
  exact ⟨n - 1, h⟩

end cara_sitting_pairs_l728_72888


namespace sum_of_number_and_reverse_l728_72817

def digit_representation (n m : ℕ) (a b : ℕ) :=
  n = 10 * a + b ∧
  m = 10 * b + a ∧
  n - m = 9 * (a * b) + 3

theorem sum_of_number_and_reverse :
  ∃ a b n m : ℕ, digit_representation n m a b ∧ n + m = 22 :=
by
  sorry

end sum_of_number_and_reverse_l728_72817


namespace combined_mpg_l728_72839

theorem combined_mpg (miles_alice : ℕ) (mpg_alice : ℕ) (miles_bob : ℕ) (mpg_bob : ℕ) :
  miles_alice = 120 ∧ mpg_alice = 30 ∧ miles_bob = 180 ∧ mpg_bob = 20 →
  (miles_alice + miles_bob) / ((miles_alice / mpg_alice) + (miles_bob / mpg_bob)) = 300 / 13 :=
by
  intros h
  sorry

end combined_mpg_l728_72839


namespace nth_equation_holds_l728_72845

theorem nth_equation_holds (n : ℕ) (h : 0 < n) :
  1 / (n + 2) + 2 / (n^2 + 2 * n) = 1 / n :=
by
  sorry

end nth_equation_holds_l728_72845


namespace each_person_pays_50_97_l728_72814

noncomputable def total_bill (original_bill : ℝ) (tip_percentage : ℝ) : ℝ :=
  original_bill + original_bill * tip_percentage

noncomputable def amount_per_person (total_bill : ℝ) (num_people : ℕ) : ℝ :=
  total_bill / num_people

theorem each_person_pays_50_97 :
  let original_bill := 139.00
  let number_of_people := 3
  let tip_percentage := 0.10
  let expected_amount := 50.97
  abs (amount_per_person (total_bill original_bill tip_percentage) number_of_people - expected_amount) < 0.01
:= sorry

end each_person_pays_50_97_l728_72814


namespace no_x_for_rational_sin_cos_l728_72864

-- Define rational predicate
def is_rational (r : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ r = a / b

-- Define the statement of the problem
theorem no_x_for_rational_sin_cos :
  ∀ x : ℝ, ¬ (is_rational (Real.sin x + Real.sqrt 2) ∧ is_rational (Real.cos x - Real.sqrt 2)) :=
by
  -- Placeholder for proof
  sorry

end no_x_for_rational_sin_cos_l728_72864


namespace max_flowers_used_min_flowers_used_l728_72846

-- Part (a) Setup
def max_flowers (C M : ℕ) (flower_views : ℕ) := 2 * C + M = flower_views
def max_T (C M : ℕ) := C + M

-- Given conditions
theorem max_flowers_used :
  (∀ C M : ℕ, max_flowers C M 36 → max_T C M = 36) :=
by sorry

-- Part (b) Setup
def min_flowers (C M : ℕ) (flower_views : ℕ) := 2 * C + M = flower_views
def min_T (C M : ℕ) := C + M

-- Given conditions
theorem min_flowers_used :
  (∀ C M : ℕ, min_flowers C M 48 → min_T C M = 24) :=
by sorry

end max_flowers_used_min_flowers_used_l728_72846


namespace number_of_teams_l728_72853

theorem number_of_teams (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
by
  sorry

end number_of_teams_l728_72853


namespace six_digit_number_multiple_of_7_l728_72825

theorem six_digit_number_multiple_of_7 (d : ℕ) (hd : d ≤ 9) :
  (∃ k : ℤ, 56782 + d * 10 = 7 * k) ↔ (d = 0 ∨ d = 7) := by
sorry

end six_digit_number_multiple_of_7_l728_72825


namespace min_value_x_plus_3y_min_value_xy_l728_72882

variable {x y : ℝ}

theorem min_value_x_plus_3y (hx : x > 0) (hy : y > 0) (h : 1/x + 3/y = 1) : x + 3 * y ≥ 16 :=
sorry

theorem min_value_xy (hx : x > 0) (hy : y > 0) (h : 1/x + 3/y = 1) : x * y ≥ 12 :=
sorry

end min_value_x_plus_3y_min_value_xy_l728_72882


namespace middle_number_is_9_point_5_l728_72883

theorem middle_number_is_9_point_5 (x y z : ℝ) 
  (h1 : x + y = 15) (h2 : x + z = 18) (h3 : y + z = 22) : y = 9.5 := 
by {
  sorry
}

end middle_number_is_9_point_5_l728_72883


namespace Megan_not_lead_plays_l728_72806

-- Define the problem's conditions as variables
def total_plays : ℕ := 100
def lead_play_ratio : ℤ := 80

-- Define the proposition we want to prove
theorem Megan_not_lead_plays : 
  (total_plays - (total_plays * lead_play_ratio / 100)) = 20 := 
by sorry

end Megan_not_lead_plays_l728_72806
