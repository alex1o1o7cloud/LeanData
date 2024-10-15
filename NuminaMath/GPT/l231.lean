import Mathlib

namespace NUMINAMATH_GPT_solve_inequality_l231_23149

theorem solve_inequality :
  {x : ℝ | 8*x^3 - 6*x^2 + 5*x - 5 < 0} = {x : ℝ | x < 1/2} :=
sorry

end NUMINAMATH_GPT_solve_inequality_l231_23149


namespace NUMINAMATH_GPT_evaluate_product_l231_23192

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem evaluate_product : 
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) *
  (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) = 2657205 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_product_l231_23192


namespace NUMINAMATH_GPT_big_SUV_wash_ratio_l231_23130

-- Defining constants for time taken for various parts of the car
def time_windows : ℕ := 4
def time_body : ℕ := 7
def time_tires : ℕ := 4
def time_waxing : ℕ := 9

-- Time taken to wash one normal car
def time_normal_car : ℕ := time_windows + time_body + time_tires + time_waxing

-- Given total time William spent washing all vehicles
def total_time : ℕ := 96

-- Time taken for two normal cars
def time_two_normal_cars : ℕ := 2 * time_normal_car

-- Time taken for the big SUV
def time_big_SUV : ℕ := total_time - time_two_normal_cars

-- Ratio of time taken to wash the big SUV to the time taken to wash a normal car
def time_ratio : ℕ := time_big_SUV / time_normal_car

theorem big_SUV_wash_ratio : time_ratio = 2 := by
  sorry

end NUMINAMATH_GPT_big_SUV_wash_ratio_l231_23130


namespace NUMINAMATH_GPT_numeral_of_place_face_value_difference_l231_23199

theorem numeral_of_place_face_value_difference (P F : ℕ) (H : P - F = 63) (Hface : F = 7) : P = 70 :=
sorry

end NUMINAMATH_GPT_numeral_of_place_face_value_difference_l231_23199


namespace NUMINAMATH_GPT_Megan_pays_correct_amount_l231_23172

def original_price : ℝ := 22
def discount : ℝ := 6
def amount_paid := original_price - discount

theorem Megan_pays_correct_amount : amount_paid = 16 := by
  sorry

end NUMINAMATH_GPT_Megan_pays_correct_amount_l231_23172


namespace NUMINAMATH_GPT_shaded_area_floor_l231_23182

noncomputable def area_of_white_quarter_circle : ℝ := Real.pi / 4

noncomputable def area_of_white_per_tile : ℝ := 4 * area_of_white_quarter_circle

noncomputable def area_of_tile : ℝ := 4

noncomputable def shaded_area_per_tile : ℝ := area_of_tile - area_of_white_per_tile

noncomputable def number_of_tiles : ℕ := by
  have floor_area : ℝ := 12 * 15
  have tile_area : ℝ := 2 * 2
  exact Nat.floor (floor_area / tile_area)

noncomputable def total_shaded_area (num_tiles : ℕ) : ℝ := num_tiles * shaded_area_per_tile

theorem shaded_area_floor : total_shaded_area number_of_tiles = 180 - 45 * Real.pi := by
  sorry

end NUMINAMATH_GPT_shaded_area_floor_l231_23182


namespace NUMINAMATH_GPT_average_speed_ratio_l231_23142

theorem average_speed_ratio (t_E t_F : ℝ) (d_B d_C : ℝ) (htE : t_E = 3) (htF : t_F = 4) (hdB : d_B = 450) (hdC : d_C = 300) :
  (d_B / t_E) / (d_C / t_F) = 2 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_ratio_l231_23142


namespace NUMINAMATH_GPT_find_smallest_even_number_l231_23131

theorem find_smallest_even_number (n : ℕ) (h : n + (n + 2) + (n + 4) = 162) : n = 52 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_even_number_l231_23131


namespace NUMINAMATH_GPT_candies_bought_l231_23195

theorem candies_bought :
  ∃ (S C : ℕ), S + C = 8 ∧ 300 * S + 500 * C = 3000 ∧ C = 3 :=
by
  sorry

end NUMINAMATH_GPT_candies_bought_l231_23195


namespace NUMINAMATH_GPT_expected_total_rainfall_over_week_l231_23185

noncomputable def daily_rain_expectation : ℝ :=
  (0.5 * 0) + (0.2 * 2) + (0.3 * 5)

noncomputable def total_rain_expectation (days: ℕ) : ℝ :=
  days * daily_rain_expectation

theorem expected_total_rainfall_over_week : total_rain_expectation 7 = 13.3 :=
by 
  -- calculation of expected value here
  -- daily_rain_expectation = 1.9
  -- total_rain_expectation 7 = 7 * 1.9 = 13.3
  sorry

end NUMINAMATH_GPT_expected_total_rainfall_over_week_l231_23185


namespace NUMINAMATH_GPT_Billy_weighs_more_l231_23158

-- Variables and assumptions
variable (Billy Brad Carl : ℕ)
variable (b_weight : Billy = 159)
variable (c_weight : Carl = 145)
variable (brad_formula : Brad = Carl + 5)

-- Theorem statement to prove the required condition
theorem Billy_weighs_more :
  Billy - Brad = 9 :=
by
  -- Here we put the proof steps, but it's omitted as per instructions.
  sorry

end NUMINAMATH_GPT_Billy_weighs_more_l231_23158


namespace NUMINAMATH_GPT_money_left_in_wallet_l231_23140

def initial_amount := 106
def spent_supermarket := 31
def spent_showroom := 49

theorem money_left_in_wallet : initial_amount - spent_supermarket - spent_showroom = 26 := by
  sorry

end NUMINAMATH_GPT_money_left_in_wallet_l231_23140


namespace NUMINAMATH_GPT_girls_doctors_percentage_l231_23181

-- Define the total number of students in the class
variables (total_students : ℕ)

-- Define the proportions given in the problem
def proportion_boys : ℚ := 3 / 5
def proportion_boys_who_want_to_be_doctors : ℚ := 1 / 3
def proportion_doctors_who_are_boys : ℚ := 2 / 5

-- Compute the proportion of boys in the class who want to be doctors
def proportion_boys_as_doctors := proportion_boys * proportion_boys_who_want_to_be_doctors

-- Compute the proportion of girls in the class
def proportion_girls := 1 - proportion_boys

-- Compute the number of girls who want to be doctors compared to boys
def proportion_girls_as_doctors := (1 - proportion_doctors_who_are_boys) / proportion_doctors_who_are_boys * proportion_boys_as_doctors

-- Compute the proportion of girls who want to be doctors
def proportion_girls_who_want_to_be_doctors := proportion_girls_as_doctors / proportion_girls

-- Define the expected percentage of girls who want to be doctors
def expected_percentage_girls_who_want_to_be_doctors : ℚ := 75 / 100

-- The theorem we need to prove
theorem girls_doctors_percentage : proportion_girls_who_want_to_be_doctors * 100 = expected_percentage_girls_who_want_to_be_doctors :=
sorry

end NUMINAMATH_GPT_girls_doctors_percentage_l231_23181


namespace NUMINAMATH_GPT_election_majority_l231_23174

theorem election_majority (V : ℝ) 
  (h1 : ∃ w l : ℝ, w = 0.70 * V ∧ l = 0.30 * V ∧ w - l = 174) : 
  V = 435 :=
by
  sorry

end NUMINAMATH_GPT_election_majority_l231_23174


namespace NUMINAMATH_GPT_cylinder_surface_area_minimization_l231_23141

theorem cylinder_surface_area_minimization (S V r h : ℝ) (h₁ : π * r^2 * h = V) (h₂ : r^2 + (h / 2)^2 = S^2) : (h / r) = 2 :=
sorry

end NUMINAMATH_GPT_cylinder_surface_area_minimization_l231_23141


namespace NUMINAMATH_GPT_square_area_l231_23112

theorem square_area (XY ZQ : ℕ) (inscribed_square : Prop) : (XY = 35) → (ZQ = 65) → inscribed_square → ∃ (a : ℕ), a^2 = 2275 :=
by
  intros hXY hZQ hinscribed
  use 2275
  sorry

end NUMINAMATH_GPT_square_area_l231_23112


namespace NUMINAMATH_GPT_candy_per_bag_correct_l231_23120

def total_candy : ℕ := 648
def sister_candy : ℕ := 48
def friends : ℕ := 3
def bags : ℕ := 8

def remaining_candy (total candy_kept : ℕ) : ℕ := total - candy_kept
def candy_per_person (remaining people : ℕ) : ℕ := remaining / people
def candy_per_bag (per_person bags : ℕ) : ℕ := per_person / bags

theorem candy_per_bag_correct :
  candy_per_bag (candy_per_person (remaining_candy total_candy sister_candy) (friends + 1)) bags = 18 :=
by
  sorry

end NUMINAMATH_GPT_candy_per_bag_correct_l231_23120


namespace NUMINAMATH_GPT_no_adjacent_numbers_differ_by_10_or_multiple_10_l231_23145

theorem no_adjacent_numbers_differ_by_10_or_multiple_10 :
  ¬ ∃ (f : Fin 25 → Fin 25),
    (∀ n : Fin 25, f (n + 1) - f n = 10 ∨ (f (n + 1) - f n) % 10 = 0) :=
by
  sorry

end NUMINAMATH_GPT_no_adjacent_numbers_differ_by_10_or_multiple_10_l231_23145


namespace NUMINAMATH_GPT_modular_inverse_28_mod_29_l231_23101

theorem modular_inverse_28_mod_29 :
  28 * 28 ≡ 1 [MOD 29] :=
by
  sorry

end NUMINAMATH_GPT_modular_inverse_28_mod_29_l231_23101


namespace NUMINAMATH_GPT_fraction_value_l231_23116

theorem fraction_value :
  (2015^2 : ℤ) / (2014^2 + 2016^2 - 2) = (1 : ℚ) / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_value_l231_23116


namespace NUMINAMATH_GPT_find_g5_l231_23167

noncomputable def g (x : ℤ) : ℤ := sorry

axiom condition1 : g 1 > 1
axiom condition2 : ∀ x y : ℤ, g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom condition3 : ∀ x : ℤ, 3 * g x = g (x + 1) + 2 * x - 1

theorem find_g5 : g 5 = 248 := 
by
  sorry

end NUMINAMATH_GPT_find_g5_l231_23167


namespace NUMINAMATH_GPT_expand_fraction_product_l231_23161

theorem expand_fraction_product (x : ℝ) (hx : x ≠ 0) : 
  (3 / 4) * (8 / x^2 - 5 * x^3) = 6 / x^2 - 15 * x^3 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_expand_fraction_product_l231_23161


namespace NUMINAMATH_GPT_johns_original_earnings_l231_23136

theorem johns_original_earnings (x : ℝ) (h1 : x + 0.5 * x = 90) : x = 60 := 
by
  -- sorry indicates the proof steps are omitted
  sorry

end NUMINAMATH_GPT_johns_original_earnings_l231_23136


namespace NUMINAMATH_GPT_midpoint_trajectory_of_circle_l231_23197

theorem midpoint_trajectory_of_circle 
  (M P : ℝ × ℝ)
  (B : ℝ × ℝ)
  (hx : B = (3, 0))
  (hp : ∃(a b : ℝ), (P = (2 * a - 3, 2 * b)) ∧ (a^2 + b^2 = 1))
  (hm : M = ((P.1 + B.1) / 2, (P.2 + B.2) / 2)) :
  M.1^2 + M.2^2 - 3 * M.1 + 2 = 0 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_midpoint_trajectory_of_circle_l231_23197


namespace NUMINAMATH_GPT_increasing_sequence_range_l231_23191

theorem increasing_sequence_range (a : ℝ) (a_seq : ℕ → ℝ)
  (h₁ : ∀ (n : ℕ), n ≤ 5 → a_seq n = (5 - a) * n - 11)
  (h₂ : ∀ (n : ℕ), n > 5 → a_seq n = a ^ (n - 4))
  (h₃ : ∀ (n : ℕ), a_seq n < a_seq (n + 1)) :
  2 < a ∧ a < 5 := 
sorry

end NUMINAMATH_GPT_increasing_sequence_range_l231_23191


namespace NUMINAMATH_GPT_find_x_l231_23154

open Nat

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_x (x : ℕ) (hx : x > 0) (hprime : is_prime (x^5 + x + 1)) : x = 1 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l231_23154


namespace NUMINAMATH_GPT_burn_down_village_in_1920_seconds_l231_23117

-- Definitions of the initial conditions
def initial_cottages : Nat := 90
def burn_interval_seconds : Nat := 480
def burn_time_per_unit : Nat := 5
def max_burns_per_interval : Nat := burn_interval_seconds / burn_time_per_unit

-- Recurrence relation for the number of cottages after n intervals
def cottages_remaining (n : Nat) : Nat :=
if n = 0 then initial_cottages
else 2 * cottages_remaining (n - 1) - max_burns_per_interval

-- Time taken to burn all cottages is when cottages_remaining(n) becomes 0
def total_burn_time_seconds (intervals : Nat) : Nat :=
intervals * burn_interval_seconds

-- Main theorem statement
theorem burn_down_village_in_1920_seconds :
  ∃ n, cottages_remaining n = 0 ∧ total_burn_time_seconds n = 1920 := by
  sorry

end NUMINAMATH_GPT_burn_down_village_in_1920_seconds_l231_23117


namespace NUMINAMATH_GPT_total_length_of_sticks_l231_23155

theorem total_length_of_sticks :
  ∃ (s1 s2 s3 : ℝ), s1 = 3 ∧ s2 = 2 * s1 ∧ s3 = s2 - 1 ∧ (s1 + s2 + s3 = 14) := by
  sorry

end NUMINAMATH_GPT_total_length_of_sticks_l231_23155


namespace NUMINAMATH_GPT_problem_statement_l231_23157

theorem problem_statement (k : ℕ) (h : 35^k ∣ 1575320897) : 7^k - k^7 = 1 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l231_23157


namespace NUMINAMATH_GPT_solve_eq_sqrt_exp_l231_23153

theorem solve_eq_sqrt_exp :
  (∀ x : ℝ, (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) → (x = 2 ∨ x = -1)) :=
by
  -- Prove that the solutions are x = 2 and x = -1
  sorry

end NUMINAMATH_GPT_solve_eq_sqrt_exp_l231_23153


namespace NUMINAMATH_GPT_length_AD_of_circle_l231_23128

def circle_radius : ℝ := 8
def p_A : Prop := True  -- stand-in for the point A on the circle
def p_B : Prop := True  -- stand-in for the point B on the circle
def dist_AB : ℝ := 10
def p_D : Prop := True  -- stand-in for point D opposite B

theorem length_AD_of_circle 
  (r : ℝ := circle_radius)
  (A B D : Prop)
  (h_AB : dist_AB = 10)
  (h_radius : r = 8)
  (h_opposite : D)
  : ∃ AD : ℝ, AD = Real.sqrt 252.75 :=
sorry

end NUMINAMATH_GPT_length_AD_of_circle_l231_23128


namespace NUMINAMATH_GPT_polynomial_coefficients_sum_l231_23183

theorem polynomial_coefficients_sum :
  let a := -15
  let b := 69
  let c := -81
  let d := 27
  10 * a + 5 * b + 2 * c + d = 60 :=
by
  let a := -15
  let b := 69
  let c := -81
  let d := 27
  sorry

end NUMINAMATH_GPT_polynomial_coefficients_sum_l231_23183


namespace NUMINAMATH_GPT_passing_marks_required_l231_23132

theorem passing_marks_required (T : ℝ)
  (h1 : 0.30 * T + 60 = 0.40 * T)
  (h2 : 0.40 * T = passing_mark)
  (h3 : 0.50 * T - 40 = passing_mark) :
  passing_mark = 240 := by
  sorry

end NUMINAMATH_GPT_passing_marks_required_l231_23132


namespace NUMINAMATH_GPT_sum_of_first_five_terms_l231_23186

theorem sum_of_first_five_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q) -- geometric sequence definition
  (h3 : a 2 + a 5 = 2 * (a 4 + 2)) : 
  S 5 = 62 :=
by
  -- lean tactics would go here to provide the proof
  sorry

end NUMINAMATH_GPT_sum_of_first_five_terms_l231_23186


namespace NUMINAMATH_GPT_gas_cost_per_gallon_l231_23107

def car_mileage : Nat := 450
def car1_mpg : Nat := 50
def car2_mpg : Nat := 10
def car3_mpg : Nat := 15
def monthly_gas_cost : Nat := 56

theorem gas_cost_per_gallon (car_mileage car1_mpg car2_mpg car3_mpg monthly_gas_cost : Nat)
  (h1 : car_mileage = 450) 
  (h2 : car1_mpg = 50) 
  (h3 : car2_mpg = 10) 
  (h4 : car3_mpg = 15) 
  (h5 : monthly_gas_cost = 56) :
  monthly_gas_cost / ((car_mileage / 3) / car1_mpg + 
                      (car_mileage / 3) / car2_mpg + 
                      (car_mileage / 3) / car3_mpg) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_gas_cost_per_gallon_l231_23107


namespace NUMINAMATH_GPT_average_payment_l231_23168

theorem average_payment (total_payments : ℕ) (first_n_payments : ℕ)  (first_payment_amt : ℕ) (remaining_payment_amt : ℕ) 
  (H1 : total_payments = 104)
  (H2 : first_n_payments = 24)
  (H3 : first_payment_amt = 520)
  (H4 : remaining_payment_amt = 615)
  :
  (24 * 520 + 80 * 615) / 104 = 593.08 := 
  by 
    sorry

end NUMINAMATH_GPT_average_payment_l231_23168


namespace NUMINAMATH_GPT_find_y_given_x_l231_23176

-- Let x and y be real numbers
variables (x y : ℝ)

-- Assume x and y are inversely proportional, so their product is a constant C
variable (C : ℝ)

-- Additional conditions from the problem statement
variable (h1 : x + y = 40) (h2 : x - y = 10) (hx : x = 7)

-- Define the goal: y = 375 / 7
theorem find_y_given_x : y = 375 / 7 :=
sorry

end NUMINAMATH_GPT_find_y_given_x_l231_23176


namespace NUMINAMATH_GPT_find_value_of_expression_l231_23129

theorem find_value_of_expression (x y : ℝ) 
  (h1 : 4 * x + 2 * y = 20)
  (h2 : 2 * x + 4 * y = 16) : 
  4 * x ^ 2 + 12 * x * y + 12 * y ^ 2 = 292 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l231_23129


namespace NUMINAMATH_GPT_correct_statement_about_CH3COOK_l231_23114

def molar_mass_CH3COOK : ℝ := 98  -- in g/mol

def avogadro_number : ℝ := 6.02 * 10^23  -- molecules per mole

def hydrogen_atoms_in_CH3COOK (mol_CH3COOK : ℝ) : ℝ :=
  3 * mol_CH3COOK * avogadro_number

theorem correct_statement_about_CH3COOK (mol_CH3COOK : ℝ) (h: mol_CH3COOK = 1) :
  hydrogen_atoms_in_CH3COOK mol_CH3COOK = 3 * avogadro_number :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_about_CH3COOK_l231_23114


namespace NUMINAMATH_GPT_num_math_books_l231_23103

theorem num_math_books (total_books total_cost math_book_cost history_book_cost : ℕ) (M H : ℕ)
  (h1 : total_books = 80)
  (h2 : math_book_cost = 4)
  (h3 : history_book_cost = 5)
  (h4 : total_cost = 368)
  (h5 : M + H = total_books)
  (h6 : math_book_cost * M + history_book_cost * H = total_cost) :
  M = 32 :=
by
  sorry

end NUMINAMATH_GPT_num_math_books_l231_23103


namespace NUMINAMATH_GPT_cost_of_jacket_is_60_l231_23144

/-- Define the constants from the problem --/
def cost_of_shirt : ℕ := 8
def cost_of_pants : ℕ := 18
def shirts_bought : ℕ := 4
def pants_bought : ℕ := 2
def jackets_bought : ℕ := 2
def carrie_paid : ℕ := 94

/-- Define the problem statement --/
theorem cost_of_jacket_is_60 (total_cost jackets_cost : ℕ) 
    (H1 : total_cost = (shirts_bought * cost_of_shirt) + (pants_bought * cost_of_pants) + jackets_cost)
    (H2 : carrie_paid = total_cost / 2)
    : jackets_cost / jackets_bought = 60 := 
sorry

end NUMINAMATH_GPT_cost_of_jacket_is_60_l231_23144


namespace NUMINAMATH_GPT_negation_of_existence_l231_23123

theorem negation_of_existence (h : ¬ (∃ x : ℝ, x^2 - x - 1 > 0)) : ∀ x : ℝ, x^2 - x - 1 ≤ 0 :=
sorry

end NUMINAMATH_GPT_negation_of_existence_l231_23123


namespace NUMINAMATH_GPT_number_of_a_values_l231_23106

theorem number_of_a_values (a : ℝ) : 
  (∃ a : ℝ, ∃ b : ℝ, a = 0 ∨ a = 1) := sorry

end NUMINAMATH_GPT_number_of_a_values_l231_23106


namespace NUMINAMATH_GPT_cans_to_paint_35_rooms_l231_23119

/-- Paula the painter initially had enough paint for 45 identically sized rooms.
    Unfortunately, she lost five cans of paint, leaving her with only enough paint for 35 rooms.
    Prove that she now uses 18 cans of paint to paint the 35 rooms. -/
theorem cans_to_paint_35_rooms :
  ∀ (cans_per_room : ℕ) (total_cans : ℕ) (lost_cans : ℕ) (rooms_before : ℕ) (rooms_after : ℕ),
  rooms_before = 45 →
  lost_cans = 5 →
  rooms_after = 35 →
  rooms_before - rooms_after = cans_per_room * lost_cans →
  (cans_per_room * rooms_after) / rooms_after = 18 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cans_to_paint_35_rooms_l231_23119


namespace NUMINAMATH_GPT_triangle_angles_l231_23115

theorem triangle_angles (second_angle first_angle third_angle : ℝ) 
  (h1 : first_angle = 2 * second_angle)
  (h2 : third_angle = second_angle + 30)
  (h3 : second_angle + first_angle + third_angle = 180) :
  second_angle = 37.5 ∧ first_angle = 75 ∧ third_angle = 67.5 :=
sorry

end NUMINAMATH_GPT_triangle_angles_l231_23115


namespace NUMINAMATH_GPT_three_distinct_numbers_l231_23165

theorem three_distinct_numbers (s : ℕ) (A : Finset ℕ) (S : Finset ℕ) (hA : A = Finset.range (4 * s + 1) \ Finset.range 1)
  (hS : S ⊆ A) (hcard: S.card = 2 * s + 2) :
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x + y = 2 * z :=
by
  sorry

end NUMINAMATH_GPT_three_distinct_numbers_l231_23165


namespace NUMINAMATH_GPT_figure_perimeter_l231_23108

theorem figure_perimeter 
  (side_length : ℕ)
  (inner_large_square_sides : ℕ)
  (shared_edge_length : ℕ)
  (rectangle_dimension_1 : ℕ)
  (rectangle_dimension_2 : ℕ) 
  (h1 : side_length = 2)
  (h2 : inner_large_square_sides = 4)
  (h3 : shared_edge_length = 2)
  (h4 : rectangle_dimension_1 = 2)
  (h5 : rectangle_dimension_2 = 1) : 
  let large_square_perimeter := inner_large_square_sides * side_length
  let horizontal_perimeter := large_square_perimeter - shared_edge_length + rectangle_dimension_1 + rectangle_dimension_2
  let vertical_perimeter := large_square_perimeter
  horizontal_perimeter + vertical_perimeter = 33 := 
by
  sorry

end NUMINAMATH_GPT_figure_perimeter_l231_23108


namespace NUMINAMATH_GPT_smallest_value_fraction_l231_23151

theorem smallest_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 6) :
  ∃ k : ℝ, (∀ (x y : ℝ), (-6 ≤ x ∧ x ≤ -3) → (3 ≤ y ∧ y ≤ 6) → k ≤ (x + y) / x) ∧ k = 0 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_fraction_l231_23151


namespace NUMINAMATH_GPT_find_value_of_x2_div_y2_l231_23184

theorem find_value_of_x2_div_y2 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x ≠ y) (h5 : y ≠ z) (h6 : x ≠ z)
    (h7 : (y^2 / (x^2 - z^2) = (x^2 + y^2) / z^2))
    (h8 : (x^2 + y^2) / z^2 = x^2 / y^2) : x^2 / y^2 = 2 := by
  sorry

end NUMINAMATH_GPT_find_value_of_x2_div_y2_l231_23184


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l231_23175

def setA : Set ℝ := { x | x ≤ 4 }
def setB : Set ℝ := { x | x ≥ 1/2 }

theorem intersection_of_A_and_B : setA ∩ setB = { x | 1/2 ≤ x ∧ x ≤ 4 } := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l231_23175


namespace NUMINAMATH_GPT_tree_height_at_year_3_l231_23121

theorem tree_height_at_year_3 :
  ∃ h₃ : ℕ, h₃ = 27 ∧
  (∃ h₇ h₆ h₅ h₄ : ℕ,
   h₇ = 648 ∧
   h₆ = h₇ / 2 ∧
   h₅ = h₆ / 2 ∧
   h₄ = h₅ / 2 ∧
   h₄ = 3 * h₃) :=
by
  sorry

end NUMINAMATH_GPT_tree_height_at_year_3_l231_23121


namespace NUMINAMATH_GPT_simplify_expression_l231_23156

theorem simplify_expression (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (3 * x^2 - 2 * x - 5) / ((x - 3) * (x + 2)) - (5 * x - 6) / ((x - 3) * (x + 2)) =
  (3 * (x - (7 + Real.sqrt 37) / 6) * (x - (7 - Real.sqrt 37) / 6)) / ((x - 3) * (x + 2)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l231_23156


namespace NUMINAMATH_GPT_total_amount_l231_23110

theorem total_amount (x y z : ℝ) 
  (hy : y = 0.45 * x) 
  (hz : z = 0.30 * x) 
  (hy_value : y = 54) : 
  x + y + z = 210 := 
by
  sorry

end NUMINAMATH_GPT_total_amount_l231_23110


namespace NUMINAMATH_GPT_parabola_circle_intersection_l231_23122

theorem parabola_circle_intersection :
  (∃ x y : ℝ, y = (x - 2)^2 ∧ x + 1 = (y + 2)^2) →
  (∃ r : ℝ, ∀ x y : ℝ, (y = (x - 2)^2 ∧ x + 1 = (y + 2)^2) →
    (x - 5/2)^2 + (y + 3/2)^2 = r^2 ∧ r^2 = 3/2) :=
by
  intros
  sorry

end NUMINAMATH_GPT_parabola_circle_intersection_l231_23122


namespace NUMINAMATH_GPT_quilt_shaded_fraction_l231_23139

theorem quilt_shaded_fraction (total_squares : ℕ) (fully_shaded : ℕ) (half_shaded_squares : ℕ) (half_shades_per_square: ℕ) : 
  (((fully_shaded) + (half_shaded_squares * half_shades_per_square / 2)) / total_squares) = (1 / 4) :=
by 
  let fully_shaded := 2
  let half_shaded_squares := 4
  let half_shades_per_square := 1
  let total_squares := 16
  sorry

end NUMINAMATH_GPT_quilt_shaded_fraction_l231_23139


namespace NUMINAMATH_GPT_average_increase_l231_23127

theorem average_increase (A A' : ℕ) (runs_in_17th : ℕ) (total_innings : ℕ) (new_avg : ℕ) 
(h1 : total_innings = 17)
(h2 : runs_in_17th = 87)
(h3 : new_avg = 39)
(h4 : A' = new_avg)
(h5 : 16 * A + runs_in_17th = total_innings * new_avg) 
: A' - A = 3 := by
  sorry

end NUMINAMATH_GPT_average_increase_l231_23127


namespace NUMINAMATH_GPT_time_expression_l231_23118

theorem time_expression (h V₀ g S V t : ℝ) :
  (V = g * t + V₀) →
  (S = h + (1 / 2) * g * t^2 + V₀ * t) →
  t = (2 * (S - h)) / (V + V₀) :=
by
  intro h_eq v_eq
  sorry

end NUMINAMATH_GPT_time_expression_l231_23118


namespace NUMINAMATH_GPT_problem_proof_l231_23152

theorem problem_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b = ab → a + 4 * b = 9) ∧
  (a + b = 1 → ∀ a b,  2^a + 2^(b + 1) ≥ 4) ∧
  (a + b = ab → 1 / a^2 + 2 / b^2 = 2 / 3) ∧
  (a + b = 1 → ∀ a b,  2 * a / (a + b^2) + b / (a^2 + b) = (2 * Real.sqrt 3 / 3) + 1) :=
by
  sorry

end NUMINAMATH_GPT_problem_proof_l231_23152


namespace NUMINAMATH_GPT_skilled_picker_capacity_minimize_costs_l231_23138

theorem skilled_picker_capacity (x : ℕ) (h1 : ∀ x : ℕ, ∀ s : ℕ, s = 3 * x) (h2 : 450 * 25 = 3 * x * 25 + 600) :
  s = 30 :=
by
  sorry

theorem minimize_costs (s n m : ℕ)
(h1 : s ≤ 20)
(h2 : n ≤ 15)
(h3 : 600 = s * 30 + n * 10)
(h4 : ∀ y, y = s * 300 + n * 80) :
  m = 15 ∧ s = 15 :=
by
  sorry

end NUMINAMATH_GPT_skilled_picker_capacity_minimize_costs_l231_23138


namespace NUMINAMATH_GPT_circles_intersect_l231_23109

def circle1 (x y : ℝ) := x^2 + y^2 + 2*x + 8*y - 8 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 4*x - 4*y - 1 = 0

theorem circles_intersect : 
  (∃ (x y : ℝ), circle1 x y ∧ circle2 x y) := 
sorry

end NUMINAMATH_GPT_circles_intersect_l231_23109


namespace NUMINAMATH_GPT_analysis_duration_unknown_l231_23143

-- Definitions based on the given conditions
def number_of_bones : Nat := 206
def analysis_duration_per_bone (bone: Nat) : Nat := 5  -- assumed fixed for simplicity
-- Time spent analyzing all bones (which needs more information to be accurately known)
def total_analysis_time (bones_analyzed: Nat) (hours_per_bone: Nat) : Nat := bones_analyzed * hours_per_bone

-- Given the number of bones and duration per bone, there isn't enough information to determine the total analysis duration
theorem analysis_duration_unknown (total_bones : Nat) (duration_per_bone : Nat) (bones_remaining: Nat) (analysis_already_done : Nat) :
  total_bones = number_of_bones →
  (∀ bone, analysis_duration_per_bone bone = duration_per_bone) →
  analysis_already_done ≠ (total_bones - bones_remaining) ->
  ∃ hours_needed, hours_needed = total_analysis_time (total_bones - bones_remaining) duration_per_bone :=
by
  intros
  sorry

end NUMINAMATH_GPT_analysis_duration_unknown_l231_23143


namespace NUMINAMATH_GPT_least_possible_value_of_m_plus_n_l231_23147

theorem least_possible_value_of_m_plus_n 
(m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) 
(hgcd : Nat.gcd (m + n) 210 = 1) 
(hdiv : ∃ k, m^m = k * n^n)
(hnotdiv : ¬ ∃ k, m = k * n) : 
  m + n = 407 := 
sorry

end NUMINAMATH_GPT_least_possible_value_of_m_plus_n_l231_23147


namespace NUMINAMATH_GPT_subset_range_l231_23164

open Set

-- Definitions of sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x < a}

-- The statement of the problem
theorem subset_range (a : ℝ) (h : A ⊆ B a) : 2 ≤ a :=
sorry -- Skipping the proof

end NUMINAMATH_GPT_subset_range_l231_23164


namespace NUMINAMATH_GPT_perpendicular_vectors_m_eq_0_or_neg2_l231_23188

theorem perpendicular_vectors_m_eq_0_or_neg2
  (m : ℝ)
  (a : ℝ × ℝ := (m, 1))
  (b : ℝ × ℝ := (1, m - 1))
  (h : a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 0) :
  m = 0 ∨ m = -2 := sorry

end NUMINAMATH_GPT_perpendicular_vectors_m_eq_0_or_neg2_l231_23188


namespace NUMINAMATH_GPT_system_of_equations_correct_l231_23163

-- Define the problem conditions
variable (x y : ℝ) -- Define the productivity of large and small harvesters

-- Define the correct system of equations as per the problem
def system_correct : Prop := (2 * (2 * x + 5 * y) = 3.6) ∧ (5 * (3 * x + 2 * y) = 8)

-- State the theorem to prove the correctness of the system of equations under given conditions
theorem system_of_equations_correct (x y : ℝ) : (2 * (2 * x + 5 * y) = 3.6) ∧ (5 * (3 * x + 2 * y) = 8) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_correct_l231_23163


namespace NUMINAMATH_GPT_value_of_a_l231_23134

theorem value_of_a (a x : ℝ) (h : x = 4) (h_eq : x^2 - 3 * x = a^2) : a = 2 ∨ a = -2 :=
by
  -- The proof is omitted, but the theorem statement adheres to the problem conditions and expected result.
  sorry

end NUMINAMATH_GPT_value_of_a_l231_23134


namespace NUMINAMATH_GPT_weight_loss_challenge_l231_23162

noncomputable def percentage_weight_loss (W : ℝ) : ℝ :=
  ((W - (0.918 * W)) / W) * 100

theorem weight_loss_challenge (W : ℝ) (h : W > 0) :
  percentage_weight_loss W = 8.2 :=
by
  sorry

end NUMINAMATH_GPT_weight_loss_challenge_l231_23162


namespace NUMINAMATH_GPT_num_adults_on_field_trip_l231_23100

-- Definitions of the conditions
def num_vans : Nat := 6
def people_per_van : Nat := 9
def num_students : Nat := 40

-- The theorem to prove
theorem num_adults_on_field_trip : (num_vans * people_per_van) - num_students = 14 := by
  sorry

end NUMINAMATH_GPT_num_adults_on_field_trip_l231_23100


namespace NUMINAMATH_GPT_last_number_nth_row_sum_of_nth_row_position_of_2008_l231_23150

theorem last_number_nth_row (n : ℕ) : 
  ∃ last_number, last_number = 2^n - 1 := by
  sorry

theorem sum_of_nth_row (n : ℕ) : 
  ∃ sum_nth_row, sum_nth_row = 2^(2*n-2) + 2^(2*n-3) - 2^(n-2) := by
  sorry

theorem position_of_2008 : 
  ∃ (row : ℕ) (position : ℕ), row = 11 ∧ position = 2008 - 2^10 + 1 :=
  by sorry

end NUMINAMATH_GPT_last_number_nth_row_sum_of_nth_row_position_of_2008_l231_23150


namespace NUMINAMATH_GPT_sum_geom_seq_l231_23104

theorem sum_geom_seq (S : ℕ → ℝ) (a_n : ℕ → ℝ) (h1 : S 4 ≠ 0) 
  (h2 : S 8 / S 4 = 4) 
  (h3 : ∀ n : ℕ, S n = a_n 0 * (1 - (a_n 1 / a_n 0)^n) / (1 - a_n 1 / a_n 0)) :
  S 12 / S 4 = 13 :=
sorry

end NUMINAMATH_GPT_sum_geom_seq_l231_23104


namespace NUMINAMATH_GPT_Dorothy_found_57_pieces_l231_23178

def total_pieces_Dorothy_found 
  (B_green B_red R_red R_blue : ℕ)
  (D_red_factor D_blue_factor : ℕ)
  (H1 : B_green = 12)
  (H2 : B_red = 3)
  (H3 : R_red = 9)
  (H4 : R_blue = 11)
  (H5 : D_red_factor = 2)
  (H6 : D_blue_factor = 3) : ℕ := 
  let D_red := D_red_factor * (B_red + R_red)
  let D_blue := D_blue_factor * R_blue
  D_red + D_blue

theorem Dorothy_found_57_pieces 
  (B_green B_red R_red R_blue : ℕ)
  (D_red_factor D_blue_factor : ℕ)
  (H1 : B_green = 12)
  (H2 : B_red = 3)
  (H3 : R_red = 9)
  (H4 : R_blue = 11)
  (H5 : D_red_factor = 2)
  (H6 : D_blue_factor = 3) :
  total_pieces_Dorothy_found B_green B_red R_red R_blue D_red_factor D_blue_factor H1 H2 H3 H4 H5 H6 = 57 := by
    sorry

end NUMINAMATH_GPT_Dorothy_found_57_pieces_l231_23178


namespace NUMINAMATH_GPT_p_sq_plus_q_sq_l231_23187

theorem p_sq_plus_q_sq (p q : ℝ) (h1 : p * q = 12) (h2 : p + q = 8) : p^2 + q^2 = 40 := by
  sorry

end NUMINAMATH_GPT_p_sq_plus_q_sq_l231_23187


namespace NUMINAMATH_GPT_interval_length_implies_difference_l231_23135

theorem interval_length_implies_difference (a b : ℝ) (h : (b - 5) / 3 - (a - 5) / 3 = 15) : b - a = 45 := by
  sorry

end NUMINAMATH_GPT_interval_length_implies_difference_l231_23135


namespace NUMINAMATH_GPT_initial_crayons_count_l231_23170

theorem initial_crayons_count (C : ℕ) :
  (3 / 8) * C = 18 → C = 48 :=
by
  sorry

end NUMINAMATH_GPT_initial_crayons_count_l231_23170


namespace NUMINAMATH_GPT_exists_f_satisfying_iteration_l231_23102

-- Mathematically equivalent problem statement in Lean 4
theorem exists_f_satisfying_iteration :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, (f^[1995] n) = 2 * n :=
by
  -- Fill in proof here
  sorry

end NUMINAMATH_GPT_exists_f_satisfying_iteration_l231_23102


namespace NUMINAMATH_GPT_find_original_number_l231_23137

theorem find_original_number (r : ℝ) (h1 : r * 1.125 - r * 0.75 = 30) : r = 80 :=
by
  sorry

end NUMINAMATH_GPT_find_original_number_l231_23137


namespace NUMINAMATH_GPT_Marissa_sister_height_l231_23177

theorem Marissa_sister_height (sunflower_height_feet : ℕ) (height_difference_inches : ℕ) :
  sunflower_height_feet = 6 -> height_difference_inches = 21 -> 
  let sunflower_height_inches := sunflower_height_feet * 12
  let sister_height_inches := sunflower_height_inches - height_difference_inches
  let sister_height_feet := sister_height_inches / 12
  let sister_height_remainder_inches := sister_height_inches % 12
  sister_height_feet = 4 ∧ sister_height_remainder_inches = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Marissa_sister_height_l231_23177


namespace NUMINAMATH_GPT_electricity_usage_A_B_l231_23124

def electricity_cost (x : ℕ) : ℝ :=
  if h₁ : 0 ≤ x ∧ x ≤ 24 then 4.2 * x
  else if h₂ : 24 < x ∧ x ≤ 60 then 5.2 * x - 24
  else if h₃ : 60 < x ∧ x ≤ 100 then 6.6 * x - 108
  else if h₄ : 100 < x ∧ x ≤ 150 then 7.6 * x - 208
  else if h₅ : 150 < x ∧ x ≤ 250 then 8 * x - 268
  else 8.4 * x - 368

theorem electricity_usage_A_B (x : ℕ) (h : electricity_cost x = 486) :
  60 < x ∧ x ≤ 100 ∧ 5 * x = 450 ∧ 2 * x = 180 :=
by
  sorry

end NUMINAMATH_GPT_electricity_usage_A_B_l231_23124


namespace NUMINAMATH_GPT_calculate_bankers_discount_l231_23173

noncomputable def present_worth : ℝ := 800
noncomputable def true_discount : ℝ := 36
noncomputable def face_value : ℝ := present_worth + true_discount
noncomputable def bankers_discount : ℝ := (face_value * true_discount) / (face_value - true_discount)

theorem calculate_bankers_discount :
  bankers_discount = 37.62 := 
sorry

end NUMINAMATH_GPT_calculate_bankers_discount_l231_23173


namespace NUMINAMATH_GPT_g_diff_l231_23105

def g (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 - 2 * x - 1

theorem g_diff (x h : ℝ) : g (x + h) - g x = h * (6 * x^2 + 6 * x * h + 2 * h^2 + 10 * x + 5 * h - 2) := 
by
  sorry

end NUMINAMATH_GPT_g_diff_l231_23105


namespace NUMINAMATH_GPT_solve_first_system_solve_second_system_l231_23189

-- Define the first system of equations
def first_system (x y : ℝ) : Prop := (3 * x + 2 * y = 5) ∧ (y = 2 * x - 8)

-- Define the solution to the first system
def solution1 (x y : ℝ) : Prop := (x = 3) ∧ (y = -2)

-- Define the second system of equations
def second_system (x y : ℝ) : Prop := (2 * x - y = 10) ∧ (2 * x + 3 * y = 2)

-- Define the solution to the second system
def solution2 (x y : ℝ) : Prop := (x = 4) ∧ (y = -2)

-- Define the problem statement in Lean
theorem solve_first_system : ∃ x y : ℝ, first_system x y ↔ solution1 x y :=
by
  sorry

theorem solve_second_system : ∃ x y : ℝ, second_system x y ↔ solution2 x y :=
by
  sorry

end NUMINAMATH_GPT_solve_first_system_solve_second_system_l231_23189


namespace NUMINAMATH_GPT_polygon_sides_l231_23166

theorem polygon_sides (n : ℕ) (h : 180 * (n - 2) = 1620) : n = 11 := 
by 
  sorry

end NUMINAMATH_GPT_polygon_sides_l231_23166


namespace NUMINAMATH_GPT_area_of_rectangle_l231_23179

-- Definitions from the conditions
def breadth (b : ℝ) : Prop := b > 0
def length (l b : ℝ) : Prop := l = 3 * b
def perimeter (P l b : ℝ) : Prop := P = 2 * (l + b)

-- The main theorem we are proving
theorem area_of_rectangle (b l : ℝ) (P : ℝ) (h1 : breadth b) (h2 : length l b) (h3 : perimeter P l b) (h4 : P = 96) : l * b = 432 := 
by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l231_23179


namespace NUMINAMATH_GPT_shape_of_phi_eq_d_in_spherical_coordinates_l231_23194

theorem shape_of_phi_eq_d_in_spherical_coordinates (d : ℝ) : 
  (∃ (ρ θ : ℝ), ∀ (φ : ℝ), φ = d) ↔ ( ∃ cone_vertex : ℝ × ℝ × ℝ, ∃ opening_angle : ℝ, cone_vertex = (0, 0, 0) ∧ opening_angle = d) :=
sorry

end NUMINAMATH_GPT_shape_of_phi_eq_d_in_spherical_coordinates_l231_23194


namespace NUMINAMATH_GPT_sum_of_cubes_8001_l231_23196
-- Import the entire Mathlib library

-- Define a property on integers
def approx (x y : ℝ) := abs (x - y) < 0.000000000000004

-- Define the variables a and b
variables (a b : ℤ)

-- State the theorem
theorem sum_of_cubes_8001 (h : approx (a * b : ℝ) 19.999999999999996) : a^3 + b^3 = 8001 := 
sorry

end NUMINAMATH_GPT_sum_of_cubes_8001_l231_23196


namespace NUMINAMATH_GPT_proof_problem_l231_23148

variables {a b c : Real}

theorem proof_problem (h1 : a < 0) (h2 : |a| < |b|) (h3 : |b| < |c|) (h4 : b < 0) :
  (|a * b| < |b * c|) ∧ (a * c < |b * c|) ∧ (|a + b| < |b + c|) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l231_23148


namespace NUMINAMATH_GPT_molecular_weight_compound_l231_23169

/-- Definition of atomic weights for elements H, Cr, and O in AMU (Atomic Mass Units) --/
def atomic_weight_H : ℝ := 1.008
def atomic_weight_Cr : ℝ := 51.996
def atomic_weight_O : ℝ := 15.999

/-- Proof statement to calculate the molecular weight of a compound with 2 H, 1 Cr, and 4 O --/
theorem molecular_weight_compound :
  2 * atomic_weight_H + 1 * atomic_weight_Cr + 4 * atomic_weight_O = 118.008 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_compound_l231_23169


namespace NUMINAMATH_GPT_bill_head_circumference_l231_23125

theorem bill_head_circumference (jack_head_circumference charlie_head_circumference bill_head_circumference : ℝ) :
  jack_head_circumference = 12 →
  charlie_head_circumference = (1 / 2 * jack_head_circumference) + 9 →
  bill_head_circumference = (2 / 3 * charlie_head_circumference) →
  bill_head_circumference = 10 :=
by
  intro hj hc hb
  sorry

end NUMINAMATH_GPT_bill_head_circumference_l231_23125


namespace NUMINAMATH_GPT_angle_BDC_proof_l231_23180

noncomputable def angle_sum_triangle (angle_A angle_B angle_C : ℝ) : Prop :=
  angle_A + angle_B + angle_C = 180

-- Given conditions
def angle_A : ℝ := 70
def angle_E : ℝ := 50
def angle_C : ℝ := 40

-- The problem of proving that angle_BDC = 20 degrees
theorem angle_BDC_proof (A E C BDC : ℝ) 
  (hA : A = angle_A)
  (hE : E = angle_E)
  (hC : C = angle_C) :
  BDC = 20 :=
  sorry

end NUMINAMATH_GPT_angle_BDC_proof_l231_23180


namespace NUMINAMATH_GPT_not_square_a2_b2_ab_l231_23159

theorem not_square_a2_b2_ab (n : ℕ) (h_n : n > 2) (a : ℕ) (b : ℕ) (h_b : b = 2^(2^n))
  (h_a_odd : a % 2 = 1) (h_a_le_b : a ≤ b) (h_b_le_2a : b ≤ 2 * a) :
  ¬ ∃ k : ℕ, a^2 + b^2 - a * b = k^2 :=
by
  sorry

end NUMINAMATH_GPT_not_square_a2_b2_ab_l231_23159


namespace NUMINAMATH_GPT_coat_price_calculation_l231_23171

noncomputable def effective_price (initial_price : ℝ) 
  (reduction1 reduction2 reduction3 : ℝ) 
  (tax1 tax2 tax3 : ℝ) : ℝ :=
  let price_after_first_month := initial_price * (1 - reduction1 / 100) * (1 + tax1 / 100)
  let price_after_second_month := price_after_first_month * (1 - reduction2 / 100) * (1 + tax2 / 100)
  let price_after_third_month := price_after_second_month * (1 - reduction3 / 100) * (1 + tax3 / 100)
  price_after_third_month

noncomputable def total_percent_reduction (initial_price final_price : ℝ) : ℝ :=
  (initial_price - final_price) / initial_price * 100

theorem coat_price_calculation :
  let original_price := 500
  let price_final := effective_price original_price 10 15 20 5 8 6
  let reduction_percentage := total_percent_reduction original_price price_final
  price_final = 367.824 ∧ reduction_percentage = 26.44 :=
by
  sorry

end NUMINAMATH_GPT_coat_price_calculation_l231_23171


namespace NUMINAMATH_GPT_tangent_line_eq_l231_23146

theorem tangent_line_eq {f : ℝ → ℝ} (hf : ∀ x, f x = x - 2 * Real.log x) :
  ∃ m b, (m = -1) ∧ (b = 2) ∧ (∀ x, f x = m * x + b) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_eq_l231_23146


namespace NUMINAMATH_GPT_lcm_48_180_l231_23160

theorem lcm_48_180 : Int.lcm 48 180 = 720 := by
  sorry

end NUMINAMATH_GPT_lcm_48_180_l231_23160


namespace NUMINAMATH_GPT_graph_of_eqn_is_pair_of_lines_l231_23133

theorem graph_of_eqn_is_pair_of_lines : 
  ∃ (l₁ l₂ : ℝ × ℝ → Prop), 
  (∀ x y, l₁ (x, y) ↔ x = 2 * y) ∧ 
  (∀ x y, l₂ (x, y) ↔ x = -2 * y) ∧ 
  (∀ x y, (x^2 - 4 * y^2 = 0) ↔ (l₁ (x, y) ∨ l₂ (x, y))) :=
by
  sorry

end NUMINAMATH_GPT_graph_of_eqn_is_pair_of_lines_l231_23133


namespace NUMINAMATH_GPT_sum_of_roots_l231_23190

theorem sum_of_roots (a1 a2 a3 a4 a5 : ℤ)
  (h_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧
                a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧
                a3 ≠ a4 ∧ a3 ≠ a5 ∧
                a4 ≠ a5)
  (h_poly : (104 - a1) * (104 - a2) * (104 - a3) * (104 - a4) * (104 - a5) = 2012) :
  a1 + a2 + a3 + a4 + a5 = 17 := by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l231_23190


namespace NUMINAMATH_GPT_part_A_part_B_part_D_l231_23198

variables (c d : ℤ)

def multiple_of_5 (x : ℤ) : Prop := ∃ k : ℤ, x = 5 * k
def multiple_of_10 (x : ℤ) : Prop := ∃ k : ℤ, x = 10 * k

-- Given conditions
axiom h1 : multiple_of_5 c
axiom h2 : multiple_of_10 d

-- Problems to prove
theorem part_A : multiple_of_5 d := by sorry
theorem part_B : multiple_of_5 (c - d) := by sorry
theorem part_D : multiple_of_5 (c + d) := by sorry

end NUMINAMATH_GPT_part_A_part_B_part_D_l231_23198


namespace NUMINAMATH_GPT_fraction_pattern_l231_23193

theorem fraction_pattern (n m k : ℕ) (h : n / m = k * n / (k * m)) : (n + m) / m = (k * n + k * m) / (k * m) := by
  sorry

end NUMINAMATH_GPT_fraction_pattern_l231_23193


namespace NUMINAMATH_GPT_reflect_point_example_l231_23113

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def reflect_over_x_axis (P : Point3D) : Point3D :=
  { x := P.x, y := -P.y, z := -P.z }

theorem reflect_point_example :
  reflect_over_x_axis ⟨2, 3, 4⟩ = ⟨2, -3, -4⟩ :=
by
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_reflect_point_example_l231_23113


namespace NUMINAMATH_GPT_quadratic_roots_distinct_real_l231_23111

theorem quadratic_roots_distinct_real (a b c : ℝ) (h_eq : 2 * a = 2 ∧ 2 * b + -3 = b ∧ 2 * c + 1 = c) :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (∀ x : ℝ, (2 * x^2 + (-3) * x + 1 = 0) ↔ (x = x1 ∨ x = x2)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_distinct_real_l231_23111


namespace NUMINAMATH_GPT_max_books_per_student_l231_23126

-- Define the variables and conditions
variables (students : ℕ) (not_borrowed5 borrowed1_20 borrowed2_25 borrowed3_30 borrowed5_20 : ℕ)
variables (avg_books_per_student : ℕ)
variables (remaining_books : ℕ) (max_books : ℕ)

-- Assume given conditions
def conditions : Prop :=
  students = 100 ∧ 
  not_borrowed5 = 5 ∧ 
  borrowed1_20 = 20 ∧ 
  borrowed2_25 = 25 ∧ 
  borrowed3_30 = 30 ∧ 
  borrowed5_20 = 20 ∧ 
  avg_books_per_student = 3

-- Prove the maximum number of books any single student could have borrowed is 50
theorem max_books_per_student (students not_borrowed5 borrowed1_20 borrowed2_25 borrowed3_30 borrowed5_20 avg_books_per_student : ℕ) (max_books : ℕ) :
  conditions students not_borrowed5 borrowed1_20 borrowed2_25 borrowed3_30 borrowed5_20 avg_books_per_student →
  max_books = 50 :=
by
  sorry

end NUMINAMATH_GPT_max_books_per_student_l231_23126
