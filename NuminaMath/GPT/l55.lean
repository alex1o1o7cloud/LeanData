import Mathlib

namespace quadrilateral_sides_equal_l55_55681

theorem quadrilateral_sides_equal (a b c d : ℕ) (h1 : a ∣ b + c + d) (h2 : b ∣ a + c + d) (h3 : c ∣ a + b + d) (h4 : d ∣ a + b + c) : a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d :=
sorry

end quadrilateral_sides_equal_l55_55681


namespace tv_sets_sales_decrease_l55_55905

theorem tv_sets_sales_decrease
  (P Q P' Q' R R': ℝ)
  (h1 : P' = 1.6 * P)
  (h2 : R' = 1.28 * R)
  (h3 : R = P * Q)
  (h4 : R' = P' * Q')
  (h5 : Q' = Q * (1 - D / 100)) :
  D = 20 :=
by
  sorry

end tv_sets_sales_decrease_l55_55905


namespace parallelogram_area_correct_l55_55087

noncomputable def parallelogram_area (s1 s2 : ℝ) (a : ℝ) : ℝ :=
s2 * (2 * s2 * Real.sin a)

theorem parallelogram_area_correct (s2 a : ℝ) (h_pos_s2 : 0 < s2) :
  parallelogram_area (2 * s2) s2 a = 2 * s2^2 * Real.sin a :=
by
  unfold parallelogram_area
  sorry

end parallelogram_area_correct_l55_55087


namespace sufficient_not_necessary_a_equals_2_l55_55907

theorem sufficient_not_necessary_a_equals_2 {a : ℝ} :
  (∃ a : ℝ, (a = 2 ∧ 15 * a^2 = 60) → (15 * a^2 = 60) ∧ (15 * a^2 = 60 → a = 2)) → 
  (¬∀ a : ℝ, (15 * a^2 = 60) → a = 2) → 
  (a = 2 → 15 * a^2 = 60) ∧ ¬(15 * a^2 = 60 → a = 2) :=
by
  sorry

end sufficient_not_necessary_a_equals_2_l55_55907


namespace projectile_height_time_l55_55970

theorem projectile_height_time (h : ∀ t : ℝ, -16 * t^2 + 100 * t = 64 → t = 1) : (∃ t : ℝ, -16 * t^2 + 100 * t = 64 ∧ t = 1) :=
by sorry

end projectile_height_time_l55_55970


namespace contradiction_proof_l55_55761

theorem contradiction_proof (x y : ℝ) (h1 : x + y ≤ 0) (h2 : x > 0) (h3 : y > 0) : false :=
by
  sorry

end contradiction_proof_l55_55761


namespace ratio_first_term_to_common_difference_l55_55742

theorem ratio_first_term_to_common_difference
  (a d : ℝ)
  (S_n : ℕ → ℝ)
  (hS_n : ∀ n, S_n n = (n / 2) * (2 * a + (n - 1) * d))
  (h : S_n 15 = 3 * S_n 10) :
  a / d = -2 :=
by
  sorry

end ratio_first_term_to_common_difference_l55_55742


namespace C_converges_l55_55305

noncomputable def behavior_of_C (e R r : ℝ) (n : ℕ) : ℝ := e * (n^2) / (R + n * (r^2))

theorem C_converges (e R r : ℝ) (h₁ : 0 < r) : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |behavior_of_C e R r n - e / r^2| < ε := 
sorry

end C_converges_l55_55305


namespace ratio_x_y_l55_55357

variable (x y : ℝ)

-- Conditions:
-- 1. lengths of pieces
def is_square (x : ℝ) : Prop := ∃ s, x = 4 * s
def is_pentagon (y : ℝ) : Prop := ∃ t, y = 5 * t
def equal_perimeter (x y : ℝ) : Prop := x = y

-- Theorem to prove
theorem ratio_x_y (hx : is_square x) (hy : is_pentagon y) (h_perimeter : equal_perimeter x y) : x / y = 1 :=
by {
  -- Implementation of the proof
  sorry
}

end ratio_x_y_l55_55357


namespace fraction_problem_l55_55586

theorem fraction_problem (a b c d e: ℚ) (val: ℚ) (h_a: a = 1/4) (h_b: b = 1/3) 
  (h_c: c = 1/6) (h_d: d = 1/8) (h_val: val = 72) :
  (a * b * c * val + d) = 9 / 8 :=
by {
  sorry
}

end fraction_problem_l55_55586


namespace problem_condition_l55_55370

theorem problem_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x : ℝ, f x = 5 * x - 3) → (∀ x : ℝ, |x + 0.4| < b → |f x + 1| < a) ↔ (0 < a ∧ 0 < b ∧ b ≤ a / 5) := by
  sorry

end problem_condition_l55_55370


namespace geometric_sequence_properties_l55_55149

noncomputable def geometric_sequence (a2 a5 : ℕ) (n : ℕ) : ℕ :=
  3 ^ (n - 1)

noncomputable def sum_first_n_terms (n : ℕ) : ℕ :=
  (3^n - 1) / 2

def T10_sum_of_sequence : ℚ := 10/11

theorem geometric_sequence_properties :
  (geometric_sequence 3 81 2 = 3) ∧
  (geometric_sequence 3 81 5 = 81) ∧
  (sum_first_n_terms 2 = (geometric_sequence 3 81 1 + geometric_sequence 3 81 2)) ∧
  (sum_first_n_terms 5 = (geometric_sequence 3 81 1 + geometric_sequence 3 81 2 + geometric_sequence 3 81 3 + geometric_sequence 3 81 4 + geometric_sequence 3 81 5)) ∧
  T10_sum_of_sequence = 10/11 :=
by
  sorry

end geometric_sequence_properties_l55_55149


namespace binomial_coefficient_ratio_l55_55679

theorem binomial_coefficient_ratio (n k : ℕ) (h₁ : n = 4 * k + 3) (h₂ : n = 3 * k + 5) : n + k = 13 :=
by
  sorry

end binomial_coefficient_ratio_l55_55679


namespace initial_concentration_l55_55243

theorem initial_concentration (C : ℝ) 
  (hC : (C * 0.2222222222222221) + (0.25 * 0.7777777777777779) = 0.35) :
  C = 0.7 :=
sorry

end initial_concentration_l55_55243


namespace inequality_proof_l55_55589

theorem inequality_proof (a b c d : ℝ) : 
  (a^2 + b^2 + 1) * (c^2 + d^2 + 1) ≥ 2 * (a + c) * (b + d) :=
by sorry

end inequality_proof_l55_55589


namespace joe_travel_time_l55_55406

theorem joe_travel_time
  (d : ℝ) -- Total distance
  (rw : ℝ) (rr : ℝ) -- Walking and running rates
  (tw : ℝ) -- Walking time
  (tr : ℝ) -- Running time
  (h1 : tw = 9)
  (h2 : rr = 4 * rw)
  (h3 : rw * tw = d / 3)
  (h4 : rr * tr = 2 * d / 3) :
  tw + tr = 13.5 :=
by 
  sorry

end joe_travel_time_l55_55406


namespace train_ride_length_l55_55359

noncomputable def totalMinutesUntil0900 (leaveTime : Nat) (arrivalTime : Nat) : Nat :=
  arrivalTime - leaveTime

noncomputable def walkTime : Nat := 10

noncomputable def rideTime (totalTime : Nat) (walkTime : Nat) : Nat :=
  totalTime - walkTime

theorem train_ride_length (leaveTime : Nat) (arrivalTime : Nat) :
  leaveTime = 450 → arrivalTime = 540 → rideTime (totalMinutesUntil0900 leaveTime arrivalTime) walkTime = 80 :=
by
  intros h_leaveTime h_arrivalTime
  rw [h_leaveTime, h_arrivalTime]
  unfold totalMinutesUntil0900
  unfold rideTime
  unfold walkTime
  sorry

end train_ride_length_l55_55359


namespace factor_1024_count_l55_55080

theorem factor_1024_count :
  ∃ (n : ℕ), 
  (∀ (a b c : ℕ), (a >= b) → (b >= c) → (2^a * 2^b * 2^c = 1024) → a + b + c = 10) ∧ n = 14 :=
sorry

end factor_1024_count_l55_55080


namespace fraction_power_multiplication_l55_55067

theorem fraction_power_multiplication :
  ((1 : ℝ) / 3) ^ 4 * ((1 : ℝ) / 5) = ((1 : ℝ) / 405) := by
  sorry

end fraction_power_multiplication_l55_55067


namespace daphney_potatoes_l55_55002

theorem daphney_potatoes (cost_per_2kg : ℕ) (total_paid : ℕ) (amount_per_kg : ℕ) (kg_bought : ℕ) 
  (h1 : cost_per_2kg = 6) (h2 : total_paid = 15) (h3 : amount_per_kg = cost_per_2kg / 2) 
  (h4 : kg_bought = total_paid / amount_per_kg) : kg_bought = 5 :=
by
  sorry

end daphney_potatoes_l55_55002


namespace john_must_deliver_1063_pizzas_l55_55734

-- Declare all the given conditions
def car_cost : ℕ := 8000
def maintenance_cost : ℕ := 500
def pizza_income (p : ℕ) : ℕ := 12 * p
def gas_cost (p : ℕ) : ℕ := 4 * p

-- Define the function that returns the net earnings
def net_earnings (p : ℕ) := pizza_income p - gas_cost p

-- Define the total expenses
def total_expenses : ℕ := car_cost + maintenance_cost

-- Define the minimum number of pizzas John must deliver
def minimum_pizzas (p : ℕ) : Prop := net_earnings p ≥ total_expenses

-- State the theorem that needs to be proved
theorem john_must_deliver_1063_pizzas : minimum_pizzas 1063 := by
  sorry

end john_must_deliver_1063_pizzas_l55_55734


namespace luxury_class_adults_l55_55568

def total_passengers : ℕ := 300
def adult_percentage : ℝ := 0.70
def luxury_percentage : ℝ := 0.15

def total_adults (p : ℕ) : ℕ := (p * 70) / 100
def adults_in_luxury (a : ℕ) : ℕ := (a * 15) / 100

theorem luxury_class_adults :
  adults_in_luxury (total_adults total_passengers) = 31 :=
by
  sorry

end luxury_class_adults_l55_55568


namespace simplify_expression_l55_55395

theorem simplify_expression :
  2 * Real.sqrt (1 + Real.sin 8) + Real.sqrt (2 + 2 * Real.cos 8) = -2 * Real.sin 4 - 4 * Real.cos 4 :=
by
  sorry

end simplify_expression_l55_55395


namespace book_arrangement_l55_55153

theorem book_arrangement (math_books : ℕ) (english_books : ℕ) (science_books : ℕ)
  (math_different : math_books = 4) 
  (english_different : english_books = 5) 
  (science_different : science_books = 2) :
  (Nat.factorial 3) * (Nat.factorial math_books) * (Nat.factorial english_books) * (Nat.factorial science_books) = 34560 := 
by
  sorry

end book_arrangement_l55_55153


namespace circle_hyperbola_intersection_l55_55129

def hyperbola_equation (x y a : ℝ) : Prop := x^2 - y^2 = a^2
def circle_equation (x y c d r : ℝ) : Prop := (x - c)^2 + (y - d)^2 = r^2

theorem circle_hyperbola_intersection (a r : ℝ) (P Q R S : ℝ × ℝ):
  (∃ c d: ℝ, 
    circle_equation P.1 P.2 c d r ∧ 
    circle_equation Q.1 Q.2 c d r ∧ 
    circle_equation R.1 R.2 c d r ∧ 
    circle_equation S.1 S.2 c d r ∧ 
    hyperbola_equation P.1 P.2 a ∧ 
    hyperbola_equation Q.1 Q.2 a ∧ 
    hyperbola_equation R.1 R.2 a ∧ 
    hyperbola_equation S.1 S.2 a
  ) →
  (P.1^2 + P.2^2) + (Q.1^2 + Q.2^2) + (R.1^2 + R.2^2) + (S.1^2 + S.2^2) = 4 * r^2 :=
by
  sorry

end circle_hyperbola_intersection_l55_55129


namespace geometric_sequence_first_term_l55_55507

theorem geometric_sequence_first_term (S_3 S_6 : ℝ) (a_1 q : ℝ)
  (hS3 : S_3 = 6) (hS6 : S_6 = 54)
  (hS3_def : S_3 = a_1 * (1 - q^3) / (1 - q))
  (hS6_def : S_6 = a_1 * (1 - q^6) / (1 - q)) :
  a_1 = 6 / 7 := 
by
  sorry

end geometric_sequence_first_term_l55_55507


namespace find_value_of_a_l55_55997

noncomputable def log_base_four (a : ℝ) : ℝ := Real.log a / Real.log 4

theorem find_value_of_a (a : ℝ) (h : log_base_four a = (1 : ℝ) / (2 : ℝ)) : a = 2 := by
  sorry

end find_value_of_a_l55_55997


namespace trig_eq_solutions_l55_55875

open Real

theorem trig_eq_solutions (x : ℝ) :
  2 * sin x ^ 3 + 2 * sin x ^ 2 * cos x - sin x * cos x ^ 2 - cos x ^ 3 = 0 ↔
  (∃ n : ℤ, x = -π / 4 + n * π) ∨ (∃ k : ℤ, x = arctan (sqrt 2 / 2) + k * π) ∨ (∃ m : ℤ, x = -arctan (sqrt 2 / 2) + m * π) :=
by
  sorry

end trig_eq_solutions_l55_55875


namespace add_hex_numbers_l55_55410

theorem add_hex_numbers : (7 * 16^2 + 10 * 16^1 + 3) + (1 * 16^2 + 15 * 16^1 + 4) = 9 * 16^2 + 9 * 16^1 + 7 := by sorry

end add_hex_numbers_l55_55410


namespace inequality_solution_l55_55987

theorem inequality_solution (x : ℝ) :
  (6*x^2 + 24*x - 63) / ((3*x - 4)*(x + 5)) < 4 ↔ x ∈ Set.Ioo (-(5:ℝ)) (4 / 3) ∪ Set.Iio (5) ∪ Set.Ioi (4 / 3) := by
  sorry

end inequality_solution_l55_55987


namespace max_val_xa_minus_2x_max_val_ab_plus_bc_plus_ac_l55_55477

-- First proof problem
theorem max_val_xa_minus_2x (x a : ℝ) (h1 : 0 < x) (h2 : 2 * x < a) :
  ∃ y, (y = x * (a - 2 * x)) ∧ y ≤ a^2 / 8 :=
sorry

-- Second proof problem
theorem max_val_ab_plus_bc_plus_ac (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 4) :
  ab + bc + ac ≤ 4 :=
sorry

end max_val_xa_minus_2x_max_val_ab_plus_bc_plus_ac_l55_55477


namespace walter_age_in_2005_l55_55887

theorem walter_age_in_2005 
  (y : ℕ) (gy : ℕ)
  (h1 : gy = 3 * y)
  (h2 : (2000 - y) + (2000 - gy) = 3896) : y + 5 = 31 :=
by {
  sorry
}

end walter_age_in_2005_l55_55887


namespace latus_rectum_equation_l55_55429

theorem latus_rectum_equation (y x : ℝ) :
  y^2 = 4 * x → x = -1 :=
sorry

end latus_rectum_equation_l55_55429


namespace carrie_total_sales_l55_55091

theorem carrie_total_sales :
  let tomatoes := 200
  let carrots := 350
  let price_tomato := 1.0
  let price_carrot := 1.50
  (tomatoes * price_tomato + carrots * price_carrot) = 725 := by
  -- let tomatoes := 200
  -- let carrots := 350
  -- let price_tomato := 1.0
  -- let price_carrot := 1.50
  -- show (tomatoes * price_tomato + carrots * price_carrot) = 725
  sorry

end carrie_total_sales_l55_55091


namespace phase_shift_of_sine_l55_55999

theorem phase_shift_of_sine :
  let B := 5
  let C := (3 * Real.pi) / 2
  let phase_shift := C / B
  phase_shift = (3 * Real.pi) / 10 := by
    sorry

end phase_shift_of_sine_l55_55999


namespace number_of_pages_l55_55042

-- Define the conditions
def rate_of_printer_A (P : ℕ) : ℕ := P / 60
def rate_of_printer_B (P : ℕ) : ℕ := (P / 60) + 6

-- Define the combined rate condition
def combined_rate (P : ℕ) (R_A R_B : ℕ) : Prop := (R_A + R_B) = P / 24

-- The main theorem to prove
theorem number_of_pages :
  ∃ (P : ℕ), combined_rate P (rate_of_printer_A P) (rate_of_printer_B P) ∧ P = 720 := by
  sorry

end number_of_pages_l55_55042


namespace train_length_proof_l55_55509

-- Define the conditions
def time_to_cross := 12 -- Time in seconds
def speed_km_per_h := 75 -- Speed in km/h

-- Convert the speed to m/s
def speed_m_per_s := speed_km_per_h * (5 / 18 : ℚ)

-- The length of the train using the formula: length = speed * time
def length_of_train := speed_m_per_s * (time_to_cross : ℚ)

-- The theorem to prove
theorem train_length_proof : length_of_train = 250 := by
  sorry

end train_length_proof_l55_55509


namespace min_value_of_a_plus_b_l55_55998

theorem min_value_of_a_plus_b (a b : ℤ) (h_ab : a * b = 72) (h_even : a % 2 = 0) : a + b ≥ -38 :=
sorry

end min_value_of_a_plus_b_l55_55998


namespace find_x_l55_55548

variables (a b c d x : ℤ)

theorem find_x (h1 : a - b = c + d + 9) (h2 : a - c = 3) (h3 : a + b = c - d - x) : x = 3 :=
sorry

end find_x_l55_55548


namespace keith_missed_games_l55_55968

-- Define the total number of football games
def total_games : ℕ := 8

-- Define the number of games Keith attended
def attended_games : ℕ := 4

-- Define the number of games played at night (although it is not directly necessary for the proof)
def night_games : ℕ := 4

-- Define the number of games Keith missed
def missed_games : ℕ := total_games - attended_games

-- Prove that the number of games Keith missed is 4
theorem keith_missed_games : missed_games = 4 := by
  sorry

end keith_missed_games_l55_55968


namespace num_stripes_on_us_flag_l55_55654

-- Definitions based on conditions in the problem
def num_stars : ℕ := 50

def num_circles : ℕ := (num_stars / 2) - 3

def num_squares (S : ℕ) : ℕ := 2 * S + 6

def total_shapes (num_squares : ℕ) : ℕ := num_circles + num_squares

-- The theorem stating the number of stripes
theorem num_stripes_on_us_flag (S : ℕ) (h1 : num_circles = 22) (h2 : total_shapes (num_squares S) = 54) : S = 13 := by
  sorry

end num_stripes_on_us_flag_l55_55654


namespace additional_stars_needed_l55_55180

-- Defining the number of stars required per bottle
def stars_per_bottle : Nat := 85

-- Defining the number of bottles Luke needs to fill
def bottles_to_fill : Nat := 4

-- Defining the number of stars Luke has already made
def stars_made : Nat := 33

-- Calculating the number of stars Luke still needs to make
theorem additional_stars_needed : (stars_per_bottle * bottles_to_fill - stars_made) = 307 := by
  sorry  -- Proof to be provided

end additional_stars_needed_l55_55180


namespace physics_marks_l55_55332

theorem physics_marks (P C M : ℕ) 
  (h1 : P + C + M = 180) 
  (h2 : P + M = 180) 
  (h3 : P + C = 140) : 
  P = 140 := 
by 
  sorry

end physics_marks_l55_55332


namespace overhead_cost_calculation_l55_55000

-- Define the production cost per performance
def production_cost_performance : ℕ := 7000

-- Define the revenue per sold-out performance
def revenue_per_soldout_performance : ℕ := 16000

-- Define the number of performances needed to break even
def break_even_performances : ℕ := 9

-- Prove the overhead cost
theorem overhead_cost_calculation (O : ℕ) :
  (O + break_even_performances * production_cost_performance = break_even_performances * revenue_per_soldout_performance) →
  O = 81000 :=
by
  sorry

end overhead_cost_calculation_l55_55000


namespace joan_balloons_l55_55392

-- Defining the condition
def melanie_balloons : ℕ := 41
def total_balloons : ℕ := 81

-- Stating the theorem
theorem joan_balloons :
  ∃ (joan_balloons : ℕ), joan_balloons = total_balloons - melanie_balloons ∧ joan_balloons = 40 :=
by
  -- Placeholder for the proof
  sorry

end joan_balloons_l55_55392


namespace total_drivers_l55_55383

theorem total_drivers (N : ℕ) (A : ℕ) (sA sB sC sD : ℕ) (total_sampled : ℕ)
  (hA : A = 96) (hsA : sA = 12) (hsB : sB = 21) (hsC : sC = 25) (hsD : sD = 43) (htotal : total_sampled = sA + sB + sC + sD)
  (hsA_proportion : (sA : ℚ) / A = (total_sampled : ℚ) / N) : N = 808 := by
  sorry

end total_drivers_l55_55383


namespace lines_intersect_l55_55520

structure Point where
  x : ℝ
  y : ℝ

def line1 (t : ℝ) : Point :=
  ⟨1 + 2 * t, 4 - 3 * t⟩

def line2 (u : ℝ) : Point :=
  ⟨5 + 4 * u, -2 - 5 * u⟩

theorem lines_intersect (x y t u : ℝ) 
  (h1 : x = 1 + 2 * t)
  (h2 : y = 4 - 3 * t)
  (h3 : x = 5 + 4 * u)
  (h4 : y = -2 - 5 * u) :
  x = 5 ∧ y = -2 := 
sorry

end lines_intersect_l55_55520


namespace trig_identity_solution_l55_55303

-- Define the necessary trigonometric functions
noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x
noncomputable def cot (x : ℝ) : ℝ := Real.cos x / Real.sin x

-- Statement of the theorem
theorem trig_identity_solution (x : ℝ) (k : ℤ) (hcos : Real.cos x ≠ 0) (hsin : Real.sin x ≠ 0) :
  (Real.sin x) ^ 2 * tan x + (Real.cos x) ^ 2 * cot x + 2 * Real.sin x * Real.cos x = (4 * Real.sqrt 3) / 3 →
  ∃ k : ℤ, x = (-1) ^ k * (Real.pi / 6) + (Real.pi / 2) :=
sorry

end trig_identity_solution_l55_55303


namespace find_certain_number_l55_55751

theorem find_certain_number (x certain_number : ℕ) (h: x = 3) (h2: certain_number = 5 * x + 4) : certain_number = 19 :=
by
  sorry

end find_certain_number_l55_55751


namespace white_area_of_sign_remains_l55_55746

theorem white_area_of_sign_remains (h1 : (6 * 18 = 108))
  (h2 : 9 = 6 + 3)
  (h3 : 7.5 = 5 + 3 - 0.5)
  (h4 : 13 = 9 + 4)
  (h5 : 9 = 6 + 3)
  (h6 : 38.5 = 9 + 7.5 + 13 + 9)
  : 108 - 38.5 = 69.5 := by
  sorry

end white_area_of_sign_remains_l55_55746


namespace total_gas_cost_l55_55048

theorem total_gas_cost 
  (x : ℝ)
  (cost_per_person_initial : ℝ := x / 5)
  (cost_per_person_new : ℝ := x / 8)
  (cost_difference : cost_per_person_initial - cost_per_person_new = 15) :
  x = 200 :=
sorry

end total_gas_cost_l55_55048


namespace smaller_number_eq_l55_55138

variable (m n t s : ℝ)
variable (h_ratio : m / n = t)
variable (h_sum : m + n = s)
variable (h_t_gt_one : t > 1)

theorem smaller_number_eq : n = s / (1 + t) :=
by sorry

end smaller_number_eq_l55_55138


namespace solve_inequality_l55_55763

theorem solve_inequality :
  {x : ℝ | 0 ≤ x^2 - x - 2 ∧ x^2 - x - 2 ≤ 4} = {x : ℝ | (-2 ≤ x ∧ x ≤ -1) ∨ (2 ≤ x ∧ x ≤ 3)} :=
sorry

end solve_inequality_l55_55763


namespace select_2n_comparable_rectangles_l55_55517

def comparable (A B : Rectangle) : Prop :=
  -- A can be placed into B by translation and rotation
  exists f : Rectangle → Rectangle, f A = B

theorem select_2n_comparable_rectangles (n : ℕ) (h : n > 1) :
  ∃ (rectangles : List Rectangle), rectangles.length = 2 * n ∧
  ∀ (a b : Rectangle), a ∈ rectangles → b ∈ rectangles → comparable a b :=
sorry

end select_2n_comparable_rectangles_l55_55517


namespace inequality_holds_l55_55557

theorem inequality_holds (k : ℝ) : (∀ x : ℝ, x^2 + k * x + 1 > 0) ↔ (k > -2 ∧ k < 2) :=
by
  sorry

end inequality_holds_l55_55557


namespace ratio_of_pieces_l55_55747

-- Define the total length of the wire.
def total_length : ℕ := 14

-- Define the length of the shorter piece.
def shorter_piece_length : ℕ := 4

-- Define the length of the longer piece.
def longer_piece_length : ℕ := total_length - shorter_piece_length

-- Define the expected ratio of the lengths.
def ratio : ℚ := shorter_piece_length / longer_piece_length

-- State the theorem to prove.
theorem ratio_of_pieces : ratio = 2 / 5 := 
by {
  -- skip the proof
  sorry
}

end ratio_of_pieces_l55_55747


namespace line_through_point_bisects_chord_l55_55641

theorem line_through_point_bisects_chord 
  (x y : ℝ) 
  (h_parabola : y^2 = 16 * x) 
  (h_point : 8 * 2 - 1 - 15 = 0) :
  8 * x - y - 15 = 0 :=
by
  sorry

end line_through_point_bisects_chord_l55_55641


namespace no_common_real_root_l55_55407

theorem no_common_real_root (a b : ℚ) : 
  ¬ ∃ (r : ℝ), (r^5 - r - 1 = 0) ∧ (r^2 + a * r + b = 0) :=
by
  sorry

end no_common_real_root_l55_55407


namespace chocolate_bar_cost_l55_55425

def total_bars := 11
def bars_left := 7
def bars_sold := total_bars - bars_left
def total_money := 16
def cost := total_money / bars_sold

theorem chocolate_bar_cost : cost = 4 :=
by
  sorry

end chocolate_bar_cost_l55_55425


namespace difference_between_numbers_l55_55214

variable (x y : ℕ)

theorem difference_between_numbers (h1 : x + y = 34) (h2 : y = 22) : y - x = 10 := by
  sorry

end difference_between_numbers_l55_55214


namespace start_page_day2_correct_l55_55043

variables (total_pages : ℕ) (percentage_read_day1 : ℝ) (start_page_day2 : ℕ)

theorem start_page_day2_correct
  (h1 : total_pages = 200)
  (h2 : percentage_read_day1 = 0.2)
  : start_page_day2 = total_pages * percentage_read_day1 + 1 :=
by
  sorry

end start_page_day2_correct_l55_55043


namespace num_girls_l55_55772

theorem num_girls (boys girls : ℕ) (h1 : girls = boys + 228) (h2 : boys = 469) : girls = 697 :=
sorry

end num_girls_l55_55772


namespace prime_product_is_2009_l55_55710

theorem prime_product_is_2009 (a b c : ℕ) 
  (h_primeA : Prime a) 
  (h_primeB : Prime b) 
  (h_primeC : Prime c)
  (h_div1 : a ∣ (b + 8)) 
  (h_div2a : a ∣ (b^2 - 1)) 
  (h_div2c : c ∣ (b^2 - 1)) 
  (h_sum : b + c = a^2 - 1) : 
  a * b * c = 2009 := 
sorry

end prime_product_is_2009_l55_55710


namespace sum_max_min_values_of_g_l55_55927

def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2 * x - 8| + 3

theorem sum_max_min_values_of_g : (∀ x, 1 ≤ x ∧ x ≤ 7 → g x = 15 - 2 * x ∨ g x = 5) ∧ 
      (g 1 = 13 ∧ g 5 = 5)
      → (13 + 5 = 18) :=
by
  sorry

end sum_max_min_values_of_g_l55_55927


namespace equilateral_triangle_perimeter_l55_55368

theorem equilateral_triangle_perimeter (x : ℕ) (h : 2 * x = x + 15) : 
  3 * (2 * x) = 90 :=
by
  -- Definitions & hypothesis
  sorry

end equilateral_triangle_perimeter_l55_55368


namespace filling_tank_ratio_l55_55515

theorem filling_tank_ratio :
  ∀ (t : ℝ),
    (1 / 40) * t + (1 / 24) * (29.999999999999993 - t) = 1 →
    t / 29.999999999999993 = 1 / 2 :=
by
  intro t
  intro H
  sorry

end filling_tank_ratio_l55_55515


namespace non_obtuse_triangle_medians_ge_4R_l55_55051

theorem non_obtuse_triangle_medians_ge_4R
  (A B C : Type*)
  (triangle_non_obtuse : ∀ (α β γ : ℝ), α ≤ 90 ∧ β ≤ 90 ∧ γ ≤ 90)
  (m_a m_b m_c : ℝ)
  (R : ℝ)
  (h1 : AO + BO ≤ AM + BM)
  (h2 : AM = 2 * m_a / 3 ∧ BM = 2 * m_b / 3)
  (h3 : AO + BO = 2 * R)
  (h4 : m_c ≥ R) : 
  m_a + m_b + m_c ≥ 4 * R :=
by
  sorry

end non_obtuse_triangle_medians_ge_4R_l55_55051


namespace josh_money_left_l55_55164

theorem josh_money_left (initial_amount : ℝ) (first_spend : ℝ) (second_spend : ℝ) 
  (h1 : initial_amount = 9) 
  (h2 : first_spend = 1.75) 
  (h3 : second_spend = 1.25) : 
  initial_amount - first_spend - second_spend = 6 := 
by 
  sorry

end josh_money_left_l55_55164


namespace john_payment_l55_55873

def total_cost (cakes : ℕ) (cost_per_cake : ℕ) : ℕ :=
  cakes * cost_per_cake

def split_cost (total : ℕ) (people : ℕ) : ℕ :=
  total / people

theorem john_payment (cakes : ℕ) (cost_per_cake : ℕ) (people : ℕ) : 
  cakes = 3 → cost_per_cake = 12 → people = 2 → 
  split_cost (total_cost cakes cost_per_cake) people = 18 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end john_payment_l55_55873


namespace orchestra_member_count_l55_55727

theorem orchestra_member_count :
  ∃ x : ℕ, 150 ≤ x ∧ x ≤ 250 ∧ 
           x % 4 = 2 ∧
           x % 5 = 3 ∧
           x % 8 = 4 ∧
           x % 9 = 5 :=
sorry

end orchestra_member_count_l55_55727


namespace arithmetic_sequence_sum_l55_55943

variable (a : ℕ → ℕ)
variable (h_arith_seq : ∀ n : ℕ, a (n+1) - a n = a 2 - a 1)

theorem arithmetic_sequence_sum (h : a 2 + a 8 = 6) : 
  1 / 2 * 9 * (a 1 + a 9) = 27 :=
by 
  sorry

end arithmetic_sequence_sum_l55_55943


namespace f_zero_f_increasing_on_negative_l55_55176

noncomputable def f : ℝ → ℝ := sorry
variable {x : ℝ}

-- Assume f is an odd function
axiom odd_f : ∀ x, f (-x) = -f x

-- Assume f is increasing on (0, +∞)
axiom increasing_f_on_positive :
  ∀ ⦃x₁ x₂⦄, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂

-- Prove that f(0) = 0
theorem f_zero : f 0 = 0 := sorry

-- Prove that f is increasing on (-∞, 0)
theorem f_increasing_on_negative :
  ∀ ⦃x₁ x₂⦄, x₁ < x₂ → x₂ < 0 → f x₁ < f x₂ := sorry

end f_zero_f_increasing_on_negative_l55_55176


namespace percent_of_l55_55449

theorem percent_of (Part Whole : ℕ) (Percent : ℕ) (hPart : Part = 120) (hWhole : Whole = 40) :
  Percent = (Part * 100) / Whole → Percent = 300 :=
by
  sorry

end percent_of_l55_55449


namespace find_y_l55_55823

theorem find_y (y : ℕ) 
  (h : (1/8) * 2^36 = 8^y) : y = 11 :=
sorry

end find_y_l55_55823


namespace simplify_fraction_l55_55812

theorem simplify_fraction (x y z : ℝ) (hx : x = 5) (hz : z = 2) : (10 * x * y * z) / (15 * x^2 * z) = (2 * y) / 15 :=
by
  sorry

end simplify_fraction_l55_55812


namespace number_of_non_representable_l55_55947

theorem number_of_non_representable :
  ∀ (a b : ℕ), Nat.gcd a b = 1 →
  (∃ n : ℕ, ¬ ∃ x y : ℕ, n = a * x + b * y) :=
sorry

end number_of_non_representable_l55_55947


namespace max_value_of_expression_l55_55758

noncomputable def maximum_value {x y : ℝ} (hx : 0 < x) (hy : 0 < y) 
  (h : x^2 - x*y + 2*y^2 = 8) : ℝ :=
  x^2 + x*y + 2*y^2

theorem max_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : x^2 - x*y + 2*y^2 = 8) : maximum_value hx hy h = (72 + 32 * Real.sqrt 2) / 7 :=
by
  sorry

end max_value_of_expression_l55_55758


namespace scientific_notation_of_935million_l55_55489

theorem scientific_notation_of_935million :
  935000000 = 9.35 * 10 ^ 8 :=
  sorry

end scientific_notation_of_935million_l55_55489


namespace common_number_of_two_sets_l55_55420

theorem common_number_of_two_sets (a b c d e f g : ℚ) :
  (a + b + c + d) / 4 = 5 →
  (d + e + f + g) / 4 = 8 →
  (a + b + c + d + e + f + g) / 7 = 46 / 7 →
  d = 6 :=
by
  intros h₁ h₂ h₃
  sorry

end common_number_of_two_sets_l55_55420


namespace log_expression_value_l55_55007

noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_expression_value :
  log10 8 + 3 * log10 4 - 2 * log10 2 + 4 * log10 25 + log10 16 = 11 := by
  sorry

end log_expression_value_l55_55007


namespace integer_solutions_l55_55501

theorem integer_solutions :
  ∃ (a b c : ℤ), a + b + c = 24 ∧ a^2 + b^2 + c^2 = 210 ∧ a * b * c = 440 ∧
    (a = 5 ∧ b = 8 ∧ c = 11) ∨ (a = 5 ∧ b = 11 ∧ c = 8) ∨ 
    (a = 8 ∧ b = 5 ∧ c = 11) ∨ (a = 8 ∧ b = 11 ∧ c = 5) ∨
    (a = 11 ∧ b = 5 ∧ c = 8) ∨ (a = 11 ∧ b = 8 ∧ c = 5) :=
sorry

end integer_solutions_l55_55501


namespace arithmetic_expression_equiv_l55_55059

theorem arithmetic_expression_equiv :
  (-1:ℤ)^2009 * (-3) + 1 - 2^2 * 3 + (1 - 2^2) / 3 + (1 - 2 * 3)^2 = 16 := by
  sorry

end arithmetic_expression_equiv_l55_55059


namespace math_competition_l55_55263

theorem math_competition (a b c d e f g : ℕ) (h1 : a + b + c + d + e + f + g = 25)
    (h2 : b = 2 * c + f) (h3 : a = d + e + g + 1) (h4 : a = b + c) :
    b = 6 :=
by
  -- The proof is omitted as the problem requests the statement only.
  sorry

end math_competition_l55_55263


namespace certain_number_divisibility_l55_55570

-- Define the conditions and the main problem statement
theorem certain_number_divisibility (n : ℕ) (h1 : 0 < n) (h2 : n < 11) (h3 : (18888 - n) % k = 0) (h4 : n = 1) : k = 11 :=
by
  sorry

end certain_number_divisibility_l55_55570


namespace inequality_problem_l55_55496

theorem inequality_problem (a b : ℝ) (h₁ : 1/a < 1/b) (h₂ : 1/b < 0) :
  (∃ (p q : Prop), 
    (p ∧ q) ∧ 
    ((p ↔ (a + b < a * b)) ∧ 
    (¬q ↔ |a| ≤ |b|) ∧ 
    (¬q ↔ a > b) ∧ 
    (q ↔ (b / a + a / b > 2)))) :=
sorry

end inequality_problem_l55_55496


namespace total_profit_is_42000_l55_55813

noncomputable def total_profit (I_B T_B : ℝ) :=
  let I_A := 3 * I_B
  let T_A := 2 * T_B
  let profit_B := I_B * T_B
  let profit_A := I_A * T_A
  profit_A + profit_B

theorem total_profit_is_42000
  (I_B T_B : ℝ)
  (h1 : I_A = 3 * I_B)
  (h2 : T_A = 2 * T_B)
  (h3 : I_B * T_B = 6000) :
  total_profit I_B T_B = 42000 := by
  sorry

end total_profit_is_42000_l55_55813


namespace abs_lt_five_implies_interval_l55_55612

theorem abs_lt_five_implies_interval (x : ℝ) : |x| < 5 → -5 < x ∧ x < 5 := by
  sorry

end abs_lt_five_implies_interval_l55_55612


namespace find_distance_l55_55630

-- Conditions: total cost, base price, cost per mile
variables (total_cost base_price cost_per_mile : ℕ)

-- Definition of the distance as per the problem
def distance_from_home_to_hospital (total_cost base_price cost_per_mile : ℕ) : ℕ :=
  (total_cost - base_price) / cost_per_mile

-- Given values:
def total_cost_value : ℕ := 23
def base_price_value : ℕ := 3
def cost_per_mile_value : ℕ := 4

-- The theorem that encapsulates the problem statement
theorem find_distance :
  distance_from_home_to_hospital total_cost_value base_price_value cost_per_mile_value = 5 :=
by
  -- Placeholder for the proof
  sorry

end find_distance_l55_55630


namespace statement_b_statement_e_l55_55063

-- Statement (B): ∀ x, if x^3 > 0 then x > 0.
theorem statement_b (x : ℝ) : x^3 > 0 → x > 0 := sorry

-- Statement (E): ∀ x, if x < 1 then x^3 < x.
theorem statement_e (x : ℝ) : x < 1 → x^3 < x := sorry

end statement_b_statement_e_l55_55063


namespace coefficient_x7_in_expansion_l55_55201

theorem coefficient_x7_in_expansion : 
  let n := 10
  let k := 7
  let binom := Nat.choose n k
  let coeff := 1
  coeff * binom = 120 :=
by
  sorry

end coefficient_x7_in_expansion_l55_55201


namespace max_area_rectangle_with_perimeter_40_l55_55780

theorem max_area_rectangle_with_perimeter_40 :
  ∃ (l w : ℕ), 2 * l + 2 * w = 40 ∧ l * w = 100 :=
sorry

end max_area_rectangle_with_perimeter_40_l55_55780


namespace dot_product_result_parallelism_condition_l55_55744

-- Definitions of the vectors
def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 2)

-- 1. Prove the dot product result
theorem dot_product_result :
  let a_plus_b := (a.1 + b.1, a.2 + b.2)
  let a_minus_2b := (a.1 - 2 * b.1, a.2 - 2 * b.2)
  a_plus_b.1 * a_minus_2b.1 + a_plus_b.2 * a_minus_2b.2 = -14 :=
by
  sorry

-- 2. Prove parallelism condition
theorem parallelism_condition (k : ℝ) :
  let k_a_plus_b := (k * a.1 + b.1, k * a.2 + b.2)
  let a_minus_3b := (a.1 - 3 * b.1, a.2 - 3 * b.2)
  k = -1/3 → k_a_plus_b.1 * a_minus_3b.2 = k_a_plus_b.2 * a_minus_3b.1 :=
by
  sorry

end dot_product_result_parallelism_condition_l55_55744


namespace cycling_route_length_l55_55270

-- Conditions (segment lengths)
def segment1 : ℝ := 4
def segment2 : ℝ := 7
def segment3 : ℝ := 2
def segment4 : ℝ := 6
def segment5 : ℝ := 7

-- Specify the total length calculation
noncomputable def total_length : ℝ :=
  2 * (segment1 + segment2 + segment3) + 2 * (segment4 + segment5)

-- The theorem we want to prove
theorem cycling_route_length :
  total_length = 52 :=
by
  sorry

end cycling_route_length_l55_55270


namespace max_area_parabola_l55_55038

open Real

noncomputable def max_area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

theorem max_area_parabola (a b c : ℝ) 
  (ha : a^2 = (a * a))
  (hb : b^2 = (b * b))
  (hc : c^2 = (c * c))
  (centroid_cond1 : (a + b + c) = 4)
  (centroid_cond2 : (a^2 + b^2 + c^2) = 6)
  : max_area_of_triangle (a^2, a) (b^2, b) (c^2, c) = (sqrt 3) / 9 := 
sorry

end max_area_parabola_l55_55038


namespace selected_40th_is_795_l55_55144

-- Definitions of constants based on the problem conditions
def total_participants : ℕ := 1000
def selections : ℕ := 50
def equal_spacing : ℕ := total_participants / selections
def first_selected_number : ℕ := 15
def nth_selected_number (n : ℕ) : ℕ := (n - 1) * equal_spacing + first_selected_number

-- The theorem to prove the 40th selected number is 795
theorem selected_40th_is_795 : nth_selected_number 40 = 795 := 
by 
  -- Skipping the detailed proof
  sorry

end selected_40th_is_795_l55_55144


namespace solve_fractional_equation_l55_55508

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 4) : 
  (3 - x) / (x - 4) + 1 / (4 - x) = 1 → x = 3 :=
by {
  sorry
}

end solve_fractional_equation_l55_55508


namespace exists_triang_and_square_le_50_l55_55398

def is_triang_num (n : ℕ) : Prop := ∃ m : ℕ, n = m * (m + 1) / 2
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem exists_triang_and_square_le_50 : ∃ n : ℕ, n ≤ 50 ∧ is_triang_num n ∧ is_perfect_square n :=
by
  sorry

end exists_triang_and_square_le_50_l55_55398


namespace minimize_sum_of_reciprocals_l55_55123

def dataset : List ℝ := [2, 4, 6, 8]

def mean : ℝ := 5
def variance: ℝ := 5

theorem minimize_sum_of_reciprocals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : mean * a + variance * b = 1) : 
  (1 / a + 1 / b) = 20 :=
sorry

end minimize_sum_of_reciprocals_l55_55123


namespace taxi_ride_cost_l55_55016

-- Define the fixed cost
def fixed_cost : ℝ := 2.00

-- Define the cost per mile
def cost_per_mile : ℝ := 0.30

-- Define the number of miles traveled
def miles_traveled : ℝ := 7.0

-- Define the total cost calculation
def total_cost : ℝ := fixed_cost + (cost_per_mile * miles_traveled)

-- Theorem: Prove the total cost of a 7-mile taxi ride is $4.10
theorem taxi_ride_cost : total_cost = 4.10 := by
  sorry

end taxi_ride_cost_l55_55016


namespace contemporaries_probability_l55_55281

open Real

noncomputable def probability_of_contemporaries
  (born_within : ℝ) (lifespan : ℝ) : ℝ :=
  let total_area := born_within * born_within
  let side := born_within - lifespan
  let non_overlap_area := 2 * (1/2 * side * side)
  let overlap_area := total_area - non_overlap_area
  overlap_area / total_area

theorem contemporaries_probability :
  probability_of_contemporaries 300 80 = 104 / 225 := 
by
  sorry

end contemporaries_probability_l55_55281


namespace garden_area_increase_l55_55326

theorem garden_area_increase :
    let length := 60
    let width := 20
    let perimeter := 2 * (length + width)
    let side_of_square := perimeter / 4
    let area_rectangular := length * width
    let area_square := side_of_square * side_of_square
    area_square - area_rectangular = 400 :=
by
  sorry

end garden_area_increase_l55_55326


namespace planks_needed_l55_55199

theorem planks_needed (total_nails : ℕ) (nails_per_plank : ℕ) (h1 : total_nails = 4) (h2 : nails_per_plank = 2) : total_nails / nails_per_plank = 2 :=
by
  -- Prove that given the conditions, the required result is obtained
  sorry

end planks_needed_l55_55199


namespace ratio_twice_width_to_length_l55_55200

-- Given conditions:
def length_of_field : ℚ := 24
def width_of_field : ℚ := 13.5

-- The problem is to prove the ratio of twice the width to the length of the field is 9/8
theorem ratio_twice_width_to_length : 2 * width_of_field / length_of_field = 9 / 8 :=
by sorry

end ratio_twice_width_to_length_l55_55200


namespace white_tshirt_cost_l55_55471

-- Define the problem conditions
def total_tshirts : ℕ := 200
def total_minutes : ℕ := 25
def black_tshirt_cost : ℕ := 30
def revenue_per_minute : ℕ := 220

-- Prove the cost of white t-shirts given the conditions
theorem white_tshirt_cost : 
  (total_tshirts / 2) * revenue_per_minute * total_minutes 
  - (total_tshirts / 2) * black_tshirt_cost = 2500
  → 2500 / (total_tshirts / 2) = 25 :=
by
  sorry

end white_tshirt_cost_l55_55471


namespace integer_a_satisfies_equation_l55_55558

theorem integer_a_satisfies_equation (a b c : ℤ) :
  (∃ b c : ℤ, (x - a) * (x - 5) + 2 = (x + b) * (x + c)) → 
    a = 2 :=
by
  intro h_eq
  -- Proof goes here
  sorry

end integer_a_satisfies_equation_l55_55558


namespace value_of_a_plus_b_l55_55790

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  if 0 ≤ x then Real.sqrt x + 3 else a * x + b

theorem value_of_a_plus_b (a b : ℝ) 
  (h1 : ∀ x1 : ℝ, x1 ≠ 0 → ∃ x2 : ℝ, x1 ≠ x2 ∧ f x1 a b = f x2 a b)
  (h2 : f (2 * a) a b = f (3 * b) a b) :
  a + b = - (Real.sqrt 6) / 2 + 3 :=
by
  sorry

end value_of_a_plus_b_l55_55790


namespace sum_of_consecutive_integers_is_33_l55_55958

theorem sum_of_consecutive_integers_is_33 :
  ∃ (x : ℕ), x * (x + 1) = 272 ∧ x + (x + 1) = 33 :=
by
  sorry

end sum_of_consecutive_integers_is_33_l55_55958


namespace fish_market_customers_l55_55881

theorem fish_market_customers :
  let num_tuna := 10
  let weight_per_tuna := 200
  let weight_per_customer := 25
  let num_customers_no_fish := 20
  let total_tuna_weight := num_tuna * weight_per_tuna
  let num_customers_served := total_tuna_weight / weight_per_customer
  num_customers_served + num_customers_no_fish = 100 := 
by
  sorry

end fish_market_customers_l55_55881


namespace percentage_return_on_investment_l55_55808

theorem percentage_return_on_investment
  (dividend_rate : ℝ)
  (face_value : ℝ)
  (purchase_price : ℝ)
  (dividend_per_share : ℝ := (dividend_rate / 100) * face_value)
  (percentage_return : ℝ := (dividend_per_share / purchase_price) * 100)
  (h1 : dividend_rate = 15.5)
  (h2 : face_value = 50)
  (h3 : purchase_price = 31) :
  percentage_return = 25 := by
    sorry

end percentage_return_on_investment_l55_55808


namespace expression_divisible_by_a_square_l55_55445

theorem expression_divisible_by_a_square (n : ℕ) (a : ℤ) : 
  a^2 ∣ ((a * n - 1) * (a + 1) ^ n + 1) := 
sorry

end expression_divisible_by_a_square_l55_55445


namespace largest_fraction_l55_55384

noncomputable def compare_fractions : List ℚ :=
  [5 / 11, 7 / 16, 9 / 20, 11 / 23, 111 / 245, 145 / 320, 185 / 409, 211 / 465, 233 / 514]

theorem largest_fraction :
  max (5 / 11) (max (7 / 16) (max (9 / 20) (max (11 / 23) (max (111 / 245) (max (145 / 320) (max (185 / 409) (max (211 / 465) (233 / 514)))))))) = 11 / 23 := 
  sorry

end largest_fraction_l55_55384


namespace perpendicular_bisector_correct_vertex_C_correct_l55_55959

-- Define the vertices A, B, and the coordinates of the angle bisector line
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 1, y := 2 }
def B : Point := { x := -1, y := -1 }

-- The angle bisector CD equation
def angle_bisector_CD (p : Point) : Prop :=
  p.x + p.y - 1 = 0

-- The perpendicular bisector equation of side AB
def perpendicular_bisector_AB (p : Point) : Prop :=
  4 * p.x + 6 * p.y - 3 = 0

-- Coordinates of vertex C
def C_coordinates (c : Point) : Prop :=
  c.x = -1 ∧ c.y = 2

theorem perpendicular_bisector_correct :
  ∀ (M : Point), M.x = 0 ∧ M.y = 1/2 →
  ∀ (p : Point), perpendicular_bisector_AB p :=
sorry

theorem vertex_C_correct :
  ∃ (C : Point), angle_bisector_CD C ∧ (C : Point) = { x := -1, y := 2 } :=
sorry

end perpendicular_bisector_correct_vertex_C_correct_l55_55959


namespace quotient_of_x6_plus_8_by_x_minus_1_l55_55323

theorem quotient_of_x6_plus_8_by_x_minus_1 :
  ∀ (x : ℝ), x ≠ 1 →
  (∃ Q : ℝ → ℝ, x^6 + 8 = (x - 1) * Q x + 9 ∧ Q x = x^5 + x^4 + x^3 + x^2 + x + 1) := 
  by
    intros x hx
    sorry

end quotient_of_x6_plus_8_by_x_minus_1_l55_55323


namespace scarlet_savings_l55_55581

noncomputable def remaining_savings (initial_savings earrings_cost necklace_cost bracelet_cost jewelry_set_cost jewelry_set_discount sales_tax_percentage : ℝ) : ℝ :=
  let total_item_cost := earrings_cost + necklace_cost + bracelet_cost
  let discounted_jewelry_set_cost := jewelry_set_cost * (1 - jewelry_set_discount / 100)
  let total_cost_before_tax := total_item_cost + discounted_jewelry_set_cost
  let total_sales_tax := total_cost_before_tax * (sales_tax_percentage / 100)
  let final_total_cost := total_cost_before_tax + total_sales_tax
  initial_savings - final_total_cost

theorem scarlet_savings : remaining_savings 200 23 48 35 80 25 5 = 25.70 :=
by
  sorry

end scarlet_savings_l55_55581


namespace general_formula_a_n_sum_first_n_b_l55_55194

-- Define the sequence {a_n}
def a_n (n : ℕ) : ℕ := 2 * n + 1

-- Sequence property
def seq_property (n : ℕ) (S_n : ℕ) : Prop :=
  a_n n ^ 2 + 2 * a_n n = 4 * S_n + 3

-- General formula for {a_n}
theorem general_formula_a_n (n : ℕ) (hpos : ∀ n, a_n n > 0) (S_n : ℕ) (hseq : seq_property n S_n) :
  a_n n = 2 * n + 1 :=
sorry

-- Sum of the first n terms of {b_n}
def b_n (n : ℕ) : ℚ := 1 / ((a_n n) * (a_n (n + 1)))

def sum_b (n : ℕ) (T_n : ℚ) : Prop :=
  T_n = (1 / 2) * ((1 / (2 * n + 1)) - (1 / (2 * n + 3)))

theorem sum_first_n_b (n : ℕ) (hpos : ∀ n, a_n n > 0) (T_n : ℚ) :
  T_n = (n : ℚ) / (3 * (2 * n + 3)) :=
sorry

end general_formula_a_n_sum_first_n_b_l55_55194


namespace birds_are_crows_l55_55299

theorem birds_are_crows (total_birds pigeons crows sparrows parrots non_pigeons: ℕ)
    (h1: pigeons = 20)
    (h2: crows = 40)
    (h3: sparrows = 15)
    (h4: parrots = total_birds - pigeons - crows - sparrows)
    (h5: total_birds = pigeons + crows + sparrows + parrots)
    (h6: non_pigeons = total_birds - pigeons) :
    (crows * 100 / non_pigeons = 50) :=
by sorry

end birds_are_crows_l55_55299


namespace percentage_of_people_win_a_prize_l55_55819

-- Define the constants used in the problem
def totalMinnows : Nat := 600
def minnowsPerPrize : Nat := 3
def totalPlayers : Nat := 800
def minnowsLeft : Nat := 240

-- Calculate the number of minnows given away as prizes
def minnowsGivenAway : Nat := totalMinnows - minnowsLeft

-- Calculate the number of prizes given away
def prizesGivenAway : Nat := minnowsGivenAway / minnowsPerPrize

-- Calculate the percentage of people winning a prize
def percentageWinners : Nat := (prizesGivenAway * 100) / totalPlayers

-- Theorem to prove the percentage of winners
theorem percentage_of_people_win_a_prize : 
    percentageWinners = 15 := 
sorry

end percentage_of_people_win_a_prize_l55_55819


namespace Ellen_strawberries_used_l55_55588

theorem Ellen_strawberries_used :
  let yogurt := 0.1
  let orange_juice := 0.2
  let total_ingredients := 0.5
  let strawberries := total_ingredients - (yogurt + orange_juice)
  strawberries = 0.2 :=
by
  sorry

end Ellen_strawberries_used_l55_55588


namespace train_boarding_probability_l55_55057

theorem train_boarding_probability :
  (0.5 / 5) = 1 / 10 :=
by sorry

end train_boarding_probability_l55_55057


namespace domain_of_sqrt_function_l55_55235

theorem domain_of_sqrt_function (x : ℝ) :
  (x + 4 ≥ 0) ∧ (1 - x ≥ 0) ∧ (x ≠ 0) ↔ (-4 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1) := 
sorry

end domain_of_sqrt_function_l55_55235


namespace find_three_digit_number_l55_55448

noncomputable def three_digit_number := ∃ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧ 100 * x + 10 * y + z = 345 ∧
  (100 * z + 10 * y + x = 100 * x + 10 * y + z + 198) ∧
  (100 * x + 10 * z + y = 100 * x + 10 * y + z + 9) ∧
  (x^2 + y^2 + z^2 - 2 = 4 * (x + y + z))

theorem find_three_digit_number : three_digit_number :=
sorry

end find_three_digit_number_l55_55448


namespace train_return_time_l55_55376

open Real

theorem train_return_time
  (C_small : Real := 1.5)
  (C_large : Real := 3)
  (speed : Real := 10)
  (initial_connection : String := "A to C")
  (switch_interval : Real := 1) :
  (126 = 2.1 * 60) :=
sorry

end train_return_time_l55_55376


namespace infinite_indices_exist_l55_55074

theorem infinite_indices_exist (a : ℕ → ℕ) (h_seq : ∀ n, a n < a (n + 1)) :
  ∃ᶠ m in ⊤, ∃ x y h k : ℕ, 0 < h ∧ h < k ∧ k < m ∧ a m = x * a h + y * a k :=
by sorry

end infinite_indices_exist_l55_55074


namespace brownies_left_is_zero_l55_55168

-- Definitions of the conditions
def total_brownies : ℝ := 24
def tina_lunch : ℝ := 1.5 * 5
def tina_dinner : ℝ := 0.5 * 5
def tina_total : ℝ := tina_lunch + tina_dinner
def husband_total : ℝ := 0.75 * 5
def guests_total : ℝ := 2.5 * 2
def daughter_total : ℝ := 2 * 3

-- Formulate the proof statement
theorem brownies_left_is_zero :
    total_brownies - (tina_total + husband_total + guests_total + daughter_total) = 0 := by
  sorry

end brownies_left_is_zero_l55_55168


namespace yi_successful_shots_l55_55743

-- Defining the basic conditions
variables {x y : ℕ} -- Number of successful shots made by Jia and Yi respectively

-- Each hit gains 20 points and each miss deducts 12 points.
-- Both person A (Jia) and person B (Yi) made 10 shots each.
def total_shots (x y : ℕ) : Prop := 
  (20 * x - 12 * (10 - x)) + (20 * y - 12 * (10 - y)) = 208 ∧ x + y = 14 ∧ x - y = 2

theorem yi_successful_shots (x y : ℕ) (h : total_shots x y) : y = 6 := 
  by sorry

end yi_successful_shots_l55_55743


namespace rancher_steers_cows_solution_l55_55750

theorem rancher_steers_cows_solution :
  ∃ (s c : ℕ), s > 0 ∧ c > 0 ∧ (30 * s + 31 * c = 1200) ∧ (s = 9) ∧ (c = 30) :=
by
  sorry

end rancher_steers_cows_solution_l55_55750


namespace remainder_of_349_by_17_is_9_l55_55625

theorem remainder_of_349_by_17_is_9 :
  349 % 17 = 9 :=
sorry

end remainder_of_349_by_17_is_9_l55_55625


namespace geometric_sum_eight_terms_l55_55069

theorem geometric_sum_eight_terms :
  let a0 := (1 : ℚ) / 3
  let r := (1 : ℚ) / 4
  let n := 8
  let S_n := a0 * (1 - r^n) / (1 - r)
  S_n = 65535 / 147456 := by
  sorry

end geometric_sum_eight_terms_l55_55069


namespace ratio_of_u_to_v_l55_55954

theorem ratio_of_u_to_v (b u v : ℝ) (Hu : u = -b/12) (Hv : v = -b/8) : 
  u / v = 2 / 3 := 
sorry

end ratio_of_u_to_v_l55_55954


namespace perfect_square_iff_l55_55166

theorem perfect_square_iff (A : ℕ) : (∃ k : ℕ, A = k^2) ↔ (∀ n : ℕ, n > 0 → ∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ n ∣ ((A + k)^2 - A)) :=
by
  sorry

end perfect_square_iff_l55_55166


namespace probability_C_l55_55423

-- Definitions of probabilities
def P_A : ℚ := 3 / 8
def P_B : ℚ := 1 / 4
def P_D : ℚ := 1 / 8

-- Main proof statement
theorem probability_C :
  ∀ P_C : ℚ, P_A + P_B + P_C + P_D = 1 → P_C = 1 / 4 :=
by
  intro P_C h
  sorry

end probability_C_l55_55423


namespace triangle_inequality_l55_55365

variable (a b c : ℝ) -- sides of the triangle
variable (h_a h_b h_c S r R : ℝ) -- heights, area of the triangle, inradius, circumradius

-- Definitions of conditions
axiom h_def : h_a + h_b + h_c = (a + b + c) -- express heights sum in terms of sides sum (for illustrative purposes)
axiom S_def : S = 0.5 * a * h_a  -- area definition (adjust as needed)
axiom r_def : 9 * r ≤ h_a + h_b + h_c -- given in solution
axiom R_def : h_a + h_b + h_c ≤ 9 * R / 2 -- given in solution

theorem triangle_inequality :
  9 * r / (2 * S) ≤ (1 / a) + (1 / b) + (1 / c) ∧ (1 / a) + (1 / b) + (1 / c) ≤ 9 * R / (4 * S) :=
by
  sorry

end triangle_inequality_l55_55365


namespace distance_between_Sneezy_and_Grumpy_is_8_l55_55478

variables (DS DV SP VP: ℕ) (SV: ℕ)

theorem distance_between_Sneezy_and_Grumpy_is_8
  (hDS : DS = 5)
  (hDV : DV = 4)
  (hSP : SP = 10)
  (hVP : VP = 17)
  (hSV_condition1 : SV + SP > VP)
  (hSV_condition2 : SV < DS + DV)
  (hSV_condition3 : 7 < SV) :
  SV = 8 := 
sorry

end distance_between_Sneezy_and_Grumpy_is_8_l55_55478


namespace option_B_is_one_variable_quadratic_l55_55262

theorem option_B_is_one_variable_quadratic :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x : ℝ, 2 * (x - x^2) - 1 = a * x^2 + b * x + c) :=
by
  sorry

end option_B_is_one_variable_quadratic_l55_55262


namespace balloons_lost_l55_55100

theorem balloons_lost (initial remaining : ℕ) (h_initial : initial = 9) (h_remaining : remaining = 7) : initial - remaining = 2 := by
  sorry

end balloons_lost_l55_55100


namespace smallest_n_l55_55089

theorem smallest_n {n : ℕ} (h1 : n ≡ 4 [MOD 6]) (h2 : n ≡ 3 [MOD 7]) (h3 : n > 10) : n = 52 :=
sorry

end smallest_n_l55_55089


namespace softball_players_count_l55_55715

theorem softball_players_count :
  ∀ (cricket hockey football total_players softball : ℕ),
  cricket = 15 →
  hockey = 12 →
  football = 13 →
  total_players = 55 →
  total_players = cricket + hockey + football + softball →
  softball = 15 :=
by
  intros cricket hockey football total_players softball h_cricket h_hockey h_football h_total_players h_total
  sorry

end softball_players_count_l55_55715


namespace percentage_of_trout_is_correct_l55_55340

-- Define the conditions
def video_game_cost := 60
def last_weekend_earnings := 35
def earnings_per_trout := 5
def earnings_per_bluegill := 4
def total_fish_caught := 5
def additional_savings_needed := 2

-- Define the total amount needed to buy the game
def total_required_savings := video_game_cost - additional_savings_needed

-- Define the amount earned this Sunday
def earnings_this_sunday := total_required_savings - last_weekend_earnings

-- Define the number of trout and blue-gill caught thisSunday
def num_trout := 3
def num_bluegill := 2    -- Derived from the conditions

-- Theorem: given the conditions, prove that the percentage of trout is 60%
theorem percentage_of_trout_is_correct :
  (num_trout + num_bluegill = total_fish_caught) ∧
  (earnings_per_trout * num_trout + earnings_per_bluegill * num_bluegill = earnings_this_sunday) →
  100 * num_trout / total_fish_caught = 60 := 
by
  sorry

end percentage_of_trout_is_correct_l55_55340


namespace total_bills_is_126_l55_55143

noncomputable def F : ℕ := 84  -- number of 5-dollar bills
noncomputable def T : ℕ := (840 - 5 * F) / 10  -- derive T based on the total value and F
noncomputable def total_bills : ℕ := F + T

theorem total_bills_is_126 : total_bills = 126 :=
by
  -- Placeholder for the proof
  sorry

end total_bills_is_126_l55_55143


namespace points_on_opposite_sides_l55_55976

theorem points_on_opposite_sides (a : ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) 
    (hA : A = (a, 1)) 
    (hB : B = (2, a)) 
    (opposite_sides : A.1 < 0 ∧ B.1 > 0 ∨ A.1 > 0 ∧ B.1 < 0) 
    : a < 0 := 
  sorry

end points_on_opposite_sides_l55_55976


namespace cubed_multiplication_identity_l55_55400

theorem cubed_multiplication_identity : 3^3 * 6^3 = 5832 := by
  sorry

end cubed_multiplication_identity_l55_55400


namespace charlie_extra_fee_l55_55732

-- Conditions
def data_limit_week1 : ℕ := 2 -- in GB
def data_limit_week2 : ℕ := 3 -- in GB
def data_limit_week3 : ℕ := 2 -- in GB
def data_limit_week4 : ℕ := 1 -- in GB

def additional_fee_week1 : ℕ := 12 -- dollars per GB
def additional_fee_week2 : ℕ := 10 -- dollars per GB
def additional_fee_week3 : ℕ := 8 -- dollars per GB
def additional_fee_week4 : ℕ := 6 -- dollars per GB

def data_used_week1 : ℕ := 25 -- in 0.1 GB
def data_used_week2 : ℕ := 40 -- in 0.1 GB
def data_used_week3 : ℕ := 30 -- in 0.1 GB
def data_used_week4 : ℕ := 50 -- in 0.1 GB

-- Additional fee calculation
def extra_data_fee := 
  let extra_data_week1 := max (data_used_week1 - data_limit_week1 * 10) 0
  let extra_fee_week1 := extra_data_week1 * additional_fee_week1 / 10
  let extra_data_week2 := max (data_used_week2 - data_limit_week2 * 10) 0
  let extra_fee_week2 := extra_data_week2 * additional_fee_week2 / 10
  let extra_data_week3 := max (data_used_week3 - data_limit_week3 * 10) 0
  let extra_fee_week3 := extra_data_week3 * additional_fee_week3 / 10
  let extra_data_week4 := max (data_used_week4 - data_limit_week4 * 10) 0
  let extra_fee_week4 := extra_data_week4 * additional_fee_week4 / 10
  extra_fee_week1 + extra_fee_week2 + extra_fee_week3 + extra_fee_week4

-- The math proof problem
theorem charlie_extra_fee : extra_data_fee = 48 := sorry

end charlie_extra_fee_l55_55732


namespace non_vegan_gluten_cupcakes_eq_28_l55_55311

def total_cupcakes : ℕ := 80
def gluten_free_cupcakes : ℕ := total_cupcakes / 2
def vegan_cupcakes : ℕ := 24
def vegan_gluten_free_cupcakes : ℕ := vegan_cupcakes / 2
def non_vegan_cupcakes : ℕ := total_cupcakes - vegan_cupcakes
def gluten_cupcakes : ℕ := total_cupcakes - gluten_free_cupcakes
def non_vegan_gluten_cupcakes : ℕ := gluten_cupcakes - vegan_gluten_free_cupcakes

theorem non_vegan_gluten_cupcakes_eq_28 :
  non_vegan_gluten_cupcakes = 28 := by
  sorry

end non_vegan_gluten_cupcakes_eq_28_l55_55311


namespace mod_11_residue_l55_55250

theorem mod_11_residue : 
  ((312 - 3 * 52 + 9 * 165 + 6 * 22) % 11) = 2 :=
by
  sorry

end mod_11_residue_l55_55250


namespace independent_sum_of_projections_l55_55399

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem independent_sum_of_projections (A1 A2 A3 P P1 P2 P3 : ℝ × ℝ) 
  (h_eq_triangle : distance A1 A2 = distance A2 A3 ∧ distance A2 A3 = distance A3 A1)
  (h_proj_P1 : P1 = (P.1, A2.2))
  (h_proj_P2 : P2 = (P.1, A3.2))
  (h_proj_P3 : P3 = (P.1, A1.2)) :
  distance A1 P2 + distance A2 P3 + distance A3 P1 = (3 / 2) * distance A1 A2 := 
sorry

end independent_sum_of_projections_l55_55399


namespace intersect_sets_l55_55572

def M : Set ℝ := { x | x ≥ -1 }
def N : Set ℝ := { x | -2 < x ∧ x < 2 }

theorem intersect_sets :
  M ∩ N = { x | -1 ≤ x ∧ x < 2 } := by
  sorry

end intersect_sets_l55_55572


namespace intersection_of_A_and_B_l55_55136

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | x > 1}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end intersection_of_A_and_B_l55_55136


namespace value_of_2p_plus_q_l55_55609

theorem value_of_2p_plus_q (p q : ℚ) (h : p / q = 2 / 7) : 2 * p + q = (11 / 2) * p :=
by
  sorry

end value_of_2p_plus_q_l55_55609


namespace zoo_total_animals_l55_55904

theorem zoo_total_animals (penguins polar_bears : ℕ)
  (h1 : penguins = 21)
  (h2 : polar_bears = 2 * penguins) :
  penguins + polar_bears = 63 := by
   sorry

end zoo_total_animals_l55_55904


namespace Robert_more_than_Claire_l55_55015

variable (Lisa Claire Robert : ℕ)

theorem Robert_more_than_Claire (h1 : Lisa = 3 * Claire) (h2 : Claire = 10) (h3 : Robert > Claire) :
  Robert > 10 :=
by
  rw [h2] at h3
  assumption

end Robert_more_than_Claire_l55_55015


namespace initial_discount_percentage_l55_55291

-- Statement of the problem
theorem initial_discount_percentage (d : ℝ) (x : ℝ)
  (h₁ : d > 0)
  (h_staff_price : d * ((100 - x) / 100) * 0.5 = 0.225 * d) :
  x = 55 := 
sorry

end initial_discount_percentage_l55_55291


namespace fraction_a_b_l55_55817

variables {a b x y : ℝ}

theorem fraction_a_b (h1 : 4 * x - 2 * y = a) (h2 : 6 * y - 12 * x = b) (hb : b ≠ 0) :
  a / b = -1/3 := 
sorry

end fraction_a_b_l55_55817


namespace range_of_func_l55_55704

noncomputable def func (x : ℝ) : ℝ := 1 / (x - 1)

theorem range_of_func :
  (∀ y : ℝ, 
    (∃ x : ℝ, (x < 1 ∨ (2 ≤ x ∧ x < 5)) ∧ y = func x) ↔ 
    (y < 0 ∨ (1/4 < y ∧ y ≤ 1))) :=
by
  sorry

end range_of_func_l55_55704


namespace alice_students_count_l55_55652

variable (S : ℕ)
variable (students_with_own_vests := 0.20 * S)
variable (students_needing_vests := 0.80 * S)
variable (instructors : ℕ := 10)
variable (life_vests_on_hand : ℕ := 20)
variable (additional_life_vests_needed : ℕ := 22)
variable (total_life_vests_needed := life_vests_on_hand + additional_life_vests_needed)
variable (life_vests_needed_for_instructors := instructors)
variable (life_vests_needed_for_students := total_life_vests_needed - life_vests_needed_for_instructors)

theorem alice_students_count : S = 40 :=
by
  -- proof steps would go here
  sorry

end alice_students_count_l55_55652


namespace grocer_sales_l55_55665

theorem grocer_sales (sale1 sale2 sale3 sale4 sale5 sale6 : ℕ)
  (h1 : sale2 = 900)
  (h2 : sale3 = 1000)
  (h3 : sale4 = 700)
  (h4 : sale5 = 800)
  (h5 : sale6 = 900)
  (h6 : (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = 850) :
  sale1 = 800 :=
by
  sorry

end grocer_sales_l55_55665


namespace initial_walking_speed_l55_55339

open Real

theorem initial_walking_speed :
  ∃ (v : ℝ), (∀ (d : ℝ), d = 9.999999999999998 →
  (∀ (lateness_time : ℝ), lateness_time = 10 / 60 →
  ((d / v) - (d / 15) = lateness_time + lateness_time)) → v = 11.25) :=
by
  sorry

end initial_walking_speed_l55_55339


namespace set_equality_l55_55265

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hA : A = {1, 4})
variable (hB : B = {3, 4})

theorem set_equality : ({2, 5} : Set ℕ) = U \ (A ∪ B) :=
by
  sorry

end set_equality_l55_55265


namespace problem_solution_l55_55799

theorem problem_solution (x : ℝ) (h : x + 1 / x = 8) : x^2 + 1 / x^2 = 62 := 
by
  sorry

end problem_solution_l55_55799


namespace reduce_expression_l55_55480

-- Define the variables a, b, c as real numbers
variables (a b c : ℝ)

-- State the theorem with the given condition that expressions are defined and non-zero
theorem reduce_expression :
  (a^2 + b^2 - c^2 - 2*a*b) / (a^2 + c^2 - b^2 - 2*a*c) = ((a - b + c) * (a - b - c)) / ((a - c + b) * (a - c - b)) :=
by
  sorry

end reduce_expression_l55_55480


namespace sophie_total_spending_l55_55965

-- Definitions based on conditions
def num_cupcakes : ℕ := 5
def price_per_cupcake : ℝ := 2
def num_doughnuts : ℕ := 6
def price_per_doughnut : ℝ := 1
def num_slices_apple_pie : ℕ := 4
def price_per_slice_apple_pie : ℝ := 2
def num_cookies : ℕ := 15
def price_per_cookie : ℝ := 0.60

-- Total cost calculation
def total_cost : ℝ :=
  num_cupcakes * price_per_cupcake +
  num_doughnuts * price_per_doughnut +
  num_slices_apple_pie * price_per_slice_apple_pie +
  num_cookies * price_per_cookie

-- Theorem stating the total cost is 33
theorem sophie_total_spending : total_cost = 33 := by
  sorry

end sophie_total_spending_l55_55965


namespace solve_for_r_l55_55944

noncomputable def k (r : ℝ) : ℝ := 5 / (2 ^ r)

theorem solve_for_r (r : ℝ) :
  (5 = k r * 2 ^ r) ∧ (45 = k r * 8 ^ r) → r = (Real.log 9 / Real.log 2) / 2 :=
by
  intro h
  sorry

end solve_for_r_l55_55944


namespace coordinates_of_A_l55_55893

-- Defining the point A
def point_A : ℤ × ℤ := (1, -4)

-- Statement that needs to be proved
theorem coordinates_of_A :
  point_A = (1, -4) :=
by
  sorry

end coordinates_of_A_l55_55893


namespace number_less_than_one_is_correct_l55_55975

theorem number_less_than_one_is_correct : (1 - 5 = -4) :=
by
  sorry

end number_less_than_one_is_correct_l55_55975


namespace negation_of_prop_l55_55295

def prop (x : ℝ) := x^2 ≥ 0

theorem negation_of_prop:
  ¬ ∀ x : ℝ, prop x ↔ ∃ x : ℝ, x^2 < 0 := by
    sorry

end negation_of_prop_l55_55295


namespace sticker_distribution_ways_l55_55334

theorem sticker_distribution_ways : 
  ∃ ways : ℕ, ways = Nat.choose (9) (4) ∧ ways = 126 :=
by
  sorry

end sticker_distribution_ways_l55_55334


namespace length_of_each_piece_l55_55397

theorem length_of_each_piece :
  ∀ (ribbon_length remaining_length pieces : ℕ),
  ribbon_length = 51 →
  remaining_length = 36 →
  pieces = 100 →
  (ribbon_length - remaining_length) / pieces * 100 = 15 :=
by
  intros ribbon_length remaining_length pieces h1 h2 h3
  sorry

end length_of_each_piece_l55_55397


namespace square_perimeter_ratio_l55_55239

theorem square_perimeter_ratio (a₁ a₂ s₁ s₂ : ℝ) 
  (h₁ : a₁ / a₂ = 16 / 25)
  (h₂ : a₁ = s₁^2)
  (h₃ : a₂ = s₂^2) :
  (4 : ℝ) / 5 = s₁ / s₂ :=
by sorry

end square_perimeter_ratio_l55_55239


namespace pradeep_failed_marks_l55_55245

theorem pradeep_failed_marks
    (total_marks : ℕ)
    (obtained_marks : ℕ)
    (pass_percentage : ℕ)
    (pass_marks : ℕ)
    (fail_marks : ℕ)
    (total_marks_eq : total_marks = 2075)
    (obtained_marks_eq : obtained_marks = 390)
    (pass_percentage_eq : pass_percentage = 20)
    (pass_marks_eq : pass_marks = (pass_percentage * total_marks) / 100)
    (fail_marks_eq : fail_marks = pass_marks - obtained_marks) :
    fail_marks = 25 :=
by
  rw [total_marks_eq, obtained_marks_eq, pass_percentage_eq] at *
  sorry

end pradeep_failed_marks_l55_55245


namespace candy_proof_l55_55146

variable (x s t : ℤ)

theorem candy_proof (H1 : 4 * x - 15 * s = 23)
                    (H2 : 5 * x - 23 * t = 15) :
  x = 302 := by
  sorry

end candy_proof_l55_55146


namespace equal_intercepts_no_second_quadrant_l55_55688

/- Given line equation (a + 1)x + y + 2 - a = 0 and a \in ℝ. -/
def line_eq (a x y : ℝ) : Prop := (a + 1) * x + y + 2 - a = 0

/- If the line l has equal intercepts on both coordinate axes, 
   then a = 0 or a = 2. -/
theorem equal_intercepts (a : ℝ) :
  (∃ x y : ℝ, line_eq a x 0 ∧ line_eq a 0 y ∧ x = y) →
  a = 0 ∨ a = 2 :=
sorry

/- If the line l does not pass through the second quadrant,
   then a ≤ -1. -/
theorem no_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, x > 0 → y > 0 → ¬ line_eq a x y) →
  a ≤ -1 :=
sorry

end equal_intercepts_no_second_quadrant_l55_55688


namespace son_age_is_26_l55_55615

-- Definitions based on conditions in the problem
variables (S F : ℕ)
axiom cond1 : F = S + 28
axiom cond2 : F + 2 = 2 * (S + 2)

-- Statement to prove that S = 26
theorem son_age_is_26 : S = 26 :=
by 
  -- Proof steps go here
  sorry

end son_age_is_26_l55_55615


namespace kem_hourly_wage_l55_55327

theorem kem_hourly_wage (shem_total_earnings: ℝ) (shem_hours_worked: ℝ) (ratio: ℝ)
  (h1: shem_total_earnings = 80)
  (h2: shem_hours_worked = 8)
  (h3: ratio = 2.5) :
  (shem_total_earnings / shem_hours_worked) / ratio = 4 :=
by 
  sorry

end kem_hourly_wage_l55_55327


namespace part_I_part_II_l55_55271

def f (x a : ℝ) : ℝ := abs (x - a) + abs (2 * x + 1)

-- Part (I)
theorem part_I (x : ℝ) : f x 1 ≤ 3 ↔ -1 ≤ x ∧ x ≤ 1 :=
by sorry

-- Part (II)
theorem part_II (a : ℝ) : (∃ x ∈ Set.Ici a, f x a ≤ 2 * a + x) ↔ a ≥ 1 :=
by sorry

end part_I_part_II_l55_55271


namespace chad_sandwiches_l55_55546

-- Definitions representing the conditions
def crackers_per_sleeve : ℕ := 28
def sleeves_per_box : ℕ := 4
def boxes : ℕ := 5
def nights : ℕ := 56
def crackers_per_sandwich : ℕ := 2

-- Definition representing the final question about the number of sandwiches
def sandwiches_per_night (crackers_per_sleeve sleeves_per_box boxes nights crackers_per_sandwich : ℕ) : ℕ :=
  (crackers_per_sleeve * sleeves_per_box * boxes) / nights / crackers_per_sandwich

-- The theorem that states Chad makes 5 sandwiches each night
theorem chad_sandwiches :
  sandwiches_per_night crackers_per_sleeve sleeves_per_box boxes nights crackers_per_sandwich = 5 :=
by
  -- Proof outline:
  -- crackers_per_sleeve * sleeves_per_box * boxes = 28 * 4 * 5 = 560
  -- 560 / nights = 560 / 56 = 10 crackers per night
  -- 10 / crackers_per_sandwich = 10 / 2 = 5 sandwiches per night
  sorry

end chad_sandwiches_l55_55546


namespace probability_of_at_least_one_black_ball_l55_55579

noncomputable def probability_at_least_one_black_ball := 
  let total_outcomes := Nat.choose 4 2
  let favorable_outcomes := (Nat.choose 2 1) * (Nat.choose 2 1) + (Nat.choose 2 2)
  favorable_outcomes / total_outcomes

theorem probability_of_at_least_one_black_ball :
  probability_at_least_one_black_ball = 5 / 6 :=
by
  sorry

end probability_of_at_least_one_black_ball_l55_55579


namespace g_at_0_eq_1_l55_55519

noncomputable def g : ℝ → ℝ := sorry

axiom g_add (x y : ℝ) : g (x + y) = g x * g y
axiom g_deriv_at_0 : deriv g 0 = 2

theorem g_at_0_eq_1 : g 0 = 1 :=
by
  sorry

end g_at_0_eq_1_l55_55519


namespace upper_limit_of_x_l55_55258

theorem upper_limit_of_x :
  ∀ x : ℤ, (0 < x ∧ x < 7) ∧ (0 < x ∧ x < some_upper_limit) ∧ (5 > x ∧ x > -1) ∧ (3 > x ∧ x > 0) ∧ (x + 2 < 4) →
  some_upper_limit = 2 :=
by
  intros x h
  sorry

end upper_limit_of_x_l55_55258


namespace minimum_value_condition_l55_55504

def f (a x : ℝ) : ℝ := -x^3 + 0.5 * (a + 3) * x^2 - a * x - 1

theorem minimum_value_condition (a : ℝ) (h : a ≥ 3) : 
  (∃ x₀ : ℝ, f a x₀ < f a 1) ∨ (f a 1 > f a ((a/3))) := 
sorry

end minimum_value_condition_l55_55504


namespace problem1_problem2_l55_55948

-- Problem 1
theorem problem1 (a : ℝ) : 2 * a + 3 * a - 4 * a = a :=
by sorry

-- Problem 2
theorem problem2 : 
  - (1 : ℝ) ^ 2022 + (27 / 4) * (- (1 / 3) - 1) / ((-3) ^ 2) + abs (-1) = -1 :=
by sorry

end problem1_problem2_l55_55948


namespace total_questions_reviewed_l55_55635

-- Defining the conditions
def questions_per_student : Nat := 10
def students_per_class : Nat := 35
def num_classes : Nat := 5

-- Define the total number of questions that Professor Oscar must review.
def total_questions : Nat := questions_per_student * students_per_class * num_classes

-- Theorem statement to be proved
theorem total_questions_reviewed : total_questions = 1750 := by
  sorry

end total_questions_reviewed_l55_55635


namespace female_democrats_l55_55575

theorem female_democrats (F M : ℕ) (h1 : F + M = 840) (h2 : F / 2 + M / 4 = 280) : F / 2 = 140 :=
by 
  sorry

end female_democrats_l55_55575


namespace find_f_l55_55396

def f : ℝ → ℝ := sorry

theorem find_f (x : ℝ) : f (x + 2) = 2 * x + 3 → f x = 2 * x - 1 :=
by
  intro h
  -- Proof goes here 
  sorry

end find_f_l55_55396


namespace max_axbycz_value_l55_55687

theorem max_axbycz_value (a b c : ℝ) (x y z : ℝ) 
  (h_triangle: a + b > c ∧ b + c > a ∧ c + a > b)
  (h_positive: 0 < x ∧ 0 < y ∧ 0 < z)
  (h_sum : x + y + z = 1) : 
  a * x * y + b * y * z + c * z * x ≤ (a * b * c) / (2 * a * b + 2 * b * c + 2 * c * a - a^2 - b^2 - c^2) :=
  sorry

end max_axbycz_value_l55_55687


namespace maria_score_l55_55685

theorem maria_score (m j : ℕ) (h1 : m = j + 50) (h2 : (m + j) / 2 = 112) : m = 137 :=
by
  sorry

end maria_score_l55_55685


namespace solution_set_of_x_l55_55655

noncomputable def satisfies_condition (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋

theorem solution_set_of_x :
  { x : ℝ | satisfies_condition x } = { x : ℝ | 5/3 ≤ x ∧ x < 7/3 } :=
by
  sorry

end solution_set_of_x_l55_55655


namespace eggs_per_omelet_l55_55116

theorem eggs_per_omelet:
  let small_children_tickets := 53
  let older_children_tickets := 35
  let adult_tickets := 75
  let senior_tickets := 37
  let smallChildrenOmelets := small_children_tickets * 0.5
  let olderChildrenOmelets := older_children_tickets
  let adultOmelets := adult_tickets * 2
  let seniorOmelets := senior_tickets * 1.5
  let extra_omelets := 25
  let total_omelets := smallChildrenOmelets + olderChildrenOmelets + adultOmelets + seniorOmelets + extra_omelets
  let total_eggs := 584
  total_eggs / total_omelets = 2 := 
by
  sorry

end eggs_per_omelet_l55_55116


namespace reciprocal_self_eq_one_or_neg_one_l55_55924

theorem reciprocal_self_eq_one_or_neg_one (x : ℝ) (h : x = 1 / x) : x = 1 ∨ x = -1 := sorry

end reciprocal_self_eq_one_or_neg_one_l55_55924


namespace edge_length_box_l55_55699

theorem edge_length_box (n : ℝ) (h : n = 999.9999999999998) : 
  ∃ (L : ℝ), L = 1 ∧ ((L * 100) ^ 3 / 10 ^ 3) = n := 
sorry

end edge_length_box_l55_55699


namespace sum_of_ages_l55_55950

-- Define the variables for Viggo and his younger brother's ages
variables (v y : ℕ)

-- Condition: When Viggo's younger brother was 2, Viggo's age was 10 years more than twice his brother's age
def condition1 (v y : ℕ) := (y = 2 → v = 2 * y + 10)

-- Condition: Viggo's younger brother is currently 10 years old
def condition2 (y_current : ℕ) := y_current = 10

-- Define the current age of Viggo given the conditions
def viggo_current_age (v y y_current : ℕ) := v + (y_current - y)

-- Prove that the sum of their ages is 32
theorem sum_of_ages
  (v y y_current : ℕ)
  (h1 : condition1 v y)
  (h2 : condition2 y_current) :
  viggo_current_age v y y_current + y_current = 32 :=
by
  -- Apply sorry to skip the proof
  sorry

end sum_of_ages_l55_55950


namespace Jon_needs_to_wash_20_pairs_of_pants_l55_55202

theorem Jon_needs_to_wash_20_pairs_of_pants
  (machine_capacity : ℕ)
  (shirts_per_pound : ℕ)
  (pants_per_pound : ℕ)
  (num_shirts : ℕ)
  (num_loads : ℕ)
  (total_pounds : ℕ)
  (weight_of_shirts : ℕ)
  (remaining_weight : ℕ)
  (num_pairs_of_pants : ℕ) :
  machine_capacity = 5 →
  shirts_per_pound = 4 →
  pants_per_pound = 2 →
  num_shirts = 20 →
  num_loads = 3 →
  total_pounds = num_loads * machine_capacity →
  weight_of_shirts = num_shirts / shirts_per_pound →
  remaining_weight = total_pounds - weight_of_shirts →
  num_pairs_of_pants = remaining_weight * pants_per_pound →
  num_pairs_of_pants = 20 :=
by
  intros _ _ _ _ _ _ _ _ _
  sorry

end Jon_needs_to_wash_20_pairs_of_pants_l55_55202


namespace TrainTravelDays_l55_55086

-- Definition of the problem conditions
def train_start (days: ℕ) : ℕ := 
  if days = 0 then 0 -- no trains to meet on the first day
  else days -- otherwise, meet 'days' number of trains

/-- 
  Prove that if a train comes across 4 trains on its way from Amritsar to Bombay and starts at 9 am, 
  then it takes 5 days for the train to reach its destination.
-/
theorem TrainTravelDays (meet_train_count : ℕ) : meet_train_count = 4 → train_start (meet_train_count) + 1 = 5 :=
by
  intro h
  rw [h]
  sorry

end TrainTravelDays_l55_55086


namespace range_of_a_l55_55643

theorem range_of_a (a : ℝ) (h1 : 0 < a) :
  (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 ≤ 0 → x^2 - x - 6 ≤ 0) ∧
  (¬ (∀ x : ℝ, x^2 - x - 6 ≤ 0 → x^2 - 4*a*x + 3*a^2 ≤ 0)) →
  0 < a ∧ a ≤ 1 :=
sorry

end range_of_a_l55_55643


namespace find_expression_l55_55120

variables {x y : ℝ}

theorem find_expression
  (h1: 3 * x + y = 5)
  (h2: x + 3 * y = 6)
  : 10 * x^2 + 13 * x * y + 10 * y^2 = 97 :=
by
  sorry

end find_expression_l55_55120


namespace horner_evaluation_at_3_l55_55276

def f (x : ℤ) : ℤ := x^5 + 2 * x^3 + 3 * x^2 + x + 1

theorem horner_evaluation_at_3 : f 3 = 328 := by
  sorry

end horner_evaluation_at_3_l55_55276


namespace man_born_in_1892_l55_55150

-- Define the conditions and question
def man_birth_year (x : ℕ) : ℕ :=
x^2 - x

-- Conditions:
variable (x : ℕ)
-- 1. The man was born in the first half of the 20th century
variable (h1 : man_birth_year x < 1950)
-- 2. The man's age x and the conditions in the problem
variable (h2 : x^2 - x < 1950)

-- The statement we aim to prove
theorem man_born_in_1892 (x : ℕ) (h1 : man_birth_year x < 1950) (h2 : x = 44) : man_birth_year x = 1892 := by
  sorry

end man_born_in_1892_l55_55150


namespace find_XY_squared_l55_55084

variables {A B C T X Y : Type}

-- Conditions
variables (is_acute_scalene_triangle : ∀ A B C : Type, Prop) -- Assume scalene and acute properties
variable  (circumcircle : ∀ A B C : Type, Type) -- Circumcircle of the triangle
variable  (tangent_at : ∀ (ω : Type) B C, Type) -- Tangents at B and C
variables (BT CT : ℝ)
variables (BC : ℝ)
variables (projections : ∀ T (line : Type), Type)
variables (TX TY XY : ℝ)

-- Given conditions
axiom BT_value : BT = 18
axiom CT_value : CT = 18
axiom BC_value : BC = 24
axiom final_equation : TX^2 + TY^2 + XY^2 = 1552

-- Goal
theorem find_XY_squared : XY^2 = 884 := by
  sorry

end find_XY_squared_l55_55084


namespace sequence_increasing_l55_55253

theorem sequence_increasing (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ∀ n : ℕ, a^n / n^b < a^(n+1) / (n+1)^b :=
by sorry

end sequence_increasing_l55_55253


namespace sum_first_five_terms_arithmetic_sequence_l55_55538

theorem sum_first_five_terms_arithmetic_sequence (a d : ℤ)
  (h1 : a + 5 * d = 10)
  (h2 : a + 6 * d = 15)
  (h3 : a + 7 * d = 20) :
  5 * (2 * a + (5 - 1) * d) / 2 = -25 := by
  sorry

end sum_first_five_terms_arithmetic_sequence_l55_55538


namespace solve_fraction_eq_l55_55318

theorem solve_fraction_eq (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end solve_fraction_eq_l55_55318


namespace domain_of_g_l55_55646

noncomputable def g (x : ℝ) : ℝ := 1 / ⌊ x^2 - 9 * x + 21 ⌋

theorem domain_of_g :
  { x : ℝ | ∃ y : ℝ, g x = y } = { x : ℝ | x ≤ 4 ∨ x ≥ 5 } :=
by
  sorry

end domain_of_g_l55_55646


namespace number_of_valid_3_digit_numbers_l55_55829

def is_even (n : ℕ) : Prop := n % 2 = 0

def valid_3_digit_numbers_count : ℕ :=
  let digits := [(4, 8), (8, 4), (6, 6)]
  digits.length * 9

theorem number_of_valid_3_digit_numbers : valid_3_digit_numbers_count = 27 :=
by
  sorry

end number_of_valid_3_digit_numbers_l55_55829


namespace eventually_periodic_sequence_l55_55106

noncomputable def eventually_periodic (a : ℕ → ℕ) : Prop :=
  ∃ N k : ℕ, k > 0 ∧ ∀ m ≥ N, a m = a (m + k)

theorem eventually_periodic_sequence
  (a : ℕ → ℕ)
  (h_pos : ∀ n, a n > 0)
  (h_condition : ∀ n, a n * a (n + 1) = a (n + 2) * a (n + 3)) :
  eventually_periodic a :=
sorry

end eventually_periodic_sequence_l55_55106


namespace latest_start_time_l55_55840

-- Define the weights of the turkeys
def turkey_weights : List ℕ := [16, 18, 20, 22]

-- Define the roasting time per pound
def roasting_time_per_pound : ℕ := 15

-- Define the dinner time in 24-hour format
def dinner_time : ℕ := 18 * 60 -- 18:00 in minutes

-- Calculate the total roasting time
def total_roasting_time (weights : List ℕ) (time_per_pound : ℕ) : ℕ :=
  weights.foldr (λ weight acc => weight * time_per_pound + acc) 0

-- Calculate the latest start time
def latest_roasting_start_time (total_time : ℕ) (dinner_time : ℕ) : ℕ :=
  let start_time := dinner_time - total_time
  if start_time < 0 then start_time + 24 * 60 else start_time

-- Convert minutes to hours:minutes format
def time_in_hours_minutes (time : ℕ) : String :=
  let hours := time / 60
  let minutes := time % 60
  toString hours ++ ":" ++ toString minutes

theorem latest_start_time : 
  time_in_hours_minutes (latest_roasting_start_time (total_roasting_time turkey_weights roasting_time_per_pound) dinner_time) = "23:00" := by
  sorry

end latest_start_time_l55_55840


namespace problem_is_happy_number_512_l55_55290

/-- A number is a "happy number" if it is the square difference of two consecutive odd numbers. -/
def is_happy_number (x : ℕ) : Prop :=
  ∃ n : ℤ, x = 8 * n

/-- The number 512 is a "happy number". -/
theorem problem_is_happy_number_512 : is_happy_number 512 :=
  sorry

end problem_is_happy_number_512_l55_55290


namespace other_root_of_quadratic_l55_55937

theorem other_root_of_quadratic (a b k : ℝ) (h : 1^2 - (a+b) * 1 + ab * (1 - k) = 0) : 
  ∃ r : ℝ, r = a + b - 1 := 
sorry

end other_root_of_quadratic_l55_55937


namespace positive_integers_divide_n_plus_7_l55_55161

theorem positive_integers_divide_n_plus_7 (n : ℕ) (hn_pos : 0 < n) : n ∣ n + 7 ↔ n = 1 ∨ n = 7 :=
by 
  sorry

end positive_integers_divide_n_plus_7_l55_55161


namespace minimum_red_vertices_l55_55550

theorem minimum_red_vertices (n : ℕ) (h : 0 < n) :
  ∃ R : ℕ, (∀ i j : ℕ, i < n ∧ j < n →
    (i + j) % 2 = 0 → true) ∧
    R = Int.ceil (n^2 / 2 : ℝ) :=
sorry

end minimum_red_vertices_l55_55550


namespace altitude_product_difference_eq_zero_l55_55506

variables (A B C P Q H : Type*) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited Q] [Inhabited H]
variable {HP HQ BP PC AQ QC AH BH : ℝ}

-- Given conditions
axiom altitude_intersects_at_H : true
axiom HP_val : HP = 3
axiom HQ_val : HQ = 7

-- Statement to prove
theorem altitude_product_difference_eq_zero (h_BP_PC : BP * PC = 3 / (AH + 3))
                                           (h_AQ_QC : AQ * QC = 7 / (BH + 7))
                                           (h_AH_BQ_ratio : AH / BH = 3 / 7) :
  (BP * PC) - (AQ * QC) = 0 :=
by sorry

end altitude_product_difference_eq_zero_l55_55506


namespace winner_exceeds_second_opponent_l55_55503

theorem winner_exceeds_second_opponent
  (total_votes : ℕ)
  (votes_winner : ℕ)
  (votes_second : ℕ)
  (votes_third : ℕ)
  (votes_fourth : ℕ) 
  (h_votes_sum : total_votes = votes_winner + votes_second + votes_third + votes_fourth)
  (h_total_votes : total_votes = 963) 
  (h_winner_votes : votes_winner = 195) 
  (h_second_votes : votes_second = 142) 
  (h_third_votes : votes_third = 116) 
  (h_fourth_votes : votes_fourth = 90) :
  votes_winner - votes_second = 53 := by
  sorry

end winner_exceeds_second_opponent_l55_55503


namespace range_of_a_l55_55624

theorem range_of_a (m : ℝ) (a : ℝ) : 
  m ∈ Set.Icc (-1 : ℝ) (1 : ℝ) →
  (∀ x₁ x₂ : ℝ, x₁^2 - m * x₁ - 2 = 0 ∧ x₂^2 - m * x₂ - 2 = 0 → a^2 - 5 * a - 3 ≥ |x₁ - x₂|) ↔ (a ≥ 6 ∨ a ≤ -1) :=
by
  sorry

end range_of_a_l55_55624


namespace percentage_sales_other_l55_55121

theorem percentage_sales_other (p_pens p_pencils p_markers p_other : ℕ)
(h_pens : p_pens = 25)
(h_pencils : p_pencils = 30)
(h_markers : p_markers = 20)
(h_other : p_other = 100 - (p_pens + p_pencils + p_markers)): p_other = 25 :=
by
  rw [h_pens, h_pencils, h_markers] at h_other
  exact h_other


end percentage_sales_other_l55_55121


namespace minimum_perimeter_triangle_MAF_is_11_l55_55350

-- Define point, parabola, and focus
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the specific points in the problem
def A : Point := ⟨5, 3⟩

-- Parabola with the form y^2 = 4x has the focus at (1, 0)
def F : Point := ⟨1, 0⟩

-- Minimum perimeter problem for ΔMAF
noncomputable def minimum_perimeter_triangle_MAF (M : Point) : ℝ :=
  (dist (M.x, M.y) (A.x, A.y)) + (dist (M.x, M.y) (F.x, F.y))

-- The goal is to show the minimum value of the perimeter is 11
theorem minimum_perimeter_triangle_MAF_is_11 (M : Point) 
  (hM_parabola : M.y^2 = 4 * M.x) 
  (hM_not_AF : M.x ≠ (5 + (3 * ((M.y - 0) / (M.x - 1))) )) : 
  ∃ M, minimum_perimeter_triangle_MAF M = 11 :=
sorry

end minimum_perimeter_triangle_MAF_is_11_l55_55350


namespace chalk_pieces_l55_55532

theorem chalk_pieces (boxes: ℕ) (pieces_per_box: ℕ) (total_chalk: ℕ) 
  (hb: boxes = 194) (hp: pieces_per_box = 18) : 
  total_chalk = 194 * 18 :=
by 
  sorry

end chalk_pieces_l55_55532


namespace total_time_spent_l55_55856

-- Definitions based on the conditions
def number_of_chairs := 2
def number_of_tables := 2
def minutes_per_piece := 8
def total_pieces := number_of_chairs + number_of_tables

-- The statement we want to prove
theorem total_time_spent : total_pieces * minutes_per_piece = 32 :=
by
  sorry

end total_time_spent_l55_55856


namespace train_passes_man_in_approximately_18_seconds_l55_55279

noncomputable def length_of_train : ℝ := 330 -- meters
noncomputable def speed_of_train : ℝ := 60 -- kmph
noncomputable def speed_of_man : ℝ := 6 -- kmph

noncomputable def kmph_to_mps (v : ℝ) : ℝ := v * (5/18)

noncomputable def relative_speed_mps : ℝ :=
  kmph_to_mps (speed_of_train + speed_of_man)

noncomputable def time_to_pass (length : ℝ) (speed : ℝ) : ℝ := length / speed

theorem train_passes_man_in_approximately_18_seconds :
  abs (time_to_pass length_of_train relative_speed_mps - 18) < 1 :=
by
  sorry

end train_passes_man_in_approximately_18_seconds_l55_55279


namespace linear_regression_passes_through_centroid_l55_55673

noncomputable def linear_regression (a b : ℝ) (x : ℝ) : ℝ := a + b * x

theorem linear_regression_passes_through_centroid 
  (a b : ℝ) (x_bar y_bar : ℝ) 
  (h_centroid : ∀ (x y : ℝ), (x = x_bar ∧ y = y_bar) → y = linear_regression a b x) :
  linear_regression a b x_bar = y_bar :=
by
  -- proof omitted
  sorry

end linear_regression_passes_through_centroid_l55_55673


namespace investment_initial_amount_l55_55633

noncomputable def initialInvestment (final_amount : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  final_amount / interest_rate^years

theorem investment_initial_amount :
  initialInvestment 705.73 1.12 5 = 400.52 := by
  sorry

end investment_initial_amount_l55_55633


namespace pool_volume_l55_55825

variable {rate1 rate2 : ℕ}
variables {hose1 hose2 hose3 hose4 : ℕ}
variables {time : ℕ}

def hose1_rate := 2
def hose2_rate := 2
def hose3_rate := 3
def hose4_rate := 3
def fill_time := 25

def total_rate := hose1_rate + hose2_rate + hose3_rate + hose4_rate

theorem pool_volume (h : hose1 = hose1_rate ∧ hose2 = hose2_rate ∧ hose3 = hose3_rate ∧ hose4 = hose4_rate ∧ time = fill_time):
  total_rate * 60 * time = 15000 := 
by 
  sorry

end pool_volume_l55_55825


namespace fraction_is_one_over_three_l55_55165

variable (x : ℚ) -- Let the fraction x be a rational number
variable (num : ℚ) -- Let the number be a rational number

theorem fraction_is_one_over_three (h1 : num = 45) (h2 : x * num - 5 = 10) : x = 1 / 3 := by
  sorry

end fraction_is_one_over_three_l55_55165


namespace toys_per_week_production_l55_55838

-- Define the necessary conditions
def days_per_week : Nat := 4
def toys_per_day : Nat := 1500

-- Define the theorem to prove the total number of toys produced per week
theorem toys_per_week_production : 
  ∀ (days_per_week toys_per_day : Nat), 
    (days_per_week = 4) →
    (toys_per_day = 1500) →
    (days_per_week * toys_per_day = 6000) := 
by
  intros
  sorry

end toys_per_week_production_l55_55838


namespace parking_lot_wheels_l55_55151

-- Define the conditions
def num_cars : Nat := 10
def num_bikes : Nat := 2
def wheels_per_car : Nat := 4
def wheels_per_bike : Nat := 2

-- Define the total number of wheels
def total_wheels : Nat := (num_cars * wheels_per_car) + (num_bikes * wheels_per_bike)

-- State the theorem
theorem parking_lot_wheels : total_wheels = 44 :=
by
  sorry

end parking_lot_wheels_l55_55151


namespace time_in_2700_minutes_is_3_am_l55_55452

def minutes_in_hour : ℕ := 60
def hours_in_day : ℕ := 24
def current_hour : ℕ := 6
def minutes_later : ℕ := 2700

-- Calculate the final hour after adding the given minutes
def final_hour (current_hour minutes_later minutes_in_hour hours_in_day: ℕ) : ℕ :=
  (current_hour + (minutes_later / minutes_in_hour) % hours_in_day) % hours_in_day

theorem time_in_2700_minutes_is_3_am :
  final_hour current_hour minutes_later minutes_in_hour hours_in_day = 3 :=
by
  sorry

end time_in_2700_minutes_is_3_am_l55_55452


namespace octopus_legs_l55_55254

-- Definitions of octopus behavior based on the number of legs
def tells_truth (legs: ℕ) : Prop := legs = 6 ∨ legs = 8
def lies (legs: ℕ) : Prop := legs = 7

-- Statements made by the octopuses
def blue_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 28
def green_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 27
def yellow_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 26
def red_statement (legs_b legs_g legs_y legs_r: ℕ) : Prop := legs_b + legs_g + legs_y + legs_r = 25

noncomputable def legs_b := 7
noncomputable def legs_g := 6
noncomputable def legs_y := 7
noncomputable def legs_r := 7

-- Main theorem
theorem octopus_legs : 
  (tells_truth legs_g) ∧ 
  (lies legs_b) ∧ 
  (lies legs_y) ∧ 
  (lies legs_r) ∧ 
  blue_statement legs_b legs_g legs_y legs_r ∧ 
  green_statement legs_b legs_g legs_y legs_r ∧ 
  yellow_statement legs_b legs_g legs_y legs_r ∧ 
  red_statement legs_b legs_g legs_y legs_r := 
by 
  sorry

end octopus_legs_l55_55254


namespace line_circle_no_intersection_l55_55782

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) → false :=
sorry

end line_circle_no_intersection_l55_55782


namespace Euclid1976_PartA_Problem8_l55_55526

theorem  Euclid1976_PartA_Problem8 (a b c m n : ℝ) 
  (h1 : Polynomial.eval a (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 3 * Polynomial.X^2 + Polynomial.C m * Polynomial.X + Polynomial.C 24) = 0)
  (h2 : Polynomial.eval b (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 3 * Polynomial.X^2 + Polynomial.C m * Polynomial.X + Polynomial.C 24) = 0)
  (h3 : Polynomial.eval c (Polynomial.C 1 * Polynomial.X^3 - Polynomial.C 3 * Polynomial.X^2 + Polynomial.C m * Polynomial.X + Polynomial.C 24) = 0)
  (h4 : Polynomial.eval (-a) (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C n * Polynomial.X + Polynomial.C (-6)) = 0)
  (h5 : Polynomial.eval (-b) (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C n * Polynomial.X + Polynomial.C (-6)) = 0) :
  n = -1 :=
sorry

end Euclid1976_PartA_Problem8_l55_55526


namespace fraction_of_number_is_one_fifth_l55_55125

theorem fraction_of_number_is_one_fifth (N : ℕ) (f : ℚ) 
    (hN : N = 90) 
    (h : 3 + (1 / 2) * (1 / 3) * f * N = (1 / 15) * N) : 
  f = 1 / 5 := by 
  sorry

end fraction_of_number_is_one_fifth_l55_55125


namespace petya_catch_bus_l55_55183

theorem petya_catch_bus 
    (v_p v_b d : ℝ) 
    (h1 : v_b = 5 * v_p)
    (h2 : ∀ t : ℝ, 5 * v_p * t ≤ 0.6) 
    : d = 0.12 := 
sorry

end petya_catch_bus_l55_55183


namespace product_of_reciprocals_l55_55876

theorem product_of_reciprocals (x y : ℝ) (h : x + y = 6 * x * y) : (1 / x) * (1 / y) = 1 / 36 :=
by
  sorry

end product_of_reciprocals_l55_55876


namespace probability_non_defective_second_draw_l55_55160

theorem probability_non_defective_second_draw 
  (total_products : ℕ)
  (defective_products : ℕ)
  (first_draw_defective : Bool)
  (second_draw_non_defective_probability : ℚ) : 
  total_products = 100 → 
  defective_products = 3 → 
  first_draw_defective = true → 
  second_draw_non_defective_probability = 97 / 99 :=
by
  intros h_total h_defective h_first_draw
  subst h_total
  subst h_defective
  subst h_first_draw
  sorry

end probability_non_defective_second_draw_l55_55160


namespace problem1_problem2_l55_55221

-- Problem 1: Proving the equation
theorem problem1 (x : ℝ) : (x + 2) / 3 - 1 = (1 - x) / 2 → x = 1 :=
sorry

-- Problem 2: Proving the solution for the system of equations
theorem problem2 (x y : ℝ) : (x + 2 * y = 8) ∧ (3 * x - 4 * y = 4) → x = 4 ∧ y = 2 :=
sorry

end problem1_problem2_l55_55221


namespace total_games_won_l55_55547

theorem total_games_won 
  (bulls_games : ℕ) (heat_games : ℕ) (knicks_games : ℕ)
  (bulls_condition : bulls_games = 70)
  (heat_condition : heat_games = bulls_games + 5)
  (knicks_condition : knicks_games = 2 * heat_games) :
  bulls_games + heat_games + knicks_games = 295 :=
by
  sorry

end total_games_won_l55_55547


namespace range_of_function_l55_55419

theorem range_of_function :
  (∀ y : ℝ, (∃ x : ℝ, y = (x + 1) / (x ^ 2 + 1)) ↔ 0 ≤ y ∧ y ≤ 4/3) :=
by
  sorry

end range_of_function_l55_55419


namespace sheila_hourly_wage_l55_55822

def sheila_works_hours : ℕ :=
  let monday_wednesday_friday := 8 * 3
  let tuesday_thursday := 6 * 2
  monday_wednesday_friday + tuesday_thursday

def sheila_weekly_earnings : ℕ := 396
def sheila_total_hours_worked := 36
def expected_hourly_earnings := sheila_weekly_earnings / sheila_total_hours_worked

theorem sheila_hourly_wage :
  sheila_works_hours = sheila_total_hours_worked ∧
  sheila_weekly_earnings / sheila_total_hours_worked = 11 :=
by
  sorry

end sheila_hourly_wage_l55_55822


namespace minimum_perimeter_triangle_l55_55167

noncomputable def minimum_perimeter (a b c : ℝ) (cos_C : ℝ) (ha : a + b = 10) (hroot : 2 * cos_C^2 - 3 * cos_C - 2 = 0) 
  : ℝ :=
  a + b + c

theorem minimum_perimeter_triangle (a b c : ℝ) (cos_C : ℝ)
  (ha : a + b = 10)
  (hroot : 2 * cos_C^2 - 3 * cos_C - 2 = 0)
  (cos_C_valid : cos_C = -1/2) :
  (minimum_perimeter a b c cos_C ha hroot) = 10 + 5 * Real.sqrt 3 :=
sorry

end minimum_perimeter_triangle_l55_55167


namespace pokemon_cards_per_friend_l55_55560

theorem pokemon_cards_per_friend (total_cards : ℕ) (num_friends : ℕ) 
  (hc : total_cards = 56) (hf : num_friends = 4) : (total_cards / num_friends) = 14 := 
by
  sorry

end pokemon_cards_per_friend_l55_55560


namespace scientific_notation_correct_l55_55312

-- Define the problem conditions
def original_number : ℝ := 6175700

-- Define the expected output in scientific notation
def scientific_notation_representation (x : ℝ) : Prop :=
  x = 6.1757 * 10^6

-- The theorem to prove
theorem scientific_notation_correct : scientific_notation_representation original_number :=
by sorry

end scientific_notation_correct_l55_55312


namespace max_sector_area_l55_55385

theorem max_sector_area (r θ : ℝ) (S : ℝ) (h_perimeter : 2 * r + θ * r = 16)
  (h_max_area : S = 1 / 2 * θ * r^2) :
  r = 4 ∧ θ = 2 ∧ S = 16 := by
  -- sorry, the proof is expected to go here
  sorry

end max_sector_area_l55_55385


namespace muffins_per_person_l55_55529

-- Definitions based on conditions
def total_friends : ℕ := 4
def total_people : ℕ := 1 + total_friends
def total_muffins : ℕ := 20

-- Theorem statement for the proof
theorem muffins_per_person : total_muffins / total_people = 4 := by
  sorry

end muffins_per_person_l55_55529


namespace No_response_percentage_l55_55223

theorem No_response_percentage (total_guests : ℕ) (yes_percentage : ℕ) (non_respondents : ℕ) (yes_guests := total_guests * yes_percentage / 100) (no_guests := total_guests - yes_guests - non_respondents) (no_percentage := no_guests * 100 / total_guests) :
  total_guests = 200 → yes_percentage = 83 → non_respondents = 16 → no_percentage = 9 :=
by
  sorry

end No_response_percentage_l55_55223


namespace base_n_divisible_by_13_l55_55185

-- Define the polynomial f(n)
def f (n : ℕ) : ℕ := 7 + 3 * n + 5 * n^2 + 6 * n^3 + 3 * n^4 + 5 * n^5

-- The main theorem stating the result
theorem base_n_divisible_by_13 : 
  (∃ ns : Finset ℕ, ns.card = 16 ∧ ∀ n ∈ ns, 3 ≤ n ∧ n ≤ 200 ∧ f n % 13 = 0) :=
sorry

end base_n_divisible_by_13_l55_55185


namespace train_platform_time_l55_55455

theorem train_platform_time :
  ∀ (L_train L_platform T_tree S D T_platform : ℝ),
    L_train = 1200 ∧ 
    T_tree = 120 ∧ 
    L_platform = 1100 ∧ 
    S = L_train / T_tree ∧ 
    D = L_train + L_platform ∧ 
    T_platform = D / S →
    T_platform = 230 :=
by
  intros
  sorry

end train_platform_time_l55_55455


namespace B_spends_85_percent_l55_55649

def combined_salary (S_A S_B : ℝ) : Prop := S_A + S_B = 4000
def A_savings_percentage : ℝ := 0.05
def A_salary : ℝ := 3000
def B_salary : ℝ := 4000 - A_salary
def equal_savings (S_A S_B : ℝ) : Prop := A_savings_percentage * S_A = (1 - S_B / 100) * B_salary

theorem B_spends_85_percent (S_A S_B : ℝ) (B_spending_percentage : ℝ) :
  combined_salary S_A S_B ∧ S_A = A_salary ∧ equal_savings S_A B_spending_percentage → B_spending_percentage = 0.85 := by
  sorry

end B_spends_85_percent_l55_55649


namespace option_C_forms_a_set_l55_55077

-- Definition of the criteria for forming a set
def well_defined (criterion : Prop) : Prop := criterion

-- Criteria for option C: all female students in grade one of Jiu Middle School
def grade_one_students_criteria (is_female : Prop) (is_grade_one_student : Prop) : Prop :=
  is_female ∧ is_grade_one_student

-- Proof statement
theorem option_C_forms_a_set :
  ∀ (is_female : Prop) (is_grade_one_student : Prop), well_defined (grade_one_students_criteria is_female is_grade_one_student) :=
  by sorry

end option_C_forms_a_set_l55_55077


namespace inequality_always_holds_l55_55083

theorem inequality_always_holds (m : ℝ) : (-6 < m ∧ m ≤ 0) ↔ ∀ x : ℝ, 2 * m * x^2 + m * x - 3 / 4 < 0 := 
sorry

end inequality_always_holds_l55_55083


namespace least_possible_value_m_n_l55_55358

theorem least_possible_value_m_n :
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ Nat.gcd (m + n) 330 = 1 ∧ n ∣ m^m ∧ ¬(m % n = 0) ∧ (m + n = 377) :=
by
  sorry

end least_possible_value_m_n_l55_55358


namespace find_rstu_l55_55110

theorem find_rstu (a x y c : ℝ) (r s t u : ℤ) (hc : a^10 * x * y - a^8 * y - a^7 * x = a^6 * (c^3 - 1)) :
  (a^r * x - a^s) * (a^t * y - a^u) = a^6 * c^3 ∧ r * s * t * u = 0 :=
by
  sorry

end find_rstu_l55_55110


namespace necessary_and_sufficient_condition_l55_55405

theorem necessary_and_sufficient_condition (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : (a + b > a * b) ↔ (a = 1 ∨ b = 1) := 
sorry

end necessary_and_sufficient_condition_l55_55405


namespace find_larger_number_l55_55024

-- Define the conditions
variables (L S : ℕ)

theorem find_larger_number (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 :=
sorry

end find_larger_number_l55_55024


namespace problem_statement_l55_55989

variable (f : ℝ → ℝ)

theorem problem_statement (h : ∀ x : ℝ, 2 * (f x) + x * (deriv f x) > x^2) :
  ∀ x : ℝ, x^2 * f x ≥ 0 :=
by
  sorry

end problem_statement_l55_55989


namespace trucks_after_redistribution_l55_55494

/-- Problem Statement:
   Prove that the total number of trucks after redistribution is 10.
-/

theorem trucks_after_redistribution
    (num_trucks1 : ℕ)
    (boxes_per_truck1 : ℕ)
    (num_trucks2 : ℕ)
    (boxes_per_truck2 : ℕ)
    (containers_per_box : ℕ)
    (containers_per_truck_after : ℕ)
    (h1 : num_trucks1 = 7)
    (h2 : boxes_per_truck1 = 20)
    (h3 : num_trucks2 = 5)
    (h4 : boxes_per_truck2 = 12)
    (h5 : containers_per_box = 8)
    (h6 : containers_per_truck_after = 160) :
  (num_trucks1 * boxes_per_truck1 + num_trucks2 * boxes_per_truck2) * containers_per_box / containers_per_truck_after = 10 := by
  sorry

end trucks_after_redistribution_l55_55494


namespace linear_eq_substitution_l55_55158

theorem linear_eq_substitution (x y : ℝ) (h1 : 3 * x - 4 * y = 2) (h2 : x = 2 * y - 1) :
  3 * (2 * y - 1) - 4 * y = 2 :=
by
  sorry

end linear_eq_substitution_l55_55158


namespace part1_part2_l55_55894

open Complex

def equation (a z : ℂ) : Prop := z^2 - (a + I) * z - (I + 2) = 0

theorem part1 (m : ℝ) (a : ℝ) : equation a m → a = 1 := by
  sorry

theorem part2 (a : ℝ) : ¬ ∃ n : ℝ, equation a (n * I) := by
  sorry

end part1_part2_l55_55894


namespace problem1_problem2_l55_55341

open Set

variable (a : Real)

-- Problem 1: Prove the intersection M ∩ (C_R N) equals the given set
theorem problem1 :
  let M := { x : ℝ | x^2 - 3*x ≤ 10 }
  let N := { x : ℝ | 3 ≤ x ∧ x ≤ 5 }
  let C_RN := { x : ℝ | x < 3 ∨ 5 < x }
  M ∩ C_RN = { x : ℝ | -2 ≤ x ∧ x < 3 } :=
by
  sorry

-- Problem 2: Prove the range of values for a such that M ∪ N = M
theorem problem2 :
  let M := { x : ℝ | x^2 - 3*x ≤ 10 }
  let N := { x : ℝ | a+1 ≤ x ∧ x ≤ 2*a+1 }
  (M ∪ N = M) → a ≤ 2 :=
by
  sorry

end problem1_problem2_l55_55341


namespace find_number_l55_55918

theorem find_number (x : ℚ) (h : 1 + 1 / x = 5 / 2) : x = 2 / 3 :=
by
  sorry

end find_number_l55_55918


namespace point_d_lies_on_graph_l55_55257

theorem point_d_lies_on_graph : (-1 : ℝ) = -2 * (1 : ℝ) + 1 :=
by {
  sorry
}

end point_d_lies_on_graph_l55_55257


namespace Stonewall_marching_band_max_members_l55_55712

theorem Stonewall_marching_band_max_members (n : ℤ) (h1 : 30 * n % 34 = 2) (h2 : 30 * n < 1500) : 30 * n = 1260 :=
by
  sorry

end Stonewall_marching_band_max_members_l55_55712


namespace peter_twice_as_old_in_years_l55_55203

def mother_age : ℕ := 60
def harriet_current_age : ℕ := 13
def peter_current_age : ℕ := mother_age / 2
def years_later : ℕ := 4

theorem peter_twice_as_old_in_years : 
  peter_current_age + years_later = 2 * (harriet_current_age + years_later) :=
by
  -- using given conditions 
  -- Peter's current age is 30
  -- Harriet's current age is 13
  -- years_later is 4
  sorry

end peter_twice_as_old_in_years_l55_55203


namespace seating_arrangements_l55_55613

theorem seating_arrangements (n : ℕ) (hn : n = 8) : 
  ∃ (k : ℕ), k = 5760 :=
by
  sorry

end seating_arrangements_l55_55613


namespace find_f4_l55_55929

def f1 : ℝ × ℝ := (-2, -1)
def f2 : ℝ × ℝ := (-3, 2)
def f3 : ℝ × ℝ := (4, -3)
def equilibrium_condition (f4 : ℝ × ℝ) : Prop :=
  f1 + f2 + f3 + f4 = (0, 0)

-- Statement that needs to be proven
theorem find_f4 : ∃ (f4 : ℝ × ℝ), equilibrium_condition f4 :=
  by
  use (1, 2)
  sorry

end find_f4_l55_55929


namespace gcd_36745_59858_l55_55533

theorem gcd_36745_59858 : Nat.gcd 36745 59858 = 7 :=
sorry

end gcd_36745_59858_l55_55533


namespace distance_between_intersections_l55_55159

theorem distance_between_intersections :
  let a := 3
  let b := 2
  let c := -7
  let x1 := (-1 + Real.sqrt 22) / 3
  let x2 := (-1 - Real.sqrt 22) / 3
  let distance := abs (x1 - x2)
  let p := 88  -- 2^2 * 22 = 88
  let q := 9   -- 3^2 = 9
  distance = 2 * Real.sqrt 22 / 3 →
  p - q = 79 :=
by
  sorry

end distance_between_intersections_l55_55159


namespace avg_books_rounded_l55_55627

def books_read : List (ℕ × ℕ) := [(1, 4), (2, 3), (3, 6), (4, 2), (5, 4)]

noncomputable def total_books_read (books : List (ℕ × ℕ)) : ℕ :=
  books.foldl (λ acc pair => acc + pair.fst * pair.snd) 0

noncomputable def total_members (books : List (ℕ × ℕ)) : ℕ :=
  books.foldl (λ acc pair => acc + pair.snd) 0

noncomputable def average_books_read (books : List (ℕ × ℕ)) : ℤ :=
  Int.ofNat (total_books_read books) / Int.ofNat (total_members books)

theorem avg_books_rounded :
  average_books_read books_read = 3 :=
by 
  sorry

end avg_books_rounded_l55_55627


namespace number_below_267_is_301_l55_55951

-- Define the row number function
def rowNumber (n : ℕ) : ℕ :=
  Nat.sqrt n + 1

-- Define the starting number of a row
def rowStart (k : ℕ) : ℕ :=
  (k - 1) * (k - 1) + 1

-- Define the number in the row below given a number and its position in the row
def numberBelow (n : ℕ) : ℕ :=
  let k := rowNumber n
  let startK := rowStart k
  let position := n - startK
  let startNext := rowStart (k + 1)
  startNext + position

-- Prove that the number below 267 is 301
theorem number_below_267_is_301 : numberBelow 267 = 301 :=
by
  -- skip proof details, just the statement is needed
  sorry

end number_below_267_is_301_l55_55951


namespace least_number_to_subtract_l55_55603

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (k : ℕ) (hk : 42398 % 15 = k) : k = 8 :=
by
  sorry

end least_number_to_subtract_l55_55603


namespace triangle_product_l55_55293

theorem triangle_product (a b c: ℕ) (p: ℕ)
    (h1: ∃ k1 k2 k3: ℕ, a * k1 * k2 = p ∧ k2 * k3 * b = p ∧ k3 * c * a = p) 
    : (1 ≤ c ∧ c ≤ 336) :=
by
  sorry

end triangle_product_l55_55293


namespace quadratic_discriminant_l55_55189

def discriminant (a b c : ℚ) : ℚ :=
  b^2 - 4 * a * c

theorem quadratic_discriminant : discriminant 5 (5 + 1/2) (-2) = 281/4 := by
  sorry

end quadratic_discriminant_l55_55189


namespace parabola_at_point_has_value_zero_l55_55333

theorem parabola_at_point_has_value_zero (a m : ℝ) :
  (x ^ 2 + (a + 1) * x + a) = 0 -> m = 0 :=
by
  -- We know the parabola passes through the point (-1, m)
  sorry

end parabola_at_point_has_value_zero_l55_55333


namespace find_a5_l55_55857

-- Sequence definition
def a : ℕ → ℤ
| 0     => 1
| (n+1) => 2 * a n + 3

-- Theorem to prove
theorem find_a5 : a 4 = 61 := sorry

end find_a5_l55_55857


namespace total_revenue_correct_l55_55434

def KwikETaxCenter : Type := ℕ

noncomputable def federal_return_price : ℕ := 50
noncomputable def state_return_price : ℕ := 30
noncomputable def quarterly_business_taxes_price : ℕ := 80
noncomputable def international_return_price : ℕ := 100
noncomputable def value_added_service_price : ℕ := 75

noncomputable def federal_returns_sold : ℕ := 60
noncomputable def state_returns_sold : ℕ := 20
noncomputable def quarterly_returns_sold : ℕ := 10
noncomputable def international_returns_sold : ℕ := 13
noncomputable def value_added_services_sold : ℕ := 25

noncomputable def international_discount : ℕ := 20

noncomputable def calculate_total_revenue 
   (federal_price : ℕ) (state_price : ℕ) 
   (quarterly_price : ℕ) (international_price : ℕ) 
   (value_added_price : ℕ)
   (federal_sold : ℕ) (state_sold : ℕ) 
   (quarterly_sold : ℕ) (international_sold : ℕ) 
   (value_added_sold : ℕ)
   (discount : ℕ) : ℕ := 
    (federal_price * federal_sold) 
  + (state_price * state_sold) 
  + (quarterly_price * quarterly_sold) 
  + ((international_price - discount) * international_sold) 
  + (value_added_price * value_added_sold)

theorem total_revenue_correct :
  calculate_total_revenue federal_return_price state_return_price 
                          quarterly_business_taxes_price international_return_price 
                          value_added_service_price
                          federal_returns_sold state_returns_sold 
                          quarterly_returns_sold international_returns_sold 
                          value_added_services_sold 
                          international_discount = 7315 := 
  by sorry

end total_revenue_correct_l55_55434


namespace complement_set_A_in_U_l55_55966

-- Given conditions
def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {x | x ∈ U ∧ x^2 < 1}

-- Theorem to prove complement
theorem complement_set_A_in_U :
  U \ A = {-1, 1, 2} :=
by
  sorry

end complement_set_A_in_U_l55_55966


namespace fraction_problem_l55_55634

theorem fraction_problem :
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 9) = 531 / 322 :=
by sorry

end fraction_problem_l55_55634


namespace find_C_D_l55_55027

theorem find_C_D : ∃ C D, 
  (∀ x, x ≠ 3 → x ≠ 5 → (6*x - 3) / (x^2 - 8*x + 15) = C / (x - 3) + D / (x - 5)) ∧ 
  C = -15/2 ∧ D = 27/2 := by
  sorry

end find_C_D_l55_55027


namespace rectangle_area_ratio_l55_55184

theorem rectangle_area_ratio (length width diagonal : ℝ) (h_ratio : length / width = 5 / 2) (h_diagonal : diagonal = 13) :
    ∃ k : ℝ, (length * width) = k * diagonal^2 ∧ k = 10 / 29 :=
by
  sorry

end rectangle_area_ratio_l55_55184


namespace nobody_but_angela_finished_9_problems_l55_55391

theorem nobody_but_angela_finished_9_problems :
  ∀ (total_problems martha_problems : ℕ)
    (jenna_problems : ℕ → ℕ)
    (mark_problems : ℕ → ℕ),
    total_problems = 20 →
    martha_problems = 2 →
    jenna_problems martha_problems = 4 * martha_problems - 2 →
    mark_problems (jenna_problems martha_problems) = (jenna_problems martha_problems) / 2 →
    total_problems - (martha_problems + jenna_problems martha_problems + mark_problems (jenna_problems martha_problems)) = 9 :=
by
  intros total_problems martha_problems jenna_problems mark_problems h_total h_martha h_jenna h_mark
  sorry

end nobody_but_angela_finished_9_problems_l55_55391


namespace mixed_groups_count_l55_55694

theorem mixed_groups_count 
    (total_children : ℕ)
    (total_groups : ℕ)
    (group_size : ℕ)
    (total_games : ℕ)
    (boy_vs_boy_games : ℕ)
    (girl_vs_girl_games : ℕ)
    (total_mixed_games : ℕ)
    (games_per_group : ℕ)
    (mixed_group_games_per_group : ℕ)
    (mixed_groups : ℕ) :
    total_children = 90 →
    total_groups = 30 →
    group_size = 3 →
    total_games = 90 →
    boy_vs_boy_games = 30 →
    girl_vs_girl_games = 14 →
    total_mixed_games = 46 →
    games_per_group = 3 →
    mixed_group_games_per_group = 2 →
    mixed_groups = total_mixed_games / mixed_group_games_per_group →
    mixed_groups = 23 :=
by
    intros
    sorry

end mixed_groups_count_l55_55694


namespace solve_diophantine_l55_55596

theorem solve_diophantine :
  {xy : ℤ × ℤ | 5 * (xy.1 ^ 2) + 5 * xy.1 * xy.2 + 5 * (xy.2 ^ 2) = 7 * xy.1 + 14 * xy.2} = {(-1, 3), (0, 0), (1, 2)} :=
by sorry

end solve_diophantine_l55_55596


namespace range_of_a_l55_55752

theorem range_of_a (a : ℝ) (e : ℝ) (x : ℝ) (ln : ℝ → ℝ) :
  (∀ x, (1 / e) ≤ x ∧ x ≤ e → (a - x^2 = -2 * ln x)) →
  (1 ≤ a ∧ a ≤ (e^2 - 2)) :=
by
  sorry

end range_of_a_l55_55752


namespace probability_same_color_plates_l55_55864

noncomputable def choose : ℕ → ℕ → ℕ := Nat.choose

theorem probability_same_color_plates :
  (choose 6 3 : ℚ) / (choose 11 3 : ℚ) = 4 / 33 := by
  sorry

end probability_same_color_plates_l55_55864


namespace two_a_plus_two_d_eq_zero_l55_55052

theorem two_a_plus_two_d_eq_zero
  (a b c d : ℝ)
  (h₀ : a ≠ 0)
  (h₁ : b ≠ 0)
  (h₂ : c ≠ 0)
  (h₃ : d ≠ 0)
  (h₄ : ∀ x : ℝ, (2 * a * ((2 * a * x + b) / (3 * c * x + 2 * d)) + b)
                 / (3 * c * ((2 * a * x + b) / (3 * c * x + 2 * d)) + 2 * d) = x) :
  2 * a + 2 * d = 0 :=
by sorry

end two_a_plus_two_d_eq_zero_l55_55052


namespace problem_solution_l55_55960

-- Definitions for the digits and arithmetic conditions
def is_digit (n : ℕ) : Prop := n < 10

-- Problem conditions stated in Lean
variables (A B C D E : ℕ)

-- Define the conditions
axiom digits_A : is_digit A
axiom digits_B : is_digit B
axiom digits_C : is_digit C
axiom digits_D : is_digit D
axiom digits_E : is_digit E

-- Subtraction result for second equation
axiom sub_eq : A - C = A

-- Additional conditions derived from the problem
axiom add_eq : (E + E = D)

-- Now, state the problem in Lean
theorem problem_solution : D = 8 :=
sorry

end problem_solution_l55_55960


namespace part_one_costs_part_two_feasible_values_part_three_min_cost_l55_55238

noncomputable def cost_of_stationery (a b : ℕ) (cost_A_and_B₁ : 2 * a + b = 35) (cost_A_and_B₂ : a + 3 * b = 30): ℕ × ℕ :=
(a, b)

theorem part_one_costs (a b : ℕ) (h₁ : 2 * a + b = 35) (h₂ : a + 3 * b = 30): cost_of_stationery a b h₁ h₂ = (15, 5) :=
sorry

theorem part_two_feasible_values (x : ℕ) (h₁ : x + (120 - x) = 120) (h₂ : 975 ≤ 15 * x + 5 * (120 - x)) (h₃ : 15 * x + 5 * (120 - x) ≤ 1000):
  x = 38 ∨ x = 39 ∨ x = 40 :=
sorry

theorem part_three_min_cost (x : ℕ) (h₁ : x = 38 ∨ x = 39 ∨ x = 40):
  ∃ min_cost, (min_cost = 10 * 38 + 600 ∧ min_cost ≤ 10 * x + 600) :=
sorry

end part_one_costs_part_two_feasible_values_part_three_min_cost_l55_55238


namespace simplify_fraction_l55_55664

variable (x : ℕ)

theorem simplify_fraction (h : x = 3) : (x^10 + 15 * x^5 + 125) / (x^5 + 5) = 248 + 25 / 62 := by
  sorry

end simplify_fraction_l55_55664


namespace minimum_S_l55_55324

theorem minimum_S (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  S = (a + 1/a)^2 + (b + 1/b)^2 → S ≥ 8 :=
by
  sorry

end minimum_S_l55_55324


namespace find_angle_D_l55_55868

theorem find_angle_D (A B C D : ℝ) (h1 : A + B = 180) (h2 : C = D) (h3 : A = 40) (h4 : B + C = 130) : D = 40 := by
  sorry

end find_angle_D_l55_55868


namespace find_a_l55_55969

def star (a b : ℝ) : ℝ := 2 * a - b^2

theorem find_a (a : ℝ) (h : star a 5 = 9) : a = 17 := by
  sorry

end find_a_l55_55969


namespace dad_strawberries_weight_l55_55530

-- Definitions for the problem
def weight_marco := 15
def total_weight := 37

-- Theorem statement
theorem dad_strawberries_weight :
  (total_weight - weight_marco = 22) :=
by
  sorry

end dad_strawberries_weight_l55_55530


namespace cannot_obtain_100_pieces_l55_55050

theorem cannot_obtain_100_pieces : ¬ ∃ n : ℕ, 1 + 2 * n = 100 := by
  sorry

end cannot_obtain_100_pieces_l55_55050


namespace cubic_ineq_l55_55626

theorem cubic_ineq (x p q : ℝ) (h : x^3 + p * x + q = 0) : 4 * q * x ≤ p^2 := 
  sorry

end cubic_ineq_l55_55626


namespace proposition_false_at_6_l55_55047

variable (P : ℕ → Prop)

theorem proposition_false_at_6 (h1 : ∀ k : ℕ, 0 < k → P k → P (k + 1)) (h2 : ¬P 7): ¬P 6 :=
by
  sorry

end proposition_false_at_6_l55_55047


namespace natalia_total_distance_l55_55457

theorem natalia_total_distance :
  let dist_mon := 40
  let bonus_mon := 0.05 * dist_mon
  let effective_mon := dist_mon + bonus_mon
  
  let dist_tue := 50
  let bonus_tue := 0.03 * dist_tue
  let effective_tue := dist_tue + bonus_tue
  
  let dist_wed := dist_tue / 2
  let bonus_wed := 0.07 * dist_wed
  let effective_wed := dist_wed + bonus_wed
  
  let dist_thu := dist_mon + dist_wed
  let bonus_thu := 0.04 * dist_thu
  let effective_thu := dist_thu + bonus_thu
  
  let dist_fri := 1.2 * dist_thu
  let bonus_fri := 0.06 * dist_fri
  let effective_fri := dist_fri + bonus_fri
  
  let dist_sat := 0.75 * dist_fri
  let bonus_sat := 0.02 * dist_sat
  let effective_sat := dist_sat + bonus_sat
  
  let dist_sun := dist_sat - dist_wed
  let bonus_sun := 0.10 * dist_sun
  let effective_sun := dist_sun + bonus_sun
  
  effective_mon + effective_tue + effective_wed + effective_thu + effective_fri + effective_sat + effective_sun = 367.05 :=
by
  sorry

end natalia_total_distance_l55_55457


namespace harmonic_mean_of_4_and_5040_is_8_closest_l55_55773

noncomputable def harmonicMean (a b : ℕ) : ℝ :=
  (2 * a * b) / (a + b)

theorem harmonic_mean_of_4_and_5040_is_8_closest :
  abs (harmonicMean 4 5040 - 8) < 1 :=
by
  -- The proof process would go here
  sorry

end harmonic_mean_of_4_and_5040_is_8_closest_l55_55773


namespace employees_use_public_transportation_l55_55700

theorem employees_use_public_transportation 
  (total_employees : ℕ)
  (percentage_drive : ℕ)
  (half_of_non_drivers_take_transport : ℕ)
  (h1 : total_employees = 100)
  (h2 : percentage_drive = 60)
  (h3 : half_of_non_drivers_take_transport = 1 / 2) 
  : (total_employees - percentage_drive * total_employees / 100) / 2 = 20 := 
  by
  sorry

end employees_use_public_transportation_l55_55700


namespace robin_gum_total_l55_55236

theorem robin_gum_total :
  let original_gum := 18.0
  let given_gum := 44.0
  original_gum + given_gum = 62.0 := by
  sorry

end robin_gum_total_l55_55236


namespace eval_at_2_l55_55421

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem eval_at_2 : f 2 = 62 := by
  sorry

end eval_at_2_l55_55421


namespace jake_present_weight_l55_55366

theorem jake_present_weight :
  ∃ (J K L : ℕ), J = 194 ∧ J + K = 287 ∧ J - L = 2 * K ∧ J = 194 := by
  sorry

end jake_present_weight_l55_55366


namespace average_is_correct_l55_55523

def numbers : List ℕ := [1200, 1300, 1400, 1510, 1520, 1530, 1200]

def sum_of_numbers : ℕ := numbers.sum
def count_of_numbers : ℕ := numbers.length
def average_of_numbers : ℚ := sum_of_numbers / count_of_numbers

theorem average_is_correct : average_of_numbers = 1380 := 
by 
  -- Here, you would normally put the proof steps.
  sorry

end average_is_correct_l55_55523


namespace min_gx1_gx2_l55_55850

noncomputable def f (x a : ℝ) : ℝ := x - (1 / x) - a * Real.log x
noncomputable def g (x a : ℝ) : ℝ := x - (a / 2) * Real.log x

theorem min_gx1_gx2 (x1 x2 a : ℝ) (h1 : 0 < x1 ∧ x1 < Real.exp 1) (h2 : 0 < x2) (hx1x2: x1 * x2 = 1) (ha : a > 0) :
  f x1 a = 0 ∧ f x2 a = 0 →
  g x1 a - g x2 a = -2 / Real.exp 1 :=
by sorry

end min_gx1_gx2_l55_55850


namespace range_of_f_lt_f2_l55_55786

-- Definitions for the given conditions
def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def increasing_on (f : ℝ → ℝ) (S : Set ℝ) := ∀ ⦃a b : ℝ⦄, a ∈ S → b ∈ S → a < b → f a < f b

-- Lean 4 statement for the proof problem
theorem range_of_f_lt_f2 (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_increasing : increasing_on f {x | x ≤ 0}) : 
  ∀ x : ℝ, f x < f 2 → x > 2 ∨ x < -2 :=
by
  sorry

end range_of_f_lt_f2_l55_55786


namespace min_cost_and_ways_l55_55131

-- Define the cost of each package
def cost_A : ℕ := 10
def cost_B : ℕ := 5

-- Define a function to calculate the total cost given the number of each package
def total_cost (nA nB : ℕ) : ℕ := nA * cost_A + nB * cost_B

-- Define the number of friends
def num_friends : ℕ := 4

-- Prove the minimum cost is 15 yuan and there are 28 ways
theorem min_cost_and_ways :
  (∃ nA nB : ℕ, total_cost nA nB = 15 ∧ (
    (nA = 1 ∧ nB = 1 ∧ 12 = 12) ∨ 
    (nA = 0 ∧ nB = 3 ∧ 12 = 12) ∨
    (nA = 0 ∧ nB = 3 ∧ 4 = 4) → 28 = 28)) :=
sorry

end min_cost_and_ways_l55_55131


namespace tangent_line_right_triangle_l55_55256

theorem tangent_line_right_triangle {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (tangent_condition : a^2 + b^2 = c^2) : 
  (abs c)^2 = (abs a)^2 + (abs b)^2 :=
by
  sorry

end tangent_line_right_triangle_l55_55256


namespace sin_minus_cos_eq_sqrt2_l55_55309

theorem sin_minus_cos_eq_sqrt2 (x : ℝ) (hx1: 0 ≤ x) (hx2: x < 2 * Real.pi) (h: Real.sin x - Real.cos x = Real.sqrt 2) : x = (3 * Real.pi) / 4 :=
sorry

end sin_minus_cos_eq_sqrt2_l55_55309


namespace evaluate_f_x_plus_3_l55_55094

def f (x : ℝ) : ℝ := x^2

theorem evaluate_f_x_plus_3 (x : ℝ) : f (x + 3) = x^2 + 6 * x + 9 := by
  sorry

end evaluate_f_x_plus_3_l55_55094


namespace quadratic_solution_range_l55_55225

theorem quadratic_solution_range :
  ∃ x : ℝ, x^2 + 12 * x - 15 = 0 ∧ 1.1 < x ∧ x < 1.2 :=
sorry

end quadratic_solution_range_l55_55225


namespace range_of_a_l55_55967

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x < 2 → (a+1)*x > 2*a+2) → a < -1 :=
by
  sorry

end range_of_a_l55_55967


namespace points_satisfy_equation_l55_55566

theorem points_satisfy_equation :
  ∀ (x y : ℝ), x^2 - y^4 = Real.sqrt (18 * x - x^2 - 81) ↔ 
               (x = 9 ∧ y = Real.sqrt 3) ∨ (x = 9 ∧ y = -Real.sqrt 3) := 
by 
  intros x y 
  sorry

end points_satisfy_equation_l55_55566


namespace max_correct_answers_l55_55037

theorem max_correct_answers (c w b : ℕ) 
  (h1 : c + w + b = 25) 
  (h2 : 5 * c - 2 * w = 60) : 
  c ≤ 14 := 
sorry

end max_correct_answers_l55_55037


namespace triangle_area_l55_55335

theorem triangle_area (a b c : ℝ) (A B C : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) 
  (h_c : c = 2) (h_C : C = π / 3)
  (h_sin : Real.sin B = 2 * Real.sin A) :
  (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3 / 3 :=
sorry

end triangle_area_l55_55335


namespace checkerboard_no_identical_numbers_l55_55577

theorem checkerboard_no_identical_numbers :
  ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 11 ∧ 1 ≤ j ∧ j ≤ 19 → 19 * (i - 1) + j = 11 * (j - 1) + i → false :=
by
  sorry

end checkerboard_no_identical_numbers_l55_55577


namespace oil_bill_january_l55_55709

theorem oil_bill_january (F J : ℝ) (h1 : F / J = 3 / 2) (h2 : (F + 30) / J = 5 / 3) : J = 180 :=
by
  sorry

end oil_bill_january_l55_55709


namespace trees_planted_l55_55565

theorem trees_planted (interval trail_length : ℕ) (h1 : interval = 30) (h2 : trail_length = 1200) : 
  trail_length / interval = 40 :=
by
  sorry

end trees_planted_l55_55565


namespace area_difference_l55_55001

-- Definitions of the conditions
def length_rect := 60 -- length of the rectangular garden in feet
def width_rect := 20 -- width of the rectangular garden in feet

-- Compute the area of the rectangular garden
def area_rect := length_rect * width_rect

-- Compute the perimeter of the rectangular garden
def perimeter_rect := 2 * (length_rect + width_rect)

-- Compute the side length of the square garden from the same perimeter
def side_square := perimeter_rect / 4

-- Compute the area of the square garden
def area_square := side_square * side_square

-- The goal is to prove the area difference
theorem area_difference : area_square - area_rect = 400 := by
  sorry -- Proof to be completed

end area_difference_l55_55001


namespace quadratic_function_expression_l55_55617

-- Definitions based on conditions
def quadratic (f : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c
def condition1 (f : ℝ → ℝ) : Prop := (f 0 = 1)
def condition2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) - f x = 4 * x

-- The theorem we want to prove
theorem quadratic_function_expression (f : ℝ → ℝ) 
  (hf_quad : quadratic f)
  (hf_cond1 : condition1 f)
  (hf_cond2 : condition2 f) : 
  ∃ (a b c : ℝ), a = 2 ∧ b = -2 ∧ c = 1 ∧ ∀ x, f x = a * x^2 + b * x + c :=
sorry

end quadratic_function_expression_l55_55617


namespace sum_of_solutions_l55_55874

  theorem sum_of_solutions :
    (∃ x : ℝ, x = abs (2 * x - abs (50 - 2 * x)) ∧ ∃ y : ℝ, y = abs (2 * y - abs (50 - 2 * y)) ∧ ∃ z : ℝ, z = abs (2 * z - abs (50 - 2 * z)) ∧ (x + y + z = 170 / 3)) :=
  sorry
  
end sum_of_solutions_l55_55874


namespace find_divisor_l55_55601

theorem find_divisor (h : 2994 / 14.5 = 171) : 29.94 / 1.75 = 17.1 :=
by
  sorry

end find_divisor_l55_55601


namespace cone_volume_in_liters_l55_55278

theorem cone_volume_in_liters (d h : ℝ) (pi : ℝ) (liters_conversion : ℝ) :
  d = 12 → h = 10 → liters_conversion = 1000 → (1/3) * pi * (d/2)^2 * h * (1 / liters_conversion) = 0.12 * pi :=
by
  intros hd hh hc
  sorry

end cone_volume_in_liters_l55_55278


namespace average_test_score_fifty_percent_l55_55261

-- Given conditions
def percent1 : ℝ := 15
def avg1 : ℝ := 100
def percent2 : ℝ := 50
def avg3 : ℝ := 63
def overall_average : ℝ := 76.05

-- Intermediate calculations based on given conditions
def total_percent : ℝ := 100
def percent3: ℝ := total_percent - percent1 - percent2
def sum_of_weights: ℝ := overall_average * total_percent

-- Expected average of the group that is 50% of the class
theorem average_test_score_fifty_percent (X: ℝ) :
  sum_of_weights = percent1 * avg1 + percent2 * X + percent3 * avg3 → X = 78 := by
  sorry

end average_test_score_fifty_percent_l55_55261


namespace total_fish_l55_55363

variable (L A : ℕ)

theorem total_fish (h1 : L = 40) (h2 : A = L + 20) : L + A = 100 := by 
  sorry

end total_fish_l55_55363


namespace camping_trip_percentage_l55_55382

theorem camping_trip_percentage (t : ℕ) (h1 : 22 / 100 * t > 0) (h2 : 75 / 100 * (22 / 100 * t) ≤ t) :
  (88 / 100 * t) = t :=
by
  sorry

end camping_trip_percentage_l55_55382


namespace totalMoney_l55_55788

noncomputable def joannaMoney : ℕ := 8
noncomputable def brotherMoney : ℕ := 3 * joannaMoney
noncomputable def sisterMoney : ℕ := joannaMoney / 2

theorem totalMoney : joannaMoney + brotherMoney + sisterMoney = 36 := by
  sorry

end totalMoney_l55_55788


namespace negation_of_p_equiv_h_l55_55912

variable (p : ∀ x : ℝ, Real.sin x ≤ 1)
variable (h : ∃ x : ℝ, Real.sin x ≥ 1)

theorem negation_of_p_equiv_h : (¬ (∀ x : ℝ, Real.sin x ≤ 1)) ↔ (∃ x : ℝ, Real.sin x ≥ 1) :=
by
  sorry

end negation_of_p_equiv_h_l55_55912


namespace lizette_stamps_count_l55_55827

-- Conditions
def lizette_more : ℕ := 125
def minerva_stamps : ℕ := 688

-- Proof of Lizette's stamps count
theorem lizette_stamps_count : (minerva_stamps + lizette_more = 813) :=
by 
  sorry

end lizette_stamps_count_l55_55827


namespace range_of_a_l55_55401

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x = 1 → x > a) : a < 1 := 
by
  sorry

end range_of_a_l55_55401


namespace sqrt_twentyfive_eq_five_l55_55651

theorem sqrt_twentyfive_eq_five : Real.sqrt 25 = 5 := by
  sorry

end sqrt_twentyfive_eq_five_l55_55651


namespace matrix_power_problem_l55_55842

def B : Matrix (Fin 2) (Fin 2) ℤ := 
  ![![4, 1], ![0, 2]]

theorem matrix_power_problem : B^15 - 3 * B^14 = ![![4, 3], ![0, -2]] :=
  by sorry

end matrix_power_problem_l55_55842


namespace john_spent_on_candy_l55_55863

theorem john_spent_on_candy (M : ℝ) 
  (h1 : M = 29.999999999999996)
  (h2 : 1/5 + 1/3 + 1/10 = 19/30) :
  (11 / 30) * M = 11 :=
by {
  sorry
}

end john_spent_on_candy_l55_55863


namespace combined_width_approximately_8_l55_55240

noncomputable def C1 := 352 / 7
noncomputable def C2 := 528 / 7
noncomputable def C3 := 704 / 7

noncomputable def r1 := C1 / (2 * Real.pi)
noncomputable def r2 := C2 / (2 * Real.pi)
noncomputable def r3 := C3 / (2 * Real.pi)

noncomputable def W1 := r2 - r1
noncomputable def W2 := r3 - r2

noncomputable def combined_width := W1 + W2

theorem combined_width_approximately_8 :
  |combined_width - 8| < 1 :=
by
  sorry

end combined_width_approximately_8_l55_55240


namespace arithmetic_sequence_a8_l55_55910

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_a8 (h : a 7 + a 9 = 8) : a 8 = 4 := 
by 
  -- proof steps would go here
  sorry

end arithmetic_sequence_a8_l55_55910


namespace problem_statement_l55_55919

theorem problem_statement (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 ∧ x^2 + y^2 = 104 :=
by
  sorry

end problem_statement_l55_55919


namespace proof_total_distance_l55_55284

-- Define the total distance
def total_distance (D : ℕ) :=
  let by_foot := (1 : ℚ) / 6
  let by_bicycle := (1 : ℚ) / 4
  let by_bus := (1 : ℚ) / 3
  let by_car := 10
  let by_train := (1 : ℚ) / 12
  D - (by_foot + by_bicycle + by_bus + by_train) * D = by_car

-- Given proof problem
theorem proof_total_distance : ∃ D : ℕ, total_distance D ∧ D = 60 :=
sorry

end proof_total_distance_l55_55284


namespace simplify_fraction_rationalize_denominator_l55_55349

theorem simplify_fraction_rationalize_denominator :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18) = 5 * Real.sqrt 2 / 28 :=
by
  have sqrt_50 : Real.sqrt 50 = 5 * Real.sqrt 2 := sorry
  have sqrt_8 : 3 * Real.sqrt 8 = 6 * Real.sqrt 2 := sorry
  have sqrt_18 : Real.sqrt 18 = 3 * Real.sqrt 2 := sorry
  sorry

end simplify_fraction_rationalize_denominator_l55_55349


namespace incorrect_reciprocal_quotient_l55_55241

-- Definitions based on problem conditions
def identity_property (x : ℚ) : x * 1 = x := by sorry
def division_property (a b : ℚ) (h : b ≠ 0) : a / b = 0 → a = 0 := by sorry
def additive_inverse_property (x : ℚ) : x * (-1) = -x := by sorry

-- Statement that needs to be proved
theorem incorrect_reciprocal_quotient (a b : ℚ) (h1 : a ≠ 0) (h2 : b = 1 / a) : a / b ≠ 1 :=
by sorry

end incorrect_reciprocal_quotient_l55_55241


namespace greatest_value_is_B_l55_55886

def x : Int := -6

def A : Int := 2 + x
def B : Int := 2 - x
def C : Int := x - 1
def D : Int := x
def E : Int := x / 2

theorem greatest_value_is_B :
  B > A ∧ B > C ∧ B > D ∧ B > E :=
by
  sorry

end greatest_value_is_B_l55_55886


namespace sum_of_roots_l55_55196

-- States that the sum of the values of x that satisfy the given quadratic equation is 7
theorem sum_of_roots (x : ℝ) :
  (x^2 - 7 * x + 12 = 4) → (∃ a b : ℝ, x^2 - 7 * x + 8 = 0 ∧ a + b = 7) :=
by
  sorry

end sum_of_roots_l55_55196


namespace lana_spent_l55_55796

def ticket_cost : ℕ := 6
def tickets_for_friends : ℕ := 8
def extra_tickets : ℕ := 2

theorem lana_spent :
  ticket_cost * (tickets_for_friends + extra_tickets) = 60 := 
by
  sorry

end lana_spent_l55_55796


namespace probability_white_then_black_l55_55207

-- Definition of conditions
def total_balls := 5
def white_balls := 3
def black_balls := 2

def first_draw_white_probability (total white : ℕ) : ℚ :=
  white / total

def second_draw_black_probability (remaining_white remaining_black : ℕ) : ℚ :=
  remaining_black / (remaining_white + remaining_black)

-- The theorem statement
theorem probability_white_then_black :
  first_draw_white_probability total_balls white_balls *
  second_draw_black_probability (total_balls - 1) black_balls
  = 3 / 10 :=
by
  sorry

end probability_white_then_black_l55_55207


namespace dartboard_odd_score_probability_l55_55723

theorem dartboard_odd_score_probability :
  let π := Real.pi
  let r_outer := 4
  let r_inner := 2
  let area_inner := π * r_inner * r_inner
  let area_outer := π * r_outer * r_outer
  let area_annulus := area_outer - area_inner
  let area_inner_region := area_inner / 3
  let area_outer_region := area_annulus / 3
  let odd_inner_regions := 1
  let even_inner_regions := 2
  let odd_outer_regions := 2
  let even_outer_regions := 1
  let prob_odd_inner := (odd_inner_regions * area_inner_region) / (area_inner + area_annulus)
  let prob_even_inner := (even_inner_regions * area_inner_region) / (area_inner + area_annulus)
  let prob_odd_outer := (odd_outer_regions * area_outer_region) / (area_inner + area_annulus)
  let prob_even_outer := (even_outer_regions * area_outer_region) / (area_inner + area_annulus)
  let prob_odd_region := prob_odd_inner + prob_odd_outer
  let prob_even_region := prob_even_inner + prob_even_outer
  let prob_odd_score := (prob_odd_region * prob_even_region) + (prob_even_region * prob_odd_region)
  prob_odd_score = 5 / 9 :=
by
  -- Proof omitted
  sorry

end dartboard_odd_score_probability_l55_55723


namespace find_ratio_a6_b6_l55_55705

noncomputable def S (n : ℕ) : ℕ := sorry
noncomputable def T (n : ℕ) : ℕ := sorry
noncomputable def a (n : ℕ) : ℕ := sorry
noncomputable def b (n : ℕ) : ℕ := sorry

theorem find_ratio_a6_b6 
  (H1 : ∀ n: ℕ, n > 0 → (S n / T n : ℚ) = n / (2 * n + 1)) :
  (a 6 / b 6 : ℚ) = 11 / 23 :=
sorry

end find_ratio_a6_b6_l55_55705


namespace december_sales_multiple_l55_55762

   noncomputable def find_sales_multiple (A : ℝ) (x : ℝ) :=
     x * A = 0.3888888888888889 * (11 * A + x * A)

   theorem december_sales_multiple (A : ℝ) (x : ℝ) (h : find_sales_multiple A x) : x = 7 :=
   by 
     sorry
   
end december_sales_multiple_l55_55762


namespace eq_g_of_f_l55_55708

def f (x : ℝ) : ℝ := 3 * x - 5
def g (x : ℝ) : ℝ := 6 * x - 29

theorem eq_g_of_f (x : ℝ) : 2 * (f x) - 19 = g x :=
by 
  sorry

end eq_g_of_f_l55_55708


namespace final_cost_is_correct_l55_55608

noncomputable def calculate_final_cost 
  (price_orange : ℕ)
  (price_mango : ℕ)
  (increase_percent : ℕ)
  (bulk_discount_percent : ℕ)
  (sales_tax_percent : ℕ) : ℕ := 
  let new_price_orange := price_orange + (price_orange * increase_percent) / 100
  let new_price_mango := price_mango + (price_mango * increase_percent) / 100
  let total_cost_oranges := 10 * new_price_orange
  let total_cost_mangoes := 10 * new_price_mango
  let total_cost_before_discount := total_cost_oranges + total_cost_mangoes
  let discount_oranges := (total_cost_oranges * bulk_discount_percent) / 100
  let discount_mangoes := (total_cost_mangoes * bulk_discount_percent) / 100
  let total_cost_after_discount := total_cost_before_discount - discount_oranges - discount_mangoes
  let sales_tax := (total_cost_after_discount * sales_tax_percent) / 100
  total_cost_after_discount + sales_tax

theorem final_cost_is_correct :
  calculate_final_cost 40 50 15 10 8 = 100602 :=
by
  sorry

end final_cost_is_correct_l55_55608


namespace find_some_number_l55_55583

theorem find_some_number (x some_number : ℝ) (h1 : (27 / 4) * x - some_number = 3 * x + 27) (h2 : x = 12) :
  some_number = 18 :=
by
  sorry

end find_some_number_l55_55583


namespace range_of_k_l55_55892

noncomputable def quadratic_has_real_roots (k : ℝ): Prop :=
  ∃ x : ℝ, k * x^2 - 2 * x - 1 = 0

theorem range_of_k (k : ℝ) : quadratic_has_real_roots k ↔ k ≥ -1 :=
by
  sorry

end range_of_k_l55_55892


namespace abs_neg_three_l55_55260

theorem abs_neg_three : |(-3 : ℤ)| = 3 := 
by
  sorry

end abs_neg_three_l55_55260


namespace stone_hitting_ground_time_l55_55961

noncomputable def equation (s : ℝ) : ℝ := -4.5 * s^2 - 12 * s + 48

theorem stone_hitting_ground_time :
  ∃ s : ℝ, equation s = 0 ∧ s = (-8 + 16 * Real.sqrt 7) / 6 :=
by
  sorry

end stone_hitting_ground_time_l55_55961


namespace cos_alpha_plus_pi_over_3_l55_55208

theorem cos_alpha_plus_pi_over_3 (α : ℝ) (h : Real.sin (α - π / 6) = 1 / 3) : Real.cos (α + π / 3) = -1 / 3 :=
  sorry

end cos_alpha_plus_pi_over_3_l55_55208


namespace bread_pieces_total_l55_55663

def initial_slices : ℕ := 2
def pieces_per_slice (n : ℕ) : ℕ := n * 4

theorem bread_pieces_total : pieces_per_slice initial_slices = 8 :=
by
  sorry

end bread_pieces_total_l55_55663


namespace total_students_l55_55992

variable (A B AB : ℕ)

-- Conditions
axiom h1 : AB = (1 / 5) * (A + AB)
axiom h2 : AB = (1 / 4) * (B + AB)
axiom h3 : A - B = 75

-- Proof problem
theorem total_students : A + B + AB = 600 :=
by
  sorry

end total_students_l55_55992


namespace sum_three_distinct_zero_l55_55618

variable {R : Type} [Field R]

theorem sum_three_distinct_zero
  (a b c x y : R)
  (h1 : a ^ 3 + a * x + y = 0)
  (h2 : b ^ 3 + b * x + y = 0)
  (h3 : c ^ 3 + c * x + y = 0)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  a + b + c = 0 := by
  sorry

end sum_three_distinct_zero_l55_55618


namespace car_rental_cost_l55_55667

theorem car_rental_cost
  (rent_per_day : ℝ) (cost_per_mile : ℝ) (days_rented : ℕ) (miles_driven : ℝ)
  (h1 : rent_per_day = 30)
  (h2 : cost_per_mile = 0.25)
  (h3 : days_rented = 5)
  (h4 : miles_driven = 500) :
  rent_per_day * days_rented + cost_per_mile * miles_driven = 275 := 
  by
  sorry

end car_rental_cost_l55_55667


namespace Q_mul_P_plus_Q_eq_one_l55_55985

noncomputable def sqrt5_plus_2_pow (n : ℕ) :=
  (Real.sqrt 5 + 2)^(2 * n + 1)

noncomputable def P (n : ℕ) :=
  Int.floor (sqrt5_plus_2_pow n)

noncomputable def Q (n : ℕ) :=
  sqrt5_plus_2_pow n - P n

theorem Q_mul_P_plus_Q_eq_one (n : ℕ) : Q n * (P n + Q n) = 1 := by
  sorry

end Q_mul_P_plus_Q_eq_one_l55_55985


namespace least_value_x_y_z_l55_55275

theorem least_value_x_y_z (x y z : ℕ) (hx : x = 4 * y) (hy : y = 7 * z) (hz : 0 < z) : x - y - z = 19 :=
by
  -- placeholder for actual proof
  sorry

end least_value_x_y_z_l55_55275


namespace complement_of_A_cap_B_l55_55802

def set_A (x : ℝ) : Prop := x ≤ -4 ∨ x ≥ 2
def set_B (x : ℝ) : Prop := |x - 1| ≤ 3

def A_cap_B (x : ℝ) : Prop := set_A x ∧ set_B x

def complement_A_cap_B (x : ℝ) : Prop := ¬A_cap_B x

theorem complement_of_A_cap_B :
  {x : ℝ | complement_A_cap_B x} = {x : ℝ | x < 2 ∨ x > 4} :=
by
  sorry

end complement_of_A_cap_B_l55_55802


namespace union_complement_U_A_B_l55_55597

def U : Set Int := {-1, 0, 1, 2, 3}

def A : Set Int := {-1, 0, 1}

def B : Set Int := {0, 1, 2}

def complement_U_A : Set Int := {u | u ∈ U ∧ u ∉ A}

theorem union_complement_U_A_B : (complement_U_A ∪ B) = {0, 1, 2, 3} :=
by
  sorry

end union_complement_U_A_B_l55_55597


namespace find_y_given_z_25_l55_55072

theorem find_y_given_z_25 (k m x y z : ℝ) 
  (hk : y = k * x) 
  (hm : z = m * x)
  (hy5 : y = 10) 
  (hx5z15 : z = 15) 
  (hz25 : z = 25) : 
  y = 50 / 3 := 
  by sorry

end find_y_given_z_25_l55_55072


namespace negation_of_universal_l55_55599

-- Definitions based on the provided problem
def prop (x : ℝ) : Prop := x > 0 → x^2 > 0

-- Main proof problem statement
theorem negation_of_universal : 
  ¬ (∀ x : ℝ, x > 0 → x^2 > 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 ≤ 0 :=
by sorry

end negation_of_universal_l55_55599


namespace proof_problem_l55_55695

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := (x + 1) * f x

axiom domain_f : ∀ x : ℝ, true
axiom even_f : ∀ x : ℝ, f (2 * x - 1) = f (-(2 * x - 1))
axiom mono_g_neg_inf_minus_1 : ∀ x y : ℝ, x ≤ y → x ≤ -1 → y ≤ -1 → g x ≤ g y

-- Proof Problem Statement
theorem proof_problem :
  (∀ x y : ℝ, x ≤ y → -1 ≤ x → -1 ≤ y → g x ≤ g y) ∧
  (∀ a b : ℝ, g a + g b > 0 → a + b + 2 > 0) :=
by
  sorry

end proof_problem_l55_55695


namespace arithmetic_sequence_10th_term_l55_55531

theorem arithmetic_sequence_10th_term (a d : ℤ) :
    (a + 4 * d = 26) →
    (a + 7 * d = 50) →
    (a + 9 * d = 66) := by
  intros h1 h2
  sorry

end arithmetic_sequence_10th_term_l55_55531


namespace y_difference_positive_l55_55795

theorem y_difference_positive (a c y1 y2 : ℝ) (h1 : a < 0)
  (h2 : y1 = a * 1^2 + 2 * a * 1 + c)
  (h3 : y2 = a * 2^2 + 2 * a * 2 + c) : y1 - y2 > 0 := 
sorry

end y_difference_positive_l55_55795


namespace totalShortBushes_l55_55648

namespace ProofProblem

def initialShortBushes : Nat := 37
def additionalShortBushes : Nat := 20

theorem totalShortBushes :
  initialShortBushes + additionalShortBushes = 57 := by
  sorry

end ProofProblem

end totalShortBushes_l55_55648


namespace melted_mixture_weight_l55_55942

-- Let Zinc and Copper be real numbers representing their respective weights in kilograms.
variables (Zinc Copper: ℝ)
-- Assume the ratio of Zinc to Copper is 9:11.
axiom ratio_zinc_copper : Zinc / Copper = 9 / 11
-- Assume 26.1kg of Zinc has been used.
axiom zinc_value : Zinc = 26.1

-- Define the total weight of the melted mixture.
def total_weight := Zinc + Copper

-- We state the theorem to prove that the total weight of the mixture equals 58kg.
theorem melted_mixture_weight : total_weight Zinc Copper = 58 :=
by
  sorry

end melted_mixture_weight_l55_55942


namespace correct_calculation_l55_55505

theorem correct_calculation (a b : ℝ) :
  2 * a^2 * b - 3 * a^2 * b = -a^2 * b ∧
  ¬ (a^3 * a^4 = a^12) ∧
  ¬ ((-2 * a^2 * b)^3 = -6 * a^6 * b^3) ∧
  ¬ ((a + b)^2 = a^2 + b^2) :=
by
  sorry

end correct_calculation_l55_55505


namespace intersection_lines_k_l55_55216

theorem intersection_lines_k (k : ℝ) :
  (∃ (x y : ℝ), x = 2 ∧ x - y - 1 = 0 ∧ x + k * y = 0) → k = -2 :=
by
  sorry

end intersection_lines_k_l55_55216


namespace range_of_a_l55_55870

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 3| + |x - 4| < a) ↔ a > 1 :=
by
  sorry -- The proof is omitted as per the instructions.

end range_of_a_l55_55870


namespace select_people_english_japanese_l55_55785

-- Definitions based on conditions
def total_people : ℕ := 9
def english_speakers : ℕ := 7
def japanese_speakers : ℕ := 3

-- Theorem statement
theorem select_people_english_japanese (h1 : total_people = 9) 
                                      (h2 : english_speakers = 7) 
                                      (h3 : japanese_speakers = 3) :
  ∃ n, n = 20 :=
by {
  sorry
}

end select_people_english_japanese_l55_55785


namespace polygon_stats_l55_55467

-- Definitions based on the problem's conditions
def total_number_of_polygons : ℕ := 207
def median_position : ℕ := 104
def m : ℕ := 14
def sum_of_squares_of_sides : ℕ := 2860
def mean_value : ℚ := sum_of_squares_of_sides / total_number_of_polygons
def mode_median : ℚ := 11.5

-- The proof statement
theorem polygon_stats (d μ M : ℚ)
  (h₁ : μ = mean_value)
  (h₂ : d = mode_median)
  (h₃ : M = m) :
  d < μ ∧ μ < M :=
by
  rw [h₁, h₂, h₃]
  -- The exact proof steps are omitted
  sorry

end polygon_stats_l55_55467


namespace multiplication_distributive_example_l55_55766

theorem multiplication_distributive_example : 23 * 4 = 20 * 4 + 3 * 4 := by
  sorry

end multiplication_distributive_example_l55_55766


namespace inverse_function_of_13_l55_55834

noncomputable def f (x : ℝ) : ℝ := 3 * x + 4
noncomputable def f_inv (y : ℝ) : ℝ := (y - 4) / 3

theorem inverse_function_of_13 : f_inv (f_inv 13) = -1 / 3 := by
  sorry

end inverse_function_of_13_l55_55834


namespace smallest_x_solution_l55_55552

theorem smallest_x_solution :
  (∃ x : ℝ, (3 * x^2 + 36 * x - 90 = 2 * x * (x + 16)) ∧ ∀ y : ℝ, (3 * y^2 + 36 * y - 90 = 2 * y * (y + 16)) → x ≤ y) ↔ x = -10 :=
by
  sorry

end smallest_x_solution_l55_55552


namespace tan_15_pi_over_4_l55_55851

theorem tan_15_pi_over_4 : Real.tan (15 * Real.pi / 4) = -1 :=
by
-- The proof is omitted.
sorry

end tan_15_pi_over_4_l55_55851


namespace rectangles_with_equal_perimeters_can_have_different_shapes_l55_55426

theorem rectangles_with_equal_perimeters_can_have_different_shapes (l₁ w₁ l₂ w₂ : ℝ) 
  (h₁ : l₁ + w₁ = l₂ + w₂) : (l₁ ≠ l₂ ∨ w₁ ≠ w₂) :=
by
  sorry

end rectangles_with_equal_perimeters_can_have_different_shapes_l55_55426


namespace investment_B_l55_55524

theorem investment_B {x : ℝ} :
  let a_investment := 6300
  let c_investment := 10500
  let total_profit := 12100
  let a_share_profit := 3630
  (6300 / (6300 + x + 10500) = 3630 / 12100) →
  x = 13650 :=
by { sorry }

end investment_B_l55_55524


namespace total_cost_of_apples_l55_55140

variable (num_apples_per_bag cost_per_bag num_apples : ℕ)
#check num_apples_per_bag = 50
#check cost_per_bag = 8
#check num_apples = 750

theorem total_cost_of_apples : 
  (num_apples_per_bag = 50) → 
  (cost_per_bag = 8) → 
  (num_apples = 750) → 
  (num_apples / num_apples_per_bag * cost_per_bag = 120) :=
by
  intros
  sorry

end total_cost_of_apples_l55_55140


namespace original_expenditure_l55_55328

theorem original_expenditure (initial_students new_students : ℕ) (increment_expense : ℝ) (decrement_avg_expense : ℝ) (original_avg_expense : ℝ) (new_avg_expense : ℝ) 
  (total_initial_expense original_expenditure : ℝ)
  (h1 : initial_students = 35) 
  (h2 : new_students = 7) 
  (h3 : increment_expense = 42)
  (h4 : decrement_avg_expense = 1)
  (h5 : new_avg_expense = original_avg_expense - decrement_avg_expense)
  (h6 : total_initial_expense = initial_students * original_avg_expense)
  (h7 : original_expenditure = total_initial_expense)
  (h8 : 42 * new_avg_expense - original_students * original_avg_expense = increment_expense) :
  original_expenditure = 420 := 
by
  sorry

end original_expenditure_l55_55328


namespace word_count_with_a_l55_55104

-- Defining the constants for the problem
def alphabet_size : ℕ := 26
def no_a_size : ℕ := 25

-- Calculating words that contain 'A' for lengths 1 to 5
def words_with_a (len : ℕ) : ℕ :=
  alphabet_size ^ len - no_a_size ^ len

-- The main theorem statement
theorem word_count_with_a : words_with_a 1 + words_with_a 2 + words_with_a 3 + words_with_a 4 + words_with_a 5 = 2186085 :=
by
  -- Calculations are established in the problem statement
  sorry

end word_count_with_a_l55_55104


namespace number_of_propositions_is_4_l55_55378

def is_proposition (s : String) : Prop :=
  s = "The Earth is a planet in the solar system" ∨ 
  s = "{0} ∈ ℕ" ∨ 
  s = "1+1 > 2" ∨ 
  s = "Elderly people form a set"

theorem number_of_propositions_is_4 : 
  (is_proposition "The Earth is a planet in the solar system" ∨ 
   is_proposition "{0} ∈ ℕ" ∨ 
   is_proposition "1+1 > 2" ∨ 
   is_proposition "Elderly people form a set") → 
  4 = 4 :=
by
  sorry

end number_of_propositions_is_4_l55_55378


namespace evaluate_expression_l55_55424

theorem evaluate_expression : (2301 - 2222)^2 / 144 = 43 := 
by 
  sorry

end evaluate_expression_l55_55424


namespace no_solutions_in_naturals_l55_55036

theorem no_solutions_in_naturals (n k : ℕ) : ¬ (n ≤ n! - k^n ∧ n! - k^n ≤ k * n) :=
sorry

end no_solutions_in_naturals_l55_55036


namespace total_ranking_sequences_l55_55784

-- Define teams
inductive Team
| A | B | C | D

-- Define the conditions
def qualifies (t : Team) : Prop := 
  -- Each team must win its qualifying match to participate
  true

def plays_saturday (t1 t2 t3 t4 : Team) : Prop :=
  (t1 = Team.A ∧ t2 = Team.B) ∨ (t3 = Team.C ∧ t4 = Team.D)

def plays_sunday (t1 t2 t3 t4 : Team) : Prop := 
  -- Winners of Saturday's matches play for 1st and 2nd, losers play for 3rd and 4th
  true

-- Lean statement for the proof problem
theorem total_ranking_sequences : 
  (∀ t : Team, qualifies t) → 
  (∀ t1 t2 t3 t4 : Team, plays_saturday t1 t2 t3 t4) → 
  (∀ t1 t2 t3 t4 : Team, plays_sunday t1 t2 t3 t4) → 
  ∃ n : ℕ, n = 16 :=
by 
  sorry

end total_ranking_sequences_l55_55784


namespace total_green_ducks_percentage_l55_55497

def ducks_in_park_A : ℕ := 200
def green_percentage_A : ℕ := 25

def ducks_in_park_B : ℕ := 350
def green_percentage_B : ℕ := 20

def ducks_in_park_C : ℕ := 120
def green_percentage_C : ℕ := 50

def ducks_in_park_D : ℕ := 60
def green_percentage_D : ℕ := 25

def ducks_in_park_E : ℕ := 500
def green_percentage_E : ℕ := 30

theorem total_green_ducks_percentage (green_ducks_A green_ducks_B green_ducks_C green_ducks_D green_ducks_E total_ducks : ℕ)
  (h_A : green_ducks_A = ducks_in_park_A * green_percentage_A / 100)
  (h_B : green_ducks_B = ducks_in_park_B * green_percentage_B / 100)
  (h_C : green_ducks_C = ducks_in_park_C * green_percentage_C / 100)
  (h_D : green_ducks_D = ducks_in_park_D * green_percentage_D / 100)
  (h_E : green_ducks_E = ducks_in_park_E * green_percentage_E / 100)
  (h_total_ducks : total_ducks = ducks_in_park_A + ducks_in_park_B + ducks_in_park_C + ducks_in_park_D + ducks_in_park_E) :
  (green_ducks_A + green_ducks_B + green_ducks_C + green_ducks_D + green_ducks_E) * 100 / total_ducks = 2805 / 100 :=
by sorry

end total_green_ducks_percentage_l55_55497


namespace largest_c_3_in_range_l55_55803

theorem largest_c_3_in_range (c : ℝ) : 
  (∃ x : ℝ, x^2 - 7*x + c = 3) ↔ c ≤ 61 / 4 := 
by sorry

end largest_c_3_in_range_l55_55803


namespace system_of_equations_solution_l55_55073

theorem system_of_equations_solution (x y z : ℤ) :
  x^2 - 9 * y^2 - z^2 = 0 ∧ z = x - 3 * y ↔ 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (∃ k : ℤ, x = 3 * k ∧ y = k ∧ z = 0) := 
by
  sorry

end system_of_equations_solution_l55_55073


namespace nancy_hours_to_work_l55_55614

def tuition := 22000
def scholarship := 3000
def hourly_wage := 10
def parents_contribution := tuition / 2
def student_loan := 2 * scholarship
def total_financial_aid := scholarship + student_loan
def remaining_tuition := tuition - parents_contribution - total_financial_aid
def hours_to_work := remaining_tuition / hourly_wage

theorem nancy_hours_to_work : hours_to_work = 200 := by
  -- This by block demonstrates that a proof would go here
  sorry

end nancy_hours_to_work_l55_55614


namespace first_present_cost_is_18_l55_55544

-- Conditions as definitions
variables (x : ℕ)

-- Given conditions
def first_present_cost := x
def second_present_cost := x + 7
def third_present_cost := x - 11
def total_cost := first_present_cost x + second_present_cost x + third_present_cost x

-- Statement of the problem
theorem first_present_cost_is_18 (h : total_cost x = 50) : x = 18 :=
by {
  sorry  -- Proof omitted
}

end first_present_cost_is_18_l55_55544


namespace rational_solution_cos_eq_l55_55690

theorem rational_solution_cos_eq {q : ℚ} (h0 : 0 < q) (h1 : q < 1) (heq : Real.cos (3 * Real.pi * q) + 2 * Real.cos (2 * Real.pi * q) = 0) : 
  q = 2 / 3 := 
sorry

end rational_solution_cos_eq_l55_55690


namespace MEMOrable_rectangle_count_l55_55156

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

end MEMOrable_rectangle_count_l55_55156


namespace roots_quartic_sum_l55_55629

theorem roots_quartic_sum (p q r : ℝ) 
  (h1 : p^3 - 2*p^2 + 3*p - 4 = 0)
  (h2 : q^3 - 2*q^2 + 3*q - 4 = 0)
  (h3 : r^3 - 2*r^2 + 3*r - 4 = 0)
  (h4 : p + q + r = 2)
  (h5 : p*q + q*r + r*p = 3)
  (h6 : p*q*r = 4) :
  p^4 + q^4 + r^4 = 18 := sorry

end roots_quartic_sum_l55_55629


namespace domain_of_g_l55_55628

def f (x : ℝ) : Prop := x ∈ Set.Icc (-12.0) 6.0

def g (x : ℝ) : Prop := f (3 * x)

theorem domain_of_g : Set.Icc (-4.0) 2.0 = {x : ℝ | g x} := 
by 
    sorry

end domain_of_g_l55_55628


namespace digit_150_in_17_div_70_l55_55105

noncomputable def repeating_sequence_170 : List Nat := [2, 4, 2, 8, 5, 7]

theorem digit_150_in_17_div_70 : (repeating_sequence_170.get? (150 % 6 - 1) = some 7) := by
  sorry

end digit_150_in_17_div_70_l55_55105


namespace committee_formation_l55_55088

/-- Problem statement: In how many ways can a 5-person executive committee be formed if one of the 
members must be the president, given there are 30 members. --/
theorem committee_formation (n : ℕ) (k : ℕ) (h : n = 30) (h2 : k = 5) : 
  (n * Nat.choose (n - 1) (k - 1) = 712530 ) :=
by
  sorry

end committee_formation_l55_55088


namespace domain_of_function_l55_55053

theorem domain_of_function :
  ∀ x : ℝ, (x > 0) ∧ (x ≤ 2) ∧ (x ≠ 1) ↔ ∀ x, (∃ y : ℝ, y = (1 / (Real.log x / Real.log 10) + Real.sqrt (2 - x))) :=
by
  sorry

end domain_of_function_l55_55053


namespace number_of_seats_in_classroom_l55_55237

theorem number_of_seats_in_classroom 
    (seats_per_row_condition : 7 + 13 = 19) 
    (rows_condition : 8 + 14 = 21) : 
    19 * 21 = 399 := 
by 
    sorry

end number_of_seats_in_classroom_l55_55237


namespace index_difference_l55_55108

theorem index_difference (n f m : ℕ) (h_n : n = 25) (h_f : f = 8) (h_m : m = 25 - 8) :
  (n - f) / n - (n - m) / n = 9 / 25 :=
by
  -- The proof is to be completed here.
  sorry

end index_difference_l55_55108


namespace pizza_store_total_sales_l55_55845

theorem pizza_store_total_sales (pepperoni bacon cheese : ℕ) (h1 : pepperoni = 2) (h2 : bacon = 6) (h3 : cheese = 6) :
  pepperoni + bacon + cheese = 14 :=
by sorry

end pizza_store_total_sales_l55_55845


namespace find_v_l55_55337

def B : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![1, 2], ![0, 1]]

def v : Matrix (Fin 2) (Fin 1) ℚ :=
  ![![3], ![1]]

def target : Matrix (Fin 2) (Fin 1) ℚ :=
  ![![15], ![5]]

theorem find_v :
  let B2 := B * B
  let B3 := B2 * B
  let B4 := B3 * B
  (B4 + B3 + B2 + B + (1 : Matrix (Fin 2) (Fin 2) ℚ)) * v = target :=
by
  sorry

end find_v_l55_55337


namespace zongzi_problem_l55_55610

def zongzi_prices : Prop :=
  ∀ (x y : ℕ), -- x: price of red bean zongzi, y: price of meat zongzi
  10 * x + 12 * y = 136 → -- total cost for the first customer
  y = 2 * x →
  x = 4 ∧ y = 8 -- prices found

def discounted_zongzi_prices : Prop :=
  ∀ (a b : ℕ), -- a: discounted price of red bean zongzi, b: discounted price of meat zongzi
  20 * a + 30 * b = 270 → -- cost for Xiaohuan's mother
  30 * a + 20 * b = 230 → -- cost for Xiaole's mother
  a = 3 ∧ b = 7 -- discounted prices found

def zongzi_packages (m : ℕ) : Prop :=
  ∀ (a b : ℕ), -- a: discounted price of red bean zongzi, b: discounted price of meat zongzi
  a = 3 → b = 7 →
  (80 - 4 * m) * (m * a + (40 - m) * b) + (4 * m + 8) * ((40 - m) * a + m * b) = 17280 →
  m ≤ 20 / 2 → -- quantity constraint
  m = 10 -- final m value

-- Statement to prove all together
theorem zongzi_problem :
  zongzi_prices ∧ discounted_zongzi_prices ∧ ∃ (m : ℕ), zongzi_packages m :=
by sorry

end zongzi_problem_l55_55610


namespace math_problem_l55_55880

variable (x y : ℝ)

theorem math_problem (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by sorry

end math_problem_l55_55880


namespace six_n_digit_remains_divisible_by_7_l55_55244

-- Given the conditions
def is_6n_digit_number (N : ℕ) (n : ℕ) : Prop :=
  N < 10^(6*n) ∧ N ≥ 10^(6*(n-1))

def is_divisible_by_7 (N : ℕ) : Prop :=
  N % 7 = 0

-- Define new number M formed by moving the unit digit to the beginning
def new_number (N : ℕ) (n : ℕ) : ℕ :=
  let a_0 := N % 10
  let rest := N / 10
  a_0 * 10^(6*n - 1) + rest

-- The theorem statement
theorem six_n_digit_remains_divisible_by_7 (N : ℕ) (n : ℕ)
  (hN : is_6n_digit_number N n)
  (hDiv7 : is_divisible_by_7 N) : is_divisible_by_7 (new_number N n) :=
sorry

end six_n_digit_remains_divisible_by_7_l55_55244


namespace machine_production_percentage_difference_l55_55941

theorem machine_production_percentage_difference 
  (X_production_rate : ℕ := 3)
  (widgets_to_produce : ℕ := 1080)
  (difference_in_hours : ℕ := 60) :
  ((widgets_to_produce / (widgets_to_produce / X_production_rate - difference_in_hours) - 
   X_production_rate) / X_production_rate * 100) = 20 := by
  sorry

end machine_production_percentage_difference_l55_55941


namespace pencils_per_box_l55_55957

theorem pencils_per_box:
  ∀ (red_pencils blue_pencils yellow_pencils green_pencils total_pencils num_boxes : ℕ),
  red_pencils = 20 →
  blue_pencils = 2 * red_pencils →
  yellow_pencils = 40 →
  green_pencils = red_pencils + blue_pencils →
  total_pencils = red_pencils + blue_pencils + yellow_pencils + green_pencils →
  num_boxes = 8 →
  total_pencils / num_boxes = 20 :=
by
  intros red_pencils blue_pencils yellow_pencils green_pencils total_pencils num_boxes
  intros h1 h2 h3 h4 h5 h6
  sorry

end pencils_per_box_l55_55957


namespace contrapositive_example_l55_55563

theorem contrapositive_example (x : ℝ) : (x > 1 → x^2 > 1) → (x^2 ≤ 1 → x ≤ 1) :=
sorry

end contrapositive_example_l55_55563


namespace largest_common_factor_462_330_l55_55062

-- Define the factors of 462
def factors_462 : Set ℕ := {1, 2, 3, 6, 7, 14, 21, 33, 42, 66, 77, 154, 231, 462}

-- Define the factors of 330
def factors_330 : Set ℕ := {1, 2, 3, 5, 6, 10, 11, 15, 30, 33, 55, 66, 110, 165, 330}

-- Define the statement of the theorem
theorem largest_common_factor_462_330 : 
  (∀ d : ℕ, d ∈ (factors_462 ∩ factors_330) → d ≤ 66) ∧
  66 ∈ (factors_462 ∩ factors_330) :=
sorry

end largest_common_factor_462_330_l55_55062


namespace simplify_expression_l55_55769

theorem simplify_expression :
  6^6 + 6^6 + 6^6 + 6^6 + 6^6 + 6^6 = 6^7 :=
by sorry

end simplify_expression_l55_55769


namespace tan_alpha_is_three_halves_l55_55869

theorem tan_alpha_is_three_halves (α : ℝ) (h : Real.tan (α - 5 * Real.pi / 4) = 1 / 5) : 
  Real.tan α = 3 / 2 :=
by
  sorry

end tan_alpha_is_three_halves_l55_55869


namespace combined_height_of_trees_l55_55111

noncomputable def growth_rate_A (weeks : ℝ) : ℝ := (weeks / 2) * 50
noncomputable def growth_rate_B (weeks : ℝ) : ℝ := (weeks / 3) * 70
noncomputable def growth_rate_C (weeks : ℝ) : ℝ := (weeks / 4) * 90
noncomputable def initial_height_A : ℝ := 200
noncomputable def initial_height_B : ℝ := 150
noncomputable def initial_height_C : ℝ := 250
noncomputable def total_weeks : ℝ := 16
noncomputable def total_growth_A := growth_rate_A total_weeks
noncomputable def total_growth_B := growth_rate_B total_weeks
noncomputable def total_growth_C := growth_rate_C total_weeks
noncomputable def final_height_A := initial_height_A + total_growth_A
noncomputable def final_height_B := initial_height_B + total_growth_B
noncomputable def final_height_C := initial_height_C + total_growth_C
noncomputable def final_combined_height := final_height_A + final_height_B + final_height_C

theorem combined_height_of_trees :
  final_combined_height = 1733.33 := by
  sorry

end combined_height_of_trees_l55_55111


namespace find_dividend_l55_55739

theorem find_dividend (R D Q V : ℤ) (hR : R = 5) (hD1 : D = 3 * Q) (hD2 : D = 3 * R + 3) : V = D * Q + R → V = 113 :=
by 
  sorry

end find_dividend_l55_55739


namespace value_of_m_l55_55735

theorem value_of_m 
    (x : ℝ) (m : ℝ) 
    (h : 0 < x)
    (h_eq : (2 / (x - 2)) - ((2 * x - m) / (2 - x)) = 3) : 
    m = 6 := 
sorry

end value_of_m_l55_55735


namespace sandy_marbles_correct_l55_55210

namespace MarbleProblem

-- Define the number of dozens Jessica has
def jessica_dozens : ℕ := 3

-- Define the conversion from dozens to individual marbles
def dozens_to_marbles (d : ℕ) : ℕ := 12 * d

-- Calculate the number of marbles Jessica has
def jessica_marbles : ℕ := dozens_to_marbles jessica_dozens

-- Define the multiplier for Sandy's marbles
def sandy_multiplier : ℕ := 4

-- Define the number of marbles Sandy has
def sandy_marbles : ℕ := sandy_multiplier * jessica_marbles

theorem sandy_marbles_correct : sandy_marbles = 144 :=
by
  sorry

end MarbleProblem

end sandy_marbles_correct_l55_55210


namespace difference_in_surface_areas_l55_55019

-- Define the conditions: volumes and number of cubes
def V_large : ℕ := 343
def n : ℕ := 343
def V_small : ℕ := 1

-- Define the function to calculate the side length of a cube given its volume
def side_length (V : ℕ) : ℕ := V^(1/3 : ℕ)

-- Specify the side lengths of the larger and smaller cubes
def s_large : ℕ := side_length V_large
def s_small : ℕ := side_length V_small

-- Define the function to calculate the surface area of a cube given its side length
def surface_area (s : ℕ) : ℕ := 6 * s^2

-- Specify the surface areas of the larger cube and the total of the smaller cubes
def SA_large : ℕ := surface_area s_large
def SA_small_total : ℕ := n * surface_area s_small

-- State the theorem to prove
theorem difference_in_surface_areas : SA_small_total - SA_large = 1764 :=
by {
  -- Intentionally omit proof, as per instructions
  sorry
}

end difference_in_surface_areas_l55_55019


namespace average_minutes_run_l55_55017

-- Definitions
def third_graders (fi : ℕ) : ℕ := 6 * fi
def fourth_graders (fi : ℕ) : ℕ := 2 * fi
def fifth_graders (fi : ℕ) : ℕ := fi

-- Number of minutes run by each grade
def third_graders_minutes : ℕ := 10
def fourth_graders_minutes : ℕ := 18
def fifth_graders_minutes : ℕ := 8

-- Main theorem
theorem average_minutes_run 
  (fi : ℕ) 
  (t := third_graders fi) 
  (fr := fourth_graders fi) 
  (f := fifth_graders fi) 
  (minutes_total := 10 * t + 18 * fr + 8 * f) 
  (students_total := t + fr + f) :
  (students_total > 0) →
  (minutes_total : ℚ) / students_total = 104 / 9 :=
by
  sorry

end average_minutes_run_l55_55017


namespace find_B_value_l55_55463

theorem find_B_value (A B : ℕ) : (A * 100 + B * 10 + 2) - 41 = 591 → B = 3 :=
by
  sorry

end find_B_value_l55_55463


namespace mixed_number_evaluation_l55_55585

theorem mixed_number_evaluation :
  let a := (4 + 1 / 3 : ℚ)
  let b := (3 + 2 / 7 : ℚ)
  let c := (2 + 5 / 6 : ℚ)
  let d := (1 + 1 / 2 : ℚ)
  let e := (5 + 1 / 4 : ℚ)
  let f := (3 + 2 / 5 : ℚ)
  (a + b - c) * (d + e) / f = 9 + 198 / 317 :=
by {
  let a : ℚ := 4 + 1 / 3
  let b : ℚ := 3 + 2 / 7
  let c : ℚ := 2 + 5 / 6
  let d : ℚ := 1 + 1 / 2
  let e : ℚ := 5 + 1 / 4
  let f : ℚ := 3 + 2 / 5
  sorry
}

end mixed_number_evaluation_l55_55585


namespace smallest_denominator_of_sum_of_irreducible_fractions_l55_55390

theorem smallest_denominator_of_sum_of_irreducible_fractions :
  ∀ (a b : ℕ),
  Nat.Coprime a 600 → Nat.Coprime b 700 →
  (∃ c d : ℕ, Nat.Coprime c d ∧ d < 168 ∧ (7 * a + 6 * b) / Nat.gcd (7 * a + 6 * b) 4200 = c / d) →
  False :=
by
  sorry

end smallest_denominator_of_sum_of_irreducible_fractions_l55_55390


namespace find_original_amount_l55_55930

-- Let X be the original amount of money in Christina's account.
variable (X : ℝ)

-- Condition 1: Remaining balance after transferring 20% is $30,000.
def initial_transfer (X : ℝ) : Prop :=
  0.80 * X = 30000

-- Prove that the original amount before the initial transfer was $37,500.
theorem find_original_amount (h : initial_transfer X) : X = 37500 :=
  sorry

end find_original_amount_l55_55930


namespace n_minus_m_eq_zero_l55_55009

-- Definitions based on the conditions
def m : ℝ := sorry
def n : ℝ := sorry
def i := Complex.I
def condition : Prop := m + i = (1 + 2 * i) - n * i

-- The theorem stating the equivalence proof problem
theorem n_minus_m_eq_zero (h : condition) : n - m = 0 :=
sorry

end n_minus_m_eq_zero_l55_55009


namespace total_value_of_item_l55_55726

theorem total_value_of_item (V : ℝ) (h1 : 0.07 * (V - 1000) = 87.50) :
  V = 2250 :=
by
  sorry

end total_value_of_item_l55_55726


namespace correct_choice_l55_55081

variable (a b : ℝ) (p q : Prop) (x : ℝ)

-- Proposition A: Incorrect because x > 3 is a sufficient condition for x > 2.
def propositionA : Prop := (∀ x : ℝ, x > 3 → x > 2) ∧ ¬ (∀ x : ℝ, x > 2 → x > 3)

-- Proposition B: Incorrect negation form.
def propositionB : Prop := ¬ (¬p → ¬q) ∧ (q → p)

-- Proposition C: Incorrect because it should be 1/a > 1/b given 0 < a < b.
def propositionC : Prop := (a > 0 ∧ b < 0) ∧ ¬ (1/a < 1/b)

-- Proposition D: Correct negation form.
def propositionD_negation_correct : Prop := 
  (¬ ∃ x : ℝ, x^2 = 1) = ( ∀ x : ℝ, x^2 ≠ 1)

theorem correct_choice : propositionD_negation_correct := by
  sorry

end correct_choice_l55_55081


namespace exterior_angle_measure_l55_55056

theorem exterior_angle_measure (sum_interior_angles : ℝ) (h : sum_interior_angles = 1260) :
  ∃ (n : ℕ) (d : ℝ), (n - 2) * 180 = sum_interior_angles ∧ d = 360 / n ∧ d = 40 := 
by
  sorry

end exterior_angle_measure_l55_55056


namespace daily_harvest_sacks_l55_55963

theorem daily_harvest_sacks (sacks_per_section : ℕ) (num_sections : ℕ) (total_sacks : ℕ) :
  sacks_per_section = 65 → num_sections = 12 → total_sacks = sacks_per_section * num_sections → total_sacks = 780 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end daily_harvest_sacks_l55_55963


namespace number_of_valid_three_digit_numbers_l55_55044

def three_digit_numbers_count : Nat :=
  let count_numbers (last_digit : Nat) (remaining_digits : List Nat) : Nat :=
    remaining_digits.length * (remaining_digits.erase last_digit).length

  let count_when_last_digit_is_0 :=
    count_numbers 0 [1, 2, 3, 4, 5, 6, 7, 8, 9]

  let count_when_last_digit_is_5 :=
    count_numbers 5 [0, 1, 2, 3, 4, 6, 7, 8, 9]

  count_when_last_digit_is_0 + count_when_last_digit_is_5

theorem number_of_valid_three_digit_numbers : three_digit_numbers_count = 136 := by
  sorry

end number_of_valid_three_digit_numbers_l55_55044


namespace find_cost_of_fourth_cd_l55_55079

variables (cost1 cost2 cost3 cost4 : ℕ)
variables (h1 : (cost1 + cost2 + cost3) / 3 = 15)
variables (h2 : (cost1 + cost2 + cost3 + cost4) / 4 = 16)

theorem find_cost_of_fourth_cd : cost4 = 19 := 
by 
  sorry

end find_cost_of_fourth_cd_l55_55079


namespace alpha_bound_l55_55787

theorem alpha_bound (α : ℝ) (x : ℕ → ℝ) (h_x_inc : ∀ n, x n < x (n + 1))
    (x0_one : x 0 = 1) (h_alpha : α = ∑' n, x (n + 1) / (x n)^3) :
    α ≥ 3 * Real.sqrt 3 / 2 := 
sorry

end alpha_bound_l55_55787


namespace price_of_36kgs_l55_55492

namespace Apples

-- Define the parameters l and q
variables (l q : ℕ)

-- Define the conditions
def cost_first_30kgs (l : ℕ) : ℕ := 30 * l
def cost_first_15kgs : ℕ := 150
def cost_33kgs (l q : ℕ) : ℕ := (30 * l) + (3 * q)
def cost_36kgs (l q : ℕ) : ℕ := (30 * l) + (6 * q)

-- Define the hypothesis for l and q based on given conditions
axiom l_value (h1 : cost_first_15kgs = 150) : l = 10
axiom q_value (h2 : cost_33kgs l q = 333) : q = 11

-- Prove the price of 36 kilograms of apples
theorem price_of_36kgs (h1 : cost_first_15kgs = 150) (h2 : cost_33kgs l q = 333) : cost_36kgs l q = 366 :=
sorry

end Apples

end price_of_36kgs_l55_55492


namespace turtle_hare_race_headstart_l55_55255

noncomputable def hare_time_muddy (distance speed_reduction hare_speed : ℝ) : ℝ :=
  distance / (hare_speed * speed_reduction)

noncomputable def hare_time_sandy (distance hare_speed : ℝ) : ℝ :=
  distance / hare_speed

noncomputable def hare_time_regular (distance hare_speed : ℝ) : ℝ :=
  distance / hare_speed

noncomputable def turtle_time_muddy (distance turtle_speed : ℝ) : ℝ :=
  distance / turtle_speed

noncomputable def turtle_time_sandy (distance speed_increase turtle_speed : ℝ) : ℝ :=
  distance / (turtle_speed * speed_increase)

noncomputable def turtle_time_regular (distance turtle_speed : ℝ) : ℝ :=
  distance / turtle_speed

noncomputable def hare_total_time (hare_speed : ℝ) : ℝ :=
  hare_time_muddy 20 0.5 hare_speed + hare_time_sandy 10 hare_speed + hare_time_regular 20 hare_speed

noncomputable def turtle_total_time (turtle_speed : ℝ) : ℝ :=
  turtle_time_muddy 20 turtle_speed + turtle_time_sandy 10 1.5 turtle_speed + turtle_time_regular 20 turtle_speed

theorem turtle_hare_race_headstart (hare_speed turtle_speed : ℝ) (t_hs : ℝ) :
  hare_speed = 10 →
  turtle_speed = 1 →
  t_hs = 39.67 →
  hare_total_time hare_speed + t_hs = turtle_total_time turtle_speed :=
by
  intros 
  sorry

end turtle_hare_race_headstart_l55_55255


namespace pencil_notebook_cost_l55_55670

variable {p n : ℝ}

theorem pencil_notebook_cost (hp1 : 9 * p + 11 * n = 6.05) (hp2 : 6 * p + 4 * n = 2.68) :
  18 * p + 13 * n = 8.45 :=
sorry

end pencil_notebook_cost_l55_55670


namespace charity_total_cost_l55_55107

theorem charity_total_cost
  (plates : ℕ)
  (rice_cost_per_plate chicken_cost_per_plate : ℕ)
  (h1 : plates = 100)
  (h2 : rice_cost_per_plate = 10)
  (h3 : chicken_cost_per_plate = 40) :
  plates * (rice_cost_per_plate + chicken_cost_per_plate) / 100 = 50 := 
by
  sorry

end charity_total_cost_l55_55107


namespace carpet_needed_for_room_l55_55724

theorem carpet_needed_for_room
  (length_feet : ℕ) (width_feet : ℕ)
  (area_conversion_factor : ℕ)
  (length_given : length_feet = 12)
  (width_given : width_feet = 6)
  (conversion_given : area_conversion_factor = 9) :
  (length_feet * width_feet) / area_conversion_factor = 8 := 
by
  sorry

end carpet_needed_for_room_l55_55724


namespace trigonometric_identity_l55_55187

theorem trigonometric_identity
  (α : ℝ) 
  (h : Real.tan α = -1 / 2) :
  (Real.cos α - Real.sin α)^2 / Real.cos (2 * α) = 3 := 
by 
  sorry

end trigonometric_identity_l55_55187


namespace cube_surface_area_of_same_volume_as_prism_l55_55479

theorem cube_surface_area_of_same_volume_as_prism :
  let prism_length := 10
  let prism_width := 5
  let prism_height := 24
  let prism_volume := prism_length * prism_width * prism_height
  let cube_edge := (prism_volume : ℝ)^(1/3)
  let cube_surface_area := 6 * cube_edge^2
  cube_surface_area = 677.76 := by
  sorry

end cube_surface_area_of_same_volume_as_prism_l55_55479


namespace quadratic_roots_l55_55556

theorem quadratic_roots (m x1 x2 : ℝ) (h1 : x1 + x2 = 1) (h2 : x1*x1 + m*x1 + 2*m = 0) (h3 : x2*x2 + m*x2 + 2*m = 0) : x1 * x2 = -2 := 
by sorry

end quadratic_roots_l55_55556


namespace poly_coefficients_sum_l55_55962

theorem poly_coefficients_sum :
  ∀ (x A B C D : ℝ),
  (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D →
  A + B + C + D = 2 :=
by sorry

end poly_coefficients_sum_l55_55962


namespace transformed_polynomial_l55_55666

theorem transformed_polynomial (x y : ℝ) (h : y = x + 1 / x) :
  (x^4 - 2*x^3 - 3*x^2 + 2*x + 1 = 0) → (x^2 * (y^2 - y - 3) = 0) :=
by
  sorry

end transformed_polynomial_l55_55666


namespace reflect_point_l55_55809

def point_reflect_across_line (m : ℝ) :=
  (6 - m, m + 1)

theorem reflect_point (m : ℝ) :
  point_reflect_across_line m = (6 - m, m + 1) :=
  sorry

end reflect_point_l55_55809


namespace avg_of_consecutive_starting_with_b_l55_55830

variable {a : ℕ} (h : b = (a + 1 + a + 2 + a + 3 + a + 4 + a + 5 + a + 6 + a + 7) / 7)

theorem avg_of_consecutive_starting_with_b (h : b = (a + 1 + a + 2 + a + 3 + a + 4 + a + 5 + a + 6 + a + 7) / 7) :
  (a + 4 + (a + 4 + 1) + (a + 4 + 2) + (a + 4 + 3) + (a + 4 + 4) + (a + 4 + 5) + (a + 4 + 6)) / 7 = a + 7 :=
  sorry

end avg_of_consecutive_starting_with_b_l55_55830


namespace decorations_given_to_friend_l55_55622

-- Definitions of the given conditions
def boxes : ℕ := 6
def decorations_per_box : ℕ := 25
def used_decorations : ℕ := 58
def neighbor_decorations : ℕ := 75

-- The statement of the proof problem
theorem decorations_given_to_friend : 
  (boxes * decorations_per_box) - used_decorations - neighbor_decorations = 17 := 
by 
  sorry

end decorations_given_to_friend_l55_55622


namespace value_of_a_plus_d_l55_55591

variable (a b c d : ℝ)

theorem value_of_a_plus_d
  (h1 : a + b = 4)
  (h2 : b + c = 5)
  (h3 : c + d = 3) :
  a + d = 1 :=
by
sorry

end value_of_a_plus_d_l55_55591


namespace no_integer_n_such_that_squares_l55_55322

theorem no_integer_n_such_that_squares :
  ¬ ∃ n : ℤ, (∃ k1 : ℤ, 10 * n - 1 = k1 ^ 2) ∧
             (∃ k2 : ℤ, 13 * n - 1 = k2 ^ 2) ∧
             (∃ k3 : ℤ, 85 * n - 1 = k3 ^ 2) := 
by sorry

end no_integer_n_such_that_squares_l55_55322


namespace two_digit_product_l55_55092

theorem two_digit_product (x y : ℕ) (h₁ : 10 ≤ x) (h₂ : x < 100) (h₃ : 10 ≤ y) (h₄ : y < 100) (h₅ : x * y = 4320) :
  (x = 60 ∧ y = 72) ∨ (x = 72 ∧ y = 60) :=
sorry

end two_digit_product_l55_55092


namespace one_intersection_point_two_intersection_points_l55_55444

variables (k : ℝ)

-- Condition definitions
def parabola_eq (y x : ℝ) : Prop := y^2 = -4 * x
def line_eq (x y k : ℝ) : Prop := y + 1 = k * (x - 2)
def discriminant_non_negative (a b c : ℝ) : Prop := b^2 - 4 * a * c ≥ 0

-- Mathematically equivalent proof problem 1
theorem one_intersection_point (k : ℝ) : 
  (k = 1/2 ∨ k = -1 ∨ k = 0) → 
  ∃ x y : ℝ, parabola_eq y x ∧ line_eq x y k := sorry

-- Mathematically equivalent proof problem 2
theorem two_intersection_points (k : ℝ) : 
  (-1 < k ∧ k < 1/2 ∧ k ≠ 0) → 
  ∃ x₁ y₁ x₂ y₂ : ℝ, 
  (x₁ ≠ x₂ ∧ y₁ ≠ y₂) ∧ parabola_eq y₁ x₁ ∧ parabola_eq y₂ x₂ ∧ 
  line_eq x₁ y₁ k ∧ line_eq x₂ y₂ k := sorry

end one_intersection_point_two_intersection_points_l55_55444


namespace find_b_squared_l55_55433

-- Assume a and b are real numbers and positive
variables (a b : ℝ)
-- Given conditions
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom magnitude : a^2 + b^2 = 100
axiom equidistant : 2 * a - 4 * b = 7

-- Main proof statement
theorem find_b_squared : b^2 = 287 / 17 := sorry

end find_b_squared_l55_55433


namespace smallest_area_of_2020th_square_l55_55956

theorem smallest_area_of_2020th_square :
  ∃ (S : ℤ) (A : ℕ), 
    (S * S - 2019 = A) ∧ 
    (∃ k : ℕ, k * k = A) ∧ 
    (∀ (T : ℤ) (B : ℕ), ((T * T - 2019 = B) ∧ (∃ l : ℕ, l * l = B)) → (A ≤ B)) :=
sorry

end smallest_area_of_2020th_square_l55_55956


namespace vector_equation_proof_l55_55112

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C P : V)

/-- The given condition. -/
def given_condition : Prop :=
  (P - A) + 2 • (P - B) + 3 • (P - C) = 0

/-- The target equality we want to prove. -/
theorem vector_equation_proof (h : given_condition A B C P) :
  P - A = (1 / 3 : ℝ) • (B - A) + (1 / 2 : ℝ) • (C - A) :=
sorry

end vector_equation_proof_l55_55112


namespace cubic_equation_solution_bound_l55_55206

theorem cubic_equation_solution_bound (a : ℝ) :
  a ∈ Set.Ici (-15) → ∀ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ → x₂ ≠ x₃ → x₁ ≠ x₃ →
  (x₁^3 + 6 * x₁^2 + a * x₁ + 8 = 0) →
  (x₂^3 + 6 * x₂^2 + a * x₂ + 8 = 0) →
  (x₃^3 + 6 * x₃^2 + a * x₃ + 8 = 0) →
  False := 
sorry

end cubic_equation_solution_bound_l55_55206


namespace length_of_nylon_cord_l55_55118

-- Definitions based on the conditions
def tree : ℝ := 0 -- Tree as the center point (assuming a 0 for simplicity)
def distance_ran : ℝ := 30 -- Dog ran approximately 30 feet

-- The theorem to prove
theorem length_of_nylon_cord : (distance_ran / 2) = 15 := by
  -- Assuming the dog ran along the diameter of the circle
  -- and the length of the cord is the radius of that circle.
  sorry

end length_of_nylon_cord_l55_55118


namespace probability_of_region_D_l55_55783

theorem probability_of_region_D
    (P_A : ℚ) (P_B : ℚ) (P_C : ℚ) (P_D : ℚ)
    (h1 : P_A = 1/4) 
    (h2 : P_B = 1/3) 
    (h3 : P_C = 1/6) 
    (h4 : P_A + P_B + P_C + P_D = 1) : 
    P_D = 1/4 := by
    sorry

end probability_of_region_D_l55_55783


namespace minimum_fence_length_l55_55472

theorem minimum_fence_length {x y : ℝ} (hxy : x * y = 100) : 2 * (x + y) ≥ 40 :=
by
  sorry

end minimum_fence_length_l55_55472


namespace functional_eq_solution_l55_55801

noncomputable def f : ℚ → ℚ := sorry

theorem functional_eq_solution (f : ℚ → ℚ)
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1):
  ∀ x : ℚ, f x = x + 1 :=
sorry

end functional_eq_solution_l55_55801


namespace points_on_opposite_sides_of_line_l55_55058

theorem points_on_opposite_sides_of_line (a : ℝ) :
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 ↔ -7 < a ∧ a < 24 :=
by 
  sorry

end points_on_opposite_sides_of_line_l55_55058


namespace crackers_given_to_friends_l55_55417

theorem crackers_given_to_friends (crackers_per_friend : ℕ) (number_of_friends : ℕ) (h1 : crackers_per_friend = 6) (h2 : number_of_friends = 6) : (crackers_per_friend * number_of_friends) = 36 :=
by
  sorry

end crackers_given_to_friends_l55_55417


namespace quadrilateral_circumscribed_l55_55373

structure ConvexQuad (A B C D : Type) := 
  (is_convex : True)
  (P : Type)
  (interior : True)
  (angle_APB_angle_CPD_eq_angle_BPC_angle_DPA : True)
  (angle_PAD_angle_PCD_eq_angle_PAB_angle_PCB : True)
  (angle_PDC_angle_PBC_eq_angle_PDA_angle_PBA : True)

theorem quadrilateral_circumscribed (A B C D : Type) (quad : ConvexQuad A B C D) : True := 
sorry

end quadrilateral_circumscribed_l55_55373


namespace certain_event_proof_l55_55637

def Moonlight_in_front_of_bed := "depends_on_time_and_moon_position"
def Lonely_smoke_in_desert := "depends_on_specific_conditions"
def Reach_for_stars_with_hand := "physically_impossible"
def Yellow_River_flows_into_sea := "certain_event"

theorem certain_event_proof : Yellow_River_flows_into_sea = "certain_event" :=
by
  sorry

end certain_event_proof_l55_55637


namespace distance_from_neg6_to_origin_l55_55068

theorem distance_from_neg6_to_origin :
  abs (-6) = 6 :=
by
  sorry

end distance_from_neg6_to_origin_l55_55068


namespace probability_4_students_same_vehicle_l55_55209

-- Define the number of vehicles
def num_vehicles : ℕ := 3

-- Define the probability that 4 students choose the same vehicle
def probability_same_vehicle (n : ℕ) : ℚ :=
  3 / (3^(n : ℤ))

-- Prove that the probability for 4 students is 1/27
theorem probability_4_students_same_vehicle : probability_same_vehicle 4 = 1 / 27 := 
  sorry

end probability_4_students_same_vehicle_l55_55209


namespace percent_defective_shipped_l55_55220

theorem percent_defective_shipped
  (P_d : ℝ) (P_s : ℝ)
  (hP_d : P_d = 0.1)
  (hP_s : P_s = 0.05) :
  P_d * P_s = 0.005 :=
by
  sorry

end percent_defective_shipped_l55_55220


namespace foil_covered_prism_width_l55_55224

def inner_prism_length (l : ℝ) := l
def inner_prism_width (l : ℝ) := 2 * l
def inner_prism_height (l : ℝ) := l
def inner_prism_volume (l : ℝ) := l * (2 * l) * l

theorem foil_covered_prism_width :
  (∃ l : ℝ, inner_prism_volume l = 128) → (inner_prism_width l + 2 = 8) := by
sorry

end foil_covered_prism_width_l55_55224


namespace min_abs_y1_minus_4y2_l55_55993

-- Definitions based on conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : (ℝ × ℝ) := (1, 0)

noncomputable def equation_of_line (k y : ℝ) : ℝ := k * y + 1

-- The Lean theorem statement
theorem min_abs_y1_minus_4y2 {x1 y1 x2 y2 : ℝ} (H1 : parabola x1 y1) (H2 : parabola x2 y2)
    (A_in_first_quadrant : 0 < x1 ∧ 0 < y1)
    (line_through_focus : ∃ k : ℝ, x1 = equation_of_line k y1 ∧ x2 = equation_of_line k y2)
    : |y1 - 4 * y2| = 8 :=
sorry

end min_abs_y1_minus_4y2_l55_55993


namespace fill_tank_time_l55_55511

theorem fill_tank_time :
  ∀ (rate_fill rate_empty : ℝ), 
    rate_fill = 1 / 25 → 
    rate_empty = 1 / 50 → 
    (1/2) / (rate_fill - rate_empty) = 25 :=
by
  intros rate_fill rate_empty h_fill h_empty
  sorry

end fill_tank_time_l55_55511


namespace speed_in_still_water_l55_55922

-- We define the given conditions for the man's rowing speeds
def upstream_speed : ℕ := 25
def downstream_speed : ℕ := 35

-- We want to prove that the speed in still water is 30 kmph
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 30 := by
  sorry

end speed_in_still_water_l55_55922


namespace functional_equation_solution_l55_55217

noncomputable def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (f 0 ≠ 0) ∧ (∀ x y : ℝ, f (x + y) * f (x + y) = 2 * f x * f y + max (f (x * x) + f (y * y)) (f (x * x + y * y)))

theorem functional_equation_solution (f : ℝ → ℝ) :
  satisfies_conditions f → (∀ x : ℝ, f x = -1 ∨ f x = x - 1) :=
by
  intros h
  sorry

end functional_equation_solution_l55_55217


namespace max_unbounded_xy_sum_l55_55175

theorem max_unbounded_xy_sum (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  ∃ M : ℝ, ∀ z : ℝ, z > 0 → ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ (xy + 1)^2 + (x - y)^2 > z := 
  sorry

end max_unbounded_xy_sum_l55_55175


namespace cannot_form_set_of_good_friends_of_wang_ming_l55_55127

def is_well_defined_set (description : String) : Prop := sorry  -- Placeholder for the formal definition.

theorem cannot_form_set_of_good_friends_of_wang_ming :
  ¬ is_well_defined_set "Good friends of Wang Ming" :=
sorry

end cannot_form_set_of_good_friends_of_wang_ming_l55_55127


namespace sequence_arithmetic_condition_l55_55807

theorem sequence_arithmetic_condition {α β : ℝ} (hα : α ≠ 0) (hβ : β ≠ 0) (hαβ : α + β ≠ 0)
  (seq : ℕ → ℝ) (hseq : ∀ n, seq (n + 2) = (α * seq (n + 1) + β * seq n) / (α + β)) :
  ∃ α β : ℝ, (∀ a1 a2 : ℝ, α ≠ 0 ∧ β ≠ 0 ∧ α + β = 0 → seq (n + 1) - seq n = seq n - seq (n - 1)) :=
by sorry

end sequence_arithmetic_condition_l55_55807


namespace find_a_l55_55718

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then x ^ 2 + a * Real.cos (Real.pi * x) else 2

theorem find_a (a : ℝ) :
  (∀ x, f (-x) a = -f x a) → f 1 a = 2 → a = - 3 :=
by
  sorry

end find_a_l55_55718


namespace find_k_for_one_real_solution_l55_55607

theorem find_k_for_one_real_solution (k : ℤ) :
  (∀ x : ℤ, (x - 3) * (x + 2) = k + 3 * x) ↔ k = -10 := by
  sorry

end find_k_for_one_real_solution_l55_55607


namespace relationship_among_three_numbers_l55_55512

noncomputable def M (a b : ℝ) : ℝ := a^b
noncomputable def N (a b : ℝ) : ℝ := Real.log a / Real.log b
noncomputable def P (a b : ℝ) : ℝ := b^a

theorem relationship_among_three_numbers (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) : N a b < M a b ∧ M a b < P a b := 
by
  sorry

end relationship_among_three_numbers_l55_55512


namespace problem1_part1_problem1_part2_l55_55884

theorem problem1_part1 : (3 - Real.pi)^0 - 2 * Real.cos (Real.pi / 6) + abs (1 - Real.sqrt 3) + (1 / 2)⁻¹ = 2 := by
  sorry

theorem problem1_part2 {x : ℝ} : x^2 - 2 * x - 9 = 0 -> (x = 1 + Real.sqrt 10 ∨ x = 1 - Real.sqrt 10) := by
  sorry

end problem1_part1_problem1_part2_l55_55884


namespace width_of_lawn_is_60_l55_55858

-- Define the problem conditions in Lean
def length_of_lawn : ℕ := 70
def road_width : ℕ := 10
def total_road_cost : ℕ := 3600
def cost_per_sq_meter : ℕ := 3

-- Define the proof problem
theorem width_of_lawn_is_60 (W : ℕ) 
  (h1 : (road_width * W) + (road_width * length_of_lawn) - (road_width * road_width) 
        = total_road_cost / cost_per_sq_meter) : 
  W = 60 := 
by 
  sorry

end width_of_lawn_is_60_l55_55858


namespace inequality_proof_l55_55885

theorem inequality_proof
  (x y : ℝ)
  (h : x^8 + y^8 ≤ 1) :
  x^12 - y^12 + 2 * x^6 * y^6 ≤ π / 2 :=
sorry

end inequality_proof_l55_55885


namespace unique_solution_p_eq_neg8_l55_55459

theorem unique_solution_p_eq_neg8 (p : ℝ) (h : ∀ y : ℝ, 2 * y^2 - 8 * y - p = 0 → ∃! y : ℝ, 2 * y^2 - 8 * y - p = 0) : p = -8 :=
sorry

end unique_solution_p_eq_neg8_l55_55459


namespace find_x_complementary_l55_55638

-- Define the conditions.
def are_complementary (a b : ℝ) : Prop := a + b = 90

-- The main theorem statement with the condition and conclusion.
theorem find_x_complementary : ∀ x : ℝ, are_complementary (2*x) (3*x) → x = 18 := 
by
  intros x h
  -- sorry is a placeholder for the proof.
  sorry

end find_x_complementary_l55_55638


namespace product_bases_l55_55342

def base2_to_nat (s : String) : Nat :=
  s.foldl (λ acc d => acc * 2 + (d.toNat - '0'.toNat)) 0

def base3_to_nat (s : String) : Nat :=
  s.foldl (λ acc d => acc * 3 + (d.toNat - '0'.toNat)) 0

def base4_to_nat (s : String) : Nat :=
  s.foldl (λ acc d => acc * 4 + (d.toNat - '0'.toNat)) 0

theorem product_bases :
  base2_to_nat "1101" * base3_to_nat "202" * base4_to_nat "22" = 2600 :=
by
  sorry

end product_bases_l55_55342


namespace algebraic_fraction_l55_55415

theorem algebraic_fraction (x : ℝ) (h1 : 1 / 3 = 1 / 3) 
(h2 : x / Real.pi = x / Real.pi) 
(h3 : 2 / (x + 3) = 2 / (x + 3))
(h4 : (x + 2) / 3 = (x + 2) / 3) 
: 
2 / (x + 3) = 2 / (x + 3) := sorry

end algebraic_fraction_l55_55415


namespace problem_1110_1111_1112_1113_l55_55811

theorem problem_1110_1111_1112_1113 (r : ℕ) (hr : r > 5) : 
  (r^3 + r^2 + r) * (r^3 + r^2 + r + 1) * (r^3 + r^2 + r + 2) * (r^3 + r^2 + r + 3) = (r^6 + 2 * r^5 + 3 * r^4 + 5 * r^3 + 4 * r^2 + 3 * r + 1)^2 - 1 :=
by
  sorry

end problem_1110_1111_1112_1113_l55_55811


namespace tangent_line_of_circle_l55_55852

theorem tangent_line_of_circle (x y : ℝ)
    (C_def : (x - 2)^2 + (y - 3)^2 = 25)
    (P : (ℝ × ℝ)) (P_def : P = (-1, 7)) :
    (3 * x - 4 * y + 31 = 0) :=
sorry

end tangent_line_of_circle_l55_55852


namespace John_needs_more_days_l55_55388

theorem John_needs_more_days (days_worked : ℕ) (amount_earned : ℕ) :
  days_worked = 10 ∧ amount_earned = 250 ∧ 
  (∀ d : ℕ, d < days_worked → amount_earned / days_worked = amount_earned / 10) →
  ∃ more_days : ℕ, more_days = 10 ∧ amount_earned * 2 = (days_worked + more_days) * (amount_earned / days_worked) :=
sorry

end John_needs_more_days_l55_55388


namespace prime_sum_product_l55_55101

theorem prime_sum_product (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 101) : p * q = 194 :=
sorry

end prime_sum_product_l55_55101


namespace num_multiples_of_three_in_ap_l55_55117

variable (a : ℕ → ℚ)  -- Defining the arithmetic sequence

def first_term (a1 : ℚ) := a 1 = a1
def eighth_term (a8 : ℚ) := a 8 = a8
def general_term (d : ℚ) := ∀ n : ℕ, a n = 9 + (n - 1) * d
def multiple_of_three (n : ℕ) := ∃ k : ℕ, a n = 3 * k

theorem num_multiples_of_three_in_ap 
  (a : ℕ → ℚ)
  (h1 : first_term a 9)
  (h2 : eighth_term a 12) :
  ∃ n : ℕ, n = 288 ∧ ∃ l : ℕ → Prop, ∀ k : ℕ, l k → multiple_of_three a (k * 7 + 1) :=
sorry

end num_multiples_of_three_in_ap_l55_55117


namespace proof_problem_l55_55486

noncomputable def a : ℝ := 2 - 0.5
noncomputable def b : ℝ := Real.log (Real.pi) / Real.log 3
noncomputable def c : ℝ := Real.log 2 / Real.log 4

theorem proof_problem : b > a ∧ a > c := 
by
sorry

end proof_problem_l55_55486


namespace solution_set_inequality_l55_55437

theorem solution_set_inequality (a : ℝ) (x : ℝ) (h : 0 < a ∧ a < 1) : 
  ((a - x) * (x - 1 / a) > 0) ↔ (a < x ∧ x < 1 / a) :=
by
  sorry

end solution_set_inequality_l55_55437


namespace complement_A_inter_B_l55_55979

def A : Set ℝ := {x | abs (x - 2) ≤ 2}

def B : Set ℝ := {y | ∃ x, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2}

def A_inter_B : Set ℝ := A ∩ B

def C_R (s : Set ℝ) : Set ℝ := {x | x ∉ s}

theorem complement_A_inter_B :
  C_R A_inter_B = {x | x < 0} ∪ {x | x > 0} :=
by
  sorry

end complement_A_inter_B_l55_55979


namespace angle_Y_measure_l55_55777

def hexagon_interior_angle_sum (n : ℕ) : ℕ :=
  180 * (n - 2)

def supplementary (α β : ℕ) : Prop :=
  α + β = 180

def equal_angles (α β γ δ : ℕ) : Prop :=
  α = β ∧ β = γ ∧ γ = δ

theorem angle_Y_measure :
  ∀ (C H E S1 S2 Y : ℕ),
    C = E ∧ E = S1 ∧ S1 = Y →
    supplementary H S2 →
    hexagon_interior_angle_sum 6 = C + H + E + S1 + S2 + Y →
    Y = 135 :=
by
  intros C H E S1 S2 Y h1 h2 h3
  sorry

end angle_Y_measure_l55_55777


namespace rabbit_probability_l55_55860

def cube_vertices : ℕ := 8
def cube_edges : ℕ := 12
def moves : ℕ := 11
def paths_after_11_moves : ℕ := 3 ^ moves
def favorable_paths : ℕ := 24

theorem rabbit_probability :
  (favorable_paths : ℚ) / paths_after_11_moves = 24 / 177147 := by
  sorry

end rabbit_probability_l55_55860


namespace selene_total_payment_l55_55754

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

end selene_total_payment_l55_55754


namespace common_difference_is_3_l55_55198

noncomputable def whale_plankton_frenzy (x : ℝ) (y : ℝ) : Prop :=
  (9 * x + 36 * y = 450) ∧
  (x + 5 * y = 53)

theorem common_difference_is_3 :
  ∃ (x y : ℝ), whale_plankton_frenzy x y ∧ y = 3 :=
by {
  sorry
}

end common_difference_is_3_l55_55198


namespace integer_roots_condition_l55_55154

theorem integer_roots_condition (a : ℝ) (h_pos : 0 < a) :
  (∀ x y : ℤ, (a ^ 2 * x ^ 2 + a * x + 1 - 13 * a ^ 2 = 0) ∧ (a ^ 2 * y ^ 2 + a * y + 1 - 13 * a ^ 2 = 0)) ↔
  (a = 1 ∨ a = 1/3 ∨ a = 1/4) :=
by sorry

end integer_roots_condition_l55_55154


namespace perp_tangents_l55_55109

theorem perp_tangents (a b : ℝ) (h : a + b = 5) (tangent_perp : ∀ x y : ℝ, x = 1 ∧ y = 1) :
  a / b = 1 / 3 :=
sorry

end perp_tangents_l55_55109


namespace general_term_formula_minimum_sum_value_l55_55139

variable {a : ℕ → ℚ} -- The arithmetic sequence
variable {S : ℕ → ℚ} -- Sum of the first n terms of the sequence

-- Conditions
axiom a_seq_cond1 : a 2 + a 6 = 6
axiom S_sum_cond5 : S 5 = 35 / 3

-- Definitions
def a_n (n : ℕ) : ℚ := (2 / 3) * n + 1 / 3
def S_n (n : ℕ) : ℚ := (1 / 3) * (n^2 + 2 * n)

-- Hypotheses
axiom seq_def : ∀ n, a n = a_n n
axiom sum_def : ∀ n, S n = S_n n

-- Theorems to be proved
theorem general_term_formula : ∀ n, a n = (2 / 3 * n) + 1 / 3 := by sorry
theorem minimum_sum_value : ∀ n, S 1 ≤ S n := by sorry

end general_term_formula_minimum_sum_value_l55_55139


namespace find_digits_l55_55594

-- Definitions, conditions and statement of the problem
def satisfies_condition (z : ℕ) (k : ℕ) (n : ℕ) : Prop :=
  n ≥ 1 ∧ (n^9 % 10^k) / 10^(k - 1) = z

theorem find_digits (z : ℕ) (k : ℕ) :
  k ≥ 1 →
  (z = 0 ∨ z = 1 ∨ z = 3 ∨ z = 7 ∨ z = 9) →
  ∃ n, satisfies_condition z k n := 
sorry

end find_digits_l55_55594


namespace negation_of_p_l55_55402

theorem negation_of_p : (¬ ∃ x : ℕ, x^2 > 4^x) ↔ (∀ x : ℕ, x^2 ≤ 4^x) :=
by
  sorry

end negation_of_p_l55_55402


namespace spherical_coordinates_convert_l55_55759

theorem spherical_coordinates_convert (ρ θ φ ρ' θ' φ' : ℝ) 
  (h₀ : ρ > 0) 
  (h₁ : 0 ≤ θ ∧ θ < 2 * Real.pi) 
  (h₂ : 0 ≤ φ ∧ φ ≤ Real.pi) 
  (h_initial : (ρ, θ, φ) = (4, (3 * Real.pi) / 8, (9 * Real.pi) / 5)) 
  (h_final : (ρ', θ', φ') = (4, (11 * Real.pi) / 8,  Real.pi / 5)) : 
  (ρ, θ, φ) = (4, (3 * Real.pi) / 8, (9 * Real.pi) / 5) → 
  (ρ, θ, φ) = (ρ', θ', φ') := 
by
  sorry

end spherical_coordinates_convert_l55_55759


namespace product_of_p_r_s_l55_55861

-- Definition of conditions
def eq1 (p : ℕ) : Prop := 4^p + 4^3 = 320
def eq2 (r : ℕ) : Prop := 3^r + 27 = 108
def eq3 (s : ℕ) : Prop := 2^s + 7^4 = 2617

-- Main statement
theorem product_of_p_r_s (p r s : ℕ) (h1 : eq1 p) (h2 : eq2 r) (h3 : eq3 s) : p * r * s = 112 :=
by sorry

end product_of_p_r_s_l55_55861


namespace intersection_A_B_l55_55029

def A : Set ℤ := { x | (2 * x + 3) * (x - 4) < 0 }
def B : Set ℝ := { x | 0 < x ∧ x ≤ Real.exp 1 }

theorem intersection_A_B :
  { x : ℤ | x ∈ A ∧ (x : ℝ) ∈ B } = {1, 2} :=
by
  sorry

end intersection_A_B_l55_55029


namespace cube_square_third_smallest_prime_l55_55554

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

def third_smallest_prime := 5

noncomputable def cube (n : ℕ) : ℕ := n * n * n

noncomputable def square (n : ℕ) : ℕ := n * n

theorem cube_square_third_smallest_prime : cube (square third_smallest_prime) = 15625 := by
  have h1 : is_prime 2 := by sorry
  have h2 : is_prime 3 := by sorry
  have h3 : is_prime 5 := by sorry
  sorry

end cube_square_third_smallest_prime_l55_55554


namespace inequality_solution_range_l55_55527

theorem inequality_solution_range (a : ℝ) :
  (∃ (x : ℝ), |x + 1| - |x - 2| < a^2 - 4 * a) → (a > 3 ∨ a < 1) :=
by
  sorry

end inequality_solution_range_l55_55527


namespace largest_part_of_proportional_division_l55_55361

theorem largest_part_of_proportional_division :
  ∀ (x y z : ℝ),
    x + y + z = 120 ∧
    x / (1 / 2) = y / (1 / 4) ∧
    x / (1 / 2) = z / (1 / 6) →
    max x (max y z) = 60 :=
by sorry

end largest_part_of_proportional_division_l55_55361


namespace line_contains_point_l55_55729

theorem line_contains_point (k : ℝ) : 
  let x := (1 : ℝ) / 3
  let y := -2 
  let line_eq := (3 : ℝ) - 3 * k * x = 4 * y
  line_eq → k = 11 :=
by
  intro h
  sorry

end line_contains_point_l55_55729


namespace find_dividend_l55_55911

-- Given conditions as definitions
def divisor : ℕ := 16
def quotient : ℕ := 9
def remainder : ℕ := 5

-- Lean 4 statement to be proven
theorem find_dividend : divisor * quotient + remainder = 149 := by
  sorry

end find_dividend_l55_55911


namespace number_of_intersections_l55_55192

-- Definitions of the given curves.
def curve1 (x y : ℝ) : Prop := x^2 + 4*y^2 = 1
def curve2 (x y : ℝ) : Prop := 4*x^2 + y^2 = 4

-- Statement of the theorem
theorem number_of_intersections : ∃! p : ℝ × ℝ, curve1 p.1 p.2 ∧ curve2 p.1 p.2 := sorry

end number_of_intersections_l55_55192


namespace linear_equation_condition_l55_55213

theorem linear_equation_condition (a : ℝ) :
  (∃ x : ℝ, (a - 2) * x ^ (|a|⁻¹ + 3) = 0) ↔ a = -2 := 
by
  sorry

end linear_equation_condition_l55_55213


namespace symmetric_points_l55_55952

-- Let points P and Q be symmetric about the origin
variables (m n : ℤ)
axiom symmetry_condition : (m, 4) = (- (-2), -n)

theorem symmetric_points :
  m = 2 ∧ n = -4 := 
  by {
    sorry
  }

end symmetric_points_l55_55952


namespace fraction_subtraction_property_l55_55440

variable (a b c d : ℚ)

theorem fraction_subtraction_property :
  (a / b - c / d) = ((a - c) / (b + d)) → (a / c) = (b / d) ^ 2 := 
by
  sorry

end fraction_subtraction_property_l55_55440


namespace min_value_is_correct_l55_55848

noncomputable def min_value (P : ℝ × ℝ) (A B C : ℝ × ℝ) : ℝ := 
  let PA := (A.1 - P.1, A.2 - P.2)
  let PB := (B.1 - P.1, B.2 - P.2)
  let PC := (C.1 - P.1, C.2 - P.2)
  PA.1 * PB.1 + PA.2 * PB.2 +
  PB.1 * PC.1 + PB.2 * PC.2 +
  PC.1 * PA.1 + PC.2 * PA.2

theorem min_value_is_correct :
  ∃ P : ℝ × ℝ, P = (5/3, 1/3) ∧
  min_value P (1, 4) (4, 1) (0, -4) = -62/3 :=
by
  sorry

end min_value_is_correct_l55_55848


namespace teachers_like_at_least_one_l55_55774

theorem teachers_like_at_least_one (T C B N: ℕ) 
    (total_teachers : T + C + N = 90)  -- Total number of teachers plus neither equals 90
    (tea_teachers : T = 66)           -- Teachers who like tea is 66
    (coffee_teachers : C = 42)        -- Teachers who like coffee is 42
    (both_beverages : B = 3 * N)      -- Teachers who like both is three times neither
    : T + C - B = 81 :=               -- Teachers who like at least one beverage
by 
  sorry

end teachers_like_at_least_one_l55_55774


namespace cookie_price_ratio_l55_55926

theorem cookie_price_ratio (c b : ℝ) (h1 : 6 * c + 5 * b = 3 * (3 * c + 27 * b)) : c = (4 / 5) * b :=
sorry

end cookie_price_ratio_l55_55926


namespace original_average_speed_l55_55035

theorem original_average_speed :
  ∀ (D : ℝ),
  (V = D / (5 / 6)) ∧ (60 = D / (2 / 3)) → V = 48 :=
by
  sorry

end original_average_speed_l55_55035


namespace total_number_of_dresses_l55_55461

theorem total_number_of_dresses (ana_dresses lisa_more_dresses : ℕ) (h_condition : ana_dresses = 15) (h_more : lisa_more_dresses = ana_dresses + 18) : ana_dresses + lisa_more_dresses = 48 :=
by
  sorry

end total_number_of_dresses_l55_55461


namespace no_nonzero_solution_l55_55721

theorem no_nonzero_solution (a b c n : ℤ) 
  (h : 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := 
by 
  sorry

end no_nonzero_solution_l55_55721


namespace fourth_quadrant_negative_half_x_axis_upper_half_plane_l55_55502

theorem fourth_quadrant (m : ℝ) : ((-7 < m ∧ m < 3) ↔ ((m^2 - 8 * m + 15 > 0) ∧ (m^2 + 3 * m - 28 < 0))) :=
sorry

theorem negative_half_x_axis (m : ℝ) : (m = 4 ↔ ((m^2 - 8 * m + 15 < 0) ∧ (m^2 + 3 * m - 28 = 0))) :=
sorry

theorem upper_half_plane (m : ℝ) : ((m ≥ 4 ∨ m ≤ -7) ↔ (m^2 + 3 * m - 28 ≥ 0)) :=
sorry

end fourth_quadrant_negative_half_x_axis_upper_half_plane_l55_55502


namespace head_start_ratio_l55_55099

variable (Va Vb L H : ℕ)

-- Conditions
def speed_relation : Prop := Va = (4 * Vb) / 3

-- The head start fraction that makes A and B finish the race at the same time given the speed relation
theorem head_start_ratio (Va Vb L H : ℕ)
  (h1 : speed_relation Va Vb)
  (h2 : L > 0) : (H = L / 4) :=
sorry

end head_start_ratio_l55_55099


namespace wire_ratio_l55_55193

theorem wire_ratio (bonnie_pieces : ℕ) (length_per_bonnie_piece : ℕ) (roark_volume : ℕ) 
  (unit_cube_volume : ℕ) (bonnie_cube_volume : ℕ) (roark_pieces_per_unit_cube : ℕ)
  (bonnie_total_wire : ℕ := bonnie_pieces * length_per_bonnie_piece)
  (roark_total_wire : ℕ := (bonnie_cube_volume / unit_cube_volume) * roark_pieces_per_unit_cube) :
  bonnie_pieces = 12 →
  length_per_bonnie_piece = 4 →
  unit_cube_volume = 1 →
  bonnie_cube_volume = 64 →
  roark_pieces_per_unit_cube = 12 →
  (bonnie_total_wire / roark_total_wire : ℚ) = 1 / 16 :=
by sorry

end wire_ratio_l55_55193


namespace solve_quadratic_l55_55693

theorem solve_quadratic (x : ℝ) :
  25 * x^2 - 10 * x - 1000 = 0 → ∃ r s, (x + r)^2 = s ∧ s = 40.04 :=
by
  intro h
  sorry

end solve_quadratic_l55_55693


namespace regression_prediction_l55_55157

theorem regression_prediction
  (slope : ℝ) (centroid_x centroid_y : ℝ) (b : ℝ)
  (h_slope : slope = 1.23)
  (h_centroid : centroid_x = 4 ∧ centroid_y = 5)
  (h_intercept : centroid_y = slope * centroid_x + b)
  (x : ℝ) (h_x : x = 10) :
  centroid_y = 5 →
  slope = 1.23 →
  x = 10 →
  b = 5 - 1.23 * 4 →
  (slope * x + b) = 12.38 :=
by
  intros
  sorry

end regression_prediction_l55_55157


namespace min_blocks_for_wall_l55_55768

noncomputable def min_blocks_needed (length height : ℕ) (block_sizes : List (ℕ × ℕ)) : ℕ :=
  sorry

theorem min_blocks_for_wall :
  min_blocks_needed 120 8 [(1, 3), (1, 2), (1, 1)] = 404 := by
  sorry

end min_blocks_for_wall_l55_55768


namespace gcd_m_n_l55_55890

def m : ℕ := 555555555
def n : ℕ := 1111111111

theorem gcd_m_n : Nat.gcd m n = 1 := by
  sorry

end gcd_m_n_l55_55890


namespace continuity_of_f_at_2_l55_55054

def f (x : ℝ) := -2 * x^2 - 5

theorem continuity_of_f_at_2 : ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by {
  sorry
}

end continuity_of_f_at_2_l55_55054


namespace calculate_division_l55_55032

theorem calculate_division : 
  (- (1 / 28)) / ((1 / 2) - (1 / 4) + (1 / 7) - (1 / 14)) = - (1 / 9) :=
by
  sorry

end calculate_division_l55_55032


namespace complex_expression_equality_l55_55022

open Complex

theorem complex_expression_equality (i : ℂ) (h : i^2 = -1) :
  (1 + i)^20 - (1 - i)^20 = 0 := 
sorry

end complex_expression_equality_l55_55022


namespace recent_quarter_revenue_l55_55816

theorem recent_quarter_revenue :
  let revenue_year_ago : Float := 69.0
  let percentage_decrease : Float := 30.434782608695656
  let decrease_in_revenue : Float := revenue_year_ago * (percentage_decrease / 100)
  let recent_quarter_revenue := revenue_year_ago - decrease_in_revenue
  recent_quarter_revenue = 48.0 := by
  sorry

end recent_quarter_revenue_l55_55816


namespace total_weight_of_plastic_rings_l55_55691

-- Conditions
def orange_ring_weight : ℝ := 0.08
def purple_ring_weight : ℝ := 0.33
def white_ring_weight : ℝ := 0.42

-- Proof Statement
theorem total_weight_of_plastic_rings :
  orange_ring_weight + purple_ring_weight + white_ring_weight = 0.83 := by
  sorry

end total_weight_of_plastic_rings_l55_55691


namespace driver_weekly_distance_l55_55228

-- Defining the conditions
def speed_part1 : ℕ := 30  -- speed in miles per hour for the first part
def time_part1 : ℕ := 3    -- time in hours for the first part
def speed_part2 : ℕ := 25  -- speed in miles per hour for the second part
def time_part2 : ℕ := 4    -- time in hours for the second part
def days_per_week : ℕ := 6 -- number of days the driver works in a week

-- Total distance calculation each day
def distance_part1 := speed_part1 * time_part1
def distance_part2 := speed_part2 * time_part2
def daily_distance := distance_part1 + distance_part2

-- Total distance travel in a week
def weekly_distance := daily_distance * days_per_week

-- Theorem stating that weekly distance is 1140 miles
theorem driver_weekly_distance : weekly_distance = 1140 :=
by
  -- We skip the proof using sorry
  sorry

end driver_weekly_distance_l55_55228


namespace solution_l55_55571

variable (f g : ℝ → ℝ)

open Real

-- Define f(x) and g(x) as given in the problem
def isSolution (x : ℝ) : Prop :=
  f x + g x = sqrt ((1 + cos (2 * x)) / (1 - sin x)) ∧
  (∀ x, f (-x) = -f x) ∧
  (∀ x, g (-x) = g x)

-- The theorem we want to prove
theorem solution (x : ℝ) (hx : -π / 2 < x ∧ x < π / 2)
  (h : isSolution f g x) : (f x)^2 - (g x)^2 = -2 * cos x := 
sorry

end solution_l55_55571


namespace milton_books_l55_55381

theorem milton_books (Z B : ℕ) (h1 : B = 4 * Z) (h2 : Z + B = 80) : Z = 16 :=
sorry

end milton_books_l55_55381


namespace probability_prime_sum_l55_55006

def is_prime (n: ℕ) : Prop := Nat.Prime n

theorem probability_prime_sum :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let total_pairs := (primes.length.choose 2)
  let successful_pairs := [
    (2, 3), (2, 5), (2, 11), (2, 17), (2, 29)
  ]
  let num_successful_pairs := successful_pairs.length
  (num_successful_pairs : ℚ) / total_pairs = 1 / 9 :=
by
  sorry

end probability_prime_sum_l55_55006


namespace intersection_M_N_l55_55289

-- Defining set M
def M : Set ℕ := {1, 2, 3, 4}

-- Defining the set N based on the condition
def N : Set ℕ := {x | ∃ n ∈ M, x = n^2}

-- Lean statement to prove the intersection
theorem intersection_M_N : M ∩ N = {1, 4} := 
by
  sorry

end intersection_M_N_l55_55289


namespace daily_production_l55_55242

-- Define the conditions
def bottles_per_case : ℕ := 9
def num_cases : ℕ := 8000

-- State the theorem with the question and the calculated answer
theorem daily_production : bottles_per_case * num_cases = 72000 :=
by
  sorry

end daily_production_l55_55242


namespace no_nat_p_prime_and_p6_plus_6_prime_l55_55994

theorem no_nat_p_prime_and_p6_plus_6_prime (p : ℕ) (h1 : Nat.Prime p) (h2 : Nat.Prime (p^6 + 6)) : False := 
sorry

end no_nat_p_prime_and_p6_plus_6_prime_l55_55994


namespace smallest_number_divisible_l55_55013

theorem smallest_number_divisible (n : ℤ) : 
  (n + 7) % 25 = 0 ∧
  (n + 7) % 49 = 0 ∧
  (n + 7) % 15 = 0 ∧
  (n + 7) % 21 = 0 ↔ n = 3668 :=
by 
 sorry

end smallest_number_divisible_l55_55013


namespace largest_divisor_of_n_l55_55720

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : ∃ k : ℕ, n^2 = 18 * k) : ∃ l : ℕ, n = 6 * l :=
sorry

end largest_divisor_of_n_l55_55720


namespace blocks_remaining_l55_55983

def initial_blocks : ℕ := 55
def blocks_eaten : ℕ := 29

theorem blocks_remaining : initial_blocks - blocks_eaten = 26 := by
  sorry

end blocks_remaining_l55_55983


namespace no_solution_for_xx_plus_yy_eq_9z_l55_55126

theorem no_solution_for_xx_plus_yy_eq_9z (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ¬ (x^x + y^y = 9^z) :=
sorry

end no_solution_for_xx_plus_yy_eq_9z_l55_55126


namespace tangent_line_equation_l55_55454

def f (x : ℝ) : ℝ := x^2

theorem tangent_line_equation :
  let x := (1 : ℝ)
  let y := f x
  ∃ m b : ℝ, m = 2 ∧ b = 1 ∧ (2*x - y - 1 = 0) := by
  sorry

end tangent_line_equation_l55_55454


namespace quadratic_roots_always_implies_l55_55343

variable {k x1 x2 : ℝ}

theorem quadratic_roots_always_implies (h1 : k^2 > 16) 
  (h2 : x1 + x2 = -k)
  (h3 : x1 * x2 = 4) : x1^2 + x2^2 > 8 :=
by
  sorry

end quadratic_roots_always_implies_l55_55343


namespace bob_more_than_ken_l55_55248

def ken_situps : ℕ := 20

def nathan_situps : ℕ := 2 * ken_situps

def bob_situps : ℕ := (ken_situps + nathan_situps) / 2

theorem bob_more_than_ken : bob_situps - ken_situps = 10 := 
sorry

end bob_more_than_ken_l55_55248


namespace yellow_marbles_problem_l55_55142

variable (Y B R : ℕ)

theorem yellow_marbles_problem
  (h1 : Y + B + R = 19)
  (h2 : B = (3 * R) / 4)
  (h3 : R = Y + 3) :
  Y = 5 :=
by
  sorry

end yellow_marbles_problem_l55_55142


namespace ceil_sqrt_200_eq_15_l55_55310

theorem ceil_sqrt_200_eq_15 : ⌈Real.sqrt 200⌉ = 15 := 
sorry

end ceil_sqrt_200_eq_15_l55_55310


namespace middle_number_is_correct_l55_55450

theorem middle_number_is_correct (numbers : List ℝ) (h_length : numbers.length = 11)
  (h_avg11 : numbers.sum / 11 = 9.9)
  (first_6 : List ℝ) (h_first6_length : first_6.length = 6)
  (h_avg6_1 : first_6.sum / 6 = 10.5)
  (last_6 : List ℝ) (h_last6_length : last_6.length = 6)
  (h_avg6_2 : last_6.sum / 6 = 11.4) :
  (∃ m : ℝ, m ∈ first_6 ∧ m ∈ last_6 ∧ m = 22.5) :=
by
  sorry

end middle_number_is_correct_l55_55450


namespace extremely_powerful_count_l55_55778

def is_extremely_powerful (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ b % 2 = 1 ∧ a^b = n

noncomputable def count_extremely_powerful_below (m : ℕ) : ℕ :=
  Nat.card { n : ℕ | is_extremely_powerful n ∧ n < m }

theorem extremely_powerful_count : count_extremely_powerful_below 5000 = 19 :=
by
  sorry

end extremely_powerful_count_l55_55778


namespace largest_angle_of_triangle_l55_55466

theorem largest_angle_of_triangle (x : ℝ) 
  (h1 : 35 + 70 + x = 180) : 75 = max (max 35 70) x := 
sorry

end largest_angle_of_triangle_l55_55466


namespace jenny_ate_65_chocolates_l55_55493

noncomputable def chocolates_eaten_by_Jenny : ℕ :=
  let chocolates_mike := 20
  let chocolates_john := chocolates_mike / 2
  let combined_chocolates := chocolates_mike + chocolates_john
  let twice_combined_chocolates := 2 * combined_chocolates
  5 + twice_combined_chocolates

theorem jenny_ate_65_chocolates :
  chocolates_eaten_by_Jenny = 65 :=
by
  -- Skipping the proof details
  sorry

end jenny_ate_65_chocolates_l55_55493


namespace xy_identity_l55_55765

theorem xy_identity (x y : ℝ) (h1 : x + y = 11) (h2 : x * y = 24) : (x^2 + y^2) * (x + y) = 803 := by
  sorry

end xy_identity_l55_55765


namespace solve_for_x_l55_55702

theorem solve_for_x (x : ℤ) (h : 3 * x - 7 = 11) : x = 6 :=
by
  sorry

end solve_for_x_l55_55702


namespace quadrilateral_is_parallelogram_l55_55130

theorem quadrilateral_is_parallelogram
  (AB BC CD DA : ℝ)
  (K L M N : ℝ)
  (H₁ : K = (AB + BC) / 2)
  (H₂ : L = (BC + CD) / 2)
  (H₃ : M = (CD + DA) / 2)
  (H₄ : N = (DA + AB) / 2)
  (H : K + M + L + N = (AB + BC + CD + DA) / 2)
  : ∃ P Q R S : ℝ, P ≠ Q ∧ Q ≠ R ∧ R ≠ S ∧ S ≠ P ∧ 
    (P + R = AB) ∧ (Q + S = CD)  := 
sorry

end quadrilateral_is_parallelogram_l55_55130


namespace amazing_rectangle_area_unique_l55_55562

def isAmazingRectangle (a b : ℕ) : Prop :=
  a = 2 * b ∧ a * b = 3 * (2 * (a + b))

theorem amazing_rectangle_area_unique :
  ∃ (a b : ℕ), isAmazingRectangle a b ∧ a * b = 162 :=
by
  sorry

end amazing_rectangle_area_unique_l55_55562


namespace sum_of_first_9_terms_l55_55246

variable (a : ℕ → ℤ)
variable (S : ℕ → ℤ)
variable (a1 : ℤ)
variable (d : ℤ)

-- Given is that the sequence is arithmetic.
-- Given a1 is the first term, and d is the common difference, we can define properties based on the conditions.
def is_arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = a1 + (n - 1) * d

def sum_first_n_terms (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

-- Given condition: 2a_1 + a_13 = -9.
def given_condition (a : ℕ → ℤ) (a1 : ℤ) (d : ℤ) : Prop :=
  2 * a1 + (a1 + 12 * d) = -9

theorem sum_of_first_9_terms (a : ℕ → ℤ) (S : ℕ → ℤ) (a1 d : ℤ)
  (h_arith : is_arithmetic_sequence a a1 d)
  (h_sum : sum_first_n_terms S a)
  (h_cond : given_condition a a1 d) :
  S 9 = -27 :=
sorry

end sum_of_first_9_terms_l55_55246


namespace card_collection_problem_l55_55647

theorem card_collection_problem 
  (m : ℕ) 
  (h : (2 * m + 1) / 3 = 56) : 
  m = 84 :=
sorry

end card_collection_problem_l55_55647


namespace polar_bear_daily_salmon_consumption_l55_55428

/-- Polar bear's fish consumption conditions and daily salmon amount calculation -/
theorem polar_bear_daily_salmon_consumption (h1: ℝ) (h2: ℝ) : 
  (h1 = 0.2) → (h2 = 0.6) → (h2 - h1 = 0.4) :=
by
  sorry

end polar_bear_daily_salmon_consumption_l55_55428


namespace monotonic_decreasing_interval_of_f_l55_55753

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem monotonic_decreasing_interval_of_f :
  ∀ x : ℝ, -1 < x ∧ x < 1 → deriv f x < 0 :=
by
  sorry

end monotonic_decreasing_interval_of_f_l55_55753


namespace find_a_value_l55_55516

theorem find_a_value :
  let center := (0.5, Real.sqrt 2)
  let line_dist (a : ℝ) := (abs (0.5 * a + Real.sqrt 2 - Real.sqrt 2)) / Real.sqrt (a^2 + 1)
  line_dist a = Real.sqrt 2 / 4 ↔ (a = 1 ∨ a = -1) :=
by
  sorry

end find_a_value_l55_55516


namespace symmetric_about_origin_l55_55521

theorem symmetric_about_origin (x y : ℝ) :
  (∀ (x y : ℝ), (x*y - x^2 = 1) → ((-x)*(-y) - (-x)^2 = 1)) :=
by
  intros x y h
  sorry

end symmetric_about_origin_l55_55521


namespace arithmetic_sequence_contains_term_l55_55888

theorem arithmetic_sequence_contains_term (a1 : ℤ) (d : ℤ) (k : ℕ) (h1 : a1 = 3) (h2 : d = 9) :
  ∃ n : ℕ, (a1 + (n - 1) * d) = 3 * 4 ^ k := by
  sorry

end arithmetic_sequence_contains_term_l55_55888


namespace complement_of_A_union_B_in_U_l55_55676

def U : Set ℝ := { x | -5 < x ∧ x < 5 }
def A : Set ℝ := { x | x^2 - 4*x - 5 < 0 }
def B : Set ℝ := { x | -2 < x ∧ x < 4 }

theorem complement_of_A_union_B_in_U :
  (U \ (A ∪ B)) = { x | -5 < x ∧ x ≤ -2 } := by
  sorry

end complement_of_A_union_B_in_U_l55_55676


namespace factorization_correct_l55_55308

theorem factorization_correct (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) :=
by
  -- The actual proof will be written here.
  sorry

end factorization_correct_l55_55308


namespace files_remaining_l55_55098

def initial_music_files : ℕ := 27
def initial_video_files : ℕ := 42
def initial_doc_files : ℕ := 12
def compression_ratio_music : ℕ := 2
def compression_ratio_video : ℕ := 3
def files_deleted : ℕ := 11

def compressed_music_files : ℕ := initial_music_files * compression_ratio_music
def compressed_video_files : ℕ := initial_video_files * compression_ratio_video
def total_compressed_files : ℕ := compressed_music_files + compressed_video_files + initial_doc_files

theorem files_remaining : total_compressed_files - files_deleted = 181 := by
  -- we skip the proof for now
  sorry

end files_remaining_l55_55098


namespace shopkeeper_discount_l55_55179

theorem shopkeeper_discount :
  let CP := 100
  let SP_with_discount := 119.7
  let SP_without_discount := 126
  let discount := SP_without_discount - SP_with_discount
  let discount_percentage := (discount / SP_without_discount) * 100
  discount_percentage = 5 := sorry

end shopkeeper_discount_l55_55179


namespace tan_alpha_proof_l55_55714

theorem tan_alpha_proof 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 2)
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_proof_l55_55714


namespace inscribed_sphere_radius_l55_55060

theorem inscribed_sphere_radius (h1 h2 h3 h4 : ℝ) (S1 S2 S3 S4 V : ℝ)
  (h1_ge : h1 ≥ 1) (h2_ge : h2 ≥ 1) (h3_ge : h3 ≥ 1) (h4_ge : h4 ≥ 1)
  (volume : V = (1/3) * S1 * h1)
  : (∃ r : ℝ, 3 * V = (S1 + S2 + S3 + S4) * r ∧ r = 1 / 4) :=
by
  sorry

end inscribed_sphere_radius_l55_55060


namespace trig_identity_l55_55436

theorem trig_identity : 4 * Real.cos (50 * Real.pi / 180) - Real.tan (40 * Real.pi / 180) = Real.sqrt 3 :=
by sorry

end trig_identity_l55_55436


namespace profit_8000_l55_55435

noncomputable def profit (selling_price increase : ℝ) : ℝ :=
  (selling_price - 40 + increase) * (500 - 10 * increase)

theorem profit_8000 (increase : ℝ) :
  profit 50 increase = 8000 →
  ((increase = 10 ∧ (50 + increase = 60) ∧ (500 - 10 * increase = 400)) ∨ 
   (increase = 30 ∧ (50 + increase = 80) ∧ (500 - 10 * increase = 200))) :=
by
  sorry

end profit_8000_l55_55435


namespace percent_of_workday_in_meetings_l55_55767

theorem percent_of_workday_in_meetings (h1 : 9 > 0) (m1 m2 : ℕ) (h2 : m1 = 45) (h3 : m2 = 2 * m1) : 
  (135 / 540 : ℚ) * 100 = 25 := 
by
  -- Just for structure, the proof should go here
  sorry

end percent_of_workday_in_meetings_l55_55767


namespace find_largest_beta_l55_55280

theorem find_largest_beta (α : ℝ) (r : ℕ → ℝ) (C : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < 1)
  (h3 : ∀ n, ∀ m ≠ n, dist (r n) (r m) ≥ (r n) ^ α)
  (h4 : ∀ n, r n ≤ r (n + 1)) 
  (h5 : ∀ n, r n ≥ C * n ^ (1 / (2 * (1 - α)))) :
  ∀ β, (∃ C > 0, ∀ n, r n ≥ C * n ^ β) → β ≤ 1 / (2 * (1 - α)) :=
sorry

end find_largest_beta_l55_55280


namespace triangle_angles_l55_55920

variable (A B C : ℝ)

def is_triangle (A B C : ℝ) : Prop :=
  A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = 180

theorem triangle_angles (x : ℝ) (hA : A = x) (hB : B = 2 * A) (hC : C + A + B = 180) :
  A = x ∧ B = 2 * x ∧ C = 180 - 3 * x := by
  -- proof goes here
  sorry

end triangle_angles_l55_55920


namespace calc_value_of_ab_bc_ca_l55_55451

theorem calc_value_of_ab_bc_ca (a b c : ℝ) (h1 : a + b + c = 35) (h2 : ab + bc + ca = 320) (h3 : abc = 600) : 
  (a + b) * (b + c) * (c + a) = 10600 := 
by sorry

end calc_value_of_ab_bc_ca_l55_55451


namespace point_in_second_quadrant_l55_55945

def is_in_second_quadrant (x y : ℤ) : Prop :=
  x < 0 ∧ y > 0

theorem point_in_second_quadrant : is_in_second_quadrant (-2) 3 :=
by
  sorry

end point_in_second_quadrant_l55_55945


namespace a_range_l55_55356

open Set

variable (A B : Set Real) (a : Real)

def A_def : Set Real := {x | 3 * x + 1 < 4}
def B_def : Set Real := {x | x - a < 0}
def intersection_eq : A ∩ B = A := sorry

theorem a_range : a ≥ 1 :=
  by
  have hA : A = {x | x < 1} := sorry
  have hB : B = {x | x < a} := sorry
  have h_intersection : (A ∩ B) = A := sorry
  sorry

end a_range_l55_55356


namespace smallest_positive_integer_congruence_l55_55351

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ, 5 * x ≡ 14 [MOD 31] ∧ 0 < x ∧ x < 31 := 
sorry

end smallest_positive_integer_congruence_l55_55351


namespace find_n_values_l55_55096

-- Define a function that calculates the polynomial expression
def prime_expression (n : ℕ) : ℕ :=
  n^4 - 27 * n^2 + 121

-- State the problem as a theorem
theorem find_n_values (n : ℕ) (h : Nat.Prime (prime_expression n)) : n = 2 ∨ n = 5 :=
  sorry

end find_n_values_l55_55096


namespace ratio_size12_to_size6_l55_55981

-- Definitions based on conditions
def cheerleaders_size2 : ℕ := 4
def cheerleaders_size6 : ℕ := 10
def total_cheerleaders : ℕ := 19
def cheerleaders_size12 : ℕ := total_cheerleaders - (cheerleaders_size2 + cheerleaders_size6)

-- Proof statement
theorem ratio_size12_to_size6 : cheerleaders_size12.toFloat / cheerleaders_size6.toFloat = 1 / 2 := sorry

end ratio_size12_to_size6_l55_55981


namespace joe_first_lift_weight_l55_55661

variable (x y : ℕ)

def joe_lift_conditions (x y : ℕ) : Prop :=
  x + y = 600 ∧ 2 * x = y + 300

theorem joe_first_lift_weight (x y : ℕ) (h : joe_lift_conditions x y) : x = 300 :=
by
  sorry

end joe_first_lift_weight_l55_55661


namespace square_b_perimeter_l55_55731

theorem square_b_perimeter (a b : ℝ) 
  (ha : a^2 = 65) 
  (prob : (65 - b^2) / 65 = 0.7538461538461538) : 
  4 * b = 16 :=
by 
  sorry

end square_b_perimeter_l55_55731


namespace minimum_value_expression_l55_55133

theorem minimum_value_expression (a : ℝ) (h : a > 0) : 
  a + (a + 4) / a ≥ 5 :=
sorry

end minimum_value_expression_l55_55133


namespace find_missing_fraction_l55_55514

def f1 := 1/3
def f2 := 1/2
def f3 := 1/5
def f4 := 1/4
def f5 := -9/20
def f6 := -9/20
def total_sum := 45/100
def missing_fraction := 1/15

theorem find_missing_fraction : f1 + f2 + f3 + f4 + f5 + f6 + missing_fraction = total_sum :=
by
  sorry

end find_missing_fraction_l55_55514


namespace trajectory_equation_of_point_M_l55_55814

variables {x y a b : ℝ}

theorem trajectory_equation_of_point_M :
  (a^2 + b^2 = 100) →
  (x = a / (1 + 4)) →
  (y = 4 * b / (1 + 4)) →
  16 * x^2 + y^2 = 64 :=
by
  intros h1 h2 h3
  sorry

end trajectory_equation_of_point_M_l55_55814


namespace sin_alpha_cos_2beta_l55_55282

theorem sin_alpha_cos_2beta :
  ∀ α β : ℝ, 3 * Real.sin α - Real.sin β = Real.sqrt 10 ∧ α + β = Real.pi / 2 →
  Real.sin α = 3 * Real.sqrt 10 / 10 ∧ Real.cos (2 * β) = 4 / 5 :=
by
  intros α β h
  sorry

end sin_alpha_cos_2beta_l55_55282


namespace intersection_complement_eq_C_l55_55621

def A := { x : ℝ | -3 < x ∧ x < 6 }
def B := { x : ℝ | 2 < x ∧ x < 7 }
def complement_B := { x : ℝ | x ≤ 2 ∨ x ≥ 7 }
def C := { x : ℝ | -3 < x ∧ x ≤ 2 }

theorem intersection_complement_eq_C :
  A ∩ complement_B = C :=
sorry

end intersection_complement_eq_C_l55_55621


namespace product_odd_integers_lt_20_l55_55653

/--
The product of all odd positive integers strictly less than 20 is a positive number ending with the digit 5.
-/
theorem product_odd_integers_lt_20 :
  let nums := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
  let product := List.prod nums
  (product > 0) ∧ (product % 10 = 5) :=
by
  sorry

end product_odd_integers_lt_20_l55_55653


namespace base_angle_isosceles_triangle_l55_55487

theorem base_angle_isosceles_triangle (α : ℝ) (hα : α = 108) (isosceles : ∀ (a b c : ℝ), a = b ∨ b = c ∨ c = a) : α = 108 →
  α + β + β = 180 → β = 36 :=
by
  sorry

end base_angle_isosceles_triangle_l55_55487


namespace complex_series_sum_eq_zero_l55_55833

open Complex

theorem complex_series_sum_eq_zero {ω : ℂ} (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^18 + ω^27 + ω^36 + ω^45 + ω^54 + ω^63 + ω^72 + ω^81 + ω^90 = 0 := by
  sorry

end complex_series_sum_eq_zero_l55_55833


namespace find_total_sales_l55_55903

theorem find_total_sales
  (S : ℝ)
  (h_comm1 : ∀ x, x ≤ 5000 → S = 0.9 * x → S = 16666.67 → false)
  (h_comm2 : S > 5000 → S - (500 + 0.05 * (S - 5000)) = 15000):
  S = 16052.63 :=
by
  sorry

end find_total_sales_l55_55903


namespace probability_correct_l55_55620

def total_chips : ℕ := 15
def total_ways_to_draw_2_chips : ℕ := Nat.choose 15 2

def chips_same_color : ℕ := 3 * (Nat.choose 5 2)
def chips_same_number : ℕ := 5 * (Nat.choose 3 2)
def favorable_outcomes : ℕ := chips_same_color + chips_same_number

def probability_same_color_or_number : ℚ := favorable_outcomes / total_ways_to_draw_2_chips

theorem probability_correct :
  probability_same_color_or_number = 3 / 7 :=
by sorry

end probability_correct_l55_55620


namespace total_surface_area_first_rectangular_parallelepiped_equals_22_l55_55346

theorem total_surface_area_first_rectangular_parallelepiped_equals_22
  (x y z : ℝ)
  (h1 : (x + 1) * (y + 1) * (z + 1) = x * y * z + 18)
  (h2 : 2 * ((x + 1) * (y + 1) + (y + 1) * (z + 1) + (z + 1) * (x + 1)) = 2 * (x * y + x * z + y * z) + 30) :
  2 * (x * y + x * z + y * z) = 22 := sorry

end total_surface_area_first_rectangular_parallelepiped_equals_22_l55_55346


namespace sum_of_fourth_powers_eq_square_of_sum_of_squares_l55_55500

theorem sum_of_fourth_powers_eq_square_of_sum_of_squares 
  (x1 x2 x3 : ℝ) (p q n : ℝ)
  (h1 : x1^3 + p*x1^2 + q*x1 + n = 0)
  (h2 : x2^3 + p*x2^2 + q*x2 + n = 0)
  (h3 : x3^3 + p*x3^2 + q*x3 + n = 0)
  (h_rel : q^2 = 2 * n * p) :
  x1^4 + x2^4 + x3^4 = (x1^2 + x2^2 + x3^2)^2 := 
sorry

end sum_of_fourth_powers_eq_square_of_sum_of_squares_l55_55500


namespace total_homework_pages_l55_55549

theorem total_homework_pages (R : ℕ) (H1 : R + 3 = 8) : R + (R + 3) = 13 :=
by sorry

end total_homework_pages_l55_55549


namespace find_polynomial_l55_55882

-- Define the polynomial function and the constant
variables {F : Type*} [Field F]

-- The main condition of the problem
def satisfies_condition (p : F → F) (c : F) :=
  ∀ x : F, p (p x) = x * p x + c * x^2

-- Prove the correct answers
theorem find_polynomial (p : F → F) (c : F) : 
  (c = 0 → ∀ x, p x = x) ∧ (c = -2 → ∀ x, p x = -x) :=
by
  sorry

end find_polynomial_l55_55882


namespace roof_shingle_width_l55_55749

theorem roof_shingle_width (L A W : ℕ) (hL : L = 10) (hA : A = 70) (hArea : A = L * W) : W = 7 :=
by
  sorry

end roof_shingle_width_l55_55749


namespace maurice_rides_l55_55408

theorem maurice_rides (M : ℕ) 
    (h1 : ∀ m_attended : ℕ, m_attended = 8)
    (h2 : ∀ matt_other : ℕ, matt_other = 16)
    (h3 : ∀ total_matt : ℕ, total_matt = matt_other + m_attended)
    (h4 : total_matt = 3 * M) : M = 8 :=
by 
  sorry

end maurice_rides_l55_55408


namespace find_third_number_in_proportion_l55_55379

theorem find_third_number_in_proportion (x : ℝ) (third_number : ℝ) (h1 : x = 0.9) (h2 : 0.75 / 6 = x / third_number) : third_number = 5 := by
  sorry

end find_third_number_in_proportion_l55_55379


namespace find_q_l55_55447

def P (q x : ℝ) : ℝ := x^4 + 2 * q * x^3 - 3 * x^2 + 2 * q * x + 1

theorem find_q (q : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 < 0 ∧ x2 < 0 ∧ P q x1 = 0 ∧ P q x2 = 0) → q < 1 / 4 :=
by
  sorry

end find_q_l55_55447


namespace polynomial_coeff_fraction_eq_neg_122_div_121_l55_55974

theorem polynomial_coeff_fraction_eq_neg_122_div_121
  (a0 a1 a2 a3 a4 a5 : ℤ)
  (h1 : (2 - 1) ^ 5 = a0 + a1 * 1 + a2 * 1^2 + a3 * 1^3 + a4 * 1^4 + a5 * 1^5)
  (h2 : (2 - (-1)) ^ 5 = a0 + a1 * (-1) + a2 * (-1)^2 + a3 * (-1)^3 + a4 * (-1)^4 + a5 * (-1)^5)
  (h_sum1 : a0 + a1 + a2 + a3 + a4 + a5 = 1)
  (h_sum2 : a0 - a1 + a2 - a3 + a4 - a5 = 243) :
  (a0 + a2 + a4) / (a1 + a3 + a5) = - 122 / 121 :=
sorry

end polynomial_coeff_fraction_eq_neg_122_div_121_l55_55974


namespace Cody_book_series_total_count_l55_55555

theorem Cody_book_series_total_count :
  ∀ (weeks: ℕ) (books_first_week: ℕ) (books_second_week: ℕ) (books_per_week_after: ℕ),
    weeks = 7 ∧ books_first_week = 6 ∧ books_second_week = 3 ∧ books_per_week_after = 9 →
    (books_first_week + books_second_week + (weeks - 2) * books_per_week_after) = 54 :=
by
  sorry

end Cody_book_series_total_count_l55_55555


namespace length_PQ_is_5_l55_55259

/-
Given:
- Point P with coordinates (3, 4, 5)
- Point Q is the projection of P onto the xOy plane

Show:
- The length of the segment PQ is 5
-/

def P : ℝ × ℝ × ℝ := (3, 4, 5)
def Q : ℝ × ℝ × ℝ := (3, 4, 0)

theorem length_PQ_is_5 : dist P Q = 5 := by
  sorry

end length_PQ_is_5_l55_55259


namespace gp_sum_l55_55046

theorem gp_sum (x : ℕ) (h : (30 + x) / (10 + x) = (60 + x) / (30 + x)) :
  x = 30 ∧ (10 + x) + (30 + x) + (60 + x) + (120 + x) = 340 :=
by {
  sorry
}

end gp_sum_l55_55046


namespace polygon_sides_l55_55475

theorem polygon_sides (n : ℕ) (h : n - 1 = 2022) : n = 2023 :=
by
  sorry

end polygon_sides_l55_55475


namespace walter_age_at_2003_l55_55604

theorem walter_age_at_2003 :
  ∀ (w : ℕ),
  (1998 - w) + (1998 - 3 * w) = 3860 → 
  w + 5 = 39 :=
by
  intros w h
  sorry

end walter_age_at_2003_l55_55604


namespace root_in_interval_iff_a_range_l55_55211

def f (a x : ℝ) : ℝ := 2 * a * x ^ 2 + 2 * x - 3 - a

theorem root_in_interval_iff_a_range (a : ℝ) :
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f a x = 0) ↔ (1 ≤ a ∨ a ≤ - (3 + Real.sqrt 7) / 2) :=
sorry

end root_in_interval_iff_a_range_l55_55211


namespace square_area_l55_55682

theorem square_area : ∃ (s: ℝ), (∀ x: ℝ, x^2 + 4*x + 1 = 7 → ∃ t: ℝ, t = x ∧ ∃ x2: ℝ, (x2 - x)^2 = s^2 ∧ ∀ y : ℝ, y = 7 ∧ y = x2^2 + 4*x2 + 1) ∧ s^2 = 40 :=
by
  sorry

end square_area_l55_55682


namespace int_solution_l55_55003

theorem int_solution (n : ℕ) (h1 : n ≥ 1) (h2 : n^2 ∣ 2^n + 1) : n = 1 ∨ n = 3 :=
by
  sorry

end int_solution_l55_55003


namespace no_int_solutions_p_mod_4_neg_1_l55_55040

theorem no_int_solutions_p_mod_4_neg_1 :
  ∀ (p n : ℕ), (p % 4 = 3) → (∀ x y : ℕ, x^2 + y^2 ≠ p^n) :=
by
  intros
  sorry

end no_int_solutions_p_mod_4_neg_1_l55_55040


namespace visited_neither_l55_55028

theorem visited_neither (total : ℕ) (iceland : ℕ) (norway : ℕ) (both : ℕ) 
  (h1 : total = 100) 
  (h2 : iceland = 55) 
  (h3 : norway = 43) 
  (h4 : both = 61) : 
  (total - (iceland + norway - both)) = 63 := 
by 
  sorry

end visited_neither_l55_55028


namespace correlate_height_weight_l55_55925

-- Define the problems as types
def heightWeightCorrelated : Prop := true
def distanceTimeConstantSpeed : Prop := true
def heightVisionCorrelated : Prop := false
def volumeEdgeLengthCorrelated : Prop := true

-- Define the equivalence for the problem
def correlated : Prop := heightWeightCorrelated

-- Now state that correlated == heightWeightCorrelated
theorem correlate_height_weight : correlated = heightWeightCorrelated :=
by sorry

end correlate_height_weight_l55_55925


namespace min_distinct_lines_for_polyline_l55_55953

theorem min_distinct_lines_for_polyline (n : ℕ) (h_n : n = 31) : 
  ∃ (k : ℕ), 9 ≤ k ∧ k ≤ 31 ∧ 
  (∀ (s : Fin n → Fin 31), 
     ∀ i j, i ≠ j → s i ≠ s j) := 
sorry

end min_distinct_lines_for_polyline_l55_55953


namespace number_of_tiles_l55_55935

open Real

noncomputable def room_length : ℝ := 10
noncomputable def room_width : ℝ := 15
noncomputable def tile_length : ℝ := 5 / 12
noncomputable def tile_width : ℝ := 2 / 3

theorem number_of_tiles :
  (room_length * room_width) / (tile_length * tile_width) = 540 := by
  sorry

end number_of_tiles_l55_55935


namespace area_of_quadrilateral_l55_55498

theorem area_of_quadrilateral (A B C D H : Type) (AB BC : Real)
    (angle_ABC angle_ADC : Real) (BH h : Real)
    (H1 : AB = BC) (H2 : angle_ABC = 90 ∧ angle_ADC = 90)
    (H3 : BH = h) :
    (∃ area : Real, area = h^2) :=
by
  sorry

end area_of_quadrilateral_l55_55498


namespace central_symmetry_preserves_distance_l55_55650

variables {Point : Type} [MetricSpace Point]

def central_symmetry (O A A' B B' : Point) : Prop :=
  dist O A = dist O A' ∧ dist O B = dist O B'

theorem central_symmetry_preserves_distance {O A A' B B' : Point}
  (h : central_symmetry O A A' B B') : dist A B = dist A' B' :=
sorry

end central_symmetry_preserves_distance_l55_55650


namespace freshman_to_sophomore_ratio_l55_55103

variable (f s : ℕ)

-- Define the participants from freshmen and sophomores
def freshmen_participants : ℕ := (3 * f) / 7
def sophomores_participants : ℕ := (2 * s) / 3

-- Theorem: There are 14/9 times as many freshmen as sophomores
theorem freshman_to_sophomore_ratio (h : freshmen_participants f = sophomores_participants s) : 
  9 * f = 14 * s :=
by
  sorry

end freshman_to_sophomore_ratio_l55_55103


namespace no_rational_solution_l55_55806

/-- Prove that the only rational solution to the equation x^3 + 3y^3 + 9z^3 = 9xyz is x = y = z = 0. -/
theorem no_rational_solution : ∀ (x y z : ℚ), x^3 + 3 * y^3 + 9 * z^3 = 9 * x * y * z → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intro x y z h
  sorry

end no_rational_solution_l55_55806


namespace find_Sn_find_Tn_l55_55543

def Sn (n : ℕ) : ℕ := n^2 + n

def Tn (n : ℕ) : ℚ := (n : ℚ) / (n + 1)

section
variables {a₁ d : ℕ}

-- Given conditions
axiom S5 : 5 * a₁ + 10 * d = 30
axiom S10 : 10 * a₁ + 45 * d = 110

-- Problem statement 1
theorem find_Sn (n : ℕ) : Sn n = n^2 + n :=
sorry

-- Problem statement 2
theorem find_Tn (n : ℕ) : Tn n = (n : ℚ) / (n + 1) :=
sorry

end

end find_Sn_find_Tn_l55_55543


namespace acute_angle_sine_l55_55590
--import Lean library

-- Define the problem conditions and statement
theorem acute_angle_sine (a : ℝ) (h1 : 0 < a) (h2 : a < π / 2) (h3 : Real.sin a = 0.6) :
  π / 6 < a ∧ a < π / 4 :=
by 
  sorry

end acute_angle_sine_l55_55590


namespace fraction_meaningful_iff_l55_55418

theorem fraction_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 := sorry

end fraction_meaningful_iff_l55_55418


namespace sum_of_integers_l55_55034

theorem sum_of_integers (n : ℤ) (h : n * (n + 2) = 20400) : n + (n + 2) = 286 ∨ n + (n + 2) = -286 :=
by
  sorry

end sum_of_integers_l55_55034


namespace inverse_variation_l55_55456

theorem inverse_variation (a b k : ℝ) (h1 : a * b^3 = k) (h2 : 8 * 1^3 = k) : (∃ a, b = 4 → a = 1 / 8) :=
by
  sorry

end inverse_variation_l55_55456


namespace at_least_one_inequality_false_l55_55574

open Classical

theorem at_least_one_inequality_false (a b c d : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) :
  ¬ (a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) :=
by
  sorry

end at_least_one_inequality_false_l55_55574


namespace chairlift_halfway_l55_55717

theorem chairlift_halfway (total_chairs current_chair halfway_chair : ℕ) 
  (h_total_chairs : total_chairs = 96)
  (h_current_chair : current_chair = 66) : halfway_chair = 18 :=
sorry

end chairlift_halfway_l55_55717


namespace defect_rate_product_l55_55372

theorem defect_rate_product (P1_defect P2_defect : ℝ) (h1 : P1_defect = 0.10) (h2 : P2_defect = 0.03) : 
  ((1 - P1_defect) * (1 - P2_defect)) = 0.873 → (1 - ((1 - P1_defect) * (1 - P2_defect)) = 0.127) :=
by
  intro h
  sorry

end defect_rate_product_l55_55372


namespace third_month_sale_l55_55301

theorem third_month_sale (s3 : ℝ)
  (s1 s2 s4 s5 s6 : ℝ)
  (h1 : s1 = 2435)
  (h2 : s2 = 2920)
  (h4 : s4 = 3230)
  (h5 : s5 = 2560)
  (h6 : s6 = 1000)
  (average : (s1 + s2 + s3 + s4 + s5 + s6) / 6 = 2500) :
  s3 = 2855 := 
by sorry

end third_month_sale_l55_55301


namespace ratio_is_9_l55_55097

-- Define the set of numbers
def set_of_numbers := { x : ℕ | ∃ n, n ≤ 8 ∧ x = 10^n }

-- Define the sum of the geometric series excluding the largest element
def sum_of_others : ℕ := (Finset.range 8).sum (λ n => 10^n)

-- Define the largest element
def largest_element := 10^8

-- Define the ratio of the largest element to the sum of the other elements
def ratio := largest_element / sum_of_others

-- Problem statement: The ratio is 9
theorem ratio_is_9 : ratio = 9 := by
  sorry

end ratio_is_9_l55_55097


namespace sum_of_edges_96_l55_55272

noncomputable def volume (a r : ℝ) : ℝ := 
  (a / r) * a * (a * r)

noncomputable def surface_area (a r : ℝ) : ℝ := 
  2 * ((a^2) / r + a^2 + a^2 * r)

noncomputable def sum_of_edges (a r : ℝ) : ℝ := 
  4 * ((a / r) + a + (a * r))

theorem sum_of_edges_96 :
  (∃ (a r : ℝ), volume a r = 512 ∧ surface_area a r = 384 ∧ sum_of_edges a r = 96) :=
by
  have a := 8
  have r := 1
  have h_volume : volume a r = 512 := sorry
  have h_surface_area : surface_area a r = 384 := sorry
  have h_sum_of_edges : sum_of_edges a r = 96 := sorry
  exact ⟨a, r, h_volume, h_surface_area, h_sum_of_edges⟩

end sum_of_edges_96_l55_55272


namespace molecular_weight_of_10_moles_of_Al2S3_l55_55940

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_S : ℝ := 32.06

-- Define the molecular weight calculation for Al2S3
def molecular_weight_Al2S3 : ℝ :=
  (2 * atomic_weight_Al) + (3 * atomic_weight_S)

-- Define the molecular weight for 10 moles of Al2S3
def molecular_weight_10_moles_Al2S3 : ℝ :=
  10 * molecular_weight_Al2S3

-- The theorem to prove
theorem molecular_weight_of_10_moles_of_Al2S3 :
  molecular_weight_10_moles_Al2S3 = 1501.4 :=
by
  -- skip the proof
  sorry

end molecular_weight_of_10_moles_of_Al2S3_l55_55940


namespace hoseok_value_l55_55008

theorem hoseok_value (x : ℕ) (h : x - 10 = 15) : x + 5 = 30 :=
by
  sorry

end hoseok_value_l55_55008


namespace circle_equation_l55_55680

theorem circle_equation (x y : ℝ) (h1 : (1 - 1)^2 + (1 - 1)^2 = 2) (h2 : (0 - 1)^2 + (0 - 1)^2 = r_sq) :
  (x - 1)^2 + (y - 1)^2 = 2 :=
sorry

end circle_equation_l55_55680


namespace min_ab_bound_l55_55934

theorem min_ab_bound (a b n : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_n : 0 < n) 
                      (h : ∀ i j, i ≤ n → j ≤ n → Nat.gcd (a + i) (b + j) > 1) :
  ∃ c > 0, min a b > c^n * n^(n/2) :=
sorry

end min_ab_bound_l55_55934


namespace breakable_iff_composite_l55_55698

-- Definitions directly from the problem conditions
def is_breakable (n : ℕ) : Prop :=
  ∃ (a b x y : ℕ), a > 0 ∧ b > 0 ∧ x > 0 ∧ y > 0 ∧ a + b = n ∧ (x / a : ℚ) + (y / b : ℚ) = 1

def is_composite (n : ℕ) : Prop :=
  ∃ (s t : ℕ), s > 1 ∧ t > 1 ∧ n = s * t

-- The proof statement
theorem breakable_iff_composite (n : ℕ) : is_breakable n ↔ is_composite n := sorry

end breakable_iff_composite_l55_55698


namespace solution_to_g_inv_2_l55_55162

noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := 1 / (c * x + d)

theorem solution_to_g_inv_2 (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
    ∃ x : ℝ, g x c d = 2 ↔ x = (1 - 2 * d) / (2 * c) :=
by
  sorry

end solution_to_g_inv_2_l55_55162


namespace rainfall_on_wednesday_l55_55481

theorem rainfall_on_wednesday 
  (rain_on_monday : ℝ)
  (rain_on_tuesday : ℝ)
  (total_rain : ℝ) 
  (hmonday : rain_on_monday = 0.16666666666666666) 
  (htuesday : rain_on_tuesday = 0.4166666666666667) 
  (htotal : total_rain = 0.6666666666666666) :
  total_rain - (rain_on_monday + rain_on_tuesday) = 0.0833333333333333 :=
by
  -- Proof would go here
  sorry

end rainfall_on_wednesday_l55_55481


namespace total_molecular_weight_l55_55616

-- Define atomic weights
def atomic_weight (element : String) : Float :=
  match element with
  | "K"  => 39.10
  | "Cr" => 51.996
  | "O"  => 16.00
  | "Fe" => 55.845
  | "S"  => 32.07
  | "Mn" => 54.938
  | _    => 0.0

-- Molecular weights of compounds
def molecular_weight_K2Cr2O7 : Float := 
  2 * atomic_weight "K" + 2 * atomic_weight "Cr" + 7 * atomic_weight "O"

def molecular_weight_Fe2_SO4_3 : Float := 
  2 * atomic_weight "Fe" + 3 * atomic_weight "S" + 12 * atomic_weight "O"

def molecular_weight_KMnO4 : Float := 
  atomic_weight "K" + atomic_weight "Mn" + 4 * atomic_weight "O"

-- Proof statement 
theorem total_molecular_weight :
  4 * molecular_weight_K2Cr2O7 + 3 * molecular_weight_Fe2_SO4_3 + 5 * molecular_weight_KMnO4 = 3166.658 :=
by
  sorry

end total_molecular_weight_l55_55616


namespace extreme_values_a_1_turning_point_a_8_l55_55792

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 - (a + 2) * x + a * Real.log x

def turning_point (g : ℝ → ℝ) (P : ℝ × ℝ) (h : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), x ≠ P.1 → (g x - h x) / (x - P.1) > 0

theorem extreme_values_a_1 :
  (∀ (x : ℝ), f x 1 ≤ f (1/2) 1 → f x 1 = f (1/2) 1) ∧ (∀ (x : ℝ), f x 1 ≥ f 1 1 → f x 1 = f 1 1) :=
sorry

theorem turning_point_a_8 :
  ∀ (x₀ : ℝ), x₀ = 2 → turning_point (f · 8) (x₀, f x₀ 8) (λ x => (2 * x₀ + 8 / x₀ - 10) * (x - x₀) + x₀^2 - 10 * x₀ + 8 * Real.log x₀) :=
sorry

end extreme_values_a_1_turning_point_a_8_l55_55792


namespace roger_individual_pouches_per_pack_l55_55078

variable (members : ℕ) (coaches : ℕ) (helpers : ℕ) (packs : ℕ)

-- Given conditions
def total_people (members coaches helpers : ℕ) : ℕ := members + coaches + helpers
def pouches_per_pack (total_people packs : ℕ) : ℕ := total_people / packs

-- Specific values from the problem
def roger_total_people : ℕ := total_people 13 3 2
def roger_packs : ℕ := 3

-- The problem statement to prove:
theorem roger_individual_pouches_per_pack : pouches_per_pack roger_total_people roger_packs = 6 :=
by
  sorry

end roger_individual_pouches_per_pack_l55_55078


namespace minimum_jumps_l55_55222

theorem minimum_jumps (a b : ℕ) (h : 2 * a + 3 * b = 2016) : a + b = 673 :=
sorry

end minimum_jumps_l55_55222


namespace conclusion_A_conclusion_B_conclusion_C1_conclusion_C2_l55_55542

variable {r a b x1 y1 x2 y2 : ℝ} -- variables used in the problem

-- conditions
def circle1 : x1^2 + y1^2 = r^2 := sorry -- Circle C1 equation
def circle2 : (x1 + a)^2 + (y1 + b)^2 = r^2 := sorry -- Circle C2 equation
def r_positive : r > 0 := sorry -- r > 0
def not_both_zero : ¬ (a = 0 ∧ b = 0) := sorry -- a, b are not both zero
def distinct_points : x1 ≠ x2 ∧ y1 ≠ y2 := sorry -- A(x1, y1) and B(x2, y2) are distinct

-- Proofs to be provided for each of the conclusions
theorem conclusion_A : 2 * a * x1 + 2 * b * y1 + a^2 + b^2 = 0 := sorry
theorem conclusion_B : a * (x1 - x2) + b * (y1 - y2) = 0 := sorry
theorem conclusion_C1 : x1 + x2 = -a := sorry
theorem conclusion_C2 : y1 + y2 = -b := sorry

end conclusion_A_conclusion_B_conclusion_C1_conclusion_C2_l55_55542


namespace flour_masses_l55_55287

theorem flour_masses (x : ℝ) (h: 
    (x * (1 + x / 100) + (x + 10) * (1 + (x + 10) / 100) = 112.5)) :
    x = 35 ∧ (x + 10) = 45 :=
by 
  sorry

end flour_masses_l55_55287


namespace sum_of_first_110_terms_l55_55464

theorem sum_of_first_110_terms
  (a d : ℝ)
  (h1 : (10 : ℝ) * (2 * a + (10 - 1) * d) / 2 = 100)
  (h2 : (100 : ℝ) * (2 * a + (100 - 1) * d) / 2 = 10) :
  (110 : ℝ) * (2 * a + (110 - 1) * d) / 2 = -110 :=
  sorry

end sum_of_first_110_terms_l55_55464


namespace pencils_placed_by_dan_l55_55319

-- Definitions based on the conditions provided
def pencils_in_drawer : ℕ := 43
def initial_pencils_on_desk : ℕ := 19
def new_total_pencils : ℕ := 78

-- The statement to be proven
theorem pencils_placed_by_dan : pencils_in_drawer + initial_pencils_on_desk + 16 = new_total_pencils :=
by
  sorry

end pencils_placed_by_dan_l55_55319


namespace find_xyz_l55_55821

theorem find_xyz (x y z : ℝ) :
  x - y + z = 2 ∧
  x^2 + y^2 + z^2 = 30 ∧
  x^3 - y^3 + z^3 = 116 →
  (x = -1 ∧ y = 2 ∧ z = 5) ∨
  (x = -1 ∧ y = -5 ∧ z = -2) ∨
  (x = -2 ∧ y = 1 ∧ z = 5) ∨
  (x = -2 ∧ y = -5 ∧ z = -1) ∨
  (x = 5 ∧ y = 1 ∧ z = -2) ∨
  (x = 5 ∧ y = 2 ∧ z = -1) := by
  sorry

end find_xyz_l55_55821


namespace compare_exponents_l55_55913

def a : ℝ := 2^(4/3)
def b : ℝ := 4^(2/5)
def c : ℝ := 25^(1/3)

theorem compare_exponents : b < a ∧ a < c :=
by
  have h1 : a = 2^(4/3) := rfl
  have h2 : b = 4^(2/5) := rfl
  have h3 : c = 25^(1/3) := rfl
  -- These are used to indicate the definitions, not the proof steps
  sorry

end compare_exponents_l55_55913


namespace train_speed_l55_55443

def length_of_train : ℝ := 250
def length_of_bridge : ℝ := 120
def time_taken : ℝ := 20
noncomputable def total_distance : ℝ := length_of_train + length_of_bridge
noncomputable def speed_of_train : ℝ := total_distance / time_taken

theorem train_speed : speed_of_train = 18.5 :=
  by sorry

end train_speed_l55_55443


namespace sym_coords_origin_l55_55119

theorem sym_coords_origin (a b : ℝ) (h : |a - 3| + (b + 4)^2 = 0) :
  (-a, -b) = (-3, 4) :=
sorry

end sym_coords_origin_l55_55119


namespace pyramid_cross_section_distance_l55_55908

theorem pyramid_cross_section_distance
  (area1 area2 : ℝ) (distance : ℝ)
  (h1 : area1 = 100 * Real.sqrt 3) 
  (h2 : area2 = 225 * Real.sqrt 3) 
  (h3 : distance = 5) : 
  ∃ h : ℝ, h = 15 :=
by
  sorry

end pyramid_cross_section_distance_l55_55908


namespace sum_of_angles_of_solutions_l55_55805

theorem sum_of_angles_of_solutions : 
  ∀ (z : ℂ), z^5 = 32 * Complex.I → ∃ θs : Fin 5 → ℝ, 
  (∀ k, 0 ≤ θs k ∧ θs k < 360) ∧ (θs 0 + θs 1 + θs 2 + θs 3 + θs 4 = 810) :=
by
  sorry

end sum_of_angles_of_solutions_l55_55805


namespace larger_circle_radius_l55_55692

theorem larger_circle_radius (r R : ℝ) 
  (h : (π * R^2) / (π * r^2) = 5 / 2) : 
  R = r * Real.sqrt 2.5 :=
sorry

end larger_circle_radius_l55_55692


namespace value_of_expression_l55_55021

theorem value_of_expression (x y z : ℝ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4) : 
  (x - 2 * y + 3 * z) / (x + y + z) = 8 / 9 := 
  sorry

end value_of_expression_l55_55021


namespace number_of_positive_integers_with_positive_log_l55_55446

theorem number_of_positive_integers_with_positive_log (b : ℕ) (h : ∃ n : ℕ, n > 0 ∧ b ^ n = 1024) : 
  ∃ L, L = 4 :=
sorry

end number_of_positive_integers_with_positive_log_l55_55446


namespace arithmetic_sequence_sum_l55_55902

theorem arithmetic_sequence_sum (S_n : ℕ → ℕ) (a_n : ℕ → ℕ) (c : ℤ) :
  (∀ n : ℕ, 0 < n → S_n n = n^2 + c) →
  a_n 1 = 1 + c →
  (∀ n, 1 < n → a_n n = S_n n - S_n (n - 1)) →
  (∀ n : ℕ, 0 < n → a_n n = 1 + (n - 1) * 2) →
  c = 0 ∧ (∀ n : ℕ, 0 < n → a_n n = 2 * n - 1) :=
by
  sorry

end arithmetic_sequence_sum_l55_55902


namespace inequality_and_equality_conditions_l55_55933

theorem inequality_and_equality_conditions (a b c : ℝ) 
  (h : (a + 1) * (b + 1) * (c + 1) = 8) :
  a + b + c ≥ 3 ∧ abc ≤ 1 ∧ ((a + b + c = 3) → (a = 1 ∧ b = 1 ∧ c = 1)) := 
by 
  sorry

end inequality_and_equality_conditions_l55_55933


namespace total_pears_after_giving_away_l55_55316

def alyssa_pears : ℕ := 42
def nancy_pears : ℕ := 17
def carlos_pears : ℕ := 25
def pears_given_away_per_person : ℕ := 5

theorem total_pears_after_giving_away :
  (alyssa_pears + nancy_pears + carlos_pears) - (3 * pears_given_away_per_person) = 69 :=
by
  sorry

end total_pears_after_giving_away_l55_55316


namespace average_height_is_64_l55_55891

noncomputable def Parker (H_D : ℝ) : ℝ := H_D - 4
noncomputable def Daisy (H_R : ℝ) : ℝ := H_R + 8
noncomputable def Reese : ℝ := 60

theorem average_height_is_64 :
  let H_R := Reese 
  let H_D := Daisy H_R
  let H_P := Parker H_D
  (H_P + H_D + H_R) / 3 = 64 := sorry

end average_height_is_64_l55_55891


namespace even_function_l55_55733

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

def f (x : ℝ) : ℝ := (x + 2)^2 + (2 * x - 1)^2

theorem even_function : is_even_function f :=
by
  sorry

end even_function_l55_55733


namespace b_remaining_work_days_l55_55325

-- Definitions of the conditions
def together_work (a b: ℕ) := a + b = 12
def alone_work (a: ℕ) := a = 20
def c_work (c: ℕ) := c = 30
def initial_work_days := 5

-- Question to prove:
theorem b_remaining_work_days (a b c : ℕ) (h1 : together_work a b) (h2 : alone_work a) (h3 : c_work c) : 
  let b_rate := 1 / 30 
  let remaining_work := 25 / 60
  let work_to_days := remaining_work / b_rate
  work_to_days = 12.5 := 
sorry

end b_remaining_work_days_l55_55325


namespace sum_a_b_l55_55839

theorem sum_a_b (a b : ℚ) (h1 : a + 3 * b = 27) (h2 : 5 * a + 2 * b = 40) : a + b = 161 / 13 :=
  sorry

end sum_a_b_l55_55839


namespace problem_1_split_terms_problem_2_split_terms_l55_55636

-- Problem 1 Lean statement
theorem problem_1_split_terms :
  (28 + 5/7) + (-25 - 1/7) = 3 + 4/7 := 
  sorry
  
-- Problem 2 Lean statement
theorem problem_2_split_terms :
  (-2022 - 2/7) + (-2023 - 4/7) + 4046 - 1/7 = 0 := 
  sorry

end problem_1_split_terms_problem_2_split_terms_l55_55636


namespace strawberry_growth_rate_l55_55483

theorem strawberry_growth_rate
  (initial_plants : ℕ)
  (months : ℕ)
  (plants_given_away : ℕ)
  (total_plants_after : ℕ)
  (growth_rate : ℕ)
  (h_initial : initial_plants = 3)
  (h_months : months = 3)
  (h_given_away : plants_given_away = 4)
  (h_total_after : total_plants_after = 20)
  (h_equation : initial_plants + growth_rate * months - plants_given_away = total_plants_after) :
  growth_rate = 7 :=
sorry

end strawberry_growth_rate_l55_55483


namespace problem_part1_problem_part2_l55_55273

theorem problem_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
  a + b + c ≥ 1 / Real.sqrt a + 1 / Real.sqrt b + 1 / Real.sqrt c := 
sorry

theorem problem_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
  a^2 + b^2 + c^2 ≥ Real.sqrt a + Real.sqrt b + Real.sqrt c :=
sorry

end problem_part1_problem_part2_l55_55273


namespace repeating_decimal_eq_fraction_l55_55348

-- Define the repeating decimal 0.363636... as a limit of its geometric series representation
noncomputable def repeating_decimal := ∑' n : ℕ, (36 / 100^(n + 1))

-- Define the fraction
def fraction := 4 / 11

theorem repeating_decimal_eq_fraction : repeating_decimal = fraction := by sorry

end repeating_decimal_eq_fraction_l55_55348


namespace average_difference_is_7_l55_55611

/-- The differences between Mia's and Liam's study times for each day in one week -/
def daily_differences : List ℤ := [15, -5, 25, 0, -15, 20, 10]

/-- The number of days in a week -/
def number_of_days : ℕ := 7

/-- The total difference over the week -/
def total_difference : ℤ := daily_differences.sum

/-- The average difference per day -/
def average_difference_per_day : ℚ := total_difference / number_of_days

theorem average_difference_is_7 : average_difference_per_day = 7 := by 
  sorry

end average_difference_is_7_l55_55611


namespace minimum_m_value_l55_55540

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem minimum_m_value :
  (∃ m, ∀ (x1 x2 x3 : ℝ), 0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 ≤ Real.pi → |f x1 - f x2| + |f x2 - f x3| ≤ m) ∧
  ∀ m', (∀ (x1 x2 x3 : ℝ), 0 ≤ x1 ∧ x1 < x2 ∧ x2 < x3 ∧ x3 ≤ Real.pi → |f x1 - f x2| + |f x2 - f x3| ≤ m') → 3 + Real.sqrt 3 / 2 ≤ m' :=
by
  sorry

end minimum_m_value_l55_55540


namespace cuts_needed_l55_55132

-- Define the length of the wood in centimeters
def wood_length_cm : ℕ := 400

-- Define the length of each stake in centimeters
def stake_length_cm : ℕ := 50

-- Define the expected number of cuts needed
def expected_cuts : ℕ := 7

-- The main theorem stating the equivalence
theorem cuts_needed (wood_length stake_length : ℕ) (h1 : wood_length = 400) (h2 : stake_length = 50) :
  (wood_length / stake_length) - 1 = expected_cuts :=
sorry

end cuts_needed_l55_55132


namespace quadratic_ineq_solution_l55_55755

theorem quadratic_ineq_solution (x : ℝ) : x^2 - x - 6 < 0 ↔ -2 < x ∧ x < 3 := 
sorry

end quadratic_ineq_solution_l55_55755


namespace smallest_x_solution_l55_55580

theorem smallest_x_solution :
  (∃ x : ℚ, abs (4 * x + 3) = 30 ∧ ∀ y : ℚ, abs (4 * y + 3) = 30 → x ≤ y) ↔ x = -33 / 4 := by
  sorry

end smallest_x_solution_l55_55580


namespace equation_is_linear_l55_55854

-- Define the conditions and the proof statement
theorem equation_is_linear (m n : ℕ) : 3 * x ^ (2 * m + 1) - 2 * y ^ (n - 1) = 7 → (2 * m + 1 = 1) ∧ (n - 1 = 1) → m = 0 ∧ n = 2 :=
by
  sorry

end equation_is_linear_l55_55854


namespace prob_triangle_includes_G_l55_55122

-- Definitions based on conditions in the problem
def total_triangles : ℕ := 6
def triangles_including_G : ℕ := 4

-- The theorem statement proving the probability
theorem prob_triangle_includes_G : (triangles_including_G : ℚ) / total_triangles = 2 / 3 :=
by
  sorry

end prob_triangle_includes_G_l55_55122


namespace convert_base_3_to_base_10_l55_55915

theorem convert_base_3_to_base_10 : 
  (1 * 3^4 + 2 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0) = 142 :=
by
  sorry

end convert_base_3_to_base_10_l55_55915


namespace publishing_company_break_even_l55_55470

theorem publishing_company_break_even : 
  ∀ (F V P : ℝ) (x : ℝ), F = 35630 ∧ V = 11.50 ∧ P = 20.25 →
  (P * x = F + V * x) → x = 4074 :=
by
  intros F V P x h_eq h_rev
  sorry

end publishing_company_break_even_l55_55470


namespace volunteer_distribution_l55_55321

theorem volunteer_distribution :
  let students := 5
  let projects := 4
  let combinations := Nat.choose students 2
  let permutations := Nat.factorial projects
  combinations * permutations = 240 := 
by
  sorry

end volunteer_distribution_l55_55321


namespace distance_after_rest_l55_55371

-- Define the conditions
def distance_before_rest := 0.75
def total_distance := 1.0

-- State the theorem
theorem distance_after_rest :
  total_distance - distance_before_rest = 0.25 :=
by sorry

end distance_after_rest_l55_55371


namespace corrected_observations_mean_l55_55004

noncomputable def corrected_mean (mean incorrect correct: ℚ) (n: ℕ) : ℚ :=
  let S_incorrect := mean * n
  let Difference := correct - incorrect
  let S_corrected := S_incorrect + Difference
  S_corrected / n

theorem corrected_observations_mean:
  corrected_mean 36 23 34 50 = 36.22 := by
  sorry

end corrected_observations_mean_l55_55004


namespace arithmetic_sequence_sum_l55_55662

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ)
  (h1 : ∀ n, S n = n * ((a 1 + a n) / 2))
  (h2 : S 9 = 27) :
  a 4 + a 6 = 6 := 
sorry

end arithmetic_sequence_sum_l55_55662


namespace Q1_Q2_l55_55824

noncomputable def prob_A_scores_3_out_of_4 (p_A_serves : ℚ) (p_A_scores_A_serves: ℚ) (p_A_scores_B_serves: ℚ) : ℚ :=
  by
    -- Placeholder probability function
    sorry

theorem Q1 (p_A_serves : ℚ := 2/3) (p_A_scores_A_serves: ℚ := 2/3) (p_A_scores_B_serves: ℚ := 1/2) :
  prob_A_scores_3_out_of_4 p_A_serves p_A_scores_A_serves p_A_scores_B_serves = 1/3 :=
  by
    -- Proof of the theorem
    sorry

noncomputable def prob_X_lessthan_or_equal_4 (p_A_serves: ℚ) (p_A_scores_A_serves: ℚ) (p_A_scores_B_serves: ℚ) : ℚ :=
  by
    -- Placeholder probability function
    sorry

theorem Q2 (p_A_serves: ℚ := 2/3) (p_A_scores_A_serves: ℚ := 2/3) (p_A_scores_B_serves: ℚ := 1/2) :
  prob_X_lessthan_or_equal_4 p_A_serves p_A_scores_A_serves p_A_scores_B_serves = 3/4 :=
  by
    -- Proof of the theorem
    sorry

end Q1_Q2_l55_55824


namespace total_cubes_proof_l55_55186

def Grady_initial_red_cubes := 20
def Grady_initial_blue_cubes := 15
def Gage_initial_red_cubes := 10
def Gage_initial_blue_cubes := 12
def Harper_initial_red_cubes := 8
def Harper_initial_blue_cubes := 10

def Gage_red_received := (2 / 5) * Grady_initial_red_cubes
def Gage_blue_received := (1 / 3) * Grady_initial_blue_cubes

def Grady_red_after_Gage := Grady_initial_red_cubes - Gage_red_received
def Grady_blue_after_Gage := Grady_initial_blue_cubes - Gage_blue_received

def Harper_red_received := (1 / 4) * Grady_red_after_Gage
def Harper_blue_received := (1 / 2) * Grady_blue_after_Gage

def Gage_total_red := Gage_initial_red_cubes + Gage_red_received
def Gage_total_blue := Gage_initial_blue_cubes + Gage_blue_received

def Harper_total_red := Harper_initial_red_cubes + Harper_red_received
def Harper_total_blue := Harper_initial_blue_cubes + Harper_blue_received

def Gage_total_cubes := Gage_total_red + Gage_total_blue
def Harper_total_cubes := Harper_total_red + Harper_total_blue

def Gage_Harper_total_cubes := Gage_total_cubes + Harper_total_cubes

theorem total_cubes_proof : Gage_Harper_total_cubes = 61 := by
  sorry

end total_cubes_proof_l55_55186


namespace ratio_of_scores_l55_55684

theorem ratio_of_scores 
  (u v : ℝ) 
  (h1 : u > v) 
  (h2 : u - v = (u + v) / 2) 
  : v / u = 1 / 3 :=
sorry

end ratio_of_scores_l55_55684


namespace theater_ticket_sales_l55_55491

-- Definitions of the given constants and initialization
def R : ℕ := 25

-- Conditions based on the problem statement
def condition_horror (H : ℕ) := H = 3 * R + 18
def condition_action (A : ℕ) := A = 2 * R
def condition_comedy (C H : ℕ) := 4 * H = 5 * C

-- Desired outcomes based on the solutions
def desired_horror := 93
def desired_action := 50
def desired_comedy := 74

theorem theater_ticket_sales
  (H A C : ℕ)
  (h1 : condition_horror H)
  (h2 : condition_action A)
  (h3 : condition_comedy C H)
  : H = desired_horror ∧ A = desired_action ∧ C = desired_comedy :=
by {
    sorry
}

end theater_ticket_sales_l55_55491


namespace arithmetic_seq_sum_l55_55632

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h₁ : ∀ n k : ℕ, a (n + k) = a n + k * d) 
  (h₂ : a 5 + a 6 + a 7 + a 8 = 20) : a 1 + a 12 = 10 := 
by 
  sorry

end arithmetic_seq_sum_l55_55632


namespace base7_addition_l55_55230

theorem base7_addition (X Y : ℕ) (h1 : Y + 2 = X) (h2 : X + 5 = 8) : X + Y = 4 :=
by
  sorry

end base7_addition_l55_55230


namespace min_max_values_l55_55791

noncomputable def f (x : ℝ) : ℝ := 1 + 3 * x - x^3

theorem min_max_values : 
  (∃ x : ℝ, f x = -1) ∧ (∃ x : ℝ, f x = 3) :=
by
  sorry

end min_max_values_l55_55791


namespace find_one_third_of_product_l55_55745

theorem find_one_third_of_product : (1 / 3) * (7 * 9) = 21 :=
by
  -- The proof will be filled here
  sorry

end find_one_third_of_product_l55_55745


namespace inequality_proof_l55_55085

theorem inequality_proof (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a > b) : 
  1 / (a * b^2) > 1 / (a^2 * b) :=
sorry

end inequality_proof_l55_55085


namespace find_C_l55_55330

theorem find_C (A B C : ℕ) (h0 : 3 * A - A = 10) (h1 : B + A = 12) (h2 : C - B = 6) (h3 : A ≠ B) (h4 : B ≠ C) (h5 : C ≠ A) 
: C = 13 :=
sorry

end find_C_l55_55330


namespace vacation_cost_in_usd_l55_55866

theorem vacation_cost_in_usd :
  let n := 7
  let rent_per_person_eur := 65
  let transport_per_person_usd := 25
  let food_per_person_gbp := 50
  let activities_per_person_jpy := 2750
  let eur_to_usd := 1.20
  let gbp_to_usd := 1.40
  let jpy_to_usd := 0.009
  let total_rent_usd := n * rent_per_person_eur * eur_to_usd
  let total_transport_usd := n * transport_per_person_usd
  let total_food_usd := n * food_per_person_gbp * gbp_to_usd
  let total_activities_usd := n * activities_per_person_jpy * jpy_to_usd
  let total_cost_usd := total_rent_usd + total_transport_usd + total_food_usd + total_activities_usd
  total_cost_usd = 1384.25 := by
    sorry

end vacation_cost_in_usd_l55_55866


namespace building_height_l55_55377

noncomputable def height_of_building (flagpole_height shadow_of_flagpole shadow_of_building : ℝ) : ℝ :=
  (flagpole_height / shadow_of_flagpole) * shadow_of_building

theorem building_height : height_of_building 18 45 60 = 24 := by {
  sorry
}

end building_height_l55_55377


namespace maximum_profit_l55_55413

def radioactive_marble_problem : ℕ :=
    let total_marbles := 100
    let radioactive_marbles := 1
    let non_radioactive_profit := 1
    let measurement_cost := 1
    let max_profit := 92 
    max_profit

theorem maximum_profit 
    (total_marbles : ℕ := 100) 
    (radioactive_marbles : ℕ := 1) 
    (non_radioactive_profit : ℕ := 1) 
    (measurement_cost : ℕ := 1) :
    radioactive_marble_problem = 92 :=
by sorry

end maximum_profit_l55_55413


namespace five_consecutive_product_div_24_l55_55163

theorem five_consecutive_product_div_24 (n : ℤ) : 
  24 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) := 
sorry

end five_consecutive_product_div_24_l55_55163


namespace reciprocal_of_3_div_2_l55_55041

def reciprocal (a : ℚ) : ℚ := a⁻¹

theorem reciprocal_of_3_div_2 : reciprocal (3 / 2) = 2 / 3 :=
by
  -- proof would go here
  sorry

end reciprocal_of_3_div_2_l55_55041


namespace ratio_of_M_to_N_l55_55145

theorem ratio_of_M_to_N 
  (M Q P N : ℝ) 
  (h1 : M = 0.4 * Q) 
  (h2 : Q = 0.25 * P) 
  (h3 : N = 0.75 * P) : 
  M / N = 2 / 15 := 
sorry

end ratio_of_M_to_N_l55_55145


namespace paint_coverage_is_10_l55_55528

noncomputable def paintCoverage (cost_per_quart : ℝ) (cube_edge_length : ℝ) (total_cost : ℝ) : ℝ :=
  let total_surface_area := 6 * (cube_edge_length ^ 2)
  let number_of_quarts := total_cost / cost_per_quart
  total_surface_area / number_of_quarts

theorem paint_coverage_is_10 :
  paintCoverage 3.2 10 192 = 10 :=
by
  sorry

end paint_coverage_is_10_l55_55528


namespace work_completion_problem_l55_55442

theorem work_completion_problem :
  (∃ x : ℕ, 9 * (1 / 45 + 1 / x) + 23 * (1 / x) = 1) → x = 40 :=
sorry

end work_completion_problem_l55_55442


namespace class_total_students_l55_55826

-- Definitions based on the conditions
def number_students_group : ℕ := 12
def frequency_group : ℚ := 0.25

-- Statement of the problem in Lean
theorem class_total_students (n : ℕ) (h : frequency_group = number_students_group / n) : n = 48 :=
by
  sorry

end class_total_students_l55_55826


namespace total_cartons_packed_l55_55578

-- Define the given conditions
def cans_per_carton : ℕ := 20
def cartons_loaded : ℕ := 40
def cans_left : ℕ := 200

-- Formalize the proof problem
theorem total_cartons_packed : cartons_loaded + (cans_left / cans_per_carton) = 50 := by
  sorry

end total_cartons_packed_l55_55578


namespace total_number_recruits_l55_55252

theorem total_number_recruits 
  (x y z : ℕ)
  (h1 : x = 50)
  (h2 : y = 100)
  (h3 : z = 170)
  (h4 : x = 4 * (y - 50) ∨ y = 4 * (z - 170) ∨ x = 4 * (z - 170)) : 
  171 + (z - 170) = 211 :=
by
  sorry

end total_number_recruits_l55_55252


namespace simplify_and_evaluate_l55_55551

variable (a : ℚ)
variable (a_val : a = -1/2)

theorem simplify_and_evaluate : (4 - 3 * a) * (1 + 2 * a) - 3 * a * (1 - 2 * a) = 3 := by
  sorry

end simplify_and_evaluate_l55_55551


namespace g_g1_eq_43_l55_55352

def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 1

theorem g_g1_eq_43 : g (g 1) = 43 :=
by
  sorry

end g_g1_eq_43_l55_55352


namespace mean_equal_implication_l55_55137

theorem mean_equal_implication (y : ℝ) :
  (7 + 10 + 15 + 23 = 55) →
  (55 / 4 = 13.75) →
  (18 + y + 30 = 48 + y) →
  (48 + y) / 3 = 13.75 →
  y = -6.75 :=
by 
  intros h1 h2 h3 h4
  -- The steps would be applied here to prove y = -6.75
  sorry

end mean_equal_implication_l55_55137


namespace find_m_l55_55605

theorem find_m (m : ℝ) (h1 : ∀ x y : ℝ, (x ^ 2 + (y - 2) ^ 2 = 1) → (y = x / m ∨ y = -x / m)) (h2 : 0 < m) :
  m = (Real.sqrt 3) / 3 :=
by
  sorry

end find_m_l55_55605


namespace problem_proof_l55_55553

noncomputable def f : ℝ → ℝ := sorry

theorem problem_proof (h1 : ∀ x : ℝ, f (-x) = f x)
    (h2 : ∀ x y : ℝ, x < y ∧ y ≤ -1 → f x < f y) : 
    f 2 < f (-3 / 2) ∧ f (-3 / 2) < f (-1) :=
by
  sorry

end problem_proof_l55_55553


namespace range_of_x_l55_55490

theorem range_of_x (x : ℝ) : (6 - 2 * x) ≠ 0 ↔ x ≠ 3 := 
by {
  sorry
}

end range_of_x_l55_55490


namespace difference_five_three_numbers_specific_number_condition_l55_55716

def is_five_three_number (A : ℕ) : Prop :=
  let a := A / 1000
  let b := (A % 1000) / 100
  let c := (A % 100) / 10
  let d := A % 10
  a = 5 + c ∧ b = 3 + d

def M (A : ℕ) : ℕ :=
  let a := A / 1000
  let b := (A % 1000) / 100
  let c := (A % 100) / 10
  let d := A % 10
  a + c + 2 * (b + d)

def N (A : ℕ) : ℕ :=
  let b := (A % 1000) / 100
  b - 3

noncomputable def largest_five_three_number := 9946
noncomputable def smallest_five_three_number := 5300

theorem difference_five_three_numbers :
  largest_five_three_number - smallest_five_three_number = 4646 := by
  sorry

noncomputable def specific_five_three_number := 5401

theorem specific_number_condition {A : ℕ} (hA : is_five_three_number A) :
  (M A) % (N A) = 0 ∧ (M A) / (N A) % 5 = 0 → A = specific_five_three_number := by
  sorry

end difference_five_three_numbers_specific_number_condition_l55_55716


namespace linda_original_savings_l55_55836

theorem linda_original_savings (S : ℝ) (h1 : 3 / 4 * S = 300 + 300) :
  S = 1200 :=
by
  sorry -- The proof is not required.

end linda_original_savings_l55_55836


namespace count_multiples_3_or_4_but_not_6_l55_55600

def multiples_between (m n k : Nat) : Nat :=
  (k / m) + (k / n) - (k / (m * n))

theorem count_multiples_3_or_4_but_not_6 :
  let count_multiples (d : Nat) := (3000 / d)
  let multiples_of_3 := count_multiples 3
  let multiples_of_4 := count_multiples 4
  let multiples_of_6 := count_multiples 6
  multiples_of_3 + multiples_of_4 - multiples_of_6 = 1250 := by
  sorry

end count_multiples_3_or_4_but_not_6_l55_55600


namespace largest_class_students_l55_55835

theorem largest_class_students (x : ℕ)
  (h1 : x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = 115) : x = 27 := 
by 
  sorry

end largest_class_students_l55_55835


namespace smallest_b_base_l55_55683

theorem smallest_b_base :
  ∃ b : ℕ, b^2 ≤ 25 ∧ 25 < b^3 ∧ (∀ c : ℕ, c < b → ¬(c^2 ≤ 25 ∧ 25 < c^3)) :=
sorry

end smallest_b_base_l55_55683


namespace bob_max_candies_l55_55982

theorem bob_max_candies (b : ℕ) (h : b + 2 * b = 30) : b = 10 := 
sorry

end bob_max_candies_l55_55982


namespace geometric_series_S_n_div_a_n_l55_55619

-- Define the conditions and the properties of the geometric sequence
variables (a_3 a_5 a_4 a_6 S_n a_n : ℝ) (n : ℕ)
variable (q : ℝ) -- common ratio of the geometric sequence

-- Conditions given in the problem
axiom h1 : a_3 + a_5 = 5 / 4
axiom h2 : a_4 + a_6 = 5 / 8

-- The value we want to prove
theorem geometric_series_S_n_div_a_n : 
  (a_3 + a_5) * q = 5 / 8 → 
  q = 1 / 2 → 
  S_n = a_n * (2^n - 1) :=
by
  intros h1 h2
  sorry

end geometric_series_S_n_div_a_n_l55_55619


namespace cost_of_toys_l55_55837

theorem cost_of_toys (x y : ℝ) (h1 : x + y = 40) (h2 : 90 / x = 150 / y) :
  x = 15 ∧ y = 25 :=
sorry

end cost_of_toys_l55_55837


namespace smallest_positive_real_number_l55_55657

noncomputable def smallest_x : ℝ := 71 / 8

theorem smallest_positive_real_number (x : ℝ) (h₁ : ∀ y : ℝ, 0 < y ∧ (⌊y^2⌋ - y * ⌊y⌋ = 7) → x ≤ y) (h₂ : 0 < x) (h₃ : ⌊x^2⌋ - x * ⌊x⌋ = 7) : x = smallest_x :=
sorry

end smallest_positive_real_number_l55_55657


namespace problem_statement_l55_55232

theorem problem_statement (m n : ℕ) (hm : m ≠ 0) (hn : n ≠ 0) (hprod : m * n = 5000) 
  (h_m_not_div_10 : ¬ ∃ k, m = 10 * k) (h_n_not_div_10 : ¬ ∃ k, n = 10 * k) :
  m + n = 633 :=
sorry

end problem_statement_l55_55232


namespace fraction_multiplication_l55_55314

theorem fraction_multiplication :
  (2 / 3) * (3 / 8) = (1 / 4) :=
sorry

end fraction_multiplication_l55_55314


namespace abs_w_unique_l55_55115

theorem abs_w_unique (w : ℂ) (h : w^2 - 6 * w + 40 = 0) : ∃! x : ℝ, x = Complex.abs w ∧ x = Real.sqrt 40 := by
  sorry

end abs_w_unique_l55_55115


namespace snake_body_length_l55_55862

theorem snake_body_length (l h : ℝ) (h_head: h = l / 10) (h_length: l = 10) : l - h = 9 := 
by 
  rw [h_length, h_head] 
  norm_num
  sorry

end snake_body_length_l55_55862


namespace ada_original_seat_l55_55485

-- Define the problem conditions
def initial_seats : List ℕ := [1, 2, 3, 4, 5]  -- seat numbers

def bea_move (seat : ℕ) : ℕ := seat + 2  -- Bea moves 2 seats to the right
def ceci_move (seat : ℕ) : ℕ := seat - 1  -- Ceci moves 1 seat to the left
def switch (seats : (ℕ × ℕ)) : (ℕ × ℕ) := (seats.2, seats.1)  -- Dee and Edie switch seats

-- The final seating positions (end seats are 1 or 5 for Ada)
axiom ada_end_seat : ∃ final_seat : ℕ, final_seat ∈ [1, 5]  -- Ada returns to an end seat

-- Prove Ada was originally sitting in seat 2
theorem ada_original_seat (final_seat : ℕ) (h₁ : ∃ (s₁ s₂ : ℕ), s₁ ≠ s₂ ∧ bea_move s₁ ≠ final_seat ∧ ceci_move s₂ ≠ final_seat ∧ switch (s₁, s₂).2 ≠ final_seat) : 2 ∈ initial_seats :=
by
  sorry

end ada_original_seat_l55_55485


namespace options_equal_results_l55_55576

theorem options_equal_results :
  (4^3 ≠ 3^4) ∧
  ((-5)^3 = (-5^3)) ∧
  ((-6)^2 ≠ -6^2) ∧
  ((- (5/2))^2 ≠ (- (2/5))^2) :=
by {
  sorry
}

end options_equal_results_l55_55576


namespace find_k_l55_55055

theorem find_k : ∃ b k : ℝ, (∀ x : ℝ, (x + b)^2 = x^2 - 20 * x + k) ∧ k = 100 := by
  sorry

end find_k_l55_55055


namespace problem_statement_l55_55756

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def pow_log2 (x : ℝ) : ℝ := x ^ log2 x

theorem problem_statement (a b c : ℝ)
  (h0 : 1 ≤ a)
  (h1 : 1 ≤ b)
  (h2 : 1 ≤ c)
  (h3 : a * b * c = 10)
  (h4 : pow_log2 a * pow_log2 b * pow_log2 c ≥ 10) :
  a + b + c = 12 := by
  sorry

end problem_statement_l55_55756


namespace intersection_of_A_and_B_l55_55181

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-2, -1, 0, 1}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} :=
by
  sorry

end intersection_of_A_and_B_l55_55181


namespace train_speed_l55_55128

theorem train_speed (distance time : ℕ) (h1 : distance = 180) (h2 : time = 9) : distance / time = 20 := by
  sorry

end train_speed_l55_55128


namespace intersection_A_B_l55_55832

open Set

-- Given definitions of sets A and B
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 - 2 * x ≥ 0}

-- Theorem statement
theorem intersection_A_B :
  A ∩ B = {-1, 0, 2} :=
sorry

end intersection_A_B_l55_55832


namespace object_speed_l55_55853

namespace problem

noncomputable def speed_in_miles_per_hour (distance_in_feet : ℕ) (time_in_seconds : ℕ) : ℝ :=
  let distance_in_miles := distance_in_feet / 5280
  let time_in_hours := time_in_seconds / 3600
  distance_in_miles / time_in_hours

theorem object_speed 
  (distance_in_feet : ℕ)
  (time_in_seconds : ℕ)
  (h : distance_in_feet = 80 ∧ time_in_seconds = 2) :
  speed_in_miles_per_hour distance_in_feet time_in_seconds = 27.27 :=
by
  sorry

end problem

end object_speed_l55_55853


namespace noemi_initial_amount_l55_55191

-- Define the conditions
def lost_on_roulette : Int := 400
def lost_on_blackjack : Int := 500
def still_has : Int := 800
def total_lost : Int := lost_on_roulette + lost_on_blackjack

-- Define the theorem to be proven
theorem noemi_initial_amount : total_lost + still_has = 1700 := by
  -- The proof will be added here
  sorry

end noemi_initial_amount_l55_55191


namespace students_per_class_l55_55049

-- Define the conditions
variables (c : ℕ) (h_c : c ≥ 1) (s : ℕ)

-- Define the total number of books read by one student per year
def books_per_student_per_year := 5 * 12

-- Define the total number of students
def total_number_of_students := c * s

-- Define the total number of books read by the entire student body
def total_books_read := total_number_of_students * books_per_student_per_year

-- The given condition that the entire student body reads 60 books in one year
axiom total_books_eq_60 : total_books_read = 60

theorem students_per_class (h_c : c ≥ 1) : s = 1 / c :=
by sorry

end students_per_class_l55_55049


namespace abe_age_sum_l55_55458

theorem abe_age_sum (h : abe_age = 29) : abe_age + (abe_age - 7) = 51 :=
by
  sorry

end abe_age_sum_l55_55458


namespace rectangle_length_reduction_30_percent_l55_55675

variables (L W : ℝ) (x : ℝ)

theorem rectangle_length_reduction_30_percent
  (h : 1 = (1 - x / 100) * 1.4285714285714287) :
  x = 30 :=
sorry

end rectangle_length_reduction_30_percent_l55_55675


namespace red_shoes_drawn_l55_55883

-- Define the main conditions
def total_shoes : ℕ := 8
def red_shoes : ℕ := 4
def green_shoes : ℕ := 4
def probability_red : ℝ := 0.21428571428571427

-- Problem statement in Lean
theorem red_shoes_drawn (x : ℕ) (hx : ↑x / total_shoes = probability_red) : x = 2 := by
  sorry

end red_shoes_drawn_l55_55883


namespace uncovered_side_length_l55_55286

theorem uncovered_side_length {L W : ℕ} (h1 : L * W = 680) (h2 : 2 * W + L = 74) : L = 40 :=
sorry

end uncovered_side_length_l55_55286


namespace range_of_a_l55_55639

theorem range_of_a 
{α : Type*} [LinearOrderedField α] (a : α) 
(h : ∃ x, x = 3 ∧ (x - a) * (x + 2 * a - 1) ^ 2 * (x - 3 * a) ≤ 0) :
a = -1 ∨ (1 ≤ a ∧ a ≤ 3) :=
sorry

end range_of_a_l55_55639


namespace mr_johnson_pill_intake_l55_55020

theorem mr_johnson_pill_intake (total_days : ℕ) (remaining_pills : ℕ) (fraction : ℚ) (dose : ℕ)
  (h1 : total_days = 30)
  (h2 : remaining_pills = 12)
  (h3 : fraction = 4 / 5) :
  dose = 2 :=
by
  sorry

end mr_johnson_pill_intake_l55_55020


namespace trigonometric_identity_l55_55404

variable {α : ℝ}

theorem trigonometric_identity (h : Real.tan α = 3) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 3 * Real.cos α) = 5 / 6 :=
by
  sorry

end trigonometric_identity_l55_55404


namespace triangle_to_rectangle_ratio_l55_55536

def triangle_perimeter := 60
def rectangle_perimeter := 60

def is_equilateral_triangle (side_length: ℝ) : Prop :=
  3 * side_length = triangle_perimeter

def is_valid_rectangle (length width: ℝ) : Prop :=
  2 * (length + width) = rectangle_perimeter ∧ length = 2 * width

theorem triangle_to_rectangle_ratio (s l w: ℝ) 
  (ht: is_equilateral_triangle s) 
  (hr: is_valid_rectangle l w) : 
  s / w = 2 := by
  sorry

end triangle_to_rectangle_ratio_l55_55536


namespace incorrect_expression_l55_55678

variable (x y : ℝ)

theorem incorrect_expression (h : x > y) (hnx : x < 0) (hny : y < 0) : x^2 - 3 ≤ y^2 - 3 := by
sorry

end incorrect_expression_l55_55678


namespace cost_of_song_book_l55_55846

theorem cost_of_song_book 
  (flute_cost : ℝ) 
  (stand_cost : ℝ) 
  (total_cost : ℝ) 
  (h1 : flute_cost = 142.46) 
  (h2 : stand_cost = 8.89) 
  (h3 : total_cost = 158.35) : 
  total_cost - (flute_cost + stand_cost) = 7.00 := 
by 
  sorry

end cost_of_song_book_l55_55846


namespace polynomial_divisible_by_square_l55_55980

def f (x : ℝ) (a1 a2 a3 a4 : ℝ) : ℝ := x^4 + a1 * x^3 + a2 * x^2 + a3 * x + a4
def f' (x : ℝ) (a1 a2 a3 : ℝ) : ℝ := 4 * x^3 + 3 * a1 * x^2 + 2 * a2 * x + a3

theorem polynomial_divisible_by_square (x0 a1 a2 a3 a4 : ℝ) 
  (h1 : f x0 a1 a2 a3 a4 = 0) 
  (h2 : f' x0 a1 a2 a3 = 0) : 
  ∃ g : ℝ → ℝ, ∀ x : ℝ, f x a1 a2 a3 a4 = (x - x0)^2 * (g x) :=
sorry

end polynomial_divisible_by_square_l55_55980


namespace additional_discount_percentage_l55_55012

def initial_price : ℝ := 2000
def gift_cards : ℝ := 200
def initial_discount_rate : ℝ := 0.15
def final_price : ℝ := 1330

theorem additional_discount_percentage :
  let discounted_price := initial_price * (1 - initial_discount_rate)
  let price_after_gift := discounted_price - gift_cards
  let additional_discount := price_after_gift - final_price
  let additional_discount_percentage := (additional_discount / price_after_gift) * 100
  additional_discount_percentage = 11.33 :=
by
  let discounted_price := initial_price * (1 - initial_discount_rate)
  let price_after_gift := discounted_price - gift_cards
  let additional_discount := price_after_gift - final_price
  let additional_discount_percentage := (additional_discount / price_after_gift) * 100
  show additional_discount_percentage = 11.33
  sorry

end additional_discount_percentage_l55_55012


namespace max_a_is_fractional_value_l55_55430

theorem max_a_is_fractional_value (a k : ℝ) (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = x^2 - (k^2 - 5 * a * k + 3) * x + 7)
  (h_k : 0 ≤ k ∧ k ≤ 2)
  (x1 x2 : ℝ)
  (h_x1 : k ≤ x1 ∧ x1 ≤ k + a)
  (h_x2 : k + 2 * a ≤ x2 ∧ x2 ≤ k + 4 * a)
  (h_fx1_fx2 : f x1 ≥ f x2) :
  a = (2 * Real.sqrt 6 - 4) / 5 :=
sorry

end max_a_is_fractional_value_l55_55430


namespace find_A_from_equation_l55_55598

variable (A B C D : ℕ)
variable (diff_numbers : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
variable (eq1 : A * 1000 + B * 100 + 82 - 900 + C * 10 + 9 = 4000 + 900 + 30 + D)

theorem find_A_from_equation (A B C D : ℕ) (diff_numbers : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (eq1 : A * 1000 + B * 100 + 82 - (900 + C * 10 + 9) = 4000 + 900 + 30 + D) : A = 5 :=
by sorry

end find_A_from_equation_l55_55598


namespace monkey_farm_l55_55689

theorem monkey_farm (x y : ℕ) 
  (h1 : y = 14 * x + 48) 
  (h2 : y = 18 * x - 64) : 
  x = 28 ∧ y = 440 := 
by 
  sorry

end monkey_farm_l55_55689


namespace sum_of_digits_is_21_l55_55306

theorem sum_of_digits_is_21 :
  ∃ (a b c d : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) ∧ 
  ((10 * a + b) * (10 * c + b) = 111 * d) ∧ 
  (d = 9) ∧ 
  (a + b + c + d = 21) := by
  sorry

end sum_of_digits_is_21_l55_55306


namespace students_without_A_l55_55364

theorem students_without_A 
  (total_students : ℕ) 
  (A_in_literature : ℕ) 
  (A_in_science : ℕ) 
  (A_in_both : ℕ) 
  (h_total_students : total_students = 35)
  (h_A_in_literature : A_in_literature = 10)
  (h_A_in_science : A_in_science = 15)
  (h_A_in_both : A_in_both = 5) :
  total_students - (A_in_literature + A_in_science - A_in_both) = 15 :=
by {
  sorry
}

end students_without_A_l55_55364


namespace application_schemes_eq_l55_55113

noncomputable def number_of_application_schemes (graduates : ℕ) (universities : ℕ) : ℕ :=
  universities ^ graduates

theorem application_schemes_eq : 
  number_of_application_schemes 5 3 = 3 ^ 5 := 
by 
  -- proof goes here
  sorry

end application_schemes_eq_l55_55113


namespace sym_diff_A_B_l55_55018

open Set

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

-- Definition of the symmetric difference
def sym_diff (A B : Set ℕ) : Set ℕ := {x | (x ∈ A ∨ x ∈ B) ∧ x ∉ (A ∩ B)}

theorem sym_diff_A_B : sym_diff A B = {0, 3} := 
by 
  sorry

end sym_diff_A_B_l55_55018


namespace least_area_of_figure_l55_55831

theorem least_area_of_figure (c : ℝ) (hc : c > 1) : 
  ∃ A : ℝ, A = (4 / 3) * (c - 1)^(3 / 2) :=
by
  sorry

end least_area_of_figure_l55_55831


namespace modular_inverse_calculation_l55_55545

theorem modular_inverse_calculation : 
  (3 * (49 : ℤ) + 12 * (40 : ℤ)) % 65 = 42 := 
by
  sorry

end modular_inverse_calculation_l55_55545


namespace probability_red_or_blue_l55_55432

theorem probability_red_or_blue :
  ∀ (total_marbles white_marbles green_marbles red_blue_marbles : ℕ),
    total_marbles = 90 →
    (white_marbles : ℝ) / total_marbles = 1 / 6 →
    (green_marbles : ℝ) / total_marbles = 1 / 5 →
    white_marbles = 15 →
    green_marbles = 18 →
    red_blue_marbles = total_marbles - (white_marbles + green_marbles) →
    (red_blue_marbles : ℝ) / total_marbles = 19 / 30 :=
by
  intros total_marbles white_marbles green_marbles red_blue_marbles
  intros h_total_marbles h_white_prob h_green_prob h_white_count h_green_count h_red_blue_count
  sorry

end probability_red_or_blue_l55_55432


namespace fenced_area_with_cutout_l55_55804

theorem fenced_area_with_cutout :
  let rectangle_length : ℕ := 20
  let rectangle_width : ℕ := 16
  let cutout_length : ℕ := 4
  let cutout_width : ℕ := 4
  rectangle_length * rectangle_width - cutout_length * cutout_width = 304 := by
  sorry

end fenced_area_with_cutout_l55_55804


namespace squares_and_sqrt_l55_55797

variable (a b c : ℤ)

theorem squares_and_sqrt (ha : a = 10001) (hb : b = 100010001) (hc : c = 1000200030004000300020001) :
∃ x y z : ℤ, x = a^2 ∧ y = b^2 ∧ z = Int.sqrt c ∧ x = 100020001 ∧ y = 10002000300020001 ∧ z = 1000100010001 :=
by
  use a^2, b^2, Int.sqrt c
  rw [ha, hb, hc]
  sorry

end squares_and_sqrt_l55_55797


namespace who_plays_piano_l55_55320

theorem who_plays_piano 
  (A : Prop)
  (B : Prop)
  (C : Prop)
  (hA : A = True)
  (hB : B = False)
  (hC : A = False)
  (only_one_true : (A ∧ ¬B ∧ ¬C) ∨ (¬A ∧ B ∧ ¬C) ∨ (¬A ∧ ¬B ∧ C)) : B = True := 
sorry

end who_plays_piano_l55_55320


namespace max_traffic_flow_at_v_40_traffic_flow_at_least_10_thousand_l55_55367

-- Define the traffic flow function
noncomputable def traffic_flow (v : ℝ) : ℝ := 920 * v / (v^2 + 3 * v + 1600)

-- Condition: v > 0
axiom v_pos (v : ℝ) : v > 0 → traffic_flow v ≥ 0

-- Prove that the average speed v = 40 results in the maximum traffic flow y = 920/83 ≈ 11.08
theorem max_traffic_flow_at_v_40 : traffic_flow 40 = 920 / 83 :=
sorry

-- Prove that to ensure the traffic flow is at least 10 thousand vehicles per hour,
-- the average speed v should be in the range [25, 64]
theorem traffic_flow_at_least_10_thousand (v : ℝ) (h : traffic_flow v ≥ 10) : 25 ≤ v ∧ v ≤ 64 :=
sorry

end max_traffic_flow_at_v_40_traffic_flow_at_least_10_thousand_l55_55367


namespace xy_yz_zx_equal_zero_l55_55147

noncomputable def side1 (x y z : ℝ) : ℝ := 1 / abs (x^2 + 2 * y * z)
noncomputable def side2 (x y z : ℝ) : ℝ := 1 / abs (y^2 + 2 * z * x)
noncomputable def side3 (x y z : ℝ) : ℝ := 1 / abs (z^2 + 2 * x * y)

def non_degenerate_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem xy_yz_zx_equal_zero
  (x y z : ℝ)
  (h1 : non_degenerate_triangle (side1 x y z) (side2 x y z) (side3 x y z)) :
  xy + yz + zx = 0 := sorry

end xy_yz_zx_equal_zero_l55_55147


namespace initial_number_of_trees_l55_55354

theorem initial_number_of_trees (trees_removed remaining_trees initial_trees : ℕ) 
  (h1 : trees_removed = 4) 
  (h2 : remaining_trees = 2) 
  (h3 : remaining_trees + trees_removed = initial_trees) : 
  initial_trees = 6 :=
by
  sorry

end initial_number_of_trees_l55_55354


namespace whipped_cream_needed_l55_55304

/- Problem conditions -/
def pies_per_day : ℕ := 3
def days : ℕ := 11
def pies_total : ℕ := pies_per_day * days
def pies_eaten_by_tiffany : ℕ := 4
def pies_remaining : ℕ := pies_total - pies_eaten_by_tiffany
def whipped_cream_per_pie : ℕ := 2

/- Proof statement -/
theorem whipped_cream_needed : whipped_cream_per_pie * pies_remaining = 58 := by
  sorry

end whipped_cream_needed_l55_55304


namespace train_speed_l55_55977

theorem train_speed (distance time : ℝ) (h1 : distance = 450) (h2 : time = 8) : distance / time = 56.25 := by
  sorry

end train_speed_l55_55977


namespace more_perfect_squares_with_7_digit_17th_l55_55387

noncomputable def seventeenth_digit (n : ℕ) : ℕ :=
  (n / 10^16) % 10

theorem more_perfect_squares_with_7_digit_17th
  (h_bound : ∀ n, n < 10^10 → (n * n) < 10^20)
  (h_representation : ∀ m, m < 10^20 → ∃ n, n < 10^10 ∧ m = n * n) :
  (∃ majority_digit_7 : ℕ,
    (∃ majority_digit_8 : ℕ,
      ∀ n, seventeenth_digit (n * n) = 7 → majority_digit_7 > majority_digit_8)
  ) :=
sorry

end more_perfect_squares_with_7_digit_17th_l55_55387


namespace initial_juggling_objects_l55_55567

theorem initial_juggling_objects (x : ℕ) : (∀ i : ℕ, i = 5 → x + 2*i = 13) → x = 3 :=
by 
  intro h
  sorry

end initial_juggling_objects_l55_55567


namespace circle_with_all_three_colors_l55_55820

-- Define color type using an inductive type with three colors
inductive Color
| red
| green
| blue

-- Define a function that assigns a color to each point in the plane
def color_function (point : ℝ × ℝ) : Color := sorry

-- Define the main theorem stating that for any coloring, there exists a circle that contains points of all three colors
theorem circle_with_all_three_colors (color_func : ℝ × ℝ → Color) (exists_red : ∃ p : ℝ × ℝ, color_func p = Color.red)
                                      (exists_green : ∃ p : ℝ × ℝ, color_func p = Color.green) 
                                      (exists_blue : ∃ p : ℝ × ℝ, color_func p = Color.blue) :
    ∃ (c : ℝ × ℝ) (r : ℝ), ∃ p1 p2 p3 : ℝ × ℝ, 
             color_func p1 = Color.red ∧ color_func p2 = Color.green ∧ color_func p3 = Color.blue ∧ 
             (dist p1 c = r) ∧ (dist p2 c = r) ∧ (dist p3 c = r) :=
by 
  sorry

end circle_with_all_three_colors_l55_55820


namespace correct_average_l55_55973

theorem correct_average (n : ℕ) (wrong_avg : ℕ) (wrong_num correct_num : ℕ) (correct_avg : ℕ)
  (h1 : n = 10) 
  (h2 : wrong_avg = 21)
  (h3 : wrong_num = 26)
  (h4 : correct_num = 36)
  (h5 : correct_avg = 22) :
  (wrong_avg * n + (correct_num - wrong_num)) / n = correct_avg :=
by
  sorry

end correct_average_l55_55973


namespace find_x_y_l55_55898

theorem find_x_y (x y : ℤ) (hx : 0 < x) (hy : 0 < y) (h : (x + y * Complex.I)^2 = (7 + 24 * Complex.I)) :
  x + y * Complex.I = 4 + 3 * Complex.I :=
by
  sorry

end find_x_y_l55_55898


namespace range_of_m_l55_55011

theorem range_of_m (m : ℝ) (p : Prop) (q : Prop)
  (hp : (2 * m)^2 - 4 ≥ 0 ↔ p)
  (hq : 1 < (Real.sqrt (5 + m)) / (Real.sqrt 5) ∧ (Real.sqrt (5 + m)) / (Real.sqrt 5) < 2 ↔ q)
  (hnq : ¬q = False)
  (hpq : (p ∧ q) = False) :
  0 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l55_55011


namespace max_xy_l55_55412

theorem max_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 5 * x + 6 * y < 90) :
  xy * (90 - 5 * x - 6 * y) ≤ 900 := by
  sorry

end max_xy_l55_55412


namespace lindas_initial_candies_l55_55828

theorem lindas_initial_candies (candies_given : ℝ) (candies_left : ℝ) (initial_candies : ℝ) : 
  candies_given = 28 ∧ candies_left = 6 → initial_candies = candies_given + candies_left → initial_candies = 34 := 
by 
  sorry

end lindas_initial_candies_l55_55828


namespace probability_not_all_same_color_l55_55005

def num_colors := 3
def draws := 3
def total_outcomes := num_colors ^ draws

noncomputable def prob_same_color : ℚ := (3 / total_outcomes)
noncomputable def prob_not_same_color : ℚ := 1 - prob_same_color

theorem probability_not_all_same_color :
  prob_not_same_color = 8 / 9 :=
by
  sorry

end probability_not_all_same_color_l55_55005


namespace bob_needs_8_additional_wins_to_afford_puppy_l55_55737

variable (n : ℕ) (grand_prize_per_win : ℝ) (total_cost : ℝ)

def bob_total_wins_to_afford_puppy : Prop :=
  total_cost = 1000 ∧ grand_prize_per_win = 100 ∧ n = (total_cost / grand_prize_per_win) - 2

theorem bob_needs_8_additional_wins_to_afford_puppy :
  bob_total_wins_to_afford_puppy 8 100 1000 :=
by {
  sorry
}

end bob_needs_8_additional_wins_to_afford_puppy_l55_55737


namespace last_years_rate_per_mile_l55_55148

-- Definitions from the conditions
variables (m : ℕ) (x : ℕ)

-- Condition 1: This year, walkers earn $2.75 per mile
def amount_per_mile_this_year : ℝ := 2.75

-- Condition 2: Last year's winner collected $44
def last_years_total_amount : ℕ := 44

-- Condition 3: Elroy will walk 5 more miles than last year's winner
def elroy_walks_more_miles (m : ℕ) : ℕ := m + 5

-- The main goal is to prove that last year's rate per mile was $4 given the conditions
theorem last_years_rate_per_mile (h1 : last_years_total_amount = m * x)
  (h2 : last_years_total_amount = (elroy_walks_more_miles m) * amount_per_mile_this_year) :
  x = 4 :=
by {
  sorry
}

end last_years_rate_per_mile_l55_55148


namespace arithmetic_sequence_statements_l55_55453

/-- 
Given the arithmetic sequence {a_n} with first term a_1 > 0 and the sum of the first n terms denoted as S_n, 
prove the following statements based on the condition S_8 = S_16:
  1. d > 0
  2. a_{13} < 0
  3. The maximum value of S_n is S_{12}
  4. When S_n < 0, the minimum value of n is 25
--/
theorem arithmetic_sequence_statements (a_1 d : ℤ) (S : ℕ → ℤ)
  (h1 : a_1 > 0)
  (h2 : S 8 = S 16)
  (hS8 : S 8 = 8 * a_1 + 28 * d)
  (hS16 : S 16 = 16 * a_1 + 120 * d) :
  (d > 0) ∨ 
  (a_1 + 12 * d < 0) ∨ 
  (∀ n, n ≠ 12 → S n ≤ S 12) ∨ 
  (∀ n, S n < 0 → n ≥ 25) :=
sorry

end arithmetic_sequence_statements_l55_55453


namespace johns_cycling_speed_needed_l55_55064

theorem johns_cycling_speed_needed 
  (swim_speed : Float := 3)
  (swim_distance : Float := 0.5)
  (run_speed : Float := 8)
  (run_distance : Float := 4)
  (total_time : Float := 3)
  (bike_distance : Float := 20) :
  (bike_distance / (total_time - (swim_distance / swim_speed + run_distance / run_speed))) = 60 / 7 := 
  by
  sorry

end johns_cycling_speed_needed_l55_55064


namespace parabola_line_intersection_l55_55738

theorem parabola_line_intersection (p : ℝ) (hp : p > 0) 
  (line_eq : ∃ b : ℝ, ∀ x : ℝ, 2 * x + b = 2 * x - p/2) 
  (focus := (p / 4, 0))
  (point_A := (0, -p / 2))
  (area_OAF : 1 / 2 * (p / 4) * (p / 2) = 1) : 
  p = 4 :=
sorry

end parabola_line_intersection_l55_55738


namespace negation_of_p_l55_55936

variable (p : Prop) (n : ℕ)

def proposition_p := ∃ n : ℕ, n^2 > 2^n

theorem negation_of_p : ¬ proposition_p ↔ ∀ n : ℕ, n^2 <= 2^n :=
by
  sorry

end negation_of_p_l55_55936


namespace circle_area_x2_y2_eq_102_l55_55978

theorem circle_area_x2_y2_eq_102 :
  ∀ (x y : ℝ), (x + 9)^2 + (y - 3)^2 = 102 → π * 102 = 102 * π :=
by
  intros
  sorry

end circle_area_x2_y2_eq_102_l55_55978


namespace range_of_sqrt_meaningful_real_l55_55277

theorem range_of_sqrt_meaningful_real (x : ℝ) : (x - 1 ≥ 0) ↔ (x ≥ 1) :=
by
  sorry

end range_of_sqrt_meaningful_real_l55_55277


namespace find_growth_rate_l55_55706

noncomputable def donation_first_day : ℝ := 10000
noncomputable def donation_third_day : ℝ := 12100
noncomputable def growth_rate (x : ℝ) : Prop :=
  (donation_first_day * (1 + x) ^ 2 = donation_third_day)

theorem find_growth_rate : ∃ x : ℝ, growth_rate x ∧ x = 0.1 :=
by
  sorry

end find_growth_rate_l55_55706


namespace product_of_numbers_l55_55336

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 404) : x * y = 40 := sorry

end product_of_numbers_l55_55336


namespace evaluate_expression_l55_55469

theorem evaluate_expression (a b : ℕ) (h1 : a = 3) (h2 : b = 2) : (a^3 + b)^2 - (a^3 - b)^2 = 216 :=
by
  -- Proof is not required, add sorry to skip the proof
  sorry

end evaluate_expression_l55_55469


namespace inhabitable_land_fraction_l55_55859

theorem inhabitable_land_fraction (total_surface not_water_covered initially_inhabitable tech_advancement_viable : ℝ)
  (h1 : not_water_covered = 1 / 3 * total_surface)
  (h2 : initially_inhabitable = 1 / 3 * not_water_covered)
  (h3 : tech_advancement_viable = 1 / 2 * (not_water_covered - initially_inhabitable)) :
  (initially_inhabitable + tech_advancement_viable) / total_surface = 2 / 9 := 
sorry

end inhabitable_land_fraction_l55_55859


namespace car_speed_l55_55696

theorem car_speed (v : ℝ) (h : (1 / v) * 3600 = (1 / 450) * 3600 + 2) : v = 360 :=
by
  sorry

end car_speed_l55_55696


namespace last_four_digits_5_to_2019_l55_55541

theorem last_four_digits_5_to_2019 :
  ∃ (x : ℕ), (5^2019) % 10000 = x ∧ x = 8125 :=
by
  sorry

end last_four_digits_5_to_2019_l55_55541


namespace harris_flour_amount_l55_55595

noncomputable def flour_needed_by_cakes (cakes : ℕ) : ℕ := cakes * 100

noncomputable def traci_flour : ℕ := 500

noncomputable def total_cakes : ℕ := 9

theorem harris_flour_amount : flour_needed_by_cakes total_cakes - traci_flour = 400 := 
by
  sorry

end harris_flour_amount_l55_55595


namespace medium_bed_rows_l55_55939

theorem medium_bed_rows (large_top_beds : ℕ) (large_bed_rows : ℕ) (large_bed_seeds_per_row : ℕ) 
                         (medium_beds : ℕ) (medium_bed_seeds_per_row : ℕ) (total_seeds : ℕ) :
    large_top_beds = 2 ∧ large_bed_rows = 4 ∧ large_bed_seeds_per_row = 25 ∧
    medium_beds = 2 ∧ medium_bed_seeds_per_row = 20 ∧ total_seeds = 320 →
    ((total_seeds - (large_top_beds * large_bed_rows * large_bed_seeds_per_row)) / medium_bed_seeds_per_row) = 6 :=
by
  intro conditions
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := conditions
  sorry

end medium_bed_rows_l55_55939


namespace pipe_fills_entire_cistern_in_77_minutes_l55_55427

-- Define the time taken to fill 1/11 of the cistern
def time_to_fill_one_eleven_cistern : ℕ := 7

-- Define the fraction of the cistern filled in a certain time
def fraction_filled (t : ℕ) : ℚ := t / time_to_fill_one_eleven_cistern * (1 / 11)

-- Define the problem statement
theorem pipe_fills_entire_cistern_in_77_minutes : 
  fraction_filled 77 = 1 := by
  sorry

end pipe_fills_entire_cistern_in_77_minutes_l55_55427


namespace length_of_plot_l55_55226

theorem length_of_plot 
  (b : ℝ)
  (H1 : 2 * (b + 20) + 2 * b = 5300 / 26.50)
  : (b + 20 = 60) :=
sorry

end length_of_plot_l55_55226


namespace max_profit_l55_55949

noncomputable def C (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x < 80 then (1 / 3) * x^2 + 10 * x
  else 51 * x + 10000 / x - 1450

noncomputable def L (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x < 80 then -(1 / 3) * x^2 + 40 * x - 250
  else -(x + 10000 / x) + 1200

theorem max_profit :
  ∃ x : ℝ, (L x) = 1000 ∧ x = 100 :=
by
  sorry

end max_profit_l55_55949


namespace log_base_243_l55_55865

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_base_243 : log_base 3 243 = 5 := by
  -- this is the statement, proof is omitted
  sorry

end log_base_243_l55_55865


namespace solve_quadratic_eqn_l55_55247

theorem solve_quadratic_eqn:
  (∃ x: ℝ, (x + 10)^2 = (4 * x + 6) * (x + 8)) ↔ 
  (∀ x: ℝ, x = 2.131 ∨ x = -8.131) := 
by
  sorry

end solve_quadratic_eqn_l55_55247


namespace simplify_proof_l55_55285

noncomputable def simplify_expression (a b c d x y : ℝ) (h : c * x ≠ d * y) : ℝ :=
  (c * x * (b^2 * x^2 - 4 * b^2 * y^2 + a^2 * y^2) 
  - d * y * (b^2 * x^2 - 2 * a^2 * x^2 - 3 * a^2 * y^2)) / (c * x - d * y)

theorem simplify_proof (a b c d x y : ℝ) (h : c * x ≠ d * y) :
  simplify_expression a b c d x y h = b^2 * x^2 + a^2 * y^2 :=
by sorry

end simplify_proof_l55_55285


namespace rationalize_fraction_l55_55414

theorem rationalize_fraction :
  (5 : ℚ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18 + Real.sqrt 32) = 
  (5 * Real.sqrt 2) / 36 :=
by
  sorry

end rationalize_fraction_l55_55414


namespace function_properties_l55_55416

variable (f : ℝ → ℝ)
variable (h1 : ∀ x y : ℝ, 2 * f x * f y = f (x + y) + f (x - y))
variable (h2 : f 1 = -1)

theorem function_properties :
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ x : ℝ, f x + f (1 - x) = 0) :=
sorry

end function_properties_l55_55416


namespace fabric_cost_equation_l55_55728

theorem fabric_cost_equation (x : ℝ) :
  (3 * x + 5 * (138 - x) = 540) :=
sorry

end fabric_cost_equation_l55_55728


namespace Barbara_Mike_ratio_is_one_half_l55_55369

-- Define the conditions
def Mike_age_current : ℕ := 16
def Mike_age_future : ℕ := 24
def Barbara_age_future : ℕ := 16

-- Define Barbara's current age based on the conditions
def Barbara_age_current : ℕ := Mike_age_current - (Mike_age_future - Barbara_age_future)

-- Define the ratio of Barbara's age to Mike's age
def ratio_Barbara_Mike : ℚ := Barbara_age_current / Mike_age_current

-- Prove that the ratio is 1:2
theorem Barbara_Mike_ratio_is_one_half : ratio_Barbara_Mike = 1 / 2 := by
  sorry

end Barbara_Mike_ratio_is_one_half_l55_55369


namespace age_problem_l55_55525

-- Define the conditions
variables (a b c : ℕ)

-- Assumptions based on conditions
theorem age_problem (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 37) : b = 14 :=
by {
  sorry   -- Placeholder for the detailed proof
}

end age_problem_l55_55525


namespace difference_in_tiles_l55_55674

theorem difference_in_tiles (n : ℕ) (hn : n = 9) : (n + 1)^2 - n^2 = 19 :=
by sorry

end difference_in_tiles_l55_55674


namespace joe_initial_paint_amount_l55_55386

theorem joe_initial_paint_amount (P : ℝ) 
  (h1 : (2/3) * P + (1/15) * P = 264) : P = 360 :=
sorry

end joe_initial_paint_amount_l55_55386


namespace john_profit_l55_55996

theorem john_profit (cost price : ℕ) (n : ℕ) (h1 : cost = 4) (h2 : price = 8) (h3 : n = 30) : 
  n * (price - cost) = 120 :=
by
  -- The proof goes here
  sorry

end john_profit_l55_55996


namespace max_value_2x_minus_y_l55_55928

theorem max_value_2x_minus_y 
  (x y : ℝ) 
  (h1 : x - y + 1 ≥ 0) 
  (h2 : y + 1 ≥ 0) 
  (h3 : x + y + 1 ≤ 0) : 
  2 * x - y ≤ 1 :=
sorry

end max_value_2x_minus_y_l55_55928


namespace bird_weights_l55_55776

variables (A B V G : ℕ)

theorem bird_weights : 
  A + B + V + G = 32 ∧ 
  V < G ∧ 
  V + G < B ∧ 
  A < V + B ∧ 
  G + B < A + V 
  → 
  (A = 13 ∧ V = 4 ∧ G = 5 ∧ B = 10) :=
sorry

end bird_weights_l55_55776


namespace set_nonempty_iff_nonneg_l55_55593

theorem set_nonempty_iff_nonneg (a : ℝ) :
  (∃ x : ℝ, x^2 ≤ a) ↔ a ≥ 0 :=
sorry

end set_nonempty_iff_nonneg_l55_55593


namespace tank_ratio_l55_55895

theorem tank_ratio (V1 V2 : ℝ) (h1 : 0 < V1) (h2 : 0 < V2) (h1_full : 3 / 4 * V1 - 7 / 20 * V2 = 0) (h2_full : 1 / 4 * V2 + 7 / 20 * V2 = 3 / 5 * V2) :
  V1 / V2 = 7 / 9 :=
by
  sorry

end tank_ratio_l55_55895


namespace gcd_84_210_l55_55909

theorem gcd_84_210 : Nat.gcd 84 210 = 42 :=
by {
  sorry
}

end gcd_84_210_l55_55909


namespace grid_3x3_unique_72_l55_55218

theorem grid_3x3_unique_72 :
  ∃ (f : Fin 3 → Fin 3 → ℕ), 
    (∀ (i j : Fin 3), 1 ≤ f i j ∧ f i j ≤ 9) ∧
    (∀ (i j k : Fin 3), j < k → f i j < f i k) ∧
    (∀ (i j k : Fin 3), i < k → f i j < f k j) ∧
    f 0 0 = 1 ∧ f 1 1 = 5 ∧ f 2 2 = 8 ∧
    (∃! (g : Fin 3 → Fin 3 → ℕ), 
      (∀ (i j : Fin 3), 1 ≤ g i j ∧ g i j ≤ 9) ∧
      (∀ (i j k : Fin 3), j < k → g i j < g i k) ∧
      (∀ (i j k : Fin 3), i < k → g i j < g k j) ∧
      g 0 0 = 1 ∧ g 1 1 = 5 ∧ g 2 2 = 8) :=
sorry

end grid_3x3_unique_72_l55_55218


namespace ellipse_range_l55_55569

theorem ellipse_range (t : ℝ) (x y : ℝ) :
  (10 - t > 0) → (t - 4 > 0) → (10 - t ≠ t - 4) →
  (t ∈ (Set.Ioo 4 7 ∪ Set.Ioo 7 10)) :=
by
  intros h1 h2 h3
  sorry

end ellipse_range_l55_55569


namespace math_books_count_l55_55393

theorem math_books_count (M H : ℕ) (h1 : M + H = 80) (h2 : 4 * M + 5 * H = 373) : M = 27 :=
by
  sorry

end math_books_count_l55_55393


namespace hyperbola_asymptote_slope_l55_55642

theorem hyperbola_asymptote_slope :
  ∀ {x y : ℝ}, (x^2 / 144 - y^2 / 81 = 1) → (∃ m : ℝ, ∀ x, y = m * x ∨ y = -m * x ∧ m = 3 / 4) :=
by
  sorry

end hyperbola_asymptote_slope_l55_55642


namespace angle_A_value_sin_BC_value_l55_55431

open Real

noncomputable def triangleABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ 
  A + B + C = π 

theorem angle_A_value (A B C : ℝ) (h : triangleABC a b c A B C) (h1 : cos 2 * A - 3 * cos (B + C) = 1) : 
  A = π / 3 :=
sorry

theorem sin_BC_value (A B C S b c : ℝ) (h : triangleABC a b c A B C)
  (hA : A = π / 3) (hS : S = 5 * sqrt 3) (hb : b = 5) : 
  sin B * sin C = 5 / 7 :=
sorry

end angle_A_value_sin_BC_value_l55_55431


namespace cos_alpha_l55_55438

-- Define the conditions
variable (α : Real)
variable (x y r : Real)
-- Given the point (-3, 4)
def point_condition (x : Real) (y : Real) : Prop := x = -3 ∧ y = 4

-- Define r as the distance
def radius_condition (x y r : Real) : Prop := r = Real.sqrt (x ^ 2 + y ^ 2)

-- Prove that cos α and cos 2α are the given values
theorem cos_alpha (α : Real) (x y r : Real) (h1 : point_condition x y) (h2 : radius_condition x y r) :
  Real.cos α = -3 / 5 ∧ Real.cos (2 * α) = -7 / 25 :=
by
  sorry

end cos_alpha_l55_55438


namespace original_number_of_men_l55_55307

theorem original_number_of_men (M : ℕ) : 
  (∀ t : ℕ, (t = 8) -> (8:ℕ) * M = 8 * 10 / (M - 3) ) -> ( M = 12 ) :=
by sorry

end original_number_of_men_l55_55307


namespace average_of_remaining_two_nums_l55_55484

theorem average_of_remaining_two_nums (S S4 : ℕ) (h1 : S / 6 = 8) (h2 : S4 / 4 = 5) :
  ((S - S4) / 2 = 14) :=
by 
  sorry

end average_of_remaining_two_nums_l55_55484


namespace correct_train_process_l55_55798

-- Define each step involved in the train process
inductive Step
| buy_ticket
| wait_for_train
| check_ticket
| board_train
| repair_train

open Step

-- Define each condition as a list of steps
def process_a : List Step := [buy_ticket, wait_for_train, check_ticket, board_train]
def process_b : List Step := [wait_for_train, buy_ticket, board_train, check_ticket]
def process_c : List Step := [buy_ticket, wait_for_train, board_train, check_ticket]
def process_d : List Step := [repair_train, buy_ticket, check_ticket, board_train]

-- Define the correct process
def correct_process : List Step := [buy_ticket, wait_for_train, check_ticket, board_train]

-- The theorem to prove that process A is the correct representation
theorem correct_train_process : process_a = correct_process :=
by {
  sorry
}

end correct_train_process_l55_55798


namespace geom_seq_m_value_l55_55725

/-- Given a geometric sequence {a_n} with a1 = 1 and common ratio q ≠ 1,
    if a_m = a_1 * a_2 * a_3 * a_4 * a_5, then m = 11. -/
theorem geom_seq_m_value (q : ℝ) (h_q : q ≠ 1) :
  ∃ (m : ℕ), (m = 11) ∧ (∃ a : ℕ → ℝ, a 1 = 1 ∧ (∀ n, a (n + 1) = a n * q ) ∧ (a m = a 1 * a 2 * a 3 * a 4 * a 5)) :=
by
  sorry

end geom_seq_m_value_l55_55725


namespace solve_for_x_l55_55468

def custom_mul (a b : ℤ) : ℤ := a * b + a + b

theorem solve_for_x (x : ℤ) :
  custom_mul 3 (3 * x - 1) = 27 → x = 7 / 3 := by
sorry

end solve_for_x_l55_55468


namespace total_tickets_sold_l55_55465

def ticket_prices : Nat := 25
def senior_ticket_price : Nat := 15
def total_receipts : Nat := 9745
def senior_tickets_sold : Nat := 348
def adult_tickets_sold : Nat := (total_receipts - senior_ticket_price * senior_tickets_sold) / ticket_prices

theorem total_tickets_sold : adult_tickets_sold + senior_tickets_sold = 529 :=
by
  sorry

end total_tickets_sold_l55_55465


namespace usual_time_to_catch_bus_l55_55251

theorem usual_time_to_catch_bus (S T : ℝ) (h : S / (4 / 5 * S) = (T + 3) / T) : T = 12 :=
by 
  sorry

end usual_time_to_catch_bus_l55_55251


namespace max_a_value_l55_55317

theorem max_a_value (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) (h4 : d < 50) :
  a ≤ 2924 :=
by sorry

end max_a_value_l55_55317


namespace inverse_of_inverse_at_9_l55_55707

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5

noncomputable def f_inv (x : ℝ) : ℝ := (x - 5) / 4

theorem inverse_of_inverse_at_9 : f_inv (f_inv 9) = -1 :=
by
  sorry

end inverse_of_inverse_at_9_l55_55707


namespace least_number_of_teams_l55_55879

/-- A coach has 30 players in a team. If he wants to form teams of at most 7 players each for a tournament, we aim to prove that the least number of teams that he needs is 5. -/
theorem least_number_of_teams (players teams : ℕ) 
  (h_players : players = 30) 
  (h_teams : ∀ t, t ≤ 7 → t ∣ players) : teams = 5 := by
  sorry

end least_number_of_teams_l55_55879


namespace round_310242_to_nearest_thousand_l55_55602

-- Define the conditions and the target statement
def round_to_nearest_thousand (n : ℕ) : ℕ :=
  if (n % 1000) < 500 then (n / 1000) * 1000 else (n / 1000 + 1) * 1000

theorem round_310242_to_nearest_thousand :
  round_to_nearest_thousand 310242 = 310000 :=
by
  sorry

end round_310242_to_nearest_thousand_l55_55602


namespace rectangle_measurement_error_l55_55170

theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) 
  (h1 : 0 < L) (h2 : 0 < W) 
  (h3 : A = L * W)
  (h4 : A' = L * (1 + x / 100) * W * (1 - 4 / 100))
  (h5 : A' = A * (100.8 / 100)) :
  x = 5 :=
by
  sorry

end rectangle_measurement_error_l55_55170


namespace find_a9_l55_55946

noncomputable def a : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+1) => a n + n

theorem find_a9 : a 9 = 37 := by
  sorry

end find_a9_l55_55946


namespace florist_sold_16_roses_l55_55141

-- Definitions for initial and final states
def initial_roses : ℕ := 37
def picked_roses : ℕ := 19
def final_roses : ℕ := 40

-- Defining the variable for number of roses sold
variable (x : ℕ)

-- The statement to prove
theorem florist_sold_16_roses
  (h : initial_roses - x + picked_roses = final_roses) : x = 16 := 
by
  -- Placeholder for proof
  sorry

end florist_sold_16_roses_l55_55141


namespace find_multiple_of_diff_l55_55592

theorem find_multiple_of_diff (n sum diff remainder k : ℕ) 
  (hn : n = 220070) 
  (hs : sum = 555 + 445) 
  (hd : diff = 555 - 445)
  (hr : remainder = 70)
  (hmod : n % sum = remainder) 
  (hquot : n / sum = k) :
  ∃ k, k = 2 ∧ k * diff = n / sum := 
by 
  sorry

end find_multiple_of_diff_l55_55592


namespace acute_triangle_incorrect_option_l55_55794

theorem acute_triangle_incorrect_option (A B C : ℝ) (hA : 0 < A ∧ A < 90) (hB : 0 < B ∧ B < 90) (hC : 0 < C ∧ C < 90)
  (angle_sum : A + B + C = 180) (h_order : A > B ∧ B > C) : ¬(B + C < 90) :=
sorry

end acute_triangle_incorrect_option_l55_55794


namespace inequality_solution_l55_55815

theorem inequality_solution (x : ℝ) : 3 * x^2 + 9 * x + 6 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 := 
sorry

end inequality_solution_l55_55815


namespace find_k_l55_55955

noncomputable def digit_sum (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem find_k :
  ∃ k : ℕ, digit_sum (5 * (5 * (10 ^ (k - 1) - 1) / 9)) = 600 ∧ k = 87 :=
by
  sorry

end find_k_l55_55955


namespace max_value_of_z_l55_55302

theorem max_value_of_z (x y : ℝ) (h1 : x + 2 * y - 5 ≥ 0) (h2 : x - 2 * y + 3 ≥ 0) (h3 : x - 5 ≤ 0) :
  ∃ x y, x + y = 9 :=
by {
  sorry
}

end max_value_of_z_l55_55302


namespace hexagon_piece_area_l55_55205

theorem hexagon_piece_area (A : ℝ) (n : ℕ) (h1 : A = 21.12) (h2 : n = 6) : 
  A / n = 3.52 :=
by
  -- The proof will go here
  sorry

end hexagon_piece_area_l55_55205


namespace sin_13pi_over_4_eq_neg_sqrt2_over_2_l55_55197

theorem sin_13pi_over_4_eq_neg_sqrt2_over_2 : Real.sin (13 * Real.pi / 4) = -Real.sqrt 2 / 2 := 
by 
  sorry

end sin_13pi_over_4_eq_neg_sqrt2_over_2_l55_55197


namespace cone_height_l55_55347

theorem cone_height (V : ℝ) (h r : ℝ) (π : ℝ) (h_eq_r : h = r) (volume_eq : V = 12288 * π) (V_def : V = (1/3) * π * r^3) : h = 36 := 
by
  sorry

end cone_height_l55_55347


namespace bobby_initial_candy_l55_55561

theorem bobby_initial_candy (candy_ate_start candy_ate_more candy_left : ℕ)
  (h1 : candy_ate_start = 9) (h2 : candy_ate_more = 5) (h3 : candy_left = 8) :
  candy_ate_start + candy_ate_more + candy_left = 22 :=
by
  rw [h1, h2, h3]
  -- sorry


end bobby_initial_candy_l55_55561


namespace prove_logical_proposition_l55_55296

theorem prove_logical_proposition (p q : Prop) (hp : p) (hq : ¬q) : (¬p ∨ ¬q) :=
by
  sorry

end prove_logical_proposition_l55_55296


namespace rounded_product_less_than_original_l55_55669

theorem rounded_product_less_than_original
  (x y z : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (hxy : x > 2 * y) :
  (x + z) * (y - z) < x * y :=
by
  sorry

end rounded_product_less_than_original_l55_55669


namespace inf_many_solutions_to_ineq_l55_55439

theorem inf_many_solutions_to_ineq (x : ℕ) : (15 < 2 * x + 20) ↔ x ≥ 1 :=
by
  sorry

end inf_many_solutions_to_ineq_l55_55439


namespace press_t_denomination_l55_55522

def press_f_rate_per_minute := 1000
def press_t_rate_per_minute := 200
def time_in_seconds := 3
def f_denomination := 5
def additional_amount := 50

theorem press_t_denomination : 
  ∃ (x : ℝ), 
  (3 * (5 * (1000 / 60))) = (3 * (x * (200 / 60)) + 50) → 
  x = 20 := 
by 
  -- Proof logic here
  sorry

end press_t_denomination_l55_55522


namespace cafeteria_apples_pies_l55_55300

theorem cafeteria_apples_pies (initial_apples handed_out_apples apples_per_pie remaining_apples pies : ℕ) 
    (h_initial: initial_apples = 62) 
    (h_handed_out: handed_out_apples = 8) 
    (h_apples_per_pie: apples_per_pie = 9)
    (h_remaining: remaining_apples = initial_apples - handed_out_apples) 
    (h_pies: pies = remaining_apples / apples_per_pie) : 
    pies = 6 := by
  sorry

end cafeteria_apples_pies_l55_55300


namespace arithmetic_problem_l55_55789

theorem arithmetic_problem : 
  let part1 := (20 / 100) * 120
  let part2 := (25 / 100) * 250
  let part3 := (15 / 100) * 80
  let sum := part1 + part2 + part3
  let subtract := (10 / 100) * 600
  sum - subtract = 38.5 := by
  sorry

end arithmetic_problem_l55_55789


namespace range_of_m_l55_55267

noncomputable def f (m x : ℝ) : ℝ := 2 * m * x^2 - 2 * (4 - m) * x + 1
noncomputable def g (m x : ℝ) : ℝ := m * x

theorem range_of_m :
  (∀ x : ℝ, f m x > 0 ∨ g m x > 0) → 0 < m ∧ m < 8 :=
sorry

end range_of_m_l55_55267


namespace common_difference_arithmetic_progression_l55_55535

theorem common_difference_arithmetic_progression {n : ℕ} (x y : ℝ) (a : ℕ → ℝ) 
  (h : ∀ k : ℕ, k ≤ n → a (k+1) = a k + (y - x) / (n + 1)) 
  : (∃ d : ℝ, ∀ i : ℕ, i ≤ n + 1 → a (i+1) = x + i * d) ∧ d = (y - x) / (n + 1) := 
by
  sorry

end common_difference_arithmetic_progression_l55_55535


namespace estimate_shaded_area_l55_55338

theorem estimate_shaded_area 
  (side_length : ℝ)
  (points_total : ℕ)
  (points_shaded : ℕ)
  (area_shaded_estimation : ℝ) :
  side_length = 6 →
  points_total = 800 →
  points_shaded = 200 →
  area_shaded_estimation = (36 * (200 / 800)) →
  area_shaded_estimation = 9 :=
by
  intros h_side_length h_points_total h_points_shaded h_area_shaded_estimation
  rw [h_side_length, h_points_total, h_points_shaded] at *
  norm_num at h_area_shaded_estimation
  exact h_area_shaded_estimation

end estimate_shaded_area_l55_55338


namespace absolute_value_inequality_solution_l55_55677

theorem absolute_value_inequality_solution (x : ℝ) :
  abs ((3 * x + 2) / (x + 2)) > 3 ↔ (x < -2) ∨ (-2 < x ∧ x < -4 / 3) :=
by
  sorry

end absolute_value_inequality_solution_l55_55677


namespace average_of_ABC_l55_55964

theorem average_of_ABC (A B C : ℤ)
  (h1 : 101 * C - 202 * A = 404)
  (h2 : 101 * B + 303 * A = 505)
  (h3 : 101 * A + 101 * B + 101 * C = 303) :
  (A + B + C) / 3 = 3 :=
by
  sorry

end average_of_ABC_l55_55964


namespace range_of_a_l55_55713

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, (x^2 + a*x + 4 < 0)) ↔ (a < -4 ∨ a > 4) :=
by
  sorry

end range_of_a_l55_55713


namespace find_radius_l55_55770

def radius_of_circle (d : ℤ) (PQ : ℕ) (QR : ℕ) (r : ℕ) : Prop := 
  let PR := PQ + QR
  (PQ * PR = (d - r) * (d + r)) ∧ (d = 15) ∧ (PQ = 11) ∧ (QR = 8) ∧ (r = 4)

-- Now stating the theorem to prove the radius r given the conditions
theorem find_radius (r : ℕ) : radius_of_circle 15 11 8 r := by
  sorry

end find_radius_l55_55770


namespace sin_cos_sum_eq_l55_55818

theorem sin_cos_sum_eq (θ : ℝ) 
  (h1 : θ ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.tan (θ + π / 4) = 1 / 2): 
  Real.sin θ + Real.cos θ = -Real.sqrt 10 / 5 := 
  sorry

end sin_cos_sum_eq_l55_55818


namespace solution_set_of_inequality_l55_55269

theorem solution_set_of_inequality (x : ℝ) : x^2 - |x| - 2 < 0 ↔ -2 < x ∧ x < 2 := 
by
  sorry

end solution_set_of_inequality_l55_55269


namespace find_x_l55_55375

def perpendicular_vectors_solution (x : ℝ) : Prop :=
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (3, -2)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 2 / 3

theorem find_x (x : ℝ) : perpendicular_vectors_solution x := sorry

end find_x_l55_55375


namespace operation_two_three_l55_55288

def operation (a b : ℕ) : ℤ := 4 * a ^ 2 - 4 * b ^ 2

theorem operation_two_three : operation 2 3 = -20 :=
by
  sorry

end operation_two_three_l55_55288


namespace first_other_factor_of_lcm_l55_55135

theorem first_other_factor_of_lcm (A B hcf lcm : ℕ) (h1 : A = 368) (h2 : hcf = 23) (h3 : lcm = hcf * 16 * X) :
  X = 1 :=
by
  sorry

end first_other_factor_of_lcm_l55_55135


namespace fish_added_l55_55025

theorem fish_added (x : ℕ) (hx : x + (x - 4) = 20) : x - 4 = 8 := by
  sorry

end fish_added_l55_55025


namespace sector_central_angle_l55_55764

theorem sector_central_angle (r θ : ℝ) 
  (h1 : 1 = (1 / 2) * 2 * r) 
  (h2 : 2 = θ * r) : θ = 2 := 
sorry

end sector_central_angle_l55_55764


namespace matrix_no_solution_neg_two_l55_55169

-- Define the matrix and vector equation
def matrix_equation (a x y : ℝ) : Prop :=
  (a * x + 2 * y = a + 2) ∧ (2 * x + a * y = 2 * a)

-- Define the condition for no solution
def no_solution_condition (a : ℝ) : Prop :=
  (a/2 = 2/a) ∧ (a/2 ≠ (a + 2) / (2 * a))

-- Theorem stating that a = -2 is the necessary condition for no solution
theorem matrix_no_solution_neg_two (a : ℝ) : no_solution_condition a → a = -2 := by
  sorry

end matrix_no_solution_neg_two_l55_55169


namespace decompose_fraction1_decompose_fraction2_l55_55268

-- Define the first problem as a theorem
theorem decompose_fraction1 (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (2 / (x^2 - 1)) = (1 / (x - 1)) - (1 / (x + 1)) :=
sorry  -- Proof required

-- Define the second problem as a theorem
theorem decompose_fraction2 (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (2 * x / (x^2 - 1)) = (1 / (x - 1)) + (1 / (x + 1)) :=
sorry  -- Proof required

end decompose_fraction1_decompose_fraction2_l55_55268


namespace percentage_increase_l55_55931

theorem percentage_increase (regular_rate : ℝ) (regular_hours total_compensation total_hours_worked : ℝ)
  (h1 : regular_rate = 20)
  (h2 : regular_hours = 40)
  (h3 : total_compensation = 1000)
  (h4 : total_hours_worked = 45.714285714285715) :
  let overtime_hours := total_hours_worked - regular_hours
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := total_compensation - regular_pay
  let overtime_rate := overtime_pay / overtime_hours
  let percentage_increase := ((overtime_rate - regular_rate) / regular_rate) * 100
  percentage_increase = 75 := 
by
  sorry

end percentage_increase_l55_55931


namespace min_performances_l55_55249

theorem min_performances (n_pairs_per_show m n_singers : ℕ) (h1 : n_singers = 8) (h2 : n_pairs_per_show = 6) 
  (condition : 6 * m = 28 * 3) : m = 14 :=
by
  -- Use the assumptions to prove the statement
  sorry

end min_performances_l55_55249


namespace smallest_number_is_1013_l55_55329

def smallest_number_divisible (n : ℕ) : Prop :=
  n - 5 % Nat.lcm 12 (Nat.lcm 16 (Nat.lcm 18 (Nat.lcm 21 28))) = 0

theorem smallest_number_is_1013 : smallest_number_divisible 1013 :=
by
  sorry

end smallest_number_is_1013_l55_55329


namespace al_sandwiches_correct_l55_55264

-- Definitions based on the given conditions
def num_breads := 5
def num_meats := 7
def num_cheeses := 6
def total_combinations := num_breads * num_meats * num_cheeses

def turkey_swiss := num_breads -- disallowed turkey/Swiss cheese combinations
def multigrain_turkey := num_cheeses -- disallowed multi-grain bread/turkey combinations

def al_sandwiches := total_combinations - turkey_swiss - multigrain_turkey

-- The theorem to prove
theorem al_sandwiches_correct : al_sandwiches = 199 := 
by sorry

end al_sandwiches_correct_l55_55264


namespace dayAfter73DaysFromFridayAnd9WeeksLater_l55_55781

-- Define the days of the week as a data type
inductive Weekday
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
deriving DecidableEq, Repr

open Weekday

-- Function to calculate the day of the week after a given number of days
def addDays (start_day : Weekday) (days : ℕ) : Weekday :=
  match start_day with
  | Sunday    => match days % 7 with | 0 => Sunday    | 1 => Monday | 2 => Tuesday | 3 => Wednesday | 4 => Thursday | 5 => Friday | 6 => Saturday | _ => Sunday
  | Monday    => match days % 7 with | 0 => Monday    | 1 => Tuesday | 2 => Wednesday | 3 => Thursday | 4 => Friday | 5 => Saturday | 6 => Sunday | _ => Monday
  | Tuesday   => match days % 7 with | 0 => Tuesday   | 1 => Wednesday | 2 => Thursday | 3 => Friday | 4 => Saturday | 5 => Sunday | 6 => Monday | _ => Tuesday
  | Wednesday => match days % 7 with | 0 => Wednesday | 1 => Thursday | 2 => Friday | 3 => Saturday | 4 => Sunday | 5 => Monday | 6 => Tuesday | _ => Wednesday
  | Thursday  => match days % 7 with | 0 => Thursday  | 1 => Friday | 2 => Saturday | 3 => Sunday | 4 => Monday | 5 => Tuesday | 6 => Wednesday | _ => Thursday
  | Friday    => match days % 7 with | 0 => Friday    | 1 => Saturday | 2 => Sunday | 3 => Monday | 4 => Tuesday | 5 => Wednesday | 6 => Thursday | _ => Friday
  | Saturday  => match days % 7 with | 0 => Saturday  | 1 => Sunday | 2 => Monday | 3 => Tuesday | 4 => Wednesday | 5 => Thursday | 6 => Friday | _ => Saturday

-- Theorem that proves the required solution
theorem dayAfter73DaysFromFridayAnd9WeeksLater : addDays Friday 73 = Monday ∧ addDays Monday (9 * 7) = Monday := 
by
  -- Placeholder to acknowledge proof requirements
  sorry

end dayAfter73DaysFromFridayAnd9WeeksLater_l55_55781


namespace percent_change_area_decrease_l55_55534

theorem percent_change_area_decrease (L W : ℝ) (hL : L > 0) (hW : W > 0) :
    let A_initial := L * W
    let L_new := 1.60 * L
    let W_new := 0.40 * W
    let A_new := L_new * W_new
    let percent_change := (A_new - A_initial) / A_initial * 100
    percent_change = -36 :=
by
  sorry

end percent_change_area_decrease_l55_55534


namespace find_angle_BXY_l55_55923

noncomputable def angle_AXE (angle_CYX : ℝ) : ℝ := 3 * angle_CYX - 108

theorem find_angle_BXY
  (AB_parallel_CD : Prop)
  (h_parallel : ∀ (AXE CYX : ℝ), angle_AXE CYX = AXE)
  (x : ℝ) :
  (angle_AXE x = x) → x = 54 :=
by
  intro h₁
  unfold angle_AXE at h₁
  sorry

end find_angle_BXY_l55_55923


namespace factorize_expression_l55_55353

variable (x y : ℝ)

theorem factorize_expression : 9 * x^2 * y - y = y * (3 * x + 1) * (3 * x - 1) := 
by
  sorry

end factorize_expression_l55_55353


namespace income_M_l55_55986

variable (M N O : ℝ)

theorem income_M (h1 : (M + N) / 2 = 5050) 
                  (h2 : (N + O) / 2 = 6250) 
                  (h3 : (M + O) / 2 = 5200) : 
                  M = 2666.67 := 
by 
  sorry

end income_M_l55_55986


namespace three_sum_eq_nine_seven_five_l55_55362

theorem three_sum_eq_nine_seven_five {a b c : ℝ} 
    (h1 : b + c = 15 - 2 * a)
    (h2 : a + c = -10 - 4 * b)
    (h3 : a + b = 8 - 2 * c) : 
    3 * a + 3 * b + 3 * c = 9.75 := 
by
    sorry

end three_sum_eq_nine_seven_five_l55_55362


namespace robert_balls_l55_55315

theorem robert_balls (R T : ℕ) (hR : R = 25) (hT : T = 40 / 2) : R + T = 45 :=
by
  sorry

end robert_balls_l55_55315


namespace min_value_expression_l55_55360

theorem min_value_expression : ∀ x : ℝ, (x + 1)^2 * (x + 2)^2 * (x + 3)^2 * (x + 4)^2 + 2025 ≥ 3625 :=
by
  sorry

end min_value_expression_l55_55360


namespace train_distance_difference_l55_55697

theorem train_distance_difference (t : ℝ) (D₁ D₂ : ℝ)
(h_speed1 : D₁ = 20 * t)
(h_speed2 : D₂ = 25 * t)
(h_total_dist : D₁ + D₂ = 540) :
  D₂ - D₁ = 60 :=
by {
  -- These are the conditions as stated in step c)
  sorry
}

end train_distance_difference_l55_55697


namespace staircase_perimeter_l55_55513

theorem staircase_perimeter (area : ℝ) (side_length : ℝ) (num_sides : ℕ) (right_angles : Prop) :
  area = 85 ∧ side_length = 1 ∧ num_sides = 10 ∧ right_angles → 
  ∃ perimeter : ℝ, perimeter = 30.5 :=
by
  intro h
  sorry

end staircase_perimeter_l55_55513


namespace range_of_a_l55_55102

theorem range_of_a (a b : ℝ) (h : a - 4 * Real.sqrt b = 2 * Real.sqrt (a - b)) : 
  a ∈ {x | 0 ≤ x} ∧ ((a = 0) ∨ (4 ≤ a ∧ a ≤ 20)) :=
by
  sorry

end range_of_a_l55_55102


namespace number_of_feasible_networks_10_l55_55800

-- Definitions based on conditions
def feasible_networks (n : ℕ) : ℕ :=
if n = 0 then 1 else 2 ^ (n - 1)

-- The proof problem statement
theorem number_of_feasible_networks_10 : feasible_networks 10 = 512 := by
  -- proof goes here
  sorry

end number_of_feasible_networks_10_l55_55800


namespace James_watch_time_l55_55499

def Jeopardy_length : ℕ := 20
def Wheel_of_Fortune_length : ℕ := Jeopardy_length * 2
def Jeopardy_episodes : ℕ := 2
def Wheel_of_Fortune_episodes : ℕ := 2

theorem James_watch_time :
  (Jeopardy_episodes * Jeopardy_length + Wheel_of_Fortune_episodes * Wheel_of_Fortune_length) / 60 = 2 :=
by
  sorry

end James_watch_time_l55_55499


namespace distance_to_Tianbo_Mountain_l55_55155

theorem distance_to_Tianbo_Mountain : ∀ (x y : ℝ), 
  (x ≠ 0) ∧ 
  (y = 3) ∧ 
  (∀ v, v = (4 * y + x) * ((2 * x - 8) / v)) ∧ 
  (2 * (y * x) = 8 * y + x^2 - 4 * x) 
  → 
  (x + y = 9) := 
by
  sorry

end distance_to_Tianbo_Mountain_l55_55155


namespace petya_time_comparison_l55_55760

theorem petya_time_comparison (V a : ℝ) (hV : V > 0) (ha : a > 0) :
  (a / V) < ((a / (2.5 * V)) + (a / (1.6 * V))) := by
  have T_planned : ℝ := a / V
  have T_first_half : ℝ := a / (2.5 * V)
  have T_second_half : ℝ := a / (1.6 * V)
  have T_real : ℝ := T_first_half + T_second_half
  sorry

end petya_time_comparison_l55_55760


namespace technology_courses_correct_l55_55972

variable (m : ℕ)

def subject_courses := m
def arts_courses := subject_courses + 9
def technology_courses := 1 / 3 * arts_courses + 5

theorem technology_courses_correct : technology_courses = 1 / 3 * m + 8 := by
  sorry

end technology_courses_correct_l55_55972


namespace skirt_price_l55_55901

theorem skirt_price (S : ℝ) 
  (h1 : 2 * 5 = 10) 
  (h2 : 1 * 4 = 4) 
  (h3 : 6 * (5 / 2) = 15) 
  (h4 : 10 + 4 + 15 + 4 * S = 53) 
  : S = 6 :=
sorry

end skirt_price_l55_55901


namespace total_exercise_hours_l55_55409

theorem total_exercise_hours (natasha_minutes_per_day : ℕ) (natasha_days : ℕ)
  (esteban_minutes_per_day : ℕ) (esteban_days : ℕ)
  (h_n : natasha_minutes_per_day = 30) (h_nd : natasha_days = 7)
  (h_e : esteban_minutes_per_day = 10) (h_ed : esteban_days = 9) :
  (natasha_minutes_per_day * natasha_days + esteban_minutes_per_day * esteban_days) / 60 = 5 :=
by
  sorry

end total_exercise_hours_l55_55409


namespace basis_service_B_l55_55841

def vector := ℤ × ℤ

def not_collinear (v1 v2 : vector) : Prop :=
  v1.1 * v2.2 ≠ v1.2 * v2.1

def A : vector × vector := ((0, 0), (2, 3))
def B : vector × vector := ((-1, 3), (5, -2))
def C : vector × vector := ((3, 4), (6, 8))
def D : vector × vector := ((2, -3), (-2, 3))

theorem basis_service_B : not_collinear B.1 B.2 := by
  sorry

end basis_service_B_l55_55841


namespace number_of_green_hats_l55_55606

theorem number_of_green_hats 
  (B G : ℕ) 
  (h1 : B + G = 85) 
  (h2 : 6 * B + 7 * G = 550) 
  : G = 40 :=
sorry

end number_of_green_hats_l55_55606


namespace problem1_part1_problem1_part2_problem2_l55_55071

noncomputable def problem1_condition1 (m : ℕ) (a : ℕ) : Prop := 4^m = a
noncomputable def problem1_condition2 (n : ℕ) (b : ℕ) : Prop := 8^n = b

theorem problem1_part1 (m n a b : ℕ) (h1 : 4^m = a) (h2 : 8^n = b) : 2^(2*m + 3*n) = a * b :=
by sorry

theorem problem1_part2 (m n a b : ℕ) (h1 : 4^m = a) (h2 : 8^n = b) : 2^(4*m - 6*n) = (a^2) / (b^2) :=
by sorry

theorem problem2 (x : ℕ) (h : 2 * 8^x * 16 = 2^23) : x = 6 :=
by sorry

end problem1_part1_problem1_part2_problem2_l55_55071


namespace sphere_surface_area_l55_55938

theorem sphere_surface_area (V : ℝ) (π : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : V = 36 * π) 
  (h2 : V = (4 / 3) * π * r^3) 
  (h3 : A = 4 * π * r^2) 
  : A = 36 * π :=
by
  sorry

end sphere_surface_area_l55_55938


namespace max_profit_l55_55476

noncomputable def profit_function (x : ℕ) : ℝ :=
  if x ≤ 400 then
    300 * x - (1 / 2) * x^2 - 20000
  else
    60000 - 100 * x

theorem max_profit : 
  (∀ x ≥ 0, profit_function x ≤ 25000) ∧ (profit_function 300 = 25000) :=
by 
  sorry

end max_profit_l55_55476


namespace rhombus_longer_diagonal_length_l55_55740

theorem rhombus_longer_diagonal_length
  (side_length : ℕ) (shorter_diagonal : ℕ) 
  (side_length_eq : side_length = 53) 
  (shorter_diagonal_eq : shorter_diagonal = 50) : 
  ∃ longer_diagonal : ℕ, longer_diagonal = 94 := by
  sorry

end rhombus_longer_diagonal_length_l55_55740


namespace anthony_has_more_pairs_l55_55703

theorem anthony_has_more_pairs (scott_pairs : ℕ) (anthony_pairs : ℕ) (jim_pairs : ℕ) :
  (scott_pairs = 7) →
  (anthony_pairs = 3 * scott_pairs) →
  (jim_pairs = anthony_pairs - 2) →
  (anthony_pairs - jim_pairs = 2) :=
by
  intro h_scott h_anthony h_jim
  sorry

end anthony_has_more_pairs_l55_55703


namespace main_inequality_l55_55701

theorem main_inequality (a b c d : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_d_pos : 0 < d) (h_sum : a + b + c + d = 1) :
  (b * c * d) / (1 - a)^2 + (c * d * a) / (1 - b)^2 + (d * a * b) / (1 - c)^2 + (a * b * c) / (1 - d)^2 ≤ 1 / 9 :=
by
  sorry

end main_inequality_l55_55701


namespace center_and_radius_of_circle_l55_55090

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 6 * y + 6 = 0

-- State the theorem
theorem center_and_radius_of_circle :
  (∃ x₀ y₀ r, (∀ x y, circle_eq x y ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ∧
  x₀ = 1 ∧ y₀ = -3 ∧ r = 2) :=
by
  -- Proof is omitted
  sorry

end center_and_radius_of_circle_l55_55090


namespace shopkeeper_loss_percent_l55_55345

theorem shopkeeper_loss_percent
  (initial_value : ℝ)
  (profit_percent : ℝ)
  (loss_percent : ℝ)
  (remaining_value_percent : ℝ)
  (profit_percent_10 : profit_percent = 0.10)
  (loss_percent_70 : loss_percent = 0.70)
  (initial_value_100 : initial_value = 100)
  (remaining_value_percent_30 : remaining_value_percent = 0.30)
  (selling_price : ℝ := initial_value * (1 + profit_percent))
  (remaining_value : ℝ := initial_value * remaining_value_percent)
  (remaining_selling_price : ℝ := remaining_value * (1 + profit_percent))
  (loss_value : ℝ := initial_value - remaining_selling_price)
  (shopkeeper_loss_percent : ℝ := loss_value / initial_value * 100) : 
  shopkeeper_loss_percent = 67 :=
sorry

end shopkeeper_loss_percent_l55_55345


namespace isosceles_triangle_l55_55847

theorem isosceles_triangle
  (α β γ : ℝ)
  (triangle_sum : α + β + γ = Real.pi)
  (second_triangle_angle1 : α + β < Real.pi)
  (second_triangle_angle2 : α + γ < Real.pi) :
  β = γ := 
sorry

end isosceles_triangle_l55_55847


namespace factor_expression_l55_55182

theorem factor_expression (x : ℝ) : 12 * x ^ 2 + 8 * x = 4 * x * (3 * x + 2) :=
by
  sorry

end factor_expression_l55_55182


namespace g_at_3_l55_55991

noncomputable def g (x : ℝ) : ℝ := sorry

theorem g_at_3 (h : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = 2) : g 3 = 0 := by
  sorry

end g_at_3_l55_55991


namespace total_crayons_l55_55045

def original_crayons := 41
def added_crayons := 12

theorem total_crayons : original_crayons + added_crayons = 53 := by
  sorry

end total_crayons_l55_55045


namespace xiao_ming_actual_sleep_time_l55_55274

def required_sleep_time : ℝ := 9
def recorded_excess_sleep_time : ℝ := 0.4
def actual_sleep_time (required : ℝ) (excess : ℝ) : ℝ := required + excess

theorem xiao_ming_actual_sleep_time :
  actual_sleep_time required_sleep_time recorded_excess_sleep_time = 9.4 := 
by
  sorry

end xiao_ming_actual_sleep_time_l55_55274


namespace zach_fill_time_l55_55916

theorem zach_fill_time : 
  ∀ (t : ℕ), 
  (∀ (max_time max_rate zach_rate popped total : ℕ), 
    max_time = 30 → 
    max_rate = 2 → 
    zach_rate = 3 → 
    popped = 10 → 
    total = 170 → 
    (max_time * max_rate + t * zach_rate - popped = total) → 
    t = 40) := 
sorry

end zach_fill_time_l55_55916


namespace min_value_expression_l55_55623

theorem min_value_expression : ∀ x y : ℝ, 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 9 :=
by
  intro x y
  sorry

end min_value_expression_l55_55623


namespace least_possible_multiple_l55_55871

theorem least_possible_multiple (x y z k : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hk : 1 ≤ k)
  (h1 : 3 * x = k * z) (h2 : 4 * y = k * z) (h3 : x - y + z = 19) : 3 * x = 12 :=
by
  sorry

end least_possible_multiple_l55_55871


namespace jim_saves_money_by_buying_gallon_l55_55757

theorem jim_saves_money_by_buying_gallon :
  let gallon_price := 8
  let bottle_price := 3
  let ounces_per_gallon := 128
  let ounces_per_bottle := 16
  (ounces_per_gallon / ounces_per_bottle) * bottle_price - gallon_price = 16 :=
by
  sorry

end jim_saves_money_by_buying_gallon_l55_55757


namespace nine_point_circle_equation_l55_55849

theorem nine_point_circle_equation 
  (α β γ : ℝ) 
  (x y z : ℝ) :
  (x^2 * (Real.sin α) * (Real.cos α) + y^2 * (Real.sin β) * (Real.cos β) + z^2 * (Real.sin γ) * (Real.cos γ) = 
  y * z * (Real.sin α) + x * z * (Real.sin β) + x * y * (Real.sin γ))
:= sorry

end nine_point_circle_equation_l55_55849


namespace fermats_little_theorem_poly_binom_coeff_divisible_by_prime_l55_55061

variable (p : ℕ) [Fact (Nat.Prime p)]

theorem fermats_little_theorem_poly (X : ℤ) :
  (X + 1) ^ p = X ^ p + 1 := by
    sorry

theorem binom_coeff_divisible_by_prime {k : ℕ} (hkp : 1 ≤ k ∧ k < p) :
  p ∣ Nat.choose p k := by
    sorry

end fermats_little_theorem_poly_binom_coeff_divisible_by_prime_l55_55061


namespace bruised_more_than_wormy_l55_55331

noncomputable def total_apples : ℕ := 85
noncomputable def fifth_of_apples (n : ℕ) : ℕ := n / 5
noncomputable def apples_left_to_eat_raw : ℕ := 42

noncomputable def wormy_apples : ℕ := fifth_of_apples total_apples
noncomputable def total_non_raw_eatable_apples : ℕ := total_apples - apples_left_to_eat_raw
noncomputable def bruised_apples : ℕ := total_non_raw_eatable_apples - wormy_apples

theorem bruised_more_than_wormy :
  bruised_apples - wormy_apples = 43 - 17 :=
by sorry

end bruised_more_than_wormy_l55_55331


namespace area_of_annulus_l55_55344

-- Define the conditions
def concentric_circles (r s : ℝ) (h : r > s) (x : ℝ) := 
  r^2 = s^2 + x^2

-- State the theorem
theorem area_of_annulus (r s x : ℝ) (h : r > s) (h₁ : concentric_circles r s h x) :
  π * x^2 = π * r^2 - π * s^2 :=
by 
  rw [concentric_circles] at h₁
  sorry

end area_of_annulus_l55_55344


namespace ratio_of_tetrahedron_to_cube_volume_l55_55394

theorem ratio_of_tetrahedron_to_cube_volume (x : ℝ) (hx : 0 < x) :
  let V_cube := x^3
  let a_tetrahedron := (x * Real.sqrt 3) / 2
  let V_tetrahedron := (a_tetrahedron^3 * Real.sqrt 2) / 12
  (V_tetrahedron / V_cube) = (Real.sqrt 6 / 32) :=
by
  sorry

end ratio_of_tetrahedron_to_cube_volume_l55_55394


namespace value_of_expression_l55_55564

def g (x : ℝ) (p q r s t : ℝ) : ℝ :=
  p * x^4 + q * x^3 + r * x^2 + s * x + t

theorem value_of_expression (p q r s t : ℝ) (h : g (-1) p q r s t = 4) :
  12 * p - 6 * q + 3 * r - 2 * s + t = 13 :=
sorry

end value_of_expression_l55_55564


namespace problem_l55_55124

theorem problem (r : ℝ) (h : (r + 1/r)^4 = 17) : r^6 + 1/r^6 = 1 * Real.sqrt 17 - 6 :=
sorry

end problem_l55_55124


namespace work_days_l55_55266

theorem work_days (A B C : ℝ) (h₁ : A + B = 1 / 15) (h₂ : C = 1 / 7.5) : 1 / (A + B + C) = 5 :=
by
  sorry

end work_days_l55_55266


namespace correct_condition_l55_55640

section proof_problem

variable (a : ℝ)

def cond1 : Prop := (a ^ 6 / a ^ 3 = a ^ 2)
def cond2 : Prop := (2 * a ^ 2 + 3 * a ^ 3 = 5 * a ^ 5)
def cond3 : Prop := (a ^ 4 * a ^ 2 = a ^ 8)
def cond4 : Prop := ((-a ^ 3) ^ 2 = a ^ 6)

theorem correct_condition : cond4 a :=
by
  sorry

end proof_problem

end correct_condition_l55_55640


namespace no_solution_l55_55482

theorem no_solution (n : ℕ) (k : ℕ) (hn : Prime n) (hk : 0 < k) :
  ¬ (n ≤ n.factorial - k ^ n ∧ n.factorial - k ^ n ≤ k * n) :=
by
  sorry

end no_solution_l55_55482


namespace pell_eq_unique_fund_sol_l55_55899

theorem pell_eq_unique_fund_sol (x y x_0 y_0 : ℕ) 
  (h1 : x_0^2 - 2003 * y_0^2 = 1) 
  (h2 : ∀ x y, x > 0 ∧ y > 0 → x^2 - 2003 * y^2 = 1 → ∃ n : ℕ, x + Real.sqrt 2003 * y = (x_0 + Real.sqrt 2003 * y_0)^n)
  (hx_pos : x > 0) 
  (hy_pos : y > 0)
  (h_sol : x^2 - 2003 * y^2 = 1) 
  (hprime : ∀ p : ℕ, Prime p → p ∣ x → p ∣ x_0)
  : x = x_0 ∧ y = y_0 :=
sorry

end pell_eq_unique_fund_sol_l55_55899


namespace M1_M2_product_l55_55313

theorem M1_M2_product (M_1 M_2 : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 →
  (42 * x - 51) / (x^2 - 5 * x + 6) = (M_1 / (x - 2)) + (M_2 / (x - 3))) →
  M_1 * M_2 = -2981.25 :=
by
  intros h
  sorry

end M1_M2_product_l55_55313


namespace revenue_percentage_l55_55889

theorem revenue_percentage (R C : ℝ) (hR_pos : R > 0) (hC_pos : C > 0) :
  let projected_revenue := 1.20 * R
  let actual_revenue := 0.75 * R
  (actual_revenue / projected_revenue) * 100 = 62.5 := by
  sorry

end revenue_percentage_l55_55889


namespace square_perimeter_l55_55855

theorem square_perimeter (s : ℝ) (h₁ : s^2 = 625) : 4 * s = 100 := 
sorry

end square_perimeter_l55_55855


namespace intersection_of_A_and_B_l55_55173

def setA (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1
def setB (x : ℝ) : Prop := 0 < x ∧ x ≤ 2
def setIntersection (x : ℝ) : Prop := 0 < x ∧ x ≤ 1

theorem intersection_of_A_and_B :
  ∀ x, (setA x ∧ setB x) ↔ setIntersection x := 
by sorry

end intersection_of_A_and_B_l55_55173


namespace units_digit_of_large_power_l55_55843

theorem units_digit_of_large_power
  (units_147_1997_pow2999: ℕ) 
  (h1 : units_147_1997_pow2999 = (147 ^ 1997) % 10)
  (h2 : ∀ k, (7 ^ (k * 4 + 1)) % 10 = 7)
  (h3 : ∀ m, (7 ^ (m * 4 + 3)) % 10 = 3)
  : units_147_1997_pow2999 % 10 = 3 :=
sorry

end units_digit_of_large_power_l55_55843


namespace min_fraction_sum_l55_55039

theorem min_fraction_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) : 
  (∀ x, x = 1 / m + 2 / n → x ≥ 8) :=
  sorry

end min_fraction_sum_l55_55039


namespace meaningful_expression_iff_l55_55518

theorem meaningful_expression_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (Real.sqrt (x - 3))) ↔ x > 3 := by
  sorry

end meaningful_expression_iff_l55_55518


namespace shapeB_is_symmetric_to_original_l55_55380

-- Assume a simple type to represent our shapes
inductive Shape
| shapeA
| shapeB
| shapeC
| shapeD
| shapeE
| originalShape

-- Define the symmetry condition
def is_symmetric (s1 s2 : Shape) : Prop := sorry  -- this would be the condition to check symmetry

-- The theorem to prove that shapeB is symmetric to the original shape
theorem shapeB_is_symmetric_to_original :
  is_symmetric Shape.shapeB Shape.originalShape :=
sorry

end shapeB_is_symmetric_to_original_l55_55380


namespace halfway_between_3_4_and_5_7_l55_55660

-- Define the two fractions
def frac1 := 3/4
def frac2 := 5/7

-- Define the average function for two fractions
def halfway_fract (a b : ℚ) : ℚ := (a + b) / 2

-- Prove that the halfway fraction between 3/4 and 5/7 is 41/56
theorem halfway_between_3_4_and_5_7 : 
  halfway_fract frac1 frac2 = 41/56 := 
by 
  sorry

end halfway_between_3_4_and_5_7_l55_55660


namespace transformed_function_zero_l55_55231

-- Definitions based on conditions
def f : ℝ → ℝ → ℝ := sorry  -- Assume this is the given function f(x, y)

-- Transformed function according to symmetry and reflections
def transformed_f (x y : ℝ) : Prop := f (y + 2) (x - 2) = 0

-- Lean statement to be proved
theorem transformed_function_zero (x y : ℝ) : transformed_f x y := sorry

end transformed_function_zero_l55_55231


namespace percentage_reduction_l55_55645

theorem percentage_reduction 
  (original_employees : ℝ)
  (new_employees : ℝ)
  (h1 : original_employees = 208.04597701149424)
  (h2 : new_employees = 181) :
  ((original_employees - new_employees) / original_employees) * 100 = 13.00 :=
by
  sorry

end percentage_reduction_l55_55645


namespace num_ordered_triples_l55_55914

theorem num_ordered_triples :
  (∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a ∣ b ∧ a ∣ c ∧ a + b + c = 100) :=
  sorry

end num_ordered_triples_l55_55914


namespace abs_diff_l55_55227

theorem abs_diff (m n : ℝ) (h_avg : (m + n + 9 + 8 + 10) / 5 = 9) (h_var : ((m^2 + n^2 + 81 + 64 + 100) / 5) - 81 = 2) :
  |m - n| = 4 := by
  sorry

end abs_diff_l55_55227


namespace percentage_silver_cars_after_shipment_l55_55075

-- Definitions for conditions
def initialCars : ℕ := 40
def initialSilverPerc : ℝ := 0.15
def newShipmentCars : ℕ := 80
def newShipmentNonSilverPerc : ℝ := 0.30

-- Proof statement that needs to be proven
theorem percentage_silver_cars_after_shipment :
  let initialSilverCars := initialSilverPerc * initialCars
  let newShipmentSilverPerc := 1 - newShipmentNonSilverPerc
  let newShipmentSilverCars := newShipmentSilverPerc * newShipmentCars
  let totalSilverCars := initialSilverCars + newShipmentSilverCars
  let totalCars := initialCars + newShipmentCars
  (totalSilverCars / totalCars) * 100 = 51.67 :=
by
  sorry

end percentage_silver_cars_after_shipment_l55_55075


namespace time_difference_leak_l55_55172

/-- 
The machine usually fills one barrel in 3 minutes. 
However, with a leak, it takes 5 minutes to fill one barrel. 
Given that it takes 24 minutes longer to fill 12 barrels with the leak, prove that it will take 2n minutes longer to fill n barrels with the leak.
-/
theorem time_difference_leak (n : ℕ) : 
  (3 * 12 + 24 = 5 * 12) →
  (5 * n) - (3 * n) = 2 * n :=
by
  intros h
  sorry

end time_difference_leak_l55_55172


namespace line_intersects_ellipse_possible_slopes_l55_55014

theorem line_intersects_ellipse_possible_slopes :
  ∀ m : ℝ, (∃ x y : ℝ, y = m * x + 3 ∧ 4 * x^2 + 25 * y^2 = 100) ↔
    (m ≤ -Real.sqrt (1 / 20) ∨ m ≥ Real.sqrt (1 / 20)) :=
by
  sorry

end line_intersects_ellipse_possible_slopes_l55_55014


namespace extreme_points_inequality_l55_55093

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 + m * Real.log (1 + x)

-- Given m > 0 and f(x) has extreme points x1 and x2 such that x1 < x2
theorem extreme_points_inequality {m x1 x2 : ℝ} (h_m : m > 0)
    (h_extreme1 : x1 = (-1 - Real.sqrt (1 - 2 * m)) / 2)
    (h_extreme2 : x2 = (-1 + Real.sqrt (1 - 2 * m)) / 2)
    (h_order : x1 < x2) :
    2 * f x2 m > -x1 + 2 * x1 * Real.log 2 := sorry

end extreme_points_inequality_l55_55093


namespace intersection_point_parallel_line_through_intersection_l55_55114

-- Definitions for the problem
def l1 (x y : ℝ) : Prop := x + 8 * y + 7 = 0
def l2 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def l3 (x y : ℝ) : Prop := x + y + 1 = 0
def parallel (x y c : ℝ) : Prop := x + y + c = 0
def point (x y : ℝ) : Prop := x = 1 ∧ y = -1

-- (1) Proof that the intersection point of l1 and l2 is (1, -1)
theorem intersection_point : ∃ (x y : ℝ), l1 x y ∧ l2 x y ∧ point x y :=
by 
  sorry

-- (2) Proof that the line passing through the intersection point of l1 and l2
-- which is parallel to l3 is x + y = 0
theorem parallel_line_through_intersection : ∃ (c : ℝ), parallel 1 (-1) c ∧ c = 0 :=
by 
  sorry

end intersection_point_parallel_line_through_intersection_l55_55114


namespace annual_interest_rate_l55_55294

noncomputable def compound_interest_rate (P A : ℝ) (n t : ℕ) (r : ℝ) : Prop :=
  A = P * (1 + r / n) ^ (n * t)

theorem annual_interest_rate :
  compound_interest_rate 150 181.50 2 1 (0.2 : ℝ) :=
by
  unfold compound_interest_rate
  sorry

end annual_interest_rate_l55_55294


namespace find_z_l55_55988

noncomputable def w : ℝ := sorry
noncomputable def x : ℝ := (5 * w) / 4
noncomputable def y : ℝ := 1.40 * w

theorem find_z (z : ℝ) : x = (1 - z / 100) * y → z = 10.71 :=
by
  sorry

end find_z_l55_55988


namespace y_days_do_work_l55_55656

theorem y_days_do_work (d : ℝ) (h : (1 / 30) + (1 / d) = 1 / 18) : d = 45 := 
by
  sorry

end y_days_do_work_l55_55656


namespace parallel_lines_slope_l55_55741

theorem parallel_lines_slope (d : ℝ) (h : 3 = 4 * d) : d = 3 / 4 :=
by
  sorry

end parallel_lines_slope_l55_55741


namespace num_five_dollar_coins_l55_55878

theorem num_five_dollar_coins (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 5 * y = 125) : y = 15 :=
by
  sorry -- Proof to be completed

end num_five_dollar_coins_l55_55878


namespace max_value_y_l55_55510

noncomputable def y (x : ℝ) : ℝ := 3 - 3*x - 1/x

theorem max_value_y : (∃ x > 0, ∀ x' > 0, y x' ≤ y x) ∧ (y (1 / Real.sqrt 3) = 3 - 2 * Real.sqrt 3) :=
by
  sorry

end max_value_y_l55_55510


namespace equilateral_triangle_grid_l55_55422

noncomputable def number_of_triangles (n : ℕ) : ℕ :=
1 + 3 + 5 + 7 + 9 + 1 + 2 + 3 + 4 + 3 + 1 + 2 + 3 + 1 + 2 + 1

theorem equilateral_triangle_grid (n : ℕ) (h : n = 5) : number_of_triangles n = 48 := by
  sorry

end equilateral_triangle_grid_l55_55422


namespace consecutive_numbers_count_l55_55174

-- Definitions and conditions
variables (n : ℕ) (x : ℕ)
axiom avg_condition : (2 * 33 = 2 * x + n - 1)
axiom highest_num_condition : (x + (n - 1) = 36)

-- Thm statement
theorem consecutive_numbers_count : n = 7 :=
by
  sorry

end consecutive_numbers_count_l55_55174


namespace john_volunteer_hours_l55_55736

noncomputable def total_volunteer_hours :=
  let first_six_months_hours := 2 * 3 * 6
  let next_five_months_hours := 1 * 2 * 4 * 5
  let december_hours := 3 * 2
  first_six_months_hours + next_five_months_hours + december_hours

theorem john_volunteer_hours : total_volunteer_hours = 82 := by
  sorry

end john_volunteer_hours_l55_55736


namespace arithmetic_geometric_sequence_l55_55066

theorem arithmetic_geometric_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (q : ℕ)
  (h₀ : ∀ n, a n = 2^(n-1))
  (h₁ : a 1 = 1)
  (h₂ : a 1 + a 2 + a 3 = 7)
  (h₃ : q > 0) :
  (∀ n, a n = 2^(n-1)) ∧ (∀ n, S n = 2^n - 1) :=
by {
  sorry
}

end arithmetic_geometric_sequence_l55_55066


namespace area_of_quadrilateral_is_195_l55_55573

-- Definitions and conditions
def diagonal_length : ℝ := 26
def offset1 : ℝ := 9
def offset2 : ℝ := 6

-- Prove the area of the quadrilateral is 195 cm²
theorem area_of_quadrilateral_is_195 :
  1 / 2 * diagonal_length * offset1 + 1 / 2 * diagonal_length * offset2 = 195 := 
by
  -- The proof steps would go here
  sorry

end area_of_quadrilateral_is_195_l55_55573


namespace part1_part2_l55_55587

variable (a : ℝ)

-- Defining the set A
def setA (a : ℝ) : Set ℝ := { x : ℝ | (x - 2) * (x - 3 * a - 1) < 0 }

-- Part 1: For a = 2, setB should be {x | 2 < x < 7}
theorem part1 : setA 2 = { x : ℝ | 2 < x ∧ x < 7 } :=
by
  sorry

-- Part 2: If setA a = setB, then a = -1
theorem part2 (B : Set ℝ) (h : setA a = B) : a = -1 :=
by
  sorry

end part1_part2_l55_55587


namespace perimeter_of_regular_pentagon_is_75_l55_55722

-- Define the side length and the property of the figure
def side_length : ℝ := 15
def is_regular_pentagon : Prop := true  -- assuming this captures the regular pentagon property

-- Define the perimeter calculation based on the conditions
def perimeter (n : ℕ) (side_length : ℝ) := n * side_length

-- The theorem to prove
theorem perimeter_of_regular_pentagon_is_75 :
  is_regular_pentagon → perimeter 5 side_length = 75 :=
by
  intro _ -- We don't need to use is_regular_pentagon directly
  rw [side_length]
  norm_num
  sorry

end perimeter_of_regular_pentagon_is_75_l55_55722


namespace division_remainder_l55_55190

theorem division_remainder : 4053 % 23 = 5 :=
by
  sorry

end division_remainder_l55_55190


namespace monotonic_intervals_max_min_values_l55_55877

def f (x : ℝ) := x^3 - 3*x
def f_prime (x : ℝ) := 3*(x-1)*(x+1)

theorem monotonic_intervals :
  (∀ x : ℝ, x < -1 → 0 < f_prime x) ∧ (∀ x : ℝ, -1 < x ∧ x < 1 → f_prime x < 0) ∧ (∀ x : ℝ, x > 1 → 0 < f_prime x) :=
  by
  sorry

theorem max_min_values :
  ∀ x ∈ Set.Icc (-1 : ℝ) 3, f x ≤ 18 ∧ f x ≥ -2 ∧ 
  (f 1 = -2) ∧
  (f 3 = 18) :=
  by
  sorry

end monotonic_intervals_max_min_values_l55_55877


namespace min_value_fraction_l55_55234

theorem min_value_fraction (x : ℝ) (h : x > 9) : (x^2 + 81) / (x - 9) ≥ 27 := 
  sorry

end min_value_fraction_l55_55234


namespace tangent_line_through_origin_l55_55033

theorem tangent_line_through_origin (f : ℝ → ℝ) (x : ℝ) (H1 : ∀ x < 0, f x = Real.log (-x))
  (H2 : ∀ x < 0, DifferentiableAt ℝ f x) (H3 : ∀ (x₀ : ℝ), x₀ < 0 → x₀ = -Real.exp 1 → deriv f x₀ = -1 / Real.exp 1)
  : ∀ x, -Real.exp 1 = x → ∀ y, y = -1 / Real.exp 1 * x → y = 0 → y = -1 / Real.exp 1 * x :=
by
  sorry

end tangent_line_through_origin_l55_55033


namespace slope_of_line_determined_by_solutions_l55_55076

theorem slope_of_line_determined_by_solutions (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : 3 / x₁ +  4 / y₁ = 0)
  (h₂ : 3 / x₂ + 4 / y₂ = 0) :
  (y₂ - y₁) / (x₂ - x₁) = -4 / 3 :=
sorry

end slope_of_line_determined_by_solutions_l55_55076


namespace angle_bc_l55_55411

variables (a b c : ℝ → ℝ → Prop) (theta : ℝ)

-- Definitions of parallelism and angle conditions
def parallel (x y : ℝ → ℝ → Prop) : Prop := ∀ p q r s : ℝ, x p q → y r s → p - q = r - s

def angle_between (x y : ℝ → ℝ → Prop) (θ : ℝ) : Prop := sorry  -- Assume we have a definition for angle between lines

-- Given conditions
axiom parallel_ab : parallel a b
axiom angle_ac : angle_between a c theta

-- Theorem statement
theorem angle_bc : angle_between b c theta :=
sorry

end angle_bc_l55_55411


namespace trip_first_part_length_l55_55867

theorem trip_first_part_length
  (total_distance : ℝ := 50)
  (first_speed : ℝ := 66)
  (second_speed : ℝ := 33)
  (average_speed : ℝ := 44) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ total_distance) ∧ 44 = total_distance / (x / first_speed + (total_distance - x) / second_speed) ∧ x = 25 :=
by
  sorry

end trip_first_part_length_l55_55867


namespace product_of_numbers_larger_than_reciprocal_eq_neg_one_l55_55195

theorem product_of_numbers_larger_than_reciprocal_eq_neg_one :
  ∃ x y : ℝ, x ≠ y ∧ (x = 1 / x + 2) ∧ (y = 1 / y + 2) ∧ x * y = -1 :=
by
  sorry

end product_of_numbers_larger_than_reciprocal_eq_neg_one_l55_55195


namespace find_m_range_of_x_l55_55775

def f (m x : ℝ) : ℝ := (m^2 - 1) * x + m^2 - 3 * m + 2

theorem find_m (m : ℝ) (H_dec : m^2 - 1 < 0) (H_f1 : f m 1 = 0) : 
  m = 1 / 2 :=
sorry

theorem range_of_x (x : ℝ) :
  f (1 / 2) (x + 1) ≥ x^2 ↔ -3 / 4 ≤ x ∧ x ≤ 0 :=
sorry

end find_m_range_of_x_l55_55775


namespace sarah_homework_problems_l55_55658

theorem sarah_homework_problems (math_pages reading_pages problems_per_page : ℕ) 
  (h1 : math_pages = 4) 
  (h2 : reading_pages = 6) 
  (h3 : problems_per_page = 4) : 
  (math_pages + reading_pages) * problems_per_page = 40 :=
by 
  sorry

end sarah_homework_problems_l55_55658


namespace average_score_group2_l55_55297

-- Total number of students
def total_students : ℕ := 50

-- Overall average score
def overall_average_score : ℝ := 92

-- Number of students from 1 to 30
def group1_students : ℕ := 30

-- Average score of students from 1 to 30
def group1_average_score : ℝ := 90

-- Total number of students - group1_students = 50 - 30 = 20
def group2_students : ℕ := total_students - group1_students

-- Lean 4 statement to prove the average score of students with student numbers 31 to 50 is 95
theorem average_score_group2 :
  (overall_average_score * total_students = group1_average_score * group1_students + x * group2_students) →
  x = 95 :=
sorry

end average_score_group2_l55_55297


namespace cube_volume_edge_length_range_l55_55495

theorem cube_volume_edge_length_range (a : ℝ) (h : a^3 = 9) : 2 < a ∧ a < 2.5 :=
by {
    -- proof will go here
    sorry
}

end cube_volume_edge_length_range_l55_55495


namespace greatest_term_in_expansion_l55_55906

theorem greatest_term_in_expansion :
  ∃ k : ℕ, k = 63 ∧
  (∀ n : ℕ, n ∈ (Finset.range 101) → n ≠ k → 
    (Nat.choose 100 n * (Real.sqrt 3)^n) < 
    (Nat.choose 100 k * (Real.sqrt 3)^k)) :=
by
  sorry

end greatest_term_in_expansion_l55_55906


namespace math_expression_evaluation_l55_55082

theorem math_expression_evaluation :
  36 + (120 / 15) + (15 * 19) - 150 - (450 / 9) = 129 :=
by
  sorry

end math_expression_evaluation_l55_55082


namespace x_intercept_of_line_l2_l55_55441

theorem x_intercept_of_line_l2 :
  ∀ (l1 l2 : ℝ → ℝ),
  (∀ x y, 2 * x - y + 3 = 0 → l1 x = y) →
  (∀ x y, 2 * x - y - 6 = 0 → l2 x = y) →
  l1 0 = 6 →
  l2 0 = -6 →
  l2 3 = 0 :=
by
  sorry

end x_intercept_of_line_l2_l55_55441


namespace slices_with_all_toppings_l55_55171

theorem slices_with_all_toppings (p m o a b c x total : ℕ) 
  (pepperoni_slices : p = 8)
  (mushrooms_slices : m = 12)
  (olives_slices : o = 14)
  (total_slices : total = 16)
  (inclusion_exclusion : p + m + o - a - b - c - 2 * x = total) :
  x = 4 := 
by
  rw [pepperoni_slices, mushrooms_slices, olives_slices, total_slices] at inclusion_exclusion
  sorry

end slices_with_all_toppings_l55_55171


namespace jen_visits_exactly_two_countries_l55_55771

noncomputable def probability_of_visiting_exactly_two_countries (p_chile p_madagascar p_japan p_egypt : ℝ) : ℝ :=
  let p_chile_madagascar := (p_chile * p_madagascar) * (1 - p_japan) * (1 - p_egypt)
  let p_chile_japan := (p_chile * p_japan) * (1 - p_madagascar) * (1 - p_egypt)
  let p_chile_egypt := (p_chile * p_egypt) * (1 - p_madagascar) * (1 - p_japan)
  let p_madagascar_japan := (p_madagascar * p_japan) * (1 - p_chile) * (1 - p_egypt)
  let p_madagascar_egypt := (p_madagascar * p_egypt) * (1 - p_chile) * (1 - p_japan)
  let p_japan_egypt := (p_japan * p_egypt) * (1 - p_chile) * (1 - p_madagascar)
  p_chile_madagascar + p_chile_japan + p_chile_egypt + p_madagascar_japan + p_madagascar_egypt + p_japan_egypt

theorem jen_visits_exactly_two_countries :
  probability_of_visiting_exactly_two_countries 0.4 0.35 0.2 0.15 = 0.2432 :=
by
  sorry

end jen_visits_exactly_two_countries_l55_55771


namespace necessary_but_not_sufficient_l55_55810

variable {a b : ℝ}

theorem necessary_but_not_sufficient : (a < b + 1) ∧ ¬ (a < b + 1 → a < b) :=
by
  sorry

end necessary_but_not_sufficient_l55_55810


namespace total_toucans_l55_55984

def initial_toucans : Nat := 2

def new_toucans : Nat := 1

theorem total_toucans : initial_toucans + new_toucans = 3 := by
  sorry

end total_toucans_l55_55984


namespace mr_blue_expected_rose_petals_l55_55779

def mr_blue_flower_bed_rose_petals (length_paces : ℕ) (width_paces : ℕ) (pace_length_ft : ℝ) (petals_per_sqft : ℝ) : ℝ :=
  let length_ft := length_paces * pace_length_ft
  let width_ft := width_paces * pace_length_ft
  let area_sqft := length_ft * width_ft
  area_sqft * petals_per_sqft

theorem mr_blue_expected_rose_petals :
  mr_blue_flower_bed_rose_petals 18 24 1.5 0.4 = 388.8 :=
by
  simp [mr_blue_flower_bed_rose_petals]
  norm_num

end mr_blue_expected_rose_petals_l55_55779


namespace geometric_progression_l55_55023

theorem geometric_progression (b q : ℝ) :
  (b + b*q + b*q^2 + b*q^3 = -40) ∧ 
  (b^2 + (b*q)^2 + (b*q^2)^2 + (b*q^3)^2 = 3280) →
  (b = 2 ∧ q = -3) ∨ (b = -54 ∧ q = -1/3) :=
by sorry

end geometric_progression_l55_55023


namespace change_is_24_l55_55971

-- Define the prices and quantities
def price_basketball_card : ℕ := 3
def price_baseball_card : ℕ := 4
def num_basketball_cards : ℕ := 2
def num_baseball_cards : ℕ := 5
def money_paid : ℕ := 50

-- Define the total cost
def total_cost : ℕ := (num_basketball_cards * price_basketball_card) + (num_baseball_cards * price_baseball_card)

-- Define the change received
def change_received : ℕ := money_paid - total_cost

-- Prove that the change received is $24
theorem change_is_24 : change_received = 24 := by
  -- the proof will go here
  sorry

end change_is_24_l55_55971


namespace simplify_expression_l55_55177

theorem simplify_expression (a : ℝ) (h : a < 1 / 4) : 4 * (4 * a - 1)^2 = (1 - 4 * a) ^ 2 :=
by
  sorry

end simplify_expression_l55_55177


namespace total_difference_in_cents_l55_55582

variable (q : ℕ)

def charles_quarters := 6 * q + 2
def charles_dimes := 3 * q - 2

def richard_quarters := 2 * q + 10
def richard_dimes := 4 * q + 3

def cents_from_quarters (n : ℕ) : ℕ := 25 * n
def cents_from_dimes (n : ℕ) : ℕ := 10 * n

theorem total_difference_in_cents : 
  (cents_from_quarters (charles_quarters q) + cents_from_dimes (charles_dimes q)) - 
  (cents_from_quarters (richard_quarters q) + cents_from_dimes (richard_dimes q)) = 
  90 * q - 250 :=
by
  sorry

end total_difference_in_cents_l55_55582


namespace XiaoKang_min_sets_pushups_pullups_l55_55204

theorem XiaoKang_min_sets_pushups_pullups (x y : ℕ) (hx : x ≥ 100) (hy : y ≥ 106) (h : 8 * x + 5 * y = 9050) :
  x ≥ 100 ∧ y ≥ 106 :=
by {
  sorry  -- proof not required as per instruction
}

end XiaoKang_min_sets_pushups_pullups_l55_55204


namespace layla_goals_l55_55584

variable (L K : ℕ)
variable (average_score : ℕ := 92)
variable (goals_difference : ℕ := 24)
variable (total_games : ℕ := 4)

theorem layla_goals :
  K = L - goals_difference →
  (L + K) = (average_score * total_games) →
  L = 196 :=
by
  sorry

end layla_goals_l55_55584


namespace coefficient_of_x4_l55_55672

theorem coefficient_of_x4 (n : ℕ) (f : ℕ → ℕ → ℝ)
  (h1 : (2 : ℕ) ^ n = 256) :
  (f 8 4) * (2 : ℕ) ^ 4 = 1120 :=
by
  sorry

end coefficient_of_x4_l55_55672


namespace clea_escalator_time_l55_55844

theorem clea_escalator_time (x y k : ℕ) (h1 : 90 * x = y) (h2 : 30 * (x + k) = y) :
  (y / k) = 45 := by
  sorry

end clea_escalator_time_l55_55844


namespace volume_tetrahedron_l55_55719

variables (AB AC AD : ℝ) (β γ D : ℝ)
open Real

/-- Prove that the volume of tetrahedron ABCD is equal to 
    (AB * AC * AD * sin β * sin γ * sin D) / 6,
    where β and γ are the plane angles at vertex A opposite to edges AB and AC, 
    and D is the dihedral angle at edge AD. 
-/
theorem volume_tetrahedron (h₁: β ≠ 0) (h₂: γ ≠ 0) (h₃: D ≠ 0):
  (AB * AC * AD * sin β * sin γ * sin D) / 6 =
    abs (AB * AC * AD * sin β * sin γ * sin D) / 6 :=
by sorry

end volume_tetrahedron_l55_55719


namespace unique_integer_solution_m_l55_55065

theorem unique_integer_solution_m {m : ℤ} (h : ∀ x : ℤ, |2 * x - m| ≤ 1 → x = 2) : m = 4 := 
sorry

end unique_integer_solution_m_l55_55065


namespace geometric_sequence_product_l55_55730

theorem geometric_sequence_product {a : ℕ → ℝ} 
(h₁ : a 1 = 2) 
(h₂ : a 5 = 8) 
(h_geom : ∀ n, a (n+1) / a n = a (n+2) / a (n+1)) :
a 2 * a 3 * a 4 = 64 := 
sorry

end geometric_sequence_product_l55_55730


namespace mailman_junk_mail_l55_55897

theorem mailman_junk_mail (total_mail : ℕ) (magazines : ℕ) (junk_mail : ℕ) 
  (h1 : total_mail = 11) (h2 : magazines = 5) (h3 : junk_mail = total_mail - magazines) : junk_mail = 6 := by
  sorry

end mailman_junk_mail_l55_55897


namespace maps_skipped_l55_55460

-- Definitions based on conditions
def total_pages := 372
def pages_read := 125
def pages_left := 231

-- Statement to be proven
theorem maps_skipped : total_pages - (pages_read + pages_left) = 16 :=
by
  sorry

end maps_skipped_l55_55460


namespace unique_solution_m_n_l55_55917

theorem unique_solution_m_n (m n : ℕ) (h1 : m > 1) (h2 : (n - 1) % (m - 1) = 0) 
  (h3 : ¬ ∃ k : ℕ, n = m ^ k) :
  ∃! (a b c : ℕ), a + m * b = n ∧ a + b = m * c := 
sorry

end unique_solution_m_n_l55_55917


namespace parabola_focus_l55_55668

-- Define the given conditions
def parabola_equation (x : ℝ) : ℝ := 4 * x^2 - 8 * x + 5

-- The proof statement that we need to show the focus of the given parabola
theorem parabola_focus :
  (∃ (h k : ℝ), (k = 1) ∧ (h = 1) ∧ (parabola_equation h = k) ∧ ((h, k + 1 / (4 * 4)) = (1, 17 / 16))) := 
sorry

end parabola_focus_l55_55668


namespace cut_half_meter_from_cloth_l55_55178

theorem cut_half_meter_from_cloth (initial_length : ℝ) (cut_length : ℝ) : 
  initial_length = 8 / 15 → cut_length = 1 / 30 → initial_length - cut_length = 1 / 2 := 
by
  intros h_initial h_cut
  sorry

end cut_half_meter_from_cloth_l55_55178


namespace common_root_for_equations_l55_55921

theorem common_root_for_equations : 
  ∃ p x : ℤ, 3 * x^2 - 4 * x + p - 2 = 0 ∧ x^2 - 2 * p * x + 5 = 0 ∧ p = 3 ∧ x = 1 :=
by
  sorry

end common_root_for_equations_l55_55921


namespace problem_solution_l55_55283

def seq (a : ℕ → ℝ) (a1 : a 1 = 0) (rec : ∀ n, a (n + 1) = (a n - Real.sqrt 3) / (1 + Real.sqrt 3 * a n)) : Prop :=
  a 6 = Real.sqrt 3

theorem problem_solution (a : ℕ → ℝ) (h1 : a 1 = 0) (hrec : ∀ n, a (n + 1) = (a n - Real.sqrt 3) / (1 + Real.sqrt 3 * a n)) : 
  seq a h1 hrec :=
by
  sorry

end problem_solution_l55_55283


namespace cube_vertex_numbering_impossible_l55_55188

-- Definition of the cube problem
def vertex_numbering_possible : Prop :=
  ∃ (v : Fin 8 → ℕ), (∀ i, 1 ≤ v i ∧ v i ≤ 8) ∧
    (∀ (e1 e2 : (Fin 8 × Fin 8)), e1 ≠ e2 → (v e1.1 + v e1.2 ≠ v e2.1 + v e2.2))

theorem cube_vertex_numbering_impossible : ¬ vertex_numbering_possible :=
sorry

end cube_vertex_numbering_impossible_l55_55188


namespace factor_expression_l55_55671

theorem factor_expression (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) =
  (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) :=
by
  sorry

end factor_expression_l55_55671


namespace negation_of_proposition_l55_55711
open Real

theorem negation_of_proposition :
  ¬ (∃ x₀ : ℝ, (2/x₀) + log x₀ ≤ 0) ↔ ∀ x : ℝ, (2/x) + log x > 0 :=
by
  sorry

end negation_of_proposition_l55_55711


namespace number_of_women_in_preston_after_one_year_l55_55070

def preston_is_25_times_leesburg (preston leesburg : ℕ) : Prop := 
  preston = 25 * leesburg

def leesburg_population : ℕ := 58940

def women_percentage_leesburg : ℕ := 40

def women_percentage_preston : ℕ := 55

def growth_rate_leesburg : ℝ := 0.025

def growth_rate_preston : ℝ := 0.035

theorem number_of_women_in_preston_after_one_year : 
  ∀ (preston leesburg : ℕ), 
  preston_is_25_times_leesburg preston leesburg → 
  leesburg = 58940 → 
  (women_percentage_preston : ℝ) / 100 * (preston * (1 + growth_rate_preston) : ℝ) = 838788 :=
by 
  sorry

end number_of_women_in_preston_after_one_year_l55_55070


namespace weight_of_four_cakes_l55_55539

variable (C B : ℕ)  -- We declare C and B as natural numbers representing the weights in grams.

def cake_bread_weight_conditions (C B : ℕ) : Prop :=
  (3 * C + 5 * B = 1100) ∧ (C = B + 100)

theorem weight_of_four_cakes (C B : ℕ) 
  (h : cake_bread_weight_conditions C B) : 
  4 * C = 800 := 
by 
  {sorry}

end weight_of_four_cakes_l55_55539


namespace more_wrappers_than_bottle_caps_at_park_l55_55233

-- Define the number of bottle caps and wrappers found at the park.
def bottle_caps_found : ℕ := 11
def wrappers_found : ℕ := 28

-- State the theorem to prove the number of more wrappers than bottle caps found at the park is 17.
theorem more_wrappers_than_bottle_caps_at_park : wrappers_found - bottle_caps_found = 17 :=
by
  -- proof goes here
  sorry

end more_wrappers_than_bottle_caps_at_park_l55_55233


namespace find_m_l55_55374

def A : Set ℤ := {-1, 1}
def B (m : ℤ) : Set ℤ := {x | m * x = 1}

theorem find_m (m : ℤ) (h : B m ⊆ A) : m = 0 ∨ m = 1 ∨ m = -1 := 
sorry

end find_m_l55_55374


namespace coffee_mug_cost_l55_55488

theorem coffee_mug_cost (bracelet_cost gold_heart_necklace_cost total_change total_money_spent : ℤ)
    (bracelets_count gold_heart_necklace_count mugs_count : ℤ)
    (h_bracelet_cost : bracelet_cost = 15)
    (h_gold_heart_necklace_cost : gold_heart_necklace_cost = 10)
    (h_total_change : total_change = 15)
    (h_total_money_spent : total_money_spent = 100)
    (h_bracelets_count : bracelets_count = 3)
    (h_gold_heart_necklace_count : gold_heart_necklace_count = 2)
    (h_mugs_count : mugs_count = 1) :
    mugs_count * ((total_money_spent - total_change) - (bracelets_count * bracelet_cost + gold_heart_necklace_count * gold_heart_necklace_cost)) = 20 :=
by
  sorry

end coffee_mug_cost_l55_55488


namespace ratio_of_roses_l55_55215

-- Definitions for conditions
def roses_two_days_ago : ℕ := 50
def roses_yesterday : ℕ := roses_two_days_ago + 20
def roses_total : ℕ := 220
def roses_today : ℕ := roses_total - roses_two_days_ago - roses_yesterday

-- Lean statement to prove the ratio of roses planted today to two days ago is 2
theorem ratio_of_roses :
  roses_today / roses_two_days_ago = 2 :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_roses_l55_55215


namespace quadrilateral_area_sum_l55_55900

theorem quadrilateral_area_sum (a b : ℤ) (h1 : a > b) (h2 : b > 0) 
  (h3 : a^2 * b = 36) : a + b = 4 := 
sorry

end quadrilateral_area_sum_l55_55900


namespace growth_rate_equation_l55_55030

-- Given conditions
def revenue_january : ℕ := 36
def revenue_march : ℕ := 48

-- Problem statement
theorem growth_rate_equation (x : ℝ) 
  (h_january : revenue_january = 36)
  (h_march : revenue_march = 48) :
  36 * (1 + x) ^ 2 = 48 :=
sorry

end growth_rate_equation_l55_55030


namespace max_dist_2_minus_2i_l55_55990

open Complex

noncomputable def max_dist (z1 : ℂ) : ℝ :=
  Complex.abs 1 + Complex.abs z1

theorem max_dist_2_minus_2i :
  max_dist (2 - 2*I) = 1 + 2 * Real.sqrt 2 := by
  sorry

end max_dist_2_minus_2i_l55_55990


namespace curve_is_line_l55_55403

-- Define the polar equation as a condition
def polar_eq (r θ : ℝ) : Prop := r = 2 / (2 * Real.sin θ - Real.cos θ)

-- Define what it means for a curve to be a line
def is_line (x y : ℝ) : Prop := x + 2 * y = 2

-- The main statement to prove
theorem curve_is_line (r θ : ℝ) (x y : ℝ) (hr : polar_eq r θ) (hx : x = r * Real.cos θ) (hy : y = r * Real.sin θ) :
  is_line x y :=
sorry

end curve_is_line_l55_55403


namespace difference_in_floors_l55_55212

-- Given conditions
variable (FA FB FC : ℕ)
variable (h1 : FA = 4)
variable (h2 : FC = 5 * FB - 6)
variable (h3 : FC = 59)

-- The statement to prove
theorem difference_in_floors : FB - FA = 9 :=
by 
  -- Placeholder proof
  sorry

end difference_in_floors_l55_55212


namespace adjacent_sum_constant_l55_55031

theorem adjacent_sum_constant (x y : ℤ) (k : ℤ) (h1 : 2 + x = k) (h2 : x + y = k) (h3 : y + 5 = k) : x - y = 3 := 
by 
  sorry

end adjacent_sum_constant_l55_55031


namespace series_sum_eq_one_sixth_l55_55292

noncomputable def series_sum := 
  ∑' n : ℕ, (3^n) / ((7^ (2^n)) + 1)

theorem series_sum_eq_one_sixth : series_sum = 1 / 6 := 
  sorry

end series_sum_eq_one_sixth_l55_55292


namespace bananas_used_l55_55026

-- Define the conditions
def bananas_per_loaf := 4
def loaves_monday := 3
def loaves_tuesday := 2 * loaves_monday

-- Define the total bananas used
def bananas_monday := loaves_monday * bananas_per_loaf
def bananas_tuesday := loaves_tuesday * bananas_per_loaf
def total_bananas := bananas_monday + bananas_tuesday

-- Theorem statement to prove the total bananas used is 36
theorem bananas_used : total_bananas = 36 := by
  sorry

end bananas_used_l55_55026


namespace part_a_part_b_l55_55473

-- Part (a)
theorem part_a (x : ℝ) (h : x > 0) : x^3 - 3*x ≥ -2 :=
sorry

-- Part (b)
theorem part_b (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 * y / z) + (y^2 * z / x) + (z^2 * x / y) + 2 * ((y / (x * z)) + (z / (x * y)) + (x / (y * z))) ≥ 9 :=
sorry

end part_a_part_b_l55_55473


namespace grandpa_rank_l55_55995

theorem grandpa_rank (mom dad grandpa : ℕ) 
  (h1 : mom < dad) 
  (h2 : dad < grandpa) : 
  ∀ rank: ℕ, rank = 3 := 
by
  sorry

end grandpa_rank_l55_55995


namespace smallest_d_for_inverse_l55_55659

def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 7

theorem smallest_d_for_inverse (d : ℝ) : 
  (∀ x1 x2 : ℝ, d ≤ x1 → d ≤ x2 → g x1 = g x2 → x1 = x2) → d = 3 :=
by
  sorry

end smallest_d_for_inverse_l55_55659


namespace fourth_term_geom_progression_l55_55219

theorem fourth_term_geom_progression : 
  ∀ (a b c : ℝ), 
    a = 4^(1/2) → 
    b = 4^(1/3) → 
    c = 4^(1/6) → 
    ∃ d : ℝ, d = 1 ∧ b / a = c / b ∧ c / b = 4^(1/6) / 4^(1/3) :=
by
  sorry

end fourth_term_geom_progression_l55_55219


namespace range_of_a_l55_55932

theorem range_of_a
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_neg_x : ∀ x, x ≤ 0 → f x = 2 * x + x^2)
  (h_three_solutions : ∃ x1 x2 x3, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = 2 * a^2 + a ∧ f x2 = 2 * a^2 + a ∧ f x3 = 2 * a^2 + a) :
  -1 < a ∧ a < 1/2 :=
sorry

end range_of_a_l55_55932


namespace least_common_multiple_of_20_45_75_l55_55010

theorem least_common_multiple_of_20_45_75 :
  Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
sorry

end least_common_multiple_of_20_45_75_l55_55010


namespace points_on_fourth_board_l55_55631

theorem points_on_fourth_board (P_1 P_2 P_3 P_4 : ℕ)
 (h1 : P_1 = 30)
 (h2 : P_2 = 38)
 (h3 : P_3 = 41) :
  P_4 = 34 :=
sorry

end points_on_fourth_board_l55_55631


namespace tan_arithmetic_seq_value_l55_55134

variable {a : ℕ → ℝ}
variable (d : ℝ)

-- Define the arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a n = a 0 + n * d

-- Given conditions and the final proof goal
theorem tan_arithmetic_seq_value (h_arith : arithmetic_seq a d)
    (h_sum : a 0 + a 6 + a 12 = Real.pi) :
    Real.tan (a 1 + a 11) = -Real.sqrt 3 := sorry

end tan_arithmetic_seq_value_l55_55134


namespace total_nails_to_cut_l55_55559

theorem total_nails_to_cut :
  let dogs := 4 
  let legs_per_dog := 4
  let nails_per_dog_leg := 4
  let parrots := 8
  let legs_per_parrot := 2
  let nails_per_parrot_leg := 3
  let extra_nail := 1
  let total_dog_nails := dogs * legs_per_dog * nails_per_dog_leg
  let total_parrot_nails := (parrots * legs_per_parrot * nails_per_parrot_leg) + extra_nail
  total_dog_nails + total_parrot_nails = 113 :=
sorry

end total_nails_to_cut_l55_55559


namespace vector_parallel_condition_l55_55896

def OA : ℝ × ℝ := (3, -4)
def OB : ℝ × ℝ := (6, -3)
def OC (m : ℝ) : ℝ × ℝ := (2 * m, m + 1)

def AB (OA OB : ℝ × ℝ) : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)

theorem vector_parallel_condition (m : ℝ) (h_parallel : AB OA OB = (3, 1) ∧ 
    (∀ k : ℝ, 2*m = 3*k ∧ m + 1 = k)) : m = -3 :=
by
  sorry

end vector_parallel_condition_l55_55896


namespace muffin_cost_ratio_l55_55152

theorem muffin_cost_ratio (m b : ℝ) 
  (h1 : 5 * m + 4 * b = 20)
  (h2 : 3 * (5 * m + 4 * b) = 60)
  (h3 : 3 * m + 18 * b = 60) :
  m / b = 13 / 4 :=
by
  sorry

end muffin_cost_ratio_l55_55152


namespace probability_in_given_interval_l55_55462

noncomputable def length_interval (a b : ℝ) : ℝ := b - a

noncomputable def probability_in_interval (a b c d : ℝ) : ℝ :=
  (length_interval a b) / (length_interval c d)

theorem probability_in_given_interval : 
  probability_in_interval (-1) 1 (-2) 3 = 2 / 5 :=
by
  sorry

end probability_in_given_interval_l55_55462


namespace opposite_neg_fraction_l55_55474

theorem opposite_neg_fraction : -(- (1/2023)) = 1/2023 := 
by 
  sorry

end opposite_neg_fraction_l55_55474


namespace quadratic_has_distinct_real_roots_l55_55537

theorem quadratic_has_distinct_real_roots :
  let a := 2
  let b := 3
  let c := -4
  (b^2 - 4 * a * c) > 0 := by
  sorry

end quadratic_has_distinct_real_roots_l55_55537


namespace remainder_div_14_l55_55748

def S : ℕ := 11065 + 11067 + 11069 + 11071 + 11073 + 11075 + 11077

theorem remainder_div_14 : S % 14 = 7 :=
by
  sorry

end remainder_div_14_l55_55748


namespace bus_driver_total_compensation_l55_55298

-- Definitions of conditions
def regular_rate : ℝ := 16
def regular_hours : ℝ := 40
def overtime_rate : ℝ := regular_rate * 1.75
def total_hours : ℝ := 65
def total_compensation : ℝ := (regular_rate * regular_hours) + (overtime_rate * (total_hours - regular_hours))

-- Theorem stating the total compensation
theorem bus_driver_total_compensation : total_compensation = 1340 :=
by
  sorry

end bus_driver_total_compensation_l55_55298


namespace product_and_divisibility_l55_55389

theorem product_and_divisibility (n : ℕ) (h : n = 3) :
  (n-1) * n * (n+1) * (n+2) * (n+3) = 720 ∧ ¬ (720 % 11 = 0) :=
by
  sorry

end product_and_divisibility_l55_55389


namespace quadratic_equal_roots_iff_l55_55095

theorem quadratic_equal_roots_iff (k : ℝ) :
  (∃ x : ℝ, x^2 - k * x + 9 = 0 ∧ x^2 - k * x + 9 = 0 ∧ x = x) ↔ k^2 = 36 :=
by
  sorry

end quadratic_equal_roots_iff_l55_55095


namespace sum_first_100_odd_l55_55644

theorem sum_first_100_odd :
  (Finset.sum (Finset.range 100) (λ x => 2 * (x + 1) - 1)) = 10000 := by
  sorry

end sum_first_100_odd_l55_55644


namespace merchant_gross_profit_l55_55229

noncomputable def grossProfit (purchase_price : ℝ) (selling_price : ℝ) (discount : ℝ) : ℝ :=
  (selling_price - discount * selling_price) - purchase_price

theorem merchant_gross_profit :
  let P := 56
  let S := (P / 0.70 : ℝ)
  let discount := 0.20
  grossProfit P S discount = 8 := 
by
  let P := 56
  let S := (P / 0.70 : ℝ)
  let discount := 0.20
  unfold grossProfit
  sorry

end merchant_gross_profit_l55_55229


namespace max_superior_squares_l55_55872

theorem max_superior_squares (n : ℕ) (h : n > 2004) :
  ∃ superior_squares_count : ℕ, superior_squares_count = n * (n - 2004) := 
sorry

end max_superior_squares_l55_55872


namespace arithmetic_expression_value_l55_55793

theorem arithmetic_expression_value :
  (19 + 43 / 151) * 151 = 2910 :=
by {
  sorry
}

end arithmetic_expression_value_l55_55793


namespace evaluate_expression_l55_55355

variable (a b c : ℝ)

theorem evaluate_expression 
  (h : a / (20 - a) + b / (75 - b) + c / (55 - c) = 8) :
  4 / (20 - a) + 15 / (75 - b) + 11 / (55 - c) = 8.8 :=
sorry

end evaluate_expression_l55_55355


namespace fixed_monthly_fee_l55_55686

theorem fixed_monthly_fee (f h : ℝ) 
  (feb_bill : f + h = 18.72)
  (mar_bill : f + 3 * h = 33.78) :
  f = 11.19 :=
by
  sorry

end fixed_monthly_fee_l55_55686
