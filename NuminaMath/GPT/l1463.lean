import Mathlib

namespace NUMINAMATH_GPT_pairs_satisfying_equation_l1463_146395

theorem pairs_satisfying_equation :
  ∀ x y : ℝ, (x ^ 4 + 1) * (y ^ 4 + 1) = 4 * x^2 * y^2 ↔ (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
by
  intros x y
  sorry

end NUMINAMATH_GPT_pairs_satisfying_equation_l1463_146395


namespace NUMINAMATH_GPT_find_X_l1463_146331

variable (E X : ℕ)

-- Theorem statement
theorem find_X (hE : E = 9)
              (hSum : E * 100 + E * 10 + E + E * 100 + E * 10 + E = 1798) :
              X = 7 :=
sorry

end NUMINAMATH_GPT_find_X_l1463_146331


namespace NUMINAMATH_GPT_chess_tournament_participants_l1463_146375

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 136) : n = 17 :=
by {
  sorry -- Proof will be here.
}

end NUMINAMATH_GPT_chess_tournament_participants_l1463_146375


namespace NUMINAMATH_GPT_equalities_implied_by_sum_of_squares_l1463_146300

variable {a b c d : ℝ}

theorem equalities_implied_by_sum_of_squares (h1 : a = b) (h2 : c = d) : 
  (a - b) ^ 2 + (c - d) ^ 2 = 0 :=
sorry

end NUMINAMATH_GPT_equalities_implied_by_sum_of_squares_l1463_146300


namespace NUMINAMATH_GPT_value_of_x_squared_plus_y_squared_l1463_146322

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : x^2 = 8 * x + y) (h2 : y^2 = x + 8 * y) (h3 : x ≠ y) : 
  x^2 + y^2 = 63 := sorry

end NUMINAMATH_GPT_value_of_x_squared_plus_y_squared_l1463_146322


namespace NUMINAMATH_GPT_doctor_visit_cost_l1463_146330

theorem doctor_visit_cost (cast_cost : ℝ) (insurance_coverage : ℝ) (out_of_pocket : ℝ) (visit_cost : ℝ) :
  cast_cost = 200 → insurance_coverage = 0.60 → out_of_pocket = 200 → 0.40 * (visit_cost + cast_cost) = out_of_pocket → visit_cost = 300 :=
by
  intros h_cast h_insurance h_out_of_pocket h_equation
  sorry

end NUMINAMATH_GPT_doctor_visit_cost_l1463_146330


namespace NUMINAMATH_GPT_investment_period_two_years_l1463_146328

theorem investment_period_two_years
  (P : ℝ) (r : ℝ) (A : ℝ) (n : ℕ) (hP : P = 6000) (hr : r = 0.10) (hA : A = 7260) (hn : n = 1) : 
  ∃ t : ℝ, t = 2 ∧ A = P * (1 + r / n) ^ (n * t) :=
by
  sorry

end NUMINAMATH_GPT_investment_period_two_years_l1463_146328


namespace NUMINAMATH_GPT_problem_statement_l1463_146397

-- Definitions of the sets P and Q
def P : Set ℝ := {x : ℝ | x > 1}
def Q : Set ℝ := {x : ℝ | abs x > 0}

-- Statement of the problem to prove that P is not a subset of Q
theorem problem_statement : ¬ (P ⊆ Q) :=
sorry

end NUMINAMATH_GPT_problem_statement_l1463_146397


namespace NUMINAMATH_GPT_time_to_cross_bridge_l1463_146358

def train_length : ℕ := 600  -- train length in meters
def bridge_length : ℕ := 100  -- overbridge length in meters
def speed_km_per_hr : ℕ := 36  -- speed of the train in kilometers per hour

-- Convert speed from km/h to m/s
def speed_m_per_s : ℕ := speed_km_per_hr * 1000 / 3600

-- Compute the total distance
def total_distance : ℕ := train_length + bridge_length

-- Prove the time to cross the overbridge
theorem time_to_cross_bridge : total_distance / speed_m_per_s = 70 := by
  sorry

end NUMINAMATH_GPT_time_to_cross_bridge_l1463_146358


namespace NUMINAMATH_GPT_four_digit_number_conditions_l1463_146384

theorem four_digit_number_conditions :
  ∃ (a b c d : ℕ), 
    (a < 10) ∧ (b < 10) ∧ (c < 10) ∧ (d < 10) ∧ 
    (a * 1000 + b * 100 + c * 10 + d = 10 * 23) ∧ 
    (a + b + c + d = 26) ∧ 
    ((b * d / 10) % 10 = a + c) ∧ 
    ∃ (n : ℕ), (b * d - c^2 = 2^n) ∧ 
    (a * 1000 + b * 100 + c * 10 + d = 1979) :=
sorry

end NUMINAMATH_GPT_four_digit_number_conditions_l1463_146384


namespace NUMINAMATH_GPT_part1_part2_l1463_146363

noncomputable def f (x : ℝ) (a : ℝ) := (Real.exp x / x) - Real.log x + x - a

theorem part1 (x : ℝ) (a : ℝ) :
    (∀ x > 0, f x a ≥ 0) → a ≤ Real.exp 1 + 1 :=
sorry

theorem part2 (x1 x2 : ℝ) (a : ℝ) :
  f x1 a = 0 → f x2 a = 0 → x1 < 1 → 1 < x2 → x1 * x2 < 1 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1463_146363


namespace NUMINAMATH_GPT_first_term_value_l1463_146329

noncomputable def find_first_term (a r : ℝ) := a / (1 - r) = 27 ∧ a^2 / (1 - r^2) = 108

theorem first_term_value :
  ∃ (a r : ℝ), find_first_term a r ∧ a = 216 / 31 :=
by
  sorry

end NUMINAMATH_GPT_first_term_value_l1463_146329


namespace NUMINAMATH_GPT_FerrisWheelCostIsTwo_l1463_146335

noncomputable def costFerrisWheel (rollerCoasterCost multipleRideDiscount coupon totalTicketsBought : ℝ) : ℝ :=
  totalTicketsBought + multipleRideDiscount + coupon - rollerCoasterCost

theorem FerrisWheelCostIsTwo :
  let rollerCoasterCost := 7.0
  let multipleRideDiscount := 1.0
  let coupon := 1.0
  let totalTicketsBought := 7.0
  costFerrisWheel rollerCoasterCost multipleRideDiscount coupon totalTicketsBought = 2.0 :=
by
  sorry

end NUMINAMATH_GPT_FerrisWheelCostIsTwo_l1463_146335


namespace NUMINAMATH_GPT_samantha_trip_l1463_146390

theorem samantha_trip (a b c d x : ℕ)
  (h1 : 1 ≤ a) (h2 : a + b + c + d ≤ 10) 
  (h3 : 1000 * d + 100 * c + 10 * b + a - (1000 * a + 100 * b + 10 * c + d) = 60 * x)
  : a^2 + b^2 + c^2 + d^2 = 83 :=
sorry

end NUMINAMATH_GPT_samantha_trip_l1463_146390


namespace NUMINAMATH_GPT_find_common_ratio_l1463_146326

noncomputable def a_n (n : ℕ) (q : ℚ) : ℚ :=
  if n = 1 then 1 / 8 else (q^(n - 1)) * (1 / 8)

theorem find_common_ratio (q : ℚ) :
  (a_n 4 q = -1) ↔ (q = -2) :=
by
  sorry

end NUMINAMATH_GPT_find_common_ratio_l1463_146326


namespace NUMINAMATH_GPT_quadratic_min_value_l1463_146313

theorem quadratic_min_value (p q r : ℝ) (h : ∀ x : ℝ, x^2 + p * x + q + r ≥ -r) : q = p^2 / 4 :=
sorry

end NUMINAMATH_GPT_quadratic_min_value_l1463_146313


namespace NUMINAMATH_GPT_internet_bill_proof_l1463_146314

variable (current_bill : ℕ)
variable (internet_bill_30Mbps : ℕ)
variable (annual_savings : ℕ)
variable (additional_amount_20Mbps : ℕ)

theorem internet_bill_proof
  (h1 : current_bill = 20)
  (h2 : internet_bill_30Mbps = 40)
  (h3 : annual_savings = 120)
  (monthly_savings : ℕ := annual_savings / 12)
  (h4 : monthly_savings = 10)
  (h5 : internet_bill_30Mbps - (current_bill + additional_amount_20Mbps) = 10) :
  additional_amount_20Mbps = 10 :=
by
  sorry

end NUMINAMATH_GPT_internet_bill_proof_l1463_146314


namespace NUMINAMATH_GPT_directrix_of_parabola_l1463_146303

-- Define the parabola x^2 = 16y
def parabola (x y : ℝ) : Prop := x^2 = 16 * y

-- Define the directrix equation
def directrix (y : ℝ) : Prop := y = -4

-- Theorem stating that the directrix of the given parabola is y = -4
theorem directrix_of_parabola : ∀ x y: ℝ, parabola x y → ∃ y, directrix y :=
by
  sorry

end NUMINAMATH_GPT_directrix_of_parabola_l1463_146303


namespace NUMINAMATH_GPT_set_A_correct_l1463_146309

-- Definition of the sets and conditions
def A : Set ℤ := {-3, 0, 2, 6}
def B : Set ℤ := {-1, 3, 5, 8}

theorem set_A_correct : 
  (∃ a1 a2 a3 a4 : ℤ, A = {a1, a2, a3, a4} ∧ 
  {a1 + a2 + a3, a1 + a2 + a4, a1 + a3 + a4, a2 + a3 + a4} = B) → 
  A = {-3, 0, 2, 6} :=
by 
  sorry

end NUMINAMATH_GPT_set_A_correct_l1463_146309


namespace NUMINAMATH_GPT_f_1_eq_zero_l1463_146398

-- Given a function f with the specified properties
variable {f : ℝ → ℝ}

-- Given 1) the domain of the function
axiom domain_f : ∀ x, (x < 0 ∨ x > 0) → true 

-- Given 2) the functional equation
axiom functional_eq_f : ∀ x₁ x₂, (x₁ < 0 ∨ x₁ > 0) ∧ (x₂ < 0 ∨ x₂ > 0) → f (x₁ * x₂) = f x₁ + f x₂

-- Prove that f(1) = 0
theorem f_1_eq_zero : f 1 = 0 := 
  sorry

end NUMINAMATH_GPT_f_1_eq_zero_l1463_146398


namespace NUMINAMATH_GPT_probability_sum_9_is_correct_l1463_146304

def num_faces : ℕ := 6

def possible_outcomes : ℕ := num_faces * num_faces

def favorable_outcomes : ℕ := 4  -- (3,6), (6,3), (4,5), (5,4)

def probability_sum_9 : ℚ := favorable_outcomes / possible_outcomes

theorem probability_sum_9_is_correct :
  probability_sum_9 = 1/9 :=
sorry

end NUMINAMATH_GPT_probability_sum_9_is_correct_l1463_146304


namespace NUMINAMATH_GPT_ellipse_slope_product_l1463_146350

variables {a b x1 y1 x2 y2 : ℝ} (h₁ : a > b) (h₂ : b > 0) (h₃ : (b^2 * x1^2 + a^2 * y1^2 = a^2 * b^2) ∧ (b^2 * x2^2 + a^2 * y2^2 = a^2 * b^2))

theorem ellipse_slope_product : 
  (a > b) → (b > 0) → (b^2 * x1^2 + a^2 * y1^2 = a^2 * b^2) → 
  (b^2 * x2^2 + a^2 * y2^2 = a^2 * b^2) → 
  ( (y1 + y2)/(x1 + x2) ) * ( (y1 - y2)/(x1 - x2) ) = - (b^2 / a^2) :=
by
  intros ha hb hxy1 hxy2
  sorry

end NUMINAMATH_GPT_ellipse_slope_product_l1463_146350


namespace NUMINAMATH_GPT_smallest_whole_number_greater_than_triangle_perimeter_l1463_146361

theorem smallest_whole_number_greater_than_triangle_perimeter 
  (a b : ℝ) (h_a : a = 7) (h_b : b = 23) :
  ∀ c : ℝ, 16 < c ∧ c < 30 → ⌈a + b + c⌉ = 60 :=
by
  intros c h
  rw [h_a, h_b]
  sorry

end NUMINAMATH_GPT_smallest_whole_number_greater_than_triangle_perimeter_l1463_146361


namespace NUMINAMATH_GPT_batsman_average_30_matches_l1463_146378

theorem batsman_average_30_matches (avg_20_matches : ℕ -> ℚ) (avg_10_matches : ℕ -> ℚ)
  (h1 : avg_20_matches 20 = 40)
  (h2 : avg_10_matches 10 = 20)
  : (20 * (avg_20_matches 20) + 10 * (avg_10_matches 10)) / 30 = 33.33 := by
  sorry

end NUMINAMATH_GPT_batsman_average_30_matches_l1463_146378


namespace NUMINAMATH_GPT_value_of_m_l1463_146343

theorem value_of_m (a b c : ℤ) (m : ℤ) (h1 : 0 ≤ m) (h2 : m ≤ 26) 
  (h3 : (a + b + c) % 27 = m) (h4 : ((a - b) * (b - c) * (c - a)) % 27 = m) : 
  m = 0 :=
  by
  -- Proof is to be filled in
  sorry

end NUMINAMATH_GPT_value_of_m_l1463_146343


namespace NUMINAMATH_GPT_n_pow_8_minus_1_divisible_by_480_l1463_146324

theorem n_pow_8_minus_1_divisible_by_480 (n : ℤ) (h1 : ¬ (2 ∣ n)) (h2 : ¬ (3 ∣ n)) (h3 : ¬ (5 ∣ n)) : 
  480 ∣ (n^8 - 1) := 
sorry

end NUMINAMATH_GPT_n_pow_8_minus_1_divisible_by_480_l1463_146324


namespace NUMINAMATH_GPT_roundness_of_8000000_l1463_146320

def is_prime (n : Nat) : Prop := sorry

def prime_factors_exponents (n : Nat) : List (Nat × Nat) := sorry

def roundness (n : Nat) : Nat := 
  (prime_factors_exponents n).foldr (λ p acc => p.2 + acc) 0

theorem roundness_of_8000000 : roundness 8000000 = 15 :=
sorry

end NUMINAMATH_GPT_roundness_of_8000000_l1463_146320


namespace NUMINAMATH_GPT_trapezoid_height_l1463_146336

theorem trapezoid_height (BC AD AB CD h : ℝ) (hBC : BC = 4) (hAD : AD = 25) (hAB : AB = 20) (hCD : CD = 13) :
  h = 12 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_height_l1463_146336


namespace NUMINAMATH_GPT_train_passes_tree_in_20_seconds_l1463_146355

def train_passing_time 
  (length_of_train : ℕ)
  (speed_kmh : ℕ)
  (conversion_factor : ℚ) : ℚ :=
  length_of_train / (speed_kmh * conversion_factor)

theorem train_passes_tree_in_20_seconds 
  (length_of_train : ℕ := 350)
  (speed_kmh : ℕ := 63)
  (conversion_factor : ℚ := 1000 / 3600) : 
  train_passing_time length_of_train speed_kmh conversion_factor = 20 :=
  sorry

end NUMINAMATH_GPT_train_passes_tree_in_20_seconds_l1463_146355


namespace NUMINAMATH_GPT_pow_mod_eq_l1463_146334

theorem pow_mod_eq (n : ℕ) : 
  (3^n % 5 = 3 % 5) → 
  (3^(n+1) % 5 = (3 * 3^n) % 5) → 
  (3^(n+2) % 5 = (3 * 3^(n+1)) % 5) → 
  (3^(n+3) % 5 = (3 * 3^(n+2)) % 5) → 
  (3^4 % 5 = 1 % 5) → 
  (2023 % 4 = 3) → 
  (3^2023 % 5 = 2 % 5) :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_pow_mod_eq_l1463_146334


namespace NUMINAMATH_GPT_number_of_students_l1463_146367

theorem number_of_students (total_stars : ℕ) (stars_per_student : ℕ) (h1 : total_stars = 372) (h2 : stars_per_student = 3) : total_stars / stars_per_student = 124 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_l1463_146367


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1463_146315

-- Define the sets A and B based on the given conditions
def A := {x : ℝ | x > 1}
def B := {x : ℝ | x ≤ 3}

-- Lean statement to prove the intersection of A and B matches the correct answer
theorem intersection_of_A_and_B : (A ∩ B) = {x : ℝ | 1 < x ∧ x ≤ 3} :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_of_A_and_B_l1463_146315


namespace NUMINAMATH_GPT_find_x_l1463_146389

theorem find_x (x : ℕ) (h1 : 8 = 2 ^ 3) (h2 : 32 = 2 ^ 5) :
  (2^(x+2) * 8^(x-1) = 32^3) ↔ (x = 4) :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1463_146389


namespace NUMINAMATH_GPT_ticket_price_values_l1463_146341

theorem ticket_price_values : 
  ∃ (x_values : Finset ℕ), 
    (∀ x ∈ x_values, x ∣ 60 ∧ x ∣ 80) ∧ 
    x_values.card = 6 :=
by
  sorry

end NUMINAMATH_GPT_ticket_price_values_l1463_146341


namespace NUMINAMATH_GPT_probability_red_and_at_least_one_even_l1463_146374

-- Definitions based on conditions
def total_balls : ℕ := 12
def red_balls : Finset ℕ := {1, 2, 3, 4, 5, 6}
def black_balls : Finset ℕ := {7, 8, 9, 10, 11, 12}

-- Condition to check if a ball is red
def is_red (n : ℕ) : Prop := n ∈ red_balls

-- Condition to check if a ball has an even number
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Total number of ways to draw two balls with replacement
def total_ways : ℕ := total_balls * total_balls

-- Number of ways to draw both red balls
def red_red_ways : ℕ := Finset.card red_balls * Finset.card red_balls

-- Number of ways to draw both red balls with none even
def red_odd_numbers : Finset ℕ := {1, 3, 5}
def red_red_odd_ways : ℕ := Finset.card red_odd_numbers * Finset.card red_odd_numbers

-- Number of ways to draw both red balls with at least one even
def desired_outcomes : ℕ := red_red_ways - red_red_odd_ways

-- The probability
def probability : ℚ := desired_outcomes / total_ways

theorem probability_red_and_at_least_one_even :
  probability = 3 / 16 :=
by
  sorry

end NUMINAMATH_GPT_probability_red_and_at_least_one_even_l1463_146374


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1463_146386

def quadratic_inequality (x : ℝ) : Prop :=
  x^2 - 3 * x + 2 < 0

def necessary_condition_A (x : ℝ) : Prop :=
  -1 < x ∧ x < 2

def necessary_condition_D (x : ℝ) : Prop :=
  -2 < x ∧ x < 2

theorem necessary_but_not_sufficient :
  (∀ x, quadratic_inequality x → ∃ x, necessary_condition_A x ∧ ¬(quadratic_inequality x ∧ necessary_condition_A x)) ∧ 
  (∀ x, quadratic_inequality x → ∃ x, necessary_condition_D x ∧ ¬(quadratic_inequality x ∧ necessary_condition_D x)) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1463_146386


namespace NUMINAMATH_GPT_find_x_l1463_146352

variable (x : ℝ)

theorem find_x (h : (15 - 2 + 4 / 1 / 2) * x = 77) : x = 77 / (15 - 2 + 4 / 1 / 2) :=
by sorry

end NUMINAMATH_GPT_find_x_l1463_146352


namespace NUMINAMATH_GPT_root_power_sum_eq_l1463_146323

open Real

theorem root_power_sum_eq :
  ∀ {a b c : ℝ},
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (a^3 - 3 * a + 1 = 0) → (b^3 - 3 * b + 1 = 0) → (c^3 - 3 * c + 1 = 0) →
  a^8 + b^8 + c^8 = 186 :=
by
  intros a b c h1 h2 h3 ha hb hc
  sorry

end NUMINAMATH_GPT_root_power_sum_eq_l1463_146323


namespace NUMINAMATH_GPT_calculate_T6_l1463_146327

noncomputable def T (y : ℝ) (m : ℕ) : ℝ := y^m + 1 / y^m

theorem calculate_T6 (y : ℝ) (h : y + 1 / y = 5) : T y 6 = 12098 := 
by
  sorry

end NUMINAMATH_GPT_calculate_T6_l1463_146327


namespace NUMINAMATH_GPT_janice_time_left_l1463_146356

def time_before_movie : ℕ := 2 * 60
def homework_time : ℕ := 30
def cleaning_time : ℕ := homework_time / 2
def walking_dog_time : ℕ := homework_time + 5
def taking_trash_time : ℕ := homework_time * 1 / 6

theorem janice_time_left : time_before_movie - (homework_time + cleaning_time + walking_dog_time + taking_trash_time) = 35 :=
by
  sorry

end NUMINAMATH_GPT_janice_time_left_l1463_146356


namespace NUMINAMATH_GPT_land_value_moon_l1463_146371

-- Define the conditions
def surface_area_earth : ℕ := 200
def surface_area_ratio : ℕ := 5
def value_ratio : ℕ := 6
def total_value_earth : ℕ := 80

-- Define the question and the expected answer
noncomputable def total_value_moon : ℕ := 96

-- State the proof problem
theorem land_value_moon :
  (surface_area_earth / surface_area_ratio * value_ratio) * (surface_area_earth / surface_area_ratio) = total_value_moon := 
sorry

end NUMINAMATH_GPT_land_value_moon_l1463_146371


namespace NUMINAMATH_GPT_no_coprime_xy_multiple_l1463_146307

theorem no_coprime_xy_multiple (n : ℕ) (hn : ∀ d : ℕ, d ∣ n → d^2 ∣ n → d = 1)
  (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) (h_coprime : Nat.gcd x y = 1) :
  ¬ ((x^n + y^n) % ((x + y)^3) = 0) :=
by
  sorry

end NUMINAMATH_GPT_no_coprime_xy_multiple_l1463_146307


namespace NUMINAMATH_GPT_slow_car_speed_l1463_146373

theorem slow_car_speed (x : ℝ) (hx : 0 < x) (distance : ℝ) (delay : ℝ) (fast_factor : ℝ) :
  distance = 60 ∧ delay = 0.5 ∧ fast_factor = 1.5 ∧ 
  (distance / x) - (distance / (fast_factor * x)) = delay → 
  x = 40 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_slow_car_speed_l1463_146373


namespace NUMINAMATH_GPT_coin_flip_prob_nickel_halfdollar_heads_l1463_146366

def coin_prob : ℚ :=
  let total_outcomes := 2^5
  let successful_outcomes := 2^3
  successful_outcomes / total_outcomes

theorem coin_flip_prob_nickel_halfdollar_heads :
  coin_prob = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_coin_flip_prob_nickel_halfdollar_heads_l1463_146366


namespace NUMINAMATH_GPT_determine_color_sum_or_product_l1463_146317

theorem determine_color_sum_or_product {x : ℕ → ℝ} (h_distinct: ∀ i j : ℕ, i < j → x i < x j) (x_pos : ∀ i : ℕ, x i > 0) :
  ∃ c : ℕ → ℝ, (∀ i : ℕ, c i > 0) ∧
  (∀ i j : ℕ, i < j → (∃ r1 r2 : ℕ, (r1 ≠ r2) ∧ (c r1 + c r2 = x₆₄ + x₆₃) ∧ (c r1 * c r2 = x₆₄ * x₆₃))) :=
sorry

end NUMINAMATH_GPT_determine_color_sum_or_product_l1463_146317


namespace NUMINAMATH_GPT_total_cost_correct_l1463_146357

def cost_barette : ℕ := 3
def cost_comb : ℕ := 1

def kristine_barrettes : ℕ := 1
def kristine_combs : ℕ := 1

def crystal_barrettes : ℕ := 3
def crystal_combs : ℕ := 1

def total_spent (cost_barette : ℕ) (cost_comb : ℕ) 
  (kristine_barrettes : ℕ) (kristine_combs : ℕ) 
  (crystal_barrettes : ℕ) (crystal_combs : ℕ) : ℕ :=
  (kristine_barrettes * cost_barette + kristine_combs * cost_comb) + 
  (crystal_barrettes * cost_barette + crystal_combs * cost_comb)

theorem total_cost_correct :
  total_spent cost_barette cost_comb kristine_barrettes kristine_combs crystal_barrettes crystal_combs = 14 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l1463_146357


namespace NUMINAMATH_GPT_remainder_proof_l1463_146393

theorem remainder_proof (n : ℤ) (h : n % 6 = 1) : (3 * (n + 1812)) % 6 = 3 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_proof_l1463_146393


namespace NUMINAMATH_GPT_egyptian_fraction_decomposition_l1463_146349

theorem egyptian_fraction_decomposition (n : ℕ) (hn : 0 < n) : 
  (2 : ℚ) / (2 * n + 1) = (1 : ℚ) / (n + 1) + (1 : ℚ) / ((n + 1) * (2 * n + 1)) := 
by {
  sorry
}

end NUMINAMATH_GPT_egyptian_fraction_decomposition_l1463_146349


namespace NUMINAMATH_GPT_sum_of_bases_is_20_l1463_146399

theorem sum_of_bases_is_20
  (B1 B2 : ℕ)
  (G1 : ℚ)
  (G2 : ℚ)
  (hG1_B1 : G1 = (4 * B1 + 5) / (B1^2 - 1))
  (hG2_B1 : G2 = (5 * B1 + 4) / (B1^2 - 1))
  (hG1_B2 : G1 = (3 * B2) / (B2^2 - 1))
  (hG2_B2 : G2 = (6 * B2) / (B2^2 - 1)) :
  B1 + B2 = 20 :=
sorry

end NUMINAMATH_GPT_sum_of_bases_is_20_l1463_146399


namespace NUMINAMATH_GPT_tank_ratio_l1463_146325

variable (C D : ℝ)
axiom h1 : 3 / 4 * C = 2 / 5 * D

theorem tank_ratio : C / D = 8 / 15 := by
  sorry

end NUMINAMATH_GPT_tank_ratio_l1463_146325


namespace NUMINAMATH_GPT_scientific_notation_of_218000000_l1463_146332

theorem scientific_notation_of_218000000 :
  218000000 = 2.18 * 10^8 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_218000000_l1463_146332


namespace NUMINAMATH_GPT_red_to_blue_l1463_146316

def is_red (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m ^ 2020

def is_blue (n : ℕ) : Prop :=
  ¬ is_red n ∧ ∃ m : ℕ, n = m ^ 2019

theorem red_to_blue (n : ℕ) (hn : n > 10^100000000) (hnred : is_red n) 
    (hn1red : is_red (n+1)) :
    ∃ (k : ℕ), 1 ≤ k ∧ k ≤ 2019 ∧ is_blue (n + k) :=
sorry

end NUMINAMATH_GPT_red_to_blue_l1463_146316


namespace NUMINAMATH_GPT_flowchart_output_correct_l1463_146301

-- Define the conditions of the problem
def program_flowchart (initial : ℕ) : ℕ :=
  let step1 := initial * 2
  let step2 := step1 * 2
  let step3 := step2 * 2
  step3

-- State the proof problem
theorem flowchart_output_correct : program_flowchart 1 = 8 :=
by
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_flowchart_output_correct_l1463_146301


namespace NUMINAMATH_GPT_center_of_circle_l1463_146308

theorem center_of_circle (x y : ℝ) : (x^2 + y^2 - 10 * x + 4 * y + 13 = 0) → (x - y = 7) :=
by
  -- Statement, proof omitted
  sorry

end NUMINAMATH_GPT_center_of_circle_l1463_146308


namespace NUMINAMATH_GPT_luther_latest_line_count_l1463_146344

theorem luther_latest_line_count :
  let silk := 10
  let cashmere := silk / 2
  let blended := 2
  silk + cashmere + blended = 17 :=
by
  sorry

end NUMINAMATH_GPT_luther_latest_line_count_l1463_146344


namespace NUMINAMATH_GPT_quadratic_range_extrema_l1463_146362

def quadratic (x : ℝ) : ℝ := x^2 - 4 * x + 2

theorem quadratic_range_extrema :
  let y := quadratic
  ∃ x_max x_min,
    (x_min = -2 ∧ y x_min = -2) ∧
    (x_max = -2 ∧ y x_max = 14 ∨ x_max = 5 ∧ y x_max = 7) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_range_extrema_l1463_146362


namespace NUMINAMATH_GPT_work_completion_by_b_l1463_146347

theorem work_completion_by_b (a_days : ℕ) (a_solo_days : ℕ) (a_b_combined_days : ℕ) (b_days : ℕ) :
  a_days = 12 ∧ a_solo_days = 3 ∧ a_b_combined_days = 5 → b_days = 15 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_by_b_l1463_146347


namespace NUMINAMATH_GPT_exponentiation_identity_l1463_146310

theorem exponentiation_identity (x : ℝ) : (-x^7)^4 = x^28 := 
sorry

end NUMINAMATH_GPT_exponentiation_identity_l1463_146310


namespace NUMINAMATH_GPT_problem_statement_l1463_146385

theorem problem_statement :
  ((8^5 / 8^2) * 2^10 - 2^2) = 2^19 - 4 := 
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l1463_146385


namespace NUMINAMATH_GPT_min_value_a1_plus_a7_l1463_146342

theorem min_value_a1_plus_a7 (a : ℕ → ℝ) (r : ℝ) 
  (h1 : ∀ n, a n > 0) 
  (h2 : ∀ n, a (n+1) = a n * r) 
  (h3 : a 3 * a 5 = 64) : 
  a 1 + a 7 ≥ 16 := 
sorry

end NUMINAMATH_GPT_min_value_a1_plus_a7_l1463_146342


namespace NUMINAMATH_GPT_geometric_sequence_theorem_l1463_146377

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n+1) = a n * r

def holds_condition (a : ℕ → ℝ) : Prop := 
  a 1 * a 10 = -2

theorem geometric_sequence_theorem (a : ℕ → ℝ) (h_geo : geometric_sequence a) (h_cond : holds_condition a) : a 4 * a 7 = -2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_theorem_l1463_146377


namespace NUMINAMATH_GPT_initial_paintings_l1463_146372

theorem initial_paintings (paintings_per_day : ℕ) (days : ℕ) (total_paintings : ℕ) (initial_paintings : ℕ) 
  (h1 : paintings_per_day = 2) 
  (h2 : days = 30) 
  (h3 : total_paintings = 80) 
  (h4 : total_paintings = initial_paintings + paintings_per_day * days) : 
  initial_paintings = 20 := by
  sorry

end NUMINAMATH_GPT_initial_paintings_l1463_146372


namespace NUMINAMATH_GPT_two_digit_numbers_tens_greater_ones_l1463_146306

theorem two_digit_numbers_tens_greater_ones : 
  ∃ (count : ℕ), count = 45 ∧ ∀ (n : ℕ), 10 ≤ n ∧ n < 100 → 
    let tens := n / 10;
    let ones := n % 10;
    tens > ones → count = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_two_digit_numbers_tens_greater_ones_l1463_146306


namespace NUMINAMATH_GPT_restore_salary_l1463_146392

variable (W : ℝ) -- Define the initial wage as a real number
variable (newWage : ℝ := 0.7 * W) -- New wage after a 30% reduction

-- Define the hypothesis for the initial wage reduction
theorem restore_salary : (100 * (W / (0.7 * W) - 1)) = 42.86 :=
by
  sorry

end NUMINAMATH_GPT_restore_salary_l1463_146392


namespace NUMINAMATH_GPT_seat_number_X_l1463_146359

theorem seat_number_X (X : ℕ) (h1 : 42 - 30 = X - 6) : X = 18 :=
by
  sorry

end NUMINAMATH_GPT_seat_number_X_l1463_146359


namespace NUMINAMATH_GPT_min_value_of_expression_l1463_146348

theorem min_value_of_expression (a b c : ℝ) (hb : b > a) (ha : a > c) (hc : b ≠ 0) :
  ∃ l : ℝ, l = 5.5 ∧ l ≤ (a + b)^2 / b^2 + (b + c)^2 / b^2 + (c + a)^2 / b^2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l1463_146348


namespace NUMINAMATH_GPT_pentagon_angles_sum_l1463_146346

theorem pentagon_angles_sum {α β γ δ ε : ℝ} (h1 : α + β + γ + δ + ε = 180) (h2 : α = 50) :
  β + ε = 230 := 
sorry

end NUMINAMATH_GPT_pentagon_angles_sum_l1463_146346


namespace NUMINAMATH_GPT_minimum_quadratic_value_l1463_146388

theorem minimum_quadratic_value (h : ℝ) (x : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → (x - h)^2 + 1 ≥ 10) ∧ (∃ x, 1 ≤ x ∧ x ≤ 3 ∧ (x - h)^2 + 1 = 10) 
  ↔ h = -2 ∨ h = 6 :=
by
  sorry

end NUMINAMATH_GPT_minimum_quadratic_value_l1463_146388


namespace NUMINAMATH_GPT_stan_average_speed_l1463_146387

/-- Given two trips with specified distances and times, prove that the overall average speed is 55 mph. -/
theorem stan_average_speed :
  let distance1 := 300
  let hours1 := 5
  let minutes1 := 20
  let distance2 := 360
  let hours2 := 6
  let minutes2 := 40
  let total_distance := distance1 + distance2
  let total_time := (hours1 + minutes1 / 60) + (hours2 + minutes2 / 60)
  total_distance / total_time = 55 := 
sorry

end NUMINAMATH_GPT_stan_average_speed_l1463_146387


namespace NUMINAMATH_GPT_parabola_intersections_l1463_146376

theorem parabola_intersections :
  ∃ y1 y2, (∀ x y, (y = 2 * x^2 + 5 * x + 1 ∧ y = - x^2 + 4 * x + 6) → 
     (x = ( -1 + Real.sqrt 61) / 6 ∧ y = y1) ∨ (x = ( -1 - Real.sqrt 61) / 6 ∧ y = y2)) := 
by
  sorry

end NUMINAMATH_GPT_parabola_intersections_l1463_146376


namespace NUMINAMATH_GPT_inequality_holds_l1463_146396

theorem inequality_holds (x : ℝ) (n : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 1) (h3 : n > 0) : 
  (1 + x) ^ n ≥ (1 - x) ^ n + 2 * n * x * (1 - x ^ 2) ^ ((n - 1) / 2) :=
sorry

end NUMINAMATH_GPT_inequality_holds_l1463_146396


namespace NUMINAMATH_GPT_trajectory_equation_l1463_146353

open Real

-- Define points M and N
def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

-- Define the moving point P
def P (x y : ℝ) : Prop := 
  (4 * Real.sqrt ((x + 2) ^ 2 + y ^ 2) + 4 * (x - 2) = 0) → 
  (y ^ 2 = -8 * x)

-- The theorem stating the desired proof problem
theorem trajectory_equation (x y : ℝ) : P x y :=
sorry

end NUMINAMATH_GPT_trajectory_equation_l1463_146353


namespace NUMINAMATH_GPT_birthday_paradox_l1463_146368

-- Defining the problem conditions
def people (n : ℕ) := n ≥ 367

-- Using the Pigeonhole Principle as a condition
def pigeonhole_principle (pigeonholes pigeons : ℕ) := pigeonholes < pigeons

-- Stating the final proposition
theorem birthday_paradox (n : ℕ) (days_in_year : ℕ) (h1 : days_in_year = 366) (h2 : people n) : pigeonhole_principle days_in_year n :=
sorry

end NUMINAMATH_GPT_birthday_paradox_l1463_146368


namespace NUMINAMATH_GPT_total_number_of_birds_l1463_146365

def geese : ℕ := 58
def ducks : ℕ := 37
def swans : ℕ := 42

theorem total_number_of_birds : geese + ducks + swans = 137 := by
  sorry

end NUMINAMATH_GPT_total_number_of_birds_l1463_146365


namespace NUMINAMATH_GPT_part1_part2_l1463_146379

-- We state the problem conditions and theorems to be proven accordingly
variable (A B C : Real) (a b c : Real)

-- Condition 1: In triangle ABC, opposite sides a, b, c with angles A, B, C such that a sin(B - C) = b sin(A - C)
axiom condition1 (A B C : Real) (a b c : Real) : a * Real.sin (B - C) = b * Real.sin (A - C)

-- Question 1: Prove that a = b under the given conditions
theorem part1 (A B C : Real) (a b c : Real) (h1 : a * Real.sin (B - C) = b * Real.sin (A - C)) : a = b := sorry

-- Condition 2: If c = 5 and cos C = 12/13
axiom condition2 (c : Real) : c = 5
axiom condition3 (C : Real) : Real.cos C = 12 / 13

-- Question 2: Prove that the area of triangle ABC is 125/4 under the given conditions
theorem part2 (A B C : Real) (a b c : Real) (h1 : a * Real.sin (B - C) = b * Real.sin (A - C)) 
               (h2 : c = 5) (h3 : Real.cos C = 12 / 13): (1 / 2) * a * b * (Real.sin C) = 125 / 4 := sorry

end NUMINAMATH_GPT_part1_part2_l1463_146379


namespace NUMINAMATH_GPT_dance_lesson_cost_l1463_146354

-- Define the conditions
variable (total_lessons : Nat) (free_lessons : Nat) (paid_lessons_cost : Nat)

-- State the problem with the given conditions
theorem dance_lesson_cost
  (h1 : total_lessons = 10)
  (h2 : free_lessons = 2)
  (h3 : paid_lessons_cost = 80) :
  let number_of_paid_lessons := total_lessons - free_lessons
  number_of_paid_lessons ≠ 0 -> 
  (paid_lessons_cost / number_of_paid_lessons) = 10 := by
  sorry

end NUMINAMATH_GPT_dance_lesson_cost_l1463_146354


namespace NUMINAMATH_GPT_taco_castle_parking_lot_l1463_146345

variable (D F T V : ℕ)

theorem taco_castle_parking_lot (h1 : F = D / 3) (h2 : F = 2 * T) (h3 : V = T / 2) (h4 : V = 5) : D = 60 :=
by
  sorry

end NUMINAMATH_GPT_taco_castle_parking_lot_l1463_146345


namespace NUMINAMATH_GPT_b_bound_for_tangent_parallel_l1463_146394

theorem b_bound_for_tangent_parallel (b : ℝ) (c : ℝ) :
  (∃ x : ℝ, 3 * x^2 - x + b = 0) → b ≤ 1/12 :=
by
  intros h
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_b_bound_for_tangent_parallel_l1463_146394


namespace NUMINAMATH_GPT_jacket_spending_l1463_146369

def total_spent : ℝ := 14.28
def spent_on_shorts : ℝ := 9.54
def spent_on_jacket : ℝ := 4.74

theorem jacket_spending :
  spent_on_jacket = total_spent - spent_on_shorts :=
by sorry

end NUMINAMATH_GPT_jacket_spending_l1463_146369


namespace NUMINAMATH_GPT_toll_for_18_wheel_truck_l1463_146337

-- Definitions based on conditions
def num_axles (total_wheels : ℕ) (wheels_front_axle : ℕ) (wheels_per_other_axle : ℕ) : ℕ :=
  1 + (total_wheels - wheels_front_axle) / wheels_per_other_axle

def toll (x : ℕ) : ℝ :=
  0.50 + 0.50 * (x - 2)

-- The problem statement to prove
theorem toll_for_18_wheel_truck : toll (num_axles 18 2 4) = 2.00 := by
  sorry

end NUMINAMATH_GPT_toll_for_18_wheel_truck_l1463_146337


namespace NUMINAMATH_GPT_birds_never_gather_44_l1463_146382

theorem birds_never_gather_44 :
    ∀ (position : Fin 44 → Nat), 
    (∀ (i : Fin 44), position i ≤ 44) →
    (∀ (i j : Fin 44), position i ≠ position j) →
    ∃ (S : Nat), S % 4 = 2 →
    ∀ (moves : (Fin 44 → Fin 44) → (Fin 44 → Fin 44)),
    ¬(∃ (tree : Nat), ∀ (i : Fin 44), position i = tree) := 
sorry

end NUMINAMATH_GPT_birds_never_gather_44_l1463_146382


namespace NUMINAMATH_GPT_tail_length_10_l1463_146360

theorem tail_length_10 (length_body tail_length head_length width height overall_length: ℝ) 
  (h1 : tail_length = (1 / 2) * length_body)
  (h2 : head_length = (1 / 6) * length_body)
  (h3 : height = 1.5 * width)
  (h4 : overall_length = length_body + tail_length)
  (h5 : overall_length = 30)
  (h6 : width = 12) :
  tail_length = 10 :=
by
  sorry

end NUMINAMATH_GPT_tail_length_10_l1463_146360


namespace NUMINAMATH_GPT_probability_kyle_catherine_not_david_l1463_146339

/--
Kyle, David, and Catherine each try independently to solve a problem. 
Their individual probabilities for success are 1/3, 2/7, and 5/9.
Prove that the probability that Kyle and Catherine, but not David, will solve the problem is 25/189.
-/
theorem probability_kyle_catherine_not_david :
  let P_K := 1 / 3
  let P_D := 2 / 7
  let P_C := 5 / 9
  let P_D_c := 1 - P_D
  P_K * P_C * P_D_c = 25 / 189 :=
by
  sorry

end NUMINAMATH_GPT_probability_kyle_catherine_not_david_l1463_146339


namespace NUMINAMATH_GPT_total_goals_proof_l1463_146312

-- Definitions based on the conditions
def first_half_team_a := 8
def first_half_team_b := first_half_team_a / 2
def first_half_team_c := first_half_team_b * 2

def second_half_team_a := first_half_team_c
def second_half_team_b := first_half_team_a
def second_half_team_c := second_half_team_b + 3

-- Total scores for each team
def total_team_a := first_half_team_a + second_half_team_a
def total_team_b := first_half_team_b + second_half_team_b
def total_team_c := first_half_team_c + second_half_team_c

-- Total goals for all teams
def total_goals := total_team_a + total_team_b + total_team_c

-- The theorem to be proved
theorem total_goals_proof : total_goals = 47 := by
  sorry

end NUMINAMATH_GPT_total_goals_proof_l1463_146312


namespace NUMINAMATH_GPT_ages_total_l1463_146302

variable (A B C : ℕ)

theorem ages_total (h1 : A = B + 2) (h2 : B = 2 * C) (h3 : B = 10) : A + B + C = 27 :=
by
  sorry

end NUMINAMATH_GPT_ages_total_l1463_146302


namespace NUMINAMATH_GPT_initial_ratio_of_milk_to_water_l1463_146364

theorem initial_ratio_of_milk_to_water 
  (M W : ℕ) 
  (h1 : M + 10 + W = 30)
  (h2 : (M + 10) * 2 = W * 5)
  (h3 : M + W = 20) : 
  M = 11 ∧ W = 9 := 
by 
  sorry

end NUMINAMATH_GPT_initial_ratio_of_milk_to_water_l1463_146364


namespace NUMINAMATH_GPT_solve_system_of_equations_solve_fractional_equation_l1463_146333

noncomputable def solution1 (x y : ℚ) := (3 * x - 5 * y = 3) ∧ (x / 2 - y / 3 = 1) ∧ (x = 8 / 3) ∧ (y = 1)

noncomputable def solution2 (x : ℚ) := (x / (x - 1) + 1 = 3 / (2 * x - 2)) ∧ (x = 5 / 4)

theorem solve_system_of_equations (x y : ℚ) : solution1 x y := by
  sorry

theorem solve_fractional_equation (x : ℚ) : solution2 x := by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_solve_fractional_equation_l1463_146333


namespace NUMINAMATH_GPT_current_swans_number_l1463_146383

noncomputable def swans_doubling (S : ℕ) : Prop :=
  let S_after_10_years := S * 2^5 -- Doubling every 2 years for 10 years results in multiplying by 2^5
  S_after_10_years = 480

theorem current_swans_number (S : ℕ) (h : swans_doubling S) : S = 15 := by
  sorry

end NUMINAMATH_GPT_current_swans_number_l1463_146383


namespace NUMINAMATH_GPT_oil_level_drop_l1463_146370

noncomputable def stationary_tank_radius : ℝ := 100
noncomputable def stationary_tank_height : ℝ := 25
noncomputable def truck_tank_radius : ℝ := 7
noncomputable def truck_tank_height : ℝ := 10

noncomputable def π : ℝ := Real.pi
noncomputable def truck_tank_volume := π * truck_tank_radius^2 * truck_tank_height
noncomputable def stationary_tank_area := π * stationary_tank_radius^2

theorem oil_level_drop (volume_truck: ℝ) (area_stationary: ℝ) : volume_truck = 490 * π → area_stationary = π * 10000 → (volume_truck / area_stationary) = 0.049 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_oil_level_drop_l1463_146370


namespace NUMINAMATH_GPT_minimum_bottles_needed_l1463_146381

theorem minimum_bottles_needed (medium_volume jumbo_volume : ℕ) (h_medium : medium_volume = 120) (h_jumbo : jumbo_volume = 2000) : 
  let minimum_bottles := (jumbo_volume + medium_volume - 1) / medium_volume
  minimum_bottles = 17 :=
by
  sorry

end NUMINAMATH_GPT_minimum_bottles_needed_l1463_146381


namespace NUMINAMATH_GPT_book_has_50_pages_l1463_146321

noncomputable def sentences_per_hour : ℕ := 200
noncomputable def hours_to_read : ℕ := 50
noncomputable def sentences_per_paragraph : ℕ := 10
noncomputable def paragraphs_per_page : ℕ := 20

theorem book_has_50_pages :
  (sentences_per_hour * hours_to_read) / sentences_per_paragraph / paragraphs_per_page = 50 :=
by
  sorry

end NUMINAMATH_GPT_book_has_50_pages_l1463_146321


namespace NUMINAMATH_GPT_problem_I_problem_II_l1463_146340

-- Problem (I)
def A : Set ℝ := { x | x > 2 ∨ x < -1 }
def B : Set ℝ := { x | -3 ≤ x ∧ x ≤ 3 }
def A_inter_B : Set ℝ := { x | (-3 ≤ x ∧ x < -1) ∨ (2 < x ∧ x ≤ 3) }

theorem problem_I : A ∩ B = A_inter_B :=
by
  sorry

-- Problem (II)
def C (m : ℝ) : Set ℝ := { x | 2 * m - 1 < x ∧ x < m + 1 }

theorem problem_II (m : ℝ) : (C m ⊆ B) → m ≥ -1 :=
by
  sorry

end NUMINAMATH_GPT_problem_I_problem_II_l1463_146340


namespace NUMINAMATH_GPT_blocks_total_l1463_146311

theorem blocks_total (blocks_initial : ℕ) (blocks_added : ℕ) (total_blocks : ℕ) 
  (h1 : blocks_initial = 86) (h2 : blocks_added = 9) : total_blocks = 95 :=
by
  sorry

end NUMINAMATH_GPT_blocks_total_l1463_146311


namespace NUMINAMATH_GPT_cake_piece_volume_l1463_146391

theorem cake_piece_volume (h : ℝ) (d : ℝ) (n : ℕ) (V_piece : ℝ) : 
  h = 1/2 ∧ d = 16 ∧ n = 8 → V_piece = 4 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_cake_piece_volume_l1463_146391


namespace NUMINAMATH_GPT_decreasing_interval_f_l1463_146305

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 (4*x - x^2)

theorem decreasing_interval_f : ∀ x, (2 < x) ∧ (x < 4) → f x < f (2 : ℝ) :=
by
sorry

end NUMINAMATH_GPT_decreasing_interval_f_l1463_146305


namespace NUMINAMATH_GPT_compute_expression_l1463_146338

theorem compute_expression (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49 / (x - 3)^2 = 23 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l1463_146338


namespace NUMINAMATH_GPT_proof_complement_union_l1463_146318

open Set

variable (U A B: Set Nat)

def complement_equiv_union (U A B: Set Nat) : Prop :=
  (U \ A) ∪ B = {0, 2, 3, 6}

theorem proof_complement_union: 
  U = {0, 1, 3, 5, 6, 8} → 
  A = {1, 5, 8} → 
  B = {2} → 
  complement_equiv_union U A B :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_proof_complement_union_l1463_146318


namespace NUMINAMATH_GPT_at_least_50_singers_l1463_146319

def youth_summer_village (total people_not_working people_with_families max_subset : ℕ) : Prop :=
  total = 100 ∧ 
  people_not_working = 50 ∧ 
  people_with_families = 25 ∧ 
  max_subset = 50

theorem at_least_50_singers (S : ℕ) (h : youth_summer_village 100 50 25 50) : S ≥ 50 :=
by
  obtain ⟨h1, h2, h3, h4⟩ := h
  sorry

end NUMINAMATH_GPT_at_least_50_singers_l1463_146319


namespace NUMINAMATH_GPT_increased_speed_l1463_146351

theorem increased_speed (S : ℝ) : 
  (∀ (usual_speed : ℝ) (usual_time : ℝ) (distance : ℝ), 
    usual_speed = 20 ∧ distance = 100 ∧ usual_speed * usual_time = distance ∧ S * (usual_time - 1) = distance) → 
  S = 25 :=
by
  intros h1
  sorry

end NUMINAMATH_GPT_increased_speed_l1463_146351


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1463_146380

theorem solution_set_of_inequality (x : ℝ) : (|2 * x - 1| < 1) ↔ (0 < x ∧ x < 1) :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1463_146380
