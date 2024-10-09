import Mathlib

namespace nearest_integer_is_11304_l564_56401

def nearest_integer_to_a_plus_b_pow_six (a b : ℝ) (h : b = Real.sqrt 5) : ℝ :=
  (a + b) ^ 6

theorem nearest_integer_is_11304 : nearest_integer_to_a_plus_b_pow_six 3 (Real.sqrt 5) rfl = 11304 := 
  sorry

end nearest_integer_is_11304_l564_56401


namespace min_points_necessary_l564_56426

noncomputable def min_points_on_circle (circumference : ℝ) (dist1 dist2 : ℝ) : ℕ :=
  1304

theorem min_points_necessary :
  ∀ (circumference : ℝ) (dist1 dist2 : ℝ),
  circumference = 1956 →
  dist1 = 1 →
  dist2 = 2 →
  (min_points_on_circle circumference dist1 dist2) = 1304 :=
sorry

end min_points_necessary_l564_56426


namespace minimum_value_of_sum_l564_56469

noncomputable def left_focus (a b c : ℝ) : ℝ := -c 

noncomputable def right_focus (a b c : ℝ) : ℝ := c

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1^2 + v.2^2)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
magnitude (q.1 - p.1, q.2 - p.2)

def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 12) = 1

def P_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola_eq P.1 P.2

theorem minimum_value_of_sum (P : ℝ × ℝ) (A : ℝ × ℝ) (F F' : ℝ × ℝ) (a b c : ℝ)
  (h1 : F = (-c, 0)) (h2 : F' = (c, 0)) (h3 : A = (1, 4)) (h4 : 2 * a = 4)
  (h5 : c^2 = a^2 + b^2) (h6 : P_on_hyperbola P) :
  (|distance P F| + |distance P A|) ≥ 9 :=
sorry

end minimum_value_of_sum_l564_56469


namespace melanie_total_dimes_l564_56482

theorem melanie_total_dimes (d_1 d_2 d_3 : ℕ) (h₁ : d_1 = 19) (h₂ : d_2 = 39) (h₃ : d_3 = 25) : d_1 + d_2 + d_3 = 83 := by
  sorry

end melanie_total_dimes_l564_56482


namespace businessman_earnings_l564_56485

theorem businessman_earnings : 
  let P : ℝ := 1000
  let day1_stock := 1000 / P
  let day2_stock := 1000 / (P * 1.1)
  let day3_stock := 1000 / (P * 1.1^2)
  let value_on_day4 stock := stock * (P * 1.1^3)
  let total_earnings := value_on_day4 day1_stock + value_on_day4 day2_stock + value_on_day4 day3_stock
  total_earnings = 3641 := sorry

end businessman_earnings_l564_56485


namespace determine_n_l564_56477

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem determine_n (n : ℕ) (h1 : binom n 2 + binom n 1 = 6) : n = 3 := 
by
  sorry

end determine_n_l564_56477


namespace minimum_positive_Sn_l564_56472

theorem minimum_positive_Sn (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ n, a (n+1) = a n + d) →
  a 11 / a 10 < -1 →
  (∃ N, ∀ n > N, S n < S (n + 1) ∧ S 1 ≤ S n ∧ ∀ n > N, S n < 0) →
  S 19 > 0 ∧ ∀ k < 19, S k > S 19 → S 19 < 0 →
  n = 19 :=
by
  sorry

end minimum_positive_Sn_l564_56472


namespace complement_intersection_eq_l564_56480

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection_eq :
  U \ (A ∩ B) = {1, 4, 5} := by
  sorry

end complement_intersection_eq_l564_56480


namespace find_first_discount_percentage_l564_56455

def first_discount_percentage 
  (price_initial : ℝ) 
  (price_final : ℝ) 
  (discount_x : ℝ) 
  : Prop := 
  price_initial * (1 - discount_x / 100) * 0.9 * 0.95 = price_final

theorem find_first_discount_percentage :
  first_discount_percentage 9941.52 6800 20.02 :=
by
  sorry

end find_first_discount_percentage_l564_56455


namespace real_roots_of_quad_eq_l564_56428

theorem real_roots_of_quad_eq (p q a : ℝ) (h : p^2 - 4 * q > 0) : 
  (2 * a - p)^2 + 3 * (p^2 - 4 * q) > 0 := 
by
  sorry

end real_roots_of_quad_eq_l564_56428


namespace bailey_discount_l564_56478

noncomputable def discount_percentage (total_cost_without_discount amount_spent : ℝ) : ℝ :=
  ((total_cost_without_discount - amount_spent) / total_cost_without_discount) * 100

theorem bailey_discount :
  let guest_sets := 2
  let master_sets := 4
  let price_guest := 40
  let price_master := 50
  let amount_spent := 224
  let total_cost_without_discount := (guest_sets * price_guest) + (master_sets * price_master)
  discount_percentage total_cost_without_discount amount_spent = 20 := 
by
  sorry

end bailey_discount_l564_56478


namespace least_number_subtracted_divisible_17_l564_56497

theorem least_number_subtracted_divisible_17 :
  ∃ n : ℕ, 165826 - n % 17 = 0 ∧ n = 12 :=
by
  use 12
  sorry  -- Proof will go here.

end least_number_subtracted_divisible_17_l564_56497


namespace total_time_spent_l564_56474

-- Define the conditions
def t1 : ℝ := 2.5
def t2 : ℝ := 3 * t1

-- Define the theorem to prove
theorem total_time_spent : t1 + t2 = 10 := by
  sorry

end total_time_spent_l564_56474


namespace toy_ratio_l564_56499

variable (Jaxon : ℕ) (Gabriel : ℕ) (Jerry : ℕ)

theorem toy_ratio (h1 : Jerry = Gabriel + 8) 
                  (h2 : Jaxon = 15)
                  (h3 : Gabriel + Jerry + Jaxon = 83) :
                  Gabriel / Jaxon = 2 := 
by
  sorry

end toy_ratio_l564_56499


namespace roots_of_polynomial_l564_56464

noncomputable def poly (x : ℝ) : ℝ := x^5 - 3*x^4 + 3*x^3 - x^2 - 4*x + 4

theorem roots_of_polynomial :
  ∀ x : ℝ, poly x = 0 ↔ (x = -1 ∨ x = 1 ∨ x = 2) :=
by
  sorry

end roots_of_polynomial_l564_56464


namespace solve_for_t_l564_56410

variable (A P0 r t : ℝ)

theorem solve_for_t (h : A = P0 * Real.exp (r * t)) : t = (Real.log (A / P0)) / r :=
  by
  sorry

end solve_for_t_l564_56410


namespace rational_expression_l564_56462

theorem rational_expression {x : ℚ} : (∃ a : ℚ, x / (x^2 + x + 1) = a) → (∃ b : ℚ, x^2 / (x^4 + x^2 + 1) = b) := by
  sorry

end rational_expression_l564_56462


namespace find_angle_C_find_a_and_b_l564_56481

-- Conditions from the problem
variables {A B C : ℝ} {a b c : ℝ}
variables {m n : ℝ × ℝ}
variables (h1 : m = (Real.sin A, Real.sin B - Real.sin C))
variables (h2 : n = (a - Real.sqrt 3 * b, b + c))
variables (h3 : m.1 * n.1 + m.2 * n.2 = 0)
variables (h4 : ∀ θ ∈ Set.Ioo 0 Real.pi, θ ≠ C → Real.cos θ = (a^2 + b^2 - c^2) / (2 * a * b))

-- Hypotheses for part (2)
variables (circumradius : ℝ) (area : ℝ)
variables (h5 : circumradius = 2)
variables (h6 : area = Real.sqrt 3)
variables (h7 : a > b)

-- Theorem statement for part (1)
theorem find_angle_C (h1 : m = (Real.sin A, Real.sin B - Real.sin C))
  (h2 : n = (a - Real.sqrt 3 * b, b + c))
  (h3 : m.1 * n.1 + m.2 * n.2 = 0)
  (h4 : ∀ C ∈ Set.Ioo 0 Real.pi, Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) : 
  C = Real.pi / 6 := sorry

-- Theorem statement for part (2)
theorem find_a_and_b (circumradius : ℝ) (area : ℝ) (a b : ℝ)
  (h5 : circumradius = 2) (h6 : area = Real.sqrt 3) (h7 : a > b)
  (h8 : ∀ C ∈ Set.Ioo 0 Real.pi, Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b))
  (h9 : Real.sin C ≠ 0): 
  a = 2 * Real.sqrt 3 ∧ b = 2 := sorry

end find_angle_C_find_a_and_b_l564_56481


namespace min_value_of_inverse_sum_l564_56488

noncomputable def minimumValue (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1 / 3) : ℝ :=
  9 + 6 * Real.sqrt 2

theorem min_value_of_inverse_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1 / 3) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 / 3 ∧ (1/x + 1/y) = 9 + 6 * Real.sqrt 2 := by
  sorry

end min_value_of_inverse_sum_l564_56488


namespace rectangle_area_l564_56487

theorem rectangle_area (length width : ℝ) 
  (h1 : width = 0.9 * length) 
  (h2 : length = 15) : 
  length * width = 202.5 := 
by
  sorry

end rectangle_area_l564_56487


namespace panda_bamboo_consumption_l564_56413

theorem panda_bamboo_consumption (x : ℝ) (h : 0.40 * x = 16) : x = 40 :=
  sorry

end panda_bamboo_consumption_l564_56413


namespace petya_run_12_seconds_l564_56495

-- Define the conditions
variable (petya_speed classmates_speed : ℕ → ℕ) -- speeds of Petya and his classmates
variable (total_distance : ℕ := 100) -- each participant needs to run 100 meters
variable (initial_total_distance_run : ℕ := 288) -- total distance run by all in the first 12 seconds
variable (remaining_distance_when_petya_finished : ℕ := 40) -- remaining distance for others when Petya finished
variable (time_to_first_finish : ℕ) -- the time Petya takes to finish the race

-- Assume constant speeds for all participants
axiom constant_speed_petya (t : ℕ) : petya_speed t = petya_speed 0
axiom constant_speed_classmates (t : ℕ) : classmates_speed t = classmates_speed 0

-- Summarized total distances run by participants
axiom total_distance_run_all (t : ℕ) :
  petya_speed t * t + classmates_speed t * t = initial_total_distance_run + remaining_distance_when_petya_finished + (total_distance - remaining_distance_when_petya_finished) * 3

-- Given conditions converted to Lean
axiom initial_distance_run (t : ℕ) :
  t = 12 → petya_speed t * t + classmates_speed t * t = initial_total_distance_run

axiom petya_completion (t : ℕ) :
  t = time_to_first_finish → petya_speed t * t = total_distance

axiom remaining_distance_classmates (t : ℕ) :
  t = time_to_first_finish → classmates_speed t * (t - time_to_first_finish) = remaining_distance_when_petya_finished
  
-- Define the proof goal using the conditions
theorem petya_run_12_seconds (d : ℕ) :
  (∃ t, t = 12 ∧ d = petya_speed t * t) → d = 80 :=
by
  sorry

end petya_run_12_seconds_l564_56495


namespace misha_students_count_l564_56470

theorem misha_students_count :
  (∀ n : ℕ, n = 60 → (exists better worse : ℕ, better = n - 1 ∧  worse = n - 1)) →
  (∀ n : ℕ, n = 60 → (better + worse + 1 = 119)) :=
by
  sorry

end misha_students_count_l564_56470


namespace total_students_are_45_l564_56400

theorem total_students_are_45 (burgers hot_dogs students : ℕ)
  (h1 : burgers = 30)
  (h2 : burgers = 2 * hot_dogs)
  (h3 : students = burgers + hot_dogs) : students = 45 :=
sorry

end total_students_are_45_l564_56400


namespace true_q_if_not_p_and_p_or_q_l564_56451

variables {p q : Prop}

theorem true_q_if_not_p_and_p_or_q (h1 : ¬p) (h2 : p ∨ q) : q :=
by 
  sorry

end true_q_if_not_p_and_p_or_q_l564_56451


namespace sum_of_squared_residuals_l564_56420

theorem sum_of_squared_residuals (S : ℝ) (r : ℝ) (hS : S = 100) (hr : r = 0.818) : 
    S * (1 - r^2) = 33.0876 :=
by
  rw [hS, hr]
  sorry

end sum_of_squared_residuals_l564_56420


namespace largest_xy_l564_56429

-- Define the problem conditions
def conditions (x y : ℕ) : Prop := 27 * x + 35 * y ≤ 945 ∧ x > 0 ∧ y > 0

-- Define the largest value of xy
def largest_xy_value : ℕ := 234

-- Prove that the largest possible value of xy given conditions is 234
theorem largest_xy (x y : ℕ) (h : conditions x y) : x * y ≤ largest_xy_value :=
sorry

end largest_xy_l564_56429


namespace range_of_a_l564_56445

theorem range_of_a {a : ℝ} : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ - 3 / 5 < a ∧ a ≤ 1 := sorry

end range_of_a_l564_56445


namespace probability_sum_sixteen_l564_56456

-- Define the probabilities involved
def probability_of_coin_fifteen := 1 / 2
def probability_of_die_one := 1 / 6

-- Define the combined probability
def combined_probability : ℚ := probability_of_coin_fifteen * probability_of_die_one

theorem probability_sum_sixteen : combined_probability = 1 / 12 := by
  sorry

end probability_sum_sixteen_l564_56456


namespace alcohol_to_water_ratio_l564_56493

variable {V p q : ℚ}

def alcohol_volume_jar1 (V p : ℚ) : ℚ := (2 * p) / (2 * p + 3) * V
def water_volume_jar1 (V p : ℚ) : ℚ := 3 / (2 * p + 3) * V
def alcohol_volume_jar2 (V q : ℚ) : ℚ := q / (q + 2) * 2 * V
def water_volume_jar2 (V q : ℚ) : ℚ := 2 / (q + 2) * 2 * V

def total_alcohol_volume (V p q : ℚ) : ℚ :=
  alcohol_volume_jar1 V p + alcohol_volume_jar2 V q

def total_water_volume (V p q : ℚ) : ℚ :=
  water_volume_jar1 V p + water_volume_jar2 V q

theorem alcohol_to_water_ratio (V p q : ℚ) :
  (total_alcohol_volume V p q) / (total_water_volume V p q) = (2 * p + 2 * q) / (3 * p + q + 10) :=
by
  sorry

end alcohol_to_water_ratio_l564_56493


namespace total_apples_collected_l564_56444

variable (dailyPicks : ℕ) (days : ℕ) (remainingPicks : ℕ)

theorem total_apples_collected (h1 : dailyPicks = 4) (h2 : days = 30) (h3 : remainingPicks = 230) :
  dailyPicks * days + remainingPicks = 350 :=
by
  sorry

end total_apples_collected_l564_56444


namespace lesser_of_two_numbers_l564_56414

theorem lesser_of_two_numbers (x y : ℝ) (h1 : x + y = 70) (h2 : x * y = 1050) : min x y = 30 :=
sorry

end lesser_of_two_numbers_l564_56414


namespace number_of_nickels_l564_56467

def dimes : ℕ := 10
def pennies_per_dime : ℕ := 10
def pennies_per_nickel : ℕ := 5
def total_pennies : ℕ := 150

theorem number_of_nickels (total_value_dimes : ℕ := dimes * pennies_per_dime)
  (pennies_needed_from_nickels : ℕ := total_pennies - total_value_dimes)
  (n : ℕ) : n = pennies_needed_from_nickels / pennies_per_nickel → n = 10 := by
  sorry

end number_of_nickels_l564_56467


namespace sampling_interval_l564_56441

theorem sampling_interval (total_students sample_size k : ℕ) (h1 : total_students = 1200) (h2 : sample_size = 40) (h3 : k = total_students / sample_size) : k = 30 :=
by
  sorry

end sampling_interval_l564_56441


namespace molecular_weight_H2O_correct_l564_56416

-- Define the atomic weights of hydrogen and oxygen, and the molecular weight of H2O
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight calculation of H2O
def molecular_weight_H2O : ℝ := (2 * atomic_weight_H) + atomic_weight_O

-- Theorem to state the molecular weight of H2O is approximately 18.016 g/mol
theorem molecular_weight_H2O_correct : molecular_weight_H2O = 18.016 :=
by
  -- Putting the value and calculation
  sorry

end molecular_weight_H2O_correct_l564_56416


namespace units_digit_F500_is_7_l564_56425

def F (n : ℕ) : ℕ := 2 ^ (2 ^ (2 * n)) + 1

theorem units_digit_F500_is_7 : (F 500) % 10 = 7 := 
  sorry

end units_digit_F500_is_7_l564_56425


namespace find_first_odd_number_l564_56431

theorem find_first_odd_number (x : ℤ)
  (h : 8 * x = 3 * (x + 4) + 2 * (x + 2) + 5) : x = 7 :=
by
  sorry

end find_first_odd_number_l564_56431


namespace product_of_roots_l564_56442

variable {k m x1 x2 : ℝ}

theorem product_of_roots (h1 : 4 * x1 ^ 2 - k * x1 - m = 0) (h2 : 4 * x2 ^ 2 - k * x2 - m = 0) (h3 : x1 ≠ x2) :
  x1 * x2 = -m / 4 :=
sorry

end product_of_roots_l564_56442


namespace minValue_equality_l564_56436

noncomputable def minValue (a b c : ℝ) : ℝ :=
  (a + 3 * b) * (b + 3 * c) * (a * c + 3)

theorem minValue_equality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 8) :
  minValue a b c = 48 :=
sorry

end minValue_equality_l564_56436


namespace find_x_l564_56446

open Real

noncomputable def satisfies_equation (x : ℝ) : Prop :=
  log (x - 1) / log 3 + log (x^2 - 1) / log (sqrt 3) + log (x - 1) / log (1 / 3) = 3

theorem find_x : ∃ x : ℝ, 1 < x ∧ satisfies_equation x ∧ x = sqrt (1 + 3 * sqrt 3) := by
  sorry

end find_x_l564_56446


namespace geometric_sequence_a7_l564_56457

theorem geometric_sequence_a7
  (a : ℕ → ℤ)
  (is_geom_seq : ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = a n * r)
  (h1 : a 1 = -16)
  (h4 : a 4 = 8) :
  a 7 = -4 := 
sorry

end geometric_sequence_a7_l564_56457


namespace tommy_saw_100_wheels_l564_56459

-- Define the parameters
def trucks : ℕ := 12
def cars : ℕ := 13
def wheels_per_truck : ℕ := 4
def wheels_per_car : ℕ := 4

-- Define the statement to prove
theorem tommy_saw_100_wheels : (trucks * wheels_per_truck + cars * wheels_per_car) = 100 := by
  sorry 

end tommy_saw_100_wheels_l564_56459


namespace correct_statements_l564_56466

theorem correct_statements : 
    let statement1 := "The regression effect is characterized by the relevant exponent R^{2}. The larger the R^{2}, the better the fitting effect."
    let statement2 := "The properties of a sphere are inferred from the properties of a circle by analogy."
    let statement3 := "Any two complex numbers cannot be compared in size."
    let statement4 := "Flowcharts are often used to represent some dynamic processes, usually with a 'starting point' and an 'ending point'."
    true -> (statement1 = "correct" ∧ statement2 = "correct" ∧ statement3 = "incorrect" ∧ statement4 = "incorrect") :=
by
  -- proof
  sorry

end correct_statements_l564_56466


namespace condition_necessary_but_not_sufficient_l564_56417

variable (a b : ℝ)

theorem condition_necessary_but_not_sufficient (h : a ≠ 1 ∨ b ≠ 2) : (a + b ≠ 3) ∧ ¬(a + b ≠ 3 → a ≠ 1 ∨ b ≠ 2) :=
by
  --Proof will go here
  sorry

end condition_necessary_but_not_sufficient_l564_56417


namespace grassy_width_excluding_path_l564_56475

theorem grassy_width_excluding_path
  (l : ℝ) (w : ℝ) (p : ℝ)
  (h1: l = 110) (h2: w = 65) (h3: p = 2.5) :
  w - 2 * p = 60 :=
by
  sorry

end grassy_width_excluding_path_l564_56475


namespace MaireadRan40Miles_l564_56422

def MaireadRanMiles (R : ℝ) (W : ℝ) (J : ℝ) : Prop :=
  W = (3 / 5) * R ∧ J = 3 * R ∧ R + W + J = 184

theorem MaireadRan40Miles : ∃ R W J, MaireadRanMiles R W J ∧ R = 40 :=
by sorry

end MaireadRan40Miles_l564_56422


namespace cost_of_one_book_l564_56454

theorem cost_of_one_book (s b c : ℕ) (h1 : s > 18) (h2 : b > 1) (h3 : c > b) (h4 : s * b * c = 3203) (h5 : s ≤ 36) : c = 11 :=
by
  sorry

end cost_of_one_book_l564_56454


namespace customer_total_payment_l564_56427

structure PaymentData where
  rate : ℕ
  discount1 : ℕ
  lateFee1 : ℕ
  discount2 : ℕ
  lateFee2 : ℕ
  discount3 : ℕ
  lateFee3 : ℕ
  discount4 : ℕ
  lateFee4 : ℕ
  onTime1 : Bool
  onTime2 : Bool
  onTime3 : Bool
  onTime4 : Bool

noncomputable def monthlyPayment (rate discount late_fee : ℕ) (onTime : Bool) : ℕ :=
  if onTime then rate - (rate * discount / 100) else rate + (rate * late_fee / 100)

theorem customer_total_payment (data : PaymentData) : 
  monthlyPayment data.rate data.discount1 data.lateFee1 data.onTime1 +
  monthlyPayment data.rate data.discount2 data.lateFee2 data.onTime2 +
  monthlyPayment data.rate data.discount3 data.lateFee3 data.onTime3 +
  monthlyPayment data.rate data.discount4 data.lateFee4 data.onTime4 = 195 := by
  sorry

end customer_total_payment_l564_56427


namespace seeds_in_fourth_pot_l564_56407

-- Define the conditions as variables
def total_seeds : ℕ := 10
def number_of_pots : ℕ := 4
def seeds_per_pot : ℕ := 3

-- Define the theorem to prove the quantity of seeds planted in the fourth pot
theorem seeds_in_fourth_pot :
  (total_seeds - (seeds_per_pot * (number_of_pots - 1))) = 1 := by
  sorry

end seeds_in_fourth_pot_l564_56407


namespace problem_1_problem_2_l564_56486

-- First problem: Find the solution set for the inequality |x - 1| + |x + 2| ≥ 5
theorem problem_1 (x : ℝ) : (|x - 1| + |x + 2| ≥ 5) ↔ (x ≤ -3 ∨ x ≥ 2) :=
sorry

-- Second problem: Find the range of real number a such that |x - a| + |x + 2| ≤ |x + 4| for all x in [0, 1]
theorem problem_2 (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |x - a| + |x + 2| ≤ |x + 4|) ↔ (-1 ≤ a ∧ a ≤ 2) :=
sorry

end problem_1_problem_2_l564_56486


namespace pigeons_among_non_sparrows_l564_56461

theorem pigeons_among_non_sparrows (P_total P_parrots P_peacocks P_sparrows : ℝ)
    (h1 : P_total = 20)
    (h2 : P_parrots = 30)
    (h3 : P_peacocks = 15)
    (h4 : P_sparrows = 35) :
    (P_total / (100 - P_sparrows)) * 100 = 30.77 :=
by
  -- Proof will be provided here
  sorry

end pigeons_among_non_sparrows_l564_56461


namespace number_of_alligators_l564_56489

theorem number_of_alligators (A : ℕ) 
  (num_snakes : ℕ := 18) 
  (total_eyes : ℕ := 56) 
  (eyes_per_snake : ℕ := 2) 
  (eyes_per_alligator : ℕ := 2) 
  (snakes_eyes : ℕ := num_snakes * eyes_per_snake) 
  (alligators_eyes : ℕ := A * eyes_per_alligator) 
  (total_animals_eyes : ℕ := snakes_eyes + alligators_eyes) 
  (total_eyes_eq : total_animals_eyes = total_eyes) 
: A = 10 :=
by 
  sorry

end number_of_alligators_l564_56489


namespace somu_present_age_l564_56424

theorem somu_present_age (S F : ℕ) (h1 : S = (1 / 3) * F)
    (h2 : S - 5 = (1 / 5) * (F - 5)) : S = 10 := by
  sorry

end somu_present_age_l564_56424


namespace calculate_parallel_segment_length_l564_56494

theorem calculate_parallel_segment_length :
  ∀ (d : ℝ), 
    ∃ (X Y Z P : Type) 
    (XY YZ XZ : ℝ), 
    XY = 490 ∧ 
    YZ = 520 ∧ 
    XZ = 560 ∧ 
    ∃ (D D' E E' F F' : Type),
      (D ≠ E ∧ E ≠ F ∧ F ≠ D') ∧  
      (XZ - (d * (520/490) + d * (520/560))) = d → d = 268.148148 :=
by
  sorry

end calculate_parallel_segment_length_l564_56494


namespace max_min_values_l564_56432

noncomputable def max_value (x y z w : ℝ) : ℝ :=
  if x^2 + y^2 + z^2 + w^2 + x + 2 * y + 3 * z + 4 * w = 17 / 2 then
    max (x + y + z + w) 3
  else
    0

noncomputable def min_value (x y z w : ℝ) : ℝ :=
  if x^2 + y^2 + z^2 + w^2 + x + 2 * y + 3 * z + 4 * w = 17 / 2 then
    min (x + y + z + w) (-2 + 5 / 2 * Real.sqrt 2)
  else
    0

theorem max_min_values (x y z w : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_z : 0 ≤ z) (h_nonneg_w : 0 ≤ w)
  (h_eqn : x^2 + y^2 + z^2 + w^2 + x + 2 * y + 3 * z + 4 * w = 17 / 2) :
  (x + y + z + w ≤ 3) ∧ (x + y + z + w ≥ -2 + 5 / 2 * Real.sqrt 2) :=
by
  sorry

end max_min_values_l564_56432


namespace new_rate_ratio_l564_56408

/--
Hephaestus charged 3 golden apples for the first six months and raised his rate halfway through the year.
Apollo paid 54 golden apples in total for the entire year.
The ratio of the new rate to the old rate is 2.
-/
theorem new_rate_ratio
  (old_rate new_rate : ℕ)
  (total_payment : ℕ)
  (H1 : old_rate = 3)
  (H2 : total_payment = 54)
  (H3 : ∀ R : ℕ, new_rate = R * old_rate ∧ total_payment = 18 + 18 * R) :
  ∃ (R : ℕ), R = 2 :=
by {
  sorry
}

end new_rate_ratio_l564_56408


namespace triangle_ABC_is_acute_l564_56405

theorem triangle_ABC_is_acute (A B C : ℝ) (a b c : ℝ) 
  (h1: a^2 + b^2 >= c^2) (h2: b^2 + c^2 >= a^2) (h3: c^2 + a^2 >= b^2)
  (h4: (Real.sin A + Real.sin B) / (Real.sin B + Real.sin C) = 9 / 11)
  (h5: (Real.sin B + Real.sin C) / (Real.sin C + Real.sin A) = 11 / 10) : 
  A < π / 2 ∧ B < π / 2 ∧ C < π / 2 :=
sorry

end triangle_ABC_is_acute_l564_56405


namespace shaded_quilt_fraction_l564_56483

-- Define the basic structure of the problem using conditions from step a

def is_unit_square (s : ℕ) : Prop := s = 1

def grid_size : ℕ := 4
def total_squares : ℕ := grid_size * grid_size

def shaded_squares : ℕ := 2
def half_shaded_squares : ℕ := 4

def fraction_shaded (shaded: ℕ) (total: ℕ) : ℚ := shaded / total

theorem shaded_quilt_fraction :
  fraction_shaded (shaded_squares + half_shaded_squares / 2) total_squares = 1 / 4 :=
by
  sorry

end shaded_quilt_fraction_l564_56483


namespace chocolate_cost_is_correct_l564_56411

def total_spent : ℕ := 13
def candy_bar_cost : ℕ := 7
def chocolate_cost : ℕ := total_spent - candy_bar_cost

theorem chocolate_cost_is_correct : chocolate_cost = 6 :=
by
  sorry

end chocolate_cost_is_correct_l564_56411


namespace circumscribed_circle_radius_l564_56435

noncomputable def radius_of_circumscribed_circle (b c : ℝ) (A : ℝ) : ℝ :=
  let a := Real.sqrt (b^2 + c^2 - 2 * b * c * Real.cos A)
  let R := a / (2 * Real.sin A)
  R

theorem circumscribed_circle_radius (b c : ℝ) (A : ℝ) (hb : b = 4) (hc : c = 2) (hA : A = Real.pi / 3) :
  radius_of_circumscribed_circle b c A = 2 := by
  sorry

end circumscribed_circle_radius_l564_56435


namespace problem_1_problem_2_l564_56452

noncomputable def problem_1_solution : Set ℝ := {6, -2}
noncomputable def problem_2_solution : Set ℝ := {2 + Real.sqrt 7, 2 - Real.sqrt 7}

theorem problem_1 :
  {x : ℝ | x^2 - 4 * x - 12 = 0} = problem_1_solution :=
by
  sorry

theorem problem_2 :
  {x : ℝ | x^2 - 4 * x - 3 = 0} = problem_2_solution :=
by
  sorry

end problem_1_problem_2_l564_56452


namespace team_X_played_24_games_l564_56465

def games_played_X (x : ℕ) : ℕ := x
def games_played_Y (x : ℕ) : ℕ := x + 9
def games_won_X (x : ℕ) : ℚ := 3 / 4 * x
def games_won_Y (x : ℕ) : ℚ := 2 / 3 * (x + 9)

theorem team_X_played_24_games (x : ℕ) 
  (h1 : games_won_Y x = games_won_X x + 4) : games_played_X x = 24 :=
by
  sorry

end team_X_played_24_games_l564_56465


namespace total_food_items_in_one_day_l564_56440

-- Define the food consumption for each individual
def JorgeCroissants := 7
def JorgeCakes := 18
def JorgePizzas := 30

def GiulianaCroissants := 5
def GiulianaCakes := 14
def GiulianaPizzas := 25

def MatteoCroissants := 6
def MatteoCakes := 16
def MatteoPizzas := 28

-- Define the total number of each food type consumed
def totalCroissants := JorgeCroissants + GiulianaCroissants + MatteoCroissants
def totalCakes := JorgeCakes + GiulianaCakes + MatteoCakes
def totalPizzas := JorgePizzas + GiulianaPizzas + MatteoPizzas

-- The theorem statement
theorem total_food_items_in_one_day : 
  totalCroissants + totalCakes + totalPizzas = 149 :=
by
  -- Proof is omitted
  sorry

end total_food_items_in_one_day_l564_56440


namespace minimum_inverse_sum_l564_56479

theorem minimum_inverse_sum (a b : ℝ) (h1 : (a > 0) ∧ (b > 0)) 
  (h2 : 3 * a + 4 * b = 55) : 
  (1 / a) + (1 / b) ≥ (7 + 4 * Real.sqrt 3) / 55 :=
sorry

end minimum_inverse_sum_l564_56479


namespace find_sum_l564_56419

variable {α : Type*} [LinearOrderedField α] (a : ℕ → α)

-- Geometric sequence condition
def geometric_seq (a : ℕ → α) (r : α) := ∀ n : ℕ, a (n + 1) = a n * r

theorem find_sum (r : α)
  (h1 : geometric_seq a r)
  (h2 : a 4 + a 7 = 2)
  (h3 : a 5 * a 6 = -8) :
  a 1 + a 10 = -7 := 
sorry

end find_sum_l564_56419


namespace new_person_weight_l564_56418

noncomputable def weight_of_new_person (W : ℝ) : ℝ :=
  W + 61 - 25

theorem new_person_weight {W : ℝ} : 
  ((W + 61 - 25) / 12 = W / 12 + 3) → 
  weight_of_new_person W = 61 :=
by
  intro h
  sorry

end new_person_weight_l564_56418


namespace probability_of_A_given_B_l564_56412

-- Definitions of events
def tourist_attractions : List String := ["Pengyuan", "Jiuding Mountain", "Garden Expo Park", "Yunlong Lake", "Pan'an Lake"]

-- Probabilities for each scenario
noncomputable def P_AB : ℝ := 8 / 25
noncomputable def P_B : ℝ := 20 / 25
noncomputable def P_A_given_B : ℝ := 2 / 5

-- Proof statement
theorem probability_of_A_given_B : (P_AB / P_B) = P_A_given_B :=
by
  sorry

end probability_of_A_given_B_l564_56412


namespace arithmetic_sequence_common_difference_l564_56496

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) 
    (h1 : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 81)
    (h2 : a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 171) : 
    ∃ d, d = 10 := 
by 
  sorry

end arithmetic_sequence_common_difference_l564_56496


namespace days_worked_together_l564_56406

theorem days_worked_together (W : ℝ) (h1 : ∀ (a b : ℝ), (a + b) * 40 = W) 
                             (h2 : ∀ a, a * 16 = W) 
                             (x : ℝ) 
                             (h3 : (x * (W / 40) + 12 * (W / 16)) = W) : 
                             x = 10 := 
by
  sorry

end days_worked_together_l564_56406


namespace candidate_lost_by_2340_votes_l564_56491

theorem candidate_lost_by_2340_votes
  (total_votes : ℝ)
  (candidate_percentage : ℝ)
  (rival_percentage : ℝ)
  (candidate_votes : ℝ)
  (rival_votes : ℝ)
  (votes_difference : ℝ)
  (h1 : total_votes = 7800)
  (h2 : candidate_percentage = 0.35)
  (h3 : rival_percentage = 0.65)
  (h4 : candidate_votes = candidate_percentage * total_votes)
  (h5 : rival_votes = rival_percentage * total_votes)
  (h6 : votes_difference = rival_votes - candidate_votes) :
  votes_difference = 2340 :=
by
  sorry

end candidate_lost_by_2340_votes_l564_56491


namespace train_length_is_120_l564_56415

noncomputable def length_of_train (speed_kmh : ℝ) (time_s : ℝ) (bridge_length_m : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  let total_distance := speed_ms * time_s
  total_distance - bridge_length_m

theorem train_length_is_120 :
  length_of_train 70 13.884603517432893 150 = 120 :=
by
  sorry

end train_length_is_120_l564_56415


namespace sector_area_l564_56447

theorem sector_area (θ r arc_length : ℝ) (h_arc_length : arc_length = r * θ) (h_values : θ = 2 ∧ arc_length = 2) :
  1 / 2 * r^2 * θ = 1 := by
  sorry

end sector_area_l564_56447


namespace folded_quadrilateral_has_perpendicular_diagonals_l564_56439

-- Define a quadrilateral and its properties
structure Quadrilateral :=
(A B C D : ℝ × ℝ)

structure Point :=
(x y : ℝ)

-- Define the diagonals within a quadrilateral
def diagonal1 (q : Quadrilateral) : ℝ × ℝ := (q.A.1 - q.C.1, q.A.2 - q.C.2)
def diagonal2 (q : Quadrilateral) : ℝ × ℝ := (q.B.1 - q.D.1, q.B.2 - q.D.2)

-- Define dot product to check perpendicularity
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Condition when folding quadrilateral vertices to a common point ensures no gaps or overlaps
def folding_condition (q : Quadrilateral) (P : Point) : Prop :=
sorry -- Detailed folding condition logic here if needed

-- The statement we need to prove
theorem folded_quadrilateral_has_perpendicular_diagonals (q : Quadrilateral) (P : Point)
    (h_folding : folding_condition q P)
    : dot_product (diagonal1 q) (diagonal2 q) = 0 :=
sorry

end folded_quadrilateral_has_perpendicular_diagonals_l564_56439


namespace degrees_to_radians_90_l564_56473

theorem degrees_to_radians_90 : (90 : ℝ) * (Real.pi / 180) = (Real.pi / 2) :=
by
  sorry

end degrees_to_radians_90_l564_56473


namespace base8_1724_to_base10_l564_56409

/-- Define the base conversion function from base-eight to base-ten -/
def base8_to_base10 (d3 d2 d1 d0 : ℕ) : ℕ :=
  d3 * 8^3 + d2 * 8^2 + d1 * 8^1 + d0 * 8^0

/-- Base-eight representation conditions for the number 1724 -/
def base8_1724_digits := (1, 7, 2, 4)

/-- Prove the base-ten equivalent of the base-eight number 1724 is 980 -/
theorem base8_1724_to_base10 : base8_to_base10 1 7 2 4 = 980 :=
  by
    -- skipping the proof; just state that it is a theorem to be proved.
    sorry

end base8_1724_to_base10_l564_56409


namespace lillian_candies_l564_56449

theorem lillian_candies (initial_candies : ℕ) (additional_candies : ℕ) (total_candies : ℕ) :
  initial_candies = 88 → additional_candies = 5 → total_candies = initial_candies + additional_candies → total_candies = 93 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end lillian_candies_l564_56449


namespace find_a3_l564_56492

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a : ℕ → α) (q : α) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem find_a3
  (a : ℕ → α) (q : α)
  (h_geom : geometric_sequence a q)
  (h_a1 : a 1 = 2)
  (h_cond : a 3 * a 5 = 4 * (a 6) ^ 2) :
  a 3 = 1 :=
by
  sorry

end find_a3_l564_56492


namespace rectangle_area_l564_56460

theorem rectangle_area (w l: ℝ) (h1: l = 2 * w) (h2: 2 * l + 2 * w = 4) : l * w = 8 / 9 := by
  sorry

end rectangle_area_l564_56460


namespace waiter_tip_amount_l564_56421

theorem waiter_tip_amount (n n_no_tip E : ℕ) (h_n : n = 10) (h_no_tip : n_no_tip = 5) (h_E : E = 15) :
  (E / (n - n_no_tip) = 3) :=
by
  -- Proof goes here (we are only writing the statement with sorry)
  sorry

end waiter_tip_amount_l564_56421


namespace max_value_x_plus_y_l564_56490

theorem max_value_x_plus_y (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 48) (hx_mult_4 : x % 4 = 0) : x + y ≤ 49 :=
sorry

end max_value_x_plus_y_l564_56490


namespace no_preimage_iff_lt_one_l564_56448

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem no_preimage_iff_lt_one (k : ℝ) :
  (∀ x : ℝ, f x ≠ k) ↔ k < 1 := 
by
  sorry

end no_preimage_iff_lt_one_l564_56448


namespace cube_faces_opposite_10_is_8_l564_56498

theorem cube_faces_opposite_10_is_8 (nums : Finset ℕ) (h_nums : nums = {6, 7, 8, 9, 10, 11})
  (sum_lateral_first : ℕ) (h_sum_lateral_first : sum_lateral_first = 36)
  (sum_lateral_second : ℕ) (h_sum_lateral_second : sum_lateral_second = 33)
  (faces_opposite_10 : ℕ) (h_faces_opposite_10 : faces_opposite_10 ∈ nums) :
  faces_opposite_10 = 8 :=
by
  sorry

end cube_faces_opposite_10_is_8_l564_56498


namespace isosceles_triangle_l564_56402

theorem isosceles_triangle (a b c : ℝ) (h : a^2 * (b - c) + b^2 * (c - a) + c^2 * (a - b) = 0) : 
  a = b ∨ b = c ∨ c = a :=
sorry

end isosceles_triangle_l564_56402


namespace maximum_area_of_enclosed_poly_l564_56471

theorem maximum_area_of_enclosed_poly (k : ℕ) : 
  ∃ (A : ℕ), (A = 4 * k + 1) :=
sorry

end maximum_area_of_enclosed_poly_l564_56471


namespace xy_sum_square_l564_56463

theorem xy_sum_square (x y : ℕ) (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h1 : x * y + x + y = 119)
  (h2 : x^2 * y + x * y^2 = 1680) :
  x^2 + y^2 = 1057 := by
  sorry

end xy_sum_square_l564_56463


namespace perfect_square_trinomial_l564_56438

theorem perfect_square_trinomial (a k : ℝ) : (∃ b : ℝ, (a^2 + 2*k*a + 9 = (a + b)^2)) ↔ (k = 3 ∨ k = -3) := 
by
  sorry

end perfect_square_trinomial_l564_56438


namespace piglet_straws_l564_56443

theorem piglet_straws (total_straws : ℕ) (straws_adult_pigs_ratio : ℚ) (straws_piglets_ratio : ℚ) (number_piglets : ℕ) :
  total_straws = 300 →
  straws_adult_pigs_ratio = 3/5 →
  straws_piglets_ratio = 1/3 →
  number_piglets = 20 →
  (total_straws * straws_piglets_ratio) / number_piglets = 5 := 
by
  intros
  sorry

end piglet_straws_l564_56443


namespace dishwasher_spending_l564_56430

theorem dishwasher_spending (E : ℝ) (h1 : E > 0) 
    (rent : ℝ := 0.40 * E)
    (left_over : ℝ := 0.28 * E)
    (spent : ℝ := 0.72 * E)
    (dishwasher : ℝ := spent - rent)
    (difference : ℝ := rent - dishwasher) :
    ((difference / rent) * 100) = 20 := 
by
  sorry

end dishwasher_spending_l564_56430


namespace quadratic_solution_1_quadratic_solution_2_l564_56476

theorem quadratic_solution_1 (x : ℝ) : x^2 - 8 * x + 12 = 0 ↔ x = 2 ∨ x = 6 := 
by
  sorry

theorem quadratic_solution_2 (x : ℝ) : (x - 3)^2 = 2 * x * (x - 3) ↔ x = 3 ∨ x = -3 := 
by
  sorry

end quadratic_solution_1_quadratic_solution_2_l564_56476


namespace problem_1_problem_2_l564_56468

def op (x y : ℝ) : ℝ := 3 * x - y

theorem problem_1 (x : ℝ) : op x (op 2 3) = 1 ↔ x = 4 / 3 := by
  -- definitions from conditions
  let def_op_2_3 := op 2 3
  let eq1 := op x def_op_2_3
  -- problem in lean representation
  sorry

theorem problem_2 (x : ℝ) : op (x ^ 2) 2 = 10 ↔ x = 2 ∨ x = -2 := by
  -- problem in lean representation
  sorry

end problem_1_problem_2_l564_56468


namespace range_of_z_l564_56450

theorem range_of_z (α β : ℝ) (z : ℝ) (h1 : -2 < α) (h2 : α ≤ 3) (h3 : 2 < β) (h4 : β ≤ 4) (h5 : z = 2 * α - (1 / 2) * β) :
  -6 < z ∧ z < 5 :=
by
  sorry

end range_of_z_l564_56450


namespace lana_picked_37_roses_l564_56437

def total_flowers_picked (used : ℕ) (extra : ℕ) := used + extra

def picked_roses (total : ℕ) (tulips : ℕ) := total - tulips

theorem lana_picked_37_roses :
    ∀ (tulips used extra : ℕ), tulips = 36 → used = 70 → extra = 3 → 
    picked_roses (total_flowers_picked used extra) tulips = 37 :=
by
  intros tulips used extra htulips husd hextra
  sorry

end lana_picked_37_roses_l564_56437


namespace central_angle_of_sector_l564_56404

theorem central_angle_of_sector
  (r : ℝ) (S_sector : ℝ) (alpha : ℝ) (h₁ : r = 2) (h₂ : S_sector = (2 / 5) * Real.pi)
  (h₃ : S_sector = (1 / 2) * alpha * r^2) : alpha = Real.pi / 5 :=
by
  sorry

end central_angle_of_sector_l564_56404


namespace age_of_youngest_l564_56423

theorem age_of_youngest
  (y : ℕ)
  (h1 : 4 * 25 = y + (y + 2) + (y + 7) + (y + 11)) : y = 20 :=
by
  sorry

end age_of_youngest_l564_56423


namespace computer_price_problem_l564_56453

theorem computer_price_problem (x : ℝ) (h : x + 0.30 * x = 351) : x + 351 = 621 :=
by
  sorry

end computer_price_problem_l564_56453


namespace remaining_wire_in_cm_l564_56484

theorem remaining_wire_in_cm (total_mm : ℝ) (per_mobile_mm : ℝ) (conversion_factor : ℝ) :
  total_mm = 117.6 →
  per_mobile_mm = 4 →
  conversion_factor = 10 →
  ((total_mm % per_mobile_mm) / conversion_factor) = 0.16 :=
by
  intros htotal hmobile hconv
  sorry

end remaining_wire_in_cm_l564_56484


namespace commutative_matrices_implies_fraction_l564_56458

-- Definitions
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, 3], ![4, 5]]
def B (a b c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![![a, b], ![c, d]]

-- Theorem Statement
theorem commutative_matrices_implies_fraction (a b c d : ℝ) 
    (h1 : A * B a b c d = B a b c d * A) 
    (h2 : 4 * b ≠ c) : 
    (a - d) / (c - 4 * b) = 3 / 8 :=
by
  sorry

end commutative_matrices_implies_fraction_l564_56458


namespace number_of_terminating_decimals_l564_56403

theorem number_of_terminating_decimals (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 150) :
  ∃ m, m = 50 ∧ 
  ∀ n, (1 ≤ n ∧ n ≤ 150) → (∃ k, n = 3 * k) →
  m = 50 :=
by 
  sorry

end number_of_terminating_decimals_l564_56403


namespace add_neg3_and_2_mul_neg3_and_2_l564_56434

theorem add_neg3_and_2 : -3 + 2 = -1 := 
by
  sorry

theorem mul_neg3_and_2 : (-3) * 2 = -6 := 
by
  sorry

end add_neg3_and_2_mul_neg3_and_2_l564_56434


namespace points_on_line_initial_l564_56433

theorem points_on_line_initial (x : ℕ) (h : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_initial_l564_56433
