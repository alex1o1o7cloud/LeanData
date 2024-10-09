import Mathlib

namespace task_D_is_suitable_l1203_120393

-- Definitions of the tasks
def task_A := "Investigating the age distribution of your classmates"
def task_B := "Understanding the ratio of male to female students in the eighth grade of your school"
def task_C := "Testing the urine samples of athletes who won championships at the Olympics"
def task_D := "Investigating the sleeping conditions of middle school students in Lishui City"

-- Definition of suitable_for_sampling_survey condition
def suitable_for_sampling_survey (task : String) : Prop :=
  task = task_D

-- Theorem statement
theorem task_D_is_suitable : suitable_for_sampling_survey task_D := by
  -- the proof is omitted
  sorry

end task_D_is_suitable_l1203_120393


namespace paula_remaining_money_l1203_120319

theorem paula_remaining_money (initial_amount cost_per_shirt cost_of_pants : ℕ) 
                             (num_shirts : ℕ) (H1 : initial_amount = 109)
                             (H2 : cost_per_shirt = 11) (H3 : num_shirts = 2)
                             (H4 : cost_of_pants = 13) :
  initial_amount - (num_shirts * cost_per_shirt + cost_of_pants) = 74 := 
by
  -- Calculation of total spent and remaining would go here.
  sorry

end paula_remaining_money_l1203_120319


namespace time_of_free_fall_l1203_120388

theorem time_of_free_fall (h : ℝ) (t : ℝ) (height_fall_eq : h = 4.9 * t^2) (initial_height : h = 490) : t = 10 :=
by
  -- Proof is omitted
  sorry

end time_of_free_fall_l1203_120388


namespace probability_volleyball_is_one_third_l1203_120394

-- Define the total number of test items
def total_test_items : ℕ := 3

-- Define the number of favorable outcomes for hitting the wall with a volleyball
def favorable_outcomes_volleyball : ℕ := 1

-- Define the probability calculation
def probability_hitting_wall_with_volleyball : ℚ :=
  favorable_outcomes_volleyball / total_test_items

-- Prove the probability is 1/3
theorem probability_volleyball_is_one_third :
  probability_hitting_wall_with_volleyball = 1 / 3 := 
sorry

end probability_volleyball_is_one_third_l1203_120394


namespace one_eighth_of_N_l1203_120381

theorem one_eighth_of_N
  (N : ℝ)
  (h : (6 / 11) * N = 48) : (1 / 8) * N = 11 :=
sorry

end one_eighth_of_N_l1203_120381


namespace Ava_watch_minutes_l1203_120358

theorem Ava_watch_minutes (hours_watched : ℕ) (minutes_per_hour : ℕ) (h : hours_watched = 4) (m : minutes_per_hour = 60) : 
  hours_watched * minutes_per_hour = 240 :=
by
  sorry

end Ava_watch_minutes_l1203_120358


namespace inequality_solution_sets_min_value_exists_l1203_120317

-- Define the function f
def f (x : ℝ) (m : ℝ) : ℝ := m * x^2 - 2 * x - 3

-- Existence of roots at -1 and n
def roots_of_quadratic (m : ℝ) (n : ℝ) : Prop :=
  m * (-1)^2 - 2 * (-1) - 3 = 0 ∧ m * n^2 - 2 * n - 3 = 0 ∧ m > 0

-- Main problem statements
theorem inequality_solution_sets (a : ℝ) (m : ℝ) (n : ℝ)
  (h1 : roots_of_quadratic m n) (h2 : m = 1) (h3 : n = 3) (h4 : a > 0) :
  if 0 < a ∧ a ≤ 1 then 
    ∀ x : ℝ, x > 2 / a ∨ x < 2
  else if 1 < a ∧ a < 2 then
    ∀ x : ℝ, x > 2 ∨ x < 2 / a
  else 
    False :=
sorry

theorem min_value_exists (a : ℝ) (m : ℝ)
  (h1 : 0 < a ∧ a < 1) (h2 : m = 1) (h3 : f (a^2) m - 3*a^3 = -5) :
  a = (Real.sqrt 5 - 1) / 2 :=
sorry

end inequality_solution_sets_min_value_exists_l1203_120317


namespace find_x_l1203_120300

theorem find_x (x : ℝ) (h : (1 / 2) * x + (1 / 3) * x = (1 / 4) * x + 7) : x = 12 :=
by
  sorry

end find_x_l1203_120300


namespace polynomial_solution_l1203_120315

noncomputable def q (x : ℝ) : ℝ :=
  -20 / 93 * x^3 - 110 / 93 * x^2 - 372 / 93 * x - 525 / 93

theorem polynomial_solution :
  (q 1 = -11) ∧
  (q 2 = -15) ∧
  (q 3 = -25) ∧
  (q 5 = -65) :=
by
  sorry

end polynomial_solution_l1203_120315


namespace intersection_with_y_axis_l1203_120320

-- Define the original linear function
def original_function (x : ℝ) : ℝ := -2 * x + 3

-- Define the function after moving it up by 2 units
def moved_up_function (x : ℝ) : ℝ := original_function x + 2

-- State the theorem to prove the intersection with the y-axis
theorem intersection_with_y_axis : moved_up_function 0 = 5 :=
by
  sorry

end intersection_with_y_axis_l1203_120320


namespace smallest_m_l1203_120334

theorem smallest_m (m : ℕ) (p q : ℤ) (h_eq : 12 * p * p - m * p + 432 = 0) (h_sum : p + q = m / 12) (h_prod : p * q = 36) :
  m = 144 :=
by
  sorry

end smallest_m_l1203_120334


namespace find_function_l1203_120364

/-- A function f satisfies the equation f(x) + (x + 1/2) * f(1 - x) = 1. -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x + (x + 1 / 2) * f (1 - x) = 1

/-- We want to prove two things:
 1) f(0) = 2 and f(1) = -2
 2) f(x) =  2 / (1 - 2x) for x ≠ 1/2
 -/
theorem find_function (f : ℝ → ℝ) (h : satisfies_equation f) :
  (f 0 = 2 ∧ f 1 = -2) ∧ (∀ x : ℝ, x ≠ 1 / 2 → f x = 2 / (1 - 2 * x)) ∧ (f (1 / 2) = 1 / 2) :=
by
  sorry

end find_function_l1203_120364


namespace teams_match_count_l1203_120337

theorem teams_match_count
  (n : ℕ)
  (h : n = 6)
: (n * (n - 1)) / 2 = 15 := by
  sorry

end teams_match_count_l1203_120337


namespace class_books_transfer_l1203_120340

theorem class_books_transfer :
  ∀ (A B n : ℕ), 
    A = 200 → B = 200 → 
    (B + n = 3/2 * (A - n)) →
    n = 40 :=
by sorry

end class_books_transfer_l1203_120340


namespace find_x_l1203_120304

theorem find_x (y x : ℝ) (h : x / (x - 1) = (y^2 + 2 * y - 1) / (y^2 + 2 * y - 2)) : 
  x = y^2 + 2 * y - 1 := 
sorry

end find_x_l1203_120304


namespace no_internal_angle_less_than_60_l1203_120370

-- Define the concept of a Δ-curve
def delta_curve (K : Type) : Prop := sorry

-- Define the concept of a bicentric Δ-curve
def bicentric_delta_curve (K : Type) : Prop := sorry

-- Define the concept of internal angles of a Δ-curve
def has_internal_angle (K : Type) (A : ℝ) : Prop := sorry

-- The Lean statement for the problem
theorem no_internal_angle_less_than_60 (K : Type) 
  (h1 : delta_curve K) 
  (h2 : has_internal_angle K 60 ↔ bicentric_delta_curve K) :
  (∀ A < 60, ¬has_internal_angle K A) ∧ (has_internal_angle K 60 → bicentric_delta_curve K) := 
sorry

end no_internal_angle_less_than_60_l1203_120370


namespace trig_identity_tangent_l1203_120374

variable {θ : ℝ}

theorem trig_identity_tangent (h : Real.tan θ = 2) : 
  (Real.sin θ * (Real.cos θ * Real.cos θ - Real.sin θ * Real.sin θ)) / (Real.cos θ - Real.sin θ) = 6 / 5 := 
sorry

end trig_identity_tangent_l1203_120374


namespace Marta_books_directly_from_bookstore_l1203_120352

theorem Marta_books_directly_from_bookstore :
  let total_books_sale := 5
  let price_per_book_sale := 10
  let total_books_online := 2
  let total_cost_online := 40
  let total_spent := 210
  let cost_of_books_directly := 3 * total_cost_online
  let total_cost_sale := total_books_sale * price_per_book_sale
  let cost_per_book_directly := cost_of_books_directly / (total_cost_online / total_books_online)
  total_spent = total_cost_sale + total_cost_online + cost_of_books_directly ∧ (cost_of_books_directly / cost_per_book_directly) = 2 :=
by
  sorry

end Marta_books_directly_from_bookstore_l1203_120352


namespace best_fitting_model_is_model3_l1203_120389

-- Definitions of the coefficients of determination for the models
def R2_model1 : ℝ := 0.60
def R2_model2 : ℝ := 0.90
def R2_model3 : ℝ := 0.98
def R2_model4 : ℝ := 0.25

-- The best fitting effect corresponds to the highest R^2 value
theorem best_fitting_model_is_model3 :
  R2_model3 = max (max R2_model1 R2_model2) (max R2_model3 R2_model4) :=
by {
  -- Proofblock is skipped, using sorry
  sorry
}

end best_fitting_model_is_model3_l1203_120389


namespace sin_X_value_l1203_120344

theorem sin_X_value (a b X : ℝ) (h₁ : (1/2) * a * b * Real.sin X = 72) (h₂ : Real.sqrt (a * b) = 16) :
  Real.sin X = 9 / 16 := by
  sorry

end sin_X_value_l1203_120344


namespace find_integer_to_satisfy_eq_l1203_120325

theorem find_integer_to_satisfy_eq (n : ℤ) (h : n - 5 = 2) : n = 7 :=
sorry

end find_integer_to_satisfy_eq_l1203_120325


namespace russel_carousel_rides_l1203_120396

variable (tickets_used : Nat) (tickets_shooting : Nat) (tickets_carousel : Nat)
variable (total_tickets : Nat)
variable (times_shooting : Nat)

theorem russel_carousel_rides :
    times_shooting = 2 →
    tickets_shooting = 5 →
    tickets_carousel = 3 →
    total_tickets = 19 →
    tickets_used = total_tickets - (times_shooting * tickets_shooting) →
    tickets_used / tickets_carousel = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end russel_carousel_rides_l1203_120396


namespace deepak_age_l1203_120332

theorem deepak_age
  (A D : ℕ)
  (h1 : A / D = 5 / 7)
  (h2 : A + 6 = 36) :
  D = 42 :=
by sorry

end deepak_age_l1203_120332


namespace largest_inscribed_rectangle_l1203_120324

theorem largest_inscribed_rectangle {a b m : ℝ} (h : m ≥ b) :
  ∃ (base height area : ℝ),
    base = a * (b + m) / m ∧ 
    height = (b + m) / 2 ∧ 
    area = a * (b + m)^2 / (2 * m) :=
sorry

end largest_inscribed_rectangle_l1203_120324


namespace my_op_evaluation_l1203_120380

def my_op (x y : Int) : Int := x * y - 3 * x + y

theorem my_op_evaluation : my_op 5 3 - my_op 3 5 = -8 := by 
  sorry

end my_op_evaluation_l1203_120380


namespace infinitely_many_good_numbers_seven_does_not_divide_good_number_l1203_120339

-- Define what it means for a number to be good
def is_good_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a + b = n ∧ (a * b) ∣ (n^2 + n + 1)

-- Part (a): Show that there are infinitely many good numbers
theorem infinitely_many_good_numbers : ∃ (f : ℕ → ℕ), ∀ n, is_good_number (f n) :=
sorry

-- Part (b): Show that if n is a good number, then 7 does not divide n
theorem seven_does_not_divide_good_number (n : ℕ) (h : is_good_number n) : ¬ (7 ∣ n) :=
sorry

end infinitely_many_good_numbers_seven_does_not_divide_good_number_l1203_120339


namespace words_difference_l1203_120361

-- Definitions based on conditions.
def right_hand_speed (words_per_minute : ℕ) := 10
def left_hand_speed (words_per_minute : ℕ) := 7
def time_duration (minutes : ℕ) := 5

-- Problem statement
theorem words_difference :
  let right_hand_words := right_hand_speed 0 * time_duration 0
  let left_hand_words := left_hand_speed 0 * time_duration 0
  (right_hand_words - left_hand_words) = 15 :=
by
  sorry

end words_difference_l1203_120361


namespace inequalities_validity_l1203_120341

theorem inequalities_validity (x y a b : ℝ) (hx : x ≤ a) (hy : y ≤ b) (hstrict : x < a ∨ y < b) :
  (x + y ≤ a + b) ∧
  ¬((x - y ≤ a - b) ∧ (x * y ≤ a * b) ∧ (x / y ≤ a / b)) :=
by
  -- Here is where the proof would go.
  sorry

end inequalities_validity_l1203_120341


namespace largest_possible_b_l1203_120312

theorem largest_possible_b (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) (h4 : a * b * c = 360) : b = 10 :=
sorry

end largest_possible_b_l1203_120312


namespace product_a3_a10_a17_l1203_120365

-- Let's define the problem setup
variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a r : α) (n : ℕ) : α := a * r ^ (n - 1)

theorem product_a3_a10_a17 
  (a r : α)
  (h1 : geometric_sequence a r 2 + geometric_sequence a r 18 = -15) 
  (h2 : geometric_sequence a r 2 * geometric_sequence a r 18 = 16) 
  (ha2pos : geometric_sequence a r 18 ≠ 0) 
  (h3 : r < 0) :
  geometric_sequence a r 3 * geometric_sequence a r 10 * geometric_sequence a r 17 = -64 :=
sorry

end product_a3_a10_a17_l1203_120365


namespace minimize_expression_l1203_120373

theorem minimize_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (cond1 : x + y > z) (cond2 : y + z > x) (cond3 : z + x > y) :
  (x + y + z) * (1 / (x + y - z) + 1 / (y + z - x) + 1 / (z + x - y)) ≥ 9 :=
by
  sorry

end minimize_expression_l1203_120373


namespace multiplication_is_valid_l1203_120328

-- Define that the three-digit number n = 306
def three_digit_number := 306

-- The multiplication by 1995 should result in the defined product
def valid_multiplication (n : ℕ) := 1995 * n

theorem multiplication_is_valid : valid_multiplication three_digit_number = 1995 * 306 := by
  -- Since we only need the statement, we use sorry here
  sorry

end multiplication_is_valid_l1203_120328


namespace find_t_l1203_120384

def vector := (ℝ × ℝ)

def a : vector := (-3, 4)
def b : vector := (-1, 5)
def c : vector := (2, 3)

def parallel (v1 v2 : vector) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem find_t (t : ℝ) : 
  parallel (a.1 - c.1, a.2 - c.2) ((2 * t) + b.1, (3 * t) + b.2) ↔ t = -24 / 17 :=
by
  sorry

end find_t_l1203_120384


namespace greatest_constant_right_triangle_l1203_120301

theorem greatest_constant_right_triangle (a b c : ℝ) (h : c^2 = a^2 + b^2) (K : ℝ) 
    (hK : (a^2 + b^2) / (a^2 + b^2 + c^2) > K) : 
    K ≤ 1 / 2 :=
by 
  sorry

end greatest_constant_right_triangle_l1203_120301


namespace sum_of_squares_l1203_120359

theorem sum_of_squares (a b c : ℝ) (h_arith : a + b + c = 30) (h_geom : a * b * c = 216) 
(h_harm : 1/a + 1/b + 1/c = 3/4) : a^2 + b^2 + c^2 = 576 := 
by 
  sorry

end sum_of_squares_l1203_120359


namespace average_speed_v2_l1203_120355

theorem average_speed_v2 (v1 : ℝ) (t : ℝ) (S1 : ℝ) (S2 : ℝ) : 
  (v1 = 30) → (t = 30) → (S1 = 800) → (S2 = 200) → 
  (v2 = (v1 - (S1 - S2) / t) ∨ v2 = (v1 + (S1 - S2) / t)) :=
by
  intros h1 h2 h3 h4
  sorry

end average_speed_v2_l1203_120355


namespace Sn_minimum_value_l1203_120385

theorem Sn_minimum_value {a : ℕ → ℤ} (n : ℕ) (S : ℕ → ℤ)
  (h1 : a 1 = -11)
  (h2 : a 4 + a 6 = -6)
  (S_def : ∀ n, S n = n * (-12 + n)) :
  ∃ n, S n = S 6 :=
sorry

end Sn_minimum_value_l1203_120385


namespace percentage_of_boys_l1203_120360

theorem percentage_of_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) 
  (total_students_eq : total_students = 42)
  (ratio_eq : boy_ratio = 3 ∧ girl_ratio = 4) :
  (boy_ratio + girl_ratio) = 7 ∧ (total_students / 7 * boy_ratio * 100 / total_students : ℚ) = 42.86 :=
by
  sorry

end percentage_of_boys_l1203_120360


namespace initialPersonsCount_l1203_120357

noncomputable def numberOfPersonsInitially (increaseInAverageWeight kg_diff : ℝ) : ℝ :=
  kg_diff / increaseInAverageWeight

theorem initialPersonsCount :
  numberOfPersonsInitially 2.5 20 = 8 := by
  sorry

end initialPersonsCount_l1203_120357


namespace find_big_bonsai_cost_l1203_120369

-- Given definitions based on conditions
def small_bonsai_cost : ℕ := 30
def num_small_bonsai_sold : ℕ := 3
def num_big_bonsai_sold : ℕ := 5
def total_earnings : ℕ := 190

-- Define the function to calculate total earnings from bonsai sales
def calculate_total_earnings (big_bonsai_cost: ℕ) : ℕ :=
  (num_small_bonsai_sold * small_bonsai_cost) + (num_big_bonsai_sold * big_bonsai_cost)

-- The theorem state
theorem find_big_bonsai_cost (B : ℕ) : calculate_total_earnings B = total_earnings → B = 20 :=
by
  sorry

end find_big_bonsai_cost_l1203_120369


namespace probability_two_blue_marbles_l1203_120397

theorem probability_two_blue_marbles (h_red: ℕ := 3) (h_blue: ℕ := 4) (h_white: ℕ := 9) :
  (h_blue / (h_red + h_blue + h_white)) * ((h_blue - 1) / ((h_red + h_blue + h_white) - 1)) = 1 / 20 :=
by sorry

end probability_two_blue_marbles_l1203_120397


namespace total_amount_correct_l1203_120354

noncomputable def total_amount (p_a r_a t_a p_b r_b t_b p_c r_c t_c : ℚ) : ℚ :=
  let final_price (p r t : ℚ) := p - (p * r / 100) + ((p - (p * r / 100)) * t / 100)
  final_price p_a r_a t_a + final_price p_b r_b t_b + final_price p_c r_c t_c

theorem total_amount_correct :
  total_amount 2500 6 10 3150 8 12 1000 5 7 = 6847.26 :=
by
  sorry

end total_amount_correct_l1203_120354


namespace valid_permutations_remainder_l1203_120336

def countValidPermutations : Nat :=
  let total := (Finset.range 3).sum (fun j =>
    Nat.choose 3 (j + 2) * Nat.choose 5 j * Nat.choose 7 (j + 3))
  total % 1000

theorem valid_permutations_remainder :
  countValidPermutations = 60 := 
  sorry

end valid_permutations_remainder_l1203_120336


namespace percentage_of_x_l1203_120307

theorem percentage_of_x (x y : ℝ) (h1 : y = x / 4) (p : ℝ) (h2 : p / 100 * x = 20 / 100 * y) : p = 5 :=
by sorry

end percentage_of_x_l1203_120307


namespace pascal_sixth_element_row_20_l1203_120313

theorem pascal_sixth_element_row_20 : (Nat.choose 20 5) = 7752 := 
  by
  sorry

end pascal_sixth_element_row_20_l1203_120313


namespace find_prices_l1203_120329

variables (C S : ℕ) -- Using natural numbers to represent rubles

theorem find_prices (h1 : C + S = 2500) (h2 : 4 * C + 3 * S = 8870) :
  C = 1370 ∧ S = 1130 :=
by
  sorry

end find_prices_l1203_120329


namespace compare_x_y_l1203_120390

theorem compare_x_y :
  let x := 123456789 * 123456786
  let y := 123456788 * 123456787
  x < y := sorry

end compare_x_y_l1203_120390


namespace rowing_time_to_place_and_back_l1203_120309

open Real

/-- Definitions of the problem conditions -/
def rowing_speed_still_water : ℝ := 5
def current_speed : ℝ := 1
def distance_to_place : ℝ := 2.4

/-- Proof statement: the total time taken to row to the place and back is 1 hour -/
theorem rowing_time_to_place_and_back :
  (distance_to_place / (rowing_speed_still_water + current_speed)) + 
  (distance_to_place / (rowing_speed_still_water - current_speed)) =
  1 := by
  sorry

end rowing_time_to_place_and_back_l1203_120309


namespace inequality_chain_l1203_120382

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem inequality_chain (a b : ℝ) (h1 : b > a) (h2 : a > 3) :
  f b < f ((a + b) / 2) ∧ f ((a + b) / 2) < f (Real.sqrt (a * b)) ∧ f (Real.sqrt (a * b)) < f a :=
by
  sorry

end inequality_chain_l1203_120382


namespace save_percentage_l1203_120350

theorem save_percentage (I S : ℝ) 
  (h1 : 1.5 * I - 2 * S + (I - S) = 2 * (I - S))
  (h2 : I ≠ 0) : 
  S / I = 0.5 :=
by sorry

end save_percentage_l1203_120350


namespace inscribed_circle_radius_l1203_120379

noncomputable def radius_inscribed_circle (DE DF EF : ℝ) : ℝ := 
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s

theorem inscribed_circle_radius :
  radius_inscribed_circle 8 5 9 = 6 * Real.sqrt 11 / 11 :=
by
  sorry

end inscribed_circle_radius_l1203_120379


namespace hiring_manager_acceptance_l1203_120323

theorem hiring_manager_acceptance :
  let average_age := 31
  let std_dev := 9
  let max_diff_ages := 19
  let k := max_diff_ages / (2 * std_dev)
  k = 19 / 18 :=
by
  let average_age := 31
  let std_dev := 9
  let max_diff_ages := 19
  let k := max_diff_ages / (2 * std_dev)
  show k = 19 / 18
  sorry

end hiring_manager_acceptance_l1203_120323


namespace remainder_2017_div_89_l1203_120348

theorem remainder_2017_div_89 : 2017 % 89 = 59 :=
by
  sorry

end remainder_2017_div_89_l1203_120348


namespace functional_relationship_optimizing_profit_l1203_120386

-- Define the scope of the problem with conditions and proof statements

variables (x : ℝ) (y : ℝ)

-- Conditions
def price_condition := 44 ≤ x ∧ x ≤ 52
def sales_function := y = -10 * x + 740
def profit_function (x : ℝ) := -10 * x^2 + 1140 * x - 29600

-- Lean statement to prove the first part
theorem functional_relationship (h₁ : 44 ≤ x) (h₂ : x ≤ 52) : y = -10 * x + 740 := by
  sorry

-- Lean statement to prove the second part
theorem optimizing_profit (h₃ : 44 ≤ x) (h₄ : x ≤ 52) : (profit_function 52 = 2640 ∧ (∀ x, (44 ≤ x ∧ x ≤ 52) → profit_function x ≤ 2640)) := by
  sorry

end functional_relationship_optimizing_profit_l1203_120386


namespace lemonade_syrup_parts_l1203_120363

theorem lemonade_syrup_parts (L : ℝ) :
  (L = 2 / 0.75) →
  (L = 2.6666666666666665) :=
by
  sorry

end lemonade_syrup_parts_l1203_120363


namespace problem1_problem2_l1203_120331

noncomputable def f (x a : ℝ) := x - (x^2 + a * x) / Real.exp x

theorem problem1 (x : ℝ) : (f x 1) ≥ 0 := by
  sorry

theorem problem2 (x : ℝ) : (1 - (Real.log x) / x) * (f x (-1)) > 1 - 1/(Real.exp 2) := by
  sorry

end problem1_problem2_l1203_120331


namespace isosceles_base_lines_l1203_120366
open Real

theorem isosceles_base_lines {x y : ℝ} (h1 : 7 * x - y - 9 = 0) (h2 : x + y - 7 = 0) (hx : x = 3) (hy : y = -8) :
  (x - 3 * y - 27 = 0) ∨ (3 * x + y - 1 = 0) :=
sorry

end isosceles_base_lines_l1203_120366


namespace total_students_in_school_l1203_120316

noncomputable def total_students (girls boys : ℕ) (ratio_girls boys_ratio : ℕ) : ℕ :=
  let parts := ratio_girls + boys_ratio
  let students_per_part := girls / ratio_girls
  students_per_part * parts

theorem total_students_in_school (girls : ℕ) (ratio_girls boys_ratio : ℕ) (h1 : ratio_girls = 5) (h2 : boys_ratio = 8) (h3 : girls = 160) :
  total_students girls boys_ratio ratio_girls = 416 :=
  by
  -- proof would go here
  sorry

end total_students_in_school_l1203_120316


namespace jar_filled_fraction_l1203_120372

variable (S L : ℝ)

-- Conditions
axiom h1 : S * (1/3) = L * (1/2)

-- Statement of the problem
theorem jar_filled_fraction :
  (L * (1/2)) + (S * (1/3)) = L := by
sorry

end jar_filled_fraction_l1203_120372


namespace evaluate_expression_l1203_120367

theorem evaluate_expression : 3^(2 + 3 + 4) - (3^2 * 3^3 + 3^4) = 19359 :=
by
  sorry

end evaluate_expression_l1203_120367


namespace part1_part2_l1203_120327

open Set

namespace ProofProblem

variable (m : ℝ)

def A (m : ℝ) := {x : ℝ | 0 < x - m ∧ x - m < 3}
def B := {x : ℝ | x ≤ 0 ∨ x ≥ 3}

theorem part1 : (A 1 ∩ B) = {x : ℝ | 3 ≤ x ∧ x < 4} := by
  sorry

theorem part2 : (∀ m, (A m ∪ B) = B ↔ (m ≥ 3 ∨ m ≤ -3)) := by
  sorry

end ProofProblem

end part1_part2_l1203_120327


namespace residue_mod_13_l1203_120321

theorem residue_mod_13 : 
  (156 % 13 = 0) ∧ (52 % 13 = 0) ∧ (182 % 13 = 0) ∧ (26 % 13 = 0) →
  (156 + 3 * 52 + 4 * 182 + 6 * 26) % 13 = 0 :=
by
  intros h
  sorry

end residue_mod_13_l1203_120321


namespace serving_guests_possible_iff_even_l1203_120346

theorem serving_guests_possible_iff_even (n : ℕ) : 
  (∀ seats : Finset ℕ, ∀ p : ℕ → ℕ, (∀ i : ℕ, i < n → p i ∈ seats) → 
    (∀ i j : ℕ, i < j → p i ≠ p j) → (n % 2 = 0)) = (n % 2 = 0) :=
by sorry

end serving_guests_possible_iff_even_l1203_120346


namespace initial_percentage_of_alcohol_l1203_120368

theorem initial_percentage_of_alcohol :
  ∃ P : ℝ, (P / 100 * 11) = (33 / 100 * 14) :=
by
  use 42
  sorry

end initial_percentage_of_alcohol_l1203_120368


namespace max_servings_l1203_120371

def servings_prepared (peppers brynza tomatoes cucumbers : ℕ) : ℕ :=
  min (peppers)
      (min (brynza / 75)
           (min (tomatoes / 2) (cucumbers / 2)))

theorem max_servings :
  servings_prepared 60 4200 116 117 = 56 :=
by sorry

end max_servings_l1203_120371


namespace ones_digit_power_sum_l1203_120383

noncomputable def ones_digit_of_power_sum_is_5 : Prop :=
  (1^2010 + 2^2010 + 3^2010 + 4^2010 + 5^2010 + 6^2010 + 7^2010 + 8^2010 + 9^2010 + 10^2010) % 10 = 5

theorem ones_digit_power_sum : ones_digit_of_power_sum_is_5 :=
  sorry

end ones_digit_power_sum_l1203_120383


namespace pictures_left_after_deletion_l1203_120351

variable (zoo museum deleted : ℕ)

def total_pictures_taken (zoo museum : ℕ) : ℕ := zoo + museum

def pictures_remaining (total deleted : ℕ) : ℕ := total - deleted

theorem pictures_left_after_deletion (h1 : zoo = 50) (h2 : museum = 8) (h3 : deleted = 38) :
  pictures_remaining (total_pictures_taken zoo museum) deleted = 20 :=
by
  sorry

end pictures_left_after_deletion_l1203_120351


namespace ken_summit_time_l1203_120395

variables (t : ℕ) (s : ℕ)

/--
Sari and Ken climb up a mountain. 
Ken climbs at a constant pace of 500 meters per hour,
and reaches the summit after \( t \) hours starting from 10:00.
Sari starts climbing 2 hours before Ken at 08:00 and is 50 meters behind Ken when he reaches the summit.
Sari is already 700 meters ahead of Ken when he starts climbing.
Prove that Ken reaches the summit at 15:00.
-/
theorem ken_summit_time (h1 : 500 * t = s * (t + 2) + 50)
  (h2 : s * 2 = 700) : t + 10 = 15 :=

sorry

end ken_summit_time_l1203_120395


namespace find_result_l1203_120375

theorem find_result : ∀ (x : ℝ), x = 1 / 3 → 5 - 7 * x = 8 / 3 := by
  intros x hx
  sorry

end find_result_l1203_120375


namespace can_form_triangle_l1203_120377

theorem can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

example : can_form_triangle 8 6 3 := by
  sorry

end can_form_triangle_l1203_120377


namespace rowing_speed_in_still_water_l1203_120399

theorem rowing_speed_in_still_water (d t1 t2 : ℝ) 
  (h1 : d = 750) (h2 : t1 = 675) (h3 : t2 = 450) : 
  (d / t1 + (d / t2 - d / t1) / 2) = 1.389 := 
by
  sorry

end rowing_speed_in_still_water_l1203_120399


namespace hexahedron_volume_l1203_120345

open Real

noncomputable def volume_of_hexahedron (AB A1B1 AA1 : ℝ) : ℝ :=
  let S_base := (3 * sqrt 3 / 2) * AB^2
  let S_top := (3 * sqrt 3 / 2) * A1B1^2
  let h := AA1
  (1 / 3) * h * (S_base + sqrt (S_base * S_top) + S_top)

theorem hexahedron_volume : volume_of_hexahedron 2 3 (sqrt 10) = 57 * sqrt 3 / 2 := by
  sorry

end hexahedron_volume_l1203_120345


namespace remainder_expr_div_by_5_l1203_120353

theorem remainder_expr_div_by_5 (n : ℤ) : 
  (7 - 2 * n + (n + 5)) % 5 = (-n + 2) % 5 := 
sorry

end remainder_expr_div_by_5_l1203_120353


namespace work_days_for_A_l1203_120311

/-- If A is thrice as fast as B and together they can do a work in 15 days, A alone can do the work in 20 days. -/
theorem work_days_for_A (Wb : ℕ) (Wa : ℕ) (H_wa : Wa = 3 * Wb) (H_total : (Wa + Wb) * 15 = Wa * 20) : A_work_days = 20 :=
by
  sorry

end work_days_for_A_l1203_120311


namespace expected_non_empty_urns_correct_l1203_120342

open ProbabilityTheory

noncomputable def expected_non_empty_urns (n k : ℕ) : ℝ :=
  n * (1 - (1 - 1 / n) ^ k)

theorem expected_non_empty_urns_correct (n k : ℕ) : expected_non_empty_urns n k = n * (1 - ((n - 1) / n) ^ k) :=
by 
  sorry

end expected_non_empty_urns_correct_l1203_120342


namespace find_f_2017_l1203_120347

def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom even_f_shifted : ∀ x : ℝ, f (1 - x) = f (x + 1)
axiom f_neg_one : f (-1) = 2

theorem find_f_2017 : f 2017 = -2 :=
by
  sorry

end find_f_2017_l1203_120347


namespace smallest_possible_value_abs_sum_l1203_120314

theorem smallest_possible_value_abs_sum : 
  ∀ (x : ℝ), 
    (|x + 3| + |x + 6| + |x + 7| + 2) ≥ 8 :=
by
  sorry

end smallest_possible_value_abs_sum_l1203_120314


namespace Joan_seashells_l1203_120330

theorem Joan_seashells (J_J : ℕ) (J : ℕ) (h : J + J_J = 14) (hJJ : J_J = 8) : J = 6 :=
by
  sorry

end Joan_seashells_l1203_120330


namespace find_e_m_l1203_120322

variable {R : Type} [Field R]

def matrix_B (e : R) : Matrix (Fin 2) (Fin 2) R :=
  !![3, 4; 6, e]

theorem find_e_m (e m : R) (hB_inv : (matrix_B e)⁻¹ = m • (matrix_B e)) :
  e = -3 ∧ m = (1 / 11) := by
  sorry

end find_e_m_l1203_120322


namespace kids_go_to_camp_l1203_120318

theorem kids_go_to_camp (total_kids: Nat) (kids_stay_home: Nat) 
  (h1: total_kids = 1363293) (h2: kids_stay_home = 907611) : total_kids - kids_stay_home = 455682 :=
by
  have h_total : total_kids = 1363293 := h1
  have h_stay_home : kids_stay_home = 907611 := h2
  sorry

end kids_go_to_camp_l1203_120318


namespace minimize_area_eq_l1203_120303

theorem minimize_area_eq {l : ℝ → ℝ → Prop}
  (P : ℝ × ℝ) (A B : ℝ × ℝ)
  (condition1 : l P.1 P.2)
  (condition2 : A.1 > 0 ∧ A.2 = 0)
  (condition3 : B.1 = 0 ∧ B.2 > 0)
  (line_eq : ∀ x y : ℝ, l x y ↔ (2 * x + y = 4)) :
  ∀ (a b : ℝ), a = 2 → b = 4 → 2 * P.1 + P.2 = 4 :=
by sorry

end minimize_area_eq_l1203_120303


namespace cube_face_parallel_probability_l1203_120392

theorem cube_face_parallel_probability :
  ∃ (n m : ℕ), (n = 15) ∧ (m = 3) ∧ (m / n = (1 / 5 : ℝ)) := 
sorry

end cube_face_parallel_probability_l1203_120392


namespace probability_open_lock_l1203_120387

/-- Given 5 keys and only 2 can open the lock, the probability of opening the lock by selecting one key randomly is 0.4. -/
theorem probability_open_lock (k : Finset ℕ) (h₁ : k.card = 5) (s : Finset ℕ) (h₂ : s.card = 2 ∧ s ⊆ k) :
  ∃ p : ℚ, p = 0.4 :=
by
  sorry

end probability_open_lock_l1203_120387


namespace triangle_base_length_l1203_120326

theorem triangle_base_length (base : ℝ) (h1 : ∃ (side : ℝ), side = 6 ∧ (side^2 = (base * 12) / 2)) : base = 6 :=
sorry

end triangle_base_length_l1203_120326


namespace cos_double_angle_l1203_120335

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 3 / 5) : Real.cos (2 * theta) = -7 / 25 :=
by
  sorry

end cos_double_angle_l1203_120335


namespace solve_system_of_equations_l1203_120302

theorem solve_system_of_equations :
  ∀ x y : ℝ,
  (y^2 + 2*x*y + x^2 - 6*y - 6*x + 5 = 0) ∧ (y - x + 1 = x^2 - 3*x) ∧ (x ≠ 0) ∧ (x ≠ 3) →
  (x, y) = (-1, 2) ∨ (x, y) = (2, -1) ∨ (x, y) = (-2, 7) :=
by
  sorry

end solve_system_of_equations_l1203_120302


namespace white_tshirts_per_pack_l1203_120333

def packs_of_white := 3
def packs_of_blue := 2
def blue_in_each_pack := 4
def total_tshirts := 26

theorem white_tshirts_per_pack :
  ∃ W : ℕ, packs_of_white * W + packs_of_blue * blue_in_each_pack = total_tshirts ∧ W = 6 :=
by
  sorry

end white_tshirts_per_pack_l1203_120333


namespace digit_d_for_5678d_is_multiple_of_9_l1203_120349

def is_multiple_of_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem digit_d_for_5678d_is_multiple_of_9 : 
  ∃ d : ℕ, d < 10 ∧ is_multiple_of_9 (56780 + d) ∧ d = 1 :=
by
  sorry

end digit_d_for_5678d_is_multiple_of_9_l1203_120349


namespace number_of_people_study_only_cooking_l1203_120310

def total_yoga : Nat := 25
def total_cooking : Nat := 18
def total_weaving : Nat := 10
def cooking_and_yoga : Nat := 5
def all_three : Nat := 4
def cooking_and_weaving : Nat := 5

theorem number_of_people_study_only_cooking :
  (total_cooking - (cooking_and_yoga + cooking_and_weaving - all_three)) = 12 :=
by
  sorry

end number_of_people_study_only_cooking_l1203_120310


namespace g_of_zero_l1203_120343

theorem g_of_zero (f g : ℤ → ℤ) (h₁ : ∀ x, f x = 2 * x + 3) (h₂ : ∀ x, g (x + 2) = f x) : 
  g 0 = -1 :=
by
  sorry

end g_of_zero_l1203_120343


namespace chord_length_count_l1203_120305

noncomputable def number_of_chords (d r : ℕ) : ℕ := sorry

theorem chord_length_count {d r : ℕ} (h1 : d = 12) (h2 : r = 13) :
  number_of_chords d r = 17 :=
sorry

end chord_length_count_l1203_120305


namespace find_number_l1203_120338

theorem find_number (x : ℝ) : (35 - x) * 2 + 12 = 72 → ((35 - x) * 2 + 12) / 8 = 9 → x = 5 :=
by
  -- assume the first condition
  intro h1
  -- assume the second condition
  intro h2
  -- the proof goes here
  sorry

end find_number_l1203_120338


namespace mean_visits_between_200_and_300_l1203_120378

def monday_visits := 300
def tuesday_visits := 400
def wednesday_visits := 300
def thursday_visits := 200
def friday_visits := 200

def total_visits := monday_visits + tuesday_visits + wednesday_visits + thursday_visits + friday_visits
def number_of_days := 5
def mean_visits_per_day := total_visits / number_of_days

theorem mean_visits_between_200_and_300 : 200 ≤ mean_visits_per_day ∧ mean_visits_per_day ≤ 300 :=
by sorry

end mean_visits_between_200_and_300_l1203_120378


namespace certain_number_is_45_l1203_120356

-- Define the variables and condition
def x : ℝ := 45
axiom h : x * 7 = 0.35 * 900

-- The statement we need to prove
theorem certain_number_is_45 : x = 45 :=
by
  sorry

end certain_number_is_45_l1203_120356


namespace circle_radius_zero_l1203_120306

theorem circle_radius_zero :
  ∀ (x y : ℝ),
    (4 * x^2 - 8 * x + 4 * y^2 - 16 * y + 20 = 0) →
    ((x - 1)^2 + (y - 2)^2 = 0) → 
    0 = 0 :=
by
  intros x y h_eq h_circle
  sorry

end circle_radius_zero_l1203_120306


namespace ratio_of_black_to_white_areas_l1203_120398

theorem ratio_of_black_to_white_areas :
  let π := Real.pi
  let radii := [2, 4, 6, 8]
  let areas := [π * (radii[0])^2, π * (radii[1])^2, π * (radii[2])^2, π * (radii[3])^2]
  let black_areas := [areas[0], areas[2] - areas[1]]
  let white_areas := [areas[1] - areas[0], areas[3] - areas[2]]
  let total_black_area := black_areas.sum
  let total_white_area := white_areas.sum
  let ratio := total_black_area / total_white_area
  ratio = 3 / 5 := sorry

end ratio_of_black_to_white_areas_l1203_120398


namespace product_of_two_numbers_l1203_120308

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 404) : x * y = 86 :=
sorry

end product_of_two_numbers_l1203_120308


namespace proof_l1203_120391

noncomputable def question := ∀ x : ℝ, (0.12 * x = 36) → (0.5 * (0.4 * 0.3 * x) = 18) 

theorem proof : question :=
by
  intro x
  intro h
  sorry

end proof_l1203_120391


namespace product_of_coprime_numbers_l1203_120376

variable {a b c : ℕ}

def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem product_of_coprime_numbers (h1 : coprime a b) (h2 : a * b = c) : Nat.lcm a b = c := by
  sorry

end product_of_coprime_numbers_l1203_120376


namespace log_49_48_in_terms_of_a_and_b_l1203_120362

-- Define the constants and hypotheses
variable (a b : ℝ)
variable (h1 : a = Real.logb 7 3)
variable (h2 : b = Real.logb 7 4)

-- Define the statement to be proved
theorem log_49_48_in_terms_of_a_and_b (a b : ℝ) (h1 : a = Real.logb 7 3) (h2 : b = Real.logb 7 4) :
  Real.logb 49 48 = (a + 2 * b) / 2 :=
by
  sorry

end log_49_48_in_terms_of_a_and_b_l1203_120362
