import Mathlib

namespace NUMINAMATH_GPT_primes_solution_l2319_231982

theorem primes_solution (p : ℕ) (n : ℕ) (h_prime : Prime p) (h_nat : 0 < n) : 
  (p^2 + n^2 = 3 * p * n + 1) ↔ (p = 3 ∧ n = 1) ∨ (p = 3 ∧ n = 8) := sorry

end NUMINAMATH_GPT_primes_solution_l2319_231982


namespace NUMINAMATH_GPT_history_paper_pages_l2319_231941

theorem history_paper_pages (p d : ℕ) (h1 : p = 11) (h2 : d = 3) : p * d = 33 :=
by
  sorry

end NUMINAMATH_GPT_history_paper_pages_l2319_231941


namespace NUMINAMATH_GPT_ball_hits_ground_at_10_over_7_l2319_231980

def ball_hits_ground (t : ℚ) : Prop :=
  -4.9 * t^2 + 3.5 * t + 5 = 0

theorem ball_hits_ground_at_10_over_7 : ball_hits_ground (10 / 7) :=
by
  sorry

end NUMINAMATH_GPT_ball_hits_ground_at_10_over_7_l2319_231980


namespace NUMINAMATH_GPT_necessary_not_sufficient_l2319_231964

theorem necessary_not_sufficient (m a : ℝ) (h : a ≠ 0) :
  (|m| = a → m = -a ∨ m = a) ∧ ¬ (m = -a ∨ m = a → |m| = a) :=
by
  sorry

end NUMINAMATH_GPT_necessary_not_sufficient_l2319_231964


namespace NUMINAMATH_GPT_thomas_probability_of_two_pairs_l2319_231900

def number_of_ways_to_choose_five_socks := Nat.choose 12 5
def number_of_ways_to_choose_two_pairs_of_colors := Nat.choose 4 2
def number_of_ways_to_choose_one_color_for_single_sock := Nat.choose 2 1
def number_of_ways_to_choose_two_socks_from_three := Nat.choose 3 2
def number_of_ways_to_choose_one_sock_from_three := Nat.choose 3 1

theorem thomas_probability_of_two_pairs : 
  number_of_ways_to_choose_five_socks = 792 →
  number_of_ways_to_choose_two_pairs_of_colors = 6 →
  number_of_ways_to_choose_one_color_for_single_sock = 2 →
  number_of_ways_to_choose_two_socks_from_three = 3 →
  number_of_ways_to_choose_one_sock_from_three = 3 →
  6 * 2 * 3 * 3 * 3 = 324 →
  (324 : ℚ) / 792 = 9 / 22 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_thomas_probability_of_two_pairs_l2319_231900


namespace NUMINAMATH_GPT_largest_integer_less_than_100_with_remainder_5_l2319_231913

theorem largest_integer_less_than_100_with_remainder_5 :
  ∃ n, (n < 100 ∧ n % 8 = 5) ∧ ∀ m, (m < 100 ∧ m % 8 = 5) → m ≤ n :=
sorry

end NUMINAMATH_GPT_largest_integer_less_than_100_with_remainder_5_l2319_231913


namespace NUMINAMATH_GPT_incorrect_description_is_A_l2319_231906

-- Definitions for the conditions
def description_A := "Increasing the concentration of reactants increases the percentage of activated molecules, accelerating the reaction rate."
def description_B := "Increasing the pressure of a gaseous reaction system increases the number of activated molecules per unit volume, accelerating the rate of the gas reaction."
def description_C := "Raising the temperature of the reaction increases the percentage of activated molecules, increases the probability of effective collisions, and increases the reaction rate."
def description_D := "Catalysts increase the reaction rate by changing the reaction path and lowering the activation energy required for the reaction."

-- Problem Statement
theorem incorrect_description_is_A :
  description_A ≠ correct :=
  sorry

end NUMINAMATH_GPT_incorrect_description_is_A_l2319_231906


namespace NUMINAMATH_GPT_paving_cost_correct_l2319_231977

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate : ℝ := 400
def area : ℝ := length * width
def cost : ℝ := area * rate

theorem paving_cost_correct :
  cost = 8250 := by
  sorry

end NUMINAMATH_GPT_paving_cost_correct_l2319_231977


namespace NUMINAMATH_GPT_total_vacations_and_classes_l2319_231924

def kelvin_classes := 90
def grant_vacations := 4 * kelvin_classes
def total := grant_vacations + kelvin_classes

theorem total_vacations_and_classes :
  total = 450 :=
by
  sorry

end NUMINAMATH_GPT_total_vacations_and_classes_l2319_231924


namespace NUMINAMATH_GPT_compare_exponents_l2319_231999

theorem compare_exponents (n : ℕ) (hn : n > 8) :
  let a := Real.sqrt n
  let b := Real.sqrt (n + 1)
  a^b > b^a :=
sorry

end NUMINAMATH_GPT_compare_exponents_l2319_231999


namespace NUMINAMATH_GPT_rectangle_error_percent_deficit_l2319_231919

theorem rectangle_error_percent_deficit (L W : ℝ) (p : ℝ) 
    (h1 : L > 0) (h2 : W > 0)
    (h3 : 1.05 * (1 - p) = 1.008) :
    p = 0.04 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_error_percent_deficit_l2319_231919


namespace NUMINAMATH_GPT_find_m_value_l2319_231956

noncomputable def hyperbola_m_value (m : ℝ) : Prop :=
  let a := 1
  let b := 2 * a
  m = -(1/4)

theorem find_m_value :
  (∀ x y : ℝ, x^2 + m * y^2 = 1 → b = 2 * a) → hyperbola_m_value m :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_m_value_l2319_231956


namespace NUMINAMATH_GPT_carrots_remaining_l2319_231966

theorem carrots_remaining 
  (total_carrots : ℕ)
  (weight_20_carrots : ℕ)
  (removed_carrots : ℕ)
  (avg_weight_remaining : ℕ)
  (avg_weight_removed : ℕ)
  (h1 : total_carrots = 20)
  (h2 : weight_20_carrots = 3640)
  (h3 : removed_carrots = 4)
  (h4 : avg_weight_remaining = 180)
  (h5 : avg_weight_removed = 190) :
  total_carrots - removed_carrots = 16 :=
by 
  -- h1 : 20 carrots in total
  -- h2 : total weight of 20 carrots is 3640 grams
  -- h3 : 4 carrots are removed
  -- h4 : average weight of remaining carrots is 180 grams
  -- h5 : average weight of removed carrots is 190 grams
  sorry

end NUMINAMATH_GPT_carrots_remaining_l2319_231966


namespace NUMINAMATH_GPT_MountainRidgeAcademy_l2319_231937

theorem MountainRidgeAcademy (j s : ℕ) 
  (h1 : 3/4 * j = 1/2 * s) : s = 3/2 * j := 
by 
  sorry

end NUMINAMATH_GPT_MountainRidgeAcademy_l2319_231937


namespace NUMINAMATH_GPT_quadratic_square_binomial_l2319_231921

theorem quadratic_square_binomial (d : ℝ) : (∃ b : ℝ, (x : ℝ) -> (x + b)^2 = x^2 + 110 * x + d) ↔ d = 3025 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_square_binomial_l2319_231921


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2319_231954

noncomputable def given_expression (a : ℝ) : ℝ :=
  (a-4) / a / ((a+2) / (a^2 - 2 * a) - (a-1) / (a^2 - 4 * a + 4))

theorem simplify_and_evaluate (a : ℝ) (h : a^2 - 4 * a + 3 = 0) : given_expression a = 1 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2319_231954


namespace NUMINAMATH_GPT_car_speeds_l2319_231909

noncomputable def distance_between_places : ℝ := 135
noncomputable def departure_time_diff : ℝ := 4 -- large car departs 4 hours before small car
noncomputable def arrival_time_diff : ℝ := 0.5 -- small car arrives 30 minutes earlier than large car
noncomputable def speed_ratio : ℝ := 5 / 2 -- ratio of speeds (small car : large car)

theorem car_speeds (v_small v_large : ℝ) (h1 : v_small / v_large = speed_ratio) :
    v_small = 45 ∧ v_large = 18 :=
sorry

end NUMINAMATH_GPT_car_speeds_l2319_231909


namespace NUMINAMATH_GPT_minimum_distance_l2319_231991

open Real

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y + 4 = 0
def parabola_eq (x y : ℝ) : Prop := y^2 = 8 * x

theorem minimum_distance :
  ∃ (A B : ℝ × ℝ), circle_eq A.1 A.2 ∧ parabola_eq B.1 B.2 ∧ dist A B = 1 / 2 :=
sorry

end NUMINAMATH_GPT_minimum_distance_l2319_231991


namespace NUMINAMATH_GPT_find_AC_l2319_231908

theorem find_AC (A B C : ℝ) (r1 r2 : ℝ) (AB : ℝ) (AC : ℝ) 
  (h_rad1 : r1 = 1) (h_rad2 : r2 = 3) (h_AB : AB = 2 * Real.sqrt 5) 
  (h_AC : AC = AB / 4) :
  AC = Real.sqrt 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_AC_l2319_231908


namespace NUMINAMATH_GPT_perimeter_eq_28_l2319_231965

theorem perimeter_eq_28 (PQ QR TS TU : ℝ) (h2 : PQ = 4) (h3 : QR = 4) 
(h5 : TS = 8) (h7 : TU = 4) : 
PQ + QR + TS + TS - TU + TU + TU = 28 := by
  sorry

end NUMINAMATH_GPT_perimeter_eq_28_l2319_231965


namespace NUMINAMATH_GPT_mail_difference_eq_15_l2319_231989

variable (Monday Tuesday Wednesday Thursday : ℕ)
variable (total : ℕ)

theorem mail_difference_eq_15
  (h1 : Monday = 65)
  (h2 : Tuesday = Monday + 10)
  (h3 : Wednesday = Tuesday - 5)
  (h4 : total = 295)
  (h5 : total = Monday + Tuesday + Wednesday + Thursday) :
  Thursday - Wednesday = 15 := 
  by
  sorry

end NUMINAMATH_GPT_mail_difference_eq_15_l2319_231989


namespace NUMINAMATH_GPT_gcd_and_sum_of_1729_and_867_l2319_231996

-- Given numbers
def a := 1729
def b := 867

-- Define the problem statement
theorem gcd_and_sum_of_1729_and_867 : Nat.gcd a b = 1 ∧ a + b = 2596 := by
  sorry

end NUMINAMATH_GPT_gcd_and_sum_of_1729_and_867_l2319_231996


namespace NUMINAMATH_GPT_valid_q_values_l2319_231967

theorem valid_q_values (q : ℕ) (h : q > 0) :
  q = 3 ∨ q = 4 ∨ q = 9 ∨ q = 28 ↔ ((5 * q + 40) / (3 * q - 8)) * (3 * q - 8) = 5 * q + 40 :=
by
  sorry

end NUMINAMATH_GPT_valid_q_values_l2319_231967


namespace NUMINAMATH_GPT_percentage_of_boys_from_schoolA_study_science_l2319_231934

variable (T : ℝ) -- Total number of boys in the camp
variable (schoolA_boys : ℝ)
variable (science_boys : ℝ)

noncomputable def percentage_science_boys := (science_boys / schoolA_boys) * 100

theorem percentage_of_boys_from_schoolA_study_science 
  (h1 : schoolA_boys = 0.20 * T)
  (h2 : science_boys = schoolA_boys - 56)
  (h3 : T = 400) :
  percentage_science_boys science_boys schoolA_boys = 30 := 
by sorry

end NUMINAMATH_GPT_percentage_of_boys_from_schoolA_study_science_l2319_231934


namespace NUMINAMATH_GPT_four_digit_number_8802_l2319_231925

theorem four_digit_number_8802 (x : ℕ) (a b c d : ℕ) (h1 : 1000 ≤ x ∧ x ≤ 9999)
  (h2 : x = 1000 * a + 100 * b + 10 * c + d)
  (h3 : a ≠ 0)  -- since a 4-digit number cannot start with 0
  (h4 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10) : 
  x + 8802 = 1099 + 8802 :=
by
  sorry

end NUMINAMATH_GPT_four_digit_number_8802_l2319_231925


namespace NUMINAMATH_GPT_min_value_fractions_l2319_231963

open Real

theorem min_value_fractions (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 3) :
  3 ≤ (1 / (2 * a + b) + 1 / (2 * b + c) + 1 / (2 * c + a)) :=
sorry

end NUMINAMATH_GPT_min_value_fractions_l2319_231963


namespace NUMINAMATH_GPT_number_of_pairs_sold_l2319_231938

-- Define the conditions
def total_amount_made : ℝ := 588
def average_price_per_pair : ℝ := 9.8

-- The theorem we want to prove
theorem number_of_pairs_sold : total_amount_made / average_price_per_pair = 60 := 
by sorry

end NUMINAMATH_GPT_number_of_pairs_sold_l2319_231938


namespace NUMINAMATH_GPT_equal_integers_l2319_231969

theorem equal_integers (a b : ℕ)
  (h : ∀ n : ℕ, n > 0 → a > 0 → b > 0 → (a^n + n) ∣ (b^n + n)) : a = b := 
sorry

end NUMINAMATH_GPT_equal_integers_l2319_231969


namespace NUMINAMATH_GPT_jaylen_bell_peppers_ratio_l2319_231959

theorem jaylen_bell_peppers_ratio :
  ∃ j_bell_p, ∃ k_bell_p, ∃ j_green_b, ∃ k_green_b, ∃ j_carrots, ∃ j_cucumbers, ∃ j_total_veg,
  j_carrots = 5 ∧
  j_cucumbers = 2 ∧
  k_bell_p = 2 ∧
  k_green_b = 20 ∧
  j_green_b = 20 / 2 - 3 ∧
  j_total_veg = 18 ∧
  j_carrots + j_cucumbers + j_green_b + j_bell_p = j_total_veg ∧
  j_bell_p / k_bell_p = 2 :=
sorry

end NUMINAMATH_GPT_jaylen_bell_peppers_ratio_l2319_231959


namespace NUMINAMATH_GPT_vikki_hourly_pay_rate_l2319_231950

-- Define the variables and conditions
def hours_worked : ℝ := 42
def tax_rate : ℝ := 0.20
def insurance_rate : ℝ := 0.05
def union_dues : ℝ := 5
def net_pay : ℝ := 310

-- Define Vikki's hourly pay rate (we will solve for this)
variable (hourly_pay : ℝ)

-- Define the gross earnings
def gross_earnings (hourly_pay : ℝ) : ℝ := hours_worked * hourly_pay

-- Define the total deductions
def total_deductions (hourly_pay : ℝ) : ℝ := (tax_rate * gross_earnings hourly_pay) + (insurance_rate * gross_earnings hourly_pay) + union_dues

-- Define the net pay
def calculate_net_pay (hourly_pay : ℝ) : ℝ := gross_earnings hourly_pay - total_deductions hourly_pay

-- Prove the solution
theorem vikki_hourly_pay_rate : calculate_net_pay hourly_pay = net_pay → hourly_pay = 10 := by
  sorry

end NUMINAMATH_GPT_vikki_hourly_pay_rate_l2319_231950


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l2319_231902

theorem isosceles_triangle_perimeter :
  (∃ x y : ℝ, x^2 - 6*x + 8 = 0 ∧ y^2 - 6*y + 8 = 0 ∧ (x = 2 ∧ y = 4) ∧ 2 + 4 + 4 = 10) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l2319_231902


namespace NUMINAMATH_GPT_value_of_MN_l2319_231968

theorem value_of_MN (M N : ℝ) (log : ℝ → ℝ → ℝ)
    (h1 : log (M ^ 2) N = log N (M ^ 2))
    (h2 : M ≠ N)
    (h3 : M * N > 0)
    (h4 : M ≠ 1)
    (h5 : N ≠ 1) :
    M * N = N^(1/2) :=
  sorry

end NUMINAMATH_GPT_value_of_MN_l2319_231968


namespace NUMINAMATH_GPT_probability_of_scoring_l2319_231930

theorem probability_of_scoring :
  ∀ (p : ℝ), (p + (1 / 3) * p = 1) → (p = 3 / 4) → (p * (1 - p) = 3 / 16) :=
by
  intros p h1 h2
  sorry

end NUMINAMATH_GPT_probability_of_scoring_l2319_231930


namespace NUMINAMATH_GPT_point_on_hyperbola_l2319_231979

theorem point_on_hyperbola (x y : ℝ) (h_eqn : y = -4 / x) (h_point : x = -2 ∧ y = 2) : x * y = -4 := 
by
  intros
  sorry

end NUMINAMATH_GPT_point_on_hyperbola_l2319_231979


namespace NUMINAMATH_GPT_arith_seq_sum_proof_l2319_231920

open Function

variable (a : ℕ → ℕ) -- Define the arithmetic sequence
variables (S : ℕ → ℕ) -- Define the sum function of the sequence

-- Conditions: S_8 = 9 and S_5 = 6
axiom S8 : S 8 = 9
axiom S5 : S 5 = 6

-- Mathematical equivalence
theorem arith_seq_sum_proof : S 13 = 13 :=
sorry

end NUMINAMATH_GPT_arith_seq_sum_proof_l2319_231920


namespace NUMINAMATH_GPT_speed_of_first_train_l2319_231905

noncomputable def speed_of_second_train : ℝ := 40 -- km/h
noncomputable def length_of_first_train : ℝ := 125 -- m
noncomputable def length_of_second_train : ℝ := 125.02 -- m
noncomputable def time_to_pass_each_other : ℝ := 1.5 / 60 -- hours (converted from minutes)

theorem speed_of_first_train (V1 V2 : ℝ) 
  (h1 : V2 = speed_of_second_train)
  (h2 : 125 + 125.02 = 250.02) 
  (h3 : 1.5 / 60 = 0.025) :
  V1 - V2 = 10.0008 → V1 = 50 :=
by 
  sorry

end NUMINAMATH_GPT_speed_of_first_train_l2319_231905


namespace NUMINAMATH_GPT_sequence_geometric_l2319_231957

theorem sequence_geometric {a_n : ℕ → ℕ} (S : ℕ → ℕ) (a1 a2 a3 : ℕ) 
(hS : ∀ n, S n = 2 * a_n n - a_n 1) 
(h_arith : 2 * (a_n 2 + 1) = a_n 3 + a_n 1) : 
  ∀ n, a_n n = 2 ^ n :=
sorry

end NUMINAMATH_GPT_sequence_geometric_l2319_231957


namespace NUMINAMATH_GPT_vector_c_condition_l2319_231986

variables (a b c : ℝ × ℝ)

def is_perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0
def is_parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ v = (k * w.1, k * w.2)

theorem vector_c_condition (a b c : ℝ × ℝ) 
  (ha : a = (1, 2)) (hb : b = (2, -3)) 
  (hc : c = (7 / 2, -7 / 4)) :
  is_perpendicular c a ∧ is_parallel b (a - c) :=
sorry

end NUMINAMATH_GPT_vector_c_condition_l2319_231986


namespace NUMINAMATH_GPT_parabola_ellipse_tangency_l2319_231985

theorem parabola_ellipse_tangency :
  ∃ (a b : ℝ), (∀ x y, y = x^2 - 5 → (x^2 / a) + (y^2 / b) = 1) →
               (∃ x, y = x^2 - 5 ∧ (x^2 / a) + ((x^2 - 5)^2 / b) = 1) ∧
               a = 1/10 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_parabola_ellipse_tangency_l2319_231985


namespace NUMINAMATH_GPT_debby_soda_bottles_l2319_231932

noncomputable def total_bottles (d t : ℕ) : ℕ := d * t

theorem debby_soda_bottles :
  ∀ (d t: ℕ), d = 9 → t = 40 → total_bottles d t = 360 :=
by
  intros d t h1 h2
  sorry

end NUMINAMATH_GPT_debby_soda_bottles_l2319_231932


namespace NUMINAMATH_GPT_tetrahedron_planes_count_l2319_231927

def tetrahedron_planes : ℕ :=
  let vertices := 4
  let midpoints := 6
  -- The total number of planes calculated by considering different combinations
  4      -- planes formed by three vertices
  + 6    -- planes formed by two vertices and one midpoint
  + 12   -- planes formed by one vertex and two midpoints
  + 7    -- planes formed by three midpoints

theorem tetrahedron_planes_count :
  tetrahedron_planes = 29 :=
by
  sorry

end NUMINAMATH_GPT_tetrahedron_planes_count_l2319_231927


namespace NUMINAMATH_GPT_reflection_eq_l2319_231947

theorem reflection_eq (x y : ℝ) : 
    let line_eq (x y : ℝ) := 2 * x + 3 * y - 5 = 0 
    let reflection_eq (x y : ℝ) := 3 * x + 2 * y - 5 = 0 
    (∀ (x y : ℝ), line_eq x y ↔ reflection_eq y x) →
    reflection_eq x y :=
by
    sorry

end NUMINAMATH_GPT_reflection_eq_l2319_231947


namespace NUMINAMATH_GPT_minimum_value_omega_l2319_231940

variable (f : ℝ → ℝ) (ω ϕ T : ℝ) (x : ℝ)
variable (h_zero : 0 < ω) (h_phi_range : 0 < ϕ ∧ ϕ < π)
variable (h_period : T = 2 * π / ω)
variable (h_f_period : f T = sqrt 3 / 2)
variable (h_zero_of_f : f (π / 9) = 0)
variable (h_f_def : ∀ x, f x = cos (ω * x + ϕ))

theorem minimum_value_omega : ω = 3 := by sorry

end NUMINAMATH_GPT_minimum_value_omega_l2319_231940


namespace NUMINAMATH_GPT_distance_travelled_by_gavril_l2319_231939

noncomputable def smartphoneFullyDischargesInVideoWatching : ℝ := 3
noncomputable def smartphoneFullyDischargesInPlayingTetris : ℝ := 5
noncomputable def speedForHalfDistanceFirst : ℝ := 80
noncomputable def speedForHalfDistanceSecond : ℝ := 60
noncomputable def averageSpeed (distance speed time : ℝ) :=
  distance / time = speed

theorem distance_travelled_by_gavril : 
  ∃ S : ℝ, 
    (∃ t : ℝ, 
      (t / 2 / smartphoneFullyDischargesInVideoWatching + t / 2 / smartphoneFullyDischargesInPlayingTetris = 1) ∧ 
      (S / 2 / t / 2 = speedForHalfDistanceFirst) ∧
      (S / 2 / t / 2 = speedForHalfDistanceSecond)) ∧
     S = 257 := 
sorry

end NUMINAMATH_GPT_distance_travelled_by_gavril_l2319_231939


namespace NUMINAMATH_GPT_hyperbola_t_square_l2319_231904

theorem hyperbola_t_square (t : ℝ)
  (h1 : ∃ a : ℝ, ∀ (x y : ℝ), (y^2 / 4) - (5 * x^2 / 64) = 1 ↔ ((x, y) = (2, t) ∨ (x, y) = (4, -3) ∨ (x, y) = (0, -2))) :
  t^2 = 21 / 4 :=
by
  -- We need to prove t² = 21/4 given the conditions
  sorry

end NUMINAMATH_GPT_hyperbola_t_square_l2319_231904


namespace NUMINAMATH_GPT_inequality_implies_double_l2319_231994

-- Define the condition
variables {x y : ℝ}

theorem inequality_implies_double (h : x < y) : 2 * x < 2 * y :=
  sorry

end NUMINAMATH_GPT_inequality_implies_double_l2319_231994


namespace NUMINAMATH_GPT_new_average_daily_production_l2319_231960

theorem new_average_daily_production 
  (n : ℕ) 
  (avg_past_n_days : ℕ) 
  (today_production : ℕ)
  (new_avg_production : ℕ)
  (hn : n = 5) 
  (havg : avg_past_n_days = 60) 
  (htoday : today_production = 90) 
  (hnew_avg : new_avg_production = 65)
  : (n + 1 = 6) ∧ ((n * 60 + today_production) = 390) ∧ (390 / 6 = 65) :=
by
  sorry

end NUMINAMATH_GPT_new_average_daily_production_l2319_231960


namespace NUMINAMATH_GPT_ava_planted_more_trees_l2319_231926

theorem ava_planted_more_trees (L : ℕ) (h1 : 9 + L = 15) : 9 - L = 3 := 
by
  sorry

end NUMINAMATH_GPT_ava_planted_more_trees_l2319_231926


namespace NUMINAMATH_GPT_incorrect_statement_for_proportional_function_l2319_231918

theorem incorrect_statement_for_proportional_function (x y : ℝ) : y = -5 * x →
  ¬ (∀ x, (x > 0 → y > 0) ∧ (x < 0 → y < 0)) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_for_proportional_function_l2319_231918


namespace NUMINAMATH_GPT_packages_per_truck_l2319_231974

theorem packages_per_truck (total_packages : ℕ) (number_of_trucks : ℕ) (h1 : total_packages = 490) (h2 : number_of_trucks = 7) :
  (total_packages / number_of_trucks) = 70 := by
  sorry

end NUMINAMATH_GPT_packages_per_truck_l2319_231974


namespace NUMINAMATH_GPT_kelly_total_apples_l2319_231975

variable (initial_apples : ℕ) (additional_apples : ℕ)

theorem kelly_total_apples (h1 : initial_apples = 56) (h2 : additional_apples = 49) :
  initial_apples + additional_apples = 105 :=
by
  sorry

end NUMINAMATH_GPT_kelly_total_apples_l2319_231975


namespace NUMINAMATH_GPT_emily_dog_count_l2319_231942

theorem emily_dog_count (dogs : ℕ) 
  (food_per_day_per_dog : ℕ := 250) 
  (vacation_days : ℕ := 14)
  (total_food_kg : ℕ := 14)
  (kg_to_grams : ℕ := 1000) 
  (total_food_grams : ℕ := total_food_kg * kg_to_grams)
  (food_needed_per_dog : ℕ := food_per_day_per_dog * vacation_days) 
  (total_food_needed : ℕ := dogs * food_needed_per_dog) 
  (h : total_food_needed = total_food_grams) : 
  dogs = 4 := 
sorry

end NUMINAMATH_GPT_emily_dog_count_l2319_231942


namespace NUMINAMATH_GPT_convert_13_to_binary_l2319_231972

def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (n : Nat) (acc : List Nat) : List Nat :=
      if n = 0 then acc
      else aux (n / 2) ((n % 2) :: acc)
    aux n []

theorem convert_13_to_binary : decimal_to_binary 13 = [1, 1, 0, 1] :=
  by
    sorry -- Proof to be provided

end NUMINAMATH_GPT_convert_13_to_binary_l2319_231972


namespace NUMINAMATH_GPT_cos_135_eq_neg_sqrt2_div_2_sin_135_eq_sqrt2_div_2_l2319_231907

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by sorry

theorem sin_135_eq_sqrt2_div_2 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_GPT_cos_135_eq_neg_sqrt2_div_2_sin_135_eq_sqrt2_div_2_l2319_231907


namespace NUMINAMATH_GPT_star_operation_l2319_231945

def new_op (a b : ℝ) : ℝ :=
  a^2 + b^2 - a * b

theorem star_operation (x y : ℝ) : 
  new_op (x + 2 * y) (y + 3 * x) = 7 * x^2 + 3 * y^2 + 3 * (x * y) :=
by
  sorry

end NUMINAMATH_GPT_star_operation_l2319_231945


namespace NUMINAMATH_GPT_square_side_measurement_error_l2319_231946

theorem square_side_measurement_error {S S' : ℝ} (h1 : S' = S * Real.sqrt 1.0816) :
  ((S' - S) / S) * 100 = 4 := by
  sorry

end NUMINAMATH_GPT_square_side_measurement_error_l2319_231946


namespace NUMINAMATH_GPT_only_A_can_form_triangle_l2319_231929

/--
Prove that from the given sets of lengths, only the set {5cm, 8cm, 12cm} can form a valid triangle.

Given:
- A: 5 cm, 8 cm, 12 cm
- B: 2 cm, 3 cm, 6 cm
- C: 3 cm, 3 cm, 6 cm
- D: 4 cm, 7 cm, 11 cm

We need to show that only Set A satisfies the triangle inequality theorem.
-/
theorem only_A_can_form_triangle :
  (∀ (a b c : ℕ), a = 5 ∧ b = 8 ∧ c = 12 → a + b > c ∧ a + c > b ∧ b + c > a) ∧
  (∀ (a b c : ℕ), a = 2 ∧ b = 3 ∧ c = 6 → ¬(a + b > c ∧ a + c > b ∧ b + c > a)) ∧
  (∀ (a b c : ℕ), a = 3 ∧ b = 3 ∧ c = 6 → ¬(a + b > c ∧ a + c > b ∧ b + c > a)) ∧
  (∀ (a b c : ℕ), a = 4 ∧ b = 7 ∧ c = 11 → ¬(a + b > c ∧ a + c > b ∧ b + c > a)) :=
by
  sorry -- Proof to be provided

end NUMINAMATH_GPT_only_A_can_form_triangle_l2319_231929


namespace NUMINAMATH_GPT_car_a_distance_behind_car_b_l2319_231988

theorem car_a_distance_behind_car_b :
  ∃ D : ℝ, D = 40 ∧ 
    (∀ (t : ℝ), t = 4 →
    ((58 - 50) * t + 8) = D + 8)
  := by
  sorry

end NUMINAMATH_GPT_car_a_distance_behind_car_b_l2319_231988


namespace NUMINAMATH_GPT_stickers_given_to_sister_l2319_231949

variable (initial bought birthday used left given : ℕ)

theorem stickers_given_to_sister :
  (initial = 20) →
  (bought = 12) →
  (birthday = 20) →
  (used = 8) →
  (left = 39) →
  (given = (initial + bought + birthday - used - left)) →
  given = 5 := by
  intros
  sorry

end NUMINAMATH_GPT_stickers_given_to_sister_l2319_231949


namespace NUMINAMATH_GPT_line_equation_l2319_231997

-- Define the structure of a point
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define the projection condition
def projection_condition (P : Point) (l : ℤ → ℤ → Prop) : Prop :=
  l P.x P.y ∧ ∀ (Q : Point), l Q.x Q.y → (Q.x ^ 2 + Q.y ^ 2) ≥ (P.x ^ 2 + P.y ^ 2)

-- Define the point P(-2, 1)
def P : Point := ⟨ -2, 1 ⟩

-- Define line l
def line_l (x y : ℤ) : Prop := 2 * x - y + 5 = 0

-- Theorem statement
theorem line_equation :
  projection_condition P line_l → ∀ (x y : ℤ), line_l x y ↔ 2 * x - y + 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_equation_l2319_231997


namespace NUMINAMATH_GPT_tan_alpha_implies_fraction_l2319_231903

theorem tan_alpha_implies_fraction (α : ℝ) (h : Real.tan α = -3/2) : 
  (Real.sin α + 2 * Real.cos α) / (Real.cos α - Real.sin α) = 1 / 5 := 
sorry

end NUMINAMATH_GPT_tan_alpha_implies_fraction_l2319_231903


namespace NUMINAMATH_GPT_range_of_a_l2319_231931

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = Real.exp x) :
  (∀ x : ℝ, f x ≥ Real.exp x + a) ↔ a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2319_231931


namespace NUMINAMATH_GPT_mirror_side_length_l2319_231948

theorem mirror_side_length
  (width_wall : ℝ)
  (length_wall : ℝ)
  (area_wall : ℝ)
  (area_mirror : ℝ)
  (side_length_mirror : ℝ)
  (h1 : width_wall = 32)
  (h2 : length_wall = 20.25)
  (h3 : area_wall = width_wall * length_wall)
  (h4 : area_mirror = area_wall / 2)
  (h5 : side_length_mirror * side_length_mirror = area_mirror)
  : side_length_mirror = 18 := by
  sorry

end NUMINAMATH_GPT_mirror_side_length_l2319_231948


namespace NUMINAMATH_GPT_american_summits_more_water_l2319_231978

-- Definitions based on the conditions
def FosterFarmsChickens := 45
def AmericanSummitsWater := 2 * FosterFarmsChickens
def HormelChickens := 3 * FosterFarmsChickens
def BoudinButchersChickens := HormelChickens / 3
def TotalItems := 375
def ItemsByFourCompanies := FosterFarmsChickens + AmericanSummitsWater + HormelChickens + BoudinButchersChickens
def DelMonteWater := TotalItems - ItemsByFourCompanies
def WaterDifference := AmericanSummitsWater - DelMonteWater

theorem american_summits_more_water : WaterDifference = 30 := by
  sorry

end NUMINAMATH_GPT_american_summits_more_water_l2319_231978


namespace NUMINAMATH_GPT_linear_system_solution_l2319_231917

theorem linear_system_solution (a b : ℝ) 
  (h1 : 3 * a + 2 * b = 5) 
  (h2 : 2 * a + 3 * b = 4) : 
  a - b = 1 := 
by
  sorry

end NUMINAMATH_GPT_linear_system_solution_l2319_231917


namespace NUMINAMATH_GPT_minimum_sides_of_polygon_l2319_231936

theorem minimum_sides_of_polygon (θ : ℝ) (hθ : θ = 25.5) : ∃ n : ℕ, n = 240 ∧ ∀ k : ℕ, (k * θ) % 360 = 0 → k = n := 
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_minimum_sides_of_polygon_l2319_231936


namespace NUMINAMATH_GPT_smallest_period_pi_max_value_min_value_l2319_231971

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2 * x)

open Real

theorem smallest_period_pi : ∀ x, f (x + π) = f x := by
  unfold f
  intros
  sorry

theorem max_value : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f x ≤ 1 + sqrt 2 := by
  unfold f
  intros
  sorry

theorem min_value : ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f x ≥ 0 := by
  unfold f
  intros
  sorry

end NUMINAMATH_GPT_smallest_period_pi_max_value_min_value_l2319_231971


namespace NUMINAMATH_GPT_freshmen_and_sophomores_without_pet_l2319_231984

theorem freshmen_and_sophomores_without_pet (total_students : ℕ) 
                                             (freshmen_sophomores_percent : ℕ)
                                             (pet_ownership_fraction : ℕ)
                                             (h_total : total_students = 400)
                                             (h_percent : freshmen_sophomores_percent = 50)
                                             (h_fraction : pet_ownership_fraction = 5) : 
                                             (total_students * freshmen_sophomores_percent / 100 - 
                                             total_students * freshmen_sophomores_percent / 100 / pet_ownership_fraction) = 160 :=
by
  sorry

end NUMINAMATH_GPT_freshmen_and_sophomores_without_pet_l2319_231984


namespace NUMINAMATH_GPT_janice_homework_time_l2319_231976

variable (H : ℝ)
variable (cleaning_room walk_dog take_trash : ℝ)

-- Conditions from the problem translated directly
def cleaning_room_time : cleaning_room = H / 2 := sorry
def walk_dog_time : walk_dog = H + 5 := sorry
def take_trash_time : take_trash = H / 6 := sorry
def total_time_before_movie : 35 + (H + cleaning_room + walk_dog + take_trash) = 120 := sorry

-- The main theorem to prove
theorem janice_homework_time (H : ℝ)
        (cleaning_room : ℝ := H / 2)
        (walk_dog : ℝ := H + 5)
        (take_trash : ℝ := H / 6) :
    H + cleaning_room + walk_dog + take_trash + 35 = 120 → H = 30 :=
by
  sorry

end NUMINAMATH_GPT_janice_homework_time_l2319_231976


namespace NUMINAMATH_GPT_ratio_perimeter_triangle_square_l2319_231935

/-
  Suppose a square piece of paper with side length 4 units is folded in half diagonally.
  The folded paper is then cut along the fold, producing two right-angled triangles.
  We need to prove that the ratio of the perimeter of one of the triangles to the perimeter of the original square is (1/2) + (sqrt 2 / 4).
-/
theorem ratio_perimeter_triangle_square:
  let side_length := 4
  let triangle_leg := side_length
  let hypotenuse := Real.sqrt (triangle_leg ^ 2 + triangle_leg ^ 2)
  let perimeter_triangle := triangle_leg + triangle_leg + hypotenuse
  let perimeter_square := 4 * side_length
  let ratio := perimeter_triangle / perimeter_square
  ratio = (1 / 2) + (Real.sqrt 2 / 4) :=
by
  sorry

end NUMINAMATH_GPT_ratio_perimeter_triangle_square_l2319_231935


namespace NUMINAMATH_GPT_min_value_fraction_sum_l2319_231987

theorem min_value_fraction_sum (m n : ℝ) (h₁ : 0 < m) (h₂ : 0 < n) (h₃ : 2 * m + n = 1) : 
  (1 / m + 2 / n) ≥ 8 :=
sorry

end NUMINAMATH_GPT_min_value_fraction_sum_l2319_231987


namespace NUMINAMATH_GPT_cost_price_A_l2319_231943

-- Establishing the definitions based on the conditions from a)

def profit_A_to_B (CP_A : ℝ) : ℝ := 1.20 * CP_A
def profit_B_to_C (CP_B : ℝ) : ℝ := 1.25 * CP_B
def price_paid_by_C : ℝ := 222

-- Stating the theorem to be proven:
theorem cost_price_A (CP_A : ℝ) (H : profit_B_to_C (profit_A_to_B CP_A) = price_paid_by_C) : CP_A = 148 :=
by 
  sorry

end NUMINAMATH_GPT_cost_price_A_l2319_231943


namespace NUMINAMATH_GPT_B_days_to_complete_work_l2319_231901

theorem B_days_to_complete_work (A_days : ℕ) (efficiency_less_percent : ℕ) 
  (hA : A_days = 12) (hB_efficiency : efficiency_less_percent = 20) :
  let A_work_rate := 1 / 12
  let B_work_rate := (1 - (20 / 100)) * A_work_rate
  let B_days := 1 / B_work_rate
  B_days = 15 :=
by
  sorry

end NUMINAMATH_GPT_B_days_to_complete_work_l2319_231901


namespace NUMINAMATH_GPT_solution1_solution2_l2319_231923

open Real

noncomputable def problem1 (a b : ℝ) : Prop :=
a = 2 ∧ b = 2

noncomputable def problem2 (b : ℝ) : Prop :=
b = (2 * (sqrt 3 + sqrt 2)) / 3

theorem solution1 (a b : ℝ) (c : ℝ) (C : ℝ) (area : ℝ)
  (h1 : c = 2)
  (h2 : C = π / 3)
  (h3 : area = sqrt 3)
  (h4 : (1 / 2) * a * b * sin C = area) :
  problem1 a b :=
by sorry

theorem solution2 (a b : ℝ) (c : ℝ) (C : ℝ) (cosA : ℝ)
  (h1 : c = 2)
  (h2 : C = π / 3)
  (h3 : cosA = sqrt 3 / 3)
  (h4 : sin (arccos (sqrt 3 / 3)) = sqrt 6 / 3)
  (h5 : (a / (sqrt 6 / 3)) = (2 / (sqrt 3 / 2)))
  (h6 : ((b / ((3 + sqrt 6) / 6)) = (2 / (sqrt 3 / 2)))) :
  problem2 b :=
by sorry

end NUMINAMATH_GPT_solution1_solution2_l2319_231923


namespace NUMINAMATH_GPT_max_value_6a_3b_10c_l2319_231961

theorem max_value_6a_3b_10c (a b c : ℝ) (h : 9 * a ^ 2 + 4 * b ^ 2 + 25 * c ^ 2 = 1) : 
  6 * a + 3 * b + 10 * c ≤ (Real.sqrt 41) / 2 :=
sorry

end NUMINAMATH_GPT_max_value_6a_3b_10c_l2319_231961


namespace NUMINAMATH_GPT_large_circuit_longer_l2319_231981

theorem large_circuit_longer :
  ∀ (small_circuit_length large_circuit_length : ℕ),
  ∀ (laps_jana laps_father : ℕ),
  laps_jana = 3 →
  laps_father = 4 →
  (laps_father * large_circuit_length = 2 * (laps_jana * small_circuit_length)) →
  small_circuit_length = 400 →
  large_circuit_length - small_circuit_length = 200 :=
by
  intros small_circuit_length large_circuit_length laps_jana laps_father
  intros h_jana_laps h_father_laps h_distance h_small_length
  sorry

end NUMINAMATH_GPT_large_circuit_longer_l2319_231981


namespace NUMINAMATH_GPT_fifth_friend_payment_l2319_231933

def contributions (a b c d e : ℕ) : Prop :=
  a + b + c + d + e = 120 ∧
  a = (1 / 3 : ℕ) * (b + c + d + e) ∧
  b = (1 / 4 : ℕ) * (a + c + d + e) ∧
  c = (1 / 5 : ℕ) * (a + b + d + e)

theorem fifth_friend_payment (a b c d e : ℕ) (h : contributions a b c d e) : e = 13 :=
sorry

end NUMINAMATH_GPT_fifth_friend_payment_l2319_231933


namespace NUMINAMATH_GPT_number_of_solutions_l2319_231992

def f (x : ℝ) : ℝ := |1 - 2 * x|

theorem number_of_solutions :
  (∃ n : ℕ, n = 8 ∧ ∀ x ∈ [0,1], f (f (f x)) = (1 / 2) * x) :=
sorry

end NUMINAMATH_GPT_number_of_solutions_l2319_231992


namespace NUMINAMATH_GPT_canned_food_total_bins_l2319_231916

theorem canned_food_total_bins :
  let soup_bins := 0.125
  let vegetable_bins := 0.125
  let pasta_bins := 0.5
  soup_bins + vegetable_bins + pasta_bins = 0.75 := 
by
  sorry

end NUMINAMATH_GPT_canned_food_total_bins_l2319_231916


namespace NUMINAMATH_GPT_find_number_l2319_231911

theorem find_number:
  ∃ x : ℝ, x + 1.35 + 0.123 = 1.794 ∧ x = 0.321 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2319_231911


namespace NUMINAMATH_GPT_domain_of_sqrt_cos_function_l2319_231995

theorem domain_of_sqrt_cos_function:
  (∀ k : ℤ, ∀ x : ℝ, 2 * Real.cos x + 1 ≥ 0 ↔ x ∈ Set.Icc (2 * k * Real.pi - 2 * Real.pi / 3) (2 * k * Real.pi + 2 * Real.pi / 3)) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_sqrt_cos_function_l2319_231995


namespace NUMINAMATH_GPT_train_combined_distance_l2319_231998

/-- Prove that the combined distance covered by three trains is 3480 km,
    given their respective speeds and travel times. -/
theorem train_combined_distance : 
  let speed_A := 150 -- Speed of Train A in km/h
  let time_A := 8     -- Time Train A travels in hours
  let speed_B := 180 -- Speed of Train B in km/h
  let time_B := 6     -- Time Train B travels in hours
  let speed_C := 120 -- Speed of Train C in km/h
  let time_C := 10    -- Time Train C travels in hours
  let distance_A := speed_A * time_A -- Distance covered by Train A
  let distance_B := speed_B * time_B -- Distance covered by Train B
  let distance_C := speed_C * time_C -- Distance covered by Train C
  let combined_distance := distance_A + distance_B + distance_C -- Combined distance covered by all trains
  combined_distance = 3480 :=
by
  sorry

end NUMINAMATH_GPT_train_combined_distance_l2319_231998


namespace NUMINAMATH_GPT_no_real_solution_l2319_231910

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 6

-- Lean statement: prove that the equation x^2 - 4x + 6 = 0 has no real solution
theorem no_real_solution : ¬ ∃ x : ℝ, f x = 0 :=
sorry

end NUMINAMATH_GPT_no_real_solution_l2319_231910


namespace NUMINAMATH_GPT_sum_inequality_l2319_231928

open Real

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ (a * b / (a + b)) + (b * c / (b + c)) + (c * a / (c + a)) + 
             (1 / 2) * ((a * b / c) + (b * c / a) + (c * a / b)) :=
by
  sorry

end NUMINAMATH_GPT_sum_inequality_l2319_231928


namespace NUMINAMATH_GPT_ratio_of_volumes_l2319_231951

def cone_radius_X := 10
def cone_height_X := 15
def cone_radius_Y := 15
def cone_height_Y := 10

noncomputable def volume_cone (r h : ℝ) := (1 / 3) * Real.pi * r^2 * h

noncomputable def volume_X := volume_cone cone_radius_X cone_height_X
noncomputable def volume_Y := volume_cone cone_radius_Y cone_height_Y

theorem ratio_of_volumes : volume_X / volume_Y = 2 / 3 := sorry

end NUMINAMATH_GPT_ratio_of_volumes_l2319_231951


namespace NUMINAMATH_GPT_arithmetic_sequence_n_equals_8_l2319_231914

theorem arithmetic_sequence_n_equals_8
  (a : ℕ → ℝ) 
  (h_arith : ∀ n m, a (n + 1) - a n = a (m + 1) - a m) 
  (h2 : a 2 + a 5 = 18)
  (h3 : a 3 * a 4 = 32)
  (h_n : ∃ n, a n = 128) :
  ∃ n, a n = 128 ∧ n = 8 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_n_equals_8_l2319_231914


namespace NUMINAMATH_GPT_halfway_fraction_l2319_231962

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/7) : (a + b) / 2 = 41/56 :=
by
  sorry

end NUMINAMATH_GPT_halfway_fraction_l2319_231962


namespace NUMINAMATH_GPT_fraction_expression_equiv_l2319_231970

theorem fraction_expression_equiv:
  ((5 / 2) / (1 / 2) * (5 / 2)) / ((5 / 2) * (1 / 2) / (5 / 2)) = 25 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_expression_equiv_l2319_231970


namespace NUMINAMATH_GPT_geometric_series_q_and_S6_l2319_231955

theorem geometric_series_q_and_S6 (a : ℕ → ℝ) (q : ℝ) (S_6 : ℝ) 
  (ha_pos : ∀ n, a n > 0)
  (ha2 : a 2 = 3)
  (ha4 : a 4 = 27) :
  q = 3 ∧ S_6 = 364 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_q_and_S6_l2319_231955


namespace NUMINAMATH_GPT_fraction_of_orange_juice_correct_l2319_231944

-- Define the capacities of the pitchers
def capacity := 800

-- Define the fractions of orange juice and apple juice in the first pitcher
def orangeJuiceFraction1 := 1 / 4
def appleJuiceFraction1 := 1 / 8

-- Define the fractions of orange juice and apple juice in the second pitcher
def orangeJuiceFraction2 := 1 / 5
def appleJuiceFraction2 := 1 / 10

-- Define the total volumes of the contents in each pitcher
def totalVolume := 2 * capacity -- total volume in the large container after pouring

-- Define the orange juice volumes in each pitcher
def orangeJuiceVolume1 := orangeJuiceFraction1 * capacity
def orangeJuiceVolume2 := orangeJuiceFraction2 * capacity

-- Calculate the total volume of orange juice in the large container
def totalOrangeJuiceVolume := orangeJuiceVolume1 + orangeJuiceVolume2

-- Define the fraction of orange juice in the large container
def orangeJuiceFraction := totalOrangeJuiceVolume / totalVolume

theorem fraction_of_orange_juice_correct :
  orangeJuiceFraction = 9 / 40 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_orange_juice_correct_l2319_231944


namespace NUMINAMATH_GPT_find_y_l2319_231922

theorem find_y (x y : ℤ) (h₁ : x ^ 2 + x + 4 = y - 4) (h₂ : x = 3) : y = 20 :=
by 
  sorry

end NUMINAMATH_GPT_find_y_l2319_231922


namespace NUMINAMATH_GPT_books_loaned_out_l2319_231953

/-- 
Given:
- There are 75 books in a special collection at the beginning of the month.
- By the end of the month, 70 percent of books that were loaned out are returned.
- There are 60 books in the special collection at the end of the month.
Prove:
- The number of books loaned out during the month is 50.
-/
theorem books_loaned_out (x : ℝ) (h1 : 75 - 0.3 * x = 60) : x = 50 :=
by
  sorry

end NUMINAMATH_GPT_books_loaned_out_l2319_231953


namespace NUMINAMATH_GPT_tangent_line_at_x_2_increasing_on_1_to_infinity_l2319_231973

noncomputable def f (a x : ℝ) : ℝ := (1/2) * x^2 + a * Real.log x

-- Subpart I
theorem tangent_line_at_x_2 (a b : ℝ) :
  (a / 2 + 2 = 1) ∧ (2 + a * Real.log 2 = 2 + b) → (a = -2 ∧ b = -2 * Real.log 2) :=
by
  sorry

-- Subpart II
theorem increasing_on_1_to_infinity (a : ℝ) :
  (∀ x > 1, (x + a / x) ≥ 0) → (a ≥ -1) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_x_2_increasing_on_1_to_infinity_l2319_231973


namespace NUMINAMATH_GPT_prime_ge_7_not_divisible_by_40_l2319_231958

theorem prime_ge_7_not_divisible_by_40 (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_7 : p ≥ 7) : ¬ (40 ∣ (p^3 - 1)) :=
sorry

end NUMINAMATH_GPT_prime_ge_7_not_divisible_by_40_l2319_231958


namespace NUMINAMATH_GPT_count_numbers_1000_to_5000_l2319_231993

def countFourDigitNumbersInRange (lower upper : ℕ) : ℕ :=
  if lower <= upper then upper - lower + 1 else 0

theorem count_numbers_1000_to_5000 : countFourDigitNumbersInRange 1000 5000 = 4001 :=
by
  sorry

end NUMINAMATH_GPT_count_numbers_1000_to_5000_l2319_231993


namespace NUMINAMATH_GPT_part1_monotonicity_part2_minimum_range_l2319_231912

noncomputable def f (k x : ℝ) : ℝ := (k + x) / (x - 1) * Real.log x

theorem part1_monotonicity (x : ℝ) (h : x ≠ 1) :
    k = 0 → f k x = (x / (x - 1)) * Real.log x ∧ 
    (0 < x ∧ x < 1 ∨ 1 < x) → Monotone (f k) :=
sorry

theorem part2_minimum_range (k : ℝ) :
    (∃ x ∈ Set.Ioi 1, IsLocalMin (f k) x) ↔ k ∈ Set.Ioi 1 :=
sorry

end NUMINAMATH_GPT_part1_monotonicity_part2_minimum_range_l2319_231912


namespace NUMINAMATH_GPT_b7_value_l2319_231983

theorem b7_value (a : ℕ → ℚ) (b : ℕ → ℚ)
  (h₀a : a 0 = 3) (h₀b : b 0 = 4)
  (h₁ : ∀ n, a (n + 1) = a n ^ 2 / b n)
  (h₂ : ∀ n, b (n + 1) = b n ^ 2 / a n) :
  b 7 = 4 ^ 730 / 3 ^ 1093 :=
by
  sorry

end NUMINAMATH_GPT_b7_value_l2319_231983


namespace NUMINAMATH_GPT_eden_bears_count_l2319_231915

-- Define the main hypothesis
def initial_bears : Nat := 20
def favorite_bears : Nat := 8
def remaining_bears := initial_bears - favorite_bears

def number_of_sisters : Nat := 3
def bears_per_sister := remaining_bears / number_of_sisters

def eden_initial_bears : Nat := 10
def eden_final_bears := eden_initial_bears + bears_per_sister

theorem eden_bears_count : eden_final_bears = 14 :=
by
  unfold eden_final_bears eden_initial_bears bears_per_sister remaining_bears initial_bears favorite_bears
  norm_num
  sorry

end NUMINAMATH_GPT_eden_bears_count_l2319_231915


namespace NUMINAMATH_GPT_divisor_of_number_l2319_231990

theorem divisor_of_number : 
  ∃ D, 
    let x := 75 
    let R' := 7 
    let Q := R' + 8 
    x = D * Q + 0 :=
by
  sorry

end NUMINAMATH_GPT_divisor_of_number_l2319_231990


namespace NUMINAMATH_GPT_binom_coefficient_largest_l2319_231952

theorem binom_coefficient_largest (n : ℕ) (h : (n / 2) + 1 = 7) : n = 12 :=
by
  sorry

end NUMINAMATH_GPT_binom_coefficient_largest_l2319_231952
