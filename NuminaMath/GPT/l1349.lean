import Mathlib

namespace new_person_weight_l1349_134941

theorem new_person_weight :
  (8 * 2.5 + 75 = 95) :=
by sorry

end new_person_weight_l1349_134941


namespace spoon_less_than_fork_l1349_134903

-- Define the initial price of spoon and fork in kopecks
def initial_price (x : ℕ) : Prop :=
  x > 100 -- ensuring the spoon's sale price remains positive

-- Define the sale price of the spoon
def spoon_sale_price (x : ℕ) : ℕ :=
  x - 100

-- Define the sale price of the fork
def fork_sale_price (x : ℕ) : ℕ :=
  x / 10

-- Prove that the spoon's sale price can be less than the fork's sale price
theorem spoon_less_than_fork (x : ℕ) (h : initial_price x) : 
  spoon_sale_price x < fork_sale_price x :=
by
  sorry

end spoon_less_than_fork_l1349_134903


namespace intersecting_point_value_l1349_134956

theorem intersecting_point_value
  (b a : ℤ)
  (h1 : a = -2 * 2 + b)
  (h2 : 2 = -2 * a + b) :
  a = 2 :=
by
  sorry

end intersecting_point_value_l1349_134956


namespace weekend_price_is_correct_l1349_134935

-- Define the original price of the jacket
def original_price : ℝ := 250

-- Define the first discount rate (40%)
def first_discount_rate : ℝ := 0.40

-- Define the additional weekend discount rate (10%)
def additional_discount_rate : ℝ := 0.10

-- Define a function to apply the first discount
def apply_first_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

-- Define a function to apply the additional discount
def apply_additional_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

-- Using both discounts, calculate the final weekend price
def weekend_price : ℝ :=
  apply_additional_discount (apply_first_discount original_price first_discount_rate) additional_discount_rate

-- The final theorem stating the expected weekend price is $135
theorem weekend_price_is_correct : weekend_price = 135 := by
  sorry

end weekend_price_is_correct_l1349_134935


namespace find_range_of_a_l1349_134916

noncomputable def range_of_a : Set ℝ :=
  {a | (∀ x : ℝ, x^2 - 2 * x > a) ∨ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0)}

theorem find_range_of_a :
  {a : ℝ | (∀ x : ℝ, x^2 - 2 * x > a) ∨ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0)} = 
  {a | (-2 < a ∧ a < -1) ∨ (1 ≤ a)} :=
by
  sorry

end find_range_of_a_l1349_134916


namespace geometric_sequence_angle_count_l1349_134953

theorem geometric_sequence_angle_count :
  (∃ θs : Finset ℝ, (∀ θ ∈ θs, 0 < θ ∧ θ < 2 * π ∧ ¬ ∃ k : ℕ, θ = k * (π / 2)) 
                    ∧ θs.card = 4
                    ∧ ∀ θ ∈ θs, ∃ a b c : ℝ, (a, b, c) = (Real.sin θ, Real.cos θ, Real.tan θ) 
                                             ∨ (a, b) = (Real.sin θ, Real.tan θ) 
                                             ∨ (a, b) = (Real.cos θ, Real.tan θ)
                                             ∧ b = a * c) :=
sorry

end geometric_sequence_angle_count_l1349_134953


namespace average_in_all_6_subjects_l1349_134955

-- Definitions of the conditions
def average_in_5_subjects : ℝ := 74
def marks_in_6th_subject : ℝ := 104
def num_subjects_total : ℝ := 6

-- Proof that the average in all 6 subjects is 79
theorem average_in_all_6_subjects :
  (average_in_5_subjects * 5 + marks_in_6th_subject) / num_subjects_total = 79 := by
  sorry

end average_in_all_6_subjects_l1349_134955


namespace solution_is_correct_l1349_134977

def valid_triple (a b c : ℕ) : Prop :=
  (Nat.gcd a 20 = b) ∧ (Nat.gcd b 15 = c) ∧ (Nat.gcd a c = 5)

def is_solution_set (triples : Set (ℕ × ℕ × ℕ)) : Prop :=
  ∀ a b c, (a, b, c) ∈ triples ↔ 
    (valid_triple a b c) ∧ 
    ((∃ k, a = 20 * k ∧ b = 20 ∧ c = 5) ∨
    (∃ k, a = 20 * k - 10 ∧ b = 10 ∧ c = 5) ∨
    (∃ k, a = 10 * k - 5 ∧ b = 5 ∧ c = 5))

theorem solution_is_correct : ∃ S, is_solution_set S :=
sorry

end solution_is_correct_l1349_134977


namespace math_problem_l1349_134937

theorem math_problem (n d : ℕ) (h1 : 0 < n) (h2 : d < 10)
  (h3 : 3 * n^2 + 2 * n + d = 263)
  (h4 : 3 * n^2 + 2 * n + 4 = 396 + 7 * d) :
  n + d = 11 :=
by {
  sorry
}

end math_problem_l1349_134937


namespace chord_midpoint_line_eqn_l1349_134986

-- Definitions of points and the ellipse condition
def P : ℝ × ℝ := (3, 2)

def is_midpoint (P E F : ℝ × ℝ) := 
  P.1 = (E.1 + F.1) / 2 ∧ P.2 = (E.2 + F.2) / 2

def ellipse (x y : ℝ) := 
  4 * x^2 + 9 * y^2 = 144

theorem chord_midpoint_line_eqn
  (E F : ℝ × ℝ) 
  (h1 : is_midpoint P E F)
  (h2 : ellipse E.1 E.2)
  (h3 : ellipse F.1 F.2):
  ∃ (m b : ℝ), (P.2 = m * P.1 + b) ∧ (2 * P.1 + 3 * P.2 - 12 = 0) :=
by 
  sorry

end chord_midpoint_line_eqn_l1349_134986


namespace count_remainders_gte_l1349_134992

def remainder (a N : ℕ) : ℕ := a % N

theorem count_remainders_gte (N : ℕ) : 
  (∀ a, a > 0 → remainder a 1000 > remainder a 1001 → N ≤ 1000000) →
  N = 499500 :=
by
  sorry

end count_remainders_gte_l1349_134992


namespace union_M_N_l1349_134907

-- Definitions based on conditions
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {x | ∃ a, a ∈ M ∧ x = 2 * a}

-- The theorem to be proven
theorem union_M_N : M ∪ N = {0, 1, 2, 4} := by
  sorry

end union_M_N_l1349_134907


namespace minimum_m_plus_n_l1349_134942

theorem minimum_m_plus_n
  (m n : ℝ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h_ellipse : 1 / m + 4 / n = 1) :
  m + n = 9 :=
sorry

end minimum_m_plus_n_l1349_134942


namespace find_i_value_for_S_i_l1349_134928

theorem find_i_value_for_S_i :
  ∃ (i : ℕ), (3 * 6 - 2 ≤ i ∧ i < 3 * 6 + 1) ∧ (1000 ≤ 31 * 2^6) ∧ (31 * 2^6 ≤ 3000) ∧ i = 2 :=
by sorry

end find_i_value_for_S_i_l1349_134928


namespace initial_average_weight_l1349_134972

theorem initial_average_weight
  (A : ℝ)
  (h : 30 * 27.4 - 10 = 29 * A) : 
  A = 28 := 
by
  sorry

end initial_average_weight_l1349_134972


namespace visitors_that_day_l1349_134990

theorem visitors_that_day (total_visitors : ℕ) (previous_day_visitors : ℕ) 
  (h_total : total_visitors = 406) (h_previous : previous_day_visitors = 274) : 
  total_visitors - previous_day_visitors = 132 :=
by
  sorry

end visitors_that_day_l1349_134990


namespace find_c_eq_neg_9_over_4_l1349_134934

theorem find_c_eq_neg_9_over_4 (c x : ℚ) (h₁ : 3 * x + 5 = 1) (h₂ : c * x - 8 = -5) :
  c = -9 / 4 :=
sorry

end find_c_eq_neg_9_over_4_l1349_134934


namespace product_of_binomials_l1349_134970

theorem product_of_binomials (x : ℝ) : 
  (4 * x - 3) * (2 * x + 7) = 8 * x^2 + 22 * x - 21 := by
  sorry

end product_of_binomials_l1349_134970


namespace gina_order_rose_cups_l1349_134976

theorem gina_order_rose_cups 
  (rose_cups_per_hour : ℕ) 
  (lily_cups_per_hour : ℕ) 
  (total_lily_cups_order : ℕ) 
  (total_pay : ℕ) 
  (pay_per_hour : ℕ) 
  (total_hours_worked : ℕ) 
  (hours_spent_with_lilies : ℕ)
  (hours_spent_with_roses : ℕ) 
  (rose_cups_order : ℕ) :
  rose_cups_per_hour = 6 →
  lily_cups_per_hour = 7 →
  total_lily_cups_order = 14 →
  total_pay = 90 →
  pay_per_hour = 30 →
  total_hours_worked = total_pay / pay_per_hour →
  hours_spent_with_lilies = total_lily_cups_order / lily_cups_per_hour →
  hours_spent_with_roses = total_hours_worked - hours_spent_with_lilies →
  rose_cups_order = rose_cups_per_hour * hours_spent_with_roses →
  rose_cups_order = 6 := 
by
  sorry

end gina_order_rose_cups_l1349_134976


namespace determine_lunch_break_duration_lunch_break_duration_in_minutes_l1349_134958

noncomputable def painter_lunch_break_duration (j h L : ℝ) : Prop :=
  (10 - L) * (j + h) = 0.6 ∧
  (8 - L) * h = 0.3 ∧
  (5 - L) * j = 0.1

theorem determine_lunch_break_duration (j h : ℝ) :
  ∃ L : ℝ, painter_lunch_break_duration j h L ∧ L = 0.8 :=
by sorry

theorem lunch_break_duration_in_minutes (j h : ℝ) :
  ∃ L : ℝ, painter_lunch_break_duration j h L ∧ L * 60 = 48 :=
by sorry

end determine_lunch_break_duration_lunch_break_duration_in_minutes_l1349_134958


namespace total_number_of_wheels_l1349_134984

-- Define the conditions as hypotheses
def cars := 2
def wheels_per_car := 4

def bikes := 2
def trashcans := 1
def wheels_per_bike_or_trashcan := 2

def roller_skates_pair := 1
def wheels_per_skate := 4

def tricycle := 1
def wheels_per_tricycle := 3

-- Prove the total number of wheels
theorem total_number_of_wheels :
  cars * wheels_per_car +
  (bikes + trashcans) * wheels_per_bike_or_trashcan +
  (roller_skates_pair * 2) * wheels_per_skate +
  tricycle * wheels_per_tricycle 
  = 25 :=
by
  sorry

end total_number_of_wheels_l1349_134984


namespace prime_count_60_to_70_l1349_134914

theorem prime_count_60_to_70 : ∃ primes : Finset ℕ, primes.card = 2 ∧ ∀ p ∈ primes, 60 < p ∧ p < 70 ∧ Nat.Prime p :=
by
  sorry

end prime_count_60_to_70_l1349_134914


namespace correct_calculation_l1349_134994

theorem correct_calculation :
  (3 * Real.sqrt 2) * (2 * Real.sqrt 3) = 6 * Real.sqrt 6 :=
by sorry

end correct_calculation_l1349_134994


namespace find_f_l1349_134995

theorem find_f (f : ℕ → ℕ) :
  (∀ a b c : ℕ, ((f a + f b + f c) - a * b - b * c - c * a) ∣ (a * f a + b * f b + c * f c - 3 * a * b * c)) →
  (∀ n : ℕ, f n = n * n) :=
sorry

end find_f_l1349_134995


namespace cylinder_height_comparison_l1349_134967

theorem cylinder_height_comparison (r1 h1 r2 h2 : ℝ)
  (volume_eq : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_relation : r2 = 1.2 * r1) :
  h1 = 1.44 * h2 :=
by {
  -- Proof steps here, not required per instruction
  sorry
}

end cylinder_height_comparison_l1349_134967


namespace expand_expression_l1349_134945

theorem expand_expression (y z : ℝ) : 
  -2 * (5 * y^3 - 3 * y^2 * z + 4 * y * z^2 - z^3) = -10 * y^3 + 6 * y^2 * z - 8 * y * z^2 + 2 * z^3 :=
by sorry

end expand_expression_l1349_134945


namespace original_price_of_cycle_l1349_134905

theorem original_price_of_cycle (P : ℝ) (h1 : 1440 = P + 0.6 * P) : P = 900 :=
by
  sorry

end original_price_of_cycle_l1349_134905


namespace regression_decrease_by_three_l1349_134915

-- Given a regression equation \hat y = 2 - 3 \hat x
def regression_equation (x : ℝ) : ℝ :=
  2 - 3 * x

-- Prove that when x increases by one unit, \hat y decreases by 3 units
theorem regression_decrease_by_three (x : ℝ) :
  regression_equation (x + 1) - regression_equation x = -3 :=
by
  -- proof
  sorry

end regression_decrease_by_three_l1349_134915


namespace problem1_problem2_l1349_134938

-- Lean statement for Problem 1
theorem problem1 (x : ℝ) : x^2 * x^3 - x^5 = 0 := 
by sorry

-- Lean statement for Problem 2
theorem problem2 (a : ℝ) : (a + 1)^2 + 2 * a * (a - 1) = 3 * a^2 + 1 :=
by sorry

end problem1_problem2_l1349_134938


namespace problem_l1349_134991

-- Definitions based on the provided conditions
def frequency_varies (freq : Real) : Prop := true -- Placeholder definition
def probability_is_stable (prob : Real) : Prop := true -- Placeholder definition
def is_random_event (event : Type) : Prop := true -- Placeholder definition
def is_random_experiment (experiment : Type) : Prop := true -- Placeholder definition
def is_sum_of_events (event1 event2 : Prop) : Prop := event1 ∨ event2 -- Definition of sum of events
def mutually_exclusive (A B : Prop) : Prop := ¬(A ∧ B) -- Definition of mutually exclusive events
def complementary_events (A B : Prop) : Prop := A ↔ ¬B -- Definition of complementary events
def equally_likely_events (events : List Prop) : Prop := true -- Placeholder definition

-- Translation of the questions and correct answers
theorem problem (freq prob : Real) (event experiment : Type) (A B : Prop) (events : List Prop) :
  (¬(frequency_varies freq = probability_is_stable prob)) ∧ -- 1
  ((is_random_event event) ≠ (is_random_experiment experiment)) ∧ -- 2
  (probability_is_stable prob) ∧ -- 3
  (is_sum_of_events A B) ∧ -- 4
  (mutually_exclusive A B → ¬(probability_is_stable (1 - prob))) ∧ -- 5
  (¬(equally_likely_events events)) :=  -- 6
by
  sorry

end problem_l1349_134991


namespace absolute_value_condition_l1349_134900

theorem absolute_value_condition (a : ℝ) (h : |a| = -a) : a = 0 ∨ a < 0 :=
by
  sorry

end absolute_value_condition_l1349_134900


namespace constant_value_l1349_134963

noncomputable def find_constant (p q : ℚ) (h : p / q = 4 / 5) : ℚ :=
    let C := 0.5714285714285714 - (2 * q - p) / (2 * q + p)
    C

theorem constant_value (p q : ℚ) (h : p / q = 4 / 5) :
    find_constant p q h = 0.14285714285714285 := by
    sorry

end constant_value_l1349_134963


namespace ellipse_foci_distance_l1349_134952

noncomputable def distance_between_foci : ℝ :=
  let a := 20
  let b := 10
  2 * Real.sqrt (a ^ 2 - b ^ 2)

theorem ellipse_foci_distance : distance_between_foci = 20 * Real.sqrt 3 := by
  sorry

end ellipse_foci_distance_l1349_134952


namespace sum_of_three_numbers_is_seventy_l1349_134999

theorem sum_of_three_numbers_is_seventy
  (a b c : ℝ)
  (h1 : a ≤ b ∧ b ≤ c)
  (h2 : (a + b + c) / 3 = a + 20)
  (h3 : (a + b + c) / 3 = c - 30)
  (h4 : b = 10)
  (h5 : a + c = 60) :
  a + b + c = 70 :=
  sorry

end sum_of_three_numbers_is_seventy_l1349_134999


namespace least_value_of_q_minus_p_l1349_134964

variables (y p q : ℝ)

/-- Triangle side lengths -/
def BC := y + 7
def AC := y + 3
def AB := 2 * y + 1

/-- Given conditions for triangle inequalities and angle B being the largest -/
def triangle_inequality_conditions :=
  (y + 7 + (y + 3) > 2 * y + 1) ∧
  (y + 7 + (2 * y + 1) > y + 3) ∧
  ((y + 3) + (2 * y + 1) > y + 7)

def angle_largest_conditions :=
  (2 * y + 1 > y + 3) ∧
  (2 * y + 1 > y + 7)

/-- Prove the least possible value of q - p given the conditions -/
theorem least_value_of_q_minus_p
  (h1 : triangle_inequality_conditions y)
  (h2 : angle_largest_conditions y)
  (h3 : 6 < y)
  (h4 : y < 8) :
  q - p = 2 := sorry

end least_value_of_q_minus_p_l1349_134964


namespace linda_change_l1349_134910

-- Defining the conditions
def cost_per_banana : ℝ := 0.30
def number_of_bananas : ℕ := 5
def amount_paid : ℝ := 10.00

-- Proving the statement
theorem linda_change :
  amount_paid - (number_of_bananas * cost_per_banana) = 8.50 :=
by
  sorry

end linda_change_l1349_134910


namespace roger_forgot_lawns_l1349_134929

theorem roger_forgot_lawns
  (dollars_per_lawn : ℕ)
  (total_lawns : ℕ)
  (total_earned : ℕ)
  (actual_mowed_lawns : ℕ)
  (forgotten_lawns : ℕ)
  (h1 : dollars_per_lawn = 9)
  (h2 : total_lawns = 14)
  (h3 : total_earned = 54)
  (h4 : actual_mowed_lawns = total_earned / dollars_per_lawn) :
  forgotten_lawns = total_lawns - actual_mowed_lawns :=
  sorry

end roger_forgot_lawns_l1349_134929


namespace part_I_5_continuous_part_I_6_not_continuous_part_II_min_k_for_8_continuous_part_III_min_k_for_20_continuous_l1349_134985

def is_continuous_representable (m : ℕ) (Q : List ℤ) : Prop :=
  ∀ n ∈ (List.range (m + 1)).tail, ∃ (sublist : List ℤ), sublist ≠ [] ∧ sublist ∈ Q.sublists' ∧ sublist.sum = n

theorem part_I_5_continuous :
  is_continuous_representable 5 [2, 1, 4] :=
sorry

theorem part_I_6_not_continuous :
  ¬is_continuous_representable 6 [2, 1, 4] :=
sorry

theorem part_II_min_k_for_8_continuous (Q : List ℤ) :
  is_continuous_representable 8 Q → Q.length ≥ 4 :=
sorry

theorem part_III_min_k_for_20_continuous (Q : List ℤ) 
  (h : is_continuous_representable 20 Q) (h_sum : Q.sum < 20) :
  Q.length ≥ 7 :=
sorry

end part_I_5_continuous_part_I_6_not_continuous_part_II_min_k_for_8_continuous_part_III_min_k_for_20_continuous_l1349_134985


namespace average_rounds_rounded_is_3_l1349_134948

-- Definitions based on conditions
def golfers : List ℕ := [3, 4, 3, 6, 2, 4]
def rounds : List ℕ := [0, 1, 2, 3, 4, 5]

noncomputable def total_rounds : ℕ :=
  List.sum (List.zipWith (λ g r => g * r) golfers rounds)

def total_golfers : ℕ := List.sum golfers

noncomputable def average_rounds : ℕ :=
  Int.natAbs (Int.ofNat total_rounds / total_golfers).toNat

theorem average_rounds_rounded_is_3 : average_rounds = 3 := by
  sorry

end average_rounds_rounded_is_3_l1349_134948


namespace triangle_side_b_range_l1349_134959

noncomputable def sin60 := Real.sin (Real.pi / 3)

theorem triangle_side_b_range (a b : ℝ) (A : ℝ)
  (ha : a = 2)
  (hA : A = 60 * Real.pi / 180)
  (h_2solutions : b * sin60 < a ∧ a < b) :
  (2 < b ∧ b < 4 * Real.sqrt 3 / 3) :=
by
  sorry

end triangle_side_b_range_l1349_134959


namespace proposition_A_proposition_B_proposition_C_proposition_D_l1349_134981

theorem proposition_A (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (1 / a < 1 → a ≠ 1) :=
by {
  sorry
}

theorem proposition_B : (¬ ∀ x : ℝ, x^2 + x + 1 < 0) → (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≥ 0) :=
by {
  sorry
}

theorem proposition_C : ¬ ∀ x ≠ 0, x + 1 / x ≥ 2 :=
by {
  sorry
}

theorem proposition_D (m : ℝ) : (∃ x : ℝ, (1 < x ∧ x < 2) ∧ x^2 + m * x + 4 < 0) → m < -4 :=
by {
  sorry
}

end proposition_A_proposition_B_proposition_C_proposition_D_l1349_134981


namespace fruit_order_count_l1349_134954

-- Define the initial conditions
def apples := 3
def oranges := 2
def bananas := 2
def totalFruits := apples + oranges + bananas -- which is 7

-- Calculate the factorial of a number
def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

-- Noncomputable definition to skip proof
noncomputable def distinctOrders : ℕ :=
  fact totalFruits / (fact apples * fact oranges * fact bananas)

-- Lean statement expressing that the number of distinct orders is 210
theorem fruit_order_count : distinctOrders = 210 :=
by
  sorry

end fruit_order_count_l1349_134954


namespace calculate_expression_l1349_134987

variable (a : ℝ)

theorem calculate_expression : (-a) ^ 2 * (-a ^ 5) ^ 4 / a ^ 12 * (-2 * a ^ 4) = -2 * a ^ 14 := 
by sorry

end calculate_expression_l1349_134987


namespace oranges_thrown_away_l1349_134965

theorem oranges_thrown_away (original_oranges: ℕ) (new_oranges: ℕ) (total_oranges: ℕ) (x: ℕ)
  (h1: original_oranges = 5) (h2: new_oranges = 28) (h3: total_oranges = 31) :
  original_oranges - x + new_oranges = total_oranges → x = 2 :=
by
  intros h_eq
  -- Proof omitted
  sorry

end oranges_thrown_away_l1349_134965


namespace speed_against_current_l1349_134908

theorem speed_against_current (V_m V_c : ℝ) (h1 : V_m + V_c = 20) (h2 : V_c = 1) :
  V_m - V_c = 18 :=
by
  sorry

end speed_against_current_l1349_134908


namespace mod_3_power_87_plus_5_l1349_134988

theorem mod_3_power_87_plus_5 :
  (3 ^ 87 + 5) % 11 = 3 := 
by
  sorry

end mod_3_power_87_plus_5_l1349_134988


namespace black_equals_sum_of_white_l1349_134966

theorem black_equals_sum_of_white :
  ∃ (a b c d : ℤ) (a_neq_zero : a ≠ 0) (b_neq_zero : b ≠ 0) (c_neq_zero : c ≠ 0) (d_neq_zero : d ≠ 0),
    (c + d * Real.sqrt 7 = (Real.sqrt (a + b * Real.sqrt 2) + Real.sqrt (a - b * Real.sqrt 2))^2) :=
by
  sorry

end black_equals_sum_of_white_l1349_134966


namespace number_of_possible_k_values_l1349_134924

theorem number_of_possible_k_values : 
  ∃ k_values : Finset ℤ, 
    (∀ k ∈ k_values, ∃ (x y : ℤ), y = x - 3 ∧ y = k * x - k) ∧
    k_values.card = 3 := 
sorry

end number_of_possible_k_values_l1349_134924


namespace find_m_range_l1349_134920

def vector_a : ℝ × ℝ := (1, 2)
def dot_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)
def is_acute (a b : ℝ × ℝ) : Prop := dot_product a b > 0

theorem find_m_range (m : ℝ) :
  is_acute vector_a (4, m) → m ∈ Set.Ioo (-2 : ℝ) 8 ∪ Set.Ioi 8 := 
by
  sorry

end find_m_range_l1349_134920


namespace exists_integers_cubes_sum_product_l1349_134943

theorem exists_integers_cubes_sum_product :
  ∃ (a b : ℤ), a^3 + b^3 = 91 ∧ a * b = 12 :=
by
  sorry

end exists_integers_cubes_sum_product_l1349_134943


namespace problem_statement_l1349_134902
-- Broader import to bring in necessary library components.

-- Definition of the equation that needs to be satisfied by the points.
def satisfies_equation (x y : ℝ) : Prop := 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Definitions of the two lines that form the solution set.
def line1 (x y : ℝ) : Prop := y = -x - 2
def line2 (x y : ℝ) : Prop := y = -2 * x + 1

-- Prove that the set of points that satisfy the given equation is the union of the two lines.
theorem problem_statement (x y : ℝ) : satisfies_equation x y ↔ line1 x y ∨ line2 x y :=
sorry

end problem_statement_l1349_134902


namespace max_value_expression_l1349_134917

theorem max_value_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 := 
by sorry

end max_value_expression_l1349_134917


namespace ratio_of_hours_l1349_134982

theorem ratio_of_hours (x y z : ℕ) 
  (h1 : x + y + z = 157) 
  (h2 : z = y - 8) 
  (h3 : z = 56) 
  (h4 : y = x + 10) : 
  (y / gcd y x) = 32 ∧ (x / gcd y x) = 27 := 
by 
  sorry

end ratio_of_hours_l1349_134982


namespace find_number_l1349_134932

theorem find_number (y : ℝ) (h : 0.25 * 820 = 0.15 * y - 20) : y = 1500 :=
by
  sorry

end find_number_l1349_134932


namespace opposite_sides_range_l1349_134968

theorem opposite_sides_range (a : ℝ) : (2 * 1 + 3 * a + 1) * (2 * a - 3 * 1 + 1) < 0 ↔ -1 < a ∧ a < 1 := sorry

end opposite_sides_range_l1349_134968


namespace necessary_not_sufficient_x2_minus_3x_plus_2_l1349_134973

theorem necessary_not_sufficient_x2_minus_3x_plus_2 (m : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ m → x^2 - 3 * x + 2 ≤ 0) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ m ∧ ¬(x^2 - 3 * x + 2 ≤ 0)) →
  m ≥ 2 :=
sorry

end necessary_not_sufficient_x2_minus_3x_plus_2_l1349_134973


namespace painters_work_l1349_134921

theorem painters_work (w1 w2 : ℕ) (d1 d2 : ℚ) (C : ℚ) (h1 : w1 * d1 = C) (h2 : w2 * d2 = C) (p : w1 = 5) (t : d1 = 1.6) (a : w2 = 4) : d2 = 2 := 
by
  sorry

end painters_work_l1349_134921


namespace solve_x_value_l1349_134923
-- Import the necessary libraries

-- Define the problem and the main theorem
theorem solve_x_value (x : ℝ) (h : 3 / x^2 = x / 27) : x = 3 * Real.sqrt 3 :=
by
  sorry

end solve_x_value_l1349_134923


namespace muffins_divide_equally_l1349_134933

theorem muffins_divide_equally (friends : ℕ) (total_muffins : ℕ) (Jessie_and_friends : ℕ) (muffins_per_person : ℕ) :
  friends = 6 →
  total_muffins = 35 →
  Jessie_and_friends = friends + 1 →
  muffins_per_person = total_muffins / Jessie_and_friends →
  muffins_per_person = 5 :=
by
  intros h_friends h_muffins h_people h_division
  sorry

end muffins_divide_equally_l1349_134933


namespace negative_real_root_range_l1349_134975

theorem negative_real_root_range (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ (1 / Real.pi) ^ x = (1 + a) / (1 - a)) ↔ 0 < a ∧ a < 1 :=
by
  sorry

end negative_real_root_range_l1349_134975


namespace solve_fractional_equation_l1349_134962

theorem solve_fractional_equation : ∀ x : ℝ, (2 * x / (x - 1) = 3) ↔ x = 3 := 
by
  sorry

end solve_fractional_equation_l1349_134962


namespace roots_conditions_l1349_134947

theorem roots_conditions (α β m n : ℝ) (h_pos : β > 0)
  (h1 : α + 2 * β = -m)
  (h2 : 2 * α * β + β^2 = -3)
  (h3 : α * β^2 = -n)
  (h4 : α^2 + 2 * β^2 = 6) : 
  m = 0 ∧ n = 2 := by
  sorry

end roots_conditions_l1349_134947


namespace polynomial_remainder_l1349_134998

theorem polynomial_remainder :
  ∀ (q : Polynomial ℚ),
  (q.eval 2 = 8) →
  (q.eval (-3) = -10) →
  ∃ c d : ℚ, (q = (Polynomial.C (c : ℚ) * (Polynomial.X - Polynomial.C 2) * (Polynomial.X + Polynomial.C 3)) + (Polynomial.C 3.6 * Polynomial.X + Polynomial.C 0.8)) :=
by intros q h1 h2; sorry

end polynomial_remainder_l1349_134998


namespace pen_and_notebook_cost_l1349_134906

theorem pen_and_notebook_cost (pen_cost : ℝ) (notebook_cost : ℝ) 
  (h1 : pen_cost = 4.5) 
  (h2 : pen_cost = notebook_cost + 1.8) : 
  pen_cost + notebook_cost = 7.2 := 
  by
    sorry

end pen_and_notebook_cost_l1349_134906


namespace price_of_child_ticket_l1349_134930

theorem price_of_child_ticket (C : ℝ) 
  (adult_ticket_price : ℝ := 8) 
  (total_tickets_sold : ℕ := 34) 
  (adult_tickets_sold : ℕ := 12) 
  (total_revenue : ℝ := 236) 
  (h1 : 12 * adult_ticket_price + (34 - 12) * C = total_revenue) :
  C = 6.36 :=
by
  sorry

end price_of_child_ticket_l1349_134930


namespace solve_system_l1349_134940

theorem solve_system (x y : ℝ) :
  (x^3 - x + 1 = y^2 ∧ y^3 - y + 1 = x^2) ↔ ((x = 1 ∨ x = -1) ∧ (y = 1 ∨ y = -1)) :=
by
  sorry

end solve_system_l1349_134940


namespace motorboat_speeds_l1349_134951

theorem motorboat_speeds (v a x : ℝ) (d : ℝ)
  (h1 : ∀ t1 t2 t1' t2', 
        t1 = d / (v - a) ∧ t1' = d / (v + x - a) ∧ 
        t2 = d / (v + a) ∧ t2' = d / (v + a - x) ∧ 
        (t1 - t1' = t2' - t2)) 
        : x = 2 * a := 
sorry

end motorboat_speeds_l1349_134951


namespace call_center_agents_ratio_l1349_134913

theorem call_center_agents_ratio
  (a b : ℕ) -- Number of agents in teams A and B
  (x : ℝ) -- Calls each member of team B processes
  (h1 : (a : ℝ) / (b : ℝ) = 5 / 8)
  (h2 : b * x * 4 / 7 + a * 6 / 5 * x * 3 / 7 = b * x + a * 6 / 5 * x) :
  (a : ℝ) / (b : ℝ) = 5 / 8 :=
by
  sorry

end call_center_agents_ratio_l1349_134913


namespace shaded_square_cover_columns_l1349_134911

def triangular_number (n : Nat) : Nat := n * (n + 1) / 2

theorem shaded_square_cover_columns :
  ∃ n : Nat, 
    triangular_number n = 136 ∧ 
    ∀ i : Fin 10, ∃ k ≤ n, (triangular_number k) % 10 = i.val :=
sorry

end shaded_square_cover_columns_l1349_134911


namespace smallest_k_equals_26_l1349_134960

open Real

-- Define the condition
def cos_squared_eq_one (θ : ℝ) : Prop :=
  cos θ ^ 2 = 1

-- Define the requirement for θ to be in the form 180°n
def theta_condition (n : ℤ) : Prop :=
  ∃ (k : ℤ), k ^ 2 + k + 81 = 180 * n

-- The problem statement in Lean: Find the smallest positive integer k such that
-- cos squared of (k^2 + k + 81) degrees = 1
noncomputable def smallest_k_satisfying_cos (k : ℤ) : Prop :=
  (∃ n : ℤ, theta_condition n ∧
   cos_squared_eq_one (k ^ 2 + k + 81)) ∧ (∀ m : ℤ, m > 0 ∧ m < k → 
   (∃ n : ℤ, theta_condition n ∧
   cos_squared_eq_one (m ^ 2 + m + 81)) → false)

theorem smallest_k_equals_26 : smallest_k_satisfying_cos 26 := 
  sorry

end smallest_k_equals_26_l1349_134960


namespace Richard_remaining_distance_l1349_134925

theorem Richard_remaining_distance
  (total_distance : ℕ)
  (day1_distance : ℕ)
  (day2_distance : ℕ)
  (day3_distance : ℕ)
  (half_and_subtract : day2_distance = (day1_distance / 2) - 6)
  (total_distance_to_walk : total_distance = 70)
  (distance_day1 : day1_distance = 20)
  (distance_day3 : day3_distance = 10)
  : total_distance - (day1_distance + day2_distance + day3_distance) = 36 :=
  sorry

end Richard_remaining_distance_l1349_134925


namespace original_price_of_article_l1349_134974

theorem original_price_of_article (SP : ℝ) (profit_percent : ℝ) (CP : ℝ) (hSP : SP = 374) (hprofit : profit_percent = 0.10) : 
  CP = 340 ↔ SP = CP * (1 + profit_percent) :=
by 
  sorry

end original_price_of_article_l1349_134974


namespace squares_form_acute_triangle_l1349_134919

theorem squares_form_acute_triangle (a b c x y z d : ℝ)
    (h_triangle : ∀ x y z : ℝ, (x > 0 ∧ y > 0 ∧ z > 0) → (x + y > z) ∧ (x + z > y) ∧ (y + z > x))
    (h_acute : ∀ x y z : ℝ, (x^2 + y^2 > z^2) ∧ (x^2 + z^2 > y^2) ∧ (y^2 + z^2 > x^2))
    (h_inscribed_squares : x = a ^ 2 * b * c / (d * a + b * c) ∧
                           y = b ^ 2 * a * c / (d * b + a * c) ∧
                           z = c ^ 2 * a * b / (d * c + a * b)) :
    (x + y > z) ∧ (x + z > y) ∧ (y + z > x) ∧
    (x^2 + y^2 > z^2) ∧ (x^2 + z^2 > y^2) ∧ (y^2 + z^2 > x^2) :=
sorry

end squares_form_acute_triangle_l1349_134919


namespace find_sum_of_x_and_y_l1349_134909

theorem find_sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 8 * x - 4 * y - 20) : x + y = 2 := 
by
  sorry

end find_sum_of_x_and_y_l1349_134909


namespace sequence_sum_l1349_134904

theorem sequence_sum : (1 - 3 + 5 - 7 + 9 - 11 + 13 - 15 + 17 - 19) = -10 :=
by
  sorry

end sequence_sum_l1349_134904


namespace output_is_three_l1349_134927

-- Define the initial values
def initial_a : ℕ := 1
def initial_b : ℕ := 2

-- Define the final value of a after the computation
def final_a : ℕ := initial_a + initial_b

-- The theorem stating that the final value of a is 3
theorem output_is_three : final_a = 3 := by
  sorry

end output_is_three_l1349_134927


namespace paint_mixture_replacement_l1349_134950

theorem paint_mixture_replacement :
  ∃ x y : ℝ,
    (0.5 * (1 - x) + 0.35 * x = 0.45) ∧
    (0.6 * (1 - y) + 0.45 * y = 0.55) ∧
    (x = 1 / 3) ∧
    (y = 1 / 3) :=
sorry

end paint_mixture_replacement_l1349_134950


namespace car_a_has_higher_avg_speed_l1349_134961

-- Definitions of the conditions for Car A
def distance_car_a : ℕ := 120
def speed_segment_1_car_a : ℕ := 60
def distance_segment_1_car_a : ℕ := 40
def speed_segment_2_car_a : ℕ := 40
def distance_segment_2_car_a : ℕ := 40
def speed_segment_3_car_a : ℕ := 80
def distance_segment_3_car_a : ℕ := distance_car_a - distance_segment_1_car_a - distance_segment_2_car_a

-- Definitions of the conditions for Car B
def distance_car_b : ℕ := 120
def time_segment_1_car_b : ℕ := 1
def speed_segment_1_car_b : ℕ := 60
def time_segment_2_car_b : ℕ := 1
def speed_segment_2_car_b : ℕ := 40
def total_time_car_b : ℕ := 3
def distance_segment_1_car_b := speed_segment_1_car_b * time_segment_1_car_b
def distance_segment_2_car_b := speed_segment_2_car_b * time_segment_2_car_b
def time_segment_3_car_b := total_time_car_b - time_segment_1_car_b - time_segment_2_car_b
def distance_segment_3_car_b := distance_car_b - distance_segment_1_car_b - distance_segment_2_car_b
def speed_segment_3_car_b := distance_segment_3_car_b / time_segment_3_car_b

-- Total Time for Car A
def time_car_a := distance_segment_1_car_a / speed_segment_1_car_a
                + distance_segment_2_car_a / speed_segment_2_car_a
                + distance_segment_3_car_a / speed_segment_3_car_a

-- Average Speed for Car A
def avg_speed_car_a := distance_car_a / time_car_a

-- Total Time for Car B
def time_car_b := total_time_car_b

-- Average Speed for Car B
def avg_speed_car_b := distance_car_b / time_car_b

-- Proof that Car A has a higher average speed than Car B
theorem car_a_has_higher_avg_speed : avg_speed_car_a > avg_speed_car_b := by sorry

end car_a_has_higher_avg_speed_l1349_134961


namespace solution_l1349_134989

theorem solution (x : ℝ) (h : 6 ∈ ({2, 4, x * x - x} : Set ℝ)) : x = 3 ∨ x = -2 := 
by 
  sorry

end solution_l1349_134989


namespace equivalent_modulo_l1349_134996

theorem equivalent_modulo :
  ∃ (n : ℤ), 0 ≤ n ∧ n < 31 ∧ -250 ≡ n [ZMOD 31] ∧ n = 29 := 
by
  sorry

end equivalent_modulo_l1349_134996


namespace find_point_C_l1349_134922

def point := ℝ × ℝ
def is_midpoint (M A B : point) : Prop := (2 * M.1 = A.1 + B.1) ∧ (2 * M.2 = A.2 + B.2)

-- Variables for known points
def A : point := (2, 8)
def M : point := (4, 11)
def L : point := (6, 6)

-- The proof problem: Prove the coordinates of point C
theorem find_point_C (C : point) (B : point) :
  is_midpoint M A B →
  -- (additional conditions related to the angle bisector can be added if specified)
  C = (14, 2) :=
sorry

end find_point_C_l1349_134922


namespace units_sold_at_original_price_l1349_134939

-- Define the necessary parameters and assumptions
variables (a x y : ℝ)
variables (total_units sold_original sold_discount sold_offseason : ℝ)
variables (purchase_price sell_price discount_price clearance_price : ℝ)

-- Define specific conditions
def purchase_units := total_units = 1000
def selling_price := sell_price = 1.25 * a
def discount_cond := discount_price = 1.25 * 0.9 * a
def clearance_cond := clearance_price = 1.25 * 0.60 * a
def holiday_limit := y ≤ 100
def profitability_condition := 1.25 * x + 1.25 * 0.9 * y + 1.25 * 0.60 * (1000 - x - y) > 1000 * a

-- The theorem asserting at least 426 units sold at the original price ensures profitability
theorem units_sold_at_original_price (h1 : total_units = 1000)
  (h2 : sell_price = 1.25 * a) (h3 : discount_price = 1.25 * 0.9 * a)
  (h4 : clearance_price = 1.25 * 0.60 * a) (h5 : y ≤ 100)
  (h6 : 1.25 * x + 1.25 * 0.9 * y + 1.25 * 0.60 * (1000 - x - y) > 1000 * a) :
  x ≥ 426 :=
by
  sorry

end units_sold_at_original_price_l1349_134939


namespace problem1_problem2_l1349_134957

-- First proof problem
theorem problem1 (a b : ℝ) : a^4 + 6 * a^2 * b^2 + b^4 ≥ 4 * a * b * (a^2 + b^2) :=
by sorry

-- Second proof problem
theorem problem2 (a b : ℝ) : ∃ (x : ℝ), 
  (∀ (x : ℝ), |2 * x - a^4 + (1 - 6 * a^2 * b^2 - b^4)| + 2 * |x - (2 * a^3 * b + 2 * a * b^3 - 1)| ≥ 1) ∧
  ∃ (x : ℝ), |2 * x - a^4 + (1 - 6 * a^2 * b^2 - b^4)| + 2 * |x - (2 * a^3 * b + 2 * a * b^3 - 1)| = 1 :=
by sorry

end problem1_problem2_l1349_134957


namespace number_of_initials_sets_l1349_134993

-- Define the letters and the range
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}

-- Number of letters
def number_of_letters : ℕ := letters.card

-- Length of the initials set
def length_of_initials : ℕ := 4

-- Proof statement
theorem number_of_initials_sets : (number_of_letters ^ length_of_initials) = 10000 := by
  sorry

end number_of_initials_sets_l1349_134993


namespace jerry_bought_one_pound_of_pasta_sauce_l1349_134946

-- Definitions of the given conditions
def cost_mustard_oil_per_liter : ℕ := 13
def liters_mustard_oil : ℕ := 2
def cost_pasta_per_pound : ℕ := 4
def pounds_pasta : ℕ := 3
def cost_pasta_sauce_per_pound : ℕ := 5
def leftover_amount : ℕ := 7
def initial_amount : ℕ := 50

-- The goal to prove
theorem jerry_bought_one_pound_of_pasta_sauce :
  (initial_amount - leftover_amount - liters_mustard_oil * cost_mustard_oil_per_liter 
  - pounds_pasta * cost_pasta_per_pound) / cost_pasta_sauce_per_pound = 1 :=
by
  sorry

end jerry_bought_one_pound_of_pasta_sauce_l1349_134946


namespace avg_annual_growth_rate_profit_exceeds_340_l1349_134978

variable (P2018 P2020 : ℝ)
variable (r : ℝ)

theorem avg_annual_growth_rate :
    P2018 = 200 → P2020 = 288 →
    (1 + r)^2 = P2020 / P2018 →
    r = 0.2 :=
by
  intros hP2018 hP2020 hGrowth
  sorry

theorem profit_exceeds_340 (P2020 : ℝ) (r : ℝ) :
    P2020 = 288 → r = 0.2 →
    P2020 * (1 + r) > 340 :=
by
  intros hP2020 hr
  sorry

end avg_annual_growth_rate_profit_exceeds_340_l1349_134978


namespace probability_60_or_more_points_l1349_134901

theorem probability_60_or_more_points :
  let five_choose k := Nat.choose 5 k
  let prob_correct (k : Nat) := (five_choose k) * (1 / 2)^5
  let prob_at_least_3_correct := prob_correct 3 + prob_correct 4 + prob_correct 5
  prob_at_least_3_correct = 1 / 2 := 
sorry

end probability_60_or_more_points_l1349_134901


namespace sequence_formula_l1349_134931

theorem sequence_formula (a : ℕ → ℕ) (n : ℕ) (h : ∀ n ≥ 1, a n = a (n - 1) + n^3) : 
  a n = (n * (n + 1) / 2) ^ 2 := sorry

end sequence_formula_l1349_134931


namespace maximum_gcd_of_sequence_l1349_134997

def a_n (n : ℕ) : ℕ := 100 + n^2

def d_n (n : ℕ) : ℕ := Nat.gcd (a_n n) (a_n (n + 1))

theorem maximum_gcd_of_sequence : ∃ n : ℕ, ∀ m : ℕ, d_n n ≤ d_n m ∧ d_n n = 401 := sorry

end maximum_gcd_of_sequence_l1349_134997


namespace Cary_height_is_72_l1349_134944

variable (Cary_height Bill_height Jan_height : ℕ)

-- Conditions
axiom Bill_height_is_half_Cary_height : Bill_height = Cary_height / 2
axiom Jan_height_is_6_inches_taller_than_Bill : Jan_height = Bill_height + 6
axiom Jan_height_is_42 : Jan_height = 42

-- Theorem statement
theorem Cary_height_is_72 : Cary_height = 72 := 
by
  sorry

end Cary_height_is_72_l1349_134944


namespace sum_first_five_terms_geometric_sequence_l1349_134971

theorem sum_first_five_terms_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ):
  (∀ n, a (n+1) = a 1 * (1/2) ^ n) →
  a 1 = 16 →
  1/2 * (a 4 + a 7) = 9 / 8 →
  S 5 = (a 1 * (1 - (1 / 2) ^ 5)) / (1 - 1 / 2) →
  S 5 = 31 := by
  sorry

end sum_first_five_terms_geometric_sequence_l1349_134971


namespace insulation_cost_l1349_134980

def rectangular_prism_surface_area (l w h : ℕ) : ℕ :=
2 * l * w + 2 * l * h + 2 * w * h

theorem insulation_cost
  (l w h : ℕ) (cost_per_square_foot : ℕ)
  (h_l : l = 6) (h_w : w = 3) (h_h : h = 2) (h_cost : cost_per_square_foot = 20) :
  rectangular_prism_surface_area l w h * cost_per_square_foot = 1440 := 
sorry

end insulation_cost_l1349_134980


namespace probability_A_l1349_134926

variable (A B : Prop)
variable (P : Prop → ℝ)

axiom prob_B : P B = 0.4
axiom prob_A_and_B : P (A ∧ B) = 0.15
axiom prob_notA_and_notB : P (¬ A ∧ ¬ B) = 0.5499999999999999

theorem probability_A : P A = 0.20 :=
by sorry

end probability_A_l1349_134926


namespace jiujiang_liansheng_sampling_l1349_134969

def bag_numbers : List ℕ := [7, 17, 27, 37, 47]

def systematic_sampling (N n : ℕ) (selected_bags : List ℕ) : Prop :=
  ∃ k i, k = N / n ∧ ∀ j, j < List.length selected_bags → selected_bags.get? j = some (i + k * j)

theorem jiujiang_liansheng_sampling :
  systematic_sampling 50 5 bag_numbers :=
by
  sorry

end jiujiang_liansheng_sampling_l1349_134969


namespace quadratic_inequality_solution_set_l1349_134949

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2 * x - 3 < 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end quadratic_inequality_solution_set_l1349_134949


namespace necklaces_caught_l1349_134979

noncomputable def total_necklaces_caught (boudreaux rhonda latch cecilia : ℕ) : ℕ :=
  boudreaux + rhonda + latch + cecilia

theorem necklaces_caught :
  ∃ (boudreaux rhonda latch cecilia : ℕ), 
    boudreaux = 12 ∧
    rhonda = boudreaux / 2 ∧
    latch = 3 * rhonda - 4 ∧
    cecilia = latch + 3 ∧
    total_necklaces_caught boudreaux rhonda latch cecilia = 49 ∧
    (total_necklaces_caught boudreaux rhonda latch cecilia) % 7 = 0 :=
by
  sorry

end necklaces_caught_l1349_134979


namespace school_orchestra_members_l1349_134918

theorem school_orchestra_members (total_members can_play_violin can_play_keyboard neither : ℕ)
    (h1 : total_members = 42)
    (h2 : can_play_violin = 25)
    (h3 : can_play_keyboard = 22)
    (h4 : neither = 3) :
    (can_play_violin + can_play_keyboard) - (total_members - neither) = 8 :=
by
  sorry

end school_orchestra_members_l1349_134918


namespace binary_calculation_l1349_134912

theorem binary_calculation :
  let b1 := 0b110110
  let b2 := 0b101110
  let b3 := 0b100
  let expected_result := 0b11100011110
  ((b1 * b2) / b3) = expected_result := by
  sorry

end binary_calculation_l1349_134912


namespace surface_area_of_cube_l1349_134936

-- Define the volume condition
def volume_of_cube (s : ℝ) := s^3 = 125

-- Define the conversion from decimeters to centimeters
def decimeters_to_centimeters (d : ℝ) := d * 10

-- Define the surface area formula for one side of the cube
def surface_area_one_side (s_cm : ℝ) := s_cm^2

-- Prove that given the volume condition, the surface area of one side is 2500 cm²
theorem surface_area_of_cube
  (s : ℝ)
  (h : volume_of_cube s)
  (s_cm : ℝ := decimeters_to_centimeters s) :
  surface_area_one_side s_cm = 2500 :=
by
  sorry

end surface_area_of_cube_l1349_134936


namespace find_k_l1349_134983

-- Define vectors a, b, and c
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 3)
def c (k : ℝ) : ℝ × ℝ := (k, 2)

-- Define the dot product function for two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the condition for perpendicular vectors
def perpendicular_condition (k : ℝ) : Prop :=
  dot_product (a.1 - k, -1) b = 0

-- State the theorem
theorem find_k : ∃ k : ℝ, perpendicular_condition k ∧ k = 0 := by
  sorry

end find_k_l1349_134983
