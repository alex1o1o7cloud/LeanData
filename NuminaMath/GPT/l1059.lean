import Mathlib

namespace inequality_proof_l1059_105914

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (1 / a + 1 / b + 9 / c + 25 / d) ≥ (100 / (a + b + c + d)) :=
by
  sorry

end inequality_proof_l1059_105914


namespace find_N_sum_e_l1059_105980

theorem find_N_sum_e (N : ℝ) (e1 e2 : ℝ) :
  (2 * abs (2 - e1) = N) ∧
  (2 * abs (2 - e2) = N) ∧
  (e1 ≠ e2) ∧
  (e1 + e2 = 4) →
  N = 0 :=
by
  sorry

end find_N_sum_e_l1059_105980


namespace range_of_a_l1059_105951

noncomputable def A (x : ℝ) : Prop := x < -2 ∨ x ≥ 1
noncomputable def B (x : ℝ) (a : ℝ) : Prop := x ≥ a

theorem range_of_a (a : ℝ) : (∀ x, A x ∨ B x a) ↔ a ≤ -2 :=
by sorry

end range_of_a_l1059_105951


namespace dinosaur_book_cost_l1059_105908

theorem dinosaur_book_cost (D : ℕ) : 
  (11 + D + 7 = 37) → (D = 19) := 
by 
  intro h
  sorry

end dinosaur_book_cost_l1059_105908


namespace necessary_but_not_sufficient_range_m_l1059_105976

namespace problem

variable (m x y : ℝ)

/-- Propositions for m -/
def P := (1 < m ∧ m < 4) 
def Q := (2 < m ∧ m < 3) ∨ (3 < m ∧  m < 4)

/-- Statements that P => Q is necessary but not sufficient -/
theorem necessary_but_not_sufficient (hP : 1 < m ∧ m < 4) : 
  ((m-1) * (m-4) < 0) ∧ (Q m) :=
by 
  sorry

theorem range_m (h1 : ¬ (P m ∧ Q m)) (h2 : P m ∨ Q m) : 
  1 < m ∧ m ≤ 2 ∨ m = 3 :=
by
  sorry

end problem

end necessary_but_not_sufficient_range_m_l1059_105976


namespace weight_of_each_dumbbell_l1059_105958

-- Definitions based on conditions
def initial_dumbbells : Nat := 4
def added_dumbbells : Nat := 2
def total_dumbbells : Nat := initial_dumbbells + added_dumbbells -- 6
def total_weight : Nat := 120

-- Theorem statement
theorem weight_of_each_dumbbell (h : total_dumbbells = 6) (w : total_weight = 120) :
  total_weight / total_dumbbells = 20 :=
by
  -- Proof is to be written here
  sorry

end weight_of_each_dumbbell_l1059_105958


namespace ratio_solves_for_x_l1059_105978

theorem ratio_solves_for_x (x : ℝ) (h : 0.60 / x = 6 / 4) : x = 0.4 :=
by
  -- The formal proof would go here.
  sorry

end ratio_solves_for_x_l1059_105978


namespace Tim_eats_91_pickle_slices_l1059_105926

theorem Tim_eats_91_pickle_slices :
  let Sammy := 25
  let Tammy := 3 * Sammy
  let Ron := Tammy - 0.15 * Tammy
  let Amy := Sammy + 0.50 * Sammy
  let CombinedTotal := Ron + Amy
  let Tim := CombinedTotal - 0.10 * CombinedTotal
  Tim = 91 :=
by
  admit

end Tim_eats_91_pickle_slices_l1059_105926


namespace solve_quadratic_l1059_105929

theorem solve_quadratic : ∀ x : ℝ, 3 * x^2 - 2 * x - 1 = 0 ↔ (x = -1/3 ∨ x = 1) := by
  intro x
  sorry

end solve_quadratic_l1059_105929


namespace scott_invests_l1059_105906

theorem scott_invests (x r : ℝ) (h1 : 2520 = x + 1260) (h2 : 2520 * 0.08 = x * r) : r = 0.16 :=
by
  -- Proof goes here
  sorry

end scott_invests_l1059_105906


namespace original_amount_l1059_105936

theorem original_amount (X : ℝ) (h : 0.05 * X = 25) : X = 500 :=
sorry

end original_amount_l1059_105936


namespace mike_age_l1059_105925

theorem mike_age : ∀ (m M : ℕ), m = M - 18 ∧ m + M = 54 → m = 18 :=
by
  intros m M
  intro h
  sorry

end mike_age_l1059_105925


namespace minimum_value_of_expression_l1059_105959

theorem minimum_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : 
  ∃ m : ℝ, (m = 8) ∧ (∀ z : ℝ, z = (y / x) + (4 / y) → z ≥ m) :=
sorry

end minimum_value_of_expression_l1059_105959


namespace explicit_expression_solve_inequality_l1059_105954

noncomputable def f (n : ℝ) (x : ℝ) : ℝ := (n^2 - 3*n + 3) * x^(n+1)

theorem explicit_expression (h_power : ∀ n x, f n x = x^3)
  (h_odd : ∀ x, f 2 x = -f 2 (-x)) :
  (∀ n x, f n x = x^3) :=
by
  sorry

theorem solve_inequality (h_power : ∀ n x, f n x = x^3)
  (h_odd : ∀ x, f 2 x = -f 2 (-x))
  (f_eq : ∀ n x, f n x = x^3) :
  ∀ x, (x + 1)^3 + (3 - 2*x)^3 > 0 → x < 4 :=
by
  sorry

end explicit_expression_solve_inequality_l1059_105954


namespace train_distance_proof_l1059_105934

theorem train_distance_proof (c₁ c₂ c₃ : ℝ) : 
  (5 / c₁ + 5 / c₂ = 15) →
  (5 / c₂ + 5 / c₃ = 11) →
  ∀ (x : ℝ), (x / c₁ = 10 / c₂ + (10 + x) / c₃) →
  x = 27.5 := 
by
  sorry

end train_distance_proof_l1059_105934


namespace arithmetic_seq_general_term_geometric_seq_general_term_l1059_105961

theorem arithmetic_seq_general_term (a : ℕ → ℝ) (h1 : a 1 + a 2 = 10) (h2 : a 4 - a 3 = 2) :
  ∀ n, a n = 2 * n + 2 :=
by sorry

theorem geometric_seq_general_term (a b : ℕ → ℝ) (h1 : a 1 + a 2 = 10) (h2 : a 4 - a 3 = 2)
  (h3 : b 2 = a 3) (h4 : b 3 = a 7) :
  ∀ n, b n = 2 ^ (n + 1) :=
by sorry

end arithmetic_seq_general_term_geometric_seq_general_term_l1059_105961


namespace greater_solution_of_quadratic_eq_l1059_105965

theorem greater_solution_of_quadratic_eq (x : ℝ) : 
  (∀ y : ℝ, y^2 + 20 * y - 96 = 0 → (y = 4)) :=
sorry

end greater_solution_of_quadratic_eq_l1059_105965


namespace largest_divisor_of_expression_l1059_105998

theorem largest_divisor_of_expression (n : ℤ) : 6 ∣ (n^4 - n) := 
sorry

end largest_divisor_of_expression_l1059_105998


namespace geometric_sequence_common_ratio_l1059_105902

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 + a 3 = 10) 
  (h2 : a 4 + a 6 = 5/4) 
  (h_sequence : ∀ n, a n = a 1 * q ^ (n - 1)) : 
  q = 1/2 :=
sorry

end geometric_sequence_common_ratio_l1059_105902


namespace calculate_fraction_l1059_105947

theorem calculate_fraction :
  (-1 / 42) / (1 / 6 - 3 / 14 + 2 / 3 - 2 / 7) = -1 / 14 :=
by
  sorry

end calculate_fraction_l1059_105947


namespace soda_cost_l1059_105931

variable (b s : ℕ)

theorem soda_cost (h1 : 2 * b + s = 210) (h2 : b + 2 * s = 240) : s = 90 := by
  sorry

end soda_cost_l1059_105931


namespace simplify_polynomial_l1059_105950

theorem simplify_polynomial :
  (6 * p ^ 4 + 2 * p ^ 3 - 8 * p + 9) + (-3 * p ^ 3 + 7 * p ^ 2 - 5 * p - 1) = 
  6 * p ^ 4 - p ^ 3 + 7 * p ^ 2 - 13 * p + 8 :=
by
  sorry

end simplify_polynomial_l1059_105950


namespace olivia_spent_89_l1059_105938

-- Define initial and subsequent amounts
def initial_amount : ℕ := 100
def atm_amount : ℕ := 148
def after_supermarket : ℕ := 159

-- Total amount before supermarket
def total_before_supermarket : ℕ := initial_amount + atm_amount

-- Amount spent
def amount_spent : ℕ := total_before_supermarket - after_supermarket

-- Proof that Olivia spent 89 dollars
theorem olivia_spent_89 : amount_spent = 89 := sorry

end olivia_spent_89_l1059_105938


namespace tripling_base_exponent_l1059_105930

variables (a b x : ℝ)

theorem tripling_base_exponent (b_ne_zero : b ≠ 0) (r_def : (3 * a)^(3 * b) = a^b * x^b) : x = 27 * a^2 :=
by
  -- Proof omitted as requested
  sorry

end tripling_base_exponent_l1059_105930


namespace mother_gave_80_cents_l1059_105973

theorem mother_gave_80_cents (father_uncles_gift : Nat) (spent_on_candy current_amount : Nat) (gift_from_father gift_from_uncle add_gift_from_uncle : Nat) (x : Nat) :
  father_uncles_gift = gift_from_father + gift_from_uncle ∧
  father_uncles_gift = 110 ∧
  spent_on_candy = 50 ∧
  current_amount = 140 ∧
  gift_from_father = 40 ∧
  gift_from_uncle = 70 ∧
  add_gift_from_uncle = 70 ∧
  x = current_amount + spent_on_candy - father_uncles_gift ∧
  x = 190 - 110 ∨
  x = 80 :=
  sorry

end mother_gave_80_cents_l1059_105973


namespace combined_weight_of_daughter_and_child_l1059_105928

theorem combined_weight_of_daughter_and_child 
  (G D C : ℝ)
  (h1 : G + D + C = 110)
  (h2 : C = 1/5 * G)
  (h3 : D = 50) :
  D + C = 60 :=
sorry

end combined_weight_of_daughter_and_child_l1059_105928


namespace slope_line_point_l1059_105940

theorem slope_line_point (m b : ℝ) (h_slope : m = 3) (h_point : 2 = m * 5 + b) : m + b = -10 :=
by
  sorry

end slope_line_point_l1059_105940


namespace outerCircumference_is_correct_l1059_105956

noncomputable def π : ℝ := Real.pi  
noncomputable def innerCircumference : ℝ := 352 / 7
noncomputable def width : ℝ := 4.001609997739084

noncomputable def radius_inner : ℝ := innerCircumference / (2 * π)
noncomputable def radius_outer : ℝ := radius_inner + width
noncomputable def outerCircumference : ℝ := 2 * π * radius_outer

theorem outerCircumference_is_correct : outerCircumference = 341.194 := by
  sorry

end outerCircumference_is_correct_l1059_105956


namespace triangle_angle_not_greater_than_60_l1059_105964

theorem triangle_angle_not_greater_than_60 (A B C : Real) (h1 : A + B + C = 180) 
  : A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 :=
by {
  sorry
}

end triangle_angle_not_greater_than_60_l1059_105964


namespace work_hours_l1059_105945

namespace JohnnyWork

variable (dollarsPerHour : ℝ) (totalDollars : ℝ)

theorem work_hours 
  (h_wage : dollarsPerHour = 3.25)
  (h_earned : totalDollars = 26) 
  : (totalDollars / dollarsPerHour) = 8 := 
by
  rw [h_wage, h_earned]
  -- proof goes here
  sorry

end JohnnyWork

end work_hours_l1059_105945


namespace prove_a_is_perfect_square_l1059_105987

-- Definition of a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Main theorem statement
theorem prove_a_is_perfect_square 
  (a b : ℕ) 
  (hb_odd : b % 2 = 1) 
  (h_integer : ∃ k : ℕ, ((a + b) * (a + b) + 4 * a) = k * a * b) :
  is_perfect_square a :=
sorry

end prove_a_is_perfect_square_l1059_105987


namespace problem1_problem2_l1059_105970

theorem problem1 (n : ℕ) : 2^n + 3 = k * k → n = 0 :=
by
  intros
  sorry 

theorem problem2 (n : ℕ) : 2^n + 1 = x * x → n = 3 :=
by
  intros
  sorry 

end problem1_problem2_l1059_105970


namespace simplify_nested_fraction_l1059_105974

theorem simplify_nested_fraction :
  (1 : ℚ) / (1 + (1 / (3 + (1 / 4)))) = 13 / 17 :=
by
  sorry

end simplify_nested_fraction_l1059_105974


namespace fifth_score_l1059_105910

theorem fifth_score (r : ℕ) 
  (h1 : r % 5 = 0)
  (h2 : (60 + 75 + 85 + 95 + r) / 5 = 80) : 
  r = 85 := by 
  sorry

end fifth_score_l1059_105910


namespace oldest_child_age_l1059_105996

def avg (a b c : ℕ) : ℕ := (a + b + c) / 3

theorem oldest_child_age (a b : ℕ) (h1 : avg a b x = 10) (h2 : a = 8) (h3 : b = 11) : x = 11 :=
by
  sorry

end oldest_child_age_l1059_105996


namespace smallest_base_conversion_l1059_105921

theorem smallest_base_conversion :
  let n1 := 8 * 9 + 5 -- 85 in base 9
  let n2 := 2 * 6^2 + 1 * 6 -- 210 in base 6
  let n3 := 1 * 4^3 -- 1000 in base 4
  let n4 := 1 * 2^7 - 1 -- 1111111 in base 2
  n3 < n1 ∧ n3 < n2 ∧ n3 < n4 :=
by
  let n1 := 8 * 9 + 5
  let n2 := 2 * 6^2 + 1 * 6
  let n3 := 1 * 4^3
  let n4 := 1 * 2^7 - 1
  sorry

end smallest_base_conversion_l1059_105921


namespace people_in_first_group_l1059_105989

-- Conditions
variables (P W : ℕ) (people_work_rate same_work_rate : ℕ)

-- Given conditions as Lean definitions
-- P people can do 3W in 3 days implies the work rate of the group is W per day
def first_group_work_rate : ℕ := 3 * W / 3

-- 9 people can do 9W in 3 days implies the work rate of these 9 people is 3W per day
def second_group_work_rate : ℕ := 9 * W / 3

-- The work rates are proportional to the number of people
def proportional_work_rate : Prop := P / 9 = first_group_work_rate / second_group_work_rate

-- Lean theorem statement for proof
theorem people_in_first_group (h1 : first_group_work_rate = W) (h2 : second_group_work_rate = 3 * W) :
  P = 3 :=
by
  sorry

end people_in_first_group_l1059_105989


namespace product_of_values_l1059_105952

-- Define the condition
def satisfies_eq (x : ℝ) : Prop := |2 * x| + 4 = 38

-- State the theorem
theorem product_of_values : ∃ x1 x2 : ℝ, satisfies_eq x1 ∧ satisfies_eq x2 ∧ x1 * x2 = -289 := 
by
  sorry

end product_of_values_l1059_105952


namespace quadratic_solution_l1059_105977

theorem quadratic_solution (a b : ℚ) (h : a * 1^2 + b * 1 + 1 = 0) : 3 - a - b = 4 := 
by
  sorry

end quadratic_solution_l1059_105977


namespace cubic_root_identity_l1059_105999

theorem cubic_root_identity (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a * b + a * c + b * c = -3)
  (h3 : a * b * c = -2) : 
  a * (b + c) ^ 2 + b * (c + a) ^ 2 + c * (a + b) ^ 2 = -6 := 
by
  sorry

end cubic_root_identity_l1059_105999


namespace paco_initial_cookies_l1059_105944

-- Define the given conditions
def cookies_given : ℕ := 14
def cookies_eaten : ℕ := 10
def cookies_left : ℕ := 12

-- Proposition to prove: Paco initially had 36 cookies
theorem paco_initial_cookies : (cookies_given + cookies_eaten + cookies_left = 36) :=
by
  sorry

end paco_initial_cookies_l1059_105944


namespace speed_of_current_is_2_l1059_105953

noncomputable def speed_current : ℝ :=
  let still_water_speed := 14  -- kmph
  let distance_m := 40         -- meters
  let time_s := 8.9992800576   -- seconds
  let distance_km := distance_m / 1000
  let time_h := time_s / 3600
  let downstream_speed := distance_km / time_h
  downstream_speed - still_water_speed

theorem speed_of_current_is_2 :
  speed_current = 2 :=
by
  sorry

end speed_of_current_is_2_l1059_105953


namespace fraction_zero_implies_x_is_neg_2_l1059_105904

theorem fraction_zero_implies_x_is_neg_2 {x : ℝ} 
  (h₁ : x^2 - 4 = 0)
  (h₂ : x^2 - 4 * x + 4 ≠ 0) 
  : x = -2 := 
by
  sorry

end fraction_zero_implies_x_is_neg_2_l1059_105904


namespace find_a_l1059_105907

open Set

noncomputable def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
noncomputable def B (a : ℝ) : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}

theorem find_a (a : ℝ) (h : (A ∪ B a) ⊆ (A ∩ B a)) : a = 1 :=
sorry

end find_a_l1059_105907


namespace gunther_typing_l1059_105939

theorem gunther_typing :
  ∀ (wpm : ℚ), (wpm = 160 / 3) → 480 * wpm = 25598 :=
by
  intros wpm h
  sorry

end gunther_typing_l1059_105939


namespace sum_of_cubes_l1059_105923

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 20) : x^3 + y^3 = 87.5 := 
by 
  sorry

end sum_of_cubes_l1059_105923


namespace find_n_l1059_105942

def valid_n (n : ℕ) : Prop :=
  0 ≤ n ∧ n ≤ 14 ∧ n ≡ 10403 [MOD 15]

theorem find_n : ∃ n, valid_n n ∧ n = 8 :=
by
  sorry

end find_n_l1059_105942


namespace find_k_l1059_105909

-- Define the vectors a, b, and c
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (0, 1)

-- Define the vector c involving variable k
variables (k : ℝ)
def vec_c : ℝ × ℝ := (k, -2)

-- Define the combined vector (a + 2b)
def combined_vec : ℝ × ℝ := (vec_a.1 + 2 * vec_b.1, vec_a.2 + 2 * vec_b.2)

-- Define the dot product function
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- State the theorem to prove
theorem find_k (h : dot_product combined_vec (vec_c k) = 0) : k = 8 :=
by sorry

end find_k_l1059_105909


namespace calculate_difference_of_squares_l1059_105992

theorem calculate_difference_of_squares : (640^2 - 360^2) = 280000 := by
  sorry

end calculate_difference_of_squares_l1059_105992


namespace rectangle_area_l1059_105990

theorem rectangle_area (x : ℝ) (h : (x - 3) * (2 * x + 3) = 4 * x - 9) : x = 7 / 2 :=
sorry

end rectangle_area_l1059_105990


namespace inexperienced_sailors_count_l1059_105905

theorem inexperienced_sailors_count
  (I E : ℕ)
  (h1 : I + E = 17)
  (h2 : ∀ (rate_inexperienced hourly_rate experienced_rate : ℕ), hourly_rate = 10 → experienced_rate = 12 → rate_inexperienced = 2400)
  (h3 : ∀ (total_income experienced_salary : ℕ), total_income = 34560 → experienced_salary = 2880)
  (h4 : ∀ (monthly_income : ℕ), monthly_income = 34560)
  : I = 5 := sorry

end inexperienced_sailors_count_l1059_105905


namespace find_a6_l1059_105920

variable (a : ℕ → ℝ)

-- condition: a_2 + a_8 = 16
axiom h1 : a 2 + a 8 = 16

-- condition: a_4 = 1
axiom h2 : a 4 = 1

-- question: Prove that a_6 = 15
theorem find_a6 : a 6 = 15 :=
sorry

end find_a6_l1059_105920


namespace sara_ticket_cost_l1059_105994

noncomputable def calc_ticket_price : ℝ :=
  let rented_movie_cost := 1.59
  let bought_movie_cost := 13.95
  let total_cost := 36.78
  let total_tickets := 2
  let spent_on_tickets := total_cost - (rented_movie_cost + bought_movie_cost)
  spent_on_tickets / total_tickets

theorem sara_ticket_cost : calc_ticket_price = 10.62 := by
  sorry

end sara_ticket_cost_l1059_105994


namespace inverse_implies_negation_l1059_105903

-- Let's define p as a proposition
variable (p : Prop)

-- The inverse of a proposition p, typically the implication of not p implies not q
def inverse (p q : Prop) := ¬p → ¬q

-- The negation of a proposition p is just ¬p
def negation (p : Prop) := ¬p

-- The math problem statement. Prove that if the inverse of p is true, the negation of p is true.
theorem inverse_implies_negation (q : Prop) (h : inverse p q) : negation q := by
  sorry

end inverse_implies_negation_l1059_105903


namespace math_proof_l1059_105927

noncomputable def problem (a b : ℝ) : Prop :=
  a - b = 2 ∧ a^2 + b^2 = 25 → a * b = 10.5

-- We state the problem as a theorem:
theorem math_proof (a b : ℝ) (h1: a - b = 2) (h2: a^2 + b^2 = 25) : a * b = 10.5 :=
by {
  sorry -- Proof goes here
}

end math_proof_l1059_105927


namespace map_length_representation_l1059_105995

theorem map_length_representation (a b : ℕ) (h : a = 15 ∧ b = 90) : b * (20 / a) = 120 :=
by
  sorry

end map_length_representation_l1059_105995


namespace div_five_times_eight_by_ten_l1059_105957

theorem div_five_times_eight_by_ten : (5 * 8) / 10 = 4 := by
  sorry

end div_five_times_eight_by_ten_l1059_105957


namespace total_fish_caught_l1059_105901

-- Definitions based on conditions
def sums : List ℕ := [7, 9, 14, 14, 19, 21]

-- Statement of the proof problem
theorem total_fish_caught : 
  (∃ (a b c d : ℕ), [a+b, a+c, a+d, b+c, b+d, c+d] = sums) → 
  ∃ (a b c d : ℕ), a + b + c + d = 28 :=
by 
  sorry

end total_fish_caught_l1059_105901


namespace wire_length_ratio_l1059_105912

open Real

noncomputable def bonnie_wire_length : ℝ := 12 * 8
noncomputable def bonnie_cube_volume : ℝ := 8^3
noncomputable def roark_unit_cube_volume : ℝ := 2^3
noncomputable def roark_number_of_cubes : ℝ := bonnie_cube_volume / roark_unit_cube_volume
noncomputable def roark_wire_length_per_cube : ℝ := 12 * 2
noncomputable def roark_total_wire_length : ℝ := roark_number_of_cubes * roark_wire_length_per_cube
noncomputable def bonnie_to_roark_wire_ratio := bonnie_wire_length / roark_total_wire_length

theorem wire_length_ratio : bonnie_to_roark_wire_ratio = (1 : ℝ) / 16 :=
by
  sorry

end wire_length_ratio_l1059_105912


namespace pie_eating_contest_l1059_105986

theorem pie_eating_contest :
  let first_student_round1 := (5 : ℚ) / 6
  let first_student_round2 := (1 : ℚ) / 6
  let second_student_total := (2 : ℚ) / 3
  let first_student_total := first_student_round1 + first_student_round2
  first_student_total - second_student_total = 1 / 3 :=
by
  sorry

end pie_eating_contest_l1059_105986


namespace balls_in_rightmost_box_l1059_105993

theorem balls_in_rightmost_box (a : ℕ → ℕ)
  (h₀ : a 1 = 7)
  (h₁ : ∀ i, 1 ≤ i ∧ i ≤ 1990 → a i + a (i + 1) + a (i + 2) + a (i + 3) = 30) :
  a 1993 = 7 :=
sorry

end balls_in_rightmost_box_l1059_105993


namespace kitchen_width_l1059_105919

theorem kitchen_width (length : ℕ) (height : ℕ) (rate : ℕ) (hours : ℕ) (coats : ℕ) 
  (total_painted : ℕ) (half_walls_area : ℕ) (total_walls_area : ℕ)
  (width : ℕ) : 
  length = 12 ∧ height = 10 ∧ rate = 40 ∧ hours = 42 ∧ coats = 3 ∧ 
  total_painted = rate * hours ∧ total_painted = coats * total_walls_area ∧
  half_walls_area = 2 * length * height ∧ total_walls_area = half_walls_area + 2 * width * height ∧
  2 * (total_walls_area - half_walls_area / 2) = 2 * width * height →
  width = 16 := 
by
  sorry

end kitchen_width_l1059_105919


namespace student_contribution_is_4_l1059_105984

-- Definitions based on the conditions in the problem statement
def total_contribution := 90
def available_class_funds := 14
def number_of_students := 19

-- The theorem statement to be proven
theorem student_contribution_is_4 : 
  (total_contribution - available_class_funds) / number_of_students = 4 :=
by
  sorry  -- Proof is not required as per the instructions

end student_contribution_is_4_l1059_105984


namespace original_number_of_men_l1059_105917

theorem original_number_of_men (M : ℤ) (h1 : 8 * M = 5 * (M + 10)) : M = 17 := by
  -- Proof goes here
  sorry

end original_number_of_men_l1059_105917


namespace find_smallest_number_l1059_105971

theorem find_smallest_number (a b c : ℕ) 
  (h1 : a ≤ b) (h2 : b ≤ c)
  (h3 : b = 31)
  (h4 : c = b + 6)
  (h5 : (a + b + c) / 3 = 30) :
  a = 22 := 
sorry

end find_smallest_number_l1059_105971


namespace probability_of_matching_pair_l1059_105933

theorem probability_of_matching_pair (blackSocks blueSocks : ℕ) (h_black : blackSocks = 12) (h_blue : blueSocks = 10) : 
  let totalSocks := blackSocks + blueSocks
  let totalWays := Nat.choose totalSocks 2
  let blackPairWays := Nat.choose blackSocks 2
  let bluePairWays := Nat.choose blueSocks 2
  let matchingPairWays := blackPairWays + bluePairWays
  totalWays = 231 ∧ matchingPairWays = 111 → (matchingPairWays : ℚ) / totalWays = 111 / 231 := 
by
  intros
  sorry

end probability_of_matching_pair_l1059_105933


namespace yellow_highlighters_l1059_105955

def highlighters (pink blue yellow total : Nat) : Prop :=
  (pink + blue + yellow = total)

theorem yellow_highlighters (h : highlighters 3 5 y 15) : y = 7 :=
by 
  sorry

end yellow_highlighters_l1059_105955


namespace rate_of_interest_is_8_l1059_105988

def principal_B : ℕ := 5000
def time_B : ℕ := 2
def principal_C : ℕ := 3000
def time_C : ℕ := 4
def total_interest : ℕ := 1760

theorem rate_of_interest_is_8 :
  ∃ (R : ℝ), ((principal_B * R * time_B) / 100 + (principal_C * R * time_C) / 100 = total_interest) → R = 8 := 
by
  sorry

end rate_of_interest_is_8_l1059_105988


namespace range_of_m_no_zeros_inequality_when_m_zero_l1059_105900

-- Statement for Problem 1
theorem range_of_m_no_zeros (m : ℝ) (h : ∀ x : ℝ, (x^2 + m * x + m) * Real.exp x ≠ 0) : 0 < m ∧ m < 4 :=
sorry

-- Statement for Problem 2
theorem inequality_when_m_zero (x : ℝ) : 
  (x^2) * (Real.exp x) ≥ x^2 + x^3 :=
sorry

end range_of_m_no_zeros_inequality_when_m_zero_l1059_105900


namespace max_value_ineq_l1059_105937

theorem max_value_ineq (x y : ℝ) (h : x^2 + y^2 = 20) : xy + 8*x + y ≤ 42 := by
  sorry

end max_value_ineq_l1059_105937


namespace acute_angles_theorem_l1059_105991

open Real

variable (α β : ℝ)

-- Given conditions
def conditions : Prop :=
  0 < α ∧ α < π / 2 ∧
  0 < β ∧ β < π / 2 ∧
  tan α = 1 / 7 ∧
  sin β = sqrt 10 / 10

-- Proof goal
def proof_goal : Prop :=
  α + 2 * β = π / 4

-- The final theorem
theorem acute_angles_theorem (h : conditions α β) : proof_goal α β :=
  sorry

end acute_angles_theorem_l1059_105991


namespace geometric_sequence_k_eq_6_l1059_105916

theorem geometric_sequence_k_eq_6 
  (a : ℕ → ℝ) (q : ℝ) (k : ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a n = a 1 * q ^ (n - 1))
  (h3 : q ≠ 1)
  (h4 : q ≠ -1)
  (h5 : a k = a 2 * a 5) :
  k = 6 :=
sorry

end geometric_sequence_k_eq_6_l1059_105916


namespace John_walked_miles_to_park_l1059_105960

theorem John_walked_miles_to_park :
  ∀ (total_skateboarded_miles skateboarded_first_leg skateboarded_return_leg walked_miles : ℕ),
    total_skateboarded_miles = 24 →
    skateboarded_first_leg = 10 →
    skateboarded_return_leg = 10 →
    total_skateboarded_miles = skateboarded_first_leg + skateboarded_return_leg + walked_miles →
    walked_miles = 4 :=
by
  intros total_skateboarded_miles skateboarded_first_leg skateboarded_return_leg walked_miles
  intro h1 h2 h3 h4
  sorry

end John_walked_miles_to_park_l1059_105960


namespace sum_of_cubes_of_roots_l1059_105997

theorem sum_of_cubes_of_roots (r1 r2 r3 : ℂ) (h1 : r1 + r2 + r3 = 3) (h2 : r1 * r2 + r1 * r3 + r2 * r3 = 0) (h3 : r1 * r2 * r3 = -1) : 
  r1^3 + r2^3 + r3^3 = 24 :=
  sorry

end sum_of_cubes_of_roots_l1059_105997


namespace rowing_speed_downstream_correct_l1059_105983

/-- Given:
- The speed of the man upstream V_upstream is 20 kmph.
- The speed of the man in still water V_man is 40 kmph.
Prove:
- The speed of the man rowing downstream V_downstream is 60 kmph.
-/
def rowing_speed_downstream : Prop :=
  let V_upstream := 20
  let V_man := 40
  let V_s := V_man - V_upstream
  let V_downstream := V_man + V_s
  V_downstream = 60

theorem rowing_speed_downstream_correct : rowing_speed_downstream := by
  sorry

end rowing_speed_downstream_correct_l1059_105983


namespace quadratic_other_root_l1059_105941

theorem quadratic_other_root (a : ℝ) (h1 : ∃ (x : ℝ), x^2 - 2 * x + a = 0 ∧ x = -1) :
  ∃ (x2 : ℝ), x2^2 - 2 * x2 + a = 0 ∧ x2 = 3 :=
sorry

end quadratic_other_root_l1059_105941


namespace discount_percentage_of_sale_l1059_105946

theorem discount_percentage_of_sale (initial_price sale_coupon saved_amount final_price : ℝ)
    (h1 : initial_price = 125)
    (h2 : sale_coupon = 10)
    (h3 : saved_amount = 44)
    (h4 : final_price = 81) :
    ∃ x : ℝ, x = 0.20 ∧ 
             (initial_price - initial_price * x - sale_coupon) - 
             0.10 * (initial_price - initial_price * x - sale_coupon) = final_price :=
by
  -- Proof should be constructed here
  sorry

end discount_percentage_of_sale_l1059_105946


namespace add_salt_solution_l1059_105969

theorem add_salt_solution
  (initial_amount : ℕ) (added_concentration : ℕ) (desired_concentration : ℕ)
  (initial_concentration : ℝ) :
  initial_amount = 50 ∧ initial_concentration = 0.4 ∧ added_concentration = 10 ∧ desired_concentration = 25 →
  (∃ (x : ℕ), x = 50 ∧ 
    (initial_concentration * initial_amount + 0.1 * x) / (initial_amount + x) = 0.25) :=
by
  sorry

end add_salt_solution_l1059_105969


namespace find_values_of_a_b_solve_inequality_l1059_105924

variable (a b : ℝ)
variable (h1 : ∀ x : ℝ, a * x^2 + b * x + 2 = 0 ↔ x = -1/2 ∨ x = 2)

theorem find_values_of_a_b (h2 : a = -2) (h3 : b = 3) : 
  a = -2 ∧ b = 3 :=
by
  constructor
  exact h2
  exact h3


theorem solve_inequality 
  (h2 : a = -2) (h3 : b = 3) :
  ∀ x : ℝ, (a * x^2 + b * x - 1 > 0) ↔ (1/2 < x ∧ x < 1) :=
by
  sorry

end find_values_of_a_b_solve_inequality_l1059_105924


namespace product_is_cube_l1059_105922

/-
  Given conditions:
    - a, b, and c are distinct composite natural numbers.
    - None of a, b, and c are divisible by any of the integers from 2 to 100 inclusive.
    - a, b, and c are the smallest possible numbers satisfying the above conditions.

  We need to prove that their product a * b * c is a cube of a natural number.
-/

noncomputable def is_composite (n : ℕ) : Prop :=
  ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ n = p * q

theorem product_is_cube (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : is_composite a) (h5 : is_composite b) (h6 : is_composite c)
  (h7 : ∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ a))
  (h8 : ∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ b))
  (h9 : ∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ c))
  (h10 : ∀ (d e f : ℕ), is_composite d → is_composite e → is_composite f → d ≠ e → e ≠ f → d ≠ f → 
         (∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ d)) → (∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ e)) → (∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ f)) →
         (d * e * f ≥ a * b * c)) :
  ∃ (n : ℕ), a * b * c = n ^ 3 :=
by
  sorry

end product_is_cube_l1059_105922


namespace sequence_product_l1059_105918

-- Definitions and conditions
def geometric_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

def a4_value (a : ℕ → ℕ) : Prop :=
a 4 = 2

-- The statement to be proven
theorem sequence_product (a : ℕ → ℕ) (q : ℕ) (h_geo_seq : geometric_sequence a q) (h_a4 : a4_value a) :
  a 2 * a 3 * a 5 * a 6 = 16 :=
sorry

end sequence_product_l1059_105918


namespace work_completion_days_l1059_105981

-- We assume D is a certain number of days and W is some amount of work
variables (D W : ℕ)

-- Define the rate at which 3 people can do 3W work in D days
def rate_3_people : ℚ := 3 * W / D

-- Define the rate at which 5 people can do 5W work in D days
def rate_5_people : ℚ := 5 * W / D

-- The problem states that both rates must be equal
theorem work_completion_days : (3 * D) = D / 3 :=
by sorry

end work_completion_days_l1059_105981


namespace second_candidate_extra_marks_l1059_105911

theorem second_candidate_extra_marks (T : ℝ) (marks_40_percent : ℝ) (marks_passing : ℝ) (marks_60_percent : ℝ) 
  (h1 : marks_40_percent = 0.40 * T)
  (h2 : marks_passing = 160)
  (h3 : marks_60_percent = 0.60 * T)
  (h4 : marks_passing = marks_40_percent + 40) :
  (marks_60_percent - marks_passing) = 20 :=
by
  sorry

end second_candidate_extra_marks_l1059_105911


namespace cannot_form_62_cents_with_six_coins_l1059_105949

-- Define the coin denominations and their values
structure Coin :=
  (value : ℕ)
  (count : ℕ)

def penny : Coin := ⟨1, 6⟩
def nickel : Coin := ⟨5, 6⟩
def dime : Coin := ⟨10, 6⟩
def quarter : Coin := ⟨25, 6⟩
def halfDollar : Coin := ⟨50, 6⟩

-- Define the main theorem statement
theorem cannot_form_62_cents_with_six_coins :
  ¬ (∃ (p n d q h : ℕ),
      p + n + d + q + h = 6 ∧
      1 * p + 5 * n + 10 * d + 25 * q + 50 * h = 62) :=
sorry

end cannot_form_62_cents_with_six_coins_l1059_105949


namespace incorrect_method_D_l1059_105932

-- Conditions definitions
def conditionA (locus : Set α) (cond : α → Prop) :=
  ∀ p, (p ∈ locus ↔ cond p)

def conditionB (locus : Set α) (cond : α → Prop) :=
  ∀ p, (cond p ↔ p ∈ locus)

def conditionC (locus : Set α) (cond : α → Prop) :=
  ∀ p, (¬ (p ∈ locus) ↔ ¬ (cond p))

def conditionD (locus : Set α) (cond : α → Prop) :=
  ∀ p, (p ∈ locus → cond p) ∧ (∃ p, cond p ∧ ¬ (p ∈ locus))

def conditionE (locus : Set α) (cond : α → Prop) :=
  ∀ p, (cond p ↔ p ∈ locus)

-- Main theorem
theorem incorrect_method_D {α : Type} (locus : Set α) (cond : α → Prop) :
  conditionD locus cond →
  ¬ (conditionA locus cond) ∧
  ¬ (conditionB locus cond) ∧
  ¬ (conditionC locus cond) ∧
  ¬ (conditionE locus cond) :=
  sorry

end incorrect_method_D_l1059_105932


namespace original_price_l1059_105979

-- Definitions based on the conditions
def selling_price : ℝ := 1080
def gain_percent : ℝ := 80

-- The proof problem: Prove that the cost price is Rs. 600
theorem original_price (CP : ℝ) (h_sp : CP + CP * (gain_percent / 100) = selling_price) : CP = 600 :=
by
  -- We skip the proof itself
  sorry

end original_price_l1059_105979


namespace average_speed_calculation_l1059_105943

-- Define constants and conditions
def speed_swimming : ℝ := 1
def speed_running : ℝ := 6
def distance : ℝ := 1  -- We use a generic distance d = 1 (assuming normalized unit distance)

-- Proof statement
theorem average_speed_calculation :
  (2 * distance) / ((distance / speed_swimming) + (distance / speed_running)) = 12 / 7 :=
by
  sorry

end average_speed_calculation_l1059_105943


namespace find_points_l1059_105972

def clubsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem find_points (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (y = x ∨ y = -x) := by
  sorry

end find_points_l1059_105972


namespace part1_part2_l1059_105915

-- Part 1
theorem part1 (n : ℕ) (hn : n ≠ 0) (d : ℕ) (hd : d ∣ 2 * n^2) : 
  ∀ m : ℕ, ¬ (m ≠ 0 ∧ m^2 = n^2 + d) :=
by
  sorry 

-- Part 2
theorem part2 (n : ℕ) (hn : n ≠ 0) : 
  ∀ d : ℕ, (d ∣ 3 * n^2 ∧ ∃ m : ℕ, m ≠ 0 ∧ m^2 = n^2 + d) → d = 3 * n^2 :=
by
  sorry

end part1_part2_l1059_105915


namespace find_principal_l1059_105963

variable (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)

theorem find_principal (h₁ : SI = 8625) (h₂ : R = 50 / 3) (h₃ : T = 3 / 4) :
  SI = (P * R * T) / 100 → P = 69000 := sorry

end find_principal_l1059_105963


namespace solve_equation_l1059_105985

theorem solve_equation (x : ℝ) :
    x^6 - 22 * x^2 - Real.sqrt 21 = 0 ↔ x = Real.sqrt ((Real.sqrt 21 + 5) / 2) ∨ x = -Real.sqrt ((Real.sqrt 21 + 5) / 2) := by
  sorry

end solve_equation_l1059_105985


namespace coat_price_reduction_l1059_105962

theorem coat_price_reduction 
    (original_price : ℝ) 
    (reduction_amount : ℝ) 
    (h1 : original_price = 500) 
    (h2 : reduction_amount = 300) : 
    (reduction_amount / original_price) * 100 = 60 := 
by 
  sorry

end coat_price_reduction_l1059_105962


namespace box_volume_l1059_105935

theorem box_volume (a b c : ℝ) (H1 : a * b = 15) (H2 : b * c = 10) (H3 : c * a = 6) : a * b * c = 30 := 
sorry

end box_volume_l1059_105935


namespace exists_contiguous_figure_l1059_105968

-- Definition of the type for different types of rhombuses
inductive RhombusType
| wide
| narrow

-- Definition of a figure composed of rhombuses
structure Figure where
  count_wide : ℕ
  count_narrow : ℕ
  connected : Prop

-- Statement of the proof problem
theorem exists_contiguous_figure : ∃ (f : Figure), f.count_wide = 3 ∧ f.count_narrow = 8 ∧ f.connected :=
sorry

end exists_contiguous_figure_l1059_105968


namespace average_children_per_grade_average_girls_per_grade_average_boys_per_grade_average_club_members_per_grade_l1059_105975

theorem average_children_per_grade (G3_girls G3_boys G3_club : ℕ) 
                                  (G4_girls G4_boys G4_club : ℕ) 
                                  (G5_girls G5_boys G5_club : ℕ) 
                                  (H1 : G3_girls = 28) 
                                  (H2 : G3_boys = 35) 
                                  (H3 : G3_club = 12) 
                                  (H4 : G4_girls = 45) 
                                  (H5 : G4_boys = 42) 
                                  (H6 : G4_club = 15) 
                                  (H7 : G5_girls = 38) 
                                  (H8 : G5_boys = 51) 
                                  (H9 : G5_club = 10) :
   (63 + 87 + 89) / 3 = 79.67 :=
by sorry

theorem average_girls_per_grade (G3_girls G4_girls G5_girls : ℕ) 
                                (H1 : G3_girls = 28) 
                                (H2 : G4_girls = 45) 
                                (H3 : G5_girls = 38) :
   (28 + 45 + 38) / 3 = 37 :=
by sorry

theorem average_boys_per_grade (G3_boys G4_boys G5_boys : ℕ)
                               (H1 : G3_boys = 35) 
                               (H2 : G4_boys = 42) 
                               (H3 : G5_boys = 51) :
   (35 + 42 + 51) / 3 = 42.67 :=
by sorry

theorem average_club_members_per_grade (G3_club G4_club G5_club : ℕ) 
                                       (H1 : G3_club = 12)
                                       (H2 : G4_club = 15)
                                       (H3 : G5_club = 10) :
   (12 + 15 + 10) / 3 = 12.33 :=
by sorry

end average_children_per_grade_average_girls_per_grade_average_boys_per_grade_average_club_members_per_grade_l1059_105975


namespace min_value_inequality_l1059_105948

open Real

theorem min_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 27) : 3 * a + 2 * b + c ≥ 18 := 
sorry

end min_value_inequality_l1059_105948


namespace domain_log_function_l1059_105913

/-- The quadratic expression x^2 - 2x + 3 is always positive. -/
lemma quadratic_positive (x : ℝ) : x^2 - 2*x + 3 > 0 :=
by
  sorry

/-- The domain of the function y = log(x^2 - 2x + 3) is all real numbers. -/
theorem domain_log_function : ∀ x : ℝ, ∃ y : ℝ, y = Real.log (x^2 - 2*x + 3) :=
by
  have h := quadratic_positive
  sorry

end domain_log_function_l1059_105913


namespace mice_meet_after_three_days_l1059_105982

theorem mice_meet_after_three_days 
  (thickness : ℕ) 
  (first_day_distance : ℕ) 
  (big_mouse_double_progress : ℕ → ℕ) 
  (small_mouse_half_remain_distance : ℕ → ℕ) 
  (days : ℕ) 
  (big_mouse_distance : ℚ) : 
  thickness = 5 ∧ 
  first_day_distance = 1 ∧ 
  (∀ n, big_mouse_double_progress n = 2 ^ (n - 1)) ∧ 
  (∀ n, small_mouse_half_remain_distance n = 5 - (5 / 2 ^ (n - 1))) ∧ 
  days = 3 → 
  big_mouse_distance = 3 + 8 / 17 := 
by
  sorry

end mice_meet_after_three_days_l1059_105982


namespace scientific_notation_of_19672_l1059_105967

theorem scientific_notation_of_19672 :
  ∃ a b, 19672 = a * 10^b ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.9672 ∧ b = 4 :=
sorry

end scientific_notation_of_19672_l1059_105967


namespace ratio_cereal_A_to_B_l1059_105966

-- Definitions translated from conditions
def sugar_percentage_A : ℕ := 10
def sugar_percentage_B : ℕ := 2
def desired_sugar_percentage : ℕ := 6

-- The theorem based on the question and correct answer
theorem ratio_cereal_A_to_B :
  let difference_A := sugar_percentage_A - desired_sugar_percentage
  let difference_B := desired_sugar_percentage - sugar_percentage_B
  difference_A = 4 ∧ difference_B = 4 → 
  difference_B / difference_A = 1 :=
by
  intros
  sorry

end ratio_cereal_A_to_B_l1059_105966
