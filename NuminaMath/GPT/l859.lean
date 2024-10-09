import Mathlib

namespace light_bulb_arrangement_l859_85925

theorem light_bulb_arrangement :
  let B := 6
  let R := 7
  let W := 9
  let total_arrangements := Nat.choose (B + R) B * Nat.choose (B + R + 1) W
  total_arrangements = 3435432 :=
by
  sorry

end light_bulb_arrangement_l859_85925


namespace frank_more_miles_than_jim_in_an_hour_l859_85951

theorem frank_more_miles_than_jim_in_an_hour
    (jim_distance : ℕ) (jim_time : ℕ)
    (frank_distance : ℕ) (frank_time : ℕ)
    (h_jim : jim_distance = 16)
    (h_jim_time : jim_time = 2)
    (h_frank : frank_distance = 20)
    (h_frank_time : frank_time = 2) :
    (frank_distance / frank_time) - (jim_distance / jim_time) = 2 := 
by
  -- Placeholder for the proof, no proof steps included as instructed.
  sorry

end frank_more_miles_than_jim_in_an_hour_l859_85951


namespace find_integer_tuples_l859_85968

theorem find_integer_tuples (a b c x y z : ℤ) :
  a + b + c = x * y * z →
  x + y + z = a * b * c →
  a ≥ b → b ≥ c → c ≥ 1 →
  x ≥ y → y ≥ z → z ≥ 1 →
  (a, b, c, x, y, z) = (2, 2, 2, 6, 1, 1) ∨
  (a, b, c, x, y, z) = (5, 2, 1, 8, 1, 1) ∨
  (a, b, c, x, y, z) = (3, 3, 1, 7, 1, 1) ∨
  (a, b, c, x, y, z) = (3, 2, 1, 6, 2, 1) :=
by
  sorry

end find_integer_tuples_l859_85968


namespace monotonic_increasing_range_l859_85907

theorem monotonic_increasing_range (a : ℝ) :
  (∀ x : ℝ, (3*x^2 + 2*x - a) ≥ 0) ↔ (a ≤ -1/3) :=
by
  sorry

end monotonic_increasing_range_l859_85907


namespace total_votes_l859_85957

theorem total_votes (V : ℝ) (C R : ℝ) 
  (hC : C = 0.10 * V)
  (hR1 : R = 0.10 * V + 16000)
  (hR2 : R = 0.90 * V) :
  V = 20000 :=
by
  sorry

end total_votes_l859_85957


namespace find_angle_A_l859_85922

theorem find_angle_A (A B C a b c : ℝ)
  (h1 : A + B + C = Real.pi)
  (h2 : B = (A + C) / 2)
  (h3 : 2 * b ^ 2 = 3 * a * c) :
  A = Real.pi / 2 ∨ A = Real.pi / 6 :=
by
  sorry

end find_angle_A_l859_85922


namespace alice_age_2005_l859_85947

-- Definitions
variables (x : ℕ) (age_Alice_2000 age_Grandmother_2000 : ℕ)
variables (born_Alice born_Grandmother : ℕ)

-- Conditions
def alice_grandmother_relation_at_2000 := age_Alice_2000 = x ∧ age_Grandmother_2000 = 3 * x
def birth_year_sum := born_Alice + born_Grandmother = 3870
def birth_year_Alice := born_Alice = 2000 - x
def birth_year_Grandmother := born_Grandmother = 2000 - 3 * x

-- Proving the main statement: Alice's age at the end of 2005
theorem alice_age_2005 : 
  alice_grandmother_relation_at_2000 x age_Alice_2000 age_Grandmother_2000 ∧ 
  birth_year_sum born_Alice born_Grandmother ∧ 
  birth_year_Alice x born_Alice ∧ 
  birth_year_Grandmother x born_Grandmother 
  → 2005 - 2000 + age_Alice_2000 = 37 := 
by 
  intros
  sorry

end alice_age_2005_l859_85947


namespace max_floor_l859_85915

theorem max_floor (x : ℝ) (h : ⌊(x + 4) / 10⌋ = 5) : ⌊(6 * x) / 5⌋ = 67 :=
  sorry

end max_floor_l859_85915


namespace sandy_total_spent_on_clothes_l859_85980

theorem sandy_total_spent_on_clothes :
  let shorts := 13.99
  let shirt := 12.14 
  let jacket := 7.43
  shorts + shirt + jacket = 33.56 := 
by
  sorry

end sandy_total_spent_on_clothes_l859_85980


namespace smallest_term_l859_85908

theorem smallest_term (a1 d : ℕ) (h_a1 : a1 = 7) (h_d : d = 7) :
  ∃ n : ℕ, (a1 + (n - 1) * d) > 150 ∧ (a1 + (n - 1) * d) % 5 = 0 ∧
  (∀ m : ℕ, (a1 + (m - 1) * d) > 150 ∧ (a1 + (m - 1) * d) % 5 = 0 → (a1 + (m - 1) * d) ≥ (a1 + (n - 1) * d)) → a1 + (n - 1) * d = 175 :=
by
  -- We need to prove given the conditions.
  sorry

end smallest_term_l859_85908


namespace parallelogram_construction_l859_85948

theorem parallelogram_construction 
  (α : ℝ) (hα : 0 ≤ α ∧ α < 180)
  (A B : (ℝ × ℝ))
  (in_angle : (∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ α ∧ 
               ∃ θ' : ℝ, 0 ≤ θ' ∧ θ' ≤ α))
  (C D : (ℝ × ℝ)) :
  ∃ O : (ℝ × ℝ), 
    O = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ 
    O = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) :=
sorry

end parallelogram_construction_l859_85948


namespace sum_of_six_consecutive_odd_numbers_l859_85963

theorem sum_of_six_consecutive_odd_numbers (a b c d e f : ℕ) 
  (ha : 135135 = a * b * c * d * e * f)
  (hb : a < b) (hc : b < c) (hd : c < d) (he : d < e) (hf : e < f)
  (hzero : a % 2 = 1) (hone : b % 2 = 1) (htwo : c % 2 = 1) 
  (hthree : d % 2 = 1) (hfour : e % 2 = 1) (hfive : f % 2 = 1) :
  a + b + c + d + e + f = 48 := by
  sorry

end sum_of_six_consecutive_odd_numbers_l859_85963


namespace smallest_constant_N_l859_85903

theorem smallest_constant_N (a : ℝ) (ha : a > 0) : 
  let b := a
  let c := a
  (a = b ∧ b = c) → (a^2 + b^2 + c^2) / (a + b + c) > (0 : ℝ) := 
by
  -- Assuming the proof steps are written here
  sorry

end smallest_constant_N_l859_85903


namespace total_stones_is_odd_l859_85998

variable (d : ℕ) (total_distance : ℕ)

theorem total_stones_is_odd (h1 : d = 10) (h2 : total_distance = 4800) :
  ∃ (N : ℕ), N % 2 = 1 ∧ total_distance = ((N - 1) * 2 * d) :=
by
  -- Let's denote the number of stones as N
  -- Given dx = 10 and total distance as 4800, we want to show that N is odd and 
  -- satisfies the equation: total_distance = ((N - 1) * 2 * d)
  sorry

end total_stones_is_odd_l859_85998


namespace polynomials_exist_l859_85995

theorem polynomials_exist (p : ℕ) (hp : Nat.Prime p) :
  ∃ (P Q : Polynomial ℤ),
  ¬(Polynomial.degree P = 0) ∧ ¬(Polynomial.degree Q = 0) ∧
  (∀ n, (Polynomial.coeff (P * Q) n).natAbs % p =
    if n = 0 then 1
    else if n = 4 then 1
    else if n = 2 then p - 2
    else 0) :=
sorry

end polynomials_exist_l859_85995


namespace rate_of_interest_l859_85939

-- Define the conditions
def P : ℝ := 1200
def SI : ℝ := 432
def T (R : ℝ) : ℝ := R

-- Define the statement to be proven
theorem rate_of_interest (R : ℝ) (h : SI = (P * R * T R) / 100) : R = 6 :=
by sorry

end rate_of_interest_l859_85939


namespace frac_y_over_x_plus_y_eq_one_third_l859_85913

theorem frac_y_over_x_plus_y_eq_one_third (x y : ℝ) (h : y / x = 1 / 2) : y / (x + y) = 1 / 3 := by
  sorry

end frac_y_over_x_plus_y_eq_one_third_l859_85913


namespace tea_consumption_eq1_tea_consumption_eq2_l859_85921

theorem tea_consumption_eq1 (k : ℝ) (w_sunday t_sunday w_wednesday : ℝ) (h1 : w_sunday * t_sunday = k) 
  (h2 : w_wednesday = 4) : 
  t_wednesday = 6 := 
  by sorry

theorem tea_consumption_eq2 (k : ℝ) (w_sunday t_sunday t_thursday : ℝ) (h1 : w_sunday * t_sunday = k) 
  (h2 : t_thursday = 2) : 
  w_thursday = 12 := 
  by sorry

end tea_consumption_eq1_tea_consumption_eq2_l859_85921


namespace maximum_tangency_circles_l859_85962

/-- Points \( P_1, P_2, \ldots, P_n \) are in the plane
    Real numbers \( r_1, r_2, \ldots, r_n \) are such that the distance between \( P_i \) and \( P_j \) is \( r_i + r_j \) for \( i \ne j \).
    -/
theorem maximum_tangency_circles (n : ℕ) (P : Fin n → ℝ × ℝ) (r : Fin n → ℝ)
  (h : ∀ i j : Fin n, i ≠ j → dist (P i) (P j) = r i + r j) : n ≤ 4 :=
sorry

end maximum_tangency_circles_l859_85962


namespace correct_equation_l859_85978

-- Define the initial deposit
def initial_deposit : ℝ := 2500

-- Define the total amount after one year with interest tax deducted
def total_amount : ℝ := 2650

-- Define the annual interest rate
variable (x : ℝ)

-- Define the interest tax rate
def interest_tax_rate : ℝ := 0.20

-- Define the equation for the total amount after one year considering the tax
theorem correct_equation :
  initial_deposit * (1 + (1 - interest_tax_rate) * x) = total_amount :=
sorry

end correct_equation_l859_85978


namespace persons_in_boat_l859_85937

theorem persons_in_boat (W1 W2 new_person_weight : ℝ) (n : ℕ)
  (hW1 : W1 = 55)
  (h_new_person : new_person_weight = 50)
  (hW2 : W2 = W1 - 5) :
  (n * W1 + new_person_weight) / (n + 1) = W2 → false :=
by
  intros h_eq
  sorry

end persons_in_boat_l859_85937


namespace minimum_value_l859_85901

def f (x a : ℝ) : ℝ := x^3 - a*x^2 - a^2*x
def f_prime (x a : ℝ) : ℝ := 3*x^2 - 2*a*x - a^2

theorem minimum_value (a : ℝ) (hf_prime : f_prime 1 a = 0) (ha : a = -3) : ∃ x : ℝ, f x a = -5 := 
sorry

end minimum_value_l859_85901


namespace equation_of_tangent_circle_l859_85979

-- Define the point and conditional tangency
def center : ℝ × ℝ := (5, 4)
def tangent_to_x_axis : Prop := true -- Placeholder for the tangency condition, which is encoded in our reasoning

-- Define the proof statement
theorem equation_of_tangent_circle :
  (∀ (x y : ℝ), tangent_to_x_axis → 
  (center = (5, 4)) → 
  ((x - 5) ^ 2 + (y - 4) ^ 2 = 16)) := 
sorry

end equation_of_tangent_circle_l859_85979


namespace ratio_of_bronze_to_silver_l859_85942

def total_gold_coins := 3500
def num_chests := 5
def total_silver_coins := 500
def coins_per_chest := 1000

-- Definitions based on the conditions to be used in the proof
def gold_coins_per_chest := total_gold_coins / num_chests
def silver_coins_per_chest := total_silver_coins / num_chests
def bronze_coins_per_chest := coins_per_chest - gold_coins_per_chest - silver_coins_per_chest
def bronze_to_silver_ratio := bronze_coins_per_chest / silver_coins_per_chest

theorem ratio_of_bronze_to_silver : bronze_to_silver_ratio = 2 := 
by
  sorry

end ratio_of_bronze_to_silver_l859_85942


namespace intersection_with_complement_N_l859_85912

open Set Real

def M : Set ℝ := {x | x^2 - 4 * x + 3 < 0}
def N : Set ℝ := {x | 0 < x ∧ x < 2}
def complement_N : Set ℝ := {x | x ≤ 0 ∨ x ≥ 2}

theorem intersection_with_complement_N : M ∩ complement_N = Ico 2 3 :=
by {
  sorry
}

end intersection_with_complement_N_l859_85912


namespace solve_equation_1_solve_equation_2_l859_85943

theorem solve_equation_1 (x : ℝ) : x^2 - 7 * x = 0 ↔ (x = 0 ∨ x = 7) :=
by sorry

theorem solve_equation_2 (x : ℝ) : 2 * x^2 - 6 * x + 1 = 0 ↔ (x = (3 + Real.sqrt 7) / 2 ∨ x = (3 - Real.sqrt 7) / 2) :=
by sorry

end solve_equation_1_solve_equation_2_l859_85943


namespace valid_x_for_sqrt_l859_85964

theorem valid_x_for_sqrt (x : ℝ) (hx : x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 3) : x ≥ 2 ↔ x = 3 := 
sorry

end valid_x_for_sqrt_l859_85964


namespace raine_change_l859_85938

noncomputable def price_bracelet : ℝ := 15
noncomputable def price_necklace : ℝ := 10
noncomputable def price_mug : ℝ := 20
noncomputable def price_keychain : ℝ := 5

noncomputable def quantity_bracelet : ℕ := 3
noncomputable def quantity_necklace : ℕ := 2
noncomputable def quantity_mug : ℕ := 1
noncomputable def quantity_keychain : ℕ := 4

noncomputable def discount_rate : ℝ := 0.12

noncomputable def amount_given : ℝ := 100

-- The total cost before discount
noncomputable def total_before_discount : ℝ := 
  quantity_bracelet * price_bracelet + 
  quantity_necklace * price_necklace + 
  quantity_mug * price_mug + 
  quantity_keychain * price_keychain

-- The discount amount
noncomputable def discount_amount : ℝ := total_before_discount * discount_rate

-- The final amount Raine has to pay after discount
noncomputable def final_amount : ℝ := total_before_discount - discount_amount

-- The change Raine gets back
noncomputable def change : ℝ := amount_given - final_amount

theorem raine_change : change = 7.60 := 
by sorry

end raine_change_l859_85938


namespace boat_distance_against_stream_l859_85982

-- Define the conditions
variable (v_s : ℝ)
variable (speed_still_water : ℝ := 9)
variable (distance_downstream : ℝ := 13)

-- Assert the given condition
axiom condition : speed_still_water + v_s = distance_downstream

-- Prove the required distance against the stream
theorem boat_distance_against_stream : (speed_still_water - (distance_downstream - speed_still_water)) = 5 :=
by
  sorry

end boat_distance_against_stream_l859_85982


namespace complex_number_solution_l859_85910

variable (z : ℂ)
variable (i : ℂ)

theorem complex_number_solution (h : (1 - i)^2 / z = 1 + i) (hi : i^2 = -1) : z = -1 - i :=
sorry

end complex_number_solution_l859_85910


namespace amount_of_brown_paint_l859_85981

-- Definition of the conditions
def white_paint : ℕ := 20
def green_paint : ℕ := 15
def total_paint : ℕ := 69

-- Theorem statement for the amount of brown paint
theorem amount_of_brown_paint : (total_paint - (white_paint + green_paint)) = 34 :=
by
  sorry

end amount_of_brown_paint_l859_85981


namespace number_of_correct_answers_l859_85954

theorem number_of_correct_answers (C W : ℕ) (h1 : C + W = 100) (h2 : 5 * C - 2 * W = 210) : C = 58 :=
sorry

end number_of_correct_answers_l859_85954


namespace sophomores_in_seminar_l859_85961

theorem sophomores_in_seminar (P Q x y : ℕ)
  (h1 : P + Q = 50)
  (h2 : x = y)
  (h3 : x = (1 / 5 : ℚ) * P)
  (h4 : y = (1 / 4 : ℚ) * Q) :
  P = 22 :=
by
  sorry

end sophomores_in_seminar_l859_85961


namespace part_I_solution_set_part_II_solution_range_l859_85974

-- Part I: Defining the function and proving the solution set for m = 3
def f (x m : ℝ) : ℝ := |x + 1| + |m - x|

theorem part_I_solution_set (x : ℝ) :
  (f x 3 ≥ 6) ↔ (x ≤ -2 ∨ x ≥ 4) :=
sorry

-- Part II: Proving the range of values for m such that f(x) ≥ 8 for any real number x
theorem part_II_solution_range (m : ℝ) :
  (∀ x : ℝ, f x m ≥ 8) ↔ (m ≤ -9 ∨ m ≥ 7) :=
sorry

end part_I_solution_set_part_II_solution_range_l859_85974


namespace roots_of_quadratic_discriminant_positive_l859_85924

theorem roots_of_quadratic_discriminant_positive {a b c : ℝ} (h : b^2 - 4 * a * c > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) :=
by {
  sorry
}

end roots_of_quadratic_discriminant_positive_l859_85924


namespace find_r_fourth_l859_85996

theorem find_r_fourth (r : ℝ) (h : (r + 1 / r)^2 = 5) : r^4 + 1 / r^4 = 7 :=
by
  sorry

end find_r_fourth_l859_85996


namespace probability_reach_2C_l859_85959

noncomputable def f (x C : ℝ) : ℝ :=
  x / (2 * C)

theorem probability_reach_2C (x C : ℝ) (hC : 0 < C) (hx : 0 < x ∧ x < 2 * C) :
  f x C = x / (2 * C) := 
by
  sorry

end probability_reach_2C_l859_85959


namespace unique_intersection_point_l859_85967

theorem unique_intersection_point (c : ℝ) :
  (∀ x : ℝ, (|x - 20| + |x + 18| = x + c) → (x = 18 - 2 \/ x = 38 - x \/ x = 2 - 3 * x)) →
  c = 18 :=
by
  sorry

end unique_intersection_point_l859_85967


namespace count_multiples_4_6_10_less_300_l859_85977

theorem count_multiples_4_6_10_less_300 : 
  ∃ n, n = 4 ∧ ∀ k ∈ { k : ℕ | k < 300 ∧ (k % 4 = 0) ∧ (k % 6 = 0) ∧ (k % 10 = 0) }, k = 60 * ((k / 60) + 1) - 60 :=
sorry

end count_multiples_4_6_10_less_300_l859_85977


namespace complex_quadrant_l859_85916

open Complex

noncomputable def z : ℂ := (2 * I) / (1 - I)

theorem complex_quadrant (z : ℂ) (h : (1 - I) * z = 2 * I) : 
  z.re < 0 ∧ z.im > 0 := by
  sorry

end complex_quadrant_l859_85916


namespace find_smaller_integer_l859_85956

theorem find_smaller_integer
  (x y : ℤ)
  (h1 : x + y = 30)
  (h2 : 2 * y = 5 * x - 10) :
  x = 10 :=
by
  -- proof would go here
  sorry

end find_smaller_integer_l859_85956


namespace apples_in_box_at_first_l859_85902

noncomputable def initial_apples (X : ℕ) : Prop :=
  (X / 2 - 25 = 6)

theorem apples_in_box_at_first (X : ℕ) : initial_apples X ↔ X = 62 :=
by
  sorry

end apples_in_box_at_first_l859_85902


namespace max_non_colored_cubes_l859_85945

open Nat

-- Define the conditions
def isRectangularPrism (length width height volume : ℕ) := length * width * height = volume

-- The theorem stating the equivalent math proof problem
theorem max_non_colored_cubes (length width height : ℕ) (h₁ : isRectangularPrism length width height 1024) :
(length > 2 ∧ width > 2 ∧ height > 2) → (length - 2) * (width - 2) * (height - 2) = 504 := by
  sorry

end max_non_colored_cubes_l859_85945


namespace no_couples_next_to_each_other_l859_85990

def factorial (n: Nat): Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (m n p q: Nat): Nat :=
  factorial m - n * factorial (m - 1) + p * factorial (m - 2) - q * factorial (m - 3)

theorem no_couples_next_to_each_other :
  arrangements 7 8 24 32 + 16 * factorial 3 = 1488 :=
by
  -- Here we state that the calculation of special arrangements equals 1488.
  sorry

end no_couples_next_to_each_other_l859_85990


namespace amateur_definition_l859_85920
-- Import necessary libraries

-- Define the meaning of "amateur" and state that it is "amateurish" or "non-professional"
def meaning_of_amateur : String :=
  "amateurish or non-professional"

-- The main statement asserting that the meaning of "amateur" is indeed "amateurish" or "non-professional"
theorem amateur_definition : meaning_of_amateur = "amateurish or non-professional" :=
by
  -- The proof is trivial and assumed to be correct
  sorry

end amateur_definition_l859_85920


namespace inequality_le_one_equality_case_l859_85936

open Real

theorem inequality_le_one (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 1) :
    (ab / (a^5 + b^5 + ab) + bc / (b^5 + c^5 + bc) + ca / (c^5 + a^5 + ca) ≤ 1) :=
sorry

theorem equality_case (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 1) :
    (ab / (a^5 + b^5 + ab) + bc / (b^5 + c^5 + bc) + ca / (c^5 + a^5 + ca) = 1) ↔ (a = 1 ∧ b = 1 ∧ c = 1) :=
sorry

end inequality_le_one_equality_case_l859_85936


namespace arithmetic_sequence_terms_l859_85971

theorem arithmetic_sequence_terms (a d n : ℕ) 
  (h_sum_first_3 : 3 * a + 3 * d = 34)
  (h_sum_last_3 : 3 * a + 3 * d * (n - 1) = 146)
  (h_sum_all : n * (2 * a + (n - 1) * d) = 2 * 390) : 
  n = 13 :=
by
  sorry

end arithmetic_sequence_terms_l859_85971


namespace efficiency_of_worker_p_more_than_q_l859_85927

noncomputable def worker_p_rate : ℚ := 1 / 22
noncomputable def combined_rate : ℚ := 1 / 12

theorem efficiency_of_worker_p_more_than_q
  (W_p : ℚ) (W_q : ℚ)
  (h1 : W_p = worker_p_rate)
  (h2 : W_p + W_q = combined_rate) : (W_p / W_q) = 6 / 5 :=
by
  sorry

end efficiency_of_worker_p_more_than_q_l859_85927


namespace min_boys_needed_l859_85941

theorem min_boys_needed
  (T : ℕ) -- total apples
  (n : ℕ) -- total number of boys
  (x : ℕ) -- number of boys collecting 20 apples each
  (y : ℕ) -- number of boys collecting 20% of total apples each
  (h1 : n = x + y)
  (h2 : T = 20 * x + Nat.div (T * 20 * y) 100)
  (hx_pos : x > 0) 
  (hy_pos : y > 0) : n ≥ 2 :=
sorry

end min_boys_needed_l859_85941


namespace geometric_sequence_20_sum_is_2_pow_20_sub_1_l859_85997

def geometric_sequence_sum_condition (a : ℕ → ℕ) (q : ℕ) : Prop :=
  (a 1 * q + 2 * a 1 = 4) ∧ (a 1 ^ 2 * q ^ 4 = a 1 * q ^ 4)

noncomputable def geometric_sequence_sum (a : ℕ → ℕ) (q : ℕ) : ℕ :=
  (a 1 * (1 - q ^ 20)) / (1 - q)

theorem geometric_sequence_20_sum_is_2_pow_20_sub_1 (a : ℕ → ℕ) (q : ℕ) 
  (h : geometric_sequence_sum_condition a q) : 
  geometric_sequence_sum a q =  2 ^ 20 - 1 := 
sorry

end geometric_sequence_20_sum_is_2_pow_20_sub_1_l859_85997


namespace intersection_A_B_l859_85931

-- Definitions of the sets A and B
def set_A : Set ℝ := { x | 3 ≤ x ∧ x ≤ 10 }
def set_B : Set ℝ := { x | 2 < x ∧ x < 7 }

-- Theorem statement to prove the intersection
theorem intersection_A_B : set_A ∩ set_B = { x | 3 ≤ x ∧ x < 7 } := by
  sorry

end intersection_A_B_l859_85931


namespace range_of_alpha_l859_85952

variable {x : ℝ}

noncomputable def curve (x : ℝ) : ℝ := x^3 - x + 2

theorem range_of_alpha (x : ℝ) (α : ℝ) (h : α = Real.arctan (3*x^2 - 1)) :
  α ∈ Set.Ico 0 (Real.pi / 2) ∪ Set.Ico (3 * Real.pi / 4) Real.pi :=
sorry

end range_of_alpha_l859_85952


namespace width_of_room_l859_85946

-- Define the givens
def length_of_room : ℝ := 5.5
def total_cost : ℝ := 20625
def rate_per_sq_meter : ℝ := 1000

-- Define the required proof statement
theorem width_of_room : (total_cost / rate_per_sq_meter) / length_of_room = 3.75 :=
by
  sorry

end width_of_room_l859_85946


namespace sample_average_l859_85923

theorem sample_average (x : ℝ) 
  (h1 : (1 + 3 + 2 + 5 + x) / 5 = 3) : x = 4 := 
by 
  sorry

end sample_average_l859_85923


namespace find_other_number_l859_85934

theorem find_other_number (m n : ℕ) (H1 : n = 26) 
  (H2 : Nat.lcm n m = 52) (H3 : Nat.gcd n m = 8) : m = 16 := by
  sorry

end find_other_number_l859_85934


namespace min_value_abc_l859_85965

theorem min_value_abc : 
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    (a^b % 10 = 4) ∧ (b^c % 10 = 2) ∧ (c^a % 10 = 9) ∧ 
    (a + b + c = 17) :=
  by {
    sorry
  }

end min_value_abc_l859_85965


namespace cross_section_area_ratio_correct_l859_85989

variable (α : ℝ)
noncomputable def cross_section_area_ratio : ℝ := 2 * (Real.cos α)

theorem cross_section_area_ratio_correct (α : ℝ) : 
  cross_section_area_ratio α = 2 * Real.cos α :=
by
  unfold cross_section_area_ratio
  sorry

end cross_section_area_ratio_correct_l859_85989


namespace handshakes_at_convention_l859_85904

theorem handshakes_at_convention (num_gremlins : ℕ) (num_imps : ℕ) 
  (H_gremlins_shake : num_gremlins = 25) (H_imps_shake_gremlins : num_imps = 20) : 
  let handshakes_among_gremlins := num_gremlins * (num_gremlins - 1) / 2
  let handshakes_between_imps_and_gremlins := num_imps * num_gremlins
  let total_handshakes := handshakes_among_gremlins + handshakes_between_imps_and_gremlins
  total_handshakes = 800 := 
by 
  sorry

end handshakes_at_convention_l859_85904


namespace find_a_l859_85930

noncomputable def binomial_expansion_term_coefficient
  (n : ℕ) (r : ℕ) (a : ℝ) (x : ℝ) : ℝ :=
  (2^(n-r)) * ((-a)^r) * (Nat.choose n r) * (x^(n - 2*r))

theorem find_a 
  (a : ℝ)
  (h : binomial_expansion_term_coefficient 7 5 a 1 = 84) 
  : a = -1 :=
sorry

end find_a_l859_85930


namespace solution_set_inequality_l859_85983

theorem solution_set_inequality (x : ℝ) : (1 / x ≤ 1 / 3) ↔ (x ≥ 3 ∨ x < 0) := by
  sorry

end solution_set_inequality_l859_85983


namespace approximate_number_of_fish_l859_85988

/-
  In a pond, 50 fish were tagged and returned. 
  Later, in another catch of 50 fish, 2 were tagged. 
  Assuming the proportion of tagged fish in the second catch approximates that of the pond,
  prove that the total number of fish in the pond is approximately 1250.
-/

theorem approximate_number_of_fish (N : ℕ) 
  (tagged_in_pond : ℕ := 50) 
  (total_in_second_catch : ℕ := 50) 
  (tagged_in_second_catch : ℕ := 2) 
  (proportion_approx : tagged_in_second_catch / total_in_second_catch = tagged_in_pond / N) :
  N = 1250 :=
by
  sorry

end approximate_number_of_fish_l859_85988


namespace probability_of_two_hearts_and_three_diff_suits_l859_85984

def prob_two_hearts_and_three_diff_suits (n : ℕ) : ℚ :=
  if n = 5 then 135 / 1024 else 0

theorem probability_of_two_hearts_and_three_diff_suits :
  prob_two_hearts_and_three_diff_suits 5 = 135 / 1024 :=
by
  sorry

end probability_of_two_hearts_and_three_diff_suits_l859_85984


namespace increasing_function_solve_inequality_find_range_l859_85906

noncomputable def f : ℝ → ℝ := sorry
def a1 := ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f (-x) = -f x
def a2 := f 1 = 1
def a3 := ∀ m n : ℝ, -1 ≤ m ∧ m ≤ 1 ∧ -1 ≤ n ∧ n ≤ 1 ∧ m + n ≠ 0 → (f m + f n) / (m + n) > 0

-- Statement for question (1)
theorem increasing_function : 
  (∀ x y : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1 ∧ x < y → f x < f y) :=
by 
  apply sorry

-- Statement for question (2)
theorem solve_inequality (x : ℝ) :
  (f (x^2 - 1) + f (3 - 3*x) < 0 ↔ 1 < x ∧ x ≤ 4/3) :=
by 
  apply sorry

-- Statement for question (3)
theorem find_range (t : ℝ) :
  (∀ x a : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ a ∧ a ≤ 1 → f x ≤ t^2 - 2*a*t + 1) 
  ↔ (2 ≤ t ∨ t ≤ -2 ∨ t = 0) :=
by 
  apply sorry

end increasing_function_solve_inequality_find_range_l859_85906


namespace clock_correction_l859_85991

def gain_per_day : ℚ := 13 / 4
def hours_per_day : ℕ := 24
def days_passed : ℕ := 9
def extra_hours : ℕ := 8
def total_hours : ℕ := days_passed * hours_per_day + extra_hours
def gain_per_hour : ℚ := gain_per_day / hours_per_day
def total_gain : ℚ := total_hours * gain_per_hour
def required_correction : ℚ := 30.33

theorem clock_correction :
  total_gain = required_correction :=
  by sorry

end clock_correction_l859_85991


namespace exponent_properties_l859_85929

variables (a : ℝ) (m n : ℕ)
-- Conditions
axiom h1 : a^m = 3
axiom h2 : a^n = 2

-- Goal
theorem exponent_properties :
  a^(m + n) = 6 :=
by
  sorry

end exponent_properties_l859_85929


namespace original_price_of_table_l859_85917

noncomputable def original_price (sale_price : ℝ) (discount_rate : ℝ) : ℝ :=
  sale_price / (1 - discount_rate)

theorem original_price_of_table
  (d : ℝ) (p' : ℝ) (h_d : d = 0.10) (h_p' : p' = 450) :
  original_price p' d = 500 := by
  rw [h_d, h_p']
  -- Calculating the original price
  show original_price 450 0.10 = 500
  sorry

end original_price_of_table_l859_85917


namespace minimize_sum_AP_BP_l859_85905

def point := (ℝ × ℝ)

def A : point := (-1, 0)
def B : point := (1, 0)
def center : point := (3, 4)
def radius : ℝ := 2

def on_circle (P : point) : Prop := (P.1 - center.1)^2 + (P.2 - center.2)^2 = radius^2

def AP_squared (P : point) : ℝ := (P.1 - A.1)^2 + (P.2 - A.2)^2
def BP_squared (P : point) : ℝ := (P.1 - B.1)^2 + (P.2 - B.2)^2
def sum_AP_BP_squared (P : point) : ℝ := AP_squared P + BP_squared P

theorem minimize_sum_AP_BP :
  ∀ P : point, on_circle P → sum_AP_BP_squared P = AP_squared (9/5, 12/5) + BP_squared (9/5, 12/5) → 
  P = (9/5, 12/5) :=
sorry

end minimize_sum_AP_BP_l859_85905


namespace inverse_proportion_decreases_l859_85935

theorem inverse_proportion_decreases {x : ℝ} (h : x > 0 ∨ x < 0) : 
  y = 3 / x → ∀ (x1 x2 : ℝ), (x1 > 0 ∨ x1 < 0) → (x2 > 0 ∨ x2 < 0) → x1 < x2 → (3 / x1) > (3 / x2) := 
by
  sorry

end inverse_proportion_decreases_l859_85935


namespace rectangle_width_l859_85928

theorem rectangle_width (L W : ℝ) (h1 : 2 * (L + W) = 16) (h2 : W = L + 2) : W = 5 :=
by
  sorry

end rectangle_width_l859_85928


namespace negation_of_diagonals_equal_l859_85933

-- Define a rectangle type and a function for the diagonals being equal
structure Rectangle :=
  (a b c d : ℝ) -- Assuming rectangle sides

-- Assume a function that checks if the diagonals of a given rectangle are equal
def diagonals_are_equal (r : Rectangle) : Prop :=
  sorry -- The actual function definition is omitted for this context

-- The proof problem
theorem negation_of_diagonals_equal :
  ¬ (∀ r : Rectangle, diagonals_are_equal r) ↔ (∃ r : Rectangle, ¬ diagonals_are_equal r) :=
by
  sorry

end negation_of_diagonals_equal_l859_85933


namespace cake_stand_cost_calculation_l859_85987

-- Define the constants given in the problem
def flour_cost : ℕ := 5
def money_given : ℕ := 43
def change_received : ℕ := 10

-- Define the cost of the cake stand based on the problem's conditions
def cake_stand_cost : ℕ := (money_given - change_received) - flour_cost

-- The theorem we want to prove
theorem cake_stand_cost_calculation : cake_stand_cost = 28 :=
by
  sorry

end cake_stand_cost_calculation_l859_85987


namespace find_g3_l859_85993

variable (g : ℝ → ℝ)

axiom condition_g :
  ∀ x : ℝ, x ≠ 1 / 2 → g x + g ((x + 2) / (2 - 4 * x)) = 2 * x

theorem find_g3 : g 3 = 9 / 2 :=
  by
    sorry

end find_g3_l859_85993


namespace goods_train_length_is_280_meters_l859_85958

def speed_of_man_train_kmph : ℝ := 80
def speed_of_goods_train_kmph : ℝ := 32
def time_to_pass_seconds : ℝ := 9

theorem goods_train_length_is_280_meters :
  let relative_speed_kmph := speed_of_man_train_kmph + speed_of_goods_train_kmph
  let relative_speed_mps := relative_speed_kmph * (1000 / 3600)
  let length_of_goods_train := relative_speed_mps * time_to_pass_seconds
  abs (length_of_goods_train - 280) < 1 :=
by
  -- skipping the proof
  sorry

end goods_train_length_is_280_meters_l859_85958


namespace trigonometric_identity_l859_85949

theorem trigonometric_identity (alpha : ℝ) (h : Real.tan alpha = 2 * Real.tan (π / 5)) :
  (Real.cos (alpha - 3 * π / 10) / Real.sin (alpha - π / 5)) = 3 :=
by
  sorry

end trigonometric_identity_l859_85949


namespace proof_a6_bounds_l859_85914

theorem proof_a6_bounds (a : ℝ) (h : a^5 - a^3 + a = 2) : 3 < a^6 ∧ a^6 < 4 :=
by
  sorry

end proof_a6_bounds_l859_85914


namespace converse_x_gt_y_then_x_gt_abs_y_is_true_l859_85919

theorem converse_x_gt_y_then_x_gt_abs_y_is_true :
  (∀ x y : ℝ, (x > y) → (x > |y|)) → (∀ x y : ℝ, (x > |y|) → (x > y)) :=
by
  sorry

end converse_x_gt_y_then_x_gt_abs_y_is_true_l859_85919


namespace cos_double_angle_l859_85986

theorem cos_double_angle (θ : ℝ) (h : Real.tan θ = -1/3) : Real.cos (2 * θ) = 4/5 :=
sorry

end cos_double_angle_l859_85986


namespace print_time_is_fifteen_l859_85944

noncomputable def time_to_print (total_pages rate : ℕ) := 
  (total_pages : ℚ) / rate

theorem print_time_is_fifteen :
  let rate := 24
  let total_pages := 350
  let time := time_to_print total_pages rate
  round time = 15 := by
  let rate := 24
  let total_pages := 350
  let time := time_to_print total_pages rate
  have time_val : time = (350 : ℚ) / 24 := by rfl
  let rounded_time := round time
  have rounded_time_val : rounded_time = 15 := by sorry
  exact rounded_time_val

end print_time_is_fifteen_l859_85944


namespace sum_of_20th_and_30th_triangular_numbers_l859_85955

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_of_20th_and_30th_triangular_numbers :
  triangular_number 20 + triangular_number 30 = 675 :=
by
  sorry

end sum_of_20th_and_30th_triangular_numbers_l859_85955


namespace range_of_x_l859_85950

variable (f : ℝ → ℝ)

def even_function :=
  ∀ x : ℝ, f (-x) = f x

def monotonically_decreasing :=
  ∀ x y : ℝ, 0 ≤ x → x ≤ y → f y ≤ f x

def f_value_at_2 := f 2 = 0

theorem range_of_x (h1 : even_function f) (h2 : monotonically_decreasing f) (h3 : f_value_at_2 f) :
  { x : ℝ | f (x - 1) > 0 } = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end range_of_x_l859_85950


namespace debby_deletion_l859_85909

theorem debby_deletion :
  ∀ (zoo_pics museum_pics remaining_pics deleted_pics : ℕ),
    zoo_pics = 24 →
    museum_pics = 12 →
    remaining_pics = 22 →
    deleted_pics = zoo_pics + museum_pics - remaining_pics →
    deleted_pics = 14 :=
by
  intros zoo_pics museum_pics remaining_pics deleted_pics h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end debby_deletion_l859_85909


namespace chocolate_chips_per_family_member_l859_85994

def total_family_members : ℕ := 4
def batches_choco_chip : ℕ := 3
def batches_double_choco_chip : ℕ := 2
def batches_white_choco_chip : ℕ := 1
def cookies_per_batch_choco_chip : ℕ := 12
def cookies_per_batch_double_choco_chip : ℕ := 10
def cookies_per_batch_white_choco_chip : ℕ := 15
def choco_chips_per_cookie_choco_chip : ℕ := 2
def choco_chips_per_cookie_double_choco_chip : ℕ := 4
def choco_chips_per_cookie_white_choco_chip : ℕ := 3

theorem chocolate_chips_per_family_member :
  (batches_choco_chip * cookies_per_batch_choco_chip * choco_chips_per_cookie_choco_chip +
   batches_double_choco_chip * cookies_per_batch_double_choco_chip * choco_chips_per_cookie_double_choco_chip +
   batches_white_choco_chip * cookies_per_batch_white_choco_chip * choco_chips_per_cookie_white_choco_chip) / 
   total_family_members = 49 :=
by
  sorry

end chocolate_chips_per_family_member_l859_85994


namespace parallel_lines_l859_85940

theorem parallel_lines (a : ℝ) : (2 * a = a * (a + 4)) → a = -2 :=
by
  intro h
  sorry

end parallel_lines_l859_85940


namespace radii_of_cylinder_and_cone_are_equal_l859_85953

theorem radii_of_cylinder_and_cone_are_equal
  (h : ℝ)
  (r : ℝ)
  (V_cylinder : ℝ := π * r^2 * h)
  (V_cone : ℝ := (1/3) * π * r^2 * h)
  (volume_ratio : V_cylinder / V_cone = 3) :
  r = r :=
by
  sorry

end radii_of_cylinder_and_cone_are_equal_l859_85953


namespace evaluate_f_at_3_over_4_l859_85973

def g (x : ℝ) : ℝ := 1 - x^2

noncomputable def f (y : ℝ) : ℝ := (1 - y) / y

theorem evaluate_f_at_3_over_4 (h : g (x : ℝ) = 1 - x^2) (x_ne_zero : x ≠ 0) :
  f (3 / 4) = 3 :=
by
  sorry

end evaluate_f_at_3_over_4_l859_85973


namespace younger_brother_age_l859_85975

variable (x y : ℕ)

theorem younger_brother_age :
  x + y = 46 →
  y = x / 3 + 10 →
  y = 19 :=
by
  intros h1 h2
  sorry

end younger_brother_age_l859_85975


namespace common_points_line_circle_l859_85966

theorem common_points_line_circle (a b : ℝ) :
    (∃ x y : ℝ, x / a + y / b = 1 ∧ x^2 + y^2 = 1) →
    (1 / (a * a) + 1 / (b * b) ≥ 1) :=
by
  sorry

end common_points_line_circle_l859_85966


namespace molecular_weight_of_N2O5_is_correct_l859_85992

-- Definitions for atomic weights
def atomic_weight_N : ℚ := 14.01
def atomic_weight_O : ℚ := 16.00

-- Define the molecular weight calculation for N2O5
def molecular_weight_N2O5 : ℚ := (2 * atomic_weight_N) + (5 * atomic_weight_O)

-- The theorem to prove
theorem molecular_weight_of_N2O5_is_correct : molecular_weight_N2O5 = 108.02 := by
  -- Proof here
  sorry

end molecular_weight_of_N2O5_is_correct_l859_85992


namespace real_solution_unique_l859_85970

theorem real_solution_unique (x : ℝ) (h : x^4 + (2 - x)^4 + 2 * x = 34) : x = 0 :=
sorry

end real_solution_unique_l859_85970


namespace probability_hare_claims_not_hare_then_not_rabbit_l859_85926

noncomputable def probability_hare_given_claims : ℚ := (27 / 59)

theorem probability_hare_claims_not_hare_then_not_rabbit
  (population : ℚ) (hares : ℚ) (rabbits : ℚ)
  (belief_hare_not_hare : ℚ) (belief_hare_not_rabbit : ℚ)
  (belief_rabbit_not_hare : ℚ) (belief_rabbit_not_rabbit : ℚ) :
  population = 1 ∧ hares = 1/2 ∧ rabbits = 1/2 ∧
  belief_hare_not_hare = 1/4 ∧ belief_hare_not_rabbit = 3/4 ∧
  belief_rabbit_not_hare = 2/3 ∧ belief_rabbit_not_rabbit = 1/3 →
  (27 / 59) = probability_hare_given_claims :=
sorry

end probability_hare_claims_not_hare_then_not_rabbit_l859_85926


namespace anne_cleaning_time_l859_85969

theorem anne_cleaning_time :
  ∃ (A B : ℝ), (4 * (B + A) = 1) ∧ (3 * (B + 2 * A) = 1) ∧ (1 / A = 12) :=
by
  sorry

end anne_cleaning_time_l859_85969


namespace find_integer_n_l859_85960

def s : List ℤ := [8, 11, 12, 14, 15]

theorem find_integer_n (n : ℤ) (h : (s.sum + n) / (s.length + 1) = (25 / 100) * (s.sum / s.length) + (s.sum / s.length)) : n = 30 := by
  sorry

end find_integer_n_l859_85960


namespace remainder_2_power_404_l859_85911

theorem remainder_2_power_404 (y : ℕ) (h_y : y = 2^101) :
  (2^404 + 404) % (2^203 + 2^101 + 1) = 403 := by
sorry

end remainder_2_power_404_l859_85911


namespace triangle_third_side_l859_85900

open Nat

theorem triangle_third_side (a b c : ℝ) (h1 : a = 4) (h2 : b = 9) (h3 : c > 0) :
  (5 < c ∧ c < 13) ↔ c = 6 :=
by
  sorry

end triangle_third_side_l859_85900


namespace weight_of_11_25m_rod_l859_85999

noncomputable def weight_per_meter (total_weight : ℝ) (length : ℝ) : ℝ :=
  total_weight / length

def weight_of_rod (weight_per_length : ℝ) (length : ℝ) : ℝ :=
  weight_per_length * length

theorem weight_of_11_25m_rod :
  let total_weight_8m := 30.4
  let length_8m := 8.0
  let length_11_25m := 11.25
  let weight_per_length := weight_per_meter total_weight_8m length_8m
  weight_of_rod weight_per_length length_11_25m = 42.75 :=
by sorry

end weight_of_11_25m_rod_l859_85999


namespace total_crayons_is_12_l859_85918

-- Definitions
def initial_crayons : ℕ := 9
def added_crayons : ℕ := 3

-- Goal to prove
theorem total_crayons_is_12 : initial_crayons + added_crayons = 12 :=
by
  sorry

end total_crayons_is_12_l859_85918


namespace least_n_for_multiple_of_8_l859_85932

def is_positive_integer (n : ℕ) : Prop := n > 0

def is_multiple_of_8 (k : ℕ) : Prop := ∃ m : ℕ, k = 8 * m

theorem least_n_for_multiple_of_8 :
  ∀ n : ℕ, (is_positive_integer n → is_multiple_of_8 (Nat.factorial n)) → n ≥ 6 :=
by
  sorry

end least_n_for_multiple_of_8_l859_85932


namespace find_a_and_b_l859_85972

theorem find_a_and_b (a b : ℝ) 
  (curve : ∀ x : ℝ, y = x^2 + a * x + b) 
  (tangent : ∀ x : ℝ, y - b = a * x) 
  (tangent_line : ∀ x y : ℝ, x + y = 1) :
  a = -1 ∧ b = 1 := 
by 
  sorry

end find_a_and_b_l859_85972


namespace sum_proper_divisors_81_l859_85985

theorem sum_proper_divisors_81 :
  let proper_divisors : List Nat := [1, 3, 9, 27]
  List.sum proper_divisors = 40 :=
by
  sorry

end sum_proper_divisors_81_l859_85985


namespace largest_among_four_numbers_l859_85976

theorem largest_among_four_numbers
  (a b : ℝ)
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : a + b = 1) :
  b > max (max (1/2) (2 * a * b)) (a^2 + b^2) := 
sorry

end largest_among_four_numbers_l859_85976
