import Mathlib

namespace fraction_meaningful_range_l75_75035

-- Define the condition where the fraction is not undefined.
def meaningful_fraction (x : ℝ) : Prop := x - 5 ≠ 0

-- Prove the range of x which makes the fraction meaningful.
theorem fraction_meaningful_range (x : ℝ) : meaningful_fraction x ↔ x ≠ 5 :=
by
  sorry

end fraction_meaningful_range_l75_75035


namespace middle_number_is_40_l75_75740

theorem middle_number_is_40 (A B C : ℕ) (h1 : C = 56) (h2 : C - A = 32) (h3 : B / C = 5 / 7) : B = 40 :=
  sorry

end middle_number_is_40_l75_75740


namespace timothy_movies_count_l75_75778

variable (T : ℕ)

def timothy_movies_previous_year (T : ℕ) :=
  let timothy_2010 := T + 7
  let theresa_2010 := 2 * (T + 7)
  let theresa_previous := T / 2
  T + timothy_2010 + theresa_2010 + theresa_previous = 129

theorem timothy_movies_count (T : ℕ) (h : timothy_movies_previous_year T) : T = 24 := 
by 
  sorry

end timothy_movies_count_l75_75778


namespace probability_of_Ace_then_King_l75_75246

def numAces : ℕ := 4
def numKings : ℕ := 4
def totalCards : ℕ := 52

theorem probability_of_Ace_then_King : 
  (numAces / totalCards) * (numKings / (totalCards - 1)) = 4 / 663 :=
by
  sorry

end probability_of_Ace_then_King_l75_75246


namespace ratio_of_vanilla_chips_l75_75995

-- Definitions from the conditions
variable (V_c S_c V_v S_v : ℕ)
variable (H1 : V_c = S_c + 5)
variable (H2 : S_c = 25)
variable (H3 : V_v = 20)
variable (H4 : V_c + S_c + V_v + S_v = 90)

-- The statement we want to prove
theorem ratio_of_vanilla_chips : S_v / V_v = 3 / 4 := by
  sorry

end ratio_of_vanilla_chips_l75_75995


namespace tee_shirts_with_60_feet_of_material_l75_75381

def tee_shirts (f t : ℕ) : ℕ := t / f

theorem tee_shirts_with_60_feet_of_material :
  tee_shirts 4 60 = 15 :=
by
  sorry

end tee_shirts_with_60_feet_of_material_l75_75381


namespace teagan_total_cost_l75_75134

theorem teagan_total_cost :
  let reduction_percentage := 20
  let original_price_shirt := 60
  let original_price_jacket := 90
  let reduced_price_shirt := original_price_shirt * (100 - reduction_percentage) / 100
  let reduced_price_jacket := original_price_jacket * (100 - reduction_percentage) / 100
  let cost_5_shirts := 5 * reduced_price_shirt
  let cost_10_jackets := 10 * reduced_price_jacket
  let total_cost := cost_5_shirts + cost_10_jackets
  total_cost = 960 := by
  sorry

end teagan_total_cost_l75_75134


namespace triangle_perimeter_upper_bound_l75_75260

theorem triangle_perimeter_upper_bound (a b : ℕ) (s : ℕ) (h₁ : a = 7) (h₂ : b = 23) 
  (h₃ : 16 < s) (h₄ : s < 30) : 
  ∃ n : ℕ, n = 60 ∧ n > a + b + s := 
by
  sorry

end triangle_perimeter_upper_bound_l75_75260


namespace permutation_probability_l75_75045

theorem permutation_probability (total_digits: ℕ) (zeros: ℕ) (ones: ℕ) 
  (total_permutations: ℕ) (favorable_permutations: ℕ) (probability: ℚ)
  (h1: total_digits = 6) 
  (h2: zeros = 2) 
  (h3: ones = 4) 
  (h4: total_permutations = 2 ^ total_digits) 
  (h5: favorable_permutations = Nat.choose total_digits zeros) 
  (h6: probability = favorable_permutations / total_permutations) : 
  probability = 15 / 64 := 
sorry

end permutation_probability_l75_75045


namespace failed_english_is_45_l75_75284

-- Definitions of the given conditions
def total_students : ℝ := 1 -- representing 100%
def failed_hindi : ℝ := 0.35
def failed_both : ℝ := 0.2
def passed_both : ℝ := 0.4

-- The goal is to prove that the percentage of students who failed in English is 45%

theorem failed_english_is_45 :
  let failed_at_least_one := total_students - passed_both
  let failed_english := failed_at_least_one - failed_hindi + failed_both
  failed_english = 0.45 :=
by
  -- The steps and manipulation will go here, but for now we skip with sorry
  sorry

end failed_english_is_45_l75_75284


namespace company_pays_300_per_month_l75_75788

theorem company_pays_300_per_month
  (length width height : ℝ)
  (total_volume : ℝ)
  (cost_per_box_per_month : ℝ)
  (h1 : length = 15)
  (h2 : width = 12)
  (h3 : height = 10)
  (h4 : total_volume = 1080000)
  (h5 : cost_per_box_per_month = 0.5) :
  (total_volume / (length * width * height)) * cost_per_box_per_month = 300 := by
  sorry

end company_pays_300_per_month_l75_75788


namespace solve_equation_l75_75997

theorem solve_equation (x : ℝ) : 
  (x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 ∨ x = -3 + Real.sqrt 6 ∨ x = -3 - Real.sqrt 6) ↔ 
  (x^4 / (2 * x + 1) + x^2 = 6 * (2 * x + 1)) := by
  sorry

end solve_equation_l75_75997


namespace solid_brick_height_l75_75297

theorem solid_brick_height (n c base_perimeter height : ℕ) 
  (h1 : n = 42) 
  (h2 : c = 1) 
  (h3 : base_perimeter = 18)
  (h4 : n % base_area = 0)
  (h5 : 2 * (length + width) = base_perimeter)
  (h6 : base_area * height = n) : 
  height = 3 :=
by sorry

end solid_brick_height_l75_75297


namespace percentage_decrease_10_l75_75108

def stocks_decrease (F J M : ℝ) (X : ℝ) : Prop :=
  J = F * (1 - X / 100) ∧
  J = M * 1.20 ∧
  M = F * 0.7500000000000007

theorem percentage_decrease_10 {F J M X : ℝ} (h : stocks_decrease F J M X) :
  X = 9.99999999999992 :=
by
  sorry

end percentage_decrease_10_l75_75108


namespace find_q_l75_75211

theorem find_q (p q : ℝ) (hp : 1 < p) (hq : 1 < q) (hcond1 : 1/p + 1/q = 1) (hcond2 : p * q = 9) :
    q = (9 + 3 * Real.sqrt 5) / 2 ∨ q = (9 - 3 * Real.sqrt 5) / 2 :=
by
  sorry

end find_q_l75_75211


namespace y_value_when_x_neg_one_l75_75618

theorem y_value_when_x_neg_one (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2 * t) 
  (h2 : y = t^2 + 3 * t + 6) 
  (h3 : x = -1) : 
  y = 16 := 
by sorry

end y_value_when_x_neg_one_l75_75618


namespace selling_price_to_achieve_profit_l75_75859

theorem selling_price_to_achieve_profit (num_pencils : ℝ) (cost_per_pencil : ℝ) (desired_profit : ℝ) (selling_price : ℝ) :
  num_pencils = 1800 →
  cost_per_pencil = 0.15 →
  desired_profit = 100 →
  selling_price = 0.21 :=
by
  sorry

end selling_price_to_achieve_profit_l75_75859


namespace smallest_positive_n_l75_75508

theorem smallest_positive_n (n : ℕ) (h1 : 0 < n) (h2 : gcd (8 * n - 3) (6 * n + 4) > 1) : n = 1 :=
sorry

end smallest_positive_n_l75_75508


namespace both_solve_prob_l75_75125

variable (a b : ℝ) -- Define a and b as real numbers

-- Define the conditions
def not_solve_prob_A := (0 ≤ a) ∧ (a ≤ 1)
def not_solve_prob_B := (0 ≤ b) ∧ (b ≤ 1)
def independent := true -- independence is implicit by the question

-- Define the statement of the proof
theorem both_solve_prob (h1 : not_solve_prob_A a) (h2 : not_solve_prob_B b) :
  (1 - a) * (1 - b) = (1 - a) * (1 - b) :=
by sorry

end both_solve_prob_l75_75125


namespace cristina_speed_cristina_running_speed_l75_75806

theorem cristina_speed 
  (head_start : ℕ)
  (nicky_speed : ℕ)
  (catch_up_time : ℕ)
  (distance : ℕ := head_start + (nicky_speed * catch_up_time))
  : distance / catch_up_time = 6
  := by
  sorry

-- Given conditions used as definitions in Lean 4:
-- head_start = 36 (meters)
-- nicky_speed = 3 (meters/second)
-- catch_up_time = 12 (seconds)

theorem cristina_running_speed
  (head_start : ℕ := 36)
  (nicky_speed : ℕ := 3)
  (catch_up_time : ℕ := 12)
  : (head_start + (nicky_speed * catch_up_time)) / catch_up_time = 6
  := by
  sorry

end cristina_speed_cristina_running_speed_l75_75806


namespace proof_strictly_increasing_sequence_l75_75875

noncomputable def exists_strictly_increasing_sequence : Prop :=
  ∃ a : ℕ → ℕ, 
    (∀ n m : ℕ, n < m → a n < a m) ∧
    (∀ m : ℕ, ∃ i j : ℕ, m = a i + a j) ∧
    (∀ n : ℕ, 0 < n → a n > n^2 / 16)

theorem proof_strictly_increasing_sequence : exists_strictly_increasing_sequence :=
  sorry

end proof_strictly_increasing_sequence_l75_75875


namespace equation_of_midpoint_trajectory_l75_75962

theorem equation_of_midpoint_trajectory
  (M : ℝ × ℝ)
  (hM : M.1 ^ 2 + M.2 ^ 2 = 1)
  (N : ℝ × ℝ := (2, 0))
  (P : ℝ × ℝ := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)) :
  (P.1 - 1) ^ 2 + P.2 ^ 2 = 1 / 4 := 
sorry

end equation_of_midpoint_trajectory_l75_75962


namespace total_cost_of_topsoil_l75_75439

-- Definitions
def cost_per_cubic_foot : ℝ := 8
def cubic_yard_to_cubic_foot : ℝ := 27
def volume_in_cubic_yards : ℕ := 8

-- The total cost of 8 cubic yards of topsoil
theorem total_cost_of_topsoil : volume_in_cubic_yards * cubic_yard_to_cubic_foot * cost_per_cubic_foot = 1728 := by
  sorry

end total_cost_of_topsoil_l75_75439


namespace range_a_l75_75259

variable (a : ℝ)

def p := (∀ x : ℝ, x^2 + x + a > 0)
def q := ∃ x y : ℝ, x^2 - 2 * a * x + 1 ≤ y

theorem range_a :
  ({a : ℝ | (p a ∧ ¬q a) ∨ (¬p a ∧ q a)} = {a : ℝ | a < -1} ∪ {a : ℝ | 1 / 4 < a ∧ a < 1}) := 
by
  sorry

end range_a_l75_75259


namespace monthly_earnings_is_correct_l75_75154

-- Conditions as definitions

def first_floor_cost_per_room : ℕ := 15
def second_floor_cost_per_room : ℕ := 20
def first_floor_rooms : ℕ := 3
def second_floor_rooms : ℕ := 3
def third_floor_rooms : ℕ := 3
def occupied_third_floor_rooms : ℕ := 2

-- Calculated values from conditions
def third_floor_cost_per_room : ℕ := 2 * first_floor_cost_per_room

-- Total earnings on each floor
def first_floor_earnings : ℕ := first_floor_cost_per_room * first_floor_rooms
def second_floor_earnings : ℕ := second_floor_cost_per_room * second_floor_rooms
def third_floor_earnings : ℕ := third_floor_cost_per_room * occupied_third_floor_rooms

-- Total monthly earnings
def total_monthly_earnings : ℕ :=
  first_floor_earnings + second_floor_earnings + third_floor_earnings

theorem monthly_earnings_is_correct : total_monthly_earnings = 165 := by
  -- proof omitted
  sorry

end monthly_earnings_is_correct_l75_75154


namespace gcd_of_a_and_b_lcm_of_a_and_b_l75_75315

def a : ℕ := 2 * 3 * 7
def b : ℕ := 2 * 3 * 3 * 5

theorem gcd_of_a_and_b : Nat.gcd a b = 6 := by
  sorry

theorem lcm_of_a_and_b : Nat.lcm a b = 630 := by
  sorry

end gcd_of_a_and_b_lcm_of_a_and_b_l75_75315


namespace solve_system_of_equations_l75_75235

theorem solve_system_of_equations :
  ∀ (x y : ℝ),
  (3 * x - 2 * y = 7) →
  (2 * x + 3 * y = 8) →
  x = 37 / 13 :=
by
  intros x y h1 h2
  -- to prove x = 37 / 13 from the given system of equations
  sorry

end solve_system_of_equations_l75_75235


namespace intersection_of_A_and_B_l75_75855

open Set

def A := {x : ℝ | 2 + x ≥ 4}
def B := {x : ℝ | -1 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x ≤ 5} := sorry

end intersection_of_A_and_B_l75_75855


namespace remainder_of_9_pow_1995_mod_7_l75_75096

theorem remainder_of_9_pow_1995_mod_7 : (9^1995) % 7 = 1 := 
by 
sorry

end remainder_of_9_pow_1995_mod_7_l75_75096


namespace tom_sold_price_l75_75527

noncomputable def original_price : ℝ := 200
noncomputable def tripled_price (price : ℝ) : ℝ := 3 * price
noncomputable def sold_price (price : ℝ) : ℝ := 0.4 * price

theorem tom_sold_price : sold_price (tripled_price original_price) = 240 := 
by
  sorry

end tom_sold_price_l75_75527


namespace total_children_estimate_l75_75081

theorem total_children_estimate (k m n : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : k > 0) 
(h4 : n ≤ m) (h5 : n ≤ k) (h6 : m ≤ k) :
  (∃ (total : ℕ), total = k * m / n) :=
sorry

end total_children_estimate_l75_75081


namespace math_problem_l75_75312

variable (a : ℝ) (m n : ℝ)

theorem math_problem
  (h1 : a^m = 3)
  (h2 : a^n = 2) :
  a^(2*m + 3*n) = 72 := 
  sorry

end math_problem_l75_75312


namespace range_of_a_for_inequality_l75_75838

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ p q : ℝ, (0 < p ∧ p < 1) → (0 < q ∧ q < 1) → p ≠ q → (f a p - f a q) / (p - q) > 1) ↔ 3 ≤ a :=
sorry

end range_of_a_for_inequality_l75_75838


namespace bobs_walking_rate_l75_75906

theorem bobs_walking_rate (distance_XY : ℕ) 
  (yolanda_rate : ℕ) 
  (bob_distance_when_met : ℕ) 
  (yolanda_extra_hour : ℕ)
  (meet_covered_distance : distance_XY = yolanda_rate * (bob_distance_when_met / yolanda_rate + yolanda_extra_hour - 1 + bob_distance_when_met / bob_distance_when_met)) 
  (yolanda_distance_when_met : yolanda_rate * (bob_distance_when_met / yolanda_rate + yolanda_extra_hour - 1) + bob_distance_when_met = distance_XY) 
  : 
  (bob_distance_when_met / (bob_distance_when_met / yolanda_rate + yolanda_extra_hour - 1) = yolanda_rate) :=
  sorry

end bobs_walking_rate_l75_75906


namespace find_four_numbers_l75_75505

theorem find_four_numbers (a b c d : ℕ) : 
  a + b + c + d = 45 ∧ (∃ k : ℕ, a + 2 = k ∧ b - 2 = k ∧ 2 * c = k ∧ d / 2 = k) → (a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20) :=
by
  sorry

end find_four_numbers_l75_75505


namespace part1_part2_l75_75480

open Real

def f (x a : ℝ) : ℝ :=
  x^2 + a * x + 3

theorem part1 (x : ℝ) (h : x^2 - 4 * x + 3 < 0) :
  1 < x ∧ x < 3 :=
  sorry

theorem part2 (a : ℝ) (h : ∀ x, f x a > 0) :
  -2 * sqrt 3 < a ∧ a < 2 * sqrt 3 :=
  sorry

end part1_part2_l75_75480


namespace find_g_g2_l75_75753

def g (x : ℝ) : ℝ := 2 * x^3 - 3 * x + 1

theorem find_g_g2 : g (g 2) = 2630 := by
  sorry

end find_g_g2_l75_75753


namespace marks_in_math_l75_75140

theorem marks_in_math (e p c b : ℕ) (avg : ℚ) (n : ℕ) (total_marks_other_subjects : ℚ) :
  e = 45 →
  p = 52 →
  c = 47 →
  b = 55 →
  avg = 46.8 →
  n = 5 →
  total_marks_other_subjects = (e + p + c + b : ℕ) →
  (avg * n) - total_marks_other_subjects = 35 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end marks_in_math_l75_75140


namespace isosceles_triangle_sine_base_angle_l75_75305

theorem isosceles_triangle_sine_base_angle (m : ℝ) (θ : ℝ) 
  (h1 : m > 0)
  (h2 : θ > 0 ∧ θ < π / 2)
  (h_base_height : m * (Real.sin θ) = (m * 2 * (Real.sin θ) * (Real.cos θ))) :
  Real.sin θ = (Real.sqrt 15) / 4 := 
sorry

end isosceles_triangle_sine_base_angle_l75_75305


namespace doughnuts_per_box_l75_75840

theorem doughnuts_per_box
  (total_doughnuts : ℕ)
  (boxes_sold : ℕ)
  (doughnuts_given_away : ℕ)
  (doughnuts_per_box : ℕ)
  (h1 : total_doughnuts = 300)
  (h2 : boxes_sold = 27)
  (h3 : doughnuts_given_away = 30) :
  doughnuts_per_box = (total_doughnuts - doughnuts_given_away) / boxes_sold := by
  -- proof goes here
  sorry

end doughnuts_per_box_l75_75840


namespace sum_even_minus_odd_from_1_to_100_l75_75247

noncomputable def sum_even_numbers : Nat :=
  (List.range' 2 99 2).sum

noncomputable def sum_odd_numbers : Nat :=
  (List.range' 1 100 2).sum

theorem sum_even_minus_odd_from_1_to_100 :
  sum_even_numbers - sum_odd_numbers = 50 :=
by
  sorry

end sum_even_minus_odd_from_1_to_100_l75_75247


namespace automobile_travel_distance_l75_75396

variable (a r : ℝ)

theorem automobile_travel_distance (h : r ≠ 0) :
  (a / 4) * (240 / 1) * (1 / (3 * r)) = (20 * a) / r := 
by
  sorry

end automobile_travel_distance_l75_75396


namespace range_of_m_l75_75089

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x^4) / Real.log 3
noncomputable def g (x : ℝ) : ℝ := x * f x

theorem range_of_m (m : ℝ) : g (1 - m) < g (2 * m) → m > 1 / 3 :=
  by
  sorry

end range_of_m_l75_75089


namespace range_of_a_l75_75733

noncomputable def f (a x : ℝ) : ℝ := Real.sin x + 0.5 * x^2 - a * x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ici 0, f a x ≥ 0) ↔ a ≤ 1 :=
by
  sorry

end range_of_a_l75_75733


namespace triangle_solution_proof_l75_75244

noncomputable def solve_triangle_proof (a b c : ℝ) (alpha beta gamma : ℝ) : Prop :=
  a = 631.28 ∧
  alpha = 63 + 35 / 60 + 30 / 3600 ∧
  b - c = 373 ∧
  beta = 88 + 12 / 60 + 15 / 3600 ∧
  gamma = 28 + 12 / 60 + 15 / 3600 ∧
  b = 704.55 ∧
  c = 331.55

theorem triangle_solution_proof : solve_triangle_proof 631.28 704.55 331.55 (63 + 35 / 60 + 30 / 3600) (88 + 12 / 60 + 15 / 3600) (28 + 12 / 60 + 15 / 3600) :=
  by { sorry }

end triangle_solution_proof_l75_75244


namespace interior_lattice_points_of_triangle_l75_75576

-- Define the vertices of the triangle
def A : (ℤ × ℤ) := (0, 99)
def B : (ℤ × ℤ) := (5, 100)
def C : (ℤ × ℤ) := (2003, 500)

-- The problem is to find the number of interior lattice points
-- according to Pick's Theorem (excluding boundary points).

theorem interior_lattice_points_of_triangle :
  let I : ℤ := 0 -- number of interior lattice points
  I = 0 :=
by
  sorry

end interior_lattice_points_of_triangle_l75_75576


namespace solve_x_l75_75660

variable (x : ℝ)

def vector_a := (2, 1)
def vector_b := (1, x)

def vectors_parallel : Prop :=
  let a_plus_b := (2 + 1, 1 + x)
  let a_minus_b := (2 - 1, 1 - x)
  a_plus_b.1 * a_minus_b.2 = a_plus_b.2 * a_minus_b.1

theorem solve_x (hx : vectors_parallel x) : x = 1/2 := by
  sorry

end solve_x_l75_75660


namespace boys_neither_happy_nor_sad_l75_75968

theorem boys_neither_happy_nor_sad : 
  (∀ children total happy sad neither boys girls happy_boys sad_girls : ℕ,
    total = 60 →
    happy = 30 →
    sad = 10 →
    neither = 20 →
    boys = 19 →
    girls = 41 →
    happy_boys = 6 →
    sad_girls = 4 →
    (boys - (happy_boys + (sad - sad_girls))) = 7) :=
by
  intros children total happy sad neither boys girls happy_boys sad_girls
  sorry

end boys_neither_happy_nor_sad_l75_75968


namespace prime_of_two_pow_sub_one_prime_l75_75807

theorem prime_of_two_pow_sub_one_prime {n : ℕ} (h : Nat.Prime (2^n - 1)) : Nat.Prime n :=
sorry

end prime_of_two_pow_sub_one_prime_l75_75807


namespace base_8_to_base_4_l75_75285

theorem base_8_to_base_4 (n : ℕ) (h : n = 6 * 8^2 + 5 * 8^1 + 3 * 8^0) : 
  (n : ℕ) = 1 * 4^4 + 2 * 4^3 + 2 * 4^2 + 2 * 4^1 + 3 * 4^0 :=
by
  -- Conversion proof goes here
  sorry

end base_8_to_base_4_l75_75285


namespace ratio_problem_l75_75626

theorem ratio_problem
  (A B C : ℚ)
  (h : A / B = 2 / 1)
  (h1 : B / C = 1 / 4) :
  (3 * A + 2 * B) / (4 * C - A) = 4 / 7 := 
sorry

end ratio_problem_l75_75626


namespace line_equation_l75_75436

theorem line_equation (x y : ℝ) (h : ∀ x : ℝ, (x - 2) * 1 = y) : x - y - 2 = 0 :=
sorry

end line_equation_l75_75436


namespace find_interest_rate_l75_75309

theorem find_interest_rate (P r : ℝ) 
  (h1 : 100 = P * (1 + 2 * r)) 
  (h2 : 200 = P * (1 + 6 * r)) : 
  r = 0.5 :=
sorry

end find_interest_rate_l75_75309


namespace inequality_range_l75_75729

theorem inequality_range (a : ℝ) : 
  (∀ x y : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ 2 ≤ y ∧ y ≤ 3 → x * y ≤ a * x^2 + 2 * y^2) ↔ a ≥ -1 := by 
  sorry

end inequality_range_l75_75729


namespace sets_given_to_friend_l75_75500

theorem sets_given_to_friend (total_cards : ℕ) (total_given_away : ℕ) (sets_brother : ℕ) 
  (sets_sister : ℕ) (cards_per_set : ℕ) (sets_friend : ℕ) 
  (h1 : total_cards = 365) 
  (h2 : total_given_away = 195) 
  (h3 : sets_brother = 8) 
  (h4 : sets_sister = 5) 
  (h5 : cards_per_set = 13) 
  (h6 : total_given_away = (sets_brother + sets_sister + sets_friend) * cards_per_set) : 
  sets_friend = 2 :=
by
  sorry

end sets_given_to_friend_l75_75500


namespace sqrt_inequality_l75_75957

open Real

theorem sqrt_inequality (x y z : ℝ) (hx : 1 < x) (hy : 1 < y) (hz : 1 < z) 
  (h : 1 / x + 1 / y + 1 / z = 2) : 
  sqrt (x + y + z) ≥ sqrt (x - 1) + sqrt (y - 1) + sqrt (z - 1) :=
sorry

end sqrt_inequality_l75_75957


namespace decrement_value_is_15_l75_75985

noncomputable def decrement_value (n : ℕ) (original_mean updated_mean : ℕ) : ℕ :=
  (n * original_mean - n * updated_mean) / n

theorem decrement_value_is_15 : decrement_value 50 200 185 = 15 :=
by
  sorry

end decrement_value_is_15_l75_75985


namespace xiao_ming_fails_the_test_probability_l75_75666

def probability_scoring_above_80 : ℝ := 0.69
def probability_scoring_between_70_and_79 : ℝ := 0.15
def probability_scoring_between_60_and_69 : ℝ := 0.09

theorem xiao_ming_fails_the_test_probability :
  1 - (probability_scoring_above_80 + probability_scoring_between_70_and_79 + probability_scoring_between_60_and_69) = 0.07 :=
by
  sorry

end xiao_ming_fails_the_test_probability_l75_75666


namespace find_t_and_m_l75_75971

theorem find_t_and_m 
  (t m : ℝ) 
  (ineq : ∀ x : ℝ, x^2 - 3 * x + t < 0 ↔ 1 < x ∧ x < m) : 
  t = 2 ∧ m = 2 :=
sorry

end find_t_and_m_l75_75971


namespace atomic_weight_Oxygen_l75_75411

theorem atomic_weight_Oxygen :
  ∀ (Ba_atomic_weight S_atomic_weight : ℝ),
    (Ba_atomic_weight = 137.33) →
    (S_atomic_weight = 32.07) →
    (Ba_atomic_weight + S_atomic_weight + 4 * 15.9 = 233) →
    15.9 = 233 - 137.33 - 32.07 / 4 := 
by
  intros Ba_atomic_weight S_atomic_weight hBa hS hm
  sorry

end atomic_weight_Oxygen_l75_75411


namespace digits_difference_l75_75927

/-- Given a two-digit number represented as 10X + Y and the number obtained by interchanging its digits as 10Y + X,
    if the difference between the original number and the interchanged number is 81, 
    then the difference between the tens digit X and the units digit Y is 9. -/
theorem digits_difference (X Y : ℕ) (h : (10 * X + Y) - (10 * Y + X) = 81) : X - Y = 9 :=
by
  sorry

end digits_difference_l75_75927


namespace find_k_l75_75898

theorem find_k (m n k : ℝ) 
  (h1 : 3^m = k) 
  (h2 : 5^n = k) 
  (h3 : 1/m + 1/n = 2) : 
  k = Real.sqrt 15 := 
sorry

end find_k_l75_75898


namespace benny_lost_books_l75_75450

-- Define the initial conditions
def sandy_books : ℕ := 10
def tim_books : ℕ := 33
def total_books : ℕ := sandy_books + tim_books
def remaining_books : ℕ := 19

-- Define the proof problem to find out the number of books Benny lost
theorem benny_lost_books : total_books - remaining_books = 24 :=
by
  sorry -- Insert proof here

end benny_lost_books_l75_75450


namespace quadratic_graph_above_x_axis_l75_75811

theorem quadratic_graph_above_x_axis (a b c : ℝ) :
  ¬ ((b^2 - 4*a*c < 0) ↔ ∀ x : ℝ, a*x^2 + b*x + c > 0) :=
sorry

end quadratic_graph_above_x_axis_l75_75811


namespace greatest_number_of_large_chips_l75_75787

theorem greatest_number_of_large_chips (s l p : ℕ) (h1 : s + l = 60) (h2 : s = l + p) 
  (hp_prime : Nat.Prime p) (hp_div : p ∣ l) : l ≤ 29 :=
by
  sorry

end greatest_number_of_large_chips_l75_75787


namespace ellipse_hyperbola_proof_l75_75846

noncomputable def ellipse_and_hyperbola_condition (a b : ℝ) : Prop :=
  (a > b ∧ b > 0) ∧ (a^2 - b^2 = 5) ∧ (a^2 = 11 * b^2)

theorem ellipse_hyperbola_proof : 
  ∀ (a b : ℝ), ellipse_and_hyperbola_condition a b → b^2 = 0.5 :=
by
  intros a b h
  sorry

end ellipse_hyperbola_proof_l75_75846


namespace algebraic_expression_result_l75_75745

theorem algebraic_expression_result (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2 * x^2 - 12 = -11 :=
by
  sorry

end algebraic_expression_result_l75_75745


namespace cone_altitude_ratio_l75_75688

variable (r h : ℝ)
variable (radius_condition : r > 0)
variable (volume_condition : (1 / 3) * π * r^2 * h = (1 / 3) * (4 / 3) * π * r^3)

theorem cone_altitude_ratio {r h : ℝ}
  (radius_condition : r > 0) 
  (volume_condition : (1 / 3) * π * r^2 * h = (1 / 3) * (4 / 3) * π * r^3) : 
  h / r = 4 / 3 := by
  sorry

end cone_altitude_ratio_l75_75688


namespace probability_correct_arrangement_l75_75342

-- Definitions for conditions
def characters := {c : String | c = "医" ∨ c = "国"}

def valid_arrangements : Finset (List String) := 
    {["医", "医", "国"], ["医", "国", "医"], ["国", "医", "医"]}

def correct_arrangements : Finset (List String) := 
    {["医", "医", "国"], ["医", "国", "医"]}

-- Theorem statement
theorem probability_correct_arrangement :
  (correct_arrangements.card : ℚ) / (valid_arrangements.card : ℚ) = 1 / 3 :=
by
  sorry

end probability_correct_arrangement_l75_75342


namespace solve_r_l75_75232

theorem solve_r (k r : ℝ) (h1 : 3 = k * 2^r) (h2 : 15 = k * 4^r) : 
  r = Real.log 5 / Real.log 2 := 
sorry

end solve_r_l75_75232


namespace gideon_fraction_of_marbles_l75_75113

variable (f : ℝ)

theorem gideon_fraction_of_marbles (marbles : ℝ) (age_now : ℝ) (age_future : ℝ) (remaining_marbles : ℝ) (future_age_with_remaining_marbles : Bool)
  (h1 : marbles = 100)
  (h2 : age_now = 45)
  (h3 : age_future = age_now + 5)
  (h4 : remaining_marbles = 2 * (1 - f) * marbles)
  (h5 : remaining_marbles = age_future)
  (h6 : future_age_with_remaining_marbles = (age_future = 50)) :
  f = 3 / 4 :=
by
  sorry

end gideon_fraction_of_marbles_l75_75113


namespace train_length_l75_75924

open Real

/--
A train of a certain length can cross an electric pole in 30 sec with a speed of 43.2 km/h.
Prove that the length of the train is 360 meters.
-/
theorem train_length (t : ℝ) (v_kmh : ℝ) (length : ℝ) 
  (h_time : t = 30) 
  (h_speed_kmh : v_kmh = 43.2) 
  (h_length : length = v_kmh * (t * (1000 / 3600))) : 
  length = 360 := 
by
  -- skip the actual proof steps
  sorry

end train_length_l75_75924


namespace percentage_increase_l75_75402

variable (m y : ℝ)

theorem percentage_increase (h : x = y + (m / 100) * y) : x = ((100 + m) / 100) * y := by
  sorry

end percentage_increase_l75_75402


namespace real_roots_prime_equation_l75_75528

noncomputable def has_rational_roots (p q : ℕ) : Prop :=
  ∃ x : ℚ, x^2 + p^2 * x + q^3 = 0

theorem real_roots_prime_equation (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  has_rational_roots p q ↔ (p = 3 ∧ q = 2) :=
sorry

end real_roots_prime_equation_l75_75528


namespace min_a_l75_75648

theorem min_a (a : ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → (x + y) * (1/x + a/y) ≥ 25) : a ≥ 16 :=
sorry  -- Proof is omitted

end min_a_l75_75648


namespace prism_volume_l75_75941

theorem prism_volume (x y z : ℝ) 
  (h1 : x * y = 24) 
  (h2 : x * z = 32) 
  (h3 : y * z = 48) : 
  x * y * z = 192 :=
by
  sorry

end prism_volume_l75_75941


namespace no_nat_triplet_exists_l75_75919

theorem no_nat_triplet_exists (x y z : ℕ) : ¬ (x ^ 2 + y ^ 2 = 7 * z ^ 2) := 
sorry

end no_nat_triplet_exists_l75_75919


namespace domain_of_log_function_l75_75586

noncomputable def domain_f (k : ℤ) : Set ℝ :=
  {x : ℝ | (2 * k * Real.pi - Real.pi / 3 < x ∧ x < 2 * k * Real.pi + Real.pi / 3) ∨
           (2 * k * Real.pi + 2 * Real.pi / 3 < x ∧ x < 2 * k * Real.pi + 4 * Real.pi / 3)}

theorem domain_of_log_function :
  ∀ x : ℝ, (∃ k : ℤ, (2 * k * Real.pi - Real.pi / 3 < x ∧ x < 2 * k * Real.pi + Real.pi / 3) ∨
                      (2 * k * Real.pi + 2 * Real.pi / 3 < x ∧ x < 2 * k * Real.pi + 4 * Real.pi / 3))
  ↔ (3 - 4 * Real.sin x ^ 2 > 0) :=
by {
  sorry
}

end domain_of_log_function_l75_75586


namespace solve_digits_l75_75731

variables (h t u : ℕ)

theorem solve_digits :
  (u = h + 6) →
  (u + h = 16) →
  (∀ (x y z : ℕ), 100 * h + 10 * t + u + 100 * u + 10 * t + h = 100 * x + 10 * y + z ∧ y = 9 ∧ z = 6) →
  (h = 5 ∧ t = 4 ∧ u = 11) :=
sorry

end solve_digits_l75_75731


namespace volume_of_prism_l75_75550

variable (x y z : ℝ)
variable (h1 : x * y = 15)
variable (h2 : y * z = 10)
variable (h3 : x * z = 6)

theorem volume_of_prism : x * y * z = 30 :=
by
  sorry

end volume_of_prism_l75_75550


namespace max_area_of_garden_l75_75615

theorem max_area_of_garden (total_fence : ℝ) (gate : ℝ) (remaining_fence := total_fence - gate) :
  total_fence = 60 → gate = 4 → (remaining_fence / 2) * (remaining_fence / 2) = 196 :=
by 
  sorry

end max_area_of_garden_l75_75615


namespace boat_stream_ratio_l75_75230

theorem boat_stream_ratio (B S : ℝ) (h : 1 / (B - S) = 2 * (1 / (B + S))) : B / S = 3 :=
by
  sorry

end boat_stream_ratio_l75_75230


namespace sin_75_eq_sqrt6_add_sqrt2_div4_l75_75212

theorem sin_75_eq_sqrt6_add_sqrt2_div4 :
  Real.sin (75 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
sorry

end sin_75_eq_sqrt6_add_sqrt2_div4_l75_75212


namespace point_on_x_axis_coord_l75_75891

theorem point_on_x_axis_coord (m : ℝ) (h : (m - 1, 2 * m).snd = 0) : (m - 1, 2 * m) = (-1, 0) :=
by
  sorry

end point_on_x_axis_coord_l75_75891


namespace cos_probability_ge_one_half_in_range_l75_75993

theorem cos_probability_ge_one_half_in_range :
  let interval_length := (Real.pi / 2) - (- (Real.pi / 2))
  let favorable_length := (Real.pi / 3) - (- (Real.pi / 3))
  (favorable_length / interval_length) = (2 / 3) := by
  sorry

end cos_probability_ge_one_half_in_range_l75_75993


namespace segment_proportionality_l75_75510

variable (a b c x : ℝ)

theorem segment_proportionality (ha : a ≠ 0) (hc : c ≠ 0) 
  (h : x = a * (b / c)) : 
  (x / a) = (b / c) := 
by
  sorry

end segment_proportionality_l75_75510


namespace side_length_square_l75_75350

theorem side_length_square (s : ℝ) (h1 : ∃ (s : ℝ), (s > 0)) (h2 : 6 * s^2 = 3456) : s = 24 :=
sorry

end side_length_square_l75_75350


namespace find_a_b_find_m_l75_75548

-- Define the parabola and the points it passes through
def parabola (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- The conditions based on the given problem
def condition1 (a b : ℝ) : Prop := parabola a b 1 = -2
def condition2 (a b : ℝ) : Prop := parabola a b (-2) = 13

-- Part 1: Proof for a and b
theorem find_a_b : ∃ a b : ℝ, condition1 a b ∧ condition2 a b ∧ a = 1 ∧ b = -4 :=
by sorry

-- Part 2: Given y equation and the specific points
def parabola2 (x : ℝ) : ℝ := x^2 - 4 * x + 1

-- Conditions for the second part
def condition3 : Prop := parabola2 5 = 6
def condition4 (m : ℝ) : Prop := parabola2 m = 12 - 6

-- Theorem statement for the second part
theorem find_m : ∃ m : ℝ, condition3 ∧ condition4 m ∧ m = -1 :=
by sorry

end find_a_b_find_m_l75_75548


namespace quadratic_inequality_solution_l75_75693

theorem quadratic_inequality_solution (a b c : ℝ) (h_solution_set : ∀ x, ax^2 + bx + c < 0 ↔ x < -1 ∨ x > 3) :
  (a < 0) ∧
  (a + b + c > 0) ∧
  (∀ x, cx^2 - bx + a < 0 ↔ -1/3 < x ∧ x < 1) :=
by
  sorry

end quadratic_inequality_solution_l75_75693


namespace sum_of_x_and_y_l75_75295

theorem sum_of_x_and_y (x y : ℝ) 
  (h₁ : |x| + x + 5 * y = 2)
  (h₂ : |y| - y + x = 7) : 
  x + y = 3 := 
sorry

end sum_of_x_and_y_l75_75295


namespace sequence_k_value_l75_75705

theorem sequence_k_value
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ m n : ℕ, a (m + n) = a m * a n)
  (hk1 : ∀ k : ℕ, a (k + 1) = 1024) :
  ∃ k : ℕ, k = 9 :=
by {
  sorry
}

end sequence_k_value_l75_75705


namespace average_value_of_T_l75_75717

def average_T (boys girls : ℕ) (starts_with_boy : Bool) (ends_with_girl : Bool) : ℕ :=
  if boys = 9 ∧ girls = 15 ∧ starts_with_boy ∧ ends_with_girl then 12 else 0

theorem average_value_of_T :
  average_T 9 15 true true = 12 :=
sorry

end average_value_of_T_l75_75717


namespace blue_chip_value_l75_75630

noncomputable def yellow_chip_value := 2
noncomputable def green_chip_value := 5
noncomputable def total_product_value := 16000
noncomputable def num_yellow_chips := 4

def blue_chip_points (b n : ℕ) :=
  yellow_chip_value ^ num_yellow_chips * b ^ n * green_chip_value ^ n = total_product_value

theorem blue_chip_value (b : ℕ) (n : ℕ) (h : blue_chip_points b n) (hn : b^n = 8) : b = 8 :=
by
  have h1 : ∀ k : ℕ, k ^ n = 8 → k = 8 ∧ n = 3 := sorry
  exact (h1 b hn).1

end blue_chip_value_l75_75630


namespace bacon_needed_l75_75103

def eggs_per_plate : ℕ := 2
def bacon_per_plate : ℕ := 2 * eggs_per_plate
def customers : ℕ := 14
def bacon_total (eggs_per_plate bacon_per_plate customers : ℕ) : ℕ := customers * bacon_per_plate

theorem bacon_needed : bacon_total eggs_per_plate bacon_per_plate customers = 56 :=
by
  sorry

end bacon_needed_l75_75103


namespace positive_distinct_solutions_conditons_l75_75214

-- Definitions corresponding to the conditions in the problem
variables {x y z a b : ℝ}

-- The statement articulates the condition
theorem positive_distinct_solutions_conditons (h1 : x + y + z = a) (h2 : x^2 + y^2 + z^2 = b^2) (h3 : xy = z^2) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0) (h7 : x ≠ y) (h8 : y ≠ z) (h9 : x ≠ z) : 
  b^2 ≥ a^2 / 2 :=
sorry

end positive_distinct_solutions_conditons_l75_75214


namespace actual_average_speed_l75_75738

variable {t : ℝ} (h₁ : t > 0) -- ensure that time is positive
variable {v : ℝ} 

theorem actual_average_speed (h₂ : v > 0)
  (h3 : v * t = (v + 12) * (3 / 4 * t)) : v = 36 :=
by
  sorry

end actual_average_speed_l75_75738


namespace hypotenuse_length_l75_75862

theorem hypotenuse_length (x y h : ℝ)
  (hx : (1 / 3) * π * y * x^2 = 1620 * π)
  (hy : (1 / 3) * π * x * y^2 = 3240 * π) :
  h = Real.sqrt 507 :=
by
  sorry

end hypotenuse_length_l75_75862


namespace numValidRoutesJackToJill_l75_75976

noncomputable def numPaths (n m : ℕ) : ℕ :=
  Nat.choose (n + m) n

theorem numValidRoutesJackToJill : 
  let totalRoutes := numPaths 5 3
  let pathsViaDanger := numPaths 2 2 * numPaths 3 1
  totalRoutes - pathsViaDanger = 32 :=
by
  let totalRoutes := numPaths 5 3
  let pathsViaDanger := numPaths 2 2 * numPaths 3 1
  show totalRoutes - pathsViaDanger = 32
  sorry

end numValidRoutesJackToJill_l75_75976


namespace students_didnt_make_cut_l75_75012

theorem students_didnt_make_cut (g b c : ℕ) (hg : g = 15) (hb : b = 25) (hc : c = 7) : g + b - c = 33 := by
  sorry

end students_didnt_make_cut_l75_75012


namespace sufficient_but_not_necessary_l75_75944

theorem sufficient_but_not_necessary (x y : ℝ) :
  (x + y = 1 → xy ≤ 1 / 4) ∧ (∃ x y : ℝ, xy ≤ 1 / 4 ∧ x + y ≠ 1) := by
  sorry

end sufficient_but_not_necessary_l75_75944


namespace Alyssa_spent_in_total_l75_75217

def amount_paid_for_grapes : ℝ := 12.08
def refund_for_cherries : ℝ := 9.85
def total_spent : ℝ := amount_paid_for_grapes - refund_for_cherries

theorem Alyssa_spent_in_total : total_spent = 2.23 := by
  sorry

end Alyssa_spent_in_total_l75_75217


namespace arithmetic_sequence_n_l75_75943

theorem arithmetic_sequence_n (a_n : ℕ → ℕ) (S_n : ℕ) (n : ℕ) 
  (h1 : ∀ i, a_n i = 20 + (i - 1) * (54 - 20) / (n - 1)) 
  (h2 : S_n = 37 * n) 
  (h3 : S_n = 999) : 
  n = 27 :=
by sorry

end arithmetic_sequence_n_l75_75943


namespace time_for_first_three_workers_l75_75712

def work_rate_equations (a b c d : ℝ) :=
  a + b + c + d = 1 / 6 ∧
  2 * a + (1 / 2) * b + c + d = 1 / 6 ∧
  (1 / 2) * a + 2 * b + c + d = 1 / 4

theorem time_for_first_three_workers (a b c d : ℝ) (h : work_rate_equations a b c d) :
  (a + b + c) * 6 = 1 :=
sorry

end time_for_first_three_workers_l75_75712


namespace carbon_emission_l75_75220

theorem carbon_emission (x y : ℕ) (h1 : x + y = 70) (h2 : x = 5 * y - 8) : y = 13 ∧ x = 57 := by
  sorry

end carbon_emission_l75_75220


namespace total_people_3522_l75_75813

def total_people (M W: ℕ) : ℕ := M + W

theorem total_people_3522 
    (M W: ℕ) 
    (h1: M / 9 * 45 + W / 12 * 60 = 17760)
    (h2: M % 9 = 0)
    (h3: W % 12 = 0) : 
    total_people M W = 3552 :=
by {
  sorry
}

end total_people_3522_l75_75813


namespace red_light_probability_l75_75014

theorem red_light_probability :
  let red_duration := 30
  let yellow_duration := 5
  let green_duration := 40
  let total_duration := red_duration + yellow_duration + green_duration
  let probability_of_red := (red_duration:ℝ) / total_duration
  probability_of_red = 2 / 5 := by
    sorry

end red_light_probability_l75_75014


namespace distance_to_intersection_of_quarter_circles_eq_zero_l75_75561

open Real

theorem distance_to_intersection_of_quarter_circles_eq_zero (s : ℝ) :
  let A := (0, 0)
  let B := (s, 0)
  let C := (s, s)
  let D := (0, s)
  let center := (s / 2, s / 2)
  let arc_from_A := {p : ℝ × ℝ | p.1^2 + p.2^2 = s^2}
  let arc_from_C := {p : ℝ × ℝ | (p.1 - s)^2 + (p.2 - s)^2 = s^2}
  (center ∈ arc_from_A ∧ center ∈ arc_from_C) →
  let (ix, iy) := (s / 2, s / 2)
  dist (ix, iy) center = 0 :=
by
  sorry

end distance_to_intersection_of_quarter_circles_eq_zero_l75_75561


namespace correct_expression_l75_75582

variable (a b : ℝ)

theorem correct_expression : (∃ x, x = 3 * a + b^2) ∧ 
    (x = (3 * a + b)^2 ∨ x = 3 * (a + b)^2 ∨ x = 3 * a + b^2 ∨ x = (a + 3 * b)^2) → 
    x = 3 * a + b^2 := by sorry

end correct_expression_l75_75582


namespace angle_cosine_third_quadrant_l75_75701

theorem angle_cosine_third_quadrant (B : ℝ) (h1 : π < B ∧ B < 3 * π / 2) (h2 : Real.sin B = 4 / 5) :
  Real.cos B = -3 / 5 :=
sorry

end angle_cosine_third_quadrant_l75_75701


namespace area_ratio_l75_75502

noncomputable def AreaOfTrapezoid (AD BC : ℝ) (R : ℝ) : ℝ :=
  let s_π := Real.pi
  let height1 := 2 -- One of the heights considered
  let height2 := 14 -- Another height considered
  (AD + BC) / 2 * height1  -- First case area
  -- Here we assume the area uses sine which is arc-related, but provide fixed coefficients for area representation

noncomputable def AreaOfRectangle (R : ℝ) : ℝ :=
  let d := 2 * R
  -- Using the equation for area discussed
  d * d / 2

theorem area_ratio (AD BC : ℝ) (R : ℝ) (hAD : AD = 16) (hBC : BC = 12) (hR : R = 10) :
  let area_trap := AreaOfTrapezoid AD BC R
  let area_rect := AreaOfRectangle R
  area_trap / area_rect = 1 / 2 ∨ area_trap / area_rect = 49 / 50 :=
by
  sorry

end area_ratio_l75_75502


namespace students_play_both_l75_75117

def students_total : ℕ := 25
def hockey_players : ℕ := 15
def basketball_players : ℕ := 16
def neither_players : ℕ := 4

theorem students_play_both : 
  (students_total - neither_players) + (hockey_players + basketball_players - students_total + neither_players - students_total) = 10 :=
by 
  sorry

end students_play_both_l75_75117


namespace max_b_value_l75_75192

theorem max_b_value
  (a b c : ℕ)
  (h1 : 1 < c)
  (h2 : c < b)
  (h3 : b < a)
  (h4 : a * b * c = 240) : b = 10 :=
  sorry

end max_b_value_l75_75192


namespace bottles_per_day_l75_75774

theorem bottles_per_day (b d : ℕ) (h1 : b = 8066) (h2 : d = 74) : b / d = 109 :=
by {
  sorry
}

end bottles_per_day_l75_75774


namespace line_through_intersections_of_circles_l75_75839

-- Define the first circle
def circle₁ (x y : ℝ) : Prop :=
  x^2 + y^2 = 10

-- Define the second circle
def circle₂ (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 20

-- The statement of the mathematically equivalent proof problem
theorem line_through_intersections_of_circles : 
    (∃ (x y : ℝ), circle₁ x y ∧ circle₂ x y) → (∃ (x y : ℝ), x + 3 * y - 5 = 0) :=
by
  intro h
  sorry

end line_through_intersections_of_circles_l75_75839


namespace intersection_with_y_axis_l75_75032

theorem intersection_with_y_axis :
  ∃ (y : ℝ), (y = -x^2 + 3*x - 4) ∧ (x = 0) ∧ (y = -4) := 
by
  sorry

end intersection_with_y_axis_l75_75032


namespace correct_regression_line_l75_75850

theorem correct_regression_line (h_neg_corr: ∀ x: ℝ, ∀ y: ℝ, y = -10*x + 200 ∨ y = 10*x + 200 ∨ y = -10*x - 200 ∨ y = 10*x - 200) 
(h_slope_neg : ∀ a b: ℝ, a < 0) 
(h_y_intercept: ∀ x: ℝ, x = 0 → 200 > 0 → y = 200) : 
∃ y: ℝ, y = -10*x + 200 :=
by
-- the proof will go here
sorry

end correct_regression_line_l75_75850


namespace length_of_hypotenuse_l75_75525

theorem length_of_hypotenuse (a b : ℝ) (h1 : a = 15) (h2 : b = 21) : 
hypotenuse_length = Real.sqrt (a^2 + b^2) :=
by
  rw [h1, h2]
  sorry

end length_of_hypotenuse_l75_75525


namespace first_player_wins_l75_75935

-- Define the game state and requirements
inductive Player
| first : Player
| second : Player

-- Game state consists of a number of stones and whose turn it is
structure GameState where
  stones : Nat
  player : Player

-- Define a simple transition for the game
def take_stones (s : GameState) (n : Nat) : GameState :=
  { s with stones := s.stones - n, player := Player.second }

-- Determine if a player can take n stones
def can_take (s : GameState) (n : Nat) : Prop :=
  n >= 1 ∧ n <= 4 ∧ n <= s.stones

-- Define victory condition
def wins (s : GameState) : Prop :=
  s.stones = 0 ∧ s.player = Player.second

-- Prove that if the first player starts with 18 stones and picks 3 stones initially,
-- they can ensure victory
theorem first_player_wins :
  ∀ (s : GameState),
    s.stones = 18 ∧ s.player = Player.first →
    can_take s 3 →
    wins (take_stones s 3)
:= by
  sorry

end first_player_wins_l75_75935


namespace polynomial_remainder_l75_75363

theorem polynomial_remainder (x : ℤ) :
  let poly := x^5 + 3*x^3 + 1
  let divisor := (x + 1)^2
  let remainder := 5*x + 9
  ∃ q : ℤ, poly = divisor * q + remainder := by
  sorry

end polynomial_remainder_l75_75363


namespace contrapositive_proposition_l75_75345

theorem contrapositive_proposition (α : ℝ) :
  (α = π / 4 → Real.tan α = 1) ↔ (Real.tan α ≠ 1 → α ≠ π / 4) :=
by
  sorry

end contrapositive_proposition_l75_75345


namespace total_age_difference_is_twelve_l75_75539

variable {A B C : ℕ}

theorem total_age_difference_is_twelve (h1 : A + B > B + C) (h2 : C = A - 12) :
  (A + B) - (B + C) = 12 :=
by
  sorry

end total_age_difference_is_twelve_l75_75539


namespace range_of_a_l75_75324

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬((a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0)) → (-2 < a ∧ a ≤ 6/5) :=
by
  sorry

end range_of_a_l75_75324


namespace fish_in_third_tank_l75_75304

-- Definitions of the conditions
def first_tank_goldfish : ℕ := 7
def first_tank_beta_fish : ℕ := 8
def first_tank_fish : ℕ := first_tank_goldfish + first_tank_beta_fish

def second_tank_fish : ℕ := 2 * first_tank_fish

def third_tank_fish : ℕ := second_tank_fish / 3

-- The statement to prove
theorem fish_in_third_tank : third_tank_fish = 10 := by
  sorry

end fish_in_third_tank_l75_75304


namespace sum_of_coordinates_is_17_over_3_l75_75975

theorem sum_of_coordinates_is_17_over_3
  (f : ℝ → ℝ)
  (h1 : 5 = 3 * f 2) :
  (5 / 3 + 4) = 17 / 3 :=
by
  have h2 : f 2 = 5 / 3 := by
    linarith
  have h3 : f⁻¹ (5 / 3) = 2 := by
    sorry -- we do not know more properties of f to conclude this proof step
  have h4 : 2 * f⁻¹ (5 / 3) = 4 := by
    sorry -- similarly, assume for now the desired property
  exact sorry -- finally putting everything together

end sum_of_coordinates_is_17_over_3_l75_75975


namespace pascals_triangle_53_rows_l75_75808

theorem pascals_triangle_53_rows : 
  ∃! row, (∃ k, 1 ≤ k ∧ k ≤ row ∧ 53 = Nat.choose row k) ∧ 
          (∀ k, 1 ≤ k ∧ k ≤ row → 53 = Nat.choose row k → row = 53) :=
sorry

end pascals_triangle_53_rows_l75_75808


namespace chromosome_structure_l75_75488

-- Definitions related to the conditions of the problem
def chromosome : Type := sorry  -- Define type for chromosome (hypothetical representation)
def has_centromere (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome has centromere
def contains_one_centromere (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome contains one centromere
def has_one_chromatid (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome has one chromatid
def has_two_chromatids (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome has two chromatids
def is_chromatin (c : chromosome) : Prop := sorry  -- Predicate indicating a chromosome is chromatin

-- Define the problem statement
theorem chromosome_structure (c : chromosome) :
  contains_one_centromere c ∧ ¬has_one_chromatid c ∧ ¬has_two_chromatids c ∧ ¬is_chromatin c := sorry

end chromosome_structure_l75_75488


namespace warriors_won_40_games_l75_75782

variable (H F W K R S : ℕ)

-- Conditions as given in the problem
axiom hawks_won_more_games_than_falcons : H > F
axiom knights_won_more_than_30 : K > 30
axiom warriors_won_more_than_knights_but_fewer_than_royals : W > K ∧ W < R
axiom squires_tied_with_falcons : S = F

-- The proof statement
theorem warriors_won_40_games : W = 40 :=
sorry

end warriors_won_40_games_l75_75782


namespace calculate_expression_l75_75344

theorem calculate_expression : 7 + 15 / 3 - 5 * 2 = 2 :=
by sorry

end calculate_expression_l75_75344


namespace students_in_neither_l75_75267

def total_students := 60
def students_in_art := 40
def students_in_music := 30
def students_in_both := 15

theorem students_in_neither : total_students - (students_in_art - students_in_both + students_in_music - students_in_both + students_in_both) = 5 :=
by
  sorry

end students_in_neither_l75_75267


namespace number_of_blue_socks_l75_75947

theorem number_of_blue_socks (x : ℕ) (h : ((6 + x ^ 2 - x) / ((6 + x) * (5 + x)) = 1/5)) : x = 4 := 
sorry

end number_of_blue_socks_l75_75947


namespace total_boxes_is_4575_l75_75206

-- Define the number of boxes in each warehouse
def num_boxes_in_warehouse_A (x : ℕ) := x
def num_boxes_in_warehouse_B (x : ℕ) := 3 * x
def num_boxes_in_warehouse_C (x : ℕ) := (3 * x) / 2 + 100
def num_boxes_in_warehouse_D (x : ℕ) := 2 * ((3 * x) / 2 + 100) - 50
def num_boxes_in_warehouse_E (x : ℕ) := x + (2 * ((3 * x) / 2 + 100) - 50) - 200

-- Define the condition that warehouse B has 300 more boxes than warehouse E
def condition_B_E (x : ℕ) := 3 * x = num_boxes_in_warehouse_E x + 300

-- Define the total number of boxes calculation
def total_boxes (x : ℕ) := 
    num_boxes_in_warehouse_A x +
    num_boxes_in_warehouse_B x +
    num_boxes_in_warehouse_C x +
    num_boxes_in_warehouse_D x +
    num_boxes_in_warehouse_E x

-- The statement of the problem
theorem total_boxes_is_4575 (x : ℕ) (h : condition_B_E x) : total_boxes x = 4575 :=
by
    sorry

end total_boxes_is_4575_l75_75206


namespace number_of_real_roots_l75_75759

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then 2010 * x + Real.log x / Real.log 2010
  else if x < 0 then - (2010 * (-x) + Real.log (-x) / Real.log 2010)
  else 0

theorem number_of_real_roots : 
  (∃ x1 x2 x3 : ℝ, 
    f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ 
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) ∧ 
    ∀ x y z : ℝ, 
    (f x = 0 ∧ f y = 0 ∧ f z = 0 → 
    (x = y ∨ x = z ∨ y = z)) 
  :=
by
  sorry

end number_of_real_roots_l75_75759


namespace find_multiplier_l75_75845

theorem find_multiplier (x : ℝ) (y : ℝ) (h1 : x = 62.5) (h2 : (y * (x + 5)) / 5 - 5 = 22) : y = 2 :=
sorry

end find_multiplier_l75_75845


namespace average_of_pqrs_l75_75860

theorem average_of_pqrs (p q r s : ℚ) (h : (5/4) * (p + q + r + s) = 20) : ((p + q + r + s) / 4) = 4 :=
sorry

end average_of_pqrs_l75_75860


namespace dawn_monthly_payments_l75_75317

theorem dawn_monthly_payments (annual_salary : ℕ) (saved_per_month : ℕ)
  (h₁ : annual_salary = 48000)
  (h₂ : saved_per_month = 400)
  (h₃ : ∀ (monthly_salary : ℕ), saved_per_month = (10 * monthly_salary) / 100):
  annual_salary / saved_per_month = 12 :=
by
  sorry

end dawn_monthly_payments_l75_75317


namespace collinear_points_x_value_l75_75892

theorem collinear_points_x_value :
  (∀ A B C : ℝ × ℝ, A = (-1, 1) → B = (2, -4) → C = (x, -9) → 
                    (∃ x : ℝ, x = 5)) :=
by sorry

end collinear_points_x_value_l75_75892


namespace max_M_range_a_l75_75069

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a / x + x * Real.log x
def g (x : ℝ) : ℝ := x^3 - x^2 - 3

theorem max_M (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : x1 ≤ 2) (h3 : 0 ≤ x2) (h4 : x2 ≤ 2) : 
  4 ≤ g x1 - g x2 :=
sorry

theorem range_a (a : ℝ) (s t : ℝ) (h1 : 1 / 2 ≤ s) (h2 : s ≤ 2) (h3 : 1 / 2 ≤ t) (h4 : t ≤ 2) : 
  1 ≤ a ∧ f s a ≥ g t :=
sorry

end max_M_range_a_l75_75069


namespace determine_abc_l75_75405

theorem determine_abc (a b c : ℕ) (h1 : a * b * c = 2^4 * 3^2 * 5^3) 
  (h2 : gcd a b = 15) (h3 : gcd a c = 5) (h4 : gcd b c = 20) : 
  a = 15 ∧ b = 60 ∧ c = 20 :=
by
  sorry

end determine_abc_l75_75405


namespace dance_contradiction_l75_75179

variable {Boy Girl : Type}
variable {danced_with : Boy → Girl → Prop}

theorem dance_contradiction
    (H1 : ¬ ∃ g : Boy, ∀ f : Girl, danced_with g f)
    (H2 : ∀ f : Girl, ∃ g : Boy, danced_with g f) :
    ∃ (g g' : Boy) (f f' : Girl),
        danced_with g f ∧ ¬ danced_with g f' ∧
        danced_with g' f' ∧ ¬ danced_with g' f :=
by
  -- Proof will be inserted here
  sorry

end dance_contradiction_l75_75179


namespace domain_f_x_plus_2_l75_75028

-- Define the function f and its properties
variable (f : ℝ → ℝ)

-- Define the given condition: the domain of y = f(2x - 3) is [-2, 3]
def domain_f_2x_minus_3 : Set ℝ :=
  {x | -2 ≤ x ∧ x ≤ 3}

-- Express this condition formally
axiom domain_f_2x_minus_3_axiom :
  ∀ (x : ℝ), (x ∈ domain_f_2x_minus_3) → (2 * x - 3 ∈ Set.Icc (-7 : ℝ) 3)

-- Prove the desired result: the domain of y = f(x + 2) is [-9, 1]
theorem domain_f_x_plus_2 :
  ∀ (x : ℝ), (x ∈ Set.Icc (-9 : ℝ) 1) ↔ ((x + 2) ∈ Set.Icc (-7 : ℝ) 3) :=
sorry

end domain_f_x_plus_2_l75_75028


namespace fraction_value_l75_75901

theorem fraction_value
  (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x^2 - 2*x + 1) / (x^2 - 1) = 1 - Real.sqrt 2 :=
by
  sorry

end fraction_value_l75_75901


namespace ratio_yellow_jelly_beans_l75_75215

theorem ratio_yellow_jelly_beans :
  let bag_A_total := 24
  let bag_B_total := 30
  let bag_C_total := 32
  let bag_D_total := 34
  let bag_A_yellow_ratio := 0.40
  let bag_B_yellow_ratio := 0.30
  let bag_C_yellow_ratio := 0.25 
  let bag_D_yellow_ratio := 0.10
  let bag_A_yellow := bag_A_total * bag_A_yellow_ratio
  let bag_B_yellow := bag_B_total * bag_B_yellow_ratio
  let bag_C_yellow := bag_C_total * bag_C_yellow_ratio
  let bag_D_yellow := bag_D_total * bag_D_yellow_ratio
  let total_yellow := bag_A_yellow + bag_B_yellow + bag_C_yellow + bag_D_yellow
  let total_beans := bag_A_total + bag_B_total + bag_C_total + bag_D_total
  (total_yellow / total_beans) = 0.25 := by
  sorry

end ratio_yellow_jelly_beans_l75_75215


namespace comb_n_plus_1_2_l75_75021

theorem comb_n_plus_1_2 (n : ℕ) (h : 0 < n) : 
  (n + 1).choose 2 = (n + 1) * n / 2 :=
by sorry

end comb_n_plus_1_2_l75_75021


namespace find_D_l75_75111

-- We define the points E and F
def E : ℝ × ℝ := (-3, -2)
def F : ℝ × ℝ := (5, 10)

-- Definition of point D with the given conditions
def D : ℝ × ℝ := (3, 7)

-- We state the main theorem to prove that D is such that ED = 2 * DF given E and F
theorem find_D (D : ℝ × ℝ) (ED_DF_relation : dist E D = 2 * dist D F) : D = (3, 7) :=
sorry

end find_D_l75_75111


namespace car_bus_washing_inconsistency_l75_75023

theorem car_bus_washing_inconsistency :
  ∀ (C B : ℕ), 
    C % 2 = 0 →
    B % 2 = 1 →
    7 * C + 18 * B = 309 →
    3 + 8 + 5 + C + B = 15 →
    false :=
by
  sorry

end car_bus_washing_inconsistency_l75_75023


namespace find_point_B_l75_75339

def line_segment_parallel_to_x_axis (A B : (ℝ × ℝ)) : Prop :=
  A.snd = B.snd

def length_3 (A B : (ℝ × ℝ)) : Prop :=
  abs (A.fst - B.fst) = 3

theorem find_point_B (A B : (ℝ × ℝ))
  (h₁ : A = (3, 2))
  (h₂ : line_segment_parallel_to_x_axis A B)
  (h₃ : length_3 A B) :
  B = (0, 2) ∨ B = (6, 2) :=
sorry

end find_point_B_l75_75339


namespace range_of_m_l75_75352

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem range_of_m:
  ∀ m : ℝ, 
  (∀ x, 0 ≤ x ∧ x ≤ m → f x ≤ -3) ∧ 
  (∃ x, 0 ≤ x ∧ x ≤ m ∧ f x = -4) → 
  1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_l75_75352


namespace slope_of_line_l75_75826

theorem slope_of_line (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 3 / x + 4 / y = 0) : 
  ∃ m : ℝ, m = -4 / 3 := 
sorry

end slope_of_line_l75_75826


namespace find_principal_l75_75644

variable (P R : ℝ)
variable (condition1 : P + (P * R * 2) / 100 = 660)
variable (condition2 : P + (P * R * 7) / 100 = 1020)

theorem find_principal : P = 516 := by
  sorry

end find_principal_l75_75644


namespace common_noninteger_root_eq_coeffs_l75_75174

theorem common_noninteger_root_eq_coeffs (p1 p2 q1 q2 : ℤ) (α : ℝ) :
  (α^2 + (p1: ℝ) * α + (q1: ℝ) = 0) ∧ (α^2 + (p2: ℝ) * α + (q2: ℝ) = 0) ∧ ¬(∃ (k : ℤ), α = k) → p1 = p2 ∧ q1 = q2 :=
by {
  sorry
}

end common_noninteger_root_eq_coeffs_l75_75174


namespace repeated_root_cubic_l75_75659

theorem repeated_root_cubic (p : ℝ) :
  (∃ x : ℝ, (3 * x^3 - (p + 1) * x^2 + 4 * x - 12 = 0) ∧ (9 * x^2 - 2 * (p + 1) * x + 4 = 0)) →
  (p = 5 ∨ p = -7) :=
by
  sorry

end repeated_root_cubic_l75_75659


namespace discriminant_eq_complete_square_form_l75_75628

theorem discriminant_eq_complete_square_form (a b c t : ℝ) (h : a ≠ 0) (ht : a * t^2 + b * t + c = 0) :
  (b^2 - 4 * a * c) = (2 * a * t + b)^2 := 
sorry

end discriminant_eq_complete_square_form_l75_75628


namespace draw_9_cards_ensure_even_product_l75_75071

theorem draw_9_cards_ensure_even_product :
  ∀ (cards : Finset ℕ), (∀ x ∈ cards, 1 ≤ x ∧ x ≤ 16) →
  (cards.card = 9) →
  (∃ (subset : Finset ℕ), subset ⊆ cards ∧ ∃ k ∈ subset, k % 2 = 0) :=
by
  sorry

end draw_9_cards_ensure_even_product_l75_75071


namespace johns_initial_playtime_l75_75902

theorem johns_initial_playtime :
  ∃ (x : ℝ), (14 * x = 0.40 * (14 * x + 84)) → x = 4 :=
by
  sorry

end johns_initial_playtime_l75_75902


namespace find_pairs_l75_75495

theorem find_pairs (a b : ℕ) (h1 : a + b = 60) (h2 : Nat.lcm a b = 72) : (a = 24 ∧ b = 36) ∨ (a = 36 ∧ b = 24) := 
sorry

end find_pairs_l75_75495


namespace xiaoming_statement_incorrect_l75_75033

theorem xiaoming_statement_incorrect (s : ℕ) : 
    let x_h := 3
    let x_m := 6
    let steps_xh := (x_h - 1) * s
    let steps_xm := (x_m - 1) * s
    (steps_xm ≠ 2 * steps_xh) :=
by
  let x_h := 3
  let x_m := 6
  let steps_xh := (x_h - 1) * s
  let steps_xm := (x_m - 1) * s
  sorry

end xiaoming_statement_incorrect_l75_75033


namespace bridge_length_is_correct_l75_75066

noncomputable def length_of_bridge (train_length : ℝ) (time : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let distance_covered := speed_mps * time
  distance_covered - train_length

theorem bridge_length_is_correct :
  length_of_bridge 100 16.665333439991468 54 = 149.97999909987152 :=
by sorry

end bridge_length_is_correct_l75_75066


namespace part1_price_reduction_maintains_profit_part2_profit_reach_460_impossible_l75_75462

-- Define initial conditions
def cost_price : ℝ := 20
def initial_selling_price : ℝ := 40
def initial_sales_volume : ℝ := 20
def price_decrease_per_kg : ℝ := 1
def sales_increase_per_kg : ℝ := 2
def original_profit : ℝ := 400

-- Part (1) statement
theorem part1_price_reduction_maintains_profit :
  ∃ x : ℝ, (initial_selling_price - x - cost_price) * (initial_sales_volume + sales_increase_per_kg * x) = original_profit ∧ x = 20 := 
sorry

-- Part (2) statement
theorem part2_profit_reach_460_impossible :
  ¬∃ y : ℝ, (initial_selling_price - y - cost_price) * (initial_sales_volume + sales_increase_per_kg * y) = 460 :=
sorry

end part1_price_reduction_maintains_profit_part2_profit_reach_460_impossible_l75_75462


namespace proof_m_plus_n_l75_75559

variable (m n : ℚ) -- Defining m and n as rational numbers (ℚ)
-- Conditions from the problem:
axiom condition1 : 2 * m + 5 * n + 8 = 1
axiom condition2 : m - n - 3 = 1

-- Proof statement (theorem) that needs to be established:
theorem proof_m_plus_n : m + n = -2/7 :=
by
-- Since the proof is not required, we use "sorry" to placeholder the proof.
sorry

end proof_m_plus_n_l75_75559


namespace calculate_truncated_cone_volume_l75_75816

noncomputable def volume_of_truncated_cone (R₁ R₂ h : ℝ) :
    ℝ := ((1 / 3) * Real.pi * h * (R₁ ^ 2 + R₁ * R₂ + R₂ ^ 2))

theorem calculate_truncated_cone_volume : 
    volume_of_truncated_cone 10 5 10 = (1750 / 3) * Real.pi := by
sorry

end calculate_truncated_cone_volume_l75_75816


namespace initial_average_marks_is_90_l75_75385

def incorrect_average_marks (A : ℝ) : Prop :=
  let wrong_sum := 10 * A
  let correct_sum := 10 * 95
  wrong_sum + 50 = correct_sum

theorem initial_average_marks_is_90 : ∃ A : ℝ, incorrect_average_marks A ∧ A = 90 :=
by
  use 90
  unfold incorrect_average_marks
  simp
  sorry

end initial_average_marks_is_90_l75_75385


namespace project_selection_l75_75858

noncomputable def binomial : ℕ → ℕ → ℕ 
| n, 0 => 1
| 0, k => 0
| n+1, k+1 => binomial n k + binomial n (k+1)

theorem project_selection :
  (binomial 5 2 * binomial 3 2) + (binomial 3 1 * binomial 5 1) = 45 := 
sorry

end project_selection_l75_75858


namespace algebraic_expression_value_l75_75829

theorem algebraic_expression_value (a : ℝ) (h : a = Real.sqrt 6 + 2) : a^2 - 4 * a + 4 = 6 :=
by
  sorry

end algebraic_expression_value_l75_75829


namespace cos_A_minus_B_l75_75994

theorem cos_A_minus_B (A B : Real) 
  (h1 : Real.sin A + Real.sin B = -1) 
  (h2 : Real.cos A + Real.cos B = 1/2) :
  Real.cos (A - B) = -3/8 :=
by
  sorry

end cos_A_minus_B_l75_75994


namespace find_C_l75_75430

noncomputable def A := {x : ℝ | x^2 - 5 * x + 6 = 0}
noncomputable def B (a : ℝ) := {x : ℝ | a * x - 6 = 0}
def C := {a : ℝ | (A ∪ (B a)) = A}

theorem find_C : C = {0, 2, 3} := by
  sorry

end find_C_l75_75430


namespace solution_to_largest_four_digit_fulfilling_conditions_l75_75908

def largest_four_digit_fulfilling_conditions : Prop :=
  ∃ (N : ℕ), N < 10000 ∧ N ≡ 2 [MOD 11] ∧ N ≡ 4 [MOD 7] ∧ N = 9979

theorem solution_to_largest_four_digit_fulfilling_conditions : largest_four_digit_fulfilling_conditions :=
  sorry

end solution_to_largest_four_digit_fulfilling_conditions_l75_75908


namespace books_left_to_read_l75_75627

theorem books_left_to_read (total_books : ℕ) (books_mcgregor : ℕ) (books_floyd : ℕ) : total_books = 89 → books_mcgregor = 34 → books_floyd = 32 → 
  (total_books - (books_mcgregor + books_floyd) = 23) :=
by
  intros h1 h2 h3
  sorry

end books_left_to_read_l75_75627


namespace number_exceeds_its_3_over_8_part_by_20_l75_75867

theorem number_exceeds_its_3_over_8_part_by_20 (x : ℝ) (h : x = (3 / 8) * x + 20) : x = 32 :=
by
  sorry

end number_exceeds_its_3_over_8_part_by_20_l75_75867


namespace exists_n_good_not_n_add_1_good_l75_75076

-- Define the sum of digits function S
def S (k : ℕ) : ℕ := (k.digits 10).sum

-- Define what it means for a number to be n-good
def n_good (a n : ℕ) : Prop :=
  ∃ (a_seq : Fin (n + 1) → ℕ), (a_seq 0 = a) ∧ (∀ i : Fin n, a_seq i.succ = a_seq i - S (a_seq i))

-- Define the main theorem
theorem exists_n_good_not_n_add_1_good : ∀ n : ℕ, ∃ a : ℕ, n_good a n ∧ ¬n_good a (n + 1) :=
by
  sorry

end exists_n_good_not_n_add_1_good_l75_75076


namespace intersection_PQ_l75_75099

def setP  := {x : ℝ | x * (x - 1) ≥ 0}
def setQ := {y : ℝ | ∃ x : ℝ, y = 3 * x^2 + 1}

theorem intersection_PQ : {x : ℝ | x > 1} = {z : ℝ | z ∈ setP ∧ z ∈ setQ} :=
by
  sorry

end intersection_PQ_l75_75099


namespace right_triangle_side_length_l75_75327

theorem right_triangle_side_length (c a b : ℕ) (hc : c = 13) (ha : a = 12) (hypotenuse_eq : c ^ 2 = a ^ 2 + b ^ 2) : b = 5 :=
sorry

end right_triangle_side_length_l75_75327


namespace candy_sharing_l75_75316

theorem candy_sharing (Hugh_candy Tommy_candy Melany_candy shared_candy : ℕ) 
  (h1 : Hugh_candy = 8) (h2 : Tommy_candy = 6) (h3 : shared_candy = 7) :
  Hugh_candy + Tommy_candy + Melany_candy = 3 * shared_candy →
  Melany_candy = 7 :=
by
  intro h
  sorry

end candy_sharing_l75_75316


namespace relationship_abc_l75_75252

noncomputable def a := (1 / 3 : ℝ) ^ (2 / 3)
noncomputable def b := (2 / 3 : ℝ) ^ (1 / 3)
noncomputable def c := Real.logb (1/2) (1/3)

theorem relationship_abc : c > b ∧ b > a :=
by
  sorry

end relationship_abc_l75_75252


namespace license_plate_palindrome_probability_l75_75061

theorem license_plate_palindrome_probability : 
  let p := 775 
  let q := 67600  
  p + q = 776 :=
by
  let p := 775
  let q := 67600
  show p + q = 776
  sorry

end license_plate_palindrome_probability_l75_75061


namespace union_of_A_and_B_l75_75750

namespace SetProof

def A : Set ℝ := {x | x^2 ≤ 4}
def B : Set ℝ := {x | x > 0}
def expectedUnion : Set ℝ := {x | -2 ≤ x}

theorem union_of_A_and_B : (A ∪ B) = expectedUnion := by
  sorry

end SetProof

end union_of_A_and_B_l75_75750


namespace prime_divisors_of_50_fact_eq_15_l75_75465

theorem prime_divisors_of_50_fact_eq_15 :
  ∃ P : Finset Nat, (∀ p ∈ P, Prime p ∧ p ∣ (Nat.factorial 50)) ∧ P.card = 15 := by
  sorry

end prime_divisors_of_50_fact_eq_15_l75_75465


namespace union_of_A_and_B_intersection_of_complement_A_and_B_l75_75149

-- Define sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x | 3 < 2 * x - 1 ∧ 2 * x - 1 < 19}

-- Define the universal set here, which encompass all real numbers
def universal_set : Set ℝ := {x | true}

-- Define the complement of A with respect to the real numbers
def C_R (S : Set ℝ) : Set ℝ := {x | x ∉ S}

-- Prove that A ∪ B is {x | 2 < x < 10}
theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} := by
  sorry

-- Prove that (C_R A) ∩ B is {x | 2 < x < 3 ∨ 7 < x < 10}
theorem intersection_of_complement_A_and_B : (C_R A) ∪ B = {x | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10)} := by
  sorry

end union_of_A_and_B_intersection_of_complement_A_and_B_l75_75149


namespace axis_of_symmetry_l75_75415

theorem axis_of_symmetry {a b c : ℝ} (h1 : (2 : ℝ) * (a * 2 + b) + c = 5) (h2 : (4 : ℝ) * (a * 4 + b) + c = 5) : 
  (2 + 4) / 2 = 3 := 
by 
  sorry

end axis_of_symmetry_l75_75415


namespace find_a_l75_75336

theorem find_a (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_asymptote : ∀ x : ℝ, x = π/2 ∨ x = 3*π/2 ∨ x = -π/2 ∨ x = -3*π/2 → b*x = π/2 ∨ b*x = 3*π/2 ∨ b*x = -π/2 ∨ b*x = -3*π/2)
  (h_amplitude : ∀ x : ℝ, |a * (1 / Real.cos (b * x))| ≤ 3): 
  a = 3 := 
sorry

end find_a_l75_75336


namespace expression_multiple_l75_75264

theorem expression_multiple :
  let a : ℚ := 1/2
  let b : ℚ := 1/3
  (a - b) / (1/78) = 13 :=
by
  sorry

end expression_multiple_l75_75264


namespace smallest_positive_integer_congruence_l75_75048

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ, 5 * x ≡ 18 [MOD 31] ∧ 0 < x ∧ x < 31 ∧ x = 16 := 
by sorry

end smallest_positive_integer_congruence_l75_75048


namespace sum_of_cubes_mod_4_l75_75210

theorem sum_of_cubes_mod_4 :
  let b := 2
  let n := 2010
  ( (n * (n + 1) / 2) ^ 2 ) % (b ^ 2) = 1 :=
by
  let b := 2
  let n := 2010
  sorry

end sum_of_cubes_mod_4_l75_75210


namespace part_one_part_two_l75_75109

theorem part_one (g : ℝ → ℝ) (h : ∀ x, g x = |x - 1| + 2) : {x : ℝ | |g x| < 5} = {x : ℝ | -2 < x ∧ x < 4} :=
sorry

theorem part_two (f g : ℝ → ℝ) (h1 : ∀ x, f x = |2 * x - a| + |2 * x + 3|) (h2 : ∀ x, g x = |x - 1| + 2) 
(h3 : ∀ x1 : ℝ, ∃ x2 : ℝ, f x1 = g x2) : {a : ℝ | a ≥ -1 ∨ a ≤ -5} :=
sorry

end part_one_part_two_l75_75109


namespace range_of_a_l75_75216

theorem range_of_a (a : ℝ) : (∃ x y : ℝ, x^2 / a^2 + y^2 / (a + 6) = 1 ∧ (a^2 > a + 6 ∧ a + 6 > 0)) → (a > 3 ∨ (-6 < a ∧ a < -2)) :=
by
  intro h
  sorry

end range_of_a_l75_75216


namespace geometric_sequence_solution_l75_75182

-- Assume we have a type for real numbers
variable {R : Type} [LinearOrderedField R]

theorem geometric_sequence_solution (a b c : R)
  (h1 : -1 ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) (h5 : -9 ≠ 0)
  (h : ∃ r : R, r ≠ 0 ∧ (a = r * -1) ∧ (b = r * a) ∧ (c = r * b) ∧ (-9 = r * c)) :
  b = -3 ∧ a * c = 9 := by
  sorry

end geometric_sequence_solution_l75_75182


namespace no_integer_solutions_l75_75770

theorem no_integer_solutions (a b : ℤ) : ¬ (3 * a ^ 2 = b ^ 2 + 1) :=
  sorry

end no_integer_solutions_l75_75770


namespace line_equation_through_P_and_equidistant_from_A_B_l75_75084

theorem line_equation_through_P_and_equidistant_from_A_B (P A B : ℝ × ℝ) (hP : P = (1, 2)) (hA : A = (2, 3)) (hB : B = (4, -5)) :
  (∃ l : ℝ × ℝ → Prop, ∀ x y, l (x, y) ↔ 4 * x + y - 6 = 0 ∨ 3 * x + 2 * y - 7 = 0) :=
sorry

end line_equation_through_P_and_equidistant_from_A_B_l75_75084


namespace perp_lines_of_parallel_planes_l75_75188

variables {Line Plane : Type} 
variables (m n : Line) (α β : Plane)
variable (is_parallel : Line → Plane → Prop)
variable (is_perpendicular : Line → Plane → Prop)
variable (planes_parallel : Plane → Plane → Prop)
variable (lines_perpendicular : Line → Line → Prop)

-- Given Conditions
variables (h1 : planes_parallel α β) (h2 : is_perpendicular m α) (h3 : is_parallel n β)

-- Prove that
theorem perp_lines_of_parallel_planes (h1 : planes_parallel α β) (h2 : is_perpendicular m α) (h3 : is_parallel n β) : lines_perpendicular m n := 
sorry

end perp_lines_of_parallel_planes_l75_75188


namespace sphere_radius_equal_l75_75748

theorem sphere_radius_equal (r : ℝ) 
  (hvol : (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2) : r = 3 :=
sorry

end sphere_radius_equal_l75_75748


namespace multiples_of_3_or_4_probability_l75_75052

theorem multiples_of_3_or_4_probability :
  let total_cards := 36
  let multiples_of_3 := 12
  let multiples_of_4 := 9
  let multiples_of_both := 3
  let favorable_outcomes := multiples_of_3 + multiples_of_4 - multiples_of_both
  let probability := (favorable_outcomes : ℚ) / total_cards
  probability = 1 / 2 :=
by
  sorry

end multiples_of_3_or_4_probability_l75_75052


namespace intersection_M_N_l75_75228

def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt x}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l75_75228


namespace find_principal_l75_75692

-- Define the conditions
def interest_rate : ℝ := 0.05
def time_period : ℕ := 10
def interest_less_than_principal : ℝ := 3100

-- Define the principal
def principal : ℝ := 6200

-- The theorem statement
theorem find_principal :
  ∃ P : ℝ, P - interest_less_than_principal = P * interest_rate * time_period ∧ P = principal :=
by
  sorry

end find_principal_l75_75692


namespace phi_value_l75_75821

noncomputable def f (x φ : ℝ) := Real.sin (2 * x + φ)

theorem phi_value (φ : ℝ) (h1 : ∀ x : ℝ, f x φ ≤ |f (π / 6) φ|) (h2 : f (π / 3) φ > f (π / 2) φ) : φ = π / 6 :=
by
  sorry

end phi_value_l75_75821


namespace car_speed_proof_l75_75128

noncomputable def car_speed_second_hour 
  (speed_first_hour: ℕ) (average_speed: ℕ) (total_time: ℕ) 
  (speed_second_hour: ℕ) : Prop :=
  (speed_first_hour = 80) ∧ (average_speed = 70) ∧ (total_time = 2) → speed_second_hour = 60

theorem car_speed_proof : 
  car_speed_second_hour 80 70 2 60 := by
  sorry

end car_speed_proof_l75_75128


namespace check_line_properties_l75_75735

-- Define the conditions
def line_equation (x y : ℝ) : Prop := y + 7 = -x - 3

-- Define the point and slope
def point_and_slope (x y : ℝ) (m : ℝ) : Prop := (x, y) = (-3, -7) ∧ m = -1

-- State the theorem to prove
theorem check_line_properties :
  ∃ x y m, line_equation x y ∧ point_and_slope x y m :=
sorry

end check_line_properties_l75_75735


namespace problem_a1_value_l75_75133

theorem problem_a1_value (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) (h : ∀ x : ℝ, x^10 = a + a₁ * (x - 1) + a₂ * (x - 1)^2 + a₃ * (x - 1)^3 + a₄ * (x - 1)^4 + a₅ * (x - 1)^5 + a₆ * (x - 1)^6 + a₇ * (x - 1)^7 + a₈ * (x - 1)^8 + a₉ * (x - 1)^9 + a₁₀ * (x - 1)^10) :
  a₁ = 10 :=
sorry

end problem_a1_value_l75_75133


namespace placements_for_nine_squares_l75_75190

-- Define the parameters and conditions of the problem
def countPlacements (n : ℕ) : ℕ := sorry

theorem placements_for_nine_squares : countPlacements 9 = 25 := sorry

end placements_for_nine_squares_l75_75190


namespace trapezoid_area_l75_75243

open Real

theorem trapezoid_area 
  (r : ℝ) (BM CD AB : ℝ) (radius_nonneg : 0 ≤ r) 
  (BM_positive : 0 < BM) (CD_positive : 0 < CD) (AB_positive : 0 < AB)
  (circle_radius : r = 4) (BM_length : BM = 16) (CD_length : CD = 3) :
  let height := 2 * r
  let base_sum := AB + CD
  let area := height * base_sum / 2
  AB = BM + 8 → area = 108 :=
by
  intro hyp
  sorry

end trapezoid_area_l75_75243


namespace riya_speed_l75_75234

theorem riya_speed 
  (R : ℝ)
  (priya_speed : ℝ) 
  (time : ℝ) 
  (distance : ℝ)
  (h_priya_speed : priya_speed = 22)
  (h_time : time = 1)
  (h_distance : distance = 43)
  : R + priya_speed * time = distance → R = 21 :=
by 
  sorry

end riya_speed_l75_75234


namespace frank_reading_days_l75_75153

-- Define the parameters
def pages_weekdays : ℚ := 5.7
def pages_weekends : ℚ := 9.5
def total_pages : ℚ := 576
def pages_per_week : ℚ := (pages_weekdays * 5) + (pages_weekends * 2)

-- Define the property to be proved
theorem frank_reading_days : 
  (total_pages / pages_per_week).floor * 7 + 
  (total_pages - (total_pages / pages_per_week).floor * pages_per_week) / pages_weekdays 
  = 85 := 
  by
    sorry

end frank_reading_days_l75_75153


namespace Peggy_dolls_l75_75072

theorem Peggy_dolls (initial_dolls granny_dolls birthday_dolls : ℕ) (h1 : initial_dolls = 6) (h2 : granny_dolls = 30) (h3 : birthday_dolls = granny_dolls / 2) : 
  initial_dolls + granny_dolls + birthday_dolls = 51 := by
  sorry

end Peggy_dolls_l75_75072


namespace log_condition_necessary_not_sufficient_l75_75949

noncomputable def base_of_natural_logarithm := Real.exp 1

variable (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : 0 < b) (h4 : b ≠ 1)

theorem log_condition_necessary_not_sufficient (h : 0 < a ∧ a < b ∧ b < 1) :
  (Real.log 2 / Real.log a > Real.log base_of_natural_logarithm / Real.log b) :=
sorry

end log_condition_necessary_not_sufficient_l75_75949


namespace incorrect_statement_D_l75_75912

theorem incorrect_statement_D
  (passes_through_center : ∀ (x_vals y_vals : List ℝ), ∃ (regression_line : ℝ → ℝ), 
    regression_line (x_vals.sum / x_vals.length) = (y_vals.sum / y_vals.length))
  (higher_r2_better_fit : ∀ (r2 : ℝ), r2 > 0 → ∃ (residual_sum_squares : ℝ), residual_sum_squares < (1 - r2))
  (slope_interpretation : ∀ (x : ℝ), (0.2 * x + 0.8) - (0.2 * (x - 1) + 0.8) = 0.2)
  (chi_squared_k2 : ∀ (X Y : Type) [Fintype X] [Fintype Y] (k : ℝ), (k > 0) → 
    ∃ (confidence : ℝ), confidence > 0) :
  ¬(∀ (X Y : Type) [Fintype X] [Fintype Y] (k : ℝ), k > 0 → 
    ∃ (confidence : ℝ), confidence < 0) :=
by
  sorry

end incorrect_statement_D_l75_75912


namespace donuts_Niraek_covers_l75_75842

/- Define the radii of the donut holes -/
def radius_Niraek : ℕ := 5
def radius_Theo : ℕ := 9
def radius_Akshaj : ℕ := 10
def radius_Lily : ℕ := 7

/- Define the surface areas of the donut holes -/
def surface_area (r : ℕ) : ℕ := 4 * r * r

/- Compute the surface areas -/
def sa_Niraek := surface_area radius_Niraek
def sa_Theo := surface_area radius_Theo
def sa_Akshaj := surface_area radius_Akshaj
def sa_Lily := surface_area radius_Lily

/- Define a function to compute the LCM of a list of natural numbers -/
def lcm_of_list (l : List ℕ) : ℕ := l.foldr Nat.lcm 1

/- Compute the lcm of the surface areas -/
def lcm_surface_areas := lcm_of_list [sa_Niraek, sa_Theo, sa_Akshaj, sa_Lily]

/- Compute the answer -/
def num_donuts_Niraek_covers := lcm_surface_areas / sa_Niraek

/- Prove the statement -/
theorem donuts_Niraek_covers : num_donuts_Niraek_covers = 63504 :=
by
  /- Skipping the proof for now -/
  sorry

end donuts_Niraek_covers_l75_75842


namespace cost_price_l75_75253

theorem cost_price (SP : ℝ) (profit_percent : ℝ) (C : ℝ) 
  (h1 : SP = 400) 
  (h2 : profit_percent = 25) 
  (h3 : SP = C + (profit_percent / 100) * C) : 
  C = 320 := 
by
  sorry

end cost_price_l75_75253


namespace area_of_small_parallelograms_l75_75303

theorem area_of_small_parallelograms (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  (1 : ℝ) / (m * n : ℝ) = 1 / (m * n) :=
by
  sorry

end area_of_small_parallelograms_l75_75303


namespace max_ab_bc_cd_da_l75_75176

theorem max_ab_bc_cd_da (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) (h₄ : a + b + c + d = 200) :
  ab + bc + cd + da ≤ 10000 :=
sorry

end max_ab_bc_cd_da_l75_75176


namespace inequality_subtraction_l75_75755

theorem inequality_subtraction (a b : ℝ) (h : a < b) : a - 5 < b - 5 := 
by {
  sorry
}

end inequality_subtraction_l75_75755


namespace number_of_five_ruble_coins_l75_75785

theorem number_of_five_ruble_coins (total_coins a b c : Nat) (h1 : total_coins = 25) (h2 : 19 = total_coins - a) (h3 : 20 = total_coins - b) (h4 : 16 = total_coins - c) :
  total_coins - (a + b + c) = 5 :=
by
  sorry

end number_of_five_ruble_coins_l75_75785


namespace range_of_a_l75_75409

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) ↔ (-2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2) :=
sorry

end range_of_a_l75_75409


namespace polynomial_sum_evaluation_l75_75093

noncomputable def q1 : Polynomial ℤ := Polynomial.X^3
noncomputable def q2 : Polynomial ℤ := Polynomial.X^2 + Polynomial.X + 1
noncomputable def q3 : Polynomial ℤ := Polynomial.X - 1
noncomputable def q4 : Polynomial ℤ := Polynomial.X^2 + 1

theorem polynomial_sum_evaluation :
  q1.eval 3 + q2.eval 3 + q3.eval 3 + q4.eval 3 = 52 :=
by
  sorry

end polynomial_sum_evaluation_l75_75093


namespace evaluate_expression_l75_75573

variable {R : Type} [CommRing R]

theorem evaluate_expression (x y z w : R) :
  (x - (y - 3 * z + w)) - ((x - y + w) - 3 * z) = 6 * z - 2 * w :=
by
  sorry

end evaluate_expression_l75_75573


namespace saree_original_price_l75_75521

theorem saree_original_price :
  ∃ P : ℝ, (0.95 * 0.88 * P = 334.4) ∧ (P = 400) :=
by
  sorry

end saree_original_price_l75_75521


namespace possible_values_a1_l75_75126

theorem possible_values_a1 (m : ℕ) (h_m_pos : 0 < m)
    (a : ℕ → ℕ) (h_seq : ∀ n, a n.succ = if a n < 2^m then a n ^ 2 + 2^m else a n / 2)
    (h1 : ∀ n, a n > 0) :
    (∀ n, ∃ k : ℕ, a n = 2^k) ↔ (m = 2 ∧ ∃ ℓ : ℕ, a 0 = 2 ^ ℓ ∧ 0 < ℓ) :=
by sorry

end possible_values_a1_l75_75126


namespace combinations_sol_eq_l75_75909

theorem combinations_sol_eq (x : ℕ) (h : Nat.choose 10 x = Nat.choose 10 (3 * x - 2)) : x = 1 ∨ x = 3 := sorry

end combinations_sol_eq_l75_75909


namespace ball_draw_probability_l75_75016

/-- 
Four balls labeled with numbers 1, 2, 3, 4 are placed in an urn. 
A ball is drawn, its number is recorded, and then the ball is returned to the urn. 
This process is repeated three times. Each ball is equally likely to be drawn on each occasion. 
Given that the sum of the numbers recorded is 7, the probability that the ball numbered 2 was drawn twice is 1/4. 
-/
theorem ball_draw_probability :
  let draws := [(1, 1, 5),(1, 2, 4),(1, 3, 3),(2, 2, 3)]
  (3 / 12 = 1 / 4) :=
by
  sorry

end ball_draw_probability_l75_75016


namespace area_increase_percentage_l75_75793

variable (r : ℝ) (π : ℝ := Real.pi)

theorem area_increase_percentage (h₁ : r > 0) (h₂ : π > 0) : 
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * (new_radius)^2
  (new_area - original_area) / original_area * 100 = 525 := 
by
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * (new_radius)^2
  sorry

end area_increase_percentage_l75_75793


namespace abs_lt_inequality_solution_l75_75574

theorem abs_lt_inequality_solution (x : ℝ) : |x - 1| < 2 ↔ -1 < x ∧ x < 3 :=
by sorry

end abs_lt_inequality_solution_l75_75574


namespace sum_of_three_consecutive_integers_divisible_by_3_l75_75065

theorem sum_of_three_consecutive_integers_divisible_by_3 (a : ℤ) :
  ∃ k : ℤ, k = 3 ∧ (a - 1 + a + (a + 1)) % k = 0 :=
by
  use 3
  sorry

end sum_of_three_consecutive_integers_divisible_by_3_l75_75065


namespace lines_intersect_lines_perpendicular_lines_coincide_lines_parallel_l75_75801

/- Define lines l1 and l2 -/
def l1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + a^2 - 1 = 0

/- Prove intersection condition -/
theorem lines_intersect (a : ℝ) : (∃ x y, l1 a x y ∧ l2 a x y) ↔ (a ≠ -1 ∧ a ≠ 2) := 
sorry

/- Prove perpendicular condition -/
theorem lines_perpendicular (a : ℝ) : (∃ x1 y1 x2 y2, l1 a x1 y1 ∧ l2 a x2 y2 ∧ x1 * x2 + y1 * y2 = 0) ↔ (a = 2 / 3) :=
sorry

/- Prove coincident condition -/
theorem lines_coincide (a : ℝ) : (∀ x y, l1 a x y ↔ l2 a x y) ↔ (a = 2) := 
sorry

/- Prove parallel condition -/
theorem lines_parallel (a : ℝ) : (∀ x1 y1 x2 y2, l1 a x1 y1 → l2 a x2 y2 → (x1 * y2 - y1 * x2) = 0) ↔ (a = -1) := 
sorry

end lines_intersect_lines_perpendicular_lines_coincide_lines_parallel_l75_75801


namespace parallel_planes_l75_75379

variables {Point Line Plane : Type}
variables (a : Line) (α β : Plane)

-- Conditions
def line_perpendicular_plane (l: Line) (p: Plane) : Prop := sorry
def planes_parallel (p₁ p₂: Plane) : Prop := sorry

-- Problem statement
theorem parallel_planes (h1: line_perpendicular_plane a α) 
                        (h2: line_perpendicular_plane a β) : 
                        planes_parallel α β :=
sorry

end parallel_planes_l75_75379


namespace Kevin_crates_per_week_l75_75261

theorem Kevin_crates_per_week (a b c : ℕ) (h₁ : a = 13) (h₂ : b = 20) (h₃ : c = 17) :
  a + b + c = 50 :=
by 
  sorry

end Kevin_crates_per_week_l75_75261


namespace simplify_fraction_l75_75779

theorem simplify_fraction (a b : ℕ) (h : a ≠ b) : (2 * a) / (2 * b) = a / b :=
sorry

end simplify_fraction_l75_75779


namespace minimum_value_expression_l75_75928

theorem minimum_value_expression (a b : ℝ) (h : a * b > 0) : 
  ∃ m : ℝ, (∀ x y : ℝ, x * y > 0 → (4 * y / x + (x - 2 * y) / y) ≥ m) ∧ m = 2 :=
by
  sorry

end minimum_value_expression_l75_75928


namespace original_distance_cycled_l75_75663

theorem original_distance_cycled
  (x t d : ℝ)
  (h1 : d = x * t)
  (h2 : d = (x + 1/4) * (3/4 * t))
  (h3 : d = (x - 1/4) * (t + 3)) :
  d = 4.5 := 
sorry

end original_distance_cycled_l75_75663


namespace find_d_l75_75797

theorem find_d (a d : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 45 * d) : d = 49 :=
sorry

end find_d_l75_75797


namespace hired_year_l75_75104

theorem hired_year (A W : ℕ) (Y : ℕ) (retire_year : ℕ) 
    (hA : A = 30) 
    (h_rule : A + W = 70) 
    (h_retire : retire_year = 2006) 
    (h_employment : retire_year - Y = W) 
    : Y = 1966 := 
by 
  -- proofs are skipped with 'sorry'
  sorry

end hired_year_l75_75104


namespace compute_value_l75_75766

theorem compute_value : 12 - 4 * (5 - 10)^3 = 512 :=
by
  sorry

end compute_value_l75_75766


namespace greatest_root_of_g_l75_75721

noncomputable def g (x : ℝ) : ℝ := 16 * x^4 - 20 * x^2 + 5

theorem greatest_root_of_g :
  ∃ r : ℝ, r = Real.sqrt 5 / 2 ∧ (forall x, g x ≤ g r) :=
sorry

end greatest_root_of_g_l75_75721


namespace f_23_plus_f_neg14_l75_75060

noncomputable def f : ℝ → ℝ := sorry

axiom periodic_f : ∀ x, f (x + 5) = f x
axiom odd_f : ∀ x, f (-x) = -f x
axiom f_one : f 1 = 1
axiom f_two : f 2 = 2

theorem f_23_plus_f_neg14 : f 23 + f (-14) = -1 := by
  sorry

end f_23_plus_f_neg14_l75_75060


namespace find_m_value_l75_75756

-- Defining the hyperbola equation and the conditions
def hyperbola_eq (x y : ℝ) (m : ℝ) : Prop :=
  (x^2 / m) - (y^2 / 4) = 1

-- Definition of the focal distance
def focal_distance (c : ℝ) :=
  2 * c = 6

-- Definition of the relationship c^2 = a^2 + b^2 for hyperbolas
def hyperbola_focal_distance_eq (m : ℝ) (c b : ℝ) : Prop :=
  c^2 = m + b^2

-- Stating that the hyperbola has the given focal distance
def given_focal_distance : Prop :=
  focal_distance 3

-- Stating the given condition on b²
def given_b_squared : Prop :=
  4 = 4

-- The main theorem stating that m = 5 given the conditions.
theorem find_m_value (m : ℝ) : 
  (hyperbola_eq 1 1 m) → given_focal_distance → given_b_squared → m = 5 :=
by
  sorry

end find_m_value_l75_75756


namespace contradiction_assumption_l75_75241

variable (x y z : ℝ)

/-- The negation of "at least one is positive" for proof by contradiction is 
    "all three numbers are non-positive". -/
theorem contradiction_assumption (h : ¬ (x > 0 ∨ y > 0 ∨ z > 0)) : 
  (x ≤ 0 ∧ y ≤ 0 ∧ z ≤ 0) :=
by
  sorry

end contradiction_assumption_l75_75241


namespace area_increase_percentage_l75_75198

-- Define the original dimensions l and w as non-zero real numbers
variables (l w : ℝ) (hl : l ≠ 0) (hw : w ≠ 0)

-- Define the new dimensions after increase
def new_length := 1.15 * l
def new_width := 1.25 * w

-- Define the original and new areas
def original_area := l * w
def new_area := new_length l * new_width w

-- The statement to prove
theorem area_increase_percentage :
  ((new_area l w - original_area l w) / original_area l w) * 100 = 43.75 :=
by
  sorry

end area_increase_percentage_l75_75198


namespace quilt_cost_calculation_l75_75476

theorem quilt_cost_calculation :
  let length := 12
  let width := 15
  let cost_per_sq_foot := 70
  let sales_tax_rate := 0.05
  let discount_rate := 0.10
  let area := length * width
  let cost_before_discount := area * cost_per_sq_foot
  let discount_amount := cost_before_discount * discount_rate
  let cost_after_discount := cost_before_discount - discount_amount
  let sales_tax_amount := cost_after_discount * sales_tax_rate
  let total_cost := cost_after_discount + sales_tax_amount
  total_cost = 11907 := by
  {
    sorry
  }

end quilt_cost_calculation_l75_75476


namespace garden_ratio_l75_75047

theorem garden_ratio (L W : ℝ) (h1 : 2 * L + 2 * W = 180) (h2 : L = 60) : L / W = 2 :=
by
  -- this is where you would put the proof
  sorry

end garden_ratio_l75_75047


namespace units_digit_of_n_l75_75888

theorem units_digit_of_n
  (m n : ℕ)
  (h1 : m * n = 23^7)
  (h2 : m % 10 = 9) : n % 10 = 3 :=
sorry

end units_digit_of_n_l75_75888


namespace set_of_a_where_A_subset_B_l75_75948

variable {a x : ℝ}

theorem set_of_a_where_A_subset_B (h : ∀ x, (2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5) → (3 ≤ x ∧ x ≤ 22)) :
  6 ≤ a ∧ a ≤ 9 :=
by
  sorry

end set_of_a_where_A_subset_B_l75_75948


namespace find_x_plus_y_l75_75457

theorem find_x_plus_y
  (x y : ℤ)
  (hx : |x| = 2)
  (hy : |y| = 3)
  (hxy : x > y) : x + y = -1 := 
sorry

end find_x_plus_y_l75_75457


namespace value_of_f_at_5_l75_75330

def f (x : ℝ) : ℝ := 4 * x + 2

theorem value_of_f_at_5 : f 5 = 22 :=
by
  sorry

end value_of_f_at_5_l75_75330


namespace marbles_leftover_l75_75129

theorem marbles_leftover (g j : ℕ) (hg : g % 8 = 5) (hj : j % 8 = 6) :
  ((g + 5 + j) % 8) = 0 :=
by
  sorry

end marbles_leftover_l75_75129


namespace maximum_median_soda_shop_l75_75887

noncomputable def soda_shop_median (total_cans : ℕ) (total_customers : ℕ) (min_cans_per_customer : ℕ) : ℝ :=
  if total_cans = 300 ∧ total_customers = 120 ∧ min_cans_per_customer = 1 then 3.5 else sorry

theorem maximum_median_soda_shop : soda_shop_median 300 120 1 = 3.5 :=
by
  sorry

end maximum_median_soda_shop_l75_75887


namespace find_sum_due_l75_75896

variable (BD TD FV : ℝ)

-- given conditions
def condition_1 : Prop := BD = 80
def condition_2 : Prop := TD = 70
def condition_3 : Prop := BD = TD + (TD * BD / FV)

-- goal statement
theorem find_sum_due (h1 : condition_1 BD) (h2 : condition_2 TD) (h3 : condition_3 BD TD FV) : FV = 560 :=
by
  sorry

end find_sum_due_l75_75896


namespace find_other_number_l75_75146

theorem find_other_number (A B : ℕ) (H1 : Nat.lcm A B = 2310) (H2 : Nat.gcd A B = 30) (H3 : A = 770) : B = 90 :=
  by
  sorry

end find_other_number_l75_75146


namespace v_not_closed_under_operations_l75_75040

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def v : Set ℕ := {n | ∃ m : ℕ, n = m * m}

def addition_followed_by_multiplication (a b : ℕ) : ℕ :=
  (a + b) * a

def multiplication_followed_by_addition (a b : ℕ) : ℕ :=
  (a * b) + a

def division_followed_by_subtraction (a b : ℕ) : ℕ :=
  if b ≠ 0 then (a / b) - b else 0

def extraction_root_followed_by_multiplication (a b : ℕ) : ℕ :=
  (Nat.sqrt a) * (Nat.sqrt b)

theorem v_not_closed_under_operations : 
  ¬ (∀ a ∈ v, ∀ b ∈ v, addition_followed_by_multiplication a b ∈ v) ∧
  ¬ (∀ a ∈ v, ∀ b ∈ v, multiplication_followed_by_addition a b ∈ v) ∧
  ¬ (∀ a ∈ v, ∀ b ∈ v, division_followed_by_subtraction a b ∈ v) ∧
  ¬ (∀ a ∈ v, ∀ b ∈ v, extraction_root_followed_by_multiplication a b ∈ v) :=
sorry

end v_not_closed_under_operations_l75_75040


namespace handshakes_at_meetup_l75_75085

theorem handshakes_at_meetup :
  let gremlins := 25
  let imps := 20
  let sprites := 10
  ∃ (total_handshakes : ℕ), total_handshakes = 1095 :=
by
  sorry

end handshakes_at_meetup_l75_75085


namespace find_DG_l75_75358

theorem find_DG (a b S k l DG BC : ℕ) (h1: S = 17 * (a + b)) (h2: S % a = 0) (h3: S % b = 0) (h4: a = S / k) (h5: b = S / l) (h6: BC = 17) (h7: (k - 17) * (l - 17) = 289) : DG = 306 :=
by
  sorry

end find_DG_l75_75358


namespace find_b_l75_75001

theorem find_b
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = (1/12) * x^2 + a * x + b)
  (A C: ℝ × ℝ)
  (hA : A = (x1, 0))
  (hC : C = (x2, 0))
  (T : ℝ × ℝ)
  (hT : T = (3, 3))
  (h_TA : dist (3, 3) (x1, 0) = dist (3, 3) (0, b))
  (h_TB : dist (3, 3) (0, b) = dist (3, 3) (x2, 0))
  (vietas : x1 * x2 = 12 * b)
  : b = -6 := 
sorry

end find_b_l75_75001


namespace find_a2_b2_l75_75225

theorem find_a2_b2 (a b : ℝ) (h : a^2 * b^2 + a^2 + b^2 + 16 = 10 * a * b) : a^2 + b^2 = 8 :=
by
  sorry

end find_a2_b2_l75_75225


namespace initial_markers_count_l75_75132

   -- Let x be the initial number of markers Megan had.
   variable (x : ℕ)

   -- Conditions:
   def robert_gave_109_markers : Prop := true
   def total_markers_after_adding : ℕ := 326
   def markers_added_by_robert : ℕ := 109

   -- The total number of markers Megan has now is 326.
   def total_markers_eq (x : ℕ) : Prop := x + markers_added_by_robert = total_markers_after_adding

   -- Prove that initially Megan had 217 markers.
   theorem initial_markers_count : total_markers_eq 217 := by
     sorry
   
end initial_markers_count_l75_75132


namespace f_4_1981_eq_l75_75820

def f : ℕ → ℕ → ℕ
| 0, y     => y + 1
| (x + 1), 0 => f x 1
| (x + 1), (y + 1) => f x (f (x + 1) y)

theorem f_4_1981_eq : f 4 1981 = 2 ^ 16 - 3 := sorry

end f_4_1981_eq_l75_75820


namespace probability_at_least_one_expired_l75_75433

theorem probability_at_least_one_expired (total_bottles expired_bottles selected_bottles : ℕ) : 
  total_bottles = 10 → expired_bottles = 3 → selected_bottles = 3 → 
  (∃ probability, probability = 17 / 24) :=
by
  sorry

end probability_at_least_one_expired_l75_75433


namespace solve_for_x_minus_y_l75_75341

theorem solve_for_x_minus_y (x y : ℝ) 
  (h1 : 3 * x - 5 * y = 5)
  (h2 : x / (x + y) = 5 / 7) : 
  x - y = 3 := 
by 
  -- Proof would go here
  sorry

end solve_for_x_minus_y_l75_75341


namespace nearest_integer_ratio_l75_75481

variable (a b : ℝ)

-- Given condition and constraints
def condition : Prop := (a > b) ∧ (b > 0) ∧ (a + b) / 2 = 3 * Real.sqrt (a * b)

-- Main statement to prove
theorem nearest_integer_ratio (h : condition a b) : Int.floor (a / b) = 34 ∨ Int.floor (a / b) = 33 := sorry

end nearest_integer_ratio_l75_75481


namespace range_of_a_l75_75142

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 3| + |x - a| ≥ 3) ↔ a ≤ 0 ∨ a ≥ 6 :=
by
  sorry

end range_of_a_l75_75142


namespace cube_difference_l75_75164

variable (x : ℝ)

theorem cube_difference (h : x - 1/x = 5) : x^3 - 1/x^3 = 120 := by
  sorry

end cube_difference_l75_75164


namespace exists_m_n_l75_75148

theorem exists_m_n (p : ℕ) (hp : p > 10) [hp_prime : Fact (Nat.Prime p)] :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m + n < p ∧ (5^m * 7^n - 1) % p = 0 :=
sorry

end exists_m_n_l75_75148


namespace time_saved_is_six_minutes_l75_75553

-- Conditions
def distance_monday : ℝ := 3
def distance_wednesday : ℝ := 4
def distance_friday : ℝ := 5

def speed_monday : ℝ := 6
def speed_wednesday : ℝ := 4
def speed_friday : ℝ := 5

def speed_constant : ℝ := 5

-- Question (proof statement)
theorem time_saved_is_six_minutes : 
  (distance_monday / speed_monday + distance_wednesday / speed_wednesday + distance_friday / speed_friday) - (distance_monday + distance_wednesday + distance_friday) / speed_constant = 0.1 :=
by
  sorry

end time_saved_is_six_minutes_l75_75553


namespace boys_and_girls_arrangement_l75_75346

theorem boys_and_girls_arrangement : 
  ∃ (arrangements : ℕ), arrangements = 48 :=
  sorry

end boys_and_girls_arrangement_l75_75346


namespace faith_overtime_hours_per_day_l75_75199

noncomputable def normal_pay_per_hour : ℝ := 13.50
noncomputable def normal_daily_hours : ℕ := 8
noncomputable def normal_weekly_days : ℕ := 5
noncomputable def total_weekly_earnings : ℝ := 675
noncomputable def overtime_rate_multiplier : ℝ := 1.5

noncomputable def normal_weekly_hours := normal_daily_hours * normal_weekly_days
noncomputable def normal_weekly_earnings := normal_pay_per_hour * normal_weekly_hours
noncomputable def overtime_earnings := total_weekly_earnings - normal_weekly_earnings
noncomputable def overtime_pay_per_hour := normal_pay_per_hour * overtime_rate_multiplier
noncomputable def total_overtime_hours := overtime_earnings / overtime_pay_per_hour
noncomputable def overtime_hours_per_day := total_overtime_hours / normal_weekly_days

theorem faith_overtime_hours_per_day :
  overtime_hours_per_day = 1.33 := 
by 
  sorry

end faith_overtime_hours_per_day_l75_75199


namespace incorrect_regression_intercept_l75_75183

theorem incorrect_regression_intercept (points : List (ℕ × ℝ)) (h_points : points = [(1, 0.5), (2, 0.8), (3, 1.0), (4, 1.2), (5, 1.5)]) :
  ¬ (∃ (a : ℝ), a = 0.26 ∧ ∀ x : ℕ, x ∈ ([1, 2, 3, 4, 5] : List ℕ) → (∃ y : ℝ, y = 0.24 * x + a)) := sorry

end incorrect_regression_intercept_l75_75183


namespace parabola_standard_equation_l75_75722

theorem parabola_standard_equation :
  ∃ m : ℝ, (∀ x y : ℝ, (x^2 = 2 * m * y ↔ (0, -6) ∈ ({p | 3 * p.1 - 4 * p.2 - 24 = 0}))) → 
  (x^2 = -24 * y) := 
by {
  sorry
}

end parabola_standard_equation_l75_75722


namespace decreasing_power_function_l75_75598

open Nat

/-- For the power function y = x^(m^2 + 2*m - 3) (where m : ℕ) 
    to be a decreasing function in the interval (0, +∞), prove that m = 0. -/
theorem decreasing_power_function (m : ℕ) (h : m^2 + 2 * m - 3 < 0) : m = 0 := 
by
  sorry

end decreasing_power_function_l75_75598


namespace interest_rate_for_first_part_l75_75886

def sum_amount : ℝ := 2704
def part2 : ℝ := 1664
def part1 : ℝ := sum_amount - part2
def rate2 : ℝ := 0.05
def years2 : ℝ := 3
def interest2 : ℝ := part2 * rate2 * years2
def years1 : ℝ := 8

theorem interest_rate_for_first_part (r1 : ℝ) :
  part1 * r1 * years1 = interest2 → r1 = 0.03 :=
by
  sorry

end interest_rate_for_first_part_l75_75886


namespace solve_for_x_l75_75254

theorem solve_for_x (x : ℚ) (h : 5 * (x - 6) = 3 * (3 - 3 * x) + 9) : x = 24 / 7 :=
sorry

end solve_for_x_l75_75254


namespace remainder_equality_l75_75376

theorem remainder_equality 
  (Q Q' S S' E s s' : ℕ) 
  (Q_gt_Q' : Q > Q')
  (h1 : Q % E = S)
  (h2 : Q' % E = S')
  (h3 : (Q^2 * Q') % E = s)
  (h4 : (S^2 * S') % E = s') :
  s = s' :=
sorry

end remainder_equality_l75_75376


namespace required_integer_l75_75321

def digits_sum_to (n : ℕ) (sum : ℕ) : Prop :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 + d2 + d3 + d4 = sum

def middle_digits_sum_to (n : ℕ) (sum : ℕ) : Prop :=
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  d2 + d3 = sum

def thousands_minus_units (n : ℕ) (diff : ℕ) : Prop :=
  let d1 := n / 1000
  let d4 := n % 10
  d1 - d4 = diff

def divisible_by (n : ℕ) (d : ℕ) : Prop :=
  n % d = 0

theorem required_integer : 
  ∃ (n : ℕ), 
    1000 ≤ n ∧ n < 10000 ∧ 
    digits_sum_to n 18 ∧ 
    middle_digits_sum_to n 9 ∧ 
    thousands_minus_units n 3 ∧ 
    divisible_by n 9 ∧ 
    n = 6453 :=
by
  sorry

end required_integer_l75_75321


namespace larry_jogs_first_week_days_l75_75965

-- Defining the constants and conditions
def daily_jogging_time := 30 -- Larry jogs for 30 minutes each day
def total_jogging_time_in_hours := 4 -- Total jogging time in two weeks in hours
def total_jogging_time_in_minutes := total_jogging_time_in_hours * 60 -- Convert hours to minutes
def jogging_days_in_second_week := 5 -- Larry jogs 5 days in the second week
def daily_jogging_time_in_week2 := jogging_days_in_second_week * daily_jogging_time -- Total jogging time in minutes in the second week

-- Theorem statement
theorem larry_jogs_first_week_days : 
  (total_jogging_time_in_minutes - daily_jogging_time_in_week2) / daily_jogging_time = 3 :=
by
  -- Definitions and conditions used above should directly appear from the problem statement
  sorry

end larry_jogs_first_week_days_l75_75965


namespace fraction_power_minus_one_l75_75239

theorem fraction_power_minus_one :
  (5 / 3) ^ 4 - 1 = 544 / 81 := 
by
  sorry

end fraction_power_minus_one_l75_75239


namespace original_area_of_circle_l75_75464

theorem original_area_of_circle
  (A₀ : ℝ) -- original area
  (r₀ r₁ : ℝ) -- original and new radius
  (π : ℝ := 3.14)
  (h_area : A₀ = π * r₀^2)
  (h_area_increase : π * r₁^2 = 9 * A₀)
  (h_circumference_increase : 2 * π * r₁ - 2 * π * r₀ = 50.24) :
  A₀ = 50.24 :=
by
  sorry

end original_area_of_circle_l75_75464


namespace solid_produces_quadrilateral_l75_75166

-- Define the solids and their properties
inductive Solid 
| cone 
| cylinder 
| sphere

-- Define the condition for a plane cut resulting in a quadrilateral cross-section
def can_produce_quadrilateral_cross_section (s : Solid) : Prop :=
  match s with
  | Solid.cone => False
  | Solid.cylinder => True
  | Solid.sphere => False

-- Theorem to prove that only a cylinder can produce a quadrilateral cross-section
theorem solid_produces_quadrilateral : 
  ∃ s : Solid, can_produce_quadrilateral_cross_section s :=
by
  existsi Solid.cylinder
  trivial

end solid_produces_quadrilateral_l75_75166


namespace ball_hits_ground_time_l75_75920

theorem ball_hits_ground_time :
  ∀ t : ℝ, y = -20 * t^2 + 30 * t + 60 → y = 0 → t = (3 + Real.sqrt 57) / 4 := by
  sorry

end ball_hits_ground_time_l75_75920


namespace bird_costs_l75_75616

-- Define the cost of a small bird and a large bird
def cost_small_bird (x : ℕ) := x
def cost_large_bird (x : ℕ) := 2 * x

-- Define total cost calculations for the first and second ladies
def cost_first_lady (x : ℕ) := 5 * cost_large_bird x + 3 * cost_small_bird x
def cost_second_lady (x : ℕ) := 5 * cost_small_bird x + 3 * cost_large_bird x

-- State the main theorem
theorem bird_costs (x : ℕ) (hx : cost_first_lady x = cost_second_lady x + 20) : 
(cost_small_bird x = 10) ∧ (cost_large_bird x = 20) := 
by {
  sorry
}

end bird_costs_l75_75616


namespace rectangle_side_ratio_l75_75686

theorem rectangle_side_ratio (s x y : ℝ) 
  (h1 : 8 * (x * y) = (9 - 1) * s^2) 
  (h2 : s + 4 * y = 3 * s) 
  (h3 : 2 * x + y = 3 * s) : 
  x / y = 2.5 :=
by
  sorry

end rectangle_side_ratio_l75_75686


namespace pens_cost_l75_75453

theorem pens_cost (pens_pack_cost : ℝ) (pens_pack_quantity : ℕ) (total_pens : ℕ) (unit_price : ℝ) (total_cost : ℝ)
  (h1 : pens_pack_cost = 45) (h2 : pens_pack_quantity = 150) (h3 : total_pens = 3600) (h4 : unit_price = pens_pack_cost / pens_pack_quantity)
  (h5 : total_cost = total_pens * unit_price) : total_cost = 1080 := by
  sorry

end pens_cost_l75_75453


namespace extreme_points_f_l75_75289

theorem extreme_points_f (a b : ℝ)
  (h1 : 3 * (-2)^2 + 2 * a * (-2) + b = 0)
  (h2 : 3 * 4^2 + 2 * a * 4 + b = 0) :
  a - b = 21 :=
sorry

end extreme_points_f_l75_75289


namespace time_to_fill_tank_l75_75434

-- Define the rates of the pipes
def rate_first_fill : ℚ := 1 / 15
def rate_second_fill : ℚ := 1 / 15
def rate_outlet_empty : ℚ := -1 / 45

-- Define the combined rate
def combined_rate : ℚ := rate_first_fill + rate_second_fill + rate_outlet_empty

-- Define the time to fill the tank
def fill_time (rate : ℚ) : ℚ := 1 / rate

theorem time_to_fill_tank : fill_time combined_rate = 9 := 
by 
  -- Proof omitted
  sorry

end time_to_fill_tank_l75_75434


namespace intersection_points_l75_75091

variables (a b c : ℝ)
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

theorem intersection_points :
  ∃ p q : ℝ × ℝ, 
    (p = (0, c) ∨ p = (-1, a - b + c)) ∧ 
    (q = (0, c) ∨ q = (-1, a - b + c)) ∧
    p ≠ q ∧
    (∃ x : ℝ, (x, ax^2 + bx + c) = p) ∧
    (∃ x : ℝ, (x, -ax^3 + bx + c) = q) :=
by
  sorry

end intersection_points_l75_75091


namespace problem_statement_l75_75562

theorem problem_statement (x : ℝ) (hx : x^2 + 1/(x^2) = 2) : x^4 + 1/(x^4) = 2 := by
  sorry

end problem_statement_l75_75562


namespace jason_car_cost_l75_75610

theorem jason_car_cost
    (down_payment : ℕ := 8000)
    (monthly_payment : ℕ := 525)
    (months : ℕ := 48)
    (interest_rate : ℝ := 0.05) :
    (down_payment + monthly_payment * months + interest_rate * (monthly_payment * months)) = 34460 := 
by
  sorry

end jason_car_cost_l75_75610


namespace initial_paint_amount_l75_75612

theorem initial_paint_amount (P : ℝ) (h1 : P > 0) (h2 : (1 / 4) * P + (1 / 3) * (3 / 4) * P = 180) : P = 360 := by
  sorry

end initial_paint_amount_l75_75612


namespace cupcakes_frosted_in_10_minutes_l75_75213

theorem cupcakes_frosted_in_10_minutes :
  let cagney_rate := 1 / 25 -- Cagney's rate in cupcakes per second
  let lacey_rate := 1 / 35 -- Lacey's rate in cupcakes per second
  let total_time := 600 -- Total time in seconds for 10 minutes
  let lacey_break := 60 -- Break duration in seconds
  let lacey_work_time := total_time - lacey_break
  let cupcakes_by_cagney := total_time / 25 
  let cupcakes_by_lacey := lacey_work_time / 35
  cupcakes_by_cagney + cupcakes_by_lacey = 39 := 
by {
  sorry
}

end cupcakes_frosted_in_10_minutes_l75_75213


namespace n_plus_one_sum_of_three_squares_l75_75658

theorem n_plus_one_sum_of_three_squares (n x : ℤ) (h1 : n > 1) (h2 : 3 * n + 1 = x^2) :
  ∃ a b c : ℤ, n + 1 = a^2 + b^2 + c^2 :=
by
  sorry

end n_plus_one_sum_of_three_squares_l75_75658


namespace matrix_pow_A_50_l75_75485

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![5, 2], ![-16, -6]]

theorem matrix_pow_A_50 :
  A ^ 50 = ![![301, 100], ![-800, -249]] :=
by
  sorry

end matrix_pow_A_50_l75_75485


namespace unique_records_l75_75583

variable (Samantha_records : Nat)
variable (shared_records : Nat)
variable (Lily_unique_records : Nat)

theorem unique_records (h1 : Samantha_records = 24) (h2 : shared_records = 15) (h3 : Lily_unique_records = 9) :
  let Samantha_unique_records := Samantha_records - shared_records
  Samantha_unique_records + Lily_unique_records = 18 :=
by
  sorry

end unique_records_l75_75583


namespace correct_computation_l75_75236

theorem correct_computation (x : ℕ) (h : x - 20 = 52) : x / 4 = 18 :=
  sorry

end correct_computation_l75_75236


namespace find_x_l75_75996

theorem find_x (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 14) : x = 12 := by
  sorry

end find_x_l75_75996


namespace nurses_count_l75_75563

theorem nurses_count (total personnel_ratio d_ratio n_ratio : ℕ)
  (ratio_eq: personnel_ratio = 280)
  (ratio_condition: d_ratio = 5)
  (person_count: n_ratio = 9) :
  n_ratio * (personnel_ratio / (d_ratio + n_ratio)) = 180 := by
  -- Total personnel = 280
  -- Ratio of doctors to nurses = 5/9
  -- Prove that the number of nurses is 180
  -- sorry is used to skip proof
  sorry

end nurses_count_l75_75563


namespace find_common_real_root_l75_75438

theorem find_common_real_root :
  ∃ (m a : ℝ), (a^2 + m * a + 2 = 0) ∧ (a^2 + 2 * a + m = 0) ∧ m = -3 ∧ a = 1 :=
by
  -- Skipping the proof
  sorry

end find_common_real_root_l75_75438


namespace largest_divisor_same_remainder_l75_75175

theorem largest_divisor_same_remainder (n : ℕ) (h : 17 % n = 30 % n) : n = 13 :=
sorry

end largest_divisor_same_remainder_l75_75175


namespace total_tiles_in_room_l75_75877

theorem total_tiles_in_room (s : ℕ) (hs : 6 * s - 5 = 193) : s^2 = 1089 :=
by sorry

end total_tiles_in_room_l75_75877


namespace tangent_line_through_point_l75_75980

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x

theorem tangent_line_through_point (x y : ℝ) (h₁ : y = 2 * Real.log x - x) (h₂ : (1 : ℝ)  ≠ 0) 
  (h₃ : (-1 : ℝ) ≠ 0):
  (x - y - 2 = 0) :=
sorry

end tangent_line_through_point_l75_75980


namespace ax5_by5_eq_6200_div_29_l75_75431

variables (a b x y : ℝ)

-- Given conditions
axiom h1 : a * x + b * y = 5
axiom h2 : a * x^2 + b * y^2 = 11
axiom h3 : a * x^3 + b * y^3 = 30
axiom h4 : a * x^4 + b * y^4 = 80

-- Statement to prove
theorem ax5_by5_eq_6200_div_29 : a * x^5 + b * y^5 = 6200 / 29 :=
by
  sorry

end ax5_by5_eq_6200_div_29_l75_75431


namespace probability_even_first_odd_second_l75_75375

-- Definitions based on the conditions
def die_sides : Finset ℕ := {1, 2, 3, 4, 5, 6}
def even_numbers : Finset ℕ := {2, 4, 6}
def odd_numbers : Finset ℕ := {1, 3, 5}

-- Probability calculations
def prob_even := (even_numbers.card : ℚ) / (die_sides.card : ℚ)
def prob_odd := (odd_numbers.card : ℚ) / (die_sides.card : ℚ)

-- Proof statement
theorem probability_even_first_odd_second :
  prob_even * prob_odd = 1 / 4 :=
by
  sorry

end probability_even_first_odd_second_l75_75375


namespace equal_roots_quadratic_l75_75978

theorem equal_roots_quadratic (k : ℝ) : (∃ (x : ℝ), x*(x + 2) + k = 0 ∧ ∀ y z, (y, z) = (x, x)) → k = 1 :=
sorry

end equal_roots_quadratic_l75_75978


namespace length_BD_l75_75320

/-- Points A, B, C, and D lie on a line in that order. We are given:
  AB = 2 cm,
  AC = 5 cm, and
  CD = 3 cm.
Then, we need to show that the length of BD is 6 cm. -/
theorem length_BD :
  ∀ (A B C D : ℕ),
  A + B = 2 → A + C = 5 → C + D = 3 →
  D - B = 6 :=
by
  intros A B C D h1 h2 h3
  -- Proof steps to be filled in
  sorry

end length_BD_l75_75320


namespace remaining_value_subtract_70_percent_from_4500_l75_75335

theorem remaining_value_subtract_70_percent_from_4500 (num : ℝ) 
  (h : 0.36 * num = 2376) : 4500 - 0.70 * num = -120 :=
by
  sorry

end remaining_value_subtract_70_percent_from_4500_l75_75335


namespace tickets_left_l75_75196

-- Definitions for the conditions given in the problem
def tickets_whack_a_mole : ℕ := 32
def tickets_skee_ball : ℕ := 25
def tickets_spent_on_hat : ℕ := 7

-- The main proof statement to verify
theorem tickets_left : tickets_whack_a_mole + tickets_skee_ball - tickets_spent_on_hat = 50 :=
by
  sorry

end tickets_left_l75_75196


namespace fourth_quadrant_for_m_negative_half_x_axis_for_m_upper_half_plane_for_m_l75_75960

open Complex

def inFourthQuadrant (m : ℝ) : Prop :=
  (m^2 - 8*m + 15) > 0 ∧ (m^2 + 3*m - 28) < 0

def onNegativeHalfXAxis (m : ℝ) : Prop :=
  (m^2 - 8*m + 15) < 0 ∧ (m^2 + 3*m - 28) = 0

def inUpperHalfPlaneIncludingRealAxis (m : ℝ) : Prop :=
  (m^2 + 3*m - 28) ≥ 0

theorem fourth_quadrant_for_m (m : ℝ) :
  (-7 < m ∧ m < 3) ↔ inFourthQuadrant m := 
sorry

theorem negative_half_x_axis_for_m (m : ℝ) :
  (m = 4) ↔ onNegativeHalfXAxis m :=
sorry

theorem upper_half_plane_for_m (m : ℝ) :
  (m ≤ -7 ∨ m ≥ 4) ↔ inUpperHalfPlaneIncludingRealAxis m :=
sorry

end fourth_quadrant_for_m_negative_half_x_axis_for_m_upper_half_plane_for_m_l75_75960


namespace total_age_proof_l75_75760

noncomputable def total_age : ℕ :=
  let susan := 15
  let arthur := susan + 2
  let bob := 11
  let tom := bob - 3
  let emily := susan / 2
  let david := (arthur + tom + emily) / 3
  susan + arthur + tom + bob + emily + david

theorem total_age_proof : total_age = 70 := by
  unfold total_age
  sorry

end total_age_proof_l75_75760


namespace time_period_simple_interest_l75_75513

theorem time_period_simple_interest 
  (P : ℝ) (R18 R12 : ℝ) (additional_interest : ℝ) (T : ℝ) :
  P = 2500 →
  R18 = 0.18 →
  R12 = 0.12 →
  additional_interest = 300 →
  P * R18 * T = P * R12 * T + additional_interest →
  T = 2 :=
by
  intros P_val R18_val R12_val add_int_val interest_eq
  rw [P_val, R18_val, R12_val, add_int_val] at interest_eq
  -- Continue the proof here
  sorry

end time_period_simple_interest_l75_75513


namespace total_workers_in_workshop_l75_75222

-- Definition of average salary calculation
def average_salary (total_salary : ℕ) (workers : ℕ) : ℕ := total_salary / workers

theorem total_workers_in_workshop :
  ∀ (W T R : ℕ),
  T = 5 →
  average_salary ((W - T) * 750) (W - T) = 700 →
  average_salary (T * 900) T = 900 →
  average_salary (W * 750) W = 750 →
  W = T + R →
  W = 20 :=
by
  sorry

end total_workers_in_workshop_l75_75222


namespace number_of_true_propositions_is_two_l75_75181

def proposition1 (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (1 + x) = f (1 - x)

def proposition2 : Prop :=
∀ x : ℝ, 2 * Real.sin x * Real.cos (abs x) -- minimum period not 1
  -- We need to define proper periodicity which is complex; so here's a simplified representation
  ≠ 2 * Real.sin (x + 1) * Real.cos (abs (x + 1))

def increasing_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) > a n

def proposition3 (k : ℝ) : Prop :=
∀ n : ℕ, n > 0 → increasing_sequence (fun n => n^2 + k * n + 2)

def condition (f : ℝ → ℝ) (k : ℝ) : Prop :=
proposition1 f ∧ proposition2 ∧ proposition3 k

theorem number_of_true_propositions_is_two (f : ℝ → ℝ) (k : ℝ) :
  condition f k → 2 = 2 :=
by
  sorry

end number_of_true_propositions_is_two_l75_75181


namespace workers_are_280_women_l75_75479

variables (W : ℕ) 
          (workers_without_retirement_plan : ℕ := W / 3)
          (women_without_retirement_plan : ℕ := (workers_without_retirement_plan * 1) / 10)
          (workers_with_retirement_plan : ℕ := W * 2 / 3)
          (men_with_retirement_plan : ℕ := (workers_with_retirement_plan * 4) / 10)
          (total_men : ℕ := (workers_without_retirement_plan * 9) / 30)
          (total_workers := total_men / (9 / 30))
          (number_of_women : ℕ := total_workers - 120)

theorem workers_are_280_women : total_workers = 400 ∧ number_of_women = 280 :=
by sorry

end workers_are_280_women_l75_75479


namespace correct_expression_must_hold_l75_75771

variable {f : ℝ → ℝ}

-- Conditions
axiom increasing_function : ∀ x y : ℝ, x < y → f x < f y
axiom positive_function : ∀ x : ℝ, f x > 0

-- Problem Statement
theorem correct_expression_must_hold : 3 * f (-2) > 2 * f (-3) := by
  sorry

end correct_expression_must_hold_l75_75771


namespace smallest_angle_CBD_l75_75827

-- Definitions for given conditions
def angle_ABC : ℝ := 40
def angle_ABD : ℝ := 15

-- Theorem statement
theorem smallest_angle_CBD : ∃ (angle_CBD : ℝ), angle_CBD = angle_ABC - angle_ABD := by
  use 25
  sorry

end smallest_angle_CBD_l75_75827


namespace cost_of_article_l75_75432

variable (C : ℝ)
variable (SP1 SP2 : ℝ)
variable (G : ℝ)

theorem cost_of_article (h1 : SP1 = 380) 
                        (h2 : SP2 = 420)
                        (h3 : SP1 = C + G)
                        (h4 : SP2 = C + G + 0.08 * G) :
  C = 120 :=
by
  sorry

end cost_of_article_l75_75432


namespace large_circle_radius_l75_75386

noncomputable def radius_of_large_circle : ℝ :=
  let r_small := 1
  let side_length := 2 * r_small
  let diagonal_length := Real.sqrt (side_length ^ 2 + side_length ^ 2)
  let radius_large := (diagonal_length / 2) + r_small
  radius_large + r_small

theorem large_circle_radius :
  radius_of_large_circle = Real.sqrt 2 + 2 :=
by
  sorry

end large_circle_radius_l75_75386


namespace fraction_simplification_l75_75633

theorem fraction_simplification :
    1 + (1 / (1 + (1 / (2 + (1 / 3))))) = 17 / 10 := by
  sorry

end fraction_simplification_l75_75633


namespace man_walking_time_l75_75791

theorem man_walking_time (D V_w V_m T : ℝ) (t : ℝ) :
  D = V_w * T →
  D_w = V_m * t →
  D - V_m * t = V_w * (T - t) →
  T - (T - t) = 16 →
  t = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end man_walking_time_l75_75791


namespace problem1_problem2_l75_75581

variable (m n x y : ℝ)

theorem problem1 : 4 * m * n^3 * (2 * m^2 - (3 / 4) * m * n^2) = 8 * m^3 * n^3 - 3 * m^2 * n^5 := sorry

theorem problem2 : (x - 6 * y^2) * (3 * x^3 + y) = 3 * x^4 + x * y - 18 * x^3 * y^2 - 6 * y^3 := sorry

end problem1_problem2_l75_75581


namespace max_receptivity_compare_receptivity_l75_75197

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then -0.1 * x^2 + 2.6 * x + 43
  else if 10 < x ∧ x <= 16 then 59
  else if 16 < x ∧ x <= 30 then -3 * x + 107
  else 0 -- To cover the case when x is outside the given ranges

-- Problem 1
theorem max_receptivity :
  f 10 = 59 ∧ ∀ x, 10 < x ∧ x ≤ 16 → f x = 59 :=
by
  sorry

-- Problem 2
theorem compare_receptivity :
  f 5 > f 20 :=
by
  sorry

end max_receptivity_compare_receptivity_l75_75197


namespace min_distance_between_intersections_range_of_a_l75_75506

variable {a : ℝ}

/-- Given the function f(x) = x^2 - 2ax - 2(a + 1), 
1. Prove that the graph of function f(x) always intersects the x-axis at two distinct points.
2. For all x in the interval (-1, ∞), prove that f(x) + 3 ≥ 0 implies a ≤ sqrt 2 - 1. --/

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x - 2 * (a + 1)

theorem min_distance_between_intersections (a : ℝ) : 
  ∃ x₁ x₂ : ℝ, (f x₁ a = 0) ∧ (f x₂ a = 0) ∧ (x₁ ≠ x₂) ∧ (dist x₁ x₂ = 2) := sorry

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, -1 < x → f x a + 3 ≥ 0) → a ≤ Real.sqrt 2 - 1 := sorry

end min_distance_between_intersections_range_of_a_l75_75506


namespace inscribed_circle_radius_l75_75754

/-- Define a square SEAN with side length 2. -/
def square_side_length : ℝ := 2

/-- Define a quarter-circle of radius 1. -/
def quarter_circle_radius : ℝ := 1

/-- Hypothesis: The radius of the largest circle that can be inscribed in the remaining figure. -/
theorem inscribed_circle_radius :
  let S : ℝ := square_side_length
  let R : ℝ := quarter_circle_radius
  ∃ (r : ℝ), (r = 5 - 3 * Real.sqrt 2) := 
sorry

end inscribed_circle_radius_l75_75754


namespace prime_numbers_eq_l75_75461

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_numbers_eq 
  (p q r : ℕ)
  (hp : is_prime p)
  (hq : is_prime q)
  (hr : is_prime r)
  (h : p * (p - 7) + q * (q - 7) = r * (r - 7)) :
  (p = 2 ∧ q = 5 ∧ r = 7) ∨ (p = 2 ∧ q = 5 ∧ r = 7) ∨
  (p = 7 ∧ q = 5 ∧ r = 5) ∨ (p = 5 ∧ q = 7 ∧ r = 5) ∨
  (p = 5 ∧ q = 7 ∧ r = 2) ∨ (p = 7 ∧ q = 5 ∧ r = 2) ∨
  (p = 7 ∧ q = 3 ∧ r = 3) ∨ (p = 3 ∧ q = 7 ∧ r = 3) ∨
  (∃ (a : ℕ), is_prime a ∧ p = a ∧ q = 7 ∧ r = a) ∨
  (∃ (a : ℕ), is_prime a ∧ p = 7 ∧ q = a ∧ r = a) :=
sorry

end prime_numbers_eq_l75_75461


namespace cars_with_air_bags_l75_75564

/--
On a car lot with 65 cars:
- Some have air-bags.
- 30 have power windows.
- 12 have both air-bag and power windows.
- 2 have neither air-bag nor power windows.

Prove that the number of cars with air-bags is 45.
-/
theorem cars_with_air_bags 
    (total_cars : ℕ)
    (cars_with_power_windows : ℕ)
    (cars_with_both : ℕ)
    (cars_with_neither : ℕ)
    (total_cars_eq : total_cars = 65)
    (cars_with_power_windows_eq : cars_with_power_windows = 30)
    (cars_with_both_eq : cars_with_both = 12)
    (cars_with_neither_eq : cars_with_neither = 2) :
    ∃ (A : ℕ), A = 45 :=
by
  sorry

end cars_with_air_bags_l75_75564


namespace simplify_and_evaluate_expression_l75_75866

theorem simplify_and_evaluate_expression (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -2) (hx3 : x ≠ 2) :
  ( ( (x^2 + 4) / x - 4) / ((x^2 - 4) / (x^2 + 2 * x)) = x - 2 ) ∧ 
  ( (x = 1) → ((x^2 + 4) / x - 4) / ((x^2 - 4) / (x^2 + 2 * x)) = -1 ) :=
by
  sorry

end simplify_and_evaluate_expression_l75_75866


namespace correct_calculation_l75_75973

theorem correct_calculation (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 :=
by sorry

end correct_calculation_l75_75973


namespace kelsey_remaining_half_speed_l75_75584

variable (total_hours : ℝ) (first_half_speed : ℝ) (total_distance : ℝ) (remaining_half_time : ℝ) (remaining_half_distance : ℝ)

axiom h1 : total_hours = 10
axiom h2 : first_half_speed = 25
axiom h3 : total_distance = 400
axiom h4 : remaining_half_time = total_hours - total_distance / (2 * first_half_speed)
axiom h5 : remaining_half_distance = total_distance / 2

theorem kelsey_remaining_half_speed :
  remaining_half_distance / remaining_half_time = 100
:=
by
  sorry

end kelsey_remaining_half_speed_l75_75584


namespace smallest_number_of_three_integers_l75_75151

theorem smallest_number_of_three_integers 
  (a b c : ℕ) 
  (hpos1 : 0 < a) (hpos2 : 0 < b) (hpos3 : 0 < c) 
  (hmean : (a + b + c) / 3 = 24)
  (hmed : b = 23)
  (hlargest : b + 4 = c) 
  : a = 22 :=
by
  sorry

end smallest_number_of_three_integers_l75_75151


namespace solve_for_y_l75_75595

theorem solve_for_y 
  (a b c d y : ℚ) 
  (h₀ : a ≠ b) 
  (h₁ : a ≠ 0) 
  (h₂ : c ≠ d) 
  (h₃ : (b + y) / (a + y) = d / c) : 
  y = (a * d - b * c) / (c - d) :=
by
  sorry

end solve_for_y_l75_75595


namespace proof_problem_l75_75120

-- Condition for the first part: a quadratic inequality having a solution set
def quadratic_inequality (a : ℝ) :=
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → a * x^2 - 3 * x + 2 ≤ 0

-- Condition for the second part: the solution set of a rational inequality
def rational_inequality_solution (a : ℝ) (b : ℝ) :=
  ∀ x : ℝ, (x + 3) / (a * x - b) > 0 ↔ (x < -3 ∨ x > 2)

theorem proof_problem {a : ℝ} {b : ℝ} :
  (quadratic_inequality a → a = 1 ∧ b = 2) ∧ 
  (rational_inequality_solution 1 2) :=
by
  sorry

end proof_problem_l75_75120


namespace total_students_is_100_l75_75904

-- Definitions of the conditions
def largest_class_students : Nat := 24
def decrement : Nat := 2

-- Let n be the number of classes, which is given by 5
def num_classes : Nat := 5

-- The number of students in each class
def students_in_class (n : Nat) : Nat := 
  if n = 1 then largest_class_students
  else largest_class_students - decrement * (n - 1)

-- Total number of students in the school
def total_students : Nat :=
  List.sum (List.map students_in_class (List.range num_classes))

-- Theorem to prove that total_students equals 100
theorem total_students_is_100 : total_students = 100 := by
  sorry

end total_students_is_100_l75_75904


namespace eq1_solution_eq2_solution_l75_75591

theorem eq1_solution (x : ℝ) : (3 * x * (x - 1) = 2 - 2 * x) ↔ (x = 1 ∨ x = -2/3) :=
sorry

theorem eq2_solution (x : ℝ) : (3 * x^2 - 6 * x + 2 = 0) ↔ (x = 1 + (Real.sqrt 3) / 3 ∨ x = 1 - (Real.sqrt 3) / 3) :=
sorry

end eq1_solution_eq2_solution_l75_75591


namespace derivative_at_3_l75_75291

noncomputable def f (x : ℝ) := x^2

theorem derivative_at_3 : deriv f 3 = 6 := by
  sorry

end derivative_at_3_l75_75291


namespace pow_sum_nineteen_eq_zero_l75_75463

variable {a b c : ℝ}

theorem pow_sum_nineteen_eq_zero (h₁ : a + b + c = 0) (h₂ : a^3 + b^3 + c^3 = 0) : a^19 + b^19 + c^19 = 0 :=
sorry

end pow_sum_nineteen_eq_zero_l75_75463


namespace total_molecular_weight_is_1317_12_l75_75195

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_S : ℝ := 32.06
def atomic_weight_H : ℝ := 1.01
def atomic_weight_O : ℝ := 16.00
def atomic_weight_C : ℝ := 12.01

def molecular_weight_Al2S3 : ℝ := (2 * atomic_weight_Al) + (3 * atomic_weight_S)
def molecular_weight_H2O : ℝ := (2 * atomic_weight_H) + (1 * atomic_weight_O)
def molecular_weight_CO2 : ℝ := (1 * atomic_weight_C) + (2 * atomic_weight_O)

def total_weight_7_Al2S3 : ℝ := 7 * molecular_weight_Al2S3
def total_weight_5_H2O : ℝ := 5 * molecular_weight_H2O
def total_weight_4_CO2 : ℝ := 4 * molecular_weight_CO2

def total_molecular_weight : ℝ := total_weight_7_Al2S3 + total_weight_5_H2O + total_weight_4_CO2

theorem total_molecular_weight_is_1317_12 : total_molecular_weight = 1317.12 := by
  sorry

end total_molecular_weight_is_1317_12_l75_75195


namespace find_x_l75_75604

variables (a b c k : ℝ) (h : k ≠ 0)

theorem find_x (x y z : ℝ)
  (h1 : (xy + k) / (x + y) = a)
  (h2 : (xz + k) / (x + z) = b)
  (h3 : (yz + k) / (y + z) = c) :
  x = 2 * a * b * c * d / (b * (a * c - k) + c * (a * b - k) - a * (b * c - k)) := sorry

end find_x_l75_75604


namespace geometric_sequence_problem_l75_75676

noncomputable def a₂ (a₁ q : ℝ) : ℝ := a₁ * q
noncomputable def a₃ (a₁ q : ℝ) : ℝ := a₁ * q^2
noncomputable def a₄ (a₁ q : ℝ) : ℝ := a₁ * q^3
noncomputable def S₆ (a₁ q : ℝ) : ℝ := (a₁ * (1 - q^6)) / (1 - q)

theorem geometric_sequence_problem
  (a₁ q : ℝ)
  (h1 : a₁ * a₂ a₁ q * a₃ a₁ q = 27)
  (h2 : a₂ a₁ q + a₄ a₁ q = 30)
  : ((a₁ = 1 ∧ q = 3) ∨ (a₁ = -1 ∧ q = -3))
    ∧ (if a₁ = 1 ∧ q = 3 then S₆ a₁ q = 364 else true)
    ∧ (if a₁ = -1 ∧ q = -3 then S₆ a₁ q = -182 else true) :=
by
  -- Proof goes here
  sorry

end geometric_sequence_problem_l75_75676


namespace morgan_first_sat_score_l75_75435

theorem morgan_first_sat_score (x : ℝ) (h : 1.10 * x = 1100) : x = 1000 :=
sorry

end morgan_first_sat_score_l75_75435


namespace find_m_l75_75399

open Real

/-- Define Circle C1 and C2 as having the given equations
and verify their internal tangency to find the possible m values -/
theorem find_m (m : ℝ) :
  (∃ (x y : ℝ), (x - m)^2 + (y + 2)^2 = 9) ∧ 
  (∃ (x y : ℝ), (x + 1)^2 + (y - m)^2 = 4) ∧ 
  (by exact (sqrt ((m + 1)^2 + (-2 - m)^2)) = 3 - 2) → 
  m = -2 ∨ m = -1 := 
sorry -- Proof is omitted

end find_m_l75_75399


namespace sum_even_squares_sum_odd_squares_l75_75511

open scoped BigOperators

def sumOfSquaresEven (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, (2 * (i + 1))^2

def sumOfSquaresOdd (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, (2 * i + 1)^2

theorem sum_even_squares (n : ℕ) :
  sumOfSquaresEven n = (2 * n * (n - 1) * (2 * n - 1)) / 3 := by
    sorry

theorem sum_odd_squares (n : ℕ) :
  sumOfSquaresOdd n = (n * (4 * n^2 - 1)) / 3 := by
    sorry

end sum_even_squares_sum_odd_squares_l75_75511


namespace cakes_bought_l75_75650

theorem cakes_bought (initial_cakes remaining_cakes : ℕ) (h_initial : initial_cakes = 155) (h_remaining : remaining_cakes = 15) : initial_cakes - remaining_cakes = 140 :=
by {
  sorry
}

end cakes_bought_l75_75650


namespace william_max_riding_time_l75_75790

theorem william_max_riding_time (x : ℝ) :
  (2 * x + 2 * 1.5 + 2 * (1 / 2 * x) = 21) → (x = 6) :=
by
  sorry

end william_max_riding_time_l75_75790


namespace range_of_b_l75_75370

theorem range_of_b :
  (∀ b, (∀ x : ℝ, x ≥ 1 → Real.log (2^x - b) ≥ 0) → b ≤ 1) :=
sorry

end range_of_b_l75_75370


namespace men_in_first_group_l75_75080

theorem men_in_first_group (x : ℕ) :
  (20 * 48 = x * 80) → x = 12 :=
by
  intro h_eq
  have : x = (20 * 48) / 80 := sorry
  exact this

end men_in_first_group_l75_75080


namespace price_of_orange_l75_75546

-- Define relevant conditions
def price_apple : ℝ := 1.50
def morning_apples : ℕ := 40
def morning_oranges : ℕ := 30
def afternoon_apples : ℕ := 50
def afternoon_oranges : ℕ := 40
def total_sales : ℝ := 205

-- Define the proof problem
theorem price_of_orange (O : ℝ) 
  (h : (morning_apples * price_apple + morning_oranges * O) + 
       (afternoon_apples * price_apple + afternoon_oranges * O) = total_sales) : 
  O = 1 :=
by
  sorry

end price_of_orange_l75_75546


namespace original_price_l75_75597

theorem original_price (x: ℝ) (h1: x * 1.1 * 0.8 = 2) : x = 25 / 11 :=
by
  sorry

end original_price_l75_75597


namespace multiplication_correct_l75_75675

theorem multiplication_correct :
  375680169467 * 4565579427629 = 1715110767607750737263 :=
  by sorry

end multiplication_correct_l75_75675


namespace rotate_D_90_clockwise_l75_75571

-- Define the point D with its coordinates.
structure Point where
  x : Int
  y : Int

-- Define the original point D.
def D : Point := { x := -3, y := -8 }

-- Define the rotation transformation.
def rotate90Clockwise (p : Point) : Point :=
  { x := p.y, y := -p.x }

-- Statement to be proven.
theorem rotate_D_90_clockwise :
  rotate90Clockwise D = { x := -8, y := 3 } :=
sorry

end rotate_D_90_clockwise_l75_75571


namespace simplify_expression_l75_75187

theorem simplify_expression : 
  (1 / (1 / (1 / 3) ^ 1 + 1 / (1 / 3) ^ 2 + 1 / (1 / 3) ^ 3 + 1 / (1 / 3) ^ 4)) = 1 / 120 :=
by
  sorry

end simplify_expression_l75_75187


namespace cylinder_surface_area_is_128pi_l75_75286

noncomputable def cylinder_total_surface_area (h r : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

theorem cylinder_surface_area_is_128pi :
  cylinder_total_surface_area 12 4 = 128 * Real.pi :=
by
  sorry

end cylinder_surface_area_is_128pi_l75_75286


namespace log_product_eq_3_div_4_l75_75098

theorem log_product_eq_3_div_4 : (Real.log 3 / Real.log 4) * (Real.log 8 / Real.log 9) = 3 / 4 :=
by
  sorry

end log_product_eq_3_div_4_l75_75098


namespace B_pow_97_l75_75974

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 1, 0],
  ![0, 0, 1],
  ![1, 0, 0]
]

theorem B_pow_97 : B ^ 97 = B := by
  sorry

end B_pow_97_l75_75974


namespace john_total_payment_l75_75049

theorem john_total_payment :
  let cost_per_appointment := 400
  let total_appointments := 3
  let pet_insurance_cost := 100
  let insurance_coverage := 0.80
  let first_appointment_cost := cost_per_appointment
  let subsequent_appointments := total_appointments - 1
  let subsequent_appointments_cost := subsequent_appointments * cost_per_appointment
  let covered_cost := subsequent_appointments_cost * insurance_coverage
  let uncovered_cost := subsequent_appointments_cost - covered_cost
  let total_cost := first_appointment_cost + pet_insurance_cost + uncovered_cost
  total_cost = 660 :=
by
  sorry

end john_total_payment_l75_75049


namespace loss_percentage_is_17_l75_75068

noncomputable def loss_percentage (CP SP : ℝ) := ((CP - SP) / CP) * 100

theorem loss_percentage_is_17 :
  let CP : ℝ := 1500
  let SP : ℝ := 1245
  loss_percentage CP SP = 17 :=
by
  sorry

end loss_percentage_is_17_l75_75068


namespace correct_statements_l75_75945

variable (P Q : Prop)

-- Define statements
def is_neg_false_if_orig_true := (P → ¬P) = False
def is_converse_not_nec_true_if_orig_true := (P → Q) → ¬(Q → P)
def is_neg_true_if_converse_true := (Q → P) → (¬P → ¬Q)
def is_neg_true_if_contrapositive_true := (¬Q → ¬P) → (¬P → False)

-- Main proposition
theorem correct_statements : 
  is_converse_not_nec_true_if_orig_true P Q ∧ 
  is_neg_true_if_converse_true P Q :=
by
  sorry

end correct_statements_l75_75945


namespace line_parabola_one_point_l75_75732

theorem line_parabola_one_point (k : ℝ) :
  (∃ x y : ℝ, y = k * x + 2 ∧ y^2 = 8 * x) 
  → (k = 0 ∨ k = 1) := 
by 
  sorry

end line_parabola_one_point_l75_75732


namespace number_of_students_l75_75382

theorem number_of_students (B S : ℕ) 
  (h1 : S = 9 * B + 1) 
  (h2 : S = 10 * B - 10) : 
  S = 100 := 
by 
  { sorry }

end number_of_students_l75_75382


namespace kitten_length_after_4_months_l75_75145

theorem kitten_length_after_4_months
  (initial_length : ℕ)
  (doubled_length_2_weeks : ℕ)
  (final_length_4_months : ℕ)
  (h1 : initial_length = 4)
  (h2 : doubled_length_2_weeks = initial_length * 2)
  (h3 : final_length_4_months = doubled_length_2_weeks * 2) :
  final_length_4_months = 16 := 
by
  sorry

end kitten_length_after_4_months_l75_75145


namespace cos_six_arccos_two_fifths_l75_75356

noncomputable def arccos (x : ℝ) : ℝ := Real.arccos x
noncomputable def cos (x : ℝ) : ℝ := Real.cos x

theorem cos_six_arccos_two_fifths : cos (6 * arccos (2 / 5)) = 12223 / 15625 := 
by
  sorry

end cos_six_arccos_two_fifths_l75_75356


namespace solution_set_inequalities_l75_75625

theorem solution_set_inequalities (a b x : ℝ) (h1 : ∃ x, x > a ∧ x < b) :
  (x < 1 - a ∧ x < 1 - b) ↔ x < 1 - b :=
by
  sorry

end solution_set_inequalities_l75_75625


namespace find_b_for_square_binomial_l75_75710

theorem find_b_for_square_binomial 
  (b : ℝ)
  (u t : ℝ)
  (h₁ : u^2 = 4)
  (h₂ : 2 * t * u = 8)
  (h₃ : b = t^2) : b = 4 := 
  sorry

end find_b_for_square_binomial_l75_75710


namespace sum_of_integers_is_34_l75_75208

theorem sum_of_integers_is_34 (a b : ℕ) (h1 : a - b = 6) (h2 : a * b = 272) (h3a : a > 0) (h3b : b > 0) : a + b = 34 :=
  sorry

end sum_of_integers_is_34_l75_75208


namespace modular_inverse_sum_correct_l75_75677

theorem modular_inverse_sum_correct :
  (3 * 8 + 9 * 13) % 56 = 29 :=
by
  sorry

end modular_inverse_sum_correct_l75_75677


namespace sum_of_cosines_bounds_l75_75611

theorem sum_of_cosines_bounds (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : 0 ≤ x₁ ∧ x₁ ≤ π / 2)
  (h₂ : 0 ≤ x₂ ∧ x₂ ≤ π / 2)
  (h₃ : 0 ≤ x₃ ∧ x₃ ≤ π / 2)
  (h₄ : 0 ≤ x₄ ∧ x₄ ≤ π / 2)
  (h₅ : 0 ≤ x₅ ∧ x₅ ≤ π / 2)
  (sum_sines_eq : Real.sin x₁ + Real.sin x₂ + Real.sin x₃ + Real.sin x₄ + Real.sin x₅ = 3) : 
  2 ≤ Real.cos x₁ + Real.cos x₂ + Real.cos x₃ + Real.cos x₄ + Real.cos x₅ ∧ 
      Real.cos x₁ + Real.cos x₂ + Real.cos x₃ + Real.cos x₄ + Real.cos x₅ ≤ 4 :=
by
  sorry

end sum_of_cosines_bounds_l75_75611


namespace intersection_l75_75053

noncomputable def M : Set ℝ := { x : ℝ | Real.sqrt (x + 1) ≥ 0 }
noncomputable def N : Set ℝ := { x : ℝ | x^2 + x - 2 < 0 }

theorem intersection (x : ℝ) : x ∈ (M ∩ N) ↔ -1 ≤ x ∧ x < 1 := by
  sorry

end intersection_l75_75053


namespace circle_integer_points_l75_75227

theorem circle_integer_points (m n : ℤ) (h : ∃ m n : ℤ, m^2 + n^2 = r ∧ 
  ∃ p q : ℤ, m^2 + n^2 = p ∧ ∃ s t : ℤ, m^2 + n^2 = q ∧ ∃ u v : ℤ, m^2 + n^2 = s ∧ 
  ∃ j k : ℤ, m^2 + n^2 = t ∧ ∃ l w : ℤ, m^2 + n^2 = u ∧ ∃ x y : ℤ, m^2 + n^2 = v ∧ 
  ∃ i b : ℤ, m^2 + n^2 = w ∧ ∃ c d : ℤ, m^2 + n^2 = b ) :
  ∃ r, r = 25 := by
    sorry

end circle_integer_points_l75_75227


namespace intersection_of_A_and_B_l75_75333

open Set

def A : Set ℤ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℤ := {-1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l75_75333


namespace find_second_number_l75_75672

theorem find_second_number (x y z : ℚ) (h_sum : x + y + z = 120)
  (h_ratio1 : x = (3 / 4) * y) (h_ratio2 : z = (7 / 4) * y) :
  y = 240 / 7 :=
by {
  -- Definitions provided from conditions
  sorry  -- Proof omitted
}

end find_second_number_l75_75672


namespace coords_of_P_max_PA_distance_l75_75878

open Real

noncomputable def A : (ℝ × ℝ) := (0, -5)

def on_circle (P : ℝ × ℝ) : Prop :=
  ∃ x y, x = P.1 ∧ y = P.2 ∧ (x - 2)^2 + (y + 3)^2 = 2

def max_PA_distance (P : (ℝ × ℝ)) : Prop :=
  dist P A = max (dist (3, -2) A) (dist (1, -4) A)

theorem coords_of_P_max_PA_distance (P : (ℝ × ℝ)) :
  on_circle P →
  max_PA_distance P →
  P = (3, -2) :=
  sorry

end coords_of_P_max_PA_distance_l75_75878


namespace total_birds_correct_l75_75700

-- Define the conditions
def number_of_trees : ℕ := 7
def blackbirds_per_tree : ℕ := 3
def magpies : ℕ := 13

-- Define the total number of blackbirds using the conditions
def total_blackbirds : ℕ := number_of_trees * blackbirds_per_tree

-- Define the total number of birds using the total number of blackbirds and the number of magpies
def total_birds : ℕ := total_blackbirds + magpies

-- The theorem statement that should be proven
theorem total_birds_correct : total_birds = 34 := 
sorry

end total_birds_correct_l75_75700


namespace linear_function_does_not_pass_first_quadrant_l75_75636

theorem linear_function_does_not_pass_first_quadrant (k b : ℝ) (h : ∀ x : ℝ, y = k * x + b) :
  k = -1 → b = -2 → ¬∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = k * x + b :=
by
  sorry

end linear_function_does_not_pass_first_quadrant_l75_75636


namespace neg_p_true_l75_75752

theorem neg_p_true :
  ∀ (x : ℝ), -2 < x ∧ x < 2 → |x - 1| + |x + 2| < 6 :=
by
  sorry

end neg_p_true_l75_75752


namespace sum_of_sequence_l75_75258

theorem sum_of_sequence (n : ℕ) (h : n ≥ 2) 
  (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, n ≥ 2 → S n * S (n-1) + a n = 0) :
  S n = 2 / (2 * n - 1) := by
  sorry

end sum_of_sequence_l75_75258


namespace boat_speed_still_water_l75_75034

theorem boat_speed_still_water (v c : ℝ) (h1 : v + c = 13) (h2 : v - c = 4) : v = 8.5 :=
by sorry

end boat_speed_still_water_l75_75034


namespace base_conversion_equivalence_l75_75122

theorem base_conversion_equivalence :
  ∃ (n : ℕ), (∃ (C B : ℕ), C < 9 ∧ B < 6 ∧ n = 9 * C + B) ∧
             (∃ (C B : ℕ), C < 9 ∧ B < 6 ∧ n = 6 * B + C) ∧
             n = 0 := 
by 
  sorry

end base_conversion_equivalence_l75_75122


namespace find_inverse_mod_36_l75_75135

-- Given condition
def inverse_mod_17 := (17 * 23) % 53 = 1

-- Definition for the problem statement
def inverse_mod_36 : Prop := (36 * 30) % 53 = 1

theorem find_inverse_mod_36 (h : inverse_mod_17) : inverse_mod_36 :=
sorry

end find_inverse_mod_36_l75_75135


namespace inequality_property_l75_75374

theorem inequality_property (a b : ℝ) (h : a > b) : -5 * a < -5 * b := sorry

end inequality_property_l75_75374


namespace symmetric_circle_eq_of_given_circle_eq_l75_75614

theorem symmetric_circle_eq_of_given_circle_eq
  (x y : ℝ)
  (eq1 : (x - 1)^2 + (y - 2)^2 = 1)
  (line_eq : y = x) :
  (x - 2)^2 + (y - 1)^2 = 1 := by
  sorry

end symmetric_circle_eq_of_given_circle_eq_l75_75614


namespace negate_even_condition_l75_75095

theorem negate_even_condition (a b c : ℤ) :
  (¬(∀ a b c : ℤ, ∃ x : ℚ, a * x^2 + b * x + c = 0 → Even a ∧ Even b ∧ Even c)) →
  (¬Even a ∨ ¬Even b ∨ ¬Even c) :=
by
  sorry

end negate_even_condition_l75_75095


namespace cos_eight_arccos_one_fourth_l75_75646

theorem cos_eight_arccos_one_fourth :
  Real.cos (8 * Real.arccos (1 / 4)) = 172546 / 1048576 :=
sorry

end cos_eight_arccos_one_fourth_l75_75646


namespace geometric_mean_2_6_l75_75393

theorem geometric_mean_2_6 : ∃ x : ℝ, x^2 = 2 * 6 ∧ (x = 2 * Real.sqrt 3 ∨ x = - (2 * Real.sqrt 3)) :=
by
  sorry

end geometric_mean_2_6_l75_75393


namespace intersection_of_A_and_B_l75_75879

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {x | ∃ a ∈ A, x = 2 * a - 1}

-- The main statement to prove
theorem intersection_of_A_and_B : A ∩ B = {1, 3} :=
by {
  sorry
}

end intersection_of_A_and_B_l75_75879


namespace find_13_points_within_radius_one_l75_75067

theorem find_13_points_within_radius_one (points : Fin 25 → ℝ × ℝ)
  (h : ∀ i j k : Fin 25, min (dist (points i) (points j)) (min (dist (points i) (points k)) (dist (points j) (points k))) < 1) :
  ∃ (subset : Finset (Fin 25)), subset.card = 13 ∧ ∃ (center : ℝ × ℝ), ∀ i ∈ subset, dist (points i) center < 1 :=
  sorry

end find_13_points_within_radius_one_l75_75067


namespace train_pass_tree_in_time_l75_75874

-- Definitions from the given conditions
def train_length : ℚ := 270  -- length in meters
def train_speed_km_per_hr : ℚ := 108  -- speed in km/hr

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s (v : ℚ) : ℚ := v * (5 / 18)

-- Speed of the train in m/s
def train_speed_m_per_s : ℚ := km_per_hr_to_m_per_s train_speed_km_per_hr

-- Question translated into a proof problem
theorem train_pass_tree_in_time :
  train_length / train_speed_m_per_s = 9 :=
by
  sorry

end train_pass_tree_in_time_l75_75874


namespace probability_of_same_color_correct_l75_75514

def total_plates : ℕ := 13
def red_plates : ℕ := 7
def blue_plates : ℕ := 6

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def total_ways_to_choose_two : ℕ := choose total_plates 2
noncomputable def ways_to_choose_two_red : ℕ := choose red_plates 2
noncomputable def ways_to_choose_two_blue : ℕ := choose blue_plates 2

noncomputable def ways_to_choose_two_same_color : ℕ :=
  ways_to_choose_two_red + ways_to_choose_two_blue

noncomputable def probability_same_color : ℚ :=
  ways_to_choose_two_same_color / total_ways_to_choose_two

theorem probability_of_same_color_correct :
  probability_same_color = 4 / 9 := by
  sorry

end probability_of_same_color_correct_l75_75514


namespace line_parallel_eq_l75_75100

theorem line_parallel_eq (x y : ℝ) (h1 : 3 * x - y = 6) (h2 : x = -2 ∧ y = 3) :
  ∃ m b, m = 3 ∧ b = 9 ∧ y = m * x + b :=
by
  sorry

end line_parallel_eq_l75_75100


namespace evaluate_f_at_2_l75_75833

def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1

theorem evaluate_f_at_2 : f 2 = 259 := 
by
  -- Substitute x = 2 into the polynomial and simplify the expression.
  sorry

end evaluate_f_at_2_l75_75833


namespace has_real_root_neg_one_l75_75560

theorem has_real_root_neg_one : 
  (-1)^2 - (-1) - 2 = 0 :=
by 
  sorry

end has_real_root_neg_one_l75_75560


namespace sum_of_values_of_n_l75_75348

theorem sum_of_values_of_n (n₁ n₂ : ℚ) (h1 : 3 * n₁ - 8 = 5) (h2 : 3 * n₂ - 8 = -5) : n₁ + n₂ = 16 / 3 := 
by {
  -- Use the provided conditions to solve the problem
  sorry 
}

end sum_of_values_of_n_l75_75348


namespace number_of_numbers_in_last_group_l75_75119

theorem number_of_numbers_in_last_group :
  ∃ n : ℕ, (60 * 13) = (57 * 6) + 50 + (61 * n) ∧ n = 6 :=
sorry

end number_of_numbers_in_last_group_l75_75119


namespace ordering_y1_y2_y3_l75_75444

-- Conditions
def A (y₁ : ℝ) : Prop := ∃ b : ℝ, y₁ = -4^2 + 2*4 + b
def B (y₂ : ℝ) : Prop := ∃ b : ℝ, y₂ = -(-1)^2 + 2*(-1) + b
def C (y₃ : ℝ) : Prop := ∃ b : ℝ, y₃ = -(1)^2 + 2*1 + b

-- Question translated to a proof problem
theorem ordering_y1_y2_y3 (y₁ y₂ y₃ : ℝ) :
  A y₁ → B y₂ → C y₃ → y₁ < y₂ ∧ y₂ < y₃ :=
sorry

end ordering_y1_y2_y3_l75_75444


namespace path_count_l75_75112

theorem path_count (f : ℕ → (ℤ × ℤ)) :
  (∀ n, (f (n + 1)).1 = (f n).1 + 1 ∨ (f (n + 1)).2 = (f n).2 + 1) ∧
  f 0 = (-6, -6) ∧ f 24 = (6, 6) ∧
  (∀ n, ¬(-3 ≤ (f n).1 ∧ (f n).1 ≤ 3 ∧ -3 ≤ (f n).2 ∧ (f n).2 ≤ 3)) →
  ∃ N, N = 2243554 :=
by {
  sorry
}

end path_count_l75_75112


namespace find_s_l75_75242

noncomputable def g (x : ℝ) (p q r s : ℝ) : ℝ := x^4 + p * x^3 + q * x^2 + r * x + s

theorem find_s (p q r s : ℝ)
  (h1 : ∀ (x : ℝ), g x p q r s = (x + 1) * (x + 10) * (x + 10) * (x + 10))
  (h2 : p + q + r + s = 2673) :
  s = 1000 := 
  sorry

end find_s_l75_75242


namespace more_boys_than_girls_l75_75873

noncomputable def class1_4th_girls : ℕ := 12
noncomputable def class1_4th_boys : ℕ := 13
noncomputable def class2_4th_girls : ℕ := 15
noncomputable def class2_4th_boys : ℕ := 11

noncomputable def class1_5th_girls : ℕ := 9
noncomputable def class1_5th_boys : ℕ := 13
noncomputable def class2_5th_girls : ℕ := 10
noncomputable def class2_5th_boys : ℕ := 11

noncomputable def total_4th_girls : ℕ := class1_4th_girls + class2_4th_girls
noncomputable def total_4th_boys : ℕ := class1_4th_boys + class2_4th_boys

noncomputable def total_5th_girls : ℕ := class1_5th_girls + class2_5th_girls
noncomputable def total_5th_boys : ℕ := class1_5th_boys + class2_5th_boys

noncomputable def total_girls : ℕ := total_4th_girls + total_5th_girls
noncomputable def total_boys : ℕ := total_4th_boys + total_5th_boys

theorem more_boys_than_girls :
  (total_boys - total_girls) = 2 :=
by
  -- placeholder for the proof
  sorry

end more_boys_than_girls_l75_75873


namespace outfits_count_l75_75728

-- Definitions of various clothing counts
def numRedShirts : ℕ := 7
def numGreenShirts : ℕ := 3
def numPants : ℕ := 8
def numBlueShoes : ℕ := 5
def numRedShoes : ℕ := 5
def numGreenHats : ℕ := 10
def numRedHats : ℕ := 6

-- Statement of the theorem based on the problem description
theorem outfits_count :
  (numRedShirts * numPants * numBlueShoes * numGreenHats) + 
  (numGreenShirts * numPants * (numBlueShoes + numRedShoes) * numRedHats) = 4240 := 
by
  -- No proof required, only the statement is needed
  sorry

end outfits_count_l75_75728


namespace rosie_pies_l75_75377

def number_of_pies (apples : ℕ) : ℕ := sorry

theorem rosie_pies (h : number_of_pies 9 = 2) : number_of_pies 27 = 6 :=
by sorry

end rosie_pies_l75_75377


namespace ordered_pairs_satisfy_conditions_l75_75398

theorem ordered_pairs_satisfy_conditions :
  ∀ (a b : ℕ), 0 < a → 0 < b → (a^2 + b^2 + 25 = 15 * a * b) → Nat.Prime (a^2 + a * b + b^2) →
  (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) :=
by
  intros a b ha hb h1 h2
  sorry

end ordered_pairs_satisfy_conditions_l75_75398


namespace radius_range_l75_75158

noncomputable def circle_eq (x y r : ℝ) := x^2 + y^2 = r^2

def point_P_on_line_AB (m n : ℝ) := 4 * m + 3 * n - 24 = 0

def point_P_in_interval (m : ℝ) := 0 ≤ m ∧ m ≤ 6

theorem radius_range {r : ℝ} :
  (∀ (m n x y : ℝ), point_P_in_interval m →
     circle_eq x y r →
     circle_eq ((x + m) / 2) ((y + n) / 2) r → 
     point_P_on_line_AB m n ∧
     (4 * r ^ 2 ≤ (25 / 9) * m ^ 2 - (64 / 3) * m + 64 ∧
     (25 / 9) * m ^ 2 - (64 / 3) * m + 64 ≤ 36 * r ^ 2)) →
  (8 / 3 ≤ r ∧ r < 12 / 5) :=
sorry

end radius_range_l75_75158


namespace parabola_vertex_coordinates_l75_75568

theorem parabola_vertex_coordinates :
  ∀ x : ℝ, (3 * (x - 7) ^ 2 + 5) = 3 * (x - 7) ^ 2 + 5 := by
  sorry

end parabola_vertex_coordinates_l75_75568


namespace quadratic_roots_l75_75936

theorem quadratic_roots (x : ℝ) (h : x^2 - 1 = 3) : x = 2 ∨ x = -2 :=
by
  sorry

end quadratic_roots_l75_75936


namespace eight_in_C_l75_75368

def C : Set ℕ := {x | 1 ≤ x ∧ x ≤ 10}

theorem eight_in_C : 8 ∈ C :=
by {
  sorry
}

end eight_in_C_l75_75368


namespace largest_number_among_options_l75_75475

theorem largest_number_among_options :
  let A := 0.983
  let B := 0.9829
  let C := 0.9831
  let D := 0.972
  let E := 0.9819
  C > A ∧ C > B ∧ C > D ∧ C > E :=
by
  sorry

end largest_number_among_options_l75_75475


namespace rod_volume_proof_l75_75865

-- Definitions based on given conditions
def original_length : ℝ := 2
def increase_in_surface_area : ℝ := 0.6
def rod_volume : ℝ := 0.3

-- Problem statement
theorem rod_volume_proof
  (len : ℝ)
  (inc_surface_area : ℝ)
  (vol : ℝ)
  (h_len : len = original_length)
  (h_inc_surface_area : inc_surface_area = increase_in_surface_area) :
  vol = rod_volume :=
sorry

end rod_volume_proof_l75_75865


namespace average_of_multiples_l75_75682

theorem average_of_multiples (n : ℕ) (hn : n > 0) :
  (60.5 : ℚ) = ((n / 2) * (11 + 11 * n)) / n → n = 10 :=
by
  sorry

end average_of_multiples_l75_75682


namespace find_k_l75_75702

theorem find_k (k : ℝ) : -x^2 - (k + 10) * x - 8 = -(x - 2) * (x - 4) → k = -16 := by
  intro h
  sorry

end find_k_l75_75702


namespace rectangle_perimeter_eq_l75_75992

noncomputable def rectangle_perimeter (z w : ℕ) : ℕ :=
  let longer_side := w
  let shorter_side := (z - w) / 2
  2 * longer_side + 2 * shorter_side

theorem rectangle_perimeter_eq (z w : ℕ) : rectangle_perimeter z w = w + z := by
  sorry

end rectangle_perimeter_eq_l75_75992


namespace drums_filled_per_day_l75_75781

-- Definition of given conditions
def pickers : ℕ := 266
def total_drums : ℕ := 90
def total_days : ℕ := 5

-- Statement to prove
theorem drums_filled_per_day : (total_drums / total_days) = 18 := by
  sorry

end drums_filled_per_day_l75_75781


namespace geometric_sequence_15th_term_l75_75723

theorem geometric_sequence_15th_term :
  let a_1 := 27
  let r := (1 : ℚ) / 6
  let a_15 := a_1 * r ^ 14
  a_15 = 1 / 14155776 := by
  sorry

end geometric_sequence_15th_term_l75_75723


namespace walnut_trees_initial_count_l75_75696

theorem walnut_trees_initial_count (x : ℕ) (h : x + 6 = 10) : x = 4 := 
by
  sorry

end walnut_trees_initial_count_l75_75696


namespace find_B_l75_75017

structure Point where
  x : Int
  y : Int

def vector_sub (p1 p2 : Point) : Point :=
  ⟨p1.x - p2.x, p1.y - p2.y⟩

def O : Point := ⟨0, 0⟩
def A : Point := ⟨-1, 2⟩
def BA : Point := ⟨3, 3⟩
def B : Point := ⟨-4, -1⟩

theorem find_B :
  vector_sub A BA = B :=
by
  sorry

end find_B_l75_75017


namespace bears_on_each_shelf_l75_75795

theorem bears_on_each_shelf (initial_bears : ℕ) (additional_bears : ℕ) (shelves : ℕ) (total_bears : ℕ) (bears_per_shelf : ℕ) :
  initial_bears = 5 → additional_bears = 7 → shelves = 2 → total_bears = initial_bears + additional_bears → bears_per_shelf = total_bears / shelves → bears_per_shelf = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end bears_on_each_shelf_l75_75795


namespace part1_l75_75489

variables {a b c : ℝ}
theorem part1 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h : a / (b + c) = b / (c + a) - c / (a + b)) : 
    b / (c + a) ≥ (Real.sqrt 17 - 1) / 4 :=
sorry

end part1_l75_75489


namespace water_consumption_comparison_l75_75884

-- Define the given conditions
def waterConsumptionWest : ℝ := 21428
def waterConsumptionNonWest : ℝ := 26848.55
def waterConsumptionRussia : ℝ := 302790.13

-- Theorem statement to prove that the water consumption per person matches the given values
theorem water_consumption_comparison :
  waterConsumptionWest = 21428 ∧
  waterConsumptionNonWest = 26848.55 ∧
  waterConsumptionRussia = 302790.13 :=
by
  -- Sorry to skip the proof
  sorry

end water_consumption_comparison_l75_75884


namespace exponent_division_is_equal_l75_75802

variable (a : ℝ) 

theorem exponent_division_is_equal :
  (a^11) / (a^2) = a^9 := 
sorry

end exponent_division_is_equal_l75_75802


namespace original_square_area_l75_75599

theorem original_square_area :
  ∀ (a b : ℕ), 
  (a * a = 24 * 1 * 1 + b * b ∧ 
  ((∃ m n : ℕ, (a + b = m ∧ a - b = n ∧ m * n = 24) ∨ 
  (a + b = n ∧ a - b = m ∧ m * n = 24)))) →
  a * a = 25 :=
by
  sorry

end original_square_area_l75_75599


namespace alpha_arctan_l75_75157

open Real

theorem alpha_arctan {α : ℝ} (h1 : α ∈ Set.Ioo 0 (π/4)) (h2 : tan (α + (π/4)) = 2 * cos (2 * α)) : 
  α = arctan (2 - sqrt 3) := by
  sorry

end alpha_arctan_l75_75157


namespace sin_x_correct_l75_75477

noncomputable def sin_x (a b c : ℝ) (x : ℝ) : ℝ :=
  2 * a * b * c / Real.sqrt (a^4 + 2 * a^2 * b^2 * (c^2 - 1) + b^4)

theorem sin_x_correct (a b c x : ℝ) 
  (h₁ : a > b) 
  (h₂ : b > 0) 
  (h₃ : c > 0) 
  (h₄ : 0 < x ∧ x < Real.pi / 2) 
  (h₅ : Real.tan x = 2 * a * b * c / (a^2 - b^2)) :
  Real.sin x = sin_x a b c x :=
sorry

end sin_x_correct_l75_75477


namespace students_before_intersection_equal_l75_75670

-- Define the conditions
def students_after_stop : Nat := 58
def percentage : Real := 0.40
def percentage_students_entered : Real := 12

-- Define the target number of students before stopping
def students_before_stop (total_after : Nat) (entered : Nat) : Nat :=
  total_after - entered

-- State the proof problem
theorem students_before_intersection_equal :
  ∃ (x : Nat), 
  percentage * (x : Real) = percentage_students_entered ∧ 
  students_before_stop students_after_stop x = 28 :=
by
  sorry

end students_before_intersection_equal_l75_75670


namespace perfect_squares_represented_as_diff_of_consecutive_cubes_l75_75184

theorem perfect_squares_represented_as_diff_of_consecutive_cubes : ∃ (count : ℕ), 
  count = 40 ∧ 
  ∀ n : ℕ, 
  (∃ a : ℕ, a^2 = ( ( n + 1 )^3 - n^3 ) ∧ a^2 < 20000) → count = 40 := by 
sorry

end perfect_squares_represented_as_diff_of_consecutive_cubes_l75_75184


namespace range_of_a_l75_75512

theorem range_of_a (a : ℝ) : (∀ x : ℝ, ¬ (ax^2 - ax + 1 ≤ 0)) ↔ 0 ≤ a ∧ a < 4 :=
by
  sorry

end range_of_a_l75_75512


namespace men_joined_l75_75942

-- Definitions for initial conditions
def initial_men : ℕ := 10
def initial_days : ℕ := 50
def extended_days : ℕ := 25

-- Theorem stating the number of men who joined the camp
theorem men_joined (x : ℕ) 
    (initial_food : initial_men * initial_days = (initial_men + x) * extended_days) : 
    x = 10 := 
sorry

end men_joined_l75_75942


namespace factorize_poly1_factorize_poly2_l75_75240

variable (a b m n : ℝ)

theorem factorize_poly1 : 3 * a^2 - 6 * a * b + 3 * b^2 = 3 * (a - b)^2 :=
sorry

theorem factorize_poly2 : 4 * m^2 - 9 * n^2 = (2 * m - 3 * n) * (2 * m + 3 * n) :=
sorry

end factorize_poly1_factorize_poly2_l75_75240


namespace mark_brought_in_4_times_more_cans_l75_75741

theorem mark_brought_in_4_times_more_cans (M J R : ℕ) (h1 : M = 100) 
  (h2 : J = 2 * R + 5) (h3 : M + J + R = 135) : M / J = 4 :=
by sorry

end mark_brought_in_4_times_more_cans_l75_75741


namespace interpretation_of_neg_two_pow_six_l75_75667

theorem interpretation_of_neg_two_pow_six :
  - (2^6) = -(6 * 2) :=
by
  sorry

end interpretation_of_neg_two_pow_six_l75_75667


namespace mean_proportional_l75_75204

theorem mean_proportional (a b c : ℝ) (ha : a = 1) (hb : b = 2) (h : c ^ 2 = a * b) : c = Real.sqrt 2 :=
by
  sorry

end mean_proportional_l75_75204


namespace find_fraction_divide_equal_l75_75954

theorem find_fraction_divide_equal (x : ℚ) : 
  (3 * x = (1 / (5 / 2))) → (x = 2 / 15) :=
by
  intro h
  sorry

end find_fraction_divide_equal_l75_75954


namespace simplify_expression_l75_75130

theorem simplify_expression (a : ℝ) (h₁ : a ≠ 2) (h₂ : a ≠ -2) : 
  (2 * a / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2)) :=
by
  -- proof to be added
  sorry

end simplify_expression_l75_75130


namespace shortest_distance_between_circles_l75_75983

-- Conditions
def first_circle (x y : ℝ) : Prop := x^2 - 10 * x + y^2 - 4 * y - 7 = 0
def second_circle (x y : ℝ) : Prop := x^2 + 14 * x + y^2 + 6 * y + 49 = 0

-- Goal: Prove the shortest distance between the two circles is 4
theorem shortest_distance_between_circles : 
  -- Given conditions about the equations of the circles
  (∀ x y : ℝ, first_circle x y ↔ (x - 5)^2 + (y - 2)^2 = 36) ∧ 
  (∀ x y : ℝ, second_circle x y ↔ (x + 7)^2 + (y + 3)^2 = 9) →
  -- Assert the shortest distance between the two circles is 4
  13 - (6 + 3) = 4 :=
by
  sorry

end shortest_distance_between_circles_l75_75983


namespace range_of_function_l75_75147

noncomputable def function_range: Set ℝ :=
  { y | ∃ x, y = (1/2)^(x^2 - 2*x + 2) }

theorem range_of_function :
  function_range = {y | 0 < y ∧ y ≤ 1/2} :=
sorry

end range_of_function_l75_75147


namespace coby_travel_time_l75_75429

def travel_time (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

theorem coby_travel_time :
  let wash_to_idaho_distance := 640
  let idaho_to_nevada_distance := 550
  let wash_to_idaho_speed := 80
  let idaho_to_nevada_speed := 50
  travel_time wash_to_idaho_distance wash_to_idaho_speed + travel_time idaho_to_nevada_distance idaho_to_nevada_speed = 19 := by
  sorry

end coby_travel_time_l75_75429


namespace sum_of_fraction_components_l75_75359

def repeating_decimal_to_fraction (a b : ℕ) := a = 35 ∧ b = 99 ∧ Nat.gcd a b = 1

theorem sum_of_fraction_components :
  ∃ a b : ℕ, repeating_decimal_to_fraction a b ∧ a + b = 134 :=
by {
  sorry
}

end sum_of_fraction_components_l75_75359


namespace num_students_above_120_l75_75160

noncomputable def class_size : ℤ := 60
noncomputable def mean_score : ℝ := 110
noncomputable def std_score : ℝ := sorry  -- We do not know σ explicitly
noncomputable def probability_100_to_110 : ℝ := 0.35

def normal_distribution (x : ℝ) : Prop :=
  sorry -- placeholder for the actual normal distribution formula N(110, σ^2)

theorem num_students_above_120 :
  ∃ (students_above_120 : ℤ),
  (class_size = 60) ∧
  (∀ score, normal_distribution score → (100 ≤ score ∧ score ≤ 110) → probability_100_to_110 = 0.35) →
  students_above_120 = 9 :=
sorry

end num_students_above_120_l75_75160


namespace student_score_in_first_subject_l75_75698

theorem student_score_in_first_subject 
  (x : ℝ)  -- Percentage in the first subject
  (w : ℝ)  -- Constant weight (as all subjects have same weight)
  (S2_score : ℝ)  -- Score in the second subject
  (S3_score : ℝ)  -- Score in the third subject
  (target_avg : ℝ) -- Target average score
  (hS2 : S2_score = 70)  -- Second subject score is 70%
  (hS3 : S3_score = 80)  -- Third subject score is 80%
  (havg : (x + S2_score + S3_score) / 3 = target_avg) :  -- The desired average is equal to the target average
  target_avg = 70 → x = 60 :=   -- Target average score is 70%
by
  sorry

end student_score_in_first_subject_l75_75698


namespace janets_garden_area_l75_75856

theorem janets_garden_area :
  ∃ (s l : ℕ), 2 * (s + l) = 24 ∧ (l + 1) = 3 * (s + 1) ∧ 6 * (s + 1 - 1) * 6 * (l + 1 - 1) = 576 := 
by
  sorry

end janets_garden_area_l75_75856


namespace cone_volume_from_half_sector_l75_75603

theorem cone_volume_from_half_sector (r l : ℝ) (h : ℝ) 
    (h_r : r = 3) (h_l : l = 6) (h_h : h = 3 * Real.sqrt 3) : 
    (1 / 3) * Real.pi * r^2 * h = 9 * Real.pi * Real.sqrt 3 := 
by
  -- Sorry to skip the proof
  sorry

end cone_volume_from_half_sector_l75_75603


namespace unoccupied_seats_in_business_class_l75_75629

/-
Define the numbers for each class and the number of people in each.
-/
def first_class_seats : Nat := 10
def business_class_seats : Nat := 30
def economy_class_seats : Nat := 50
def people_in_first_class : Nat := 3
def people_in_economy_class : Nat := economy_class_seats / 2
def people_in_business_and_first_class : Nat := people_in_economy_class
def people_in_business_class : Nat := people_in_business_and_first_class - people_in_first_class

/-
Prove that the number of unoccupied seats in business class is 8.
-/
theorem unoccupied_seats_in_business_class :
  business_class_seats - people_in_business_class = 8 :=
sorry

end unoccupied_seats_in_business_class_l75_75629


namespace sequence_unique_integers_l75_75551

theorem sequence_unique_integers (a : ℕ → ℤ) 
  (H_inf_pos : ∀ N : ℤ, ∃ n : ℕ, n > 0 ∧ a n > N) 
  (H_inf_neg : ∀ N : ℤ, ∃ n : ℕ, n > 0 ∧ a n < N)
  (H_diff_remainders : ∀ n : ℕ, n > 0 → ∀ i j : ℕ, (1 ≤ i ∧ i ≤ n) → (1 ≤ j ∧ j ≤ n) → i ≠ j → (a i % ↑n) ≠ (a j % ↑n)) :
  ∀ m : ℤ, ∃! n : ℕ, a n = m := sorry

end sequence_unique_integers_l75_75551


namespace shirts_and_pants_neither_plaid_nor_purple_l75_75397

variable (total_shirts total_pants plaid_shirts purple_pants : Nat)

def non_plaid_shirts (total_shirts plaid_shirts : Nat) : Nat := total_shirts - plaid_shirts
def non_purple_pants (total_pants purple_pants : Nat) : Nat := total_pants - purple_pants

theorem shirts_and_pants_neither_plaid_nor_purple :
  total_shirts = 5 → total_pants = 24 → plaid_shirts = 3 → purple_pants = 5 →
  non_plaid_shirts total_shirts plaid_shirts + non_purple_pants total_pants purple_pants = 21 :=
by
  intros
  -- Placeholder for proof to ensure the theorem builds correctly
  sorry

end shirts_and_pants_neither_plaid_nor_purple_l75_75397


namespace height_difference_l75_75641

theorem height_difference (B_height A_height : ℝ) (h : A_height = 0.6 * B_height) :
  (B_height - A_height) / A_height * 100 = 66.67 := 
sorry

end height_difference_l75_75641


namespace distance_between_ports_l75_75378

theorem distance_between_ports (x : ℝ) (speed_ship : ℝ) (speed_water : ℝ) (time_difference : ℝ) 
  (speed_downstream := speed_ship + speed_water) 
  (speed_upstream := speed_ship - speed_water) 
  (time_downstream := x / speed_downstream) 
  (time_upstream := x / speed_upstream) 
  (h : time_downstream + time_difference = time_upstream) 
  (h_ship : speed_ship = 26)
  (h_water : speed_water = 2)
  (h_time : time_difference = 3) : x = 504 :=
by
  -- The proof is omitted 
  sorry

end distance_between_ports_l75_75378


namespace comprehensive_score_correct_l75_75914

-- Conditions
def theoreticalWeight : ℝ := 0.20
def designWeight : ℝ := 0.50
def presentationWeight : ℝ := 0.30

def theoreticalScore : ℕ := 95
def designScore : ℕ := 88
def presentationScore : ℕ := 90

-- Calculate comprehensive score
def comprehensiveScore : ℝ :=
  theoreticalScore * theoreticalWeight +
  designScore * designWeight +
  presentationScore * presentationWeight

-- Lean statement to prove the comprehensive score using the conditions
theorem comprehensive_score_correct :
  comprehensiveScore = 90 := 
  sorry

end comprehensive_score_correct_l75_75914


namespace list_price_proof_l75_75124

theorem list_price_proof (x : ℝ) :
  (0.15 * (x - 15) = 0.25 * (x - 25)) → x = 40 :=
by
  sorry

end list_price_proof_l75_75124


namespace common_ratio_of_geometric_series_l75_75765

variables (a r S : ℝ)

theorem common_ratio_of_geometric_series
  (h1 : S = a / (1 - r))
  (h2 : r^4 * S = S / 81) :
  r = 1 / 3 :=
by
  sorry

end common_ratio_of_geometric_series_l75_75765


namespace simplify_expr_to_polynomial_l75_75424

namespace PolynomialProof

-- Define the given polynomial expressions
def expr1 (x : ℕ) := (3 * x^2 + 4 * x + 8) * (x - 2)
def expr2 (x : ℕ) := (x - 2) * (x^2 + 5 * x - 72)
def expr3 (x : ℕ) := (4 * x - 15) * (x - 2) * (x + 6)

-- Define the full polynomial expression
def full_expr (x : ℕ) := expr1 x - expr2 x + expr3 x

-- Our goal is to prove that full_expr == 6 * x^3 - 4 * x^2 - 26 * x + 20
theorem simplify_expr_to_polynomial (x : ℕ) : 
  full_expr x = 6 * x^3 - 4 * x^2 - 26 * x + 20 := by
  sorry

end PolynomialProof

end simplify_expr_to_polynomial_l75_75424


namespace evaluate_x_l75_75395

variable {R : Type*} [LinearOrderedField R]

theorem evaluate_x (m n k x : R) (hm : m ≠ 0) (hn : n ≠ 0) (h : m ≠ n) (h_eq : (x + m)^2 - (x + n)^2 = k * (m - n)^2) :
  x = ((k - 1) * (m + n) - 2 * k * n) / 2 :=
by
  sorry

end evaluate_x_l75_75395


namespace thermostat_range_l75_75555

theorem thermostat_range (T : ℝ) : 
  |T - 22| ≤ 6 ↔ 16 ≤ T ∧ T ≤ 28 := 
by sorry

end thermostat_range_l75_75555


namespace find_natural_n_l75_75955

theorem find_natural_n (n x y k : ℕ) (h_rel_prime : Nat.gcd x y = 1) (h_k_gt_one : k > 1) (h_eq : 3^n = x^k + y^k) :
  n = 2 := by
  sorry

end find_natural_n_l75_75955


namespace edward_money_l75_75389

theorem edward_money (X : ℝ) (H1 : X - 130 - 0.25 * (X - 130) = 270) : X = 490 :=
by
  sorry

end edward_money_l75_75389


namespace pyramid_volume_of_unit_cube_l75_75880

noncomputable def volume_of_pyramid : ℝ :=
  let s := (Real.sqrt 2) / 2
  let base_area := (Real.sqrt 3) / 8
  let height := 1
  (1 / 3) * base_area * height

theorem pyramid_volume_of_unit_cube :
  volume_of_pyramid = (Real.sqrt 3) / 24 := by
  sorry

end pyramid_volume_of_unit_cube_l75_75880


namespace ratio_of_sums_of_sides_and_sines_l75_75251

theorem ratio_of_sums_of_sides_and_sines (A B C : ℝ) (a b c : ℝ) (hA : A = 60) (ha : a = 3) 
  (h : a / Real.sin A = b / Real.sin B ∧ a / Real.sin A = c / Real.sin C) : 
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 3 := 
by 
  sorry

end ratio_of_sums_of_sides_and_sines_l75_75251


namespace proof_problem_l75_75094

noncomputable def f (x : ℝ) := 3 * Real.sin x + 2 * Real.cos x + 1

theorem proof_problem (a b c : ℝ) (h : ∀ x : ℝ, a * f x + b * f (x - c) = 1) :
  (b * Real.cos c / a) = -1 :=
sorry

end proof_problem_l75_75094


namespace prob_two_fours_l75_75265

-- Define the sample space for a fair die
def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- The probability of rolling a 4 on a fair die
def prob_rolling_four : ℚ := 1 / 6

-- Probability of two independent events both resulting in rolling a 4
def prob_both_rolling_four : ℚ := (prob_rolling_four) * (prob_rolling_four)

-- Prove that the probability of rolling two 4s in two independent die rolls is 1/36
theorem prob_two_fours : prob_both_rolling_four = 1 / 36 := by
  sorry

end prob_two_fours_l75_75265


namespace baseball_card_decrease_l75_75876

theorem baseball_card_decrease (x : ℝ) (h : (1 - x / 100) * (1 - x / 100) = 0.64) : x = 20 :=
by
  sorry

end baseball_card_decrease_l75_75876


namespace no_valid_pairs_l75_75958

theorem no_valid_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ¬ (a * b + 82 = 25 * Nat.lcm a b + 15 * Nat.gcd a b) :=
by {
  sorry
}

end no_valid_pairs_l75_75958


namespace cogs_produced_after_speed_increase_l75_75864

-- Define the initial conditions of the problem
def initial_cogs := 60
def initial_rate := 15
def increased_rate := 60
def average_output := 24

-- Variables to represent the number of cogs produced after the speed increase and the total time taken for each phase
variable (x : ℕ)

-- Assuming the equations representing the conditions
def initial_time := initial_cogs / initial_rate
def increased_time := x / increased_rate

def total_cogs := initial_cogs + x
def total_time := initial_time + increased_time

-- Define the overall average output equation
def average_eq := average_output * total_time = total_cogs

-- The proposition we want to prove
theorem cogs_produced_after_speed_increase : x = 60 :=
by
  -- Using the equation from the conditions
  have h1 : average_eq := sorry
  sorry

end cogs_produced_after_speed_increase_l75_75864


namespace nim_maximum_product_l75_75637

def nim_max_product (x y : ℕ) : ℕ :=
43 * 99 * x * y

theorem nim_maximum_product :
  ∃ x y : ℕ, (43 ≠ 0) ∧ (99 ≠ 0) ∧ (x ≠ 0) ∧ (y ≠ 0) ∧
  (43 + 99 + x + y = 0) ∧ (nim_max_product x y = 7704) :=
sorry

end nim_maximum_product_l75_75637


namespace line_through_circle_center_l75_75736

theorem line_through_circle_center {m : ℝ} :
  (∃ (x y : ℝ), x - 2*y + m = 0 ∧ x^2 + y^2 + 2*x - 4*y = 0) → m = 5 :=
by
  sorry

end line_through_circle_center_l75_75736


namespace mike_earnings_first_job_l75_75250

def total_earnings := 160
def hours_second_job := 12
def hourly_wage_second_job := 9
def earnings_second_job := hours_second_job * hourly_wage_second_job
def earnings_first_job := total_earnings - earnings_second_job

theorem mike_earnings_first_job : 
  earnings_first_job = 160 - (12 * 9) := by
  -- omitted proof
  sorry

end mike_earnings_first_job_l75_75250


namespace pairs_of_natural_numbers_l75_75092

theorem pairs_of_natural_numbers (a b : ℕ) (h₁ : b ∣ a + 1) (h₂ : a ∣ b + 1) :
    (a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3) ∨ (a = 2 ∧ b = 1) ∨ (a = 3 ∧ b = 2) :=
by {
  sorry
}

end pairs_of_natural_numbers_l75_75092


namespace third_angle_of_triangle_l75_75343

theorem third_angle_of_triangle (a b : ℝ) (h₁ : a = 25) (h₂ : b = 70) : 180 - a - b = 85 := 
by
  sorry

end third_angle_of_triangle_l75_75343


namespace value_of_a_l75_75156

noncomputable def f (a x : ℝ) : ℝ := a^x + Real.logb a (x^2 + 1)

theorem value_of_a (a : ℝ) (h : f a 1 + f a 2 = a^2 + a + 2) : a = Real.sqrt 10 :=
by
  sorry

end value_of_a_l75_75156


namespace ratio_of_speeds_l75_75022

theorem ratio_of_speeds (v_A v_B : ℝ) (t : ℝ) (hA : v_A = 120 / t) (hB : v_B = 60 / t) : v_A / v_B = 2 :=
by {
  sorry
}

end ratio_of_speeds_l75_75022


namespace cost_to_treat_dog_l75_75982

variable (D : ℕ)
variable (cost_cat : ℕ := 40)
variable (num_dogs : ℕ := 20)
variable (num_cats : ℕ := 60)
variable (total_paid : ℕ := 3600)

theorem cost_to_treat_dog : 20 * D + 60 * cost_cat = total_paid → D = 60 := by
  intros h
  -- Proof goes here
  sorry

end cost_to_treat_dog_l75_75982


namespace find_a_and_b_l75_75029

theorem find_a_and_b (a b : ℝ) :
  (∀ x, y = a + b / x) →
  (y = 3 → x = 2) →
  (y = -1 → x = -4) →
  a + b = 4 :=
by sorry

end find_a_and_b_l75_75029


namespace find_original_number_l75_75673

theorem find_original_number (k : ℤ) (h : 25 * k = N + 4) : ∃ N, N = 21 :=
by
  sorry

end find_original_number_l75_75673


namespace bottle_caps_given_l75_75857

variable (initial_caps : ℕ) (final_caps : ℕ) (caps_given_by_rebecca : ℕ)

theorem bottle_caps_given (h1: initial_caps = 7) (h2: final_caps = 9) : caps_given_by_rebecca = 2 :=
by
  -- The proof will be filled here
  sorry

end bottle_caps_given_l75_75857


namespace problem1_problem2_l75_75863

-- Problem (1)
theorem problem1 (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a^3 + b^3 >= a*b^2 + a^2*b := 
sorry

-- Problem (2)
theorem problem2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : 
  (1 + y) / x < 2 ∨ (1 + x) / y < 2 := 
sorry

end problem1_problem2_l75_75863


namespace frac_eq_three_l75_75006

theorem frac_eq_three (a b c : ℝ) 
  (h₁ : a / b = 4 / 3) (h₂ : (a + c) / (b - c) = 5 / 2) : 
  (3 * a + 2 * b) / (3 * a - 2 * b) = 3 :=
by
  sorry

end frac_eq_three_l75_75006


namespace conic_section_eccentricity_l75_75007

theorem conic_section_eccentricity (m : ℝ) (h : 2 * 8 = m^2) :
    (∃ e : ℝ, ((e = (Real.sqrt 2) / 2) ∨ (e = Real.sqrt 3))) :=
by
  sorry

end conic_section_eccentricity_l75_75007


namespace find_number_l75_75452

theorem find_number (x : ℝ) (h : (3/4 : ℝ) * x = 93.33333333333333) : x = 124.44444444444444 := 
by
  -- Proof to be filled in
  sorry

end find_number_l75_75452


namespace determine_r_l75_75747

theorem determine_r (S : ℕ → ℤ) (r : ℤ) (n : ℕ) (h1 : 2 ≤ n) (h2 : ∀ k, S k = 2^k + r) : 
  r = -1 :=
sorry

end determine_r_l75_75747


namespace five_b_value_l75_75306

theorem five_b_value (a b : ℚ) (h1 : 3 * a + 4 * b = 0) (h2 : a = b - 3) : 5 * b = 45 / 7 :=
by
  sorry

end five_b_value_l75_75306


namespace solution_set_of_inequality_l75_75966

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - 5 * x - 14 < 0} = {x : ℝ | -2 < x ∧ x < 7} :=
by
  sorry

end solution_set_of_inequality_l75_75966


namespace regular_nonagon_diagonals_correct_l75_75737

def regular_nonagon_diagonals : Nat :=
  let vertices := 9
  let total_line_segments := Nat.choose vertices 2
  let sides := vertices
  total_line_segments - sides
  
theorem regular_nonagon_diagonals_correct : regular_nonagon_diagonals = 27 := by
  sorry

end regular_nonagon_diagonals_correct_l75_75737


namespace event_day_price_l75_75799

theorem event_day_price (original_price : ℝ) (first_discount second_discount : ℝ)
  (h1 : original_price = 250) (h2 : first_discount = 0.4) (h3 : second_discount = 0.25) : 
  ∃ discounted_price : ℝ, 
  discounted_price = (original_price * (1 - first_discount)) * (1 - second_discount) → 
  discounted_price = 112.5 :=
by
  use (250 * (1 - 0.4) * (1 - 0.25))
  sorry

end event_day_price_l75_75799


namespace average_homework_time_decrease_l75_75171

theorem average_homework_time_decrease
  (initial_time final_time : ℝ)
  (rate_of_decrease : ℝ)
  (h1 : initial_time = 100)
  (h2 : final_time = 70) :
  initial_time * (1 - rate_of_decrease)^2 = final_time :=
by
  rw [h1, h2]
  sorry

end average_homework_time_decrease_l75_75171


namespace point_on_x_axis_l75_75647

theorem point_on_x_axis (a : ℝ) (h : (1, a + 1).snd = 0) : a = -1 :=
by
  sorry

end point_on_x_axis_l75_75647


namespace find_xyz_sum_cube_l75_75643

variable (x y z c d : ℝ) 

theorem find_xyz_sum_cube (h1 : x * y * z = c) (h2 : 1 / x^3 + 1 / y^3 + 1 / z^3 = d) :
  (x + y + z)^3 = d * c^3 + 3 * c - 3 * c * d := 
by
  sorry

end find_xyz_sum_cube_l75_75643


namespace find_width_of_room_eq_l75_75224

noncomputable def total_cost : ℝ := 20625
noncomputable def rate_per_sqm : ℝ := 1000
noncomputable def length_of_room : ℝ := 5.5
noncomputable def area_paved : ℝ := total_cost / rate_per_sqm
noncomputable def width_of_room : ℝ := area_paved / length_of_room

theorem find_width_of_room_eq :
  width_of_room = 3.75 :=
sorry

end find_width_of_room_eq_l75_75224


namespace minimum_value_2x_plus_y_l75_75123

theorem minimum_value_2x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + y + 6 = x * y) : 
  2 * x + y ≥ 12 := 
sorry

end minimum_value_2x_plus_y_l75_75123


namespace average_weight_of_removed_onions_l75_75540

theorem average_weight_of_removed_onions (total_weight_40_onions : ℝ := 7680)
    (average_weight_35_onions : ℝ := 190)
    (number_of_onions_removed : ℕ := 5)
    (total_onions_initial : ℕ := 40)
    (total_number_of_remaining_onions : ℕ := 35) :
    (total_weight_40_onions - total_number_of_remaining_onions * average_weight_35_onions) / number_of_onions_removed = 206 :=
by
    sorry

end average_weight_of_removed_onions_l75_75540


namespace decagon_perimeter_l75_75046

-- Define the number of sides in a decagon
def num_sides : ℕ := 10

-- Define the length of each side in the decagon
def side_length : ℕ := 3

-- Define the perimeter of a decagon given the number of sides and the side length
def perimeter (n : ℕ) (s : ℕ) : ℕ := n * s

-- State the theorem we want to prove: the perimeter of our given regular decagon
theorem decagon_perimeter : perimeter num_sides side_length = 30 := 
by sorry

end decagon_perimeter_l75_75046


namespace problem_l75_75144

noncomputable def f : ℝ → ℝ := sorry 

theorem problem
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_func : ∀ x : ℝ, f (2 + x) = -f (2 - x))
  (h_value : f (-3) = -2) :
  f 2007 = 2 :=
sorry

end problem_l75_75144


namespace cloth_sold_l75_75849

theorem cloth_sold (C S M : ℚ) (P : ℚ) (hP : P = 1 / 3) (hG : 10 * S = (1 / 3) * (M * C)) (hS : S = (4 / 3) * C) : M = 40 := by
  sorry

end cloth_sold_l75_75849


namespace probability_one_in_first_20_rows_l75_75437

theorem probability_one_in_first_20_rows :
  let total_elements := 210
  let number_of_ones := 39
  (number_of_ones / total_elements : ℚ) = 13 / 70 :=
by
  sorry

end probability_one_in_first_20_rows_l75_75437


namespace find_x_for_condition_l75_75299

def f (x : ℝ) : ℝ := 3 * x - 5

theorem find_x_for_condition :
  (2 * f 1 - 16 = f (1 - 6)) :=
by
  sorry

end find_x_for_condition_l75_75299


namespace zero_in_M_l75_75501

def M : Set Int := {-1, 0, 1}

theorem zero_in_M : 0 ∈ M :=
  by
  -- Proof is omitted
  sorry

end zero_in_M_l75_75501


namespace total_problems_is_correct_l75_75585

/-- Definition of the number of pages of math homework. -/
def math_pages : ℕ := 2

/-- Definition of the number of pages of reading homework. -/
def reading_pages : ℕ := 4

/-- Definition that each page of homework contains 5 problems. -/
def problems_per_page : ℕ := 5

/-- The proof statement: given the number of pages of math and reading homework,
    and the number of problems per page, prove that the total number of problems is 30. -/
theorem total_problems_is_correct : (math_pages + reading_pages) * problems_per_page = 30 := by
  sorry

end total_problems_is_correct_l75_75585


namespace find_n_l75_75708

theorem find_n
    (h : Real.arctan (1 / 2) + Real.arctan (1 / 3) + Real.arctan (1 / 7) + Real.arctan (1 / n) = Real.pi / 2) :
    n = 46 :=
sorry

end find_n_l75_75708


namespace maximum_distance_between_balls_l75_75952

theorem maximum_distance_between_balls 
  (a b c : ℝ) 
  (aluminum_ball_heavier : true) -- Implicitly understood property rather than used in calculation directly
  (wood_ball_lighter : true) -- Implicitly understood property rather than used in calculation directly
  : ∃ d : ℝ, d = Real.sqrt (a^2 + b^2 + c^2) → d = Real.sqrt (3^2 + 4^2 + 2^2) := 
by
  use Real.sqrt (3^2 + 4^2 + 2^2)
  sorry

end maximum_distance_between_balls_l75_75952


namespace chinese_mathematical_system_l75_75070

noncomputable def problem_statement : Prop :=
  ∃ (x : ℕ) (y : ℕ),
    7 * x + 7 = y ∧ 
    9 * (x - 1) = y

theorem chinese_mathematical_system :
  problem_statement := by
  sorry

end chinese_mathematical_system_l75_75070


namespace sum_of_coordinates_of_other_endpoint_l75_75314

theorem sum_of_coordinates_of_other_endpoint
  (x y : ℝ)
  (h1 : (1 + x) / 2 = 5)
  (h2 : (2 + y) / 2 = 6) :
  x + y = 19 :=
by
  sorry

end sum_of_coordinates_of_other_endpoint_l75_75314


namespace value_of_f1_l75_75691

variable (f : ℝ → ℝ)
open Function

theorem value_of_f1
  (h : ∀ x y : ℝ, f (f (x - y)) = f x * f y - f x + f y - 2 * x * y + 2 * x - 2 * y) :
  f 1 = -1 :=
sorry

end value_of_f1_l75_75691


namespace coprime_unique_residues_non_coprime_same_residue_l75_75711

-- Part (a)

theorem coprime_unique_residues (m k : ℕ) (h : m.gcd k = 1) : 
  ∃ (a : Fin m → ℕ) (b : Fin k → ℕ), 
    ∀ (i : Fin m) (j : Fin k), 
      ∀ (i' : Fin m) (j' : Fin k), 
        (i, j) ≠ (i', j') → (a i * b j) % (m * k) ≠ (a i' * b j') % (m * k) := 
sorry

-- Part (b)

theorem non_coprime_same_residue (m k : ℕ) (h : m.gcd k > 1) : 
  ∀ (a : Fin m → ℕ) (b : Fin k → ℕ), 
    ∃ (i : Fin m) (j : Fin k) (i' : Fin m) (j' : Fin k), 
      (i, j) ≠ (i', j') ∧ (a i * b j) % (m * k) = (a i' * b j') % (m * k) := 
sorry

end coprime_unique_residues_non_coprime_same_residue_l75_75711


namespace water_flow_speed_l75_75557

/-- A person rows a boat for 15 li. If he rows at his usual speed,
the time taken to row downstream is 5 hours less than rowing upstream.
If he rows at twice his usual speed, the time taken to row downstream
is only 1 hour less than rowing upstream. 
Prove that the speed of the water flow is 2 li/hour.
-/
theorem water_flow_speed (y x : ℝ)
  (h1 : 15 / (y - x) - 15 / (y + x) = 5)
  (h2 : 15 / (2 * y - x) - 15 / (2 * y + x) = 1) :
  x = 2 := 
sorry

end water_flow_speed_l75_75557


namespace recurring_decimal_sum_l75_75138

noncomputable def x : ℚ := 1 / 3

noncomputable def y : ℚ := 14 / 999

noncomputable def z : ℚ := 5 / 9999

theorem recurring_decimal_sum :
  x + y + z = 3478 / 9999 := by
  sorry

end recurring_decimal_sum_l75_75138


namespace customers_per_table_l75_75030

theorem customers_per_table (total_tables : ℝ) (left_tables : ℝ) (total_customers : ℕ)
  (h1 : total_tables = 44.0)
  (h2 : left_tables = 12.0)
  (h3 : total_customers = 256) :
  total_customers / (total_tables - left_tables) = 8 :=
by {
  sorry
}

end customers_per_table_l75_75030


namespace max_b_n_occurs_at_n_l75_75057

def a_n (n : ℕ) (a1 : ℚ) (d : ℚ) : ℚ :=
  a1 + (n-1) * d

def S_n (n : ℕ) (a1 : ℚ) (d : ℚ) : ℚ :=
  n * a1 + (n * (n-1) / 2) * d

def b_n (n : ℕ) (an : ℚ) : ℚ :=
  (1 + an) / an

theorem max_b_n_occurs_at_n :
  ∀ (n : ℕ) (a1 d : ℚ),
  (a1 = -5/2) →
  (S_n 4 a1 d = 2 * S_n 2 a1 d + 4) →
  n = 4 := sorry

end max_b_n_occurs_at_n_l75_75057


namespace cot_30_plus_cot_75_eq_2_l75_75110

noncomputable def cot (θ : ℝ) : ℝ := 1 / Real.tan θ

theorem cot_30_plus_cot_75_eq_2 : cot 30 + cot 75 = 2 := by sorry

end cot_30_plus_cot_75_eq_2_l75_75110


namespace days_for_Q_wages_l75_75238

variables (P Q S : ℝ) (D : ℝ)

theorem days_for_Q_wages (h1 : S = 24 * P) (h2 : S = 15 * (P + Q)) : S = D * Q → D = 40 :=
by
  sorry

end days_for_Q_wages_l75_75238


namespace intersection_A_B_l75_75207

open Set

def SetA : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def SetB : Set ℤ := {x | 0 ≤ x ∧ x ≤ 4}

theorem intersection_A_B :
  (SetA ∩ SetB) = ( {0, 2, 4} : Set ℤ ) :=
by
  sorry

end intersection_A_B_l75_75207


namespace sum_repeating_decimals_as_fraction_l75_75414

-- Definitions for repeating decimals
def rep2 : ℝ := 0.2222
def rep02 : ℝ := 0.0202
def rep0002 : ℝ := 0.00020002

-- Prove the sum of the repeating decimals is equal to the given fraction
theorem sum_repeating_decimals_as_fraction :
  rep2 + rep02 + rep0002 = (2224 / 9999 : ℝ) :=
sorry

end sum_repeating_decimals_as_fraction_l75_75414


namespace radius_of_circle_l75_75115

/-- Given the equation of a circle x^2 + y^2 - 8 = 2x + 4y,
    we need to prove that the radius of the circle is sqrt 13. -/
theorem radius_of_circle : 
    ∃ r : ℝ, (∀ x y : ℝ, x^2 + y^2 - 8 = 2*x + 4*y → r = Real.sqrt 13) :=
by
    sorry

end radius_of_circle_l75_75115


namespace no_real_x_solution_l75_75999

open Real

-- Define the conditions.
def log_defined (x : ℝ) : Prop :=
  0 < x + 5 ∧ 0 < x - 3 ∧ 0 < x^2 - 7*x - 18

-- Define the equation to prove.
def log_eqn (x : ℝ) : Prop :=
  log (x + 5) + log (x - 3) = log (x^2 - 7*x - 18)

-- The mathematicall equivalent proof problem.
theorem no_real_x_solution : ¬∃ x : ℝ, log_defined x ∧ log_eqn x :=
by
  sorry

end no_real_x_solution_l75_75999


namespace arc_lengths_l75_75410

-- Definitions for the given conditions
def circumference : ℝ := 80  -- Circumference of the circle

-- Angles in degrees
def angle_AOM : ℝ := 45
def angle_MOB : ℝ := 90

-- Radius of the circle using the formula C = 2 * π * r
noncomputable def radius : ℝ := circumference / (2 * Real.pi)

-- Calculate the arc lengths using the angles
noncomputable def arc_length_AM : ℝ := (angle_AOM / 360) * circumference
noncomputable def arc_length_MB : ℝ := (angle_MOB / 360) * circumference

-- The theorem stating the required lengths
theorem arc_lengths (h : circumference = 80 ∧ angle_AOM = 45 ∧ angle_MOB = 90) :
  arc_length_AM = 10 ∧ arc_length_MB = 20 :=
by
  sorry

end arc_lengths_l75_75410


namespace number_of_female_students_l75_75460

variable (n m : ℕ)

theorem number_of_female_students (hn : n ≥ 0) (hm : m ≥ 0) (hmn : m ≤ n) : n - m = n - m :=
by
  sorry

end number_of_female_students_l75_75460


namespace closest_point_on_line_l75_75632

theorem closest_point_on_line (x y : ℝ) (h : y = (x - 3) / 3) : 
  (∃ p : ℝ × ℝ, p = (4, -2) ∧ ∀ q : ℝ × ℝ, (q.1, q.2) = (x, y) ∧ q ≠ p → dist p q ≥ dist p (33/10, 1/10)) :=
sorry

end closest_point_on_line_l75_75632


namespace product_quantities_l75_75592

theorem product_quantities (a b x y : ℝ) 
  (h1 : a * x + b * y = 1500)
  (h2 : (a + 1.5) * (x - 10) + (b + 1) * y = 1529)
  (h3 : (a + 1) * (x - 5) + (b + 1) * y = 1563.5)
  (h4 : 205 < 2 * x + y ∧ 2 * x + y < 210) :
  (x + 2 * y = 186) ∧ (73 ≤ x ∧ x ≤ 75) :=
by
  sorry

end product_quantities_l75_75592


namespace difference_apples_peaches_pears_l75_75482

-- Definitions based on the problem conditions
def apples : ℕ := 60
def peaches : ℕ := 3 * apples
def pears : ℕ := apples / 2

-- Statement of the proof problem
theorem difference_apples_peaches_pears : (apples + peaches) - pears = 210 := by
  sorry

end difference_apples_peaches_pears_l75_75482


namespace value_of_abc_l75_75580

noncomputable def f (x a b c : ℝ) := |(1 - x^2) * (x^2 + a * x + b)| - c

theorem value_of_abc :
  (∀ x : ℝ, f (x + 4) 8 15 9 = f (-x) 8 15 9) ∧
  (∃ x : ℝ, f x 8 15 9 = 0) ∧
  (∃ x : ℝ, f (-(x-4)) 8 15 9 = 0) ∧
  (∀ c : ℝ, c ≠ 0) →
  8 + 15 + 9 = 32 :=
by sorry

end value_of_abc_l75_75580


namespace sqrt_31_between_5_and_6_l75_75010

theorem sqrt_31_between_5_and_6
  (h1 : Real.sqrt 25 = 5)
  (h2 : Real.sqrt 36 = 6)
  (h3 : 25 < 31)
  (h4 : 31 < 36) :
  5 < Real.sqrt 31 ∧ Real.sqrt 31 < 6 :=
sorry

end sqrt_31_between_5_and_6_l75_75010


namespace total_turnips_grown_l75_75000

theorem total_turnips_grown 
  (melanie_turnips : ℕ) 
  (benny_turnips : ℕ) 
  (jack_turnips : ℕ) 
  (lynn_turnips : ℕ) : 
  melanie_turnips = 1395 ∧
  benny_turnips = 11380 ∧
  jack_turnips = 15825 ∧
  lynn_turnips = 23500 → 
  melanie_turnips + benny_turnips + jack_turnips + lynn_turnips = 52100 :=
by
  intros h
  rcases h with ⟨hm, hb, hj, hl⟩
  sorry

end total_turnips_grown_l75_75000


namespace average_customers_per_day_l75_75412

-- Define the number of customers each day:
def customers_per_day : List ℕ := [10, 12, 15, 13, 18, 16, 11]

-- Define the total number of days in a week
def days_in_week : ℕ := 7

-- Define the theorem stating the average number of daily customers
theorem average_customers_per_day :
  (customers_per_day.sum : ℚ) / days_in_week = 13.57 :=
by
  sorry

end average_customers_per_day_l75_75412


namespace which_polygon_covers_ground_l75_75262

def is_tessellatable (n : ℕ) : Prop :=
  let interior_angle := (n - 2) * 180 / n
  360 % interior_angle = 0

theorem which_polygon_covers_ground :
  is_tessellatable 6 ∧ ¬is_tessellatable 5 ∧ ¬is_tessellatable 8 ∧ ¬is_tessellatable 12 :=
by
  sorry

end which_polygon_covers_ground_l75_75262


namespace gumballs_initial_count_l75_75940

theorem gumballs_initial_count (x : ℝ) (h : (0.75 ^ 3) * x = 27) : x = 64 :=
by
  sorry

end gumballs_initial_count_l75_75940


namespace jacket_initial_reduction_l75_75679

theorem jacket_initial_reduction (x : ℝ) :
  (1 - x / 100) * 1.53846 = 1 → x = 35 :=
by
  sorry

end jacket_initial_reduction_l75_75679


namespace cylinder_problem_l75_75467

theorem cylinder_problem (r h : ℝ) (h1 : π * r^2 * h = 2) (h2 : 2 * π * r * h + 2 * π * r^2 = 12) :
  1 / r + 1 / h = 3 :=
sorry

end cylinder_problem_l75_75467


namespace find_hyperbola_m_l75_75394

theorem find_hyperbola_m (m : ℝ) (h : m > 0) : 
  (∀ x y : ℝ, (x^2 / m - y^2 / 3 = 1 → y = 1 / 2 * x)) → m = 12 :=
by
  intros
  sorry

end find_hyperbola_m_l75_75394


namespace exists_n_in_range_multiple_of_11_l75_75449

def is_multiple_of_11 (n : ℕ) : Prop :=
  (3 * n^5 + 4 * n^4 + 5 * n^3 + 7 * n^2 + 6 * n + 2) % 11 = 0

theorem exists_n_in_range_multiple_of_11 : ∃ n : ℕ, (2 ≤ n ∧ n ≤ 101) ∧ is_multiple_of_11 n :=
sorry

end exists_n_in_range_multiple_of_11_l75_75449


namespace solve_system_of_equations_l75_75388

theorem solve_system_of_equations (a b : ℝ) (h1 : a^2 ≠ 1) (h2 : b^2 ≠ 1) (h3 : a ≠ b) : 
  (∃ x y : ℝ, 
    (x - y) / (1 - x * y) = 2 * a / (1 + a^2) ∧ (x + y) / (1 + x * y) = 2 * b / (1 + b^2) ∧
    ((x = (a * b + 1) / (a + b) ∧ y = (a * b - 1) / (a - b)) ∨ 
     (x = (a + b) / (a * b + 1) ∧ y = (a - b) / (a * b - 1)))) :=
by
  sorry

end solve_system_of_equations_l75_75388


namespace rectangle_width_length_ratio_l75_75545

theorem rectangle_width_length_ratio (w l : ℕ) 
  (h1 : l = 12) 
  (h2 : 2 * w + 2 * l = 36) : 
  w / l = 1 / 2 := 
by 
  sorry

end rectangle_width_length_ratio_l75_75545


namespace man_l75_75423

-- Defining the conditions as variables in Lean
variables (S : ℕ) (M : ℕ)
-- Given conditions
def son_present_age := S = 25
def man_present_age := M = S + 27

-- Goal: the ratio of the man's age to the son's age in two years is 2:1
theorem man's_age_ratio_in_two_years (h1 : son_present_age S) (h2 : man_present_age S M) :
  (M + 2) / (S + 2) = 2 := sorry

end man_l75_75423


namespace difference_between_relations_l75_75202

-- Definitions based on conditions
def functional_relationship 
  (f : α → β) (x : α) (y : β) : Prop :=
  f x = y

def correlation_relationship (X Y : Type) : Prop :=
  ∃ (X_rand : X → ℝ) (Y_rand : Y → ℝ), 
    ∀ (x : X), ∃ (y : Y), X_rand x ≠ Y_rand y

-- Theorem stating the problem
theorem difference_between_relations :
  (∀ (f : α → β) (x : α) (y : β), functional_relationship f x y) ∧ 
  (∀ (X Y : Type), correlation_relationship X Y) :=
sorry

end difference_between_relations_l75_75202


namespace smallest_b_to_the_a_l75_75972

theorem smallest_b_to_the_a (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = 2^2023) : b^a = 1 :=
by
  -- Proof steps go here
  sorry

end smallest_b_to_the_a_l75_75972


namespace tan_theta_minus_pi_over_4_l75_75056

theorem tan_theta_minus_pi_over_4 (θ : ℝ) (h : Real.cos θ - 3 * Real.sin θ = 0) :
  Real.tan (θ - Real.pi / 4) = -1 / 2 :=
sorry

end tan_theta_minus_pi_over_4_l75_75056


namespace sum_inf_evaluation_eq_9_by_80_l75_75442

noncomputable def infinite_sum_evaluation : ℝ := ∑' n, (2 * n) / (n^4 + 16)

theorem sum_inf_evaluation_eq_9_by_80 :
  infinite_sum_evaluation = 9 / 80 :=
by
  sorry

end sum_inf_evaluation_eq_9_by_80_l75_75442


namespace rectangle_length_width_difference_l75_75008

theorem rectangle_length_width_difference
  (x y : ℝ)
  (h1 : x + y = 40)
  (h2 : x^2 + y^2 = 800) :
  x - y = 0 :=
sorry

end rectangle_length_width_difference_l75_75008


namespace max_airlines_l75_75083

-- Definitions for the conditions
-- There are 200 cities
def num_cities : ℕ := 200

-- Calculate the total number of city pairs
def num_city_pairs (n : ℕ) : ℕ := (n * (n - 1)) / 2

def total_city_pairs : ℕ := num_city_pairs num_cities

-- Minimum spanning tree concept
def min_flights_per_airline (n : ℕ) : ℕ := n - 1

def total_flights_required : ℕ := num_cities * min_flights_per_airline num_cities

-- Claim: Maximum number of airlines
theorem max_airlines (n : ℕ) (h : n = 200) : ∃ m : ℕ, m = (total_city_pairs / (min_flights_per_airline n)) ∧ m = 100 :=
by sorry

end max_airlines_l75_75083


namespace complete_square_proof_l75_75494

def complete_square (x : ℝ) : Prop :=
  x^2 - 2 * x - 8 = 0 -> (x - 1)^2 = 9

theorem complete_square_proof (x : ℝ) :
  complete_square x :=
sorry

end complete_square_proof_l75_75494


namespace proof_volume_l75_75678

noncomputable def volume_set (a b c h r : ℝ) : ℝ := 
  let v_box := a * b * c
  let v_extensions := 2 * (a * b * h) + 2 * (a * c * h) + 2 * (b * c * h)
  let v_cylinder := Real.pi * r^2 * h
  let v_spheres := 8 * (1/6) * (Real.pi * r^3)
  v_box + v_extensions + v_cylinder + v_spheres

theorem proof_volume : 
  let a := 2; let b := 3; let c := 6
  let r := 2; let h := 3
  volume_set a b c h r = (540 + 48 * Real.pi) / 3 ∧ (540 + 48 + 3) = 591 :=
by 
  sorry

end proof_volume_l75_75678


namespace find_y_l75_75537

noncomputable def x : Real := 1.6666666666666667
def y : Real := 5

theorem find_y (h : x ≠ 0) (h1 : (x * y) / 3 = x^2) : y = 5 := 
by sorry

end find_y_l75_75537


namespace circle_area_percentage_decrease_l75_75185

theorem circle_area_percentage_decrease (r : ℝ) (A : ℝ := Real.pi * r^2) 
  (r' : ℝ := 0.5 * r) (A' : ℝ := Real.pi * (r')^2) :
  (A - A') / A * 100 = 75 := 
by
  sorry

end circle_area_percentage_decrease_l75_75185


namespace sum_of_a_b_l75_75325

theorem sum_of_a_b (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 + b^2 = 25) : a + b = 7 ∨ a + b = -7 := 
by 
  sorry

end sum_of_a_b_l75_75325


namespace average_length_of_strings_l75_75027

theorem average_length_of_strings {l1 l2 l3 : ℝ} (h1 : l1 = 2) (h2 : l2 = 6) (h3 : l3 = 9) : 
  (l1 + l2 + l3) / 3 = 17 / 3 :=
by
  sorry

end average_length_of_strings_l75_75027


namespace slices_leftover_is_9_l75_75851

-- Conditions and definitions
def total_pizzas : ℕ := 2
def slices_per_pizza : ℕ := 12
def bob_ate : ℕ := slices_per_pizza / 2
def tom_ate : ℕ := slices_per_pizza / 3
def sally_ate : ℕ := slices_per_pizza / 6
def jerry_ate : ℕ := slices_per_pizza / 4

-- Calculate total slices eaten and left over
def total_slices_eaten : ℕ := bob_ate + tom_ate + sally_ate + jerry_ate
def total_slices_available : ℕ := total_pizzas * slices_per_pizza
def slices_leftover : ℕ := total_slices_available - total_slices_eaten

-- Theorem to prove the number of slices left over
theorem slices_leftover_is_9 : slices_leftover = 9 := by
  -- Proof: omitted, add relevant steps here
  sorry

end slices_leftover_is_9_l75_75851


namespace min_number_of_trials_sum_15_min_number_of_trials_sum_at_least_15_l75_75499

noncomputable def min_trials_sum_of_15 : ℕ :=
  15

noncomputable def min_trials_sum_at_least_15 : ℕ :=
  8

theorem min_number_of_trials_sum_15 (x : ℕ) :
  (∀ (x : ℕ), (103/108 : ℝ)^x < (1/2 : ℝ) → x >= min_trials_sum_of_15) := sorry

theorem min_number_of_trials_sum_at_least_15 (x : ℕ) :
  (∀ (x : ℕ), (49/54 : ℝ)^x < (1/2 : ℝ) → x >= min_trials_sum_at_least_15) := sorry

end min_number_of_trials_sum_15_min_number_of_trials_sum_at_least_15_l75_75499


namespace solve_equation_l75_75503

theorem solve_equation (x : ℝ) (h1 : x + 2 ≠ 0) (h2 : 3 - x ≠ 0) :
  (3 * x - 5) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ↔ x = -15 / 2 :=
by
  sorry

end solve_equation_l75_75503


namespace chameleon_color_change_l75_75835

variable (x : ℕ)

-- Initial and final conditions definitions.
def total_chameleons : ℕ := 140
def initial_blue_chameleons : ℕ := 5 * x 
def initial_red_chameleons : ℕ := 140 - 5 * x 
def final_blue_chameleons : ℕ := x
def final_red_chameleons : ℕ := 3 * (140 - 5 * x )

-- Proof statement
theorem chameleon_color_change :
  (140 - 5 * x) * 3 + x = 140 → 4 * x = 80 :=
by sorry

end chameleon_color_change_l75_75835


namespace total_pamphlets_correct_l75_75522

-- Define the individual printing rates and hours
def Mike_pre_break_rate := 600
def Mike_pre_break_hours := 9
def Mike_post_break_rate := Mike_pre_break_rate / 3
def Mike_post_break_hours := 2

def Leo_pre_break_rate := 2 * Mike_pre_break_rate
def Leo_pre_break_hours := Mike_pre_break_hours / 3
def Leo_post_first_break_rate := Leo_pre_break_rate / 2
def Leo_post_second_break_rate := Leo_post_first_break_rate / 2

def Sally_pre_break_rate := 3 * Mike_pre_break_rate
def Sally_pre_break_hours := Mike_post_break_hours / 2
def Sally_post_break_rate := Leo_post_first_break_rate
def Sally_post_break_hours := 1

-- Calculate the total number of pamphlets printed by each person
def Mike_pamphlets := 
  (Mike_pre_break_rate * Mike_pre_break_hours) + (Mike_post_break_rate * Mike_post_break_hours)

def Leo_pamphlets := 
  (Leo_pre_break_rate * 1) + (Leo_post_first_break_rate * 1) + (Leo_post_second_break_rate * 1)

def Sally_pamphlets := 
  (Sally_pre_break_rate * Sally_pre_break_hours) + (Sally_post_break_rate * Sally_post_break_hours)

-- Calculate the total number of pamphlets printed by all three
def total_pamphlets := Mike_pamphlets + Leo_pamphlets + Sally_pamphlets

theorem total_pamphlets_correct : total_pamphlets = 10700 := by
  sorry

end total_pamphlets_correct_l75_75522


namespace product_of_000412_and_9243817_is_closest_to_3600_l75_75287

def product_closest_to (x y value: ℝ) : Prop := (abs (x * y - value) < min (abs (x * y - 350)) (min (abs (x * y - 370)) (min (abs (x * y - 3700)) (abs (x * y - 4000)))))

theorem product_of_000412_and_9243817_is_closest_to_3600 :
  product_closest_to 0.000412 9243817 3600 :=
by
  sorry

end product_of_000412_and_9243817_is_closest_to_3600_l75_75287


namespace scientific_notation_of_29_47_thousand_l75_75209

theorem scientific_notation_of_29_47_thousand :
  (29.47 * 1000 = 2.947 * 10^4) :=
sorry

end scientific_notation_of_29_47_thousand_l75_75209


namespace net_sales_revenue_l75_75441

-- Definition of the conditions
def regression (x : ℝ) : ℝ := 8.5 * x + 17.5

-- Statement of the theorem
theorem net_sales_revenue (x : ℝ) (h : x = 10) : (regression x - x) = 92.5 :=
by {
  -- No proof required as per instruction; use sorry.
  sorry
}

end net_sales_revenue_l75_75441


namespace maciek_total_cost_l75_75087

-- Define the cost of pretzels without discount
def pretzel_price : ℝ := 4.0

-- Define the discounted price of pretzels when buying 3 or more packs
def pretzel_discount_price : ℝ := 3.5

-- Define the cost of chips without discount
def chips_price : ℝ := 7.0

-- Define the discounted price of chips when buying 2 or more packs
def chips_discount_price : ℝ := 6.0

-- Define the number of pretzels Maciek buys
def pretzels_bought : ℕ := 3

-- Define the number of chips Maciek buys
def chips_bought : ℕ := 4

-- Calculate the total cost of pretzels
def pretzel_cost : ℝ :=
  if pretzels_bought >= 3 then pretzels_bought * pretzel_discount_price else pretzels_bought * pretzel_price

-- Calculate the total cost of chips
def chips_cost : ℝ :=
  if chips_bought >= 2 then chips_bought * chips_discount_price else chips_bought * chips_price

-- Calculate the total amount Maciek needs to pay
def total_cost : ℝ :=
  pretzel_cost + chips_cost

theorem maciek_total_cost :
  total_cost = 34.5 :=
by 
  sorry

end maciek_total_cost_l75_75087


namespace find_p_l75_75661

theorem find_p (p : ℕ) : 18^3 = (16^2 / 4) * 2^(8 * p) → p = 0 := 
by 
  sorry

end find_p_l75_75661


namespace johnny_guitar_practice_l75_75420

theorem johnny_guitar_practice :
  ∃ x : ℕ, (∃ d : ℕ, d = 20 ∧ ∀ n : ℕ, (n = x - d ∧ n = x / 2)) ∧ (x + 80 = 3 * x) :=
by
  sorry

end johnny_guitar_practice_l75_75420


namespace ratio_y_to_x_l75_75594

-- Definitions based on conditions
variable (c : ℝ) -- Cost price
def x : ℝ := 0.8 * c -- Selling price for a loss of 20%
def y : ℝ := 1.25 * c -- Selling price for a gain of 25%

-- Statement to prove the ratio of y to x
theorem ratio_y_to_x : y / x = 25 / 16 := by
  -- skip the proof
  sorry

end ratio_y_to_x_l75_75594


namespace no_valid_conference_division_l75_75624

theorem no_valid_conference_division (num_teams : ℕ) (matches_per_team : ℕ) :
  num_teams = 30 → matches_per_team = 82 → 
  ¬ ∃ (k : ℕ) (x y z : ℕ), k + (num_teams - k) = num_teams ∧
                          x + y + z = (num_teams * matches_per_team) / 2 ∧
                          z = ((x + y + z) / 2) := 
by
  sorry

end no_valid_conference_division_l75_75624


namespace travel_days_l75_75471

variable (a b d : ℕ)

theorem travel_days (h1 : a + d = 11) (h2 : b + d = 21) (h3 : a + b = 12) : a + b + d = 22 :=
by sorry

end travel_days_l75_75471


namespace N_subset_M_l75_75486

-- Definitions of sets M and N
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | x * x - x < 0 }

-- Proof statement: N is a subset of M
theorem N_subset_M : N ⊆ M :=
sorry

end N_subset_M_l75_75486


namespace compare_fractions_l75_75380

theorem compare_fractions (x y : ℝ) (n : ℕ) (h1 : 0 < x) (h2 : x < 1) (h3 : 0 < y) (h4 : y < 1) (h5 : 0 < n) :
  (x^n / (1 - x^2) + y^n / (1 - y^2)) ≥ ((x^n + y^n) / (1 - x * y)) :=
by sorry

end compare_fractions_l75_75380


namespace circumscribed_circle_radius_l75_75422

-- Definitions of side lengths
def a : ℕ := 5
def b : ℕ := 12

-- Defining the hypotenuse based on the Pythagorean theorem
def hypotenuse (a b : ℕ) : ℕ := Nat.sqrt (a * a + b * b)

-- Radius of the circumscribed circle of a right triangle
def radius (hypotenuse : ℕ) : ℕ := hypotenuse / 2

-- Theorem: The radius of the circumscribed circle of the right triangle is 13 / 2 = 6.5
theorem circumscribed_circle_radius : 
  radius (hypotenuse a b) = 13 / 2 :=
by
  sorry

end circumscribed_circle_radius_l75_75422


namespace find_f_neg3_l75_75652

noncomputable def f : ℝ → ℝ
| x => if x > 0 then x^2 - 2 * x else -(x^2 - 2 * -x)

theorem find_f_neg3 (h_odd : ∀ x : ℝ, f (-x) = -f x) (h_pos : ∀ x : ℝ, 0 < x → f x = x^2 - 2 * x) : f (-3) = -3 :=
by
  sorry

end find_f_neg3_l75_75652


namespace kaleb_toys_can_buy_l75_75621

theorem kaleb_toys_can_buy (saved_money : ℕ) (allowance_received : ℕ) (allowance_increase_percent : ℕ) (toy_cost : ℕ) (half_total_spend : ℕ) :
  saved_money = 21 →
  allowance_received = 15 →
  allowance_increase_percent = 20 →
  toy_cost = 6 →
  half_total_spend = (saved_money + allowance_received) / 2 →
  (half_total_spend / toy_cost) = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end kaleb_toys_can_buy_l75_75621


namespace new_bookstore_acquisition_l75_75524

theorem new_bookstore_acquisition (x : ℝ) 
  (h1 : (1 / 2) * x + (1 / 4) * x + 50 = x - 200) : x = 1000 :=
by {
  sorry
}

end new_bookstore_acquisition_l75_75524


namespace find_first_offset_l75_75979

theorem find_first_offset (x : ℝ) : 
  let area := 180
  let diagonal := 24
  let offset2 := 6
  (area = (diagonal * (x + offset2)) / 2) -> x = 9 :=
sorry

end find_first_offset_l75_75979


namespace Nicky_wait_time_l75_75159

theorem Nicky_wait_time (x : ℕ) (h1 : x + (4 * x + 14) = 114) : x = 20 :=
by {
  sorry
}

end Nicky_wait_time_l75_75159


namespace restaurant_meals_l75_75601

theorem restaurant_meals (k a : ℕ) (ratio_kids_to_adults : k / a = 10 / 7) (kids_meals_sold : k = 70) : a = 49 :=
by
  sorry

end restaurant_meals_l75_75601


namespace fred_earned_63_dollars_l75_75822

-- Definitions for the conditions
def initial_money_fred : ℕ := 23
def initial_money_jason : ℕ := 46
def money_per_car : ℕ := 5
def money_per_lawn : ℕ := 10
def money_per_dog : ℕ := 3
def total_money_after_chores : ℕ := 86
def cars_washed : ℕ := 4
def lawns_mowed : ℕ := 3
def dogs_walked : ℕ := 7

-- The equivalent proof problem in Lean
theorem fred_earned_63_dollars :
  (initial_money_fred + (cars_washed * money_per_car) + 
      (lawns_mowed * money_per_lawn) + 
      (dogs_walked * money_per_dog) = total_money_after_chores) → 
  ((cars_washed * money_per_car) + 
      (lawns_mowed * money_per_lawn) + 
      (dogs_walked * money_per_dog) = 63) :=
by
  sorry

end fred_earned_63_dollars_l75_75822


namespace no_two_digit_factorization_2023_l75_75167

theorem no_two_digit_factorization_2023 :
  ¬ ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 2023 := 
by
  sorry

end no_two_digit_factorization_2023_l75_75167


namespace num_audio_cassettes_in_second_set_l75_75631

-- Define the variables and constants
def costOfAudio (A : ℕ) : ℕ := A
def costOfVideo (V : ℕ) : ℕ := V
def totalCost (numOfAudio : ℕ) (numOfVideo : ℕ) (A : ℕ) (V : ℕ) : ℕ :=
  numOfAudio * (costOfAudio A) + numOfVideo * (costOfVideo V)

-- Given conditions
def condition1 (A V : ℕ) : Prop := ∃ X : ℕ, totalCost X 4 A V = 1350
def condition2 (A V : ℕ) : Prop := totalCost 7 3 A V = 1110
def condition3 : Prop := costOfVideo 300 = 300

-- Main theorem to prove: The number of audio cassettes in the second set is 7
theorem num_audio_cassettes_in_second_set :
  ∃ (A : ℕ), condition1 A 300 ∧ condition2 A 300 ∧ condition3 →
  7 = 7 :=
by
  sorry

end num_audio_cassettes_in_second_set_l75_75631


namespace part_a_part_b_l75_75805

theorem part_a {a b c : ℝ} : ∃ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 :=
sorry

theorem part_b {a b c : ℝ} : (a + b + c) ^ 2 ≥ 3 * (a * b + b * c + c * a) :=
sorry

end part_a_part_b_l75_75805


namespace exponent_of_term_on_right_side_l75_75694

theorem exponent_of_term_on_right_side
  (s m : ℕ) 
  (h1 : (2^16) * (25^s) = 5 * (10^m))
  (h2 : m = 16) : m = 16 := 
by
  sorry

end exponent_of_term_on_right_side_l75_75694


namespace ones_mult_palindrome_l75_75058

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 
  digits = digits.reverse

def ones (k : ℕ) : ℕ := (10 ^ k - 1) / 9

theorem ones_mult_palindrome (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  is_palindrome (ones m * ones n) ↔ (m = n ∧ m ≤ 9 ∧ n ≤ 9) := 
sorry

end ones_mult_palindrome_l75_75058


namespace pencil_case_solution_part1_pencil_case_solution_part2_1_pencil_case_solution_part2_2_l75_75365

section pencil_case_problem

variables (x m : ℕ)

-- Part 1: The cost prices of each $A$ type and $B$ type pencil cases.
def cost_price_A (x : ℕ) : Prop := 
  (800 : ℝ) / x = (1000 : ℝ) / (x + 2)

-- Part 2.1: Maximum quantity of $B$ type pencil cases.
def max_quantity_B (m : ℕ) : Prop := 
  3 * m - 50 + m ≤ 910

-- Part 2.2: Number of different scenarios for purchasing the pencil cases.
def profit_condition (m : ℕ) : Prop := 
  4 * (3 * m - 50) + 5 * m > 3795

theorem pencil_case_solution_part1 (hA : cost_price_A x) : 
  x = 8 := 
sorry

theorem pencil_case_solution_part2_1 (hB : max_quantity_B m) : 
  m ≤ 240 := 
sorry

theorem pencil_case_solution_part2_2 (hB : max_quantity_B m) (hp : profit_condition m) : 
  236 ≤ m ∧ m ≤ 240 := 
sorry

end pencil_case_problem

end pencil_case_solution_part1_pencil_case_solution_part2_1_pencil_case_solution_part2_2_l75_75365


namespace product_is_zero_l75_75819

variables {a b c d : ℤ}

def system_of_equations (a b c d : ℤ) :=
  2 * a + 3 * b + 5 * c + 7 * d = 34 ∧
  3 * (d + c) = b ∧
  3 * b + c = a ∧
  c - 1 = d

theorem product_is_zero (h : system_of_equations a b c d) : 
  a * b * c * d = 0 :=
sorry

end product_is_zero_l75_75819


namespace true_propositions_count_l75_75371

theorem true_propositions_count (b : ℤ) :
  (b = 3 → b^2 = 9) → 
  (∃! p : Prop, p = (b^2 ≠ 9 → b ≠ 3) ∨ p = (b ≠ 3 → b^2 ≠ 9) ∨ p = (b^2 = 9 → b = 3) ∧ (p = (b^2 ≠ 9 → b ≠ 3))) :=
sorry

end true_propositions_count_l75_75371


namespace paul_bags_on_saturday_l75_75558

-- Definitions and Conditions
def total_cans : ℕ := 72
def cans_per_bag : ℕ := 8
def extra_bags : ℕ := 3

-- Statement of the problem
theorem paul_bags_on_saturday (S : ℕ) :
  S * cans_per_bag = total_cans - (extra_bags * cans_per_bag) →
  S = 6 :=
sorry

end paul_bags_on_saturday_l75_75558


namespace gold_bars_distribution_l75_75074

theorem gold_bars_distribution 
  (initial_gold : ℕ) 
  (lost_gold : ℕ) 
  (num_friends : ℕ) 
  (remaining_gold : ℕ)
  (each_friend_gets : ℕ) :
  initial_gold = 100 →
  lost_gold = 20 →
  num_friends = 4 →
  remaining_gold = initial_gold - lost_gold →
  each_friend_gets = remaining_gold / num_friends →
  each_friend_gets = 20 :=
by
  intros
  sorry

end gold_bars_distribution_l75_75074


namespace vector_projection_condition_l75_75268

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + 3 * t, 3 + 2 * t)
noncomputable def line_m (s : ℝ) : ℝ × ℝ := (4 + 2 * s, 5 + 3 * s)

def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem vector_projection_condition 
  (t s : ℝ)
  (C : ℝ × ℝ := line_l t)
  (D : ℝ × ℝ := line_m s)
  (Q : ℝ × ℝ)
  (hQ : is_perpendicular (Q.1 - C.1, Q.2 - C.2) (2, 3))
  (v1 v2 : ℝ)
  (hv_sum : v1 + v2 = 3)
  (hv_def : ∃ k : ℝ, v1 = 3 * k ∧ v2 = -2 * k)
  : (v1, v2) = (9, -6) := 
sorry

end vector_projection_condition_l75_75268


namespace find_f2_plus_g2_l75_75714

variable (f g : ℝ → ℝ)

def even_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x
def odd_function (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

theorem find_f2_plus_g2 (hf : even_function f) (hg : odd_function g) (h : ∀ x, f x - g x = x^3 - 2 * x^2) :
  f 2 + g 2 = -16 :=
sorry

end find_f2_plus_g2_l75_75714


namespace range_j_l75_75498

def h (x : ℝ) : ℝ := 2 * x + 3

def j (x : ℝ) : ℝ := h (h (h (h x)))

theorem range_j : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → 61 ≤ j x ∧ j x ≤ 93) := 
by 
  sorry

end range_j_l75_75498


namespace find_k_value_l75_75803

theorem find_k_value (k : ℚ) (h1 : (3, -5) ∈ {p : ℚ × ℚ | p.snd = k * p.fst}) (h2 : k ≠ 0) : k = -5 / 3 :=
sorry

end find_k_value_l75_75803


namespace determine_f_l75_75913

noncomputable def f : ℝ → ℝ := sorry

axiom f_0 : f 0 = 1
axiom f_eq (x y : ℝ) : f (x * y + 1) = f x * f y - f y - x + 2

theorem determine_f (x : ℝ) : f x = x + 1 := by
  sorry

end determine_f_l75_75913


namespace find_k_l75_75534

def a : ℝ × ℝ := (2, 1)
def b (k : ℝ) : ℝ × ℝ := (-2, k)
def vec_op (a b : ℝ × ℝ) : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)

noncomputable def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_k (k : ℝ) : dot_prod a (vec_op a (b k)) = 0 → k = 14 :=
by
  sorry

end find_k_l75_75534


namespace percentage_of_invalid_votes_l75_75143

-- Candidate A got 60% of the total valid votes.
-- The total number of votes is 560000.
-- The number of valid votes polled in favor of candidate A is 285600.
variable (total_votes valid_votes_A : ℝ)
variable (percent_A : ℝ := 0.60)
variable (valid_votes total_invalid_votes percent_invalid_votes : ℝ)

axiom h1 : total_votes = 560000
axiom h2 : valid_votes_A = 285600
axiom h3 : valid_votes_A = percent_A * valid_votes
axiom h4 : total_invalid_votes = total_votes - valid_votes
axiom h5 : percent_invalid_votes = (total_invalid_votes / total_votes) * 100

theorem percentage_of_invalid_votes : percent_invalid_votes = 15 := by
  sorry

end percentage_of_invalid_votes_l75_75143


namespace joan_apples_l75_75984

def initial_apples : ℕ := 43
def additional_apples : ℕ := 27
def total_apples (initial additional: ℕ) := initial + additional

theorem joan_apples : total_apples initial_apples additional_apples = 70 := by
  sorry

end joan_apples_l75_75984


namespace solution_for_x_l75_75139

theorem solution_for_x : ∀ (x : ℚ), (10 * x^2 + 9 * x - 2 = 0) ∧ (30 * x^2 + 59 * x - 6 = 0) → x = 1 / 5 :=
by
  sorry

end solution_for_x_l75_75139


namespace number_of_B_is_14_l75_75263

-- Define the problem conditions
variable (num_students : ℕ)
variable (num_A num_B num_C num_D : ℕ)
variable (h1 : num_A = 8 * num_B / 10)
variable (h2 : num_C = 13 * num_B / 10)
variable (h3 : num_D = 5 * num_B / 10)
variable (h4 : num_students = 50)
variable (h5 : num_A + num_B + num_C + num_D = num_students)

-- Formalize the statement to be proved
theorem number_of_B_is_14 :
  num_B = 14 := by
  sorry

end number_of_B_is_14_l75_75263


namespace probability_no_adjacent_same_roll_l75_75059

theorem probability_no_adjacent_same_roll :
  let A := 1 -- rolls a six-sided die
  let B := 2 -- rolls a six-sided die
  let C := 3 -- rolls a six-sided die
  let D := 4 -- rolls a six-sided die
  let E := 5 -- rolls a six-sided die
  let people := [A, B, C, D, E]
  -- A and C are required to roll different numbers
  let prob_A_C_diff := 5 / 6
  -- B must roll different from A and C
  let prob_B_diff := 4 / 6
  -- D must roll different from C and A
  let prob_D_diff := 4 / 6
  -- E must roll different from D and A
  let prob_E_diff := 3 / 6
  (prob_A_C_diff * prob_B_diff * prob_D_diff * prob_E_diff) = 10 / 27 :=
by
  sorry

end probability_no_adjacent_same_roll_l75_75059


namespace line_length_400_l75_75929

noncomputable def length_of_line (speed_march_kmh speed_run_kmh total_time_min: ℝ) : ℝ :=
  let speed_march_mpm := (speed_march_kmh * 1000) / 60
  let speed_run_mpm := (speed_run_kmh * 1000) / 60
  let len_eq := 1 / (speed_run_mpm - speed_march_mpm) + 1 / (speed_run_mpm + speed_march_mpm)
  (total_time_min * 200 * len_eq) * 400 / len_eq

theorem line_length_400 :
  length_of_line 8 12 7.2 = 400 := by
  sorry

end line_length_400_l75_75929


namespace magical_stack_130_cards_l75_75772

theorem magical_stack_130_cards (n : ℕ) (h1 : 2 * n > 0) (h2 : ∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ 2 * (n - k + 1) = 131 ∨ 
                                   (n + 1) ≤ k ∧ k ≤ 2 * n ∧ 2 * k - 1 = 131) : 2 * n = 130 :=
by
  sorry

end magical_stack_130_cards_l75_75772


namespace graph_inequality_solution_l75_75907

noncomputable def solution_set : Set (Real × Real) := {
  p | let x := p.1
       let y := p.2
       (y^2 - (Real.arcsin (Real.sin x))^2) *
       (y^2 - (Real.arcsin (Real.sin (x + Real.pi / 3)))^2) *
       (y^2 - (Real.arcsin (Real.sin (x - Real.pi / 3)))^2) < 0
}

theorem graph_inequality_solution
  (x y : ℝ) :
  (y^2 - (Real.arcsin (Real.sin x))^2) *
  (y^2 - (Real.arcsin (Real.sin (x + Real.pi / 3)))^2) *
  (y^2 - (Real.arcsin (Real.sin (x - Real.pi / 3)))^2) < 0 ↔
  (x, y) ∈ solution_set :=
by
  sorry

end graph_inequality_solution_l75_75907


namespace number_of_markings_l75_75817

def markings (L : ℕ → ℕ) := ∀ n, (n > 0) → L n = L (n - 1) + 1

theorem number_of_markings : ∃ L : ℕ → ℕ, (∀ n, n = 1 → L n = 2) ∧ markings L ∧ L 200 = 201 := 
sorry

end number_of_markings_l75_75817


namespace fraction_problem_l75_75055

def fractions : List (ℚ) := [4/3, 7/5, 12/10, 23/20, 45/40, 89/80]
def subtracted_value : ℚ := -8

theorem fraction_problem :
  (fractions.sum - subtracted_value) = -163 / 240 := by
  sorry

end fraction_problem_l75_75055


namespace tiles_difference_ninth_eighth_rectangle_l75_75075

theorem tiles_difference_ninth_eighth_rectangle : 
  let width (n : Nat) := 2 * n
  let height (n : Nat) := n
  let tiles (n : Nat) := width n * height n
  tiles 9 - tiles 8 = 34 :=
by
  intro width height tiles
  sorry

end tiles_difference_ninth_eighth_rectangle_l75_75075


namespace loss_per_metre_proof_l75_75934

-- Define the given conditions
def cost_price_per_metre : ℕ := 66
def quantity_sold : ℕ := 200
def total_selling_price : ℕ := 12000

-- Define total cost price based on cost price per metre and quantity sold
def total_cost_price : ℕ := cost_price_per_metre * quantity_sold

-- Define total loss based on total cost price and total selling price
def total_loss : ℕ := total_cost_price - total_selling_price

-- Define loss per metre
def loss_per_metre : ℕ := total_loss / quantity_sold

-- The theorem we need to prove:
theorem loss_per_metre_proof : loss_per_metre = 6 :=
  by
    sorry

end loss_per_metre_proof_l75_75934


namespace proof_valid_x_values_l75_75152

noncomputable def valid_x_values (x : ℝ) : Prop :=
  (x^2 + 2*x^3 - 3*x^4) / (x + 2*x^2 - 3*x^3) ≤ 1

theorem proof_valid_x_values :
  {x : ℝ | valid_x_values x} = {x : ℝ | (x < -1) ∨ (x > -1 ∧ x < 0) ∨ (x > 0 ∧ x < 1)} :=
by {
  sorry
}

end proof_valid_x_values_l75_75152


namespace find_a_l75_75307

theorem find_a 
  (a : ℝ)
  (h : ∀ n : ℕ, (n.choose 2) * 2^(5-2) * a^2 = 80 → n = 5) :
  a = 1 ∨ a = -1 :=
by
  sorry

end find_a_l75_75307


namespace at_most_one_solution_l75_75063

theorem at_most_one_solution (a b c : ℝ) (hapos : 0 < a) (hbpos : 0 < b) (hcpos : 0 < c) :
  ∃! x : ℝ, a * x + b * ⌊x⌋ - c = 0 :=
sorry

end at_most_one_solution_l75_75063


namespace equation_transformation_l75_75764

theorem equation_transformation (x y: ℝ) (h : 2 * x - 3 * y = 6) : 
  y = (2 * x - 6) / 3 := 
by
  sorry

end equation_transformation_l75_75764


namespace find_y_l75_75168

theorem find_y (x y : ℤ) (h1 : x + 2 * y = 100) (h2 : x = 50) : y = 25 :=
by
  sorry

end find_y_l75_75168


namespace identify_quadratic_equation_l75_75687

def is_quadratic (eq : String) : Prop :=
  eq = "a * x^2 + b * x + c = 0"  /-
  This definition is a placeholder for checking if a 
  given equation is in the quadratic form. In practice,
  more advanced techniques like parsing and formally
  verifying the quadratic form would be used. -/

theorem identify_quadratic_equation :
  (is_quadratic "2 * x^2 - x - 3 = 0") :=
by
  sorry

end identify_quadratic_equation_l75_75687


namespace determine_k_l75_75834

theorem determine_k (k : ℕ) : 2^2004 - 2^2003 - 2^2002 + 2^2001 = k * 2^2001 → k = 3 :=
by
  intro h
  -- now we would proceed to prove it, but we'll skip proof here
  sorry

end determine_k_l75_75834


namespace min_area_triangle_l75_75421

theorem min_area_triangle (m n : ℝ) (h1 : (1 : ℝ) / m + (2 : ℝ) / n = 1) (h2 : m > 0) (h3 : n > 0) :
  ∃ A B C : ℝ, 
  ((0 < A) ∧ (0 < B) ∧ ((1 : ℝ) / A + (2 : ℝ) / B = 1) ∧ (A * B = C) ∧ (2 / C = mn)) ∧ (C = 4) :=
by
  sorry

end min_area_triangle_l75_75421


namespace inequality_solution_set_l75_75054

theorem inequality_solution_set : 
  { x : ℝ | (x + 1) / (x + 2) < 0 } = { x : ℝ | -2 < x ∧ x < -1 } := 
by
  sorry 

end inequality_solution_set_l75_75054


namespace range_of_a_l75_75608

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - a| < 4) → -1 < a ∧ a < 7 :=
  sorry

end range_of_a_l75_75608


namespace option_B_correct_l75_75102

theorem option_B_correct (x y : ℝ) : 
  x * y^2 - y^2 * x = 0 :=
by sorry

end option_B_correct_l75_75102


namespace mean_score_for_exam_l75_75518

variable (M SD : ℝ)

-- Define the conditions
def condition1 : Prop := 58 = M - 2 * SD
def condition2 : Prop := 98 = M + 3 * SD

-- The problem statement
theorem mean_score_for_exam (h1 : condition1 M SD) (h2 : condition2 M SD) : M = 74 :=
sorry

end mean_score_for_exam_l75_75518


namespace total_distance_correct_l75_75357

noncomputable def total_distance_covered (rA rB rC : ℝ) (revA revB revC : ℕ) : ℝ :=
  let pi := Real.pi
  let circumference (r : ℝ) := 2 * pi * r
  let distance (r : ℝ) (rev : ℕ) := circumference r * rev
  distance rA revA + distance rB revB + distance rC revC

theorem total_distance_correct :
  total_distance_covered 22.4 35.7 55.9 600 450 375 = 316015.4 :=
by
  sorry

end total_distance_correct_l75_75357


namespace area_of_square_field_l75_75653

theorem area_of_square_field (s : ℕ) (A : ℕ) (cost_per_meter : ℕ) 
  (total_cost : ℕ) (gate_width : ℕ) (num_gates : ℕ) 
  (h1 : cost_per_meter = 1)
  (h2 : total_cost = 666)
  (h3 : gate_width = 1)
  (h4 : num_gates = 2)
  (h5 : (4 * s - num_gates * gate_width) * cost_per_meter = total_cost) :
  A = s * s → A = 27889 :=
by
  sorry

end area_of_square_field_l75_75653


namespace find_xy_integers_l75_75515

theorem find_xy_integers (x y : ℤ) (h : x^3 + 2 * x * y = 7) :
  (x, y) = (-7, -25) ∨ (x, y) = (-1, -4) ∨ (x, y) = (1, 3) ∨ (x, y) = (7, -24) :=
sorry

end find_xy_integers_l75_75515


namespace corveus_sleep_hours_l75_75203

-- Definition of the recommended hours of sleep per day
def recommended_sleep_per_day : ℕ := 6

-- Definition of the hours of sleep Corveus lacks per week
def lacking_sleep_per_week : ℕ := 14

-- Definition of days in a week
def days_in_week : ℕ := 7

-- Prove that Corveus sleeps 4 hours per day given the conditions
theorem corveus_sleep_hours :
  (recommended_sleep_per_day * days_in_week - lacking_sleep_per_week) / days_in_week = 4 :=
by
  -- The proof steps would go here
  sorry

end corveus_sleep_hours_l75_75203


namespace consecutive_page_sum_l75_75459

theorem consecutive_page_sum (n : ℕ) (h : n * (n + 1) = 2156) : n + (n + 1) = 93 :=
sorry

end consecutive_page_sum_l75_75459


namespace extreme_value_f_max_b_a_plus_1_l75_75408

noncomputable def f (x : ℝ) := Real.exp x - x + (1/2)*x^2

noncomputable def g (x : ℝ) (a b : ℝ) := (1/2)*x^2 + a*x + b

theorem extreme_value_f :
  ∃ x, deriv f x = 0 ∧ f x = 3 / 2 :=
sorry

theorem max_b_a_plus_1 (a : ℝ) (b : ℝ) :
  (∀ x, f x ≥ g x a b) → b * (a+1) ≤ (a+1)^2 - (a+1)^2 * Real.log (a+1) :=
sorry

end extreme_value_f_max_b_a_plus_1_l75_75408


namespace darnel_difference_l75_75150

theorem darnel_difference (sprint_1 jog_1 sprint_2 jog_2 sprint_3 jog_3 : ℝ)
  (h_sprint_1 : sprint_1 = 0.8932)
  (h_jog_1 : jog_1 = 0.7683)
  (h_sprint_2 : sprint_2 = 0.9821)
  (h_jog_2 : jog_2 = 0.4356)
  (h_sprint_3 : sprint_3 = 1.2534)
  (h_jog_3 : jog_3 = 0.6549) :
  (sprint_1 + sprint_2 + sprint_3 - (jog_1 + jog_2 + jog_3)) = 1.2699 := by
  sorry

end darnel_difference_l75_75150


namespace percentage_decrease_in_savings_l75_75707

theorem percentage_decrease_in_savings (I : ℝ) (F : ℝ) (IncPercent : ℝ) (decPercent : ℝ)
  (h1 : I = 125) (h2 : IncPercent = 0.25) (h3 : F = 125) :
  let P := (I * (1 + IncPercent))
  ∃ decPercent, decPercent = ((P - F) / P) * 100 ∧ decPercent = 20 := 
by
  sorry

end percentage_decrease_in_savings_l75_75707


namespace petya_friends_count_l75_75889

-- Define the number of classmates
def total_classmates : ℕ := 28

-- Each classmate has a unique number of friends from 0 to 27
def unique_friends (n : ℕ) : Prop :=
  n ≥ 0 ∧ n < total_classmates

-- We state the problem where Petya's number of friends is to be proven as 14
theorem petya_friends_count (friends : ℕ) (h : unique_friends friends) : friends = 14 :=
sorry

end petya_friends_count_l75_75889


namespace inequality_solution_set_l75_75938

theorem inequality_solution_set (x : ℝ) : |x - 5| + |x + 3| ≤ 10 ↔ -4 ≤ x ∧ x ≤ 6 :=
by
  sorry

end inequality_solution_set_l75_75938


namespace find_d_l75_75331

-- Definitions of the conditions
variables (r s t u d : ℤ)

-- Assume r, s, t, and u are positive integers
axiom r_pos : r > 0
axiom s_pos : s > 0
axiom t_pos : t > 0
axiom u_pos : u > 0

-- Given conditions
axiom h1 : r ^ 5 = s ^ 4
axiom h2 : t ^ 3 = u ^ 2
axiom h3 : t - r = 19
axiom h4 : d = u - s

-- Proof statement
theorem find_d : d = 757 :=
by sorry

end find_d_l75_75331


namespace hall_length_l75_75939

variable (breadth length : ℝ)

def condition1 : Prop := length = breadth + 5
def condition2 : Prop := length * breadth = 750

theorem hall_length : condition1 breadth length ∧ condition2 breadth length → length = 30 :=
by
  intros
  sorry

end hall_length_l75_75939


namespace evaluate_expression_l75_75426

noncomputable def complex_numbers_condition (a b : ℂ) := a ≠ 0 ∧ b ≠ 0 ∧ (a^2 + a * b + b^2 = 0)

theorem evaluate_expression (a b : ℂ) (h : complex_numbers_condition a b) : 
  (a^5 + b^5) / (a + b)^5 = -2 := 
by
  sorry

end evaluate_expression_l75_75426


namespace num_valid_m_divisors_of_1750_l75_75041

theorem num_valid_m_divisors_of_1750 : 
  ∃! (m : ℕ) (h1 : m > 0), ∃ (k : ℕ), k > 0 ∧ 1750 = k * (m^2 - 4) :=
sorry

end num_valid_m_divisors_of_1750_l75_75041


namespace determine_a_range_l75_75118

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x
noncomputable def g (x : ℝ) : ℝ := Real.exp x

theorem determine_a_range (e : ℝ) (he : e = Real.exp 1) :
  ∃ a_range : Set ℝ, a_range = Set.Icc 1 (e + 1 / e) :=
by 
  sorry

end determine_a_range_l75_75118


namespace find_c_l75_75837

theorem find_c (x : ℝ) (c : ℝ) (h1 : 3 * x + 5 = 4) (h2 : c * x + 6 = 3) : c = 9 :=
by
  sorry

end find_c_l75_75837


namespace sum_a1_a3_a5_l75_75533

-- Definitions
variable (a : ℕ → ℕ)
variable (b : ℕ → ℕ)

-- Conditions
axiom initial_condition : a 1 = 16
axiom relationship_ak_bk : ∀ k, b k = a k / 2
axiom ak_next : ∀ k, a (k + 1) = a k + 2 * (b k)

-- Theorem Statement
theorem sum_a1_a3_a5 : a 1 + a 3 + a 5 = 336 :=
by
  sorry

end sum_a1_a3_a5_l75_75533


namespace skier_total_time_l75_75587

variable (t1 t2 t3 : ℝ)

-- Conditions
def condition1 : Prop := t1 + t2 = 40.5
def condition2 : Prop := t2 + t3 = 37.5
def condition3 : Prop := 1 / t2 = 2 / (t1 + t3)

-- Theorem to prove total time is 58.5 minutes
theorem skier_total_time (h1 : condition1 t1 t2) (h2 : condition2 t2 t3) (h3 : condition3 t1 t2 t3) : t1 + t2 + t3 = 58.5 := 
by
  sorry

end skier_total_time_l75_75587


namespace smartphone_price_l75_75024

theorem smartphone_price (S : ℝ) (pc_price : ℝ) (tablet_price : ℝ) 
  (total_cost : ℝ) (h1 : pc_price = S + 500) 
  (h2 : tablet_price = 2 * S + 500) 
  (h3 : S + pc_price + tablet_price = 2200) : 
  S = 300 :=
by
  sorry

end smartphone_price_l75_75024


namespace total_selling_price_correct_l75_75362

-- Define the conditions
def metres_of_cloth : ℕ := 500
def loss_per_metre : ℕ := 5
def cost_price_per_metre : ℕ := 41
def selling_price_per_metre : ℕ := cost_price_per_metre - loss_per_metre
def expected_total_selling_price : ℕ := 18000

-- Define the theorem
theorem total_selling_price_correct : 
  selling_price_per_metre * metres_of_cloth = expected_total_selling_price := 
by
  sorry

end total_selling_price_correct_l75_75362


namespace union_comm_inter_comm_union_assoc_inter_assoc_inter_union_distrib_union_inter_distrib_union_idem_inter_idem_de_morgan_union_de_morgan_inter_l75_75899

open Set

variables {α : Type*} (A B C : Set α)

-- Commutativity
theorem union_comm : A ∪ B = B ∪ A := sorry
theorem inter_comm : A ∩ B = B ∩ A := sorry

-- Associativity
theorem union_assoc : A ∪ (B ∪ C) = (A ∪ B) ∪ C := sorry
theorem inter_assoc : A ∩ (B ∩ C) = (A ∩ B) ∩ C := sorry

-- Distributivity
theorem inter_union_distrib : A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) := sorry
theorem union_inter_distrib : A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C) := sorry

-- Idempotence
theorem union_idem : A ∪ A = A := sorry
theorem inter_idem : A ∩ A = A := sorry

-- De Morgan's Laws
theorem de_morgan_union : compl (A ∪ B) = compl A ∩ compl B := sorry
theorem de_morgan_inter : compl (A ∩ B) = compl A ∪ compl B := sorry

end union_comm_inter_comm_union_assoc_inter_assoc_inter_union_distrib_union_inter_distrib_union_idem_inter_idem_de_morgan_union_de_morgan_inter_l75_75899


namespace functional_equation_solution_l75_75683

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, 
  (∀ x y : ℝ, 
      y * f (2 * x) - x * f (2 * y) = 8 * x * y * (x^2 - y^2)
  ) → (∃ c : ℝ, ∀ x : ℝ, f x = x^3 + c * x) :=
by { sorry }

end functional_equation_solution_l75_75683


namespace symmetric_point_l75_75364

theorem symmetric_point (P Q : ℝ × ℝ)
  (l : ℝ → ℝ)
  (P_coords : P = (-1, 2))
  (l_eq : ∀ x, l x = x - 1) :
  Q = (3, -2) :=
by
  sorry

end symmetric_point_l75_75364


namespace baseball_cap_problem_l75_75249

theorem baseball_cap_problem 
  (n_first_week n_second_week n_third_week n_fourth_week total_caps : ℕ) 
  (h2 : n_second_week = 400) 
  (h3 : n_third_week = 300) 
  (h4 : n_fourth_week = (n_first_week + n_second_week + n_third_week) / 3) 
  (h_total : n_first_week + n_second_week + n_third_week + n_fourth_week = 1360) : 
  n_first_week = 320 := 
by 
  sorry

end baseball_cap_problem_l75_75249


namespace range_g_l75_75828

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + 2 * Real.arcsin x

theorem range_g : 
  (∀ x, -1 ≤ x ∧ x ≤ 1 → -Real.pi / 2 ≤ g x ∧ g x ≤ 3 * Real.pi / 2) := 
by {
  sorry
}

end range_g_l75_75828


namespace optionA_incorrect_optionB_incorrect_optionC_incorrect_optionD_correct_l75_75956

theorem optionA_incorrect (a x : ℝ) : 3 * a * x^2 - 6 * a * x ≠ 3 * (a * x^2 - 2 * a * x) :=
by sorry

theorem optionB_incorrect (a x : ℝ) : (x + a) * (x - a) ≠ x^2 - a^2 :=
by sorry

theorem optionC_incorrect (a b : ℝ) : a^2 + 2 * a * b - 4 * b^2 ≠ (a + 2 * b)^2 :=
by sorry

theorem optionD_correct (a x : ℝ) : -a * x^2 + 2 * a * x - a = -a * (x - 1)^2 :=
by sorry

end optionA_incorrect_optionB_incorrect_optionC_incorrect_optionD_correct_l75_75956


namespace part1_part2_l75_75278

theorem part1 (n : ℕ) (students : Finset ℕ) (d : students → students → ℕ) :
  (∀ (a b : students), a ≠ b → d a b ≠ d b a) →
  (∀ (a b c : students), a ≠ b ∧ b ≠ c ∧ a ≠ c → d a b ≠ d b c ∧ d a b ≠ d a c ∧ d a c ≠ d b c) →
  (students.card = 2 * n + 1) →
  ∃ a b : students, a ≠ b ∧
  (∀ c : students, c ≠ a → d a c > d a b) ∧ 
  (∀ c : students, c ≠ b → d b c > d b a) :=
sorry

theorem part2 (n : ℕ) (students : Finset ℕ) (d : students → students → ℕ) :
  (∀ (a b : students), a ≠ b → d a b ≠ d b a) →
  (∀ (a b c : students), a ≠ b ∧ b ≠ c ∧ a ≠ c → d a b ≠ d b c ∧ d a b ≠ d a c ∧ d a c ≠ d b c) →
  (students.card = 2 * n + 1) →
  ∃ c : students, ∀ a : students, ¬ (∀ b : students, b ≠ a → d b a < d b c ∧ d a c < d a b) :=
sorry

end part1_part2_l75_75278


namespace min_jellybeans_l75_75079

theorem min_jellybeans (n : ℕ) (h1 : n ≥ 150) (h2 : n % 17 = 15) : n = 151 :=
by { sorry }

end min_jellybeans_l75_75079


namespace zero_of_f_inequality_l75_75946

noncomputable def f (x : ℝ) : ℝ := 2^(-x) - Real.log (x^3 + 1)

variable (a b c x : ℝ)
variable (h : 0 < a ∧ a < b ∧ b < c)
variable (hx : f x = 0)
variable (h₀ : f a * f b * f c < 0)

theorem zero_of_f_inequality :
  ¬ (x > c) :=
by 
  sorry

end zero_of_f_inequality_l75_75946


namespace common_roots_correct_l75_75483

noncomputable section
def common_roots_product (A B : ℝ) : ℝ :=
  let p := sorry
  let q := sorry
  p * q

theorem common_roots_correct (A B : ℝ) (h1 : ∀ x, x^3 + 2*A*x + 20 = 0 → x = p ∨ x = q ∨ x = r) 
    (h2 : ∀ x, x^3 + B*x^2 + 100 = 0 → x = p ∨ x = q ∨ x = s)
    (h_sum1 : p + q + r = 0) 
    (h_sum2 : p + q + s = -B)
    (h_prod1 : p * q * r = -20) 
    (h_prod2 : p * q * s = -100) : 
    common_roots_product A B = 10 * (2000)^(1/3) ∧ 15 = 10 + 3 + 2 :=
by
  sorry

end common_roots_correct_l75_75483


namespace find_r_power_4_l75_75427

variable {r : ℝ}

theorem find_r_power_4 (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := 
sorry

end find_r_power_4_l75_75427


namespace greatest_integer_value_l75_75620

theorem greatest_integer_value (x : ℤ) : ∃ x, (∀ y, (x^2 + 2 * x + 10) % (x - 3) = 0 → x ≥ y) → x = 28 :=
by
  sorry

end greatest_integer_value_l75_75620


namespace solve_ordered_pair_l75_75276

theorem solve_ordered_pair : 
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^y + 3 = y^x ∧ 2 * x^y = y^x + 11 ∧ x = 14 ∧ y = 1 :=
by
  sorry

end solve_ordered_pair_l75_75276


namespace sales_tax_difference_l75_75372

theorem sales_tax_difference:
  let original_price := 50 
  let discount_rate := 0.10 
  let sales_tax_rate_1 := 0.08
  let sales_tax_rate_2 := 0.075 
  let discounted_price := original_price * (1 - discount_rate) 
  let sales_tax_1 := discounted_price * sales_tax_rate_1 
  let sales_tax_2 := discounted_price * sales_tax_rate_2 
  sales_tax_1 - sales_tax_2 = 0.225 := by
  sorry

end sales_tax_difference_l75_75372


namespace probability_of_at_least_ten_heads_in_twelve_given_first_two_heads_l75_75329

-- Define a fair coin
inductive Coin
| Heads
| Tails

def fair_coin : List Coin := [Coin.Heads, Coin.Tails]

-- Define a function to calculate the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  Nat.descFactorial n k / k.factorial

-- Define a function to calculate the probability of at least 8 heads in 10 flips
def prob_at_least_eight_heads_in_ten : ℚ :=
  (binomial 10 8 + binomial 10 9 + binomial 10 10) / (2 ^ 10)

-- Define our theorem statement
theorem probability_of_at_least_ten_heads_in_twelve_given_first_two_heads :
    (prob_at_least_eight_heads_in_ten = 7 / 128) :=
  by
    -- The proof steps can be written here later
    sorry

end probability_of_at_least_ten_heads_in_twelve_given_first_two_heads_l75_75329


namespace find_linear_function_l75_75549

theorem find_linear_function (f : ℝ → ℝ) (hf_inc : ∀ x y, x < y → f x < f y)
  (hf_lin : ∃ a b, a > 0 ∧ ∀ x, f x = a * x + b)
  (h_comp : ∀ x, f (f x) = 4 * x + 3) :
  ∀ x, f x = 2 * x + 1 :=
by
  sorry

end find_linear_function_l75_75549


namespace solution_set_I_range_of_a_l75_75664

-- Define the function f(x) = |x + a| - |x + 1|
def f (x a : ℝ) : ℝ := abs (x + a) - abs (x + 1)

-- Part (I)
theorem solution_set_I (a : ℝ) : 
  (f a a > 1) ↔ (a < -2/3 ∨ a > 2) := by
  sorry

-- Part (II)
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≤ 2 * a) ↔ (a ≥ 1/3) := by
  sorry

end solution_set_I_range_of_a_l75_75664


namespace lassis_from_mangoes_l75_75097

def ratio (lassis mangoes : ℕ) : Prop := lassis = 11 * mangoes / 2

theorem lassis_from_mangoes (mangoes : ℕ) (h : mangoes = 10) : ratio 55 mangoes :=
by
  rw [h]
  unfold ratio
  sorry

end lassis_from_mangoes_l75_75097


namespace find_n_l75_75669

theorem find_n (n : ℕ) (h : 2^n = 2 * 16^2 * 4^3) : n = 15 :=
by
  sorry

end find_n_l75_75669


namespace radius_of_circle_l75_75536

theorem radius_of_circle (r : ℝ) (h : 3 * (2 * Real.pi * r) = 2 * (Real.pi * r ^ 2)) : r = 3 := by
  sorry

end radius_of_circle_l75_75536


namespace smallest_pencils_l75_75107

theorem smallest_pencils (P : ℕ) :
  (P > 2) ∧
  (P % 5 = 2) ∧
  (P % 9 = 2) ∧
  (P % 11 = 2) →
  P = 497 := by
  sorry

end smallest_pencils_l75_75107


namespace x_is_one_if_pure_imaginary_l75_75832

theorem x_is_one_if_pure_imaginary
  (x : ℝ)
  (h1 : x^2 - 1 = 0)
  (h2 : x^2 + 3 * x + 2 ≠ 0) :
  x = 1 :=
sorry

end x_is_one_if_pure_imaginary_l75_75832


namespace number_of_outfits_l75_75719

theorem number_of_outfits (shirts pants : ℕ) (h_shirts : shirts = 5) (h_pants : pants = 3) 
    : shirts * pants = 15 := by
  sorry

end number_of_outfits_l75_75719


namespace books_loaned_out_during_month_l75_75454

-- Define the initial conditions
def initial_books : ℕ := 75
def remaining_books : ℕ := 65
def loaned_out_percentage : ℝ := 0.80
def returned_books_ratio : ℝ := loaned_out_percentage
def not_returned_ratio : ℝ := 1 - returned_books_ratio
def difference : ℕ := initial_books - remaining_books

-- Define the main theorem
theorem books_loaned_out_during_month : ∃ (x : ℕ), not_returned_ratio * (x : ℝ) = (difference : ℝ) ∧ x = 50 :=
by
  existsi 50
  simp [not_returned_ratio, difference]
  sorry

end books_loaned_out_during_month_l75_75454


namespace provisions_remaining_days_l75_75417

-- Definitions based on the conditions
def initial_men : ℕ := 1000
def initial_provisions_days : ℕ := 60
def days_elapsed : ℕ := 15
def reinforcement_men : ℕ := 1250

-- Mathematical computation for Lean
def total_provisions : ℕ := initial_men * initial_provisions_days
def provisions_left : ℕ := initial_men * (initial_provisions_days - days_elapsed)
def total_men_after_reinforcement : ℕ := initial_men + reinforcement_men

-- Statement to prove
theorem provisions_remaining_days : provisions_left / total_men_after_reinforcement = 20 :=
by
  -- The proof steps will be filled here, but for now, we use sorry to skip them.
  sorry

end provisions_remaining_days_l75_75417


namespace net_income_on_15th_day_l75_75283

noncomputable def net_income_15th_day : ℝ :=
  let earnings_15th_day := 3 * (3 ^ 14)
  let tax := 0.10 * earnings_15th_day
  let earnings_after_tax := earnings_15th_day - tax
  earnings_after_tax - 100

theorem net_income_on_15th_day :
  net_income_15th_day = 12913916.3 := by
  sorry

end net_income_on_15th_day_l75_75283


namespace ms_lee_class_difference_l75_75761

noncomputable def boys_and_girls_difference (ratio_b : ℕ) (ratio_g : ℕ) (total_students : ℕ) : ℕ :=
  let x := total_students / (ratio_b + ratio_g)
  let boys := ratio_b * x
  let girls := ratio_g * x
  girls - boys

theorem ms_lee_class_difference :
  boys_and_girls_difference 3 4 42 = 6 :=
by
  sorry

end ms_lee_class_difference_l75_75761


namespace minimum_discount_correct_l75_75989

noncomputable def minimum_discount (total_weight: ℝ) (cost_price: ℝ) (sell_price: ℝ) 
                                   (profit_required: ℝ) : ℝ :=
  let first_half_profit := (total_weight / 2) * (sell_price - cost_price)
  let second_half_profit_with_discount (x: ℝ) := (total_weight / 2) * (sell_price * x - cost_price)
  let required_profit_condition (x: ℝ) := first_half_profit + second_half_profit_with_discount x ≥ profit_required
  (1 - (7 / 11))

theorem minimum_discount_correct : minimum_discount 1000 7 10 2000 = 4 / 11 := 
by {
  -- We need to solve the inequality step by step to reach the final answer
  sorry
}

end minimum_discount_correct_l75_75989


namespace mental_math_quiz_l75_75038

theorem mental_math_quiz : ∃ (q_i q_c : ℕ), q_c + q_i = 100 ∧ 10 * q_c - 5 * q_i = 850 ∧ q_i = 10 :=
by
  sorry

end mental_math_quiz_l75_75038


namespace peter_reads_more_books_l75_75484

-- Definitions and conditions
def total_books : ℕ := 20
def peter_percentage_read : ℕ := 40
def brother_percentage_read : ℕ := 10

def percentage_to_count (percentage : ℕ) (total : ℕ) : ℕ := (percentage * total) / 100

-- Main statement to prove
theorem peter_reads_more_books :
  percentage_to_count peter_percentage_read total_books - percentage_to_count brother_percentage_read total_books = 6 :=
by
  sorry

end peter_reads_more_books_l75_75484


namespace number_of_games_played_l75_75556

-- Define our conditions
def teams : ℕ := 14
def games_per_pair : ℕ := 5

-- Define the function to calculate the number of combinations
def combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the expected total games
def total_games : ℕ := 455

-- Statement asserting that given the conditions, the number of games played in the season is total_games
theorem number_of_games_played : (combinations teams 2) * games_per_pair = total_games := 
by 
  sorry

end number_of_games_played_l75_75556


namespace ellipse_and_triangle_properties_l75_75127

noncomputable def standard_equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  1/2 * a * b

theorem ellipse_and_triangle_properties :
  (∀ x y : ℝ, standard_equation_of_ellipse x y ↔ (x, y) = (1, 3/2) ∨ (x, y) = (1, -3/2)) ∧
  area_triangle 2 3 = 3 :=
by
  sorry

end ellipse_and_triangle_properties_l75_75127


namespace pyramid_boxes_l75_75825

theorem pyramid_boxes (a₁ a₂ aₙ : ℕ) (d : ℕ) (n : ℕ) (Sₙ : ℕ) 
  (h₁ : a₁ = 12) 
  (h₂ : a₂ = 15) 
  (h₃ : aₙ = 39) 
  (h₄ : d = 3) 
  (h₅ : a₂ = a₁ + d)
  (h₆ : aₙ = a₁ + (n - 1) * d) 
  (h₇ : Sₙ = n * (a₁ + aₙ) / 2) :
  Sₙ = 255 :=
by
  sorry

end pyramid_boxes_l75_75825


namespace monica_read_books_l75_75332

theorem monica_read_books (x : ℕ) 
    (h1 : 2 * (2 * x) + 5 = 69) : 
    x = 16 :=
by 
  sorry

end monica_read_books_l75_75332


namespace InequalityProof_l75_75681

theorem InequalityProof (m n : ℝ) (h : m > n) : m / 4 > n / 4 :=
by sorry

end InequalityProof_l75_75681


namespace power_difference_divisible_l75_75155

-- Define the variables and conditions
variables {a b c : ℤ} {n : ℕ}

-- Condition: a - b is divisible by c
def is_divisible (a b c : ℤ) : Prop := ∃ k : ℤ, a - b = k * c

-- Lean proof statement
theorem power_difference_divisible {a b c : ℤ} {n : ℕ} (h : is_divisible a b c) : c ∣ (a^n - b^n) :=
  sorry

end power_difference_divisible_l75_75155


namespace range_of_a_l75_75847

theorem range_of_a (a : ℝ) (x y : ℝ) (hxy : x * y > 0) (hx : 0 < x) (hy : 0 < y) :
  (x + y) * (1 / x + a / y) ≥ 9 → a ≥ 4 :=
by
  intro h
  sorry

end range_of_a_l75_75847


namespace least_n_divisible_by_25_and_7_l75_75221

theorem least_n_divisible_by_25_and_7 (n : ℕ) (h1 : n > 1) (h2 : n % 25 = 1) (h3 : n % 7 = 1) : n = 126 :=
by
  sorry

end least_n_divisible_by_25_and_7_l75_75221


namespace abs_sum_neq_zero_iff_or_neq_zero_l75_75768

variable {x y : ℝ}

theorem abs_sum_neq_zero_iff_or_neq_zero (x y : ℝ) :
  (|x| + |y| ≠ 0) ↔ (x ≠ 0 ∨ y ≠ 0) := by
  sorry

end abs_sum_neq_zero_iff_or_neq_zero_l75_75768


namespace prob1_prob2_prob3_l75_75854

-- Define the sequences for rows ①, ②, and ③
def seq1 (n : ℕ) : ℤ := (-2) ^ n
def seq2 (m : ℕ) : ℤ := (-2) ^ (m - 1)
def seq3 (m : ℕ) : ℤ := (-2) ^ (m - 1) - 1

-- Prove the $n^{th}$ number in row ①
theorem prob1 (n : ℕ) : seq1 n = (-2) ^ n :=
by sorry

-- Prove the relationship between $m^{th}$ numbers in row ② and row ③
theorem prob2 (m : ℕ) : seq3 m = seq2 m - 1 :=
by sorry

-- Prove the value of $x + y + z$ where $x$, $y$, and $z$ are the $2019^{th}$ numbers in rows ①, ②, and ③, respectively
theorem prob3 : seq1 2019 + seq2 2019 + seq3 2019 = -1 :=
by sorry

end prob1_prob2_prob3_l75_75854


namespace proof_remove_terms_sum_is_one_l75_75282

noncomputable def remove_terms_sum_is_one : Prop :=
  let initial_sum := (1/2) + (1/4) + (1/6) + (1/8) + (1/10) + (1/12)
  let terms_to_remove := (1/8) + (1/10)
  initial_sum - terms_to_remove = 1

theorem proof_remove_terms_sum_is_one : remove_terms_sum_is_one :=
by
  -- proof will go here but is not required
  sorry

end proof_remove_terms_sum_is_one_l75_75282


namespace constant_is_5_variables_are_n_and_S_l75_75086

-- Define the conditions
def cost_per_box : ℕ := 5
def total_cost (n : ℕ) : ℕ := n * cost_per_box

-- Define the statement to be proved
-- constant is 5
theorem constant_is_5 : cost_per_box = 5 := 
by sorry

-- variables are n and S, where S is total_cost n
theorem variables_are_n_and_S (n : ℕ) : 
    ∃ S : ℕ, S = total_cost n :=
by sorry

end constant_is_5_variables_are_n_and_S_l75_75086


namespace fg_sqrt2_eq_neg5_l75_75523

noncomputable def f (x : ℝ) : ℝ := 4 - 3 * x
noncomputable def g (x : ℝ) : ℝ := x^2 + 1

theorem fg_sqrt2_eq_neg5 : f (g (Real.sqrt 2)) = -5 := by
  sorry

end fg_sqrt2_eq_neg5_l75_75523


namespace cubic_geometric_sequence_conditions_l75_75844

-- Conditions from the problem
def cubic_eq (a b c x : ℝ) : Prop := x^3 + a * x^2 + b * x + c = 0

-- The statement to be proven
theorem cubic_geometric_sequence_conditions (a b c : ℝ) :
  (∃ x q : ℝ, q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1 ∧ 
    cubic_eq a b c x ∧ cubic_eq a b c (x*q) ∧ cubic_eq a b c (x*q^2)) → 
  (b^3 = a^3 * c ∧ c ≠ 0 ∧ -a^3 < c ∧ c < a^3 / 27 ∧ a < m ∧ m < - a / 3) :=
by 
  sorry

end cubic_geometric_sequence_conditions_l75_75844


namespace det_scaled_matrix_l75_75178

variable {R : Type*} [CommRing R]

def det2x2 (a b c d : R) : R := a * d - b * c

theorem det_scaled_matrix 
  (x y z w : R) 
  (h : det2x2 x y z w = 3) : 
  det2x2 (3 * x) (3 * y) (6 * z) (6 * w) = 54 := by
  sorry

end det_scaled_matrix_l75_75178


namespace feeding_amount_per_horse_per_feeding_l75_75882

-- Define the conditions as constants
def num_horses : ℕ := 25
def feedings_per_day : ℕ := 2
def half_ton_in_pounds : ℕ := 1000
def bags_needed : ℕ := 60
def days : ℕ := 60

-- Statement of the problem
theorem feeding_amount_per_horse_per_feeding :
  (bags_needed * half_ton_in_pounds / days / feedings_per_day) / num_horses = 20 := by
  -- Assume conditions are satisfied
  sorry

end feeding_amount_per_horse_per_feeding_l75_75882


namespace range_of_sum_of_reciprocals_l75_75796

theorem range_of_sum_of_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) :
  ∃ (r : ℝ), ∀ t ∈ Set.Ici (3 + 2 * Real.sqrt 2), t = (1 / x + 1 / y) := 
sorry

end range_of_sum_of_reciprocals_l75_75796


namespace probability_of_observing_change_l75_75922

noncomputable def traffic_light_cycle := 45 + 5 + 45
noncomputable def observable_duration := 5 + 5 + 5
noncomputable def probability_observe_change := observable_duration / (traffic_light_cycle : ℝ)

theorem probability_of_observing_change :
  probability_observe_change = (3 / 19 : ℝ) :=
  by sorry

end probability_of_observing_change_l75_75922


namespace expression_value_l75_75136

theorem expression_value : 
  ∀ (x y z: ℤ), x = 2 ∧ y = -3 ∧ z = 1 → x^2 + y^2 - z^2 - 2*x*y = 24 := by
  sorry

end expression_value_l75_75136


namespace find_stream_speed_l75_75419

variable (boat_speed dist_downstream dist_upstream : ℝ)
variable (stream_speed : ℝ)

noncomputable def speed_of_stream (boat_speed dist_downstream dist_upstream : ℝ) : ℝ :=
  let t_downstream := dist_downstream / (boat_speed + stream_speed)
  let t_upstream := dist_upstream / (boat_speed - stream_speed)
  if t_downstream = t_upstream then stream_speed else 0

theorem find_stream_speed
  (h : speed_of_stream 20 26 14 stream_speed = stream_speed) :
  stream_speed = 6 :=
sorry

end find_stream_speed_l75_75419


namespace transistors_in_2010_l75_75639

theorem transistors_in_2010 
  (initial_transistors : ℕ) 
  (initial_year : ℕ) 
  (final_year : ℕ) 
  (doubling_period : ℕ)
  (initial_transistors_eq: initial_transistors = 500000)
  (initial_year_eq: initial_year = 1985)
  (final_year_eq: final_year = 2010)
  (doubling_period_eq : doubling_period = 2) :
  initial_transistors * 2^((final_year - initial_year) / doubling_period) = 2048000000 := 
by 
  -- the proof goes here
  sorry

end transistors_in_2010_l75_75639


namespace jogging_track_circumference_l75_75373

/-- 
Given:
- Deepak's speed = 20 km/hr
- His wife's speed = 12 km/hr
- They meet for the first time in 32 minutes

Then:
The circumference of the jogging track is 17.0667 km.
-/
theorem jogging_track_circumference (deepak_speed : ℝ) (wife_speed : ℝ) (meet_time : ℝ)
  (h1 : deepak_speed = 20)
  (h2 : wife_speed = 12)
  (h3 : meet_time = (32 / 60) ) : 
  ∃ circumference : ℝ, circumference = 17.0667 :=
by
  sorry

end jogging_track_circumference_l75_75373


namespace subtraction_proof_l75_75504

theorem subtraction_proof :
  2000000000000 - 1111111111111 - 222222222222 = 666666666667 :=
by sorry

end subtraction_proof_l75_75504


namespace rhombus_area_l75_75226

-- Definitions
def side_length := 25 -- cm
def diagonal1 := 30 -- cm

-- Statement to prove
theorem rhombus_area (s : ℝ) (d1 : ℝ) (d2 : ℝ) 
  (h_s : s = 25) 
  (h_d1 : d1 = 30)
  (h_side : s^2 = (d1/2)^2 + (d2/2)^2) :
  (d1 * d2) / 2 = 600 :=
by sorry

end rhombus_area_l75_75226


namespace compute_expression_l75_75783

theorem compute_expression (a b : ℕ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b^3) / (a^2 - 2*a*b + b^2 + a*b) = 5 :=
by
  have h : a = 3 := h1
  have k : b = 2 := h2
  rw [h, k]
  sorry

end compute_expression_l75_75783


namespace distance_A_to_B_is_64_yards_l75_75990

theorem distance_A_to_B_is_64_yards :
  let south1 := 50
  let west := 80
  let north := 20
  let east := 30
  let south2 := 10
  let net_south := south1 + south2 - north
  let net_west := west - east
  let distance := Real.sqrt ((net_south ^ 2) + (net_west ^ 2))
  distance = 64 :=
  by
  let south1 := 50
  let west := 80
  let north := 20
  let east := 30
  let south2 := 10
  let net_south := south1 + south2 - north
  let net_west := west - east
  let distance := Real.sqrt ((net_south ^ 2) + (net_west ^ 2))
  sorry

end distance_A_to_B_is_64_yards_l75_75990


namespace bananas_per_friend_l75_75792

-- Define constants and conditions
def totalBananas : Nat := 40
def totalFriends : Nat := 40

-- Define the main theorem to prove
theorem bananas_per_friend : totalBananas / totalFriends = 1 := by
  sorry

end bananas_per_friend_l75_75792


namespace black_to_white_ratio_l75_75565

/-- 
Given:
- The original square pattern consists of 13 black tiles and 23 white tiles
- Attaching a border of black tiles around the original 6x6 square pattern results in an 8x8 square pattern

To prove:
- The ratio of black tiles to white tiles in the extended 8x8 pattern is 41/23.
-/
theorem black_to_white_ratio (b_orig w_orig b_added b_total w_total : ℕ) 
  (h_black_orig: b_orig = 13)
  (h_white_orig: w_orig = 23)
  (h_size_orig: 6 * 6 = b_orig + w_orig)
  (h_size_ext: 8 * 8 = (b_orig + b_added) + w_orig)
  (h_b_added: b_added = 28)
  (h_b_total: b_total = b_orig + b_added)
  (h_w_total: w_total = w_orig)
  :
  b_total / w_total = 41 / 23 :=
by
  sorry

end black_to_white_ratio_l75_75565


namespace avg_salary_feb_mar_apr_may_l75_75910

def avg_salary_4_months : ℝ := 8000
def salary_jan : ℝ := 3700
def salary_may : ℝ := 6500
def total_salary_4_months := 4 * avg_salary_4_months
def total_salary_feb_mar_apr := total_salary_4_months - salary_jan
def total_salary_feb_mar_apr_may := total_salary_feb_mar_apr + salary_may

theorem avg_salary_feb_mar_apr_may : total_salary_feb_mar_apr_may / 4 = 8700 := by
  sorry

end avg_salary_feb_mar_apr_may_l75_75910


namespace residue_11_pow_1234_mod_19_l75_75470

theorem residue_11_pow_1234_mod_19 : 
  (11 ^ 1234) % 19 = 11 := 
by
  sorry

end residue_11_pow_1234_mod_19_l75_75470


namespace lilibeth_and_friends_strawberries_l75_75642

-- Define the conditions
def baskets_filled_by_lilibeth : ℕ := 6
def strawberries_per_basket : ℕ := 50
def friends_count : ℕ := 3

-- Define the total number of strawberries picked by Lilibeth and her friends 
def total_strawberries_picked : ℕ :=
  (baskets_filled_by_lilibeth * strawberries_per_basket) * (1 + friends_count)

-- The theorem to prove
theorem lilibeth_and_friends_strawberries : total_strawberries_picked = 1200 := 
by
  sorry

end lilibeth_and_friends_strawberries_l75_75642


namespace num_chairs_l75_75347

variable (C : Nat)
variable (tables_sticks : Nat := 6 * 9)
variable (stools_sticks : Nat := 4 * 2)
variable (total_sticks_needed : Nat := 34 * 5)
variable (total_sticks_chairs : Nat := 6 * C)

theorem num_chairs (h : total_sticks_chairs + tables_sticks + stools_sticks = total_sticks_needed) : C = 18 := 
by sorry

end num_chairs_l75_75347


namespace b_alone_completion_days_l75_75026

theorem b_alone_completion_days (Rab : ℝ) (w_12_days : (1 / (Rab + 4 * Rab)) = 12⁻¹) : 
    (1 / Rab = 60) :=
sorry

end b_alone_completion_days_l75_75026


namespace ratio_one_six_to_five_eighths_l75_75328

theorem ratio_one_six_to_five_eighths : (1 / 6) / (5 / 8) = 4 / 15 := by
  sorry

end ratio_one_six_to_five_eighths_l75_75328


namespace triangle_altitude_l75_75868

theorem triangle_altitude (A b : ℝ) (h : ℝ) 
  (hA : A = 750) 
  (hb : b = 50) 
  (area_formula : A = (1 / 2) * b * h) : 
  h = 30 :=
  sorry

end triangle_altitude_l75_75868


namespace dryer_sheets_per_load_l75_75739

theorem dryer_sheets_per_load (loads_per_week : ℕ) (cost_of_box : ℝ) (sheets_per_box : ℕ)
  (annual_savings : ℝ) (weeks_in_year : ℕ) (x : ℕ)
  (h1 : loads_per_week = 4)
  (h2 : cost_of_box = 5.50)
  (h3 : sheets_per_box = 104)
  (h4 : annual_savings = 11)
  (h5 : weeks_in_year = 52)
  (h6 : annual_savings = 2 * cost_of_box)
  (h7 : sheets_per_box * 2 = weeks_in_year * (loads_per_week * x)):
  x = 1 :=
by
  sorry

end dryer_sheets_per_load_l75_75739


namespace simplify_complex_expr_l75_75544

theorem simplify_complex_expr : 
  ∀ (i : ℂ), i^2 = -1 → ( (2 + 4 * i) / (2 - 4 * i) - (2 - 4 * i) / (2 + 4 * i) )
  = -8/5 + (16/5 : ℂ) * i :=
by
  intro i h_i_squared
  sorry

end simplify_complex_expr_l75_75544


namespace milk_mixture_l75_75590

theorem milk_mixture (x : ℝ) : 
  (2.4 + 0.1 * x) / (8 + x) = 0.2 → x = 8 :=
by
  sorry

end milk_mixture_l75_75590


namespace determine_a_l75_75478

theorem determine_a (a : ℕ)
  (h1 : 2 / (2 + 3 + a) = 1 / 3) : a = 1 :=
by
  sorry

end determine_a_l75_75478


namespace num_four_letter_initials_l75_75233

theorem num_four_letter_initials : 
  (10 : ℕ)^4 = 10000 := 
by 
  sorry

end num_four_letter_initials_l75_75233


namespace stream_speed_l75_75064

theorem stream_speed (v : ℝ) (t : ℝ) (h1 : t > 0)
  (h2 : ∃ k : ℝ, k = 2 * t)
  (h3 : (9 + v) * t = (9 - v) * (2 * t)) :
  v = 3 := 
sorry

end stream_speed_l75_75064


namespace correct_ordering_of_fractions_l75_75443

theorem correct_ordering_of_fractions :
  let a := (6 : ℚ) / 17
  let b := (8 : ℚ) / 25
  let c := (10 : ℚ) / 31
  let d := (1 : ℚ) / 3
  b < d ∧ d < c ∧ c < a :=
by
  sorry

end correct_ordering_of_fractions_l75_75443


namespace intersection_of_A_and_B_l75_75275

def A : Set ℝ := { x | x ≥ 0 }
def B : Set ℝ := { x | -1 ≤ x ∧ x < 2 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | 0 ≤ x ∧ x < 2 } := 
by
  sorry

end intersection_of_A_and_B_l75_75275


namespace cube_positive_integers_solution_l75_75077

theorem cube_positive_integers_solution (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) :
  (∃ k : ℕ, 2^(Nat.factorial a) + 2^(Nat.factorial b) + 2^(Nat.factorial c) = k^3) ↔ 
    ( (a = 1 ∧ b = 1 ∧ c = 2) ∨ 
      (a = 1 ∧ b = 2 ∧ c = 1) ∨ 
      (a = 2 ∧ b = 1 ∧ c = 1) ) :=
by
  sorry

end cube_positive_integers_solution_l75_75077


namespace part_a_part_b_part_c_l75_75020

-- Definitions for the convex polyhedron, volume, and surface area
structure ConvexPolyhedron :=
  (volume : ℝ)
  (surface_area : ℝ)

variable {P : ConvexPolyhedron}

-- Statement for Part (a)
theorem part_a (r : ℝ) (h_r : r ≤ P.surface_area) :
  P.volume / P.surface_area ≥ r / 3 := sorry

-- Statement for Part (b)
theorem part_b :
  Exists (fun r : ℝ => r = P.volume / P.surface_area) := sorry

-- Definitions and conditions for the outer and inner polyhedron
structure ConvexPolyhedronPair :=
  (outer_polyhedron : ConvexPolyhedron)
  (inner_polyhedron : ConvexPolyhedron)

variable {CP : ConvexPolyhedronPair}

-- Statement for Part (c)
theorem part_c :
  3 * CP.outer_polyhedron.volume / CP.outer_polyhedron.surface_area ≥
  CP.inner_polyhedron.volume / CP.inner_polyhedron.surface_area := sorry

end part_a_part_b_part_c_l75_75020


namespace area_enclosed_is_one_third_l75_75991

theorem area_enclosed_is_one_third :
  ∫ x in (0:ℝ)..1, (x^(1/2) - x^2 : ℝ) = (1/3 : ℝ) :=
by
  sorry

end area_enclosed_is_one_third_l75_75991


namespace smallest_nat_divisible_by_225_l75_75697

def has_digits_0_or_1 (n : ℕ) : Prop := 
  ∀ (d : ℕ), d ∈ n.digits 10 → d = 0 ∨ d = 1

def divisible_by_225 (n : ℕ) : Prop := 225 ∣ n

theorem smallest_nat_divisible_by_225 :
  ∃ (n : ℕ), has_digits_0_or_1 n ∧ divisible_by_225 n 
    ∧ ∀ (m : ℕ), has_digits_0_or_1 m ∧ divisible_by_225 m → n ≤ m 
    ∧ n = 11111111100 := 
  sorry

end smallest_nat_divisible_by_225_l75_75697


namespace correct_operation_l75_75577

theorem correct_operation (a : ℝ) : a^8 / a^2 = a^6 :=
by
  -- proof will go here, let's use sorry to indicate it's unfinished
  sorry

end correct_operation_l75_75577


namespace find_number_l75_75654

variable {x : ℝ}

theorem find_number (h : (30 / 100) * x = (40 / 100) * 40) : x = 160 / 3 :=
by
  sorry

end find_number_l75_75654


namespace average_eq_one_half_l75_75106

variable (w x y : ℝ)

-- Conditions
variables (h1 : 2 / w + 2 / x = 2 / y)
variables (h2 : w * x = y)

theorem average_eq_one_half : (w + x) / 2 = 1 / 2 :=
by
  sorry

end average_eq_one_half_l75_75106


namespace parabola_translation_l75_75780

theorem parabola_translation :
  ∀ (x y : ℝ), (y = 2 * (x - 3) ^ 2) ↔ ∃ t : ℝ, t = x - 3 ∧ y = 2 * t ^ 2 :=
by sorry

end parabola_translation_l75_75780


namespace items_counted_l75_75194

def convert_counter (n : Nat) : Nat := sorry

theorem items_counted
  (counter_reading : Nat) 
  (condition_1 : ∀ d, d ∈ [5, 6, 7] → ¬(d ∈ [0, 1, 2, 3, 4, 8, 9]))
  (condition_2 : ∀ d1 d2, d1 = 4 → d2 = 8 → ¬(d2 = 5 ∨ d2 = 6 ∨ d2 = 7)) :
  convert_counter 388 = 151 :=
sorry

end items_counted_l75_75194


namespace number_is_square_plus_opposite_l75_75088

theorem number_is_square_plus_opposite (x : ℝ) (hx : x = x^2 + -x) : x = 0 ∨ x = 2 :=
by sorry

end number_is_square_plus_opposite_l75_75088


namespace problem1_sin_cos_problem2_linear_combination_l75_75257

/-- Problem 1: Prove that sin(α) * cos(α) = -2/5 given that the terminal side of angle α passes through (-1, 2) --/
theorem problem1_sin_cos (α : ℝ) (x y : ℝ) (h1 : x = -1) (h2 : y = 2) (h3 : Real.sqrt (x^2 + y^2) ≠ 0) :
  Real.sin α * Real.cos α = -2 / 5 :=
by
  sorry

/-- Problem 2: Prove that 10sin(α) + 3cos(α) = 0 given that the terminal side of angle α lies on the line y = -3x --/
theorem problem2_linear_combination (α : ℝ) (x y : ℝ) (h1 : y = -3 * x) (h2 : (x = -1 ∧ y = 3) ∨ (x = 1 ∧ y = -3)) (h3 : Real.sqrt (x^2 + y^2) ≠ 0) :
  10 * Real.sin α + 3 / Real.cos α = 0 :=
by
  sorry

end problem1_sin_cos_problem2_linear_combination_l75_75257


namespace record_loss_of_300_l75_75853

-- Definitions based on conditions
def profit (x : Int) : String := "+" ++ toString x
def loss (x : Int) : String := "-" ++ toString x

-- The theorem to prove that a loss of 300 is recorded as "-300" based on the recording system
theorem record_loss_of_300 : loss 300 = "-300" :=
by
  sorry

end record_loss_of_300_l75_75853


namespace certain_number_is_60_l75_75490

theorem certain_number_is_60 
  (A J C : ℕ) 
  (h1 : A = 4) 
  (h2 : C = 8) 
  (h3 : A = (1 / 2) * J) :
  3 * (A + J + C) = 60 :=
by sorry

end certain_number_is_60_l75_75490


namespace negation_of_forall_exp_gt_zero_l75_75334

open Real

theorem negation_of_forall_exp_gt_zero : 
  (¬ (∀ x : ℝ, exp x > 0)) ↔ (∃ x : ℝ, exp x ≤ 0) :=
by
  sorry

end negation_of_forall_exp_gt_zero_l75_75334


namespace ordering_of_four_numbers_l75_75466

variable (m n α β : ℝ)
variable (h1 : m < n)
variable (h2 : α < β)
variable (h3 : 2 * (α - m) * (α - n) - 7 = 0)
variable (h4 : 2 * (β - m) * (β - n) - 7 = 0)

theorem ordering_of_four_numbers : α < m ∧ m < n ∧ n < β :=
by
  sorry

end ordering_of_four_numbers_l75_75466


namespace number_of_teams_l75_75655

theorem number_of_teams (x : ℕ) (h : (x * (x - 1)) / 2 = 28) : x = 8 :=
sorry

end number_of_teams_l75_75655


namespace stacked_cubes_surface_area_is_945_l75_75039

def volumes : List ℕ := [512, 343, 216, 125, 64, 27, 8, 1]

def side_length (v : ℕ) : ℕ := v^(1/3)

def num_visible_faces (i : ℕ) : ℕ :=
  if i == 0 then 5 else 3 -- Bottom cube has 5 faces visible, others have 3 due to rotation

def surface_area (s : ℕ) (faces : ℕ) : ℕ :=
  faces * s^2

def total_surface_area (volumes : List ℕ) : ℕ :=
  (volumes.zipWith surface_area (volumes.enum.map (λ (i, v) => num_visible_faces i))).sum

theorem stacked_cubes_surface_area_is_945 :
  total_surface_area volumes = 945 := 
by 
  sorry

end stacked_cubes_surface_area_is_945_l75_75039


namespace power_of_54_l75_75937

theorem power_of_54 (a b : ℕ) (h_a_pos : a > 0) (h_b_pos : b > 0) 
(h_eq : 54^a = a^b) : ∃ k : ℕ, a = 54^k := by
  sorry

end power_of_54_l75_75937


namespace solution_set_of_inequality_l75_75248

theorem solution_set_of_inequality :
  { x : ℝ | 3 ≤ |2 * x - 5| ∧ |2 * x - 5| < 9 } = { x : ℝ | (-2 < x ∧ x ≤ 1) ∨ (4 ≤ x ∧ x < 7) } :=
by 
  -- Conditions and steps omitted for the sake of the statement.
  sorry

end solution_set_of_inequality_l75_75248


namespace general_term_sum_first_n_terms_l75_75893

variable {a : ℕ → ℝ} {b : ℕ → ℝ} {S : ℕ → ℝ}
variable (d : ℝ) (h1 : d ≠ 0)
variable (a10 : a 10 = 19)
variable (geo_seq : ∀ {x y z}, x * z = y ^ 2 → x = 1 → y = a 2 → z = a 5)
variable (arith_seq : ∀ n, a n = a 1 + (n - 1) * d)

-- General term of the arithmetic sequence
theorem general_term (a_1 : ℝ) (h1 : a 1 = a_1) : a n = 2 * n - 1 :=
sorry

-- Sum of the first n terms of the sequence b_n
theorem sum_first_n_terms (n : ℕ) : S n = (2 * n - 3) * 2^(n + 1) + 6 :=
sorry

end general_term_sum_first_n_terms_l75_75893


namespace paul_lost_crayons_l75_75313

theorem paul_lost_crayons :
  ∀ (initial_crayons given_crayons left_crayons lost_crayons : ℕ),
    initial_crayons = 1453 →
    given_crayons = 563 →
    left_crayons = 332 →
    lost_crayons = (initial_crayons - given_crayons) - left_crayons →
    lost_crayons = 558 :=
by
  intros initial_crayons given_crayons left_crayons lost_crayons
  intros h_initial h_given h_left h_lost
  sorry

end paul_lost_crayons_l75_75313


namespace sheila_paintings_l75_75296

theorem sheila_paintings (a b : ℕ) (h1 : a = 9) (h2 : b = 9) : a + b = 18 :=
by
  sorry

end sheila_paintings_l75_75296


namespace find_vector_at_6_l75_75657

structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vec_add (v1 v2 : Vector3D) : Vector3D :=
  { x := v1.x + v2.x, y := v1.y + v2.y, z := v1.z + v2.z }

def vec_scale (c : ℝ) (v : Vector3D) : Vector3D :=
  { x := c * v.x, y := c * v.y, z := c * v.z }

noncomputable def vector_at_t (a d : Vector3D) (t : ℝ) : Vector3D :=
  vec_add a (vec_scale t d)

theorem find_vector_at_6 :
  let a := { x := 2, y := -1, z := 3 }
  let d := { x := 1, y := 2, z := -1 }
  vector_at_t a d 6 = { x := 8, y := 11, z := -3 } :=
by
  sorry

end find_vector_at_6_l75_75657


namespace ducks_drinking_l75_75988

theorem ducks_drinking (total_d : ℕ) (drank_before : ℕ) (drank_after : ℕ) :
  total_d = 20 → drank_before = 11 → drank_after = total_d - (drank_before + 1) → drank_after = 8 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end ducks_drinking_l75_75988


namespace pipes_fill_time_l75_75730

noncomputable def filling_time (P X Y Z : ℝ) : ℝ :=
  P / (X + Y + Z)

theorem pipes_fill_time (P : ℝ) (X Y Z : ℝ)
  (h1 : X + Y = P / 3) 
  (h2 : X + Z = P / 6) 
  (h3 : Y + Z = P / 4.5) :
  filling_time P X Y Z = 36 / 13 := by
  sorry

end pipes_fill_time_l75_75730


namespace angle_between_lines_in_folded_rectangle_l75_75575

theorem angle_between_lines_in_folded_rectangle
  (a b : ℝ) 
  (h : b > a)
  (dihedral_angle : ℝ)
  (h_dihedral_angle : dihedral_angle = 18) :
  ∃ (angle_AC_MN : ℝ), angle_AC_MN = 90 :=
by
  sorry

end angle_between_lines_in_folded_rectangle_l75_75575


namespace square_side_length_l75_75830

theorem square_side_length (length_rect width_rect : ℕ) (h_length : length_rect = 400) (h_width : width_rect = 300)
  (h_perimeter : 4 * side_length = 2 * (2 * (length_rect + width_rect))) : side_length = 700 := by
  -- Proof goes here
  sorry

end square_side_length_l75_75830


namespace monotonically_increasing_l75_75009

variable {R : Type} [LinearOrderedField R]

def f (x : R) : R := 3 * x + 1

theorem monotonically_increasing : ∀ x₁ x₂ : R, x₁ < x₂ → f x₁ < f x₂ :=
by
  intro x₁ x₂ h
 -- this is where the proof would go
  sorry

end monotonically_increasing_l75_75009


namespace sqrt_fraction_simplification_l75_75406

theorem sqrt_fraction_simplification :
  (Real.sqrt ((25 / 49) - (16 / 81)) = (Real.sqrt 1241) / 63) := by
  sorry

end sqrt_fraction_simplification_l75_75406


namespace domain_of_sqrt_cosine_sub_half_l75_75237

theorem domain_of_sqrt_cosine_sub_half :
  {x : ℝ | ∃ k : ℤ, (2 * k * π - π / 3) ≤ x ∧ x ≤ (2 * k * π + π / 3)} =
  {x : ℝ | ∃ k : ℤ, 2 * k * π - π / 3 ≤ x ∧ x ≤ 2 * k * π + π / 3} :=
by sorry

end domain_of_sqrt_cosine_sub_half_l75_75237


namespace Matilda_correct_age_l75_75519

def Louis_age : ℕ := 14
def Jerica_age : ℕ := 2 * Louis_age
def Matilda_age : ℕ := Jerica_age + 7

theorem Matilda_correct_age : Matilda_age = 35 :=
by
  -- Proof needs to be filled here
  sorry

end Matilda_correct_age_l75_75519


namespace parabola_y1_gt_y2_l75_75746

variable {x1 x2 y1 y2 : ℝ}

theorem parabola_y1_gt_y2 
  (hx1 : -4 < x1 ∧ x1 < -2) 
  (hx2 : 0 < x2 ∧ x2 < 2) 
  (hy1 : y1 = x1^2) 
  (hy2 : y2 = x2^2) : 
  y1 > y2 :=
by 
  sorry

end parabola_y1_gt_y2_l75_75746


namespace M_values_l75_75446

theorem M_values (m n p M : ℝ) (h1 : M = m / (n + p)) (h2 : M = n / (p + m)) (h3 : M = p / (m + n)) :
  M = 1 / 2 ∨ M = -1 :=
by
  sorry

end M_values_l75_75446


namespace x_axis_intercept_of_line_l75_75977

theorem x_axis_intercept_of_line (x : ℝ) : (∃ x, 2*x + 1 = 0) → x = - 1 / 2 :=
  by
    intro h
    obtain ⟨x, h1⟩ := h
    have : 2 * x + 1 = 0 := h1
    linarith [this]

end x_axis_intercept_of_line_l75_75977


namespace line_in_slope_intercept_form_l75_75926

variable (x y : ℝ)

def line_eq (x y : ℝ) : Prop :=
  (3 : ℝ) * (x - 2) - (4 : ℝ) * (y + 1) = 0

theorem line_in_slope_intercept_form (x y : ℝ) (h: line_eq x y) :
  y = (3 / 4) * x - 5 / 2 :=
sorry

end line_in_slope_intercept_form_l75_75926


namespace last_four_digits_of_7_pow_5000_l75_75036

theorem last_four_digits_of_7_pow_5000 (h : 7 ^ 250 ≡ 1 [MOD 1250]) : 7 ^ 5000 ≡ 1 [MOD 1250] :=
by
  -- Proof (will be omitted)
  sorry

end last_four_digits_of_7_pow_5000_l75_75036


namespace initial_water_percentage_l75_75383

variable (W : ℝ) -- Initial percentage of water in the milk

theorem initial_water_percentage 
  (final_water_content : ℝ := 2) 
  (pure_milk_added : ℝ := 15) 
  (initial_milk_volume : ℝ := 10)
  (final_mixture_volume : ℝ := initial_milk_volume + pure_milk_added)
  (water_equation : W / 100 * initial_milk_volume = final_water_content / 100 * final_mixture_volume) 
  : W = 5 :=
by
  sorry

end initial_water_percentage_l75_75383


namespace problem_a_problem_b_l75_75718

-- Problem a conditions and statement
def digit1a : Nat := 1
def digit2a : Nat := 4
def digit3a : Nat := 2
def digit4a : Nat := 8
def digit5a : Nat := 5

theorem problem_a : (digit1a * 100000 + digit2a * 10000 + digit3a * 1000 + digit4a * 100 + digit5a * 10 + 7) * 5 = 
                    7 * (digit1a * 100000 + digit2a * 10000 + digit3a * 1000 + digit4a * 100 + digit5a * 10 + 285) := by
  sorry

-- Problem b conditions and statement
def digit1b : Nat := 4
def digit2b : Nat := 2
def digit3b : Nat := 8
def digit4b : Nat := 5
def digit5b : Nat := 7

theorem problem_b : (1 * 100000 + digit1b * 10000 + digit2b * 1000 + digit3b * 100 + digit4b * 10 + digit5b) * 3 = 
                    (digit1b * 100000 + digit2b * 10000 + digit3b * 1000 + digit4b * 100 + digit5b * 10 + 1) := by
  sorry

end problem_a_problem_b_l75_75718


namespace remainder_of_sum_mod_13_l75_75542

theorem remainder_of_sum_mod_13 :
  ∀ (D : ℕ) (k1 k2 : ℕ),
    D = 13 →
    (242 = k1 * D + 8) →
    (698 = k2 * D + 9) →
    (242 + 698) % D = 4 :=
by
  intros D k1 k2 hD h242 h698
  sorry

end remainder_of_sum_mod_13_l75_75542


namespace positive_difference_perimeters_l75_75413

def perimeter_rectangle (length : ℕ) (width : ℕ) : ℕ :=
  2 * (length + width)

def perimeter_cross_shape : ℕ := 
  let top_and_bottom := 3 + 3 -- top and bottom edges
  let left_and_right := 3 + 3 -- left and right edges
  let internal_subtraction := 4
  top_and_bottom + left_and_right - internal_subtraction

theorem positive_difference_perimeters :
  let length := 4
  let width := 3
  perimeter_rectangle length width - perimeter_cross_shape = 6 :=
by
  let length := 4
  let width := 3
  sorry

end positive_difference_perimeters_l75_75413


namespace part1_part2_l75_75923

theorem part1 (a b : ℝ) (h1 : a - b = 4) (h2 : a * b = 3) : a^2 + b^2 = 22 :=
sorry

theorem part2 (a b : ℝ) (h1 : a - b = 4) (h2 : a * b = 3) : (a - 2) * (b + 2) = 7 :=
sorry

end part1_part2_l75_75923


namespace people_on_bus_now_l75_75831

variable (x : ℕ)

def original_people_on_bus : ℕ := 38
def people_got_on_bus (x : ℕ) : ℕ := x
def people_left_bus (x : ℕ) : ℕ := x + 9

theorem people_on_bus_now (x : ℕ) : original_people_on_bus - people_left_bus x + people_got_on_bus x = 29 := 
by
  sorry

end people_on_bus_now_l75_75831


namespace ratio_w_y_l75_75931

open Real

theorem ratio_w_y (w x y z : ℝ) (h1 : w / x = 5 / 2) (h2 : y / z = 3 / 2) (h3 : z / x = 1 / 4) : w / y = 20 / 3 :=
by
  sorry

end ratio_w_y_l75_75931


namespace binomial_sum_of_coefficients_l75_75445

-- Given condition: for the third term in the expansion, the binomial coefficient is 15
def binomial_coefficient_condition (n : ℕ) := Nat.choose n 2 = 15

-- The goal: the sum of the coefficients of all terms in the expansion is 1/64
theorem binomial_sum_of_coefficients (n : ℕ) (h : binomial_coefficient_condition n) :
  (1:ℚ) / (2 : ℚ)^6 = 1 / 64 :=
by 
  have h₁ : n = 6 := by sorry -- Solve for n using the given condition.
  sorry -- Prove the sum of coefficients when x is 1.

end binomial_sum_of_coefficients_l75_75445


namespace cash_price_of_television_l75_75905

variable (DownPayment : ℕ := 120)
variable (MonthlyPayment : ℕ := 30)
variable (NumberOfMonths : ℕ := 12)
variable (Savings : ℕ := 80)

-- Define the total installment cost
def TotalInstallment := DownPayment + MonthlyPayment * NumberOfMonths

-- The main statement to prove
theorem cash_price_of_television : (TotalInstallment - Savings) = 400 := by
  sorry

end cash_price_of_television_l75_75905


namespace constant_function_solution_l75_75491

theorem constant_function_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, 2 * f x = f (x + y) + f (x + 2 * y)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c := by
  sorry

end constant_function_solution_l75_75491


namespace min_sum_of_factors_l75_75541

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 1176) :
  a + b + c ≥ 59 :=
sorry

end min_sum_of_factors_l75_75541


namespace optimal_selling_price_l75_75606

-- Define the constants given in the problem
def purchase_price : ℝ := 40
def initial_selling_price : ℝ := 50
def initial_sales_volume : ℝ := 50

-- Define the function that represents the profit based on the change in price x
def profit (x : ℝ) : ℝ := (initial_selling_price + x) * (initial_sales_volume - x) - (initial_sales_volume - x) * purchase_price

-- State the theorem
theorem optimal_selling_price : ∃ x : ℝ, profit x = -x^2 + 40*x + 500 ∧ (initial_selling_price + x = 70) :=
by
  sorry

end optimal_selling_price_l75_75606


namespace janous_inequality_l75_75360

theorem janous_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
sorry

end janous_inequality_l75_75360


namespace gcd_three_digit_palindromes_l75_75703

theorem gcd_three_digit_palindromes : ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) → Nat.gcd (102 * a + 10 * b) 1 = 1 :=
by
  intros a b h
  sorry

end gcd_three_digit_palindromes_l75_75703


namespace charlie_original_price_l75_75593

theorem charlie_original_price (acorns_Alice acorns_Bob acorns_Charlie ν_Alice ν_Bob discount price_Charlie_before_discount price_Charlie_after_discount total_paid_by_AliceBob total_acorns_AliceBob average_price_per_acorn price_per_acorn_Alice price_per_acorn_Bob total_paid_Alice total_paid_Bob: ℝ) :
  acorns_Alice = 3600 →
  acorns_Bob = 2400 →
  acorns_Charlie = 4500 →
  ν_Bob = 6000 →
  ν_Alice = 9 * ν_Bob →
  price_per_acorn_Bob = ν_Bob / acorns_Bob →
  price_per_acorn_Alice = ν_Alice / acorns_Alice →
  total_paid_Alice = acorns_Alice * price_per_acorn_Alice →
  total_paid_Bob = ν_Bob →
  total_paid_by_AliceBob = total_paid_Alice + total_paid_Bob →
  total_acorns_AliceBob = acorns_Alice + acorns_Bob →
  average_price_per_acorn = total_paid_by_AliceBob / total_acorns_AliceBob →
  discount = 10 / 100 →
  price_Charlie_after_discount = average_price_per_acorn * (1 - discount) →
  price_Charlie_before_discount = average_price_per_acorn →
  price_Charlie_before_discount = 14.50 →
  price_per_acorn_Alice = 22.50 →
  price_Charlie_before_discount * acorns_Charlie = 4500 * 14.50 :=
by sorry

end charlie_original_price_l75_75593


namespace percentage_reduced_l75_75280

theorem percentage_reduced (P : ℝ) (h : (85 * P / 100) - 11 = 23) : P = 40 :=
by 
  sorry

end percentage_reduced_l75_75280


namespace remainder_sum_mod_53_l75_75369

theorem remainder_sum_mod_53 (a b c d : ℕ)
  (h1 : a % 53 = 31)
  (h2 : b % 53 = 45)
  (h3 : c % 53 = 17)
  (h4 : d % 53 = 6) :
  (a + b + c + d) % 53 = 46 := 
sorry

end remainder_sum_mod_53_l75_75369


namespace number_of_elements_l75_75530

theorem number_of_elements
  (init_avg : ℕ → ℝ)
  (correct_avg : ℕ → ℝ)
  (incorrect_num correct_num : ℝ)
  (h1 : ∀ n : ℕ, init_avg n = 17)
  (h2 : ∀ n : ℕ, correct_avg n = 20)
  (h3 : incorrect_num = 26)
  (h4 : correct_num = 56)
  : ∃ n : ℕ, n = 10 := sorry

end number_of_elements_l75_75530


namespace yoongi_hoseok_age_sum_l75_75623

-- Definitions of given conditions
def age_aunt : ℕ := 38
def diff_aunt_yoongi : ℕ := 23
def diff_yoongi_hoseok : ℕ := 4

-- Definitions related to ages of Yoongi and Hoseok derived from given conditions
def age_yoongi : ℕ := age_aunt - diff_aunt_yoongi
def age_hoseok : ℕ := age_yoongi - diff_yoongi_hoseok

-- The theorem we need to prove
theorem yoongi_hoseok_age_sum : age_yoongi + age_hoseok = 26 := by
  sorry

end yoongi_hoseok_age_sum_l75_75623


namespace incorrect_statement_about_zero_l75_75950

theorem incorrect_statement_about_zero :
  ¬ (0 > 0) :=
by
  sorry

end incorrect_statement_about_zero_l75_75950


namespace probability_of_odd_sum_l75_75013

theorem probability_of_odd_sum (P : ℝ → Prop) 
    (P_even_sum : ℝ)
    (P_odd_sum : ℝ)
    (h1 : P_even_sum = 2 * P_odd_sum) 
    (h2 : P_even_sum + P_odd_sum = 1) :
    P_odd_sum = 4/9 := 
sorry

end probability_of_odd_sum_l75_75013


namespace frisbee_total_distance_correct_l75_75713

-- Define the conditions
def bess_distance_per_throw : ℕ := 20 * 2 -- 20 meters out and 20 meters back = 40 meters
def bess_number_of_throws : ℕ := 4
def holly_distance_per_throw : ℕ := 8
def holly_number_of_throws : ℕ := 5

-- Calculate total distances
def bess_total_distance : ℕ := bess_distance_per_throw * bess_number_of_throws
def holly_total_distance : ℕ := holly_distance_per_throw * holly_number_of_throws
def total_distance : ℕ := bess_total_distance + holly_total_distance

-- The proof statement
theorem frisbee_total_distance_correct :
  total_distance = 200 :=
by
  -- proof goes here (we use sorry to skip the proof)
  sorry

end frisbee_total_distance_correct_l75_75713


namespace head_start_l75_75841

theorem head_start (V_b : ℝ) (S : ℝ) : 
  ((7 / 4) * V_b) = V_b → 
  196 = (196 - S) → 
  S = 84 := 
sorry

end head_start_l75_75841


namespace rectangular_prism_diagonals_l75_75301

theorem rectangular_prism_diagonals
  (num_vertices : ℕ) (num_edges : ℕ)
  (h1 : num_vertices = 12) (h2 : num_edges = 18) :
  (total_diagonals : ℕ) → total_diagonals = 20 :=
by
  sorry

end rectangular_prism_diagonals_l75_75301


namespace find_f_2012_l75_75407

variable (f : ℕ → ℝ)

axiom f_one : f 1 = 3997
axiom recurrence : ∀ x, f x - f (x + 1) = 1

theorem find_f_2012 : f 2012 = 1986 :=
by
  -- Skipping proof
  sorry

end find_f_2012_l75_75407


namespace problem_dividing_remainder_l75_75804

-- The conditions exported to Lean
def tiling_count (n : ℕ) : ℕ :=
  -- This function counts the number of valid tilings for a board size n with all colors used
  sorry

def remainder_when_divide (num divisor : ℕ) : ℕ := num % divisor

-- The statement problem we need to prove
theorem problem_dividing_remainder :
  remainder_when_divide (tiling_count 9) 1000 = 545 := 
sorry

end problem_dividing_remainder_l75_75804


namespace factorization_of_square_difference_l75_75255

variable (t : ℝ)

theorem factorization_of_square_difference : t^2 - 144 = (t - 12) * (t + 12) := 
sorry

end factorization_of_square_difference_l75_75255


namespace neg_sqrt_17_estimate_l75_75050

theorem neg_sqrt_17_estimate : -5 < -Real.sqrt 17 ∧ -Real.sqrt 17 < -4 := by
  sorry

end neg_sqrt_17_estimate_l75_75050


namespace Tina_profit_l75_75870

variables (x : ℝ) (profit_per_book : ℝ) (number_of_people : ℕ) (cost_per_book : ℝ)
           (books_per_customer : ℕ) (total_profit : ℝ) (total_cost : ℝ) (total_books_sold : ℕ)

theorem Tina_profit :
  (number_of_people = 4) →
  (cost_per_book = 5) →
  (books_per_customer = 2) →
  (total_profit = 120) →
  (books_per_customer * number_of_people = total_books_sold) →
  (cost_per_book * total_books_sold = total_cost) →
  (total_profit = total_books_sold * x - total_cost) →
  x = 20 :=
by
  intros
  sorry


end Tina_profit_l75_75870


namespace intersection_eq_l75_75271

def M : Set ℝ := { x | -1 < x ∧ x < 3 }
def N : Set ℝ := { x | -2 < x ∧ x < 1 }

theorem intersection_eq : M ∩ N = { x | -1 < x ∧ x < 1 } :=
by
  sorry

end intersection_eq_l75_75271


namespace collinear_points_min_value_l75_75689

open Real

/-- Let \(\overrightarrow{e_{1}}\) and \(\overrightarrow{e_{2}}\) be two non-collinear vectors in a plane,
    \(\overrightarrow{AB} = (a-1) \overrightarrow{e_{1}} + \overrightarrow{e_{2}}\),
    \(\overrightarrow{AC} = b \overrightarrow{e_{1}} - 2 \overrightarrow{e_{2}}\),
    with \(a > 0\) and \(b > 0\). 
    If points \(A\), \(B\), and \(C\) are collinear, then the minimum value of \(\frac{1}{a} + \frac{2}{b}\) is \(4\). -/
theorem collinear_points_min_value 
  (e1 e2 : ℝ) 
  (H_non_collinear : (e1 ≠ 0 ∨ e2 ≠ 0))
  (a b : ℝ) 
  (H_a_pos : a > 0) 
  (H_b_pos : b > 0)
  (H_collinear : ∃ x : ℝ, (a - 1) * e1 + e2 = x * (b * e1 - 2 * e2)) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + (1/2) * b = 1 ∧ (∀ a b : ℝ, (1/a) + (2/b) ≥ 4) :=
sorry

end collinear_points_min_value_l75_75689


namespace prob_of_two_reds_is_7_over_20_expected_value_is_2_19_l75_75073

section card_draws

/-- A card draw experiment with 10 cards: 5 red, 3 white, 2 blue. --/
inductive CardColor
| red
| white
| blue

def bag : List CardColor := List.replicate 5 CardColor.red ++ List.replicate 3 CardColor.white ++ List.replicate 2 CardColor.blue

/-- Probability of drawing exactly 2 red cards in up to 3 draws with the given conditions. --/
def prob_two_reds : ℚ :=
  (5 / 10) * (5 / 10) + 
  (5 / 10) * (2 / 10) * (5 / 10) + 
  (2 / 10) * (5 / 10) * (5 / 10)

theorem prob_of_two_reds_is_7_over_20 : prob_two_reds = 7 / 20 :=
  sorry

/-- Probability distribution of the number of draws necessary. --/
def prob_ξ_1 : ℚ := 3 / 10
def prob_ξ_2 : ℚ := 21 / 100
def prob_ξ_3 : ℚ := 49 / 100
def expected_value_ξ : ℚ :=
  1 * prob_ξ_1 + 2 * prob_ξ_2 + 3 * prob_ξ_3

theorem expected_value_is_2_19 : expected_value_ξ = 219 / 100 :=
  sorry

end card_draws

end prob_of_two_reds_is_7_over_20_expected_value_is_2_19_l75_75073


namespace proportional_segments_l75_75516

-- Define the problem
theorem proportional_segments :
  ∀ (a b c d : ℕ), a = 6 → b = 9 → c = 12 → (a * d = b * c) → d = 18 :=
by
  intros a b c d ha hb hc hrat
  rw [ha, hb, hc] at hrat
  exact sorry

end proportional_segments_l75_75516


namespace number_of_participants_l75_75384

theorem number_of_participants (n : ℕ) (h : n * (n - 1) / 2 = 171) : n = 19 :=
by
  sorry

end number_of_participants_l75_75384


namespace solve_for_b_l75_75762

theorem solve_for_b (b x : ℚ)
  (h₁ : 3 * x + 5 = 1)
  (h₂ : b * x + 6 = 0) :
  b = 9 / 2 :=
sorry   -- The proof is omitted as per instruction.

end solve_for_b_l75_75762


namespace smallest_base10_integer_l75_75932

theorem smallest_base10_integer :
  ∃ (n A B : ℕ), 
    (A < 5) ∧ (B < 7) ∧ 
    (n = 6 * A) ∧ 
    (n = 8 * B) ∧ 
    n = 24 := 
sorry

end smallest_base10_integer_l75_75932


namespace rationalize_denominator_l75_75273

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end rationalize_denominator_l75_75273


namespace geometric_sequence_sum_S8_l75_75487

noncomputable def sum_geometric_seq (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_S8 (a q : ℝ) (h1 : q ≠ 1)
  (h2 : sum_geometric_seq a q 4 = -5)
  (h3 : sum_geometric_seq a q 6 = 21 * sum_geometric_seq a q 2) :
  sum_geometric_seq a q 8 = -85 :=
sorry

end geometric_sequence_sum_S8_l75_75487


namespace R_and_D_calculation_l75_75836

-- Define the given conditions and required calculation
def R_and_D_t : ℝ := 2640.92
def delta_APL_t_plus_1 : ℝ := 0.12

theorem R_and_D_calculation :
  (R_and_D_t / delta_APL_t_plus_1) = 22008 := by sorry

end R_and_D_calculation_l75_75836


namespace fixed_line_of_midpoint_l75_75596

theorem fixed_line_of_midpoint
  (A B : ℝ × ℝ)
  (H : ∀ (P : ℝ × ℝ), (P = A ∨ P = B) → (P.1^2 / 3 - P.2^2 / 6 = 1))
  (slope_l : (B.2 - A.2) / (B.1 - A.1) = 2)
  (midpoint_lies : (A.1 + B.1) / 2 = (A.2 + B.2) / 2) :
  ∀ (M : ℝ × ℝ), (M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2) → M.1 - M.2 = 0 :=
by
  sorry

end fixed_line_of_midpoint_l75_75596


namespace number_of_rectangular_arrays_l75_75223

theorem number_of_rectangular_arrays (n : ℕ) (h : n = 48) : 
  ∃ k : ℕ, (k = 6 ∧ ∀ m p : ℕ, m * p = n → m ≥ 3 → p ≥ 3 → m = 3 ∨ m = 4 ∨ m = 6 ∨ m = 8 ∨ m = 12 ∨ m = 16 ∨ m = 24) :=
by
  sorry

end number_of_rectangular_arrays_l75_75223


namespace arithmetic_first_term_l75_75387

theorem arithmetic_first_term (a : ℕ) (d : ℕ) (T : ℕ → ℕ) (k : ℕ) :
  (∀ n : ℕ, T n = n * (2 * a + (n - 1) * d) / 2) →
  (∀ n : ℕ, T (4 * n) / T n = k) →
  d = 5 →
  k = 16 →
  a = 3 := 
by
  sorry

end arithmetic_first_term_l75_75387


namespace sara_initial_quarters_l75_75458

theorem sara_initial_quarters (total_quarters: ℕ) (dad_gave: ℕ) (initial_quarters: ℕ) 
  (h1: dad_gave = 49) (h2: total_quarters = 70) (h3: total_quarters = initial_quarters + dad_gave) :
  initial_quarters = 21 := 
by {
  sorry
}

end sara_initial_quarters_l75_75458


namespace range_of_m_l75_75933

theorem range_of_m (x m : ℝ) (h1 : (2 * x + m) / (x - 1) = 1) (h2 : x ≥ 0) : m ≤ -1 ∧ m ≠ -2 :=
sorry

end range_of_m_l75_75933


namespace winning_candidate_percentage_l75_75078

theorem winning_candidate_percentage (v1 v2 v3 : ℕ) (h1 : v1 = 1136) (h2 : v2 = 7636) (h3 : v3 = 11628) :
  ((v3: ℝ) / (v1 + v2 + v3)) * 100 = 57 := by
  sorry

end winning_candidate_percentage_l75_75078


namespace necessary_condition_not_sufficient_condition_l75_75428

noncomputable def zero_point (a : ℝ) : Prop :=
  ∃ x : ℝ, 3^x + a - 1 = 0

noncomputable def decreasing_log (a : ℝ) : Prop :=
  0 < a ∧ a < 1

theorem necessary_condition (a : ℝ) (h : zero_point a) : 0 < a ∧ a < 1 := sorry

theorem not_sufficient_condition (a : ℝ) (h : 0 < a ∧ a < 1) : ¬(zero_point a) := sorry

end necessary_condition_not_sufficient_condition_l75_75428


namespace length_linear_function_alpha_increase_l75_75404

variable (l : ℝ) (l₀ : ℝ) (t : ℝ) (α : ℝ)

theorem length_linear_function 
  (h_formula : l = l₀ * (1 + α * t)) : 
  ∃ (f : ℝ → ℝ), (∀ t, f t = l₀ + l₀ * α * t ∧ (l = f t)) :=
by {
  -- Proof would go here
  sorry
}

theorem alpha_increase 
  (h_formula : l = l₀ * (1 + α * t))
  (h_initial : t = 1) :
  α = (l - l₀) / l₀ :=
by {
  -- Proof would go here
  sorry
}

end length_linear_function_alpha_increase_l75_75404


namespace relationship_y1_y2_y3_l75_75769

-- Define the quadratic function
def quadratic (x : ℝ) (k : ℝ) : ℝ :=
  -(x - 2) ^ 2 + k

-- Define the points A, B, and C
def A (y1 k : ℝ) := ∃ y1, quadratic (-1 / 2) k = y1
def B (y2 k : ℝ) := ∃ y2, quadratic (1) k = y2
def C (y3 k : ℝ) := ∃ y3, quadratic (4) k = y3

theorem relationship_y1_y2_y3 (y1 y2 y3 k: ℝ)
  (hA : A y1 k)
  (hB : B y2 k)
  (hC : C y3 k) :
  y1 < y3 ∧ y3 < y2 :=
  sorry

end relationship_y1_y2_y3_l75_75769


namespace James_age_is_47_5_l75_75617

variables (James_Age Mara_Age : ℝ)

def condition1 : Prop := James_Age = 3 * Mara_Age - 20
def condition2 : Prop := James_Age + Mara_Age = 70

theorem James_age_is_47_5 (h1 : condition1 James_Age Mara_Age) (h2 : condition2 James_Age Mara_Age) : James_Age = 47.5 :=
by
  sorry

end James_age_is_47_5_l75_75617


namespace students_not_yet_pictured_l75_75526

def students_in_class : ℕ := 24
def students_before_lunch : ℕ := students_in_class / 3
def students_after_lunch_before_gym : ℕ := 10
def total_students_pictures_taken : ℕ := students_before_lunch + students_after_lunch_before_gym

theorem students_not_yet_pictured : total_students_pictures_taken = 18 → students_in_class - total_students_pictures_taken = 6 := by
  intros h
  rw [h]
  rfl

end students_not_yet_pictured_l75_75526


namespace first_term_of_geometric_sequence_l75_75300

theorem first_term_of_geometric_sequence (a r : ℚ) 
  (h1 : a * r = 18) 
  (h2 : a * r^2 = 24) : 
  a = 27 / 2 := 
sorry

end first_term_of_geometric_sequence_l75_75300


namespace complement_of_A_with_respect_to_U_l75_75319

def U : Set ℤ := {1, 2, 3, 4, 5}
def A : Set ℤ := {x | abs (x - 3) < 2}
def C_UA : Set ℤ := { x | x ∈ U ∧ x ∉ A }

theorem complement_of_A_with_respect_to_U :
  C_UA = {1, 5} :=
by
  sorry

end complement_of_A_with_respect_to_U_l75_75319


namespace tangent_line_at_point_l75_75355

noncomputable def tangent_line_equation (x y : ℝ) : Prop :=
x + 4 * y - 3 = 0

theorem tangent_line_at_point (x y : ℝ) (h₁ : y = 1 / x^2) (h₂ : x = 2) (h₃ : y = 1/4) :
  tangent_line_equation x y :=
by 
  sorry

end tangent_line_at_point_l75_75355


namespace xiao_ming_runs_distance_l75_75963

theorem xiao_ming_runs_distance 
  (num_trees : ℕ) 
  (first_tree : ℕ) 
  (last_tree : ℕ) 
  (distance_between_trees : ℕ) 
  (gap_count : ℕ) 
  (total_distance : ℕ)
  (h1 : num_trees = 200) 
  (h2 : first_tree = 1) 
  (h3 : last_tree = 200) 
  (h4 : distance_between_trees = 6) 
  (h5 : gap_count = last_tree - first_tree)
  (h6 : total_distance = gap_count * distance_between_trees) :
  total_distance = 1194 :=
sorry

end xiao_ming_runs_distance_l75_75963


namespace alice_more_than_half_sum_l75_75170

-- Conditions
def row_of_fifty_coins (denominations : List ℤ) : Prop :=
  denominations.length = 50 ∧ (List.sum denominations) % 2 = 1

def alice_starts (denominations : List ℤ) : Prop := True
def bob_follows (denominations : List ℤ) : Prop := True
def alternating_selection (denominations : List ℤ) : Prop := True

-- Question/Proof Goal
theorem alice_more_than_half_sum (denominations : List ℤ) 
  (h1 : row_of_fifty_coins denominations)
  (h2 : alice_starts denominations)
  (h3 : bob_follows denominations)
  (h4 : alternating_selection denominations) :
  ∃ s_A : ℤ, s_A > (List.sum denominations) / 2 ∧ s_A ≤ List.sum denominations :=
sorry

end alice_more_than_half_sum_l75_75170


namespace max_b_no_lattice_point_l75_75757

theorem max_b_no_lattice_point (m : ℚ) (x : ℤ) (b : ℚ) :
  (y = mx + 3) → (0 < x ∧ x ≤ 50) → (2/5 < m ∧ m < b) → 
  ∀ (x : ℕ), y ≠ m * x + 3 →
  b = 11/51 :=
sorry

end max_b_no_lattice_point_l75_75757


namespace michelle_drives_294_miles_l75_75724

theorem michelle_drives_294_miles
  (total_distance : ℕ)
  (michelle_drives : ℕ)
  (katie_drives : ℕ)
  (tracy_drives : ℕ)
  (h1 : total_distance = 1000)
  (h2 : michelle_drives = 3 * katie_drives)
  (h3 : tracy_drives = 2 * michelle_drives + 20)
  (h4 : katie_drives + michelle_drives + tracy_drives = total_distance) :
  michelle_drives = 294 := by
  sorry

end michelle_drives_294_miles_l75_75724


namespace smaller_number_is_four_l75_75690

theorem smaller_number_is_four (x y : ℝ) (h1 : x + y = 18) (h2 : x - y = 10) : y = 4 :=
by
  sorry

end smaller_number_is_four_l75_75690


namespace sugar_at_home_l75_75916

-- Definitions based on conditions
def bags_of_sugar := 2
def cups_per_bag := 6
def cups_for_batter_per_12_cupcakes := 1
def cups_for_frosting_per_12_cupcakes := 2
def dozens_of_cupcakes := 5

-- Calculation of total sugar needed and bought, in terms of definitions
def total_cupcakes := dozens_of_cupcakes * 12
def total_sugar_needed_for_batter := (total_cupcakes / 12) * cups_for_batter_per_12_cupcakes
def total_sugar_needed_for_frosting := dozens_of_cupcakes * cups_for_frosting_per_12_cupcakes
def total_sugar_needed := total_sugar_needed_for_batter + total_sugar_needed_for_frosting
def total_sugar_bought := bags_of_sugar * cups_per_bag

-- The statement to be proven in Lean
theorem sugar_at_home : total_sugar_needed - total_sugar_bought = 3 := by
  sorry

end sugar_at_home_l75_75916


namespace num_parallelogram_even_l75_75367

-- Define the conditions of the problem in Lean
def isosceles_right_triangle (base_length : ℕ) := 
  base_length = 2

def square (side_length : ℕ) := 
  side_length = 1

def parallelogram (sides_length : ℕ) (diagonals_length : ℕ) := 
  sides_length = 1 ∧ diagonals_length = 1

-- Main statement to prove
theorem num_parallelogram_even (num_triangles num_squares num_parallelograms : ℕ)
  (Htriangle : ∀ t, t < num_triangles → isosceles_right_triangle 2)
  (Hsquare : ∀ s, s < num_squares → square 1)
  (Hparallelogram : ∀ p, p < num_parallelograms → parallelogram 1 1) :
  num_parallelograms % 2 = 0 := 
sorry

end num_parallelogram_even_l75_75367


namespace geometric_sequence_implies_condition_counterexample_condition_does_not_imply_geometric_sequence_geometric_sequence_sufficient_not_necessary_l75_75767

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a n ≠ 0 ∧ (a (n + 1) = a n * (a (n + 1) / a n))

theorem geometric_sequence_implies_condition (a : ℕ → ℝ) :
  is_geometric_sequence a → ∀ n, (a n)^2 = a (n - 1) * a (n + 1) := sorry

theorem counterexample_condition_does_not_imply_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, (a n)^2 = a (n - 1) * a (n + 1)) → ¬ is_geometric_sequence a := sorry

theorem geometric_sequence_sufficient_not_necessary (a : ℕ → ℝ) :
  (is_geometric_sequence a → ∀ n, (a n)^2 = a (n - 1) * a (n + 1)) ∧
  ((∀ n, (a n)^2 = a (n - 1) * a (n + 1)) → ¬ is_geometric_sequence a) := by
  exact ⟨geometric_sequence_implies_condition a, counterexample_condition_does_not_imply_geometric_sequence a⟩

end geometric_sequence_implies_condition_counterexample_condition_does_not_imply_geometric_sequence_geometric_sequence_sufficient_not_necessary_l75_75767


namespace three_digit_ends_in_5_divisible_by_5_l75_75468

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ends_in_5 (n : ℕ) : Prop := n % 10 = 5

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

theorem three_digit_ends_in_5_divisible_by_5 (N : ℕ) 
  (h1 : is_three_digit N) 
  (h2 : ends_in_5 N) : is_divisible_by_5 N := 
sorry

end three_digit_ends_in_5_divisible_by_5_l75_75468


namespace range_of_a_l75_75744

open Real

theorem range_of_a {a : ℝ} :
  (∃ x : ℝ, sqrt (3 * x + 6) + sqrt (14 - x) > a) → a < 8 :=
by
  intro h
  sorry

end range_of_a_l75_75744


namespace equation_holds_true_l75_75588

theorem equation_holds_true (a b : ℝ) (h₁ : a ≠ 0) (h₂ : 2 * b - a ≠ 0) :
  ((a + 2 * b) / a = b / (2 * b - a)) ↔ 
  (a = -b * (1 + Real.sqrt 17) / 2 ∨ a = -b * (1 - Real.sqrt 17) / 2) := 
sorry

end equation_holds_true_l75_75588


namespace correct_transformation_l75_75191

theorem correct_transformation (a b : ℝ) (hb : b ≠ 0): (a / b = 2 * a / (2 * b)) :=
by
  sorry

end correct_transformation_l75_75191


namespace triangle_area_correct_l75_75340

def vector_2d (x y : ℝ) : ℝ × ℝ := (x, y)

def area_of_triangle (a b : ℝ × ℝ) : ℝ :=
  0.5 * abs (a.1 * b.2 - a.2 * b.1)

def a : ℝ × ℝ := vector_2d 3 2
def b : ℝ × ℝ := vector_2d 1 5

theorem triangle_area_correct : area_of_triangle a b = 6.5 :=
by
  sorry

end triangle_area_correct_l75_75340


namespace possible_values_for_a_l75_75895

def setM : Set ℝ := {x | x^2 + x - 6 = 0}
def setN (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

theorem possible_values_for_a (a : ℝ) : (∀ x, x ∈ setN a → x ∈ setM) ↔ (a = -1 ∨ a = 0 ∨ a = 2 / 3) := 
by
  sorry

end possible_values_for_a_l75_75895


namespace machine_rate_ratio_l75_75578

theorem machine_rate_ratio (A B : ℕ) (h1 : ∃ A : ℕ, 8 * A = 8 * A)
  (h2 : ∃ W : ℕ, W = 8 * A)
  (h3 : ∃ W1 : ℕ, W1 = 6 * A)
  (h4 : ∃ W2 : ℕ, W2 = 2 * A)
  (h5 : ∃ B : ℕ, 8 * B = 2 * A) :
  (B:ℚ) / (A:ℚ) = 1 / 4 :=
by sorry

end machine_rate_ratio_l75_75578


namespace scientific_notation_of_population_l75_75852

theorem scientific_notation_of_population :
  (141260 : ℝ) = 1.4126 * 10^5 :=
sorry

end scientific_notation_of_population_l75_75852


namespace minimum_value_frac_l75_75964

theorem minimum_value_frac (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (2 / a) + (3 / b) ≥ 5 + 2 * Real.sqrt 6 := 
sorry

end minimum_value_frac_l75_75964


namespace intersection_M_N_l75_75517

-- Define the set M based on the given condition
def M : Set ℝ := { x | x^2 > 1 }

-- Define the set N based on the given elements
def N : Set ℝ := { x | x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 }

-- Prove that the intersection of M and N is {-2, 2}
theorem intersection_M_N : M ∩ N = { -2, 2 } := by
  sorry

end intersection_M_N_l75_75517


namespace planes_parallel_or_intersect_l75_75824

variables {Plane : Type} {Line : Type}
variables (α β : Plane) (a b : Line)

-- Conditions
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def not_parallel (l1 l2 : Line) : Prop := sorry

-- Given conditions
axiom h₁ : line_in_plane a α
axiom h₂ : line_in_plane b β
axiom h₃ : not_parallel a b

-- The theorem statement
theorem planes_parallel_or_intersect : (exists l : Line, line_in_plane l α ∧ line_in_plane l β) ∨ (α = β) :=
sorry

end planes_parallel_or_intersect_l75_75824


namespace edge_length_in_mm_l75_75520

-- Definitions based on conditions
def cube_volume (a : ℝ) : ℝ := a^3

axiom volume_of_dice : cube_volume 2 = 8

-- Statement of the theorem to be proved
theorem edge_length_in_mm : ∃ (a : ℝ), cube_volume a = 8 ∧ a * 10 = 20 := sorry

end edge_length_in_mm_l75_75520


namespace intersection_of_sets_l75_75622

variable (A B : Set ℝ) (x : ℝ)

def setA : Set ℝ := { x | x > 0 }
def setB : Set ℝ := { x | -1 < x ∧ x ≤ 2 }

theorem intersection_of_sets : A ∩ B = { x | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_of_sets_l75_75622


namespace mass_increase_l75_75638

theorem mass_increase (ρ₁ ρ₂ m₁ m₂ a₁ a₂ : ℝ) (cond1 : ρ₂ = 2 * ρ₁) 
                      (cond2 : a₂ = 2 * a₁) (cond3 : m₁ = ρ₁ * (a₁^3)) 
                      (cond4 : m₂ = ρ₂ * (a₂^3)) : 
                      ((m₂ - m₁) / m₁) * 100 = 1500 := by
  sorry

end mass_increase_l75_75638


namespace angle_problem_l75_75915

theorem angle_problem (θ : ℝ) (h1 : 90 - θ = 0.4 * (180 - θ)) (h2 : 180 - θ = 2 * θ) : θ = 30 :=
by
  sorry

end angle_problem_l75_75915


namespace variance_of_data_set_l75_75961

open Real

def dataSet := [11, 12, 15, 18, 13, 15]

theorem variance_of_data_set :
  let mean := (11 + 12 + 15 + 13 + 18 + 15) / 6
  let variance := (1 / 6) * ((11 - mean)^2 + (12 - mean)^2 + (15 - mean)^2 + (13 - mean)^2 + (18 - mean)^2 + (15 - mean)^2)
  variance = 16 / 3 :=
by
  let mean := (11 + 12 + 15 + 13 + 18 + 15) / 6
  let variance := (1 / 6) * ((11 - mean)^2 + (12 - mean)^2 + (15 - mean)^2 + (13 - mean)^2 + (18 - mean)^2 + (15 - mean)^2)
  have h : mean = 14 := sorry
  have h_variance : variance = 16 / 3 := sorry
  exact h_variance

end variance_of_data_set_l75_75961


namespace tan_sum_pi_eighths_l75_75671

theorem tan_sum_pi_eighths : (Real.tan (Real.pi / 8) + Real.tan (3 * Real.pi / 8) = 2 * Real.sqrt 2) :=
by
  sorry

end tan_sum_pi_eighths_l75_75671


namespace intersection_hyperbola_circle_l75_75695

theorem intersection_hyperbola_circle :
  {p : ℝ × ℝ | p.1^2 - 9 * p.2^2 = 36 ∧ p.1^2 + p.2^2 = 36} = {(6, 0), (-6, 0)} :=
by sorry

end intersection_hyperbola_circle_l75_75695


namespace current_at_time_l75_75812

noncomputable def I (t : ℝ) : ℝ := 5 * (Real.sin (100 * Real.pi * t + Real.pi / 3))

theorem current_at_time (t : ℝ) (h : t = 1 / 200) : I t = 5 / 2 := by
  sorry

end current_at_time_l75_75812


namespace max_distinct_terms_degree_6_l75_75715

-- Step 1: Define the variables and conditions
def polynomial_max_num_terms (deg : ℕ) (vars : ℕ) : ℕ :=
  Nat.choose (deg + vars - 1) (vars - 1)

-- Step 2: State the specific problem
theorem max_distinct_terms_degree_6 :
  polynomial_max_num_terms 6 5 = 210 :=
by
  sorry

end max_distinct_terms_degree_6_l75_75715


namespace non_planar_characterization_l75_75758

-- Definitions:
structure Graph where
  V : ℕ
  E : ℕ
  F : ℕ

def is_planar (G : Graph) : Prop :=
  G.V - G.E + G.F = 2

def edge_inequality (G : Graph) : Prop :=
  G.E ≤ 3 * G.V - 6

def has_subgraph_K5_or_K33 (G : Graph) : Prop := sorry -- Placeholder for the complex subgraph check

-- Theorem statement:
theorem non_planar_characterization (G : Graph) (hV : G.V ≥ 3) :
  ¬ is_planar G ↔ ¬ edge_inequality G ∨ has_subgraph_K5_or_K33 G := sorry

end non_planar_characterization_l75_75758


namespace ratio_Jane_to_John_l75_75042

-- Define the conditions as given in the problem.
variable (J N : ℕ) -- total products inspected by John and Jane
variable (rJ rN rT : ℚ) -- rejection rates for John, Jane, and total

-- Setting up the provided conditions
axiom h1 : rJ = 0.005 -- John rejected 0.5% of the products he inspected
axiom h2 : rN = 0.007 -- Jane rejected 0.7% of the products she inspected
axiom h3 : rT = 0.0075 -- 0.75% of the total products were rejected

-- Prove the ratio of products inspected by Jane to products inspected by John is 5
theorem ratio_Jane_to_John : (rJ * J + rN * N) = rT * (J + N) → N = 5 * J :=
by 
  sorry

end ratio_Jane_to_John_l75_75042


namespace choose_3_from_12_l75_75763

theorem choose_3_from_12 : (Nat.choose 12 3) = 220 := by
  sorry

end choose_3_from_12_l75_75763


namespace tickets_needed_to_ride_l75_75529

noncomputable def tickets_required : Float :=
let ferris_wheel := 3.5
let roller_coaster := 8.0
let bumper_cars := 5.0
let additional_ride_discount := 0.5
let newspaper_coupon := 1.5
let teacher_discount := 2.0

let total_cost_without_discounts := ferris_wheel + roller_coaster + bumper_cars
let total_additional_discounts := additional_ride_discount * 2
let total_coupons_discounts := newspaper_coupon + teacher_discount

let total_cost_with_discounts := total_cost_without_discounts - total_additional_discounts - total_coupons_discounts
total_cost_with_discounts

theorem tickets_needed_to_ride : tickets_required = 12.0 := by
  sorry

end tickets_needed_to_ride_l75_75529


namespace evaluate_expression_l75_75390

theorem evaluate_expression : 
  (4 * 6 / (12 * 16)) * (8 * 12 * 16 / (4 * 6 * 8)) = 1 :=
by
  sorry

end evaluate_expression_l75_75390


namespace winning_candidate_percentage_l75_75881

/-- 
In an election, a candidate won by a majority of 1040 votes out of a total of 5200 votes.
Prove that the winning candidate received 60% of the votes.
-/
theorem winning_candidate_percentage {P : ℝ} (h_majority : (P * 5200) - ((1 - P) * 5200) = 1040) : P = 0.60 := 
by
  sorry

end winning_candidate_percentage_l75_75881


namespace percent_covered_by_larger_triangles_l75_75699

-- Define the number of small triangles in one large hexagon
def total_small_triangles := 16

-- Define the number of small triangles that are part of the larger triangles within one hexagon
def small_triangles_in_larger_triangles := 9

-- Calculate the fraction of the area of the hexagon covered by larger triangles
def fraction_covered_by_larger_triangles := 
  small_triangles_in_larger_triangles / total_small_triangles

-- Define the expected result as a fraction of the total area
def expected_fraction := 56 / 100

-- The proof problem in Lean 4 statement:
theorem percent_covered_by_larger_triangles
  (h1 : fraction_covered_by_larger_triangles = 9 / 16) :
  fraction_covered_by_larger_triangles = expected_fraction :=
  by
    sorry

end percent_covered_by_larger_triangles_l75_75699


namespace eight_percent_of_fifty_is_four_l75_75062

theorem eight_percent_of_fifty_is_four : 0.08 * 50 = 4 := by
  sorry

end eight_percent_of_fifty_is_four_l75_75062


namespace roots_satisfy_conditions_l75_75002

variable (a x1 x2 : ℝ)

-- Conditions
def condition1 : Prop := x1 * x2 + x1 + x2 - a = 0
def condition2 : Prop := x1 * x2 - a * (x1 + x2) + 1 = 0

-- Derived quadratic equation
def quadratic_eq : Prop := ∃ x : ℝ, x^2 - x + (a - 1) = 0

theorem roots_satisfy_conditions (h1: condition1 a x1 x2) (h2: condition2 a x1 x2) : quadratic_eq a :=
  sorry

end roots_satisfy_conditions_l75_75002


namespace exists_nat_solution_for_A_415_l75_75662

theorem exists_nat_solution_for_A_415 : ∃ (m n : ℕ), 3 * m^2 * n = n^3 + 415 := by
  sorry

end exists_nat_solution_for_A_415_l75_75662


namespace triangle_arithmetic_angles_l75_75685

/-- The angles in a triangle are in arithmetic progression and the side lengths are 6, 7, and y.
    The sum of the possible values of y equals a + sqrt b + sqrt c,
    where a, b, and c are positive integers. Prove that a + b + c = 68. -/
theorem triangle_arithmetic_angles (y : ℝ) (a b c : ℕ) (h1 : a = 3) (h2 : b = 22) (h3 : c = 43) :
    (∃ y1 y2 : ℝ, y1 = 3 + Real.sqrt 22 ∧ y2 = Real.sqrt 43 ∧ (y = y1 ∨ y = y2))
    → a + b + c = 68 :=
by
  sorry

end triangle_arithmetic_angles_l75_75685


namespace value_of_b_l75_75401

theorem value_of_b (b : ℝ) (h1 : 1/2 * (b / 3) * b = 6) (h2 : b ≥ 0) : b = 6 := sorry

end value_of_b_l75_75401


namespace ratio_a_c_l75_75497

theorem ratio_a_c {a b c : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : ∀ x : ℝ, x^2 + a * x + b = 0 → 2 * x^2 + 2 * a * x + 2 * b = 0)
  (h2 : ∀ x : ℝ, x^2 + b * x + c = 0 → x^2 + b * x + c = 0) :
  a / c = 1 / 8 :=
by
  sorry

end ratio_a_c_l75_75497


namespace product_of_last_two_digits_l75_75469

theorem product_of_last_two_digits (A B : ℕ) (h1 : A + B = 12) (h2 : (10 * A + B) % 8 = 0) : A * B = 32 :=
by
  sorry

end product_of_last_two_digits_l75_75469


namespace max_poly_l75_75848

noncomputable def poly (a b : ℝ) : ℝ :=
  a^4 * b + a^3 * b + a^2 * b + a * b + a * b^2 + a * b^3 + a * b^4

theorem max_poly (a b : ℝ) (h : a + b = 4) :
  ∃ (a b : ℝ) (h : a + b = 4), poly a b = (7225 / 56) :=
sorry

end max_poly_l75_75848


namespace decrease_in_combined_area_l75_75634

theorem decrease_in_combined_area (r1 r2 r3 : ℝ) :
    let π := Real.pi
    let A_original := π * (r1 ^ 2) + π * (r2 ^ 2) + π * (r3 ^ 2)
    let r1' := r1 * 0.5
    let r2' := r2 * 0.5
    let r3' := r3 * 0.5
    let A_new := π * (r1' ^ 2) + π * (r2' ^ 2) + π * (r3' ^ 2)
    let Decrease := A_original - A_new
    Decrease = 0.75 * π * (r1 ^ 2) + 0.75 * π * (r2 ^ 2) + 0.75 * π * (r3 ^ 2) :=
by
  sorry

end decrease_in_combined_area_l75_75634


namespace center_of_circle_polar_eq_l75_75294

theorem center_of_circle_polar_eq (ρ θ : ℝ) : 
    (∀ ρ θ, ρ = 2 * Real.cos θ ↔ (ρ * Real.cos θ - 1)^2 + (ρ * Real.sin θ)^2 = 1) → 
    ∃ x y : ℝ, x = 1 ∧ y = 0 :=
by
  sorry

end center_of_circle_polar_eq_l75_75294


namespace c_minus_3_eq_neg3_l75_75871

variable (g : ℝ → ℝ)
variable (c : ℝ)

-- defining conditions
axiom invertible_g : Function.Injective g
axiom g_c_eq_3 : g c = 3
axiom g_3_eq_5 : g 3 = 5

-- The goal is to prove that c - 3 = -3
theorem c_minus_3_eq_neg3 : c - 3 = -3 :=
by
  sorry

end c_minus_3_eq_neg3_l75_75871


namespace complex_fraction_l75_75082

open Complex

theorem complex_fraction
  (a b : ℂ)
  (h : (a + b) / (a - b) - (a - b) / (a + b) = 2) :
  (a^4 + b^4) / (a^4 - b^4) - (a^4 - b^4) / (a^4 + b^4) = ((a^2 + b^2) - 3) / 3 := 
by
  sorry

end complex_fraction_l75_75082


namespace central_angle_of_sector_l75_75474

noncomputable def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
noncomputable def arc_length (r α : ℝ) : ℝ := r * α

theorem central_angle_of_sector :
  ∀ (r α : ℝ),
    circumference r = 2 * Real.pi + 2 →
    arc_length r α = 2 * Real.pi - 2 →
    α = Real.pi - 1 :=
by
  intros r α hcirc harc
  sorry

end central_angle_of_sector_l75_75474


namespace friends_bought_boxes_l75_75656

def rainbow_colors : ℕ := 7
def total_pencils : ℕ := 56
def pencils_per_box : ℕ := rainbow_colors

theorem friends_bought_boxes (emily_box : ℕ := 1) :
  (total_pencils / pencils_per_box) - emily_box = 7 := by
  sorry

end friends_bought_boxes_l75_75656


namespace inequality_solution_l75_75998

theorem inequality_solution (x : ℝ) : 
  (x - 1) / (2 * x + 1) ≤ 0 ↔ -1 / 2 < x ∧ x ≤ 1 :=
sorry

end inequality_solution_l75_75998


namespace T_n_formula_l75_75869

-- Define the given sequence sum S_n
def S (n : ℕ) : ℚ := (n^2 : ℚ) / 2 + (3 * n : ℚ) / 2

-- Define the general term a_n for the sequence {a_n}
def a (n : ℕ) : ℚ := if n = 1 then 2 else n + 1

-- Define the sequence b_n
def b (n : ℕ) : ℚ := a (n + 2) - a n + 1 / (a (n + 2) * a n)

-- Define the sum of the first n terms of the sequence {b_n}
def T (n : ℕ) : ℚ := 2 * n + 5 / 12 - (2 * n + 5) / (2 * (n + 2) * (n + 3))

-- Prove the equality of T_n with the given expression
theorem T_n_formula (n : ℕ) : T n = 2 * n + 5 / 12 - (2 * n + 5) / (2 * (n + 2) * (n + 3)) := sorry

end T_n_formula_l75_75869


namespace line_passes_through_fixed_point_l75_75810

-- Given a line equation kx - y + 1 - 3k = 0
def line_equation (k x y : ℝ) : Prop := k * x - y + 1 - 3 * k = 0

-- We need to prove that this line passes through the point (3,1)
theorem line_passes_through_fixed_point (k : ℝ) : line_equation k 3 1 :=
by
  sorry

end line_passes_through_fixed_point_l75_75810


namespace factor_poly_l75_75274

theorem factor_poly : 
  ∃ (a b c d e f : ℤ), a < d ∧ (a*x^2 + b*x + c)*(d*x^2 + e*x + f) = (x^2 + 6*x + 9 - 64*x^4) ∧ 
  (a = -8 ∧ b = 1 ∧ c = 3 ∧ d = 8 ∧ e = 1 ∧ f = 3) := 
sorry

end factor_poly_l75_75274


namespace B_subset_A_implies_m_le_5_l75_75640

variable (A B : Set ℝ)
variable (m : ℝ)

def setA : Set ℝ := {x | 4 ≤ x ∧ x ≤ 8}
def setB (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2 * m - 2}

theorem B_subset_A_implies_m_le_5 :
  B ⊆ A → (∀ k : ℝ, k ∈ setB m → k ∈ setA) → m ≤ 5 :=
by
  sorry

end B_subset_A_implies_m_le_5_l75_75640


namespace power_function_below_identity_l75_75967

theorem power_function_below_identity {α : ℝ} :
  (∀ x : ℝ, 1 < x → x^α < x) → α < 1 :=
by
  intro h
  sorry

end power_function_below_identity_l75_75967


namespace minimum_value_l75_75131

theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / (x + 1) + 9 / y = 1) : 4 * x + y ≥ 21 :=
sorry

end minimum_value_l75_75131


namespace find_m_values_l75_75162

def is_solution (m : ℝ) : Prop :=
  let A : Set ℝ := {1, -2}
  let B : Set ℝ := {x | m * x + 1 = 0}
  B ⊆ A

theorem find_m_values :
  {m : ℝ | is_solution m} = {0, -1, 1 / 2} :=
by
  sorry

end find_m_values_l75_75162


namespace sum_of_x_values_l75_75177

theorem sum_of_x_values (x : ℂ) (h₁ : x ≠ -3) (h₂ : 3 = (x^3 - 3 * x^2 - 10 * x) / (x + 3)) : x + (5 - x) = 5 :=
sorry

end sum_of_x_values_l75_75177


namespace find_a_minus_b_l75_75566

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b * x + 1

theorem find_a_minus_b (a b : ℝ)
  (h1 : deriv (f a b) 1 = -2)
  (h2 : deriv (f a b) (2 / 3) = 0) :
  a - b = 10 :=
sorry

end find_a_minus_b_l75_75566


namespace total_students_l75_75570

theorem total_students (n x : ℕ) (h1 : 3 * n + 48 = 6 * n) (h2 : 4 * n + x = 2 * n + 2 * x) : n = 16 :=
by
  sorry

end total_students_l75_75570


namespace min_fraction_expression_l75_75189

theorem min_fraction_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : 1 / a + 1 / b = 1) : 
  ∃ a b, ∃ (h : 1 / a + 1 / b = 1), a > 1 ∧ b > 1 ∧ (1 / (a - 1) + 4 / (b - 1)) = 4 := 
by 
  sorry

end min_fraction_expression_l75_75189


namespace coordinates_of_point_l75_75890

theorem coordinates_of_point (x : ℝ) (P : ℝ × ℝ) (h : P = (1 - x, 2 * x + 1)) (y_axis : P.1 = 0) : P = (0, 3) :=
by
  sorry

end coordinates_of_point_l75_75890


namespace percentage_decrease_is_20_l75_75005

-- Define the original and new prices in Rs.
def original_price : ℕ := 775
def new_price : ℕ := 620

-- Define the decrease in price
def decrease_in_price : ℕ := original_price - new_price

-- Define the formula to calculate the percentage decrease
def percentage_decrease (orig_price new_price : ℕ) : ℕ :=
  (decrease_in_price * 100) / orig_price

-- Prove that the percentage decrease is 20%
theorem percentage_decrease_is_20 :
  percentage_decrease original_price new_price = 20 :=
by
  sorry

end percentage_decrease_is_20_l75_75005


namespace find_mn_expression_l75_75900

-- Define the conditions
variables (m n : ℤ)
axiom abs_m_eq_3 : |m| = 3
axiom abs_n_eq_2 : |n| = 2
axiom m_lt_n : m < n

-- State the problem
theorem find_mn_expression : m^2 + m * n + n^2 = 7 ∨ m^2 + m * n + n^2 = 19 :=
by
  sorry

end find_mn_expression_l75_75900


namespace mean_of_quadrilateral_angles_l75_75794

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l75_75794


namespace moores_law_2000_l75_75322

noncomputable def number_of_transistors (year : ℕ) : ℕ :=
  if year = 1990 then 1000000
  else 1000000 * 2 ^ ((year - 1990) / 2)

theorem moores_law_2000 :
  number_of_transistors 2000 = 32000000 :=
by
  unfold number_of_transistors
  rfl

end moores_law_2000_l75_75322


namespace tim_meditation_time_l75_75509

-- Definitions of the conditions:
def time_reading_week (t_reading : ℕ) : Prop := t_reading = 14
def twice_as_much_reading (t_reading t_meditate : ℕ) : Prop := t_reading = 2 * t_meditate

-- The theorem to prove:
theorem tim_meditation_time (t_reading t_meditate_per_day : ℕ) 
  (h1 : time_reading_week t_reading)
  (h2 : twice_as_much_reading t_reading (7 * t_meditate_per_day)) :
  t_meditate_per_day = 1 :=
by
  sorry

end tim_meditation_time_l75_75509


namespace ages_of_patients_l75_75538

theorem ages_of_patients (x y : ℕ) 
  (h1 : x - y = 44) 
  (h2 : x * y = 1280) : 
  (x = 64 ∧ y = 20) ∨ (x = 20 ∧ y = 64) := by
  sorry

end ages_of_patients_l75_75538


namespace rancher_total_animals_l75_75043

theorem rancher_total_animals
  (H C : ℕ) (h1 : C = 5 * H) (h2 : C = 140) :
  C + H = 168 := 
sorry

end rancher_total_animals_l75_75043


namespace chick_hits_at_least_five_l75_75885

theorem chick_hits_at_least_five (x y z : ℕ) (h1 : 9 * x + 5 * y + 2 * z = 61) (h2 : x + y + z = 10) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) : x ≥ 5 :=
sorry

end chick_hits_at_least_five_l75_75885


namespace polynomial_no_positive_real_roots_l75_75173

theorem polynomial_no_positive_real_roots : 
  ¬ ∃ x : ℝ, x > 0 ∧ x^3 + 6 * x^2 + 11 * x + 6 = 0 :=
sorry

end polynomial_no_positive_real_roots_l75_75173


namespace triangle_angles_l75_75684

theorem triangle_angles
  (A B C M : Type)
  (ortho_divides_height_A : ∀ (H_AA1 : ℝ), ∃ (H_AM : ℝ), H_AA1 = H_AM * 3 ∧ H_AM = 2 * H_AA1 / 3)
  (ortho_divides_height_B : ∀ (H_BB1 : ℝ), ∃ (H_BM : ℝ), H_BB1 = H_BM * 5 / 2 ∧ H_BM = 3 * H_BB1 / 5) :
  ∃ α β γ : ℝ, α = 60 + 40 / 60 ∧ β = 64 + 36 / 60 ∧ γ = 54 + 44 / 60 :=
by { 
  sorry 
}

end triangle_angles_l75_75684


namespace no_valid_middle_number_l75_75743

theorem no_valid_middle_number
    (x : ℤ)
    (h1 : (x % 2 = 1))
    (h2 : 3 * x + 12 = x^2 + 20) :
    false :=
by
    sorry

end no_valid_middle_number_l75_75743


namespace probability_order_correct_l75_75279

inductive Phenomenon
| Certain
| VeryLikely
| Possible
| Impossible
| NotVeryLikely

open Phenomenon

def probability_order : Phenomenon → ℕ
| Certain       => 5
| VeryLikely    => 4
| Possible      => 3
| NotVeryLikely => 2
| Impossible    => 1

theorem probability_order_correct :
  [Certain, VeryLikely, Possible, NotVeryLikely, Impossible] =
  [Certain, VeryLikely, Possible, NotVeryLikely, Impossible] :=
by
  -- skips the proof
  sorry

end probability_order_correct_l75_75279


namespace isosceles_triangle_same_area_l75_75843

-- Given conditions of the original isosceles triangle
def original_base : ℝ := 10
def original_side : ℝ := 13

-- The problem states that an isosceles triangle has the base 10 cm and side lengths 13 cm, 
-- we need to show there's another isosceles triangle with a different base but the same area.
theorem isosceles_triangle_same_area : 
  ∃ (new_base : ℝ) (new_side : ℝ), 
    new_base ≠ original_base ∧ 
    (∃ (h1 h2: ℝ), 
      h1 = 12 ∧ 
      h2 = 5 ∧
      1/2 * original_base * h1 = 60 ∧ 
      1/2 * new_base * h2 = 60) := 
sorry

end isosceles_triangle_same_area_l75_75843


namespace cost_per_tshirt_l75_75651
-- Import necessary libraries

-- Define the given conditions
def t_shirts : ℕ := 20
def total_cost : ℝ := 199

-- Define the target proof statement
theorem cost_per_tshirt : (total_cost / t_shirts) = 9.95 := 
sorry

end cost_per_tshirt_l75_75651


namespace complex_evaluation_l75_75883

theorem complex_evaluation (a b : ℂ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a^2 + a * b + b^2 = 0) : 
  (a^9 + b^9) / (a + b)^9 = -2 := 
by 
  sorry

end complex_evaluation_l75_75883


namespace total_seeds_in_watermelons_l75_75981

def slices1 : ℕ := 40
def seeds_per_slice1 : ℕ := 60
def slices2 : ℕ := 30
def seeds_per_slice2 : ℕ := 80
def slices3 : ℕ := 50
def seeds_per_slice3 : ℕ := 40

theorem total_seeds_in_watermelons :
  (slices1 * seeds_per_slice1) + (slices2 * seeds_per_slice2) + (slices3 * seeds_per_slice3) = 6800 := by
  sorry

end total_seeds_in_watermelons_l75_75981


namespace perfect_power_transfer_l75_75293

-- Given Conditions
variables {x y z : ℕ}

-- Definition of what it means to be a perfect seventh power
def is_perfect_seventh_power (n : ℕ) :=
  ∃ k : ℕ, n = k^7

-- The proof problem
theorem perfect_power_transfer 
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (h : is_perfect_seventh_power (x^3 * y^5 * z^6)) :
  is_perfect_seventh_power (x^5 * y^6 * z^3) := by
  sorry

end perfect_power_transfer_l75_75293


namespace jesses_room_length_l75_75338

theorem jesses_room_length 
  (width : ℝ)
  (tile_area : ℝ)
  (num_tiles : ℕ)
  (total_area : ℝ := num_tiles * tile_area) 
  (room_length : ℝ := total_area / width)
  (hw : width = 12)
  (hta : tile_area = 4)
  (hnt : num_tiles = 6) :
  room_length = 2 :=
by
  -- proof omitted
  sorry

end jesses_room_length_l75_75338


namespace speed_of_boat_l75_75798

-- Given conditions
variables (V_b : ℝ) (V_s : ℝ) (T : ℝ) (D : ℝ)

-- Problem statement in Lean
theorem speed_of_boat (h1 : V_s = 5) (h2 : T = 1) (h3 : D = 45) :
  D = T * (V_b + V_s) → V_b = 40 := 
by
  intro h4
  rw [h1, h2, h3] at h4
  linarith

end speed_of_boat_l75_75798


namespace total_books_borrowed_lunchtime_correct_l75_75645

def shelf_A_borrowed (X : ℕ) : Prop :=
  110 - X = 60 ∧ X = 50

def shelf_B_borrowed (Y : ℕ) : Prop :=
  150 - 50 + 20 - Y = 80 ∧ Y = 80

def shelf_C_borrowed (Z : ℕ) : Prop :=
  210 - 45 = 165 ∧ 165 - 130 = Z ∧ Z = 35

theorem total_books_borrowed_lunchtime_correct :
  ∃ (X Y Z : ℕ),
    shelf_A_borrowed X ∧
    shelf_B_borrowed Y ∧
    shelf_C_borrowed Z ∧
    X + Y + Z = 165 :=
by
  sorry

end total_books_borrowed_lunchtime_correct_l75_75645


namespace original_price_of_suit_l75_75552

theorem original_price_of_suit (P : ℝ) (hP : 0.70 * 1.30 * P = 182) : P = 200 :=
by
  sorry

end original_price_of_suit_l75_75552


namespace find_A_plus_B_plus_C_plus_D_l75_75473

noncomputable def A : ℤ := -7
noncomputable def B : ℕ := 8
noncomputable def C : ℤ := 21
noncomputable def D : ℕ := 1

def conditions_satisfied : Prop :=
  D > 0 ∧
  ¬∃ p : ℕ, Nat.Prime p ∧ p^2 ∣ B ∧ p ≠ 1 ∧ p ≠ B ∧ p ≥ 2 ∧
  Int.gcd A (Int.gcd C (Int.ofNat D)) = 1

theorem find_A_plus_B_plus_C_plus_D : conditions_satisfied → A + B + C + D = 23 :=
by
  intro h
  sorry

end find_A_plus_B_plus_C_plus_D_l75_75473


namespace minimum_point_translation_l75_75665

theorem minimum_point_translation (x y : ℝ) : 
  (∀ (x : ℝ), y = 2 * |x| - 4) →
  x = 0 →
  y = -4 →
  (∀ (x y : ℝ), x_new = x + 3 ∧ y_new = y + 4) →
  (x_new, y_new) = (3, 0) :=
sorry

end minimum_point_translation_l75_75665


namespace susan_homework_time_l75_75674

theorem susan_homework_time :
  ∀ (start finish practice : ℕ),
  start = 119 ->
  practice = 240 ->
  finish = practice - 25 ->
  (start < finish) ->
  (finish - start) = 96 :=
by
  intros start finish practice h_start h_practice h_finish h_lt
  sorry

end susan_homework_time_l75_75674


namespace sum_of_digits_of_multiple_of_990_l75_75456

theorem sum_of_digits_of_multiple_of_990 (a b c : ℕ) (h₀ : a < 10 ∧ b < 10 ∧ c < 10)
  (h₁ : ∃ (d e f g : ℕ), 123000 + 10000 * d + 1000 * e + 100 * a + 10 * b + c = 123000 + 9000 + 900 + 90 + 9 + 0)
  (h2 : (123000 + 10000 * d + 1000 * e + 100 * a + 10 * b + c) % 990 = 0) :
  a + b + c = 12 :=
by {
  sorry
}

end sum_of_digits_of_multiple_of_990_l75_75456


namespace gcd_16016_20020_l75_75015

theorem gcd_16016_20020 : Int.gcd 16016 20020 = 4004 :=
by
  sorry

end gcd_16016_20020_l75_75015


namespace minimum_erasures_correct_l75_75911

open Nat List

-- define a function that checks if a number represented as a list of digits is a palindrome
def is_palindrome (l : List ℕ) : Prop :=
  l = l.reverse

-- the given problem statement
def given_number := [1, 2, 3, 2, 3, 3, 1, 4]

-- function to find the minimum erasures to make a list a palindrome
noncomputable def min_erasures_to_palindrome (l : List ℕ) : ℕ :=
  sorry -- function implementation skipped

-- the main theorem statement
theorem minimum_erasures_correct : min_erasures_to_palindrome given_number = 3 :=
  sorry

end minimum_erasures_correct_l75_75911


namespace cost_price_percentage_l75_75496

theorem cost_price_percentage (MP CP : ℝ) 
  (h1 : MP * 0.9 = CP * (72 / 70))
  (h2 : CP / MP * 100 = 87.5) :
  CP / MP = 0.875 :=
by {
  sorry
}

end cost_price_percentage_l75_75496


namespace intersection_is_empty_l75_75554

def A : Set ℝ := { α | ∃ k : ℤ, α = (5 * k * Real.pi) / 3 }
def B : Set ℝ := { β | ∃ k : ℤ, β = (3 * k * Real.pi) / 2 }

theorem intersection_is_empty : A ∩ B = ∅ :=
by
  sorry

end intersection_is_empty_l75_75554


namespace Q_proper_subset_P_l75_75987

open Set

def P : Set ℝ := { x | x ≥ 1 }
def Q : Set ℝ := { 2, 3 }

theorem Q_proper_subset_P : Q ⊂ P :=
by
  sorry

end Q_proper_subset_P_l75_75987


namespace james_marbles_l75_75815

def marbles_in_bag_D (bag_C : ℕ) := 2 * bag_C - 1
def marbles_in_bag_E (bag_A : ℕ) := bag_A / 2
def marbles_in_bag_G (bag_E : ℕ) := bag_E

theorem james_marbles :
    ∀ (A B C D E F G : ℕ),
      A = 4 →
      B = 3 →
      C = 5 →
      D = marbles_in_bag_D C →
      E = marbles_in_bag_E A →
      F = 3 →
      G = marbles_in_bag_G E →
      28 - (D + F) + 4 = 20 := by
    intros A B C D E F G hA hB hC hD hE hF hG
    sorry

end james_marbles_l75_75815


namespace find_eccentricity_l75_75418

variable (a b : ℝ) (ha : a > 0) (hb : b > 0)
variable (asymp_cond : b / a = 1 / 2)

theorem find_eccentricity : ∃ e : ℝ, e = Real.sqrt 5 / 2 :=
by
  let c := Real.sqrt ((a^2 + b^2) / 4)
  let e := c / a
  use e
  sorry

end find_eccentricity_l75_75418


namespace trigonometric_identity_l75_75472

theorem trigonometric_identity (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin A * Real.cos B * Real.cos C + Real.cos A * Real.sin B * Real.cos C + 
  Real.cos A * Real.cos B * Real.sin C = Real.sin A * Real.sin B * Real.sin C :=
by 
  sorry

end trigonometric_identity_l75_75472


namespace wastewater_volume_2013_l75_75403

variable (x_2013 x_2014 : ℝ)
variable (condition1 : x_2014 = 38000)
variable (condition2 : x_2014 = 1.6 * x_2013)

theorem wastewater_volume_2013 : x_2013 = 23750 := by
  sorry

end wastewater_volume_2013_l75_75403


namespace compound_interest_for_2_years_l75_75290

noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

noncomputable def compound_interest (P R T : ℝ) : ℝ := P * (1 + R / 100)^T - P

theorem compound_interest_for_2_years 
  (P : ℝ) (R : ℝ) (T : ℝ) (S : ℝ)
  (h1 : S = 600)
  (h2 : R = 5)
  (h3 : T = 2)
  (h4 : simple_interest P R T = S)
  : compound_interest P R T = 615 := 
sorry

end compound_interest_for_2_years_l75_75290


namespace bobbo_minimum_speed_increase_l75_75114

theorem bobbo_minimum_speed_increase
  (initial_speed: ℝ)
  (river_width : ℝ)
  (current_speed : ℝ)
  (waterfall_distance : ℝ)
  (midpoint_distance : ℝ)
  (required_increase: ℝ) :
  initial_speed = 2 ∧ river_width = 100 ∧ current_speed = 5 ∧ waterfall_distance = 175 ∧ midpoint_distance = 50 ∧ required_increase = 3 → 
  (required_increase = (50 / (50 / current_speed)) - initial_speed) := 
by
  sorry

end bobbo_minimum_speed_increase_l75_75114


namespace train2_length_is_230_l75_75266

noncomputable def train_length_proof : Prop :=
  let speed1_kmph := 120
  let speed2_kmph := 80
  let length_train1 := 270
  let time_cross := 9
  let speed1_mps := speed1_kmph * 1000 / 3600
  let speed2_mps := speed2_kmph * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := relative_speed * time_cross
  let length_train2 := total_distance - length_train1
  length_train2 = 230

theorem train2_length_is_230 : train_length_proof :=
  by
    sorry

end train2_length_is_230_l75_75266


namespace angle_C_in_triangle_l75_75281

theorem angle_C_in_triangle (A B C : ℝ) 
  (hA : A = 90) (hB : B = 50) : (A + B + C = 180) → C = 40 :=
by
  intro hSum
  rw [hA, hB] at hSum
  linarith

end angle_C_in_triangle_l75_75281


namespace relationship_among_a_b_c_l75_75784

noncomputable def a : ℝ := (0.6:ℝ) ^ (0.2:ℝ)
noncomputable def b : ℝ := (0.2:ℝ) ^ (0.2:ℝ)
noncomputable def c : ℝ := (0.2:ℝ) ^ (0.6:ℝ)

theorem relationship_among_a_b_c : a > b ∧ b > c :=
by
  -- The proof can be added here if needed
  sorry

end relationship_among_a_b_c_l75_75784


namespace solve_system_l75_75809

noncomputable def solutions (a b c : ℝ) : Prop :=
  a^4 - b^4 = c ∧ b^4 - c^4 = a ∧ c^4 - a^4 = b

theorem solve_system :
  { (a, b, c) | solutions a b c } =
  { (0, 0, 0), (0, 1, -1), (-1, 0, 1), (1, -1, 0) } :=
by
  sorry

end solve_system_l75_75809


namespace find_m_for_parallel_lines_l75_75776

open Real

theorem find_m_for_parallel_lines :
  ∀ (m : ℝ),
    (∀ x y : ℝ, 3 * x + 4 * y - 3 = 0 → 6 * x + m * y + 14 = 0 → 3 * m = 4 * 6) →
    m = 8 :=
by
  intro m h
  have H : 3 * m = 4 * 6 := h 0 0 sorry sorry
  linarith

end find_m_for_parallel_lines_l75_75776


namespace line_through_points_l75_75917

variable (A1 B1 A2 B2 : ℝ)

def line1 : Prop := -7 * A1 + 9 * B1 = 1
def line2 : Prop := -7 * A2 + 9 * B2 = 1

theorem line_through_points (h1 : line1 A1 B1) (h2 : line1 A2 B2) :
  ∃ (k : ℝ), (∀ (x y : ℝ), y - B1 = k * (x - A1)) ∧ (-7 * (x : ℝ) + 9 * y = 1) := 
by sorry

end line_through_points_l75_75917


namespace perimeter_of_square_l75_75366

-- Definitions based on problem conditions
def is_square_divided_into_four_congruent_rectangles (s : ℝ) (rect_perimeter : ℝ) : Prop :=
  rect_perimeter = 30 ∧ s > 0

-- Statement of the theorem to be proved
theorem perimeter_of_square (s : ℝ) (rect_perimeter : ℝ) (h : is_square_divided_into_four_congruent_rectangles s rect_perimeter) :
  4 * s = 48 :=
by sorry

end perimeter_of_square_l75_75366


namespace slices_per_banana_l75_75165

-- Define conditions
def yogurts : ℕ := 5
def slices_per_yogurt : ℕ := 8
def bananas : ℕ := 4
def total_slices_needed : ℕ := yogurts * slices_per_yogurt

-- Statement to prove
theorem slices_per_banana : total_slices_needed / bananas = 10 := by sorry

end slices_per_banana_l75_75165


namespace anne_wandered_hours_l75_75773

noncomputable def speed : ℝ := 2 -- miles per hour
noncomputable def distance : ℝ := 6 -- miles

theorem anne_wandered_hours (t : ℝ) (h : distance = speed * t) : t = 3 := by
  sorry

end anne_wandered_hours_l75_75773


namespace blueberries_per_basket_l75_75391

-- Definitions based on the conditions
def total_blueberries : ℕ := 200
def total_baskets : ℕ := 10

-- Statement to be proven
theorem blueberries_per_basket : total_blueberries / total_baskets = 20 := 
by
  sorry

end blueberries_per_basket_l75_75391


namespace equation_represents_single_point_l75_75619

theorem equation_represents_single_point (d : ℝ) :
  (∀ x y : ℝ, 3*x^2 + 4*y^2 + 6*x - 8*y + d = 0 ↔ (x = -1 ∧ y = 1)) → d = 7 :=
sorry

end equation_represents_single_point_l75_75619


namespace problem_statement_l75_75535

theorem problem_statement :
  (∀ x : ℝ, |x| < 2 → x < 3) ∧
  (∀ x : ℝ, ¬ (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ∧
  (-1 < m ∧ m < 0 → ∀ a b : ℝ, a ≠ b → (a * b > 0)) :=
by
  sorry

end problem_statement_l75_75535


namespace parabola_intersects_x_axis_two_points_l75_75349

theorem parabola_intersects_x_axis_two_points (m : ℝ) : 
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ mx^2 + (m-3)*x - 1 = 0 :=
by
  sorry

end parabola_intersects_x_axis_two_points_l75_75349


namespace tan_subtraction_l75_75492

theorem tan_subtraction (α β : ℝ) (h₁ : Real.tan α = 11) (h₂ : Real.tan β = 5) : 
  Real.tan (α - β) = 3 / 28 := 
  sorry

end tan_subtraction_l75_75492


namespace problem_l75_75600

theorem problem : 3^128 + 8^5 / 8^3 = 65 := sorry

end problem_l75_75600


namespace Nick_raising_money_l75_75734

theorem Nick_raising_money :
  let chocolate_oranges := 20
  let oranges_price := 10
  let candy_bars := 160
  let bars_price := 5
  let total_amount := chocolate_oranges * oranges_price + candy_bars * bars_price
  total_amount = 1000 := 
by
  sorry

end Nick_raising_money_l75_75734


namespace intersection_M_N_l75_75986

noncomputable def M : Set ℝ := {-1, 0, 1}
noncomputable def N : Set ℝ := {x | x * (x - 2) ≤ 0}

theorem intersection_M_N : M ∩ N = {0, 1} :=
by {
  sorry
}

end intersection_M_N_l75_75986


namespace houses_with_only_one_pet_l75_75205

theorem houses_with_only_one_pet (h_total : ∃ t : ℕ, t = 75)
                                 (h_dogs : ∃ d : ℕ, d = 40)
                                 (h_cats : ∃ c : ℕ, c = 30)
                                 (h_dogs_and_cats : ∃ dc : ℕ, dc = 10)
                                 (h_birds : ∃ b : ℕ, b = 8)
                                 (h_cats_and_birds : ∃ cb : ℕ, cb = 5)
                                 (h_no_dogs_and_birds : ∀ db : ℕ, ¬ (∃ db : ℕ, db = 1)) :
  ∃ n : ℕ, n = 48 :=
by
  have only_dogs := 40 - 10
  have only_cats := 30 - 10 - 5
  have only_birds := 8 - 5
  have result := only_dogs + only_cats + only_birds
  exact ⟨result, sorry⟩

end houses_with_only_one_pet_l75_75205


namespace remainder_when_3_pow_2020_div_73_l75_75455

theorem remainder_when_3_pow_2020_div_73 :
  (3^2020 % 73) = 8 := 
sorry

end remainder_when_3_pow_2020_div_73_l75_75455


namespace sally_eggs_l75_75668

def dozen := 12
def total_eggs := 48

theorem sally_eggs : total_eggs / dozen = 4 := by
  -- Normally a proof would follow here, but we will use sorry to skip it
  sorry

end sally_eggs_l75_75668


namespace final_result_l75_75969

def a : ℕ := 2548
def b : ℕ := 364
def hcd := Nat.gcd a b
def result := hcd + 8 - 12

theorem final_result : result = 360 := by
  sorry

end final_result_l75_75969


namespace Mika_water_left_l75_75818

theorem Mika_water_left :
  let initial_amount := 7 / 2
  let used_amount := 9 / 4
  initial_amount - used_amount = 5 / 4 :=
by
  let initial_amount := 7 / 2
  let used_amount := 9 / 4
  show initial_amount - used_amount = 5 / 4
  sorry

end Mika_water_left_l75_75818


namespace least_positive_integer_solution_l75_75231

theorem least_positive_integer_solution :
  ∃ b : ℕ, b ≡ 1 [MOD 4] ∧ b ≡ 2 [MOD 5] ∧ b ≡ 3 [MOD 6] ∧ b = 37 :=
by
  sorry

end least_positive_integer_solution_l75_75231


namespace total_area_of_removed_triangles_l75_75018

theorem total_area_of_removed_triangles (side_length : ℝ) (half_leg_length : ℝ) :
  side_length = 16 →
  half_leg_length = side_length / 4 →
  4 * (1 / 2) * half_leg_length^2 = 32 :=
by
  intro h_side_length h_half_leg_length
  simp [h_side_length, h_half_leg_length]
  sorry

end total_area_of_removed_triangles_l75_75018


namespace two_digit_divisors_1995_l75_75137

theorem two_digit_divisors_1995 :
  (∃ (n : Finset ℕ), (∀ x ∈ n, 10 ≤ x ∧ x < 100 ∧ 1995 % x = 0) ∧ n.card = 6 ∧ ∃ y ∈ n, y = 95) :=
by
  sorry

end two_digit_divisors_1995_l75_75137


namespace acceleration_inverse_square_distance_l75_75218

noncomputable def s (t : ℝ) : ℝ := t^(2/3)

noncomputable def v (t : ℝ) : ℝ := (deriv s t : ℝ)

noncomputable def a (t : ℝ) : ℝ := (deriv v t : ℝ)

theorem acceleration_inverse_square_distance
  (t : ℝ) (h : t ≠ 0) :
  ∃ k : ℝ, k = -2/9 ∧ a t = k / (s t)^2 :=
sorry

end acceleration_inverse_square_distance_l75_75218


namespace average_speed_of_train_l75_75607

theorem average_speed_of_train
  (d1 d2 : ℝ) (t1 t2 : ℝ)
  (h1 : d1 = 290) (h2 : d2 = 400) (h3 : t1 = 4.5) (h4 : t2 = 5.5) :
  ((d1 + d2) / (t1 + t2)) = 69 :=
by
  -- proof steps can be filled in later
  sorry

end average_speed_of_train_l75_75607


namespace Niko_total_profit_l75_75720

-- Definitions based on conditions
def cost_per_pair : ℕ := 2
def total_pairs : ℕ := 9
def profit_margin_4_pairs : ℚ := 0.25
def profit_per_other_pair : ℚ := 0.2
def pairs_with_margin : ℕ := 4
def pairs_with_fixed_profit : ℕ := 5

-- Calculations based on definitions
def total_cost : ℚ := total_pairs * cost_per_pair
def profit_on_margin_pairs : ℚ := pairs_with_margin * (profit_margin_4_pairs * cost_per_pair)
def profit_on_fixed_profit_pairs : ℚ := pairs_with_fixed_profit * profit_per_other_pair
def total_profit : ℚ := profit_on_margin_pairs + profit_on_fixed_profit_pairs

-- Statement to prove
theorem Niko_total_profit : total_profit = 3 := by
  sorry

end Niko_total_profit_l75_75720


namespace prove_axisymmetric_char4_l75_75775

-- Predicates representing whether a character is an axisymmetric figure
def is_axisymmetric (ch : Char) : Prop := sorry

-- Definitions for the conditions given in the problem
def char1 := '月'
def char2 := '右'
def char3 := '同'
def char4 := '干'

-- Statement that needs to be proven
theorem prove_axisymmetric_char4 (h1 : ¬ is_axisymmetric char1) 
                                  (h2 : ¬ is_axisymmetric char2) 
                                  (h3 : ¬ is_axisymmetric char3) : 
                                  is_axisymmetric char4 :=
sorry

end prove_axisymmetric_char4_l75_75775


namespace smaller_number_is_24_l75_75400

theorem smaller_number_is_24 (x y : ℕ) (h1 : x + y = 44) (h2 : 5 * x = 6 * y) : x = 24 :=
by
  sorry

end smaller_number_is_24_l75_75400


namespace contractor_engaged_days_l75_75569

variable (x : ℕ)
variable (days_absent : ℕ) (wage_per_day fine_per_day_Rs total_payment_Rs : ℝ)

theorem contractor_engaged_days :
  days_absent = 10 →
  wage_per_day = 25 →
  fine_per_day_Rs = 7.5 →
  total_payment_Rs = 425 →
  (x * wage_per_day - days_absent * fine_per_day_Rs = total_payment_Rs) →
  x = 20 :=
by
  sorry

end contractor_engaged_days_l75_75569


namespace min_value_5_5_l75_75493

noncomputable def given_expression (x y z : ℝ) : ℝ :=
  (6 * z) / (2 * x + y) + (6 * x) / (y + 2 * z) + (4 * y) / (x + z)

theorem min_value_5_5 (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x + y + z = 1) :
  given_expression x y z ≥ 5.5 :=
sorry

end min_value_5_5_l75_75493


namespace evaluate_polynomial_l75_75613

theorem evaluate_polynomial (x : ℝ) (h : x^2 - 3 * x - 9 = 0) (hx : 0 < x) : 
  x^4 - 3 * x^3 - 9 * x^2 + 27 * x - 8 = -8 := 
sorry

end evaluate_polynomial_l75_75613


namespace eq_positive_root_a_value_l75_75605

theorem eq_positive_root_a_value (x a : ℝ) (hx : x > 0) :
  ((x + a) / (x + 3) - 2 / (x + 3) = 0) → a = 5 :=
by
  sorry

end eq_positive_root_a_value_l75_75605


namespace ab_sum_l75_75532

theorem ab_sum (a b : ℝ) (h₁ : ∀ x : ℝ, (x + a) * (x + 8) = x^2 + b * x + 24) (h₂ : 8 * a = 24) : a + b = 14 :=
by
  sorry

end ab_sum_l75_75532


namespace max_value_func1_l75_75970

theorem max_value_func1 (x : ℝ) (h : 0 < x ∧ x < 2) : 
  ∃ y, y = x * (4 - 2 * x) ∧ (∀ z, z = x * (4 - 2 * x) → z ≤ 2) :=
sorry

end max_value_func1_l75_75970


namespace solve_for_a_and_b_range_of_f_when_x_lt_zero_l75_75302

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (1 + a * (2 ^ x)) / (2 ^ x + b)

theorem solve_for_a_and_b (a b : ℝ) :
  f a b 1 = 3 ∧
  f a b (-1) = -3 →
  a = 1 ∧ b = -1 :=
by
  sorry

theorem range_of_f_when_x_lt_zero (x : ℝ) :
  ∀ x < 0, f 1 (-1) x < -1 :=
by 
  sorry

end solve_for_a_and_b_range_of_f_when_x_lt_zero_l75_75302


namespace leak_empties_cistern_in_12_hours_l75_75725

theorem leak_empties_cistern_in_12_hours 
  (R : ℝ) (L : ℝ)
  (h1 : R = 1 / 4) 
  (h2 : R - L = 1 / 6) : 
  1 / L = 12 := 
by
  -- proof will go here
  sorry

end leak_empties_cistern_in_12_hours_l75_75725


namespace three_hour_classes_per_week_l75_75800

theorem three_hour_classes_per_week (x : ℕ) : 
  (24 * (3 * x + 4 + 4) = 336) → x = 2 := by {
  sorry
}

end three_hour_classes_per_week_l75_75800


namespace equal_areas_greater_perimeter_l75_75789

noncomputable def side_length_square := Real.sqrt 3 + 3

noncomputable def length_rectangle := Real.sqrt 72 + 3 * Real.sqrt 6
noncomputable def width_rectangle := Real.sqrt 2

noncomputable def area_square := (side_length_square) ^ 2

noncomputable def area_rectangle := length_rectangle * width_rectangle

noncomputable def perimeter_square := 4 * side_length_square

noncomputable def perimeter_rectangle := 2 * (length_rectangle + width_rectangle)

theorem equal_areas : area_square = area_rectangle := sorry

theorem greater_perimeter : perimeter_square < perimeter_rectangle := sorry

end equal_areas_greater_perimeter_l75_75789


namespace right_triangle_area_l75_75716

theorem right_triangle_area (base hypotenuse : ℕ) (h_base : base = 8) (h_hypotenuse : hypotenuse = 10) :
  ∃ height : ℕ, height^2 = hypotenuse^2 - base^2 ∧ (base * height) / 2 = 24 :=
by
  sorry

end right_triangle_area_l75_75716


namespace num_ordered_triples_l75_75025

theorem num_ordered_triples 
  (a b c : ℕ)
  (h_cond1 : 1 ≤ a ∧ a ≤ b ∧ b ≤ c)
  (h_cond2 : a * b * c = 4 * (a * b + b * c + c * a)) : 
  ∃ (n : ℕ), n = 5 :=
sorry

end num_ordered_triples_l75_75025


namespace symmetric_function_exists_l75_75361

-- Define the main sets A and B with given cardinalities
def A := { n : ℕ // n < 2011^2 }
def B := { n : ℕ // n < 2010 }

-- The main theorem to prove
theorem symmetric_function_exists :
  ∃ (f : A × A → B), 
  (∀ x y, f (x, y) = f (y, x)) ∧ 
  (∀ g : A → B, ∃ (a1 a2 : A), g a1 = f (a1, a2) ∧ g a2 = f (a1, a2) ∧ a1 ≠ a2) :=
sorry

end symmetric_function_exists_l75_75361


namespace remaining_length_after_cut_l75_75272

/- Definitions -/
def original_length (a b : ℕ) : ℕ := 5 * a + 4 * b
def rectangle_perimeter (a b : ℕ) : ℕ := 2 * (a + b)
def remaining_length (a b : ℕ) : ℕ := original_length a b - rectangle_perimeter a b

/- Theorem statement -/
theorem remaining_length_after_cut (a b : ℕ) : remaining_length a b = 3 * a + 2 * b := 
by 
  sorry

end remaining_length_after_cut_l75_75272


namespace wendy_facial_products_l75_75277

def total_time (P : ℕ) : ℕ :=
  5 * (P - 1) + 30

theorem wendy_facial_products :
  (total_time 6 = 55) :=
by
  sorry

end wendy_facial_products_l75_75277


namespace test_group_type_A_probability_atleast_one_type_A_group_probability_l75_75044

noncomputable def probability_type_A_group : ℝ :=
  let pA := 2 / 3
  let pB := 1 / 2
  let P_A1 := 2 * (1 - pA) * pA
  let P_A2 := pA * pA
  let P_B0 := (1 - pB) * (1 - pB)
  let P_B1 := 2 * (1 - pB) * pB
  P_B0 * P_A1 + P_B0 * P_A2 + P_B1 * P_A2

theorem test_group_type_A_probability :
  probability_type_A_group = 4 / 9 :=
by
  sorry

noncomputable def at_least_one_type_A_in_3_groups : ℝ :=
  let P_type_A_group := 4 / 9
  1 - (1 - P_type_A_group) ^ 3

theorem atleast_one_type_A_group_probability :
  at_least_one_type_A_in_3_groups = 604 / 729 :=
by
  sorry

end test_group_type_A_probability_atleast_one_type_A_group_probability_l75_75044


namespace stickers_left_correct_l75_75163

-- Define the initial number of stickers and number of stickers given away
def n_initial : ℝ := 39.0
def n_given_away : ℝ := 22.0

-- Proof statement: The number of stickers left at the end is 17.0
theorem stickers_left_correct : n_initial - n_given_away = 17.0 := by
  sorry

end stickers_left_correct_l75_75163


namespace no_representation_of_216p3_l75_75004

theorem no_representation_of_216p3 (p : ℕ) (hp_prime : Nat.Prime p)
  (hp_form : ∃ m : ℤ, p = 4 * m + 1) : ¬ ∃ x y z : ℤ, 216 * (p ^ 3) = x^2 + y^2 + z^9 := by
  sorry

end no_representation_of_216p3_l75_75004


namespace problem_1_problem_2_l75_75416

open Set Real

noncomputable def A : Set ℝ := {x | x^2 - 3 * x - 18 ≤ 0}

noncomputable def B (m : ℝ) : Set ℝ := {x | m - 8 ≤ x ∧ x ≤ m + 4}

theorem problem_1 : (m = 3) → ((compl A) ∩ (B m) = {x | (-5 ≤ x ∧ x < -3) ∨ (6 < x ∧ x ≤ 7)}) :=
by
  sorry

theorem problem_2 : (A ∩ (B m) = A) → (2 ≤ m ∧ m ≤ 5) :=
by
  sorry

end problem_1_problem_2_l75_75416


namespace inscribed_circle_radius_l75_75288

theorem inscribed_circle_radius (a b c r : ℝ) (h : a^2 + b^2 = c^2) (h' : r = (a + b - c) / 2) : r = (a + b - c) / 2 :=
by
  sorry

end inscribed_circle_radius_l75_75288


namespace value_of_x_squared_plus_reciprocal_squared_l75_75953

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (h : x^4 + (1 / x^4) = 23) :
  x^2 + (1 / x^2) = 5 := by
  sorry

end value_of_x_squared_plus_reciprocal_squared_l75_75953


namespace pizza_share_l75_75727

theorem pizza_share :
  forall (friends : ℕ) (leftover_pizza : ℚ), friends = 4 -> leftover_pizza = 5/6 -> (leftover_pizza / friends) = (5 / 24) :=
by
  intros friends leftover_pizza h_friends h_leftover_pizza
  sorry

end pizza_share_l75_75727


namespace ship_distances_l75_75635

-- Define the conditions based on the initial problem statement
variables (f : ℕ → ℝ)
def distances_at_known_times : Prop :=
  f 0 = 49 ∧ f 2 = 25 ∧ f 3 = 121

-- Define the questions to prove the distances at unknown times
def distance_at_time_1 : Prop :=
  f 1 = 1

def distance_at_time_4 : Prop :=
  f 4 = 289

-- The proof problem
theorem ship_distances
  (f : ℕ → ℝ)
  (hf : ∀ t, ∃ a b c, f t = a*t^2 + b*t + c)
  (h_known : distances_at_known_times f) :
  distance_at_time_1 f ∧ distance_at_time_4 f :=
by
  sorry

end ship_distances_l75_75635


namespace range_of_a_l75_75567

theorem range_of_a (a : ℚ) (h₀ : 0 < a) (h₁ : ∃ n : ℕ, (2 * n - 1 = 2007) ∧ (-a < n ∧ n < a)) :
  1003 < a ∧ a ≤ 1004 :=
sorry

end range_of_a_l75_75567


namespace max_distance_l75_75292

-- Definition of curve C₁ in rectangular coordinates.
def C₁_rectangular (x y : ℝ) : Prop := x^2 + y^2 - 2 * y = 0

-- Definition of curve C₂ in its general form.
def C₂_general (x y : ℝ) : Prop := 4 * x + 3 * y - 8 = 0

-- Coordinates of point M, the intersection of C₂ with x-axis.
def M : ℝ × ℝ := (2, 0)

-- Condition that N is a moving point on curve C₁.
def N (x y : ℝ) : Prop := C₁_rectangular x y

-- Maximum distance |MN|.
theorem max_distance (x y : ℝ) (hN : N x y) : 
  dist (2, 0) (x, y) ≤ Real.sqrt 5 + 1 := by
  sorry

end max_distance_l75_75292


namespace number_of_cheeses_per_pack_l75_75270

-- Definitions based on the conditions
def packs : ℕ := 3
def cost_per_cheese : ℝ := 0.10
def total_amount_paid : ℝ := 6

-- Theorem statement to prove the number of string cheeses in each pack
theorem number_of_cheeses_per_pack : 
  (total_amount_paid / (packs : ℝ)) / cost_per_cheese = 20 :=
sorry

end number_of_cheeses_per_pack_l75_75270


namespace inequality_system_solution_exists_l75_75925

theorem inequality_system_solution_exists (a : ℝ) : (∃ x : ℝ, x ≥ -1 ∧ 2 * x < a) ↔ a > -2 := 
sorry

end inequality_system_solution_exists_l75_75925


namespace gasoline_storage_l75_75704

noncomputable def total_distance : ℕ := 280 * 2

noncomputable def miles_per_segment : ℕ := 40

noncomputable def gasoline_consumption : ℕ := 8

noncomputable def total_segments : ℕ := total_distance / miles_per_segment

noncomputable def total_gasoline : ℕ := total_segments * gasoline_consumption

noncomputable def number_of_refills : ℕ := 14

theorem gasoline_storage (storage_capacity : ℕ) (h : number_of_refills * storage_capacity = total_gasoline) :
  storage_capacity = 8 :=
by
  sorry

end gasoline_storage_l75_75704


namespace percentage_increase_x_y_l75_75579

theorem percentage_increase_x_y (Z Y X : ℝ) (h1 : Z = 300) (h2 : Y = 1.20 * Z) (h3 : X = 1110 - Y - Z) :
  ((X - Y) / Y) * 100 = 25 :=
by
  sorry

end percentage_increase_x_y_l75_75579


namespace area_of_shaded_region_l75_75353

theorem area_of_shaded_region : 
  let side_length := 4
  let radius := side_length / 2 
  let area_of_square := side_length * side_length 
  let area_of_one_quarter_circle := (pi * radius * radius) / 4
  let total_area_of_quarter_circles := 4 * area_of_one_quarter_circle 
  let area_of_shaded_region := area_of_square - total_area_of_quarter_circles 
  area_of_shaded_region = 16 - 4 * pi :=
by
  let side_length := 4
  let radius := side_length / 2
  let area_of_square := side_length * side_length
  let area_of_one_quarter_circle := (pi * radius * radius) / 4
  let total_area_of_quarter_circles := 4 * area_of_one_quarter_circle
  let area_of_shaded_region := area_of_square - total_area_of_quarter_circles
  sorry

end area_of_shaded_region_l75_75353


namespace find_x_l75_75229

noncomputable def eq_num (x : ℝ) : Prop :=
  9 - 3 / (1 / 3) + x = 3

theorem find_x : ∃ x : ℝ, eq_num x ∧ x = 3 := 
by
  sorry

end find_x_l75_75229


namespace lambda_phi_relation_l75_75169

-- Define the context and conditions
variables (A B C D M N : Type) -- Points on the triangle with D being the midpoint of BC
variables (AB AC BC BN BM MN : ℝ) -- Lengths
variables (lambda phi : ℝ) -- Ratios given in the problem

-- Conditions
-- 1. M is a point on the median AD of triangle ABC
variable (h1 : M = D ∨ M = A ∨ M = D) -- Simplified condition stating M's location
-- 2. The line BM intersects the side AC at point N
variable (h2 : N = M ∧ N ≠ A ∧ N ≠ C) -- Defining the intersection point
-- 3. AB is tangent to the circumcircle of triangle NBC
variable (h3 : tangent AB (circumcircle N B C))
-- 4. BC = lambda BN
variable (h4 : BC = lambda * BN)
-- 5. BM = phi * MN
variable (h5 : BM = phi * MN)

-- Goal
theorem lambda_phi_relation : phi = lambda ^ 2 :=
sorry

end lambda_phi_relation_l75_75169


namespace find_side_b_l75_75019

theorem find_side_b
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : 2 * Real.sin B = Real.sin A + Real.sin C)
  (h2 : Real.cos B = 3 / 5)
  (h3 : (1 / 2) * a * c * Real.sin B = 4) :
  b = 4 * Real.sqrt 6 / 3 := 
sorry

end find_side_b_l75_75019


namespace trig_identity_equiv_l75_75921

theorem trig_identity_equiv (α : ℝ) (h : Real.sin (Real.pi - α) = -2 * Real.cos (-α)) : 
  Real.sin (2 * α) - Real.cos α ^ 2 = -1 :=
by
  sorry

end trig_identity_equiv_l75_75921


namespace total_cats_correct_l75_75440

-- Jamie's cats
def Jamie_Persian_cats : ℕ := 4
def Jamie_Maine_Coons : ℕ := 2

-- Gordon's cats
def Gordon_Persian_cats : ℕ := Jamie_Persian_cats / 2
def Gordon_Maine_Coons : ℕ := Jamie_Maine_Coons + 1

-- Hawkeye's cats
def Hawkeye_Persian_cats : ℕ := 0
def Hawkeye_Maine_Coons : ℕ := Gordon_Maine_Coons - 1

-- Total cats for each person
def Jamie_total_cats : ℕ := Jamie_Persian_cats + Jamie_Maine_Coons
def Gordon_total_cats : ℕ := Gordon_Persian_cats + Gordon_Maine_Coons
def Hawkeye_total_cats : ℕ := Hawkeye_Persian_cats + Hawkeye_Maine_Coons

-- Proof that the total number of cats is 13
theorem total_cats_correct : Jamie_total_cats + Gordon_total_cats + Hawkeye_total_cats = 13 :=
by sorry

end total_cats_correct_l75_75440


namespace perpendicular_sum_value_of_m_l75_75037

-- Let a and b be defined as vectors in R^2
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (2, m)

-- Define the dot product for vectors in R^2
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the condition for perpendicular vectors using dot product
def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

-- Define the sum of two vectors
def vector_sum (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- State our proof problem
theorem perpendicular_sum_value_of_m :
  is_perpendicular (vector_sum vector_a (vector_b (-7 / 2))) vector_a :=
by
  -- Proof omitted
  sorry

end perpendicular_sum_value_of_m_l75_75037


namespace Mike_ride_distance_l75_75897

theorem Mike_ride_distance 
  (M : ℕ)
  (total_cost_Mike : ℝ)
  (total_cost_Annie : ℝ)
  (h1 : total_cost_Mike = 4.50 + 0.30 * M)
  (h2: total_cost_Annie = 15.00)
  (h3: total_cost_Mike = total_cost_Annie) : 
  M = 35 := 
by
  sorry

end Mike_ride_distance_l75_75897


namespace jill_llamas_count_l75_75451

theorem jill_llamas_count : 
  let initial_llamas := 9
  let pregnant_with_twins := 5
  let calves_from_single := initial_llamas
  let calves_from_twins := pregnant_with_twins * 2
  let initial_herd := initial_llamas + pregnant_with_twins + calves_from_single + calves_from_twins
  let traded_calves := 8
  let new_adult_llamas := 2
  let herd_after_trade := initial_herd - traded_calves + new_adult_llamas
  let sell_fraction := 1 / 3
  let herd_after_sell := herd_after_trade - (herd_after_trade * sell_fraction)
  herd_after_sell = 18 := 
by
  -- Definitions for the conditions
  let initial_llamas := 9
  let pregnant_with_twins := 5
  let calves_from_single := initial_llamas
  let calves_from_twins := pregnant_with_twins * 2
  let initial_herd := initial_llamas + pregnant_with_twins + calves_from_single + calves_from_twins
  let traded_calves := 8
  let new_adult_llamas := 2
  let herd_after_trade := initial_herd - traded_calves + new_adult_llamas
  let sell_fraction := 1 / 3
  let herd_after_sell := herd_after_trade - (herd_after_trade * sell_fraction)
  -- Proof will be filled in here.
  sorry

end jill_llamas_count_l75_75451


namespace daily_profit_at_35_yuan_selling_price_for_600_profit_selling_price_impossible_for_900_profit_l75_75609

-- Definitions based on given conditions
noncomputable def purchase_price : ℝ := 30
noncomputable def max_selling_price : ℝ := 55
noncomputable def daily_sales_volume (x : ℝ) : ℝ := -2 * x + 140

-- Definition of daily profit based on selling price x
noncomputable def daily_profit (x : ℝ) : ℝ := (x - purchase_price) * daily_sales_volume x

-- Lean 4 statements for the proofs
theorem daily_profit_at_35_yuan : daily_profit 35 = 350 := sorry

theorem selling_price_for_600_profit : ∃ x, 30 ≤ x ∧ x ≤ 55 ∧ daily_profit x = 600 ∧ x = 40 := sorry

theorem selling_price_impossible_for_900_profit :
  ∀ x, 30 ≤ x ∧ x ≤ 55 → daily_profit x ≠ 900 := sorry

end daily_profit_at_35_yuan_selling_price_for_600_profit_selling_price_impossible_for_900_profit_l75_75609


namespace situps_together_l75_75814

theorem situps_together (hani_rate diana_rate : ℕ) (diana_situps diana_time hani_situps total_situps : ℕ)
  (h1 : hani_rate = diana_rate + 3)
  (h2 : diana_rate = 4)
  (h3 : diana_situps = 40)
  (h4 : diana_time = diana_situps / diana_rate)
  (h5 : hani_situps = hani_rate * diana_time)
  (h6 : total_situps = diana_situps + hani_situps) : 
  total_situps = 110 :=
sorry

end situps_together_l75_75814


namespace value_of_expression_l75_75180

theorem value_of_expression
  (a b x y : ℝ)
  (h1 : a + b = 0)
  (h2 : x * y = 1) : 
  2 * (a + b) + (7 / 4) * (x * y) = 7 / 4 := 
sorry

end value_of_expression_l75_75180


namespace sequoia_taller_than_maple_l75_75256

def height_maple_tree : ℚ := 13 + 3/4
def height_sequoia : ℚ := 20 + 1/2

theorem sequoia_taller_than_maple : (height_sequoia - height_maple_tree) = 6 + 3/4 :=
by
  sorry

end sequoia_taller_than_maple_l75_75256


namespace computation_equal_l75_75959

theorem computation_equal (a b c d : ℕ) (inv : ℚ → ℚ) (mul : ℚ → ℕ → ℚ) : 
  a = 3 → b = 1 → c = 6 → d = 2 → 
  inv ((a^b - d + c^2 + b) : ℚ) * 6 = (3 / 19) := by
  intros ha hb hc hd
  rw [ha, hb, hc, hd]
  sorry

end computation_equal_l75_75959


namespace measure_α_l75_75186

noncomputable def measure_α_proof (AB BC : ℝ) (h1: AB = 1) (h2 : BC = 2) : ℝ :=
  let α := 120
  α

theorem measure_α (AB BC : ℝ) (h1: AB = 1) (h2 : BC = 2) : measure_α_proof AB BC h1 h2 = 120 :=
  sorry

end measure_α_l75_75186


namespace x_squared_y_squared_iff_x_squared_y_squared_not_sufficient_x_squared_y_squared_necessary_l75_75011

theorem x_squared_y_squared_iff (x y : ℝ) : x ^ 2 = y ^ 2 ↔ x = y ∨ x = -y := by
  sorry

theorem x_squared_y_squared_not_sufficient (x y : ℝ) : (x ^ 2 = y ^ 2) → (x = y ∨ x = -y) := by
  sorry

theorem x_squared_y_squared_necessary (x y : ℝ) : (x = y) → (x ^ 2 = y ^ 2) := by
  sorry

end x_squared_y_squared_iff_x_squared_y_squared_not_sufficient_x_squared_y_squared_necessary_l75_75011


namespace area_ratio_l75_75311

theorem area_ratio (A B C D E : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  (AB BC AC AD AE : ℝ) (ADE_ratio : ℝ) :
  AB = 25 ∧ BC = 39 ∧ AC = 42 ∧ AD = 19 ∧ AE = 14 →
  ADE_ratio = 19 / 56 :=
by sorry

end area_ratio_l75_75311


namespace largest_prime_factor_of_1729_l75_75602

def prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_factor_of_1729 :
  1729 = 13 * 7 * 19 ∧ prime 13 ∧ prime 7 ∧ prime 19 → ∃ p, prime p ∧ p ∣ 1729 ∧
  ∀ q, prime q → q ∣ 1729 → q ≤ p :=
  by
  sorry

end largest_prime_factor_of_1729_l75_75602


namespace distance_travelled_first_hour_l75_75823

noncomputable def initial_distance (x : ℕ) : Prop :=
  let distance_travelled := (12 / 2) * (2 * x + (12 - 1) * 2)
  distance_travelled = 552

theorem distance_travelled_first_hour : ∃ x : ℕ, initial_distance x ∧ x = 35 :=
by
  use 35
  unfold initial_distance
  sorry

end distance_travelled_first_hour_l75_75823


namespace increase_80_by_50_percent_l75_75749

theorem increase_80_by_50_percent :
  let initial_number : ℕ := 80
  let increase_percentage : ℝ := 0.5
  initial_number + (initial_number * increase_percentage) = 120 :=
by
  sorry

end increase_80_by_50_percent_l75_75749


namespace Mark_average_speed_l75_75200

theorem Mark_average_speed 
  (start_time : ℝ) (end_time : ℝ) (distance : ℝ)
  (h1 : start_time = 8.5) (h2 : end_time = 14.75) (h3 : distance = 210) :
  distance / (end_time - start_time) = 33.6 :=
by 
  sorry

end Mark_average_speed_l75_75200


namespace problem_statement_l75_75245

theorem problem_statement (a b c x y z : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < x) (h5 : 0 < y) (h6 : 0 < z)
  (h7 : a^2 + b^2 + c^2 = 16) (h8 : x^2 + y^2 + z^2 = 49) (h9 : a * x + b * y + c * z = 28) : 
  (a + b + c) / (x + y + z) = 4 / 7 := 
by
  sorry

end problem_statement_l75_75245


namespace octagon_area_sum_l75_75447

theorem octagon_area_sum :
  let A1 := 2024
  let a := 1012
  let b := 506
  let c := 2
  a + b + c = 1520 := by
    sorry

end octagon_area_sum_l75_75447


namespace TimSpentTotal_l75_75742

variable (LunchCost : ℝ) (TipPercentage : ℝ)

def TotalAmountSpent (LunchCost : ℝ) (TipPercentage : ℝ) : ℝ := 
  LunchCost + (LunchCost * TipPercentage)

theorem TimSpentTotal (h1 : LunchCost = 50.50) (h2 : TipPercentage = 0.20) :
  TotalAmountSpent LunchCost TipPercentage = 60.60 := by
  sorry

end TimSpentTotal_l75_75742


namespace ellipse_to_parabola_standard_eq_l75_75951

theorem ellipse_to_parabola_standard_eq :
  ∀ (x y : ℝ), (x^2 / 25 + y^2 / 16 = 1) → (y^2 = 12 * x) :=
by
  sorry

end ellipse_to_parabola_standard_eq_l75_75951


namespace minimum_value_l75_75201

noncomputable def min_value_of_expression (a b c : ℝ) : ℝ :=
  1/a + 2/b + 4/c

theorem minimum_value (a b c : ℝ) (h₀ : c > 0) (h₁ : a ≠ 0) (h₂ : b ≠ 0)
    (h₃ : 4 * a^2 - 2 * a * b + b^2 - c = 0)
    (h₄ : ∀ x y, 4*x^2 - 2*x*y + y^2 - c = 0 → |2*x + y| ≤ |2*a + b|)
    : min_value_of_expression a b c = -1 :=
sorry

end minimum_value_l75_75201


namespace checkerboard_pattern_exists_l75_75930

-- Definitions for the given conditions
def is_black_white_board (board : ℕ × ℕ → Prop) (n : ℕ) : Prop :=
  ∀ i j, i < n ∧ j < n → (board (i, j) = true ∨ board (i, j) = false)

def boundary_cells_black (board : ℕ × ℕ → Prop) (n : ℕ) : Prop :=
  ∀ i, (i < n → (board (i, 0) = true ∧ board (i, n-1) = true ∧ 
                  board (0, i) = true ∧ board (n-1, i) = true))

def no_monochromatic_square (board : ℕ × ℕ → Prop) (n : ℕ) : Prop :=
  ∀ i j, i < n-1 ∧ j < n-1 → ¬(board (i, j) = board (i+1, j) ∧ 
                               board (i, j) = board (i, j+1) ∧ 
                               board (i, j) = board (i+1, j+1))

def exists_checkerboard_2x2 (board : ℕ × ℕ → Prop) (n : ℕ) : Prop :=
  ∃ i j, i < n-1 ∧ j < n-1 ∧ 
         (board (i, j) ≠ board (i+1, j) ∧ board (i, j) ≠ board (i, j+1) ∧ 
          board (i+1, j) ≠ board (i+1, j+1) ∧ board (i, j+1) ≠ board (i+1, j+1))

-- The theorem statement
theorem checkerboard_pattern_exists (board : ℕ × ℕ → Prop) (n : ℕ) 
  (coloring : is_black_white_board board n)
  (boundary_black : boundary_cells_black board n)
  (no_mono_2x2 : no_monochromatic_square board n) : 
  exists_checkerboard_2x2 board n :=
by
  sorry

end checkerboard_pattern_exists_l75_75930


namespace insurance_percentage_l75_75323

noncomputable def total_pills_per_year : ℕ := 2 * 365

noncomputable def cost_per_pill : ℕ := 5

noncomputable def total_medication_cost_per_year : ℕ := total_pills_per_year * cost_per_pill

noncomputable def doctor_visits_per_year : ℕ := 2

noncomputable def cost_per_doctor_visit : ℕ := 400

noncomputable def total_doctor_cost_per_year : ℕ := doctor_visits_per_year * cost_per_doctor_visit

noncomputable def total_yearly_cost_without_insurance : ℕ := total_medication_cost_per_year + total_doctor_cost_per_year

noncomputable def total_payment_per_year : ℕ := 1530

noncomputable def insurance_coverage_per_year : ℕ := total_yearly_cost_without_insurance - total_payment_per_year

theorem insurance_percentage:
  (insurance_coverage_per_year * 100) / total_medication_cost_per_year = 80 :=
by sorry

end insurance_percentage_l75_75323


namespace min_xy_of_conditions_l75_75903

open Real

theorem min_xy_of_conditions
  (x y : ℝ)
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy_eq : 1 / (2 + x) + 1 / (2 + y) = 1 / 3) : 
  xy ≥ 16 :=
by
  sorry

end min_xy_of_conditions_l75_75903


namespace value_of_expression_l75_75392

variables {A B C : ℚ}

def conditions (A B C : ℚ) : Prop := A / B = 3 / 2 ∧ B / C = 2 / 5

theorem value_of_expression (h : conditions A B C) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
sorry

end value_of_expression_l75_75392


namespace part1_part2_l75_75706

theorem part1 (a : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = |2 * x - a| + |x - 1|) :
  (∀ x, f x + |x - 1| ≥ 2) → (a ≤ 0 ∨ a ≥ 4) :=
by sorry

theorem part2 (a : ℝ) (f : ℝ → ℝ) (h_a : a < 2) (h_f : ∀ x, f x = |2 * x - a| + |x - 1|) :
  (∀ x, f x ≥ a - 1) → (a = 4 / 3) :=
by sorry

end part1_part2_l75_75706


namespace arithmetic_square_root_of_16_is_4_l75_75003

theorem arithmetic_square_root_of_16_is_4 : ∃ x : ℤ, x * x = 16 ∧ x = 4 := 
sorry

end arithmetic_square_root_of_16_is_4_l75_75003


namespace average_age_of_new_men_is_30_l75_75507

noncomputable def average_age_of_two_new_men (A : ℝ) : ℝ :=
  let total_age_before : ℝ := 8 * A
  let total_age_after : ℝ := 8 * (A + 2)
  let age_of_replaced_men : ℝ := 21 + 23
  let total_age_of_new_men : ℝ := total_age_after - total_age_before + age_of_replaced_men
  total_age_of_new_men / 2

theorem average_age_of_new_men_is_30 (A : ℝ) : 
  average_age_of_two_new_men A = 30 :=
by 
  sorry

end average_age_of_new_men_is_30_l75_75507


namespace lucas_fib_relation_l75_75425

noncomputable def α := (1 + Real.sqrt 5) / 2
noncomputable def β := (1 - Real.sqrt 5) / 2
def Fib : ℕ → ℝ
| 0       => 0
| 1       => 1
| (n + 2) => Fib n + Fib (n + 1)

def Lucas : ℕ → ℝ
| 0       => 2
| 1       => 1
| (n + 2) => Lucas n + Lucas (n + 1)

theorem lucas_fib_relation (n : ℕ) (hn : 1 ≤ n) :
  Lucas (2 * n + 1) + (-1)^(n+1) = Fib (2 * n) * Fib (2 * n + 1) := sorry

end lucas_fib_relation_l75_75425


namespace correct_answers_proof_l75_75298

variable (n p q s c : ℕ)
variable (total_questions points_per_correct penalty_per_wrong total_score correct_answers : ℕ)

def num_questions := 20
def points_correct := 5
def penalty_wrong := 1
def total_points := 76

theorem correct_answers_proof :
  (total_questions * points_per_correct - (total_questions - correct_answers) * penalty_wrong) = total_points →
  correct_answers = 16 :=
by {
  sorry
}

end correct_answers_proof_l75_75298


namespace not_divisible_l75_75031

theorem not_divisible (x y : ℕ) (hx : x % 61 ≠ 0) (hy : y % 61 ≠ 0) (h : (7 * x + 34 * y) % 61 = 0) : (5 * x + 16 * y) % 61 ≠ 0 := 
sorry

end not_divisible_l75_75031


namespace solve_equation_l75_75326

theorem solve_equation (x y z : ℤ) (h : 19 * (x + y) + z = 19 * (-x + y) - 21) (hx : x = 1) : z = -59 := by
  sorry

end solve_equation_l75_75326


namespace find_c_l75_75141

theorem find_c (c : ℝ) :
  (∀ x : ℝ, -2 * x^2 + c * x - 8 < 0 ↔ x ∈ Set.Iio 2 ∪ Set.Ioi 6) → c = 16 :=
by
  intros h
  sorry

end find_c_l75_75141


namespace problem_to_prove_l75_75105

theorem problem_to_prove
  (α : ℝ)
  (h : Real.sin (3 * Real.pi / 2 + α) = 1 / 3) :
  Real.cos (Real.pi - 2 * α) = -7 / 9 :=
by
  sorry -- proof required

end problem_to_prove_l75_75105


namespace count_sums_of_fours_and_fives_l75_75547

theorem count_sums_of_fours_and_fives :
  ∃ n, (∀ x y : ℕ, 4 * x + 5 * y = 1800 ↔ (x = 0 ∨ x ≤ 1800) ∧ (y = 0 ∨ y ≤ 1800)) ∧ n = 201 :=
by
  -- definition and theorem statement is complete. The proof is omitted.
  sorry

end count_sums_of_fours_and_fives_l75_75547


namespace solve_equation_l75_75649

variable {x y : ℝ}

theorem solve_equation (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2: y ≠ 4) (h : (3 / x) + (2 / y) = 5 / 6) :
  x = 18 * y / (5 * y - 12) :=
sorry

end solve_equation_l75_75649


namespace check_point_on_curve_l75_75269

def point_on_curve (x y : ℝ) : Prop :=
  x^2 - x * y + 2 * y + 1 = 0

theorem check_point_on_curve :
  point_on_curve 0 (-1/2) :=
by
  sorry

end check_point_on_curve_l75_75269


namespace select_numbers_with_sum_713_l75_75543

noncomputable def is_suitable_sum (numbers : List ℤ) : Prop :=
  ∃ subset : List ℤ, subset ⊆ numbers ∧ (subset.sum % 10000 = 713)

theorem select_numbers_with_sum_713 :
  ∀ numbers : List ℤ, 
  numbers.length = 1000 → 
  (∀ n ∈ numbers, n % 2 = 1 ∧ n % 5 ≠ 0) →
  is_suitable_sum numbers :=
sorry

end select_numbers_with_sum_713_l75_75543


namespace remainder_three_l75_75786

-- Define the condition that x % 6 = 3
def condition (x : ℕ) : Prop := x % 6 = 3

-- Proof statement that if condition is met, then (3 * x) % 6 = 3
theorem remainder_three {x : ℕ} (h : condition x) : (3 * x) % 6 = 3 :=
sorry

end remainder_three_l75_75786


namespace equivalent_annual_rate_l75_75090

theorem equivalent_annual_rate :
  ∀ (annual_rate compounding_periods: ℝ), annual_rate = 0.08 → compounding_periods = 4 → 
  ((1 + (annual_rate / compounding_periods)) ^ compounding_periods - 1) * 100 = 8.24 :=
by
  intros annual_rate compounding_periods h_rate h_periods
  sorry

end equivalent_annual_rate_l75_75090


namespace economical_shower_heads_l75_75872

theorem economical_shower_heads (x T : ℕ) (x_pos : 0 < x)
    (students : ℕ := 100)
    (preheat_time_per_shower : ℕ := 3)
    (shower_time_per_group : ℕ := 12) :
  (T = preheat_time_per_shower * x + shower_time_per_group * (students / x)) →
  (students * preheat_time_per_shower + shower_time_per_group * students / x = T) →
  x = 20 := by
  sorry

end economical_shower_heads_l75_75872


namespace rita_needs_9_months_l75_75354

def total_required_hours : ℕ := 4000
def backstroke_hours : ℕ := 100
def breaststroke_hours : ℕ := 40
def butterfly_hours : ℕ := 320
def monthly_practice_hours : ℕ := 400

def hours_already_completed : ℕ := backstroke_hours + breaststroke_hours + butterfly_hours
def remaining_hours : ℕ := total_required_hours - hours_already_completed
def months_needed : ℕ := (remaining_hours + monthly_practice_hours - 1) / monthly_practice_hours -- Ceiling division

theorem rita_needs_9_months :
  months_needed = 9 := by
  sorry

end rita_needs_9_months_l75_75354


namespace sum_of_reciprocals_of_geometric_sequence_is_two_l75_75310

theorem sum_of_reciprocals_of_geometric_sequence_is_two
  (a1 q : ℝ)
  (pos_terms : 0 < a1)
  (S P M : ℝ)
  (sum_eq : S = 9)
  (product_eq : P = 81 / 4)
  (sum_of_terms : S = a1 * (1 - q^4) / (1 - q))
  (product_of_terms : P = a1 * a1 * q * q * (a1*q*q) * (q*a1) )
  (sum_of_reciprocals : M = (q^4 - 1) / (a1 * (q^4 - q^3)))
  : M = 2 :=
sorry

end sum_of_reciprocals_of_geometric_sequence_is_two_l75_75310


namespace grandmother_times_older_l75_75448

variables (M G Gr : ℕ)

-- Conditions
def MilenasAge : Prop := M = 7
def GrandfatherAgeRelation : Prop := Gr = G + 2
def AgeDifferenceRelation : Prop := Gr - M = 58

-- Theorem to prove
theorem grandmother_times_older (h1 : MilenasAge M) (h2 : GrandfatherAgeRelation G Gr) (h3 : AgeDifferenceRelation M Gr) :
  G / M = 9 :=
sorry

end grandmother_times_older_l75_75448


namespace inequality_transform_l75_75531

variable {x y : ℝ}

theorem inequality_transform (h : x < y) : - (x / 2) > - (y / 2) :=
sorry

end inequality_transform_l75_75531


namespace worst_player_is_niece_l75_75116

structure Player where
  name : String
  sex : String
  generation : Nat

def grandmother := Player.mk "Grandmother" "Female" 1
def niece := Player.mk "Niece" "Female" 2
def grandson := Player.mk "Grandson" "Male" 3
def son_in_law := Player.mk "Son-in-law" "Male" 2

def worst_player : Player := niece
def best_player : Player := grandmother

-- Conditions
def cousin_check : worst_player ≠ best_player ∧
                   worst_player.generation ≠ best_player.generation ∧ 
                   worst_player.sex ≠ best_player.sex := 
  by sorry

-- Prove that the worst player is the niece
theorem worst_player_is_niece : worst_player = niece :=
  by sorry

end worst_player_is_niece_l75_75116


namespace bird_population_in_1997_l75_75918

theorem bird_population_in_1997 
  (k : ℝ)
  (pop_1995 pop_1996 pop_1998 : ℝ)
  (h1 : pop_1995 = 45)
  (h2 : pop_1996 = 70)
  (h3 : pop_1998 = 145)
  (h4 : pop_1997 - pop_1995 = k * pop_1996)
  (h5 : pop_1998 - pop_1996 = k * pop_1997) : 
  pop_1997 = 105 :=
by
  sorry

end bird_population_in_1997_l75_75918


namespace machine_makes_12_shirts_l75_75572

def shirts_per_minute : ℕ := 2
def minutes_worked : ℕ := 6

def total_shirts_made : ℕ := shirts_per_minute * minutes_worked

theorem machine_makes_12_shirts :
  total_shirts_made = 12 :=
by
  -- proof placeholder
  sorry

end machine_makes_12_shirts_l75_75572


namespace evaluate_expression_l75_75051

theorem evaluate_expression (x y : ℝ) (h1 : 2 * x + 3 * y = 5) (h2 : x = 4) :
  3 * x^2 + 12 * x * y + y^2 = 1 := 
sorry

end evaluate_expression_l75_75051


namespace sum_of_arithmetic_series_l75_75894

theorem sum_of_arithmetic_series (a1 an : ℕ) (d n : ℕ) (s : ℕ) :
  a1 = 2 ∧ an = 100 ∧ d = 2 ∧ n = (an - a1) / d + 1 ∧ s = n * (a1 + an) / 2 → s = 2550 :=
by
  sorry

end sum_of_arithmetic_series_l75_75894


namespace solution_set_l75_75680

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- conditions
axiom differentiable_on_f : ∀ x < 0, DifferentiableAt ℝ f x
axiom derivative_f_x : ∀ x < 0, deriv f x = f' x

axiom condition_3fx_xf'x : ∀ x < 0, 3 * f x + x * f' x > 0

-- goal
theorem solution_set :
  ∀ x, (-2020 < x ∧ x < -2017) ↔ ((x + 2017)^3 * f (x + 2017) + 27 * f (-3) > 0) :=
by
  sorry

end solution_set_l75_75680


namespace work_completion_days_l75_75861

noncomputable def A_days : ℝ := 20
noncomputable def B_days : ℝ := 35
noncomputable def C_days : ℝ := 50

noncomputable def A_work_rate : ℝ := 1 / A_days
noncomputable def B_work_rate : ℝ := 1 / B_days
noncomputable def C_work_rate : ℝ := 1 / C_days

noncomputable def combined_work_rate : ℝ := A_work_rate + B_work_rate + C_work_rate
noncomputable def total_days : ℝ := 1 / combined_work_rate

theorem work_completion_days : total_days = 700 / 69 :=
by
  -- Proof steps would go here
  sorry

end work_completion_days_l75_75861


namespace reflection_line_sum_l75_75337

theorem reflection_line_sum (m b : ℝ) :
  (∀ (x y x' y' : ℝ), (x, y) = (2, 5) → (x', y') = (6, 1) →
  y' = m * x' + b ∧ y = m * x + b) → 
  m + b = 0 :=
sorry

end reflection_line_sum_l75_75337


namespace find_k_l75_75101

open Classical

theorem find_k 
    (z x y k : ℝ) 
    (k_pos_int : k > 0 ∧ ∃ n : ℕ, k = n)
    (prop1 : z - y = k * x)
    (prop2 : x - z = k * y)
    (cond : z = (5 / 3) * (x - y)) :
    k = 3 :=
by
  sorry

end find_k_l75_75101


namespace suitcase_combinations_l75_75589

def count_odd_numbers (n : Nat) : Nat := n / 2

def count_multiples_of_4 (n : Nat) : Nat := n / 4

def count_multiples_of_5 (n : Nat) : Nat := n / 5

theorem suitcase_combinations : count_odd_numbers 40 * count_multiples_of_4 40 * count_multiples_of_5 40 = 1600 :=
by
  sorry

end suitcase_combinations_l75_75589


namespace minnie_takes_longer_l75_75308

def minnie_speed_flat := 25 -- kph
def minnie_speed_downhill := 35 -- kph
def minnie_speed_uphill := 10 -- kph

def penny_speed_flat := 35 -- kph
def penny_speed_downhill := 45 -- kph
def penny_speed_uphill := 15 -- kph

def distance_flat := 25 -- km
def distance_downhill := 20 -- km
def distance_uphill := 15 -- km

noncomputable def minnie_time := 
  (distance_uphill / minnie_speed_uphill) + 
  (distance_downhill / minnie_speed_downhill) + 
  (distance_flat / minnie_speed_flat) -- hours

noncomputable def penny_time := 
  (distance_uphill / penny_speed_uphill) + 
  (distance_downhill / penny_speed_downhill) + 
  (distance_flat / penny_speed_flat) -- hours

noncomputable def minnie_time_minutes := minnie_time * 60 -- minutes
noncomputable def penny_time_minutes := penny_time * 60 -- minutes

noncomputable def time_difference := minnie_time_minutes - penny_time_minutes -- minutes

theorem minnie_takes_longer : time_difference = 130 :=
  sorry

end minnie_takes_longer_l75_75308


namespace find_a_minus_b_l75_75318

-- Define the given function
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 3 * a * x + 4

-- Define the condition for the function being even
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define the function f(x) with given parameters
theorem find_a_minus_b (a b : ℝ) (h_dom_range : ∀ x : ℝ, b - 3 ≤ x → x ≤ 2 * b) (h_even_f : is_even (f a)) :
  a - b = -1 :=
  sorry

end find_a_minus_b_l75_75318


namespace equal_sums_of_squares_l75_75751

-- Define the coordinates of a rectangle in a 3D space.
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define vertices of the rectangle.
def A : Point3D := ⟨0, 0, 0⟩
def B (a : ℝ) : Point3D := ⟨a, 0, 0⟩
def C (a b : ℝ) : Point3D := ⟨a, b, 0⟩
def D (b : ℝ) : Point3D := ⟨0, b, 0⟩

-- Distance squared between two points in 3D space.
def distance_squared (M N : Point3D) : ℝ :=
  (M.x - N.x)^2 + (M.y - N.y)^2 + (M.z - N.z)^2

-- Prove that the sums of the squares of the distances between an arbitrary point M and opposite vertices of the rectangle are equal.
theorem equal_sums_of_squares (a b : ℝ) (M : Point3D) :
  distance_squared M A + distance_squared M (C a b) = distance_squared M (B a) + distance_squared M (D b) :=
by
  sorry

end equal_sums_of_squares_l75_75751


namespace Sherman_weekly_driving_time_l75_75161

theorem Sherman_weekly_driving_time (daily_commute : ℕ := 30) (weekend_drive : ℕ := 2) : 
  (5 * (2 * daily_commute) / 60 + 2 * weekend_drive) = 9 := 
by
  sorry

end Sherman_weekly_driving_time_l75_75161


namespace tan_alpha_plus_pi_over_4_sin_2alpha_fraction_l75_75709

-- Question 1 (Proving tan(alpha + pi/4) = -3 given tan(alpha) = 2)
theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 :=
sorry

-- Question 2 (Proving the given fraction equals 1 given tan(alpha) = 2)
theorem sin_2alpha_fraction (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (2 * α) / 
   (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2 * α) - 1)) = 1 :=
sorry

end tan_alpha_plus_pi_over_4_sin_2alpha_fraction_l75_75709


namespace sum_of_arithmetic_sequence_l75_75726

theorem sum_of_arithmetic_sequence (S : ℕ → ℕ):
  (S 4 = S 8 - S 4) →
  (S 4 = S 12 - S 8) →
  (S 4 = S 16 - S 12) →
  S 16 / S 4 = 10 :=
by
  intros h1 h2 h3
  sorry

end sum_of_arithmetic_sequence_l75_75726


namespace sum_of_first_3n_terms_l75_75219

-- Define the sums of the geometric sequence
variable (S_n S_2n S_3n : ℕ)

-- Given conditions
variable (h1 : S_n = 48)
variable (h2 : S_2n = 60)

-- The statement we need to prove
theorem sum_of_first_3n_terms (S_n S_2n S_3n : ℕ) (h1 : S_n = 48) (h2 : S_2n = 60) :
  S_3n = 63 := by
  sorry

end sum_of_first_3n_terms_l75_75219


namespace rectangle_area_increase_l75_75777

theorem rectangle_area_increase (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let A_original := L * W
  let A_new := (2 * L) * (2 * W)
  (A_new - A_original) / A_original * 100 = 300 := by
  sorry

end rectangle_area_increase_l75_75777


namespace average_goal_l75_75193

-- Define the list of initial rolls
def initial_rolls : List ℕ := [1, 3, 2, 4, 3, 5, 3, 4, 4, 2]

-- Define the next roll
def next_roll : ℕ := 2

-- Define the goal for the average
def goal_average : ℕ := 3

-- The theorem to prove that Ronald's goal for the average of all his rolls is 3
theorem average_goal : (List.sum (initial_rolls ++ [next_roll]) / (List.length (initial_rolls ++ [next_roll]))) = goal_average :=
by
  -- The proof will be provided later
  sorry

end average_goal_l75_75193


namespace circle_area_l75_75121

theorem circle_area (r : ℝ) (h1 : 5 * (1 / (2 * Real.pi * r)) = 2 * r) : π * r^2 = 5 / 4 := by
  sorry

end circle_area_l75_75121


namespace perimeter_C_is_74_l75_75172

/-- Definitions of side lengths based on given perimeters -/
def side_length_A (p_A : ℕ) : ℕ :=
  p_A / 4

def side_length_B (p_B : ℕ) : ℕ :=
  p_B / 4

/-- Definition of side length of C in terms of side lengths of A and B -/
def side_length_C (s_A s_B : ℕ) : ℚ :=
  (s_A : ℚ) / 2 + 2 * (s_B : ℚ)

/-- Definition of perimeter in terms of side length -/
def perimeter (s : ℚ) : ℚ :=
  4 * s

/-- Theorem statement: the perimeter of square C is 74 -/
theorem perimeter_C_is_74 (p_A p_B : ℕ) (h₁ : p_A = 20) (h₂ : p_B = 32) :
  perimeter (side_length_C (side_length_A p_A) (side_length_B p_B)) = 74 := by
  sorry

end perimeter_C_is_74_l75_75172


namespace number_of_square_integers_l75_75351

theorem number_of_square_integers (n : ℤ) (h1 : (0 ≤ n) ∧ (n < 30)) :
  (∃ (k : ℕ), n = 0 ∨ n = 15 ∨ n = 24 → ∃ (k : ℕ), n / (30 - n) = k^2) :=
by
  sorry

end number_of_square_integers_l75_75351
