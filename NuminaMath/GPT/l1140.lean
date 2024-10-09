import Mathlib

namespace combine_ingredients_l1140_114059

theorem combine_ingredients : 
  ∃ (water flour salt : ℕ), 
    water = 10 ∧ flour = 16 ∧ salt = 1 / 2 * flour ∧ 
    (water + flour = 26) ∧ (salt = 8) :=
by
  sorry

end combine_ingredients_l1140_114059


namespace find_slope_of_parallel_line_l1140_114042

-- Define the condition that line1 is parallel to line2.
def lines_parallel (k : ℝ) : Prop :=
  k = -3

-- The theorem that proves the condition given.
theorem find_slope_of_parallel_line (k : ℝ) (h : lines_parallel k) : k = -3 :=
by
  exact h

end find_slope_of_parallel_line_l1140_114042


namespace dice_total_correct_l1140_114053

-- Define the problem conditions
def IvanDice (x : ℕ) : ℕ := x
def JerryDice (x : ℕ) : ℕ := (1 / 2 * x) ^ 2

-- Define the total dice function
def totalDice (x : ℕ) : ℕ := IvanDice x + JerryDice x

-- The theorem to prove the answer
theorem dice_total_correct (x : ℕ) : totalDice x = x + (1 / 4) * x ^ 2 := 
  sorry

end dice_total_correct_l1140_114053


namespace lcm_division_l1140_114072

open Nat

-- Define the LCM function for a list of integers
def list_lcm (l : List Nat) : Nat := l.foldr (fun a b => Nat.lcm a b) 1

-- Define the sequence ranges
def range1 := List.range' 20 21 -- From 20 to 40 inclusive
def range2 := List.range' 41 10 -- From 41 to 50 inclusive

-- Define P and Q
def P : Nat := list_lcm range1
def Q : Nat := Nat.lcm P (list_lcm range2)

-- The theorem statement
theorem lcm_division : (Q / P) = 55541 := by
  sorry

end lcm_division_l1140_114072


namespace not_both_perfect_squares_l1140_114041

theorem not_both_perfect_squares (n : ℕ) (hn : 0 < n) : 
  ¬ (∃ a b : ℕ, (n+1) * 2^n = a^2 ∧ (n+3) * 2^(n + 2) = b^2) :=
sorry

end not_both_perfect_squares_l1140_114041


namespace total_length_of_ropes_l1140_114087

theorem total_length_of_ropes 
  (L : ℕ)
  (first_used second_used : ℕ)
  (h1 : first_used = 42) 
  (h2 : second_used = 12) 
  (h3 : (L - second_used) = 4 * (L - first_used)) :
  2 * L = 104 :=
by
  -- We skip the proof for now
  sorry

end total_length_of_ropes_l1140_114087


namespace solve_x_squared_plus_y_squared_l1140_114085

-- Variables
variables {x y : ℝ}

-- Conditions
def cond1 : (x + y)^2 = 36 := sorry
def cond2 : x * y = 8 := sorry

-- Theorem stating the problem's equivalent proof
theorem solve_x_squared_plus_y_squared : x^2 + y^2 = 20 := sorry

end solve_x_squared_plus_y_squared_l1140_114085


namespace num_large_posters_l1140_114012

-- Define the constants
def total_posters : ℕ := 50
def small_posters : ℕ := total_posters * 2 / 5
def medium_posters : ℕ := total_posters / 2
def large_posters : ℕ := total_posters - (small_posters + medium_posters)

-- Theorem to prove the number of large posters
theorem num_large_posters : large_posters = 5 :=
by
  sorry

end num_large_posters_l1140_114012


namespace slope_proof_l1140_114096

noncomputable def slope_between_midpoints : ℚ :=
  let p1 := (2, 3)
  let p2 := (4, 5)
  let q1 := (7, 3)
  let q2 := (8, 7)

  let midpoint (a b : ℚ × ℚ) : ℚ × ℚ := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

  let m1 := midpoint p1 p2
  let m2 := midpoint q1 q2

  (m2.2 - m1.2) / (m2.1 - m1.1)

theorem slope_proof : slope_between_midpoints = 2 / 9 := by
  sorry

end slope_proof_l1140_114096


namespace train_passing_time_l1140_114022

theorem train_passing_time 
  (length_of_train : ℕ) 
  (length_of_platform : ℕ) 
  (time_to_pass_pole : ℕ) 
  (speed_of_train : ℕ) 
  (combined_length : ℕ) 
  (time_to_pass_platform : ℕ) 
  (h1 : length_of_train = 240) 
  (h2 : length_of_platform = 650)
  (h3 : time_to_pass_pole = 24)
  (h4 : speed_of_train = length_of_train / time_to_pass_pole)
  (h5 : combined_length = length_of_train + length_of_platform)
  (h6 : time_to_pass_platform = combined_length / speed_of_train) : 
  time_to_pass_platform = 89 :=
sorry

end train_passing_time_l1140_114022


namespace meal_combinations_correct_l1140_114075

-- Define the given conditions
def number_of_entrees : Nat := 4
def number_of_drinks : Nat := 4
def number_of_desserts : Nat := 2

-- Define the total number of meal combinations to prove
def total_meal_combinations : Nat := number_of_entrees * number_of_drinks * number_of_desserts

-- The theorem we want to prove
theorem meal_combinations_correct : total_meal_combinations = 32 := 
by 
  sorry

end meal_combinations_correct_l1140_114075


namespace eval_expression_l1140_114090

theorem eval_expression : 8 / 4 - 3^2 - 10 + 5 * 2 = -7 :=
by
  sorry

end eval_expression_l1140_114090


namespace lab_preparation_is_correct_l1140_114054

def correct_operation (m_CuSO4 : ℝ) (m_CuSO4_5H2O : ℝ) (V_solution : ℝ) : Prop :=
  let molar_mass_CuSO4 := 160 -- g/mol
  let molar_mass_CuSO4_5H2O := 250 -- g/mol
  let desired_concentration := 0.1 -- mol/L
  let desired_volume := 0.480 -- L
  let prepared_volume := 0.500 -- L
  (m_CuSO4 = 8.0 ∧ V_solution = 0.500 ∧ m_CuSO4_5H2O = 12.5 ∧ desired_concentration * prepared_volume * molar_mass_CuSO4_5H2O = 12.5)

-- Example proof statement to show the problem with "sorry"
theorem lab_preparation_is_correct : correct_operation 8.0 12.5 0.500 :=
by
  sorry

end lab_preparation_is_correct_l1140_114054


namespace correct_multiplication_l1140_114081

variable {a : ℕ} -- Assume 'a' to be a natural number for simplicity in this example

theorem correct_multiplication : (3 * a) * (4 * a^2) = 12 * a^3 := by
  sorry

end correct_multiplication_l1140_114081


namespace interest_for_20000_l1140_114099

-- Definition of simple interest
def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := P * r * t

variables (P1 P2 I1 I2 r : ℝ)
-- Given conditions
def h1 := (P1 = 5000)
def h2 := (I1 = 250)
def h3 := (r = I1 / P1)
-- Question condition
def h4 := (P2 = 20000)
def t := 1

theorem interest_for_20000 :
  P1 = 5000 →
  I1 = 250 →
  P2 = 20000 →
  r = I1 / P1 →
  simple_interest P2 r t = 1000 :=
by
  intros
  -- Proof goes here
  sorry

end interest_for_20000_l1140_114099


namespace sum_ge_3_implies_one_ge_2_l1140_114002

theorem sum_ge_3_implies_one_ge_2 (a b : ℕ) (h : a + b ≥ 3) : a ≥ 2 ∨ b ≥ 2 :=
by
  sorry

end sum_ge_3_implies_one_ge_2_l1140_114002


namespace isosceles_triangle_perimeter_correct_l1140_114016

noncomputable def isosceles_triangle_perimeter (x y : ℝ) : ℝ :=
  if x = y then 2 * x + y else if (2 * x > y ∧ y > 2 * x - y) ∨ (2 * y > x ∧ x > 2 * y - x) then 2 * y + x else 0

theorem isosceles_triangle_perimeter_correct (x y : ℝ) (h : |x - 5| + (y - 8)^2 = 0) :
  isosceles_triangle_perimeter x y = 18 ∨ isosceles_triangle_perimeter x y = 21 := by
sorry

end isosceles_triangle_perimeter_correct_l1140_114016


namespace length_PQ_eq_b_l1140_114025

open Real

variables {a b : ℝ} (h : a > b) (p : ℝ × ℝ) (h₁ : (p.fst / a) ^ 2 + (p.snd / b) ^ 2 = 1)
variables (F₁ F₂ : ℝ × ℝ) (P Q : ℝ × ℝ)
variable (Q_on_segment : Q.1 = (F₁.1 + F₂.1) / 2)
variable (equal_inradii : inradius (triangle P Q F₁) = inradius (triangle P Q F₂))

theorem length_PQ_eq_b : dist P Q = b :=
by
  sorry

end length_PQ_eq_b_l1140_114025


namespace josanna_next_test_score_l1140_114015

theorem josanna_next_test_score :
  let scores := [75, 85, 65, 95, 70]
  let current_sum := scores.sum
  let current_average := current_sum / scores.length
  let desired_average := current_average + 10
  let new_test_count := scores.length + 1
  let desired_sum := desired_average * new_test_count
  let required_score := desired_sum - current_sum
  required_score = 138 :=
by
  sorry

end josanna_next_test_score_l1140_114015


namespace find_abc_l1140_114056

theorem find_abc
  (a b c : ℝ)
  (h : ∀ x y z : ℝ, |a * x + b * y + c * z| + |b * x + c * y + a * z| + |c * x + a * y + b * z| = |x| + |y| + |z|):
  (a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = -1 ∧ b = 0 ∧ c = 0) ∨ 
  (a = 0 ∧ b = 1 ∧ c = 0) ∨ (a = 0 ∧ b = -1 ∧ c = 0) ∨ 
  (a = 0 ∧ b = 0 ∧ c = 1) ∨ (a = 0 ∧ b = 0 ∧ c = -1) :=
sorry

end find_abc_l1140_114056


namespace expand_product_l1140_114061

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 := 
  sorry

end expand_product_l1140_114061


namespace BRAIN_7225_cycle_line_number_l1140_114017

def BRAIN_cycle : Nat := 5
def _7225_cycle : Nat := 4

theorem BRAIN_7225_cycle_line_number : Nat.lcm BRAIN_cycle _7225_cycle = 20 :=
by
  sorry

end BRAIN_7225_cycle_line_number_l1140_114017


namespace pairing_probability_l1140_114011

variable {students : Fin 28} (Alex Jamie : Fin 28)

theorem pairing_probability (h1 : ∀ (i j : Fin 28), i ≠ j) :
  ∃ p : ℚ, p = 1 / 27 ∧ 
  (∃ (A_J_pairs : Finset (Fin 28) × Finset (Fin 28)),
  A_J_pairs.1 = {Alex} ∧ A_J_pairs.2 = {Jamie}) -> p = 1 / 27
:= sorry

end pairing_probability_l1140_114011


namespace quad_function_intersects_x_axis_l1140_114048

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quad_function_intersects_x_axis (m : ℝ) :
  (discriminant (2 * m) (8 * m + 1) (8 * m) ≥ 0) ↔ (m ≥ -1/16 ∧ m ≠ 0) :=
by
  sorry

end quad_function_intersects_x_axis_l1140_114048


namespace total_students_l1140_114039

-- Define the conditions
variables (S : ℕ) -- total number of students
variable (h1 : (3/5 : ℚ) * S + (1/5 : ℚ) * S + 10 = S)

-- State the theorem
theorem total_students (HS : S = 50) : 3 / 5 * S + 1 / 5 * S + 10 = S := by
  -- Here we declare the proof is to be filled in later.
  sorry

end total_students_l1140_114039


namespace polygon_problem_l1140_114086

theorem polygon_problem 
  (D : ℕ → ℕ) (m x : ℕ) 
  (H1 : ∀ n, D n = n * (n - 3) / 2)
  (H2 : D m = 3 * D (m - 3))
  (H3 : D (m + x) = 7 * D m) :
  m = 9 ∧ x = 12 ∧ (m + x) - m = 12 :=
by {
  -- the proof would go here, skipped as per the instructions.
  sorry
}

end polygon_problem_l1140_114086


namespace remainder_of_sum_div_8_l1140_114060

theorem remainder_of_sum_div_8 :
  let a := 2356789
  let b := 211
  (a + b) % 8 = 0 := 
by 
  sorry

end remainder_of_sum_div_8_l1140_114060


namespace john_school_year_hours_l1140_114027

noncomputable def requiredHoursPerWeek (summerHoursPerWeek : ℕ) (summerWeeks : ℕ) 
                                       (summerEarnings : ℕ) (schoolWeeks : ℕ) 
                                       (schoolEarnings : ℕ) : ℕ :=
    schoolEarnings * summerHoursPerWeek * summerWeeks / (summerEarnings * schoolWeeks)

theorem john_school_year_hours :
  ∀ (summerHoursPerWeek summerWeeks summerEarnings schoolWeeks schoolEarnings : ℕ),
    summerHoursPerWeek = 40 →
    summerWeeks = 10 →
    summerEarnings = 4000 →
    schoolWeeks = 50 →
    schoolEarnings = 4000 →
    requiredHoursPerWeek summerHoursPerWeek summerWeeks summerEarnings schoolWeeks schoolEarnings = 8 :=
by
  intros
  sorry

end john_school_year_hours_l1140_114027


namespace slower_train_speed_l1140_114069

theorem slower_train_speed (v : ℝ) (L : ℝ) (faster_speed_km_hr : ℝ) (time_sec : ℝ) (relative_speed : ℝ) 
  (hL : L = 70) (hfaster_speed_km_hr : faster_speed_km_hr = 50)
  (htime_sec : time_sec = 36) (hrelative_speed : relative_speed = (faster_speed_km_hr - v) * (1000 / 3600)) :
  140 = relative_speed * time_sec → v = 36 := 
by
  -- Proof omitted
  sorry

end slower_train_speed_l1140_114069


namespace exist_three_primes_sum_to_30_l1140_114033

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def less_than_twenty (n : ℕ) : Prop := n < 20

theorem exist_three_primes_sum_to_30 : 
  ∃ A B C : ℕ, is_prime A ∧ is_prime B ∧ is_prime C ∧ 
  less_than_twenty A ∧ less_than_twenty B ∧ less_than_twenty C ∧ 
  A + B + C = 30 :=
by 
  -- assume A = 2, prime and less than 20
  -- find B, C such that B and C are primes less than 20 and A + B + C = 30
  sorry

end exist_three_primes_sum_to_30_l1140_114033


namespace roberto_outfits_l1140_114028

theorem roberto_outfits (trousers shirts jackets : ℕ) (restricted_shirt restricted_jacket : ℕ) 
  (h_trousers : trousers = 5) 
  (h_shirts : shirts = 6) 
  (h_jackets : jackets = 4) 
  (h_restricted_shirt : restricted_shirt = 1) 
  (h_restricted_jacket : restricted_jacket = 1) : 
  ((trousers * shirts * jackets) - (restricted_shirt * restricted_jacket * trousers) = 115) := 
  by 
    sorry

end roberto_outfits_l1140_114028


namespace gain_percentage_calculation_l1140_114047

theorem gain_percentage_calculation 
  (C S : ℝ)
  (h1 : 30 * S = 40 * C) :
  (10 * S / (30 * C)) * 100 = 44.44 :=
by
  sorry

end gain_percentage_calculation_l1140_114047


namespace petya_coloring_failure_7_petya_coloring_failure_10_l1140_114080

theorem petya_coloring_failure_7 :
  ¬ ∀ (points : Fin 200 → Fin 7) (segments : ∀ (i j : Fin 200), i ≠ j → Fin 7),
  ∃ (colors : ∀ (i j : Fin 200), i ≠ j → Fin 7),
  ∀ (i j : Fin 200) (h : i ≠ j),
    (segments i j h ≠ points i) ∧ (segments i j h ≠ points j) :=
sorry

theorem petya_coloring_failure_10 :
  ¬ ∀ (points : Fin 200 → Fin 10) (segments : ∀ (i j : Fin 200), i ≠ j → Fin 10),
  ∃ (colors : ∀ (i j : Fin 200), i ≠ j → Fin 10),
  ∀ (i j : Fin 200) (h : i ≠ j),
    (segments i j h ≠ points i) ∧ (segments i j h ≠ points j) :=
sorry

end petya_coloring_failure_7_petya_coloring_failure_10_l1140_114080


namespace transformed_curve_is_circle_l1140_114055

open Real

def polar_curve (ρ θ : ℝ) : Prop :=
  ρ^2 = 12 / (3 * cos θ^2 + 4 * sin θ^2)

def cartesian_curve (x y: ℝ) : Prop :=
  3 * x^2 + 4 * y^2 = 12

def transformation (x y x' y' : ℝ) : Prop :=
  x' = x / 2 ∧ y' = y * sqrt (3 / 3)

theorem transformed_curve_is_circle (x y x' y' : ℝ) 
  (h1: cartesian_curve x y) (h2: transformation x y x' y') : 
  (x'^2 + y'^2 = 1) :=
sorry

end transformed_curve_is_circle_l1140_114055


namespace which_set_can_form_triangle_l1140_114057

-- Definition of the triangle inequality theorem
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Conditions for each set of line segments
def setA := (2, 6, 8)
def setB := (4, 6, 7)
def setC := (5, 6, 12)
def setD := (2, 3, 6)

-- Proof problem statement
theorem which_set_can_form_triangle : 
  triangle_inequality 2 6 8 = false ∧
  triangle_inequality 4 6 7 = true ∧
  triangle_inequality 5 6 12 = false ∧
  triangle_inequality 2 3 6 = false := 
by
  sorry -- Proof omitted

end which_set_can_form_triangle_l1140_114057


namespace positive_diff_of_squares_l1140_114046

theorem positive_diff_of_squares (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 10) : a^2 - b^2 = 400 := by
  sorry

end positive_diff_of_squares_l1140_114046


namespace beetle_distance_l1140_114014

theorem beetle_distance :
  let p1 := 3
  let p2 := -5
  let p3 := 7
  let dist1 := Int.natAbs (p2 - p1)
  let dist2 := Int.natAbs (p3 - p2)
  dist1 + dist2 = 20 :=
by
  let p1 := 3
  let p2 := -5
  let p3 := 7
  let dist1 := Int.natAbs (p2 - p1)
  let dist2 := Int.natAbs (p3 - p2)
  show dist1 + dist2 = 20
  sorry

end beetle_distance_l1140_114014


namespace functional_form_of_f_l1140_114093

variable (f : ℝ → ℝ)

-- Define the condition as an axiom
axiom cond_f : ∀ (x y : ℝ), |f (x + y) - f (x - y) - y| ≤ y^2

-- State the theorem to be proved
theorem functional_form_of_f :
  ∃ c : ℝ, ∀ x : ℝ, f x = x / 2 + c := by
  sorry

end functional_form_of_f_l1140_114093


namespace max_height_of_basketball_l1140_114064

def h (t : ℝ) : ℝ := -5 * t^2 + 50 * t + 2

theorem max_height_of_basketball : ∃ t : ℝ, h t = 127 :=
by
  use 5
  sorry

end max_height_of_basketball_l1140_114064


namespace johnny_savings_l1140_114034

variable (S : ℤ) -- The savings in September.

theorem johnny_savings :
  (S + 49 + 46 - 58 = 67) → (S = 30) :=
by
  intro h
  sorry

end johnny_savings_l1140_114034


namespace eval_expression_l1140_114094

theorem eval_expression (x : ℝ) (h : x = Real.sqrt 2 + 1) : (x + 1) / (x - 1) = 1 + Real.sqrt 2 := 
by
  sorry

end eval_expression_l1140_114094


namespace each_niece_gets_13_l1140_114009

-- Define the conditions
def total_sandwiches : ℕ := 143
def number_of_nieces : ℕ := 11

-- Prove that each niece can get 13 ice cream sandwiches
theorem each_niece_gets_13 : total_sandwiches / number_of_nieces = 13 :=
by
  -- Proof omitted
  sorry

end each_niece_gets_13_l1140_114009


namespace runners_meet_again_l1140_114063

theorem runners_meet_again :
    ∀ t : ℝ,
      t ≠ 0 →
      (∃ k : ℤ, 3.8 * t - 4 * t = 400 * k) ∧
      (∃ m : ℤ, 4.2 * t - 4 * t = 400 * m) ↔
      t = 2000 := 
by
  sorry

end runners_meet_again_l1140_114063


namespace arithmetic_sequence_problem_l1140_114045

noncomputable def a1 := 3
noncomputable def S (n : ℕ) (a1 d : ℕ) : ℕ := n * (a1 + (n - 1) * d / 2)

theorem arithmetic_sequence_problem (d : ℕ) 
  (h1 : S 1 a1 d = 3) 
  (h2 : S 1 a1 d / 2 + S 4 a1 d / 4 = 18) : 
  S 5 a1 d = 75 :=
sorry

end arithmetic_sequence_problem_l1140_114045


namespace monotonically_decreasing_interval_l1140_114092

noncomputable def f (x : ℝ) : ℝ :=
  (2 * Real.exp 2) * Real.exp (x - 2) - 2 * x + 1/2 * x^2

theorem monotonically_decreasing_interval :
  ∀ x : ℝ, x < 0 → ((2 * Real.exp x - 2 + x) < 0) :=
by
  sorry

end monotonically_decreasing_interval_l1140_114092


namespace smallest_positive_whole_number_divisible_by_first_five_primes_l1140_114070

def is_prime (n : Nat) : Prop := Nat.Prime n

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

def smallest_positive_divisible (lst : List Nat) : Nat :=
  List.foldl (· * ·) 1 lst

theorem smallest_positive_whole_number_divisible_by_first_five_primes :
  smallest_positive_divisible first_five_primes = 2310 := by
  sorry

end smallest_positive_whole_number_divisible_by_first_five_primes_l1140_114070


namespace find_num_white_balls_l1140_114049

theorem find_num_white_balls
  (W : ℕ)
  (total_balls : ℕ := 15 + W)
  (prob_black : ℚ := 7 / total_balls)
  (given_prob : ℚ := 0.38095238095238093) :
  prob_black = given_prob → W = 3 :=
by
  intro h
  sorry

end find_num_white_balls_l1140_114049


namespace solve_for_x_l1140_114010

theorem solve_for_x (x : ℝ) (h : (1 / (Real.sqrt x + Real.sqrt (x - 2)) + 1 / (Real.sqrt x + Real.sqrt (x + 2)) = 1 / 4)) : x = 257 / 16 := by
  sorry

end solve_for_x_l1140_114010


namespace quadrant_classification_l1140_114005

theorem quadrant_classification :
  ∀ (x y : ℝ), (4 * x - 3 * y = 24) → (|x| = |y|) → 
  ((x > 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)) :=
by
  intros x y h_line h_eqdist
  sorry

end quadrant_classification_l1140_114005


namespace perfect_squares_perfect_square_plus_one_l1140_114037

theorem perfect_squares : (∃ n : ℕ, 2^n + 3 = (x : ℕ)^2) ↔ n = 0 ∨ n = 3 :=
by
  sorry

theorem perfect_square_plus_one : (∃ n : ℕ, 2^n + 1 = (x : ℕ)^2) ↔ n = 3 :=
by
  sorry

end perfect_squares_perfect_square_plus_one_l1140_114037


namespace solve_for_y_l1140_114026

theorem solve_for_y (x y : ℝ) (h₁ : x - y = 16) (h₂ : x + y = 4) : y = -6 := 
by 
  sorry

end solve_for_y_l1140_114026


namespace total_wheels_in_garage_l1140_114024

def total_wheels (bicycles tricycles unicycles : ℕ) (bicycle_wheels tricycle_wheels unicycle_wheels : ℕ) :=
  bicycles * bicycle_wheels + tricycles * tricycle_wheels + unicycles * unicycle_wheels

theorem total_wheels_in_garage :
  total_wheels 3 4 7 2 3 1 = 25 := by
  -- Calculation shows:
  -- (3 * 2) + (4 * 3) + (7 * 1) = 6 + 12 + 7 = 25
  sorry

end total_wheels_in_garage_l1140_114024


namespace max_experiments_fibonacci_search_l1140_114089

-- Define the conditions and the theorem
def is_unimodal (f : ℕ → ℕ) : Prop :=
  ∃ k, ∀ n m, (n < k ∧ k ≤ m) → f n < f k ∧ f k > f m

def fibonacci_search_experiments (n : ℕ) : ℕ :=
  -- Placeholder function representing the steps of Fibonacci search
  if n <= 1 then n else fibonacci_search_experiments (n - 1) + fibonacci_search_experiments (n - 2)

theorem max_experiments_fibonacci_search (f : ℕ → ℕ) (n : ℕ) (hn : n = 33) (hf : is_unimodal f) : fibonacci_search_experiments n ≤ 7 :=
  sorry

end max_experiments_fibonacci_search_l1140_114089


namespace length_greater_than_width_l1140_114058

theorem length_greater_than_width
  (perimeter : ℕ)
  (P : perimeter = 150)
  (l w difference : ℕ)
  (L : l = 60)
  (W : w = 45)
  (D : difference = l - w) :
  difference = 15 :=
by
  sorry

end length_greater_than_width_l1140_114058


namespace triangle_perimeter_l1140_114073

theorem triangle_perimeter (x : ℕ) (hx1 : x % 2 = 1) (hx2 : 5 < x) (hx3 : x < 11) : 
  (3 + 8 + x = 18) ∨ (3 + 8 + x = 20) :=
sorry

end triangle_perimeter_l1140_114073


namespace symmetric_points_x_axis_l1140_114077

theorem symmetric_points_x_axis (a b : ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) 
  (hP : P = (a - 3, 1)) (hQ : Q = (2, b + 1)) (hSymm : P.1 = Q.1 ∧ P.2 = -Q.2) :
  a + b = 3 :=
by 
  sorry

end symmetric_points_x_axis_l1140_114077


namespace johns_monthly_earnings_l1140_114068

variable (work_days : ℕ) (hours_per_day : ℕ) (former_wage : ℝ) (raise_percentage : ℝ) (days_in_month : ℕ)

def johns_earnings (work_days hours_per_day : ℕ) (former_wage raise_percentage : ℝ) (days_in_month : ℕ) : ℝ :=
  let days_worked := days_in_month / 2
  let total_hours := days_worked * hours_per_day
  let raise := former_wage * raise_percentage
  let new_wage := former_wage + raise
  total_hours * new_wage

theorem johns_monthly_earnings (work_days : ℕ := 15) (hours_per_day : ℕ := 12) (former_wage : ℝ := 20) (raise_percentage : ℝ := 0.3) (days_in_month : ℕ := 30) :
  johns_earnings work_days hours_per_day former_wage raise_percentage days_in_month = 4680 :=
by
  sorry

end johns_monthly_earnings_l1140_114068


namespace inequality_k_distance_comparison_l1140_114031

theorem inequality_k (k : ℝ) (x : ℝ) : 
  -3 < k ∧ k ≤ 0 → 2 * k * x^2 + k * x - 3/8 < 0 := sorry

theorem distance_comparison (a b : ℝ) (hab : a ≠ b) : 
  (abs ((a^2 + b^2) / 2 - (a + b)^2 / 4) > abs (a * b - (a + b)^2 / 4)) := sorry

end inequality_k_distance_comparison_l1140_114031


namespace log_comparison_l1140_114091

theorem log_comparison (a b c : ℝ) (h₁ : a = Real.log 6 / Real.log 4) (h₂ : b = Real.log 3 / Real.log 2) (h₃ : c = 3/2) : b > c ∧ c > a := 
by 
  sorry

end log_comparison_l1140_114091


namespace bowling_average_decrease_l1140_114038

theorem bowling_average_decrease
    (initial_average : ℝ) (wickets_last_match : ℝ) (runs_last_match : ℝ)
    (average_decrease : ℝ) (W : ℝ)
    (H_initial : initial_average = 12.4)
    (H_wickets_last_match : wickets_last_match = 6)
    (H_runs_last_match : runs_last_match = 26)
    (H_average_decrease : average_decrease = 0.4) :
    W = 115 :=
by
  sorry

end bowling_average_decrease_l1140_114038


namespace int_solution_exists_l1140_114065

theorem int_solution_exists (x y : ℤ) (h : x + y = 5) : x = 2 ∧ y = 3 := 
by
  sorry

end int_solution_exists_l1140_114065


namespace imaginary_part_of_complex_l1140_114018

theorem imaginary_part_of_complex : ∀ z : ℂ, z = i^2 * (1 + i) → z.im = -1 :=
by
  intro z
  intro h
  sorry

end imaginary_part_of_complex_l1140_114018


namespace rectangle_area_l1140_114071

-- Define the conditions as hypotheses in Lean 4
variable (x : ℤ)
variable (area : ℤ := 864)
variable (width : ℤ := x - 12)

-- State the theorem to prove the relation between length and area
theorem rectangle_area (h : x * width = area) : x * (x - 12) = 864 :=
by 
  sorry

end rectangle_area_l1140_114071


namespace problem_1_problem_2_l1140_114066

def condition_p (x : ℝ) : Prop := 4 * x ^ 2 + 12 * x - 7 ≤ 0
def condition_q (a x : ℝ) : Prop := a - 3 ≤ x ∧ x ≤ a + 3

-- Problem 1: When a=0, if p is true and q is false, the range of real numbers x
theorem problem_1 (x : ℝ) :
  condition_p x ∧ ¬ condition_q 0 x ↔ -7/2 ≤ x ∧ x < -3 := sorry

-- Problem 2: If p is a sufficient condition for q, the range of real numbers a
theorem problem_2 (a : ℝ) :
  (∀ x : ℝ, condition_p x → condition_q a x) ↔ -5/2 ≤ a ∧ a ≤ -1/2 := sorry

end problem_1_problem_2_l1140_114066


namespace min_sum_rect_box_l1140_114082

-- Define the main theorem with the given constraints
theorem min_sum_rect_box (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_vol : a * b * c = 2002) : a + b + c ≥ 38 :=
  sorry

end min_sum_rect_box_l1140_114082


namespace find_noon_temperature_l1140_114084

theorem find_noon_temperature (T T₄₀₀ T₈₀₀ : ℝ) 
  (h1 : T₄₀₀ = T + 8)
  (h2 : T₈₀₀ = T₄₀₀ - 11)
  (h3 : T₈₀₀ = T + 1) : 
  T = 4 :=
by
  sorry

end find_noon_temperature_l1140_114084


namespace tomatoes_picked_today_l1140_114023

theorem tomatoes_picked_today (initial yesterday_picked left_after_yesterday today_picked : ℕ)
  (h1 : initial = 160)
  (h2 : yesterday_picked = 56)
  (h3 : left_after_yesterday = 104)
  (h4 : initial - yesterday_picked = left_after_yesterday) :
  today_picked = 56 :=
by
  sorry

end tomatoes_picked_today_l1140_114023


namespace bess_milk_daily_l1140_114079

-- Definitions based on conditions from step a)
variable (B : ℕ) -- B is the number of pails Bess gives every day

def BrownieMilk : ℕ := 3 * B
def DaisyMilk : ℕ := B + 1
def TotalDailyMilk : ℕ := B + BrownieMilk B + DaisyMilk B

-- Conditions definition to be used in Lean to ensure the equivalence
axiom weekly_milk_total : 7 * TotalDailyMilk B = 77
axiom daily_milk_eq : TotalDailyMilk B = 11

-- Prove that Bess gives 2 pails of milk everyday
theorem bess_milk_daily : B = 2 :=
by
  sorry

end bess_milk_daily_l1140_114079


namespace find_a2_l1140_114032

variable (a : ℕ → ℝ) (d : ℝ)

axiom arithmetic_seq (n : ℕ) : a (n + 1) = a n + d
axiom common_diff : d = 2
axiom geometric_mean : (a 4) ^ 2 = (a 5) * (a 2)

theorem find_a2 : a 2 = -8 := 
by 
  sorry

end find_a2_l1140_114032


namespace simplify_fraction_l1140_114013

theorem simplify_fraction : (5 + 4 - 3) / (5 + 4 + 3) = 1 / 2 := 
by {
  sorry
}

end simplify_fraction_l1140_114013


namespace good_numbers_l1140_114098

def is_good (n : ℕ) : Prop :=
  ∀ d, d ∣ n → (d + 1) ∣ (n + 1)

theorem good_numbers (n : ℕ) :
  is_good n ↔ n = 1 ∨ (Nat.Prime n ∧ Odd n) :=
by
  sorry

end good_numbers_l1140_114098


namespace choir_members_count_l1140_114020

theorem choir_members_count : ∃ n : ℕ, n = 226 ∧ 
  (n % 10 = 6) ∧ 
  (n % 11 = 6) ∧ 
  (200 < n ∧ n < 300) :=
by
  sorry

end choir_members_count_l1140_114020


namespace min_value_one_over_a_plus_one_over_b_point_P_outside_ellipse_l1140_114083

variable (a b : ℝ)
-- Conditions: a and b are positive real numbers and (a + b)x - 1 ≤ x^2 for all x > 0
variables (ha : a > 0) (hb : b > 0) (h : ∀ x : ℝ, 0 < x → (a + b) * x - 1 ≤ x^2)

-- Question 1: Prove that the minimum value of 1/a + 1/b is 2
theorem min_value_one_over_a_plus_one_over_b : (1 : ℝ) / a + (1 : ℝ) / b = 2 := 
sorry

-- Question 2: Determine point P(1, -1) relative to the ellipse x^2/a^2 + y^2/b^2 = 1
theorem point_P_outside_ellipse : (1 : ℝ)^2 / (a^2) + (-1 : ℝ)^2 / (b^2) > 1 :=
sorry

end min_value_one_over_a_plus_one_over_b_point_P_outside_ellipse_l1140_114083


namespace raja_journey_distance_l1140_114006

theorem raja_journey_distance
  (T : ℝ) (D : ℝ)
  (H1 : T = 10)
  (H2 : ∀ t1 t2, t1 = D / 42 ∧ t2 = D / 48 → T = t1 + t2) :
  D = 224 :=
by
  sorry

end raja_journey_distance_l1140_114006


namespace angle_of_inclination_l1140_114044

theorem angle_of_inclination (m : ℝ) (h : m = -1) : 
  ∃ α : ℝ, α = 3 * Real.pi / 4 := 
sorry

end angle_of_inclination_l1140_114044


namespace tree_growth_rate_l1140_114043

noncomputable def growth_rate_per_week (initial_height final_height : ℝ) (months weeks_per_month : ℕ) : ℝ :=
  (final_height - initial_height) / (months * weeks_per_month)

theorem tree_growth_rate :
  growth_rate_per_week 10 42 4 4 = 2 := 
by
  sorry

end tree_growth_rate_l1140_114043


namespace initial_money_correct_l1140_114050

def initial_money (total: ℕ) (allowance: ℕ): ℕ :=
  total - allowance

theorem initial_money_correct: initial_money 18 8 = 10 :=
  by sorry

end initial_money_correct_l1140_114050


namespace number_of_distinct_intersections_l1140_114001

/-- The problem is to prove that the number of distinct intersection points
in the xy-plane for the graphs of the given equations is exactly 4. -/
theorem number_of_distinct_intersections :
  ∃ (S : Finset (ℝ × ℝ)), 
  (∀ p : ℝ × ℝ, p ∈ S ↔
    ((p.1 + p.2 = 7 ∨ 2 * p.1 - 3 * p.2 + 1 = 0) ∧
     (p.1 - p.2 - 2 = 0 ∨ 3 * p.1 + 2 * p.2 - 10 = 0))) ∧
  S.card = 4 :=
sorry

end number_of_distinct_intersections_l1140_114001


namespace meal_cost_l1140_114021

theorem meal_cost (M : ℝ) (h1 : 3 * M + 15 = 45) : M = 10 :=
by
  sorry

end meal_cost_l1140_114021


namespace max_gold_coins_l1140_114062

theorem max_gold_coins (n k : ℕ) (h1 : n = 13 * k + 3) (h2 : n < 110) : n ≤ 107 :=
by
  sorry

end max_gold_coins_l1140_114062


namespace area_of_triangle_is_sqrt_5_sum_of_tangents_eq_1_l1140_114088

-- Definitions and conditions
variable {A B C a b c : ℝ}
variable (cosA : ℝ) (sinA : ℝ)
variable (area : ℝ)
variable (tanA tanB tanC : ℝ)

-- Given conditions
axiom angle_identity : b^2 + c^2 = 3 * b * c * cosA
axiom sin_cos_identity : sinA^2 + cosA^2 = 1
axiom law_of_cosines : a^2 = b^2 + c^2 - 2 * b * c * cosA

-- Part (1) statement
theorem area_of_triangle_is_sqrt_5 (B_eq_C : B = C) (a_eq_2 : a = 2) 
    (cosA_eq_2_3 : cosA = 2/3) 
    (b_eq_sqrt6 : b = Real.sqrt 6) 
    (sinA_eq_sqrt5_3 : sinA = Real.sqrt 5 / 3) 
    : area = Real.sqrt 5 := sorry

-- Part (2) statement
theorem sum_of_tangents_eq_1 (tanA_eq : tanA = sinA / cosA)
    (tanB_eq : tanB = sinA * sinA / (cosA * cosA))
    (tanC_eq : tanC = sinA * sinA / (cosA * cosA))
    : (tanA / tanB) + (tanA / tanC) = 1 := sorry

end area_of_triangle_is_sqrt_5_sum_of_tangents_eq_1_l1140_114088


namespace reciprocal_neg_two_l1140_114097

theorem reciprocal_neg_two : 1 / (-2) = - (1 / 2) :=
by
  sorry

end reciprocal_neg_two_l1140_114097


namespace stream_current_rate_l1140_114007

theorem stream_current_rate (r w : ℝ) (h1 : 18 / (r + w) + 4 = 18 / (r - w))
  (h2 : 18 / (3 * r + w) + 2 = 18 / (3 * r - w)) : w = 3 :=
  sorry

end stream_current_rate_l1140_114007


namespace S_17_33_50_sum_l1140_114040

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    - (n / 2)
  else
    (n + 1) / 2

theorem S_17_33_50_sum : S 17 + S 33 + S 50 = 1 :=
by
  sorry

end S_17_33_50_sum_l1140_114040


namespace part1_part2_l1140_114029

theorem part1 (u v w : ℤ) (h_uv : gcd u v = 1) (h_vw : gcd v w = 1) (h_wu : gcd w u = 1) 
: gcd (u * v + v * w + w * u) (u * v * w) = 1 :=
sorry

theorem part2 (u v w : ℤ) (b := u * v + v * w + w * u) (c := u * v * w) (h : gcd b c = 1) 
: gcd u v = 1 ∧ gcd v w = 1 ∧ gcd w u = 1 :=
sorry

end part1_part2_l1140_114029


namespace average_squares_of_first_10_multiples_of_7_correct_l1140_114078

def first_10_multiples_of_7 : List ℕ := List.map (fun n => 7 * n) (List.range 10)

def squares (l : List ℕ) : List ℕ := List.map (fun n => n * n) l

def sum (l : List ℕ) : ℕ := List.foldr (· + ·) 0 l

theorem average_squares_of_first_10_multiples_of_7_correct :
  (sum (squares first_10_multiples_of_7) / 10 : ℚ) = 1686.5 :=
by
  sorry

end average_squares_of_first_10_multiples_of_7_correct_l1140_114078


namespace num_partitions_of_staircase_l1140_114003

-- Definition of a staircase
def is_staircase (n : ℕ) (cells : ℕ × ℕ → Prop) : Prop :=
  ∀ (i j : ℕ), 1 ≤ j → j ≤ i → i ≤ n → cells (i, j)

-- Number of partitions of a staircase of height n
def num_partitions (n : ℕ) : ℕ :=
  2^(n-1)

theorem num_partitions_of_staircase (n : ℕ) (cells : ℕ × ℕ → Prop) :
  is_staircase n cells → (∃ p : ℕ, p = num_partitions n) :=
by
  intro h
  use (2^(n-1))
  sorry

end num_partitions_of_staircase_l1140_114003


namespace range_of_a_l1140_114051

theorem range_of_a (a : ℝ) : (5 - a > 1) → (a < 4) := 
by
  sorry

end range_of_a_l1140_114051


namespace sin_neg_45_l1140_114074

theorem sin_neg_45 :
  Real.sin (-45 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
sorry

end sin_neg_45_l1140_114074


namespace largest_number_is_56_l1140_114067

-- Definitions based on the conditions
def ratio_three_five_seven (a b c : ℕ) : Prop :=
  3 * c = a ∧ 5 * c = b ∧ 7 * c = c

def difference_is_32 (a c : ℕ) : Prop :=
  c - a = 32

-- Statement of the proof
theorem largest_number_is_56 (a b c : ℕ) (h1 : ratio_three_five_seven a b c) (h2 : difference_is_32 a c) : c = 56 :=
by
  sorry

end largest_number_is_56_l1140_114067


namespace angle_sum_420_l1140_114076

theorem angle_sum_420 (A B C D E F : ℝ) (hE : E = 30) : 
  A + B + C + D + E + F = 420 :=
by
  sorry

end angle_sum_420_l1140_114076


namespace number_of_large_balls_l1140_114036

def smallBallRubberBands : ℕ := 50
def largeBallRubberBands : ℕ := 300
def totalRubberBands : ℕ := 5000
def smallBallsMade : ℕ := 22

def rubberBandsUsedForSmallBalls := smallBallsMade * smallBallRubberBands
def remainingRubberBands := totalRubberBands - rubberBandsUsedForSmallBalls

theorem number_of_large_balls :
  (remainingRubberBands / largeBallRubberBands) = 13 := by
  sorry

end number_of_large_balls_l1140_114036


namespace find_expression_for_f_l1140_114030

theorem find_expression_for_f (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = x^2 + 6 * x) :
  ∀ x, f x = x^2 + 8 * x + 7 :=
by
  sorry

end find_expression_for_f_l1140_114030


namespace simplify_expression_l1140_114095

theorem simplify_expression (θ : ℝ) : 
  ((1 + Real.sin θ ^ 2) ^ 2 - Real.cos θ ^ 4) * ((1 + Real.cos θ ^ 2) ^ 2 - Real.sin θ ^ 4) = 4 * Real.sin (2 * θ) ^ 2 :=
by 
  sorry

end simplify_expression_l1140_114095


namespace part_I_part_II_l1140_114035

variables {x a : ℝ} (p : Prop) (q : Prop)

-- Proposition p
def prop_p (x a : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0 ∧ a > 0

-- Proposition q
def prop_q (x : ℝ) : Prop := (x^2 - 2*x - 8 ≤ 0) ∧ (x^2 + 3*x - 10 > 0)

-- Part (I)
theorem part_I (a : ℝ) (h : a = 1) : (prop_p x a) → (prop_q x) → (2 < x ∧ x < 4) :=
by
  sorry

-- Part (II)
theorem part_II (a : ℝ) : ¬(∃ x, prop_p x a) → ¬(∃ x, prop_q x) → (1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end part_I_part_II_l1140_114035


namespace cello_viola_pairs_are_70_l1140_114008

-- Given conditions
def cellos : ℕ := 800
def violas : ℕ := 600
def pair_probability : ℝ := 0.00014583333333333335

-- Theorem statement translating the mathematical problem
theorem cello_viola_pairs_are_70 (n : ℕ) (h1 : cellos = 800) (h2 : violas = 600) (h3 : pair_probability = 0.00014583333333333335) :
  n = 70 :=
sorry

end cello_viola_pairs_are_70_l1140_114008


namespace ellen_smoothie_l1140_114019

theorem ellen_smoothie :
  let yogurt := 0.1
  let orange_juice := 0.2
  let total_ingredients := 0.5
  let strawberries_used := total_ingredients - (yogurt + orange_juice)
  strawberries_used = 0.2 := by
  sorry

end ellen_smoothie_l1140_114019


namespace jimmy_change_l1140_114004

def cost_of_pens (num_pens : ℕ) (cost_per_pen : ℕ): ℕ := num_pens * cost_per_pen
def cost_of_notebooks (num_notebooks : ℕ) (cost_per_notebook : ℕ): ℕ := num_notebooks * cost_per_notebook
def cost_of_folders (num_folders : ℕ) (cost_per_folder : ℕ): ℕ := num_folders * cost_per_folder

def total_cost : ℕ :=
  cost_of_pens 3 1 + cost_of_notebooks 4 3 + cost_of_folders 2 5

def paid_amount : ℕ := 50

theorem jimmy_change : paid_amount - total_cost = 25 := by
  sorry

end jimmy_change_l1140_114004


namespace triangle_area_is_24_l1140_114000

def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (0, 6)
def C : point := (8, 10)

def triangle_area (A B C : point) : ℝ := 
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_is_24 : triangle_area A B C = 24 :=
by
  -- Insert proof here
  sorry

end triangle_area_is_24_l1140_114000


namespace max_leap_years_l1140_114052

theorem max_leap_years (years : ℕ) (leap_interval : ℕ) (total_years : ℕ) (leap_years : ℕ)
  (h1 : leap_interval = 5)
  (h2 : total_years = 200)
  (h3 : years = total_years / leap_interval) :
  leap_years = 40 :=
by
  sorry

end max_leap_years_l1140_114052
