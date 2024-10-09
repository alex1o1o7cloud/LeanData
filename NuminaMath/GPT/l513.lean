import Mathlib

namespace at_least_one_less_than_zero_l513_51300

theorem at_least_one_less_than_zero {a b : ℝ} (h: a + b < 0) : a < 0 ∨ b < 0 := 
by 
  sorry

end at_least_one_less_than_zero_l513_51300


namespace geometric_sequence_a5_eq_neg1_l513_51328

-- Definitions for the conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

def roots_of_quadratic (a3 a7 : ℝ) : Prop :=
  a3 + a7 = -4 ∧ a3 * a7 = 1

-- The statement to prove
theorem geometric_sequence_a5_eq_neg1 {a : ℕ → ℝ}
  (h_geo : is_geometric_sequence a)
  (h_roots : roots_of_quadratic (a 3) (a 7)) :
  a 5 = -1 :=
sorry

end geometric_sequence_a5_eq_neg1_l513_51328


namespace number_of_refills_l513_51352

variable (totalSpent costPerRefill : ℕ)
variable (h1 : totalSpent = 40)
variable (h2 : costPerRefill = 10)

theorem number_of_refills (h1 h2 : totalSpent = 40) (h2 : costPerRefill = 10) :
  totalSpent / costPerRefill = 4 := by
  sorry

end number_of_refills_l513_51352


namespace initial_population_l513_51366

-- Define the initial population
variable (P : ℝ)

-- Define the conditions
theorem initial_population
  (h1 : P * 1.25 * 0.8 * 1.1 * 0.85 * 1.3 + 150 = 25000) :
  P = 24850 :=
by
  sorry

end initial_population_l513_51366


namespace tan_of_acute_angle_l513_51327

theorem tan_of_acute_angle (A : ℝ) (hA1 : 0 < A ∧ A < π / 2)
  (hA2 : 4 * (Real.sin A)^2 - 4 * Real.sin A * Real.cos A + (Real.cos A)^2 = 0) :
  Real.tan A = 1 / 2 :=
by
  sorry

end tan_of_acute_angle_l513_51327


namespace range_of_x_l513_51385

theorem range_of_x (a b x : ℝ) (h : a ≠ 0) 
  (ineq : |a + b| + |a - b| ≥ |a| * |x - 2|) : 
  0 ≤ x ∧ x ≤ 4 :=
  sorry

end range_of_x_l513_51385


namespace find_x_coordinate_l513_51347

theorem find_x_coordinate 
  (x : ℝ)
  (h1 : (0, 0) = (0, 0))
  (h2 : (0, 4) = (0, 4))
  (h3 : (x, 4) = (x, 4))
  (h4 : (x, 0) = (x, 0))
  (h5 : 0.4 * (4 * x) = 8)
  : x = 5 := 
sorry

end find_x_coordinate_l513_51347


namespace correct_average_weight_l513_51330

theorem correct_average_weight (avg_weight : ℝ) (num_boys : ℕ) (incorrect_weight correct_weight : ℝ)
  (h1 : avg_weight = 58.4) (h2 : num_boys = 20) (h3 : incorrect_weight = 56) (h4 : correct_weight = 62) :
  (avg_weight * ↑num_boys + (correct_weight - incorrect_weight)) / ↑num_boys = 58.7 := by
  sorry

end correct_average_weight_l513_51330


namespace part_one_part_two_l513_51397

-- Part (1)
theorem part_one (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a^2 + 3 * b^2 ≥ 2 * b * (a + b) :=
sorry

-- Part (2)
theorem part_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) : 
  2 * a + b = 8 :=
sorry

end part_one_part_two_l513_51397


namespace secant_length_l513_51363

theorem secant_length
  (A B C D E : ℝ)
  (AB : A - B = 7)
  (BC : B - C = 7)
  (AD : A - D = 10)
  (pos : A > E ∧ D > E):
  E - D = 0.2 :=
by
  sorry

end secant_length_l513_51363


namespace number_of_assembled_desks_and_chairs_students_cannot_complete_tasks_simultaneously_l513_51314

-- Defining the conditions
def wooden_boards_type_A := 400
def wooden_boards_type_B := 500
def desk_needs_type_A := 2
def desk_needs_type_B := 1
def chair_needs_type_A := 1
def chair_needs_type_B := 2
def total_students := 30
def desk_assembly_time := 10
def chair_assembly_time := 7

-- Theorem for the number of assembled desks and chairs
theorem number_of_assembled_desks_and_chairs :
  ∃ x y : ℕ, 2 * x + y = wooden_boards_type_A ∧ x + 2 * y = wooden_boards_type_B ∧ x = 100 ∧ y = 200 :=
by {
  sorry
}

-- Theorem for the feasibility of students completing the tasks simultaneously
theorem students_cannot_complete_tasks_simultaneously :
  ¬ ∃ a : ℕ, (a ≤ total_students) ∧ (total_students - a > 0) ∧ 
  (100 / a) * desk_assembly_time = (200 / (total_students - a)) * chair_assembly_time :=
by {
  sorry
}

end number_of_assembled_desks_and_chairs_students_cannot_complete_tasks_simultaneously_l513_51314


namespace solve_fabric_price_l513_51312

-- Defining the variables
variables (x y : ℕ)

-- Conditions as hypotheses
def condition1 := 7 * x = 9 * y
def condition2 := x - y = 36

-- Theorem statement to prove the system of equations
theorem solve_fabric_price (h1 : condition1 x y) (h2 : condition2 x y) :
  (7 * x = 9 * y) ∧ (x - y = 36) :=
by
  -- No proof is provided
  sorry

end solve_fabric_price_l513_51312


namespace arithmetic_sequence_seventh_term_l513_51360

variable (a1 a15 : ℚ)
variable (n : ℕ) (a7 : ℚ)

-- Given conditions
def first_term (a1 : ℚ) : Prop := a1 = 3
def last_term (a15 : ℚ) : Prop := a15 = 72
def total_terms (n : ℕ) : Prop := n = 15

-- Arithmetic sequence formula
def common_difference (d : ℚ) : Prop := d = (72 - 3) / (15 - 1)
def nth_term (a_n : ℚ) (a1 : ℚ) (n : ℕ) (d : ℚ) : Prop := a_n = a1 + (n - 1) * d

-- Prove that the 7th term is approximately 33
theorem arithmetic_sequence_seventh_term :
  ∀ (a1 a15 : ℚ) (n : ℕ), first_term a1 → last_term a15 → total_terms n → ∃ a7 : ℚ, 
  nth_term a7 a1 7 ((a15 - a1) / (n - 1)) ∧ (33 - 0.5) < a7 ∧ a7 < (33 + 0.5) :=
by {
  sorry
}

end arithmetic_sequence_seventh_term_l513_51360


namespace min_reciprocal_sum_l513_51370

theorem min_reciprocal_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1) : 
  (1 / a) + (1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_reciprocal_sum_l513_51370


namespace solve_quadratic_l513_51325

theorem solve_quadratic : ∃ x : ℝ, (x^2 - 2 * x - 8 = 0 ↔ x = 4 ∨ x = -2) :=
sorry

end solve_quadratic_l513_51325


namespace find_triples_l513_51368

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem find_triples (a b c : ℕ) :
  is_prime (a^2 + 1) ∧
  is_prime (b^2 + 1) ∧
  (a^2 + 1) * (b^2 + 1) = c^2 + 1 →
  (a = 1 ∧ b = 2 ∧ c = 3) ∨ (a = 2 ∧ b = 1 ∧ c = 3) :=
by
  sorry

end find_triples_l513_51368


namespace total_profit_l513_51398

theorem total_profit (investment_B : ℝ) (period_B : ℝ) (profit_B : ℝ) (investment_A : ℝ) (period_A : ℝ) (total_profit : ℝ)
  (h1 : investment_A = 3 * investment_B)
  (h2 : period_A = 2 * period_B)
  (h3 : profit_B = 6000)
  (h4 : profit_B / (profit_A * 6 + profit_B) = profit_B) : total_profit = 7 * 6000 :=
by 
  sorry

#print axioms total_profit

end total_profit_l513_51398


namespace g_negative_example1_g_negative_example2_g_negative_example3_l513_51389

noncomputable def g (a : ℚ) : ℚ := sorry

axiom g_mul (a b : ℚ) (ha : 0 < a) (hb : 0 < b) : g (a * b) = g a + g b
axiom g_prime (p : ℕ) (hp : Nat.Prime p) : g (p * p) = p

theorem g_negative_example1 : g (8/81) < 0 := sorry
theorem g_negative_example2 : g (25/72) < 0 := sorry
theorem g_negative_example3 : g (49/18) < 0 := sorry

end g_negative_example1_g_negative_example2_g_negative_example3_l513_51389


namespace complex_exp_power_cos_angle_l513_51367

theorem complex_exp_power_cos_angle (z : ℂ) (h : z + 1/z = 2 * Complex.cos (Real.pi / 36)) :
    z^1000 + 1/(z^1000) = 2 * Complex.cos (Real.pi * 2 / 9) :=
by
  sorry

end complex_exp_power_cos_angle_l513_51367


namespace rectangle_area_eq_l513_51359

theorem rectangle_area_eq (a b c d x y z w : ℝ)
  (h1 : a = x + y) (h2 : b = y + z) (h3 : c = z + w) (h4 : d = w + x) :
  a + c = b + d :=
by
  sorry

end rectangle_area_eq_l513_51359


namespace pentagon_area_l513_51337

open Function 

/-
Given a convex pentagon FGHIJ with the following properties:
  1. ∠F = ∠G = 100°
  2. JF = FG = GH = 3
  3. HI = IJ = 5
Prove that the area of pentagon FGHIJ is approximately 15.2562 square units.
-/

noncomputable def area_pentagon_FGHIJ : ℝ :=
  let sin100 := Real.sin (100 * Real.pi / 180)
  let area_FGJ := (3 * 3 * sin100) / 2
  let area_HIJ := (5 * 5 * Real.sqrt 3) / 4
  area_FGJ + area_HIJ

theorem pentagon_area : abs (area_pentagon_FGHIJ - 15.2562) < 0.0001 := by
  sorry

end pentagon_area_l513_51337


namespace trick_proof_l513_51383

-- Defining the number of fillings and total pastries based on combinations
def num_fillings := 10

def total_pastries : ℕ := (num_fillings * (num_fillings - 1)) / 2

-- Definition stating that the smallest number of pastries n such that Vasya can always determine at least one filling of any remaining pastry
def min_n := 36

-- The theorem stating the proof problem
theorem trick_proof (n m: ℕ) (h1: n = 10) (h2: m = (n * (n - 1)) / 2) : min_n = 36 :=
by
  sorry

end trick_proof_l513_51383


namespace total_pencils_l513_51320

theorem total_pencils  (a b c : Nat) (total : Nat) 
(h₀ : a = 43) 
(h₁ : b = 19) 
(h₂ : c = 16) 
(h₃ : total = a + b + c) : 
total = 78 := 
by
  sorry

end total_pencils_l513_51320


namespace probability_sum_5_l513_51343

theorem probability_sum_5 :
  let total_outcomes := 36
  let favorable_outcomes := 4
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 9 :=
by
  -- proof omitted
  sorry

end probability_sum_5_l513_51343


namespace subtraction_of_fractions_l513_51336

theorem subtraction_of_fractions : (5 / 9) - (1 / 6) = 7 / 18 :=
by
  sorry

end subtraction_of_fractions_l513_51336


namespace length_of_shorter_train_l513_51334

noncomputable def relativeSpeedInMS (speed1_kmh speed2_kmh : ℝ) : ℝ :=
  (speed1_kmh + speed2_kmh) * (5 / 18)

noncomputable def totalDistanceCovered (relativeSpeed_ms time_s : ℝ) : ℝ :=
  relativeSpeed_ms * time_s

noncomputable def lengthOfShorterTrain (longerTrainLength_m time_s : ℝ) (speed1_kmh speed2_kmh : ℝ) : ℝ :=
  let relativeSpeed_ms := relativeSpeedInMS speed1_kmh speed2_kmh
  let totalDistance := totalDistanceCovered relativeSpeed_ms time_s
  totalDistance - longerTrainLength_m

theorem length_of_shorter_train :
  lengthOfShorterTrain 160 10.07919366450684 60 40 = 117.8220467912412 := 
sorry

end length_of_shorter_train_l513_51334


namespace toy_value_l513_51331

theorem toy_value
  (t : ℕ)                 -- total number of toys
  (W : ℕ)                 -- total worth in dollars
  (v : ℕ)                 -- value of one specific toy
  (x : ℕ)                 -- value of one of the other toys
  (h1 : t = 9)            -- condition 1: total number of toys
  (h2 : W = 52)           -- condition 2: total worth
  (h3 : v = 12)           -- condition 3: value of one specific toy
  (h4 : (t - 1) * x + v = W) -- condition 4: equation based on the problem
  : x = 5 :=              -- theorem statement: other toy's value
by {
  -- proof goes here
  sorry
}

end toy_value_l513_51331


namespace interior_triangle_area_l513_51376

theorem interior_triangle_area (a b c : ℝ)
  (ha : a^2 = 36) (hb : b^2 = 64) (hc : c^2 = 100)
  (hpythagorean : a^2 + b^2 = c^2) :
  1/2 * a * b = 24 :=
by
  sorry

end interior_triangle_area_l513_51376


namespace hyperbola_center_l513_51302

theorem hyperbola_center :
  ∃ (center : ℝ × ℝ), center = (2.5, 4) ∧
    (∀ x y : ℝ, 9 * x^2 - 45 * x - 16 * y^2 + 128 * y + 207 = 0 ↔ 
      (1/1503) * (36 * (x - 2.5)^2 - 64 * (y - 4)^2) = 1) :=
sorry

end hyperbola_center_l513_51302


namespace last_four_digits_5_pow_2017_l513_51364

theorem last_four_digits_5_pow_2017 : (5 ^ 2017) % 10000 = 3125 :=
by sorry

end last_four_digits_5_pow_2017_l513_51364


namespace find_a_l513_51355

theorem find_a (x : ℝ) (a : ℝ)
  (h1 : 3 * x - 4 = a)
  (h2 : (x + a) / 3 = 1)
  (h3 : (x = (a + 4) / 3) → (x = 3 - a → ((a + 4) / 3 = 2 * (3 - a)))) :
  a = 2 :=
sorry

end find_a_l513_51355


namespace lines_coplanar_parameter_l513_51318

/-- 
  Two lines are given in parametric form: 
  L1: (2 + 2s, 4s, -3 + rs)
  L2: (-1 + 3t, 2t, 1 + 2t)
  Prove that if these lines are coplanar, then r = 4.
-/
theorem lines_coplanar_parameter (s t r : ℝ) :
  ∃ (k : ℝ), 
  (∀ s t, 
    ∃ (k₁ k₂ : ℝ), k₁ * k₂ ≠ 0
      ∧
      (2 + 2 * s, 4 * s, -3 + r * s) = (k * (-1 + 3 * t), k * 2 * t, k * (1 + 2 * t))
  ) → r = 4 := sorry

end lines_coplanar_parameter_l513_51318


namespace platform_length_l513_51307

theorem platform_length 
  (train_length : ℝ) (train_speed_kmph : ℝ) (time_s : ℝ) (platform_length : ℝ)
  (H1 : train_length = 360) 
  (H2 : train_speed_kmph = 45) 
  (H3 : time_s = 40)
  (H4 : platform_length = (train_speed_kmph * 1000 / 3600 * time_s) - train_length ) :
  platform_length = 140 :=
by {
 sorry
}

end platform_length_l513_51307


namespace probability_of_matching_pair_l513_51341

def total_socks : ℕ := 12 + 6 + 9
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

def black_pairs : ℕ := choose_two 12
def white_pairs : ℕ := choose_two 6
def blue_pairs : ℕ := choose_two 9

def total_pairs : ℕ := choose_two total_socks
def matching_pairs : ℕ := black_pairs + white_pairs + blue_pairs

def probability : ℚ := matching_pairs / total_pairs

theorem probability_of_matching_pair :
  probability = 1 / 3 :=
by
  -- The proof will go here
  sorry

end probability_of_matching_pair_l513_51341


namespace cement_mixture_weight_l513_51339

theorem cement_mixture_weight 
  (W : ℝ)
  (h1 : W = (2/5) * W + (1/6) * W + (1/10) * W + (1/8) * W + 12) :
  W = 57.6 := by
  sorry

end cement_mixture_weight_l513_51339


namespace complete_square_form_l513_51309

theorem complete_square_form (x : ℝ) (a : ℝ) 
  (h : x^2 - 2 * x - 4 = 0) : (x - 1)^2 = a ↔ a = 5 :=
by
  sorry

end complete_square_form_l513_51309


namespace nth_row_equation_l513_51304

theorem nth_row_equation (n : ℕ) : 2 * n + 1 = (n + 1) ^ 2 - n ^ 2 := 
sorry

end nth_row_equation_l513_51304


namespace proof_range_of_a_l513_51316

/-- p is the proposition that for all x in [1,2], x^2 - a ≥ 0 --/
def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

/-- q is the proposition that there exists an x0 in ℝ such that x0^2 + (a-1)x0 + 1 < 0 --/
def q (a : ℝ) : Prop := ∃ x0 : ℝ, x0^2 + (a-1)*x0 + 1 < 0

theorem proof_range_of_a (a : ℝ) : (p a ∨ q a) ∧ (¬p a ∧ ¬q a) → (a ≥ -1 ∧ a ≤ 1) ∨ a > 3 :=
by
  sorry -- proof will be filled out here

end proof_range_of_a_l513_51316


namespace hyperbola_foci_distance_l513_51310

-- Definitions based on the problem conditions
def hyperbola (x y : ℝ) : Prop := x^2 - (y^2 / 9) = 1

def foci_distance (PF1 : ℝ) : Prop := PF1 = 5

-- Main theorem stating the problem and expected outcome
theorem hyperbola_foci_distance (x y PF2 : ℝ) 
  (P_on_hyperbola : hyperbola x y) 
  (PF1_dist : foci_distance (dist (x, y) (some_focal_point_x1, 0))) :
  dist (x, y) (some_focal_point_x2, 0) = 7 ∨ dist (x, y) (some_focal_point_x2, 0) = 3 :=
sorry

end hyperbola_foci_distance_l513_51310


namespace range_of_a_l513_51345

noncomputable
def proposition_p (x : ℝ) : Prop := abs (x - (3 / 4)) <= (1 / 4)
noncomputable
def proposition_q (x a : ℝ) : Prop := (x - a) * (x - a - 1) <= 0

theorem range_of_a :
  (∀ x : ℝ, proposition_p x → ∃ x : ℝ, proposition_q x a) ∧
  (∃ x : ℝ, ¬(proposition_p x → proposition_q x a )) →
  0 ≤ a ∧ a ≤ (1 / 2) :=
sorry

end range_of_a_l513_51345


namespace right_triangle_median_to_hypotenuse_l513_51381

theorem right_triangle_median_to_hypotenuse 
    {DEF : Type} [MetricSpace DEF] 
    (D E F M : DEF) 
    (h_triangle : dist D E = 15 ∧ dist D F = 20 ∧ dist E F = 25) 
    (h_midpoint : dist D M = dist E M ∧ dist D E = 2 * dist D M ∧ dist E F * dist E F = dist E D * dist E D + dist D F * dist D F) :
    dist F M = 12.5 :=
by sorry

end right_triangle_median_to_hypotenuse_l513_51381


namespace derivative_y_l513_51372

open Real

noncomputable def y (x : ℝ) : ℝ :=
  log (2 * x - 3 + sqrt (4 * x ^ 2 - 12 * x + 10)) -
  sqrt (4 * x ^ 2 - 12 * x + 10) * arctan (2 * x - 3)

theorem derivative_y (x : ℝ) : 
  (deriv y x) = - arctan (2 * x - 3) / sqrt (4 * x ^ 2 - 12 * x + 10) :=
by
  sorry

end derivative_y_l513_51372


namespace arithmetic_square_root_16_l513_51323

theorem arithmetic_square_root_16 : ∃ (x : ℝ), x * x = 16 ∧ x ≥ 0 ∧ x = 4 := by
  sorry

end arithmetic_square_root_16_l513_51323


namespace Tia_drove_192_more_miles_l513_51393

noncomputable def calculate_additional_miles (s_C t_C : ℝ) : ℝ :=
  let d_C := s_C * t_C
  let d_M := (s_C + 8) * (t_C + 3)
  let d_T := (s_C + 12) * (t_C + 4)
  d_T - d_C

theorem Tia_drove_192_more_miles (s_C t_C : ℝ) (h1 : d_M = d_C + 120) (h2 : d_M = (s_C + 8) * (t_C + 3)) : calculate_additional_miles s_C t_C = 192 :=
by {
  sorry
}

end Tia_drove_192_more_miles_l513_51393


namespace cross_area_l513_51344

variables (R : ℝ) (A : ℝ × ℝ) (φ : ℝ)
  -- Radius R of the circle, Point A inside the circle, and angle φ in radians

-- Define the area of the cross formed by rotated lines
def area_of_cross (R : ℝ) (φ : ℝ) : ℝ :=
  2 * φ * R^2

theorem cross_area (R : ℝ) (A : ℝ × ℝ) (φ : ℝ) (hR : 0 < R) (hA : dist A (0, 0) < R) :
  area_of_cross R φ = 2 * φ * R^2 := 
sorry

end cross_area_l513_51344


namespace time_45_minutes_after_10_20_is_11_05_l513_51313

def time := Nat × Nat -- Represents time as (hours, minutes)

noncomputable def add_minutes (t : time) (m : Nat) : time :=
  let (hours, minutes) := t
  let total_minutes := minutes + m
  let new_hours := hours + total_minutes / 60
  let new_minutes := total_minutes % 60
  (new_hours, new_minutes)

theorem time_45_minutes_after_10_20_is_11_05 :
  add_minutes (10, 20) 45 = (11, 5) :=
  sorry

end time_45_minutes_after_10_20_is_11_05_l513_51313


namespace uncle_ben_eggs_l513_51369

noncomputable def total_eggs (total_chickens : ℕ) (roosters : ℕ) (non_egg_laying_hens : ℕ) (eggs_per_hen : ℕ) : ℕ :=
  let total_hens := total_chickens - roosters
  let egg_laying_hens := total_hens - non_egg_laying_hens
  egg_laying_hens * eggs_per_hen

theorem uncle_ben_eggs :
  total_eggs 440 39 15 3 = 1158 :=
by
  unfold total_eggs
  -- Correct steps to prove the theorem can be skipped with sorry
  sorry

end uncle_ben_eggs_l513_51369


namespace biased_die_expected_value_is_neg_1_5_l513_51322

noncomputable def biased_die_expected_value : ℚ :=
  let prob_123 := (1 / 6 : ℚ) + (1 / 6) + (1 / 6)
  let prob_456 := (1 / 2 : ℚ)
  let gain := prob_123 * 2
  let loss := prob_456 * -5
  gain + loss

theorem biased_die_expected_value_is_neg_1_5 :
  biased_die_expected_value = - (3 / 2 : ℚ) :=
by
  -- We skip the detailed proof steps here.
  sorry

end biased_die_expected_value_is_neg_1_5_l513_51322


namespace people_on_train_after_third_stop_l513_51375

variable (initial_people : ℕ) (off_1 boarded_1 off_2 boarded_2 off_3 boarded_3 : ℕ)

def people_after_first_stop (initial : ℕ) (off_1 boarded_1 : ℕ) : ℕ :=
  initial - off_1 + boarded_1

def people_after_second_stop (first_stop : ℕ) (off_2 boarded_2 : ℕ) : ℕ :=
  first_stop - off_2 + boarded_2

def people_after_third_stop (second_stop : ℕ) (off_3 boarded_3 : ℕ) : ℕ :=
  second_stop - off_3 + boarded_3

theorem people_on_train_after_third_stop :
  people_after_third_stop (people_after_second_stop (people_after_first_stop initial_people off_1 boarded_1) off_2 boarded_2) off_3 boarded_3 = 42 :=
  by
    have initial_people := 48
    have off_1 := 12
    have boarded_1 := 7
    have off_2 := 15
    have boarded_2 := 9
    have off_3 := 6
    have boarded_3 := 11
    sorry

end people_on_train_after_third_stop_l513_51375


namespace class_ratio_and_percentage_l513_51317

theorem class_ratio_and_percentage:
  ∀ (female male : ℕ), female = 15 → male = 25 →
  (∃ ratio_n ratio_d : ℕ, gcd ratio_n ratio_d = 1 ∧ ratio_n = 5 ∧ ratio_d = 8 ∧
  ratio_n / ratio_d = male / (female + male))
  ∧
  (∃ percentage : ℕ, percentage = 40 ∧ percentage = 100 * (male - female) / male) :=
by
  intros female male hf hm
  have h1 : female = 15 := hf
  have h2 : male = 25 := hm
  sorry

end class_ratio_and_percentage_l513_51317


namespace evaluate_expression_l513_51329

-- Given conditions 
def x := 3
def y := 2

-- Prove that y + y(y^x + x!) evaluates to 30.
theorem evaluate_expression : y + y * (y^x + Nat.factorial x) = 30 := by
  sorry

end evaluate_expression_l513_51329


namespace defect_rate_probability_l513_51356

theorem defect_rate_probability (p : ℝ) (n : ℕ) (ε : ℝ) (q : ℝ) : 
  p = 0.02 →
  n = 800 →
  ε = 0.01 →
  q = 1 - p →
  1 - (p * q) / (n * ε^2) = 0.755 :=
by
  intro hp hn he hq
  rw [hp, hn, he, hq]
  -- Calculation steps can be verified here
  sorry

end defect_rate_probability_l513_51356


namespace eleonora_age_l513_51326

-- Definitions
def age_eleonora (e m : ℕ) : Prop :=
m - e = 3 * (2 * e - m) ∧ 3 * e + (m + 2 * e) = 100

-- Theorem stating that Eleonora's age is 15
theorem eleonora_age (e m : ℕ) (h : age_eleonora e m) : e = 15 :=
sorry

end eleonora_age_l513_51326


namespace line_through_P_origin_line_through_P_perpendicular_to_l3_l513_51377

-- Define lines l1, l2, l3
def l1 (x y : ℝ) := 3 * x + 4 * y - 2 = 0
def l2 (x y : ℝ) := 2 * x + y + 2 = 0
def l3 (x y : ℝ) := x - 2 * y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Prove the equations of the lines passing through P
theorem line_through_P_origin : ∃ A B C : ℝ, A * -2 + B * 2 + C = 0 ∧ A * 0 + B * 0 + C = 0 ∧ A = 1 ∧ B = 1 ∧ C = 0 :=
by sorry

theorem line_through_P_perpendicular_to_l3 : ∃ A B C : ℝ, A * -2 + B * 2 + C = 0 ∧ A * P.1 + B * P.2 + C = 0 ∧ A = 2 ∧ B = 1 ∧ C = 2 :=
by sorry

end line_through_P_origin_line_through_P_perpendicular_to_l3_l513_51377


namespace smallest_number_satisfying_conditions_l513_51357

theorem smallest_number_satisfying_conditions :
  ∃ (n : ℕ), n % 6 = 2 ∧ n % 7 = 3 ∧ n % 8 = 4 ∧ ∀ m, (m % 6 = 2 → m % 7 = 3 → m % 8 = 4 → n ≤ m) :=
  sorry

end smallest_number_satisfying_conditions_l513_51357


namespace scientific_notation_of_0_000000032_l513_51365

theorem scientific_notation_of_0_000000032 :
  0.000000032 = 3.2 * 10^(-8) :=
by
  -- skipping the proof
  sorry

end scientific_notation_of_0_000000032_l513_51365


namespace rate_up_the_mountain_l513_51346

noncomputable def mountain_trip_rate (R : ℝ) : ℝ := 1.5 * R

theorem rate_up_the_mountain : 
  ∃ R : ℝ, (2 * 1.5 * R = 18) ∧ (1.5 * R = 9) → R = 6 :=
by
  sorry

end rate_up_the_mountain_l513_51346


namespace intersection_A_B_at_1_range_of_a_l513_51338

-- Problem definitions
def set_A (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def set_B (x a : ℝ) : Prop := x^2 - 2*a*x - 1 ≤ 0 ∧ a > 0

-- Question (I) If a = 1, find A ∩ B
theorem intersection_A_B_at_1 : (∀ x : ℝ, set_A x ∧ set_B x 1 ↔ (1 < x ∧ x ≤ 1 + Real.sqrt 2)) := sorry

-- Question (II) If A ∩ B contains exactly one integer, find the range of a.
theorem range_of_a (h : ∃ x : ℤ, set_A x ∧ set_B x 2) : 3 / 4 ≤ 2 ∧ 2 < 4 / 3 := sorry

end intersection_A_B_at_1_range_of_a_l513_51338


namespace soda_cost_l513_51374

-- Definitions based on conditions of the problem
variable (b s : ℤ)
variable (h1 : 4 * b + 3 * s = 540)
variable (h2 : 3 * b + 2 * s = 390)

-- The theorem to prove the cost of a soda
theorem soda_cost : s = 60 := by
  sorry

end soda_cost_l513_51374


namespace larger_segment_length_l513_51362

theorem larger_segment_length (a b c : ℕ) (h : ℝ) (x : ℝ)
  (ha : a = 50) (hb : b = 90) (hc : c = 110)
  (hyp1 : a^2 = x^2 + h^2)
  (hyp2 : b^2 = (c - x)^2 + h^2) :
  110 - x = 80 :=
by {
  sorry
}

end larger_segment_length_l513_51362


namespace clarence_oranges_left_l513_51386

-- Definitions based on the conditions in the problem
def initial_oranges : ℕ := 5
def oranges_from_joyce : ℕ := 3
def total_oranges_after_joyce : ℕ := initial_oranges + oranges_from_joyce
def oranges_given_to_bob : ℕ := total_oranges_after_joyce / 2
def oranges_left : ℕ := total_oranges_after_joyce - oranges_given_to_bob

-- Proof statement that needs to be proven
theorem clarence_oranges_left : oranges_left = 4 :=
by
  sorry

end clarence_oranges_left_l513_51386


namespace k_9_pow_4_eq_81_l513_51373

theorem k_9_pow_4_eq_81 
  (h k : ℝ → ℝ) 
  (hk1 : ∀ (x : ℝ), x ≥ 1 → h (k x) = x^3) 
  (hk2 : ∀ (x : ℝ), x ≥ 1 → k (h x) = x^4) 
  (k81_eq_9 : k 81 = 9) :
  (k 9)^4 = 81 :=
by
  sorry

end k_9_pow_4_eq_81_l513_51373


namespace expression_evaluation_l513_51388

theorem expression_evaluation : |1 - Real.sqrt 3| + 2 * Real.cos (Real.pi / 6) - Real.sqrt 12 - 2023 = -2024 := 
by {
    sorry
}

end expression_evaluation_l513_51388


namespace find_M_l513_51380

theorem find_M : 995 + 997 + 999 + 1001 + 1003 = 5100 - 104 :=
by 
  sorry

end find_M_l513_51380


namespace ratio_of_two_numbers_l513_51349

variable {a b : ℝ}

theorem ratio_of_two_numbers
  (h1 : a + b = 7 * (a - b))
  (h2 : 0 < b)
  (h3 : a > b) :
  a / b = 4 / 3 := by
  sorry

end ratio_of_two_numbers_l513_51349


namespace part1_part2_l513_51321

noncomputable def f (x a : ℝ) := x * Real.log (x + 1) + (1/2 - a) * x + 2 - a

noncomputable def g (x a : ℝ) := f x a + Real.log (x + 1) + 1/2 * x

theorem part1 (a : ℝ) (x : ℝ) (h : x > 0) : 
  (a ≤ 2 → ∀ x, g x a > 0) ∧ 
  (a > 2 → ∀ x, x < Real.exp (a - 2) - 1 → g x a < 0) ∧
  (a > 2 → ∀ x, x > Real.exp (a - 2) - 1 → g x a > 0) :=
sorry

theorem part2 (a : ℤ) : 
  (∃ x ≥ 0, f x a < 0) → a ≥ 3 :=
sorry

end part1_part2_l513_51321


namespace art_gallery_total_pieces_l513_51315

theorem art_gallery_total_pieces :
  ∃ T : ℕ, 
    (1/3 : ℝ) * T + (2/3 : ℝ) * (1/3 : ℝ) * T + 400 + 3 * (1/18 : ℝ) * T + 2 * (1/18 : ℝ) * T = T :=
sorry

end art_gallery_total_pieces_l513_51315


namespace least_possible_BC_l513_51301

-- Define given lengths
def AB := 7 -- cm
def AC := 18 -- cm
def DC := 10 -- cm
def BD := 25 -- cm

-- Define the proof statement
theorem least_possible_BC : 
  ∃ (BC : ℕ), (BC > AC - AB) ∧ (BC > BD - DC) ∧ BC = 16 := by
  sorry

end least_possible_BC_l513_51301


namespace total_spent_l513_51308

-- Define the conditions
def cost_fix_automobile := 350
def cost_fix_formula (S : ℕ) := 3 * S + 50

-- Prove the total amount spent is $450
theorem total_spent (S : ℕ) (h : cost_fix_automobile = cost_fix_formula S) :
  S + cost_fix_automobile = 450 :=
by
  sorry

end total_spent_l513_51308


namespace running_speed_is_24_l513_51333

def walk_speed := 8 -- km/h
def walk_time := 3 -- hours
def run_time := 1 -- hour

def walk_distance := walk_speed * walk_time

def run_speed := walk_distance / run_time

theorem running_speed_is_24 : run_speed = 24 := 
by
  sorry

end running_speed_is_24_l513_51333


namespace convert_length_convert_area_convert_time_convert_mass_l513_51335

theorem convert_length (cm : ℕ) : cm = 7 → (cm : ℚ) / 100 = 7 / 100 :=
by sorry

theorem convert_area (dm2 : ℕ) : dm2 = 35 → (dm2 : ℚ) / 100 = 7 / 20 :=
by sorry

theorem convert_time (min : ℕ) : min = 45 → (min : ℚ) / 60 = 3 / 4 :=
by sorry

theorem convert_mass (g : ℕ) : g = 2500 → (g : ℚ) / 1000 = 5 / 2 :=
by sorry

end convert_length_convert_area_convert_time_convert_mass_l513_51335


namespace c_plus_d_l513_51395

theorem c_plus_d (c d : ℝ)
  (h1 : c^3 - 12 * c^2 + 15 * c - 36 = 0)
  (h2 : 6 * d^3 - 36 * d^2 - 150 * d + 1350 = 0) :
  c + d = 7 := 
  sorry

end c_plus_d_l513_51395


namespace students_not_coming_l513_51396

-- Define the conditions
def pieces_per_student : ℕ := 4
def pieces_made_last_monday : ℕ := 40
def pieces_made_upcoming_monday : ℕ := 28

-- Define the number of students not coming to class
theorem students_not_coming :
  (pieces_made_last_monday / pieces_per_student) - 
  (pieces_made_upcoming_monday / pieces_per_student) = 3 :=
by sorry

end students_not_coming_l513_51396


namespace george_blocks_l513_51391

theorem george_blocks (num_boxes : ℕ) (blocks_per_box : ℕ) (total_blocks : ℕ) :
  num_boxes = 2 → blocks_per_box = 6 → total_blocks = num_boxes * blocks_per_box → total_blocks = 12 := by
  intros h_num_boxes h_blocks_per_box h_blocks_equal
  rw [h_num_boxes, h_blocks_per_box] at h_blocks_equal
  exact h_blocks_equal

end george_blocks_l513_51391


namespace digit_B_l513_51305

def is_valid_digit (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 7

def unique_digits (A B C D E F G : ℕ) : Prop :=
  is_valid_digit A ∧ is_valid_digit B ∧ is_valid_digit C ∧ is_valid_digit D ∧ 
  is_valid_digit E ∧ is_valid_digit F ∧ is_valid_digit G ∧ 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ 
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ 
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ 
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ 
  E ≠ F ∧ E ≠ G ∧ 
  F ≠ G

def total_sum (A B C D E F G : ℕ) : ℕ :=
  (A + B + C) + (A + E + F) + (C + D + E) + (B + D + G) + (B + F) + (G + E)

theorem digit_B (A B C D E F G : ℕ) 
  (h1 : unique_digits A B C D E F G)
  (h2 : total_sum A B C D E F G = 65) : B = 7 := 
sorry

end digit_B_l513_51305


namespace cannot_determine_right_triangle_l513_51350

def is_right_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

theorem cannot_determine_right_triangle :
  ∀ A B C : ℝ, 
    (A = 2 * B ∧ A = 3 * C) →
    ¬ is_right_triangle A B C :=
by
  intro A B C h
  have h1 : A = 2 * B := h.1
  have h2 : A = 3 * C := h.2
  sorry

end cannot_determine_right_triangle_l513_51350


namespace solution_l513_51371

theorem solution (y : ℚ) (h : (1/3 : ℚ) + 1/y = 7/9) : y = 9/4 :=
by
  sorry

end solution_l513_51371


namespace no_perfect_squares_in_sequence_l513_51306

def tau (a : ℕ) : ℕ := sorry -- Define tau function here

def a_seq (k : ℕ) (n : ℕ) : ℕ :=
  if n = 0 then k else tau (a_seq k (n-1))

theorem no_perfect_squares_in_sequence (k : ℕ) (hk : Prime k) :
  ∀ n : ℕ, ∃ m : ℕ, a_seq k n = m * m → False :=
sorry

end no_perfect_squares_in_sequence_l513_51306


namespace find_parabola_vertex_l513_51311

-- Define the parabola with specific roots.
def parabola (x : ℝ) : ℝ := -x^2 + 2 * x + 24

-- Define the vertex of the parabola.
def vertex : ℝ × ℝ := (1, 25)

-- Prove that the vertex of the parabola is indeed at (1, 25).
theorem find_parabola_vertex : vertex = (1, 25) :=
  sorry

end find_parabola_vertex_l513_51311


namespace percent_of_men_tenured_l513_51399

theorem percent_of_men_tenured (total_professors : ℕ) (women_percent tenured_percent women_tenured_or_both_percent men_percent tenured_men_percent : ℝ)
  (h1 : women_percent = 70 / 100)
  (h2 : tenured_percent = 70 / 100)
  (h3 : women_tenured_or_both_percent = 90 / 100)
  (h4 : men_percent = 30 / 100)
  (h5 : total_professors > 0)
  (h6 : tenured_men_percent = (2/3)) :
  tenured_men_percent * 100 = 66.67 :=
by sorry

end percent_of_men_tenured_l513_51399


namespace cooper_pies_days_l513_51394

theorem cooper_pies_days :
  ∃ d : ℕ, 7 * d - 50 = 34 ∧ d = 12 :=
by
  sorry

end cooper_pies_days_l513_51394


namespace sum_of_digits_of_2010_l513_51361

noncomputable def sum_of_base6_digits (n : ℕ) : ℕ :=
  (n.digits 6).sum

theorem sum_of_digits_of_2010 : sum_of_base6_digits 2010 = 10 := by
  sorry

end sum_of_digits_of_2010_l513_51361


namespace new_average_age_l513_51390

theorem new_average_age (n_students : ℕ) (average_student_age : ℕ) (teacher_age : ℕ)
  (h_students : n_students = 50)
  (h_average_student_age : average_student_age = 14)
  (h_teacher_age : teacher_age = 65) :
  (n_students * average_student_age + teacher_age) / (n_students + 1) = 15 :=
by
  sorry

end new_average_age_l513_51390


namespace tan_angle_sum_l513_51384

theorem tan_angle_sum
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  Real.tan (α + π / 4) = 3 / 22 :=
by
  sorry

end tan_angle_sum_l513_51384


namespace probability_of_region_C_l513_51324

theorem probability_of_region_C (pA pB pC : ℚ) 
  (h1 : pA = 1/2) 
  (h2 : pB = 1/5) 
  (h3 : pA + pB + pC = 1) : 
  pC = 3/10 := 
sorry

end probability_of_region_C_l513_51324


namespace ratio_of_nuts_to_raisins_l513_51348

theorem ratio_of_nuts_to_raisins 
  (R N : ℝ) 
  (h_ratio : 3 * R = 0.2727272727272727 * (3 * R + 4 * N)) : 
  N = 2 * R := 
sorry

end ratio_of_nuts_to_raisins_l513_51348


namespace right_triangle_can_form_isosceles_l513_51303

-- Definitions for the problem
structure RightTriangle :=
  (a b : ℝ) -- The legs of the right triangle
  (c : ℝ)  -- The hypotenuse of the right triangle
  (h1 : c = Real.sqrt (a ^ 2 + b ^ 2)) -- Pythagoras theorem

-- The triangle attachment requirement definition
def IsoscelesTriangleAttachment (rightTriangle : RightTriangle) : Prop :=
  ∃ (b1 b2 : ℝ), -- Two base sides of the new triangle sharing one side with the right triangle
    (b1 ≠ b2) ∧ -- They should be different to not overlap
    (b1 = rightTriangle.a ∨ b1 = rightTriangle.b) ∧ -- Share one side with the right triangle
    (b2 ≠ rightTriangle.a ∧ b2 ≠ rightTriangle.b) ∧ -- Ensure non-overlapping
    (b1^2 + b2^2 = rightTriangle.c^2)

-- The statement to prove
theorem right_triangle_can_form_isosceles (T : RightTriangle) : IsoscelesTriangleAttachment T :=
sorry

end right_triangle_can_form_isosceles_l513_51303


namespace increasing_exponential_function_l513_51382

theorem increasing_exponential_function (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) :
  (∀ x y : ℝ, x < y → (a ^ x) < (a ^ y)) → (1 < a) :=
by
  sorry

end increasing_exponential_function_l513_51382


namespace percentage_discount_l513_51319

theorem percentage_discount (C S S' : ℝ) (h1 : S = 1.14 * C) (h2 : S' = 2.20 * C) :
  (S' - S) / S' * 100 = 48.18 :=
by 
  sorry

end percentage_discount_l513_51319


namespace inequality_problem_l513_51378

-- Given a < b < 0, we want to prove a^2 > ab > b^2
theorem inequality_problem (a b : ℝ) (h : a < b ∧ b < 0) : a^2 > ab ∧ ab > b^2 :=
sorry

end inequality_problem_l513_51378


namespace relationship_between_y_values_l513_51340

theorem relationship_between_y_values 
  (m : ℝ) 
  (y1 y2 y3 : ℝ)
  (h1 : y1 = (-1 : ℝ) ^ 2 + 2 * (-1 : ℝ) + m) 
  (h2 : y2 = (3 : ℝ) ^ 2 + 2 * (3 : ℝ) + m) 
  (h3 : y3 = ((1 / 2) : ℝ) ^ 2 + 2 * ((1 / 2) : ℝ) + m) : 
  y2 > y3 ∧ y3 > y1 := 
by 
  sorry

end relationship_between_y_values_l513_51340


namespace vote_difference_l513_51358

-- Definitions of initial votes for and against the policy
def vote_initial_for (x y : ℕ) : Prop := x + y = 450
def initial_margin (x y m : ℕ) : Prop := y > x ∧ y - x = m

-- Definitions of votes for and against in the second vote
def vote_second_for (x' y' : ℕ) : Prop := x' + y' = 450
def second_margin (x' y' m : ℕ) : Prop := x' - y' = 3 * m
def second_vote_ratio (x' y : ℕ) : Prop := x' = 10 * y / 9

-- Theorem to prove the increase in votes
theorem vote_difference (x y x' y' m : ℕ)
  (hi : vote_initial_for x y)
  (hm : initial_margin x y m)
  (hs : vote_second_for x' y')
  (hsm : second_margin x' y' m)
  (hr : second_vote_ratio x' y) : 
  x' - x = 52 :=
sorry

end vote_difference_l513_51358


namespace apple_price_difference_l513_51354

variable (S R F : ℝ)

theorem apple_price_difference (h1 : S + R > R + F) (h2 : F = S - 250) :
  (S + R) - (R + F) = 250 :=
by
  sorry

end apple_price_difference_l513_51354


namespace winston_initial_quarters_l513_51342

-- Defining the conditions
def spent_candy := 50 -- 50 cents spent on candy
def remaining_cents := 300 -- 300 cents left

-- Defining the value of a quarter in cents
def value_of_quarter := 25

-- Calculating the number of quarters Winston initially had
def initial_quarters := (spent_candy + remaining_cents) / value_of_quarter

-- Proof statement
theorem winston_initial_quarters : initial_quarters = 14 := 
by sorry

end winston_initial_quarters_l513_51342


namespace simplified_radical_formula_l513_51387

theorem simplified_radical_formula (y : ℝ) (hy : 0 ≤ y):
  Real.sqrt (48 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) = 120 * y * Real.sqrt (3 * y) :=
by
  sorry

end simplified_radical_formula_l513_51387


namespace number_of_customers_l513_51351

theorem number_of_customers 
  (total_cartons : ℕ) 
  (damaged_cartons : ℕ) 
  (accepted_cartons : ℕ) 
  (customers : ℕ) 
  (h1 : total_cartons = 400)
  (h2 : damaged_cartons = 60)
  (h3 : accepted_cartons = 160)
  (h_eq_per_customer : (total_cartons / customers) - damaged_cartons = accepted_cartons / customers) :
  customers = 4 :=
sorry

end number_of_customers_l513_51351


namespace least_possible_value_z_minus_x_l513_51353

theorem least_possible_value_z_minus_x (x y z : ℤ) (h1 : Even x) (h2 : Odd y) (h3 : Odd z) (h4 : x < y) (h5 : y < z) (h6 : y - x > 5) : z - x = 9 := 
sorry

end least_possible_value_z_minus_x_l513_51353


namespace Mirella_read_purple_books_l513_51379

theorem Mirella_read_purple_books (P : ℕ) 
  (pages_per_purple_book : ℕ := 230)
  (pages_per_orange_book : ℕ := 510)
  (orange_books_read : ℕ := 4)
  (extra_orange_pages : ℕ := 890)
  (total_orange_pages : ℕ := orange_books_read * pages_per_orange_book)
  (total_purple_pages : ℕ := P * pages_per_purple_book)
  (condition : total_orange_pages - total_purple_pages = extra_orange_pages) :
  P = 5 := 
by 
  sorry

end Mirella_read_purple_books_l513_51379


namespace monthly_fixed_cost_is_correct_l513_51332

-- Definitions based on the conditions in the problem
def production_cost_per_component : ℕ := 80
def shipping_cost_per_component : ℕ := 5
def components_per_month : ℕ := 150
def minimum_price_per_component : ℕ := 195

-- Monthly fixed cost definition based on the provided solution
def monthly_fixed_cost := components_per_month * (minimum_price_per_component - (production_cost_per_component + shipping_cost_per_component))

-- Theorem stating that the calculated fixed cost is correct.
theorem monthly_fixed_cost_is_correct : monthly_fixed_cost = 16500 :=
by
  unfold monthly_fixed_cost
  norm_num
  sorry

end monthly_fixed_cost_is_correct_l513_51332


namespace speed_ratio_l513_51392

variable (v1 v2 : ℝ) -- Speeds of A and B respectively
variable (dA dB : ℝ) -- Distances to destinations A and B respectively

-- Conditions:
-- 1. Both reach their destinations in 1 hour
def condition_1 : Prop := dA = v1 ∧ dB = v2

-- 2. When they swap destinations, A takes 35 minutes more to reach B's destination
def condition_2 : Prop := dB / v1 = dA / v2 + 35 / 60

-- Given these conditions, prove that the ratio of v1 to v2 is 3
theorem speed_ratio (h1 : condition_1 v1 v2 dA dB) (h2 : condition_2 v1 v2 dA dB) : v1 = 3 * v2 :=
sorry

end speed_ratio_l513_51392
