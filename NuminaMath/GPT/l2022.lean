import Mathlib

namespace people_going_to_movie_l2022_202214

variable (people_per_car : ℕ) (number_of_cars : ℕ)

theorem people_going_to_movie (h1 : people_per_car = 6) (h2 : number_of_cars = 18) : 
    (people_per_car * number_of_cars) = 108 := 
by
  sorry

end people_going_to_movie_l2022_202214


namespace non_empty_solution_set_range_l2022_202208

theorem non_empty_solution_set_range {a : ℝ} 
  (h : ∃ x : ℝ, |x + 2| + |x - 3| ≤ a) : 
  a ≥ 5 :=
sorry

end non_empty_solution_set_range_l2022_202208


namespace new_students_admitted_l2022_202249

theorem new_students_admitted (orig_students : ℕ := 35) (increase_cost : ℕ := 42) (orig_expense : ℕ := 400) (dim_avg_expense : ℤ := 1) :
  ∃ (x : ℕ), x = 7 :=
by
  sorry

end new_students_admitted_l2022_202249


namespace boat_travel_distance_downstream_l2022_202220

-- Define the given conditions
def speed_boat_still : ℝ := 22
def speed_stream : ℝ := 5
def time_downstream : ℝ := 5

-- Define the effective speed and the computed distance
def effective_speed_downstream : ℝ := speed_boat_still + speed_stream
def distance_traveled_downstream : ℝ := effective_speed_downstream * time_downstream

-- State the proof problem that distance_traveled_downstream is 135 km
theorem boat_travel_distance_downstream :
  distance_traveled_downstream = 135 :=
by
  -- The proof will go here
  sorry

end boat_travel_distance_downstream_l2022_202220


namespace a_3_def_a_4_def_a_r_recurrence_l2022_202273

-- Define minimally the structure of the problem.
noncomputable def a_r (r : ℕ) : ℕ := -- Definition for minimum phone calls required.
by sorry

-- Assertions for the specific cases provided.
theorem a_3_def : a_r 3 = 3 :=
by
  -- Proof is omitted with sorry.
  sorry

theorem a_4_def : a_r 4 = 4 :=
by
  -- Proof is omitted with sorry.
  sorry

theorem a_r_recurrence (r : ℕ) (hr : r ≥ 3) : a_r r ≤ a_r (r - 1) + 2 :=
by
  -- Proof is omitted with sorry.
  sorry

end a_3_def_a_4_def_a_r_recurrence_l2022_202273


namespace eduardo_ate_fraction_of_remaining_l2022_202248

theorem eduardo_ate_fraction_of_remaining (init_cookies : ℕ) (nicole_fraction : ℚ) (remaining_percent : ℚ) :
  init_cookies = 600 →
  nicole_fraction = 2 / 5 →
  remaining_percent = 24 / 100 →
  (360 - (600 * 24 / 100)) / 360 = 3 / 5 := by
  sorry

end eduardo_ate_fraction_of_remaining_l2022_202248


namespace savings_increase_l2022_202275

variable (I : ℝ) -- Initial income
variable (E : ℝ) -- Initial expenditure
variable (S : ℝ) -- Initial savings
variable (I_new : ℝ) -- New income
variable (E_new : ℝ) -- New expenditure
variable (S_new : ℝ) -- New savings

theorem savings_increase (h1 : E = 0.75 * I) 
                         (h2 : I_new = 1.20 * I) 
                         (h3 : E_new = 1.10 * E) : 
                         (S_new - S) / S * 100 = 50 :=
by 
  have h4 : S = 0.25 * I := by sorry
  have h5 : E_new = 0.825 * I := by sorry
  have h6 : S_new = 0.375 * I := by sorry
  have increase : (S_new - S) / S * 100 = 50 := by sorry
  exact increase

end savings_increase_l2022_202275


namespace jason_initial_cards_l2022_202284

/-- Jason initially had some Pokemon cards, Alyssa bought him 224 more, 
and now Jason has 900 Pokemon cards in total.
Prove that initially Jason had 676 Pokemon cards. -/
theorem jason_initial_cards (a b c : ℕ) (h_a : a = 224) (h_b : b = 900) (h_cond : b = a + 676) : 676 = c :=
by 
  sorry

end jason_initial_cards_l2022_202284


namespace updated_mean_of_decrement_l2022_202251

theorem updated_mean_of_decrement 
  (mean_initial : ℝ)
  (num_observations : ℕ)
  (decrement_per_observation : ℝ)
  (h1 : mean_initial = 200)
  (h2 : num_observations = 50)
  (h3 : decrement_per_observation = 6) : 
  (mean_initial * num_observations - decrement_per_observation * num_observations) / num_observations = 194 :=
by
  sorry

end updated_mean_of_decrement_l2022_202251


namespace compute_fraction_power_mul_l2022_202242

theorem compute_fraction_power_mul : ((1 / 3: ℚ) ^ 4) * (1 / 5) = (1 / 405) := by
  -- proof goes here
  sorry

end compute_fraction_power_mul_l2022_202242


namespace height_of_E_l2022_202203

variable {h_E h_F h_G h_H : ℝ}

theorem height_of_E (h1 : h_E + h_F + h_G + h_H = 2 * (h_E + h_F))
                    (h2 : (h_E + h_F) / 2 = (h_E + h_G) / 2 - 4)
                    (h3 : h_H = h_E - 10)
                    (h4 : h_F + h_G = 288) :
  h_E = 139 :=
by
  sorry

end height_of_E_l2022_202203


namespace solve_for_x_l2022_202221

theorem solve_for_x :
  ∃ (x : ℝ), x ≠ 0 ∧ (5 * x)^10 = (10 * x)^5 ∧ x = 2 / 5 :=
by
  sorry

end solve_for_x_l2022_202221


namespace fraction_of_crop_to_CD_is_correct_l2022_202241

-- Define the trapezoid with given conditions
structure Trapezoid :=
  (AB CD AD BC : ℝ)
  (angleA angleD : ℝ)
  (h: ℝ) -- height
  (Area Trapezoid total_area close_area_to_CD: ℝ) 

-- Assumptions
axiom AB_eq_CD (T : Trapezoid) : T.AB = 150 
axiom CD_eq_CD (T : Trapezoid) : T.CD = 200
axiom AD_eq_CD (T : Trapezoid) : T.AD = 130
axiom BC_eq_CD (T : Trapezoid) : T.BC = 130
axiom angleA_eq_75 (T : Trapezoid) : T.angleA = 75
axiom angleD_eq_75 (T : Trapezoid) : T.angleD = 75

-- The fraction calculation
noncomputable def fraction_to_CD (T : Trapezoid) : ℝ :=
  T.close_area_to_CD / T.total_area

-- Theorem stating the fraction of the crop that is brought to the longer base CD is 15/28
theorem fraction_of_crop_to_CD_is_correct (T : Trapezoid) 
  (h_pos : 0 < T.h)
  (total_area_def : T.total_area = (T.AB + T.CD) * T.h / 2)
  (close_area_def : T.close_area_to_CD = ((T.h / 4) * (T.AB + T.CD))) : 
  fraction_to_CD T = 15 / 28 :=
  sorry

end fraction_of_crop_to_CD_is_correct_l2022_202241


namespace find_x_l2022_202236

open Real

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (2, -1)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_x (x : ℝ) : 
  let ab := (a.1 + x * b.1, a.2 + x * b.2)
  let minus_b := (-b.1, -b.2)
  dot_product ab minus_b = 0 
  → x = -2 / 5 :=
by
  intros
  sorry

end find_x_l2022_202236


namespace smallest_k_exists_l2022_202289

theorem smallest_k_exists : ∃ (k : ℕ), k > 0 ∧ (∃ (n m : ℕ), n > 0 ∧ m > 0 ∧ k = 19^n - 5^m) ∧ k = 14 :=
by 
  sorry

end smallest_k_exists_l2022_202289


namespace evaluate_expression_l2022_202279

theorem evaluate_expression :
    123 - (45 * (9 - 6) - 78) + (0 / 1994) = 66 :=
by
  sorry

end evaluate_expression_l2022_202279


namespace fixed_monthly_fee_december_l2022_202252

theorem fixed_monthly_fee_december (x y : ℝ) 
    (h1 : x + y = 15.00) 
    (h2 : x + 2 + 3 * y = 25.40) : 
    x = 10.80 :=
by
  sorry

end fixed_monthly_fee_december_l2022_202252


namespace vector_subtraction_correct_l2022_202238

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-3, 4)

theorem vector_subtraction_correct : (a - b) = (5, -3) :=
by 
  have h1 : a = (2, 1) := by rfl
  have h2 : b = (-3, 4) := by rfl
  sorry

end vector_subtraction_correct_l2022_202238


namespace reservoir_water_level_l2022_202259

theorem reservoir_water_level (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : 
  ∃ y : ℝ, y = 6 + 0.3 * x :=
by sorry

end reservoir_water_level_l2022_202259


namespace min_value_of_expression_ge_9_l2022_202271

theorem min_value_of_expression_ge_9 
    (x : ℝ)
    (h1 : -2 < x ∧ x < -1)
    (m n : ℝ)
    (a b : ℝ)
    (ha : a = -2)
    (hb : b = -1)
    (h2 : mn > 0)
    (h3 : m * a + n * b + 1 = 0) :
    (2 / m) + (1 / n) ≥ 9 := by
  sorry

end min_value_of_expression_ge_9_l2022_202271


namespace impossible_to_create_3_piles_l2022_202257

-- Defining similar piles
def similar (x y : ℝ) : Prop :=
  x / y ≤ Real.sqrt 2 ∧ y / x ≤ Real.sqrt 2

-- Main theorem statement
theorem impossible_to_create_3_piles (initial_pile : ℝ) (h_initial : initial_pile > 0) :
  ∀ (x y z : ℝ), 
  x + y + z = initial_pile → 
  similar x y ∧ similar y z ∧ similar z x → 
  false := 
by 
  sorry

end impossible_to_create_3_piles_l2022_202257


namespace blake_change_l2022_202210

-- Definitions based on conditions
def n_l : ℕ := 4
def n_c : ℕ := 6
def p_l : ℕ := 2
def p_c : ℕ := 4 * p_l
def amount_given : ℕ := 6 * 10

-- Total cost calculations derived from the conditions
def total_cost_lollipops : ℕ := n_l * p_l
def total_cost_chocolates : ℕ := n_c * p_c
def total_cost : ℕ := total_cost_lollipops + total_cost_chocolates

-- Calculating the change
def change : ℕ := amount_given - total_cost

-- Theorem stating the final answer
theorem blake_change : change = 4 := sorry

end blake_change_l2022_202210


namespace inequality_solution_l2022_202290

theorem inequality_solution (x : ℝ) (h_pos : 0 < x) :
  (3 / 8 + |x - 14 / 24| < 8 / 12) ↔ x ∈ Set.Ioo (7 / 24) (7 / 8) :=
by
  sorry

end inequality_solution_l2022_202290


namespace find_alpha_minus_beta_find_cos_2alpha_minus_beta_l2022_202261

-- Definitions and assumptions
variables (α β : ℝ)
axiom sin_alpha : Real.sin α = (Real.sqrt 5) / 5
axiom sin_beta : Real.sin β = (3 * Real.sqrt 10) / 10
axiom alpha_acute : 0 < α ∧ α < Real.pi / 2
axiom beta_acute : 0 < β ∧ β < Real.pi / 2

-- Statement to prove α - β = -π/4
theorem find_alpha_minus_beta : α - β = -Real.pi / 4 :=
sorry

-- Given α - β = -π/4, statement to prove cos(2α - β) = 3√10 / 10
theorem find_cos_2alpha_minus_beta (h : α - β = -Real.pi / 4) : Real.cos (2 * α - β) = (3 * Real.sqrt 10) / 10 :=
sorry

end find_alpha_minus_beta_find_cos_2alpha_minus_beta_l2022_202261


namespace percentage_increase_l2022_202200

-- Define the initial and final prices as constants
def P_inicial : ℝ := 5.00
def P_final : ℝ := 5.55

-- Define the percentage increase proof
theorem percentage_increase : ((P_final - P_inicial) / P_inicial) * 100 = 11 := 
by
  sorry

end percentage_increase_l2022_202200


namespace baseball_football_difference_is_five_l2022_202240

-- Define the conditions
def total_cards : ℕ := 125
def baseball_cards : ℕ := 95
def some_more : ℕ := baseball_cards - 3 * (total_cards - baseball_cards)

-- Define the number of football cards
def football_cards : ℕ := total_cards - baseball_cards

-- Define the difference between the number of baseball cards and three times the number of football cards
def difference : ℕ := baseball_cards - 3 * football_cards

-- Statement of the proof
theorem baseball_football_difference_is_five : difference = 5 := 
by
  sorry

end baseball_football_difference_is_five_l2022_202240


namespace evaluate_box_2_neg1_0_l2022_202211

def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

theorem evaluate_box_2_neg1_0 : box 2 (-1) 0 = -1/2 := 
by
  sorry

end evaluate_box_2_neg1_0_l2022_202211


namespace range_of_a_l2022_202266

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, abs (2 * x - 1) + abs (x + 2) ≥ a^2 + (1 / 2) * a + 2) →
  -1 ≤ a ∧ a ≤ (1 / 2) := by
sorry

end range_of_a_l2022_202266


namespace circumference_of_back_wheel_l2022_202222

theorem circumference_of_back_wheel
  (C_f : ℝ) (C_b : ℝ) (D : ℝ) (N_b : ℝ)
  (h1 : C_f = 30)
  (h2 : D = 1650)
  (h3 : (N_b + 5) * C_f = D)
  (h4 : N_b * C_b = D) :
  C_b = 33 :=
sorry

end circumference_of_back_wheel_l2022_202222


namespace solve_fraction_problem_l2022_202223

noncomputable def x_value (a b c d : ℤ) : ℝ :=
  (a + b * Real.sqrt c) / d

theorem solve_fraction_problem (a b c d : ℤ) (h1 : x_value a b c d = (5 + 5 * Real.sqrt 5) / 4)
  (h2 : (4 * x_value a b c d) / 5 - 2 = 5 / x_value a b c d) :
  (a * c * d) / b = 20 := by
  sorry

end solve_fraction_problem_l2022_202223


namespace wall_length_l2022_202235

theorem wall_length (side_mirror : ℝ) (width_wall : ℝ) (length_wall : ℝ) 
  (h_mirror: side_mirror = 18) 
  (h_width: width_wall = 32)
  (h_area: (side_mirror ^ 2) * 2 = width_wall * length_wall):
  length_wall = 20.25 := 
by 
  -- The following 'sorry' is a placeholder for the proof
  sorry

end wall_length_l2022_202235


namespace three_digit_number_divisible_by_7_l2022_202263

theorem three_digit_number_divisible_by_7 (t : ℕ) :
  (n : ℕ) = 600 + 10 * t + 5 →
  n ≥ 100 ∧ n < 1000 →
  n % 10 = 5 →
  (n / 100) % 10 = 6 →
  n % 7 = 0 →
  n = 665 :=
by
  sorry

end three_digit_number_divisible_by_7_l2022_202263


namespace alice_bob_sum_is_42_l2022_202278

theorem alice_bob_sum_is_42 :
  ∃ (A B : ℕ), 
    (1 ≤ A ∧ A ≤ 60) ∧ 
    (1 ≤ B ∧ B ≤ 60) ∧ 
    Nat.Prime B ∧ B > 10 ∧ 
    (∀ n : ℕ, n < 5 → (A + B) % n ≠ 0) ∧ 
    ∃ k : ℕ, 150 * B + A = k * k ∧ 
    A + B = 42 :=
by 
  sorry

end alice_bob_sum_is_42_l2022_202278


namespace find_number_l2022_202287

theorem find_number (N : ℝ) (h1 : (3 / 10) * N = 64.8) : N = 216 ∧ (1 / 3) * (1 / 4) * N = 18 := 
by 
  sorry

end find_number_l2022_202287


namespace sum_of_three_consecutive_even_numbers_l2022_202224

theorem sum_of_three_consecutive_even_numbers (m : ℤ) (h : ∃ k, m = 2 * k) : 
  m + (m + 2) + (m + 4) = 3 * m + 6 :=
by
  sorry

end sum_of_three_consecutive_even_numbers_l2022_202224


namespace sum_of_squares_of_roots_l2022_202232

-- Define the roots of the polynomial and Vieta's conditions
variables {p q r : ℝ}

-- Given conditions from Vieta's formulas
def vieta_conditions (p q r : ℝ) : Prop :=
  p + q + r = 7 / 3 ∧
  p * q + p * r + q * r = 2 / 3 ∧
  p * q * r = 4 / 3

-- Statement that sum of squares of roots equals to 37/9 given Vieta's conditions
theorem sum_of_squares_of_roots 
  (h : vieta_conditions p q r) : 
  p^2 + q^2 + r^2 = 37 / 9 := 
sorry

end sum_of_squares_of_roots_l2022_202232


namespace max_quarters_is_13_l2022_202207

noncomputable def number_of_quarters (total_value : ℝ) (quarters nickels dimes : ℝ) : Prop :=
  total_value = 4.55 ∧
  quarters = nickels ∧
  dimes = quarters / 2 ∧
  (0.25 * quarters + 0.05 * nickels + 0.05 * quarters / 2 = 4.55)

theorem max_quarters_is_13 : ∃ q : ℝ, number_of_quarters 4.55 q q (q / 2) ∧ q = 13 :=
by
  sorry

end max_quarters_is_13_l2022_202207


namespace PQ_ratio_l2022_202282

-- Definitions
def hexagon_area : ℕ := 7
def base_of_triangle : ℕ := 4

-- Conditions
def PQ_bisects_area (A : ℕ) : Prop :=
  A = hexagon_area / 2

def area_below_PQ (U T : ℚ) : Prop :=
  U + T = hexagon_area / 2 ∧ U = 1

def triangle_area (T b : ℚ) : ℚ :=
  1/2 * b * (5/4)

def XQ_QY_ratio (XQ QY : ℚ) : ℚ :=
  XQ / QY

-- Theorem Statement
theorem PQ_ratio (XQ QY : ℕ) (h1 : PQ_bisects_area (hexagon_area / 2))
  (h2 : area_below_PQ 1 (triangle_area (5/2) base_of_triangle))
  (h3 : XQ + QY = base_of_triangle) : XQ_QY_ratio XQ QY = 1 := sorry

end PQ_ratio_l2022_202282


namespace charles_whistles_l2022_202219

theorem charles_whistles (S C : ℕ) (h1 : S = 45) (h2 : S = C + 32) : C = 13 := 
by
  sorry

end charles_whistles_l2022_202219


namespace south_side_students_count_l2022_202264

variables (N : ℕ)
def students_total := 41
def difference := 3

theorem south_side_students_count (N : ℕ) (h₁ : 2 * N + difference = students_total) : N + difference = 22 :=
sorry

end south_side_students_count_l2022_202264


namespace blue_ball_higher_probability_l2022_202281

noncomputable def probability_blue_ball_higher : ℝ :=
  let p (k : ℕ) : ℝ := 1 / (2^k : ℝ)
  let same_bin_prob := ∑' k : ℕ, (p (k + 1))^2
  let higher_prob := (1 - same_bin_prob) / 2
  higher_prob

theorem blue_ball_higher_probability :
  probability_blue_ball_higher = 1 / 3 :=
by
  sorry

end blue_ball_higher_probability_l2022_202281


namespace find_A_from_AB9_l2022_202209

theorem find_A_from_AB9 (A B : ℕ) (h1 : 0 ≤ A ∧ A ≤ 9) (h2 : 0 ≤ B ∧ B ≤ 9) (h3 : 100 * A + 10 * B + 9 = 459) : A = 4 :=
sorry

end find_A_from_AB9_l2022_202209


namespace students_same_group_in_all_lessons_l2022_202276

theorem students_same_group_in_all_lessons (students : Fin 28 → Fin 3 × Fin 3 × Fin 3) :
  ∃ (i j : Fin 28), i ≠ j ∧ students i = students j :=
by
  sorry

end students_same_group_in_all_lessons_l2022_202276


namespace trigonometric_expression_value_l2022_202244

theorem trigonometric_expression_value (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin α ^ 2 - Real.cos α ^ 2) / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 3 / 5 := 
sorry

end trigonometric_expression_value_l2022_202244


namespace find_pairs_l2022_202255

theorem find_pairs (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n)
  (h3 : (m^2 - n) ∣ (m + n^2)) (h4 : (n^2 - m) ∣ (n + m^2)) :
  (m = 2 ∧ n = 2) ∨ (m = 3 ∧ n = 3) ∨ (m = 1 ∧ n = 2) ∨ (m = 2 ∧ n = 1) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 3) := by
  sorry

end find_pairs_l2022_202255


namespace pair_cannot_appear_l2022_202291

theorem pair_cannot_appear :
  ¬ ∃ (sequence_of_pairs : List (ℤ × ℤ)), 
    (1, 2) ∈ sequence_of_pairs ∧ 
    (2022, 2023) ∈ sequence_of_pairs ∧ 
    ∀ (a b : ℤ) (seq : List (ℤ × ℤ)), 
      (a, b) ∈ seq → 
      ((-a, -b) ∈ seq ∨ (-b, a+b) ∈ seq ∨ 
      ∃ (c d : ℤ), ((a+c, b+d) ∈ seq ∧ (c, d) ∈ seq)) := 
sorry

end pair_cannot_appear_l2022_202291


namespace number_of_male_students_l2022_202237

noncomputable def avg_all : ℝ := 90
noncomputable def avg_male : ℝ := 84
noncomputable def avg_female : ℝ := 92
noncomputable def count_female : ℕ := 24

theorem number_of_male_students (M : ℕ) (T : ℕ) :
  avg_all * (M + count_female) = avg_male * M + avg_female * count_female →
  T = M + count_female →
  M = 8 :=
by
  intro h_avg h_count
  sorry

end number_of_male_students_l2022_202237


namespace percentage_more_l2022_202213

variable (J : ℝ) -- Juan's income
noncomputable def Tim_income := 0.60 * J -- T = 0.60J
noncomputable def Mart_income := 0.84 * J -- M = 0.84J

theorem percentage_more {J : ℝ} (T := Tim_income J) (M := Mart_income J) :
  ((M - T) / T) * 100 = 40 := by
  sorry

end percentage_more_l2022_202213


namespace triangular_pyramid_volume_l2022_202292

theorem triangular_pyramid_volume (a b c : ℝ) 
  (h1 : 1 / 2 * a * b = 6) 
  (h2 : 1 / 2 * a * c = 4) 
  (h3 : 1 / 2 * b * c = 3) : 
  (1 / 3) * (1 / 2) * a * b * c = 4 := by 
  sorry

end triangular_pyramid_volume_l2022_202292


namespace bella_earrings_l2022_202265

theorem bella_earrings (B M R : ℝ) 
  (h1 : B = 0.25 * M) 
  (h2 : M = 2 * R) 
  (h3 : B + M + R = 70) : 
  B = 10 := by 
  sorry

end bella_earrings_l2022_202265


namespace find_original_denominator_l2022_202260

theorem find_original_denominator (d : ℕ) (h : (3 + 7) / (d + 7) = 1 / 3) : d = 23 :=
sorry

end find_original_denominator_l2022_202260


namespace volleyball_problem_correct_l2022_202299

noncomputable def volleyball_problem : Nat :=
  let total_players := 16
  let triplets : Finset String := {"Alicia", "Amanda", "Anna"}
  let twins : Finset String := {"Beth", "Brenda"}
  let remaining_players := total_players - triplets.card - twins.card
  let no_triplets_no_twins := Nat.choose remaining_players 6
  let one_triplet_no_twins := triplets.card * Nat.choose remaining_players 5
  let no_triplets_one_twin := twins.card * Nat.choose remaining_players 5
  no_triplets_no_twins + one_triplet_no_twins + no_triplets_one_twin

theorem volleyball_problem_correct : volleyball_problem = 2772 := by
  sorry

end volleyball_problem_correct_l2022_202299


namespace polina_pizza_combinations_correct_l2022_202283

def polina_pizza_combinations : Nat :=
  let total_toppings := 5
  let possible_combinations := total_toppings * (total_toppings - 1) / 2
  possible_combinations

theorem polina_pizza_combinations_correct :
  polina_pizza_combinations = 10 :=
by
  sorry

end polina_pizza_combinations_correct_l2022_202283


namespace basketball_team_wins_l2022_202293

-- Define the known quantities
def games_won_initial : ℕ := 60
def games_total_initial : ℕ := 80
def games_left : ℕ := 50
def total_games : ℕ := games_total_initial + games_left
def desired_win_fraction : ℚ := 3 / 4

-- The main goal: Prove that the team must win 38 of the remaining 50 games to reach the desired win fraction
theorem basketball_team_wins :
  ∃ x : ℕ, x = 38 ∧ (games_won_initial + x : ℚ) / total_games = desired_win_fraction :=
by
  sorry

end basketball_team_wins_l2022_202293


namespace students_can_be_helped_on_fourth_day_l2022_202294

theorem students_can_be_helped_on_fourth_day : 
  ∀ (total_books first_day_students second_day_students third_day_students books_per_student : ℕ),
  total_books = 120 →
  first_day_students = 4 →
  second_day_students = 5 →
  third_day_students = 6 →
  books_per_student = 5 →
  (total_books - (first_day_students * books_per_student + second_day_students * books_per_student + third_day_students * books_per_student)) / books_per_student = 9 :=
by
  intros total_books first_day_students second_day_students third_day_students books_per_student h_total h_first h_second h_third h_books_per_student
  sorry

end students_can_be_helped_on_fourth_day_l2022_202294


namespace student_count_l2022_202228

noncomputable def numberOfStudents (decreaseInAverageWeight totalWeightDecrease : ℕ) : ℕ :=
  totalWeightDecrease / decreaseInAverageWeight

theorem student_count 
  (decreaseInAverageWeight : ℕ)
  (totalWeightDecrease : ℕ)
  (condition_avg_weight_decrease : decreaseInAverageWeight = 4)
  (condition_weight_difference : totalWeightDecrease = 92 - 72) :
  numberOfStudents decreaseInAverageWeight totalWeightDecrease = 5 := by 
  -- We are not providing the proof details as per the instruction
  sorry

end student_count_l2022_202228


namespace asian_countries_visited_l2022_202256

theorem asian_countries_visited (total_countries europe_countries south_america_countries remaining_asian_countries : ℕ)
  (h1 : total_countries = 42)
  (h2 : europe_countries = 20)
  (h3 : south_america_countries = 10)
  (h4 : remaining_asian_countries = (total_countries - (europe_countries + south_america_countries)) / 2) :
  remaining_asian_countries = 6 :=
by sorry

end asian_countries_visited_l2022_202256


namespace value_of_f_at_sqrt2_l2022_202254

noncomputable def f (x : ℝ) : ℝ := x^5 - 5 * x^4 + 10 * x^3 - 10 * x^2 + 5 * x - 1

theorem value_of_f_at_sqrt2 :
  f (1 + Real.sqrt 2) = 4 * Real.sqrt 2 := by
  sorry

end value_of_f_at_sqrt2_l2022_202254


namespace find_focus_with_larger_x_l2022_202258

def hyperbola_foci_coordinates : Prop :=
  let center := (5, 10)
  let a := 7
  let b := 3
  let c := Real.sqrt (a^2 + b^2)
  let focus1 := (5 + c, 10)
  let focus2 := (5 - c, 10)
  focus1 = (5 + Real.sqrt 58, 10)
  
theorem find_focus_with_larger_x : hyperbola_foci_coordinates := 
  by
    sorry

end find_focus_with_larger_x_l2022_202258


namespace isosceles_triangle_inequality_degenerate_triangle_a_zero_degenerate_triangle_double_b_l2022_202202

section isosceles_triangle

variables (a b k : ℝ)

/-- Prove the inequality for an isosceles triangle -/
theorem isosceles_triangle_inequality (h_perimeter : k = a + 2 * b) (ha_pos : a > 0) :
  k / 2 < a + b ∧ a + b < 3 * k / 4 :=
sorry

/-- Prove the inequality for degenerate triangle with a = 0 -/
theorem degenerate_triangle_a_zero (b k : ℝ) (h_perimeter : k = 2 * b) :
  k / 2 ≤ b ∧ b < 3 * k / 4 :=
sorry

/-- Prove the inequality for degenerate triangle with a = 2b -/
theorem degenerate_triangle_double_b (b k : ℝ) (h_perimeter : k = 4 * b) :
  k / 2 < b ∧ b ≤ 3 * k / 4 :=
sorry

end isosceles_triangle

end isosceles_triangle_inequality_degenerate_triangle_a_zero_degenerate_triangle_double_b_l2022_202202


namespace find_unknown_number_l2022_202218

def op (a b : ℝ) := a * (b ^ (1 / 2))

theorem find_unknown_number (x : ℝ) (h : op 4 x = 12) : x = 9 :=
by
  sorry

end find_unknown_number_l2022_202218


namespace avg_annual_growth_rate_equation_l2022_202246

variable (x : ℝ)
def foreign_trade_income_2007 : ℝ := 250 -- million yuan
def foreign_trade_income_2009 : ℝ := 360 -- million yuan

theorem avg_annual_growth_rate_equation :
  2.5 * (1 + x) ^ 2 = 3.6 := sorry

end avg_annual_growth_rate_equation_l2022_202246


namespace find_x2_times_x1_plus_x3_l2022_202267

noncomputable def a := Real.sqrt 2023
noncomputable def x1 := -Real.sqrt 7
noncomputable def x2 := 1 / a
noncomputable def x3 := Real.sqrt 7

theorem find_x2_times_x1_plus_x3 :
  let x1 := -Real.sqrt 7
  let x2 := 1 / Real.sqrt 2023
  let x3 := Real.sqrt 7
  x2 * (x1 + x3) = 0 :=
by
  sorry

end find_x2_times_x1_plus_x3_l2022_202267


namespace complement_A_in_U_l2022_202230

universe u

-- Define the universal set U and set A.
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5, 7}

-- Define the complement of A in U.
def complement (A U: Set ℕ) : Set ℕ :=
  {x ∈ U | x ∉ A}

-- Statement to prove.
theorem complement_A_in_U :
  complement A U = {2, 4, 6} :=
sorry

end complement_A_in_U_l2022_202230


namespace maximum_number_of_buses_l2022_202243

-- Definitions of the conditions
def bus_stops : ℕ := 9 -- There are 9 bus stops in the city.
def max_common_stops : ℕ := 1 -- Any two buses have at most one common stop.
def stops_per_bus : ℕ := 3 -- Each bus stops at exactly three stops.

-- The main theorem statement
theorem maximum_number_of_buses (n : ℕ) :
  (bus_stops = 9) →
  (max_common_stops = 1) →
  (stops_per_bus = 3) →
  n ≤ 12 :=
sorry

end maximum_number_of_buses_l2022_202243


namespace inequality_for_distinct_integers_l2022_202206

-- Define the necessary variables and conditions
variable {a b c : ℤ}

-- Ensure a, b, and c are pairwise distinct integers
def pairwise_distinct (a b c : ℤ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a

-- The main theorem statement
theorem inequality_for_distinct_integers 
  (h : pairwise_distinct a b c) : 
  (a^3 + b^3 + c^3) / 3 ≥ a * b * c + Real.sqrt (3 * (a * b + b * c + c * a + 1)) :=
by
  sorry

end inequality_for_distinct_integers_l2022_202206


namespace total_guitars_sold_l2022_202286

theorem total_guitars_sold (total_revenue : ℕ) (price_electric : ℕ) (price_acoustic : ℕ)
  (num_electric_sold : ℕ) (num_acoustic_sold : ℕ) 
  (h1 : total_revenue = 3611) (h2 : price_electric = 479) 
  (h3 : price_acoustic = 339) (h4 : num_electric_sold = 4) 
  (h5 : num_acoustic_sold * price_acoustic + num_electric_sold * price_electric = total_revenue) :
  num_electric_sold + num_acoustic_sold = 9 :=
sorry

end total_guitars_sold_l2022_202286


namespace max_value_f_at_e_l2022_202201

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_f_at_e (h : 0 < x) : 
  ∃ e : ℝ, (∀ x : ℝ, 0 < x → f x ≤ f e) ∧ e = Real.exp 1 :=
by
  sorry

end max_value_f_at_e_l2022_202201


namespace problem_1_l2022_202216

theorem problem_1 (x : ℝ) (h : x^2 = 2) : (3 * x)^2 - 4 * (x^3)^2 = -14 :=
by {
  sorry
}

end problem_1_l2022_202216


namespace original_triangle_area_l2022_202205

-- Define the conditions
def dimensions_quadrupled (original_area new_area : ℝ) : Prop :=
  4^2 * original_area = new_area

-- Define the statement to be proved
theorem original_triangle_area {new_area : ℝ} (h : new_area = 64) :
  ∃ (original_area : ℝ), dimensions_quadrupled original_area new_area ∧ original_area = 4 :=
by
  sorry

end original_triangle_area_l2022_202205


namespace find_slope_of_q_l2022_202280

theorem find_slope_of_q (j : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + 3 → y = j * x + 1 → x = 1 → y = 5) → j = 4 := 
by
  intro h
  sorry

end find_slope_of_q_l2022_202280


namespace quadratic_term_free_solution_l2022_202247

theorem quadratic_term_free_solution (m : ℝ) : 
  (∀ x : ℝ, ∃ (p : ℝ → ℝ), (x + m) * (x^2 + 2*x - 1) = p x + (2 + m) * x^2) → m = -2 :=
by
  intro H
  sorry

end quadratic_term_free_solution_l2022_202247


namespace intersection_of_sets_l2022_202268

theorem intersection_of_sets (A B : Set ℕ) (hA : A = {1, 2, 3}) (hB : B = {3, 4}) :
  A ∩ B = {3} :=
by
  rw [hA, hB]
  exact sorry

end intersection_of_sets_l2022_202268


namespace XYZStockPriceIs75_l2022_202233

/-- XYZ stock price model 
Starts at $50, increases by 200% in first year, 
then decreases by 50% in second year.
-/
def XYZStockPriceEndOfSecondYear : ℝ :=
  let initialPrice := 50
  let firstYearIncreaseRate := 2.0
  let secondYearDecreaseRate := 0.5
  let priceAfterFirstYear := initialPrice * (1 + firstYearIncreaseRate)
  let priceAfterSecondYear := priceAfterFirstYear * (1 - secondYearDecreaseRate)
  priceAfterSecondYear

theorem XYZStockPriceIs75 : XYZStockPriceEndOfSecondYear = 75 := by
  sorry

end XYZStockPriceIs75_l2022_202233


namespace beads_currently_have_l2022_202298

-- Definitions of the conditions
def friends : Nat := 6
def beads_per_bracelet : Nat := 8
def additional_beads_needed : Nat := 12

-- Theorem statement
theorem beads_currently_have : (beads_per_bracelet * friends - additional_beads_needed) = 36 := by
  sorry

end beads_currently_have_l2022_202298


namespace determinant_of_A_l2022_202217

def A : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![3, 0, -2],
  ![8, 5, -4],
  ![3, 3, 6]
]

theorem determinant_of_A : A.det = 108 := by
  sorry

end determinant_of_A_l2022_202217


namespace orchestra_admission_l2022_202270

theorem orchestra_admission (x v c t: ℝ) 
  -- Conditions
  (h1 : v = 1.25 * 1.6 * x)
  (h2 : c = 0.8 * x)
  (h3 : t = 0.4 * x)
  (h4 : v + c + t = 32) :
  -- Conclusion
  v = 20 ∧ c = 8 ∧ t = 4 :=
sorry

end orchestra_admission_l2022_202270


namespace solve_frac_eq_l2022_202204

-- Define the fractional function
def frac_eq (x : ℝ) : Prop := (x + 2) / (x - 1) = 0

-- State the theorem
theorem solve_frac_eq : frac_eq (-2) :=
by
  unfold frac_eq
  -- Use sorry to skip the proof
  sorry

end solve_frac_eq_l2022_202204


namespace sqrt_meaningful_range_l2022_202234

-- Define the condition
def sqrt_condition (x : ℝ) : Prop := 1 - 3 * x ≥ 0

-- State the theorem
theorem sqrt_meaningful_range (x : ℝ) (h : sqrt_condition x) : x ≤ 1 / 3 :=
sorry

end sqrt_meaningful_range_l2022_202234


namespace hot_dogs_remainder_l2022_202231

theorem hot_dogs_remainder : 25197641 % 6 = 1 :=
by
  sorry

end hot_dogs_remainder_l2022_202231


namespace minimize_a2_b2_l2022_202225

theorem minimize_a2_b2 (a b t : ℝ) (h : 2 * a + b = 2 * t) : ∃ a b, (2 * a + b = 2 * t) ∧ (a^2 + b^2 = 4 * t^2 / 5) :=
by
  sorry

end minimize_a2_b2_l2022_202225


namespace simplify_sqrt_l2022_202262

theorem simplify_sqrt (a b : ℝ) (hb : b > 0) : 
  Real.sqrt (20 * a^3 * b^2) = 2 * a * b * Real.sqrt (5 * a) :=
by
  sorry

end simplify_sqrt_l2022_202262


namespace value_added_after_doubling_l2022_202253

theorem value_added_after_doubling (x v : ℝ) (h1 : x = 4) (h2 : 2 * x + v = x / 2 + 20) : v = 14 :=
by
  sorry

end value_added_after_doubling_l2022_202253


namespace relationship_between_m_and_n_l2022_202215

theorem relationship_between_m_and_n (f : ℝ → ℝ) (a : ℝ) 
  (h_even : ∀ x : ℝ, f x = f (-x)) 
  (h_mono_inc : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y) 
  (m_def : f (-1) = f 1) 
  (n_def : f (a^2 + 2*a + 3) > f 1) :
  f (-1) < f (a^2 + 2*a + 3) := 
by 
  sorry

end relationship_between_m_and_n_l2022_202215


namespace sequence_general_term_l2022_202212

theorem sequence_general_term (a : ℕ+ → ℤ) (h₁ : a 1 = 2) (h₂ : ∀ n : ℕ+, a (n + 1) = a n - 1) :
  ∀ n : ℕ+, a n = 3 - n := 
sorry

end sequence_general_term_l2022_202212


namespace new_energy_vehicle_sales_growth_l2022_202229

theorem new_energy_vehicle_sales_growth (x : ℝ) :
  let sales_jan := 64
  let sales_feb := 64 * (1 + x)
  let sales_mar := 64 * (1 + x)^2
  (sales_jan + sales_feb + sales_mar = 244) :=
sorry

end new_energy_vehicle_sales_growth_l2022_202229


namespace count_two_digit_numbers_l2022_202239

theorem count_two_digit_numbers : (99 - 10 + 1) = 90 := by
  sorry

end count_two_digit_numbers_l2022_202239


namespace total_amount_shared_l2022_202296

theorem total_amount_shared (z : ℝ) (hz : z = 150) (hy : y = 1.20 * z) (hx : x = 1.25 * y) : 
  x + y + z = 555 :=
by
  sorry

end total_amount_shared_l2022_202296


namespace jasmine_spent_l2022_202285

theorem jasmine_spent 
  (original_cost : ℝ)
  (discount : ℝ)
  (h_original : original_cost = 35)
  (h_discount : discount = 17) : 
  original_cost - discount = 18 := 
by
  sorry

end jasmine_spent_l2022_202285


namespace perimeter_less_than_1_km_perimeter_less_than_1_km_zero_thickness_perimeter_to_area_ratio_l2022_202226

section Problem

-- Definitions based on the problem conditions

-- Condition: Side length of each square is 1 cm
def side_length : ℝ := 1

-- Condition: Thickness of the nail for parts a) and b)
def nail_thickness_a := 0.1
def nail_thickness_b := 0

-- Given a perimeter P and area S, the perimeter cannot exceed certain thresholds based on problem analysis

theorem perimeter_less_than_1_km (P : ℝ) (S : ℝ) (r : ℝ) (h1 : 2 * S ≥ r * P) (h2 : r = 0.1) : P < 1000 * 100 :=
  sorry

theorem perimeter_less_than_1_km_zero_thickness (P : ℝ) (S : ℝ) (r : ℝ) (h1 : 2 * S ≥ r * P) (h2 : r = 0) : P < 1000 * 100 :=
  sorry

theorem perimeter_to_area_ratio (P : ℝ) (S : ℝ) (h : P / S ≤ 700) : P / S < 100000 :=
  sorry

end Problem

end perimeter_less_than_1_km_perimeter_less_than_1_km_zero_thickness_perimeter_to_area_ratio_l2022_202226


namespace determine_m_range_l2022_202227

variable {R : Type} [OrderedCommGroup R]

-- Define the odd function f: ℝ → ℝ
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Define the increasing function f: ℝ → ℝ
def increasing_function (f : ℝ → ℝ) := ∀ x y, x < y → f x < f y

-- Define the main theorem
theorem determine_m_range (f : ℝ → ℝ) (odd_f : odd_function f) (inc_f : increasing_function f) :
    (∀ θ : ℝ, f (Real.cos (2 * θ) - 5) + f (2 * m + 4 * Real.sin θ) > 0) → m > 5 :=
by
  sorry

end determine_m_range_l2022_202227


namespace trig_identity_proof_l2022_202272

variable (α : ℝ)

theorem trig_identity_proof : 
  16 * (Real.sin α)^5 - 20 * (Real.sin α)^3 + 5 * Real.sin α = Real.sin (5 * α) :=
  sorry

end trig_identity_proof_l2022_202272


namespace quadrilateral_side_length_eq_12_l2022_202274

-- Definitions
def EF : ℝ := 7
def FG : ℝ := 15
def GH : ℝ := 7
def HE : ℝ := 12
def EH : ℝ := 12

-- Statement to prove that EH = 12 given the definition and conditions
theorem quadrilateral_side_length_eq_12
  (EF_eq : EF = 7)
  (FG_eq : FG = 15)
  (GH_eq : GH = 7)
  (HE_eq : HE = 12)
  (EH_eq : EH = 12) : 
  EH = 12 :=
sorry

end quadrilateral_side_length_eq_12_l2022_202274


namespace find_m_plus_n_l2022_202269

theorem find_m_plus_n (x : ℝ) (m n : ℕ) (h₁ : (1 + Real.sin x) / (Real.cos x) = 22 / 7) 
                      (h₂ : (1 + Real.cos x) / (Real.sin x) = m / n) :
                      m + n = 44 := by
  sorry

end find_m_plus_n_l2022_202269


namespace fraction_of_tank_used_l2022_202288

theorem fraction_of_tank_used (speed : ℝ) (fuel_efficiency : ℝ) (initial_fuel : ℝ) (time_traveled : ℝ)
  (h_speed : speed = 40) (h_fuel_eff : fuel_efficiency = 1 / 40) (h_initial_fuel : initial_fuel = 12) 
  (h_time : time_traveled = 5) : 
  (speed * time_traveled * fuel_efficiency) / initial_fuel = 5 / 12 :=
by
  -- Here the proof would go, but we add sorry to indicate it's incomplete.
  sorry

end fraction_of_tank_used_l2022_202288


namespace evaluate_expression_at_3_l2022_202277

theorem evaluate_expression_at_3 :
  ((3^(3^2))^(3^3)) = 3^(243) := 
by 
  sorry

end evaluate_expression_at_3_l2022_202277


namespace arithmetic_sequence_term_l2022_202250

theorem arithmetic_sequence_term {a : ℕ → ℤ} 
  (h1 : a 4 = -4) 
  (h2 : a 8 = 4) : 
  a 12 = 12 := 
by 
  sorry

end arithmetic_sequence_term_l2022_202250


namespace parabola_has_one_x_intercept_l2022_202245

-- Define the equation of the parabola.
def parabola (y : ℝ) : ℝ := -3 * y ^ 2 + 2 * y + 4

-- Prove that the number of x-intercepts of the graph of the parabola is 1.
theorem parabola_has_one_x_intercept : (∃! y : ℝ, parabola y = 4) :=
by
  sorry

end parabola_has_one_x_intercept_l2022_202245


namespace cubic_polynomial_evaluation_l2022_202295

theorem cubic_polynomial_evaluation
  (f : ℚ → ℚ)
  (cubic_f : ∃ a b c d : ℚ, ∀ x, f x = a*x^3 + b*x^2 + c*x + d)
  (h1 : f (-2) = -4)
  (h2 : f 3 = -9)
  (h3 : f (-4) = -16) :
  f 1 = -23 :=
sorry

end cubic_polynomial_evaluation_l2022_202295


namespace ratio_KL_eq_3_over_5_l2022_202297

theorem ratio_KL_eq_3_over_5
  (K L : ℤ)
  (h : ∀ (x : ℝ), x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 → 
    (K : ℝ) / (x + 3) + (L : ℝ) / (x^2 - 3 * x) = (x^2 - x + 5) / (x^3 + x^2 - 9 * x)):
  (K : ℝ) / (L : ℝ) = 3 / 5 :=
by
  sorry

end ratio_KL_eq_3_over_5_l2022_202297
