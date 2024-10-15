import Mathlib

namespace NUMINAMATH_GPT_mul_powers_same_base_l2282_228209

theorem mul_powers_same_base (x : ℝ) : (x ^ 8) * (x ^ 2) = x ^ 10 :=
by
  exact sorry

end NUMINAMATH_GPT_mul_powers_same_base_l2282_228209


namespace NUMINAMATH_GPT_monomial_sum_mn_l2282_228251

-- Define the conditions as Lean definitions
def is_monomial_sum (x y : ℕ) (m n : ℕ) : Prop :=
  ∃ k : ℕ, (x ^ 2) * (y ^ m) + (x ^ n) * (y ^ 3) = x ^ k

-- State our main theorem
theorem monomial_sum_mn (x y : ℕ) (m n : ℕ) (h : is_monomial_sum x y m n) : m + n = 5 :=
sorry  -- Completion of the proof is not required

end NUMINAMATH_GPT_monomial_sum_mn_l2282_228251


namespace NUMINAMATH_GPT_geese_ratio_l2282_228241

/-- Define the problem conditions --/

def lily_ducks := 20
def lily_geese := 10

def rayden_ducks : ℕ := 3 * lily_ducks
def total_lily_animals := lily_ducks + lily_geese
def total_rayden_animals := total_lily_animals + 70
def rayden_geese := total_rayden_animals - rayden_ducks

/-- Prove the desired ratio of the number of geese Rayden bought to the number of geese Lily bought --/
theorem geese_ratio : rayden_geese / lily_geese = 4 :=
sorry

end NUMINAMATH_GPT_geese_ratio_l2282_228241


namespace NUMINAMATH_GPT_find_k_l2282_228236

-- Define the sum of even integers from 2 to 2k
def sum_even_integers (k : ℕ) : ℕ :=
  2 * (k * (k + 1)) / 2

-- Define the condition that this sum equals 132
def sum_condition (t : ℕ) (k : ℕ) : Prop :=
  sum_even_integers k = t

theorem find_k (k : ℕ) (t : ℕ) (h₁ : t = 132) (h₂ : sum_condition t k) : k = 11 := by
  sorry

end NUMINAMATH_GPT_find_k_l2282_228236


namespace NUMINAMATH_GPT_smallest_pos_integer_l2282_228282

-- Definitions based on the given conditions
def arithmetic_seq (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d
def sum_seq (a1 d : ℤ) (n : ℕ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

-- Given conditions
def condition1 (a1 d : ℤ) : Prop := arithmetic_seq a1 d 11 - arithmetic_seq a1 d 8 = 3
def condition2 (a1 d : ℤ) : Prop := sum_seq a1 d 11 - sum_seq a1 d 8 = 3

-- The claim we want to prove
theorem smallest_pos_integer 
  (n : ℕ) (a1 d : ℤ) 
  (h1 : condition1 a1 d) 
  (h2 : condition2 a1 d) : n = 10 :=
by
  sorry

end NUMINAMATH_GPT_smallest_pos_integer_l2282_228282


namespace NUMINAMATH_GPT_percent_reduction_l2282_228270

def original_price : ℕ := 500
def reduction_amount : ℕ := 400

theorem percent_reduction : (reduction_amount * 100) / original_price = 80 := by
  sorry

end NUMINAMATH_GPT_percent_reduction_l2282_228270


namespace NUMINAMATH_GPT_describe_S_is_two_rays_l2282_228200

def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ common : ℝ, 
     (common = 5 ∧ (p.1 + 3 = common ∧ p.2 - 2 ≥ common ∨ p.1 + 3 ≥ common ∧ p.2 - 2 = common)) ∨
     (common = p.1 + 3 ∧ (5 = common ∧ p.2 - 2 ≥ common ∨ 5 ≥ common ∧ p.2 - 2 = common)) ∨
     (common = p.2 - 2 ∧ (5 = common ∧ p.1 + 3 ≥ common ∨ 5 ≥ common ∧ p.1 + 3 = common))}

theorem describe_S_is_two_rays :
  S = {p : ℝ × ℝ | (p.1 = 2 ∧ p.2 ≥ 7) ∨ (p.2 = 7 ∧ p.1 ≥ 2)} :=
  by
    sorry

end NUMINAMATH_GPT_describe_S_is_two_rays_l2282_228200


namespace NUMINAMATH_GPT_find_f_2_l2282_228247

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_2 (h1 : ∀ x1 x2 : ℝ, f (x1 * x2) = f x1 + f x2) (h2 : f 8 = 3) : f 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_f_2_l2282_228247


namespace NUMINAMATH_GPT_intersection_of_complements_l2282_228214

theorem intersection_of_complements {U S T : Set ℕ}
  (hU : U = {1, 2, 3, 4, 5, 6, 7, 8})
  (hS : S = {1, 3, 5})
  (hT : T = {3, 6}) :
  (U \ S) ∩ (U \ T) = {2, 4, 7, 8} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_complements_l2282_228214


namespace NUMINAMATH_GPT_square_ratio_l2282_228234

theorem square_ratio (x y : ℝ) (hx : x = 60 / 17) (hy : y = 780 / 169) : 
  x / y = 169 / 220 :=
by
  sorry

end NUMINAMATH_GPT_square_ratio_l2282_228234


namespace NUMINAMATH_GPT_handshake_problem_l2282_228223

theorem handshake_problem :
  let companies := 5
  let representatives_per_company := 5
  let total_people := companies * representatives_per_company
  let possible_handshakes_per_person := total_people - representatives_per_company - 1
  let total_possible_handshakes := total_people * possible_handshakes_per_person
  let unique_handshakes := total_possible_handshakes / 2
  unique_handshakes = 250 :=
by 
  let companies := 5
  let representatives_per_company := 5
  let total_people := companies * representatives_per_company
  let possible_handshakes_per_person := total_people - representatives_per_company - 1
  let total_possible_handshakes := total_people * possible_handshakes_per_person
  let unique_handshakes := total_possible_handshakes / 2
  sorry

end NUMINAMATH_GPT_handshake_problem_l2282_228223


namespace NUMINAMATH_GPT_center_cell_value_l2282_228258

open Matrix Finset

def table := Matrix (Fin 3) (Fin 3) ℝ

def row_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 0 2 = 1) ∧ 
  (T 1 0 * T 1 1 * T 1 2 = 1) ∧ 
  (T 2 0 * T 2 1 * T 2 2 = 1)

def col_products (T : table) : Prop :=
  (T 0 0 * T 1 0 * T 2 0 = 1) ∧ 
  (T 0 1 * T 1 1 * T 2 1 = 1) ∧ 
  (T 0 2 * T 1 2 * T 2 2 = 1)

def square_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 1 0 * T 1 1 = 2) ∧ 
  (T 0 1 * T 0 2 * T 1 1 * T 1 2 = 2) ∧ 
  (T 1 0 * T 1 1 * T 2 0 * T 2 1 = 2) ∧ 
  (T 1 1 * T 1 2 * T 2 1 * T 2 2 = 2)

theorem center_cell_value (T : table) 
  (h_row : row_products T) 
  (h_col : col_products T) 
  (h_square : square_products T) : 
  T 1 1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_center_cell_value_l2282_228258


namespace NUMINAMATH_GPT_verify_distinct_outcomes_l2282_228276

def i : ℂ := Complex.I

theorem verify_distinct_outcomes :
  ∃! S, ∀ n : ℤ, n % 8 = n → S = i^n + i^(-n)
  := sorry

end NUMINAMATH_GPT_verify_distinct_outcomes_l2282_228276


namespace NUMINAMATH_GPT_total_kids_in_lawrence_county_l2282_228263

theorem total_kids_in_lawrence_county :
  ∀ (h c T : ℕ), h = 274865 → c = 38608 → T = h + c → T = 313473 :=
by
  intros h c T h_eq c_eq T_eq
  rw [h_eq, c_eq] at T_eq
  exact T_eq

end NUMINAMATH_GPT_total_kids_in_lawrence_county_l2282_228263


namespace NUMINAMATH_GPT_remainder_is_neg_x_plus_60_l2282_228287

theorem remainder_is_neg_x_plus_60 (R : Polynomial ℝ) :
  (R.eval 10 = 50) ∧ (R.eval 50 = 10) → 
  ∃ Q : Polynomial ℝ, R = (Polynomial.X - 10) * (Polynomial.X - 50) * Q + (- Polynomial.X + 60) :=
by
  sorry

end NUMINAMATH_GPT_remainder_is_neg_x_plus_60_l2282_228287


namespace NUMINAMATH_GPT_derivative_at_1_l2282_228229

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem derivative_at_1 : deriv f 1 = 0 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_derivative_at_1_l2282_228229


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l2282_228252

theorem repeating_decimal_to_fraction : (0.7 + 23 / 99 / 10) = (62519 / 66000) := by
  sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l2282_228252


namespace NUMINAMATH_GPT_find_c_l2282_228211

open Real

-- Definition of the quadratic expression in question
def expr (x y c : ℝ) : ℝ := 5 * x^2 - 8 * c * x * y + (4 * c^2 + 3) * y^2 - 5 * x - 5 * y + 7

-- The theorem to prove that the minimum value of this expression being 0 over all (x, y) implies c = 4
theorem find_c :
  (∀ x y : ℝ, expr x y c ≥ 0) → (∃ x y : ℝ, expr x y c = 0) → c = 4 := 
by 
  sorry

end NUMINAMATH_GPT_find_c_l2282_228211


namespace NUMINAMATH_GPT_temperature_difference_l2282_228238

variable (highest_temp : ℤ)
variable (lowest_temp : ℤ)

theorem temperature_difference : 
  highest_temp = 2 ∧ lowest_temp = -8 → (highest_temp - lowest_temp = 10) := by
  sorry

end NUMINAMATH_GPT_temperature_difference_l2282_228238


namespace NUMINAMATH_GPT_smallest_x_l2282_228279

theorem smallest_x (x : ℝ) (h : 4 * x^2 + 6 * x + 1 = 5) : x = -2 :=
sorry

end NUMINAMATH_GPT_smallest_x_l2282_228279


namespace NUMINAMATH_GPT_amount_paid_l2282_228290

theorem amount_paid (cost_price : ℝ) (percent_more : ℝ) (h1 : cost_price = 6525) (h2 : percent_more = 0.24) : 
  cost_price + percent_more * cost_price = 8091 :=
by 
  -- Proof here
  sorry

end NUMINAMATH_GPT_amount_paid_l2282_228290


namespace NUMINAMATH_GPT_rowing_time_to_place_and_back_l2282_228215

def speed_man_still_water : ℝ := 8 -- km/h
def speed_river : ℝ := 2 -- km/h
def total_distance : ℝ := 7.5 -- km

theorem rowing_time_to_place_and_back :
  let V_m := speed_man_still_water
  let V_r := speed_river
  let D := total_distance / 2
  let V_up := V_m - V_r
  let V_down := V_m + V_r
  let T_up := D / V_up
  let T_down := D / V_down
  T_up + T_down = 1 :=
by
  sorry

end NUMINAMATH_GPT_rowing_time_to_place_and_back_l2282_228215


namespace NUMINAMATH_GPT_find_abc_l2282_228267

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

axiom condition1 : a * b = 45 * (3 : ℝ)^(1/3)
axiom condition2 : a * c = 75 * (3 : ℝ)^(1/3)
axiom condition3 : b * c = 30 * (3 : ℝ)^(1/3)

theorem find_abc : a * b * c = 75 * (2 : ℝ)^(1/2) := sorry

end NUMINAMATH_GPT_find_abc_l2282_228267


namespace NUMINAMATH_GPT_inequality_proof_l2282_228255

-- Let x and y be real numbers such that x > y
variables {x y : ℝ} (hx : x > y)

-- We need to prove -2x < -2y
theorem inequality_proof (hx : x > y) : -2 * x < -2 * y :=
sorry

end NUMINAMATH_GPT_inequality_proof_l2282_228255


namespace NUMINAMATH_GPT_range_of_a_l2282_228285

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (1/2)^x = 3 * a + 2 ∧ x < 0) ↔ (a > -1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2282_228285


namespace NUMINAMATH_GPT_perfect_square_tens_place_l2282_228294

/-- A whole number ending in 5 can only be a perfect square if the tens place is 2. -/
theorem perfect_square_tens_place (n : ℕ) (h₁ : n % 10 = 5) : ∃ k : ℕ, n = k * k → (n / 10) % 10 = 2 :=
sorry

end NUMINAMATH_GPT_perfect_square_tens_place_l2282_228294


namespace NUMINAMATH_GPT_num_clerks_l2282_228296

def manager_daily_salary := 5
def clerk_daily_salary := 2
def num_managers := 2
def total_daily_salary := 16

theorem num_clerks (c : ℕ) (h1 : num_managers * manager_daily_salary + c * clerk_daily_salary = total_daily_salary) : c = 3 :=
by 
  sorry

end NUMINAMATH_GPT_num_clerks_l2282_228296


namespace NUMINAMATH_GPT_find_a_l2282_228297

theorem find_a (a : ℝ) : (∀ x : ℝ, (x + 1) * (x - 3) = x^2 + a * x - 3) → a = -2 :=
  by
    sorry

end NUMINAMATH_GPT_find_a_l2282_228297


namespace NUMINAMATH_GPT_cone_height_l2282_228264

theorem cone_height 
  (sector_radius : ℝ) 
  (central_angle : ℝ) 
  (sector_radius_eq : sector_radius = 3) 
  (central_angle_eq : central_angle = 2 * π / 3) : 
  ∃ h : ℝ, h = 2 * Real.sqrt 2 :=
by
  -- Formalize conditions
  let r := 1
  let l := sector_radius
  let θ := central_angle

  -- Combine conditions
  have r_eq : r = 1 := by sorry

  -- Calculate height using Pythagorean theorem
  let h := (l^2 - r^2).sqrt

  use h
  have h_eq : h = 2 * Real.sqrt 2 := by sorry
  exact h_eq

end NUMINAMATH_GPT_cone_height_l2282_228264


namespace NUMINAMATH_GPT_area_of_inscribed_rectangle_l2282_228295

theorem area_of_inscribed_rectangle (h_triangle_altitude : 12 > 0)
  (h_segment_XZ : 15 > 0)
  (h_PQ_eq_one_third_PS : ∀ PQ PS : ℚ, PS = 3 * PQ) :
  ∃ PQ PS : ℚ, 
    (YM = 12) ∧
    (XZ = 15) ∧
    (PQ = (15 / 8 : ℚ)) ∧
    (PS = 3 * PQ) ∧ 
    ((PQ * PS) = (675 / 64 : ℚ)) :=
by
  -- Proof would go here.
  sorry

end NUMINAMATH_GPT_area_of_inscribed_rectangle_l2282_228295


namespace NUMINAMATH_GPT_sample_size_l2282_228280

theorem sample_size (T : ℕ) (f_C : ℚ) (samples_C : ℕ) (n : ℕ) 
    (hT : T = 260)
    (hfC : f_C = 3 / 13)
    (hsamples_C : samples_C = 3) : n = 13 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sample_size_l2282_228280


namespace NUMINAMATH_GPT_find_interest_rate_l2282_228272

theorem find_interest_rate (P : ℕ) (diff : ℕ) (T : ℕ) (I2_rate : ℕ) (r : ℚ) 
  (hP : P = 15000) (hdiff : diff = 900) (hT : T = 2) (hI2_rate : I2_rate = 12)
  (h : P * (r / 100) * T = P * (I2_rate / 100) * T + diff) :
  r = 15 :=
sorry

end NUMINAMATH_GPT_find_interest_rate_l2282_228272


namespace NUMINAMATH_GPT_find_a_l2282_228266

theorem find_a (a x : ℝ) (h1 : 3 * x + 5 = 11) (h2 : 6 * x + 3 * a = 22) : a = 10 / 3 :=
by
  -- the proof will go here
  sorry

end NUMINAMATH_GPT_find_a_l2282_228266


namespace NUMINAMATH_GPT_p_sufficient_not_necessary_q_l2282_228284

def p (x : ℝ) := 0 < x ∧ x < 2
def q (x : ℝ) := -1 < x ∧ x < 3

theorem p_sufficient_not_necessary_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end NUMINAMATH_GPT_p_sufficient_not_necessary_q_l2282_228284


namespace NUMINAMATH_GPT_book_price_l2282_228227

theorem book_price (x : ℕ) : 
  9 * x ≤ 1100 ∧ 13 * x ≤ 1500 → x = 123 :=
sorry

end NUMINAMATH_GPT_book_price_l2282_228227


namespace NUMINAMATH_GPT_sequence1_sixth_seventh_terms_sequence2_sixth_term_sequence3_ninth_tenth_terms_l2282_228292

def seq1 (n : ℕ) : ℕ := 2 * (n + 1)
def seq2 (n : ℕ) : ℕ := 3 * 2 ^ n
def seq3 (n : ℕ) : ℕ :=
  if n % 2 = 0 then 36 + n
  else 10 + n
  
theorem sequence1_sixth_seventh_terms :
  seq1 5 = 12 ∧ seq1 6 = 14 :=
by
  sorry

theorem sequence2_sixth_term :
  seq2 5 = 96 :=
by
  sorry

theorem sequence3_ninth_tenth_terms :
  seq3 8 = 44 ∧ seq3 9 = 19 :=
by
  sorry

end NUMINAMATH_GPT_sequence1_sixth_seventh_terms_sequence2_sixth_term_sequence3_ninth_tenth_terms_l2282_228292


namespace NUMINAMATH_GPT_inequalities_in_quadrants_l2282_228260

theorem inequalities_in_quadrants (x y : ℝ) :
  (y > - (1 / 2) * x + 6) ∧ (y > 3 * x - 4) → (x > 0) ∧ (y > 0) :=
  sorry

end NUMINAMATH_GPT_inequalities_in_quadrants_l2282_228260


namespace NUMINAMATH_GPT_minimum_possible_value_of_Box_l2282_228291

theorem minimum_possible_value_of_Box : 
  ∃ (a b Box : ℤ), 
    (a ≠ b) ∧ (a ≠ Box) ∧ (b ≠ Box) ∧
    (a * b = 15) ∧ 
    (∀ x : ℤ, (a * x + b) * (b * x + a) = 15 * x ^ 2 + Box * x + 15) ∧ 
    (∃ p q : ℤ, (p * q = 15 ∧ p ≠ q ∧ p ≠ 34 ∧ q ≠ 34) → (Box = p^2 + q^2)) ∧ 
    Box = 34 :=
by
  sorry

end NUMINAMATH_GPT_minimum_possible_value_of_Box_l2282_228291


namespace NUMINAMATH_GPT_squirrel_cones_l2282_228288

theorem squirrel_cones :
  ∃ (x y : ℕ), 
    x + y < 25 ∧ 
    2 * x > y + 26 ∧ 
    2 * y > x - 4 ∧
    x = 17 ∧ 
    y = 7 :=
by
  sorry

end NUMINAMATH_GPT_squirrel_cones_l2282_228288


namespace NUMINAMATH_GPT_circle_center_x_coordinate_eq_l2282_228246

theorem circle_center_x_coordinate_eq (a : ℝ) (h : (∃ k : ℝ, ∀ x y : ℝ, x^2 + y^2 - a * x = k) ∧ (1 = a / 2)) : a = 2 :=
sorry

end NUMINAMATH_GPT_circle_center_x_coordinate_eq_l2282_228246


namespace NUMINAMATH_GPT_james_proof_l2282_228257

def james_pages_per_hour 
  (writes_some_pages_an_hour : ℕ)
  (writes_5_pages_to_2_people_each_day : ℕ)
  (hours_spent_writing_per_week : ℕ) 
  (writes_total_pages_per_day : ℕ)
  (writes_total_pages_per_week : ℕ) 
  (pages_per_hour : ℕ) 
: Prop :=
  writes_some_pages_an_hour = writes_5_pages_to_2_people_each_day / hours_spent_writing_per_week

theorem james_proof
  (writes_some_pages_an_hour : ℕ := 10)
  (writes_5_pages_to_2_people_each_day : ℕ := 5 * 2)
  (hours_spent_writing_per_week : ℕ := 7)
  (writes_total_pages_per_day : ℕ := writes_5_pages_to_2_people_each_day)
  (writes_total_pages_per_week : ℕ := writes_total_pages_per_day * 7)
  (pages_per_hour : ℕ := writes_total_pages_per_week / hours_spent_writing_per_week)
: writes_some_pages_an_hour = pages_per_hour :=
by {
  sorry 
}

end NUMINAMATH_GPT_james_proof_l2282_228257


namespace NUMINAMATH_GPT_find_number_l2282_228222

theorem find_number (x : ℤ) (h : x + 2 - 3 = 7) : x = 8 :=
sorry

end NUMINAMATH_GPT_find_number_l2282_228222


namespace NUMINAMATH_GPT_additional_savings_l2282_228235

def initial_price : Float := 30
def discount1 : Float := 5
def discount2_percent : Float := 0.25

def price_after_discount1_then_discount2 : Float := 
  (initial_price - discount1) * (1 - discount2_percent)

def price_after_discount2_then_discount1 : Float := 
  initial_price * (1 - discount2_percent) - discount1

theorem additional_savings :
  price_after_discount1_then_discount2 - price_after_discount2_then_discount1 = 1.25 := by
  sorry

end NUMINAMATH_GPT_additional_savings_l2282_228235


namespace NUMINAMATH_GPT_minimum_number_of_colors_l2282_228271

theorem minimum_number_of_colors (n : ℕ) (h_n : 2 ≤ n) :
  ∀ (f : (Fin n) → ℕ),
  (∀ i j : Fin n, i ≠ j → f i ≠ f j) →
  (∃ c : ℕ, c = n) :=
by sorry

end NUMINAMATH_GPT_minimum_number_of_colors_l2282_228271


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l2282_228281

-- Condition: x1 and x2 are the roots of the quadratic equation x^2 - 2(m+2)x + m^2 = 0
variables {x1 x2 m : ℝ}
axiom roots_quadratic_equation : x1^2 - 2*(m+2) * x1 + m^2 = 0 ∧ x2^2 - 2*(m+2) * x2 + m^2 = 0

-- 1. When m = 0, the roots of the equation are 0 and 4
theorem problem_1 (h : m = 0) : x1 = 0 ∧ x2 = 4 :=
by 
  sorry

-- 2. If (x1 - 2)(x2 - 2) = 41, then m = 9
theorem problem_2 (h : (x1 - 2) * (x2 - 2) = 41) : m = 9 :=
by
  sorry

-- 3. Given an isosceles triangle ABC with one side length 9, if x1 and x2 are the lengths of the other two sides, 
--    prove that the perimeter is 19.
theorem problem_3 (h1 : x1 + x2 > 9) (h2 : 9 + x1 > x2) (h3 : 9 + x2 > x1) : x1 = 1 ∧ x2 = 9 ∧ (x1 + x2 + 9) = 19 :=
by 
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l2282_228281


namespace NUMINAMATH_GPT_x_varies_as_z_l2282_228204

variable {x y z : ℝ}
variable (k j : ℝ)
variable (h1 : x = k * y^3)
variable (h2 : y = j * z^(1/3))

theorem x_varies_as_z (m : ℝ) (h3 : m = k * j^3) : x = m * z := by
  sorry

end NUMINAMATH_GPT_x_varies_as_z_l2282_228204


namespace NUMINAMATH_GPT_undefined_values_of_fraction_l2282_228212

theorem undefined_values_of_fraction (b : ℝ) : b^2 - 9 = 0 ↔ b = 3 ∨ b = -3 := by
  sorry

end NUMINAMATH_GPT_undefined_values_of_fraction_l2282_228212


namespace NUMINAMATH_GPT_range_of_a_l2282_228213

def my_Op (a b : ℝ) : ℝ := a - 2 * b

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, my_Op x 3 > 0 → my_Op x a > a) ↔ (∀ x : ℝ, x > 6 → x > 3 * a) → a ≤ 2 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l2282_228213


namespace NUMINAMATH_GPT_systematic_sampling_interval_l2282_228245

def population_size : ℕ := 2000
def sample_size : ℕ := 50
def interval (N n : ℕ) : ℕ := N / n

theorem systematic_sampling_interval :
  interval population_size sample_size = 40 := by
  sorry

end NUMINAMATH_GPT_systematic_sampling_interval_l2282_228245


namespace NUMINAMATH_GPT_edward_money_left_l2282_228207

def earnings_from_lawns (lawns_mowed : Nat) (dollar_per_lawn : Nat) : Nat :=
  lawns_mowed * dollar_per_lawn

def earnings_from_gardens (gardens_cleaned : Nat) (dollar_per_garden : Nat) : Nat :=
  gardens_cleaned * dollar_per_garden

def total_earnings (earnings_lawns : Nat) (earnings_gardens : Nat) : Nat :=
  earnings_lawns + earnings_gardens

def total_expenses (fuel_expense : Nat) (equipment_expense : Nat) : Nat :=
  fuel_expense + equipment_expense

def total_earnings_with_savings (total_earnings : Nat) (savings : Nat) : Nat :=
  total_earnings + savings

def money_left (earnings_with_savings : Nat) (expenses : Nat) : Nat :=
  earnings_with_savings - expenses

theorem edward_money_left : 
  let lawns_mowed := 5
  let dollar_per_lawn := 8
  let gardens_cleaned := 3
  let dollar_per_garden := 12
  let fuel_expense := 10
  let equipment_expense := 15
  let savings := 7
  let earnings_lawns := earnings_from_lawns lawns_mowed dollar_per_lawn
  let earnings_gardens := earnings_from_gardens gardens_cleaned dollar_per_garden
  let total_earnings_work := total_earnings earnings_lawns earnings_gardens
  let expenses := total_expenses fuel_expense equipment_expense
  let earnings_with_savings := total_earnings_with_savings total_earnings_work savings
  money_left earnings_with_savings expenses = 58
:= by sorry

end NUMINAMATH_GPT_edward_money_left_l2282_228207


namespace NUMINAMATH_GPT_solve_fractional_equation_l2282_228283

theorem solve_fractional_equation (x : ℝ) :
  (x ≠ 5) ∧ (x ≠ 6) ∧ ((x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1)) / ((x - 5) * (x - 6) * (x - 5)) = 1 ↔ 
  x = 2 ∨ x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_solve_fractional_equation_l2282_228283


namespace NUMINAMATH_GPT_largest_prime_form_2pow_n_plus_nsq_minus_1_less_than_100_l2282_228202

def is_prime (n : ℕ) : Prop := sorry -- Use inbuilt primality function or define it

def expression (n : ℕ) : ℕ := 2^n + n^2 - 1

theorem largest_prime_form_2pow_n_plus_nsq_minus_1_less_than_100 :
  ∃ m, is_prime m ∧ (∃ n, is_prime n ∧ expression n = m ∧ m < 100) ∧
        ∀ k, is_prime k ∧ (∃ n, is_prime n ∧ expression n = k ∧ k < 100) → k <= m :=
  sorry

end NUMINAMATH_GPT_largest_prime_form_2pow_n_plus_nsq_minus_1_less_than_100_l2282_228202


namespace NUMINAMATH_GPT_fraction_evaluation_l2282_228244

theorem fraction_evaluation :
  (2 + 3 * 6) / (23 + 6) = 20 / 29 := by
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_fraction_evaluation_l2282_228244


namespace NUMINAMATH_GPT_f_f_n_plus_n_eq_n_plus_1_l2282_228228

-- Define the function f : ℕ+ → ℕ+ satisfying the given condition
axiom f : ℕ+ → ℕ+

-- Define that for all positive integers n, f satisfies the condition f(f(n)) + f(n+1) = n + 2
axiom f_condition : ∀ n : ℕ+, f (f n) + f (n + 1) = n + 2

-- State that we want to prove that f(f(n) + n) = n + 1 for all positive integers n
theorem f_f_n_plus_n_eq_n_plus_1 : ∀ n : ℕ+, f (f n + n) = n + 1 := 
by sorry

end NUMINAMATH_GPT_f_f_n_plus_n_eq_n_plus_1_l2282_228228


namespace NUMINAMATH_GPT_triangle_max_distance_product_l2282_228269

open Real

noncomputable def max_product_of_distances
  (a b c : ℝ) (P : {p : ℝ × ℝ // True}) : ℝ :=
  let h_a := 1 -- placeholder for actual distance calculation
  let h_b := 1 -- placeholder for actual distance calculation
  let h_c := 1 -- placeholder for actual distance calculation
  h_a * h_b * h_c

theorem triangle_max_distance_product
  (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5)
  (P : {p : ℝ × ℝ // True}) :
  max_product_of_distances a b c P = (16/15 : ℝ) :=
sorry

end NUMINAMATH_GPT_triangle_max_distance_product_l2282_228269


namespace NUMINAMATH_GPT_proof_problem_l2282_228278

variable {a1 a2 b1 b2 b3 : ℝ}

-- Condition: -2, a1, a2, -8 form an arithmetic sequence
def arithmetic_sequence (a1 a2 : ℝ) : Prop :=
  a2 - a1 = -2 / 3 * (-2 - 8)

-- Condition: -2, b1, b2, b3, -8 form a geometric sequence
def geometric_sequence (b1 b2 b3 : ℝ) : Prop :=
  b2^2 = (-2) * (-8) ∧ b1^2 = (-2) * b2 ∧ b3^2 = b2 * (-8)

theorem proof_problem (h1 : arithmetic_sequence a1 a2) (h2 : geometric_sequence b1 b2 b3) : b2 * (a2 - a1) = 8 :=
by
  admit -- Convert to sorry to skip the proof

end NUMINAMATH_GPT_proof_problem_l2282_228278


namespace NUMINAMATH_GPT_pasha_wins_9_games_l2282_228231

theorem pasha_wins_9_games :
  ∃ w l : ℕ, (w + l = 12) ∧ (2^w * (2^l - 1) - (2^l - 1) * 2^(w - 1) = 2023) ∧ (w = 9) :=
by
  sorry

end NUMINAMATH_GPT_pasha_wins_9_games_l2282_228231


namespace NUMINAMATH_GPT_container_volume_ratio_l2282_228249

theorem container_volume_ratio
  (A B C : ℚ)  -- A is the volume of the first container, B is the volume of the second container, C is the volume of the third container
  (h1 : (8 / 9) * A = (7 / 9) * B)  -- Condition: First container was 8/9 full and second container gets filled to 7/9 after transfer.
  (h2 : (7 / 9) * B + (1 / 2) * C = C)  -- Condition: Mixing contents from second and third containers completely fill third container.
  : A / C = 63 / 112 := sorry  -- We need to prove this.

end NUMINAMATH_GPT_container_volume_ratio_l2282_228249


namespace NUMINAMATH_GPT_find_f_on_interval_l2282_228242

/-- Representation of periodic and even functions along with specific interval definition -/
noncomputable def f (x : ℝ) : ℝ := 
if 2 ≤ x ∧ x ≤ 3 then -2*(x-3)^2 + 4 else 0 -- Define f(x) on [2,3], otherwise undefined

/-- Main proof statement -/
theorem find_f_on_interval :
  (∀ x, f x = f (x + 2)) ∧  -- f(x) is periodic with period 2
  (∀ x, f x = f (-x)) ∧   -- f(x) is even
  (∀ x, 2 ≤ x ∧ x ≤ 3 → f x = -2*(x-3)^2 + 4) →
  (∀ x, 0 ≤ x ∧ x ≤ 2 → f x = -2*(x-1)^2 + 4) :=
sorry

end NUMINAMATH_GPT_find_f_on_interval_l2282_228242


namespace NUMINAMATH_GPT_total_spokes_in_garage_l2282_228220

theorem total_spokes_in_garage :
  let bicycle1_spokes := 12 + 10
  let bicycle2_spokes := 14 + 12
  let bicycle3_spokes := 10 + 14
  let tricycle_spokes := 14 + 12 + 16
  bicycle1_spokes + bicycle2_spokes + bicycle3_spokes + tricycle_spokes = 114 :=
by
  let bicycle1_spokes := 12 + 10
  let bicycle2_spokes := 14 + 12
  let bicycle3_spokes := 10 + 14
  let tricycle_spokes := 14 + 12 + 16
  show bicycle1_spokes + bicycle2_spokes + bicycle3_spokes + tricycle_spokes = 114
  sorry

end NUMINAMATH_GPT_total_spokes_in_garage_l2282_228220


namespace NUMINAMATH_GPT_andrew_daily_work_hours_l2282_228232

theorem andrew_daily_work_hours (total_hours : ℝ) (days : ℝ) (h1 : total_hours = 7.5) (h2 : days = 3) : total_hours / days = 2.5 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_andrew_daily_work_hours_l2282_228232


namespace NUMINAMATH_GPT_jerry_average_increase_l2282_228253

-- Definitions of conditions
def first_three_tests_average (avg : ℕ) : Prop := avg = 85
def fourth_test_score (score : ℕ) : Prop := score = 97
def desired_average_increase (increase : ℕ) : Prop := increase = 3

-- The theorem to prove
theorem jerry_average_increase
  (first_avg first_avg_value : ℕ)
  (fourth_score fourth_score_value : ℕ)
  (increase_points : ℕ)
  (h1 : first_three_tests_average first_avg)
  (h2 : fourth_test_score fourth_score)
  (h3 : desired_average_increase increase_points) :
  fourth_score = 97 → (first_avg + fourth_score) / 4 = 88 → increase_points = 3 :=
by
  intros _ _
  sorry

end NUMINAMATH_GPT_jerry_average_increase_l2282_228253


namespace NUMINAMATH_GPT_solve_equation_l2282_228218

theorem solve_equation : ∀ (x : ℝ), x ≠ -3 → x ≠ 3 → 
  (x / (x + 3) + 6 / (x^2 - 9) = 1 / (x - 3)) → x = 1 :=
by
  intros x hx1 hx2 h
  sorry

end NUMINAMATH_GPT_solve_equation_l2282_228218


namespace NUMINAMATH_GPT_num_triangles_square_even_num_triangles_rect_even_l2282_228289

-- Problem (a): Proving that the number of triangles is even 
theorem num_triangles_square_even (a : ℕ) (n : ℕ) (h : a * a = n * (3 * 4 / 2)) : 
  n % 2 = 0 :=
sorry

-- Problem (b): Proving that the number of triangles is even
theorem num_triangles_rect_even (L W k : ℕ) (hL : L = k * 2) (hW : W = k * 1) (h : L * W = k * 1 * 2 / 2) :
  k % 2 = 0 :=
sorry

end NUMINAMATH_GPT_num_triangles_square_even_num_triangles_rect_even_l2282_228289


namespace NUMINAMATH_GPT_find_C_l2282_228275

def A : ℝ × ℝ := (2, 8)
def M : ℝ × ℝ := (4, 11)
def L : ℝ × ℝ := (6, 6)

theorem find_C (C : ℝ × ℝ) (B : ℝ × ℝ) :
  -- Median condition: M is the midpoint of A and B
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  -- Given coordinates for A, M, L
  A = (2, 8) → M = (4, 11) → L = (6, 6) →
  -- Correct answer
  C = (14, 2) :=
by
  intros hmedian hA hM hL
  sorry

end NUMINAMATH_GPT_find_C_l2282_228275


namespace NUMINAMATH_GPT_max_regions_divided_by_lines_l2282_228237

theorem max_regions_divided_by_lines (m n : ℕ) (hm : m ≠ 0) (hn : n ≠ 0) :
  ∃ r : ℕ, r = m * n + 2 * m + 2 * n - 1 :=
by
  sorry

end NUMINAMATH_GPT_max_regions_divided_by_lines_l2282_228237


namespace NUMINAMATH_GPT_emily_small_gardens_l2282_228239

theorem emily_small_gardens (total_seeds planted_big_garden seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 41) 
  (h2 : planted_big_garden = 29) 
  (h3 : seeds_per_small_garden = 4) : 
  (total_seeds - planted_big_garden) / seeds_per_small_garden = 3 := 
by
  sorry

end NUMINAMATH_GPT_emily_small_gardens_l2282_228239


namespace NUMINAMATH_GPT_tshirts_per_package_l2282_228262

-- Definitions based on the conditions
def total_tshirts : ℕ := 70
def num_packages : ℕ := 14

-- Theorem to prove the number of t-shirts per package
theorem tshirts_per_package : total_tshirts / num_packages = 5 := by
  -- The proof is omitted, only the statement is provided as required.
  sorry

end NUMINAMATH_GPT_tshirts_per_package_l2282_228262


namespace NUMINAMATH_GPT_smallest_x_l2282_228268

theorem smallest_x (x : ℕ) :
  (x % 6 = 5) ∧ (x % 7 = 6) ∧ (x % 8 = 7) → x = 167 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_l2282_228268


namespace NUMINAMATH_GPT_pool_one_quarter_capacity_at_6_l2282_228219

-- Variables and parameters
variables (volume : ℕ → ℝ) (T : ℕ)

-- Conditions
def doubles_every_hour : Prop :=
  ∀ t, volume (t + 1) = 2 * volume t

def full_capacity_at_8 : Prop :=
  volume 8 = T

def one_quarter_capacity (t : ℕ) : Prop :=
  volume t = T / 4

-- Theorem to prove
theorem pool_one_quarter_capacity_at_6 (h1 : doubles_every_hour volume) (h2 : full_capacity_at_8 volume T) : one_quarter_capacity volume T 6 :=
sorry

end NUMINAMATH_GPT_pool_one_quarter_capacity_at_6_l2282_228219


namespace NUMINAMATH_GPT_probability_A_mc_and_B_tf_probability_at_least_one_mc_l2282_228261

-- Define the total number of questions
def total_questions : ℕ := 5

-- Define the number of multiple choice questions and true or false questions
def multiple_choice_questions : ℕ := 3
def true_false_questions : ℕ := 2

-- First proof problem: Probability that A draws a multiple-choice question and B draws a true or false question
theorem probability_A_mc_and_B_tf :
  (multiple_choice_questions * true_false_questions : ℚ) / (total_questions * (total_questions - 1)) = 3 / 10 :=
by
  sorry

-- Second proof problem: Probability that at least one of A and B draws a multiple-choice question
theorem probability_at_least_one_mc :
  1 - (true_false_questions * (true_false_questions - 1) : ℚ) / (total_questions * (total_questions - 1)) = 9 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_A_mc_and_B_tf_probability_at_least_one_mc_l2282_228261


namespace NUMINAMATH_GPT_fraction_product_is_one_l2282_228243

theorem fraction_product_is_one : 
  (1 / 4) * (1 / 5) * (1 / 6) * 120 = 1 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_product_is_one_l2282_228243


namespace NUMINAMATH_GPT_erica_pie_fraction_as_percentage_l2282_228217

theorem erica_pie_fraction_as_percentage (apple_pie_fraction : ℚ) (cherry_pie_fraction : ℚ) 
  (h1 : apple_pie_fraction = 1 / 5) 
  (h2 : cherry_pie_fraction = 3 / 4) 
  (common_denominator : ℚ := 20) : 
  (apple_pie_fraction + cherry_pie_fraction) * 100 = 95 :=
by
  sorry

end NUMINAMATH_GPT_erica_pie_fraction_as_percentage_l2282_228217


namespace NUMINAMATH_GPT_diplomats_not_speaking_russian_l2282_228205

-- Definitions to formalize the problem
def total_diplomats : ℕ := 150
def speak_french : ℕ := 17
def speak_both_french_and_russian : ℕ := (10 * total_diplomats) / 100
def speak_neither_french_nor_russian : ℕ := (20 * total_diplomats) / 100

-- Theorem to prove the desired quantity
theorem diplomats_not_speaking_russian : 
  speak_neither_french_nor_russian + (speak_french - speak_both_french_and_russian) = 32 := by
  sorry

end NUMINAMATH_GPT_diplomats_not_speaking_russian_l2282_228205


namespace NUMINAMATH_GPT_line_through_longest_chord_l2282_228230

-- Define the point M and the circle equation
def M : ℝ × ℝ := (3, -1)
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + y - 2 = 0

-- Define the standard form of the circle equation
def standard_circle_eqn (x y : ℝ) : Prop := (x - 2)^2 + (y + 1/2)^2 = 25/4

-- Define the line equation
def line_eqn (x y : ℝ) : Prop := x + 2 * y - 2 = 0

-- Theorem: Equation of the line containing the longest chord passing through M
theorem line_through_longest_chord : 
  (circle_eqn 3 (-1)) → 
  ∀ (x y : ℝ), standard_circle_eqn x y → ∃ (k b : ℝ), line_eqn x y :=
by
  -- Proof goes here
  intro h1 x y h2
  sorry

end NUMINAMATH_GPT_line_through_longest_chord_l2282_228230


namespace NUMINAMATH_GPT_hyperbola_center_l2282_228254

theorem hyperbola_center (x y : ℝ) :
  ( ∃ (h k : ℝ), ∀ (x y : ℝ), (4 * x - 8)^2 / 9^2 - (5 * y - 15)^2 / 7^2 = 1 → (h, k) = (2, 3) ) :=
by
  existsi 2
  existsi 3
  intros x y h
  sorry

end NUMINAMATH_GPT_hyperbola_center_l2282_228254


namespace NUMINAMATH_GPT_find_base_s_l2282_228210

-- Definitions based on the conditions.
def five_hundred_thirty_base (s : ℕ) : ℕ := 5 * s^2 + 3 * s
def four_hundred_fifty_base (s : ℕ) : ℕ := 4 * s^2 + 5 * s
def one_thousand_one_hundred_base (s : ℕ) : ℕ := s^3 + s^2

-- The theorem to prove.
theorem find_base_s : (∃ s : ℕ, five_hundred_thirty_base s + four_hundred_fifty_base s = one_thousand_one_hundred_base s) → s = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_base_s_l2282_228210


namespace NUMINAMATH_GPT_range_of_m_l2282_228240

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, mx^2 - 4*x + 1 = 0 ∧ ∀ y : ℝ, mx^2 - 4*x + 1 = 0 → y = x) → m ≤ 4 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2282_228240


namespace NUMINAMATH_GPT_min_score_seventh_shot_to_break_record_shots_hitting_10_to_break_record_when_7th_shot_is_8_necessary_shot_of_10_when_7th_shot_is_10_l2282_228286

-- Definitions for the problem conditions
def initial_points : ℕ := 52
def record_points : ℕ := 89
def max_shots : ℕ := 10
def points_range : Finset ℕ := Finset.range 11 \ {0}

-- Lean statement for the first question
theorem min_score_seventh_shot_to_break_record (x₇ : ℕ) (h₁: x₇ ∈ points_range) :
  initial_points + x₇ + 30 > record_points ↔ x₇ ≥ 8 :=
by sorry

-- Lean statement for the second question
theorem shots_hitting_10_to_break_record_when_7th_shot_is_8 (x₈ x₉ x₁₀ : ℕ)
  (h₂ : 8 ∈ points_range) 
  (h₃ : x₈ ∈ points_range) (h₄ : x₉ ∈ points_range) (h₅ : x₁₀ ∈ points_range) :
  initial_points + 8 + x₈ + x₉ + x₁₀ > record_points ↔ (x₈ = 10 ∧ x₉ = 10 ∧ x₁₀ = 10) :=
by sorry

-- Lean statement for the third question
theorem necessary_shot_of_10_when_7th_shot_is_10 (x₈ x₉ x₁₀ : ℕ)
  (h₆ : 10 ∈ points_range)
  (h₇ : x₈ ∈ points_range) (h₈ : x₉ ∈ points_range) (h₉ : x₁₀ ∈ points_range) :
  initial_points + 10 + x₈ + x₉ + x₁₀ > record_points ↔ (x₈ = 10 ∨ x₉ = 10 ∨ x₁₀ = 10) :=
by sorry

end NUMINAMATH_GPT_min_score_seventh_shot_to_break_record_shots_hitting_10_to_break_record_when_7th_shot_is_8_necessary_shot_of_10_when_7th_shot_is_10_l2282_228286


namespace NUMINAMATH_GPT_whisky_replacement_l2282_228233

variable (V : ℝ) (x : ℝ)

theorem whisky_replacement (h_condition : 0.40 * V - 0.40 * x + 0.19 * x = 0.26 * V) : 
  x = (2 / 3) * V := 
sorry

end NUMINAMATH_GPT_whisky_replacement_l2282_228233


namespace NUMINAMATH_GPT_latte_price_l2282_228293

theorem latte_price
  (almond_croissant_price salami_croissant_price plain_croissant_price focaccia_price total_spent : ℝ)
  (lattes_count : ℕ)
  (H1 : almond_croissant_price = 4.50)
  (H2 : salami_croissant_price = 4.50)
  (H3 : plain_croissant_price = 3.00)
  (H4 : focaccia_price = 4.00)
  (H5 : total_spent = 21.00)
  (H6 : lattes_count = 2) :
  (total_spent - (almond_croissant_price + salami_croissant_price + plain_croissant_price + focaccia_price)) / lattes_count = 2.50 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_latte_price_l2282_228293


namespace NUMINAMATH_GPT_industrial_park_investment_l2282_228259

noncomputable def investment_in_projects : Prop :=
  ∃ (x : ℝ), 0.054 * x + 0.0828 * (2000 - x) = 122.4 ∧ x = 1500 ∧ (2000 - x) = 500

theorem industrial_park_investment :
  investment_in_projects :=
by
  have h : ∃ (x : ℝ), 0.054 * x + 0.0828 * (2000 - x) = 122.4 ∧ x = 1500 ∧ (2000 - x) = 500 := 
    sorry
  exact h

end NUMINAMATH_GPT_industrial_park_investment_l2282_228259


namespace NUMINAMATH_GPT_find_number_with_10_questions_l2282_228225

theorem find_number_with_10_questions (n : ℕ) (h : n ≤ 1000) : n = 300 :=
by
  sorry

end NUMINAMATH_GPT_find_number_with_10_questions_l2282_228225


namespace NUMINAMATH_GPT_fill_time_with_leak_l2282_228299

theorem fill_time_with_leak (A L : ℝ) (hA : A = 1 / 5) (hL : L = 1 / 10) :
  1 / (A - L) = 10 :=
by 
  sorry

end NUMINAMATH_GPT_fill_time_with_leak_l2282_228299


namespace NUMINAMATH_GPT_percentage_increase_visitors_l2282_228226

theorem percentage_increase_visitors
  (original_visitors : ℕ)
  (original_fee : ℝ := 1)
  (fee_reduction : ℝ := 0.25)
  (visitors_increase : ℝ := 0.20) :
  ((original_visitors + (visitors_increase * original_visitors)) / original_visitors - 1) * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_visitors_l2282_228226


namespace NUMINAMATH_GPT_largest_angle_is_75_l2282_228224

-- Let the measures of the angles be represented as 3x, 4x, and 5x for some value x
variable (x : ℝ)

-- Define the angles based on the given ratio
def angle1 := 3 * x
def angle2 := 4 * x
def angle3 := 5 * x

-- The sum of the angles in a triangle is 180 degrees
axiom angle_sum : angle1 + angle2 + angle3 = 180

-- Prove that the largest angle is 75 degrees
theorem largest_angle_is_75 : 5 * (180 / 12) = 75 :=
by
  -- Proof is not required as per the instructions
  sorry

end NUMINAMATH_GPT_largest_angle_is_75_l2282_228224


namespace NUMINAMATH_GPT_megan_folders_count_l2282_228221

theorem megan_folders_count (init_files deleted_files files_per_folder : ℕ) (h₁ : init_files = 93) (h₂ : deleted_files = 21) (h₃ : files_per_folder = 8) :
  (init_files - deleted_files) / files_per_folder = 9 :=
by
  sorry

end NUMINAMATH_GPT_megan_folders_count_l2282_228221


namespace NUMINAMATH_GPT_ryan_distance_correct_l2282_228248

-- Definitions of the conditions
def billy_distance : ℝ := 30
def madison_distance : ℝ := billy_distance * 1.2
def ryan_distance : ℝ := madison_distance * 0.5

-- Statement to prove
theorem ryan_distance_correct : ryan_distance = 18 := by
  sorry

end NUMINAMATH_GPT_ryan_distance_correct_l2282_228248


namespace NUMINAMATH_GPT_fraction_of_subsets_l2282_228203

theorem fraction_of_subsets (S T : ℕ) (hS : S = 2^10) (hT : T = Nat.choose 10 3) :
    (T:ℚ) / (S:ℚ) = 15 / 128 :=
by sorry

end NUMINAMATH_GPT_fraction_of_subsets_l2282_228203


namespace NUMINAMATH_GPT_LukaLemonadeSolution_l2282_228256

def LukaLemonadeProblem : Prop :=
  ∃ (L S W : ℕ), 
    (S = 3 * L) ∧
    (W = 3 * S) ∧
    (L = 4) ∧
    (W = 36)

theorem LukaLemonadeSolution : LukaLemonadeProblem :=
  by sorry

end NUMINAMATH_GPT_LukaLemonadeSolution_l2282_228256


namespace NUMINAMATH_GPT_find_integer_l2282_228206

def satisfies_conditions (x : ℕ) (m n : ℕ) : Prop :=
  x + 100 = m ^ 2 ∧ x + 168 = n ^ 2 ∧ m > 0 ∧ n > 0

theorem find_integer (x m n : ℕ) (h : satisfies_conditions x m n) : x = 156 :=
sorry

end NUMINAMATH_GPT_find_integer_l2282_228206


namespace NUMINAMATH_GPT_tree_height_at_2_years_l2282_228273

-- Define the conditions
def triples_height (height : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, height (n + 1) = 3 * height n

def height_at_5_years (height : ℕ → ℕ) : Prop :=
  height 5 = 243

-- Set up the problem statement
theorem tree_height_at_2_years (height : ℕ → ℕ) 
  (H1 : triples_height height) 
  (H2 : height_at_5_years height) : 
  height 2 = 9 :=
sorry

end NUMINAMATH_GPT_tree_height_at_2_years_l2282_228273


namespace NUMINAMATH_GPT_average_hit_targets_formula_average_hit_targets_ge_half_l2282_228208

open Nat Real

noncomputable def average_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - (1 : ℝ) / n) ^ n)

theorem average_hit_targets_formula (n : ℕ) (h : 0 < n) :
  average_hit_targets n = n * (1 - (1 - (1 : ℝ) / n) ^ n) :=
by
  sorry

theorem average_hit_targets_ge_half (n : ℕ) (h : 0 < n) :
  average_hit_targets n ≥ n / 2 :=
by
  sorry

end NUMINAMATH_GPT_average_hit_targets_formula_average_hit_targets_ge_half_l2282_228208


namespace NUMINAMATH_GPT_number_of_rectangles_l2282_228201

theorem number_of_rectangles (horizontal_lines : Fin 6) (vertical_lines : Fin 5) 
                             (point : ℕ × ℕ) (h₁ : point = (3, 4)) : 
  ∃ ways : ℕ, ways = 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_rectangles_l2282_228201


namespace NUMINAMATH_GPT_basic_cable_cost_l2282_228216

variable (B M S : ℝ)

def CostOfMovieChannels (B : ℝ) : ℝ := B + 12
def CostOfSportsChannels (M : ℝ) : ℝ := M - 3

theorem basic_cable_cost :
  let M := CostOfMovieChannels B
  let S := CostOfSportsChannels M
  B + M + S = 36 → B = 5 :=
by
  intro h
  let M := CostOfMovieChannels B
  let S := CostOfSportsChannels M
  sorry

end NUMINAMATH_GPT_basic_cable_cost_l2282_228216


namespace NUMINAMATH_GPT_value_b_minus_a_l2282_228250

theorem value_b_minus_a (a b : ℝ) (h₁ : a + b = 507) (h₂ : (a - b) / b = 1 / 7) : b - a = -34.428571 :=
by
  sorry

end NUMINAMATH_GPT_value_b_minus_a_l2282_228250


namespace NUMINAMATH_GPT_solve_quadratic_eq_l2282_228298

theorem solve_quadratic_eq (a c : ℝ) (h1 : a + c = 31) (h2 : a < c) (h3 : (24:ℝ)^2 - 4 * a * c = 0) : a = 9 ∧ c = 22 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_quadratic_eq_l2282_228298


namespace NUMINAMATH_GPT_range_of_m_l2282_228265

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ 4) → 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → 
  -3 ≤ m ∧ m ≤ 5 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_m_l2282_228265


namespace NUMINAMATH_GPT_greatest_fourth_term_l2282_228277

theorem greatest_fourth_term (a d : ℕ) (h1 : a > 0) (h2 : d > 0) 
  (h3 : 5 * a + 10 * d = 50) (h4 : a + 2 * d = 10) : 
  a + 3 * d = 14 :=
by {
  -- We introduced the given constraints and now need a proof
  sorry
}

end NUMINAMATH_GPT_greatest_fourth_term_l2282_228277


namespace NUMINAMATH_GPT_correlation_identification_l2282_228274

noncomputable def relationship (a b : Type) : Prop := 
  ∃ (f : a → b), true

def correlation (a b : Type) : Prop :=
  relationship a b ∧ relationship b a

def deterministic (a b : Type) : Prop :=
  ∀ x y : a, ∃! z : b, true

def age_wealth : Prop := correlation ℕ ℝ
def point_curve_coordinates : Prop := deterministic (ℝ × ℝ) (ℝ × ℝ)
def apple_production_climate : Prop := correlation ℝ ℝ
def tree_diameter_height : Prop := correlation ℝ ℝ

theorem correlation_identification :
  age_wealth ∧ apple_production_climate ∧ tree_diameter_height ∧ ¬point_curve_coordinates := 
by
  -- proof of these properties
  sorry

end NUMINAMATH_GPT_correlation_identification_l2282_228274
