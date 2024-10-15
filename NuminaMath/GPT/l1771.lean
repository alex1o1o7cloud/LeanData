import Mathlib

namespace NUMINAMATH_GPT_geometric_series_sum_l1771_177125

theorem geometric_series_sum :
  let a := 2
  let r := 2
  let n := 11
  let S := a * (r^n - 1) / (r - 1)
  S = 4094 := by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1771_177125


namespace NUMINAMATH_GPT_field_area_is_243_l1771_177176

noncomputable def field_area (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 72) : ℝ :=
  w * l

theorem field_area_is_243 (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 72) : field_area w l h1 h2 = 243 :=
  sorry

end NUMINAMATH_GPT_field_area_is_243_l1771_177176


namespace NUMINAMATH_GPT_mean_square_sum_l1771_177162

theorem mean_square_sum (x y z : ℝ) 
  (h1 : x + y + z = 27)
  (h2 : x * y * z = 216)
  (h3 : x * y + y * z + z * x = 162) : 
  x^2 + y^2 + z^2 = 405 :=
by
  sorry

end NUMINAMATH_GPT_mean_square_sum_l1771_177162


namespace NUMINAMATH_GPT_union_of_intervals_l1771_177115

open Set

theorem union_of_intervals :
  let M := { x : ℝ | 1 < x ∧ x ≤ 3 }
  let N := { x : ℝ | 2 < x ∧ x ≤ 5 }
  M ∪ N = { x : ℝ | 1 < x ∧ x ≤ 5 } :=
by
  let M := { x : ℝ | 1 < x ∧ x ≤ 3 }
  let N := { x : ℝ | 2 < x ∧ x ≤ 5 }
  sorry

end NUMINAMATH_GPT_union_of_intervals_l1771_177115


namespace NUMINAMATH_GPT_problem_1_problem_2_l1771_177127

-- Definitions of sets A and B
def A : Set ℝ := { x : ℝ | x^2 - 2 * x - 3 ≤ 0 }
def B (m : ℝ) : Set ℝ := { x : ℝ | m - 1 ≤ x ∧ x ≤ m + 1 }

-- Problem 1: Prove that if A ∩ B = [1, 3], then m = 2
theorem problem_1 (m : ℝ) (h : (A ∩ B m) = {x : ℝ | 1 ≤ x ∧ x ≤ 3}) : m = 2 :=
sorry

-- Problem 2: Prove that if A ⊆ complement ℝ B m, then m > 4 or m < -2
theorem problem_2 (m : ℝ) (h : A ⊆ { x : ℝ | x < m - 1 ∨ x > m + 1 }) : m > 4 ∨ m < -2 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1771_177127


namespace NUMINAMATH_GPT_power_expression_result_l1771_177141

theorem power_expression_result : (-2)^2004 + (-2)^2005 = -2^2004 :=
by
  sorry

end NUMINAMATH_GPT_power_expression_result_l1771_177141


namespace NUMINAMATH_GPT_solution_x_x_sub_1_eq_x_l1771_177113

theorem solution_x_x_sub_1_eq_x (x : ℝ) : x * (x - 1) = x ↔ (x = 0 ∨ x = 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_x_x_sub_1_eq_x_l1771_177113


namespace NUMINAMATH_GPT_pies_sold_l1771_177121

-- Define the conditions in Lean
def num_cakes : ℕ := 453
def price_per_cake : ℕ := 12
def total_earnings : ℕ := 6318
def price_per_pie : ℕ := 7

-- Define the problem
theorem pies_sold (P : ℕ) (h1 : num_cakes * price_per_cake + P * price_per_pie = total_earnings) : P = 126 := 
by 
  sorry

end NUMINAMATH_GPT_pies_sold_l1771_177121


namespace NUMINAMATH_GPT_cats_left_in_store_l1771_177164

theorem cats_left_in_store 
  (initial_siamese : ℕ := 25)
  (initial_persian : ℕ := 18)
  (initial_house : ℕ := 12)
  (initial_maine_coon : ℕ := 10)
  (sold_siamese : ℕ := 6)
  (sold_persian : ℕ := 4)
  (sold_maine_coon : ℕ := 3)
  (sold_house : ℕ := 0)
  (remaining_siamese : ℕ := 19)
  (remaining_persian : ℕ := 14)
  (remaining_house : ℕ := 12)
  (remaining_maine_coon : ℕ := 7) : 
  initial_siamese - sold_siamese = remaining_siamese ∧
  initial_persian - sold_persian = remaining_persian ∧
  initial_house - sold_house = remaining_house ∧
  initial_maine_coon - sold_maine_coon = remaining_maine_coon :=
by sorry

end NUMINAMATH_GPT_cats_left_in_store_l1771_177164


namespace NUMINAMATH_GPT_overall_gain_percent_l1771_177182

variables (C_A S_A C_B S_B : ℝ)

def cost_price_A (n : ℝ) : ℝ := n * C_A
def selling_price_A (n : ℝ) : ℝ := n * S_A

def cost_price_B (n : ℝ) : ℝ := n * C_B
def selling_price_B (n : ℝ) : ℝ := n * S_B

theorem overall_gain_percent :
  (selling_price_A 25 = cost_price_A 50) →
  (selling_price_B 30 = cost_price_B 60) →
  ((S_A - C_A) / C_A * 100 = 100) ∧ ((S_B - C_B) / C_B * 100 = 100) :=
by
  sorry

end NUMINAMATH_GPT_overall_gain_percent_l1771_177182


namespace NUMINAMATH_GPT_josh_bottle_caps_l1771_177170

/--
Suppose:
1. 7 bottle caps weigh exactly one ounce.
2. Josh's entire bottle cap collection weighs 18 pounds exactly.
3. There are 16 ounces in 1 pound.
We aim to show that Josh has 2016 bottle caps in his collection.
-/
theorem josh_bottle_caps :
  (7 : ℕ) * (1 : ℕ) = (7 : ℕ) → 
  (18 : ℕ) * (16 : ℕ) = (288 : ℕ) →
  (288 : ℕ) * (7 : ℕ) = (2016 : ℕ) :=
by
  intros h1 h2;
  exact sorry

end NUMINAMATH_GPT_josh_bottle_caps_l1771_177170


namespace NUMINAMATH_GPT_geometric_sequence_formula_and_sum_l1771_177161

theorem geometric_sequence_formula_and_sum (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (h1 : a 1 = 2) 
  (h2 : ∀ n, a (n+1) = 2 * a n) 
  (h_arith : a 1 = 2 ∧ 2 * (a 3 + 1) = a 1 + a 4)
  (h_b : ∀ n, b n = Nat.log2 (a n)) :
  (∀ n, a n = 2 ^ n) ∧ (S n = (n * (n + 1)) / 2) := 
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_formula_and_sum_l1771_177161


namespace NUMINAMATH_GPT_jo_page_an_hour_ago_l1771_177169

variables (total_pages current_page hours_left : ℕ)
variables (steady_reading_rate : ℕ)
variables (page_an_hour_ago : ℕ)

-- Conditions
def conditions := 
  steady_reading_rate * hours_left = total_pages - current_page ∧
  total_pages = 210 ∧
  current_page = 90 ∧
  hours_left = 4 ∧
  page_an_hour_ago = current_page - steady_reading_rate

-- Theorem to prove that Jo was on page 60 an hour ago
theorem jo_page_an_hour_ago (h : conditions total_pages current_page hours_left steady_reading_rate page_an_hour_ago) : 
  page_an_hour_ago = 60 :=
sorry

end NUMINAMATH_GPT_jo_page_an_hour_ago_l1771_177169


namespace NUMINAMATH_GPT_min_value_of_expression_l1771_177118

theorem min_value_of_expression (x y : ℝ) : (2 * x * y - 3) ^ 2 + (x - y) ^ 2 ≥ 1 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1771_177118


namespace NUMINAMATH_GPT_percent_of_x_l1771_177168

theorem percent_of_x
  (x y z : ℝ)
  (h1 : 0.45 * z = 1.20 * y)
  (h2 : z = 2 * x) :
  y = 0.75 * x :=
sorry

end NUMINAMATH_GPT_percent_of_x_l1771_177168


namespace NUMINAMATH_GPT_average_of_seven_consecutive_l1771_177138

theorem average_of_seven_consecutive (
  a : ℤ 
  ) (c : ℤ) 
  (h1 : c = (a + 1 + a + 2 + a + 3 + a + 4 + a + 5 + a + 6 + a + 7) / 7) : 
  (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7 = a + 7 := 
by 
  sorry

end NUMINAMATH_GPT_average_of_seven_consecutive_l1771_177138


namespace NUMINAMATH_GPT_g_recursion_relation_l1771_177149

noncomputable def g (n : ℕ) : ℝ :=
  (3 + 2 * Real.sqrt 3) / 6 * ((2 + Real.sqrt 3) / 2)^n +
  (3 - 2 * Real.sqrt 3) / 6 * ((2 - Real.sqrt 3) / 2)^n

theorem g_recursion_relation (n : ℕ) : g (n + 1) - 2 * g n + g (n - 1) = 0 :=
  sorry

end NUMINAMATH_GPT_g_recursion_relation_l1771_177149


namespace NUMINAMATH_GPT_correct_coefficient_l1771_177191

-- Definitions based on given conditions
def isMonomial (expr : String) : Prop := true

def coefficient (expr : String) : ℚ :=
  if expr = "-a/3" then -1/3 else 0

-- Statement to prove
theorem correct_coefficient : coefficient "-a/3" = -1/3 :=
by
  sorry

end NUMINAMATH_GPT_correct_coefficient_l1771_177191


namespace NUMINAMATH_GPT_minimal_side_length_of_room_l1771_177126

theorem minimal_side_length_of_room (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ S : ℕ, S = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_minimal_side_length_of_room_l1771_177126


namespace NUMINAMATH_GPT_john_pays_more_than_jane_l1771_177122

noncomputable def original_price : ℝ := 34.00
noncomputable def discount : ℝ := 0.10
noncomputable def tip_percent : ℝ := 0.15

noncomputable def discounted_price : ℝ := original_price - (discount * original_price)
noncomputable def john_tip : ℝ := tip_percent * original_price
noncomputable def john_total : ℝ := discounted_price + john_tip
noncomputable def jane_tip : ℝ := tip_percent * discounted_price
noncomputable def jane_total : ℝ := discounted_price + jane_tip

theorem john_pays_more_than_jane : john_total - jane_total = 0.51 := by
  sorry

end NUMINAMATH_GPT_john_pays_more_than_jane_l1771_177122


namespace NUMINAMATH_GPT_program_output_for_six_l1771_177199

-- Define the factorial function
def factorial : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

-- The theorem we want to prove
theorem program_output_for_six : factorial 6 = 720 := by
  sorry

end NUMINAMATH_GPT_program_output_for_six_l1771_177199


namespace NUMINAMATH_GPT_at_least_one_false_l1771_177187

theorem at_least_one_false (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : ¬ ((a + b < c + d) ∧ ((a + b) * (c + d) < a * b + c * d) ∧ ((a + b) * c * d < a * b * (c + d))) :=
  by
  sorry

end NUMINAMATH_GPT_at_least_one_false_l1771_177187


namespace NUMINAMATH_GPT_swimming_speed_l1771_177143

theorem swimming_speed (v : ℝ) (water_speed : ℝ) (distance : ℝ) (time : ℝ) 
  (h1 : water_speed = 2) 
  (h2 : distance = 14) 
  (h3 : time = 3.5) 
  (h4 : distance = (v - water_speed) * time) : 
  v = 6 := 
by
  sorry

end NUMINAMATH_GPT_swimming_speed_l1771_177143


namespace NUMINAMATH_GPT_harry_terry_difference_l1771_177144

theorem harry_terry_difference :
  let H := 12 - (3 * 4)
  let T := 12 - (3 * 4) -- Correcting Terry's mistake
  H - T = 0 := by
  sorry

end NUMINAMATH_GPT_harry_terry_difference_l1771_177144


namespace NUMINAMATH_GPT_simplify_expression_l1771_177196

variable (m n : ℝ)

theorem simplify_expression : -2 * (m - n) = -2 * m + 2 * n := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1771_177196


namespace NUMINAMATH_GPT_problem_statement_l1771_177174

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 2) :
  (1 < b ∧ b < 2) ∧ (ab < 1) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1771_177174


namespace NUMINAMATH_GPT_evaluate_polynomial_103_l1771_177150

theorem evaluate_polynomial_103 :
  103 ^ 4 - 4 * 103 ^ 3 + 6 * 103 ^ 2 - 4 * 103 + 1 = 108243216 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_103_l1771_177150


namespace NUMINAMATH_GPT_rationalize_denominator_correct_l1771_177154

noncomputable def rationalize_denominator : ℚ :=
  let A := 5
  let B := 49
  let C := 21
  A + B + C

theorem rationalize_denominator_correct :
  (let A := 5
   let B := 49
   let C := 21
   (A + B + C) = 75) :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_correct_l1771_177154


namespace NUMINAMATH_GPT_min_cos_for_sqrt_l1771_177151

theorem min_cos_for_sqrt (x : ℝ) (h : 2 * Real.cos x - 1 ≥ 0) : Real.cos x ≥ 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_min_cos_for_sqrt_l1771_177151


namespace NUMINAMATH_GPT_trig_identity_l1771_177111

theorem trig_identity : (Real.cos (15 * Real.pi / 180))^2 - (Real.sin (15 * Real.pi / 180))^2 = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1771_177111


namespace NUMINAMATH_GPT_fraction_students_above_eight_l1771_177186

theorem fraction_students_above_eight (total_students S₈ : ℕ) (below_eight_percent : ℝ)
    (num_below_eight : total_students * below_eight_percent = 10) 
    (total_equals : total_students = 50) 
    (students_eight : S₈ = 24) :
    (total_students - (total_students * below_eight_percent + S₈)) / S₈ = 2 / 3 := 
by 
  -- Solution steps can go here 
  sorry

end NUMINAMATH_GPT_fraction_students_above_eight_l1771_177186


namespace NUMINAMATH_GPT_freddy_call_duration_l1771_177103

theorem freddy_call_duration (total_cost : ℕ) (local_cost_per_minute : ℕ) (international_cost_per_minute : ℕ) (local_duration : ℕ)
  (total_cost_eq : total_cost = 1000) -- cost in cents
  (local_cost_eq : local_cost_per_minute = 5)
  (international_cost_eq : international_cost_per_minute = 25)
  (local_duration_eq : local_duration = 45) :
  (total_cost - local_duration * local_cost_per_minute) / international_cost_per_minute = 31 :=
by
  sorry

end NUMINAMATH_GPT_freddy_call_duration_l1771_177103


namespace NUMINAMATH_GPT_evaluate_expression_l1771_177157

theorem evaluate_expression : 6 - 8 * (9 - 4^2) * 5 = 286 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1771_177157


namespace NUMINAMATH_GPT_possible_rectangular_arrays_l1771_177124

theorem possible_rectangular_arrays (n : ℕ) (h : n = 48) :
  ∃ (m k : ℕ), m * k = n ∧ 2 ≤ m ∧ 2 ≤ k :=
sorry

end NUMINAMATH_GPT_possible_rectangular_arrays_l1771_177124


namespace NUMINAMATH_GPT_largest_minus_smallest_l1771_177197

-- Define the given conditions
def A : ℕ := 10 * 2 + 9
def B : ℕ := A - 16
def C : ℕ := B * 3

-- Statement to prove
theorem largest_minus_smallest : C - B = 26 := by
  sorry

end NUMINAMATH_GPT_largest_minus_smallest_l1771_177197


namespace NUMINAMATH_GPT_coefficient_of_x3y0_l1771_177140

def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

def f (m n : ℕ) : ℕ :=
  binomial_coeff 6 m * binomial_coeff 4 n

theorem coefficient_of_x3y0 :
  f 3 0 = 20 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_of_x3y0_l1771_177140


namespace NUMINAMATH_GPT_molecular_weight_one_mole_of_AlOH3_l1771_177165

variable (MW_7_moles : ℕ) (MW : ℕ)

theorem molecular_weight_one_mole_of_AlOH3 (h : MW_7_moles = 546) : MW = 78 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_one_mole_of_AlOH3_l1771_177165


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1771_177193

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2 - 1) : 
  (1 - 1 / (a + 1)) * ((a^2 + 2 * a + 1) / a) = Real.sqrt 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_simplify_and_evaluate_l1771_177193


namespace NUMINAMATH_GPT_problem_statement_l1771_177160

variable {x y : ℤ}

def is_multiple_of_5 (n : ℤ) : Prop := ∃ m : ℤ, n = 5 * m
def is_multiple_of_10 (n : ℤ) : Prop := ∃ m : ℤ, n = 10 * m

theorem problem_statement (hx : is_multiple_of_5 x) (hy : is_multiple_of_10 y) :
  (is_multiple_of_5 (x + y)) ∧ (x + y ≥ 15) :=
sorry

end NUMINAMATH_GPT_problem_statement_l1771_177160


namespace NUMINAMATH_GPT_triangle_inequality_range_x_l1771_177102

theorem triangle_inequality_range_x (x : ℝ) :
  let a := 3;
  let b := 8;
  let c := 1 + 2 * x;
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ↔ (2 < x ∧ x < 5) :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_range_x_l1771_177102


namespace NUMINAMATH_GPT_second_concert_attendance_l1771_177106

def first_concert_attendance : ℕ := 65899
def additional_people : ℕ := 119

theorem second_concert_attendance : first_concert_attendance + additional_people = 66018 := 
by 
  -- Proof is not discussed here, only the statement is required.
sorry

end NUMINAMATH_GPT_second_concert_attendance_l1771_177106


namespace NUMINAMATH_GPT_smallest_value_of_N_l1771_177158

theorem smallest_value_of_N (l m n : ℕ) (N : ℕ) (h1 : (l-1) * (m-1) * (n-1) = 270) (h2 : N = l * m * n): 
  N = 420 :=
sorry

end NUMINAMATH_GPT_smallest_value_of_N_l1771_177158


namespace NUMINAMATH_GPT_motorbike_speed_l1771_177105

noncomputable def speed_of_motorbike 
  (V_train : ℝ) 
  (t_overtake : ℝ) 
  (train_length_m : ℝ) : ℝ :=
  V_train - (train_length_m / 1000) * (3600 / t_overtake)

theorem motorbike_speed : 
  speed_of_motorbike 100 80 800.064 = 63.99712 :=
by
  -- this is where the proof steps would go
  sorry

end NUMINAMATH_GPT_motorbike_speed_l1771_177105


namespace NUMINAMATH_GPT_initial_people_in_castle_l1771_177181

theorem initial_people_in_castle (P : ℕ) (provisions : ℕ → ℕ → ℕ) :
  (provisions P 90) - (provisions P 30) = provisions (P - 100) 90 ↔ P = 300 :=
by
  sorry

end NUMINAMATH_GPT_initial_people_in_castle_l1771_177181


namespace NUMINAMATH_GPT_percentage_of_acid_in_original_mixture_l1771_177172

theorem percentage_of_acid_in_original_mixture
  (a w : ℚ)
  (h1 : a / (a + w + 2) = 18 / 100)
  (h2 : (a + 2) / (a + w + 4) = 30 / 100) :
  (a / (a + w)) * 100 = 29 := 
sorry

end NUMINAMATH_GPT_percentage_of_acid_in_original_mixture_l1771_177172


namespace NUMINAMATH_GPT_max_sum_abc_l1771_177100

theorem max_sum_abc (a b c : ℝ) (h : a + b + c = a^2 + b^2 + c^2) : a + b + c ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_sum_abc_l1771_177100


namespace NUMINAMATH_GPT_michael_will_meet_two_times_l1771_177167

noncomputable def michael_meetings : ℕ :=
  let michael_speed := 6 -- feet per second
  let pail_distance := 300 -- feet
  let truck_speed := 12 -- feet per second
  let truck_stop_time := 20 -- seconds
  let initial_distance := pail_distance -- feet
  let michael_position (t: ℕ) := michael_speed * t
  let truck_position (cycle: ℕ) := pail_distance * cycle
  let truck_cycle_time := pail_distance / truck_speed + truck_stop_time -- seconds per cycle
  let truck_position_at_time (t: ℕ) := 
    let cycle := t / truck_cycle_time
    let remaining_time := t % truck_cycle_time
    if remaining_time < (pail_distance / truck_speed) then 
      truck_position cycle + truck_speed * remaining_time
    else 
      truck_position cycle + pail_distance
  let distance_between := 
    λ (t: ℕ) => truck_position_at_time t - michael_position t
  let meet_time := 
    λ (t: ℕ) => if distance_between t = 0 then 1 else 0
  let total_meetings := 
    (List.range 300).map meet_time -- estimating within 300 seconds
    |> List.sum
  total_meetings

theorem michael_will_meet_two_times : michael_meetings = 2 :=
  sorry

end NUMINAMATH_GPT_michael_will_meet_two_times_l1771_177167


namespace NUMINAMATH_GPT_derivative_of_gx_eq_3x2_l1771_177156

theorem derivative_of_gx_eq_3x2 (f : ℝ → ℝ) : (∀ x : ℝ, f x = (x + 1) * (x^2 - x + 1)) → (∀ x : ℝ, deriv f x = 3 * x^2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_derivative_of_gx_eq_3x2_l1771_177156


namespace NUMINAMATH_GPT_right_triangle_leg_length_l1771_177112

theorem right_triangle_leg_length (a b c : ℕ) (h₁ : a = 8) (h₂ : c = 17) (h₃ : a^2 + b^2 = c^2) : b = 15 := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_leg_length_l1771_177112


namespace NUMINAMATH_GPT_min_value_of_f_l1771_177132

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + Real.sin (2 * x)

theorem min_value_of_f : 
  ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y :=
sorry

end NUMINAMATH_GPT_min_value_of_f_l1771_177132


namespace NUMINAMATH_GPT_range_of_x_l1771_177155

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) + 1

theorem range_of_x (x : ℝ) (h : f (2 * x - 1) + f (4 - x^2) > 2) : x ∈ Set.Ioo (-1 : ℝ) 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l1771_177155


namespace NUMINAMATH_GPT_probability_even_sum_is_correct_l1771_177173

noncomputable def probability_even_sum : ℚ :=
  let p_even_first := (2 : ℚ) / 5
  let p_odd_first := (3 : ℚ) / 5
  let p_even_second := (1 : ℚ) / 4
  let p_odd_second := (3 : ℚ) / 4

  let p_both_even := p_even_first * p_even_second
  let p_both_odd := p_odd_first * p_odd_second

  p_both_even + p_both_odd

theorem probability_even_sum_is_correct : probability_even_sum = 11 / 20 := by
  sorry

end NUMINAMATH_GPT_probability_even_sum_is_correct_l1771_177173


namespace NUMINAMATH_GPT_value_of_expression_l1771_177116

theorem value_of_expression : (20 * 24) / (2 * 0 + 2 * 4) = 60 := sorry

end NUMINAMATH_GPT_value_of_expression_l1771_177116


namespace NUMINAMATH_GPT_initial_reading_times_per_day_l1771_177189

-- Definitions based on the conditions

/-- Number of pages Jessy plans to read initially in each session is 6. -/
def session_pages : ℕ := 6

/-- Jessy needs to read 140 pages in one week. -/
def total_pages : ℕ := 140

/-- Jessy reads an additional 2 pages per day to achieve her goal. -/
def additional_daily_pages : ℕ := 2

/-- Days in a week -/
def days_in_week : ℕ := 7

-- Proving Jessy's initial plan for reading times per day
theorem initial_reading_times_per_day (x : ℕ) (h : days_in_week * (session_pages * x + additional_daily_pages) = total_pages) : 
    x = 3 := by
  -- skipping the proof itself
  sorry

end NUMINAMATH_GPT_initial_reading_times_per_day_l1771_177189


namespace NUMINAMATH_GPT_sales_tax_per_tire_l1771_177194

def cost_per_tire : ℝ := 7
def number_of_tires : ℕ := 4
def final_total_cost : ℝ := 30

theorem sales_tax_per_tire :
  (final_total_cost - number_of_tires * cost_per_tire) / number_of_tires = 0.5 :=
sorry

end NUMINAMATH_GPT_sales_tax_per_tire_l1771_177194


namespace NUMINAMATH_GPT_total_crayons_l1771_177175

-- We're given the conditions
def crayons_per_child : ℕ := 6
def number_of_children : ℕ := 12

-- We need to prove the total number of crayons.
theorem total_crayons (c : ℕ := crayons_per_child) (n : ℕ := number_of_children) : (c * n) = 72 := by
  sorry

end NUMINAMATH_GPT_total_crayons_l1771_177175


namespace NUMINAMATH_GPT_value_of_k_l1771_177192

open Nat

def perm (n r : ℕ) : ℕ := factorial n / factorial (n - r)
def comb (n r : ℕ) : ℕ := factorial n / (factorial r * factorial (n - r))

theorem value_of_k : ∃ k : ℕ, perm 32 6 = k * comb 32 6 ∧ k = 720 := by
  use 720
  unfold perm comb
  sorry

end NUMINAMATH_GPT_value_of_k_l1771_177192


namespace NUMINAMATH_GPT_path_inequality_l1771_177184

theorem path_inequality
  (f : ℕ → ℕ → ℝ) :
  f 1 6 * f 2 5 * f 3 4 + f 1 5 * f 2 4 * f 3 6 + f 1 4 * f 2 6 * f 3 5 ≥
  f 1 6 * f 2 4 * f 3 5 + f 1 5 * f 2 6 * f 3 4 + f 1 4 * f 2 5 * f 3 6 :=
sorry

end NUMINAMATH_GPT_path_inequality_l1771_177184


namespace NUMINAMATH_GPT_water_percentage_l1771_177139

theorem water_percentage (P : ℕ) : 
  let initial_volume := 300
  let final_volume := initial_volume + 100
  let desired_water_percentage := 70
  let water_added := 100
  let final_water_amount := desired_water_percentage * final_volume / 100
  let current_water_amount := P * initial_volume / 100

  current_water_amount + water_added = final_water_amount → 
  P = 60 :=
by sorry

end NUMINAMATH_GPT_water_percentage_l1771_177139


namespace NUMINAMATH_GPT_rectangle_to_cylinder_max_volume_ratio_l1771_177114

/-- Given a rectangle with a perimeter of 12 and converting it into a cylinder 
with the height being the same as the width of the rectangle, prove that the 
ratio of the circumference of the cylinder's base to its height when the volume 
is maximized is 2:1. -/
theorem rectangle_to_cylinder_max_volume_ratio : 
  ∃ (x : ℝ), (2 * x + 2 * (6 - x)) = 12 → 2 * (6 - x) / x = 2 :=
sorry

end NUMINAMATH_GPT_rectangle_to_cylinder_max_volume_ratio_l1771_177114


namespace NUMINAMATH_GPT_distance_BC_in_circle_l1771_177183

theorem distance_BC_in_circle
    (r : ℝ) (A B C : ℝ × ℝ)
    (h_radius : r = 10)
    (h_diameter : dist A B = 2 * r)
    (h_chord : dist A C = 12) :
    dist B C = 16 := by
  sorry

end NUMINAMATH_GPT_distance_BC_in_circle_l1771_177183


namespace NUMINAMATH_GPT_total_visible_legs_l1771_177107

-- Defining the conditions
def num_crows : ℕ := 4
def num_pigeons : ℕ := 3
def num_flamingos : ℕ := 5
def num_sparrows : ℕ := 8

def legs_per_crow : ℕ := 2
def legs_per_pigeon : ℕ := 2
def legs_per_flamingo : ℕ := 3
def legs_per_sparrow : ℕ := 2

-- Formulating the theorem that we need to prove
theorem total_visible_legs :
  (num_crows * legs_per_crow) +
  (num_pigeons * legs_per_pigeon) +
  (num_flamingos * legs_per_flamingo) +
  (num_sparrows * legs_per_sparrow) = 45 := by sorry

end NUMINAMATH_GPT_total_visible_legs_l1771_177107


namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_l1771_177109

theorem eccentricity_of_hyperbola (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_asymp : 3 * a + b = 0) :
    let c := Real.sqrt (a^2 + b^2)
    let e := c / a
    e = Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_eccentricity_of_hyperbola_l1771_177109


namespace NUMINAMATH_GPT_smallest_number_exceeding_triangle_perimeter_l1771_177120

theorem smallest_number_exceeding_triangle_perimeter (a b : ℕ) (a_eq_7 : a = 7) (b_eq_21 : b = 21) :
  ∃ P : ℕ, (∀ c : ℝ, 14 < c ∧ c < 28 → a + b + c < P) ∧ P = 56 := by
  sorry

end NUMINAMATH_GPT_smallest_number_exceeding_triangle_perimeter_l1771_177120


namespace NUMINAMATH_GPT_cube_red_face_probability_l1771_177159

theorem cube_red_face_probability :
  let faces_total := 6
  let red_faces := 3
  let probability_red := red_faces / faces_total
  probability_red = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cube_red_face_probability_l1771_177159


namespace NUMINAMATH_GPT_simplify_sqrt_expression_l1771_177117

theorem simplify_sqrt_expression :
  ( (Real.sqrt 112 + Real.sqrt 567) / Real.sqrt 175) = 13 / 5 := by
  -- conditions for simplification
  have h1 : Real.sqrt 112 = 4 * Real.sqrt 7 := sorry
  have h2 : Real.sqrt 567 = 9 * Real.sqrt 7 := sorry
  have h3 : Real.sqrt 175 = 5 * Real.sqrt 7 := sorry
  
  -- Use the conditions to simplify the expression
  rw [h1, h2, h3]
  -- Further simplification to achieve the result 13 / 5
  sorry

end NUMINAMATH_GPT_simplify_sqrt_expression_l1771_177117


namespace NUMINAMATH_GPT_sea_lions_at_zoo_l1771_177171

def ratio_sea_lions_to_penguins (S P : ℕ) : Prop := P = 11 * S / 4
def ratio_sea_lions_to_flamingos (S F : ℕ) : Prop := F = 7 * S / 4
def penguins_more_sea_lions (S P : ℕ) : Prop := P = S + 84
def flamingos_more_penguins (P F : ℕ) : Prop := F = P + 42

theorem sea_lions_at_zoo (S P F : ℕ)
  (h1 : ratio_sea_lions_to_penguins S P)
  (h2 : ratio_sea_lions_to_flamingos S F)
  (h3 : penguins_more_sea_lions S P)
  (h4 : flamingos_more_penguins P F) :
  S = 42 :=
sorry

end NUMINAMATH_GPT_sea_lions_at_zoo_l1771_177171


namespace NUMINAMATH_GPT_simplify_trig_identity_l1771_177198

theorem simplify_trig_identity (α : ℝ) :
  (Real.cos (Real.pi / 3 + α) + Real.sin (Real.pi / 6 + α)) = Real.cos α :=
by
  sorry

end NUMINAMATH_GPT_simplify_trig_identity_l1771_177198


namespace NUMINAMATH_GPT_paper_thickness_after_folding_five_times_l1771_177129

-- Definitions of initial conditions
def initial_thickness : ℝ := 0.1
def num_folds : ℕ := 5

-- Target thickness after folding
def final_thickness (init_thickness : ℝ) (folds : ℕ) : ℝ :=
  (2 ^ folds) * init_thickness

-- Statement of the theorem
theorem paper_thickness_after_folding_five_times :
  final_thickness initial_thickness num_folds = 3.2 :=
by
  -- The proof (the implementation is replaced with sorry)
  sorry

end NUMINAMATH_GPT_paper_thickness_after_folding_five_times_l1771_177129


namespace NUMINAMATH_GPT_events_equally_likely_iff_N_eq_18_l1771_177108

variable (N : ℕ)

-- Define the number of combinations in the draws
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the sums of selecting balls
noncomputable def S_63_10 (N: ℕ) : ℕ := sorry -- placeholder definition
noncomputable def S_44_8 (N: ℕ) : ℕ := sorry 

-- Condition for the events being equally likely
theorem events_equally_likely_iff_N_eq_18 : 
  (S_63_10 N * C N 8 = S_44_8 N * C N 10) ↔ N = 18 :=
sorry

end NUMINAMATH_GPT_events_equally_likely_iff_N_eq_18_l1771_177108


namespace NUMINAMATH_GPT_campers_rowing_afternoon_l1771_177148

theorem campers_rowing_afternoon (morning_rowing morning_hiking total : ℕ) 
  (h1 : morning_rowing = 41) 
  (h2 : morning_hiking = 4) 
  (h3 : total = 71) : 
  total - (morning_rowing + morning_hiking) = 26 :=
by
  sorry

end NUMINAMATH_GPT_campers_rowing_afternoon_l1771_177148


namespace NUMINAMATH_GPT_chessboard_colorings_l1771_177110

-- Definitions based on conditions
def valid_chessboard_colorings_count : ℕ :=
  2 ^ 33

-- Theorem statement with the question, conditions, and the correct answer
theorem chessboard_colorings : 
  valid_chessboard_colorings_count = 2 ^ 33 := by
  sorry

end NUMINAMATH_GPT_chessboard_colorings_l1771_177110


namespace NUMINAMATH_GPT_smallest_positive_period_f_max_value_f_interval_min_value_f_interval_l1771_177179

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x - Real.cos x) + 1

theorem smallest_positive_period_f : ∃ k > 0, ∀ x, f (x + k) = f x := 
sorry

theorem max_value_f_interval : ∃ x ∈ Set.Icc (Real.pi / 8) (3 * Real.pi / 4), f x = Real.sqrt 2 :=
sorry

theorem min_value_f_interval : ∃ x ∈ Set.Icc (Real.pi / 8) (3 * Real.pi / 4), f x = -1 :=
sorry

end NUMINAMATH_GPT_smallest_positive_period_f_max_value_f_interval_min_value_f_interval_l1771_177179


namespace NUMINAMATH_GPT_intersection_A_B_l1771_177178

-- Define set A
def A : Set ℝ := { y | ∃ x : ℝ, y = Real.log x }

-- Define set B
def B : Set ℝ := { x | ∃ y : ℝ, y = Real.sqrt x }

-- Prove that the intersection of sets A and B is [0, +∞)
theorem intersection_A_B : A ∩ B = { x | 0 ≤ x } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1771_177178


namespace NUMINAMATH_GPT_arithmetic_sequence_value_l1771_177137

theorem arithmetic_sequence_value (a : ℝ) 
  (h1 : 2 * (2 * a + 1) = (a - 1) + (a + 4)) : a = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_value_l1771_177137


namespace NUMINAMATH_GPT_quadratic_variation_y_l1771_177119

theorem quadratic_variation_y (k : ℝ) (x y : ℝ) (h1 : y = k * x^2) (h2 : (25 : ℝ) = k * (5 : ℝ)^2) :
  y = 25 :=
by
sorry

end NUMINAMATH_GPT_quadratic_variation_y_l1771_177119


namespace NUMINAMATH_GPT_tan_product_eq_three_l1771_177104

noncomputable def tan_pi_over_9 : ℝ := Real.tan (Real.pi / 9)
noncomputable def tan_2pi_over_9 : ℝ := Real.tan (2 * Real.pi / 9)
noncomputable def tan_4pi_over_9 : ℝ := Real.tan (4 * Real.pi / 9)

theorem tan_product_eq_three : tan_pi_over_9 * tan_2pi_over_9 * tan_4pi_over_9 = 3 := by
    sorry

end NUMINAMATH_GPT_tan_product_eq_three_l1771_177104


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_l1771_177134

theorem sum_of_roots_of_quadratic (m n : ℝ) (h1 : m = 2 * n) (h2 : ∀ x : ℝ, x ^ 2 + m * x + n = 0) :
    m + n = 3 / 2 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_l1771_177134


namespace NUMINAMATH_GPT_real_part_implies_value_of_a_l1771_177166

theorem real_part_implies_value_of_a (a b : ℝ) (h : a = 2 * b) (hb : b = 1) : a = 2 := by
  sorry

end NUMINAMATH_GPT_real_part_implies_value_of_a_l1771_177166


namespace NUMINAMATH_GPT_intersect_at_one_point_l1771_177180

theorem intersect_at_one_point (a : ℝ) : 
  (a * (4 * 4) + 4 * 4 * 6 = 0) -> a = 2 / (3: ℝ) :=
by sorry

end NUMINAMATH_GPT_intersect_at_one_point_l1771_177180


namespace NUMINAMATH_GPT_sum_of_x_and_y_l1771_177131

theorem sum_of_x_and_y (x y : ℝ) 
  (h1 : (x - 1) ^ 3 + 1997 * (x - 1) = -1)
  (h2 : (y - 1) ^ 3 + 1997 * (y - 1) = 1) : 
  x + y = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l1771_177131


namespace NUMINAMATH_GPT_sqrt_2023_irrational_l1771_177153

theorem sqrt_2023_irrational : ¬ ∃ (r : ℚ), r^2 = 2023 := by
  sorry

end NUMINAMATH_GPT_sqrt_2023_irrational_l1771_177153


namespace NUMINAMATH_GPT_smallest_7_heavy_three_digit_number_l1771_177188

def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem smallest_7_heavy_three_digit_number :
  ∃ n : ℕ, is_three_digit n ∧ is_7_heavy n ∧ (∀ m : ℕ, is_three_digit m ∧ is_7_heavy m → n ≤ m) ∧
  n = 103 := 
by
  sorry

end NUMINAMATH_GPT_smallest_7_heavy_three_digit_number_l1771_177188


namespace NUMINAMATH_GPT_percentage_cross_pollinated_l1771_177145

-- Definitions and known conditions:
variables (F C T : ℕ)
variables (h1 : F + C = 221)
variables (h2 : F = 3 * T / 4)
variables (h3 : T = F + 39 + C)

-- Theorem statement for the percentage of cross-pollinated trees
theorem percentage_cross_pollinated : ((C : ℚ) / T) * 100 = 10 :=
by sorry

end NUMINAMATH_GPT_percentage_cross_pollinated_l1771_177145


namespace NUMINAMATH_GPT_units_digit_G_1000_l1771_177136

def G (n : ℕ) : ℕ := 3 ^ (3 ^ n) + 1

theorem units_digit_G_1000 : G 1000 % 10 = 2 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_G_1000_l1771_177136


namespace NUMINAMATH_GPT_Tamara_is_95_inches_l1771_177146

/- Defining the basic entities: Kim's height (K), Tamara's height, Gavin's height -/
def Kim_height (K : ℝ) := K
def Tamara_height (K : ℝ) := 3 * K - 4
def Gavin_height (K : ℝ) := 2 * K + 6

/- Combined height equation -/
def combined_height (K : ℝ) := (Tamara_height K) + (Kim_height K) + (Gavin_height K) = 200

/- Given that Kim's height satisfies the combined height condition,
   proving that Tamara's height is 95 inches -/
theorem Tamara_is_95_inches (K : ℝ) (h : combined_height K) : Tamara_height K = 95 :=
by
  sorry

end NUMINAMATH_GPT_Tamara_is_95_inches_l1771_177146


namespace NUMINAMATH_GPT_taxi_ride_cost_l1771_177147

theorem taxi_ride_cost (initial_cost : ℝ) (cost_first_3_miles : ℝ) (rate_first_3_miles : ℝ) (rate_after_3_miles : ℝ) (total_miles : ℝ) (remaining_miles : ℝ) :
  initial_cost = 2.00 ∧ rate_first_3_miles = 0.30 ∧ rate_after_3_miles = 0.40 ∧ total_miles = 8 ∧ total_miles - 3 = remaining_miles →
  initial_cost + 3 * rate_first_3_miles + remaining_miles * rate_after_3_miles = 4.90 :=
sorry

end NUMINAMATH_GPT_taxi_ride_cost_l1771_177147


namespace NUMINAMATH_GPT_divisible_by_11_l1771_177152

theorem divisible_by_11 (n : ℤ) : (11 ∣ (n^2001 - n^4)) ↔ (n % 11 = 0 ∨ n % 11 = 1) :=
by
  sorry

end NUMINAMATH_GPT_divisible_by_11_l1771_177152


namespace NUMINAMATH_GPT_percent_value_quarters_l1771_177135

noncomputable def value_in_cents (dimes quarters nickels : ℕ) : ℕ := 
  (dimes * 10) + (quarters * 25) + (nickels * 5)

noncomputable def percent_in_quarters (quarters total_value : ℕ) : ℚ := 
  (quarters * 25 : ℚ) / total_value * 100

theorem percent_value_quarters 
  (h_dimes : ℕ := 80) 
  (h_quarters : ℕ := 30) 
  (h_nickels : ℕ := 40) 
  (h_total_value := value_in_cents h_dimes h_quarters h_nickels) : 
  percent_in_quarters h_quarters h_total_value = 42.86 :=
by sorry

end NUMINAMATH_GPT_percent_value_quarters_l1771_177135


namespace NUMINAMATH_GPT_curve_crosses_itself_l1771_177163

-- Definitions of the parametric equations
def x (t k : ℝ) : ℝ := t^2 + k
def y (t k : ℝ) : ℝ := t^3 - k * t + 5

-- The main theorem statement
theorem curve_crosses_itself (k : ℝ) (ha : ℝ) (hb : ℝ) :
  ha ≠ hb →
  x ha k = x hb k →
  y ha k = y hb k →
  k = 9 ∧ x ha k = 18 ∧ y ha k = 5 :=
by
  sorry

end NUMINAMATH_GPT_curve_crosses_itself_l1771_177163


namespace NUMINAMATH_GPT_find_sum_xyz_l1771_177128

-- Define the problem
def system_of_equations (x y z : ℝ) : Prop :=
  x^2 + x * y + y^2 = 27 ∧
  y^2 + y * z + z^2 = 9 ∧
  z^2 + z * x + x^2 = 36

-- The main theorem to be proved
theorem find_sum_xyz (x y z : ℝ) (h : system_of_equations x y z) : 
  x * y + y * z + z * x = 18 :=
sorry

end NUMINAMATH_GPT_find_sum_xyz_l1771_177128


namespace NUMINAMATH_GPT_cylinder_ratio_max_volume_l1771_177177

theorem cylinder_ratio_max_volume 
    (l w : ℝ) 
    (r : ℝ) 
    (h : ℝ)
    (H_perimeter : 2 * l + 2 * w = 12)
    (H_length_circumference : l = 2 * π * r)
    (H_width_height : w = h) :
    (∀ V : ℝ, V = π * r^2 * h) →
    (∀ r : ℝ, r = 2 / π) →
    ((2 * π * r) / h = 2) :=
sorry

end NUMINAMATH_GPT_cylinder_ratio_max_volume_l1771_177177


namespace NUMINAMATH_GPT_sqrt_xyz_ge_sqrt_x_add_sqrt_y_add_sqrt_z_l1771_177142

open Real

theorem sqrt_xyz_ge_sqrt_x_add_sqrt_y_add_sqrt_z (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z ≥ x * y + y * z + z * x) :
  sqrt (x * y * z) ≥ sqrt x + sqrt y + sqrt z :=
by
  sorry

end NUMINAMATH_GPT_sqrt_xyz_ge_sqrt_x_add_sqrt_y_add_sqrt_z_l1771_177142


namespace NUMINAMATH_GPT_quadratics_root_k_value_l1771_177190

theorem quadratics_root_k_value :
  (∀ k : ℝ, (∀ x : ℝ, x^2 + k * x + 6 = 0 → (x = 2 ∨ ∃ x1 : ℝ, x1 * 2 = 6 ∧ x1 + 2 = k)) → 
  (x = 2 → ∃ x1 : ℝ, x1 = 3 ∧ k = -5)) := 
sorry

end NUMINAMATH_GPT_quadratics_root_k_value_l1771_177190


namespace NUMINAMATH_GPT_units_digit_of_power_ends_in_nine_l1771_177101

theorem units_digit_of_power_ends_in_nine (n : ℕ) (h : (3^n) % 10 = 9) : n % 4 = 2 :=
sorry

end NUMINAMATH_GPT_units_digit_of_power_ends_in_nine_l1771_177101


namespace NUMINAMATH_GPT_stratified_sampling_medium_supermarkets_l1771_177133

theorem stratified_sampling_medium_supermarkets
  (large_supermarkets : ℕ)
  (medium_supermarkets : ℕ)
  (small_supermarkets : ℕ)
  (sample_size : ℕ)
  (total_supermarkets : ℕ)
  (medium_proportion : ℚ) :
  large_supermarkets = 200 →
  medium_supermarkets = 400 →
  small_supermarkets = 1400 →
  sample_size = 100 →
  total_supermarkets = large_supermarkets + medium_supermarkets + small_supermarkets →
  medium_proportion = (medium_supermarkets : ℚ) / (total_supermarkets : ℚ) →
  medium_supermarkets_to_sample = sample_size * medium_proportion →
  medium_supermarkets_to_sample = 20 :=
sorry

end NUMINAMATH_GPT_stratified_sampling_medium_supermarkets_l1771_177133


namespace NUMINAMATH_GPT_dan_picked_more_apples_l1771_177195

-- Define the number of apples picked by Benny and Dan
def apples_picked_by_benny := 2
def apples_picked_by_dan := 9

-- Lean statement to prove the given condition
theorem dan_picked_more_apples :
  apples_picked_by_dan - apples_picked_by_benny = 7 := 
sorry

end NUMINAMATH_GPT_dan_picked_more_apples_l1771_177195


namespace NUMINAMATH_GPT_train_length_l1771_177130

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (length_train : ℝ) 
  (h_speed : speed_kmph = 50)
  (h_time : time_sec = 18) 
  (h_length : length_train = 250) : 
  (speed_kmph * 1000 / 3600) * time_sec = length_train :=
by 
  rw [h_speed, h_time, h_length]
  sorry

end NUMINAMATH_GPT_train_length_l1771_177130


namespace NUMINAMATH_GPT_sin_15_cos_15_l1771_177123

theorem sin_15_cos_15 : (Real.sin (15 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_sin_15_cos_15_l1771_177123


namespace NUMINAMATH_GPT_factor_expression_l1771_177185

theorem factor_expression (x y : ℝ) :
  66 * x^5 - 165 * x^9 + 99 * x^5 * y = 33 * x^5 * (2 - 5 * x^4 + 3 * y) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1771_177185
