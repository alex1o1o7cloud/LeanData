import Mathlib

namespace NUMINAMATH_GPT_value_of_x_l1202_120284

theorem value_of_x (x : ℝ) (h : 0.5 * x = 0.25 * 1500 - 30) : x = 690 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l1202_120284


namespace NUMINAMATH_GPT_ratio_girls_to_boys_l1202_120227

variable (g b : ℕ)

-- Conditions: total students are 30, six more girls than boys.
def total_students : Prop := g + b = 30
def six_more_girls : Prop := g = b + 6

-- Proof that the ratio of girls to boys is 3:2.
theorem ratio_girls_to_boys (ht : total_students g b) (hs : six_more_girls g b) : g / b = 3 / 2 :=
  sorry

end NUMINAMATH_GPT_ratio_girls_to_boys_l1202_120227


namespace NUMINAMATH_GPT_sum_of_consecutive_even_numbers_l1202_120277

theorem sum_of_consecutive_even_numbers (n : ℤ) 
  (h : n + 4 = 14) : n + (n + 2) + (n + 4) + (n + 6) = 52 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_even_numbers_l1202_120277


namespace NUMINAMATH_GPT_martin_discounted_tickets_l1202_120273

-- Definitions of the problem conditions
def total_tickets (F D : ℕ) := F + D = 10
def total_cost (F D : ℕ) := 2 * F + (16/10) * D = 184/10

-- Statement of the proof
theorem martin_discounted_tickets (F D : ℕ) (h1 : total_tickets F D) (h2 : total_cost F D) :
  D = 4 :=
sorry

end NUMINAMATH_GPT_martin_discounted_tickets_l1202_120273


namespace NUMINAMATH_GPT_collinear_points_d_value_l1202_120250

theorem collinear_points_d_value (a b c d : ℚ)
  (h1 : b = a)
  (h2 : c = -(a+1)/2)
  (collinear : (4 * d * (4 * a + 5) + a + 1 = 0)) :
  d = 9/20 :=
by {
  sorry
}

end NUMINAMATH_GPT_collinear_points_d_value_l1202_120250


namespace NUMINAMATH_GPT_wine_remaining_percentage_l1202_120294

theorem wine_remaining_percentage :
  let initial_wine := 250.0 -- initial wine in liters
  let daily_fraction := (249.0 / 250.0)
  let days := 50
  let remaining_wine := (daily_fraction ^ days) * initial_wine
  let percentage_remaining := (remaining_wine / initial_wine) * 100
  percentage_remaining = 81.846 :=
by
  sorry

end NUMINAMATH_GPT_wine_remaining_percentage_l1202_120294


namespace NUMINAMATH_GPT_perfect_square_condition_l1202_120256

theorem perfect_square_condition (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
    (gcd_xyz : Nat.gcd (Nat.gcd x y) z = 1)
    (hx_dvd : x ∣ y * z * (x + y + z))
    (hy_dvd : y ∣ x * z * (x + y + z))
    (hz_dvd : z ∣ x * y * (x + y + z))
    (sum_dvd : x + y + z ∣ x * y * z) :
  ∃ m : ℕ, m * m = x * y * z * (x + y + z) := sorry

end NUMINAMATH_GPT_perfect_square_condition_l1202_120256


namespace NUMINAMATH_GPT_overall_average_length_of_ropes_l1202_120210

theorem overall_average_length_of_ropes :
  let ropes := 6
  let third_part := ropes / 3
  let average1 := 70
  let average2 := 85
  let length1 := third_part * average1
  let length2 := (ropes - third_part) * average2
  let total_length := length1 + length2
  let overall_average := total_length / ropes
  overall_average = 80 := by
sorry

end NUMINAMATH_GPT_overall_average_length_of_ropes_l1202_120210


namespace NUMINAMATH_GPT_sequence_properties_l1202_120207

theorem sequence_properties (a : ℕ → ℕ) (S : ℕ → ℕ) :
  a 1 = 1 →
  (∀ n : ℕ, a (n + 1) + a n = 4 * n) →
  (∀ n : ℕ, a n = 2 * n - 1) ∧ (a 2023 = 4045) :=
by
  sorry

end NUMINAMATH_GPT_sequence_properties_l1202_120207


namespace NUMINAMATH_GPT_number_of_valid_n_l1202_120237

-- The definition for determining the number of positive integers n ≤ 2000 that can be represented as
-- floor(x) + floor(4x) + floor(5x) = n for some real number x.

noncomputable def count_valid_n : ℕ :=
  (200 : ℕ) * 3 + (200 : ℕ) * 2 + 1 + 1

theorem number_of_valid_n : count_valid_n = 802 :=
  sorry

end NUMINAMATH_GPT_number_of_valid_n_l1202_120237


namespace NUMINAMATH_GPT_incorrect_operation_l1202_120267

noncomputable def a : ℤ := -2

def operation_A (a : ℤ) : ℤ := abs a
def operation_B (a : ℤ) : ℤ := abs (a - 2) + abs (a + 1)
def operation_C (a : ℤ) : ℤ := -a ^ 3 + a + (-a) ^ 2
def operation_D (a : ℤ) : ℤ := abs a ^ 2

theorem incorrect_operation :
  operation_D a ≠ abs 4 :=
by
  sorry

end NUMINAMATH_GPT_incorrect_operation_l1202_120267


namespace NUMINAMATH_GPT_triangle_ABC_right_angled_l1202_120245
open Real

theorem triangle_ABC_right_angled (A B C : ℝ) (a b c : ℝ)
  (h1 : cos (2 * A) - cos (2 * B) = 2 * sin C ^ 2)
  (h2 : a = sin A) (h3 : b = sin B) (h4 : c = sin C)
  : a^2 + c^2 = b^2 :=
by sorry

end NUMINAMATH_GPT_triangle_ABC_right_angled_l1202_120245


namespace NUMINAMATH_GPT_spring_stretch_150N_l1202_120254

-- Definitions for the conditions
def spring_stretch (weight : ℕ) : ℕ :=
  if weight = 100 then 20 else sorry

-- The theorem to prove
theorem spring_stretch_150N : spring_stretch 150 = 30 := by
  sorry

end NUMINAMATH_GPT_spring_stretch_150N_l1202_120254


namespace NUMINAMATH_GPT_range_of_a_function_greater_than_exp_neg_x_l1202_120290

noncomputable def f (x a : ℝ) : ℝ := Real.log x + a / x

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 0 < x ∧ f x a = 0) → (0 < a ∧ a ≤ 1 / Real.exp 1) :=
sorry

theorem function_greater_than_exp_neg_x (a : ℝ) (h : a ≥ 2 / Real.exp 1) (x : ℝ) (hx : 0 < x) : f x a > Real.exp (-x) :=
sorry

end NUMINAMATH_GPT_range_of_a_function_greater_than_exp_neg_x_l1202_120290


namespace NUMINAMATH_GPT_find_a_and_theta_find_max_min_g_l1202_120231

noncomputable def f (x a θ : ℝ) : ℝ := (a + 2 * (Real.cos x)^2) * Real.cos (2 * x + θ)

-- Provided conditions
variable (a : ℝ)
variable (θ : ℝ)
variable (is_odd : ∀ x, f x a θ = -f (-x) a θ)
variable (f_pi_over_4 : f ((Real.pi) / 4) a θ = 0)
variable (theta_in_range : 0 < θ ∧ θ < Real.pi)

-- To Prove
theorem find_a_and_theta :
  a = -1 ∧ θ = (Real.pi / 2) :=
sorry

-- Define g(x) and its domain
noncomputable def g (x : ℝ) : ℝ := f x (-1) (Real.pi / 2) + f (x + (Real.pi / 3)) (-1) (Real.pi / 2)

-- Provided domain condition
variable (x_in_domain : 0 ≤ x ∧ x ≤ (Real.pi / 4))

-- To Prove maximum and minimum value of g(x)
theorem find_max_min_g :
  (∀ x, x ∈ Set.Icc (0 : ℝ) (Real.pi / 4) → -((Real.sqrt 3) / 2) ≤ g x ∧ g x ≤ (Real.sqrt 3) / 2)
  ∧ ∃ x_min, g x_min = -((Real.sqrt 3) / 2) ∧ x_min = (Real.pi / 8)
  ∧ ∃ x_max, g x_max = ((Real.sqrt 3) / 2) ∧ x_max = (Real.pi / 4) :=
sorry

end NUMINAMATH_GPT_find_a_and_theta_find_max_min_g_l1202_120231


namespace NUMINAMATH_GPT_iced_coffee_cost_is_2_l1202_120243

def weekly_latte_cost := 4 * 5
def annual_latte_cost := weekly_latte_cost * 52
def weekly_iced_coffee_cost (x : ℝ) := x * 3
def annual_iced_coffee_cost (x : ℝ) := weekly_iced_coffee_cost x * 52
def total_annual_coffee_cost (x : ℝ) := annual_latte_cost + annual_iced_coffee_cost x
def reduced_spending_goal (x : ℝ) := 0.75 * total_annual_coffee_cost x
def saved_amount := 338

theorem iced_coffee_cost_is_2 :
  ∃ x : ℝ, (total_annual_coffee_cost x - reduced_spending_goal x = saved_amount) → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_iced_coffee_cost_is_2_l1202_120243


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1202_120239

theorem geometric_sequence_common_ratio (a : ℕ → ℕ) (q : ℕ) (h2 : a 2 = 8) (h5 : a 5 = 64)
  (h_geom : ∀ n, a (n+1) = a n * q) : q = 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1202_120239


namespace NUMINAMATH_GPT_relay_race_time_l1202_120216

theorem relay_race_time (M S J T : ℕ) 
(hJ : J = 30)
(hS : S = J + 10)
(hM : M = 2 * S)
(hT : T = M - 7) : 
M + S + J + T = 223 :=
by sorry

end NUMINAMATH_GPT_relay_race_time_l1202_120216


namespace NUMINAMATH_GPT_find_x_l1202_120219

variable (x : ℝ)

theorem find_x (h : 0.60 * x = (1/3) * x + 110) : x = 412.5 :=
sorry

end NUMINAMATH_GPT_find_x_l1202_120219


namespace NUMINAMATH_GPT_gcd_lcm_product_75_90_l1202_120272

theorem gcd_lcm_product_75_90 :
  let a := 75
  let b := 90
  let gcd_ab := Int.gcd a b
  let lcm_ab := Int.lcm a b
  gcd_ab * lcm_ab = 6750 :=
by
  let a := 75
  let b := 90
  let gcd_ab := Int.gcd a b
  let lcm_ab := Int.lcm a b
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_75_90_l1202_120272


namespace NUMINAMATH_GPT_zero_in_interval_l1202_120211

open Real

noncomputable def f (x : ℝ) : ℝ := log x + x - 3

theorem zero_in_interval (a b : ℕ) (h1 : b - a = 1) (h2 : 1 ≤ a) (h3 : 1 ≤ b) 
  (h4 : f a < 0) (h5 : 0 < f b) : a + b = 5 :=
sorry

end NUMINAMATH_GPT_zero_in_interval_l1202_120211


namespace NUMINAMATH_GPT_Olga_paints_zero_boards_l1202_120223

variable (t p q t' : ℝ)
variable (rv ro : ℝ)

-- Conditions
axiom Valera_solo_trip : 2 * t + p = 2
axiom Valera_and_Olga_painting_time : 2 * t' + q = 3
axiom Valera_painting_rate : rv = 11 / p
axiom Valera_Omega_painting_rate : rv * q + ro * q = 9
axiom Valera_walk_faster : t' > t

-- Question: How many boards will Olga be able to paint alone if she needs to return home 1 hour after leaving?
theorem Olga_paints_zero_boards :
  t' > 1 → 0 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_Olga_paints_zero_boards_l1202_120223


namespace NUMINAMATH_GPT_vasya_expected_area_greater_l1202_120258

/-- Vasya and Asya roll dice to cut out shapes and determine whose expected area is greater. -/
theorem vasya_expected_area_greater :
  let A : ℕ := 1
  let B : ℕ := 2
  (6 * 7 * (2 * 7 / 6) * 21 / 6) < (21 * 91 / 6) := 
by
  sorry

end NUMINAMATH_GPT_vasya_expected_area_greater_l1202_120258


namespace NUMINAMATH_GPT_proof_problem_l1202_120224

def pos_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
∀ n, 4 * S n = (a n + 1) ^ 2

def sequence_condition (a : ℕ → ℝ) : Prop :=
a 0 = 1 ∧ ∀ n, a (n + 1) - a n = 2

def sum_sequence_T (a : ℕ → ℝ) (T : ℕ → ℝ) :=
∀ n, T n = (1 - 1 / (2 * n + 1))

def range_k (T : ℕ → ℝ) (k : ℝ) : Prop :=
∀ n, T n ≥ k → k ≤ 2 / 3

theorem proof_problem (a : ℕ → ℝ) (S T : ℕ → ℝ) (k : ℝ) :
  pos_sequence a S → sequence_condition a → sum_sequence_T a T → range_k T k :=
by sorry

end NUMINAMATH_GPT_proof_problem_l1202_120224


namespace NUMINAMATH_GPT_largest_number_l1202_120233

theorem largest_number (a b c : ℕ) (h1: a ≤ b) (h2: b ≤ c) 
  (h3: (a + b + c) = 90) (h4: b = 32) (h5: b = a + 4) : c = 30 :=
sorry

end NUMINAMATH_GPT_largest_number_l1202_120233


namespace NUMINAMATH_GPT_five_cubic_km_to_cubic_meters_l1202_120202

theorem five_cubic_km_to_cubic_meters (km_to_m : 1 = 1000) : 
  5 * (1000 ^ 3) = 5000000000 := 
by
  sorry

end NUMINAMATH_GPT_five_cubic_km_to_cubic_meters_l1202_120202


namespace NUMINAMATH_GPT_exist_pair_lcm_gcd_l1202_120213

theorem exist_pair_lcm_gcd (a b: ℤ) : 
  ∃ a b : ℤ, Int.lcm a b - Int.gcd a b = 19 := 
sorry

end NUMINAMATH_GPT_exist_pair_lcm_gcd_l1202_120213


namespace NUMINAMATH_GPT_imaginary_unit_cubic_l1202_120287

def imaginary_unit_property (i : ℂ) : Prop :=
  i^2 = -1

theorem imaginary_unit_cubic (i : ℂ) (h : imaginary_unit_property i) : 1 + i^3 = 1 - i :=
  sorry

end NUMINAMATH_GPT_imaginary_unit_cubic_l1202_120287


namespace NUMINAMATH_GPT_picnic_students_count_l1202_120274

theorem picnic_students_count (x : ℕ) (h1 : (x / 2) + (x / 3) + (x / 4) = 65) : x = 60 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_picnic_students_count_l1202_120274


namespace NUMINAMATH_GPT_value_of_d_l1202_120238

theorem value_of_d (d y : ℤ) (h₁ : y = 2) (h₂ : 5 * y^2 - 8 * y + 55 = d) : d = 59 := by
  sorry

end NUMINAMATH_GPT_value_of_d_l1202_120238


namespace NUMINAMATH_GPT_mario_total_flowers_l1202_120208

def hibiscus_flower_count (n : ℕ) : ℕ :=
  let h1 := 2 + 3 * n
  let h2 := (2 * 2) + 4 * n
  let h3 := (4 * (2 * 2)) + 5 * n
  h1 + h2 + h3

def rose_flower_count (n : ℕ) : ℕ :=
  let r1 := 3 + 2 * n
  let r2 := 5 + 3 * n
  r1 + r2

def sunflower_flower_count (n : ℕ) : ℕ :=
  6 * 2^n

def total_flower_count (n : ℕ) : ℕ :=
  hibiscus_flower_count n + rose_flower_count n + sunflower_flower_count n

theorem mario_total_flowers :
  total_flower_count 2 = 88 :=
by
  unfold total_flower_count hibiscus_flower_count rose_flower_count sunflower_flower_count
  norm_num

end NUMINAMATH_GPT_mario_total_flowers_l1202_120208


namespace NUMINAMATH_GPT_proportion_condition_l1202_120200

variable (a b c d a₁ b₁ c₁ d₁ : ℚ)

theorem proportion_condition
  (h₁ : a / b = c / d)
  (h₂ : a₁ / b₁ = c₁ / d₁) :
  (a + a₁) / (b + b₁) = (c + c₁) / (d + d₁) ↔ a * d₁ + a₁ * d = b₁ * c + b * c₁ := by
  sorry

end NUMINAMATH_GPT_proportion_condition_l1202_120200


namespace NUMINAMATH_GPT_problem1_problem2_l1202_120286

def setA : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}

def setB (m : ℝ) : Set ℝ := {x | (x - m + 2) * (x - m - 2) ≤ 0}

-- Problem 1: prove that if A ∩ B = {x | 0 ≤ x ≤ 3}, then m = 2
theorem problem1 (m : ℝ) : (setA ∩ setB m = {x | 0 ≤ x ∧ x ≤ 3}) → m = 2 :=
by
  sorry

-- Problem 2: prove that if A ⊆ complement of B, then m ∈ (-∞, -3) ∪ (5, +∞)
theorem problem2 (m : ℝ) : (setA ⊆ (fun x => x ∉ setB m)) → (m < -3 ∨ m > 5) :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1202_120286


namespace NUMINAMATH_GPT_find_circle_radius_l1202_120212

-- Definitions of given distances and the parallel chord condition
def isChordParallelToDiameter (c d : ℝ × ℝ) (radius distance1 distance2 : ℝ) : Prop :=
  let p1 := distance1
  let p2 := distance2
  p1 = 5 ∧ p2 = 12 ∧ 
  -- Assuming distances from the end of the diameter to the ends of the chord
  true

-- The main theorem which states the radius of the circle given the conditions
theorem find_circle_radius
  (diameter chord : ℝ × ℝ)
  (R p1 p2 : ℝ)
  (h1 : isChordParallelToDiameter diameter chord R p1 p2) :
  R = 6.5 :=
  by
    sorry

end NUMINAMATH_GPT_find_circle_radius_l1202_120212


namespace NUMINAMATH_GPT_min_value_x2_plus_y2_l1202_120288

theorem min_value_x2_plus_y2 :
  ∀ x y : ℝ, (x + 5)^2 + (y - 12)^2 = 196 → x^2 + y^2 ≥ 1 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_min_value_x2_plus_y2_l1202_120288


namespace NUMINAMATH_GPT_solve_inequalities_l1202_120262

theorem solve_inequalities :
  {x : ℝ | -3 < x ∧ x ≤ -2 ∨ 1 ≤ x ∧ x ≤ 2} =
  { x : ℝ | (5 / (x + 3) ≥ 1) ∧ (x^2 + x - 2 ≥ 0) } :=
sorry

end NUMINAMATH_GPT_solve_inequalities_l1202_120262


namespace NUMINAMATH_GPT_min_inv_sum_l1202_120292

noncomputable def minimum_value_condition (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ 1 = 2*a + b

theorem min_inv_sum (a b : ℝ) (h : minimum_value_condition a b) : 
  ∃ a b : ℝ, (1 / a + 1 / b = 3 + 2 * Real.sqrt 2) := 
by 
  have h1 : a > 0 := h.1;
  have h2 : b > 0 := h.2.1;
  have h3 : 1 = 2 * a + b := h.2.2;
  sorry

end NUMINAMATH_GPT_min_inv_sum_l1202_120292


namespace NUMINAMATH_GPT_divide_payment_correctly_l1202_120270

-- Define the number of logs contributed by each person
def logs_troikin : ℕ := 3
def logs_pyaterkin : ℕ := 5
def logs_bestoplivny : ℕ := 0

-- Define the total number of logs
def total_logs : ℕ := logs_troikin + logs_pyaterkin + logs_bestoplivny

-- Define the total number of logs used equally
def logs_per_person : ℚ := total_logs / 3

-- Define the total payment made by Bestoplivny 
def total_payment : ℕ := 80

-- Define the cost per log
def cost_per_log : ℚ := total_payment / logs_per_person

-- Define the contribution of each person to Bestoplivny
def bestoplivny_from_troikin : ℚ := logs_troikin - logs_per_person
def bestoplivny_from_pyaterkin : ℚ := logs_pyaterkin - (logs_per_person - bestoplivny_from_troikin)

-- Define the kopecks received by Troikina and Pyaterkin
def kopecks_troikin : ℚ := bestoplivny_from_troikin * cost_per_log
def kopecks_pyaterkin : ℚ := bestoplivny_from_pyaterkin * cost_per_log

-- Main theorem to prove the correct division of kopecks
theorem divide_payment_correctly : kopecks_troikin = 10 ∧ kopecks_pyaterkin = 70 :=
by
  -- ... Proof goes here
  sorry

end NUMINAMATH_GPT_divide_payment_correctly_l1202_120270


namespace NUMINAMATH_GPT_sin_law_of_sines_l1202_120276

theorem sin_law_of_sines (a b : ℝ) (sin_A sin_B : ℝ)
  (h1 : a = 3)
  (h2 : b = 4)
  (h3 : sin_A = 3 / 5) :
  sin_B = 4 / 5 := 
sorry

end NUMINAMATH_GPT_sin_law_of_sines_l1202_120276


namespace NUMINAMATH_GPT_parabola_difference_eq_l1202_120242

variable (a b c : ℝ)

def original_parabola (x : ℝ) : ℝ := a * x^2 + b * x + c
def reflected_parabola (x : ℝ) : ℝ := -(a * x^2 + b * x + c)
def translated_original (x : ℝ) : ℝ := a * x^2 + b * x + c + 3
def translated_reflection (x : ℝ) : ℝ := -(a * x^2 + b * x + c) - 3

theorem parabola_difference_eq (x : ℝ) :
  (translated_original a b c x) - (translated_reflection a b c x) = 2 * a * x^2 + 2 * b * x + 2 * c + 6 :=
by 
  sorry

end NUMINAMATH_GPT_parabola_difference_eq_l1202_120242


namespace NUMINAMATH_GPT_athlete_weight_l1202_120235

theorem athlete_weight (a b c : ℤ) (k₁ k₂ k₃ : ℤ)
  (h1 : (a + b + c) / 3 = 42)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 43)
  (h4 : a = 5 * k₁)
  (h5 : b = 5 * k₂)
  (h6 : c = 5 * k₃) :
  b = 40 :=
by
  sorry

end NUMINAMATH_GPT_athlete_weight_l1202_120235


namespace NUMINAMATH_GPT_arith_seq_sum_nine_l1202_120260

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arith_seq := ∀ n : ℕ, a n = a 0 + (n - 1) * (a 1 - a 0)

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := 
  ∀ n : ℕ, S n = (n / 2) * (a 0 + a (n - 1))

theorem arith_seq_sum_nine (h_seq : arith_seq a) (h_sum : sum_first_n_terms a S) (h_S9 : S 9 = 18) : 
  a 2 + a 5 + a 8 = 6 :=
  sorry

end NUMINAMATH_GPT_arith_seq_sum_nine_l1202_120260


namespace NUMINAMATH_GPT_length_OP_l1202_120278

noncomputable def right_triangle_length_OP (XY XZ YZ : ℝ) (rO rP : ℝ) : ℝ :=
  let O := rO
  let P := rP
  -- Coordinates of point Y and Z can be O = (0, r), P = (OP, r)
  25 -- directly from the given correct answer

theorem length_OP (XY XZ YZ : ℝ) (rO rP : ℝ) (hXY : XY = 7) (hXZ : XZ = 24) (hYZ : YZ = 25) 
  (hO : rO = YZ - rO) (hP : rP = YZ - rP) : 
  right_triangle_length_OP XY XZ YZ rO rP = 25 :=
sorry

end NUMINAMATH_GPT_length_OP_l1202_120278


namespace NUMINAMATH_GPT_Glorys_favorite_number_l1202_120261

variable (M G : ℝ)

theorem Glorys_favorite_number :
  (M = G / 3) →
  (M + G = 600) →
  (G = 450) :=
by
sorry

end NUMINAMATH_GPT_Glorys_favorite_number_l1202_120261


namespace NUMINAMATH_GPT_square_side_measurement_error_l1202_120230

theorem square_side_measurement_error (S S' : ℝ) (h1 : S' = S * Real.sqrt 1.0404) : 
  (S' - S) / S * 100 = 2 :=
by
  sorry

end NUMINAMATH_GPT_square_side_measurement_error_l1202_120230


namespace NUMINAMATH_GPT_RebeccaHasTwentyMarbles_l1202_120269

variable (groups : ℕ) (marbles_per_group : ℕ) (total_marbles : ℕ)

def totalMarbles (g m : ℕ) : ℕ :=
  g * m

theorem RebeccaHasTwentyMarbles
  (h1 : groups = 5)
  (h2 : marbles_per_group = 4)
  (h3 : total_marbles = totalMarbles groups marbles_per_group) :
  total_marbles = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_RebeccaHasTwentyMarbles_l1202_120269


namespace NUMINAMATH_GPT_krishan_nandan_investment_ratio_l1202_120247

theorem krishan_nandan_investment_ratio
    (X t : ℝ) (k : ℝ)
    (h1 : X * t = 6000)
    (h2 : X * t + k * X * 2 * t = 78000) :
    k = 6 := by
  sorry

end NUMINAMATH_GPT_krishan_nandan_investment_ratio_l1202_120247


namespace NUMINAMATH_GPT_prove_n_eq_one_l1202_120259

-- Definitions of the vectors a and b
def vector_a (n : ℝ) : ℝ × ℝ := (1, n)
def vector_b (n : ℝ) : ℝ × ℝ := (-1, n - 2)

-- Definition of collinearity between two vectors
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)

-- Theorem to prove that if a and b are collinear, then n = 1
theorem prove_n_eq_one (n : ℝ) (h_collinear : collinear (vector_a n) (vector_b n)) : n = 1 :=
sorry

end NUMINAMATH_GPT_prove_n_eq_one_l1202_120259


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_nine_l1202_120280

variable {α : Type*} [LinearOrderedField α]

/-- An arithmetic sequence (a_n) is defined by a starting term a_1 and a common difference d. -/
def arithmetic_seq (a d n : α) : α := a + (n - 1) * d

/-- The sum of the first n terms of an arithmetic sequence. -/
def arithmetic_sum (a d n : α) : α := n / 2 * (2 * a + (n - 1) * d)

/-- Prove that for a given arithmetic sequence where a_2 + a_4 + a_9 = 24, the sum of the first 9 terms is 72. -/
theorem arithmetic_sequence_sum_nine 
  {a d : α}
  (h : arithmetic_seq a d 2 + arithmetic_seq a d 4 + arithmetic_seq a d 9 = 24) :
  arithmetic_sum a d 9 = 72 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_nine_l1202_120280


namespace NUMINAMATH_GPT_simplify_polynomial_l1202_120244

theorem simplify_polynomial : 
  (5 - 3 * x - 7 * x^2 + 3 + 12 * x - 9 * x^2 - 8 + 15 * x + 21 * x^2) = (5 * x^2 + 24 * x) :=
by 
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l1202_120244


namespace NUMINAMATH_GPT_quadratic_has_one_solution_at_zero_l1202_120298

theorem quadratic_has_one_solution_at_zero (k : ℝ) :
  ((k - 2) * (0 : ℝ)^2 + 3 * (0 : ℝ) + k^2 - 4 = 0) →
  (3^2 - 4 * (k - 2) * (k^2 - 4) = 0) → k = -2 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_quadratic_has_one_solution_at_zero_l1202_120298


namespace NUMINAMATH_GPT_eu_countries_2012_forms_set_l1202_120214

def higher_level_skills_students := false -- Condition A can't form a set.
def tall_trees := false -- Condition B can't form a set.
def developed_cities := false -- Condition D can't form a set.
def eu_countries_2012 := true -- Condition C forms a set.

theorem eu_countries_2012_forms_set : 
  higher_level_skills_students = false ∧ tall_trees = false ∧ developed_cities = false ∧ eu_countries_2012 = true :=
by {
  sorry
}

end NUMINAMATH_GPT_eu_countries_2012_forms_set_l1202_120214


namespace NUMINAMATH_GPT_box_surface_area_l1202_120265

variables (a b c : ℝ)

noncomputable def sum_edges : ℝ := 4 * (a + b + c)
noncomputable def diagonal_length : ℝ := Real.sqrt (a^2 + b^2 + c^2)
noncomputable def surface_area : ℝ := 2 * (a * b + b * c + c * a)

/- The problem states that the sum of the lengths of the edges and the diagonal length gives us these values. -/
theorem box_surface_area (h1 : sum_edges a b c = 168) (h2 : diagonal_length a b c = 25) : surface_area a b c = 1139 :=
sorry

end NUMINAMATH_GPT_box_surface_area_l1202_120265


namespace NUMINAMATH_GPT_simplify_expression_l1202_120297

theorem simplify_expression (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ 2) (h3 : x ≠ 4) (h4 : x ≠ 5) :
  ( (x^2 - 4*x + 3) / (x^2 - 6*x + 9) ) / ( (x^2 - 6*x + 8) / (x^2 - 8*x + 15) ) =
  ( (x - 1) * (x - 5) ) / ( (x - 3) * (x - 4) * (x - 2) ) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1202_120297


namespace NUMINAMATH_GPT_angle_measure_l1202_120253

theorem angle_measure (α : ℝ) (h1 : α - (90 - α) = 20) : α = 55 := by
  -- Proof to be provided here
  sorry

end NUMINAMATH_GPT_angle_measure_l1202_120253


namespace NUMINAMATH_GPT_root_of_polynomial_l1202_120215

theorem root_of_polynomial (a b : ℝ) (h₁ : a^4 + a^3 - 1 = 0) (h₂ : b^4 + b^3 - 1 = 0) : 
  (ab : ℝ) → ab * ab * ab * ab * ab * ab + ab * ab * ab * ab + ab * ab * ab - ab * ab - 1 = 0 :=
sorry

end NUMINAMATH_GPT_root_of_polynomial_l1202_120215


namespace NUMINAMATH_GPT_duration_of_each_class_is_3_l1202_120268

theorem duration_of_each_class_is_3
    (weeks : ℕ) 
    (x : ℝ) 
    (weekly_additional_class_hours : ℝ) 
    (homework_hours_per_week : ℝ) 
    (total_hours : ℝ) 
    (h1 : weeks = 24)
    (h2 : weekly_additional_class_hours = 4)
    (h3 : homework_hours_per_week = 4)
    (h4 : total_hours = 336) :
    (2 * x + weekly_additional_class_hours + homework_hours_per_week) * weeks = total_hours → x = 3 := 
by 
  sorry

end NUMINAMATH_GPT_duration_of_each_class_is_3_l1202_120268


namespace NUMINAMATH_GPT_minimum_possible_value_of_Box_l1202_120279

theorem minimum_possible_value_of_Box :
  ∃ a b : ℤ, a ≠ b ∧ a * b = 45 ∧ 
    (∀ c d : ℤ, c * d = 45 → c^2 + d^2 ≥ 106) ∧ a^2 + b^2 = 106 :=
by
  sorry

end NUMINAMATH_GPT_minimum_possible_value_of_Box_l1202_120279


namespace NUMINAMATH_GPT_unsuitable_temperature_for_refrigerator_l1202_120218

theorem unsuitable_temperature_for_refrigerator:
  let avg_temp := -18
  let variation := 2
  let min_temp := avg_temp - variation
  let max_temp := avg_temp + variation
  let temp_A := -17
  let temp_B := -18
  let temp_C := -19
  let temp_D := -22
  temp_D < min_temp ∨ temp_D > max_temp := by
  sorry

end NUMINAMATH_GPT_unsuitable_temperature_for_refrigerator_l1202_120218


namespace NUMINAMATH_GPT_totalProblemsSolved_l1202_120263

-- Given conditions
def initialProblemsSolved : Nat := 45
def additionalProblemsSolved : Nat := 18

-- Statement to prove the total problems solved equals 63
theorem totalProblemsSolved : initialProblemsSolved + additionalProblemsSolved = 63 := 
by
  sorry

end NUMINAMATH_GPT_totalProblemsSolved_l1202_120263


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_l1202_120289

theorem arithmetic_sequence_a5 {a : ℕ → ℝ} (h₁ : a 2 + a 8 = 16) : a 5 = 8 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_l1202_120289


namespace NUMINAMATH_GPT_parabola_line_intersection_l1202_120283

theorem parabola_line_intersection :
  ∀ (x y : ℝ), 
  (y = 20 * x^2 + 19 * x) ∧ (y = 20 * x + 19) →
  y = 20 * x^3 + 19 * x^2 :=
by sorry

end NUMINAMATH_GPT_parabola_line_intersection_l1202_120283


namespace NUMINAMATH_GPT_part_I_part_II_l1202_120222

theorem part_I : 
  (∀ x : ℝ, |x - (2 : ℝ)| ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) :=
  sorry

theorem part_II :
  (∀ a b c : ℝ, a - 2 * b + c = 2 → a^2 + b^2 + c^2 ≥ 2 / 3) :=
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1202_120222


namespace NUMINAMATH_GPT_five_a_squared_plus_one_divisible_by_three_l1202_120264

theorem five_a_squared_plus_one_divisible_by_three (a : ℤ) (h : a % 3 ≠ 0) : (5 * a^2 + 1) % 3 = 0 :=
sorry

end NUMINAMATH_GPT_five_a_squared_plus_one_divisible_by_three_l1202_120264


namespace NUMINAMATH_GPT_qiqi_initial_batteries_qiqi_jiajia_difference_after_transfer_l1202_120203

variable (m : Int)

theorem qiqi_initial_batteries (m : Int) : 
  let Qiqi_initial := 2 * m - 2
  Qiqi_initial = 2 * m - 2 := sorry

theorem qiqi_jiajia_difference_after_transfer (m : Int) : 
  let Qiqi_after := 2 * m - 2 - 2
  let Jiajia_after := m + 2
  Qiqi_after - Jiajia_after = m - 6 := sorry

end NUMINAMATH_GPT_qiqi_initial_batteries_qiqi_jiajia_difference_after_transfer_l1202_120203


namespace NUMINAMATH_GPT_swim_team_girls_l1202_120271

-- Definitions using the given conditions
variables (B G : ℕ)
theorem swim_team_girls (h1 : G = 5 * B) (h2 : G + B = 96) : G = 80 :=
sorry

end NUMINAMATH_GPT_swim_team_girls_l1202_120271


namespace NUMINAMATH_GPT_investment_time_l1202_120204

theorem investment_time (P R diff : ℝ) (T : ℕ) 
  (hP : P = 1500)
  (hR : R = 0.10)
  (hdiff : diff = 15)
  (h1 : P * ((1 + R) ^ T - 1) - (P * R * T) = diff) 
  : T = 2 := 
by
  -- proof steps here
  sorry

end NUMINAMATH_GPT_investment_time_l1202_120204


namespace NUMINAMATH_GPT_outfit_combinations_l1202_120221

theorem outfit_combinations (shirts ties hat_choices : ℕ) (h_shirts : shirts = 8) (h_ties : ties = 7) (h_hat_choices : hat_choices = 3) : shirts * ties * hat_choices = 168 := by
  sorry

end NUMINAMATH_GPT_outfit_combinations_l1202_120221


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1202_120275

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }
def expected_intersection : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_of_A_and_B : A ∩ B = expected_intersection :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1202_120275


namespace NUMINAMATH_GPT_point_not_in_region_l1202_120226

theorem point_not_in_region : ¬ (3 * 2 + 2 * 0 < 6) :=
by simp [lt_irrefl]

end NUMINAMATH_GPT_point_not_in_region_l1202_120226


namespace NUMINAMATH_GPT_sequence_initial_value_l1202_120241

theorem sequence_initial_value (a : ℕ → ℚ) 
  (h : ∀ n : ℕ, a (n + 1)^2 - a (n + 1) = a n) : a 1 = 0 ∨ a 1 = 2 :=
sorry

end NUMINAMATH_GPT_sequence_initial_value_l1202_120241


namespace NUMINAMATH_GPT_curve_y_all_real_l1202_120217

theorem curve_y_all_real (y : ℝ) : ∃ (x : ℝ), 2 * x * |x| + y^2 = 1 :=
sorry

end NUMINAMATH_GPT_curve_y_all_real_l1202_120217


namespace NUMINAMATH_GPT_distinct_ordered_pairs_l1202_120225

/-- There are 9 distinct ordered pairs of positive integers (m, n) such that the sum of the 
    reciprocals of m and n equals 1/6. -/
theorem distinct_ordered_pairs : 
  ∃ (s : Finset (ℕ × ℕ)), s.card = 9 ∧ 
  ∀ (p : ℕ × ℕ), p ∈ s → 
    (0 < p.1 ∧ 0 < p.2) ∧ 
    (1 / (p.1 : ℚ) + 1 / (p.2 : ℚ) = 1 / 6) :=
sorry

end NUMINAMATH_GPT_distinct_ordered_pairs_l1202_120225


namespace NUMINAMATH_GPT_option_D_is_divisible_by_9_l1202_120291

theorem option_D_is_divisible_by_9 (k : ℕ) (hk : k > 0) : 9 ∣ 3 * (2 + 7^k) := 
sorry

end NUMINAMATH_GPT_option_D_is_divisible_by_9_l1202_120291


namespace NUMINAMATH_GPT_polygon_interior_angle_eq_l1202_120299

theorem polygon_interior_angle_eq (n : ℕ) (h : ∀ i, 1 ≤ i → i ≤ n → (interior_angle : ℝ) = 108) : n = 5 := 
sorry

end NUMINAMATH_GPT_polygon_interior_angle_eq_l1202_120299


namespace NUMINAMATH_GPT_product_of_good_numbers_does_not_imply_sum_digits_property_l1202_120252

-- Define what it means for a number to be "good".
def is_good (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 0 ∨ d = 1

-- Define the sum of the digits of a number
def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main theorem statement
theorem product_of_good_numbers_does_not_imply_sum_digits_property :
  ∀ (A B : ℕ), is_good A → is_good B → is_good (A * B) →
  ¬ (sum_digits (A * B) = sum_digits A * sum_digits B) :=
by
  intros A B hA hB hAB
  -- The detailed proof is not provided here, hence we use sorry to skip it.
  sorry

end NUMINAMATH_GPT_product_of_good_numbers_does_not_imply_sum_digits_property_l1202_120252


namespace NUMINAMATH_GPT_third_vertex_coordinates_l1202_120249

theorem third_vertex_coordinates (x : ℝ) (h : 6 * |x| = 96) : x = 16 ∨ x = -16 :=
by
  sorry

end NUMINAMATH_GPT_third_vertex_coordinates_l1202_120249


namespace NUMINAMATH_GPT_symmetric_point_x_axis_l1202_120228

theorem symmetric_point_x_axis (P : ℝ × ℝ) (hx : P = (2, 3)) : P.1 = 2 ∧ P.2 = -3 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_symmetric_point_x_axis_l1202_120228


namespace NUMINAMATH_GPT_problem_1_problem_2_l1202_120248

variable (x y : ℝ)
noncomputable def x_val : ℝ := 2 + Real.sqrt 3
noncomputable def y_val : ℝ := 2 - Real.sqrt 3

theorem problem_1 :
  3 * x_val^2 + 5 * x_val * y_val + 3 * y_val^2 = 47 := sorry

theorem problem_2 :
  Real.sqrt (x_val / y_val) + Real.sqrt (y_val / x_val) = 4 := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1202_120248


namespace NUMINAMATH_GPT_find_y_l1202_120229

theorem find_y (x y : ℤ) (h1 : x + y = 300) (h2 : x - y = 200) : y = 50 :=
sorry

end NUMINAMATH_GPT_find_y_l1202_120229


namespace NUMINAMATH_GPT_domain_and_parity_range_of_a_l1202_120234

noncomputable def f (a x : ℝ) := Real.log (1 + x) / Real.log a
noncomputable def g (a x : ℝ) := Real.log (1 - x) / Real.log a

theorem domain_and_parity (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (∀ x, f a x * g a x = f a (-x) * g a (-x)) ∧ (∀ x, -1 < x ∧ x < 1) :=
sorry

theorem range_of_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : f a 1 + g a (1/4) < 1) :
  (a ∈ (Set.Ioo 0 1 ∪ Set.Ioi (3/2))) :=
sorry

end NUMINAMATH_GPT_domain_and_parity_range_of_a_l1202_120234


namespace NUMINAMATH_GPT_find_c_l1202_120205

variable (y c : ℝ)

theorem find_c (h : y > 0) (h_expr : (7 * y / 20 + c * y / 10) = 0.6499999999999999 * y) : c = 3 := by
  sorry

end NUMINAMATH_GPT_find_c_l1202_120205


namespace NUMINAMATH_GPT_arithmetic_seq_properties_l1202_120251

theorem arithmetic_seq_properties (a : ℕ → ℝ) (d a1 : ℝ) (S : ℕ → ℝ) :
  (a 1 + a 3 = 8) ∧ (a 4 ^ 2 = a 2 * a 9) →
  ((a1 = 4 ∧ d = 0 ∧ (∀ n, S n = 4 * n)) ∨
   (a1 = 1 ∧ d = 3 ∧ (∀ n, S n = (3 * n^2 - n) / 2))) := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_properties_l1202_120251


namespace NUMINAMATH_GPT_div_add_example_l1202_120232

theorem div_add_example : 150 / (10 / 2) + 5 = 35 := by
  sorry

end NUMINAMATH_GPT_div_add_example_l1202_120232


namespace NUMINAMATH_GPT_comparison_among_abc_l1202_120281

noncomputable def a : ℝ := 2^(1/5)
noncomputable def b : ℝ := (1/5)^2
noncomputable def c : ℝ := Real.log (1/5) / Real.log 2

theorem comparison_among_abc : a > b ∧ b > c :=
by
  -- Assume the necessary conditions and the conclusion.
  sorry

end NUMINAMATH_GPT_comparison_among_abc_l1202_120281


namespace NUMINAMATH_GPT_part_one_part_two_l1202_120266

noncomputable def f (a x : ℝ) := a * Real.log x - x + 1

theorem part_one (a : ℝ) (h : ∀ x > 0, f a x ≤ 0) : a = 1 := 
sorry

theorem part_two (h₁ : ∀ x > 0, f 1 x ≤ 0) (x : ℝ) (h₂ : 0 < x) (h₃ : x < Real.pi / 2) :
  Real.exp x * Real.sin x - x > f 1 x :=
sorry

end NUMINAMATH_GPT_part_one_part_two_l1202_120266


namespace NUMINAMATH_GPT_total_tape_length_l1202_120255

-- Definitions based on the problem conditions
def first_side_songs : ℕ := 6
def second_side_songs : ℕ := 4
def song_length : ℕ := 4

-- Statement to prove the total tape length is 40 minutes
theorem total_tape_length : (first_side_songs + second_side_songs) * song_length = 40 := by
  sorry

end NUMINAMATH_GPT_total_tape_length_l1202_120255


namespace NUMINAMATH_GPT_find_f_of_2_l1202_120236

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f_of_2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := by
  sorry

end NUMINAMATH_GPT_find_f_of_2_l1202_120236


namespace NUMINAMATH_GPT_length_of_AE_l1202_120240

theorem length_of_AE (AD AE EB EF: ℝ) (h_AD: AD = 80) (h_EB: EB = 40) (h_EF: EF = 30) 
  (h_eq_area: 2 * ((EB * EF) + (1 / 2) * (ED * (AD - EF))) = AD * (AD - AE)) : AE = 15 :=
  sorry

end NUMINAMATH_GPT_length_of_AE_l1202_120240


namespace NUMINAMATH_GPT_debby_candy_problem_l1202_120293

theorem debby_candy_problem (D : ℕ) (sister_candy : ℕ) (eaten : ℕ) (remaining : ℕ) 
  (h1 : sister_candy = 42) (h2 : eaten = 35) (h3 : remaining = 39) :
  D + sister_candy - eaten = remaining ↔ D = 32 :=
by
  sorry

end NUMINAMATH_GPT_debby_candy_problem_l1202_120293


namespace NUMINAMATH_GPT_tessellation_coloring_l1202_120257

theorem tessellation_coloring :
  ∀ (T : Type) (colors : T → ℕ) (adjacent : T → T → Prop),
    (∀ t1 t2, adjacent t1 t2 → colors t1 ≠ colors t2) → 
    (∃ c1 c2 c3, ∀ t, colors t = c1 ∨ colors t = c2 ∨ colors t = c3) :=
sorry

end NUMINAMATH_GPT_tessellation_coloring_l1202_120257


namespace NUMINAMATH_GPT_johnson_class_more_students_l1202_120296

theorem johnson_class_more_students
  (finley_class_students : ℕ)
  (johnson_class_students : ℕ)
  (h_finley : finley_class_students = 24)
  (h_johnson : johnson_class_students = 22) :
  johnson_class_students - finley_class_students / 2 = 10 :=
  sorry

end NUMINAMATH_GPT_johnson_class_more_students_l1202_120296


namespace NUMINAMATH_GPT_sin_double_angle_solution_l1202_120206

theorem sin_double_angle_solution (φ : ℝ) 
  (h : (7 / 13) + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := 
sorry

end NUMINAMATH_GPT_sin_double_angle_solution_l1202_120206


namespace NUMINAMATH_GPT_proof_problem_l1202_120246

noncomputable def question (a b c : ℝ) : ℝ := 
  (a ^ 2 * b ^ 2) / ((a ^ 2 + b * c) * (b ^ 2 + a * c)) +
  (a ^ 2 * c ^ 2) / ((a ^ 2 + b * c) * (c ^ 2 + a * b)) +
  (b ^ 2 * c ^ 2) / ((b ^ 2 + a * c) * (c ^ 2 + a * b))

theorem proof_problem (a b c : ℝ) (h : a ≠ 0) (h1 : b ≠ 0) (h2 : c ≠ 0) 
  (h3 : a ^ 2 + b ^ 2 + c ^ 2 = a * b + b * c + c * a ) : 
  question a b c = 1 := 
by 
  sorry

end NUMINAMATH_GPT_proof_problem_l1202_120246


namespace NUMINAMATH_GPT_find_a_l1202_120220

theorem find_a (a : ℝ) : (∀ x : ℝ, (x^2 - 4 * x + a) + |x - 3| ≤ 5) → (∃ x : ℝ, x = 3) → a = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1202_120220


namespace NUMINAMATH_GPT_graph_EQ_a_l1202_120209

theorem graph_EQ_a (x y : ℝ) : (x - 2) * (y + 3) = 0 ↔ x = 2 ∨ y = -3 :=
by sorry

end NUMINAMATH_GPT_graph_EQ_a_l1202_120209


namespace NUMINAMATH_GPT_triangles_from_sticks_l1202_120201

theorem triangles_from_sticks (a1 a2 a3 a4 a5 a6 : ℕ) (h_diff: a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a1 ≠ a6 
∧ a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a2 ≠ a6 
∧ a3 ≠ a4 ∧ a3 ≠ a5 ∧ a3 ≠ a6 
∧ a4 ≠ a5 ∧ a4 ≠ a6 
∧ a5 ≠ a6) (h_order: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6) : 
  (a1 + a3 > a5 ∧ a1 + a5 > a3 ∧ a3 + a5 > a1) ∧ 
  (a2 + a4 > a6 ∧ a2 + a6 > a4 ∧ a4 + a6 > a2) :=
by
  sorry

end NUMINAMATH_GPT_triangles_from_sticks_l1202_120201


namespace NUMINAMATH_GPT_eq_of_divisibility_l1202_120285

theorem eq_of_divisibility (a b : ℕ) (h : (a^2 + b^2) ∣ (a * b)) : a = b :=
  sorry

end NUMINAMATH_GPT_eq_of_divisibility_l1202_120285


namespace NUMINAMATH_GPT_hyperbola_range_m_l1202_120295

theorem hyperbola_range_m (m : ℝ) : (m - 2) * (m - 6) < 0 ↔ 2 < m ∧ m < 6 :=
by sorry

end NUMINAMATH_GPT_hyperbola_range_m_l1202_120295


namespace NUMINAMATH_GPT_transformed_center_is_correct_l1202_120282

-- Definition for transformations
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

def translate_right (p : ℝ × ℝ) (dx : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2)

def translate_up (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + dy)

-- Given conditions
def initial_center : ℝ × ℝ := (4, -3)
def reflection_center := reflect_x initial_center
def translated_right_center := translate_right reflection_center 5
def final_center := translate_up translated_right_center 3

-- The statement to be proved
theorem transformed_center_is_correct : final_center = (9, 6) :=
by
  sorry

end NUMINAMATH_GPT_transformed_center_is_correct_l1202_120282
