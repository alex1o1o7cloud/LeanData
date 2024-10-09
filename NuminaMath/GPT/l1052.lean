import Mathlib

namespace find_a_l1052_105213

def are_parallel (a : ℝ) : Prop :=
  (a + 1) = (2 - a)

theorem find_a (a : ℝ) (h : are_parallel a) : a = 0 :=
sorry

end find_a_l1052_105213


namespace sum_four_least_tau_equals_eight_l1052_105273

def tau (n : ℕ) : ℕ := n.divisors.card

theorem sum_four_least_tau_equals_eight :
  ∃ n1 n2 n3 n4 : ℕ, 
    tau n1 + tau (n1 + 1) = 8 ∧ 
    tau n2 + tau (n2 + 1) = 8 ∧
    tau n3 + tau (n3 + 1) = 8 ∧
    tau n4 + tau (n4 + 1) = 8 ∧
    n1 + n2 + n3 + n4 = 80 := 
sorry

end sum_four_least_tau_equals_eight_l1052_105273


namespace smallest_positive_angle_l1052_105202

theorem smallest_positive_angle (α : ℝ) (h : (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3)) = (Real.sin α, Real.cos α)) : 
  α = 11 * Real.pi / 6 := by
sorry

end smallest_positive_angle_l1052_105202


namespace students_exceed_pets_by_70_l1052_105298

theorem students_exceed_pets_by_70 :
  let n_classrooms := 5
  let students_per_classroom := 22
  let rabbits_per_classroom := 3
  let hamsters_per_classroom := 5
  let total_students := students_per_classroom * n_classrooms
  let total_rabbits := rabbits_per_classroom * n_classrooms
  let total_hamsters := hamsters_per_classroom * n_classrooms
  let total_pets := total_rabbits + total_hamsters
  total_students - total_pets = 70 :=
  by
    sorry

end students_exceed_pets_by_70_l1052_105298


namespace two_sin_cos_75_eq_half_l1052_105288

noncomputable def two_sin_cos_of_75_deg : ℝ :=
  2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180)

theorem two_sin_cos_75_eq_half : two_sin_cos_of_75_deg = 1 / 2 :=
by
  -- The steps to prove this theorem are omitted deliberately
  sorry

end two_sin_cos_75_eq_half_l1052_105288


namespace sufficient_condition_l1052_105281

theorem sufficient_condition (a b : ℝ) (h : |a + b| > 1) : |a| + |b| > 1 := 
by sorry

end sufficient_condition_l1052_105281


namespace average_reading_days_l1052_105270

theorem average_reading_days :
  let days_participated := [2, 3, 4, 5, 6]
  let students := [5, 4, 7, 3, 6]
  let total_days := List.zipWith (· * ·) days_participated students |>.sum
  let total_students := students.sum
  let average := total_days / total_students
  average = 4.04 := sorry

end average_reading_days_l1052_105270


namespace AngiesClassGirlsCount_l1052_105244

theorem AngiesClassGirlsCount (n_girls n_boys : ℕ) (total_students : ℕ)
  (h1 : n_girls = 2 * (total_students / 5))
  (h2 : n_boys = 3 * (total_students / 5))
  (h3 : n_girls + n_boys = 20)
  : n_girls = 8 :=
by
  sorry

end AngiesClassGirlsCount_l1052_105244


namespace initial_population_of_town_l1052_105205

theorem initial_population_of_town 
  (final_population : ℝ) 
  (growth_rate : ℝ) 
  (years : ℕ) 
  (initial_population : ℝ) 
  (h : final_population = initial_population * (1 + growth_rate) ^ years) : 
  initial_population = 297500 / (1 + 0.07) ^ 10 :=
by
  sorry

end initial_population_of_town_l1052_105205


namespace evaluate_expression_l1052_105287

variable (g : ℕ → ℕ)
variable (g_inv : ℕ → ℕ)
variable (h1 : g 4 = 7)
variable (h2 : g 6 = 2)
variable (h3 : g 3 = 6)

theorem evaluate_expression : g_inv (g_inv 6 + g_inv 7) = 4 := by
  sorry

end evaluate_expression_l1052_105287


namespace denominator_divisor_zero_l1052_105200

theorem denominator_divisor_zero (n : ℕ) : n ≠ 0 → (∀ d, d ≠ 0 → d / n ≠ d / 0) :=
by
  sorry

end denominator_divisor_zero_l1052_105200


namespace volume_of_cube_with_surface_area_l1052_105290

theorem volume_of_cube_with_surface_area (S : ℝ) (hS : S = 294) : 
  ∃ V : ℝ, V = 343 :=
by
  let s := (S / 6).sqrt
  have hs : s = 7 := by sorry
  use s ^ 3
  simp [hs]
  exact sorry

end volume_of_cube_with_surface_area_l1052_105290


namespace average_students_l1052_105228

def ClassGiraffe : ℕ := 225

def ClassElephant (giraffe: ℕ) : ℕ := giraffe + 48

def ClassRabbit (giraffe: ℕ) : ℕ := giraffe - 24

theorem average_students (giraffe : ℕ) (elephant : ℕ) (rabbit : ℕ) :
  giraffe = 225 → elephant = giraffe + 48 → rabbit = giraffe - 24 →
  (giraffe + elephant + rabbit) / 3 = 233 := by
  sorry

end average_students_l1052_105228


namespace Wilson_sledding_l1052_105282

variable (T S : ℕ)

theorem Wilson_sledding (h1 : S = T / 2) (h2 : (2 * T) + (3 * S) = 14) : T = 4 := by
  sorry

end Wilson_sledding_l1052_105282


namespace radius_correct_l1052_105256

noncomputable def radius_of_circle (chord_length tang_secant_segment : ℝ) : ℝ :=
  let r := 6.25
  r

theorem radius_correct
  (chord_length : ℝ)
  (tangent_secant_segment : ℝ)
  (parallel_secant_internal_segment : ℝ)
  : chord_length = 10 ∧ parallel_secant_internal_segment = 12 → radius_of_circle chord_length parallel_secant_internal_segment = 6.25 :=
by
  intros h
  sorry

end radius_correct_l1052_105256


namespace speed_of_stream_l1052_105291

theorem speed_of_stream (x : ℝ) (boat_speed : ℝ) (distance_one_way : ℝ) (total_time : ℝ) 
  (h1 : boat_speed = 16) 
  (h2 : distance_one_way = 7560) 
  (h3 : total_time = 960) 
  (h4 : (distance_one_way / (boat_speed + x)) + (distance_one_way / (boat_speed - x)) = total_time) 
  : x = 2 := 
  sorry

end speed_of_stream_l1052_105291


namespace Lorelai_jellybeans_correct_l1052_105266

def Gigi_jellybeans : ℕ := 15
def Rory_jellybeans : ℕ := Gigi_jellybeans + 30
def Total_jellybeans : ℕ := Rory_jellybeans + Gigi_jellybeans
def Lorelai_jellybeans : ℕ := 3 * Total_jellybeans

theorem Lorelai_jellybeans_correct : Lorelai_jellybeans = 180 := by
  sorry

end Lorelai_jellybeans_correct_l1052_105266


namespace focus_of_parabola_l1052_105269

theorem focus_of_parabola (m : ℝ) (m_nonzero : m ≠ 0) :
    ∃ (focus_x focus_y : ℝ), (focus_x, focus_y) = (m, 0) ∧
        ∀ (y : ℝ), (x = 1/(4*m) * y^2) := 
sorry

end focus_of_parabola_l1052_105269


namespace machine_P_additional_hours_unknown_l1052_105278

noncomputable def machine_A_rate : ℝ := 1.0000000000000013

noncomputable def machine_Q_rate : ℝ := machine_A_rate + 0.10 * machine_A_rate

noncomputable def total_sprockets : ℝ := 110

noncomputable def machine_Q_hours : ℝ := total_sprockets / machine_Q_rate

variable (x : ℝ) -- additional hours taken by Machine P

theorem machine_P_additional_hours_unknown :
  ∃ x, total_sprockets / machine_Q_rate + x = total_sprockets / ((total_sprockets + total_sprockets / machine_Q_rate * x) / total_sprockets) :=
sorry

end machine_P_additional_hours_unknown_l1052_105278


namespace number_of_pieces_of_paper_l1052_105253

def three_digit_number_with_unique_digits (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ n / 100 ≠ (n / 10) % 10 ∧ n / 100 ≠ n % 10 ∧ (n / 10) % 10 ≠ n % 10

theorem number_of_pieces_of_paper (n : ℕ) (k : ℕ) (h1 : three_digit_number_with_unique_digits n) (h2 : 2331 = k * n) : k = 9 :=
by
  sorry

end number_of_pieces_of_paper_l1052_105253


namespace count_perfect_squares_divisible_by_36_l1052_105243

theorem count_perfect_squares_divisible_by_36 :
  let N := 10000
  let max_square := 10^8
  let multiple := 36
  let valid_divisor := 1296
  let count_multiples := 277
  (∀ N : ℕ, N^2 < max_square → (∃ k : ℕ, N = k * multiple ∧ k < N)) → 
  ∃ cnt : ℕ, cnt = count_multiples := 
by {
  sorry
}

end count_perfect_squares_divisible_by_36_l1052_105243


namespace weights_sum_l1052_105246

theorem weights_sum (e f g h : ℕ) (h₁ : e + f = 280) (h₂ : f + g = 230) (h₃ : e + h = 300) : g + h = 250 := 
by 
  sorry

end weights_sum_l1052_105246


namespace negation_of_real_root_proposition_l1052_105275

theorem negation_of_real_root_proposition :
  (¬ ∃ m : ℝ, ∃ (x : ℝ), x^2 + m * x + 1 = 0) ↔ (∀ m : ℝ, ∀ (x : ℝ), x^2 + m * x + 1 ≠ 0) :=
by
  sorry

end negation_of_real_root_proposition_l1052_105275


namespace probability_all_six_draws_white_l1052_105248

theorem probability_all_six_draws_white :
  let total_balls := 14
  let white_balls := 7
  let single_draw_white_probability := (white_balls : ℚ) / total_balls
  (single_draw_white_probability ^ 6 = (1 : ℚ) / 64) :=
by
  sorry

end probability_all_six_draws_white_l1052_105248


namespace find_f_sqrt_5753_l1052_105251

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_sqrt_5753 (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x * y) = x * f y + y * f x)
  (h2 : ∀ x y : ℝ, f (x + y) = f (x * 1993) + f (y * 1993)) :
  f (Real.sqrt 5753) = 0 :=
sorry

end find_f_sqrt_5753_l1052_105251


namespace product_of_digits_l1052_105289

-- Define the conditions and state the theorem
theorem product_of_digits (A B : ℕ) (h1 : (10 * A + B) % 12 = 0) (h2 : A + B = 12) : A * B = 32 :=
  sorry

end product_of_digits_l1052_105289


namespace probability_of_two_same_color_l1052_105218

noncomputable def probability_at_least_two_same_color (reds whites blues greens : ℕ) (total_draws : ℕ) : ℚ :=
  have total_marbles := reds + whites + blues + greens
  let total_combinations := Nat.choose total_marbles total_draws
  let two_reds := Nat.choose reds 2 * (total_marbles - 2)
  let two_whites := Nat.choose whites 2 * (total_marbles - 2)
  let two_blues := Nat.choose blues 2 * (total_marbles - 2)
  let two_greens := Nat.choose greens 2 * (total_marbles - 2)
  
  let all_reds := Nat.choose reds 3
  let all_whites := Nat.choose whites 3
  let all_blues := Nat.choose blues 3
  let all_greens := Nat.choose greens 3
  
  let desired_outcomes := two_reds + two_whites + two_blues + two_greens +
                          all_reds + all_whites + all_blues + all_greens
                          
  (desired_outcomes : ℚ) / (total_combinations : ℚ)

theorem probability_of_two_same_color : probability_at_least_two_same_color 6 7 8 4 3 = 69 / 115 := 
by
  sorry

end probability_of_two_same_color_l1052_105218


namespace ratio_of_coconut_flavored_red_jelly_beans_l1052_105214

theorem ratio_of_coconut_flavored_red_jelly_beans :
  ∀ (total_jelly_beans jelly_beans_coconut_flavored : ℕ)
    (three_fourths_red : total_jelly_beans > 0 ∧ (3/4 : ℝ) * total_jelly_beans = 3 * (total_jelly_beans / 4))
    (h1 : jelly_beans_coconut_flavored = 750)
    (h2 : total_jelly_beans = 4000),
  (250 : ℝ)/(3000 : ℝ) = 1/4 :=
by
  intros total_jelly_beans jelly_beans_coconut_flavored three_fourths_red h1 h2
  sorry

end ratio_of_coconut_flavored_red_jelly_beans_l1052_105214


namespace reduced_rectangle_area_l1052_105227

theorem reduced_rectangle_area
  (w h : ℕ) (hw : w = 5) (hh : h = 7)
  (new_w : ℕ) (h_reduced_area : new_w = w - 2 ∧ new_w * h = 21)
  (reduced_h : ℕ) (hr : reduced_h = h - 1) :
  (new_w * reduced_h = 18) :=
by
  sorry

end reduced_rectangle_area_l1052_105227


namespace quadratic_polynomial_exists_l1052_105212

theorem quadratic_polynomial_exists (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ p : ℝ → ℝ, (∀ x, p x = (a^2 + ab + b^2 + ac + bc + c^2) * x^2 
                   - (a + b) * (b + c) * (a + c) * x 
                   + abc * (a + b + c))
              ∧ p a = a^4 
              ∧ p b = b^4 
              ∧ p c = c^4 := 
sorry

end quadratic_polynomial_exists_l1052_105212


namespace necessary_but_not_sufficient_cond_l1052_105226

open Set

variable {α : Type*} (A B C : Set α)

/-- Mathematical equivalent proof problem statement -/
theorem necessary_but_not_sufficient_cond (h1 : A ∪ B = C) (h2 : ¬ B ⊆ A) (hA : A.Nonempty) (hB : B.Nonempty) (hC : C.Nonempty) :
  (∀ x, x ∈ A → x ∈ C) ∧ (∃ y ∈ C, y ∉ A) :=
by
  sorry

end necessary_but_not_sufficient_cond_l1052_105226


namespace sum_of_squares_is_289_l1052_105254

theorem sum_of_squares_is_289 (x y : ℤ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 :=
by
  sorry

end sum_of_squares_is_289_l1052_105254


namespace sin_cos_solution_set_l1052_105233
open Real

theorem sin_cos_solution_set :
  {x : ℝ | ∃ k : ℤ, x = k * π + (-1)^k * (π / 6) - (π / 3)} =
  {x : ℝ | sin x + sqrt 3 * cos x = 1} :=
by sorry

end sin_cos_solution_set_l1052_105233


namespace checkered_rectangles_containing_one_gray_cell_l1052_105283

def total_number_of_rectangles_with_one_gray_cell :=
  let gray_cells := 40
  let blue_cells := 36
  let red_cells := 4
  
  let blue_rectangles_each := 4
  let red_rectangles_each := 8
  
  (blue_cells * blue_rectangles_each) + (red_cells * red_rectangles_each)

theorem checkered_rectangles_containing_one_gray_cell : total_number_of_rectangles_with_one_gray_cell = 176 :=
by 
  sorry

end checkered_rectangles_containing_one_gray_cell_l1052_105283


namespace problem_statement_l1052_105221

theorem problem_statement (p x : ℝ) (h : 0 ≤ p ∧ p ≤ 4) :
  (x^2 + p*x > 4*x + p - 3) ↔ (x > 3 ∨ x < -1) := by
sorry

end problem_statement_l1052_105221


namespace sum_two_numbers_l1052_105255

theorem sum_two_numbers (x y : ℝ) (h₁ : x * y = 16) (h₂ : 1 / x = 3 * (1 / y)) : x + y = 16 * Real.sqrt 3 / 3 :=
by
  -- Proof follows the steps outlined in the solution, but this is where the proof ends for now.
  sorry

end sum_two_numbers_l1052_105255


namespace limit_of_hours_for_overtime_l1052_105216

theorem limit_of_hours_for_overtime
  (R : Real) (O : Real) (total_compensation : Real) (total_hours_worked : Real) (L : Real)
  (hR : R = 14)
  (hO : O = 1.75 * R)
  (hTotalCompensation : total_compensation = 998)
  (hTotalHoursWorked : total_hours_worked = 57.88)
  (hEquation : (R * L) + ((total_hours_worked - L) * O) = total_compensation) :
  L = 40 := 
  sorry

end limit_of_hours_for_overtime_l1052_105216


namespace calculate_result_l1052_105207

def binary_op (x y : ℝ) : ℝ := x^2 + y^2

theorem calculate_result (h : ℝ) : binary_op (binary_op h h) (binary_op h h) = 8 * h^4 :=
by
  sorry

end calculate_result_l1052_105207


namespace population_ratio_l1052_105296

variables (Px Py Pz : ℕ)

theorem population_ratio (h1 : Py = 2 * Pz) (h2 : Px = 8 * Py) : Px / Pz = 16 :=
by
  sorry

end population_ratio_l1052_105296


namespace difference_max_min_students_l1052_105203

-- Definitions for problem conditions
def total_students : ℕ := 50
def shanghai_university_min : ℕ := 40
def shanghai_university_max : ℕ := 45
def shanghai_normal_university_min : ℕ := 16
def shanghai_normal_university_max : ℕ := 20

-- Lean statement for the math proof problem
theorem difference_max_min_students :
  (∀ (a b : ℕ), shanghai_university_min ≤ a ∧ a ≤ shanghai_university_max →
                shanghai_normal_university_min ≤ b ∧ b ≤ shanghai_normal_university_max →
                15 ≤ a + b - total_students ∧ a + b - total_students ≤ 15) →
  (∀ (a b : ℕ), shanghai_university_min ≤ a ∧ a ≤ shanghai_university_max →
                shanghai_normal_university_min ≤ b ∧ b ≤ shanghai_normal_university_max →
                6 ≤ a + b - total_students ∧ a + b - total_students ≤ 6) →
  (∃ M m : ℕ, 
    (M = 15) ∧ 
    (m = 6) ∧ 
    (M - m = 9)) :=
by
  sorry

end difference_max_min_students_l1052_105203


namespace total_food_pounds_l1052_105261

theorem total_food_pounds (chicken hamburger hot_dogs sides : ℕ) 
  (h1 : chicken = 16) 
  (h2 : hamburger = chicken / 2) 
  (h3 : hot_dogs = hamburger + 2) 
  (h4 : sides = hot_dogs / 2) : 
  chicken + hamburger + hot_dogs + sides = 39 := 
  by 
    sorry

end total_food_pounds_l1052_105261


namespace time_addition_correct_l1052_105220

def start_time := (3, 0, 0) -- Representing 3:00:00 PM as (hours, minutes, seconds)
def additional_time := (315, 78, 30) -- Representing additional time as (hours, minutes, seconds)

noncomputable def resulting_time (start add : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  let (sh, sm, ss) := start -- start hours, minutes, seconds
  let (ah, am, as) := add -- additional hours, minutes, seconds
  let total_seconds := ss + as
  let extra_minutes := total_seconds / 60
  let remaining_seconds := total_seconds % 60
  let total_minutes := sm + am + extra_minutes
  let extra_hours := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  let total_hours := sh + ah + extra_hours
  let resulting_hours := (total_hours % 12) -- Modulo 12 for wrap-around
  (resulting_hours, remaining_minutes, remaining_seconds)

theorem time_addition_correct :
  let (A, B, C) := resulting_time start_time additional_time
  A + B + C = 55 := by
  sorry

end time_addition_correct_l1052_105220


namespace a_number_M_middle_digit_zero_l1052_105260

theorem a_number_M_middle_digit_zero (d e f M : ℕ) (h1 : M = 36 * d + 6 * e + f)
  (h2 : M = 64 * f + 8 * e + d) (hd : d < 6) (he : e < 6) (hf : f < 6) : e = 0 :=
by sorry

end a_number_M_middle_digit_zero_l1052_105260


namespace most_precise_value_l1052_105211

def D := 3.27645
def error := 0.00518
def D_upper := D + error
def D_lower := D - error
def rounded_D_upper := Float.round (D_upper * 10) / 10
def rounded_D_lower := Float.round (D_lower * 10) / 10

theorem most_precise_value :
  rounded_D_upper = 3.3 ∧ rounded_D_lower = 3.3 → rounded_D_upper = 3.3 :=
by sorry

end most_precise_value_l1052_105211


namespace ratio_B_to_A_l1052_105295

theorem ratio_B_to_A (A B C : ℝ) 
  (hA : A = 1 / 21) 
  (hC : C = 2 * B) 
  (h_sum : A + B + C = 1 / 3) : 
  B / A = 2 := 
by 
  /- Proof goes here, but it's omitted as per instructions -/
  sorry

end ratio_B_to_A_l1052_105295


namespace johnny_hours_second_job_l1052_105208

theorem johnny_hours_second_job (x : ℕ) (h_eq : 5 * (69 + 10 * x) = 445) : x = 2 :=
by 
  -- The proof will go here, but we skip it as per the instructions
  sorry

end johnny_hours_second_job_l1052_105208


namespace pair_B_equal_l1052_105247

theorem pair_B_equal : (∀ x : ℝ, 4 * x^4 = |x|) :=
by sorry

end pair_B_equal_l1052_105247


namespace odometer_problem_l1052_105259

theorem odometer_problem
  (a b c : ℕ) -- a, b, c are natural numbers
  (h1 : 1 ≤ a) -- condition (a ≥ 1)
  (h2 : a + b + c ≤ 7) -- condition (a + b + c ≤ 7)
  (h3 : 99 * (c - a) % 55 = 0) -- 99(c - a) must be divisible by 55
  (h4 : 100 * a + 10 * b + c < 1000) -- ensuring a, b, c keeps numbers within 3-digits
  (h5 : 100 * c + 10 * b + a < 1000) -- ensuring a, b, c keeps numbers within 3-digits
  : a^2 + b^2 + c^2 = 37 := sorry

end odometer_problem_l1052_105259


namespace difference_cubed_divisible_by_27_l1052_105276

theorem difference_cubed_divisible_by_27 (a b : ℤ) :
    ((3 * a + 2) ^ 3 - (3 * b + 2) ^ 3) % 27 = 0 := 
by
  sorry

end difference_cubed_divisible_by_27_l1052_105276


namespace remaining_amount_eq_40_l1052_105257

-- Definitions and conditions
def initial_amount : ℕ := 100
def food_spending : ℕ := 20
def rides_spending : ℕ := 2 * food_spending
def total_spending : ℕ := food_spending + rides_spending

-- The proposition to be proved
theorem remaining_amount_eq_40 :
  initial_amount - total_spending = 40 :=
by
  sorry

end remaining_amount_eq_40_l1052_105257


namespace salt_solution_concentration_l1052_105252

theorem salt_solution_concentration (m x : ℝ) (h1 : m > 30) (h2 : (m * m / 100) = ((m - 20) / 100) * (m + 2 * x)) :
  x = 10 * m / (m + 20) :=
sorry

end salt_solution_concentration_l1052_105252


namespace cereal_discount_l1052_105232

theorem cereal_discount (milk_normal_cost milk_discounted_cost total_savings milk_quantity cereal_quantity: ℝ) 
  (total_milk_savings cereal_savings_per_box: ℝ) 
  (h1: milk_normal_cost = 3)
  (h2: milk_discounted_cost = 2)
  (h3: total_savings = 8)
  (h4: milk_quantity = 3)
  (h5: cereal_quantity = 5)
  (h6: total_milk_savings = milk_quantity * (milk_normal_cost - milk_discounted_cost)) 
  (h7: total_milk_savings + cereal_quantity * cereal_savings_per_box = total_savings):
  cereal_savings_per_box = 1 :=
by 
  sorry

end cereal_discount_l1052_105232


namespace intersection_A_B_l1052_105277

def A : Set ℝ := { x | 1 < x - 1 ∧ x - 1 ≤ 3 }
def B : Set ℝ := { 2, 3, 4 }

theorem intersection_A_B : A ∩ B = {3, 4} := 
by 
  sorry

end intersection_A_B_l1052_105277


namespace find_specific_linear_function_l1052_105229

-- Define the linear function with given conditions
def linear_function (k b : ℝ) (x : ℝ) := k * x + b

-- Define the condition that the point lies on the line
def passes_through (k b : ℝ) (x y : ℝ) := y = linear_function k b x

-- Define the condition that slope is negative
def slope_negative (k : ℝ) := k < 0

-- The specific function we want to prove
def specific_linear_function (x : ℝ) := -x + 1

-- The theorem to prove
theorem find_specific_linear_function : 
  ∃ (k b : ℝ), slope_negative k ∧ passes_through k b 0 1 ∧ 
  ∀ x, linear_function k b x = specific_linear_function x :=
by
  sorry

end find_specific_linear_function_l1052_105229


namespace expected_score_shooting_competition_l1052_105294

theorem expected_score_shooting_competition (hit_rate : ℝ)
  (miss_both_score : ℝ) (hit_one_score : ℝ) (hit_both_score : ℝ)
  (prob_0 : ℝ) (prob_10 : ℝ) (prob_15 : ℝ) :
  hit_rate = 4 / 5 →
  miss_both_score = 0 →
  hit_one_score = 10 →
  hit_both_score = 15 →
  prob_0 = (1 - 4 / 5) * (1 - 4 / 5) →
  prob_10 = 2 * (4 / 5) * (1 - 4 / 5) →
  prob_15 = (4 / 5) * (4 / 5) →
  (0 * prob_0 + 10 * prob_10 + 15 * prob_15) = 12.8 :=
by
  intros h_hit_rate h_miss_both_score h_hit_one_score h_hit_both_score
         h_prob_0 h_prob_10 h_prob_15
  sorry

end expected_score_shooting_competition_l1052_105294


namespace pipeA_fills_tank_in_56_minutes_l1052_105215

-- Define the relevant variables and conditions.
variable (t : ℕ) -- Time for Pipe A to fill the tank in minutes

-- Condition: Pipe B fills the tank 7 times faster than Pipe A
def pipeB_time (t : ℕ) := t / 7

-- Combined rate of Pipe A and Pipe B filling the tank in 7 minutes
def combined_rate (t : ℕ) := (1 / t) + (1 / pipeB_time t)

-- Given the combined rate fills the tank in 7 minutes
def combined_rate_equals (t : ℕ) := combined_rate t = 1 / 7

-- The proof statement
theorem pipeA_fills_tank_in_56_minutes (t : ℕ) (h : combined_rate_equals t) : t = 56 :=
sorry

end pipeA_fills_tank_in_56_minutes_l1052_105215


namespace shortest_distance_between_tracks_l1052_105237

noncomputable def rational_man_track (x y : ℝ) : Prop :=
x^2 + y^2 = 1

noncomputable def irrational_man_track (x y : ℝ) : Prop :=
(x + 1)^2 + y^2 = 9

noncomputable def shortest_distance : ℝ :=
0

theorem shortest_distance_between_tracks :
  ∀ (A B : ℝ × ℝ), 
  rational_man_track A.1 A.2 → 
  irrational_man_track B.1 B.2 → 
  dist A B = shortest_distance := sorry

end shortest_distance_between_tracks_l1052_105237


namespace arithmetic_sequence_ninth_term_l1052_105217

theorem arithmetic_sequence_ninth_term
  (a d : ℤ)
  (h1 : a + 2 * d = 23)
  (h2 : a + 5 * d = 29) :
  a + 8 * d = 35 :=
sorry

end arithmetic_sequence_ninth_term_l1052_105217


namespace complex_number_solution_l1052_105284

theorem complex_number_solution (z : ℂ) (h : z / Complex.I = 3 - Complex.I) : z = 1 + 3 * Complex.I :=
sorry

end complex_number_solution_l1052_105284


namespace rainfall_third_day_is_18_l1052_105222

-- Define the conditions including the rainfall for each day
def rainfall_first_day : ℕ := 4
def rainfall_second_day : ℕ := 5 * rainfall_first_day
def rainfall_third_day : ℕ := (rainfall_first_day + rainfall_second_day) - 6

-- Prove that the rainfall on the third day is 18 inches
theorem rainfall_third_day_is_18 : rainfall_third_day = 18 :=
by
  -- Use the definitions and directly state that the proof follows
  sorry

end rainfall_third_day_is_18_l1052_105222


namespace exact_sequence_a2007_l1052_105230

theorem exact_sequence_a2007 (a : ℕ → ℤ) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 0) 
  (exact : ∀ n m : ℕ, n > m → a n ^ 2 - a m ^ 2 = a (n - m) * a (n + m)) :
  a 2007 = -1 := 
sorry

end exact_sequence_a2007_l1052_105230


namespace number_of_whole_numbers_l1052_105234

theorem number_of_whole_numbers (x y : ℝ) (hx : 2 < x ∧ x < 3) (hy : 8 < y ∧ y < 9) : 
  ∃ (n : ℕ), n = 6 := by
  sorry

end number_of_whole_numbers_l1052_105234


namespace factorization1_factorization2_factorization3_l1052_105293

-- Problem 1
theorem factorization1 (a x : ℝ) : a * x^2 - 4 * a = a * (x + 2) * (x - 2) :=
sorry

-- Problem 2
theorem factorization2 (m x y : ℝ) : m * x^2 + 2 * m * x * y + m * y^2 = m * (x + y)^2 :=
sorry

-- Problem 3
theorem factorization3 (a b : ℝ) : (1 / 2) * a^2 - a * b + (1 / 2) * b^2 = (1 / 2) * (a - b)^2 :=
sorry

end factorization1_factorization2_factorization3_l1052_105293


namespace solve_for_z_l1052_105239

theorem solve_for_z (z : ℂ) (h : 5 - 3 * (I * z) = 3 + 5 * (I * z)) : z = I / 4 :=
sorry

end solve_for_z_l1052_105239


namespace michael_birth_year_l1052_105258

theorem michael_birth_year (first_AMC8_year : ℕ) (tenth_AMC8_year : ℕ) (age_during_tenth_AMC8 : ℕ) 
  (h1 : first_AMC8_year = 1985) (h2 : tenth_AMC8_year = (first_AMC8_year + 9)) (h3 : age_during_tenth_AMC8 = 15) :
  (tenth_AMC8_year - age_during_tenth_AMC8) = 1979 :=
by
  sorry

end michael_birth_year_l1052_105258


namespace total_number_of_crickets_l1052_105204

def initial_crickets : ℝ := 7.0
def additional_crickets : ℝ := 11.0
def total_crickets : ℝ := 18.0

theorem total_number_of_crickets :
  initial_crickets + additional_crickets = total_crickets :=
by
  sorry

end total_number_of_crickets_l1052_105204


namespace roger_bike_rides_total_l1052_105292

theorem roger_bike_rides_total 
  (r1 : ℕ) (h1 : r1 = 2) 
  (r2 : ℕ) (h2 : r2 = 5 * r1) 
  (r : ℕ) (h : r = r1 + r2) : 
  r = 12 := 
by
  sorry

end roger_bike_rides_total_l1052_105292


namespace sqrt_calculation_l1052_105224

theorem sqrt_calculation : Real.sqrt ((5: ℝ)^2 - (4: ℝ)^2 - (3: ℝ)^2) = 0 := 
by
  -- The proof is skipped
  sorry

end sqrt_calculation_l1052_105224


namespace cos_half_pi_plus_alpha_l1052_105238

open Real

noncomputable def alpha : ℝ := sorry

theorem cos_half_pi_plus_alpha :
  let a := (1 / 3, tan alpha)
  let b := (cos alpha, 1)
  ((1 / 3) / (cos alpha) = (tan alpha) / 1) →
  cos (pi / 2 + alpha) = -1 / 3 :=
by
  intros
  sorry

end cos_half_pi_plus_alpha_l1052_105238


namespace probability_roots_real_l1052_105245

-- Define the polynomial
def polynomial (b : ℝ) (x : ℝ) : ℝ :=
  x^4 + 3*b*x^3 + (3*b - 5)*x^2 + (-6*b + 4)*x - 3

-- Define the intervals for b
def interval_b1 := Set.Icc (-(15:ℝ)) (20:ℝ)
def interval_b2 := Set.Icc (-(15:ℝ)) (-2/3)
def interval_b3 := Set.Icc (4/3) (20:ℝ)

-- Calculate the lengths of the intervals
def length_interval (a b : ℝ) : ℝ := b - a

noncomputable def length_b1 := length_interval (-(15:ℝ)) (20:ℝ)
noncomputable def length_b2 := length_interval (-(15:ℝ)) (-2/3)
noncomputable def length_b3 := length_interval (4/3) (20:ℝ)
noncomputable def effective_length := length_b2 + length_b3

-- The probability is the ratio of effective lengths
noncomputable def probability := effective_length / length_b1

-- The theorem we want to prove
theorem probability_roots_real : probability = 33/35 :=
  sorry

end probability_roots_real_l1052_105245


namespace value_of_pq_s_l1052_105280

-- Definitions for the problem
def polynomial_divisible (p q s : ℚ) : Prop :=
  ∀ x : ℚ, (x^3 + 4 * x^2 + 16 * x + 8) ∣ (x^4 + 6 * x^3 + 8 * p * x^2 + 6 * q * x + s)

-- The main theorem statement to prove
theorem value_of_pq_s (p q s : ℚ) (h : polynomial_divisible p q s) : (p + q) * s = 332 / 3 :=
sorry -- Proof omitted

end value_of_pq_s_l1052_105280


namespace matt_without_calculator_5_minutes_l1052_105225

-- Define the conditions
def time_with_calculator (problems : Nat) : Nat := 2 * problems
def time_without_calculator (problems : Nat) (x : Nat) : Nat := x * problems
def time_saved (problems : Nat) (x : Nat) : Nat := time_without_calculator problems x - time_with_calculator problems

-- State the problem
theorem matt_without_calculator_5_minutes (x : Nat) :
  (time_saved 20 x = 60) → x = 5 := by
  sorry

end matt_without_calculator_5_minutes_l1052_105225


namespace var_of_or_l1052_105272

theorem var_of_or (p q : Prop) (h : ¬ (p ∧ q)) : (p ∨ q = true) ∨ (p ∨ q = false) :=
by
  sorry

end var_of_or_l1052_105272


namespace universal_quantifiers_are_true_l1052_105223

-- Declare the conditions as hypotheses
theorem universal_quantifiers_are_true :
  (∀ x : ℝ, x^2 - x + 0.25 ≥ 0) ∧ (∀ x : ℝ, x^2 + 2 * x + 2 > 0) :=
by
  sorry -- Proof skipped

end universal_quantifiers_are_true_l1052_105223


namespace find_geometric_sequence_term_l1052_105240

noncomputable def geometric_sequence_term (a q : ℝ) (n : ℕ) : ℝ := a * q ^ (n - 1)

theorem find_geometric_sequence_term (a : ℝ) (q : ℝ)
  (h1 : a * (1 - q ^ 3) / (1 - q) = 7)
  (h2 : a * (1 - q ^ 6) / (1 - q) = 63) :
  ∀ n : ℕ, geometric_sequence_term a q n = 2^(n-1) :=
by
  sorry

end find_geometric_sequence_term_l1052_105240


namespace find_d_plus_f_l1052_105250

noncomputable def a : ℂ := sorry
noncomputable def c : ℂ := sorry
noncomputable def e : ℂ := -2 * a - c
noncomputable def d : ℝ := sorry
noncomputable def f : ℝ := sorry

theorem find_d_plus_f (a c e : ℂ) (d f : ℝ) (h₁ : e = -2 * a - c) (h₂ : a.im + d + f = 4) (h₃ : a.re + c.re + e.re = 0) (h₄ : 2 + d + f = 4) : d + f = 2 :=
by
  -- proof goes here
  sorry

end find_d_plus_f_l1052_105250


namespace route_a_faster_by_8_minutes_l1052_105299

theorem route_a_faster_by_8_minutes :
  let route_a_distance := 8 -- miles
  let route_a_speed := 40 -- miles per hour
  let route_b_distance := 9 -- miles
  let route_b_speed := 45 -- miles per hour
  let route_b_stop := 8 -- minutes
  let time_route_a := route_a_distance / route_a_speed * 60 -- time in minutes
  let time_route_b := (route_b_distance / route_b_speed) * 60 + route_b_stop -- time in minutes
  time_route_b - time_route_a = 8 :=
by
  sorry

end route_a_faster_by_8_minutes_l1052_105299


namespace ratio_surface_area_cube_to_octahedron_l1052_105236

noncomputable def cube_side_length := 1

noncomputable def surface_area_cube (s : ℝ) : ℝ := 6 * s^2

noncomputable def edge_length_octahedron := 1

-- Surface area formula for a regular octahedron with side length e is 2 * sqrt(3) * e^2
noncomputable def surface_area_octahedron (e : ℝ) : ℝ := 2 * Real.sqrt 3 * e^2

-- Finally, we want to prove that the ratio of the surface area of the cube to that of the octahedron is sqrt(3)
theorem ratio_surface_area_cube_to_octahedron :
  surface_area_cube cube_side_length / surface_area_octahedron edge_length_octahedron = Real.sqrt 3 :=
by sorry

end ratio_surface_area_cube_to_octahedron_l1052_105236


namespace equation_descr_circle_l1052_105274

theorem equation_descr_circle : ∀ (x y : ℝ), (x - 0) ^ 2 + (y - 0) ^ 2 = 25 → ∃ (c : ℝ × ℝ) (r : ℝ), c = (0, 0) ∧ r = 5 ∧ ∀ (p : ℝ × ℝ), (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 :=
by
  sorry

end equation_descr_circle_l1052_105274


namespace find_sequence_l1052_105242

noncomputable def sequence_satisfies (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (1 / 2) * (a n + 1 / (a n))

theorem find_sequence (a : ℕ → ℝ) (S : ℕ → ℝ)
    (h_pos : ∀ n, 0 < a n)
    (h_S : sequence_satisfies a S) :
    ∀ n, a n = Real.sqrt n - Real.sqrt (n - 1) :=
sorry

end find_sequence_l1052_105242


namespace eccentricity_range_l1052_105263

-- We start with the given problem and conditions
variables {a c b : ℝ}
def C1 := ∀ x y, x^2 + 2 * c * x + y^2 = 0
def C2 := ∀ x y, x^2 - 2 * c * x + y^2 = 0
def ellipse := ∀ x y, x^2 / a^2 + y^2 / b^2 = 1

-- Ellipse semi-latus rectum condition and circles inside the ellipse
axiom h1 : c = b^2 / a
axiom h2 : a > 2 * c

-- Proving the range of the eccentricity
theorem eccentricity_range : 0 < c / a ∧ c / a < 1 / 2 :=
by
  sorry

end eccentricity_range_l1052_105263


namespace totalCoatsCollected_l1052_105201

-- Definitions from the conditions
def highSchoolCoats : Nat := 6922
def elementarySchoolCoats : Nat := 2515

-- Theorem that proves the total number of coats collected
theorem totalCoatsCollected : highSchoolCoats + elementarySchoolCoats = 9437 := by
  sorry

end totalCoatsCollected_l1052_105201


namespace equivalent_equation_l1052_105268

theorem equivalent_equation (x y : ℝ) 
  (x_ne_0 : x ≠ 0) (x_ne_3 : x ≠ 3) 
  (y_ne_0 : y ≠ 0) (y_ne_5 : y ≠ 5)
  (main_equation : (3 / x) + (4 / y) = 1 / 3) : 
  x = 9 * y / (y - 12) :=
sorry

end equivalent_equation_l1052_105268


namespace minimum_value_expression_l1052_105241

theorem minimum_value_expression (p q r s t u v w : ℝ) (h1 : p > 0) (h2 : q > 0) 
    (h3 : r > 0) (h4 : s > 0) (h5 : t > 0) (h6 : u > 0) (h7 : v > 0) (h8 : w > 0)
    (hpqrs : p * q * r * s = 16) (htuvw : t * u * v * w = 25) 
    (hptqu : p * t = q * u ∧ q * u = r * v ∧ r * v = s * w) : 
    (p * t) ^ 2 + (q * u) ^ 2 + (r * v) ^ 2 + (s * w) ^ 2 = 80 := sorry

end minimum_value_expression_l1052_105241


namespace student_range_exact_student_count_l1052_105262

-- Definitions for the conditions
def retail_price (x : ℕ) : ℕ := 240
def wholesale_price (x : ℕ) : ℕ := 260 / (x + 60)

def student_conditions (x : ℕ) : Prop := (x < 250) ∧ (x + 60 ≥ 250)
def wholesale_retail_equation (a : ℕ) : Prop := (240^2 / a) * 240 = (260 / (a+60)) * 288

-- Proofs of the required statements
theorem student_range (x : ℕ) (hc : student_conditions x) : 190 ≤ x ∧ x < 250 :=
by {
  sorry
}

theorem exact_student_count (a : ℕ) (heq : wholesale_retail_equation a) : a = 200 :=
by {
  sorry
}

end student_range_exact_student_count_l1052_105262


namespace div_by_17_l1052_105279

theorem div_by_17 (n : ℕ) (h : ¬ 17 ∣ n) : 17 ∣ (n^8 + 1) ∨ 17 ∣ (n^8 - 1) := 
by sorry

end div_by_17_l1052_105279


namespace loaves_of_bread_l1052_105206

variable (B : ℕ) -- Number of loaves of bread Erik bought
variable (total_money : ℕ := 86) -- Money given to Erik
variable (money_left : ℕ := 59) -- Money left after purchase
variable (cost_bread : ℕ := 3) -- Cost of each loaf of bread
variable (cost_oj : ℕ := 6) -- Cost of each carton of orange juice
variable (num_oj : ℕ := 3) -- Number of cartons of orange juice bought

theorem loaves_of_bread (h1 : total_money - money_left = num_oj * cost_oj + B * cost_bread) : B = 3 := 
by sorry

end loaves_of_bread_l1052_105206


namespace john_total_hours_l1052_105231

def wall_area (length : ℕ) (width : ℕ) := length * width

def total_area (num_walls : ℕ) (wall_area : ℕ) := num_walls * wall_area

def time_to_paint (area : ℕ) (time_per_square_meter : ℕ) := area * time_per_square_meter

def hours_to_minutes (hours : ℕ) := hours * 60

def total_hours (painting_time : ℕ) (spare_time : ℕ) := painting_time + spare_time

theorem john_total_hours 
  (length width num_walls time_per_square_meter spare_hours : ℕ) 
  (H_length : length = 2) 
  (H_width : width = 3) 
  (H_num_walls : num_walls = 5)
  (H_time_per_square_meter : time_per_square_meter = 10)
  (H_spare_hours : spare_hours = 5) :
  total_hours (time_to_paint (total_area num_walls (wall_area length width)) time_per_square_meter / hours_to_minutes 1) spare_hours = 10 := 
by 
    rw [H_length, H_width, H_num_walls, H_time_per_square_meter, H_spare_hours]
    sorry

end john_total_hours_l1052_105231


namespace hh_of_2_eq_91265_l1052_105235

def h (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 - x + 1

theorem hh_of_2_eq_91265 : h (h 2) = 91265 := by
  sorry

end hh_of_2_eq_91265_l1052_105235


namespace ze_age_conditions_l1052_105265

theorem ze_age_conditions 
  (z g t : ℕ)
  (h1 : z = 2 * g + 3 * t)
  (h2 : 2 * (z + 15) = 2 * (g + 15) + 3 * (t + 15))
  (h3 : 2 * (g + 15) = 3 * (t + 15)) :
  z = 45 ∧ t = 5 :=
by
  sorry

end ze_age_conditions_l1052_105265


namespace A_superset_C_l1052_105297

-- Definitions of the sets as given in the problem statement
def U : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {-1, 3}
def C : Set ℝ := {x | -1 < x ∧ x < 3}

-- Statement to be proved: A ⊇ C
theorem A_superset_C : A ⊇ C :=
by sorry

end A_superset_C_l1052_105297


namespace find_amount_l1052_105286

-- Definitions based on the conditions provided
def gain : ℝ := 0.70
def gain_percent : ℝ := 1.0

-- The theorem statement
theorem find_amount (h : gain_percent = 1) : ∀ (amount : ℝ), amount = gain / (gain_percent / 100) → amount = 70 :=
by
  intros amount h_calc
  sorry

end find_amount_l1052_105286


namespace value_of_k_l1052_105209

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

noncomputable def f (k x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

theorem value_of_k (k : ℝ) :
  is_even_function (f k) → k = 1 :=
by {
  sorry
}

end value_of_k_l1052_105209


namespace basketball_cards_price_l1052_105271

theorem basketball_cards_price :
  let toys_cost := 3 * 10
  let shirts_cost := 5 * 6
  let total_cost := 70
  let basketball_cards_cost := total_cost - (toys_cost + shirts_cost)
  let packs_of_cards := 2
  (basketball_cards_cost / packs_of_cards) = 5 :=
by
  sorry

end basketball_cards_price_l1052_105271


namespace geometric_sequence_logarithm_identity_l1052_105267

variable {a : ℕ+ → ℝ}

-- Assumptions
def common_ratio (a : ℕ+ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ+, a (n + 1) = r * a n

theorem geometric_sequence_logarithm_identity
  (r : ℝ)
  (hr : r = -Real.sqrt 2)
  (h : common_ratio a r) :
  Real.log (a 2017)^2 - Real.log (a 2016)^2 = Real.log 2 :=
by
  sorry

end geometric_sequence_logarithm_identity_l1052_105267


namespace cherries_per_quart_of_syrup_l1052_105285

-- Definitions based on conditions
def time_to_pick_cherries : ℚ := 2
def cherries_picked_in_time : ℚ := 300
def time_to_make_syrup : ℚ := 3
def total_time_for_all_syrup : ℚ := 33
def total_quarts : ℚ := 9

-- Derivation of how many cherries are needed per quart
theorem cherries_per_quart_of_syrup : 
  (cherries_picked_in_time / time_to_pick_cherries) * (total_time_for_all_syrup - total_quarts * time_to_make_syrup) / total_quarts = 100 :=
by
  repeat { sorry }

end cherries_per_quart_of_syrup_l1052_105285


namespace alpha_plus_beta_l1052_105210

noncomputable def alpha_beta (α β : ℝ) : Prop :=
  ∀ x : ℝ, ((x - α) / (x + β)) = ((x^2 - 54 * x + 621) / (x^2 + 42 * x - 1764))

theorem alpha_plus_beta : ∃ α β : ℝ, α + β = 86 ∧ alpha_beta α β :=
by
  sorry

end alpha_plus_beta_l1052_105210


namespace no_real_solution_ratio_l1052_105249

theorem no_real_solution_ratio (x : ℝ) : (x + 3) / (2 * x + 5) = (5 * x + 4) / (8 * x + 5) → false :=
by {
  sorry
}

end no_real_solution_ratio_l1052_105249


namespace rectangle_area_l1052_105264

structure Rectangle where
  length : ℕ    -- Length of the rectangle in cm
  width : ℕ     -- Width of the rectangle in cm
  perimeter : ℕ -- Perimeter of the rectangle in cm
  h : length = width + 4 -- Distance condition from the diagonal intersection

theorem rectangle_area (r : Rectangle) (h_perim : r.perimeter = 56) : r.length * r.width = 192 := by
  sorry

end rectangle_area_l1052_105264


namespace number_of_elements_less_than_2004_l1052_105219

theorem number_of_elements_less_than_2004 (f : ℕ → ℕ) 
    (h0 : f 0 = 0) 
    (h1 : ∀ n : ℕ, (f (2 * n + 1)) ^ 2 - (f (2 * n)) ^ 2 = 6 * f n + 1) 
    (h2 : ∀ n : ℕ, f (2 * n) > f n) 
  : ∃ m : ℕ,  m = 128 ∧ ∀ x : ℕ, f x < 2004 → x < m := sorry

end number_of_elements_less_than_2004_l1052_105219
