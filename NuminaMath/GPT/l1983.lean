import Mathlib

namespace NUMINAMATH_GPT_jeans_price_increase_l1983_198383

theorem jeans_price_increase (M R C : ℝ) (hM : M = 100) 
  (hR : R = M * 1.4)
  (hC : C = R * 1.1) : 
  (C - M) / M * 100 = 54 :=
by
  sorry

end NUMINAMATH_GPT_jeans_price_increase_l1983_198383


namespace NUMINAMATH_GPT_production_rate_l1983_198301

theorem production_rate (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x * x * x = x) → (y * y * z) / x^2 = y^2 * z / x^2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_production_rate_l1983_198301


namespace NUMINAMATH_GPT_total_eggs_collected_by_all_four_l1983_198332

def benjamin_eggs := 6
def carla_eggs := 3 * benjamin_eggs
def trisha_eggs := benjamin_eggs - 4
def david_eggs := 2 * trisha_eggs

theorem total_eggs_collected_by_all_four :
  benjamin_eggs + carla_eggs + trisha_eggs + david_eggs = 30 := by
  sorry

end NUMINAMATH_GPT_total_eggs_collected_by_all_four_l1983_198332


namespace NUMINAMATH_GPT_hyperbola_representation_iff_l1983_198330

theorem hyperbola_representation_iff (m : ℝ) :
  (∃ x y : ℝ, (x^2) / (2 + m) - (y^2) / (m + 1) = 1) ↔ (m > -1 ∨ m < -2) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_representation_iff_l1983_198330


namespace NUMINAMATH_GPT_candy_distribution_l1983_198353

theorem candy_distribution (candy_total friends : ℕ) (candies : List ℕ) :
  candy_total = 47 ∧ friends = 5 ∧ List.length candies = friends ∧
  (∀ n ∈ candies, n = 9) → (47 % 5 = 2) :=
by
  sorry

end NUMINAMATH_GPT_candy_distribution_l1983_198353


namespace NUMINAMATH_GPT_solve_for_x_l1983_198399

theorem solve_for_x (x : ℝ) (h : 0.4 * x = (1 / 3) * x + 110) : x = 1650 :=
by sorry

end NUMINAMATH_GPT_solve_for_x_l1983_198399


namespace NUMINAMATH_GPT_xiaoming_mirrored_time_l1983_198331

-- Define the condition: actual time is 7:10 AM.
def actual_time : (ℕ × ℕ) := (7, 10)

-- Define a function to compute the mirrored time given an actual time.
def mirror_time (h m : ℕ) : (ℕ × ℕ) :=
  let mirrored_minute := if m = 0 then 0 else 60 - m
  let mirrored_hour := if m = 0 then if h = 12 then 12 else (12 - h) % 12
                        else if h = 12 then 11 else (11 - h) % 12
  (mirrored_hour, mirrored_minute)

-- Our goal is to verify that the mirrored time of 7:10 is 4:50.
theorem xiaoming_mirrored_time : mirror_time 7 10 = (4, 50) :=
by
  -- Proof will verify that mirror_time (7, 10) evaluates to (4, 50).
  sorry

end NUMINAMATH_GPT_xiaoming_mirrored_time_l1983_198331


namespace NUMINAMATH_GPT_motorist_travel_distance_l1983_198328

def total_distance_traveled (time_first_half time_second_half speed_first_half speed_second_half : ℕ) : ℕ :=
  (speed_first_half * time_first_half) + (speed_second_half * time_second_half)

theorem motorist_travel_distance :
  total_distance_traveled 3 3 60 48 = 324 :=
by sorry

end NUMINAMATH_GPT_motorist_travel_distance_l1983_198328


namespace NUMINAMATH_GPT_initial_percentage_filled_l1983_198352

theorem initial_percentage_filled {P : ℝ} 
  (h1 : 45 + (P / 100) * 100 = (3 / 4) * 100) : 
  P = 30 := by
  sorry

end NUMINAMATH_GPT_initial_percentage_filled_l1983_198352


namespace NUMINAMATH_GPT_sin_70_equals_1_minus_2a_squared_l1983_198337

variable (a : ℝ)

theorem sin_70_equals_1_minus_2a_squared (h : Real.sin (10 * Real.pi / 180) = a) :
  Real.sin (70 * Real.pi / 180) = 1 - 2 * a^2 := 
sorry

end NUMINAMATH_GPT_sin_70_equals_1_minus_2a_squared_l1983_198337


namespace NUMINAMATH_GPT_clea_escalator_time_standing_l1983_198339

noncomputable def escalator_time (c : ℕ) : ℝ :=
  let s := (7 * c) / 5
  let d := 72 * c
  let t := d / s
  t

theorem clea_escalator_time_standing (c : ℕ) (h1 : 72 * c = 72 * c) (h2 : 30 * (c + (7 * c) / 5) = 72 * c): escalator_time c = 51 :=
by
  sorry

end NUMINAMATH_GPT_clea_escalator_time_standing_l1983_198339


namespace NUMINAMATH_GPT_tan_seven_pi_over_four_l1983_198390

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 :=
by
  sorry

end NUMINAMATH_GPT_tan_seven_pi_over_four_l1983_198390


namespace NUMINAMATH_GPT_function_passes_through_fixed_point_l1983_198341

noncomputable def given_function (a : ℝ) (x : ℝ) : ℝ :=
  a^(x - 1) + 7

theorem function_passes_through_fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  given_function a 1 = 8 :=
by
  sorry

end NUMINAMATH_GPT_function_passes_through_fixed_point_l1983_198341


namespace NUMINAMATH_GPT_denominator_of_fraction_l1983_198351

theorem denominator_of_fraction (n : ℕ) (h1 : n = 20) (h2 : num = 35) (dec_value : ℝ) (h3 : dec_value = 2 / 10^n) : denom = 175 * 10^20 :=
by
  sorry

end NUMINAMATH_GPT_denominator_of_fraction_l1983_198351


namespace NUMINAMATH_GPT_min_value_l1983_198312

variable (d : ℕ) (a_n S_n : ℕ → ℕ)
variable (a1 : ℕ) (H1 : d ≠ 0)
variable (H2 : a1 = 1)
variable (H3 : (a_n 3)^2 = a1 * (a_n 13))
variable (H4 : a_n n = a1 + (n - 1) * d)
variable (H5 : S_n n = (n * (a1 + a_n n)) / 2)

theorem min_value (n : ℕ) (Hn : 1 ≤ n) : 
  ∃ n, ∀ m, 1 ≤ m → (2 * S_n n + 16) / (a_n n + 3) ≥ (2 * S_n m + 16) / (a_n m + 3) ∧ (2 * S_n n + 16) / (a_n n + 3) = 4 :=
sorry

end NUMINAMATH_GPT_min_value_l1983_198312


namespace NUMINAMATH_GPT_candidates_appeared_equal_l1983_198361

theorem candidates_appeared_equal 
  (A_candidates B_candidates : ℕ)
  (A_selected B_selected : ℕ)
  (h1 : 6 * A_candidates = A_selected * 100)
  (h2 : 7 * B_candidates = B_selected * 100)
  (h3 : B_selected = A_selected + 83)
  (h4 : A_candidates = B_candidates):
  A_candidates = 8300 :=
by
  sorry

end NUMINAMATH_GPT_candidates_appeared_equal_l1983_198361


namespace NUMINAMATH_GPT_friend_time_to_read_book_l1983_198385

-- Define the conditions and variables
def my_reading_time : ℕ := 240 -- 4 hours in minutes
def speed_ratio : ℕ := 2 -- I read at half the speed of my friend

-- Define the variable for my friend's reading time which we need to find
def friend_reading_time : ℕ := my_reading_time / speed_ratio

-- The theorem statement that given the conditions, the friend's reading time is 120 minutes
theorem friend_time_to_read_book : friend_reading_time = 120 := sorry

end NUMINAMATH_GPT_friend_time_to_read_book_l1983_198385


namespace NUMINAMATH_GPT_evaluate_expression_l1983_198368

theorem evaluate_expression : (528 * 528) - (527 * 529) = 1 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1983_198368


namespace NUMINAMATH_GPT_local_extrema_l1983_198363

-- Defining the function y = 1 + 3x - x^3
def y (x : ℝ) : ℝ := 1 + 3 * x - x ^ 3

-- Statement of the problem to be proved
theorem local_extrema :
  (∃ x : ℝ, x = -1 ∧ y x = -1 ∧ ∀ ε > 0, ∃ δ > 0, ∀ z, abs (z + 1) < δ → y z ≥ y (-1)) ∧
  (∃ x : ℝ, x = 1 ∧ y x = 3 ∧ ∀ ε > 0, ∃ δ > 0, ∀ z, abs (z - 1) < δ → y z ≤ y 1) :=
by sorry

end NUMINAMATH_GPT_local_extrema_l1983_198363


namespace NUMINAMATH_GPT_algebraic_expression_value_l1983_198343

theorem algebraic_expression_value (x : ℝ) (h : x^2 - 2*x - 2 = 0) : 3*x^2 - 6*x + 9 = 15 := by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1983_198343


namespace NUMINAMATH_GPT_find_m_l1983_198338

def f (x m : ℝ) : ℝ := x ^ 2 - 3 * x + m
def g (x m : ℝ) : ℝ := 2 * x ^ 2 - 6 * x + 5 * m

theorem find_m (m : ℝ) (h : 3 * f 3 m = 2 * g 3 m) : m = 0 :=
by sorry

end NUMINAMATH_GPT_find_m_l1983_198338


namespace NUMINAMATH_GPT_density_of_second_part_l1983_198320

theorem density_of_second_part (ρ₁ : ℝ) (V₁ V : ℝ) (m₁ m : ℝ) (h₁ : ρ₁ = 2700) (h₂ : V₁ = 0.25 * V) (h₃ : m₁ = 0.4 * m) :
  (0.6 * m) / (0.75 * V) = 2160 :=
by
  --- Proof omitted
  sorry

end NUMINAMATH_GPT_density_of_second_part_l1983_198320


namespace NUMINAMATH_GPT_estimated_red_balls_l1983_198302

-- Definitions based on conditions
def total_balls : ℕ := 15
def black_ball_frequency : ℝ := 0.6
def red_ball_frequency : ℝ := 1 - black_ball_frequency

-- Theorem stating the proof problem
theorem estimated_red_balls :
  (total_balls : ℝ) * red_ball_frequency = 6 := by
  sorry

end NUMINAMATH_GPT_estimated_red_balls_l1983_198302


namespace NUMINAMATH_GPT_integer_solutions_count_correct_1992_integer_solutions_count_correct_1993_integer_solutions_count_correct_1994_l1983_198355

noncomputable def count_integer_solutions (n : ℕ) : ℕ :=
  if n = 1992 then 90
  else if n = 1993 then 6
  else if n = 1994 then 6
  else 0

theorem integer_solutions_count_correct_1992 :
  count_integer_solutions 1992 = 90 :=
by
  sorry

theorem integer_solutions_count_correct_1993 :
  count_integer_solutions 1993 = 6 :=
by
  sorry

theorem integer_solutions_count_correct_1994 :
  count_integer_solutions 1994 = 6 :=
by
  sorry

example :
  count_integer_solutions 1992 = 90 ∧
  count_integer_solutions 1993 = 6 ∧
  count_integer_solutions 1994 = 6 :=
by
  exact ⟨integer_solutions_count_correct_1992, integer_solutions_count_correct_1993, integer_solutions_count_correct_1994⟩

end NUMINAMATH_GPT_integer_solutions_count_correct_1992_integer_solutions_count_correct_1993_integer_solutions_count_correct_1994_l1983_198355


namespace NUMINAMATH_GPT_marathon_problem_l1983_198389

-- Defining the given conditions in the problem.
def john_position_right := 28
def john_position_left := 42
def mike_ahead := 10

-- Define total participants.
def total_participants := john_position_right + john_position_left - 1

-- Define Mike's positions based on the given conditions.
def mike_position_left := john_position_left - mike_ahead
def mike_position_right := john_position_right - mike_ahead

-- Proposition combining all the facts.
theorem marathon_problem :
  total_participants = 69 ∧ mike_position_left = 32 ∧ mike_position_right = 18 := by 
     sorry

end NUMINAMATH_GPT_marathon_problem_l1983_198389


namespace NUMINAMATH_GPT_sqrt_3_between_inequalities_l1983_198333

theorem sqrt_3_between_inequalities (n : ℕ) (h1 : 1 + (3 : ℝ) / (n + 1) < Real.sqrt 3) (h2 : Real.sqrt 3 < 1 + (3 : ℝ) / n) : n = 4 := 
sorry

end NUMINAMATH_GPT_sqrt_3_between_inequalities_l1983_198333


namespace NUMINAMATH_GPT_smallest_base10_integer_exists_l1983_198334

theorem smallest_base10_integer_exists : ∃ (n a b : ℕ), a > 2 ∧ b > 2 ∧ n = 1 * a + 3 ∧ n = 3 * b + 1 ∧ n = 10 := by
  sorry

end NUMINAMATH_GPT_smallest_base10_integer_exists_l1983_198334


namespace NUMINAMATH_GPT_trigonometric_identity_l1983_198379

theorem trigonometric_identity 
  (θ : ℝ) 
  (h : Real.tan θ = 2) : 
  (Real.cos θ - Real.sin θ) / (Real.cos θ + Real.sin θ) = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1983_198379


namespace NUMINAMATH_GPT_log_expression_evaluation_l1983_198306

theorem log_expression_evaluation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (Real.log (x^2) / Real.log (y^8)) * (Real.log (y^3) / Real.log (x^7)) * (Real.log (x^4) / Real.log (y^5)) * (Real.log (y^5) / Real.log (x^4)) * (Real.log (x^7) / Real.log (y^3)) =
  (1 / 4) * (Real.log x / Real.log y) := 
by
  sorry

end NUMINAMATH_GPT_log_expression_evaluation_l1983_198306


namespace NUMINAMATH_GPT_Yoque_borrowed_150_l1983_198398

noncomputable def Yoque_borrowed_amount (X : ℝ) : Prop :=
  1.10 * X = 11 * 15

theorem Yoque_borrowed_150 (X : ℝ) : Yoque_borrowed_amount X → X = 150 :=
by
  -- proof will be filled in
  sorry

end NUMINAMATH_GPT_Yoque_borrowed_150_l1983_198398


namespace NUMINAMATH_GPT_algebra_inequality_l1983_198382

variable {x y z : ℝ}

theorem algebra_inequality
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x^3 * (y^2 + z^2)^2 + y^3 * (z^2 + x^2)^2 + z^3 * (x^2 + y^2)^2
  ≥ x * y * z * (x * y * (x + y)^2 + y * z * (y + z)^2 + z * x * (z + x)^2) :=
sorry

end NUMINAMATH_GPT_algebra_inequality_l1983_198382


namespace NUMINAMATH_GPT_correct_relative_pronoun_used_l1983_198377

theorem correct_relative_pronoun_used (option : String) :
  (option = "where") ↔
  "Giving is a universal opportunity " ++ option ++ " regardless of your age, profession, religion, and background, you have the capacity to create change." =
  "Giving is a universal opportunity where regardless of your age, profession, religion, and background, you have the capacity to create change." :=
by
  sorry

end NUMINAMATH_GPT_correct_relative_pronoun_used_l1983_198377


namespace NUMINAMATH_GPT_midpoint_ellipse_trajectory_l1983_198350

theorem midpoint_ellipse_trajectory (x y x0 y0 x1 y1 x2 y2 : ℝ) :
  (x0 / 12) + (y0 / 8) = 1 →
  (x1^2 / 24) + (y1^2 / 16) = 1 →
  (x2^2 / 24) + (y2^2 / 16) = 1 →
  x = (x1 + x2) / 2 →
  y = (y1 + y2) / 2 →
  ∃ x y, ((x - 1)^2 / (5 / 2)) + ((y - 1)^2 / (5 / 3)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_ellipse_trajectory_l1983_198350


namespace NUMINAMATH_GPT_B_squared_B_sixth_l1983_198324

noncomputable def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![0, 3], ![2, -1]]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℤ :=
  1

theorem B_squared :
  B * B = 3 * B - I := by
  sorry

theorem B_sixth :
  B^6 = 84 * B - 44 * I := by
  sorry

end NUMINAMATH_GPT_B_squared_B_sixth_l1983_198324


namespace NUMINAMATH_GPT_evaluate_power_l1983_198317

theorem evaluate_power (n : ℕ) (h : 3^(2 * n) = 81) : 9^(n + 1) = 729 :=
by sorry

end NUMINAMATH_GPT_evaluate_power_l1983_198317


namespace NUMINAMATH_GPT_train_pass_jogger_time_l1983_198360

noncomputable def jogger_speed_km_per_hr : ℝ := 9
noncomputable def train_speed_km_per_hr : ℝ := 60
noncomputable def initial_distance_m : ℝ := 350
noncomputable def train_length_m : ℝ := 250

noncomputable def relative_speed_m_per_s : ℝ := 
  ((train_speed_km_per_hr - jogger_speed_km_per_hr) * 1000) / 3600

noncomputable def total_distance_m : ℝ := initial_distance_m + train_length_m

noncomputable def time_to_pass_s : ℝ := total_distance_m / relative_speed_m_per_s

theorem train_pass_jogger_time :
  abs (time_to_pass_s - 42.35) < 0.01 :=
by 
  sorry

end NUMINAMATH_GPT_train_pass_jogger_time_l1983_198360


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1983_198376

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (3 / (a - 1) + (a - 3) / (a^2 - 1)) / (a / (a + 1)) = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1983_198376


namespace NUMINAMATH_GPT_range_of_a_l1983_198394

open Set

theorem range_of_a (a : ℝ) (h1 : (∃ x, a^x > 1 ∧ x < 0) ∨ (∀ x, ax^2 - x + a ≥ 0))
  (h2 : ¬((∃ x, a^x > 1 ∧ x < 0) ∧ (∀ x, ax^2 - x + a ≥ 0))) :
  a ∈ (Ioo 0 (1/2)) ∪ (Ici 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l1983_198394


namespace NUMINAMATH_GPT_product_mnp_l1983_198365

theorem product_mnp (a x y z c : ℕ) (m n p : ℕ) :
  (a ^ 8 * x * y * z - a ^ 7 * y * z - a ^ 6 * x * z = a ^ 5 * (c ^ 5 - 1) ∧
   (a ^ m * x * z - a ^ n) * (a ^ p * y * z - a ^ 3) = a ^ 5 * c ^ 5) →
  m = 5 ∧ n = 4 ∧ p = 3 ∧ m * n * p = 60 :=
by
  sorry

end NUMINAMATH_GPT_product_mnp_l1983_198365


namespace NUMINAMATH_GPT_problem_part1_problem_part2_problem_part3_l1983_198367

open Set

noncomputable def U := ℝ
noncomputable def A := { x : ℝ | x < -4 ∨ x > 1 }
noncomputable def B := { x : ℝ | -3 ≤ x - 1 ∧ x - 1 ≤ 2 }

theorem problem_part1 :
  A ∩ B = { x : ℝ | 1 < x ∧ x ≤ 3 } := by sorry

theorem problem_part2 :
  compl A ∪ compl B = { x : ℝ | x ≤ 1 ∨ x > 3 } := by sorry

theorem problem_part3 (k : ℝ) :
  { x : ℝ | 2 * k - 1 ≤ x ∧ x ≤ 2 * k + 1 } ⊆ A → k > 1 := by sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_problem_part3_l1983_198367


namespace NUMINAMATH_GPT_area_of_TURS_eq_area_of_PQRS_l1983_198369

-- Definition of the rectangle PQRS
structure Rectangle where
  length : ℕ
  width : ℕ
  area : ℕ

-- Definition of the trapezoid TURS
structure Trapezoid where
  base1 : ℕ
  base2 : ℕ
  height : ℕ
  area : ℕ

-- Condition: PQRS is a rectangle whose area is 20 square units
def PQRS : Rectangle := { length := 5, width := 4, area := 20 }

-- Question: Prove the area of TURS equals area of PQRS
theorem area_of_TURS_eq_area_of_PQRS (TURS_area : ℕ) : TURS_area = PQRS.area :=
  sorry

end NUMINAMATH_GPT_area_of_TURS_eq_area_of_PQRS_l1983_198369


namespace NUMINAMATH_GPT_simplify_expression_l1983_198309

theorem simplify_expression: 3 * Real.sqrt 48 - 6 * Real.sqrt (1 / 3) + (Real.sqrt 3 - 1) ^ 2 = 8 * Real.sqrt 3 + 4 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1983_198309


namespace NUMINAMATH_GPT_proof_of_diagonals_and_angles_l1983_198344

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

def sum_of_internal_angles (n : ℕ) : ℕ := (n - 2) * 180

theorem proof_of_diagonals_and_angles :
  let p_diagonals := number_of_diagonals 5
  let o_diagonals := number_of_diagonals 8
  let total_diagonals := p_diagonals + o_diagonals
  let p_internal_angles := sum_of_internal_angles 5
  let o_internal_angles := sum_of_internal_angles 8
  let total_internal_angles := p_internal_angles + o_internal_angles
  total_diagonals = 25 ∧ total_internal_angles = 1620 :=
by
  sorry

end NUMINAMATH_GPT_proof_of_diagonals_and_angles_l1983_198344


namespace NUMINAMATH_GPT_negation_of_proposition_l1983_198316

theorem negation_of_proposition :
  (¬ (∃ x_0 : ℝ, x_0 ≤ 0 ∧ x_0^2 ≥ 0)) ↔ (∀ x : ℝ, x ≤ 0 → x^2 < 0) :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l1983_198316


namespace NUMINAMATH_GPT_relatively_prime_powers_of_two_l1983_198327

theorem relatively_prime_powers_of_two (a : ℤ) (h₁ : a % 2 = 1) (n m : ℕ) (h₂ : n ≠ m) :
  Int.gcd (a^(2^n) + 2^(2^n)) (a^(2^m) + 2^(2^m)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_relatively_prime_powers_of_two_l1983_198327


namespace NUMINAMATH_GPT_probability_quadrant_l1983_198374

theorem probability_quadrant
    (r : ℝ) (x y : ℝ)
    (h : x^2 + y^2 ≤ r^2) :
    (∃ p : ℝ, p = (1 : ℚ)/4) :=
by
  sorry

end NUMINAMATH_GPT_probability_quadrant_l1983_198374


namespace NUMINAMATH_GPT_total_attendance_l1983_198370

theorem total_attendance (A C : ℕ) (ticket_sales : ℕ) (adult_ticket_cost child_ticket_cost : ℕ) (total_collected : ℕ)
    (h1 : C = 18) (h2 : ticket_sales = 50) (h3 : adult_ticket_cost = 8) (h4 : child_ticket_cost = 1)
    (h5 : ticket_sales = adult_ticket_cost * A + child_ticket_cost * C) :
    A + C = 22 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_attendance_l1983_198370


namespace NUMINAMATH_GPT_hypotenuse_length_l1983_198305

variables (a b c : ℝ)

-- Definitions from conditions
def right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def sum_of_squares_is_2000 (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 = 2000

def perimeter_is_60 (a b c : ℝ) : Prop :=
  a + b + c = 60

theorem hypotenuse_length (a b c : ℝ)
  (h1 : right_angled_triangle a b c)
  (h2 : sum_of_squares_is_2000 a b c)
  (h3 : perimeter_is_60 a b c) :
  c = 10 * Real.sqrt 10 :=
sorry

end NUMINAMATH_GPT_hypotenuse_length_l1983_198305


namespace NUMINAMATH_GPT_base3_to_base5_conversion_l1983_198315

-- Define the conversion from base 3 to decimal
def base3_to_decimal (n : ℕ) : ℕ :=
  n % 10 * 1 + (n / 10 % 10) * 3 + (n / 100 % 10) * 9 + (n / 1000 % 10) * 27 + (n / 10000 % 10) * 81

-- Define the conversion from decimal to base 5
def decimal_to_base5 (n : ℕ) : ℕ :=
  n % 5 + (n / 5 % 5) * 10 + (n / 25 % 5) * 100

-- The initial number in base 3
def initial_number_base3 : ℕ := 10121

-- The final number in base 5
def final_number_base5 : ℕ := 342

-- The theorem that states the conversion result
theorem base3_to_base5_conversion :
  decimal_to_base5 (base3_to_decimal initial_number_base3) = final_number_base5 :=
by
  sorry

end NUMINAMATH_GPT_base3_to_base5_conversion_l1983_198315


namespace NUMINAMATH_GPT_ice_cream_vendor_l1983_198342

theorem ice_cream_vendor (M : ℕ) (h3 : 50 - (3 / 5) * 50 = 20) (h4 : (2 / 3) * M = 2 * M / 3) 
  (h5 : (50 - 30) + M - (2 * M / 3) = 38) :
  M = 12 :=
by
  sorry

end NUMINAMATH_GPT_ice_cream_vendor_l1983_198342


namespace NUMINAMATH_GPT_prime_pairs_solution_l1983_198359

theorem prime_pairs_solution (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  7 * p * q^2 + p = q^3 + 43 * p^3 + 1 ↔ (p = 2 ∧ q = 7) :=
by
  sorry

end NUMINAMATH_GPT_prime_pairs_solution_l1983_198359


namespace NUMINAMATH_GPT_problems_per_page_l1983_198311

theorem problems_per_page (pages_math pages_reading total_problems x : ℕ) (h1 : pages_math = 2) (h2 : pages_reading = 4) (h3 : total_problems = 30) : 
  (pages_math + pages_reading) * x = total_problems → x = 5 := by
  sorry

end NUMINAMATH_GPT_problems_per_page_l1983_198311


namespace NUMINAMATH_GPT_response_percentage_is_50_l1983_198397

-- Define the initial number of friends
def initial_friends := 100

-- Define the number of friends Mark kept initially
def kept_friends := 40

-- Define the number of friends Mark contacted
def contacted_friends := initial_friends - kept_friends

-- Define the number of friends Mark has after some responded
def remaining_friends := 70

-- Define the number of friends who responded to Mark's contact
def responded_friends := remaining_friends - kept_friends

-- Define the percentage of contacted friends who responded
def response_percentage := (responded_friends / contacted_friends) * 100

theorem response_percentage_is_50 :
  response_percentage = 50 := by
  sorry

end NUMINAMATH_GPT_response_percentage_is_50_l1983_198397


namespace NUMINAMATH_GPT_line_through_point_with_equal_intercepts_l1983_198356

theorem line_through_point_with_equal_intercepts (x y : ℝ) :
  (∃ b : ℝ, 3 * x + y = 0) ∨ (∃ b : ℝ, x - y + 4 = 0) ∨ (∃ b : ℝ, x + y - 2 = 0) :=
  sorry

end NUMINAMATH_GPT_line_through_point_with_equal_intercepts_l1983_198356


namespace NUMINAMATH_GPT_range_of_b_l1983_198304

theorem range_of_b (M : Set (ℝ × ℝ)) (N : ℝ → ℝ → Set (ℝ × ℝ)) :
  (∀ m : ℝ, (∃ x y : ℝ, (x, y) ∈ M ∧ (x, y) ∈ (N m b))) ↔ b ∈ Set.Icc (- Real.sqrt 6 / 2) (Real.sqrt 6 / 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_l1983_198304


namespace NUMINAMATH_GPT_A_is_9_years_older_than_B_l1983_198391

-- Define the conditions
variables (A_years B_years : ℕ)

def given_conditions : Prop :=
  B_years = 39 ∧ A_years + 10 = 2 * (B_years - 10)

-- Theorem to prove the correct answer
theorem A_is_9_years_older_than_B (h : given_conditions A_years B_years) : A_years - B_years = 9 :=
by
  sorry

end NUMINAMATH_GPT_A_is_9_years_older_than_B_l1983_198391


namespace NUMINAMATH_GPT_product_of_undefined_x_l1983_198319

-- Define the quadratic equation condition
def quad_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- The main theorem to prove the product of all x such that the expression is undefined
theorem product_of_undefined_x :
  (∃ x₁ x₂ : ℝ, quad_eq 1 4 3 x₁ ∧ quad_eq 1 4 3 x₂ ∧ x₁ * x₂ = 3) :=
by
  sorry

end NUMINAMATH_GPT_product_of_undefined_x_l1983_198319


namespace NUMINAMATH_GPT_central_angle_is_two_l1983_198386

noncomputable def central_angle_of_sector (r l : ℝ) (h1 : 2 * r + l = 4) (h2 : (1 / 2) * l * r = 1) : ℝ :=
  l / r

theorem central_angle_is_two (r l : ℝ) (h1 : 2 * r + l = 4) (h2 : (1 / 2) * l * r = 1) : central_angle_of_sector r l h1 h2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_central_angle_is_two_l1983_198386


namespace NUMINAMATH_GPT_natural_numbers_satisfy_equation_l1983_198323

theorem natural_numbers_satisfy_equation:
  ∀ (n k : ℕ), (k^5 + 5 * n^4 = 81 * k) ↔ (n = 2 ∧ k = 1) :=
by
  sorry

end NUMINAMATH_GPT_natural_numbers_satisfy_equation_l1983_198323


namespace NUMINAMATH_GPT_min_area_triangle_l1983_198392

theorem min_area_triangle (m n : ℝ) (h : m^2 + n^2 = 1/3) : ∃ S, S = 3 :=
by
  sorry

end NUMINAMATH_GPT_min_area_triangle_l1983_198392


namespace NUMINAMATH_GPT_probability_of_desired_roll_l1983_198303

-- Definitions of six-sided dice rolls and probability results
def is_greater_than_four (n : ℕ) : Prop := n > 4
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5

-- Definitions of probabilities based on dice outcomes
def prob_greater_than_four : ℚ := 2 / 6
def prob_prime : ℚ := 3 / 6

-- Definition of joint probability for independent events
def joint_prob : ℚ := prob_greater_than_four * prob_prime

-- Theorem to prove
theorem probability_of_desired_roll : joint_prob = 1 / 6 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_desired_roll_l1983_198303


namespace NUMINAMATH_GPT_triangle_BD_length_l1983_198313

theorem triangle_BD_length 
  (A B C D : Type) 
  (hAC : AC = 8) 
  (hBC : BC = 8) 
  (hAD : AD = 6) 
  (hCD : CD = 5) : BD = 6 :=
  sorry

end NUMINAMATH_GPT_triangle_BD_length_l1983_198313


namespace NUMINAMATH_GPT_spheres_in_base_l1983_198307

theorem spheres_in_base (n : ℕ) (T_n : ℕ) (total_spheres : ℕ) :
  (total_spheres = 165) →
  (total_spheres = (1 / 6 : ℚ) * ↑n * ↑(n + 1) * ↑(n + 2)) →
  (T_n = n * (n + 1) / 2) →
  n = 9 →
  T_n = 45 :=
by
  intros _ _ _ _
  sorry

end NUMINAMATH_GPT_spheres_in_base_l1983_198307


namespace NUMINAMATH_GPT_typing_speed_ratio_l1983_198396

theorem typing_speed_ratio (T M : ℝ) 
  (h1 : T + M = 12) 
  (h2 : T + 1.25 * M = 14) : 
  M / T = 2 :=
by 
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_typing_speed_ratio_l1983_198396


namespace NUMINAMATH_GPT_power_computation_l1983_198364

theorem power_computation : (12 ^ (12 / 2)) = 2985984 := by
  sorry

end NUMINAMATH_GPT_power_computation_l1983_198364


namespace NUMINAMATH_GPT_average_student_headcount_l1983_198318

theorem average_student_headcount (h1 : ℕ := 10900) (h2 : ℕ := 10500) (h3 : ℕ := 10700) (h4 : ℕ := 11300) : 
  (h1 + h2 + h3 + h4) / 4 = 10850 := 
by 
  sorry

end NUMINAMATH_GPT_average_student_headcount_l1983_198318


namespace NUMINAMATH_GPT_curve_symmetric_origin_l1983_198388

theorem curve_symmetric_origin (x y : ℝ) (h : 3*x^2 - 8*x*y + 2*y^2 = 0) :
  3*(-x)^2 - 8*(-x)*(-y) + 2*(-y)^2 = 3*x^2 - 8*x*y + 2*y^2 :=
sorry

end NUMINAMATH_GPT_curve_symmetric_origin_l1983_198388


namespace NUMINAMATH_GPT_pool_fill_time_l1983_198373

theorem pool_fill_time:
  ∀ (A B C D : ℚ),
  (A + B - D = 1 / 6) →
  (A + C - D = 1 / 5) →
  (B + C - D = 1 / 4) →
  (A + B + C - D = 1 / 3) →
  (1 / (A + B + C) = 60 / 23) :=
by intros A B C D h1 h2 h3 h4; sorry

end NUMINAMATH_GPT_pool_fill_time_l1983_198373


namespace NUMINAMATH_GPT_sum_of_solutions_l1983_198393

theorem sum_of_solutions :
  let a := -48
  let b := 110
  let c := 165
  ( ∀ x1 x2 : ℝ, (a * x1^2 + b * x1 + c = 0) ∧ (a * x2^2 + b * x2 + c = 0) → x1 ≠ x2 → (x1 + x2) = 55 / 24 ) :=
by
  let a := -48
  let b := 110
  let c := 165
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l1983_198393


namespace NUMINAMATH_GPT_find_marked_price_l1983_198395

theorem find_marked_price (cp : ℝ) (d : ℝ) (p : ℝ) (x : ℝ) (h1 : cp = 80) (h2 : d = 0.3) (h3 : p = 0.05) :
  (1 - d) * x = cp * (1 + p) → x = 120 :=
by
  sorry

end NUMINAMATH_GPT_find_marked_price_l1983_198395


namespace NUMINAMATH_GPT_total_cost_price_of_items_l1983_198325

/-- 
  Definition of the selling prices of the items A, B, and C.
  Definition of the profit percentages of the items A, B, and C.
  The statement is the total cost price calculation.
-/
def ItemA_SP : ℝ := 800
def ItemA_Profit : ℝ := 0.25
def ItemB_SP : ℝ := 1200
def ItemB_Profit : ℝ := 0.20
def ItemC_SP : ℝ := 1500
def ItemC_Profit : ℝ := 0.30

theorem total_cost_price_of_items :
  let CP_A := ItemA_SP / (1 + ItemA_Profit)
  let CP_B := ItemB_SP / (1 + ItemB_Profit)
  let CP_C := ItemC_SP / (1 + ItemC_Profit)
  CP_A + CP_B + CP_C = 2793.85 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_price_of_items_l1983_198325


namespace NUMINAMATH_GPT_polynomial_solution_l1983_198308

noncomputable def roots (a b c : ℤ) : Set ℝ :=
  { x : ℝ | a * x ^ 2 + b * x + c = 0 }

theorem polynomial_solution :
  let x1 := (1 + Real.sqrt 13) / 2
  let x2 := (1 - Real.sqrt 13) / 2
  x1 ∈ roots 1 (-1) (-3) → x2 ∈ roots 1 (-1) (-3) →
  ((x1^5 - 20) * (3*x2^4 - 2*x2 - 35) = -1063) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_solution_l1983_198308


namespace NUMINAMATH_GPT_find_correct_quotient_l1983_198380

theorem find_correct_quotient 
  (Q : ℕ)
  (D : ℕ)
  (h1 : D = 21 * Q)
  (h2 : D = 12 * 35) : 
  Q = 20 := 
by 
  sorry

end NUMINAMATH_GPT_find_correct_quotient_l1983_198380


namespace NUMINAMATH_GPT_dragon_heads_belong_to_dragons_l1983_198378

def truthful (H : ℕ) : Prop := 
  H = 1 ∨ H = 3

def lying (H : ℕ) : Prop := 
  H = 2 ∨ H = 4

def head1_statement : Prop := truthful 1
def head2_statement : Prop := truthful 3
def head3_statement : Prop := ¬ truthful 2
def head4_statement : Prop := lying 3

theorem dragon_heads_belong_to_dragons :
  head1_statement ∧ head2_statement ∧ head3_statement ∧ head4_statement →
  (∀ H, (truthful H ↔ H = 1 ∨ H = 3) ∧ (lying H ↔ H = 2 ∨ H = 4)) :=
by
  sorry

end NUMINAMATH_GPT_dragon_heads_belong_to_dragons_l1983_198378


namespace NUMINAMATH_GPT_problem_statement_l1983_198314

theorem problem_statement (x : ℕ) (h : 423 - x = 421) : (x * 423) + 421 = 1267 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1983_198314


namespace NUMINAMATH_GPT_larger_number_is_1641_l1983_198321

theorem larger_number_is_1641 (L S : ℕ) (h1 : L - S = 1370) (h2 : L = 6 * S + 15) : L = 1641 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_1641_l1983_198321


namespace NUMINAMATH_GPT_pi_is_irrational_l1983_198322

theorem pi_is_irrational (π : ℝ) (h : π = Real.pi) :
  ¬ ∃ (a b : ℤ), b ≠ 0 ∧ π = a / b :=
by
  sorry

end NUMINAMATH_GPT_pi_is_irrational_l1983_198322


namespace NUMINAMATH_GPT_probability_divisible_by_5_l1983_198345

def spinner_nums : List ℕ := [1, 2, 3, 5]

def total_outcomes (spins : ℕ) : ℕ :=
  List.length spinner_nums ^ spins

def count_divisible_by_5 (spins : ℕ) : ℕ :=
  let units_digit := 1
  let rest_combinations := (List.length spinner_nums) ^ (spins - units_digit)
  rest_combinations

theorem probability_divisible_by_5 : 
  let spins := 3 
  let successful_cases := count_divisible_by_5 spins
  let all_cases := total_outcomes spins
  successful_cases / all_cases = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_divisible_by_5_l1983_198345


namespace NUMINAMATH_GPT_min_points_to_guarantee_victory_l1983_198366

noncomputable def points_distribution (pos : ℕ) : ℕ :=
  match pos with
  | 1 => 7
  | 2 => 4
  | 3 => 2
  | _ => 0

def max_points_per_race : ℕ := 7
def num_races : ℕ := 3

theorem min_points_to_guarantee_victory : ∃ min_points, min_points = 18 ∧ 
  (∀ other_points, other_points < 18) := 
by {
  sorry
}

end NUMINAMATH_GPT_min_points_to_guarantee_victory_l1983_198366


namespace NUMINAMATH_GPT_graph_of_equation_represents_three_lines_l1983_198375

theorem graph_of_equation_represents_three_lines (x y : ℝ) :
  (x^2 * (x + y + 2) = y^2 * (x + y + 2)) →
  (∃ (a b c : ℝ), (a * x + b * y + c = 0) ∧
    ((a * x + b * y + c = 0) ∧ (a * x + b * y + c ≠ 0)) ∨
    ((a * x + b * y + c = 0) ∨ (a * x + b * y + c ≠ 0)) ∨
    (a * x + b * y + c = 0)) :=
by
  sorry

end NUMINAMATH_GPT_graph_of_equation_represents_three_lines_l1983_198375


namespace NUMINAMATH_GPT_bob_total_distance_l1983_198346

theorem bob_total_distance:
  let time1 := 1.5
  let speed1 := 60
  let time2 := 2
  let speed2 := 45
  (time1 * speed1) + (time2 * speed2) = 180 := 
  by
  sorry

end NUMINAMATH_GPT_bob_total_distance_l1983_198346


namespace NUMINAMATH_GPT_range_of_x_l1983_198335

theorem range_of_x (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_x_l1983_198335


namespace NUMINAMATH_GPT_no_function_f_satisfies_condition_l1983_198347

theorem no_function_f_satisfies_condition :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = f x + y^2 :=
by
  sorry

end NUMINAMATH_GPT_no_function_f_satisfies_condition_l1983_198347


namespace NUMINAMATH_GPT_binomial_coeff_sum_l1983_198340

theorem binomial_coeff_sum {a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℤ}
  (h : (1 - x)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7) :
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| = 128 :=
by
  sorry

end NUMINAMATH_GPT_binomial_coeff_sum_l1983_198340


namespace NUMINAMATH_GPT_problem_solution_l1983_198381

-- Definitions based on conditions given in the problem statement
def validExpression (n : ℕ) : ℕ := 
  sorry -- Placeholder for function defining valid expressions

def T (n : ℕ) : ℕ := 
  if n = 1 then 1 
  else validExpression n

def R (n : ℕ) : ℕ := T n % 4

def computeSum (k : ℕ) : ℕ := 
  (List.range k).map R |>.sum

-- Lean theorem statement to be proven
theorem problem_solution : 
  computeSum 1000001 = 320 := 
sorry

end NUMINAMATH_GPT_problem_solution_l1983_198381


namespace NUMINAMATH_GPT_ratio_docking_to_license_l1983_198362

noncomputable def Mitch_savings : ℕ := 20000
noncomputable def boat_cost_per_foot : ℕ := 1500
noncomputable def license_and_registration_fees : ℕ := 500
noncomputable def max_boat_length : ℕ := 12

theorem ratio_docking_to_license :
  let remaining_amount := Mitch_savings - license_and_registration_fees
  let cost_of_longest_boat := boat_cost_per_foot * max_boat_length
  let docking_fees := remaining_amount - cost_of_longest_boat
  docking_fees / license_and_registration_fees = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_docking_to_license_l1983_198362


namespace NUMINAMATH_GPT_range_of_m_l1983_198310

-- Definition of the propositions and conditions
def p (m : ℝ) : Prop := m > 2
def q (m : ℝ) : Prop := 1 ≤ m ∧ m ≤ 3
def prop (m : ℝ) : Prop := (¬(p m ∧ q m) ∧ (p m ∨ q m))

-- The proof statement showing the range of m
theorem range_of_m (m : ℝ) : prop m ↔ (1 ≤ m ∧ m ≤ 2) ∨ (m > 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1983_198310


namespace NUMINAMATH_GPT_people_left_line_l1983_198371

theorem people_left_line (L : ℕ) (h_initial : 31 - L + 25 = 31) : L = 25 :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_people_left_line_l1983_198371


namespace NUMINAMATH_GPT_product_of_three_numbers_l1983_198354

theorem product_of_three_numbers (x y z : ℝ) 
  (h1 : x + y + z = 20) 
  (h2 : x = 4 * (y + z)) 
  (h3 : y = 7 * z) :
  x * y * z = 28 := 
by 
  sorry

end NUMINAMATH_GPT_product_of_three_numbers_l1983_198354


namespace NUMINAMATH_GPT_find_x_l1983_198384

-- Definition of the binary operation
def binary_operation (a b c d : ℤ) : ℤ × ℤ :=
  (a - c, b + d)

-- Definition of our main theorem to be proved
theorem find_x (x y : ℤ) (h : binary_operation x y 2 3 = (4, 5)) : x = 6 :=
  by sorry

end NUMINAMATH_GPT_find_x_l1983_198384


namespace NUMINAMATH_GPT_polar_to_cartesian_coordinates_l1983_198300

theorem polar_to_cartesian_coordinates (ρ θ : ℝ) (hρ : ρ = 2) (hθ : θ = 5 * Real.pi / 6) :
  (ρ * Real.cos θ, ρ * Real.sin θ) = (-Real.sqrt 3, 1) :=
by
  sorry

end NUMINAMATH_GPT_polar_to_cartesian_coordinates_l1983_198300


namespace NUMINAMATH_GPT_total_students_accommodated_l1983_198349

structure BusConfig where
  columns : ℕ
  rows : ℕ
  broken_seats : ℕ

structure SplitBusConfig where
  columns : ℕ
  left_rows : ℕ
  right_rows : ℕ
  broken_seats : ℕ

structure ComplexBusConfig where
  columns : ℕ
  rows : ℕ
  special_rows_broken_seats : ℕ

def bus1 : BusConfig := { columns := 4, rows := 10, broken_seats := 2 }
def bus2 : BusConfig := { columns := 5, rows := 8, broken_seats := 4 }
def bus3 : BusConfig := { columns := 3, rows := 12, broken_seats := 3 }
def bus4 : SplitBusConfig := { columns := 4, left_rows := 6, right_rows := 8, broken_seats := 1 }
def bus5 : SplitBusConfig := { columns := 6, left_rows := 8, right_rows := 10, broken_seats := 5 }
def bus6 : ComplexBusConfig := { columns := 5, rows := 10, special_rows_broken_seats := 4 }

theorem total_students_accommodated :
  let seats_bus1 := (bus1.columns * bus1.rows) - bus1.broken_seats;
  let seats_bus2 := (bus2.columns * bus2.rows) - bus2.broken_seats;
  let seats_bus3 := (bus3.columns * bus3.rows) - bus3.broken_seats;
  let seats_bus4 := (bus4.columns * bus4.left_rows) + (bus4.columns * bus4.right_rows) - bus4.broken_seats;
  let seats_bus5 := (bus5.columns * bus5.left_rows) + (bus5.columns * bus5.right_rows) - bus5.broken_seats;
  let seats_bus6 := (bus6.columns * bus6.rows) - bus6.special_rows_broken_seats;
  seats_bus1 + seats_bus2 + seats_bus3 + seats_bus4 + seats_bus5 + seats_bus6 = 311 :=
sorry

end NUMINAMATH_GPT_total_students_accommodated_l1983_198349


namespace NUMINAMATH_GPT_find_x1_l1983_198358

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 3)
  (h3 : x1 + x2 + x3 + x4 = 2) : 
  x1 = 4 / 5 :=
sorry

end NUMINAMATH_GPT_find_x1_l1983_198358


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1983_198348

def isEllipse (a b : ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, f x y = 1

theorem necessary_but_not_sufficient_condition (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  isEllipse a b (λ x y => a * x^2 + b * y^2) → ¬(∃ x y : ℝ, a * x^2 + b * y^2 = 1) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1983_198348


namespace NUMINAMATH_GPT_shaded_area_percentage_correct_l1983_198357

-- Define a square and the conditions provided
def square (side_length : ℕ) : ℕ := side_length ^ 2

-- Define conditions
def EFGH_side_length : ℕ := 6
def total_area : ℕ := square EFGH_side_length

def shaded_area_1 : ℕ := square 2
def shaded_area_2 : ℕ := square 4 - square 3
def shaded_area_3 : ℕ := square 6 - square 5

def total_shaded_area : ℕ := shaded_area_1 + shaded_area_2 + shaded_area_3

def shaded_percentage : ℚ := total_shaded_area / total_area * 100

-- Statement of the theorem to prove
theorem shaded_area_percentage_correct :
  shaded_percentage = 61.11 := by sorry

end NUMINAMATH_GPT_shaded_area_percentage_correct_l1983_198357


namespace NUMINAMATH_GPT_remainder_of_power_mod_l1983_198326

theorem remainder_of_power_mod :
  (5^2023) % 11 = 4 :=
  by
    sorry

end NUMINAMATH_GPT_remainder_of_power_mod_l1983_198326


namespace NUMINAMATH_GPT_min_value_fraction_l1983_198387

variable (a b : ℝ)

theorem min_value_fraction (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + 2 * b = 1) : 
  (1 / a + 1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_value_fraction_l1983_198387


namespace NUMINAMATH_GPT_no_positive_solution_for_special_k_l1983_198329
open Nat

theorem no_positive_solution_for_special_k (p : ℕ) (hp : p.Prime) (hmod : p % 4 = 3) :
    ¬ ∃ n m k : ℕ, (n > 0) ∧ (m > 0) ∧ (k = p^2) ∧ (n^2 + m^2 = k * (m^4 + n)) :=
sorry

end NUMINAMATH_GPT_no_positive_solution_for_special_k_l1983_198329


namespace NUMINAMATH_GPT_largest_divisor_of_expression_of_even_x_l1983_198372

theorem largest_divisor_of_expression_of_even_x (x : ℤ) (h_even : ∃ k : ℤ, x = 2 * k) :
  ∃ (d : ℤ), d = 240 ∧ d ∣ ((8 * x + 2) * (8 * x + 4) * (4 * x + 2)) :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_of_expression_of_even_x_l1983_198372


namespace NUMINAMATH_GPT_shooting_competition_hits_l1983_198336

noncomputable def a1 : ℝ := 1
noncomputable def d : ℝ := 0.5
noncomputable def S_n (n : ℝ) : ℝ := (n / 2) * (2 * a1 + (n - 1) * d)

theorem shooting_competition_hits (n : ℝ) (h : S_n n = 7) : 25 - n = 21 :=
by
  -- sequence of proof steps
  sorry

end NUMINAMATH_GPT_shooting_competition_hits_l1983_198336
