import Mathlib

namespace NUMINAMATH_GPT_range_of_a_l1745_174541

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 + a * x + a = 0) → a ∈ Set.Iic 0 ∪ Set.Ici 4 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1745_174541


namespace NUMINAMATH_GPT_work_completion_time_l1745_174526

theorem work_completion_time (x : ℕ) (h1 : ∀ B, ∀ A, A = 2 * B) (h2 : (1/x + 1/(2*x)) * 4 = 1) : x = 12 := 
sorry

end NUMINAMATH_GPT_work_completion_time_l1745_174526


namespace NUMINAMATH_GPT_batsman_average_after_11th_inning_l1745_174507

theorem batsman_average_after_11th_inning 
  (x : ℝ) 
  (h1 : (10 * x + 95) / 11 = x + 5) : 
  x + 5 = 45 :=
by 
  sorry

end NUMINAMATH_GPT_batsman_average_after_11th_inning_l1745_174507


namespace NUMINAMATH_GPT_sum_of_possible_g9_values_l1745_174578

def f (x : ℝ) : ℝ := x^2 - 6 * x + 14

def g (y : ℝ) : ℝ := 3 * y + 2

theorem sum_of_possible_g9_values : ∀ {x1 x2 : ℝ}, f x1 = 9 → f x2 = 9 → g x1 + g x2 = 22 := by
  intros
  sorry

end NUMINAMATH_GPT_sum_of_possible_g9_values_l1745_174578


namespace NUMINAMATH_GPT_maria_sold_in_first_hour_l1745_174586

variable (x : ℕ)

-- Conditions
def sold_in_first_hour := x
def sold_in_second_hour := 2
def average_sold_in_two_hours := 6

-- Proof Goal
theorem maria_sold_in_first_hour :
  (sold_in_first_hour + sold_in_second_hour) / 2 = average_sold_in_two_hours → sold_in_first_hour = 10 :=
by
  sorry

end NUMINAMATH_GPT_maria_sold_in_first_hour_l1745_174586


namespace NUMINAMATH_GPT_triplet_sum_not_equal_two_l1745_174594

theorem triplet_sum_not_equal_two :
  ¬((1.2 + -2.2 + 2) = 2) ∧ ¬((- 4 / 3 + - 2 / 3 + 3) = 2) :=
by
  sorry

end NUMINAMATH_GPT_triplet_sum_not_equal_two_l1745_174594


namespace NUMINAMATH_GPT_tangent_to_parabola_k_l1745_174521

theorem tangent_to_parabola_k (k : ℝ) :
  (∃ (x y : ℝ), 4 * x + 7 * y + k = 0 ∧ y^2 = 32 * x ∧ 
  ∀ (a b : ℝ) (ha : a * y^2 + b * y + k = 0), b^2 - 4 * a * k = 0) → k = 98 :=
by
  sorry

end NUMINAMATH_GPT_tangent_to_parabola_k_l1745_174521


namespace NUMINAMATH_GPT_arithmetic_sequence_terms_l1745_174577

theorem arithmetic_sequence_terms (a : ℕ → ℕ) (n : ℕ)
  (h1 : a 1 + a 2 + a 3 = 34)
  (h2 : a n + a (n - 1) + a (n - 2) = 146)
  (h3 : n * (a 1 + a n) = 780) : n = 13 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_terms_l1745_174577


namespace NUMINAMATH_GPT_number_of_bushes_needed_l1745_174525

-- Definitions from the conditions
def containers_per_bush : ℕ := 10
def containers_per_zucchini : ℕ := 3
def zucchinis_required : ℕ := 72

-- Statement to prove
theorem number_of_bushes_needed : 
  ∃ bushes_needed : ℕ, bushes_needed = 22 ∧ 
  (zucchinis_required * containers_per_zucchini + containers_per_bush - 1) / containers_per_bush = bushes_needed := 
by
  sorry

end NUMINAMATH_GPT_number_of_bushes_needed_l1745_174525


namespace NUMINAMATH_GPT_white_wash_cost_l1745_174597

noncomputable def room_length : ℝ := 25
noncomputable def room_width : ℝ := 15
noncomputable def room_height : ℝ := 12
noncomputable def door_height : ℝ := 6
noncomputable def door_width : ℝ := 3
noncomputable def window_height : ℝ := 4
noncomputable def window_width : ℝ := 3
noncomputable def num_windows : ℕ := 3
noncomputable def cost_per_sqft : ℝ := 3

theorem white_wash_cost :
  let wall_area := 2 * (room_length * room_height + room_width * room_height)
  let door_area := door_height * door_width
  let window_area := window_height * window_width
  let total_non_white_wash_area := door_area + ↑num_windows * window_area
  let white_wash_area := wall_area - total_non_white_wash_area
  let total_cost := white_wash_area * cost_per_sqft
  total_cost = 2718 :=  
by
  sorry

end NUMINAMATH_GPT_white_wash_cost_l1745_174597


namespace NUMINAMATH_GPT_speed_ratio_l1745_174551

noncomputable def k_value {u v x y : ℝ} (h_uv : u > 0) (h_v : v > 0) (h_x : x > 0) (h_y : y > 0) 
  (h_ratio : u / v = ((x + y) / (u - v)) / ((x + y) / (u + v))) : ℝ :=
  1 + Real.sqrt 2

theorem speed_ratio (u v x y : ℝ) (h_uv : u > 0) (h_v : v > 0) (h_x : x > 0) (h_y : y > 0) 
  (h_ratio : u / v = ((x + y) / (u - v)) / ((x + y) / (u + v))) : 
  u / v = k_value h_uv h_v h_x h_y h_ratio :=
sorry

end NUMINAMATH_GPT_speed_ratio_l1745_174551


namespace NUMINAMATH_GPT_standard_equation_of_parabola_l1745_174579

theorem standard_equation_of_parabola (F : ℝ × ℝ) (hF : F.1 + 2 * F.2 + 3 = 0) :
  (∃ y₀: ℝ, y₀ < 0 ∧ F = (0, y₀) ∧ ∀ x: ℝ, x ^ 2 = - 6 * y₀ * x) ∨
  (∃ x₀: ℝ, x₀ < 0 ∧ F = (x₀, 0) ∧ ∀ y: ℝ, y ^ 2 = - 12 * x₀ * y) :=
sorry

end NUMINAMATH_GPT_standard_equation_of_parabola_l1745_174579


namespace NUMINAMATH_GPT_part1_part2_l1745_174532

noncomputable def f (x a : ℝ) : ℝ := |x - 2 * a| + |x - 3 * a|

theorem part1 (a : ℝ) (h_min : ∃ x, f x a = 2) : |a| = 2 := by
  sorry

theorem part2 (m : ℝ)
  (h_condition : ∀ x : ℝ, ∃ a : ℝ, -2 ≤ a ∧ a ≤ 2 ∧ (m^2 - |m| - f x a) < 0) :
  -1 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1745_174532


namespace NUMINAMATH_GPT_tony_solving_puzzles_time_l1745_174574

theorem tony_solving_puzzles_time : ∀ (warm_up_time long_puzzle_ratio num_long_puzzles : ℕ),
  warm_up_time = 10 →
  long_puzzle_ratio = 3 →
  num_long_puzzles = 2 →
  (warm_up_time + long_puzzle_ratio * warm_up_time * num_long_puzzles) = 70 :=
by
  intros
  sorry

end NUMINAMATH_GPT_tony_solving_puzzles_time_l1745_174574


namespace NUMINAMATH_GPT_numDogsInPetStore_l1745_174558

-- Definitions from conditions
variables {D P : Nat}

-- Theorem statement - no proof provided
theorem numDogsInPetStore (h1 : D + P = 15) (h2 : 4 * D + 2 * P = 42) : D = 6 :=
by
  sorry

end NUMINAMATH_GPT_numDogsInPetStore_l1745_174558


namespace NUMINAMATH_GPT_probability_of_exactly_one_red_ball_l1745_174564

-- Definitions based on the conditions:
def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3
def draw_count : ℕ := 2

-- Required to calculate combinatory values
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Definitions of probabilities (though we won't use them explicitly for the statement):
def total_events : ℕ := choose total_balls draw_count
def no_red_ball_events : ℕ := choose white_balls draw_count
def one_red_ball_events : ℕ := choose red_balls 1 * choose white_balls 1

-- Probability Functions (for context):
def probability (events : ℕ) (total_events : ℕ) : ℚ := events / total_events

-- Lean 4 statement:
theorem probability_of_exactly_one_red_ball :
  probability one_red_ball_events total_events = 3/5 := by
  sorry

end NUMINAMATH_GPT_probability_of_exactly_one_red_ball_l1745_174564


namespace NUMINAMATH_GPT_solve_for_a_l1745_174595

theorem solve_for_a (a : ℤ) :
  (|2 * a + 1| = 3) ↔ (a = 1 ∨ a = -2) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l1745_174595


namespace NUMINAMATH_GPT_correct_average_weight_l1745_174557

-- Definitions
def initial_average_weight : ℝ := 58.4
def number_of_boys : ℕ := 20
def misread_weight_initial : ℝ := 56
def misread_weight_correct : ℝ := 68

-- Correct average weight
theorem correct_average_weight : 
  let initial_total_weight := initial_average_weight * (number_of_boys : ℝ)
  let difference := misread_weight_correct - misread_weight_initial
  let correct_total_weight := initial_total_weight + difference
  let correct_average_weight := correct_total_weight / (number_of_boys : ℝ)
  correct_average_weight = 59 :=
by
  -- Insert the proof steps if needed
  sorry

end NUMINAMATH_GPT_correct_average_weight_l1745_174557


namespace NUMINAMATH_GPT_part_one_part_two_l1745_174576

variable {a b c : ℝ}
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : a^2 + b^2 + 4*c^2 = 3)

theorem part_one (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  a + b + 2*c ≤ 3 :=
sorry

theorem part_two (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + 4*c^2 = 3) (h_b_eq_2c : b = 2*c) :
  1/a + 1/c ≥ 3 :=
sorry

end NUMINAMATH_GPT_part_one_part_two_l1745_174576


namespace NUMINAMATH_GPT_coffee_last_days_l1745_174546

theorem coffee_last_days (weight : ℕ) (cups_per_lb : ℕ) (cups_per_day : ℕ) 
  (h_weight : weight = 3) 
  (h_cups_per_lb : cups_per_lb = 40) 
  (h_cups_per_day : cups_per_day = 3) : 
  (weight * cups_per_lb) / cups_per_day = 40 := 
by 
  sorry

end NUMINAMATH_GPT_coffee_last_days_l1745_174546


namespace NUMINAMATH_GPT_eq1_eq2_eq3_eq4_l1745_174524

/-
  First, let's define each problem and then state the equivalency of the solutions.
  We will assume the real number type for the domain of x.
-/

-- Assume x is a real number
variable (x : ℝ)

theorem eq1 (x : ℝ) : (x - 3)^2 = 4 -> (x = 5 ∨ x = 1) := sorry

theorem eq2 (x : ℝ) : x^2 - 5 * x + 1 = 0 -> (x = (5 - Real.sqrt 21) / 2 ∨ x = (5 + Real.sqrt 21) / 2) := sorry

theorem eq3 (x : ℝ) : x * (3 * x - 2) = 2 * (3 * x - 2) -> (x = 2 / 3 ∨ x = 2) := sorry

theorem eq4 (x : ℝ) : (x + 1)^2 = 4 * (1 - x)^2 -> (x = 1 / 3 ∨ x = 3) := sorry

end NUMINAMATH_GPT_eq1_eq2_eq3_eq4_l1745_174524


namespace NUMINAMATH_GPT_vector_addition_l1745_174581

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-3, 4)

theorem vector_addition :
  2 • a + b = (1, 2) :=
by
  sorry

end NUMINAMATH_GPT_vector_addition_l1745_174581


namespace NUMINAMATH_GPT_percentage_not_speaking_French_is_60_l1745_174516

-- Define the number of students who speak English well and those who do not.
def speakEnglishWell : Nat := 20
def doNotSpeakEnglish : Nat := 60

-- Calculate the total number of students who speak French.
def speakFrench : Nat := speakEnglishWell + doNotSpeakEnglish

-- Define the total number of students surveyed.
def totalStudents : Nat := 200

-- Calculate the number of students who do not speak French.
def doNotSpeakFrench : Nat := totalStudents - speakFrench

-- Calculate the percentage of students who do not speak French.
def percentageDoNotSpeakFrench : Float := (doNotSpeakFrench.toFloat / totalStudents.toFloat) * 100

-- Theorem asserting the percentage of students who do not speak French is 60%.
theorem percentage_not_speaking_French_is_60 : percentageDoNotSpeakFrench = 60 := by
  sorry

end NUMINAMATH_GPT_percentage_not_speaking_French_is_60_l1745_174516


namespace NUMINAMATH_GPT_find_x_l1745_174566

variables (a b x : ℝ)
variables (pos_a : a > 0) (pos_b : b > 0) (pos_x : x > 0)

theorem find_x : ((2 * a) ^ (2 * b) = (a^2) ^ b * x ^ b) → (x = 4) := by
  sorry

end NUMINAMATH_GPT_find_x_l1745_174566


namespace NUMINAMATH_GPT_sufficient_condition_l1745_174596

-- Definitions of propositions p and q
variables (p q : Prop)

-- Theorem statement
theorem sufficient_condition (h : ¬(p ∨ q)) : ¬p :=
by sorry

end NUMINAMATH_GPT_sufficient_condition_l1745_174596


namespace NUMINAMATH_GPT_gear_revolutions_l1745_174548

theorem gear_revolutions (t : ℝ) (r_p r_q : ℝ) (h1 : r_q = 40) (h2 : t = 20)
 (h3 : (r_q / 60) * t = ((r_p / 60) * t) + 10) :
 r_p = 10 :=
 sorry

end NUMINAMATH_GPT_gear_revolutions_l1745_174548


namespace NUMINAMATH_GPT_total_number_of_workers_l1745_174501

theorem total_number_of_workers 
    (W : ℕ) 
    (average_salary_all : ℕ := 8000) 
    (average_salary_technicians : ℕ := 12000) 
    (average_salary_rest : ℕ := 6000) 
    (total_salary_all : ℕ := average_salary_all * W) 
    (salary_technicians : ℕ := 6 * average_salary_technicians) 
    (N : ℕ := W - 6) 
    (salary_rest : ℕ := average_salary_rest * N) 
    (salary_equation : total_salary_all = salary_technicians + salary_rest) 
  : W = 18 := 
sorry

end NUMINAMATH_GPT_total_number_of_workers_l1745_174501


namespace NUMINAMATH_GPT_unit_digit_2_pow_2024_l1745_174531

theorem unit_digit_2_pow_2024 : (2 ^ 2024) % 10 = 6 := by
  -- We observe the repeating pattern in the unit digits of powers of 2:
  -- 2^1 = 2 -> unit digit is 2
  -- 2^2 = 4 -> unit digit is 4
  -- 2^3 = 8 -> unit digit is 8
  -- 2^4 = 16 -> unit digit is 6
  -- The cycle repeats every 4 powers: 2, 4, 8, 6
  -- 2024 ≡ 0 (mod 4), so it corresponds to the unit digit of 2^4, which is 6
  sorry

end NUMINAMATH_GPT_unit_digit_2_pow_2024_l1745_174531


namespace NUMINAMATH_GPT_solve_congruence_l1745_174585

theorem solve_congruence :
  ∃ a m : ℕ, (8 * (x : ℕ) + 1) % 12 = 5 % 12 ∧ m ≥ 2 ∧ a < m ∧ x ≡ a [MOD m] ∧ a + m = 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_congruence_l1745_174585


namespace NUMINAMATH_GPT_range_of_a_l1745_174569

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - 1| - |x - 2| < a^2 + a + 1) →
  (a < -1 ∨ a > 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1745_174569


namespace NUMINAMATH_GPT_combine_like_terms_problem1_combine_like_terms_problem2_l1745_174543

-- Problem 1 Statement
theorem combine_like_terms_problem1 (x y : ℝ) : 
  2*x - (x - y) + (x + y) = 2*x + 2*y :=
by
  sorry

-- Problem 2 Statement
theorem combine_like_terms_problem2 (x : ℝ) : 
  3*x^2 - 9*x + 2 - x^2 + 4*x - 6 = 2*x^2 - 5*x - 4 :=
by
  sorry

end NUMINAMATH_GPT_combine_like_terms_problem1_combine_like_terms_problem2_l1745_174543


namespace NUMINAMATH_GPT_missed_both_shots_l1745_174553

variables (p q : Prop)

theorem missed_both_shots : (¬p ∧ ¬q) ↔ ¬(p ∨ q) :=
by sorry

end NUMINAMATH_GPT_missed_both_shots_l1745_174553


namespace NUMINAMATH_GPT_second_car_mileage_l1745_174588

theorem second_car_mileage (x : ℝ) : 
  (150 / 50) + (150 / x) + (150 / 15) = 56 / 2 → x = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_second_car_mileage_l1745_174588


namespace NUMINAMATH_GPT_sin_240_eq_neg_sqrt3_div_2_l1745_174528

theorem sin_240_eq_neg_sqrt3_div_2 : Real.sin (240 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_sin_240_eq_neg_sqrt3_div_2_l1745_174528


namespace NUMINAMATH_GPT_graduation_ceremony_chairs_l1745_174518

theorem graduation_ceremony_chairs (g p t a : ℕ) 
  (h_g : g = 50) 
  (h_p : p = 2 * g) 
  (h_t : t = 20) 
  (h_a : a = t / 2) : 
  g + p + t + a = 180 :=
by
  sorry

end NUMINAMATH_GPT_graduation_ceremony_chairs_l1745_174518


namespace NUMINAMATH_GPT_total_fruit_punch_eq_21_l1745_174556

def orange_punch : ℝ := 4.5
def cherry_punch := 2 * orange_punch
def apple_juice := cherry_punch - 1.5

theorem total_fruit_punch_eq_21 : orange_punch + cherry_punch + apple_juice = 21 := by 
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_total_fruit_punch_eq_21_l1745_174556


namespace NUMINAMATH_GPT_sqrt_of_4_equals_2_l1745_174593

theorem sqrt_of_4_equals_2 : Real.sqrt 4 = 2 :=
by sorry

end NUMINAMATH_GPT_sqrt_of_4_equals_2_l1745_174593


namespace NUMINAMATH_GPT_gcd_459_357_l1745_174559

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_GPT_gcd_459_357_l1745_174559


namespace NUMINAMATH_GPT_find_ab_sum_eq_42_l1745_174552

noncomputable def find_value (a b : ℝ) : ℝ := a^2 * b + a * b^2

theorem find_ab_sum_eq_42 (a b : ℝ) (h1 : a + b = 6) (h2 : a * b = 7) : find_value a b = 42 := by
  sorry

end NUMINAMATH_GPT_find_ab_sum_eq_42_l1745_174552


namespace NUMINAMATH_GPT_eliana_steps_total_l1745_174587

def eliana_walks_first_day_steps := 200 + 300
def eliana_walks_second_day_steps := 2 * eliana_walks_first_day_steps
def eliana_walks_third_day_steps := eliana_walks_second_day_steps + 100
def eliana_total_steps := eliana_walks_first_day_steps + eliana_walks_second_day_steps + eliana_walks_third_day_steps

theorem eliana_steps_total : eliana_total_steps = 2600 := by
  sorry

end NUMINAMATH_GPT_eliana_steps_total_l1745_174587


namespace NUMINAMATH_GPT_x_varies_as_half_power_of_z_l1745_174535

variable {x y z : ℝ} -- declare variables as real numbers

-- Assume the conditions, which are the relationships between x, y, and z
variable (k j : ℝ) (k_pos : k > 0) (j_pos : j > 0)
axiom xy_relationship : ∀ y, x = k * y^2
axiom yz_relationship : ∀ z, y = j * z^(1/4)

-- The theorem we want to prove
theorem x_varies_as_half_power_of_z (z : ℝ) (h : z ≥ 0) : ∃ m, m > 0 ∧ x = m * z^(1/2) :=
sorry

end NUMINAMATH_GPT_x_varies_as_half_power_of_z_l1745_174535


namespace NUMINAMATH_GPT_john_worked_period_l1745_174502

theorem john_worked_period (A : ℝ) (n : ℕ) (h1 : 6 * A = 1 / 2 * (6 * A + n * A)) : n + 1 = 7 :=
by
  sorry

end NUMINAMATH_GPT_john_worked_period_l1745_174502


namespace NUMINAMATH_GPT_perfect_square_trinomial_m_value_l1745_174537

theorem perfect_square_trinomial_m_value (m : ℤ) :
  (∃ a : ℤ, ∀ y : ℤ, y^2 + my + 9 = (y + a)^2) ↔ (m = 6 ∨ m = -6) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_m_value_l1745_174537


namespace NUMINAMATH_GPT_range_of_a_l1745_174570

-- Define the set M
def M : Set ℝ := { x | -1 ≤ x ∧ x < 2 }

-- Define the set N
def N (a : ℝ) : Set ℝ := { x | x ≤ a }

-- The theorem to be proved
theorem range_of_a (a : ℝ) (h : (M ∩ N a).Nonempty) : a ≥ -1 := sorry

end NUMINAMATH_GPT_range_of_a_l1745_174570


namespace NUMINAMATH_GPT_parabola_exists_l1745_174555

noncomputable def parabola_conditions (a b : ℝ) : Prop :=
  (a + b = -3) ∧ (4 * a - 2 * b = 12)

noncomputable def translated_min_equals_six (m : ℝ) : Prop :=
  (m > 0) ∧ ((-1 - 2 + m)^2 - 3 = 6) ∨ ((3 - 2 - m)^2 - 3 = 6)

theorem parabola_exists (a b m : ℝ) (x y : ℝ) :
  parabola_conditions a b → y = x^2 + b * x + 1 → translated_min_equals_six m →
  (y = x^2 - 4 * x + 1) ∧ (m = 6 ∨ m = 4) := 
by 
  sorry

end NUMINAMATH_GPT_parabola_exists_l1745_174555


namespace NUMINAMATH_GPT_circumference_divided_by_diameter_l1745_174545

noncomputable def radius : ℝ := 15
noncomputable def circumference : ℝ := 90
noncomputable def diameter : ℝ := 2 * radius

theorem circumference_divided_by_diameter :
  circumference / diameter = 3 := by
  sorry

end NUMINAMATH_GPT_circumference_divided_by_diameter_l1745_174545


namespace NUMINAMATH_GPT_probability_exactly_one_first_class_l1745_174511

-- Define the probabilities
def prob_first_class_first_intern : ℚ := 2 / 3
def prob_first_class_second_intern : ℚ := 3 / 4
def prob_not_first_class_first_intern : ℚ := 1 - prob_first_class_first_intern
def prob_not_first_class_second_intern : ℚ := 1 - prob_first_class_second_intern

-- Define the event A, which is the event that exactly one of the two parts is of first-class quality
def prob_event_A : ℚ :=
  (prob_first_class_first_intern * prob_not_first_class_second_intern) +
  (prob_not_first_class_first_intern * prob_first_class_second_intern)

theorem probability_exactly_one_first_class (h1 : prob_first_class_first_intern = 2 / 3) 
    (h2 : prob_first_class_second_intern = 3 / 4) 
    (h3 : prob_event_A = 
          (prob_first_class_first_intern * (1 - prob_first_class_second_intern)) + 
          ((1 - prob_first_class_first_intern) * prob_first_class_second_intern)) : 
  prob_event_A = 5 / 12 := 
  sorry

end NUMINAMATH_GPT_probability_exactly_one_first_class_l1745_174511


namespace NUMINAMATH_GPT_son_l1745_174520

variable (S M : ℤ)

-- Conditions
def condition1 : Prop := M = S + 24
def condition2 : Prop := M + 2 = 2 * (S + 2)

theorem son's_age : condition1 S M ∧ condition2 S M → S = 22 :=
by
  sorry

end NUMINAMATH_GPT_son_l1745_174520


namespace NUMINAMATH_GPT_bryan_more_than_ben_l1745_174504

theorem bryan_more_than_ben :
  let Bryan_candies := 50
  let Ben_candies := 20
  Bryan_candies - Ben_candies = 30 :=
by
  let Bryan_candies := 50
  let Ben_candies := 20
  sorry

end NUMINAMATH_GPT_bryan_more_than_ben_l1745_174504


namespace NUMINAMATH_GPT_negation_of_exists_l1745_174565

theorem negation_of_exists (x : ℝ) : 
  ¬ (∃ x : ℝ, 2 * x^2 + 2 * x - 1 ≤ 0) ↔ ∀ x : ℝ, 2 * x^2 + 2 * x - 1 > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_l1745_174565


namespace NUMINAMATH_GPT_competition_scores_l1745_174503

theorem competition_scores (n d : ℕ) (h_n : 1 < n)
  (h_total_score : d * (n * (n + 1)) / 2 = 26 * n) :
  (n, d) = (3, 13) ∨ (n, d) = (12, 4) ∨ (n, d) = (25, 2) :=
by
  sorry

end NUMINAMATH_GPT_competition_scores_l1745_174503


namespace NUMINAMATH_GPT_base_five_product_l1745_174589

open Nat

/-- Definition of the base 5 representation of 131 and 21 --/
def n131 := 1 * 5^2 + 3 * 5^1 + 1 * 5^0
def n21 := 2 * 5^1 + 1 * 5^0

/-- Definition of the expected result in base 5 --/
def expected_result := 3 * 5^3 + 2 * 5^2 + 5 * 5^1 + 1 * 5^0

/-- Claim to prove that the product of 131_5 and 21_5 equals 3251_5 --/
theorem base_five_product : n131 * n21 = expected_result := by sorry

end NUMINAMATH_GPT_base_five_product_l1745_174589


namespace NUMINAMATH_GPT_vector_calculation_l1745_174538

def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, -2)

theorem vector_calculation : 2 • a - b = (5, 8) := by
  sorry

end NUMINAMATH_GPT_vector_calculation_l1745_174538


namespace NUMINAMATH_GPT_abs_neg_three_l1745_174571

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_three_l1745_174571


namespace NUMINAMATH_GPT_tenth_term_arithmetic_sequence_l1745_174505

def arithmetic_sequence (a1 d : ℚ) (n : ℕ) : ℚ :=
  a1 + (n - 1) * d

theorem tenth_term_arithmetic_sequence :
  arithmetic_sequence (1 / 2) (1 / 2) 10 = 5 :=
by
  sorry

end NUMINAMATH_GPT_tenth_term_arithmetic_sequence_l1745_174505


namespace NUMINAMATH_GPT_symmetric_point_origin_l1745_174560

-- Define the original point
def original_point : ℝ × ℝ := (4, -1)

-- Define a function to find the symmetric point with respect to the origin
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- State the theorem
theorem symmetric_point_origin : symmetric_point original_point = (-4, 1) :=
sorry

end NUMINAMATH_GPT_symmetric_point_origin_l1745_174560


namespace NUMINAMATH_GPT_soldier_rearrangement_20x20_soldier_rearrangement_21x21_l1745_174591

theorem soldier_rearrangement_20x20 (d : ℝ) : d ≤ 10 * Real.sqrt 2 :=
by
  -- Problem (a) setup and conditions
  sorry

theorem soldier_rearrangement_21x21 (d : ℝ) : d ≤ 10 * Real.sqrt 2 :=
by
  -- Problem (b) setup and conditions
  sorry

end NUMINAMATH_GPT_soldier_rearrangement_20x20_soldier_rearrangement_21x21_l1745_174591


namespace NUMINAMATH_GPT_find_phi_l1745_174529

theorem find_phi (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) :
  (∀ x, 2 * Real.sin (2 * x + φ - π / 6) = 2 * Real.cos (2 * x)) → φ = 5 * π / 6 :=
by
  sorry

end NUMINAMATH_GPT_find_phi_l1745_174529


namespace NUMINAMATH_GPT_total_oysters_and_crabs_is_195_l1745_174549

-- Define the initial conditions
def oysters_day1 : ℕ := 50
def crabs_day1 : ℕ := 72

-- Define the calculations for the second day
def oysters_day2 : ℕ := oysters_day1 / 2
def crabs_day2 : ℕ := crabs_day1 * 2 / 3

-- Define the total counts over the two days
def total_oysters : ℕ := oysters_day1 + oysters_day2
def total_crabs : ℕ := crabs_day1 + crabs_day2
def total_count : ℕ := total_oysters + total_crabs

-- The goal specification
theorem total_oysters_and_crabs_is_195 : total_count = 195 :=
by
  sorry

end NUMINAMATH_GPT_total_oysters_and_crabs_is_195_l1745_174549


namespace NUMINAMATH_GPT_line_passes_through_point_l1745_174530

theorem line_passes_through_point (k : ℝ) :
  ∀ k : ℝ, (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 :=
by
  intro k
  sorry

end NUMINAMATH_GPT_line_passes_through_point_l1745_174530


namespace NUMINAMATH_GPT_equal_sides_length_of_isosceles_right_triangle_l1745_174523

noncomputable def isosceles_right_triangle (a c : ℝ) : Prop :=
  c^2 = 2 * a^2 ∧ a^2 + a^2 + c^2 = 725

theorem equal_sides_length_of_isosceles_right_triangle (a c : ℝ) 
  (h : isosceles_right_triangle a c) : 
  a = 13.5 :=
by
  sorry

end NUMINAMATH_GPT_equal_sides_length_of_isosceles_right_triangle_l1745_174523


namespace NUMINAMATH_GPT_employees_participating_in_game_l1745_174522

theorem employees_participating_in_game 
  (managers players : ℕ)
  (teams people_per_team : ℕ)
  (h_teams : teams = 3)
  (h_people_per_team : people_per_team = 2)
  (h_managers : managers = 3)
  (h_total_players : players = teams * people_per_team) :
  players - managers = 3 :=
sorry

end NUMINAMATH_GPT_employees_participating_in_game_l1745_174522


namespace NUMINAMATH_GPT_complementary_event_A_l1745_174514

-- Define the events
def EventA (defective : ℕ) : Prop := defective ≥ 2

def ComplementaryEvent (defective : ℕ) : Prop := defective ≤ 1

-- Question: Prove that the complementary event of event A ("at least 2 defective products") 
-- is "at most 1 defective product" given the conditions.
theorem complementary_event_A (defective : ℕ) (total : ℕ) (h_total : total = 10) :
  EventA defective ↔ ComplementaryEvent defective :=
by sorry

end NUMINAMATH_GPT_complementary_event_A_l1745_174514


namespace NUMINAMATH_GPT_compute_expression_l1745_174582

theorem compute_expression :
    (3 + 5)^2 + (3^2 + 5^2 + 3 * 5) = 113 := 
by sorry

end NUMINAMATH_GPT_compute_expression_l1745_174582


namespace NUMINAMATH_GPT_candles_lit_time_correct_l1745_174580

noncomputable def candle_time : String :=
  let initial_length := 1 -- Since the length is uniform, we use 1
  let rateA := initial_length / (6 * 60) -- Rate at which Candle A burns out
  let rateB := initial_length / (8 * 60) -- Rate at which Candle B burns out
  let t := 320 -- The time in minutes that satisfy the condition
  let time_lit := (16 * 60 - t) / 60 -- Convert minutes to hours
  if time_lit = 10 + 40 / 60 then "10:40 AM" else "Unknown"

theorem candles_lit_time_correct :
  candle_time = "10:40 AM" := 
by
  sorry

end NUMINAMATH_GPT_candles_lit_time_correct_l1745_174580


namespace NUMINAMATH_GPT_anne_speed_ratio_l1745_174584

variable (B A A' : ℝ) (hours_to_clean_together : ℝ) (hours_to_clean_with_new_anne : ℝ)

-- Conditions
def cleaning_condition_1 := (A + B) * 4 = 1 -- Combined rate for 4 hours
def cleaning_condition_2 := A = 1 / 12      -- Anne's rate alone
def cleaning_condition_3 := (A' + B) * 3 = 1 -- Combined rate for 3 hours with new Anne's rate

-- Theorem to Prove
theorem anne_speed_ratio (h1 : cleaning_condition_1 B A)
                         (h2 : cleaning_condition_2 A)
                         (h3 : cleaning_condition_3 B A') :
                         (A' / A) = 2 :=
by sorry

end NUMINAMATH_GPT_anne_speed_ratio_l1745_174584


namespace NUMINAMATH_GPT_cost_of_eraser_l1745_174539

theorem cost_of_eraser
  (total_money: ℕ)
  (n_sharpeners n_notebooks n_erasers n_highlighters: ℕ)
  (price_sharpener price_notebook price_highlighter: ℕ)
  (heaven_spent brother_spent remaining_money final_spent: ℕ) :
  total_money = 100 →
  n_sharpeners = 2 →
  price_sharpener = 5 →
  n_notebooks = 4 →
  price_notebook = 5 →
  n_highlighters = 1 →
  price_highlighter = 30 →
  heaven_spent = n_sharpeners * price_sharpener + n_notebooks * price_notebook →
  brother_spent = 30 →
  remaining_money = total_money - heaven_spent →
  final_spent = remaining_money - brother_spent →
  final_spent = 40 →
  n_erasers = 10 →
  ∀ cost_per_eraser: ℕ, final_spent = cost_per_eraser * n_erasers →
  cost_per_eraser = 4 := by
  intros h_total_money h_n_sharpeners h_price_sharpener h_n_notebooks h_price_notebook
    h_n_highlighters h_price_highlighter h_heaven_spent h_brother_spent h_remaining_money
    h_final_spent h_n_erasers cost_per_eraser h_final_cost
  sorry

end NUMINAMATH_GPT_cost_of_eraser_l1745_174539


namespace NUMINAMATH_GPT_number_of_eggs_in_each_basket_l1745_174542

theorem number_of_eggs_in_each_basket 
  (total_blue_eggs : ℕ)
  (total_yellow_eggs : ℕ)
  (h1 : total_blue_eggs = 30)
  (h2 : total_yellow_eggs = 42)
  (exists_basket_count : ∃ n : ℕ, 6 ≤ n ∧ total_blue_eggs % n = 0 ∧ total_yellow_eggs % n = 0) :
  ∃ n : ℕ, n = 6 := 
sorry

end NUMINAMATH_GPT_number_of_eggs_in_each_basket_l1745_174542


namespace NUMINAMATH_GPT_students_exceed_guinea_pigs_l1745_174573

theorem students_exceed_guinea_pigs :
  let classrooms := 5
  let students_per_classroom := 20
  let guinea_pigs_per_classroom := 3
  let total_students := classrooms * students_per_classroom
  let total_guinea_pigs := classrooms * guinea_pigs_per_classroom
  total_students - total_guinea_pigs = 85 :=
by
  -- using the conditions and correct answer identified above
  let classrooms := 5
  let students_per_classroom := 20
  let guinea_pigs_per_classroom := 3
  let total_students := classrooms * students_per_classroom
  let total_guinea_pigs := classrooms * guinea_pigs_per_classroom
  show total_students - total_guinea_pigs = 85
  sorry

end NUMINAMATH_GPT_students_exceed_guinea_pigs_l1745_174573


namespace NUMINAMATH_GPT_larger_integer_is_50_l1745_174583

-- Definition of the problem conditions.
def is_two_digit (x : ℕ) : Prop := 10 ≤ x ∧ x ≤ 99

def problem_conditions (m n : ℕ) : Prop := 
  is_two_digit m ∧ is_two_digit n ∧
  (m + n) / 2 = m + n / 100

-- Statement of the proof problem.
theorem larger_integer_is_50 (m n : ℕ) (h : problem_conditions m n) : max m n = 50 :=
  sorry

end NUMINAMATH_GPT_larger_integer_is_50_l1745_174583


namespace NUMINAMATH_GPT_cos_value_l1745_174562

theorem cos_value (α : ℝ) (h : Real.sin (π/4 + α) = 1/3) :
  Real.cos (π/2 - 2*α) = -7/9 :=
sorry

end NUMINAMATH_GPT_cos_value_l1745_174562


namespace NUMINAMATH_GPT_percent_employed_females_l1745_174536

theorem percent_employed_females (percent_employed : ℝ) (percent_employed_males : ℝ) :
  percent_employed = 0.64 →
  percent_employed_males = 0.55 →
  (percent_employed - percent_employed_males) / percent_employed * 100 = 14.0625 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_percent_employed_females_l1745_174536


namespace NUMINAMATH_GPT_inequality_pos_real_l1745_174500

theorem inequality_pos_real (
  a b c : ℝ
) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  abc ≥ (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ∧ 
  (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ≥ (a + b - c) * (b + c - a) * (c + a - b) := 
sorry

end NUMINAMATH_GPT_inequality_pos_real_l1745_174500


namespace NUMINAMATH_GPT_minimum_value_2_l1745_174544

noncomputable def minimum_value (x y : ℝ) : ℝ := 2 * x + 3 * y ^ 2

theorem minimum_value_2 (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h : x + 2 * y = 1) : minimum_value x y = 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_2_l1745_174544


namespace NUMINAMATH_GPT_find_a_l1745_174506

theorem find_a 
  {a : ℝ} 
  (h : ∀ x : ℝ, (ax / (x - 1) < 1) ↔ (x < 1 ∨ x > 3)) : 
  a = 2 / 3 := 
sorry

end NUMINAMATH_GPT_find_a_l1745_174506


namespace NUMINAMATH_GPT_prime_divisibility_l1745_174508

theorem prime_divisibility (a b : ℕ) (ha_prime : Nat.Prime a) (hb_prime : Nat.Prime b) (ha_gt7 : a > 7) (hb_gt7 : b > 7) :
  290304 ∣ (a^2 - 1) * (b^2 - 1) * (a^6 - b^6) := 
by
  sorry

end NUMINAMATH_GPT_prime_divisibility_l1745_174508


namespace NUMINAMATH_GPT_dividing_by_10_l1745_174540

theorem dividing_by_10 (x : ℤ) (h : x + 8 = 88) : x / 10 = 8 :=
by
  sorry

end NUMINAMATH_GPT_dividing_by_10_l1745_174540


namespace NUMINAMATH_GPT_largest_divisor_of_expression_l1745_174554

theorem largest_divisor_of_expression (x : ℤ) (h_odd : x % 2 = 1) : 
  1200 ∣ ((10 * x - 4) * (10 * x) * (5 * x + 15)) := 
  sorry

end NUMINAMATH_GPT_largest_divisor_of_expression_l1745_174554


namespace NUMINAMATH_GPT_integral_abs_x_plus_2_eq_29_div_2_integral_inv_x_minus_1_eq_1_l1745_174547

open Real

noncomputable def integral_abs_x_plus_2 : ℝ :=
  ∫ x in (-4 : ℝ)..(3 : ℝ), |x + 2|

noncomputable def integral_inv_x_minus_1 : ℝ :=
  ∫ x in (2 : ℝ)..(Real.exp 1 + 1 : ℝ), 1 / (x - 1)

theorem integral_abs_x_plus_2_eq_29_div_2 :
  integral_abs_x_plus_2 = 29 / 2 :=
sorry

theorem integral_inv_x_minus_1_eq_1 :
  integral_inv_x_minus_1 = 1 :=
sorry

end NUMINAMATH_GPT_integral_abs_x_plus_2_eq_29_div_2_integral_inv_x_minus_1_eq_1_l1745_174547


namespace NUMINAMATH_GPT_a10_eq_neg12_l1745_174599

variable (a_n : ℕ → ℤ)
variable (S_n : ℕ → ℤ)
variable (d a1 : ℤ)

-- Conditions of the problem
axiom arithmetic_sequence : ∀ n : ℕ, a_n n = a1 + (n - 1) * d
axiom sum_of_first_n_terms : ∀ n : ℕ, S_n n = n * (2 * a1 + (n - 1) * d) / 2
axiom a2_eq_4 : a_n 2 = 4
axiom S8_eq_neg8 : S_n 8 = -8

-- The statement to prove
theorem a10_eq_neg12 : a_n 10 = -12 :=
sorry

end NUMINAMATH_GPT_a10_eq_neg12_l1745_174599


namespace NUMINAMATH_GPT_log_inequality_l1745_174512

noncomputable def log3_2 : ℝ := Real.log 2 / Real.log 3
noncomputable def log2_3 : ℝ := Real.log 3 / Real.log 2
noncomputable def log2_5 : ℝ := Real.log 5 / Real.log 2

theorem log_inequality :
  let a := log3_2;
  let b := log2_3;
  let c := log2_5;
  a < b ∧ b < c :=
  by
  sorry

end NUMINAMATH_GPT_log_inequality_l1745_174512


namespace NUMINAMATH_GPT_find_selling_price_l1745_174567

def cost_price : ℝ := 59
def selling_price_for_loss : ℝ := 52
def loss := cost_price - selling_price_for_loss

theorem find_selling_price (sp : ℝ) : (sp - cost_price = loss) → sp = 66 :=
by
  sorry

end NUMINAMATH_GPT_find_selling_price_l1745_174567


namespace NUMINAMATH_GPT_number_of_items_l1745_174550

theorem number_of_items {a n : ℕ} (h1 : ∀ x, x = 100 * a + 99) (h2 : 200 * 100 + 83 = 20083) : 
  (n * (100 * a + 99) = 20083) → (n = 17 ∨ n = 117) :=
by 
  sorry

end NUMINAMATH_GPT_number_of_items_l1745_174550


namespace NUMINAMATH_GPT_plane_parallel_l1745_174534

-- Definitions for planes and lines within a plane
variable (Plane : Type) (Line : Type)
variables (lines_in_plane1 : Set Line)
variables (parallel_to_plane2 : Line → Prop)
variables (Plane1 Plane2 : Plane)

-- Conditions
axiom infinite_lines_in_plane1_parallel_to_plane2 : ∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l
axiom planes_are_parallel : ∀ (P1 P2 : Plane), (∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l) → P1 = Plane1 → P2 = Plane2 → (Plane1 ≠ Plane2 ∧ (∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l))

-- The proof that Plane 1 and Plane 2 are parallel based on the conditions
theorem plane_parallel : Plane1 ≠ Plane2 → ∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l → (∀ l : Line, l ∈ lines_in_plane1 → parallel_to_plane2 l) := 
by
  sorry

end NUMINAMATH_GPT_plane_parallel_l1745_174534


namespace NUMINAMATH_GPT_force_required_l1745_174527

theorem force_required 
  (F : ℕ → ℕ)
  (h_inv : ∀ L L' : ℕ, F L * L = F L' * L')
  (h1 : F 12 = 300) :
  F 18 = 200 :=
by
  sorry

end NUMINAMATH_GPT_force_required_l1745_174527


namespace NUMINAMATH_GPT_find_greater_number_l1745_174592

theorem find_greater_number (x y : ℕ) 
  (h1 : x + y = 40)
  (h2 : x - y = 12) : x = 26 :=
by
  sorry

end NUMINAMATH_GPT_find_greater_number_l1745_174592


namespace NUMINAMATH_GPT_ab_necessary_not_sufficient_l1745_174513

theorem ab_necessary_not_sufficient (a b : ℝ) : 
  (ab > 0) ↔ ((a ≠ 0) ∧ (b ≠ 0) ∧ ((b / a + a / b > 2) → (ab > 0))) := 
sorry

end NUMINAMATH_GPT_ab_necessary_not_sufficient_l1745_174513


namespace NUMINAMATH_GPT_desired_value_l1745_174598

noncomputable def find_sum (a b c : ℝ) (p q r : ℝ) : ℝ :=
  a / p + b / q + c / r

theorem desired_value (a b c : ℝ) (h1 : p = a / 2) (h2 : q = b / 2) (h3 : r = c / 2) :
  find_sum a b c p q r = 6 :=
by
  sorry

end NUMINAMATH_GPT_desired_value_l1745_174598


namespace NUMINAMATH_GPT_five_in_range_for_all_b_l1745_174509

noncomputable def f (x b : ℝ) := x^2 + b * x - 3

theorem five_in_range_for_all_b : ∀ (b : ℝ), ∃ (x : ℝ), f x b = 5 := by 
  sorry

end NUMINAMATH_GPT_five_in_range_for_all_b_l1745_174509


namespace NUMINAMATH_GPT_intersection_complement_l1745_174563

open Set

-- Definitions from the problem
def U : Set ℝ := univ
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {y | 0 < y}

-- The proof statement
theorem intersection_complement : A ∩ (compl B) = Ioc (-1 : ℝ) 0 := by
  sorry

end NUMINAMATH_GPT_intersection_complement_l1745_174563


namespace NUMINAMATH_GPT_average_age_l1745_174561

variable (John Mary Tonya : ℕ)

theorem average_age (h1 : John = 2 * Mary) (h2 : John = Tonya / 2) (h3 : Tonya = 60) : 
  (John + Mary + Tonya) / 3 = 35 :=
by
  sorry

end NUMINAMATH_GPT_average_age_l1745_174561


namespace NUMINAMATH_GPT_probability_of_sequence_l1745_174519

noncomputable def prob_first_card_diamond : ℚ := 13 / 52
noncomputable def prob_second_card_spade_given_first_diamond : ℚ := 13 / 51
noncomputable def prob_third_card_heart_given_first_diamond_and_second_spade : ℚ := 13 / 50

theorem probability_of_sequence : 
  prob_first_card_diamond * prob_second_card_spade_given_first_diamond * 
  prob_third_card_heart_given_first_diamond_and_second_spade = 169 / 10200 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_probability_of_sequence_l1745_174519


namespace NUMINAMATH_GPT_amount_subtracted_for_new_ratio_l1745_174510

theorem amount_subtracted_for_new_ratio (x a : ℝ) (h1 : 3 * x = 72) (h2 : 8 * x = 192)
(h3 : (3 * x - a) / (8 * x - a) = 4 / 9) : a = 24 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_amount_subtracted_for_new_ratio_l1745_174510


namespace NUMINAMATH_GPT_quadratic_coefficients_l1745_174590

theorem quadratic_coefficients :
  ∀ x : ℝ, x * (x + 2) = 5 * (x - 2) → ∃ a b c : ℝ, a = 1 ∧ b = -3 ∧ c = 10 ∧ a * x^2 + b * x + c = 0 := by
  intros x h
  use 1, -3, 10
  sorry

end NUMINAMATH_GPT_quadratic_coefficients_l1745_174590


namespace NUMINAMATH_GPT_math_problem_l1745_174517

theorem math_problem (a b : ℝ) (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 :=
by
  -- The proof will be here
  sorry

end NUMINAMATH_GPT_math_problem_l1745_174517


namespace NUMINAMATH_GPT_hash_op_8_4_l1745_174568

def hash_op (a b : ℕ) : ℕ := a + a / b - 2

theorem hash_op_8_4 : hash_op 8 4 = 8 := 
by 
  -- The proof is left as an exercise, indicated by sorry.
  sorry

end NUMINAMATH_GPT_hash_op_8_4_l1745_174568


namespace NUMINAMATH_GPT_negation_equivalence_l1745_174533

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀^2 + x₀ - 2 < 0) ↔ (∀ x₀ : ℝ, x₀^2 + x₀ - 2 ≥ 0) :=
by sorry

end NUMINAMATH_GPT_negation_equivalence_l1745_174533


namespace NUMINAMATH_GPT_Allen_age_difference_l1745_174515

theorem Allen_age_difference (M A : ℕ) (h1 : M = 30) (h2 : (A + 3) + (M + 3) = 41) : M - A = 25 :=
by
  sorry

end NUMINAMATH_GPT_Allen_age_difference_l1745_174515


namespace NUMINAMATH_GPT_distance_center_to_plane_l1745_174575

noncomputable def sphere_center_to_plane_distance 
  (volume : ℝ) (AB AC : ℝ) (angleACB : ℝ) : ℝ :=
  let R := (3 * volume / 4 / Real.pi)^(1 / 3);
  let circumradius := AB / (2 * Real.sin (angleACB / 2));
  Real.sqrt (R^2 - circumradius^2)

theorem distance_center_to_plane 
  (volume : ℝ) (AB : ℝ) (angleACB : ℝ)
  (h_volume : volume = 500 * Real.pi / 3)
  (h_AB : AB = 4 * Real.sqrt 3)
  (h_angleACB : angleACB = Real.pi / 3) :
  sphere_center_to_plane_distance volume AB angleACB = 3 :=
by
  sorry

end NUMINAMATH_GPT_distance_center_to_plane_l1745_174575


namespace NUMINAMATH_GPT_problem_statement_l1745_174572

section

variable {f : ℝ → ℝ}

-- Conditions
axiom even_function (h : ∀ x : ℝ, f (-x) = f x) : ∀ x, f (-x) = f x 
axiom monotonically_increasing (h : ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y) :
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

-- Goal
theorem problem_statement 
  (h_even : ∀ x, f (-x) = f x)
  (h_mono : ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y) :
  f (-Real.log 2 / Real.log 3) > f (Real.log 2 / Real.log 3) ∧ f (Real.log 2 / Real.log 3) > f 0 := 
sorry

end

end NUMINAMATH_GPT_problem_statement_l1745_174572
